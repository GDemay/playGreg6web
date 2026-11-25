// Unit & integration tests for W74 + W76 HUD overlay in web/index.html & web/ui/overlay.css:
// - Real-vs-smoke bootMode badge (#boot-mode-badge / data-boot-mode="idle")
// - Split-module load progress status line (#side-module-status)
// - Live WebGPU validation error metric (#gpu-validation-errors)
// - W76 debloated 4-section Astryx 1:1 layout, collapsed Diagnostics & Tools disclosure,
//   collapsed status pill, and self-contained COOP/COEP-safe Astryx stylesheet
// - Preservation of all existing HUD IDs, COI/WebGPU checks, Wasm64 `address: 'i64'`
//   detection, valid HTML5 structure, and zero hardcoded host paths or game files.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

import { HudController, BOOT_MODE, DEFAULT_TOGGLE_KEY } from '../web/ui/hud_controller.js';

const require = createRequire(import.meta.url);
const { GpuBootPanelController } = require('../web/ui/gpu_boot_panel.js');
const { DEFAULT_FADE_OUT_MS } = require('../web/ui/splash_overlay.js');

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

function extractElementById(html, id) {
  const escaped = id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const re = new RegExp(
    `<([a-zA-Z0-9-]+)\\b([^>]*\\bid="${escaped}"[^>]*)>([\\s\\S]*?)<\\/\\1>`,
    'i'
  );
  const match = re.exec(html);
  if (!match) return null;
  const [, tag, rawAttrs, innerText] = match;
  const attrs = {};
  for (const m of rawAttrs.matchAll(/\b([a-zA-Z0-9:-]+)="([^"]*)"/g)) {
    attrs[m[1]] = m[2];
  }
  return { tag: tag.toLowerCase(), attrs, innerText: innerText.trim(), raw: match[0] };
}

function extractTagBlock(html, openRe, closeTag) {
  const startMatch = openRe.exec(html);
  assert.ok(startMatch, `Expected opening pattern ${openRe} in web/index.html`);
  const startIdx = startMatch.index;
  const endIdx = html.indexOf(closeTag, startIdx);
  assert.ok(endIdx !== -1, `Expected closing tag ${closeTag} after ${startMatch[0]}`);
  return html.slice(startIdx, endIdx + closeTag.length);
}

function createMockDisplayElement(initial = {}) {
  const attributes = new Map();
  return {
    textContent: initial.textContent ?? '',
    title: initial.title ?? '',
    hidden: initial.hidden ?? false,
    dataset: { ...(initial.dataset || {}) },
    setAttribute(name, value) {
      attributes.set(name, String(value));
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
  };
}

test('W74 web/index.html is valid HTML5 with all required existing & new HUD IDs and zero duplicates', () => {
  assert.match(indexHtml, /^\s*<!doctype html>/i, 'web/index.html must start with HTML5 doctype');
  assert.match(indexHtml, /<html\s+lang="en"[^>]*>/i);
  assert.match(indexHtml, /<meta\s+charset="utf-8"\s*\/>/i);
  assert.match(indexHtml, /<head>[\s\S]*<\/head>/i);
  assert.match(indexHtml, /<body>[\s\S]*<\/body>/i);
  assert.match(indexHtml, /<\/html>\s*$/i);

  const requiredIds = [
    // Existing viewport, splash, HUD & panel IDs (must stay 100% intact)
    'viewport',
    'splash-overlay-root',
    'hud',
    'hud-status-dot',
    'hud-status',
    'hud-toggle',
    'hud-body',
    'hud-refresh-status',
    'hud-clear-log',
    'hud-log',
    'cap-coi',
    'cap-webgpu',
    'cap-wasm64',
    'cap-opfs',
    'cap-gamepad',
    'srv-manifest',
    'srv-bootset',
    'srv-wasm',
    'srv-shaders',
    'metric-fps',
    'metric-gpu-packets',
    'metric-opfs-hits',
    'metric-pointer-lock',
    'metric-gamepad',
    'btn-smoke-demo',
    'btn-restart-session',
    'btn-record-trace',
    'btn-export-trace',
    'btn-clear-cache',
    'btn-export-saves',
    'btn-export-crash',
    'btn-export-missing-shaders',
    'select-input-profile',
    'gpu-boot-panel-root',
    // W74 HUD diagnostic indicators
    'boot-mode-badge',
    'side-module-status',
    'gpu-validation-errors',
  ];

  for (const id of requiredIds) {
    assert.match(
      indexHtml,
      new RegExp(`\\bid="${id}"`),
      `web/index.html must define element with id="${id}"`
    );
  }

  // Zero duplicate id attributes.
  const idMatches = [...indexHtml.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
  const seen = new Set();
  for (const id of idMatches) {
    assert.equal(seen.has(id), false, `Duplicate element id="${id}" found in web/index.html`);
    seen.add(id);
  }
});

test('W74 boot-mode badge, split-module status line, and GPU validation error indicator are accessible and unobtrusive inside #hud', () => {
  const hudStart = indexHtml.indexOf('<div id="hud"');
  const scriptStart = indexHtml.indexOf('<script src="./runtime/pgim_preloader.js">');
  assert.ok(hudStart !== -1 && scriptStart > hudStart, 'Expected #hud container before scripts');
  const hudMarkup = indexHtml.slice(hudStart, scriptStart);

  // 1. #boot-mode-badge
  const bootBadge = extractElementById(hudMarkup, 'boot-mode-badge');
  assert.ok(bootBadge, '#boot-mode-badge must be nested inside #hud');
  assert.equal(bootBadge.attrs['data-boot-mode'], 'idle');
  assert.equal(bootBadge.attrs['data-state'], 'info');
  assert.equal(bootBadge.attrs.role, 'status');
  assert.equal(bootBadge.attrs['aria-live'], 'polite');
  assert.ok(bootBadge.attrs['aria-label'], '#boot-mode-badge must define aria-label');
  assert.equal(bootBadge.innerText, 'Idle');

  // 2. #side-module-status
  const sideStatus = extractElementById(hudMarkup, 'side-module-status');
  assert.ok(sideStatus, '#side-module-status must be nested inside #hud');
  assert.equal(sideStatus.attrs['data-loaded'], '0');
  assert.equal(sideStatus.attrs['data-total'], '0');
  assert.equal(sideStatus.attrs.role, 'status');
  assert.equal(sideStatus.attrs['aria-live'], 'polite');

  // 3. #gpu-validation-errors
  const gpuErrPill = extractElementById(hudMarkup, 'gpu-validation-errors');
  assert.ok(gpuErrPill, '#gpu-validation-errors must be nested inside #hud');
  assert.equal(gpuErrPill.attrs['data-count'], '0');
  assert.equal(gpuErrPill.attrs['data-state'], 'ok');
  assert.equal(gpuErrPill.attrs.role, 'status');
  assert.equal(gpuErrPill.attrs['aria-live'], 'polite');
  assert.ok(gpuErrPill.attrs.title, '#gpu-validation-errors must provide a tooltip summary');
  assert.ok(gpuErrPill.attrs['aria-label'], '#gpu-validation-errors must define aria-label');
  assert.equal(gpuErrPill.innerText, '0');

  // Verify unobtrusive styling rules exist in overlay.css for all boot modes, 0-side-module hiding, and GPU errors.
  assert.match(overlayCss, /\[data-boot-mode="idle"\]/);
  assert.match(overlayCss, /\[data-boot-mode="real-wasm64"\]/);
  assert.match(overlayCss, /\[data-boot-mode="smoke"\]/);
  assert.match(overlayCss, /\.hud-side-modules\[data-total="0"\]\[data-loaded="0"\][\s\S]*display:\s*none/);
  assert.match(overlayCss, /\.hud-gpu-errors\[data-state="err"\]/);
  assert.match(overlayCss, /#viewport\s*\{[\s\S]*inset:\s*0/);
  assert.match(overlayCss, /#hud\s*\{[\s\S]*top:\s*12px[\s\S]*left:\s*12px/);
});

test('W74 preserves COI/WebGPU checks, Wasm64 address: i64 detection, and valid ES module syntax', () => {
  const moduleScript = extractModuleScript(indexHtml);

  // Native Node ESM syntax check.
  const check = spawnSync(process.execPath, ['--input-type=module', '--check'], {
    input: moduleScript,
    encoding: 'utf8',
  });
  assert.equal(
    check.status,
    0,
    `Inline <script type="module"> failed syntax check:\n${check.stderr}`
  );

  // Preserve Wasm64 spec-standard `{ address: 'i64' }` probe ahead of legacy `{ index: 'i64' }`.
  const addressIdx = moduleScript.search(/address\s*:\s*['"]i64['"]/);
  const indexIdx = moduleScript.search(/index\s*:\s*['"]i64['"]/);
  assert.ok(addressIdx !== -1, 'web/index.html must preserve Wasm64 address: "i64" probe');
  assert.ok(indexIdx !== -1, 'web/index.html must preserve legacy index: "i64" fallback');
  assert.ok(addressIdx < indexIdx, 'address: "i64" must be probed before index: "i64"');

  // Preserve COI/SAB and WebGPU guards.
  assert.match(moduleScript, /self\.crossOriginIsolated\s*===\s*true/);
  assert.match(moduleScript, /typeof\s+SharedArrayBuffer\s*===\s*['"]function['"]/);
  assert.match(moduleScript, /['"]gpu['"]\s+in\s+navigator/);
  assert.match(moduleScript, /Missing COOP\/COEP/);
  assert.match(moduleScript, /WebGPU unavailable/);

  // Verify W74 HUD display elements and W69/W72/W73 hooks are wired in the inline script.
  assert.match(moduleScript, /getElementById\(\s*['"]boot-mode-badge['"]\s*\)/);
  assert.match(moduleScript, /getElementById\(\s*['"]side-module-status['"]\s*\)/);
  assert.match(moduleScript, /getElementById\(\s*['"]gpu-validation-errors['"]\s*\)/);
  assert.match(moduleScript, /displayElements\s*:\s*\{[\s\S]*bootModeBadge[\s\S]*sideModuleStatusEl[\s\S]*validationCountEl/);
  assert.match(moduleScript, /\bsetBootMode\b/);
  assert.match(moduleScript, /\bupdateBootStatus\b/);
  assert.match(moduleScript, /\bsetSideModules\b/);
  assert.match(moduleScript, /\brecordValidationError\b/);
  assert.match(moduleScript, /\bonGpuValidationError\b/);
  assert.match(moduleScript, /data\.sideModuleCount/);
});

test('W74 web/index.html contains zero forbidden hardcoded host paths or copyrighted game archive references', () => {
  const forbiddenPatterns = [
    { name: 'Windows drive path', re: /\b[A-Za-z]:\\(?:Users|Windows|Program Files|Temp|Games)/i },
    { name: 'macOS /Users/ host path', re: /\/Users\/[A-Za-z0-9._-]+\// },
    { name: 'Linux /home/ host path', re: /\/home\/[A-Za-z0-9._-]+\// },
    { name: 'file:// URL', re: /\bfile:\/\//i },
    { name: 'local-data/ or work/ dump reference', re: /\b(?:local-data|work)\/[A-Za-z0-9._-]+/ },
    { name: 'copyrighted game archive extension', re: /\.(?:rpf|xvd|xvc|ttpf|awc)\b/i },
  ];

  for (const { name, re } of forbiddenPatterns) {
    assert.doesNotMatch(
      indexHtml,
      re,
      `web/index.html must not contain ${name}`
    );
  }
});

test('W74 integration: HudController drives #boot-mode-badge, #side-module-status, and #gpu-validation-errors cleanly', () => {
  // Confirm initial attributes from web/index.html match our mock elements.
  const badgeSpec = extractElementById(indexHtml, 'boot-mode-badge');
  const sideSpec = extractElementById(indexHtml, 'side-module-status');
  const errSpec = extractElementById(indexHtml, 'gpu-validation-errors');
  assert.ok(badgeSpec && sideSpec && errSpec);

  const bootModeBadge = createMockDisplayElement({
    textContent: badgeSpec.innerText,
    dataset: { bootMode: badgeSpec.attrs['data-boot-mode'], state: badgeSpec.attrs['data-state'] },
  });
  const sideModuleStatusEl = createMockDisplayElement({
    textContent: sideSpec.innerText,
    dataset: { loaded: sideSpec.attrs['data-loaded'], total: sideSpec.attrs['data-total'] },
  });
  const validationCountEl = createMockDisplayElement({
    textContent: errSpec.innerText,
    title: errSpec.attrs.title,
    dataset: { count: errSpec.attrs['data-count'], state: errSpec.attrs['data-state'] },
  });

  const logs = [];
  const controller = new HudController({
    hud: {
      appendLog(message, level) {
        logs.push({ message, level });
      },
    },
    displayElements: {
      bootModeBadge,
      sideModuleStatusEl,
      validationCountEl,
    },
  });

  // 1. Initial idle state.
  assert.equal(bootModeBadge.textContent, 'Idle');
  assert.equal(bootModeBadge.dataset.bootMode, BOOT_MODE.IDLE);
  assert.equal(bootModeBadge.dataset.state, 'info');
  assert.equal(sideModuleStatusEl.dataset.total, '0');
  assert.equal(sideModuleStatusEl.dataset.loaded, '0');
  assert.equal(validationCountEl.textContent, '0');
  assert.equal(validationCountEl.dataset.count, '0');
  assert.equal(validationCountEl.dataset.state, 'ok');

  // 2. Switch to Real Wasm64 + split-module loading progress.
  controller.setBootMode('real-wasm64', { updateHudStatus: false });
  assert.equal(bootModeBadge.textContent, 'Real Wasm64');
  assert.equal(bootModeBadge.dataset.bootMode, BOOT_MODE.REAL_WASM64);
  assert.equal(bootModeBadge.dataset.state, 'ok');

  controller.setSideModuleProgress({
    sideModuleCount: 2,
    loadedCount: 0,
    sideModuleUrls: ['/dist/guest_part_0.wasm', '/dist/guest_part_1.wasm'],
  });
  assert.equal(sideModuleStatusEl.dataset.total, '2');
  assert.equal(sideModuleStatusEl.dataset.loaded, '0');
  assert.equal(sideModuleStatusEl.dataset.state, 'info');
  assert.match(sideModuleStatusEl.textContent, /0\/2/);

  controller.recordSideModuleLoaded('/dist/guest_part_0.wasm', { sideModuleCount: 2 });
  assert.equal(sideModuleStatusEl.dataset.loaded, '1');
  assert.match(sideModuleStatusEl.textContent, /1\/2 \(50%\)/);

  controller.recordSideModuleLoaded('/dist/guest_part_1.wasm', { sideModuleCount: 2 });
  assert.equal(sideModuleStatusEl.dataset.loaded, '2');
  assert.equal(sideModuleStatusEl.dataset.state, 'ok');
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 2/2 loaded');

  // 3. Switch to Smoke Demo mode.
  controller.setBootMode('smoke', { updateHudStatus: false });
  assert.equal(bootModeBadge.textContent, 'Smoke Demo');
  assert.equal(bootModeBadge.dataset.bootMode, BOOT_MODE.SMOKE);
  assert.equal(bootModeBadge.dataset.state, 'warn');

  // 4. Live WebGPU validation error updates pill count/state and sanitizes host paths.
  controller.recordValidationError({
    type: 'validation',
    message: 'RenderPipeline layout mismatch at C:\\Users\\alice\\secret\\shader.wgsl',
  });
  const validationSummary = controller.getValidationSummary();
  assert.equal(validationSummary.count, 1);
  assert.equal(validationCountEl.textContent, '1');
  assert.equal(validationCountEl.dataset.count, '1');
  assert.equal(validationCountEl.dataset.state, 'err');
  assert.doesNotMatch(validationSummary.statusText, /alice/i);
  assert.doesNotMatch(validationSummary.statusText, /C:\\Users/i);
  assert.equal(logs.length, 1);
  assert.equal(logs[0].level, 'error');

  // 5. Clear validation errors resets pill cleanly.
  controller.clearValidationErrors();
  assert.equal(validationCountEl.textContent, '0');
  assert.equal(validationCountEl.dataset.count, '0');
  assert.equal(validationCountEl.dataset.state, 'ok');

  controller.destroy();
});

test('W76 web/index.html loads self-contained Astryx overlay.css with zero external CDN assets (COOP/COEP safe)', () => {
  assert.equal(fs.existsSync(OVERLAY_CSS_PATH), true, 'web/ui/overlay.css must exist');
  assert.match(indexHtml, /<html[^>]*data-astryx-theme="astryx"/);
  assert.match(indexHtml, /<link\s+rel="stylesheet"\s+href="\.\/ui\/overlay\.css"\s*\/?>/);

  // Must never pull un-CORP'd external fonts or stylesheets that COEP blocks.
  assert.doesNotMatch(indexHtml, /https?:\/\/[^\s"'<>]+(?:\.css|fonts\.googleapis|fonts\.gstatic|unpkg\.com|cdn)/i);
  assert.doesNotMatch(overlayCss, /@import\s+url\(\s*['"]?https?:/i);
  assert.doesNotMatch(overlayCss, /url\(\s*['"]?https?:/i);

  // Attribution + Astryx tokens and slim 320px overlay / pill collapse geometry.
  assert.match(overlayCss, /Astryx/i);
  assert.match(overlayCss, /--color-background-body:\s*#111112/i);
  assert.match(overlayCss, /--color-accent:\s*#dfe2e5/i);
  assert.match(overlayCss, /--color-on-accent:\s*#15110c/i);
  assert.match(overlayCss, /--radius-full:\s*9999px/);
  assert.match(overlayCss, /max-width:\s*320px/);
  assert.match(overlayCss, /#hud\.collapsed[\s\S]*border-radius:\s*var\(--radius-full\)/);
  assert.match(overlayCss, /\.astryx-button\b/);
  assert.match(overlayCss, /\.astryx-badge\b/);
  assert.match(overlayCss, /\.astryx-progress-bar-track\b/);
  assert.match(overlayCss, /\.astryx-collapsible\b/);
});

test('W76 HUD defaults to 4 clean visible sections with Diagnostics & Tools collapsed by default', () => {
  // 1. Header bar: brand, Astryx boot-mode badge, collapsed pill FPS readout, and Hide button.
  assert.match(indexHtml, /id="boot-mode-badge"/);
  assert.match(indexHtml, /id="hud-pill-fps"/);
  assert.match(indexHtml, /id="hud-toggle"[^>]*>Hide<\/button>/);

  // 2. Boot progress row: stage label, thin Astryx progress bar, elapsed time, side-module status line.
  assert.match(indexHtml, /id="hud-boot-stage"/);
  assert.match(indexHtml, /id="hud-boot-progress-bar"[^>]*role="progressbar"/);
  assert.match(indexHtml, /id="hud-boot-progress-fill"/);
  assert.match(indexHtml, /id="hud-boot-elapsed"/);
  assert.match(indexHtml, /id="side-module-status"/);

  // 3. Compact 3-metric strip: Frames, FPS, GPU Errors.
  const metricsStrip = extractTagBlock(
    indexHtml,
    /<section\s+class="hud-metrics-strip"[^>]*>/,
    '</section>'
  );
  assert.match(metricsStrip, /id="metric-frames"/);
  assert.match(metricsStrip, /id="metric-fps"/);
  assert.match(metricsStrip, /id="gpu-validation-errors"/);

  // 4. Primary actions row has ONLY 2 primary buttons: Restart and Export Report.
  const primaryActions = extractTagBlock(
    indexHtml,
    /<div\s+class="hud-primary-actions"[^>]*>/,
    '</div>'
  );
  const primaryButtons = [...primaryActions.matchAll(/<button\b[^>]*id="([^"]+)"/g)].map((m) => m[1]);
  assert.deepEqual(primaryButtons, ['btn-restart-session', 'btn-export-crash']);

  // Conditional banners start hidden by default.
  assert.match(indexHtml, /<div\s+id="hud-capability-banner"[^>]*\bhidden\b/);
  assert.match(indexHtml, /<div\s+id="hud-validation-banner"[^>]*\bhidden\b/);

  // Compact collapsible Log is a <details> closed by default.
  assert.match(indexHtml, /<details\s+id="hud-log-disclosure"\s+class="astryx-collapsible">/);
  assert.doesNotMatch(indexHtml, /<details\s+id="hud-log-disclosure"[^>]*\bopen\b/);

  // Diagnostics & Tools disclosure is a <details> closed by default and contains all secondary controls.
  assert.match(indexHtml, /<details\s+id="hud-diagnostics-disclosure"\s+class="astryx-collapsible">/);
  assert.doesNotMatch(indexHtml, /<details\s+id="hud-diagnostics-disclosure"[^>]*\bopen\b/);

  const diagBlock = extractTagBlock(
    indexHtml,
    /<details\s+id="hud-diagnostics-disclosure"[^>]*>/,
    '</details>'
  );

  const tuckedDiagnosticIds = [
    'cap-coi',
    'cap-webgpu',
    'cap-wasm64',
    'cap-opfs',
    'cap-gamepad',
    'hud-refresh-status',
    'srv-manifest',
    'srv-bootset',
    'srv-wasm',
    'srv-shaders',
    'metric-gpu-packets',
    'metric-opfs-hits',
    'metric-pointer-lock',
    'metric-gamepad',
    'btn-smoke-demo',
    'btn-record-trace',
    'btn-export-trace',
    'btn-clear-cache',
    'btn-export-saves',
    'btn-export-missing-shaders',
    'select-input-profile',
    'gpu-boot-panel-root',
  ];

  for (const id of tuckedDiagnosticIds) {
    assert.match(
      diagBlock,
      new RegExp(`id="${id}"`),
      `Secondary diagnostic element id="${id}" must be tucked inside collapsed #hud-diagnostics-disclosure`
    );
  }
});

test('W145 web/index.html defines scripted-input badge, F3/?hud=0 visibility hooks, and Diagnostics & Tools telemetry summary slots', () => {
  const hudStart = indexHtml.indexOf('<div id="hud"');
  const scriptStart = indexHtml.indexOf('<script src="./runtime/pgim_preloader.js">');
  assert.ok(hudStart !== -1 && scriptStart > hudStart, 'Expected #hud container before scripts');
  const hudMarkup = indexHtml.slice(hudStart, scriptStart);

  // Clean [hidden] guard for #hud and #scripted-input-badge.
  assert.match(indexHtml, /#hud\[hidden\][\s\S]*display:\s*none\s*!important/);
  assert.match(indexHtml, /#scripted-input-badge\[hidden\][\s\S]*display:\s*none\s*!important/);
  assert.match(indexHtml, /<div\s+id="hud"[^>]*data-hud-visible="true"/);

  // Scripted-input badge starts hidden and accessible inside #hud.
  const scriptBadge = extractElementById(hudMarkup, 'scripted-input-badge');
  assert.ok(scriptBadge, '#scripted-input-badge must be nested inside #hud');
  assert.equal(scriptBadge.attrs['data-active'], 'false');
  assert.equal(scriptBadge.attrs['data-preset'], '');
  assert.equal(scriptBadge.attrs.role, 'status');
  assert.equal(scriptBadge.attrs['aria-live'], 'polite');
  assert.equal(scriptBadge.attrs['aria-hidden'], 'true');
  assert.match(scriptBadge.raw, /\bhidden\b/);
  assert.equal(scriptBadge.innerText, '');

  // GPU compat, HTTPFS/cache, and shader hit-rate summary slots live inside collapsed Diagnostics & Tools and default to '-'.
  const diagBlock = extractTagBlock(
    indexHtml,
    /<details\s+id="hud-diagnostics-disclosure"[^>]*>/,
    '</details>'
  );
  for (const id of ['metric-shader-hit-rate', 'metric-gpu-compat', 'metric-httpfs-summary']) {
    const slot = extractElementById(diagBlock, id);
    assert.ok(slot, `#${id} must be tucked inside collapsed #hud-diagnostics-disclosure`);
    assert.equal(slot.innerText, '-', `#${id} must default gracefully to '-' before telemetry arrives`);
  }

  // Inline module wires F3 shortcut, ?hud=0 / ?hideHud=1, scriptedInputBadge, and W114 compat/swapchain/httpfs forwarding.
  const moduleScript = extractModuleScript(indexHtml);
  assert.match(moduleScript, /getElementById\(\s*['"]scripted-input-badge['"]\s*\)/);
  assert.match(moduleScript, /getElementById\(\s*['"]metric-shader-hit-rate['"]\s*\)/);
  assert.match(moduleScript, /getElementById\(\s*['"]metric-gpu-compat['"]\s*\)/);
  assert.match(moduleScript, /getElementById\(\s*['"]metric-httpfs-summary['"]\s*\)/);
  assert.match(moduleScript, /enableToggleShortcut:\s*true/);
  assert.match(moduleScript, /toggleShortcutKey:\s*['"]F3['"]/);
  assert.match(moduleScript, /hudRootEl:\s*hudEl/);
  assert.match(moduleScript, /scriptedInputBadge:\s*scriptedInputBadgeEl/);
  assert.match(moduleScript, /\bisVisible\b/);
  assert.match(moduleScript, /\bsetVisible\b/);
  assert.match(moduleScript, /\btoggleVisible\b/);
  assert.match(moduleScript, /\bsetScriptedInputStatus\b/);
  assert.match(moduleScript, /\bupdateGpuBootTelemetry\b/);
  assert.match(moduleScript, /\bcompatStats\b/);
  assert.match(moduleScript, /\bswapchain\b/);
  assert.match(moduleScript, /\bhttpfsStats\b/);
});

test('W145 inline helpers parse ?hud=0 / ?hideHud=1 / ?inputScript, guard editable targets, and normalize GPU compat/HTTPFS payloads', () => {
  const moduleScript = extractModuleScript(indexHtml);

  const helperSliceMatch =
    /const SCRIPTED_INPUT_PRESET_RE[\s\S]*?function buildGpuBootTelemetryPayload\([\s\S]*?\r?\n    \}/.exec(
      moduleScript
    );
  assert.ok(helperSliceMatch, 'Expected W145 helper block in web/index.html inline script');

  const factory = new Function(
    `${helperSliceMatch[0]}\nreturn { parseHudQueryOptions, isEditableEventTarget, buildGpuBootTelemetryPayload, sanitizeScriptedInputPreset };`
  );
  const {
    parseHudQueryOptions,
    isEditableEventTarget,
    buildGpuBootTelemetryPayload,
  } = factory();

  // 1. Clean capture URL flags (?hud=0, ?hideHud=1, case-insensitive) and ?inputScript=<name>.
  assert.deepEqual(parseHudQueryOptions(''), { hudVisible: true, inputScript: null });
  assert.deepEqual(parseHudQueryOptions('?hud=0'), { hudVisible: false, inputScript: null });
  assert.deepEqual(parseHudQueryOptions('?HUD=false'), { hudVisible: false, inputScript: null });
  assert.deepEqual(parseHudQueryOptions('?hud=off'), { hudVisible: false, inputScript: null });
  assert.deepEqual(parseHudQueryOptions('?hud=hidden'), { hudVisible: false, inputScript: null });
  assert.deepEqual(parseHudQueryOptions('?hideHud=1'), { hudVisible: false, inputScript: null });
  assert.deepEqual(parseHudQueryOptions('?hide_hud=true'), { hudVisible: false, inputScript: null });
  assert.deepEqual(parseHudQueryOptions('?hud=1'), { hudVisible: true, inputScript: null });
  assert.deepEqual(parseHudQueryOptions('?hud=0&inputScript=start-screen'), {
    hudVisible: false,
    inputScript: 'start-screen',
  });
  assert.deepEqual(parseHudQueryOptions('?inputScript=menu-flow'), {
    hudVisible: true,
    inputScript: 'menu-flow',
  });
  // Reject path-traversal / script injection in ?inputScript.
  assert.deepEqual(parseHudQueryOptions('?inputScript=../../secret'), {
    hudVisible: true,
    inputScript: null,
  });
  assert.deepEqual(parseHudQueryOptions('?inputScript=<script>alert(1)</script>'), {
    hudVisible: true,
    inputScript: null,
  });

  // 2. Editable target guard for F3/F9 keyboard shortcuts.
  assert.equal(isEditableEventTarget({ tagName: 'INPUT' }), true);
  assert.equal(isEditableEventTarget({ tagName: 'textarea' }), true);
  assert.equal(isEditableEventTarget({ tagName: 'SELECT' }), true);
  assert.equal(isEditableEventTarget({ tagName: 'DIV', isContentEditable: true }), true);
  assert.equal(
    isEditableEventTarget({
      tagName: 'DIV',
      getAttribute: (name) => (name === 'contenteditable' ? 'true' : null),
    }),
    true
  );
  assert.equal(isEditableEventTarget({ tagName: 'CANVAS' }), false);
  assert.equal(isEditableEventTarget({ tagName: 'BODY' }), false);
  assert.equal(isEditableEventTarget(null), false);

  // 3. Telemetry forwarding payload normalization (compat -> compatStats, httpfs -> httpfsStats).
  assert.equal(buildGpuBootTelemetryPayload({ fps: 60, gpuPackets: 10 }), null);
  const normalized = buildGpuBootTelemetryPayload({
    compat: { bindGroupKindMismatches: 2, autoLayoutFallbacks: 1, bc3dFallbacks: 3 },
    swapchain: { format: 'bgra8unorm', usesBlitFallback: true, blitCount: 4 },
    httpfs: { pgimFullFetches: 1, slabCacheHits: 12, slabCacheMisses: 2, syncFastPathHits: 7 },
    fallbackStats: { totalLookups: 20, packHits: 17, fallbackHits: 3, builtinHits: 0 },
  });
  assert.ok(normalized);
  assert.deepEqual(normalized.compatStats, {
    bindGroupKindMismatches: 2,
    autoLayoutFallbacks: 1,
    bc3dFallbacks: 3,
  });
  assert.deepEqual(normalized.swapchain, {
    format: 'bgra8unorm',
    usesBlitFallback: true,
    blitCount: 4,
  });
  assert.deepEqual(normalized.httpfsStats, {
    pgimFullFetches: 1,
    slabCacheHits: 12,
    slabCacheMisses: 2,
    syncFastPathHits: 7,
  });
  assert.equal(normalized.fallbackStats.packHits, 17);
});

test('W145 integration: F3 toggle, ?hud=0 clean capture, scripted-input badge, and compat/swapchain/httpfs forwarding into GpuBootPanelController', () => {
  const hudRootEl = createMockDisplayElement({
    dataset: { hudVisible: 'true' },
  });
  const scriptedInputBadge = createMockDisplayElement({
    hidden: true,
    dataset: { active: 'false', preset: '' },
  });

  const listeners = new Map();
  const keyTarget = {
    addEventListener(type, fn) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      listeners.get(type)?.delete(fn);
    },
    dispatch(type, event) {
      for (const fn of [...(listeners.get(type) || [])]) {
        fn(event);
      }
    },
    listenerCount(type) {
      return listeners.get(type)?.size ?? 0;
    },
  };

  function makePanelEl(tagName = 'DIV') {
    const attrs = new Map();
    const el = {
      tagName: tagName.toUpperCase(),
      className: '',
      textContent: '',
      hidden: false,
      style: {},
      children: [],
      parentNode: null,
      setAttribute(k, v) {
        attrs.set(k, String(v));
      },
      getAttribute(k) {
        return attrs.has(k) ? attrs.get(k) : null;
      },
      appendChild(child) {
        if (child && typeof child === 'object') child.parentNode = el;
        el.children.push(child);
        return child;
      },
      removeChild(child) {
        const idx = el.children.indexOf(child);
        if (idx !== -1) el.children.splice(idx, 1);
        if (child && typeof child === 'object') child.parentNode = null;
        return child;
      },
      addEventListener() {},
      removeEventListener() {},
    };
    return el;
  }

  const panelRoot = makePanelEl('DIV');
  const panelDoc = {
    head: makePanelEl('HEAD'),
    body: makePanelEl('BODY'),
    createElement: (tag) => makePanelEl(tag),
    querySelector: () => null,
  };

  const gpuBootPanel = new GpuBootPanelController({
    container: panelRoot,
    documentObj: panelDoc,
    collapsed: true,
    now: () => 1700000000000,
  });

  // Start hidden as in `?hud=0&inputScript=menu-flow`.
  const controller = new HudController({
    gpuBootPanel,
    visible: false,
    scriptedInputPreset: 'menu-flow',
    enableToggleShortcut: true,
    toggleShortcutTarget: keyTarget,
    toggleShortcutKey: DEFAULT_TOGGLE_KEY,
    displayElements: {
      hudRootEl,
      scriptedInputBadge,
    },
  });

  // Initial `?hud=0` state hides HUD cleanly while scripted-input badge is populated.
  assert.equal(controller.isVisible(), false);
  assert.equal(hudRootEl.hidden, true);
  assert.equal(hudRootEl.getAttribute('aria-hidden'), 'true');
  assert.equal(hudRootEl.dataset.hudVisible, 'false');
  assert.equal(scriptedInputBadge.hidden, false);
  assert.equal(scriptedInputBadge.textContent, 'Scripted input: menu-flow');
  assert.equal(scriptedInputBadge.dataset.active, 'true');
  assert.equal(scriptedInputBadge.dataset.preset, 'menu-flow');

  // F3 repeat and text inputs must be ignored.
  let prevented = false;
  keyTarget.dispatch('keydown', {
    key: 'F3',
    code: 'F3',
    repeat: true,
    target: { tagName: 'BODY' },
    preventDefault() {
      prevented = true;
    },
  });
  assert.equal(controller.isVisible(), false);
  assert.equal(prevented, false);

  keyTarget.dispatch('keydown', {
    key: 'F3',
    code: 'F3',
    repeat: false,
    target: { tagName: 'INPUT' },
    preventDefault() {
      prevented = true;
    },
  });
  assert.equal(controller.isVisible(), false);
  assert.equal(prevented, false);

  // Clean F3 press shows the HUD, second F3 press hides it again.
  keyTarget.dispatch('keydown', {
    key: 'F3',
    code: 'F3',
    repeat: false,
    target: { tagName: 'CANVAS' },
    preventDefault() {
      prevented = true;
    },
  });
  assert.equal(prevented, true);
  assert.equal(controller.isVisible(), true);
  assert.equal(hudRootEl.hidden, false);
  assert.equal(hudRootEl.dataset.hudVisible, 'true');

  keyTarget.dispatch('keydown', {
    key: 'F3',
    code: 'F3',
    repeat: false,
    target: { tagName: 'CANVAS' },
    preventDefault() {},
  });
  assert.equal(controller.isVisible(), false);
  assert.equal(hudRootEl.hidden, true);

  // Clearing scripted input hides the badge cleanly.
  controller.setScriptedInputStatus(null, { updateHudStatus: false });
  assert.equal(scriptedInputBadge.hidden, true);
  assert.equal(scriptedInputBadge.dataset.active, 'false');

  // Forwarding compat, swapchain, httpfs, and fallbackStats updates GpuBootPanelController state.
  const panelState = controller.updateGpuBootTelemetry({
    compatStats: { bindGroupKindMismatches: 1, autoLayoutFallbacks: 2, bc3dFallbacks: 1 },
    swapchain: { format: 'bgra8unorm', usesBlitFallback: true, blitCount: 5 },
    httpfsStats: { pgimFullFetches: 1, slabCacheHits: 18, slabCacheMisses: 2, syncFastPathHits: 9 },
    fallbackStats: { totalLookups: 10, packHits: 9, fallbackHits: 1, builtinHits: 0 },
  });
  assert.ok(panelState);
  assert.equal(panelState.compatStats.bindGroupKindMismatches, 1);
  assert.equal(panelState.compatStats.autoLayoutFallbacks, 2);
  assert.equal(panelState.compatStats.bc3dFallbacks, 1);
  assert.equal(panelState.compatStats.swapchainFormat, 'bgra8unorm');
  assert.equal(panelState.compatStats.usesBlitFallback, true);
  assert.equal(panelState.compatStats.blitCount, 5);
  assert.equal(panelState.ioStats.pgimFullFetches, 1);
  assert.equal(panelState.ioStats.slabCacheHits, 18);
  assert.equal(panelState.ioStats.slabCacheMisses, 2);
  assert.equal(panelState.ioStats.syncFastPathHits, 9);
  assert.equal(panelState.hitRateText, '90.0%');

  controller.destroy();
  gpuBootPanel.destroy();
  assert.equal(keyTarget.listenerCount('keydown'), 0);
});

test('W330 web/ui/overlay.css aligns .pg6-splash--fading transition with W311 fadeOutMs and preserves Astryx splash tokens', () => {
  assert.equal(DEFAULT_FADE_OUT_MS, 220);

  // :root splash fade duration tokens match W311 DEFAULT_FADE_OUT_MS (220ms).
  assert.match(
    overlayCss,
    new RegExp(`--pg6-splash-fade-out:\\s*${DEFAULT_FADE_OUT_MS}ms;`),
    'overlay.css must define --pg6-splash-fade-out matching W311 DEFAULT_FADE_OUT_MS'
  );
  assert.match(
    overlayCss,
    new RegExp(`--pg6-splash-fade:\\s*${DEFAULT_FADE_OUT_MS}ms;`),
    'overlay.css must align default --pg6-splash-fade with W311 DEFAULT_FADE_OUT_MS'
  );
  assert.match(overlayCss, /--pg6-splash-ease:\s*cubic-bezier\(/);

  // Smooth opacity transition on splash root and .pg6-splash--fading (standalone and #splash-overlay-root).
  assert.match(
    overlayCss,
    /\.pg6-splash,\s*\r?\n#splash-overlay-root\s+\.pg6-splash\s*\{[\s\S]*?opacity:\s*1;[\s\S]*?transition:\s*opacity\s+var\(--pg6-splash-fade,\s*220ms\)\s+var\(--pg6-splash-ease[\s\S]*?will-change:\s*opacity;/
  );
  assert.match(
    overlayCss,
    /\.pg6-splash\.pg6-splash--fading,\s*\r?\n#splash-overlay-root\s+\.pg6-splash\.pg6-splash--fading\s*\{[\s\S]*?opacity:\s*0;[\s\S]*?pointer-events:\s*none;[\s\S]*?transition:\s*opacity\s+var\(--pg6-splash-fade,\s*220ms\)/
  );
  assert.match(
    overlayCss,
    /\.pg6-splash\.pg6-splash--fading\s+\.pg6-splash__panel,\s*\r?\n#splash-overlay-root\s+\.pg6-splash\.pg6-splash--fading\s+\.pg6-splash__panel\s*\{[\s\S]*?opacity:\s*0;[\s\S]*?pointer-events:\s*none;/
  );
});

test('W330 web/ui/overlay.css disables splash & HUD transitions under prefers-reduced-motion and [data-reduced-motion="true"]', () => {
  const reducedMotionMatch = /@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)\s*\{([\s\S]*)\}\s*$/.exec(
    overlayCss
  );
  assert.ok(
    reducedMotionMatch,
    'overlay.css must end with a @media (prefers-reduced-motion: reduce) cascade override block'
  );
  const reducedMotionBody = reducedMotionMatch[1];

  // Zero out motion custom properties.
  assert.match(reducedMotionBody, /--duration-fast:\s*0ms;/);
  assert.match(reducedMotionBody, /--pg6-splash-fade:\s*0ms;/);
  assert.match(reducedMotionBody, /--pg6-splash-fade-out:\s*0ms;/);

  // Cover HUD, splash fading state, panel, progress bars, and spinners.
  for (const selector of [
    '#hud',
    '.pg6-splash',
    '.pg6-splash.pg6-splash--fading',
    '#splash-overlay-root .pg6-splash',
    '#splash-overlay-root .pg6-splash.pg6-splash--fading',
    '.pg6-splash__panel',
    '#splash-overlay-root .pg6-splash__panel',
    '.pg6-splash__bar-fill',
    '.pg6-splash__stream-fill',
    '.astryx-button',
    '.astryx-progress-bar-fill',
    '.astryx-chevron',
  ]) {
    assert.ok(
      reducedMotionBody.includes(selector),
      `prefers-reduced-motion block must cover "${selector}"`
    );
  }
  assert.match(reducedMotionBody, /transition:\s*none\s*!important;/);
  assert.match(reducedMotionBody, /transition-duration:\s*0ms\s*!important;/);
  assert.match(reducedMotionBody, /transition-delay:\s*0ms\s*!important;/);
  assert.match(reducedMotionBody, /transform:\s*none\s*!important;/);
  assert.match(
    reducedMotionBody,
    /\.pg6-splash__spinner-ring[\s\S]*\.pg6-splash__greg[\s\S]*animation:\s*none\s*!important;/
  );

  // Programmatic [data-reduced-motion="true"] attribute also disables fade transitions & spinner rotation.
  assert.match(
    overlayCss,
    /\.pg6-splash\[data-reduced-motion="true"\][\s\S]*--pg6-splash-fade:\s*0ms;[\s\S]*transition:\s*none\s*!important;/
  );
  assert.match(
    overlayCss,
    /\.pg6-splash\[data-reduced-motion="true"\]\s+\.pg6-splash__spinner-ring[\s\S]*animation:\s*none\s*!important;/
  );
});

test('W330 web/ui/overlay.css adds clean 1080p/4K/narrow responsive sizing while preserving compact Astryx #hud invariants', () => {
  // Base 1080p Astryx compact top-left #hud card invariants remain intact.
  assert.match(
    overlayCss,
    /#hud\s*\{[\s\S]*?position:\s*fixed;[\s\S]*?top:\s*12px;[\s\S]*?left:\s*12px;[\s\S]*?width:\s*min\(320px,\s*calc\(100vw - 24px\)\);[\s\S]*?max-width:\s*320px;[\s\S]*?border:\s*none;[\s\S]*?border-radius:\s*var\(--radius-container\);[\s\S]*?contain:\s*layout style paint;/
  );
  assert.match(overlayCss, /#hud\[data-hud-visible="false"\]\s*\{[\s\S]*?display:\s*none\s*!important;/);
  assert.match(overlayCss, /#hud\.collapsed\s*\{[\s\S]*?border-radius:\s*var\(--radius-full\);/);
  assert.doesNotMatch(overlayCss, /#6db3ff/i, 'overlay.css must never reintroduce legacy neon blue #6db3ff');

  // 4K viewport breakpoint scales gently without bloating the compact top-left #hud card.
  const fourKMatch = /@media\s*\(\s*min-width\s*:\s*2560px\s*\)\s*\{([\s\S]*?)\r?\n\}/.exec(overlayCss);
  assert.ok(fourKMatch, 'overlay.css must define a @media (min-width: 2560px) responsive block for 4K viewports');
  const fourKBody = fourKMatch[1];
  const fourKMaxWidth = /#hud\s*\{[\s\S]*?max-width:\s*(\d+)px;/.exec(fourKBody);
  assert.ok(fourKMaxWidth, '4K breakpoint must cap #hud max-width cleanly');
  assert.ok(
    Number(fourKMaxWidth[1]) >= 320 && Number(fourKMaxWidth[1]) <= 360,
    `4K #hud max-width (${fourKMaxWidth[1]}px) must stay compact (320px..360px)`
  );
  assert.match(fourKBody, /#splash-overlay-root\s+\.pg6-splash__panel\s*\{[\s\S]*?max-width:\s*368px;/);

  // Narrow viewport breakpoints tighten margins & stack primary actions without overflow.
  const narrowMatch = /@media\s*\(\s*max-width\s*:\s*640px\s*\)\s*\{([\s\S]*?)\r?\n\}/.exec(overlayCss);
  assert.ok(narrowMatch, 'overlay.css must define a @media (max-width: 640px) responsive block for narrow viewports');
  assert.match(narrowMatch[1], /#hud\s*\{[\s\S]*?top:\s*8px;[\s\S]*?left:\s*8px;[\s\S]*?calc\(100vw - 16px\)/);
  assert.match(narrowMatch[1], /#splash-overlay-root\s+\.pg6-splash__panel\s*\{[\s\S]*?calc\(100vw - 16px\)/);

  const tinyMatch = /@media\s*\(\s*max-width\s*:\s*380px\s*\)\s*\{([\s\S]*?)\r?\n\}/.exec(overlayCss);
  assert.ok(tinyMatch, 'overlay.css must define a @media (max-width: 380px) responsive block for ultra-narrow viewports');
  assert.match(tinyMatch[1], /\.hud-primary-actions[\s\S]*?grid-template-columns:\s*1fr;/);

  const shortMatch = /@media\s*\(\s*max-height\s*:\s*600px\s*\)\s*\{([\s\S]*?)\r?\n\}/.exec(overlayCss);
  assert.ok(shortMatch, 'overlay.css must define a @media (max-height: 600px) responsive block for short viewports');
  assert.match(shortMatch[1], /#hud-body\s*\{[\s\S]*?max-height:\s*calc\(100vh - 56px\);/);
});
