// Unit & integration tests for W74 HUD indicators in web/index.html:
// - Real-vs-smoke bootMode badge (#boot-mode-badge / data-boot-mode="idle")
// - Split-module load progress status line (#side-module-status)
// - Live WebGPU validation error pill (#gpu-validation-errors)
// - Preservation of all existing HUD IDs, COI/WebGPU checks, Wasm64 `address: 'i64'`
//   detection, valid HTML5 structure, and zero hardcoded host paths or game files.

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
const indexHtml = fs.readFileSync(INDEX_HTML_PATH, 'utf8');

function extractModuleScript(html) {
  const match = /<script\s+type="module">([\s\S]*?)<\/script>/i.exec(html);
  assert.ok(match, 'web/index.html must contain a <script type="module"> block');
  return match[1];
}

function extractStyleBlock(html) {
  const match = /<style>([\s\S]*?)<\/style>/i.exec(html);
  assert.ok(match, 'web/index.html must contain a <style> block');
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
  assert.match(indexHtml, /<html\s+lang="en">/i);
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
    // W74 new HUD diagnostic indicators
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

test('W74 boot-mode badge, split-module status line, and GPU validation error pill are accessible and unobtrusive inside #hud', () => {
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

  // Verify unobtrusive styling rules exist for all boot modes, 0-side-module hiding, and GPU errors.
  const css = extractStyleBlock(indexHtml);
  assert.match(css, /\[data-boot-mode="idle"\]/);
  assert.match(css, /\[data-boot-mode="real-wasm64"\]/);
  assert.match(css, /\[data-boot-mode="smoke"\]/);
  assert.match(css, /\.hud-side-modules\[data-total="0"\]\[data-loaded="0"\][\s\S]*display:\s*none/);
  assert.match(css, /\.hud-gpu-errors\[data-state="err"\]/);
  assert.match(css, /#viewport\s*\{[\s\S]*inset:\s*0/);
  assert.match(css, /#hud\s*\{[\s\S]*top:\s*12px[\s\S]*left:\s*12px/);
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
