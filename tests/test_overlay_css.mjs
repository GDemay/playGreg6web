// Unit tests for W226 HUD/overlay CSS containment & clean capture styling in web/ui/overlay.css:
// - Balanced braces, parentheses, and CSS comments (valid syntax)
// - CSS layout/paint containment (`contain: layout style paint`) on the floating HUD panel
//   and collapsible diagnostics/log panels so live telemetry never triggers full-viewport
//   repaints over the 4K WebGPU canvas
// - `[hidden]` and clean-capture visibility rules enforce `display: none !important`
// - `@media (prefers-reduced-motion: reduce)` disables transition delays for instant
//   HUD and splash hiding
// - Zero regressions on existing Astryx overlay selectors, variables, responsiveness,
//   and COOP/COEP self-containment.

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const INDEX_HTML_PATH = path.join(REPO_ROOT, 'web', 'index.html');
const OVERLAY_CSS_PATH = path.join(REPO_ROOT, 'web', 'ui', 'overlay.css');
const indexHtml = fs.readFileSync(INDEX_HTML_PATH, 'utf8');
const overlayCss = fs.readFileSync(OVERLAY_CSS_PATH, 'utf8');

/**
 * Strips CSS block comments while verifying every opened comment is properly closed.
 *
 * @param {string} css
 * @returns {string}
 */
function stripCssComments(css) {
  let out = '';
  let i = 0;
  while (i < css.length) {
    if (css[i] === '/' && css[i + 1] === '*') {
      const end = css.indexOf('*/', i + 2);
      assert.notEqual(end, -1, `Unterminated CSS comment starting at offset ${i}`);
      out += ' ';
      i = end + 2;
      continue;
    }
    if (css[i] === '*' && css[i + 1] === '/') {
      assert.fail(`Unexpected closing CSS comment "*/" at offset ${i}`);
    }
    out += css[i];
    i += 1;
  }
  return out;
}

/**
 * Parses top-level CSS rules and nested `@media` rule blocks into structured entries.
 *
 * @param {string} css
 * @returns {{ rules: Array<{ selector: string, body: string }>, mediaBlocks: Array<{ prelude: string, rules: Array<{ selector: string, body: string }> }> }}
 */
function parseCssBlocks(css) {
  const clean = stripCssComments(css);
  const rules = [];
  const mediaBlocks = [];

  function parseRange(start, end, targetRules, collectMedia) {
    let i = start;
    while (i < end) {
      while (i < end && /\s/.test(clean[i])) i += 1;
      if (i >= end) break;

      const openBrace = clean.indexOf('{', i);
      if (openBrace === -1 || openBrace >= end) break;

      const header = clean.slice(i, openBrace).trim().replace(/\s+/g, ' ');
      let depth = 0;
      let quote = null;
      let closeBrace = -1;
      for (let j = openBrace; j < end; j += 1) {
        const ch = clean[j];
        if (quote) {
          if (ch === '\\') {
            j += 1;
          } else if (ch === quote) {
            quote = null;
          }
          continue;
        }
        if (ch === '"' || ch === "'") {
          quote = ch;
        } else if (ch === '{') {
          depth += 1;
        } else if (ch === '}') {
          depth -= 1;
          if (depth === 0) {
            closeBrace = j;
            break;
          }
        }
      }
      assert.notEqual(closeBrace, -1, `Unclosed CSS block for "${header}"`);

      if (collectMedia && /^@media\b/i.test(header)) {
        const innerRules = [];
        parseRange(openBrace + 1, closeBrace, innerRules, false);
        mediaBlocks.push({ prelude: header, rules: innerRules });
      } else {
        targetRules.push({
          selector: header,
          body: clean.slice(openBrace + 1, closeBrace).trim(),
        });
      }
      i = closeBrace + 1;
    }
  }

  parseRange(0, clean.length, rules, true);
  return { rules, mediaBlocks };
}

function normalizeSelector(selector) {
  return selector.trim().replace(/\s+/g, ' ');
}

function findRuleBody(rules, selector) {
  const needle = normalizeSelector(selector);
  const bodies = [];
  for (const rule of rules) {
    const parts = rule.selector.split(',').map(normalizeSelector);
    if (normalizeSelector(rule.selector) === needle || parts.includes(needle)) {
      bodies.push(rule.body);
    }
  }
  return bodies.join('\n');
}

const parsedCss = parseCssBlocks(overlayCss);

test('W226 web/ui/overlay.css has balanced braces, parentheses, and comments with zero external URLs', () => {
  assert.ok(overlayCss.length > 0, 'web/ui/overlay.css must not be empty');
  const clean = stripCssComments(overlayCss);

  let braceDepth = 0;
  let parenDepth = 0;
  let quote = null;
  for (let i = 0; i < clean.length; i += 1) {
    const ch = clean[i];
    if (quote) {
      if (ch === '\\') {
        i += 1;
      } else if (ch === quote) {
        quote = null;
      }
      continue;
    }
    if (ch === '"' || ch === "'") {
      quote = ch;
    } else if (ch === '{') {
      braceDepth += 1;
    } else if (ch === '}') {
      braceDepth -= 1;
      assert.ok(braceDepth >= 0, `Extra closing brace "}" at offset ${i}`);
    } else if (ch === '(') {
      parenDepth += 1;
    } else if (ch === ')') {
      parenDepth -= 1;
      assert.ok(parenDepth >= 0, `Extra closing parenthesis ")" at offset ${i}`);
    }
  }

  assert.equal(quote, null, 'Unterminated string literal in web/ui/overlay.css');
  assert.equal(braceDepth, 0, 'Unbalanced curly braces in web/ui/overlay.css');
  assert.equal(parenDepth, 0, 'Unbalanced parentheses in web/ui/overlay.css');

  // Remain COOP/COEP self-contained and free of host paths / game archive names.
  assert.doesNotMatch(overlayCss, /@import\s+url\(\s*['"]?https?:/i);
  assert.doesNotMatch(overlayCss, /url\(\s*['"]?https?:/i);
  assert.doesNotMatch(overlayCss, /\b[A-Za-z]:\\(?:Users|Windows|Program Files|Temp|Games)/i);
  assert.doesNotMatch(overlayCss, /\.(?:rpf|xvd|xvc|ttpf|awc)\b/i);
});

test('W226 enforces CSS layout/paint containment on floating HUD and collapsible diagnostics/log panels', () => {
  const containedSelectors = [
    '#hud',
    '#hud-body',
    '.hud-metrics-strip',
    '.astryx-collapsible-content',
    '#hud-log',
    '#gpu-boot-panel-root',
  ];

  for (const selector of containedSelectors) {
    const body = findRuleBody(parsedCss.rules, selector);
    assert.ok(body.length > 0, `Expected rule for selector "${selector}" in web/ui/overlay.css`);
    assert.match(
      body,
      /\bcontain\s*:\s*layout\s+style\s+paint\s*;/,
      `Selector "${selector}" must declare "contain: layout style paint;" to prevent full-viewport repaints`
    );
  }

  // `.astryx-collapsible` should avoid paint clipping so `.astryx-collapsible-trigger`'s
  // `-8px` horizontal hover bleed remains intact inside `#hud-body`.
  const collapsibleBody = findRuleBody(parsedCss.rules, '.astryx-collapsible');
  assert.doesNotMatch(
    collapsibleBody,
    /\bcontain\s*:[^;]*\bpaint\b/,
    '.astryx-collapsible must not apply paint containment that clips negative-margin trigger hover fills'
  );
});

test('W226 enforces [hidden] and clean-capture HUD visibility rules with display: none !important', () => {
  const hiddenBody = findRuleBody(parsedCss.rules, '[hidden]');
  assert.ok(hiddenBody.length > 0, 'Expected global [hidden] rule in web/ui/overlay.css');
  assert.match(
    hiddenBody,
    /\bdisplay\s*:\s*none\s*!important\s*;/,
    '[hidden] must enforce "display: none !important;"'
  );

  const hudHiddenAttrBody = findRuleBody(parsedCss.rules, '#hud[data-hud-visible="false"]');
  assert.ok(
    hudHiddenAttrBody.length > 0,
    'Expected #hud[data-hud-visible="false"] clean-capture rule in web/ui/overlay.css'
  );
  assert.match(
    hudHiddenAttrBody,
    /\bdisplay\s*:\s*none\s*!important\s*;/,
    '#hud[data-hud-visible="false"] must enforce "display: none !important;"'
  );

  // Preserve existing banner [hidden] selector.
  const bannerHiddenBody = findRuleBody(parsedCss.rules, '.astryx-banner[hidden]');
  assert.match(
    bannerHiddenBody,
    /\bdisplay\s*:\s*none\s*!important\s*;/,
    '.astryx-banner[hidden] must preserve "display: none !important;"'
  );
});

test('W226 prefers-reduced-motion disables transitions and delays for instant HUD and splash hiding', () => {
  const reducedMotionBlock = parsedCss.mediaBlocks.find((block) =>
    /@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)/i.test(block.prelude)
  );
  assert.ok(
    reducedMotionBlock,
    'web/ui/overlay.css must include a @media (prefers-reduced-motion: reduce) block'
  );

  const rootTokensBody = findRuleBody(reducedMotionBlock.rules, ':root');
  assert.match(
    rootTokensBody,
    /--duration-fast\s*:\s*0m?s\s*;/,
    'prefers-reduced-motion must zero out --duration-fast'
  );
  assert.match(
    rootTokensBody,
    /--pg6-splash-fade\s*:\s*0m?s\s*;/,
    'prefers-reduced-motion must zero out --pg6-splash-fade'
  );

  const instantSelectors = [
    '#hud',
    '#splash-overlay-root',
    '.pg6-splash',
    '.pg6-splash__bar-fill',
    '.pg6-splash__stream-fill',
    '.astryx-button',
    '.astryx-progress-bar-fill',
    '.astryx-collapsible-trigger',
    '.astryx-chevron',
    '#gpu-boot-panel-root .pg6-gpu-panel__progress-fill',
  ];

  for (const selector of instantSelectors) {
    const body = findRuleBody(reducedMotionBlock.rules, selector);
    assert.ok(
      body.length > 0,
      `prefers-reduced-motion must cover selector "${selector}"`
    );
    assert.match(
      body,
      /\btransition\s*:\s*none\s*!important\s*;/,
      `prefers-reduced-motion must set "transition: none !important;" on "${selector}"`
    );
    assert.match(
      body,
      /\btransition-delay\s*:\s*0m?s\s*!important\s*;/,
      `prefers-reduced-motion must set "transition-delay: 0ms !important;" on "${selector}"`
    );
    assert.match(
      body,
      /\btransition-duration\s*:\s*0m?s\s*!important\s*;/,
      `prefers-reduced-motion must set "transition-duration: 0ms !important;" on "${selector}"`
    );
  }
});

test('W226 preserves 100% of existing Astryx overlay selectors, tokens, and responsive geometry', () => {
  const rootBody = findRuleBody(parsedCss.rules, ':root');
  const requiredTokens = [
    '--font-family-body',
    '--font-family-heading',
    '--font-family-code',
    '--color-background-body',
    '--color-background-surface',
    '--color-background-card',
    '--color-background-popover',
    '--color-background-muted',
    '--color-background-recessed',
    '--color-accent',
    '--color-on-accent',
    '--color-text-primary',
    '--color-text-strong',
    '--color-text-secondary',
    '--color-success',
    '--color-warning',
    '--color-error',
    '--color-info',
    '--radius-inner',
    '--radius-element',
    '--radius-container',
    '--radius-full',
    '--shadow-med',
    '--shadow-high',
    '--duration-fast',
    '--ease-standard',
  ];

  for (const token of requiredTokens) {
    assert.match(
      rootBody,
      new RegExp(`${token}\\s*:`),
      `Existing CSS custom property "${token}" must remain defined in :root`
    );
  }

  const requiredSelectors = [
    ':root',
    '[data-astryx-theme="astryx"]',
    '#viewport',
    '#hud',
    '#hud.collapsed',
    '#hud.collapsed #hud-body',
    '.hud-header',
    '#hud.collapsed .hud-header',
    '.hud-title-group',
    '.astryx-status-dot',
    '.astryx-status-dot[data-level="ok"]',
    '.astryx-status-dot[data-level="warn"]',
    '.astryx-status-dot[data-level="error"]',
    '.astryx-status-dot[data-level="info"]',
    '.hud-brand',
    '.hud-pill-meta',
    '#hud.collapsed .hud-pill-meta',
    '.hud-actions',
    '.astryx-badge',
    '.astryx-badge[data-boot-mode="real-wasm64"]',
    '.astryx-badge[data-boot-mode="smoke"]',
    '.astryx-badge[data-boot-mode="idle"]',
    '.astryx-button',
    '.astryx-button:focus-visible',
    '.astryx-button[data-size="xs"]',
    '.astryx-button[data-variant="primary"]',
    '.astryx-button[data-variant="secondary"]',
    '.astryx-button[data-variant="ghost"]',
    '.astryx-button[data-active="true"]',
    '.astryx-button[data-recording="true"]',
    '#hud-body',
    '.astryx-banner',
    '.astryx-banner[hidden]',
    '.astryx-banner[data-status="warning"]',
    '.astryx-banner[data-status="error"]',
    '.hud-boot-section',
    '#hud-boot-stage',
    '#hud-status',
    '#hud-boot-elapsed',
    '.astryx-progress-bar-track',
    '.astryx-progress-bar-fill',
    '.hud-side-modules',
    '.hud-metrics-strip',
    '.hud-metric-cell',
    '.hud-metric-value',
    '.hud-gpu-errors[data-state="err"]',
    '.hud-primary-actions',
    '.astryx-collapsible',
    '.astryx-collapsible-trigger',
    '.astryx-chevron',
    '.astryx-collapsible[open] > .astryx-collapsible-trigger .astryx-chevron',
    '.astryx-collapsible-content',
    '#hud-log-disclosure:not([open]) #hud-clear-log',
    '.diag-group:has(#gpu-boot-panel-root:empty)',
    '#hud-log',
    '.log-entry',
    '.diag-group',
    '.badge-grid',
    '.kv-list',
    '.kv-row',
    '.kv-value',
    '.controls-grid',
    '.control-row',
    '.astryx-select',
    '#gpu-boot-panel-root',
    '#gpu-boot-panel-root:empty',
    '#gpu-boot-panel-root .pg6-gpu-panel',
    '#gpu-boot-panel-root .pg6-gpu-panel__progress-fill',
  ];

  for (const selector of requiredSelectors) {
    assert.ok(
      findRuleBody(parsedCss.rules, selector).length > 0,
      `Existing overlay selector "${selector}" must remain present without regression`
    );
  }

  // Responsive viewport-clamped HUD dimensions remain intact.
  const hudBody = findRuleBody(parsedCss.rules, '#hud');
  assert.match(hudBody, /width:\s*min\(320px,\s*calc\(100vw - 24px\)\)/);
  assert.match(hudBody, /max-width:\s*320px/);
  assert.match(hudBody, /backdrop-filter:\s*blur\(18px\)/);

  const hudCollapsedBody = findRuleBody(parsedCss.rules, '#hud.collapsed');
  assert.match(hudCollapsedBody, /max-width:\s*calc\(100vw - 24px\)/);

  const hudScrollBody = findRuleBody(parsedCss.rules, '#hud-body');
  assert.match(hudScrollBody, /max-height:\s*calc\(100vh - 72px\)/);
});

test('W232 web/index.html defines a clean inline SVG data-URI favicon in <head> to prevent /favicon.ico 404s', () => {
  const headMatch = /<head>([\s\S]*?)<\/head>/i.exec(indexHtml);
  assert.ok(headMatch, 'web/index.html must contain a <head> block');
  const headHtml = headMatch[1];

  const iconMatch = /<link\b([^>]*\brel="icon"[^>]*)\/?>/i.exec(headHtml);
  assert.ok(iconMatch, 'web/index.html <head> must contain a <link rel="icon"> tag');
  const attrs = iconMatch[1];

  assert.match(attrs, /\btype="image\/svg\+xml"/i, 'Favicon link must declare type="image/svg+xml"');
  const hrefMatch = /\bhref="(data:image\/svg\+xml,[^"]+)"/i.exec(attrs);
  assert.ok(hrefMatch, 'Favicon link must use an inline data:image/svg+xml,... URI');

  const dataUri = hrefMatch[1];
  const encodedSvg = dataUri.slice('data:image/svg+xml,'.length);

  // Must be compact and safely percent-encoded (no raw #, <, >, or whitespace).
  assert.ok(encodedSvg.length > 0 && encodedSvg.length < 512, 'Inline SVG favicon payload must remain compact');
  assert.doesNotMatch(encodedSvg, /[#<>\s]/, 'Inline SVG data-URI must percent-encode #, <, >, and spaces');

  const decodedSvg = decodeURIComponent(encodedSvg);
  assert.match(decodedSvg, /^<svg\b[^>]*xmlns=['"]http:\/\/www\.w3\.org\/2000\/svg['"][^>]*>/);
  assert.match(decodedSvg, /\bviewBox=['"]0 0 32 32['"]/);
  assert.match(decodedSvg, /<\/svg>$/);
  assert.match(decodedSvg, /<(?:rect|path|polygon|circle)\b/, 'Favicon must use minimal geometric SVG primitives');
  assert.doesNotMatch(decodedSvg, /<script\b|on[a-z]+\s*=|https?:\/\/(?!www\.w3\.org\/2000\/svg)|\bid=/i);
});

test('W232/W252 web/ui/overlay.css provides smooth fade-out helpers and Astryx-matched splash card styling', () => {
  const rootBody = findRuleBody(parsedCss.rules, ':root');
  for (const token of [
    '--pg6-splash-fade',
    '--pg6-splash-ease',
    '--pg6-splash-backdrop',
    '--pg6-splash-backdrop-strong',
    '--pg6-splash-pill-border',
  ]) {
    assert.match(
      rootBody,
      new RegExp(`${token}\\s*:`),
      `Splash custom property "${token}" must be defined in :root`
    );
  }

  // Smooth fade-out transition helpers remain 100% intact.
  const fadeTransitionBody = findRuleBody(parsedCss.rules, '.pg6-fade-transition');
  assert.match(fadeTransitionBody, /\bopacity\s*:\s*1\s*;/);
  assert.match(fadeTransitionBody, /\btransition\s*:[\s\S]*opacity\s+var\(--pg6-splash-fade/);

  const fadeOutBody = findRuleBody(parsedCss.rules, '.pg6-fade-out');
  assert.match(fadeOutBody, /\bopacity\s*:\s*0\s*;/);
  assert.match(fadeOutBody, /\bpointer-events\s*:\s*none\s*;/);
  assert.match(fadeOutBody, /\btransition\s*:[\s\S]*opacity\s+var\(--pg6-splash-fade/);

  const splashRootFadeBody = findRuleBody(parsedCss.rules, '#splash-overlay-root .pg6-splash');
  assert.match(splashRootFadeBody, /\btransition\s*:\s*opacity\s+var\(--pg6-splash-fade/);

  const splashRootFadingBody = findRuleBody(
    parsedCss.rules,
    '#splash-overlay-root .pg6-splash.pg6-splash--fading'
  );
  assert.match(splashRootFadingBody, /\bopacity\s*:\s*0\s*;/);
  assert.match(splashRootFadingBody, /\bpointer-events\s*:\s*none\s*;/);

  // Helper classes align with Astryx borderless surface tokens.
  const frostedBody = findRuleBody(parsedCss.rules, '.pg6-splash-frosted');
  assert.match(frostedBody, /\bbackground\s*:\s*var\(--color-background-surface\)\s*;/);
  assert.match(frostedBody, /\bborder\s*:\s*none\s*;/);
  assert.match(frostedBody, /\bborder-radius\s*:\s*var\(--radius-container\)\s*;/);
  assert.match(frostedBody, /\bbox-shadow\s*:\s*var\(--shadow-high\)\s*;/);
  assert.match(frostedBody, /\bbackdrop-filter\s*:\s*blur\(18px\)\s*;/);
  assert.match(frostedBody, /-webkit-backdrop-filter\s*:\s*blur\(18px\)\s*;/);

  const statusPillBody = findRuleBody(parsedCss.rules, '.pg6-splash-status-pill');
  assert.match(statusPillBody, /\bborder\s*:\s*none\s*;/);
  assert.match(statusPillBody, /\bborder-radius\s*:\s*var\(--radius-full\)\s*;/);
  assert.match(statusPillBody, /\bbackground\s*:\s*var\(--color-background-surface\)\s*;/);
  assert.match(statusPillBody, /\bbackdrop-filter\s*:\s*blur\(18px\)\s*;/);
  assert.match(statusPillBody, /-webkit-backdrop-filter\s*:\s*blur\(18px\)\s*;/);

  // W252: Splash panel matches #hud Astryx elevated borderless card 1:1.
  const splashPanelBody = findRuleBody(parsedCss.rules, '#splash-overlay-root .pg6-splash__panel');
  assert.match(splashPanelBody, /\bwidth\s*:\s*min\(340px,\s*calc\(100vw - 24px\)\)\s*;/);
  assert.match(splashPanelBody, /\bmax-width\s*:\s*340px\s*;/);
  assert.match(splashPanelBody, /\bpadding\s*:\s*12px\s+14px\s*;/);
  assert.match(splashPanelBody, /\bgap\s*:\s*8px\s*;/);
  assert.match(splashPanelBody, /\bbackground\s*:\s*var\(--color-background-surface\)\s*;/);
  assert.match(splashPanelBody, /\bborder\s*:\s*none\s*;/);
  assert.match(splashPanelBody, /\bborder-radius\s*:\s*var\(--radius-container\)\s*;/);
  assert.match(splashPanelBody, /\bbox-shadow\s*:\s*var\(--shadow-high\)\s*;/);
  assert.match(splashPanelBody, /\bbackdrop-filter\s*:\s*blur\(18px\)\s*;/);
  assert.match(splashPanelBody, /-webkit-backdrop-filter\s*:\s*blur\(18px\)\s*;/);
  assert.match(splashPanelBody, /\bcolor\s*:\s*var\(--color-text-primary\)\s*;/);
  assert.match(splashPanelBody, /\bfont-family\s*:\s*var\(--font-family-body\)\s*;/);
  assert.match(splashPanelBody, /\bfont-size\s*:\s*12px\s*;/);
  assert.match(splashPanelBody, /\bline-height\s*:\s*1\.4\s*;/);
  assert.match(splashPanelBody, /\bcontain\s*:\s*layout\s+style\s+paint\s*;/);

  // W252: No nested dark inner pill box inside the splash card.
  for (const rowSelector of [
    '#splash-overlay-root .pg6-splash__stage-row',
    '#splash-overlay-root .pg6-splash__status-row',
  ]) {
    const rowBody = findRuleBody(parsedCss.rules, rowSelector);
    assert.match(rowBody, /\bbackground\s*:\s*transparent\s*;/);
    assert.match(rowBody, /\bborder\s*:\s*none\s*;/);
    assert.match(rowBody, /\bpadding\s*:\s*0\s*;/);
    assert.match(rowBody, /\bbox-shadow\s*:\s*none\s*;/);
    assert.match(rowBody, /\bbackdrop-filter\s*:\s*none\s*;/);
  }

  // W252: Crisp Astryx stage & meta typography.
  const splashStageBody = findRuleBody(parsedCss.rules, '#splash-overlay-root .pg6-splash__stage');
  assert.match(splashStageBody, /\bfont-family\s*:\s*var\(--font-family-body\)\s*;/);
  assert.match(splashStageBody, /\bfont-size\s*:\s*12px\s*;/);
  assert.match(splashStageBody, /\bfont-weight\s*:\s*var\(--font-weight-medium\)\s*;/);
  assert.match(splashStageBody, /\bcolor\s*:\s*var\(--color-text-primary\)\s*;/);

  const splashMetaBody = findRuleBody(parsedCss.rules, '#splash-overlay-root .pg6-splash__meta');
  assert.match(splashMetaBody, /\bfont-size\s*:\s*11px\s*;/);
  assert.match(splashMetaBody, /\bcolor\s*:\s*var\(--color-text-secondary\)\s*;/);
  assert.match(splashMetaBody, /\bfont-variant-numeric\s*:\s*tabular-nums\s*;/);

  // W252: Astryx 4px monochrome progress bar matching #hud (zero neon blue).
  const splashBarBody = findRuleBody(parsedCss.rules, '#splash-overlay-root .pg6-splash__bar');
  assert.match(splashBarBody, /\bheight\s*:\s*4px\s*;/);
  assert.match(splashBarBody, /\bborder\s*:\s*none\s*;/);
  assert.match(splashBarBody, /\bborder-radius\s*:\s*var\(--radius-full\)\s*;/);
  assert.match(splashBarBody, /\bbackground\s*:\s*var\(--color-track\)\s*;/);

  const splashBarFillBody = findRuleBody(parsedCss.rules, '#splash-overlay-root .pg6-splash__bar-fill');
  assert.match(splashBarFillBody, /\bbackground\s*:\s*var\(--color-accent\)\s*;/);
  assert.doesNotMatch(splashBarFillBody, /#6db3ff/i);

  const splashBarOkBody = findRuleBody(
    parsedCss.rules,
    '#splash-overlay-root .pg6-splash__bar-fill[data-state="ok"]'
  );
  assert.match(splashBarOkBody, /\bbackground\s*:\s*var\(--color-success\)\s*;/);

  // W252: Astryx 14px monochrome spinner + crisp 22px Spinning Greg icon.
  const splashSpinnerBody = findRuleBody(parsedCss.rules, '#splash-overlay-root .pg6-splash__spinner');
  assert.match(splashSpinnerBody, /\bwidth\s*:\s*14px\s*;/);
  assert.match(splashSpinnerBody, /\bheight\s*:\s*14px\s*;/);
  assert.match(splashSpinnerBody, /\btransition\s*:/);

  const splashSpinnerRingBody = findRuleBody(
    parsedCss.rules,
    '#splash-overlay-root .pg6-splash__spinner-ring'
  );
  assert.match(splashSpinnerRingBody, /\bwidth\s*:\s*14px\s*;/);
  assert.match(splashSpinnerRingBody, /\bheight\s*:\s*14px\s*;/);
  assert.match(splashSpinnerRingBody, /\bborder\s*:\s*2px\s+solid\s+var\(--color-track\)\s*;/);
  assert.match(splashSpinnerRingBody, /\bborder-top-color\s*:\s*var\(--color-accent\)\s*;/);

  const splashSpinnerGregBody = findRuleBody(
    parsedCss.rules,
    '#splash-overlay-root .pg6-splash__spinner--greg'
  );
  assert.match(splashSpinnerGregBody, /\bwidth\s*:\s*22px\s*;/);
  assert.match(splashSpinnerGregBody, /\bheight\s*:\s*22px\s*;/);

  const splashGregBody = findRuleBody(parsedCss.rules, '#splash-overlay-root .pg6-splash__greg');
  assert.match(splashGregBody, /\bwidth\s*:\s*22px\s*;/);
  assert.match(splashGregBody, /\bheight\s*:\s*22px\s*;/);
  assert.match(splashGregBody, /\bborder\s*:\s*none\s*;/);

  assert.doesNotMatch(overlayCss, /#6db3ff/i, 'overlay.css must not use legacy neon blue #6db3ff');

  // Reduced-motion block must also disable transitions on the splash helpers and bar fills.
  const reducedMotionBlock = parsedCss.mediaBlocks.find((block) =>
    /@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)/i.test(block.prelude)
  );
  assert.ok(reducedMotionBlock, 'Expected prefers-reduced-motion block');
  for (const selector of [
    '.pg6-fade-transition',
    '.pg6-fade-out',
    '#splash-overlay-root .pg6-splash',
    '#splash-overlay-root .pg6-splash__spinner',
    '#splash-overlay-root .pg6-splash__bar-fill',
    '#splash-overlay-root .pg6-splash__stream-fill',
  ]) {
    const body = findRuleBody(reducedMotionBlock.rules, selector);
    assert.match(
      body,
      /\btransition\s*:\s*none\s*!important\s*;/,
      `prefers-reduced-motion must disable transitions on "${selector}"`
    );
  }
});

test('viewport preserves 16:9 console aspect ratio with object-fit: contain and pure black letterbox background', () => {
  const viewportBody = findRuleBody(parsedCss.rules, '#viewport');
  assert.ok(viewportBody.length > 0, 'Expected #viewport rule in web/ui/overlay.css');
  assert.match(viewportBody, /\bposition\s*:\s*fixed\s*;/);
  assert.match(viewportBody, /\binset\s*:\s*0\s*;/);
  assert.match(viewportBody, /\bwidth\s*:\s*100%\s*;/);
  assert.match(viewportBody, /\bheight\s*:\s*100%\s*;/);
  assert.match(
    viewportBody,
    /\bobject-fit\s*:\s*contain\s*;/,
    '#viewport must declare "object-fit: contain;" to prevent non-uniform 16:9 stretching across wide windows'
  );
  assert.match(viewportBody, /\bobject-position\s*:\s*center\s*;/);
  assert.match(
    viewportBody,
    /\bbackground\s*:\s*#000(?:000)?\s*;/i,
    '#viewport must use pure black #000 letterbox/pillarbox background'
  );

  const fullscreenBody = findRuleBody(parsedCss.rules, '#viewport:fullscreen');
  assert.match(fullscreenBody, /\bobject-fit\s*:\s*contain\s*;/);
  assert.match(fullscreenBody, /\bbackground\s*:\s*#000(?:000)?\s*;/i);

  const backdropBody = findRuleBody(parsedCss.rules, '#viewport::backdrop');
  assert.match(backdropBody, /\bbackground\s*:\s*#000(?:000)?\s*;/i);
});
