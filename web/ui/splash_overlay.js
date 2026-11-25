'use strict';

// Browser local splash screen & live boot-stage overlay for playgta6web.
// Bridges the moment the user opens the local server to the first real WebGPU
// frame (FIRST_PRESENT / BinkVideoPresenter). Accepts BootTracer (W40) snapshots
// for progress, stage labels, streaming readiness, and unresolved-import / trap
// diagnostics. Never bundles copyrighted game splash images — only an optional
// runtime splashImageUrl (e.g. /game/xbs_splash.png) with a clean CSS/SVG fallback.
// Zero external dependencies; UMD export for Browser (globalThis.PlayGta6SplashOverlay)
// and Node.js (module.exports).

(function (root) {
  /** Guest boot milestones in chronological order (mirrors BootTracer BOOT_MILESTONES). */
  const BOOT_STAGE_ORDER = Object.freeze([
    'PGIM_LOAD',
    'TLS_INIT',
    'CRT_INITTERM',
    'ENTRY_CALL',
    'XMEM_INIT',
    'GDK_INIT',
    'D3D12_CREATE_DEVICE',
    'BINK_OPEN',
    'FIRST_PRESENT',
  ]);

  /** Human-readable labels for each boot stage. */
  const BOOT_STAGE_LABELS = Object.freeze({
    NOT_STARTED: 'Waiting to start',
    PGIM_LOAD: 'Loading guest image',
    TLS_INIT: 'Initializing TLS',
    CRT_INITTERM: 'Running CRT startup',
    ENTRY_CALL: 'Calling guest entry',
    XMEM_INIT: 'Initializing XMem',
    GDK_INIT: 'Initializing GDK runtime',
    D3D12_CREATE_DEVICE: 'Creating D3D12 device',
    BINK_OPEN: 'Opening Bink stream',
    FIRST_PRESENT: 'First frame presented',
  });

  const DEFAULT_FADE_MS = 400;
  const DEFAULT_GREG_IMAGE_URL = './assets/greg.png';
  const DEFAULT_GREG_CLICK_THRESHOLD = 3;
  const GREG_SECRET_WORD = 'greg';
  const STAGE_NOT_STARTED = 'NOT_STARTED';
  const STYLE_ATTR = 'data-pg6-splash-style';
  const ROOT_CLASS = 'pg6-splash';
  const REDUCED_MOTION_QUERY = '(prefers-reduced-motion: reduce)';
  const SCRIPTED_INPUT_PRESET_RE = /^[a-z0-9][a-z0-9_-]{0,31}$/i;
  const DISMISS_REASON_RE = /^[a-z0-9][a-z0-9_.:-]{0,63}$/i;
  const HUD_HIDDEN_VALUES = new Set(['0', 'false', 'off', 'no', 'none', 'hidden']);
  const GREG_ENABLED_VALUES = new Set(['1', 'true', 'on', 'yes']);
  const DISMISS_REASONS = Object.freeze({
    FIRST_PRESENT: 'first_present',
    IMMEDIATE: 'immediate',
    HUD_HIDDEN: 'hud_hidden',
    INSTANT_HIDE: 'instant_hide',
    REDUCED_MOTION: 'reduced_motion',
    MANUAL: 'manual',
  });

  const FALLBACK_SVG =
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 360" role="img" aria-label="playgta6web">' +
    '<defs><linearGradient id="pg6g" x1="0" y1="0" x2="1" y2="1">' +
    '<stop offset="0%" stop-color="#0b1220"/><stop offset="55%" stop-color="#132033"/>' +
    '<stop offset="100%" stop-color="#1a2a3a"/></linearGradient></defs>' +
    '<rect width="640" height="360" fill="url(#pg6g)"/>' +
    '<text x="320" y="176" text-anchor="middle" fill="#e8eef7" font-family="Segoe UI, system-ui, sans-serif" ' +
    'font-size="36" font-weight="600" letter-spacing="0.08em">playgta6web</text>' +
    '<text x="320" y="214" text-anchor="middle" fill="#8fa3bc" font-family="Segoe UI, system-ui, sans-serif" ' +
    'font-size="14" letter-spacing="0.18em">LOCAL BOOT</text></svg>';

  const SHARED_CSS =
    '.' +
    ROOT_CLASS +
    '{' +
    'position:fixed;inset:0;z-index:9999;display:flex;flex-direction:column;align-items:center;' +
    'justify-content:flex-end;box-sizing:border-box;padding:0 1.5rem 2.5rem;' +
    'background:#0b1220;color:#e8eef7;font-family:Segoe UI,system-ui,sans-serif;' +
    'opacity:1;transition:opacity var(--pg6-splash-fade,400ms) ease;' +
    'pointer-events:auto;' +
    '}' +
    '.' +
    ROOT_CLASS +
    '[hidden]{display:none!important;}' +
    '.' +
    ROOT_CLASS +
    '.pg6-splash--fading{opacity:0;pointer-events:none;}' +
    '.' +
    ROOT_CLASS +
    '__media{position:absolute;inset:0;overflow:hidden;}' +
    '.' +
    ROOT_CLASS +
    '__img,.pg6-splash__fallback{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;}' +
    '.' +
    ROOT_CLASS +
    '__fallback{display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#0b1220,#1a2a3a);}' +
    '.' +
    ROOT_CLASS +
    '__fallback svg{width:100%;height:100%;}' +
    '.' +
    ROOT_CLASS +
    '__panel{position:relative;z-index:1;width:min(36rem,100%);display:flex;flex-direction:column;gap:0.55rem;}' +
    '.' +
    ROOT_CLASS +
    '__stage-row{display:flex;align-items:center;gap:0.6rem;min-height:1.5rem;}' +
    '.' +
    ROOT_CLASS +
    '__spinner{position:relative;flex:0 0 auto;width:1.25rem;height:1.25rem;display:inline-flex;' +
    'align-items:center;justify-content:center;cursor:pointer;user-select:none;}' +
    '.' +
    ROOT_CLASS +
    '__spinner--greg{width:1.65rem;height:1.65rem;}' +
    '.' +
    ROOT_CLASS +
    '__spinner-ring{box-sizing:border-box;width:1.1rem;height:1.1rem;border-radius:50%;' +
    'border:2px solid rgba(255,255,255,0.18);border-top-color:#6db3ff;animation:pg6-spin 850ms linear infinite;}' +
    '.' +
    ROOT_CLASS +
    '__greg{width:1.65rem;height:1.65rem;object-fit:contain;border-radius:50%;' +
    'display:block;animation:pg6-spin 1200ms linear infinite;}' +
    '.' +
    ROOT_CLASS +
    '__spinner[hidden],.' +
    ROOT_CLASS +
    '__spinner-ring[hidden],.' +
    ROOT_CLASS +
    '__greg[hidden]{display:none!important;}' +
    '.' +
    ROOT_CLASS +
    '__stage{font-size:0.95rem;letter-spacing:0.04em;opacity:0.95;}' +
    '.' +
    ROOT_CLASS +
    '__bar,.pg6-splash__stream{height:0.4rem;border-radius:999px;background:rgba(255,255,255,0.12);overflow:hidden;}' +
    '.' +
    ROOT_CLASS +
    '__bar-fill,.pg6-splash__stream-fill{height:100%;width:0%;background:#6db3ff;transition:width 180ms ease;}' +
    '.' +
    ROOT_CLASS +
    '__stream{opacity:0.85;}' +
    '.' +
    ROOT_CLASS +
    '__stream[hidden],.pg6-splash__badge[hidden],.pg6-splash__scripted[hidden]{display:none!important;}' +
    '.' +
    ROOT_CLASS +
    '__scripted{align-self:flex-start;padding:0.2rem 0.55rem;border-radius:999px;' +
    'background:rgba(109,179,255,0.16);border:1px solid rgba(109,179,255,0.4);' +
    'color:#c8e3ff;font-size:0.74rem;letter-spacing:0.03em;line-height:1.25;}' +
    '.' +
    ROOT_CLASS +
    '__meta{display:flex;justify-content:space-between;gap:1rem;font-size:0.78rem;opacity:0.8;}' +
    '.' +
    ROOT_CLASS +
    '__badge{margin-top:0.15rem;padding:0.45rem 0.65rem;border-radius:0.4rem;' +
    'background:rgba(180,60,60,0.22);border:1px solid rgba(255,140,140,0.35);' +
    'color:#ffc9c9;font-size:0.78rem;line-height:1.35;}' +
    '@keyframes pg6-spin{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}' +
    '.' +
    ROOT_CLASS +
    '[data-reduced-motion="true"] .' +
    ROOT_CLASS +
    '__spinner-ring,.' +
    ROOT_CLASS +
    '[data-reduced-motion="true"] .' +
    ROOT_CLASS +
    '__greg{animation:none!important;}' +
    '@media (prefers-reduced-motion: reduce){.' +
    ROOT_CLASS +
    ',.' +
    ROOT_CLASS +
    '__bar-fill,.' +
    ROOT_CLASS +
    '__stream-fill{transition:none!important;}.' +
    ROOT_CLASS +
    '__spinner-ring,.' +
    ROOT_CLASS +
    '__greg{animation:none!important;}}';

  /**
   * Strictly validate a scripted-input preset identifier.
   * Rejects paths, URLs, control characters, whitespace-containing text, and
   * non-string values to `null`.
   *
   * @param {unknown} raw
   * @returns {string|null}
   */
  function sanitizeScriptedInputPreset(raw) {
    if (typeof raw !== 'string') {
      return null;
    }
    const trimmed = raw.trim();
    if (!trimmed || /[\x00-\x1F\x7F]/.test(trimmed)) {
      return null;
    }
    if (!SCRIPTED_INPUT_PRESET_RE.test(trimmed)) {
      return null;
    }
    return trimmed;
  }

  /**
   * Strictly validate a dismissal reason tag for telemetry.
   * Rejects paths, URLs, control characters, whitespace, and oversized strings.
   *
   * @param {unknown} raw
   * @returns {string|null}
   */
  function sanitizeDismissReason(raw) {
    if (typeof raw !== 'string') {
      return null;
    }
    const trimmed = raw.trim();
    if (!trimmed || /[\x00-\x1F\x7F]/.test(trimmed)) {
      return null;
    }
    if (!DISMISS_REASON_RE.test(trimmed)) {
      return null;
    }
    return trimmed;
  }

  function readLastQueryParam(search, targetKey) {
    if (search === null || search === undefined || search === '') {
      return null;
    }
    try {
      const normalizedTarget = String(targetKey).trim().toLowerCase();
      let params = null;
      if (typeof URLSearchParams === 'function' && search instanceof URLSearchParams) {
        params = search;
      } else {
        let raw = null;
        if (typeof search === 'string') {
          raw = search;
        } else if (typeof search === 'object' && typeof search.search === 'string') {
          raw = search.search;
        }
        if (typeof raw !== 'string' || raw.length === 0) {
          return null;
        }
        const hashIdx = raw.indexOf('#');
        const withoutHash = hashIdx >= 0 ? raw.slice(0, hashIdx) : raw;
        const qIdx = withoutHash.indexOf('?');
        const query = qIdx >= 0 ? withoutHash.slice(qIdx + 1) : withoutHash;
        if (!query) {
          return null;
        }
        if (typeof URLSearchParams === 'function') {
          params = new URLSearchParams(query);
        } else {
          let lastValue = null;
          for (const part of query.split('&')) {
            if (!part) continue;
            const eqIdx = part.indexOf('=');
            const rawKey = eqIdx >= 0 ? part.slice(0, eqIdx) : part;
            const rawVal = eqIdx >= 0 ? part.slice(eqIdx + 1) : '';
            const key = decodeURIComponent(rawKey).trim().toLowerCase();
            if (key === normalizedTarget) {
              lastValue = decodeURIComponent(rawVal).trim().toLowerCase();
            }
          }
          return lastValue;
        }
      }

      if (!params) {
        return null;
      }
      let lastValue = null;
      for (const [key, value] of params.entries()) {
        if (String(key).trim().toLowerCase() === normalizedTarget) {
          lastValue = String(value).trim().toLowerCase();
        }
      }
      return lastValue;
    } catch {
      return null;
    }
  }

  /**
   * Detect whether a URL query string / URLSearchParams / Location disables the HUD
   * (e.g. `?hud=0`, `?hud=false`, `?hud=off`, `?hud=no`). Never throws.
   *
   * @param {unknown} search
   * @returns {boolean}
   */
  function isHudHiddenQuery(search) {
    const value = readLastQueryParam(search, 'hud');
    return value !== null && HUD_HIDDEN_VALUES.has(value);
  }

  /**
   * Detect whether a URL query string / URLSearchParams / Location enables the
   * Spinning Greg easter egg (e.g. `?greg=1`, `?greg=true`, `?greg=on`, `?greg=yes`).
   * Never throws.
   *
   * @param {unknown} search
   * @returns {boolean}
   */
  function isGregQuery(search) {
    const value = readLastQueryParam(search, 'greg');
    return value !== null && GREG_ENABLED_VALUES.has(value);
  }

  function isEditableEventTarget(target) {
    if (!target || typeof target !== 'object') {
      return false;
    }
    if (target.isContentEditable === true) {
      return true;
    }
    if (typeof target.getAttribute === 'function') {
      try {
        const ce = target.getAttribute('contenteditable');
        if (ce === '' || ce === 'true' || ce === 'plaintext-only') {
          return true;
        }
      } catch {
        // ignore
      }
    }
    const tag = typeof target.tagName === 'string' ? target.tagName.toUpperCase() : '';
    return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
  }

  function resolveSearchSource(opts, doc) {
    if (opts.locationSearch !== undefined && opts.locationSearch !== null) {
      return opts.locationSearch;
    }
    if (opts.search !== undefined && opts.search !== null) {
      return opts.search;
    }
    try {
      if (opts.windowObj && opts.windowObj.location && typeof opts.windowObj.location.search === 'string') {
        return opts.windowObj.location.search;
      }
    } catch {
      // ignore
    }
    try {
      if (doc && doc.defaultView && doc.defaultView.location && typeof doc.defaultView.location.search === 'string') {
        return doc.defaultView.location.search;
      }
    } catch {
      // ignore
    }
    try {
      if (doc && doc.location && typeof doc.location.search === 'string') {
        return doc.location.search;
      }
    } catch {
      // ignore
    }
    try {
      if (typeof window !== 'undefined' && window && window.location && typeof window.location.search === 'string') {
        return window.location.search;
      }
    } catch {
      // ignore
    }
    return null;
  }

  function clampPct(value) {
    const n = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(n)) return 0;
    if (n <= 0) return 0;
    if (n >= 100) return 100;
    return n;
  }

  function stageIndex(stage) {
    if (typeof stage !== 'string' || stage.length === 0) return -1;
    const upper = stage.trim().toUpperCase();
    if (upper === STAGE_NOT_STARTED) return -1;
    return BOOT_STAGE_ORDER.indexOf(upper);
  }

  function labelForStage(stage) {
    if (typeof stage !== 'string' || stage.length === 0) {
      return BOOT_STAGE_LABELS.NOT_STARTED;
    }
    const upper = stage.trim().toUpperCase();
    return BOOT_STAGE_LABELS[upper] || upper;
  }

  /**
   * Compute a 0..100 boot progress percentage from a BootTracer-style snapshot.
   * Prefers an explicit `progressPct`, otherwise derives it from `stage` /
   * `milestonesReached` against `BOOT_STAGE_ORDER`.
   *
   * @param {object|null|undefined} snapshot
   * @returns {number}
   */
  function computeBootProgress(snapshot) {
    if (!snapshot || typeof snapshot !== 'object') {
      return 0;
    }
    if (snapshot.progressPct !== undefined && snapshot.progressPct !== null) {
      return Math.round(clampPct(snapshot.progressPct));
    }
    const idx = stageIndex(snapshot.stage);
    if (idx >= 0) {
      return Math.round(((idx + 1) / BOOT_STAGE_ORDER.length) * 100);
    }
    const reached = Number(snapshot.milestonesReached);
    if (Number.isFinite(reached) && reached > 0) {
      return Math.round(clampPct((reached / BOOT_STAGE_ORDER.length) * 100));
    }
    return 0;
  }

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function formatElapsed(ms) {
    const n = typeof ms === 'number' && Number.isFinite(ms) ? Math.max(0, ms) : 0;
    if (n < 1000) return `${Math.round(n)} ms`;
    return `${(n / 1000).toFixed(1)} s`;
  }

  function streamingPct(streamingProgress) {
    if (streamingProgress === undefined || streamingProgress === null) return null;
    if (typeof streamingProgress === 'number') return clampPct(streamingProgress);
    if (typeof streamingProgress === 'object') {
      if (streamingProgress.pct !== undefined) return clampPct(streamingProgress.pct);
      if (streamingProgress.progressPct !== undefined) return clampPct(streamingProgress.progressPct);
      const loaded = Number(streamingProgress.loadedBytes);
      const total = Number(streamingProgress.totalBytes);
      if (Number.isFinite(loaded) && Number.isFinite(total) && total > 0) {
        return clampPct((loaded / total) * 100);
      }
    }
    return null;
  }

  function formatTrapBadge(traps, unresolvedImports) {
    const imports = asArray(unresolvedImports);
    const trapList = asArray(traps);
    const parts = [];

    if (imports.length > 0) {
      const top = imports
        .slice(0, 3)
        .map((entry) => {
          if (typeof entry === 'string') return entry;
          if (entry && typeof entry === 'object') {
            return entry.name || (entry.dll && entry.symbol ? `${entry.dll}!${entry.symbol}` : entry.symbol) || null;
          }
          return null;
        })
        .filter(Boolean);
      if (top.length > 0) {
        parts.push(`Unresolved imports (${imports.length}): ${top.join(', ')}`);
      } else {
        parts.push(`Unresolved imports: ${imports.length}`);
      }
    }

    if (trapList.length > 0) {
      const latest = trapList[trapList.length - 1];
      const kind =
        latest && typeof latest === 'object'
          ? latest.kind || 'TRAP'
          : typeof latest === 'string'
            ? latest
            : 'TRAP';
      parts.push(`Traps: ${trapList.length} (latest ${kind})`);
    }

    return parts.length > 0 ? parts.join(' · ') : '';
  }

  function ensureSharedStyle(doc) {
    if (!doc || typeof doc.createElement !== 'function') return null;
    const head = doc.head || (doc.getElementsByTagName && doc.getElementsByTagName('head')[0]) || null;
    if (!head) return null;
    if (typeof doc.querySelector === 'function') {
      const existing = doc.querySelector(`style[${STYLE_ATTR}]`);
      if (existing) return existing;
    }
    const style = doc.createElement('style');
    style.setAttribute(STYLE_ATTR, '1');
    style.textContent = SHARED_CSS;
    if (typeof head.appendChild === 'function') {
      head.appendChild(style);
    }
    return style;
  }

  function createEl(doc, tag, className, attrs) {
    const el = doc.createElement(tag);
    if (className) el.className = className;
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null) continue;
        if (key === 'text') {
          el.textContent = String(value);
        } else if (key === 'html') {
          el.innerHTML = String(value);
        } else {
          el.setAttribute(key, String(value));
        }
      }
    }
    return el;
  }

  /**
   * Controllable splash / boot-stage overlay.
   */
  class SplashOverlayController {
    /**
     * @param {object} [options]
     * @param {Document|object} [options.documentObj]
     * @param {Window|object|null} [options.windowObj]
     * @param {Element|object|null} [options.parent]
     * @param {Element|object|null} [options.root]
     * @param {string|null} [options.splashImageUrl]
     * @param {number} [options.fadeMs]
     * @param {(() => void)|null} [options.onDismiss]
     * @param {((fn: Function, ms: number) => any)} [options.setTimeout]
     * @param {((id: any) => void)} [options.clearTimeout]
     * @param {(() => number)} [options.nowFn]
     * @param {boolean} [options.reducedMotion]
     * @param {((query: string) => { matches?: boolean } | null)} [options.matchMedia]
     * @param {boolean} [options.instantHide]
     * @param {string|URLSearchParams|object|null} [options.locationSearch]
     * @param {string|URLSearchParams|object|null} [options.search]
     * @param {string|null} [options.scriptedInput]
     * @param {boolean} [options.gregMode]
     * @param {number} [options.gregClickThreshold=3]
     * @param {string|null} [options.gregImageUrl]
     * @param {boolean} [options.autoMount=true]
     * @param {boolean} [options.visible=true]
     */
    constructor(options = {}) {
      if (options !== undefined && (options === null || typeof options !== 'object')) {
        throw new TypeError('SplashOverlayController options must be an object');
      }

      const opts = options || {};
      const doc =
        opts.documentObj ||
        (typeof document !== 'undefined' ? document : null) ||
        (root && root.document) ||
        null;
      if (!doc || typeof doc.createElement !== 'function') {
        throw new TypeError('SplashOverlayController requires a document with createElement');
      }

      if (opts.onDismiss !== undefined && opts.onDismiss !== null && typeof opts.onDismiss !== 'function') {
        throw new TypeError('onDismiss must be a function or null');
      }
      if (opts.setTimeout !== undefined && typeof opts.setTimeout !== 'function') {
        throw new TypeError('setTimeout must be a function');
      }
      if (opts.clearTimeout !== undefined && typeof opts.clearTimeout !== 'function') {
        throw new TypeError('clearTimeout must be a function');
      }
      if (opts.nowFn !== undefined && opts.nowFn !== null && typeof opts.nowFn !== 'function') {
        throw new TypeError('nowFn must be a function');
      }
      if (
        opts.reducedMotion !== undefined &&
        opts.reducedMotion !== null &&
        typeof opts.reducedMotion !== 'boolean'
      ) {
        throw new TypeError('reducedMotion must be a boolean');
      }
      if (
        opts.matchMedia !== undefined &&
        opts.matchMedia !== null &&
        typeof opts.matchMedia !== 'function'
      ) {
        throw new TypeError('matchMedia must be a function');
      }
      if (
        opts.instantHide !== undefined &&
        opts.instantHide !== null &&
        typeof opts.instantHide !== 'boolean'
      ) {
        throw new TypeError('instantHide must be a boolean');
      }
      if (
        opts.windowObj !== undefined &&
        opts.windowObj !== null &&
        typeof opts.windowObj !== 'object'
      ) {
        throw new TypeError('windowObj must be an object or null');
      }
      if (
        opts.gregMode !== undefined &&
        opts.gregMode !== null &&
        typeof opts.gregMode !== 'boolean'
      ) {
        throw new TypeError('gregMode must be a boolean');
      }
      if (
        opts.gregClickThreshold !== undefined &&
        opts.gregClickThreshold !== null &&
        (!Number.isInteger(opts.gregClickThreshold) || opts.gregClickThreshold < 1)
      ) {
        throw new TypeError('gregClickThreshold must be a positive integer');
      }
      if (
        opts.gregImageUrl !== undefined &&
        opts.gregImageUrl !== null &&
        typeof opts.gregImageUrl !== 'string'
      ) {
        throw new TypeError('gregImageUrl must be a string or null');
      }

      this._document = doc;
      this._window = opts.windowObj && typeof opts.windowObj === 'object' ? opts.windowObj : null;
      this._parent =
        opts.parent ||
        opts.mountEl ||
        (doc.body && typeof doc.body.appendChild === 'function' ? doc.body : null) ||
        (typeof doc.documentElement !== 'undefined' ? doc.documentElement : null);
      this._splashImageUrl =
        typeof opts.splashImageUrl === 'string' && opts.splashImageUrl.length > 0
          ? opts.splashImageUrl
          : null;
      this._gregImageUrl =
        typeof opts.gregImageUrl === 'string' && opts.gregImageUrl.trim().length > 0
          ? opts.gregImageUrl.trim()
          : DEFAULT_GREG_IMAGE_URL;
      this._gregClickThreshold =
        typeof opts.gregClickThreshold === 'number'
          ? opts.gregClickThreshold
          : DEFAULT_GREG_CLICK_THRESHOLD;
      this._fadeMs =
        typeof opts.fadeMs === 'number' && Number.isFinite(opts.fadeMs) && opts.fadeMs >= 0
          ? opts.fadeMs
          : DEFAULT_FADE_MS;
      this._onDismiss = typeof opts.onDismiss === 'function' ? opts.onDismiss : null;
      this._setTimeout =
        typeof opts.setTimeout === 'function'
          ? opts.setTimeout
          : typeof setTimeout === 'function'
            ? setTimeout
            : (fn) => {
                fn();
                return 0;
              };
      this._clearTimeout =
        typeof opts.clearTimeout === 'function'
          ? opts.clearTimeout
          : typeof clearTimeout === 'function'
            ? clearTimeout
            : () => {};
      this._nowFn = typeof opts.nowFn === 'function' ? opts.nowFn : () => Date.now();
      this._reducedMotionOverride =
        typeof opts.reducedMotion === 'boolean' ? opts.reducedMotion : null;
      this._matchMedia = typeof opts.matchMedia === 'function' ? opts.matchMedia : null;

      const searchSource = resolveSearchSource(opts, doc);
      this._hudHidden = isHudHiddenQuery(searchSource);
      this._instantHide =
        typeof opts.instantHide === 'boolean' ? opts.instantHide : this._hudHidden;
      this._gregMode =
        typeof opts.gregMode === 'boolean' ? opts.gregMode : isGregQuery(searchSource);
      this._gregClickCount = 0;
      this._gregKeyBuffer = '';
      this._onSpinnerClick = null;
      this._onKeyDown = null;
      this._keyTarget = null;

      const autoHiddenByHud =
        this._hudHidden && opts.visible === undefined && opts.instantHide !== false;

      this._destroyed = false;
      this._visible = autoHiddenByHud ? false : opts.visible !== false;
      this._dismissed = autoHiddenByHud;
      this._fading = false;
      this._fadeTimer = null;
      this._pendingDismissReason = null;
      this._dismissedAtMs = autoHiddenByHud ? this._safeNow() : null;
      this._dismissCount = autoHiddenByHud ? 1 : 0;
      this._fadeDurationMs = autoHiddenByHud ? 0 : null;
      this._dismissReason = autoHiddenByHud ? DISMISS_REASONS.HUD_HIDDEN : null;
      this._scriptedInputPreset = sanitizeScriptedInputPreset(opts.scriptedInput);
      this._imageFailed = false;
      this._usingFallback = this._splashImageUrl === null;
      this._progressPct = 0;
      this._streamPct = null;
      this._stage = STAGE_NOT_STARTED;
      this._stageLabel = BOOT_STAGE_LABELS.NOT_STARTED;
      this._statusText = this._stageLabel;
      this._elapsedMs = 0;
      this._badgeText = '';
      this._readyForSplash = false;
      this._lastSnapshot = null;

      this._styleEl = ensureSharedStyle(doc);
      this._root = opts.root && typeof opts.root === 'object' ? opts.root : this._buildDom();
      this._bindElements();

      if (opts.autoMount !== false && this._parent && this._root && !this._root.parentNode) {
        try {
          this._parent.appendChild(this._root);
        } catch {
          // Parent mocks may omit appendChild; tests can still drive the root directly.
        }
      }

      this._applyVisibility(this._visible);
      this._syncChrome();
    }

    _safeNow() {
      try {
        const value = this._nowFn();
        if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
          return value;
        }
      } catch {
        // Faulty injected clocks must never throw out of the overlay.
      }
      return 0;
    }

    _isReducedMotion() {
      if (this._reducedMotionOverride !== null) {
        return this._reducedMotionOverride;
      }
      let mediaFn = this._matchMedia;
      if (!mediaFn && this._window && typeof this._window.matchMedia === 'function') {
        mediaFn = this._window.matchMedia.bind(this._window);
      } else if (!mediaFn && this._document && this._document.defaultView && typeof this._document.defaultView.matchMedia === 'function') {
        mediaFn = this._document.defaultView.matchMedia.bind(this._document.defaultView);
      } else if (!mediaFn && typeof window !== 'undefined' && window && typeof window.matchMedia === 'function') {
        mediaFn = window.matchMedia.bind(window);
      } else if (!mediaFn && root && typeof root.matchMedia === 'function') {
        mediaFn = root.matchMedia.bind(root);
      }
      if (!mediaFn) {
        return false;
      }
      try {
        const result = mediaFn(REDUCED_MOTION_QUERY);
        return Boolean(result && result.matches);
      } catch {
        return false;
      }
    }

    _buildDom() {
      const doc = this._document;
      const root = createEl(doc, 'div', ROOT_CLASS, {
        role: 'status',
        'aria-live': 'polite',
        'aria-atomic': 'true',
        'aria-busy': 'true',
      });
      const initialFadeMs = this._isReducedMotion() || this._instantHide ? 0 : this._fadeMs;
      root.style.setProperty('--pg6-splash-fade', `${initialFadeMs}ms`);

      const media = createEl(doc, 'div', `${ROOT_CLASS}__media`);
      const img = createEl(doc, 'img', `${ROOT_CLASS}__img`, {
        alt: 'Local splash',
        decoding: 'async',
      });
      const fallback = createEl(doc, 'div', `${ROOT_CLASS}__fallback`);
      fallback.innerHTML = FALLBACK_SVG;
      media.appendChild(img);
      media.appendChild(fallback);

      const panel = createEl(doc, 'div', `${ROOT_CLASS}__panel`);
      const scripted = createEl(doc, 'div', `${ROOT_CLASS}__scripted`, {
        'aria-hidden': 'true',
        'data-active': 'false',
      });
      scripted.hidden = true;
      if (scripted.style) scripted.style.display = 'none';

      const stageRow = createEl(doc, 'div', `${ROOT_CLASS}__stage-row`);
      const spinner = createEl(doc, 'span', `${ROOT_CLASS}__spinner`, {
        'aria-hidden': 'true',
        'data-greg-mode': 'false',
      });
      const spinnerRing = createEl(doc, 'span', `${ROOT_CLASS}__spinner-ring`, {
        'aria-hidden': 'true',
      });
      const gregImg = createEl(doc, 'img', `${ROOT_CLASS}__greg`, {
        src: this._gregImageUrl,
        alt: 'Greg',
        decoding: 'async',
      });
      gregImg.hidden = true;
      if (gregImg.style) gregImg.style.display = 'none';
      spinner.appendChild(spinnerRing);
      spinner.appendChild(gregImg);

      const stage = createEl(doc, 'div', `${ROOT_CLASS}__stage`);
      stageRow.appendChild(spinner);
      stageRow.appendChild(stage);
      const bar = createEl(doc, 'div', `${ROOT_CLASS}__bar`, {
        role: 'progressbar',
        'aria-valuemin': '0',
        'aria-valuemax': '100',
        'aria-valuenow': '0',
      });
      const barFill = createEl(doc, 'div', `${ROOT_CLASS}__bar-fill`);
      bar.appendChild(barFill);

      const stream = createEl(doc, 'div', `${ROOT_CLASS}__stream`);
      stream.hidden = true;
      const streamFill = createEl(doc, 'div', `${ROOT_CLASS}__stream-fill`);
      stream.appendChild(streamFill);

      const meta = createEl(doc, 'div', `${ROOT_CLASS}__meta`);
      const pct = createEl(doc, 'span', `${ROOT_CLASS}__pct`);
      const elapsed = createEl(doc, 'span', `${ROOT_CLASS}__elapsed`);
      meta.appendChild(pct);
      meta.appendChild(elapsed);

      const badge = createEl(doc, 'div', `${ROOT_CLASS}__badge`);
      badge.hidden = true;

      panel.appendChild(scripted);
      panel.appendChild(stageRow);
      panel.appendChild(bar);
      panel.appendChild(stream);
      panel.appendChild(meta);
      panel.appendChild(badge);

      root.appendChild(media);
      root.appendChild(panel);

      this._els = {
        root,
        media,
        img,
        fallback,
        scripted,
        stageRow,
        spinner,
        spinnerRing,
        gregImg,
        stage,
        bar,
        barFill,
        stream,
        streamFill,
        pct,
        elapsed,
        badge,
      };

      return root;
    }

    _resolveKeyTarget() {
      if (this._window && typeof this._window.addEventListener === 'function') {
        return this._window;
      }
      if (this._document && typeof this._document.addEventListener === 'function') {
        return this._document;
      }
      try {
        if (
          this._document &&
          this._document.defaultView &&
          typeof this._document.defaultView.addEventListener === 'function'
        ) {
          return this._document.defaultView;
        }
      } catch {
        // ignore
      }
      try {
        if (typeof window !== 'undefined' && window && typeof window.addEventListener === 'function') {
          return window;
        }
      } catch {
        // ignore
      }
      return null;
    }

    _bindElements() {
      const els = this._els || {};
      if (!els.root) {
        this._els = {
          root: this._root,
          img: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__img`) : null,
          fallback: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__fallback`) : null,
          scripted: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__scripted`) : null,
          stageRow: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__stage-row`) : null,
          spinner: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__spinner`) : null,
          spinnerRing: this._root.querySelector
            ? this._root.querySelector(`.${ROOT_CLASS}__spinner-ring`)
            : null,
          gregImg: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__greg`) : null,
          stage: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__stage`) : null,
          bar: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__bar`) : null,
          barFill: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__bar-fill`) : null,
          stream: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__stream`) : null,
          streamFill: this._root.querySelector
            ? this._root.querySelector(`.${ROOT_CLASS}__stream-fill`)
            : null,
          pct: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__pct`) : null,
          elapsed: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__elapsed`) : null,
          badge: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__badge`) : null,
        };
      }

      const img = this._els.img;
      if (img) {
        this._onImageError = () => {
          this._imageFailed = true;
          this._usingFallback = true;
          this._syncImage();
        };
        if (typeof img.addEventListener === 'function') {
          img.addEventListener('error', this._onImageError);
        } else {
          img.onerror = this._onImageError;
        }
      }

      const spinner = this._els.spinner;
      if (spinner && typeof spinner.addEventListener === 'function') {
        this._onSpinnerClick = () => {
          if (this._destroyed || !this._visible || this._dismissed) {
            return;
          }
          this._gregClickCount += 1;
          if (this._gregClickCount >= this._gregClickThreshold) {
            this._gregClickCount = 0;
            this.toggleGregMode();
          }
        };
        try {
          spinner.addEventListener('click', this._onSpinnerClick);
        } catch {
          this._onSpinnerClick = null;
        }
      }

      const keyTarget = this._resolveKeyTarget();
      if (keyTarget) {
        this._onKeyDown = (event) => {
          if (this._destroyed || !this._visible || this._dismissed) {
            return;
          }
          if (!event || typeof event.key !== 'string') {
            return;
          }
          if (event.ctrlKey || event.metaKey || event.altKey) {
            return;
          }
          if (isEditableEventTarget(event.target)) {
            return;
          }
          const ch = event.key.toLowerCase();
          if (ch.length !== 1) {
            return;
          }
          this._gregKeyBuffer = (this._gregKeyBuffer + ch).slice(-GREG_SECRET_WORD.length);
          if (this._gregKeyBuffer === GREG_SECRET_WORD) {
            this._gregKeyBuffer = '';
            this.toggleGregMode();
          }
        };
        try {
          keyTarget.addEventListener('keydown', this._onKeyDown);
          this._keyTarget = keyTarget;
        } catch {
          this._onKeyDown = null;
          this._keyTarget = null;
        }
      }

      this._syncImage();
    }

    _syncImage() {
      const img = this._els && this._els.img;
      const fallback = this._els && this._els.fallback;
      const useImage = Boolean(this._splashImageUrl) && !this._imageFailed;

      if (img) {
        try {
          if (useImage) {
            if (img.getAttribute && img.getAttribute('src') !== this._splashImageUrl) {
              img.setAttribute('src', this._splashImageUrl);
            } else if (img.src !== this._splashImageUrl) {
              img.src = this._splashImageUrl;
            }
            img.hidden = false;
            if (img.style) img.style.display = '';
          } else {
            img.removeAttribute && img.removeAttribute('src');
            img.hidden = true;
            if (img.style) img.style.display = 'none';
          }
        } catch {
          // Image attribute updates must never throw out of the overlay.
          this._imageFailed = true;
          this._usingFallback = true;
        }
      }

      if (fallback) {
        fallback.hidden = useImage;
        if (fallback.style) {
          fallback.style.display = useImage ? 'none' : '';
        }
      }

      this._usingFallback = !useImage;
    }

    _removeFadingClass(el) {
      if (!el) return;
      if (el.classList && typeof el.classList.remove === 'function') {
        el.classList.remove('pg6-splash--fading');
      } else if (typeof el.className === 'string') {
        el.className = el.className
          .split(/\s+/)
          .filter((c) => c && c !== 'pg6-splash--fading')
          .join(' ');
      }
    }

    _applyVisibility(visible) {
      const root = this._root;
      if (!root) return;
      try {
        if (visible) {
          root.hidden = false;
          if (root.style) root.style.display = '';
          this._removeFadingClass(root);
        } else {
          root.hidden = true;
          if (root.style) root.style.display = 'none';
          this._removeFadingClass(root);
        }
      } catch {
        // Visibility updates must never throw.
      }
      this._visible = Boolean(visible);
    }

    _syncSpinner() {
      const els = this._els || {};
      const spinner = els.spinner;
      const spinnerRing = els.spinnerRing;
      const gregImg = els.gregImg;
      const reducedMotion = this._isReducedMotion();
      const activeGreg = Boolean(this._gregMode);
      const hideSpinner = Boolean(this._dismissed);

      if (this._root) {
        try {
          this._root.setAttribute('data-greg-mode', activeGreg ? 'true' : 'false');
          this._root.setAttribute('data-reduced-motion', reducedMotion ? 'true' : 'false');
        } catch {
          // ignore
        }
      }

      if (spinner) {
        try {
          spinner.hidden = hideSpinner;
          if (spinner.style) spinner.style.display = hideSpinner ? 'none' : '';
          spinner.setAttribute('data-greg-mode', activeGreg ? 'true' : 'false');
          spinner.setAttribute('data-reduced-motion', reducedMotion ? 'true' : 'false');
          const gregClass = `${ROOT_CLASS}__spinner--greg`;
          if (spinner.classList && typeof spinner.classList.add === 'function' && typeof spinner.classList.remove === 'function') {
            if (activeGreg) {
              spinner.classList.add(gregClass);
            } else {
              spinner.classList.remove(gregClass);
            }
          } else if (typeof spinner.className === 'string') {
            const tokens = spinner.className.split(/\s+/).filter((c) => c && c !== gregClass);
            if (activeGreg) tokens.push(gregClass);
            spinner.className = tokens.join(' ');
          }
        } catch {
          // Spinner updates must never throw.
        }
      }

      if (spinnerRing) {
        try {
          spinnerRing.hidden = activeGreg;
          if (spinnerRing.style) {
            spinnerRing.style.display = activeGreg ? 'none' : '';
            spinnerRing.style.animation = reducedMotion ? 'none' : '';
          }
        } catch {
          // ignore
        }
      }

      if (gregImg) {
        try {
          if (gregImg.getAttribute && gregImg.getAttribute('src') !== this._gregImageUrl) {
            gregImg.setAttribute('src', this._gregImageUrl);
          } else if (gregImg.src !== this._gregImageUrl) {
            gregImg.src = this._gregImageUrl;
          }
          if (gregImg.getAttribute && gregImg.getAttribute('alt') !== 'Greg') {
            gregImg.setAttribute('alt', 'Greg');
          }
          gregImg.hidden = !activeGreg;
          if (gregImg.style) {
            gregImg.style.display = activeGreg ? '' : 'none';
            gregImg.style.animation = reducedMotion ? 'none' : '';
          }
        } catch {
          // ignore
        }
      }
    }

    _syncChrome() {
      const els = this._els || {};
      const pct = Math.round(clampPct(this._progressPct));
      const stageText = this._statusText || this._stageLabel || labelForStage(this._stage);

      if (els.stage) els.stage.textContent = stageText;
      if (els.pct) els.pct.textContent = `${pct}%`;
      if (els.elapsed) els.elapsed.textContent = formatElapsed(this._elapsedMs);

      if (els.barFill && els.barFill.style) {
        els.barFill.style.width = `${pct}%`;
      }
      if (els.bar) {
        try {
          els.bar.setAttribute('aria-valuenow', String(pct));
          els.bar.setAttribute('aria-label', `Boot progress ${pct}%`);
        } catch {
          // ignore
        }
      }

      const streamPct = this._streamPct;
      if (els.stream) {
        if (streamPct === null || streamPct === undefined) {
          els.stream.hidden = true;
        } else {
          els.stream.hidden = false;
          if (els.streamFill && els.streamFill.style) {
            els.streamFill.style.width = `${Math.round(clampPct(streamPct))}%`;
          }
        }
      }

      if (els.badge) {
        if (this._badgeText) {
          els.badge.hidden = false;
          els.badge.textContent = this._badgeText;
        } else {
          els.badge.hidden = true;
          els.badge.textContent = '';
        }
      }

      if (els.scripted) {
        try {
          if (this._scriptedInputPreset) {
            els.scripted.hidden = false;
            if (els.scripted.style) els.scripted.style.display = '';
            els.scripted.textContent = `Scripted input: ${this._scriptedInputPreset}`;
            els.scripted.setAttribute('aria-hidden', 'false');
            els.scripted.setAttribute('data-active', 'true');
            els.scripted.setAttribute('data-preset', this._scriptedInputPreset);
          } else {
            els.scripted.hidden = true;
            if (els.scripted.style) els.scripted.style.display = 'none';
            els.scripted.textContent = '';
            els.scripted.setAttribute('aria-hidden', 'true');
            els.scripted.setAttribute('data-active', 'false');
            if (typeof els.scripted.removeAttribute === 'function') {
              els.scripted.removeAttribute('data-preset');
            }
          }
        } catch {
          // Scripted-input pill updates must never throw.
        }
      }

      this._syncSpinner();

      if (this._root) {
        try {
          this._root.setAttribute('aria-busy', this._dismissed || this._readyForSplash ? 'false' : 'true');
          this._root.setAttribute('aria-label', stageText);
        } catch {
          // ignore
        }
      }
    }

    _clearFadeTimer() {
      if (this._fadeTimer !== null && this._fadeTimer !== undefined) {
        try {
          this._clearTimeout(this._fadeTimer);
        } catch {
          // ignore
        }
        this._fadeTimer = null;
      }
      this._fading = false;
    }

    _fireDismiss() {
      if (typeof this._onDismiss === 'function') {
        try {
          this._onDismiss();
        } catch {
          // Callback errors must not break dismiss.
        }
      }
    }

    /**
     * Apply a BootTracer-compatible snapshot and refresh the overlay chrome.
     *
     * @param {object|null|undefined} snapshot
     * @returns {object} Current controller state.
     */
    updateBootState(snapshot) {
      if (this._destroyed) {
        return this.getState();
      }
      const snap = snapshot && typeof snapshot === 'object' ? snapshot : {};
      this._lastSnapshot = snap;

      this._stage =
        typeof snap.stage === 'string' && snap.stage.length > 0
          ? snap.stage.trim().toUpperCase()
          : STAGE_NOT_STARTED;
      this._stageLabel = labelForStage(this._stage);
      this._progressPct = computeBootProgress(snap);
      this._elapsedMs =
        typeof snap.elapsedMs === 'number' && Number.isFinite(snap.elapsedMs) ? snap.elapsedMs : 0;
      this._readyForSplash = Boolean(snap.readyForSplash) || this._stage === 'FIRST_PRESENT';
      this._streamPct = streamingPct(snap.streamingProgress);

      const traps = snap.traps !== undefined ? snap.traps : snap.recentTraps;
      this._badgeText = formatTrapBadge(traps, snap.unresolvedImports);

      if (typeof snap.statusText === 'string' && snap.statusText.trim().length > 0) {
        this._statusText = snap.statusText.trim();
      } else {
        this._statusText = this._stageLabel;
      }

      this._syncChrome();

      if (this._readyForSplash && !this._dismissed) {
        // Caller may still want an explicit onFirstPresent(); do not auto-dismiss here.
      }

      return this.getState();
    }

    /**
     * Dynamically set or clear the local splash image URL (e.g. after `/api/status`
     * discovers `/game/xbs_splash.png`). Resets prior image-error state when URL changes.
     *
     * @param {string|null|undefined} url
     * @returns {object} Current controller state.
     */
    setSplashImageUrl(url) {
      if (this._destroyed) {
        return this.getState();
      }
      const nextUrl = typeof url === 'string' && url.trim().length > 0 ? url.trim() : null;
      if (nextUrl !== this._splashImageUrl) {
        this._splashImageUrl = nextUrl;
        this._imageFailed = false;
        this._syncImage();
      }
      return this.getState();
    }

    /** Alias used by BootOrchestrator (W44). */
    setSplashImage(url) {
      return this.setSplashImageUrl(url);
    }

    /**
     * Set or clear the optional scripted-input indicator pill. Strictly sanitizes
     * the preset identifier against `/^[a-z0-9][a-z0-9_-]{0,31}$/i` (invalid values
     * and `null`/`undefined` clear and hide the pill).
     *
     * @param {string|null|undefined} presetOrNull
     * @returns {object} Current controller state.
     */
    setScriptedInput(presetOrNull) {
      if (this._destroyed) {
        return this.getState();
      }
      this._scriptedInputPreset = sanitizeScriptedInputPreset(presetOrNull);
      this._syncChrome();
      return this.getState();
    }

    /**
     * Enable or disable the Spinning Greg loading spinner easter egg.
     *
     * @param {boolean} enabled
     * @returns {object} Current controller state.
     */
    setGregMode(enabled) {
      if (this._destroyed) {
        return this.getState();
      }
      this._gregMode = Boolean(enabled);
      this._gregClickCount = 0;
      this._gregKeyBuffer = '';
      this._syncSpinner();
      return this.getState();
    }

    /**
     * Toggle the Spinning Greg loading spinner easter egg.
     *
     * @returns {object} Current controller state.
     */
    toggleGregMode() {
      if (this._destroyed) {
        return this.getState();
      }
      return this.setGregMode(!this._gregMode);
    }

    _completeDismiss(effectiveFadeMs, startedAtMs, reason) {
      const completedAtMs = this._safeNow();
      let fadeDurationMs = 0;
      if (effectiveFadeMs > 0) {
        const measured = completedAtMs - startedAtMs;
        fadeDurationMs = Number.isFinite(measured) && measured > 0 ? measured : effectiveFadeMs;
      }
      this._dismissedAtMs = completedAtMs;
      this._dismissCount += 1;
      this._fadeDurationMs = fadeDurationMs;
      this._dismissReason =
        sanitizeDismissReason(reason) ||
        this._pendingDismissReason ||
        DISMISS_REASONS.MANUAL;
      this._pendingDismissReason = null;
      this._fading = false;
      this._applyVisibility(false);
      this._syncSpinner();
      this._fireDismiss();
    }

    /**
     * Signal that the first WebGPU / Bink frame has been presented and dismiss the overlay.
     * Supports immediate dismissal via `{ immediate: true }`, `{ instant: true }`,
     * `{ instantHide: true }`, `{ fadeMs: 0 }`, constructor `instantHide: true`,
     * `?hud=0`, or `prefers-reduced-motion`.
     *
     * @param {{ fadeMs?: number, reducedMotion?: boolean, immediate?: boolean, instant?: boolean, instantHide?: boolean, reason?: string }} [options]
     * @returns {object}
     */
    onFirstPresent(options) {
      if (this._destroyed) {
        return this.getState();
      }
      this._readyForSplash = true;
      this._stage = 'FIRST_PRESENT';
      this._stageLabel = labelForStage(this._stage);
      this._progressPct = 100;
      this._statusText = this._stageLabel;
      this._syncChrome();

      const opts = options && typeof options === 'object' ? { ...options } : {};
      if (opts.reason === undefined) {
        opts.reason = DISMISS_REASONS.FIRST_PRESENT;
      }
      return this.dismiss(opts);
    }

    /**
     * Immediately dismiss the overlay with zero transition delay (no CSS fade timer).
     * Cancels any in-flight fade animation and hides synchronously so automated video
     * captures and `FIRST_PRESENT` Bink2 frames are never occluded. Idempotent once dismissed.
     *
     * @param {string|{ reason?: string }} [options]
     * @returns {object}
     */
    dismissImmediate(options) {
      if (this._destroyed) {
        return this.getState();
      }
      let opts;
      if (typeof options === 'string') {
        opts = { reason: options };
      } else if (options && typeof options === 'object') {
        opts = { ...options };
      } else {
        opts = {};
      }
      if (opts.reason === undefined) {
        opts.reason = DISMISS_REASONS.IMMEDIATE;
      }
      opts.immediate = true;
      opts.fadeMs = 0;
      return this.dismiss(opts);
    }

    /**
     * Smoothly cross-fade the overlay out, then hide it. Idempotent.
     * Honors `immediate` / `instant` / `instantHide`, `?hud=0`, and
     * `prefers-reduced-motion` (or `reducedMotion: true`) by skipping transition
     * delay and hiding synchronously. Calling with immediate mode while a fade
     * is in flight cancels the pending timer and completes dismissal immediately.
     *
     * @param {{ fadeMs?: number, reducedMotion?: boolean, immediate?: boolean, instant?: boolean, instantHide?: boolean, reason?: string }} [options]
     * @returns {object}
     */
    dismiss(options = {}) {
      if (this._destroyed) {
        return this.getState();
      }

      const opts = options && typeof options === 'object' ? options : {};
      const explicitImmediate =
        typeof opts.immediate === 'boolean'
          ? opts.immediate
          : typeof opts.instant === 'boolean'
            ? opts.instant
            : typeof opts.instantHide === 'boolean'
              ? opts.instantHide
              : null;
      const requestedFadeMs =
        typeof opts.fadeMs === 'number' && Number.isFinite(opts.fadeMs)
          ? Math.max(0, opts.fadeMs)
          : this._fadeMs;
      const reducedMotion =
        typeof opts.reducedMotion === 'boolean'
          ? opts.reducedMotion
          : this._isReducedMotion();
      const useInstant = explicitImmediate !== null ? explicitImmediate : this._instantHide;
      const effectiveFadeMs = useInstant || reducedMotion ? 0 : requestedFadeMs;

      const explicitReason = sanitizeDismissReason(opts.reason);
      const defaultReason =
        explicitImmediate === true
          ? DISMISS_REASONS.IMMEDIATE
          : this._hudHidden && useInstant
            ? DISMISS_REASONS.HUD_HIDDEN
            : useInstant
              ? DISMISS_REASONS.INSTANT_HIDE
              : reducedMotion
                ? DISMISS_REASONS.REDUCED_MOTION
                : DISMISS_REASONS.MANUAL;
      const resolvedReason = explicitReason || defaultReason;

      if (this._dismissed) {
        if (this._fading && (explicitImmediate === true || effectiveFadeMs === 0)) {
          const upgradedReason =
            explicitReason ||
            (explicitImmediate === true
              ? DISMISS_REASONS.IMMEDIATE
              : this._pendingDismissReason || resolvedReason);
          this._clearFadeTimer();
          if (this._root) {
            try {
              this._root.style && this._root.style.setProperty('--pg6-splash-fade', '0ms');
              this._root.setAttribute('aria-busy', 'false');
            } catch {
              // ignore
            }
          }
          this._completeDismiss(0, this._safeNow(), upgradedReason);
        }
        return this.getState();
      }

      const startedAtMs = this._safeNow();

      this._dismissed = true;
      this._pendingDismissReason = resolvedReason;
      this._clearFadeTimer();
      this._syncSpinner();

      if (this._root) {
        try {
          this._root.style && this._root.style.setProperty('--pg6-splash-fade', `${effectiveFadeMs}ms`);
          if (effectiveFadeMs > 0) {
            if (this._root.classList && typeof this._root.classList.add === 'function') {
              this._root.classList.add('pg6-splash--fading');
            } else if (typeof this._root.className === 'string') {
              this._root.className = `${this._root.className} pg6-splash--fading`.trim();
            }
          }
          this._root.setAttribute('aria-busy', 'false');
        } catch {
          // ignore
        }
      }

      if (effectiveFadeMs <= 0) {
        this._completeDismiss(0, startedAtMs, resolvedReason);
        return this.getState();
      }

      this._fading = true;
      this._fadeTimer = this._setTimeout(() => {
        this._fadeTimer = null;
        if (this._destroyed) {
          return;
        }
        this._completeDismiss(effectiveFadeMs, startedAtMs, resolvedReason);
      }, effectiveFadeMs);

      return this.getState();
    }

    /**
     * Show the overlay (clears a prior hide/dismiss; idempotent when already visible
     * and active). Pass `{ reset: true }` to reset boot stage/progress on restart.
     *
     * @param {{ reset?: boolean }} [options]
     * @returns {object}
     */
    show(options) {
      if (this._destroyed) return this.getState();
      const reset = Boolean(options && typeof options === 'object' && options.reset === true);
      if (this._visible && !this._dismissed && !this._fading && !reset) {
        return this.getState();
      }
      this._clearFadeTimer();
      this._dismissed = false;
      this._fading = false;
      this._pendingDismissReason = null;
      if (reset) {
        this._stage = STAGE_NOT_STARTED;
        this._stageLabel = BOOT_STAGE_LABELS.NOT_STARTED;
        this._statusText = this._stageLabel;
        this._progressPct = 0;
        this._streamPct = null;
        this._elapsedMs = 0;
        this._badgeText = '';
        this._readyForSplash = false;
      }
      if (this._root && this._root.style && typeof this._root.style.setProperty === 'function') {
        try {
          const fadeMs = this._isReducedMotion() || this._instantHide ? 0 : this._fadeMs;
          this._root.style.setProperty('--pg6-splash-fade', `${fadeMs}ms`);
        } catch {
          // ignore
        }
      }
      this._applyVisibility(true);
      this._syncChrome();
      return this.getState();
    }

    /** Immediately hide the overlay without firing onDismiss. Idempotent when already hidden. */
    hide() {
      if (this._destroyed) return this.getState();
      if (!this._visible && !this._fading) {
        return this.getState();
      }
      const wasFading = this._fading;
      this._clearFadeTimer();
      this._fading = false;
      this._pendingDismissReason = null;
      if (wasFading) {
        this._dismissed = false;
      }
      this._applyVisibility(false);
      return this.getState();
    }

    /**
     * Return a sanitized, JSON-safe telemetry snapshot of overlay visibility,
     * dismissal timing/counts/reason, reduced-motion/instant-hide state, and
     * scripted-input preset.
     *
     * @returns {object}
     */
    getStats() {
      return {
        visible: Boolean(this._visible),
        dismissed: Boolean(this._dismissed),
        fading: Boolean(this._fading),
        destroyed: Boolean(this._destroyed),
        dismissedAtMs:
          typeof this._dismissedAtMs === 'number' && Number.isFinite(this._dismissedAtMs) && this._dismissedAtMs >= 0
            ? this._dismissedAtMs
            : null,
        dismissCount:
          typeof this._dismissCount === 'number' && Number.isFinite(this._dismissCount) && this._dismissCount >= 0
            ? Math.floor(this._dismissCount)
            : 0,
        fadeDurationMs:
          typeof this._fadeDurationMs === 'number' &&
          Number.isFinite(this._fadeDurationMs) &&
          this._fadeDurationMs >= 0
            ? this._fadeDurationMs
            : null,
        dismissReason: sanitizeDismissReason(this._dismissReason),
        fadeMs: this._fadeMs,
        instantHide: Boolean(this._instantHide),
        hudHidden: Boolean(this._hudHidden),
        reducedMotion: this._isReducedMotion(),
        stage: this._stage,
        progressPct: Math.round(clampPct(this._progressPct)),
        readyForSplash: Boolean(this._readyForSplash),
        scriptedInputPreset: this._scriptedInputPreset,
        gregMode: Boolean(this._gregMode),
      };
    }

    /** @returns {object} */
    getState() {
      return {
        visible: this._visible,
        dismissed: this._dismissed,
        fading: this._fading,
        destroyed: this._destroyed,
        progressPct: Math.round(clampPct(this._progressPct)),
        streamPct: this._streamPct === null || this._streamPct === undefined
          ? null
          : Math.round(clampPct(this._streamPct)),
        stage: this._stage,
        stageLabel: this._stageLabel,
        statusText: this._statusText,
        elapsedMs: this._elapsedMs,
        badgeText: this._badgeText,
        readyForSplash: this._readyForSplash,
        splashImageUrl: this._splashImageUrl,
        usingFallback: this._usingFallback,
        imageFailed: this._imageFailed,
        fadeMs: this._fadeMs,
        instantHide: this._instantHide,
        hudHidden: this._hudHidden,
        dismissedAtMs: this._dismissedAtMs,
        dismissCount: this._dismissCount,
        fadeDurationMs: this._fadeDurationMs,
        dismissReason: this._dismissReason,
        reducedMotion: this._isReducedMotion(),
        scriptedInputPreset: this._scriptedInputPreset,
        gregMode: Boolean(this._gregMode),
        root: this._root,
      };
    }

    /** Tear down DOM listeners / nodes. Idempotent. */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;
      this._clearFadeTimer();
      this._pendingDismissReason = null;
      this._gregClickCount = 0;
      this._gregKeyBuffer = '';

      const img = this._els && this._els.img;
      if (img && this._onImageError) {
        try {
          if (typeof img.removeEventListener === 'function') {
            img.removeEventListener('error', this._onImageError);
          } else {
            img.onerror = null;
          }
        } catch {
          // ignore
        }
      }

      const spinner = this._els && this._els.spinner;
      if (spinner && this._onSpinnerClick) {
        try {
          if (typeof spinner.removeEventListener === 'function') {
            spinner.removeEventListener('click', this._onSpinnerClick);
          }
        } catch {
          // ignore
        }
      }
      this._onSpinnerClick = null;

      if (this._keyTarget && this._onKeyDown) {
        try {
          if (typeof this._keyTarget.removeEventListener === 'function') {
            this._keyTarget.removeEventListener('keydown', this._onKeyDown);
          }
        } catch {
          // ignore
        }
      }
      this._onKeyDown = null;
      this._keyTarget = null;

      if (this._root && this._root.parentNode && typeof this._root.parentNode.removeChild === 'function') {
        try {
          this._root.parentNode.removeChild(this._root);
        } catch {
          // ignore
        }
      }

      this._visible = false;
      this._els = null;
      this._root = null;
      this._onDismiss = null;
    }
  }

  const exportedApi = {
    BOOT_STAGE_ORDER,
    BOOT_STAGE_LABELS,
    DISMISS_REASONS,
    computeBootProgress,
    sanitizeScriptedInputPreset,
    sanitizeDismissReason,
    isHudHiddenQuery,
    isGregQuery,
    SplashOverlayController,
    DEFAULT_FADE_MS,
    DEFAULT_GREG_IMAGE_URL,
    DEFAULT_GREG_CLICK_THRESHOLD,
    GREG_SECRET_WORD,
    REDUCED_MOTION_QUERY,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6SplashOverlay = exportedApi;
  } else if (root) {
    root.PlayGta6SplashOverlay = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
