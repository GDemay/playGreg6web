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
  const STAGE_NOT_STARTED = 'NOT_STARTED';
  const STYLE_ATTR = 'data-pg6-splash-style';
  const ROOT_CLASS = 'pg6-splash';

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
    '__stream[hidden],.pg6-splash__badge[hidden]{display:none!important;}' +
    '.' +
    ROOT_CLASS +
    '__meta{display:flex;justify-content:space-between;gap:1rem;font-size:0.78rem;opacity:0.8;}' +
    '.' +
    ROOT_CLASS +
    '__badge{margin-top:0.15rem;padding:0.45rem 0.65rem;border-radius:0.4rem;' +
    'background:rgba(180,60,60,0.22);border:1px solid rgba(255,140,140,0.35);' +
    'color:#ffc9c9;font-size:0.78rem;line-height:1.35;}';

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
     * @param {Element|object|null} [options.parent]
     * @param {Element|object|null} [options.root]
     * @param {string|null} [options.splashImageUrl]
     * @param {number} [options.fadeMs]
     * @param {(() => void)|null} [options.onDismiss]
     * @param {((fn: Function, ms: number) => any)} [options.setTimeout]
     * @param {((id: any) => void)} [options.clearTimeout]
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

      this._document = doc;
      this._parent =
        opts.parent ||
        opts.mountEl ||
        (doc.body && typeof doc.body.appendChild === 'function' ? doc.body : null) ||
        (typeof doc.documentElement !== 'undefined' ? doc.documentElement : null);
      this._splashImageUrl =
        typeof opts.splashImageUrl === 'string' && opts.splashImageUrl.length > 0
          ? opts.splashImageUrl
          : null;
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

      this._destroyed = false;
      this._visible = opts.visible !== false;
      this._dismissed = false;
      this._fading = false;
      this._fadeTimer = null;
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

    _buildDom() {
      const doc = this._document;
      const root = createEl(doc, 'div', ROOT_CLASS, {
        role: 'status',
        'aria-live': 'polite',
        'aria-atomic': 'true',
        'aria-busy': 'true',
      });
      root.style.setProperty('--pg6-splash-fade', `${this._fadeMs}ms`);

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
      const stage = createEl(doc, 'div', `${ROOT_CLASS}__stage`);
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

      panel.appendChild(stage);
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

    _bindElements() {
      const els = this._els || {};
      if (!els.root) {
        this._els = {
          root: this._root,
          img: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__img`) : null,
          fallback: this._root.querySelector ? this._root.querySelector(`.${ROOT_CLASS}__fallback`) : null,
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

    _applyVisibility(visible) {
      const root = this._root;
      if (!root) return;
      try {
        if (visible) {
          root.hidden = false;
          if (root.style) root.style.display = '';
          if (root.classList && typeof root.classList.remove === 'function') {
            root.classList.remove('pg6-splash--fading');
          } else if (typeof root.className === 'string') {
            root.className = root.className
              .split(/\s+/)
              .filter((c) => c && c !== 'pg6-splash--fading')
              .join(' ');
          }
        } else {
          root.hidden = true;
          if (root.style) root.style.display = 'none';
        }
      } catch {
        // Visibility updates must never throw.
      }
      this._visible = Boolean(visible);
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
     * Signal that the first WebGPU / Bink frame has been presented and start the dismiss fade.
     *
     * @param {{ fadeMs?: number }} [options]
     * @returns {object}
     */
    onFirstPresent(options) {
      this._readyForSplash = true;
      this._stage = 'FIRST_PRESENT';
      this._stageLabel = labelForStage(this._stage);
      this._progressPct = 100;
      this._statusText = this._stageLabel;
      this._syncChrome();
      return this.dismiss(options);
    }

    /**
     * Smoothly cross-fade the overlay out, then hide it. Idempotent.
     *
     * @param {{ fadeMs?: number }} [options]
     * @returns {object}
     */
    dismiss(options = {}) {
      if (this._destroyed) {
        return this.getState();
      }
      if (this._dismissed) {
        return this.getState();
      }

      const fadeMs =
        options && typeof options === 'object' && typeof options.fadeMs === 'number' && Number.isFinite(options.fadeMs)
          ? Math.max(0, options.fadeMs)
          : this._fadeMs;

      this._dismissed = true;
      this._clearFadeTimer();

      if (this._root) {
        try {
          this._root.style && this._root.style.setProperty('--pg6-splash-fade', `${fadeMs}ms`);
          if (this._root.classList && typeof this._root.classList.add === 'function') {
            this._root.classList.add('pg6-splash--fading');
          } else if (typeof this._root.className === 'string') {
            this._root.className = `${this._root.className} pg6-splash--fading`.trim();
          }
          this._root.setAttribute('aria-busy', 'false');
        } catch {
          // ignore
        }
      }

      if (fadeMs <= 0) {
        this._fading = false;
        this._applyVisibility(false);
        this._fireDismiss();
        return this.getState();
      }

      this._fading = true;
      this._fadeTimer = this._setTimeout(() => {
        this._fadeTimer = null;
        this._fading = false;
        this._applyVisibility(false);
        this._fireDismiss();
      }, fadeMs);

      return this.getState();
    }

    /** Show the overlay (clears a prior hide; does not undo a completed dismiss fade class). */
    show() {
      if (this._destroyed) return this.getState();
      this._clearFadeTimer();
      this._dismissed = false;
      this._fading = false;
      this._applyVisibility(true);
      this._syncChrome();
      return this.getState();
    }

    /** Immediately hide the overlay without firing onDismiss. */
    hide() {
      if (this._destroyed) return this.getState();
      this._clearFadeTimer();
      this._fading = false;
      this._applyVisibility(false);
      return this.getState();
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
    computeBootProgress,
    SplashOverlayController,
    DEFAULT_FADE_MS,
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
