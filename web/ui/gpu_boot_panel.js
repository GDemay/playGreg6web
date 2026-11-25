'use strict';

// Live Frame-1 GPU & Missing-Shader Diagnostics Panel for playgta6web (Track 3/5 G29).
// Consumes BootTracer (W40) snapshots, WgpuCommandProcessor.getMissingShaders() /
// ShaderFallbackRouter (S12/G28) stats, and SplashFrameBridge (G26) frame stats.
// Renders a compact, accessible collapsible diagnostics region with a deduplicated
// missing-shader shopping list and sanitized JSON export (zero host paths, zero raw bytes).
// Zero external dependencies; UMD export for Browser (globalThis.PlayGta6GpuBootPanel)
// and Node.js (module.exports).

(function (root) {
  const DEFAULT_MAX_ROWS = 32;
  const DEFAULT_MAX_TOP_MISSING_SHADERS = 8;
  const MAX_TOP_MISSING_SHADERS_CAP = 32;
  const MAX_GPU_ADAPTER_LENGTH = 96;
  const EXPORT_FORMAT = 'playgta6-missing-shaders-v1';
  const EXPORT_SCHEMA_VERSION = 1;
  const STAGE_NOT_STARTED = 'NOT_STARTED';
  const PLACEHOLDER_DASH = '—';
  const STYLE_ATTR = 'data-pg6-gpu-boot-panel-style';
  const ROOT_CLASS = 'pg6-gpu-panel';
  const DEFAULT_ARIA_LABEL = 'Frame-1 GPU & Missing-Shader Diagnostics';
  const U64_MAX = 0xffffffffffffffffn;

  const ANSI_ESCAPE_RE = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;
  const CONTROL_CHAR_RE = /[\u0000-\u001f\u007f-\u009f]/g;
  const URL_TOKEN_RE = /[a-zA-Z][a-zA-Z0-9+.-]*:\/\/[^\s"'`<>)\]},;]*/g;
  const WIN_DRIVE_PATH_RE = /[a-zA-Z]:[\\/][^\s"'`<>|*?)\]},;]*/g;
  const BACKSLASH_TOKEN_RE = /[^\s"'`<>|*?)\]},;]*\\[^\s"'`<>|*?)\]},;]*/g;
  const RELATIVE_PATH_TOKEN_RE = /(^|[\s"'`(\[={<,:;|])\.\.?\/[^\s"'`<>)\]},;]*/g;
  const POSIX_ROOT_PATH_RE =
    /(^|[\s"'`(\[={<,:;|])\/(?:(?:users|library|applications|volumes|home|root|tmp|var|etc|opt|usr|mnt|media|private|srv|proc|sys|dev|run|cygdrive|windows|programdata)(?:\/[^\s"'`<>)\]},;]*)*|[a-z0-9._-]+(?:\/[a-z0-9._-]+)+)/gi;
  const ROOT_REL_USER_PATH_RE =
    /(^|[\s"'`(\[={<,:;|])(?:users|home|appdata)\/[^\s"'`<>)\]},;]+/gi;
  const HOST_PATH_PREFIX_RE =
    /^(?:[a-zA-Z]:[\\/]|\\\\|[a-zA-Z][a-zA-Z0-9+.-]*:\/\/|\.\.?[\\/]|\/(?:users|library|applications|volumes|home|root|tmp|var|etc|opt|usr|mnt|media|private|srv|proc|sys|dev|run|cygdrive|windows|programdata)(?:\/|$))/i;
  const REDACTED_TOKEN_RE = /\[REDACTED(?:_[A-Z0-9_]+)?\]/gi;

  /** Allowlist of canonical WebGPU swapchain / render-target format tokens. */
  const ALLOWED_SWAPCHAIN_FORMATS = Object.freeze([
    'bgra8unorm',
    'bgra8unorm-srgb',
    'rgba8unorm',
    'rgba8unorm-srgb',
    'rgba16float',
    'rgb10a2unorm',
    'rgb10a2uint',
    'rg11b10ufloat',
    'rgb9e5ufloat',
    'rgba32float',
    'r8unorm',
    'rg8unorm',
    'r16float',
    'rg16float',
    'r32float',
    'rg32float',
  ]);

  const ALLOWED_SWAPCHAIN_FORMAT_SET = new Set(ALLOWED_SWAPCHAIN_FORMATS);

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

  const BOOT_STAGE_SET = new Set(BOOT_STAGE_ORDER);

  /** Canonical WGPK / ShaderFallbackRouter stage codes. */
  const STAGE_NAMES = Object.freeze({
    1: 'vertex',
    2: 'fragment',
    3: 'compute',
  });

  const STAGE_ALIASES = Object.freeze({
    vertex: 'vertex',
    vs: 'vertex',
    vert: 'vertex',
    '1': 'vertex',
    fragment: 'fragment',
    fs: 'fragment',
    frag: 'fragment',
    pixel: 'fragment',
    ps: 'fragment',
    '2': 'fragment',
    compute: 'compute',
    cs: 'compute',
    comp: 'compute',
    '3': 'compute',
  });

  let nextPanelId = 1;

  const SHARED_CSS =
    '.' +
    ROOT_CLASS +
    '{' +
    'box-sizing:border-box;width:min(30rem,100%);border-radius:0.5rem;' +
    'background:rgba(11,18,32,0.92);color:#e8eef7;border:1px solid rgba(143,163,188,0.28);' +
    'font-family:Segoe UI,system-ui,sans-serif;font-size:0.8rem;line-height:1.35;' +
    'overflow:hidden;box-shadow:0 8px 24px rgba(0,0,0,0.35);' +
    '}' +
    '.' +
    ROOT_CLASS +
    '[hidden]{display:none!important;}' +
    '.' +
    ROOT_CLASS +
    '__header{display:flex;align-items:center;justify-content:space-between;gap:0.5rem;' +
    'padding:0.5rem 0.7rem;background:rgba(255,255,255,0.04);' +
    'border-bottom:1px solid rgba(143,163,188,0.18);}' +
    '.' +
    ROOT_CLASS +
    '__heading{display:flex;align-items:center;gap:0.45rem;min-width:0;}' +
    '.' +
    ROOT_CLASS +
    '__title{font-weight:600;letter-spacing:0.02em;white-space:nowrap;}' +
    '.' +
    ROOT_CLASS +
    '__summary-badge{font-size:0.72rem;padding:0.1rem 0.4rem;border-radius:999px;' +
    'background:rgba(109,179,255,0.16);color:#a9d2ff;white-space:nowrap;}' +
    '.' +
    ROOT_CLASS +
    '__actions{display:flex;align-items:center;gap:0.35rem;flex-shrink:0;}' +
    '.' +
    ROOT_CLASS +
    '__btn{appearance:none;border:1px solid rgba(143,163,188,0.32);border-radius:0.3rem;' +
    'background:rgba(255,255,255,0.07);color:#e8eef7;font:inherit;font-size:0.74rem;' +
    'padding:0.2rem 0.5rem;cursor:pointer;}' +
    '.' +
    ROOT_CLASS +
    '__btn:hover{background:rgba(255,255,255,0.14);}' +
    '.' +
    ROOT_CLASS +
    '__body{padding:0.65rem 0.7rem;display:flex;flex-direction:column;gap:0.65rem;}' +
    '.' +
    ROOT_CLASS +
    '__body[hidden],.' +
    ROOT_CLASS +
    '__overflow[hidden]{display:none!important;}' +
    '.' +
    ROOT_CLASS +
    '__section{display:flex;flex-direction:column;gap:0.35rem;}' +
    '.' +
    ROOT_CLASS +
    '__section-title{font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#8fa3bc;}' +
    '.' +
    ROOT_CLASS +
    '__grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:0.35rem;}' +
    '.' +
    ROOT_CLASS +
    '__grid--4{grid-template-columns:repeat(4,minmax(0,1fr));}' +
    '.' +
    ROOT_CLASS +
    '__metric{padding:0.3rem 0.45rem;border-radius:0.35rem;background:rgba(255,255,255,0.04);}' +
    '.' +
    ROOT_CLASS +
    '__metric--wide{grid-column:1 / -1;}' +
    '.' +
    ROOT_CLASS +
    '__metric-label{display:block;font-size:0.68rem;color:#8fa3bc;}' +
    '.' +
    ROOT_CLASS +
    '__metric-value{display:block;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;' +
    'font-size:0.78rem;font-weight:600;color:#e8eef7;}' +
    '.' +
    ROOT_CLASS +
    '__top-missing-shaders{font-size:0.72rem;word-break:break-all;}' +
    '.' +
    ROOT_CLASS +
    '__gpu-adapter{font-size:0.74rem;word-break:break-word;}' +
    '.' +
    ROOT_CLASS +
    '__progress-bar{height:0.35rem;border-radius:999px;background:rgba(255,255,255,0.12);overflow:hidden;}' +
    '.' +
    ROOT_CLASS +
    '__progress-fill{height:100%;width:0%;background:#6db3ff;transition:width 150ms ease;}' +
    '.' +
    ROOT_CLASS +
    '__table{width:100%;border-collapse:collapse;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;' +
    'font-size:0.74rem;}' +
    '.' +
    ROOT_CLASS +
    '__table th,.' +
    ROOT_CLASS +
    '__table td{text-align:left;padding:0.22rem 0.35rem;border-bottom:1px solid rgba(143,163,188,0.14);}' +
    '.' +
    ROOT_CLASS +
    '__table th{color:#8fa3bc;font-weight:600;}' +
    '.' +
    ROOT_CLASS +
    '__cell-hits{text-align:right!important;}' +
    '.' +
    ROOT_CLASS +
    '__empty-cell{color:#8fa3bc;text-align:center!important;padding:0.45rem!important;}' +
    '.' +
    ROOT_CLASS +
    '__overflow{font-size:0.74rem;color:#a9d2ff;text-align:right;padding-top:0.15rem;}';

  function clampPct(value) {
    const n = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(n) || n <= 0) return 0;
    if (n >= 100) return 100;
    return n;
  }

  function nonNegativeInt(value, fallback = 0) {
    if (value === undefined || value === null || value === '') return fallback;
    const n = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(n) || n < 0) return fallback;
    return Math.floor(n);
  }

  function nonNegativeFinite(value) {
    if (value === undefined || value === null || value === '') return null;
    const n = typeof value === 'number' ? value : Number(value);
    if (!Number.isFinite(n) || n < 0) return null;
    return n;
  }

  function safeRead(obj, key) {
    if (!obj || typeof obj !== 'object') return undefined;
    try {
      return obj[key];
    } catch {
      return undefined;
    }
  }

  function normalizeHashHex(raw) {
    if (typeof raw === 'bigint') {
      if (raw < 0n || raw > U64_MAX) return null;
      return `0x${raw.toString(16).padStart(16, '0')}`;
    }
    if (typeof raw === 'number') {
      if (!Number.isSafeInteger(raw) || raw < 0) return null;
      return `0x${BigInt(raw).toString(16).padStart(16, '0')}`;
    }
    if (typeof raw === 'string') {
      const trimmed = raw.trim();
      const hex =
        trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
      if (!/^[0-9a-fA-F]{1,16}$/.test(hex)) {
        return null;
      }
      return `0x${BigInt(`0x${hex}`).toString(16).padStart(16, '0')}`;
    }
    return null;
  }

  function normalizeStageName(stage) {
    if (typeof stage === 'number') {
      return STAGE_NAMES[stage] || 'unknown';
    }
    if (typeof stage === 'string') {
      const key = stage.trim().toLowerCase();
      if (Object.prototype.hasOwnProperty.call(STAGE_ALIASES, key)) {
        return STAGE_ALIASES[key];
      }
    }
    return 'unknown';
  }

  /**
   * Sanitize a guest boot stage identifier using a strict allowlist pattern so host
   * filesystem paths, URLs, or arbitrary strings can never appear in UI or exports.
   *
   * @param {unknown} rawStage
   * @returns {string}
   */
  function sanitizeBootStage(rawStage) {
    if (typeof rawStage !== 'string' || rawStage.trim().length === 0) {
      return STAGE_NOT_STARTED;
    }
    const upper = rawStage.trim().toUpperCase();
    if (upper === STAGE_NOT_STARTED || BOOT_STAGE_SET.has(upper)) {
      return upper;
    }
    // Reject anything containing path separators, colons, dots, spaces, or query chars.
    if (/^[A-Z][A-Z0-9_]{0,63}$/.test(upper)) {
      return upper;
    }
    return 'UNKNOWN';
  }

  function normalizeBootSummary(raw) {
    if (!raw || typeof raw !== 'object') {
      return {
        stage: STAGE_NOT_STARTED,
        milestonesReached: 0,
        totalMilestones: BOOT_STAGE_ORDER.length,
        progressPct: 0,
        readyForSplash: false,
      };
    }

    const stage = sanitizeBootStage(safeRead(raw, 'stage'));
    const stageIdx = BOOT_STAGE_ORDER.indexOf(stage);

    const rawMilestones = safeRead(raw, 'milestones');
    const rawTotal = safeRead(raw, 'totalMilestones');
    const totalMilestones =
      typeof rawTotal === 'number' && Number.isInteger(rawTotal) && rawTotal >= 1
        ? rawTotal
        : Array.isArray(rawMilestones) && rawMilestones.length > 0
          ? rawMilestones.length
          : BOOT_STAGE_ORDER.length;

    const rawReached = safeRead(raw, 'milestonesReached');
    let milestonesReached;
    if (rawReached !== undefined && rawReached !== null) {
      milestonesReached = nonNegativeInt(rawReached, 0);
    } else if (Array.isArray(rawMilestones)) {
      let count = 0;
      for (let i = 0; i < rawMilestones.length; i += 1) {
        if (safeRead(rawMilestones[i], 'reached')) count += 1;
      }
      milestonesReached = count;
    } else if (stageIdx >= 0) {
      milestonesReached = stageIdx + 1;
    } else {
      milestonesReached = 0;
    }

    const rawProgress = safeRead(raw, 'progressPct');
    let progressPct;
    if (rawProgress !== undefined && rawProgress !== null) {
      progressPct = Math.round(clampPct(rawProgress));
    } else if (stageIdx >= 0) {
      progressPct = Math.round(((stageIdx + 1) / BOOT_STAGE_ORDER.length) * 100);
    } else if (milestonesReached > 0 && totalMilestones > 0) {
      progressPct = Math.round(clampPct((milestonesReached / totalMilestones) * 100));
    } else {
      progressPct = 0;
    }

    const readyForSplash =
      Boolean(safeRead(raw, 'readyForSplash')) ||
      stage === 'FIRST_PRESENT' ||
      stage === 'FIRST_VISIBLE_FRAME';

    return {
      stage,
      milestonesReached,
      totalMilestones,
      progressPct,
      readyForSplash,
    };
  }

  function normalizeFrameStats(raw) {
    if (!raw || typeof raw !== 'object') {
      return {
        framesPresented: 0,
        droppedFrames: 0,
      };
    }
    return {
      framesPresented: nonNegativeInt(safeRead(raw, 'framesPresented'), 0),
      droppedFrames: nonNegativeInt(safeRead(raw, 'droppedFrames'), 0),
    };
  }

  function normalizeFallbackStats(raw) {
    if (!raw || typeof raw !== 'object') {
      return {
        totalLookups: 0,
        packHits: 0,
        fallbackHits: 0,
        builtinHits: 0,
        hitRatePct: 0,
        overflowMissingCount: 0,
      };
    }

    const packHits = nonNegativeInt(safeRead(raw, 'packHits'), 0);
    const fallbackHits = nonNegativeInt(safeRead(raw, 'fallbackHits'), 0);
    const builtinHits = nonNegativeInt(safeRead(raw, 'builtinHits'), 0);
    const sumHits = packHits + fallbackHits + builtinHits;
    const rawTotal = safeRead(raw, 'totalLookups');
    const totalLookups =
      rawTotal !== undefined && rawTotal !== null ? nonNegativeInt(rawTotal, sumHits) : sumHits;
    const overflowMissingCount = nonNegativeInt(safeRead(raw, 'overflowMissingCount'), 0);

    const rawHitRate = safeRead(raw, 'hitRatePct');
    let hitRatePct;
    if (rawHitRate !== undefined && rawHitRate !== null && Number.isFinite(Number(rawHitRate))) {
      hitRatePct = Math.round(clampPct(Number(rawHitRate)) * 100) / 100;
    } else {
      hitRatePct = totalLookups > 0 ? Math.round((packHits / totalLookups) * 10000) / 100 : 0;
    }

    return {
      totalLookups,
      packHits,
      fallbackHits,
      builtinHits,
      hitRatePct,
      overflowMissingCount,
    };
  }

  function formatHitRatePct(hitRatePct) {
    const pct = clampPct(hitRatePct);
    return `${pct.toFixed(1)}%`;
  }

  /**
   * Sanitize a WebGPU swapchain / render-target format token against a strict allowlist
   * so host paths, URLs, or arbitrary strings can never appear in UI or exports.
   *
   * @param {unknown} rawFormat
   * @returns {string|null}
   */
  function sanitizeSwapchainFormat(rawFormat) {
    if (typeof rawFormat !== 'string') {
      return null;
    }
    const token = rawFormat.trim().toLowerCase();
    if (token.length === 0 || token.length > 32) {
      return null;
    }
    return ALLOWED_SWAPCHAIN_FORMAT_SET.has(token) ? token : null;
  }

  function createDefaultCompatStats() {
    return {
      bindGroupKindMismatches: 0,
      autoLayoutFallbacks: 0,
      bc3dFallbacks: 0,
      bc3dFallbackUploadsSkipped: 0,
      swapchainFormat: null,
      usesBlitFallback: false,
      blitCount: 0,
    };
  }

  function createDefaultIoStats() {
    return {
      pgimFullFetches: 0,
      slabCacheHits: 0,
      slabCacheMisses: 0,
      syncFastPathHits: 0,
    };
  }

  function pickFirstDefined(sources, keys) {
    for (let i = 0; i < sources.length; i += 1) {
      const src = sources[i];
      if (!src || typeof src !== 'object') continue;
      for (let j = 0; j < keys.length; j += 1) {
        const val = safeRead(src, keys[j]);
        if (val !== undefined) {
          return { found: true, value: val };
        }
      }
    }
    return { found: false, value: undefined };
  }

  function extractCompatStats(source, base) {
    let next = base ? { ...base } : createDefaultCompatStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawCompat =
      safeRead(source, 'compatStats') !== undefined
        ? safeRead(source, 'compatStats')
        : safeRead(source, 'gpuCompatStats') !== undefined
          ? safeRead(source, 'gpuCompatStats')
          : safeRead(source, 'gpu');
    if (rawCompat === null) {
      next = createDefaultCompatStats();
    }
    const compatObj = rawCompat && typeof rawCompat === 'object' ? rawCompat : null;

    const rawRenderState = safeRead(source, 'renderStateStats');
    const renderStateObj =
      rawRenderState && typeof rawRenderState === 'object' ? rawRenderState : null;

    const rawTexture = safeRead(source, 'textureStats');
    const textureObj = rawTexture && typeof rawTexture === 'object' ? rawTexture : null;

    const rawSwapchain =
      safeRead(source, 'swapchainStats') !== undefined
        ? safeRead(source, 'swapchainStats')
        : safeRead(source, 'swapchain');
    const swapchainObj = rawSwapchain && typeof rawSwapchain === 'object' ? rawSwapchain : null;

    const rawFrame = safeRead(source, 'frameStats');
    const frameObj = rawFrame && typeof rawFrame === 'object' ? rawFrame : null;

    const bindMismatch = pickFirstDefined(
      [source, compatObj, renderStateObj, frameObj],
      ['bindGroupKindMismatches']
    );
    if (bindMismatch.found) {
      next.bindGroupKindMismatches = nonNegativeInt(bindMismatch.value, 0);
    }

    const autoLayout = pickFirstDefined(
      [source, compatObj, renderStateObj, frameObj],
      ['autoLayoutFallbacks']
    );
    if (autoLayout.found) {
      next.autoLayoutFallbacks = nonNegativeInt(autoLayout.value, 0);
    }

    const bc3d = pickFirstDefined([source, compatObj, textureObj, frameObj], ['bc3dFallbacks']);
    if (bc3d.found) {
      next.bc3dFallbacks = nonNegativeInt(bc3d.value, 0);
    }

    const bc3dSkipped = pickFirstDefined(
      [source, compatObj, textureObj, frameObj],
      ['bc3dFallbackUploadsSkipped']
    );
    if (bc3dSkipped.found) {
      next.bc3dFallbackUploadsSkipped = nonNegativeInt(bc3dSkipped.value, 0);
    }

    let swapFormat = pickFirstDefined(
      [source, compatObj, swapchainObj, frameObj],
      ['swapchainFormat']
    );
    if (!swapFormat.found && swapchainObj) {
      swapFormat = pickFirstDefined([swapchainObj], ['streamFormat', 'format']);
    }
    if (swapFormat.found) {
      next.swapchainFormat = sanitizeSwapchainFormat(swapFormat.value);
    }

    const usesBlit = pickFirstDefined(
      [source, compatObj, swapchainObj, frameObj],
      ['usesBlitFallback']
    );
    if (usesBlit.found) {
      next.usesBlitFallback = Boolean(usesBlit.value);
    }

    const blitCount = pickFirstDefined([source, compatObj, swapchainObj, frameObj], ['blitCount']);
    if (blitCount.found) {
      next.blitCount = nonNegativeInt(blitCount.value, 0);
    }

    return next;
  }

  function extractIoStats(source, base) {
    let next = base ? { ...base } : createDefaultIoStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawIo =
      safeRead(source, 'ioStats') !== undefined
        ? safeRead(source, 'ioStats')
        : safeRead(source, 'httpfsStats') !== undefined
          ? safeRead(source, 'httpfsStats')
          : safeRead(source, 'io');
    if (rawIo === null) {
      next = createDefaultIoStats();
    }
    const ioObj = rawIo && typeof rawIo === 'object' ? rawIo : null;

    const pgim = pickFirstDefined([source, ioObj], ['pgimFullFetches']);
    if (pgim.found) {
      next.pgimFullFetches = nonNegativeInt(pgim.value, 0);
    }

    const slabHits = pickFirstDefined([source, ioObj], ['slabCacheHits']);
    if (slabHits.found) {
      next.slabCacheHits = nonNegativeInt(slabHits.value, 0);
    }

    const slabMisses = pickFirstDefined([source, ioObj], ['slabCacheMisses']);
    if (slabMisses.found) {
      next.slabCacheMisses = nonNegativeInt(slabMisses.value, 0);
    }

    const syncFast = pickFirstDefined([source, ioObj], ['syncFastPathHits']);
    if (syncFast.found) {
      next.syncFastPathHits = nonNegativeInt(syncFast.value, 0);
    }

    return next;
  }

  function formatBlitFallbackText(usesBlitFallback, blitCount) {
    const count = nonNegativeInt(blitCount, 0);
    return usesBlitFallback ? `Active (${count})` : `Direct (${count})`;
  }

  function sanitizeNonNegativeNumber(value) {
    if (typeof value === 'number') {
      return Number.isFinite(value) && value >= 0 ? value : null;
    }
    if (typeof value === 'string') {
      const trimmed = value.trim();
      if (trimmed.length === 0 || trimmed.length > 32) return null;
      if (!/^\d+(?:\.\d+)?$/.test(trimmed)) return null;
      const n = Number(trimmed);
      return Number.isFinite(n) && n >= 0 ? n : null;
    }
    return null;
  }

  function sanitizeNonNegativeInt(value, fallback = 0) {
    const n = sanitizeNonNegativeNumber(value);
    return n === null ? fallback : Math.floor(n);
  }

  function formatOnePercentLowFps(value) {
    const n = sanitizeNonNegativeNumber(value);
    return n === null ? PLACEHOLDER_DASH : `${n.toFixed(1)} fps`;
  }

  function formatP99FrameMs(value) {
    const n = sanitizeNonNegativeNumber(value);
    return n === null ? PLACEHOLDER_DASH : `${n.toFixed(2)} ms`;
  }

  function createDefaultPerfStats() {
    return {
      onePercentLowFps: null,
      p99FrameMs: null,
      zeroCopyUploads: 0,
      totalElidedStateCalls: 0,
      bindGroupCacheHits: 0,
      topMissingShaders: [],
      topMissingShaderOverflow: 0,
    };
  }

  /**
   * Normalize, deduplicate, sanitize, and cap a `topMissingShaders` list so only
   * canonical 64-bit hex shader hashes (plus optional stage/hit counts) are retained
   * and zero host paths or raw bytes can ever leak into UI or exports.
   *
   * @param {unknown} rawList
   * @param {number} [maxItems=8]
   * @returns {{ items: Array<{ hash: string, stage: string|null, hits: number|null }>, overflowCount: number }}
   */
  function normalizeTopMissingShaders(rawList, maxItems = DEFAULT_MAX_TOP_MISSING_SHADERS) {
    const cap =
      Number.isInteger(maxItems) && maxItems >= 1
        ? Math.min(maxItems, MAX_TOP_MISSING_SHADERS_CAP)
        : DEFAULT_MAX_TOP_MISSING_SHADERS;

    if (!rawList || typeof rawList === 'string') {
      return { items: [], overflowCount: 0 };
    }

    let rawItems;
    if (Array.isArray(rawList)) {
      rawItems = rawList;
    } else if (typeof rawList[Symbol.iterator] === 'function') {
      try {
        rawItems = Array.from(rawList);
      } catch {
        return { items: [], overflowCount: 0 };
      }
    } else {
      return { items: [], overflowCount: 0 };
    }

    const merged = new Map();
    let hasNumericHits = false;

    for (let i = 0; i < rawItems.length; i += 1) {
      const entry = rawItems[i];
      let hash = null;
      let stage = null;
      let hits = null;

      if (typeof entry === 'string' || typeof entry === 'number' || typeof entry === 'bigint') {
        hash = normalizeHashHex(entry);
      } else if (entry && typeof entry === 'object' && !ArrayBuffer.isView(entry)) {
        const rawHash =
          safeRead(entry, 'hash') !== undefined
            ? safeRead(entry, 'hash')
            : safeRead(entry, 'shaderHash') !== undefined
              ? safeRead(entry, 'shaderHash')
              : safeRead(entry, 'hashHex');
        hash = normalizeHashHex(rawHash);
        if (hash !== null) {
          const rawStage = safeRead(entry, 'stage');
          if (rawStage !== undefined && rawStage !== null) {
            stage = normalizeStageName(rawStage);
          }
          const rawHits =
            safeRead(entry, 'hits') !== undefined
              ? safeRead(entry, 'hits')
              : safeRead(entry, 'hitCount') !== undefined
                ? safeRead(entry, 'hitCount')
                : safeRead(entry, 'drawCount') !== undefined
                  ? safeRead(entry, 'drawCount')
                  : safeRead(entry, 'count');
          if (rawHits !== undefined && rawHits !== null) {
            const parsedHits = sanitizeNonNegativeNumber(rawHits);
            if (parsedHits !== null) {
              hits = Math.floor(parsedHits);
            }
          }
        }
      }

      if (hash === null) continue;
      if (hits !== null) {
        hasNumericHits = true;
      }

      const key = `${hash}|${stage || ''}`;
      const existing = merged.get(key);
      if (!existing) {
        merged.set(key, { hash, stage, hits });
      } else if (hits !== null) {
        existing.hits = existing.hits === null ? hits : existing.hits + hits;
      }
    }

    const rows = Array.from(merged.values());
    if (hasNumericHits) {
      rows.sort(
        (a, b) =>
          (b.hits === null ? -1 : b.hits) - (a.hits === null ? -1 : a.hits) ||
          a.hash.localeCompare(b.hash) ||
          String(a.stage || '').localeCompare(String(b.stage || ''))
      );
    }

    const items = rows.slice(0, cap);
    const overflowCount = Math.max(0, rows.length - items.length);
    return { items, overflowCount };
  }

  /**
   * Format a sanitized list of top missing shader hashes for compact UI rendering.
   *
   * @param {Array<{ hash?: string, hits?: number|null }|string>|null|undefined} items
   * @param {number} [overflowCount=0]
   * @param {number} [maxItems=8]
   * @returns {string}
   */
  function formatTopMissingShaders(
    items,
    overflowCount = 0,
    maxItems = DEFAULT_MAX_TOP_MISSING_SHADERS
  ) {
    if (!Array.isArray(items) || items.length === 0) {
      return PLACEHOLDER_DASH;
    }

    const cap =
      Number.isInteger(maxItems) && maxItems >= 1
        ? Math.min(maxItems, MAX_TOP_MISSING_SHADERS_CAP)
        : DEFAULT_MAX_TOP_MISSING_SHADERS;
    const parts = [];
    let validCount = 0;

    for (let i = 0; i < items.length; i += 1) {
      const entry = items[i];
      let hash = null;
      let hits = null;
      if (typeof entry === 'string' || typeof entry === 'number' || typeof entry === 'bigint') {
        hash = normalizeHashHex(entry);
      } else if (entry && typeof entry === 'object' && !ArrayBuffer.isView(entry)) {
        const rawHash =
          safeRead(entry, 'hash') !== undefined
            ? safeRead(entry, 'hash')
            : safeRead(entry, 'shaderHash') !== undefined
              ? safeRead(entry, 'shaderHash')
              : safeRead(entry, 'hashHex');
        hash = normalizeHashHex(rawHash);
        const rawHits =
          safeRead(entry, 'hits') !== undefined
            ? safeRead(entry, 'hits')
            : safeRead(entry, 'hitCount') !== undefined
              ? safeRead(entry, 'hitCount')
              : safeRead(entry, 'drawCount') !== undefined
                ? safeRead(entry, 'drawCount')
                : safeRead(entry, 'count');
        const parsedHits = sanitizeNonNegativeNumber(rawHits);
        if (parsedHits !== null) {
          hits = Math.floor(parsedHits);
        }
      }
      if (hash === null) continue;
      validCount += 1;
      if (parts.length < cap) {
        parts.push(hits !== null ? `${hash} (${hits})` : hash);
      }
    }

    if (parts.length === 0) {
      return PLACEHOLDER_DASH;
    }

    const extraOverflow = sanitizeNonNegativeInt(overflowCount, 0);
    const localOverflow = Math.max(0, validCount - parts.length);
    const totalOverflow = extraOverflow + localOverflow;
    const joined = parts.join(', ');
    return totalOverflow > 0 ? `${joined} (+${totalOverflow} more)` : joined;
  }

  function extractPerfStats(source, base, maxTopMissingShaders = DEFAULT_MAX_TOP_MISSING_SHADERS) {
    let next = base
      ? {
          ...base,
          topMissingShaders: Array.isArray(base.topMissingShaders)
            ? base.topMissingShaders.map((row) => ({ ...row }))
            : [],
        }
      : createDefaultPerfStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawPerf =
      safeRead(source, 'perfStats') !== undefined
        ? safeRead(source, 'perfStats')
        : safeRead(source, 'perf');
    if (rawPerf === null) {
      next = createDefaultPerfStats();
    }
    const perfObj = rawPerf && typeof rawPerf === 'object' ? rawPerf : null;

    const rawFrames =
      safeRead(source, 'frameStats') !== undefined
        ? safeRead(source, 'frameStats')
        : safeRead(source, 'frames');
    const framesObj = rawFrames && typeof rawFrames === 'object' ? rawFrames : null;

    const rawBink =
      safeRead(source, 'binkStats') !== undefined
        ? safeRead(source, 'binkStats')
        : safeRead(source, 'bink') !== undefined
          ? safeRead(source, 'bink')
          : safeRead(source, 'videoStats');
    const binkObj = rawBink && typeof rawBink === 'object' ? rawBink : null;

    const rawTexture = safeRead(source, 'textureStats');
    const textureObj = rawTexture && typeof rawTexture === 'object' ? rawTexture : null;

    const rawDraw =
      safeRead(source, 'drawStats') !== undefined
        ? safeRead(source, 'drawStats')
        : safeRead(source, 'drawExecutorStats');
    const drawObj = rawDraw && typeof rawDraw === 'object' ? rawDraw : null;

    const rawRenderState = safeRead(source, 'renderStateStats');
    const renderStateObj =
      rawRenderState && typeof rawRenderState === 'object' ? rawRenderState : null;

    const rawFallback = safeRead(source, 'fallbackStats');
    const fallbackObj = rawFallback && typeof rawFallback === 'object' ? rawFallback : null;

    const rawMissingSummary = safeRead(source, 'missingShaderSummary');
    const missingSummaryObj =
      rawMissingSummary && typeof rawMissingSummary === 'object' ? rawMissingSummary : null;

    const onePctLow = pickFirstDefined(
      [source, perfObj, framesObj],
      ['onePercentLowFps', 'fps1PercentLow', 'low1PctFps']
    );
    if (onePctLow.found) {
      next.onePercentLowFps = sanitizeNonNegativeNumber(onePctLow.value);
    }

    const p99 = pickFirstDefined([source, perfObj, framesObj], ['p99FrameMs', 'p99FrameTimeMs']);
    if (p99.found) {
      next.p99FrameMs = sanitizeNonNegativeNumber(p99.value);
    }

    const zeroCopy = pickFirstDefined(
      [source, perfObj, binkObj, textureObj, framesObj],
      ['zeroCopyUploads', 'directZeroCopyUploads', 'binkZeroCopyUploads']
    );
    if (zeroCopy.found) {
      next.zeroCopyUploads = sanitizeNonNegativeInt(zeroCopy.value, 0);
    }

    const elided = pickFirstDefined(
      [source, perfObj, drawObj, renderStateObj, framesObj],
      ['totalElidedStateCalls', 'elidedStateCalls']
    );
    if (elided.found) {
      next.totalElidedStateCalls = sanitizeNonNegativeInt(elided.value, 0);
    } else {
      const skipSources = [source, perfObj, drawObj, renderStateObj];
      for (let i = 0; i < skipSources.length; i += 1) {
        const src = skipSources[i];
        if (!src || typeof src !== 'object') continue;
        const pSkips = safeRead(src, 'redundantPipelineSkips');
        const bgSkips = safeRead(src, 'redundantBindGroupSkips');
        const vbSkips = safeRead(src, 'redundantVertexBufferSkips');
        const ibSkips = safeRead(src, 'redundantIndexBufferSkips');
        if (
          pSkips !== undefined ||
          bgSkips !== undefined ||
          vbSkips !== undefined ||
          ibSkips !== undefined
        ) {
          next.totalElidedStateCalls =
            sanitizeNonNegativeInt(pSkips, 0) +
            sanitizeNonNegativeInt(bgSkips, 0) +
            sanitizeNonNegativeInt(vbSkips, 0) +
            sanitizeNonNegativeInt(ibSkips, 0);
          break;
        }
      }
    }

    let bgCacheHits = pickFirstDefined(
      [source, perfObj, drawObj, renderStateObj, framesObj],
      ['bindGroupCacheHits']
    );
    if (!bgCacheHits.found && renderStateObj) {
      bgCacheHits = pickFirstDefined([renderStateObj], ['presentxBindGroupCacheHits']);
    }
    if (bgCacheHits.found) {
      next.bindGroupCacheHits = sanitizeNonNegativeInt(bgCacheHits.value, 0);
    }

    const topMissing = pickFirstDefined(
      [source, perfObj, fallbackObj, missingSummaryObj],
      ['topMissingShaders']
    );
    if (topMissing.found) {
      if (topMissing.value === null) {
        next.topMissingShaders = [];
        next.topMissingShaderOverflow = 0;
      } else {
        const normalized = normalizeTopMissingShaders(topMissing.value, maxTopMissingShaders);
        next.topMissingShaders = normalized.items;
        next.topMissingShaderOverflow = normalized.overflowCount;
      }
    }

    return next;
  }

  /**
   * Sanitize a WebGPU adapter label (or `{ vendor, architecture }` object) into a
   * lowercase safe `vendor / architecture` string of at most `MAX_GPU_ADAPTER_LENGTH`
   * (96) characters with ANSI escapes, control characters, URLs, and host paths stripped.
   *
   * @param {unknown} rawAdapter
   * @returns {string|null}
   */
  function sanitizeGpuAdapter(rawAdapter) {
    if (rawAdapter === null || rawAdapter === undefined) {
      return null;
    }

    let raw = null;
    if (typeof rawAdapter === 'string') {
      raw = rawAdapter;
    } else if (
      typeof rawAdapter === 'object' &&
      !Array.isArray(rawAdapter) &&
      !ArrayBuffer.isView(rawAdapter) &&
      !(rawAdapter instanceof ArrayBuffer)
    ) {
      const rawVendor = safeRead(rawAdapter, 'vendor');
      const rawArch =
        safeRead(rawAdapter, 'architecture') !== undefined
          ? safeRead(rawAdapter, 'architecture')
          : safeRead(rawAdapter, 'arch');
      const vendor = typeof rawVendor === 'string' ? rawVendor.trim() : '';
      const arch = typeof rawArch === 'string' ? rawArch.trim() : '';
      if (vendor && arch) {
        raw = `${vendor} / ${arch}`;
      } else if (vendor || arch) {
        raw = vendor || arch;
      } else {
        const nested =
          safeRead(rawAdapter, 'gpuAdapter') !== undefined
            ? safeRead(rawAdapter, 'gpuAdapter')
            : safeRead(rawAdapter, 'adapterLabel');
        if (typeof nested === 'string') {
          raw = nested;
        }
      }
    }

    if (typeof raw !== 'string' || raw.length === 0) {
      return null;
    }

    const cleanedRaw = raw
      .replace(ANSI_ESCAPE_RE, ' ')
      .replace(/[\r\n\t]+/g, ' ')
      .replace(CONTROL_CHAR_RE, '')
      .trim();

    if (cleanedRaw.length === 0 || HOST_PATH_PREFIX_RE.test(cleanedRaw)) {
      return null;
    }

    let out = cleanedRaw
      .replace(URL_TOKEN_RE, ' ')
      .replace(WIN_DRIVE_PATH_RE, ' ')
      .replace(BACKSLASH_TOKEN_RE, ' ')
      .replace(RELATIVE_PATH_TOKEN_RE, '$1')
      .replace(POSIX_ROOT_PATH_RE, '$1')
      .replace(ROOT_REL_USER_PATH_RE, '$1')
      .replace(REDACTED_TOKEN_RE, ' ')
      .replace(/\.\.+/g, ' ')
      .toLowerCase()
      .replace(/[^a-z0-9 /._+()-]+/g, ' ')
      .replace(/\(\s*\)/g, ' ')
      .replace(/\s+/g, ' ')
      .replace(/\s*\/\s*/g, ' / ')
      .replace(/(?:\s*\/\s*){2,}/g, ' / ')
      .replace(/^[\s/._+-]+|[\s/._+-]+$/g, '');

    if (out.length > MAX_GPU_ADAPTER_LENGTH) {
      out = out.slice(0, MAX_GPU_ADAPTER_LENGTH).replace(/[\s/._+-]+$/, '');
    }

    if (!/[a-z0-9]/.test(out)) {
      return null;
    }

    return out.length > 0 ? out : null;
  }

  /**
   * Format a non-negative millisecond duration (e.g. `firstVisibleFrameMs`, `blackLoadingMs`)
   * for compact UI rendering, returning `—` when absent or invalid.
   *
   * @param {unknown} value
   * @returns {string}
   */
  function formatDurationMs(value) {
    const n = sanitizeNonNegativeNumber(value);
    return n === null ? PLACEHOLDER_DASH : `${n.toFixed(1)} ms`;
  }

  function createDefaultAdapterStats() {
    return {
      gpuAdapter: null,
    };
  }

  function createDefaultVisibilityStats() {
    return {
      firstVisibleFrameMs: null,
      blackLoadingMs: null,
      blackPresentsTotal: 0,
    };
  }

  function createDefaultReflectionStats() {
    return {
      reflectedStages: 0,
      inlineSamplerStages: 0,
      resourceHeapBindings: 0,
    };
  }

  function pickFirstDefinedWithSource(sources, keys) {
    for (let i = 0; i < sources.length; i += 1) {
      const entry = sources[i];
      const src = entry && entry.obj;
      if (!src || typeof src !== 'object') continue;
      for (let j = 0; j < keys.length; j += 1) {
        const val = safeRead(src, keys[j]);
        if (val !== undefined) {
          return { found: true, value: val, explicitNull: Boolean(entry.explicitNull) };
        }
      }
    }
    return { found: false, value: undefined, explicitNull: false };
  }

  function extractAdapterStats(source, base) {
    let next = base ? { ...base } : createDefaultAdapterStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawAdapterStats = safeRead(source, 'adapterStats');
    if (rawAdapterStats === null) {
      next = createDefaultAdapterStats();
    }
    const adapterStatsObj =
      rawAdapterStats && typeof rawAdapterStats === 'object' ? rawAdapterStats : null;

    const rawBoot =
      safeRead(source, 'bootSummary') !== undefined
        ? safeRead(source, 'bootSummary')
        : safeRead(source, 'boot');
    const bootObj = rawBoot && typeof rawBoot === 'object' ? rawBoot : null;

    const rawCompat =
      safeRead(source, 'compatStats') !== undefined
        ? safeRead(source, 'compatStats')
        : safeRead(source, 'gpuCompatStats') !== undefined
          ? safeRead(source, 'gpuCompatStats')
          : safeRead(source, 'gpu');
    const compatObj = rawCompat && typeof rawCompat === 'object' ? rawCompat : null;

    const rawDevice =
      safeRead(source, 'deviceStats') !== undefined
        ? safeRead(source, 'deviceStats')
        : safeRead(source, 'adapterInfo');
    const deviceObj = rawDevice && typeof rawDevice === 'object' ? rawDevice : null;

    const picked = pickFirstDefinedWithSource(
      [
        { obj: source, explicitNull: true },
        { obj: adapterStatsObj, explicitNull: true },
        { obj: bootObj, explicitNull: false },
        { obj: compatObj, explicitNull: false },
        { obj: deviceObj, explicitNull: false },
      ],
      ['gpuAdapter', 'adapterLabel', 'adapterInfo']
    );

    if (picked.found) {
      if (picked.value === null) {
        if (picked.explicitNull) {
          next.gpuAdapter = null;
        }
      } else {
        next.gpuAdapter = sanitizeGpuAdapter(picked.value);
      }
    } else if (
      deviceObj &&
      (safeRead(deviceObj, 'vendor') !== undefined ||
        safeRead(deviceObj, 'architecture') !== undefined ||
        safeRead(deviceObj, 'arch') !== undefined)
    ) {
      next.gpuAdapter = sanitizeGpuAdapter(deviceObj);
    }

    return next;
  }

  function extractVisibilityStats(source, base) {
    let next = base ? { ...base } : createDefaultVisibilityStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawVis =
      safeRead(source, 'visibilityStats') !== undefined
        ? safeRead(source, 'visibilityStats')
        : safeRead(source, 'visibleFrameStats');
    if (rawVis === null) {
      next = createDefaultVisibilityStats();
    }
    const visObj = rawVis && typeof rawVis === 'object' ? rawVis : null;

    const rawBoot =
      safeRead(source, 'bootSummary') !== undefined
        ? safeRead(source, 'bootSummary')
        : safeRead(source, 'boot');
    const bootObj = rawBoot && typeof rawBoot === 'object' ? rawBoot : null;

    const rawSwapchain =
      safeRead(source, 'swapchainStats') !== undefined
        ? safeRead(source, 'swapchainStats')
        : safeRead(source, 'swapchain');
    const swapchainObj = rawSwapchain && typeof rawSwapchain === 'object' ? rawSwapchain : null;

    const rawFrame =
      safeRead(source, 'frameStats') !== undefined
        ? safeRead(source, 'frameStats')
        : safeRead(source, 'frames');
    const frameObj = rawFrame && typeof rawFrame === 'object' ? rawFrame : null;

    const rawGpu = safeRead(source, 'gpu');
    const gpuObj = rawGpu && typeof rawGpu === 'object' ? rawGpu : null;

    const rawPerf =
      safeRead(source, 'perfStats') !== undefined
        ? safeRead(source, 'perfStats')
        : safeRead(source, 'perf');
    const perfObj = rawPerf && typeof rawPerf === 'object' ? rawPerf : null;

    const sources = [
      { obj: source, explicitNull: true },
      { obj: visObj, explicitNull: true },
      { obj: bootObj, explicitNull: false },
      { obj: swapchainObj, explicitNull: false },
      { obj: frameObj, explicitNull: false },
      { obj: gpuObj, explicitNull: false },
      { obj: perfObj, explicitNull: false },
    ];

    const firstVisible = pickFirstDefinedWithSource(sources, [
      'firstVisibleFrameMs',
      'firstVisiblePresentMs',
    ]);
    if (firstVisible.found) {
      if (firstVisible.value === null) {
        if (firstVisible.explicitNull) {
          next.firstVisibleFrameMs = null;
        }
      } else {
        next.firstVisibleFrameMs = sanitizeNonNegativeNumber(firstVisible.value);
      }
    }

    const blackLoading = pickFirstDefinedWithSource(sources, ['blackLoadingMs']);
    if (blackLoading.found) {
      if (blackLoading.value === null) {
        if (blackLoading.explicitNull) {
          next.blackLoadingMs = null;
        }
      } else {
        next.blackLoadingMs = sanitizeNonNegativeNumber(blackLoading.value);
      }
    }

    const blackPresents = pickFirstDefinedWithSource(sources, ['blackPresentsTotal']);
    if (blackPresents.found) {
      if (blackPresents.value === null) {
        if (blackPresents.explicitNull) {
          next.blackPresentsTotal = 0;
        }
      } else {
        next.blackPresentsTotal = sanitizeNonNegativeInt(blackPresents.value, 0);
      }
    }

    return next;
  }

  function extractReflectionStats(source, base) {
    let next = base ? { ...base } : createDefaultReflectionStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawRefl =
      safeRead(source, 'reflectionStats') !== undefined
        ? safeRead(source, 'reflectionStats')
        : safeRead(source, 'shaderReflectionStats') !== undefined
          ? safeRead(source, 'shaderReflectionStats')
          : safeRead(source, 'reflection') !== undefined
            ? safeRead(source, 'reflection')
            : safeRead(source, 's44Stats');
    if (rawRefl === null) {
      next = createDefaultReflectionStats();
    }
    const reflObj = rawRefl && typeof rawRefl === 'object' ? rawRefl : null;

    const rawShaderPack =
      safeRead(source, 'shaderPackStats') !== undefined
        ? safeRead(source, 'shaderPackStats')
        : safeRead(source, 'packStats');
    const shaderPackObj = rawShaderPack && typeof rawShaderPack === 'object' ? rawShaderPack : null;

    const rawGpu = safeRead(source, 'gpu');
    const gpuObj = rawGpu && typeof rawGpu === 'object' ? rawGpu : null;

    let reflectedStages = pickFirstDefined(
      [source, reflObj, shaderPackObj, gpuObj],
      ['reflectedStages']
    );
    if (!reflectedStages.found && reflObj) {
      reflectedStages = pickFirstDefined([reflObj], ['stageCount', 'shaderCount']);
    }
    if (reflectedStages.found) {
      next.reflectedStages = sanitizeNonNegativeInt(reflectedStages.value, 0);
    }

    const inlineSamplerStages = pickFirstDefined(
      [source, reflObj, shaderPackObj, gpuObj],
      ['inlineSamplerStages']
    );
    if (inlineSamplerStages.found) {
      next.inlineSamplerStages = sanitizeNonNegativeInt(inlineSamplerStages.value, 0);
    }

    const resourceHeapBindings = pickFirstDefined(
      [source, reflObj, shaderPackObj, gpuObj],
      ['resourceHeapBindings']
    );
    if (resourceHeapBindings.found) {
      next.resourceHeapBindings = sanitizeNonNegativeInt(resourceHeapBindings.value, 0);
    }

    return next;
  }

  /**
   * Normalize and sanitize a single missing-shader entry into a canonical row
   * (`hash` hex, `stage`, `hits`, optional numeric timestamps, and display `text`).
   * Returns `null` if the entry lacks a valid 64-bit hash (rejecting paths/buffers).
   *
   * @param {object|null|undefined} entry
   * @returns {{ hash: string, stage: string, hits: number, firstSeenMs: number|null, lastSeenMs: number|null, text: string }|null}
   */
  function formatMissingShaderRow(entry) {
    if (!entry || typeof entry !== 'object') {
      return null;
    }

    const rawHash =
      safeRead(entry, 'hash') !== undefined
        ? safeRead(entry, 'hash')
        : safeRead(entry, 'shaderHash') !== undefined
          ? safeRead(entry, 'shaderHash')
          : safeRead(entry, 'hashHex');

    const hash = normalizeHashHex(rawHash);
    if (hash === null) {
      return null;
    }

    const stage = normalizeStageName(safeRead(entry, 'stage'));

    const rawHits =
      safeRead(entry, 'hits') !== undefined
        ? safeRead(entry, 'hits')
        : safeRead(entry, 'hitCount') !== undefined
          ? safeRead(entry, 'hitCount')
          : safeRead(entry, 'drawCount') !== undefined
            ? safeRead(entry, 'drawCount')
            : safeRead(entry, 'count');

    const hits = rawHits === undefined || rawHits === null ? 1 : nonNegativeInt(rawHits, 0);
    const firstSeenMs = nonNegativeFinite(safeRead(entry, 'firstSeenMs'));
    const lastSeenMs = nonNegativeFinite(safeRead(entry, 'lastSeenMs'));

    return {
      hash,
      stage,
      hits,
      firstSeenMs,
      lastSeenMs,
      text: `${hash} | ${stage} | ${hits}`,
    };
  }

  function deduplicateMissingShaders(rawList) {
    if (!rawList || typeof rawList === 'string') {
      return [];
    }

    let items;
    if (Array.isArray(rawList)) {
      items = rawList;
    } else if (typeof rawList[Symbol.iterator] === 'function') {
      try {
        items = Array.from(rawList);
      } catch {
        return [];
      }
    } else {
      return [];
    }

    const merged = new Map();
    for (let i = 0; i < items.length; i += 1) {
      const row = formatMissingShaderRow(items[i]);
      if (!row) continue;

      const key = `${row.hash}|${row.stage}`;
      const existing = merged.get(key);
      if (!existing) {
        merged.set(key, {
          hash: row.hash,
          stage: row.stage,
          hits: row.hits,
          firstSeenMs: row.firstSeenMs,
          lastSeenMs: row.lastSeenMs,
          text: row.text,
        });
      } else {
        existing.hits += row.hits;
        if (row.firstSeenMs !== null) {
          existing.firstSeenMs =
            existing.firstSeenMs === null
              ? row.firstSeenMs
              : Math.min(existing.firstSeenMs, row.firstSeenMs);
        }
        if (row.lastSeenMs !== null) {
          existing.lastSeenMs =
            existing.lastSeenMs === null
              ? row.lastSeenMs
              : Math.max(existing.lastSeenMs, row.lastSeenMs);
        }
        existing.text = `${existing.hash} | ${existing.stage} | ${existing.hits}`;
      }
    }

    const rows = Array.from(merged.values());
    rows.sort(
      (a, b) =>
        b.hits - a.hits || a.hash.localeCompare(b.hash) || a.stage.localeCompare(b.stage)
    );
    return rows;
  }

  function formatIsoTimestamp(value) {
    if (typeof value === 'string') {
      const parsed = Date.parse(value);
      if (Number.isFinite(parsed)) {
        return new Date(parsed).toISOString();
      }
    }
    if (typeof value === 'number' && Number.isFinite(value)) {
      return new Date(value).toISOString();
    }
    return new Date().toISOString();
  }

  /**
   * Build a sanitized, JSON-serializable missing-shader shopping list export payload.
   * Uses strict allowlist projection so zero host paths and zero raw bytes can ever leak.
   *
   * Accepts either an array of raw missing-shader rows or a state object
   * `{ bootSummary, missingShaders, fallbackStats, frameStats, exportedAt }`.
   *
   * @param {Array|object|null|undefined} input
   * @returns {object}
   */
  function buildMissingShaderExport(input) {
    const isArrayInput = Array.isArray(input);
    const source = !isArrayInput && input && typeof input === 'object' ? input : {};

    const rawMissing = isArrayInput ? input : safeRead(source, 'missingShaders');
    const rows = deduplicateMissingShaders(rawMissing);

    const boot = normalizeBootSummary(
      safeRead(source, 'bootSummary') !== undefined
        ? safeRead(source, 'bootSummary')
        : safeRead(source, 'boot')
    );
    const frameStats = normalizeFrameStats(safeRead(source, 'frameStats'));
    const fallbackStats = normalizeFallbackStats(safeRead(source, 'fallbackStats'));
    const compatStats = extractCompatStats(source, null);
    const ioStats = extractIoStats(source, null);
    const rawMaxTop = safeRead(source, 'maxTopMissingShaders');
    const maxTop =
      Number.isInteger(rawMaxTop) && rawMaxTop >= 1 ? rawMaxTop : DEFAULT_MAX_TOP_MISSING_SHADERS;
    const perfStats = extractPerfStats(source, null, maxTop);
    const adapterStats = extractAdapterStats(source, null);
    const visibilityStats = extractVisibilityStats(source, null);
    const reflectionStats = extractReflectionStats(source, null);

    let exportedAtValue = safeRead(source, 'exportedAt');
    if (exportedAtValue === undefined) {
      const nowFn = safeRead(source, 'now');
      if (typeof nowFn === 'function') {
        try {
          exportedAtValue = nowFn();
        } catch {
          exportedAtValue = Date.now();
        }
      }
    }

    const missingShaders = rows.map((row) => {
      const item = {
        hash: row.hash,
        stage: row.stage,
        hits: row.hits,
      };
      if (row.firstSeenMs !== null) {
        item.firstSeenMs = row.firstSeenMs;
      }
      if (row.lastSeenMs !== null) {
        item.lastSeenMs = row.lastSeenMs;
      }
      return item;
    });

    return {
      format: EXPORT_FORMAT,
      schemaVersion: EXPORT_SCHEMA_VERSION,
      exportedAt: formatIsoTimestamp(exportedAtValue),
      boot: {
        stage: boot.stage,
        milestonesReached: boot.milestonesReached,
        totalMilestones: boot.totalMilestones,
        progressPct: boot.progressPct,
        readyForSplash: boot.readyForSplash,
      },
      gpu: {
        framesPresented: frameStats.framesPresented,
        droppedFrames: frameStats.droppedFrames,
        totalLookups: fallbackStats.totalLookups,
        packHits: fallbackStats.packHits,
        fallbackHits: fallbackStats.fallbackHits,
        builtinHits: fallbackStats.builtinHits,
        hitRatePct: fallbackStats.hitRatePct,
        uniqueMissing: missingShaders.length,
        overflowMissingCount: fallbackStats.overflowMissingCount,
        bindGroupKindMismatches: compatStats.bindGroupKindMismatches,
        autoLayoutFallbacks: compatStats.autoLayoutFallbacks,
        bc3dFallbacks: compatStats.bc3dFallbacks,
        bc3dFallbackUploadsSkipped: compatStats.bc3dFallbackUploadsSkipped,
        swapchainFormat: compatStats.swapchainFormat,
        usesBlitFallback: compatStats.usesBlitFallback,
        blitCount: compatStats.blitCount,
        gpuAdapter: adapterStats.gpuAdapter,
        firstVisibleFrameMs: visibilityStats.firstVisibleFrameMs,
        blackLoadingMs: visibilityStats.blackLoadingMs,
        blackPresentsTotal: visibilityStats.blackPresentsTotal,
      },
      io: {
        pgimFullFetches: ioStats.pgimFullFetches,
        slabCacheHits: ioStats.slabCacheHits,
        slabCacheMisses: ioStats.slabCacheMisses,
        syncFastPathHits: ioStats.syncFastPathHits,
      },
      reflection: {
        reflectedStages: reflectionStats.reflectedStages,
        inlineSamplerStages: reflectionStats.inlineSamplerStages,
        resourceHeapBindings: reflectionStats.resourceHeapBindings,
      },
      perf: {
        onePercentLowFps: perfStats.onePercentLowFps,
        p99FrameMs: perfStats.p99FrameMs,
        zeroCopyUploads: perfStats.zeroCopyUploads,
        totalElidedStateCalls: perfStats.totalElidedStateCalls,
        bindGroupCacheHits: perfStats.bindGroupCacheHits,
        topMissingShaders: perfStats.topMissingShaders.map((row) => {
          const item = { hash: row.hash };
          if (row.stage !== null) item.stage = row.stage;
          if (row.hits !== null) item.hits = row.hits;
          return item;
        }),
      },
      missingShaderCount: missingShaders.length,
      missingShaders,
    };
  }

  function ensureSharedStyle(doc) {
    if (!doc || typeof doc.createElement !== 'function') return null;
    try {
      const head =
        doc.head ||
        (typeof doc.getElementsByTagName === 'function' && doc.getElementsByTagName('head')[0]) ||
        null;
      if (!head) return null;
      if (typeof doc.querySelector === 'function') {
        const existing = doc.querySelector(`style[${STYLE_ATTR}]`);
        if (existing) return existing;
      }
      const style = doc.createElement('style');
      if (typeof style.setAttribute === 'function') {
        style.setAttribute(STYLE_ATTR, '1');
      }
      style.textContent = SHARED_CSS;
      if (typeof head.appendChild === 'function') {
        head.appendChild(style);
      }
      return style;
    } catch {
      return null;
    }
  }

  function createEl(doc, tag, className, attrs) {
    const el = doc.createElement(tag);
    if (className) el.className = className;
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null) continue;
        if (key === 'text') {
          el.textContent = String(value);
        } else if (typeof el.setAttribute === 'function') {
          el.setAttribute(key, String(value));
        }
      }
    }
    return el;
  }

  function clearElementChildren(el) {
    if (!el) return;
    try {
      if (Array.isArray(el.children) && typeof el.removeChild === 'function') {
        while (el.children.length > 0) {
          el.removeChild(el.children[el.children.length - 1]);
        }
        return;
      }
      if (typeof el.removeChild === 'function') {
        while (el.firstChild) {
          el.removeChild(el.firstChild);
        }
        return;
      }
      if (typeof el.replaceChildren === 'function') {
        el.replaceChildren();
      }
    } catch {
      // Ignore DOM cleanup errors on partial mocks.
    }
  }

  /**
   * Live Frame-1 GPU & Missing-Shader Diagnostics Panel controller.
   */
  class GpuBootPanelController {
    /**
     * @param {object} [options]
     * @param {Element|object|null} [options.container]
     * @param {Document|object} [options.documentObj]
     * @param {number} [options.maxRows=32]
     * @param {((jsonText: string, payload: object) => void)|null} [options.onExportMissingShaders]
     * @param {boolean} [options.collapsed=false]
     * @param {boolean} [options.visible=true]
     * @param {boolean} [options.autoMount=true]
     * @param {string} [options.ariaLabel]
     * @param {() => number} [options.now]
     * @param {Element|object|null} [options.root]
     */
    constructor(options = {}) {
      if (options !== undefined && (options === null || typeof options !== 'object')) {
        throw new TypeError('GpuBootPanelController options must be an object');
      }

      const opts = options || {};
      const doc =
        opts.documentObj ||
        (typeof document !== 'undefined' ? document : null) ||
        (root && root.document) ||
        null;
      if (!doc || typeof doc.createElement !== 'function') {
        throw new TypeError('GpuBootPanelController requires a document with createElement');
      }

      const maxRows = opts.maxRows !== undefined ? opts.maxRows : DEFAULT_MAX_ROWS;
      if (!Number.isInteger(maxRows) || maxRows < 1) {
        throw new RangeError(`maxRows must be a positive integer, got ${maxRows}`);
      }

      const maxTopMissingShaders =
        opts.maxTopMissingShaders !== undefined
          ? opts.maxTopMissingShaders
          : DEFAULT_MAX_TOP_MISSING_SHADERS;
      if (!Number.isInteger(maxTopMissingShaders) || maxTopMissingShaders < 1) {
        throw new RangeError(
          `maxTopMissingShaders must be a positive integer, got ${maxTopMissingShaders}`
        );
      }

      if (
        opts.onExportMissingShaders !== undefined &&
        opts.onExportMissingShaders !== null &&
        typeof opts.onExportMissingShaders !== 'function'
      ) {
        throw new TypeError('onExportMissingShaders must be a function or null');
      }

      if (opts.now !== undefined && typeof opts.now !== 'function') {
        throw new TypeError('now must be a function');
      }

      this._document = doc;
      this._container =
        opts.container ||
        opts.parent ||
        opts.mountEl ||
        (doc.body && typeof doc.body.appendChild === 'function' ? doc.body : null) ||
        (typeof doc.documentElement !== 'undefined' ? doc.documentElement : null);
      this._maxRows = maxRows;
      this._maxTopMissingShaders = Math.min(maxTopMissingShaders, MAX_TOP_MISSING_SHADERS_CAP);
      this._onExportMissingShaders =
        typeof opts.onExportMissingShaders === 'function' ? opts.onExportMissingShaders : null;
      this._ariaLabel =
        typeof opts.ariaLabel === 'string' && opts.ariaLabel.trim().length > 0
          ? opts.ariaLabel.trim()
          : DEFAULT_ARIA_LABEL;
      this._now = typeof opts.now === 'function' ? opts.now : () => Date.now();

      this._panelId = `pg6-gpu-panel-body-${nextPanelId++}`;
      this._destroyed = false;
      this._visible = opts.visible !== false;
      this._collapsed = Boolean(opts.collapsed);
      this._exportCount = 0;

      this._boot = normalizeBootSummary(null);
      this._frameStats = normalizeFrameStats(null);
      this._fallbackStats = normalizeFallbackStats(null);
      this._compatStats = createDefaultCompatStats();
      this._ioStats = createDefaultIoStats();
      this._perfStats = createDefaultPerfStats();
      this._adapterStats = createDefaultAdapterStats();
      this._visibilityStats = createDefaultVisibilityStats();
      this._reflectionStats = createDefaultReflectionStats();
      this._missingShaders = [];
      this._renderedCache = Object.create(null);

      this._styleEl = ensureSharedStyle(doc);
      this._root = opts.root && typeof opts.root === 'object' ? opts.root : this._buildDom();
      this._bindListeners();

      if (opts.autoMount !== false && this._container && this._root && !this._root.parentNode) {
        try {
          if (typeof this._container.appendChild === 'function') {
            this._container.appendChild(this._root);
          }
        } catch {
          // Container mocks may omit appendChild; controller still remains usable.
        }
      }

      this._applyVisibility(this._visible);
      this._applyCollapsed(this._collapsed);
      this._render();
    }

    _buildMetricCard(doc, label, valueClass, initialText) {
      const card = createEl(doc, 'div', `${ROOT_CLASS}__metric`);
      const lbl = createEl(doc, 'span', `${ROOT_CLASS}__metric-label`, { text: label });
      const val = createEl(doc, 'span', `${ROOT_CLASS}__metric-value ${valueClass}`, {
        text: initialText,
      });
      card.appendChild(lbl);
      card.appendChild(val);
      return { card, val };
    }

    _buildDom() {
      const doc = this._document;
      const rootEl = createEl(doc, 'section', ROOT_CLASS, {
        role: 'region',
        'aria-label': this._ariaLabel,
        'data-collapsed': String(this._collapsed),
      });

      const header = createEl(doc, 'div', `${ROOT_CLASS}__header`);
      const heading = createEl(doc, 'div', `${ROOT_CLASS}__heading`);
      const title = createEl(doc, 'span', `${ROOT_CLASS}__title`, {
        text: 'GPU & Shader Diagnostics',
      });
      const summaryBadge = createEl(doc, 'span', `${ROOT_CLASS}__summary-badge`, {
        text: 'NOT_STARTED · 0%',
      });
      heading.appendChild(title);
      heading.appendChild(summaryBadge);

      const actions = createEl(doc, 'div', `${ROOT_CLASS}__actions`);
      const exportBtn = createEl(doc, 'button', `${ROOT_CLASS}__btn ${ROOT_CLASS}__export-btn`, {
        type: 'button',
        'aria-label': 'Export missing shaders JSON',
        text: 'Export JSON',
      });
      const toggleBtn = createEl(doc, 'button', `${ROOT_CLASS}__btn ${ROOT_CLASS}__toggle-btn`, {
        type: 'button',
        'aria-expanded': String(!this._collapsed),
        'aria-controls': this._panelId,
        text: this._collapsed ? 'Expand' : 'Collapse',
      });
      actions.appendChild(exportBtn);
      actions.appendChild(toggleBtn);

      header.appendChild(heading);
      header.appendChild(actions);

      const body = createEl(doc, 'div', `${ROOT_CLASS}__body`, {
        id: this._panelId,
      });

      // --- Boot milestone section ---
      const bootSection = createEl(doc, 'div', `${ROOT_CLASS}__section ${ROOT_CLASS}__boot`);
      bootSection.appendChild(
        createEl(doc, 'div', `${ROOT_CLASS}__section-title`, { text: 'Guest Boot Milestone' })
      );
      const bootGrid = createEl(doc, 'div', `${ROOT_CLASS}__grid`);
      const bootStageMetric = this._buildMetricCard(
        doc,
        'Stage',
        `${ROOT_CLASS}__boot-stage`,
        STAGE_NOT_STARTED
      );
      const bootMilestonesMetric = this._buildMetricCard(
        doc,
        'Milestones',
        `${ROOT_CLASS}__boot-milestones`,
        `0/${BOOT_STAGE_ORDER.length}`
      );
      const bootReadyMetric = this._buildMetricCard(
        doc,
        'Splash Ready',
        `${ROOT_CLASS}__boot-ready`,
        'Pending'
      );
      bootGrid.appendChild(bootStageMetric.card);
      bootGrid.appendChild(bootMilestonesMetric.card);
      bootGrid.appendChild(bootReadyMetric.card);

      const progressRow = createEl(doc, 'div', `${ROOT_CLASS}__progress-row`);
      const bootProgress = createEl(doc, 'span', `${ROOT_CLASS}__boot-progress`, { text: '0%' });
      const progressBar = createEl(doc, 'div', `${ROOT_CLASS}__progress-bar`, {
        role: 'progressbar',
        'aria-label': 'Guest boot progress',
        'aria-valuemin': '0',
        'aria-valuemax': '100',
        'aria-valuenow': '0',
      });
      const progressFill = createEl(doc, 'div', `${ROOT_CLASS}__progress-fill`);
      progressBar.appendChild(progressFill);
      progressRow.appendChild(bootProgress);
      progressRow.appendChild(progressBar);

      bootSection.appendChild(bootGrid);
      bootSection.appendChild(progressRow);

      // --- GPU Frame-1 counters section ---
      const countersSection = createEl(
        doc,
        'div',
        `${ROOT_CLASS}__section ${ROOT_CLASS}__counters`
      );
      countersSection.appendChild(
        createEl(doc, 'div', `${ROOT_CLASS}__section-title`, { text: 'GPU Frame-1 Counters' })
      );
      const countersGrid = createEl(doc, 'div', `${ROOT_CLASS}__grid`);
      const framesPresentedMetric = this._buildMetricCard(
        doc,
        'Frames Presented',
        `${ROOT_CLASS}__frames-presented`,
        '0'
      );
      const droppedFramesMetric = this._buildMetricCard(
        doc,
        'Dropped Frames',
        `${ROOT_CLASS}__dropped-frames`,
        '0'
      );
      const hitRateMetric = this._buildMetricCard(
        doc,
        'Pack Hit Rate',
        `${ROOT_CLASS}__hit-rate`,
        '0.0%'
      );
      const packHitsMetric = this._buildMetricCard(
        doc,
        'Pack Hits',
        `${ROOT_CLASS}__pack-hits`,
        '0'
      );
      const fallbackHitsMetric = this._buildMetricCard(
        doc,
        'Fallback Hits',
        `${ROOT_CLASS}__fallback-hits`,
        '0'
      );
      const builtinHitsMetric = this._buildMetricCard(
        doc,
        'Builtin Hits',
        `${ROOT_CLASS}__builtin-hits`,
        '0'
      );

      countersGrid.appendChild(framesPresentedMetric.card);
      countersGrid.appendChild(droppedFramesMetric.card);
      countersGrid.appendChild(hitRateMetric.card);
      countersGrid.appendChild(packHitsMetric.card);
      countersGrid.appendChild(fallbackHitsMetric.card);
      countersGrid.appendChild(builtinHitsMetric.card);
      countersSection.appendChild(countersGrid);

      // --- WebGPU compatibility & swapchain diagnostics section ---
      const compatSection = createEl(
        doc,
        'div',
        `${ROOT_CLASS}__section ${ROOT_CLASS}__compat`,
        {
          role: 'group',
          'aria-label': 'WebGPU Compatibility & Swapchain',
        }
      );
      compatSection.appendChild(
        createEl(doc, 'div', `${ROOT_CLASS}__section-title`, { text: 'WebGPU Compatibility' })
      );
      const compatGrid = createEl(doc, 'div', `${ROOT_CLASS}__grid`);
      const bindMismatchesMetric = this._buildMetricCard(
        doc,
        'Bind Mismatches',
        `${ROOT_CLASS}__bind-mismatches`,
        '0'
      );
      const autoLayoutMetric = this._buildMetricCard(
        doc,
        'Auto-Layout',
        `${ROOT_CLASS}__auto-layout-fallbacks`,
        '0'
      );
      const bc3dFallbacksMetric = this._buildMetricCard(
        doc,
        '3D BC Fallbacks',
        `${ROOT_CLASS}__bc3d-fallbacks`,
        '0'
      );
      const bc3dSkippedMetric = this._buildMetricCard(
        doc,
        '3D BC Skipped',
        `${ROOT_CLASS}__bc3d-uploads-skipped`,
        '0'
      );
      const swapchainFormatMetric = this._buildMetricCard(
        doc,
        'Swapchain',
        `${ROOT_CLASS}__swapchain-format`,
        PLACEHOLDER_DASH
      );
      const blitFallbackMetric = this._buildMetricCard(
        doc,
        'Blit Fallback',
        `${ROOT_CLASS}__blit-fallback`,
        formatBlitFallbackText(false, 0)
      );
      compatGrid.appendChild(bindMismatchesMetric.card);
      compatGrid.appendChild(autoLayoutMetric.card);
      compatGrid.appendChild(bc3dFallbacksMetric.card);
      compatGrid.appendChild(bc3dSkippedMetric.card);
      compatGrid.appendChild(swapchainFormatMetric.card);
      compatGrid.appendChild(blitFallbackMetric.card);
      compatSection.appendChild(compatGrid);

      // --- HTTPFS & cache acceleration telemetry section ---
      const ioSection = createEl(
        doc,
        'div',
        `${ROOT_CLASS}__section ${ROOT_CLASS}__io`,
        {
          role: 'group',
          'aria-label': 'HTTPFS & Cache Telemetry',
        }
      );
      ioSection.appendChild(
        createEl(doc, 'div', `${ROOT_CLASS}__section-title`, { text: 'HTTPFS & Cache' })
      );
      const ioGrid = createEl(doc, 'div', `${ROOT_CLASS}__grid ${ROOT_CLASS}__grid--4`);
      const pgimFetchesMetric = this._buildMetricCard(
        doc,
        'PGIM Fetches',
        `${ROOT_CLASS}__pgim-fetches`,
        '0'
      );
      const slabHitsMetric = this._buildMetricCard(
        doc,
        'Slab Hits',
        `${ROOT_CLASS}__slab-hits`,
        '0'
      );
      const slabMissesMetric = this._buildMetricCard(
        doc,
        'Slab Misses',
        `${ROOT_CLASS}__slab-misses`,
        '0'
      );
      const syncFastHitsMetric = this._buildMetricCard(
        doc,
        'Sync Fast Path',
        `${ROOT_CLASS}__sync-fast-hits`,
        '0'
      );
      ioGrid.appendChild(pgimFetchesMetric.card);
      ioGrid.appendChild(slabHitsMetric.card);
      ioGrid.appendChild(slabMissesMetric.card);
      ioGrid.appendChild(syncFastHitsMetric.card);
      ioSection.appendChild(ioGrid);

      // --- Performance, Bink2 zero-copy, state-elision & top missing shaders section (W212) ---
      const perfSection = createEl(
        doc,
        'div',
        `${ROOT_CLASS}__section ${ROOT_CLASS}__perf`,
        {
          role: 'group',
          'aria-label': 'Performance & GPU Telemetry',
        }
      );
      perfSection.appendChild(
        createEl(doc, 'div', `${ROOT_CLASS}__section-title`, {
          text: 'Performance & GPU Telemetry',
        })
      );
      const perfGrid = createEl(doc, 'div', `${ROOT_CLASS}__grid`);
      const onePercentLowFpsMetric = this._buildMetricCard(
        doc,
        '1% Low FPS',
        `${ROOT_CLASS}__one-percent-low-fps`,
        PLACEHOLDER_DASH
      );
      const p99FrameMsMetric = this._buildMetricCard(
        doc,
        'P99 Frame',
        `${ROOT_CLASS}__p99-frame-ms`,
        PLACEHOLDER_DASH
      );
      const zeroCopyUploadsMetric = this._buildMetricCard(
        doc,
        '4K Zero-Copy',
        `${ROOT_CLASS}__zero-copy-uploads`,
        '0'
      );
      const elidedStateCallsMetric = this._buildMetricCard(
        doc,
        'State Elided',
        `${ROOT_CLASS}__elided-state-calls`,
        '0'
      );
      const bindGroupCacheHitsMetric = this._buildMetricCard(
        doc,
        'BG Cache Hits',
        `${ROOT_CLASS}__bind-group-cache-hits`,
        '0'
      );
      const topMissingCountMetric = this._buildMetricCard(
        doc,
        'Top Missing',
        `${ROOT_CLASS}__top-missing-count`,
        '0'
      );
      const topMissingShadersMetric = this._buildMetricCard(
        doc,
        'Top Missing Shader Hashes',
        `${ROOT_CLASS}__top-missing-shaders`,
        PLACEHOLDER_DASH
      );
      topMissingShadersMetric.card.className = `${ROOT_CLASS}__metric ${ROOT_CLASS}__metric--wide`;

      perfGrid.appendChild(onePercentLowFpsMetric.card);
      perfGrid.appendChild(p99FrameMsMetric.card);
      perfGrid.appendChild(zeroCopyUploadsMetric.card);
      perfGrid.appendChild(elidedStateCallsMetric.card);
      perfGrid.appendChild(bindGroupCacheHitsMetric.card);
      perfGrid.appendChild(topMissingCountMetric.card);
      perfGrid.appendChild(topMissingShadersMetric.card);
      perfSection.appendChild(perfGrid);

      // --- Active WebGPU adapter, visible-frame readiness & S44 shader reflection section (W240) ---
      const adapterReflectionSection = createEl(
        doc,
        'div',
        `${ROOT_CLASS}__section ${ROOT_CLASS}__adapter-reflection`,
        {
          role: 'group',
          'aria-label': 'Adapter, Visibility & Shader Reflection',
        }
      );
      adapterReflectionSection.appendChild(
        createEl(doc, 'div', `${ROOT_CLASS}__section-title`, {
          text: 'Adapter, Visibility & Shader Reflection',
        })
      );
      const adapterReflectionGrid = createEl(doc, 'div', `${ROOT_CLASS}__grid`);
      const gpuAdapterMetric = this._buildMetricCard(
        doc,
        'GPU Adapter',
        `${ROOT_CLASS}__gpu-adapter`,
        PLACEHOLDER_DASH
      );
      gpuAdapterMetric.card.className = `${ROOT_CLASS}__metric ${ROOT_CLASS}__metric--wide`;
      const firstVisibleFrameMsMetric = this._buildMetricCard(
        doc,
        'First Visible',
        `${ROOT_CLASS}__first-visible-frame-ms`,
        PLACEHOLDER_DASH
      );
      const blackLoadingMsMetric = this._buildMetricCard(
        doc,
        'Black Loading',
        `${ROOT_CLASS}__black-loading-ms`,
        PLACEHOLDER_DASH
      );
      const blackPresentsTotalMetric = this._buildMetricCard(
        doc,
        'Black Presents',
        `${ROOT_CLASS}__black-presents-total`,
        '0'
      );
      const reflectedStagesMetric = this._buildMetricCard(
        doc,
        'Reflected Stages',
        `${ROOT_CLASS}__reflected-stages`,
        '0'
      );
      const inlineSamplerStagesMetric = this._buildMetricCard(
        doc,
        'Inline Samplers',
        `${ROOT_CLASS}__inline-sampler-stages`,
        '0'
      );
      const resourceHeapBindingsMetric = this._buildMetricCard(
        doc,
        'Heap Bindings',
        `${ROOT_CLASS}__resource-heap-bindings`,
        '0'
      );

      adapterReflectionGrid.appendChild(gpuAdapterMetric.card);
      adapterReflectionGrid.appendChild(firstVisibleFrameMsMetric.card);
      adapterReflectionGrid.appendChild(blackLoadingMsMetric.card);
      adapterReflectionGrid.appendChild(blackPresentsTotalMetric.card);
      adapterReflectionGrid.appendChild(reflectedStagesMetric.card);
      adapterReflectionGrid.appendChild(inlineSamplerStagesMetric.card);
      adapterReflectionGrid.appendChild(resourceHeapBindingsMetric.card);
      adapterReflectionSection.appendChild(adapterReflectionGrid);

      // --- Missing-shader table section ---
      const missingSection = createEl(doc, 'div', `${ROOT_CLASS}__section ${ROOT_CLASS}__missing`);
      const missingHeader = createEl(
        doc,
        'div',
        `${ROOT_CLASS}__section-title ${ROOT_CLASS}__missing-header`,
        { text: 'Missing Shaders (0)' }
      );
      const table = createEl(doc, 'table', `${ROOT_CLASS}__table`, {
        'aria-label': 'Missing shaders',
      });
      const thead = createEl(doc, 'thead');
      const headRow = createEl(doc, 'tr');
      headRow.appendChild(createEl(doc, 'th', null, { scope: 'col', text: 'Hash' }));
      headRow.appendChild(createEl(doc, 'th', null, { scope: 'col', text: 'Stage' }));
      headRow.appendChild(
        createEl(doc, 'th', `${ROOT_CLASS}__cell-hits`, { scope: 'col', text: 'Hits' })
      );
      thead.appendChild(headRow);
      const tbody = createEl(doc, 'tbody', `${ROOT_CLASS}__table-body`);
      table.appendChild(thead);
      table.appendChild(tbody);

      const overflowEl = createEl(doc, 'div', `${ROOT_CLASS}__overflow`);
      overflowEl.hidden = true;

      missingSection.appendChild(missingHeader);
      missingSection.appendChild(table);
      missingSection.appendChild(overflowEl);

      body.appendChild(bootSection);
      body.appendChild(countersSection);
      body.appendChild(compatSection);
      body.appendChild(ioSection);
      body.appendChild(perfSection);
      body.appendChild(adapterReflectionSection);
      body.appendChild(missingSection);

      rootEl.appendChild(header);
      rootEl.appendChild(body);

      this._els = {
        root: rootEl,
        header,
        summaryBadge,
        exportBtn,
        toggleBtn,
        body,
        bootStage: bootStageMetric.val,
        bootMilestones: bootMilestonesMetric.val,
        bootReady: bootReadyMetric.val,
        bootProgress,
        progressBar,
        progressFill,
        framesPresented: framesPresentedMetric.val,
        droppedFrames: droppedFramesMetric.val,
        packHits: packHitsMetric.val,
        fallbackHits: fallbackHitsMetric.val,
        builtinHits: builtinHitsMetric.val,
        hitRate: hitRateMetric.val,
        bindGroupKindMismatches: bindMismatchesMetric.val,
        autoLayoutFallbacks: autoLayoutMetric.val,
        bc3dFallbacks: bc3dFallbacksMetric.val,
        bc3dFallbackUploadsSkipped: bc3dSkippedMetric.val,
        swapchainFormat: swapchainFormatMetric.val,
        blitFallback: blitFallbackMetric.val,
        pgimFullFetches: pgimFetchesMetric.val,
        slabCacheHits: slabHitsMetric.val,
        slabCacheMisses: slabMissesMetric.val,
        syncFastPathHits: syncFastHitsMetric.val,
        onePercentLowFps: onePercentLowFpsMetric.val,
        p99FrameMs: p99FrameMsMetric.val,
        zeroCopyUploads: zeroCopyUploadsMetric.val,
        totalElidedStateCalls: elidedStateCallsMetric.val,
        bindGroupCacheHits: bindGroupCacheHitsMetric.val,
        topMissingCount: topMissingCountMetric.val,
        topMissingShaders: topMissingShadersMetric.val,
        gpuAdapter: gpuAdapterMetric.val,
        firstVisibleFrameMs: firstVisibleFrameMsMetric.val,
        blackLoadingMs: blackLoadingMsMetric.val,
        blackPresentsTotal: blackPresentsTotalMetric.val,
        reflectedStages: reflectedStagesMetric.val,
        inlineSamplerStages: inlineSamplerStagesMetric.val,
        resourceHeapBindings: resourceHeapBindingsMetric.val,
        missingHeader,
        table,
        tbody,
        overflowEl,
      };

      return rootEl;
    }

    _bindListeners() {
      if (!this._els && this._root && typeof this._root.querySelector === 'function') {
        const q = (sel) => {
          try {
            return this._root.querySelector(sel);
          } catch {
            return null;
          }
        };
        this._els = {
          root: this._root,
          summaryBadge: q(`.${ROOT_CLASS}__summary-badge`),
          exportBtn: q(`.${ROOT_CLASS}__export-btn`),
          toggleBtn: q(`.${ROOT_CLASS}__toggle-btn`),
          body: q(`.${ROOT_CLASS}__body`),
          bootStage: q(`.${ROOT_CLASS}__boot-stage`),
          bootMilestones: q(`.${ROOT_CLASS}__boot-milestones`),
          bootReady: q(`.${ROOT_CLASS}__boot-ready`),
          bootProgress: q(`.${ROOT_CLASS}__boot-progress`),
          progressBar: q(`.${ROOT_CLASS}__progress-bar`),
          progressFill: q(`.${ROOT_CLASS}__progress-fill`),
          framesPresented: q(`.${ROOT_CLASS}__frames-presented`),
          droppedFrames: q(`.${ROOT_CLASS}__dropped-frames`),
          packHits: q(`.${ROOT_CLASS}__pack-hits`),
          fallbackHits: q(`.${ROOT_CLASS}__fallback-hits`),
          builtinHits: q(`.${ROOT_CLASS}__builtin-hits`),
          hitRate: q(`.${ROOT_CLASS}__hit-rate`),
          bindGroupKindMismatches: q(`.${ROOT_CLASS}__bind-mismatches`),
          autoLayoutFallbacks: q(`.${ROOT_CLASS}__auto-layout-fallbacks`),
          bc3dFallbacks: q(`.${ROOT_CLASS}__bc3d-fallbacks`),
          bc3dFallbackUploadsSkipped: q(`.${ROOT_CLASS}__bc3d-uploads-skipped`),
          swapchainFormat: q(`.${ROOT_CLASS}__swapchain-format`),
          blitFallback: q(`.${ROOT_CLASS}__blit-fallback`),
          pgimFullFetches: q(`.${ROOT_CLASS}__pgim-fetches`),
          slabCacheHits: q(`.${ROOT_CLASS}__slab-hits`),
          slabCacheMisses: q(`.${ROOT_CLASS}__slab-misses`),
          syncFastPathHits: q(`.${ROOT_CLASS}__sync-fast-hits`),
          onePercentLowFps: q(`.${ROOT_CLASS}__one-percent-low-fps`),
          p99FrameMs: q(`.${ROOT_CLASS}__p99-frame-ms`),
          zeroCopyUploads: q(`.${ROOT_CLASS}__zero-copy-uploads`),
          totalElidedStateCalls: q(`.${ROOT_CLASS}__elided-state-calls`),
          bindGroupCacheHits: q(`.${ROOT_CLASS}__bind-group-cache-hits`),
          topMissingCount: q(`.${ROOT_CLASS}__top-missing-count`),
          topMissingShaders: q(`.${ROOT_CLASS}__top-missing-shaders`),
          gpuAdapter: q(`.${ROOT_CLASS}__gpu-adapter`),
          firstVisibleFrameMs: q(`.${ROOT_CLASS}__first-visible-frame-ms`),
          blackLoadingMs: q(`.${ROOT_CLASS}__black-loading-ms`),
          blackPresentsTotal: q(`.${ROOT_CLASS}__black-presents-total`),
          reflectedStages: q(`.${ROOT_CLASS}__reflected-stages`),
          inlineSamplerStages: q(`.${ROOT_CLASS}__inline-sampler-stages`),
          resourceHeapBindings: q(`.${ROOT_CLASS}__resource-heap-bindings`),
          missingHeader: q(`.${ROOT_CLASS}__missing-header`),
          table: q(`.${ROOT_CLASS}__table`),
          tbody: q(`.${ROOT_CLASS}__table-body`),
          overflowEl: q(`.${ROOT_CLASS}__overflow`),
        };
      }

      const els = this._els || {};
      this._onToggleClick = () => {
        if (!this._destroyed) {
          this.toggle();
        }
      };
      this._onExportClick = () => {
        if (!this._destroyed) {
          this.exportMissingShadersJson();
        }
      };

      if (els.toggleBtn) {
        try {
          if (typeof els.toggleBtn.addEventListener === 'function') {
            els.toggleBtn.addEventListener('click', this._onToggleClick);
          } else {
            els.toggleBtn.onclick = this._onToggleClick;
          }
        } catch {
          // Ignore listener errors on lightweight mocks.
        }
      }

      if (els.exportBtn) {
        try {
          if (typeof els.exportBtn.addEventListener === 'function') {
            els.exportBtn.addEventListener('click', this._onExportClick);
          } else {
            els.exportBtn.onclick = this._onExportClick;
          }
        } catch {
          // Ignore listener errors on lightweight mocks.
        }
      }
    }

    _applyVisibility(visible) {
      this._visible = Boolean(visible);
      const rootEl = this._root;
      if (!rootEl) return;
      try {
        rootEl.hidden = !this._visible;
        if (rootEl.style) {
          rootEl.style.display = this._visible ? '' : 'none';
        }
      } catch {
        // Visibility updates must never throw.
      }
    }

    _applyCollapsed(collapsed) {
      this._collapsed = Boolean(collapsed);
      const els = this._els || {};
      try {
        if (this._root && typeof this._root.setAttribute === 'function') {
          this._root.setAttribute('data-collapsed', String(this._collapsed));
        }
        if (els.body) {
          els.body.hidden = this._collapsed;
          if (els.body.style) {
            els.body.style.display = this._collapsed ? 'none' : '';
          }
        }
        if (els.toggleBtn) {
          if (typeof els.toggleBtn.setAttribute === 'function') {
            els.toggleBtn.setAttribute('aria-expanded', String(!this._collapsed));
          }
          els.toggleBtn.textContent = this._collapsed ? 'Expand' : 'Collapse';
        }
      } catch {
        // Collapse updates must never throw.
      }
    }

    _setTextIfChanged(cacheKey, el, nextText) {
      if (!el) return false;
      const str = String(nextText);
      if (this._renderedCache[cacheKey] === str) {
        return false;
      }
      el.textContent = str;
      this._renderedCache[cacheKey] = str;
      return true;
    }

    _setAttrIfChanged(cacheKey, el, attrName, nextValue) {
      if (!el || typeof el.setAttribute !== 'function') return false;
      const str = String(nextValue);
      if (this._renderedCache[cacheKey] === str) {
        return false;
      }
      el.setAttribute(attrName, str);
      this._renderedCache[cacheKey] = str;
      return true;
    }

    _setStylePropIfChanged(cacheKey, el, prop, nextValue) {
      if (!el || !el.style) return false;
      const str = String(nextValue);
      if (this._renderedCache[cacheKey] === str) {
        return false;
      }
      el.style[prop] = str;
      this._renderedCache[cacheKey] = str;
      return true;
    }

    _render() {
      const els = this._els;
      if (!els) return;

      const boot = this._boot;
      const frames = this._frameStats;
      const fb = this._fallbackStats;
      const compat = this._compatStats;
      const io = this._ioStats;
      const perf = this._perfStats;
      const adapter = this._adapterStats;
      const visibility = this._visibilityStats;
      const reflection = this._reflectionStats;

      try {
        this._setTextIfChanged('summaryBadge', els.summaryBadge, `${boot.stage} · ${boot.progressPct}%`);
        this._setTextIfChanged('bootStage', els.bootStage, boot.stage);
        this._setTextIfChanged(
          'bootMilestones',
          els.bootMilestones,
          `${boot.milestonesReached}/${boot.totalMilestones}`
        );
        this._setTextIfChanged('bootProgress', els.bootProgress, `${boot.progressPct}%`);
        this._setTextIfChanged('bootReady', els.bootReady, boot.readyForSplash ? 'Ready' : 'Pending');
        this._setStylePropIfChanged('progressFill.width', els.progressFill, 'width', `${boot.progressPct}%`);
        this._setAttrIfChanged(
          'progressBar.aria-valuenow',
          els.progressBar,
          'aria-valuenow',
          boot.progressPct
        );

        this._setTextIfChanged('framesPresented', els.framesPresented, frames.framesPresented);
        this._setTextIfChanged('droppedFrames', els.droppedFrames, frames.droppedFrames);
        this._setTextIfChanged('packHits', els.packHits, fb.packHits);
        this._setTextIfChanged('fallbackHits', els.fallbackHits, fb.fallbackHits);
        this._setTextIfChanged('builtinHits', els.builtinHits, fb.builtinHits);
        this._setTextIfChanged('hitRate', els.hitRate, formatHitRatePct(fb.hitRatePct));

        this._setTextIfChanged(
          'bindGroupKindMismatches',
          els.bindGroupKindMismatches,
          compat.bindGroupKindMismatches
        );
        this._setTextIfChanged(
          'autoLayoutFallbacks',
          els.autoLayoutFallbacks,
          compat.autoLayoutFallbacks
        );
        this._setTextIfChanged('bc3dFallbacks', els.bc3dFallbacks, compat.bc3dFallbacks);
        this._setTextIfChanged(
          'bc3dFallbackUploadsSkipped',
          els.bc3dFallbackUploadsSkipped,
          compat.bc3dFallbackUploadsSkipped
        );
        this._setTextIfChanged(
          'swapchainFormat',
          els.swapchainFormat,
          compat.swapchainFormat || PLACEHOLDER_DASH
        );
        if (els.blitFallback) {
          this._setTextIfChanged(
            'blitFallback',
            els.blitFallback,
            formatBlitFallbackText(compat.usesBlitFallback, compat.blitCount)
          );
          this._setAttrIfChanged(
            'blitFallback.data-uses-blit',
            els.blitFallback,
            'data-uses-blit',
            compat.usesBlitFallback
          );
          this._setAttrIfChanged(
            'blitFallback.data-blit-count',
            els.blitFallback,
            'data-blit-count',
            compat.blitCount
          );
        }

        this._setTextIfChanged('pgimFullFetches', els.pgimFullFetches, io.pgimFullFetches);
        this._setTextIfChanged('slabCacheHits', els.slabCacheHits, io.slabCacheHits);
        this._setTextIfChanged('slabCacheMisses', els.slabCacheMisses, io.slabCacheMisses);
        this._setTextIfChanged('syncFastPathHits', els.syncFastPathHits, io.syncFastPathHits);

        // W212: Performance, 4K Bink2 zero-copy, state elision, and top missing shader hashes.
        this._setTextIfChanged(
          'onePercentLowFps',
          els.onePercentLowFps,
          formatOnePercentLowFps(perf.onePercentLowFps)
        );
        this._setTextIfChanged(
          'p99FrameMs',
          els.p99FrameMs,
          formatP99FrameMs(perf.p99FrameMs)
        );
        this._setTextIfChanged('zeroCopyUploads', els.zeroCopyUploads, perf.zeroCopyUploads);
        this._setTextIfChanged(
          'totalElidedStateCalls',
          els.totalElidedStateCalls,
          perf.totalElidedStateCalls
        );
        this._setTextIfChanged(
          'bindGroupCacheHits',
          els.bindGroupCacheHits,
          perf.bindGroupCacheHits
        );
        this._setTextIfChanged(
          'topMissingCount',
          els.topMissingCount,
          perf.topMissingShaders.length
        );
        this._setTextIfChanged(
          'topMissingShaders',
          els.topMissingShaders,
          formatTopMissingShaders(
            perf.topMissingShaders,
            perf.topMissingShaderOverflow,
            this._maxTopMissingShaders
          )
        );

        // W240: Active WebGPU adapter, visible-frame readiness & S44 shader reflection breakdown.
        this._setTextIfChanged(
          'gpuAdapter',
          els.gpuAdapter,
          adapter.gpuAdapter || PLACEHOLDER_DASH
        );
        this._setTextIfChanged(
          'firstVisibleFrameMs',
          els.firstVisibleFrameMs,
          formatDurationMs(visibility.firstVisibleFrameMs)
        );
        this._setTextIfChanged(
          'blackLoadingMs',
          els.blackLoadingMs,
          formatDurationMs(visibility.blackLoadingMs)
        );
        this._setTextIfChanged(
          'blackPresentsTotal',
          els.blackPresentsTotal,
          visibility.blackPresentsTotal
        );
        this._setTextIfChanged(
          'reflectedStages',
          els.reflectedStages,
          reflection.reflectedStages
        );
        this._setTextIfChanged(
          'inlineSamplerStages',
          els.inlineSamplerStages,
          reflection.inlineSamplerStages
        );
        this._setTextIfChanged(
          'resourceHeapBindings',
          els.resourceHeapBindings,
          reflection.resourceHeapBindings
        );

        const totalMissing = this._missingShaders.length;
        const visibleRows = this._missingShaders.slice(0, this._maxRows);
        const overflowCount = Math.max(0, totalMissing - visibleRows.length);

        this._setTextIfChanged('missingHeader', els.missingHeader, `Missing Shaders (${totalMissing})`);

        if (els.tbody) {
          const tableFingerprint =
            visibleRows.length === 0
              ? '__EMPTY__'
              : visibleRows.map((row) => `${row.hash}|${row.stage}|${row.hits}`).join('\n');
          if (this._renderedCache.tbodyFingerprint !== tableFingerprint) {
            clearElementChildren(els.tbody);
            const doc = this._document;
            if (visibleRows.length === 0) {
              const emptyRow = createEl(doc, 'tr', `${ROOT_CLASS}__empty-row`);
              const emptyCell = createEl(doc, 'td', `${ROOT_CLASS}__empty-cell`, {
                colspan: '3',
                text: 'No missing shaders recorded',
              });
              emptyRow.appendChild(emptyCell);
              els.tbody.appendChild(emptyRow);
            } else {
              for (let i = 0; i < visibleRows.length; i += 1) {
                const row = visibleRows[i];
                const tr = createEl(doc, 'tr', `${ROOT_CLASS}__shader-row`);
                tr.appendChild(
                  createEl(doc, 'td', `${ROOT_CLASS}__cell-hash`, { text: row.hash })
                );
                tr.appendChild(
                  createEl(doc, 'td', `${ROOT_CLASS}__cell-stage`, { text: row.stage })
                );
                tr.appendChild(
                  createEl(doc, 'td', `${ROOT_CLASS}__cell-hits`, { text: String(row.hits) })
                );
                els.tbody.appendChild(tr);
              }
            }
            this._renderedCache.tbodyFingerprint = tableFingerprint;
          }
        }

        if (els.overflowEl && this._renderedCache.overflowCount !== overflowCount) {
          if (overflowCount > 0) {
            els.overflowEl.hidden = false;
            if (els.overflowEl.style) els.overflowEl.style.display = '';
            els.overflowEl.textContent = `+${overflowCount} more`;
          } else {
            els.overflowEl.hidden = true;
            if (els.overflowEl.style) els.overflowEl.style.display = 'none';
            els.overflowEl.textContent = '';
          }
          this._renderedCache.overflowCount = overflowCount;
        }
      } catch {
        // Render errors on partial/failing DOM mocks must never escape.
      }
    }

    /**
     * Apply partial or full telemetry snapshots and refresh the diagnostics panel.
     *
     * @param {{
     *   bootSummary?: object|null,
     *   missingShaders?: Array|null,
     *   fallbackStats?: object|null,
     *   frameStats?: object|null,
     *   compatStats?: object|null,
     *   renderStateStats?: object|null,
     *   textureStats?: object|null,
     *   swapchainStats?: object|null,
     *   ioStats?: object|null,
     *   bindGroupKindMismatches?: number,
     *   autoLayoutFallbacks?: number,
     *   bc3dFallbacks?: number,
     *   bc3dFallbackUploadsSkipped?: number,
     *   swapchainFormat?: string|null,
     *   usesBlitFallback?: boolean,
     *   blitCount?: number,
     *   pgimFullFetches?: number,
     *   slabCacheHits?: number,
     *   slabCacheMisses?: number,
     *   syncFastPathHits?: number,
     * }} [snapshot]
     * @returns {object} Current controller state.
     */
    update(snapshot) {
      if (this._destroyed) {
        return this.getState();
      }
      if (!snapshot || typeof snapshot !== 'object') {
        return this.getState();
      }

      const bootSummary = safeRead(snapshot, 'bootSummary');
      if (bootSummary !== undefined) {
        this._boot = normalizeBootSummary(bootSummary);
      }

      const missingShaders = safeRead(snapshot, 'missingShaders');
      if (missingShaders !== undefined) {
        this._missingShaders = deduplicateMissingShaders(missingShaders);
      }

      const fallbackStats = safeRead(snapshot, 'fallbackStats');
      if (fallbackStats !== undefined) {
        this._fallbackStats = normalizeFallbackStats(fallbackStats);
      }

      const frameStats = safeRead(snapshot, 'frameStats');
      if (frameStats !== undefined) {
        this._frameStats = normalizeFrameStats(frameStats);
      }

      this._compatStats = extractCompatStats(snapshot, this._compatStats);
      this._ioStats = extractIoStats(snapshot, this._ioStats);
      this._perfStats = extractPerfStats(snapshot, this._perfStats, this._maxTopMissingShaders);
      this._adapterStats = extractAdapterStats(snapshot, this._adapterStats);
      this._visibilityStats = extractVisibilityStats(snapshot, this._visibilityStats);
      this._reflectionStats = extractReflectionStats(snapshot, this._reflectionStats);

      this._render();
      return this.getState();
    }

    /**
     * Reset all telemetry counters, adapter/visibility/reflection diagnostics, and
     * missing-shader rows back to clean initial defaults while preserving mounted DOM.
     *
     * @returns {object} Current controller state after reset.
     */
    reset() {
      if (this._destroyed) {
        return this.getState();
      }
      this._boot = normalizeBootSummary(null);
      this._frameStats = normalizeFrameStats(null);
      this._fallbackStats = normalizeFallbackStats(null);
      this._compatStats = createDefaultCompatStats();
      this._ioStats = createDefaultIoStats();
      this._perfStats = createDefaultPerfStats();
      this._adapterStats = createDefaultAdapterStats();
      this._visibilityStats = createDefaultVisibilityStats();
      this._reflectionStats = createDefaultReflectionStats();
      this._missingShaders = [];
      this._exportCount = 0;

      this._render();
      return this.getState();
    }

    /**
     * Build and return a sanitized JSON string of the current missing-shader shopping list,
     * invoking `onExportMissingShaders(jsonText, payload)` if configured.
     *
     * @returns {string}
     */
    exportMissingShadersJson() {
      const payload = buildMissingShaderExport({
        bootSummary: this._boot,
        missingShaders: this._missingShaders,
        fallbackStats: this._fallbackStats,
        frameStats: this._frameStats,
        compatStats: this._compatStats,
        ioStats: this._ioStats,
        perfStats: this._perfStats,
        adapterStats: this._adapterStats,
        visibilityStats: this._visibilityStats,
        reflectionStats: this._reflectionStats,
        maxTopMissingShaders: this._maxTopMissingShaders,
        now: this._now,
      });
      const jsonText = JSON.stringify(payload, null, 2);

      if (!this._destroyed) {
        this._exportCount += 1;
        if (typeof this._onExportMissingShaders === 'function') {
          try {
            this._onExportMissingShaders(jsonText, payload);
          } catch {
            // Isolate callback errors so diagnostics export never breaks the caller.
          }
        }
      }

      return jsonText;
    }

    /**
     * Toggle the collapsible diagnostics body (or set collapsed explicitly when given a boolean).
     *
     * @param {boolean} [forceCollapsed]
     * @returns {object}
     */
    toggle(forceCollapsed) {
      if (this._destroyed) {
        return this.getState();
      }
      const next = typeof forceCollapsed === 'boolean' ? forceCollapsed : !this._collapsed;
      this._applyCollapsed(next);
      return this.getState();
    }

    /** Show the panel root element. */
    show() {
      if (this._destroyed) {
        return this.getState();
      }
      this._applyVisibility(true);
      return this.getState();
    }

    /** Hide the panel root element. */
    hide() {
      if (this._destroyed) {
        return this.getState();
      }
      this._applyVisibility(false);
      return this.getState();
    }

    /** @returns {object} Snapshot of current panel state. */
    getState() {
      const visibleMissingShaders = this._missingShaders
        .slice(0, this._maxRows)
        .map((row) => ({ ...row }));
      const totalMissingShaders = this._missingShaders.length;
      return {
        visible: this._visible,
        collapsed: this._collapsed,
        destroyed: this._destroyed,
        maxRows: this._maxRows,
        boot: { ...this._boot },
        frameStats: { ...this._frameStats },
        fallbackStats: { ...this._fallbackStats },
        compatStats: { ...this._compatStats },
        ioStats: { ...this._ioStats },
        perfStats: {
          onePercentLowFps: this._perfStats.onePercentLowFps,
          p99FrameMs: this._perfStats.p99FrameMs,
          zeroCopyUploads: this._perfStats.zeroCopyUploads,
          totalElidedStateCalls: this._perfStats.totalElidedStateCalls,
          bindGroupCacheHits: this._perfStats.bindGroupCacheHits,
          topMissingShaders: this._perfStats.topMissingShaders.map((row) => ({ ...row })),
          topMissingShaderOverflow: this._perfStats.topMissingShaderOverflow,
        },
        gpuAdapter: this._adapterStats.gpuAdapter,
        firstVisibleFrameMs: this._visibilityStats.firstVisibleFrameMs,
        blackLoadingMs: this._visibilityStats.blackLoadingMs,
        blackPresentsTotal: this._visibilityStats.blackPresentsTotal,
        reflectedStages: this._reflectionStats.reflectedStages,
        inlineSamplerStages: this._reflectionStats.inlineSamplerStages,
        resourceHeapBindings: this._reflectionStats.resourceHeapBindings,
        visibilityStats: { ...this._visibilityStats },
        reflectionStats: { ...this._reflectionStats },
        hitRateText: formatHitRatePct(this._fallbackStats.hitRatePct),
        missingShaders: this._missingShaders.map((row) => ({ ...row })),
        visibleMissingShaders,
        totalMissingShaders,
        overflowCount: Math.max(0, totalMissingShaders - visibleMissingShaders.length),
        exportCount: this._exportCount,
        root: this._root,
      };
    }

    /** @returns {object} Snapshot of current panel state (alias of `getState()`). */
    getSnapshot() {
      return this.getState();
    }

    /** Tear down DOM event listeners and detach the panel node. Idempotent. */
    destroy() {
      if (this._destroyed) {
        return;
      }
      this._destroyed = true;

      const els = this._els || {};
      if (els.toggleBtn && this._onToggleClick) {
        try {
          if (typeof els.toggleBtn.removeEventListener === 'function') {
            els.toggleBtn.removeEventListener('click', this._onToggleClick);
          } else {
            els.toggleBtn.onclick = null;
          }
        } catch {
          // ignore
        }
      }
      if (els.exportBtn && this._onExportClick) {
        try {
          if (typeof els.exportBtn.removeEventListener === 'function') {
            els.exportBtn.removeEventListener('click', this._onExportClick);
          } else {
            els.exportBtn.onclick = null;
          }
        } catch {
          // ignore
        }
      }

      if (
        this._root &&
        this._root.parentNode &&
        typeof this._root.parentNode.removeChild === 'function'
      ) {
        try {
          this._root.parentNode.removeChild(this._root);
        } catch {
          // ignore
        }
      }

      this._visible = false;
      this._els = null;
      this._root = null;
      this._renderedCache = Object.create(null);
      this._onExportMissingShaders = null;
      this._onToggleClick = null;
      this._onExportClick = null;
    }
  }

  const exportedApi = Object.freeze({
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_TOP_MISSING_SHADERS,
    MAX_GPU_ADAPTER_LENGTH,
    EXPORT_FORMAT,
    EXPORT_SCHEMA_VERSION,
    BOOT_STAGE_ORDER,
    STAGE_NAMES,
    ALLOWED_SWAPCHAIN_FORMATS,
    sanitizeSwapchainFormat,
    sanitizeGpuAdapter,
    formatDurationMs,
    formatOnePercentLowFps,
    formatP99FrameMs,
    normalizeTopMissingShaders,
    formatTopMissingShaders,
    formatMissingShaderRow,
    buildMissingShaderExport,
    GpuBootPanelController,
    GpuBootPanel: GpuBootPanelController,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6GpuBootPanel = exportedApi;
  } else if (root) {
    root.PlayGta6GpuBootPanel = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
