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

  const ALLOWED_SWAPCHAIN_COLOR_MODES = Object.freeze([
    'sdr-srgb',
    'sdr-linear',
    'sdr-tonemap',
    'sdr-passthrough',
    'direct-srgb',
    'srgb-encode',
    'hdr-linear',
    'hdr-tonemap',
    'hdr-tonemap-aces',
    'hdr-tonemap-reinhard',
    'hdr-scrgb',
    'hdr-pq',
  ]);
  const ALLOWED_SWAPCHAIN_COLOR_MODE_SET = new Set(ALLOWED_SWAPCHAIN_COLOR_MODES);
  const ALLOWED_TONE_MAP_MODE_SET = new Set(['none', 'reinhard', 'aces']);

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

  /** Post-present M4 boot milestones (mirrors BootTracer OPTIONAL + FAST_BOOT_MILESTONES). */
  const POST_PRESENT_BOOT_STAGES = Object.freeze([
    'FIRST_VISIBLE_FRAME',
    'START_SCREEN',
    'INTRO_VIDEO',
    'MAIN_MENU',
    'IN_GAME',
  ]);

  /** Complete ordered list of core and post-present M4 boot milestones. */
  const ALL_BOOT_STAGE_ORDER = Object.freeze([...BOOT_STAGE_ORDER, ...POST_PRESENT_BOOT_STAGES]);

  const BOOT_STAGE_SET = new Set(BOOT_STAGE_ORDER);
  const POST_PRESENT_STAGE_SET = new Set(POST_PRESENT_BOOT_STAGES);
  const ALL_BOOT_STAGE_SET = new Set(ALL_BOOT_STAGE_ORDER);

  const PLAYABILITY_STATE = Object.freeze({
    BOOTING: 'BOOTING',
    FIRST_PRESENT: 'FIRST_PRESENT',
    START_SCREEN: 'START_SCREEN',
    INTRO_VIDEO: 'INTRO_VIDEO',
    MAIN_MENU: 'MAIN_MENU',
    IN_GAME: 'IN_GAME',
  });

  const ALLOWED_AUDIO_OUTPUT_STATES = Object.freeze(['active', 'suspended', 'headless-drain']);
  const ALLOWED_AUDIO_OUTPUT_STATE_SET = new Set(ALLOWED_AUDIO_OUTPUT_STATES);

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
    if (upper === STAGE_NOT_STARTED || ALL_BOOT_STAGE_SET.has(upper)) {
      return upper;
    }
    // Reject anything containing path separators, colons, dots, spaces, or query chars.
    if (/^[A-Z][A-Z0-9_]{0,63}$/.test(upper)) {
      return upper;
    }
    return 'UNKNOWN';
  }

  function resolvePlayabilityState(stage) {
    if (stage === 'IN_GAME') return PLAYABILITY_STATE.IN_GAME;
    if (stage === 'MAIN_MENU') return PLAYABILITY_STATE.MAIN_MENU;
    if (stage === 'INTRO_VIDEO') return PLAYABILITY_STATE.INTRO_VIDEO;
    if (stage === 'START_SCREEN') return PLAYABILITY_STATE.START_SCREEN;
    if (stage === 'FIRST_VISIBLE_FRAME' || stage === 'FIRST_PRESENT') {
      return PLAYABILITY_STATE.FIRST_PRESENT;
    }
    return PLAYABILITY_STATE.BOOTING;
  }

  function normalizeBootSummary(raw) {
    if (!raw || typeof raw !== 'object') {
      return {
        stage: STAGE_NOT_STARTED,
        playabilityState: PLAYABILITY_STATE.BOOTING,
        milestonesReached: 0,
        totalMilestones: BOOT_STAGE_ORDER.length,
        progressPct: 0,
        readyForSplash: false,
      };
    }

    const stage = sanitizeBootStage(safeRead(raw, 'stage'));
    const stageIdx = BOOT_STAGE_ORDER.indexOf(stage);
    const isPostPresent = POST_PRESENT_STAGE_SET.has(stage);

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
    } else if (isPostPresent) {
      milestonesReached = BOOT_STAGE_ORDER.length;
    } else {
      milestonesReached = 0;
    }

    const rawProgress = safeRead(raw, 'progressPct');
    let progressPct;
    if (rawProgress !== undefined && rawProgress !== null) {
      progressPct = Math.round(clampPct(rawProgress));
    } else if (isPostPresent) {
      progressPct = 100;
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
      isPostPresent;

    return {
      stage,
      playabilityState: resolvePlayabilityState(stage),
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

  function sanitizeToneMapMode(rawMode) {
    if (typeof rawMode !== 'string') {
      return null;
    }
    const token = rawMode.trim().toLowerCase();
    return ALLOWED_TONE_MAP_MODE_SET.has(token) ? token : null;
  }

  /**
   * Sanitize a swapchain color presentation mode against a strict allowlist
   * (or derive a canonical mode from toneMapMode / isHdrStream / applySrgbEncode / swapchainFormat).
   *
   * @param {unknown} rawMode
   * @param {{ toneMapMode?: string|null, isHdrStream?: boolean, toneMapActive?: boolean, applySrgbEncode?: boolean|null, swapchainFormat?: string|null }} [ctx]
   * @returns {string|null}
   */
  function sanitizeSwapchainColorMode(rawMode, ctx) {
    if (typeof rawMode === 'string') {
      const token = rawMode
        .trim()
        .toLowerCase()
        .replace(/[\s_]+/g, '-');
      if (token.length > 0 && token.length <= 32 && ALLOWED_SWAPCHAIN_COLOR_MODE_SET.has(token)) {
        return token;
      }
    }
    if (ctx && typeof ctx === 'object') {
      const toneMapMode = sanitizeToneMapMode(ctx.toneMapMode);
      const isHdr = Boolean(ctx.isHdrStream);
      const toneMapActive = Boolean(ctx.toneMapActive);
      const applySrgb = typeof ctx.applySrgbEncode === 'boolean' ? ctx.applySrgbEncode : null;
      const fmt = sanitizeSwapchainFormat(ctx.swapchainFormat);
      if (toneMapActive) {
        if (toneMapMode === 'aces') return 'hdr-tonemap-aces';
        if (toneMapMode === 'reinhard') return 'hdr-tonemap-reinhard';
        return 'hdr-tonemap';
      }
      if (isHdr || fmt === 'rgba16float' || fmt === 'rg11b10ufloat' || fmt === 'rgba32float') {
        if (toneMapMode === 'aces') return 'hdr-tonemap-aces';
        if (toneMapMode === 'reinhard') return 'hdr-tonemap-reinhard';
        return 'hdr-linear';
      }
      if (applySrgb === true) {
        return 'srgb-encode';
      }
      if (fmt && fmt.endsWith('-srgb')) {
        return 'sdr-srgb';
      }
      if (fmt && (fmt === 'bgra8unorm' || fmt === 'rgba8unorm' || fmt === 'rgb10a2unorm')) {
        return 'sdr-passthrough';
      }
    }
    return null;
  }

  function createDefaultSwapchainColorStats() {
    return {
      swapchainColorMode: null,
      toneMapMode: null,
      isHdrStream: false,
      toneMapActive: false,
      applySrgbEncode: null,
    };
  }

  function extractSwapchainColorStats(source, base, compatStats) {
    let next = base ? { ...base } : createDefaultSwapchainColorStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawSwapchain =
      safeRead(source, 'swapchainStats') !== undefined
        ? safeRead(source, 'swapchainStats')
        : safeRead(source, 'swapchain');
    if (rawSwapchain === null) {
      next = createDefaultSwapchainColorStats();
    }
    const swapchainObj = rawSwapchain && typeof rawSwapchain === 'object' ? rawSwapchain : null;
    const rawCompat =
      safeRead(source, 'compatStats') !== undefined
        ? safeRead(source, 'compatStats')
        : safeRead(source, 'gpuCompatStats') !== undefined
          ? safeRead(source, 'gpuCompatStats')
          : safeRead(source, 'gpu');
    const compatObj = rawCompat && typeof rawCompat === 'object' ? rawCompat : null;
    const rawColor = safeRead(source, 'swapchainColorStats');
    const colorObj = rawColor && typeof rawColor === 'object' ? rawColor : null;
    const sources = [source, colorObj, swapchainObj, compatObj];

    const toneMap = pickFirstDefined(sources, ['toneMapMode', 'tone_map_mode']);
    if (toneMap.found) {
      next.toneMapMode = sanitizeToneMapMode(toneMap.value);
    }
    const hdrStream = pickFirstDefined(sources, ['isHdrStream', 'is_hdr_stream']);
    if (hdrStream.found) {
      next.isHdrStream = Boolean(hdrStream.value);
    }
    const toneActive = pickFirstDefined(sources, ['toneMapActive', 'tone_map_active']);
    if (toneActive.found) {
      next.toneMapActive = Boolean(toneActive.value);
    }
    const applySrgb = pickFirstDefined(sources, ['applySrgbEncode', 'apply_srgb_encode']);
    if (applySrgb.found) {
      next.applySrgbEncode = typeof applySrgb.value === 'boolean' ? applySrgb.value : null;
    }

    const explicitColorMode = pickFirstDefined(sources, [
      'swapchainColorMode',
      'swapchain_color_mode',
      'colorMode',
    ]);
    const swapchainFormat =
      (compatStats && compatStats.swapchainFormat) ||
      sanitizeSwapchainFormat(
        pickFirstDefined(sources, ['swapchainFormat', 'streamFormat', 'format']).value
      );
    if (explicitColorMode.found) {
      if (explicitColorMode.value === null) {
        next.swapchainColorMode = null;
      } else {
        const sanitized = sanitizeSwapchainColorMode(explicitColorMode.value);
        next.swapchainColorMode =
          sanitized !== null
            ? sanitized
            : sanitizeSwapchainColorMode(null, {
                toneMapMode: next.toneMapMode,
                isHdrStream: next.isHdrStream,
                toneMapActive: next.toneMapActive,
                applySrgbEncode: next.applySrgbEncode,
                swapchainFormat,
              });
      }
    } else if (
      toneMap.found ||
      hdrStream.found ||
      toneActive.found ||
      applySrgb.found ||
      swapchainFormat !== null
    ) {
      const derived = sanitizeSwapchainColorMode(null, {
        toneMapMode: next.toneMapMode,
        isHdrStream: next.isHdrStream,
        toneMapActive: next.toneMapActive,
        applySrgbEncode: next.applySrgbEncode,
        swapchainFormat,
      });
      if (derived !== null) {
        next.swapchainColorMode = derived;
      }
    }

    return next;
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
   * Sanitize and clamp a unit-interval ratio (e.g. `fps30ComplianceRate`,
   * `vramBudgetUtilization`, `bindGroupHitRate`) into `[0, 1]`, returning `null`
   * for missing, non-finite, boolean, object, XSS, or host-path values.
   *
   * @param {unknown} value
   * @returns {number|null}
   */
  function sanitizeUnitRatio(value) {
    let n = null;
    if (typeof value === 'number') {
      if (!Number.isFinite(value)) return null;
      n = value;
    } else if (typeof value === 'string') {
      const trimmed = value.trim();
      if (trimmed.length === 0 || trimmed.length > 32) return null;
      if (!/^[+-]?\d+(?:\.\d+)?$/.test(trimmed)) return null;
      const parsed = Number(trimmed);
      if (!Number.isFinite(parsed)) return null;
      n = parsed;
    } else {
      return null;
    }
    if (n <= 0) return 0;
    if (n >= 1) return 1;
    return n;
  }

  /**
   * Sanitize a non-negative finite telemetry scalar, normalizing `-0` to `0` and
   * returning `null` for negative, non-finite, boolean, XSS, or host-path values.
   *
   * @param {unknown} value
   * @returns {number|null}
   */
  function sanitizeCleanNonNegativeNumber(value) {
    const n = sanitizeNonNegativeNumber(value);
    return n === null ? null : n === 0 ? 0 : n;
  }

  /**
   * Sanitize a positive integer count (`>= 1`, e.g. `mailboxLanes`), returning `null`
   * for missing, zero, negative, non-finite, boolean, XSS, or host-path values.
   *
   * @param {unknown} value
   * @returns {number|null}
   */
  function sanitizePositiveInt(value) {
    const n = sanitizeNonNegativeNumber(value);
    if (n === null) return null;
    const floored = Math.floor(n);
    if (!Number.isSafeInteger(floored) || floored < 1) return null;
    return floored;
  }

  /**
   * Format a clamped `[0, 1]` unit ratio as a percentage string (`0.0%`–`100.0%`),
   * returning `—` when absent or invalid.
   *
   * @param {unknown} value
   * @returns {string}
   */
  function formatRatioPct(value) {
    const ratio = sanitizeUnitRatio(value);
    return ratio === null ? PLACEHOLDER_DASH : `${(ratio * 100).toFixed(1)}%`;
  }

  /**
   * Format a non-negative frame jitter duration in milliseconds (`0.00 ms`),
   * returning `—` when absent or invalid.
   *
   * @param {unknown} value
   * @returns {string}
   */
  function formatFrameJitterMs(value) {
    const n = sanitizeCleanNonNegativeNumber(value);
    return n === null ? PLACEHOLDER_DASH : `${n.toFixed(2)} ms`;
  }

  /**
   * Format a non-negative memory footprint in megabytes (`0.0 MB`),
   * returning `—` when absent or invalid.
   *
   * @param {unknown} value
   * @returns {string}
   */
  function formatMemoryMb(value) {
    const n = sanitizeCleanNonNegativeNumber(value);
    return n === null ? PLACEHOLDER_DASH : `${n.toFixed(1)} MB`;
  }

  /**
   * Sanitize a percentage value into `[0, 100]` (rounded to 1 decimal place),
   * returning `null` for invalid, negative, non-finite, boolean, or hostile values.
   *
   * @param {unknown} value
   * @param {boolean} [isUnitRatio=false]
   * @returns {number|null}
   */
  function sanitizePercentage(value, isUnitRatio = false) {
    const n = sanitizeNonNegativeNumber(value);
    if (n === null) return null;
    const scaled = isUnitRatio && n <= 1 ? n * 100 : n;
    const clamped = Math.min(100, Math.max(0, scaled));
    const rounded = Math.round(clamped * 10) / 10;
    return rounded === 0 ? 0 : rounded;
  }

  /**
   * Strictly allowlist an audio output state to `'active' | 'suspended' | 'headless-drain'`,
   * returning `null` for missing, unknown, or hostile values.
   *
   * @param {unknown} value
   * @returns {'active'|'suspended'|'headless-drain'|null}
   */
  function sanitizeAudioOutputState(value) {
    if (typeof value !== 'string') return null;
    const norm = value.trim().toLowerCase().replace(/_/g, '-');
    if (!norm || norm.length > 32) return null;
    if (norm === 'running' || norm === 'playing') return 'active';
    if (norm === 'headless' || norm === 'drain' || norm === 'draining') return 'headless-drain';
    if (norm === 'paused' || norm === 'closed' || norm === 'idle' || norm === 'muted') return 'suspended';
    return ALLOWED_AUDIO_OUTPUT_STATE_SET.has(norm) ? norm : null;
  }

  function sanitizeBigIntOrNonNegativeInt(value, fallback = 0) {
    if (typeof value === 'bigint') {
      if (value < 0n) return fallback;
      return value > BigInt(Number.MAX_SAFE_INTEGER) ? Number.MAX_SAFE_INTEGER : Number(value);
    }
    return sanitizeNonNegativeInt(value, fallback);
  }

  function formatAudioFrameCount(value) {
    const n = sanitizeBigIntOrNonNegativeInt(value, null);
    if (n === null) return PLACEHOLDER_DASH;
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
    if (n >= 10_000) return `${(n / 1_000).toFixed(1)}K`;
    return String(n);
  }

  function formatOpfsBytes(bytes) {
    const n = sanitizeBigIntOrNonNegativeInt(bytes, null);
    if (n === null || n === 0) return PLACEHOLDER_DASH;
    if (n >= 1024 * 1024 * 1024) return `${(n / (1024 * 1024 * 1024)).toFixed(1)} GiB`;
    if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
    if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${n} B`;
  }

  function formatMemoryBytes(bytes) {
    const n = sanitizeBigIntOrNonNegativeInt(bytes, null);
    if (n === null) return PLACEHOLDER_DASH;
    if (n >= 1024 * 1024 * 1024) return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
    if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
    if (n >= 1024) return `${(n / 1024).toFixed(1)} KiB`;
    return `${n} B`;
  }

  function createDefaultGpuMemoryStats() {
    return {
      liveBufferBytes: null,
      liveTextureBytes: null,
      peakBufferBytes: null,
      peakTextureBytes: null,
      stagingPoolBytes: null,
      totalLiveGpuBytes: null,
      wasmLinearMemoryBytes: null,
    };
  }

  function extractGpuMemoryStats(source, base) {
    let next = base ? { ...base } : createDefaultGpuMemoryStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawGpuMem =
      safeRead(source, 'gpuMemoryStats') !== undefined
        ? safeRead(source, 'gpuMemoryStats')
        : safeRead(source, 'gpuMemory') !== undefined
          ? safeRead(source, 'gpuMemory')
          : safeRead(source, 'gpu_memory');
    if (rawGpuMem === null) {
      next = createDefaultGpuMemoryStats();
    }
    const gpuMemObj = rawGpuMem && typeof rawGpuMem === 'object' ? rawGpuMem : null;
    const rawMemory =
      safeRead(source, 'memoryStats') !== undefined
        ? safeRead(source, 'memoryStats')
        : safeRead(source, 'memory') !== undefined
          ? safeRead(source, 'memory')
          : safeRead(source, 'memory_stats');
    const memoryObj = rawMemory && typeof rawMemory === 'object' ? rawMemory : null;
    const rawGpu = safeRead(source, 'gpu');
    const gpuObj = rawGpu && typeof rawGpu === 'object' ? rawGpu : null;
    const rawFrame = safeRead(source, 'frameStats');
    const frameObj = rawFrame && typeof rawFrame === 'object' ? rawFrame : null;
    const nestedGpuMemInGpu =
      gpuObj && typeof safeRead(gpuObj, 'gpuMemory') === 'object'
        ? safeRead(gpuObj, 'gpuMemory')
        : null;

    const sources = [
      { obj: source, explicitNull: true },
      { obj: gpuMemObj, explicitNull: true },
      { obj: memoryObj, explicitNull: false },
      { obj: nestedGpuMemInGpu, explicitNull: false },
      { obj: gpuObj, explicitNull: false },
      { obj: frameObj, explicitNull: false },
    ];

    const applyByteField = (prop, keys) => {
      const picked = pickFirstDefinedWithSource(sources, keys);
      if (picked.found) {
        if (picked.value === null) {
          if (picked.explicitNull) {
            next[prop] = null;
          }
        } else {
          next[prop] = sanitizeBigIntOrNonNegativeInt(picked.value, null);
        }
      }
    };

    applyByteField('liveBufferBytes', ['liveBufferBytes', 'live_buffer_bytes']);
    applyByteField('liveTextureBytes', ['liveTextureBytes', 'live_texture_bytes']);
    applyByteField('peakBufferBytes', ['peakBufferBytes', 'peak_buffer_bytes']);
    applyByteField('peakTextureBytes', ['peakTextureBytes', 'peak_texture_bytes']);
    applyByteField('stagingPoolBytes', ['stagingPoolBytes', 'staging_pool_bytes']);
    applyByteField('totalLiveGpuBytes', [
      'totalLiveGpuBytes',
      'total_live_gpu_bytes',
      'vramBytes',
      'vram_bytes',
    ]);
    applyByteField('wasmLinearMemoryBytes', ['wasmLinearMemoryBytes', 'wasm_linear_memory_bytes']);

    if (
      next.totalLiveGpuBytes === null &&
      (next.liveBufferBytes !== null || next.liveTextureBytes !== null)
    ) {
      next.totalLiveGpuBytes =
        (next.liveBufferBytes !== null ? next.liveBufferBytes : 0) +
        (next.liveTextureBytes !== null ? next.liveTextureBytes : 0);
    }

    return next;
  }

  function createDefaultShaderStreamingStats() {
    return {
      rangedShaderReady: null,
      rangedShaderFetches: null,
      rangedShaderErrors: null,
      asyncShaderPendingCount: null,
      rangedPackActive: null,
    };
  }

  function extractShaderStreamingStats(source, base) {
    let next = base ? { ...base } : createDefaultShaderStreamingStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawStream =
      safeRead(source, 'shaderStreamingStats') !== undefined
        ? safeRead(source, 'shaderStreamingStats')
        : safeRead(source, 'shaderStreaming') !== undefined
          ? safeRead(source, 'shaderStreaming')
          : safeRead(source, 'shaderPackStats') !== undefined
            ? safeRead(source, 'shaderPackStats')
            : safeRead(source, 'packStats');
    if (rawStream === null) {
      next = createDefaultShaderStreamingStats();
    }
    const streamObj = rawStream && typeof rawStream === 'object' ? rawStream : null;
    const rawGpu = safeRead(source, 'gpu');
    const gpuObj = rawGpu && typeof rawGpu === 'object' ? rawGpu : null;

    const sources = [
      { obj: source, explicitNull: true },
      { obj: streamObj, explicitNull: true },
      { obj: gpuObj, explicitNull: false },
    ];

    const applyIntField = (prop, keys) => {
      const picked = pickFirstDefinedWithSource(sources, keys);
      if (picked.found) {
        if (picked.value === null) {
          if (picked.explicitNull) {
            next[prop] = null;
          }
        } else {
          next[prop] = sanitizeBigIntOrNonNegativeInt(picked.value, null);
        }
      }
    };

    applyIntField('rangedShaderReady', ['rangedShaderReady', 'ranged_shader_ready']);
    applyIntField('rangedShaderFetches', ['rangedShaderFetches', 'ranged_shader_fetches']);
    applyIntField('rangedShaderErrors', ['rangedShaderErrors', 'ranged_shader_errors']);
    applyIntField('asyncShaderPendingCount', [
      'asyncShaderPendingCount',
      'async_shader_pending_count',
    ]);

    const active = pickFirstDefinedWithSource(sources, ['rangedPackActive', 'ranged_pack_active']);
    if (active.found) {
      if (active.value === null) {
        if (active.explicitNull) {
          next.rangedPackActive = null;
        }
      } else {
        next.rangedPackActive = Boolean(active.value);
      }
    }

    return next;
  }

  function formatShaderStreamingText(stats) {
    if (!stats || typeof stats !== 'object') {
      return PLACEHOLDER_DASH;
    }
    const hasReady = typeof stats.rangedShaderReady === 'number';
    const hasFetches = typeof stats.rangedShaderFetches === 'number';
    if (!hasReady && !hasFetches) {
      return stats.rangedPackActive === true ? 'Active' : PLACEHOLDER_DASH;
    }
    const ready = hasReady ? stats.rangedShaderReady : 0;
    const fetches = hasFetches ? stats.rangedShaderFetches : 0;
    const pending =
      typeof stats.asyncShaderPendingCount === 'number' && stats.asyncShaderPendingCount > 0
        ? ` (${stats.asyncShaderPendingCount} pending)`
        : '';
    return `${ready} ready · ${fetches} fetch${fetches === 1 ? '' : 'es'}${pending}`;
  }

  function triggerBrowserJsonDownload(doc, filename, jsonText) {
    try {
      if (
        !doc ||
        typeof doc.createElement !== 'function' ||
        typeof Blob !== 'function' ||
        typeof URL === 'undefined' ||
        !URL ||
        typeof URL.createObjectURL !== 'function'
      ) {
        return false;
      }
      const anchor = doc.createElement('a');
      if (!anchor || typeof anchor.click !== 'function') {
        return false;
      }
      const blob = new Blob([jsonText], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      try {
        anchor.href = url;
        anchor.download = filename;
        if (anchor.style) {
          anchor.style.display = 'none';
        }
        if (doc.body && typeof doc.body.appendChild === 'function') {
          doc.body.appendChild(anchor);
        }
        anchor.click();
        if (anchor.parentNode && typeof anchor.parentNode.removeChild === 'function') {
          anchor.parentNode.removeChild(anchor);
        }
      } finally {
        if (typeof URL.revokeObjectURL === 'function') {
          URL.revokeObjectURL(url);
        }
      }
      return true;
    } catch {
      return false;
    }
  }

  function formatOpfsHitRate(value) {
    const pct = sanitizePercentage(value, false);
    return pct === null ? PLACEHOLDER_DASH : `${pct.toFixed(1)}%`;
  }

  /**
   * Format a positive integer I/O mailbox lane count (`>= 1`) and optional live
   * active/peak concurrency metrics, returning `—` when absent or invalid.
   * Backward-compatible: `formatMailboxLanes(8)` -> `'8'`;
   * `formatMailboxLanes(8, { peakMailboxLanes: 8 })` -> `'8 lanes (peak 8)'`;
   * `formatMailboxLanes({ mailboxLanes: 8, activeMailboxLanes: 6, peakMailboxLanes: 8 })` -> `'6/8 lanes (peak 8)'`.
   *
   * @param {unknown} value
   * @param {object|null|undefined} [opts]
   * @returns {string}
   */
  function formatMailboxLanes(value, opts) {
    let lanes = null;
    let active = null;
    let peak = null;

    if (value && typeof value === 'object' && !Array.isArray(value)) {
      lanes = sanitizePositiveInt(
        safeRead(value, 'mailboxLanes') !== undefined
          ? safeRead(value, 'mailboxLanes')
          : safeRead(value, 'lanes')
      );
      const rawActive =
        safeRead(value, 'activeMailboxLanes') !== undefined
          ? safeRead(value, 'activeMailboxLanes')
          : safeRead(value, 'httpfsActiveConcurrency');
      if (rawActive !== undefined && rawActive !== null) {
        active = sanitizeNonNegativeInt(rawActive, null);
      }
      const rawPeak =
        safeRead(value, 'peakMailboxLanes') !== undefined
          ? safeRead(value, 'peakMailboxLanes')
          : safeRead(value, 'httpfsPeakConcurrency');
      if (rawPeak !== undefined && rawPeak !== null) {
        peak = sanitizeNonNegativeInt(rawPeak, null);
      }
    } else {
      lanes = sanitizePositiveInt(value);
      if (opts && typeof opts === 'object') {
        const rawActive =
          safeRead(opts, 'activeMailboxLanes') !== undefined
            ? safeRead(opts, 'activeMailboxLanes')
            : safeRead(opts, 'httpfsActiveConcurrency');
        if (rawActive !== undefined && rawActive !== null) {
          active = sanitizeNonNegativeInt(rawActive, null);
        }
        const rawPeak =
          safeRead(opts, 'peakMailboxLanes') !== undefined
            ? safeRead(opts, 'peakMailboxLanes')
            : safeRead(opts, 'httpfsPeakConcurrency');
        if (rawPeak !== undefined && rawPeak !== null) {
          peak = sanitizeNonNegativeInt(rawPeak, null);
        }
      }
    }

    if (lanes === null && active === null && peak === null) {
      return PLACEHOLDER_DASH;
    }
    if (lanes !== null && active === null && peak === null) {
      return String(lanes);
    }
    if (active !== null && lanes !== null) {
      return peak !== null ? `${active}/${lanes} lanes (peak ${peak})` : `${active}/${lanes} lanes`;
    }
    if (lanes !== null && peak !== null) {
      return `${lanes} lanes (peak ${peak})`;
    }
    if (peak !== null) {
      return `peak ${peak}`;
    }
    return String(active);
  }

  function createDefaultPacingStats() {
    return {
      fps30ComplianceRate: null,
      frameJitterMs: null,
    };
  }

  function createDefaultMemoryStats() {
    return {
      jsHeapUsedMb: null,
      vramBudgetUtilization: null,
      bindGroupHitRate: null,
    };
  }

  function createDefaultMailboxStats() {
    return {
      mailboxLanes: null,
      activeMailboxLanes: null,
      peakMailboxLanes: null,
      rangeCacheHitRatePct: null,
    };
  }

  function createDefaultAudioStats() {
    return {
      audioOutputState: null,
      guestRingAttached: false,
      guestFramesDrained: 0,
      framesDrained: 0,
      underruns: 0,
      guestUnderruns: 0,
      overruns: 0,
    };
  }

  function createDefaultOpfsStats() {
    return {
      opfsHits: 0,
      opfsMisses: 0,
      opfsWrites: 0,
      opfsInvalidations: 0,
      bytesFromOpfs: 0,
      bytesFromNetwork: 0,
      opfsHitRatePct: null,
    };
  }

  function extractPacingStats(source, base) {
    let next = base ? { ...base } : createDefaultPacingStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawPacing =
      safeRead(source, 'pacingStats') !== undefined
        ? safeRead(source, 'pacingStats')
        : safeRead(source, 'pacing');
    if (rawPacing === null) {
      next = createDefaultPacingStats();
    }
    const pacingObj = rawPacing && typeof rawPacing === 'object' ? rawPacing : null;

    const rawPerf =
      safeRead(source, 'perfStats') !== undefined
        ? safeRead(source, 'perfStats')
        : safeRead(source, 'perf');
    const perfObj = rawPerf && typeof rawPerf === 'object' ? rawPerf : null;

    const rawFrames =
      safeRead(source, 'frameStats') !== undefined
        ? safeRead(source, 'frameStats')
        : safeRead(source, 'frames');
    const framesObj = rawFrames && typeof rawFrames === 'object' ? rawFrames : null;

    const rawBoot =
      safeRead(source, 'bootSummary') !== undefined
        ? safeRead(source, 'bootSummary')
        : safeRead(source, 'boot');
    const bootObj = rawBoot && typeof rawBoot === 'object' ? rawBoot : null;

    const sources = [
      { obj: source, explicitNull: true },
      { obj: pacingObj, explicitNull: true },
      { obj: perfObj, explicitNull: false },
      { obj: framesObj, explicitNull: false },
      { obj: bootObj, explicitNull: false },
    ];

    const compliance = pickFirstDefinedWithSource(sources, ['fps30ComplianceRate']);
    if (compliance.found) {
      if (compliance.value === null) {
        if (compliance.explicitNull) {
          next.fps30ComplianceRate = null;
        }
      } else {
        next.fps30ComplianceRate = sanitizeUnitRatio(compliance.value);
      }
    }

    const jitter = pickFirstDefinedWithSource(sources, ['frameJitterMs']);
    if (jitter.found) {
      if (jitter.value === null) {
        if (jitter.explicitNull) {
          next.frameJitterMs = null;
        }
      } else {
        next.frameJitterMs = sanitizeCleanNonNegativeNumber(jitter.value);
      }
    }

    return next;
  }

  function extractMemoryStats(source, base) {
    let next = base ? { ...base } : createDefaultMemoryStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawMemory =
      safeRead(source, 'memoryStats') !== undefined
        ? safeRead(source, 'memoryStats')
        : safeRead(source, 'memory') !== undefined
          ? safeRead(source, 'memory')
          : safeRead(source, 'vramStats') !== undefined
            ? safeRead(source, 'vramStats')
            : safeRead(source, 'vram');
    if (rawMemory === null) {
      next = createDefaultMemoryStats();
    }
    const memoryObj = rawMemory && typeof rawMemory === 'object' ? rawMemory : null;

    const rawPerf =
      safeRead(source, 'perfStats') !== undefined
        ? safeRead(source, 'perfStats')
        : safeRead(source, 'perf');
    const perfObj = rawPerf && typeof rawPerf === 'object' ? rawPerf : null;

    const rawResourceTable =
      safeRead(source, 'resourceTableStats') !== undefined
        ? safeRead(source, 'resourceTableStats')
        : safeRead(source, 'resourceTable') !== undefined
          ? safeRead(source, 'resourceTable')
          : safeRead(source, 'bindGroupStats');
    const resourceTableObj =
      rawResourceTable && typeof rawResourceTable === 'object' ? rawResourceTable : null;

    const rawRenderState = safeRead(source, 'renderStateStats');
    const renderStateObj =
      rawRenderState && typeof rawRenderState === 'object' ? rawRenderState : null;

    const rawGpu =
      safeRead(source, 'compatStats') !== undefined
        ? safeRead(source, 'compatStats')
        : safeRead(source, 'gpuCompatStats') !== undefined
          ? safeRead(source, 'gpuCompatStats')
          : safeRead(source, 'gpu');
    const gpuObj = rawGpu && typeof rawGpu === 'object' ? rawGpu : null;

    const rawBoot =
      safeRead(source, 'bootSummary') !== undefined
        ? safeRead(source, 'bootSummary')
        : safeRead(source, 'boot');
    const bootObj = rawBoot && typeof rawBoot === 'object' ? rawBoot : null;

    const sources = [
      { obj: source, explicitNull: true },
      { obj: memoryObj, explicitNull: true },
      { obj: perfObj, explicitNull: false },
      { obj: resourceTableObj, explicitNull: false },
      { obj: renderStateObj, explicitNull: false },
      { obj: gpuObj, explicitNull: false },
      { obj: bootObj, explicitNull: false },
    ];

    const jsHeap = pickFirstDefinedWithSource(sources, ['jsHeapUsedMb']);
    if (jsHeap.found) {
      if (jsHeap.value === null) {
        if (jsHeap.explicitNull) {
          next.jsHeapUsedMb = null;
        }
      } else {
        next.jsHeapUsedMb = sanitizeCleanNonNegativeNumber(jsHeap.value);
      }
    }

    const vramUtil = pickFirstDefinedWithSource(sources, ['vramBudgetUtilization']);
    if (vramUtil.found) {
      if (vramUtil.value === null) {
        if (vramUtil.explicitNull) {
          next.vramBudgetUtilization = null;
        }
      } else {
        next.vramBudgetUtilization = sanitizeUnitRatio(vramUtil.value);
      }
    }

    const bgHitRate = pickFirstDefinedWithSource(sources, [
      'bindGroupHitRate',
      'bindGroupCacheHitRate',
    ]);
    if (bgHitRate.found) {
      if (bgHitRate.value === null) {
        if (bgHitRate.explicitNull) {
          next.bindGroupHitRate = null;
        }
      } else {
        next.bindGroupHitRate = sanitizeUnitRatio(bgHitRate.value);
      }
    }

    return next;
  }

  function extractMailboxStats(source, base) {
    let next = base ? { ...base } : createDefaultMailboxStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawIo =
      safeRead(source, 'ioStats') !== undefined
        ? safeRead(source, 'ioStats')
        : safeRead(source, 'httpfsStats') !== undefined
          ? safeRead(source, 'httpfsStats')
          : safeRead(source, 'io');
    const rawMailbox =
      safeRead(source, 'mailboxStats') !== undefined
        ? safeRead(source, 'mailboxStats')
        : safeRead(source, 'mailbox');
    if (rawIo === null || rawMailbox === null) {
      next = createDefaultMailboxStats();
    }
    const ioObj = rawIo && typeof rawIo === 'object' ? rawIo : null;
    const mailboxObj = rawMailbox && typeof rawMailbox === 'object' ? rawMailbox : null;

    const rawBoot =
      safeRead(source, 'bootSummary') !== undefined
        ? safeRead(source, 'bootSummary')
        : safeRead(source, 'boot');
    const bootObj = rawBoot && typeof rawBoot === 'object' ? rawBoot : null;

    const sources = [
      { obj: source, explicitNull: true },
      { obj: mailboxObj, explicitNull: true },
      { obj: ioObj, explicitNull: true },
      { obj: bootObj, explicitNull: false },
    ];

    const picked = pickFirstDefinedWithSource(sources, ['mailboxLanes', 'mailbox_lanes']);
    if (picked.found) {
      if (picked.value === null) {
        if (picked.explicitNull) {
          next.mailboxLanes = null;
        }
      } else {
        next.mailboxLanes = sanitizePositiveInt(picked.value);
      }
    }

    const pickedActive = pickFirstDefinedWithSource(sources, [
      'activeMailboxLanes',
      'active_mailbox_lanes',
      'httpfsActiveConcurrency',
    ]);
    if (pickedActive.found) {
      if (pickedActive.value === null) {
        if (pickedActive.explicitNull) {
          next.activeMailboxLanes = null;
        }
      } else {
        const parsed = sanitizeNonNegativeNumber(pickedActive.value);
        next.activeMailboxLanes = parsed === null ? null : Math.floor(parsed);
      }
    }

    const pickedPeak = pickFirstDefinedWithSource(sources, [
      'peakMailboxLanes',
      'peak_mailbox_lanes',
      'httpfsPeakConcurrency',
      'httpfs_peak_concurrency',
    ]);
    if (pickedPeak.found) {
      if (pickedPeak.value === null) {
        if (pickedPeak.explicitNull) {
          next.peakMailboxLanes = null;
        }
      } else {
        const parsed = sanitizeNonNegativeNumber(pickedPeak.value);
        next.peakMailboxLanes = parsed === null ? null : Math.floor(parsed);
      }
    } else if (pickedActive.found && next.activeMailboxLanes !== null) {
      next.peakMailboxLanes = Math.max(next.peakMailboxLanes ?? 0, next.activeMailboxLanes);
    }

    const pickedHitPct = pickFirstDefinedWithSource(sources, [
      'rangeCacheHitRatePct',
      'range_cache_hit_rate_pct',
      'slabHitRatePct',
      'slab_hit_rate_pct',
    ]);
    if (pickedHitPct.found) {
      if (pickedHitPct.value === null) {
        if (pickedHitPct.explicitNull) {
          next.rangeCacheHitRatePct = null;
        }
      } else {
        next.rangeCacheHitRatePct = sanitizePercentage(pickedHitPct.value, false);
      }
    } else {
      const pickedHitRatio = pickFirstDefinedWithSource(sources, ['rangeCacheHitRate', 'slabHitRate']);
      if (pickedHitRatio.found) {
        if (pickedHitRatio.value === null) {
          if (pickedHitRatio.explicitNull) {
            next.rangeCacheHitRatePct = null;
          }
        } else {
          next.rangeCacheHitRatePct = sanitizePercentage(pickedHitRatio.value, true);
        }
      }
    }

    return next;
  }

  function extractAudioStats(source, base) {
    let next = base ? { ...base } : createDefaultAudioStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawAudio =
      safeRead(source, 'audioStats') !== undefined
        ? safeRead(source, 'audioStats')
        : safeRead(source, 'audio');
    if (rawAudio === null) {
      next = createDefaultAudioStats();
    }
    const audioObj = rawAudio && typeof rawAudio === 'object' ? rawAudio : null;

    const sources = [
      { obj: source, explicitNull: true },
      { obj: audioObj, explicitNull: true },
    ];

    const statePicked = pickFirstDefinedWithSource(sources, [
      'audioOutputState',
      'audio_output_state',
    ]);
    const nestedStatePicked = audioObj
      ? pickFirstDefinedWithSource([{ obj: audioObj, explicitNull: true }], [
          'outputState',
          'output_state',
        ])
      : { found: false, value: undefined, explicitNull: false };
    const effectiveState = statePicked.found ? statePicked : nestedStatePicked;
    if (effectiveState.found) {
      if (effectiveState.value === null) {
        if (effectiveState.explicitNull) {
          next.audioOutputState = null;
        }
      } else {
        next.audioOutputState = sanitizeAudioOutputState(effectiveState.value);
      }
    }

    const ringPicked = pickFirstDefinedWithSource(sources, [
      'guestRingAttached',
      'guest_ring_attached',
      'ringAttached',
    ]);
    if (ringPicked.found) {
      next.guestRingAttached = ringPicked.value === true;
    }

    const guestDrained = pickFirstDefinedWithSource(sources, [
      'guestFramesDrained',
      'guest_frames_drained',
    ]);
    if (guestDrained.found) {
      next.guestFramesDrained = sanitizeBigIntOrNonNegativeInt(guestDrained.value, 0);
    }

    const drained = pickFirstDefinedWithSource(sources, [
      'framesDrained',
      'frames_drained',
      'drainedFrames',
    ]);
    if (drained.found) {
      next.framesDrained = sanitizeBigIntOrNonNegativeInt(drained.value, 0);
    }

    const underruns = pickFirstDefinedWithSource(sources, ['underruns', 'underrunCount']);
    if (underruns.found) {
      next.underruns = sanitizeBigIntOrNonNegativeInt(underruns.value, 0);
    }

    const guestUnderruns = pickFirstDefinedWithSource(sources, [
      'guestUnderruns',
      'guest_underruns',
    ]);
    if (guestUnderruns.found) {
      next.guestUnderruns = sanitizeBigIntOrNonNegativeInt(guestUnderruns.value, 0);
    }

    const overruns = pickFirstDefinedWithSource(sources, ['overruns', 'overrunCount', 'overflows']);
    if (overruns.found) {
      next.overruns = sanitizeBigIntOrNonNegativeInt(overruns.value, 0);
    }

    return next;
  }

  function extractOpfsStats(source, base) {
    let next = base ? { ...base } : createDefaultOpfsStats();
    if (!source || typeof source !== 'object') {
      return next;
    }

    const rawOpfs =
      safeRead(source, 'opfsStats') !== undefined
        ? safeRead(source, 'opfsStats')
        : safeRead(source, 'opfsTelemetry') !== undefined
          ? safeRead(source, 'opfsTelemetry')
          : safeRead(source, 'preloadTelemetry') !== undefined
            ? safeRead(source, 'preloadTelemetry')
            : safeRead(source, 'opfs');
    if (rawOpfs === null) {
      next = createDefaultOpfsStats();
    }
    const opfsObj = rawOpfs && typeof rawOpfs === 'object' ? rawOpfs : null;

    const rawIo =
      safeRead(source, 'ioStats') !== undefined
        ? safeRead(source, 'ioStats')
        : safeRead(source, 'httpfsStats') !== undefined
          ? safeRead(source, 'httpfsStats')
          : safeRead(source, 'io');
    const ioObj = rawIo && typeof rawIo === 'object' ? rawIo : null;

    const sources = [
      { obj: source, explicitNull: true },
      { obj: opfsObj, explicitNull: true },
      { obj: ioObj, explicitNull: false },
    ];

    let countsTouched = false;

    const hits = pickFirstDefinedWithSource(sources, ['opfsHits', 'opfs_hits', 'opfsCacheHits']);
    if (hits.found) {
      next.opfsHits = sanitizeBigIntOrNonNegativeInt(hits.value, 0);
      countsTouched = true;
    }

    const misses = pickFirstDefinedWithSource(sources, [
      'opfsMisses',
      'opfs_misses',
      'opfsCacheMisses',
    ]);
    if (misses.found) {
      next.opfsMisses = sanitizeBigIntOrNonNegativeInt(misses.value, 0);
      countsTouched = true;
    }

    const writes = pickFirstDefinedWithSource(sources, ['opfsWrites', 'opfs_writes']);
    if (writes.found) {
      next.opfsWrites = sanitizeBigIntOrNonNegativeInt(writes.value, 0);
    }

    const invalidations = pickFirstDefinedWithSource(sources, [
      'opfsInvalidations',
      'opfs_invalidations',
    ]);
    if (invalidations.found) {
      next.opfsInvalidations = sanitizeBigIntOrNonNegativeInt(invalidations.value, 0);
    }

    const bytesFromOpfs = pickFirstDefinedWithSource(sources, ['bytesFromOpfs', 'bytes_from_opfs']);
    if (bytesFromOpfs.found) {
      next.bytesFromOpfs = sanitizeBigIntOrNonNegativeInt(bytesFromOpfs.value, 0);
      countsTouched = true;
    }

    const bytesFromNetwork = pickFirstDefinedWithSource(sources, [
      'bytesFromNetwork',
      'bytes_from_network',
    ]);
    if (bytesFromNetwork.found) {
      next.bytesFromNetwork = sanitizeBigIntOrNonNegativeInt(bytesFromNetwork.value, 0);
      countsTouched = true;
    }

    const explicitPct = pickFirstDefinedWithSource(sources, [
      'opfsHitRatePct',
      'opfs_hit_rate_pct',
      'warmBootHitRatePct',
    ]);
    const explicitRatio = pickFirstDefinedWithSource(sources, ['opfsHitRate', 'opfs_hit_rate']);
    if (explicitPct.found) {
      if (explicitPct.value === null) {
        if (explicitPct.explicitNull) {
          next.opfsHitRatePct = null;
        }
      } else {
        next.opfsHitRatePct = sanitizePercentage(explicitPct.value, false);
      }
    } else if (explicitRatio.found) {
      if (explicitRatio.value === null) {
        if (explicitRatio.explicitNull) {
          next.opfsHitRatePct = null;
        }
      } else {
        next.opfsHitRatePct = sanitizePercentage(explicitRatio.value, true);
      }
    } else if (countsTouched) {
      const lookups = next.opfsHits + next.opfsMisses;
      const totalBytes = next.bytesFromOpfs + next.bytesFromNetwork;
      if (lookups > 0) {
        next.opfsHitRatePct = Math.round((next.opfsHits / lookups) * 1000) / 10;
      } else if (totalBytes > 0) {
        next.opfsHitRatePct = Math.round((next.bytesFromOpfs / totalBytes) * 1000) / 10;
      } else {
        next.opfsHitRatePct = null;
      }
    }

    return next;
  }

  /**
   * Build a concise, sanitized single-line diagnostics summary string from a panel
   * snapshot or raw telemetry payload. Gracefully omits null/missing optional metrics
   * and strips any host paths, URLs, or XSS payloads.
   *
   * @param {object|null|undefined} input
   * @returns {string}
   */
  function formatSummary(input) {
    const source = input && typeof input === 'object' ? input : {};
    const boot = normalizeBootSummary(
      safeRead(source, 'bootSummary') !== undefined
        ? safeRead(source, 'bootSummary')
        : safeRead(source, 'boot')
    );
    const frameStats = normalizeFrameStats(safeRead(source, 'frameStats'));
    const fallbackStats = normalizeFallbackStats(safeRead(source, 'fallbackStats'));
    const adapterStats = extractAdapterStats(source, null);
    const pacingStats = extractPacingStats(source, null);
    const memoryStats = extractMemoryStats(source, null);
    const mailboxStats = extractMailboxStats(source, null);
    const audioStats = extractAudioStats(source, null);
    const opfsStats = extractOpfsStats(source, null);

    const parts = [
      `${boot.stage} (${boot.progressPct}%)`,
      `frames=${frameStats.framesPresented}`,
      `packHit=${formatHitRatePct(fallbackStats.hitRatePct)}`,
    ];

    if (adapterStats.gpuAdapter !== null) {
      parts.push(`adapter=${adapterStats.gpuAdapter}`);
    }
    if (pacingStats.fps30ComplianceRate !== null) {
      parts.push(`30fps=${formatRatioPct(pacingStats.fps30ComplianceRate)}`);
    }
    if (pacingStats.frameJitterMs !== null) {
      parts.push(`jitter=${formatFrameJitterMs(pacingStats.frameJitterMs)}`);
    }
    if (memoryStats.jsHeapUsedMb !== null) {
      parts.push(`jsHeap=${formatMemoryMb(memoryStats.jsHeapUsedMb)}`);
    }
    if (memoryStats.vramBudgetUtilization !== null) {
      parts.push(`vram=${formatRatioPct(memoryStats.vramBudgetUtilization)}`);
    }
    if (memoryStats.bindGroupHitRate !== null) {
      parts.push(`bgHit=${formatRatioPct(memoryStats.bindGroupHitRate)}`);
    }
    if (mailboxStats.mailboxLanes !== null) {
      if (mailboxStats.activeMailboxLanes !== null && mailboxStats.peakMailboxLanes !== null) {
        parts.push(
          `mailboxLanes=${mailboxStats.activeMailboxLanes}/${mailboxStats.mailboxLanes}(peak=${mailboxStats.peakMailboxLanes})`
        );
      } else if (mailboxStats.peakMailboxLanes !== null) {
        parts.push(`mailboxLanes=${mailboxStats.mailboxLanes}(peak=${mailboxStats.peakMailboxLanes})`);
      } else {
        parts.push(`mailboxLanes=${mailboxStats.mailboxLanes}`);
      }
    }
    if (mailboxStats.rangeCacheHitRatePct !== null) {
      parts.push(`ioCacheHit=${mailboxStats.rangeCacheHitRatePct.toFixed(1)}%`);
    }
    if (
      audioStats.guestRingAttached ||
      audioStats.guestFramesDrained > 0 ||
      audioStats.framesDrained > 0 ||
      audioStats.audioOutputState !== null
    ) {
      const drainedCount =
        audioStats.guestFramesDrained > 0 ? audioStats.guestFramesDrained : audioStats.framesDrained;
      const ringTag = audioStats.guestRingAttached ? 'attached' : audioStats.audioOutputState || 'detached';
      parts.push(
        drainedCount > 0 ? `audio=${ringTag}(${formatAudioFrameCount(drainedCount)})` : `audio=${ringTag}`
      );
    }
    if (opfsStats.opfsHitRatePct !== null) {
      parts.push(`opfsHit=${opfsStats.opfsHitRatePct.toFixed(1)}%`);
    }

    return parts.join(' | ');
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
    const pacingStats = extractPacingStats(source, null);
    const memoryStats = extractMemoryStats(source, null);
    const gpuMemoryStats = extractGpuMemoryStats(source, null);
    const shaderStreamingStats = extractShaderStreamingStats(source, null);
    const swapchainColorStats = extractSwapchainColorStats(source, null, compatStats);
    const mailboxStats = extractMailboxStats(source, null);
    const audioStats = extractAudioStats(source, null);
    const opfsStats = extractOpfsStats(source, null);

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
        playabilityState: boot.playabilityState,
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
        swapchainColorMode: swapchainColorStats.swapchainColorMode,
        toneMapMode: swapchainColorStats.toneMapMode,
        isHdrStream: swapchainColorStats.isHdrStream,
        toneMapActive: swapchainColorStats.toneMapActive,
        applySrgbEncode: swapchainColorStats.applySrgbEncode,
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
      pacing: {
        fps30ComplianceRate: pacingStats.fps30ComplianceRate,
        frameJitterMs: pacingStats.frameJitterMs,
      },
      memory: {
        jsHeapUsedMb: memoryStats.jsHeapUsedMb,
        vramBudgetUtilization: memoryStats.vramBudgetUtilization,
        bindGroupHitRate: memoryStats.bindGroupHitRate,
      },
      gpuMemory: {
        liveBufferBytes: gpuMemoryStats.liveBufferBytes,
        liveTextureBytes: gpuMemoryStats.liveTextureBytes,
        peakBufferBytes: gpuMemoryStats.peakBufferBytes,
        peakTextureBytes: gpuMemoryStats.peakTextureBytes,
        stagingPoolBytes: gpuMemoryStats.stagingPoolBytes,
        totalLiveGpuBytes: gpuMemoryStats.totalLiveGpuBytes,
        wasmLinearMemoryBytes: gpuMemoryStats.wasmLinearMemoryBytes,
      },
      shaderStreaming: {
        rangedShaderReady: shaderStreamingStats.rangedShaderReady,
        rangedShaderFetches: shaderStreamingStats.rangedShaderFetches,
        rangedShaderErrors: shaderStreamingStats.rangedShaderErrors,
        asyncShaderPendingCount: shaderStreamingStats.asyncShaderPendingCount,
        rangedPackActive: shaderStreamingStats.rangedPackActive,
      },
      mailboxLanes: mailboxStats.mailboxLanes,
      httpfsConcurrency: {
        mailboxLanes: mailboxStats.mailboxLanes,
        activeMailboxLanes: mailboxStats.activeMailboxLanes,
        peakMailboxLanes: mailboxStats.peakMailboxLanes,
        rangeCacheHitRatePct: mailboxStats.rangeCacheHitRatePct,
      },
      audio: {
        outputState: audioStats.audioOutputState,
        guestRingAttached: audioStats.guestRingAttached,
        guestFramesDrained: audioStats.guestFramesDrained,
        framesDrained: audioStats.framesDrained,
        underruns: audioStats.underruns,
        guestUnderruns: audioStats.guestUnderruns,
        overruns: audioStats.overruns,
      },
      opfs: {
        opfsHits: opfsStats.opfsHits,
        opfsMisses: opfsStats.opfsMisses,
        opfsWrites: opfsStats.opfsWrites,
        opfsInvalidations: opfsStats.opfsInvalidations,
        bytesFromOpfs: opfsStats.bytesFromOpfs,
        bytesFromNetwork: opfsStats.bytesFromNetwork,
        opfsHitRatePct: opfsStats.opfsHitRatePct,
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
      this._pacingStats = createDefaultPacingStats();
      this._memoryStats = createDefaultMemoryStats();
      this._gpuMemoryStats = createDefaultGpuMemoryStats();
      this._shaderStreamingStats = createDefaultShaderStreamingStats();
      this._swapchainColorStats = createDefaultSwapchainColorStats();
      this._mailboxStats = createDefaultMailboxStats();
      this._audioStats = createDefaultAudioStats();
      this._opfsStats = createDefaultOpfsStats();
      this._missingShaders = [];
      this._renderedCache = Object.create(null);
      this._headerDirty = false;
      this._bodyDirty = false;

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
      const swapchainColorModeMetric = this._buildMetricCard(
        doc,
        'Color Mode',
        `${ROOT_CLASS}__swapchain-color-mode`,
        PLACEHOLDER_DASH
      );
      compatGrid.appendChild(bindMismatchesMetric.card);
      compatGrid.appendChild(autoLayoutMetric.card);
      compatGrid.appendChild(bc3dFallbacksMetric.card);
      compatGrid.appendChild(bc3dSkippedMetric.card);
      compatGrid.appendChild(swapchainFormatMetric.card);
      compatGrid.appendChild(blitFallbackMetric.card);
      compatGrid.appendChild(swapchainColorModeMetric.card);
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
      const shaderStreamingMetric = this._buildMetricCard(
        doc,
        'Shader Stream',
        `${ROOT_CLASS}__shader-streaming`,
        PLACEHOLDER_DASH
      );

      adapterReflectionGrid.appendChild(gpuAdapterMetric.card);
      adapterReflectionGrid.appendChild(firstVisibleFrameMsMetric.card);
      adapterReflectionGrid.appendChild(blackLoadingMsMetric.card);
      adapterReflectionGrid.appendChild(blackPresentsTotalMetric.card);
      adapterReflectionGrid.appendChild(reflectedStagesMetric.card);
      adapterReflectionGrid.appendChild(inlineSamplerStagesMetric.card);
      adapterReflectionGrid.appendChild(resourceHeapBindingsMetric.card);
      adapterReflectionGrid.appendChild(shaderStreamingMetric.card);
      adapterReflectionSection.appendChild(adapterReflectionGrid);

      // --- 30 FPS pacing, memory/VRAM & I/O mailbox telemetry section (W292) ---
      const pacingMemorySection = createEl(
        doc,
        'div',
        `${ROOT_CLASS}__section ${ROOT_CLASS}__pacing-memory`,
        {
          role: 'group',
          'aria-label': '30 FPS Pacing, Memory & I/O Telemetry',
        }
      );
      pacingMemorySection.appendChild(
        createEl(doc, 'div', `${ROOT_CLASS}__section-title`, {
          text: '30 FPS Pacing, Memory & I/O',
        })
      );
      const pacingMemoryGrid = createEl(doc, 'div', `${ROOT_CLASS}__grid`);
      const fps30ComplianceMetric = this._buildMetricCard(
        doc,
        '30 FPS Compliance',
        `${ROOT_CLASS}__fps30-compliance-rate`,
        PLACEHOLDER_DASH
      );
      const frameJitterMetric = this._buildMetricCard(
        doc,
        'Frame Jitter',
        `${ROOT_CLASS}__frame-jitter-ms`,
        PLACEHOLDER_DASH
      );
      const jsHeapUsedMetric = this._buildMetricCard(
        doc,
        'JS Heap',
        `${ROOT_CLASS}__js-heap-used-mb`,
        PLACEHOLDER_DASH
      );
      const vramBudgetMetric = this._buildMetricCard(
        doc,
        'VRAM Budget',
        `${ROOT_CLASS}__vram-budget-utilization`,
        PLACEHOLDER_DASH
      );
      const bindGroupHitRateMetric = this._buildMetricCard(
        doc,
        'BG Hit Rate',
        `${ROOT_CLASS}__bind-group-hit-rate`,
        PLACEHOLDER_DASH
      );
      const mailboxLanesMetric = this._buildMetricCard(
        doc,
        'I/O Mailbox Lanes',
        `${ROOT_CLASS}__mailbox-lanes`,
        PLACEHOLDER_DASH
      );
      const gpuVramLiveMetric = this._buildMetricCard(
        doc,
        'GPU VRAM Live',
        `${ROOT_CLASS}__gpu-vram-live`,
        PLACEHOLDER_DASH
      );
      const gpuVramPeakMetric = this._buildMetricCard(
        doc,
        'GPU VRAM Peak',
        `${ROOT_CLASS}__gpu-vram-peak`,
        PLACEHOLDER_DASH
      );
      const wasmLinearMemoryMetric = this._buildMetricCard(
        doc,
        'Wasm Memory',
        `${ROOT_CLASS}__wasm-linear-memory`,
        PLACEHOLDER_DASH
      );
      pacingMemoryGrid.appendChild(fps30ComplianceMetric.card);
      pacingMemoryGrid.appendChild(frameJitterMetric.card);
      pacingMemoryGrid.appendChild(jsHeapUsedMetric.card);
      pacingMemoryGrid.appendChild(vramBudgetMetric.card);
      pacingMemoryGrid.appendChild(bindGroupHitRateMetric.card);
      pacingMemoryGrid.appendChild(mailboxLanesMetric.card);
      pacingMemoryGrid.appendChild(gpuVramLiveMetric.card);
      pacingMemoryGrid.appendChild(gpuVramPeakMetric.card);
      pacingMemoryGrid.appendChild(wasmLinearMemoryMetric.card);
      pacingMemorySection.appendChild(pacingMemoryGrid);

      // --- Live M4 Audio Ring & OPFS Warm-Boot telemetry section (W376) ---
      const m4LiveSection = createEl(
        doc,
        'div',
        `${ROOT_CLASS}__section ${ROOT_CLASS}__m4-live`,
        {
          role: 'group',
          'aria-label': 'Live M4 Audio Ring & OPFS Warm-Boot',
        }
      );
      m4LiveSection.appendChild(
        createEl(doc, 'div', `${ROOT_CLASS}__section-title`, {
          text: 'Audio Ring & OPFS Warm-Boot',
        })
      );
      const m4LiveGrid = createEl(doc, 'div', `${ROOT_CLASS}__grid`);
      const audioRingMetric = this._buildMetricCard(
        doc,
        'Audio Ring',
        `${ROOT_CLASS}__audio-ring`,
        'Detached'
      );
      const audioOutputStateMetric = this._buildMetricCard(
        doc,
        'Audio Output',
        `${ROOT_CLASS}__audio-output-state`,
        PLACEHOLDER_DASH
      );
      const audioFramesDrainedMetric = this._buildMetricCard(
        doc,
        'Audio Drained',
        `${ROOT_CLASS}__audio-frames-drained`,
        '0'
      );
      const opfsHitRateMetric = this._buildMetricCard(
        doc,
        'OPFS Hit Rate',
        `${ROOT_CLASS}__opfs-hit-rate`,
        PLACEHOLDER_DASH
      );
      const opfsHitsMissesMetric = this._buildMetricCard(
        doc,
        'OPFS Hits/Miss',
        `${ROOT_CLASS}__opfs-hits-misses`,
        '0 / 0'
      );
      const opfsBytesMetric = this._buildMetricCard(
        doc,
        'OPFS Cached',
        `${ROOT_CLASS}__opfs-bytes`,
        PLACEHOLDER_DASH
      );
      m4LiveGrid.appendChild(audioRingMetric.card);
      m4LiveGrid.appendChild(audioOutputStateMetric.card);
      m4LiveGrid.appendChild(audioFramesDrainedMetric.card);
      m4LiveGrid.appendChild(opfsHitRateMetric.card);
      m4LiveGrid.appendChild(opfsHitsMissesMetric.card);
      m4LiveGrid.appendChild(opfsBytesMetric.card);
      m4LiveSection.appendChild(m4LiveGrid);

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
      body.appendChild(pacingMemorySection);
      body.appendChild(m4LiveSection);
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
        swapchainColorMode: swapchainColorModeMetric.val,
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
        shaderStreaming: shaderStreamingMetric.val,
        fps30ComplianceRate: fps30ComplianceMetric.val,
        frameJitterMs: frameJitterMetric.val,
        jsHeapUsedMb: jsHeapUsedMetric.val,
        vramBudgetUtilization: vramBudgetMetric.val,
        bindGroupHitRate: bindGroupHitRateMetric.val,
        mailboxLanes: mailboxLanesMetric.val,
        gpuVramLive: gpuVramLiveMetric.val,
        gpuVramPeak: gpuVramPeakMetric.val,
        wasmLinearMemory: wasmLinearMemoryMetric.val,
        audioRing: audioRingMetric.val,
        audioOutputState: audioOutputStateMetric.val,
        audioFramesDrained: audioFramesDrainedMetric.val,
        opfsHitRate: opfsHitRateMetric.val,
        opfsHitsMisses: opfsHitsMissesMetric.val,
        opfsBytes: opfsBytesMetric.val,
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
          swapchainColorMode: q(`.${ROOT_CLASS}__swapchain-color-mode`),
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
          shaderStreaming: q(`.${ROOT_CLASS}__shader-streaming`),
          fps30ComplianceRate: q(`.${ROOT_CLASS}__fps30-compliance-rate`),
          frameJitterMs: q(`.${ROOT_CLASS}__frame-jitter-ms`),
          jsHeapUsedMb: q(`.${ROOT_CLASS}__js-heap-used-mb`),
          vramBudgetUtilization: q(`.${ROOT_CLASS}__vram-budget-utilization`),
          bindGroupHitRate: q(`.${ROOT_CLASS}__bind-group-hit-rate`),
          mailboxLanes: q(`.${ROOT_CLASS}__mailbox-lanes`),
          gpuVramLive: q(`.${ROOT_CLASS}__gpu-vram-live`),
          gpuVramPeak: q(`.${ROOT_CLASS}__gpu-vram-peak`),
          wasmLinearMemory: q(`.${ROOT_CLASS}__wasm-linear-memory`),
          audioRing: q(`.${ROOT_CLASS}__audio-ring`),
          audioOutputState: q(`.${ROOT_CLASS}__audio-output-state`),
          audioFramesDrained: q(`.${ROOT_CLASS}__audio-frames-drained`),
          opfsHitRate: q(`.${ROOT_CLASS}__opfs-hit-rate`),
          opfsHitsMisses: q(`.${ROOT_CLASS}__opfs-hits-misses`),
          opfsBytes: q(`.${ROOT_CLASS}__opfs-bytes`),
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
          const jsonText = this.exportMissingShadersJson();
          if (typeof this._onExportMissingShaders !== 'function' && typeof jsonText === 'string') {
            triggerBrowserJsonDownload(this._document, 'playgta6-missing-shaders.json', jsonText);
          }
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

      if (!this._visible) {
        this._headerDirty = true;
        this._bodyDirty = true;
        return;
      }

      const boot = this._boot;
      const frames = this._frameStats;
      const fb = this._fallbackStats;
      const compat = this._compatStats;
      const io = this._ioStats;
      const perf = this._perfStats;
      const adapter = this._adapterStats;
      const visibility = this._visibilityStats;
      const reflection = this._reflectionStats;
      const pacing = this._pacingStats;
      const memory = this._memoryStats;
      const gpuMemory = this._gpuMemoryStats;
      const shaderStreaming = this._shaderStreamingStats;
      const swapchainColor = this._swapchainColorStats;
      const mailbox = this._mailboxStats;
      const audio = this._audioStats;
      const opfs = this._opfsStats;

      try {
        this._setTextIfChanged('summaryBadge', els.summaryBadge, `${boot.stage} · ${boot.progressPct}%`);
        this._headerDirty = false;

        if (this._collapsed) {
          this._bodyDirty = true;
          return;
        }
        this._bodyDirty = false;
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
        this._setTextIfChanged(
          'swapchainColorMode',
          els.swapchainColorMode,
          swapchainColor.swapchainColorMode || PLACEHOLDER_DASH
        );

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
        this._setTextIfChanged(
          'shaderStreaming',
          els.shaderStreaming,
          formatShaderStreamingText(shaderStreaming)
        );

        // W292: 30 FPS pacing, #1077 memory/VRAM telemetry & #1015 I/O mailbox lanes.
        this._setTextIfChanged(
          'fps30ComplianceRate',
          els.fps30ComplianceRate,
          formatRatioPct(pacing.fps30ComplianceRate)
        );
        this._setTextIfChanged(
          'frameJitterMs',
          els.frameJitterMs,
          formatFrameJitterMs(pacing.frameJitterMs)
        );
        this._setTextIfChanged(
          'jsHeapUsedMb',
          els.jsHeapUsedMb,
          formatMemoryMb(memory.jsHeapUsedMb)
        );
        this._setTextIfChanged(
          'vramBudgetUtilization',
          els.vramBudgetUtilization,
          formatRatioPct(memory.vramBudgetUtilization)
        );
        this._setTextIfChanged(
          'bindGroupHitRate',
          els.bindGroupHitRate,
          formatRatioPct(memory.bindGroupHitRate)
        );
        this._setTextIfChanged(
          'mailboxLanes',
          els.mailboxLanes,
          formatMailboxLanes(mailbox.mailboxLanes, mailbox)
        );
        this._setTextIfChanged(
          'gpuVramLive',
          els.gpuVramLive,
          formatMemoryBytes(gpuMemory.totalLiveGpuBytes)
        );
        const peakGpuBytes =
          gpuMemory.peakBufferBytes !== null || gpuMemory.peakTextureBytes !== null
            ? (gpuMemory.peakBufferBytes || 0) + (gpuMemory.peakTextureBytes || 0)
            : null;
        this._setTextIfChanged(
          'gpuVramPeak',
          els.gpuVramPeak,
          formatMemoryBytes(peakGpuBytes)
        );
        this._setTextIfChanged(
          'wasmLinearMemory',
          els.wasmLinearMemory,
          formatMemoryBytes(gpuMemory.wasmLinearMemoryBytes)
        );

        // W376: Live M4 audio ring & OPFS warm-boot diagnostics cards.
        this._setTextIfChanged(
          'audioRing',
          els.audioRing,
          audio.guestRingAttached ? 'Attached' : 'Detached'
        );
        this._setTextIfChanged(
          'audioOutputState',
          els.audioOutputState,
          audio.audioOutputState || PLACEHOLDER_DASH
        );
        const effectiveDrained =
          audio.guestFramesDrained > 0 ? audio.guestFramesDrained : audio.framesDrained;
        this._setTextIfChanged(
          'audioFramesDrained',
          els.audioFramesDrained,
          formatAudioFrameCount(effectiveDrained)
        );
        this._setTextIfChanged(
          'opfsHitRate',
          els.opfsHitRate,
          formatOpfsHitRate(opfs.opfsHitRatePct)
        );
        this._setTextIfChanged(
          'opfsHitsMisses',
          els.opfsHitsMisses,
          `${opfs.opfsHits} / ${opfs.opfsMisses}`
        );
        this._setTextIfChanged(
          'opfsBytes',
          els.opfsBytes,
          formatOpfsBytes(opfs.bytesFromOpfs)
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
      this._pacingStats = extractPacingStats(snapshot, this._pacingStats);
      this._memoryStats = extractMemoryStats(snapshot, this._memoryStats);
      this._gpuMemoryStats = extractGpuMemoryStats(snapshot, this._gpuMemoryStats);
      this._shaderStreamingStats = extractShaderStreamingStats(
        snapshot,
        this._shaderStreamingStats
      );
      this._swapchainColorStats = extractSwapchainColorStats(
        snapshot,
        this._swapchainColorStats,
        this._compatStats
      );
      this._mailboxStats = extractMailboxStats(snapshot, this._mailboxStats);
      this._audioStats = extractAudioStats(snapshot, this._audioStats);
      this._opfsStats = extractOpfsStats(snapshot, this._opfsStats);

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
      this._pacingStats = createDefaultPacingStats();
      this._memoryStats = createDefaultMemoryStats();
      this._gpuMemoryStats = createDefaultGpuMemoryStats();
      this._shaderStreamingStats = createDefaultShaderStreamingStats();
      this._swapchainColorStats = createDefaultSwapchainColorStats();
      this._mailboxStats = createDefaultMailboxStats();
      this._audioStats = createDefaultAudioStats();
      this._opfsStats = createDefaultOpfsStats();
      this._missingShaders = [];
      this._exportCount = 0;

      this._render();
      return this.getState();
    }

    /**
     * Build and return a sanitized JSON string of the current missing-shader shopping list
     * and live M4 diagnostics, invoking `onExportMissingShaders(jsonText, payload)` if configured.
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
        pacingStats: this._pacingStats,
        memoryStats: this._memoryStats,
        gpuMemoryStats: this._gpuMemoryStats,
        shaderStreamingStats: this._shaderStreamingStats,
        swapchainColorStats: this._swapchainColorStats,
        mailboxStats: this._mailboxStats,
        audioStats: this._audioStats,
        opfsStats: this._opfsStats,
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
     * Alias of `exportMissingShadersJson()` returning the full sanitized M4 diagnostics JSON.
     *
     * @returns {string}
     */
    exportDiagnosticsJson() {
      return this.exportMissingShadersJson();
    }

    /**
     * Toggle the collapsible diagnostics body (or set collapsed explicitly when given a boolean).
     * Syncs deferred body DOM immediately when expanded.
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
      if (!this._collapsed && this._visible && this._bodyDirty) {
        this._render();
      }
      return this.getState();
    }

    /** Show the panel root element and flush any deferred DOM updates. */
    show() {
      if (this._destroyed) {
        return this.getState();
      }
      this._applyVisibility(true);
      if (this._headerDirty || (!this._collapsed && this._bodyDirty)) {
        this._render();
      }
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
        fps30ComplianceRate: this._pacingStats.fps30ComplianceRate,
        frameJitterMs: this._pacingStats.frameJitterMs,
        jsHeapUsedMb: this._memoryStats.jsHeapUsedMb,
        vramBudgetUtilization: this._memoryStats.vramBudgetUtilization,
        bindGroupHitRate: this._memoryStats.bindGroupHitRate,
        mailboxLanes: this._mailboxStats.mailboxLanes,
        activeMailboxLanes: this._mailboxStats.activeMailboxLanes,
        peakMailboxLanes: this._mailboxStats.peakMailboxLanes,
        rangeCacheHitRatePct: this._mailboxStats.rangeCacheHitRatePct,
        audioOutputState: this._audioStats.audioOutputState,
        guestRingAttached: this._audioStats.guestRingAttached,
        guestFramesDrained: this._audioStats.guestFramesDrained,
        framesDrained: this._audioStats.framesDrained,
        opfsHits: this._opfsStats.opfsHits,
        opfsMisses: this._opfsStats.opfsMisses,
        opfsWrites: this._opfsStats.opfsWrites,
        opfsInvalidations: this._opfsStats.opfsInvalidations,
        bytesFromOpfs: this._opfsStats.bytesFromOpfs,
        bytesFromNetwork: this._opfsStats.bytesFromNetwork,
        opfsHitRatePct: this._opfsStats.opfsHitRatePct,
        pacingStats: { ...this._pacingStats },
        memoryStats: { ...this._memoryStats },
        gpuMemoryStats: { ...this._gpuMemoryStats },
        shaderStreamingStats: { ...this._shaderStreamingStats },
        swapchainColorStats: { ...this._swapchainColorStats },
        swapchainColorMode: this._swapchainColorStats.swapchainColorMode,
        liveBufferBytes: this._gpuMemoryStats.liveBufferBytes,
        liveTextureBytes: this._gpuMemoryStats.liveTextureBytes,
        peakBufferBytes: this._gpuMemoryStats.peakBufferBytes,
        peakTextureBytes: this._gpuMemoryStats.peakTextureBytes,
        stagingPoolBytes: this._gpuMemoryStats.stagingPoolBytes,
        totalLiveGpuBytes: this._gpuMemoryStats.totalLiveGpuBytes,
        wasmLinearMemoryBytes: this._gpuMemoryStats.wasmLinearMemoryBytes,
        rangedShaderReady: this._shaderStreamingStats.rangedShaderReady,
        rangedShaderFetches: this._shaderStreamingStats.rangedShaderFetches,
        rangedShaderErrors: this._shaderStreamingStats.rangedShaderErrors,
        asyncShaderPendingCount: this._shaderStreamingStats.asyncShaderPendingCount,
        rangedPackActive: this._shaderStreamingStats.rangedPackActive,
        mailboxStats: { ...this._mailboxStats },
        audioStats: { ...this._audioStats },
        opfsStats: { ...this._opfsStats },
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

    /**
     * Format a concise, sanitized single-line diagnostics summary string.
     * Gracefully omits absent/null optional pacing, memory/VRAM, and mailbox metrics.
     *
     * @returns {string}
     */
    formatSummary() {
      return formatSummary({
        boot: this._boot,
        frameStats: this._frameStats,
        fallbackStats: this._fallbackStats,
        adapterStats: this._adapterStats,
        pacingStats: this._pacingStats,
        memoryStats: this._memoryStats,
        mailboxStats: this._mailboxStats,
        audioStats: this._audioStats,
        opfsStats: this._opfsStats,
      });
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
    POST_PRESENT_BOOT_STAGES,
    ALL_BOOT_STAGE_ORDER,
    PLAYABILITY_STATE,
    STAGE_NAMES,
    ALLOWED_SWAPCHAIN_FORMATS,
    ALLOWED_SWAPCHAIN_COLOR_MODES,
    ALLOWED_AUDIO_OUTPUT_STATES,
    sanitizeSwapchainFormat,
    sanitizeSwapchainColorMode,
    sanitizeGpuAdapter,
    sanitizeUnitRatio,
    sanitizePositiveInt,
    sanitizePercentage,
    sanitizeAudioOutputState,
    formatDurationMs,
    formatOnePercentLowFps,
    formatP99FrameMs,
    formatRatioPct,
    formatFrameJitterMs,
    formatMemoryMb,
    formatMemoryBytes,
    formatMailboxLanes,
    formatAudioFrameCount,
    formatOpfsHitRate,
    formatOpfsBytes,
    formatSummary,
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
