// Guest boot milestone tracer and unresolved-import / trap collector for playgta6web.
// Zero external dependencies; clean ES module compatible with <script type="module">
// and Node.js ESM imports. Turns runtime log lines (`[boot]`, `[wasm]`, `[trap]`,
// `[unimpl]`, `[import]`) into ordered boot milestones and structured trap records.
// Every stored and exported string is sanitized so summaries never leak host paths
// or credentials.

import { sanitizeString, sanitizeValue } from './crash_report.js';

export const DEFAULT_MAX_TRAPS = 256;
export const DEFAULT_SLOW_THRESHOLD_MS = null;
export const CHROME_TRACE_CATEGORY = 'playgta6.boot';
export const DEFAULT_TRACE_PID = 1;
export const DEFAULT_TRACE_TID = 1;
export const MAX_MESSAGE_LENGTH = 512;
export const MAX_GPU_ADAPTER_LENGTH = 96;
export const STAGE_NOT_STARTED = 'NOT_STARTED';

/** Guest boot milestones in the order the real boot path reaches them. */
export const BOOT_MILESTONES = Object.freeze([
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

/** Optional post-present boot milestones that do not affect core BOOT_MILESTONES ordering or progress percentages. */
export const OPTIONAL_BOOT_MILESTONES = Object.freeze(['FIRST_VISIBLE_FRAME']);

/** Post-visible fast-boot milestones tracking progression from splash through main menu to playable in-game. */
export const FAST_BOOT_MILESTONES = Object.freeze([
  'START_SCREEN',
  'INTRO_VIDEO',
  'MAIN_MENU',
  'IN_GAME',
]);

/** Complete ordered list of all core, optional, and fast-boot milestones. */
export const ALL_BOOT_MILESTONES = Object.freeze([
  ...BOOT_MILESTONES,
  ...OPTIONAL_BOOT_MILESTONES,
  ...FAST_BOOT_MILESTONES,
]);

const POST_PRESENT_MILESTONES = Object.freeze([...OPTIONAL_BOOT_MILESTONES, ...FAST_BOOT_MILESTONES]);
const ALL_MILESTONES = ALL_BOOT_MILESTONES;
const OK_HUD_MILESTONES = new Set(['FIRST_PRESENT', ...POST_PRESENT_MILESTONES]);

/** Trap kinds mirroring the recompiler/runtime fault helpers. */
export const TRAP_KIND = Object.freeze({
  UNIMPLEMENTED_OPCODE: 'UNIMPLEMENTED_OPCODE',
  UNRESOLVED_IMPORT: 'UNRESOLVED_IMPORT',
  INDIRECT_CALL_MISS: 'INDIRECT_CALL_MISS',
  DIVIDE_ERROR: 'DIVIDE_ERROR',
  GUEST_TRAP: 'GUEST_TRAP',
  WASM_TRAP: 'WASM_TRAP',
});

const TRAP_KIND_VALUES = Object.freeze(Object.values(TRAP_KIND));
const MILESTONE_INDEX = new Map(ALL_MILESTONES.map((name, idx) => [name, idx]));
const TRAP_KIND_SET = new Set(TRAP_KIND_VALUES);

// Unresolved imports are logged once before the runtime decides whether to abort,
// so they are non-fatal by default; every other trap kind halts the guest.
const NON_FATAL_BY_DEFAULT = new Set([TRAP_KIND.UNRESOLVED_IMPORT]);

const LOG_TAG_RE = /\[(wasm|boot|trap|unimpl|import)\]\s*(.*)$/i;
const MILESTONE_TOKEN_RE = new RegExp(`\\b(${ALL_MILESTONES.join('|')})\\b`, 'gi');
const TRAP_KIND_TOKEN_RE = new RegExp(`\\b(${TRAP_KIND_VALUES.join('|')})\\b`, 'i');
const FAILURE_RE = /\b(?:fail(?:ed|s|ure)?|error|unresolved|missing|unimplemented|abort(?:ed)?)\b/i;
const HEX_RE = /\b0x([0-9a-f]{1,16})\b/gi;
const LABELED_VA_RE = /\b(?:va|rip|pc|at|addr|address|target)\s*[=:]?\s*0x([0-9a-f]{1,16})\b/i;
const NON_VA_PREFIX_RE = /\b(?:slot|rva|size|len|length|code|hr|hresult|status)\s*[=:]?\s*$/i;
const IMPORT_REF_RE =
  /\b([A-Za-z0-9_][A-Za-z0-9_.-]*\.(?:dll|exe|sys|lib))!(#\d+|[A-Za-z_?@$][A-Za-z0-9_?@$]*)/i;
const WASM_TRAP_RE =
  /\b(?:RuntimeError|unreachable|out of bounds|trap(?:ped)?|integer (?:divide by zero|overflow)|indirect call|call stack exhausted|stack overflow)\b/i;
const ANSI_ESCAPE_RE = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;
const GPU_ADAPTER_WHITESPACE_CTRL_RE = /[\r\n\t]+/g;
const GPU_ADAPTER_CONTROL_CHAR_RE = /[\u0000-\u001f\u007f-\u009f]/g;
const GPU_ADAPTER_LOWER_POSIX_PATH_RE =
  /(^|[\s"'`(\[={<,:;|])(\/(?:users|library|applications|volumes|home|root|tmp|var|etc|opt|usr|mnt|media|private|srv|proc|sys|dev|run|cygdrive)(?:\/[^\s"'`<>)\]},;]+)?)(?=[\s"'`<>)\]},;]|$)/gi;
const GPU_ADAPTER_BACKSLASH_TOKEN_RE = /[^\s"'`<>|*?)\]},;]*\\[^\s"'`<>|*?)\]},;]*/g;
const REDACTED_TOKEN_RE = /\[REDACTED(?:_[A-Z0-9_]+)?\]/g;

const MILESTONE_HEURISTICS = Object.freeze([
  [
    'PGIM_LOAD',
    /\bpgim\b.*\b(?:load(?:ed|ing)?|mapped|mount(?:ed)?|open(?:ed)?)\b|\b(?:load(?:ed|ing)?|mapped|mount(?:ed)?|open(?:ed)?)\b.*\bpgim\b/i,
  ],
  ['TLS_INIT', /\btls\b.*\b(?:init(?:iali[sz]ed|iali[sz]ing)?|callbacks?|ready|setup|alloc(?:ated)?)\b/i],
  ['CRT_INITTERM', /\b_?initterm(?:_e)?\b|\bcrt\s+init(?:iali[sz]ed)?\b/i],
  [
    'ENTRY_CALL',
    /\b(?:call(?:ing|ed)?|enter(?:ing|ed)?|jump(?:ing|ed)? to|invok(?:ing|ed))\s+(?:the\s+)?(?:guest\s+)?entry(?:[\s_-]*point)?\b|\bmainCRTStartup\b|\bWinMain\b/i,
  ],
  ['XMEM_INIT', /\bxmem(?:[\s_-]*init\w*|SetAllocationHooks|Alloc\w*|\s+(?:ready|initiali[sz]ed))\b/i],
  [
    'GDK_INIT',
    /\bXGameRuntimeInitialize\b|\bgdk(?:[\s_-]*runtime)?[\s_-]*init\w*\b|\bXGameRuntime\s+(?:ready|initiali[sz]ed)\b/i,
  ],
  ['D3D12_CREATE_DEVICE', /\bD3D12(?:Xbox)?CreateDevice\b|\bd3d12\s+device\s+(?:created|ready)\b/i],
  ['BINK_OPEN', /\bBinkOpen\w*\b|\bbink\s+(?:open(?:ed)?|stream\s+opened)\b/i],
  ['FIRST_PRESENT', /\bfirst[\s_-]*(?:swapchain[\s_-]*)?(?:present(?:ed)?|frame)\b/i],
  ['FIRST_VISIBLE_FRAME', /\bfirst[\s_-]*(?:visible|non[\s_-]*black)[\s_-]*frame\b/i],
  ['START_SCREEN', /\bstart[\s_-]*screen\b|\bpress[\s_-]*(?:any[\s_-]*button|start)\b/i],
  ['INTRO_VIDEO', /\bintro[\s_-]*(?:video|movie|cinematic|bink)\b|\bopening[\s_-]*(?:video|cinematic)\b/i],
  ['MAIN_MENU', /\bmain[\s_-]*menu\b|\btitle[\s_-]*menu\b/i],
  ['IN_GAME', /\bin[\s_-]*game\b|\bgameplay[\s_-]*(?:started|active|ready|entered)\b/i],
]);

function defaultNow() {
  if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
    return performance.now();
  }
  return Date.now();
}

function clip(text, max = MAX_MESSAGE_LENGTH) {
  const safe = sanitizeString(text);
  return safe.length > max ? `${safe.slice(0, max - 3)}...` : safe;
}

/**
 * Sanitizes a GPU adapter label (or `{ vendor, architecture }` object) into a
 * lowercase `vendor / architecture` string of at most `MAX_GPU_ADAPTER_LENGTH`
 * (96) characters with control characters, ANSI escapes, host paths, and
 * credentials stripped.
 *
 * @param {unknown} adapterLabel
 * @returns {string|null}
 */
export function sanitizeGpuAdapter(adapterLabel) {
  if (adapterLabel === null || adapterLabel === undefined) {
    return null;
  }

  let raw = null;
  if (typeof adapterLabel === 'string') {
    raw = adapterLabel;
  } else if (typeof adapterLabel === 'object' && !Array.isArray(adapterLabel)) {
    const vendor = typeof adapterLabel.vendor === 'string' ? adapterLabel.vendor.trim() : '';
    const arch = typeof adapterLabel.architecture === 'string' ? adapterLabel.architecture.trim() : '';
    if (vendor && arch) {
      raw = `${vendor} / ${arch}`;
    } else if (vendor || arch) {
      raw = vendor || arch;
    }
  }

  if (typeof raw !== 'string' || raw.length === 0) {
    return null;
  }

  let out = raw
    .replace(ANSI_ESCAPE_RE, ' ')
    .replace(GPU_ADAPTER_WHITESPACE_CTRL_RE, ' ')
    .replace(GPU_ADAPTER_CONTROL_CHAR_RE, '');

  out = sanitizeString(out);
  out = out
    .replace(GPU_ADAPTER_LOWER_POSIX_PATH_RE, '$1')
    .replace(GPU_ADAPTER_BACKSLASH_TOKEN_RE, '')
    .replace(REDACTED_TOKEN_RE, '')
    .replace(/\(\s*\)/g, '')
    .replace(/\s+/g, ' ')
    .replace(/\s*\/\s*/g, ' / ')
    .replace(/(?:\s*\/\s*){2,}/g, ' / ')
    .replace(/^[\s/]+|[\s/]+$/g, '')
    .toLowerCase();

  if (out.length > MAX_GPU_ADAPTER_LENGTH) {
    out = out.slice(0, MAX_GPU_ADAPTER_LENGTH).replace(/[\s/]+$/, '');
  }

  return out.length > 0 ? out : null;
}

/**
 * Normalizes a guest virtual address (number, bigint, or hex string) to `0x...` lowercase hex.
 *
 * @param {unknown} value
 * @returns {string|null}
 */
export function normalizeGuestVa(value) {
  try {
    if (typeof value === 'bigint') {
      return value >= 0n ? `0x${value.toString(16)}` : null;
    }
    if (typeof value === 'number') {
      return Number.isSafeInteger(value) && value >= 0 ? `0x${value.toString(16)}` : null;
    }
    if (typeof value === 'string') {
      const m = /^\s*0x([0-9a-f]{1,16})\s*$/i.exec(value);
      return m ? `0x${BigInt(`0x${m[1]}`).toString(16)}` : null;
    }
  } catch {
    return null;
  }
  return null;
}

/**
 * Extracts the most likely guest VA from free text. Prefers labeled addresses
 * (`va=0x...`, `at 0x...`, `rip=0x...`) and skips import slot / RVA offsets.
 *
 * @param {string} text
 * @returns {string|null}
 */
export function extractGuestVa(text) {
  if (typeof text !== 'string' || text.length === 0) {
    return null;
  }
  const labeled = LABELED_VA_RE.exec(text);
  if (labeled) {
    return normalizeGuestVa(`0x${labeled[1]}`);
  }
  HEX_RE.lastIndex = 0;
  let m;
  while ((m = HEX_RE.exec(text)) !== null) {
    const before = text.slice(Math.max(0, m.index - 12), m.index);
    if (!NON_VA_PREFIX_RE.test(before)) {
      return normalizeGuestVa(m[0]);
    }
  }
  return null;
}

/**
 * Extracts a `DLL!Symbol` or ordinal `DLL!#N` import reference from free text.
 *
 * @param {string} text
 * @returns {{ dll: string, symbol: string, ordinal: number|null }|null}
 */
export function extractImportRef(text) {
  if (typeof text !== 'string') {
    return null;
  }
  const m = IMPORT_REF_RE.exec(text);
  if (!m) {
    return null;
  }
  const symbol = m[2];
  return {
    dll: m[1],
    symbol,
    ordinal: symbol.startsWith('#') ? Number.parseInt(symbol.slice(1), 10) : null,
  };
}

function classifyTrapText(text) {
  const explicit = TRAP_KIND_TOKEN_RE.exec(text);
  if (explicit) {
    return explicit[1].toUpperCase();
  }
  if (/\bunresolved\s+import\b/i.test(text)) {
    return TRAP_KIND.UNRESOLVED_IMPORT;
  }
  if (/\bunimpl(?:emented)?\b/i.test(text)) {
    return TRAP_KIND.UNIMPLEMENTED_OPCODE;
  }
  if (/\bindirect\b|\bunknown[\s_-]*target\b|\bUnknownTarget\b/i.test(text)) {
    return TRAP_KIND.INDIRECT_CALL_MISS;
  }
  if (/\bdivide\b|\bdivision\b|\bDivideError\b|#DE\b/i.test(text)) {
    return TRAP_KIND.DIVIDE_ERROR;
  }
  if (/\bRuntimeError\b|\bwasm\b|\bunreachable\b/i.test(text)) {
    return TRAP_KIND.WASM_TRAP;
  }
  return TRAP_KIND.GUEST_TRAP;
}

function detectMilestones(text) {
  if (FAILURE_RE.test(text)) {
    return [];
  }
  const found = new Set();
  MILESTONE_TOKEN_RE.lastIndex = 0;
  let m;
  while ((m = MILESTONE_TOKEN_RE.exec(text)) !== null) {
    found.add(m[1].toUpperCase());
  }
  for (const [name, re] of MILESTONE_HEURISTICS) {
    if (re.test(text)) {
      found.add(name);
    }
  }
  return ALL_MILESTONES.filter((name) => found.has(name));
}

function isUnresolvedImportText(text) {
  if (/\bun(?:resolved|bound|implemented)\b|\bmissing\b|\bnot\s+found\b|\bno\s+(?:shim|thunk|export)\b/i.test(text)) {
    return true;
  }
  return !/\b(?:bound|resolved|binding\s+ok)\b/i.test(text);
}

/**
 * Parses one structured runtime log line without mutating any tracer state.
 *
 * @param {unknown} line
 * @returns {null | { type: 'milestone', tag: string, milestones: string[], message: string }
 *   | { type: 'trap', tag: string, trap: { kind: string, guestVa: string|null, dll: string|null,
 *       symbol: string|null, message: string, fatal: boolean } }}
 */
export function parseLogLine(line) {
  if (typeof line !== 'string' || line.length === 0) {
    return null;
  }
  const tagMatch = LOG_TAG_RE.exec(line);
  if (!tagMatch) {
    return null;
  }
  const tag = tagMatch[1].toLowerCase();
  const body = tagMatch[2].trim();

  const trapFrom = (kind, fatal) => {
    const ref = extractImportRef(body);
    return {
      type: 'trap',
      tag,
      trap: {
        kind,
        guestVa: extractGuestVa(body),
        dll: ref ? ref.dll : null,
        symbol: ref ? ref.symbol : null,
        message: body,
        fatal,
      },
    };
  };

  if (tag === 'trap') {
    return trapFrom(classifyTrapText(body), true);
  }
  if (tag === 'unimpl') {
    return trapFrom(TRAP_KIND.UNIMPLEMENTED_OPCODE, true);
  }
  if (tag === 'import') {
    if (!extractImportRef(body) || !isUnresolvedImportText(body)) {
      return null;
    }
    return trapFrom(TRAP_KIND.UNRESOLVED_IMPORT, false);
  }
  if (tag === 'wasm' && WASM_TRAP_RE.test(body)) {
    return trapFrom(TRAP_KIND.WASM_TRAP, true);
  }

  const milestones = detectMilestones(body);
  if (milestones.length === 0) {
    return null;
  }
  return { type: 'milestone', tag, milestones, message: body };
}

class TrapRing {
  constructor(capacity) {
    this._capacity = capacity;
    this.clear();
  }

  push(item) {
    this._items[this._writeIdx] = item;
    this._writeIdx = (this._writeIdx + 1) % this._capacity;
    if (this._count < this._capacity) {
      this._count += 1;
    }
  }

  toArray() {
    const out = new Array(this._count);
    const start = this._count < this._capacity ? 0 : this._writeIdx;
    for (let i = 0; i < this._count; i += 1) {
      out[i] = this._items[(start + i) % this._capacity];
    }
    return out;
  }

  get length() {
    return this._count;
  }

  clear() {
    this._items = new Array(this._capacity);
    this._writeIdx = 0;
    this._count = 0;
  }
}

function resolveMilestone(milestone) {
  const name = typeof milestone === 'string' ? milestone.trim().toUpperCase() : '';
  if (!MILESTONE_INDEX.has(name)) {
    throw new RangeError(`Unknown boot milestone: ${sanitizeString(String(milestone))}`);
  }
  return name;
}

function resolveTrapKind(kind) {
  if (kind === undefined || kind === null || kind === '') {
    return TRAP_KIND.GUEST_TRAP;
  }
  const name = typeof kind === 'string' ? kind.trim().toUpperCase() : '';
  if (!TRAP_KIND_SET.has(name)) {
    throw new RangeError(`Unknown trap kind: ${sanitizeString(String(kind))}`);
  }
  return name;
}

function validateSlowThresholdMs(value) {
  if (value === undefined || value === null) {
    return null;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new RangeError(`slowThresholdMs must be a non-negative finite number or null, got ${value}`);
  }
  return value;
}

function validateTraceId(value, label) {
  if (!Number.isInteger(value) || value < 0) {
    throw new RangeError(`${label} must be a non-negative integer, got ${value}`);
  }
  return value;
}

function msToMicros(ms) {
  return Number.isFinite(ms) ? Math.max(0, Math.round(ms * 1000)) : 0;
}

function clampNonNegativeFiniteMs(value) {
  return typeof value === 'number' && Number.isFinite(value) ? Math.max(0, value) : null;
}

/**
 * Tracks guest boot progress through `BOOT_MILESTONES` and collects guest traps
 * (unimplemented opcodes, unresolved imports, indirect-call misses, ...) into a
 * bounded ring plus a deduplicated unresolved-import frequency table.
 */
export class BootTracer {
  /**
   * @param {object} [options]
   * @param {number} [options.maxTraps=256] Ring capacity for recent trap records.
   * @param {() => number} [options.now] Monotonic clock in milliseconds.
   * @param {number} [options.startMs] Boot start timestamp (defaults to `now()`).
   * @param {number|null} [options.slowThresholdMs=null] Optional inter-milestone delta threshold in ms for flagging slow milestones.
   * @param {{ recordError?: Function, recordLog?: Function }} [options.crashReporter]
   * @param {{ setStatus?: Function, updateMetrics?: Function }} [options.hud]
   */
  constructor(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('BootTracer options must be an object');
    }
    const {
      maxTraps = DEFAULT_MAX_TRAPS,
      now = defaultNow,
      startMs,
      slowThresholdMs = DEFAULT_SLOW_THRESHOLD_MS,
      crashReporter = null,
      hud = null,
    } = options;

    if (!Number.isInteger(maxTraps) || maxTraps < 1) {
      throw new RangeError(`maxTraps must be a positive integer, got ${maxTraps}`);
    }
    if (typeof now !== 'function') {
      throw new TypeError('now must be a function returning milliseconds');
    }
    const validatedSlowThresholdMs = validateSlowThresholdMs(slowThresholdMs);

    this._maxTraps = maxTraps;
    this._now = now;
    this._slowThresholdMs = validatedSlowThresholdMs;
    this._crashReporter = crashReporter && typeof crashReporter === 'object' ? crashReporter : null;
    this._hud = hud && typeof hud === 'object' ? hud : null;
    this._traps = new TrapRing(maxTraps);
    this.reset(startMs);
  }

  get maxTraps() {
    return this._maxTraps;
  }

  get slowThresholdMs() {
    return this._slowThresholdMs;
  }

  get trapCount() {
    return this._trapCount;
  }

  get gpuAdapter() {
    return this._gpuAdapter;
  }

  /**
   * Clears milestones, traps, GPU adapter label, and the import table and restarts the boot clock.
   * Keeps the attached crash reporter and HUD.
   *
   * @param {number} [startMs]
   */
  reset(startMs) {
    const start = typeof startMs === 'number' && Number.isFinite(startMs) ? startMs : this._now();
    this._startMs = Number.isFinite(start) ? start : 0;
    this._lastElapsedMs = 0;
    this._lastMilestoneElapsedMs = 0;
    this._milestones = new Map();
    this._milestoneOrder = [];
    this._highestIdx = -1;
    this._gpuAdapter = null;
    this._traps.clear();
    this._trapCount = 0;
    this._trapSeq = 0;
    this._signatures = new Map();
    this._imports = new Map();
    this._lastTrap = null;
  }

  _elapsed() {
    const raw = this._now() - this._startMs;
    if (Number.isFinite(raw) && raw > this._lastElapsedMs) {
      this._lastElapsedMs = raw;
    }
    return this._lastElapsedMs;
  }

  _progressPct() {
    return this._highestIdx < 0
      ? 0
      : Math.min(100, Math.round(((this._highestIdx + 1) / BOOT_MILESTONES.length) * 100));
  }

  _milestoneElapsedMs(name) {
    const entry = this._milestones.get(name);
    return entry ? clampNonNegativeFiniteMs(entry.elapsedMs) : null;
  }

  _firstPresentMs() {
    return this._milestoneElapsedMs('FIRST_PRESENT');
  }

  _firstVisibleFrameMs() {
    return this._milestoneElapsedMs('FIRST_VISIBLE_FRAME');
  }

  _bootToStartScreenMs() {
    return this._milestoneElapsedMs('START_SCREEN');
  }

  _bootToIntroVideoMs() {
    return this._milestoneElapsedMs('INTRO_VIDEO');
  }

  _bootToMainMenuMs() {
    return this._milestoneElapsedMs('MAIN_MENU');
  }

  _bootToInGameMs() {
    return this._milestoneElapsedMs('IN_GAME');
  }

  _blackLoadingMs() {
    const firstPresentMs = this._firstPresentMs();
    const firstVisibleFrameMs = this._firstVisibleFrameMs();
    if (firstPresentMs === null || firstVisibleFrameMs === null) {
      return null;
    }
    return clampNonNegativeFiniteMs(firstVisibleFrameMs - firstPresentMs);
  }

  _recomputeMilestoneDeltas() {
    let prevElapsedMs = 0;
    let maxElapsedMs = 0;
    for (const name of this._milestoneOrder) {
      const entry = this._milestones.get(name);
      if (!entry) {
        continue;
      }
      entry.deltaFromPrevMs = Math.max(0, entry.elapsedMs - prevElapsedMs);
      entry.slow = this._slowThresholdMs !== null && entry.deltaFromPrevMs > this._slowThresholdMs;
      if (entry.elapsedMs > prevElapsedMs) {
        prevElapsedMs = entry.elapsedMs;
      }
      if (entry.elapsedMs > maxElapsedMs) {
        maxElapsedMs = entry.elapsedMs;
      }
    }
    this._lastMilestoneElapsedMs = maxElapsedMs;
  }

  _notifyHud(text, level) {
    const hud = this._hud;
    if (!hud) {
      return;
    }
    try {
      if (typeof hud.setStatus === 'function') {
        hud.setStatus(text, level);
      } else if (typeof hud.updateMetrics === 'function') {
        hud.updateMetrics({ status: text, level });
      }
    } catch {
      // Telemetry must never break the boot path.
    }
  }

  _forward(method, ...args) {
    const reporter = this._crashReporter;
    if (!reporter || typeof reporter[method] !== 'function') {
      return;
    }
    try {
      reporter[method](...args);
    } catch {
      // Telemetry must never break the boot path.
    }
  }

  /**
   * Records that a boot milestone was reached. The first-reached timestamp is kept;
   * repeats only bump `count` and `lastElapsedMs` unless `options.overwrite` is `true`.
   *
   * @param {string} milestone One of `BOOT_MILESTONES`, `OPTIONAL_BOOT_MILESTONES`, or `FAST_BOOT_MILESTONES`.
   * @param {unknown} [detail] Optional sanitized detail stored on first reach (or overwrite).
   * @param {{ overwrite?: boolean }} [options] Optional control flags.
   * @returns {{ milestone: string, index: number, elapsedMs: number, lastElapsedMs: number,
   *   count: number, detail: any, firstReach: boolean }}
   */
  markMilestone(milestone, detail, options) {
    let overwrite = false;
    if (options !== undefined) {
      if (options === null || typeof options !== 'object' || Array.isArray(options)) {
        throw new TypeError('markMilestone options must be an object');
      }
      if (options.overwrite !== undefined) {
        if (typeof options.overwrite !== 'boolean') {
          throw new TypeError(`overwrite must be a boolean, got ${typeof options.overwrite}`);
        }
        overwrite = options.overwrite;
      }
    }
    return this._markMilestone(milestone, detail, undefined, overwrite);
  }

  _markMilestone(milestone, detail, explicitElapsedMs, overwrite = false) {
    const name = resolveMilestone(milestone);
    let elapsedMs;
    if (explicitElapsedMs !== undefined) {
      elapsedMs = explicitElapsedMs;
      if (elapsedMs > this._lastElapsedMs) {
        this._lastElapsedMs = elapsedMs;
      }
    } else {
      elapsedMs = this._elapsed();
    }
    const index = MILESTONE_INDEX.get(name);
    let entry = this._milestones.get(name);
    const firstReach = entry === undefined;

    if (firstReach) {
      const safeDetail = detail === undefined ? null : sanitizeValue(detail);
      const deltaFromPrevMs = Math.max(0, elapsedMs - this._lastMilestoneElapsedMs);
      if (elapsedMs > this._lastMilestoneElapsedMs) {
        this._lastMilestoneElapsedMs = elapsedMs;
      }
      const slow = this._slowThresholdMs !== null && deltaFromPrevMs > this._slowThresholdMs;
      entry = {
        milestone: name,
        index,
        elapsedMs,
        lastElapsedMs: elapsedMs,
        deltaFromPrevMs,
        slow,
        count: 1,
        detail: safeDetail ?? null,
      };
      this._milestones.set(name, entry);
      this._milestoneOrder.push(name);
      if (index > this._highestIdx) {
        this._highestIdx = index;
      }
      const pct = this._progressPct();
      this._forward('recordLog', 'info', 'boot_tracer', `milestone ${name} reached at ${elapsedMs.toFixed(1)}ms`);
      this._notifyHud(`Boot ${pct}%: ${name}`, OK_HUD_MILESTONES.has(name) ? 'ok' : 'info');
    } else if (overwrite) {
      entry.count += 1;
      entry.elapsedMs = elapsedMs;
      entry.lastElapsedMs = elapsedMs;
      if (detail !== undefined) {
        entry.detail = sanitizeValue(detail) ?? null;
      }
      this._recomputeMilestoneDeltas();
      this._forward('recordLog', 'info', 'boot_tracer', `milestone ${name} overwritten at ${elapsedMs.toFixed(1)}ms`);
    } else {
      entry.count += 1;
      entry.lastElapsedMs = elapsedMs;
    }
    return { ...entry, firstReach };
  }

  _recordPostPresentMilestone(milestone, opts = {}, methodName) {
    if (opts !== undefined && (opts === null || typeof opts !== 'object' || Array.isArray(opts))) {
      throw new TypeError(`${methodName} options must be an object`);
    }
    let explicitElapsedMs;
    if (opts.elapsedMs !== undefined && opts.elapsedMs !== null) {
      if (typeof opts.elapsedMs !== 'number' || !Number.isFinite(opts.elapsedMs) || opts.elapsedMs < 0) {
        throw new RangeError(`elapsedMs must be a non-negative finite number, got ${opts.elapsedMs}`);
      }
      explicitElapsedMs = opts.elapsedMs;
    }
    let overwrite = false;
    if (opts.overwrite !== undefined) {
      if (typeof opts.overwrite !== 'boolean') {
        throw new TypeError(`overwrite must be a boolean, got ${typeof opts.overwrite}`);
      }
      overwrite = opts.overwrite;
    }

    const nonControlKeys = Object.keys(opts).filter((key) => key !== 'elapsedMs' && key !== 'overwrite');
    let detail;
    if (nonControlKeys.length === 0) {
      detail = undefined;
    } else if (nonControlKeys.length === 1 && nonControlKeys[0] === 'detail') {
      detail = opts.detail;
    } else {
      detail = {};
      for (const key of nonControlKeys) {
        detail[key] = opts[key];
      }
    }

    return this._markMilestone(milestone, detail, explicitElapsedMs, overwrite);
  }

  /**
   * Records the optional `FIRST_VISIBLE_FRAME` milestone (first non-black frame after
   * `FIRST_PRESENT`), preserving the first-reached `firstVisibleFrameMs` unless
   * `{ overwrite: true }` is passed, and computing
   * `blackLoadingMs = max(0, firstVisibleFrameMs - firstPresentMs)` when both exist.
   *
   * @param {object} [opts]
   * @param {number} [opts.elapsedMs] Optional explicit non-negative elapsed time in ms.
   * @param {boolean} [opts.overwrite=false] Overwrite existing first-occurrence timestamp and detail.
   * @param {unknown} [opts.detail] Optional sanitized detail metadata.
   * @returns {object} Milestone record enriched with `firstVisibleFrameMs` and `blackLoadingMs`.
   */
  recordFirstVisibleFrame(opts = {}) {
    const entry = this._recordPostPresentMilestone('FIRST_VISIBLE_FRAME', opts, 'recordFirstVisibleFrame');
    return {
      ...entry,
      firstVisibleFrameMs: this._firstVisibleFrameMs(),
      blackLoadingMs: this._blackLoadingMs(),
    };
  }

  /**
   * Records the `START_SCREEN` fast-boot milestone (idempotent first-occurrence timestamp
   * unless `{ overwrite: true }`).
   *
   * @param {object} [opts]
   * @param {number} [opts.elapsedMs] Optional explicit non-negative elapsed time in ms.
   * @param {boolean} [opts.overwrite=false] Overwrite existing first-occurrence timestamp and detail.
   * @param {unknown} [opts.detail] Optional sanitized detail metadata.
   * @returns {object} Milestone record enriched with `bootToStartScreenMs` and `blackLoadingMs`.
   */
  recordStartScreen(opts = {}) {
    const entry = this._recordPostPresentMilestone('START_SCREEN', opts, 'recordStartScreen');
    return {
      ...entry,
      bootToStartScreenMs: this._bootToStartScreenMs(),
      blackLoadingMs: this._blackLoadingMs(),
    };
  }

  /**
   * Records the `INTRO_VIDEO` fast-boot milestone (idempotent first-occurrence timestamp
   * unless `{ overwrite: true }`).
   *
   * @param {object} [opts]
   * @param {number} [opts.elapsedMs] Optional explicit non-negative elapsed time in ms.
   * @param {boolean} [opts.overwrite=false] Overwrite existing first-occurrence timestamp and detail.
   * @param {unknown} [opts.detail] Optional sanitized detail metadata.
   * @returns {object} Milestone record enriched with `bootToIntroVideoMs` and `blackLoadingMs`.
   */
  recordIntroVideo(opts = {}) {
    const entry = this._recordPostPresentMilestone('INTRO_VIDEO', opts, 'recordIntroVideo');
    return {
      ...entry,
      bootToIntroVideoMs: this._bootToIntroVideoMs(),
      blackLoadingMs: this._blackLoadingMs(),
    };
  }

  /**
   * Records the `MAIN_MENU` fast-boot milestone (idempotent first-occurrence timestamp
   * unless `{ overwrite: true }`).
   *
   * @param {object} [opts]
   * @param {number} [opts.elapsedMs] Optional explicit non-negative elapsed time in ms.
   * @param {boolean} [opts.overwrite=false] Overwrite existing first-occurrence timestamp and detail.
   * @param {unknown} [opts.detail] Optional sanitized detail metadata.
   * @returns {object} Milestone record enriched with `bootToMainMenuMs` and `blackLoadingMs`.
   */
  recordMainMenu(opts = {}) {
    const entry = this._recordPostPresentMilestone('MAIN_MENU', opts, 'recordMainMenu');
    return {
      ...entry,
      bootToMainMenuMs: this._bootToMainMenuMs(),
      blackLoadingMs: this._blackLoadingMs(),
    };
  }

  /**
   * Records the `IN_GAME` fast-boot milestone (idempotent first-occurrence timestamp
   * unless `{ overwrite: true }`).
   *
   * @param {object} [opts]
   * @param {number} [opts.elapsedMs] Optional explicit non-negative elapsed time in ms.
   * @param {boolean} [opts.overwrite=false] Overwrite existing first-occurrence timestamp and detail.
   * @param {unknown} [opts.detail] Optional sanitized detail metadata.
   * @returns {object} Milestone record enriched with `bootToInGameMs` and `blackLoadingMs`.
   */
  recordInGame(opts = {}) {
    const entry = this._recordPostPresentMilestone('IN_GAME', opts, 'recordInGame');
    return {
      ...entry,
      bootToInGameMs: this._bootToInGameMs(),
      blackLoadingMs: this._blackLoadingMs(),
    };
  }

  /**
   * Records a sanitized lowercase `vendor / architecture` GPU adapter label
   * (max 96 chars, stripped of control characters and host paths).
   *
   * @param {string|{ vendor?: string, architecture?: string }|null|undefined} adapterLabel
   * @returns {string|null} The stored sanitized GPU adapter label, or `null`.
   */
  recordGpuAdapter(adapterLabel) {
    const sanitized = sanitizeGpuAdapter(adapterLabel);
    this._gpuAdapter = sanitized;
    if (sanitized !== null) {
      this._forward('recordLog', 'info', 'boot_tracer', `gpu adapter: ${sanitized}`);
    }
    return this._gpuAdapter;
  }

  /**
   * Records a guest trap. Missing `guestVa` / `dll` / `symbol` are recovered from `message`.
   *
   * @param {object} trap
   * @param {string} [trap.kind] One of `TRAP_KIND` (defaults to `GUEST_TRAP`).
   * @param {number|bigint|string} [trap.guestVa]
   * @param {string} [trap.dll]
   * @param {string} [trap.symbol]
   * @param {string} [trap.message]
   * @param {boolean} [trap.fatal] Defaults to true for every kind except `UNRESOLVED_IMPORT`.
   * @returns {object} The stored, sanitized trap record.
   */
  recordTrap(trap = {}) {
    if (!trap || typeof trap !== 'object') {
      throw new TypeError('recordTrap requires a trap object');
    }
    const kind = resolveTrapKind(trap.kind);
    const rawMessage = typeof trap.message === 'string' ? trap.message : trap.message == null ? '' : String(trap.message);
    const message = clip(rawMessage);

    let guestVa = normalizeGuestVa(trap.guestVa);
    if (guestVa === null && trap.guestVa === undefined) {
      guestVa = extractGuestVa(rawMessage);
    }

    let dll = typeof trap.dll === 'string' && trap.dll.trim() ? clip(trap.dll.trim(), 128) : null;
    let symbol =
      typeof trap.symbol === 'string' && trap.symbol.trim()
        ? clip(trap.symbol.trim(), 256)
        : typeof trap.symbol === 'number' && Number.isInteger(trap.symbol) && trap.symbol >= 0
          ? `#${trap.symbol}`
          : null;
    if (dll === null && symbol === null) {
      const ref = extractImportRef(rawMessage);
      if (ref) {
        dll = ref.dll;
        symbol = ref.symbol;
      }
    }
    const importName = symbol !== null ? (dll !== null ? `${dll}!${symbol}` : symbol) : null;
    const fatal = typeof trap.fatal === 'boolean' ? trap.fatal : !NON_FATAL_BY_DEFAULT.has(kind);
    const elapsedMs = this._elapsed();

    const signature = `${kind}|${guestVa ?? ''}|${(dll ?? '').toLowerCase()}!${symbol ?? ''}`;
    const occurrences = (this._signatures.get(signature) ?? 0) + 1;
    this._signatures.set(signature, occurrences);

    this._trapSeq += 1;
    this._trapCount += 1;
    const record = {
      seq: this._trapSeq,
      kind,
      guestVa,
      dll,
      symbol,
      import: importName,
      message,
      fatal,
      elapsedMs,
      stage: this._stage(),
      count: occurrences,
    };
    this._traps.push(record);
    this._lastTrap = record;

    if (kind === TRAP_KIND.UNRESOLVED_IMPORT && symbol !== null) {
      const key = `${(dll ?? '').toLowerCase()}!${symbol}`;
      const existing = this._imports.get(key);
      if (existing) {
        existing.count += 1;
        existing.lastElapsedMs = elapsedMs;
        if (existing.guestVa === null && guestVa !== null) {
          existing.guestVa = guestVa;
        }
      } else {
        this._imports.set(key, {
          name: importName,
          dll,
          symbol,
          ordinal: symbol.startsWith('#') ? Number.parseInt(symbol.slice(1), 10) : null,
          guestVa,
          count: 1,
          firstElapsedMs: elapsedMs,
          lastElapsedMs: elapsedMs,
        });
      }
    }

    const where = `${guestVa ? ` @ ${guestVa}` : ''}${importName ? ` ${importName}` : ''}`;
    const summaryText = `${kind}${where}${message ? `: ${message}` : ''}`;
    if (fatal) {
      this._forward('recordError', 'wasm_trap', { name: 'GuestTrap', message: summaryText }, {
        kind,
        guestVa,
        import: importName,
        stage: record.stage,
        elapsedMs,
      });
    } else {
      this._forward('recordLog', 'warn', 'boot_tracer', summaryText);
    }
    this._notifyHud(`Trap ${kind}${where}`, fatal ? 'error' : 'warn');

    return { ...record };
  }

  /**
   * Parses a structured runtime log line and applies it as milestones or a trap record.
   *
   * @param {unknown} line
   * @returns {null | { type: 'milestone', milestones: object[] } | { type: 'trap', trap: object }}
   */
  ingestLogLine(line) {
    const parsed = parseLogLine(line);
    if (!parsed) {
      return null;
    }
    if (parsed.type === 'trap') {
      return { type: 'trap', trap: this.recordTrap(parsed.trap) };
    }
    const detail = { source: parsed.tag, line: clip(parsed.message, 256) };
    return { type: 'milestone', milestones: parsed.milestones.map((name) => this.markMilestone(name, detail)) };
  }

  _stage() {
    return this._highestIdx < 0 ? STAGE_NOT_STARTED : ALL_MILESTONES[this._highestIdx];
  }

  /**
   * Returns reached milestones in chronological first-reach order with inter-milestone
   * delta timings and cumulative elapsed time from boot start.
   *
   * @param {object} [options]
   * @param {number|null} [options.slowThresholdMs] Optional override for slow-milestone flagging.
   * @returns {Array<{ name: string, index: number, atMs: number, deltaMs: number,
   *   deltaFromPrevMs: number, cumulativeMs: number, slow: boolean }>}
   */
  getMilestoneDeltas(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('getMilestoneDeltas options must be an object');
    }
    const threshold =
      options.slowThresholdMs !== undefined
        ? validateSlowThresholdMs(options.slowThresholdMs)
        : this._slowThresholdMs;

    return this._milestoneOrder.map((name) => {
      const entry = this._milestones.get(name);
      const slow = threshold !== null && entry.deltaFromPrevMs > threshold;
      return {
        name: entry.milestone,
        index: entry.index,
        atMs: entry.elapsedMs,
        deltaMs: entry.deltaFromPrevMs,
        deltaFromPrevMs: entry.deltaFromPrevMs,
        cumulativeMs: entry.elapsedMs,
        slow,
      };
    });
  }

  /**
   * Exports a sanitized Chromium `chrome://tracing` / Perfetto JSON trace object
   * containing `ph: 'X'` boot phase spans and `ph: 'i'` instant milestone (and optional trap)
   * markers with timestamps and durations in microseconds.
   *
   * @param {object} [options]
   * @param {number} [options.pid=1] Process id for trace events.
   * @param {number} [options.tid=1] Thread id for trace events.
   * @param {string} [options.category='playgta6.boot'] Trace event category.
   * @param {boolean} [options.includeTraps=true] Whether to include trap instant markers.
   * @param {boolean} [options.includeMetadata=false] Whether to include `ph: 'M'` process/thread name events.
   * @param {number|null} [options.slowThresholdMs] Optional override for slow-milestone tagging.
   * @returns {{ traceEvents: object[], displayTimeUnit: 'ms', metadata: object }}
   */
  toChromeTraceEvents(options = {}) {
    if (options !== undefined && (options === null || typeof options !== 'object')) {
      throw new TypeError('toChromeTraceEvents options must be an object');
    }
    const {
      pid = DEFAULT_TRACE_PID,
      tid = DEFAULT_TRACE_TID,
      category = CHROME_TRACE_CATEGORY,
      includeTraps = true,
      includeMetadata = false,
      slowThresholdMs,
    } = options;

    validateTraceId(pid, 'pid');
    validateTraceId(tid, 'tid');
    const threshold =
      slowThresholdMs !== undefined ? validateSlowThresholdMs(slowThresholdMs) : this._slowThresholdMs;
    const safeCategory =
      typeof category === 'string' && category.trim().length > 0
        ? clip(category.trim(), 128)
        : CHROME_TRACE_CATEGORY;

    const metadataEvents = [];
    if (includeMetadata) {
      metadataEvents.push(
        {
          name: 'process_name',
          ph: 'M',
          pid,
          tid: 0,
          args: { name: 'playgta6web' },
        },
        {
          name: 'thread_name',
          ph: 'M',
          pid,
          tid,
          args: { name: 'guest-boot' },
        }
      );
    }

    const timedEvents = [];
    let prevAtMs = 0;
    let prevUs = 0;

    for (const name of this._milestoneOrder) {
      const entry = this._milestones.get(name);
      const atUs = msToMicros(entry.elapsedMs);
      const durUs = Math.max(0, atUs - prevUs);
      const slow = threshold !== null && entry.deltaFromPrevMs > threshold;

      timedEvents.push({
        name: `phase:${entry.milestone}`,
        cat: safeCategory,
        ph: 'X',
        ts: prevUs,
        dur: durUs,
        pid,
        tid,
        args: {
          milestone: entry.milestone,
          index: entry.index,
          fromMs: prevAtMs,
          atMs: entry.elapsedMs,
          deltaMs: entry.deltaFromPrevMs,
          deltaFromPrevMs: entry.deltaFromPrevMs,
          cumulativeMs: entry.elapsedMs,
          slow,
          count: entry.count,
        },
      });

      const markerArgs = {
        milestone: entry.milestone,
        index: entry.index,
        atMs: entry.elapsedMs,
        deltaMs: entry.deltaFromPrevMs,
        deltaFromPrevMs: entry.deltaFromPrevMs,
        cumulativeMs: entry.elapsedMs,
        slow,
        count: entry.count,
        lastElapsedMs: entry.lastElapsedMs,
      };
      if (entry.detail !== null) {
        markerArgs.detail = entry.detail;
      }

      timedEvents.push({
        name: entry.milestone,
        cat: safeCategory,
        ph: 'i',
        s: 't',
        ts: atUs,
        pid,
        tid,
        args: markerArgs,
      });

      prevAtMs = entry.elapsedMs;
      prevUs = atUs;
    }

    if (includeTraps) {
      for (const trap of this._traps.toArray()) {
        timedEvents.push({
          name: `trap:${trap.kind}`,
          cat: safeCategory,
          ph: 'i',
          s: 't',
          ts: msToMicros(trap.elapsedMs),
          pid,
          tid,
          args: {
            seq: trap.seq,
            kind: trap.kind,
            guestVa: trap.guestVa,
            dll: trap.dll,
            symbol: trap.symbol,
            import: trap.import,
            fatal: trap.fatal,
            stage: trap.stage,
            count: trap.count,
            message: trap.message,
          },
        });
      }
    }

    timedEvents.sort((a, b) => a.ts - b.ts);

    const firstPresentMs = this._firstPresentMs();
    const firstVisibleFrameMs = this._firstVisibleFrameMs();
    const blackLoadingMs = this._blackLoadingMs();
    const bootToStartScreenMs = this._bootToStartScreenMs();
    const bootToIntroVideoMs = this._bootToIntroVideoMs();
    const bootToMainMenuMs = this._bootToMainMenuMs();
    const bootToInGameMs = this._bootToInGameMs();

    const metadata = {
      source: 'playgta6web.boot_tracer',
      stage: this._stage(),
      progressPct: this._progressPct(),
      readyForSplash: this._milestones.has('FIRST_PRESENT'),
      elapsedMs: this._elapsed(),
      firstPresentMs,
      firstVisibleFrameMs,
      blackLoadingMs,
      bootToStartScreenMs,
      bootToIntroVideoMs,
      bootToMainMenuMs,
      bootToInGameMs,
      gpuAdapter: this._gpuAdapter,
      milestonesReached: this._milestones.size,
      trapCount: this._trapCount,
    };
    if (threshold !== null) {
      metadata.slowThresholdMs = threshold;
    }

    return sanitizeValue({
      traceEvents: [...metadataEvents, ...timedEvents],
      displayTimeUnit: 'ms',
      metadata,
    });
  }

  /**
   * Returns the sanitized boot progress and trap summary.
   */
  getSummary() {
    const milestones = BOOT_MILESTONES.map((name, index) => {
      const entry = this._milestones.get(name);
      return entry
        ? {
            milestone: name,
            index,
            reached: true,
            elapsedMs: entry.elapsedMs,
            lastElapsedMs: entry.lastElapsedMs,
            deltaFromPrevMs: entry.deltaFromPrevMs,
            slow: entry.slow,
            count: entry.count,
            detail: entry.detail,
          }
        : {
            milestone: name,
            index,
            reached: false,
            elapsedMs: null,
            lastElapsedMs: null,
            deltaFromPrevMs: null,
            slow: false,
            count: 0,
            detail: null,
          };
    });

    for (const name of POST_PRESENT_MILESTONES) {
      const entry = this._milestones.get(name);
      if (entry) {
        milestones.push({
          milestone: name,
          index: entry.index,
          reached: true,
          elapsedMs: entry.elapsedMs,
          lastElapsedMs: entry.lastElapsedMs,
          deltaFromPrevMs: entry.deltaFromPrevMs,
          slow: entry.slow,
          count: entry.count,
          detail: entry.detail,
        });
      }
    }

    const slowMilestones = this.getMilestoneDeltas().filter((delta) => delta.slow);

    const unresolvedImports = Array.from(this._imports.values(), (entry) => ({ ...entry })).sort(
      (a, b) => b.count - a.count || a.firstElapsedMs - b.firstElapsedMs || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0)
    );

    const firstPresentMs = this._firstPresentMs();
    const firstVisibleFrameMs = this._firstVisibleFrameMs();
    const blackLoadingMs = this._blackLoadingMs();
    const bootToStartScreenMs = this._bootToStartScreenMs();
    const bootToIntroVideoMs = this._bootToIntroVideoMs();
    const bootToMainMenuMs = this._bootToMainMenuMs();
    const bootToInGameMs = this._bootToInGameMs();

    return sanitizeValue({
      stage: this._stage(),
      progressPct: this._progressPct(),
      readyForSplash: this._milestones.has('FIRST_PRESENT'),
      elapsedMs: this._elapsed(),
      firstPresentMs,
      firstVisibleFrameMs,
      blackLoadingMs,
      bootToStartScreenMs,
      bootToIntroVideoMs,
      bootToMainMenuMs,
      bootToInGameMs,
      gpuAdapter: this._gpuAdapter,
      milestonesReached: this._milestones.size,
      milestones,
      slowThresholdMs: this._slowThresholdMs,
      slowMilestones,
      trapCount: this._trapCount,
      uniqueTrapCount: this._signatures.size,
      lastTrap: this._lastTrap ? { ...this._lastTrap } : null,
      unresolvedImports,
      recentTraps: this._traps.toArray().map((record) => ({ ...record })),
    });
  }

  /**
   * Returns a sanitized telemetry snapshot (alias for `getSummary()`).
   */
  getSnapshot() {
    return this.getSummary();
  }

  /**
   * Returns the sanitized summary object for standard `JSON.stringify(tracer)` serialization.
   */
  toJSON() {
    return this.getSummary();
  }

  /**
   * Serializes `getSummary()` to JSON.
   *
   * @param {number|string} [space=2]
   * @returns {string}
   */
  toJson(space = 2) {
    return JSON.stringify(this.getSummary(), null, space);
  }
}

export default BootTracer;
