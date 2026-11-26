'use strict';

// WebGPU RenderPipeline, ComputePipeline, and BindGroupLayout cache manager.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6PipelineCache)
// and Node.js (module.exports). Interfaces cleanly with GPUDevice, GPUShaderModule,
// WGPK 64-bit shader hashes (web/workers/wgsl_pack_loader.js), and S6 reflection metadata.
(function (root) {
  const DEFAULT_MAX_RENDER_PIPELINES = 2048;
  const DEFAULT_MAX_COMPUTE_PIPELINES = 1024;
  const DEFAULT_MAX_BIND_GROUP_LAYOUTS = 512;
  const DEFAULT_MAX_FAST_LOOKUP_ENTRIES = 2048;
  const DEFAULT_PREWARM_CONCURRENCY = 4;
  const DEFAULT_MAX_ERROR_HISTORY = 32;
  const DEFAULT_MAX_QUARANTINED_KEYS = 512;
  const DEFAULT_MAX_NEGATIVE_CACHE_ENTRIES = DEFAULT_MAX_QUARANTINED_KEYS;
  const MAX_PIPELINE_ERROR_MESSAGE_LENGTH = 512;
  const MAX_PIPELINE_ERROR_KEY_LENGTH = 64;
  const DEFAULT_MAX_VERTEX_BUFFERS = 8;
  const DEFAULT_MAX_VERTEX_ATTRIBUTES = 16;
  const DEFAULT_MAX_VERTEX_BUFFER_STRIDE = 2048;
  const DEFAULT_COLOR_FORMAT = 'bgra8unorm';
  const FRAME1_VERTEX_STRIDE_24B = 24;
  const FRAME1_CBV_MIN_BINDING_SIZE = 256;
  const DEFAULT_COMPILE_BUDGET_MS = 33.333;
  const DEFAULT_COMPILE_LATENCY_WINDOW_SIZE = 128;

  const PIPELINE_PRIORITY_TIERS = Object.freeze({
    CRITICAL: 'critical',
    HIGH: 'high',
    NORMAL: 'normal',
    LOW: 'low',
  });

  const PIPELINE_PRIORITY_RANKS = Object.freeze({
    low: 0,
    normal: 1,
    high: 2,
    critical: 3,
  });

  const PIPELINE_PRIORITY_ALIASES = Object.freeze({
    critical: 'critical',
    urgent: 'critical',
    demand: 'critical',
    drawn: 'critical',
    high: 'high',
    normal: 'normal',
    default: 'normal',
    medium: 'normal',
    low: 'low',
    background: 'low',
    prewarm: 'low',
    warmup: 'low',
    idle: 'low',
  });

  const PIPELINE_FAILURE_REASON = Object.freeze({
    VALIDATION: 'validation',
    WGSL_COMPILE: 'wgsl_compile',
    LAYOUT_MISMATCH: 'layout_mismatch',
    UNSUPPORTED_FEATURE: 'unsupported_feature',
    DEVICE_LOST: 'device_lost',
    NULL_PIPELINE: 'null_pipeline',
    OTHER: 'other',
  });

  function createEmptyQuarantineReasonCounts() {
    return {
      validation: 0,
      wgsl_compile: 0,
      layout_mismatch: 0,
      unsupported_feature: 0,
      device_lost: 0,
      null_pipeline: 0,
      other: 0,
    };
  }

  function classifyQuarantineReason(err) {
    if (err && typeof err === 'object') {
      if (err.reason === 'destroyed' || err.reason === 'unknown' || err.name === 'GPUDeviceLostError') {
        return PIPELINE_FAILURE_REASON.DEVICE_LOST;
      }
      if (err.cause && err.cause !== err && classifyQuarantineReason(err.cause) === PIPELINE_FAILURE_REASON.DEVICE_LOST) {
        return PIPELINE_FAILURE_REASON.DEVICE_LOST;
      }
    }
    const msg = typeof err === 'string'
      ? err
      : (err && typeof err.message === 'string' ? err.message : '');
    const name = err && typeof err === 'object' && typeof err.name === 'string' ? err.name : '';

    if (/\b(?:gpu\s*)?device\s+(was\s+)?(lost|destroyed|unavailable)\b|\bgpu\s*device\s*lost\b|\bno\s+gpu\s+device\b/i.test(msg)) {
      return PIPELINE_FAILURE_REASON.DEVICE_LOST;
    }
    if (/\b(?:returned|resolved)\s+(?:no|null|empty|undefined)\s+pipeline\b|\b(?:null|empty)\s+pipeline\b/i.test(msg)) {
      return PIPELINE_FAILURE_REASON.NULL_PIPELINE;
    }
    if (/\blayout\b|\bbind\s*group\b|\bbinding\s+mismatch\b|\bpipeline\s*layout\b|\b@group\s*\(/i.test(msg)) {
      return PIPELINE_FAILURE_REASON.LAYOUT_MISMATCH;
    }
    if (/\bunsupported\b|\bnot\s+supported\b|\bfeature\b|\blimit\b|\bexceeds\s+max\b|\bextension\b|\bstorage\s+texture\s+format\b/i.test(msg)) {
      return PIPELINE_FAILURE_REASON.UNSUPPORTED_FEATURE;
    }
    if (/\bwgsl\b|\btint\b|\bshader\b|\bentry\s*point\b|\bsyntax\b|\bparse\b|\bspir-?v\b|\bnaga\b/i.test(msg)) {
      return PIPELINE_FAILURE_REASON.WGSL_COMPILE;
    }
    if (
      name === 'GPUValidationError' ||
      (typeof GPUValidationError === 'function' && err instanceof GPUValidationError) ||
      /\bvalidation\b|\binvalid\b|\bformat\s+mismatch\b|\btype\s+mismatch\b/i.test(msg)
    ) {
      return PIPELINE_FAILURE_REASON.VALIDATION;
    }
    return PIPELINE_FAILURE_REASON.OTHER;
  }

  // WebGPU GPUShaderStage bitmask values with headless/Node fallbacks.
  const SHADER_STAGE_BITS = Object.freeze({
    VERTEX:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.VERTEX === 'number'
        ? GPUShaderStage.VERTEX
        : 0x1,
    FRAGMENT:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.FRAGMENT === 'number'
        ? GPUShaderStage.FRAGMENT
        : 0x2,
    COMPUTE:
      typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.COMPUTE === 'number'
        ? GPUShaderStage.COMPUTE
        : 0x4,
  });

  const STAGE_NAME_TO_BIT = Object.freeze({
    vertex: SHADER_STAGE_BITS.VERTEX,
    vs: SHADER_STAGE_BITS.VERTEX,
    fragment: SHADER_STAGE_BITS.FRAGMENT,
    fs: SHADER_STAGE_BITS.FRAGMENT,
    pixel: SHADER_STAGE_BITS.FRAGMENT,
    ps: SHADER_STAGE_BITS.FRAGMENT,
    compute: SHADER_STAGE_BITS.COMPUTE,
    cs: SHADER_STAGE_BITS.COMPUTE,
    all: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT | SHADER_STAGE_BITS.COMPUTE,
  });

  // WebGPU GPUVertexFormat byte widths (WebGPU specification table).
  const VERTEX_FORMAT_BYTE_SIZES = Object.freeze({
    uint8: 1,
    sint8: 1,
    unorm8: 1,
    snorm8: 1,
    uint8x2: 2,
    sint8x2: 2,
    unorm8x2: 2,
    snorm8x2: 2,
    uint16: 2,
    sint16: 2,
    unorm16: 2,
    snorm16: 2,
    float16: 2,
    uint8x4: 4,
    sint8x4: 4,
    unorm8x4: 4,
    snorm8x4: 4,
    uint16x2: 4,
    sint16x2: 4,
    unorm16x2: 4,
    snorm16x2: 4,
    float16x2: 4,
    float32: 4,
    uint32: 4,
    sint32: 4,
    'unorm10-10-10-2': 4,
    'unorm8x4-bgra': 4,
    uint16x4: 8,
    sint16x4: 8,
    unorm16x4: 8,
    snorm16x4: 8,
    float16x4: 8,
    float32x2: 8,
    uint32x2: 8,
    sint32x2: 8,
    float32x3: 12,
    uint32x3: 12,
    sint32x3: 12,
    float32x4: 16,
    uint32x4: 16,
    sint32x4: 16,
  });

  const VALID_PRIMITIVE_TOPOLOGIES = new Set([
    'point-list',
    'line-list',
    'line-strip',
    'triangle-list',
    'triangle-strip',
  ]);

  const VALID_BLEND_OPERATIONS = new Set([
    'add',
    'subtract',
    'reverse-subtract',
    'min',
    'max',
  ]);

  const VALID_BLEND_FACTORS = new Set([
    'zero',
    'one',
    'src',
    'one-minus-src',
    'src-alpha',
    'one-minus-src-alpha',
    'dst',
    'one-minus-dst',
    'dst-alpha',
    'one-minus-dst-alpha',
    'src-alpha-saturated',
    'constant',
    'one-minus-constant',
    'src1',
    'one-minus-src1',
    'src1-alpha',
    'one-minus-src1-alpha',
  ]);

  // Canonical W167 fallback sampler descriptors for stages declaring sampler bindings
  // without an explicit static sampler in the root signature.
  const DEFAULT_SAMPLER_DESCRIPTORS = Object.freeze({
    filtering: Object.freeze({
      label: 'fallback-sampler:filtering',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'linear',
    }),
    'non-filtering': Object.freeze({
      label: 'fallback-sampler:non-filtering',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
      magFilter: 'nearest',
      minFilter: 'nearest',
      mipmapFilter: 'nearest',
    }),
    comparison: Object.freeze({
      label: 'fallback-sampler:comparison',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
      addressModeW: 'clamp-to-edge',
      magFilter: 'linear',
      minFilter: 'linear',
      mipmapFilter: 'nearest',
      compare: 'less-equal',
    }),
  });

  const DEFAULT_SAMPLER_KIND_ALIASES = Object.freeze({
    filtering: 'filtering',
    linear: 'filtering',
    'linear-clamp': 'filtering',
    linear_clamp: 'filtering',
    bilinear: 'filtering',
    trilinear: 'filtering',
    default: 'filtering',
    'non-filtering': 'non-filtering',
    non_filtering: 'non-filtering',
    nonfiltering: 'non-filtering',
    point: 'non-filtering',
    nearest: 'non-filtering',
    'nearest-clamp': 'non-filtering',
    nearest_clamp: 'non-filtering',
    'point-clamp': 'non-filtering',
    point_clamp: 'non-filtering',
    comparison: 'comparison',
    compare: 'comparison',
    shadow: 'comparison',
    'less-equal': 'comparison',
    less_equal: 'comparison',
    pcf: 'comparison',
  });

  // Canonical G34 fallback profile aliases matching web/workers/shader_fallback_router.js.
  const FALLBACK_PROFILE_ALIASES = Object.freeze({
    default: 'default',
    minimal: 'default',
    fullscreen_triangle_cbv: 'fullscreen_triangle_cbv',
    fullscreen_cbv: 'fullscreen_triangle_cbv',
    'fullscreen-triangle-cbv': 'fullscreen_triangle_cbv',
    'fullscreen-cbv': 'fullscreen_triangle_cbv',
    textured_quad_24b: 'textured_quad_24b',
    textured_24b: 'textured_quad_24b',
    'textured-quad-24b': 'textured_quad_24b',
    'textured-24b': 'textured_quad_24b',
  });

  const FALLBACK_LABEL_RE =
    /^fallback:(default|minimal|fullscreen[_-]triangle[_-]cbv|fullscreen[_-]cbv|textured[_-]quad[_-]24b|textured[_-]24b)(?:[:\s]|$)/i;

  // Canonical Frame-1 24-byte-stride vertex layout (DrawInstanced(3, 1) / textured_quad_24b):
  // @location(0) float32x4 @ offset 0 (16 B) + @location(1) float32x2 @ offset 16 (8 B).
  const FRAME1_VERTEX_LAYOUT_24B = Object.freeze({
    arrayStride: FRAME1_VERTEX_STRIDE_24B,
    stepMode: 'vertex',
    attributes: Object.freeze([
      Object.freeze({ shaderLocation: 0, offset: 0, format: 'float32x4' }),
      Object.freeze({ shaderLocation: 1, offset: 16, format: 'float32x2' }),
    ]),
  });

  // Canonical group(0) bind-group layout entries matching G34 fallback WGSL profiles.
  const FALLBACK_BIND_GROUP_ENTRIES = Object.freeze({
    default: Object.freeze([]),
    fullscreen_triangle_cbv: Object.freeze([
      Object.freeze({
        binding: 0,
        visibility: SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT,
        buffer: Object.freeze({
          type: 'uniform',
          hasDynamicOffset: false,
          minBindingSize: FRAME1_CBV_MIN_BINDING_SIZE,
        }),
      }),
    ]),
    textured_quad_24b: Object.freeze([
      Object.freeze({
        binding: 0,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        texture: Object.freeze({
          sampleType: 'float',
          viewDimension: '2d',
          multisampled: false,
        }),
      }),
      Object.freeze({
        binding: 1,
        visibility: SHADER_STAGE_BITS.FRAGMENT,
        sampler: Object.freeze({
          type: 'filtering',
        }),
      }),
    ]),
  });

  // 64-bit FNV-1a constants matching shaders/src/convert.rs & web/workers/wgsl_pack_loader.js.
  // Computed in pure 32-bit limbs (hi/lo) on the hot path so lookup keys avoid BigInt & TextEncoder allocations.
  const FNV1A_64_INIT_HI = 0xcbf29ce4;
  const FNV1A_64_INIT_LO = 0x84222325;
  const U64_MASK = 0xffffffffffffffffn;
  const WGPK_LABEL_RE = /^wgpk:(?:vertex|fragment|compute):(0x[0-9a-f]{1,16})$/i;

  // Precomputed decimal strings for common small non-negative integers (bindings, locations, strides, masks).
  const SMALL_INT_STRINGS = new Array(257);
  for (let i = 0; i <= 256; i += 1) {
    SMALL_INT_STRINGS[i] = String(i);
  }

  // WeakMaps for stable identity of opaque WebGPU objects (GPUShaderModule, GPUBindGroupLayout, etc.)
  // and user-registered WGPK/numeric shader module IDs without leaking GC references.
  const registeredObjectIds = new WeakMap();
  const opaqueObjectIds = new WeakMap();
  const fallbackPipelineLayoutProfiles = new WeakMap();
  const gpuHandleFastCache = new WeakMap();
  let nextOpaqueObjectId = 1;
  let shaderModuleRegistryVersion = 0;

  // Reusable 8-byte scratch view for zero-allocation float64 -> 2x uint32 hashing.
  const FAST_HASH_F64 = new Float64Array(1);
  const FAST_HASH_U32 = new Uint32Array(FAST_HASH_F64.buffer);

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function formatHashHex64(value) {
    return `0x${(value & U64_MASK).toString(16).padStart(16, '0')}`;
  }

  function fnv1a64UpdateByte(state, b) {
    const lo = (state.lo ^ b) >>> 0;
    const hi = state.hi;
    const t0 = (lo & 0xffff) * 0x1b3;
    const t1 = (lo >>> 16) * 0x1b3 + (t0 >>> 16);
    state.lo = (((t1 & 0xffff) << 16) | (t0 & 0xffff)) >>> 0;
    state.hi = (Math.imul(hi, 0x1b3) + ((lo << 8) >>> 0) + (t1 >>> 16)) >>> 0;
  }

  function fnv1a64UpdateString(state, str) {
    let lo = state.lo;
    let hi = state.hi;
    const len = str.length;
    for (let i = 0; i < len; i += 1) {
      let c = str.charCodeAt(i);
      if (c < 0x80) {
        lo = (lo ^ c) >>> 0;
        const t0 = (lo & 0xffff) * 0x1b3;
        const t1 = (lo >>> 16) * 0x1b3 + (t0 >>> 16);
        hi = (Math.imul(hi, 0x1b3) + ((lo << 8) >>> 0) + (t1 >>> 16)) >>> 0;
        lo = (((t1 & 0xffff) << 16) | (t0 & 0xffff)) >>> 0;
      } else {
        // UTF-8 encode code points >= 0x80 (including surrogate pairs) identically to TextEncoder.
        if (c >= 0xd800 && c <= 0xdbff && i + 1 < len) {
          const next = str.charCodeAt(i + 1);
          if (next >= 0xdc00 && next <= 0xdfff) {
            c = 0x10000 + (((c - 0xd800) << 10) | (next - 0xdc00));
            i += 1;
          } else {
            c = 0xfffd;
          }
        } else if (c >= 0xd800 && c <= 0xdfff) {
          c = 0xfffd;
        }
        state.hi = hi;
        state.lo = lo;
        if (c < 0x800) {
          fnv1a64UpdateByte(state, 0xc0 | (c >> 6));
          fnv1a64UpdateByte(state, 0x80 | (c & 0x3f));
        } else if (c < 0x10000) {
          fnv1a64UpdateByte(state, 0xe0 | (c >> 12));
          fnv1a64UpdateByte(state, 0x80 | ((c >> 6) & 0x3f));
          fnv1a64UpdateByte(state, 0x80 | (c & 0x3f));
        } else {
          fnv1a64UpdateByte(state, 0xf0 | (c >> 18));
          fnv1a64UpdateByte(state, 0x80 | ((c >> 12) & 0x3f));
          fnv1a64UpdateByte(state, 0x80 | ((c >> 6) & 0x3f));
          fnv1a64UpdateByte(state, 0x80 | (c & 0x3f));
        }
        hi = state.hi;
        lo = state.lo;
      }
    }
    state.hi = hi;
    state.lo = lo;
  }

  function fnv1a64FinalizeHex(state) {
    return (
      '0x' +
      (state.hi >>> 0).toString(16).padStart(8, '0') +
      (state.lo >>> 0).toString(16).padStart(8, '0')
    );
  }

  /**
   * Compute a 64-bit FNV-1a hex digest (`0x` + 16 hex digits) over a UTF-8 string
   * using pure 32-bit arithmetic (zero BigInt or TextEncoder allocations).
   *
   * @param {string} str
   * @returns {string}
   */
  function fnv1a64Hex(str) {
    const state = { hi: FNV1A_64_INIT_HI, lo: FNV1A_64_INIT_LO };
    fnv1a64UpdateString(state, String(str));
    return fnv1a64FinalizeHex(state);
  }

  function formatCanonicalNumber(n) {
    if (n >= 0 && n <= 256 && (n | 0) === n) {
      return SMALL_INT_STRINGS[n];
    }
    if (Number.isNaN(n)) return 'NaN';
    if (n === Infinity) return 'Inf';
    if (n === -Infinity) return '-Inf';
    return Object.is(n, -0) ? '0' : String(n);
  }

  function emitToken(sink, str) {
    if (sink.state !== null) {
      fnv1a64UpdateString(sink.state, str);
    } else {
      sink.buf += str;
    }
  }

  function emitLenStr(sink, str) {
    emitToken(sink, formatCanonicalNumber(str.length));
    emitToken(sink, ':');
    emitToken(sink, str);
  }

  function normalizeShaderId(id) {
    if (typeof id === 'bigint') {
      return id >= 0n && id <= U64_MASK ? formatHashHex64(id) : `bi:${id.toString(10)}`;
    }
    if (typeof id === 'number') {
      if (!Number.isFinite(id)) {
        throw new RangeError(`Shader module ID must be a finite number, got: ${id}`);
      }
      return `num:${id}`;
    }
    if (typeof id === 'string' && id.length > 0) {
      return `str:${id}`;
    }
    throw new TypeError('Shader module ID must be a BigInt, finite number, or non-empty string');
  }

  /**
   * Associate a WebGPU object (e.g., `GPUShaderModule`, `GPUBindGroupLayout`, `GPUPipelineLayout`)
   * with a deterministic ID or WGPK 64-bit shader hash so `hashDescriptorKey` hashes it stably.
   *
   * @param {object} gpuObject
   * @param {bigint | number | string} id
   * @returns {string} Canonical registered ID token
   */
  function setRegisteredObjectId(gpuObject, token) {
    registeredObjectIds.set(gpuObject, token);
    shaderModuleRegistryVersion += 1;
  }

  function registerShaderModuleId(gpuObject, id) {
    if (!gpuObject || (typeof gpuObject !== 'object' && typeof gpuObject !== 'function')) {
      throw new TypeError('gpuObject must be an object');
    }
    const token = normalizeShaderId(id);
    setRegisteredObjectId(gpuObject, token);
    return token;
  }

  function getOpaqueObjectId(obj) {
    let id = opaqueObjectIds.get(obj);
    if (id === undefined) {
      id = nextOpaqueObjectId++;
      opaqueObjectIds.set(obj, id);
    }
    return id;
  }

  function isNativeWebGpuObject(val) {
    if (typeof GPUShaderModule !== 'undefined' && val instanceof GPUShaderModule) return true;
    if (typeof GPUBindGroupLayout !== 'undefined' && val instanceof GPUBindGroupLayout) return true;
    if (typeof GPUPipelineLayout !== 'undefined' && val instanceof GPUPipelineLayout) return true;
    if (typeof GPUSampler !== 'undefined' && val instanceof GPUSampler) return true;
    const tag = Object.prototype.toString.call(val);
    return tag.startsWith('[object GPU');
  }

  function isDuckTypedShaderModule(val) {
    return typeof val.getCompilationInfo === 'function';
  }

  function extractExplicitShaderId(val) {
    if (typeof val.wgpkHash === 'bigint' || typeof val.wgpkHash === 'string' || typeof val.wgpkHash === 'number') {
      return `wgpk:${normalizeShaderId(val.wgpkHash)}`;
    }
    if (typeof val.shaderHash === 'bigint' || typeof val.shaderHash === 'string' || typeof val.shaderHash === 'number') {
      return `sh:${normalizeShaderId(val.shaderHash)}`;
    }
    if (typeof val.hashHex === 'string' && val.hashHex.length > 0) {
      return `hex:${val.hashHex.toLowerCase()}`;
    }
    if (typeof val.hash === 'bigint' || typeof val.hash === 'string' || typeof val.hash === 'number') {
      return `h:${normalizeShaderId(val.hash)}`;
    }
    if (typeof val.shaderId === 'bigint' || typeof val.shaderId === 'string' || typeof val.shaderId === 'number') {
      return `sid:${normalizeShaderId(val.shaderId)}`;
    }
    if (typeof val.moduleId === 'bigint' || typeof val.moduleId === 'string' || typeof val.moduleId === 'number') {
      return `mid:${normalizeShaderId(val.moduleId)}`;
    }
    if (typeof val.label === 'string') {
      const m = WGPK_LABEL_RE.exec(val.label);
      if (m) {
        return `wgpk:0x${m[1].slice(2).toLowerCase().padStart(16, '0')}`;
      }
    }
    return null;
  }

  function hasEnumerableDataKeys(val) {
    for (const k in val) {
      if (!Object.prototype.hasOwnProperty.call(val, k)) continue;
      const v = val[k];
      if (v !== undefined && typeof v !== 'function') {
        return true;
      }
    }
    return false;
  }

  function isPlainRecordObject(val) {
    if (!val || typeof val !== 'object' || Array.isArray(val)) return false;
    if (
      val instanceof Map ||
      val instanceof Set ||
      ArrayBuffer.isView(val) ||
      val instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && val instanceof SharedArrayBuffer)
    ) {
      return false;
    }
    if (registeredObjectIds.has(val) || isDuckTypedShaderModule(val) || isNativeWebGpuObject(val)) {
      return false;
    }
    return true;
  }

  // Resolve a `module` or `layout` slot to a canonical token without allocating key arrays.
  // Returns null if `val` is a structured non-GPU object that needs generic recursive traversal.
  function resolveGpuHandleToken(val) {
    if (val === null) return 'null';
    const t = typeof val;
    if (t === 'string') {
      return val === 'auto' ? 'auto' : `s:${val.length}:${val}`;
    }
    if (t === 'bigint') {
      return `bi:${val.toString(10)}`;
    }
    if (t === 'number') {
      return `n:${formatCanonicalNumber(val)}`;
    }
    if (t !== 'object') {
      return null;
    }
    const registered = registeredObjectIds.get(val);
    if (registered !== undefined) {
      return `reg:${registered}`;
    }
    const explicitId = extractExplicitShaderId(val);
    if (explicitId !== null) {
      return `gpu:${explicitId}`;
    }
    if (isDuckTypedShaderModule(val) || isNativeWebGpuObject(val) || !hasEnumerableDataKeys(val)) {
      return `gpu-oid:${getOpaqueObjectId(val)}`;
    }
    return null;
  }

  function isFastConstantsObject(obj) {
    if (!isPlainRecordObject(obj)) return false;
    for (const k in obj) {
      if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
      const v = obj[k];
      if (v === undefined || typeof v === 'function') continue;
      if (typeof v !== 'number' && typeof v !== 'boolean') {
        return false;
      }
    }
    return true;
  }

  function emitFastConstants(obj, sink) {
    const keys = [];
    for (const k in obj) {
      if (!Object.prototype.hasOwnProperty.call(obj, k)) continue;
      const v = obj[k];
      if (v === undefined || typeof v === 'function') continue;
      keys.push(k);
    }
    if (keys.length > 1) {
      keys.sort();
    }
    emitToken(sink, '|c:');
    emitToken(sink, formatCanonicalNumber(keys.length));
    emitToken(sink, ':[');
    for (let i = 0; i < keys.length; i += 1) {
      if (i > 0) emitToken(sink, ',');
      const k = keys[i];
      const v = obj[k];
      emitLenStr(sink, k);
      if (typeof v === 'boolean') {
        emitToken(sink, v ? '=b:1' : '=b:0');
      } else {
        emitToken(sink, '=n:');
        emitToken(sink, formatCanonicalNumber(v));
      }
    }
    emitToken(sink, ']');
  }

  function isFastVertexAttribute(attr) {
    if (!isPlainRecordObject(attr)) return false;
    for (const k in attr) {
      if (!Object.prototype.hasOwnProperty.call(attr, k)) continue;
      const v = attr[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'format') {
        if (typeof v !== 'string') return false;
      } else if (k === 'offset' || k === 'shaderLocation') {
        if (typeof v !== 'number') return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastVertexBuffer(buf) {
    if (buf === null || buf === undefined) return true;
    if (!isPlainRecordObject(buf)) return false;
    for (const k in buf) {
      if (!Object.prototype.hasOwnProperty.call(buf, k)) continue;
      const v = buf[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'arrayStride') {
        if (typeof v !== 'number') return false;
      } else if (k === 'stepMode') {
        if (typeof v !== 'string') return false;
      } else if (k === 'attributes') {
        if (!Array.isArray(v)) return false;
        for (let i = 0; i < v.length; i += 1) {
          if (!isFastVertexAttribute(v[i])) return false;
        }
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastVertexState(vertex) {
    if (!isPlainRecordObject(vertex)) return false;
    for (const k in vertex) {
      if (!Object.prototype.hasOwnProperty.call(vertex, k)) continue;
      const v = vertex[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'module') {
        if (resolveGpuHandleToken(v) === null) return false;
      } else if (k === 'entryPoint') {
        if (typeof v !== 'string') return false;
      } else if (k === 'buffers') {
        if (!Array.isArray(v)) return false;
        for (let i = 0; i < v.length; i += 1) {
          if (!isFastVertexBuffer(v[i])) return false;
        }
      } else if (k === 'constants') {
        if (!isFastConstantsObject(v)) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastBlendComponent(comp) {
    if (!isPlainRecordObject(comp)) return false;
    for (const k in comp) {
      if (!Object.prototype.hasOwnProperty.call(comp, k)) continue;
      const v = comp[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'operation' || k === 'srcFactor' || k === 'dstFactor') {
        if (typeof v !== 'string') return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastBlendState(blend) {
    if (blend === null) return true;
    if (!isPlainRecordObject(blend)) return false;
    for (const k in blend) {
      if (!Object.prototype.hasOwnProperty.call(blend, k)) continue;
      const v = blend[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'color' || k === 'alpha') {
        if (!isFastBlendComponent(v)) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastColorTarget(target) {
    if (target === null || target === undefined) return true;
    if (!isPlainRecordObject(target)) return false;
    for (const k in target) {
      if (!Object.prototype.hasOwnProperty.call(target, k)) continue;
      const v = target[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'format') {
        if (typeof v !== 'string') return false;
      } else if (k === 'writeMask') {
        if (typeof v !== 'number') return false;
      } else if (k === 'blend') {
        if (!isFastBlendState(v)) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastFragmentState(fragment) {
    if (!isPlainRecordObject(fragment)) return false;
    for (const k in fragment) {
      if (!Object.prototype.hasOwnProperty.call(fragment, k)) continue;
      const v = fragment[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'module') {
        if (resolveGpuHandleToken(v) === null) return false;
      } else if (k === 'entryPoint') {
        if (typeof v !== 'string') return false;
      } else if (k === 'targets') {
        if (!Array.isArray(v)) return false;
        for (let i = 0; i < v.length; i += 1) {
          if (!isFastColorTarget(v[i])) return false;
        }
      } else if (k === 'constants') {
        if (!isFastConstantsObject(v)) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastPrimitiveState(prim) {
    if (!isPlainRecordObject(prim)) return false;
    for (const k in prim) {
      if (!Object.prototype.hasOwnProperty.call(prim, k)) continue;
      const v = prim[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'topology' || k === 'stripIndexFormat' || k === 'frontFace' || k === 'cullMode') {
        if (typeof v !== 'string') return false;
      } else if (k === 'unclippedDepth') {
        if (typeof v !== 'boolean') return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastStencilFace(face) {
    if (!isPlainRecordObject(face)) return false;
    for (const k in face) {
      if (!Object.prototype.hasOwnProperty.call(face, k)) continue;
      const v = face[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'compare' || k === 'failOp' || k === 'depthFailOp' || k === 'passOp') {
        if (typeof v !== 'string') return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastDepthStencilState(ds) {
    if (ds === null) return true;
    if (!isPlainRecordObject(ds)) return false;
    for (const k in ds) {
      if (!Object.prototype.hasOwnProperty.call(ds, k)) continue;
      const v = ds[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'format' || k === 'depthCompare') {
        if (typeof v !== 'string') return false;
      } else if (k === 'depthWriteEnabled') {
        if (typeof v !== 'boolean') return false;
      } else if (k === 'stencilFront' || k === 'stencilBack') {
        if (!isFastStencilFace(v)) return false;
      } else if (
        k === 'stencilReadMask' ||
        k === 'stencilWriteMask' ||
        k === 'depthBias' ||
        k === 'depthBiasSlopeScale' ||
        k === 'depthBiasClamp'
      ) {
        if (typeof v !== 'number') return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastMultisampleState(ms) {
    if (!isPlainRecordObject(ms)) return false;
    for (const k in ms) {
      if (!Object.prototype.hasOwnProperty.call(ms, k)) continue;
      const v = ms[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'count' || k === 'mask') {
        if (typeof v !== 'number') return false;
      } else if (k === 'alphaToCoverageEnabled') {
        if (typeof v !== 'boolean') return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastRenderDescriptor(desc) {
    if (!isPlainRecordObject(desc)) return false;
    if (desc.vertex === undefined || !isFastVertexState(desc.vertex)) {
      return false;
    }
    for (const k in desc) {
      if (!Object.prototype.hasOwnProperty.call(desc, k)) continue;
      const v = desc[k];
      if (v === undefined || typeof v === 'function') continue;
      switch (k) {
        case 'vertex':
          break;
        case 'layout':
          if (resolveGpuHandleToken(v) === null) return false;
          break;
        case 'label':
          if (typeof v !== 'string') return false;
          break;
        case 'fragment':
          if (!isFastFragmentState(v)) return false;
          break;
        case 'primitive':
          if (!isFastPrimitiveState(v)) return false;
          break;
        case 'depthStencil':
          if (!isFastDepthStencilState(v)) return false;
          break;
        case 'multisample':
          if (!isFastMultisampleState(v)) return false;
          break;
        default:
          return false;
      }
    }
    return true;
  }

  function isFastComputeState(compute) {
    if (!isPlainRecordObject(compute)) return false;
    for (const k in compute) {
      if (!Object.prototype.hasOwnProperty.call(compute, k)) continue;
      const v = compute[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'module') {
        if (resolveGpuHandleToken(v) === null) return false;
      } else if (k === 'entryPoint') {
        if (typeof v !== 'string') return false;
      } else if (k === 'constants') {
        if (!isFastConstantsObject(v)) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastComputeDescriptor(desc) {
    if (!isPlainRecordObject(desc)) return false;
    if (desc.compute === undefined || !isFastComputeState(desc.compute)) {
      return false;
    }
    for (const k in desc) {
      if (!Object.prototype.hasOwnProperty.call(desc, k)) continue;
      const v = desc[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'compute') {
        continue;
      } else if (k === 'layout') {
        if (resolveGpuHandleToken(v) === null) return false;
      } else if (k === 'label') {
        if (typeof v !== 'string') return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastBindGroupLayoutEntry(entry) {
    if (!isPlainRecordObject(entry)) return false;
    for (const k in entry) {
      if (!Object.prototype.hasOwnProperty.call(entry, k)) continue;
      const v = entry[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'binding' || k === 'visibility') {
        if (typeof v !== 'number') return false;
      } else if (k === 'buffer') {
        if (!isPlainRecordObject(v)) return false;
        for (const bk in v) {
          if (!Object.prototype.hasOwnProperty.call(v, bk)) continue;
          const bv = v[bk];
          if (bv === undefined || typeof bv === 'function') continue;
          if (bk === 'type') {
            if (typeof bv !== 'string') return false;
          } else if (bk === 'hasDynamicOffset') {
            if (typeof bv !== 'boolean') return false;
          } else if (bk === 'minBindingSize') {
            if (typeof bv !== 'number') return false;
          } else {
            return false;
          }
        }
      } else if (k === 'sampler') {
        if (!isPlainRecordObject(v)) return false;
        for (const sk in v) {
          if (!Object.prototype.hasOwnProperty.call(v, sk)) continue;
          const sv = v[sk];
          if (sv === undefined || typeof sv === 'function') continue;
          if (sk === 'type') {
            if (typeof sv !== 'string') return false;
          } else {
            return false;
          }
        }
      } else if (k === 'texture') {
        if (!isPlainRecordObject(v)) return false;
        for (const tk in v) {
          if (!Object.prototype.hasOwnProperty.call(v, tk)) continue;
          const tv = v[tk];
          if (tv === undefined || typeof tv === 'function') continue;
          if (tk === 'sampleType' || tk === 'viewDimension') {
            if (typeof tv !== 'string') return false;
          } else if (tk === 'multisampled') {
            if (typeof tv !== 'boolean') return false;
          } else {
            return false;
          }
        }
      } else if (k === 'storageTexture') {
        if (!isPlainRecordObject(v)) return false;
        for (const stk in v) {
          if (!Object.prototype.hasOwnProperty.call(v, stk)) continue;
          const stv = v[stk];
          if (stv === undefined || typeof stv === 'function') continue;
          if (stk === 'access' || stk === 'format' || stk === 'viewDimension') {
            if (typeof stv !== 'string') return false;
          } else {
            return false;
          }
        }
      } else if (k === 'externalTexture') {
        if (!isPlainRecordObject(v) || hasEnumerableDataKeys(v)) return false;
      } else {
        return false;
      }
    }
    return true;
  }

  function isFastBindGroupLayoutDescriptor(desc) {
    if (!isPlainRecordObject(desc)) return false;
    if (!Array.isArray(desc.entries)) return false;
    for (const k in desc) {
      if (!Object.prototype.hasOwnProperty.call(desc, k)) continue;
      const v = desc[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k === 'entries') {
        continue;
      } else if (k === 'label') {
        if (typeof v !== 'string') return false;
      } else {
        return false;
      }
    }
    for (let i = 0; i < desc.entries.length; i += 1) {
      if (!isFastBindGroupLayoutEntry(desc.entries[i])) return false;
    }
    return true;
  }

  function emitFastBlendComponent(comp, tag, sink) {
    emitToken(sink, tag);
    if (comp.operation !== undefined) {
      emitToken(sink, 'o:');
      emitLenStr(sink, comp.operation);
    }
    if (comp.srcFactor !== undefined) {
      emitToken(sink, ';s:');
      emitLenStr(sink, comp.srcFactor);
    }
    if (comp.dstFactor !== undefined) {
      emitToken(sink, ';d:');
      emitLenStr(sink, comp.dstFactor);
    }
    emitToken(sink, '}');
  }

  function emitFastStencilFace(face, tag, sink) {
    emitToken(sink, tag);
    if (face.compare !== undefined) {
      emitToken(sink, 'c:');
      emitLenStr(sink, face.compare);
    }
    if (face.failOp !== undefined) {
      emitToken(sink, ';f:');
      emitLenStr(sink, face.failOp);
    }
    if (face.depthFailOp !== undefined) {
      emitToken(sink, ';d:');
      emitLenStr(sink, face.depthFailOp);
    }
    if (face.passOp !== undefined) {
      emitToken(sink, ';p:');
      emitLenStr(sink, face.passOp);
    }
    emitToken(sink, '}');
  }

  function emitFastRenderDescriptor(desc, sink) {
    emitToken(sink, 'r');
    if (desc.layout !== undefined) {
      emitToken(sink, '|l:');
      emitLenStr(sink, resolveGpuHandleToken(desc.layout));
    }
    if (desc.label !== undefined) {
      emitToken(sink, '|lbl:');
      emitLenStr(sink, desc.label);
    }

    // Vertex state
    const v = desc.vertex;
    emitToken(sink, '|v:{');
    if (v.module !== undefined) {
      emitToken(sink, 'm:');
      emitLenStr(sink, resolveGpuHandleToken(v.module));
    }
    if (v.entryPoint !== undefined) {
      emitToken(sink, '|ep:');
      emitLenStr(sink, v.entryPoint);
    }
    if (v.buffers !== undefined) {
      const bufs = v.buffers;
      emitToken(sink, '|vb:');
      emitToken(sink, formatCanonicalNumber(bufs.length));
      emitToken(sink, ':[');
      for (let i = 0; i < bufs.length; i += 1) {
        if (i > 0) emitToken(sink, ',');
        const b = bufs[i];
        if (b === null) {
          emitToken(sink, 'N');
        } else if (b === undefined) {
          emitToken(sink, 'U');
        } else {
          emitToken(sink, 'B{');
          if (b.arrayStride !== undefined) {
            emitToken(sink, 'as:');
            emitToken(sink, formatCanonicalNumber(b.arrayStride));
          }
          if (b.stepMode !== undefined) {
            emitToken(sink, '|sm:');
            emitLenStr(sink, b.stepMode);
          }
          if (b.attributes !== undefined) {
            const attrs = b.attributes;
            emitToken(sink, '|a:');
            emitToken(sink, formatCanonicalNumber(attrs.length));
            emitToken(sink, ':[');
            for (let j = 0; j < attrs.length; j += 1) {
              if (j > 0) emitToken(sink, ',');
              const a = attrs[j];
              emitToken(sink, '{');
              if (a.shaderLocation !== undefined) {
                emitToken(sink, 'l:');
                emitToken(sink, formatCanonicalNumber(a.shaderLocation));
              }
              if (a.offset !== undefined) {
                emitToken(sink, ';o:');
                emitToken(sink, formatCanonicalNumber(a.offset));
              }
              if (a.format !== undefined) {
                emitToken(sink, ';f:');
                emitLenStr(sink, a.format);
              }
              emitToken(sink, '}');
            }
            emitToken(sink, ']');
          }
          emitToken(sink, '}');
        }
      }
      emitToken(sink, ']');
    }
    if (v.constants !== undefined) {
      emitFastConstants(v.constants, sink);
    }
    emitToken(sink, '}');

    // Fragment state
    if (desc.fragment !== undefined) {
      const f = desc.fragment;
      emitToken(sink, '|f:{');
      if (f.module !== undefined) {
        emitToken(sink, 'm:');
        emitLenStr(sink, resolveGpuHandleToken(f.module));
      }
      if (f.entryPoint !== undefined) {
        emitToken(sink, '|ep:');
        emitLenStr(sink, f.entryPoint);
      }
      if (f.targets !== undefined) {
        const targets = f.targets;
        emitToken(sink, '|t:');
        emitToken(sink, formatCanonicalNumber(targets.length));
        emitToken(sink, ':[');
        for (let i = 0; i < targets.length; i += 1) {
          if (i > 0) emitToken(sink, ',');
          const t = targets[i];
          if (t === null) {
            emitToken(sink, 'N');
          } else if (t === undefined) {
            emitToken(sink, 'U');
          } else {
            emitToken(sink, 'T{');
            if (t.format !== undefined) {
              emitToken(sink, 'f:');
              emitLenStr(sink, t.format);
            }
            if (t.writeMask !== undefined) {
              emitToken(sink, '|wm:');
              emitToken(sink, formatCanonicalNumber(t.writeMask));
            }
            if (t.blend !== undefined) {
              if (t.blend === null) {
                emitToken(sink, '|bl:N');
              } else {
                emitToken(sink, '|bl:{');
                if (t.blend.color !== undefined) {
                  emitFastBlendComponent(t.blend.color, 'c:{', sink);
                }
                if (t.blend.alpha !== undefined) {
                  emitFastBlendComponent(t.blend.alpha, '|a:{', sink);
                }
                emitToken(sink, '}');
              }
            }
            emitToken(sink, '}');
          }
        }
        emitToken(sink, ']');
      }
      if (f.constants !== undefined) {
        emitFastConstants(f.constants, sink);
      }
      emitToken(sink, '}');
    }

    // Primitive state
    if (desc.primitive !== undefined) {
      const p = desc.primitive;
      emitToken(sink, '|p:{');
      if (p.topology !== undefined) {
        emitToken(sink, 'top:');
        emitLenStr(sink, p.topology);
      }
      if (p.stripIndexFormat !== undefined) {
        emitToken(sink, '|sif:');
        emitLenStr(sink, p.stripIndexFormat);
      }
      if (p.frontFace !== undefined) {
        emitToken(sink, '|ff:');
        emitLenStr(sink, p.frontFace);
      }
      if (p.cullMode !== undefined) {
        emitToken(sink, '|cm:');
        emitLenStr(sink, p.cullMode);
      }
      if (p.unclippedDepth !== undefined) {
        emitToken(sink, p.unclippedDepth ? '|ud:1' : '|ud:0');
      }
      emitToken(sink, '}');
    }

    // Depth-stencil state
    if (desc.depthStencil !== undefined) {
      const ds = desc.depthStencil;
      if (ds === null) {
        emitToken(sink, '|ds:N');
      } else {
        emitToken(sink, '|ds:{');
        if (ds.format !== undefined) {
          emitToken(sink, 'f:');
          emitLenStr(sink, ds.format);
        }
        if (ds.depthWriteEnabled !== undefined) {
          emitToken(sink, ds.depthWriteEnabled ? '|dw:1' : '|dw:0');
        }
        if (ds.depthCompare !== undefined) {
          emitToken(sink, '|dc:');
          emitLenStr(sink, ds.depthCompare);
        }
        if (ds.stencilFront !== undefined) {
          emitFastStencilFace(ds.stencilFront, '|sf:{', sink);
        }
        if (ds.stencilBack !== undefined) {
          emitFastStencilFace(ds.stencilBack, '|sb:{', sink);
        }
        if (ds.stencilReadMask !== undefined) {
          emitToken(sink, '|srm:');
          emitToken(sink, formatCanonicalNumber(ds.stencilReadMask));
        }
        if (ds.stencilWriteMask !== undefined) {
          emitToken(sink, '|swm:');
          emitToken(sink, formatCanonicalNumber(ds.stencilWriteMask));
        }
        if (ds.depthBias !== undefined) {
          emitToken(sink, '|db:');
          emitToken(sink, formatCanonicalNumber(ds.depthBias));
        }
        if (ds.depthBiasSlopeScale !== undefined) {
          emitToken(sink, '|dbs:');
          emitToken(sink, formatCanonicalNumber(ds.depthBiasSlopeScale));
        }
        if (ds.depthBiasClamp !== undefined) {
          emitToken(sink, '|dbc:');
          emitToken(sink, formatCanonicalNumber(ds.depthBiasClamp));
        }
        emitToken(sink, '}');
      }
    }

    // Multisample state
    if (desc.multisample !== undefined) {
      const ms = desc.multisample;
      emitToken(sink, '|ms:{');
      if (ms.count !== undefined) {
        emitToken(sink, 'cnt:');
        emitToken(sink, formatCanonicalNumber(ms.count));
      }
      if (ms.mask !== undefined) {
        emitToken(sink, '|msk:');
        emitToken(sink, formatCanonicalNumber(ms.mask));
      }
      if (ms.alphaToCoverageEnabled !== undefined) {
        emitToken(sink, ms.alphaToCoverageEnabled ? '|atc:1' : '|atc:0');
      }
      emitToken(sink, '}');
    }
  }

  function emitFastComputeDescriptor(desc, sink) {
    emitToken(sink, 'c');
    if (desc.layout !== undefined) {
      emitToken(sink, '|l:');
      emitLenStr(sink, resolveGpuHandleToken(desc.layout));
    }
    if (desc.label !== undefined) {
      emitToken(sink, '|lbl:');
      emitLenStr(sink, desc.label);
    }
    const cs = desc.compute;
    emitToken(sink, '|cs:{');
    if (cs.module !== undefined) {
      emitToken(sink, 'm:');
      emitLenStr(sink, resolveGpuHandleToken(cs.module));
    }
    if (cs.entryPoint !== undefined) {
      emitToken(sink, '|ep:');
      emitLenStr(sink, cs.entryPoint);
    }
    if (cs.constants !== undefined) {
      emitFastConstants(cs.constants, sink);
    }
    emitToken(sink, '}');
  }

  function emitFastBindGroupLayoutDescriptor(desc, sink) {
    emitToken(sink, 'bgl');
    if (desc.label !== undefined) {
      emitToken(sink, '|lbl:');
      emitLenStr(sink, desc.label);
    }
    const entries = desc.entries;
    emitToken(sink, '|e:');
    emitToken(sink, formatCanonicalNumber(entries.length));
    emitToken(sink, ':[');
    for (let i = 0; i < entries.length; i += 1) {
      if (i > 0) emitToken(sink, ',');
      const e = entries[i];
      emitToken(sink, '{');
      if (e.binding !== undefined) {
        emitToken(sink, 'b:');
        emitToken(sink, formatCanonicalNumber(e.binding));
      }
      if (e.visibility !== undefined) {
        emitToken(sink, '|v:');
        emitToken(sink, formatCanonicalNumber(e.visibility));
      }
      if (e.buffer !== undefined) {
        const b = e.buffer;
        emitToken(sink, '|buf:{');
        if (b.type !== undefined) {
          emitToken(sink, 't:');
          emitLenStr(sink, b.type);
        }
        if (b.hasDynamicOffset !== undefined) {
          emitToken(sink, b.hasDynamicOffset ? '|do:1' : '|do:0');
        }
        if (b.minBindingSize !== undefined) {
          emitToken(sink, '|mbs:');
          emitToken(sink, formatCanonicalNumber(b.minBindingSize));
        }
        emitToken(sink, '}');
      }
      if (e.sampler !== undefined) {
        emitToken(sink, '|smp:{');
        if (e.sampler.type !== undefined) {
          emitToken(sink, 't:');
          emitLenStr(sink, e.sampler.type);
        }
        emitToken(sink, '}');
      }
      if (e.texture !== undefined) {
        const tx = e.texture;
        emitToken(sink, '|tex:{');
        if (tx.sampleType !== undefined) {
          emitToken(sink, 'st:');
          emitLenStr(sink, tx.sampleType);
        }
        if (tx.viewDimension !== undefined) {
          emitToken(sink, '|vd:');
          emitLenStr(sink, tx.viewDimension);
        }
        if (tx.multisampled !== undefined) {
          emitToken(sink, tx.multisampled ? '|ms:1' : '|ms:0');
        }
        emitToken(sink, '}');
      }
      if (e.storageTexture !== undefined) {
        const st = e.storageTexture;
        emitToken(sink, '|stx:{');
        if (st.access !== undefined) {
          emitToken(sink, 'a:');
          emitLenStr(sink, st.access);
        }
        if (st.format !== undefined) {
          emitToken(sink, '|f:');
          emitLenStr(sink, st.format);
        }
        if (st.viewDimension !== undefined) {
          emitToken(sink, '|vd:');
          emitLenStr(sink, st.viewDimension);
        }
        emitToken(sink, '}');
      }
      if (e.externalTexture !== undefined) {
        emitToken(sink, '|ext:{}');
      }
      emitToken(sink, '}');
    }
    emitToken(sink, ']');
  }

  function canonicalizeToSink(val, keyName, ancestors, sink) {
    if (val === null) {
      emitToken(sink, 'null');
      return;
    }
    const t = typeof val;
    if (t === 'undefined') {
      emitToken(sink, 'undef');
      return;
    }
    if (t === 'boolean') {
      emitToken(sink, val ? 'b:1' : 'b:0');
      return;
    }
    if (t === 'number') {
      emitToken(sink, 'n:');
      emitToken(sink, formatCanonicalNumber(val));
      return;
    }
    if (t === 'bigint') {
      emitToken(sink, 'bi:');
      emitToken(sink, val.toString(10));
      return;
    }
    if (t === 'string') {
      emitToken(sink, 's:');
      emitLenStr(sink, val);
      return;
    }
    if (t === 'symbol') {
      emitToken(sink, 'sym:');
      emitToken(sink, String(val.description || ''));
      return;
    }
    if (t === 'function') {
      emitToken(sink, 'fn');
      return;
    }

    // Check registered GPU resource / shader module IDs first.
    const registered = registeredObjectIds.get(val);
    if (registered !== undefined) {
      emitToken(sink, 'reg:');
      emitToken(sink, registered);
      return;
    }

    // Handle native/opaque WebGPU objects and shader modules before walking keys.
    const isShaderOrLayoutSlot = keyName === 'module' || keyName === 'layout';
    const isGpuHandle = isDuckTypedShaderModule(val) || isNativeWebGpuObject(val);
    if (isGpuHandle || isShaderOrLayoutSlot) {
      const explicitId = extractExplicitShaderId(val);
      if (explicitId !== null) {
        emitToken(sink, 'gpu:');
        emitToken(sink, explicitId);
        return;
      }
      if (isGpuHandle || !hasEnumerableDataKeys(val)) {
        emitToken(sink, 'gpu-oid:');
        emitToken(sink, formatCanonicalNumber(getOpaqueObjectId(val)));
        return;
      }
    }

    if (ancestors.has(val)) {
      throw new TypeError('Circular reference detected in WebGPU descriptor');
    }

    if (Array.isArray(val)) {
      ancestors.add(val);
      emitToken(sink, 'a:');
      emitToken(sink, formatCanonicalNumber(val.length));
      emitToken(sink, ':[');
      for (let i = 0; i < val.length; i += 1) {
        if (i > 0) emitToken(sink, ',');
        canonicalizeToSink(val[i], null, ancestors, sink);
      }
      emitToken(sink, ']');
      ancestors.delete(val);
      return;
    }

    if (ArrayBuffer.isView(val)) {
      const bytes = new Uint8Array(val.buffer, val.byteOffset, val.byteLength);
      const ctorName = val.constructor ? val.constructor.name : 'View';
      emitToken(sink, 'bin:');
      emitToken(sink, ctorName);
      emitToken(sink, ':');
      emitToken(sink, formatCanonicalNumber(bytes.byteLength));
      emitToken(sink, ':');
      for (let i = 0; i < bytes.length; i += 1) {
        emitToken(sink, bytes[i].toString(16).padStart(2, '0'));
      }
      return;
    }

    if (
      val instanceof ArrayBuffer ||
      (typeof SharedArrayBuffer !== 'undefined' && val instanceof SharedArrayBuffer)
    ) {
      const bytes = new Uint8Array(val);
      emitToken(sink, 'ab:');
      emitToken(sink, formatCanonicalNumber(bytes.byteLength));
      emitToken(sink, ':');
      for (let i = 0; i < bytes.length; i += 1) {
        emitToken(sink, bytes[i].toString(16).padStart(2, '0'));
      }
      return;
    }

    if (val instanceof Set) {
      ancestors.add(val);
      const items = [];
      for (const item of val) {
        const subSink = { state: null, buf: '' };
        canonicalizeToSink(item, null, ancestors, subSink);
        items.push(subSink.buf);
      }
      ancestors.delete(val);
      items.sort();
      emitToken(sink, 'set:');
      emitToken(sink, formatCanonicalNumber(items.length));
      emitToken(sink, ':[');
      for (let i = 0; i < items.length; i += 1) {
        if (i > 0) emitToken(sink, ',');
        emitToken(sink, items[i]);
      }
      emitToken(sink, ']');
      return;
    }

    if (val instanceof Map) {
      ancestors.add(val);
      const pairs = [];
      for (const [k, v] of val.entries()) {
        const kSink = { state: null, buf: '' };
        const vSink = { state: null, buf: '' };
        canonicalizeToSink(k, null, ancestors, kSink);
        canonicalizeToSink(v, null, ancestors, vSink);
        pairs.push(`${kSink.buf}=>${vSink.buf}`);
      }
      ancestors.delete(val);
      pairs.sort();
      emitToken(sink, 'map:');
      emitToken(sink, formatCanonicalNumber(pairs.length));
      emitToken(sink, ':{');
      for (let i = 0; i < pairs.length; i += 1) {
        if (i > 0) emitToken(sink, ',');
        emitToken(sink, pairs[i]);
      }
      emitToken(sink, '}');
      return;
    }

    // Plain/structured object: sort keys lexicographically for insertion-order invariance.
    ancestors.add(val);
    const keys = Object.keys(val).sort();
    let count = 0;
    for (let i = 0; i < keys.length; i += 1) {
      const v = val[keys[i]];
      if (v !== undefined && typeof v !== 'function') {
        count += 1;
      }
    }
    emitToken(sink, 'o:');
    emitToken(sink, formatCanonicalNumber(count));
    emitToken(sink, ':{');
    let emitted = 0;
    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i];
      const v = val[k];
      if (v === undefined || typeof v === 'function') {
        continue;
      }
      if (emitted > 0) emitToken(sink, ',');
      emitLenStr(sink, k);
      emitToken(sink, '=');
      canonicalizeToSink(v, k, ancestors, sink);
      emitted += 1;
    }
    emitToken(sink, '}');
    ancestors.delete(val);
  }

  function emitDescriptorToSink(obj, sink) {
    if (isFastRenderDescriptor(obj)) {
      emitFastRenderDescriptor(obj, sink);
      return;
    }
    if (isFastComputeDescriptor(obj)) {
      emitFastComputeDescriptor(obj, sink);
      return;
    }
    if (isFastBindGroupLayoutDescriptor(obj)) {
      emitFastBindGroupLayoutDescriptor(obj, sink);
      return;
    }
    canonicalizeToSink(obj, null, new Set(), sink);
  }

  /**
   * Build a compact, deterministic, 100% injective string key directly from WebGPU pipeline or
   * bind-group-layout descriptor fields without `JSON.stringify` or `Object.keys().sort()` on the
   * hot path. Covers color targets, blend state, depth/stencil, primitive topology, vertex layout,
   * multisample state, pipeline layout, and shader hashes/IDs; falls back cleanly for arbitrary JS
   * descriptors. Satisfies `fnv1a64Hex(buildFastPipelineKey(desc)) === hashDescriptorKey(desc)`.
   *
   * @param {unknown} descriptor
   * @returns {string}
   */
  function buildFastPipelineKey(descriptor) {
    const sink = { state: null, buf: '' };
    emitDescriptorToSink(descriptor, sink);
    return sink.buf;
  }

  /**
   * Compute a deterministic canonical FNV-1a 64-bit hex hash (`0x` + 16 hex digits)
   * for any WebGPU descriptor or nested JS value, independent of object key insertion order.
   * Uses a zero-`JSON.stringify`, zero-`Object.keys().sort()`, zero-`BigInt`, zero-`TextEncoder`
   * direct field streamer for standard render/compute/bind-group-layout descriptors, and supports
   * `BigInt`, nested arrays/objects, TypedArrays, `Map`/`Set`, opaque `GPUShaderModule`s,
   * and WGPK shader hashes.
   *
   * @param {unknown} obj
   * @returns {string}
   */
  function hashDescriptorKey(obj) {
    const state = { hi: FNV1A_64_INIT_HI, lo: FNV1A_64_INIT_LO };
    emitDescriptorToSink(obj, { state, buf: '' });
    return fnv1a64FinalizeHex(state);
  }

  // --- W281: Fast-Path Integer Hash & Structural Descriptor Snapshot Helpers ---

  const FAST_HASH_AUTO_HANDLE = Object.freeze({
    token: 'auto',
    h1: 0x6175746f,
    h2: 0x2f6a1b93,
  });

  function mixFastInt(state, x) {
    const v = x | 0;
    state.h1 = Math.imul(state.h1 ^ v, 0x85ebca6b) >>> 0;
    state.h2 = Math.imul(state.h2 ^ ((v >>> 16) | (v << 16)), 0xc2b2ae35) >>> 0;
  }

  function mixFastNumber(state, n) {
    const normalized = n === 0 ? 0 : n;
    if ((normalized | 0) === normalized) {
      mixFastInt(state, normalized);
      return;
    }
    FAST_HASH_F64[0] = normalized;
    mixFastInt(state, FAST_HASH_U32[0]);
    mixFastInt(state, FAST_HASH_U32[1]);
  }

  function mixFastString(state, str) {
    const len = str.length;
    let h1 = Math.imul(state.h1 ^ len, 0x85ebca6b) >>> 0;
    let h2 = Math.imul(state.h2 ^ len, 0xc2b2ae35) >>> 0;
    for (let i = 0; i < len; i += 1) {
      const c = str.charCodeAt(i);
      h1 = Math.imul(h1 ^ c, 0x01000193) >>> 0;
      h2 = Math.imul(h2 ^ ((c + i) | 0), 0x27d4eb2d) >>> 0;
    }
    state.h1 = h1;
    state.h2 = h2;
  }

  function finalizeFastHash53(state) {
    const hi21 = (state.h1 ^ (state.h1 >>> 16)) & 0x1fffff;
    const lo32 = (state.h2 ^ (state.h2 >>> 15)) >>> 0;
    return hi21 * 0x100000000 + lo32;
  }

  function resolveGpuHandleFast(val) {
    if (val === 'auto') {
      return FAST_HASH_AUTO_HANDLE;
    }
    if (val === null || val === undefined) {
      return null;
    }
    const t = typeof val;
    if (t === 'object' || t === 'function') {
      const cached = gpuHandleFastCache.get(val);
      if (
        cached !== undefined &&
        cached.regVer === shaderModuleRegistryVersion &&
        cached.label === val.label &&
        cached.wgpkHash === val.wgpkHash &&
        cached.shaderHash === val.shaderHash &&
        cached.hashHex === val.hashHex &&
        cached.hash === val.hash &&
        cached.shaderId === val.shaderId &&
        cached.moduleId === val.moduleId
      ) {
        return cached;
      }
      const token = resolveGpuHandleToken(val);
      if (token === null) {
        return null;
      }
      const st = { h1: 0x9e3779b9, h2: 0x85ebca6b };
      mixFastString(st, token);
      const entry = {
        token,
        h1: st.h1,
        h2: st.h2,
        regVer: shaderModuleRegistryVersion,
        label: val.label,
        wgpkHash: val.wgpkHash,
        shaderHash: val.shaderHash,
        hashHex: val.hashHex,
        hash: val.hash,
        shaderId: val.shaderId,
        moduleId: val.moduleId,
      };
      gpuHandleFastCache.set(val, entry);
      return entry;
    }
    const token = resolveGpuHandleToken(val);
    if (token === null) {
      return null;
    }
    const st = { h1: 0x9e3779b9, h2: 0x85ebca6b };
    mixFastString(st, token);
    return { token, h1: st.h1, h2: st.h2 };
  }

  function mixFastGpuHandle(state, handleFast) {
    mixFastInt(state, handleFast.h1);
    mixFastInt(state, handleFast.h2);
  }

  function isLikelyFallbackModule(mod) {
    if (!mod || typeof mod !== 'object') {
      return false;
    }
    if (
      mod.isFallback === true ||
      mod.source === 'builtin' ||
      (typeof mod.fallbackProfile === 'string' && mod.fallbackProfile.length > 0) ||
      (typeof mod.profile === 'string' && mod.profile.length > 0)
    ) {
      return true;
    }
    if (typeof mod.label === 'string' && FALLBACK_LABEL_RE.test(mod.label.trim())) {
      return true;
    }
    return false;
  }

  function gpuHandlesMatch(actual, expectedRef, expectedToken) {
    if (actual === expectedRef) {
      return true;
    }
    const fast = resolveGpuHandleFast(actual);
    return fast !== null && fast.token === expectedToken;
  }

  function hashAndValidateBlendComponent(comp, tag, state) {
    if (!isPlainRecordObject(comp)) return false;
    for (const k in comp) {
      if (!Object.prototype.hasOwnProperty.call(comp, k)) continue;
      const v = comp[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k !== 'operation' && k !== 'srcFactor' && k !== 'dstFactor') return false;
      if (typeof v !== 'string') return false;
    }
    mixFastInt(state, tag);
    if (comp.operation !== undefined) {
      mixFastInt(state, 0x31);
      mixFastString(state, comp.operation);
    }
    if (comp.srcFactor !== undefined) {
      mixFastInt(state, 0x32);
      mixFastString(state, comp.srcFactor);
    }
    if (comp.dstFactor !== undefined) {
      mixFastInt(state, 0x33);
      mixFastString(state, comp.dstFactor);
    }
    return true;
  }

  function hashAndValidateStencilFace(face, tag, state) {
    if (!isPlainRecordObject(face)) return false;
    for (const k in face) {
      if (!Object.prototype.hasOwnProperty.call(face, k)) continue;
      const v = face[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k !== 'compare' && k !== 'failOp' && k !== 'depthFailOp' && k !== 'passOp') return false;
      if (typeof v !== 'string') return false;
    }
    mixFastInt(state, tag);
    if (face.compare !== undefined) {
      mixFastInt(state, 0x41);
      mixFastString(state, face.compare);
    }
    if (face.failOp !== undefined) {
      mixFastInt(state, 0x42);
      mixFastString(state, face.failOp);
    }
    if (face.depthFailOp !== undefined) {
      mixFastInt(state, 0x43);
      mixFastString(state, face.depthFailOp);
    }
    if (face.passOp !== undefined) {
      mixFastInt(state, 0x44);
      mixFastString(state, face.passOp);
    }
    return true;
  }

  /**
   * Single-pass, zero-allocation, key-order-invariant 53-bit integer hash for standard
   * `GPURenderPipelineDescriptor` objects. Returns `null` for non-standard, fallback, or
   * unnormalized descriptors so callers fall back safely to canonical normalization/hashing.
   *
   * @param {unknown} desc
   * @returns {number | null}
   */
  function computeFastRenderIntHash(desc) {
    if (!isPlainRecordObject(desc)) return null;
    if (desc.layout === undefined || desc.layout === null) return null;
    if (desc.depthStencil === null) return null;
    if (desc.vertex === undefined || !isPlainRecordObject(desc.vertex)) return null;

    // Validate top-level keys (strict WebGPU GPURenderPipelineDescriptor subset).
    for (const k in desc) {
      if (!Object.prototype.hasOwnProperty.call(desc, k)) continue;
      const v = desc[k];
      if (v === undefined || typeof v === 'function') continue;
      if (
        k !== 'layout' &&
        k !== 'label' &&
        k !== 'vertex' &&
        k !== 'fragment' &&
        k !== 'primitive' &&
        k !== 'depthStencil' &&
        k !== 'multisample'
      ) {
        return null;
      }
    }

    const layoutFast = resolveGpuHandleFast(desc.layout);
    if (layoutFast === null) return null;

    const state = { h1: 0x52454e44, h2: 0x13579bdf };
    mixFastInt(state, 0x101);
    mixFastGpuHandle(state, layoutFast);

    if (desc.label !== undefined) {
      if (typeof desc.label !== 'string') return null;
      mixFastInt(state, 0x102);
      mixFastString(state, desc.label);
    }

    // Vertex state
    const v = desc.vertex;
    for (const vk in v) {
      if (!Object.prototype.hasOwnProperty.call(v, vk)) continue;
      const vv = v[vk];
      if (vv === undefined || typeof vv === 'function') continue;
      if (vk === 'module' || vk === 'entryPoint' || vk === 'buffers') {
        continue;
      }
      return null;
    }
    mixFastInt(state, 0x200);
    if (v.module !== undefined) {
      if (isLikelyFallbackModule(v.module)) return null;
      const vsFast = resolveGpuHandleFast(v.module);
      if (vsFast === null) return null;
      mixFastInt(state, 0x201);
      mixFastGpuHandle(state, vsFast);
    }
    if (v.entryPoint !== undefined) {
      if (typeof v.entryPoint !== 'string') return null;
      mixFastInt(state, 0x202);
      mixFastString(state, v.entryPoint);
    }
    if (v.buffers !== undefined) {
      if (!Array.isArray(v.buffers)) return null;
      const bufs = v.buffers;
      mixFastInt(state, 0x203);
      mixFastInt(state, bufs.length);
      for (let i = 0; i < bufs.length; i += 1) {
        const b = bufs[i];
        if (b === null) {
          mixFastInt(state, 0x210);
          continue;
        }
        if (b === undefined) {
          mixFastInt(state, 0x211);
          continue;
        }
        if (!isPlainRecordObject(b)) return null;
        for (const bk in b) {
          if (!Object.prototype.hasOwnProperty.call(b, bk)) continue;
          const bv = b[bk];
          if (bv === undefined || typeof bv === 'function') continue;
          if (bk !== 'arrayStride' && bk !== 'stepMode' && bk !== 'attributes') return null;
        }
        mixFastInt(state, 0x212);
        if (b.arrayStride !== undefined) {
          if (typeof b.arrayStride !== 'number' || !Number.isFinite(b.arrayStride)) return null;
          mixFastInt(state, 0x213);
          mixFastNumber(state, b.arrayStride);
        }
        if (b.stepMode !== undefined) {
          if (typeof b.stepMode !== 'string') return null;
          mixFastInt(state, 0x214);
          mixFastString(state, b.stepMode);
        }
        if (b.attributes !== undefined) {
          if (!Array.isArray(b.attributes)) return null;
          const attrs = b.attributes;
          mixFastInt(state, 0x215);
          mixFastInt(state, attrs.length);
          for (let j = 0; j < attrs.length; j += 1) {
            const a = attrs[j];
            if (!isPlainRecordObject(a)) return null;
            for (const ak in a) {
              if (!Object.prototype.hasOwnProperty.call(a, ak)) continue;
              const av = a[ak];
              if (av === undefined || typeof av === 'function') continue;
              if (ak !== 'shaderLocation' && ak !== 'offset' && ak !== 'format') return null;
            }
            mixFastInt(state, 0x220);
            if (a.shaderLocation !== undefined) {
              if (typeof a.shaderLocation !== 'number' || !Number.isFinite(a.shaderLocation)) {
                return null;
              }
              mixFastInt(state, 0x221);
              mixFastNumber(state, a.shaderLocation);
            }
            if (a.offset !== undefined) {
              if (typeof a.offset !== 'number' || !Number.isFinite(a.offset)) return null;
              mixFastInt(state, 0x222);
              mixFastNumber(state, a.offset);
            }
            if (a.format !== undefined) {
              if (typeof a.format !== 'string') return null;
              mixFastInt(state, 0x223);
              mixFastString(state, a.format);
            }
          }
        }
      }
    }

    // Fragment state
    if (desc.fragment !== undefined) {
      const f = desc.fragment;
      if (!isPlainRecordObject(f)) return null;
      if (!Array.isArray(f.targets) || f.targets.length === 0) return null;
      for (const fk in f) {
        if (!Object.prototype.hasOwnProperty.call(f, fk)) continue;
        const fv = f[fk];
        if (fv === undefined || typeof fv === 'function') continue;
        if (fk === 'module' || fk === 'entryPoint' || fk === 'targets') {
          continue;
        }
        return null;
      }
      mixFastInt(state, 0x300);
      if (f.module !== undefined) {
        if (isLikelyFallbackModule(f.module)) return null;
        const fsFast = resolveGpuHandleFast(f.module);
        if (fsFast === null) return null;
        mixFastInt(state, 0x301);
        mixFastGpuHandle(state, fsFast);
      }
      if (f.entryPoint !== undefined) {
        if (typeof f.entryPoint !== 'string') return null;
        mixFastInt(state, 0x302);
        mixFastString(state, f.entryPoint);
      }
      const targets = f.targets;
      mixFastInt(state, 0x303);
      mixFastInt(state, targets.length);
      for (let i = 0; i < targets.length; i += 1) {
        const t = targets[i];
        if (t === null) {
          mixFastInt(state, 0x310);
          continue;
        }
        if (t === undefined || !isPlainRecordObject(t)) return null;
        if (t.blend === null) return null;
        for (const tk in t) {
          if (!Object.prototype.hasOwnProperty.call(t, tk)) continue;
          const tv = t[tk];
          if (tv === undefined || typeof tv === 'function') continue;
          if (tk !== 'format' && tk !== 'writeMask' && tk !== 'blend') return null;
        }
        mixFastInt(state, 0x312);
        if (t.format !== undefined) {
          if (typeof t.format !== 'string') return null;
          mixFastInt(state, 0x313);
          mixFastString(state, t.format);
        }
        if (t.writeMask !== undefined) {
          if (typeof t.writeMask !== 'number' || !Number.isFinite(t.writeMask)) return null;
          mixFastInt(state, 0x314);
          mixFastNumber(state, t.writeMask);
        }
        if (t.blend !== undefined) {
          if (!isPlainRecordObject(t.blend)) return null;
          for (const blk in t.blend) {
            if (!Object.prototype.hasOwnProperty.call(t.blend, blk)) continue;
            const blv = t.blend[blk];
            if (blv === undefined || typeof blv === 'function') continue;
            if (blk !== 'color' && blk !== 'alpha') return null;
          }
          mixFastInt(state, 0x315);
          if (t.blend.color !== undefined) {
            if (!hashAndValidateBlendComponent(t.blend.color, 0x316, state)) return null;
          }
          if (t.blend.alpha !== undefined) {
            if (!hashAndValidateBlendComponent(t.blend.alpha, 0x317, state)) return null;
          }
        }
      }
    }

    // Primitive state
    if (desc.primitive !== undefined) {
      const p = desc.primitive;
      if (!isPlainRecordObject(p)) return null;
      for (const pk in p) {
        if (!Object.prototype.hasOwnProperty.call(p, pk)) continue;
        const pv = p[pk];
        if (pv === undefined || typeof pv === 'function') continue;
        if (
          pk !== 'topology' &&
          pk !== 'stripIndexFormat' &&
          pk !== 'frontFace' &&
          pk !== 'cullMode' &&
          pk !== 'unclippedDepth'
        ) {
          return null;
        }
      }
      mixFastInt(state, 0x400);
      if (p.topology !== undefined) {
        if (typeof p.topology !== 'string') return null;
        mixFastInt(state, 0x401);
        mixFastString(state, p.topology);
      }
      if (p.stripIndexFormat !== undefined) {
        if (typeof p.stripIndexFormat !== 'string') return null;
        mixFastInt(state, 0x402);
        mixFastString(state, p.stripIndexFormat);
      }
      if (p.frontFace !== undefined) {
        if (typeof p.frontFace !== 'string') return null;
        mixFastInt(state, 0x403);
        mixFastString(state, p.frontFace);
      }
      if (p.cullMode !== undefined) {
        if (typeof p.cullMode !== 'string') return null;
        mixFastInt(state, 0x404);
        mixFastString(state, p.cullMode);
      }
      if (p.unclippedDepth !== undefined) {
        if (typeof p.unclippedDepth !== 'boolean') return null;
        mixFastInt(state, p.unclippedDepth ? 0x405 : 0x406);
      }
    }

    // Depth-stencil state
    if (desc.depthStencil !== undefined) {
      const ds = desc.depthStencil;
      if (!isPlainRecordObject(ds)) return null;
      for (const dk in ds) {
        if (!Object.prototype.hasOwnProperty.call(ds, dk)) continue;
        const dv = ds[dk];
        if (dv === undefined || typeof dv === 'function') continue;
        if (
          dk !== 'format' &&
          dk !== 'depthWriteEnabled' &&
          dk !== 'depthCompare' &&
          dk !== 'stencilFront' &&
          dk !== 'stencilBack' &&
          dk !== 'stencilReadMask' &&
          dk !== 'stencilWriteMask' &&
          dk !== 'depthBias' &&
          dk !== 'depthBiasSlopeScale' &&
          dk !== 'depthBiasClamp'
        ) {
          return null;
        }
      }
      mixFastInt(state, 0x500);
      if (ds.format !== undefined) {
        if (typeof ds.format !== 'string') return null;
        mixFastInt(state, 0x501);
        mixFastString(state, ds.format);
      }
      if (ds.depthWriteEnabled !== undefined) {
        if (typeof ds.depthWriteEnabled !== 'boolean') return null;
        mixFastInt(state, ds.depthWriteEnabled ? 0x502 : 0x503);
      }
      if (ds.depthCompare !== undefined) {
        if (typeof ds.depthCompare !== 'string') return null;
        mixFastInt(state, 0x504);
        mixFastString(state, ds.depthCompare);
      }
      if (ds.stencilFront !== undefined) {
        if (!hashAndValidateStencilFace(ds.stencilFront, 0x505, state)) return null;
      }
      if (ds.stencilBack !== undefined) {
        if (!hashAndValidateStencilFace(ds.stencilBack, 0x506, state)) return null;
      }
      if (ds.stencilReadMask !== undefined) {
        if (typeof ds.stencilReadMask !== 'number' || !Number.isFinite(ds.stencilReadMask)) {
          return null;
        }
        mixFastInt(state, 0x507);
        mixFastNumber(state, ds.stencilReadMask);
      }
      if (ds.stencilWriteMask !== undefined) {
        if (typeof ds.stencilWriteMask !== 'number' || !Number.isFinite(ds.stencilWriteMask)) {
          return null;
        }
        mixFastInt(state, 0x508);
        mixFastNumber(state, ds.stencilWriteMask);
      }
      if (ds.depthBias !== undefined) {
        if (typeof ds.depthBias !== 'number' || !Number.isFinite(ds.depthBias)) return null;
        mixFastInt(state, 0x509);
        mixFastNumber(state, ds.depthBias);
      }
      if (ds.depthBiasSlopeScale !== undefined) {
        if (
          typeof ds.depthBiasSlopeScale !== 'number' ||
          !Number.isFinite(ds.depthBiasSlopeScale)
        ) {
          return null;
        }
        mixFastInt(state, 0x50a);
        mixFastNumber(state, ds.depthBiasSlopeScale);
      }
      if (ds.depthBiasClamp !== undefined) {
        if (typeof ds.depthBiasClamp !== 'number' || !Number.isFinite(ds.depthBiasClamp)) {
          return null;
        }
        mixFastInt(state, 0x50b);
        mixFastNumber(state, ds.depthBiasClamp);
      }
    }

    // Multisample state
    if (desc.multisample !== undefined) {
      const ms = desc.multisample;
      if (!isPlainRecordObject(ms)) return null;
      for (const mk in ms) {
        if (!Object.prototype.hasOwnProperty.call(ms, mk)) continue;
        const mv = ms[mk];
        if (mv === undefined || typeof mv === 'function') continue;
        if (mk !== 'count' && mk !== 'mask' && mk !== 'alphaToCoverageEnabled') return null;
      }
      mixFastInt(state, 0x600);
      if (ms.count !== undefined) {
        if (typeof ms.count !== 'number' || !Number.isFinite(ms.count)) return null;
        mixFastInt(state, 0x601);
        mixFastNumber(state, ms.count);
      }
      if (ms.mask !== undefined) {
        if (typeof ms.mask !== 'number' || !Number.isFinite(ms.mask)) return null;
        mixFastInt(state, 0x602);
        mixFastNumber(state, ms.mask);
      }
      if (ms.alphaToCoverageEnabled !== undefined) {
        if (typeof ms.alphaToCoverageEnabled !== 'boolean') return null;
        mixFastInt(state, ms.alphaToCoverageEnabled ? 0x603 : 0x604);
      }
    }

    return finalizeFastHash53(state);
  }

  /**
   * Single-pass, zero-allocation, key-order-invariant 53-bit integer hash for standard
   * `GPUComputePipelineDescriptor` objects.
   *
   * @param {unknown} desc
   * @returns {number | null}
   */
  function computeFastComputeIntHash(desc) {
    if (!isPlainRecordObject(desc)) return null;
    if (desc.layout === undefined || desc.layout === null) return null;
    if (desc.compute === undefined || !isPlainRecordObject(desc.compute)) return null;

    for (const k in desc) {
      if (!Object.prototype.hasOwnProperty.call(desc, k)) continue;
      const v = desc[k];
      if (v === undefined || typeof v === 'function') continue;
      if (k !== 'layout' && k !== 'label' && k !== 'compute') return null;
    }

    const layoutFast = resolveGpuHandleFast(desc.layout);
    if (layoutFast === null) return null;

    const cs = desc.compute;
    for (const ck in cs) {
      if (!Object.prototype.hasOwnProperty.call(cs, ck)) continue;
      const cv = cs[ck];
      if (cv === undefined || typeof cv === 'function') continue;
      if (ck !== 'module' && ck !== 'entryPoint') return null;
    }

    const state = { h1: 0x434f4d50, h2: 0x2468ace0 };
    mixFastInt(state, 0x701);
    mixFastGpuHandle(state, layoutFast);

    if (desc.label !== undefined) {
      if (typeof desc.label !== 'string') return null;
      mixFastInt(state, 0x702);
      mixFastString(state, desc.label);
    }
    if (cs.module !== undefined) {
      const csFast = resolveGpuHandleFast(cs.module);
      if (csFast === null) return null;
      mixFastInt(state, 0x703);
      mixFastGpuHandle(state, csFast);
    }
    if (cs.entryPoint !== undefined) {
      if (typeof cs.entryPoint !== 'string') return null;
      mixFastInt(state, 0x704);
      mixFastString(state, cs.entryPoint);
    }

    return finalizeFastHash53(state);
  }

  /**
   * Compute a fast 53-bit integer hash for a standard WebGPU render or compute pipeline
   * descriptor without string serialization, or return `null` when the descriptor requires
   * canonical hashing.
   *
   * @param {unknown} descriptor
   * @returns {number | null}
   */
  function computeFastDescriptorHash(descriptor) {
    if (!descriptor || typeof descriptor !== 'object') {
      return null;
    }
    if (descriptor.vertex !== undefined) {
      return computeFastRenderIntHash(descriptor);
    }
    if (descriptor.compute !== undefined) {
      return computeFastComputeIntHash(descriptor);
    }
    return null;
  }

  function captureBlendComponentSnapshot(comp) {
    if (comp === undefined) return undefined;
    return {
      operation: comp.operation,
      srcFactor: comp.srcFactor,
      dstFactor: comp.dstFactor,
    };
  }

  function matchesBlendComponentSnapshot(comp, snap) {
    if (snap === undefined) return comp === undefined;
    if (!comp || typeof comp !== 'object') return false;
    return (
      comp.operation === snap.operation &&
      comp.srcFactor === snap.srcFactor &&
      comp.dstFactor === snap.dstFactor
    );
  }

  function captureStencilFaceSnapshot(face) {
    if (face === undefined) return undefined;
    return {
      compare: face.compare,
      failOp: face.failOp,
      depthFailOp: face.depthFailOp,
      passOp: face.passOp,
    };
  }

  function matchesStencilFaceSnapshot(face, snap) {
    if (snap === undefined) return face === undefined;
    if (!face || typeof face !== 'object') return false;
    return (
      face.compare === snap.compare &&
      face.failOp === snap.failOp &&
      face.depthFailOp === snap.depthFailOp &&
      face.passOp === snap.passOp
    );
  }

  function captureRenderSnapshot(desc) {
    if (!isPlainRecordObject(desc) || !isPlainRecordObject(desc.vertex)) {
      return null;
    }
    const layoutFast =
      desc.layout !== undefined && desc.layout !== null ? resolveGpuHandleFast(desc.layout) : null;
    if (desc.layout !== undefined && desc.layout !== null && layoutFast === null) {
      return null;
    }
    const v = desc.vertex;
    if (v.constants !== undefined) return null;
    const vsFast =
      v.module !== undefined && v.module !== null ? resolveGpuHandleFast(v.module) : null;
    if (v.module !== undefined && v.module !== null && vsFast === null) {
      return null;
    }

    let vBuffers = undefined;
    if (v.buffers !== undefined) {
      if (!Array.isArray(v.buffers)) return null;
      vBuffers = new Array(v.buffers.length);
      for (let i = 0; i < v.buffers.length; i += 1) {
        const b = v.buffers[i];
        if (b === null || b === undefined) {
          vBuffers[i] = b;
        } else {
          if (!isPlainRecordObject(b)) return null;
          let attrs = undefined;
          if (b.attributes !== undefined) {
            if (!Array.isArray(b.attributes)) return null;
            attrs = new Array(b.attributes.length);
            for (let j = 0; j < b.attributes.length; j += 1) {
              const a = b.attributes[j];
              if (!isPlainRecordObject(a)) return null;
              attrs[j] = {
                shaderLocation: a.shaderLocation,
                offset: a.offset,
                format: a.format,
              };
            }
          }
          vBuffers[i] = {
            arrayStride: b.arrayStride,
            stepMode: b.stepMode,
            attributes: attrs,
          };
        }
      }
    }

    let fragmentSnap = undefined;
    if (desc.fragment !== undefined) {
      const f = desc.fragment;
      if (!isPlainRecordObject(f) || f.constants !== undefined) return null;
      const fsFast =
        f.module !== undefined && f.module !== null ? resolveGpuHandleFast(f.module) : null;
      if (f.module !== undefined && f.module !== null && fsFast === null) {
        return null;
      }
      let targetsSnap = undefined;
      if (f.targets !== undefined) {
        if (!Array.isArray(f.targets)) return null;
        targetsSnap = new Array(f.targets.length);
        for (let i = 0; i < f.targets.length; i += 1) {
          const t = f.targets[i];
          if (t === null || t === undefined) {
            targetsSnap[i] = t;
          } else {
            if (!isPlainRecordObject(t)) return null;
            let blendSnap = undefined;
            if (t.blend === null) {
              blendSnap = null;
            } else if (t.blend !== undefined) {
              if (!isPlainRecordObject(t.blend)) return null;
              blendSnap = {
                color: captureBlendComponentSnapshot(t.blend.color),
                alpha: captureBlendComponentSnapshot(t.blend.alpha),
              };
            }
            targetsSnap[i] = {
              format: t.format,
              writeMask: t.writeMask,
              blend: blendSnap,
            };
          }
        }
      }
      fragmentSnap = {
        moduleRef: f.module,
        moduleToken: fsFast ? fsFast.token : null,
        entryPoint: f.entryPoint,
        targets: targetsSnap,
      };
    }

    let primitiveSnap = undefined;
    if (desc.primitive !== undefined) {
      const p = desc.primitive;
      if (!isPlainRecordObject(p)) return null;
      primitiveSnap = {
        topology: p.topology,
        stripIndexFormat: p.stripIndexFormat,
        frontFace: p.frontFace,
        cullMode: p.cullMode,
        unclippedDepth: p.unclippedDepth,
      };
    }

    let dsSnap = undefined;
    if (desc.depthStencil === null) {
      dsSnap = null;
    } else if (desc.depthStencil !== undefined) {
      const ds = desc.depthStencil;
      if (!isPlainRecordObject(ds)) return null;
      dsSnap = {
        format: ds.format,
        depthWriteEnabled: ds.depthWriteEnabled,
        depthCompare: ds.depthCompare,
        stencilFront: captureStencilFaceSnapshot(ds.stencilFront),
        stencilBack: captureStencilFaceSnapshot(ds.stencilBack),
        stencilReadMask: ds.stencilReadMask,
        stencilWriteMask: ds.stencilWriteMask,
        depthBias: ds.depthBias,
        depthBiasSlopeScale: ds.depthBiasSlopeScale,
        depthBiasClamp: ds.depthBiasClamp,
      };
    }

    let msSnap = undefined;
    if (desc.multisample !== undefined) {
      const ms = desc.multisample;
      if (!isPlainRecordObject(ms)) return null;
      msSnap = {
        count: ms.count,
        mask: ms.mask,
        alphaToCoverageEnabled: ms.alphaToCoverageEnabled,
      };
    }

    return {
      layoutRef: desc.layout,
      layoutToken: layoutFast ? layoutFast.token : null,
      label: desc.label,
      fallbackProfile: desc.fallbackProfile,
      vertexStride: desc.vertexStride,
      vertex: {
        moduleRef: v.module,
        moduleToken: vsFast ? vsFast.token : null,
        entryPoint: v.entryPoint,
        buffers: vBuffers,
      },
      fragment: fragmentSnap,
      primitive: primitiveSnap,
      depthStencil: dsSnap,
      multisample: msSnap,
    };
  }

  function matchesRenderSnapshot(desc, snap) {
    if (!desc || typeof desc !== 'object') return false;
    if (
      desc.label !== snap.label ||
      desc.fallbackProfile !== snap.fallbackProfile ||
      desc.vertexStride !== snap.vertexStride
    ) {
      return false;
    }
    if (snap.layoutToken === null) {
      if (desc.layout !== snap.layoutRef) return false;
    } else if (!gpuHandlesMatch(desc.layout, snap.layoutRef, snap.layoutToken)) {
      return false;
    }

    // Vertex
    const v = desc.vertex;
    if (!v || typeof v !== 'object' || v.constants !== undefined) return false;
    const sv = snap.vertex;
    if (v.entryPoint !== sv.entryPoint) return false;
    if (sv.moduleToken === null) {
      if (v.module !== sv.moduleRef) return false;
    } else if (!gpuHandlesMatch(v.module, sv.moduleRef, sv.moduleToken)) {
      return false;
    }
    if (sv.buffers === undefined) {
      if (v.buffers !== undefined) return false;
    } else {
      const bufs = v.buffers;
      const sbufs = sv.buffers;
      if (!Array.isArray(bufs) || bufs.length !== sbufs.length) return false;
      for (let i = 0; i < sbufs.length; i += 1) {
        const b = bufs[i];
        const sb = sbufs[i];
        if (sb === null || sb === undefined) {
          if (b !== sb) return false;
          continue;
        }
        if (!b || typeof b !== 'object') return false;
        if (b.arrayStride !== sb.arrayStride || b.stepMode !== sb.stepMode) return false;
        if (sb.attributes === undefined) {
          if (b.attributes !== undefined) return false;
        } else {
          const attrs = b.attributes;
          const sattrs = sb.attributes;
          if (!Array.isArray(attrs) || attrs.length !== sattrs.length) return false;
          for (let j = 0; j < sattrs.length; j += 1) {
            const a = attrs[j];
            const sa = sattrs[j];
            if (
              !a ||
              typeof a !== 'object' ||
              a.shaderLocation !== sa.shaderLocation ||
              a.offset !== sa.offset ||
              a.format !== sa.format
            ) {
              return false;
            }
          }
        }
      }
    }

    // Fragment
    const sf = snap.fragment;
    if (sf === undefined) {
      if (desc.fragment !== undefined) return false;
    } else {
      const f = desc.fragment;
      if (!f || typeof f !== 'object' || f.constants !== undefined) return false;
      if (f.entryPoint !== sf.entryPoint) return false;
      if (sf.moduleToken === null) {
        if (f.module !== sf.moduleRef) return false;
      } else if (!gpuHandlesMatch(f.module, sf.moduleRef, sf.moduleToken)) {
        return false;
      }
      if (sf.targets === undefined) {
        if (f.targets !== undefined) return false;
      } else {
        const targets = f.targets;
        const stargets = sf.targets;
        if (!Array.isArray(targets) || targets.length !== stargets.length) return false;
        for (let i = 0; i < stargets.length; i += 1) {
          const t = targets[i];
          const st = stargets[i];
          if (st === null || st === undefined) {
            if (t !== st) return false;
            continue;
          }
          if (!t || typeof t !== 'object') return false;
          if (t.format !== st.format || t.writeMask !== st.writeMask) return false;
          if (st.blend === undefined || st.blend === null) {
            if (t.blend !== st.blend) return false;
          } else {
            if (!t.blend || typeof t.blend !== 'object') return false;
            if (
              !matchesBlendComponentSnapshot(t.blend.color, st.blend.color) ||
              !matchesBlendComponentSnapshot(t.blend.alpha, st.blend.alpha)
            ) {
              return false;
            }
          }
        }
      }
    }

    // Primitive
    const sp = snap.primitive;
    if (sp === undefined) {
      if (desc.primitive !== undefined) return false;
    } else {
      const p = desc.primitive;
      if (
        !p ||
        typeof p !== 'object' ||
        p.topology !== sp.topology ||
        p.stripIndexFormat !== sp.stripIndexFormat ||
        p.frontFace !== sp.frontFace ||
        p.cullMode !== sp.cullMode ||
        p.unclippedDepth !== sp.unclippedDepth
      ) {
        return false;
      }
    }

    // Depth-stencil
    const sds = snap.depthStencil;
    if (sds === undefined || sds === null) {
      if (desc.depthStencil !== sds) return false;
    } else {
      const ds = desc.depthStencil;
      if (
        !ds ||
        typeof ds !== 'object' ||
        ds.format !== sds.format ||
        ds.depthWriteEnabled !== sds.depthWriteEnabled ||
        ds.depthCompare !== sds.depthCompare ||
        ds.stencilReadMask !== sds.stencilReadMask ||
        ds.stencilWriteMask !== sds.stencilWriteMask ||
        ds.depthBias !== sds.depthBias ||
        ds.depthBiasSlopeScale !== sds.depthBiasSlopeScale ||
        ds.depthBiasClamp !== sds.depthBiasClamp ||
        !matchesStencilFaceSnapshot(ds.stencilFront, sds.stencilFront) ||
        !matchesStencilFaceSnapshot(ds.stencilBack, sds.stencilBack)
      ) {
        return false;
      }
    }

    // Multisample
    const sms = snap.multisample;
    if (sms === undefined) {
      if (desc.multisample !== undefined) return false;
    } else {
      const ms = desc.multisample;
      if (
        !ms ||
        typeof ms !== 'object' ||
        ms.count !== sms.count ||
        ms.mask !== sms.mask ||
        ms.alphaToCoverageEnabled !== sms.alphaToCoverageEnabled
      ) {
        return false;
      }
    }

    return true;
  }

  function captureComputeSnapshot(desc) {
    if (!isPlainRecordObject(desc) || !isPlainRecordObject(desc.compute)) {
      return null;
    }
    const layoutFast =
      desc.layout !== undefined && desc.layout !== null ? resolveGpuHandleFast(desc.layout) : null;
    if (desc.layout !== undefined && desc.layout !== null && layoutFast === null) {
      return null;
    }
    const cs = desc.compute;
    if (cs.constants !== undefined) return null;
    const csFast =
      cs.module !== undefined && cs.module !== null ? resolveGpuHandleFast(cs.module) : null;
    if (cs.module !== undefined && cs.module !== null && csFast === null) {
      return null;
    }
    return {
      layoutRef: desc.layout,
      layoutToken: layoutFast ? layoutFast.token : null,
      label: desc.label,
      moduleRef: cs.module,
      moduleToken: csFast ? csFast.token : null,
      entryPoint: cs.entryPoint,
    };
  }

  function matchesComputeSnapshot(desc, snap) {
    if (!desc || typeof desc !== 'object') return false;
    if (desc.label !== snap.label) return false;
    if (snap.layoutToken === null) {
      if (desc.layout !== snap.layoutRef) return false;
    } else if (!gpuHandlesMatch(desc.layout, snap.layoutRef, snap.layoutToken)) {
      return false;
    }
    const cs = desc.compute;
    if (!cs || typeof cs !== 'object' || cs.constants !== undefined) return false;
    if (cs.entryPoint !== snap.entryPoint) return false;
    if (snap.moduleToken === null) {
      if (cs.module !== snap.moduleRef) return false;
    } else if (!gpuHandlesMatch(cs.module, snap.moduleRef, snap.moduleToken)) {
      return false;
    }
    return true;
  }

  function normalizeVisibility(visibility) {
    if (typeof visibility === 'number') {
      if (!Number.isInteger(visibility) || visibility < 0) {
        throw new RangeError(`Invalid shader stage visibility bitmask: ${visibility}`);
      }
      return visibility;
    }

    if (typeof visibility === 'string') {
      const tokens = visibility
        .split(/[\s|,+/]+/)
        .map((s) => s.trim().toLowerCase())
        .filter((s) => s.length > 0);
      if (tokens.length === 0) {
        throw new RangeError('Shader stage visibility string must not be empty');
      }
      let mask = 0;
      for (let i = 0; i < tokens.length; i += 1) {
        const bit = STAGE_NAME_TO_BIT[tokens[i]];
        if (bit === undefined) {
          throw new RangeError(`Unknown shader stage visibility token: "${tokens[i]}"`);
        }
        mask |= bit;
      }
      return mask;
    }

    if (Array.isArray(visibility)) {
      if (visibility.length === 0) {
        throw new RangeError('Shader stage visibility array must not be empty');
      }
      let mask = 0;
      for (let i = 0; i < visibility.length; i += 1) {
        mask |= normalizeVisibility(visibility[i]);
      }
      return mask;
    }

    throw new TypeError('Binding visibility must be a GPUShaderStage bitmask, stage string, or stage array');
  }

  function isRtArenaReflectionBinding(raw) {
    if (!raw || typeof raw !== 'object') return false;
    const src = raw.source;
    return Boolean(src && typeof src === 'object' && (src.rt_arena === true || src.kind === 'rt_arena'));
  }

  function normalizeReflectionBinding(raw, idx, options = null) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      throw new TypeError(`Reflection binding at index ${idx} must be an object`);
    }
    const sourceEntry =
      raw.layoutEntry && typeof raw.layoutEntry === 'object' && !Array.isArray(raw.layoutEntry)
        ? raw.layoutEntry
        : raw;

    const bindingIndex =
      Number.isInteger(sourceEntry.binding) && sourceEntry.binding >= 0
        ? sourceEntry.binding
        : raw.binding;
    if (!Number.isInteger(bindingIndex) || bindingIndex < 0) {
      throw new RangeError(`Reflection binding at index ${idx} must have a non-negative integer "binding"`);
    }

    const rawVisibility =
      sourceEntry.visibility !== undefined ? sourceEntry.visibility : raw.visibility;
    const entry = {
      binding: bindingIndex,
      visibility: normalizeVisibility(rawVisibility),
    };

    if (sourceEntry.buffer && typeof sourceEntry.buffer === 'object') {
      const b = {};
      if (sourceEntry.buffer.type !== undefined) b.type = sourceEntry.buffer.type;
      const hasDynamicOffset =
        sourceEntry.buffer.hasDynamicOffset !== undefined
          ? sourceEntry.buffer.hasDynamicOffset
          : sourceEntry.buffer.has_dynamic_offset;
      if (hasDynamicOffset !== undefined) b.hasDynamicOffset = Boolean(hasDynamicOffset);
      const minBindingSize =
        sourceEntry.buffer.minBindingSize !== undefined
          ? sourceEntry.buffer.minBindingSize
          : sourceEntry.buffer.min_binding_size;
      if (minBindingSize !== undefined) b.minBindingSize = Number(minBindingSize);
      entry.buffer = b;
    }

    if (sourceEntry.sampler && typeof sourceEntry.sampler === 'object') {
      const s = {};
      if (sourceEntry.sampler.type !== undefined) {
        s.type = normalizeDefaultSamplerKind(sourceEntry.sampler.type);
      }
      entry.sampler = s;
    }

    if (sourceEntry.texture && typeof sourceEntry.texture === 'object') {
      const tx = {};
      const sampleType =
        sourceEntry.texture.sampleType !== undefined
          ? sourceEntry.texture.sampleType
          : sourceEntry.texture.sample_type;
      if (sampleType !== undefined) tx.sampleType = sampleType;
      const viewDimension =
        sourceEntry.texture.viewDimension !== undefined
          ? sourceEntry.texture.viewDimension
          : sourceEntry.texture.view_dimension;
      if (viewDimension !== undefined) tx.viewDimension = viewDimension;
      if (sourceEntry.texture.multisampled !== undefined) {
        tx.multisampled = Boolean(sourceEntry.texture.multisampled);
      }
      entry.texture = tx;
    }

    const rawStorageTex =
      sourceEntry.storageTexture !== undefined
        ? sourceEntry.storageTexture
        : sourceEntry.storage_texture;
    if (rawStorageTex && typeof rawStorageTex === 'object') {
      const st = {};
      if (rawStorageTex.access !== undefined) st.access = rawStorageTex.access;
      if (rawStorageTex.format !== undefined) st.format = rawStorageTex.format;
      const viewDimension =
        rawStorageTex.viewDimension !== undefined
          ? rawStorageTex.viewDimension
          : rawStorageTex.view_dimension;
      if (viewDimension !== undefined) st.viewDimension = viewDimension;
      entry.storageTexture = st;
    }

    const rawExternalTex =
      sourceEntry.externalTexture !== undefined
        ? sourceEntry.externalTexture
        : sourceEntry.external_texture;
    if (rawExternalTex && typeof rawExternalTex === 'object') {
      entry.externalTexture = {};
    }

    // W379: Support v3 lane-indexed heap binding arrays when explicitly opted in.
    if (options && options.bindingArrays === true) {
      const explicitArraySize =
        sourceEntry.bindingArraySize !== undefined
          ? sourceEntry.bindingArraySize
          : sourceEntry.binding_array_size !== undefined
            ? sourceEntry.binding_array_size
            : raw.source &&
                typeof raw.source === 'object' &&
                (raw.source.kind === 'lane_indexed_resource_heap' ||
                  raw.source.kind === 'lane_indexed_sampler_heap') &&
                Number.isInteger(raw.source.maxCount)
              ? raw.source.maxCount
              : undefined;
      if (Number.isInteger(explicitArraySize) && explicitArraySize >= 1 && explicitArraySize <= 256) {
        entry.bindingArraySize = explicitArraySize;
      }
    }

    return entry;
  }

  function validatePositiveInt(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got: ${value}`);
    }
    return value;
  }

  function alignUp4(value) {
    return Math.ceil(value / 4) * 4;
  }

  function alignDown(value, align) {
    if (align <= 1) return value;
    return value - (value % align);
  }

  /**
   * Normalize a fallback sampler kind or alias to a canonical WebGPU `GPUSamplerBindingType`
   * (`'filtering'`, `'non-filtering'`, or `'comparison'`). Unknown or omitted values safely
   * resolve to `'filtering'` (linear clamp-to-edge).
   *
   * @param {string | null | undefined} kind
   * @returns {'filtering' | 'non-filtering' | 'comparison'}
   */
  function normalizeDefaultSamplerKind(kind) {
    if (typeof kind !== 'string') {
      return 'filtering';
    }
    const key = kind.trim().toLowerCase();
    if (key.length === 0) {
      return 'filtering';
    }
    if (Object.prototype.hasOwnProperty.call(DEFAULT_SAMPLER_KIND_ALIASES, key)) {
      return DEFAULT_SAMPLER_KIND_ALIASES[key];
    }
    return 'filtering';
  }

  /**
   * Build a fresh mutable `GPUSamplerDescriptor` for a canonical default fallback sampler kind:
   * - `'filtering'`: linear min/mag/mipmap, clamp-to-edge
   * - `'non-filtering'`: nearest min/mag/mipmap, clamp-to-edge
   * - `'comparison'`: linear min/mag, nearest mipmap, clamp-to-edge, compare `'less-equal'`
   *
   * @param {string | null | undefined} kind
   * @returns {GPUSamplerDescriptor}
   */
  function buildDefaultSamplerDescriptor(kind) {
    const canonical = normalizeDefaultSamplerKind(kind);
    return { ...DEFAULT_SAMPLER_DESCRIPTORS[canonical] };
  }

  /**
   * Return true when a WebGPU color texture format uses sRGB transfer encoding
   * (e.g., `'rgba8unorm-srgb'`, `'bgra8unorm-srgb'`).
   *
   * @param {unknown} format
   * @returns {boolean}
   */
  function isSrgbColorFormat(format) {
    return typeof format === 'string' && format.trim().toLowerCase().endsWith('-srgb');
  }

  function descriptorHasBlendState(descriptor) {
    if (!descriptor || typeof descriptor !== 'object') {
      return false;
    }
    const targets = descriptor.fragment && descriptor.fragment.targets;
    if (!Array.isArray(targets)) {
      return false;
    }
    for (let i = 0; i < targets.length; i += 1) {
      const target = targets[i];
      if (target && typeof target === 'object' && target.blend && typeof target.blend === 'object') {
        return true;
      }
    }
    return false;
  }

  function descriptorHasSrgbTarget(descriptor) {
    if (!descriptor || typeof descriptor !== 'object') {
      return false;
    }
    const targets = descriptor.fragment && descriptor.fragment.targets;
    if (!Array.isArray(targets)) {
      return false;
    }
    for (let i = 0; i < targets.length; i += 1) {
      const target = targets[i];
      if (target && typeof target === 'object' && isSrgbColorFormat(target.format)) {
        return true;
      }
    }
    return false;
  }

  function entryHasInlineSampler(entry) {
    if (!entry || typeof entry !== 'object') {
      return false;
    }
    if (entry.sampler && typeof entry.sampler === 'object') {
      return true;
    }
    if (
      entry.inlineSampler === true ||
      (entry.inlineSampler && typeof entry.inlineSampler === 'object') ||
      entry.inline_sampler === true ||
      (entry.inline_sampler && typeof entry.inline_sampler === 'object') ||
      (Array.isArray(entry.inlineSamplerSdwords) && entry.inlineSamplerSdwords.length > 0)
    ) {
      return true;
    }
    if (entry.source && typeof entry.source === 'object') {
      if (
        entry.source.kind === 'inline_sampler' ||
        (entry.source.inline_sampler && typeof entry.source.inline_sampler === 'object') ||
        (entry.source.inlineSampler && typeof entry.source.inlineSampler === 'object')
      ) {
        return true;
      }
    }
    return false;
  }

  function descriptorHasInlineSampler(descriptor) {
    if (!descriptor || typeof descriptor !== 'object') {
      return false;
    }
    if (
      descriptor.inlineSampler === true ||
      descriptor.inline_sampler === true ||
      descriptor.hasInlineSampler === true ||
      descriptor.hasInlineSamplers === true ||
      (typeof descriptor.inlineSamplerBindings === 'number' &&
        Number.isFinite(descriptor.inlineSamplerBindings) &&
        descriptor.inlineSamplerBindings > 0)
    ) {
      return true;
    }
    const entries = Array.isArray(descriptor)
      ? descriptor
      : Array.isArray(descriptor.entries)
        ? descriptor.entries
        : Array.isArray(descriptor.bindings)
          ? descriptor.bindings
          : null;
    if (!entries) {
      return false;
    }
    for (let i = 0; i < entries.length; i += 1) {
      if (entryHasInlineSampler(entries[i])) {
        return true;
      }
    }
    return false;
  }

  function computeClampedHitRate(hits, misses) {
    const h = typeof hits === 'number' && Number.isFinite(hits) && hits > 0 ? hits : 0;
    const m = typeof misses === 'number' && Number.isFinite(misses) && misses > 0 ? misses : 0;
    const total = h + m;
    if (!(total > 0)) {
      return 0;
    }
    const rate = h / total;
    if (!Number.isFinite(rate)) {
      return 0;
    }
    return rate < 0 ? 0 : rate > 1 ? 1 : rate;
  }

  function normalizeBlendComponentInternal(rawComp, addReason) {
    if (!rawComp || typeof rawComp !== 'object' || Array.isArray(rawComp)) {
      addReason('normalized-blend-component');
      return {
        component: { operation: 'add', srcFactor: 'one', dstFactor: 'zero' },
        changed: true,
      };
    }

    let changed = false;
    let operation = rawComp.operation;
    if (typeof operation !== 'string' || !VALID_BLEND_OPERATIONS.has(operation)) {
      operation = 'add';
      changed = true;
      addReason('normalized-blend-component');
    }

    let srcFactor = rawComp.srcFactor;
    if (typeof srcFactor !== 'string' || !VALID_BLEND_FACTORS.has(srcFactor)) {
      srcFactor = 'one';
      changed = true;
      addReason('normalized-blend-component');
    }

    let dstFactor = rawComp.dstFactor;
    if (typeof dstFactor !== 'string' || !VALID_BLEND_FACTORS.has(dstFactor)) {
      dstFactor = operation === 'min' || operation === 'max' ? 'one' : 'zero';
      changed = true;
      addReason('normalized-blend-component');
    }

    if (!changed) {
      return { component: rawComp, changed: false };
    }

    return {
      component: { operation, srcFactor, dstFactor },
      changed: true,
    };
  }

  function normalizeColorTargetsInternal(targets, defaultColorFormat) {
    const reasons = [];
    const addReason = (r) => {
      if (!reasons.includes(r)) {
        reasons.push(r);
      }
    };

    if (!Array.isArray(targets) || targets.length === 0) {
      addReason('default-color-target');
      return {
        targets: [{ format: defaultColorFormat }],
        changed: true,
        reasons,
      };
    }

    let changed = false;
    const normalizedTargets = new Array(targets.length);

    for (let i = 0; i < targets.length; i += 1) {
      const rawTarget = targets[i];
      if (rawTarget === null) {
        normalizedTargets[i] = null;
        continue;
      }
      if (rawTarget === undefined) {
        normalizedTargets[i] = null;
        changed = true;
        addReason('normalized-color-target');
        continue;
      }
      if (typeof rawTarget !== 'object' || Array.isArray(rawTarget)) {
        normalizedTargets[i] = { format: defaultColorFormat };
        changed = true;
        addReason('normalized-color-target');
        continue;
      }

      let targetChanged = false;
      let nextFormat = rawTarget.format;
      if (typeof nextFormat !== 'string' || nextFormat.trim().length === 0) {
        nextFormat = defaultColorFormat;
        targetChanged = true;
        addReason('default-color-target-format');
      } else {
        const lowerFmt = nextFormat.trim().toLowerCase();
        if (lowerFmt !== nextFormat) {
          nextFormat = lowerFmt;
          targetChanged = true;
          addReason('normalized-color-target-format');
        }
      }

      let nextBlend = rawTarget.blend;
      let stripBlend = false;
      if (rawTarget.blend === null) {
        stripBlend = true;
        targetChanged = true;
        addReason('null-blend-cleared');
      } else if (rawTarget.blend !== undefined) {
        if (typeof rawTarget.blend !== 'object' || Array.isArray(rawTarget.blend)) {
          stripBlend = true;
          targetChanged = true;
          addReason('invalid-blend-cleared');
        } else {
          const colorRes = normalizeBlendComponentInternal(rawTarget.blend.color, addReason);
          const alphaRes = normalizeBlendComponentInternal(rawTarget.blend.alpha, addReason);
          if (colorRes.changed || alphaRes.changed) {
            nextBlend = {
              color: colorRes.component,
              alpha: alphaRes.component,
            };
            targetChanged = true;
          }
        }
      }

      let nextWriteMask = rawTarget.writeMask;
      if (rawTarget.writeMask !== undefined) {
        if (
          typeof rawTarget.writeMask !== 'number' ||
          !Number.isInteger(rawTarget.writeMask) ||
          rawTarget.writeMask < 0 ||
          rawTarget.writeMask > 0xf
        ) {
          nextWriteMask =
            typeof rawTarget.writeMask === 'number' && Number.isFinite(rawTarget.writeMask)
              ? Math.trunc(rawTarget.writeMask) & 0xf
              : 0xf;
          targetChanged = true;
          addReason('clamped-write-mask');
        }
      }

      if (!targetChanged) {
        normalizedTargets[i] = rawTarget;
      } else {
        changed = true;
        const copy = { ...rawTarget, format: nextFormat };
        if (stripBlend) {
          delete copy.blend;
        } else if (nextBlend !== undefined) {
          copy.blend = nextBlend;
        }
        if (nextWriteMask !== undefined) {
          copy.writeMask = nextWriteMask;
        }
        normalizedTargets[i] = copy;
      }
    }

    if (!changed) {
      return { targets, changed: false, reasons };
    }
    return { targets: normalizedTargets, changed: true, reasons };
  }

  /**
   * Normalize a built-in fallback profile name or alias to a canonical G34 profile key.
   * Unknown or omitted values safely resolve to `'default'`.
   *
   * @param {string | null | undefined} profile
   * @returns {'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b'}
   */
  function normalizeFallbackProfile(profile) {
    if (typeof profile !== 'string') {
      return 'default';
    }
    const key = profile.trim().toLowerCase();
    if (key.length === 0) {
      return 'default';
    }
    if (Object.prototype.hasOwnProperty.call(FALLBACK_PROFILE_ALIASES, key)) {
      return FALLBACK_PROFILE_ALIASES[key];
    }
    return 'default';
  }

  function detectModuleFallbackProfile(mod) {
    if (!mod || typeof mod !== 'object') {
      return null;
    }
    if (typeof mod.fallbackProfile === 'string' && mod.fallbackProfile.trim().length > 0) {
      return normalizeFallbackProfile(mod.fallbackProfile);
    }
    if (typeof mod.profile === 'string' && mod.profile.trim().length > 0) {
      return normalizeFallbackProfile(mod.profile);
    }
    if (typeof mod.label === 'string') {
      const m = FALLBACK_LABEL_RE.exec(mod.label.trim());
      if (m) {
        return normalizeFallbackProfile(m[1]);
      }
    }
    if (mod.isFallback === true || mod.source === 'builtin') {
      return 'default';
    }
    return null;
  }

  function detectFallbackProfile(descriptor, options) {
    const opts = options && typeof options === 'object' ? options : null;
    if (opts) {
      if (typeof opts.fallbackProfile === 'string' && opts.fallbackProfile.trim().length > 0) {
        return normalizeFallbackProfile(opts.fallbackProfile);
      }
      if (typeof opts.profile === 'string' && opts.profile.trim().length > 0) {
        return normalizeFallbackProfile(opts.profile);
      }
    }
    if (descriptor && typeof descriptor === 'object') {
      if (
        typeof descriptor.fallbackProfile === 'string' &&
        descriptor.fallbackProfile.trim().length > 0
      ) {
        return normalizeFallbackProfile(descriptor.fallbackProfile);
      }
      const vsProfile =
        descriptor.vertex && typeof descriptor.vertex === 'object'
          ? detectModuleFallbackProfile(descriptor.vertex.module)
          : null;
      const fsProfile =
        descriptor.fragment && typeof descriptor.fragment === 'object'
          ? detectModuleFallbackProfile(descriptor.fragment.module)
          : null;
      if (vsProfile && vsProfile !== 'default') {
        return vsProfile;
      }
      if (fsProfile && fsProfile !== 'default') {
        return fsProfile;
      }
      if (vsProfile || fsProfile) {
        return vsProfile || fsProfile;
      }
    }
    if (opts && opts.isFallback === true) {
      return 'default';
    }
    return null;
  }

  /**
   * Return the byte size of a valid WebGPU `GPUVertexFormat`, or `0` if unrecognized.
   *
   * @param {string} format
   * @returns {number}
   */
  function getVertexFormatByteSize(format) {
    if (
      typeof format === 'string' &&
      Object.prototype.hasOwnProperty.call(VERTEX_FORMAT_BYTE_SIZES, format)
    ) {
      return VERTEX_FORMAT_BYTE_SIZES[format];
    }
    return 0;
  }

  function cloneFrame1VertexLayout24B(stepMode) {
    return {
      arrayStride: FRAME1_VERTEX_STRIDE_24B,
      stepMode: stepMode === 'instance' ? 'instance' : 'vertex',
      attributes: [
        { shaderLocation: 0, offset: 0, format: 'float32x4' },
        { shaderLocation: 1, offset: 16, format: 'float32x2' },
      ],
    };
  }

  function isCanonical24BLayoutList(buffers) {
    if (!Array.isArray(buffers) || buffers.length !== 1) {
      return false;
    }
    const buf = buffers[0];
    if (!buf || typeof buf !== 'object') {
      return false;
    }
    if (buf.arrayStride !== FRAME1_VERTEX_STRIDE_24B) {
      return false;
    }
    if (buf.stepMode !== undefined && buf.stepMode !== 'vertex') {
      return false;
    }
    if (!Array.isArray(buf.attributes) || buf.attributes.length !== 2) {
      return false;
    }
    const a0 = buf.attributes[0];
    const a1 = buf.attributes[1];
    return (
      Boolean(a0) &&
      Boolean(a1) &&
      a0.shaderLocation === 0 &&
      a0.offset === 0 &&
      a0.format === 'float32x4' &&
      a1.shaderLocation === 1 &&
      a1.offset === 16 &&
      a1.format === 'float32x2'
    );
  }

  function resolveVertexLimits(options) {
    const opts = options && typeof options === 'object' ? options : {};
    const maxVertexBuffers =
      Number.isInteger(opts.maxVertexBuffers) && opts.maxVertexBuffers > 0
        ? opts.maxVertexBuffers
        : DEFAULT_MAX_VERTEX_BUFFERS;
    const maxVertexAttributes =
      Number.isInteger(opts.maxVertexAttributes) && opts.maxVertexAttributes > 0
        ? opts.maxVertexAttributes
        : DEFAULT_MAX_VERTEX_ATTRIBUTES;
    const rawMaxStride =
      Number.isInteger(opts.maxVertexBufferArrayStride) && opts.maxVertexBufferArrayStride >= 4
        ? opts.maxVertexBufferArrayStride
        : DEFAULT_MAX_VERTEX_BUFFER_STRIDE;
    const maxVertexBufferArrayStride = Math.max(4, alignDown(rawMaxStride, 4));
    return { maxVertexBuffers, maxVertexAttributes, maxVertexBufferArrayStride };
  }

  function nextFreeVertexLocation(usedLocations, maxVertexAttributes) {
    for (let loc = 0; loc < maxVertexAttributes; loc += 1) {
      if (!usedLocations.has(loc)) {
        return loc;
      }
    }
    return -1;
  }

  /**
   * Build a WebGPU-valid `GPUVertexBufferLayout[]` array for a given byte stride and/or
   * G34 fallback profile.
   *
   * - `fullscreen_triangle_cbv` / `default` -> `[]` (`@builtin(vertex_index)` shaders)
   * - `textured_quad_24b` or stride 24 -> 24B layout (`float32x4` @ 0 loc 0 + `float32x2` @ 16 loc 1)
   * - Other positive strides -> 4-byte aligned, clamped to `maxVertexBufferArrayStride`.
   *
   * @param {number} vertexStride
   * @param {{
   *   fallbackProfile?: string | null,
   *   profile?: string | null,
   *   stepMode?: 'vertex' | 'instance',
   *   maxVertexBufferArrayStride?: number,
   * }} [options]
   * @returns {Array<GPUVertexBufferLayout>}
   */
  function buildVertexBufferLayoutForStride(vertexStride, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const stepMode = opts.stepMode === 'instance' ? 'instance' : 'vertex';
    const rawProfile =
      typeof opts.fallbackProfile === 'string' && opts.fallbackProfile.trim().length > 0
        ? opts.fallbackProfile
        : typeof opts.profile === 'string' && opts.profile.trim().length > 0
          ? opts.profile
          : null;

    if (rawProfile !== null) {
      const profile = normalizeFallbackProfile(rawProfile);
      if (profile === 'fullscreen_triangle_cbv' || profile === 'default') {
        return [];
      }
      if (profile === 'textured_quad_24b') {
        return [cloneFrame1VertexLayout24B(stepMode)];
      }
    }

    if (typeof vertexStride !== 'number' || !Number.isFinite(vertexStride) || vertexStride <= 0) {
      return [];
    }

    const { maxVertexBufferArrayStride } = resolveVertexLimits(opts);
    const alignedStride = Math.min(maxVertexBufferArrayStride, alignUp4(Math.max(4, vertexStride)));

    if (alignedStride >= FRAME1_VERTEX_STRIDE_24B) {
      return [
        {
          arrayStride: alignedStride,
          stepMode,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x4' },
            { shaderLocation: 1, offset: 16, format: 'float32x2' },
          ],
        },
      ];
    }

    const format =
      alignedStride >= 16
        ? 'float32x4'
        : alignedStride >= 12
          ? 'float32x3'
          : alignedStride >= 8
            ? 'float32x2'
            : 'float32';

    return [
      {
        arrayStride: alignedStride,
        stepMode,
        attributes: [{ shaderLocation: 0, offset: 0, format }],
      },
    ];
  }

  function normalizeVertexBufferLayoutsInternal(buffers, options = {}) {
    const opts = options && typeof options === 'object' ? options : {};
    const { maxVertexBuffers, maxVertexAttributes, maxVertexBufferArrayStride } =
      resolveVertexLimits(opts);
    const reasons = [];
    const addReason = (reason) => {
      if (!reasons.includes(reason)) {
        reasons.push(reason);
      }
    };

    const rawProfile =
      typeof opts.fallbackProfile === 'string' && opts.fallbackProfile.trim().length > 0
        ? opts.fallbackProfile
        : typeof opts.profile === 'string' && opts.profile.trim().length > 0
          ? opts.profile
          : null;

    if (rawProfile !== null) {
      const profile = normalizeFallbackProfile(rawProfile);
      if (profile === 'fullscreen_triangle_cbv' || profile === 'default') {
        const alreadyEmpty = Array.isArray(buffers) && buffers.length === 0;
        if (!alreadyEmpty) {
          addReason('fallback-empty-vertex-buffers');
        }
        return { buffers: [], changed: !alreadyEmpty, reasons };
      }
      if (profile === 'textured_quad_24b') {
        const alreadyCanonical = isCanonical24BLayoutList(buffers);
        if (!alreadyCanonical) {
          addReason('fallback-24b-vertex-layout');
        }
        return {
          buffers: [cloneFrame1VertexLayout24B('vertex')],
          changed: !alreadyCanonical,
          reasons,
        };
      }
    }

    if (typeof buffers === 'number') {
      const built = buildVertexBufferLayoutForStride(buffers, opts);
      addReason('stride-to-vertex-layout');
      return { buffers: built, changed: true, reasons };
    }

    if (buffers === undefined || buffers === null) {
      if (
        typeof opts.vertexStride === 'number' &&
        Number.isFinite(opts.vertexStride) &&
        opts.vertexStride > 0
      ) {
        const built = buildVertexBufferLayoutForStride(opts.vertexStride, opts);
        addReason('vertex-stride-layout-synthesized');
        return { buffers: built, changed: true, reasons };
      }
      const changed = buffers === null;
      if (changed) {
        addReason('null-vertex-buffers-cleared');
      }
      return { buffers: [], changed, reasons };
    }

    if (!Array.isArray(buffers)) {
      addReason('invalid-vertex-buffers-cleared');
      return { buffers: [], changed: true, reasons };
    }

    if (buffers.length === 0) {
      if (
        typeof opts.vertexStride === 'number' &&
        Number.isFinite(opts.vertexStride) &&
        opts.vertexStride > 0
      ) {
        const built = buildVertexBufferLayoutForStride(opts.vertexStride, opts);
        addReason('vertex-stride-layout-synthesized');
        return { buffers: built, changed: true, reasons };
      }
      return { buffers: [], changed: false, reasons };
    }

    let changed = false;
    if (buffers.length > maxVertexBuffers) {
      changed = true;
      addReason('clamped-vertex-buffer-count');
    }

    const usedLocations = new Set();
    const normalizedBuffers = [];
    const slotCount = Math.min(buffers.length, maxVertexBuffers);

    for (let i = 0; i < slotCount; i += 1) {
      const buf = buffers[i];
      if (!buf || typeof buf !== 'object' || Array.isArray(buf)) {
        changed = true;
        addReason('dropped-invalid-vertex-buffer');
        continue;
      }

      let stepMode = 'vertex';
      if (buf.stepMode === 'instance') {
        stepMode = 'instance';
      } else if (buf.stepMode !== undefined && buf.stepMode !== 'vertex') {
        changed = true;
        addReason('normalized-step-mode');
      }

      const rawStride = buf.arrayStride;
      const isExplicitZero = rawStride === 0;
      const isValidPositiveStride =
        typeof rawStride === 'number' && Number.isFinite(rawStride) && rawStride > 0;

      let stride = 0;
      if (isValidPositiveStride) {
        const aligned = Math.min(maxVertexBufferArrayStride, alignUp4(rawStride));
        if (aligned !== rawStride) {
          changed = true;
          addReason('clamped-vertex-stride');
        }
        stride = aligned;
      } else if (isExplicitZero) {
        stride = 0;
      } else {
        changed = true;
        addReason('invalid-vertex-stride');
      }

      const rawAttrs = buf.attributes;
      const normalizedAttrs = [];

      if (Array.isArray(rawAttrs) && rawAttrs.length > 0) {
        for (let j = 0; j < rawAttrs.length; j += 1) {
          const attr = rawAttrs[j];
          if (!attr || typeof attr !== 'object' || Array.isArray(attr)) {
            changed = true;
            addReason('dropped-invalid-vertex-attribute');
            continue;
          }

          let format = attr.format;
          let fmtSize = getVertexFormatByteSize(format);
          if (fmtSize === 0) {
            format =
              stride >= 16 || stride === 0
                ? 'float32x4'
                : stride >= 12
                  ? 'float32x3'
                  : stride >= 8
                    ? 'float32x2'
                    : 'float32';
            fmtSize = getVertexFormatByteSize(format);
            changed = true;
            addReason('fallback-vertex-format');
          }

          const align = Math.min(4, fmtSize);
          let offset;
          if (typeof attr.offset !== 'number' || !Number.isFinite(attr.offset) || attr.offset < 0) {
            offset = 0;
            changed = true;
            addReason('clamped-vertex-offset');
          } else {
            const floored = Math.floor(attr.offset);
            offset = alignDown(floored, align);
            if (offset !== attr.offset) {
              changed = true;
              addReason('aligned-vertex-offset');
            }
          }

          if (isValidPositiveStride) {
            if (stride < fmtSize) {
              stride = Math.min(maxVertexBufferArrayStride, alignUp4(fmtSize));
              offset = 0;
              changed = true;
              addReason('expanded-vertex-stride');
            } else if (offset + fmtSize > stride) {
              offset = alignDown(stride - fmtSize, align);
              changed = true;
              addReason('clamped-vertex-offset');
            }
          } else if (isExplicitZero) {
            if (offset + fmtSize > maxVertexBufferArrayStride) {
              offset = alignDown(maxVertexBufferArrayStride - fmtSize, align);
              changed = true;
              addReason('clamped-vertex-offset');
            }
          } else {
            if (offset + fmtSize > maxVertexBufferArrayStride) {
              offset = alignDown(maxVertexBufferArrayStride - fmtSize, align);
              changed = true;
              addReason('clamped-vertex-offset');
            }
            const required = Math.min(maxVertexBufferArrayStride, alignUp4(offset + fmtSize));
            if (required > stride) {
              stride = required;
            }
          }

          let loc = attr.shaderLocation;
          if (
            !Number.isInteger(loc) ||
            loc < 0 ||
            loc >= maxVertexAttributes ||
            usedLocations.has(loc)
          ) {
            const nextLoc = nextFreeVertexLocation(usedLocations, maxVertexAttributes);
            if (nextLoc < 0) {
              changed = true;
              addReason('clamped-vertex-attribute-count');
              continue;
            }
            loc = nextLoc;
            changed = true;
            addReason('reassigned-vertex-location');
          }

          usedLocations.add(loc);
          normalizedAttrs.push({
            shaderLocation: loc,
            offset,
            format,
          });
        }
      } else {
        if (!Array.isArray(rawAttrs)) {
          changed = true;
        }
        if (stride > 0) {
          const synthesized = buildVertexBufferLayoutForStride(stride, {
            stepMode,
            maxVertexBufferArrayStride,
          });
          const templateAttrs = synthesized.length > 0 ? synthesized[0].attributes : [];
          for (let k = 0; k < templateAttrs.length; k += 1) {
            const tpl = templateAttrs[k];
            const loc = usedLocations.has(tpl.shaderLocation)
              ? nextFreeVertexLocation(usedLocations, maxVertexAttributes)
              : tpl.shaderLocation;
            if (loc < 0) {
              break;
            }
            usedLocations.add(loc);
            normalizedAttrs.push({
              shaderLocation: loc,
              offset: tpl.offset,
              format: tpl.format,
            });
          }
          changed = true;
          addReason(
            stride === FRAME1_VERTEX_STRIDE_24B
              ? 'synthesized-24b-attributes'
              : 'synthesized-vertex-attributes'
          );
        } else if (!isExplicitZero) {
          changed = true;
          addReason('dropped-empty-invalid-vertex-buffer');
          continue;
        }
      }

      normalizedBuffers.push({
        arrayStride: stride,
        stepMode,
        attributes: normalizedAttrs,
      });
    }

    return {
      buffers: normalizedBuffers,
      changed,
      reasons,
    };
  }

  /**
   * Normalize WebGPU vertex buffer layouts for real Frame-1 Xbox draws (including 24-byte-stride
   * `DrawInstanced(3, 1)` buffers) and G34 fallback shader profiles (`fullscreen_triangle_cbv`,
   * `textured_quad_24b`, `default`). Clamps invalid strides/offsets and deduplicates locations.
   *
   * @param {Array<GPUVertexBufferLayout> | number | null | undefined} buffers
   * @param {{
   *   fallbackProfile?: string | null,
   *   profile?: string | null,
   *   vertexStride?: number,
   *   maxVertexBuffers?: number,
   *   maxVertexAttributes?: number,
   *   maxVertexBufferArrayStride?: number,
   * }} [options]
   * @returns {Array<GPUVertexBufferLayout>}
   */
  function normalizeVertexBufferLayouts(buffers, options = {}) {
    return normalizeVertexBufferLayoutsInternal(buffers, options).buffers;
  }

  /**
   * Build a fresh `GPUBindGroupLayoutDescriptor` for group(0) matching a G34 fallback profile:
   * - `fullscreen_triangle_cbv`: `binding(0)` uniform CBV (vertex | fragment, minBindingSize 256)
   * - `textured_quad_24b`: `binding(0)` 2D float texture + `binding(1)` filtering sampler (fragment)
   * - `default`: empty `entries: []`
   *
   * @param {string | null | undefined} profile
   * @returns {GPUBindGroupLayoutDescriptor}
   */
  function buildFallbackBindGroupLayoutDescriptor(profile) {
    const canonical = normalizeFallbackProfile(profile);
    const template = FALLBACK_BIND_GROUP_ENTRIES[canonical] || FALLBACK_BIND_GROUP_ENTRIES.default;
    const entries = template.map((entry) => {
      const copy = {
        binding: entry.binding,
        visibility: entry.visibility,
      };
      if (entry.buffer) copy.buffer = { ...entry.buffer };
      if (entry.texture) copy.texture = { ...entry.texture };
      if (entry.sampler) copy.sampler = { ...entry.sampler };
      return copy;
    });
    return {
      label: `fallback-bgl:${canonical}:group0`,
      entries,
    };
  }

  /**
   * Normalize a `GPURenderPipelineDescriptor` for safe WebGPU validation without mutating the
   * caller's original descriptor. Already-valid non-fallback descriptors pass through untouched.
   *
   * @param {GPURenderPipelineDescriptor} descriptor
   * @param {{
   *   fallbackProfile?: string | null,
   *   profile?: string | null,
   *   isFallback?: boolean,
   *   vertexStride?: number,
   *   fallbackLayout?: 'auto' | GPUPipelineLayout,
   *   defaultColorFormat?: string,
   *   forceFallbackEntryPoints?: boolean,
   *   maxVertexBuffers?: number,
   *   maxVertexAttributes?: number,
   *   maxVertexBufferArrayStride?: number,
   * }} [options]
   * @returns {{
   *   descriptor: GPURenderPipelineDescriptor,
   *   normalized: boolean,
   *   vertexLayoutNormalized: boolean,
   *   layoutNormalized: boolean,
   *   fallbackProfile: 'default' | 'fullscreen_triangle_cbv' | 'textured_quad_24b' | null,
   *   reasons: Array<string>,
   * }}
   */
  function normalizeRenderPipelineDescriptor(descriptor, options = {}) {
    if (!descriptor || typeof descriptor !== 'object' || Array.isArray(descriptor)) {
      throw new TypeError('normalizeRenderPipelineDescriptor requires a descriptor object');
    }

    const opts = options && typeof options === 'object' ? options : {};
    const fallbackProfile = detectFallbackProfile(descriptor, opts);
    const defaultColorFormat =
      typeof opts.defaultColorFormat === 'string' && opts.defaultColorFormat.length > 0
        ? opts.defaultColorFormat
        : DEFAULT_COLOR_FORMAT;

    const reasons = [];
    const addReason = (r) => {
      if (!reasons.includes(r)) {
        reasons.push(r);
      }
    };

    let layoutNormalized = false;
    let vertexLayoutNormalized = false;
    let otherNormalized = false;

    const hasHelperMeta =
      Object.prototype.hasOwnProperty.call(descriptor, 'fallbackProfile') ||
      Object.prototype.hasOwnProperty.call(descriptor, 'vertexStride');

    // 1. Pipeline layout normalization
    const rawLayout = descriptor.layout;
    const hasValidRawLayout =
      rawLayout === 'auto' || (rawLayout !== null && typeof rawLayout === 'object');
    let nextLayout = rawLayout;

    if (fallbackProfile !== null) {
      if (opts.fallbackLayout !== undefined && opts.fallbackLayout !== null) {
        if (rawLayout !== opts.fallbackLayout) {
          nextLayout = opts.fallbackLayout;
          layoutNormalized = true;
          addReason('fallback-layout-supplied');
        }
      } else if (!hasValidRawLayout) {
        nextLayout = 'auto';
        layoutNormalized = true;
        addReason('missing-layout');
      } else if (
        rawLayout !== 'auto' &&
        fallbackPipelineLayoutProfiles.get(rawLayout) !== fallbackProfile
      ) {
        nextLayout = 'auto';
        layoutNormalized = true;
        addReason('fallback-layout-auto');
      }
    } else if (!hasValidRawLayout) {
      nextLayout = 'auto';
      layoutNormalized = true;
      addReason('missing-layout');
    }

    // 2. Vertex stage & vertex-buffer layout normalization
    let nextVertex = descriptor.vertex;
    if (descriptor.vertex && typeof descriptor.vertex === 'object') {
      const effectiveStride =
        typeof opts.vertexStride === 'number'
          ? opts.vertexStride
          : typeof descriptor.vertexStride === 'number'
            ? descriptor.vertexStride
            : undefined;

      const shouldInspectBuffers =
        fallbackProfile !== null ||
        descriptor.vertex.buffers !== undefined ||
        (typeof effectiveStride === 'number' &&
          Number.isFinite(effectiveStride) &&
          effectiveStride > 0);

      let nextBuffers = descriptor.vertex.buffers;
      if (shouldInspectBuffers) {
        const vblResult = normalizeVertexBufferLayoutsInternal(descriptor.vertex.buffers, {
          fallbackProfile,
          vertexStride: effectiveStride,
          maxVertexBuffers: opts.maxVertexBuffers,
          maxVertexAttributes: opts.maxVertexAttributes,
          maxVertexBufferArrayStride: opts.maxVertexBufferArrayStride,
        });
        if (vblResult.changed) {
          nextBuffers = vblResult.buffers;
          vertexLayoutNormalized = true;
          for (let i = 0; i < vblResult.reasons.length; i += 1) {
            addReason(vblResult.reasons[i]);
          }
        }
      }

      let nextVsEntryPoint = descriptor.vertex.entryPoint;
      if (
        fallbackProfile !== null &&
        (typeof nextVsEntryPoint !== 'string' ||
          nextVsEntryPoint.length === 0 ||
          opts.forceFallbackEntryPoints === true) &&
        nextVsEntryPoint !== 'vs_main'
      ) {
        nextVsEntryPoint = 'vs_main';
        otherNormalized = true;
        addReason('fallback-vertex-entrypoint');
      }

      if (vertexLayoutNormalized || nextVsEntryPoint !== descriptor.vertex.entryPoint) {
        nextVertex = {
          ...descriptor.vertex,
          entryPoint: nextVsEntryPoint,
        };
        if (shouldInspectBuffers && vertexLayoutNormalized) {
          nextVertex.buffers = nextBuffers;
        }
      }
    }

    // 3. Fragment stage normalization
    let nextFragment = descriptor.fragment;
    if (descriptor.fragment && typeof descriptor.fragment === 'object') {
      let fragChanged = false;
      let nextFsEntryPoint = descriptor.fragment.entryPoint;
      if (
        fallbackProfile !== null &&
        (typeof nextFsEntryPoint !== 'string' ||
          nextFsEntryPoint.length === 0 ||
          opts.forceFallbackEntryPoints === true) &&
        nextFsEntryPoint !== 'fs_main'
      ) {
        nextFsEntryPoint = 'fs_main';
        fragChanged = true;
        otherNormalized = true;
        addReason('fallback-fragment-entrypoint');
      }

      let nextTargets = descriptor.fragment.targets;
      const targetsResult = normalizeColorTargetsInternal(nextTargets, defaultColorFormat);
      if (targetsResult.changed) {
        nextTargets = targetsResult.targets;
        fragChanged = true;
        otherNormalized = true;
        for (let i = 0; i < targetsResult.reasons.length; i += 1) {
          addReason(targetsResult.reasons[i]);
        }
      }

      if (fragChanged) {
        nextFragment = {
          ...descriptor.fragment,
          entryPoint: nextFsEntryPoint,
          targets: nextTargets,
        };
      }
    }

    // 4. Primitive topology & depthStencil null normalization
    let nextPrimitive = descriptor.primitive;
    if (
      descriptor.primitive &&
      typeof descriptor.primitive === 'object' &&
      descriptor.primitive.topology !== undefined &&
      !VALID_PRIMITIVE_TOPOLOGIES.has(descriptor.primitive.topology)
    ) {
      nextPrimitive = {
        ...descriptor.primitive,
        topology: 'triangle-list',
      };
      otherNormalized = true;
      addReason('clamped-primitive-topology');
    }

    let stripDepthStencil = false;
    if (descriptor.depthStencil === null) {
      stripDepthStencil = true;
      otherNormalized = true;
      addReason('null-depth-stencil-cleared');
    }

    const normalized = layoutNormalized || vertexLayoutNormalized || otherNormalized;
    if (!normalized && !hasHelperMeta) {
      return {
        descriptor,
        normalized: false,
        vertexLayoutNormalized: false,
        layoutNormalized: false,
        fallbackProfile,
        reasons,
      };
    }

    const nextDescriptor = { ...descriptor };
    if (hasHelperMeta) {
      delete nextDescriptor.fallbackProfile;
      delete nextDescriptor.vertexStride;
    }
    if (stripDepthStencil) {
      delete nextDescriptor.depthStencil;
    }
    nextDescriptor.layout = nextLayout;
    if (nextVertex !== undefined) {
      nextDescriptor.vertex = nextVertex;
    }
    if (nextFragment !== undefined) {
      nextDescriptor.fragment = nextFragment;
    }
    if (nextPrimitive !== undefined) {
      nextDescriptor.primitive = nextPrimitive;
    }

    return {
      descriptor: nextDescriptor,
      normalized,
      vertexLayoutNormalized,
      layoutNormalized,
      fallbackProfile,
      reasons,
    };
  }

  function normalizeComputePipelineDescriptor(descriptor) {
    if (!descriptor || typeof descriptor !== 'object' || Array.isArray(descriptor)) {
      throw new TypeError('normalizeComputePipelineDescriptor requires a descriptor object');
    }
    const rawLayout = descriptor.layout;
    const hasValidRawLayout =
      rawLayout === 'auto' || (rawLayout !== null && typeof rawLayout === 'object');
    if (hasValidRawLayout) {
      return {
        descriptor,
        normalized: false,
        layoutNormalized: false,
      };
    }
    return {
      descriptor: {
        ...descriptor,
        layout: 'auto',
      },
      normalized: true,
      layoutNormalized: true,
    };
  }

  function classifyPipelineErrorType(err) {
    if (err && typeof err === 'object') {
      if (typeof GPUOutOfMemoryError === 'function' && err instanceof GPUOutOfMemoryError) {
        return 'out-of-memory';
      }
      if (typeof GPUValidationError === 'function' && err instanceof GPUValidationError) {
        return 'validation';
      }
      if (typeof GPUInternalError === 'function' && err instanceof GPUInternalError) {
        return 'internal';
      }
      if (err instanceof TypeError) {
        return 'type';
      }
      if (err instanceof RangeError) {
        return 'range';
      }
      const candidates = [
        err.type,
        err.name,
        err.constructor && err.constructor.name,
      ];
      for (let i = 0; i < candidates.length; i += 1) {
        const raw = candidates[i];
        if (typeof raw !== 'string') continue;
        const lower = raw.trim().toLowerCase();
        if (!lower) continue;
        if (
          lower.includes('outofmemory') ||
          lower.includes('out-of-memory') ||
          lower.includes('out_of_memory') ||
          lower === 'oom'
        ) {
          return 'out-of-memory';
        }
        if (lower.includes('validation')) {
          return 'validation';
        }
        if (lower.includes('internal')) {
          return 'internal';
        }
        if (lower === 'typeerror') {
          return 'type';
        }
        if (lower === 'rangeerror') {
          return 'range';
        }
      }
    }
    return 'unknown';
  }

  /**
   * Sanitize a pipeline creation error message so telemetry never leaks host filesystem paths,
   * embedded binary/hex/base64 dumps, control characters, or unbounded strings.
   *
   * @param {unknown} rawMessage
   * @param {number} [maxLength]
   * @returns {string}
   */
  function sanitizePipelineErrorMessage(rawMessage, maxLength = MAX_PIPELINE_ERROR_MESSAGE_LENGTH) {
    let msg;
    if (typeof rawMessage === 'string') {
      msg = rawMessage;
    } else if (rawMessage === null || rawMessage === undefined) {
      return '';
    } else if (
      typeof rawMessage === 'object' &&
      typeof rawMessage.message === 'string'
    ) {
      msg = rawMessage.message;
    } else {
      msg = String(rawMessage);
    }

    // Normalize line breaks/tabs to spaces and strip control characters.
    msg = msg.replace(/[\r\n\t]+/g, ' ').replace(/[\x00-\x1F\x7F]+/g, '');

    // Strip file:// URLs and host filesystem paths (Windows drive, UNC, and Unix paths).
    msg = msg.replace(/\bfile:\/\/[^\s"'`<>)\]]+/gi, '<path>');
    msg = msg.replace(/\\\\[^\s\\"'`<>)\]]+\\[^\s"'`<>)\]]+/g, '<path>');
    msg = msg.replace(/(["'`])[A-Za-z]:[\\\/][^"'`\r\n]+\1/g, '$1<path>$1');
    msg = msg.replace(/\b[A-Za-z]:[\\\/][^\s"'`<>)\],;]+/g, '<path>');
    msg = msg.replace(
      /(["'`])\/(?:Users|home|var|tmp|private|etc|opt|usr|Volumes|mnt|root|Library|Applications|System)\/[^"'`\r\n]+\1/gi,
      '$1<path>$1'
    );
    msg = msg.replace(
      /(^|[\s"'`(\[=,:])\/(?:[A-Za-z0-9._-]+\/)+[A-Za-z0-9._~+-]+/g,
      '$1<path>'
    );

    // Strip long embedded buffers (data URIs, byte arrays, hex dumps, base64 blobs).
    msg = msg.replace(/\bdata:[^\s;,'"`<>]+;base64,[A-Za-z0-9+/=]{16,}/gi, '<buffer>');
    msg = msg.replace(
      /(?:\b(?:0x)?[0-9a-fA-F]{1,4}\s*,\s*){16,}(?:0x)?[0-9a-fA-F]{1,4}\b/g,
      '<buffer>'
    );
    msg = msg.replace(/(?:\b[0-9a-fA-F]{2}\s+){16,}[0-9a-fA-F]{2}\b/g, '<buffer>');
    msg = msg.replace(/\b(?:0x)?[0-9a-fA-F]{48,}\b/g, '<buffer>');
    msg = msg.replace(
      /(^|[\s"'`(\[=,:])[A-Za-z0-9+/]{64,}={0,2}(?=$|[\s"'`)\]=,.;:])/g,
      '$1<buffer>'
    );

    msg = msg.replace(/\s{2,}/g, ' ').trim();

    const limit =
      Number.isInteger(maxLength) && maxLength >= 4
        ? maxLength
        : MAX_PIPELINE_ERROR_MESSAGE_LENGTH;
    if (msg.length > limit) {
      msg = `${msg.slice(0, limit - 3)}...`;
    }
    return msg;
  }

  function validatePositiveNumber(value, name) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${name} must be a finite positive number, got: ${value}`);
    }
    return value;
  }

  function validateNonNegativeInt(value, name) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative integer, got: ${value}`);
    }
    return value;
  }

  /**
   * Normalize an optional pipeline priority hint (`'critical'`, `'high'`, `'normal'`, or `'low'`).
   * Unknown or omitted values resolve safely to `'normal'`.
   *
   * @param {string | null | undefined} priority
   * @returns {'critical' | 'high' | 'normal' | 'low'}
   */
  function normalizePipelinePriority(priority) {
    if (typeof priority !== 'string') {
      return PIPELINE_PRIORITY_TIERS.NORMAL;
    }
    const key = priority.trim().toLowerCase();
    if (key.length === 0) {
      return PIPELINE_PRIORITY_TIERS.NORMAL;
    }
    if (Object.prototype.hasOwnProperty.call(PIPELINE_PRIORITY_ALIASES, key)) {
      return PIPELINE_PRIORITY_ALIASES[key];
    }
    return PIPELINE_PRIORITY_TIERS.NORMAL;
  }

  function extractPriorityOption(options) {
    if (!options || typeof options !== 'object') {
      return null;
    }
    if (options.priority !== undefined && options.priority !== null) {
      return normalizePipelinePriority(options.priority);
    }
    if (options.priorityTier !== undefined && options.priorityTier !== null) {
      return normalizePipelinePriority(options.priorityTier);
    }
    if (options.drawn === true || options.demand === true) {
      return PIPELINE_PRIORITY_TIERS.CRITICAL;
    }
    if (options.warmup === true || options.prewarm === true) {
      return PIPELINE_PRIORITY_TIERS.LOW;
    }
    return null;
  }

  function hasRenderNormalizationOptions(options) {
    if (!options || typeof options !== 'object') {
      return false;
    }
    for (const k in options) {
      if (!Object.prototype.hasOwnProperty.call(options, k)) continue;
      if (
        k === 'priority' ||
        k === 'priorityTier' ||
        k === 'drawn' ||
        k === 'demand' ||
        k === 'warmup' ||
        k === 'prewarm'
      ) {
        continue;
      }
      const v = options[k];
      if (v !== undefined && typeof v !== 'function') {
        return true;
      }
    }
    return false;
  }

  function percentileFromSorted(sorted, count, fraction) {
    if (count <= 0) {
      return 0;
    }
    const idx = Math.min(count - 1, Math.max(0, Math.ceil(fraction * count) - 1));
    return sorted[idx];
  }

  function isDescriptorWrapper(item) {
    return (
      Boolean(item) &&
      typeof item === 'object' &&
      !Array.isArray(item) &&
      Object.prototype.hasOwnProperty.call(item, 'descriptor') &&
      !Object.prototype.hasOwnProperty.call(item, 'vertex') &&
      !Object.prototype.hasOwnProperty.call(item, 'compute')
    );
  }

  /**
   * LRU-bounded WebGPU RenderPipeline, ComputePipeline, and BindGroupLayout cache
   * with async compilation in-flight promise deduplication, batch prewarm, fallback-safe
   * Frame-1 vertex/CBV layout normalization, structured error tracking, and telemetry.
   */
  class WebGpuPipelineCache {
    /**
     * @param {{
     *   device: GPUDevice,
     *   maxRenderPipelines?: number,
     *   maxComputePipelines?: number,
     *   maxBindGroupLayouts?: number,
     *   maxErrorHistory?: number,
     *   quarantineFailedKeys?: boolean,
     *   maxQuarantinedKeys?: number,
     *   defaultColorFormat?: string,
     *   onPipelineError?: (record: object) => void,
     *   now?: () => number,
     *   nowFn?: () => number,
     * }} options
     */
    constructor(options = {}) {
      if (!options || typeof options !== 'object') {
        throw new TypeError('WebGpuPipelineCache requires an options object with a WebGPU device');
      }
      if (!options.device || typeof options.device !== 'object') {
        throw new TypeError('WebGpuPipelineCache requires a valid GPUDevice in options.device');
      }

      this._device = options.device;
      this._maxRenderPipelines = validatePositiveInt(
        options.maxRenderPipelines !== undefined
          ? options.maxRenderPipelines
          : DEFAULT_MAX_RENDER_PIPELINES,
        'maxRenderPipelines'
      );
      this._maxComputePipelines = validatePositiveInt(
        options.maxComputePipelines !== undefined
          ? options.maxComputePipelines
          : DEFAULT_MAX_COMPUTE_PIPELINES,
        'maxComputePipelines'
      );
      this._maxBindGroupLayouts = validatePositiveInt(
        options.maxBindGroupLayouts !== undefined
          ? options.maxBindGroupLayouts
          : DEFAULT_MAX_BIND_GROUP_LAYOUTS,
        'maxBindGroupLayouts'
      );
      this._maxEntries =
        options.maxEntries !== undefined && options.maxEntries !== null
          ? validatePositiveInt(options.maxEntries, 'maxEntries')
          : null;
      this._maxFastLookupEntries = validatePositiveInt(
        options.maxFastLookupEntries !== undefined
          ? options.maxFastLookupEntries
          : DEFAULT_MAX_FAST_LOOKUP_ENTRIES,
        'maxFastLookupEntries'
      );
      this._compileBudgetMs =
        options.compileBudgetMs !== undefined
          ? validatePositiveNumber(options.compileBudgetMs, 'compileBudgetMs')
          : options.slowCompileBudgetMs !== undefined
            ? validatePositiveNumber(options.slowCompileBudgetMs, 'slowCompileBudgetMs')
            : DEFAULT_COMPILE_BUDGET_MS;
      this._compileLatencyWindowSize =
        options.compileLatencyWindowSize !== undefined
          ? validatePositiveInt(options.compileLatencyWindowSize, 'compileLatencyWindowSize')
          : options.latencyWindowSize !== undefined
            ? validatePositiveInt(options.latencyWindowSize, 'latencyWindowSize')
            : DEFAULT_COMPILE_LATENCY_WINDOW_SIZE;
      this._maxErrorHistory = validatePositiveInt(
        options.maxErrorHistory !== undefined
          ? options.maxErrorHistory
          : DEFAULT_MAX_ERROR_HISTORY,
        'maxErrorHistory'
      );

      if (
        options.quarantineFailedKeys !== undefined &&
        typeof options.quarantineFailedKeys !== 'boolean'
      ) {
        throw new TypeError('options.quarantineFailedKeys must be a boolean when provided');
      }
      if (
        options.enableNegativeCache !== undefined &&
        typeof options.enableNegativeCache !== 'boolean'
      ) {
        throw new TypeError('options.enableNegativeCache must be a boolean when provided');
      }
      this._quarantineFailedKeys =
        options.quarantineFailedKeys !== undefined
          ? options.quarantineFailedKeys
          : options.enableNegativeCache !== undefined
            ? options.enableNegativeCache
            : true;

      if (options.maxQuarantinedKeys !== undefined) {
        this._maxQuarantinedKeys = validatePositiveInt(
          options.maxQuarantinedKeys,
          'maxQuarantinedKeys'
        );
      } else if (options.maxNegativeCacheEntries !== undefined) {
        this._maxQuarantinedKeys = validatePositiveInt(
          options.maxNegativeCacheEntries,
          'maxNegativeCacheEntries'
        );
      } else {
        this._maxQuarantinedKeys = DEFAULT_MAX_QUARANTINED_KEYS;
      }

      if (options.now !== undefined && typeof options.now !== 'function') {
        throw new TypeError('options.now must be a function when provided');
      }
      if (options.nowFn !== undefined && typeof options.nowFn !== 'function') {
        throw new TypeError('options.nowFn must be a function when provided');
      }
      this._now = options.now || options.nowFn || defaultNow;

      if (
        options.onPipelineError !== undefined &&
        options.onPipelineError !== null &&
        typeof options.onPipelineError !== 'function'
      ) {
        throw new TypeError('options.onPipelineError must be a function when provided');
      }
      this._onPipelineError =
        typeof options.onPipelineError === 'function' ? options.onPipelineError : null;

      this._defaultColorFormat =
        typeof options.defaultColorFormat === 'string' && options.defaultColorFormat.length > 0
          ? options.defaultColorFormat
          : DEFAULT_COLOR_FORMAT;

      this._renderPipelines = new Map();
      this._computePipelines = new Map();
      this._bindGroupLayouts = new Map();
      this._fastRenderLookup = new Map();
      this._fastComputeLookup = new Map();
      this._fastRenderIntHash = new Map();
      this._fastComputeIntHash = new Map();
      this._renderDescriptorMemo = new WeakMap();
      this._computeDescriptorMemo = new WeakMap();
      this._fallbackPipelineLayouts = new Map();
      this._defaultSamplers = new Map();
      this._inFlightRender = new Map();
      this._inFlightCompute = new Map();
      this._inFlightPriorities = new Map();
      this._quarantinedKeys = new Map();
      this._quarantineHitsByReason = createEmptyQuarantineReasonCounts();
      this._quarantineFailuresByReason = createEmptyQuarantineReasonCounts();
      this._asyncPipelineCache =
        options.asyncPipelineCache && typeof options.asyncPipelineCache === 'object'
          ? options.asyncPipelineCache
          : null;
      this._activePrewarmQueues = new Set();
      this._drawnFirstPromotions = 0;
      this._generation = 0;
      this._shaderPackGeneration = 0;
      this._accessSeq = 0;
      this._hasNonDefaultPriority = false;

      // Pre-allocated rolling compile-latency window & reusable sort scratch buffer (zero-alloc percentiles).
      this._compileLatencyWindow = new Float64Array(this._compileLatencyWindowSize);
      this._compileLatencyScratch = new Float64Array(this._compileLatencyWindowSize);
      this._compileLatencyHead = 0;
      this._compileLatencyCount = 0;

      this._pipelineErrors = [];
      this._pipelineErrorSeq = 0;

      this._stats = {
        renderHits: 0,
        renderMisses: 0,
        computeHits: 0,
        computeMisses: 0,
        layoutHits: 0,
        layoutMisses: 0,
        inlineSamplerLayouts: 0,
        samplerHits: 0,
        samplerMisses: 0,
        fastLookupHits: 0,
        fastLookupMisses: 0,
        inFlightDedupes: 0,
        evictions: 0,
        reclaimedCount: 0,
        compileCount: 0,
        totalCompileMs: 0,
        maxCompileMs: 0,
        slowCompilesOverBudget: 0,
        quarantineHits: 0,
        negativeCacheEvictions: 0,
        fallbackNormalizedCount: 0,
        vertexLayoutNormalizedCount: 0,
        layoutNormalizedCount: 0,
        fallbackRecoveryCount: 0,
        pipelineErrorCount: 0,
        fallbackSamplersCreated: 0,
        fallbackSamplerBindingsSynthesized: 0,
        blendPipelineVariants: 0,
        srgbPipelineVariants: 0,
      };
    }

    /**
     * Associate a `GPUShaderModule` with its WGPK 64-bit hash or numeric ID
     * so descriptors referencing it hash deterministically.
     *
     * @param {object} shaderModule
     * @param {bigint | number | string} hashOrId
     * @returns {string}
     */
    registerShaderModule(shaderModule, hashOrId) {
      return registerShaderModuleId(shaderModule, hashOrId);
    }

    _deviceVertexLimits() {
      const limits = this._device && this._device.limits;
      if (!limits || typeof limits !== 'object') {
        return {};
      }
      const out = {};
      if (Number.isInteger(limits.maxVertexBuffers) && limits.maxVertexBuffers > 0) {
        out.maxVertexBuffers = limits.maxVertexBuffers;
      }
      if (Number.isInteger(limits.maxVertexAttributes) && limits.maxVertexAttributes > 0) {
        out.maxVertexAttributes = limits.maxVertexAttributes;
      }
      if (
        Number.isInteger(limits.maxVertexBufferArrayStride) &&
        limits.maxVertexBufferArrayStride >= 4
      ) {
        out.maxVertexBufferArrayStride = limits.maxVertexBufferArrayStride;
      }
      return out;
    }

    _recordPipelineError(kind, err, details = {}) {
      this._stats.pipelineErrorCount += 1;
      this._pipelineErrorSeq += 1;

      const rawKey = details.key;
      const sanitizedKey =
        rawKey !== undefined && rawKey !== null
          ? sanitizePipelineErrorMessage(String(rawKey), MAX_PIPELINE_ERROR_KEY_LENGTH)
          : null;

      const record = {
        seq: this._pipelineErrorSeq,
        kind,
        errorType: classifyPipelineErrorType(err),
        key: sanitizedKey,
        fallbackProfile: details.fallbackProfile || null,
        recovered: Boolean(details.recovered),
        message: sanitizePipelineErrorMessage(err),
        timestampMs: this._now(),
      };

      while (this._pipelineErrors.length >= this._maxErrorHistory) {
        this._pipelineErrors.shift();
      }
      this._pipelineErrors.push(record);

      if (this._onPipelineError) {
        try {
          this._onPipelineError({ ...record });
        } catch (_callbackErr) {
          // Never allow user error callback throws to disrupt pipeline cache error tracking.
        }
      }

      return record;
    }

    _recordNormalizationStats(norm) {
      if (!norm) return;
      if (norm.normalized) {
        this._stats.fallbackNormalizedCount += 1;
      }
      if (norm.vertexLayoutNormalized) {
        this._stats.vertexLayoutNormalizedCount += 1;
      }
      if (norm.layoutNormalized) {
        this._stats.layoutNormalizedCount += 1;
      }
    }

    _recordRenderVariantStats(descriptor) {
      if (descriptorHasBlendState(descriptor)) {
        this._stats.blendPipelineVariants += 1;
      }
      if (descriptorHasSrgbTarget(descriptor)) {
        this._stats.srgbPipelineVariants += 1;
      }
    }

    /**
     * Return a cached default fallback `GPUSampler` for the requested WebGPU sampler binding kind
     * (`'filtering'` linear clamp-to-edge, `'non-filtering'` nearest clamp-to-edge, or
     * `'comparison'` less-equal). Created lazily via `device.createSampler` and invalidated on
     * `clear()`, `reset()`, `onDeviceLost()`, and `setDevice()`.
     *
     * @param {'filtering' | 'non-filtering' | 'comparison' | string | null | undefined} [kind='filtering']
     * @returns {GPUSampler}
     */
    getOrCreateDefaultSampler(kind = 'filtering') {
      const canonical = normalizeDefaultSamplerKind(kind);
      const cached = this._defaultSamplers.get(canonical);
      if (cached !== undefined) {
        this._stats.samplerHits += 1;
        return cached;
      }
      if (typeof this._device.createSampler !== 'function') {
        throw new TypeError('device must provide a createSampler(descriptor) method');
      }

      this._stats.samplerMisses += 1;
      const descriptor = buildDefaultSamplerDescriptor(canonical);
      const key = `fallback-sampler:${canonical}`;
      let sampler;
      try {
        sampler = this._device.createSampler(descriptor);
      } catch (err) {
        this._recordPipelineError('sampler', err, { key });
        throw err;
      }

      if (!sampler || (typeof sampler !== 'object' && typeof sampler !== 'function')) {
        const emptyErr = new Error('createSampler returned empty sampler');
        this._recordPipelineError('sampler', emptyErr, { key });
        throw emptyErr;
      }

      setRegisteredObjectId(sampler, key);
      this._defaultSamplers.set(canonical, sampler);
      this._stats.fallbackSamplersCreated += 1;
      return sampler;
    }

    /**
     * Synthesize safe fallback `GPUSampler` bind-group entries for any sampler slot declared in
     * a bind-group layout or S6/S34 reflection group that does not already have an explicit static
     * or dynamic sampler bound in `existingEntries`. Does not mutate caller inputs.
     *
     * @param {Array<object> | { entries?: Array<object>, bindings?: Array<object> }} reflectionGroupOrEntries
     * @param {Iterable<{ binding: number, resource: unknown }>} [existingEntries=[]]
     * @returns {{
     *   entries: Array<{ binding: number, resource: unknown }>,
     *   synthesized: Array<{ binding: number, kind: 'filtering' | 'non-filtering' | 'comparison', resource: GPUSampler }>,
     *   synthesizedCount: number,
     * }}
     */
    synthesizeFallbackSamplerEntries(reflectionGroupOrEntries, existingEntries = []) {
      let layoutEntries;
      if (Array.isArray(reflectionGroupOrEntries)) {
        layoutEntries = reflectionGroupOrEntries;
      } else if (reflectionGroupOrEntries && typeof reflectionGroupOrEntries === 'object') {
        if (Array.isArray(reflectionGroupOrEntries.entries)) {
          layoutEntries = reflectionGroupOrEntries.entries;
        } else if (Array.isArray(reflectionGroupOrEntries.bindings)) {
          layoutEntries = reflectionGroupOrEntries.bindings;
        } else {
          throw new TypeError(
            'reflectionGroupOrEntries must be an array or an object with an "entries" or "bindings" array'
          );
        }
      } else {
        throw new TypeError('reflectionGroupOrEntries must be an array or object');
      }

      const byBinding = new Map();
      const boundSet = new Set();
      if (existingEntries !== null && existingEntries !== undefined) {
        for (const item of existingEntries) {
          if (!item || typeof item !== 'object') {
            continue;
          }
          if (!Number.isInteger(item.binding) || item.binding < 0) {
            continue;
          }
          if (item.resource !== undefined && item.resource !== null) {
            byBinding.set(item.binding, { ...item });
            boundSet.add(item.binding);
          }
        }
      }

      const synthesized = [];
      for (let i = 0; i < layoutEntries.length; i += 1) {
        const rawEntry = layoutEntries[i];
        if (!rawEntry || typeof rawEntry !== 'object' || isRtArenaReflectionBinding(rawEntry)) {
          continue;
        }
        const entry =
          rawEntry.layoutEntry && typeof rawEntry.layoutEntry === 'object'
            ? rawEntry.layoutEntry
            : rawEntry;
        const bindingIndex =
          Number.isInteger(entry.binding) && entry.binding >= 0 ? entry.binding : rawEntry.binding;
        if (
          !entry.sampler ||
          typeof entry.sampler !== 'object' ||
          !Number.isInteger(bindingIndex) ||
          bindingIndex < 0
        ) {
          continue;
        }
        if (boundSet.has(bindingIndex)) {
          continue;
        }
        const samplerKind = normalizeDefaultSamplerKind(entry.sampler.type);
        const sampler = this.getOrCreateDefaultSampler(samplerKind);
        byBinding.set(bindingIndex, { binding: bindingIndex, resource: sampler });
        boundSet.add(bindingIndex);
        synthesized.push({
          binding: bindingIndex,
          kind: samplerKind,
          resource: sampler,
        });
        this._stats.fallbackSamplerBindingsSynthesized += 1;
      }

      const entries = Array.from(byBinding.values()).sort((a, b) => a.binding - b.binding);
      return {
        entries,
        synthesized,
        synthesizedCount: synthesized.length,
      };
    }

    /**
     * Return shallow copies of sanitized pipeline creation errors retained in the bounded ring.
     *
     * @returns {Array<{
     *   seq: number,
     *   kind: string,
     *   errorType: string,
     *   key: string | null,
     *   fallbackProfile: string | null,
     *   recovered: boolean,
     *   message: string,
     *   timestampMs: number,
     * }>}
     */
    getPipelineErrors() {
      return this._pipelineErrors.map((entry) => ({ ...entry }));
    }

    /**
     * Clear the bounded pipeline creation error history ring while preserving cumulative error
     * counters and monotonic sequence numbers.
     */
    clearPipelineErrors() {
      this._pipelineErrors.length = 0;
    }

    /**
     * Return a cached group(0) `GPUBindGroupLayout` matching a G34 fallback profile
     * (`fullscreen_triangle_cbv`, `textured_quad_24b`, or `default`).
     *
     * @param {string | null | undefined} profile
     * @returns {GPUBindGroupLayout}
     */
    getOrCreateFallbackBindGroupLayout(profile) {
      const canonical = normalizeFallbackProfile(profile);
      const descriptor = buildFallbackBindGroupLayoutDescriptor(canonical);
      return this.getOrCreateBindGroupLayout(descriptor, `fallback-bgl:${canonical}`);
    }

    /**
     * Return a cached `GPUPipelineLayout` compatible with a G34 fallback profile, or `'auto'`
     * when `device.createPipelineLayout` is unavailable or fails validation.
     *
     * @param {string | null | undefined} profile
     * @returns {GPUPipelineLayout | 'auto'}
     */
    getOrCreateFallbackPipelineLayout(profile) {
      const canonical = normalizeFallbackProfile(profile);
      if (typeof this._device.createPipelineLayout !== 'function') {
        return 'auto';
      }

      const cached = this._fallbackPipelineLayouts.get(canonical);
      if (cached !== undefined) {
        return cached;
      }

      const bindGroupLayouts =
        canonical === 'default' ? [] : [this.getOrCreateFallbackBindGroupLayout(canonical)];

      try {
        const layout = this._device.createPipelineLayout({
          label: `fallback-pl:${canonical}`,
          bindGroupLayouts,
        });
        if (layout && (typeof layout === 'object' || typeof layout === 'function')) {
          setRegisteredObjectId(layout, `fallback-pl:${canonical}`);
          fallbackPipelineLayoutProfiles.set(layout, canonical);
          this._fallbackPipelineLayouts.set(canonical, layout);
          return layout;
        }
        return 'auto';
      } catch (err) {
        this._recordPipelineError('pipeline-layout', err, {
          key: `fallback-pl:${canonical}`,
          fallbackProfile: canonical,
          recovered: true,
        });
        return 'auto';
      }
    }

    /**
     * Normalize a render pipeline descriptor using this cache's device vertex limits,
     * default color format, and optional explicit fallback `GPUPipelineLayout`.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {object} [options]
     */
    normalizeRenderDescriptor(descriptor, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const detectedProfile = detectFallbackProfile(descriptor, opts);
      let fallbackLayout = opts.fallbackLayout;
      if (
        detectedProfile !== null &&
        opts.layoutMode === 'explicit' &&
        (fallbackLayout === undefined || fallbackLayout === null)
      ) {
        fallbackLayout = this.getOrCreateFallbackPipelineLayout(detectedProfile);
      }

      const mergedOpts = {
        ...this._deviceVertexLimits(),
        defaultColorFormat: this._defaultColorFormat,
        ...opts,
      };
      if (fallbackLayout !== undefined) {
        mergedOpts.fallbackLayout = fallbackLayout;
      }
      return normalizeRenderPipelineDescriptor(descriptor, mergedOpts);
    }

    _resolveKey(descriptor, keyHint) {
      if (keyHint !== undefined && keyHint !== null) {
        if (typeof keyHint === 'string') {
          if (keyHint.length === 0) {
            throw new RangeError('keyHint string must not be empty');
          }
          return keyHint;
        }
        if (typeof keyHint === 'bigint') {
          return keyHint >= 0n && keyHint <= U64_MASK
            ? formatHashHex64(keyHint)
            : `bi:${keyHint.toString(10)}`;
        }
        if (typeof keyHint === 'number') {
          if (!Number.isFinite(keyHint)) {
            throw new RangeError(`keyHint number must be finite, got: ${keyHint}`);
          }
          return `num:${keyHint}`;
        }
        return hashDescriptorKey(keyHint);
      }
      return hashDescriptorKey(descriptor);
    }

    _promoteEntryPriority(entry, priority) {
      if (!entry || priority === null || priority === undefined) {
        return;
      }
      const rank = PIPELINE_PRIORITY_RANKS[priority];
      if (rank !== undefined && rank > entry.priorityRank) {
        entry.priority = priority;
        entry.priorityRank = rank;
        if (rank !== PIPELINE_PRIORITY_RANKS.normal) {
          this._hasNonDefaultPriority = true;
        }
      }
    }

    _recordInFlightPriority(kind, key, priority) {
      if (priority === null || priority === undefined) {
        return;
      }
      const mapKey = `${kind}:${key}`;
      const existing = this._inFlightPriorities.get(mapKey);
      if (
        existing === undefined ||
        PIPELINE_PRIORITY_RANKS[priority] > PIPELINE_PRIORITY_RANKS[existing]
      ) {
        this._inFlightPriorities.set(mapKey, priority);
      }
    }

    _consumeInFlightPriority(kind, key, basePriority) {
      const mapKey = `${kind}:${key}`;
      const promoted = this._inFlightPriorities.get(mapKey);
      if (promoted === undefined) {
        return basePriority;
      }
      this._inFlightPriorities.delete(mapKey);
      if (basePriority === null || basePriority === undefined) {
        return promoted;
      }
      return PIPELINE_PRIORITY_RANKS[promoted] > PIPELINE_PRIORITY_RANKS[basePriority]
        ? promoted
        : basePriority;
    }

    _clearInFlightPriority(kind, key) {
      this._inFlightPriorities.delete(`${kind}:${key}`);
    }

    _totalCachedEntries() {
      return this._renderPipelines.size + this._computePipelines.size + this._bindGroupLayouts.size;
    }

    _evictOneFromMap(map, excludeKey) {
      if (map.size === 0) {
        return false;
      }
      let victimKey;
      let victimEntry;

      if (!this._hasNonDefaultPriority) {
        for (const [k, entry] of map.entries()) {
          if (k === excludeKey) {
            continue;
          }
          victimKey = k;
          victimEntry = entry;
          break;
        }
      } else {
        let bestRank = Infinity;
        for (const [k, entry] of map.entries()) {
          if (k === excludeKey) {
            continue;
          }
          const rank = entry.priorityRank;
          if (rank < bestRank) {
            bestRank = rank;
            victimKey = k;
            victimEntry = entry;
            if (bestRank === PIPELINE_PRIORITY_RANKS.low) {
              break;
            }
          }
        }
      }

      if (victimKey === undefined) {
        return false;
      }
      if (victimEntry !== undefined) {
        victimEntry.alive = false;
      }
      map.delete(victimKey);
      this._stats.evictions += 1;
      this._stats.reclaimedCount += 1;
      return true;
    }

    _evictOneAcrossAllMaps(excludeMap, excludeKey, maxAllowedRank) {
      const maps = [this._renderPipelines, this._computePipelines, this._bindGroupLayouts];
      let bestMap = null;
      let bestKey;
      let bestEntry = null;
      let bestRank = Infinity;
      let bestSeq = Infinity;

      for (let i = 0; i < maps.length; i += 1) {
        const map = maps[i];
        if (map.size === 0) {
          continue;
        }
        if (!this._hasNonDefaultPriority) {
          if (PIPELINE_PRIORITY_RANKS.normal > maxAllowedRank) {
            continue;
          }
          for (const [k, entry] of map.entries()) {
            if (map === excludeMap && k === excludeKey) {
              continue;
            }
            if (entry.accessSeq < bestSeq) {
              bestMap = map;
              bestKey = k;
              bestEntry = entry;
              bestRank = entry.priorityRank;
              bestSeq = entry.accessSeq;
            }
            break;
          }
          continue;
        }

        let mapBestRank = maxAllowedRank + 1;
        for (const [k, entry] of map.entries()) {
          if (map === excludeMap && k === excludeKey) {
            continue;
          }
          const rank = entry.priorityRank;
          if (rank < mapBestRank) {
            mapBestRank = rank;
            if (
              rank < bestRank ||
              (rank === bestRank && entry.accessSeq < bestSeq)
            ) {
              bestMap = map;
              bestKey = k;
              bestEntry = entry;
              bestRank = rank;
              bestSeq = entry.accessSeq;
            }
            if (rank === PIPELINE_PRIORITY_RANKS.low) {
              break;
            }
          }
        }
      }

      if (bestMap === null || bestKey === undefined) {
        return false;
      }
      if (bestEntry !== null) {
        bestEntry.alive = false;
      }
      bestMap.delete(bestKey);
      this._stats.evictions += 1;
      this._stats.reclaimedCount += 1;
      return true;
    }

    _enforceMaxEntries(excludeMap, excludeKey) {
      if (this._maxEntries === null) {
        return 0;
      }
      let removed = 0;
      while (this._totalCachedEntries() > this._maxEntries) {
        if (
          !this._evictOneAcrossAllMaps(
            excludeMap,
            excludeKey,
            PIPELINE_PRIORITY_RANKS.critical
          )
        ) {
          break;
        }
        removed += 1;
      }
      return removed;
    }

    _getLru(map, key, priority = null) {
      const entry = map.get(key);
      if (entry === undefined) {
        return undefined;
      }
      entry.lastAccessedAt = this._now();
      entry.accessSeq = ++this._accessSeq;
      if (priority !== null && priority !== undefined) {
        this._promoteEntryPriority(entry, priority);
      }
      // Refresh LRU recency by reinserting at the tail of the Map.
      map.delete(key);
      map.set(key, entry);
      return entry.value;
    }

    _setLru(map, key, value, compileMs, maxSize, priority = null) {
      const ts = this._now();
      const canonicalPriority =
        priority !== null && priority !== undefined
          ? normalizePipelinePriority(priority)
          : PIPELINE_PRIORITY_TIERS.NORMAL;
      const priorityRank = PIPELINE_PRIORITY_RANKS[canonicalPriority];
      if (priorityRank !== PIPELINE_PRIORITY_RANKS.normal) {
        this._hasNonDefaultPriority = true;
      }
      const prev = map.get(key);
      if (prev !== undefined) {
        prev.alive = false;
        map.delete(key);
      }
      const entry = {
        value,
        createdAt: ts,
        lastAccessedAt: ts,
        accessSeq: ++this._accessSeq,
        compileMs,
        priority: canonicalPriority,
        priorityRank,
        alive: true,
      };
      map.set(key, entry);
      while (map.size > maxSize) {
        if (!this._evictOneFromMap(map, key)) {
          break;
        }
      }
      if (this._maxEntries !== null) {
        this._enforceMaxEntries(map, key);
      }
      return entry;
    }

    _invalidateFastLookups() {
      this._fastRenderLookup.clear();
      this._fastComputeLookup.clear();
      this._fastRenderIntHash.clear();
      this._fastComputeIntHash.clear();
      this._renderDescriptorMemo = new WeakMap();
      this._computeDescriptorMemo = new WeakMap();
    }

    _isValidFastKey(fastKey) {
      if (typeof fastKey === 'string') {
        return fastKey.length > 0;
      }
      if (typeof fastKey === 'number') {
        return Number.isFinite(fastKey);
      }
      return typeof fastKey === 'bigint';
    }

    _touchBackingLru(lruMap, canonicalKey, entry, priority = null) {
      entry.lastAccessedAt = this._now();
      entry.accessSeq = ++this._accessSeq;
      if (priority !== null && priority !== undefined) {
        this._promoteEntryPriority(entry, priority);
      }
      lruMap.delete(canonicalKey);
      lruMap.set(canonicalKey, entry);
    }

    _setFastLookup(map, fastKey, canonicalKey, entry) {
      if (!entry || !entry.alive || !this._isValidFastKey(fastKey)) {
        return;
      }
      const normalizedKey = typeof fastKey === 'number' && fastKey === 0 ? 0 : fastKey;
      if (map.has(normalizedKey)) {
        map.delete(normalizedKey);
      }
      map.set(normalizedKey, {
        canonicalKey,
        entry,
        generation: this._generation,
      });
      while (map.size > this._maxFastLookupEntries) {
        const oldestKey = map.keys().next().value;
        map.delete(oldestKey);
      }
    }

    _getFastLookup(map, lruMap, fastKey, priority = null) {
      const normalizedKey = typeof fastKey === 'number' && fastKey === 0 ? 0 : fastKey;
      const rec = map.get(normalizedKey);
      if (rec === undefined) {
        return undefined;
      }
      if (rec.generation !== this._generation || !rec.entry.alive) {
        map.delete(normalizedKey);
        return undefined;
      }
      map.delete(normalizedKey);
      map.set(normalizedKey, rec);
      this._touchBackingLru(lruMap, rec.canonicalKey, rec.entry, priority);
      return rec.entry.value;
    }

    _setFastIntHash(map, intHash, canonicalKey, entry, snapshot) {
      if (intHash === null || !entry || !entry.alive || !snapshot) {
        return;
      }
      if (map.has(intHash)) {
        map.delete(intHash);
      }
      map.set(intHash, {
        canonicalKey,
        entry,
        generation: this._generation,
        regVer: shaderModuleRegistryVersion,
        snapshot,
      });
      while (map.size > this._maxFastLookupEntries) {
        const oldestKey = map.keys().next().value;
        map.delete(oldestKey);
      }
    }

    _tryFastRenderLookupByDescriptor(descriptor, priority = null) {
      if (!descriptor || typeof descriptor !== 'object') {
        return undefined;
      }
      const memo = this._renderDescriptorMemo.get(descriptor);
      if (memo !== undefined) {
        if (
          memo.generation === this._generation &&
          memo.entry.alive &&
          memo.regVer === shaderModuleRegistryVersion &&
          matchesRenderSnapshot(descriptor, memo.snapshot)
        ) {
          this._touchBackingLru(this._renderPipelines, memo.canonicalKey, memo.entry, priority);
          return memo.entry.value;
        }
        this._renderDescriptorMemo.delete(descriptor);
      }

      if (this._fastRenderIntHash.size === 0) {
        return undefined;
      }
      const intHash = computeFastRenderIntHash(descriptor);
      if (intHash === null) {
        return undefined;
      }
      const rec = this._fastRenderIntHash.get(intHash);
      if (rec === undefined) {
        return undefined;
      }
      if (
        rec.generation !== this._generation ||
        !rec.entry.alive ||
        rec.regVer !== shaderModuleRegistryVersion
      ) {
        this._fastRenderIntHash.delete(intHash);
        return undefined;
      }
      if (!matchesRenderSnapshot(descriptor, rec.snapshot)) {
        return undefined;
      }
      this._fastRenderIntHash.delete(intHash);
      this._fastRenderIntHash.set(intHash, rec);
      this._touchBackingLru(this._renderPipelines, rec.canonicalKey, rec.entry, priority);
      this._renderDescriptorMemo.set(descriptor, rec);
      return rec.entry.value;
    }

    _tryFastComputeLookupByDescriptor(descriptor, priority = null) {
      if (!descriptor || typeof descriptor !== 'object') {
        return undefined;
      }
      const memo = this._computeDescriptorMemo.get(descriptor);
      if (memo !== undefined) {
        if (
          memo.generation === this._generation &&
          memo.entry.alive &&
          memo.regVer === shaderModuleRegistryVersion &&
          matchesComputeSnapshot(descriptor, memo.snapshot)
        ) {
          this._touchBackingLru(this._computePipelines, memo.canonicalKey, memo.entry, priority);
          return memo.entry.value;
        }
        this._computeDescriptorMemo.delete(descriptor);
      }

      if (this._fastComputeIntHash.size === 0) {
        return undefined;
      }
      const intHash = computeFastComputeIntHash(descriptor);
      if (intHash === null) {
        return undefined;
      }
      const rec = this._fastComputeIntHash.get(intHash);
      if (rec === undefined) {
        return undefined;
      }
      if (
        rec.generation !== this._generation ||
        !rec.entry.alive ||
        rec.regVer !== shaderModuleRegistryVersion
      ) {
        this._fastComputeIntHash.delete(intHash);
        return undefined;
      }
      if (!matchesComputeSnapshot(descriptor, rec.snapshot)) {
        return undefined;
      }
      this._fastComputeIntHash.delete(intHash);
      this._fastComputeIntHash.set(intHash, rec);
      this._touchBackingLru(this._computePipelines, rec.canonicalKey, rec.entry, priority);
      this._computeDescriptorMemo.set(descriptor, rec);
      return rec.entry.value;
    }

    _populateFastRenderEntry(rawDescriptor, norm, canonicalKey, keyHint, opts, entry) {
      if (!entry || !entry.alive) {
        return;
      }
      this._setFastLookup(this._fastRenderLookup, canonicalKey, canonicalKey, entry);
      if (
        keyHint !== undefined &&
        keyHint !== null &&
        (typeof keyHint === 'string' || typeof keyHint === 'bigint' || typeof keyHint === 'number')
      ) {
        this._setFastLookup(this._fastRenderLookup, keyHint, canonicalKey, entry);
      }
      const hasCustomOpts = hasRenderNormalizationOptions(opts);
      if ((keyHint === undefined || keyHint === null) && !hasCustomOpts && rawDescriptor) {
        if (isFastRenderDescriptor(rawDescriptor)) {
          const snapshot = captureRenderSnapshot(rawDescriptor);
          if (snapshot !== null) {
            const memoRec = {
              canonicalKey,
              entry,
              generation: this._generation,
              regVer: shaderModuleRegistryVersion,
              snapshot,
            };
            this._renderDescriptorMemo.set(rawDescriptor, memoRec);
            if (!norm.normalized && norm.fallbackProfile === null) {
              const intHash = computeFastRenderIntHash(rawDescriptor);
              if (intHash !== null) {
                this._setFastIntHash(
                  this._fastRenderIntHash,
                  intHash,
                  canonicalKey,
                  entry,
                  snapshot
                );
              }
            }
          }
        }
      }
    }

    _populateFastComputeEntry(rawDescriptor, norm, canonicalKey, keyHint, entry) {
      if (!entry || !entry.alive) {
        return;
      }
      this._setFastLookup(this._fastComputeLookup, canonicalKey, canonicalKey, entry);
      if (
        keyHint !== undefined &&
        keyHint !== null &&
        (typeof keyHint === 'string' || typeof keyHint === 'bigint' || typeof keyHint === 'number')
      ) {
        this._setFastLookup(this._fastComputeLookup, keyHint, canonicalKey, entry);
      }
      if ((keyHint === undefined || keyHint === null) && rawDescriptor) {
        if (isFastComputeDescriptor(rawDescriptor)) {
          const snapshot = captureComputeSnapshot(rawDescriptor);
          if (snapshot !== null) {
            const memoRec = {
              canonicalKey,
              entry,
              generation: this._generation,
              regVer: shaderModuleRegistryVersion,
              snapshot,
            };
            this._computeDescriptorMemo.set(rawDescriptor, memoRec);
            if (!norm.normalized) {
              const intHash = computeFastComputeIntHash(rawDescriptor);
              if (intHash !== null) {
                this._setFastIntHash(
                  this._fastComputeIntHash,
                  intHash,
                  canonicalKey,
                  entry,
                  snapshot
                );
              }
            }
          }
        }
      }
    }

    _recordCompileTiming(compileMs) {
      const durationMs =
        typeof compileMs === 'number' && Number.isFinite(compileMs) && compileMs > 0
          ? compileMs
          : 0;
      this._stats.compileCount += 1;
      this._stats.totalCompileMs += durationMs;
      if (durationMs > this._stats.maxCompileMs) {
        this._stats.maxCompileMs = durationMs;
      }
      this._compileLatencyWindow[this._compileLatencyHead] = durationMs;
      this._compileLatencyHead = (this._compileLatencyHead + 1) % this._compileLatencyWindowSize;
      if (this._compileLatencyCount < this._compileLatencyWindowSize) {
        this._compileLatencyCount += 1;
      }
      if (durationMs > this._compileBudgetMs) {
        this._stats.slowCompilesOverBudget += 1;
      }
    }

    _computeCompileLatencyStats() {
      const slowCompilesOverBudget =
        typeof this._stats.slowCompilesOverBudget === 'number' &&
        Number.isFinite(this._stats.slowCompilesOverBudget) &&
        this._stats.slowCompilesOverBudget > 0
          ? this._stats.slowCompilesOverBudget
          : 0;
      const count = this._compileLatencyCount;
      if (count <= 0) {
        return {
          p50Ms: 0,
          p95Ms: 0,
          maxMs: 0,
          slowCompilesOverBudget,
          budgetMs: this._compileBudgetMs,
          sampleCount: 0,
          windowSize: this._compileLatencyWindowSize,
        };
      }
      const ring = this._compileLatencyWindow;
      const scratch = this._compileLatencyScratch;
      for (let i = 0; i < count; i += 1) {
        scratch[i] = ring[i];
      }
      const view = scratch.subarray(0, count);
      view.sort();
      const rawP50 = percentileFromSorted(view, count, 0.5);
      const rawP95 = percentileFromSorted(view, count, 0.95);
      const rawMax = view[count - 1];
      return {
        p50Ms: Number.isFinite(rawP50) && rawP50 > 0 ? rawP50 : 0,
        p95Ms: Number.isFinite(rawP95) && rawP95 > 0 ? rawP95 : 0,
        maxMs: Number.isFinite(rawMax) && rawMax > 0 ? rawMax : 0,
        slowCompilesOverBudget,
        budgetMs: this._compileBudgetMs,
        sampleCount: count,
        windowSize: this._compileLatencyWindowSize,
      };
    }

    /**
     * Return rolling compile-latency percentiles (`p50Ms`, `p95Ms`, `maxMs`) and
     * over-budget compile count (`slowCompilesOverBudget`) computed via a reusable
     * pre-allocated `Float64Array` scratch buffer.
     *
     * @returns {{
     *   p50Ms: number,
     *   p95Ms: number,
     *   maxMs: number,
     *   slowCompilesOverBudget: number,
     *   budgetMs: number,
     *   sampleCount: number,
     *   windowSize: number,
     * }}
     */
    getCompileLatencyStats() {
      return this._computeCompileLatencyStats();
    }

    _quarantineKey(key, kind, err) {
      const reasonBucket = classifyQuarantineReason(err);
      this._quarantineFailuresByReason[reasonBucket] =
        (this._quarantineFailuresByReason[reasonBucket] || 0) + 1;
      if (!this._quarantineFailedKeys || reasonBucket === PIPELINE_FAILURE_REASON.DEVICE_LOST) {
        return;
      }
      const ts = this._now();
      const sanitizedReason =
        sanitizePipelineErrorMessage(err) || `${kind} pipeline creation failed`;
      const errorType = classifyPipelineErrorType(err);
      const existing = this._quarantinedKeys.get(key);
      if (existing !== undefined) {
        existing.kind = kind;
        existing.reason = sanitizedReason;
        existing.reasonBucket = reasonBucket;
        existing.errorType = errorType;
        existing.failureCount += 1;
        existing.lastFailedAt = ts;
        // Refresh recency in bounded LRU/FIFO quarantine map.
        this._quarantinedKeys.delete(key);
        this._quarantinedKeys.set(key, existing);
        return;
      }

      this._quarantinedKeys.set(key, {
        key,
        kind,
        reason: sanitizedReason,
        reasonBucket,
        errorType,
        failureCount: 1,
        hitCount: 0,
        quarantinedAt: ts,
        lastFailedAt: ts,
        lastHitAt: null,
      });

      while (this._quarantinedKeys.size > this._maxQuarantinedKeys) {
        const oldestKey = this._quarantinedKeys.keys().next().value;
        this._quarantinedKeys.delete(oldestKey);
        this._stats.negativeCacheEvictions += 1;
      }
    }

    _isQuarantinedKey(key, kind) {
      if (!this._quarantineFailedKeys || this._quarantinedKeys.size === 0) {
        return false;
      }
      const entry = this._quarantinedKeys.get(key);
      if (entry === undefined) {
        return false;
      }
      if (kind && entry.kind && entry.kind !== kind) {
        return false;
      }
      return true;
    }

    _hitQuarantine(key, kind) {
      if (!this._quarantineFailedKeys || this._quarantinedKeys.size === 0) {
        return false;
      }
      const entry = this._quarantinedKeys.get(key);
      if (entry === undefined) {
        return false;
      }
      if (kind && entry.kind && entry.kind !== kind) {
        return false;
      }
      entry.hitCount += 1;
      entry.lastHitAt = this._now();
      // Refresh recency so actively-hit quarantined keys stay protected from eviction.
      this._quarantinedKeys.delete(key);
      this._quarantinedKeys.set(key, entry);
      this._stats.quarantineHits += 1;
      const bucket = entry.reasonBucket || PIPELINE_FAILURE_REASON.OTHER;
      this._quarantineHitsByReason[bucket] = (this._quarantineHitsByReason[bucket] || 0) + 1;
      return true;
    }

    _getQuarantinedByReason() {
      const counts = createEmptyQuarantineReasonCounts();
      for (const entry of this._quarantinedKeys.values()) {
        const bucket = entry.reasonBucket || PIPELINE_FAILURE_REASON.OTHER;
        counts[bucket] = (counts[bucket] || 0) + 1;
      }
      return counts;
    }

    /**
     * W379: Attach (or detach with null) a companion `AsyncPipelineCache` so drawn-first priority
     * promotions forward cleanly from `WebGpuPipelineCache` into the async compile queue.
     *
     * @param {object | null} asyncPipelineCache
     */
    attachAsyncPipelineCache(asyncPipelineCache) {
      this._asyncPipelineCache =
        asyncPipelineCache && typeof asyncPipelineCache === 'object' ? asyncPipelineCache : null;
    }

    _resolveOptionalKey(keyOrDescriptor, keyHint) {
      if (keyHint !== undefined && keyHint !== null) {
        try {
          return this._resolveKey(keyOrDescriptor, keyHint);
        } catch (_err) {
          return null;
        }
      }
      if (typeof keyOrDescriptor === 'string') {
        return keyOrDescriptor.length > 0 ? keyOrDescriptor : null;
      }
      if (typeof keyOrDescriptor === 'bigint' || typeof keyOrDescriptor === 'number') {
        try {
          return this._resolveKey(null, keyOrDescriptor);
        } catch (_err) {
          return null;
        }
      }
      if (keyOrDescriptor && typeof keyOrDescriptor === 'object' && !Array.isArray(keyOrDescriptor)) {
        try {
          return hashDescriptorKey(keyOrDescriptor);
        } catch (_err) {
          return null;
        }
      }
      return null;
    }

    _promoteKeyOnDraw(key, kind = null, rawHint = undefined) {
      let promoted = false;
      if (key !== null && key !== undefined) {
        if (kind === null || kind === 'render') {
          if (this._renderPipelines.has(key)) {
            this._getLru(this._renderPipelines, key, PIPELINE_PRIORITY_TIERS.CRITICAL);
          }
          if (this._inFlightRender.has(key)) {
            this._recordInFlightPriority('render', key, PIPELINE_PRIORITY_TIERS.CRITICAL);
          }
        }
        if (kind === null || kind === 'compute') {
          if (this._computePipelines.has(key)) {
            this._getLru(this._computePipelines, key, PIPELINE_PRIORITY_TIERS.CRITICAL);
          }
          if (this._inFlightCompute.has(key)) {
            this._recordInFlightPriority('compute', key, PIPELINE_PRIORITY_TIERS.CRITICAL);
          }
        }
        for (const q of this._activePrewarmQueues) {
          if (kind !== null && q.kind !== kind) continue;
          if (q.promoteKey(key)) {
            promoted = true;
          }
        }
      }

      const asyncCache = this._asyncPipelineCache;
      if (asyncCache) {
        const candidates = [];
        if (rawHint !== undefined && rawHint !== null) candidates.push(rawHint);
        if (key !== null && key !== undefined && key !== rawHint) candidates.push(key);
        for (let i = 0; i < candidates.length; i += 1) {
          const target = candidates[i];
          if (typeof asyncCache.promoteOnDraw === 'function') {
            if (asyncCache.promoteOnDraw(target, kind)) promoted = true;
          } else if (typeof asyncCache.promote === 'function') {
            if (asyncCache.promote(target, 'critical', kind)) promoted = true;
          }
        }
      }

      if (promoted) {
        this._drawnFirstPromotions += 1;
      }
      return promoted;
    }

    /**
     * W379: Promote a pipeline needed by an active 3D draw ahead of queued background prewarm work,
     * forwarding to any active prewarm pool and attached `AsyncPipelineCache`.
     *
     * @param {string | bigint | number | object} keyOrDescriptor
     * @param {string | bigint | number | object} [keyHint]
     * @param {'render' | 'compute' | null} [kind]
     * @returns {boolean}
     */
    promoteOnDraw(keyOrDescriptor, keyHint, kind = null) {
      let resolvedKind = kind;
      let resolvedHint = keyHint;
      if ((keyHint === 'render' || keyHint === 'compute') && kind === null) {
        resolvedKind = keyHint;
        resolvedHint = undefined;
      }
      const key = this._resolveOptionalKey(keyOrDescriptor, resolvedHint);
      let promoted = this._promoteKeyOnDraw(key, resolvedKind, resolvedHint !== undefined ? resolvedHint : keyOrDescriptor);
      if (!promoted && keyOrDescriptor && typeof keyOrDescriptor === 'object' && !Array.isArray(keyOrDescriptor)) {
        try {
          if (
            (resolvedKind === null || resolvedKind === 'render') &&
            (Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'vertex') ||
              Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'fragment'))
          ) {
            const norm = this.normalizeRenderDescriptor(keyOrDescriptor);
            const normKey = hashDescriptorKey(norm.descriptor);
            if (normKey !== key && this._promoteKeyOnDraw(normKey, 'render', undefined)) {
              promoted = true;
            }
          }
          if (
            (resolvedKind === null || resolvedKind === 'compute') &&
            Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'compute')
          ) {
            const norm = normalizeComputePipelineDescriptor(keyOrDescriptor);
            const normKey = hashDescriptorKey(norm.descriptor);
            if (normKey !== key && this._promoteKeyOnDraw(normKey, 'compute', undefined)) {
              promoted = true;
            }
          }
        } catch (_err) {
          // Ignore normalization failure during best-effort draw promotion.
        }
      }
      return promoted;
    }

    /**
     * W379: Alias for `promoteOnDraw` when notifying the cache of an actively drawn pipeline.
     */
    markDrawn(keyOrDescriptor, keyHint, kind = null) {
      return this.promoteOnDraw(keyOrDescriptor, keyHint, kind);
    }

    /**
     * Check whether a canonical pipeline key, keyHint, or pipeline descriptor is currently
     * quarantined due to a previous failed compilation.
     *
     * @param {string | bigint | number | object} keyOrDescriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {boolean}
     */
    isQuarantined(keyOrDescriptor, keyHint) {
      if (this._quarantinedKeys.size === 0) {
        return false;
      }
      if (keyHint !== undefined && keyHint !== null) {
        try {
          return this._quarantinedKeys.has(this._resolveKey(keyOrDescriptor, keyHint));
        } catch (_err) {
          return false;
        }
      }
      if (typeof keyOrDescriptor === 'string') {
        return keyOrDescriptor.length > 0 && this._quarantinedKeys.has(keyOrDescriptor);
      }
      if (typeof keyOrDescriptor === 'bigint' || typeof keyOrDescriptor === 'number') {
        try {
          return this._quarantinedKeys.has(this._resolveKey(null, keyOrDescriptor));
        } catch (_err) {
          return false;
        }
      }
      if (keyOrDescriptor && typeof keyOrDescriptor === 'object' && !Array.isArray(keyOrDescriptor)) {
        try {
          const directKey = hashDescriptorKey(keyOrDescriptor);
          if (this._quarantinedKeys.has(directKey)) {
            return true;
          }
          if (
            Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'vertex') ||
            Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'fragment')
          ) {
            const norm = this.normalizeRenderDescriptor(keyOrDescriptor);
            if (this._quarantinedKeys.has(hashDescriptorKey(norm.descriptor))) {
              return true;
            }
          }
          if (Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'compute')) {
            const norm = normalizeComputePipelineDescriptor(keyOrDescriptor);
            if (this._quarantinedKeys.has(hashDescriptorKey(norm.descriptor))) {
              return true;
            }
          }
        } catch (_err) {
          return false;
        }
      }
      return false;
    }

    /**
     * Return shallow sanitized copies of currently quarantined pipeline key records.
     *
     * @returns {Array<{
     *   key: string,
     *   kind: string,
     *   reason: string,
     *   errorType: string,
     *   failureCount: number,
     *   hitCount: number,
     *   quarantinedAt: number,
     *   lastFailedAt: number,
     *   lastHitAt: number | null,
     * }>}
     */
    getQuarantinedKeys() {
      const out = [];
      for (const entry of this._quarantinedKeys.values()) {
        out.push({
          key: sanitizePipelineErrorMessage(entry.key, MAX_PIPELINE_ERROR_KEY_LENGTH),
          kind: entry.kind,
          reason: entry.reason,
          reasonBucket: entry.reasonBucket || PIPELINE_FAILURE_REASON.OTHER,
          errorType: entry.errorType,
          failureCount: entry.failureCount,
          hitCount: entry.hitCount,
          quarantinedAt: entry.quarantinedAt,
          lastFailedAt: entry.lastFailedAt,
          lastHitAt: entry.lastHitAt,
        });
      }
      return out;
    }

    /**
     * W379: Return a sanitized summary of failed-pipeline quarantine counts grouped by fixed
     * reason bucket (`validation`, `wgsl_compile`, `layout_mismatch`, `unsupported_feature`,
     * `device_lost`, `null_pipeline`, `other`).
     */
    getQuarantineSummary() {
      return {
        enabled: this._quarantineFailedKeys,
        maxQuarantined: this._maxQuarantinedKeys,
        totalQuarantined: this._quarantinedKeys.size,
        quarantineHits: this._stats.quarantineHits,
        evictions: this._stats.negativeCacheEvictions,
        byReason: this._getQuarantinedByReason(),
        hitsByReason: { ...this._quarantineHitsByReason },
        failuresByReason: { ...this._quarantineFailuresByReason },
      };
    }

    /**
     * Clear all quarantined failed pipeline keys (or a single specified key/descriptor)
     * so future lookups may retry compilation.
     *
     * @param {string | bigint | number | object} [keyOrDescriptor]
     * @param {string | bigint | number | object} [keyHint]
     * @returns {number} Number of quarantined entries removed
     */
    clearQuarantine(keyOrDescriptor, keyHint) {
      if (keyOrDescriptor === undefined) {
        const removed = this._quarantinedKeys.size;
        this._quarantinedKeys.clear();
        return removed;
      }
      const key = this._resolveOptionalKey(keyOrDescriptor, keyHint);
      if (key !== null && this._quarantinedKeys.delete(key)) {
        return 1;
      }
      if (keyOrDescriptor && typeof keyOrDescriptor === 'object' && !Array.isArray(keyOrDescriptor)) {
        try {
          if (
            Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'vertex') ||
            Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'fragment')
          ) {
            const norm = this.normalizeRenderDescriptor(keyOrDescriptor);
            if (this._quarantinedKeys.delete(hashDescriptorKey(norm.descriptor))) {
              return 1;
            }
          }
          if (Object.prototype.hasOwnProperty.call(keyOrDescriptor, 'compute')) {
            const norm = normalizeComputePipelineDescriptor(keyOrDescriptor);
            if (this._quarantinedKeys.delete(hashDescriptorKey(norm.descriptor))) {
              return 1;
            }
          }
        } catch (_err) {
          return 0;
        }
      }
      return 0;
    }

    /**
     * Check whether a canonical pipeline key, keyHint, or pipeline descriptor is currently
     * in the bounded negative cache due to a previous failed/invalid compilation.
     *
     * @param {string | bigint | number | object} keyOrDescriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {boolean}
     */
    isNegativeCached(keyOrDescriptor, keyHint) {
      return this.isQuarantined(keyOrDescriptor, keyHint);
    }

    /**
     * Return shallow sanitized copies of currently negative-cached (quarantined) pipeline entries.
     *
     * @returns {Array<object>}
     */
    getNegativeCacheEntries() {
      return this.getQuarantinedKeys();
    }

    /**
     * Clear all negative-cached (quarantined) failed pipeline keys so future lookups may retry.
     *
     * @returns {number} Number of negative-cache entries removed
     */
    clearNegativeCache() {
      return this.clearQuarantine();
    }

    /**
     * Notify the cache that the active shader pack has been updated or reloaded.
     * Clears the bounded negative cache so previously uncompileable descriptors can retry
     * against the updated shaders, and bumps `shaderPackGeneration` so in-flight failures from
     * the previous shader pack cannot re-poison the negative cache. When `options.clearPipelines`
     * (or `options.invalidatePipelines`) is true, also clears cached render/compute pipelines.
     *
     * @param {{ clearPipelines?: boolean, invalidatePipelines?: boolean, resetStats?: boolean }} [options]
     * @returns {{
     *   clearedNegativeEntries: number,
     *   clearedPipelines: number,
     *   shaderPackGeneration: number,
     * }}
     */
    onShaderPackUpdated(options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      this._shaderPackGeneration += 1;
      const clearedNegativeEntries = this._quarantinedKeys.size;
      this._quarantinedKeys.clear();

      let clearedPipelines = 0;
      if (opts.clearPipelines === true || opts.invalidatePipelines === true) {
        clearedPipelines = this._renderPipelines.size + this._computePipelines.size;
        this._generation += 1;
        this._renderPipelines.clear();
        this._computePipelines.clear();
        this._inFlightRender.clear();
        this._inFlightCompute.clear();
        this._inFlightPriorities.clear();
        this._invalidateFastLookups();
      }

      if (opts.resetStats === true) {
        this._resetStatsCounters();
      }

      return {
        clearedNegativeEntries,
        clearedPipelines,
        shaderPackGeneration: this._shaderPackGeneration,
      };
    }

    _getOrCreateBindGroupLayout(descriptor, keyHint, hasInlineSamplerMeta, priority = null) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateBindGroupLayout requires a descriptor object');
      }
      if (typeof this._device.createBindGroupLayout !== 'function') {
        throw new TypeError('device must provide a createBindGroupLayout(descriptor) method');
      }

      const key = this._resolveKey(descriptor, keyHint);
      const cached = this._getLru(this._bindGroupLayouts, key, priority);
      if (cached !== undefined) {
        this._stats.layoutHits += 1;
        return cached;
      }

      this._stats.layoutMisses += 1;
      const t0 = this._now();
      let layout;
      try {
        layout = this._device.createBindGroupLayout(descriptor);
      } catch (err) {
        this._recordPipelineError('bind-group-layout', err, { key });
        throw err;
      }
      const compileMs = Math.max(0, this._now() - t0);
      this._recordCompileTiming(compileMs);
      if (hasInlineSamplerMeta || descriptorHasInlineSampler(descriptor)) {
        this._stats.inlineSamplerLayouts += 1;
      }

      if (layout && (typeof layout === 'object' || typeof layout === 'function')) {
        setRegisteredObjectId(layout, `bgl:${key}`);
      }
      this._setLru(
        this._bindGroupLayouts,
        key,
        layout,
        compileMs,
        this._maxBindGroupLayouts,
        priority
      );
      return layout;
    }

    /**
     * Return a cached `GPUBindGroupLayout` or create and cache it via `device.createBindGroupLayout`.
     *
     * @param {GPUBindGroupLayoutDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @param {{ priority?: 'critical' | 'high' | 'normal' | 'low' }} [options]
     * @returns {GPUBindGroupLayout}
     */
    getOrCreateBindGroupLayout(descriptor, keyHint, options) {
      return this._getOrCreateBindGroupLayout(
        descriptor,
        keyHint,
        false,
        extractPriorityOption(options)
      );
    }

    /**
     * Build and cache a `GPUBindGroupLayout` from compact S6 shader reflection JSON.
     * Accepts either `{ entries | bindings, label?, group? }` or an array of binding objects
     * `{ binding, visibility, buffer?, sampler?, texture?, storageTexture? }`.
     *
     * @param {object | Array<object>} reflectionGroup
     * @param {string | bigint | number | object} [keyHint]
     * @param {{ priority?: 'critical' | 'high' | 'normal' | 'low' }} [options]
     * @returns {GPUBindGroupLayout}
     */
    createBindGroupLayoutFromReflection(reflectionGroup, keyHint, options) {
      let rawEntries;
      let label;
      let hasInlineSampler = false;

      if (Array.isArray(reflectionGroup)) {
        rawEntries = reflectionGroup;
      } else if (reflectionGroup && typeof reflectionGroup === 'object') {
        if (Array.isArray(reflectionGroup.entries)) {
          rawEntries = reflectionGroup.entries;
        } else if (Array.isArray(reflectionGroup.bindings)) {
          rawEntries = reflectionGroup.bindings;
        } else {
          throw new TypeError(
            'reflectionGroup must be an array or an object with an "entries" or "bindings" array'
          );
        }
        if (typeof reflectionGroup.label === 'string' && reflectionGroup.label.length > 0) {
          label = reflectionGroup.label;
        }
        if (
          reflectionGroup.inlineSampler === true ||
          reflectionGroup.inline_sampler === true ||
          reflectionGroup.hasInlineSampler === true ||
          reflectionGroup.hasInlineSamplers === true ||
          (typeof reflectionGroup.inlineSamplerBindings === 'number' &&
            Number.isFinite(reflectionGroup.inlineSamplerBindings) &&
            reflectionGroup.inlineSamplerBindings > 0)
        ) {
          hasInlineSampler = true;
        }
      } else {
        throw new TypeError('reflectionGroup must be an array or object');
      }

      const seenBindings = new Set();
      const entries = [];
      for (let i = 0; i < rawEntries.length; i += 1) {
        const rawEntry = rawEntries[i];
        if (
          rawEntry &&
          typeof rawEntry === 'object' &&
          (rawEntry.inLayout === false ||
            rawEntry.in_layout === false ||
            isRtArenaReflectionBinding(rawEntry))
        ) {
          continue;
        }
        if (!hasInlineSampler && entryHasInlineSampler(rawEntry)) {
          hasInlineSampler = true;
        }
        const normalized = normalizeReflectionBinding(rawEntry, i, options || null);
        if (seenBindings.has(normalized.binding)) {
          throw new Error(`Duplicate reflection binding index ${normalized.binding}`);
        }
        seenBindings.add(normalized.binding);
        entries.push(normalized);
      }

      // Sort by binding index so semantically identical reflection groups share one cache key.
      entries.sort((a, b) => a.binding - b.binding);

      const descriptor = label !== undefined ? { label, entries } : { entries };
      return this._getOrCreateBindGroupLayout(
        descriptor,
        keyHint,
        hasInlineSampler,
        extractPriorityOption(options)
      );
    }

    /**
     * Fast-path O(1) lookup for an already-compiled `GPURenderPipeline` by numeric/bigint/string
     * `fastKey` / `psoId` or by descriptor object (via WeakMap memoization + 53-bit integer hash),
     * avoiding full descriptor normalization and canonical string/JSON serialization on the hot
     * per-frame draw loop.
     *
     * When called with `(fastKey, descriptor, options)`, a fast-path miss automatically falls back
     * to compiling/fetching via `descriptor` and binds `fastKey` for subsequent O(1) hits.
     * When called with a single argument `(fastKey)` or `(descriptor)`, a miss returns `null`
     * without triggering synchronous compilation.
     *
     * @param {number | bigint | string | GPURenderPipelineDescriptor} fastKeyOrDescriptor
     * @param {GPURenderPipelineDescriptor} [descriptor]
     * @param {object} [options]
     * @returns {GPURenderPipeline | null}
     */
    getFastRenderPipeline(fastKeyOrDescriptor, descriptor, options) {
      const priority = extractPriorityOption(options);
      if (
        typeof fastKeyOrDescriptor === 'string' ||
        typeof fastKeyOrDescriptor === 'number' ||
        typeof fastKeyOrDescriptor === 'bigint'
      ) {
        if (!this._isValidFastKey(fastKeyOrDescriptor)) {
          this._stats.fastLookupMisses += 1;
          return null;
        }
        const hit = this._getFastLookup(
          this._fastRenderLookup,
          this._renderPipelines,
          fastKeyOrDescriptor,
          priority
        );
        if (hit !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.renderHits += 1;
          return hit;
        }
        this._stats.fastLookupMisses += 1;
        if (descriptor && typeof descriptor === 'object') {
          return this._getOrCreateRenderPipelineSlow(descriptor, fastKeyOrDescriptor, options);
        }
        return null;
      }

      if (fastKeyOrDescriptor && typeof fastKeyOrDescriptor === 'object') {
        const hit = this._tryFastRenderLookupByDescriptor(fastKeyOrDescriptor, priority);
        if (hit !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.renderHits += 1;
          return hit;
        }
        this._stats.fastLookupMisses += 1;
        return null;
      }

      this._stats.fastLookupMisses += 1;
      return null;
    }

    /**
     * Synchronously return a cached `GPURenderPipeline` or compile via `device.createRenderPipeline`.
     * Automatically normalizes missing layouts, malformed vertex buffers, and G34 fallback profiles
     * while leaving already-valid non-fallback descriptors untouched.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @param {object} [options]
     * @returns {GPURenderPipeline}
     */
    getOrCreateRenderPipeline(descriptor, keyHint, options) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateRenderPipeline requires a descriptor object');
      }
      if (typeof this._device.createRenderPipeline !== 'function') {
        throw new TypeError('device must provide a createRenderPipeline(descriptor) method');
      }

      const priority = extractPriorityOption(options);
      if (
        keyHint !== undefined &&
        keyHint !== null &&
        (typeof keyHint === 'string' || typeof keyHint === 'bigint' || typeof keyHint === 'number')
      ) {
        if (!this._isValidFastKey(keyHint)) {
          this._resolveKey(null, keyHint);
        }
        const fastCached = this._getFastLookup(
          this._fastRenderLookup,
          this._renderPipelines,
          keyHint,
          priority
        );
        if (fastCached !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.renderHits += 1;
          return fastCached;
        }
        this._stats.fastLookupMisses += 1;
      } else if (
        (keyHint === undefined || keyHint === null) &&
        !hasRenderNormalizationOptions(options)
      ) {
        const fastCached = this._tryFastRenderLookupByDescriptor(descriptor, priority);
        if (fastCached !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.renderHits += 1;
          return fastCached;
        }
        this._stats.fastLookupMisses += 1;
      } else {
        this._stats.fastLookupMisses += 1;
      }

      return this._getOrCreateRenderPipelineSlow(descriptor, keyHint, options);
    }

    _getOrCreateRenderPipelineSlow(descriptor, keyHint, options) {
      if (typeof this._device.createRenderPipeline !== 'function') {
        throw new TypeError('device must provide a createRenderPipeline(descriptor) method');
      }

      const priority = extractPriorityOption(options);
      if (
        keyHint !== undefined &&
        keyHint !== null &&
        (typeof keyHint === 'string' || typeof keyHint === 'bigint' || typeof keyHint === 'number')
      ) {
        const hintKey = this._resolveKey(null, keyHint);
        const hintCached = this._getLru(this._renderPipelines, hintKey, priority);
        if (hintCached !== undefined) {
          const lruEntry = this._renderPipelines.get(hintKey);
          this._setFastLookup(this._fastRenderLookup, hintKey, hintKey, lruEntry);
          this._setFastLookup(this._fastRenderLookup, keyHint, hintKey, lruEntry);
          this._stats.renderHits += 1;
          return hintCached;
        }
        if (this._hitQuarantine(hintKey, 'render')) {
          return null;
        }
      }

      const opts = options && typeof options === 'object' ? options : null;
      const norm = this.normalizeRenderDescriptor(descriptor, opts || undefined);
      const effectiveDescriptor = norm.descriptor;

      const key = this._resolveKey(effectiveDescriptor, keyHint);
      const cached = this._getLru(this._renderPipelines, key, priority);
      if (cached !== undefined) {
        this._populateFastRenderEntry(
          descriptor,
          norm,
          key,
          keyHint,
          opts,
          this._renderPipelines.get(key)
        );
        this._stats.renderHits += 1;
        return cached;
      }

      if (this._hitQuarantine(key, 'render')) {
        return null;
      }

      this._stats.renderMisses += 1;
      this._recordNormalizationStats(norm);

      const canRetryAutoLayout =
        effectiveDescriptor.layout !== 'auto' &&
        Boolean((opts && opts.retryWithAutoLayout === true) || norm.fallbackProfile !== null);

      const t0 = this._now();
      let pipeline;
      try {
        pipeline = this._device.createRenderPipeline(effectiveDescriptor);
      } catch (err) {
        if (canRetryAutoLayout) {
          const retryDescriptor = {
            ...effectiveDescriptor,
            layout: 'auto',
          };
          try {
            pipeline = this._device.createRenderPipeline(retryDescriptor);
            if (!pipeline) {
              const emptyErr = new Error('createRenderPipeline returned empty pipeline');
              this._recordPipelineError('render', err, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: false,
              });
              this._recordPipelineError('render', emptyErr, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: false,
              });
              this._quarantineKey(key, 'render', emptyErr);
              return null;
            }
            this._recordPipelineError('render', err, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: true,
            });
            this._stats.fallbackRecoveryCount += 1;
            if (!norm.layoutNormalized) {
              this._stats.layoutNormalizedCount += 1;
            }
            if (!norm.normalized) {
              this._stats.fallbackNormalizedCount += 1;
            }
          } catch (retryErr) {
            this._recordPipelineError('render', err, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: false,
            });
            this._recordPipelineError('render', retryErr, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: false,
            });
            this._quarantineKey(key, 'render', retryErr);
            throw retryErr;
          }
        } else {
          this._recordPipelineError('render', err, {
            key,
            fallbackProfile: norm.fallbackProfile,
            recovered: false,
          });
          this._quarantineKey(key, 'render', err);
          throw err;
        }
      }

      if (!pipeline) {
        const emptyErr = new Error('createRenderPipeline returned empty pipeline');
        if (canRetryAutoLayout) {
          const retryDescriptor = {
            ...effectiveDescriptor,
            layout: 'auto',
          };
          try {
            pipeline = this._device.createRenderPipeline(retryDescriptor);
          } catch (retryErr) {
            this._recordPipelineError('render', emptyErr, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: false,
            });
            this._recordPipelineError('render', retryErr, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: false,
            });
            this._quarantineKey(key, 'render', retryErr);
            throw retryErr;
          }
          if (pipeline) {
            this._recordPipelineError('render', emptyErr, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: true,
            });
            this._stats.fallbackRecoveryCount += 1;
            if (!norm.layoutNormalized) {
              this._stats.layoutNormalizedCount += 1;
            }
            if (!norm.normalized) {
              this._stats.fallbackNormalizedCount += 1;
            }
          }
        }
        if (!pipeline) {
          this._recordPipelineError('render', emptyErr, {
            key,
            fallbackProfile: norm.fallbackProfile,
            recovered: false,
          });
          this._quarantineKey(key, 'render', emptyErr);
          return null;
        }
      }

      const compileMs = Math.max(0, this._now() - t0);
      this._recordCompileTiming(compileMs);
      this._recordRenderVariantStats(effectiveDescriptor);

      const lruEntry = this._setLru(
        this._renderPipelines,
        key,
        pipeline,
        compileMs,
        this._maxRenderPipelines,
        priority
      );
      this._populateFastRenderEntry(descriptor, norm, key, keyHint, opts, lruEntry);
      return pipeline;
    }

    /**
     * Asynchronously return a cached `GPURenderPipeline`, deduplicating concurrent in-flight
     * compilations for the same key. Uses `device.createRenderPipelineAsync` when available,
     * falling back to synchronous `device.createRenderPipeline`.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @param {object} [options]
     * @returns {Promise<GPURenderPipeline | null>}
     */
    getOrCreateRenderPipelineAsync(descriptor, keyHint, options) {
      if (!descriptor || typeof descriptor !== 'object') {
        return Promise.reject(
          new TypeError('getOrCreateRenderPipelineAsync requires a descriptor object')
        );
      }

      const priority = extractPriorityOption(options);
      if (
        keyHint !== undefined &&
        keyHint !== null &&
        (typeof keyHint === 'string' || typeof keyHint === 'bigint' || typeof keyHint === 'number')
      ) {
        if (!this._isValidFastKey(keyHint)) {
          try {
            this._resolveKey(null, keyHint);
          } catch (err) {
            return Promise.reject(err);
          }
        }
        const fastCached = this._getFastLookup(
          this._fastRenderLookup,
          this._renderPipelines,
          keyHint,
          priority
        );
        if (fastCached !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.renderHits += 1;
          return Promise.resolve(fastCached);
        }
        this._stats.fastLookupMisses += 1;

        let hintKey;
        try {
          hintKey = this._resolveKey(null, keyHint);
        } catch (err) {
          return Promise.reject(err);
        }
        const hintCached = this._getLru(this._renderPipelines, hintKey, priority);
        if (hintCached !== undefined) {
          const lruEntry = this._renderPipelines.get(hintKey);
          this._setFastLookup(this._fastRenderLookup, hintKey, hintKey, lruEntry);
          this._setFastLookup(this._fastRenderLookup, keyHint, hintKey, lruEntry);
          this._stats.renderHits += 1;
          return Promise.resolve(hintCached);
        }
        const hintInFlight = this._inFlightRender.get(hintKey);
        if (hintInFlight !== undefined) {
          this._recordInFlightPriority('render', hintKey, priority);
          this._stats.inFlightDedupes += 1;
          return hintInFlight;
        }
        if (this._hitQuarantine(hintKey, 'render')) {
          return Promise.resolve(null);
        }
      } else if (
        (keyHint === undefined || keyHint === null) &&
        !hasRenderNormalizationOptions(options)
      ) {
        const fastCached = this._tryFastRenderLookupByDescriptor(descriptor, priority);
        if (fastCached !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.renderHits += 1;
          return Promise.resolve(fastCached);
        }
        this._stats.fastLookupMisses += 1;
      } else {
        this._stats.fastLookupMisses += 1;
      }

      const opts = options && typeof options === 'object' ? options : null;
      let norm;
      let key;
      try {
        norm = this.normalizeRenderDescriptor(descriptor, opts || undefined);
        key = this._resolveKey(norm.descriptor, keyHint);
      } catch (err) {
        return Promise.reject(err);
      }

      const effectiveDescriptor = norm.descriptor;
      const cached = this._getLru(this._renderPipelines, key, priority);
      if (cached !== undefined) {
        this._populateFastRenderEntry(
          descriptor,
          norm,
          key,
          keyHint,
          opts,
          this._renderPipelines.get(key)
        );
        this._stats.renderHits += 1;
        return Promise.resolve(cached);
      }

      const isDrawnPriority =
        priority === PIPELINE_PRIORITY_TIERS.CRITICAL ||
        Boolean(opts && (opts.drawn === true || opts.demand === true));
      if (isDrawnPriority) {
        this._promoteKeyOnDraw(key, 'render', keyHint);
      }

      const existing = this._inFlightRender.get(key);
      if (existing !== undefined) {
        this._recordInFlightPriority('render', key, priority);
        this._stats.inFlightDedupes += 1;
        return existing;
      }

      if (this._hitQuarantine(key, 'render')) {
        return Promise.resolve(null);
      }

      const hasAsync = typeof this._device.createRenderPipelineAsync === 'function';
      const hasSync = typeof this._device.createRenderPipeline === 'function';
      if (!hasAsync && !hasSync) {
        return Promise.reject(
          new TypeError('device must provide createRenderPipelineAsync or createRenderPipeline')
        );
      }

      this._stats.renderMisses += 1;
      this._recordNormalizationStats(norm);

      const canRetryAutoLayout =
        effectiveDescriptor.layout !== 'auto' &&
        Boolean((opts && opts.retryWithAutoLayout === true) || norm.fallbackProfile !== null);

      const t0 = this._now();

      if (!hasAsync) {
        try {
          let pipeline;
          try {
            pipeline = this._device.createRenderPipeline(effectiveDescriptor);
          } catch (err) {
            if (!canRetryAutoLayout) {
              this._recordPipelineError('render', err, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: false,
              });
              this._quarantineKey(key, 'render', err);
              throw err;
            }
            const retryDescriptor = { ...effectiveDescriptor, layout: 'auto' };
            try {
              pipeline = this._device.createRenderPipeline(retryDescriptor);
              if (!pipeline) {
                const emptyErr = new Error('createRenderPipeline returned empty pipeline');
                this._recordPipelineError('render', err, {
                  key,
                  fallbackProfile: norm.fallbackProfile,
                  recovered: false,
                });
                this._recordPipelineError('render', emptyErr, {
                  key,
                  fallbackProfile: norm.fallbackProfile,
                  recovered: false,
                });
                this._quarantineKey(key, 'render', emptyErr);
                return Promise.resolve(null);
              }
              this._recordPipelineError('render', err, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: true,
              });
              this._stats.fallbackRecoveryCount += 1;
              if (!norm.layoutNormalized) {
                this._stats.layoutNormalizedCount += 1;
              }
              if (!norm.normalized) {
                this._stats.fallbackNormalizedCount += 1;
              }
            } catch (retryErr) {
              this._recordPipelineError('render', err, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: false,
              });
              this._recordPipelineError('render', retryErr, {
                key,
                fallbackProfile: norm.fallbackProfile,
                recovered: false,
              });
              this._quarantineKey(key, 'render', retryErr);
              throw retryErr;
            }
          }
          if (!pipeline) {
            const emptyErr = new Error('createRenderPipeline returned empty pipeline');
            this._recordPipelineError('render', emptyErr, {
              key,
              fallbackProfile: norm.fallbackProfile,
              recovered: false,
            });
            this._quarantineKey(key, 'render', emptyErr);
            return Promise.resolve(null);
          }
          const compileMs = Math.max(0, this._now() - t0);
          this._recordCompileTiming(compileMs);
          this._recordRenderVariantStats(effectiveDescriptor);
          const lruEntry = this._setLru(
            this._renderPipelines,
            key,
            pipeline,
            compileMs,
            this._maxRenderPipelines,
            priority
          );
          this._populateFastRenderEntry(descriptor, norm, key, keyHint, opts, lruEntry);
          return Promise.resolve(pipeline);
        } catch (err) {
          return Promise.reject(err);
        }
      }

      const generation = this._generation;
      const shaderPackGeneration = this._shaderPackGeneration;
      const invokeCreateAsync = (desc) => {
        try {
          return Promise.resolve(this._device.createRenderPipelineAsync(desc));
        } catch (syncErr) {
          return Promise.reject(syncErr);
        }
      };

      const finishSuccess = (pipeline) => {
        if (this._inFlightRender.get(key) === trackedPromise) {
          this._inFlightRender.delete(key);
        }
        const effectivePriority = this._consumeInFlightPriority('render', key, priority);
        if (!pipeline) {
          const emptyErr = new Error('createRenderPipelineAsync returned empty pipeline');
          this._recordPipelineError('render', emptyErr, {
            key,
            fallbackProfile: norm.fallbackProfile,
            recovered: false,
          });
          if (generation === this._generation && shaderPackGeneration === this._shaderPackGeneration) {
            this._quarantineKey(key, 'render', emptyErr);
          }
          return null;
        }
        const compileMs = Math.max(0, this._now() - t0);
        this._recordCompileTiming(compileMs);
        if (generation === this._generation) {
          this._recordRenderVariantStats(effectiveDescriptor);
          const lruEntry = this._setLru(
            this._renderPipelines,
            key,
            pipeline,
            compileMs,
            this._maxRenderPipelines,
            effectivePriority
          );
          this._populateFastRenderEntry(descriptor, norm, key, keyHint, opts, lruEntry);
        }
        return pipeline;
      };

      const trackedPromise = invokeCreateAsync(effectiveDescriptor).then(
        finishSuccess,
        (err) => {
          if (canRetryAutoLayout) {
            const retryDescriptor = { ...effectiveDescriptor, layout: 'auto' };
            return invokeCreateAsync(retryDescriptor).then(
              (recoveredPipeline) => {
                if (!recoveredPipeline) {
                  if (this._inFlightRender.get(key) === trackedPromise) {
                    this._inFlightRender.delete(key);
                  }
                  this._clearInFlightPriority('render', key);
                  const emptyErr = new Error('createRenderPipelineAsync returned empty pipeline');
                  this._recordPipelineError('render', err, {
                    key,
                    fallbackProfile: norm.fallbackProfile,
                    recovered: false,
                  });
                  this._recordPipelineError('render', emptyErr, {
                    key,
                    fallbackProfile: norm.fallbackProfile,
                    recovered: false,
                  });
                  if (
                    generation === this._generation &&
                    shaderPackGeneration === this._shaderPackGeneration
                  ) {
                    this._quarantineKey(key, 'render', emptyErr);
                  }
                  return null;
                }
                this._recordPipelineError('render', err, {
                  key,
                  fallbackProfile: norm.fallbackProfile,
                  recovered: true,
                });
                this._stats.fallbackRecoveryCount += 1;
                if (!norm.layoutNormalized) {
                  this._stats.layoutNormalizedCount += 1;
                }
                if (!norm.normalized) {
                  this._stats.fallbackNormalizedCount += 1;
                }
                return finishSuccess(recoveredPipeline);
              },
              (retryErr) => {
                if (this._inFlightRender.get(key) === trackedPromise) {
                  this._inFlightRender.delete(key);
                }
                this._clearInFlightPriority('render', key);
                this._recordPipelineError('render', err, {
                  key,
                  fallbackProfile: norm.fallbackProfile,
                  recovered: false,
                });
                this._recordPipelineError('render', retryErr, {
                  key,
                  fallbackProfile: norm.fallbackProfile,
                  recovered: false,
                });
                if (
                  generation === this._generation &&
                  shaderPackGeneration === this._shaderPackGeneration
                ) {
                  this._quarantineKey(key, 'render', retryErr);
                }
                throw retryErr;
              }
            );
          }

          if (this._inFlightRender.get(key) === trackedPromise) {
            this._inFlightRender.delete(key);
          }
          this._clearInFlightPriority('render', key);
          this._recordPipelineError('render', err, {
            key,
            fallbackProfile: norm.fallbackProfile,
            recovered: false,
          });
          if (
            generation === this._generation &&
            shaderPackGeneration === this._shaderPackGeneration
          ) {
            this._quarantineKey(key, 'render', err);
          }
          throw err;
        }
      );

      this._inFlightRender.set(key, trackedPromise);
      return trackedPromise;
    }

    /**
     * Convenience wrapper to synchronously create/fetch a fallback-safe render pipeline for a
     * G34 profile (`fullscreen_triangle_cbv`, `textured_quad_24b`, or `default`) with automatic
     * vertex/CBV layout normalization and auto-layout retry recovery.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {object} [options]
     * @returns {GPURenderPipeline}
     */
    getOrCreateFallbackRenderPipeline(descriptor, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const detected = detectFallbackProfile(descriptor, opts);
      const fallbackOpts = {
        ...opts,
        fallbackProfile: detected || 'default',
        retryWithAutoLayout: opts.retryWithAutoLayout !== false,
      };
      return this.getOrCreateRenderPipeline(descriptor, opts.keyHint, fallbackOpts);
    }

    /**
     * Convenience wrapper to asynchronously create/fetch a fallback-safe render pipeline for a
     * G34 profile (`fullscreen_triangle_cbv`, `textured_quad_24b`, or `default`).
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {object} [options]
     * @returns {Promise<GPURenderPipeline>}
     */
    getOrCreateFallbackRenderPipelineAsync(descriptor, options = {}) {
      const opts = options && typeof options === 'object' ? options : {};
      const detected = detectFallbackProfile(descriptor, opts);
      const fallbackOpts = {
        ...opts,
        fallbackProfile: detected || 'default',
        retryWithAutoLayout: opts.retryWithAutoLayout !== false,
      };
      return this.getOrCreateRenderPipelineAsync(descriptor, opts.keyHint, fallbackOpts);
    }

    /**
     * Fast-path O(1) lookup for an already-compiled `GPUComputePipeline` by numeric/bigint/string
     * `fastKey` / `psoId` or by descriptor object (via WeakMap memoization + 53-bit integer hash).
     *
     * @param {number | bigint | string | GPUComputePipelineDescriptor} fastKeyOrDescriptor
     * @param {GPUComputePipelineDescriptor} [descriptor]
     * @returns {GPUComputePipeline | null}
     */
    getFastComputePipeline(fastKeyOrDescriptor, descriptor, options) {
      const priority = extractPriorityOption(options);
      if (
        typeof fastKeyOrDescriptor === 'string' ||
        typeof fastKeyOrDescriptor === 'number' ||
        typeof fastKeyOrDescriptor === 'bigint'
      ) {
        if (!this._isValidFastKey(fastKeyOrDescriptor)) {
          this._stats.fastLookupMisses += 1;
          return null;
        }
        const hit = this._getFastLookup(
          this._fastComputeLookup,
          this._computePipelines,
          fastKeyOrDescriptor,
          priority
        );
        if (hit !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.computeHits += 1;
          return hit;
        }
        this._stats.fastLookupMisses += 1;
        if (descriptor && typeof descriptor === 'object') {
          return this._getOrCreateComputePipelineSlow(descriptor, fastKeyOrDescriptor, options);
        }
        return null;
      }

      if (fastKeyOrDescriptor && typeof fastKeyOrDescriptor === 'object') {
        const hit = this._tryFastComputeLookupByDescriptor(fastKeyOrDescriptor, priority);
        if (hit !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.computeHits += 1;
          return hit;
        }
        this._stats.fastLookupMisses += 1;
        return null;
      }

      this._stats.fastLookupMisses += 1;
      return null;
    }

    /**
     * Synchronously return a cached `GPUComputePipeline` or compile via `device.createComputePipeline`.
     *
     * @param {GPUComputePipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @param {{ priority?: 'critical' | 'high' | 'normal' | 'low' }} [options]
     * @returns {GPUComputePipeline}
     */
    getOrCreateComputePipeline(descriptor, keyHint, options) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateComputePipeline requires a descriptor object');
      }
      if (typeof this._device.createComputePipeline !== 'function') {
        throw new TypeError('device must provide a createComputePipeline(descriptor) method');
      }

      const priority = extractPriorityOption(options);
      if (
        keyHint !== undefined &&
        keyHint !== null &&
        (typeof keyHint === 'string' || typeof keyHint === 'bigint' || typeof keyHint === 'number')
      ) {
        if (!this._isValidFastKey(keyHint)) {
          this._resolveKey(null, keyHint);
        }
        const fastCached = this._getFastLookup(
          this._fastComputeLookup,
          this._computePipelines,
          keyHint,
          priority
        );
        if (fastCached !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.computeHits += 1;
          return fastCached;
        }
        this._stats.fastLookupMisses += 1;
      } else if (keyHint === undefined || keyHint === null) {
        const fastCached = this._tryFastComputeLookupByDescriptor(descriptor, priority);
        if (fastCached !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.computeHits += 1;
          return fastCached;
        }
        this._stats.fastLookupMisses += 1;
      } else {
        this._stats.fastLookupMisses += 1;
      }

      return this._getOrCreateComputePipelineSlow(descriptor, keyHint, options);
    }

    _getOrCreateComputePipelineSlow(descriptor, keyHint, options) {
      if (typeof this._device.createComputePipeline !== 'function') {
        throw new TypeError('device must provide a createComputePipeline(descriptor) method');
      }

      const priority = extractPriorityOption(options);
      if (
        keyHint !== undefined &&
        keyHint !== null &&
        (typeof keyHint === 'string' || typeof keyHint === 'bigint' || typeof keyHint === 'number')
      ) {
        const hintKey = this._resolveKey(null, keyHint);
        const hintCached = this._getLru(this._computePipelines, hintKey, priority);
        if (hintCached !== undefined) {
          const lruEntry = this._computePipelines.get(hintKey);
          this._setFastLookup(this._fastComputeLookup, hintKey, hintKey, lruEntry);
          this._setFastLookup(this._fastComputeLookup, keyHint, hintKey, lruEntry);
          this._stats.computeHits += 1;
          return hintCached;
        }
        if (this._hitQuarantine(hintKey, 'compute')) {
          return null;
        }
      }

      const norm = normalizeComputePipelineDescriptor(descriptor);
      const effectiveDescriptor = norm.descriptor;

      const key = this._resolveKey(effectiveDescriptor, keyHint);
      const cached = this._getLru(this._computePipelines, key, priority);
      if (cached !== undefined) {
        this._populateFastComputeEntry(
          descriptor,
          norm,
          key,
          keyHint,
          this._computePipelines.get(key)
        );
        this._stats.computeHits += 1;
        return cached;
      }

      if (this._hitQuarantine(key, 'compute')) {
        return null;
      }

      this._stats.computeMisses += 1;
      this._recordNormalizationStats(norm);

      const t0 = this._now();
      let pipeline;
      try {
        pipeline = this._device.createComputePipeline(effectiveDescriptor);
      } catch (err) {
        this._recordPipelineError('compute', err, { key });
        this._quarantineKey(key, 'compute', err);
        throw err;
      }
      if (!pipeline) {
        const emptyErr = new Error('createComputePipeline returned empty pipeline');
        this._recordPipelineError('compute', emptyErr, { key });
        this._quarantineKey(key, 'compute', emptyErr);
        return null;
      }
      const compileMs = Math.max(0, this._now() - t0);
      this._recordCompileTiming(compileMs);

      const lruEntry = this._setLru(
        this._computePipelines,
        key,
        pipeline,
        compileMs,
        this._maxComputePipelines,
        priority
      );
      this._populateFastComputeEntry(descriptor, norm, key, keyHint, lruEntry);
      return pipeline;
    }

    /**
     * Asynchronously return a cached `GPUComputePipeline`, deduplicating concurrent in-flight
     * compilations for the same key. Uses `device.createComputePipelineAsync` when available,
     * falling back to synchronous `device.createComputePipeline`.
     *
     * @param {GPUComputePipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @param {{ priority?: 'critical' | 'high' | 'normal' | 'low' }} [options]
     * @returns {Promise<GPUComputePipeline | null>}
     */
    getOrCreateComputePipelineAsync(descriptor, keyHint, options) {
      if (!descriptor || typeof descriptor !== 'object') {
        return Promise.reject(
          new TypeError('getOrCreateComputePipelineAsync requires a descriptor object')
        );
      }

      const priority = extractPriorityOption(options);
      if (
        keyHint !== undefined &&
        keyHint !== null &&
        (typeof keyHint === 'string' || typeof keyHint === 'bigint' || typeof keyHint === 'number')
      ) {
        if (!this._isValidFastKey(keyHint)) {
          try {
            this._resolveKey(null, keyHint);
          } catch (err) {
            return Promise.reject(err);
          }
        }
        const fastCached = this._getFastLookup(
          this._fastComputeLookup,
          this._computePipelines,
          keyHint,
          priority
        );
        if (fastCached !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.computeHits += 1;
          return Promise.resolve(fastCached);
        }
        this._stats.fastLookupMisses += 1;

        let hintKey;
        try {
          hintKey = this._resolveKey(null, keyHint);
        } catch (err) {
          return Promise.reject(err);
        }
        const hintCached = this._getLru(this._computePipelines, hintKey, priority);
        if (hintCached !== undefined) {
          const lruEntry = this._computePipelines.get(hintKey);
          this._setFastLookup(this._fastComputeLookup, hintKey, hintKey, lruEntry);
          this._setFastLookup(this._fastComputeLookup, keyHint, hintKey, lruEntry);
          this._stats.computeHits += 1;
          return Promise.resolve(hintCached);
        }
        const hintInFlight = this._inFlightCompute.get(hintKey);
        if (hintInFlight !== undefined) {
          this._recordInFlightPriority('compute', hintKey, priority);
          this._stats.inFlightDedupes += 1;
          return hintInFlight;
        }
        if (this._hitQuarantine(hintKey, 'compute')) {
          return Promise.resolve(null);
        }
      } else if (keyHint === undefined || keyHint === null) {
        const fastCached = this._tryFastComputeLookupByDescriptor(descriptor, priority);
        if (fastCached !== undefined) {
          this._stats.fastLookupHits += 1;
          this._stats.computeHits += 1;
          return Promise.resolve(fastCached);
        }
        this._stats.fastLookupMisses += 1;
      } else {
        this._stats.fastLookupMisses += 1;
      }

      let norm;
      let key;
      try {
        norm = normalizeComputePipelineDescriptor(descriptor);
        key = this._resolveKey(norm.descriptor, keyHint);
      } catch (err) {
        return Promise.reject(err);
      }

      const effectiveDescriptor = norm.descriptor;
      const cached = this._getLru(this._computePipelines, key, priority);
      if (cached !== undefined) {
        this._populateFastComputeEntry(
          descriptor,
          norm,
          key,
          keyHint,
          this._computePipelines.get(key)
        );
        this._stats.computeHits += 1;
        return Promise.resolve(cached);
      }

      const isDrawnPriority =
        priority === PIPELINE_PRIORITY_TIERS.CRITICAL ||
        Boolean(options && (options.drawn === true || options.demand === true));
      if (isDrawnPriority) {
        this._promoteKeyOnDraw(key, 'compute', keyHint);
      }

      const existing = this._inFlightCompute.get(key);
      if (existing !== undefined) {
        this._recordInFlightPriority('compute', key, priority);
        this._stats.inFlightDedupes += 1;
        return existing;
      }

      if (this._hitQuarantine(key, 'compute')) {
        return Promise.resolve(null);
      }

      const hasAsync = typeof this._device.createComputePipelineAsync === 'function';
      const hasSync = typeof this._device.createComputePipeline === 'function';
      if (!hasAsync && !hasSync) {
        return Promise.reject(
          new TypeError('device must provide createComputePipelineAsync or createComputePipeline')
        );
      }

      this._stats.computeMisses += 1;
      this._recordNormalizationStats(norm);
      const t0 = this._now();

      if (!hasAsync) {
        try {
          const pipeline = this._device.createComputePipeline(effectiveDescriptor);
          if (!pipeline) {
            const emptyErr = new Error('createComputePipeline returned empty pipeline');
            this._recordPipelineError('compute', emptyErr, { key });
            this._quarantineKey(key, 'compute', emptyErr);
            return Promise.resolve(null);
          }
          const compileMs = Math.max(0, this._now() - t0);
          this._recordCompileTiming(compileMs);
          const lruEntry = this._setLru(
            this._computePipelines,
            key,
            pipeline,
            compileMs,
            this._maxComputePipelines,
            priority
          );
          this._populateFastComputeEntry(descriptor, norm, key, keyHint, lruEntry);
          return Promise.resolve(pipeline);
        } catch (err) {
          this._recordPipelineError('compute', err, { key });
          this._quarantineKey(key, 'compute', err);
          return Promise.reject(err);
        }
      }

      const generation = this._generation;
      const shaderPackGeneration = this._shaderPackGeneration;
      let rawPromise;
      try {
        rawPromise = Promise.resolve(this._device.createComputePipelineAsync(effectiveDescriptor));
      } catch (err) {
        this._recordPipelineError('compute', err, { key });
        this._quarantineKey(key, 'compute', err);
        return Promise.reject(err);
      }

      const trackedPromise = rawPromise.then(
        (pipeline) => {
          if (this._inFlightCompute.get(key) === trackedPromise) {
            this._inFlightCompute.delete(key);
          }
          const effectivePriority = this._consumeInFlightPriority('compute', key, priority);
          if (!pipeline) {
            const emptyErr = new Error('createComputePipelineAsync returned empty pipeline');
            this._recordPipelineError('compute', emptyErr, { key });
            if (
              generation === this._generation &&
              shaderPackGeneration === this._shaderPackGeneration
            ) {
              this._quarantineKey(key, 'compute', emptyErr);
            }
            return null;
          }
          const compileMs = Math.max(0, this._now() - t0);
          this._recordCompileTiming(compileMs);
          if (generation === this._generation) {
            const lruEntry = this._setLru(
              this._computePipelines,
              key,
              pipeline,
              compileMs,
              this._maxComputePipelines,
              effectivePriority
            );
            this._populateFastComputeEntry(descriptor, norm, key, keyHint, lruEntry);
          }
          return pipeline;
        },
        (err) => {
          if (this._inFlightCompute.get(key) === trackedPromise) {
            this._inFlightCompute.delete(key);
          }
          this._clearInFlightPriority('compute', key);
          this._recordPipelineError('compute', err, { key });
          if (
            generation === this._generation &&
            shaderPackGeneration === this._shaderPackGeneration
          ) {
            this._quarantineKey(key, 'compute', err);
          }
          throw err;
        }
      );

      this._inFlightCompute.set(key, trackedPromise);
      return trackedPromise;
    }

    async _prewarmPool(descriptors, options, cacheMap, inFlightMap, getterAsync, isRender) {
      const opts = options || {};
      const concurrency = validatePositiveInt(
        opts.concurrency !== undefined ? opts.concurrency : DEFAULT_PREWARM_CONCURRENCY,
        'concurrency'
      );
      const defaultPriority = extractPriorityOption(opts);

      const list = Array.from(descriptors || []);
      const pipelines = new Array(list.length);
      const itemPriorities = new Array(list.length);
      let hasExplicitPriority = defaultPriority !== null;

      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        const wrapped = isDescriptorWrapper(item);
        let p = null;
        if (wrapped) {
          if (item.priority !== undefined && item.priority !== null) {
            p = normalizePipelinePriority(item.priority);
          } else if (item.priorityTier !== undefined && item.priorityTier !== null) {
            p = normalizePipelinePriority(item.priorityTier);
          } else {
            p = extractPriorityOption(item.options);
          }
        }
        if (p === null) {
          p = defaultPriority;
        }
        if (p !== null) {
          hasExplicitPriority = true;
        }
        itemPriorities[i] = p;
      }

      const workOrder = new Array(list.length);
      for (let i = 0; i < list.length; i += 1) {
        workOrder[i] = i;
      }
      if (hasExplicitPriority && list.length > 1) {
        workOrder.sort((a, b) => {
          const rankA =
            itemPriorities[a] !== null
              ? PIPELINE_PRIORITY_RANKS[itemPriorities[a]]
              : PIPELINE_PRIORITY_RANKS.normal;
          const rankB =
            itemPriorities[b] !== null
              ? PIPELINE_PRIORITY_RANKS[itemPriorities[b]]
              : PIPELINE_PRIORITY_RANKS.normal;
          return rankB - rankA || a - b;
        });
      }

      let cursor = 0;
      let compiled = 0;
      let cached = 0;
      let deduped = 0;

      const itemKeys = new Array(list.length);
      for (let i = 0; i < list.length; i += 1) {
        const item = list[i];
        const wrapped = isDescriptorWrapper(item);
        const desc = wrapped ? item.descriptor : item;
        const keyHint = wrapped ? item.keyHint : undefined;
        try {
          if (keyHint !== undefined && keyHint !== null) {
            itemKeys[i] = this._resolveKey(null, keyHint);
          } else {
            const rawItemOpts = wrapped ? item.options : undefined;
            const normPreview = isRender
              ? this.normalizeRenderDescriptor(desc, rawItemOpts)
              : normalizeComputePipelineDescriptor(desc);
            itemKeys[i] = this._resolveKey(normPreview.descriptor, undefined);
          }
        } catch (_err) {
          itemKeys[i] = null;
        }
      }

      const activeQueueHandle = {
        kind: isRender ? 'render' : 'compute',
        promoteKey(targetKey) {
          for (let pos = cursor; pos < workOrder.length; pos += 1) {
            const origIdx = workOrder[pos];
            if (itemKeys[origIdx] === targetKey) {
              const prev = itemPriorities[origIdx];
              const alreadyCritical = prev === PIPELINE_PRIORITY_TIERS.CRITICAL;
              itemPriorities[origIdx] = PIPELINE_PRIORITY_TIERS.CRITICAL;
              if (pos > cursor) {
                workOrder.splice(pos, 1);
                workOrder.splice(cursor, 0, origIdx);
                return true;
              }
              return !alreadyCritical;
            }
          }
          return false;
        },
      };
      this._activePrewarmQueues.add(activeQueueHandle);

      const worker = async () => {
        while (cursor < workOrder.length) {
          const origIdx = workOrder[cursor++];
          const item = list[origIdx];
          const wrapped = isDescriptorWrapper(item);
          const desc = wrapped ? item.descriptor : item;
          const keyHint = wrapped ? item.keyHint : undefined;
          const rawItemOpts = wrapped ? item.options : undefined;
          const effectivePriority = itemPriorities[origIdx];
          const itemOpts =
            effectivePriority !== null
              ? rawItemOpts && typeof rawItemOpts === 'object'
                ? { ...rawItemOpts, priority: effectivePriority }
                : { priority: effectivePriority }
              : rawItemOpts;
          const norm = isRender
            ? this.normalizeRenderDescriptor(desc, itemOpts)
            : normalizeComputePipelineDescriptor(desc);
          const effectiveDesc = norm.descriptor;
          const key = this._resolveKey(effectiveDesc, keyHint);
          const wasCached = cacheMap.has(key);
          const wasInFlight = !wasCached && inFlightMap.has(key);
          const wasQuarantined =
            !wasCached && !wasInFlight && this._isQuarantinedKey(key, isRender ? 'render' : 'compute');
          const pipeline = await getterAsync.call(this, desc, key, itemOpts);
          if (pipeline && keyHint === undefined) {
            const entry = cacheMap.get(key);
            if (entry !== undefined) {
              if (isRender) {
                this._populateFastRenderEntry(desc, norm, key, undefined, itemOpts || null, entry);
              } else {
                this._populateFastComputeEntry(desc, norm, key, undefined, entry);
              }
            }
          }
          pipelines[origIdx] = pipeline;
          if (wasCached) {
            cached += 1;
          } else if (wasInFlight) {
            deduped += 1;
          } else if (!wasQuarantined && pipeline !== null && pipeline !== undefined) {
            compiled += 1;
          }
        }
      };

      const workerCount = Math.min(concurrency, list.length);
      const workers = [];
      for (let i = 0; i < workerCount; i += 1) {
        workers.push(worker());
      }
      try {
        await Promise.all(workers);
      } finally {
        this._activePrewarmQueues.delete(activeQueueHandle);
      }

      return {
        total: list.length,
        compiled,
        cached,
        deduped,
        pipelines,
      };
    }

    /**
     * Batch-prewarm an iterable of render pipeline descriptors with bounded concurrency.
     *
     * @param {Iterable<GPURenderPipelineDescriptor | { descriptor: GPURenderPipelineDescriptor, keyHint?: string | bigint | number | object, options?: object }>} descriptors
     * @param {{ concurrency?: number }} [options]
     * @returns {Promise<{ total: number, compiled: number, cached: number, deduped: number, pipelines: Array<GPURenderPipeline> }>}
     */
    prewarmRenderPipelines(descriptors, options = {}) {
      return this._prewarmPool(
        descriptors,
        options,
        this._renderPipelines,
        this._inFlightRender,
        this.getOrCreateRenderPipelineAsync,
        true
      );
    }

    /**
     * Batch-prewarm an iterable of compute pipeline descriptors with bounded concurrency.
     *
     * @param {Iterable<GPUComputePipelineDescriptor | { descriptor: GPUComputePipelineDescriptor, keyHint?: string | bigint | number | object }>} descriptors
     * @param {{ concurrency?: number }} [options]
     * @returns {Promise<{ total: number, compiled: number, cached: number, deduped: number, pipelines: Array<GPUComputePipeline> }>}
     */
    prewarmComputePipelines(descriptors, options = {}) {
      return this._prewarmPool(
        descriptors,
        options,
        this._computePipelines,
        this._inFlightCompute,
        this.getOrCreateComputePipelineAsync,
        false
      );
    }

    /**
     * Evict cached render pipelines, compute pipelines, and bind group layouts
     * whose last access timestamp is strictly older than `now() - ageMs`.
     *
     * @param {number} ageMs
     * @returns {number} Total number of evicted cache entries
     */
    evictOlderThan(ageMs) {
      if (typeof ageMs !== 'number' || !Number.isFinite(ageMs) || ageMs < 0) {
        throw new RangeError(`ageMs must be a finite non-negative number, got: ${ageMs}`);
      }
      const cutoff = this._now() - ageMs;
      let removed = 0;

      const prune = (map) => {
        for (const [key, entry] of map.entries()) {
          if (entry.lastAccessedAt < cutoff) {
            entry.alive = false;
            map.delete(key);
            removed += 1;
          }
        }
      };

      prune(this._renderPipelines);
      prune(this._computePipelines);
      prune(this._bindGroupLayouts);

      this._stats.evictions += removed;
      this._stats.reclaimedCount += removed;
      return removed;
    }

    /**
     * Bounded LRU reclaim: trim total cached render pipelines, compute pipelines, and bind group
     * layouts down to at most `limit` entries, evicting lowest-priority (`low` -> `normal` ->
     * `high` -> `critical`) least-recently-used entries first.
     *
     * @param {number} limit
     * @returns {number} Number of entries reclaimed
     */
    trimTo(limit) {
      const target = validateNonNegativeInt(limit, 'limit');
      const total = this._totalCachedEntries();
      if (total <= target) {
        return 0;
      }
      if (target === 0) {
        const maps = [this._renderPipelines, this._computePipelines, this._bindGroupLayouts];
        let removed = 0;
        for (let i = 0; i < maps.length; i += 1) {
          const map = maps[i];
          for (const entry of map.values()) {
            entry.alive = false;
            removed += 1;
          }
          map.clear();
        }
        this._hasNonDefaultPriority = false;
        this._stats.evictions += removed;
        this._stats.reclaimedCount += removed;
        return removed;
      }

      let removed = 0;
      while (this._totalCachedEntries() > target) {
        if (!this._evictOneAcrossAllMaps(null, undefined, PIPELINE_PRIORITY_RANKS.critical)) {
          break;
        }
        removed += 1;
      }
      return removed;
    }

    /**
     * Evict up to `count` (or all when `count` is omitted) cached `'low'`-priority entries in
     * least-recently-used order across render pipelines, compute pipelines, and bind group layouts
     * without evicting `'normal'`, `'high'`, or `'critical'` entries.
     *
     * @param {number} [count]
     * @returns {number} Number of low-priority entries reclaimed
     */
    evictLowPriority(count) {
      if (count === undefined) {
        if (!this._hasNonDefaultPriority) {
          return 0;
        }
        const maps = [this._renderPipelines, this._computePipelines, this._bindGroupLayouts];
        let removed = 0;
        for (let i = 0; i < maps.length; i += 1) {
          const map = maps[i];
          for (const [k, entry] of map.entries()) {
            if (entry.priorityRank === PIPELINE_PRIORITY_RANKS.low) {
              entry.alive = false;
              map.delete(k);
              removed += 1;
            }
          }
        }
        this._stats.evictions += removed;
        this._stats.reclaimedCount += removed;
        return removed;
      }

      const maxToEvict = validateNonNegativeInt(count, 'count');
      if (maxToEvict === 0 || !this._hasNonDefaultPriority) {
        return 0;
      }
      let removed = 0;
      while (removed < maxToEvict) {
        if (!this._evictOneAcrossAllMaps(null, undefined, PIPELINE_PRIORITY_RANKS.low)) {
          break;
        }
        removed += 1;
      }
      return removed;
    }

    _resetStatsCounters() {
      this._stats.renderHits = 0;
      this._stats.renderMisses = 0;
      this._stats.computeHits = 0;
      this._stats.computeMisses = 0;
      this._stats.layoutHits = 0;
      this._stats.layoutMisses = 0;
      this._stats.inlineSamplerLayouts = 0;
      this._stats.samplerHits = 0;
      this._stats.samplerMisses = 0;
      this._stats.fastLookupHits = 0;
      this._stats.fastLookupMisses = 0;
      this._stats.inFlightDedupes = 0;
      this._stats.evictions = 0;
      this._stats.reclaimedCount = 0;
      this._stats.compileCount = 0;
      this._stats.totalCompileMs = 0;
      this._stats.maxCompileMs = 0;
      this._stats.slowCompilesOverBudget = 0;
      this._stats.quarantineHits = 0;
      this._stats.negativeCacheEvictions = 0;
      this._stats.fallbackNormalizedCount = 0;
      this._stats.vertexLayoutNormalizedCount = 0;
      this._stats.layoutNormalizedCount = 0;
      this._stats.fallbackRecoveryCount = 0;
      this._stats.pipelineErrorCount = 0;
      this._stats.fallbackSamplersCreated = 0;
      this._stats.fallbackSamplerBindingsSynthesized = 0;
      this._stats.blendPipelineVariants = 0;
      this._stats.srgbPipelineVariants = 0;
      this._compileLatencyHead = 0;
      this._compileLatencyCount = 0;
      this._pipelineErrors.length = 0;
      this._pipelineErrorSeq = 0;
      this._quarantineHitsByReason = createEmptyQuarantineReasonCounts();
      this._quarantineFailuresByReason = createEmptyQuarantineReasonCounts();
      this._drawnFirstPromotions = 0;
    }

    /**
     * Reset all telemetry counters and error history without evicting cached pipelines,
     * bind group layouts, fallback samplers, or quarantined keys.
     */
    resetStats() {
      this._resetStatsCounters();
    }

    /**
     * Clear all cached render pipelines, compute pipelines, bind group layouts,
     * quarantined failed keys, and detach any in-flight compilations so stale results
     * cannot repopulate the cache.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    clear(options = {}) {
      this._generation += 1;
      this._renderPipelines.clear();
      this._computePipelines.clear();
      this._bindGroupLayouts.clear();
      this._fallbackPipelineLayouts.clear();
      this._defaultSamplers.clear();
      this._inFlightRender.clear();
      this._inFlightCompute.clear();
      this._inFlightPriorities.clear();
      this._quarantinedKeys.clear();
      this._hasNonDefaultPriority = false;
      this._invalidateFastLookups();

      if (options && options.resetStats) {
        this._resetStatsCounters();
      }
    }

    /**
     * Reset all cached pipelines, layouts, in-flight promises, and quarantined keys.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    reset(options = {}) {
      this.clear(options);
    }

    /**
     * Destroy cache state by clearing all cached pipelines, layouts, fast-path lookup tables,
     * in-flight promises, and quarantined keys so stale resources cannot be reused.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    destroy(options = {}) {
      this.clear(options);
    }

    /**
     * Handle WebGPU device loss by clearing all device-owned caches, in-flight entries,
     * and quarantined failed keys.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    onDeviceLost(options = {}) {
      this.clear(options);
    }

    /**
     * Swap the underlying `GPUDevice` and clear all device-owned caches and quarantined keys.
     *
     * @param {GPUDevice} device
     * @param {{ resetStats?: boolean, force?: boolean }} [options]
     * @returns {boolean} True when the device was replaced or cache cleared
     */
    setDevice(device, options = {}) {
      if (!device || typeof device !== 'object') {
        throw new TypeError('setDevice requires a valid GPUDevice object');
      }
      if (device === this._device && !(options && options.force)) {
        return false;
      }
      this._device = device;
      this.clear(options);
      return true;
    }

    /**
     * Return a snapshot of cache hit/miss, negative-cache, deduplication, eviction, count,
     * quarantine, normalization, error tracking, and compile-time telemetry.
     */
    getStats() {
      const renderHits = this._stats.renderHits;
      const renderMisses = this._stats.renderMisses;
      const computeHits = this._stats.computeHits;
      const computeMisses = this._stats.computeMisses;
      const layoutHits = this._stats.layoutHits;
      const layoutMisses = this._stats.layoutMisses;
      const samplerHits = this._stats.samplerHits;
      const samplerMisses = this._stats.samplerMisses;
      const fastLookupHits =
        typeof this._stats.fastLookupHits === 'number' &&
        Number.isFinite(this._stats.fastLookupHits) &&
        this._stats.fastLookupHits > 0
          ? this._stats.fastLookupHits
          : 0;
      const fastLookupMisses =
        typeof this._stats.fastLookupMisses === 'number' &&
        Number.isFinite(this._stats.fastLookupMisses) &&
        this._stats.fastLookupMisses > 0
          ? this._stats.fastLookupMisses
          : 0;
      const fastLookupHitRate = computeClampedHitRate(
        this._stats.fastLookupHits,
        this._stats.fastLookupMisses
      );
      const pipelineHits = renderHits + computeHits;
      const pipelineMisses = renderMisses + computeMisses;
      const totalHits = pipelineHits + layoutHits + samplerHits;
      const totalMisses = pipelineMisses + layoutMisses + samplerMisses;
      const compileCount = this._stats.compileCount;
      const totalCompileMs =
        Number.isFinite(this._stats.totalCompileMs) && this._stats.totalCompileMs > 0
          ? this._stats.totalCompileMs
          : 0;
      const avgCompileMs =
        compileCount > 0 && Number.isFinite(totalCompileMs / compileCount)
          ? totalCompileMs / compileCount
          : 0;
      const maxCompileMs =
        Number.isFinite(this._stats.maxCompileMs) && this._stats.maxCompileMs > 0
          ? this._stats.maxCompileMs
          : 0;
      const totalHitRate = computeClampedHitRate(totalHits, totalMisses);
      const compileLatency = this._computeCompileLatencyStats();
      const reclaimedCount =
        typeof this._stats.reclaimedCount === 'number' &&
        Number.isFinite(this._stats.reclaimedCount) &&
        this._stats.reclaimedCount > 0
          ? this._stats.reclaimedCount
          : 0;

      const stats = {
        hits: totalHits,
        misses: totalMisses,
        pipelineHits,
        pipelineMisses,
        fastLookupHits,
        fastLookupMisses,
        fastLookupHitRate,
        negativeHits: this._stats.quarantineHits,
        negativeCacheHits: this._stats.quarantineHits,
        negativeCacheSize: this._quarantinedKeys.size,
        negativeCacheEvictions: this._stats.negativeCacheEvictions,
        hitRate: totalHitRate,
        layoutDedupHits: layoutHits,
        inlineSamplerLayouts: this._stats.inlineSamplerLayouts,
        avgCompileMs,
        shaderPackGeneration: this._shaderPackGeneration,
        renderHits,
        renderMisses,
        computeHits,
        computeMisses,
        layoutHits,
        layoutMisses,
        samplerHits,
        samplerMisses,
        inFlightDedupes: this._stats.inFlightDedupes,
        evictions: this._stats.evictions,
        reclaimedCount,
        renderCount: this._renderPipelines.size,
        computeCount: this._computePipelines.size,
        layoutCount: this._bindGroupLayouts.size,
        fallbackSamplerCount: this._defaultSamplers.size,
        compileCount,
        totalCompileMs,
        maxCompileMs,
        p50Ms: compileLatency.p50Ms,
        p95Ms: compileLatency.p95Ms,
        maxMs: compileLatency.maxMs,
        slowCompilesOverBudget: compileLatency.slowCompilesOverBudget,
        compileLatency,
        quarantineHits: this._stats.quarantineHits,
        quarantinedKeys: this._quarantinedKeys.size,
        fallbackNormalizedCount: this._stats.fallbackNormalizedCount,
        vertexLayoutNormalizedCount: this._stats.vertexLayoutNormalizedCount,
        layoutNormalizedCount: this._stats.layoutNormalizedCount,
        fallbackRecoveryCount: this._stats.fallbackRecoveryCount,
        pipelineErrorCount: this._stats.pipelineErrorCount,
        pipelineErrorHistorySize: this._pipelineErrors.length,
        maxErrorHistory: this._maxErrorHistory,
        fallbackSamplersCreated: this._stats.fallbackSamplersCreated,
        fallbackSamplerBindingsSynthesized: this._stats.fallbackSamplerBindingsSynthesized,
        blendPipelineVariants: this._stats.blendPipelineVariants,
        srgbPipelineVariants: this._stats.srgbPipelineVariants,
        cacheBreakdown: {
          render: {
            hits: renderHits,
            misses: renderMisses,
            hitRate: computeClampedHitRate(renderHits, renderMisses),
          },
          compute: {
            hits: computeHits,
            misses: computeMisses,
            hitRate: computeClampedHitRate(computeHits, computeMisses),
          },
          layout: {
            hits: layoutHits,
            misses: layoutMisses,
            hitRate: computeClampedHitRate(layoutHits, layoutMisses),
          },
          sampler: {
            hits: samplerHits,
            misses: samplerMisses,
            hitRate: computeClampedHitRate(samplerHits, samplerMisses),
          },
          total: {
            hits: totalHits,
            misses: totalMisses,
            hitRate: totalHitRate,
          },
        },
      };
      // W379: Attach sanitized quarantine reason breakdown & draw promotion counters as
      // non-enumerable properties so legacy deepEqual snapshots keep their exact own-enumerable shape.
      Object.defineProperties(stats, {
        quarantinedByReason: {
          value: this._getQuarantinedByReason(),
          enumerable: false,
          configurable: true,
        },
        quarantineFailuresByReason: {
          value: { ...this._quarantineFailuresByReason },
          enumerable: false,
          configurable: true,
        },
        drawnFirstPromotions: {
          value: this._drawnFirstPromotions,
          enumerable: false,
          configurable: true,
        },
      });
      return stats;
    }
  }

  const exportedApi = {
    DEFAULT_MAX_RENDER_PIPELINES,
    DEFAULT_MAX_COMPUTE_PIPELINES,
    DEFAULT_MAX_BIND_GROUP_LAYOUTS,
    DEFAULT_MAX_FAST_LOOKUP_ENTRIES,
    DEFAULT_PREWARM_CONCURRENCY,
    DEFAULT_COMPILE_BUDGET_MS,
    DEFAULT_COMPILE_LATENCY_WINDOW_SIZE,
    PIPELINE_PRIORITY_TIERS,
    PIPELINE_PRIORITY_RANKS,
    PIPELINE_FAILURE_REASON,
    DEFAULT_MAX_ERROR_HISTORY,
    DEFAULT_MAX_QUARANTINED_KEYS,
    DEFAULT_MAX_NEGATIVE_CACHE_ENTRIES,
    MAX_PIPELINE_ERROR_MESSAGE_LENGTH,
    DEFAULT_MAX_VERTEX_BUFFERS,
    DEFAULT_MAX_VERTEX_ATTRIBUTES,
    DEFAULT_MAX_VERTEX_BUFFER_STRIDE,
    FRAME1_VERTEX_STRIDE_24B,
    FRAME1_CBV_MIN_BINDING_SIZE,
    SHADER_STAGE_BITS,
    VERTEX_FORMAT_BYTE_SIZES,
    FRAME1_VERTEX_LAYOUT_24B,
    FALLBACK_BIND_GROUP_ENTRIES,
    DEFAULT_SAMPLER_DESCRIPTORS,
    fnv1a64Hex,
    buildFastPipelineKey,
    computeFastDescriptorHash,
    hashDescriptorKey,
    registerShaderModuleId,
    normalizeVisibility,
    normalizeFallbackProfile,
    normalizePipelinePriority,
    normalizeDefaultSamplerKind,
    buildDefaultSamplerDescriptor,
    isSrgbColorFormat,
    getVertexFormatByteSize,
    buildVertexBufferLayoutForStride,
    normalizeVertexBufferLayouts,
    buildFallbackBindGroupLayoutDescriptor,
    normalizeRenderPipelineDescriptor,
    sanitizePipelineErrorMessage,
    classifyQuarantineReason,
    WebGpuPipelineCache,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6PipelineCache = exportedApi;
  } else if (root) {
    root.PlayGta6PipelineCache = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
