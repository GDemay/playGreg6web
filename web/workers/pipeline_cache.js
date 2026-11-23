'use strict';

// WebGPU RenderPipeline, ComputePipeline, and BindGroupLayout cache manager.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6PipelineCache)
// and Node.js (module.exports). Interfaces cleanly with GPUDevice, GPUShaderModule,
// WGPK 64-bit shader hashes (web/workers/wgsl_pack_loader.js), and S6 reflection metadata.
(function (root) {
  const DEFAULT_MAX_RENDER_PIPELINES = 2048;
  const DEFAULT_MAX_COMPUTE_PIPELINES = 1024;
  const DEFAULT_MAX_BIND_GROUP_LAYOUTS = 512;
  const DEFAULT_PREWARM_CONCURRENCY = 4;

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

  // 64-bit FNV-1a constants matching shaders/src/convert.rs & web/workers/wgsl_pack_loader.js.
  const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
  const FNV1A_64_PRIME = 0x00000100000001b3n;
  const U64_MASK = 0xffffffffffffffffn;
  const WGPK_LABEL_RE = /^wgpk:(?:vertex|fragment|compute):(0x[0-9a-f]{1,16})$/i;

  const textEncoder = new TextEncoder();

  // WeakMaps for stable identity of opaque WebGPU objects (GPUShaderModule, GPUBindGroupLayout, etc.)
  // and user-registered WGPK/numeric shader module IDs without leaking GC references.
  const registeredObjectIds = new WeakMap();
  const opaqueObjectIds = new WeakMap();
  let nextOpaqueObjectId = 1;

  function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function formatHashHex64(value) {
    return `0x${(value & U64_MASK).toString(16).padStart(16, '0')}`;
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
  function registerShaderModuleId(gpuObject, id) {
    if (!gpuObject || (typeof gpuObject !== 'object' && typeof gpuObject !== 'function')) {
      throw new TypeError('gpuObject must be an object');
    }
    const token = normalizeShaderId(id);
    registeredObjectIds.set(gpuObject, token);
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
        return `wgpk:${formatHashHex64(BigInt(m[1]))}`;
      }
    }
    return null;
  }

  function hasEnumerableDataKeys(val) {
    const keys = Object.keys(val);
    for (let i = 0; i < keys.length; i += 1) {
      const v = val[keys[i]];
      if (v !== undefined && typeof v !== 'function') {
        return true;
      }
    }
    return false;
  }

  function canonicalizeValue(val, keyName, ancestors) {
    if (val === null) return 'null';
    const t = typeof val;
    if (t === 'undefined') return 'undef';
    if (t === 'boolean') return val ? 'b:1' : 'b:0';
    if (t === 'number') {
      if (Number.isNaN(val)) return 'n:NaN';
      if (val === Infinity) return 'n:Inf';
      if (val === -Infinity) return 'n:-Inf';
      return Object.is(val, -0) ? 'n:0' : `n:${String(val)}`;
    }
    if (t === 'bigint') {
      return `bi:${val.toString(10)}`;
    }
    if (t === 'string') {
      return `s:${val.length}:${val}`;
    }
    if (t === 'symbol') {
      return `sym:${String(val.description || '')}`;
    }
    if (t === 'function') {
      return 'fn';
    }

    // Check registered GPU resource / shader module IDs first.
    const registered = registeredObjectIds.get(val);
    if (registered !== undefined) {
      return `reg:${registered}`;
    }

    // Handle native/opaque WebGPU objects and shader modules before walking keys.
    const isShaderOrLayoutSlot = keyName === 'module' || keyName === 'layout';
    const isGpuHandle = isNativeWebGpuObject(val) || isDuckTypedShaderModule(val);
    if (isGpuHandle || isShaderOrLayoutSlot) {
      const explicitId = extractExplicitShaderId(val);
      if (explicitId !== null) {
        return `gpu:${explicitId}`;
      }
      if (isGpuHandle || !hasEnumerableDataKeys(val)) {
        return `gpu-oid:${getOpaqueObjectId(val)}`;
      }
    }

    if (ancestors.has(val)) {
      throw new TypeError('Circular reference detected in WebGPU descriptor');
    }

    if (Array.isArray(val)) {
      ancestors.add(val);
      const parts = new Array(val.length);
      for (let i = 0; i < val.length; i += 1) {
        parts[i] = canonicalizeValue(val[i], null, ancestors);
      }
      ancestors.delete(val);
      return `a:${val.length}:[${parts.join(',')}]`;
    }

    if (ArrayBuffer.isView(val)) {
      const bytes = new Uint8Array(val.buffer, val.byteOffset, val.byteLength);
      let hex = '';
      for (let i = 0; i < bytes.length; i += 1) {
        hex += bytes[i].toString(16).padStart(2, '0');
      }
      const ctorName = val.constructor ? val.constructor.name : 'View';
      return `bin:${ctorName}:${bytes.byteLength}:${hex}`;
    }

    if (val instanceof ArrayBuffer || (typeof SharedArrayBuffer !== 'undefined' && val instanceof SharedArrayBuffer)) {
      const bytes = new Uint8Array(val);
      let hex = '';
      for (let i = 0; i < bytes.length; i += 1) {
        hex += bytes[i].toString(16).padStart(2, '0');
      }
      return `ab:${bytes.byteLength}:${hex}`;
    }

    if (val instanceof Set) {
      ancestors.add(val);
      const items = [];
      for (const item of val) {
        items.push(canonicalizeValue(item, null, ancestors));
      }
      ancestors.delete(val);
      items.sort();
      return `set:${items.length}:[${items.join(',')}]`;
    }

    if (val instanceof Map) {
      ancestors.add(val);
      const pairs = [];
      for (const [k, v] of val.entries()) {
        pairs.push(`${canonicalizeValue(k, null, ancestors)}=>${canonicalizeValue(v, null, ancestors)}`);
      }
      ancestors.delete(val);
      pairs.sort();
      return `map:${pairs.length}:{${pairs.join(',')}}`;
    }

    // Plain/structured object: sort keys lexicographically for insertion-order invariance.
    ancestors.add(val);
    const keys = Object.keys(val).sort();
    const fields = [];
    for (let i = 0; i < keys.length; i += 1) {
      const k = keys[i];
      const v = val[k];
      if (v === undefined || typeof v === 'function') {
        continue;
      }
      fields.push(`${k.length}:${k}=${canonicalizeValue(v, k, ancestors)}`);
    }
    ancestors.delete(val);
    return `o:${fields.length}:{${fields.join(',')}}`;
  }

  /**
   * Compute a deterministic canonical FNV-1a 64-bit hex hash (`0x` + 16 hex digits)
   * for any WebGPU descriptor or nested JS value, independent of object key insertion order.
   * Supports `BigInt`, nested arrays/objects, TypedArrays, `Map`/`Set`, opaque `GPUShaderModule`s,
   * and WGPK shader hashes.
   *
   * @param {unknown} obj
   * @returns {string}
   */
  function hashDescriptorKey(obj) {
    const canonical = canonicalizeValue(obj, null, new Set());
    const bytes = textEncoder.encode(canonical);
    let hash = FNV1A_64_OFFSET_BASIS;
    for (let i = 0; i < bytes.length; i += 1) {
      hash ^= BigInt(bytes[i]);
      hash = (hash * FNV1A_64_PRIME) & U64_MASK;
    }
    return formatHashHex64(hash);
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

  function normalizeReflectionBinding(raw, idx) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) {
      throw new TypeError(`Reflection binding at index ${idx} must be an object`);
    }
    if (!Number.isInteger(raw.binding) || raw.binding < 0) {
      throw new RangeError(`Reflection binding at index ${idx} must have a non-negative integer "binding"`);
    }

    const entry = {
      binding: raw.binding,
      visibility: normalizeVisibility(raw.visibility),
    };

    if (raw.buffer && typeof raw.buffer === 'object') {
      const b = {};
      if (raw.buffer.type !== undefined) b.type = raw.buffer.type;
      const hasDynamicOffset =
        raw.buffer.hasDynamicOffset !== undefined
          ? raw.buffer.hasDynamicOffset
          : raw.buffer.has_dynamic_offset;
      if (hasDynamicOffset !== undefined) b.hasDynamicOffset = Boolean(hasDynamicOffset);
      const minBindingSize =
        raw.buffer.minBindingSize !== undefined
          ? raw.buffer.minBindingSize
          : raw.buffer.min_binding_size;
      if (minBindingSize !== undefined) b.minBindingSize = Number(minBindingSize);
      entry.buffer = b;
    }

    if (raw.sampler && typeof raw.sampler === 'object') {
      const s = {};
      if (raw.sampler.type !== undefined) s.type = raw.sampler.type;
      entry.sampler = s;
    }

    if (raw.texture && typeof raw.texture === 'object') {
      const tx = {};
      const sampleType =
        raw.texture.sampleType !== undefined ? raw.texture.sampleType : raw.texture.sample_type;
      if (sampleType !== undefined) tx.sampleType = sampleType;
      const viewDimension =
        raw.texture.viewDimension !== undefined
          ? raw.texture.viewDimension
          : raw.texture.view_dimension;
      if (viewDimension !== undefined) tx.viewDimension = viewDimension;
      if (raw.texture.multisampled !== undefined) tx.multisampled = Boolean(raw.texture.multisampled);
      entry.texture = tx;
    }

    const rawStorageTex = raw.storageTexture !== undefined ? raw.storageTexture : raw.storage_texture;
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

    const rawExternalTex = raw.externalTexture !== undefined ? raw.externalTexture : raw.external_texture;
    if (rawExternalTex && typeof rawExternalTex === 'object') {
      entry.externalTexture = {};
    }

    return entry;
  }

  function validatePositiveInt(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got: ${value}`);
    }
    return value;
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
   * with async compilation in-flight promise deduplication, batch prewarm, and telemetry.
   */
  class WebGpuPipelineCache {
    /**
     * @param {{
     *   device: GPUDevice,
     *   maxRenderPipelines?: number,
     *   maxComputePipelines?: number,
     *   maxBindGroupLayouts?: number,
     *   now?: () => number,
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

      if (options.now !== undefined && typeof options.now !== 'function') {
        throw new TypeError('options.now must be a function when provided');
      }
      this._now = options.now || defaultNow;

      this._renderPipelines = new Map();
      this._computePipelines = new Map();
      this._bindGroupLayouts = new Map();
      this._inFlightRender = new Map();
      this._inFlightCompute = new Map();
      this._generation = 0;

      this._stats = {
        renderHits: 0,
        renderMisses: 0,
        computeHits: 0,
        computeMisses: 0,
        layoutHits: 0,
        layoutMisses: 0,
        inFlightDedupes: 0,
        evictions: 0,
        totalCompileMs: 0,
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

    _getLru(map, key) {
      const entry = map.get(key);
      if (entry === undefined) {
        return undefined;
      }
      entry.lastAccessedAt = this._now();
      // Refresh LRU recency by reinserting at the tail of the Map.
      map.delete(key);
      map.set(key, entry);
      return entry.value;
    }

    _setLru(map, key, value, compileMs, maxSize) {
      const ts = this._now();
      if (map.has(key)) {
        map.delete(key);
      }
      map.set(key, {
        value,
        createdAt: ts,
        lastAccessedAt: ts,
        compileMs,
      });
      while (map.size > maxSize) {
        const oldestKey = map.keys().next().value;
        map.delete(oldestKey);
        this._stats.evictions += 1;
      }
    }

    /**
     * Return a cached `GPUBindGroupLayout` or create and cache it via `device.createBindGroupLayout`.
     *
     * @param {GPUBindGroupLayoutDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {GPUBindGroupLayout}
     */
    getOrCreateBindGroupLayout(descriptor, keyHint) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateBindGroupLayout requires a descriptor object');
      }
      if (typeof this._device.createBindGroupLayout !== 'function') {
        throw new TypeError('device must provide a createBindGroupLayout(descriptor) method');
      }

      const key = this._resolveKey(descriptor, keyHint);
      const cached = this._getLru(this._bindGroupLayouts, key);
      if (cached !== undefined) {
        this._stats.layoutHits += 1;
        return cached;
      }

      this._stats.layoutMisses += 1;
      const t0 = this._now();
      const layout = this._device.createBindGroupLayout(descriptor);
      const compileMs = Math.max(0, this._now() - t0);
      this._stats.totalCompileMs += compileMs;

      if (layout && (typeof layout === 'object' || typeof layout === 'function')) {
        registeredObjectIds.set(layout, `bgl:${key}`);
      }
      this._setLru(this._bindGroupLayouts, key, layout, compileMs, this._maxBindGroupLayouts);
      return layout;
    }

    /**
     * Build and cache a `GPUBindGroupLayout` from compact S6 shader reflection JSON.
     * Accepts either `{ entries | bindings, label?, group? }` or an array of binding objects
     * `{ binding, visibility, buffer?, sampler?, texture?, storageTexture? }`.
     *
     * @param {object | Array<object>} reflectionGroup
     * @param {string | bigint | number | object} [keyHint]
     * @returns {GPUBindGroupLayout}
     */
    createBindGroupLayoutFromReflection(reflectionGroup, keyHint) {
      let rawEntries;
      let label;

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
      } else {
        throw new TypeError('reflectionGroup must be an array or object');
      }

      const seenBindings = new Set();
      const entries = new Array(rawEntries.length);
      for (let i = 0; i < rawEntries.length; i += 1) {
        const normalized = normalizeReflectionBinding(rawEntries[i], i);
        if (seenBindings.has(normalized.binding)) {
          throw new Error(`Duplicate reflection binding index ${normalized.binding}`);
        }
        seenBindings.add(normalized.binding);
        entries[i] = normalized;
      }

      // Sort by binding index so semantically identical reflection groups share one cache key.
      entries.sort((a, b) => a.binding - b.binding);

      const descriptor = label !== undefined ? { label, entries } : { entries };
      return this.getOrCreateBindGroupLayout(descriptor, keyHint);
    }

    /**
     * Synchronously return a cached `GPURenderPipeline` or compile via `device.createRenderPipeline`.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {GPURenderPipeline}
     */
    getOrCreateRenderPipeline(descriptor, keyHint) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateRenderPipeline requires a descriptor object');
      }
      if (typeof this._device.createRenderPipeline !== 'function') {
        throw new TypeError('device must provide a createRenderPipeline(descriptor) method');
      }

      const key = this._resolveKey(descriptor, keyHint);
      const cached = this._getLru(this._renderPipelines, key);
      if (cached !== undefined) {
        this._stats.renderHits += 1;
        return cached;
      }

      this._stats.renderMisses += 1;
      const t0 = this._now();
      const pipeline = this._device.createRenderPipeline(descriptor);
      const compileMs = Math.max(0, this._now() - t0);
      this._stats.totalCompileMs += compileMs;

      this._setLru(this._renderPipelines, key, pipeline, compileMs, this._maxRenderPipelines);
      return pipeline;
    }

    /**
     * Asynchronously return a cached `GPURenderPipeline`, deduplicating concurrent in-flight
     * compilations for the same key. Uses `device.createRenderPipelineAsync` when available,
     * falling back to synchronous `device.createRenderPipeline`.
     *
     * @param {GPURenderPipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {Promise<GPURenderPipeline>}
     */
    getOrCreateRenderPipelineAsync(descriptor, keyHint) {
      if (!descriptor || typeof descriptor !== 'object') {
        return Promise.reject(new TypeError('getOrCreateRenderPipelineAsync requires a descriptor object'));
      }

      let key;
      try {
        key = this._resolveKey(descriptor, keyHint);
      } catch (err) {
        return Promise.reject(err);
      }

      const cached = this._getLru(this._renderPipelines, key);
      if (cached !== undefined) {
        this._stats.renderHits += 1;
        return Promise.resolve(cached);
      }

      const existing = this._inFlightRender.get(key);
      if (existing !== undefined) {
        this._stats.inFlightDedupes += 1;
        return existing;
      }

      const hasAsync = typeof this._device.createRenderPipelineAsync === 'function';
      const hasSync = typeof this._device.createRenderPipeline === 'function';
      if (!hasAsync && !hasSync) {
        return Promise.reject(
          new TypeError('device must provide createRenderPipelineAsync or createRenderPipeline')
        );
      }

      this._stats.renderMisses += 1;
      const t0 = this._now();

      if (!hasAsync) {
        try {
          const pipeline = this._device.createRenderPipeline(descriptor);
          const compileMs = Math.max(0, this._now() - t0);
          this._stats.totalCompileMs += compileMs;
          this._setLru(this._renderPipelines, key, pipeline, compileMs, this._maxRenderPipelines);
          return Promise.resolve(pipeline);
        } catch (err) {
          return Promise.reject(err);
        }
      }

      const generation = this._generation;
      let rawPromise;
      try {
        rawPromise = Promise.resolve(this._device.createRenderPipelineAsync(descriptor));
      } catch (err) {
        return Promise.reject(err);
      }

      const trackedPromise = rawPromise.then(
        (pipeline) => {
          if (this._inFlightRender.get(key) === trackedPromise) {
            this._inFlightRender.delete(key);
          }
          const compileMs = Math.max(0, this._now() - t0);
          this._stats.totalCompileMs += compileMs;
          if (generation === this._generation) {
            this._setLru(this._renderPipelines, key, pipeline, compileMs, this._maxRenderPipelines);
          }
          return pipeline;
        },
        (err) => {
          if (this._inFlightRender.get(key) === trackedPromise) {
            this._inFlightRender.delete(key);
          }
          throw err;
        }
      );

      this._inFlightRender.set(key, trackedPromise);
      return trackedPromise;
    }

    /**
     * Synchronously return a cached `GPUComputePipeline` or compile via `device.createComputePipeline`.
     *
     * @param {GPUComputePipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {GPUComputePipeline}
     */
    getOrCreateComputePipeline(descriptor, keyHint) {
      if (!descriptor || typeof descriptor !== 'object') {
        throw new TypeError('getOrCreateComputePipeline requires a descriptor object');
      }
      if (typeof this._device.createComputePipeline !== 'function') {
        throw new TypeError('device must provide a createComputePipeline(descriptor) method');
      }

      const key = this._resolveKey(descriptor, keyHint);
      const cached = this._getLru(this._computePipelines, key);
      if (cached !== undefined) {
        this._stats.computeHits += 1;
        return cached;
      }

      this._stats.computeMisses += 1;
      const t0 = this._now();
      const pipeline = this._device.createComputePipeline(descriptor);
      const compileMs = Math.max(0, this._now() - t0);
      this._stats.totalCompileMs += compileMs;

      this._setLru(this._computePipelines, key, pipeline, compileMs, this._maxComputePipelines);
      return pipeline;
    }

    /**
     * Asynchronously return a cached `GPUComputePipeline`, deduplicating concurrent in-flight
     * compilations for the same key. Uses `device.createComputePipelineAsync` when available,
     * falling back to synchronous `device.createComputePipeline`.
     *
     * @param {GPUComputePipelineDescriptor} descriptor
     * @param {string | bigint | number | object} [keyHint]
     * @returns {Promise<GPUComputePipeline>}
     */
    getOrCreateComputePipelineAsync(descriptor, keyHint) {
      if (!descriptor || typeof descriptor !== 'object') {
        return Promise.reject(new TypeError('getOrCreateComputePipelineAsync requires a descriptor object'));
      }

      let key;
      try {
        key = this._resolveKey(descriptor, keyHint);
      } catch (err) {
        return Promise.reject(err);
      }

      const cached = this._getLru(this._computePipelines, key);
      if (cached !== undefined) {
        this._stats.computeHits += 1;
        return Promise.resolve(cached);
      }

      const existing = this._inFlightCompute.get(key);
      if (existing !== undefined) {
        this._stats.inFlightDedupes += 1;
        return existing;
      }

      const hasAsync = typeof this._device.createComputePipelineAsync === 'function';
      const hasSync = typeof this._device.createComputePipeline === 'function';
      if (!hasAsync && !hasSync) {
        return Promise.reject(
          new TypeError('device must provide createComputePipelineAsync or createComputePipeline')
        );
      }

      this._stats.computeMisses += 1;
      const t0 = this._now();

      if (!hasAsync) {
        try {
          const pipeline = this._device.createComputePipeline(descriptor);
          const compileMs = Math.max(0, this._now() - t0);
          this._stats.totalCompileMs += compileMs;
          this._setLru(this._computePipelines, key, pipeline, compileMs, this._maxComputePipelines);
          return Promise.resolve(pipeline);
        } catch (err) {
          return Promise.reject(err);
        }
      }

      const generation = this._generation;
      let rawPromise;
      try {
        rawPromise = Promise.resolve(this._device.createComputePipelineAsync(descriptor));
      } catch (err) {
        return Promise.reject(err);
      }

      const trackedPromise = rawPromise.then(
        (pipeline) => {
          if (this._inFlightCompute.get(key) === trackedPromise) {
            this._inFlightCompute.delete(key);
          }
          const compileMs = Math.max(0, this._now() - t0);
          this._stats.totalCompileMs += compileMs;
          if (generation === this._generation) {
            this._setLru(this._computePipelines, key, pipeline, compileMs, this._maxComputePipelines);
          }
          return pipeline;
        },
        (err) => {
          if (this._inFlightCompute.get(key) === trackedPromise) {
            this._inFlightCompute.delete(key);
          }
          throw err;
        }
      );

      this._inFlightCompute.set(key, trackedPromise);
      return trackedPromise;
    }

    async _prewarmPool(descriptors, options, cacheMap, inFlightMap, getterAsync) {
      const opts = options || {};
      const concurrency = validatePositiveInt(
        opts.concurrency !== undefined ? opts.concurrency : DEFAULT_PREWARM_CONCURRENCY,
        'concurrency'
      );

      const list = Array.from(descriptors || []);
      const pipelines = new Array(list.length);
      let cursor = 0;
      let compiled = 0;
      let cached = 0;
      let deduped = 0;

      const worker = async () => {
        while (cursor < list.length) {
          const idx = cursor++;
          const item = list[idx];
          const desc = isDescriptorWrapper(item) ? item.descriptor : item;
          const keyHint = isDescriptorWrapper(item) ? item.keyHint : undefined;
          const key = this._resolveKey(desc, keyHint);
          const wasCached = cacheMap.has(key);
          const wasInFlight = !wasCached && inFlightMap.has(key);
          const pipeline = await getterAsync.call(this, desc, key);
          pipelines[idx] = pipeline;
          if (wasCached) {
            cached += 1;
          } else if (wasInFlight) {
            deduped += 1;
          } else {
            compiled += 1;
          }
        }
      };

      const workerCount = Math.min(concurrency, list.length);
      const workers = [];
      for (let i = 0; i < workerCount; i += 1) {
        workers.push(worker());
      }
      await Promise.all(workers);

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
     * @param {Iterable<GPURenderPipelineDescriptor | { descriptor: GPURenderPipelineDescriptor, keyHint?: string | bigint | number | object }>} descriptors
     * @param {{ concurrency?: number }} [options]
     * @returns {Promise<{ total: number, compiled: number, cached: number, deduped: number, pipelines: Array<GPURenderPipeline> }>}
     */
    prewarmRenderPipelines(descriptors, options = {}) {
      return this._prewarmPool(
        descriptors,
        options,
        this._renderPipelines,
        this._inFlightRender,
        this.getOrCreateRenderPipelineAsync
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
        this.getOrCreateComputePipelineAsync
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
            map.delete(key);
            removed += 1;
          }
        }
      };

      prune(this._renderPipelines);
      prune(this._computePipelines);
      prune(this._bindGroupLayouts);

      this._stats.evictions += removed;
      return removed;
    }

    /**
     * Clear all cached render pipelines, compute pipelines, bind group layouts,
     * and detach any in-flight compilations so stale results cannot repopulate the cache.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    clear(options = {}) {
      this._generation += 1;
      this._renderPipelines.clear();
      this._computePipelines.clear();
      this._bindGroupLayouts.clear();
      this._inFlightRender.clear();
      this._inFlightCompute.clear();

      if (options && options.resetStats) {
        this._stats.renderHits = 0;
        this._stats.renderMisses = 0;
        this._stats.computeHits = 0;
        this._stats.computeMisses = 0;
        this._stats.layoutHits = 0;
        this._stats.layoutMisses = 0;
        this._stats.inFlightDedupes = 0;
        this._stats.evictions = 0;
        this._stats.totalCompileMs = 0;
      }
    }

    /**
     * Return a snapshot of cache hit/miss, deduplication, eviction, count, and compile-time telemetry.
     */
    getStats() {
      return {
        renderHits: this._stats.renderHits,
        renderMisses: this._stats.renderMisses,
        computeHits: this._stats.computeHits,
        computeMisses: this._stats.computeMisses,
        layoutHits: this._stats.layoutHits,
        layoutMisses: this._stats.layoutMisses,
        inFlightDedupes: this._stats.inFlightDedupes,
        evictions: this._stats.evictions,
        renderCount: this._renderPipelines.size,
        computeCount: this._computePipelines.size,
        layoutCount: this._bindGroupLayouts.size,
        totalCompileMs: this._stats.totalCompileMs,
      };
    }
  }

  const exportedApi = {
    DEFAULT_MAX_RENDER_PIPELINES,
    DEFAULT_MAX_COMPUTE_PIPELINES,
    DEFAULT_MAX_BIND_GROUP_LAYOUTS,
    DEFAULT_PREWARM_CONCURRENCY,
    SHADER_STAGE_BITS,
    hashDescriptorKey,
    registerShaderModuleId,
    normalizeVisibility,
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
