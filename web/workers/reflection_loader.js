'use strict';

// S6 shader reflection manifest loader and WebGPU BindGroupLayout / PipelineLayout /
// VertexBufferLayout builder.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6ReflectionLoader)
// and Node.js (module.exports). Matches the exact JSON schema emitted by shaders/src/reflect.rs
// and integrates directly with WebGpuPipelineCache (web/workers/pipeline_cache.js).
(function (root) {
  /** Supported S6 layout-manifest.json schema_version (shaders/src/reflect.rs). */
  const LAYOUT_MANIFEST_SCHEMA_VERSION = 1;

  /** WebGPU baseline maxBindGroups (groups 0..3). */
  const MAX_WEBGPU_BIND_GROUPS = 4;

  /** Reflection sanity ceiling matching MAX_REFLECTED_GROUPS in shaders/src/reflect.rs. */
  const MAX_REFLECTED_GROUPS = 64;

  /** WebGPU GPUShaderStage bitmask constants with headless/Node fallbacks. */
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

  const VALID_VISIBILITY_MASK =
    SHADER_STAGE_BITS.VERTEX | SHADER_STAGE_BITS.FRAGMENT | SHADER_STAGE_BITS.COMPUTE;

  const STAGE_TO_VISIBILITY = Object.freeze({
    vertex: SHADER_STAGE_BITS.VERTEX,
    fragment: SHADER_STAGE_BITS.FRAGMENT,
    compute: SHADER_STAGE_BITS.COMPUTE,
  });

  const STAGE_ORDER = Object.freeze({
    vertex: 0,
    fragment: 1,
    compute: 2,
  });

  const VALID_BUFFER_TYPES = new Set(['uniform', 'storage', 'read-only-storage']);
  const VALID_SAMPLER_TYPES = new Set(['filtering', 'non-filtering', 'comparison']);
  const VALID_TEXTURE_SAMPLE_TYPES = new Set([
    'float',
    'unfilterable-float',
    'depth',
    'sint',
    'uint',
  ]);
  const VALID_TEXTURE_VIEW_DIMENSIONS = new Set([
    '1d',
    '2d',
    '2d-array',
    'cube',
    'cube-array',
    '3d',
  ]);
  const VALID_STORAGE_TEXTURE_ACCESS = new Set(['write-only', 'read-only', 'read-write']);
  const VALID_STORAGE_TEXTURE_VIEW_DIMENSIONS = new Set(['1d', '2d', '2d-array', '3d']);
  const VALID_RESOURCE_KINDS = new Set([
    'buffer',
    'sampler',
    'texture',
    'storageTexture',
    'externalTexture',
    'accelerationStructure',
  ]);
  const VALID_REGISTER_CLASSES = new Set(['b', 't', 'u', 's']);
  const VALID_STEP_MODES = new Set(['vertex', 'instance']);

  /** Complete WebGPU GPUVertexFormat byte sizes (covers all S6 io_type outputs + packed formats). */
  const VERTEX_FORMAT_BYTE_SIZES = Object.freeze({
    uint8: 1,
    uint8x2: 2,
    uint8x4: 4,
    sint8: 1,
    sint8x2: 2,
    sint8x4: 4,
    unorm8: 1,
    unorm8x2: 2,
    unorm8x4: 4,
    snorm8: 1,
    snorm8x2: 2,
    snorm8x4: 4,
    uint16: 2,
    uint16x2: 4,
    uint16x4: 8,
    sint16: 2,
    sint16x2: 4,
    sint16x4: 8,
    unorm16: 2,
    unorm16x2: 4,
    unorm16x4: 8,
    snorm16: 2,
    snorm16x2: 4,
    snorm16x4: 8,
    float16: 2,
    float16x2: 4,
    float16x4: 8,
    float32: 4,
    float32x2: 8,
    float32x3: 12,
    float32x4: 16,
    uint32: 4,
    uint32x2: 8,
    uint32x3: 12,
    uint32x4: 16,
    sint32: 4,
    sint32x2: 8,
    sint32x3: 12,
    sint32x4: 16,
    'unorm10-10-10-2': 4,
    'unorm8x4-bgra': 4,
  });

  // 64-bit FNV-1a constants matching shaders/src/convert.rs & shaders/src/reflect.rs.
  const FNV1A_64_OFFSET_BASIS = 0xcbf29ce484222325n;
  const FNV1A_64_PRIME = 0x00000100000001b3n;
  const U64_MAX = 0xffffffffffffffffn;
  const U32_MAX = 0xffffffff;

  const textEncoder = new TextEncoder();

  // ---------------------------------------------------------------------------
  // Typed Errors
  // ---------------------------------------------------------------------------

  class ReflectionError extends Error {
    constructor(message, code = 'REFLECTION_ERROR', details = undefined) {
      super(message);
      this.name = 'ReflectionError';
      this.code = code;
      if (details !== undefined) {
        this.details = details;
      }
    }
  }

  class ManifestValidationError extends ReflectionError {
    constructor(message, path = undefined) {
      super(path ? `${path}: ${message}` : message, 'MANIFEST_VALIDATION_ERROR', path ? { path } : undefined);
      this.name = 'ManifestValidationError';
      this.path = path || null;
    }
  }

  class BindingConflictError extends ReflectionError {
    constructor(group, binding, first, second, reason) {
      const firstLabel = describeBindingSide(first);
      const secondLabel = describeBindingSide(second);
      const detail = reason ? ` (${reason})` : '';
      super(
        `@group(${group}) @binding(${binding}) has incompatible layouts${detail}: ${firstLabel} vs ${secondLabel}`,
        'BINDING_CONFLICT',
        { group, binding, first, second, reason: reason || null }
      );
      this.name = 'BindingConflictError';
      this.group = group;
      this.binding = binding;
      this.first = first;
      this.second = second;
      this.reason = reason || null;
    }
  }

  class ShaderNotFoundError extends ReflectionError {
    constructor(query) {
      const display = typeof query === 'bigint' ? formatShaderHash(query) : String(query);
      super(`Shader reflection not found: ${display}`, 'SHADER_NOT_FOUND', { query });
      this.name = 'ShaderNotFoundError';
      this.query = query;
    }
  }

  class VertexLayoutError extends ReflectionError {
    constructor(message, details = undefined) {
      super(message, 'VERTEX_LAYOUT_ERROR', details);
      this.name = 'VertexLayoutError';
    }
  }

  function describeBindingSide(side) {
    if (!side) return '<unknown>';
    const stage = side.stage || 'unknown';
    const ep = side.entryPoint || '<unnamed>';
    const name = side.name || '<unnamed>';
    const layout = side.layoutJson || '<unknown>';
    return `${stage} '${ep}' (${name}) {${layout}}`;
  }

  // ---------------------------------------------------------------------------
  // Hash & Canonical Serialization Helpers (matching shaders/src/reflect.rs)
  // ---------------------------------------------------------------------------

  function isPlainObject(val) {
    return Boolean(val) && typeof val === 'object' && !Array.isArray(val);
  }

  function isNonNegativeInt(val) {
    return typeof val === 'number' && Number.isInteger(val) && val >= 0 && val <= U32_MAX;
  }

  /**
   * Compute 64-bit FNV-1a hash matching `fnv1a_64` in `shaders/src/convert.rs`.
   *
   * @param {string | Uint8Array | ArrayBuffer | ArrayBufferView} input
   * @returns {bigint}
   */
  function fnv1a64(input) {
    let bytes;
    if (typeof input === 'string') {
      bytes = textEncoder.encode(input);
    } else if (input instanceof Uint8Array) {
      bytes = input;
    } else if (ArrayBuffer.isView(input)) {
      bytes = new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    } else if (input instanceof ArrayBuffer) {
      bytes = new Uint8Array(input);
    } else {
      throw new TypeError('fnv1a64 input must be a string, Uint8Array, ArrayBuffer, or ArrayBufferView');
    }

    let hash = FNV1A_64_OFFSET_BASIS;
    for (let i = 0; i < bytes.length; i += 1) {
      hash ^= BigInt(bytes[i]);
      hash = (hash * FNV1A_64_PRIME) & U64_MAX;
    }
    return hash;
  }

  /**
   * Normalize a 64-bit shader/layout hash (`BigInt`, `0x...` hex string, bare hex string,
   * or non-negative safe integer) into an unsigned 64-bit `BigInt`.
   *
   * @param {bigint | string | number} hash
   * @returns {bigint}
   */
  function normalizeShaderHash(hash) {
    if (typeof hash === 'bigint') {
      if (hash < 0n || hash > U64_MAX) {
        throw new RangeError(`Shader hash out of u64 range: ${hash.toString()}`);
      }
      return hash;
    }

    if (typeof hash === 'number') {
      if (!Number.isInteger(hash) || !Number.isSafeInteger(hash) || hash < 0) {
        throw new RangeError(`Numeric shader hash must be a non-negative safe integer, got: ${hash}`);
      }
      return BigInt(hash);
    }

    if (typeof hash === 'string') {
      const trimmed = hash.trim();
      const rawHex =
        trimmed.startsWith('0x') || trimmed.startsWith('0X') ? trimmed.slice(2) : trimmed;
      if (!/^[0-9a-fA-F]{1,16}$/.test(rawHex)) {
        throw new RangeError(`Invalid hex shader hash: ${JSON.stringify(hash)}`);
      }
      return BigInt(`0x${rawHex}`);
    }

    throw new TypeError('Shader hash must be a BigInt, hex string, or non-negative safe integer');
  }

  /**
   * Format a 64-bit hash as canonical `"0x"` + 16 lowercase hex digits,
   * matching `format_shader_hash` in `shaders/src/convert.rs`.
   *
   * @param {bigint | string | number} hash
   * @returns {string}
   */
  function formatShaderHash(hash) {
    const value = normalizeShaderHash(hash);
    return `0x${value.toString(16).padStart(16, '0')}`;
  }

  function getLayoutMemberKind(entry) {
    let kind = null;
    let count = 0;
    if (entry.buffer !== undefined) {
      kind = 'buffer';
      count += 1;
    }
    if (entry.sampler !== undefined) {
      kind = 'sampler';
      count += 1;
    }
    if (entry.texture !== undefined) {
      kind = 'texture';
      count += 1;
    }
    if (entry.storageTexture !== undefined) {
      kind = 'storageTexture';
      count += 1;
    }
    if (entry.externalTexture !== undefined) {
      kind = 'externalTexture';
      count += 1;
    }
    return { kind, count };
  }

  function serializeLayoutMemberJson(entry) {
    if (entry.buffer) {
      const b = entry.buffer;
      return `"buffer":{"type":${JSON.stringify(b.type)},"hasDynamicOffset":${Boolean(
        b.hasDynamicOffset
      )},"minBindingSize":${b.minBindingSize || 0}}`;
    }
    if (entry.sampler) {
      return `"sampler":{"type":${JSON.stringify(entry.sampler.type)}}`;
    }
    if (entry.texture) {
      const t = entry.texture;
      return `"texture":{"sampleType":${JSON.stringify(t.sampleType)},"viewDimension":${JSON.stringify(
        t.viewDimension
      )},"multisampled":${Boolean(t.multisampled)}}`;
    }
    if (entry.storageTexture) {
      const st = entry.storageTexture;
      return `"storageTexture":{"access":${JSON.stringify(st.access)},"format":${JSON.stringify(
        st.format
      )},"viewDimension":${JSON.stringify(st.viewDimension)}}`;
    }
    if (entry.externalTexture) {
      return '"externalTexture":{}';
    }
    return '"unknown":{}';
  }

  function serializeLayoutEntryJson(entry) {
    return `{"binding":${entry.binding},"visibility":${entry.visibility},${serializeLayoutMemberJson(entry)}}`;
  }

  /**
   * Compute canonical S6 FNV-1a 64-bit content hash for a `GPUBindGroupLayoutDescriptor`,
   * matching `BindGroupLayoutDesc::content_hash()` in `shaders/src/reflect.rs`.
   *
   * @param {{ entries: Array<object> }} descriptor
   * @returns {string} Canonical `0x...` hash hex
   */
  function computeBindGroupLayoutHash(descriptor) {
    if (!descriptor || !Array.isArray(descriptor.entries)) {
      throw new TypeError('computeBindGroupLayoutHash requires a descriptor with an entries array');
    }
    const sorted = descriptor.entries.slice().sort((a, b) => a.binding - b.binding);
    const canonical = `[${sorted.map(serializeLayoutEntryJson).join(',')}]`;
    return formatShaderHash(fnv1a64(`bgl:${canonical}`));
  }

  /**
   * Compute canonical S6 FNV-1a 64-bit content hash for a pipeline layout's ordered BGL IDs,
   * matching `PipelineLayoutDesc::content_hash()` in `shaders/src/reflect.rs`.
   *
   * @param {Array<bigint | string | number>} bglIds
   * @returns {string} Canonical `0x...` hash hex
   */
  function computePipelineLayoutHash(bglIds) {
    if (!Array.isArray(bglIds)) {
      throw new TypeError('computePipelineLayoutHash requires an array of bind group layout IDs');
    }
    const formatted = bglIds.map((id) => formatShaderHash(id));
    return formatShaderHash(fnv1a64(`pl:[${formatted.join(',')}]`));
  }

  function cloneLayoutEntry(entry) {
    const out = {
      binding: entry.binding,
      visibility: entry.visibility,
    };
    if (entry.buffer) {
      out.buffer = {
        type: entry.buffer.type,
        hasDynamicOffset: Boolean(entry.buffer.hasDynamicOffset),
        minBindingSize: entry.buffer.minBindingSize,
      };
    } else if (entry.sampler) {
      out.sampler = { type: entry.sampler.type };
    } else if (entry.texture) {
      out.texture = {
        sampleType: entry.texture.sampleType,
        viewDimension: entry.texture.viewDimension,
        multisampled: Boolean(entry.texture.multisampled),
      };
    } else if (entry.storageTexture) {
      out.storageTexture = {
        access: entry.storageTexture.access,
        format: entry.storageTexture.format,
        viewDimension: entry.storageTexture.viewDimension,
      };
    } else if (entry.externalTexture) {
      out.externalTexture = {};
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // S6 Reflection Manifest Parsing & Validation
  // ---------------------------------------------------------------------------

  function validateHashKey(rawKey, path) {
    try {
      const hash = normalizeShaderHash(rawKey);
      return { hash, hashHex: formatShaderHash(hash) };
    } catch (err) {
      throw new ManifestValidationError(
        `invalid 64-bit hex hash ${JSON.stringify(rawKey)}: ${err.message}`,
        path
      );
    }
  }

  function validateStageName(rawStage, path) {
    if (typeof rawStage !== 'string') {
      throw new ManifestValidationError('stage must be a string ("vertex", "fragment", or "compute")', path);
    }
    const lower = rawStage.trim().toLowerCase();
    if (!Object.prototype.hasOwnProperty.call(STAGE_TO_VISIBILITY, lower)) {
      throw new ManifestValidationError(
        `unsupported shader stage ${JSON.stringify(rawStage)} (expected "vertex", "fragment", or "compute")`,
        path
      );
    }
    return lower;
  }

  function normalizeLayoutEntry(raw, path, fallbackVisibility) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('layout entry must be an object', path);
    }
    if (!isNonNegativeInt(raw.binding)) {
      throw new ManifestValidationError(`invalid binding index ${JSON.stringify(raw.binding)}`, `${path}.binding`);
    }

    let visibility = raw.visibility;
    if (visibility === undefined && fallbackVisibility !== undefined) {
      visibility = fallbackVisibility;
    }
    if (
      typeof visibility !== 'number' ||
      !Number.isInteger(visibility) ||
      visibility < 1 ||
      (visibility & ~VALID_VISIBILITY_MASK) !== 0
    ) {
      throw new ManifestValidationError(
        `invalid visibility bitmask ${JSON.stringify(raw.visibility)} (expected integer bitmask in 1..${VALID_VISIBILITY_MASK})`,
        `${path}.visibility`
      );
    }

    // Accept camelCase (exact S6 JSON) and snake_case aliases if passed from manual objects.
    const rawBuffer = raw.buffer;
    const rawSampler = raw.sampler;
    const rawTexture = raw.texture;
    const rawStorageTexture = raw.storageTexture !== undefined ? raw.storageTexture : raw.storage_texture;
    const rawExternalTexture = raw.externalTexture !== undefined ? raw.externalTexture : raw.external_texture;

    const presentMembers = [
      rawBuffer !== undefined ? 'buffer' : null,
      rawSampler !== undefined ? 'sampler' : null,
      rawTexture !== undefined ? 'texture' : null,
      rawStorageTexture !== undefined ? 'storageTexture' : null,
      rawExternalTexture !== undefined ? 'externalTexture' : null,
    ].filter(Boolean);

    if (presentMembers.length !== 1) {
      throw new ManifestValidationError(
        `layout entry must define exactly one resource member (buffer, sampler, texture, storageTexture, or externalTexture), found [${presentMembers.join(', ')}]`,
        path
      );
    }

    const entry = {
      binding: raw.binding,
      visibility,
    };

    const member = presentMembers[0];
    if (member === 'buffer') {
      if (!isPlainObject(rawBuffer)) {
        throw new ManifestValidationError('buffer layout must be an object', `${path}.buffer`);
      }
      const type = rawBuffer.type !== undefined ? rawBuffer.type : 'uniform';
      if (typeof type !== 'string' || !VALID_BUFFER_TYPES.has(type)) {
        throw new ManifestValidationError(
          `invalid buffer type ${JSON.stringify(rawBuffer.type)}`,
          `${path}.buffer.type`
        );
      }
      const rawDyn =
        rawBuffer.hasDynamicOffset !== undefined
          ? rawBuffer.hasDynamicOffset
          : rawBuffer.has_dynamic_offset;
      if (rawDyn !== undefined && typeof rawDyn !== 'boolean') {
        throw new ManifestValidationError('hasDynamicOffset must be a boolean', `${path}.buffer.hasDynamicOffset`);
      }
      const rawMinSize =
        rawBuffer.minBindingSize !== undefined
          ? rawBuffer.minBindingSize
          : rawBuffer.min_binding_size;
      let minBindingSize = 0;
      if (rawMinSize !== undefined) {
        if (typeof rawMinSize !== 'number' || !Number.isInteger(rawMinSize) || rawMinSize < 0) {
          throw new ManifestValidationError(
            `minBindingSize must be a non-negative integer, got ${JSON.stringify(rawMinSize)}`,
            `${path}.buffer.minBindingSize`
          );
        }
        minBindingSize = rawMinSize;
      }
      entry.buffer = {
        type,
        hasDynamicOffset: Boolean(rawDyn),
        minBindingSize,
      };
    } else if (member === 'sampler') {
      if (!isPlainObject(rawSampler)) {
        throw new ManifestValidationError('sampler layout must be an object', `${path}.sampler`);
      }
      const type = rawSampler.type !== undefined ? rawSampler.type : 'filtering';
      if (typeof type !== 'string' || !VALID_SAMPLER_TYPES.has(type)) {
        throw new ManifestValidationError(
          `invalid sampler type ${JSON.stringify(rawSampler.type)}`,
          `${path}.sampler.type`
        );
      }
      entry.sampler = { type };
    } else if (member === 'texture') {
      if (!isPlainObject(rawTexture)) {
        throw new ManifestValidationError('texture layout must be an object', `${path}.texture`);
      }
      const sampleType =
        rawTexture.sampleType !== undefined
          ? rawTexture.sampleType
          : rawTexture.sample_type !== undefined
            ? rawTexture.sample_type
            : 'float';
      if (typeof sampleType !== 'string' || !VALID_TEXTURE_SAMPLE_TYPES.has(sampleType)) {
        throw new ManifestValidationError(
          `invalid texture sampleType ${JSON.stringify(sampleType)}`,
          `${path}.texture.sampleType`
        );
      }
      const viewDimension =
        rawTexture.viewDimension !== undefined
          ? rawTexture.viewDimension
          : rawTexture.view_dimension !== undefined
            ? rawTexture.view_dimension
            : '2d';
      if (typeof viewDimension !== 'string' || !VALID_TEXTURE_VIEW_DIMENSIONS.has(viewDimension)) {
        throw new ManifestValidationError(
          `invalid texture viewDimension ${JSON.stringify(viewDimension)}`,
          `${path}.texture.viewDimension`
        );
      }
      if (rawTexture.multisampled !== undefined && typeof rawTexture.multisampled !== 'boolean') {
        throw new ManifestValidationError('multisampled must be a boolean', `${path}.texture.multisampled`);
      }
      const multisampled = Boolean(rawTexture.multisampled);
      if (multisampled && viewDimension !== '2d') {
        throw new ManifestValidationError(
          `multisampled texture must have viewDimension "2d", got ${JSON.stringify(viewDimension)}`,
          `${path}.texture`
        );
      }
      entry.texture = { sampleType, viewDimension, multisampled };
    } else if (member === 'storageTexture') {
      if (!isPlainObject(rawStorageTexture)) {
        throw new ManifestValidationError('storageTexture layout must be an object', `${path}.storageTexture`);
      }
      const access = rawStorageTexture.access !== undefined ? rawStorageTexture.access : 'write-only';
      if (typeof access !== 'string' || !VALID_STORAGE_TEXTURE_ACCESS.has(access)) {
        throw new ManifestValidationError(
          `invalid storageTexture access ${JSON.stringify(access)}`,
          `${path}.storageTexture.access`
        );
      }
      const format = rawStorageTexture.format;
      if (typeof format !== 'string' || format.length === 0) {
        throw new ManifestValidationError(
          'storageTexture format must be a non-empty string',
          `${path}.storageTexture.format`
        );
      }
      const viewDimension =
        rawStorageTexture.viewDimension !== undefined
          ? rawStorageTexture.viewDimension
          : rawStorageTexture.view_dimension !== undefined
            ? rawStorageTexture.view_dimension
            : '2d';
      if (
        typeof viewDimension !== 'string' ||
        !VALID_STORAGE_TEXTURE_VIEW_DIMENSIONS.has(viewDimension)
      ) {
        throw new ManifestValidationError(
          `invalid storageTexture viewDimension ${JSON.stringify(viewDimension)}`,
          `${path}.storageTexture.viewDimension`
        );
      }
      entry.storageTexture = { access, format, viewDimension };
    } else if (member === 'externalTexture') {
      if (!isPlainObject(rawExternalTexture)) {
        throw new ManifestValidationError('externalTexture layout must be an object', `${path}.externalTexture`);
      }
      entry.externalTexture = {};
    }

    return entry;
  }

  function parseRegisterConvention(raw) {
    if (raw === undefined || raw === null) {
      return { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } };
    }
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('register_convention must be an object', 'register_convention');
    }
    const source = raw.source !== undefined ? raw.source : 'dxc-spirv';
    if (typeof source !== 'string' || source.length === 0) {
      throw new ManifestValidationError('source must be a non-empty string', 'register_convention.source');
    }
    const shiftsRaw = raw.shifts !== undefined ? raw.shifts : {};
    if (!isPlainObject(shiftsRaw)) {
      throw new ManifestValidationError('shifts must be an object', 'register_convention.shifts');
    }
    const shifts = { b: 0, t: 0, u: 0, s: 0 };
    for (const key of ['b', 't', 'u', 's']) {
      if (shiftsRaw[key] !== undefined) {
        if (!isNonNegativeInt(shiftsRaw[key])) {
          throw new ManifestValidationError(
            `shift must be a non-negative uint32 integer, got ${JSON.stringify(shiftsRaw[key])}`,
            `register_convention.shifts.${key}`
          );
        }
        shifts[key] = shiftsRaw[key];
      }
    }
    return { source, shifts };
  }

  function parseBindingRecord(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('binding record must be an object', path);
    }
    if (!isNonNegativeInt(raw.group)) {
      throw new ManifestValidationError(`invalid group ${JSON.stringify(raw.group)}`, `${path}.group`);
    }
    if (!isNonNegativeInt(raw.binding)) {
      throw new ManifestValidationError(`invalid binding ${JSON.stringify(raw.binding)}`, `${path}.binding`);
    }
    if (raw.name !== undefined && raw.name !== null && typeof raw.name !== 'string') {
      throw new ManifestValidationError('name must be a string or null', `${path}.name`);
    }
    const resource = raw.resource;
    if (typeof resource !== 'string' || !VALID_RESOURCE_KINDS.has(resource)) {
      throw new ManifestValidationError(
        `invalid resource kind ${JSON.stringify(resource)}`,
        `${path}.resource`
      );
    }
    const inLayout = raw.in_layout !== undefined ? raw.in_layout : true;
    if (typeof inLayout !== 'boolean') {
      throw new ManifestValidationError('in_layout must be a boolean', `${path}.in_layout`);
    }

    let register = null;
    if (raw.register !== undefined && raw.register !== null) {
      if (!isPlainObject(raw.register)) {
        throw new ManifestValidationError('register must be an object or null', `${path}.register`);
      }
      if (typeof raw.register.class !== 'string' || !VALID_REGISTER_CLASSES.has(raw.register.class)) {
        throw new ManifestValidationError(
          `invalid register class ${JSON.stringify(raw.register.class)}`,
          `${path}.register.class`
        );
      }
      if (!isNonNegativeInt(raw.register.index) || !isNonNegativeInt(raw.register.space)) {
        throw new ManifestValidationError(
          'register index and space must be non-negative integers',
          `${path}.register`
        );
      }
      register = {
        class: raw.register.class,
        index: raw.register.index,
        space: raw.register.space,
      };
    }

    let spirv = null;
    if (raw.spirv !== undefined && raw.spirv !== null) {
      if (!isPlainObject(raw.spirv)) {
        throw new ManifestValidationError('spirv must be an object or null', `${path}.spirv`);
      }
      if (!isNonNegativeInt(raw.spirv.set) || !isNonNegativeInt(raw.spirv.binding)) {
        throw new ManifestValidationError('spirv set and binding must be non-negative integers', `${path}.spirv`);
      }
      spirv = { set: raw.spirv.set, binding: raw.spirv.binding };
    }

    const rootConstants = raw.root_constants !== undefined ? raw.root_constants : false;
    if (typeof rootConstants !== 'boolean') {
      throw new ManifestValidationError('root_constants must be a boolean', `${path}.root_constants`);
    }

    let bindingArray = null;
    if (raw.binding_array !== undefined && raw.binding_array !== null) {
      if (!isPlainObject(raw.binding_array)) {
        throw new ManifestValidationError('binding_array must be an object or null', `${path}.binding_array`);
      }
      const count = raw.binding_array.count;
      if (count !== null && !isNonNegativeInt(count)) {
        throw new ManifestValidationError(
          'binding_array.count must be a non-negative integer or null',
          `${path}.binding_array.count`
        );
      }
      bindingArray = { count };
    }

    return {
      group: raw.group,
      binding: raw.binding,
      name: raw.name !== undefined ? raw.name : null,
      resource,
      in_layout: inLayout,
      inLayout,
      register,
      spirv,
      root_constants: rootConstants,
      rootConstants,
      binding_array: bindingArray,
      bindingArray,
      layoutEntry: null,
    };
  }

  function parseVertexInput(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('vertex_input must be an object', path);
    }
    if (!isNonNegativeInt(raw.location)) {
      throw new ManifestValidationError(`invalid location ${JSON.stringify(raw.location)}`, `${path}.location`);
    }
    if (raw.type !== undefined && typeof raw.type !== 'string') {
      throw new ManifestValidationError('type must be a string', `${path}.type`);
    }
    if (raw.format !== undefined && raw.format !== null && typeof raw.format !== 'string') {
      throw new ManifestValidationError('format must be a string or null', `${path}.format`);
    }
    return {
      location: raw.location,
      type: raw.type !== undefined ? raw.type : 'unknown',
      format: raw.format !== undefined ? raw.format : null,
    };
  }

  function parseFragmentOutput(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('fragment_output must be an object', path);
    }
    if (!isNonNegativeInt(raw.location)) {
      throw new ManifestValidationError(`invalid location ${JSON.stringify(raw.location)}`, `${path}.location`);
    }
    if (typeof raw.type !== 'string' || raw.type.length === 0) {
      throw new ManifestValidationError('type must be a non-empty string', `${path}.type`);
    }
    return {
      location: raw.location,
      type: raw.type,
    };
  }

  function parseLoweringNeed(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('requires_lowering item must be an object', path);
    }
    if (typeof raw.kind !== 'string' || raw.kind.length === 0) {
      throw new ManifestValidationError('kind must be a non-empty string', `${path}.kind`);
    }
    if (raw.group !== undefined && raw.group !== null && !isNonNegativeInt(raw.group)) {
      throw new ManifestValidationError('group must be a non-negative integer or null', `${path}.group`);
    }
    if (raw.binding !== undefined && raw.binding !== null && !isNonNegativeInt(raw.binding)) {
      throw new ManifestValidationError('binding must be a non-negative integer or null', `${path}.binding`);
    }
    if (typeof raw.detail !== 'string') {
      throw new ManifestValidationError('detail must be a string', `${path}.detail`);
    }
    return {
      kind: raw.kind,
      group: raw.group !== undefined ? raw.group : null,
      binding: raw.binding !== undefined ? raw.binding : null,
      detail: raw.detail,
    };
  }

  function parseLimitViolation(raw, path) {
    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('limit_violation item must be an object', path);
    }
    if (typeof raw.limit !== 'string' || raw.limit.length === 0) {
      throw new ManifestValidationError('limit must be a non-empty string', `${path}.limit`);
    }
    let stage = null;
    if (raw.stage !== undefined && raw.stage !== null) {
      stage = validateStageName(raw.stage, `${path}.stage`);
    }
    if (typeof raw.count !== 'number' || !Number.isInteger(raw.count) || raw.count < 0) {
      throw new ManifestValidationError('count must be a non-negative integer', `${path}.count`);
    }
    if (typeof raw.max !== 'number' || !Number.isInteger(raw.max) || raw.max < 0) {
      throw new ManifestValidationError('max must be a non-negative integer', `${path}.max`);
    }
    return {
      limit: raw.limit,
      stage,
      count: raw.count,
      max: raw.max,
    };
  }

  /**
   * Parse and validate an S6 reflection manifest (`layout-manifest.json` emitted by
   * `shaders/src/reflect.rs`). Normalizes all 64-bit hashes (`0x...` hex and `BigInt`)
   * and resolves per-shader bind group layouts and pipeline references.
   *
   * @param {string | object} jsonOrObject
   * @returns {object} Normalized, indexed manifest object
   */
  function parseReflectionManifest(jsonOrObject) {
    let raw;
    if (typeof jsonOrObject === 'string') {
      try {
        raw = JSON.parse(jsonOrObject);
      } catch (err) {
        throw new ManifestValidationError(
          `malformed JSON: ${err instanceof Error ? err.message : String(err)}`
        );
      }
    } else {
      raw = jsonOrObject;
    }

    if (!isPlainObject(raw)) {
      throw new ManifestValidationError('manifest root must be a JSON object');
    }

    if (raw.schema_version !== LAYOUT_MANIFEST_SCHEMA_VERSION) {
      throw new ManifestValidationError(
        `unsupported schema_version ${JSON.stringify(raw.schema_version)} (expected ${LAYOUT_MANIFEST_SCHEMA_VERSION})`,
        'schema_version'
      );
    }

    const registerConvention = parseRegisterConvention(raw.register_convention);

    // 1. Parse bind_group_layouts table
    const bindGroupLayouts = new Map();
    const bindGroupLayoutsByBigInt = new Map();
    if (raw.bind_group_layouts !== undefined) {
      if (!isPlainObject(raw.bind_group_layouts)) {
        throw new ManifestValidationError('bind_group_layouts must be an object', 'bind_group_layouts');
      }
      for (const [rawId, rawBgl] of Object.entries(raw.bind_group_layouts)) {
        const bglPath = `bind_group_layouts[${JSON.stringify(rawId)}]`;
        const { hash: id, hashHex: idHex } = validateHashKey(rawId, bglPath);
        if (bindGroupLayoutsByBigInt.has(id)) {
          throw new ManifestValidationError(`duplicate bind group layout id ${idHex}`, bglPath);
        }
        if (!isPlainObject(rawBgl) || !Array.isArray(rawBgl.entries)) {
          throw new ManifestValidationError('bind group layout must be an object with an "entries" array', bglPath);
        }

        const seenBindings = new Set();
        const entries = new Array(rawBgl.entries.length);
        for (let i = 0; i < rawBgl.entries.length; i += 1) {
          const entry = normalizeLayoutEntry(rawBgl.entries[i], `${bglPath}.entries[${i}]`);
          if (seenBindings.has(entry.binding)) {
            throw new ManifestValidationError(
              `duplicate binding index ${entry.binding}`,
              `${bglPath}.entries[${i}]`
            );
          }
          seenBindings.add(entry.binding);
          entries[i] = entry;
        }
        entries.sort((a, b) => a.binding - b.binding);

        const normalizedBgl = { id, idHex, entries };
        bindGroupLayouts.set(idHex, normalizedBgl);
        bindGroupLayoutsByBigInt.set(id, normalizedBgl);
      }
    }

    // 2. Parse pipeline_layouts table
    const pipelineLayouts = new Map();
    const pipelineLayoutsByBigInt = new Map();
    if (raw.pipeline_layouts !== undefined) {
      if (!isPlainObject(raw.pipeline_layouts)) {
        throw new ManifestValidationError('pipeline_layouts must be an object', 'pipeline_layouts');
      }
      for (const [rawId, rawPl] of Object.entries(raw.pipeline_layouts)) {
        const plPath = `pipeline_layouts[${JSON.stringify(rawId)}]`;
        const { hash: id, hashHex: idHex } = validateHashKey(rawId, plPath);
        if (pipelineLayoutsByBigInt.has(id)) {
          throw new ManifestValidationError(`duplicate pipeline layout id ${idHex}`, plPath);
        }
        if (!isPlainObject(rawPl) || !Array.isArray(rawPl.bindGroupLayouts)) {
          throw new ManifestValidationError(
            'pipeline layout must be an object with a "bindGroupLayouts" array',
            plPath
          );
        }
        const bglIdHexes = new Array(rawPl.bindGroupLayouts.length);
        for (let g = 0; g < rawPl.bindGroupLayouts.length; g += 1) {
          const { hashHex: bglHex } = validateHashKey(
            rawPl.bindGroupLayouts[g],
            `${plPath}.bindGroupLayouts[${g}]`
          );
          if (!bindGroupLayouts.has(bglHex)) {
            throw new ManifestValidationError(
              `references unknown bind_group_layout ${bglHex}`,
              `${plPath}.bindGroupLayouts[${g}]`
            );
          }
          bglIdHexes[g] = bglHex;
        }
        const normalizedPl = { id, idHex, bindGroupLayouts: bglIdHexes };
        pipelineLayouts.set(idHex, normalizedPl);
        pipelineLayoutsByBigInt.set(id, normalizedPl);
      }
    }

    // 3. Parse shaders table (required)
    if (!isPlainObject(raw.shaders)) {
      throw new ManifestValidationError('shaders must be an object', 'shaders');
    }

    const shaders = new Map();
    const shadersByHash = new Map();
    const shadersByName = new Map();

    for (const [rawHash, rawShader] of Object.entries(raw.shaders)) {
      const shPath = `shaders[${JSON.stringify(rawHash)}]`;
      const { hash, hashHex } = validateHashKey(rawHash, shPath);
      if (shadersByHash.has(hash)) {
        throw new ManifestValidationError(`duplicate shader hash ${hashHex}`, shPath);
      }
      if (!isPlainObject(rawShader)) {
        throw new ManifestValidationError('shader record must be an object', shPath);
      }

      const stage = validateStageName(rawShader.stage, `${shPath}.stage`);
      const visibility = STAGE_TO_VISIBILITY[stage];

      const rawEntryPoint =
        rawShader.entry_point !== undefined ? rawShader.entry_point : rawShader.entryPoint;
      if (typeof rawEntryPoint !== 'string' || rawEntryPoint.length === 0) {
        throw new ManifestValidationError('entry_point must be a non-empty string', `${shPath}.entry_point`);
      }

      if (rawShader.name !== undefined && rawShader.name !== null && typeof rawShader.name !== 'string') {
        throw new ManifestValidationError('name must be a string', `${shPath}.name`);
      }
      const shaderName = typeof rawShader.name === 'string' && rawShader.name.length > 0 ? rawShader.name : null;

      // Validate pipeline_layout reference
      let pipelineLayoutIdHex = null;
      if (rawShader.pipeline_layout !== undefined && rawShader.pipeline_layout !== null) {
        const { hashHex: plHex } = validateHashKey(rawShader.pipeline_layout, `${shPath}.pipeline_layout`);
        if (raw.pipeline_layouts !== undefined && !pipelineLayouts.has(plHex)) {
          throw new ManifestValidationError(
            `references unknown pipeline_layout ${plHex}`,
            `${shPath}.pipeline_layout`
          );
        }
        pipelineLayoutIdHex = plHex;
      }

      // Resolve bind_group_layouts references
      const rawBglIds = rawShader.bind_group_layouts;
      const bindGroupLayoutIds = [];
      const resolvedGroupLayouts = [];

      if (rawBglIds !== undefined) {
        if (!Array.isArray(rawBglIds)) {
          throw new ManifestValidationError(
            'bind_group_layouts must be an array of layout ids',
            `${shPath}.bind_group_layouts`
          );
        }
        if (rawBglIds.length > MAX_REFLECTED_GROUPS) {
          throw new ManifestValidationError(
            `bind_group_layouts length ${rawBglIds.length} exceeds reflection limit ${MAX_REFLECTED_GROUPS}`,
            `${shPath}.bind_group_layouts`
          );
        }
        for (let g = 0; g < rawBglIds.length; g += 1) {
          const { hashHex: bglHex } = validateHashKey(
            rawBglIds[g],
            `${shPath}.bind_group_layouts[${g}]`
          );
          const bgl = bindGroupLayouts.get(bglHex);
          if (!bgl) {
            throw new ManifestValidationError(
              `references unknown bind_group_layout ${bglHex}`,
              `${shPath}.bind_group_layouts[${g}]`
            );
          }
          bindGroupLayoutIds.push(bglHex);
          resolvedGroupLayouts.push({
            group: g,
            id: bgl.id,
            idHex: bgl.idHex,
            entries: bgl.entries.map(cloneLayoutEntry),
          });
        }
      }

      // Parse layout_complete
      let layoutComplete = true;
      if (rawShader.layout_complete !== undefined) {
        if (typeof rawShader.layout_complete !== 'boolean') {
          throw new ManifestValidationError('layout_complete must be a boolean', `${shPath}.layout_complete`);
        }
        layoutComplete = rawShader.layout_complete;
      }

      // Parse workgroup_size
      let workgroupSize = null;
      if (rawShader.workgroup_size !== undefined && rawShader.workgroup_size !== null) {
        if (
          !Array.isArray(rawShader.workgroup_size) ||
          rawShader.workgroup_size.length !== 3 ||
          !rawShader.workgroup_size.every((n) => isNonNegativeInt(n) && n >= 1)
        ) {
          throw new ManifestValidationError(
            'workgroup_size must be null or a 3-element array of positive integers',
            `${shPath}.workgroup_size`
          );
        }
        workgroupSize = [
          rawShader.workgroup_size[0],
          rawShader.workgroup_size[1],
          rawShader.workgroup_size[2],
        ];
      }

      // Parse bindings metadata and cross-link with resolved bind_group_layouts entries
      const bindings = [];
      if (rawShader.bindings !== undefined) {
        if (!Array.isArray(rawShader.bindings)) {
          throw new ManifestValidationError('bindings must be an array', `${shPath}.bindings`);
        }
        for (let i = 0; i < rawShader.bindings.length; i += 1) {
          const rec = parseBindingRecord(rawShader.bindings[i], `${shPath}.bindings[${i}]`);
          const groupLayout = resolvedGroupLayouts[rec.group];
          if (groupLayout && rec.inLayout) {
            const matched = groupLayout.entries.find((e) => e.binding === rec.binding);
            if (matched) {
              rec.layoutEntry = cloneLayoutEntry(matched);
            }
          }
          bindings.push(rec);
        }
      }

      // Parse vertex_inputs
      const vertexInputs = [];
      const rawVertexInputs =
        rawShader.vertex_inputs !== undefined ? rawShader.vertex_inputs : rawShader.vertexInputs;
      if (rawVertexInputs !== undefined) {
        if (!Array.isArray(rawVertexInputs)) {
          throw new ManifestValidationError('vertex_inputs must be an array', `${shPath}.vertex_inputs`);
        }
        for (let i = 0; i < rawVertexInputs.length; i += 1) {
          vertexInputs.push(parseVertexInput(rawVertexInputs[i], `${shPath}.vertex_inputs[${i}]`));
        }
        vertexInputs.sort((a, b) => a.location - b.location);
      }

      // Parse fragment_outputs
      const fragmentOutputs = [];
      const rawFragmentOutputs =
        rawShader.fragment_outputs !== undefined
          ? rawShader.fragment_outputs
          : rawShader.fragmentOutputs;
      if (rawFragmentOutputs !== undefined) {
        if (!Array.isArray(rawFragmentOutputs)) {
          throw new ManifestValidationError(
            'fragment_outputs must be an array',
            `${shPath}.fragment_outputs`
          );
        }
        for (let i = 0; i < rawFragmentOutputs.length; i += 1) {
          fragmentOutputs.push(
            parseFragmentOutput(rawFragmentOutputs[i], `${shPath}.fragment_outputs[${i}]`)
          );
        }
        fragmentOutputs.sort((a, b) => a.location - b.location);
      }

      // Parse requires_lowering
      const requiresLowering = [];
      if (rawShader.requires_lowering !== undefined) {
        if (!Array.isArray(rawShader.requires_lowering)) {
          throw new ManifestValidationError(
            'requires_lowering must be an array',
            `${shPath}.requires_lowering`
          );
        }
        for (let i = 0; i < rawShader.requires_lowering.length; i += 1) {
          requiresLowering.push(
            parseLoweringNeed(rawShader.requires_lowering[i], `${shPath}.requires_lowering[${i}]`)
          );
        }
      }

      // Parse limit_violations
      const limitViolations = [];
      if (rawShader.limit_violations !== undefined) {
        if (!Array.isArray(rawShader.limit_violations)) {
          throw new ManifestValidationError(
            'limit_violations must be an array',
            `${shPath}.limit_violations`
          );
        }
        for (let i = 0; i < rawShader.limit_violations.length; i += 1) {
          limitViolations.push(
            parseLimitViolation(rawShader.limit_violations[i], `${shPath}.limit_violations[${i}]`)
          );
        }
      }

      const shaderRecord = {
        hash,
        hashHex,
        stage,
        visibility,
        entryPoint: rawEntryPoint,
        entry_point: rawEntryPoint,
        name: shaderName,
        pipelineLayoutId: pipelineLayoutIdHex,
        pipeline_layout: pipelineLayoutIdHex,
        bindGroupLayoutIds,
        bind_group_layouts: bindGroupLayoutIds.slice(),
        bindGroupLayouts: resolvedGroupLayouts,
        layoutComplete,
        layout_complete: layoutComplete,
        workgroupSize,
        workgroup_size: workgroupSize ? workgroupSize.slice() : null,
        bindings,
        vertexInputs,
        vertex_inputs: vertexInputs,
        fragmentOutputs,
        fragment_outputs: fragmentOutputs,
        requiresLowering,
        requires_lowering: requiresLowering,
        limitViolations,
        limit_violations: limitViolations,
      };

      shaders.set(hashHex, shaderRecord);
      shadersByHash.set(hash, shaderRecord);

      const indexName = (key) => {
        let list = shadersByName.get(key);
        if (!list) {
          list = [];
          shadersByName.set(key, list);
        }
        list.push(shaderRecord);
      };
      indexName(rawEntryPoint);
      indexName(`${stage}:${rawEntryPoint}`);
      if (shaderName && shaderName !== rawEntryPoint) {
        indexName(shaderName);
      }
    }

    // 4. Parse pre-merged pipelines table (optional)
    const pipelines = new Map();
    if (raw.pipelines !== undefined) {
      if (!isPlainObject(raw.pipelines)) {
        throw new ManifestValidationError('pipelines must be an object', 'pipelines');
      }
      for (const [rawKey, rawPipeline] of Object.entries(raw.pipelines)) {
        const pipePath = `pipelines[${JSON.stringify(rawKey)}]`;
        if (!isPlainObject(rawPipeline)) {
          throw new ManifestValidationError('pipeline record must be an object', pipePath);
        }
        if (!isPlainObject(rawPipeline.stages)) {
          throw new ManifestValidationError('stages must be an object', `${pipePath}.stages`);
        }

        const stages = {};
        const stageList = [];
        for (const [rawStage, rawShaderHash] of Object.entries(rawPipeline.stages)) {
          const stage = validateStageName(rawStage, `${pipePath}.stages.${rawStage}`);
          const { hash: shHash, hashHex: shHex } = validateHashKey(
            rawShaderHash,
            `${pipePath}.stages.${rawStage}`
          );
          if (!shadersByHash.has(shHash)) {
            throw new ManifestValidationError(
              `references unknown shader ${shHex}`,
              `${pipePath}.stages.${rawStage}`
            );
          }
          stages[stage] = shHex;
          stageList.push({ stage, hash: shHash, hashHex: shHex });
        }
        stageList.sort((a, b) => STAGE_ORDER[a.stage] - STAGE_ORDER[b.stage] || (a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : 0));
        const canonicalKey = stageList.map((s) => s.hashHex).join('+');

        let pipelineLayoutIdHex = null;
        if (rawPipeline.pipeline_layout !== undefined && rawPipeline.pipeline_layout !== null) {
          const { hashHex: plHex } = validateHashKey(
            rawPipeline.pipeline_layout,
            `${pipePath}.pipeline_layout`
          );
          if (raw.pipeline_layouts !== undefined && !pipelineLayouts.has(plHex)) {
            throw new ManifestValidationError(
              `references unknown pipeline_layout ${plHex}`,
              `${pipePath}.pipeline_layout`
            );
          }
          pipelineLayoutIdHex = plHex;
        }

        const bindGroupLayoutIds = [];
        const resolvedGroupLayouts = [];
        if (rawPipeline.bind_group_layouts !== undefined) {
          if (!Array.isArray(rawPipeline.bind_group_layouts)) {
            throw new ManifestValidationError(
              'bind_group_layouts must be an array',
              `${pipePath}.bind_group_layouts`
            );
          }
          for (let g = 0; g < rawPipeline.bind_group_layouts.length; g += 1) {
            const { hashHex: bglHex } = validateHashKey(
              rawPipeline.bind_group_layouts[g],
              `${pipePath}.bind_group_layouts[${g}]`
            );
            const bgl = bindGroupLayouts.get(bglHex);
            if (!bgl) {
              throw new ManifestValidationError(
                `references unknown bind_group_layout ${bglHex}`,
                `${pipePath}.bind_group_layouts[${g}]`
              );
            }
            bindGroupLayoutIds.push(bglHex);
            resolvedGroupLayouts.push({
              group: g,
              id: bgl.id,
              idHex: bgl.idHex,
              entries: bgl.entries.map(cloneLayoutEntry),
            });
          }
        }

        const layoutComplete =
          rawPipeline.layout_complete !== undefined ? rawPipeline.layout_complete : true;
        if (typeof layoutComplete !== 'boolean') {
          throw new ManifestValidationError(
            'layout_complete must be a boolean',
            `${pipePath}.layout_complete`
          );
        }

        const limitViolations = [];
        if (rawPipeline.limit_violations !== undefined) {
          if (!Array.isArray(rawPipeline.limit_violations)) {
            throw new ManifestValidationError(
              'limit_violations must be an array',
              `${pipePath}.limit_violations`
            );
          }
          for (let i = 0; i < rawPipeline.limit_violations.length; i += 1) {
            limitViolations.push(
              parseLimitViolation(rawPipeline.limit_violations[i], `${pipePath}.limit_violations[${i}]`)
            );
          }
        }

        pipelines.set(canonicalKey, {
          key: canonicalKey,
          rawKey,
          stages,
          stageList,
          pipelineLayoutId: pipelineLayoutIdHex,
          pipeline_layout: pipelineLayoutIdHex,
          bindGroupLayoutIds,
          bind_group_layouts: bindGroupLayoutIds.slice(),
          bindGroupLayouts: resolvedGroupLayouts,
          layoutComplete,
          layout_complete: layoutComplete,
          limitViolations,
          limit_violations: limitViolations,
        });
      }
    }

    return {
      schemaVersion: LAYOUT_MANIFEST_SCHEMA_VERSION,
      schema_version: LAYOUT_MANIFEST_SCHEMA_VERSION,
      registerConvention,
      register_convention: registerConvention,
      bindGroupLayouts,
      bindGroupLayoutsByBigInt,
      pipelineLayouts,
      pipelineLayoutsByBigInt,
      shaders,
      shadersByHash,
      shadersByName,
      pipelines,
    };
  }

  // ---------------------------------------------------------------------------
  // Multi-Stage BindGroupLayout Merging (matching shaders/src/reflect.rs)
  // ---------------------------------------------------------------------------

  /**
   * Merge two `GPUBindGroupLayoutEntry` layouts at the same `(group, binding)` following
   * the exact rules of `merge_layouts` in `shaders/src/reflect.rs`:
   * - Resource member kinds must match (`buffer`, `sampler`, `texture`, `storageTexture`, `externalTexture`).
   * - Buffers: `type` & `hasDynamicOffset` must match; `minBindingSize` takes `Math.max`.
   * - Samplers: `type` must match.
   * - Textures: `viewDimension` & `multisampled` must match; `"float"` wins over `"unfilterable-float"`.
   * - Storage textures: `access`, `format`, and `viewDimension` must match.
   *
   * Returns `{ merged, reason }` where `merged` is `null` on conflict.
   */
  function mergeLayoutEntries(existing, incoming) {
    const existingKind = getLayoutMemberKind(existing).kind;
    const incomingKind = getLayoutMemberKind(incoming).kind;

    if (existingKind !== incomingKind) {
      return {
        merged: null,
        reason: `resource kind mismatch: ${existingKind} vs ${incomingKind}`,
      };
    }

    const visibility = existing.visibility | incoming.visibility;

    if (existingKind === 'buffer') {
      const a = existing.buffer;
      const b = incoming.buffer;
      if (a.type !== b.type) {
        return {
          merged: null,
          reason: `buffer type mismatch: ${a.type} vs ${b.type}`,
        };
      }
      if (Boolean(a.hasDynamicOffset) !== Boolean(b.hasDynamicOffset)) {
        return {
          merged: null,
          reason: `buffer hasDynamicOffset mismatch: ${Boolean(a.hasDynamicOffset)} vs ${Boolean(b.hasDynamicOffset)}`,
        };
      }
      return {
        merged: {
          binding: existing.binding,
          visibility,
          buffer: {
            type: a.type,
            hasDynamicOffset: Boolean(a.hasDynamicOffset),
            minBindingSize: Math.max(a.minBindingSize || 0, b.minBindingSize || 0),
          },
        },
        reason: null,
      };
    }

    if (existingKind === 'sampler') {
      if (existing.sampler.type !== incoming.sampler.type) {
        return {
          merged: null,
          reason: `sampler type mismatch: ${existing.sampler.type} vs ${incoming.sampler.type}`,
        };
      }
      return {
        merged: {
          binding: existing.binding,
          visibility,
          sampler: { type: existing.sampler.type },
        },
        reason: null,
      };
    }

    if (existingKind === 'texture') {
      const a = existing.texture;
      const b = incoming.texture;
      if (a.viewDimension !== b.viewDimension) {
        return {
          merged: null,
          reason: `texture viewDimension mismatch: ${a.viewDimension} vs ${b.viewDimension}`,
        };
      }
      if (Boolean(a.multisampled) !== Boolean(b.multisampled)) {
        return {
          merged: null,
          reason: `texture multisampled mismatch: ${Boolean(a.multisampled)} vs ${Boolean(b.multisampled)}`,
        };
      }
      let sampleType = a.sampleType;
      if (a.sampleType !== b.sampleType) {
        const floatPair =
          (a.sampleType === 'float' && b.sampleType === 'unfilterable-float') ||
          (a.sampleType === 'unfilterable-float' && b.sampleType === 'float');
        if (!floatPair) {
          return {
            merged: null,
            reason: `texture sampleType mismatch: ${a.sampleType} vs ${b.sampleType}`,
          };
        }
        sampleType = 'float';
      }
      return {
        merged: {
          binding: existing.binding,
          visibility,
          texture: {
            sampleType,
            viewDimension: a.viewDimension,
            multisampled: Boolean(a.multisampled),
          },
        },
        reason: null,
      };
    }

    if (existingKind === 'storageTexture') {
      const a = existing.storageTexture;
      const b = incoming.storageTexture;
      if (a.access !== b.access || a.format !== b.format || a.viewDimension !== b.viewDimension) {
        return {
          merged: null,
          reason: `storageTexture mismatch: (${a.access}, ${a.format}, ${a.viewDimension}) vs (${b.access}, ${b.format}, ${b.viewDimension})`,
        };
      }
      return {
        merged: {
          binding: existing.binding,
          visibility,
          storageTexture: {
            access: a.access,
            format: a.format,
            viewDimension: a.viewDimension,
          },
        },
        reason: null,
      };
    }

    if (existingKind === 'externalTexture') {
      return {
        merged: {
          binding: existing.binding,
          visibility,
          externalTexture: {},
        },
        reason: null,
      };
    }

    return {
      merged: null,
      reason: `unsupported layout entry kind: ${existingKind}`,
    };
  }

  /**
   * Extract normalized `{ group, entry, side }` tuples from a shader reflection object.
   * Supports both parsed S6 `ShaderReflection` records (with `bindGroupLayouts`) and
   * standalone/inline reflection objects (with `bindings` containing WebGPU resource members).
   */
  function extractShaderLayoutSlots(reflection) {
    if (!isPlainObject(reflection)) {
      throw new TypeError('Shader reflection item must be an object');
    }

    const stage =
      typeof reflection.stage === 'string'
        ? validateStageName(reflection.stage, 'reflection.stage')
        : null;
    const defaultVisibility =
      typeof reflection.visibility === 'number'
        ? reflection.visibility
        : stage
          ? STAGE_TO_VISIBILITY[stage]
          : undefined;
    const entryPoint =
      reflection.entryPoint || reflection.entry_point || reflection.name || '<unnamed>';

    // Build optional name lookup from S6 `bindings` metadata table.
    const bindingNameBySlot = new Map();
    if (Array.isArray(reflection.bindings)) {
      for (let i = 0; i < reflection.bindings.length; i += 1) {
        const b = reflection.bindings[i];
        if (b && isNonNegativeInt(b.group) && isNonNegativeInt(b.binding) && typeof b.name === 'string') {
          bindingNameBySlot.set(`${b.group}:${b.binding}`, b.name);
        }
      }
    }

    const slots = [];

    // Primary path: resolved S6 `bindGroupLayouts` array (indexed by group 0..N-1).
    if (Array.isArray(reflection.bindGroupLayouts)) {
      for (let idx = 0; idx < reflection.bindGroupLayouts.length; idx += 1) {
        const bgl = reflection.bindGroupLayouts[idx];
        if (!bgl) continue;
        const group = isNonNegativeInt(bgl.group) ? bgl.group : idx;
        const rawEntries = Array.isArray(bgl.entries) ? bgl.entries : [];
        for (let j = 0; j < rawEntries.length; j += 1) {
          const entry = normalizeLayoutEntry(
            rawEntries[j],
            `reflection(${entryPoint}).bindGroupLayouts[${group}].entries[${j}]`,
            defaultVisibility
          );
          const name = bindingNameBySlot.get(`${group}:${entry.binding}`) || null;
          slots.push({
            group,
            entry,
            side: {
              stage: stage || 'unknown',
              entryPoint,
              name,
              layoutJson: serializeLayoutMemberJson(entry),
            },
          });
        }
      }
      return slots;
    }

    // Fallback path: inline `bindings` array where each element carries a WebGPU resource member.
    if (Array.isArray(reflection.bindings)) {
      for (let i = 0; i < reflection.bindings.length; i += 1) {
        const raw = reflection.bindings[i];
        if (!isPlainObject(raw)) {
          throw new ManifestValidationError('binding item must be an object', `bindings[${i}]`);
        }
        if (raw.in_layout === false || raw.inLayout === false) {
          continue;
        }
        const group = raw.group !== undefined ? raw.group : 0;
        if (!isNonNegativeInt(group)) {
          throw new ManifestValidationError(`invalid group ${JSON.stringify(group)}`, `bindings[${i}].group`);
        }
        const source = isPlainObject(raw.layoutEntry) ? raw.layoutEntry : raw;
        const entry = normalizeLayoutEntry(
          source,
          `reflection(${entryPoint}).bindings[${i}]`,
          defaultVisibility
        );
        slots.push({
          group,
          entry,
          side: {
            stage: stage || 'unknown',
            entryPoint,
            name: typeof raw.name === 'string' ? raw.name : null,
            layoutJson: serializeLayoutMemberJson(entry),
          },
        });
      }
      return slots;
    }

    throw new ManifestValidationError(
      `shader reflection '${entryPoint}' has neither bindGroupLayouts nor bindings`
    );
  }

  /**
   * Merge one or more shader reflection objects into WebGPU `GPUBindGroupLayoutDescriptor[]`,
   * ordered densely by group (`0..maxGroup`, within `0..maxBindGroups-1`) with entries
   * sorted ascending by `binding`.
   *
   * @param {Array<object> | object} reflections
   * @param {{ maxBindGroups?: number, requireLayoutComplete?: boolean }} [options]
   * @returns {Array<{ entries: Array<object> }>}
   */
  function buildBindGroupLayoutDescriptors(reflections, options = {}) {
    const list = Array.isArray(reflections)
      ? reflections
      : reflections && typeof reflections[Symbol.iterator] === 'function' && typeof reflections !== 'string'
        ? Array.from(reflections)
        : [reflections];

    const maxBindGroups =
      options.maxBindGroups !== undefined ? options.maxBindGroups : MAX_WEBGPU_BIND_GROUPS;
    if (!Number.isInteger(maxBindGroups) || maxBindGroups < 1 || maxBindGroups > MAX_REFLECTED_GROUPS) {
      throw new RangeError(
        `maxBindGroups must be an integer in 1..${MAX_REFLECTED_GROUPS}, got ${maxBindGroups}`
      );
    }

    const requireLayoutComplete = Boolean(options.requireLayoutComplete);
    const groups = new Map();
    let maxGroupSeen = -1;

    for (let i = 0; i < list.length; i += 1) {
      const refl = list[i];
      if (!isPlainObject(refl)) {
        throw new TypeError(`Shader reflection at index ${i} must be an object`);
      }
      if (requireLayoutComplete && (refl.layoutComplete === false || refl.layout_complete === false)) {
        const label = refl.hashHex || refl.entryPoint || refl.entry_point || `#${i}`;
        throw new ReflectionError(
          `Shader reflection ${label} has layout_complete=false and requires offline lowering before creating a WebGPU layout`,
          'INCOMPLETE_LAYOUT',
          { shader: label, requiresLowering: refl.requiresLowering || refl.requires_lowering || [] }
        );
      }

      const slots = extractShaderLayoutSlots(refl);
      for (let s = 0; s < slots.length; s += 1) {
        const { group, entry, side } = slots[s];
        if (group >= maxBindGroups) {
          throw new ReflectionError(
            `Bind group index ${group} exceeds WebGPU maxBindGroups limit (${maxBindGroups}, valid range 0..${maxBindGroups - 1})`,
            'GROUP_INDEX_OUT_OF_RANGE',
            { group, maxBindGroups }
          );
        }
        if (group > maxGroupSeen) {
          maxGroupSeen = group;
        }

        let groupMap = groups.get(group);
        if (!groupMap) {
          groupMap = new Map();
          groups.set(group, groupMap);
        }

        const existing = groupMap.get(entry.binding);
        if (!existing) {
          groupMap.set(entry.binding, {
            entry: cloneLayoutEntry(entry),
            side,
          });
        } else {
          const { merged, reason } = mergeLayoutEntries(existing.entry, entry);
          if (!merged) {
            throw new BindingConflictError(group, entry.binding, existing.side, side, reason);
          }
          existing.entry = merged;
        }
      }
    }

    if (maxGroupSeen < 0) {
      return [];
    }

    const descriptors = new Array(maxGroupSeen + 1);
    for (let g = 0; g <= maxGroupSeen; g += 1) {
      const groupMap = groups.get(g);
      if (!groupMap || groupMap.size === 0) {
        descriptors[g] = { entries: [] };
      } else {
        const entries = Array.from(groupMap.values())
          .map((item) => item.entry)
          .sort((a, b) => a.binding - b.binding);
        descriptors[g] = { entries };
      }
    }

    return descriptors;
  }

  // ---------------------------------------------------------------------------
  // Vertex Buffer Layout Builder
  // ---------------------------------------------------------------------------

  /**
   * Convert reflected vertex inputs (`[{ location, format, type? }]` or a vertex shader
   * reflection object) into WebGPU `GPUVertexBufferLayout[]`.
   *
   * @param {object | Array<object>} vertexReflection
   * @param {number | { stride?: number, stepMode?: 'vertex' | 'instance', formatOverrides?: Record<number, string>, attributeOffsets?: Record<number, number> }} [strideOrOptions]
   * @param {{ stepMode?: 'vertex' | 'instance', formatOverrides?: Record<number, string>, attributeOffsets?: Record<number, number> }} [extraOptions]
   * @returns {Array<{ arrayStride: number, stepMode: string, attributes: Array<{ shaderLocation: number, offset: number, format: string }> }>}
   */
  function buildVertexBufferLayouts(vertexReflection, strideOrOptions, extraOptions) {
    let rawInputs;
    if (Array.isArray(vertexReflection)) {
      rawInputs = vertexReflection;
    } else if (isPlainObject(vertexReflection)) {
      if (Array.isArray(vertexReflection.vertexInputs)) {
        rawInputs = vertexReflection.vertexInputs;
      } else if (Array.isArray(vertexReflection.vertex_inputs)) {
        rawInputs = vertexReflection.vertex_inputs;
      } else {
        throw new VertexLayoutError(
          'vertexReflection must be an array of vertex inputs or a shader reflection object with vertex_inputs / vertexInputs'
        );
      }
    } else {
      throw new VertexLayoutError(
        'vertexReflection must be a vertex shader reflection object or an array of vertex inputs'
      );
    }

    let explicitStride;
    let opts = {};
    if (typeof strideOrOptions === 'number') {
      explicitStride = strideOrOptions;
      opts = isPlainObject(extraOptions) ? extraOptions : {};
    } else if (isPlainObject(strideOrOptions)) {
      opts = strideOrOptions;
      explicitStride = opts.stride;
    } else if (strideOrOptions !== undefined && strideOrOptions !== null) {
      throw new VertexLayoutError(
        `stride must be a non-negative integer or options object, got ${typeof strideOrOptions}`
      );
    }

    const stepMode = opts.stepMode !== undefined ? opts.stepMode : 'vertex';
    if (typeof stepMode !== 'string' || !VALID_STEP_MODES.has(stepMode)) {
      throw new VertexLayoutError(
        `Invalid vertex stepMode ${JSON.stringify(stepMode)} (expected "vertex" or "instance")`
      );
    }

    const formatOverrides = isPlainObject(opts.formatOverrides) ? opts.formatOverrides : null;
    const attributeOffsets = isPlainObject(opts.attributeOffsets) ? opts.attributeOffsets : null;

    if (rawInputs.length === 0) {
      if (explicitStride !== undefined) {
        if (!Number.isInteger(explicitStride) || explicitStride < 0) {
          throw new VertexLayoutError(`stride must be a non-negative integer, got ${explicitStride}`);
        }
      }
      return [];
    }

    const normalizedInputs = rawInputs.map((item, idx) => {
      if (!isPlainObject(item)) {
        throw new VertexLayoutError(`Vertex input at index ${idx} must be an object`);
      }
      const location =
        item.location !== undefined ? item.location : item.shaderLocation;
      if (!isNonNegativeInt(location)) {
        throw new VertexLayoutError(
          `Vertex input at index ${idx} has invalid location ${JSON.stringify(location)}`
        );
      }
      const overrideFmt =
        formatOverrides && Object.prototype.hasOwnProperty.call(formatOverrides, location)
          ? formatOverrides[location]
          : undefined;
      const format = overrideFmt !== undefined ? overrideFmt : item.format;
      if (typeof format !== 'string' || !Object.prototype.hasOwnProperty.call(VERTEX_FORMAT_BYTE_SIZES, format)) {
        throw new VertexLayoutError(
          `Vertex input @location(${location}) has unsupported or missing GPUVertexFormat: ${JSON.stringify(format)}`,
          { location, format, type: item.type || null }
        );
      }

      const overrideOffset =
        attributeOffsets && Object.prototype.hasOwnProperty.call(attributeOffsets, location)
          ? attributeOffsets[location]
          : item.offset;
      if (overrideOffset !== undefined && !isNonNegativeInt(overrideOffset)) {
        throw new VertexLayoutError(
          `Vertex input @location(${location}) has invalid byte offset ${JSON.stringify(overrideOffset)}`
        );
      }

      return {
        location,
        format,
        byteSize: VERTEX_FORMAT_BYTE_SIZES[format],
        explicitOffset: overrideOffset,
      };
    });

    normalizedInputs.sort((a, b) => a.location - b.location);

    const seenLocations = new Set();
    const attributes = new Array(normalizedInputs.length);
    let runningOffset = 0;
    let maxEndByte = 0;

    for (let i = 0; i < normalizedInputs.length; i += 1) {
      const inp = normalizedInputs[i];
      if (seenLocations.has(inp.location)) {
        throw new VertexLayoutError(`Duplicate vertex input @location(${inp.location})`, {
          location: inp.location,
        });
      }
      seenLocations.add(inp.location);

      const offset = inp.explicitOffset !== undefined ? inp.explicitOffset : runningOffset;
      const endByte = offset + inp.byteSize;
      if (endByte > maxEndByte) {
        maxEndByte = endByte;
      }
      if (inp.explicitOffset === undefined) {
        runningOffset = endByte;
      } else if (endByte > runningOffset) {
        runningOffset = endByte;
      }

      attributes[i] = {
        shaderLocation: inp.location,
        offset,
        format: inp.format,
      };
    }

    let arrayStride = maxEndByte;
    if (explicitStride !== undefined) {
      if (!Number.isInteger(explicitStride) || explicitStride < 0) {
        throw new VertexLayoutError(`stride must be a non-negative integer, got ${explicitStride}`);
      }
      if (explicitStride < maxEndByte) {
        throw new VertexLayoutError(
          `Explicit vertex stride (${explicitStride}) is smaller than required attribute extent (${maxEndByte} bytes)`,
          { stride: explicitStride, requiredBytes: maxEndByte }
        );
      }
      arrayStride = explicitStride;
    }

    return [
      {
        arrayStride,
        stepMode,
        attributes,
      },
    ];
  }

  // ---------------------------------------------------------------------------
  // ReflectionRegistry
  // ---------------------------------------------------------------------------

  /**
   * Registry for S6 shader reflection manifests with multi-stage bind group layout merging,
   * WebGPU PipelineLayout construction via `WebGpuPipelineCache`, and vertex buffer layout generation.
   */
  class ReflectionRegistry {
    /**
     * @param {{ fetch?: typeof fetch, maxBindGroups?: number }} [options]
     */
    constructor(options = {}) {
      if (options !== undefined && !isPlainObject(options)) {
        throw new TypeError('ReflectionRegistry options must be an object');
      }
      this._fetch = typeof options.fetch === 'function' ? options.fetch : null;
      this._maxBindGroups =
        options.maxBindGroups !== undefined ? options.maxBindGroups : MAX_WEBGPU_BIND_GROUPS;
      if (
        !Number.isInteger(this._maxBindGroups) ||
        this._maxBindGroups < 1 ||
        this._maxBindGroups > MAX_REFLECTED_GROUPS
      ) {
        throw new RangeError(
          `maxBindGroups must be an integer in 1..${MAX_REFLECTED_GROUPS}, got ${this._maxBindGroups}`
        );
      }

      this._shadersByHash = new Map();
      this._shadersByHex = new Map();
      this._shadersByName = new Map();
      this._bindGroupLayouts = new Map();
      this._pipelineLayouts = new Map();
      this._pipelines = new Map();
      this._registerConvention = { source: 'dxc-spirv', shifts: { b: 0, t: 0, u: 0, s: 0 } };

      this._stats = {
        manifestsLoaded: 0,
        layoutsBuilt: 0,
        pipelineLayoutsBuilt: 0,
        vertexLayoutsBuilt: 0,
        lookupHits: 0,
        lookupMisses: 0,
        bindingConflicts: 0,
      };
    }

    /**
     * Parse and register an S6 reflection manifest (JSON string or parsed object).
     * Multiple manifests may be loaded; later manifests overwrite entries with the same hash.
     *
     * @param {string | object} jsonOrObject
     * @returns {object} Parsed manifest
     */
    loadManifest(jsonOrObject) {
      const parsed = parseReflectionManifest(jsonOrObject);
      this._registerConvention = parsed.registerConvention;

      for (const [idHex, bgl] of parsed.bindGroupLayouts.entries()) {
        this._bindGroupLayouts.set(idHex, bgl);
      }
      for (const [idHex, pl] of parsed.pipelineLayouts.entries()) {
        this._pipelineLayouts.set(idHex, pl);
      }
      for (const [hashHex, shader] of parsed.shaders.entries()) {
        this._shadersByHash.set(shader.hash, shader);
        this._shadersByHex.set(hashHex, shader);
        this._indexShaderName(shader.entryPoint, shader);
        this._indexShaderName(`${shader.stage}:${shader.entryPoint}`, shader);
        if (shader.name && shader.name !== shader.entryPoint) {
          this._indexShaderName(shader.name, shader);
        }
      }
      for (const [key, pipeline] of parsed.pipelines.entries()) {
        this._pipelines.set(key, pipeline);
      }

      this._stats.manifestsLoaded += 1;
      return parsed;
    }

    _indexShaderName(name, shader) {
      this._shadersByName.set(name, shader);
    }

    /**
     * Fetch an S6 `layout-manifest.json` from a URL and register it.
     *
     * @param {string} url
     * @param {typeof fetch} [fetchImpl]
     * @param {RequestInit} [fetchInit]
     * @returns {Promise<object>} Parsed manifest
     */
    async loadManifestUrl(url, fetchImpl, fetchInit) {
      if (typeof url !== 'string' || url.length === 0) {
        throw new TypeError('url must be a non-empty string');
      }
      const fetcher = fetchImpl || this._fetch || globalThis.fetch;
      if (typeof fetcher !== 'function') {
        throw new ReflectionError('No fetch implementation available for loadManifestUrl', 'NO_FETCH');
      }

      const response = await fetcher(url, fetchInit);
      if (!response || !response.ok) {
        const status = response ? response.status : 'unknown';
        const statusText = response && response.statusText ? ` ${response.statusText}` : '';
        throw new ReflectionError(
          `Failed to fetch reflection manifest from "${url}": HTTP ${status}${statusText}`,
          'FETCH_FAILED',
          { url, status }
        );
      }

      const payload =
        typeof response.text === 'function' ? await response.text() : await response.json();
      return this.loadManifest(payload);
    }

    /**
     * Look up a registered shader reflection by `BigInt` hash, safe integer, `0x...` hex string,
     * bare hex string, or shader `name` / `entry_point` / `"<stage>:<entry_point>"`.
     *
     * @param {bigint | string | number} hashOrName
     * @returns {object | undefined}
     */
    getShaderReflection(hashOrName) {
      let found;

      if (typeof hashOrName === 'bigint' || typeof hashOrName === 'number') {
        try {
          found = this._shadersByHash.get(normalizeShaderHash(hashOrName));
        } catch (_err) {
          found = undefined;
        }
      } else if (typeof hashOrName === 'string') {
        const trimmed = hashOrName.trim();
        if (trimmed.length > 0) {
          if (trimmed.startsWith('0x') || trimmed.startsWith('0X')) {
            const lower = trimmed.toLowerCase();
            found = this._shadersByHex.get(lower);
            if (!found) {
              try {
                found = this._shadersByHash.get(normalizeShaderHash(trimmed));
              } catch (_err) {
                found = undefined;
              }
            }
          } else {
            // Check name/entry_point first so names like "fade" are not mistaken for bare hex.
            found = this._shadersByName.get(trimmed);
            if (!found && /^[0-9a-fA-F]{1,16}$/.test(trimmed)) {
              try {
                found = this._shadersByHash.get(normalizeShaderHash(trimmed));
              } catch (_err) {
                found = undefined;
              }
            }
          }
        }
      }

      if (found !== undefined) {
        this._stats.lookupHits += 1;
      } else {
        this._stats.lookupMisses += 1;
      }
      return found;
    }

    /**
     * Look up a pre-merged pipeline reflection from `manifest.pipelines` by `"0x...+0x..."` key
     * or by an array of shader hashes (in any order).
     *
     * @param {string | Array<bigint | string | number>} hashesOrKey
     * @returns {object | undefined}
     */
    getPipelineReflection(hashesOrKey) {
      if (typeof hashesOrKey === 'string') {
        const direct = this._pipelines.get(hashesOrKey.trim());
        if (direct) return direct;
        const parts = hashesOrKey.split('+').map((s) => s.trim()).filter(Boolean);
        if (parts.length > 0) {
          return this.getPipelineReflection(parts);
        }
        return undefined;
      }
      if (!Array.isArray(hashesOrKey) || hashesOrKey.length === 0) {
        return undefined;
      }
      const members = [];
      for (let i = 0; i < hashesOrKey.length; i += 1) {
        const sh = this.getShaderReflection(hashesOrKey[i]);
        if (!sh) return undefined;
        members.push(sh);
      }
      members.sort(
        (a, b) =>
          STAGE_ORDER[a.stage] - STAGE_ORDER[b.stage] ||
          (a.hash < b.hash ? -1 : a.hash > b.hash ? 1 : 0)
      );
      const canonicalKey = members.map((m) => m.hashHex).join('+');
      return this._pipelines.get(canonicalKey);
    }

    /**
     * Resolve a list of shader hashes, names, or reflection objects into normalized
     * `ShaderReflection` objects.
     */
    _resolveReflections(shaderHashesOrReflections) {
      const items = Array.isArray(shaderHashesOrReflections)
        ? shaderHashesOrReflections
        : shaderHashesOrReflections &&
            typeof shaderHashesOrReflections[Symbol.iterator] === 'function' &&
            typeof shaderHashesOrReflections !== 'string'
          ? Array.from(shaderHashesOrReflections)
          : [shaderHashesOrReflections];

      const resolved = new Array(items.length);
      for (let i = 0; i < items.length; i += 1) {
        const item = items[i];
        if (isPlainObject(item)) {
          resolved[i] = item;
        } else if (typeof item === 'bigint' || typeof item === 'string' || typeof item === 'number') {
          const refl = this.getShaderReflection(item);
          if (!refl) {
            throw new ShaderNotFoundError(item);
          }
          resolved[i] = refl;
        } else {
          throw new TypeError(
            `Shader specifier at index ${i} must be a shader hash, name, or reflection object`
          );
        }
      }
      return resolved;
    }

    /**
     * Merge one or more shaders (by hash, name, or reflection object) into WebGPU
     * `GPUBindGroupLayoutDescriptor[]` sorted by group (`0..3`) and ascending `binding`,
     * OR-ing `visibility` bitmasks across stages and rejecting incompatible collisions.
     *
     * @param {Array<bigint | string | number | object> | bigint | string | number | object} shaderHashesOrReflections
     * @param {{ maxBindGroups?: number, requireLayoutComplete?: boolean }} [options]
     * @returns {Array<{ entries: Array<object> }>}
     */
    buildBindGroupLayoutDescriptors(shaderHashesOrReflections, options = {}) {
      const resolved = this._resolveReflections(shaderHashesOrReflections);
      const effectiveOptions = {
        maxBindGroups:
          options.maxBindGroups !== undefined ? options.maxBindGroups : this._maxBindGroups,
        requireLayoutComplete: Boolean(options.requireLayoutComplete),
      };

      try {
        const descriptors = buildBindGroupLayoutDescriptors(resolved, effectiveOptions);
        this._stats.layoutsBuilt += descriptors.length;
        return descriptors;
      } catch (err) {
        if (err instanceof BindingConflictError) {
          this._stats.bindingConflicts += 1;
        }
        throw err;
      }
    }

    /**
     * Build deduplicated `GPUBindGroupLayout`s via `pipelineCache.createBindGroupLayoutFromReflection(...)`
     * (falling back to `device.createBindGroupLayout(...)` when no `pipelineCache` is provided)
     * and construct a WebGPU `GPUPipelineLayout` via `device.createPipelineLayout({ bindGroupLayouts })`.
     *
     * @param {GPUDevice} device
     * @param {object | null | undefined} pipelineCache
     * @param {Array<bigint | string | number | object> | bigint | string | number | object} shaderHashesOrReflections
     * @param {{ label?: string, maxBindGroups?: number, requireLayoutComplete?: boolean }} [options]
     * @returns {GPUPipelineLayout}
     */
    buildPipelineLayout(device, pipelineCache, shaderHashesOrReflections, options = {}) {
      if (!device || typeof device.createPipelineLayout !== 'function') {
        throw new TypeError('device must provide a createPipelineLayout(descriptor) method');
      }

      const descriptors = this.buildBindGroupLayoutDescriptors(shaderHashesOrReflections, options);
      const bindGroupLayouts = new Array(descriptors.length);

      const useReflectionCache =
        Boolean(pipelineCache) && typeof pipelineCache.createBindGroupLayoutFromReflection === 'function';
      const useBasicCache =
        !useReflectionCache &&
        Boolean(pipelineCache) &&
        typeof pipelineCache.getOrCreateBindGroupLayout === 'function';

      if (!useReflectionCache && !useBasicCache && typeof device.createBindGroupLayout !== 'function') {
        throw new TypeError(
          'device must provide createBindGroupLayout(descriptor) when no pipelineCache is supplied'
        );
      }

      for (let g = 0; g < descriptors.length; g += 1) {
        const desc = descriptors[g];
        if (useReflectionCache) {
          bindGroupLayouts[g] = pipelineCache.createBindGroupLayoutFromReflection(desc);
        } else if (useBasicCache) {
          bindGroupLayouts[g] = pipelineCache.getOrCreateBindGroupLayout(desc);
        } else {
          bindGroupLayouts[g] = device.createBindGroupLayout(desc);
        }
      }

      const layoutDesc =
        options && typeof options.label === 'string' && options.label.length > 0
          ? { label: options.label, bindGroupLayouts }
          : { bindGroupLayouts };

      const pipelineLayout = device.createPipelineLayout(layoutDesc);
      this._stats.pipelineLayoutsBuilt += 1;
      return pipelineLayout;
    }

    /**
     * Build WebGPU `GPUVertexBufferLayout[]` from a vertex shader reflection (object, hash,
     * name, or raw `vertex_inputs` array) and optional explicit byte stride.
     *
     * @param {object | Array<object> | bigint | string | number} vertexReflection
     * @param {number | object} [strideOrOptions]
     * @param {object} [extraOptions]
     * @returns {Array<{ arrayStride: number, stepMode: string, attributes: Array<{ shaderLocation: number, offset: number, format: string }> }>}
     */
    buildVertexBufferLayouts(vertexReflection, strideOrOptions, extraOptions) {
      let target = vertexReflection;
      if (
        typeof vertexReflection === 'bigint' ||
        typeof vertexReflection === 'string' ||
        typeof vertexReflection === 'number'
      ) {
        const refl = this.getShaderReflection(vertexReflection);
        if (!refl) {
          throw new ShaderNotFoundError(vertexReflection);
        }
        target = refl;
      }

      const layouts = buildVertexBufferLayouts(target, strideOrOptions, extraOptions);
      this._stats.vertexLayoutsBuilt += 1;
      return layouts;
    }

    /**
     * Clear all loaded manifests and optionally reset telemetry counters.
     *
     * @param {{ resetStats?: boolean }} [options]
     */
    clear(options = {}) {
      this._shadersByHash.clear();
      this._shadersByHex.clear();
      this._shadersByName.clear();
      this._bindGroupLayouts.clear();
      this._pipelineLayouts.clear();
      this._pipelines.clear();

      if (options && options.resetStats) {
        this._stats.manifestsLoaded = 0;
        this._stats.layoutsBuilt = 0;
        this._stats.pipelineLayoutsBuilt = 0;
        this._stats.vertexLayoutsBuilt = 0;
        this._stats.lookupHits = 0;
        this._stats.lookupMisses = 0;
        this._stats.bindingConflicts = 0;
      }
    }

    /**
     * Return a telemetry snapshot for the registry.
     */
    getStats() {
      return {
        manifestsLoaded: this._stats.manifestsLoaded,
        shaderCount: this._shadersByHash.size,
        bindGroupLayoutCount: this._bindGroupLayouts.size,
        pipelineLayoutCount: this._pipelineLayouts.size,
        pipelineCount: this._pipelines.size,
        layoutsBuilt: this._stats.layoutsBuilt,
        pipelineLayoutsBuilt: this._stats.pipelineLayoutsBuilt,
        vertexLayoutsBuilt: this._stats.vertexLayoutsBuilt,
        lookupHits: this._stats.lookupHits,
        lookupMisses: this._stats.lookupMisses,
        bindingConflicts: this._stats.bindingConflicts,
      };
    }
  }

  const exportedApi = {
    LAYOUT_MANIFEST_SCHEMA_VERSION,
    MAX_WEBGPU_BIND_GROUPS,
    SHADER_STAGE_BITS,
    VERTEX_FORMAT_BYTE_SIZES,
    ReflectionError,
    ManifestValidationError,
    BindingConflictError,
    ShaderNotFoundError,
    VertexLayoutError,
    fnv1a64,
    normalizeShaderHash,
    formatShaderHash,
    computeBindGroupLayoutHash,
    computePipelineLayoutHash,
    parseReflectionManifest,
    buildBindGroupLayoutDescriptors,
    buildVertexBufferLayouts,
    ReflectionRegistry,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6ReflectionLoader = exportedApi;
  } else if (root) {
    root.PlayGta6ReflectionLoader = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
