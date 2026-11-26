'use strict';

// WebGPU indirect draw / dispatch argument helper for compute-driven meshlet culling.
// Zero external dependencies; UMD export for Web Workers (globalThis.PlayGta6IndirectDraw)
// and Node.js (module.exports). Pairs with the mesh-compute WGSL lowering (S5): a culling
// compute pass overwrites `instanceCount` (0 = culled) of per-draw records in place, and
// the render pass then consumes them through drawIndirect / drawIndexedIndirect.
//
// All records are little-endian u32 fields (baseVertex is i32), 4-byte aligned, as required
// by the WebGPU spec for indirect buffers.

(function (root) {
  const U32_MAX = 0xffffffff;
  const I32_MIN = -0x80000000;
  const I32_MAX = 0x7fffffff;
  const INDIRECT_OFFSET_ALIGNMENT = 4;
  const DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION = 65535;
  const DEFAULT_MAX_INDIRECT_BATCH_COUNT = 4096;
  const DEFAULT_CLAMP_WORKGROUP_SIZE = 64;
  const CLAMP_PARAMS_UNIFORM_BYTES = 48; // 12 x u32
  const CLAMP_PARAMS_SLOT_STRIDE = 256; // WebGPU minUniformBufferOffsetAlignment
  const DISPATCH_MESH_CLAMP_OUTPUT_BYTES = 28; // 12B dispatch + 16B drawIndirect pair (G66 #1537)
  const DEFAULT_CLAMP_PARAMS_SLOTS = 64;
  const DEFAULT_CLAMP_OUTPUT_BYTES = 65536; // 64 KiB managed clamped indirect ring

  function createEmptyStats() {
    return {
      indirectDrawsEncoded: 0,
      indirectIndexedDrawsEncoded: 0,
      indirectDispatchesEncoded: 0,
      boundsRejectedCount: 0,
      zeroBatchSkips: 0,
      batchLimitRejectedCount: 0,
      batchClampedCount: 0,
      indirectDraws: 0,
      indirectIndexedDraws: 0,
      indirectDispatches: 0,
      clampedOrSkippedInvalid: 0,
      maxMultiDrawCount: 0,
      validatedIndirectDraws: 0,
      skippedOutOfBoundsDraws: 0,
      skippedMisalignedDraws: 0,
      skippedZeroCountDraws: 0,
      batchesDispatched: 0,
      commandsValidated: 0,
      zeroCountElided: 0,
      outOfBoundsElided: 0,
      gpuClampPassesEncoded: 0,
      gpuClampCommandsClamped: 0,
      gpuClampPipelinesCreated: 0,
      gpuClampPipelineCacheHits: 0,
      gpuClampFirstInstanceZeroed: 0,
      gpuClampCountBuffersBound: 0,
      meshOrchestrationsEncoded: 0,
      meshIndirectDispatches: 0,
      meshZeroCountElided: 0,
    };
  }

  let moduleStats = createEmptyStats();

  function recordStat(field, delta = 1, instanceStats = null) {
    if (!delta) {
      return;
    }
    moduleStats[field] += delta;
    if (instanceStats) {
      instanceStats[field] += delta;
    }
  }

  function recordMaxStat(field, value, instanceStats = null) {
    if (value > moduleStats[field]) {
      moduleStats[field] = value;
    }
    if (instanceStats && value > instanceStats[field]) {
      instanceStats[field] = value;
    }
  }

  function recordEncodedKind(kind, count, instanceStats = null) {
    if (!count) {
      return;
    }
    moduleStats.validatedIndirectDraws += count;
    if (instanceStats) {
      instanceStats.validatedIndirectDraws += count;
    }
    if (kind === INDIRECT_KIND.DRAW) {
      moduleStats.indirectDrawsEncoded += count;
      moduleStats.indirectDraws += count;
      if (instanceStats) {
        instanceStats.indirectDrawsEncoded += count;
        instanceStats.indirectDraws += count;
      }
    } else if (kind === INDIRECT_KIND.DRAW_INDEXED) {
      moduleStats.indirectIndexedDrawsEncoded += count;
      moduleStats.indirectIndexedDraws += count;
      if (instanceStats) {
        instanceStats.indirectIndexedDrawsEncoded += count;
        instanceStats.indirectIndexedDraws += count;
      }
    } else if (kind === INDIRECT_KIND.DISPATCH) {
      moduleStats.indirectDispatchesEncoded += count;
      moduleStats.indirectDispatches += count;
      if (instanceStats) {
        instanceStats.indirectDispatchesEncoded += count;
        instanceStats.indirectDispatches += count;
      }
    }
  }

  function recordBoundsRejection(instanceStats = null) {
    moduleStats.boundsRejectedCount += 1;
    moduleStats.clampedOrSkippedInvalid += 1;
    if (instanceStats) {
      instanceStats.boundsRejectedCount += 1;
      instanceStats.clampedOrSkippedInvalid += 1;
    }
  }

  function isOutOfBoundsRangeStatus(status) {
    return (
      status === RANGE_OUT_OF_BOUNDS ||
      status === RANGE_INVALID_OFFSET ||
      status === RANGE_INVALID_OPT_BUFFER_SIZE ||
      status === RANGE_INVALID_BUFFER_SIZE
    );
  }

  function recordRangeRejection(status, instanceStats = null, skippedCount = 1, elided = false) {
    recordBoundsRejection(instanceStats);
    const delta = skippedCount > 0 ? skippedCount : 1;
    if (status === RANGE_MISALIGNED_OFFSET) {
      recordStat('skippedMisalignedDraws', delta, instanceStats);
    } else if (isOutOfBoundsRangeStatus(status)) {
      recordStat('skippedOutOfBoundsDraws', delta, instanceStats);
      if (elided) {
        recordStat('outOfBoundsElided', delta, instanceStats);
      }
    }
  }

  function recordBatchClamped(instanceStats = null) {
    moduleStats.batchClampedCount += 1;
    moduleStats.clampedOrSkippedInvalid += 1;
    if (instanceStats) {
      instanceStats.batchClampedCount += 1;
      instanceStats.clampedOrSkippedInvalid += 1;
    }
  }

  function recordBatchLimitRejected(instanceStats = null) {
    moduleStats.boundsRejectedCount += 1;
    moduleStats.batchLimitRejectedCount += 1;
    moduleStats.clampedOrSkippedInvalid += 1;
    if (instanceStats) {
      instanceStats.boundsRejectedCount += 1;
      instanceStats.batchLimitRejectedCount += 1;
      instanceStats.clampedOrSkippedInvalid += 1;
    }
  }

  /**
   * Return a sanitized numeric snapshot of module-level indirect execution telemetry.
   *
   * @returns {{
   *   indirectDrawsEncoded: number,
   *   indirectIndexedDrawsEncoded: number,
   *   indirectDispatchesEncoded: number,
   *   boundsRejectedCount: number,
   *   zeroBatchSkips: number,
   *   batchLimitRejectedCount: number,
   *   batchClampedCount: number,
   *   indirectDraws: number,
   *   indirectIndexedDraws: number,
   *   indirectDispatches: number,
   *   clampedOrSkippedInvalid: number,
   *   maxMultiDrawCount: number,
   *   validatedIndirectDraws: number,
   *   skippedOutOfBoundsDraws: number,
   *   skippedMisalignedDraws: number,
   *   skippedZeroCountDraws: number,
   *   batchesDispatched: number,
   *   commandsValidated: number,
   *   zeroCountElided: number,
   *   outOfBoundsElided: number
   * }}
   */
  function getStats() {
    return {
      indirectDrawsEncoded: moduleStats.indirectDrawsEncoded,
      indirectIndexedDrawsEncoded: moduleStats.indirectIndexedDrawsEncoded,
      indirectDispatchesEncoded: moduleStats.indirectDispatchesEncoded,
      boundsRejectedCount: moduleStats.boundsRejectedCount,
      zeroBatchSkips: moduleStats.zeroBatchSkips,
      batchLimitRejectedCount: moduleStats.batchLimitRejectedCount,
      batchClampedCount: moduleStats.batchClampedCount,
      indirectDraws: moduleStats.indirectDraws,
      indirectIndexedDraws: moduleStats.indirectIndexedDraws,
      indirectDispatches: moduleStats.indirectDispatches,
      clampedOrSkippedInvalid: moduleStats.clampedOrSkippedInvalid,
      maxMultiDrawCount: moduleStats.maxMultiDrawCount,
      validatedIndirectDraws: moduleStats.validatedIndirectDraws,
      skippedOutOfBoundsDraws: moduleStats.skippedOutOfBoundsDraws,
      skippedMisalignedDraws: moduleStats.skippedMisalignedDraws,
      skippedZeroCountDraws: moduleStats.skippedZeroCountDraws,
      batchesDispatched: moduleStats.batchesDispatched,
      commandsValidated: moduleStats.commandsValidated,
      zeroCountElided: moduleStats.zeroCountElided,
      outOfBoundsElided: moduleStats.outOfBoundsElided,
      gpuClampPassesEncoded: moduleStats.gpuClampPassesEncoded,
      gpuClampCommandsClamped: moduleStats.gpuClampCommandsClamped,
      gpuClampPipelinesCreated: moduleStats.gpuClampPipelinesCreated,
      gpuClampPipelineCacheHits: moduleStats.gpuClampPipelineCacheHits,
      gpuClampFirstInstanceZeroed: moduleStats.gpuClampFirstInstanceZeroed,
      gpuClampCountBuffersBound: moduleStats.gpuClampCountBuffersBound,
      meshOrchestrationsEncoded: moduleStats.meshOrchestrationsEncoded,
      meshIndirectDispatches: moduleStats.meshIndirectDispatches,
      meshZeroCountElided: moduleStats.meshZeroCountElided,
    };
  }

  /**
   * Reset module-level indirect execution telemetry counters to zero.
   */
  function resetStats() {
    moduleStats = createEmptyStats();
  }

  // WebGPU GPUBufferUsage bitmask values with headless/Node fallbacks.
  const BUFFER_USAGE = Object.freeze({
    COPY_SRC:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_SRC === 'number'
        ? GPUBufferUsage.COPY_SRC
        : 0x0004,
    COPY_DST:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.COPY_DST === 'number'
        ? GPUBufferUsage.COPY_DST
        : 0x0008,
    UNIFORM:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.UNIFORM === 'number'
        ? GPUBufferUsage.UNIFORM
        : 0x0040,
    STORAGE:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.STORAGE === 'number'
        ? GPUBufferUsage.STORAGE
        : 0x0080,
    INDIRECT:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.INDIRECT === 'number'
        ? GPUBufferUsage.INDIRECT
        : 0x0100,
  });

  const SHADER_STAGE_COMPUTE =
    typeof GPUShaderStage !== 'undefined' && typeof GPUShaderStage.COMPUTE === 'number'
      ? GPUShaderStage.COMPUTE
      : 0x0004;

  const INDIRECT_KIND = Object.freeze({
    DRAW: 'draw',
    DRAW_INDEXED: 'drawIndexed',
    DISPATCH: 'dispatch',
  });

  const INDIRECT_CLAMP_KIND = Object.freeze({
    DRAW: 'draw',
    DRAW_INDEXED: 'drawIndexed',
    DISPATCH: 'dispatch',
    DISPATCH_MESH: 'dispatchMesh',
  });

  // Field layouts: byte offsets in the record. `signed` marks i32 fields.
  const DRAW_INDIRECT_LAYOUT = Object.freeze({
    byteSize: 16,
    fields: Object.freeze({
      vertexCount: Object.freeze({ offset: 0, signed: false }),
      instanceCount: Object.freeze({ offset: 4, signed: false }),
      firstVertex: Object.freeze({ offset: 8, signed: false }),
      firstInstance: Object.freeze({ offset: 12, signed: false }),
    }),
  });
  const DRAW_INDEXED_INDIRECT_LAYOUT = Object.freeze({
    byteSize: 20,
    fields: Object.freeze({
      indexCount: Object.freeze({ offset: 0, signed: false }),
      instanceCount: Object.freeze({ offset: 4, signed: false }),
      firstIndex: Object.freeze({ offset: 8, signed: false }),
      baseVertex: Object.freeze({ offset: 12, signed: true }),
      firstInstance: Object.freeze({ offset: 16, signed: false }),
    }),
  });
  const DISPATCH_INDIRECT_LAYOUT = Object.freeze({
    byteSize: 12,
    fields: Object.freeze({
      x: Object.freeze({ offset: 0, signed: false }),
      y: Object.freeze({ offset: 4, signed: false }),
      z: Object.freeze({ offset: 8, signed: false }),
    }),
  });

  const LAYOUTS = Object.freeze({
    [INDIRECT_KIND.DRAW]: DRAW_INDIRECT_LAYOUT,
    [INDIRECT_KIND.DRAW_INDEXED]: DRAW_INDEXED_INDIRECT_LAYOUT,
    [INDIRECT_KIND.DISPATCH]: DISPATCH_INDIRECT_LAYOUT,
  });

  const DRAW_INDIRECT_BYTES = DRAW_INDIRECT_LAYOUT.byteSize;
  const DRAW_INDEXED_INDIRECT_BYTES = DRAW_INDEXED_INDIRECT_LAYOUT.byteSize;
  const DISPATCH_INDIRECT_BYTES = DISPATCH_INDIRECT_LAYOUT.byteSize;

  // WGSL structs that alias the indirect buffer in a culling compute shader.
  // A struct of N u32/i32 members has size 4N and alignment 4, so `array<T>` has stride 4N:
  // 16 for DrawIndirectArgs and 20 (NOT 16 or 32) for DrawIndexedIndirectArgs. Slot `i` lives
  // at byte offset `i * stride`, so `args[i].instance_count` is at `i * 20 + 4`.
  const WGSL_STRUCTS = Object.freeze({
    [INDIRECT_KIND.DRAW]: [
      'struct DrawIndirectArgs {',
      '  vertex_count: u32,',
      '  instance_count: u32,',
      '  first_vertex: u32,',
      '  first_instance: u32,',
      '}',
    ].join('\n'),
    [INDIRECT_KIND.DRAW_INDEXED]: [
      'struct DrawIndexedIndirectArgs {',
      '  index_count: u32,',
      '  instance_count: u32,',
      '  first_index: u32,',
      '  base_vertex: i32,',
      '  first_instance: u32,',
      '}',
    ].join('\n'),
    [INDIRECT_KIND.DISPATCH]: [
      'struct DispatchIndirectArgs {',
      '  x: u32,',
      '  y: u32,',
      '  z: u32,',
      '}',
    ].join('\n'),
  });

  /**
   * Error raised for arguments that WebGPU would reject (or that cannot be encoded).
   * `code` is a stable machine-readable identifier.
   */
  class IndirectDrawError extends Error {
    /**
     * @param {string} code
     * @param {string} message
     */
    constructor(code, message) {
      super(message);
      this.name = 'IndirectDrawError';
      this.code = code;
    }
  }

  function assertNonNegativeInteger(value, name) {
    if (!Number.isInteger(value) || value < 0) {
      throw new RangeError(`${name} must be a non-negative integer, got ${value}`);
    }
  }

  function assertPositiveInteger(value, name) {
    if (!Number.isInteger(value) || value < 1) {
      throw new RangeError(`${name} must be a positive integer, got ${value}`);
    }
  }

  function layoutFor(kind) {
    const layout = LAYOUTS[kind];
    if (!layout) {
      throw new TypeError(`unknown indirect kind '${kind}'`);
    }
    return layout;
  }

  /**
   * Byte size of one indirect record of `kind` (16 / 20 / 12).
   *
   * @param {string} kind One of INDIRECT_KIND.
   * @returns {number}
   */
  function recordByteSize(kind) {
    return layoutFor(kind).byteSize;
  }

  /**
   * WGSL source of the struct aliasing a record of `kind`.
   *
   * @param {string} kind One of INDIRECT_KIND.
   * @returns {string}
   */
  function wgslStructSource(kind) {
    layoutFor(kind);
    return WGSL_STRUCTS[kind];
  }

  /**
   * WGSL array stride (bytes) for `array<Struct>` of `kind`; equals the record size.
   *
   * @param {string} kind
   * @returns {number}
   */
  function wgslArrayStride(kind) {
    return layoutFor(kind).byteSize;
  }

  function checkField(kind, name, value, signed) {
    if (!Number.isInteger(value)) {
      throw new TypeError(`${kind}.${name} must be an integer, got ${value}`);
    }
    const min = signed ? I32_MIN : 0;
    const max = signed ? I32_MAX : U32_MAX;
    if (value < min || value > max) {
      throw new RangeError(`${kind}.${name} must be in [${min}, ${max}], got ${value}`);
    }
  }

  /**
   * Validate field types and ranges of a record and apply defaults. Only `firstVertex`,
   * `firstIndex`, `baseVertex` and `firstInstance` default to 0; counts are required.
   *
   * @param {string} kind One of INDIRECT_KIND.
   * @param {object} record
   * @returns {object} A new object with every field present.
   */
  function normalizeRecord(kind, record) {
    const layout = layoutFor(kind);
    if (record === null || typeof record !== 'object') {
      throw new TypeError(`${kind} record must be an object`);
    }
    const defaults = { firstVertex: 0, firstIndex: 0, baseVertex: 0, firstInstance: 0 };
    const out = {};
    for (const [name, spec] of Object.entries(layout.fields)) {
      let value = record[name];
      if (value === undefined && name in defaults) {
        value = defaults[name];
      }
      checkField(kind, name, value, spec.signed);
      out[name] = value;
    }
    return out;
  }

  function toDataView(target, byteOffset, byteLength, label) {
    let view;
    if (target instanceof DataView) {
      view = target;
    } else if (ArrayBuffer.isView(target)) {
      view = new DataView(target.buffer, target.byteOffset, target.byteLength);
    } else if (target instanceof ArrayBuffer) {
      view = new DataView(target);
    } else {
      throw new TypeError(`${label} expects an ArrayBuffer, DataView, or typed array`);
    }
    assertNonNegativeInteger(byteOffset, 'byteOffset');
    if (byteOffset + byteLength > view.byteLength) {
      throw new RangeError(
        `${label}: ${byteLength} bytes at offset ${byteOffset} exceed target of ${view.byteLength} bytes`
      );
    }
    return view;
  }

  /**
   * Encode one record of `kind` into `target` at `byteOffset` (little-endian).
   *
   * @param {string} kind One of INDIRECT_KIND.
   * @param {object} record
   * @param {ArrayBuffer|DataView|ArrayBufferView} target
   * @param {number} [byteOffset=0]
   * @returns {number} Bytes written.
   */
  function encodeRecord(kind, record, target, byteOffset = 0) {
    const layout = layoutFor(kind);
    const values = normalizeRecord(kind, record);
    const view = toDataView(target, byteOffset, layout.byteSize, `encode ${kind}`);
    for (const [name, spec] of Object.entries(layout.fields)) {
      if (spec.signed) {
        view.setInt32(byteOffset + spec.offset, values[name], true);
      } else {
        view.setUint32(byteOffset + spec.offset, values[name], true);
      }
    }
    return layout.byteSize;
  }

  /**
   * Decode one record of `kind` from `source` at `byteOffset`.
   *
   * @param {string} kind One of INDIRECT_KIND.
   * @param {ArrayBuffer|DataView|ArrayBufferView} source
   * @param {number} [byteOffset=0]
   * @returns {object}
   */
  function decodeRecord(kind, source, byteOffset = 0) {
    const layout = layoutFor(kind);
    const view = toDataView(source, byteOffset, layout.byteSize, `decode ${kind}`);
    const out = {};
    for (const [name, spec] of Object.entries(layout.fields)) {
      out[name] = spec.signed
        ? view.getInt32(byteOffset + spec.offset, true)
        : view.getUint32(byteOffset + spec.offset, true);
    }
    return out;
  }

  /**
   * Encode a `drawIndirect` record (16 bytes) into a new Uint8Array.
   *
   * @param {{vertexCount:number, instanceCount:number, firstVertex?:number, firstInstance?:number}} record
   * @returns {Uint8Array}
   */
  function encodeDrawIndirect(record) {
    const bytes = new Uint8Array(DRAW_INDIRECT_BYTES);
    encodeRecord(INDIRECT_KIND.DRAW, record, bytes);
    return bytes;
  }

  /**
   * Encode a `drawIndexedIndirect` record (20 bytes, i32 baseVertex) into a new Uint8Array.
   *
   * @param {{indexCount:number, instanceCount:number, firstIndex?:number, baseVertex?:number, firstInstance?:number}} record
   * @returns {Uint8Array}
   */
  function encodeDrawIndexedIndirect(record) {
    const bytes = new Uint8Array(DRAW_INDEXED_INDIRECT_BYTES);
    encodeRecord(INDIRECT_KIND.DRAW_INDEXED, record, bytes);
    return bytes;
  }

  /**
   * Encode a `dispatchWorkgroupsIndirect` record (12 bytes) into a new Uint8Array.
   *
   * @param {{x:number, y:number, z:number}} record
   * @returns {Uint8Array}
   */
  function encodeDispatchIndirect(record) {
    const bytes = new Uint8Array(DISPATCH_INDIRECT_BYTES);
    encodeRecord(INDIRECT_KIND.DISPATCH, record, bytes);
    return bytes;
  }

  /**
   * Decode a 16-byte drawIndirect record.
   * @param {ArrayBuffer|DataView|ArrayBufferView} source
   * @param {number} [byteOffset=0]
   * @returns {object}
   */
  function decodeDrawIndirect(source, byteOffset = 0) {
    return decodeRecord(INDIRECT_KIND.DRAW, source, byteOffset);
  }

  /**
   * Decode a 20-byte drawIndexedIndirect record.
   * @param {ArrayBuffer|DataView|ArrayBufferView} source
   * @param {number} [byteOffset=0]
   * @returns {object}
   */
  function decodeDrawIndexedIndirect(source, byteOffset = 0) {
    return decodeRecord(INDIRECT_KIND.DRAW_INDEXED, source, byteOffset);
  }

  /**
   * Decode a 12-byte dispatchWorkgroupsIndirect record.
   * @param {ArrayBuffer|DataView|ArrayBufferView} source
   * @param {number} [byteOffset=0]
   * @returns {object}
   */
  function decodeDispatchIndirect(source, byteOffset = 0) {
    return decodeRecord(INDIRECT_KIND.DISPATCH, source, byteOffset);
  }

  /**
   * Encode N records of `kind` at `stride` bytes apart.
   *
   * `firstInstance != 0` requires the `indirect-first-instance` feature. By default this
   * throws IndirectDrawError('FIRST_INSTANCE_UNSUPPORTED') naming the record index; with
   * `patchFirstInstance: true` the field is written as 0 and the index is reported instead.
   *
   * @param {string} kind One of INDIRECT_KIND.
   * @param {object[]} records
   * @param {object} [options]
   * @param {number} [options.stride] Byte distance between records (>= record size, multiple of 4).
   * @param {boolean} [options.indirectFirstInstance=false] Device has `indirect-first-instance`.
   * @param {boolean} [options.patchFirstInstance=false] Patch firstInstance to 0 instead of throwing.
   * @param {Uint8Array} [options.out] Destination; allocated when omitted.
   * @param {number} [options.outOffset=0] Byte offset into `out`.
   * @param {number} [options.maxIndirectBatchCount=4096] Maximum records allowed per batch.
   * @param {boolean} [options.clampBatch=false] Clamp oversized batches instead of throwing.
   * @returns {{bytes: Uint8Array, stride: number, patched: number[]}} `patched` lists record indices.
   */
  function encodeBatch(kind, records, options = {}) {
    const layout = layoutFor(kind);
    if (!Array.isArray(records)) {
      throw new TypeError('records must be an array');
    }
    const stride = options.stride === undefined ? layout.byteSize : options.stride;
    if (!Number.isInteger(stride) || stride < layout.byteSize || stride % INDIRECT_OFFSET_ALIGNMENT !== 0) {
      recordBoundsRejection(null);
      throw new RangeError(
        `stride must be a multiple of 4 and >= ${layout.byteSize} for ${kind}, got ${stride}`
      );
    }
    const outOffset = options.outOffset || 0;
    if (!Number.isInteger(outOffset) || outOffset < 0) {
      recordBoundsRejection(null);
      throw new RangeError(`outOffset must be a non-negative integer, got ${outOffset}`);
    }
    const batch = resolveBatchCount(records.length, options, null);
    if (batch.count === 0) {
      recordStat('zeroBatchSkips');
      const bytes = options.out || new Uint8Array(0);
      if (!(bytes instanceof Uint8Array)) {
        throw new TypeError('out must be a Uint8Array');
      }
      return { bytes, stride, patched: [] };
    }
    const total = stride * (batch.count - 1) + layout.byteSize;
    const bytes = options.out || new Uint8Array(total);
    if (!(bytes instanceof Uint8Array)) {
      throw new TypeError('out must be a Uint8Array');
    }
    if (!Number.isSafeInteger(outOffset + total) || outOffset + total > bytes.byteLength) {
      recordBoundsRejection(null);
      throw new RangeError(
        `encodeBatch ${kind}: ${total} bytes at offset ${outOffset} exceed target of ${bytes.byteLength} bytes`
      );
    }
    const patched = [];
    for (let i = 0; i < batch.count; i += 1) {
      let rec = records[i];
      if ('firstInstance' in layout.fields) {
        const normalized = normalizeRecord(kind, rec);
        if (normalized.firstInstance !== 0 && !options.indirectFirstInstance) {
          if (!options.patchFirstInstance) {
            throw new IndirectDrawError(
              'FIRST_INSTANCE_UNSUPPORTED',
              `record ${i}: firstInstance=${normalized.firstInstance} requires the ` +
                `'indirect-first-instance' feature`
            );
          }
          rec = { ...normalized, firstInstance: 0 };
          patched.push(i);
        }
      }
      encodeRecord(kind, rec, bytes, outOffset + i * stride);
    }
    return { bytes, stride, patched };
  }

  // Fast-path validation status codes (zero allocation on hot paths).
  const RANGE_OK = 0;
  const RANGE_INVALID_OFFSET = 1;
  const RANGE_INVALID_RECORD_SIZE = 2;
  const RANGE_NOT_INDIRECT = 3;
  const RANGE_DESTROYED_BUFFER = 4;
  const RANGE_MISALIGNED_OFFSET = 5;
  const RANGE_INVALID_OPT_BUFFER_SIZE = 6;
  const RANGE_INVALID_BUFFER_SIZE = 7;
  const RANGE_OUT_OF_BOUNDS = 8;

  const RANGE_CODE_TO_REASON = Object.freeze([
    null,
    'INVALID_OFFSET',
    'INVALID_RECORD_SIZE',
    'NOT_INDIRECT',
    'DESTROYED_BUFFER',
    'MISALIGNED_OFFSET',
    'OUT_OF_BOUNDS',
    'OUT_OF_BOUNDS',
    'OUT_OF_BOUNDS',
  ]);

  function isNonNegSafeIntFast(value) {
    if (typeof value !== 'number') {
      return false;
    }
    if (value >= 0 && value <= 0x7fffffff) {
      return (value | 0) === value;
    }
    return value > 0x7fffffff && Number.isSafeInteger(value);
  }

  function isPosSafeIntFast(value) {
    if (typeof value !== 'number') {
      return false;
    }
    if (value >= 1 && value <= 0x7fffffff) {
      return (value | 0) === value;
    }
    return value > 0x7fffffff && Number.isSafeInteger(value);
  }

  function isAligned4Fast(value) {
    return value <= 0x7fffffff ? (value & 3) === 0 : value % INDIRECT_OFFSET_ALIGNMENT === 0;
  }

  /**
   * Resolve effective known buffer byte size without allocating.
   * Returns:
   *  - `>= 0`: known effective byte size
   *  - `-1`: unknown size (neither buffer.size nor explicitBufferSize set)
   *  - `-2`: invalid explicitBufferSize
   *  - `-3`: invalid buffer.size
   */
  function resolveEffectiveBufferSizeFast(buffer, explicitBufferSize) {
    const hasOpt = explicitBufferSize !== undefined && explicitBufferSize !== null;
    const hasBuf = Boolean(buffer && buffer.size !== undefined && buffer.size !== null);
    if (hasOpt && !isNonNegSafeIntFast(explicitBufferSize)) {
      return -2;
    }
    if (hasBuf && !isNonNegSafeIntFast(buffer.size)) {
      return -3;
    }
    if (hasOpt && hasBuf) {
      return buffer.size < explicitBufferSize ? buffer.size : explicitBufferSize;
    }
    if (hasOpt) {
      return explicitBufferSize;
    }
    if (hasBuf) {
      return buffer.size;
    }
    return -1;
  }

  /**
   * Zero-allocation fast-path range validation for WebGPU indirect buffers.
   * Checks offset type/non-negativity, recordSize, INDIRECT usage, destroyed state,
   * 4-byte offset alignment, and buffer bounds whenever size is known.
   *
   * @returns {number} One of RANGE_* codes (`0` = valid).
   */
  function fastCheckIndirectRange(buffer, offset, recordSize, explicitBufferSize) {
    if (!isNonNegSafeIntFast(offset)) {
      return RANGE_INVALID_OFFSET;
    }
    if (!isPosSafeIntFast(recordSize)) {
      return RANGE_INVALID_RECORD_SIZE;
    }
    if (!buffer || typeof buffer !== 'object') {
      return RANGE_NOT_INDIRECT;
    }
    if (buffer.destroyed === true) {
      return RANGE_DESTROYED_BUFFER;
    }
    if ((buffer.usage & BUFFER_USAGE.INDIRECT) === 0) {
      return RANGE_NOT_INDIRECT;
    }
    if (!isAligned4Fast(offset)) {
      return RANGE_MISALIGNED_OFFSET;
    }
    const effectiveSize = resolveEffectiveBufferSizeFast(buffer, explicitBufferSize);
    if (effectiveSize === -2) {
      return RANGE_INVALID_OPT_BUFFER_SIZE;
    }
    if (effectiveSize === -3) {
      return RANGE_INVALID_BUFFER_SIZE;
    }
    if (effectiveSize >= 0) {
      const end = offset + recordSize;
      if (end > effectiveSize || (end > 0x7fffffff && !Number.isSafeInteger(end))) {
        return RANGE_OUT_OF_BOUNDS;
      }
    } else if (offset + recordSize > 0x7fffffff && !Number.isSafeInteger(offset + recordSize)) {
      return RANGE_OUT_OF_BOUNDS;
    }
    return RANGE_OK;
  }

  function throwForRangeCode(code, buffer, offset, recordSize, explicitBufferSize) {
    if (code === RANGE_INVALID_OFFSET) {
      throw new RangeError(`indirectOffset must be a non-negative integer, got ${offset}`);
    }
    if (code === RANGE_INVALID_RECORD_SIZE) {
      throw new RangeError(`recordSize must be a positive integer, got ${recordSize}`);
    }
    if (code === RANGE_DESTROYED_BUFFER) {
      throw new IndirectDrawError('DESTROYED_BUFFER', 'indirect buffer is destroyed');
    }
    if (code === RANGE_NOT_INDIRECT) {
      throw new IndirectDrawError('NOT_INDIRECT', 'indirect buffer lacks INDIRECT usage');
    }
    if (code === RANGE_MISALIGNED_OFFSET) {
      throw new IndirectDrawError(
        'MISALIGNED_OFFSET',
        `indirectOffset ${offset} is not a multiple of ${INDIRECT_OFFSET_ALIGNMENT}`
      );
    }
    if (code === RANGE_INVALID_OPT_BUFFER_SIZE) {
      throw new IndirectDrawError(
        'OUT_OF_BOUNDS',
        `bufferSize must be a non-negative integer, got ${explicitBufferSize}`
      );
    }
    if (code === RANGE_INVALID_BUFFER_SIZE) {
      throw new IndirectDrawError(
        'OUT_OF_BOUNDS',
        `buffer.size must be a non-negative integer, got ${buffer && buffer.size}`
      );
    }
    const effectiveSize = resolveEffectiveBufferSizeFast(buffer, explicitBufferSize);
    throw new IndirectDrawError(
      'OUT_OF_BOUNDS',
      `indirect record at ${offset}+${recordSize} exceeds buffer size ${effectiveSize >= 0 ? effectiveSize : 'MAX_SAFE_INTEGER'}`
    );
  }

  /**
   * Non-throwing, zero-allocation fast-path predicate checking whether an indirect
   * buffer range satisfies WebGPU alignment, usage, liveness, and bounds rules.
   *
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer
   * @param {number} offset
   * @param {number} recordSize
   * @param {object} [options]
   * @param {number} [options.bufferSize]
   * @returns {boolean}
   */
  function isValidIndirectRange(buffer, offset, recordSize, options) {
    const explicitSize = options !== undefined && options !== null ? options.bufferSize : undefined;
    return fastCheckIndirectRange(buffer, offset, recordSize, explicitSize) === RANGE_OK;
  }

  function resolveOutTarget(options, out) {
    if (out !== undefined && out !== null && typeof out === 'object') {
      return out;
    }
    if (
      options !== undefined &&
      options !== null &&
      typeof options === 'object' &&
      options.out !== undefined &&
      options.out !== null &&
      typeof options.out === 'object'
    ) {
      return options.out;
    }
    return null;
  }

  /**
   * Non-throwing range validation returning structured status without mutating telemetry.
   * Supports an optional pre-allocated `out` scratch object (via `options.out` or 5th arg)
   * to avoid per-draw allocations on hot paths.
   *
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer
   * @param {number} offset
   * @param {number} recordSize
   * @param {object} [options]
   * @param {number} [options.bufferSize]
   * @param {object} [options.out]
   * @param {object} [out]
   * @returns {{ok:boolean, code:string|null, offset:number, recordSize:number, bufferSize:number|null}}
   */
  function checkIndirectRange(buffer, offset, recordSize, options, out) {
    const explicitSize = options !== undefined && options !== null ? options.bufferSize : undefined;
    const status = fastCheckIndirectRange(buffer, offset, recordSize, explicitSize);
    const effectiveSize = resolveEffectiveBufferSizeFast(buffer, explicitSize);
    const outTarget = resolveOutTarget(options, out);
    const ok = status === RANGE_OK;
    const code = RANGE_CODE_TO_REASON[status];
    const bufferSize = effectiveSize >= 0 ? effectiveSize : null;
    if (outTarget !== null) {
      outTarget.ok = ok;
      outTarget.code = code;
      outTarget.offset = offset;
      outTarget.recordSize = recordSize;
      outTarget.bufferSize = bufferSize;
      return outTarget;
    }
    return {
      ok,
      code,
      offset,
      recordSize,
      bufferSize,
    };
  }

  /**
   * Return true when a CPU-side indirect record encodes zero primitives or zero workgroups
   * and can be safely elided without issuing a WebGPU pass call.
   *
   * @param {string} kind One of INDIRECT_KIND ('draw' | 'drawIndexed' | 'dispatch').
   * @param {object} record
   * @returns {boolean}
   */
  function isZeroCountRecord(kind, record) {
    layoutFor(kind);
    if (!record || typeof record !== 'object') {
      return false;
    }
    if (kind === INDIRECT_KIND.DRAW) {
      return record.vertexCount === 0 || record.instanceCount === 0;
    }
    if (kind === INDIRECT_KIND.DRAW_INDEXED) {
      return record.indexCount === 0 || record.instanceCount === 0;
    }
    return record.x === 0 || record.y === 0 || record.z === 0;
  }

  function prepareShadowSource(source) {
    if (source === undefined || source === null) {
      return null;
    }
    if (
      source instanceof Uint8Array ||
      source instanceof Uint32Array ||
      source instanceof Int32Array ||
      source instanceof DataView
    ) {
      return source;
    }
    if (ArrayBuffer.isView(source)) {
      return new DataView(source.buffer, source.byteOffset, source.byteLength);
    }
    if (source instanceof ArrayBuffer) {
      return new DataView(source);
    }
    return null;
  }

  /**
   * Zero-allocation zero-count check against a CPU shadow buffer at `byteOffset`.
   * Checks uint32 == 0 directly on Uint8Array / Uint32Array / Int32Array / DataView
   * without allocating temporary DataView wrappers.
   */
  function isZeroCountShadowAt(kind, source, byteOffset) {
    if (source === null || source === undefined) {
      return false;
    }
    const layout = LAYOUTS[kind];
    if (!layout || byteOffset < 0) {
      return false;
    }
    const recordSize = layout.byteSize;
    if (source instanceof Uint8Array) {
      if (byteOffset + recordSize > source.byteLength) {
        return false;
      }
      if ((source[byteOffset] | source[byteOffset + 1] | source[byteOffset + 2] | source[byteOffset + 3]) === 0) {
        return true;
      }
      if ((source[byteOffset + 4] | source[byteOffset + 5] | source[byteOffset + 6] | source[byteOffset + 7]) === 0) {
        return true;
      }
      if (kind === INDIRECT_KIND.DISPATCH) {
        return (
          (source[byteOffset + 8] | source[byteOffset + 9] | source[byteOffset + 10] | source[byteOffset + 11]) === 0
        );
      }
      return false;
    }
    if (source instanceof Uint32Array || source instanceof Int32Array) {
      if (byteOffset + recordSize > source.byteLength) {
        return false;
      }
      if (isAligned4Fast(byteOffset)) {
        const idx = byteOffset <= 0x7fffffff ? byteOffset >>> 2 : byteOffset / 4;
        if (source[idx] === 0 || source[idx + 1] === 0) {
          return true;
        }
        return kind === INDIRECT_KIND.DISPATCH && source[idx + 2] === 0;
      }
    }
    const view = source instanceof DataView ? source : prepareShadowSource(source);
    if (!view || byteOffset + recordSize > view.byteLength) {
      return false;
    }
    if (kind === INDIRECT_KIND.DRAW || kind === INDIRECT_KIND.DRAW_INDEXED) {
      return view.getUint32(byteOffset, true) === 0 || view.getUint32(byteOffset + 4, true) === 0;
    }
    return (
      view.getUint32(byteOffset, true) === 0 ||
      view.getUint32(byteOffset + 4, true) === 0 ||
      view.getUint32(byteOffset + 8, true) === 0
    );
  }

  function hasZeroCountOption(kind, options) {
    if (!options || typeof options !== 'object') {
      return false;
    }
    if (options.drawCount === 0) {
      return true;
    }
    const rec = options.record !== undefined ? options.record : options.args;
    if (rec && typeof rec === 'object' && isZeroCountRecord(kind, rec)) {
      return true;
    }
    if (kind === INDIRECT_KIND.DRAW) {
      if (options.vertexCount === 0 || options.instanceCount === 0) {
        return true;
      }
    } else if (kind === INDIRECT_KIND.DRAW_INDEXED) {
      if (options.indexCount === 0 || options.instanceCount === 0) {
        return true;
      }
    } else if (kind === INDIRECT_KIND.DISPATCH) {
      if (options.x === 0 || options.y === 0 || options.z === 0) {
        return true;
      }
    }
    return false;
  }

  function writeValidationResult(
    target,
    ok,
    code,
    kind,
    offset,
    recordSize,
    requiredStrideBytes,
    bufferSize,
    zeroCount
  ) {
    if (target !== null) {
      target.ok = ok;
      target.valid = ok;
      target.code = code;
      target.reason = code;
      target.kind = kind;
      target.offset = offset;
      target.recordSize = recordSize;
      target.requiredStrideBytes = requiredStrideBytes;
      target.bufferSize = bufferSize;
      target.zeroCount = zeroCount;
      return target;
    }
    return {
      ok,
      valid: ok,
      code,
      reason: code,
      kind,
      offset,
      recordSize,
      requiredStrideBytes,
      bufferSize,
      zeroCount,
    };
  }

  /**
   * Non-throwing strict validation for an indirect argument buffer draw/dispatch:
   * checks `offset >= 0`, `offset % 4 === 0`, `INDIRECT` usage, liveness, and
   * `offset + requiredStrideBytes <= bufferSize` (16B drawIndirect, 20B drawIndexedIndirect,
   * 12B dispatchWorkgroupsIndirect), plus zero-count detection. Does not mutate telemetry.
   * Supports an optional pre-allocated `out` scratch result object (via `options.out` or 5th arg)
   * for zero-allocation hot-path validation.
   *
   * @param {string} kind One of INDIRECT_KIND ('draw' | 'drawIndexed' | 'dispatch').
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer
   * @param {number} [offset=0]
   * @param {object} [options]
   * @param {number} [options.bufferSize]
   * @param {number} [options.stride]
   * @param {number} [options.requiredStrideBytes]
   * @param {number} [options.count]
   * @param {object} [options.record]
   * @param {object} [options.args]
   * @param {ArrayBuffer|DataView|ArrayBufferView} [options.shadowData]
   * @param {object} [options.out]
   * @param {object} [out]
   * @returns {{
   *   ok: boolean,
   *   valid: boolean,
   *   code: string | null,
   *   reason: string | null,
   *   kind: string,
   *   offset: number,
   *   recordSize: number,
   *   requiredStrideBytes: number,
   *   bufferSize: number | null,
   *   zeroCount: boolean
   * }}
   */
  function validateIndirectBufferArgs(kind, buffer, offset = 0, options, out) {
    const layout = LAYOUTS[kind];
    const hasOpts = options !== undefined && options !== null && typeof options === 'object';
    const outTarget = resolveOutTarget(options, out);
    const explicitSize = hasOpts ? options.bufferSize : undefined;
    const effectiveSize = resolveEffectiveBufferSizeFast(buffer, explicitSize);
    const knownBufferSize = effectiveSize >= 0 ? effectiveSize : null;

    if (!layout) {
      return writeValidationResult(
        outTarget,
        false,
        'UNKNOWN_KIND',
        kind,
        offset,
        0,
        0,
        knownBufferSize,
        false
      );
    }

    const recordSize = layout.byteSize;
    const requiredStrideBytes =
      hasOpts && options.requiredStrideBytes !== undefined
        ? options.requiredStrideBytes
        : hasOpts && options.stride !== undefined
          ? options.stride
          : recordSize;

    if (
      !isPosSafeIntFast(requiredStrideBytes) ||
      requiredStrideBytes < recordSize ||
      !isAligned4Fast(requiredStrideBytes)
    ) {
      return writeValidationResult(
        outTarget,
        false,
        'INVALID_STRIDE',
        kind,
        offset,
        recordSize,
        isPosSafeIntFast(requiredStrideBytes) ? requiredStrideBytes : 0,
        knownBufferSize,
        false
      );
    }

    if (hasOpts && options.count === 0) {
      if (!isNonNegSafeIntFast(offset)) {
        return writeValidationResult(
          outTarget,
          false,
          'INVALID_OFFSET',
          kind,
          offset,
          recordSize,
          requiredStrideBytes,
          knownBufferSize,
          true
        );
      }
      if (!isAligned4Fast(offset)) {
        return writeValidationResult(
          outTarget,
          false,
          'MISALIGNED_OFFSET',
          kind,
          offset,
          recordSize,
          requiredStrideBytes,
          knownBufferSize,
          true
        );
      }
      return writeValidationResult(
        outTarget,
        true,
        'ZERO_COUNT',
        kind,
        offset,
        recordSize,
        requiredStrideBytes,
        knownBufferSize,
        true
      );
    }

    const status = fastCheckIndirectRange(buffer, offset, requiredStrideBytes, explicitSize);
    if (status !== RANGE_OK) {
      return writeValidationResult(
        outTarget,
        false,
        RANGE_CODE_TO_REASON[status],
        kind,
        offset,
        recordSize,
        requiredStrideBytes,
        knownBufferSize,
        false
      );
    }

    const zeroCount =
      hasOpts &&
      options.elideZeroCount !== false &&
      (hasZeroCountOption(kind, options) || isZeroCountShadowAt(kind, options.shadowData, offset));

    return writeValidationResult(
      outTarget,
      true,
      zeroCount ? 'ZERO_COUNT' : null,
      kind,
      offset,
      recordSize,
      requiredStrideBytes,
      knownBufferSize,
      zeroCount
    );
  }

  /**
   * Hot-path zero-allocation indirect draw/indexed-draw argument validator.
   * Accepts either `(kind, buffer, offset, options, out)` or `(buffer, offset, optionsOrIndexed, out)`.
   * Preserves 100% of W269 bounds/alignment/liveness/zero-count checks without mutating telemetry.
   * When `out` (or `options.out`) is provided, populates and returns `out` with zero heap allocations.
   */
  function validateIndirectDrawArgs(kindOrBuffer, bufferOrOffset, offsetOrOptions, optionsOrOut, outArg) {
    if (typeof kindOrBuffer === 'string') {
      return validateIndirectBufferArgs(
        kindOrBuffer,
        bufferOrOffset,
        offsetOrOptions === undefined ? 0 : offsetOrOptions,
        optionsOrOut,
        outArg
      );
    }
    const buffer = kindOrBuffer;
    const offset = bufferOrOffset === undefined ? 0 : bufferOrOffset;
    if (typeof offsetOrOptions === 'boolean') {
      return validateIndirectBufferArgs(
        offsetOrOptions ? INDIRECT_KIND.DRAW_INDEXED : INDIRECT_KIND.DRAW,
        buffer,
        offset,
        undefined,
        optionsOrOut
      );
    }
    const options = offsetOrOptions;
    const kind =
      options !== undefined && options !== null && typeof options === 'object'
        ? options.kind !== undefined
          ? options.kind
          : options.indexed
            ? INDIRECT_KIND.DRAW_INDEXED
            : INDIRECT_KIND.DRAW
        : INDIRECT_KIND.DRAW;
    return validateIndirectBufferArgs(kind, buffer, offset, options, optionsOrOut);
  }

  function maxRecordsFittingBuffer(bufferSize, firstOffset, stride, recordSize) {
    if (firstOffset < 0 || firstOffset >= bufferSize) {
      return 0;
    }
    const remaining = bufferSize - firstOffset;
    if (remaining < recordSize) {
      return 0;
    }
    return 1 + Math.floor((remaining - recordSize) / stride);
  }

  function resolveBatchCount(requestedCount, options, statsTarget) {
    if (!isNonNegSafeIntFast(requestedCount)) {
      recordBoundsRejection(statsTarget);
      throw new RangeError(`count must be a non-negative integer, got ${requestedCount}`);
    }
    const maxIndirectBatchCount =
      options && options.maxIndirectBatchCount !== undefined
        ? options.maxIndirectBatchCount
        : DEFAULT_MAX_INDIRECT_BATCH_COUNT;
    assertPositiveInteger(maxIndirectBatchCount, 'maxIndirectBatchCount');
    if (requestedCount === 0) {
      return { count: 0, requestedCount: 0, clamped: false, maxIndirectBatchCount };
    }
    if (requestedCount > maxIndirectBatchCount) {
      if (options && options.clampBatch) {
        recordBatchClamped(statsTarget);
        return {
          count: maxIndirectBatchCount,
          requestedCount,
          clamped: true,
          maxIndirectBatchCount,
        };
      }
      recordBatchLimitRejected(statsTarget);
      throw new IndirectDrawError(
        'BATCH_LIMIT_EXCEEDED',
        `indirect batch count ${requestedCount} exceeds maxIndirectBatchCount ${maxIndirectBatchCount}`
      );
    }
    return { count: requestedCount, requestedCount, clamped: false, maxIndirectBatchCount };
  }

  /**
   * Validate an indirect buffer range as WebGPU does for draw/dispatch indirect calls:
   * INDIRECT usage, non-destroyed, offset multiple of 4, and offset + recordSize <= bufferSize
   * whenever `buffer.size` or `options.bufferSize` is known.
   *
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer A GPUBuffer (or fake with usage/size).
   * @param {number} offset
   * @param {number} recordSize
   * @param {object} [options]
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @throws {IndirectDrawError} With code NOT_INDIRECT, DESTROYED_BUFFER, MISALIGNED_OFFSET or OUT_OF_BOUNDS.
   */
  function validateIndirectRange(buffer, offset, recordSize, options) {
    const explicitSize = options !== undefined && options !== null ? options.bufferSize : undefined;
    const status = fastCheckIndirectRange(buffer, offset, recordSize, explicitSize);
    if (status === RANGE_OK) {
      return;
    }
    const statsTarget = options !== undefined && options !== null ? options._statsTarget || null : null;
    recordRangeRejection(status, statsTarget, 1);
    throwForRangeCode(status, buffer, offset, recordSize, explicitSize);
  }

  /**
   * Compute a non-throwing, validated/clamped stepping plan for a D3D12 `ExecuteIndirect`
   * style indirect draw/dispatch batch (`count >= 1`, arbitrary 4-byte-aligned `stride >= recordSize`).
   * Does not issue GPU commands or mutate telemetry.
   *
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer
   * @param {object} options
   * @param {string} [options.kind] INDIRECT_KIND ('draw' | 'drawIndexed' | 'dispatch').
   * @param {boolean} [options.indexed=false] Used when neither `kind` nor `recordSize` is set.
   * @param {number} [options.recordSize] Explicit record byte size (16 / 20 / 12).
   * @param {number} [options.count=1] Requested command count.
   * @param {number} [options.firstSlot=0]
   * @param {number} [options.baseOffset=0]
   * @param {number} [options.stride] Byte stride between commands (>= recordSize, multiple of 4).
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {number} [options.maxIndirectBatchCount=4096]
   * @param {boolean} [options.clampBatch=true] Clamp count to maxIndirectBatchCount.
   * @param {boolean} [options.clampToBuffer=true] Clamp count to records fitting in bufferSize.
   * @returns {{
   *   valid: boolean,
   *   reason: string | null,
   *   requestedCount: number,
   *   effectiveCount: number,
   *   clamped: boolean,
   *   clampedReason: string | null,
   *   recordSize: number,
   *   stride: number,
   *   firstOffset: number,
   *   lastOffset: number,
   *   requiredBytes: number,
   *   bufferSize: number | null
   * }}
   */
  function computeIndirectBatchStepPlan(buffer, options) {
    const opts = options && typeof options === 'object' ? options : {};
    let recordSize;
    if (opts.recordSize !== undefined) {
      recordSize = opts.recordSize;
    } else if (opts.kind !== undefined) {
      const layout = LAYOUTS[opts.kind];
      recordSize = layout ? layout.byteSize : -1;
    } else {
      recordSize = opts.indexed ? DRAW_INDEXED_INDIRECT_BYTES : DRAW_INDIRECT_BYTES;
    }

    const requestedCount = opts.count !== undefined ? opts.count : opts.maxCommandCount !== undefined ? opts.maxCommandCount : 1;
    const firstSlot = opts.firstSlot === undefined ? 0 : opts.firstSlot;
    const baseOffset = opts.baseOffset === undefined ? 0 : opts.baseOffset;
    const stride = opts.stride === undefined ? recordSize : opts.stride;
    const clampBatch = opts.clampBatch !== false;
    const clampToBuffer = opts.clampToBuffer !== false;
    const maxBatch =
      opts.maxIndirectBatchCount === undefined
        ? DEFAULT_MAX_INDIRECT_BATCH_COUNT
        : opts.maxIndirectBatchCount;
    const explicitSize = opts.bufferSize;
    const effectiveSize = resolveEffectiveBufferSizeFast(buffer, explicitSize);
    const knownBufferSize = effectiveSize >= 0 ? effectiveSize : null;

    const fail = (reason) => ({
      valid: false,
      reason,
      requestedCount: isNonNegSafeIntFast(requestedCount) ? requestedCount : 0,
      effectiveCount: 0,
      clamped: false,
      clampedReason: null,
      recordSize: isPosSafeIntFast(recordSize) ? recordSize : 0,
      stride: isPosSafeIntFast(stride) ? stride : 0,
      firstOffset: isNonNegSafeIntFast(baseOffset) ? baseOffset : 0,
      lastOffset: 0,
      requiredBytes: 0,
      bufferSize: knownBufferSize,
    });

    if (!isPosSafeIntFast(recordSize)) {
      return fail('INVALID_RECORD_SIZE');
    }
    if (!isNonNegSafeIntFast(firstSlot)) {
      return fail('INVALID_OFFSET');
    }
    if (!isNonNegSafeIntFast(baseOffset)) {
      return fail('INVALID_OFFSET');
    }
    if (!isPosSafeIntFast(stride) || stride < recordSize || !isAligned4Fast(stride)) {
      return fail('INVALID_STRIDE');
    }
    if (!isPosSafeIntFast(maxBatch)) {
      return fail('INVALID_BATCH_LIMIT');
    }
    if (!isNonNegSafeIntFast(requestedCount)) {
      return fail('INVALID_COUNT');
    }
    if (!buffer || typeof buffer !== 'object') {
      return fail('NOT_INDIRECT');
    }
    if (buffer.destroyed === true) {
      return fail('DESTROYED_BUFFER');
    }
    if ((buffer.usage & BUFFER_USAGE.INDIRECT) === 0) {
      return fail('NOT_INDIRECT');
    }
    if (effectiveSize === -2 || effectiveSize === -3) {
      return fail('OUT_OF_BOUNDS');
    }

    const firstOffset = baseOffset + firstSlot * stride;
    if (!isNonNegSafeIntFast(firstOffset)) {
      return fail('OUT_OF_BOUNDS');
    }
    if (!isAligned4Fast(firstOffset)) {
      return fail('MISALIGNED_OFFSET');
    }

    if (requestedCount === 0) {
      return {
        valid: true,
        reason: null,
        requestedCount: 0,
        effectiveCount: 0,
        clamped: false,
        clampedReason: null,
        recordSize,
        stride,
        firstOffset,
        lastOffset: firstOffset,
        requiredBytes: 0,
        bufferSize: knownBufferSize,
      };
    }

    let effectiveCount = requestedCount;
    let clampedByBatch = false;
    let clampedByBuffer = false;

    if (effectiveCount > maxBatch) {
      if (!clampBatch) {
        return fail('BATCH_LIMIT_EXCEEDED');
      }
      effectiveCount = maxBatch;
      clampedByBatch = true;
    }

    if (effectiveSize >= 0) {
      const maxFit = maxRecordsFittingBuffer(effectiveSize, firstOffset, stride, recordSize);
      if (effectiveCount > maxFit) {
        if (!clampToBuffer || maxFit === 0) {
          return fail('OUT_OF_BOUNDS');
        }
        effectiveCount = maxFit;
        clampedByBuffer = true;
      }
    }

    const requiredBytes = stride * (effectiveCount - 1) + recordSize;
    const endByte = firstOffset + requiredBytes;
    if (!Number.isSafeInteger(endByte) || (effectiveSize >= 0 && endByte > effectiveSize)) {
      return fail('OUT_OF_BOUNDS');
    }

    const clamped = clampedByBatch || clampedByBuffer;
    const clampedReason =
      clampedByBatch && clampedByBuffer
        ? 'CLAMPED_TO_BATCH_LIMIT_AND_BUFFER'
        : clampedByBatch
          ? 'CLAMPED_TO_BATCH_LIMIT'
          : clampedByBuffer
            ? 'CLAMPED_TO_BUFFER'
            : null;

    return {
      valid: true,
      reason: clampedReason,
      requestedCount,
      effectiveCount,
      clamped,
      clampedReason,
      recordSize,
      stride,
      firstOffset,
      lastOffset: firstOffset + (effectiveCount - 1) * stride,
      requiredBytes,
      bufferSize: knownBufferSize,
    };
  }

  /**
   * Validate an indirect batch's stride, offset alignment, buffer bounds, and
   * `maxIndirectBatchCount` guardrail without issuing pass commands.
   *
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect buffer.
   * @param {object} options
   * @param {string} [options.kind] INDIRECT_KIND.
   * @param {boolean} [options.indexed=false] Used when neither `kind` nor `recordSize` is set.
   * @param {number} [options.recordSize] Explicit record byte size (16 / 20 / 12).
   * @param {number} options.count Number of indirect records in the batch.
   * @param {number} [options.firstSlot=0]
   * @param {number} [options.baseOffset=0]
   * @param {number} [options.stride] Bytes between slots (>= recordSize, multiple of 4).
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {number} [options.maxIndirectBatchCount=4096]
   * @param {boolean} [options.clampBatch=false] Clamp oversized batches instead of throwing.
   * @returns {{count:number, requestedCount:number, clamped:boolean, recordSize:number, stride:number, firstOffset:number, requiredBytes:number}}
   */
  function validateIndirectBatch(buffer, options) {
    if (!options || typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const statsTarget = options._statsTarget || null;
    const recordSize =
      options.recordSize !== undefined
        ? options.recordSize
        : options.kind !== undefined
          ? recordByteSize(options.kind)
          : options.indexed
            ? DRAW_INDEXED_INDIRECT_BYTES
            : DRAW_INDIRECT_BYTES;
    if (!isPosSafeIntFast(recordSize)) {
      recordBoundsRejection(statsTarget);
      throw new RangeError(`recordSize must be a positive integer, got ${recordSize}`);
    }
    const stride = options.stride === undefined ? recordSize : options.stride;
    const firstSlot = options.firstSlot || 0;
    const baseOffset = options.baseOffset || 0;
    if (!isNonNegSafeIntFast(firstSlot)) {
      recordBoundsRejection(statsTarget);
      throw new RangeError(`firstSlot must be a non-negative integer, got ${firstSlot}`);
    }
    if (!isNonNegSafeIntFast(baseOffset)) {
      recordBoundsRejection(statsTarget);
      throw new RangeError(`baseOffset must be a non-negative integer, got ${baseOffset}`);
    }
    if (!isPosSafeIntFast(stride) || stride < recordSize || !isAligned4Fast(stride)) {
      recordBoundsRejection(statsTarget);
      throw new RangeError(`stride must be a multiple of 4 and >= ${recordSize}, got ${stride}`);
    }
    const batch = resolveBatchCount(options.count, options, statsTarget);
    const firstOffset = baseOffset + firstSlot * stride;
    if (batch.count === 0) {
      return {
        count: 0,
        requestedCount: batch.requestedCount,
        clamped: false,
        recordSize,
        stride,
        firstOffset,
        requiredBytes: 0,
      };
    }
    const requiredBytes = stride * (batch.count - 1) + recordSize;
    validateIndirectRange(buffer, firstOffset, requiredBytes, {
      bufferSize: options.bufferSize,
      _statsTarget: statsTarget,
    });
    return {
      count: batch.count,
      requestedCount: batch.requestedCount,
      clamped: batch.clamped,
      recordSize,
      stride,
      firstOffset,
      requiredBytes,
    };
  }

  /**
   * Validate dispatch counts against `maxComputeWorkgroupsPerDimension`.
   *
   * @param {{x:number, y?:number, z?:number}} counts
   * @param {number} [limit=65535] device.limits.maxComputeWorkgroupsPerDimension
   * @returns {{x:number, y:number, z:number}}
   * @throws {IndirectDrawError} With code DISPATCH_LIMIT.
   */
  function validateDispatchCounts(counts, limit = DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION) {
    assertPositiveInteger(limit, 'maxComputeWorkgroupsPerDimension');
    const out = {
      x: counts.x,
      y: counts.y === undefined ? 1 : counts.y,
      z: counts.z === undefined ? 1 : counts.z,
    };
    for (const axis of ['x', 'y', 'z']) {
      checkField('dispatch', axis, out[axis], false);
      if (out[axis] > limit) {
        throw new IndirectDrawError(
          'DISPATCH_LIMIT',
          `dispatch ${axis}=${out[axis]} exceeds maxComputeWorkgroupsPerDimension ${limit}`
        );
      }
    }
    return out;
  }

  /**
   * Plan the dispatches needed to cover `itemCount` items with `workgroupSize` items per
   * workgroup. When the workgroup count exceeds `limit` the work is split into several 1D
   * dispatches of at most `limit` workgroups. The shader must add `firstItem` (pass it in a
   * uniform) to `global_invocation_id.x` to address its items.
   *
   * @param {number} itemCount
   * @param {number} workgroupSize Items per workgroup (product of @workgroup_size).
   * @param {number} [limit=65535] device.limits.maxComputeWorkgroupsPerDimension
   * @returns {{x:number, y:number, z:number, firstWorkgroup:number, firstItem:number}[]}
   *   Empty when `itemCount` is 0.
   */
  function planDispatches(itemCount, workgroupSize, limit = DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION) {
    assertNonNegativeInteger(itemCount, 'itemCount');
    assertPositiveInteger(workgroupSize, 'workgroupSize');
    assertPositiveInteger(limit, 'maxComputeWorkgroupsPerDimension');
    const total = Math.ceil(itemCount / workgroupSize);
    const plan = [];
    for (let first = 0; first < total; first += limit) {
      plan.push({
        x: Math.min(limit, total - first),
        y: 1,
        z: 1,
        firstWorkgroup: first,
        firstItem: first * workgroupSize,
      });
    }
    return plan;
  }

  /**
   * Encode planned dispatches onto a compute pass encoder via `dispatchWorkgroups`.
   *
   * @param {{dispatchWorkgroups: Function}} pass
   * @param {{x:number,y:number,z:number}[]} plan From planDispatches.
   * @param {(dispatch: object, index: number) => void} [beforeEach] Hook to update the offset uniform.
   * @returns {number} Number of dispatches issued.
   */
  function encodeDispatches(pass, plan, beforeEach) {
    plan.forEach((dispatch, index) => {
      if (beforeEach) {
        beforeEach(dispatch, index);
      }
      pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    });
    return plan.length;
  }

  /**
   * Encode a validated 12-byte `dispatchWorkgroupsIndirect` record for `itemCount` items
   * (e.g. to seed a buffer a previous compute pass will refine).
   *
   * @param {number} itemCount
   * @param {number} workgroupSize
   * @param {number} [limit=65535]
   * @returns {Uint8Array}
   * @throws {IndirectDrawError} DISPATCH_LIMIT when the count does not fit one dimension.
   */
  function encodeDispatchForItems(itemCount, workgroupSize, limit = DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION) {
    assertNonNegativeInteger(itemCount, 'itemCount');
    assertPositiveInteger(workgroupSize, 'workgroupSize');
    const counts = validateDispatchCounts({ x: Math.ceil(itemCount / workgroupSize), y: 1, z: 1 }, limit);
    return encodeDispatchIndirect(counts);
  }

  function visibleAt(mask, i) {
    if (mask === undefined || mask === null) {
      return true;
    }
    return Boolean(mask[i]);
  }

  function writeEncodeSingleResult(target, offset, byteSize, executed, skipped, reason) {
    if (target !== null) {
      target.offset = offset;
      target.byteSize = byteSize;
      target.executed = executed;
      target.skipped = skipped;
      target.reason = reason;
      return target;
    }
    if (executed && !skipped) {
      return { offset, byteSize };
    }
    return {
      offset,
      byteSize,
      executed,
      skipped,
      reason,
    };
  }

  /**
   * Validate indirect offset alignment/bounds (16 bytes) and issue one `drawIndirect`.
   *
   * @param {{drawIndirect: Function}} pass GPURenderPassEncoder.
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect buffer.
   * @param {number} [indirectOffset=0]
   * @param {object} [options]
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {boolean} [options.skipInvalid=false] Skip instead of throwing on invalid args.
   * @param {object} [options.out] Optional scratch result object.
   * @param {object} [out] Optional scratch result object.
   * @returns {{offset:number, byteSize:number, executed?:boolean, skipped?:boolean, reason?:string}}
   */
  function encodeIndirectDraw(pass, buffer, indirectOffset = 0, options, out) {
    const hasOpts = options !== undefined && options !== null;
    const outTarget = resolveOutTarget(options, out);
    const statsTarget = hasOpts ? options._statsTarget || null : null;
    const skipInvalid = hasOpts && Boolean(options.skipInvalid);
    recordStat('commandsValidated', 1, statsTarget);
    if (hasOpts && options.count === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      recordStat('zeroCountElided', 1, statsTarget);
      return writeEncodeSingleResult(
        outTarget,
        indirectOffset,
        DRAW_INDIRECT_BYTES,
        false,
        true,
        'ZERO_COUNT'
      );
    }
    if (!pass || typeof pass.drawIndirect !== 'function') {
      if (skipInvalid) {
        recordBoundsRejection(statsTarget);
        return writeEncodeSingleResult(
          outTarget,
          indirectOffset,
          DRAW_INDIRECT_BYTES,
          false,
          true,
          'MISSING_PASS_METHOD'
        );
      }
      throw new TypeError('pass must implement drawIndirect');
    }
    const explicitSize = hasOpts ? options.bufferSize : undefined;
    const status = fastCheckIndirectRange(buffer, indirectOffset, DRAW_INDIRECT_BYTES, explicitSize);
    if (status !== RANGE_OK) {
      recordRangeRejection(status, statsTarget, 1, skipInvalid);
      if (skipInvalid) {
        return writeEncodeSingleResult(
          outTarget,
          indirectOffset,
          DRAW_INDIRECT_BYTES,
          false,
          true,
          RANGE_CODE_TO_REASON[status]
        );
      }
      throwForRangeCode(status, buffer, indirectOffset, DRAW_INDIRECT_BYTES, explicitSize);
    }
    if (
      hasOpts &&
      options.elideZeroCount !== false &&
      (hasZeroCountOption(INDIRECT_KIND.DRAW, options) ||
        isZeroCountShadowAt(INDIRECT_KIND.DRAW, options.shadowData, indirectOffset))
    ) {
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      recordStat('zeroCountElided', 1, statsTarget);
      return writeEncodeSingleResult(
        outTarget,
        indirectOffset,
        DRAW_INDIRECT_BYTES,
        false,
        true,
        'ZERO_COUNT'
      );
    }
    pass.drawIndirect(buffer, indirectOffset);
    recordEncodedKind(INDIRECT_KIND.DRAW, 1, statsTarget);
    return writeEncodeSingleResult(outTarget, indirectOffset, DRAW_INDIRECT_BYTES, true, false, null);
  }

  /**
   * Validate indirect offset alignment/bounds (20 bytes) and issue one `drawIndexedIndirect`.
   *
   * @param {{drawIndexedIndirect: Function}} pass GPURenderPassEncoder.
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect buffer.
   * @param {number} [indirectOffset=0]
   * @param {object} [options]
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {boolean} [options.skipInvalid=false] Skip instead of throwing on invalid args.
   * @param {object} [options.out] Optional scratch result object.
   * @param {object} [out] Optional scratch result object.
   * @returns {{offset:number, byteSize:number, executed?:boolean, skipped?:boolean, reason?:string}}
   */
  function encodeIndirectDrawIndexed(pass, buffer, indirectOffset = 0, options, out) {
    const hasOpts = options !== undefined && options !== null;
    const outTarget = resolveOutTarget(options, out);
    const statsTarget = hasOpts ? options._statsTarget || null : null;
    const skipInvalid = hasOpts && Boolean(options.skipInvalid);
    recordStat('commandsValidated', 1, statsTarget);
    if (hasOpts && options.count === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      recordStat('zeroCountElided', 1, statsTarget);
      return writeEncodeSingleResult(
        outTarget,
        indirectOffset,
        DRAW_INDEXED_INDIRECT_BYTES,
        false,
        true,
        'ZERO_COUNT'
      );
    }
    if (!pass || typeof pass.drawIndexedIndirect !== 'function') {
      if (skipInvalid) {
        recordBoundsRejection(statsTarget);
        return writeEncodeSingleResult(
          outTarget,
          indirectOffset,
          DRAW_INDEXED_INDIRECT_BYTES,
          false,
          true,
          'MISSING_PASS_METHOD'
        );
      }
      throw new TypeError('pass must implement drawIndexedIndirect');
    }
    const explicitSize = hasOpts ? options.bufferSize : undefined;
    const status = fastCheckIndirectRange(
      buffer,
      indirectOffset,
      DRAW_INDEXED_INDIRECT_BYTES,
      explicitSize
    );
    if (status !== RANGE_OK) {
      recordRangeRejection(status, statsTarget, 1, skipInvalid);
      if (skipInvalid) {
        return writeEncodeSingleResult(
          outTarget,
          indirectOffset,
          DRAW_INDEXED_INDIRECT_BYTES,
          false,
          true,
          RANGE_CODE_TO_REASON[status]
        );
      }
      throwForRangeCode(status, buffer, indirectOffset, DRAW_INDEXED_INDIRECT_BYTES, explicitSize);
    }
    if (
      hasOpts &&
      options.elideZeroCount !== false &&
      (hasZeroCountOption(INDIRECT_KIND.DRAW_INDEXED, options) ||
        isZeroCountShadowAt(INDIRECT_KIND.DRAW_INDEXED, options.shadowData, indirectOffset))
    ) {
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      recordStat('zeroCountElided', 1, statsTarget);
      return writeEncodeSingleResult(
        outTarget,
        indirectOffset,
        DRAW_INDEXED_INDIRECT_BYTES,
        false,
        true,
        'ZERO_COUNT'
      );
    }
    pass.drawIndexedIndirect(buffer, indirectOffset);
    recordEncodedKind(INDIRECT_KIND.DRAW_INDEXED, 1, statsTarget);
    return writeEncodeSingleResult(
      outTarget,
      indirectOffset,
      DRAW_INDEXED_INDIRECT_BYTES,
      true,
      false,
      null
    );
  }

  /**
   * Validate indirect offset alignment/bounds (12 bytes) and issue one `dispatchWorkgroupsIndirect`.
   *
   * @param {{dispatchWorkgroupsIndirect: Function}} pass GPUComputePassEncoder.
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect buffer.
   * @param {number} [indirectOffset=0]
   * @param {object} [options]
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {boolean} [options.skipInvalid=false] Skip instead of throwing on invalid args.
   * @param {object} [options.out] Optional scratch result object.
   * @param {object} [out] Optional scratch result object.
   * @returns {{offset:number, byteSize:number, executed?:boolean, skipped?:boolean, reason?:string}}
   */
  function encodeIndirectDispatch(pass, buffer, indirectOffset = 0, options, out) {
    const hasOpts = options !== undefined && options !== null;
    const outTarget = resolveOutTarget(options, out);
    const statsTarget = hasOpts ? options._statsTarget || null : null;
    const skipInvalid = hasOpts && Boolean(options.skipInvalid);
    recordStat('commandsValidated', 1, statsTarget);
    if (hasOpts && options.count === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      recordStat('zeroCountElided', 1, statsTarget);
      return writeEncodeSingleResult(
        outTarget,
        indirectOffset,
        DISPATCH_INDIRECT_BYTES,
        false,
        true,
        'ZERO_COUNT'
      );
    }
    if (!pass || typeof pass.dispatchWorkgroupsIndirect !== 'function') {
      if (skipInvalid) {
        recordBoundsRejection(statsTarget);
        return writeEncodeSingleResult(
          outTarget,
          indirectOffset,
          DISPATCH_INDIRECT_BYTES,
          false,
          true,
          'MISSING_PASS_METHOD'
        );
      }
      throw new TypeError('pass must implement dispatchWorkgroupsIndirect');
    }
    const explicitSize = hasOpts ? options.bufferSize : undefined;
    const status = fastCheckIndirectRange(
      buffer,
      indirectOffset,
      DISPATCH_INDIRECT_BYTES,
      explicitSize
    );
    if (status !== RANGE_OK) {
      recordRangeRejection(status, statsTarget, 1, skipInvalid);
      if (skipInvalid) {
        return writeEncodeSingleResult(
          outTarget,
          indirectOffset,
          DISPATCH_INDIRECT_BYTES,
          false,
          true,
          RANGE_CODE_TO_REASON[status]
        );
      }
      throwForRangeCode(status, buffer, indirectOffset, DISPATCH_INDIRECT_BYTES, explicitSize);
    }
    if (
      hasOpts &&
      options.elideZeroCount !== false &&
      (hasZeroCountOption(INDIRECT_KIND.DISPATCH, options) ||
        isZeroCountShadowAt(INDIRECT_KIND.DISPATCH, options.shadowData, indirectOffset))
    ) {
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      recordStat('zeroCountElided', 1, statsTarget);
      return writeEncodeSingleResult(
        outTarget,
        indirectOffset,
        DISPATCH_INDIRECT_BYTES,
        false,
        true,
        'ZERO_COUNT'
      );
    }
    pass.dispatchWorkgroupsIndirect(buffer, indirectOffset);
    recordEncodedKind(INDIRECT_KIND.DISPATCH, 1, statsTarget);
    return writeEncodeSingleResult(
      outTarget,
      indirectOffset,
      DISPATCH_INDIRECT_BYTES,
      true,
      false,
      null
    );
  }

  /**
   * Issue draws for `count` consecutive slots on a render pass encoder. Baseline WebGPU has no
   * multiDrawIndirect, so each visible slot becomes one `drawIndirect`/`drawIndexedIndirect`.
   * With `useNative` and a pass exposing `multiDrawIndirect`/`multiDrawIndexedIndirect`
   * (experimental `chromium-experimental-multi-draw-indirect`), contiguous visible runs are
   * issued as one native call each when records are tightly packed (`stride === recordSize`).
   *
   * @param {object} pass GPURenderPassEncoder.
   * @param {{usage:number,size?:number,destroyed?:boolean}} buffer Indirect buffer.
   * @param {object} options
   * @param {boolean} [options.indexed=false] Use drawIndexedIndirect records (20 bytes).
   * @param {number} [options.firstSlot=0]
   * @param {number} options.count
   * @param {number} [options.baseOffset=0] Byte offset of slot 0.
   * @param {number} [options.stride] Bytes between slots (default record size).
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {ArrayLike<boolean|number>} [options.visibility] CPU mask indexed from `firstSlot`; falsy skips.
   * @param {boolean} [options.useNative=false] Prefer native multi-draw when available.
   * @param {number} [options.maxIndirectBatchCount=4096] Maximum draws allowed per batch.
   * @param {boolean} [options.clampBatch=false] Clamp oversized batches instead of throwing.
   * @param {boolean} [options.clampToBuffer=false] Clamp count to records fitting within known buffer size.
   * @param {boolean} [options.skipInvalid=false] Skip invalid ranges/args instead of throwing.
   * @returns {{calls:number, draws:number, skipped:number, native:boolean}}
   */
  function encodeMultiDraw(pass, buffer, options) {
    if (!options || typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const statsTarget = options._statsTarget || null;
    const indexed = Boolean(options.indexed);
    const kind = indexed ? INDIRECT_KIND.DRAW_INDEXED : INDIRECT_KIND.DRAW;
    const recordSize = indexed ? DRAW_INDEXED_INDIRECT_BYTES : DRAW_INDIRECT_BYTES;
    const stride = options.stride === undefined ? recordSize : options.stride;
    const firstSlot = options.firstSlot || 0;
    const baseOffset = options.baseOffset || 0;
    const skipInvalid = Boolean(options.skipInvalid);
    const clampToBuffer = Boolean(options.clampToBuffer);
    const method = indexed ? 'drawIndexedIndirect' : 'drawIndirect';
    const nativeMethod = indexed ? 'multiDrawIndexedIndirect' : 'multiDrawIndirect';
    const native =
      Boolean(options.useNative) &&
      stride === recordSize &&
      Boolean(pass) &&
      typeof pass[nativeMethod] === 'function';

    const reqCountFallback = isNonNegSafeIntFast(options.count) && options.count > 0 ? options.count : 1;
    if (!isNonNegSafeIntFast(firstSlot)) {
      recordRangeRejection(RANGE_INVALID_OFFSET, statsTarget, reqCountFallback);
      if (skipInvalid) {
        return { calls: 0, draws: 0, skipped: isNonNegSafeIntFast(options.count) ? options.count : 0, native };
      }
      throw new RangeError(`firstSlot must be a non-negative integer, got ${firstSlot}`);
    }
    if (!isNonNegSafeIntFast(baseOffset)) {
      recordRangeRejection(RANGE_INVALID_OFFSET, statsTarget, reqCountFallback);
      if (skipInvalid) {
        return { calls: 0, draws: 0, skipped: isNonNegSafeIntFast(options.count) ? options.count : 0, native };
      }
      throw new RangeError(`baseOffset must be a non-negative integer, got ${baseOffset}`);
    }
    if (!isPosSafeIntFast(stride) || stride < recordSize || !isAligned4Fast(stride)) {
      if (isPosSafeIntFast(stride) && !isAligned4Fast(stride)) {
        recordRangeRejection(RANGE_MISALIGNED_OFFSET, statsTarget, reqCountFallback);
      } else {
        recordBoundsRejection(statsTarget);
      }
      if (skipInvalid) {
        return { calls: 0, draws: 0, skipped: isNonNegSafeIntFast(options.count) ? options.count : 0, native };
      }
      throw new RangeError(`stride must be a multiple of 4 and >= ${recordSize}, got ${stride}`);
    }

    let batch;
    if (skipInvalid) {
      const requested = options.count;
      const maxBatch =
        options.maxIndirectBatchCount !== undefined
          ? options.maxIndirectBatchCount
          : DEFAULT_MAX_INDIRECT_BATCH_COUNT;
      if (!isNonNegSafeIntFast(requested) || !isPosSafeIntFast(maxBatch)) {
        recordBoundsRejection(statsTarget);
        return { calls: 0, draws: 0, skipped: 0, native };
      }
      if (requested > maxBatch) {
        if (options.clampBatch) {
          recordBatchClamped(statsTarget);
          batch = { count: maxBatch, requestedCount: requested, clamped: true };
        } else {
          recordBatchLimitRejected(statsTarget);
          return { calls: 0, draws: 0, skipped: requested, native };
        }
      } else {
        batch = { count: requested, requestedCount: requested, clamped: false };
      }
    } else {
      batch = resolveBatchCount(options.count, options, statsTarget);
    }

    if (batch.count === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      return { calls: 0, draws: 0, skipped: 0, native };
    }

    const firstOffset = baseOffset + firstSlot * stride;
    const explicitSize = options.bufferSize;
    let effectiveCount = batch.count;
    let clampedSkipped = 0;

    if (clampToBuffer) {
      const baseStatus = fastCheckIndirectRange(buffer, firstOffset, recordSize, explicitSize);
      if (baseStatus !== RANGE_OK && baseStatus !== RANGE_OUT_OF_BOUNDS) {
        recordRangeRejection(baseStatus, statsTarget, effectiveCount, skipInvalid);
        if (skipInvalid) {
          return { calls: 0, draws: 0, skipped: effectiveCount, native };
        }
        throwForRangeCode(baseStatus, buffer, firstOffset, recordSize, explicitSize);
      }
      const effectiveSize = resolveEffectiveBufferSizeFast(buffer, explicitSize);
      if (effectiveSize >= 0 && isNonNegSafeIntFast(firstOffset)) {
        const maxFit = maxRecordsFittingBuffer(effectiveSize, firstOffset, stride, recordSize);
        if (effectiveCount > maxFit) {
          clampedSkipped = effectiveCount - maxFit;
          effectiveCount = maxFit;
          recordStat('skippedOutOfBoundsDraws', clampedSkipped, statsTarget);
          recordStat('outOfBoundsElided', clampedSkipped, statsTarget);
          if (!batch.clamped) {
            if (maxFit === 0) {
              recordBoundsRejection(statsTarget);
            } else {
              recordBatchClamped(statsTarget);
            }
          }
          if (effectiveCount === 0) {
            recordStat('commandsValidated', clampedSkipped, statsTarget);
            return { calls: 0, draws: 0, skipped: clampedSkipped, native };
          }
        }
      }
    }

    const hasVisibility = options.visibility !== undefined && options.visibility !== null;
    const shadowSource =
      options.elideZeroCount !== false ? prepareShadowSource(options.shadowData) : null;

    if (!hasVisibility && !shadowSource) {
      const spanBytes = stride * (effectiveCount - 1) + recordSize;
      const status = fastCheckIndirectRange(buffer, firstOffset, spanBytes, explicitSize);
      if (status !== RANGE_OK) {
        recordRangeRejection(status, statsTarget, effectiveCount, skipInvalid);
        recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
        if (skipInvalid) {
          return { calls: 0, draws: 0, skipped: effectiveCount + clampedSkipped, native };
        }
        throwForRangeCode(status, buffer, firstOffset, spanBytes, explicitSize);
      }
      if (native) {
        pass[nativeMethod](buffer, firstOffset, effectiveCount);
        recordEncodedKind(kind, effectiveCount, statsTarget);
        recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
        recordStat('batchesDispatched', 1, statsTarget);
        recordMaxStat('maxMultiDrawCount', effectiveCount, statsTarget);
        return { calls: 1, draws: effectiveCount, skipped: clampedSkipped, native: true };
      }
      for (let i = 0, offset = firstOffset; i < effectiveCount; i += 1, offset += stride) {
        pass[method](buffer, offset);
      }
      recordEncodedKind(kind, effectiveCount, statsTarget);
      recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
      recordStat('batchesDispatched', 1, statsTarget);
      recordMaxStat('maxMultiDrawCount', effectiveCount, statsTarget);
      return { calls: effectiveCount, draws: effectiveCount, skipped: clampedSkipped, native: false };
    }

    // Visibility-masked and/or CPU shadow zero-count elision path: validate bounds atomically
    // before emitting any pass commands.
    const visibility = hasVisibility ? options.visibility : null;
    let maxVisIdx = effectiveCount - 1;
    if (hasVisibility) {
      maxVisIdx = -1;
      for (let i = effectiveCount - 1; i >= 0; i -= 1) {
        if (visibleAt(visibility, i)) {
          maxVisIdx = i;
          break;
        }
      }
      if (maxVisIdx === -1) {
        recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
        return { calls: 0, draws: 0, skipped: effectiveCount + clampedSkipped, native };
      }
    }

    const maxVisOffset = firstOffset + maxVisIdx * stride;
    const status = fastCheckIndirectRange(buffer, maxVisOffset, recordSize, explicitSize);
    if (status !== RANGE_OK) {
      recordRangeRejection(status, statsTarget, effectiveCount, skipInvalid);
      recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
      if (skipInvalid) {
        return { calls: 0, draws: 0, skipped: effectiveCount + clampedSkipped, native };
      }
      throwForRangeCode(status, buffer, maxVisOffset, recordSize, explicitSize);
    }

    let calls = 0;
    let draws = 0;
    let zeroSkipped = 0;
    let skipped = clampedSkipped + (effectiveCount - 1 - maxVisIdx);

    if (native) {
      let i = 0;
      while (i <= maxVisIdx) {
        const isVis = !hasVisibility || visibleAt(visibility, i);
        const isZero = isVis && shadowSource !== null && isZeroCountShadowAt(kind, shadowSource, firstOffset + i * stride);
        if (!isVis || isZero) {
          if (isZero) {
            zeroSkipped += 1;
          }
          skipped += 1;
          i += 1;
          continue;
        }
        let runEnd = i + 1;
        while (runEnd <= maxVisIdx) {
          if (hasVisibility && !visibleAt(visibility, runEnd)) {
            break;
          }
          if (shadowSource !== null && isZeroCountShadowAt(kind, shadowSource, firstOffset + runEnd * stride)) {
            break;
          }
          runEnd += 1;
        }
        const runCount = runEnd - i;
        pass[nativeMethod](buffer, firstOffset + i * stride, runCount);
        calls += 1;
        draws += runCount;
        i = runEnd;
      }
    } else {
      for (let i = 0, offset = firstOffset; i <= maxVisIdx; i += 1, offset += stride) {
        if (hasVisibility && !visibleAt(visibility, i)) {
          skipped += 1;
          continue;
        }
        if (shadowSource !== null && isZeroCountShadowAt(kind, shadowSource, offset)) {
          zeroSkipped += 1;
          skipped += 1;
          continue;
        }
        pass[method](buffer, offset);
        calls += 1;
        draws += 1;
      }
    }

    recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
    if (zeroSkipped > 0) {
      recordStat('skippedZeroCountDraws', zeroSkipped, statsTarget);
      recordStat('zeroCountElided', zeroSkipped, statsTarget);
    }
    if (draws > 0) {
      recordEncodedKind(kind, draws, statsTarget);
      recordStat('batchesDispatched', 1, statsTarget);
      recordMaxStat('maxMultiDrawCount', draws, statsTarget);
    }
    return { calls, draws, skipped, native };
  }

  /**
   * Issue `dispatchWorkgroupsIndirect` for `count` consecutive slots on a compute pass encoder,
   * validating offset alignment, stride, buffer bounds, and `maxIndirectBatchCount` before
   * invoking WebGPU pass commands.
   *
   * @param {{dispatchWorkgroupsIndirect: Function}} pass GPUComputePassEncoder.
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect buffer.
   * @param {object} options
   * @param {number} options.count
   * @param {number} [options.firstSlot=0]
   * @param {number} [options.baseOffset=0]
   * @param {number} [options.stride=12] Bytes between slots (>= 12, multiple of 4).
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {ArrayLike<boolean|number>} [options.visibility] CPU mask indexed from `firstSlot`; falsy skips.
   * @param {number} [options.maxIndirectBatchCount=4096] Maximum dispatches allowed per batch.
   * @param {boolean} [options.clampBatch=false] Clamp oversized batches instead of throwing.
   * @param {boolean} [options.clampToBuffer=false] Clamp count to records fitting within known buffer size.
   * @param {boolean} [options.skipInvalid=false] Skip invalid ranges/args instead of throwing.
   * @returns {{calls:number, dispatches:number, skipped:number, clamped:boolean}}
   */
  function encodeMultiDispatchIndirect(pass, buffer, options) {
    const hasOpts = options !== undefined && options !== null && typeof options === 'object';
    const skipInvalid = hasOpts && Boolean(options.skipInvalid);
    const statsTarget = hasOpts ? options._statsTarget || null : null;
    if (!pass || typeof pass.dispatchWorkgroupsIndirect !== 'function') {
      if (skipInvalid) {
        recordBoundsRejection(statsTarget);
        return { calls: 0, dispatches: 0, skipped: 0, clamped: false };
      }
      throw new TypeError('pass must implement dispatchWorkgroupsIndirect');
    }
    if (!hasOpts) {
      throw new TypeError('options must be an object');
    }
    const recordSize = DISPATCH_INDIRECT_BYTES;
    const stride = options.stride === undefined ? recordSize : options.stride;
    const firstSlot = options.firstSlot || 0;
    const baseOffset = options.baseOffset || 0;
    const clampToBuffer = Boolean(options.clampToBuffer);

    const reqCountFallback = isNonNegSafeIntFast(options.count) && options.count > 0 ? options.count : 1;
    if (!isNonNegSafeIntFast(firstSlot)) {
      recordRangeRejection(RANGE_INVALID_OFFSET, statsTarget, reqCountFallback);
      if (skipInvalid) {
        return {
          calls: 0,
          dispatches: 0,
          skipped: isNonNegSafeIntFast(options.count) ? options.count : 0,
          clamped: false,
        };
      }
      throw new RangeError(`firstSlot must be a non-negative integer, got ${firstSlot}`);
    }
    if (!isNonNegSafeIntFast(baseOffset)) {
      recordRangeRejection(RANGE_INVALID_OFFSET, statsTarget, reqCountFallback);
      if (skipInvalid) {
        return {
          calls: 0,
          dispatches: 0,
          skipped: isNonNegSafeIntFast(options.count) ? options.count : 0,
          clamped: false,
        };
      }
      throw new RangeError(`baseOffset must be a non-negative integer, got ${baseOffset}`);
    }
    if (!isPosSafeIntFast(stride) || stride < recordSize || !isAligned4Fast(stride)) {
      if (isPosSafeIntFast(stride) && !isAligned4Fast(stride)) {
        recordRangeRejection(RANGE_MISALIGNED_OFFSET, statsTarget, reqCountFallback);
      } else {
        recordBoundsRejection(statsTarget);
      }
      if (skipInvalid) {
        return {
          calls: 0,
          dispatches: 0,
          skipped: isNonNegSafeIntFast(options.count) ? options.count : 0,
          clamped: false,
        };
      }
      throw new RangeError(`stride must be a multiple of 4 and >= ${recordSize}, got ${stride}`);
    }

    let batch;
    if (skipInvalid) {
      const requested = options.count;
      const maxBatch =
        options.maxIndirectBatchCount !== undefined
          ? options.maxIndirectBatchCount
          : DEFAULT_MAX_INDIRECT_BATCH_COUNT;
      if (!isNonNegSafeIntFast(requested) || !isPosSafeIntFast(maxBatch)) {
        recordBoundsRejection(statsTarget);
        return { calls: 0, dispatches: 0, skipped: 0, clamped: false };
      }
      if (requested > maxBatch) {
        if (options.clampBatch) {
          recordBatchClamped(statsTarget);
          batch = { count: maxBatch, requestedCount: requested, clamped: true };
        } else {
          recordBatchLimitRejected(statsTarget);
          return { calls: 0, dispatches: 0, skipped: requested, clamped: false };
        }
      } else {
        batch = { count: requested, requestedCount: requested, clamped: false };
      }
    } else {
      batch = resolveBatchCount(options.count, options, statsTarget);
    }

    if (batch.count === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      return { calls: 0, dispatches: 0, skipped: 0, clamped: false };
    }

    const firstOffset = baseOffset + firstSlot * stride;
    const explicitSize = options.bufferSize;
    let effectiveCount = batch.count;
    let clamped = batch.clamped;
    let clampedSkipped = 0;

    if (clampToBuffer) {
      const baseStatus = fastCheckIndirectRange(buffer, firstOffset, recordSize, explicitSize);
      if (baseStatus !== RANGE_OK && baseStatus !== RANGE_OUT_OF_BOUNDS) {
        recordRangeRejection(baseStatus, statsTarget, effectiveCount, skipInvalid);
        if (skipInvalid) {
          return { calls: 0, dispatches: 0, skipped: effectiveCount, clamped };
        }
        throwForRangeCode(baseStatus, buffer, firstOffset, recordSize, explicitSize);
      }
      const effectiveSize = resolveEffectiveBufferSizeFast(buffer, explicitSize);
      if (effectiveSize >= 0 && isNonNegSafeIntFast(firstOffset)) {
        const maxFit = maxRecordsFittingBuffer(effectiveSize, firstOffset, stride, recordSize);
        if (effectiveCount > maxFit) {
          clampedSkipped = effectiveCount - maxFit;
          effectiveCount = maxFit;
          recordStat('skippedOutOfBoundsDraws', clampedSkipped, statsTarget);
          recordStat('outOfBoundsElided', clampedSkipped, statsTarget);
          if (!clamped) {
            if (maxFit === 0) {
              recordBoundsRejection(statsTarget);
            } else {
              recordBatchClamped(statsTarget);
            }
          }
          clamped = true;
          if (effectiveCount === 0) {
            recordStat('commandsValidated', clampedSkipped, statsTarget);
            return { calls: 0, dispatches: 0, skipped: clampedSkipped, clamped: true };
          }
        }
      }
    }

    const hasVisibility = options.visibility !== undefined && options.visibility !== null;
    const shadowSource =
      options.elideZeroCount !== false ? prepareShadowSource(options.shadowData) : null;

    if (!hasVisibility && !shadowSource) {
      const spanBytes = stride * (effectiveCount - 1) + recordSize;
      const status = fastCheckIndirectRange(buffer, firstOffset, spanBytes, explicitSize);
      if (status !== RANGE_OK) {
        recordRangeRejection(status, statsTarget, effectiveCount, skipInvalid);
        recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
        if (skipInvalid) {
          return {
            calls: 0,
            dispatches: 0,
            skipped: effectiveCount + clampedSkipped,
            clamped,
          };
        }
        throwForRangeCode(status, buffer, firstOffset, spanBytes, explicitSize);
      }
      for (let i = 0, offset = firstOffset; i < effectiveCount; i += 1, offset += stride) {
        pass.dispatchWorkgroupsIndirect(buffer, offset);
      }
      recordEncodedKind(INDIRECT_KIND.DISPATCH, effectiveCount, statsTarget);
      recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
      recordStat('batchesDispatched', 1, statsTarget);
      recordMaxStat('maxMultiDrawCount', effectiveCount, statsTarget);
      return {
        calls: effectiveCount,
        dispatches: effectiveCount,
        skipped: clampedSkipped,
        clamped,
      };
    }

    const visibility = hasVisibility ? options.visibility : null;
    let maxVisIdx = effectiveCount - 1;
    if (hasVisibility) {
      maxVisIdx = -1;
      for (let i = effectiveCount - 1; i >= 0; i -= 1) {
        if (visibleAt(visibility, i)) {
          maxVisIdx = i;
          break;
        }
      }
      if (maxVisIdx === -1) {
        recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
        return { calls: 0, dispatches: 0, skipped: effectiveCount + clampedSkipped, clamped };
      }
    }

    const maxVisOffset = firstOffset + maxVisIdx * stride;
    const status = fastCheckIndirectRange(buffer, maxVisOffset, recordSize, explicitSize);
    if (status !== RANGE_OK) {
      recordRangeRejection(status, statsTarget, effectiveCount, skipInvalid);
      recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
      if (skipInvalid) {
        return {
          calls: 0,
          dispatches: 0,
          skipped: effectiveCount + clampedSkipped,
          clamped,
        };
      }
      throwForRangeCode(status, buffer, maxVisOffset, recordSize, explicitSize);
    }

    let dispatches = 0;
    let zeroSkipped = 0;
    let skipped = clampedSkipped + (effectiveCount - 1 - maxVisIdx);
    for (let i = 0, offset = firstOffset; i <= maxVisIdx; i += 1, offset += stride) {
      if (hasVisibility && !visibleAt(visibility, i)) {
        skipped += 1;
        continue;
      }
      if (shadowSource !== null && isZeroCountShadowAt(INDIRECT_KIND.DISPATCH, shadowSource, offset)) {
        zeroSkipped += 1;
        skipped += 1;
        continue;
      }
      pass.dispatchWorkgroupsIndirect(buffer, offset);
      dispatches += 1;
    }
    recordStat('commandsValidated', effectiveCount + clampedSkipped, statsTarget);
    if (zeroSkipped > 0) {
      recordStat('skippedZeroCountDraws', zeroSkipped, statsTarget);
      recordStat('zeroCountElided', zeroSkipped, statsTarget);
    }
    if (dispatches > 0) {
      recordEncodedKind(INDIRECT_KIND.DISPATCH, dispatches, statsTarget);
      recordStat('batchesDispatched', 1, statsTarget);
      recordMaxStat('maxMultiDrawCount', dispatches, statsTarget);
    }
    return { calls: dispatches, dispatches, skipped, clamped };
  }

  /**
   * Safe-by-default D3D12 `ExecuteIndirect` fast-path executor for `draw`, `drawIndexed`,
   * and `dispatch`. Validates argument buffer usage, liveness, 4-byte offset alignment,
   * stride, and bounds before issuing any WebGPU pass command; steps offsets (`argOffset + i * stride`)
   * for multi-draw (`count > 1`) batches, clamping or skipping invalid arguments cleanly
   * instead of triggering WebGPU validation errors.
   *
   * @param {object} pass GPURenderPassEncoder or GPUComputePassEncoder.
   * @param {string} kind INDIRECT_KIND ('draw' | 'drawIndexed' | 'dispatch').
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect argument buffer.
   * @param {number} [argOffset=0] Byte offset of the first indirect command.
   * @param {object} [options]
   * @param {number} [options.count=1] Number of commands (`maxCommandCount` alias accepted).
   * @param {number} [options.maxCommandCount] Alias for `count`.
   * @param {number} [options.stride] Byte stride between commands (default record size: 16 / 20 / 12).
   * @param {number} [options.firstSlot=0] Starting slot index added to `argOffset`.
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {number} [options.maxIndirectBatchCount=4096] Maximum commands per ExecuteIndirect batch.
   * @param {boolean} [options.clampBatch=true] Clamp oversized counts instead of rejecting.
   * @param {boolean} [options.clampToBuffer=true] Clamp counts overflowing bufferSize to fitting records.
   * @param {ArrayLike<boolean|number>} [options.visibility] Optional CPU visibility mask.
   * @param {boolean} [options.useNative=false] Use native multiDraw* when tightly packed & available.
   * @param {boolean} [options.strict=false] Throw on invalid arguments instead of skipping.
   * @returns {{
   *   ok: boolean,
   *   executed: number,
   *   calls: number,
   *   requestedCount: number,
   *   clamped: boolean,
   *   skipped: number,
   *   reason: string | null,
   *   native: boolean,
   *   firstOffset: number,
   *   stride: number
   * }}
   */
  function writeExecuteResult(
    target,
    ok,
    executed,
    calls,
    requestedCount,
    clamped,
    skipped,
    reason,
    native,
    firstOffset,
    stride
  ) {
    if (target !== null) {
      target.ok = ok;
      target.executed = executed;
      target.calls = calls;
      target.requestedCount = requestedCount;
      target.clamped = clamped;
      target.skipped = skipped;
      target.reason = reason;
      target.native = native;
      target.firstOffset = firstOffset;
      target.stride = stride;
      return target;
    }
    return {
      ok,
      executed,
      calls,
      requestedCount,
      clamped,
      skipped,
      reason,
      native,
      firstOffset,
      stride,
    };
  }

  function executeIndirect(pass, kind, buffer, argOffset = 0, options, out) {
    const layout = LAYOUTS[kind];
    const hasOpts = options !== undefined && options !== null;
    const outTarget = resolveOutTarget(options, out);
    const statsTarget = hasOpts ? options._statsTarget || null : null;
    const strict = hasOpts && Boolean(options.strict);

    if (!layout) {
      recordBoundsRejection(statsTarget);
      if (strict) {
        throw new TypeError(`unknown indirect kind '${kind}'`);
      }
      return writeExecuteResult(
        outTarget,
        false,
        0,
        0,
        0,
        false,
        0,
        'UNKNOWN_KIND',
        false,
        0,
        0
      );
    }

    const recordSize = layout.byteSize;
    const method =
      kind === INDIRECT_KIND.DRAW
        ? 'drawIndirect'
        : kind === INDIRECT_KIND.DRAW_INDEXED
          ? 'drawIndexedIndirect'
          : 'dispatchWorkgroupsIndirect';
    const nativeMethod =
      kind === INDIRECT_KIND.DRAW
        ? 'multiDrawIndirect'
        : kind === INDIRECT_KIND.DRAW_INDEXED
          ? 'multiDrawIndexedIndirect'
          : null;

    if (!pass || typeof pass[method] !== 'function') {
      recordBoundsRejection(statsTarget);
      if (strict) {
        throw new TypeError(`pass must implement ${method}`);
      }
      const reqFallback = hasOpts && isNonNegSafeIntFast(options.count) ? options.count : 1;
      return writeExecuteResult(
        outTarget,
        false,
        0,
        0,
        reqFallback,
        false,
        reqFallback,
        'MISSING_PASS_METHOD',
        false,
        isNonNegSafeIntFast(argOffset) ? argOffset : 0,
        recordSize
      );
    }

    const requestedCount = hasOpts
      ? options.count !== undefined
        ? options.count
        : options.maxCommandCount !== undefined
          ? options.maxCommandCount
          : 1
      : 1;
    const firstSlot = hasOpts && options.firstSlot !== undefined ? options.firstSlot : 0;
    const stride = hasOpts && options.stride !== undefined ? options.stride : recordSize;
    const explicitSize = hasOpts ? options.bufferSize : undefined;
    const clampBatch = !hasOpts || options.clampBatch !== false;
    const clampToBuffer = !hasOpts || options.clampToBuffer !== false;
    const maxBatch =
      hasOpts && options.maxIndirectBatchCount !== undefined
        ? options.maxIndirectBatchCount
        : DEFAULT_MAX_INDIRECT_BATCH_COUNT;

    // Ultra-fast zero-allocation path for the common single-command ExecuteIndirect (count === 1,
    // firstSlot === 0, no visibility mask).
    const hasVisibility = hasOpts && options.visibility !== undefined && options.visibility !== null;
    const elideZeroCount = !hasOpts || options.elideZeroCount !== false;
    const shadowSource =
      hasOpts && elideZeroCount && options.shadowData !== undefined && options.shadowData !== null
        ? prepareShadowSource(options.shadowData)
        : null;
    if (
      requestedCount === 1 &&
      firstSlot === 0 &&
      stride === recordSize &&
      !hasVisibility &&
      isPosSafeIntFast(maxBatch)
    ) {
      recordStat('commandsValidated', 1, statsTarget);
      const status = fastCheckIndirectRange(buffer, argOffset, recordSize, explicitSize);
      if (status === RANGE_OK) {
        if (
          hasOpts &&
          elideZeroCount &&
          (hasZeroCountOption(kind, options) || isZeroCountShadowAt(kind, shadowSource, argOffset))
        ) {
          recordStat('skippedZeroCountDraws', 1, statsTarget);
          recordStat('zeroCountElided', 1, statsTarget);
          return writeExecuteResult(
            outTarget,
            true,
            0,
            0,
            1,
            false,
            1,
            'ZERO_COUNT',
            false,
            argOffset,
            recordSize
          );
        }
        pass[method](buffer, argOffset);
        recordEncodedKind(kind, 1, statsTarget);
        return writeExecuteResult(
          outTarget,
          true,
          1,
          1,
          1,
          false,
          0,
          null,
          false,
          argOffset,
          recordSize
        );
      }
      recordRangeRejection(status, statsTarget, 1, !strict);
      if (strict) {
        throwForRangeCode(status, buffer, argOffset, recordSize, explicitSize);
      }
      return writeExecuteResult(
        outTarget,
        false,
        0,
        0,
        1,
        false,
        1,
        RANGE_CODE_TO_REASON[status],
        false,
        isNonNegSafeIntFast(argOffset) ? argOffset : 0,
        recordSize
      );
    }

    const plan = computeIndirectBatchStepPlan(buffer, {
      kind,
      count: requestedCount,
      firstSlot,
      baseOffset: argOffset,
      stride,
      bufferSize: explicitSize,
      maxIndirectBatchCount: maxBatch,
      clampBatch,
      clampToBuffer,
    });

    if (!plan.valid) {
      const skippedCount = plan.requestedCount > 0 ? plan.requestedCount : 1;
      recordStat('commandsValidated', skippedCount, statsTarget);
      if (plan.reason === 'BATCH_LIMIT_EXCEEDED') {
        recordBatchLimitRejected(statsTarget);
      } else if (
        plan.reason === 'MISALIGNED_OFFSET' ||
        (plan.reason === 'INVALID_STRIDE' && isPosSafeIntFast(stride) && !isAligned4Fast(stride))
      ) {
        recordRangeRejection(RANGE_MISALIGNED_OFFSET, statsTarget, skippedCount, !strict);
      } else if (plan.reason === 'OUT_OF_BOUNDS' || plan.reason === 'INVALID_OFFSET') {
        recordRangeRejection(RANGE_OUT_OF_BOUNDS, statsTarget, skippedCount, !strict);
      } else {
        recordBoundsRejection(statsTarget);
      }
      if (strict) {
        if (plan.reason === 'INVALID_OFFSET' || plan.reason === 'INVALID_STRIDE' || plan.reason === 'INVALID_COUNT' || plan.reason === 'INVALID_BATCH_LIMIT') {
          throw new RangeError(`executeIndirect ${kind} failed: ${plan.reason}`);
        }
        throw new IndirectDrawError(plan.reason, `executeIndirect ${kind} failed: ${plan.reason}`);
      }
      return writeExecuteResult(
        outTarget,
        false,
        0,
        0,
        plan.requestedCount,
        false,
        plan.requestedCount,
        plan.reason,
        false,
        plan.firstOffset,
        plan.stride
      );
    }

    if (plan.effectiveCount === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      return writeExecuteResult(
        outTarget,
        true,
        0,
        0,
        0,
        false,
        0,
        null,
        false,
        plan.firstOffset,
        plan.stride
      );
    }

    let oobClamped = 0;
    if (plan.clamped) {
      recordBatchClamped(statsTarget);
      const countAfterBatchClamp =
        clampBatch && plan.requestedCount > maxBatch ? maxBatch : plan.requestedCount;
      oobClamped = countAfterBatchClamp - plan.effectiveCount;
      if (oobClamped > 0) {
        recordStat('skippedOutOfBoundsDraws', oobClamped, statsTarget);
        recordStat('outOfBoundsElided', oobClamped, statsTarget);
      }
    }

    const clampedSkipped = plan.requestedCount - plan.effectiveCount;
    recordStat('commandsValidated', plan.effectiveCount + oobClamped, statsTarget);
    if (hasOpts && elideZeroCount && hasZeroCountOption(kind, options)) {
      recordStat('skippedZeroCountDraws', plan.effectiveCount, statsTarget);
      recordStat('zeroCountElided', plan.effectiveCount, statsTarget);
      return writeExecuteResult(
        outTarget,
        true,
        0,
        0,
        plan.requestedCount,
        plan.clamped,
        clampedSkipped + plan.effectiveCount,
        plan.clampedReason || 'ZERO_COUNT',
        false,
        plan.firstOffset,
        plan.stride
      );
    }

    const native =
      Boolean(hasOpts && options.useNative) &&
      nativeMethod !== null &&
      plan.stride === recordSize &&
      typeof pass[nativeMethod] === 'function';

    let calls = 0;
    let executed = 0;
    let zeroSkipped = 0;
    let skipped = clampedSkipped;

    if (!hasVisibility && !shadowSource) {
      if (native) {
        pass[nativeMethod](buffer, plan.firstOffset, plan.effectiveCount);
        calls = 1;
        executed = plan.effectiveCount;
      } else {
        for (
          let i = 0, offset = plan.firstOffset;
          i < plan.effectiveCount;
          i += 1, offset += plan.stride
        ) {
          pass[method](buffer, offset);
        }
        calls = plan.effectiveCount;
        executed = plan.effectiveCount;
      }
    } else {
      const visibility = hasVisibility ? options.visibility : null;
      if (native) {
        let i = 0;
        while (i < plan.effectiveCount) {
          const isVis = !hasVisibility || visibleAt(visibility, i);
          const isZero =
            isVis &&
            shadowSource !== null &&
            isZeroCountShadowAt(kind, shadowSource, plan.firstOffset + i * plan.stride);
          if (!isVis || isZero) {
            if (isZero) {
              zeroSkipped += 1;
            }
            skipped += 1;
            i += 1;
            continue;
          }
          let runEnd = i + 1;
          while (runEnd < plan.effectiveCount) {
            if (hasVisibility && !visibleAt(visibility, runEnd)) {
              break;
            }
            if (
              shadowSource !== null &&
              isZeroCountShadowAt(kind, shadowSource, plan.firstOffset + runEnd * plan.stride)
            ) {
              break;
            }
            runEnd += 1;
          }
          const runCount = runEnd - i;
          pass[nativeMethod](buffer, plan.firstOffset + i * plan.stride, runCount);
          calls += 1;
          executed += runCount;
          i = runEnd;
        }
      } else {
        for (
          let i = 0, offset = plan.firstOffset;
          i < plan.effectiveCount;
          i += 1, offset += plan.stride
        ) {
          if (hasVisibility && !visibleAt(visibility, i)) {
            skipped += 1;
            continue;
          }
          if (shadowSource !== null && isZeroCountShadowAt(kind, shadowSource, offset)) {
            zeroSkipped += 1;
            skipped += 1;
            continue;
          }
          pass[method](buffer, offset);
          calls += 1;
          executed += 1;
        }
      }
    }

    if (zeroSkipped > 0) {
      recordStat('skippedZeroCountDraws', zeroSkipped, statsTarget);
      recordStat('zeroCountElided', zeroSkipped, statsTarget);
    }
    if (executed > 0) {
      recordEncodedKind(kind, executed, statsTarget);
      if (plan.requestedCount > 1 || executed > 1) {
        recordStat('batchesDispatched', 1, statsTarget);
        recordMaxStat('maxMultiDrawCount', executed, statsTarget);
      }
    }

    return writeExecuteResult(
      outTarget,
      true,
      executed,
      calls,
      plan.requestedCount,
      plan.clamped,
      skipped,
      plan.clampedReason || (executed === 0 && zeroSkipped > 0 ? 'ZERO_COUNT' : null),
      native,
      plan.firstOffset,
      plan.stride
    );
  }

  /**
   * Safe-by-default `ExecuteIndirect` helper for non-indexed `drawIndirect` (16-byte record).
   */
  function executeIndirectDraw(pass, buffer, argOffset = 0, options, out) {
    return executeIndirect(pass, INDIRECT_KIND.DRAW, buffer, argOffset, options, out);
  }

  /**
   * Safe-by-default `ExecuteIndirect` helper for indexed `drawIndexedIndirect` (20-byte record).
   */
  function executeIndirectDrawIndexed(pass, buffer, argOffset = 0, options, out) {
    return executeIndirect(pass, INDIRECT_KIND.DRAW_INDEXED, buffer, argOffset, options, out);
  }

  /**
   * Safe-by-default `ExecuteIndirect` helper for compute `dispatchWorkgroupsIndirect` (12-byte record).
   */
  function executeIndirectDispatch(pass, buffer, argOffset = 0, options, out) {
    return executeIndirect(pass, INDIRECT_KIND.DISPATCH, buffer, argOffset, options, out);
  }

  function writeDispatchResult(
    target,
    ok,
    dispatched,
    executed,
    skipped,
    elided,
    reason,
    offset,
    byteSize,
    indexed
  ) {
    if (target !== null) {
      target.ok = ok;
      target.dispatched = dispatched;
      target.executed = executed;
      target.skipped = skipped;
      target.elided = elided;
      target.reason = reason;
      target.offset = offset;
      target.byteSize = byteSize;
      target.indexed = indexed;
      return target;
    }
    return {
      ok,
      dispatched,
      executed,
      skipped,
      elided,
      reason,
      offset,
      byteSize,
      indexed,
    };
  }

  function dispatchIndirectSingle(pass, kind, buffer, indirectOffset, options, out) {
    const indexed = kind === INDIRECT_KIND.DRAW_INDEXED;
    const recordSize = indexed ? DRAW_INDEXED_INDIRECT_BYTES : DRAW_INDIRECT_BYTES;
    const method = indexed ? 'drawIndexedIndirect' : 'drawIndirect';
    const hasOpts = options !== undefined && options !== null;
    const outTarget = resolveOutTarget(options, out);
    const statsTarget = hasOpts ? options._statsTarget || null : null;
    const strict = hasOpts && Boolean(options.strict);

    recordStat('commandsValidated', 1, statsTarget);

    if (hasOpts && options.count === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      recordStat('zeroCountElided', 1, statsTarget);
      return writeDispatchResult(
        outTarget,
        true,
        false,
        0,
        1,
        true,
        'ZERO_COUNT',
        indirectOffset,
        recordSize,
        indexed
      );
    }

    if (!pass || typeof pass[method] !== 'function') {
      recordBoundsRejection(statsTarget);
      if (strict) {
        throw new TypeError(`pass must implement ${method}`);
      }
      return writeDispatchResult(
        outTarget,
        false,
        false,
        0,
        1,
        true,
        'MISSING_PASS_METHOD',
        indirectOffset,
        recordSize,
        indexed
      );
    }

    const explicitSize = hasOpts ? options.bufferSize : undefined;
    const status = fastCheckIndirectRange(buffer, indirectOffset, recordSize, explicitSize);
    if (status !== RANGE_OK) {
      recordRangeRejection(status, statsTarget, 1, !strict);
      if (strict) {
        throwForRangeCode(status, buffer, indirectOffset, recordSize, explicitSize);
      }
      return writeDispatchResult(
        outTarget,
        false,
        false,
        0,
        1,
        true,
        RANGE_CODE_TO_REASON[status],
        indirectOffset,
        recordSize,
        indexed
      );
    }

    if (
      hasOpts &&
      options.elideZeroCount !== false &&
      (hasZeroCountOption(kind, options) ||
        isZeroCountShadowAt(kind, options.shadowData, indirectOffset))
    ) {
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      recordStat('zeroCountElided', 1, statsTarget);
      return writeDispatchResult(
        outTarget,
        true,
        false,
        0,
        1,
        true,
        'ZERO_COUNT',
        indirectOffset,
        recordSize,
        indexed
      );
    }

    pass[method](buffer, indirectOffset);
    recordEncodedKind(kind, 1, statsTarget);
    return writeDispatchResult(
      outTarget,
      true,
      true,
      1,
      0,
      false,
      null,
      indirectOffset,
      recordSize,
      indexed
    );
  }

  /**
   * Hot-path zero-allocation indirect draw dispatcher (16-byte DrawIndirectArgs).
   * Preserves 100% of W269 alignment, usage, liveness, bounds, and zero-count checks.
   * When `out` (via 5th arg or `options.out`) is provided, mutates and returns `out`
   * with zero per-draw object allocations.
   *
   * @param {{drawIndirect: Function}} pass GPURenderPassEncoder.
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect buffer.
   * @param {number} [indirectOffset=0]
   * @param {object} [options]
   * @param {object} [out] Optional pre-allocated scratch result object.
   */
  function dispatchIndirectDraw(pass, buffer, indirectOffset = 0, options, out) {
    return dispatchIndirectSingle(pass, INDIRECT_KIND.DRAW, buffer, indirectOffset, options, out);
  }

  /**
   * Hot-path zero-allocation indexed indirect draw dispatcher (20-byte DrawIndexedIndirectArgs).
   * Preserves 100% of W269 alignment, usage, liveness, bounds, and zero-count checks.
   * When `out` (via 5th arg or `options.out`) is provided, mutates and returns `out`
   * with zero per-draw object allocations.
   *
   * @param {{drawIndexedIndirect: Function}} pass GPURenderPassEncoder.
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect buffer.
   * @param {number} [indirectOffset=0]
   * @param {object} [options]
   * @param {object} [out] Optional pre-allocated scratch result object.
   */
  function dispatchIndexedIndirectDraw(pass, buffer, indirectOffset = 0, options, out) {
    return dispatchIndirectSingle(
      pass,
      INDIRECT_KIND.DRAW_INDEXED,
      buffer,
      indirectOffset,
      options,
      out
    );
  }

  function writeBatchDispatchResult(
    target,
    ok,
    dispatched,
    calls,
    validated,
    requestedCount,
    clamped,
    skipped,
    zeroCountElided,
    outOfBoundsElided,
    visibilitySkipped,
    reason,
    native,
    firstOffset,
    stride
  ) {
    if (target !== null) {
      target.ok = ok;
      target.dispatched = dispatched;
      target.executed = dispatched;
      target.calls = calls;
      target.validated = validated;
      target.requestedCount = requestedCount;
      target.clamped = clamped;
      target.skipped = skipped;
      target.zeroCountElided = zeroCountElided;
      target.outOfBoundsElided = outOfBoundsElided;
      target.visibilitySkipped = visibilitySkipped;
      target.reason = reason;
      target.native = native;
      target.firstOffset = firstOffset;
      target.stride = stride;
      return target;
    }
    return {
      ok,
      dispatched,
      executed: dispatched,
      calls,
      validated,
      requestedCount,
      clamped,
      skipped,
      zeroCountElided,
      outOfBoundsElided,
      visibilitySkipped,
      reason,
      native,
      firstOffset,
      stride,
    };
  }

  /**
   * Single-pass, zero-per-command-allocation batched multi-draw-indirect validation & dispatch.
   * Validates consecutive indirect command structs (`firstOffset + i * stride`) in one pass,
   * cleanly eliding zero-count and out-of-bounds commands while preserving 100% of W269 checks.
   * Supports an optional pre-allocated `out` scratch result object (via 4th arg or `options.out`).
   *
   * @param {object} pass GPURenderPassEncoder or GPUComputePassEncoder.
   * @param {{usage:number, size?:number, destroyed?:boolean}} buffer Indirect buffer.
   * @param {object} [options]
   * @param {boolean} [options.indexed=false] Use 20-byte drawIndexedIndirect records when `kind` omitted.
   * @param {string} [options.kind] Explicit INDIRECT_KIND ('draw' | 'drawIndexed' | 'dispatch').
   * @param {number} [options.count] Number of consecutive indirect command structs.
   * @param {number} [options.maxCommandCount] Alias for `count`.
   * @param {number} [options.baseOffset=0] Byte offset of slot 0 (`offset` / `argOffset` aliases accepted).
   * @param {number} [options.firstSlot=0] Starting command slot index.
   * @param {number} [options.stride] Byte stride between consecutive structs (default record size).
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {ArrayBuffer|DataView|ArrayBufferView} [options.shadowData] Optional CPU shadow bytes for zero-count elision.
   * @param {object[]} [options.records] Optional CPU record array for zero-count elision.
   * @param {ArrayLike<boolean|number>} [options.visibility] Optional CPU visibility mask.
   * @param {boolean} [options.elideZeroCount=true] Elide zero-count commands cleanly.
   * @param {boolean} [options.useNative=false] Coalesce contiguous valid runs into native multiDraw* calls when available.
   * @param {number} [options.maxIndirectBatchCount=4096] Maximum commands per batch.
   * @param {boolean} [options.clampBatch=true] Clamp oversized batches instead of rejecting.
   * @param {boolean} [options.strict=false] Throw on fatal pass/buffer/alignment/stride errors.
   * @param {object} [options.out] Optional pre-allocated scratch result object.
   * @param {object} [out] Optional pre-allocated scratch result object.
   */
  function dispatchIndirectBatch(pass, buffer, options, out) {
    const hasOpts = options !== undefined && options !== null && typeof options === 'object';
    const outTarget = resolveOutTarget(options, out);
    const statsTarget = hasOpts ? options._statsTarget || null : null;
    const strict = hasOpts && Boolean(options.strict);

    const kind =
      hasOpts && options.kind !== undefined
        ? options.kind
        : hasOpts && options.indexed
          ? INDIRECT_KIND.DRAW_INDEXED
          : INDIRECT_KIND.DRAW;
    const layout = LAYOUTS[kind];
    if (!layout) {
      recordBoundsRejection(statsTarget);
      if (strict) {
        throw new TypeError(`unknown indirect kind '${kind}'`);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        0,
        0,
        false,
        0,
        0,
        0,
        0,
        'UNKNOWN_KIND',
        false,
        0,
        0
      );
    }

    const recordSize = layout.byteSize;
    const method =
      kind === INDIRECT_KIND.DRAW
        ? 'drawIndirect'
        : kind === INDIRECT_KIND.DRAW_INDEXED
          ? 'drawIndexedIndirect'
          : 'dispatchWorkgroupsIndirect';
    const nativeMethod =
      kind === INDIRECT_KIND.DRAW
        ? 'multiDrawIndirect'
        : kind === INDIRECT_KIND.DRAW_INDEXED
          ? 'multiDrawIndexedIndirect'
          : null;

    const records = hasOpts && Array.isArray(options.records) ? options.records : null;
    const requestedCount = hasOpts
      ? options.count !== undefined
        ? options.count
        : options.maxCommandCount !== undefined
          ? options.maxCommandCount
          : records !== null
            ? records.length
            : 0
      : 0;
    const baseOffset = hasOpts
      ? options.baseOffset !== undefined
        ? options.baseOffset
        : options.offset !== undefined
          ? options.offset
          : options.argOffset !== undefined
            ? options.argOffset
            : 0
      : 0;
    const firstSlot = hasOpts && options.firstSlot !== undefined ? options.firstSlot : 0;
    const stride = hasOpts && options.stride !== undefined ? options.stride : recordSize;
    const maxBatch =
      hasOpts && options.maxIndirectBatchCount !== undefined
        ? options.maxIndirectBatchCount
        : DEFAULT_MAX_INDIRECT_BATCH_COUNT;
    const clampBatch = !hasOpts || options.clampBatch !== false;
    const explicitSize = hasOpts ? options.bufferSize : undefined;

    if (!pass || typeof pass[method] !== 'function') {
      recordBoundsRejection(statsTarget);
      if (strict) {
        throw new TypeError(`pass must implement ${method}`);
      }
      const reqValid = isNonNegSafeIntFast(requestedCount) ? requestedCount : 0;
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        0,
        reqValid,
        false,
        reqValid,
        0,
        0,
        0,
        'MISSING_PASS_METHOD',
        false,
        isNonNegSafeIntFast(baseOffset) ? baseOffset : 0,
        isPosSafeIntFast(stride) ? stride : recordSize
      );
    }

    if (!isNonNegSafeIntFast(requestedCount)) {
      recordBoundsRejection(statsTarget);
      if (strict) {
        throw new RangeError(`count must be a non-negative integer, got ${requestedCount}`);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        0,
        0,
        false,
        0,
        0,
        0,
        0,
        'INVALID_COUNT',
        false,
        isNonNegSafeIntFast(baseOffset) ? baseOffset : 0,
        isPosSafeIntFast(stride) ? stride : recordSize
      );
    }

    if (!isPosSafeIntFast(maxBatch)) {
      recordBoundsRejection(statsTarget);
      if (strict) {
        throw new RangeError(`maxIndirectBatchCount must be a positive integer, got ${maxBatch}`);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        0,
        requestedCount,
        false,
        requestedCount,
        0,
        0,
        0,
        'INVALID_BATCH_LIMIT',
        false,
        isNonNegSafeIntFast(baseOffset) ? baseOffset : 0,
        isPosSafeIntFast(stride) ? stride : recordSize
      );
    }

    if (!isNonNegSafeIntFast(firstSlot) || !isNonNegSafeIntFast(baseOffset)) {
      const skippedCount = requestedCount > 0 ? requestedCount : 1;
      recordRangeRejection(RANGE_INVALID_OFFSET, statsTarget, skippedCount, !strict);
      recordStat('commandsValidated', requestedCount, statsTarget);
      if (strict) {
        throw new RangeError(`baseOffset/firstSlot must be non-negative integers`);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        requestedCount,
        requestedCount,
        false,
        requestedCount,
        0,
        requestedCount,
        0,
        'INVALID_OFFSET',
        false,
        0,
        isPosSafeIntFast(stride) ? stride : recordSize
      );
    }

    if (!isPosSafeIntFast(stride) || stride < recordSize || !isAligned4Fast(stride)) {
      const skippedCount = requestedCount > 0 ? requestedCount : 1;
      if (isPosSafeIntFast(stride) && !isAligned4Fast(stride)) {
        recordRangeRejection(RANGE_MISALIGNED_OFFSET, statsTarget, skippedCount, false);
      } else {
        recordBoundsRejection(statsTarget);
      }
      if (strict) {
        throw new RangeError(`stride must be a multiple of 4 and >= ${recordSize}, got ${stride}`);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        0,
        requestedCount,
        false,
        requestedCount,
        0,
        0,
        0,
        'INVALID_STRIDE',
        false,
        baseOffset,
        0
      );
    }

    const firstOffset = baseOffset + firstSlot * stride;
    if (!isNonNegSafeIntFast(firstOffset)) {
      const skippedCount = requestedCount > 0 ? requestedCount : 1;
      recordRangeRejection(RANGE_OUT_OF_BOUNDS, statsTarget, skippedCount, !strict);
      recordStat('commandsValidated', requestedCount, statsTarget);
      if (strict) {
        throw new IndirectDrawError('OUT_OF_BOUNDS', 'indirect batch firstOffset overflow');
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        requestedCount,
        requestedCount,
        false,
        requestedCount,
        0,
        requestedCount,
        0,
        'OUT_OF_BOUNDS',
        false,
        0,
        stride
      );
    }

    if (!isAligned4Fast(firstOffset)) {
      const skippedCount = requestedCount > 0 ? requestedCount : 1;
      recordRangeRejection(RANGE_MISALIGNED_OFFSET, statsTarget, skippedCount, false);
      recordStat('commandsValidated', requestedCount, statsTarget);
      if (strict) {
        throwForRangeCode(RANGE_MISALIGNED_OFFSET, buffer, firstOffset, recordSize, explicitSize);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        requestedCount,
        requestedCount,
        false,
        requestedCount,
        0,
        0,
        0,
        'MISALIGNED_OFFSET',
        false,
        firstOffset,
        stride
      );
    }

    if (requestedCount === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      recordStat('skippedZeroCountDraws', 1, statsTarget);
      return writeBatchDispatchResult(
        outTarget,
        true,
        0,
        0,
        0,
        0,
        false,
        0,
        0,
        0,
        0,
        null,
        false,
        firstOffset,
        stride
      );
    }

    if (!buffer || typeof buffer !== 'object' || (buffer.usage & BUFFER_USAGE.INDIRECT) === 0) {
      recordRangeRejection(RANGE_NOT_INDIRECT, statsTarget, requestedCount, false);
      recordStat('commandsValidated', requestedCount, statsTarget);
      if (strict) {
        throwForRangeCode(RANGE_NOT_INDIRECT, buffer, firstOffset, recordSize, explicitSize);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        requestedCount,
        requestedCount,
        false,
        requestedCount,
        0,
        0,
        0,
        'NOT_INDIRECT',
        false,
        firstOffset,
        stride
      );
    }

    if (buffer.destroyed === true) {
      recordRangeRejection(RANGE_DESTROYED_BUFFER, statsTarget, requestedCount, false);
      recordStat('commandsValidated', requestedCount, statsTarget);
      if (strict) {
        throwForRangeCode(RANGE_DESTROYED_BUFFER, buffer, firstOffset, recordSize, explicitSize);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        requestedCount,
        requestedCount,
        false,
        requestedCount,
        0,
        0,
        0,
        'DESTROYED_BUFFER',
        false,
        firstOffset,
        stride
      );
    }

    const effectiveSize = resolveEffectiveBufferSizeFast(buffer, explicitSize);
    if (effectiveSize === -2 || effectiveSize === -3) {
      const code = effectiveSize === -2 ? RANGE_INVALID_OPT_BUFFER_SIZE : RANGE_INVALID_BUFFER_SIZE;
      recordRangeRejection(code, statsTarget, requestedCount, !strict);
      recordStat('commandsValidated', requestedCount, statsTarget);
      if (strict) {
        throwForRangeCode(code, buffer, firstOffset, recordSize, explicitSize);
      }
      return writeBatchDispatchResult(
        outTarget,
        false,
        0,
        0,
        requestedCount,
        requestedCount,
        false,
        requestedCount,
        0,
        requestedCount,
        0,
        'OUT_OF_BOUNDS',
        false,
        firstOffset,
        stride
      );
    }

    let effectiveCount = requestedCount;
    let clampedByBatch = false;
    if (effectiveCount > maxBatch) {
      if (!clampBatch) {
        recordBatchLimitRejected(statsTarget);
        if (strict) {
          throw new IndirectDrawError(
            'BATCH_LIMIT_EXCEEDED',
            `indirect batch count ${requestedCount} exceeds maxIndirectBatchCount ${maxBatch}`
          );
        }
        return writeBatchDispatchResult(
          outTarget,
          false,
          0,
          0,
          0,
          requestedCount,
          false,
          requestedCount,
          0,
          0,
          0,
          'BATCH_LIMIT_EXCEEDED',
          false,
          firstOffset,
          stride
        );
      }
      effectiveCount = maxBatch;
      clampedByBatch = true;
      recordBatchClamped(statsTarget);
    }

    const hasVisibility = hasOpts && options.visibility !== undefined && options.visibility !== null;
    const visibility = hasVisibility ? options.visibility : null;
    const elideZeroCount = !hasOpts || options.elideZeroCount !== false;
    const shadowSource =
      hasOpts && elideZeroCount && options.shadowData !== undefined && options.shadowData !== null
        ? prepareShadowSource(options.shadowData)
        : null;
    const batchZeroOption = hasOpts && elideZeroCount && hasZeroCountOption(kind, options);
    const native =
      Boolean(hasOpts && options.useNative) &&
      nativeMethod !== null &&
      stride === recordSize &&
      typeof pass[nativeMethod] === 'function';

    let dispatched = 0;
    let calls = 0;
    let validated = 0;
    let zeroElided = 0;
    let oobElided = 0;
    let visSkipped = 0;
    let runStartOffset = 0;
    let runCount = 0;

    for (let i = 0, offset = firstOffset; i < effectiveCount; i += 1, offset += stride) {
      const end = offset + recordSize;
      if (
        (effectiveSize >= 0 && end > effectiveSize) ||
        (end > 0x7fffffff && !Number.isSafeInteger(end))
      ) {
        if (native && runCount > 0) {
          pass[nativeMethod](buffer, runStartOffset, runCount);
          calls += 1;
          dispatched += runCount;
          runCount = 0;
        }
        const remainingOob = effectiveCount - i;
        oobElided += remainingOob;
        validated += remainingOob;
        break;
      }

      validated += 1;

      if (hasVisibility && !visibleAt(visibility, i)) {
        if (native && runCount > 0) {
          pass[nativeMethod](buffer, runStartOffset, runCount);
          calls += 1;
          dispatched += runCount;
          runCount = 0;
        }
        visSkipped += 1;
        continue;
      }

      if (
        elideZeroCount &&
        (batchZeroOption ||
          (shadowSource !== null && isZeroCountShadowAt(kind, shadowSource, offset)) ||
          (records !== null && i < records.length && isZeroCountRecord(kind, records[i])))
      ) {
        if (native && runCount > 0) {
          pass[nativeMethod](buffer, runStartOffset, runCount);
          calls += 1;
          dispatched += runCount;
          runCount = 0;
        }
        zeroElided += 1;
        continue;
      }

      if (native) {
        if (runCount === 0) {
          runStartOffset = offset;
          runCount = 1;
        } else {
          runCount += 1;
        }
      } else {
        pass[method](buffer, offset);
        calls += 1;
        dispatched += 1;
      }
    }

    if (native && runCount > 0) {
      pass[nativeMethod](buffer, runStartOffset, runCount);
      calls += 1;
      dispatched += runCount;
      runCount = 0;
    }

    recordStat('commandsValidated', validated, statsTarget);
    if (zeroElided > 0) {
      recordStat('skippedZeroCountDraws', zeroElided, statsTarget);
      recordStat('zeroCountElided', zeroElided, statsTarget);
    }
    const allOob = oobElided === effectiveCount && dispatched === 0 && zeroElided === 0 && visSkipped === 0;
    if (oobElided > 0) {
      recordStat('skippedOutOfBoundsDraws', oobElided, statsTarget);
      recordStat('outOfBoundsElided', oobElided, statsTarget);
      if (allOob) {
        recordBoundsRejection(statsTarget);
      } else if (!clampedByBatch) {
        recordBatchClamped(statsTarget);
      }
    }
    if (dispatched > 0) {
      recordEncodedKind(kind, dispatched, statsTarget);
      recordStat('batchesDispatched', 1, statsTarget);
      recordMaxStat('maxMultiDrawCount', dispatched, statsTarget);
    }

    const clampedByBuffer = oobElided > 0 && !allOob;
    const clamped = clampedByBatch || clampedByBuffer;
    const reason = allOob
      ? 'OUT_OF_BOUNDS'
      : clampedByBatch && clampedByBuffer
        ? 'CLAMPED_TO_BATCH_LIMIT_AND_BUFFER'
        : clampedByBatch
          ? 'CLAMPED_TO_BATCH_LIMIT'
          : clampedByBuffer
            ? 'CLAMPED_TO_BUFFER'
            : dispatched === 0 && zeroElided > 0
              ? 'ZERO_COUNT'
              : null;

    return writeBatchDispatchResult(
      outTarget,
      !allOob,
      dispatched,
      calls,
      validated,
      requestedCount,
      clamped,
      requestedCount - dispatched,
      zeroElided,
      oobElided,
      visSkipped,
      reason,
      native,
      firstOffset,
      stride
    );
  }

  /**
   * Fixed-record indirect argument buffer with stable per-draw slots, for compute-driven
   * culling. The GPU buffer has `INDIRECT | STORAGE | COPY_DST` (+ `COPY_SRC` by default,
   * needed for growth and debug readback). Slot `i` is at byte offset `i * stride`
   * (stride == record size). A culling compute shader binds `getStorageBinding()` as
   * `var<storage, read_write> args: array<DrawIndexedIndirectArgs>` (see wgslStructSource)
   * and writes `args[i].instance_count = 0` to cull; the CPU never reads that back, so
   * `getArgs` returns the last CPU-written values.
   */
  class IndirectArgsBuffer {
    /**
     * @param {object} options
     * @param {GPUDevice} options.device
     * @param {string} [options.kind='drawIndexed'] INDIRECT_KIND.DRAW or DRAW_INDEXED.
     * @param {number} [options.capacity=256] Initial draw capacity (slots).
     * @param {boolean} [options.copySrc=true] Add COPY_SRC (required for growth).
     * @param {boolean} [options.autoGrow=true] Grow on allocate when full.
     * @param {number} [options.growthFactor=2]
     * @param {number} [options.mergeGap=0] Dirty ranges closer than this many bytes are merged.
     * @param {boolean} [options.indirectFirstInstance=false]
     * @param {boolean} [options.patchFirstInstance=false]
     * @param {string} [options.label]
     * @param {(info:{buffer:object, oldBuffer:object, capacity:number, oldCapacity:number}) => void} [options.onGrow]
     */
    constructor(options) {
      if (!options || !options.device) {
        throw new TypeError('IndirectArgsBuffer requires a device');
      }
      const kind = options.kind || INDIRECT_KIND.DRAW_INDEXED;
      if (kind !== INDIRECT_KIND.DRAW && kind !== INDIRECT_KIND.DRAW_INDEXED) {
        throw new TypeError(`kind must be 'draw' or 'drawIndexed', got '${kind}'`);
      }
      const capacity = options.capacity === undefined ? 256 : options.capacity;
      assertPositiveInteger(capacity, 'capacity');
      this.device = options.device;
      this.kind = kind;
      this.stride = recordByteSize(kind);
      this.copySrc = options.copySrc !== false;
      this.autoGrow = options.autoGrow !== false;
      this.growthFactor = options.growthFactor === undefined ? 2 : options.growthFactor;
      if (!(this.growthFactor > 1)) {
        throw new RangeError('growthFactor must be > 1');
      }
      this.mergeGap = options.mergeGap || 0;
      this.indirectFirstInstance = Boolean(options.indirectFirstInstance);
      this.patchFirstInstance = Boolean(options.patchFirstInstance);
      const maxIndirectBatchCount =
        options.maxIndirectBatchCount === undefined
          ? DEFAULT_MAX_INDIRECT_BATCH_COUNT
          : options.maxIndirectBatchCount;
      assertPositiveInteger(maxIndirectBatchCount, 'maxIndirectBatchCount');
      this.maxIndirectBatchCount = maxIndirectBatchCount;
      this.clampBatch = Boolean(options.clampBatch);
      this.label = options.label || 'indirect-args';
      this.onGrow = options.onGrow || null;
      this.capacity = capacity;
      this.shadow = new Uint8Array(capacity * this.stride);
      this.buffer = this._createBuffer(capacity);
      this.freeSlots = [];
      this.nextSlot = 0;
      this.live = new Set();
      this.dirty = [];
      this.destroyed = false;
      this._stats = createEmptyStats();
    }

    _createBuffer(capacity) {
      let usage = BUFFER_USAGE.INDIRECT | BUFFER_USAGE.STORAGE | BUFFER_USAGE.COPY_DST;
      if (this.copySrc) {
        usage |= BUFFER_USAGE.COPY_SRC;
      }
      return this.device.createBuffer({ label: this.label, size: capacity * this.stride, usage });
    }

    _assertAlive() {
      if (this.destroyed) {
        throw new IndirectDrawError('DESTROYED', 'IndirectArgsBuffer is destroyed');
      }
    }

    _assertLive(slot) {
      assertNonNegativeInteger(slot, 'slot');
      if (!this.live.has(slot)) {
        throw new IndirectDrawError('BAD_SLOT', `slot ${slot} is not allocated`);
      }
    }

    /** Number of currently allocated slots. */
    get liveCount() {
      return this.live.size;
    }

    /**
     * Byte offset of a slot in the GPU buffer; stable across frames and growth.
     * @param {number} slot
     * @returns {number}
     */
    slotOffset(slot) {
      assertNonNegativeInteger(slot, 'slot');
      return slot * this.stride;
    }

    /**
     * Allocate a slot (reusing freed slots first, most recently freed first), optionally
     * writing initial args. Without args the record is zeroed so a stale record cannot draw.
     *
     * @param {object} [record] Initial args.
     * @returns {number} Slot index.
     */
    allocate(record) {
      this._assertAlive();
      let slot;
      if (this.freeSlots.length > 0) {
        slot = this.freeSlots.pop();
      } else {
        if (this.nextSlot >= this.capacity) {
          if (!this.autoGrow) {
            throw new IndirectDrawError('FULL', `indirect args buffer is full (${this.capacity} slots)`);
          }
          this.grow(Math.max(this.nextSlot + 1, Math.ceil(this.capacity * this.growthFactor)));
        }
        slot = this.nextSlot;
        this.nextSlot += 1;
      }
      this.live.add(slot);
      if (record) {
        this.setArgs(slot, record);
      } else {
        this.shadow.fill(0, slot * this.stride, (slot + 1) * this.stride);
        this._markDirty(slot * this.stride, this.stride);
      }
      return slot;
    }

    /**
     * Release a slot for reuse. Its record is zeroed (instanceCount 0) on the next flush.
     * @param {number} slot
     */
    free(slot) {
      this._assertAlive();
      this._assertLive(slot);
      this.live.delete(slot);
      this.freeSlots.push(slot);
      this.shadow.fill(0, slot * this.stride, (slot + 1) * this.stride);
      this._markDirty(slot * this.stride, this.stride);
    }

    /**
     * Write CPU args for a slot (staged; call flush() to upload). Applies the
     * `indirect-first-instance` policy configured on the buffer.
     *
     * @param {number} slot
     * @param {object} record
     */
    setArgs(slot, record) {
      this._assertAlive();
      this._assertLive(slot);
      const { bytes } = encodeBatch(this.kind, [record], {
        indirectFirstInstance: this.indirectFirstInstance,
        patchFirstInstance: this.patchFirstInstance,
      });
      this.shadow.set(bytes, slot * this.stride);
      this._markDirty(slot * this.stride, this.stride);
    }

    /**
     * Last CPU-written args of a slot (GPU-side culling edits are not reflected).
     * @param {number} slot
     * @returns {object}
     */
    getArgs(slot) {
      this._assertLive(slot);
      return decodeRecord(this.kind, this.shadow, slot * this.stride);
    }

    _markDirty(start, length) {
      this.dirty.push({ start, end: start + length });
    }

    /**
     * Upload dirty ranges with `queue.writeBuffer`, coalescing overlapping, adjacent (and,
     * with `mergeGap`, nearby) ranges into one write each.
     *
     * @returns {number} Number of writeBuffer calls issued.
     */
    flush() {
      this._assertAlive();
      if (this.dirty.length === 0) {
        return 0;
      }
      const sorted = this.dirty.sort((a, b) => a.start - b.start);
      const merged = [];
      for (const range of sorted) {
        const last = merged[merged.length - 1];
        if (last && range.start <= last.end + this.mergeGap) {
          last.end = Math.max(last.end, range.end);
        } else {
          merged.push({ start: range.start, end: range.end });
        }
      }
      this.dirty = [];
      for (const range of merged) {
        this.device.queue.writeBuffer(
          this.buffer,
          range.start,
          this.shadow.subarray(range.start, range.end)
        );
      }
      return merged.length;
    }

    /**
     * Storage binding info for the culling compute shader (whole buffer, offset 0).
     * Binding `size` is capacity * stride (a multiple of 4).
     *
     * @returns {{buffer: object, offset: number, size: number, stride: number, instanceCountOffset: number, bindGroupLayoutEntry: object, wgsl: string}}
     */
    getStorageBinding() {
      this._assertAlive();
      return {
        buffer: this.buffer,
        offset: 0,
        size: this.capacity * this.stride,
        stride: this.stride,
        instanceCountOffset: layoutFor(this.kind).fields.instanceCount.offset,
        bindGroupLayoutEntry: { buffer: { type: 'storage' } },
        wgsl: wgslStructSource(this.kind),
      };
    }

    /**
     * Grow to at least `minCapacity` slots: allocate a bigger buffer, copy the old contents
     * on the GPU (so culling results survive), submit, destroy the old buffer, and notify
     * `onGrow` so bind groups referencing the old buffer can be rebuilt. Slots keep indices.
     * Pending dirty ranges stay pending and target the new buffer on flush.
     *
     * @param {number} minCapacity
     * @returns {boolean} True when a reallocation happened.
     */
    grow(minCapacity) {
      this._assertAlive();
      assertPositiveInteger(minCapacity, 'minCapacity');
      if (minCapacity <= this.capacity) {
        return false;
      }
      if (!this.copySrc) {
        throw new IndirectDrawError('GROW_UNSUPPORTED', 'growth requires copySrc: true');
      }
      const oldBuffer = this.buffer;
      const oldCapacity = this.capacity;
      const newBuffer = this._createBuffer(minCapacity);
      const encoder = this.device.createCommandEncoder({ label: `${this.label}-grow` });
      encoder.copyBufferToBuffer(oldBuffer, 0, newBuffer, 0, oldCapacity * this.stride);
      this.device.queue.submit([encoder.finish()]);
      const shadow = new Uint8Array(minCapacity * this.stride);
      shadow.set(this.shadow);
      this.shadow = shadow;
      this.buffer = newBuffer;
      this.capacity = minCapacity;
      oldBuffer.destroy();
      if (this.onGrow) {
        this.onGrow({ buffer: newBuffer, oldBuffer, capacity: minCapacity, oldCapacity });
      }
      return true;
    }

    /**
     * Encode draws for `count` slots from `firstSlot` on a render pass (see encodeMultiDraw).
     *
     * @param {object} pass GPURenderPassEncoder.
     * @param {object} [options] firstSlot, count (default: all slots handed out), visibility, useNative, maxIndirectBatchCount, clampBatch.
     * @returns {{calls:number, draws:number, skipped:number, native:boolean}}
     */
    encodeDraws(pass, options = {}) {
      this._assertAlive();
      const firstSlot = options.firstSlot || 0;
      assertNonNegativeInteger(firstSlot, 'firstSlot');
      const count = options.count === undefined ? Math.max(0, this.nextSlot - firstSlot) : options.count;
      return encodeMultiDraw(pass, this.buffer, {
        ...options,
        indexed: this.kind === INDIRECT_KIND.DRAW_INDEXED,
        firstSlot,
        count,
        stride: this.stride,
        shadowData:
          options.shadowData !== undefined
            ? options.shadowData
            : options.elideZeroCount === true
              ? this.shadow
              : undefined,
        maxIndirectBatchCount:
          options.maxIndirectBatchCount === undefined
            ? this.maxIndirectBatchCount
            : options.maxIndirectBatchCount,
        clampBatch: options.clampBatch === undefined ? this.clampBatch : options.clampBatch,
        _statsTarget: this._stats,
      });
    }

    /**
     * Safe-by-default `ExecuteIndirect` draw execution covering `count` slots from `firstSlot`
     * (see `executeIndirect`).
     *
     * @param {object} pass GPURenderPassEncoder.
     * @param {object} [options] firstSlot, count (default: handed-out slots), visibility, useNative, maxIndirectBatchCount, clampBatch, clampToBuffer, strict, elideZeroCount.
     */
    executeDraws(pass, options = {}) {
      this._assertAlive();
      const firstSlot = options.firstSlot || 0;
      const count =
        options.count === undefined ? Math.max(0, this.nextSlot - firstSlot) : options.count;
      return executeIndirect(pass, this.kind, this.buffer, 0, {
        ...options,
        firstSlot,
        count,
        stride: this.stride,
        shadowData:
          options.shadowData !== undefined
            ? options.shadowData
            : options.elideZeroCount === true
              ? this.shadow
              : undefined,
        maxIndirectBatchCount:
          options.maxIndirectBatchCount === undefined
            ? this.maxIndirectBatchCount
            : options.maxIndirectBatchCount,
        clampBatch: options.clampBatch === undefined ? true : options.clampBatch,
        clampToBuffer: options.clampToBuffer === undefined ? true : options.clampToBuffer,
        _statsTarget: this._stats,
      });
    }

    /**
     * Return a sanitized numeric snapshot of this buffer's indirect execution telemetry.
     *
     * @returns {{
     *   indirectDrawsEncoded: number,
     *   indirectIndexedDrawsEncoded: number,
     *   indirectDispatchesEncoded: number,
     *   boundsRejectedCount: number,
     *   zeroBatchSkips: number,
     *   batchLimitRejectedCount: number,
     *   batchClampedCount: number,
     *   indirectDraws: number,
     *   indirectIndexedDraws: number,
     *   indirectDispatches: number,
     *   clampedOrSkippedInvalid: number,
     *   maxMultiDrawCount: number,
     *   validatedIndirectDraws: number,
     *   skippedOutOfBoundsDraws: number,
     *   skippedMisalignedDraws: number,
     *   skippedZeroCountDraws: number,
     *   batchesDispatched: number,
     *   commandsValidated: number,
     *   zeroCountElided: number,
     *   outOfBoundsElided: number,
     *   liveSlots: number,
     *   capacity: number,
     *   stride: number,
     *   maxIndirectBatchCount: number,
     *   destroyed: boolean
     * }}
     */
    getStats() {
      return {
        indirectDrawsEncoded: this._stats.indirectDrawsEncoded,
        indirectIndexedDrawsEncoded: this._stats.indirectIndexedDrawsEncoded,
        indirectDispatchesEncoded: this._stats.indirectDispatchesEncoded,
        boundsRejectedCount: this._stats.boundsRejectedCount,
        zeroBatchSkips: this._stats.zeroBatchSkips,
        batchLimitRejectedCount: this._stats.batchLimitRejectedCount,
        batchClampedCount: this._stats.batchClampedCount,
        indirectDraws: this._stats.indirectDraws,
        indirectIndexedDraws: this._stats.indirectIndexedDraws,
        indirectDispatches: this._stats.indirectDispatches,
        clampedOrSkippedInvalid: this._stats.clampedOrSkippedInvalid,
        maxMultiDrawCount: this._stats.maxMultiDrawCount,
        validatedIndirectDraws: this._stats.validatedIndirectDraws,
        skippedOutOfBoundsDraws: this._stats.skippedOutOfBoundsDraws,
        skippedMisalignedDraws: this._stats.skippedMisalignedDraws,
        skippedZeroCountDraws: this._stats.skippedZeroCountDraws,
        batchesDispatched: this._stats.batchesDispatched,
        commandsValidated: this._stats.commandsValidated,
        zeroCountElided: this._stats.zeroCountElided,
        outOfBoundsElided: this._stats.outOfBoundsElided,
        gpuClampPassesEncoded: this._stats.gpuClampPassesEncoded,
        gpuClampCommandsClamped: this._stats.gpuClampCommandsClamped,
        gpuClampPipelinesCreated: this._stats.gpuClampPipelinesCreated,
        gpuClampPipelineCacheHits: this._stats.gpuClampPipelineCacheHits,
        gpuClampFirstInstanceZeroed: this._stats.gpuClampFirstInstanceZeroed,
        gpuClampCountBuffersBound: this._stats.gpuClampCountBuffersBound,
        meshOrchestrationsEncoded: this._stats.meshOrchestrationsEncoded,
        meshIndirectDispatches: this._stats.meshIndirectDispatches,
        meshZeroCountElided: this._stats.meshZeroCountElided,
        liveSlots: this.live.size,
        capacity: this.capacity,
        stride: this.stride,
        maxIndirectBatchCount: this.maxIndirectBatchCount,
        destroyed: this.destroyed,
      };
    }

    /**
     * Reset this buffer's indirect execution telemetry counters to zero.
     */
    resetStats() {
      this._stats = createEmptyStats();
    }

    /** Destroy the GPU buffer; further use throws. Idempotent. */
    destroy() {
      if (this.destroyed) {
        return;
      }
      this.destroyed = true;
      this.buffer.destroy();
      this.dirty = [];
      this.live.clear();
      this.freeSlots = [];
    }
  }

  // ---------------------------------------------------------------------------
  // W378: GPU-Clamped ExecuteIndirect Compute Pass & Mesh Compute-Then-Draw
  // ---------------------------------------------------------------------------

  const _deviceClampPipelineCache = new WeakMap();
  const _deviceDefaultClampExecutors = new WeakMap();

  /**
   * Check whether a WebGPU device exposes the `indirect-first-instance` feature.
   *
   * @param {object | null | undefined} device
   * @returns {boolean}
   */
  function deviceSupportsIndirectFirstInstance(device) {
    if (!device || typeof device !== 'object') {
      return false;
    }
    if (device.features && typeof device.features.has === 'function') {
      return Boolean(device.features.has('indirect-first-instance'));
    }
    return false;
  }

  /**
   * Resolve `maxComputeWorkgroupsPerDimension` from a WebGPU device (default 65535).
   *
   * @param {object | null | undefined} device
   * @param {number} [overrideLimit]
   * @returns {number}
   */
  function resolveMaxComputeWorkgroups(device, overrideLimit) {
    if (isPosSafeIntFast(overrideLimit)) {
      return overrideLimit;
    }
    const raw =
      device &&
      device.limits &&
      typeof device.limits.maxComputeWorkgroupsPerDimension === 'number'
        ? Math.floor(device.limits.maxComputeWorkgroupsPerDimension)
        : 0;
    return isPosSafeIntFast(raw) ? raw : DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION;
  }

  function clampInputWordCount(kind) {
    if (kind === INDIRECT_CLAMP_KIND.DRAW) return 4;
    if (kind === INDIRECT_CLAMP_KIND.DRAW_INDEXED) return 5;
    if (kind === INDIRECT_CLAMP_KIND.DISPATCH) return 3;
    if (kind === INDIRECT_CLAMP_KIND.DISPATCH_MESH) return 3;
    throw new TypeError(`unknown indirect clamp kind '${kind}'`);
  }

  function clampOutputWordCount(kind) {
    if (kind === INDIRECT_CLAMP_KIND.DRAW) return 4;
    if (kind === INDIRECT_CLAMP_KIND.DRAW_INDEXED) return 5;
    if (kind === INDIRECT_CLAMP_KIND.DISPATCH) return 3;
    if (kind === INDIRECT_CLAMP_KIND.DISPATCH_MESH) return 7;
    throw new TypeError(`unknown indirect clamp kind '${kind}'`);
  }

  function clampKindNumericId(kind) {
    if (kind === INDIRECT_CLAMP_KIND.DRAW) return 0;
    if (kind === INDIRECT_CLAMP_KIND.DRAW_INDEXED) return 1;
    if (kind === INDIRECT_CLAMP_KIND.DISPATCH) return 2;
    if (kind === INDIRECT_CLAMP_KIND.DISPATCH_MESH) return 3;
    throw new TypeError(`unknown indirect clamp kind '${kind}'`);
  }

  /**
   * Generate a WGSL compute shader that clamps GPU-written indirect argument records
   * (`drawIndirect` 16B, `drawIndexedIndirect` 20B, `dispatchIndirect` 12B, or
   * `dispatchMesh` 12B -> 28B dispatch+draw pair) into a validated output indirect buffer
   * without CPU readbacks, mirroring the G66 (#1533 / #1537) compute clamp design.
   *
   * @param {string} kind One of INDIRECT_CLAMP_KIND ('draw' | 'drawIndexed' | 'dispatch' | 'dispatchMesh').
   * @param {{ workgroupSize?: number }} [options]
   * @returns {string}
   */
  function generateIndirectClampWgsl(kind, options) {
    const inWords = clampInputWordCount(kind);
    const outWords = clampOutputWordCount(kind);
    const wgSize =
      options && options.workgroupSize !== undefined
        ? options.workgroupSize
        : DEFAULT_CLAMP_WORKGROUP_SIZE;
    if (!Number.isInteger(wgSize) || wgSize < 1 || wgSize > 256) {
      throw new RangeError(`workgroupSize must be an integer in [1, 256], got ${wgSize}`);
    }

    const header = [
      'struct ClampParams {',
      '  src_word: u32,',
      '  stride_words: u32,',
      '  dst_word: u32,',
      '  command_count: u32,',
      '  count_word: u32,',
      '  has_count_buffer: u32,',
      '  max_element_count: u32,',
      '  max_instance_count: u32,',
      '  max_group_x: u32,',
      '  max_group_y: u32,',
      '  max_group_z: u32,',
      '  flags: u32,',
      '}',
      '@group(0) @binding(0) var<uniform> params: ClampParams;',
      '@group(0) @binding(1) var<storage, read> src_args: array<u32>;',
      '@group(0) @binding(2) var<storage, read> count_buf: array<u32>;',
      '@group(0) @binding(3) var<storage, read_write> dst_args: array<u32>;',
      '',
      `@compute @workgroup_size(${wgSize})`,
      'fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {',
      '  let i = gid.x;',
      '  if (i >= params.command_count) {',
      '    return;',
      '  }',
      `  let in_words = ${inWords}u;`,
      `  let out_words = ${outWords}u;`,
      '  let out_base = params.dst_word + i * out_words;',
      '  let dst_len = arrayLength(&dst_args);',
      '  if (out_base >= dst_len || out_words > dst_len - out_base) {',
      '    return;',
      '  }',
      '  var effective_count = params.command_count;',
      '  if (params.has_count_buffer != 0u) {',
      '    effective_count = 0u;',
      '    if (params.count_word < arrayLength(&count_buf)) {',
      '      effective_count = min(count_buf[params.count_word], params.command_count);',
      '    }',
      '  }',
      '  let src_base = params.src_word + i * params.stride_words;',
      '  let src_len = arrayLength(&src_args);',
      '  let live = i < effective_count && src_base < src_len && in_words <= src_len - src_base;',
      '  if (!live) {',
      '    for (var k = 0u; k < out_words; k = k + 1u) {',
      '      dst_args[out_base + k] = 0u;',
      '    }',
      '    return;',
      '  }',
    ];

    let body;
    if (kind === INDIRECT_CLAMP_KIND.DRAW) {
      body = [
        '  let first_vertex = src_args[src_base + 2u];',
        '  var vertex_count = 0u;',
        '  if (first_vertex < params.max_element_count) {',
        '    vertex_count = min(src_args[src_base + 0u], params.max_element_count - first_vertex);',
        '  }',
        '  let instance_count = min(src_args[src_base + 1u], params.max_instance_count);',
        '  var first_instance = src_args[src_base + 3u];',
        '  if ((params.flags & 1u) != 0u) {',
        '    first_instance = 0u;',
        '  }',
        '  dst_args[out_base + 0u] = vertex_count;',
        '  dst_args[out_base + 1u] = instance_count;',
        '  dst_args[out_base + 2u] = first_vertex;',
        '  dst_args[out_base + 3u] = first_instance;',
      ];
    } else if (kind === INDIRECT_CLAMP_KIND.DRAW_INDEXED) {
      body = [
        '  let first_index = src_args[src_base + 2u];',
        '  var index_count = 0u;',
        '  if (first_index < params.max_element_count) {',
        '    index_count = min(src_args[src_base + 0u], params.max_element_count - first_index);',
        '  }',
        '  let instance_count = min(src_args[src_base + 1u], params.max_instance_count);',
        '  let base_vertex_bits = src_args[src_base + 3u];',
        '  var first_instance = src_args[src_base + 4u];',
        '  if ((params.flags & 1u) != 0u) {',
        '    first_instance = 0u;',
        '  }',
        '  dst_args[out_base + 0u] = index_count;',
        '  dst_args[out_base + 1u] = instance_count;',
        '  dst_args[out_base + 2u] = first_index;',
        '  dst_args[out_base + 3u] = base_vertex_bits;',
        '  dst_args[out_base + 4u] = first_instance;',
      ];
    } else if (kind === INDIRECT_CLAMP_KIND.DISPATCH) {
      body = [
        '  dst_args[out_base + 0u] = min(src_args[src_base + 0u], params.max_group_x);',
        '  dst_args[out_base + 1u] = min(src_args[src_base + 1u], params.max_group_y);',
        '  dst_args[out_base + 2u] = min(src_args[src_base + 2u], params.max_group_z);',
      ];
    } else {
      // INDIRECT_CLAMP_KIND.DISPATCH_MESH: 12B mesh dispatch -> 28B (dispatch 12B + drawIndirect 16B).
      body = [
        '  let gx = min(src_args[src_base + 0u], params.max_group_x);',
        '  let gy = min(src_args[src_base + 1u], params.max_group_y);',
        '  let gz = min(src_args[src_base + 2u], params.max_group_z);',
        '  let total_groups = gx * gy * gz;',
        '  var vertex_count = 0u;',
        '  if (total_groups > 0u && params.max_element_count > 0u) {',
        '    let max_groups_before_cap = params.max_instance_count / params.max_element_count;',
        '    if (total_groups >= max_groups_before_cap) {',
        '      vertex_count = params.max_instance_count;',
        '    } else {',
        '      vertex_count = total_groups * params.max_element_count;',
        '    }',
        '  }',
        '  let inst_count = select(0u, 1u, vertex_count > 0u);',
        '  dst_args[out_base + 0u] = gx;',
        '  dst_args[out_base + 1u] = gy;',
        '  dst_args[out_base + 2u] = gz;',
        '  dst_args[out_base + 3u] = vertex_count;',
        '  dst_args[out_base + 4u] = inst_count;',
        '  dst_args[out_base + 5u] = 0u;',
        '  dst_args[out_base + 6u] = 0u;',
      ];
    }

    return header.concat(body, ['}', '']).join('\n');
  }

  /**
   * Look up or create a cached `GPUComputePipeline` for GPU-side indirect argument clamping,
   * keyed by `(device, kind, workgroupSize)` inside a device-scoped `WeakMap` so destroyed
   * devices are garbage-collected automatically.
   *
   * @param {object} device GPUDevice
   * @param {string} kind One of INDIRECT_CLAMP_KIND
   * @param {{ workgroupSize?: number, _statsTarget?: object | null }} [options]
   * @returns {{
   *   kind: string,
   *   workgroupSize: number,
   *   wgsl: string,
   *   module: object,
   *   bindGroupLayout: object | null,
   *   pipelineLayout: object | null,
   *   pipeline: object,
   *   hasDynamicParamsOffset: boolean,
   *   cacheHit: boolean
   * }}
   */
  function getOrCreateIndirectClampPipeline(device, kind, options) {
    if (
      !device ||
      typeof device !== 'object' ||
      typeof device.createShaderModule !== 'function' ||
      typeof device.createComputePipeline !== 'function'
    ) {
      throw new TypeError(
        'getOrCreateIndirectClampPipeline requires a GPUDevice with createShaderModule() and createComputePipeline()'
      );
    }
    clampInputWordCount(kind);
    const wgSize =
      options && options.workgroupSize !== undefined
        ? options.workgroupSize
        : DEFAULT_CLAMP_WORKGROUP_SIZE;
    if (!Number.isInteger(wgSize) || wgSize < 1 || wgSize > 256) {
      throw new RangeError(`workgroupSize must be an integer in [1, 256], got ${wgSize}`);
    }
    const statsTarget = options && options._statsTarget ? options._statsTarget : null;

    let perDevice = _deviceClampPipelineCache.get(device);
    if (!perDevice) {
      perDevice = new Map();
      _deviceClampPipelineCache.set(device, perDevice);
    }
    const key = `${kind}:${wgSize}`;
    const cached = perDevice.get(key);
    if (cached && cached.pipeline && cached.pipeline.destroyed !== true) {
      recordStat('gpuClampPipelineCacheHits', 1, statsTarget);
      return {
        kind: cached.kind,
        workgroupSize: cached.workgroupSize,
        wgsl: cached.wgsl,
        module: cached.module,
        bindGroupLayout: cached.bindGroupLayout,
        pipelineLayout: cached.pipelineLayout,
        pipeline: cached.pipeline,
        hasDynamicParamsOffset: cached.hasDynamicParamsOffset,
        cacheHit: true,
      };
    }

    const wgsl = generateIndirectClampWgsl(kind, { workgroupSize: wgSize });
    const module = device.createShaderModule({
      label: `playgta6:indirect_clamp:${kind}`,
      code: wgsl,
    });

    let bindGroupLayout = null;
    let pipelineLayout = null;
    let hasDynamicParamsOffset = false;
    if (
      typeof device.createBindGroupLayout === 'function' &&
      typeof device.createPipelineLayout === 'function'
    ) {
      bindGroupLayout = device.createBindGroupLayout({
        label: `playgta6:indirect_clamp_bgl:${kind}`,
        entries: [
          {
            binding: 0,
            visibility: SHADER_STAGE_COMPUTE,
            buffer: {
              type: 'uniform',
              hasDynamicOffset: true,
              minBindingSize: CLAMP_PARAMS_UNIFORM_BYTES,
            },
          },
          {
            binding: 1,
            visibility: SHADER_STAGE_COMPUTE,
            buffer: { type: 'read-only-storage' },
          },
          {
            binding: 2,
            visibility: SHADER_STAGE_COMPUTE,
            buffer: { type: 'read-only-storage' },
          },
          {
            binding: 3,
            visibility: SHADER_STAGE_COMPUTE,
            buffer: { type: 'storage' },
          },
        ],
      });
      pipelineLayout = device.createPipelineLayout({
        label: `playgta6:indirect_clamp_pl:${kind}`,
        bindGroupLayouts: [bindGroupLayout],
      });
      hasDynamicParamsOffset = true;
    }

    const pipeline = device.createComputePipeline({
      label: `playgta6:indirect_clamp_pipe:${kind}`,
      layout: pipelineLayout || 'auto',
      compute: {
        module,
        entryPoint: 'cs_main',
      },
    });
    if (!bindGroupLayout && pipeline && typeof pipeline.getBindGroupLayout === 'function') {
      bindGroupLayout = pipeline.getBindGroupLayout(0);
    }

    const entry = {
      kind,
      workgroupSize: wgSize,
      wgsl,
      module,
      bindGroupLayout,
      pipelineLayout,
      pipeline,
      hasDynamicParamsOffset,
    };
    perDevice.set(key, entry);
    recordStat('gpuClampPipelinesCreated', 1, statsTarget);
    return {
      kind: entry.kind,
      workgroupSize: entry.workgroupSize,
      wgsl: entry.wgsl,
      module: entry.module,
      bindGroupLayout: entry.bindGroupLayout,
      pipelineLayout: entry.pipelineLayout,
      pipeline: entry.pipeline,
      hasDynamicParamsOffset: entry.hasDynamicParamsOffset,
      cacheHit: false,
    };
  }

  function writeClampPassResult(
    target,
    ok,
    encoded,
    kind,
    commandCount,
    workgroupsX,
    dstBuffer,
    dstOffset,
    dstByteLength,
    recordBytes,
    stride,
    zeroedFirstInstance,
    hasCountBuffer,
    cacheHit,
    reason
  ) {
    if (target !== null) {
      target.ok = ok;
      target.encoded = encoded;
      target.kind = kind;
      target.commandCount = commandCount;
      target.workgroupsX = workgroupsX;
      target.dstBuffer = dstBuffer;
      target.dstOffset = dstOffset;
      target.dstByteLength = dstByteLength;
      target.recordBytes = recordBytes;
      target.stride = stride;
      target.zeroedFirstInstance = zeroedFirstInstance;
      target.hasCountBuffer = hasCountBuffer;
      target.cacheHit = cacheHit;
      target.reason = reason;
      return target;
    }
    return {
      ok,
      encoded,
      kind,
      commandCount,
      workgroupsX,
      dstBuffer,
      dstOffset,
      dstByteLength,
      recordBytes,
      stride,
      zeroedFirstInstance,
      hasCountBuffer,
      cacheHit,
      reason,
    };
  }

  /**
   * Reusable GPU-side `ExecuteIndirect` compute clamp pass manager (`GpuIndirectClampPass`,
   * aliased as `IndirectClampExecutor`). Caches per-kind `GPUComputePipeline` objects via
   * device WeakMap, caches bind groups per `(pipeline, srcBuffer, countBuffer, dstBuffer)`,
   * manages a 256B-aligned uniform parameter ring + optional clamped indirect output ring,
   * and encodes clamp dispatches with zero per-draw GC allocations on the hot path.
   */
  class GpuIndirectClampPass {
    /**
     * @param {object} options
     * @param {object} options.device GPUDevice
     * @param {number} [options.workgroupSize=64]
     * @param {number} [options.maxParamsSlots=64]
     * @param {number} [options.outputRingBytes=65536]
     * @param {boolean} [options.indirectFirstInstance] Override device feature check.
     * @param {number} [options.maxComputeWorkgroupsPerDimension] Override device limit.
     * @param {string} [options.label='playgta6-indirect-clamp']
     */
    constructor(options) {
      if (!options || typeof options !== 'object' || !options.device) {
        throw new TypeError('GpuIndirectClampPass requires an options object with a device');
      }
      this.device = options.device;
      this.workgroupSize =
        options.workgroupSize === undefined
          ? DEFAULT_CLAMP_WORKGROUP_SIZE
          : options.workgroupSize;
      if (!Number.isInteger(this.workgroupSize) || this.workgroupSize < 1 || this.workgroupSize > 256) {
        throw new RangeError(`workgroupSize must be an integer in [1, 256], got ${this.workgroupSize}`);
      }
      this.maxParamsSlots =
        options.maxParamsSlots === undefined ? DEFAULT_CLAMP_PARAMS_SLOTS : options.maxParamsSlots;
      assertPositiveInteger(this.maxParamsSlots, 'maxParamsSlots');
      this.outputRingBytes =
        options.outputRingBytes === undefined
          ? DEFAULT_CLAMP_OUTPUT_BYTES
          : options.outputRingBytes;
      assertPositiveInteger(this.outputRingBytes, 'outputRingBytes');
      if (!isAligned4Fast(this.outputRingBytes)) {
        this.outputRingBytes = Math.ceil(this.outputRingBytes / 4) * 4;
      }
      this.indirectFirstInstance =
        options.indirectFirstInstance !== undefined
          ? Boolean(options.indirectFirstInstance)
          : deviceSupportsIndirectFirstInstance(this.device);
      this.maxComputeWorkgroupsPerDimension = resolveMaxComputeWorkgroups(
        this.device,
        options.maxComputeWorkgroupsPerDimension
      );
      this.label = options.label || 'playgta6-indirect-clamp';
      this.destroyed = false;

      // Reusable hot-path scratch objects (zero per-draw heap allocations).
      this._paramsU32 = new Uint32Array(CLAMP_PARAMS_UNIFORM_BYTES >>> 2);
      this._dynOffsets = new Uint32Array(1);
      this._computePassDesc = { label: `${this.label}:pass` };
      this._bgEntries = [
        { binding: 0, resource: { buffer: null, offset: 0, size: CLAMP_PARAMS_UNIFORM_BYTES } },
        { binding: 1, resource: { buffer: null } },
        { binding: 2, resource: { buffer: null } },
        { binding: 3, resource: { buffer: null } },
      ];
      this._bgDescriptor = {
        label: `${this.label}:bg`,
        layout: null,
        entries: this._bgEntries,
      };
      this._scratchResult = writeClampPassResult(
        null,
        false,
        false,
        INDIRECT_CLAMP_KIND.DRAW,
        0,
        0,
        null,
        0,
        0,
        0,
        0,
        false,
        false,
        false,
        null
      );

      this._paramsBuffer = null;
      this._paramsSlotCursor = 0;
      this._outputRingBuffer = null;
      this._outputRingCursor = 0;

      // Bind-group cache keyed by integer resource IDs to avoid per-draw createBindGroup().
      this._resourceIds = new WeakMap();
      this._nextResourceId = 0;
      this._bgCache = new Map();
      this._maxBgCacheEntries = 256;
      this._stats = createEmptyStats();
      this._bgCacheHits = 0;
      this._bgCacheMisses = 0;
    }

    _getResId(obj) {
      if (!obj || typeof obj !== 'object') {
        return 0;
      }
      let id = this._resourceIds.get(obj);
      if (id === undefined) {
        this._nextResourceId += 1;
        id = this._nextResourceId;
        this._resourceIds.set(obj, id);
      }
      return id;
    }

    _ensureParamsBuffer(hasDynamicOffset) {
      const neededSlots = hasDynamicOffset ? this.maxParamsSlots : 1;
      const neededBytes = neededSlots * CLAMP_PARAMS_SLOT_STRIDE;
      if (
        this._paramsBuffer &&
        this._paramsBuffer.destroyed !== true &&
        (typeof this._paramsBuffer.size !== 'number' || this._paramsBuffer.size >= neededBytes)
      ) {
        return this._paramsBuffer;
      }
      if (this._paramsBuffer && typeof this._paramsBuffer.destroy === 'function') {
        try {
          this._paramsBuffer.destroy();
        } catch (_err) {
          // Ignore already-destroyed device buffer.
        }
      }
      this._paramsBuffer = this.device.createBuffer({
        label: `${this.label}:params`,
        size: neededBytes,
        usage: BUFFER_USAGE.UNIFORM | BUFFER_USAGE.COPY_DST,
      });
      this._paramsSlotCursor = 0;
      this._bgCache.clear();
      return this._paramsBuffer;
    }

    _ensureOutputRingBuffer(minBytes) {
      if (
        this._outputRingBuffer &&
        this._outputRingBuffer.destroyed !== true &&
        (typeof this._outputRingBuffer.size !== 'number' || this._outputRingBuffer.size >= minBytes)
      ) {
        return this._outputRingBuffer;
      }
      const nextSize = Math.max(this.outputRingBytes, Math.ceil(minBytes / 4) * 4);
      if (this._outputRingBuffer && typeof this._outputRingBuffer.destroy === 'function') {
        try {
          this._outputRingBuffer.destroy();
        } catch (_err) {
          // Ignore already-destroyed device buffer.
        }
      }
      this._outputRingBuffer = this.device.createBuffer({
        label: `${this.label}:output-ring`,
        size: nextSize,
        usage:
          BUFFER_USAGE.INDIRECT |
          BUFFER_USAGE.STORAGE |
          BUFFER_USAGE.COPY_DST |
          BUFFER_USAGE.COPY_SRC,
      });
      this.outputRingBytes = nextSize;
      this._outputRingCursor = 0;
      this._bgCache.clear();
      return this._outputRingBuffer;
    }

    _allocateOutputWindow(byteLength) {
      const alignedLen = Math.ceil(byteLength / 4) * 4;
      const buf = this._ensureOutputRingBuffer(alignedLen);
      const bufSize = typeof buf.size === 'number' ? buf.size : this.outputRingBytes;
      if (this._outputRingCursor + alignedLen > bufSize) {
        this._outputRingCursor = 0;
      }
      const offset = this._outputRingCursor;
      this._outputRingCursor += alignedLen;
      return offset;
    }

    _getOrCreateClampBindGroup(pipeInfo, paramsBuffer, srcBuffer, countBuffer, dstBuffer, paramsOffset) {
      const dynamic = pipeInfo.hasDynamicParamsOffset;
      const keyOffset = dynamic ? 0 : paramsOffset;
      const key =
        `${this._getResId(pipeInfo.pipeline)}:` +
        `${this._getResId(paramsBuffer)}:${keyOffset}:` +
        `${this._getResId(srcBuffer)}:` +
        `${this._getResId(countBuffer)}:` +
        `${this._getResId(dstBuffer)}`;

      const cached = this._bgCache.get(key);
      if (
        cached &&
        cached.bg &&
        cached.bg.destroyed !== true &&
        cached.paramsBuffer === paramsBuffer &&
        cached.srcBuffer === srcBuffer &&
        cached.countBuffer === countBuffer &&
        cached.dstBuffer === dstBuffer &&
        paramsBuffer.destroyed !== true &&
        srcBuffer.destroyed !== true &&
        countBuffer.destroyed !== true &&
        dstBuffer.destroyed !== true
      ) {
        this._bgCacheHits += 1;
        return cached.bg;
      }

      this._bgCacheMisses += 1;
      this._bgEntries[0].resource.buffer = paramsBuffer;
      this._bgEntries[0].resource.offset = keyOffset;
      this._bgEntries[0].resource.size = CLAMP_PARAMS_UNIFORM_BYTES;
      this._bgEntries[1].resource.buffer = srcBuffer;
      this._bgEntries[2].resource.buffer = countBuffer;
      this._bgEntries[3].resource.buffer = dstBuffer;
      this._bgDescriptor.layout = pipeInfo.bindGroupLayout;

      const bg = this.device.createBindGroup(this._bgDescriptor);
      if (this._bgCache.size >= this._maxBgCacheEntries) {
        const oldestKey = this._bgCache.keys().next().value;
        this._bgCache.delete(oldestKey);
      }
      this._bgCache.set(key, {
        bg,
        paramsBuffer,
        srcBuffer,
        countBuffer,
        dstBuffer,
      });
      return bg;
    }

    /**
     * Reset per-frame parameter-slot and output-ring cursors (call at frame start).
     */
    resetFrame() {
      this._paramsSlotCursor = 0;
      this._outputRingCursor = 0;
    }

    /**
     * Encode a GPU compute clamp pass for `kind` ('draw' | 'drawIndexed' | 'dispatch' | 'dispatchMesh').
     * Accepts either a `GPUCommandEncoder` (opens and ends a compute pass) or an active
     * `GPUComputePassEncoder` (encodes directly onto the open pass for batching).
     * Pass a preallocated `out` object (or `options.out`) for zero per-draw GC allocations.
     *
     * @param {object} passOrEncoder GPUCommandEncoder or GPUComputePassEncoder
     * @param {string} kind INDIRECT_CLAMP_KIND
     * @param {object} options
     * @param {object} options.srcBuffer Source GPU-written indirect buffer (STORAGE)
     * @param {number} [options.srcOffset=0] 4-byte-aligned byte offset in `srcBuffer`
     * @param {number} [options.stride] Source byte stride (>= input record bytes, multiple of 4)
     * @param {number} [options.commandCount=1] Number of commands (`count` / `maxCommandCount` aliases)
     * @param {object} [options.dstBuffer] Output indirect buffer (STORAGE | INDIRECT); uses managed ring if omitted
     * @param {number} [options.dstOffset=0] 4-byte-aligned byte offset in `dstBuffer`
     * @param {object} [options.countBuffer] Optional GPU count buffer (u32 at `countOffset`)
     * @param {number} [options.countOffset=0] 4-byte-aligned byte offset in `countBuffer`
     * @param {number} [options.maxVertexCount] Upper bound for draw `vertexCount` (default U32_MAX)
     * @param {number} [options.maxIndexCount] Upper bound for drawIndexed `indexCount` (default U32_MAX)
     * @param {number} [options.maxMeshVerticesPerGroup] Vertices per mesh group for `dispatchMesh`
     * @param {number} [options.maxMeshTotalVertices] Total vertex cap for `dispatchMesh`
     * @param {number} [options.maxInstanceCount] Upper bound for `instanceCount` (default U32_MAX)
     * @param {number} [options.maxWorkgroupsX] Upper bound for workgroups X (default device limit)
     * @param {number} [options.maxWorkgroupsY] Upper bound for workgroups Y (default device limit)
     * @param {number} [options.maxWorkgroupsZ] Upper bound for workgroups Z (default device limit)
     * @param {boolean} [options.indirectFirstInstance] Override `indirect-first-instance` gating
     * @param {number} [options.maxIndirectBatchCount=4096]
     * @param {boolean} [options.clampBatch=true]
     * @param {object} [options.out]
     * @param {object} [out]
     */
    encodeClamp(passOrEncoder, kind, options, out) {
      const hasOpts = options !== undefined && options !== null && typeof options === 'object';
      const outTarget = resolveOutTarget(options, out);
      if (this.destroyed) {
        return writeClampPassResult(
          outTarget,
          false,
          false,
          kind,
          0,
          0,
          null,
          0,
          0,
          0,
          0,
          false,
          false,
          false,
          'DESTROYED'
        );
      }

      let inWords;
      let outWords;
      try {
        inWords = clampInputWordCount(kind);
        outWords = clampOutputWordCount(kind);
      } catch (_err) {
        recordBoundsRejection(this._stats);
        return writeClampPassResult(
          outTarget,
          false,
          false,
          kind,
          0,
          0,
          null,
          0,
          0,
          0,
          0,
          false,
          false,
          false,
          'UNKNOWN_KIND'
        );
      }

      const inBytes = inWords * 4;
      const outBytes = outWords * 4;
      const isEncoder = Boolean(
        passOrEncoder && typeof passOrEncoder.beginComputePass === 'function'
      );
      const isComputePass = Boolean(
        passOrEncoder &&
          typeof passOrEncoder.setPipeline === 'function' &&
          typeof passOrEncoder.setBindGroup === 'function' &&
          typeof passOrEncoder.dispatchWorkgroups === 'function'
      );
      if (!isEncoder && !isComputePass) {
        recordBoundsRejection(this._stats);
        return writeClampPassResult(
          outTarget,
          false,
          false,
          kind,
          0,
          0,
          null,
          0,
          0,
          outBytes,
          outBytes,
          false,
          false,
          false,
          'INVALID_PASS_OR_ENCODER'
        );
      }

      if (!hasOpts) {
        recordBoundsRejection(this._stats);
        return writeClampPassResult(
          outTarget,
          false,
          false,
          kind,
          0,
          0,
          null,
          0,
          0,
          outBytes,
          outBytes,
          false,
          false,
          false,
          'INVALID_OPTIONS'
        );
      }

      const srcBuffer = options.srcBuffer !== undefined ? options.srcBuffer : options.buffer;
      const srcOffset = options.srcOffset !== undefined ? options.srcOffset : options.offset || 0;
      const stride = options.stride !== undefined ? options.stride : inBytes;
      const rawCount =
        options.commandCount !== undefined
          ? options.commandCount
          : options.count !== undefined
            ? options.count
            : options.maxCommandCount !== undefined
              ? options.maxCommandCount
              : 1;
      const maxBatch =
        options.maxIndirectBatchCount !== undefined
          ? options.maxIndirectBatchCount
          : DEFAULT_MAX_INDIRECT_BATCH_COUNT;
      const clampBatch = options.clampBatch !== false;

      if (!isNonNegSafeIntFast(rawCount)) {
        recordBoundsRejection(this._stats);
        return writeClampPassResult(
          outTarget,
          false,
          false,
          kind,
          0,
          0,
          null,
          0,
          0,
          outBytes,
          outBytes,
          false,
          false,
          false,
          'INVALID_COUNT'
        );
      }
      if (rawCount === 0) {
        recordStat('zeroBatchSkips', 1, this._stats);
        recordStat('zeroCountElided', 1, this._stats);
        return writeClampPassResult(
          outTarget,
          true,
          false,
          kind,
          0,
          0,
          options.dstBuffer || null,
          options.dstOffset || 0,
          0,
          outBytes,
          outBytes,
          false,
          false,
          false,
          'ZERO_COUNT'
        );
      }

      let commandCount = rawCount;
      if (isPosSafeIntFast(maxBatch) && commandCount > maxBatch) {
        if (!clampBatch) {
          recordBatchLimitRejected(this._stats);
          return writeClampPassResult(
            outTarget,
            false,
            false,
            kind,
            0,
            0,
            null,
            0,
            0,
            outBytes,
            outBytes,
            false,
            false,
            false,
            'BATCH_LIMIT_EXCEEDED'
          );
        }
        commandCount = maxBatch;
        recordBatchClamped(this._stats);
      }

      if (!srcBuffer || typeof srcBuffer !== 'object' || srcBuffer.destroyed === true) {
        recordRangeRejection(
          srcBuffer && srcBuffer.destroyed ? RANGE_DESTROYED_BUFFER : RANGE_NOT_INDIRECT,
          this._stats,
          commandCount,
          true
        );
        return writeClampPassResult(
          outTarget,
          false,
          false,
          kind,
          0,
          0,
          null,
          0,
          0,
          outBytes,
          outBytes,
          false,
          false,
          false,
          srcBuffer && srcBuffer.destroyed ? 'DESTROYED_BUFFER' : 'INVALID_SRC_BUFFER'
        );
      }
      if (!isNonNegSafeIntFast(srcOffset) || !isAligned4Fast(srcOffset)) {
        recordRangeRejection(RANGE_MISALIGNED_OFFSET, this._stats, commandCount, true);
        return writeClampPassResult(
          outTarget,
          false,
          false,
          kind,
          0,
          0,
          null,
          0,
          0,
          outBytes,
          outBytes,
          false,
          false,
          false,
          'MISALIGNED_OFFSET'
        );
      }
      if (!isPosSafeIntFast(stride) || stride < inBytes || !isAligned4Fast(stride)) {
        recordBoundsRejection(this._stats);
        return writeClampPassResult(
          outTarget,
          false,
          false,
          kind,
          0,
          0,
          null,
          0,
          0,
          outBytes,
          outBytes,
          false,
          false,
          false,
          'INVALID_STRIDE'
        );
      }

      const countBuffer = options.countBuffer || null;
      const countOffset = options.countOffset || 0;
      const hasCountBuffer = countBuffer !== null;
      if (hasCountBuffer) {
        if (typeof countBuffer !== 'object' || countBuffer.destroyed === true) {
          recordBoundsRejection(this._stats);
          return writeClampPassResult(
            outTarget,
            false,
            false,
            kind,
            0,
            0,
            null,
            0,
            0,
            outBytes,
            outBytes,
            false,
            true,
            false,
            'INVALID_COUNT_BUFFER'
          );
        }
        if (!isNonNegSafeIntFast(countOffset) || !isAligned4Fast(countOffset)) {
          recordRangeRejection(RANGE_MISALIGNED_OFFSET, this._stats, 1, true);
          return writeClampPassResult(
            outTarget,
            false,
            false,
            kind,
            0,
            0,
            null,
            0,
            0,
            outBytes,
            outBytes,
            false,
            true,
            false,
            'MISALIGNED_COUNT_OFFSET'
          );
        }
      }

      const dstByteLength = commandCount * outBytes;
      let dstBuffer = options.dstBuffer || null;
      let dstOffset = options.dstOffset !== undefined ? options.dstOffset : 0;
      if (dstBuffer !== null) {
        if (typeof dstBuffer !== 'object' || dstBuffer.destroyed === true) {
          recordRangeRejection(RANGE_DESTROYED_BUFFER, this._stats, commandCount, true);
          return writeClampPassResult(
            outTarget,
            false,
            false,
            kind,
            0,
            0,
            null,
            0,
            0,
            outBytes,
            outBytes,
            false,
            hasCountBuffer,
            false,
            'INVALID_DST_BUFFER'
          );
        }
        if (!isNonNegSafeIntFast(dstOffset) || !isAligned4Fast(dstOffset)) {
          recordRangeRejection(RANGE_MISALIGNED_OFFSET, this._stats, commandCount, true);
          return writeClampPassResult(
            outTarget,
            false,
            false,
            kind,
            0,
            0,
            null,
            0,
            0,
            outBytes,
            outBytes,
            false,
            hasCountBuffer,
            false,
            'MISALIGNED_DST_OFFSET'
          );
        }
        if (typeof dstBuffer.size === 'number' && dstOffset + dstByteLength > dstBuffer.size) {
          recordRangeRejection(RANGE_OUT_OF_BOUNDS, this._stats, commandCount, true);
          return writeClampPassResult(
            outTarget,
            false,
            false,
            kind,
            0,
            0,
            null,
            0,
            0,
            outBytes,
            outBytes,
            false,
            hasCountBuffer,
            false,
            'OUT_OF_BOUNDS'
          );
        }
      } else {
        dstOffset = this._allocateOutputWindow(dstByteLength);
        dstBuffer = this._outputRingBuffer;
      }

      const pipeInfo = getOrCreateIndirectClampPipeline(this.device, kind, {
        workgroupSize: this.workgroupSize,
        _statsTarget: this._stats,
      });

      const paramsBuffer = this._ensureParamsBuffer(pipeInfo.hasDynamicParamsOffset);
      let paramsOffset = 0;
      if (pipeInfo.hasDynamicParamsOffset) {
        if (this._paramsSlotCursor >= this.maxParamsSlots) {
          this._paramsSlotCursor = 0;
        }
        paramsOffset = this._paramsSlotCursor * CLAMP_PARAMS_SLOT_STRIDE;
        this._paramsSlotCursor += 1;
      }

      const allowFirstInstance =
        options.indirectFirstInstance !== undefined
          ? Boolean(options.indirectFirstInstance)
          : this.indirectFirstInstance;
      const isDrawKind =
        kind === INDIRECT_CLAMP_KIND.DRAW || kind === INDIRECT_CLAMP_KIND.DRAW_INDEXED;
      const zeroFirstInstance = isDrawKind && !allowFirstInstance;

      let maxElementCount = U32_MAX;
      if (kind === INDIRECT_CLAMP_KIND.DRAW) {
        if (isNonNegSafeIntFast(options.maxVertexCount)) {
          maxElementCount = Math.min(options.maxVertexCount, U32_MAX);
        }
      } else if (kind === INDIRECT_CLAMP_KIND.DRAW_INDEXED) {
        if (isNonNegSafeIntFast(options.maxIndexCount)) {
          maxElementCount = Math.min(options.maxIndexCount, U32_MAX);
        }
      } else if (kind === INDIRECT_CLAMP_KIND.DISPATCH_MESH) {
        maxElementCount = isNonNegSafeIntFast(options.maxMeshVerticesPerGroup)
          ? Math.min(options.maxMeshVerticesPerGroup, U32_MAX)
          : 256 * 3;
      }

      let maxInstanceCount = U32_MAX;
      if (kind === INDIRECT_CLAMP_KIND.DISPATCH_MESH) {
        if (isNonNegSafeIntFast(options.maxMeshTotalVertices)) {
          maxInstanceCount = Math.min(options.maxMeshTotalVertices, U32_MAX);
        }
      } else if (isNonNegSafeIntFast(options.maxInstanceCount)) {
        maxInstanceCount = Math.min(options.maxInstanceCount, U32_MAX);
      }

      const maxGroupDefault = this.maxComputeWorkgroupsPerDimension;
      const maxGroupX = isNonNegSafeIntFast(options.maxWorkgroupsX)
        ? Math.min(options.maxWorkgroupsX, maxGroupDefault)
        : maxGroupDefault;
      const maxGroupY = isNonNegSafeIntFast(options.maxWorkgroupsY)
        ? Math.min(options.maxWorkgroupsY, maxGroupDefault)
        : maxGroupDefault;
      const maxGroupZ = isNonNegSafeIntFast(options.maxWorkgroupsZ)
        ? Math.min(options.maxWorkgroupsZ, maxGroupDefault)
        : maxGroupDefault;

      const u32 = this._paramsU32;
      u32[0] = (srcOffset / 4) >>> 0;
      u32[1] = (stride / 4) >>> 0;
      u32[2] = (dstOffset / 4) >>> 0;
      u32[3] = commandCount >>> 0;
      u32[4] = (countOffset / 4) >>> 0;
      u32[5] = hasCountBuffer ? 1 : 0;
      u32[6] = maxElementCount >>> 0;
      u32[7] = maxInstanceCount >>> 0;
      u32[8] = maxGroupX >>> 0;
      u32[9] = maxGroupY >>> 0;
      u32[10] = maxGroupZ >>> 0;
      u32[11] = zeroFirstInstance ? 1 : 0;

      if (this.device.queue && typeof this.device.queue.writeBuffer === 'function') {
        this.device.queue.writeBuffer(paramsBuffer, paramsOffset, u32);
      }

      const effectiveCountBuf = hasCountBuffer ? countBuffer : srcBuffer;
      const bindGroup = this._getOrCreateClampBindGroup(
        pipeInfo,
        paramsBuffer,
        srcBuffer,
        effectiveCountBuf,
        dstBuffer,
        paramsOffset
      );

      const workgroupsX = Math.ceil(commandCount / this.workgroupSize);
      const computePass = isComputePass
        ? passOrEncoder
        : passOrEncoder.beginComputePass(this._computePassDesc);

      computePass.setPipeline(pipeInfo.pipeline);
      if (pipeInfo.hasDynamicParamsOffset) {
        this._dynOffsets[0] = paramsOffset;
        computePass.setBindGroup(0, bindGroup, this._dynOffsets);
      } else {
        computePass.setBindGroup(0, bindGroup);
      }
      computePass.dispatchWorkgroups(workgroupsX, 1, 1);

      if (!isComputePass && typeof computePass.end === 'function') {
        computePass.end();
      }

      recordStat('gpuClampPassesEncoded', 1, this._stats);
      recordStat('gpuClampCommandsClamped', commandCount, this._stats);
      if (zeroFirstInstance) {
        recordStat('gpuClampFirstInstanceZeroed', commandCount, this._stats);
      }
      if (hasCountBuffer) {
        recordStat('gpuClampCountBuffersBound', 1, this._stats);
      }

      return writeClampPassResult(
        outTarget,
        true,
        true,
        kind,
        commandCount,
        workgroupsX,
        dstBuffer,
        dstOffset,
        dstByteLength,
        outBytes,
        outBytes,
        zeroFirstInstance,
        hasCountBuffer,
        pipeInfo.cacheHit,
        null
      );
    }

    clampDrawIndirect(passOrEncoder, options, out) {
      return this.encodeClamp(passOrEncoder, INDIRECT_CLAMP_KIND.DRAW, options, out);
    }

    clampDrawIndexedIndirect(passOrEncoder, options, out) {
      return this.encodeClamp(passOrEncoder, INDIRECT_CLAMP_KIND.DRAW_INDEXED, options, out);
    }

    clampDispatchIndirect(passOrEncoder, options, out) {
      return this.encodeClamp(passOrEncoder, INDIRECT_CLAMP_KIND.DISPATCH, options, out);
    }

    clampDispatchMeshIndirect(passOrEncoder, options, out) {
      return this.encodeClamp(passOrEncoder, INDIRECT_CLAMP_KIND.DISPATCH_MESH, options, out);
    }

    getStats() {
      return {
        gpuClampPassesEncoded: this._stats.gpuClampPassesEncoded,
        gpuClampCommandsClamped: this._stats.gpuClampCommandsClamped,
        gpuClampPipelinesCreated: this._stats.gpuClampPipelinesCreated,
        gpuClampPipelineCacheHits: this._stats.gpuClampPipelineCacheHits,
        gpuClampFirstInstanceZeroed: this._stats.gpuClampFirstInstanceZeroed,
        gpuClampCountBuffersBound: this._stats.gpuClampCountBuffersBound,
        boundsRejectedCount: this._stats.boundsRejectedCount,
        batchClampedCount: this._stats.batchClampedCount,
        batchLimitRejectedCount: this._stats.batchLimitRejectedCount,
        zeroBatchSkips: this._stats.zeroBatchSkips,
        zeroCountElided: this._stats.zeroCountElided,
        bindGroupCacheHits: this._bgCacheHits,
        bindGroupCacheMisses: this._bgCacheMisses,
        bindGroupCacheSize: this._bgCache.size,
        indirectFirstInstance: this.indirectFirstInstance,
        maxComputeWorkgroupsPerDimension: this.maxComputeWorkgroupsPerDimension,
        destroyed: this.destroyed,
      };
    }

    resetStats() {
      this._stats = createEmptyStats();
      this._bgCacheHits = 0;
      this._bgCacheMisses = 0;
    }

    destroy() {
      if (this.destroyed) {
        return;
      }
      this.destroyed = true;
      if (this._paramsBuffer && typeof this._paramsBuffer.destroy === 'function') {
        try {
          this._paramsBuffer.destroy();
        } catch (_err) {
          // Ignore device-lost cleanup errors.
        }
      }
      if (this._outputRingBuffer && typeof this._outputRingBuffer.destroy === 'function') {
        try {
          this._outputRingBuffer.destroy();
        } catch (_err) {
          // Ignore device-lost cleanup errors.
        }
      }
      this._paramsBuffer = null;
      this._outputRingBuffer = null;
      this._bgCache.clear();
    }
  }

  /**
   * Standalone helper to encode a GPU compute clamp pass for `draw`, `drawIndexed`,
   * `dispatch`, or `dispatchMesh` indirect arguments, caching pipelines & parameter rings
   * per `GPUDevice` via `WeakMap` (or using a caller-supplied `options.clampExecutor`).
   *
   * @param {object} passOrEncoder GPUCommandEncoder or GPUComputePassEncoder
   * @param {string} kind INDIRECT_CLAMP_KIND
   * @param {object} options Must include `srcBuffer` and either `clampExecutor` or `device`
   * @param {object} [out] Optional preallocated scratch result object
   */
  function encodeClampedIndirectPass(passOrEncoder, kind, options, out) {
    const hasOpts = options !== undefined && options !== null && typeof options === 'object';
    let executor = hasOpts ? options.clampExecutor || null : null;
    if (!executor) {
      const device = hasOpts ? options.device : null;
      if (!device || typeof device !== 'object') {
        throw new TypeError(
          'encodeClampedIndirectPass requires options.clampExecutor or options.device'
        );
      }
      executor = _deviceDefaultClampExecutors.get(device);
      if (!executor || executor.destroyed) {
        executor = new GpuIndirectClampPass({
          device,
          workgroupSize: options.workgroupSize,
          indirectFirstInstance: options.indirectFirstInstance,
          maxComputeWorkgroupsPerDimension: options.maxComputeWorkgroupsPerDimension,
        });
        _deviceDefaultClampExecutors.set(device, executor);
      }
    }
    return executor.encodeClamp(passOrEncoder, kind, options, out);
  }

  function writeMeshOrchestrationResult(
    target,
    ok,
    executed,
    elided,
    indirectDispatch,
    dispatchClamped,
    drawClamped,
    indexed,
    drawsIssued,
    drawIndirectBuffer,
    drawIndirectOffset,
    reason
  ) {
    if (target !== null) {
      target.ok = ok;
      target.executed = executed;
      target.elided = elided;
      target.indirectDispatch = indirectDispatch;
      target.dispatchClamped = dispatchClamped;
      target.drawClamped = drawClamped;
      target.indexed = indexed;
      target.drawsIssued = drawsIssued;
      target.drawIndirectBuffer = drawIndirectBuffer;
      target.drawIndirectOffset = drawIndirectOffset;
      target.reason = reason;
      return target;
    }
    return {
      ok,
      executed,
      elided,
      indirectDispatch,
      dispatchClamped,
      drawClamped,
      indexed,
      drawsIssued,
      drawIndirectBuffer,
      drawIndirectOffset,
      reason,
    };
  }

  const _meshClampScratch = writeClampPassResult(
    null,
    false,
    false,
    INDIRECT_CLAMP_KIND.DISPATCH,
    0,
    0,
    null,
    0,
    0,
    0,
    0,
    false,
    false,
    false,
    null
  );
  const _meshClampOptsScratch = {
    device: null,
    clampExecutor: null,
    srcBuffer: null,
    srcOffset: 0,
    dstBuffer: null,
    dstOffset: 0,
    countBuffer: null,
    countOffset: 0,
    commandCount: 1,
    stride: undefined,
    maxVertexCount: undefined,
    maxIndexCount: undefined,
    maxInstanceCount: undefined,
    maxWorkgroupsX: undefined,
    maxWorkgroupsY: undefined,
    maxWorkgroupsZ: undefined,
    indirectFirstInstance: undefined,
  };
  const _meshDrawBatchScratch = {};

  /**
   * Orchestrate a lowered mesh-shader compute-then-draw pipeline (#1537 / #1552):
   * 1. Optional GPU clamp of indirect mesh dispatch arguments.
   * 2. Compute pass executing the lowered mesh shader (direct `dispatchWorkgroups` or
   *    indirect `dispatchWorkgroupsIndirect`), generating mesh output buffers + draw-indirect args.
   * 3. Optional GPU clamp of generated `drawIndirect` / `drawIndexedIndirect` arguments.
   * 4. Render pass executing `drawIndirect` / `drawIndexedIndirect` over the generated mesh output.
   *
   * Zero per-draw heap allocations when `out` (or `spec.out`) is supplied.
   *
   * @param {object} encoder GPUCommandEncoder (optional if both `spec.computePass` and `spec.renderPass` are supplied)
   * @param {object} spec
   * @param {object} [out] Optional preallocated scratch result object
   */
  function encodeMeshComputeThenDraw(encoder, spec, out) {
    const outTarget = resolveOutTarget(spec, out);
    if (!spec || typeof spec !== 'object') {
      recordBoundsRejection(null);
      return writeMeshOrchestrationResult(
        outTarget,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        0,
        null,
        0,
        'INVALID_SPEC'
      );
    }

    const indexed = Boolean(spec.indexed);
    const indirectDispatch = Boolean(
      spec.indirectDispatch === true || spec.dispatchIndirectBuffer
    );
    const drawCount = spec.drawCount !== undefined ? spec.drawCount : 1;
    if (!isNonNegSafeIntFast(drawCount)) {
      recordBoundsRejection(null);
      return writeMeshOrchestrationResult(
        outTarget,
        false,
        false,
        false,
        indirectDispatch,
        false,
        false,
        indexed,
        0,
        null,
        0,
        'INVALID_DRAW_COUNT'
      );
    }
    if (drawCount === 0) {
      recordStat('meshZeroCountElided', 1, null);
      recordStat('zeroCountElided', 1, null);
      return writeMeshOrchestrationResult(
        outTarget,
        true,
        false,
        true,
        indirectDispatch,
        false,
        false,
        indexed,
        0,
        spec.drawIndirectBuffer || null,
        spec.drawIndirectOffset || 0,
        'ZERO_COUNT'
      );
    }

    let gx = 0;
    let gy = 1;
    let gz = 1;
    if (!indirectDispatch) {
      gx = spec.workgroupCountX !== undefined ? spec.workgroupCountX : spec.x !== undefined ? spec.x : 0;
      gy = spec.workgroupCountY !== undefined ? spec.workgroupCountY : spec.y !== undefined ? spec.y : 1;
      gz = spec.workgroupCountZ !== undefined ? spec.workgroupCountZ : spec.z !== undefined ? spec.z : 1;
      if (
        !isNonNegSafeIntFast(gx) ||
        !isNonNegSafeIntFast(gy) ||
        !isNonNegSafeIntFast(gz)
      ) {
        recordBoundsRejection(null);
        return writeMeshOrchestrationResult(
          outTarget,
          false,
          false,
          false,
          false,
          false,
          false,
          indexed,
          0,
          null,
          0,
          'INVALID_WORKGROUP_COUNT'
        );
      }
      if (gx === 0 || gy === 0 || gz === 0) {
        recordStat('meshZeroCountElided', 1, null);
        recordStat('zeroCountElided', 1, null);
        return writeMeshOrchestrationResult(
          outTarget,
          true,
          false,
          true,
          false,
          false,
          false,
          indexed,
          0,
          spec.drawIndirectBuffer || null,
          spec.drawIndirectOffset || 0,
          'ZERO_COUNT'
        );
      }
      const maxWg = resolveMaxComputeWorkgroups(
        spec.device,
        spec.maxComputeWorkgroupsPerDimension
      );
      if (gx > maxWg) gx = maxWg;
      if (gy > maxWg) gy = maxWg;
      if (gz > maxWg) gz = maxWg;
    }

    if (!spec.meshPipeline || !spec.renderPipeline) {
      recordBoundsRejection(null);
      return writeMeshOrchestrationResult(
        outTarget,
        false,
        false,
        false,
        indirectDispatch,
        false,
        false,
        indexed,
        0,
        null,
        0,
        !spec.meshPipeline ? 'MISSING_MESH_PIPELINE' : 'MISSING_RENDER_PIPELINE'
      );
    }

    const rawDrawBuf = spec.drawIndirectBuffer || null;
    const rawDrawOff = spec.drawIndirectOffset || 0;
    const recordBytes = indexed ? DRAW_INDEXED_INDIRECT_BYTES : DRAW_INDIRECT_BYTES;
    const drawStride = spec.drawStride !== undefined ? spec.drawStride : recordBytes;

    // 1. Optional GPU clamp of indirect mesh dispatch buffer.
    let dispatchBuf = spec.dispatchIndirectBuffer || null;
    let dispatchOff = spec.dispatchIndirectOffset || 0;
    let dispatchClamped = false;
    const hasClampCap = Boolean(spec.clampExecutor || spec.device);

    if (indirectDispatch) {
      if (
        spec.clampDispatch !== false &&
        hasClampCap &&
        (spec.clampComputePass || (encoder && typeof encoder.beginComputePass === 'function'))
      ) {
        const co = _meshClampOptsScratch;
        co.device = spec.device || null;
        co.clampExecutor = spec.clampExecutor || null;
        co.srcBuffer = dispatchBuf;
        co.srcOffset = dispatchOff;
        co.dstBuffer = spec.clampedDispatchBuffer || null;
        co.dstOffset = spec.clampedDispatchOffset || 0;
        co.countBuffer = spec.dispatchCountBuffer || null;
        co.countOffset = spec.dispatchCountOffset || 0;
        co.commandCount = 1;
        co.stride = DISPATCH_INDIRECT_BYTES;
        co.maxVertexCount = undefined;
        co.maxIndexCount = undefined;
        co.maxInstanceCount = undefined;
        co.maxWorkgroupsX = spec.maxWorkgroupsX;
        co.maxWorkgroupsY = spec.maxWorkgroupsY;
        co.maxWorkgroupsZ = spec.maxWorkgroupsZ;
        co.indirectFirstInstance = undefined;
        const clampRes = encodeClampedIndirectPass(
          spec.clampComputePass || encoder,
          INDIRECT_CLAMP_KIND.DISPATCH,
          co,
          _meshClampScratch
        );
        if (!clampRes.ok) {
          return writeMeshOrchestrationResult(
            outTarget,
            false,
            false,
            false,
            true,
            false,
            false,
            indexed,
            0,
            null,
            0,
            clampRes.reason || 'DISPATCH_CLAMP_FAILED'
          );
        }
        dispatchBuf = clampRes.dstBuffer;
        dispatchOff = clampRes.dstOffset;
        dispatchClamped = clampRes.encoded;
      } else if (
        fastCheckIndirectRange(dispatchBuf, dispatchOff, DISPATCH_INDIRECT_BYTES, undefined) !==
        RANGE_OK
      ) {
        recordRangeRejection(
          fastCheckIndirectRange(dispatchBuf, dispatchOff, DISPATCH_INDIRECT_BYTES, undefined),
          null,
          1,
          true
        );
        return writeMeshOrchestrationResult(
          outTarget,
          false,
          false,
          false,
          true,
          false,
          false,
          indexed,
          0,
          null,
          0,
          'INVALID_DISPATCH_BUFFER'
        );
      }
    }

    // 2. Mesh compute pass: run lowered mesh shader writing output geometry + draw-indirect args.
    const ownedComputePass = !spec.computePass;
    const computePass =
      spec.computePass ||
      (encoder && typeof encoder.beginComputePass === 'function'
        ? encoder.beginComputePass(spec.computePassDescriptor)
        : null);
    if (!computePass || typeof computePass.setPipeline !== 'function') {
      recordBoundsRejection(null);
      return writeMeshOrchestrationResult(
        outTarget,
        false,
        false,
        false,
        indirectDispatch,
        dispatchClamped,
        false,
        indexed,
        0,
        null,
        0,
        'INVALID_COMPUTE_PASS'
      );
    }

    computePass.setPipeline(spec.meshPipeline);
    if (Array.isArray(spec.meshBindGroups)) {
      for (let i = 0; i < spec.meshBindGroups.length; i += 1) {
        const bg = spec.meshBindGroups[i];
        if (!bg) continue;
        const groupIdx = Number.isInteger(bg.groupIndex) ? bg.groupIndex : i;
        const handle = bg.bindGroup !== undefined ? bg.bindGroup : bg;
        if (bg.dynamicOffsets) {
          computePass.setBindGroup(groupIdx, handle, bg.dynamicOffsets);
        } else {
          computePass.setBindGroup(groupIdx, handle);
        }
      }
    }
    if (indirectDispatch) {
      computePass.dispatchWorkgroupsIndirect(dispatchBuf, dispatchOff);
      recordStat('meshIndirectDispatches', 1, null);
      recordEncodedKind(INDIRECT_KIND.DISPATCH, 1, null);
    } else {
      computePass.dispatchWorkgroups(gx, gy, gz);
    }
    if (ownedComputePass && typeof computePass.end === 'function') {
      computePass.end();
    }

    // 3. Optional GPU clamp of generated draw-indirect arguments.
    let finalDrawBuf = rawDrawBuf;
    let finalDrawOff = rawDrawOff;
    let finalDrawStride = drawStride;
    let drawClamped = false;
    if (
      spec.clampDrawIndirect !== false &&
      hasClampCap &&
      (spec.drawClampComputePass || (encoder && typeof encoder.beginComputePass === 'function'))
    ) {
      const co = _meshClampOptsScratch;
      co.device = spec.device || null;
      co.clampExecutor = spec.clampExecutor || null;
      co.srcBuffer = rawDrawBuf;
      co.srcOffset = rawDrawOff;
      co.dstBuffer = spec.clampedDrawBuffer || null;
      co.dstOffset = spec.clampedDrawOffset || 0;
      co.countBuffer = spec.drawCountBuffer || null;
      co.countOffset = spec.drawCountOffset || 0;
      co.commandCount = drawCount;
      co.stride = drawStride;
      co.maxVertexCount = spec.maxVertexCount;
      co.maxIndexCount = spec.maxIndexCount;
      co.maxInstanceCount = spec.maxInstanceCount;
      co.maxWorkgroupsX = undefined;
      co.maxWorkgroupsY = undefined;
      co.maxWorkgroupsZ = undefined;
      co.indirectFirstInstance = spec.indirectFirstInstance;
      const drawClampRes = encodeClampedIndirectPass(
        spec.drawClampComputePass || encoder,
        indexed ? INDIRECT_CLAMP_KIND.DRAW_INDEXED : INDIRECT_CLAMP_KIND.DRAW,
        co,
        _meshClampScratch
      );
      if (!drawClampRes.ok) {
        return writeMeshOrchestrationResult(
          outTarget,
          false,
          false,
          false,
          indirectDispatch,
          dispatchClamped,
          false,
          indexed,
          0,
          null,
          0,
          drawClampRes.reason || 'DRAW_CLAMP_FAILED'
        );
      }
      finalDrawBuf = drawClampRes.dstBuffer;
      finalDrawOff = drawClampRes.dstOffset;
      finalDrawStride = drawClampRes.stride;
      drawClamped = drawClampRes.encoded;
    }

    // 4. Render pass: bind generated mesh output buffers + pipeline and issue drawIndirect/drawIndexedIndirect.
    const ownedRenderPass = !spec.renderPass;
    const renderPass =
      spec.renderPass ||
      (encoder && typeof encoder.beginRenderPass === 'function'
        ? encoder.beginRenderPass(spec.renderPassDescriptor)
        : null);
    if (!renderPass || typeof renderPass.setPipeline !== 'function') {
      recordBoundsRejection(null);
      return writeMeshOrchestrationResult(
        outTarget,
        false,
        false,
        false,
        indirectDispatch,
        dispatchClamped,
        drawClamped,
        indexed,
        0,
        finalDrawBuf,
        finalDrawOff,
        'INVALID_RENDER_PASS'
      );
    }

    renderPass.setPipeline(spec.renderPipeline);
    if (Array.isArray(spec.renderBindGroups)) {
      for (let i = 0; i < spec.renderBindGroups.length; i += 1) {
        const bg = spec.renderBindGroups[i];
        if (!bg) continue;
        const groupIdx = Number.isInteger(bg.groupIndex) ? bg.groupIndex : i;
        const handle = bg.bindGroup !== undefined ? bg.bindGroup : bg;
        if (bg.dynamicOffsets) {
          renderPass.setBindGroup(groupIdx, handle, bg.dynamicOffsets);
        } else {
          renderPass.setBindGroup(groupIdx, handle);
        }
      }
    }
    if (Array.isArray(spec.vertexBuffers)) {
      for (let i = 0; i < spec.vertexBuffers.length; i += 1) {
        const vb = spec.vertexBuffers[i];
        if (!vb) continue;
        const slot = Number.isInteger(vb.slot) ? vb.slot : i;
        const buf = vb.buffer !== undefined ? vb.buffer : vb;
        const off = vb.offset || 0;
        if (vb.size !== undefined && vb.size !== null) {
          renderPass.setVertexBuffer(slot, buf, off, vb.size);
        } else {
          renderPass.setVertexBuffer(slot, buf, off);
        }
      }
    }
    if (indexed && spec.indexBuffer) {
      const ib = spec.indexBuffer;
      const ibBuf = ib.buffer !== undefined ? ib.buffer : ib;
      const ibFmt = ib.format || ib.indexFormat || 'uint32';
      const ibOff = ib.offset || 0;
      if (ib.size !== undefined && ib.size !== null) {
        renderPass.setIndexBuffer(ibBuf, ibFmt, ibOff, ib.size);
      } else {
        renderPass.setIndexBuffer(ibBuf, ibFmt, ibOff);
      }
    }

    let drawsIssued = 0;
    let drawOk = true;
    let drawReason = null;
    if (drawCount === 1) {
      const dr = dispatchIndirectSingle(
        renderPass,
        indexed ? INDIRECT_KIND.DRAW_INDEXED : INDIRECT_KIND.DRAW,
        finalDrawBuf,
        finalDrawOff,
        null,
        _meshDrawBatchScratch
      );
      drawOk = dr.ok;
      drawsIssued = dr.executed;
      drawReason = dr.reason;
    } else {
      const br = dispatchIndirectBatch(
        renderPass,
        finalDrawBuf,
        {
          indexed,
          count: drawCount,
          baseOffset: finalDrawOff,
          stride: finalDrawStride,
          useNative: Boolean(spec.useNativeMultiDraw),
        },
        _meshDrawBatchScratch
      );
      drawOk = br.ok;
      drawsIssued = br.dispatched;
      drawReason = br.reason;
    }

    if (ownedRenderPass && typeof renderPass.end === 'function') {
      renderPass.end();
    }

    if (drawOk && drawsIssued > 0) {
      recordStat('meshOrchestrationsEncoded', 1, null);
    }

    return writeMeshOrchestrationResult(
      outTarget,
      drawOk,
      drawOk && drawsIssued > 0,
      false,
      indirectDispatch,
      dispatchClamped,
      drawClamped,
      indexed,
      drawsIssued,
      finalDrawBuf,
      finalDrawOff,
      drawReason
    );
  }

  const exportedApi = Object.freeze({
    U32_MAX,
    INDIRECT_OFFSET_ALIGNMENT,
    DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
    DEFAULT_MAX_INDIRECT_BATCH_COUNT,
    DEFAULT_CLAMP_WORKGROUP_SIZE,
    CLAMP_PARAMS_UNIFORM_BYTES,
    CLAMP_PARAMS_SLOT_STRIDE,
    DISPATCH_MESH_CLAMP_OUTPUT_BYTES,
    DRAW_INDIRECT_BYTES,
    DRAW_INDEXED_INDIRECT_BYTES,
    DISPATCH_INDIRECT_BYTES,
    BUFFER_USAGE,
    INDIRECT_KIND,
    INDIRECT_CLAMP_KIND,
    DRAW_INDIRECT_LAYOUT,
    DRAW_INDEXED_INDIRECT_LAYOUT,
    DISPATCH_INDIRECT_LAYOUT,
    IndirectDrawError,
    recordByteSize,
    wgslStructSource,
    wgslArrayStride,
    deviceSupportsIndirectFirstInstance,
    resolveMaxComputeWorkgroups,
    generateIndirectClampWgsl,
    getOrCreateIndirectClampPipeline,
    encodeClampedIndirectPass,
    encodeMeshComputeThenDraw,
    GpuIndirectClampPass,
    IndirectClampExecutor: GpuIndirectClampPass,
    normalizeRecord,
    encodeRecord,
    decodeRecord,
    encodeDrawIndirect,
    encodeDrawIndexedIndirect,
    encodeDispatchIndirect,
    decodeDrawIndirect,
    decodeDrawIndexedIndirect,
    decodeDispatchIndirect,
    encodeBatch,
    isValidIndirectRange,
    checkIndirectRange,
    isZeroCountRecord,
    validateIndirectBufferArgs,
    validateIndirectDrawArgs,
    validateIndirectRange,
    computeIndirectBatchStepPlan,
    validateIndirectBatch,
    validateDispatchCounts,
    planDispatches,
    encodeDispatches,
    encodeDispatchForItems,
    encodeIndirectDraw,
    encodeIndirectDrawIndexed,
    encodeIndirectDispatch,
    encodeMultiDraw,
    encodeMultiDispatchIndirect,
    executeIndirect,
    executeIndirectDraw,
    executeIndirectDrawIndexed,
    executeIndirectDispatch,
    dispatchIndirectDraw,
    dispatchIndexedIndirectDraw,
    dispatchIndirectBatch,
    getStats,
    resetStats,
    IndirectArgsBuffer,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6IndirectDraw = exportedApi;
  } else if (root) {
    root.PlayGta6IndirectDraw = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
