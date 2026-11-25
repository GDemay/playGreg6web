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

  function createEmptyStats() {
    return {
      indirectDrawsEncoded: 0,
      indirectIndexedDrawsEncoded: 0,
      indirectDispatchesEncoded: 0,
      boundsRejectedCount: 0,
      zeroBatchSkips: 0,
      batchLimitRejectedCount: 0,
      batchClampedCount: 0,
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
   *   batchClampedCount: number
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
    STORAGE:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.STORAGE === 'number'
        ? GPUBufferUsage.STORAGE
        : 0x0080,
    INDIRECT:
      typeof GPUBufferUsage !== 'undefined' && typeof GPUBufferUsage.INDIRECT === 'number'
        ? GPUBufferUsage.INDIRECT
        : 0x0100,
  });

  const INDIRECT_KIND = Object.freeze({
    DRAW: 'draw',
    DRAW_INDEXED: 'drawIndexed',
    DISPATCH: 'dispatch',
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
      recordStat('boundsRejectedCount');
      throw new RangeError(
        `stride must be a multiple of 4 and >= ${layout.byteSize} for ${kind}, got ${stride}`
      );
    }
    const outOffset = options.outOffset || 0;
    if (!Number.isInteger(outOffset) || outOffset < 0) {
      recordStat('boundsRejectedCount');
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
      recordStat('boundsRejectedCount');
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

  function resolveKnownBufferSize(buffer, options) {
    const hasOptSize = Boolean(options && options.bufferSize !== undefined && options.bufferSize !== null);
    const hasBufSize = Boolean(buffer && buffer.size !== undefined && buffer.size !== null);
    if (hasOptSize && (!Number.isInteger(options.bufferSize) || options.bufferSize < 0)) {
      return { valid: false, known: true, size: options.bufferSize, source: 'bufferSize' };
    }
    if (hasBufSize && (!Number.isInteger(buffer.size) || buffer.size < 0)) {
      return { valid: false, known: true, size: buffer.size, source: 'buffer.size' };
    }
    if (hasOptSize && hasBufSize) {
      return { valid: true, known: true, size: Math.min(buffer.size, options.bufferSize) };
    }
    if (hasOptSize) {
      return { valid: true, known: true, size: options.bufferSize };
    }
    if (hasBufSize) {
      return { valid: true, known: true, size: buffer.size };
    }
    return { valid: true, known: false, size: undefined };
  }

  function resolveBatchCount(requestedCount, options, statsTarget) {
    if (!Number.isInteger(requestedCount) || requestedCount < 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
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
        recordStat('batchClampedCount', 1, statsTarget);
        return {
          count: maxIndirectBatchCount,
          requestedCount,
          clamped: true,
          maxIndirectBatchCount,
        };
      }
      recordStat('boundsRejectedCount', 1, statsTarget);
      recordStat('batchLimitRejectedCount', 1, statsTarget);
      throw new IndirectDrawError(
        'BATCH_LIMIT_EXCEEDED',
        `indirect batch count ${requestedCount} exceeds maxIndirectBatchCount ${maxIndirectBatchCount}`
      );
    }
    return { count: requestedCount, requestedCount, clamped: false, maxIndirectBatchCount };
  }

  /**
   * Validate an indirect buffer range as WebGPU does for draw/dispatch indirect calls:
   * INDIRECT usage, offset multiple of 4, and offset + recordSize <= bufferSize whenever
   * `buffer.size` or `options.bufferSize` is known.
   *
   * @param {{usage:number, size?:number}} buffer A GPUBuffer (or fake with usage/size).
   * @param {number} offset
   * @param {number} recordSize
   * @param {object} [options]
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @throws {IndirectDrawError} With code NOT_INDIRECT, MISALIGNED_OFFSET or OUT_OF_BOUNDS.
   */
  function validateIndirectRange(buffer, offset, recordSize, options) {
    const statsTarget = options && options._statsTarget ? options._statsTarget : null;
    if (!Number.isInteger(offset) || offset < 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`indirectOffset must be a non-negative integer, got ${offset}`);
    }
    if (!Number.isInteger(recordSize) || recordSize < 1) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`recordSize must be a positive integer, got ${recordSize}`);
    }
    if (!buffer || (buffer.usage & BUFFER_USAGE.INDIRECT) === 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new IndirectDrawError('NOT_INDIRECT', 'indirect buffer lacks INDIRECT usage');
    }
    if (offset % INDIRECT_OFFSET_ALIGNMENT !== 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new IndirectDrawError(
        'MISALIGNED_OFFSET',
        `indirectOffset ${offset} is not a multiple of ${INDIRECT_OFFSET_ALIGNMENT}`
      );
    }
    const resolved = resolveKnownBufferSize(buffer, options);
    if (!resolved.valid) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new IndirectDrawError(
        'OUT_OF_BOUNDS',
        `${resolved.source} must be a non-negative integer, got ${resolved.size}`
      );
    }
    if (
      resolved.known &&
      (!Number.isSafeInteger(offset + recordSize) || offset + recordSize > resolved.size)
    ) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new IndirectDrawError(
        'OUT_OF_BOUNDS',
        `indirect record at ${offset}+${recordSize} exceeds buffer size ${resolved.size}`
      );
    }
  }

  /**
   * Validate an indirect batch's stride, offset alignment, buffer bounds, and
   * `maxIndirectBatchCount` guardrail without issuing pass commands.
   *
   * @param {{usage:number, size?:number}} buffer Indirect buffer.
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
    if (!Number.isInteger(recordSize) || recordSize < 1) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`recordSize must be a positive integer, got ${recordSize}`);
    }
    const stride = options.stride === undefined ? recordSize : options.stride;
    const firstSlot = options.firstSlot || 0;
    const baseOffset = options.baseOffset || 0;
    if (!Number.isInteger(firstSlot) || firstSlot < 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`firstSlot must be a non-negative integer, got ${firstSlot}`);
    }
    if (!Number.isInteger(baseOffset) || baseOffset < 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`baseOffset must be a non-negative integer, got ${baseOffset}`);
    }
    if (!Number.isInteger(stride) || stride < recordSize || stride % INDIRECT_OFFSET_ALIGNMENT !== 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
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

  /**
   * Validate indirect offset alignment/bounds (16 bytes) and issue one `drawIndirect`.
   *
   * @param {{drawIndirect: Function}} pass GPURenderPassEncoder.
   * @param {{usage:number, size?:number}} buffer Indirect buffer.
   * @param {number} [indirectOffset=0]
   * @param {object} [options]
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @returns {{offset:number, byteSize:number}}
   */
  function encodeIndirectDraw(pass, buffer, indirectOffset = 0, options = {}) {
    if (!pass || typeof pass.drawIndirect !== 'function') {
      throw new TypeError('pass must implement drawIndirect');
    }
    validateIndirectRange(buffer, indirectOffset, DRAW_INDIRECT_BYTES, options);
    pass.drawIndirect(buffer, indirectOffset);
    recordStat('indirectDrawsEncoded', 1, options && options._statsTarget);
    return { offset: indirectOffset, byteSize: DRAW_INDIRECT_BYTES };
  }

  /**
   * Validate indirect offset alignment/bounds (20 bytes) and issue one `drawIndexedIndirect`.
   *
   * @param {{drawIndexedIndirect: Function}} pass GPURenderPassEncoder.
   * @param {{usage:number, size?:number}} buffer Indirect buffer.
   * @param {number} [indirectOffset=0]
   * @param {object} [options]
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @returns {{offset:number, byteSize:number}}
   */
  function encodeIndirectDrawIndexed(pass, buffer, indirectOffset = 0, options = {}) {
    if (!pass || typeof pass.drawIndexedIndirect !== 'function') {
      throw new TypeError('pass must implement drawIndexedIndirect');
    }
    validateIndirectRange(buffer, indirectOffset, DRAW_INDEXED_INDIRECT_BYTES, options);
    pass.drawIndexedIndirect(buffer, indirectOffset);
    recordStat('indirectIndexedDrawsEncoded', 1, options && options._statsTarget);
    return { offset: indirectOffset, byteSize: DRAW_INDEXED_INDIRECT_BYTES };
  }

  /**
   * Validate indirect offset alignment/bounds (12 bytes) and issue one `dispatchWorkgroupsIndirect`.
   *
   * @param {{dispatchWorkgroupsIndirect: Function}} pass GPUComputePassEncoder.
   * @param {{usage:number, size?:number}} buffer Indirect buffer.
   * @param {number} [indirectOffset=0]
   * @param {object} [options]
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @returns {{offset:number, byteSize:number}}
   */
  function encodeIndirectDispatch(pass, buffer, indirectOffset = 0, options = {}) {
    if (!pass || typeof pass.dispatchWorkgroupsIndirect !== 'function') {
      throw new TypeError('pass must implement dispatchWorkgroupsIndirect');
    }
    validateIndirectRange(buffer, indirectOffset, DISPATCH_INDIRECT_BYTES, options);
    pass.dispatchWorkgroupsIndirect(buffer, indirectOffset);
    recordStat('indirectDispatchesEncoded', 1, options && options._statsTarget);
    return { offset: indirectOffset, byteSize: DISPATCH_INDIRECT_BYTES };
  }

  /**
   * Issue draws for `count` consecutive slots on a render pass encoder. Baseline WebGPU has no
   * multiDrawIndirect, so each visible slot becomes one `drawIndirect`/`drawIndexedIndirect`.
   * With `useNative` and a pass exposing `multiDrawIndirect`/`multiDrawIndexedIndirect`
   * (experimental `chromium-experimental-multi-draw-indirect`), contiguous visible runs are
   * issued as one native call each. Never required.
   *
   * @param {object} pass GPURenderPassEncoder.
   * @param {{usage:number,size?:number}} buffer Indirect buffer.
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
   * @returns {{calls:number, draws:number, skipped:number, native:boolean}}
   */
  function encodeMultiDraw(pass, buffer, options) {
    if (!options || typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const statsTarget = options._statsTarget || null;
    const indexed = Boolean(options.indexed);
    const recordSize = indexed ? DRAW_INDEXED_INDIRECT_BYTES : DRAW_INDIRECT_BYTES;
    const stride = options.stride === undefined ? recordSize : options.stride;
    const firstSlot = options.firstSlot || 0;
    const baseOffset = options.baseOffset || 0;
    if (!Number.isInteger(firstSlot) || firstSlot < 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`firstSlot must be a non-negative integer, got ${firstSlot}`);
    }
    if (!Number.isInteger(baseOffset) || baseOffset < 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`baseOffset must be a non-negative integer, got ${baseOffset}`);
    }
    if (!Number.isInteger(stride) || stride < recordSize || stride % INDIRECT_OFFSET_ALIGNMENT !== 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`stride must be a multiple of 4 and >= ${recordSize}, got ${stride}`);
    }
    const batch = resolveBatchCount(options.count, options, statsTarget);
    const method = indexed ? 'drawIndexedIndirect' : 'drawIndirect';
    const nativeMethod = indexed ? 'multiDrawIndexedIndirect' : 'multiDrawIndirect';
    const native = Boolean(options.useNative) && Boolean(pass) && typeof pass[nativeMethod] === 'function';

    if (batch.count === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      return { calls: 0, draws: 0, skipped: 0, native };
    }

    const offsetOf = (i) => baseOffset + (firstSlot + i) * stride;
    const rangeOpts = { bufferSize: options.bufferSize, _statsTarget: statsTarget };
    const hasVisibility = options.visibility !== undefined && options.visibility !== null;

    if (!hasVisibility) {
      validateIndirectRange(
        buffer,
        offsetOf(0),
        stride * (batch.count - 1) + recordSize,
        rangeOpts
      );
      if (native) {
        pass[nativeMethod](buffer, offsetOf(0), batch.count);
        recordStat(indexed ? 'indirectIndexedDrawsEncoded' : 'indirectDrawsEncoded', batch.count, statsTarget);
        return { calls: 1, draws: batch.count, skipped: 0, native: true };
      }
      for (let i = 0; i < batch.count; i += 1) {
        pass[method](buffer, offsetOf(i));
      }
      recordStat(indexed ? 'indirectIndexedDrawsEncoded' : 'indirectDrawsEncoded', batch.count, statsTarget);
      return { calls: batch.count, draws: batch.count, skipped: 0, native: false };
    }

    const planned = [];
    let skipped = 0;
    let i = 0;
    while (i < batch.count) {
      if (!visibleAt(options.visibility, i)) {
        skipped += 1;
        i += 1;
        continue;
      }
      let runEnd = i + 1;
      if (native) {
        while (runEnd < batch.count && visibleAt(options.visibility, runEnd)) {
          runEnd += 1;
        }
        const offset = offsetOf(i);
        const runCount = runEnd - i;
        validateIndirectRange(buffer, offset, stride * (runCount - 1) + recordSize, rangeOpts);
        planned.push({ offset, count: runCount });
      } else {
        const offset = offsetOf(i);
        validateIndirectRange(buffer, offset, recordSize, rangeOpts);
        planned.push({ offset, count: 1 });
      }
      i = runEnd;
    }

    let calls = 0;
    let draws = 0;
    for (const item of planned) {
      if (native) {
        pass[nativeMethod](buffer, item.offset, item.count);
      } else {
        pass[method](buffer, item.offset);
      }
      calls += 1;
      draws += item.count;
    }
    if (draws > 0) {
      recordStat(indexed ? 'indirectIndexedDrawsEncoded' : 'indirectDrawsEncoded', draws, statsTarget);
    }
    return { calls, draws, skipped, native };
  }

  /**
   * Issue `dispatchWorkgroupsIndirect` for `count` consecutive slots on a compute pass encoder,
   * validating offset alignment, stride, buffer bounds, and `maxIndirectBatchCount` before
   * invoking WebGPU pass commands.
   *
   * @param {{dispatchWorkgroupsIndirect: Function}} pass GPUComputePassEncoder.
   * @param {{usage:number, size?:number}} buffer Indirect buffer.
   * @param {object} options
   * @param {number} options.count
   * @param {number} [options.firstSlot=0]
   * @param {number} [options.baseOffset=0]
   * @param {number} [options.stride=12] Bytes between slots (>= 12, multiple of 4).
   * @param {number} [options.bufferSize] Optional explicit buffer byte length.
   * @param {ArrayLike<boolean|number>} [options.visibility] CPU mask indexed from `firstSlot`; falsy skips.
   * @param {number} [options.maxIndirectBatchCount=4096] Maximum dispatches allowed per batch.
   * @param {boolean} [options.clampBatch=false] Clamp oversized batches instead of throwing.
   * @returns {{calls:number, dispatches:number, skipped:number, clamped:boolean}}
   */
  function encodeMultiDispatchIndirect(pass, buffer, options) {
    if (!pass || typeof pass.dispatchWorkgroupsIndirect !== 'function') {
      throw new TypeError('pass must implement dispatchWorkgroupsIndirect');
    }
    if (!options || typeof options !== 'object') {
      throw new TypeError('options must be an object');
    }
    const statsTarget = options._statsTarget || null;
    const recordSize = DISPATCH_INDIRECT_BYTES;
    const stride = options.stride === undefined ? recordSize : options.stride;
    const firstSlot = options.firstSlot || 0;
    const baseOffset = options.baseOffset || 0;
    if (!Number.isInteger(firstSlot) || firstSlot < 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`firstSlot must be a non-negative integer, got ${firstSlot}`);
    }
    if (!Number.isInteger(baseOffset) || baseOffset < 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`baseOffset must be a non-negative integer, got ${baseOffset}`);
    }
    if (!Number.isInteger(stride) || stride < recordSize || stride % INDIRECT_OFFSET_ALIGNMENT !== 0) {
      recordStat('boundsRejectedCount', 1, statsTarget);
      throw new RangeError(`stride must be a multiple of 4 and >= ${recordSize}, got ${stride}`);
    }
    const batch = resolveBatchCount(options.count, options, statsTarget);
    if (batch.count === 0) {
      recordStat('zeroBatchSkips', 1, statsTarget);
      return { calls: 0, dispatches: 0, skipped: 0, clamped: false };
    }

    const offsetOf = (i) => baseOffset + (firstSlot + i) * stride;
    const rangeOpts = { bufferSize: options.bufferSize, _statsTarget: statsTarget };
    const hasVisibility = options.visibility !== undefined && options.visibility !== null;

    if (!hasVisibility) {
      validateIndirectRange(
        buffer,
        offsetOf(0),
        stride * (batch.count - 1) + recordSize,
        rangeOpts
      );
      for (let i = 0; i < batch.count; i += 1) {
        pass.dispatchWorkgroupsIndirect(buffer, offsetOf(i));
      }
      recordStat('indirectDispatchesEncoded', batch.count, statsTarget);
      return { calls: batch.count, dispatches: batch.count, skipped: 0, clamped: batch.clamped };
    }

    const offsets = [];
    let skipped = 0;
    for (let i = 0; i < batch.count; i += 1) {
      if (!visibleAt(options.visibility, i)) {
        skipped += 1;
        continue;
      }
      const offset = offsetOf(i);
      validateIndirectRange(buffer, offset, recordSize, rangeOpts);
      offsets.push(offset);
    }
    for (const offset of offsets) {
      pass.dispatchWorkgroupsIndirect(buffer, offset);
    }
    if (offsets.length > 0) {
      recordStat('indirectDispatchesEncoded', offsets.length, statsTarget);
    }
    return { calls: offsets.length, dispatches: offsets.length, skipped, clamped: batch.clamped };
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
        maxIndirectBatchCount:
          options.maxIndirectBatchCount === undefined
            ? this.maxIndirectBatchCount
            : options.maxIndirectBatchCount,
        clampBatch: options.clampBatch === undefined ? this.clampBatch : options.clampBatch,
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

  const exportedApi = Object.freeze({
    U32_MAX,
    INDIRECT_OFFSET_ALIGNMENT,
    DEFAULT_MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
    DEFAULT_MAX_INDIRECT_BATCH_COUNT,
    DRAW_INDIRECT_BYTES,
    DRAW_INDEXED_INDIRECT_BYTES,
    DISPATCH_INDIRECT_BYTES,
    BUFFER_USAGE,
    INDIRECT_KIND,
    DRAW_INDIRECT_LAYOUT,
    DRAW_INDEXED_INDIRECT_LAYOUT,
    DISPATCH_INDIRECT_LAYOUT,
    IndirectDrawError,
    recordByteSize,
    wgslStructSource,
    wgslArrayStride,
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
    validateIndirectRange,
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
