'use strict';

(function (root) {
  // Binary GPU command stream frame recorder & .pg6trace exporter.
  // Packet framing contract: runtime/abi/playgta6_abi.h & docs/gpu_commands.md.
  // Exported .pg6trace byte streams are directly consumable by tools/gpu_trace.py.

  const ABI_VERSION = 1;
  const PACKET_HEADER_BYTES = 8;
  const WRITE_TEXTURE_HEADER_BYTES = 48;
  const DEFAULT_START_FRAME = 0;
  const DEFAULT_MAX_FRAMES = 60;
  const DEFAULT_MAX_BYTES = 16 * 1024 * 1024;
  const DEFAULT_MAX_RECORDED_BYTES = DEFAULT_MAX_BYTES;
  const DEFAULT_SKIP_WRITE_TEXTURE_ABOVE_BYTES = 0;
  const MAX_SKIPPED_UPLOAD_METADATA_ENTRIES = 256;
  const FRAME_CAPTURE_SCHEMA = 'playgta6-frame-capture';
  const FRAME_CAPTURE_SCHEMA_VERSION = 1;

  const OPCODES = Object.freeze({
    NOP: 0,
    BEGIN_FRAME: 1,
    CLEAR_COLOR: 2,
    CREATE_BUFFER: 3,
    WRITE_BUFFER: 4,
    CREATE_SHADER_MODULE: 5,
    DRAW: 6,
    PRESENT: 7,
    CREATE_RENDER_PIPELINE: 8,
    SET_PIPELINE: 9,
    SET_VERTEX_BUFFER: 10,
    DESTROY_RESOURCE: 11,
    CREATE_TEXTURE: 12,
    WRITE_TEXTURE: 13,
    CREATE_SAMPLER: 14,
    CREATE_BIND_GROUP: 15,
    SET_BIND_GROUP: 16,
    SET_INDEX_BUFFER: 17,
    DRAW_INDEXED: 18,
    CREATE_COMPUTE_PIPELINE: 19,
    SET_COMPUTE_PIPELINE: 20,
    DISPATCH_COMPUTE: 21,
    CREATE_BIND_GROUP_LAYOUT: 22,
    CREATE_PIPELINE_LAYOUT: 23,
    CREATE_RENDER_PIPELINE2: 24,
    CREATE_COMPUTE_PIPELINE2: 25,
    BEGIN_RENDER_PASS: 26,
    END_RENDER_PASS: 27,
    SET_VIEWPORT: 28,
    SET_SCISSOR_RECT: 29,
    SET_BLEND_CONSTANT: 30,
    SET_STENCIL_REFERENCE: 31,
    CREATE_BIND_GROUP2: 32,
    SET_BIND_GROUP2: 33,
    CREATE_SAMPLER2: 34,
    CREATE_TEXTURE2: 35,
  });

  // Canonical opcode names matching runtime/abi/playgta6_abi.h and tools/gpu_trace.py.
  const OPCODE_NAMES = Object.freeze({
    0: 'Nop',
    1: 'BeginFrame',
    2: 'ClearColor',
    3: 'CreateBuffer',
    4: 'WriteBuffer',
    5: 'CreateShaderModule',
    6: 'Draw',
    7: 'Present',
    8: 'CreateRenderPipeline',
    9: 'SetPipeline',
    10: 'SetVertexBuffer',
    11: 'DestroyResource',
    12: 'CreateTexture',
    13: 'WriteTexture',
    14: 'CreateSampler',
    15: 'CreateBindGroup',
    16: 'SetBindGroup',
    17: 'SetIndexBuffer',
    18: 'DrawIndexed',
    19: 'CreateComputePipeline',
    20: 'SetComputePipeline',
    21: 'DispatchCompute',
    22: 'CreateBindGroupLayout',
    23: 'CreatePipelineLayout',
    24: 'CreateRenderPipeline2',
    25: 'CreateComputePipeline2',
    26: 'BeginRenderPass',
    27: 'EndRenderPass',
    28: 'SetViewport',
    29: 'SetScissorRect',
    30: 'SetBlendConstant',
    31: 'SetStencilReference',
    32: 'CreateBindGroup2',
    33: 'SetBindGroup2',
    34: 'CreateSampler2',
    35: 'CreateTexture2',
  });

  function getOpcodeName(opcode) {
    return Object.prototype.hasOwnProperty.call(OPCODE_NAMES, opcode)
      ? OPCODE_NAMES[opcode]
      : `Unknown(${opcode})`;
  }

  function isBinaryInput(input) {
    if (input instanceof Uint8Array || ArrayBuffer.isView(input) || input instanceof ArrayBuffer) {
      return true;
    }
    if (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer) {
      return true;
    }
    return false;
  }

  function toUint8Array(input, caller = 'GpuFrameRecorder.recordChunk') {
    if (input instanceof Uint8Array) {
      return input;
    }
    if (ArrayBuffer.isView(input)) {
      return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    }
    if (input instanceof ArrayBuffer) {
      return new Uint8Array(input);
    }
    if (typeof SharedArrayBuffer !== 'undefined' && input instanceof SharedArrayBuffer) {
      return new Uint8Array(input);
    }
    throw new TypeError(
      `${caller} expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer`
    );
  }

  function copyBytes(view) {
    const copy = new Uint8Array(view.byteLength);
    copy.set(view);
    return copy;
  }

  function concatChunks(chunks, totalBytes) {
    const out = new Uint8Array(totalBytes);
    let offset = 0;
    for (let i = 0; i < chunks.length; i++) {
      out.set(chunks[i], offset);
      offset += chunks[i].byteLength;
    }
    return out;
  }

  function cloneCounts(counts) {
    const out = {};
    const keys = Object.keys(counts);
    for (let i = 0; i < keys.length; i++) {
      out[keys[i]] = counts[keys[i]];
    }
    return out;
  }

  function resolveMaxBytesOption(opts, fallback) {
    const hasMaxBytes = opts.maxBytes !== undefined;
    const hasMaxRecordedBytes = opts.maxRecordedBytes !== undefined;
    if (hasMaxBytes) {
      if (!Number.isInteger(opts.maxBytes) || opts.maxBytes < 1) {
        throw new RangeError(`maxBytes must be a positive integer, got ${opts.maxBytes}`);
      }
    }
    if (hasMaxRecordedBytes) {
      if (!Number.isInteger(opts.maxRecordedBytes) || opts.maxRecordedBytes < 1) {
        throw new RangeError(
          `maxRecordedBytes must be a positive integer, got ${opts.maxRecordedBytes}`
        );
      }
    }
    if (hasMaxBytes && hasMaxRecordedBytes) {
      return Math.min(opts.maxBytes, opts.maxRecordedBytes);
    }
    if (hasMaxRecordedBytes) {
      return opts.maxRecordedBytes;
    }
    if (hasMaxBytes) {
      return opts.maxBytes;
    }
    return fallback;
  }

  function resolveSkipWriteTextureAboveBytesOption(opts, fallback) {
    const hasPrimary = opts.skipWriteTextureAboveBytes !== undefined;
    const hasAlias = opts.skipLargeWriteTextureBytes !== undefined;
    if (hasPrimary) {
      if (
        !Number.isInteger(opts.skipWriteTextureAboveBytes) ||
        opts.skipWriteTextureAboveBytes < 0
      ) {
        throw new RangeError(
          `skipWriteTextureAboveBytes must be a non-negative integer, got ${opts.skipWriteTextureAboveBytes}`
        );
      }
    }
    if (hasAlias) {
      if (
        !Number.isInteger(opts.skipLargeWriteTextureBytes) ||
        opts.skipLargeWriteTextureBytes < 0
      ) {
        throw new RangeError(
          `skipLargeWriteTextureBytes must be a non-negative integer, got ${opts.skipLargeWriteTextureBytes}`
        );
      }
    }
    if (hasPrimary) {
      return opts.skipWriteTextureAboveBytes;
    }
    if (hasAlias) {
      return opts.skipLargeWriteTextureBytes;
    }
    return fallback;
  }

  function parseWriteTextureMetadata(view, payloadOffset, payloadBytes, packetBytes, frameSequence) {
    const texelBytes =
      payloadBytes > WRITE_TEXTURE_HEADER_BYTES ? payloadBytes - WRITE_TEXTURE_HEADER_BYTES : 0;
    return {
      frameSequence,
      textureId: view.getUint32(payloadOffset, true),
      mipLevel: view.getUint32(payloadOffset + 4, true),
      width: view.getUint32(payloadOffset + 20, true),
      height: view.getUint32(payloadOffset + 24, true),
      depthOrArrayLayers: view.getUint32(payloadOffset + 28, true),
      bytesPerRow: view.getUint32(payloadOffset + 32, true),
      rowsPerImage: view.getUint32(payloadOffset + 36, true),
      dataLen: view.getUint32(payloadOffset + 40, true),
      texelBytes,
      packetBytes,
    };
  }

  function cloneSkippedLargeUploads(entries) {
    const out = new Array(entries.length);
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i];
      out[i] = {
        frameSequence: e.frameSequence,
        textureId: e.textureId,
        mipLevel: e.mipLevel,
        width: e.width,
        height: e.height,
        depthOrArrayLayers: e.depthOrArrayLayers,
        bytesPerRow: e.bytesPerRow,
        rowsPerImage: e.rowsPerImage,
        dataLen: e.dataLen,
        texelBytes: e.texelBytes,
        packetBytes: e.packetBytes,
      };
    }
    return out;
  }

  /**
   * Bounded ring-buffered in-browser GPU command stream frame recorder.
   *
   * Parses raw `playgta6::GpuPacketHeader` (`<HHI`: opcode, flags, payload_bytes)
   * packets across arbitrarily chunked buffers, groups complete frames delimited by
   * `BeginFrame` (1) .. `Present` (7), evicts oldest frames when `maxFrames` or
   * `maxBytes`/`maxRecordedBytes` caps are exceeded, optionally skips oversized
   * `WriteTexture` payloads above `skipWriteTextureAboveBytes` (recording metadata
   * and counters instead), and exports `.pg6trace` byte blobs compatible with
   * `tools/gpu_trace.py`.
   */
  class GpuFrameRecorder {
    static ABI_VERSION = ABI_VERSION;
    static PACKET_HEADER_BYTES = PACKET_HEADER_BYTES;
    static WRITE_TEXTURE_HEADER_BYTES = WRITE_TEXTURE_HEADER_BYTES;
    static DEFAULT_START_FRAME = DEFAULT_START_FRAME;
    static DEFAULT_MAX_FRAMES = DEFAULT_MAX_FRAMES;
    static DEFAULT_MAX_BYTES = DEFAULT_MAX_BYTES;
    static DEFAULT_MAX_RECORDED_BYTES = DEFAULT_MAX_RECORDED_BYTES;
    static DEFAULT_SKIP_WRITE_TEXTURE_ABOVE_BYTES = DEFAULT_SKIP_WRITE_TEXTURE_ABOVE_BYTES;
    static OPCODES = OPCODES;
    static OPCODE_NAMES = OPCODE_NAMES;

    constructor(options = {}) {
      const opts = options || {};
      const maxFrames = opts.maxFrames !== undefined ? opts.maxFrames : DEFAULT_MAX_FRAMES;
      const maxBytes = resolveMaxBytesOption(opts, DEFAULT_MAX_BYTES);
      const startFrame = opts.startFrame !== undefined ? opts.startFrame : DEFAULT_START_FRAME;
      const autoStart = opts.autoStart !== undefined ? Boolean(opts.autoStart) : false;
      const stopOnBudget =
        opts.stopOnBudget !== undefined
          ? Boolean(opts.stopOnBudget)
          : opts.startFrame !== undefined;
      const skipWriteTextureAboveBytes = resolveSkipWriteTextureAboveBytesOption(
        opts,
        DEFAULT_SKIP_WRITE_TEXTURE_ABOVE_BYTES
      );

      if (!Number.isInteger(maxFrames) || maxFrames < 1) {
        throw new RangeError(`maxFrames must be a positive integer, got ${maxFrames}`);
      }
      if (!Number.isInteger(startFrame) || startFrame < 0) {
        throw new RangeError(`startFrame must be a non-negative integer, got ${startFrame}`);
      }

      this._maxFrames = maxFrames;
      this._maxBytes = maxBytes;
      this._startFrame = startFrame;
      this._stopOnBudget = stopOnBudget;
      this._skipWriteTextureAboveBytes = skipWriteTextureAboveBytes;
      this._recording = autoStart;

      this._frames = [];
      this._frameHead = 0;
      this._totalBytes = 0;
      this._totalPackets = 0;
      this._opcodeCounts = {};
      this._droppedFrames = 0;
      this._droppedRingPackets = 0;
      this._droppedBudgetFrames = 0;
      this._budgetReached = false;
      this._capturedFrames = 0;
      this._observedFrames = 0;
      this._skippedFrames = 0;
      this._skippedLargeUploadPackets = 0;
      this._skippedLargeUploadBytes = 0;
      this._skippedLargeUploads = [];
      this._nextFrameSeq = 0;

      this._activeFrame = null;
      this._pendingChunks = [];
      this._pendingBytes = 0;
      this._pendingNeededBytes = PACKET_HEADER_BYTES;
      this._skipBytesRemaining = 0;
    }

    _resetPartialState() {
      this._activeFrame = null;
      this._pendingChunks = [];
      this._pendingBytes = 0;
      this._pendingNeededBytes = PACKET_HEADER_BYTES;
      this._skipBytesRemaining = 0;
    }

    _shouldSkipLargeWriteTexture(opcode, payloadBytes) {
      if (this._skipWriteTextureAboveBytes <= 0 || opcode !== OPCODES.WRITE_TEXTURE) {
        return false;
      }
      if (payloadBytes <= WRITE_TEXTURE_HEADER_BYTES) {
        return false;
      }
      return payloadBytes - WRITE_TEXTURE_HEADER_BYTES > this._skipWriteTextureAboveBytes;
    }

    _recordSkippedLargeWriteTexture(view, payloadOffset, payloadBytes, packetLen) {
      if (!this._activeFrame || this._activeFrame.skip) {
        return false;
      }
      this._skippedLargeUploadPackets += 1;
      this._skippedLargeUploadBytes += packetLen;
      this._activeFrame.skippedLargeUploadPackets += 1;
      this._activeFrame.skippedLargeUploadBytes += packetLen;
      if (this._skippedLargeUploads.length < MAX_SKIPPED_UPLOAD_METADATA_ENTRIES) {
        this._skippedLargeUploads.push(
          parseWriteTextureMetadata(view, payloadOffset, payloadBytes, packetLen, this._nextFrameSeq)
        );
      }
      return true;
    }

    _retainedFrameCount() {
      return this._frames.length - this._frameHead;
    }

    _triggerBudgetStop() {
      this._budgetReached = true;
      this._droppedBudgetFrames += 1;
      this._droppedFrames += 1;
      this._recording = false;
      this._resetPartialState();
    }

    start() {
      this._recording = true;
      return this;
    }

    /**
     * Starts a bounded capture session with optional frame-window, byte-budget,
     * and large-upload filter controls (`startFrame`, `maxFrames`, `maxBytes` /
     * `maxRecordedBytes`, `skipWriteTextureAboveBytes`). By default stops cleanly
     * once `maxFrames` or `maxBytes` would be exceeded (`stopOnBudget: true`) and
     * clears prior captured frames (`clear: true`).
     *
     * @param {{
     *   startFrame?: number,
     *   maxFrames?: number,
     *   maxBytes?: number,
     *   maxRecordedBytes?: number,
     *   skipWriteTextureAboveBytes?: number,
     *   skipLargeWriteTextureBytes?: number,
     *   stopOnBudget?: boolean,
     *   clear?: boolean
     * }} [options]
     */
    startCapture(options = {}) {
      const opts = options || {};
      const nextStartFrame =
        opts.startFrame !== undefined ? opts.startFrame : this._startFrame;
      const nextMaxFrames =
        opts.maxFrames !== undefined ? opts.maxFrames : this._maxFrames;
      const nextMaxBytes = resolveMaxBytesOption(opts, this._maxBytes);
      const nextSkipWriteTextureAboveBytes = resolveSkipWriteTextureAboveBytesOption(
        opts,
        this._skipWriteTextureAboveBytes
      );
      const nextStopOnBudget =
        opts.stopOnBudget !== undefined ? Boolean(opts.stopOnBudget) : true;
      const shouldClear = opts.clear !== undefined ? Boolean(opts.clear) : true;

      if (!Number.isInteger(nextStartFrame) || nextStartFrame < 0) {
        throw new RangeError(
          `startFrame must be a non-negative integer, got ${nextStartFrame}`
        );
      }
      if (!Number.isInteger(nextMaxFrames) || nextMaxFrames < 1) {
        throw new RangeError(`maxFrames must be a positive integer, got ${nextMaxFrames}`);
      }

      this._startFrame = nextStartFrame;
      this._maxFrames = nextMaxFrames;
      this._maxBytes = nextMaxBytes;
      this._skipWriteTextureAboveBytes = nextSkipWriteTextureAboveBytes;
      this._stopOnBudget = nextStopOnBudget;

      if (shouldClear) {
        this.clear();
      } else {
        this._budgetReached = false;
        this._resetPartialState();
      }

      this._recording = true;
      return this;
    }

    stop() {
      this._recording = false;
      this._resetPartialState();
      return this;
    }

    clear() {
      this._frames = [];
      this._frameHead = 0;
      this._totalBytes = 0;
      this._totalPackets = 0;
      this._opcodeCounts = {};
      this._droppedFrames = 0;
      this._droppedRingPackets = 0;
      this._droppedBudgetFrames = 0;
      this._budgetReached = false;
      this._capturedFrames = 0;
      this._observedFrames = 0;
      this._skippedFrames = 0;
      this._skippedLargeUploadPackets = 0;
      this._skippedLargeUploadBytes = 0;
      this._skippedLargeUploads = [];
      this._nextFrameSeq = 0;
      this._resetPartialState();
      return this;
    }

    isRecording() {
      return this._recording;
    }

    _evictOldestFrame() {
      if (this._frameHead >= this._frames.length) {
        return false;
      }
      const evicted = this._frames[this._frameHead];
      this._frames[this._frameHead] = undefined;
      this._frameHead += 1;

      if (this._frameHead > 64 && this._frameHead * 2 >= this._frames.length) {
        this._frames = this._frames.slice(this._frameHead);
        this._frameHead = 0;
      }

      this._totalBytes -= evicted.byteLength;
      this._totalPackets -= evicted.packetCount;

      const keys = Object.keys(evicted.opcodeCounts);
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        const nextCount = (this._opcodeCounts[key] || 0) - evicted.opcodeCounts[key];
        if (nextCount > 0) {
          this._opcodeCounts[key] = nextCount;
        } else {
          delete this._opcodeCounts[key];
        }
      }

      this._droppedFrames += 1;
      this._droppedRingPackets += evicted.packetCount;
      return true;
    }

    _commitActiveFrame() {
      const active = this._activeFrame;
      this._activeFrame = null;
      if (!active || active.skip) {
        return { completed: 0, dropped: 0 };
      }

      this._observedFrames += 1;

      if (this._stopOnBudget) {
        if (
          this._retainedFrameCount() >= this._maxFrames ||
          this._totalBytes + active.byteLength > this._maxBytes
        ) {
          this._triggerBudgetStop();
          return { completed: 0, dropped: 1 };
        }
      }

      const frameBytes = concatChunks(active.chunks, active.byteLength);
      const frame = {
        sequence: this._nextFrameSeq++,
        data: frameBytes,
        byteLength: frameBytes.byteLength,
        packetCount: active.packetCount,
        opcodeCounts: active.opcodeCounts,
        skippedLargeUploadPackets: active.skippedLargeUploadPackets,
        skippedLargeUploadBytes: active.skippedLargeUploadBytes,
      };
      this._capturedFrames += 1;

      if (frame.byteLength > this._maxBytes) {
        this._droppedFrames += 1;
        this._droppedRingPackets += frame.packetCount;
        return { completed: 1, dropped: 1 };
      }

      this._frames.push(frame);
      this._totalBytes += frame.byteLength;
      this._totalPackets += frame.packetCount;

      const keys = Object.keys(frame.opcodeCounts);
      for (let i = 0; i < keys.length; i++) {
        const key = keys[i];
        this._opcodeCounts[key] = (this._opcodeCounts[key] || 0) + frame.opcodeCounts[key];
      }

      let droppedNow = 0;
      while (
        this._retainedFrameCount() > this._maxFrames ||
        this._totalBytes > this._maxBytes
      ) {
        if (!this._evictOldestFrame()) {
          break;
        }
        droppedNow += 1;
      }

      return { completed: 1, dropped: droppedNow };
    }

    _ingestPacket(opcode, packetBytesCopy) {
      if (opcode === OPCODES.BEGIN_FRAME) {
        const skip = this._observedFrames < this._startFrame;
        // Start a fresh frame; any unclosed prior frame without Present is discarded.
        this._activeFrame = skip
          ? {
              skip: true,
              chunks: null,
              byteLength: 0,
              packetCount: 0,
              opcodeCounts: null,
              skippedLargeUploadPackets: 0,
              skippedLargeUploadBytes: 0,
            }
          : {
              skip: false,
              chunks: [],
              byteLength: 0,
              packetCount: 0,
              opcodeCounts: {},
              skippedLargeUploadPackets: 0,
              skippedLargeUploadBytes: 0,
            };
      }

      if (!this._activeFrame) {
        // Ignore packets outside a BeginFrame..Present boundary.
        return { completed: 0, dropped: 0 };
      }

      if (this._activeFrame.skip) {
        if (opcode === OPCODES.PRESENT) {
          this._activeFrame = null;
          this._observedFrames += 1;
          this._skippedFrames += 1;
          this._nextFrameSeq += 1;
        }
        return { completed: 0, dropped: 0 };
      }

      const opName = getOpcodeName(opcode);
      this._activeFrame.chunks.push(packetBytesCopy);
      this._activeFrame.byteLength += packetBytesCopy.byteLength;
      this._activeFrame.packetCount += 1;
      this._activeFrame.opcodeCounts[opName] =
        (this._activeFrame.opcodeCounts[opName] || 0) + 1;

      if (opcode === OPCODES.PRESENT) {
        return this._commitActiveFrame();
      }

      return { completed: 0, dropped: 0 };
    }

    /**
     * Records a chunk of raw binary GPU command packets.
     * Supports arbitrary splits across packet headers, payloads, and frame boundaries.
     *
     * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer} bytes
     * @returns {{
     *   recordedBytes: number,
     *   packetsParsed: number,
     *   completedFrames: number,
     *   droppedFrames: number,
     *   skippedLargeUploadPackets: number,
     *   skippedLargeUploadBytes: number
     * }}
     */
    recordChunk(bytes) {
      if (!this._recording) {
        return {
          recordedBytes: 0,
          packetsParsed: 0,
          completedFrames: 0,
          droppedFrames: 0,
          skippedLargeUploadPackets: 0,
          skippedLargeUploadBytes: 0,
        };
      }

      let src = toUint8Array(bytes);
      const srcLen = src.byteLength;
      if (srcLen === 0) {
        return {
          recordedBytes: 0,
          packetsParsed: 0,
          completedFrames: 0,
          droppedFrames: 0,
          skippedLargeUploadPackets: 0,
          skippedLargeUploadBytes: 0,
        };
      }

      if (this._skipBytesRemaining > 0) {
        if (src.byteLength <= this._skipBytesRemaining) {
          this._skipBytesRemaining -= src.byteLength;
          return {
            recordedBytes: srcLen,
            packetsParsed: 0,
            completedFrames: 0,
            droppedFrames: 0,
            skippedLargeUploadPackets: 0,
            skippedLargeUploadBytes: 0,
          };
        }
        src = src.subarray(this._skipBytesRemaining);
        this._skipBytesRemaining = 0;
      }

      let buf;
      if (this._pendingBytes > 0) {
        const combinedLen = this._pendingBytes + src.byteLength;
        if (combinedLen < this._pendingNeededBytes) {
          this._pendingChunks.push(copyBytes(src));
          this._pendingBytes = combinedLen;
          return {
            recordedBytes: srcLen,
            packetsParsed: 0,
            completedFrames: 0,
            droppedFrames: 0,
            skippedLargeUploadPackets: 0,
            skippedLargeUploadBytes: 0,
          };
        }

        buf = new Uint8Array(combinedLen);
        let writeOffset = 0;
        for (let i = 0; i < this._pendingChunks.length; i++) {
          buf.set(this._pendingChunks[i], writeOffset);
          writeOffset += this._pendingChunks[i].byteLength;
        }
        buf.set(src, writeOffset);
        this._pendingChunks = [];
        this._pendingBytes = 0;
        this._pendingNeededBytes = PACKET_HEADER_BYTES;
      } else {
        buf = src;
      }

      const len = buf.byteLength;
      const view = new DataView(buf.buffer, buf.byteOffset, len);
      let offset = 0;
      let packetsParsed = 0;
      let completedFrames = 0;
      let droppedFrames = 0;
      let chunkSkippedPackets = 0;
      let chunkSkippedBytes = 0;

      while (offset + PACKET_HEADER_BYTES <= len) {
        const opcode = view.getUint16(offset, true);
        // flags at offset + 2 (uint16 LE) preserved verbatim in raw packet bytes.
        const payloadBytes = view.getUint32(offset + 4, true);
        const packetLen = PACKET_HEADER_BYTES + payloadBytes;
        const skipLargeWriteTexture = this._shouldSkipLargeWriteTexture(opcode, payloadBytes);

        if (skipLargeWriteTexture) {
          const minHeaderLen = PACKET_HEADER_BYTES + WRITE_TEXTURE_HEADER_BYTES;
          if (offset + minHeaderLen > len) {
            this._pendingNeededBytes = minHeaderLen;
            break;
          }

          if (
            this._recordSkippedLargeWriteTexture(
              view,
              offset + PACKET_HEADER_BYTES,
              payloadBytes,
              packetLen
            )
          ) {
            chunkSkippedPackets += 1;
            chunkSkippedBytes += packetLen;
          }
          packetsParsed += 1;

          if (offset + packetLen > len) {
            this._skipBytesRemaining = packetLen - (len - offset);
            offset = len;
            break;
          }
          offset += packetLen;
          continue;
        }

        if (this._stopOnBudget) {
          if (opcode === OPCODES.BEGIN_FRAME) {
            if (this._observedFrames >= this._startFrame) {
              if (
                this._retainedFrameCount() >= this._maxFrames ||
                this._totalBytes + packetLen > this._maxBytes
              ) {
                this._triggerBudgetStop();
                droppedFrames += 1;
                return {
                  recordedBytes: srcLen,
                  packetsParsed,
                  completedFrames,
                  droppedFrames,
                  skippedLargeUploadPackets: chunkSkippedPackets,
                  skippedLargeUploadBytes: chunkSkippedBytes,
                };
              }
            }
          } else if (this._activeFrame && !this._activeFrame.skip) {
            if (this._totalBytes + this._activeFrame.byteLength + packetLen > this._maxBytes) {
              this._triggerBudgetStop();
              droppedFrames += 1;
              return {
                recordedBytes: srcLen,
                packetsParsed,
                completedFrames,
                droppedFrames,
                skippedLargeUploadPackets: chunkSkippedPackets,
                skippedLargeUploadBytes: chunkSkippedBytes,
              };
            }
          }
        }

        if (offset + packetLen > len) {
          this._pendingNeededBytes = packetLen;
          break;
        }

        let packetCopy = null;
        if (opcode === OPCODES.BEGIN_FRAME) {
          if (this._observedFrames >= this._startFrame) {
            packetCopy = copyBytes(buf.subarray(offset, offset + packetLen));
          }
        } else if (this._activeFrame && !this._activeFrame.skip) {
          packetCopy = copyBytes(buf.subarray(offset, offset + packetLen));
        }

        const res = this._ingestPacket(opcode, packetCopy);
        packetsParsed += 1;
        completedFrames += res.completed;
        droppedFrames += res.dropped;
        offset += packetLen;

        if (!this._recording) {
          return {
            recordedBytes: srcLen,
            packetsParsed,
            completedFrames,
            droppedFrames,
            skippedLargeUploadPackets: chunkSkippedPackets,
            skippedLargeUploadBytes: chunkSkippedBytes,
          };
        }
      }

      if (offset < len) {
        const remainder = copyBytes(buf.subarray(offset, len));
        this._pendingChunks = [remainder];
        this._pendingBytes = remainder.byteLength;
        if (remainder.byteLength >= PACKET_HEADER_BYTES) {
          const remView = new DataView(remainder.buffer, remainder.byteOffset, remainder.byteLength);
          const remOpcode = remView.getUint16(0, true);
          const remPayloadBytes = remView.getUint32(4, true);
          if (this._shouldSkipLargeWriteTexture(remOpcode, remPayloadBytes)) {
            this._pendingNeededBytes = PACKET_HEADER_BYTES + WRITE_TEXTURE_HEADER_BYTES;
          } else {
            this._pendingNeededBytes = PACKET_HEADER_BYTES + remPayloadBytes;
          }
        } else {
          this._pendingNeededBytes = PACKET_HEADER_BYTES;
        }
      }

      return {
        recordedBytes: srcLen,
        packetsParsed,
        completedFrames,
        droppedFrames,
        skippedLargeUploadPackets: chunkSkippedPackets,
        skippedLargeUploadBytes: chunkSkippedBytes,
      };
    }

    /**
     * Exports retained complete frames as a flat `.pg6trace` Uint8Array containing
     * concatenated raw `GpuPacketHeader` + payload packets directly readable by
     * `tools/gpu_trace.py`.
     *
     * @param {{ frameStart?: number, frameCount?: number }} [options]
     * @returns {Uint8Array}
     */
    exportTraceBytes(options) {
      const opts = options || {};
      const available = this._retainedFrameCount();

      const frameStart = opts.frameStart !== undefined ? opts.frameStart : 0;
      if (!Number.isInteger(frameStart) || frameStart < 0) {
        throw new RangeError(`frameStart must be a non-negative integer, got ${frameStart}`);
      }

      let frameCount;
      if (opts.frameCount !== undefined) {
        if (!Number.isInteger(opts.frameCount) || opts.frameCount < 0) {
          throw new RangeError(`frameCount must be a non-negative integer, got ${opts.frameCount}`);
        }
        frameCount = opts.frameCount;
      } else {
        frameCount = available;
      }

      const startIdx = Math.min(frameStart, available);
      const endIdx = Math.min(startIdx + frameCount, available);
      if (endIdx <= startIdx) {
        return new Uint8Array(0);
      }

      let totalBytes = 0;
      for (let i = startIdx; i < endIdx; i++) {
        totalBytes += this._frames[this._frameHead + i].byteLength;
      }

      const out = new Uint8Array(totalBytes);
      let writeOffset = 0;
      for (let i = startIdx; i < endIdx; i++) {
        const frameData = this._frames[this._frameHead + i].data;
        out.set(frameData, writeOffset);
        writeOffset += frameData.byteLength;
      }
      return out;
    }

    /**
     * Returns JSON-safe capture telemetry including frame-window, budget,
     * ring-eviction, and large-WriteTexture-filter counters.
     */
    getStats() {
      const available = this._retainedFrameCount();
      return {
        abiVersion: ABI_VERSION,
        frameCount: available,
        totalBytes: this._totalBytes,
        totalPackets: this._totalPackets,
        recordedBytes: this._totalBytes,
        recordedPackets: this._totalPackets,
        opcodeCounts: cloneCounts(this._opcodeCounts),
        droppedFrames: this._droppedFrames,
        droppedRingPackets: this._droppedRingPackets,
        capturedFrames: this._capturedFrames,
        capturedBytes: this._totalBytes,
        droppedBudgetFrames: this._droppedBudgetFrames,
        budgetReached: this._budgetReached,
        startFrame: this._startFrame,
        observedFrames: this._observedFrames,
        skippedFrames: this._skippedFrames,
        skippedLargeUploadPackets: this._skippedLargeUploadPackets,
        skippedLargeUploadBytes: this._skippedLargeUploadBytes,
        skipWriteTextureAboveBytes: this._skipWriteTextureAboveBytes,
        stopOnBudget: this._stopOnBudget,
        recording: this._recording,
        maxFrames: this._maxFrames,
        maxBytes: this._maxBytes,
        maxRecordedBytes: this._maxBytes,
      };
    }

    /**
     * Returns a JSON-safe summary of recorded frames, byte usage, opcode histogram,
     * dropped frame/packet counts, skipped large-upload metadata, budget telemetry,
     * and recording state.
     */
    exportMetadata() {
      const available = this._retainedFrameCount();
      const frames = new Array(available);
      for (let i = 0; i < available; i++) {
        const frame = this._frames[this._frameHead + i];
        frames[i] = {
          index: i,
          sequence: frame.sequence,
          byteLength: frame.byteLength,
          packetCount: frame.packetCount,
          opcodeCounts: cloneCounts(frame.opcodeCounts),
          skippedLargeUploadPackets: frame.skippedLargeUploadPackets,
          skippedLargeUploadBytes: frame.skippedLargeUploadBytes,
        };
      }

      const stats = this.getStats();
      stats.frames = frames;
      stats.skippedLargeUploads = cloneSkippedLargeUploads(this._skippedLargeUploads);
      return stats;
    }

    /**
     * Non-invasively wraps a WgpuCommandProcessor's `executeCommandBuffer(bytes)`
     * so all command buffers executed directly or via `drainRing()` are captured
     * while recording is active. Call `detach()` on the returned handle to restore
     * the original method.
     *
     * @param {{ executeCommandBuffer: Function }} processor
     */
    createInterceptor(processor) {
      if (!processor || typeof processor.executeCommandBuffer !== 'function') {
        throw new TypeError(
          'createInterceptor requires a processor with an executeCommandBuffer(bytes) method'
        );
      }

      const recorder = this;
      const original = processor.executeCommandBuffer;
      const hadOwn = Object.prototype.hasOwnProperty.call(processor, 'executeCommandBuffer');
      const ownDescriptor = hadOwn
        ? Object.getOwnPropertyDescriptor(processor, 'executeCommandBuffer')
        : null;

      let attached = true;

      function interceptedExecuteCommandBuffer(input) {
        if (attached && recorder.isRecording() && isBinaryInput(input)) {
          recorder.recordChunk(input);
        }
        return original.apply(processor, arguments);
      }

      let patched = false;
      const canPatch =
        !Object.isFrozen(processor) &&
        (!hadOwn ? Object.isExtensible(processor) : Boolean(ownDescriptor && (ownDescriptor.writable || ownDescriptor.configurable)));

      if (canPatch) {
        try {
          processor.executeCommandBuffer = interceptedExecuteCommandBuffer;
          patched = processor.executeCommandBuffer === interceptedExecuteCommandBuffer;
        } catch (_err) {
          patched = false;
        }
      }

      function detach() {
        if (!attached) {
          return false;
        }
        attached = false;
        if (patched && processor.executeCommandBuffer === interceptedExecuteCommandBuffer) {
          if (hadOwn && ownDescriptor) {
            Object.defineProperty(processor, 'executeCommandBuffer', ownDescriptor);
          } else {
            delete processor.executeCommandBuffer;
          }
        }
        return true;
      }

      return {
        processor,
        recorder,
        executeCommandBuffer() {
          if (patched) {
            return processor.executeCommandBuffer.apply(processor, arguments);
          }
          return interceptedExecuteCommandBuffer.apply(processor, arguments);
        },
        execute() {
          return this.executeCommandBuffer.apply(this, arguments);
        },
        isAttached() {
          return attached;
        },
        detach,
        restore: detach,
      };
    }
  }

  function createInterceptor(processor, recorderOrOptions) {
    const recorder =
      recorderOrOptions instanceof GpuFrameRecorder
        ? recorderOrOptions
        : new GpuFrameRecorder(recorderOrOptions);
    return recorder.createInterceptor(processor);
  }

  function defaultNowMs() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
      return performance.now();
    }
    return Date.now();
  }

  function isPlainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function assertNonNegativeInteger(value, fieldName) {
    if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
      throw new RangeError(`${fieldName} must be a non-negative integer`);
    }
    return value;
  }

  function assertNonNegativeFiniteNumber(value, fieldName) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
      throw new RangeError(`${fieldName} must be a non-negative finite number`);
    }
    return value;
  }

  function sanitizeTraceName(rawTrace) {
    if (rawTrace === undefined || rawTrace === null) {
      return '';
    }
    if (typeof rawTrace !== 'string') {
      throw new TypeError('sidecar.trace must be a string');
    }
    const trimmed = rawTrace.trim();
    if (!trimmed) {
      return '';
    }
    const normalized = trimmed.replace(/\\/g, '/');
    const parts = normalized.split('/').filter((segment) => segment.length > 0);
    if (parts.length === 0) {
      return '';
    }
    const base = parts[parts.length - 1];
    if (base === '.' || base === '..' || /^[A-Za-z]:$/.test(base)) {
      return '';
    }
    return base;
  }

  function formatErrorMessage(err) {
    if (err instanceof Error) {
      return err.message || err.name || 'Error';
    }
    return String(err);
  }

  function cloneSidecar(sidecar) {
    return {
      schema: sidecar.schema,
      schemaVersion: sidecar.schemaVersion,
      trace: sidecar.trace,
      payloads: sidecar.payloads,
      payloadsOmitted: sidecar.payloadsOmitted,
      framesRequested: sidecar.framesRequested,
      framesComplete: sidecar.framesComplete,
      presentsSeen: sidecar.presentsSeen,
      packets: sidecar.packets,
      bytes: sidecar.bytes,
      omittedBytes: sidecar.omittedBytes,
      frames: sidecar.frames.map((f) => ({
        index: f.index,
        offset: f.offset,
        bytes: f.bytes,
        packets: f.packets,
      })),
      presents: sidecar.presents.map((p) => ({
        index: p.index,
        wallMs: p.wallMs,
        dispatches: p.dispatches,
        importCalls: p.importCalls,
        reservedBytes: p.reservedBytes,
        committedBytes: p.committedBytes,
      })),
      omitted: sidecar.omitted.map((o) => ({
        packet: o.packet,
        opcode: o.opcode,
        opcodeName: o.opcodeName,
        length: o.length,
        fnv1a64: o.fnv1a64,
      })),
    };
  }

  /**
   * Walks a flat binary `GpuPacketHeader` (`u16 opcode, u16 flags, u32 payloadSize`)
   * `.pg6trace` stream and splits it into frame slices delimited by `Present` (opcode 7),
   * matching `runtime/gpu/frame_capture.h`:
   *   - frame 0 = everything up to and including the first `Present`
   *   - frame i = packets after `Present i-1` up to and including `Present i`
   *   - trailing partial frame (if packets or truncated bytes remain after the last `Present`)
   *
   * Handles zero-length payloads and `--gpu-capture` omitted-payload packets cleanly.
   *
   * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer} bytes
   * @returns {Array<{
   *   index: number,
   *   offset: number,
   *   byteLength: number,
   *   bytes: Uint8Array,
   *   packetCount: number,
   *   opcodeCounts: Record<string, number>,
   *   hasPresent: boolean,
   *   complete: boolean,
   *   truncated: boolean,
   *   truncatedBytes: number
   * }>}
   */
  function splitTraceIntoFrames(bytes) {
    const src = toUint8Array(bytes, 'splitTraceIntoFrames');
    const len = src.byteLength;
    if (len === 0) {
      return [];
    }

    const view = new DataView(src.buffer, src.byteOffset, len);
    const frames = [];

    let offset = 0;
    let frameStart = 0;
    let packetCount = 0;
    let opcodeCounts = {};
    let truncated = false;
    let truncatedBytes = 0;

    while (offset < len) {
      if (offset + PACKET_HEADER_BYTES > len) {
        truncated = true;
        truncatedBytes = len - offset;
        break;
      }

      const opcode = view.getUint16(offset, true);
      // flags at offset + 2 (uint16 LE) preserved verbatim in frame slice.
      const payloadSize = view.getUint32(offset + 4, true);
      const packetLen = PACKET_HEADER_BYTES + payloadSize;

      if (offset + packetLen > len) {
        truncated = true;
        truncatedBytes = len - offset;
        break;
      }

      const opName = getOpcodeName(opcode);
      opcodeCounts[opName] = (opcodeCounts[opName] || 0) + 1;
      packetCount += 1;
      offset += packetLen;

      if (opcode === OPCODES.PRESENT) {
        const frameBytes = copyBytes(src.subarray(frameStart, offset));
        frames.push({
          index: frames.length,
          offset: frameStart,
          byteLength: frameBytes.byteLength,
          bytes: frameBytes,
          packetCount,
          opcodeCounts,
          hasPresent: true,
          complete: true,
          truncated: false,
          truncatedBytes: 0,
        });
        frameStart = offset;
        packetCount = 0;
        opcodeCounts = {};
      }
    }

    if (packetCount > 0 || truncated) {
      const frameBytes = copyBytes(src.subarray(frameStart, offset));
      frames.push({
        index: frames.length,
        offset: frameStart,
        byteLength: frameBytes.byteLength,
        bytes: frameBytes,
        packetCount,
        opcodeCounts,
        hasPresent: false,
        complete: false,
        truncated,
        truncatedBytes,
      });
    }

    return frames;
  }

  /**
   * Validates and normalizes a `playgta6-frame-capture` (`schema_version: 1`) JSON
   * sidecar produced by `--gpu-capture` (`runtime/gpu/frame_capture.cpp`) without
   * leaking host filesystem paths.
   *
   * @param {string|object} jsonOrObject
   */
  function parseFrameCaptureSidecar(jsonOrObject) {
    let raw;
    if (typeof jsonOrObject === 'string') {
      try {
        raw = JSON.parse(jsonOrObject);
      } catch (err) {
        throw new SyntaxError(`Invalid frame capture sidecar JSON: ${formatErrorMessage(err)}`);
      }
    } else if (isPlainObject(jsonOrObject)) {
      raw = jsonOrObject;
    } else {
      throw new TypeError('parseFrameCaptureSidecar expects a JSON string or plain object');
    }

    if (!isPlainObject(raw)) {
      throw new TypeError('Frame capture sidecar root must be a JSON object');
    }

    if (raw.schema !== FRAME_CAPTURE_SCHEMA) {
      throw new Error(
        `Unsupported frame capture sidecar schema: expected "${FRAME_CAPTURE_SCHEMA}"`
      );
    }

    const schemaVersion = raw.schema_version !== undefined ? raw.schema_version : raw.schemaVersion;
    if (schemaVersion !== FRAME_CAPTURE_SCHEMA_VERSION) {
      throw new Error(
        `Unsupported frame capture sidecar schema_version: expected ${FRAME_CAPTURE_SCHEMA_VERSION}`
      );
    }

    if (raw.payloads !== 'omitted' && raw.payloads !== 'kept') {
      throw new Error('Frame capture sidecar payloads must be "omitted" or "kept"');
    }

    const trace = sanitizeTraceName(raw.trace);

    if (!Array.isArray(raw.frames)) {
      throw new TypeError('Frame capture sidecar frames must be an array');
    }
    const frames = raw.frames.map((entry, idx) => {
      if (!isPlainObject(entry)) {
        throw new TypeError(`sidecar.frames[${idx}] must be an object`);
      }
      return {
        index: assertNonNegativeInteger(entry.index, `sidecar.frames[${idx}].index`),
        offset: assertNonNegativeInteger(entry.offset, `sidecar.frames[${idx}].offset`),
        bytes: assertNonNegativeInteger(entry.bytes, `sidecar.frames[${idx}].bytes`),
        packets: assertNonNegativeInteger(entry.packets, `sidecar.frames[${idx}].packets`),
      };
    });

    if (raw.presents !== undefined && !Array.isArray(raw.presents)) {
      throw new TypeError('Frame capture sidecar presents must be an array');
    }
    const presents = (raw.presents || []).map((entry, idx) => {
      if (!isPlainObject(entry)) {
        throw new TypeError(`sidecar.presents[${idx}] must be an object`);
      }
      const wallMsRaw = entry.wall_ms !== undefined ? entry.wall_ms : entry.wallMs;
      const importCallsRaw =
        entry.import_calls !== undefined ? entry.import_calls : entry.importCalls;
      const reservedBytesRaw =
        entry.reserved_bytes !== undefined ? entry.reserved_bytes : entry.reservedBytes;
      const committedBytesRaw =
        entry.committed_bytes !== undefined ? entry.committed_bytes : entry.committedBytes;
      return {
        index: assertNonNegativeInteger(entry.index, `sidecar.presents[${idx}].index`),
        wallMs: assertNonNegativeFiniteNumber(wallMsRaw, `sidecar.presents[${idx}].wall_ms`),
        dispatches:
          entry.dispatches === undefined
            ? 0
            : assertNonNegativeInteger(entry.dispatches, `sidecar.presents[${idx}].dispatches`),
        importCalls:
          importCallsRaw === undefined
            ? 0
            : assertNonNegativeInteger(importCallsRaw, `sidecar.presents[${idx}].import_calls`),
        reservedBytes:
          reservedBytesRaw === undefined
            ? 0
            : assertNonNegativeInteger(
                reservedBytesRaw,
                `sidecar.presents[${idx}].reserved_bytes`
              ),
        committedBytes:
          committedBytesRaw === undefined
            ? 0
            : assertNonNegativeInteger(
                committedBytesRaw,
                `sidecar.presents[${idx}].committed_bytes`
              ),
      };
    });

    if (raw.omitted !== undefined && !Array.isArray(raw.omitted)) {
      throw new TypeError('Frame capture sidecar omitted must be an array');
    }
    const omitted = (raw.omitted || []).map((entry, idx) => {
      if (!isPlainObject(entry)) {
        throw new TypeError(`sidecar.omitted[${idx}] must be an object`);
      }
      const packet = assertNonNegativeInteger(entry.packet, `sidecar.omitted[${idx}].packet`);
      const opcode = assertNonNegativeInteger(entry.opcode, `sidecar.omitted[${idx}].opcode`);
      if (opcode > 0xffff) {
        throw new RangeError(`sidecar.omitted[${idx}].opcode must fit in uint16`);
      }
      const length = assertNonNegativeInteger(entry.length, `sidecar.omitted[${idx}].length`);
      if (typeof entry.fnv1a64 !== 'string' || !/^0x[0-9a-fA-F]{1,16}$/.test(entry.fnv1a64)) {
        throw new TypeError(`sidecar.omitted[${idx}].fnv1a64 must be a 0x-prefixed hex string`);
      }
      return {
        packet,
        opcode,
        opcodeName: getOpcodeName(opcode),
        length,
        fnv1a64: entry.fnv1a64.toLowerCase(),
      };
    });

    const framesRequestedRaw =
      raw.frames_requested !== undefined ? raw.frames_requested : raw.framesRequested;
    const framesCompleteRaw =
      raw.frames_complete !== undefined ? raw.frames_complete : raw.framesComplete;
    const presentsSeenRaw =
      raw.presents_seen !== undefined ? raw.presents_seen : raw.presentsSeen;
    const omittedBytesRaw =
      raw.omitted_bytes !== undefined ? raw.omitted_bytes : raw.omittedBytes;

    const sumFramePackets = frames.reduce((acc, f) => acc + f.packets, 0);
    const sumFrameBytes = frames.reduce((acc, f) => acc + f.bytes, 0);
    const sumOmittedBytes = omitted.reduce((acc, o) => acc + o.length, 0);

    const framesRequested =
      framesRequestedRaw === undefined
        ? frames.length
        : assertNonNegativeInteger(framesRequestedRaw, 'sidecar.frames_requested');
    const framesComplete =
      framesCompleteRaw === undefined
        ? frames.length
        : assertNonNegativeInteger(framesCompleteRaw, 'sidecar.frames_complete');
    const presentsSeen =
      presentsSeenRaw === undefined
        ? presents.length
        : assertNonNegativeInteger(presentsSeenRaw, 'sidecar.presents_seen');
    const packets =
      raw.packets === undefined
        ? sumFramePackets
        : assertNonNegativeInteger(raw.packets, 'sidecar.packets');
    const bytes =
      raw.bytes === undefined
        ? sumFrameBytes
        : assertNonNegativeInteger(raw.bytes, 'sidecar.bytes');
    const omittedBytes =
      omittedBytesRaw === undefined
        ? sumOmittedBytes
        : assertNonNegativeInteger(omittedBytesRaw, 'sidecar.omitted_bytes');

    return {
      schema: FRAME_CAPTURE_SCHEMA,
      schemaVersion: FRAME_CAPTURE_SCHEMA_VERSION,
      trace,
      payloads: raw.payloads,
      payloadsOmitted: raw.payloads === 'omitted',
      framesRequested,
      framesComplete,
      presentsSeen,
      packets,
      bytes,
      omittedBytes,
      frames,
      presents,
      omitted,
    };
  }

  function resolveProcessorInvoker(processor) {
    if (!processor || typeof processor !== 'object') {
      throw new TypeError(
        'GpuTraceReplayer requires a processor with executeBatch(bytes) or executeCommandBuffer(bytes)'
      );
    }
    if (typeof processor.executeBatch === 'function') {
      return (slice) => processor.executeBatch(slice);
    }
    if (typeof processor.executeCommandBuffer === 'function') {
      return (slice) => processor.executeCommandBuffer(slice);
    }
    if (typeof processor.execute === 'function') {
      return (slice) => processor.execute(slice);
    }
    throw new TypeError(
      'GpuTraceReplayer requires a processor with executeBatch(bytes) or executeCommandBuffer(bytes)'
    );
  }

  /**
   * Replays `.pg6trace` and native `--gpu-capture` (`playgta6-frame-capture`)
   * binary GPU command streams frame-by-frame into any processor exposing
   * `executeBatch(bytes)` or `executeCommandBuffer(bytes)` (such as `WgpuCommandProcessor`).
   */
  class GpuTraceReplayer {
    static ABI_VERSION = ABI_VERSION;
    static PACKET_HEADER_BYTES = PACKET_HEADER_BYTES;
    static FRAME_CAPTURE_SCHEMA = FRAME_CAPTURE_SCHEMA;
    static FRAME_CAPTURE_SCHEMA_VERSION = FRAME_CAPTURE_SCHEMA_VERSION;
    static OPCODES = OPCODES;
    static OPCODE_NAMES = OPCODE_NAMES;

    constructor(options = {}) {
      const opts = options || {};
      if (opts.now !== undefined && typeof opts.now !== 'function') {
        throw new TypeError('GpuTraceReplayer option now must be a function');
      }
      this._now = opts.now || defaultNowMs;
      this._loaded = false;
      this._frames = [];
      this._sidecar = null;
      this._sourceByteLength = 0;
      this._totalBytes = 0;
      this._totalPackets = 0;
      this._completeFrameCount = 0;
      this._truncatedFrameCount = 0;
      this._opcodeCounts = {};
      this._resetReplayStats();
    }

    _resetReplayStats() {
      this._replayCount = 0;
      this._replaySucceeded = 0;
      this._replayFailed = 0;
      this._replayTotalDurationMs = 0;
      this._replayTotalPackets = 0;
      this._replayTotalBytes = 0;
      this._replayErrors = [];
    }

    /**
     * Loads a flat `.pg6trace` binary command stream and optional
     * `playgta6-frame-capture` JSON sidecar.
     *
     * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer} bytes
     * @param {string|object|null} [sidecar]
     * @returns {GpuTraceReplayer}
     */
    loadTrace(bytes, sidecar) {
      const src = toUint8Array(bytes, 'GpuTraceReplayer.loadTrace');
      const parsedSidecar =
        sidecar !== undefined && sidecar !== null ? parseFrameCaptureSidecar(sidecar) : null;
      const frames = splitTraceIntoFrames(src);

      let totalBytes = 0;
      let totalPackets = 0;
      let completeFrameCount = 0;
      let truncatedFrameCount = 0;
      const opcodeCounts = {};

      for (let i = 0; i < frames.length; i++) {
        const frame = frames[i];
        totalBytes += frame.byteLength;
        totalPackets += frame.packetCount;
        if (frame.hasPresent) {
          completeFrameCount += 1;
        }
        if (frame.truncated) {
          truncatedFrameCount += 1;
        }
        const keys = Object.keys(frame.opcodeCounts);
        for (let k = 0; k < keys.length; k++) {
          const opName = keys[k];
          opcodeCounts[opName] = (opcodeCounts[opName] || 0) + frame.opcodeCounts[opName];
        }
      }

      this._loaded = true;
      this._frames = frames;
      this._sidecar = parsedSidecar;
      this._sourceByteLength = src.byteLength;
      this._totalBytes = totalBytes;
      this._totalPackets = totalPackets;
      this._completeFrameCount = completeFrameCount;
      this._truncatedFrameCount = truncatedFrameCount;
      this._opcodeCounts = opcodeCounts;
      this._resetReplayStats();

      return this;
    }

    getFrameCount() {
      return this._frames.length;
    }

    getFrame(index) {
      if (!Number.isInteger(index) || index < 0 || index >= this._frames.length) {
        throw new RangeError(
          `Frame index ${index} out of range (loaded frames: ${this._frames.length})`
        );
      }
      const frame = this._frames[index];
      return {
        index: frame.index,
        offset: frame.offset,
        byteLength: frame.byteLength,
        bytes: copyBytes(frame.bytes),
        packetCount: frame.packetCount,
        opcodeCounts: cloneCounts(frame.opcodeCounts),
        hasPresent: frame.hasPresent,
        complete: frame.complete,
        truncated: frame.truncated,
        truncatedBytes: frame.truncatedBytes,
      };
    }

    /**
     * Replays a single loaded frame slice into `processor`.
     *
     * @param {number} index
     * @param {{ executeBatch?: Function, executeCommandBuffer?: Function, execute?: Function }} processor
     */
    replayFrame(index, processor) {
      const invoke = resolveProcessorInvoker(processor);
      if (!Number.isInteger(index) || index < 0 || index >= this._frames.length) {
        throw new RangeError(
          `Frame index ${index} out of range (loaded frames: ${this._frames.length})`
        );
      }

      const frame = this._frames[index];
      const sliceCopy = copyBytes(frame.bytes);
      const startMs = this._now();
      let ok = true;
      let error = null;
      let result = null;

      try {
        const ret = invoke(sliceCopy);
        if (ret !== undefined) {
          result = ret;
        }
      } catch (err) {
        ok = false;
        error = formatErrorMessage(err);
      }

      const rawElapsed = this._now() - startMs;
      const durationMs = Number.isFinite(rawElapsed) && rawElapsed >= 0 ? rawElapsed : 0;

      this._replayCount += 1;
      if (ok) {
        this._replaySucceeded += 1;
      } else {
        this._replayFailed += 1;
        this._replayErrors.push({ frameIndex: index, error });
      }
      this._replayTotalDurationMs += durationMs;
      this._replayTotalPackets += frame.packetCount;
      this._replayTotalBytes += frame.byteLength;

      return {
        frameIndex: index,
        packetCount: frame.packetCount,
        byteLength: frame.byteLength,
        opcodeCounts: cloneCounts(frame.opcodeCounts),
        hasPresent: frame.hasPresent,
        truncated: frame.truncated,
        ok,
        error,
        durationMs,
        result,
      };
    }

    /**
     * Replays all loaded frames in order into `processor` and returns per-frame
     * results plus aggregate execution telemetry.
     *
     * @param {{ executeBatch?: Function, executeCommandBuffer?: Function, execute?: Function }} processor
     * @param {{ stopOnError?: boolean }} [options]
     */
    replayAll(processor, options = {}) {
      resolveProcessorInvoker(processor);
      const opts = options || {};
      const stopOnError = Boolean(opts.stopOnError);

      const results = [];
      const errors = [];
      const opcodeCounts = {};
      let succeededFrames = 0;
      let failedFrames = 0;
      let totalPackets = 0;
      let totalBytes = 0;
      let totalDurationMs = 0;
      let truncatedFrames = 0;

      for (let i = 0; i < this._frames.length; i++) {
        const frameRes = this.replayFrame(i, processor);
        results.push(frameRes);
        totalPackets += frameRes.packetCount;
        totalBytes += frameRes.byteLength;
        totalDurationMs += frameRes.durationMs;
        if (frameRes.truncated) {
          truncatedFrames += 1;
        }
        const keys = Object.keys(frameRes.opcodeCounts);
        for (let k = 0; k < keys.length; k++) {
          const opName = keys[k];
          opcodeCounts[opName] = (opcodeCounts[opName] || 0) + frameRes.opcodeCounts[opName];
        }
        if (frameRes.ok) {
          succeededFrames += 1;
        } else {
          failedFrames += 1;
          errors.push({ frameIndex: frameRes.frameIndex, error: frameRes.error });
          if (stopOnError) {
            break;
          }
        }
      }

      return {
        ok: failedFrames === 0,
        frameCount: this._frames.length,
        replayedFrames: results.length,
        succeededFrames,
        failedFrames,
        totalPackets,
        totalBytes,
        totalDurationMs,
        opcodeCounts,
        truncatedFrames,
        payloadsOmitted: Boolean(this._sidecar && this._sidecar.payloadsOmitted),
        omittedPayloadCount: this._sidecar ? this._sidecar.omitted.length : 0,
        omittedBytes: this._sidecar ? this._sidecar.omittedBytes : 0,
        errors,
        frames: results,
      };
    }

    /**
     * Returns a JSON-safe summary of the loaded trace, normalized sidecar (if any),
     * and cumulative replay telemetry.
     */
    getSummary() {
      return {
        abiVersion: ABI_VERSION,
        loaded: this._loaded,
        frameCount: this._frames.length,
        completeFrameCount: this._completeFrameCount,
        truncatedFrameCount: this._truncatedFrameCount,
        totalBytes: this._totalBytes,
        sourceByteLength: this._sourceByteLength,
        totalPackets: this._totalPackets,
        opcodeCounts: cloneCounts(this._opcodeCounts),
        sidecar: this._sidecar ? cloneSidecar(this._sidecar) : null,
        payloadsOmitted: Boolean(this._sidecar && this._sidecar.payloadsOmitted),
        omittedPayloadCount: this._sidecar ? this._sidecar.omitted.length : 0,
        omittedBytes: this._sidecar ? this._sidecar.omittedBytes : 0,
        replay: {
          replayedFrames: this._replayCount,
          succeededFrames: this._replaySucceeded,
          failedFrames: this._replayFailed,
          totalDurationMs: this._replayTotalDurationMs,
          totalPackets: this._replayTotalPackets,
          totalBytes: this._replayTotalBytes,
          errors: this._replayErrors.map((e) => ({
            frameIndex: e.frameIndex,
            error: e.error,
          })),
        },
      };
    }
  }

  const exportsObj = {
    GpuFrameRecorder,
    GpuTraceReplayer,
    createInterceptor,
    splitTraceIntoFrames,
    parseFrameCaptureSidecar,
    FRAME_CAPTURE_SCHEMA,
    FRAME_CAPTURE_SCHEMA_VERSION,
    ABI_VERSION,
    PACKET_HEADER_BYTES,
    WRITE_TEXTURE_HEADER_BYTES,
    DEFAULT_START_FRAME,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_RECORDED_BYTES,
    DEFAULT_SKIP_WRITE_TEXTURE_ABOVE_BYTES,
    OPCODES,
    OPCODE_NAMES,
    getOpcodeName,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6GpuRecorder = exportsObj;
  } else if (root) {
    root.PlayGta6GpuRecorder = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
