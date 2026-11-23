'use strict';

(function (root) {
  // Binary GPU command stream frame recorder & .pg6trace exporter.
  // Packet framing contract: runtime/abi/playgta6_abi.h & docs/gpu_commands.md.
  // Exported .pg6trace byte streams are directly consumable by tools/gpu_trace.py.

  const ABI_VERSION = 1;
  const PACKET_HEADER_BYTES = 8;
  const DEFAULT_MAX_FRAMES = 60;
  const DEFAULT_MAX_BYTES = 16 * 1024 * 1024;

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

  function toUint8Array(input) {
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
      'GpuFrameRecorder.recordChunk expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer'
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

  /**
   * Bounded ring-buffered in-browser GPU command stream frame recorder.
   *
   * Parses raw `playgta6::GpuPacketHeader` (`<HHI`: opcode, flags, payload_bytes)
   * packets across arbitrarily chunked buffers, groups complete frames delimited by
   * `BeginFrame` (1) .. `Present` (7), evicts oldest frames when `maxFrames` or
   * `maxBytes` caps are exceeded, and exports `.pg6trace` byte blobs compatible
   * with `tools/gpu_trace.py`.
   */
  class GpuFrameRecorder {
    static ABI_VERSION = ABI_VERSION;
    static PACKET_HEADER_BYTES = PACKET_HEADER_BYTES;
    static DEFAULT_MAX_FRAMES = DEFAULT_MAX_FRAMES;
    static DEFAULT_MAX_BYTES = DEFAULT_MAX_BYTES;
    static OPCODES = OPCODES;
    static OPCODE_NAMES = OPCODE_NAMES;

    constructor(options = {}) {
      const opts = options || {};
      const maxFrames = opts.maxFrames !== undefined ? opts.maxFrames : DEFAULT_MAX_FRAMES;
      const maxBytes = opts.maxBytes !== undefined ? opts.maxBytes : DEFAULT_MAX_BYTES;
      const autoStart = opts.autoStart !== undefined ? Boolean(opts.autoStart) : false;

      if (!Number.isInteger(maxFrames) || maxFrames < 1) {
        throw new RangeError(`maxFrames must be a positive integer, got ${maxFrames}`);
      }
      if (!Number.isInteger(maxBytes) || maxBytes < 1) {
        throw new RangeError(`maxBytes must be a positive integer, got ${maxBytes}`);
      }

      this._maxFrames = maxFrames;
      this._maxBytes = maxBytes;
      this._recording = autoStart;

      this._frames = [];
      this._frameHead = 0;
      this._totalBytes = 0;
      this._totalPackets = 0;
      this._opcodeCounts = {};
      this._droppedFrames = 0;
      this._capturedFrames = 0;
      this._nextFrameSeq = 0;

      this._activeFrame = null;
      this._pendingChunks = [];
      this._pendingBytes = 0;
      this._pendingNeededBytes = PACKET_HEADER_BYTES;
    }

    _resetPartialState() {
      this._activeFrame = null;
      this._pendingChunks = [];
      this._pendingBytes = 0;
      this._pendingNeededBytes = PACKET_HEADER_BYTES;
    }

    _retainedFrameCount() {
      return this._frames.length - this._frameHead;
    }

    start() {
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
      this._capturedFrames = 0;
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
      return true;
    }

    _commitActiveFrame() {
      const active = this._activeFrame;
      this._activeFrame = null;
      if (!active) {
        return { completed: 0, dropped: 0 };
      }

      const frameBytes = concatChunks(active.chunks, active.byteLength);
      const frame = {
        sequence: this._nextFrameSeq++,
        data: frameBytes,
        byteLength: frameBytes.byteLength,
        packetCount: active.packetCount,
        opcodeCounts: active.opcodeCounts,
      };
      this._capturedFrames += 1;

      if (frame.byteLength > this._maxBytes) {
        this._droppedFrames += 1;
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
        // Start a fresh frame; any unclosed prior frame without Present is discarded.
        this._activeFrame = {
          chunks: [],
          byteLength: 0,
          packetCount: 0,
          opcodeCounts: {},
        };
      }

      if (!this._activeFrame) {
        // Ignore packets outside a BeginFrame..Present boundary.
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
     * @returns {{ recordedBytes: number, packetsParsed: number, completedFrames: number, droppedFrames: number }}
     */
    recordChunk(bytes) {
      if (!this._recording) {
        return {
          recordedBytes: 0,
          packetsParsed: 0,
          completedFrames: 0,
          droppedFrames: 0,
        };
      }

      const src = toUint8Array(bytes);
      const srcLen = src.byteLength;
      if (srcLen === 0) {
        return {
          recordedBytes: 0,
          packetsParsed: 0,
          completedFrames: 0,
          droppedFrames: 0,
        };
      }

      let buf;
      if (this._pendingBytes > 0) {
        const combinedLen = this._pendingBytes + srcLen;
        if (combinedLen < this._pendingNeededBytes) {
          this._pendingChunks.push(copyBytes(src));
          this._pendingBytes = combinedLen;
          return {
            recordedBytes: srcLen,
            packetsParsed: 0,
            completedFrames: 0,
            droppedFrames: 0,
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

      while (offset + PACKET_HEADER_BYTES <= len) {
        const opcode = view.getUint16(offset, true);
        // flags at offset + 2 (uint16 LE) preserved verbatim in raw packet bytes.
        const payloadBytes = view.getUint32(offset + 4, true);
        const packetLen = PACKET_HEADER_BYTES + payloadBytes;

        if (offset + packetLen > len) {
          this._pendingNeededBytes = packetLen;
          break;
        }

        const packetCopy = copyBytes(buf.subarray(offset, offset + packetLen));
        const res = this._ingestPacket(opcode, packetCopy);
        packetsParsed += 1;
        completedFrames += res.completed;
        droppedFrames += res.dropped;
        offset += packetLen;
      }

      if (offset < len) {
        const remainder = copyBytes(buf.subarray(offset, len));
        this._pendingChunks = [remainder];
        this._pendingBytes = remainder.byteLength;
        if (remainder.byteLength >= PACKET_HEADER_BYTES) {
          const remView = new DataView(remainder.buffer, remainder.byteOffset, remainder.byteLength);
          this._pendingNeededBytes = PACKET_HEADER_BYTES + remView.getUint32(4, true);
        } else {
          this._pendingNeededBytes = PACKET_HEADER_BYTES;
        }
      }

      return {
        recordedBytes: srcLen,
        packetsParsed,
        completedFrames,
        droppedFrames,
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
     * Returns a JSON-safe summary of recorded frames, byte usage, opcode histogram,
     * dropped frame count, and recording state.
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
        };
      }

      return {
        abiVersion: ABI_VERSION,
        frameCount: available,
        totalBytes: this._totalBytes,
        totalPackets: this._totalPackets,
        opcodeCounts: cloneCounts(this._opcodeCounts),
        droppedFrames: this._droppedFrames,
        capturedFrames: this._capturedFrames,
        recording: this._recording,
        maxFrames: this._maxFrames,
        maxBytes: this._maxBytes,
        frames,
      };
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

  const exportsObj = {
    GpuFrameRecorder,
    createInterceptor,
    ABI_VERSION,
    PACKET_HEADER_BYTES,
    DEFAULT_MAX_FRAMES,
    DEFAULT_MAX_BYTES,
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
