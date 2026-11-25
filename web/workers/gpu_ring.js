'use strict';

(function (root) {
  // Binary-compatible with playgta6::gpu::GpuRingHeader (runtime/gpu/gpu_ring.h).
  const GPU_RING_MAGIC = 0x47505552; // "GPUR"
  const GPU_RING_VERSION = 1;
  const GPU_RING_HEADER_BYTES = 64;
  const GPU_RING_HEADER_WORDS = 16;
  const DEFAULT_CAPACITY_BYTES = 65536;
  // Binary-compatible with playgta6::GpuPacketHeader (docs/gpu_commands.md):
  //   u16 opcode (0..2), u16 flags (2..4), u32 payload_bytes (4..8).
  const GPU_PACKET_HEADER_BYTES = 8;
  const DEFAULT_BATCH_MAX_PACKETS = 256;

  // 32-bit word indices within the 64-byte header (offsets 0..64).
  const HEADER_INDEX = Object.freeze({
    MAGIC: 0,           // offset 0
    VERSION: 1,         // offset 4
    CAPACITY_BYTES: 2,  // offset 8
    RESERVED0: 3,       // offset 12
    WRITE_POS: 4,       // offset 16 (monotonic u32 byte counter)
    READ_POS: 5,        // offset 20 (monotonic u32 byte counter)
    DROPPED_PACKETS: 6, // offset 24
    NOTIFY_SEQ: 7,      // offset 28
  });

  function isPowerOfTwo(value) {
    return (
      Number.isInteger(value) &&
      value > 0 &&
      value <= 0x80000000 &&
      (value & (value - 1)) === 0
    );
  }

  function sanitizeNonNegInt(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return Math.floor(value);
  }

  function sanitizeNonNegNumber(value) {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      return 0;
    }
    return value;
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
      'GpuRingBuffer.writeBytes expects a Uint8Array, ArrayBufferView, ArrayBuffer, or SharedArrayBuffer'
    );
  }

  /**
   * Lock-free Single-Producer / Single-Consumer (SPSC) GPU command ring buffer
   * backed by a SharedArrayBuffer:
   *   [0..64)                    : 64-byte atomic control header (GpuRingHeader)
   *   [64..64 + capacity_bytes)  : power-of-two circular byte payload region
   */
  class GpuRingBuffer {
    static MAGIC = GPU_RING_MAGIC;
    static VERSION = GPU_RING_VERSION;
    static HEADER_BYTES = GPU_RING_HEADER_BYTES;
    static HEADER_WORDS = GPU_RING_HEADER_WORDS;
    static HEADER_INDEX = HEADER_INDEX;
    static PACKET_HEADER_BYTES = GPU_PACKET_HEADER_BYTES;
    static DEFAULT_BATCH_MAX_PACKETS = DEFAULT_BATCH_MAX_PACKETS;

    static byteLength(capacityBytes) {
      if (!isPowerOfTwo(capacityBytes)) {
        throw new RangeError(
          `capacityBytes must be a power of two in (0, 2^31], got ${capacityBytes}`
        );
      }
      return GPU_RING_HEADER_BYTES + capacityBytes;
    }

    static create(capacityBytes = DEFAULT_CAPACITY_BYTES) {
      if (typeof SharedArrayBuffer === 'undefined') {
        throw new Error('SharedArrayBuffer is unavailable (requires Cross-Origin Isolation)');
      }
      const totalBytes = GpuRingBuffer.byteLength(capacityBytes);
      const sab = new SharedArrayBuffer(totalBytes);
      return new GpuRingBuffer(sab, { capacityBytes, initialize: true });
    }

    static attach(sab, byteOffset = 0) {
      return new GpuRingBuffer(sab, { byteOffset, initialize: false });
    }

    constructor(sab, options = {}) {
      if (typeof SharedArrayBuffer === 'undefined' || !(sab instanceof SharedArrayBuffer)) {
        throw new TypeError('GpuRingBuffer requires a SharedArrayBuffer');
      }

      const byteOffset = options.byteOffset !== undefined ? options.byteOffset : 0;
      if (!Number.isInteger(byteOffset) || byteOffset < 0 || (byteOffset & 3) !== 0) {
        throw new RangeError(`byteOffset must be a non-negative 4-byte aligned integer, got ${byteOffset}`);
      }
      if (byteOffset + GPU_RING_HEADER_BYTES > sab.byteLength) {
        throw new RangeError(
          `SharedArrayBuffer too small (${sab.byteLength} bytes; need at least ${byteOffset + GPU_RING_HEADER_BYTES})`
        );
      }

      this._sab = sab;
      this._byteOffset = byteOffset;
      this._headerI32 = new Int32Array(sab, byteOffset, GPU_RING_HEADER_WORDS);
      this._headerU32 = new Uint32Array(sab, byteOffset, GPU_RING_HEADER_WORDS);

      const existingMagic = Atomics.load(this._headerU32, HEADER_INDEX.MAGIC);
      const shouldInit =
        options.initialize === true ||
        (options.initialize === undefined && existingMagic === 0);

      if (shouldInit) {
        const availableDataBytes = sab.byteLength - byteOffset - GPU_RING_HEADER_BYTES;
        const capacityBytes =
          options.capacityBytes !== undefined ? options.capacityBytes : availableDataBytes;
        if (!isPowerOfTwo(capacityBytes)) {
          throw new RangeError(
            `capacityBytes must be a power of two in (0, 2^31], got ${capacityBytes}`
          );
        }
        if (capacityBytes > availableDataBytes) {
          throw new RangeError(
            `capacityBytes (${capacityBytes}) exceeds available SharedArrayBuffer bytes (${availableDataBytes})`
          );
        }

        for (let i = 0; i < GPU_RING_HEADER_WORDS; i++) {
          Atomics.store(this._headerU32, i, 0);
        }
        Atomics.store(this._headerU32, HEADER_INDEX.VERSION, GPU_RING_VERSION);
        Atomics.store(this._headerU32, HEADER_INDEX.CAPACITY_BYTES, capacityBytes);
        Atomics.store(this._headerU32, HEADER_INDEX.RESERVED0, 0);
        Atomics.store(this._headerU32, HEADER_INDEX.WRITE_POS, 0);
        Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, 0);
        Atomics.store(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 0);
        Atomics.store(this._headerU32, HEADER_INDEX.NOTIFY_SEQ, 0);
        // Publish magic last so concurrent attachers observe initialized metadata.
        Atomics.store(this._headerU32, HEADER_INDEX.MAGIC, GPU_RING_MAGIC);

        this._capacityBytes = capacityBytes;
      } else {
        if (existingMagic !== GPU_RING_MAGIC) {
          throw new Error(
            `Invalid GpuRingHeader magic: expected 0x${GPU_RING_MAGIC.toString(16)}, got 0x${existingMagic.toString(16)}`
          );
        }
        const version = Atomics.load(this._headerU32, HEADER_INDEX.VERSION);
        if (version !== GPU_RING_VERSION) {
          throw new Error(
            `Unsupported GpuRingHeader version: expected ${GPU_RING_VERSION}, got ${version}`
          );
        }
        const capacityBytes = Atomics.load(this._headerU32, HEADER_INDEX.CAPACITY_BYTES);
        if (!isPowerOfTwo(capacityBytes)) {
          throw new RangeError(`Invalid GpuRingHeader capacity_bytes: ${capacityBytes}`);
        }
        const requiredBytes = byteOffset + GPU_RING_HEADER_BYTES + capacityBytes;
        if (sab.byteLength < requiredBytes) {
          throw new RangeError(
            `SharedArrayBuffer (${sab.byteLength} bytes) is smaller than header requires (${requiredBytes} bytes)`
          );
        }
        this._capacityBytes = capacityBytes;
      }

      this._mask = (this._capacityBytes - 1) >>> 0;
      this._data = new Uint8Array(
        sab,
        byteOffset + GPU_RING_HEADER_BYTES,
        this._capacityBytes
      );
      this._dataView = new DataView(
        sab,
        byteOffset + GPU_RING_HEADER_BYTES,
        this._capacityBytes
      );
      this._headerScratch = new Uint8Array(GPU_PACKET_HEADER_BYTES);
      this._headerScratchView = new DataView(this._headerScratch.buffer);
      this._scratch = null;

      // Local throughput & backpressure telemetry (kept outside the 64-byte SAB ABI header).
      this._batchesDrained = 0;
      this._packetsDrained = 0;
      this._maxBatchSize = 0;
      this._backpressureWaits = 0;
      this._peakOccupancyBytes = 0;
    }

    _recordOccupancy(bytes) {
      if (typeof bytes !== 'number' || !Number.isFinite(bytes) || bytes <= 0) {
        return;
      }
      const clamped = bytes > this._capacityBytes ? this._capacityBytes : Math.floor(bytes);
      if (clamped > this._peakOccupancyBytes) {
        this._peakOccupancyBytes = clamped;
      }
    }

    get sab() {
      return this._sab;
    }

    get byteOffset() {
      return this._byteOffset;
    }

    get capacityBytes() {
      return this._capacityBytes;
    }

    get magic() {
      return Atomics.load(this._headerU32, HEADER_INDEX.MAGIC);
    }

    get version() {
      return Atomics.load(this._headerU32, HEADER_INDEX.VERSION);
    }

    get writePos() {
      return Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
    }

    get readPos() {
      return Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
    }

    get droppedPackets() {
      return Atomics.load(this._headerU32, HEADER_INDEX.DROPPED_PACKETS);
    }

    get notifySeq() {
      return Atomics.load(this._headerU32, HEADER_INDEX.NOTIFY_SEQ);
    }

    availableBytes() {
      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      const avail = (writePos - readPos) >>> 0;
      this._recordOccupancy(avail);
      return avail;
    }

    freeBytes() {
      const used = this.availableBytes();
      return used >= this._capacityBytes ? 0 : this._capacityBytes - used;
    }

    /**
     * Records an external producer backpressure wait (e.g. Atomics.wait / retry backoff).
     *
     * @param {number} [count=1]
     * @returns {number} Updated backpressure wait count.
     */
    recordBackpressureWait(count = 1) {
      const n = sanitizeNonNegInt(count);
      if (n > 0) {
        this._backpressureWaits += n;
      }
      return sanitizeNonNegInt(this._backpressureWaits);
    }

    /**
     * All-or-nothing byte write supporting wrap-around.
     * Returns false (and increments dropped_packets) without modifying write_pos
     * or ring contents when the ring does not have enough free bytes.
     *
     * @param {Uint8Array|ArrayBufferView|ArrayBuffer|SharedArrayBuffer} input
     * @returns {boolean}
     */
    writeBytes(input) {
      const src = toUint8Array(input);
      const len = src.byteLength;
      if (len === 0) {
        return true;
      }
      if (len > this._capacityBytes) {
        Atomics.add(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 1);
        return false;
      }

      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      const used = (writePos - readPos) >>> 0;
      this._recordOccupancy(used);
      const free = used >= this._capacityBytes ? 0 : this._capacityBytes - used;
      if (len > free) {
        this._backpressureWaits += 1;
        Atomics.add(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 1);
        return false;
      }

      const offset = (writePos & this._mask) >>> 0;
      const firstLen = Math.min(len, this._capacityBytes - offset);
      this._data.set(src.subarray(0, firstLen), offset);
      if (len > firstLen) {
        this._data.set(src.subarray(firstLen, len), 0);
      }

      const nextWrite = (writePos + len) >>> 0;
      Atomics.store(this._headerU32, HEADER_INDEX.WRITE_POS, nextWrite);
      Atomics.add(this._headerI32, HEADER_INDEX.NOTIFY_SEQ, 1);
      Atomics.notify(this._headerI32, HEADER_INDEX.NOTIFY_SEQ, 1);
      this._recordOccupancy(used + len);
      return true;
    }

    /**
     * Drains all currently queued bytes and passes a single contiguous Uint8Array
     * to `callback`.
     * - Zero-copy subarray directly over the SharedArrayBuffer when not wrapping.
     * - Linearized into a reusable scratch Uint8Array on wrap-around so framed
     *   GpuPacketHeader streams are never split or corrupted at the ring boundary.
     *
     * @param {(bytes: Uint8Array) => void} callback
     * @returns {number} Number of bytes drained.
     */
    drainTo(callback) {
      if (typeof callback !== 'function') {
        throw new TypeError('GpuRingBuffer.drainTo requires a callback function');
      }

      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const avail = (writePos - readPos) >>> 0;
      if (avail === 0) {
        return 0;
      }
      this._recordOccupancy(avail);

      const offset = (readPos & this._mask) >>> 0;
      let slice;
      if (offset + avail <= this._capacityBytes) {
        slice = this._data.subarray(offset, offset + avail);
      } else {
        if (!this._scratch || this._scratch.byteLength < avail) {
          this._scratch = new Uint8Array(avail);
        }
        const firstLen = this._capacityBytes - offset;
        const secondLen = avail - firstLen;
        this._scratch.set(this._data.subarray(offset, this._capacityBytes), 0);
        this._scratch.set(this._data.subarray(0, secondLen), firstLen);
        slice = this._scratch.subarray(0, avail);
      }

      const nextRead = (readPos + avail) >>> 0;
      try {
        callback(slice);
      } finally {
        Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, nextRead);
        Atomics.notify(this._headerI32, HEADER_INDEX.READ_POS, 1);
      }
      return avail;
    }

    /**
     * Drains up to `maxPackets` complete framed GpuPacketHeader packets in a single
     * bounded batch, publishing the updated `read_pos` once atomically at batch end.
     * - Non-wrapping packets are passed as zero-copy subarrays directly over the SAB.
     * - Only a packet that physically straddles the ring wrap boundary is linearized
     *   into the reusable internal scratch buffer.
     * - If `handler` throws mid-batch, only packets that completed prior to the error
     *   are committed via `read_pos`; the failing packet and all subsequent packets
     *   remain unread in the ring.
     * - Views (`packet.bytes` / `packet.payload`) are only guaranteed valid for the
     *   duration of the `handler` call; copy them if retaining across drains.
     *
     * @param {(packet: {opcode: number, flags: number, payloadBytes: number, payload: Uint8Array, bytes: Uint8Array, byteLength: number, wrapped: boolean}, index: number) => void} handler
     * @param {number} [maxPackets=256]
     * @returns {{packets: number, bytes: number, remainingBytes: number, wrapped: boolean}}
     */
    drainBatch(handler, maxPackets = DEFAULT_BATCH_MAX_PACKETS) {
      if (typeof handler !== 'function') {
        throw new TypeError('GpuRingBuffer.drainBatch requires a handler function');
      }
      let limit;
      if (maxPackets === undefined || maxPackets === null) {
        limit = DEFAULT_BATCH_MAX_PACKETS;
      } else if (
        typeof maxPackets !== 'number' ||
        !Number.isSafeInteger(maxPackets) ||
        maxPackets < 1
      ) {
        throw new RangeError(
          `GpuRingBuffer.drainBatch maxPackets must be a positive safe integer, got ${maxPackets}`
        );
      } else {
        limit = maxPackets;
      }

      const readPos = Atomics.load(this._headerU32, HEADER_INDEX.READ_POS);
      const writePos = Atomics.load(this._headerU32, HEADER_INDEX.WRITE_POS);
      const avail = (writePos - readPos) >>> 0;
      if (avail === 0) {
        return { packets: 0, bytes: 0, remainingBytes: 0, wrapped: false };
      }

      const boundedAvail = avail > this._capacityBytes ? this._capacityBytes : avail;
      this._recordOccupancy(boundedAvail);

      const startOffset = (readPos & this._mask) >>> 0;
      let cursor = 0;
      let completedPackets = 0;
      let completedBytes = 0;

      try {
        while (completedPackets < limit) {
          const remaining = boundedAvail - cursor;
          if (remaining < GPU_PACKET_HEADER_BYTES) {
            break;
          }

          const pktOffset = (((readPos + cursor) >>> 0) & this._mask) >>> 0;
          let opcode;
          let flags;
          let payloadBytes;

          if (pktOffset + GPU_PACKET_HEADER_BYTES <= this._capacityBytes) {
            opcode = this._dataView.getUint16(pktOffset, true);
            flags = this._dataView.getUint16(pktOffset + 2, true);
            payloadBytes = this._dataView.getUint32(pktOffset + 4, true);
          } else {
            const firstHdr = this._capacityBytes - pktOffset;
            this._headerScratch.set(
              this._data.subarray(pktOffset, this._capacityBytes),
              0
            );
            this._headerScratch.set(
              this._data.subarray(0, GPU_PACKET_HEADER_BYTES - firstHdr),
              firstHdr
            );
            opcode = this._headerScratchView.getUint16(0, true);
            flags = this._headerScratchView.getUint16(2, true);
            payloadBytes = this._headerScratchView.getUint32(4, true);
          }

          if (payloadBytes > remaining - GPU_PACKET_HEADER_BYTES) {
            break;
          }

          const packetLen = GPU_PACKET_HEADER_BYTES + payloadBytes;
          let bytes;
          let payload;
          let packetWrapped = false;

          if (pktOffset + packetLen <= this._capacityBytes) {
            bytes = this._data.subarray(pktOffset, pktOffset + packetLen);
            payload = this._data.subarray(
              pktOffset + GPU_PACKET_HEADER_BYTES,
              pktOffset + packetLen
            );
          } else {
            packetWrapped = true;
            if (!this._scratch || this._scratch.byteLength < packetLen) {
              this._scratch = new Uint8Array(packetLen);
            }
            const firstLen = this._capacityBytes - pktOffset;
            const secondLen = packetLen - firstLen;
            this._scratch.set(this._data.subarray(pktOffset, this._capacityBytes), 0);
            this._scratch.set(this._data.subarray(0, secondLen), firstLen);
            bytes = this._scratch.subarray(0, packetLen);
            payload = this._scratch.subarray(GPU_PACKET_HEADER_BYTES, packetLen);
          }

          handler(
            {
              opcode,
              flags,
              payloadBytes,
              payload,
              bytes,
              byteLength: packetLen,
              wrapped: packetWrapped,
            },
            completedPackets
          );

          cursor += packetLen;
          completedBytes = cursor;
          completedPackets += 1;
        }
      } finally {
        if (completedBytes > 0) {
          const nextRead = (readPos + completedBytes) >>> 0;
          Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, nextRead);
          Atomics.notify(this._headerI32, HEADER_INDEX.READ_POS, 1);
        }
        if (completedPackets > 0) {
          this._batchesDrained += 1;
          this._packetsDrained += completedPackets;
          if (completedPackets > this._maxBatchSize) {
            this._maxBatchSize = completedPackets;
          }
        }
      }

      const batchWrapped =
        completedBytes > 0 && startOffset + completedBytes > this._capacityBytes;

      return {
        packets: completedPackets,
        bytes: completedBytes,
        remainingBytes: boundedAvail - completedBytes,
        wrapped: batchWrapped,
      };
    }

    reset() {
      Atomics.store(this._headerU32, HEADER_INDEX.WRITE_POS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.READ_POS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.DROPPED_PACKETS, 0);
      Atomics.store(this._headerU32, HEADER_INDEX.NOTIFY_SEQ, 0);
      this._batchesDrained = 0;
      this._packetsDrained = 0;
      this._maxBatchSize = 0;
      this._backpressureWaits = 0;
      this._peakOccupancyBytes = 0;
    }

    getStats() {
      const writePos = this.writePos;
      const readPos = this.readPos;
      const availableBytes = (writePos - readPos) >>> 0;
      this._recordOccupancy(availableBytes);
      const freeBytes =
        availableBytes >= this._capacityBytes ? 0 : this._capacityBytes - availableBytes;
      const batchesDrained = sanitizeNonNegInt(this._batchesDrained);
      const packetsDrained = sanitizeNonNegInt(this._packetsDrained);
      const maxBatchSize = sanitizeNonNegInt(this._maxBatchSize);
      const avgBatchSize =
        batchesDrained > 0 ? sanitizeNonNegNumber(packetsDrained / batchesDrained) : 0;
      const backpressureWaits = sanitizeNonNegInt(this._backpressureWaits);
      const peakOccupancyBytes = Math.min(
        this._capacityBytes,
        sanitizeNonNegInt(this._peakOccupancyBytes)
      );
      return {
        capacityBytes: this._capacityBytes,
        availableBytes,
        freeBytes,
        writePos,
        readPos,
        droppedPackets: this.droppedPackets,
        notifySeq: this.notifySeq,
        batchesDrained,
        maxBatchSize,
        avgBatchSize,
        packetsDrained,
        backpressureWaits,
        peakOccupancyBytes,
      };
    }
  }

  const exportsObj = {
    GpuRingBuffer,
    GPU_RING_MAGIC,
    GPU_RING_VERSION,
    GPU_RING_HEADER_BYTES,
    GPU_RING_HEADER_WORDS,
    GPU_PACKET_HEADER_BYTES,
    DEFAULT_CAPACITY_BYTES,
    DEFAULT_BATCH_MAX_PACKETS,
    HEADER_INDEX,
    isPowerOfTwo,
  };

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6GpuRing = exportsObj;
  } else if (root) {
    root.PlayGta6GpuRing = exportsObj;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportsObj;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
