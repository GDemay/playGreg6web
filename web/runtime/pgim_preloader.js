'use strict';

// Streaming PGIM v1 guest image + boot shader bundle preloader.
// Validates the PGIM header from the first HTTP 206 Range chunk, then streams the rest of the
// container in parallel Range chunks (or a single GET when Range is unsupported) into one
// contiguous ArrayBuffer / SharedArrayBuffer before guest entry.
// The binary layout matches recomp/src/image.rs (authoritative) and runtime/os/image_loader.h.
// Zero external dependencies; UMD export for Browser/Worker (globalThis.PlayGta6PgimPreloader)
// and Node.js (module.exports).

(function (root) {

  /** Container magic ("PGIM"). */
  const PGIM_MAGIC = 'PGIM';
  /** Container magic as raw ASCII bytes. */
  const PGIM_MAGIC_BYTES = Object.freeze([0x50, 0x47, 0x49, 0x4d]);
  /** Container format version (IMAGE_VERSION / pgim::kVersion). */
  const PGIM_VERSION = 1;
  /** Header size in bytes (IMAGE_HEADER_BYTES / pgim::kHeaderBytes). */
  const PGIM_HEADER_BYTES = 64;
  /** Section table entry size in bytes (IMAGE_SECTION_ENTRY_BYTES / pgim::kSectionEntryBytes). */
  const PGIM_SECTION_ENTRY_BYTES = 64;
  /** Payload alignment / guest page size (IMAGE_PAGE_SIZE / pgim::kPayloadAlignment). */
  const PGIM_PAGE_SIZE = 4096;

  /** Header flags (FLAG_* in image.rs, kFlag* in image_loader.h). */
  const PGIM_FLAGS = Object.freeze({
    HEADERS_MAPPED: 1 << 0,
    EXECUTABLE_BYTES_OMITTED: 1 << 1,
    RELOCS_STRIPPED: 1 << 2,
  });

  /** Section flags (SECTION_FLAG_* in image.rs, kSectionFlag* in image_loader.h). */
  const PGIM_SECTION_FLAGS = Object.freeze({
    HEADERS: 1 << 0,
    BYTES_OMITTED: 1 << 1,
  });

  /** Section protection bits (kProt* in image_loader.h). */
  const PGIM_PROT = Object.freeze({ READ: 1 << 0, WRITE: 1 << 1, EXECUTE: 1 << 2 });

  /** S6 reflection manifest schema version accepted by web/workers/reflection_loader.js. */
  const REFLECTION_SCHEMA_VERSION = 1;
  const WGPK_MAGIC_BYTES = [0x57, 0x47, 0x50, 0x4b];
  const WGPK_VERSION = 1;
  const WGPK_HEADER_BYTES = 16;
  const WGPK_ENTRY_BYTES = 20;

  const SECTION_NAME_BYTES = 16;
  const KNOWN_HEADER_FLAGS = PGIM_FLAGS.HEADERS_MAPPED | PGIM_FLAGS.EXECUTABLE_BYTES_OMITTED | PGIM_FLAGS.RELOCS_STRIPPED;
  const KNOWN_SECTION_FLAGS = PGIM_SECTION_FLAGS.HEADERS | PGIM_SECTION_FLAGS.BYTES_OMITTED;
  const U64_MAX = 0xffffffffffffffffn;
  const DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024;
  const DEFAULT_CONCURRENCY = 4;
  const DEFAULT_MAX_RETRIES = 3;
  const DEFAULT_BASE_RETRY_DELAY_MS = 50;

  function defaultSleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function isRetryableHttpStatus(status) {
    return status === 408 || status === 425 || status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
  }

  function isRetryableNetworkError(err) {
    if (!err || typeof err !== 'object') return false;
    if (err.name === 'AbortError' || err.name === 'TimeoutError') return false;
    if (err instanceof PgimPreloadError) {
      return err.code === 'NETWORK_ERROR' || (err.code === 'HTTP_ERROR' && isRetryableHttpStatus(err.status));
    }
    // Fetch network failures typically surface as TypeError.
    return err.name === 'TypeError' || err.code === 'ECONNRESET' || err.code === 'ETIMEDOUT';
  }

  /** Error raised for corrupt/truncated containers and failed transfers. `code` mirrors ImageError. */
  class PgimPreloadError extends Error {
    constructor(code, message, details = {}) {
      super(message, details.cause !== undefined ? { cause: details.cause } : undefined);
      this.name = 'PgimPreloadError';
      this.code = code;
      for (const [key, value] of Object.entries(details)) {
        if (key !== 'cause') this[key] = value;
      }
    }
  }

  function isSharedArrayBuffer(buf) {
    return typeof SharedArrayBuffer !== 'undefined' && buf instanceof SharedArrayBuffer;
  }

  function toBytes(input, label = 'buffer') {
    if (input instanceof Uint8Array) return input;
    if (ArrayBuffer.isView(input)) return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
    if (input instanceof ArrayBuffer || isSharedArrayBuffer(input)) return new Uint8Array(input);
    throw new TypeError(`Expected ${label} to be an ArrayBuffer, SharedArrayBuffer, or ArrayBufferView`);
  }

  function hexBytes(bytes) {
    return Array.from(bytes, (b) => `0x${b.toString(16).padStart(2, '0')}`).join(' ');
  }

  function now() {
    return typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now();
  }

  /**
   * FNV-1a 64 of `input` (matches `fnv1a_64` in recomp/src/image.rs). Uses 16-bit limbs so large
   * payloads hash without per-byte BigInt arithmetic.
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} input
   * @returns {bigint}
   */
  function pgimFnv1a64(input) {
    const bytes = toBytes(input, 'input');
    let h0 = 0x2325;
    let h1 = 0x8422;
    let h2 = 0x9ce4;
    let h3 = 0xcbf2;
    for (let i = 0; i < bytes.length; i += 1) {
      h0 ^= bytes[i];
      // prime = 2^40 + 0x1b3: limb i contributes 0x1b3 to limb i and 0x100 to limb i + 2.
      const t0 = h0 * 0x1b3;
      let t1 = h1 * 0x1b3 + (t0 >>> 16);
      let t2 = h2 * 0x1b3 + h0 * 0x100;
      let t3 = h3 * 0x1b3 + h1 * 0x100;
      h0 = t0 & 0xffff;
      h1 = t1 & 0xffff;
      t2 += t1 >>> 16;
      h2 = t2 & 0xffff;
      t3 += t2 >>> 16;
      h3 = t3 & 0xffff;
    }
    return (BigInt(h3) << 48n) | (BigInt(h2) << 32n) | (BigInt(h1) << 16n) | BigInt(h0);
  }

  /**
   * Parse and strictly validate the 64-byte PGIM v1 header (same checks and order as `read_image`).
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} input At least the first 64 bytes.
   * @param {{ totalBytes?: number | null }} [options] Transport size (Content-Length / Content-Range
   *   total / buffer length); when given, the header `file_size` must equal it.
   */
  function parsePgimHeader(input, options = {}) {
    const bytes = toBytes(input, 'PGIM header');
    if (bytes.byteLength < PGIM_HEADER_BYTES) {
      throw new PgimPreloadError(
        'TRUNCATED',
        `truncated PGIM image header: needed ${PGIM_HEADER_BYTES} bytes, only ${bytes.byteLength} available`,
        { needed: PGIM_HEADER_BYTES, available: bytes.byteLength }
      );
    }
    for (let i = 0; i < 4; i += 1) {
      if (bytes[i] !== PGIM_MAGIC_BYTES[i]) {
        throw new PgimPreloadError(
          'BAD_MAGIC',
          `invalid PGIM image magic [${hexBytes(bytes.subarray(0, 4))}] (expected "${PGIM_MAGIC}")`
        );
      }
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, PGIM_HEADER_BYTES);
    const version = view.getUint32(4, true);
    if (version !== PGIM_VERSION) {
      throw new PgimPreloadError('UNSUPPORTED_VERSION', `unsupported PGIM image version ${version} (expected ${PGIM_VERSION})`, {
        version,
      });
    }
    const badField = (field, value, extra = '') =>
      new PgimPreloadError('BAD_HEADER_FIELD', `invalid PGIM header field ${field} = 0x${value.toString(16)}${extra}`, {
        field,
        value,
      });
    const expectU32 = (at, field, expected) => {
      const value = view.getUint32(at, true);
      if (value !== expected) throw badField(field, value);
    };
    expectU32(8, 'header_bytes', PGIM_HEADER_BYTES);
    expectU32(12, 'section_entry_bytes', PGIM_SECTION_ENTRY_BYTES);
    expectU32(20, 'page_size', PGIM_PAGE_SIZE);
    const flags = view.getUint32(16, true);
    if ((flags & ~KNOWN_HEADER_FLAGS) !== 0) throw badField('flags', flags);

    const imageBase = view.getBigUint64(24, true);
    const sizeOfImage = view.getUint32(32, true);
    const sectionAlignment = view.getUint32(36, true);
    const entryPointRva = view.getUint32(40, true);
    const sectionCount = view.getUint32(44, true);
    const dir64Count = view.getUint32(48, true);
    const highlowCount = view.getUint32(52, true);
    const fileSizeBig = view.getBigUint64(56, true);

    if (imageBase + BigInt(sizeOfImage) > U64_MAX) throw badField('image_base + size_of_image', imageBase);
    if (fileSizeBig > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw badField('file_size', fileSizeBig, ' (exceeds the JavaScript addressable range)');
    }
    const fileSize = Number(fileSizeBig);
    const { totalBytes = null } = options;
    if (totalBytes !== null && totalBytes !== undefined) {
      if (!Number.isSafeInteger(totalBytes) || totalBytes < 0) {
        throw new TypeError(`totalBytes must be a non-negative safe integer, got: ${totalBytes}`);
      }
      if (fileSize !== totalBytes) {
        throw badField('file_size', fileSizeBig, ` (transport reports ${totalBytes} bytes)`);
      }
    }

    // All counts are u32, so these stay well inside Number.MAX_SAFE_INTEGER.
    const relocTableOffset = PGIM_HEADER_BYTES + sectionCount * PGIM_SECTION_ENTRY_BYTES;
    const tablesEnd = relocTableOffset + (dir64Count + highlowCount) * 4;
    if (tablesEnd > fileSize) {
      throw new PgimPreloadError(
        'TRUNCATED',
        `truncated PGIM section and relocation tables: needed ${tablesEnd} bytes, only ${fileSize} available`,
        { needed: tablesEnd, available: fileSize }
      );
    }

    return {
      magic: PGIM_MAGIC,
      version,
      flags,
      headersMapped: (flags & PGIM_FLAGS.HEADERS_MAPPED) !== 0,
      executableBytesOmitted: (flags & PGIM_FLAGS.EXECUTABLE_BYTES_OMITTED) !== 0,
      relocsStripped: (flags & PGIM_FLAGS.RELOCS_STRIPPED) !== 0,
      imageBase,
      sizeOfImage,
      sectionAlignment,
      entryPointRva,
      entryPoint: entryPointRva === 0 ? 0n : imageBase + BigInt(entryPointRva),
      sectionCount,
      dir64Count,
      highlowCount,
      relocCount: dir64Count + highlowCount,
      fileSize,
      sectionTableOffset: PGIM_HEADER_BYTES,
      relocTableOffset,
      tablesEnd,
    };
  }

  const nameDecoder = new TextDecoder('utf-8', { fatal: true });

  function sectionAt(sections, rva) {
    let lo = 0;
    let hi = sections.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (sections[mid].rva <= rva) lo = mid + 1;
      else hi = mid;
    }
    if (lo === 0) return null;
    const section = sections[lo - 1];
    return rva < section.rva + section.virtualSize ? section : null;
  }

  /**
   * Parse and validate the header, section table, and relocation table of a complete container
   * (every structural check of `read_image` + `GuestImage::validate`). Payload checksums are only
   * verified with `verifyChecksums` (the C++ loader verifies them while mapping anyway).
   *
   * @param {ArrayBuffer | SharedArrayBuffer | ArrayBufferView} input Whole container.
   * @param {{ verifyChecksums?: boolean }} [options]
   */
  function parsePgimLayout(input, options = {}) {
    const bytes = toBytes(input, 'PGIM image');
    const header = parsePgimHeader(bytes, { totalBytes: bytes.byteLength });
    const verifyChecksums = Boolean(options.verifyChecksums);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const fileLen = bytes.byteLength;
    const invalid = (index, reason) =>
      new PgimPreloadError('INVALID_SECTION', `PGIM section ${index}: ${reason}`, { index, reason });

    const sections = new Array(header.sectionCount);
    const payloads = [];
    for (let index = 0; index < header.sectionCount; index += 1) {
      const at = PGIM_HEADER_BYTES + index * PGIM_SECTION_ENTRY_BYTES;
      const rawName = bytes.subarray(at, at + SECTION_NAME_BYTES);
      let nameLen = rawName.indexOf(0);
      if (nameLen < 0) nameLen = SECTION_NAME_BYTES;
      for (let i = nameLen; i < SECTION_NAME_BYTES; i += 1) {
        if (rawName[i] !== 0) throw invalid(index, 'name padding is not zero');
      }
      let name;
      try {
        name = nameDecoder.decode(rawName.slice(0, nameLen));
      } catch {
        throw invalid(index, 'name is not UTF-8');
      }
      const protection = view.getUint32(at + 24, true);
      if ((protection & ~0b111) !== 0) throw invalid(index, 'unknown protection bits');
      const sectionFlags = view.getUint32(at + 28, true);
      if ((sectionFlags & ~KNOWN_SECTION_FLAGS) !== 0) throw invalid(index, 'unknown section flags');
      if (view.getUint32(at + 60, true) !== 0) throw invalid(index, 'reserved field is not zero');
      const offsetBig = view.getBigUint64(at + 32, true);
      const lengthBig = view.getBigUint64(at + 40, true);
      const checksum = view.getBigUint64(at + 48, true);

      let payloadOffset = 0;
      let payloadLength = 0;
      if (lengthBig === 0n) {
        if (offsetBig !== 0n) throw invalid(index, 'empty payload with non-zero offset');
      } else {
        if (offsetBig % BigInt(PGIM_PAGE_SIZE) !== 0n) {
          throw new PgimPreloadError(
            'PAYLOAD_MISALIGNED',
            `PGIM section ${index} payload offset 0x${offsetBig.toString(16)} is not ${PGIM_PAGE_SIZE}-byte aligned`,
            { index, offset: offsetBig }
          );
        }
        const end = offsetBig + lengthBig;
        if (end > BigInt(fileLen) || offsetBig < BigInt(header.tablesEnd)) {
          throw new PgimPreloadError(
            'PAYLOAD_OUT_OF_RANGE',
            `PGIM section ${index} payload [0x${offsetBig.toString(16)}..+0x${lengthBig.toString(16)}] is outside the container (0x${fileLen.toString(16)} bytes) or its tables`,
            { index, offset: offsetBig, length: lengthBig }
          );
        }
        payloadOffset = Number(offsetBig);
        payloadLength = Number(lengthBig);
        payloads.push([payloadOffset, payloadOffset + payloadLength, index]);
      }
      if (verifyChecksums) {
        const actual = pgimFnv1a64(bytes.subarray(payloadOffset, payloadOffset + payloadLength));
        if (actual !== checksum) {
          throw new PgimPreloadError(
            'CHECKSUM_MISMATCH',
            `PGIM section ${index} checksum mismatch: stored 0x${checksum.toString(16).padStart(16, '0')}, computed 0x${actual.toString(16).padStart(16, '0')}`,
            { index, expected: checksum, actual }
          );
        }
      }
      sections[index] = {
        name,
        rva: view.getUint32(at + 16, true),
        virtualSize: view.getUint32(at + 20, true),
        protection,
        flags: sectionFlags,
        isHeaders: (sectionFlags & PGIM_SECTION_FLAGS.HEADERS) !== 0,
        bytesOmitted: (sectionFlags & PGIM_SECTION_FLAGS.BYTES_OMITTED) !== 0,
        payloadOffset,
        payloadLength,
        checksum,
        characteristics: view.getUint32(at + 56, true),
      };
    }

    payloads.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    for (let i = 1; i < payloads.length; i += 1) {
      if (payloads[i - 1][1] > payloads[i][0]) {
        throw new PgimPreloadError('PAYLOAD_OVERLAP', `PGIM section ${payloads[i][2]} payload overlaps another payload`, {
          index: payloads[i][2],
        });
      }
    }

    const readRvas = (start, count) => {
      const out = new Uint32Array(count);
      for (let i = 0; i < count; i += 1) {
        const rva = view.getUint32(start + i * 4, true);
        if (i > 0 && out[i - 1] >= rva) {
          throw new PgimPreloadError('RELOCATIONS_UNSORTED', `PGIM relocation table is not strictly ascending at 0x${rva.toString(16)}`, {
            rva,
          });
        }
        out[i] = rva;
      }
      return out;
    };
    const dir64 = readRvas(header.relocTableOffset, header.dir64Count);
    const highlow = readRvas(header.relocTableOffset + header.dir64Count * 4, header.highlowCount);

    // GuestImage::validate
    const firstIsHeaders = sections.length > 0 && sections[0].isHeaders;
    if (header.headersMapped !== firstIsHeaders) throw invalid(0, 'headers flag does not match the first section');
    const omit = header.executableBytesOmitted;
    for (let index = 0; index < sections.length; index += 1) {
      const s = sections[index];
      const execute = (s.protection & PGIM_PROT.EXECUTE) !== 0;
      if (s.virtualSize === 0) throw invalid(index, 'zero in-memory size');
      if (s.payloadLength > s.virtualSize) throw invalid(index, 'initialised bytes exceed in-memory size');
      if (s.isHeaders && (index !== 0 || s.rva !== 0)) throw invalid(index, 'headers pseudo-section must be first and at RVA 0');
      if (s.bytesOmitted && !(omit && execute && s.payloadLength === 0)) {
        throw invalid(index, 'omitted bytes only allowed for empty executable sections');
      }
      if (omit && execute && !s.isHeaders && !s.bytesOmitted) throw invalid(index, 'executable section bytes must be omitted');
      if (s.rva + s.virtualSize > header.sizeOfImage) {
        throw new PgimPreloadError(
          'SECTION_OUT_OF_IMAGE',
          `PGIM section ${s.name} [0x${s.rva.toString(16)}..+0x${s.virtualSize.toString(16)}] exceeds SizeOfImage 0x${header.sizeOfImage.toString(16)}`,
          { index }
        );
      }
      if (index > 0) {
        const prev = sections[index - 1];
        if (prev.rva + prev.virtualSize > s.rva) {
          throw new PgimPreloadError('OVERLAPPING_SECTIONS', `PGIM sections ${prev.name} and ${s.name} overlap`, { index });
        }
      }
    }

    // Merge both lists in (rva, kind) order, as validate() walks the combined sorted table.
    let i = 0;
    let j = 0;
    let prevRva = -1;
    let prevEnd = 0;
    while (i < dir64.length || j < highlow.length) {
      let rva;
      let width;
      if (j >= highlow.length || (i < dir64.length && dir64[i] < highlow[j])) {
        rva = dir64[i++];
        width = 8;
      } else {
        rva = highlow[j++];
        width = 4;
      }
      if (prevRva >= rva) {
        throw new PgimPreloadError('RELOCATIONS_UNSORTED', `PGIM relocation table is not strictly ascending at 0x${rva.toString(16)}`, {
          rva,
        });
      }
      if (prevEnd > rva) {
        throw new PgimPreloadError(
          'OVERLAPPING_RELOCATIONS',
          `PGIM relocation sites 0x${prevRva.toString(16)} and 0x${rva.toString(16)} overlap`,
          { first: prevRva, second: rva }
        );
      }
      const end = rva + width;
      if (end > header.sizeOfImage) {
        throw new PgimPreloadError(
          'RELOC_SITE_OUT_OF_RANGE',
          `PGIM relocation site 0x${rva.toString(16)} (+${width}) is outside 0x${header.sizeOfImage.toString(16)}`,
          { rva }
        );
      }
      const first = sectionAt(sections, rva);
      const last = sectionAt(sections, end - 1);
      const mapped =
        first !== null &&
        last !== null &&
        !first.bytesOmitted &&
        !last.bytesOmitted &&
        (first === last || first.rva + first.virtualSize === last.rva);
      if (!mapped) {
        throw new PgimPreloadError(
          'RELOC_SITE_UNMAPPED',
          `PGIM relocation site 0x${rva.toString(16)} is not inside a mapped section with stored bytes`,
          { rva }
        );
      }
      prevRva = rva;
      prevEnd = end;
    }

    return { header, sections, dir64, highlow };
  }

  function abortReason(signal) {
    if (signal && signal.reason !== undefined) return signal.reason;
    return new DOMException('The operation was aborted.', 'AbortError');
  }

  function throwIfAborted(signal) {
    if (signal && signal.aborted) throw abortReason(signal);
  }

  function raceAbort(promise, signal) {
    if (!signal) return promise;
    if (signal.aborted) {
      Promise.resolve(promise).catch(() => {});
      return Promise.reject(abortReason(signal));
    }
    return new Promise((resolve, reject) => {
      const onAbort = () => reject(abortReason(signal));
      signal.addEventListener('abort', onAbort, { once: true });
      Promise.resolve(promise).then(
        (value) => {
          signal.removeEventListener('abort', onAbort);
          resolve(value);
        },
        (err) => {
          signal.removeEventListener('abort', onAbort);
          reject(err);
        }
      );
    });
  }

  /** Child controller that aborts when `outer` does; returns `{ controller, unlink }`. */
  function linkSignal(outer) {
    const controller = new AbortController();
    if (!outer) return { controller, unlink: () => {} };
    if (outer.aborted) {
      controller.abort(abortReason(outer));
      return { controller, unlink: () => {} };
    }
    const onAbort = () => controller.abort(abortReason(outer));
    outer.addEventListener('abort', onAbort, { once: true });
    return { controller, unlink: () => outer.removeEventListener('abort', onAbort) };
  }

  function headerValue(response, name) {
    const headers = response && response.headers;
    if (!headers || typeof headers.get !== 'function') return null;
    const value = headers.get(name);
    return value === undefined ? null : value;
  }

  function contentLength(response) {
    const encoding = headerValue(response, 'content-encoding');
    if (encoding && encoding.trim().toLowerCase() !== 'identity') return null;
    const raw = headerValue(response, 'content-length');
    if (raw === null || !/^\s*\d+\s*$/.test(raw)) return null;
    const value = Number(raw);
    return Number.isSafeInteger(value) ? value : null;
  }

  function parseContentRange(raw) {
    if (typeof raw !== 'string') return null;
    const match = /^bytes\s+(\d+)-(\d+)\/(\d+|\*)$/i.exec(raw.trim());
    if (!match) return null;
    const start = Number(match[1]);
    const end = Number(match[2]);
    if (end < start) return null;
    return { start, end, total: match[3] === '*' ? null : Number(match[3]) };
  }

  function makeEmitter(onProgress, extra = null) {
    if (typeof onProgress !== 'function') return () => {};
    return (phase, loadedBytes, totalBytes) => {
      const pct = totalBytes ? Math.min(100, (loadedBytes / totalBytes) * 100) : 0;
      const event = { phase, loadedBytes, totalBytes, pct };
      onProgress(extra ? { ...event, ...extra } : event);
    };
  }

  function httpError(url, status, what = 'PGIM image') {
    if (status === 404 || status === 410) {
      return new PgimPreloadError('NOT_FOUND', `${what} not found: ${url} (HTTP ${status})`, { status, url });
    }
    return new PgimPreloadError('HTTP_ERROR', `${what} request failed: ${url} (HTTP ${status})`, { status, url });
  }

  function isPlainObject(value) {
    return value !== null && typeof value === 'object' && !Array.isArray(value);
  }

  function validateWgslPackBytes(bytes, url) {
    const fail = (reason) => new PgimPreloadError('BAD_SHADER_PACK', `invalid WGPK shader pack ${url}: ${reason}`, { url });
    if (bytes.byteLength < WGPK_HEADER_BYTES) throw fail(`truncated header (${bytes.byteLength} bytes)`);
    for (let i = 0; i < 4; i += 1) {
      if (bytes[i] !== WGPK_MAGIC_BYTES[i]) throw fail(`bad magic [${hexBytes(bytes.subarray(0, 4))}] (expected "WGPK")`);
    }
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(4, true);
    if (version !== WGPK_VERSION) throw fail(`unsupported version ${version} (expected ${WGPK_VERSION})`);
    const entryCount = view.getUint32(8, true);
    const tableEnd = WGPK_HEADER_BYTES + entryCount * WGPK_ENTRY_BYTES;
    if (tableEnd > bytes.byteLength) throw fail(`entry table needs ${tableEnd} bytes, only ${bytes.byteLength} available`);
    return bytes;
  }

  function parseReflectionManifestJson(bytes, url) {
    const fail = (reason, cause) =>
      new PgimPreloadError('BAD_REFLECTION_MANIFEST', `invalid reflection manifest ${url}: ${reason}`, { url, cause });
    let raw;
    try {
      raw = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
    } catch (err) {
      throw fail(`malformed JSON (${err instanceof Error ? err.message : String(err)})`, err);
    }
    if (!isPlainObject(raw)) throw fail('manifest root must be a JSON object');
    if (raw.schema_version !== REFLECTION_SCHEMA_VERSION) {
      throw fail(`unsupported schema_version ${JSON.stringify(raw.schema_version)} (expected ${REFLECTION_SCHEMA_VERSION})`);
    }
    return raw;
  }

  /**
   * Streams a PGIM guest image (and optional boot shader assets) into memory before guest entry.
   */
  class PgimPreloader {
  /**
   * @param {{
   *   fetchImpl?: typeof fetch,
   *   rangeCache?: { read(path: string, offset: number, length: number): Promise<Uint8Array> } | null,
   *   chunkSize?: number,
   *   concurrency?: number,
   *   maxRetries?: number,
   *   baseRetryDelayMs?: number,
   *   sleep?: (ms: number) => Promise<void>,
   *   useSharedArrayBuffer?: boolean,
   *   targetBuffer?: ArrayBuffer | SharedArrayBuffer | null,
   * }} [options] With `rangeCache`, image chunks are read through `rangeCache.read(url, ...)`
   *   (e.g. web/workers/range_cache.js) and `url` must be a path that cache accepts.
   */
  constructor({
    fetchImpl,
    rangeCache = null,
    chunkSize = DEFAULT_CHUNK_SIZE,
    concurrency = DEFAULT_CONCURRENCY,
    maxRetries = DEFAULT_MAX_RETRIES,
    baseRetryDelayMs = DEFAULT_BASE_RETRY_DELAY_MS,
    sleep = defaultSleep,
    useSharedArrayBuffer = false,
    targetBuffer = null,
  } = {}) {
    const impl =
      fetchImpl !== undefined && fetchImpl !== null
        ? fetchImpl
        : typeof globalThis.fetch === 'function'
          ? globalThis.fetch.bind(globalThis)
          : null;
    if (typeof impl !== 'function') {
      throw new TypeError('PgimPreloader requires a fetchImpl function (no global fetch available)');
    }
    if (rangeCache !== null && (typeof rangeCache !== 'object' || typeof rangeCache.read !== 'function')) {
      throw new TypeError('rangeCache must expose read(path, offset, length)');
    }
    if (!Number.isSafeInteger(chunkSize) || chunkSize < PGIM_HEADER_BYTES) {
      throw new RangeError(`chunkSize must be an integer >= ${PGIM_HEADER_BYTES}, got: ${chunkSize}`);
    }
    if (!Number.isSafeInteger(concurrency) || concurrency < 1) {
      throw new RangeError(`concurrency must be a positive integer, got: ${concurrency}`);
    }
    if (!Number.isSafeInteger(maxRetries) || maxRetries < 0) {
      throw new RangeError(`maxRetries must be a non-negative integer, got: ${maxRetries}`);
    }
    if (!Number.isFinite(baseRetryDelayMs) || baseRetryDelayMs < 0) {
      throw new RangeError(`baseRetryDelayMs must be a non-negative number, got: ${baseRetryDelayMs}`);
    }
    if (typeof sleep !== 'function') {
      throw new TypeError('sleep must be a function');
    }
    if (
      targetBuffer !== null &&
      !(targetBuffer instanceof ArrayBuffer || isSharedArrayBuffer(targetBuffer))
    ) {
      throw new TypeError('targetBuffer must be an ArrayBuffer, SharedArrayBuffer, or null');
    }
    this.fetchImpl = impl;
    this.rangeCache = rangeCache;
    this.chunkSize = chunkSize;
    this.concurrency = concurrency;
    this.maxRetries = maxRetries;
    this.baseRetryDelayMs = baseRetryDelayMs;
    this.sleep = sleep;
    this.useSharedArrayBuffer = Boolean(useSharedArrayBuffer);
    this.targetBuffer = targetBuffer;
  }

  /**
   * Validate and stream a PGIM container into one contiguous Uint8Array.
   *
   * @param {string} url
   * @param {{ onProgress?: (e: { phase: string, loadedBytes: number, totalBytes: number | null, pct: number }) => void,
   *           signal?: AbortSignal, verifyChecksums?: boolean }} [options]
   */
  async preloadGuestImage(url, { onProgress, signal, verifyChecksums = false } = {}) {
    if (typeof url !== 'string' || url.length === 0) throw new TypeError('PGIM url must be a non-empty string');
    throwIfAborted(signal);
    const started = now();
    const emit = makeEmitter(onProgress);
    const ctx = { url, signal, emit, requests: 0 };

    emit('probe', 0, null);
    const probe = await this._probe(ctx);
    let loaded;
    if (this.rangeCache) {
      loaded = await this._loadViaRangeCache(ctx, probe);
    } else if (probe.acceptRanges === false) {
      const response = await this._request(ctx, { method: 'GET' });
      if (!response.ok) throw httpError(url, response.status);
      loaded = await this._consumeFullBody(ctx, response, probe.totalBytes);
    } else {
      loaded = await this._loadRanged(ctx, probe);
    }
    throwIfAborted(signal);

    const layout = parsePgimLayout(loaded.bytes, { verifyChecksums });
    const totalBytes = loaded.bytes.byteLength;
    const durationMs = now() - started;
    emit('complete', totalBytes, totalBytes);
    return {
      bytes: loaded.bytes,
      buffer: loaded.bytes.buffer,
      header: layout.header,
      layout,
      shared: loaded.shared,
      transport: loaded.transport,
      requests: ctx.requests,
      totalBytes,
      bytesLoaded: totalBytes,
      timingMs: durationMs,
      durationMs,
    };
  }

  /**
   * Stream and validate a PGIM v1 guest image. Returns `{ header, buffer, bytesLoaded, durationMs }`.
   *
   * @param {string} url
   * @param {{ onProgress?: Function, signal?: AbortSignal, verifyChecksums?: boolean }} [options]
   */
  async preloadPgim(url, options = {}) {
    const result = await this.preloadGuestImage(url, options);
    return {
      header: result.header,
      buffer: result.buffer,
      bytesLoaded: result.bytesLoaded,
      durationMs: result.durationMs,
      layout: result.layout,
      bytes: result.bytes,
      shared: result.shared,
      transport: result.transport,
      requests: result.requests,
    };
  }

  /**
   * Stream a `.wgsl.pack` bundle and validate the WGPK magic/version/entry table header.
   *
   * @param {string} url
   * @param {{ onProgress?: Function, signal?: AbortSignal }} [options]
   */
  async preloadShaderPack(url, { onProgress, signal } = {}) {
    if (typeof url !== 'string' || url.length === 0) throw new TypeError('shader pack url must be a non-empty string');
    throwIfAborted(signal);
    const started = now();
    const emit = makeEmitter(onProgress);
    const bytes = await this._fetchAsset(url, 'WGPK shader pack', signal, emit);
    const validated = validateWgslPackBytes(bytes, url);
    return {
      buffer: validated.buffer.byteLength === validated.byteLength && validated.byteOffset === 0
        ? validated.buffer
        : validated.slice().buffer,
      bytes: validated,
      bytesLoaded: validated.byteLength,
      durationMs: now() - started,
    };
  }

  /**
   * Fetch and validate an S6 reflection / layout manifest JSON document.
   *
   * @param {string} url
   * @param {{ onProgress?: Function, signal?: AbortSignal }} [options]
   */
  async preloadReflectionManifest(url, { onProgress, signal } = {}) {
    if (typeof url !== 'string' || url.length === 0) {
      throw new TypeError('reflection manifest url must be a non-empty string');
    }
    throwIfAborted(signal);
    const started = now();
    const emit = makeEmitter(onProgress);
    const bytes = await this._fetchAsset(url, 'reflection manifest', signal, emit);
    const manifest = parseReflectionManifestJson(bytes, url);
    return {
      manifest,
      bytes,
      bytesLoaded: bytes.byteLength,
      durationMs: now() - started,
    };
  }

  /**
   * Preload the PGIM image plus optional `.wgsl.pack` and S6 reflection manifest in parallel.
   * Omitted URLs and HTTP 404/410 for optional shader/reflection assets resolve to `null`.
   *
   * @param {{ pgimUrl: string, shaderPackUrl?: string | null, reflectionUrl?: string | null,
   *           onProgress?: (e: object) => void, signal?: AbortSignal, verifyChecksums?: boolean }} options
   *   Progress events carry an extra `asset` field: `pgim`, `shaderPack`, `reflection`, or `bundle`.
   */
  async preloadBootBundle({
    pgimUrl,
    shaderPackUrl = null,
    reflectionUrl = null,
    onProgress,
    signal,
    verifyChecksums = false,
  } = {}) {
    if (typeof pgimUrl !== 'string' || pgimUrl.length === 0) throw new TypeError('pgimUrl must be a non-empty string');
    throwIfAborted(signal);
    const started = now();
    const timingMs = { pgim: null, shaderPack: null, reflection: null, total: 0 };
    const { controller, unlink } = linkSignal(signal);
    const inner = controller.signal;
    const timed = async (key, fn) => {
      const t0 = now();
      try {
        return await fn();
      } finally {
        timingMs[key] = now() - t0;
      }
    };
    const forward = (asset) =>
      typeof onProgress === 'function' ? (event) => onProgress({ ...event, asset }) : undefined;

    const optionalAsset = async (key, url, label, validate) => {
      if (!url) return null;
      try {
        return await timed(key, async () =>
          validate(await this._fetchAsset(url, label, inner, makeEmitter(onProgress, { asset: key })), url)
        );
      } catch (err) {
        if (err instanceof PgimPreloadError && err.code === 'NOT_FOUND') {
          timingMs[key] = timingMs[key] ?? now() - started;
          if (typeof onProgress === 'function') {
            onProgress({ phase: 'optional-missing', asset: key, loadedBytes: 0, totalBytes: 0, pct: 0, status: err.status });
          }
          return null;
        }
        throw err;
      }
    };

    const tasks = [
      timed('pgim', () =>
        this.preloadPgim(pgimUrl, { onProgress: forward('pgim'), signal: inner, verifyChecksums })
      ),
      optionalAsset('shaderPack', shaderPackUrl, 'WGPK shader pack', validateWgslPackBytes),
      optionalAsset('reflection', reflectionUrl, 'reflection manifest', parseReflectionManifestJson),
    ];

    let results;
    try {
      results = await Promise.all(
        tasks.map((task) =>
          task.catch((err) => {
            controller.abort(err);
            throw err;
          })
        )
      );
    } catch (err) {
      throwIfAborted(signal);
      throw err;
    } finally {
      unlink();
    }
    const [image, shaderPackBytes, reflectionManifest] = results;
    timingMs.total = now() - started;
    if (typeof onProgress === 'function') {
      onProgress({
        phase: 'complete',
        asset: 'bundle',
        loadedBytes: image.bytesLoaded,
        totalBytes: image.bytesLoaded,
        pct: 100,
      });
    }
    return {
      pgim: image,
      pgimBytes: image.bytes,
      pgimHeader: image.header,
      pgimLayout: image.layout,
      buffer: image.buffer,
      header: image.header,
      shared: image.shared,
      shaderPackBytes,
      reflectionManifest,
      timingMs,
      durationMs: timingMs.total,
      bytesLoaded: image.bytesLoaded,
    };
  }

  async _request(ctx, init, signal = ctx.signal) {
    let attempt = 0;
    for (;;) {
      throwIfAborted(signal);
      ctx.requests += 1;
      try {
        const response = await raceAbort(
          Promise.resolve().then(() => this.fetchImpl(ctx.url, { ...init, signal })),
          signal
        );
        if (isRetryableHttpStatus(response.status) && attempt < this.maxRetries) {
          const delayMs = this.baseRetryDelayMs * Math.pow(2, attempt);
          attempt += 1;
          await raceAbort(Promise.resolve().then(() => this.sleep(delayMs)), signal);
          throwIfAborted(signal);
          continue;
        }
        return response;
      } catch (err) {
        if (signal && signal.aborted) throw abortReason(signal);
        if (err instanceof PgimPreloadError && err.code !== 'NETWORK_ERROR') throw err;
        const wrapped =
          err instanceof PgimPreloadError
            ? err
            : new PgimPreloadError(
                'NETWORK_ERROR',
                `PGIM image request failed: ${ctx.url} (${err && err.message ? err.message : err})`,
                { url: ctx.url, cause: err }
              );
        if (attempt < this.maxRetries && isRetryableNetworkError(wrapped)) {
          const delayMs = this.baseRetryDelayMs * Math.pow(2, attempt);
          attempt += 1;
          await raceAbort(Promise.resolve().then(() => this.sleep(delayMs)), signal);
          throwIfAborted(signal);
          continue;
        }
        throw wrapped;
      }
    }
  }

    async _probe(ctx) {
      let response;
      try {
        response = await this._request(ctx, { method: 'HEAD' });
      } catch (err) {
        if (ctx.signal && ctx.signal.aborted) throw err;
        return { totalBytes: null, acceptRanges: null }; // HEAD unsupported: probe with a Range GET.
      }
      if (response.status === 404 || response.status === 410) throw httpError(ctx.url, response.status);
      if (!response.ok) return { totalBytes: null, acceptRanges: null };
      const totalBytes = contentLength(response);
      if (totalBytes !== null && totalBytes < PGIM_HEADER_BYTES) {
        throw new PgimPreloadError(
          'TRUNCATED',
          `truncated PGIM image header: needed ${PGIM_HEADER_BYTES} bytes, server reports ${totalBytes}`,
          { needed: PGIM_HEADER_BYTES, available: totalBytes }
        );
      }
      const ranges = headerValue(response, 'accept-ranges');
      let acceptRanges = null;
      if (ranges !== null) {
        if (/\bbytes\b/i.test(ranges)) acceptRanges = true;
        else if (ranges.trim().toLowerCase() === 'none') acceptRanges = false;
      }
      return { totalBytes, acceptRanges };
    }

  _allocate(totalBytes) {
    if (this.targetBuffer) {
      if (this.targetBuffer.byteLength < totalBytes) {
        throw new PgimPreloadError(
          'ALLOCATION_FAILED',
          `targetBuffer is ${this.targetBuffer.byteLength} bytes but PGIM needs ${totalBytes}`
        );
      }
      return {
        bytes: new Uint8Array(this.targetBuffer, 0, totalBytes),
        shared: isSharedArrayBuffer(this.targetBuffer),
      };
    }
    if (this.useSharedArrayBuffer && typeof SharedArrayBuffer === 'function') {
      try {
        return { bytes: new Uint8Array(new SharedArrayBuffer(totalBytes)), shared: true };
      } catch {
        // Fall back to a regular ArrayBuffer below.
      }
    }
    try {
      return { bytes: new Uint8Array(new ArrayBuffer(totalBytes)), shared: false };
    } catch (err) {
      throw new PgimPreloadError('ALLOCATION_FAILED', `cannot allocate ${totalBytes} bytes for the PGIM image`, { cause: err });
    }
  }

    async _loadRanged(ctx, probe) {
      const firstLen = probe.totalBytes !== null ? Math.min(this.chunkSize, probe.totalBytes) : this.chunkSize;
      const response = await this._request(ctx, { method: 'GET', headers: { Range: `bytes=0-${firstLen - 1}` } });
      if (response.status === 200) {
        // Server ignored the Range header: consume the full body as a single GET.
        return this._consumeFullBody(ctx, response, probe.totalBytes);
      }
      if (response.status === 416) {
        throw new PgimPreloadError('TRUNCATED', `truncated PGIM image header: ${ctx.url} is empty (HTTP 416)`, {
          needed: PGIM_HEADER_BYTES,
          available: 0,
        });
      }
      if (response.status !== 206) throw httpError(ctx.url, response.status);

      const range = parseContentRange(headerValue(response, 'content-range'));
      const first = new Uint8Array(await raceAbort(response.arrayBuffer(), ctx.signal));
      throwIfAborted(ctx.signal);
      if (range && (range.start !== 0 || range.end - range.start + 1 !== first.byteLength)) {
        throw new PgimPreloadError('RANGE_MISMATCH', `PGIM first chunk Content-Range does not match the ${first.byteLength} received bytes`);
      }
      if (range && range.total !== null && probe.totalBytes !== null && range.total !== probe.totalBytes) {
        throw new PgimPreloadError(
          'SIZE_MISMATCH',
          `PGIM size changed between HEAD (${probe.totalBytes}) and Range response (${range.total})`
        );
      }
      const transportTotal = range && range.total !== null ? range.total : probe.totalBytes;
      const header = parsePgimHeader(first, { totalBytes: transportTotal });
      const total = header.fileSize;
      if (first.byteLength !== Math.min(firstLen, total)) {
        throw new PgimPreloadError('RANGE_MISMATCH', `PGIM first chunk returned ${first.byteLength} bytes, expected ${Math.min(firstLen, total)}`);
      }
      const dest = this._allocate(total);
      dest.bytes.set(first, 0);
      ctx.emit('header', first.byteLength, total);

      await this._parallelChunks(ctx, dest.bytes, first.byteLength, async (start, end, signal) => {
        const res = await this._request(ctx, { method: 'GET', headers: { Range: `bytes=${start}-${end - 1}` } }, signal);
        if (res.status !== 206) {
          throw res.status === 200
            ? new PgimPreloadError('RANGE_NOT_HONORED', `server ignored Range bytes=${start}-${end - 1} for ${ctx.url}`)
            : httpError(ctx.url, res.status);
        }
        const cr = parseContentRange(headerValue(res, 'content-range'));
        if (cr && (cr.start !== start || cr.end !== end - 1 || (cr.total !== null && cr.total !== total))) {
          throw new PgimPreloadError('RANGE_MISMATCH', `unexpected Content-Range for bytes=${start}-${end - 1}`);
        }
        return new Uint8Array(await raceAbort(res.arrayBuffer(), signal));
      });
      return { ...dest, transport: 'range' };
    }

    async _loadViaRangeCache(ctx, probe) {
      const cache = this.rangeCache;
      const read = async (offset, length, signal) => {
        throwIfAborted(signal);
        ctx.requests += 1;
        return toBytes(await raceAbort(Promise.resolve().then(() => cache.read(ctx.url, offset, length)), signal), 'rangeCache chunk');
      };
      const firstLen = probe.totalBytes !== null ? Math.min(this.chunkSize, probe.totalBytes) : this.chunkSize;
      const first = await read(0, firstLen, ctx.signal);
      const header = parsePgimHeader(first, { totalBytes: probe.totalBytes });
      const total = header.fileSize;
      if (first.byteLength !== Math.min(firstLen, total)) {
        throw new PgimPreloadError('RANGE_MISMATCH', `rangeCache returned ${first.byteLength} bytes, expected ${Math.min(firstLen, total)}`);
      }
      const dest = this._allocate(total);
      dest.bytes.set(first, 0);
      ctx.emit('header', first.byteLength, total);
      await this._parallelChunks(ctx, dest.bytes, first.byteLength, (start, end, signal) => read(start, end - start, signal));
      return { ...dest, transport: 'range-cache' };
    }

    async _parallelChunks(ctx, dest, startOffset, fetchChunk) {
      const total = dest.byteLength;
      const chunks = [];
      for (let start = startOffset; start < total; start += this.chunkSize) {
        chunks.push([start, Math.min(total, start + this.chunkSize)]);
      }
      if (chunks.length === 0) return;
      const { controller, unlink } = linkSignal(ctx.signal);
      let next = 0;
      let loaded = startOffset;
      let failure = null;
      const worker = async () => {
        while (failure === null && next < chunks.length) {
          const [start, end] = chunks[next];
          next += 1;
          try {
            const data = await fetchChunk(start, end, controller.signal);
            if (failure !== null) return;
            if (data.byteLength !== end - start) {
              throw new PgimPreloadError(
                'RANGE_MISMATCH',
                `PGIM chunk bytes=${start}-${end - 1} returned ${data.byteLength} bytes, expected ${end - start}`
              );
            }
            dest.set(data, start);
            loaded += data.byteLength;
            ctx.emit('download', loaded, total);
          } catch (err) {
            if (failure === null) {
              failure = err;
              controller.abort(err);
            }
            return;
          }
        }
      };
      try {
        await Promise.all(Array.from({ length: Math.min(this.concurrency, chunks.length) }, worker));
      } finally {
        unlink();
      }
      throwIfAborted(ctx.signal);
      if (failure !== null) throw failure;
    }

    async _consumeFullBody(ctx, response, knownTotal) {
      const declared = contentLength(response);
      if (declared !== null && knownTotal !== null && declared !== knownTotal) {
        throw new PgimPreloadError('SIZE_MISMATCH', `PGIM size changed between HEAD (${knownTotal}) and GET (${declared})`);
      }
      const expected = declared !== null ? declared : knownTotal;
      const body = response.body;
      if (!body || typeof body.getReader !== 'function') {
        const whole = new Uint8Array(await raceAbort(response.arrayBuffer(), ctx.signal));
        throwIfAborted(ctx.signal);
        const header = parsePgimHeader(whole, { totalBytes: expected });
        if (whole.byteLength !== header.fileSize) {
          throw new PgimPreloadError('TRUNCATED', `truncated PGIM image: needed ${header.fileSize} bytes, only ${whole.byteLength} available`, {
            needed: header.fileSize,
            available: whole.byteLength,
          });
        }
        const dest = this._allocate(header.fileSize);
        dest.bytes.set(whole, 0);
        ctx.emit('download', whole.byteLength, header.fileSize);
        return { ...dest, transport: 'full' };
      }

      const reader = body.getReader();
      let pending = [];
      let pendingBytes = 0;
      let dest = null;
      let total = 0;
      let loaded = 0;
      const write = (chunk) => {
        if (loaded + chunk.byteLength > total) {
          throw new PgimPreloadError('SIZE_MISMATCH', `PGIM body is larger than its header file_size (${total} bytes)`);
        }
        dest.bytes.set(chunk, loaded);
        loaded += chunk.byteLength;
      };
      try {
        for (;;) {
          const { done, value } = await raceAbort(reader.read(), ctx.signal);
          if (done) break;
          const chunk = toBytes(value, 'response chunk');
          if (dest === null) {
            pending.push(chunk);
            pendingBytes += chunk.byteLength;
            if (pendingBytes < PGIM_HEADER_BYTES) continue;
            const head = new Uint8Array(PGIM_HEADER_BYTES);
            let cursor = 0;
            for (const part of pending) {
              const take = Math.min(part.byteLength, PGIM_HEADER_BYTES - cursor);
              head.set(part.subarray(0, take), cursor);
              cursor += take;
              if (cursor === PGIM_HEADER_BYTES) break;
            }
            const header = parsePgimHeader(head, { totalBytes: expected });
            total = header.fileSize;
            dest = this._allocate(total);
            for (const part of pending) write(part);
            pending = null;
            ctx.emit('header', loaded, total);
          } else {
            write(chunk);
            ctx.emit('download', loaded, total);
          }
        }
      } catch (err) {
        reader.cancel(err).catch(() => {});
        throwIfAborted(ctx.signal);
        throw err;
      }
      throwIfAborted(ctx.signal);
      if (dest === null) {
        throw new PgimPreloadError(
          'TRUNCATED',
          `truncated PGIM image header: needed ${PGIM_HEADER_BYTES} bytes, only ${pendingBytes} available`,
          { needed: PGIM_HEADER_BYTES, available: pendingBytes }
        );
      }
      if (loaded !== total) {
        throw new PgimPreloadError('TRUNCATED', `truncated PGIM image: needed ${total} bytes, only ${loaded} available`, {
          needed: total,
          available: loaded,
        });
      }
      return { ...dest, transport: 'full' };
    }

    async _fetchAsset(url, what, signal, emit) {
      const ctx = { url, signal, emit, requests: 0 };
      const response = await this._request(ctx, { method: 'GET' });
      if (!response.ok) throw httpError(url, response.status, what);
      const declared = contentLength(response);
      emit('download', 0, declared);
      const bytes = new Uint8Array(await raceAbort(response.arrayBuffer(), signal));
      throwIfAborted(signal);
      emit('complete', bytes.byteLength, bytes.byteLength);
      return bytes;
    }
  }


  const exportedApi = Object.freeze({
    PGIM_MAGIC,
    PGIM_MAGIC_BYTES,
    PGIM_VERSION,
    PGIM_HEADER_BYTES,
    PGIM_SECTION_ENTRY_BYTES,
    PGIM_PAGE_SIZE,
    PGIM_FLAGS,
    PGIM_SECTION_FLAGS,
    PGIM_PROT,
    PgimPreloadError,
    pgimFnv1a64,
    parsePgimHeader,
    parsePgimLayout,
    PgimPreloader,
    default: PgimPreloader,
  });

  if (typeof globalThis !== 'undefined') {
    globalThis.PlayGta6PgimPreloader = exportedApi;
  } else if (root) {
    root.PlayGta6PgimPreloader = exportedApi;
  }

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = exportedApi;
  }
})(typeof globalThis !== 'undefined' ? globalThis : typeof self !== 'undefined' ? self : this);
