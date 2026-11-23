'use strict';

// Streaming browser-side RPF7/8/9 TOC & Virtual File System reader.
// Fetches ONLY header + TOC + names table + requested sub-file byte ranges via an
// async byte-range reader or RangeCache (HTTP 206 Partial Content), never downloading
// entire multi-GB archives.
//
// Compatible with classic Web Workers (importScripts) and Node CommonJS/ESM tests.
// Zero external dependencies and zero embedded game keys.

const RPF9_MAGIC = 'RPF9';
const RPF8_MAGIC = 'RPF8';
const RPF7_MAGIC = 'RPF7';

// Header layout matching tools/src/rpf9.rs (24 bytes, little-endian):
//   0x00..0x04: magic ("RPF9" / "RPF8" / "RPF7")
//   0x04..0x08: entryCount (u32 LE)
//   0x08..0x0C: namesSize (u32 LE)
//   0x0C..0x10: encryptionTag (u32 LE)
//   0x10..0x14: metadataSize (u32 LE)
//   0x14..0x18: flags (u32 LE)
const RPF9_HEADER_SIZE = 24;

// Compact TOC entry stride matching tools/src/rpf9.rs (16 bytes, little-endian):
//
// Common word 0 (bytes 0..4):
//   bits  0..23: nameOffset (u24, offset into NUL-terminated UTF-8 names table)
//   bits 24..31: flags (u8):
//     0x80 (RPF_ENTRY_FLAG_DIRECTORY):     1 = Directory entry, 0 = File entry
//     0x40 (RPF_ENTRY_FLAG_COMPRESSED):    1 = File payload is compressed on disk
//     0x20 (RPF_ENTRY_FLAG_RESOURCE):      1 = RAGE paged resource file
//     0x10 (RPF_ENTRY_FLAG_ENCRYPTED):     1 = Per-entry encrypted payload
//     0x08 (RPF_ENTRY_FLAG_OFFSET_SHIFT9): 1 = File dataOffset word is in 512-byte sectors
//                                              (supports archives up to 2 TiB)
//
// Directory entry (flags & RPF_ENTRY_FLAG_DIRECTORY):
//   bytes  0..4:  (flags << 24) | (nameOffset & 0x00FFFFFF)
//   bytes  4..8:  firstChildIndex (u32 LE)
//   bytes  8..12: childCount (u32 LE)
//   bytes 12..16: reserved (u32 LE, 0)
//
// File entry (!(flags & RPF_ENTRY_FLAG_DIRECTORY)):
//   bytes  0..4:  (flags << 24) | (nameOffset & 0x00FFFFFF)
//   bytes  4..8:  dataOffset (u32 LE; byte offset, or 512-byte sector index if OFFSET_SHIFT9 set)
//   bytes  8..12: uncompressedSize (u32 LE)
//   bytes 12..16: compressedSize (u32 LE; 0 when stored uncompressed)
const RPF9_ENTRY_STRIDE = 16;
const RPF_SECTOR_SIZE = 512;

// Known RAGE/RPF encryption tags matching tools/src/rpf9.rs.
const RPF_ENCRYPTION_NONE = 0x00000000;
const RPF_ENCRYPTION_OPEN = 0x4e45504f; // "OPEN" LE
const RPF_ENCRYPTION_AES = 0x0ffffff9;
const RPF_ENCRYPTION_NG = 0x0fefffff;

// TOC entry bit flags (stored in bits 24..31 of word 0).
const RPF_ENTRY_FLAG_DIRECTORY = 0x80;
const RPF_ENTRY_FLAG_COMPRESSED = 0x40;
const RPF_ENTRY_FLAG_RESOURCE = 0x20;
const RPF_ENTRY_FLAG_ENCRYPTED = 0x10;
const RPF_ENTRY_FLAG_OFFSET_SHIFT9 = 0x08;

const MAX_U24 = 0x00ffffff;
const MAX_U32 = 0xffffffff;
const MAX_U64_BIGINT = 0xffffffffffffffffn;
const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

const utf8Decoder = new TextDecoder('utf-8');
const utf8Encoder = new TextEncoder();

class RpfFormatError extends Error {
  constructor(code, message, details = undefined) {
    super(message);
    this.name = 'RpfFormatError';
    this.code = code;
    if (details !== undefined) {
      this.details = details;
    }
  }
}

class EncryptedArchiveError extends RpfFormatError {
  constructor(encryptionTag, encryptionLabel, message = undefined) {
    const tag = encryptionTag >>> 0;
    const label = encryptionLabel || classifyEncryption(tag).label;
    const hex = `0x${tag.toString(16).toUpperCase().padStart(8, '0')}`;
    super(
      'ENCRYPTED_ARCHIVE',
      message ||
        `RPF archive is encrypted (tag ${hex} / ${label}) and no external decryptBlock hook was supplied`,
      { encryptionTag: tag, encryptionLabel: label }
    );
    this.name = 'EncryptedArchiveError';
    this.encryptionTag = tag;
    this.encryptionLabel = label;
  }
}

function toUint8Array(input, label = 'bytes') {
  if (input instanceof Uint8Array) {
    return input;
  }
  if (input instanceof ArrayBuffer) {
    return new Uint8Array(input);
  }
  if (ArrayBuffer.isView(input)) {
    return new Uint8Array(input.buffer, input.byteOffset, input.byteLength);
  }
  throw new RpfFormatError('INVALID_BUFFER', `${label} must be a Uint8Array or ArrayBuffer`);
}

function classifyEncryption(encryptionTag) {
  const tag = encryptionTag >>> 0;
  switch (tag) {
    case RPF_ENCRYPTION_NONE:
      return { tag, label: 'none', isEncrypted: false };
    case RPF_ENCRYPTION_OPEN:
      return { tag, label: 'open', isEncrypted: false };
    case RPF_ENCRYPTION_AES:
      return { tag, label: 'aes', isEncrypted: true };
    case RPF_ENCRYPTION_NG:
      return { tag, label: 'ng', isEncrypted: true };
    default:
      return { tag, label: 'custom', isEncrypted: true };
  }
}

function detectMagicVersion(u8) {
  const b0 = u8[0];
  const b1 = u8[1];
  const b2 = u8[2];
  const b3 = u8[3];

  // 'R'=0x52, 'P'=0x50, 'F'=0x46
  if (b0 === 0x52 && b1 === 0x50 && b2 === 0x46) {
    if (b3 === 0x39) return { magic: RPF9_MAGIC, version: 9 };
    if (b3 === 0x38) return { magic: RPF8_MAGIC, version: 8 };
    if (b3 === 0x37) return { magic: RPF7_MAGIC, version: 7 };
  }

  const hex = [b0, b1, b2, b3]
    .map((b) => b.toString(16).toUpperCase().padStart(2, '0'))
    .join(' ');
  throw new RpfFormatError(
    'INVALID_MAGIC',
    `Invalid RPF magic bytes: ${hex} (expected RPF7, RPF8, or RPF9)`,
    { magicBytes: [b0, b1, b2, b3] }
  );
}

function parseTotalSizeBigInt(totalSize) {
  if (totalSize === undefined || totalSize === null) {
    return null;
  }
  if (typeof totalSize === 'bigint') {
    if (totalSize < 0n || totalSize > MAX_SAFE_BIGINT) {
      throw new RpfFormatError('SIZE_OVERFLOW', `Invalid RPF totalSize: ${totalSize}`);
    }
    return totalSize;
  }
  if (typeof totalSize !== 'number' || !Number.isSafeInteger(totalSize) || totalSize < 0) {
    throw new RpfFormatError('SIZE_OVERFLOW', `Invalid RPF totalSize: ${totalSize}`);
  }
  return BigInt(totalSize);
}

/**
 * Parses and validates an RPF7/RPF8/RPF9 24-byte archive header.
 *
 * @param {Uint8Array|ArrayBuffer|ArrayBufferView} bytes - Prefix buffer containing at least 24 bytes.
 * @param {number|bigint|null|undefined} [totalSize] - Optional total archive byte size for bounds checks.
 * @returns {object} Parsed header metadata including computed tocBytes and namesOffset.
 */
function parseRpfHeader(bytes, totalSize = null) {
  const u8 = toUint8Array(bytes, 'header bytes');

  if (u8.byteLength < 4) {
    throw new RpfFormatError(
      'TRUNCATED_HEADER',
      `Truncated RPF header: got ${u8.byteLength} bytes, need at least ${RPF9_HEADER_SIZE} bytes`,
      { actual: u8.byteLength, expected: RPF9_HEADER_SIZE }
    );
  }

  const { magic, version } = detectMagicVersion(u8);

  if (u8.byteLength < RPF9_HEADER_SIZE) {
    throw new RpfFormatError(
      'TRUNCATED_HEADER',
      `Truncated ${magic} header: got ${u8.byteLength} bytes, need at least ${RPF9_HEADER_SIZE} bytes`,
      { actual: u8.byteLength, expected: RPF9_HEADER_SIZE }
    );
  }

  const totalSizeBig = parseTotalSizeBigInt(totalSize);
  const headerSizeBig = BigInt(RPF9_HEADER_SIZE);

  if (totalSizeBig !== null && totalSizeBig < headerSizeBig) {
    throw new RpfFormatError(
      'TRUNCATED_ARCHIVE',
      `Truncated ${magic} archive: header requires at least ${RPF9_HEADER_SIZE} bytes, but file is ${totalSizeBig} bytes`,
      { requiredBytes: RPF9_HEADER_SIZE, actualBytes: Number(totalSizeBig) }
    );
  }

  const view = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const entryCount = view.getUint32(4, true);
  const namesSize = view.getUint32(8, true);
  const encryptionTag = view.getUint32(12, true);
  const metadataSize = view.getUint32(16, true);
  const flags = view.getUint32(20, true);

  const enc = classifyEncryption(encryptionTag);

  const tocOffsetBig = headerSizeBig;
  const tocBytesBig = BigInt(entryCount) * BigInt(RPF9_ENTRY_STRIDE);
  const namesOffsetBig = tocOffsetBig + tocBytesBig;
  const metadataOffsetBig = namesOffsetBig + BigInt(namesSize);
  const minIndexBytesBig = metadataOffsetBig + BigInt(metadataSize);

  if (minIndexBytesBig > MAX_U64_BIGINT || minIndexBytesBig > MAX_SAFE_BIGINT) {
    throw new RpfFormatError(
      'SIZE_OVERFLOW',
      `${magic} header metadata sizes overflow safe integer bounds`
    );
  }

  if (totalSizeBig !== null && minIndexBytesBig > totalSizeBig) {
    throw new RpfFormatError(
      'TRUNCATED_ARCHIVE',
      `Truncated ${magic} archive: header requires at least ${minIndexBytesBig} bytes, but file is ${totalSizeBig} bytes`,
      { requiredBytes: Number(minIndexBytesBig), actualBytes: Number(totalSizeBig) }
    );
  }

  return {
    magic,
    version,
    headerSize: RPF9_HEADER_SIZE,
    entryCount,
    namesSize,
    encryptionTag: enc.tag,
    encryptionLabel: enc.label,
    isEncrypted: enc.isEncrypted,
    metadataSize,
    flags,
    totalSize: totalSizeBig === null ? null : Number(totalSizeBig),
    tocOffset: Number(tocOffsetBig),
    tocBytes: Number(tocBytesBig),
    namesOffset: Number(namesOffsetBig),
    metadataOffset: Number(metadataOffsetBig),
    minIndexBytes: Number(minIndexBytesBig),
  };
}

function readCString(namesBytes, nameOffset) {
  if (namesBytes.byteLength === 0) {
    if (nameOffset === 0) return '';
    throw new RpfFormatError(
      'INVALID_TOC',
      `TOC nameOffset ${nameOffset} out of bounds for empty names table`
    );
  }
  if (nameOffset >= namesBytes.byteLength) {
    throw new RpfFormatError(
      'INVALID_TOC',
      `TOC nameOffset ${nameOffset} exceeds names table size ${namesBytes.byteLength}`
    );
  }
  let end = nameOffset;
  while (end < namesBytes.byteLength && namesBytes[end] !== 0) {
    end += 1;
  }
  return utf8Decoder.decode(namesBytes.subarray(nameOffset, end));
}

function parseTocEntry(tocBytes, byteOffset, namesBytes, entryIndex = 0) {
  const u8 = toUint8Array(tocBytes, 'tocBytes');
  if (byteOffset < 0 || byteOffset + RPF9_ENTRY_STRIDE > u8.byteLength) {
    throw new RpfFormatError(
      'INVALID_TOC',
      `TOC entry ${entryIndex} at byte offset ${byteOffset} exceeds TOC buffer length ${u8.byteLength}`
    );
  }

  const view = new DataView(u8.buffer, u8.byteOffset + byteOffset, RPF9_ENTRY_STRIDE);
  const w0 = view.getUint32(0, true);
  const w1 = view.getUint32(4, true);
  const w2 = view.getUint32(8, true);
  const w3 = view.getUint32(12, true);

  const nameOffset = w0 & MAX_U24;
  const flags = (w0 >>> 24) & 0xff;
  const name = namesBytes ? readCString(namesBytes, nameOffset) : '';

  if ((flags & RPF_ENTRY_FLAG_DIRECTORY) !== 0) {
    return {
      kind: 'directory',
      index: entryIndex,
      name,
      nameOffset,
      firstChildIndex: w1,
      childCount: w2,
      reserved: w3,
      flags,
    };
  }

  const isCompressed = (flags & RPF_ENTRY_FLAG_COMPRESSED) !== 0;
  const isResource = (flags & RPF_ENTRY_FLAG_RESOURCE) !== 0;
  const isEncrypted = (flags & RPF_ENTRY_FLAG_ENCRYPTED) !== 0;
  const isSectorOffset = (flags & RPF_ENTRY_FLAG_OFFSET_SHIFT9) !== 0;

  const dataOffset = isSectorOffset ? w1 * RPF_SECTOR_SIZE : w1;
  const uncompressedSize = w2;
  const compressedSize = w3;
  const storedSize = isCompressed ? compressedSize : uncompressedSize;

  return {
    kind: 'file',
    index: entryIndex,
    name,
    nameOffset,
    dataOffset,
    uncompressedSize,
    compressedSize,
    storedSize,
    flags,
    isCompressed,
    isResource,
    isEncrypted,
  };
}

/**
 * Normalizes a virtual RPF path:
 * - Converts backslashes to forward slashes
 * - Strips leading/trailing slashes and collapses duplicate slashes
 * - Drops '.' segments, rejects '..' traversal and NUL bytes
 * - Returns both display path (preserving segment case) and lowercase lookup key
 *
 * @param {string} virtualPath
 * @returns {{ path: string, key: string, segments: string[] }}
 */
function normalizeVirtualPath(virtualPath) {
  if (virtualPath === undefined || virtualPath === null || virtualPath === '') {
    return { path: '', key: '', segments: [] };
  }
  if (typeof virtualPath !== 'string') {
    throw new RpfFormatError('INVALID_PATH', 'Virtual path must be a string');
  }
  if (virtualPath.includes('\0')) {
    throw new RpfFormatError('INVALID_PATH', `Virtual path contains NUL byte: ${virtualPath}`);
  }

  const rawSegments = virtualPath.replace(/\\/g, '/').split('/');
  const segments = [];
  for (const seg of rawSegments) {
    if (!seg || seg === '.') {
      continue;
    }
    if (seg === '..') {
      throw new RpfFormatError('INVALID_PATH', `Path traversal is not allowed: ${virtualPath}`);
    }
    segments.push(seg);
  }

  const normalized = segments.join('/');
  return {
    path: normalized,
    key: normalized.toLowerCase(),
    segments,
  };
}

function cloneNodeStat(node, pathOverride = undefined, extra = undefined) {
  const entry = node.entry;
  const outPath = pathOverride !== undefined ? pathOverride : node.path;
  if (entry.kind === 'directory') {
    return Object.assign(
      {
        path: outPath,
        name: node.name,
        kind: 'directory',
        isDirectory: true,
        isFile: false,
        entryIndex: entry.index,
        firstChildIndex: entry.firstChildIndex,
        childCount: entry.childCount,
        flags: entry.flags,
      },
      extra || null
    );
  }
  return Object.assign(
    {
      path: outPath,
      name: node.name,
      kind: 'file',
      isDirectory: false,
      isFile: true,
      entryIndex: entry.index,
      dataOffset: entry.dataOffset,
      size: entry.uncompressedSize,
      uncompressedSize: entry.uncompressedSize,
      compressedSize: entry.compressedSize,
      storedSize: entry.storedSize,
      flags: entry.flags,
      isCompressed: entry.isCompressed,
      isResource: entry.isResource,
      isEncrypted: entry.isEncrypted,
    },
    extra || null
  );
}

function wrapRangeReader(rangeReader) {
  if (typeof rangeReader === 'function') {
    return async (offset, length) => toUint8Array(await rangeReader(offset, length), 'rangeReader result');
  }
  if (rangeReader && typeof rangeReader.read === 'function') {
    return async (offset, length) =>
      toUint8Array(await rangeReader.read(offset, length), 'rangeReader.read result');
  }
  throw new RpfFormatError(
    'INVALID_READER',
    'rangeReader must be a function (offset, length) => Promise<Uint8Array> or an object with .read(offset, length)'
  );
}

class RpfArchiveVfs {
  constructor(header, readRange, entries, options = {}) {
    this.header = header;
    this.archivePath = typeof options.archivePath === 'string' ? options.archivePath : null;
    this._readRange = readRange;
    this._decryptBlock = typeof options.decryptBlock === 'function' ? options.decryptBlock : null;
    this._decompressEntry =
      typeof options.decompressEntry === 'function' ? options.decompressEntry : null;
    this._entries = entries;
    this._nodesByKey = new Map();
    this._childrenByDirKey = new Map();

    this._buildIndex();
  }

  _buildIndex() {
    const entries = this._entries;
    if (entries.length === 0) {
      const syntheticRoot = {
        path: '',
        key: '',
        name: '',
        entry: {
          kind: 'directory',
          index: 0,
          name: '',
          nameOffset: 0,
          firstChildIndex: 0,
          childCount: 0,
          reserved: 0,
          flags: RPF_ENTRY_FLAG_DIRECTORY,
        },
      };
      this._nodesByKey.set('', syntheticRoot);
      this._childrenByDirKey.set('', []);
      return;
    }

    const rootEntry = entries[0];
    if (rootEntry.kind !== 'directory') {
      throw new RpfFormatError('INVALID_TOC', 'TOC entry 0 must be the root directory');
    }

    const visited = new Uint8Array(entries.length);
    const rootNode = {
      path: '',
      key: '',
      name: '',
      entry: rootEntry,
    };
    this._nodesByKey.set('', rootNode);
    visited[0] = 1;

    const stack = [rootNode];
    while (stack.length > 0) {
      const dirNode = stack.pop();
      const dirEntry = dirNode.entry;
      const children = [];
      this._childrenByDirKey.set(dirNode.key, children);

      if (dirEntry.childCount === 0) {
        continue;
      }

      const start = dirEntry.firstChildIndex;
      const end = start + dirEntry.childCount;
      if (start < 0 || end > entries.length) {
        throw new RpfFormatError(
          'INVALID_TOC',
          `Directory entry ${dirEntry.index} child range [${start}..${end}) exceeds entryCount ${entries.length}`
        );
      }

      for (let idx = start; idx < end; idx += 1) {
        if (visited[idx]) {
          throw new RpfFormatError(
            'INVALID_TOC',
            `Cycle or duplicate parent reference detected at TOC entry ${idx}`
          );
        }
        visited[idx] = 1;

        const child = entries[idx];
        const cleanName = child.name.replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');
        if (!cleanName || cleanName.includes('/') || cleanName === '.' || cleanName === '..') {
          throw new RpfFormatError(
            'INVALID_TOC',
            `Invalid child entry name "${child.name}" at TOC entry ${idx}`
          );
        }

        if (child.kind === 'file' && this.header.totalSize !== null) {
          const endByte = child.dataOffset + child.storedSize;
          if (endByte > this.header.totalSize) {
            throw new RpfFormatError(
              'TRUNCATED_ARCHIVE',
              `File entry "${cleanName}" range [${child.dataOffset}..${endByte}) exceeds archive size ${this.header.totalSize}`
            );
          }
        }

        const childPath = dirNode.path ? `${dirNode.path}/${cleanName}` : cleanName;
        const childKey = childPath.toLowerCase();
        const childNode = {
          path: childPath,
          key: childKey,
          name: cleanName,
          entry: child,
        };

        this._nodesByKey.set(childKey, childNode);
        children.push(childNode);

        if (child.kind === 'directory') {
          stack.push(childNode);
        }
      }
    }
  }

  /**
   * Opens an RPF archive via any async byte-range reader, fetching ONLY the header,
   * TOC entry table, and names table.
   *
   * @param {Function|object} rangeReader - `(offset, length) => Promise<Uint8Array>` or `{ read }`.
   * @param {number|bigint|null} [totalSize] - Optional total byte size of the archive.
   * @param {object} [options] - Optional `{ decryptBlock, decompressEntry, archivePath }`.
   * @returns {Promise<RpfArchiveVfs>}
   */
  static async open(rangeReader, totalSize = null, options = {}) {
    const readRange = wrapRangeReader(rangeReader);
    const decryptBlock = typeof options.decryptBlock === 'function' ? options.decryptBlock : null;

    const headerBytes = await readRange(0, RPF9_HEADER_SIZE);
    const header = parseRpfHeader(headerBytes, totalSize);

    if (header.isEncrypted && !decryptBlock) {
      throw new EncryptedArchiveError(header.encryptionTag, header.encryptionLabel);
    }

    const indexBytesToRead = header.tocBytes + header.namesSize;
    let tocBytes = new Uint8Array(0);
    let namesBytes = new Uint8Array(0);

    if (indexBytesToRead > 0) {
      const indexBuf = await readRange(header.tocOffset, indexBytesToRead);
      if (indexBuf.byteLength < indexBytesToRead) {
        throw new RpfFormatError(
          'TRUNCATED_ARCHIVE',
          `Truncated ${header.magic} TOC/names block: expected ${indexBytesToRead} bytes at offset ${header.tocOffset}, got ${indexBuf.byteLength} bytes`,
          {
            offset: header.tocOffset,
            expectedBytes: indexBytesToRead,
            actualBytes: indexBuf.byteLength,
          }
        );
      }

      tocBytes = indexBuf.subarray(0, header.tocBytes);
      namesBytes = indexBuf.subarray(header.tocBytes, header.tocBytes + header.namesSize);

      if (header.isEncrypted) {
        if (tocBytes.byteLength > 0) {
          tocBytes = toUint8Array(
            await decryptBlock(tocBytes.slice(), {
              region: 'toc',
              encryptionTag: header.encryptionTag,
              encryptionLabel: header.encryptionLabel,
              header,
            }),
            'decryptBlock(toc) result'
          );
        }
        if (namesBytes.byteLength > 0) {
          namesBytes = toUint8Array(
            await decryptBlock(namesBytes.slice(), {
              region: 'names',
              encryptionTag: header.encryptionTag,
              encryptionLabel: header.encryptionLabel,
              header,
            }),
            'decryptBlock(names) result'
          );
        }
      }
    }

    const entries = new Array(header.entryCount);
    for (let i = 0; i < header.entryCount; i += 1) {
      entries[i] = parseTocEntry(tocBytes, i * RPF9_ENTRY_STRIDE, namesBytes, i);
    }

    return new RpfArchiveVfs(header, readRange, entries, options);
  }

  /**
   * Opens an RPF archive backed by a PlayGta6RangeCache instance.
   *
   * @param {object} rangeCache - RangeCache instance exposing `read(relPath, offset, length)`.
   * @param {string} archivePath - Relative archive path (e.g. `'x64a.rpf'` or `'update/update.rpf'`).
   * @param {number|bigint|null} [totalSize] - Optional total archive byte size.
   * @param {object} [options] - Optional `{ decryptBlock, decompressEntry }`.
   * @returns {Promise<RpfArchiveVfs>}
   */
  static async openFromRangeCache(rangeCache, archivePath, totalSize = null, options = {}) {
    if (!rangeCache || typeof rangeCache.read !== 'function') {
      throw new RpfFormatError(
        'INVALID_READER',
        'openFromRangeCache requires a RangeCache instance with a .read(path, offset, length) method'
      );
    }
    if (typeof archivePath !== 'string' || archivePath.length === 0) {
      throw new RpfFormatError('INVALID_PATH', 'archivePath must be a non-empty string');
    }

    const reader = (offset, length) => rangeCache.read(archivePath, offset, length);
    return RpfArchiveVfs.open(reader, totalSize, Object.assign({}, options, { archivePath }));
  }

  exists(virtualPath) {
    let norm;
    try {
      norm = normalizeVirtualPath(virtualPath);
    } catch {
      return false;
    }
    return this._nodesByKey.has(norm.key);
  }

  stat(virtualPath) {
    const norm = normalizeVirtualPath(virtualPath);
    const node = this._nodesByKey.get(norm.key);
    if (!node) {
      return null;
    }
    return cloneNodeStat(node);
  }

  listDir(dirPath = '') {
    const norm = normalizeVirtualPath(dirPath);
    const node = this._nodesByKey.get(norm.key);
    if (!node) {
      throw new RpfFormatError('PATH_NOT_FOUND', `Directory not found in RPF archive: ${dirPath}`);
    }
    if (node.entry.kind !== 'directory') {
      throw new RpfFormatError('NOT_A_DIRECTORY', `Path is not a directory: ${dirPath}`);
    }
    const children = this._childrenByDirKey.get(node.key) || [];
    return children.map((child) => cloneNodeStat(child));
  }

  /**
   * Streams a byte range from a virtual file inside the archive.
   * Requests ONLY `[entry.dataOffset + offset, ...]` from the underlying range reader,
   * cleanly clamped to EOF.
   *
   * @param {string} virtualPath
   * @param {number} [offset=0]
   * @param {number|null|undefined} [length] - Omit to read through EOF.
   * @param {object} [options] - Optional `{ raw: boolean }`.
   * @returns {Promise<Uint8Array>}
   */
  async readFileRange(virtualPath, offset = 0, length = undefined, options = {}) {
    const norm = normalizeVirtualPath(virtualPath);
    const node = this._nodesByKey.get(norm.key);
    if (!node) {
      throw new RpfFormatError('PATH_NOT_FOUND', `File not found in RPF archive: ${virtualPath}`);
    }
    if (node.entry.kind !== 'file') {
      throw new RpfFormatError('NOT_A_FILE', `Path is a directory, not a file: ${virtualPath}`);
    }

    if (!Number.isSafeInteger(offset) || offset < 0) {
      throw new RangeError(`Invalid readFileRange offset: ${offset}`);
    }
    if (length !== undefined && length !== null && (!Number.isSafeInteger(length) || length < 0)) {
      throw new RangeError(`Invalid readFileRange length: ${length}`);
    }

    const entry = node.entry;
    const readRaw = Boolean(options && options.raw);

    if (entry.isCompressed && !readRaw) {
      if (!this._decompressEntry) {
        throw new RpfFormatError(
          'COMPRESSED_ENTRY_UNSUPPORTED',
          `File "${node.path}" is compressed and no decompressEntry hook was provided`
        );
      }
      if (offset >= entry.uncompressedSize || length === 0) {
        return new Uint8Array(0);
      }
      let compressedBytes = await this._readRange(entry.dataOffset, entry.storedSize);
      if (this.header.isEncrypted || entry.isEncrypted) {
        if (!this._decryptBlock) {
          throw new EncryptedArchiveError(this.header.encryptionTag, this.header.encryptionLabel);
        }
        compressedBytes = toUint8Array(
          await this._decryptBlock(compressedBytes, {
            region: 'file',
            encryptionTag: this.header.encryptionTag,
            encryptionLabel: this.header.encryptionLabel,
            header: this.header,
            entry,
            offset: 0,
            length: entry.storedSize,
          }),
          'decryptBlock(file) result'
        );
      }
      const decompressed = toUint8Array(
        await this._decompressEntry(compressedBytes, entry),
        'decompressEntry result'
      );
      const end =
        length === undefined || length === null
          ? decompressed.byteLength
          : Math.min(decompressed.byteLength, offset + length);
      return decompressed.slice(offset, end);
    }

    const fileSize = readRaw ? entry.storedSize : entry.uncompressedSize;
    if (offset >= fileSize || length === 0) {
      return new Uint8Array(0);
    }

    const requested = length === undefined || length === null ? fileSize - offset : length;
    const clampedLength = Math.min(requested, fileSize - offset);
    if (clampedLength <= 0) {
      return new Uint8Array(0);
    }

    if ((this.header.isEncrypted || entry.isEncrypted) && !this._decryptBlock) {
      throw new EncryptedArchiveError(this.header.encryptionTag, this.header.encryptionLabel);
    }

    const archiveOffset = entry.dataOffset + offset;
    let bytes = await this._readRange(archiveOffset, clampedLength);
    if (bytes.byteLength > clampedLength) {
      bytes = bytes.subarray(0, clampedLength);
    }

    if (this.header.isEncrypted || entry.isEncrypted) {
      bytes = toUint8Array(
        await this._decryptBlock(bytes, {
          region: 'file',
          encryptionTag: this.header.encryptionTag,
          encryptionLabel: this.header.encryptionLabel,
          header: this.header,
          entry,
          offset,
          length: clampedLength,
        }),
        'decryptBlock(file) result'
      );
    }

    return bytes;
  }

  async readFile(virtualPath, options = {}) {
    return this.readFileRange(virtualPath, 0, undefined, options);
  }
}

/**
 * Multi-archive virtual file system mount table with RAGE-style override priority
 * (later mounts override earlier mounts by default).
 */
class RpfVfsMountTable {
  constructor() {
    this._mounts = [];
    this._nextId = 1;
    this._nextOrder = 1;
  }

  mount(mountPrefix, vfs, options = {}) {
    if (
      !vfs ||
      typeof vfs.exists !== 'function' ||
      typeof vfs.stat !== 'function' ||
      typeof vfs.listDir !== 'function' ||
      typeof vfs.readFileRange !== 'function'
    ) {
      throw new RpfFormatError(
        'INVALID_MOUNT',
        'mount() requires an RpfArchiveVfs-compatible instance'
      );
    }

    const norm = normalizeVirtualPath(mountPrefix);
    const id = `mount_${this._nextId}`;
    this._nextId += 1;
    const order = this._nextOrder;
    this._nextOrder += 1;
    const priority = Number.isFinite(options.priority) ? options.priority : 0;

    const record = {
      id,
      prefix: norm.path,
      prefixKey: norm.key,
      prefixSegments: norm.segments,
      vfs,
      priority,
      order,
    };
    this._mounts.push(record);
    return {
      id: record.id,
      prefix: record.prefix,
      priority: record.priority,
      order: record.order,
    };
  }

  unmount(idOrVfs) {
    const before = this._mounts.length;
    this._mounts = this._mounts.filter((m) => m.id !== idOrVfs && m.vfs !== idOrVfs);
    return this._mounts.length !== before;
  }

  _orderedMounts() {
    // Higher explicit priority wins; for equal priority, later mount (higher order) wins.
    return this._mounts
      .slice()
      .sort((a, b) => (b.priority !== a.priority ? b.priority - a.priority : b.order - a.order));
  }

  listMounts() {
    return this._orderedMounts().map((m) => ({
      id: m.id,
      prefix: m.prefix,
      priority: m.priority,
      order: m.order,
      archivePath: m.vfs.archivePath || null,
    }));
  }

  _matchMount(mount, pathKey, pathDisplay) {
    if (mount.prefixKey === '') {
      return { matched: true, relPath: pathDisplay, relKey: pathKey };
    }
    if (pathKey === mount.prefixKey) {
      return { matched: true, relPath: '', relKey: '' };
    }
    if (pathKey.startsWith(`${mount.prefixKey}/`)) {
      const relStart = mount.prefixKey.length + 1;
      return {
        matched: true,
        relPath: pathDisplay.slice(relStart),
        relKey: pathKey.slice(relStart),
      };
    }
    return { matched: false, relPath: '', relKey: '' };
  }

  _isVirtualPrefixDir(pathKey) {
    if (pathKey === '') return true;
    const prefixCheck = `${pathKey}/`;
    for (const m of this._mounts) {
      if (m.prefixKey === pathKey || m.prefixKey.startsWith(prefixCheck)) {
        return true;
      }
    }
    return false;
  }

  resolve(virtualPath) {
    const norm = normalizeVirtualPath(virtualPath);
    const ordered = this._orderedMounts();
    for (const m of ordered) {
      const match = this._matchMount(m, norm.key, norm.path);
      if (!match.matched) continue;
      if (m.vfs.exists(match.relPath)) {
        const st = m.vfs.stat(match.relPath);
        const fullPath =
          m.prefix && st.path ? `${m.prefix}/${st.path}` : m.prefix || st.path;
        return {
          mountId: m.id,
          mountPrefix: m.prefix,
          vfs: m.vfs,
          archivePath: st.path,
          stat: Object.assign({}, st, {
            path: fullPath,
            mountId: m.id,
            mountPrefix: m.prefix,
          }),
        };
      }
    }
    return null;
  }

  exists(virtualPath) {
    let norm;
    try {
      norm = normalizeVirtualPath(virtualPath);
    } catch {
      return false;
    }
    if (this.resolve(norm.path) !== null) {
      return true;
    }
    return this._isVirtualPrefixDir(norm.key);
  }

  stat(virtualPath) {
    const norm = normalizeVirtualPath(virtualPath);
    const resolved = this.resolve(norm.path);
    if (resolved) {
      return resolved.stat;
    }
    if (this._isVirtualPrefixDir(norm.key)) {
      const name = norm.segments.length > 0 ? norm.segments[norm.segments.length - 1] : '';
      return {
        path: norm.path,
        name,
        kind: 'directory',
        isDirectory: true,
        isFile: false,
        virtual: true,
      };
    }
    return null;
  }

  listDir(dirPath = '') {
    const norm = normalizeVirtualPath(dirPath);
    const ordered = this._orderedMounts();

    const winning = this.resolve(norm.path);
    if (winning && winning.stat.isFile) {
      throw new RpfFormatError('NOT_A_DIRECTORY', `Path is not a directory: ${dirPath}`);
    }

    const byNameKey = new Map();
    let foundAnyDir = norm.key === '';

    // Add children from matching mounts in highest-to-lowest priority order so higher
    // priority mounts win on name collisions.
    for (const m of ordered) {
      const match = this._matchMount(m, norm.key, norm.path);
      if (match.matched && m.vfs.exists(match.relPath)) {
        const st = m.vfs.stat(match.relPath);
        if (st && st.isDirectory) {
          foundAnyDir = true;
          const entries = m.vfs.listDir(match.relPath);
          for (const entry of entries) {
            const nameKey = entry.name.toLowerCase();
            if (!byNameKey.has(nameKey)) {
              const fullPath =
                m.prefix && entry.path ? `${m.prefix}/${entry.path}` : m.prefix || entry.path;
              byNameKey.set(
                nameKey,
                Object.assign({}, entry, {
                  path: fullPath,
                  mountId: m.id,
                  mountPrefix: m.prefix,
                })
              );
            }
          }
        }
      }

      // Also expose virtual mount prefix subdirectories visible from dirPath.
      if (m.prefixSegments.length > norm.segments.length) {
        let prefixMatches = true;
        for (let i = 0; i < norm.segments.length; i += 1) {
          if (m.prefixSegments[i].toLowerCase() !== norm.segments[i].toLowerCase()) {
            prefixMatches = false;
            break;
          }
        }
        if (prefixMatches) {
          foundAnyDir = true;
          const nextSeg = m.prefixSegments[norm.segments.length];
          const nextKey = nextSeg.toLowerCase();
          if (!byNameKey.has(nextKey)) {
            const fullPath = norm.path ? `${norm.path}/${nextSeg}` : nextSeg;
            byNameKey.set(nextKey, {
              path: fullPath,
              name: nextSeg,
              kind: 'directory',
              isDirectory: true,
              isFile: false,
              virtual: true,
            });
          }
        }
      }
    }

    if (!foundAnyDir) {
      throw new RpfFormatError('PATH_NOT_FOUND', `Directory not found in mount table: ${dirPath}`);
    }

    return Array.from(byNameKey.values()).sort((a, b) =>
      a.name.toLowerCase().localeCompare(b.name.toLowerCase())
    );
  }

  async readFileRange(virtualPath, offset = 0, length = undefined, options = {}) {
    const resolved = this.resolve(virtualPath);
    if (!resolved) {
      throw new RpfFormatError('PATH_NOT_FOUND', `File not found in mount table: ${virtualPath}`);
    }
    if (!resolved.stat.isFile) {
      throw new RpfFormatError('NOT_A_FILE', `Path is a directory, not a file: ${virtualPath}`);
    }
    return resolved.vfs.readFileRange(resolved.archivePath, offset, length, options);
  }

  async readFile(virtualPath, options = {}) {
    return this.readFileRange(virtualPath, 0, undefined, options);
  }
}

/**
 * Builds a valid synthetic multi-directory RPF9/8/7 byte image for unit tests.
 *
 * @param {Array<object>|Record<string, Uint8Array|ArrayBuffer|string>} files
 * @param {object} [options]
 * @returns {Uint8Array}
 */
function buildSyntheticRpf9Buffer(files = [], options = {}) {
  const magic = options.magic || RPF9_MAGIC;
  if (magic !== RPF9_MAGIC && magic !== RPF8_MAGIC && magic !== RPF7_MAGIC) {
    throw new RpfFormatError('INVALID_MAGIC', `Unsupported synthetic RPF magic: ${magic}`);
  }
  const encryptionTag = (options.encryptionTag ?? RPF_ENCRYPTION_NONE) >>> 0;
  const headerFlags = (options.flags ?? 0) >>> 0;
  const sectorAlign = Boolean(options.sectorAlign);
  const paddingAfterToc =
    Number.isInteger(options.paddingAfterToc) && options.paddingAfterToc > 0
      ? options.paddingAfterToc
      : 0;
  const encryptBlock = typeof options.encryptBlock === 'function' ? options.encryptBlock : null;

  let metadataBytes = new Uint8Array(0);
  if (options.metadata !== undefined && options.metadata !== null) {
    metadataBytes = toUint8Array(options.metadata, 'metadata');
  } else if (Number.isInteger(options.metadataSize) && options.metadataSize > 0) {
    metadataBytes = new Uint8Array(options.metadataSize);
  }

  // Normalize file descriptors (accepts Array or plain Object map).
  const descriptors = [];
  if (Array.isArray(files)) {
    for (const item of files) {
      if (!item || typeof item.path !== 'string') {
        throw new RpfFormatError('INVALID_PATH', 'Synthetic file entry requires a string path');
      }
      descriptors.push(item);
    }
  } else if (files && typeof files === 'object') {
    for (const [filePath, data] of Object.entries(files)) {
      descriptors.push({ path: filePath, data });
    }
  } else {
    throw new RpfFormatError('INVALID_TOC', 'files must be an array or path-to-bytes map');
  }

  // Build hierarchical directory tree rooted at "".
  const root = {
    kind: 'directory',
    name: '',
    children: new Map(),
    flags: RPF_ENTRY_FLAG_DIRECTORY,
  };

  for (const desc of descriptors) {
    const norm = normalizeVirtualPath(desc.path);
    if (norm.segments.length === 0) {
      continue;
    }

    let cursor = root;
    const isExplicitDir = desc.kind === 'directory';
    const dirLimit = isExplicitDir ? norm.segments.length : norm.segments.length - 1;

    for (let i = 0; i < dirLimit; i += 1) {
      const seg = norm.segments[i];
      const segKey = seg.toLowerCase();
      let next = cursor.children.get(segKey);
      if (!next) {
        next = {
          kind: 'directory',
          name: seg,
          children: new Map(),
          flags: RPF_ENTRY_FLAG_DIRECTORY,
        };
        cursor.children.set(segKey, next);
      } else if (next.kind !== 'directory') {
        throw new RpfFormatError(
          'INVALID_TOC',
          `Path conflict in synthetic RPF builder at segment "${seg}" of "${desc.path}"`
        );
      }
      cursor = next;
    }

    if (!isExplicitDir) {
      const fileName = norm.segments[norm.segments.length - 1];
      const fileKey = fileName.toLowerCase();
      let payload;
      if (typeof desc.data === 'string') {
        payload = utf8Encoder.encode(desc.data);
      } else if (desc.data === undefined || desc.data === null) {
        payload = new Uint8Array(0);
      } else {
        payload = toUint8Array(desc.data, `data for ${desc.path}`);
      }

      let entryFlags = (desc.flags ?? 0) & ~RPF_ENTRY_FLAG_DIRECTORY;
      if (desc.isCompressed) entryFlags |= RPF_ENTRY_FLAG_COMPRESSED;
      if (desc.isResource) entryFlags |= RPF_ENTRY_FLAG_RESOURCE;
      if (desc.isEncrypted) entryFlags |= RPF_ENTRY_FLAG_ENCRYPTED;
      if (sectorAlign) entryFlags |= RPF_ENTRY_FLAG_OFFSET_SHIFT9;

      const isCompressed = (entryFlags & RPF_ENTRY_FLAG_COMPRESSED) !== 0;
      const uncompressedSize =
        desc.uncompressedSize !== undefined ? desc.uncompressedSize : payload.byteLength;
      const compressedSize = isCompressed
        ? desc.compressedSize !== undefined
          ? desc.compressedSize
          : payload.byteLength
        : 0;

      cursor.children.set(fileKey, {
        kind: 'file',
        name: fileName,
        data: payload,
        flags: entryFlags,
        uncompressedSize,
        compressedSize,
      });
    }
  }

  // Breadth-first flattening ensures each directory's direct children are strictly contiguous.
  const flatEntries = [root];
  const bfsQueue = [root];
  while (bfsQueue.length > 0) {
    const dir = bfsQueue.shift();
    const sortedChildren = Array.from(dir.children.values()).sort((a, b) =>
      a.name.toLowerCase().localeCompare(b.name.toLowerCase())
    );
    dir.firstChildIndex = sortedChildren.length > 0 ? flatEntries.length : 0;
    dir.childCount = sortedChildren.length;

    for (const child of sortedChildren) {
      flatEntries.push(child);
      if (child.kind === 'directory') {
        bfsQueue.push(child);
      }
    }
  }

  // Build NUL-terminated UTF-8 names table.
  const nameChunks = [];
  let namesSize = 0;
  for (const entry of flatEntries) {
    if (namesSize > MAX_U24) {
      throw new RpfFormatError('SIZE_OVERFLOW', 'Synthetic RPF names table exceeds u24 limit');
    }
    entry.nameOffset = namesSize;
    const encoded = utf8Encoder.encode(entry.name);
    const withNul = new Uint8Array(encoded.byteLength + 1);
    withNul.set(encoded, 0);
    withNul[encoded.byteLength] = 0;
    nameChunks.push(withNul);
    namesSize += withNul.byteLength;
  }

  const namesBytes = new Uint8Array(namesSize);
  let nameCursor = 0;
  for (const chunk of nameChunks) {
    namesBytes.set(chunk, nameCursor);
    nameCursor += chunk.byteLength;
  }

  const tocBytesLen = flatEntries.length * RPF9_ENTRY_STRIDE;
  const namesOffset = RPF9_HEADER_SIZE + tocBytesLen;
  const metadataOffset = namesOffset + namesSize;
  let payloadCursor = metadataOffset + metadataBytes.byteLength + paddingAfterToc;

  // Assign file payload offsets.
  for (const entry of flatEntries) {
    if (entry.kind !== 'file') continue;
    if (sectorAlign) {
      const rem = payloadCursor % RPF_SECTOR_SIZE;
      if (rem !== 0) {
        payloadCursor += RPF_SECTOR_SIZE - rem;
      }
    }
    entry.dataOffset = payloadCursor;
    payloadCursor += entry.data.byteLength;
  }

  const totalSize = payloadCursor;
  const out = new Uint8Array(totalSize);
  const outView = new DataView(out.buffer);

  // Write 24-byte header.
  out[0] = magic.charCodeAt(0);
  out[1] = magic.charCodeAt(1);
  out[2] = magic.charCodeAt(2);
  out[3] = magic.charCodeAt(3);
  outView.setUint32(4, flatEntries.length, true);
  outView.setUint32(8, namesSize, true);
  outView.setUint32(12, encryptionTag, true);
  outView.setUint32(16, metadataBytes.byteLength, true);
  outView.setUint32(20, headerFlags, true);

  // Build TOC buffer.
  const tocBuf = new Uint8Array(tocBytesLen);
  const tocView = new DataView(tocBuf.buffer);
  for (let i = 0; i < flatEntries.length; i += 1) {
    const entry = flatEntries[i];
    const base = i * RPF9_ENTRY_STRIDE;
    const w0 = (((entry.flags & 0xff) << 24) | (entry.nameOffset & MAX_U24)) >>> 0;
    tocView.setUint32(base, w0, true);
    if (entry.kind === 'directory') {
      tocView.setUint32(base + 4, entry.firstChildIndex >>> 0, true);
      tocView.setUint32(base + 8, entry.childCount >>> 0, true);
      tocView.setUint32(base + 12, 0, true);
    } else {
      const useSector = (entry.flags & RPF_ENTRY_FLAG_OFFSET_SHIFT9) !== 0;
      const storedOffsetWord = useSector ? entry.dataOffset / RPF_SECTOR_SIZE : entry.dataOffset;
      if (!Number.isInteger(storedOffsetWord) || storedOffsetWord > MAX_U32) {
        throw new RpfFormatError(
          'SIZE_OVERFLOW',
          `File offset ${entry.dataOffset} cannot be encoded in 32-bit TOC offset word`
        );
      }
      tocView.setUint32(base + 4, storedOffsetWord >>> 0, true);
      tocView.setUint32(base + 8, entry.uncompressedSize >>> 0, true);
      tocView.setUint32(base + 12, entry.compressedSize >>> 0, true);
    }
  }

  const finalToc = encryptBlock
    ? toUint8Array(encryptBlock(tocBuf.slice(), { region: 'toc', encryptionTag }), 'encryptBlock(toc)')
    : tocBuf;
  const finalNames = encryptBlock
    ? toUint8Array(
        encryptBlock(namesBytes.slice(), { region: 'names', encryptionTag }),
        'encryptBlock(names)'
      )
    : namesBytes;

  out.set(finalToc, RPF9_HEADER_SIZE);
  out.set(finalNames, namesOffset);
  if (metadataBytes.byteLength > 0) {
    out.set(metadataBytes, metadataOffset);
  }

  // Write file payloads.
  for (const entry of flatEntries) {
    if (entry.kind !== 'file' || entry.data.byteLength === 0) continue;
    const bytesToWrite =
      encryptBlock && (classifyEncryption(encryptionTag).isEncrypted || (entry.flags & RPF_ENTRY_FLAG_ENCRYPTED) !== 0)
        ? toUint8Array(
            encryptBlock(entry.data.slice(), {
              region: 'file',
              encryptionTag,
              name: entry.name,
            }),
            'encryptBlock(file)'
          )
        : entry.data;
    out.set(bytesToWrite, entry.dataOffset);
  }

  return out;
}

const exportedApi = {
  RPF9_MAGIC,
  RPF8_MAGIC,
  RPF7_MAGIC,
  RPF9_HEADER_SIZE,
  RPF9_ENTRY_STRIDE,
  RPF_SECTOR_SIZE,
  RPF_ENCRYPTION_NONE,
  RPF_ENCRYPTION_OPEN,
  RPF_ENCRYPTION_AES,
  RPF_ENCRYPTION_NG,
  RPF_ENTRY_FLAG_DIRECTORY,
  RPF_ENTRY_FLAG_COMPRESSED,
  RPF_ENTRY_FLAG_RESOURCE,
  RPF_ENTRY_FLAG_ENCRYPTED,
  RPF_ENTRY_FLAG_OFFSET_SHIFT9,
  RpfFormatError,
  EncryptedArchiveError,
  classifyEncryption,
  parseRpfHeader,
  parseTocEntry,
  normalizeVirtualPath,
  RpfArchiveVfs,
  RpfVfsMountTable,
  buildSyntheticRpf9Buffer,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Rpf9Vfs = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
