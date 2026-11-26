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

// Big-endian / synthetic ASCII FourCC magics alongside real RAGE little-endian FourCC magics
// (0x52504639 LE = [0x39, 0x46, 0x50, 0x52] = "9FPR", etc.).
const RPF9_MAGIC_BE = RPF9_MAGIC;
const RPF8_MAGIC_BE = RPF8_MAGIC;
const RPF7_MAGIC_BE = RPF7_MAGIC;
const RPF9_MAGIC_LE = '9FPR';
const RPF8_MAGIC_LE = '8FPR';
const RPF7_MAGIC_LE = '7FPR';
const RPF9_MAGIC_U32 = 0x52504639;
const RPF8_MAGIC_U32 = 0x52504638;
const RPF7_MAGIC_U32 = 0x52504637;

// Header layout matching tools/src/rpf9.rs (24 bytes, little-endian):
//   0x00..0x04: magic ("9FPR"/"8FPR"/"7FPR" LE or synthetic "RPF9"/"RPF8"/"RPF7" BE)
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
const MAX_U32_BIGINT = 0xffffffffn;
const MAX_U64_BIGINT = 0xffffffffffffffffn;
const MAX_SAFE_BIGINT = BigInt(Number.MAX_SAFE_INTEGER);

const DEFAULT_PATH_CACHE_CAPACITY = 1024;
const DEFAULT_NEGATIVE_CACHE_CAPACITY = 2048;
const DEFAULT_DIR_CACHE_CAPACITY = 256;
const DEFAULT_TOC_CACHE_CAPACITY = 64;

const utf8Decoder = new TextDecoder('utf-8');
const utf8Encoder = new TextEncoder();
const ASCII_NORM_SCRATCH_CAPACITY = 1024;
const ASCII_NORM_SCRATCH = new Uint16Array(ASCII_NORM_SCRATCH_CAPACITY);

function normalizeCacheCapacity(value, defaultCapacity) {
  if (value === undefined || value === null) {
    return defaultCapacity;
  }
  if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
    return Math.floor(value);
  }
  return defaultCapacity;
}

/**
 * Standard unsigned 32-bit RAGE Jenkins One-At-A-Time (JOAAT) hash over
 * lowercased UTF-8 bytes. Uses a zero-allocation fast path for ASCII strings.
 *
 * @param {string} input
 * @returns {number} Unsigned 32-bit integer hash.
 */
function joaatHash(input) {
  if (typeof input !== 'string') {
    throw new RpfFormatError('INVALID_PATH', 'Hash input must be a string');
  }
  const len = input.length;
  let hash = 0;
  let asciiOnly = true;
  for (let i = 0; i < len; i += 1) {
    let c = input.charCodeAt(i);
    if (c >= 0x80) {
      asciiOnly = false;
      break;
    }
    if (c >= 0x41 && c <= 0x5a) {
      c += 0x20;
    }
    hash = (hash + c) | 0;
    hash = (hash + (hash << 10)) | 0;
    hash ^= hash >>> 6;
  }
  if (!asciiOnly) {
    const bytes = utf8Encoder.encode(input.toLowerCase());
    hash = 0;
    for (let i = 0; i < bytes.length; i += 1) {
      hash = (hash + bytes[i]) | 0;
      hash = (hash + (hash << 10)) | 0;
      hash ^= hash >>> 6;
    }
  }
  hash = (hash + (hash << 3)) | 0;
  hash ^= hash >>> 11;
  hash = (hash + (hash << 15)) | 0;
  return hash >>> 0;
}

/**
 * Computes the normalized lowercase 32-bit JOAAT hash for a virtual RPF path.
 * Uses a zero-allocation single-pass ASCII fast path that canonicalizes slashes,
 * backslashes, uppercase ASCII, and `.` segments inline while rejecting `..` and NUL.
 *
 * @param {string} virtualPath
 * @returns {number} Unsigned 32-bit integer hash.
 */
function hashVirtualPath(virtualPath) {
  if (virtualPath === undefined || virtualPath === null || virtualPath === '') {
    return 0;
  }
  if (typeof virtualPath !== 'string') {
    throw new RpfFormatError('INVALID_PATH', 'Virtual path must be a string');
  }

  const len = virtualPath.length;
  let hash = 0;
  let hasOutput = false;
  let i = 0;

  while (i < len) {
    // Skip leading/duplicate slashes or backslashes.
    while (i < len) {
      const c = virtualPath.charCodeAt(i);
      if (c === 0) {
        throw new RpfFormatError('INVALID_PATH', `Virtual path contains NUL byte: ${virtualPath}`);
      }
      if (c !== 0x2f && c !== 0x5c) {
        break;
      }
      i += 1;
    }
    if (i >= len) {
      break;
    }

    const segStart = i;
    let asciiOnly = true;
    while (i < len) {
      const c = virtualPath.charCodeAt(i);
      if (c === 0) {
        throw new RpfFormatError('INVALID_PATH', `Virtual path contains NUL byte: ${virtualPath}`);
      }
      if (c === 0x2f || c === 0x5c) {
        break;
      }
      if (c >= 0x80) {
        asciiOnly = false;
        break;
      }
      i += 1;
    }

    if (!asciiOnly) {
      const norm = normalizeVirtualPath(virtualPath);
      return joaatHash(norm.key);
    }

    const segLen = i - segStart;
    if (segLen === 1 && virtualPath.charCodeAt(segStart) === 0x2e) {
      continue;
    }
    if (
      segLen === 2 &&
      virtualPath.charCodeAt(segStart) === 0x2e &&
      virtualPath.charCodeAt(segStart + 1) === 0x2e
    ) {
      throw new RpfFormatError('INVALID_PATH', `Path traversal is not allowed: ${virtualPath}`);
    }

    if (hasOutput) {
      hash = (hash + 0x2f) | 0;
      hash = (hash + (hash << 10)) | 0;
      hash ^= hash >>> 6;
    }
    for (let j = segStart; j < i; j += 1) {
      let c = virtualPath.charCodeAt(j);
      if (c >= 0x41 && c <= 0x5a) {
        c += 0x20;
      }
      hash = (hash + c) | 0;
      hash = (hash + (hash << 10)) | 0;
      hash ^= hash >>> 6;
    }
    hasOutput = true;
  }

  hash = (hash + (hash << 3)) | 0;
  hash ^= hash >>> 11;
  hash = (hash + (hash << 15)) | 0;
  return hash >>> 0;
}

function isAsciiWhitespace(c) {
  return c === 0x20 || (c >= 0x09 && c <= 0x0d);
}

/**
 * Zero-allocation parser for an explicit `0x`-prefixed 32-bit hex hash string.
 * Returns unsigned u32 `[0..0xFFFFFFFF]` on match, or `-1` otherwise.
 */
function parseExplicitHexHash(str) {
  const len = str.length;
  let start = 0;
  let end = len;
  while (start < end && isAsciiWhitespace(str.charCodeAt(start))) {
    start += 1;
  }
  while (end > start && isAsciiWhitespace(str.charCodeAt(end - 1))) {
    end -= 1;
  }
  if (
    end - start < 3 ||
    str.charCodeAt(start) !== 0x30 ||
    (str.charCodeAt(start + 1) !== 0x78 && str.charCodeAt(start + 1) !== 0x58)
  ) {
    return -1;
  }
  start += 2;
  const digitCount = end - start;
  if (digitCount < 1 || digitCount > 8) {
    return -1;
  }
  let acc = 0;
  for (let i = start; i < end; i += 1) {
    const c = str.charCodeAt(i);
    let nibble;
    if (c >= 0x30 && c <= 0x39) {
      nibble = c - 0x30;
    } else if (c >= 0x41 && c <= 0x46) {
      nibble = c - 0x37;
    } else if (c >= 0x61 && c <= 0x66) {
      nibble = c - 0x57;
    } else {
      return -1;
    }
    acc = ((acc << 4) | nibble) >>> 0;
  }
  return acc;
}

function parseHashKey(hashInput) {
  if (typeof hashInput === 'number') {
    if ((hashInput >>> 0) !== hashInput) {
      throw new RpfFormatError(
        'INVALID_HASH',
        `RPF entry hash must be an unsigned 32-bit integer: ${hashInput}`
      );
    }
    return hashInput >>> 0;
  }
  if (typeof hashInput === 'bigint') {
    if (hashInput < 0n || hashInput > MAX_U32_BIGINT) {
      throw new RpfFormatError(
        'INVALID_HASH',
        `RPF entry hash BigInt must be an unsigned 32-bit integer: ${hashInput}`
      );
    }
    return Number(hashInput) >>> 0;
  }
  if (typeof hashInput === 'string') {
    const len = hashInput.length;
    let start = 0;
    let end = len;
    while (start < end && isAsciiWhitespace(hashInput.charCodeAt(start))) {
      start += 1;
    }
    while (end > start && isAsciiWhitespace(hashInput.charCodeAt(end - 1))) {
      end -= 1;
    }
    if (
      end - start >= 2 &&
      hashInput.charCodeAt(start) === 0x30 &&
      (hashInput.charCodeAt(start + 1) === 0x78 || hashInput.charCodeAt(start + 1) === 0x58)
    ) {
      start += 2;
    }
    const digitCount = end - start;
    if (digitCount < 1 || digitCount > 8) {
      throw new RpfFormatError('INVALID_HASH', `Invalid hex RPF entry hash: ${hashInput}`);
    }
    let acc = 0;
    for (let i = start; i < end; i += 1) {
      const c = hashInput.charCodeAt(i);
      let nibble;
      if (c >= 0x30 && c <= 0x39) {
        nibble = c - 0x30;
      } else if (c >= 0x41 && c <= 0x46) {
        nibble = c - 0x37;
      } else if (c >= 0x61 && c <= 0x66) {
        nibble = c - 0x57;
      } else {
        throw new RpfFormatError('INVALID_HASH', `Invalid hex RPF entry hash: ${hashInput}`);
      }
      acc = ((acc << 4) | nibble) >>> 0;
    }
    return acc;
  }
  throw new RpfFormatError(
    'INVALID_HASH',
    'RPF entry hash must be an unsigned 32-bit integer, BigInt, or hex string'
  );
}

/**
 * O(1) bounded LRU cache backed by JavaScript Map insertion order.
 */
class BoundedLruCache {
  constructor(capacity = DEFAULT_PATH_CACHE_CAPACITY) {
    this._capacity = normalizeCacheCapacity(capacity, DEFAULT_PATH_CACHE_CAPACITY);
    this._map = new Map();
    this._hits = 0;
    this._misses = 0;
    this._evictions = 0;
  }

  get capacity() {
    return this._capacity;
  }

  get size() {
    return this._map.size;
  }

  get hits() {
    return this._hits;
  }

  get misses() {
    return this._misses;
  }

  get evictions() {
    return this._evictions;
  }

  get(key) {
    if (this._capacity <= 0) {
      this._misses += 1;
      return undefined;
    }
    const val = this._map.get(key);
    if (val === undefined) {
      this._misses += 1;
      return undefined;
    }
    this._map.delete(key);
    this._map.set(key, val);
    this._hits += 1;
    return val;
  }

  getIfPresent(key) {
    if (this._capacity <= 0) {
      return undefined;
    }
    const val = this._map.get(key);
    if (val === undefined) {
      return undefined;
    }
    this._map.delete(key);
    this._map.set(key, val);
    this._hits += 1;
    return val;
  }

  recordMiss() {
    this._misses += 1;
  }

  set(key, value) {
    if (this._capacity <= 0 || value === undefined) {
      return;
    }
    if (this._map.has(key)) {
      this._map.delete(key);
    } else if (this._map.size >= this._capacity) {
      const oldestKey = this._map.keys().next().value;
      this._map.delete(oldestKey);
      this._evictions += 1;
    }
    this._map.set(key, value);
  }

  has(key) {
    return this._capacity > 0 && this._map.has(key);
  }

  delete(key) {
    return this._map.delete(key);
  }

  clear() {
    this._map.clear();
  }

  resetStats() {
    this._hits = 0;
    this._misses = 0;
    this._evictions = 0;
  }
}

/**
 * O(1) bounded LRU negative-lookup cache backed by JavaScript Map insertion order.
 * Keyed by normalized lowercase virtual path with bounded non-canonical raw-path
 * aliases so repeated missing probes (canonical or mixed-case/backslash) resolve
 * in O(1) without repeating path normalization or polluting the positive cache.
 */
class BoundedNegativeLookupCache {
  constructor(capacity = DEFAULT_NEGATIVE_CACHE_CAPACITY) {
    this._capacity = normalizeCacheCapacity(capacity, DEFAULT_NEGATIVE_CACHE_CAPACITY);
    this._map = new Map();
    this._rawToNorm = new Map();
    this._hits = 0;
    this._misses = 0;
    this._evictions = 0;
  }

  get capacity() {
    return this._capacity;
  }

  get size() {
    return this._map.size;
  }

  get hits() {
    return this._hits;
  }

  get misses() {
    return this._misses;
  }

  get evictions() {
    return this._evictions;
  }

  _deleteRawAliases(aliases) {
    if (aliases === null || aliases === undefined) {
      return;
    }
    if (typeof aliases === 'string') {
      this._rawToNorm.delete(aliases);
      return;
    }
    for (const raw of aliases) {
      this._rawToNorm.delete(raw);
    }
  }

  _attachRawAlias(normKey, rawKey) {
    if (typeof rawKey !== 'string' || rawKey === normKey || this._rawToNorm.has(rawKey)) {
      return;
    }
    const aliases = this._map.get(normKey);
    if (aliases === undefined) {
      return;
    }
    if (aliases === null) {
      this._map.set(normKey, rawKey);
      this._rawToNorm.set(rawKey, normKey);
    } else if (typeof aliases === 'string') {
      const set = new Set([aliases, rawKey]);
      this._map.set(normKey, set);
      this._rawToNorm.set(rawKey, normKey);
    } else if (aliases.size < 4) {
      aliases.add(rawKey);
      this._rawToNorm.set(rawKey, normKey);
    }
  }

  getByRaw(rawKey) {
    if (this._capacity <= 0) {
      return null;
    }
    if (this._map.has(rawKey)) {
      const aliases = this._map.get(rawKey);
      this._map.delete(rawKey);
      this._map.set(rawKey, aliases);
      this._hits += 1;
      return rawKey;
    }
    const normKey = this._rawToNorm.get(rawKey);
    if (normKey !== undefined) {
      const aliases = this._map.get(normKey);
      if (aliases !== undefined) {
        this._map.delete(normKey);
        this._map.set(normKey, aliases);
        this._hits += 1;
        return normKey;
      }
      this._rawToNorm.delete(rawKey);
    }
    return null;
  }

  getByNormalized(normKey, rawKey = undefined) {
    if (this._capacity <= 0) {
      return false;
    }
    const aliases = this._map.get(normKey);
    if (aliases === undefined) {
      return false;
    }
    this._map.delete(normKey);
    this._map.set(normKey, aliases);
    if (rawKey !== undefined && rawKey !== normKey) {
      this._attachRawAlias(normKey, rawKey);
    }
    this._hits += 1;
    return true;
  }

  set(normKey, rawKey = undefined) {
    this._misses += 1;
    if (this._capacity <= 0) {
      return;
    }
    if (this._map.has(normKey)) {
      const aliases = this._map.get(normKey);
      this._map.delete(normKey);
      this._map.set(normKey, aliases);
    } else {
      if (this._map.size >= this._capacity) {
        const oldestKey = this._map.keys().next().value;
        const oldestAliases = this._map.get(oldestKey);
        this._map.delete(oldestKey);
        this._deleteRawAliases(oldestAliases);
        this._evictions += 1;
      }
      this._map.set(normKey, null);
    }
    if (rawKey !== undefined && rawKey !== normKey) {
      this._attachRawAlias(normKey, rawKey);
    }
  }

  has(normKey) {
    return this._capacity > 0 && this._map.has(normKey);
  }

  delete(normKey) {
    const aliases = this._map.get(normKey);
    if (aliases === undefined) {
      return false;
    }
    this._map.delete(normKey);
    this._deleteRawAliases(aliases);
    return true;
  }

  clear() {
    this._map.clear();
    this._rawToNorm.clear();
  }

  resetStats() {
    this._hits = 0;
    this._misses = 0;
    this._evictions = 0;
  }
}

/**
 * Bounded LRU cache for parsed RPF TOC headers, entry tables, and pre-built
 * O(1) path/hash indexes so repeatedly opened archives skip TOC fetches and
 * index rebuilds.
 */
class RpfTocCache {
  constructor(options = {}) {
    const cap =
      typeof options === 'number'
        ? options
        : options && options.capacity !== undefined
          ? options.capacity
          : options && options.maxEntries !== undefined
            ? options.maxEntries
            : DEFAULT_TOC_CACHE_CAPACITY;
    this._lru = new BoundedLruCache(normalizeCacheCapacity(cap, DEFAULT_TOC_CACHE_CAPACITY));
  }

  get(key) {
    if (typeof key !== 'string' || key.length === 0) {
      return undefined;
    }
    return this._lru.get(key);
  }

  set(key, tocBundle) {
    if (typeof key !== 'string' || key.length === 0 || !tocBundle) {
      return;
    }
    this._lru.set(key, tocBundle);
  }

  has(key) {
    return typeof key === 'string' && key.length > 0 && this._lru.has(key);
  }

  delete(key) {
    if (typeof key !== 'string' || key.length === 0) {
      return false;
    }
    return this._lru.delete(key);
  }

  clear() {
    this._lru.clear();
  }

  getStats() {
    return {
      size: this._lru.size,
      capacity: this._lru.capacity,
      hits: this._lru.hits,
      misses: this._lru.misses,
      evictions: this._lru.evictions,
    };
  }

  resetStats() {
    this._lru.resetStats();
  }
}

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

  // Real RAGE little-endian FourCC: '9'/'8'/'7', 'F'=0x46, 'P'=0x50, 'R'=0x52
  // (e.g. u32 LE 0x52504639 -> [0x39, 0x46, 0x50, 0x52] = "9FPR").
  if (b1 === 0x46 && b2 === 0x50 && b3 === 0x52) {
    if (b0 === 0x39) {
      return {
        magic: RPF9_MAGIC,
        rawMagic: RPF9_MAGIC_LE,
        magicEndian: 'little',
        version: 9,
      };
    }
    if (b0 === 0x38) {
      return {
        magic: RPF8_MAGIC,
        rawMagic: RPF8_MAGIC_LE,
        magicEndian: 'little',
        version: 8,
      };
    }
    if (b0 === 0x37) {
      return {
        magic: RPF7_MAGIC,
        rawMagic: RPF7_MAGIC_LE,
        magicEndian: 'little',
        version: 7,
      };
    }
  }

  // Synthetic / big-endian ASCII FourCC: 'R'=0x52, 'P'=0x50, 'F'=0x46, '9'/'8'/'7'.
  if (b0 === 0x52 && b1 === 0x50 && b2 === 0x46) {
    if (b3 === 0x39) {
      return {
        magic: RPF9_MAGIC,
        rawMagic: RPF9_MAGIC_BE,
        magicEndian: 'big',
        version: 9,
      };
    }
    if (b3 === 0x38) {
      return {
        magic: RPF8_MAGIC,
        rawMagic: RPF8_MAGIC_BE,
        magicEndian: 'big',
        version: 8,
      };
    }
    if (b3 === 0x37) {
      return {
        magic: RPF7_MAGIC,
        rawMagic: RPF7_MAGIC_BE,
        magicEndian: 'big',
        version: 7,
      };
    }
  }

  const hex = [b0, b1, b2, b3]
    .map((b) => b.toString(16).toUpperCase().padStart(2, '0'))
    .join(' ');
  throw new RpfFormatError(
    'INVALID_MAGIC',
    `Invalid RPF magic bytes: ${hex} (expected 9FPR/8FPR/7FPR or RPF9/RPF8/RPF7)`,
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

  const { magic, rawMagic, magicEndian, version } = detectMagicVersion(u8);

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
    rawMagic,
    magicEndian,
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
  if (!Number.isInteger(nameOffset) || nameOffset < 0) {
    throw new RpfFormatError('INVALID_TOC', `Invalid TOC nameOffset: ${nameOffset}`);
  }
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
  if (end >= namesBytes.byteLength) {
    throw new RpfFormatError(
      'INVALID_TOC',
      `Unterminated TOC name string at nameOffset ${nameOffset} (namesSize ${namesBytes.byteLength})`
    );
  }
  return utf8Decoder.decode(namesBytes.subarray(nameOffset, end));
}

function parseTocEntry(tocBytes, byteOffset, namesBytes, entryIndex = 0) {
  const u8 = toUint8Array(tocBytes, 'tocBytes');
  if (
    !Number.isInteger(byteOffset) ||
    byteOffset < 0 ||
    byteOffset + RPF9_ENTRY_STRIDE > u8.byteLength
  ) {
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

  const dataOffsetBig = isSectorOffset ? BigInt(w1) * BigInt(RPF_SECTOR_SIZE) : BigInt(w1);
  const uncompressedSize = w2;
  const compressedSize = w3;
  const storedSize = isCompressed ? compressedSize : uncompressedSize;

  if (isCompressed && compressedSize === 0 && uncompressedSize > 0) {
    throw new RpfFormatError(
      'INVALID_TOC',
      `Compressed TOC entry ${entryIndex} has compressedSize=0 with uncompressedSize=${uncompressedSize}`
    );
  }

  if (
    dataOffsetBig > MAX_SAFE_BIGINT ||
    dataOffsetBig + BigInt(storedSize) > MAX_SAFE_BIGINT ||
    dataOffsetBig + BigInt(uncompressedSize) > MAX_SAFE_BIGINT
  ) {
    throw new RpfFormatError(
      'SIZE_OVERFLOW',
      `TOC entry ${entryIndex} byte range overflows safe integer bounds`
    );
  }

  const dataOffset = Number(dataOffsetBig);

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
    isSectorOffset,
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

/**
 * Zero-allocation fast path for extracting the normalized lowercase lookup key
 * of a virtual RPF path. Returns `virtualPath` directly without splitting or
 * allocating when it is already canonical lowercase ASCII with no leading,
 * trailing, or duplicate slashes, backslashes, NUL bytes, or `.` / `..` segments.
 *
 * @param {string} virtualPath
 * @returns {string}
 */
function normalizePathKeyFast(virtualPath) {
  if (virtualPath === undefined || virtualPath === null || virtualPath === '') {
    return '';
  }
  if (typeof virtualPath !== 'string') {
    throw new RpfFormatError('INVALID_PATH', 'Virtual path must be a string');
  }
  const len = virtualPath.length;
  let segStart = 0;
  let isCanonical = true;
  for (let i = 0; i < len; i += 1) {
    const c = virtualPath.charCodeAt(i);
    if (c === 0) {
      throw new RpfFormatError('INVALID_PATH', `Virtual path contains NUL byte: ${virtualPath}`);
    }
    if (c === 0x2f) {
      const segLen = i - segStart;
      if (
        segLen === 0 ||
        (segLen === 1 && virtualPath.charCodeAt(segStart) === 0x2e) ||
        (segLen === 2 &&
          virtualPath.charCodeAt(segStart) === 0x2e &&
          virtualPath.charCodeAt(segStart + 1) === 0x2e)
      ) {
        isCanonical = false;
        break;
      }
      segStart = i + 1;
    } else if (c === 0x5c || (c >= 0x41 && c <= 0x5a) || c >= 0x80) {
      isCanonical = false;
      break;
    }
  }
  if (isCanonical) {
    const tailLen = len - segStart;
    if (
      tailLen === 0 ||
      (tailLen === 1 && virtualPath.charCodeAt(segStart) === 0x2e) ||
      (tailLen === 2 &&
        virtualPath.charCodeAt(segStart) === 0x2e &&
        virtualPath.charCodeAt(segStart + 1) === 0x2e)
    ) {
      isCanonical = false;
    }
  }
  if (isCanonical) {
    return virtualPath;
  }

  // Single-pass zero-intermediate-allocation ASCII canonicalization for non-canonical
  // mixed-case, backslash, or redundant-slash paths within scratch capacity.
  if (len <= ASCII_NORM_SCRATCH_CAPACITY) {
    let outLen = 0;
    let i = 0;
    let asciiOnly = true;
    while (i < len) {
      while (i < len) {
        const c = virtualPath.charCodeAt(i);
        if (c === 0) {
          throw new RpfFormatError('INVALID_PATH', `Virtual path contains NUL byte: ${virtualPath}`);
        }
        if (c !== 0x2f && c !== 0x5c) {
          break;
        }
        i += 1;
      }
      if (i >= len) {
        break;
      }
      const sStart = i;
      while (i < len) {
        const c = virtualPath.charCodeAt(i);
        if (c === 0) {
          throw new RpfFormatError('INVALID_PATH', `Virtual path contains NUL byte: ${virtualPath}`);
        }
        if (c === 0x2f || c === 0x5c) {
          break;
        }
        if (c >= 0x80) {
          asciiOnly = false;
          break;
        }
        i += 1;
      }
      if (!asciiOnly) {
        break;
      }
      const sLen = i - sStart;
      if (sLen === 1 && virtualPath.charCodeAt(sStart) === 0x2e) {
        continue;
      }
      if (
        sLen === 2 &&
        virtualPath.charCodeAt(sStart) === 0x2e &&
        virtualPath.charCodeAt(sStart + 1) === 0x2e
      ) {
        throw new RpfFormatError('INVALID_PATH', `Path traversal is not allowed: ${virtualPath}`);
      }
      if (outLen > 0) {
        ASCII_NORM_SCRATCH[outLen] = 0x2f;
        outLen += 1;
      }
      for (let j = sStart; j < i; j += 1) {
        let c = virtualPath.charCodeAt(j);
        if (c >= 0x41 && c <= 0x5a) {
          c += 0x20;
        }
        ASCII_NORM_SCRATCH[outLen] = c;
        outLen += 1;
      }
    }
    if (asciiOnly) {
      if (outLen === 0) {
        return '';
      }
      return String.fromCharCode.apply(null, ASCII_NORM_SCRATCH.subarray(0, outLen));
    }
  }

  return normalizeVirtualPath(virtualPath).key;
}

/**
 * Writes a TOC node's stat fields into `target` in-place (zero allocations when
 * `target` is a pre-allocated scratch object). Safely clears stale fields from
 * the opposite entry kind if `target` is reused across file/directory probes.
 */
function writeNodeStat(
  node,
  target,
  pathOverride = undefined,
  hasMount = false,
  mountId = undefined,
  mountPrefix = undefined
) {
  const entry = node.entry;
  const outPath = pathOverride !== undefined ? pathOverride : node.path;
  if (entry.kind === 'directory') {
    target.path = outPath;
    target.name = node.name;
    target.kind = 'directory';
    target.isDirectory = true;
    target.isFile = false;
    target.entryIndex = entry.index;
    target.firstChildIndex = entry.firstChildIndex;
    target.childCount = entry.childCount;
    target.flags = entry.flags;
    if (target.dataOffset !== undefined) target.dataOffset = undefined;
    if (target.size !== undefined) target.size = undefined;
    if (target.uncompressedSize !== undefined) target.uncompressedSize = undefined;
    if (target.compressedSize !== undefined) target.compressedSize = undefined;
    if (target.storedSize !== undefined) target.storedSize = undefined;
    if (target.isCompressed !== undefined) target.isCompressed = undefined;
    if (target.isResource !== undefined) target.isResource = undefined;
    if (target.isEncrypted !== undefined) target.isEncrypted = undefined;
    if (target.virtual !== undefined) target.virtual = undefined;
  } else {
    target.path = outPath;
    target.name = node.name;
    target.kind = 'file';
    target.isDirectory = false;
    target.isFile = true;
    target.entryIndex = entry.index;
    target.dataOffset = entry.dataOffset;
    target.size = entry.uncompressedSize;
    target.uncompressedSize = entry.uncompressedSize;
    target.compressedSize = entry.compressedSize;
    target.storedSize = entry.storedSize;
    target.flags = entry.flags;
    target.isCompressed = entry.isCompressed;
    target.isResource = entry.isResource;
    target.isEncrypted = entry.isEncrypted;
    if (target.firstChildIndex !== undefined) target.firstChildIndex = undefined;
    if (target.childCount !== undefined) target.childCount = undefined;
    if (target.virtual !== undefined) target.virtual = undefined;
  }
  if (hasMount) {
    target.mountId = mountId;
    target.mountPrefix = mountPrefix;
  }
  return target;
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
    const pathCacheCap = normalizeCacheCapacity(
      options.pathCacheCapacity !== undefined
        ? options.pathCacheCapacity
        : options.resolvedPathCacheSize,
      DEFAULT_PATH_CACHE_CAPACITY
    );
    const negativeCacheCap = normalizeCacheCapacity(
      options.maxNegativeEntries !== undefined
        ? options.maxNegativeEntries
        : options.negativeCacheCapacity !== undefined
          ? options.negativeCacheCapacity
          : options.negativeLookupCacheCapacity,
      DEFAULT_NEGATIVE_CACHE_CAPACITY
    );
    this._resolvedPathCache = new BoundedLruCache(pathCacheCap);
    this._negativeLookupCache = new BoundedNegativeLookupCache(negativeCacheCap);
    this._stats = {
      lookupHits: 0,
      lookupMisses: 0,
      pathIndexHits: 0,
      negativeLookupHits: 0,
      fastHashHits: 0,
      batchStatCalls: 0,
      batchStatHits: 0,
      tocCacheHits: options._fromTocCache ? 1 : 0,
      compressedEntriesRead: 0,
      rawEntriesRead: 0,
      bytesRead: 0,
    };

    const prebuilt = options._prebuiltIndex;
    if (
      prebuilt &&
      prebuilt.nodesByKey instanceof Map &&
      prebuilt.childrenByDirKey instanceof Map &&
      prebuilt.nodesByHash instanceof Map
    ) {
      this._nodesByKey = prebuilt.nodesByKey;
      this._childrenByDirKey = prebuilt.childrenByDirKey;
      this._nodesByHash = prebuilt.nodesByHash;
      this._hashCollisions = prebuilt.hashCollisions || 0;
    } else {
      this._nodesByKey = new Map();
      this._childrenByDirKey = new Map();
      this._nodesByHash = new Map();
      this._hashCollisions = 0;
      this._buildIndex();
    }
  }

  /**
   * Returns sanitized numeric-only lookup, hash-index, negative-cache, TOC-cache,
   * and read telemetry counters (zero filenames, archive paths, or host paths).
   *
   * @returns {{
   *   entryCount: number,
   *   indexedEntries: number,
   *   indexedEntryCount: number,
   *   hashIndexSize: number,
   *   hashCollisions: number,
   *   lookupHits: number,
   *   lookupMisses: number,
   *   pathIndexHits: number,
   *   negativeLookupHits: number,
   *   negativeLookupEntries: number,
   *   negativeLookupCapacity: number,
   *   negativeLookupEvictions: number,
   *   fastHashHits: number,
   *   batchStatCalls: number,
   *   batchStatHits: number,
   *   tocCacheHits: number,
   *   tocCacheMisses: number,
   *   resolvedPathCacheSize: number,
   *   resolvedPathCacheCapacity: number,
   *   resolvedPathCacheEvictions: number,
   *   compressedEntriesRead: number,
   *   rawEntriesRead: number,
   *   bytesRead: number
   * }}
   */
  getStats() {
    const indexedCount = this._nodesByKey.size;
    return {
      entryCount: this._entries.length,
      indexedEntries: indexedCount,
      indexedEntryCount: indexedCount,
      hashIndexSize: this._nodesByHash.size,
      hashCollisions: this._hashCollisions,
      lookupHits: this._stats.lookupHits,
      lookupMisses: this._stats.lookupMisses,
      pathIndexHits: this._stats.pathIndexHits,
      negativeLookupHits: this._stats.negativeLookupHits,
      negativeLookupEntries: this._negativeLookupCache.size,
      negativeLookupCapacity: this._negativeLookupCache.capacity,
      negativeLookupEvictions: this._negativeLookupCache.evictions,
      fastHashHits: this._stats.fastHashHits,
      batchStatCalls: this._stats.batchStatCalls,
      batchStatHits: this._stats.batchStatHits,
      tocCacheHits: this._stats.tocCacheHits,
      tocCacheMisses: this._resolvedPathCache.misses,
      resolvedPathCacheSize: this._resolvedPathCache.size,
      resolvedPathCacheCapacity: this._resolvedPathCache.capacity,
      resolvedPathCacheEvictions: this._resolvedPathCache.evictions,
      compressedEntriesRead: this._stats.compressedEntriesRead,
      rawEntriesRead: this._stats.rawEntriesRead,
      bytesRead: this._stats.bytesRead,
    };
  }

  resetStats() {
    this._stats.lookupHits = 0;
    this._stats.lookupMisses = 0;
    this._stats.pathIndexHits = 0;
    this._stats.negativeLookupHits = 0;
    this._stats.fastHashHits = 0;
    this._stats.batchStatCalls = 0;
    this._stats.batchStatHits = 0;
    this._stats.tocCacheHits = 0;
    this._stats.compressedEntriesRead = 0;
    this._stats.rawEntriesRead = 0;
    this._stats.bytesRead = 0;
    this._resolvedPathCache.resetStats();
    this._negativeLookupCache.resetStats();
  }

  clearPathCache() {
    this._resolvedPathCache.clear();
    this._negativeLookupCache.clear();
  }

  clearCaches() {
    this._resolvedPathCache.clear();
    this._negativeLookupCache.clear();
  }

  reset() {
    this.clearCaches();
    this.resetStats();
  }

  _insertHashIndex(hash, node) {
    const existing = this._nodesByHash.get(hash);
    if (existing === undefined) {
      this._nodesByHash.set(hash, node);
    } else if (Array.isArray(existing)) {
      existing.push(node);
      this._hashCollisions += 1;
    } else {
      this._nodesByHash.set(hash, [existing, node]);
      this._hashCollisions += 1;
    }
  }

  _buildIndex() {
    const entries = this._entries;
    if (entries.length === 0) {
      const rootHash = joaatHash('');
      const syntheticRoot = {
        path: '',
        key: '',
        name: '',
        pathHash: rootHash,
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
      this._insertHashIndex(rootHash, syntheticRoot);
      return;
    }

    const rootEntry = entries[0];
    if (rootEntry.kind !== 'directory') {
      throw new RpfFormatError('INVALID_TOC', 'TOC entry 0 must be the root directory');
    }

    const visited = new Uint8Array(entries.length);
    const rootHash = joaatHash('');
    const rootNode = {
      path: '',
      key: '',
      name: '',
      pathHash: rootHash,
      entry: rootEntry,
    };
    this._nodesByKey.set('', rootNode);
    this._insertHashIndex(rootHash, rootNode);
    visited[0] = 1;

    const stack = [rootNode];
    const headerTocNamesEndBig = BigInt(this.header.namesOffset) + BigInt(this.header.namesSize);
    const totalSizeBig = this.header.totalSize !== null ? BigInt(this.header.totalSize) : null;
    while (stack.length > 0) {
      const dirNode = stack.pop();
      const dirEntry = dirNode.entry;
      const children = [];
      this._childrenByDirKey.set(dirNode.key, children);

      if (dirEntry.childCount === 0) {
        continue;
      }

      const start = dirEntry.firstChildIndex;
      const childCount = dirEntry.childCount;
      if (
        !Number.isInteger(start) ||
        !Number.isInteger(childCount) ||
        start < 0 ||
        childCount < 0
      ) {
        throw new RpfFormatError(
          'INVALID_TOC',
          `Directory entry ${dirEntry.index} has invalid child bounds (${start}, ${childCount})`
        );
      }
      const endBig = BigInt(start) + BigInt(childCount);
      if (endBig > BigInt(entries.length)) {
        throw new RpfFormatError(
          'INVALID_TOC',
          `Directory entry ${dirEntry.index} child range [${start}..${endBig}) exceeds entryCount ${entries.length}`
        );
      }
      const end = Number(endBig);

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

        if (child.kind === 'file') {
          if (
            !Number.isSafeInteger(child.dataOffset) ||
            child.dataOffset < 0 ||
            !Number.isSafeInteger(child.storedSize) ||
            child.storedSize < 0 ||
            !Number.isSafeInteger(child.uncompressedSize) ||
            child.uncompressedSize < 0
          ) {
            throw new RpfFormatError(
              'SIZE_OVERFLOW',
              `File entry "${cleanName}" has invalid offset or size`
            );
          }
          const dataOffsetBig = BigInt(child.dataOffset);
          const storedSizeBig = BigInt(child.storedSize);
          const endByteBig = dataOffsetBig + storedSizeBig;
          if (endByteBig > MAX_SAFE_BIGINT) {
            throw new RpfFormatError(
              'SIZE_OVERFLOW',
              `File entry "${cleanName}" range overflows safe integer bounds`
            );
          }
          if (child.storedSize > 0 && dataOffsetBig < headerTocNamesEndBig) {
            throw new RpfFormatError(
              'INVALID_TOC',
              `File entry "${cleanName}" dataOffset ${child.dataOffset} overlaps archive header/TOC/names (${headerTocNamesEndBig})`
            );
          }
          if (totalSizeBig !== null && endByteBig > totalSizeBig) {
            throw new RpfFormatError(
              'TRUNCATED_ARCHIVE',
              `File entry "${cleanName}" range [${child.dataOffset}..${endByteBig}) exceeds archive size ${this.header.totalSize}`
            );
          }
        }

        const childPath = dirNode.path ? `${dirNode.path}/${cleanName}` : cleanName;
        const childKey = childPath.toLowerCase();
        if (this._nodesByKey.has(childKey)) {
          throw new RpfFormatError(
            'INVALID_TOC',
            `Duplicate entry path "${childKey}" detected at TOC entry ${idx}`
          );
        }
        const childHash = joaatHash(childKey);
        const childNode = {
          path: childPath,
          key: childKey,
          name: cleanName,
          pathHash: childHash,
          entry: child,
        };

        this._nodesByKey.set(childKey, childNode);
        this._insertHashIndex(childHash, childNode);
        children.push(childNode);

        if (child.kind === 'directory') {
          stack.push(childNode);
        }
      }
    }
  }

  /**
   * Opens an RPF archive via any async byte-range reader, fetching ONLY the header,
   * TOC entry table, and names table. When `options.tocCache` is supplied alongside
   * `options.tocCacheKey` or `options.archivePath`, cached TOC tables and pre-built
   * indexes are reused in O(1) with zero TOC range reads.
   *
   * @param {Function|object} rangeReader - `(offset, length) => Promise<Uint8Array>` or `{ read }`.
   * @param {number|bigint|null} [totalSize] - Optional total byte size of the archive.
   * @param {object} [options] - Optional `{ decryptBlock, decompressEntry, archivePath, tocCache, tocCacheKey, pathCacheCapacity }`.
   * @returns {Promise<RpfArchiveVfs>}
   */
  static async open(rangeReader, totalSize = null, options = {}) {
    const readRange = wrapRangeReader(rangeReader);
    const decryptBlock = typeof options.decryptBlock === 'function' ? options.decryptBlock : null;
    const tocCache =
      options && options.tocCache && typeof options.tocCache.get === 'function' && typeof options.tocCache.set === 'function'
        ? options.tocCache
        : null;

    let tocCacheKey = null;
    if (tocCache) {
      if (typeof options.tocCacheKey === 'string' && options.tocCacheKey.length > 0) {
        tocCacheKey = options.tocCacheKey;
      } else if (typeof options.archivePath === 'string' && options.archivePath.length > 0) {
        const sizeTag = totalSize === null || totalSize === undefined ? '' : String(totalSize);
        tocCacheKey = `${options.archivePath}\0${sizeTag}`;
      }
    }

    const forceReload = Boolean(options && (options.reload || options.forceReload));
    if (tocCache && tocCacheKey && !forceReload) {
      const cachedToc = tocCache.get(tocCacheKey);
      if (cachedToc) {
        if (cachedToc.header.isEncrypted && !decryptBlock) {
          throw new EncryptedArchiveError(
            cachedToc.header.encryptionTag,
            cachedToc.header.encryptionLabel
          );
        }
        return new RpfArchiveVfs(
          cachedToc.header,
          readRange,
          cachedToc.entries,
          Object.assign({}, options, {
            _prebuiltIndex: cachedToc,
            _fromTocCache: true,
          })
        );
      }
    }

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

    const vfs = new RpfArchiveVfs(header, readRange, entries, options);
    if (tocCache && tocCacheKey) {
      tocCache.set(tocCacheKey, {
        header,
        entries,
        nodesByKey: vfs._nodesByKey,
        childrenByDirKey: vfs._childrenByDirKey,
        nodesByHash: vfs._nodesByHash,
        hashCollisions: vfs._hashCollisions,
      });
    }
    return vfs;
  }

  /**
   * Opens an RPF archive backed by a PlayGta6RangeCache instance.
   *
   * @param {object} rangeCache - RangeCache instance exposing `read(relPath, offset, length)`.
   * @param {string} archivePath - Relative archive path (e.g. `'x64a.rpf'` or `'update/update.rpf'`).
   * @param {number|bigint|null} [totalSize] - Optional total archive byte size.
   * @param {object} [options] - Optional `{ decryptBlock, decompressEntry, tocCache, tocCacheKey, pathCacheCapacity }`.
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

  _resolveNode(virtualPath) {
    let rawKey;
    if (virtualPath === undefined || virtualPath === null) {
      rawKey = '';
    } else if (typeof virtualPath === 'string') {
      rawKey = virtualPath;
    } else {
      // Delegates non-string validation to normalizeVirtualPath (throws INVALID_PATH).
      normalizeVirtualPath(virtualPath);
      return null;
    }

    const cached = this._resolvedPathCache.getIfPresent(rawKey);
    if (cached !== undefined) {
      this._stats.tocCacheHits += 1;
      this._stats.pathIndexHits += 1;
      return cached;
    }

    if (this._negativeLookupCache.getByRaw(rawKey) !== null) {
      this._stats.negativeLookupHits += 1;
      this._stats.tocCacheHits += 1;
      return null;
    }

    const normKey = normalizePathKeyFast(rawKey);
    if (normKey !== rawKey && this._negativeLookupCache.getByNormalized(normKey, rawKey)) {
      this._stats.negativeLookupHits += 1;
      this._stats.tocCacheHits += 1;
      return null;
    }

    this._resolvedPathCache.recordMiss();
    const node = this._nodesByKey.get(normKey) || null;
    if (node !== null) {
      this._stats.pathIndexHits += 1;
      this._resolvedPathCache.set(rawKey, node);
      return node;
    }

    this._negativeLookupCache.set(normKey, rawKey);
    return null;
  }

  _lookupNodesByHash(pathHash) {
    const h = parseHashKey(pathHash);
    const bucket = this._nodesByHash.get(h);
    if (bucket === undefined) {
      return null;
    }
    return Array.isArray(bucket) ? bucket : [bucket];
  }

  resolve(virtualPath) {
    const node = this._resolveNode(virtualPath);
    if (!node) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    return {
      path: node.path,
      key: node.key,
      name: node.name,
      archivePath: node.path,
      stat: cloneNodeStat(node),
    };
  }

  lookupByPath(virtualPath) {
    const node = this._resolveNode(virtualPath);
    if (!node) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    return {
      path: node.path,
      key: node.key,
      name: node.name,
      archivePath: node.path,
      pathHash: node.pathHash,
      stat: cloneNodeStat(node),
    };
  }

  lookupByHash(pathHash) {
    const h = parseHashKey(pathHash);
    const bucket = this._nodesByHash.get(h);
    if (bucket === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    const isArr = Array.isArray(bucket);
    const node = isArr ? bucket[0] : bucket;
    return {
      path: node.path,
      key: node.key,
      name: node.name,
      archivePath: node.path,
      pathHash: node.pathHash,
      collisionCount: isArr ? bucket.length - 1 : 0,
      stat: cloneNodeStat(node),
    };
  }

  resolveByHash(pathHash) {
    return this.lookupByHash(pathHash);
  }

  /**
   * Zero-allocation O(1) hash lookup supporting uint32, BigInt u32, or hex string
   * hashes and an optional pre-allocated `out` scratch result object (reusing
  /**
   * Zero-allocation O(1) path lookup supporting an optional pre-allocated `out`
   * scratch object (reusing `out.stat` when present).
   *
   * @param {string} virtualPath
   * @param {object} [out]
   * @returns {object|null}
   */
  resolveByPathFast(virtualPath, out = undefined) {
    const node = this._resolveNode(virtualPath);
    if (!node) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    const target = out !== undefined && out !== null && typeof out === 'object' ? out : {};
    const statTarget =
      target.stat !== undefined && target.stat !== null && typeof target.stat === 'object'
        ? target.stat
        : {};
    writeNodeStat(node, statTarget);
    target.path = node.path;
    target.key = node.key;
    target.name = node.name;
    target.archivePath = node.path;
    target.pathHash = node.pathHash;
    target.stat = statTarget;
    return target;
  }

  /**
   * Zero-allocation O(1) hash lookup supporting uint32, BigInt u32, or hex string
   * hashes and an optional pre-allocated `out` scratch result object (reusing
   * `out.stat` when present). Preserves exact `resolveByHash()` return shape when
   * `out` is omitted.
   *
   * @param {number|bigint|string} hashU32OrBigInt
   * @param {object} [out]
   * @returns {object|null}
   */
  resolveByHashFast(hashU32OrBigInt, out = undefined) {
    const h = parseHashKey(hashU32OrBigInt);
    const bucket = this._nodesByHash.get(h);
    if (bucket === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    this._stats.fastHashHits += 1;
    let node;
    let collisionCount;
    if (Array.isArray(bucket)) {
      node = bucket[0];
      collisionCount = bucket.length - 1;
    } else {
      node = bucket;
      collisionCount = 0;
    }
    const target = out !== undefined && out !== null && typeof out === 'object' ? out : {};
    const statTarget =
      target.stat !== undefined && target.stat !== null && typeof target.stat === 'object'
        ? target.stat
        : {};
    writeNodeStat(node, statTarget);
    target.path = node.path;
    target.key = node.key;
    target.name = node.name;
    target.archivePath = node.path;
    target.pathHash = node.pathHash;
    target.collisionCount = collisionCount;
    target.stat = statTarget;
    return target;
  }

  _statByHashFast(h, out) {
    const bucket = this._nodesByHash.get(h);
    if (bucket === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    this._stats.fastHashHits += 1;
    const node = Array.isArray(bucket) ? bucket[0] : bucket;
    const target = out !== undefined && out !== null && typeof out === 'object' ? out : {};
    return writeNodeStat(node, target);
  }

  /**
   * Zero-allocation O(1) stat probe accepting either a virtual path string or a
   * 32-bit JOAAT hash (number, BigInt u32, or explicit `0x`-prefixed hex string),
   * optionally writing into a pre-allocated `out` scratch object. Preserves exact
   * `stat()` return shape when `out` is omitted.
   *
   * @param {string|number|bigint} pathOrHash
   * @param {object} [out]
   * @returns {object|null}
   */
  statEntryFast(pathOrHash, out = undefined) {
    if (typeof pathOrHash === 'number' || typeof pathOrHash === 'bigint') {
      return this._statByHashFast(parseHashKey(pathOrHash), out);
    }
    if (typeof pathOrHash === 'string') {
      const explicitHash = parseExplicitHexHash(pathOrHash);
      if (explicitHash >= 0) {
        return this._statByHashFast(explicitHash, out);
      }
    }
    const node = this._resolveNode(pathOrHash);
    if (!node) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    const target = out !== undefined && out !== null && typeof out === 'object' ? out : {};
    return writeNodeStat(node, target);
  }

  /**
   * Resolves a list (or iterable) of hot boot/menu RPF paths or hashes in a single
   * pass using O(1) index hits and the bounded negative-lookup LRU cache.
   *
   * @param {Iterable<string|number|bigint>} paths
   * @param {Array<object|null>} [out]
   * @returns {Array<object|null>}
   */
  batchStat(paths, out = undefined) {
    if (paths === null || paths === undefined) {
      this._stats.batchStatCalls += 1;
      if (Array.isArray(out)) {
        out.length = 0;
        return out;
      }
      return [];
    }
    const isArr = Array.isArray(paths);
    if (!isArr && typeof paths[Symbol.iterator] !== 'function') {
      throw new RpfFormatError(
        'INVALID_PATH',
        'batchStat() requires an array or iterable of virtual paths or hashes'
      );
    }
    this._stats.batchStatCalls += 1;
    const useOut = Array.isArray(out);
    if (isArr) {
      const len = paths.length;
      const results = useOut ? out : new Array(len);
      if (useOut) {
        results.length = len;
      }
      for (let i = 0; i < len; i += 1) {
        const slot =
          useOut && results[i] !== null && typeof results[i] === 'object' ? results[i] : undefined;
        const st = this.statEntryFast(paths[i], slot);
        if (st !== null) {
          this._stats.batchStatHits += 1;
        }
        results[i] = st;
      }
      return results;
    }
    const results = useOut ? out : [];
    if (useOut) {
      results.length = 0;
    }
    let idx = 0;
    for (const item of paths) {
      const st = this.statEntryFast(item);
      if (st !== null) {
        this._stats.batchStatHits += 1;
      }
      results[idx] = st;
      idx += 1;
    }
    return results;
  }

  lookupAllByHash(pathHash) {
    const nodes = this._lookupNodesByHash(pathHash);
    if (!nodes) {
      this._stats.lookupMisses += 1;
      return [];
    }
    this._stats.lookupHits += 1;
    return nodes.map((node) => ({
      path: node.path,
      key: node.key,
      name: node.name,
      archivePath: node.path,
      pathHash: node.pathHash,
      stat: cloneNodeStat(node),
    }));
  }

  hasHash(pathHash) {
    const h = parseHashKey(pathHash);
    const found = this._nodesByHash.has(h);
    if (found) {
      this._stats.lookupHits += 1;
    } else {
      this._stats.lookupMisses += 1;
    }
    return found;
  }

  exists(virtualPath) {
    let node;
    try {
      node = this._resolveNode(virtualPath);
    } catch {
      this._stats.lookupMisses += 1;
      return false;
    }
    if (node) {
      this._stats.lookupHits += 1;
      return true;
    }
    this._stats.lookupMisses += 1;
    return false;
  }

  stat(virtualPath) {
    const node = this._resolveNode(virtualPath);
    if (!node) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    return cloneNodeStat(node);
  }

  open(virtualPath) {
    const node = this._resolveNode(virtualPath);
    if (!node) {
      this._stats.lookupMisses += 1;
      throw new RpfFormatError('PATH_NOT_FOUND', `File not found in RPF archive: ${virtualPath}`);
    }
    this._stats.lookupHits += 1;
    if (node.entry.kind !== 'file') {
      throw new RpfFormatError('NOT_A_FILE', `Path is a directory, not a file: ${virtualPath}`);
    }
    const stat = cloneNodeStat(node);
    return {
      path: node.path,
      name: node.name,
      size: node.entry.uncompressedSize,
      stat,
      read: (offset = 0, length = undefined, options = {}) =>
        this._readNodeRange(node, offset, length, options),
      readAll: (options = {}) => this._readNodeRange(node, 0, undefined, options),
    };
  }

  listDir(dirPath = '') {
    const node = this._resolveNode(dirPath);
    if (!node) {
      this._stats.lookupMisses += 1;
      throw new RpfFormatError('PATH_NOT_FOUND', `Directory not found in RPF archive: ${dirPath}`);
    }
    this._stats.lookupHits += 1;
    if (node.entry.kind !== 'directory') {
      throw new RpfFormatError('NOT_A_DIRECTORY', `Path is not a directory: ${dirPath}`);
    }
    const children = this._childrenByDirKey.get(node.key) || [];
    return children.map((child) => cloneNodeStat(child));
  }

  async _readNodeRange(node, offset = 0, length = undefined, options = {}) {
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
      const out = decompressed.slice(offset, end);
      this._stats.compressedEntriesRead += 1;
      this._stats.bytesRead += out.byteLength;
      return out;
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

    const archiveOffsetBig = BigInt(entry.dataOffset) + BigInt(offset);
    const archiveEndBig = archiveOffsetBig + BigInt(clampedLength);
    if (archiveEndBig > MAX_SAFE_BIGINT) {
      throw new RpfFormatError(
        'SIZE_OVERFLOW',
        `File range read [${archiveOffsetBig}..${archiveEndBig}) overflows safe integer bounds`
      );
    }
    if (this.header.totalSize !== null && archiveEndBig > BigInt(this.header.totalSize)) {
      throw new RpfFormatError(
        'TRUNCATED_ARCHIVE',
        `File range read [${archiveOffsetBig}..${archiveEndBig}) exceeds archive size ${this.header.totalSize}`
      );
    }
    const archiveOffset = Number(archiveOffsetBig);
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

    this._stats.rawEntriesRead += 1;
    this._stats.bytesRead += bytes.byteLength;
    return bytes;
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
    const node = this._resolveNode(virtualPath);
    if (!node) {
      this._stats.lookupMisses += 1;
      throw new RpfFormatError('PATH_NOT_FOUND', `File not found in RPF archive: ${virtualPath}`);
    }
    this._stats.lookupHits += 1;
    if (node.entry.kind !== 'file') {
      throw new RpfFormatError('NOT_A_FILE', `Path is a directory, not a file: ${virtualPath}`);
    }
    return this._readNodeRange(node, offset, length, options);
  }

  async readFile(virtualPath, options = {}) {
    return this.readFileRange(virtualPath, 0, undefined, options);
  }

  async readEntry(virtualPath, offset = 0, length = undefined, options = {}) {
    if (offset !== null && typeof offset === 'object') {
      return this.readFileRange(virtualPath, 0, undefined, offset);
    }
    return this.readFileRange(virtualPath, offset, length, options);
  }

  async readEntryByHash(pathHash, offset = 0, length = undefined, options = {}) {
    let effOffset = offset;
    let effLength = length;
    let effOptions = options;
    if (offset !== null && typeof offset === 'object') {
      effOffset = 0;
      effLength = undefined;
      effOptions = offset;
    }
    const nodes = this._lookupNodesByHash(pathHash);
    if (!nodes) {
      this._stats.lookupMisses += 1;
      throw new RpfFormatError('PATH_NOT_FOUND', `Hash not found in RPF archive: ${pathHash}`);
    }
    let node = nodes[0];
    if (nodes.length > 1) {
      if (effOptions && typeof effOptions.expectedPath === 'string') {
        const expectedKey = normalizeVirtualPath(effOptions.expectedPath).key;
        const matched = nodes.find((candidate) => candidate.key === expectedKey);
        if (!matched) {
          this._stats.lookupMisses += 1;
          throw new RpfFormatError(
            'PATH_NOT_FOUND',
            `Expected path not found in hash collision bucket: ${effOptions.expectedPath}`
          );
        }
        node = matched;
      } else {
        throw new RpfFormatError(
          'HASH_COLLISION',
          `Ambiguous 32-bit hash ${pathHash} matched ${nodes.length} entries; supply expectedPath to disambiguate`,
          { matchCount: nodes.length }
        );
      }
    }
    this._stats.lookupHits += 1;
    if (node.entry.kind !== 'file') {
      throw new RpfFormatError('NOT_A_FILE', `Hash target is a directory, not a file: ${pathHash}`);
    }
    return this._readNodeRange(node, effOffset, effLength, effOptions);
  }
}

/**
 * Multi-archive virtual file system mount table with RAGE-style override priority
 * (later mounts override earlier mounts by default), O(1) normalized lowercase
 * path & 32-bit JOAAT hash lookup indexes, and bounded LRU TOC resolution caches.
 */
class RpfVfsMountTable {
  constructor(options = {}) {
    this._mounts = [];
    this._sortedMounts = [];
    this._pathIndex = new Map();
    this._hashIndex = new Map();
    this._hashCollisions = 0;
    this._virtualDirIndex = new Map();
    this._unindexedMounts = [];
    this._nextId = 1;
    this._nextOrder = 1;
    const pathCacheCap = normalizeCacheCapacity(
      options && options.pathCacheCapacity !== undefined
        ? options.pathCacheCapacity
        : options && options.resolvedPathCacheSize,
      DEFAULT_PATH_CACHE_CAPACITY
    );
    const negativeCacheCap = normalizeCacheCapacity(
      options && options.maxNegativeEntries !== undefined
        ? options.maxNegativeEntries
        : options && options.negativeCacheCapacity !== undefined
          ? options.negativeCacheCapacity
          : options && options.negativeLookupCacheCapacity,
      DEFAULT_NEGATIVE_CACHE_CAPACITY
    );
    const dirCacheCap = normalizeCacheCapacity(
      options && options.dirCacheCapacity,
      DEFAULT_DIR_CACHE_CAPACITY
    );
    this._resolvedPathCache = new BoundedLruCache(pathCacheCap);
    this._negativeLookupCache = new BoundedNegativeLookupCache(negativeCacheCap);
    this._dirListingCache = new BoundedLruCache(dirCacheCap);
    this._tocCache = options && options.tocCache ? options.tocCache : null;
    this._stats = {
      lookupHits: 0,
      lookupMisses: 0,
      pathIndexHits: 0,
      negativeLookupHits: 0,
      fastHashHits: 0,
      batchStatCalls: 0,
      batchStatHits: 0,
      tocCacheHits: 0,
      compressedEntriesRead: 0,
      rawEntriesRead: 0,
      bytesRead: 0,
    };
    this._seedRootVirtualDir();
  }

  _seedRootVirtualDir() {
    this._virtualDirIndex.set('', {
      path: '',
      name: '',
      kind: 'directory',
      isDirectory: true,
      isFile: false,
      virtual: true,
    });
  }

  /**
   * Returns sanitized numeric-only mount table lookup, hash-index, negative-cache,
   * TOC-cache, and read telemetry (zero filenames, mount prefixes, archive paths, or host paths).
   *
   * @returns {{
   *   mountedArchives: number,
   *   indexedEntries: number,
   *   indexedEntryCount: number,
   *   hashIndexSize: number,
   *   hashCollisions: number,
   *   lookupHits: number,
   *   lookupMisses: number,
   *   pathIndexHits: number,
   *   negativeLookupHits: number,
   *   negativeLookupEntries: number,
   *   negativeLookupCapacity: number,
   *   negativeLookupEvictions: number,
   *   fastHashHits: number,
   *   batchStatCalls: number,
   *   batchStatHits: number,
   *   tocCacheHits: number,
   *   tocCacheMisses: number,
   *   resolvedPathCacheSize: number,
   *   resolvedPathCacheCapacity: number,
   *   resolvedPathCacheEvictions: number,
   *   dirListingCacheSize: number,
   *   compressedEntriesRead: number,
   *   rawEntriesRead: number,
   *   bytesRead: number
   * }}
   */
  getStats() {
    const indexedCount = this._pathIndex.size;
    return {
      mountedArchives: this._mounts.length,
      indexedEntries: indexedCount,
      indexedEntryCount: indexedCount,
      hashIndexSize: this._hashIndex.size,
      hashCollisions: this._hashCollisions,
      lookupHits: this._stats.lookupHits,
      lookupMisses: this._stats.lookupMisses,
      pathIndexHits: this._stats.pathIndexHits,
      negativeLookupHits: this._stats.negativeLookupHits,
      negativeLookupEntries: this._negativeLookupCache.size,
      negativeLookupCapacity: this._negativeLookupCache.capacity,
      negativeLookupEvictions: this._negativeLookupCache.evictions,
      fastHashHits: this._stats.fastHashHits,
      batchStatCalls: this._stats.batchStatCalls,
      batchStatHits: this._stats.batchStatHits,
      tocCacheHits: this._stats.tocCacheHits,
      tocCacheMisses: this._resolvedPathCache.misses + this._dirListingCache.misses,
      resolvedPathCacheSize: this._resolvedPathCache.size,
      resolvedPathCacheCapacity: this._resolvedPathCache.capacity,
      resolvedPathCacheEvictions: this._resolvedPathCache.evictions,
      dirListingCacheSize: this._dirListingCache.size,
      compressedEntriesRead: this._stats.compressedEntriesRead,
      rawEntriesRead: this._stats.rawEntriesRead,
      bytesRead: this._stats.bytesRead,
    };
  }

  resetStats() {
    this._stats.lookupHits = 0;
    this._stats.lookupMisses = 0;
    this._stats.pathIndexHits = 0;
    this._stats.negativeLookupHits = 0;
    this._stats.fastHashHits = 0;
    this._stats.batchStatCalls = 0;
    this._stats.batchStatHits = 0;
    this._stats.tocCacheHits = 0;
    this._stats.compressedEntriesRead = 0;
    this._stats.rawEntriesRead = 0;
    this._stats.bytesRead = 0;
    this._resolvedPathCache.resetStats();
    this._negativeLookupCache.resetStats();
    this._dirListingCache.resetStats();
  }

  clearPathCache() {
    this._resolvedPathCache.clear();
    this._negativeLookupCache.clear();
    this._dirListingCache.clear();
  }

  clearCaches() {
    this._resolvedPathCache.clear();
    this._negativeLookupCache.clear();
    this._dirListingCache.clear();
  }

  reset() {
    this._mounts = [];
    this._rebuildIndex();
    this.resetStats();
  }

  _upsertHashIndex(pathHash, fullKey, record) {
    const existing = this._hashIndex.get(pathHash);
    if (existing === undefined) {
      this._hashIndex.set(pathHash, record);
      return;
    }
    if (!Array.isArray(existing)) {
      if (existing.fullKey === fullKey) {
        this._hashIndex.set(pathHash, record);
      } else {
        this._hashIndex.set(pathHash, [existing, record]);
        this._hashCollisions += 1;
      }
      return;
    }
    for (let i = 0; i < existing.length; i += 1) {
      if (existing[i].fullKey === fullKey) {
        existing[i] = record;
        return;
      }
    }
    existing.push(record);
    this._hashCollisions += 1;
  }

  _indexVirtualPrefix(mount) {
    for (let i = 0; i < mount.prefixSegments.length; i += 1) {
      const segPath = mount.prefixSegments.slice(0, i + 1).join('/');
      const segKey = segPath.toLowerCase();
      if (!this._virtualDirIndex.has(segKey)) {
        this._virtualDirIndex.set(segKey, {
          path: segPath,
          name: mount.prefixSegments[i],
          kind: 'directory',
          isDirectory: true,
          isFile: false,
          virtual: true,
        });
      }
    }
  }

  _indexMount(mount, overwrite = true) {
    const vfs = mount.vfs;
    if (vfs && vfs._nodesByKey instanceof Map) {
      for (const [nodeKey, node] of vfs._nodesByKey.entries()) {
        const fullKey = mount.prefixKey
          ? nodeKey
            ? `${mount.prefixKey}/${nodeKey}`
            : mount.prefixKey
          : nodeKey;
        if (!overwrite && this._pathIndex.has(fullKey)) {
          continue;
        }
        const fullPath = mount.prefix
          ? node.path
            ? `${mount.prefix}/${node.path}`
            : mount.prefix
          : node.path;
        const pathHash =
          !mount.prefixKey && typeof node.pathHash === 'number'
            ? node.pathHash
            : joaatHash(fullKey);
        const entryRecord = {
          mount,
          mountId: mount.id,
          mountPrefix: mount.prefix,
          vfs,
          archivePath: node.path,
          archiveKey: nodeKey,
          fullPath,
          fullKey,
          pathHash,
          node,
          stat: null,
          fromPathIndex: true,
        };
        this._pathIndex.set(fullKey, entryRecord);
        this._upsertHashIndex(pathHash, fullKey, entryRecord);
      }
      return;
    }

    // Fallback indexing for custom duck-typed VFS instances without _nodesByKey.
    try {
      const rootStat = vfs.stat('');
      if (!rootStat) {
        this._unindexedMounts.push(mount);
        return;
      }
      const visited = new Set(['']);
      const queue = [{ archivePath: '', archiveKey: '', stat: rootStat }];
      let qHead = 0;
      while (qHead < queue.length) {
        const item = queue[qHead];
        qHead += 1;
        const fullKey = mount.prefixKey
          ? item.archiveKey
            ? `${mount.prefixKey}/${item.archiveKey}`
            : mount.prefixKey
          : item.archiveKey;
        if (overwrite || !this._pathIndex.has(fullKey)) {
          const fullPath = mount.prefix
            ? item.archivePath
              ? `${mount.prefix}/${item.archivePath}`
              : mount.prefix
            : item.archivePath;
          const pathHash = joaatHash(fullKey);
          const entryRecord = {
            mount,
            mountId: mount.id,
            mountPrefix: mount.prefix,
            vfs,
            archivePath: item.archivePath,
            archiveKey: item.archiveKey,
            fullPath,
            fullKey,
            pathHash,
            node: null,
            stat: item.stat,
            fromPathIndex: true,
          };
          this._pathIndex.set(fullKey, entryRecord);
          this._upsertHashIndex(pathHash, fullKey, entryRecord);
        }
        if (item.stat.isDirectory) {
          const children = vfs.listDir(item.archivePath);
          for (const child of children) {
            const childNorm = normalizeVirtualPath(child.path);
            if (visited.has(childNorm.key)) continue;
            visited.add(childNorm.key);
            queue.push({ archivePath: child.path, archiveKey: childNorm.key, stat: child });
          }
        }
      }
    } catch {
      this._unindexedMounts.push(mount);
    }
  }

  _rebuildIndex() {
    this._pathIndex.clear();
    this._hashIndex.clear();
    this._hashCollisions = 0;
    this._virtualDirIndex.clear();
    this._resolvedPathCache.clear();
    this._negativeLookupCache.clear();
    this._dirListingCache.clear();
    this._unindexedMounts = [];
    this._seedRootVirtualDir();
    this._sortedMounts = this._orderedMounts();
    for (const m of this._sortedMounts) {
      this._indexMount(m, false);
      this._indexVirtualPrefix(m);
    }
  }

  _writeMountedStat(indexed, target = {}) {
    if (indexed.node) {
      return writeNodeStat(
        indexed.node,
        target,
        indexed.fullPath,
        true,
        indexed.mountId,
        indexed.mountPrefix
      );
    }
    Object.assign(target, indexed.stat);
    target.path = indexed.fullPath;
    target.mountId = indexed.mountId;
    target.mountPrefix = indexed.mountPrefix;
    if (target.virtual !== undefined) target.virtual = undefined;
    return target;
  }

  _writeVirtualDirStat(virtualDir, target = {}) {
    target.path = virtualDir.path;
    target.name = virtualDir.name;
    target.kind = 'directory';
    target.isDirectory = true;
    target.isFile = false;
    target.virtual = true;
    if (target.entryIndex !== undefined) target.entryIndex = undefined;
    if (target.firstChildIndex !== undefined) target.firstChildIndex = undefined;
    if (target.childCount !== undefined) target.childCount = undefined;
    if (target.dataOffset !== undefined) target.dataOffset = undefined;
    if (target.size !== undefined) target.size = undefined;
    if (target.uncompressedSize !== undefined) target.uncompressedSize = undefined;
    if (target.compressedSize !== undefined) target.compressedSize = undefined;
    if (target.storedSize !== undefined) target.storedSize = undefined;
    if (target.flags !== undefined) target.flags = undefined;
    if (target.isCompressed !== undefined) target.isCompressed = undefined;
    if (target.isResource !== undefined) target.isResource = undefined;
    if (target.isEncrypted !== undefined) target.isEncrypted = undefined;
    if (target.mountId !== undefined) target.mountId = undefined;
    if (target.mountPrefix !== undefined) target.mountPrefix = undefined;
    return target;
  }

  _buildMountedStat(indexed) {
    return this._writeMountedStat(indexed, {});
  }

  _getEntryMeta(indexed) {
    if (indexed.node) {
      const entry = indexed.node.entry;
      return {
        isFile: entry.kind === 'file',
        isCompressed: Boolean(entry.isCompressed),
        uncompressedSize: entry.uncompressedSize || 0,
        storedSize: entry.storedSize || 0,
      };
    }
    const st = indexed.stat;
    return {
      isFile: Boolean(st && st.isFile),
      isCompressed: Boolean(st && st.isCompressed),
      uncompressedSize: st && Number.isFinite(st.uncompressedSize) ? st.uncompressedSize : (st && st.size) || 0,
      storedSize: st && Number.isFinite(st.storedSize) ? st.storedSize : (st && st.size) || 0,
    };
  }

  _lookupEntry(norm) {
    const indexed = this._pathIndex.get(norm.key);
    if (indexed) {
      return indexed;
    }
    if (this._unindexedMounts.length > 0) {
      for (const m of this._unindexedMounts) {
        const match = this._matchMount(m, norm.key, norm.path);
        if (!match.matched) continue;
        if (m.vfs.exists(match.relPath)) {
          const st = m.vfs.stat(match.relPath);
          const fullPath =
            m.prefix && st.path ? `${m.prefix}/${st.path}` : m.prefix || st.path;
          return {
            mount: m,
            mountId: m.id,
            mountPrefix: m.prefix,
            vfs: m.vfs,
            archivePath: st.path,
            archiveKey: match.relKey,
            fullPath,
            node: null,
            stat: st,
            fromPathIndex: false,
          };
        }
      }
    }
    return null;
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
    this._sortedMounts = this._orderedMounts();
    if (this._sortedMounts[0] === record) {
      this._indexMount(record, true);
      this._indexVirtualPrefix(record);
      this._resolvedPathCache.clear();
      this._negativeLookupCache.clear();
      this._dirListingCache.clear();
    } else {
      this._rebuildIndex();
    }
    return {
      id: record.id,
      prefix: record.prefix,
      priority: record.priority,
      order: record.order,
    };
  }

  reload(idOrVfs = undefined, newVfs = undefined) {
    if (idOrVfs === undefined || idOrVfs === null) {
      this._rebuildIndex();
      return true;
    }
    const target = this._mounts.find((m) => m.id === idOrVfs || m.vfs === idOrVfs);
    if (!target) {
      return false;
    }
    if (newVfs !== undefined) {
      if (
        !newVfs ||
        typeof newVfs.exists !== 'function' ||
        typeof newVfs.stat !== 'function' ||
        typeof newVfs.listDir !== 'function' ||
        typeof newVfs.readFileRange !== 'function'
      ) {
        throw new RpfFormatError(
          'INVALID_MOUNT',
          'reload() requires an RpfArchiveVfs-compatible instance'
        );
      }
      target.vfs = newVfs;
    }
    this._rebuildIndex();
    return true;
  }

  unmount(idOrVfs, options = {}) {
    const before = this._mounts.length;
    this._mounts = this._mounts.filter((m) => m.id !== idOrVfs && m.vfs !== idOrVfs);
    if (this._mounts.length === before) {
      return false;
    }
    this._rebuildIndex();
    if (this._mounts.length === 0 || (options && options.resetStats)) {
      this.resetStats();
    }
    return true;
  }

  _orderedMounts() {
    // Higher explicit priority wins; for equal priority, later mount (higher order) wins.
    return this._mounts
      .slice()
      .sort((a, b) => (b.priority !== a.priority ? b.priority - a.priority : b.order - a.order));
  }

  listMounts() {
    return this._sortedMounts.map((m) => ({
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
    return this._virtualDirIndex.has(pathKey);
  }

  _resolveRaw(virtualPath) {
    let rawKey;
    if (virtualPath === undefined || virtualPath === null) {
      rawKey = '';
    } else if (typeof virtualPath === 'string') {
      rawKey = virtualPath;
    } else {
      normalizeVirtualPath(virtualPath);
      return null;
    }

    const cached = this._resolvedPathCache.getIfPresent(rawKey);
    if (cached !== undefined) {
      this._stats.tocCacheHits += 1;
      if (cached.vfs !== undefined && cached.fromPathIndex !== false) {
        this._stats.pathIndexHits += 1;
      }
      return cached;
    }

    const canCacheNegatives = this._unindexedMounts.length === 0;
    if (canCacheNegatives) {
      if (this._negativeLookupCache.getByRaw(rawKey) !== null) {
        this._stats.negativeLookupHits += 1;
        this._stats.tocCacheHits += 1;
        return null;
      }
    }

    const normKey = normalizePathKeyFast(rawKey);
    if (
      canCacheNegatives &&
      normKey !== rawKey &&
      this._negativeLookupCache.getByNormalized(normKey, rawKey)
    ) {
      this._stats.negativeLookupHits += 1;
      this._stats.tocCacheHits += 1;
      return null;
    }

    this._resolvedPathCache.recordMiss();
    let indexed = this._pathIndex.get(normKey) || null;
    if (indexed !== null) {
      this._stats.pathIndexHits += 1;
      this._resolvedPathCache.set(rawKey, indexed);
      return indexed;
    }

    if (!canCacheNegatives) {
      const norm = normalizeVirtualPath(rawKey);
      indexed = this._lookupEntry(norm);
      if (indexed !== null) {
        this._resolvedPathCache.set(rawKey, indexed);
        return indexed;
      }
    }

    const virtualDir = this._virtualDirIndex.get(normKey) || null;
    if (virtualDir !== null) {
      if (canCacheNegatives) {
        this._resolvedPathCache.set(rawKey, virtualDir);
      }
      return virtualDir;
    }

    if (canCacheNegatives) {
      this._negativeLookupCache.set(normKey, rawKey);
    }
    return null;
  }

  _resolveCached(virtualPath) {
    const resolved = this._resolveRaw(virtualPath);
    if (!resolved) {
      return { key: '', indexed: null, virtualDir: null };
    }
    if (resolved.vfs !== undefined) {
      return { key: resolved.fullKey || '', indexed: resolved, virtualDir: null };
    }
    return { key: resolved.path ? resolved.path.toLowerCase() : '', indexed: null, virtualDir: resolved };
  }

  _lookupRecordsByHash(pathHash) {
    const h = parseHashKey(pathHash);
    const bucket = this._hashIndex.get(h);
    if (bucket === undefined) {
      return null;
    }
    return Array.isArray(bucket) ? bucket : [bucket];
  }

  resolve(virtualPath) {
    const resolved = this._resolveRaw(virtualPath);
    if (!resolved || resolved.vfs === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    return {
      mountId: resolved.mountId,
      mountPrefix: resolved.mountPrefix,
      vfs: resolved.vfs,
      archivePath: resolved.archivePath,
      stat: this._buildMountedStat(resolved),
    };
  }

  lookupByPath(virtualPath) {
    const resolved = this._resolveRaw(virtualPath);
    if (!resolved || resolved.vfs === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    return {
      mountId: resolved.mountId,
      mountPrefix: resolved.mountPrefix,
      vfs: resolved.vfs,
      archivePath: resolved.archivePath,
      pathHash: resolved.pathHash,
      stat: this._buildMountedStat(resolved),
    };
  }

  lookupByHash(pathHash) {
    const h = parseHashKey(pathHash);
    const bucket = this._hashIndex.get(h);
    if (bucket === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    const isArr = Array.isArray(bucket);
    const indexed = isArr ? bucket[0] : bucket;
    return {
      mountId: indexed.mountId,
      mountPrefix: indexed.mountPrefix,
      vfs: indexed.vfs,
      archivePath: indexed.archivePath,
      pathHash: indexed.pathHash,
      collisionCount: isArr ? bucket.length - 1 : 0,
      stat: this._buildMountedStat(indexed),
    };
  }

  resolveByHash(pathHash) {
    return this.lookupByHash(pathHash);
  }

  /**
   * Zero-allocation O(1) path lookup across mounted archives supporting an optional
   * pre-allocated `out` scratch result object (reusing `out.stat` when present).
   *
   * @param {string} virtualPath
   * @param {object} [out]
   * @returns {object|null}
   */
  resolveByPathFast(virtualPath, out = undefined) {
    const resolved = this._resolveRaw(virtualPath);
    if (!resolved || resolved.vfs === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    const target = out !== undefined && out !== null && typeof out === 'object' ? out : {};
    const statTarget =
      target.stat !== undefined && target.stat !== null && typeof target.stat === 'object'
        ? target.stat
        : {};
    this._writeMountedStat(resolved, statTarget);
    target.mountId = resolved.mountId;
    target.mountPrefix = resolved.mountPrefix;
    target.vfs = resolved.vfs;
    target.archivePath = resolved.archivePath;
    target.pathHash = resolved.pathHash;
    target.stat = statTarget;
    return target;
  }

  /**
   * Zero-allocation O(1) hash lookup across mounted archives supporting uint32,
   * BigInt u32, or hex string hashes and an optional pre-allocated `out` scratch
   * result object (reusing `out.stat` when present). Preserves exact
   * `resolveByHash()` return shape when `out` is omitted.
   *
   * @param {number|bigint|string} hashU32OrBigInt
   * @param {object} [out]
   * @returns {object|null}
   */
  resolveByHashFast(hashU32OrBigInt, out = undefined) {
    const h = parseHashKey(hashU32OrBigInt);
    const bucket = this._hashIndex.get(h);
    if (bucket === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    this._stats.fastHashHits += 1;
    let indexed;
    let collisionCount;
    if (Array.isArray(bucket)) {
      indexed = bucket[0];
      collisionCount = bucket.length - 1;
    } else {
      indexed = bucket;
      collisionCount = 0;
    }
    const target = out !== undefined && out !== null && typeof out === 'object' ? out : {};
    const statTarget =
      target.stat !== undefined && target.stat !== null && typeof target.stat === 'object'
        ? target.stat
        : {};
    this._writeMountedStat(indexed, statTarget);
    target.mountId = indexed.mountId;
    target.mountPrefix = indexed.mountPrefix;
    target.vfs = indexed.vfs;
    target.archivePath = indexed.archivePath;
    target.pathHash = indexed.pathHash;
    target.collisionCount = collisionCount;
    target.stat = statTarget;
    return target;
  }

  _statByHashFast(h, out) {
    const bucket = this._hashIndex.get(h);
    if (bucket === undefined) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    this._stats.fastHashHits += 1;
    const indexed = Array.isArray(bucket) ? bucket[0] : bucket;
    const target = out !== undefined && out !== null && typeof out === 'object' ? out : {};
    return this._writeMountedStat(indexed, target);
  }

  /**
   * Zero-allocation O(1) stat probe across mounted archives accepting either a
   * virtual path string or a 32-bit JOAAT hash (number, BigInt u32, or explicit
   * `0x`-prefixed hex string), optionally writing into a pre-allocated `out`
   * scratch object. Preserves exact `stat()` return shape when `out` is omitted.
   *
   * @param {string|number|bigint} pathOrHash
   * @param {object} [out]
   * @returns {object|null}
   */
  statEntryFast(pathOrHash, out = undefined) {
    if (typeof pathOrHash === 'number' || typeof pathOrHash === 'bigint') {
      return this._statByHashFast(parseHashKey(pathOrHash), out);
    }
    if (typeof pathOrHash === 'string') {
      const explicitHash = parseExplicitHexHash(pathOrHash);
      if (explicitHash >= 0) {
        return this._statByHashFast(explicitHash, out);
      }
    }
    const resolved = this._resolveRaw(pathOrHash);
    if (!resolved) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    const target = out !== undefined && out !== null && typeof out === 'object' ? out : {};
    if (resolved.vfs !== undefined) {
      return this._writeMountedStat(resolved, target);
    }
    return this._writeVirtualDirStat(resolved, target);
  }

  /**
   * Resolves a list (or iterable) of hot boot/menu RPF paths or hashes across
   * mounted archives in a single pass using O(1) index hits and the bounded
   * negative-lookup LRU cache.
   *
   * @param {Iterable<string|number|bigint>} paths
   * @param {Array<object|null>} [out]
   * @returns {Array<object|null>}
   */
  batchStat(paths, out = undefined) {
    if (paths === null || paths === undefined) {
      this._stats.batchStatCalls += 1;
      if (Array.isArray(out)) {
        out.length = 0;
        return out;
      }
      return [];
    }
    const isArr = Array.isArray(paths);
    if (!isArr && typeof paths[Symbol.iterator] !== 'function') {
      throw new RpfFormatError(
        'INVALID_PATH',
        'batchStat() requires an array or iterable of virtual paths or hashes'
      );
    }
    this._stats.batchStatCalls += 1;
    const useOut = Array.isArray(out);
    if (isArr) {
      const len = paths.length;
      const results = useOut ? out : new Array(len);
      if (useOut) {
        results.length = len;
      }
      for (let i = 0; i < len; i += 1) {
        const slot =
          useOut && results[i] !== null && typeof results[i] === 'object' ? results[i] : undefined;
        const st = this.statEntryFast(paths[i], slot);
        if (st !== null) {
          this._stats.batchStatHits += 1;
        }
        results[i] = st;
      }
      return results;
    }
    const results = useOut ? out : [];
    if (useOut) {
      results.length = 0;
    }
    let idx = 0;
    for (const item of paths) {
      const st = this.statEntryFast(item);
      if (st !== null) {
        this._stats.batchStatHits += 1;
      }
      results[idx] = st;
      idx += 1;
    }
    return results;
  }

  lookupAllByHash(pathHash) {
    const records = this._lookupRecordsByHash(pathHash);
    if (!records) {
      this._stats.lookupMisses += 1;
      return [];
    }
    this._stats.lookupHits += 1;
    return records.map((indexed) => ({
      mountId: indexed.mountId,
      mountPrefix: indexed.mountPrefix,
      vfs: indexed.vfs,
      archivePath: indexed.archivePath,
      pathHash: indexed.pathHash,
      stat: this._buildMountedStat(indexed),
    }));
  }

  hasHash(pathHash) {
    const h = parseHashKey(pathHash);
    const found = this._hashIndex.has(h);
    if (found) {
      this._stats.lookupHits += 1;
    } else {
      this._stats.lookupMisses += 1;
    }
    return found;
  }

  exists(virtualPath) {
    let resolved;
    try {
      resolved = this._resolveRaw(virtualPath);
    } catch {
      this._stats.lookupMisses += 1;
      return false;
    }
    if (resolved !== null) {
      this._stats.lookupHits += 1;
      return true;
    }
    this._stats.lookupMisses += 1;
    return false;
  }

  stat(virtualPath) {
    const resolved = this._resolveRaw(virtualPath);
    if (!resolved) {
      this._stats.lookupMisses += 1;
      return null;
    }
    this._stats.lookupHits += 1;
    if (resolved.vfs !== undefined) {
      return this._buildMountedStat(resolved);
    }
    return Object.assign({}, resolved);
  }

  open(virtualPath) {
    const resolved = this._resolveRaw(virtualPath);
    if (!resolved || resolved.vfs === undefined) {
      this._stats.lookupMisses += 1;
      throw new RpfFormatError('PATH_NOT_FOUND', `File not found in mount table: ${virtualPath}`);
    }
    this._stats.lookupHits += 1;
    const stat = this._buildMountedStat(resolved);
    if (!stat.isFile) {
      throw new RpfFormatError('NOT_A_FILE', `Path is a directory, not a file: ${virtualPath}`);
    }
    return {
      path: stat.path,
      name: stat.name,
      size: stat.size,
      mountId: resolved.mountId,
      mountPrefix: resolved.mountPrefix,
      archivePath: resolved.archivePath,
      stat,
      read: (offset = 0, length = undefined, options = {}) =>
        this._readIndexedRange(resolved, offset, length, options),
      readAll: (options = {}) => this._readIndexedRange(resolved, 0, undefined, options),
    };
  }

  listDir(dirPath = '') {
    const dirKey = normalizePathKeyFast(dirPath);
    const cachedDir = this._dirListingCache.get(dirKey);
    if (cachedDir !== undefined) {
      this._stats.tocCacheHits += 1;
      return cachedDir.map((entry) => Object.assign({}, entry));
    }

    const norm = normalizeVirtualPath(dirPath);

    const ordered = this._sortedMounts;

    const winning = this._lookupEntry(norm);
    if (winning && this._getEntryMeta(winning).isFile) {
      throw new RpfFormatError('NOT_A_DIRECTORY', `Path is not a directory: ${dirPath}`);
    }

    const byNameKey = new Map();
    let foundAnyDir = norm.key === '';

    // Add children from matching mounts in highest-to-lowest priority order so higher
    // priority mounts win on name collisions.
    for (const m of ordered) {
      const match = this._matchMount(m, norm.key, norm.path);
      if (match.matched) {
        let dirEntries = null;
        if (m.vfs._nodesByKey instanceof Map && m.vfs._childrenByDirKey instanceof Map) {
          const dirNode = m.vfs._nodesByKey.get(match.relKey);
          if (dirNode && dirNode.entry.kind === 'directory') {
            foundAnyDir = true;
            const children = m.vfs._childrenByDirKey.get(dirNode.key) || [];
            dirEntries = children.map((child) => cloneNodeStat(child));
          }
        } else if (m.vfs.exists(match.relPath)) {
          const st = m.vfs.stat(match.relPath);
          if (st && st.isDirectory) {
            foundAnyDir = true;
            dirEntries = m.vfs.listDir(match.relPath);
          }
        }

        if (dirEntries) {
          for (const entry of dirEntries) {
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

    const sortedEntries = Array.from(byNameKey.values()).sort((a, b) =>
      a.name.toLowerCase().localeCompare(b.name.toLowerCase())
    );
    if (this._unindexedMounts.length === 0) {
      this._dirListingCache.set(norm.key, sortedEntries);
    }
    return sortedEntries.map((entry) => Object.assign({}, entry));
  }

  async _readIndexedRange(indexed, offset = 0, length = undefined, options = {}) {
    const meta = this._getEntryMeta(indexed);
    const readRaw = Boolean(options && options.raw);
    const isCompressedRead = meta.isCompressed && !readRaw;
    const effectiveSize = isCompressedRead
      ? meta.uncompressedSize
      : readRaw
        ? meta.storedSize
        : meta.uncompressedSize;

    const useDirectNode =
      indexed.node &&
      typeof indexed.vfs._readNodeRange === 'function' &&
      indexed.vfs.readFileRange === RpfArchiveVfs.prototype.readFileRange;

    const bytes = useDirectNode
      ? await indexed.vfs._readNodeRange(indexed.node, offset, length, options)
      : await indexed.vfs.readFileRange(indexed.archivePath, offset, length, options);

    if (offset < effectiveSize && length !== 0) {
      if (isCompressedRead) {
        this._stats.compressedEntriesRead += 1;
      } else {
        this._stats.rawEntriesRead += 1;
      }
      this._stats.bytesRead += bytes.byteLength;
    }
    return bytes;
  }

  async readFileRange(virtualPath, offset = 0, length = undefined, options = {}) {
    const resolved = this._resolveRaw(virtualPath);
    if (!resolved || resolved.vfs === undefined) {
      this._stats.lookupMisses += 1;
      throw new RpfFormatError('PATH_NOT_FOUND', `File not found in mount table: ${virtualPath}`);
    }
    this._stats.lookupHits += 1;
    if (!this._getEntryMeta(resolved).isFile) {
      throw new RpfFormatError('NOT_A_FILE', `Path is a directory, not a file: ${virtualPath}`);
    }
    return this._readIndexedRange(resolved, offset, length, options);
  }

  async readFile(virtualPath, options = {}) {
    return this.readFileRange(virtualPath, 0, undefined, options);
  }

  async readEntry(virtualPath, offset = 0, length = undefined, options = {}) {
    if (offset !== null && typeof offset === 'object') {
      return this.readFileRange(virtualPath, 0, undefined, offset);
    }
    return this.readFileRange(virtualPath, offset, length, options);
  }

  async readEntryByHash(pathHash, offset = 0, length = undefined, options = {}) {
    let effOffset = offset;
    let effLength = length;
    let effOptions = options;
    if (offset !== null && typeof offset === 'object') {
      effOffset = 0;
      effLength = undefined;
      effOptions = offset;
    }
    const records = this._lookupRecordsByHash(pathHash);
    if (!records) {
      this._stats.lookupMisses += 1;
      throw new RpfFormatError('PATH_NOT_FOUND', `Hash not found in mount table: ${pathHash}`);
    }
    let indexed = records[0];
    if (records.length > 1) {
      if (effOptions && typeof effOptions.expectedPath === 'string') {
        const expectedKey = normalizeVirtualPath(effOptions.expectedPath).key;
        const matched = records.find((candidate) => candidate.fullKey === expectedKey);
        if (!matched) {
          this._stats.lookupMisses += 1;
          throw new RpfFormatError(
            'PATH_NOT_FOUND',
            `Expected path not found in mount hash collision bucket: ${effOptions.expectedPath}`
          );
        }
        indexed = matched;
      } else {
        throw new RpfFormatError(
          'HASH_COLLISION',
          `Ambiguous 32-bit hash ${pathHash} matched ${records.length} mount entries; supply expectedPath to disambiguate`,
          { matchCount: records.length }
        );
      }
    }
    this._stats.lookupHits += 1;
    if (!this._getEntryMeta(indexed).isFile) {
      throw new RpfFormatError('NOT_A_FILE', `Hash target is a directory, not a file: ${pathHash}`);
    }
    return this._readIndexedRange(indexed, effOffset, effLength, effOptions);
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
  if (
    magic !== RPF9_MAGIC &&
    magic !== RPF8_MAGIC &&
    magic !== RPF7_MAGIC &&
    magic !== RPF9_MAGIC_LE &&
    magic !== RPF8_MAGIC_LE &&
    magic !== RPF7_MAGIC_LE
  ) {
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
  let bfsHead = 0;
  while (bfsHead < bfsQueue.length) {
    const dir = bfsQueue[bfsHead];
    bfsHead += 1;
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
  RPF9_MAGIC_BE,
  RPF8_MAGIC_BE,
  RPF7_MAGIC_BE,
  RPF9_MAGIC_LE,
  RPF8_MAGIC_LE,
  RPF7_MAGIC_LE,
  RPF9_MAGIC_U32,
  RPF8_MAGIC_U32,
  RPF7_MAGIC_U32,
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
  DEFAULT_PATH_CACHE_CAPACITY,
  DEFAULT_NEGATIVE_CACHE_CAPACITY,
  DEFAULT_DIR_CACHE_CAPACITY,
  DEFAULT_TOC_CACHE_CAPACITY,
  RpfFormatError,
  EncryptedArchiveError,
  classifyEncryption,
  parseRpfHeader,
  parseTocEntry,
  normalizeVirtualPath,
  normalizePathKeyFast,
  joaatHash,
  hashVirtualPath,
  RpfTocCache,
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
