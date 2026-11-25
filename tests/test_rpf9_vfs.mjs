import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const rpfMod = require('../web/workers/rpf9_vfs.js');
const rangeCacheMod = require('../web/workers/range_cache.js');

const {
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
  DEFAULT_PATH_CACHE_CAPACITY,
  DEFAULT_DIR_CACHE_CAPACITY,
  DEFAULT_TOC_CACHE_CAPACITY,
  RpfFormatError,
  EncryptedArchiveError,
  classifyEncryption,
  parseRpfHeader,
  normalizeVirtualPath,
  joaatHash,
  hashVirtualPath,
  RpfTocCache,
  RpfArchiveVfs,
  RpfVfsMountTable,
  buildSyntheticRpf9Buffer,
} = rpfMod;

const { RangeCache } = rangeCacheMod;

const decoder = new TextDecoder('utf-8');
const encoder = new TextEncoder();

function makeHeaderBytes({
  magic = RPF9_MAGIC,
  entryCount = 0,
  namesSize = 0,
  encryptionTag = RPF_ENCRYPTION_NONE,
  metadataSize = 0,
  flags = 0,
} = {}) {
  const buf = new Uint8Array(RPF9_HEADER_SIZE);
  const view = new DataView(buf.buffer);
  for (let i = 0; i < 4; i += 1) {
    buf[i] = magic.charCodeAt(i);
  }
  view.setUint32(4, entryCount >>> 0, true);
  view.setUint32(8, namesSize >>> 0, true);
  view.setUint32(12, encryptionTag >>> 0, true);
  view.setUint32(16, metadataSize >>> 0, true);
  view.setUint32(20, flags >>> 0, true);
  return buf;
}

function makePatternBytes(length, seed = 11) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (i * 37 + seed) & 0xff;
  }
  return out;
}

function createTrackingRangeReader(buffer) {
  const reads = [];
  let totalBytesRead = 0;

  async function readRange(offset, length) {
    reads.push({ offset, length });
    if (offset >= buffer.byteLength || length <= 0) {
      return new Uint8Array(0);
    }
    const end = Math.min(buffer.byteLength, offset + length);
    const slice = buffer.slice(offset, end);
    totalBytesRead += slice.byteLength;
    return slice;
  }

  return {
    readRange,
    reads,
    getTotalBytesRead: () => totalBytesRead,
  };
}

function createMockHttpRangeFetch(filesMap) {
  const requests = [];
  let totalBytesServed = 0;

  async function fetchImpl(url, init = {}) {
    const rangeHeader = (init.headers && (init.headers.Range || init.headers.range)) || null;
    const relPath = url.replace(/^.*\/data\//, '');
    requests.push({ url, relPath, rangeHeader });

    const data = filesMap.get(relPath);
    if (!data) {
      return {
        ok: false,
        status: 404,
        headers: new Map(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    assert.ok(rangeHeader, 'Expected HTTP Range header for RangeCache fetch');
    const match = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
    assert.ok(match, `Unexpected Range header: ${rangeHeader}`);
    const reqStart = Number(match[1]);
    const reqEnd = Number(match[2]);

    if (reqStart >= data.byteLength) {
      return {
        ok: false,
        status: 416,
        headers: new Map([['Content-Range', `bytes */${data.byteLength}`]]),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const actualEnd = Math.min(reqEnd, data.byteLength - 1);
    const slice = data.slice(reqStart, actualEnd + 1);
    totalBytesServed += slice.byteLength;

    return {
      ok: true,
      status: 206,
      headers: new Map([
        ['Content-Range', `bytes ${reqStart}-${actualEnd}/${data.byteLength}`],
        ['Content-Length', String(slice.byteLength)],
      ]),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  }

  return {
    fetchImpl,
    requests,
    getTotalBytesServed: () => totalBytesServed,
  };
}

test('parseRpfHeader parses RPF9, RPF8, and RPF7 headers and computes TOC/names offsets', () => {
  for (const [magic, version] of [
    [RPF9_MAGIC, 9],
    [RPF8_MAGIC, 8],
    [RPF7_MAGIC, 7],
  ]) {
    const raw = makeHeaderBytes({
      magic,
      entryCount: 128,
      namesSize: 4096,
      encryptionTag: RPF_ENCRYPTION_OPEN,
      metadataSize: 1024,
      flags: 0x00000003,
    });
    const totalSize = 10 * 1024 * 1024;
    const parsed = parseRpfHeader(raw, totalSize);

    assert.equal(parsed.magic, magic);
    assert.equal(parsed.version, version);
    assert.equal(parsed.headerSize, RPF9_HEADER_SIZE);
    assert.equal(parsed.entryCount, 128);
    assert.equal(parsed.namesSize, 4096);
    assert.equal(parsed.encryptionTag, RPF_ENCRYPTION_OPEN);
    assert.equal(parsed.encryptionLabel, 'open');
    assert.equal(parsed.isEncrypted, false);
    assert.equal(parsed.metadataSize, 1024);
    assert.equal(parsed.flags, 0x00000003);
    assert.equal(parsed.totalSize, totalSize);
    assert.equal(parsed.tocOffset, RPF9_HEADER_SIZE);
    assert.equal(parsed.tocBytes, 128 * RPF9_ENTRY_STRIDE);
    assert.equal(parsed.namesOffset, RPF9_HEADER_SIZE + 128 * RPF9_ENTRY_STRIDE);
    assert.equal(parsed.metadataOffset, RPF9_HEADER_SIZE + 128 * RPF9_ENTRY_STRIDE + 4096);
    assert.equal(parsed.minIndexBytes, RPF9_HEADER_SIZE + 128 * RPF9_ENTRY_STRIDE + 4096 + 1024);
  }

  // Also verify subarray with non-zero byteOffset and omitted totalSize.
  const padded = new Uint8Array(40);
  const hdr = makeHeaderBytes({ magic: RPF9_MAGIC, entryCount: 4, namesSize: 32 });
  padded.set(hdr, 12);
  const sub = padded.subarray(12, 12 + RPF9_HEADER_SIZE);
  const parsedSub = parseRpfHeader(sub);
  assert.equal(parsedSub.magic, RPF9_MAGIC);
  assert.equal(parsedSub.entryCount, 4);
  assert.equal(parsedSub.tocBytes, 64);
  assert.equal(parsedSub.namesOffset, 88);
  assert.equal(parsedSub.totalSize, null);
});

test('parseRpfHeader classifies all known and unknown encryption tags', () => {
  const cases = [
    [RPF_ENCRYPTION_NONE, 'none', false],
    [RPF_ENCRYPTION_OPEN, 'open', false],
    [RPF_ENCRYPTION_AES, 'aes', true],
    [RPF_ENCRYPTION_NG, 'ng', true],
    [0xdeadbeef, 'custom', true],
  ];

  for (const [tag, expectedLabel, expectedEncrypted] of cases) {
    const cls = classifyEncryption(tag);
    assert.equal(cls.tag, tag >>> 0);
    assert.equal(cls.label, expectedLabel);
    assert.equal(cls.isEncrypted, expectedEncrypted);

    const parsed = parseRpfHeader(makeHeaderBytes({ encryptionTag: tag }), 4096);
    assert.equal(parsed.encryptionLabel, expectedLabel);
    assert.equal(parsed.isEncrypted, expectedEncrypted);
  }
});

test('parseRpfHeader rejects invalid magic, truncated header, truncated archive, and invalid totalSize', () => {
  const badMagic = makeHeaderBytes({ magic: 'PK\x03\x04' });
  assert.throws(
    () => parseRpfHeader(badMagic, 1024),
    (err) => err instanceof RpfFormatError && err.code === 'INVALID_MAGIC'
  );

  const unsupportedRpf6 = makeHeaderBytes({ magic: 'RPF6' });
  assert.throws(
    () => parseRpfHeader(unsupportedRpf6, 1024),
    (err) => err instanceof RpfFormatError && err.code === 'INVALID_MAGIC'
  );

  assert.throws(
    () => parseRpfHeader(encoder.encode('RPF'), 3),
    (err) => err instanceof RpfFormatError && err.code === 'TRUNCATED_HEADER'
  );

  assert.throws(
    () => parseRpfHeader(encoder.encode('RPF9\x01\x00\x00\x00'), 8),
    (err) => err instanceof RpfFormatError && err.code === 'TRUNCATED_HEADER'
  );

  // Header requires 24 + 100*16 + 512 = 2136 bytes, but archive is only 1000 bytes.
  const truncatedArchive = makeHeaderBytes({ entryCount: 100, namesSize: 512 });
  assert.throws(
    () => parseRpfHeader(truncatedArchive, 1000),
    (err) =>
      err instanceof RpfFormatError &&
      err.code === 'TRUNCATED_ARCHIVE' &&
      err.details.requiredBytes === 2136 &&
      err.details.actualBytes === 1000
  );

  assert.throws(
    () => parseRpfHeader(makeHeaderBytes(), -1),
    (err) => err instanceof RpfFormatError && err.code === 'SIZE_OVERFLOW'
  );

  assert.throws(
    () => parseRpfHeader(makeHeaderBytes(), Number.MAX_SAFE_INTEGER + 100),
    (err) => err instanceof RpfFormatError && err.code === 'SIZE_OVERFLOW'
  );
});

test('RpfArchiveVfs streams ONLY header + TOC + requested sub-file ranges over tracking reader', async () => {
  const hugeUntouched = makePatternBytes(65536, 99);
  const hudBytes = makePatternBytes(2048, 7);
  const metaText = '<?xml version="1.0"?><LevelMeta name="ViceCity"/>';

  const archive = buildSyntheticRpf9Buffer(
    [
      { path: 'x64/textures/hud.ytd', data: hudBytes, isResource: true },
      { path: 'x64/textures/unused_huge.ytd', data: hugeUntouched },
      { path: 'common/data/levels.meta', data: metaText },
      { path: 'common/data/empty.dat', data: new Uint8Array(0) },
    ],
    { metadataSize: 64, flags: 0x12 }
  );

  const tracker = createTrackingRangeReader(archive);
  const vfs = await RpfArchiveVfs.open(tracker.readRange, archive.byteLength);

  // Verify open() fetched ONLY header (24 B) + TOC + names (not metadata or 64 KiB file payloads!).
  const expectedIndexBytes = RPF9_HEADER_SIZE + vfs.header.tocBytes + vfs.header.namesSize;
  assert.equal(tracker.reads.length, 2);
  assert.deepEqual(tracker.reads[0], { offset: 0, length: RPF9_HEADER_SIZE });
  assert.deepEqual(tracker.reads[1], {
    offset: RPF9_HEADER_SIZE,
    length: vfs.header.tocBytes + vfs.header.namesSize,
  });
  assert.equal(tracker.getTotalBytesRead(), expectedIndexBytes);
  assert.ok(tracker.getTotalBytesRead() < 1024, 'open() must never download the 64 KiB archive body');

  // Verify directory traversal, stat, and case-insensitive / backslash path normalization.
  assert.equal(vfs.exists(''), true);
  assert.equal(vfs.exists('/'), true);
  assert.equal(vfs.exists('\\'), true);
  assert.equal(vfs.exists('x64/textures/hud.ytd'), true);
  assert.equal(vfs.exists('/X64\\TEXTURES//HUD.YTD'), true);
  assert.equal(vfs.exists('./x64/./textures/hud.ytd'), true);
  assert.equal(vfs.exists('x64/textures/missing.ytd'), false);
  assert.equal(vfs.exists('../escape.ytd'), false);

  assert.throws(
    () => normalizeVirtualPath('x64/../secret'),
    (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
  );

  const rootEntries = vfs.listDir('');
  assert.deepEqual(
    rootEntries.map((e) => `${e.name}:${e.kind}`),
    ['common:directory', 'x64:directory']
  );

  const texEntries = vfs.listDir('\\X64\\Textures\\');
  assert.deepEqual(
    texEntries.map((e) => `${e.name}:${e.kind}`),
    ['hud.ytd:file', 'unused_huge.ytd:file']
  );

  const hudStat = vfs.stat('/X64\\textures\\HUD.YTD');
  assert.ok(hudStat);
  assert.equal(hudStat.path, 'x64/textures/hud.ytd');
  assert.equal(hudStat.isFile, true);
  assert.equal(hudStat.isDirectory, false);
  assert.equal(hudStat.size, hudBytes.byteLength);
  assert.equal(hudStat.uncompressedSize, hudBytes.byteLength);
  assert.equal(hudStat.compressedSize, 0);
  assert.equal(hudStat.isResource, true);
  assert.equal(hudStat.isCompressed, false);

  // Partial range read from middle of hud.ytd.
  const slice = await vfs.readFileRange('X64\\Textures\\HUD.YTD', 100, 64);
  assert.deepEqual(slice, hudBytes.subarray(100, 164));
  assert.equal(tracker.reads.length, 3);
  assert.deepEqual(tracker.reads[2], { offset: hudStat.dataOffset + 100, length: 64 });

  // EOF clamping: request 500 bytes starting 30 bytes before EOF -> only 30 bytes requested & returned.
  const tail = await vfs.readFileRange('x64/textures/hud.ytd', hudBytes.byteLength - 30, 500);
  assert.equal(tail.byteLength, 30);
  assert.deepEqual(tail, hudBytes.subarray(hudBytes.byteLength - 30));
  assert.equal(tracker.reads.length, 4);
  assert.deepEqual(tracker.reads[3], {
    offset: hudStat.dataOffset + hudBytes.byteLength - 30,
    length: 30,
  });

  // Reading at/past EOF or length 0 performs ZERO underlying range reads.
  const pastEof = await vfs.readFileRange('x64/textures/hud.ytd', hudBytes.byteLength + 10, 64);
  assert.equal(pastEof.byteLength, 0);
  const zeroLen = await vfs.readFileRange('x64/textures/hud.ytd', 0, 0);
  assert.equal(zeroLen.byteLength, 0);
  const emptyFile = await vfs.readFileRange('common/data/empty.dat', 0, 16);
  assert.equal(emptyFile.byteLength, 0);
  assert.equal(tracker.reads.length, 4, 'EOF and 0-length reads must not issue range requests');

  // Verify full read of small XML file works while 64 KiB untouched file was NEVER read.
  const metaOut = await vfs.readFile('common/data/levels.meta');
  assert.equal(decoder.decode(metaOut), metaText);
  assert.ok(
    tracker.getTotalBytesRead() < expectedIndexBytes + 256,
    `Expected < ${expectedIndexBytes + 256} total bytes read, got ${tracker.getTotalBytesRead()}`
  );

  // Error cases for wrong file/dir operations.
  assert.throws(
    () => vfs.listDir('x64/textures/hud.ytd'),
    (err) => err instanceof RpfFormatError && err.code === 'NOT_A_DIRECTORY'
  );
  await assert.rejects(
    () => vfs.readFileRange('x64/textures', 0, 16),
    (err) => err instanceof RpfFormatError && err.code === 'NOT_A_FILE'
  );
  await assert.rejects(
    () => vfs.readFileRange('x64/does_not_exist.ytd', 0, 16),
    (err) => err instanceof RpfFormatError && err.code === 'PATH_NOT_FOUND'
  );
});

test('RpfArchiveVfs.openFromRangeCache streams chunks via real RangeCache without fetching middle payload', async () => {
  const untouchedBig = makePatternBytes(40 * 1024, 55);
  const targetPayload = makePatternBytes(700, 123);

  const archive = buildSyntheticRpf9Buffer(
    [
      { path: 'assets/a_untouched_40k.bin', data: untouchedBig },
      { path: 'assets/z_target.bin', data: targetPayload },
    ],
    { sectorAlign: true }
  );

  const filesMap = new Map([['x64/world.rpf', archive]]);
  const http = createMockHttpRangeFetch(filesMap);
  const cache = new RangeCache({
    dataBase: '/data/',
    chunkSize: 1024,
    maxBytes: 16 * 1024 * 1024,
    fetchImpl: http.fetchImpl,
  });

  const vfs = await RpfArchiveVfs.openFromRangeCache(cache, 'x64/world.rpf', archive.byteLength);
  assert.equal(vfs.archivePath, 'x64/world.rpf');

  // Opening should only fetch chunk 0 (1 KiB) for header + TOC + names.
  assert.equal(http.requests.length, 1);
  assert.equal(http.requests[0].rangeHeader, 'bytes=0-1023');

  // Verify sector alignment flag decoded exact byte offset.
  const targetStat = vfs.stat('assets/z_target.bin');
  assert.equal(targetStat.dataOffset % RPF_SECTOR_SIZE, 0);
  assert.ok(targetStat.dataOffset > 40 * 1024);

  // Read 200 bytes from target file near the end of the archive.
  const part1 = await vfs.readFileRange('assets/z_target.bin', 100, 200);
  assert.deepEqual(part1, targetPayload.subarray(100, 300));

  const requestsAfterFirstRead = http.requests.length;
  assert.ok(
    http.getTotalBytesServed() <= 3 * 1024,
    `Expected <= 3 KiB transferred over HTTP 206, got ${http.getTotalBytesServed()}`
  );

  // Second read of same range hits RangeCache LRU with zero new network fetches.
  const part2 = await vfs.readFileRange('assets/z_target.bin', 100, 200);
  assert.deepEqual(part2, targetPayload.subarray(100, 300));
  assert.equal(http.requests.length, requestsAfterFirstRead);
  assert.ok(cache.getStats().hits > 0);
});

test('RpfArchiveVfs rejects encrypted archives without hook and supports pluggable decryptBlock / decompressEntry hooks', async () => {
  // 1. Encrypted archive without decryptBlock hook fails immediately with EncryptedArchiveError
  //    and never fetches TOC/names.
  for (const [tag, label] of [
    [RPF_ENCRYPTION_AES, 'aes'],
    [RPF_ENCRYPTION_NG, 'ng'],
    [0x13572468, 'custom'],
  ]) {
    const encArchive = buildSyntheticRpf9Buffer(
      [{ path: 'secret/config.xml', data: '<cfg/>' }],
      { encryptionTag: tag }
    );
    const tracker = createTrackingRangeReader(encArchive);

    await assert.rejects(
      () => RpfArchiveVfs.open(tracker.readRange, encArchive.byteLength),
      (err) =>
        err instanceof EncryptedArchiveError &&
        err.code === 'ENCRYPTED_ARCHIVE' &&
        err.encryptionTag === tag &&
        err.encryptionLabel === label
    );
    assert.equal(tracker.reads.length, 1, 'Must reject after reading only the 24-byte header');
  }

  // 2. OPEN encryption tag (0x4E45504F) is unencrypted and opens cleanly without hook.
  const openArchive = buildSyntheticRpf9Buffer(
    [{ path: 'open/readme.txt', data: 'open-ok' }],
    { encryptionTag: RPF_ENCRYPTION_OPEN }
  );
  const openVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(openArchive).readRange,
    openArchive.byteLength
  );
  assert.equal(openVfs.header.encryptionLabel, 'open');
  assert.equal(decoder.decode(await openVfs.readFile('open/readme.txt')), 'open-ok');

  // 3. Pluggable decryptBlock hook (synthetic XOR transform, no keys embedded in library).
  const xorByte = 0x5a;
  const xorTransform = (bytes) => {
    const out = new Uint8Array(bytes.byteLength);
    for (let i = 0; i < bytes.byteLength; i += 1) {
      out[i] = bytes[i] ^ xorByte;
    }
    return out;
  };

  const hookedArchive = buildSyntheticRpf9Buffer(
    [{ path: 'x64/scripts/main.ysc', data: 'SCRIPT_PAYLOAD_OK' }],
    { encryptionTag: RPF_ENCRYPTION_NG, encryptBlock: xorTransform }
  );
  const seenRegions = [];
  const hookedVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(hookedArchive).readRange,
    hookedArchive.byteLength,
    {
      decryptBlock: (bytes, ctx) => {
        seenRegions.push(ctx.region);
        return xorTransform(bytes);
      },
    }
  );
  const scriptBytes = await hookedVfs.readFileRange('x64/scripts/main.ysc', 7, 7);
  assert.equal(decoder.decode(scriptBytes), 'PAYLOAD');
  assert.deepEqual(seenRegions, ['toc', 'names', 'file']);

  // 4. Per-entry encrypted flag and compressed flag hooks.
  const specialArchive = buildSyntheticRpf9Buffer([
    { path: 'entry/enc.bin', data: encoder.encode('secret'), isEncrypted: true },
    {
      path: 'entry/comp.bin',
      data: encoder.encode('RAWCOMP'),
      isCompressed: true,
      uncompressedSize: 12,
      compressedSize: 7,
    },
  ]);

  const plainVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(specialArchive).readRange,
    specialArchive.byteLength
  );
  await assert.rejects(
    () => plainVfs.readFile('entry/enc.bin'),
    (err) => err instanceof EncryptedArchiveError
  );
  await assert.rejects(
    () => plainVfs.readFile('entry/comp.bin'),
    (err) => err instanceof RpfFormatError && err.code === 'COMPRESSED_ENTRY_UNSUPPORTED'
  );
  // Raw read of compressed entry returns stored compressed bytes directly.
  assert.equal(decoder.decode(await plainVfs.readFile('entry/comp.bin', { raw: true })), 'RAWCOMP');

  // With decompressEntry hook, sliced read returns decompressed slice.
  const decompVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(specialArchive).readRange,
    specialArchive.byteLength,
    {
      decompressEntry: () => encoder.encode('DECOMPRESSED'),
    }
  );
  assert.equal(decoder.decode(await decompVfs.readFileRange('entry/comp.bin', 2, 8)), 'COMPRESS');
});

test('RpfArchiveVfs supports RPF7/RPF8 archives, empty archives, and rejects malformed TOC cycles', async () => {
  // RPF7 & RPF8 round-trip.
  for (const magic of [RPF7_MAGIC, RPF8_MAGIC]) {
    const buf = buildSyntheticRpf9Buffer([{ path: 'dir/file.txt', data: `hello-${magic}` }], {
      magic,
    });
    const vfs = await RpfArchiveVfs.open(createTrackingRangeReader(buf).readRange, buf.byteLength);
    assert.equal(vfs.header.magic, magic);
    assert.equal(decoder.decode(await vfs.readFile('dir/file.txt')), `hello-${magic}`);
  }

  // Empty archive (0 files -> root directory only).
  const emptyArchive = buildSyntheticRpf9Buffer([]);
  const emptyVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(emptyArchive).readRange,
    emptyArchive.byteLength
  );
  assert.deepEqual(emptyVfs.listDir(''), []);
  assert.equal(emptyVfs.exists('anything'), false);

  // Malformed TOC: directory cycle (entry 1 points back to entry 0).
  const cyclic = new Uint8Array(RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE + 8);
  cyclic.set(makeHeaderBytes({ entryCount: 2, namesSize: 8 }), 0);
  const view = new DataView(cyclic.buffer);
  // Entry 0 (root): child [1..2)
  view.setUint32(RPF9_HEADER_SIZE + 0, (RPF_ENTRY_FLAG_DIRECTORY << 24) | 0, true);
  view.setUint32(RPF9_HEADER_SIZE + 4, 1, true);
  view.setUint32(RPF9_HEADER_SIZE + 8, 1, true);
  // Entry 1 (dir "loop"): child [0..1) -> cycle!
  view.setUint32(RPF9_HEADER_SIZE + 16, (RPF_ENTRY_FLAG_DIRECTORY << 24) | 1, true);
  view.setUint32(RPF9_HEADER_SIZE + 20, 0, true);
  view.setUint32(RPF9_HEADER_SIZE + 24, 1, true);
  const namesOff = RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE;
  cyclic.set(encoder.encode('\0loop\0'), namesOff);

  await assert.rejects(
    () => RpfArchiveVfs.open(createTrackingRangeReader(cyclic).readRange, cyclic.byteLength),
    (err) => err instanceof RpfFormatError && err.code === 'INVALID_TOC'
  );
});

test('RpfVfsMountTable mounts multiple archives with later-mount override precedence', async () => {
  const baseX64Buf = buildSyntheticRpf9Buffer([
    { path: 'textures/hud.ytd', data: 'HUD_V1_BASE' },
    { path: 'textures/map.ytd', data: 'MAP_V1_BASE' },
  ]);
  const patchX64Buf = buildSyntheticRpf9Buffer([
    { path: 'textures/hud.ytd', data: 'HUD_V2_PATCH' },
    { path: 'textures/dlc.ytd', data: 'DLC_V2_PATCH' },
  ]);
  const updateBuf = buildSyntheticRpf9Buffer([
    { path: 'common/data/dlclist.xml', data: '<dlclist>v1</dlclist>' },
  ]);
  const rootHotfixBuf = buildSyntheticRpf9Buffer([
    { path: 'x64/textures/hud.ytd', data: 'HUD_V3_HOTFIX' },
  ]);

  const baseX64 = await RpfArchiveVfs.open(
    createTrackingRangeReader(baseX64Buf).readRange,
    baseX64Buf.byteLength,
    { archivePath: 'x64a.rpf' }
  );
  const patchX64 = await RpfArchiveVfs.open(
    createTrackingRangeReader(patchX64Buf).readRange,
    patchX64Buf.byteLength,
    { archivePath: 'x64_patch.rpf' }
  );
  const updateVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(updateBuf).readRange,
    updateBuf.byteLength,
    { archivePath: 'update.rpf' }
  );
  const rootHotfix = await RpfArchiveVfs.open(
    createTrackingRangeReader(rootHotfixBuf).readRange,
    rootHotfixBuf.byteLength,
    { archivePath: 'hotfix.rpf' }
  );

  const table = new RpfVfsMountTable();
  const mBase = table.mount('x64/', baseX64);
  const mPatch = table.mount('x64/', patchX64);
  table.mount('update/', updateVfs);

  // patchX64 overrides baseX64 for hud.ytd; map.ytd falls through to baseX64.
  assert.equal(decoder.decode(await table.readFile('x64/textures/hud.ytd')), 'HUD_V2_PATCH');
  assert.equal(decoder.decode(await table.readFile('X64\\TEXTURES\\MAP.YTD')), 'MAP_V1_BASE');
  assert.equal(decoder.decode(await table.readFile('x64/textures/dlc.ytd')), 'DLC_V2_PATCH');
  assert.equal(
    decoder.decode(await table.readFile('update/common/data/dlclist.xml')),
    '<dlclist>v1</dlclist>'
  );

  // Merged directory listing combines entries across mounts with higher-priority winning.
  const mergedTextures = table.listDir('x64/textures');
  assert.deepEqual(
    mergedTextures.map((e) => `${e.name}:${e.mountId}`),
    [`dlc.ytd:${mPatch.id}`, `hud.ytd:${mPatch.id}`, `map.ytd:${mBase.id}`]
  );

  // Root listing exposes both 'update' and 'x64' mount prefixes.
  const rootDirs = table.listDir('');
  assert.deepEqual(
    rootDirs.map((e) => e.name),
    ['update', 'x64']
  );

  // Later root mount overrides x64/textures/hud.ytd globally.
  const mHotfix = table.mount('', rootHotfix);
  assert.equal(decoder.decode(await table.readFile('x64/textures/hud.ytd')), 'HUD_V3_HOTFIX');
  assert.equal(
    decoder.decode(await table.readFileRange('x64/textures/hud.ytd', 4, 9)),
    'V3_HOTFIX'
  );

  // Unmounting hotfix restores patchX64 precedence cleanly.
  assert.equal(table.unmount(mHotfix.id), true);
  assert.equal(table.unmount('nonexistent'), false);
  assert.equal(decoder.decode(await table.readFile('x64/textures/hud.ytd')), 'HUD_V2_PATCH');
  assert.equal(table.listMounts().length, 3);
});

test('parseRpfHeader accepts real little-endian 9FPR/8FPR/7FPR magics (0x52504639 LE) alongside BE magics', () => {
  // Construct real little-endian 0x52504639 magic via DataView u32 LE -> [0x39, 0x46, 0x50, 0x52] ("9FPR").
  const le9Raw = new Uint8Array(RPF9_HEADER_SIZE);
  const le9View = new DataView(le9Raw.buffer);
  le9View.setUint32(0, RPF9_MAGIC_U32, true);
  le9View.setUint32(4, 64, true);
  le9View.setUint32(8, 2048, true);
  le9View.setUint32(12, RPF_ENCRYPTION_OPEN, true);
  le9View.setUint32(16, 512, true);
  le9View.setUint32(20, 0x00000005, true);

  assert.deepEqual(Array.from(le9Raw.subarray(0, 4)), [0x39, 0x46, 0x50, 0x52]);

  const parsedLe9 = parseRpfHeader(le9Raw, 8 * 1024 * 1024);
  assert.equal(parsedLe9.magic, RPF9_MAGIC);
  assert.equal(parsedLe9.rawMagic, RPF9_MAGIC_LE);
  assert.equal(parsedLe9.magicEndian, 'little');
  assert.equal(parsedLe9.version, 9);
  assert.equal(parsedLe9.entryCount, 64);
  assert.equal(parsedLe9.namesSize, 2048);
  assert.equal(parsedLe9.encryptionLabel, 'open');
  assert.equal(parsedLe9.metadataSize, 512);
  assert.equal(parsedLe9.flags, 0x00000005);

  for (const [rawMagic, expectedMagic, expectedVersion, expectedEndian] of [
    [RPF9_MAGIC_LE, RPF9_MAGIC, 9, 'little'],
    [RPF8_MAGIC_LE, RPF8_MAGIC, 8, 'little'],
    [RPF7_MAGIC_LE, RPF7_MAGIC, 7, 'little'],
    [RPF9_MAGIC_BE, RPF9_MAGIC, 9, 'big'],
    [RPF8_MAGIC_BE, RPF8_MAGIC, 8, 'big'],
    [RPF7_MAGIC_BE, RPF7_MAGIC, 7, 'big'],
  ]) {
    const parsed = parseRpfHeader(makeHeaderBytes({ magic: rawMagic, entryCount: 2, namesSize: 16 }), 4096);
    assert.equal(parsed.magic, expectedMagic);
    assert.equal(parsed.rawMagic, rawMagic);
    assert.equal(parsed.magicEndian, expectedEndian);
    assert.equal(parsed.version, expectedVersion);
  }

  // Unsupported LE version "6FPR" and truncated "9FPR" header are cleanly rejected.
  assert.throws(
    () => parseRpfHeader(makeHeaderBytes({ magic: '6FPR' }), 1024),
    (err) => err instanceof RpfFormatError && err.code === 'INVALID_MAGIC'
  );
  assert.throws(
    () => parseRpfHeader(encoder.encode('9FPR\x01\x00\x00\x00'), 8),
    (err) => err instanceof RpfFormatError && err.code === 'TRUNCATED_HEADER'
  );
});

test('RpfArchiveVfs opens, resolves, and streams a synthetic RPF9 archive with little-endian "9FPR" (39 46 50 52) magic', async () => {
  const untouchedBig = makePatternBytes(32768, 41);
  const shaderBlob = makePatternBytes(1536, 17);
  const manifestText = '{"archive":"appdata0.rpf","magic":"9FPR"}';

  const archive = buildSyntheticRpf9Buffer(
    [
      { path: 'common/data/boot_manifest.json', data: manifestText },
      { path: 'x64/shaders/main_ps.dxil', data: shaderBlob, isResource: true },
      { path: 'x64/textures/unused_32k.ytd', data: untouchedBig },
    ],
    {
      magic: RPF9_MAGIC_LE,
      encryptionTag: RPF_ENCRYPTION_OPEN,
      metadataSize: 32,
      flags: 0x00000009,
    }
  );

  // Confirm raw on-disk header magic is exact little-endian 0x52504639 (39 46 50 52 = "9FPR").
  assert.deepEqual(Array.from(archive.subarray(0, 4)), [0x39, 0x46, 0x50, 0x52]);
  assert.equal(new DataView(archive.buffer, archive.byteOffset, 4).getUint32(0, true), RPF9_MAGIC_U32);

  const tracker = createTrackingRangeReader(archive);
  const vfs = await RpfArchiveVfs.open(tracker.readRange, archive.byteLength, {
    archivePath: 'appdata0.rpf',
  });

  // Header parse checks.
  assert.equal(vfs.header.magic, RPF9_MAGIC);
  assert.equal(vfs.header.rawMagic, RPF9_MAGIC_LE);
  assert.equal(vfs.header.magicEndian, 'little');
  assert.equal(vfs.header.version, 9);
  assert.equal(vfs.header.encryptionLabel, 'open');

  // TOC parse & resolution checks.
  assert.equal(tracker.reads.length, 2);
  assert.ok(tracker.getTotalBytesRead() < 1024, 'open() must fetch only header + TOC + names');
  assert.equal(vfs.exists('common/data/boot_manifest.json'), true);
  assert.equal(vfs.exists('/X64\\SHADERS\\MAIN_PS.DXIL'), true);
  assert.deepEqual(
    vfs.listDir('x64').map((e) => `${e.name}:${e.kind}`),
    ['shaders:directory', 'textures:directory']
  );

  const shaderStat = vfs.stat('x64/shaders/main_ps.dxil');
  assert.ok(shaderStat);
  assert.equal(shaderStat.isFile, true);
  assert.equal(shaderStat.isResource, true);
  assert.equal(shaderStat.size, shaderBlob.byteLength);

  // Streaming sub-range read checks.
  const partial = await vfs.readFileRange('x64/shaders/main_ps.dxil', 256, 128);
  assert.deepEqual(partial, shaderBlob.subarray(256, 384));
  assert.deepEqual(tracker.reads[2], { offset: shaderStat.dataOffset + 256, length: 128 });

  // Mount table integration & full file read.
  const table = new RpfVfsMountTable();
  table.mount('appdata/', vfs);
  const resolved = table.resolve('APPDATA\\Common\\Data\\boot_manifest.json');
  assert.ok(resolved);
  assert.equal(resolved.archivePath, 'common/data/boot_manifest.json');
  assert.equal(decoder.decode(await table.readFile('appdata/common/data/boot_manifest.json')), manifestText);

  // Confirm 32 KiB untouched payload was never fetched.
  assert.ok(tracker.getTotalBytesRead() < 2048);
});

test('web/workers/rpf9_vfs.js loads cleanly in a classic Web Worker global context (UMD)', () => {
  const src = fs.readFileSync(path.join(repoRoot, 'web', 'workers', 'rpf9_vfs.js'), 'utf8');
  const sandbox = {
    TextDecoder,
    TextEncoder,
    Uint8Array,
    ArrayBuffer,
    DataView,
    BigInt,
  };
  sandbox.globalThis = sandbox;
  vm.runInNewContext(src, sandbox, { filename: 'rpf9_vfs.js' });

  assert.ok(sandbox.PlayGta6Rpf9Vfs, 'Expected globalThis.PlayGta6Rpf9Vfs UMD export');
  assert.equal(sandbox.PlayGta6Rpf9Vfs.RPF9_MAGIC, 'RPF9');
  assert.equal(sandbox.PlayGta6Rpf9Vfs.RPF9_MAGIC_BE, 'RPF9');
  assert.equal(sandbox.PlayGta6Rpf9Vfs.RPF9_MAGIC_LE, '9FPR');
  assert.equal(sandbox.PlayGta6Rpf9Vfs.RPF9_MAGIC_U32, 0x52504639);
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.parseRpfHeader, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.joaatHash, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.hashVirtualPath, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.RpfTocCache, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.RpfArchiveVfs, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.RpfVfsMountTable, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.buildSyntheticRpf9Buffer, 'function');
  assert.equal(sandbox.PlayGta6Rpf9Vfs.DEFAULT_PATH_CACHE_CAPACITY, DEFAULT_PATH_CACHE_CAPACITY);
  assert.equal(sandbox.PlayGta6Rpf9Vfs.DEFAULT_TOC_CACHE_CAPACITY, DEFAULT_TOC_CACHE_CAPACITY);
});

test('RpfArchiveVfs O(1) normalized lookup, open/readEntry, traversal rejection, and sanitized getStats()', async () => {
  const rawPayload = encoder.encode('RAW_TEXTURE_BYTES_0123456789');
  const compStored = encoder.encode('CMP_DAT');
  const compExpanded = encoder.encode('DECOMPRESSED_ASSET_BODY');

  const archive = buildSyntheticRpf9Buffer([
    { path: 'x64/Textures/Hud_Main.ytd', data: rawPayload, isResource: true },
    {
      path: 'Common/Data/World_Cfg.bin',
      data: compStored,
      isCompressed: true,
      uncompressedSize: compExpanded.byteLength,
      compressedSize: compStored.byteLength,
    },
  ]);

  const vfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(archive).readRange,
    archive.byteLength,
    {
      archivePath: '/synthetic_host_dir/x64a.rpf',
      decompressEntry: () => compExpanded,
    }
  );

  // Initial stats: all counters zero, indexedEntries populated.
  const initStats = vfs.getStats();
  assert.equal(initStats.lookupHits, 0);
  assert.equal(initStats.lookupMisses, 0);
  assert.equal(initStats.compressedEntriesRead, 0);
  assert.equal(initStats.rawEntriesRead, 0);
  assert.equal(initStats.bytesRead, 0);
  assert.ok(initStats.indexedEntries >= 5);

  // Case-insensitive + backslash + leading/duplicate slash normalized lookups.
  const res1 = vfs.resolve('///X64\\\\textures///HUD_MAIN.YTD');
  assert.ok(res1);
  assert.equal(res1.path, 'x64/Textures/Hud_Main.ytd');
  assert.equal(res1.key, 'x64/textures/hud_main.ytd');
  assert.equal(res1.stat.size, rawPayload.byteLength);

  const st1 = vfs.stat('\\\\COMMON\\data\\.\\world_cfg.bin');
  assert.ok(st1);
  assert.equal(st1.isCompressed, true);

  assert.equal(vfs.resolve('x64/textures/missing.ytd'), null);
  assert.equal(vfs.stat('common/data/nope.bin'), null);

  // Traversal ('..') and NUL-byte ('\0') rejection.
  for (const badPath of [
    '../escape.ytd',
    'x64/../secret.ytd',
    'x64\\..\\secret.ytd',
    'x64/..',
    '..',
    'x64/textures/hud.ytd\0.png',
    '\0',
  ]) {
    assert.throws(
      () => vfs.resolve(badPath),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
    );
    assert.throws(
      () => vfs.stat(badPath),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
    );
    assert.throws(
      () => vfs.open(badPath),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
    );
    await assert.rejects(
      () => vfs.readEntry(badPath),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
    );
    assert.equal(vfs.exists(badPath), false);
  }

  // open() + handle.read() + handle.readAll() operate on normalized path.
  const handle = vfs.open('/X64\\TEXTURES\\hud_main.ytd');
  assert.equal(handle.path, 'x64/Textures/Hud_Main.ytd');
  assert.equal(handle.size, rawPayload.byteLength);
  const partialRaw = await handle.read(4, 7);
  assert.equal(decoder.decode(partialRaw), 'TEXTURE');

  // readEntry() for full raw, full compressed, ranged compressed, and compressed-as-raw.
  const fullComp = await vfs.readEntry('\\COMMON\\DATA\\WORLD_CFG.BIN');
  assert.equal(decoder.decode(fullComp), 'DECOMPRESSED_ASSET_BODY');

  const sliceComp = await vfs.readEntry('common/data/world_cfg.bin', 13, 5);
  assert.equal(decoder.decode(sliceComp), 'ASSET');

  const rawOfComp = await vfs.readEntry('COMMON\\data\\world_cfg.bin', { raw: true });
  assert.equal(decoder.decode(rawOfComp), 'CMP_DAT');

  // Zero-length / past-EOF reads do not increment entry read or byte counters.
  assert.equal((await vfs.readEntry('x64/textures/hud_main.ytd', 0, 0)).byteLength, 0);
  assert.equal((await vfs.readEntry('x64/textures/hud_main.ytd', 9999, 10)).byteLength, 0);

  // Verify exact counters:
  // Hits: resolve(1) + stat(1) + open(1) + readEntry x5(5) = 8
  // Misses: resolve(1) + stat(1) + exists(badPath x7 = 7) = 9
  // Raw reads: handle.read(7 B) + rawOfComp(7 B) = 2 reads, 14 B
  // Compressed reads: fullComp(23 B) + sliceComp(5 B) = 2 reads, 28 B
  // Total bytesRead = 14 + 28 = 42
  const stats = vfs.getStats();
  assert.equal(stats.lookupHits, 8);
  assert.equal(stats.lookupMisses, 9);
  assert.equal(stats.rawEntriesRead, 2);
  assert.equal(stats.compressedEntriesRead, 2);
  assert.equal(stats.bytesRead, 42);

  // Strict sanitization check: numeric-only fields, zero filenames or host paths leaked.
  for (const [key, val] of Object.entries(stats)) {
    assert.equal(typeof val, 'number', `RpfArchiveVfs.getStats().${key} must be numeric`);
    assert.ok(Number.isFinite(val) && val >= 0);
  }
  const serialized = JSON.stringify(stats);
  assert.equal(serialized.includes('synthetic_host_dir'), false);
  assert.equal(serialized.includes('Hud_Main'), false);
  assert.equal(serialized.includes('x64a.rpf'), false);
});

test('RpfVfsMountTable O(1) Map path index, priority/unmount cleanup, and sanitized read telemetry', async () => {
  const baseFiles = [];
  for (let i = 0; i < 64; i += 1) {
    baseFiles.push({ path: `Models/Prop_${i}.ydr`, data: `BASE_PROP_${i}` });
  }
  baseFiles.push({ path: 'Textures/Shared.ytd', data: 'SHARED_BASE_DATA' });
  baseFiles.push({
    path: 'Data/Compressed.bin',
    data: encoder.encode('ZBLOB'),
    isCompressed: true,
    uncompressedSize: 16,
    compressedSize: 5,
  });

  const baseBuf = buildSyntheticRpf9Buffer(baseFiles);
  const patchBuf = buildSyntheticRpf9Buffer([
    { path: 'Textures/Shared.ytd', data: 'SHARED_PATCH_V2!' },
    { path: 'Textures/OnlyPatch.ytd', data: 'PATCH_ONLY_BYTES' },
  ]);
  const lowPrioBuf = buildSyntheticRpf9Buffer([
    { path: 'Textures/Shared.ytd', data: 'LOW_PRIO_IGNORED' },
    { path: 'Textures/LowOnly.ytd', data: 'LOW_ONLY_BYTES' },
  ]);

  const baseVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(baseBuf).readRange,
    baseBuf.byteLength,
    {
      archivePath: '/synthetic_private/base.rpf',
      decompressEntry: () => encoder.encode('EXPANDED_16BYTES'),
    }
  );
  const patchVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(patchBuf).readRange,
    patchBuf.byteLength,
    { archivePath: '/synthetic_private/patch.rpf' }
  );
  const lowPrioVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(lowPrioBuf).readRange,
    lowPrioBuf.byteLength,
    { archivePath: '/synthetic_private/low.rpf' }
  );

  const table = new RpfVfsMountTable();
  const mBase = table.mount('x64/', baseVfs, { priority: 10 });
  const mPatch = table.mount('x64/', patchVfs, { priority: 20 });
  const mLow = table.mount('x64/', lowPrioVfs, { priority: 5 });

  assert.ok(table._pathIndex instanceof Map, 'Expected Map<normalizedLowerPath, entry> index');
  const afterMountStats = table.getStats();
  assert.equal(afterMountStats.mountedArchives, 3);
  assert.ok(afterMountStats.indexedEntries >= 68);

  // Verify O(1) index hits DO NOT call vfs.exists() or vfs.stat() on any mounted archive.
  let scannedVfsCalls = 0;
  for (const v of [baseVfs, patchVfs, lowPrioVfs]) {
    const origExists = v.exists.bind(v);
    const origStat = v.stat.bind(v);
    v.exists = (...args) => {
      scannedVfsCalls += 1;
      return origExists(...args);
    };
    v.stat = (...args) => {
      scannedVfsCalls += 1;
      return origStat(...args);
    };
  }

  // Case-insensitive + backslash + leading slash O(1) resolve, stat, open, readEntry.
  const resolvedShared = table.resolve('///X64\\\\TEXTURES//shared.ytd');
  assert.ok(resolvedShared);
  assert.equal(resolvedShared.mountId, mPatch.id);

  const lowStat = table.stat('\\x64\\textures\\LOWONLY.YTD');
  assert.ok(lowStat);
  assert.equal(lowStat.mountId, mLow.id);

  const propHandle = table.open('/X64/models\\PROP_42.YDR');
  assert.equal(decoder.decode(await propHandle.readAll()), 'BASE_PROP_42');

  const sharedBytes = await table.readEntry('X64\\Textures\\SHARED.YTD');
  assert.equal(decoder.decode(sharedBytes), 'SHARED_PATCH_V2!');

  const compBytes = await table.readEntry('/x64/data/COMPRESSED.BIN');
  assert.equal(decoder.decode(compBytes), 'EXPANDED_16BYTES');

  const compRawBytes = await table.readEntry('x64\\data\\compressed.bin', { raw: true });
  assert.equal(decoder.decode(compRawBytes), 'ZBLOB');

  // Lookup misses.
  assert.equal(table.resolve('x64/textures/missing_asset.ytd'), null);
  assert.equal(table.stat('x64/models/prop_999.ydr'), null);
  assert.equal(table.exists('x64/does/not/exist'), false);

  // Confirm ZERO linear archive scans occurred during all O(1) lookups above!
  assert.equal(scannedVfsCalls, 0, 'O(1) mount table lookups must not scan mounted VFS instances');

  // Path traversal and NUL-byte rejection on mount table.
  for (const badPath of ['../x64/textures/shared.ytd', 'x64/../secret', 'x64\\..\\secret', 'x64/a\0b']) {
    assert.throws(
      () => table.resolve(badPath),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
    );
    assert.throws(
      () => table.stat(badPath),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
    );
    assert.throws(
      () => table.open(badPath),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
    );
    await assert.rejects(
      () => table.readEntry(badPath),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
    );
    assert.equal(table.exists(badPath), false);
  }

  // Verify exact telemetry counters before unmount:
  // Hits: resolve(1) + stat(1) + open(1) + readEntry x3(3) = 6
  // Misses: resolve(1) + stat(1) + exists(1) + exists(badPath x4 = 4) = 7
  // Raw reads: propHandle.readAll(12 B) + sharedBytes(16 B) + compRawBytes(5 B) = 3 reads, 33 B
  // Compressed reads: compBytes(16 B) = 1 read, 16 B
  // Total bytesRead = 33 + 16 = 49
  const preUnmountStats = table.getStats();
  assert.equal(preUnmountStats.lookupHits, 6);
  assert.equal(preUnmountStats.lookupMisses, 7);
  assert.equal(preUnmountStats.rawEntriesRead, 3);
  assert.equal(preUnmountStats.compressedEntriesRead, 1);
  assert.equal(preUnmountStats.bytesRead, 49);

  // Sanitization check: strictly numeric values, zero paths/filenames exposed.
  for (const [key, val] of Object.entries(preUnmountStats)) {
    assert.equal(typeof val, 'number', `RpfVfsMountTable.getStats().${key} must be numeric`);
    assert.ok(Number.isFinite(val) && val >= 0);
  }
  const serializedTableStats = JSON.stringify(preUnmountStats);
  assert.equal(serializedTableStats.includes('private'), false);
  assert.equal(serializedTableStats.includes('patch.rpf'), false);
  assert.equal(serializedTableStats.includes('Shared.ytd'), false);

  // Partial unmount: unmounting mPatch removes OnlyPatch.ytd from index and restores mBase for Shared.ytd.
  assert.equal(table.unmount('nonexistent_mount'), false);
  assert.equal(table.unmount(mPatch.id), true);
  assert.equal(table.getStats().mountedArchives, 2);
  assert.equal(table.resolve('x64/textures/onlypatch.ytd'), null);
  assert.equal(table.getStats().lookupMisses, 8);
  const restoredShared = table.resolve('x64/textures/shared.ytd');
  assert.ok(restoredShared);
  assert.equal(restoredShared.mountId, mBase.id);
  assert.equal(decoder.decode(await table.readEntry('x64/textures/shared.ytd')), 'SHARED_BASE_DATA');

  // Unmount remaining archives -> index cleared and counters reset cleanly.
  assert.equal(table.unmount(mBase.id), true);
  assert.equal(table.unmount(mLow.id), true);
  const emptyStats = table.getStats();
  assert.equal(emptyStats.mountedArchives, 0);
  assert.equal(emptyStats.indexedEntries, 0);
  assert.equal(emptyStats.lookupHits, 0);
  assert.equal(emptyStats.lookupMisses, 0);
  assert.equal(emptyStats.compressedEntriesRead, 0);
  assert.equal(emptyStats.rawEntriesRead, 0);
  assert.equal(emptyStats.bytesRead, 0);

  // Lookup after full unmount misses cleanly and increments lookupMisses.
  assert.equal(table.resolve('x64/textures/shared.ytd'), null);
  assert.equal(table.getStats().lookupMisses, 1);
  assert.equal(table.getStats().lookupHits, 0);
});

test('joaatHash and hashVirtualPath compute deterministic 32-bit RAGE JOAAT hashes with normalization', () => {
  assert.equal(joaatHash(''), 0);
  assert.equal(joaatHash('a'), 0xca2e9442);
  assert.equal(joaatHash('A'), 0xca2e9442);
  assert.equal(joaatHash('x64/ui/fonts/menu.gfx'), joaatHash('X64/UI/FONTS/MENU.GFX'));

  // Non-ASCII UTF-8 path folds case deterministically and returns unsigned u32.
  const utf8A = joaatHash('ui/fonts/Café_Title.gfx');
  const utf8B = joaatHash('ui/fonts/café_title.gfx');
  assert.equal(utf8A, utf8B);
  assert.ok(Number.isInteger(utf8A) && utf8A >= 0 && utf8A <= 0xffffffff);

  // Virtual path normalization produces identical hashes for equivalent variants.
  const canonical = hashVirtualPath('x64/ui/fonts/menu_main.gfx');
  assert.equal(hashVirtualPath('///X64\\\\UI\\./Fonts//MENU_MAIN.GFX'), canonical);
  assert.equal(hashVirtualPath('\\x64/ui/fonts/menu_main.gfx/'), canonical);

  assert.throws(
    () => hashVirtualPath('../escape/menu.gfx'),
    (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
  );
  assert.throws(
    () => joaatHash(12345),
    (err) => err instanceof RpfFormatError && err.code === 'INVALID_PATH'
  );
});

test('RpfArchiveVfs hash-indexed lookup, collision disambiguation, and bounded LRU resolved-path caching', async () => {
  const archive = buildSyntheticRpf9Buffer([
    { path: 'x64/ui/fonts/hud_font.gfx', data: 'FONT_PAYLOAD_HUD' },
    { path: 'x64/ui/menus/pause_menu.gfx', data: 'MENU_PAYLOAD_PAUSE' },
    { path: 'x64/movies/intro_logo.bik', data: 'VIDEO_BIK_INTRO_01' },
    { path: 'x64/movies/attract_loop.bik', data: 'VIDEO_BIK_ATTRACT' },
    { path: 'x64/ui/menus/options_menu.gfx', data: 'MENU_PAYLOAD_OPTS' },
  ]);

  const vfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(archive).readRange,
    archive.byteLength,
    {
      archivePath: 'x64/ui_pack.rpf',
      pathCacheCapacity: 3,
    }
  );

  const initStats = vfs.getStats();
  assert.equal(initStats.indexedEntryCount, initStats.indexedEntries);
  assert.ok(initStats.indexedEntryCount >= 9);
  assert.equal(initStats.tocCacheHits, 0);
  assert.equal(initStats.resolvedPathCacheCapacity, 3);
  assert.equal(initStats.resolvedPathCacheSize, 0);
  assert.equal(initStats.resolvedPathCacheEvictions, 0);

  // Hash-indexed lookups by uint32 and hex string.
  const fontHash = hashVirtualPath('x64/ui/fonts/hud_font.gfx');
  const fontHex = `0x${fontHash.toString(16).padStart(8, '0')}`;
  assert.equal(vfs.hasHash(fontHash), true);
  assert.equal(vfs.hasHash(fontHex), true);
  assert.equal(vfs.hasHash(0xdeadbeef), false);

  const byHash = vfs.lookupByHash(fontHash);
  assert.ok(byHash);
  assert.equal(byHash.path, 'x64/ui/fonts/hud_font.gfx');
  assert.equal(byHash.pathHash, fontHash);
  assert.equal(byHash.collisionCount, 0);
  assert.equal(byHash.stat.size, 'FONT_PAYLOAD_HUD'.length);

  const byHashAlias = vfs.resolveByHash(fontHex);
  assert.equal(byHashAlias.path, byHash.path);

  const byPath = vfs.lookupByPath('\\X64\\UI\\Fonts\\HUD_FONT.GFX');
  assert.ok(byPath);
  assert.equal(byPath.pathHash, fontHash);

  assert.equal(decoder.decode(await vfs.readEntryByHash(fontHash)), 'FONT_PAYLOAD_HUD');
  assert.equal(decoder.decode(await vfs.readEntryByHash(fontHex, 5, 7)), 'PAYLOAD');
  assert.equal(vfs.lookupByHash(0xdeadbeef), null);
  assert.deepEqual(vfs.lookupAllByHash(0xdeadbeef), []);

  await assert.rejects(
    () => vfs.readEntryByHash(0xdeadbeef),
    (err) => err instanceof RpfFormatError && err.code === 'PATH_NOT_FOUND'
  );
  await assert.rejects(
    () => vfs.readEntryByHash(hashVirtualPath('x64/ui/fonts')),
    (err) => err instanceof RpfFormatError && err.code === 'NOT_A_FILE'
  );
  for (const badHash of [-1, 0x100000000, 1.5, 'not-a-hash', '0x123456789', {}]) {
    assert.throws(
      () => vfs.lookupByHash(badHash),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_HASH'
    );
  }

  // Verify bounded LRU resolved-path caching & eviction order (capacity = 3).
  vfs.clearPathCache();
  vfs.resetStats();

  const pA = 'x64/ui/fonts/hud_font.gfx';
  const pB = 'x64/ui/menus/pause_menu.gfx';
  const pC = 'x64/movies/intro_logo.bik';
  const pD = 'x64/movies/attract_loop.bik';

  assert.ok(vfs.resolve(pA)); // miss -> cache: [A]
  assert.ok(vfs.resolve(pB)); // miss -> cache: [A, B]
  assert.ok(vfs.resolve(pC)); // miss -> cache: [A, B, C]
  assert.equal(vfs.getStats().tocCacheHits, 0);
  assert.equal(vfs.getStats().resolvedPathCacheSize, 3);
  assert.equal(vfs.getStats().resolvedPathCacheEvictions, 0);

  // Touch A -> cache hit + promote A to MRU: [B, C, A].
  assert.ok(vfs.stat(pA));
  assert.equal(vfs.getStats().tocCacheHits, 1);

  // Insert D -> evicts B (LRU): [C, A, D].
  assert.ok(vfs.resolve(pD));
  assert.equal(vfs.getStats().resolvedPathCacheSize, 3);
  assert.equal(vfs.getStats().resolvedPathCacheEvictions, 1);

  // A is still cached (hit -> tocCacheHits = 2); B was evicted (miss -> evicts C, evictions = 2).
  assert.ok(vfs.resolve(pA));
  assert.equal(vfs.getStats().tocCacheHits, 2);
  assert.ok(vfs.resolve(pB));
  assert.equal(vfs.getStats().tocCacheHits, 2);
  assert.equal(vfs.getStats().resolvedPathCacheEvictions, 2);
  assert.equal(vfs.getStats().lookupHits, 7);

  // Negative caching: repeated misses hit resolved-path LRU while still counting lookupMisses.
  assert.equal(vfs.resolve('x64/ui/missing_screen.gfx'), null);
  const hitsBeforeNegRepeat = vfs.getStats().tocCacheHits;
  assert.equal(vfs.resolve('x64/ui/missing_screen.gfx'), null);
  assert.equal(vfs.getStats().tocCacheHits, hitsBeforeNegRepeat + 1);
  assert.equal(vfs.getStats().lookupMisses, 2);

  // Synthetic hash collision bucket disambiguation test.
  const nodeA = vfs._nodesByKey.get('x64/ui/fonts/hud_font.gfx');
  const nodeB = vfs._nodesByKey.get('x64/ui/menus/pause_menu.gfx');
  const syntheticCollHash = 0x11223344;
  vfs._nodesByHash.set(syntheticCollHash, [nodeA, nodeB]);
  const collAll = vfs.lookupAllByHash(syntheticCollHash);
  assert.equal(collAll.length, 2);
  assert.equal(vfs.lookupByHash(syntheticCollHash).collisionCount, 1);
  await assert.rejects(
    () => vfs.readEntryByHash(syntheticCollHash),
    (err) => err instanceof RpfFormatError && err.code === 'HASH_COLLISION'
  );
  const disambiguated = await vfs.readEntryByHash(syntheticCollHash, {
    expectedPath: 'X64\\UI\\Menus\\PAUSE_MENU.GFX',
  });
  assert.equal(decoder.decode(disambiguated), 'MENU_PAYLOAD_PAUSE');

  // Capacity 0 disables resolved-path LRU caching cleanly.
  const uncachedVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(archive).readRange,
    archive.byteLength,
    { pathCacheCapacity: 0 }
  );
  assert.ok(uncachedVfs.resolve(pA));
  assert.ok(uncachedVfs.resolve(pA));
  assert.equal(uncachedVfs.getStats().tocCacheHits, 0);
  assert.equal(uncachedVfs.getStats().resolvedPathCacheSize, 0);
  assert.equal(uncachedVfs.getStats().lookupHits, 2);
});

test('RpfTocCache caches parsed TOC + indexes across RpfArchiveVfs.open and enforces LRU capacity', async () => {
  const uiBuf = buildSyntheticRpf9Buffer([
    { path: 'ui/fonts/title.gfx', data: 'TITLE_FONT_V1' },
    { path: 'ui/menus/main.gfx', data: 'MAIN_MENU_V1' },
  ]);
  const videoBuf = buildSyntheticRpf9Buffer([
    { path: 'movies/splash.bik', data: 'SPLASH_BIK_STREAM' },
  ]);
  const audioBuf = buildSyntheticRpf9Buffer([
    { path: 'audio/ui_click.awc', data: 'AWC_CLICK_DATA' },
  ]);

  const tocCache = new RpfTocCache({ capacity: 2 });

  // First open of ui.rpf fetches header + TOC + names (2 range reads).
  const tracker1 = createTrackingRangeReader(uiBuf);
  const vfs1 = await RpfArchiveVfs.open(tracker1.readRange, uiBuf.byteLength, {
    archivePath: 'x64/ui.rpf',
    tocCache,
  });
  assert.equal(tracker1.reads.length, 2);
  assert.equal(tocCache.getStats().size, 1);
  assert.equal(tocCache.getStats().hits, 0);
  assert.equal(tocCache.getStats().misses, 1);
  assert.equal(vfs1.getStats().tocCacheHits, 0);

  // Second open of ui.rpf hits RpfTocCache with ZERO header/TOC range reads!
  const tracker2 = createTrackingRangeReader(uiBuf);
  const vfs2 = await RpfArchiveVfs.open(tracker2.readRange, uiBuf.byteLength, {
    archivePath: 'x64/ui.rpf',
    tocCache,
  });
  assert.equal(tracker2.reads.length, 0, 'TOC cache hit must perform zero header/TOC reads');
  assert.equal(tocCache.getStats().hits, 1);
  assert.equal(vfs2.getStats().tocCacheHits, 1);
  assert.equal(vfs2.getStats().indexedEntryCount, vfs1.getStats().indexedEntryCount);

  // Payload read on cached-TOC VFS still streams exact bytes through tracker2.
  const menuData = await vfs2.readFile('ui/menus/main.gfx');
  assert.equal(decoder.decode(menuData), 'MAIN_MENU_V1');
  assert.equal(tracker2.reads.length, 1);

  // Open 2nd and 3rd archives -> capacity 2 evicts LRU archive (x64/ui.rpf).
  await RpfArchiveVfs.open(createTrackingRangeReader(videoBuf).readRange, videoBuf.byteLength, {
    archivePath: 'x64/video.rpf',
    tocCache,
  });
  assert.equal(tocCache.getStats().size, 2);
  assert.equal(tocCache.getStats().evictions, 0);

  await RpfArchiveVfs.open(createTrackingRangeReader(audioBuf).readRange, audioBuf.byteLength, {
    archivePath: 'x64/audio.rpf',
    tocCache,
  });
  assert.equal(tocCache.getStats().size, 2);
  assert.equal(tocCache.getStats().evictions, 1);

  // Encrypted archive in RpfTocCache still enforces decryptBlock requirement on hit.
  const xorByte = 0x3c;
  const xorFn = (bytes) => {
    const out = new Uint8Array(bytes.byteLength);
    for (let i = 0; i < bytes.byteLength; i += 1) {
      out[i] = bytes[i] ^ xorByte;
    }
    return out;
  };
  const encBuf = buildSyntheticRpf9Buffer([{ path: 'ui/secret.gfx', data: 'SECRET_UI' }], {
    encryptionTag: RPF_ENCRYPTION_NG,
    encryptBlock: xorFn,
  });
  const encCache = new RpfTocCache(4);
  const encVfs1 = await RpfArchiveVfs.open(
    createTrackingRangeReader(encBuf).readRange,
    encBuf.byteLength,
    { archivePath: 'x64/enc.rpf', tocCache: encCache, decryptBlock: xorFn }
  );
  assert.equal(decoder.decode(await encVfs1.readFile('ui/secret.gfx')), 'SECRET_UI');

  await assert.rejects(
    () =>
      RpfArchiveVfs.open(createTrackingRangeReader(encBuf).readRange, encBuf.byteLength, {
        archivePath: 'x64/enc.rpf',
        tocCache: encCache,
      }),
    (err) => err instanceof EncryptedArchiveError
  );

  const encTracker2 = createTrackingRangeReader(encBuf);
  const encVfs2 = await RpfArchiveVfs.open(encTracker2.readRange, encBuf.byteLength, {
    archivePath: 'x64/enc.rpf',
    tocCache: encCache,
    decryptBlock: xorFn,
  });
  assert.equal(encTracker2.reads.length, 0);
  assert.equal(decoder.decode(await encVfs2.readFile('ui/secret.gfx')), 'SECRET_UI');

  // Verify RpfTocCache.getStats() is strictly numeric and sanitized.
  for (const [k, v] of Object.entries(tocCache.getStats())) {
    assert.equal(typeof v, 'number', `RpfTocCache.getStats().${k} must be numeric`);
    assert.ok(Number.isFinite(v) && v >= 0);
  }
});

test('RpfVfsMountTable hash index, LRU resolved-path & dir cache invalidation on mount/unmount', async () => {
  const baseBuf = buildSyntheticRpf9Buffer([
    { path: 'ui/fonts/main.gfx', data: 'BASE_MAIN_FONT' },
    { path: 'ui/menus/title.gfx', data: 'BASE_TITLE_MENU' },
  ]);
  const patchBuf = buildSyntheticRpf9Buffer([
    { path: 'ui/fonts/main.gfx', data: 'PATCH_MAIN_FONT_V2' },
    { path: 'ui/menus/dlc_banner.gfx', data: 'PATCH_DLC_BANNER' },
  ]);

  const baseVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(baseBuf).readRange,
    baseBuf.byteLength,
    { archivePath: 'base_ui.rpf' }
  );
  const patchVfs = await RpfArchiveVfs.open(
    createTrackingRangeReader(patchBuf).readRange,
    patchBuf.byteLength,
    { archivePath: 'patch_ui.rpf' }
  );

  const table = new RpfVfsMountTable({ pathCacheCapacity: 16, dirCacheCapacity: 8 });
  const mBase = table.mount('x64/', baseVfs, { priority: 10 });

  // Populate resolved-path LRU cache & dir listing cache.
  const mainFontPath = 'X64\\UI\\Fonts\\MAIN.GFX';
  const mainFontHash = hashVirtualPath('x64/ui/fonts/main.gfx');
  assert.equal(table.resolve(mainFontPath).mountId, mBase.id);
  assert.equal(table.resolve(mainFontPath).mountId, mBase.id);
  assert.equal(table.getStats().tocCacheHits, 1);

  const menusBefore = table.listDir('x64/ui/menus');
  assert.deepEqual(
    menusBefore.map((e) => e.name),
    ['title.gfx']
  );
  // Mutating returned array entry must not corrupt cached listing.
  menusBefore[0].name = 'MUTATED';
  const menusCached = table.listDir('x64/ui/menus');
  assert.deepEqual(
    menusCached.map((e) => e.name),
    ['title.gfx']
  );
  assert.equal(table.getStats().tocCacheHits, 2);

  // Hash lookup on mount table before patch.
  assert.equal(table.hasHash(mainFontHash), true);
  assert.equal(table.lookupByHash(mainFontHash).mountId, mBase.id);
  assert.equal(decoder.decode(await table.readEntryByHash(mainFontHash)), 'BASE_MAIN_FONT');

  // Mount higher-priority patch -> invalidates caches and replaces hash bucket in place (0 false collisions).
  const mPatch = table.mount('x64/', patchVfs, { priority: 20 });
  assert.equal(table.getStats().resolvedPathCacheSize, 0);
  assert.equal(table.getStats().dirListingCacheSize, 0);
  assert.equal(table.getStats().hashCollisions, 0);
  assert.equal(table.resolve(mainFontPath).mountId, mPatch.id);
  assert.equal(table.lookupByHash(mainFontHash).mountId, mPatch.id);
  assert.equal(decoder.decode(await table.readEntryByHash(mainFontHash)), 'PATCH_MAIN_FONT_V2');
  assert.deepEqual(
    table.listDir('x64/ui/menus').map((e) => e.name),
    ['dlc_banner.gfx', 'title.gfx']
  );

  // Unmount patch -> invalidates caches and cleanly restores base mount for path & hash lookups.
  assert.equal(table.unmount(mPatch.id), true);
  assert.equal(table.resolve(mainFontPath).mountId, mBase.id);
  assert.equal(table.lookupByHash(mainFontHash).mountId, mBase.id);
  assert.equal(decoder.decode(await table.readEntryByHash(mainFontHash)), 'BASE_MAIN_FONT');
  assert.equal(table.lookupByHash(hashVirtualPath('x64/ui/menus/dlc_banner.gfx')), null);
  assert.deepEqual(
    table.listDir('x64/ui/menus').map((e) => e.name),
    ['title.gfx']
  );
});

test('Large synthetic TOC (5,200+ entries) O(1) hash/path lookup correctness and benchmark', async () => {
  const categories = [
    { dir: 'x64/ui/menus', ext: 'gfx', count: 1400 },
    { dir: 'x64/ui/fonts', ext: 'gfx', count: 1000 },
    { dir: 'x64/movies', ext: 'bik', count: 1000 },
    { dir: 'x64/textures/hud', ext: 'ytd', count: 1000 },
    { dir: 'common/data/levels', ext: 'meta', count: 800 },
  ];

  const files = [];
  const allPaths = [];
  let seq = 0;
  for (const cat of categories) {
    for (let i = 0; i < cat.count; i += 1) {
      const bucket = String(i % 10).padStart(2, '0');
      const relPath = `${cat.dir}/b${bucket}/asset_${seq}.${cat.ext}`;
      const payload = `P${seq}`;
      files.push({ path: relPath, data: payload });
      allPaths.push({ path: relPath, payload });
      seq += 1;
    }
  }
  assert.equal(files.length, 5200);

  const archive = buildSyntheticRpf9Buffer(files);
  const tracker = createTrackingRangeReader(archive);
  const tocCache = new RpfTocCache({ capacity: 8 });
  const vfs = await RpfArchiveVfs.open(tracker.readRange, archive.byteLength, {
    archivePath: 'x64/large_ui_bundle.rpf',
    tocCache,
    pathCacheCapacity: 512,
  });

  const openStats = vfs.getStats();
  assert.ok(openStats.indexedEntryCount >= 5250, `Expected >= 5250 indexed entries, got ${openStats.indexedEntryCount}`);
  assert.equal(openStats.indexedEntries, openStats.indexedEntryCount);
  assert.ok(openStats.hashIndexSize >= 5250);
  assert.equal(tracker.reads.length, 2, 'open() must only fetch header + TOC/names even for 5,200+ files');

  // Spot-check first, middle, last, and mixed-case/backslash normalized entries + hash reads.
  const sampleIndices = [0, 1, 1399, 1400, 2599, 3400, 4400, 5198, 5199];
  for (const idx of sampleIndices) {
    const item = allPaths[idx];
    const noisyPath = `///${item.path.toUpperCase().replace(/\//g, '\\\\')}`;
    const res = vfs.lookupByPath(noisyPath);
    assert.ok(res, `Expected lookupByPath hit for ${item.path}`);
    assert.equal(res.path, item.path);

    const h = hashVirtualPath(item.path);
    assert.equal(res.pathHash, h);
    const byHash = vfs.lookupByHash(h);
    assert.ok(byHash, `Expected lookupByHash hit for ${item.path}`);
    assert.equal(byHash.path, item.path);

    const bytes = await vfs.readEntryByHash(h);
    assert.equal(decoder.decode(bytes), item.payload);
  }

  // Mount into RpfVfsMountTable and verify ZERO linear archive scans across 2,000+ lookups.
  const table = new RpfVfsMountTable({ pathCacheCapacity: 256 });
  table.mount('update/', vfs);
  assert.equal(table.getStats().indexedEntryCount, openStats.indexedEntryCount);

  let scannedVfsCalls = 0;
  const origExists = vfs.exists.bind(vfs);
  const origStat = vfs.stat.bind(vfs);
  vfs.exists = (...args) => {
    scannedVfsCalls += 1;
    return origExists(...args);
  };
  vfs.stat = (...args) => {
    scannedVfsCalls += 1;
    return origStat(...args);
  };

  // Hot working set of 100 menu/font/video assets queried 20x each (2,000 path lookups + 1,000 hash lookups).
  const hotPaths = [];
  const hotHashes = [];
  for (let i = 0; i < 100; i += 1) {
    const idx = (i * 51) % allPaths.length;
    const mountedPath = `update/${allPaths[idx].path}`;
    hotPaths.push(mountedPath);
    hotHashes.push(hashVirtualPath(mountedPath));
  }

  const startMs = Date.now();
  for (let round = 0; round < 20; round += 1) {
    for (let i = 0; i < hotPaths.length; i += 1) {
      const hit = table.resolve(hotPaths[i]);
      assert.ok(hit);
    }
  }
  for (let round = 0; round < 10; round += 1) {
    for (let i = 0; i < hotHashes.length; i += 1) {
      const hit = table.lookupByHash(hotHashes[i]);
      assert.ok(hit);
    }
  }
  const elapsedMs = Date.now() - startMs;

  assert.equal(scannedVfsCalls, 0, '5,200+ entry lookups must never fall back to linear VFS scans');
  assert.ok(
    elapsedMs < 1500,
    `Expected 3,000 O(1) lookups over 5,200+ TOC entries to finish quickly (<1500ms), took ${elapsedMs}ms`
  );

  const tableStats = table.getStats();
  assert.equal(tableStats.lookupHits, 3000);
  assert.equal(tableStats.lookupMisses, 0);
  assert.equal(tableStats.tocCacheHits, 1900, '19 warm rounds x 100 hot paths should hit resolved-path LRU');
  assert.equal(tableStats.resolvedPathCacheSize, 100);
  assert.ok(tableStats.resolvedPathCacheSize <= tableStats.resolvedPathCacheCapacity);

  // Instant re-open of 5,200+ entry TOC via shared RpfTocCache performs zero range reads.
  const trackerReopen = createTrackingRangeReader(archive);
  const vfsReopened = await RpfArchiveVfs.open(trackerReopen.readRange, archive.byteLength, {
    archivePath: 'x64/large_ui_bundle.rpf',
    tocCache,
  });
  assert.equal(trackerReopen.reads.length, 0);
  assert.equal(vfsReopened.getStats().tocCacheHits, 1);
  assert.equal(vfsReopened.getStats().indexedEntryCount, openStats.indexedEntryCount);

  // Verify strict numeric-only sanitization on all stats objects.
  for (const statsObj of [vfs.getStats(), tableStats, vfsReopened.getStats(), tocCache.getStats()]) {
    for (const [key, val] of Object.entries(statsObj)) {
      assert.equal(typeof val, 'number', `Stat field ${key} must be numeric`);
      assert.ok(Number.isFinite(val) && val >= 0);
    }
  }
});
