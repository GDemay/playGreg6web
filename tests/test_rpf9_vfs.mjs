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
  RpfFormatError,
  EncryptedArchiveError,
  classifyEncryption,
  parseRpfHeader,
  normalizeVirtualPath,
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
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.parseRpfHeader, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.RpfArchiveVfs, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.RpfVfsMountTable, 'function');
  assert.equal(typeof sandbox.PlayGta6Rpf9Vfs.buildSyntheticRpf9Buffer, 'function');
});
