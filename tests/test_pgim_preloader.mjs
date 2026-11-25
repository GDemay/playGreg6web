import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pgimMod = require('../web/runtime/pgim_preloader.js');
const {
  PGIM_MAGIC,
  PGIM_MAGIC_BYTES,
  PGIM_VERSION,
  PGIM_HEADER_BYTES,
  PGIM_SECTION_ENTRY_BYTES,
  PGIM_PAGE_SIZE,
  PGIM_FLAGS,
  PGIM_SECTION_FLAGS,
  PGIM_PROT,
  PROGRAM_MANIFEST_SCHEMA,
  PROGRAM_MANIFEST_SCHEMA_VERSION,
  PgimPreloadError,
  PgimPreloader,
  parsePgimHeader,
  parsePgimLayout,
  resolveSafeManifestUrl,
  parseProgramManifest,
  parseBootManifest,
  preloadProgramModules,
  pgimFnv1a64,
} = pgimMod;
import { buildWgslPackBuffer, parseWgslPack } from '../web/workers/wgsl_pack_loader.js';

test('UMD exports PlayGta6PgimPreloader on globalThis and module.exports', () => {
  assert.equal(globalThis.PlayGta6PgimPreloader, pgimMod);
  assert.equal(typeof pgimMod.PgimPreloader, 'function');
  assert.equal(typeof pgimMod.parsePgimHeader, 'function');
  assert.equal(pgimMod.default, pgimMod.PgimPreloader);
});

// ---------------------------------------------------------------------------------------------
// Synthetic PGIM v1 writer mirroring `write_image` in recomp/src/image.rs byte-for-byte.
// ---------------------------------------------------------------------------------------------

function fnvReference(bytes) {
  let hash = 0xcbf29ce484222325n;
  for (const b of bytes) {
    hash ^= BigInt(b);
    hash = (hash * 0x100000001b3n) & 0xffffffffffffffffn;
  }
  return hash;
}

function alignUp(value, align) {
  return Math.ceil(value / align) * align;
}

function filled(length, seed) {
  const out = new Uint8Array(length);
  let x = seed >>> 0 || 1;
  for (let i = 0; i < length; i += 1) {
    x = (Math.imul(x, 1103515245) + 12345) >>> 0;
    out[i] = x >>> 24;
  }
  return out;
}

/**
 * @param {{
 *   imageBase?: bigint, sizeOfImage?: number, sectionAlignment?: number, entryPointRva?: number,
 *   flags?: number, sections?: Array<{ name: string, rva: number, virtualSize: number, protection: number,
 *     flags?: number, data?: Uint8Array, characteristics?: number }>, dir64?: number[], highlow?: number[],
 * }} spec
 */
function buildPgim(spec = {}) {
  const {
    imageBase = 0x140000000n,
    sizeOfImage = 0x6000,
    sectionAlignment = 0x1000,
    entryPointRva = 0x1010,
    flags = PGIM_FLAGS.HEADERS_MAPPED,
    sections = defaultSections(),
    dir64 = [0x2000, 0x2010],
    highlow = [0x2020],
  } = spec;
  const tablesEnd = PGIM_HEADER_BYTES + sections.length * PGIM_SECTION_ENTRY_BYTES + (dir64.length + highlow.length) * 4;
  let cursor = tablesEnd;
  const offsets = sections.map((s) => {
    const len = s.data ? s.data.byteLength : 0;
    if (len === 0) return 0;
    const offset = alignUp(cursor, PGIM_PAGE_SIZE);
    cursor = offset + len;
    return offset;
  });
  const fileSize = cursor;
  const out = new Uint8Array(fileSize);
  const view = new DataView(out.buffer);
  out.set(PGIM_MAGIC_BYTES, 0);
  view.setUint32(4, PGIM_VERSION, true);
  view.setUint32(8, PGIM_HEADER_BYTES, true);
  view.setUint32(12, PGIM_SECTION_ENTRY_BYTES, true);
  view.setUint32(16, flags, true);
  view.setUint32(20, PGIM_PAGE_SIZE, true);
  view.setBigUint64(24, imageBase, true);
  view.setUint32(32, sizeOfImage, true);
  view.setUint32(36, sectionAlignment, true);
  view.setUint32(40, entryPointRva, true);
  view.setUint32(44, sections.length, true);
  view.setUint32(48, dir64.length, true);
  view.setUint32(52, highlow.length, true);
  view.setBigUint64(56, BigInt(fileSize), true);
  sections.forEach((s, i) => {
    const at = PGIM_HEADER_BYTES + i * PGIM_SECTION_ENTRY_BYTES;
    out.set(new TextEncoder().encode(s.name), at);
    const data = s.data || new Uint8Array(0);
    view.setUint32(at + 16, s.rva, true);
    view.setUint32(at + 20, s.virtualSize, true);
    view.setUint32(at + 24, s.protection, true);
    view.setUint32(at + 28, s.flags || 0, true);
    view.setBigUint64(at + 32, BigInt(offsets[i]), true);
    view.setBigUint64(at + 40, BigInt(data.byteLength), true);
    view.setBigUint64(at + 48, fnvReference(data), true);
    view.setUint32(at + 56, s.characteristics || 0, true);
    view.setUint32(at + 60, 0, true);
    if (data.byteLength) out.set(data, offsets[i]);
  });
  let r = PGIM_HEADER_BYTES + sections.length * PGIM_SECTION_ENTRY_BYTES;
  for (const rva of [...dir64, ...highlow]) {
    view.setUint32(r, rva, true);
    r += 4;
  }
  return out;
}

function defaultSections() {
  return [
    { name: '<headers>', rva: 0, virtualSize: 0x400, protection: PGIM_PROT.READ, flags: PGIM_SECTION_FLAGS.HEADERS, data: filled(0x400, 1) },
    {
      name: '.text',
      rva: 0x1000,
      virtualSize: 0x1000,
      protection: PGIM_PROT.READ | PGIM_PROT.EXECUTE,
      data: filled(0xe00, 2),
      characteristics: 0x60000020,
    },
    {
      name: '.data',
      rva: 0x2000,
      virtualSize: 0x2000,
      protection: PGIM_PROT.READ | PGIM_PROT.WRITE,
      data: filled(0x1800, 3),
      characteristics: 0xc0000040,
    },
    { name: '.bss', rva: 0x4000, virtualSize: 0x2000, protection: PGIM_PROT.READ | PGIM_PROT.WRITE, characteristics: 0xc0000080 },
  ];
}

/** Larger image (~40 KiB) so small chunk sizes produce many parallel Range requests. */
function buildLargePgim() {
  return buildPgim({
    sizeOfImage: 0x10000,
    sections: [
      { name: '<headers>', rva: 0, virtualSize: 0x400, protection: PGIM_PROT.READ, flags: PGIM_SECTION_FLAGS.HEADERS, data: filled(0x400, 7) },
      { name: '.text', rva: 0x1000, virtualSize: 0x6000, protection: PGIM_PROT.READ | PGIM_PROT.EXECUTE, data: filled(0x5f00, 8) },
      { name: '.rdata', rva: 0x7000, virtualSize: 0x3000, protection: PGIM_PROT.READ, data: filled(0x2a10, 9) },
      { name: '.data', rva: 0xa000, virtualSize: 0x4000, protection: PGIM_PROT.READ | PGIM_PROT.WRITE, data: filled(0x1234, 10) },
    ],
    dir64: [0x7000, 0x7008, 0xa100],
    highlow: [0x7010, 0xa200],
  });
}

function mutate(bytes, fn) {
  const copy = bytes.slice();
  fn(new DataView(copy.buffer), copy);
  return copy;
}

function expectCode(code) {
  return (err) => {
    assert.ok(err instanceof PgimPreloadError, `expected PgimPreloadError, got ${err}`);
    assert.equal(err.code, code, err.message);
    return true;
  };
}

// ---------------------------------------------------------------------------------------------
// Mock HTTP server (fetch implementation) with HEAD + Range support toggles.
// ---------------------------------------------------------------------------------------------

function delay(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(timer);
        reject(signal.reason);
      },
      { once: true }
    );
  });
}

function createServer(files, options = {}) {
  const {
    supportRange = true,
    headStatus = 200,
    acceptRanges = supportRange ? 'bytes' : 'none',
    delayMs = 0,
    headContentLength = null,
    failStatuses = null,
  } = options;
  const log = [];
  let inFlight = 0;
  let maxInFlight = 0;
  const remainingFails = new Map();
  if (failStatuses) {
    for (const [url, statuses] of Object.entries(failStatuses)) {
      remainingFails.set(url, statuses.slice());
    }
  }
  const fetchImpl = async (url, init = {}) => {
    const method = init.method || 'GET';
    const range = init.headers?.Range ?? init.headers?.range ?? null;
    log.push({ url, method, range });
    if (init.signal?.aborted) throw init.signal.reason;
    inFlight += 1;
    maxInFlight = Math.max(maxInFlight, inFlight);
    try {
      if (delayMs) await delay(delayMs, init.signal);
      const queued = remainingFails.get(url);
      if (queued && queued.length > 0) {
        const statusOrError = queued.shift();
        if (typeof statusOrError === 'number') {
          return new Response(null, { status: statusOrError });
        }
        throw statusOrError;
      }
      const body = files[url];
      if (body === undefined) return new Response(null, { status: 404 });
      if (method === 'HEAD') {
        if (headStatus !== 200) return new Response(null, { status: headStatus });
        const headers = { 'content-length': String(headContentLength ?? body.byteLength) };
        if (acceptRanges !== null) headers['accept-ranges'] = acceptRanges;
        return new Response(null, { status: 200, headers });
      }
      if (range && supportRange) {
        const m = /^bytes=(\d+)-(\d+)$/.exec(range);
        const start = Number(m[1]);
        if (start >= body.byteLength) return new Response(null, { status: 416 });
        const end = Math.min(Number(m[2]), body.byteLength - 1);
        return new Response(body.slice(start, end + 1), {
          status: 206,
          headers: {
            'content-range': `bytes ${start}-${end}/${body.byteLength}`,
            'content-length': String(end - start + 1),
          },
        });
      }
      return new Response(body.slice(), { status: 200, headers: { 'content-length': String(body.byteLength) } });
    } finally {
      inFlight -= 1;
    }
  };
  return {
    fetchImpl,
    log,
    get maxInFlight() {
      return maxInFlight;
    },
  };
}

function assertMonotonic(events, total) {
  let lastLoaded = -1;
  let lastPct = -1;
  for (const e of events) {
    assert.ok(e.loadedBytes >= lastLoaded, `loadedBytes regressed: ${e.loadedBytes} < ${lastLoaded}`);
    assert.ok(e.pct >= lastPct, `pct regressed: ${e.pct} < ${lastPct}`);
    assert.ok(e.pct >= 0 && e.pct <= 100);
    lastLoaded = e.loadedBytes;
    lastPct = e.pct;
    if (e.phase !== 'probe') assert.equal(e.totalBytes, total);
  }
  const last = events[events.length - 1];
  assert.equal(last.phase, 'complete');
  assert.equal(last.loadedBytes, total);
  assert.equal(last.pct, 100);
}

// ---------------------------------------------------------------------------------------------
// Constants, checksum, header parsing.
// ---------------------------------------------------------------------------------------------

test('constants match recomp/src/image.rs and runtime/os/image_loader.h', () => {
  assert.equal(PGIM_MAGIC, 'PGIM');
  assert.deepEqual([...PGIM_MAGIC_BYTES], [0x50, 0x47, 0x49, 0x4d]);
  assert.equal(PGIM_VERSION, 1);
  assert.equal(PGIM_HEADER_BYTES, 64);
  assert.equal(PGIM_SECTION_ENTRY_BYTES, 64);
  assert.equal(PGIM_PAGE_SIZE, 4096);
  assert.deepEqual({ ...PGIM_FLAGS }, { HEADERS_MAPPED: 1, EXECUTABLE_BYTES_OMITTED: 2, RELOCS_STRIPPED: 4 });
  assert.deepEqual({ ...PGIM_SECTION_FLAGS }, { HEADERS: 1, BYTES_OMITTED: 2 });
  assert.deepEqual({ ...PGIM_PROT }, { READ: 1, WRITE: 2, EXECUTE: 4 });
});

test('pgimFnv1a64 matches FNV-1a 64 reference vectors and BigInt reference', () => {
  assert.equal(pgimFnv1a64(new Uint8Array(0)), 0xcbf29ce484222325n);
  assert.equal(pgimFnv1a64(new TextEncoder().encode('a')), 0xaf63dc4c8601ec8cn);
  assert.equal(pgimFnv1a64(new TextEncoder().encode('foobar')), 0x85944171f73967e8n);
  for (const seed of [1, 2, 3, 99]) {
    const data = filled(1000 + seed, seed);
    assert.equal(pgimFnv1a64(data), fnvReference(data));
  }
  const ff = new Uint8Array(4096).fill(0xff);
  assert.equal(pgimFnv1a64(ff), fnvReference(ff));
});

test('parsePgimHeader returns structured metadata for a valid PGIM v1 image', () => {
  const bytes = buildPgim({ flags: PGIM_FLAGS.HEADERS_MAPPED | PGIM_FLAGS.RELOCS_STRIPPED });
  const h = parsePgimHeader(bytes, { totalBytes: bytes.byteLength });
  assert.equal(h.magic, 'PGIM');
  assert.equal(h.version, 1);
  assert.equal(h.imageBase, 0x140000000n);
  assert.equal(typeof h.imageBase, 'bigint');
  assert.equal(h.sizeOfImage, 0x6000);
  assert.equal(h.sectionAlignment, 0x1000);
  assert.equal(h.entryPointRva, 0x1010);
  assert.equal(h.entryPoint, 0x140001010n);
  assert.equal(h.sectionCount, 4);
  assert.equal(h.dir64Count, 2);
  assert.equal(h.highlowCount, 1);
  assert.equal(h.relocCount, 3);
  assert.equal(h.fileSize, bytes.byteLength);
  assert.equal(h.headersMapped, true);
  assert.equal(h.relocsStripped, true);
  assert.equal(h.executableBytesOmitted, false);
  assert.equal(h.relocTableOffset, 64 + 4 * 64);
  assert.equal(h.tablesEnd, 64 + 4 * 64 + 3 * 4);
  // Only the first 64 bytes are needed when no transport size is supplied.
  assert.equal(parsePgimHeader(bytes.subarray(0, 64)).fileSize, bytes.byteLength);
  // ArrayBuffer and unaligned views are accepted.
  assert.equal(parsePgimHeader(bytes.slice().buffer).sectionCount, 4);
  const shifted = new Uint8Array(bytes.byteLength + 3);
  shifted.set(bytes, 3);
  assert.equal(parsePgimHeader(shifted.subarray(3)).imageBase, 0x140000000n);
});

test('parsePgimHeader rejects bad magic, version, truncation, and corrupt fields', () => {
  const good = buildPgim();
  assert.throws(() => parsePgimHeader(good.subarray(0, 63)), expectCode('TRUNCATED'));
  assert.throws(() => parsePgimHeader(new Uint8Array(0)), /needed 64 bytes, only 0 available/);
  assert.throws(() => parsePgimHeader(mutate(good, (v, b) => b.set([0x4d, 0x5a, 0, 0], 0))), expectCode('BAD_MAGIC'));
  assert.throws(() => parsePgimHeader(mutate(good, (v, b) => b.set([0x50, 0x47, 0x49, 0x4e], 0))), /expected "PGIM"/);
  assert.throws(() => parsePgimHeader(mutate(good, (v) => v.setUint32(4, 2, true))), expectCode('UNSUPPORTED_VERSION'));
  assert.throws(() => parsePgimHeader(mutate(good, (v) => v.setUint32(8, 32, true))), (e) => e.field === 'header_bytes');
  assert.throws(() => parsePgimHeader(mutate(good, (v) => v.setUint32(12, 40, true))), (e) => e.field === 'section_entry_bytes');
  assert.throws(() => parsePgimHeader(mutate(good, (v) => v.setUint32(20, 65536, true))), (e) => e.field === 'page_size');
  assert.throws(() => parsePgimHeader(mutate(good, (v) => v.setUint32(16, 0x8, true))), (e) => e.field === 'flags');
  assert.throws(
    () => parsePgimHeader(mutate(good, (v) => v.setBigUint64(24, 0xfffffffffffff000n, true))),
    (e) => e.code === 'BAD_HEADER_FIELD' && e.field === 'image_base + size_of_image'
  );
  assert.throws(() => parsePgimHeader(good, { totalBytes: good.byteLength + 1 }), (e) => e.field === 'file_size');
  assert.throws(
    () => parsePgimHeader(mutate(good, (v) => v.setBigUint64(56, 1n << 60n, true))),
    (e) => e.field === 'file_size'
  );
  // Section/relocation tables larger than the declared container.
  assert.throws(() => parsePgimHeader(mutate(good, (v) => v.setUint32(44, 1000, true))), expectCode('TRUNCATED'));
  assert.throws(() => parsePgimHeader(mutate(good, (v) => v.setUint32(52, 0xffffffff, true))), /relocation tables/);
  assert.throws(() => parsePgimHeader('PGIM'), TypeError);
});

test('parsePgimLayout parses sections and relocations and validates like read_image', () => {
  const bytes = buildPgim();
  const layout = parsePgimLayout(bytes, { verifyChecksums: true });
  assert.deepEqual(
    layout.sections.map((s) => [s.name, s.rva, s.virtualSize, s.protection, s.isHeaders, s.payloadLength]),
    [
      ['<headers>', 0, 0x400, 1, true, 0x400],
      ['.text', 0x1000, 0x1000, 5, false, 0xe00],
      ['.data', 0x2000, 0x2000, 3, false, 0x1800],
      ['.bss', 0x4000, 0x2000, 3, false, 0],
    ]
  );
  for (const s of layout.sections) {
    if (s.payloadLength) assert.equal(s.payloadOffset % PGIM_PAGE_SIZE, 0);
  }
  assert.equal(layout.sections[1].characteristics, 0x60000020);
  assert.deepEqual([...layout.dir64], [0x2000, 0x2010]);
  assert.deepEqual([...layout.highlow], [0x2020]);
  assert.equal(layout.sections[2].checksum, fnvReference(filled(0x1800, 3)));
});

test('parsePgimLayout rejects corrupt section and relocation tables', () => {
  const good = buildPgim();
  const sec = (i, off) => PGIM_HEADER_BYTES + i * PGIM_SECTION_ENTRY_BYTES + off;
  const payloadOffset = (i) => Number(new DataView(good.buffer).getBigUint64(sec(i, 32), true));
  // Checksums only verified on request.
  const flipped = mutate(good, (v, b) => {
    b[payloadOffset(2) + 5] ^= 0xff;
  });
  assert.doesNotThrow(() => parsePgimLayout(flipped));
  assert.throws(() => parsePgimLayout(flipped, { verifyChecksums: true }), expectCode('CHECKSUM_MISMATCH'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setBigUint64(sec(1, 32), BigInt(payloadOffset(1) + 8), true))), expectCode('PAYLOAD_MISALIGNED'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setBigUint64(sec(1, 40), 0x100000n, true))), expectCode('PAYLOAD_OUT_OF_RANGE'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setBigUint64(sec(2, 32), BigInt(payloadOffset(1)), true))), expectCode('PAYLOAD_OVERLAP'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(sec(1, 24), 8, true))), /unknown protection bits/);
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(sec(1, 28), 4, true))), /unknown section flags/);
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(sec(1, 60), 1, true))), /reserved field/);
  assert.throws(() => parsePgimLayout(mutate(good, (v, b) => { b[sec(1, 0) + 7] = 0x41; })), /name padding/);
  assert.throws(() => parsePgimLayout(mutate(good, (v, b) => { b[sec(1, 0)] = 0xff; })), /not UTF-8/);
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(sec(3, 20), 0x3000, true))), expectCode('SECTION_OUT_OF_IMAGE'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(sec(2, 16), 0x1800, true))), expectCode('OVERLAPPING_SECTIONS'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(16, 0, true))), /headers flag/);
  const relocAt = PGIM_HEADER_BYTES + 4 * PGIM_SECTION_ENTRY_BYTES;
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(relocAt + 4, 0x1ff0, true))), expectCode('RELOCATIONS_UNSORTED'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(relocAt + 8, 0x2014, true))), expectCode('OVERLAPPING_RELOCATIONS'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(relocAt + 8, 0x2000, true))), expectCode('RELOCATIONS_UNSORTED'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(relocAt + 8, 0x5ffe, true))), expectCode('RELOC_SITE_OUT_OF_RANGE'));
  assert.throws(() => parsePgimLayout(mutate(good, (v) => v.setUint32(relocAt + 8, 0x800, true))), expectCode('RELOC_SITE_UNMAPPED'));
  // Truncated container (file_size no longer matches the buffer).
  assert.throws(() => parsePgimLayout(good.subarray(0, good.byteLength - 1)), (e) => e.field === 'file_size');
});

test('parsePgimLayout enforces executable-bytes-omitted rules', () => {
  const omitted = buildPgim({
    flags: PGIM_FLAGS.HEADERS_MAPPED | PGIM_FLAGS.EXECUTABLE_BYTES_OMITTED,
    sections: [
      { name: '<headers>', rva: 0, virtualSize: 0x400, protection: PGIM_PROT.READ, flags: PGIM_SECTION_FLAGS.HEADERS, data: filled(0x400, 4) },
      { name: '.text', rva: 0x1000, virtualSize: 0x1000, protection: PGIM_PROT.READ | PGIM_PROT.EXECUTE, flags: PGIM_SECTION_FLAGS.BYTES_OMITTED },
      { name: '.data', rva: 0x2000, virtualSize: 0x1000, protection: PGIM_PROT.READ | PGIM_PROT.WRITE, data: filled(0x800, 5) },
    ],
    dir64: [0x2000],
    highlow: [],
  });
  assert.equal(parsePgimLayout(omitted, { verifyChecksums: true }).sections[1].bytesOmitted, true);
  const relocInText = mutate(omitted, (v) => v.setUint32(PGIM_HEADER_BYTES + 3 * PGIM_SECTION_ENTRY_BYTES, 0x1000, true));
  assert.throws(() => parsePgimLayout(relocInText), expectCode('RELOC_SITE_UNMAPPED'));
  const notOmitted = mutate(omitted, (v) => v.setUint32(PGIM_HEADER_BYTES + PGIM_SECTION_ENTRY_BYTES + 28, 0, true));
  assert.throws(() => parsePgimLayout(notOmitted), /must be omitted/);
});

// ---------------------------------------------------------------------------------------------
// PgimPreloader streaming.
// ---------------------------------------------------------------------------------------------

test('PgimPreloader constructor validates options', () => {
  const fetchImpl = async () => new Response(null);
  assert.throws(() => new PgimPreloader({ fetchImpl, chunkSize: 10 }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, concurrency: 0 }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, rangeCache: {} }), TypeError);
  assert.throws(() => new PgimPreloader({ fetchImpl: 'nope' }), TypeError);
  const p = new PgimPreloader({ fetchImpl });
  assert.equal(p.chunkSize, 4 * 1024 * 1024);
  assert.equal(p.concurrency, 4);
  assert.equal(p.useSharedArrayBuffer, false);
});

test('preloadGuestImage streams parallel HTTP 206 chunks with monotonic progress', async () => {
  const image = buildLargePgim();
  const server = createServer({ '/data/GTA6.pgim': image }, { delayMs: 3 });
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096, concurrency: 3 });
  const events = [];
  const result = await preloader.preloadGuestImage('/data/GTA6.pgim', { onProgress: (e) => events.push(e) });

  assert.deepEqual(result.bytes, image);
  assert.ok(result.bytes.buffer instanceof ArrayBuffer);
  assert.equal(result.shared, false);
  assert.equal(result.transport, 'range');
  assert.equal(result.totalBytes, image.byteLength);
  assert.equal(result.header.imageBase, 0x140000000n);
  assert.equal(result.layout.sections.length, 4);
  assert.ok(result.timingMs >= 0);

  const chunkCount = Math.ceil(image.byteLength / 4096);
  assert.equal(server.log[0].method, 'HEAD');
  const gets = server.log.filter((r) => r.method === 'GET');
  assert.equal(gets.length, chunkCount);
  assert.equal(result.requests, chunkCount + 1);
  assert.equal(gets[0].range, 'bytes=0-4095');
  const covered = gets.map((r) => r.range).sort((a, b) => Number(/=(\d+)/.exec(a)[1]) - Number(/=(\d+)/.exec(b)[1]));
  assert.equal(covered[covered.length - 1], `bytes=${(chunkCount - 1) * 4096}-${image.byteLength - 1}`);
  assert.ok(server.maxInFlight > 1, 'chunks should be fetched in parallel');
  assert.ok(server.maxInFlight <= 3, `concurrency exceeded: ${server.maxInFlight}`);

  assert.equal(events[0].phase, 'probe');
  assert.equal(events[0].totalBytes, null);
  assert.equal(events[1].phase, 'header');
  assert.equal(events[1].loadedBytes, 4096);
  assert.equal(events.filter((e) => e.phase === 'download').length, chunkCount - 1);
  assertMonotonic(events, image.byteLength);
});

test('preloadGuestImage backs the image with SharedArrayBuffer when requested', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image });
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 8192, useSharedArrayBuffer: true });
  const result = await preloader.preloadGuestImage('img.pgim', { verifyChecksums: true });
  assert.ok(result.bytes.buffer instanceof SharedArrayBuffer);
  assert.equal(result.shared, true);
  assert.equal(result.bytes.byteLength, image.byteLength);
  assert.deepEqual(new Uint8Array(result.bytes), image);
});

test('preloadGuestImage handles a single-chunk image', async () => {
  const image = buildPgim();
  const server = createServer({ 'small.pgim': image });
  const result = await new PgimPreloader({ fetchImpl: server.fetchImpl }).preloadGuestImage('small.pgim');
  assert.deepEqual(result.bytes, image);
  assert.deepEqual(
    server.log.map((r) => [r.method, r.range]),
    [
      ['HEAD', null],
      ['GET', `bytes=0-${image.byteLength - 1}`],
    ]
  );
});

test('preloadGuestImage supports AbortSignal cancellation', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image }, { delayMs: 5 });
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096, concurrency: 2 });
  const controller = new AbortController();
  const events = [];
  await assert.rejects(
    preloader.preloadGuestImage('img.pgim', {
      signal: controller.signal,
      onProgress: (e) => {
        events.push(e);
        if (e.phase === 'download') controller.abort();
      },
    }),
    (err) => err.name === 'AbortError'
  );
  const gets = server.log.filter((r) => r.method === 'GET').length;
  assert.ok(gets < Math.ceil(image.byteLength / 4096), `expected cancellation to stop fetching (${gets} GETs)`);
  assert.ok(!events.some((e) => e.phase === 'complete'));

  // Pre-aborted signals reject without touching the network.
  const before = server.log.length;
  const pre = new AbortController();
  pre.abort(new Error('user cancelled'));
  await assert.rejects(preloader.preloadGuestImage('img.pgim', { signal: pre.signal }), /user cancelled/);
  assert.equal(server.log.length, before);
});

test('preloadGuestImage aborts while waiting on a hung request', async () => {
  const image = buildPgim();
  const fetchImpl = (url, init) =>
    init.method === 'HEAD'
      ? Promise.resolve(new Response(null, { headers: { 'content-length': String(image.byteLength) } }))
      : new Promise(() => {}); // never settles and ignores the signal
  const controller = new AbortController();
  const pending = new PgimPreloader({ fetchImpl }).preloadGuestImage('x.pgim', { signal: controller.signal });
  setTimeout(() => controller.abort(), 5);
  await assert.rejects(pending, (err) => err.name === 'AbortError');
});

test('preloadGuestImage falls back to a single GET when the server ignores Range', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image }, { supportRange: false, acceptRanges: null });
  const events = [];
  const result = await new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096 }).preloadGuestImage('img.pgim', {
    onProgress: (e) => events.push(e),
  });
  assert.deepEqual(result.bytes, image);
  assert.equal(result.transport, 'full');
  assert.deepEqual(
    server.log.map((r) => r.method),
    ['HEAD', 'GET']
  );
  assertMonotonic(events, image.byteLength);
});

test('preloadGuestImage skips Range requests when HEAD reports Accept-Ranges: none', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image }, { supportRange: false, acceptRanges: 'none' });
  const result = await new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096, useSharedArrayBuffer: true }).preloadGuestImage('img.pgim');
  assert.deepEqual(new Uint8Array(result.bytes), image);
  assert.ok(result.bytes.buffer instanceof SharedArrayBuffer);
  assert.deepEqual(
    server.log.map((r) => [r.method, r.range]),
    [
      ['HEAD', null],
      ['GET', null],
    ]
  );
});

test('preloadGuestImage probes with a Range GET when HEAD is unsupported', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image }, { headStatus: 405 });
  const result = await new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 16384 }).preloadGuestImage('img.pgim');
  assert.deepEqual(result.bytes, image);
  assert.equal(server.log[1].range, 'bytes=0-16383');
  assert.equal(server.log.filter((r) => r.method === 'GET').length, Math.ceil(image.byteLength / 16384));
});

test('preloadGuestImage rejects corrupt or missing images before streaming the rest', async () => {
  const image = buildLargePgim();
  const badMagic = mutate(image, (v, b) => b.set([0x4d, 0x5a, 0x90, 0x00], 0));
  const server = createServer({ 'bad.pgim': badMagic, 'tiny.pgim': image.slice(0, 32) });
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096 });
  await assert.rejects(preloader.preloadGuestImage('bad.pgim'), expectCode('BAD_MAGIC'));
  assert.equal(server.log.filter((r) => r.url === 'bad.pgim').length, 2, 'only HEAD + first chunk');

  await assert.rejects(preloader.preloadGuestImage('missing.pgim'), (e) => e.code === 'NOT_FOUND' && /HTTP 404/.test(e.message));
  await assert.rejects(preloader.preloadGuestImage('tiny.pgim'), expectCode('TRUNCATED'));

  // Header file_size disagreeing with the HTTP size.
  const wrongSize = mutate(image, (v) => v.setBigUint64(56, BigInt(image.byteLength + 4096), true));
  const s2 = createServer({ 'w.pgim': wrongSize });
  await assert.rejects(new PgimPreloader({ fetchImpl: s2.fetchImpl, chunkSize: 4096 }).preloadGuestImage('w.pgim'), (e) => e.field === 'file_size');

  // Layout corruption past the first chunk is caught after download.
  const relocAt = PGIM_HEADER_BYTES + 4 * PGIM_SECTION_ENTRY_BYTES;
  const badReloc = mutate(image, (v) => v.setUint32(relocAt, 0xffff0, true));
  const s3 = createServer({ 'r.pgim': badReloc });
  await assert.rejects(new PgimPreloader({ fetchImpl: s3.fetchImpl, chunkSize: 4096 }).preloadGuestImage('r.pgim'), /relocation/);

  // Checksum verification on request.
  const flipped = mutate(image, (v, b) => {
    b[b.byteLength - 1] ^= 0x55;
  });
  const s4 = createServer({ 'c.pgim': flipped });
  const p4 = new PgimPreloader({ fetchImpl: s4.fetchImpl, chunkSize: 4096 });
  await p4.preloadGuestImage('c.pgim');
  await assert.rejects(p4.preloadGuestImage('c.pgim', { verifyChecksums: true }), expectCode('CHECKSUM_MISMATCH'));
});

test('preloadGuestImage fails cleanly when a chunk request errors', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image });
  const fetchImpl = async (url, init) => {
    if (init.headers?.Range === 'bytes=12288-16383') return new Response(null, { status: 500 });
    return server.fetchImpl(url, init);
  };
  await assert.rejects(
    new PgimPreloader({ fetchImpl, chunkSize: 4096, concurrency: 2, maxRetries: 0 }).preloadGuestImage('img.pgim'),
    (e) => e.code === 'HTTP_ERROR' && e.status === 500
  );
  const network = async (url, init) => {
    if (init.method === 'HEAD') return server.fetchImpl(url, init);
    throw new TypeError('fetch failed');
  };
  await assert.rejects(
    new PgimPreloader({ fetchImpl: network, maxRetries: 0 }).preloadGuestImage('img.pgim'),
    expectCode('NETWORK_ERROR')
  );
});

test('preloadGuestImage reads chunks through a RangeCache when provided', async () => {
  const image = buildLargePgim();
  const reads = [];
  const rangeCache = {
    async read(path, offset, length) {
      reads.push([path, offset, length]);
      return image.slice(offset, Math.min(image.byteLength, offset + length));
    },
  };
  const server = createServer({ 'game/GTA6.pgim': image });
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, rangeCache, chunkSize: 8192 });
  const result = await preloader.preloadGuestImage('game/GTA6.pgim');
  assert.deepEqual(result.bytes, image);
  assert.equal(result.transport, 'range-cache');
  assert.deepEqual(reads[0], ['game/GTA6.pgim', 0, 8192]);
  assert.equal(reads.length, Math.ceil(image.byteLength / 8192));
  assert.deepEqual(
    server.log.map((r) => r.method),
    ['HEAD']
  );
});

// ---------------------------------------------------------------------------------------------
// preloadBootBundle.
// ---------------------------------------------------------------------------------------------

function s6Manifest() {
  return {
    schema_version: 1,
    register_convention: { constant_buffer: 'b', texture: 't', sampler: 's', uav: 'u' },
    bind_group_layouts: { '0x472f25790659490f': { entries: [] } },
    shaders: {},
  };
}

test('preloadBootBundle preloads PGIM + WGPK shader pack + S6 reflection manifest', async () => {
  const image = buildLargePgim();
  const pack = new Uint8Array(
    buildWgslPackBuffer([
      { hash: 0x1122334455667788n, stage: 'vertex', wgsl: '@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }' },
      { hash: 0x99aabbccddeeff00n, stage: 'fragment', wgsl: '@fragment fn main() {}' },
    ])
  );
  const manifest = s6Manifest();
  const server = createServer(
    {
      'boot/GTA6.pgim': image,
      'boot/shaders.wgsl.pack': pack,
      'boot/reflection-manifest.json': new TextEncoder().encode(JSON.stringify(manifest)),
    },
    { delayMs: 1 }
  );
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096, useSharedArrayBuffer: true });
  const events = [];
  const result = await preloader.preloadBootBundle({
    pgimUrl: 'boot/GTA6.pgim',
    shaderPackUrl: 'boot/shaders.wgsl.pack',
    reflectionUrl: 'boot/reflection-manifest.json',
    onProgress: (e) => events.push(e),
  });
  assert.deepEqual(new Uint8Array(result.pgimBytes), image);
  assert.ok(result.pgimBytes.buffer instanceof SharedArrayBuffer);
  assert.equal(result.shared, true);
  assert.equal(result.pgimHeader.entryPoint, 0x140001010n);
  assert.deepEqual(result.shaderPackBytes, pack);
  assert.equal(parseWgslPack(result.shaderPackBytes).size, 2);
  assert.deepEqual(result.reflectionManifest, manifest);
  for (const key of ['pgim', 'shaderPack', 'reflection', 'total']) {
    assert.equal(typeof result.timingMs[key], 'number', key);
    assert.ok(result.timingMs[key] >= 0);
  }
  assert.ok(result.timingMs.total >= result.timingMs.pgim);
  assertMonotonic(events.filter((e) => e.asset === 'pgim'), image.byteLength);
  assert.ok(events.some((e) => e.asset === 'shaderPack' && e.phase === 'complete' && e.loadedBytes === pack.byteLength));
  assert.ok(events.some((e) => e.asset === 'reflection' && e.phase === 'complete'));
  assert.deepEqual(events[events.length - 1], {
    phase: 'complete',
    asset: 'bundle',
    loadedBytes: image.byteLength,
    totalBytes: image.byteLength,
    pct: 100,
  });
});

test('preloadBootBundle returns null for omitted optional assets', async () => {
  const image = buildPgim();
  const server = createServer({ 'a.pgim': image });
  const result = await new PgimPreloader({ fetchImpl: server.fetchImpl }).preloadBootBundle({ pgimUrl: 'a.pgim' });
  assert.deepEqual(result.pgimBytes, image);
  assert.equal(result.shaderPackBytes, null);
  assert.equal(result.reflectionManifest, null);
  assert.equal(result.timingMs.shaderPack, null);
  assert.equal(result.timingMs.reflection, null);
  await assert.rejects(new PgimPreloader({ fetchImpl: server.fetchImpl }).preloadBootBundle({}), TypeError);
});

test('preloadBootBundle soft-fails optional shader/reflection URLs that 404', async () => {
  const image = buildPgim();
  const events = [];
  const server = createServer({ 'a.pgim': image });
  const result = await new PgimPreloader({ fetchImpl: server.fetchImpl }).preloadBootBundle({
    pgimUrl: 'a.pgim',
    shaderPackUrl: 'missing.wgsl.pack',
    reflectionUrl: 'nope.json',
    onProgress: (e) => events.push(e),
  });
  assert.deepEqual(result.pgimBytes, image);
  assert.equal(result.shaderPackBytes, null);
  assert.equal(result.reflectionManifest, null);
  assert.ok(events.some((e) => e.phase === 'optional-missing' && e.asset === 'shaderPack'));
  assert.ok(events.some((e) => e.phase === 'optional-missing' && e.asset === 'reflection'));
});

test('preloadBootBundle rejects corrupt shader packs and reflection manifests', async () => {
  const image = buildLargePgim();
  const pack = new Uint8Array(buildWgslPackBuffer([{ hash: 1n, stage: 'compute', wgsl: '@compute @workgroup_size(1) fn main() {}' }]));
  const badPack = pack.slice();
  badPack[0] = 0x58;
  const server = createServer(
    {
      'g.pgim': image,
      'bad.wgsl.pack': badPack,
      'trunc.wgsl.pack': pack.slice(0, 20),
      'v2.json': new TextEncoder().encode(JSON.stringify({ schema_version: 2 })),
      'broken.json': new TextEncoder().encode('{"schema_version": 1,'),
    },
    { delayMs: 2 }
  );
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096 });
  await assert.rejects(preloader.preloadBootBundle({ pgimUrl: 'g.pgim', shaderPackUrl: 'bad.wgsl.pack' }), (e) => e.code === 'BAD_SHADER_PACK' && /magic/.test(e.message));
  await assert.rejects(preloader.preloadBootBundle({ pgimUrl: 'g.pgim', shaderPackUrl: 'trunc.wgsl.pack' }), /entry table/);
  await assert.rejects(preloader.preloadBootBundle({ pgimUrl: 'g.pgim', reflectionUrl: 'v2.json' }), (e) => e.code === 'BAD_REFLECTION_MANIFEST' && /schema_version/.test(e.message));
  await assert.rejects(preloader.preloadBootBundle({ pgimUrl: 'g.pgim', reflectionUrl: 'broken.json' }), /malformed JSON/);
});

test('preloadBootBundle propagates AbortSignal cancellation to every asset', async () => {
  const image = buildLargePgim();
  const server = createServer(
    { 'g.pgim': image, 's.wgsl.pack': new Uint8Array(buildWgslPackBuffer([])), 'r.json': new TextEncoder().encode('{"schema_version":1}') },
    { delayMs: 20 }
  );
  const controller = new AbortController();
  const pending = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096 }).preloadBootBundle({
    pgimUrl: 'g.pgim',
    shaderPackUrl: 's.wgsl.pack',
    reflectionUrl: 'r.json',
    signal: controller.signal,
  });
  setTimeout(() => controller.abort(), 5);
  await assert.rejects(pending, (err) => err.name === 'AbortError');
});

test('preloadPgim returns { header, buffer, bytesLoaded, durationMs } for a 64-bit imageBase', async () => {
  const image = buildPgim({ imageBase: 0x140000000n, entryPointRva: 0x1010 });
  const server = createServer({ 'guest.pgim': image });
  const result = await new PgimPreloader({ fetchImpl: server.fetchImpl }).preloadPgim('guest.pgim');
  assert.equal(result.header.imageBase, 0x140000000n);
  assert.equal(result.header.entryPointRva, 0x1010);
  assert.ok(result.buffer instanceof ArrayBuffer);
  assert.equal(result.buffer.byteLength, image.byteLength);
  assert.deepEqual(new Uint8Array(result.buffer), image);
  assert.equal(result.bytesLoaded, image.byteLength);
  assert.ok(result.durationMs >= 0);
});

test('preloadShaderPack validates WGPK magic and preloadReflectionManifest validates schema', async () => {
  const pack = new Uint8Array(
    buildWgslPackBuffer([{ hash: 0xabcden, stage: 'vertex', wgsl: '@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }' }])
  );
  const manifest = s6Manifest();
  const server = createServer({
    'ok.wgsl.pack': pack,
    'ok.json': new TextEncoder().encode(JSON.stringify(manifest)),
    'bad.wgsl.pack': new Uint8Array([1, 2, 3, 4]),
  });
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl });
  const shader = await preloader.preloadShaderPack('ok.wgsl.pack');
  assert.equal(parseWgslPack(shader.bytes).size, 1);
  assert.equal(shader.bytesLoaded, pack.byteLength);
  const reflection = await preloader.preloadReflectionManifest('ok.json');
  assert.deepEqual(reflection.manifest, manifest);
  await assert.rejects(preloader.preloadShaderPack('bad.wgsl.pack'), (e) => e.code === 'BAD_SHADER_PACK');
});

test('preloadPgim retries transient HTTP 503 and network errors with backoff', async () => {
  const image = buildPgim();
  const sleeps = [];
  const server = createServer(
    { 'retry.pgim': image },
    {
      failStatuses: {
        'retry.pgim': [503, 503, new TypeError('socket hang up')],
      },
    }
  );
  const result = await new PgimPreloader({
    fetchImpl: server.fetchImpl,
    maxRetries: 3,
    baseRetryDelayMs: 5,
    sleep: async (ms) => {
      sleeps.push(ms);
    },
  }).preloadPgim('retry.pgim');
  assert.deepEqual(new Uint8Array(result.buffer), image);
  assert.deepEqual(sleeps, [5, 10, 20]);
  assert.ok(server.log.filter((r) => r.method === 'GET' || r.method === 'HEAD').length >= 4);
});

test('preloadPgim honors AbortSignal during retry backoff', async () => {
  const image = buildPgim();
  const server = createServer({ 'x.pgim': image }, { failStatuses: { 'x.pgim': [503, 503, 503, 503] } });
  const controller = new AbortController();
  const pending = new PgimPreloader({
    fetchImpl: server.fetchImpl,
    maxRetries: 5,
    baseRetryDelayMs: 30,
  }).preloadPgim('x.pgim', { signal: controller.signal });
  setTimeout(() => controller.abort(), 10);
  await assert.rejects(pending, (err) => err.name === 'AbortError');
});

// ---------------------------------------------------------------------------------------------
// W70: Multi-module program-manifest.json parsing, URL safety, preloading & boot-manifest fallback.
// ---------------------------------------------------------------------------------------------

function jsonBytes(obj) {
  return new TextEncoder().encode(JSON.stringify(obj));
}

function buildModulePgim(seed, { imageBase = 0x140000000n, textBytes = 0xe00, dataBytes = 0x1800 } = {}) {
  const textVSize = alignUp(Math.max(0x1000, textBytes), 0x1000);
  const dataRva = 0x1000 + textVSize;
  const dataVSize = alignUp(Math.max(0x1000, dataBytes), 0x1000);
  const sizeOfImage = dataRva + dataVSize;
  return buildPgim({
    imageBase,
    sizeOfImage,
    entryPointRva: 0x1010,
    sections: [
      { name: '<headers>', rva: 0, virtualSize: 0x400, protection: PGIM_PROT.READ, flags: PGIM_SECTION_FLAGS.HEADERS, data: filled(0x400, seed) },
      {
        name: '.text',
        rva: 0x1000,
        virtualSize: textVSize,
        protection: PGIM_PROT.READ | PGIM_PROT.EXECUTE,
        data: filled(textBytes, seed + 1),
      },
      {
        name: '.data',
        rva: dataRva,
        virtualSize: dataVSize,
        protection: PGIM_PROT.READ | PGIM_PROT.WRITE,
        data: filled(dataBytes, seed + 2),
      },
    ],
    dir64: [dataRva],
    highlow: [],
  });
}

function sampleProgramManifest() {
  return {
    schema: 'playgta6-recomp-program',
    schema_version: 1,
    modules: [
      {
        file: 'vcruntime140.dll',
        dll: 'vcruntime140.dll',
        module: 'vcruntime140',
        role: 'dll',
        base: '0x100000000',
        size_of_image: '0x4000',
        entry_point: '0x100001010',
        build_manifest: { path: 'vcruntime140/build-manifest.json', fnv1a64: '0x0000000000000001' },
      },
      {
        file: 'vcruntime140_1.dll',
        dll: 'vcruntime140_1.dll',
        module: 'vcruntime140_1',
        role: 'dll',
        base: '0x100010000',
        size_of_image: '0x4000',
        entry_point: '0x100011010',
        build_manifest: { path: 'vcruntime140_1/build-manifest.json', fnv1a64: '0x0000000000000002' },
      },
      {
        file: 'msvcp140.dll',
        dll: 'msvcp140.dll',
        module: 'msvcp140',
        role: 'dll',
        base: '0x100020000',
        size_of_image: '0x6000',
        entry_point: '0x100021010',
        build_manifest: { path: 'msvcp140/build-manifest.json', fnv1a64: '0x0000000000000003' },
      },
      {
        file: 'game_scarlett_master.exe',
        dll: 'game_scarlett_master.exe',
        module: 'main',
        role: 'exe',
        base: '0x140000000',
        size_of_image: '0x10000',
        entry_point: '0x140001010',
        build_manifest: { path: 'main/build-manifest.json', fnv1a64: '0x0000000000000004' },
      },
    ],
    init_order: ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll', 'game_scarlett_master.exe'],
    files: [
      { path: 'pg6_program.cpp', size: 1200 },
      { path: 'program-report.txt', size: 400 },
    ],
  };
}

test('W70 exports multi-module manifest helpers on UMD global and module.exports', () => {
  assert.equal(PROGRAM_MANIFEST_SCHEMA, 'playgta6-recomp-program');
  assert.equal(PROGRAM_MANIFEST_SCHEMA_VERSION, 1);
  assert.equal(typeof parseProgramManifest, 'function');
  assert.equal(typeof parseBootManifest, 'function');
  assert.equal(typeof resolveSafeManifestUrl, 'function');
  assert.equal(typeof preloadProgramModules, 'function');
  assert.equal(typeof PgimPreloader.prototype.preloadProgramModules, 'function');
  assert.equal(globalThis.PlayGta6PgimPreloader.parseProgramManifest, parseProgramManifest);
  assert.equal(globalThis.PlayGta6PgimPreloader.preloadProgramModules, preloadProgramModules);
});

test('parseProgramManifest discovers all 4 guest modules in init order from object, string, or bytes', () => {
  const manifest = sampleProgramManifest();
  for (const input of [manifest, JSON.stringify(manifest), jsonBytes(manifest), jsonBytes(manifest).buffer]) {
    const parsed = parseProgramManifest(input, { baseUrl: '/boot/program-manifest.json' });
    assert.equal(parsed.schema, 'playgta6-recomp-program');
    assert.equal(parsed.schemaVersion, 1);
    assert.equal(parsed.baseUrl, '/boot/');
    assert.deepEqual(parsed.initOrder, ['vcruntime140', 'vcruntime140_1', 'msvcp140', 'main']);
    assert.deepEqual(
      parsed.modules.map((m) => [m.name, m.url, m.path, m.role]),
      [
        ['vcruntime140', '/boot/vcruntime140/image/guest.pgim', 'vcruntime140/image/guest.pgim', 'dll'],
        ['vcruntime140_1', '/boot/vcruntime140_1/image/guest.pgim', 'vcruntime140_1/image/guest.pgim', 'dll'],
        ['msvcp140', '/boot/msvcp140/image/guest.pgim', 'msvcp140/image/guest.pgim', 'dll'],
        ['main', '/boot/main/image/guest.pgim', 'main/image/guest.pgim', 'exe'],
      ]
    );
    assert.equal(parsed.modules[3].buildManifestUrl, '/boot/main/build-manifest.json');
    assert.equal(parsed.modules[3].base, '0x140000000');
    assert.equal(parsed.modules[3].entryPoint, '0x140001010');
  }
});

test('parseProgramManifest handles arbitrary module lists, explicit pgim paths, files[] lookup, and origin bases', () => {
  // String module list.
  const simple = parseProgramManifest(
    { schema: 'playgta6-recomp-program', modules: ['alpha', 'beta', 'gamma'] },
    { baseUrl: 'boot/program-manifest.json' }
  );
  assert.deepEqual(
    simple.modules.map((m) => [m.name, m.url]),
    [
      ['alpha', 'boot/alpha/image/guest.pgim'],
      ['beta', 'boot/beta/image/guest.pgim'],
      ['gamma', 'boot/gamma/image/guest.pgim'],
    ]
  );

  // Explicit per-module paths and files[] fallback (ignoring .pgim.json).
  const custom = parseProgramManifest(
    {
      schema: 'playgta6-recomp-program',
      schema_version: 1,
      modules: [
        { module: 'helper', pgimPath: './helper/custom.pgim' },
        { module: 'core' },
        { name: 'entry', image: { path: '/boot/entry/bin/game.pgim', container_bytes: 8192 } },
      ],
      files: [
        { path: 'core/image/core_module.pgim', size: 4096 },
        { path: 'core/image/core_module.pgim.json', size: 256 },
      ],
    },
    { baseUrl: 'http://127.0.0.1:8080/boot/program-manifest.json' }
  );
  assert.deepEqual(
    custom.modules.map((m) => [m.name, m.url, m.expectedBytes]),
    [
      ['helper', 'http://127.0.0.1:8080/boot/helper/custom.pgim', null],
      ['core', 'http://127.0.0.1:8080/boot/core/image/core_module.pgim', 4096],
      ['entry', 'http://127.0.0.1:8080/boot/entry/bin/game.pgim', 8192],
    ]
  );

  // Same-origin full URL inside base dir is accepted.
  const sameOrigin = resolveSafeManifestUrl(
    'http://127.0.0.1:8080/boot/main/image/guest.pgim',
    'http://127.0.0.1:8080/boot/program-manifest.json'
  );
  assert.equal(sameOrigin.url, 'http://127.0.0.1:8080/boot/main/image/guest.pgim');
  assert.equal(sameOrigin.path, 'main/image/guest.pgim');
});

test('parseProgramManifest rejects malformed manifests, unsupported schema, and duplicate modules', () => {
  assert.throws(() => parseProgramManifest('{not valid json'), expectCode('BAD_PROGRAM_MANIFEST'));
  assert.throws(() => parseProgramManifest('[]'), expectCode('BAD_PROGRAM_MANIFEST'));
  assert.throws(() => parseProgramManifest({ schema: 'other-schema', modules: ['main'] }), expectCode('BAD_PROGRAM_MANIFEST'));
  assert.throws(() => parseProgramManifest({ schema: 'playgta6-recomp-program', schema_version: 2, modules: ['main'] }), expectCode('BAD_PROGRAM_MANIFEST'));
  assert.throws(() => parseProgramManifest({ schema: 'playgta6-recomp-program', modules: [] }), expectCode('BAD_PROGRAM_MANIFEST'));
  assert.throws(() => parseProgramManifest({ schema: 'playgta6-recomp-program' }), expectCode('BAD_PROGRAM_MANIFEST'));
  assert.throws(
    () => parseProgramManifest({ schema: 'playgta6-recomp-program', modules: ['main', 'MAIN'] }),
    (e) => e.code === 'BAD_PROGRAM_MANIFEST' && /duplicate/.test(e.message)
  );
});

test('parseProgramManifest and resolveSafeManifestUrl strictly reject unsafe and traversal URLs', () => {
  const unsafePaths = [
    '../secret/guest.pgim',
    'main/../../secret.pgim',
    '%2e%2e/secret.pgim',
    '%2e%2e%2fsecret.pgim',
    '%252e%252e/secret.pgim',
    'main\\..\\..\\secret.pgim',
    'main%5cimage%5cguest.pgim',
    'C:\\game\\guest.pgim',
    '//evil.example/main/image/guest.pgim',
    'https://evil.example/boot/main/image/guest.pgim',
    'http://127.0.0.1:9999/boot/main/image/guest.pgim',
    'http://127.0.0.1:8080/boot/sub/../main/image/guest.pgim',
    'http://user:pass@127.0.0.1:8080/boot/main/image/guest.pgim',
    'javascript:alert(1)//.pgim',
    'data:application/octet-stream,PGIM.pgim',
    'file:///etc/passwd.pgim',
    '/etc/secret.pgim',
    '/other/main/image/guest.pgim',
    'main/image/guest.pgim\0.pgim',
    'main/image/guest%00.pgim',
    'main/image/guest.pgim?cache=1',
    'main/image/guest.pgim#frag',
    'main/image/guest.exe',
    'main/image/.pgim',
    'main//image/guest.pgim',
    ' main/image/guest.pgim',
    '%E0%A4%A/guest.pgim',
  ];

  for (const badPath of unsafePaths) {
    assert.throws(
      () =>
        parseProgramManifest(
          { schema: 'playgta6-recomp-program', modules: [{ module: 'main', pgimUrl: badPath }] },
          { baseUrl: 'http://127.0.0.1:8080/boot/program-manifest.json' }
        ),
      expectCode('UNSAFE_URL'),
      `expected UNSAFE_URL for pgimUrl=${JSON.stringify(badPath)}`
    );
  }

  // Unsafe module names.
  for (const badMod of ['../evil', 'main/sub', 'main\\sub', '..main', 'mod%2e%2e', 'http://evil', 'a b', '.hidden']) {
    assert.throws(
      () => parseProgramManifest({ schema: 'playgta6-recomp-program', modules: [badMod] }, { baseUrl: '/boot/program-manifest.json' }),
      expectCode('UNSAFE_URL'),
      `expected UNSAFE_URL for module=${JSON.stringify(badMod)}`
    );
  }

  // Unsafe files[] .pgim path and unsafe build_manifest.path.
  assert.throws(
    () =>
      parseProgramManifest(
        {
          schema: 'playgta6-recomp-program',
          modules: [{ module: 'main' }],
          files: [{ path: '../escape.pgim' }],
        },
        { baseUrl: '/boot/program-manifest.json' }
      ),
    expectCode('UNSAFE_URL')
  );
  assert.throws(
    () =>
      parseProgramManifest(
        {
          schema: 'playgta6-recomp-program',
          modules: [{ module: 'main', build_manifest: { path: '../../etc/passwd' } }],
        },
        { baseUrl: '/boot/program-manifest.json' }
      ),
    expectCode('UNSAFE_URL')
  );
  // Root-relative path escaping a relative base directory.
  assert.throws(
    () => resolveSafeManifestUrl('/boot/main/image/guest.pgim', 'boot/program-manifest.json'),
    expectCode('UNSAFE_URL')
  );
});

test('parseBootManifest supports single-module boot-manifest.json shapes and rejects traversal', () => {
  const fromDirect = parseBootManifest(
    { pgim: 'main/image/guest.pgim', shaderPackUrl: 'shaders/boot.wgsl.pack' },
    { baseUrl: '/boot/boot-manifest.json' }
  );
  assert.equal(fromDirect.modules.length, 1);
  assert.equal(fromDirect.pgimUrl, '/boot/main/image/guest.pgim');
  assert.equal(fromDirect.shaderPackUrl, '/boot/shaders/boot.wgsl.pack');

  const fromBuildSchema = parseBootManifest(
    {
      schema: 'playgta6-recomp-build',
      schema_version: 1,
      module: 'main',
      files: [{ path: 'image/guest.pgim', size: 12345 }],
    },
    { baseUrl: '/boot/boot-manifest.json' }
  );
  assert.equal(fromBuildSchema.pgimUrl, '/boot/image/guest.pgim');
  assert.equal(fromBuildSchema.modules[0].expectedBytes, 12345);

  assert.throws(() => parseBootManifest({ pgim: '../evil.pgim' }, { baseUrl: '/boot/boot-manifest.json' }), expectCode('UNSAFE_URL'));
  assert.throws(() => parseBootManifest({ schema_version: 1 }), expectCode('BAD_BOOT_MANIFEST'));
});

test('preloadProgramModules preloads all 4 guest modules with unified aggregate byte-level progress', async () => {
  const vcruntime = buildModulePgim(10, { imageBase: 0x100000000n, textBytes: 0x1400, dataBytes: 0x1200 });
  const vcruntime1 = buildModulePgim(20, { imageBase: 0x100010000n, textBytes: 0x1000, dataBytes: 0x1000 });
  const msvcp = buildModulePgim(30, { imageBase: 0x100020000n, textBytes: 0x2600, dataBytes: 0x1e00 });
  const mainImg = buildModulePgim(40, { imageBase: 0x140000000n, textBytes: 0x4200, dataBytes: 0x2800 });
  const expectedTotal = vcruntime.byteLength + vcruntime1.byteLength + msvcp.byteLength + mainImg.byteLength;

  const server = createServer(
    {
      '/boot/program-manifest.json': jsonBytes(sampleProgramManifest()),
      '/boot/vcruntime140/image/guest.pgim': vcruntime,
      '/boot/vcruntime140_1/image/guest.pgim': vcruntime1,
      '/boot/msvcp140/image/guest.pgim': msvcp,
      '/boot/main/image/guest.pgim': mainImg,
    },
    { delayMs: 2 }
  );

  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096, concurrency: 3 });
  const events = [];
  const result = await preloader.preloadProgramModules('/boot/program-manifest.json', {
    onProgress: (e) => events.push(e),
    verifyChecksums: true,
  });

  assert.equal(result.source, 'program-manifest');
  assert.equal(result.manifestUrl, '/boot/program-manifest.json');
  assert.equal(result.usedFallback, false);
  assert.equal(result.moduleCount, 4);
  assert.equal(result.totalBytes, expectedTotal);
  assert.equal(result.bytesLoaded, expectedTotal);
  assert.deepEqual(
    result.modules.map((m) => m.name),
    ['vcruntime140', 'vcruntime140_1', 'msvcp140', 'main']
  );
  assert.deepEqual(result.modulesByName.vcruntime140.bytes, vcruntime);
  assert.deepEqual(result.modulesByName.vcruntime140_1.bytes, vcruntime1);
  assert.deepEqual(result.modulesByName.msvcp140.bytes, msvcp);
  assert.deepEqual(result.modulesByName.main.bytes, mainImg);
  assert.equal(result.mainModule, result.modulesByName.main);
  assert.equal(result.mainModule.header.imageBase, 0x140000000n);

  // Exactly 1 HEAD per module (no duplicate probes).
  const heads = server.log.filter((r) => r.method === 'HEAD');
  assert.equal(heads.length, 4);

  // Unified aggregate progress assertions.
  assert.equal(events[0].phase, 'manifest');
  assert.equal(events[1].phase, 'probe');
  assert.equal(events[1].totalBytes, expectedTotal);
  assert.equal(events[1].moduleCount, 4);
  assert.equal(events[1].moduleIndex, 0);
  assert.equal(events[1].currentModule, 'vcruntime140');

  let prevLoaded = -1;
  let prevPct = -1;
  for (const e of events) {
    assert.ok(e.loadedBytes >= prevLoaded, `loadedBytes regressed: ${e.loadedBytes} < ${prevLoaded}`);
    assert.ok(e.pct >= prevPct, `pct regressed: ${e.pct} < ${prevPct}`);
    assert.ok(e.pct >= 0 && e.pct <= 100);
    if (e.phase !== 'manifest') {
      assert.equal(e.totalBytes, expectedTotal);
      assert.equal(e.moduleCount, 4);
      assert.ok(e.moduleIndex >= 0 && e.moduleIndex < 4);
      assert.equal(typeof e.currentModule, 'string');
    }
    prevLoaded = e.loadedBytes;
    prevPct = e.pct;
  }

  const moduleCompletes = events.filter((e) => e.phase === 'module-complete');
  assert.deepEqual(
    moduleCompletes.map((e) => [e.moduleIndex, e.currentModule]),
    [
      [0, 'vcruntime140'],
      [1, 'vcruntime140_1'],
      [2, 'msvcp140'],
      [3, 'main'],
    ]
  );
  // Only ONE terminal 'complete' event at the very end.
  const completes = events.filter((e) => e.phase === 'complete');
  assert.equal(completes.length, 1);
  assert.equal(events[events.length - 1], completes[0]);
  assert.equal(completes[0].loadedBytes, expectedTotal);
  assert.equal(completes[0].totalBytes, expectedTotal);
  assert.equal(completes[0].pct, 100);
  assert.equal(completes[0].moduleIndex, 3);
  assert.equal(completes[0].currentModule, 'main');
});

test('preloadProgramModules warms OPFS RangeCache and supports SharedArrayBuffer and top-level helper', async () => {
  const m1 = buildModulePgim(51, { imageBase: 0x100000000n });
  const m2 = buildModulePgim(52, { imageBase: 0x140000000n });
  const files = {
    '/boot/program-manifest.json': jsonBytes({ schema: 'playgta6-recomp-program', modules: ['dep', 'main'] }),
    '/boot/dep/image/guest.pgim': m1,
    '/boot/main/image/guest.pgim': m2,
  };
  const server = createServer(files);
  const reads = [];
  const rangeCache = {
    async read(path, offset, length) {
      reads.push([path, offset, length]);
      const data = files[path];
      return data.slice(offset, Math.min(data.byteLength, offset + length));
    },
  };

  const res = await preloadProgramModules('/boot/program-manifest.json', {
    fetchImpl: server.fetchImpl,
    rangeCache,
    chunkSize: 4096,
    useSharedArrayBuffer: true,
  });

  assert.equal(res.moduleCount, 2);
  assert.equal(res.modulesByName.dep.transport, 'range-cache');
  assert.equal(res.modulesByName.main.transport, 'range-cache');
  assert.ok(res.modulesByName.dep.bytes.buffer instanceof SharedArrayBuffer);
  assert.ok(res.modulesByName.main.bytes.buffer instanceof SharedArrayBuffer);
  assert.ok(reads.some(([p]) => p === '/boot/dep/image/guest.pgim'));
  assert.ok(reads.some(([p]) => p === '/boot/main/image/guest.pgim'));
  // Only manifest GET + 2 module HEADs hit HTTP directly; all chunk reads went through rangeCache.
  assert.deepEqual(
    server.log.map((r) => [r.method, r.url]),
    [
      ['GET', '/boot/program-manifest.json'],
      ['HEAD', '/boot/dep/image/guest.pgim'],
      ['HEAD', '/boot/main/image/guest.pgim'],
    ]
  );
});

test('preloadProgramModules gracefully falls back to boot-manifest.json when program-manifest.json returns 404', async () => {
  const mainImg = buildModulePgim(61, { imageBase: 0x140000000n });
  const server = createServer({
    '/boot/boot-manifest.json': jsonBytes({ module: 'main', pgim: 'main/image/guest.pgim' }),
    '/boot/main/image/guest.pgim': mainImg,
  });

  const events = [];
  const result = await new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096 }).preloadProgramModules(
    '/boot/program-manifest.json',
    { onProgress: (e) => events.push(e) }
  );

  assert.equal(result.source, 'boot-manifest');
  assert.equal(result.requestedManifestUrl, '/boot/program-manifest.json');
  assert.equal(result.manifestUrl, '/boot/boot-manifest.json');
  assert.equal(result.usedFallback, true);
  assert.equal(result.moduleCount, 1);
  assert.deepEqual(result.mainModule.bytes, mainImg);
  assert.equal(result.totalBytes, mainImg.byteLength);
  assert.ok(events.some((e) => e.phase === 'fallback' && e.fallbackTo === '/boot/boot-manifest.json'));
  assert.equal(events[events.length - 1].phase, 'complete');
  assert.equal(events[events.length - 1].pct, 100);
});

test('preloadProgramModules falls back to fallbackPgimUrl when both manifests 404, and does not mask 500 or corrupt manifests', async () => {
  const mainImg = buildModulePgim(71, { imageBase: 0x140000000n });
  const server = createServer({
    '/boot/guest.pgim': mainImg,
    '/err/program-manifest.json': new TextEncoder().encode('{broken'),
    '/err/boot-manifest.json': jsonBytes({ pgim: '/boot/guest.pgim' }),
  });

  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, maxRetries: 0 });
  const fallbackRes = await preloader.preloadProgramModules('/boot/program-manifest.json', {
    fallbackPgimUrl: '/boot/guest.pgim',
  });
  assert.equal(fallbackRes.source, 'pgim-fallback');
  assert.equal(fallbackRes.usedFallback, true);
  assert.equal(fallbackRes.moduleCount, 1);
  assert.deepEqual(fallbackRes.mainModule.bytes, mainImg);

  // Without fallbackPgimUrl, double-404 rejects with NOT_FOUND.
  await assert.rejects(
    preloader.preloadProgramModules('/boot/program-manifest.json'),
    expectCode('NOT_FOUND')
  );

  // Malformed program-manifest.json must NOT fall back to boot-manifest.json.
  await assert.rejects(
    preloader.preloadProgramModules('/err/program-manifest.json'),
    expectCode('BAD_PROGRAM_MANIFEST')
  );

  // HTTP 500 on program-manifest.json must NOT fall back to boot-manifest.json.
  const s500 = createServer(
    {
      '/boot/program-manifest.json': jsonBytes(sampleProgramManifest()),
      '/boot/boot-manifest.json': jsonBytes({ pgim: 'guest.pgim' }),
      '/boot/guest.pgim': mainImg,
    },
    { failStatuses: { '/boot/program-manifest.json': [500] } }
  );
  await assert.rejects(
    new PgimPreloader({ fetchImpl: s500.fetchImpl, maxRetries: 0 }).preloadProgramModules('/boot/program-manifest.json'),
    (e) => e.code === 'HTTP_ERROR' && e.status === 500
  );
});

test('preloadProgramModules works when HEAD is 405 and handles AbortSignal, missing modules, and targetBuffer guard', async () => {
  const m1 = buildModulePgim(81, { imageBase: 0x100000000n });
  const m2 = buildModulePgim(82, { imageBase: 0x140000000n });
  const manifestBytes = jsonBytes({ schema: 'playgta6-recomp-program', modules: ['vcruntime140', 'main'] });

  // HEAD 405 fallback (size discovered during first Range chunk per module).
  const s405 = createServer(
    {
      '/boot/program-manifest.json': manifestBytes,
      '/boot/vcruntime140/image/guest.pgim': m1,
      '/boot/main/image/guest.pgim': m2,
    },
    { headStatus: 405 }
  );
  const events405 = [];
  const r405 = await new PgimPreloader({ fetchImpl: s405.fetchImpl, chunkSize: 4096 }).preloadProgramModules(
    '/boot/program-manifest.json',
    { onProgress: (e) => events405.push(e) }
  );
  assert.equal(r405.totalBytes, m1.byteLength + m2.byteLength);
  assert.equal(events405[events405.length - 1].phase, 'complete');
  assert.equal(events405[events405.length - 1].pct, 100);

  // Mid-transfer AbortSignal stops remaining modules.
  const sSlow = createServer(
    {
      '/boot/program-manifest.json': manifestBytes,
      '/boot/vcruntime140/image/guest.pgim': m1,
      '/boot/main/image/guest.pgim': m2,
    },
    { delayMs: 4 }
  );
  const controller = new AbortController();
  const abortEvents = [];
  await assert.rejects(
    new PgimPreloader({ fetchImpl: sSlow.fetchImpl, chunkSize: 4096 }).preloadProgramModules('/boot/program-manifest.json', {
      signal: controller.signal,
      onProgress: (e) => {
        abortEvents.push(e);
        if (e.phase === 'module-complete' && e.moduleIndex === 0) controller.abort();
      },
    }),
    (err) => err.name === 'AbortError'
  );
  assert.ok(!abortEvents.some((e) => e.phase === 'complete'));
  assert.equal(
    sSlow.log.filter((r) => r.method === 'GET' && r.url === '/boot/main/image/guest.pgim').length,
    0,
    'aborted preload must not start GETs for subsequent modules'
  );

  // Missing module fails fast during HEAD probe before downloading module 0.
  const sMissing = createServer({
    '/boot/program-manifest.json': manifestBytes,
    '/boot/vcruntime140/image/guest.pgim': m1,
  });
  await assert.rejects(
    new PgimPreloader({ fetchImpl: sMissing.fetchImpl }).preloadProgramModules('/boot/program-manifest.json'),
    expectCode('NOT_FOUND')
  );
  assert.equal(
    sMissing.log.filter((r) => r.method === 'GET' && r.url.endsWith('.pgim')).length,
    0,
    'missing module probe should fail before streaming PGIM chunks'
  );

  // targetBuffer cannot be shared across multiple modules.
  const sBuf = createServer({
    '/boot/program-manifest.json': manifestBytes,
    '/boot/vcruntime140/image/guest.pgim': m1,
    '/boot/main/image/guest.pgim': m2,
  });
  await assert.rejects(
    new PgimPreloader({ fetchImpl: sBuf.fetchImpl, targetBuffer: new ArrayBuffer(1024 * 1024) }).preloadProgramModules(
      '/boot/program-manifest.json'
    ),
    expectCode('ALLOCATION_FAILED')
  );

  // Unsafe manifestUrl / bootManifestUrl rejected.
  await assert.rejects(
    new PgimPreloader({ fetchImpl: sBuf.fetchImpl }).preloadProgramModules('../secret/program-manifest.json'),
    expectCode('UNSAFE_URL')
  );
  await assert.rejects(
    new PgimPreloader({ fetchImpl: sBuf.fetchImpl }).preloadProgramModules('/missing/program-manifest.json', {
      bootManifestUrl: 'javascript:alert(1)',
    }),
    expectCode('UNSAFE_URL')
  );
});

test('preloadBootBundle integrates multi-module programManifestUrl alongside shader pack and reflection manifest', async () => {
  const vcruntime = buildModulePgim(91, { imageBase: 0x100000000n });
  const mainImg = buildModulePgim(92, { imageBase: 0x140000000n });
  const pack = new Uint8Array(
    buildWgslPackBuffer([{ hash: 0x1234n, stage: 'vertex', wgsl: '@vertex fn main() -> @builtin(position) vec4f { return vec4f(); }' }])
  );
  const reflection = s6Manifest();
  const server = createServer({
    '/boot/program-manifest.json': jsonBytes({
      schema: 'playgta6-recomp-program',
      schema_version: 1,
      modules: [
        { module: 'vcruntime140', role: 'dll' },
        { module: 'main', role: 'exe' },
      ],
    }),
    '/boot/vcruntime140/image/guest.pgim': vcruntime,
    '/boot/main/image/guest.pgim': mainImg,
    '/shaders/boot.wgsl.pack': pack,
    '/shaders/reflection-manifest.json': jsonBytes(reflection),
  });

  const events = [];
  const bundle = await new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096 }).preloadBootBundle({
    programManifestUrl: '/boot/program-manifest.json',
    shaderPackUrl: '/shaders/boot.wgsl.pack',
    reflectionUrl: '/shaders/reflection-manifest.json',
    onProgress: (e) => events.push(e),
  });

  assert.equal(bundle.moduleCount, 2);
  assert.equal(bundle.bytesLoaded, vcruntime.byteLength + mainImg.byteLength);
  assert.deepEqual(bundle.pgimBytes, mainImg);
  assert.deepEqual(bundle.modulesByName.vcruntime140.bytes, vcruntime);
  assert.deepEqual(bundle.modulesByName.main.bytes, mainImg);
  assert.deepEqual(bundle.shaderPackBytes, pack);
  assert.deepEqual(bundle.reflectionManifest, reflection);
  assert.equal(events[events.length - 1].asset, 'bundle');
  assert.equal(events[events.length - 1].phase, 'complete');
  assert.equal(events[events.length - 1].moduleCount, 2);
  assert.equal(events[events.length - 1].loadedBytes, vcruntime.byteLength + mainImg.byteLength);
});

// ---------------------------------------------------------------------------------------------
// W147: Throughput telemetry, retry/abort counters, and bounded exponential chunk retry backoff.
// ---------------------------------------------------------------------------------------------

test('W147 constructor validates retryDelayMs, maxRetryDelayMs, sleepFn, nowFn and starts with idle zero stats', () => {
  const fetchImpl = async () => new Response(null);
  assert.throws(() => new PgimPreloader({ fetchImpl, maxRetries: -1 }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, maxRetries: 1.5 }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, retryDelayMs: -1 }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, retryDelayMs: Number.NaN }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, maxRetryDelayMs: -5 }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, maxRetryDelayMs: Number.POSITIVE_INFINITY }), RangeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, sleepFn: 'not-a-fn' }), TypeError);
  assert.throws(() => new PgimPreloader({ fetchImpl, nowFn: 123 }), TypeError);

  const customSleep = async () => {};
  const customNow = () => 42;
  const p = new PgimPreloader({
    fetchImpl,
    maxRetries: 2,
    retryDelayMs: 15,
    baseRetryDelayMs: 99,
    maxRetryDelayMs: 40,
    sleepFn: customSleep,
    nowFn: customNow,
  });
  assert.equal(p.maxRetries, 2);
  assert.equal(p.retryDelayMs, 15);
  assert.equal(p.baseRetryDelayMs, 15);
  assert.equal(p.maxRetryDelayMs, 40);
  assert.equal(p.sleepFn, customSleep);
  assert.equal(p.sleep, customSleep);
  assert.equal(p.nowFn, customNow);

  const initial = p.getStats();
  assert.deepEqual(initial, {
    state: 'idle',
    bytesLoaded: 0,
    totalBytes: 0,
    chunksFetched: 0,
    requests: 0,
    durationMs: 0,
    elapsedMs: 0,
    bytesPerSec: 0,
    peakBytesPerSec: 0,
    throughputMBps: 0,
    parallelMode: false,
    retryCount: 0,
    abortedCount: 0,
  });
  assert.deepEqual(p.getSnapshot(), initial);
});

test('W147 tracks finite bytesPerSec and peakBytesPerSec with deterministic nowFn across varying chunk speeds', async () => {
  const image = buildLargePgim();
  const chunkSize = 4096;
  const server = createServer({ '/data/game.pgim': image });
  let clockMs = 1000;
  let getIdx = 0;
  // Chunk 0: 100ms, Chunk 1: 25ms (fast burst -> peak), remaining chunks: 200ms each (slower tail)
  const chunkDurations = [100, 25];
  const fetchImpl = async (url, init) => {
    const res = await server.fetchImpl(url, init);
    if (init.method === 'GET') {
      const step = getIdx < chunkDurations.length ? chunkDurations[getIdx] : 200;
      getIdx += 1;
      clockMs += step;
    }
    return res;
  };

  const preloader = new PgimPreloader({
    fetchImpl,
    chunkSize,
    concurrency: 1,
    nowFn: () => clockMs,
  });

  const result = await preloader.preloadGuestImage('/data/game.pgim');
  const stats = preloader.getStats();

  assert.deepEqual(preloader.getSnapshot(), stats);
  assert.equal(stats.state, 'complete');
  assert.equal(stats.bytesLoaded, image.byteLength);
  assert.equal(stats.totalBytes, image.byteLength);
  assert.equal(stats.retryCount, 0);
  assert.equal(stats.abortedCount, 0);
  assert.ok(Number.isFinite(stats.durationMs) && stats.durationMs > 0);
  assert.ok(Number.isFinite(stats.bytesPerSec) && stats.bytesPerSec > 0);
  assert.ok(Number.isFinite(stats.peakBytesPerSec) && stats.peakBytesPerSec > stats.bytesPerSec);
  // Burst chunk was 4096 bytes in 25ms = 163840 bytes/sec.
  assert.equal(Math.round(stats.peakBytesPerSec), Math.round((chunkSize * 1000) / 25));
  assert.equal(result.bytesPerSec, stats.bytesPerSec);
  assert.equal(result.peakBytesPerSec, stats.peakBytesPerSec);
  assert.equal(result.retryCount, 0);
  assert.equal(result.abortedCount, 0);
});

test('W147 sanitizes non-finite, zero-elapsed, and backwards nowFn clocks to finite >= 0 telemetry', async () => {
  const image = buildPgim();
  const server = createServer({ 'z.pgim': image });

  for (const makeNowFn of [
    () => () => 500, // zero elapsed duration
    () => () => Number.NaN, // NaN clock
    () => () => Number.POSITIVE_INFINITY, // Infinity clock
    () => {
      let t = 2000;
      return () => {
        t -= 50; // backwards clock
        return t;
      };
    },
  ]) {
    const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, nowFn: makeNowFn() });
    const res = await preloader.preloadGuestImage('z.pgim');
    const stats = preloader.getStats();
    assert.equal(stats.state, 'complete');
    assert.equal(stats.durationMs, 0);
    assert.equal(stats.bytesPerSec, 0);
    assert.equal(stats.peakBytesPerSec, 0);
    assert.equal(res.bytesPerSec, 0);
    assert.equal(res.peakBytesPerSec, 0);
    assert.ok(Number.isFinite(stats.bytesPerSec));
    assert.ok(Number.isFinite(stats.peakBytesPerSec));
  }
});

test('W147 retries transient HTTP 5xx, 429, and network chunk failures with bounded exponential backoff', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image });
  const targetRange = 'bytes=4096-8191';
  const failures = [502, 429, 503, new TypeError('connection reset')];
  const sleeps = [];
  let clockMs = 100;

  const fetchImpl = async (url, init) => {
    if (init.method === 'GET' && init.headers?.Range === targetRange && failures.length > 0) {
      const next = failures.shift();
      if (typeof next === 'number') return new Response(null, { status: next });
      throw next;
    }
    const res = await server.fetchImpl(url, init);
    if (init.method === 'GET') clockMs += 10;
    return res;
  };

  const preloader = new PgimPreloader({
    fetchImpl,
    chunkSize: 4096,
    concurrency: 2,
    maxRetries: 4,
    retryDelayMs: 25,
    maxRetryDelayMs: 60,
    sleepFn: async (ms) => {
      sleeps.push(ms);
      clockMs += ms;
    },
    nowFn: () => clockMs,
  });

  const result = await preloader.preloadGuestImage('img.pgim');
  assert.deepEqual(result.bytes, image);
  // Exponential backoff: 25, 50, capped at 60, capped at 60.
  assert.deepEqual(sleeps, [25, 50, 60, 60]);
  const stats = preloader.getStats();
  assert.equal(stats.state, 'complete');
  assert.equal(stats.retryCount, 4);
  assert.equal(stats.abortedCount, 0);
  assert.equal(result.retryCount, 4);
  assert.equal(result.abortedCount, 0);
  assert.ok(stats.bytesPerSec > 0 && Number.isFinite(stats.bytesPerSec));
  assert.ok(stats.peakBytesPerSec >= stats.bytesPerSec);
});

test('W147 reports retry exhaustion cleanly for 5xx, 429, and network errors', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image });
  const targetRange = 'bytes=8192-12287';

  // Exhausted 503 on a chunk.
  {
    const sleeps = [];
    const fetchImpl = async (url, init) => {
      if (init.method === 'GET' && init.headers?.Range === targetRange) {
        return new Response(null, { status: 503 });
      }
      return server.fetchImpl(url, init);
    };
    const preloader = new PgimPreloader({
      fetchImpl,
      chunkSize: 4096,
      concurrency: 1,
      maxRetries: 2,
      retryDelayMs: 10,
      sleepFn: async (ms) => sleeps.push(ms),
    });
    await assert.rejects(
      preloader.preloadGuestImage('img.pgim'),
      (err) => err instanceof PgimPreloadError && err.code === 'HTTP_ERROR' && err.status === 503
    );
    assert.deepEqual(sleeps, [10, 20]);
    const stats = preloader.getStats();
    assert.equal(stats.state, 'error');
    assert.equal(stats.retryCount, 2);
    assert.equal(stats.abortedCount, 0);
  }

  // Exhausted 429 on a chunk.
  {
    const sleeps = [];
    const fetchImpl = async (url, init) => {
      if (init.method === 'GET' && init.headers?.Range === targetRange) {
        return new Response(null, { status: 429 });
      }
      return server.fetchImpl(url, init);
    };
    const preloader = new PgimPreloader({
      fetchImpl,
      chunkSize: 4096,
      concurrency: 1,
      maxRetries: 2,
      retryDelayMs: 12,
      sleepFn: async (ms) => sleeps.push(ms),
    });
    await assert.rejects(
      preloader.preloadGuestImage('img.pgim'),
      (err) => err instanceof PgimPreloadError && err.code === 'HTTP_ERROR' && err.status === 429
    );
    assert.deepEqual(sleeps, [12, 24]);
    assert.equal(preloader.getStats().retryCount, 2);
    assert.equal(preloader.getStats().abortedCount, 0);
  }

  // Exhausted transient network error.
  {
    const sleeps = [];
    const fetchImpl = async (url, init) => {
      if (init.method === 'GET' && init.headers?.Range === targetRange) {
        throw new TypeError('socket closed');
      }
      return server.fetchImpl(url, init);
    };
    const preloader = new PgimPreloader({
      fetchImpl,
      chunkSize: 4096,
      concurrency: 1,
      maxRetries: 2,
      retryDelayMs: 8,
      sleepFn: async (ms) => sleeps.push(ms),
    });
    await assert.rejects(preloader.preloadGuestImage('img.pgim'), expectCode('NETWORK_ERROR'));
    assert.deepEqual(sleeps, [8, 16]);
    assert.equal(preloader.getStats().retryCount, 2);
    assert.equal(preloader.getStats().abortedCount, 0);
  }
});

test('W147 never retries non-retryable 404, 400, or AbortError failures', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image });

  // 404 on missing image -> 0 retries, 0 sleeps.
  {
    const sleeps = [];
    const preloader = new PgimPreloader({
      fetchImpl: server.fetchImpl,
      maxRetries: 3,
      retryDelayMs: 10,
      sleepFn: async (ms) => sleeps.push(ms),
    });
    await assert.rejects(preloader.preloadGuestImage('missing.pgim'), (e) => e.code === 'NOT_FOUND' && e.status === 404);
    assert.deepEqual(sleeps, []);
    assert.equal(preloader.getStats().retryCount, 0);
    assert.equal(preloader.getStats().abortedCount, 0);
    assert.equal(preloader.getStats().state, 'error');
  }

  // 400 on a chunk -> 0 retries, 0 sleeps.
  {
    const sleeps = [];
    const fetchImpl = async (url, init) => {
      if (init.method === 'GET' && init.headers?.Range === 'bytes=4096-8191') {
        return new Response(null, { status: 400 });
      }
      return server.fetchImpl(url, init);
    };
    const preloader = new PgimPreloader({
      fetchImpl,
      chunkSize: 4096,
      concurrency: 1,
      maxRetries: 3,
      retryDelayMs: 10,
      sleepFn: async (ms) => sleeps.push(ms),
    });
    await assert.rejects(
      preloader.preloadGuestImage('img.pgim'),
      (e) => e instanceof PgimPreloadError && e.code === 'HTTP_ERROR' && e.status === 400
    );
    assert.deepEqual(sleeps, []);
    assert.equal(preloader.getStats().retryCount, 0);
    assert.equal(preloader.getStats().abortedCount, 0);
  }

  // Direct AbortError thrown by fetchImpl -> never retried, increments abortedCount.
  {
    const sleeps = [];
    let getCalls = 0;
    const fetchImpl = async (url, init) => {
      if (init.method === 'HEAD') return server.fetchImpl(url, init);
      getCalls += 1;
      throw new DOMException('fetch aborted internally', 'AbortError');
    };
    const preloader = new PgimPreloader({
      fetchImpl,
      maxRetries: 3,
      retryDelayMs: 10,
      sleepFn: async (ms) => sleeps.push(ms),
    });
    await assert.rejects(preloader.preloadGuestImage('img.pgim'), (e) => e.name === 'AbortError');
    assert.equal(getCalls, 1);
    assert.deepEqual(sleeps, []);
    assert.equal(preloader.getStats().retryCount, 0);
    assert.equal(preloader.getStats().abortedCount, 1);
    assert.equal(preloader.getStats().state, 'aborted');
  }
});

test('W147 honors AbortSignal immediately during retry backoff and tracks abortedCount + clean reset', async () => {
  const image = buildLargePgim();
  const server = createServer({ 'img.pgim': image });

  // Pre-aborted signal increments abortedCount without touching network or sleepFn.
  {
    const sleeps = [];
    let fetchCalls = 0;
    const preloader = new PgimPreloader({
      fetchImpl: async (...args) => {
        fetchCalls += 1;
        return server.fetchImpl(...args);
      },
      maxRetries: 3,
      retryDelayMs: 10,
      sleepFn: async (ms) => sleeps.push(ms),
    });
    const controller = new AbortController();
    controller.abort();
    await assert.rejects(
      preloader.preloadGuestImage('img.pgim', { signal: controller.signal }),
      (e) => e.name === 'AbortError'
    );
    assert.equal(fetchCalls, 0);
    assert.deepEqual(sleeps, []);
    assert.equal(preloader.getStats().abortedCount, 1);
    assert.equal(preloader.getStats().retryCount, 0);
    assert.equal(preloader.getStats().state, 'aborted');
  }

  // Abort during retry backoff cancels immediately even if sleepFn never resolves on its own.
  {
    const controller = new AbortController();
    let chunkAttempts = 0;
    let sleepStartedResolve;
    const sleepStarted = new Promise((r) => {
      sleepStartedResolve = r;
    });
    const fetchImpl = async (url, init) => {
      if (init.method === 'GET' && init.headers?.Range === 'bytes=4096-8191') {
        chunkAttempts += 1;
        return new Response(null, { status: 503 });
      }
      return server.fetchImpl(url, init);
    };
    const preloader = new PgimPreloader({
      fetchImpl,
      chunkSize: 4096,
      concurrency: 1,
      maxRetries: 5,
      retryDelayMs: 60000,
      sleepFn: () => {
        sleepStartedResolve();
        return new Promise(() => {}); // hangs forever unless aborted
      },
    });
    const pending = preloader.preloadGuestImage('img.pgim', { signal: controller.signal });
    await sleepStarted;
    controller.abort();
    await assert.rejects(pending, (e) => e.name === 'AbortError');
    assert.equal(chunkAttempts, 1, 'must not issue a second attempt after abort during backoff');
    assert.equal(preloader.getStats().abortedCount, 1);
    assert.equal(preloader.getStats().retryCount, 1);
    assert.equal(preloader.getStats().state, 'aborted');

    // Explicit resetStats() / reset() clears all telemetry counters cleanly.
    const cleared = preloader.resetStats();
    assert.deepEqual(cleared, {
      state: 'idle',
      bytesLoaded: 0,
      totalBytes: 0,
      chunksFetched: 0,
      requests: 0,
      durationMs: 0,
      elapsedMs: 0,
      bytesPerSec: 0,
      peakBytesPerSec: 0,
      throughputMBps: 0,
      parallelMode: false,
      retryCount: 0,
      abortedCount: 0,
    });
    assert.deepEqual(preloader.getStats(), cleared);
  }

  // Starting a new preload automatically resets telemetry from the previous run.
  {
    let clockMs = 0;
    let failOnce = true;
    const fetchImpl = async (url, init) => {
      if (init.method === 'GET' && init.headers?.Range === 'bytes=4096-8191' && failOnce) {
        failOnce = false;
        return new Response(null, { status: 503 });
      }
      const res = await server.fetchImpl(url, init);
      if (init.method === 'GET') clockMs += 20;
      return res;
    };
    const preloader = new PgimPreloader({
      fetchImpl,
      chunkSize: 4096,
      concurrency: 1,
      maxRetries: 2,
      retryDelayMs: 5,
      sleepFn: async (ms) => {
        clockMs += ms;
      },
      nowFn: () => clockMs,
    });

    // Run 1: triggers 1 retry.
    await preloader.preloadGuestImage('img.pgim');
    assert.equal(preloader.getStats().retryCount, 1);
    assert.equal(preloader.getStats().abortedCount, 0);

    // Run 2: aborts immediately -> resets retryCount to 0, sets abortedCount to 1.
    const aborted = new AbortController();
    aborted.abort();
    await assert.rejects(preloader.preloadGuestImage('img.pgim', { signal: aborted.signal }));
    assert.equal(preloader.getStats().retryCount, 0);
    assert.equal(preloader.getStats().abortedCount, 1);

    // Run 3: succeeds cleanly -> resets abortedCount to 0, retryCount to 0, fresh throughput.
    await preloader.preloadGuestImage('img.pgim');
    const finalStats = preloader.getStats();
    assert.equal(finalStats.state, 'complete');
    assert.equal(finalStats.retryCount, 0);
    assert.equal(finalStats.abortedCount, 0);
    assert.ok(finalStats.bytesPerSec > 0);
    assert.ok(finalStats.peakBytesPerSec >= finalStats.bytesPerSec);

    // reset() alias also works.
    preloader.reset();
    assert.equal(preloader.getStats().state, 'idle');
    assert.equal(preloader.getStats().bytesPerSec, 0);
    assert.equal(preloader.getStats().peakBytesPerSec, 0);
  }
});

// ---------------------------------------------------------------------------------------------
// W208: Parallel ranged prefetch optimizations, chunkBytes option, and throughput telemetry.
// ---------------------------------------------------------------------------------------------

test('W208 supports chunkBytes option, exports defaults, and validates per-call overrides', async () => {
  assert.equal(pgimMod.DEFAULT_CHUNK_BYTES, 4 * 1024 * 1024);
  assert.equal(pgimMod.DEFAULT_CONCURRENCY, 4);

  const image = buildLargePgim();
  const server = createServer({ '/w208/game.pgim': image }, { delayMs: 2 });

  assert.throws(() => new PgimPreloader({ fetchImpl: server.fetchImpl, chunkBytes: 16 }), RangeError);
  const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkBytes: 8192, concurrency: 3 });
  assert.equal(preloader.chunkBytes, 8192);
  assert.equal(preloader.chunkSize, 8192);
  assert.equal(preloader.concurrency, 3);

  await assert.rejects(
    preloader.preloadGuestImage('/w208/game.pgim', { chunkBytes: 32 }),
    RangeError
  );
  await assert.rejects(
    preloader.preloadGuestImage('/w208/game.pgim', { concurrency: 0 }),
    RangeError
  );

  // Per-call override with smaller chunkBytes + bounded concurrency.
  const res = await preloader.preloadGuestImage('/w208/game.pgim', { chunkBytes: 4096, concurrency: 2 });
  assert.deepEqual(res.bytes, image);
  assert.equal(res.chunksFetched, Math.ceil(image.byteLength / 4096));
  assert.equal(res.parallelMode, true);
  assert.ok(server.maxInFlight <= 2, `per-call concurrency 2 exceeded: ${server.maxInFlight}`);
});

test('W208 parallel Range chunk reassembly preserves exact bytes under out-of-order chunk latency', async () => {
  const image = buildLargePgim();
  const chunkBytes = 3072; // intentionally non-power-of-two / non-page-aligned boundary
  const expectedChunks = Math.ceil(image.byteLength / chunkBytes);
  let inFlight = 0;
  let maxInFlight = 0;
  let chunkIndex = 0;

  const fetchImpl = async (url, init = {}) => {
    const method = init.method || 'GET';
    if (method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: {
          'content-length': String(image.byteLength),
          'accept-ranges': 'bytes',
        },
      });
    }
    const range = init.headers?.Range;
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    assert.ok(match, `expected Range header, got ${range}`);
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), image.byteLength - 1);
    const slice = image.slice(start, end + 1);
    const myIdx = chunkIndex++;

    inFlight += 1;
    maxInFlight = Math.max(maxInFlight, inFlight);
    try {
      // Reverse latency so later chunks finish before earlier chunks.
      const waitMs = Math.max(1, (expectedChunks - (myIdx % expectedChunks)) * 2);
      await delay(waitMs, init.signal);
      // Stream chunk in multiple tiny sub-slices to exercise direct-write ReadableStream path.
      const stream = new ReadableStream({
        start(controller) {
          const mid = Math.floor(slice.byteLength / 2);
          if (mid > 0) controller.enqueue(slice.subarray(0, mid));
          controller.enqueue(slice.subarray(mid));
          controller.close();
        },
      });
      return new Response(stream, {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${image.byteLength}`,
          'content-length': String(slice.byteLength),
        },
      });
    } finally {
      inFlight -= 1;
    }
  };

  let clockMs = 500;
  const timedFetch = async (url, init) => {
    const res = await fetchImpl(url, init);
    if (init?.method === 'GET') clockMs += 10;
    return res;
  };

  const preloader = new PgimPreloader({
    fetchImpl: timedFetch,
    chunkBytes,
    concurrency: 4,
    nowFn: () => clockMs,
  });

  const result = await preloader.preloadGuestImage('/w208/out-of-order.pgim', { verifyChecksums: true });
  assert.deepEqual(result.bytes, image);
  assert.equal(result.bytesLoaded, image.byteLength);
  assert.equal(result.totalBytes, image.byteLength);
  assert.equal(result.chunksFetched, expectedChunks);
  assert.equal(result.parallelMode, true);
  assert.equal(result.transport, 'range');
  assert.ok(maxInFlight > 1 && maxInFlight <= 4, `expected 1 < maxInFlight <= 4, got ${maxInFlight}`);

  const stats = preloader.getStats();
  assert.equal(stats.bytesLoaded, image.byteLength);
  assert.equal(stats.totalBytes, image.byteLength);
  assert.equal(stats.chunksFetched, expectedChunks);
  assert.equal(stats.elapsedMs, stats.durationMs);
  assert.ok(stats.elapsedMs > 0);
  assert.equal(stats.parallelMode, true);
  const expectedMBps = (image.byteLength * 1000) / (stats.elapsedMs * 1024 * 1024);
  assert.ok(Math.abs(stats.throughputMBps - expectedMBps) < 1e-9);
  assert.equal(result.elapsedMs, stats.elapsedMs);
  assert.equal(result.throughputMBps, stats.throughputMBps);
});

test('W208 falls back gracefully to single streaming fetch with parallelMode=false when Range is unsupported', async () => {
  const image = buildLargePgim();
  let clockMs = 100;

  // Case 1: Accept-Ranges: none
  {
    const server = createServer({ 'no-range.pgim': image }, { supportRange: false, acceptRanges: 'none' });
    const fetchImpl = async (url, init) => {
      const res = await server.fetchImpl(url, init);
      if (init?.method === 'GET') clockMs += 50;
      return res;
    };
    const preloader = new PgimPreloader({ fetchImpl, chunkBytes: 4096, concurrency: 4, nowFn: () => clockMs });
    const result = await preloader.preloadPgim('no-range.pgim');
    assert.deepEqual(new Uint8Array(result.buffer), image);
    assert.equal(result.transport, 'full');
    assert.equal(result.parallelMode, false);
    assert.equal(result.chunksFetched, 1);
    assert.equal(result.bytesLoaded, image.byteLength);
    assert.equal(result.totalBytes, image.byteLength);
    assert.equal(result.elapsedMs, 50);
    assert.ok(result.throughputMBps > 0 && Number.isFinite(result.throughputMBps));

    const stats = preloader.getStats();
    assert.equal(stats.parallelMode, false);
    assert.equal(stats.chunksFetched, 1);
    assert.equal(stats.elapsedMs, 50);
    assert.equal(stats.throughputMBps, result.throughputMBps);
  }

  // Case 2: Server omits Accept-Ranges and ignores Range header (returns 200 OK stream)
  {
    const server = createServer({ 'ignored-range.pgim': image }, { supportRange: false, acceptRanges: null });
    const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkBytes: 4096, concurrency: 4 });
    const result = await preloader.preloadGuestImage('ignored-range.pgim');
    assert.deepEqual(result.bytes, image);
    assert.equal(result.transport, 'full');
    assert.equal(result.parallelMode, false);
    assert.equal(result.chunksFetched, 1);
    assert.equal(preloader.getStats().parallelMode, false);
    assert.equal(preloader.getStats().chunksFetched, 1);
  }
});

test('W208 retries mid-stream chunk body errors with backoff without corrupting reassembled bytes or double-counting', async () => {
  const image = buildLargePgim();
  const chunkBytes = 4096;
  const expectedChunks = Math.ceil(image.byteLength / chunkBytes);
  const flakyRange = 'bytes=8192-12287';
  let flakyAttempts = 0;
  const sleeps = [];
  let clockMs = 1000;

  const fetchImpl = async (url, init = {}) => {
    if (init.method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: { 'content-length': String(image.byteLength), 'accept-ranges': 'bytes' },
      });
    }
    const range = init.headers?.Range;
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), image.byteLength - 1);
    const slice = image.slice(start, end + 1);
    clockMs += 5;

    if (range === flakyRange) {
      flakyAttempts += 1;
      if (flakyAttempts <= 2) {
        // Emit corrupted/partial bytes then throw a transient network error mid-stream.
        const corruptPrefix = new Uint8Array(512).fill(0xee);
        const brokenStream = new ReadableStream({
          pull(controller) {
            if (!this._sent) {
              this._sent = true;
              controller.enqueue(corruptPrefix);
            } else {
              controller.error(new TypeError('stream reset by peer'));
            }
          },
        });
        return new Response(brokenStream, {
          status: 206,
          headers: {
            'content-range': `bytes ${start}-${end}/${image.byteLength}`,
            'content-length': String(slice.byteLength),
          },
        });
      }
    }

    return new Response(slice, {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${end}/${image.byteLength}`,
        'content-length': String(slice.byteLength),
      },
    });
  };

  const preloader = new PgimPreloader({
    fetchImpl,
    chunkBytes,
    concurrency: 3,
    maxRetries: 3,
    retryDelayMs: 15,
    sleepFn: async (ms) => {
      sleeps.push(ms);
      clockMs += ms;
    },
    nowFn: () => clockMs,
  });

  const result = await preloader.preloadGuestImage('/w208/flaky-stream.pgim', { verifyChecksums: true });
  assert.deepEqual(result.bytes, image);
  assert.equal(flakyAttempts, 3);
  assert.deepEqual(sleeps, [15, 30]);
  assert.equal(result.retryCount, 2);
  assert.equal(result.chunksFetched, expectedChunks);
  assert.equal(result.bytesLoaded, image.byteLength);
  assert.equal(result.parallelMode, true);
  assert.equal(preloader.getStats().chunksFetched, expectedChunks);
  assert.equal(preloader.getStats().retryCount, 2);
});

test('W208 preloadShaderPack uses parallel Range chunks when supported and falls back seamlessly when unsupported', async () => {
  // Build a larger WGPK shader pack (> 16 KiB) to span multiple 4 KiB Range chunks.
  const shaders = [];
  for (let i = 0; i < 40; i += 1) {
    shaders.push({
      hash: BigInt(0x1000 + i),
      stage: i % 2 === 0 ? 'vertex' : 'fragment',
      wgsl: `// shader ${i}\n${'/* pad */ '.repeat(45)}\n@fragment fn main_${i}() {}`,
    });
  }
  const pack = new Uint8Array(buildWgslPackBuffer(shaders));
  assert.ok(pack.byteLength > 16384, `expected >16 KiB shader pack, got ${pack.byteLength}`);
  const chunkBytes = 4096;
  const expectedChunks = Math.ceil(pack.byteLength / chunkBytes);

  // Parallel ranged path (Accept-Ranges: bytes + Content-Length).
  {
    const server = createServer({ '/shaders/boot.wgsl.pack': pack }, { delayMs: 2 });
    const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkBytes, concurrency: 4 });
    const res = await preloader.preloadShaderPack('/shaders/boot.wgsl.pack');
    assert.deepEqual(res.bytes, pack);
    assert.equal(parseWgslPack(res.bytes).size, 40);
    assert.equal(res.transport, 'range');
    assert.equal(res.parallelMode, true);
    assert.equal(res.chunksFetched, expectedChunks);
    assert.equal(res.bytesLoaded, pack.byteLength);
    assert.equal(res.totalBytes, pack.byteLength);
    assert.ok(server.maxInFlight > 1 && server.maxInFlight <= 4);
    assert.equal(preloader.getStats().parallelMode, true);
    assert.equal(preloader.getStats().chunksFetched, expectedChunks);
  }

  // Fallback path when Accept-Ranges: none.
  {
    const server = createServer({ '/shaders/boot.wgsl.pack': pack }, { supportRange: false, acceptRanges: 'none' });
    const preloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkBytes, concurrency: 4 });
    const res = await preloader.preloadShaderPack('/shaders/boot.wgsl.pack');
    assert.deepEqual(res.bytes, pack);
    assert.equal(res.transport, 'full');
    assert.equal(res.parallelMode, false);
    assert.equal(res.chunksFetched, 1);
    assert.equal(preloader.getStats().parallelMode, false);
    assert.equal(preloader.getStats().chunksFetched, 1);
  }
});

test('W208 aborts in-flight parallel Range chunk stream readers cleanly', async () => {
  const image = buildLargePgim();
  const controller = new AbortController();
  let cancelledReaders = 0;

  const fetchImpl = async (url, init = {}) => {
    if (init.method === 'HEAD') {
      return new Response(null, {
        status: 200,
        headers: { 'content-length': String(image.byteLength), 'accept-ranges': 'bytes' },
      });
    }
    const range = init.headers?.Range;
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(match[1]);
    const end = Math.min(Number(match[2]), image.byteLength - 1);
    const slice = image.slice(start, end + 1);
    if (start === 0) {
      return new Response(slice, {
        status: 206,
        headers: {
          'content-range': `bytes ${start}-${end}/${image.byteLength}`,
          'content-length': String(slice.byteLength),
        },
      });
    }
    // Hang chunk streams until aborted and count cancellations.
    const hangingStream = new ReadableStream({
      pull() {
        return new Promise(() => {});
      },
      cancel() {
        cancelledReaders += 1;
      },
    });
    return new Response(hangingStream, {
      status: 206,
      headers: {
        'content-range': `bytes ${start}-${end}/${image.byteLength}`,
        'content-length': String(slice.byteLength),
      },
    });
  };

  const preloader = new PgimPreloader({ fetchImpl, chunkBytes: 4096, concurrency: 3 });
  const pending = preloader.preloadGuestImage('/w208/abort-stream.pgim', { signal: controller.signal });
  setTimeout(() => controller.abort(), 15);
  await assert.rejects(pending, (err) => err.name === 'AbortError');
  assert.ok(cancelledReaders >= 1, 'expected hung chunk readers to be cancelled on abort');
  const stats = preloader.getStats();
  assert.equal(stats.state, 'aborted');
  assert.equal(stats.abortedCount, 1);
  assert.equal(stats.chunksFetched, 1); // only chunk 0 completed before abort
});
