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
  PgimPreloadError,
  PgimPreloader,
  parsePgimHeader,
  parsePgimLayout,
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
