import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const pgimMod = require('../web/runtime/pgim_preloader.js');
const orchestratorMod = require('../web/runtime/boot_orchestrator.js');

const {
  PGIM_MAGIC_BYTES,
  PGIM_VERSION,
  PGIM_HEADER_BYTES,
  PGIM_SECTION_ENTRY_BYTES,
  PGIM_PAGE_SIZE,
  PGIM_FLAGS,
  PGIM_SECTION_FLAGS,
  PGIM_PROT,
  DEFAULT_OPFS_CACHE_MODE,
  PgimPreloader,
  buildPgimOpfsFingerprint,
  buildManifestOpfsFingerprint,
  releaseBootBundleBytes,
} = pgimMod;

const {
  BOOT_PHASE,
  BootOrchestrator,
  transferPreloadPayloadToWorker,
} = orchestratorMod;

// ---------------------------------------------------------------------------------------------
// Synthetic PGIM v1 + mock OPFS & HTTP helpers
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

function buildPgim(seed = 1, { imageBase = 0x140000000n, textBytes = 0x1800, dataBytes = 0x1400 } = {}) {
  const sections = [
    {
      name: '<headers>',
      rva: 0,
      virtualSize: 0x400,
      protection: PGIM_PROT.READ,
      flags: PGIM_SECTION_FLAGS.HEADERS,
      data: filled(0x400, seed),
    },
    {
      name: '.text',
      rva: 0x1000,
      virtualSize: alignUp(textBytes, 0x1000),
      protection: PGIM_PROT.READ | PGIM_PROT.EXECUTE,
      data: filled(textBytes, seed + 1),
      characteristics: 0x60000020,
    },
    {
      name: '.data',
      rva: 0x1000 + alignUp(textBytes, 0x1000),
      virtualSize: alignUp(dataBytes, 0x1000),
      protection: PGIM_PROT.READ | PGIM_PROT.WRITE,
      data: filled(dataBytes, seed + 2),
      characteristics: 0xc0000040,
    },
  ];
  const dataRva = sections[2].rva;
  const sizeOfImage = dataRva + sections[2].virtualSize;
  const dir64 = [dataRva, dataRva + 0x10];
  const highlow = [dataRva + 0x20];
  const tablesEnd =
    PGIM_HEADER_BYTES + sections.length * PGIM_SECTION_ENTRY_BYTES + (dir64.length + highlow.length) * 4;
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
  view.setUint32(16, PGIM_FLAGS.HEADERS_MAPPED, true);
  view.setUint32(20, PGIM_PAGE_SIZE, true);
  view.setBigUint64(24, imageBase, true);
  view.setUint32(32, sizeOfImage, true);
  view.setUint32(36, 0x1000, true);
  view.setUint32(40, 0x1010, true);
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

function jsonBytes(obj) {
  return new TextEncoder().encode(JSON.stringify(obj));
}

function createMockNativeStorage(hooks = {}) {
  const dirs = new Map();
  const getOrCreateBucket = (name) => {
    if (!dirs.has(name)) dirs.set(name, new Map());
    return dirs.get(name);
  };
  const makeDirHandle = (dirName) => {
    const bucket = getOrCreateBucket(dirName);
    return {
      async getDirectoryHandle(subName, { create = false } = {}) {
        const key = `${dirName}/${subName}`;
        if (!dirs.has(key) && !create) {
          throw new DOMException(`Directory not found: ${subName}`, 'NotFoundError');
        }
        return makeDirHandle(key);
      },
      async getFileHandle(name, { create = false } = {}) {
        if (!bucket.has(name)) {
          if (!create) {
            throw new DOMException(`File not found: ${name}`, 'NotFoundError');
          }
          bucket.set(name, new Uint8Array(0));
        }
        return {
          name,
          async getFile() {
            const data = bucket.get(name) || new Uint8Array(0);
            return {
              size: data.byteLength,
              async arrayBuffer() {
                return data.slice().buffer;
              },
            };
          },
          async createWritable() {
            let pending = new Uint8Array(0);
            return {
              async write(chunk) {
                if (typeof hooks.onWrite === 'function') {
                  await hooks.onWrite(name, chunk);
                }
                if (chunk instanceof Uint8Array) {
                  if (chunk.buffer.detached) {
                    throw new TypeError('Cannot write detached ArrayBuffer to OPFS');
                  }
                  pending = chunk.slice();
                } else if (ArrayBuffer.isView(chunk)) {
                  pending = new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength).slice();
                } else if (chunk instanceof ArrayBuffer) {
                  pending = new Uint8Array(chunk).slice();
                }
              },
              async close() {
                bucket.set(name, pending);
              },
              async abort() {},
            };
          },
        };
      },
      async removeEntry(name) {
        if (!bucket.has(name)) {
          throw new DOMException(`File not found: ${name}`, 'NotFoundError');
        }
        bucket.delete(name);
      },
    };
  };

  let getDirectoryCalls = 0;
  const storageObj = {
    async getDirectory() {
      getDirectoryCalls += 1;
      return makeDirHandle('root');
    },
  };

  return {
    storageObj,
    dirs,
    get getDirectoryCalls() {
      return getDirectoryCalls;
    },
  };
}

function createMemoryOpfsStore(hooks = {}) {
  const entries = new Map();
  const reads = [];
  const writes = [];
  const deletes = [];
  return {
    entries,
    reads,
    writes,
    deletes,
    async read(key, meta) {
      reads.push({ key, meta });
      if (typeof hooks.onRead === 'function') {
        return hooks.onRead(key, meta, entries);
      }
      const val = entries.get(key);
      return val ? val.slice() : null;
    },
    async write(key, bytes, meta) {
      writes.push({ key, byteLength: bytes.byteLength, detached: Boolean(bytes.buffer && bytes.buffer.detached), meta });
      if (typeof hooks.onWrite === 'function') {
        return hooks.onWrite(key, bytes, meta, entries);
      }
      if (bytes.buffer && bytes.buffer.detached) {
        throw new TypeError('Cannot write detached buffer');
      }
      entries.set(key, new Uint8Array(bytes).slice());
      return true;
    },
    async delete(key) {
      deletes.push(key);
      if (typeof hooks.onDelete === 'function') {
        return hooks.onDelete(key, entries);
      }
      return entries.delete(key);
    },
  };
}

function createHttpServer(initialFiles, options = {}) {
  const { supportRange = true, headStatus = 200, supportConditional = true } = options;
  const files = new Map();
  for (const [url, entry] of Object.entries(initialFiles)) {
    files.set(url, {
      bytes: entry.bytes,
      etag: entry.etag !== undefined ? entry.etag : null,
      lastModified: entry.lastModified !== undefined ? entry.lastModified : null,
      omitHeadEtag: Boolean(entry.omitHeadEtag),
    });
  }
  const log = [];
  const fetchImpl = async (url, init = {}) => {
    const method = init.method || 'GET';
    const headersIn = init.headers || {};
    const range = headersIn.Range ?? headersIn.range ?? null;
    const ifNoneMatch = headersIn['If-None-Match'] ?? headersIn['if-none-match'] ?? null;
    const ifModifiedSince = headersIn['If-Modified-Since'] ?? headersIn['if-modified-since'] ?? null;
    log.push({ url, method, range, ifNoneMatch, ifModifiedSince });
    if (init.signal?.aborted) throw init.signal.reason;
    const entry = files.get(url);
    if (!entry) return new Response(null, { status: 404 });
    const { bytes, etag, lastModified, omitHeadEtag } = entry;

    if (method === 'HEAD') {
      if (headStatus !== 200) return new Response(null, { status: headStatus });
      const h = { 'content-length': String(bytes.byteLength), 'accept-ranges': supportRange ? 'bytes' : 'none' };
      if (etag !== null && !omitHeadEtag) h.etag = etag;
      if (lastModified !== null) h['last-modified'] = lastModified;
      return new Response(null, { status: 200, headers: h });
    }

    if (supportConditional && method === 'GET' && !range) {
      if (ifNoneMatch !== null && etag !== null && ifNoneMatch === etag) {
        const h = { etag };
        if (lastModified !== null) h['last-modified'] = lastModified;
        return new Response(null, { status: 304, headers: h });
      }
      if (ifNoneMatch === null && ifModifiedSince !== null && lastModified !== null && ifModifiedSince === lastModified) {
        return new Response(null, { status: 304, headers: { 'last-modified': lastModified } });
      }
    }

    if (range && supportRange) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(range);
      const start = Number(m[1]);
      if (start >= bytes.byteLength) return new Response(null, { status: 416 });
      const end = Math.min(Number(m[2]), bytes.byteLength - 1);
      const h = {
        'content-range': `bytes ${start}-${end}/${bytes.byteLength}`,
        'content-length': String(end - start + 1),
      };
      if (etag !== null) h.etag = etag;
      if (lastModified !== null) h['last-modified'] = lastModified;
      return new Response(bytes.slice(start, end + 1), { status: 206, headers: h });
    }

    const h = { 'content-length': String(bytes.byteLength) };
    if (etag !== null) h.etag = etag;
    if (lastModified !== null) h['last-modified'] = lastModified;
    return new Response(bytes.slice(), { status: 200, headers: h });
  };

  return {
    fetchImpl,
    log,
    setFile(url, entry) {
      files.set(url, {
        bytes: entry.bytes,
        etag: entry.etag !== undefined ? entry.etag : null,
        lastModified: entry.lastModified !== undefined ? entry.lastModified : null,
        omitHeadEtag: Boolean(entry.omitHeadEtag),
      });
    },
    count(method, url = null) {
      return log.filter((r) => r.method === method && (url === null || r.url === url)).length;
    },
  };
}

function hasBinaryBufferReference(value, seen = new Set()) {
  if (!value || typeof value !== 'object') return false;
  if (
    value instanceof ArrayBuffer ||
    (typeof SharedArrayBuffer !== 'undefined' && value instanceof SharedArrayBuffer) ||
    ArrayBuffer.isView(value)
  ) {
    return true;
  }
  if (seen.has(value)) return false;
  seen.add(value);
  for (const key of Object.getOwnPropertyNames(value)) {
    let child;
    try {
      child = value[key];
    } catch {
      continue;
    }
    if (hasBinaryBufferReference(child, seen)) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------------------------
// W375 Tests
// ---------------------------------------------------------------------------------------------

test('W375 PgimPreloader auto-enables native OPFS when navigator.storage.getDirectory is available and honors explicit opt-out', async () => {
  assert.equal(DEFAULT_OPFS_CACHE_MODE, 'auto');
  const image = buildPgim(10);
  const server = createHttpServer({ '/boot/guest.pgim': { bytes: image, etag: '"auto-v1"' } });

  // 1. Default in Node (no navigator.storage.getDirectory): gracefully disabled with zero errors.
  const nodePreloader = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096 });
  assert.equal(nodePreloader.opfsCache, 'auto');
  assert.equal(nodePreloader.opfsEnabled, false);
  assert.equal(nodePreloader.getStats().opfsEnabled, false);

  // 2. Auto-detects via opfsStorageObj.getDirectory without specifying opfsCache.
  const nativeMock = createMockNativeStorage();
  const autoCold = new PgimPreloader({
    fetchImpl: server.fetchImpl,
    opfsStorageObj: nativeMock.storageObj,
    chunkSize: 4096,
  });
  assert.equal(autoCold.opfsEnabled, true);
  assert.equal(autoCold.getStats().opfsEnabled, true);
  const coldRes = await autoCold.preloadGuestImage('/boot/guest.pgim');
  await autoCold.flushOpfsWrites();
  assert.equal(coldRes.opfsCacheHit, false);
  assert.equal(autoCold.getStats().opfsWrites, 1);
  assert.equal(autoCold.getStats().opfsMisses, 1);
  assert.equal(autoCold.getStats().opfsHitRatePct, 0);
  assert.ok(nativeMock.getDirectoryCalls >= 1);

  const getsAfterCold = server.count('GET', '/boot/guest.pgim');
  const autoWarm = new PgimPreloader({
    fetchImpl: server.fetchImpl,
    opfsStorageObj: nativeMock.storageObj,
    chunkSize: 4096,
  });
  const warmRes = await autoWarm.preloadGuestImage('/boot/guest.pgim');
  assert.equal(warmRes.opfsCacheHit, true);
  assert.deepEqual(warmRes.bytes, image);
  assert.equal(server.count('GET', '/boot/guest.pgim'), getsAfterCold, 'auto-enabled OPFS warm hit skips HTTP GET');
  assert.equal(autoWarm.getStats().opfsEnabled, true);
  assert.equal(autoWarm.getStats().opfsHits, 1);
  assert.equal(autoWarm.getStats().opfsHitRatePct, 100);
  assert.equal(autoWarm.getStats().bytesFromOpfs, image.byteLength);
  assert.equal(autoWarm.getStats().bytesFromNetwork, 0);

  // 3. Auto-detects globalThis.navigator.storage.getDirectory automatically (Chrome live path).
  const origNavDesc = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  const globalNative = createMockNativeStorage();
  Object.defineProperty(globalThis, 'navigator', {
    value: { storage: globalNative.storageObj },
    configurable: true,
    writable: true,
  });
  try {
    const chromeAuto = new PgimPreloader({ fetchImpl: server.fetchImpl, chunkSize: 4096 });
    assert.equal(chromeAuto.opfsEnabled, true);
    await chromeAuto.preloadGuestImage('/boot/guest.pgim');
    await chromeAuto.flushOpfsWrites();
    assert.equal(chromeAuto.getStats().opfsWrites, 1);

    // Explicit opt-out (`false` and `null`) strictly disables OPFS even when global navigator.storage exists.
    const optOutFalse = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: false });
    assert.equal(optOutFalse.opfsEnabled, false);
    assert.equal(optOutFalse.getStats().opfsEnabled, false);

    const optOutNull = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: null });
    assert.equal(optOutNull.opfsEnabled, false);
    assert.equal(optOutNull.getStats().opfsEnabled, false);
  } finally {
    if (origNavDesc) {
      Object.defineProperty(globalThis, 'navigator', origNavDesc);
    } else {
      delete globalThis.navigator;
    }
  }

  // 4. Restricted browser where navigator.storage getter throws SecurityError stays safe.
  const restrictedNavDesc = Object.getOwnPropertyDescriptor(globalThis, 'navigator');
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      get storage() {
        throw new DOMException('Blocked by sandbox', 'SecurityError');
      },
    },
    configurable: true,
    writable: true,
  });
  try {
    const restricted = new PgimPreloader({ fetchImpl: server.fetchImpl });
    assert.equal(restricted.opfsEnabled, false);
  } finally {
    if (restrictedNavDesc) {
      Object.defineProperty(globalThis, 'navigator', restrictedNavDesc);
    } else {
      delete globalThis.navigator;
    }
  }
});

test('W375 background OPFS write-through survives immediate ArrayBuffer detachment by Worker transfer and preserves W258 zero-retention', async () => {
  const image = buildPgim(33);
  const server = createHttpServer({ '/boot/guest.pgim': { bytes: image, etag: '"detach-v1"' } });

  let releaseWrite;
  let writeStarted = false;
  const writeGate = new Promise((resolve) => {
    releaseWrite = resolve;
  });
  const nativeMock = createMockNativeStorage({
    async onWrite(name) {
      if (name.endsWith('.pgim')) {
        writeStarted = true;
        await writeGate;
      }
    },
  });

  const preloader = new PgimPreloader({
    fetchImpl: server.fetchImpl,
    opfsStorageObj: nativeMock.storageObj,
    chunkSize: 4096,
  });

  const bundle = await preloader.preloadBootBundle({ pgimUrl: '/boot/guest.pgim' });
  assert.equal(bundle.opfsCacheHit, false);
  assert.equal(preloader.getStats().opfsWrites, 0, 'OPFS write is still gated in background');

  // Simulate immediate zero-copy Worker transfer that DETACHES result.buffer BEFORE OPFS write runs!
  const origPgimBuffer = bundle.buffer;
  assert.equal(origPgimBuffer.detached, false);
  const transferredBufs = [];
  const fakeWorker = {
    postMessage(msg, transferList = []) {
      for (const item of transferList) {
        if (item instanceof ArrayBuffer) {
          transferredBufs.push(structuredClone(item, { transfer: [item] }));
        }
      }
    },
  };
  const handoff = transferPreloadPayloadToWorker(fakeWorker, { type: 'init', bundle }, { releaseTarget: bundle });
  assert.ok(handoff.transferredBuffers >= 1);
  assert.equal(origPgimBuffer.detached, true, 'PGIM ArrayBuffer must be detached before OPFS write completes');
  assert.equal(bundle.buffer, null);
  assert.equal(bundle.pgimBytes, null);
  assert.ok(transferredBufs.some((b) => b.byteLength === image.byteLength));

  // Now unblock the gated OPFS write and verify it writes the full intact image despite detachment!
  releaseWrite();
  await preloader.flushOpfsWrites();
  assert.equal(writeStarted, true);
  assert.equal(preloader.getStats().opfsWrites, 1, 'opfsWrites updates cleanly after run.complete()');
  assert.equal(hasBinaryBufferReference(preloader), false, 'W258 zero-retention holds after write');

  // Warm boot from the same OPFS storage reads intact image with 0 HTTP GETs!
  const getsBeforeWarm = server.count('GET', '/boot/guest.pgim');
  const warmPreloader = new PgimPreloader({
    fetchImpl: server.fetchImpl,
    opfsStorageObj: nativeMock.storageObj,
    chunkSize: 4096,
  });
  const warmBundle = await warmPreloader.preloadBootBundle({ pgimUrl: '/boot/guest.pgim' });
  assert.equal(warmBundle.opfsCacheHit, true);
  assert.deepEqual(warmBundle.pgimBytes, image);
  assert.equal(server.count('GET', '/boot/guest.pgim'), getsBeforeWarm);
  assert.equal(warmPreloader.getStats().opfsHitRatePct, 100);
  assert.equal(releaseBootBundleBytes(warmBundle).releasedBytes, image.byteLength);
});

test('W375 guest.pgim invalidates stale OPFS entry on Last-Modified change when ETag is absent', async () => {
  const imgV1 = buildPgim(50, { textBytes: 0x1800, dataBytes: 0x1400 });
  const imgV2 = buildPgim(99, { textBytes: 0x1800, dataBytes: 0x1400 });
  assert.equal(imgV1.byteLength, imgV2.byteLength, 'same byteLength isolates Last-Modified validator');
  assert.notDeepEqual(imgV1, imgV2);

  const server = createHttpServer({
    '/boot/guest.pgim': {
      bytes: imgV1,
      etag: null,
      lastModified: 'Thu, 01 Jan 2026 00:00:00 GMT',
    },
  });
  const store = createMemoryOpfsStore();

  // Cold boot #1 (no ETag, has Last-Modified).
  const p1 = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const r1 = await p1.preloadGuestImage('/boot/guest.pgim');
  await p1.flushOpfsWrites();
  assert.equal(r1.opfsCacheHit, false);
  assert.deepEqual(r1.bytes, imgV1);

  // Warm boot #2 (same Last-Modified + same size -> OPFS hit, 0 GETs).
  const getsAfterCold = server.count('GET');
  const p2 = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const r2 = await p2.preloadGuestImage('/boot/guest.pgim');
  assert.equal(r2.opfsCacheHit, true);
  assert.deepEqual(r2.bytes, imgV1);
  assert.equal(server.count('GET'), getsAfterCold);

  // Rebuild with identical size but newer Last-Modified -> MUST invalidate and refetch!
  server.setFile('/boot/guest.pgim', {
    bytes: imgV2,
    etag: null,
    lastModified: 'Thu, 01 Jan 2026 00:05:00 GMT',
  });
  const p3 = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const r3 = await p3.preloadGuestImage('/boot/guest.pgim', { verifyChecksums: true });
  await p3.flushOpfsWrites();
  assert.equal(r3.opfsCacheHit, false);
  assert.deepEqual(r3.bytes, imgV2);
  assert.equal(p3.getStats().opfsInvalidations, 1);
  assert.equal(p3.getStats().opfsMisses, 1);
  assert.equal(p3.getStats().opfsWrites, 1);

  // Warm boot #4 hits v2 with 0 GETs.
  const getsAfterV2 = server.count('GET');
  const p4 = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const r4 = await p4.preloadGuestImage('/boot/guest.pgim');
  assert.equal(r4.opfsCacheHit, true);
  assert.deepEqual(r4.bytes, imgV2);
  assert.equal(server.count('GET'), getsAfterV2);
});

test('W375 caches program-manifest.json and boot-manifest.json in versioned OPFS with HEAD, 304 Not Modified, invalidation, and QuotaExceededError resilience', async () => {
  const depImg = buildPgim(11, { imageBase: 0x100000000n });
  const mainImg = buildPgim(12, { imageBase: 0x140000000n });
  const manifestV1 = jsonBytes({
    schema: 'playgta6-recomp-program',
    schema_version: 1,
    modules: [
      { module: 'dep', role: 'dll' },
      { module: 'main', role: 'exe' },
    ],
  });
  // Second manifest with a different module list (validates rotation).
  const manifestV2 = jsonBytes({
    schema: 'playgta6-recomp-program',
    schema_version: 1,
    modules: [{ module: 'main', role: 'exe' }],
  });

  const server = createHttpServer({
    '/prog/program-manifest.json': { bytes: manifestV1, etag: '"pm-v1"' },
    '/prog/dep/image/guest.pgim': { bytes: depImg, etag: '"dep-v1"' },
    '/prog/main/image/guest.pgim': { bytes: mainImg, etag: '"main-v1"' },
  });
  const store = createMemoryOpfsStore();

  // Fingerprint sanity check.
  const mfp1 = buildManifestOpfsFingerprint('/prog/program-manifest.json?sig=1', '"pm-v1"', manifestV1.byteLength, 1);
  const mfp2 = buildManifestOpfsFingerprint('/prog/program-manifest.json', '"pm-v1"', manifestV1.byteLength, 2);
  assert.match(mfp1.dataKey, /^manifest-v1-.*\.json$/);
  assert.notEqual(mfp1.metaKey, mfp2.metaKey, 'cache version namespaces manifest keys');

  // 1. Cold boot persists manifest + both modules.
  const p1 = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const r1 = await p1.preloadProgramModules('/prog/program-manifest.json');
  await p1.flushOpfsWrites();
  assert.equal(r1.manifestOpfsCacheHit, false);
  assert.equal(r1.moduleCount, 2);
  assert.equal(p1.getStats().opfsWrites, 3);
  assert.equal(p1.getStats().opfsMisses, 3);
  assert.equal(p1.getStats().bytesFromNetwork, manifestV1.byteLength + depImg.byteLength + mainImg.byteLength);

  // 2. Warm boot via HEAD validation serves manifest + both modules from OPFS with 0 GETs!
  const getsAfterCold = server.count('GET');
  const p2 = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const r2 = await p2.preloadProgramModules('/prog/program-manifest.json');
  assert.equal(r2.manifestOpfsCacheHit, true);
  assert.equal(r2.modulesByName.dep.opfsCacheHit, true);
  assert.equal(r2.modulesByName.main.opfsCacheHit, true);
  assert.equal(server.count('GET'), getsAfterCold);
  assert.equal(p2.getStats().opfsHits, 3);
  assert.equal(p2.getStats().opfsHitRatePct, 100);
  assert.equal(p2.getStats().bytesFromOpfs, manifestV1.byteLength + depImg.byteLength + mainImg.byteLength);
  assert.equal(p2.getStats().bytesFromNetwork, 0);

  // 3. Warm boot when HEAD returns 405 uses conditional If-None-Match -> 304 Not Modified!
  const server405 = createHttpServer(
    {
      '/prog/program-manifest.json': { bytes: manifestV1, etag: '"pm-v1"' },
      '/prog/dep/image/guest.pgim': { bytes: depImg, etag: '"dep-v1"' },
      '/prog/main/image/guest.pgim': { bytes: mainImg, etag: '"main-v1"' },
    },
    { headStatus: 405, supportConditional: true }
  );
  const p304 = new PgimPreloader({ fetchImpl: server405.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const r304 = await p304.preloadProgramModules('/prog/program-manifest.json');
  assert.equal(r304.manifestOpfsCacheHit, true);
  assert.equal(r304.modulesByName.dep.opfsCacheHit, true);
  assert.equal(r304.modulesByName.main.opfsCacheHit, true);
  const manifestGetReq = server405.log.find((r) => r.method === 'GET' && r.url === '/prog/program-manifest.json');
  assert.ok(manifestGetReq, 'conditional GET should be sent when HEAD returns 405');
  assert.equal(manifestGetReq.ifNoneMatch, '"pm-v1"');
  assert.equal(p304.getStats().opfsHits, 3);
  assert.equal(p304.getStats().bytesFromNetwork, 0);

  // 4. Warm boot with opfsValidateWithHead: false issues ZERO HTTP requests total!
  const reqsBeforeOffline = server.log.length;
  const pOffline = new PgimPreloader({
    fetchImpl: server.fetchImpl,
    opfsCache: store,
    opfsValidateWithHead: false,
  });
  const rOffline = await pOffline.preloadProgramModules('/prog/program-manifest.json');
  assert.equal(rOffline.manifestOpfsCacheHit, true);
  assert.equal(server.log.length, reqsBeforeOffline, 'zero network requests when opfsValidateWithHead=false');
  assert.equal(pOffline.getStats().opfsHits, 3);
  assert.equal(pOffline.getStats().opfsHitRatePct, 100);

  // 5. ETag rotation on program-manifest.json invalidates stale manifest and serves v2!
  server.setFile('/prog/program-manifest.json', { bytes: manifestV2, etag: '"pm-v2"' });
  const pRot = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const rRot = await pRot.preloadProgramModules('/prog/program-manifest.json');
  await pRot.flushOpfsWrites();
  assert.equal(rRot.manifestOpfsCacheHit, false);
  assert.equal(rRot.moduleCount, 1);
  assert.equal(rRot.modules[0].name, 'main');
  assert.equal(rRot.modules[0].opfsCacheHit, true);
  assert.equal(pRot.getStats().opfsInvalidations, 1);
  assert.equal(pRot.getStats().opfsMisses, 1);
  assert.equal(pRot.getStats().opfsHits, 1);
  assert.equal(pRot.getStats().opfsHitRatePct, 50);
  assert.equal(pRot.getStats().opfsWrites, 1);

  // 6. Corrupt manifest JSON in OPFS self-heals without failing boot!
  const mfpV2 = buildManifestOpfsFingerprint('/prog/program-manifest.json', '"pm-v2"', manifestV2.byteLength, 1);
  store.entries.set(mfpV2.dataKey, new TextEncoder().encode('{broken-json-in-opfs'));
  const pHeal = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: store, chunkSize: 4096 });
  const rHeal = await pHeal.preloadProgramModules('/prog/program-manifest.json');
  await pHeal.flushOpfsWrites();
  assert.equal(rHeal.manifestOpfsCacheHit, false);
  assert.equal(rHeal.moduleCount, 1);
  assert.equal(pHeal.getStats().opfsInvalidations, 1);
  assert.equal(pHeal.getStats().opfsWrites, 1);

  // 7. boot-manifest.json fallback path also caches & hits OPFS cleanly!
  const bootManifestBytes = jsonBytes({ module: 'main', pgim: 'main/image/guest.pgim' });
  const fbServer = createHttpServer({
    '/fb/boot-manifest.json': { bytes: bootManifestBytes, etag: '"bm-v1"' },
    '/fb/main/image/guest.pgim': { bytes: mainImg, etag: '"fb-main-v1"' },
  });
  const fbStore = createMemoryOpfsStore();
  const fbCold = new PgimPreloader({ fetchImpl: fbServer.fetchImpl, opfsCache: fbStore, chunkSize: 4096 });
  const fbRes1 = await fbCold.preloadProgramModules('/fb/program-manifest.json');
  await fbCold.flushOpfsWrites();
  assert.equal(fbRes1.source, 'boot-manifest');
  assert.equal(fbRes1.manifestOpfsCacheHit, false);

  const fbGetsBeforeWarm = fbServer.count('GET', '/fb/boot-manifest.json');
  const fbWarm = new PgimPreloader({ fetchImpl: fbServer.fetchImpl, opfsCache: fbStore, chunkSize: 4096 });
  const fbRes2 = await fbWarm.preloadProgramModules('/fb/program-manifest.json');
  assert.equal(fbRes2.source, 'boot-manifest');
  assert.equal(fbRes2.manifestOpfsCacheHit, true);
  assert.equal(fbRes2.mainModule.opfsCacheHit, true);
  assert.equal(fbServer.count('GET', '/fb/boot-manifest.json'), fbGetsBeforeWarm);
  assert.equal(fbWarm.getStats().opfsHits, 2);
  assert.equal(fbWarm.getStats().opfsHitRatePct, 100);

  // 8. QuotaExceededError on manifest write never fails boot!
  const quotaStore = createMemoryOpfsStore({
    async onWrite() {
      throw new DOMException('Disk quota exceeded', 'QuotaExceededError');
    },
  });
  const pQuota = new PgimPreloader({ fetchImpl: server.fetchImpl, opfsCache: quotaStore, chunkSize: 4096 });
  const rQuota = await pQuota.preloadProgramModules('/prog/program-manifest.json');
  await pQuota.flushOpfsWrites();
  assert.equal(rQuota.moduleCount, 1);
  assert.equal(pQuota.getStats().opfsWrites, 0);
});

test('W375 BootOrchestrator forwards programManifestUrl into preloadBootBundle and exposes sanitized OPFS warm-boot telemetry', async () => {
  const depImg = buildPgim(70, { imageBase: 0x100000000n });
  const mainImg = buildPgim(71, { imageBase: 0x140000000n });
  const manifestBytes = jsonBytes({
    schema: 'playgta6-recomp-program',
    schema_version: 1,
    modules: [
      { module: 'dep', role: 'dll' },
      { module: 'main', role: 'exe' },
    ],
  });

  const statusPayload = {
    ok: true,
    coopCoep: true,
    hasWasmBinary: true,
    runtimeWasm: '/dist/runtime.wasm',
    programManifestUrl: '/boot/program-manifest.json',
    pgimUrl: null,
    gameBase: '/game/',
  };

  const assetServer = createHttpServer({
    '/boot/program-manifest.json': { bytes: manifestBytes, etag: '"orch-pm-v1"' },
    '/boot/dep/image/guest.pgim': { bytes: depImg, etag: '"orch-dep-v1"' },
    '/boot/main/image/guest.pgim': { bytes: mainImg, etag: '"orch-main-v1"' },
  });

  const combinedFetch = async (url, init) => {
    if (url === '/api/status') {
      return {
        ok: true,
        status: 200,
        async json() {
          return statusPayload;
        },
      };
    }
    return assetServer.fetchImpl(url, init);
  };

  const nativeMock = createMockNativeStorage();

  // Cold boot through BootOrchestrator with real PgimPreloader (auto OPFS) + buffer release after launch!
  let coldObservedModules = 0;
  const coldPreloader = new PgimPreloader({
    fetchImpl: combinedFetch,
    opfsStorageObj: nativeMock.storageObj,
    chunkSize: 4096,
  });
  const coldOrch = new BootOrchestrator({
    fetchImpl: combinedFetch,
    preloader: coldPreloader,
    releasePreloadBuffers: true,
    onLaunchRuntime: async (payload) => {
      coldObservedModules = payload.bundle.moduleCount;
      // Simulate zero-copy transfer detaching ArrayBuffers right inside onLaunchRuntime!
      const fakeWorker = {
        postMessage(_msg, transferList = []) {
          for (const item of transferList) {
            if (item instanceof ArrayBuffer) {
              structuredClone(item, { transfer: [item] });
            }
          }
        },
      };
      coldOrch.transferBundleToWorker(fakeWorker, { type: 'init' });
    },
  });

  const coldStart = await coldOrch.start();
  assert.equal(coldStart.ok, true);
  assert.equal(coldStart.phase, BOOT_PHASE.LAUNCHING_RUNTIME);
  assert.equal(coldObservedModules, 2);

  await coldOrch.flushOpfsWrites();
  const coldStats = coldOrch.getStats();
  const coldDiag = coldOrch.getDiagnosticSnapshot();

  assert.equal(coldStats.opfsEnabled, true);
  assert.equal(coldStats.opfsHits, 0);
  assert.equal(coldStats.opfsMisses, 3);
  assert.equal(coldStats.opfsWrites, 3, 'all 3 background OPFS writes succeeded despite immediate Worker transfer + release');
  assert.equal(coldStats.opfsInvalidations, 0);
  assert.equal(coldStats.bytesFromOpfs, 0);
  assert.equal(coldStats.bytesFromNetwork, manifestBytes.byteLength + depImg.byteLength + mainImg.byteLength);
  assert.equal(coldStats.opfsHitRatePct, 0);

  assert.equal(coldDiag.opfsEnabled, true);
  assert.equal(coldDiag.opfsWrites, 3);
  assert.equal(coldDiag.bytesFromNetwork, coldStats.bytesFromNetwork);
  assert.equal(coldDiag.opfsHitRatePct, 0);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(coldStats)));
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(coldDiag)));

  // Warm boot #2 through a fresh BootOrchestrator + fresh PgimPreloader: 100% OPFS hit rate, 0 GETs!
  const getsAfterCold = assetServer.count('GET');
  const warmPreloader = new PgimPreloader({
    fetchImpl: combinedFetch,
    opfsStorageObj: nativeMock.storageObj,
    chunkSize: 4096,
  });
  let warmBundleMeta = null;
  const warmOrch = new BootOrchestrator({
    fetchImpl: combinedFetch,
    preloader: warmPreloader,
    onLaunchRuntime: async (payload) => {
      warmBundleMeta = {
        opfsCacheHit: payload.bundle.opfsCacheHit,
        manifestOpfsCacheHit: payload.bundle.manifestOpfsCacheHit,
        moduleCount: payload.bundle.moduleCount,
        mainBytes: payload.bundle.pgimBytes.slice(),
      };
    },
  });

  const warmStart = await warmOrch.start();
  assert.equal(warmStart.ok, true);
  assert.equal(warmBundleMeta.opfsCacheHit, true);
  assert.equal(warmBundleMeta.manifestOpfsCacheHit, true);
  assert.equal(warmBundleMeta.moduleCount, 2);
  assert.deepEqual(warmBundleMeta.mainBytes, mainImg);
  assert.equal(assetServer.count('GET'), getsAfterCold, 'warm boot issues zero HTTP GETs for manifest or modules');

  const warmStats = warmOrch.getStats();
  const warmDiag = warmOrch.getDiagnosticSnapshot();
  assert.equal(warmStats.opfsEnabled, true);
  assert.equal(warmStats.opfsHits, 3);
  assert.equal(warmStats.opfsMisses, 0);
  assert.equal(warmStats.opfsWrites, 0);
  assert.equal(warmStats.opfsInvalidations, 0);
  assert.equal(warmStats.bytesFromOpfs, manifestBytes.byteLength + depImg.byteLength + mainImg.byteLength);
  assert.equal(warmStats.bytesFromNetwork, 0);
  assert.equal(warmStats.opfsHitRatePct, 100);

  assert.equal(warmDiag.opfsEnabled, true);
  assert.equal(warmDiag.opfsHits, 3);
  assert.equal(warmDiag.bytesFromOpfs, warmStats.bytesFromOpfs);
  assert.equal(warmDiag.bytesFromNetwork, 0);
  assert.equal(warmDiag.opfsHitRatePct, 100);
});

test('W375 BootOrchestrator forwards programManifestUrl alongside pgimUrl and omits it when absent', async () => {
  const seenCalls = [];
  const spyPreloader = {
    async preloadBootBundle(args) {
      seenCalls.push(args);
      return {
        pgim: { header: { magic: 'PGIM' }, bytesLoaded: 64 },
        shaderPackBytes: null,
        reflectionManifest: null,
        bytesLoaded: 64,
      };
    },
    getStats() {
      return {
        opfsEnabled: true,
        opfsHits: 2,
        opfsMisses: 1,
        opfsWrites: 1,
        opfsInvalidations: 0,
        bytesFromOpfs: 120000,
        bytesFromNetwork: 4096,
      };
    },
  };

  // Case 1: Both pgimUrl and programManifestUrl present -> forwards programManifestUrl!
  const orchWithManifest = new BootOrchestrator({
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      async json() {
        return {
          ok: true,
          hasWasmBinary: true,
          pgimUrl: '/boot/guest.pgim',
          programManifestUrl: '/boot/program-manifest.json',
        };
      },
    }),
    preloader: spyPreloader,
    onLaunchRuntime: async () => {},
  });
  await orchWithManifest.start();
  assert.equal(seenCalls.length, 1);
  assert.equal(seenCalls[0].pgimUrl, '/boot/guest.pgim');
  assert.equal(seenCalls[0].programManifestUrl, '/boot/program-manifest.json');
  const stats = orchWithManifest.getStats();
  assert.equal(stats.opfsEnabled, true);
  assert.equal(stats.opfsHits, 2);
  assert.equal(stats.opfsMisses, 1);
  assert.ok(Math.abs(stats.opfsHitRatePct - (200 / 3)) < 1e-6);

  // Case 2: Only pgimUrl present -> programManifestUrl property is NOT added!
  const orchWithoutManifest = new BootOrchestrator({
    fetchImpl: async () => ({
      ok: true,
      status: 200,
      async json() {
        return {
          ok: true,
          hasWasmBinary: true,
          pgimUrl: '/boot/guest.pgim',
        };
      },
    }),
    preloader: spyPreloader,
    onLaunchRuntime: async () => {},
  });
  await orchWithoutManifest.start();
  assert.equal(seenCalls.length, 2);
  assert.equal('programManifestUrl' in seenCalls[1], false);
});
