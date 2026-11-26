import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const ioWorker = require('../web/workers/io_worker.js');
const rangeCacheMod = require('../web/workers/range_cache.js');
const opfsCacheMod = require('../web/storage/opfs_cache.js');
const pgimPreloaderMod = require('../web/runtime/pgim_preloader.js');
const httpfsMailboxMod = require('../web/workers/httpfs_mailbox.js');

const {
  ImmutableHttpfsCache,
  IoWorkerController,
  buildFileFingerprint,
  opfsL2RangeFileName,
} = ioWorker;
const { RangeCache } = rangeCacheMod;
const { OpfsChunkCache, makeChunkKey } = opfsCacheMod;
const {
  PgimPreloader,
  preloadProgramModules,
  PGIM_MAGIC_BYTES,
  PGIM_VERSION,
  PGIM_HEADER_BYTES,
  PGIM_SECTION_ENTRY_BYTES,
  PGIM_PAGE_SIZE,
  PGIM_FLAGS,
  PGIM_PROT,
  pgimFnv1a64,
} = pgimPreloaderMod;
const { createHttpfsMailboxBuffer } = httpfsMailboxMod;

function buildMinimalPgimBytes(seed = 1) {
  const secData = makePatternBytes(0x1000, seed);
  const dataOffset = PGIM_PAGE_SIZE;
  const fileSize = dataOffset + secData.byteLength;
  const out = new Uint8Array(fileSize);
  const view = new DataView(out.buffer);
  out.set(PGIM_MAGIC_BYTES, 0);
  view.setUint32(4, PGIM_VERSION, true);
  view.setUint32(8, PGIM_HEADER_BYTES, true);
  view.setUint32(12, PGIM_SECTION_ENTRY_BYTES, true);
  view.setUint32(16, 0, true);
  view.setUint32(20, PGIM_PAGE_SIZE, true);
  view.setBigUint64(24, 0x140000000n, true);
  view.setUint32(32, 0x3000, true);
  view.setUint32(36, 0x1000, true);
  view.setUint32(40, 0x1000, true);
  view.setUint32(44, 1, true); // 1 section
  view.setUint32(48, 0, true); // dir64
  view.setUint32(52, 0, true); // highlow
  view.setBigUint64(56, BigInt(fileSize), true);

  const at = PGIM_HEADER_BYTES;
  out.set(new TextEncoder().encode('.text'), at);
  view.setUint32(at + 16, 0x1000, true);
  view.setUint32(at + 20, secData.byteLength, true);
  view.setUint32(at + 24, PGIM_PROT.READ | PGIM_PROT.EXECUTE, true);
  view.setUint32(at + 28, 0, true);
  view.setBigUint64(at + 32, BigInt(dataOffset), true);
  view.setBigUint64(at + 40, BigInt(secData.byteLength), true);
  view.setBigUint64(at + 48, pgimFnv1a64(secData), true);
  view.setUint32(at + 56, 0x60000020, true);
  view.setUint32(at + 60, 0, true);
  out.set(secData, dataOffset);
  return out;
}

function makePatternBytes(length, seed = 1) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 31) + seed * 17 + (i >>> 8)) & 0xff;
  }
  return out;
}

/**
 * In-memory OPFS FileSystemDirectoryHandle mock with full instrumentation:
 * - Supports keys()/entries(), getFileHandle({ create }), removeEntry()
 * - Enforces single active SyncAccessHandle lock per file (throws NoModificationAllowedError
 *   on concurrent lock collision just like Chromium OPFS!)
 * - Supports createWritable() fallback and optional async write delay
 */
function createInstrumentedOpfsDir(options = {}) {
  const files = new Map(); // name -> Uint8Array
  const activeSyncLocks = new Set();
  let syncDelayMs = Number.isFinite(options.syncDelayMs) ? options.syncDelayMs : 0;
  let writableDelayMs = Number.isFinite(options.writableDelayMs) ? options.writableDelayMs : 0;
  const allowSyncHandle = options.allowSyncHandle !== false;
  const throwSyncError = options.throwSyncError || null;

  const stats = {
    keysCalls: 0,
    entriesCalls: 0,
    getFileHandleCreateFalse: 0,
    getFileHandleCreateTrue: 0,
    syncHandleOpens: 0,
    syncHandleLockCollisions: 0,
    writableOpens: 0,
    getFileReads: 0,
    removeEntryCalls: 0,
  };

  function makeNotFound(name) {
    const err = new Error(`OPFS file not found: ${name}`);
    err.name = 'NotFoundError';
    return err;
  }

  function makeLockError(name) {
    const err = new Error(`OPFS SyncAccessHandle already open for ${name}`);
    err.name = 'NoModificationAllowedError';
    return err;
  }

  const dirHandle = {
    files,
    stats,
    setWriteDelay(ms) {
      syncDelayMs = ms;
      writableDelayMs = ms;
    },
    async *keys() {
      stats.keysCalls += 1;
      for (const name of Array.from(files.keys())) {
        yield name;
      }
    },
    async *entries() {
      stats.entriesCalls += 1;
      for (const name of Array.from(files.keys())) {
        const snap = files.get(name) || new Uint8Array(0);
        yield [
          name,
          {
            kind: 'file',
            name,
            async getFile() {
              const cur = (files.get(name) || snap).slice();
              return {
                size: cur.byteLength,
                lastModified: Date.now(),
                async arrayBuffer() {
                  return cur.buffer.slice(cur.byteOffset, cur.byteOffset + cur.byteLength);
                },
              };
            },
          },
        ];
      }
    },
    async getFileHandle(name, opts = {}) {
      const create = Boolean(opts && opts.create);
      if (create) {
        stats.getFileHandleCreateTrue += 1;
        if (!files.has(name)) {
          files.set(name, new Uint8Array(0));
        }
      } else {
        stats.getFileHandleCreateFalse += 1;
        if (!files.has(name)) {
          throw makeNotFound(name);
        }
      }

      const handle = {
        kind: 'file',
        name,
        async getFile() {
          stats.getFileReads += 1;
          const snap = (files.get(name) || new Uint8Array(0)).slice();
          return {
            size: snap.byteLength,
            async arrayBuffer() {
              return snap.buffer.slice(snap.byteOffset, snap.byteOffset + snap.byteLength);
            },
          };
        },
        async createWritable() {
          stats.writableOpens += 1;
          let staged = new Uint8Array(0);
          return {
            async write(data) {
              if (writableDelayMs > 0) {
                await new Promise((r) => setTimeout(r, writableDelayMs));
              }
              if (data instanceof Uint8Array) {
                if (data.buffer && data.buffer.detached) {
                  throw new TypeError('Cannot write detached ArrayBufferView');
                }
                staged = data.slice();
              } else if (data instanceof ArrayBuffer) {
                if (data.detached) {
                  throw new TypeError('Cannot write detached ArrayBuffer');
                }
                staged = new Uint8Array(data.slice(0));
              } else if (typeof data === 'string') {
                staged = new TextEncoder().encode(data);
              } else {
                staged = new Uint8Array(0);
              }
            },
            async close() {
              files.set(name, staged);
            },
            async abort() {},
          };
        },
      };

      if (allowSyncHandle) {
        handle.createSyncAccessHandle = async () => {
          stats.syncHandleOpens += 1;
          if (throwSyncError) {
            throw throwSyncError;
          }
          if (activeSyncLocks.has(name)) {
            stats.syncHandleLockCollisions += 1;
            throw makeLockError(name);
          }
          activeSyncLocks.add(name);
          if (syncDelayMs > 0) {
            await new Promise((r) => setTimeout(r, syncDelayMs));
          }
          let cur = (files.get(name) || new Uint8Array(0)).slice();
          let closed = false;
          return {
            getSize() {
              return cur.byteLength;
            },
            truncate(newLen) {
              const next = new Uint8Array(Math.max(0, newLen));
              next.set(cur.subarray(0, Math.min(cur.byteLength, next.byteLength)), 0);
              cur = next;
            },
            write(buf, writeOpts = {}) {
              const view =
                buf instanceof Uint8Array
                  ? buf
                  : buf instanceof ArrayBuffer
                    ? new Uint8Array(buf)
                    : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
              if (view.buffer && view.buffer.detached) {
                throw new TypeError('Cannot write detached ArrayBufferView');
              }
              const at = Number.isInteger(writeOpts.at) ? writeOpts.at : 0;
              const end = at + view.byteLength;
              if (end > cur.byteLength) {
                const grown = new Uint8Array(end);
                grown.set(cur, 0);
                cur = grown;
              }
              cur.set(view, at);
              return view.byteLength;
            },
            read(buf, readOpts = {}) {
              const view =
                buf instanceof Uint8Array
                  ? buf
                  : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
              const at = Number.isInteger(readOpts.at) ? readOpts.at : 0;
              const take = Math.max(0, Math.min(view.byteLength, cur.byteLength - at));
              view.set(cur.subarray(at, at + take), 0);
              return take;
            },
            flush() {
              files.set(name, cur.slice());
            },
            close() {
              if (!closed) {
                closed = true;
                files.set(name, cur.slice());
                activeSyncLocks.delete(name);
              }
            },
          };
        };
      }

      return handle;
    },
    async removeEntry(name) {
      stats.removeEntryCalls += 1;
      if (!files.has(name)) {
        throw makeNotFound(name);
      }
      files.delete(name);
    },
  };

  return dirHandle;
}

function createMockOpfsStorage(rootSubdirs = new Map()) {
  const getDirectoryCalls = { count: 0 };
  return {
    getDirectoryCalls,
    rootSubdirs,
    async getDirectory() {
      getDirectoryCalls.count += 1;
      return {
        async getDirectoryHandle(dirName) {
          if (!rootSubdirs.has(dirName)) {
            rootSubdirs.set(dirName, createInstrumentedOpfsDir());
          }
          return rootSubdirs.get(dirName);
        },
      };
    },
  };
}

function createMockHttpServer(entriesMap, options = {}) {
  const requests = [];
  const headRequests = [];
  const getRequests = [];
  const headStatusOverride = options.headStatusOverride || null;

  function makeHeaders(obj) {
    const map = new Map();
    for (const [k, v] of Object.entries(obj)) {
      if (v !== null && v !== undefined) {
        map.set(k.toLowerCase(), String(v));
      }
    }
    return {
      get(name) {
        const val = map.get(String(name).toLowerCase());
        return val === undefined ? null : val;
      },
    };
  }

  async function fetchImpl(url, init = {}) {
    const method = String((init && init.method) || 'GET').toUpperCase();
    const rangeHeader =
      init && init.headers
        ? init.headers.Range || init.headers.range || null
        : null;
    const rec = { url, method, range: rangeHeader };
    requests.push(rec);
    if (method === 'HEAD') {
      headRequests.push(rec);
    } else {
      getRequests.push(rec);
    }

    const baseUrl = String(url).split('?')[0];
    const entry =
      entriesMap.get(url) ||
      entriesMap.get(baseUrl) ||
      entriesMap.get(baseUrl.toLowerCase()) ||
      null;

    if (!entry) {
      return {
        ok: false,
        status: 404,
        headers: makeHeaders({}),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const { bytes, etag } = entry;
    const total = bytes.byteLength;

    if (method === 'HEAD') {
      if (headStatusOverride !== null) {
        return {
          ok: false,
          status: headStatusOverride,
          headers: makeHeaders({}),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      return {
        ok: true,
        status: 200,
        headers: makeHeaders({
          'Content-Length': String(total),
          'Accept-Ranges': 'bytes',
          ...(etag ? { ETag: etag } : {}),
        }),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (rangeHeader && /^bytes=(\d+)-(\d*)$/.test(rangeHeader)) {
      const m = rangeHeader.match(/^bytes=(\d+)-(\d*)$/);
      const start = Number(m[1]);
      const end = m[2] ? Math.min(Number(m[2]), total - 1) : total - 1;
      if (start >= total) {
        return {
          ok: false,
          status: 416,
          headers: makeHeaders({
            'Content-Range': `bytes */${total}`,
            ...(etag ? { ETag: etag } : {}),
          }),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      const slice = bytes.slice(start, end + 1);
      return {
        ok: true,
        status: 206,
        headers: makeHeaders({
          'Content-Range': `bytes ${start}-${end}/${total}`,
          'Content-Length': String(slice.byteLength),
          'Accept-Ranges': 'bytes',
          ...(etag ? { ETag: etag } : {}),
        }),
        async arrayBuffer() {
          return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
        },
      };
    }

    const full = bytes.slice();
    return {
      ok: true,
      status: 200,
      headers: makeHeaders({
        'Content-Length': String(total),
        'Accept-Ranges': 'bytes',
        ...(etag ? { ETag: etag } : {}),
      }),
      async arrayBuffer() {
        return full.buffer.slice(full.byteOffset, full.byteOffset + full.byteLength);
      },
    };
  }

  return { fetchImpl, requests, headRequests, getRequests };
}

test('W369: live-boot simulation populates OPFS L2 on cold boot in O(1) and hits OPFS L2 on warm boot across 8 mailbox lanes without pre-seeded fingerprints', async () => {
  const slabSize = 4096;
  const coreBvh4 = makePatternBytes(slabSize * 8, 0x33);
  const shadersRpf = makePatternBytes(slabSize * 4, 0x77);
  const serverEntries = new Map([
    ['/game/core_bvh4.rpf', { bytes: coreBvh4, etag: '"core-bvh4-v1"' }],
    ['/game/shaders.rpf', { bytes: shadersRpf, etag: '"shaders-v1"' }],
  ]);

  const sharedOpfsDir = createInstrumentedOpfsDir();

  // --- COLD BOOT ---
  const coldServer = createMockHttpServer(serverEntries);
  const coldController = new IoWorkerController({
    fetchImpl: coldServer.fetchImpl,
    opfsDir: sharedOpfsDir,
    slabSize,
    slabMaxBytes: slabSize * 32,
  });
  await coldController.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });

  const coldCoreHandle = await coldController.openHandle('G:\\CORE_BVH4.RPF');
  const coldShadersHandle = await coldController.openHandle('/game/shaders.rpf');
  assert.ok(coldCoreHandle > 0 && coldShadersHandle > 0);

  // Stream all 8 slabs of core_bvh4.rpf and all 4 slabs of shaders.rpf concurrently.
  const coldReads = [];
  for (let s = 0; s < 8; s += 1) {
    coldReads.push(coldController.readHandle(coldCoreHandle, s * slabSize + 64, 512));
  }
  for (let s = 0; s < 4; s += 1) {
    coldReads.push(coldController.readHandle(coldShadersHandle, s * slabSize + 16, 256));
  }
  const coldResults = await Promise.all(coldReads);
  for (let s = 0; s < 8; s += 1) {
    assert.deepEqual(coldResults[s], coreBvh4.subarray(s * slabSize + 64, s * slabSize + 576));
  }
  for (let s = 0; s < 4; s += 1) {
    assert.deepEqual(
      coldResults[8 + s],
      shadersRpf.subarray(s * slabSize + 16, s * slabSize + 272),
    );
  }
  await coldController.flushOpfsL2Writes();

  const coldStats = coldController.getStats();
  assert.equal(coldStats.opfsL2Writes, 12, 'Cold boot must persist all 12 slabs of core_bvh4.rpf + shaders.rpf');
  assert.equal(coldStats.opfsWrites, 12, 'Top-level opfsWrites must report aggregate OPFS writes');
  assert.equal(coldStats.opfsErrors, 0);
  assert.equal(coldStats.opfsL2WriteErrors, 0);
  assert.equal(sharedOpfsDir.stats.keysCalls + sharedOpfsDir.stats.entriesCalls, 1, 'Cold boot must scan OPFS directory at most once');
  assert.equal(sharedOpfsDir.stats.getFileHandleCreateFalse, 0, 'Cold boot misses must be O(1) with zero getFileHandle(create:false) calls');
  assert.equal(coldServer.headRequests.length, 0, 'Cold boot on empty OPFS dir must issue 0 HEAD requests');
  assert.equal(sharedOpfsDir.stats.syncHandleLockCollisions, 0);
  assert.equal(sharedOpfsDir.files.size, 12);

  // --- WARM BOOT (brand-new controller, NO pre-seeded fingerprints, mixed case + 8 concurrent reads) ---
  const warmServer = createMockHttpServer(serverEntries);
  const warmController = new IoWorkerController({
    fetchImpl: warmServer.fetchImpl,
    opfsDir: sharedOpfsDir,
    slabSize,
    slabMaxBytes: slabSize * 32,
  });
  await warmController.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
  const warmGetsAfterInit = warmServer.getRequests.length;

  // Open with different casing & drive path ('G:\\core_bvh4.rpf' vs cold 'G:\\CORE_BVH4.RPF')
  // and read all 8 slabs concurrently (simulating 8 mailbox lanes racing on first warm access).
  const warmCoreHandle = await warmController.openHandle('G:\\core_bvh4.rpf', { probeOnOpen: false });
  const warmShadersHandle = await warmController.openHandle('G:\\SHADERS.RPF');
  assert.ok(warmCoreHandle > 0 && warmShadersHandle > 0);

  const warmReads = [];
  for (let s = 0; s < 8; s += 1) {
    warmReads.push(warmController.readHandle(warmCoreHandle, s * slabSize + 64, 512));
  }
  for (let s = 0; s < 4; s += 1) {
    warmReads.push(warmController.readHandle(warmShadersHandle, s * slabSize + 16, 256));
  }
  const warmResults = await Promise.all(warmReads);
  for (let s = 0; s < 8; s += 1) {
    assert.deepEqual(warmResults[s], coreBvh4.subarray(s * slabSize + 64, s * slabSize + 576));
  }
  for (let s = 0; s < 4; s += 1) {
    assert.deepEqual(
      warmResults[8 + s],
      shadersRpf.subarray(s * slabSize + 16, s * slabSize + 272),
    );
  }
  await warmController.flushOpfsL2Writes();

  const warmStats = warmController.getStats();
  assert.equal(
    warmServer.getRequests.length - warmGetsAfterInit,
    0,
    'Warm boot must serve ALL 12 slabs (including slab 0) from OPFS L2 with 0 HTTP Range GETs',
  );
  assert.equal(
    warmServer.headRequests.length,
    2,
    'Warm boot must coalesce 8 concurrent lanes into exactly 1 lightweight 0-byte HEAD probe per hot RPF',
  );
  assert.equal(warmStats.opfsL2Hits, 12);
  assert.equal(warmStats.opfsHits, 12, 'Top-level opfsHits must reflect OPFS L2 hits');
  assert.equal(warmStats.opfsL2Misses, 0);
  assert.equal(warmStats.opfsL2Writes, 0, 'Warm boot hits must not rewrite existing OPFS slabs');
  assert.equal(warmStats.immutableCache.bytesFetched, 0);

  // --- WARM BOOT with HEAD returning 405 Method Not Allowed degrades gracefully ---
  const noHeadServer = createMockHttpServer(serverEntries, { headStatusOverride: 405 });
  const fallbackController = new IoWorkerController({
    fetchImpl: noHeadServer.fetchImpl,
    opfsDir: sharedOpfsDir,
    slabSize,
    slabMaxBytes: slabSize * 16,
  });
  await fallbackController.init({ dataBase: '/data/', bootBase: '/boot/', gameBase: '/game/' });
  const fbGetsAfterInit = noHeadServer.getRequests.length;

  const fbHandle = await fallbackController.openHandle('G:\\shaders.rpf');
  assert.deepEqual(await fallbackController.readHandle(fbHandle, 0, 256), shadersRpf.subarray(0, 256));
  assert.deepEqual(
    await fallbackController.readHandle(fbHandle, slabSize, 256),
    shadersRpf.subarray(slabSize, slabSize + 256),
  );
  assert.equal(
    noHeadServer.getRequests.length - fbGetsAfterInit,
    1,
    'When HEAD returns 405, only slab 0 falls back to Range GET; slab 1 still hits OPFS L2',
  );
  assert.equal(fallbackController.getStats().opfsL2Hits, 1);
});

test('W369: ETag and size rotation invalidate stale OPFS entries cleanly across ImmutableHttpfsCache and OpfsChunkCache', async () => {
  const slabSize = 2048;
  const v1 = makePatternBytes(slabSize * 2, 0x11);
  const v2 = makePatternBytes(slabSize * 2, 0x22); // same size, rotated ETag
  const v3 = makePatternBytes(slabSize * 2 + 512, 0x44); // changed size + ETag

  const opfsDir = createInstrumentedOpfsDir();
  const entries = new Map([['/game/core_bvh4.rpf', { bytes: v1, etag: '"etag-v1"' }]]);

  // Seed v1 into opfsDir.
  const s1 = createMockHttpServer(entries);
  const c1 = new ImmutableHttpfsCache({ fetchImpl: s1.fetchImpl, opfsDir, slabSize, slabMaxBytes: slabSize * 4 });
  assert.deepEqual(await c1.read('/game/core_bvh4.rpf', 0, v1.byteLength), v1);
  await c1.flushOpfsL2Writes();
  assert.equal(opfsDir.files.size, 2);

  // Rotate server to v2 (same size, new ETag). Fresh warm boot must detect new ETag via HEAD,
  // invalidate stale v1 OPFS files, and fetch fresh v2 bytes.
  entries.set('/game/core_bvh4.rpf', { bytes: v2, etag: '"etag-v2"' });
  const s2 = createMockHttpServer(entries);
  const c2 = new ImmutableHttpfsCache({ fetchImpl: s2.fetchImpl, opfsDir, slabSize, slabMaxBytes: slabSize * 4 });
  const gotV2 = await c2.read('/game/core_bvh4.rpf', 0, v2.byteLength);
  await c2.flushOpfsL2Writes();
  assert.deepEqual(gotV2, v2);
  assert.equal(c2.getStats().opfsL2Hits, 0);
  assert.ok(c2.getStats().opfsL2Invalidations >= 2);
  assert.equal(c2.getStats().opfsL2Writes, 2);
  assert.equal(opfsDir.files.size, 2, 'Stale v1 OPFS files must be removed when replaced by v2');

  // Rotate server to v3 (size + ETag change).
  entries.set('/game/core_bvh4.rpf', { bytes: v3, etag: '"etag-v3"' });
  const s3 = createMockHttpServer(entries);
  const c3 = new ImmutableHttpfsCache({ fetchImpl: s3.fetchImpl, opfsDir, slabSize, slabMaxBytes: slabSize * 8 });
  const gotV3 = await c3.read('/game/core_bvh4.rpf', 0, v3.byteLength);
  await c3.flushOpfsL2Writes();
  assert.deepEqual(gotV3, v3);
  assert.equal(opfsDir.files.size, 3);

  // Verify OpfsChunkCache disk reconciliation preserves `_fp_` fileFingerprint across restarts.
  const fpA = buildFileFingerprint(slabSize * 2, '"chunk-v1"');
  const fpB = buildFileFingerprint(slabSize * 2, '"chunk-v2"');
  const chunkKeyA = makeChunkKey({ path: 'game/shaders.rpf', offset: 0, length: slabSize, fileFingerprint: fpA });
  const chunkDir = createInstrumentedOpfsDir();
  chunkDir.files.set(chunkKeyA, v1.subarray(0, slabSize).slice());

  const storageObj = {
    async getDirectory() {
      return {
        async getDirectoryHandle() {
          return chunkDir;
        },
      };
    },
  };
  const reconciledCache = new OpfsChunkCache({ storageObj });
  await reconciledCache.init();

  // Warm hit with matching fileFingerprint must NOT falsely invalidate disk-reconciled entry.
  const hitEntry = await reconciledCache.getEntry(chunkKeyA, { fileFingerprint: fpA });
  assert.equal(hitEntry.hit, true, 'Disk-reconciled _fp_ file must HIT on matching fileFingerprint');
  assert.deepEqual(hitEntry.data, v1.subarray(0, slabSize));

  // Stale mismatch with fpB must miss and invalidate cleanly.
  const missEntry = await reconciledCache.getEntry(chunkKeyA, { fileFingerprint: fpB });
  assert.equal(missEntry.hit, false);
  assert.equal(missEntry.invalidated, true);
  assert.equal(chunkDir.files.has(chunkKeyA), false);
});

test('W369: ?opfsL2=0 (opfsL2Enabled: false) strictly bypasses OPFS dir opening, reads, and writes', async () => {
  const slabSize = 2048;
  const rpfBytes = makePatternBytes(slabSize * 2, 0x55);
  const server = createMockHttpServer(
    new Map([['/game/core_bvh4.rpf', { bytes: rpfBytes, etag: '"optout-v1"' }]]),
  );

  let openOpfsCalled = 0;
  const fakeRangeCacheMod = {
    RangeCache,
    async openOpfsCacheDir() {
      openOpfsCalled += 1;
      return createInstrumentedOpfsDir();
    },
  };

  const controller = new IoWorkerController({
    fetchImpl: server.fetchImpl,
    rangeCacheModule: fakeRangeCacheMod,
    slabSize,
    slabMaxBytes: slabSize * 4,
  });

  const ready = await controller.init({
    dataBase: '/data/',
    bootBase: '/boot/',
    gameBase: '/game/',
    opfsL2Enabled: false,
  });

  assert.equal(openOpfsCalled, 0, 'opfsL2Enabled: false must not call openOpfsCacheDir');
  assert.equal(ready.opfs, false);
  assert.equal(controller.immutableCache.opfsDir, null);
  assert.equal(controller.rangeCache.opfsDir, null);

  const handle = await controller.openHandle('G:\\core_bvh4.rpf');
  assert.deepEqual(await controller.readHandle(handle, 0, slabSize), rpfBytes.subarray(0, slabSize));
  await controller.flushOpfsL2Writes();
  const stats = controller.getStats();
  assert.equal(stats.opfsHits, 0);
  assert.equal(stats.opfsWrites, 0);
  assert.equal(stats.opfsL2Hits, 0);
  assert.equal(stats.opfsL2Writes, 0);
});

test('W369: concurrent SyncAccessHandle write deduplication prevents lock collision and pending writes serve RAM-evicted reads', async () => {
  const slabSize = 2048;
  const rpfBytes = makePatternBytes(slabSize * 3, 0x66);
  const server = createMockHttpServer(
    new Map([['/game/shaders.rpf', { bytes: rpfBytes, etag: '"dedup-v1"' }]]),
  );

  // Slow down OPFS writes so multiple write-throughs overlap in flight.
  const slowOpfsDir = createInstrumentedOpfsDir({ syncDelayMs: 25 });
  const cache = new ImmutableHttpfsCache({
    fetchImpl: server.fetchImpl,
    opfsDir: slowOpfsDir,
    slabSize,
    slabMaxBytes: slabSize, // tiny 1-slab RAM LRU forces immediate eviction of slab 0 when slab 1 loads!
  });

  // Read slab 0 -> populates RAM + starts slow async OPFS write for slab 0.
  const s0 = await cache.read('/game/shaders.rpf', 0, slabSize);
  assert.deepEqual(s0, rpfBytes.subarray(0, slabSize));

  // Trigger a duplicate write-through for slab 0 while first write is STILL in flight!
  const dupQueued = cache._maybeWriteThroughOpfsL2('/game/shaders.rpf', 0, slabSize, s0);
  assert.equal(dupQueued, false, 'In-flight writeKey must be deduplicated to prevent SyncAccessHandle lock collision');

  // Read slab 1 -> evicts slab 0 from 1-slab RAM LRU while slab 0 write is still in flight!
  const s1 = await cache.read('/game/shaders.rpf', slabSize, slabSize);
  assert.deepEqual(s1, rpfBytes.subarray(slabSize, slabSize * 2));
  const getsBeforeRetry = server.getRequests.length;

  // Re-read slab 0 BEFORE disk write completes -> must hit pending/in-flight write buffer with 0 HTTP GETs!
  const s0Again = await cache.read('/game/shaders.rpf', 0, 512);
  assert.deepEqual(s0Again, rpfBytes.subarray(0, 512));
  assert.equal(server.getRequests.length, getsBeforeRetry, 'Re-reading RAM-evicted slab with in-flight OPFS write must not re-fetch HTTP');
  assert.equal(cache.getStats().opfsL2Hits, 1);

  await cache.flushOpfsL2Writes();
  assert.equal(slowOpfsDir.stats.syncHandleLockCollisions, 0, 'Zero SyncAccessHandle lock collisions allowed');
  assert.equal(cache.getStats().opfsL2WriteErrors, 0);
  assert.equal(slowOpfsDir.files.size, 2);
});

test('W369: handleMessage serializes attachMailbox behind async init() so opfsDir and handles never race', async () => {
  const slabSize = 2048;
  const rpfBytes = makePatternBytes(slabSize * 2, 0x88);
  const server = createMockHttpServer(
    new Map([['/game/core_bvh4.rpf', { bytes: rpfBytes, etag: '"race-v1"' }]]),
  );
  const opfsDir = createInstrumentedOpfsDir();

  // Delay openOpfsCacheDir to reproduce Chrome worker startup timing where loader.js posts
  // { init: true } followed immediately by { attachMailbox: true }.
  const delayedRangeCacheMod = {
    RangeCache,
    async openOpfsCacheDir() {
      await new Promise((r) => setTimeout(r, 30));
      return opfsDir;
    },
  };

  const posted = [];
  const controller = new IoWorkerController({
    fetchImpl: server.fetchImpl,
    rangeCacheModule: delayedRangeCacheMod,
    postMessage: (m) => posted.push(m),
    slabSize,
    slabMaxBytes: slabSize * 8,
  });

  const sab = createHttpfsMailboxBuffer();
  // Fire both messages back-to-back without awaiting init first (exact loader.js pattern).
  const pInit = controller.handleMessage({
    init: true,
    dataBase: '/data/',
    bootBase: '/boot/',
    gameBase: '/game/',
  });
  const pAttach = controller.handleMessage({
    attachMailbox: true,
    mailboxSab: sab,
    id: 42,
  });
  await Promise.all([pInit, pAttach]);

  assert.equal(controller.immutableCache.opfsDir, opfsDir, 'OPFS dir must be bound before mailbox starts serving');
  assert.equal(controller.mailboxLanes, 1);
  const readyIdx = posted.findIndex((m) => m && m.ready === true);
  const attachIdx = posted.findIndex((m) => m && m.mailboxAttached === true);
  assert.ok(readyIdx >= 0 && attachIdx > readyIdx, 'attachMailbox must complete AFTER init finishes');

  // Verify controller handles work cleanly after serialized init + attachMailbox.
  const h = await controller.openHandle('G:\\core_bvh4.rpf');
  assert.ok(h > 0);
  assert.deepEqual(await controller.readHandle(h, 0, slabSize), rpfBytes.subarray(0, slabSize));
  await controller.flushOpfsL2Writes();
  assert.equal(controller.getStats().opfsL2Writes, 1);
  await controller.detachMailbox();
});

test('W369: RangeCache uses O(1) OPFS directory index on cold misses and SyncAccessHandle on writes', async () => {
  const chunkSize = 1024;
  const totalBytes = chunkSize * 12;
  const payload = makePatternBytes(totalBytes, 0x99);
  const server = createMockHttpServer(
    new Map([['/data/stream.bin', { bytes: payload, etag: '"rc-v1"' }]]),
  );
  const opfsDir = createInstrumentedOpfsDir();

  const coldRc = new RangeCache({
    dataBase: '/data/',
    chunkSize,
    maxBytes: chunkSize * 4,
    opfsDir,
    fetchImpl: server.fetchImpl,
  });

  // Read all 12 chunks cold.
  const gotCold = await coldRc.read('stream.bin', 0, totalBytes);
  assert.deepEqual(gotCold, payload);
  await coldRc.flushOpfsWrites();

  assert.equal(opfsDir.stats.keysCalls + opfsDir.stats.entriesCalls, 1, 'RangeCache cold misses must scan OPFS dir at most once');
  assert.equal(opfsDir.stats.getFileHandleCreateFalse, 0, 'RangeCache cold misses must not call getFileHandle(create:false)');
  assert.ok(opfsDir.stats.syncHandleOpens >= 12, 'RangeCache OPFS writes must prefer createSyncAccessHandle');
  assert.equal(opfsDir.files.size, 12);

  // Warm RangeCache over same opfsDir hits OPFS with 0 HTTP fetches.
  const warmServer = createMockHttpServer(
    new Map([['/data/stream.bin', { bytes: payload, etag: '"rc-v1"' }]]),
  );
  const warmRc = new RangeCache({
    dataBase: '/data/',
    chunkSize,
    maxBytes: chunkSize * 16,
    opfsDir,
    fetchImpl: warmServer.fetchImpl,
  });
  const gotWarm = await warmRc.read('stream.bin', 0, totalBytes);
  assert.deepEqual(gotWarm, payload);
  assert.equal(warmServer.requests.length, 0, 'Warm RangeCache must serve all 12 chunks from OPFS with 0 HTTP requests');
  assert.equal(warmRc.getStats().opfsHits, 12);
});

test('W369: PgimPreloader auto-enables native OPFS, survives caller ArrayBuffer transfer/detachment, hits OPFS on warm boot, and honors ?opfsL2=0', async () => {
  const pgimBytes = buildMinimalPgimBytes(0xab);
  const pgimUrl = '/boot/game_recomp.pgim';
  const serverEntries = new Map([[pgimUrl, { bytes: pgimBytes, etag: '"pgim-v1"' }]]);
  const mockStorage = createMockOpfsStorage();

  // Simulate app_bootstrap.js constructing PgimPreloader WITHOUT passing `opfsCache` (undefined).
  const coldServer = createMockHttpServer(serverEntries);
  const coldPreloader = new PgimPreloader({
    fetchImpl: coldServer.fetchImpl,
    opfsStorageObj: mockStorage,
    parallelChunks: false,
  });
  assert.ok(coldPreloader._opfsStore !== null, 'PgimPreloader must auto-enable native OPFS when getDirectory() is available');

  // Slow OPFS write slightly so the caller's zero-copy buffer transfer detaches `res.buffer` BEFORE OPFS write runs!
  const pgimDir = await (await mockStorage.getDirectory()).getDirectoryHandle('playgta6-pgim-cache', { create: true });
  pgimDir.setWriteDelay(15);

  const coldRes = await coldPreloader.preloadPgim(pgimUrl);
  assert.equal(coldRes.opfsCacheHit, false);

  // Immediately transfer/detach coldRes.buffer just like web/loader.js W351 postMessage(ioInit, [pgimBuf])!
  const transferred = structuredClone(coldRes.buffer, { transfer: [coldRes.buffer] });
  assert.equal(coldRes.buffer.detached, true, 'Caller buffer must be detached immediately after preload resolves');
  assert.deepEqual(new Uint8Array(transferred), pgimBytes);

  await coldPreloader.flushOpfsWrites();
  const coldStats = coldPreloader.getStats();
  assert.equal(coldStats.opfsWrites, 1, 'Detaching caller ArrayBuffer must NOT abort async OPFS write');
  assert.equal(coldStats.opfsInvalidations, 0);
  assert.equal(coldStats.opfsHits, 0);
  assert.equal(coldStats.bytesFromNetwork, pgimBytes.byteLength);

  // --- WARM BOOT: fresh PgimPreloader hits OPFS with 0 HTTP GETs ---
  pgimDir.setWriteDelay(0);
  const warmServer = createMockHttpServer(serverEntries);
  const warmPreloader = new PgimPreloader({
    fetchImpl: warmServer.fetchImpl,
    opfsStorageObj: mockStorage,
    parallelChunks: false,
  });
  const warmRes = await warmPreloader.preloadPgim(pgimUrl);
  assert.equal(warmRes.opfsCacheHit, true);
  assert.deepEqual(new Uint8Array(warmRes.buffer), pgimBytes);
  assert.equal(warmServer.getRequests.length, 0, 'Warm PGIM boot must skip HTTP GET completely');
  assert.equal(warmServer.headRequests.length, 1, 'Warm PGIM boot must validate ETag via 1 HEAD request');
  const warmStats = warmPreloader.getStats();
  assert.equal(warmStats.opfsHits, 1);
  assert.equal(warmStats.opfsMisses, 0);
  assert.equal(warmStats.opfsWrites, 0);

  // --- ETag rotation invalidates stale PGIM and re-caches fresh bytes ---
  const pgimV2 = buildMinimalPgimBytes(0xcd);
  serverEntries.set(pgimUrl, { bytes: pgimV2, etag: '"pgim-v2"' });
  const rotServer = createMockHttpServer(serverEntries);
  const rotPreloader = new PgimPreloader({
    fetchImpl: rotServer.fetchImpl,
    opfsStorageObj: mockStorage,
    parallelChunks: false,
  });
  const rotRes = await rotPreloader.preloadPgim(pgimUrl);
  await rotPreloader.flushOpfsWrites();
  assert.equal(rotRes.opfsCacheHit, false);
  assert.deepEqual(new Uint8Array(rotRes.buffer), pgimV2);
  assert.equal(rotPreloader.getStats().opfsInvalidations, 1);
  assert.equal(rotPreloader.getStats().opfsWrites, 1);

  // --- Explicit opfsCache: null disables OPFS even when storage is available ---
  const disabledPreloader = new PgimPreloader({
    fetchImpl: rotServer.fetchImpl,
    opfsStorageObj: mockStorage,
    opfsCache: null,
  });
  assert.equal(disabledPreloader._opfsStore, null);

  // --- ?opfsL2=0 query param disables auto-enabled native OPFS ---
  const prevLoc = globalThis.location;
  try {
    Object.defineProperty(globalThis, 'location', {
      value: { search: '?opfsL2=0' },
      configurable: true,
      writable: true,
    });
    const optOutPreloader = new PgimPreloader({
      fetchImpl: rotServer.fetchImpl,
      opfsStorageObj: mockStorage,
      parallelChunks: false,
    });
    assert.equal(optOutPreloader._opfsStore, null, '?opfsL2=0 must disable auto-enabled native OPFS in PgimPreloader');

    const optOutRes = await optOutPreloader.preloadPgim(pgimUrl);
    await optOutPreloader.flushOpfsWrites();
    assert.equal(optOutRes.opfsCacheHit, false);
    assert.equal(optOutPreloader.getStats().opfsHits, 0);
    assert.equal(optOutPreloader.getStats().opfsWrites, 0);
  } finally {
    if (prevLoc === undefined) {
      delete globalThis.location;
    } else {
      Object.defineProperty(globalThis, 'location', {
        value: prevLoc,
        configurable: true,
        writable: true,
      });
    }
  }
});
