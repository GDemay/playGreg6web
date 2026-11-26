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

const userdataMod = require('../web/workers/userdata.js');
const schedulerMod = require('../web/workers/io_scheduler.js');
const rpfMod = require('../web/workers/rpf9_vfs.js');

const {
  UserdataStore,
  OpfsUserdataBackend,
  InMemoryUserdataBackend,
  createOpfsUserdataBackend,
  createMemoryUserdataBackend,
  probeOpfsSupport,
  DEFAULT_OPFS_DIR_NAME,
  OPFS_RECORD_MAGIC,
  OPFS_RECORD_VERSION,
  OPFS_RECORD_HEADER_SIZE,
  STORAGE_MODES,
  BACKEND_KINDS,
  QUARANTINE_PREFIX,
  TEMP_SUFFIX,
  encodeOpfsKeyFilename,
  encodeOpfsRecord,
  decodeOpfsRecord,
  computeChecksum,
} = userdataMod;

const {
  IO_PRIORITY,
  IoPriorityScheduler,
  classifyIoPriority,
  isWorldStructureAssetPath,
  isWorldStreamingAssetPath,
} = schedulerMod;

const {
  RPF9_MAGIC,
  RPF9_HEADER_SIZE,
  RPF9_ENTRY_STRIDE,
  RPF_SECTOR_SIZE,
  RPF_ENCRYPTION_NONE,
  RPF_ENTRY_FLAG_DIRECTORY,
  RPF_ENTRY_FLAG_COMPRESSED,
  RPF_ENTRY_FLAG_OFFSET_SHIFT9,
  RpfFormatError,
  parseTocEntry,
  normalizeVirtualPath,
  normalizePathKeyFast,
  joaatHash,
  hashVirtualPath,
  RpfArchiveVfs,
  RpfVfsMountTable,
  buildSyntheticRpf9Buffer,
} = rpfMod;

const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8');

function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function makeBytes(seed, length) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (seed + i * 31) & 0xff;
  }
  return out;
}

function createMockOpfsRoot(options = {}) {
  const mode = options.mode || 'sync'; // 'sync' | 'writable' | 'sync-fails-writable-ok' | 'both'
  const dirs = new Map();

  function getOrCreateDirMap(name) {
    let files = dirs.get(name);
    if (!files) {
      files = new Map();
      dirs.set(name, files);
    }
    return files;
  }

  function makeNotFound(name) {
    const err = new Error(`NotFoundError: ${name}`);
    err.name = 'NotFoundError';
    return err;
  }

  function makeFileHandle(filesMap, fileName) {
    const handle = {
      kind: 'file',
      name: fileName,
    };

    if (mode === 'sync' || mode === 'both') {
      handle.createSyncAccessHandle = async () => {
        let closed = false;
        let data = (filesMap.get(fileName) || new Uint8Array(0)).slice();
        const ensureOpen = () => {
          if (closed) throw new Error('SyncAccessHandle already closed');
        };
        return {
          getSize() {
            ensureOpen();
            return data.byteLength;
          },
          truncate(newSize) {
            ensureOpen();
            if (newSize === 0) {
              data = new Uint8Array(0);
            } else if (newSize < data.byteLength) {
              data = data.slice(0, newSize);
            } else if (newSize > data.byteLength) {
              const next = new Uint8Array(newSize);
              next.set(data, 0);
              data = next;
            }
          },
          write(buf, writeOpts = {}) {
            ensureOpen();
            const src = buf instanceof Uint8Array ? buf : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
            const at = Number.isInteger(writeOpts.at) ? writeOpts.at : 0;
            const end = at + src.byteLength;
            if (end > data.byteLength) {
              const next = new Uint8Array(end);
              next.set(data, 0);
              data = next;
            }
            data.set(src, at);
            filesMap.set(fileName, data.slice());
            return src.byteLength;
          },
          read(buf, readOpts = {}) {
            ensureOpen();
            const dst = buf instanceof Uint8Array ? buf : new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
            const at = Number.isInteger(readOpts.at) ? readOpts.at : 0;
            if (at >= data.byteLength) return 0;
            const n = Math.min(dst.byteLength, data.byteLength - at);
            dst.set(data.subarray(at, at + n), 0);
            return n;
          },
          flush() {
            ensureOpen();
            filesMap.set(fileName, data.slice());
          },
          close() {
            closed = true;
          },
        };
      };
    } else if (mode === 'sync-fails-writable-ok') {
      handle.createSyncAccessHandle = async () => {
        const err = new Error('createSyncAccessHandle is only available in dedicated workers');
        err.name = 'NotAllowedError';
        throw err;
      };
    }

    if (mode === 'writable' || mode === 'sync-fails-writable-ok' || mode === 'both') {
      handle.createWritable = async () => {
        let chunks = [];
        let aborted = false;
        return {
          async write(chunk) {
            const u8 =
              chunk instanceof Uint8Array
                ? chunk.slice()
                : chunk instanceof ArrayBuffer
                  ? new Uint8Array(chunk.slice(0))
                  : new Uint8Array(chunk.buffer, chunk.byteOffset, chunk.byteLength).slice();
            chunks.push(u8);
          },
          async close() {
            if (aborted) return;
            const total = chunks.reduce((sum, c) => sum + c.byteLength, 0);
            const merged = new Uint8Array(total);
            let off = 0;
            for (const c of chunks) {
              merged.set(c, off);
              off += c.byteLength;
            }
            filesMap.set(fileName, merged);
          },
          async abort() {
            aborted = true;
            chunks = [];
          },
        };
      };
      handle.getFile = async () => {
        const bytes = filesMap.get(fileName);
        if (!bytes) throw makeNotFound(fileName);
        const copy = bytes.slice();
        return {
          size: copy.byteLength,
          async arrayBuffer() {
            return copy.buffer;
          },
        };
      };
    }

    return handle;
  }

  function makeDirHandle(dirName) {
    const filesMap = getOrCreateDirMap(dirName);
    return {
      kind: 'directory',
      name: dirName,
      async getFileHandle(name, opts = {}) {
        if (!filesMap.has(name)) {
          if (opts.create) {
            filesMap.set(name, new Uint8Array(0));
          } else {
            throw makeNotFound(name);
          }
        }
        return makeFileHandle(filesMap, name);
      },
      async removeEntry(name) {
        if (!filesMap.has(name)) {
          throw makeNotFound(name);
        }
        filesMap.delete(name);
      },
      async *entries() {
        for (const name of filesMap.keys()) {
          yield [name, makeFileHandle(filesMap, name)];
        }
      },
    };
  }

  return {
    dirs,
    getDirFiles(dirName = DEFAULT_OPFS_DIR_NAME) {
      return getOrCreateDirMap(dirName);
    },
    async getDirectoryHandle(name, opts = {}) {
      if (!dirs.has(name)) {
        if (opts.create) {
          dirs.set(name, new Map());
        } else {
          throw makeNotFound(name);
        }
      }
      return makeDirHandle(name);
    },
  };
}

function createMockFsAdapter() {
  const dirs = new Set();
  const files = new Map();
  const mtimes = new Map();
  return {
    dirs,
    files,
    mtimes,
    mkdirTree(dir) {
      const parts = dir.split('/').filter(Boolean);
      let cur = '';
      for (const p of parts) {
        cur += `/${p}`;
        dirs.add(cur);
      }
    },
    writeFile(filePath, data) {
      files.set(filePath, new Uint8Array(data));
    },
    utime(filePath, _a, m) {
      mtimes.set(filePath, m);
    },
  };
}

function createSimpleMockIndexedDB() {
  const stores = new Map();
  return {
    open(dbName) {
      const req = { result: null, error: null, onupgradeneeded: null, onsuccess: null, onerror: null };
      queueMicrotask(() => {
        if (!stores.has(dbName)) {
          stores.set(dbName, new Map());
        }
        const map = stores.get(dbName);
        const db = {
          objectStoreNames: { contains: () => true },
          createObjectStore() {},
          transaction() {
            const tx = {
              oncomplete: null,
              onerror: null,
              onabort: null,
              objectStore() {
                const mk = (fn) => (...args) => {
                  const r = { result: undefined, error: null, onsuccess: null, onerror: null };
                  queueMicrotask(() => {
                    r.result = fn(...args);
                    if (r.onsuccess) r.onsuccess({ target: r });
                    queueMicrotask(() => {
                      if (tx.oncomplete) tx.oncomplete({ target: tx });
                    });
                  });
                  return r;
                };
                return {
                  put: mk((rec) => {
                    map.set(rec.path, { ...rec, data: rec.data instanceof Uint8Array ? rec.data.slice() : rec.data });
                    return rec.path;
                  }),
                  get: mk((k) => {
                    const rec = map.get(k);
                    return rec ? { ...rec, data: rec.data instanceof Uint8Array ? rec.data.slice() : rec.data } : undefined;
                  }),
                  delete: mk((k) => {
                    map.delete(k);
                  }),
                  getAll: mk(() =>
                    Array.from(map.values()).map((r) => ({
                      ...r,
                      data: r.data instanceof Uint8Array ? r.data.slice() : r.data,
                    })),
                  ),
                };
              },
            };
            return tx;
          },
          close() {},
        };
        req.result = db;
        if (req.onsuccess) req.onsuccess({ target: req });
      });
      return req;
    },
  };
}

// ---------------------------------------------------------------------------
// Part A: Persistent OPFS Savegame Storage (web/workers/userdata.js)
// ---------------------------------------------------------------------------

test('W381 userdata: exports OPFS/memory backends, codec, and UMD Worker global bindings', () => {
  assert.equal(DEFAULT_OPFS_DIR_NAME, 'playgta6-userdata');
  assert.equal(OPFS_RECORD_MAGIC, 'PGUD');
  assert.equal(OPFS_RECORD_VERSION, 1);
  assert.equal(OPFS_RECORD_HEADER_SIZE, 24);
  assert.equal(STORAGE_MODES.OPFS, 'opfs');
  assert.equal(BACKEND_KINDS.OPFS, 'opfs');
  assert.equal(typeof OpfsUserdataBackend, 'function');
  assert.equal(typeof InMemoryUserdataBackend, 'function');
  assert.equal(typeof createOpfsUserdataBackend, 'function');
  assert.equal(typeof createMemoryUserdataBackend, 'function');
  assert.equal(typeof probeOpfsSupport, 'function');
  assert.equal(typeof encodeOpfsKeyFilename, 'function');
  assert.equal(typeof encodeOpfsRecord, 'function');
  assert.equal(typeof decodeOpfsRecord, 'function');

  // Verify classic Web Worker UMD sandbox execution.
  const src = fs.readFileSync(path.join(repoRoot, 'web/workers/userdata.js'), 'utf8');
  const sandbox = {
    TextEncoder,
    TextDecoder,
    Uint8Array,
    Int16Array,
    ArrayBuffer,
    Map,
    Set,
    Promise,
    Date,
    Error,
    TypeError,
    RangeError,
    Number,
    Array,
    Math,
    JSON,
    setTimeout,
    clearTimeout,
    performance,
  };
  sandbox.globalThis = sandbox;
  sandbox.self = sandbox;
  vm.runInNewContext(src, sandbox, { filename: 'web/workers/userdata.js' });
  assert.equal(typeof sandbox.PlayGta6Userdata.OpfsUserdataBackend, 'function');
  assert.equal(typeof sandbox.PlayGta6Userdata.InMemoryUserdataBackend, 'function');
  assert.equal(typeof sandbox.PlayGta6Userdata.probeOpfsSupport, 'function');
});

test('W381 userdata: OPFS record codec and filename mapping enforce /userdata/ confinement and detect corruption', () => {
  // Valid live, temp, and quarantine keys map to flat, traversal-free OPFS filenames.
  const liveFile = encodeOpfsKeyFilename('/userdata/xgs/story/slot0.sav');
  const tmpFile = encodeOpfsKeyFilename('/userdata/xgs/story/slot0.sav.tmp');
  const quarFile = encodeOpfsKeyFilename(`${QUARANTINE_PREFIX}1000-1-corrupt/xgs/story/slot0.sav`);
  assert.match(liveFile, /^u_[0-9a-f]{32}\.pgud$/);
  assert.match(tmpFile, /^t_[0-9a-f]{32}\.pgud\.tmp$/);
  assert.match(quarFile, /^q_[0-9a-f]{32}\.pgud$/);
  assert.equal(liveFile.includes('/'), false);
  assert.equal(liveFile.includes('..'), false);

  // Traversal and outside-/userdata/ keys must be rejected before touching OPFS.
  for (const bad of [
    '/etc/passwd',
    '/userdata/../escape.sav',
    '/userdata/saves/../../escape.sav',
    '/userdata/saves\\slot0.sav',
    '/userdata/slot0\0.sav',
    '/userdata/',
    `${QUARANTINE_PREFIX}../escape.sav`,
  ]) {
    assert.throws(() => encodeOpfsKeyFilename(bad), `Expected rejection for ${bad}`);
  }

  // Round-trip binary record envelope.
  const payload = makeBytes(77, 128);
  const encoded = encodeOpfsRecord({
    path: '/userdata/xgs/story/slot0.sav',
    data: payload,
    mtime: 1700001234567,
    size: payload.byteLength,
    checksum: computeChecksum(payload),
  });
  const decoded = decodeOpfsRecord(encoded);
  assert.equal(decoded.ok, true);
  assert.equal(decoded.record.path, '/userdata/xgs/story/slot0.sav');
  assert.equal(decoded.record.mtime, 1700001234567);
  assert.equal(decoded.record.size, 128);
  assert.deepEqual(decoded.record.data, payload);

  // Detect header truncation, bad magic, bad version, header checksum tamper, payload truncation, trailing bytes, payload bitflip.
  assert.equal(decodeOpfsRecord(encoded.slice(0, 10)).reason, 'truncated_header');

  const badMagic = encoded.slice();
  badMagic[0] = 0x00;
  assert.equal(decodeOpfsRecord(badMagic).reason, 'invalid_magic');

  const badVersion = encoded.slice();
  badVersion[4] = 99;
  assert.equal(decodeOpfsRecord(badVersion).reason, 'unsupported_version');

  const badHeaderMeta = encoded.slice();
  badHeaderMeta[OPFS_RECORD_HEADER_SIZE + 2] ^= 0xff;
  assert.equal(decodeOpfsRecord(badHeaderMeta).reason, 'header_checksum_mismatch');

  const truncatedPayload = encoded.slice(0, encoded.byteLength - 10);
  const truncDec = decodeOpfsRecord(truncatedPayload);
  assert.equal(truncDec.ok, false);
  assert.equal(truncDec.reason, 'truncated_payload');
  assert.equal(truncDec.partialPath, '/userdata/xgs/story/slot0.sav');

  const trailing = new Uint8Array(encoded.byteLength + 4);
  trailing.set(encoded, 0);
  assert.equal(decodeOpfsRecord(trailing).reason, 'trailing_bytes');

  const flippedPayload = encoded.slice();
  flippedPayload[flippedPayload.byteLength - 1] ^= 0x55;
  const flipDec = decodeOpfsRecord(flippedPayload);
  assert.equal(flipDec.ok, false);
  assert.equal(flipDec.reason, 'checksum_mismatch');
  assert.equal(flipDec.partialPath, '/userdata/xgs/story/slot0.sav');
});

test('W381 userdata: OPFS FileSystemSyncAccessHandle fast path persists Story Mode saves across reloads', async () => {
  const opfsRoot = createMockOpfsRoot({ mode: 'sync' });
  const store1 = new UserdataStore({
    storageMode: 'opfs',
    opfsRoot,
    opfsDirName: 'gta6-story-saves',
  });

  assert.equal(store1.getStats().backendKind, 'opfs');
  assert.equal(store1.getBackendInfo().opfsDirName, 'gta6-story-saves');

  const slot0 = makeBytes(11, 2048);
  const settings = encoder.encode(JSON.stringify({ subtitles: true, aimAssist: 'partial' }));
  await store1.saveFile('/userdata/xgs/story/slot0.sav', slot0, 1700500001000);
  await store1.saveFile('/userdata/profiles/p0/settings.json', settings, 1700500002000);

  const info1 = store1.getBackendInfo();
  assert.equal(info1.backendKind, 'opfs');
  assert.equal(info1.opfsInitialized, true);
  assert.equal(info1.ioMode, 'sync-access-handle');
  assert.ok(store1.getStats().opfsSyncWrites >= 4); // 2 atomic saves x (tmp + live)
  assert.equal(store1.getStats().opfsWritableWrites, 0);

  // Confirm no leftover .tmp files exist in OPFS directory.
  const rawFiles = opfsRoot.getDirFiles('gta6-story-saves');
  assert.equal(rawFiles.size, 2);
  for (const name of rawFiles.keys()) {
    assert.equal(name.endsWith(TEMP_SUFFIX), false);
  }
  store1.close();

  // Simulate full browser reload with a brand-new UserdataStore pointing at the same OPFS root.
  const store2 = new UserdataStore({
    storageMode: 'auto',
    opfsRoot,
    opfsDirName: 'gta6-story-saves',
  });
  const fsAdapter = createMockFsAdapter();
  const restored = await store2.restoreAll(fsAdapter);
  assert.deepEqual(restored, { restored: 2, skipped: 0, timedOut: false, error: null });
  assert.deepEqual(fsAdapter.files.get('/userdata/xgs/story/slot0.sav'), slot0);
  assert.deepEqual(fsAdapter.files.get('/userdata/profiles/p0/settings.json'), settings);
  assert.equal(fsAdapter.mtimes.get('/userdata/xgs/story/slot0.sav'), 1700500001000);

  const fetched = await store2.getFile('/userdata/xgs/story/slot0.sav');
  assert.deepEqual(fetched.data, slot0);
  assert.deepEqual(await store2.listByPrefix('/userdata/xgs/story'), ['/userdata/xgs/story/slot0.sav']);

  await store2.deleteFile('/userdata/xgs/story/slot0.sav');
  assert.equal(await store2.getFile('/userdata/xgs/story/slot0.sav'), null);
  assert.equal(rawFiles.size, 1);
  store2.close();
});

test('W381 userdata: OPFS async createWritable() path and sync-to-writable fallback persist cleanly across reloads', async () => {
  for (const mode of ['writable', 'sync-fails-writable-ok']) {
    const opfsRoot = createMockOpfsRoot({ mode });
    const store1 = new UserdataStore({ storageMode: 'opfs', opfsRoot });
    const blob = makeBytes(44, 512);
    await store1.saveFile('/userdata/saves/autosave.sav', blob, 1700600000000);
    assert.equal(store1.getBackendInfo().ioMode, 'writable');
    assert.ok(store1.getStats().opfsWritableWrites >= 2);
    store1.close();

    const store2 = new UserdataStore({ storageMode: 'opfs', opfsRoot });
    const loaded = await store2.getFile('/userdata/saves/autosave.sav');
    assert.deepEqual(loaded.data, blob);
    assert.equal(loaded.mtime, 1700600000000);
    store2.close();
  }
});

test('W381 userdata: OPFS recovers interrupted .tmp files and quarantines corrupt records without crashing boot', async () => {
  const opfsRoot = createMockOpfsRoot({ mode: 'sync' });
  const store1 = new UserdataStore({ storageMode: 'opfs', opfsRoot });

  const healthy = encoder.encode('HEALTHY_STORY_SAVE_V1');
  const victimFlip = makeBytes(9, 64);
  const victimTrunc = makeBytes(17, 64);

  await store1.saveFile('/userdata/saves/healthy.sav', healthy, 1000);
  await store1.saveFile('/userdata/saves/flip.sav', victimFlip, 1001);
  await store1.saveFile('/userdata/saves/trunc.sav', victimTrunc, 1002);
  store1.close();

  const dirFiles = opfsRoot.getDirFiles(DEFAULT_OPFS_DIR_NAME);

  // 1. Corrupt flip.sav payload byte in OPFS storage.
  const flipKey = encodeOpfsKeyFilename('/userdata/saves/flip.sav');
  const flipRaw = dirFiles.get(flipKey).slice();
  flipRaw[flipRaw.byteLength - 1] ^= 0xff;
  dirFiles.set(flipKey, flipRaw);

  // 2. Truncate trunc.sav in OPFS storage.
  const truncKey = encodeOpfsKeyFilename('/userdata/saves/trunc.sav');
  const truncRaw = dirFiles.get(truncKey);
  dirFiles.set(truncKey, truncRaw.slice(0, truncRaw.byteLength - 12));

  // 3. Inject totally garbled live .pgud file AND an interrupted/truncated .pgud.tmp file.
  dirFiles.set('u_deadbeef000000000000000000000001.pgud', new Uint8Array([0xde, 0xad, 0xbe, 0xef, 0x01]));
  dirFiles.set('t_deadbeef000000000000000000000002.pgud.tmp', new Uint8Array([0x50, 0x47, 0x55]));

  // Reopen and restoreAll(): must NOT crash, must restore healthy.sav, clean .tmp, and quarantine 3 corrupt files.
  const store2 = new UserdataStore({ storageMode: 'opfs', opfsRoot });
  const fsAdapter = createMockFsAdapter();
  const res = await store2.restoreAll(fsAdapter);

  assert.equal(res.error, null);
  assert.equal(res.timedOut, false);
  assert.equal(res.restored, 1);
  assert.equal(res.skipped, 3);
  assert.deepEqual(fsAdapter.files.get('/userdata/saves/healthy.sav'), healthy);
  assert.equal(fsAdapter.files.has('/userdata/saves/flip.sav'), false);
  assert.equal(fsAdapter.files.has('/userdata/saves/trunc.sav'), false);

  // Orphaned .tmp and corrupt live files must be deleted from OPFS.
  assert.equal(dirFiles.has('t_deadbeef000000000000000000000002.pgud.tmp'), false);
  assert.equal(dirFiles.has('u_deadbeef000000000000000000000001.pgud'), false);
  assert.equal(dirFiles.has(flipKey), false);
  assert.equal(dirFiles.has(truncKey), false);

  // Corrupt entries are quarantined and healthy save remains listed cleanly.
  const quarantined = await store2.listQuarantined();
  assert.equal(quarantined.length, 3);
  assert.deepEqual(await store2.listByPrefix('/userdata/'), ['/userdata/saves/healthy.sav']);
  assert.ok(store2.getStats().corruptRecoveries >= 4);
  store2.close();
});

test('W381 userdata: probeOpfsSupport and graceful OPFS -> IndexedDB -> InMemoryUserdataBackend fallback', async () => {
  // 1. probeOpfsSupport without OPFS in Node.js never throws.
  const noOpfs = await probeOpfsSupport();
  assert.equal(noOpfs.supported, false);
  assert.equal(noOpfs.reason, 'opfs_unavailable');

  // 2. probeOpfsSupport with working mock root.
  const okProbe = await probeOpfsSupport({ opfsRoot: createMockOpfsRoot({ mode: 'both' }), probeCapabilities: true });
  assert.equal(okProbe.supported, true);
  assert.equal(okProbe.syncAccessHandle, true);
  assert.equal(okProbe.writable, true);
  assert.equal(okProbe.reason, null);

  // 3. probeOpfsSupport when getOpfsRoot throws SecurityError.
  const secProbe = await probeOpfsSupport({
    getOpfsRoot: async () => {
      const e = new Error('Storage access blocked in sandboxed iframe');
      e.name = 'SecurityError';
      throw e;
    },
  });
  assert.equal(secProbe.supported, false);
  assert.equal(secProbe.reason, 'SecurityError');

  // 4. OPFS init SecurityError falls back cleanly to IndexedDB when available.
  const mockIdb = createSimpleMockIndexedDB();
  const idbFallbackStore = new UserdataStore({
    storageMode: 'opfs',
    indexedDB: mockIdb,
    getOpfsRoot: async () => {
      const e = new Error('OPFS denied');
      e.name = 'SecurityError';
      throw e;
    },
  });
  assert.equal(idbFallbackStore.getBackendInfo().backendKind, 'opfs');
  await idbFallbackStore.saveFile('/userdata/saves/slot0.sav', encoder.encode('IDB_FALLBACK_OK'), 100);
  assert.equal(idbFallbackStore.getBackendInfo().backendKind, 'indexeddb');
  assert.equal(idbFallbackStore.getBackendInfo().fallbackReason, 'SecurityError');
  assert.equal(idbFallbackStore.getStats().backendKind, 'indexeddb');
  assert.equal(idbFallbackStore.getStats().backendFallbacks, 1);
  assert.equal(decoder.decode((await idbFallbackStore.getFile('/userdata/saves/slot0.sav')).data), 'IDB_FALLBACK_OK');
  idbFallbackStore.close();

  // 5. OPFS QuotaExceededError during init with no IndexedDB falls back cleanly to InMemoryUserdataBackend.
  const memFallbackStore = new UserdataStore({
    storageMode: 'opfs',
    indexedDB: null,
    getOpfsRoot: async () => {
      const e = new Error('OPFS quota unavailable');
      e.name = 'QuotaExceededError';
      throw e;
    },
  });
  const restoreRes = await memFallbackStore.restoreAll(createMockFsAdapter());
  assert.deepEqual(restoreRes, { restored: 0, skipped: 0, timedOut: false, error: null });
  assert.equal(memFallbackStore.getBackendInfo().backendKind, 'memory');
  assert.equal(memFallbackStore.getBackendInfo().fallbackReason, 'QuotaExceededError');
  await memFallbackStore.saveFile('/userdata/saves/mem.sav', encoder.encode('MEM_OK'), 200);
  assert.equal(decoder.decode((await memFallbackStore.getFile('/userdata/saves/mem.sav')).data), 'MEM_OK');

  // 6. Explicit memory mode & bare Node.js auto mode both work without throwing.
  const explicitMem = new UserdataStore({ storageMode: 'memory' });
  assert.equal(explicitMem.getBackendInfo().backendKind, 'memory');
  assert.equal(explicitMem.getBackendInfo().fallbackReason, null);
  await explicitMem.saveFile('/userdata/cfg.json', '{"ok":true}', 300);
  assert.equal(decoder.decode((await explicitMem.getFile('/userdata/cfg.json')).data), '{"ok":true}');

  const bareAuto = new UserdataStore();
  assert.equal(bareAuto.getBackendInfo().backendKind, 'memory');
  assert.ok(bareAuto.getBackendInfo().fallbackReason);

  // 7. Invalid storageMode throws RangeError.
  assert.throws(() => new UserdataStore({ storageMode: 'invalid-backend' }), RangeError);

  // 8. Standalone factory helpers work directly.
  const memBackend = createMemoryUserdataBackend([
    { path: '/userdata/seed.bin', data: new Uint8Array([1, 2, 3]), mtime: 50 },
  ]);
  assert.equal(memBackend instanceof InMemoryUserdataBackend, true);
  assert.equal(memBackend.size, 1);

  const opfsBackend = createOpfsUserdataBackend({ opfsRoot: createMockOpfsRoot({ mode: 'sync' }) });
  assert.equal(opfsBackend instanceof OpfsUserdataBackend, true);
});

// ---------------------------------------------------------------------------
// Part B: Priority-Tiered IO Scheduling & 3D Asset Streaming (web/workers/io_scheduler.js)
// ---------------------------------------------------------------------------

test('W381 io_scheduler: classifyIoPriority classifies 3D world streaming assets while preserving existing rules', () => {
  // Structural world placement & collision assets -> HIGH.
  for (const p of [
    'x64/levels/vice/downtown.ymap',
    'x64/levels/vice/props.YTYP?v=2',
    'x64\\levels\\vice\\terrain_01.ybn',
  ]) {
    assert.equal(classifyIoPriority(p), IO_PRIORITY.HIGH, `Expected HIGH for ${p}`);
    assert.equal(isWorldStructureAssetPath(p), true);
    assert.equal(isWorldStreamingAssetPath(p), true);
  }

  // On-demand 3D mesh, texture dictionary, fragment, and audio wave container assets -> NORMAL.
  for (const p of [
    'x64/models/building_01.ydr',
    'x64/textures/downtown_props.ytd',
    'x64/models/ped_stream.ydd',
    'x64/vehicles/cheetah.yft',
    'x64/audio/sfx/city_ambience.awc',
  ]) {
    assert.equal(classifyIoPriority(p), IO_PRIORITY.NORMAL, `Expected NORMAL for ${p}`);
    assert.equal(isWorldStructureAssetPath(p), false);
    assert.equal(isWorldStreamingAssetPath(p), true);
  }

  // Prefetch overrides world streaming extensions; boot-critical RPFs stay HIGH; shaders/.pgim stay CRITICAL.
  assert.equal(classifyIoPriority('prefetch/distant_city.ytd'), IO_PRIORITY.PREFETCH);
  assert.equal(classifyIoPriority('x64/speculative/far_block.ymap'), IO_PRIORITY.PREFETCH);
  assert.equal(classifyIoPriority('x64/models/prop.ydr.prefetch'), IO_PRIORITY.PREFETCH);
  assert.equal(classifyIoPriority('rpf://game/levels.rpf/downtown.ydr'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('shaders/world_pbr.wgsl.pack'), IO_PRIORITY.CRITICAL);
  assert.equal(classifyIoPriority('guest/main.pgim'), IO_PRIORITY.CRITICAL);
  assert.equal(classifyIoPriority('movies/intro.bk2'), IO_PRIORITY.HIGH);
  assert.equal(classifyIoPriority('x64/data/ui/fonts.rpf'), IO_PRIORITY.HIGH);

  // Explicit caller override still takes precedence.
  assert.equal(classifyIoPriority('x64/levels/vice/downtown.ymap', IO_PRIORITY.PREFETCH), IO_PRIORITY.PREFETCH);
  assert.equal(classifyIoPriority('x64/models/hero.ydr', IO_PRIORITY.CRITICAL), IO_PRIORITY.CRITICAL);
});

test('W381 io_scheduler: reservedForegroundSlots prevents PREFETCH from starving arriving CRITICAL/HIGH foreground reads', async () => {
  let mockNow = 0;
  const gates = new Map();
  const started = [];

  const rangeCache = {
    read(filePath, offset, length) {
      started.push(filePath);
      const gate = deferred();
      gates.set(filePath, gate);
      return gate.promise;
    },
  };

  // Default maxConcurrency=6 reserves 2 foreground slots (maxPrefetchConcurrency=4).
  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 6,
    now: () => mockNow,
    sleep: async () => {},
  });
  assert.equal(scheduler.reservedForegroundSlots, 2);
  assert.equal(scheduler.maxPrefetchConcurrency, 4);

  // Flood 8 PREFETCH reads: only 4 start; 2 slots remain open for instant foreground preemption!
  const prefPromises = [];
  for (let i = 0; i < 8; i += 1) {
    prefPromises.push(
      scheduler.scheduleRead(`prefetch/tile_${i}.ytd`, 0, 64, { priority: IO_PRIORITY.PREFETCH }),
    );
  }

  let stats = scheduler.getStats();
  assert.equal(stats.inFlight, 4);
  assert.equal(stats.activeByPriority.PREFETCH, 4);
  assert.equal(stats.queued, 4);
  assert.equal(stats.prefetchDeferredByCap, 1);

  // At t=50, CRITICAL shader pack and HIGH world .ymap arrive: BOTH must dispatch IMMEDIATELY with 0ms wait!
  mockNow = 50;
  const pCrit = scheduler.scheduleRead('shaders/world.wgsl.pack', 0, 128);
  const pYmap = scheduler.scheduleRead('x64/levels/vice/beach.ymap', 0, 128);

  stats = scheduler.getStats();
  assert.equal(stats.inFlight, 6);
  assert.equal(stats.activeByPriority.CRITICAL, 1);
  assert.equal(stats.activeByPriority.HIGH, 1);
  assert.equal(stats.activeByPriority.PREFETCH, 4);
  assert.equal(stats.foregroundPreemptions, 2);
  assert.equal(stats.tierWaitMs.CRITICAL.max, 0);
  assert.equal(stats.tierWaitMs.HIGH.max, 0);
  assert.equal(stats.worldAssetReads, 9); // 8 .ytd + 1 .ymap

  // Resolve foreground reads first; reserved slots stay protected while 4 prefetches remain in flight.
  gates.get('shaders/world.wgsl.pack').resolve(makeBytes(1, 128));
  gates.get('x64/levels/vice/beach.ymap').resolve(makeBytes(2, 128));
  await Promise.all([pCrit, pYmap]);

  assert.equal(scheduler.getStats().inFlight, 4);
  assert.equal(scheduler.getStats().activeByPriority.PREFETCH, 4);

  // Resolve all prefetches wave by wave; all 8 complete cleanly.
  while (scheduler.getStats().inFlight > 0 || scheduler.getStats().queued > 0 || gates.size > 0) {
    for (const [k, g] of Array.from(gates.entries())) {
      gates.delete(k);
      g.resolve(makeBytes(3, 64));
    }
    await new Promise((r) => setImmediate(r));
  }
  await Promise.all(prefPromises);
  assert.equal(scheduler.getStats().completed, 10);
  assert.equal(scheduler.getStats().queued, 0);
  assert.equal(scheduler.getStats().inFlight, 0);

  // Verify maxPrefetchConcurrency: Infinity allows full saturation when explicitly opted in.
  const satGates = [];
  const fullSat = new IoPriorityScheduler({
    rangeCache: {
      read() {
        const g = deferred();
        satGates.push(g);
        return g.promise;
      },
    },
    maxConcurrency: 4,
    maxPrefetchConcurrency: Infinity,
  });
  assert.equal(fullSat.reservedForegroundSlots, 0);
  assert.equal(fullSat.maxPrefetchConcurrency, Infinity);
  const satPromises = [];
  for (let i = 0; i < 4; i += 1) {
    satPromises.push(fullSat.scheduleRead(`prefetch/sat_${i}.ydr`, 0, 16));
  }
  assert.equal(fullSat.getStats().inFlight, 4);
  assert.equal(fullSat.getStats().prefetchDeferredByCap, 0);
  for (const g of satGates) g.resolve(makeBytes(9, 16));
  await Promise.all(satPromises);
});

test('W381 io_scheduler: anti-starvation aging drains queued NORMAL and PREFETCH across repeated foreground bursts', async () => {
  let mockNow = 0;
  const dispatchOrder = [];
  const gates = new Map();

  const rangeCache = {
    read(filePath, offset, length) {
      dispatchOrder.push(filePath);
      const gate = deferred();
      gates.set(filePath, gate);
      return gate.promise;
    },
  };

  const scheduler = new IoPriorityScheduler({
    rangeCache,
    maxConcurrency: 2,
    reservedForegroundSlots: 1,
    agingWindowMs: 200,
    now: () => mockNow,
    sleep: async () => {},
  });

  // t=0: slot 1 occupied by prefetch-0, slot 2 reserved; prefetch-1 & normal-1 queue up.
  const pPref0 = scheduler.scheduleRead('prefetch/bg_0.ytd', 0, 16);
  const pPref1 = scheduler.scheduleRead('prefetch/bg_1.ytd', 0, 16);
  const pNorm1 = scheduler.scheduleRead('x64/models/prop_1.ydr', 0, 16); // NORMAL dispatches into slot 2!
  const pNorm2 = scheduler.scheduleRead('x64/models/prop_2.ydr', 0, 16); // queued NORMAL

  assert.deepEqual(dispatchOrder, ['prefetch/bg_0.ytd', 'x64/models/prop_1.ydr']);

  // t=100: HIGH burst arrives before aging window elapses -> HIGH runs ahead of queued NORMAL/PREFETCH.
  mockNow = 100;
  const pHigh1 = scheduler.scheduleRead('x64/levels/vice/block_1.ymap', 0, 16);
  gates.get('x64/models/prop_1.ydr').resolve(makeBytes(1, 16));
  await pNorm1;
  await Promise.resolve();
  assert.equal(dispatchOrder[2], 'x64/levels/vice/block_1.ymap');

  // t=500 (> 2 aging windows): another fresh HIGH burst arrives, but aged PREFETCH (enq 0, eff HIGH)
  // and aged NORMAL (enq 0, eff HIGH) must promote and make bounded progress!
  mockNow = 500;
  const pHigh2 = scheduler.scheduleRead('x64/levels/vice/block_2.ymap', 0, 16);
  gates.get('x64/levels/vice/block_1.ymap').resolve(makeBytes(2, 16));
  await pHigh1;
  await Promise.resolve();

  // One of the aged t=0 tasks (prop_2 or bg_1) MUST have promoted ahead of fresh block_2.ymap!
  assert.ok(
    dispatchOrder[3] === 'x64/models/prop_2.ydr' || dispatchOrder[3] === 'prefetch/bg_1.ytd',
    `Expected aged task promotion at index 3, got ${dispatchOrder[3]}`,
  );
  assert.ok(scheduler.getStats().starvationPromotions >= 1);

  // Release remaining tasks.
  mockNow = 600;
  while (scheduler.getStats().inFlight > 0 || scheduler.getStats().queued > 0 || gates.size > 0) {
    for (const [k, g] of Array.from(gates.entries())) {
      gates.delete(k);
      g.resolve(makeBytes(5, 16));
    }
    await new Promise((r) => setImmediate(r));
  }
  await Promise.all([pPref0, pPref1, pNorm2, pHigh2]);

  const finalStats = scheduler.getStats();
  assert.equal(finalStats.completed, 6);
  assert.ok(finalStats.starvationPromotions >= 1);
  assert.equal(finalStats.starvationPromotions, finalStats.agedPromotions);

  // Snapshot isolation on tierWaitMs and clean resetStats().
  finalStats.tierWaitMs.PREFETCH.max = 999999;
  assert.notEqual(scheduler.getStats().tierWaitMs.PREFETCH.max, 999999);

  scheduler.resetStats();
  const afterReset = scheduler.getStats();
  assert.equal(afterReset.foregroundPreemptions, 0);
  assert.equal(afterReset.prefetchDeferredByCap, 0);
  assert.equal(afterReset.starvationPromotions, 0);
  assert.equal(afterReset.worldAssetReads, 0);
  assert.equal(afterReset.tierWaitMs.PREFETCH.max, 0);
});

// ---------------------------------------------------------------------------
// Part C: RPF9 VFS Bounds Hardening & Fast-Path Lookup (web/workers/rpf9_vfs.js)
// ---------------------------------------------------------------------------

function makeHeaderBuf({ entryCount = 1, namesSize = 0, metadataSize = 0 } = {}) {
  const buf = new Uint8Array(RPF9_HEADER_SIZE);
  const view = new DataView(buf.buffer);
  for (let i = 0; i < 4; i += 1) buf[i] = RPF9_MAGIC.charCodeAt(i);
  view.setUint32(4, entryCount >>> 0, true);
  view.setUint32(8, namesSize >>> 0, true);
  view.setUint32(12, RPF_ENCRYPTION_NONE, true);
  view.setUint32(16, metadataSize >>> 0, true);
  view.setUint32(20, 0, true);
  return buf;
}

function makeReader(buf) {
  return async (offset, length) => buf.slice(offset, Math.min(buf.byteLength, offset + length));
}

test('W381 rpf9_vfs: hardens names-table bounds (unterminated strings), sector-offset/size bounds, overlap, and directory TOC corruption', async () => {
  // 1. Unterminated string at end of names table (no trailing NUL byte) MUST throw INVALID_TOC.
  {
    const names = encoder.encode('\0unterminated'); // no trailing '\0'!
    const total = RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE + names.byteLength + 16;
    const buf = new Uint8Array(total);
    buf.set(makeHeaderBuf({ entryCount: 2, namesSize: names.byteLength }), 0);
    const v = new DataView(buf.buffer);
    v.setUint32(RPF9_HEADER_SIZE, (RPF_ENTRY_FLAG_DIRECTORY << 24) | 0, true);
    v.setUint32(RPF9_HEADER_SIZE + 4, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 8, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 16, 1, true); // file entry pointing to "unterminated"
    v.setUint32(RPF9_HEADER_SIZE + 20, total - 16, true);
    v.setUint32(RPF9_HEADER_SIZE + 24, 4, true);
    buf.set(names, RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE);

    await assert.rejects(
      () => RpfArchiveVfs.open(makeReader(buf), buf.byteLength),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_TOC',
    );
  }

  // 2. nameOffset >= namesSize MUST throw INVALID_TOC.
  {
    const toc = new Uint8Array(RPF9_ENTRY_STRIDE);
    new DataView(toc.buffer).setUint32(0, 99, true); // nameOffset 99 >= namesSize 4
    assert.throws(
      () => parseTocEntry(toc, 0, encoder.encode('\0a\0\0'), 0),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_TOC',
    );
  }

  // 3. Compressed entry with compressedSize === 0 and uncompressedSize > 0 MUST throw INVALID_TOC.
  {
    const toc = new Uint8Array(RPF9_ENTRY_STRIDE);
    const tv = new DataView(toc.buffer);
    tv.setUint32(0, (RPF_ENTRY_FLAG_COMPRESSED << 24) | 1, true);
    tv.setUint32(4, 512, true);
    tv.setUint32(8, 1024, true); // uncompressedSize = 1024
    tv.setUint32(12, 0, true); // compressedSize = 0 -> corrupt!
    assert.throws(
      () => parseTocEntry(toc, 0, encoder.encode('\0bad.ydr\0'), 1),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_TOC',
    );
  }

  // 4. OFFSET_SHIFT9 512-byte sector offset exceeding archive totalSize MUST throw TRUNCATED_ARCHIVE.
  {
    const names = encoder.encode('\0big.ytd\0');
    const total = RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE + names.byteLength + 512;
    const buf = new Uint8Array(total);
    buf.set(makeHeaderBuf({ entryCount: 2, namesSize: names.byteLength }), 0);
    const v = new DataView(buf.buffer);
    v.setUint32(RPF9_HEADER_SIZE, (RPF_ENTRY_FLAG_DIRECTORY << 24) | 0, true);
    v.setUint32(RPF9_HEADER_SIZE + 4, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 8, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 16, (RPF_ENTRY_FLAG_OFFSET_SHIFT9 << 24) | 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 20, 100, true); // 100 * 512 = 51200 >> total
    v.setUint32(RPF9_HEADER_SIZE + 24, 64, true);
    buf.set(names, RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE);

    await assert.rejects(
      () => RpfArchiveVfs.open(makeReader(buf), buf.byteLength),
      (err) => err instanceof RpfFormatError && err.code === 'TRUNCATED_ARCHIVE',
    );
  }

  // 5. Non-empty file payload overlapping header/TOC/names MUST throw INVALID_TOC.
  {
    const names = encoder.encode('\0overlap.ydr\0');
    const total = RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE + names.byteLength + 64;
    const buf = new Uint8Array(total);
    buf.set(makeHeaderBuf({ entryCount: 2, namesSize: names.byteLength }), 0);
    const v = new DataView(buf.buffer);
    v.setUint32(RPF9_HEADER_SIZE, (RPF_ENTRY_FLAG_DIRECTORY << 24) | 0, true);
    v.setUint32(RPF9_HEADER_SIZE + 4, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 8, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 16, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 20, 4, true); // offset 4 overlaps header!
    v.setUint32(RPF9_HEADER_SIZE + 24, 16, true);
    buf.set(names, RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE);

    await assert.rejects(
      () => RpfArchiveVfs.open(makeReader(buf), buf.byteLength),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_TOC',
    );
  }

  // 6. Directory firstChildIndex + childCount u32 overflow AND duplicate parent/sibling names MUST throw INVALID_TOC.
  {
    const names = encoder.encode('\0a.ydr\0A.YDR\0');
    const dataOff = RPF9_HEADER_SIZE + 3 * RPF9_ENTRY_STRIDE + names.byteLength;
    const buf = new Uint8Array(dataOff + 16);
    buf.set(makeHeaderBuf({ entryCount: 3, namesSize: names.byteLength }), 0);
    const v = new DataView(buf.buffer);
    v.setUint32(RPF9_HEADER_SIZE, (RPF_ENTRY_FLAG_DIRECTORY << 24) | 0, true);
    v.setUint32(RPF9_HEADER_SIZE + 4, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 8, 2, true);
    // Two case-insensitively identical siblings "a.ydr" and "A.YDR".
    v.setUint32(RPF9_HEADER_SIZE + 16, 1, true);
    v.setUint32(RPF9_HEADER_SIZE + 20, dataOff, true);
    v.setUint32(RPF9_HEADER_SIZE + 24, 4, true);
    v.setUint32(RPF9_HEADER_SIZE + 32, 7, true);
    v.setUint32(RPF9_HEADER_SIZE + 36, dataOff + 4, true);
    v.setUint32(RPF9_HEADER_SIZE + 40, 4, true);
    buf.set(names, RPF9_HEADER_SIZE + 3 * RPF9_ENTRY_STRIDE);

    await assert.rejects(
      () => RpfArchiveVfs.open(makeReader(buf), buf.byteLength),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_TOC',
    );

    // Directory u32 wraparound child bounds (firstChildIndex = 0xFFFFFFFF, childCount = 2).
    const wrapBuf = new Uint8Array(RPF9_HEADER_SIZE + RPF9_ENTRY_STRIDE + 4);
    wrapBuf.set(makeHeaderBuf({ entryCount: 1, namesSize: 4 }), 0);
    const wv = new DataView(wrapBuf.buffer);
    wv.setUint32(RPF9_HEADER_SIZE, (RPF_ENTRY_FLAG_DIRECTORY << 24) | 0, true);
    wv.setUint32(RPF9_HEADER_SIZE + 4, 0xffffffff, true);
    wv.setUint32(RPF9_HEADER_SIZE + 8, 2, true);
    await assert.rejects(
      () => RpfArchiveVfs.open(makeReader(wrapBuf), wrapBuf.byteLength),
      (err) => err instanceof RpfFormatError && err.code === 'INVALID_TOC',
    );
  }
});

test('W381 rpf9_vfs: zero-alloc ASCII canonicalization, hashVirtualPath parity, resolveByPathFast, and bounded cache hygiene', async () => {
  // 1. normalizePathKeyFast & hashVirtualPath parity across canonical, mixed-case, backslash, long, and UTF-8 paths.
  const samples = [
    '',
    '/',
    '\\\\',
    'x64/models/prop_01.ydr',
    '///X64\\\\Models//./PROP_01.YDR/',
    '\\Common/Data/./Levels\\Vice_City.ymap',
    'ui/fonts/Café_Title.gfx',
    `${'a/'.repeat(600)}tail_ASSET.ytd`, // > 1024 chars fallback path
  ];
  for (const s of samples) {
    const refNorm = normalizeVirtualPath(s);
    assert.equal(normalizePathKeyFast(s), refNorm.key, `normalizePathKeyFast mismatch for ${s.slice(0, 40)}`);
    assert.equal(hashVirtualPath(s), joaatHash(refNorm.key), `hashVirtualPath mismatch for ${s.slice(0, 40)}`);
  }

  // Canonical string returns exact same reference (zero allocation).
  const canon = 'x64/models/vehicle_cheetah.yft';
  assert.equal(normalizePathKeyFast(canon), canon);

  // Traversal and NUL rejection on both fast functions.
  for (const bad of ['../secret.ydr', 'x64\\..\\secret.ydr', 'x64/bad\0.ydr']) {
    assert.throws(() => normalizePathKeyFast(bad), (e) => e instanceof RpfFormatError && e.code === 'INVALID_PATH');
    assert.throws(() => hashVirtualPath(bad), (e) => e instanceof RpfFormatError && e.code === 'INVALID_PATH');
  }

  // 2. resolveByPathFast & bounded cache hygiene on 3D world assets.
  const files = [];
  for (let i = 0; i < 64; i += 1) {
    files.push({ path: `x64/world/block_${i}.ydr`, data: `MESH_${i}`, isResource: true });
    files.push({ path: `x64/world/block_${i}.ytd`, data: `TEX_${i}`, isResource: true });
  }
  const archive = buildSyntheticRpf9Buffer(files, { sectorAlign: true });
  const vfs = await RpfArchiveVfs.open(makeReader(archive), archive.byteLength, {
    archivePath: 'x64/world_3d.rpf',
    pathCacheCapacity: 16,
    maxNegativeEntries: 16,
  });

  const scratch = { stat: {} };
  const statRef = scratch.stat;
  const hit1 = vfs.resolveByPathFast('///X64\\\\WORLD\\BLOCK_12.YDR', scratch);
  assert.equal(hit1, scratch);
  assert.equal(hit1.stat, statRef);
  assert.equal(hit1.path, 'x64/world/block_12.ydr');
  assert.equal(hit1.pathHash, hashVirtualPath('x64/world/block_12.ydr'));
  assert.equal(hit1.stat.isFile, true);
  assert.equal(hit1.stat.isResource, true);
  assert.equal(hit1.stat.dataOffset % RPF_SECTOR_SIZE, 0);

  // Switch reused scratch to directory and back to file cleanly.
  const dirHit = vfs.resolveByPathFast('x64/world', scratch);
  assert.equal(dirHit, scratch);
  assert.equal(dirHit.stat.isDirectory, true);
  assert.equal(dirHit.stat.size, undefined);

  const hit2 = vfs.resolveByPathFast('x64/world/block_5.ytd', scratch);
  assert.equal(hit2, scratch);
  assert.equal(hit2.stat.isFile, true);
  assert.equal(hit2.stat.childCount, undefined);

  // Mount table resolveByPathFast parity + bounded cache hygiene under heavy probe churn.
  const table = new RpfVfsMountTable({ pathCacheCapacity: 16, maxNegativeEntries: 16 });
  const mount = table.mount('levels/', vfs);

  const mScratch = { stat: {} };
  const mStatRef = mScratch.stat;
  const mHit = table.resolveByPathFast('\\LEVELS\\X64\\world\\BLOCK_33.YDR', mScratch);
  assert.equal(mHit, mScratch);
  assert.equal(mHit.stat, mStatRef);
  assert.equal(mHit.mountId, mount.id);
  assert.equal(mHit.stat.path, 'levels/x64/world/block_33.ydr');
  assert.equal(table.resolveByPathFast('levels/x64/world/missing.ydr', mScratch), null);

  for (let i = 0; i < 200; i += 1) {
    assert.ok(table.resolveByPathFast(`levels/x64/world/block_${i % 64}.ydr`, mScratch));
    assert.equal(table.resolveByPathFast(`levels/x64/world/missing_${i}.ytd`, mScratch), null);
  }

  const tStats = table.getStats();
  assert.ok(tStats.resolvedPathCacheSize <= 16);
  assert.ok(tStats.negativeLookupEntries <= 16);
  assert.ok(tStats.resolvedPathCacheEvictions > 0);
  assert.ok(tStats.negativeLookupEvictions > 0);
  for (const [k, val] of Object.entries(tStats)) {
    assert.equal(typeof val, 'number', `Mount table stat ${k} must remain numeric`);
    assert.ok(Number.isFinite(val) && val >= 0);
  }
});
