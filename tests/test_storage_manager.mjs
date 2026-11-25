import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  BrowserStorageManager,
  CACHE_ENTRY_KINDS,
  CACHE_INDEX_VERSION,
  DEFAULT_CACHE_INDEX_FILENAME,
  DEFAULT_CRITICAL_THRESHOLD_RATIO,
  DEFAULT_OPFS_CACHE_DIR,
  DEFAULT_QUOTA_BUDGET_RATIO,
  DEFAULT_USERDATA_DB_NAME,
  DEFAULT_USERDATA_STORE_NAME,
  DEFAULT_WARN_THRESHOLD_RATIO,
  QUOTA_PRESSURE_LEVELS,
  SNAPSHOT_FORMAT,
  SNAPSHOT_VERSION,
  USERDATA_PREFIX,
  classifyQuotaPressure,
  decodeBase64,
  encodeBase64,
  formatBytes,
  inferCacheEntryKind,
  validateCacheKey,
  validateUserdataPath,
} from '../web/storage/storage_manager.js';

const require = createRequire(import.meta.url);
const userdataMod = require('../web/workers/userdata.js');
const rangeCacheMod = require('../web/workers/range_cache.js');

function makePatternBytes(length, seed = 29) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = (i * 37 + seed) & 0xff;
  }
  return out;
}

function createMockOpfsDirHandle(name = '', initialEntries = null) {
  // Stores either Uint8Array (file) or nested mock directory handle.
  const entries = new Map();
  if (initialEntries) {
    for (const [k, v] of Object.entries(initialEntries)) {
      entries.set(k, v instanceof Uint8Array ? v.slice() : v);
    }
  }

  const handle = {
    kind: 'directory',
    name,
    _entries: entries,
    async getDirectoryHandle(dirName, opts = {}) {
      const existing = entries.get(dirName);
      if (existing && existing.kind === 'directory') {
        return existing;
      }
      if (!opts.create) {
        const err = new Error(`NotFoundError: ${dirName}`);
        err.name = 'NotFoundError';
        throw err;
      }
      const child = createMockOpfsDirHandle(dirName);
      entries.set(dirName, child);
      return child;
    },
    async getFileHandle(fileName, opts = {}) {
      if (!entries.has(fileName)) {
        if (!opts.create) {
          const err = new Error(`NotFoundError: ${fileName}`);
          err.name = 'NotFoundError';
          throw err;
        }
        entries.set(fileName, new Uint8Array(0));
      }
      return {
        kind: 'file',
        name: fileName,
        async getFile() {
          const data = entries.get(fileName);
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
              pending = chunk instanceof Uint8Array ? chunk.slice() : new Uint8Array(chunk);
            },
            async close() {
              entries.set(fileName, pending);
            },
          };
        },
      };
    },
    async *entries() {
      for (const [entryName, value] of entries.entries()) {
        if (value && value.kind === 'directory') {
          yield [entryName, value];
        } else {
          yield [
            entryName,
            {
              kind: 'file',
              name: entryName,
              async getFile() {
                return {
                  size: value.byteLength,
                  async arrayBuffer() {
                    return value.slice().buffer;
                  },
                };
              },
            },
          ];
        }
      }
    },
    async removeEntry(entryName, opts = {}) {
      if (!entries.has(entryName)) {
        const err = new Error(`NotFoundError: ${entryName}`);
        err.name = 'NotFoundError';
        throw err;
      }
      const target = entries.get(entryName);
      if (target && target.kind === 'directory' && target._entries.size > 0 && !opts.recursive) {
        const err = new Error(`InvalidModificationError: ${entryName} is not empty`);
        err.name = 'InvalidModificationError';
        throw err;
      }
      entries.delete(entryName);
    },
  };

  return handle;
}

function createMockStorageObj({
  usage = 0,
  quota = 0,
  persisted = false,
  persistResult = true,
  opfsRoot = null,
} = {}) {
  let isPersisted = persisted;
  let persistCalls = 0;
  return {
    get persistCalls() {
      return persistCalls;
    },
    async estimate() {
      return { usage, quota };
    },
    async persisted() {
      return isPersisted;
    },
    async persist() {
      persistCalls += 1;
      isPersisted = Boolean(persistResult);
      return isPersisted;
    },
    async getDirectory() {
      if (!opfsRoot) {
        throw new Error('OPFS unavailable');
      }
      return opfsRoot;
    },
  };
}

function createMockIndexedDB({ omitClear = false } = {}) {
  const databases = new Map();

  function schedule(fn) {
    queueMicrotask(fn);
  }

  function makeStore(recordsMap) {
    const store = {
      put(record) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        schedule(() => {
          const clone = {
            ...record,
            data: record.data instanceof Uint8Array ? record.data.slice() : record.data,
          };
          recordsMap.set(record.path, clone);
          req.result = record.path;
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      delete(key) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        schedule(() => {
          recordsMap.delete(key);
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      getAll() {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        schedule(() => {
          req.result = Array.from(recordsMap.values()).map((r) => ({
            ...r,
            data: r.data instanceof Uint8Array ? r.data.slice() : r.data,
          }));
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
    };

    if (!omitClear) {
      store.clear = () => {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        schedule(() => {
          recordsMap.clear();
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      };
    }

    return store;
  }

  return {
    databases,
    open(dbName) {
      const req = {
        result: null,
        error: null,
        onupgradeneeded: null,
        onsuccess: null,
        onerror: null,
      };

      schedule(() => {
        let entry = databases.get(dbName);
        const isNew = !entry;
        if (isNew) {
          entry = { stores: new Map() };
          databases.set(dbName, entry);
        }

        const db = {
          name: dbName,
          objectStoreNames: {
            contains: (name) => entry.stores.has(name),
          },
          createObjectStore(name) {
            if (!entry.stores.has(name)) {
              entry.stores.set(name, new Map());
            }
            return makeStore(entry.stores.get(name));
          },
          transaction(storeName) {
            if (!entry.stores.has(storeName)) {
              throw new Error(`Object store not found: ${storeName}`);
            }
            const tx = {
              error: null,
              oncomplete: null,
              onerror: null,
              onabort: null,
              objectStore(name) {
                const map = entry.stores.get(name);
                const inner = makeStore(map);
                const wrap = (fn) => (...args) => {
                  const innerReq = fn(...args);
                  schedule(() => {
                    schedule(() => {
                      if (tx.oncomplete) tx.oncomplete({ target: tx });
                    });
                  });
                  return innerReq;
                };
                const wrapped = {
                  put: wrap(inner.put),
                  delete: wrap(inner.delete),
                  getAll: wrap(inner.getAll),
                };
                if (inner.clear) {
                  wrapped.clear = wrap(inner.clear);
                }
                return wrapped;
              },
            };
            return tx;
          },
          close() {},
        };

        req.result = db;
        if (isNew && req.onupgradeneeded) {
          req.onupgradeneeded({ target: req });
        }
        if (req.onsuccess) {
          req.onsuccess({ target: req });
        }
      });

      return req;
    },
  };
}

// --- Constants & formatting tests ---

test('storage_manager constants match range_cache.js and userdata.js contracts', () => {
  assert.equal(DEFAULT_OPFS_CACHE_DIR, 'playgta6-cache');
  assert.equal(DEFAULT_USERDATA_DB_NAME, userdataMod.DEFAULT_DB_NAME);
  assert.equal(DEFAULT_USERDATA_STORE_NAME, userdataMod.DEFAULT_STORE_NAME);
  assert.equal(USERDATA_PREFIX, userdataMod.USERDATA_PREFIX);
});

test('formatBytes formats byte sizes cleanly across units and edge cases', () => {
  assert.equal(formatBytes(0), '0 B');
  assert.equal(formatBytes(-42), '0 B');
  assert.equal(formatBytes(Number.NaN), '0 B');
  assert.equal(formatBytes(Number.POSITIVE_INFINITY), '0 B');
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(1024), '1 KiB');
  assert.equal(formatBytes(1536), '1.5 KiB');
  assert.equal(formatBytes(256 * 1024), '256 KiB');
  assert.equal(formatBytes(1024 * 1024), '1 MiB');
  assert.equal(formatBytes(2.5 * 1024 * 1024 * 1024), '2.5 GiB');
  assert.equal(formatBytes(1024 ** 4), '1 TiB');
});

test('encodeBase64 / decodeBase64 round-trip empty, arbitrary binary, and reject invalid input', () => {
  assert.equal(encodeBase64(new Uint8Array(0)), '');
  assert.deepEqual(decodeBase64(''), new Uint8Array(0));

  for (const len of [1, 2, 3, 4, 15, 16, 17, 257, 4099]) {
    const raw = makePatternBytes(len, len);
    const b64 = encodeBase64(raw);
    const decoded = decodeBase64(b64);
    assert.deepEqual(decoded, raw);
  }

  assert.throws(() => decodeBase64('abc'), /length/i);
  assert.throws(() => decodeBase64('!!!!'), /invalid base64/i);
});

// --- Quota & Persistence tests ---

test('getQuotaEstimate and requestPersistentStorage query navigator.storage cleanly', async () => {
  const storageObj = createMockStorageObj({
    usage: 256 * 1024 * 1024,
    quota: 1024 * 1024 * 1024,
    persisted: false,
    persistResult: true,
  });

  const manager = new BrowserStorageManager({
    storageObj,
    opfsRoot: null,
    indexedDBFactory: null,
  });

  const q1 = await manager.getQuotaEstimate();
  assert.deepEqual(q1, {
    supported: true,
    usageBytes: 256 * 1024 * 1024,
    quotaBytes: 1024 * 1024 * 1024,
    availableBytes: 768 * 1024 * 1024,
    percentUsed: 25,
    persisted: false,
  });

  const p = await manager.requestPersistentStorage();
  assert.deepEqual(p, { supported: true, persisted: true });
  assert.equal(storageObj.persistCalls, 1);

  const q2 = await manager.getQuotaEstimate();
  assert.equal(q2.persisted, true);
});

// --- OPFS Range Cache tests ---

test('inspectOpfsCache and clearOpfsCache inspect and purge playgta6-cache entries created by RangeCache', async () => {
  const root = createMockOpfsDirHandle('');
  const cacheDir = await root.getDirectoryHandle(DEFAULT_OPFS_CACHE_DIR, { create: true });

  // Write real RangeCache chunk filenames into playgta6-cache to prove interop.
  const chunkAName = rangeCacheMod.opfsChunkFileName('x64a.rpf', 0, 262144);
  const chunkBName = rangeCacheMod.opfsChunkFileName('x64a.rpf', 1, 262144);
  const chunkA = makePatternBytes(1024, 1);
  const chunkB = makePatternBytes(2048, 2);

  for (const [name, bytes] of [
    [chunkBName, chunkB],
    [chunkAName, chunkA],
  ]) {
    const fh = await cacheDir.getFileHandle(name, { create: true });
    const w = await fh.createWritable();
    await w.write(bytes);
    await w.close();
  }

  const manager = new BrowserStorageManager({
    storageObj: createMockStorageObj({ opfsRoot: root }),
    indexedDBFactory: null,
  });

  const inspected = await manager.inspectOpfsCache();
  assert.equal(inspected.supported, true);
  assert.equal(inspected.dirName, 'playgta6-cache');
  assert.equal(inspected.entryCount, 2);
  assert.equal(inspected.totalBytes, 3072);
  // Entries must be sorted deterministically by name.
  assert.deepEqual(inspected.entries, [
    { name: chunkAName, size: 1024 },
    { name: chunkBName, size: 2048 },
  ]);

  const cleared = await manager.clearOpfsCache();
  assert.deepEqual(cleared, {
    supported: true,
    removedCount: 2,
    freedBytes: 3072,
  });

  const afterClear = await manager.inspectOpfsCache();
  assert.equal(afterClear.supported, true);
  assert.equal(afterClear.entryCount, 0);
  assert.equal(afterClear.totalBytes, 0);
  assert.deepEqual(afterClear.entries, []);
});

test('inspectOpfsCache and clearOpfsCache handle missing playgta6-cache subdir gracefully when OPFS is supported', async () => {
  const emptyRoot = createMockOpfsDirHandle('');
  const manager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: emptyRoot,
    indexedDBFactory: null,
  });

  const inspected = await manager.inspectOpfsCache();
  assert.deepEqual(inspected, {
    supported: true,
    dirName: 'playgta6-cache',
    entryCount: 0,
    totalBytes: 0,
    entries: [],
  });

  const cleared = await manager.clearOpfsCache();
  assert.deepEqual(cleared, {
    supported: true,
    removedCount: 0,
    freedBytes: 0,
  });
});

// --- IndexedDB /userdata Save Management tests ---

test('listUserdataFiles, exportUserdataSnapshot -> clearUserdata -> importUserdataSnapshot round-trip', async () => {
  const mockIdb = createMockIndexedDB();

  // Populate via real UserdataStore from web/workers/userdata.js to verify 100% schema compatibility.
  const store = new userdataMod.UserdataStore({ indexedDB: mockIdb });
  const saveBytes = makePatternBytes(64, 77);
  const settingsBytes = new TextEncoder().encode('{"subtitles":true,"graphics":"high"}');
  await store.saveFile('/userdata/saves/slot2.sav', saveBytes, 1700000002000);
  await store.saveFile('/userdata/config.json', settingsBytes, 1700000001000);
  store.close();

  const manager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: null,
    indexedDBFactory: mockIdb,
    now: () => 1700000099000,
  });

  // 1. Deterministic listing sorted by path.
  const listed = await manager.listUserdataFiles();
  assert.equal(listed.supported, true);
  assert.equal(listed.dbName, 'playgta6-userdata');
  assert.equal(listed.storeName, 'files');
  assert.equal(listed.fileCount, 2);
  assert.equal(listed.totalBytes, saveBytes.byteLength + settingsBytes.byteLength);
  assert.deepEqual(listed.files, [
    {
      path: '/userdata/config.json',
      byteLength: settingsBytes.byteLength,
      updatedAt: 1700000001000,
    },
    {
      path: '/userdata/saves/slot2.sav',
      byteLength: saveBytes.byteLength,
      updatedAt: 1700000002000,
    },
  ]);

  // 2. Export portable JSON/base64 snapshot and verify JSON round-trip.
  const snapshot = await manager.exportUserdataSnapshot();
  assert.equal(snapshot.format, SNAPSHOT_FORMAT);
  assert.equal(snapshot.version, SNAPSHOT_VERSION);
  assert.equal(snapshot.supported, true);
  assert.equal(snapshot.fileCount, 2);
  assert.equal(snapshot.totalBytes, saveBytes.byteLength + settingsBytes.byteLength);

  const serializedJson = JSON.stringify(snapshot);

  // 3. Clear userdata store completely.
  const cleared = await manager.clearUserdata();
  assert.deepEqual(cleared, {
    supported: true,
    clearedCount: 2,
    freedBytes: saveBytes.byteLength + settingsBytes.byteLength,
  });

  const afterClear = await manager.listUserdataFiles();
  assert.equal(afterClear.fileCount, 0);
  assert.equal(afterClear.totalBytes, 0);

  // 4. Import snapshot back from serialized JSON string.
  const imported = await manager.importUserdataSnapshot(serializedJson, { overwrite: true });
  assert.deepEqual(imported, {
    supported: true,
    importedCount: 2,
    skippedCount: 0,
    importedBytes: saveBytes.byteLength + settingsBytes.byteLength,
  });

  // 5. Verify UserdataStore can read restored records with exact byte contents & mtimes.
  const verifyStore = new userdataMod.UserdataStore({ indexedDB: mockIdb });
  const restoredFiles = await verifyStore.listFiles();
  verifyStore.close();

  const byPath = new Map(restoredFiles.map((f) => [f.path, f]));
  assert.deepEqual(byPath.get('/userdata/saves/slot2.sav').data, saveBytes);
  assert.equal(byPath.get('/userdata/saves/slot2.sav').mtime, 1700000002000);
  assert.deepEqual(byPath.get('/userdata/config.json').data, settingsBytes);
  assert.equal(byPath.get('/userdata/config.json').mtime, 1700000001000);

  // 6. Test overwrite: false skips already-present entries and only imports new ones.
  const partialSnapshot = {
    format: SNAPSHOT_FORMAT,
    version: SNAPSHOT_VERSION,
    files: [
      {
        path: '/userdata/config.json',
        encoding: 'base64',
        dataBase64: encodeBase64(new TextEncoder().encode('{"overwritten":true}')),
        updatedAt: 1700000050000,
      },
      {
        path: '/userdata/saves/slot3.sav',
        encoding: 'base64',
        dataBase64: encodeBase64(new Uint8Array([1, 2, 3, 4])),
        updatedAt: 1700000050000,
      },
    ],
  };

  const merged = await manager.importUserdataSnapshot(partialSnapshot, { overwrite: false });
  assert.deepEqual(merged, {
    supported: true,
    importedCount: 1,
    skippedCount: 1,
    importedBytes: 4,
  });

  const checkStore = new userdataMod.UserdataStore({ indexedDB: mockIdb });
  const finalFiles = await checkStore.listFiles();
  checkStore.close();
  const finalMap = new Map(finalFiles.map((f) => [f.path, f]));
  assert.deepEqual(finalMap.get('/userdata/config.json').data, settingsBytes);
  assert.deepEqual(finalMap.get('/userdata/saves/slot3.sav').data, new Uint8Array([1, 2, 3, 4]));
});

test('clearUserdata falls back to per-key delete when IndexedDB store lacks clear()', async () => {
  const mockIdb = createMockIndexedDB({ omitClear: true });
  const store = new userdataMod.UserdataStore({ indexedDB: mockIdb });
  await store.saveFile('/userdata/a.sav', new Uint8Array([10, 20, 30]), 1000);
  store.close();

  const manager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: null,
    indexedDBFactory: mockIdb,
  });

  const cleared = await manager.clearUserdata();
  assert.deepEqual(cleared, { supported: true, clearedCount: 1, freedBytes: 3 });

  const listed = await manager.listUserdataFiles();
  assert.equal(listed.fileCount, 0);
});

test('importUserdataSnapshot rejects path traversal and invalid paths atomically without mutating IndexedDB', async () => {
  const mockIdb = createMockIndexedDB();
  const manager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: null,
    indexedDBFactory: mockIdb,
  });

  const traversalPaths = [
    '../etc/passwd',
    '/etc/passwd',
    '/userdata',
    '/userdata/',
    '/userdata/../etc/passwd',
    '/userdata/saves/../../etc/passwd',
    '/userdata/./save.bin',
    '/userdata/a//b.bin',
    '/userdata/dir/',
    '/userdata/sub\\save.bin',
    '/userdata/bad\0name.bin',
  ];

  for (const badPath of traversalPaths) {
    assert.throws(() => validateUserdataPath(badPath), `validateUserdataPath should reject ${badPath}`);
    await assert.rejects(
      () =>
        manager.importUserdataSnapshot({
          format: SNAPSHOT_FORMAT,
          version: SNAPSHOT_VERSION,
          files: [
            {
              path: '/userdata/legit.sav',
              encoding: 'base64',
              dataBase64: encodeBase64(new Uint8Array([1, 2, 3])),
              updatedAt: 1000,
            },
            {
              path: badPath,
              encoding: 'base64',
              dataBase64: encodeBase64(new Uint8Array([4, 5, 6])),
              updatedAt: 1000,
            },
          ],
        }),
      `importUserdataSnapshot should reject ${badPath}`
    );
  }

  // Because validation is atomic, even '/userdata/legit.sav' must NOT have been written.
  const listed = await manager.listUserdataFiles();
  assert.equal(listed.fileCount, 0);
});

// --- Combined Summary & Graceful Fallback tests ---

test('getSummary combines quota, OPFS cache, and userdata metrics into a HUD-friendly summary', async () => {
  const opfsRoot = createMockOpfsDirHandle('', {
    [DEFAULT_OPFS_CACHE_DIR]: createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, {
      'c_262144_0_61.bin': makePatternBytes(4096, 1),
    }),
  });
  const storageObj = createMockStorageObj({
    usage: 10 * 1024 * 1024,
    quota: 100 * 1024 * 1024,
    persisted: true,
    opfsRoot,
  });
  const mockIdb = createMockIndexedDB();
  const store = new userdataMod.UserdataStore({ indexedDB: mockIdb });
  await store.saveFile('/userdata/profile.sav', makePatternBytes(2048, 5), 1700000000000);
  store.close();

  const manager = new BrowserStorageManager({
    storageObj,
    indexedDBFactory: mockIdb,
  });

  const summary = await manager.getSummary();
  assert.equal(summary.quota.supported, true);
  assert.equal(summary.quota.percentUsed, 10);
  assert.equal(summary.quota.persisted, true);
  assert.equal(summary.quota.usageFormatted, '10 MiB');
  assert.equal(summary.quota.quotaFormatted, '100 MiB');
  assert.equal(summary.quota.availableFormatted, '90 MiB');

  assert.equal(summary.opfsCache.supported, true);
  assert.equal(summary.opfsCache.dirName, 'playgta6-cache');
  assert.equal(summary.opfsCache.entryCount, 1);
  assert.equal(summary.opfsCache.totalBytes, 4096);
  assert.equal(summary.opfsCache.totalFormatted, '4 KiB');

  assert.equal(summary.userdata.supported, true);
  assert.equal(summary.userdata.fileCount, 1);
  assert.equal(summary.userdata.totalBytes, 2048);
  assert.equal(summary.userdata.totalFormatted, '2 KiB');

  assert.equal(summary.totalManagedBytes, 6144);
  assert.equal(summary.totalManagedFormatted, '6 KiB');
});

test('BrowserStorageManager degrades gracefully when Storage, OPFS, and IndexedDB APIs are unavailable', async () => {
  const manager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: null,
    indexedDBFactory: null,
  });

  const quota = await manager.getQuotaEstimate();
  assert.deepEqual(quota, {
    supported: false,
    usageBytes: 0,
    quotaBytes: 0,
    availableBytes: 0,
    percentUsed: 0,
    persisted: false,
  });

  const persist = await manager.requestPersistentStorage();
  assert.deepEqual(persist, { supported: false, persisted: false });

  const opfsInspect = await manager.inspectOpfsCache();
  assert.deepEqual(opfsInspect, {
    supported: false,
    dirName: 'playgta6-cache',
    entryCount: 0,
    totalBytes: 0,
    entries: [],
  });

  const opfsClear = await manager.clearOpfsCache();
  assert.deepEqual(opfsClear, { supported: false, removedCount: 0, freedBytes: 0 });

  const userdataList = await manager.listUserdataFiles();
  assert.equal(userdataList.supported, false);
  assert.deepEqual(userdataList.files, []);

  const exported = await manager.exportUserdataSnapshot();
  assert.equal(exported.supported, false);
  assert.deepEqual(exported.files, []);

  const imported = await manager.importUserdataSnapshot({
    format: SNAPSHOT_FORMAT,
    version: SNAPSHOT_VERSION,
    files: [
      {
        path: '/userdata/save.bin',
        encoding: 'base64',
        dataBase64: encodeBase64(new Uint8Array([1, 2])),
      },
    ],
  });
  assert.deepEqual(imported, {
    supported: false,
    importedCount: 0,
    skippedCount: 0,
    importedBytes: 0,
  });

  const cleared = await manager.clearUserdata();
  assert.deepEqual(cleared, { supported: false, clearedCount: 0, freedBytes: 0 });

  const summary = await manager.getSummary();
  assert.equal(summary.quota.supported, false);
  assert.equal(summary.opfsCache.supported, false);
  assert.equal(summary.userdata.supported, false);
  assert.equal(summary.totalManagedBytes, 0);
  assert.equal(summary.totalManagedFormatted, '0 B');
  assert.equal(summary.pressure, 'unknown');
  assert.equal(summary.quota.pressure, 'unknown');
  assert.equal(summary.quota.usageRatio, null);
  assert.deepEqual(summary.eviction, {
    evictedEntries: 0,
    evictedBytes: 0,
    evictedFormatted: '0 B',
    cleanupCount: 0,
    lastCleanupAtMs: null,
  });
});

// --- W156 Quota Pressure Classification, Eviction Telemetry & Sanitized Usage Snapshot tests ---

test('quota pressure defaults and classifyQuotaPressure classify ok/warn/critical/unknown with safe clamping', () => {
  assert.equal(DEFAULT_WARN_THRESHOLD_RATIO, 0.75);
  assert.equal(DEFAULT_CRITICAL_THRESHOLD_RATIO, 0.9);
  assert.deepEqual(QUOTA_PRESSURE_LEVELS, {
    OK: 'ok',
    WARN: 'warn',
    CRITICAL: 'critical',
    UNKNOWN: 'unknown',
  });
  assert.equal(Object.isFrozen(QUOTA_PRESSURE_LEVELS), true);

  // Positional (usageBytes, quotaBytes) boundaries with default thresholds (0.75 / 0.90)
  assert.equal(classifyQuotaPressure(0, 1000), 'ok');
  assert.equal(classifyQuotaPressure(749, 1000), 'ok');
  assert.equal(classifyQuotaPressure(750, 1000), 'warn');
  assert.equal(classifyQuotaPressure(899, 1000), 'warn');
  assert.equal(classifyQuotaPressure(900, 1000), 'critical');
  assert.equal(classifyQuotaPressure(1000, 1000), 'critical');

  // Safe numeric clamping for over-quota and negative usage
  assert.equal(classifyQuotaPressure(2500, 1000), 'critical');
  assert.equal(classifyQuotaPressure(-50, 1000), 'ok');

  // Non-positive quota, non-finite inputs, and unsupported estimates -> 'unknown'
  assert.equal(classifyQuotaPressure(500, 0), 'unknown');
  assert.equal(classifyQuotaPressure(0, 0), 'unknown');
  assert.equal(classifyQuotaPressure(500, -100), 'unknown');
  assert.equal(classifyQuotaPressure(Number.NaN, 1000), 'unknown');
  assert.equal(classifyQuotaPressure(500, Number.POSITIVE_INFINITY), 'unknown');
  assert.equal(classifyQuotaPressure(null), 'unknown');
  assert.equal(classifyQuotaPressure(undefined), 'unknown');
  assert.equal(classifyQuotaPressure({ supported: false, usageBytes: 950, quotaBytes: 1000 }), 'unknown');
  assert.equal(classifyQuotaPressure({ usageRatio: Number.NaN }), 'unknown');

  // Object shapes: getQuotaEstimate(), navigator.storage.estimate(), and usageRatio
  assert.equal(classifyQuotaPressure({ supported: true, usageBytes: 760, quotaBytes: 1000 }), 'warn');
  assert.equal(classifyQuotaPressure({ usage: 910, quota: 1000 }), 'critical');
  assert.equal(classifyQuotaPressure({ supported: true, usageRatio: 0.4 }), 'ok');
  assert.equal(classifyQuotaPressure({ supported: true, usageRatio: 0.8 }), 'warn');
  assert.equal(classifyQuotaPressure({ supported: true, usageRatio: 1.5 }), 'critical');

  // Configurable thresholds & threshold clamping/normalization
  assert.equal(
    classifyQuotaPressure(600, 1000, { warnThresholdRatio: 0.5, criticalThresholdRatio: 0.8 }),
    'warn'
  );
  assert.equal(
    classifyQuotaPressure(800, 1000, { warnThresholdRatio: 0.5, criticalThresholdRatio: 0.8 }),
    'critical'
  );
  // Inverted thresholds clamp warn down to critical (0.6 / 0.6)
  assert.equal(
    classifyQuotaPressure(599, 1000, { warnThresholdRatio: 0.95, criticalThresholdRatio: 0.6 }),
    'ok'
  );
  assert.equal(
    classifyQuotaPressure(600, 1000, { warnThresholdRatio: 0.95, criticalThresholdRatio: 0.6 }),
    'critical'
  );
  // Only warnThresholdRatio raised above default critical lifts critical to match
  assert.equal(classifyQuotaPressure(920, 1000, { warnThresholdRatio: 0.95 }), 'ok');
  assert.equal(classifyQuotaPressure(950, 1000, { warnThresholdRatio: 0.95 }), 'critical');
  // Non-finite or out-of-range thresholds safely clamp/fallback
  assert.equal(
    classifyQuotaPressure(800, 1000, { warnThresholdRatio: Number.NaN, criticalThresholdRatio: 1.8 }),
    'warn'
  );
  assert.equal(
    classifyQuotaPressure(0, 1000, { warnThresholdRatio: -0.5, criticalThresholdRatio: 0.5 }),
    'warn'
  );
});

test('BrowserStorageManager configures thresholds and nowFn clock cleanly while preserving now fallback', async () => {
  let nowVal = 5000;
  let legacyNowCalls = 0;

  const manager = new BrowserStorageManager({
    storageObj: createMockStorageObj({ usage: 650, quota: 1000 }),
    opfsRoot: null,
    indexedDBFactory: null,
    warnThresholdRatio: 0.6,
    criticalThresholdRatio: 0.85,
    now: () => {
      legacyNowCalls += 1;
      return 9999;
    },
    nowFn: () => nowVal,
  });

  assert.equal(manager.warnThresholdRatio, 0.6);
  assert.equal(manager.criticalThresholdRatio, 0.85);
  assert.equal(manager.classifyQuotaPressure(600, 1000), 'warn');
  assert.equal(manager.classifyQuotaPressure(850, 1000), 'critical');
  assert.equal(manager.classifyQuotaPressure(600, 1000, { warnThresholdRatio: 0.7 }), 'ok');

  const estimate = await manager.getQuotaEstimate();
  assert.equal(manager.classifyQuotaPressure(estimate), 'warn');
  assert.equal(legacyNowCalls, 0);

  const stats = manager.getStats();
  assert.equal(stats.lastQuotaPressure, 'warn');
  assert.equal(stats.lastQuotaUsageRatio, 0.65);
  assert.equal(stats.lastQuotaSampleAtMs, 5000);

  // Legacy `now` option still works when `nowFn` is omitted
  const legacyMgr = new BrowserStorageManager({
    storageObj: createMockStorageObj({ usage: 100, quota: 1000 }),
    opfsRoot: null,
    indexedDBFactory: null,
    now: () => 4242,
  });
  await legacyMgr.getQuotaEstimate();
  assert.equal(legacyMgr.getStats().lastQuotaSampleAtMs, 4242);
});

test('eviction telemetry tracks evictedEntries, evictedBytes, cleanupCount, and lastCleanupAtMs via nowFn', async () => {
  let clockMs = 10000;
  const opfsRoot = createMockOpfsDirHandle('', {
    [DEFAULT_OPFS_CACHE_DIR]: createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, {
      'chunk_a.bin': makePatternBytes(1024, 1),
      'chunk_b.bin': makePatternBytes(2048, 2),
    }),
  });
  const mockIdb = createMockIndexedDB();
  const store = new userdataMod.UserdataStore({ indexedDB: mockIdb });
  await store.saveFile('/userdata/slot1.sav', makePatternBytes(512, 3), 1000);
  await store.saveFile('/userdata/slot2.sav', makePatternBytes(256, 4), 2000);
  store.close();

  const manager = new BrowserStorageManager({
    storageObj: createMockStorageObj({ usage: 800, quota: 1000, opfsRoot }),
    indexedDBFactory: mockIdb,
    nowFn: () => clockMs,
  });

  // Initial stats before any cleanup
  const initialStats = manager.getStats();
  assert.deepEqual(initialStats, {
    evictedEntries: 0,
    evictedBytes: 0,
    evictedFormatted: '0 B',
    cleanupCount: 0,
    lastCleanupAtMs: null,
    warnThresholdRatio: 0.75,
    criticalThresholdRatio: 0.9,
    lastQuotaPressure: 'unknown',
    lastQuotaUsageRatio: null,
    lastQuotaSampleAtMs: null,
  });

  // Mutating returned stats object must not affect internal counters
  initialStats.evictedEntries = 999;
  assert.equal(manager.getStats().evictedEntries, 0);

  // 1. OPFS cache cleanup
  const opfsCleared = await manager.clearOpfsCache();
  assert.deepEqual(opfsCleared, { supported: true, removedCount: 2, freedBytes: 3072 });
  assert.equal(manager.getStats().evictedEntries, 2);
  assert.equal(manager.getStats().evictedBytes, 3072);
  assert.equal(manager.getStats().evictedFormatted, '3 KiB');
  assert.equal(manager.getStats().cleanupCount, 1);
  assert.equal(manager.getStats().lastCleanupAtMs, 10000);

  // 2. Userdata cleanup at a later timestamp accumulates totals
  clockMs = 25000;
  const userdataCleared = await manager.clearUserdata();
  assert.deepEqual(userdataCleared, { supported: true, clearedCount: 2, freedBytes: 768 });
  assert.equal(manager.getStats().evictedEntries, 4);
  assert.equal(manager.getStats().evictedBytes, 3840);
  assert.equal(manager.getStats().cleanupCount, 2);
  assert.equal(manager.getStats().lastCleanupAtMs, 25000);

  // 3. Second OPFS cleanup when empty still records cleanupCount and timestamp
  clockMs = 30000;
  const opfsClearedAgain = await manager.clearOpfsCache();
  assert.deepEqual(opfsClearedAgain, { supported: true, removedCount: 0, freedBytes: 0 });
  assert.equal(manager.getStats().evictedEntries, 4);
  assert.equal(manager.getStats().evictedBytes, 3840);
  assert.equal(manager.getStats().cleanupCount, 3);
  assert.equal(manager.getStats().lastCleanupAtMs, 30000);

  // 4. resetStats clears all telemetry cleanly
  const reset = manager.resetStats();
  assert.equal(reset.evictedEntries, 0);
  assert.equal(reset.evictedBytes, 0);
  assert.equal(reset.cleanupCount, 0);
  assert.equal(reset.lastCleanupAtMs, null);
  assert.equal(reset.lastQuotaPressure, 'unknown');
  assert.equal(reset.lastQuotaUsageRatio, null);
  assert.equal(reset.lastQuotaSampleAtMs, null);
});

test('eviction telemetry ignores unsupported cleanups and handles zero or invalid clock outputs safely', async () => {
  // Unsupported APIs must NOT increment cleanupCount or touch lastCleanupAtMs
  const unsupportedMgr = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: null,
    indexedDBFactory: null,
    nowFn: () => 12345,
  });
  await unsupportedMgr.clearOpfsCache();
  await unsupportedMgr.clearUserdata();
  assert.equal(unsupportedMgr.getStats().cleanupCount, 0);
  assert.equal(unsupportedMgr.getStats().evictedEntries, 0);
  assert.equal(unsupportedMgr.getStats().evictedBytes, 0);
  assert.equal(unsupportedMgr.getStats().lastCleanupAtMs, null);

  // Zero timestamp (0 ms) is preserved as 0 (not treated as falsy null)
  const zeroClockMgr = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: createMockOpfsDirHandle(''),
    indexedDBFactory: null,
    nowFn: () => 0,
  });
  const missingDirClear = await zeroClockMgr.clearOpfsCache();
  assert.deepEqual(missingDirClear, { supported: true, removedCount: 0, freedBytes: 0 });
  assert.equal(zeroClockMgr.getStats().cleanupCount, 1);
  assert.equal(zeroClockMgr.getStats().lastCleanupAtMs, 0);

  // Throwing or non-finite clock must never break cleanup
  const badClockMgr = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: createMockOpfsDirHandle(''),
    indexedDBFactory: null,
    nowFn: () => {
      throw new Error('clock fault');
    },
  });
  const res = await badClockMgr.clearOpfsCache();
  assert.equal(res.supported, true);
  assert.equal(badClockMgr.getStats().cleanupCount, 1);
  assert.equal(badClockMgr.getStats().lastCleanupAtMs, null);
});

test('getSummary and getUsageSnapshot expose quota pressure, eviction telemetry, and sanitized output', async () => {
  let clockMs = 1700001000000;
  let dynamicUsage = 80 * 1024 * 1024;
  const dynamicQuota = 100 * 1024 * 1024;

  const opfsRoot = createMockOpfsDirHandle('', {
    'custom-secret-cache-dir': createMockOpfsDirHandle('custom-secret-cache-dir', {
      'secret_chunk_01.bin': makePatternBytes(4096, 9),
    }),
  });
  const storageObj = {
    async estimate() {
      return { usage: dynamicUsage, quota: dynamicQuota };
    },
    async persisted() {
      return true;
    },
    async getDirectory() {
      return opfsRoot;
    },
  };
  const mockIdb = createMockIndexedDB();
  const store = new userdataMod.UserdataStore({
    indexedDB: mockIdb,
    dbName: 'custom-secret-db',
    storeName: 'custom-secret-store',
  });
  await store.saveFile('/userdata/private/player_slot.sav', makePatternBytes(2048, 11), 1700000500000);
  store.close();

  const manager = new BrowserStorageManager({
    storageObj,
    opfsDirName: 'custom-secret-cache-dir',
    indexedDBFactory: mockIdb,
    dbName: 'custom-secret-db',
    storeName: 'custom-secret-store',
    warnThresholdRatio: 0.75,
    criticalThresholdRatio: 0.9,
    nowFn: () => clockMs,
  });

  // At 80% usage -> 'warn'
  const warnSummary = await manager.getSummary();
  assert.equal(warnSummary.pressure, 'warn');
  assert.equal(warnSummary.quota.pressure, 'warn');
  assert.equal(warnSummary.quota.usageRatio, 0.8);
  assert.equal(warnSummary.quota.warnThresholdRatio, 0.75);
  assert.equal(warnSummary.quota.criticalThresholdRatio, 0.9);
  assert.deepEqual(warnSummary.eviction, {
    evictedEntries: 0,
    evictedBytes: 0,
    evictedFormatted: '0 B',
    cleanupCount: 0,
    lastCleanupAtMs: null,
  });

  // Trigger a cleanup, then escalate usage to 95% -> 'critical'
  clockMs = 1700001005000;
  await manager.clearOpfsCache();
  dynamicUsage = 95 * 1024 * 1024;
  clockMs = 1700001010000;

  const snapshot = await manager.getUsageSnapshot();
  assert.equal(snapshot.capturedAtMs, 1700001010000);
  assert.equal(snapshot.pressure, 'critical');
  assert.deepEqual(snapshot.thresholds, {
    warnThresholdRatio: 0.75,
    criticalThresholdRatio: 0.9,
  });
  assert.equal(snapshot.quota.supported, true);
  assert.equal(snapshot.quota.usageBytes, 95 * 1024 * 1024);
  assert.equal(snapshot.quota.quotaBytes, 100 * 1024 * 1024);
  assert.equal(snapshot.quota.availableBytes, 5 * 1024 * 1024);
  assert.equal(snapshot.quota.usageRatio, 0.95);
  assert.equal(snapshot.quota.percentUsed, 95);
  assert.equal(snapshot.quota.persisted, true);
  assert.equal(snapshot.quota.pressure, 'critical');
  assert.deepEqual(snapshot.opfsCache, {
    supported: true,
    entryCount: 0,
    totalBytes: 0,
    totalFormatted: '0 B',
  });
  assert.deepEqual(snapshot.userdata, {
    supported: true,
    fileCount: 1,
    totalBytes: 2048,
    totalFormatted: '2 KiB',
  });
  assert.equal(snapshot.totalManagedBytes, 2048);
  assert.equal(snapshot.totalManagedFormatted, '2 KiB');
  assert.deepEqual(snapshot.eviction, {
    evictedEntries: 1,
    evictedBytes: 4096,
    evictedFormatted: '4 KiB',
    cleanupCount: 1,
    lastCleanupAtMs: 1700001005000,
  });

  // Snapshot must be JSON-serializable and sanitized (no raw paths, filenames, or custom dir/db labels)
  const serialized = JSON.stringify(snapshot);
  assert.equal(serialized.includes('secret_chunk_01.bin'), false);
  assert.equal(serialized.includes('player_slot.sav'), false);
  assert.equal(serialized.includes('custom-secret-cache-dir'), false);
  assert.equal(serialized.includes('custom-secret-db'), false);
  assert.equal(serialized.includes('custom-secret-store'), false);
  assert.equal('entries' in snapshot.opfsCache, false);
  assert.equal('files' in snapshot.userdata, false);
});

// --- W211 OPFS Quota-Aware LRU Eviction & Integrity-Keyed Shader/PGIM Cache tests ---

test('validateCacheKey and inferCacheEntryKind enforce safe keys and classify pgim/shader-pack/chunk entries', () => {
  assert.equal(DEFAULT_CACHE_INDEX_FILENAME, '.playgta6-cache-index.json');
  assert.equal(CACHE_INDEX_VERSION, 1);
  assert.equal(DEFAULT_QUOTA_BUDGET_RATIO, 0.6);
  assert.equal(Object.isFrozen(CACHE_ENTRY_KINDS), true);

  assert.equal(validateCacheKey('boot.pgim'), 'boot.pgim');
  assert.equal(validateCacheKey('shaders/frame1.wgsl.pack'), 'shaders/frame1.wgsl.pack');
  assert.equal(validateCacheKey('c_262144_0_78363461.bin'), 'c_262144_0_78363461.bin');

  for (const badKey of [
    '',
    '/boot.pgim',
    'shaders/',
    '../escape.pgim',
    'shaders/../boot.pgim',
    'shaders/./pack.wgsl.pack',
    'shaders//pack.wgsl.pack',
    'shaders\\pack.wgsl.pack',
    'boot\0.pgim',
    DEFAULT_CACHE_INDEX_FILENAME,
  ]) {
    assert.throws(() => validateCacheKey(badKey), `validateCacheKey should reject "${badKey}"`);
  }

  assert.equal(inferCacheEntryKind('boot/gta6.pgim'), CACHE_ENTRY_KINDS.PGIM);
  assert.equal(inferCacheEntryKind('BOOT.PGIM'), CACHE_ENTRY_KINDS.PGIM);
  assert.equal(inferCacheEntryKind('shaders/main.wgsl.pack'), CACHE_ENTRY_KINDS.SHADER_PACK);
  assert.equal(inferCacheEntryKind('custom/pass.spv'), CACHE_ENTRY_KINDS.SHADER_PACK);
  assert.equal(inferCacheEntryKind('c_262144_12_abcdef.bin'), CACHE_ENTRY_KINDS.RANGE_CHUNK);
  assert.equal(inferCacheEntryKind('misc/data.bin'), CACHE_ENTRY_KINDS.GENERIC);
});

test('integrity-keyed shader/PGIM cache validates ETag, versionKey, and byte size with automatic invalidation', async () => {
  let clockMs = 1700002000000;
  const opfsRoot = createMockOpfsDirHandle('');
  const manager = new BrowserStorageManager({
    storageObj: createMockStorageObj({ usage: 0, quota: 10 * 1024 * 1024, opfsRoot }),
    indexedDBFactory: null,
    nowFn: () => clockMs,
  });

  const pgimBytes = makePatternBytes(4096, 10);
  const shaderBytes = makePatternBytes(2048, 20);

  const putPgim = await manager.putCacheEntry('boot/gta6.pgim', pgimBytes, {
    etag: '"etag-pgim-v1"',
    versionKey: 'pgim-sha256-aaa',
    pinned: true,
  });
  assert.equal(putPgim.supported, true);
  assert.equal(putPgim.stored, true);
  assert.equal(putPgim.replaced, false);
  assert.equal(putPgim.versionChanged, false);
  assert.deepEqual(putPgim.meta, {
    key: 'boot/gta6.pgim',
    sizeBytes: 4096,
    lastAccessedMs: 1700002000000,
    createdAtMs: 1700002000000,
    etag: '"etag-pgim-v1"',
    versionKey: 'pgim-sha256-aaa',
    pinned: true,
    kind: 'pgim',
  });

  clockMs += 1000;
  await manager.putCacheEntry('shaders/main.wgsl.pack', shaderBytes, {
    etag: '"etag-shader-v1"',
    versionKey: 'shader-pack-v1',
    pinned: false,
  });

  // Internal index file must NOT leak into inspectOpfsCache().
  const inspected = await manager.inspectOpfsCache();
  assert.equal(inspected.entryCount, 2);
  assert.equal(inspected.totalBytes, 6144);
  assert.deepEqual(
    inspected.entries.map((e) => e.name),
    ['boot/gta6.pgim', 'shaders/main.wgsl.pack']
  );

  // Matching ETag & versionKey -> hit and updates lastAccessedMs.
  clockMs += 5000;
  const hitPgim = await manager.getCacheEntry('boot/gta6.pgim', {
    etag: '"etag-pgim-v1"',
    versionKey: 'pgim-sha256-aaa',
    expectedSizeBytes: 4096,
  });
  assert.equal(hitPgim.hit, true);
  assert.equal(hitPgim.invalidated, false);
  assert.deepEqual(hitPgim.data, pgimBytes);
  assert.equal(hitPgim.meta.lastAccessedMs, 1700002006000);

  // ETag mismatch invalidates & removes shader pack from OPFS.
  clockMs += 1000;
  const etagMiss = await manager.getCacheEntry('shaders/main.wgsl.pack', {
    etag: '"etag-shader-v2"',
  });
  assert.equal(etagMiss.hit, false);
  assert.equal(etagMiss.invalidated, true);
  assert.equal(etagMiss.reason, 'etag-mismatch');
  assert.equal(etagMiss.data, null);
  assert.equal(manager.getCacheEntryMeta('shaders/main.wgsl.pack'), null);
  assert.equal((await manager.inspectOpfsCache()).entryCount, 1);

  // Re-store shader pack under v2, then test versionKey mismatch invalidation.
  await manager.putCacheEntry('shaders/main.wgsl.pack', shaderBytes, {
    etag: '"etag-shader-v2"',
    versionKey: 'shader-pack-v2',
  });
  const versionMiss = await manager.getCacheEntry('shaders/main.wgsl.pack', {
    versionKey: 'shader-pack-v3',
  });
  assert.equal(versionMiss.hit, false);
  assert.equal(versionMiss.invalidated, true);
  assert.equal(versionMiss.reason, 'version-mismatch');
  assert.equal(manager.getCacheEntryMeta('shaders/main.wgsl.pack'), null);

  // Size integrity mismatch also invalidates and removes entry.
  await manager.putCacheEntry('shaders/corrupt.wgsl.pack', makePatternBytes(512, 33), {
    versionKey: 'v1',
  });
  const sizeMiss = await manager.getCacheEntry('shaders/corrupt.wgsl.pack', {
    versionKey: 'v1',
    expectedSizeBytes: 1024,
  });
  assert.equal(sizeMiss.hit, false);
  assert.equal(sizeMiss.invalidated, true);
  assert.equal(sizeMiss.reason, 'size-mismatch');

  // Replacing an existing entry with a new versionKey counts as versionChanged + invalidation.
  const replaced = await manager.putCacheEntry('boot/gta6.pgim', makePatternBytes(4096, 99), {
    etag: '"etag-pgim-v2"',
    versionKey: 'pgim-sha256-bbb',
    pinned: true,
  });
  assert.equal(replaced.replaced, true);
  assert.equal(replaced.versionChanged, true);

  const stats = manager.getCacheStats();
  assert.equal(stats.hits, 1);
  assert.equal(stats.misses, 3);
  assert.equal(stats.invalidations, 4); // etag-mismatch + version-mismatch + size-mismatch + versionChanged replace
  assert.equal(stats.entryCount, 1);
  assert.equal(stats.totalBytes, 4096);
});

test('invalidateStaleCacheEntries bulk-purges mismatched shader packs and persists metadata across reload', async () => {
  let clockMs = 100000;
  const opfsRoot = createMockOpfsDirHandle('');

  const writer = new BrowserStorageManager({
    storageObj: null,
    opfsRoot,
    indexedDBFactory: null,
    nowFn: () => clockMs,
  });

  await writer.putCacheEntry('boot/active.pgim', makePatternBytes(2048, 1), {
    versionKey: 'pgim-v2',
    etag: '"pgim-2"',
    pinned: true,
  });
  clockMs += 100;
  await writer.putCacheEntry('shaders/old_a.wgsl.pack', makePatternBytes(1024, 2), {
    versionKey: 'shaders-v1',
    etag: '"s1"',
  });
  clockMs += 100;
  await writer.putCacheEntry('shaders/old_b.wgsl.pack', makePatternBytes(1024, 3), {
    versionKey: 'shaders-v1',
    etag: '"s1"',
    pinned: true,
  });
  clockMs += 100;
  await writer.putCacheEntry('shaders/fresh.wgsl.pack', makePatternBytes(1536, 4), {
    versionKey: 'shaders-v2',
    etag: '"s2"',
  });

  // Simulate browser reload with a brand-new manager sharing the same OPFS root.
  const reloaded = new BrowserStorageManager({
    storageObj: null,
    opfsRoot,
    indexedDBFactory: null,
    nowFn: () => clockMs + 500,
  });

  const synced = await reloaded.syncCacheIndex();
  assert.equal(synced.supported, true);
  assert.equal(synced.entryCount, 4);
  assert.equal(synced.totalBytes, 5632);
  assert.equal(synced.discoveredCount, 0);
  assert.equal(reloaded.getCacheEntryMeta('boot/active.pgim').pinned, true);
  assert.equal(reloaded.getCacheEntryMeta('boot/active.pgim').versionKey, 'pgim-v2');

  // Bulk invalidate stale shader packs while skipping pinned entries first.
  const bulk1 = await reloaded.invalidateStaleCacheEntries({
    kind: CACHE_ENTRY_KINDS.SHADER_PACK,
    expectedVersionKey: 'shaders-v2',
    includePinned: false,
  });
  assert.equal(bulk1.invalidatedCount, 1);
  assert.deepEqual(bulk1.invalidatedKeys, ['shaders/old_a.wgsl.pack']);
  assert.equal(bulk1.freedBytes, 1024);
  assert.equal(bulk1.skippedPinnedCount, 1);

  // Per-key manifest map invalidation with includePinned: true (default) clears stale pinned pack too.
  const bulk2 = await reloaded.invalidateStaleCacheEntries({
    entries: {
      'boot/active.pgim': { versionKey: 'pgim-v2', etag: '"pgim-2"' },
      'shaders/old_b.wgsl.pack': { versionKey: 'shaders-v2' },
    },
  });
  assert.equal(bulk2.invalidatedCount, 1);
  assert.deepEqual(bulk2.invalidatedKeys, ['shaders/old_b.wgsl.pack']);
  assert.equal(bulk2.freedBytes, 1024);

  // Active PGIM and fresh shader pack remain intact.
  assert.deepEqual(
    reloaded.listCacheEntries().map((e) => e.key),
    ['boot/active.pgim', 'shaders/fresh.wgsl.pack']
  );
});

test('evictToBudget enforces deterministic LRU eviction order including touch promotion and timestamp ties', async () => {
  let clockMs = 1000;
  const opfsRoot = createMockOpfsDirHandle('');
  const manager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot,
    indexedDBFactory: null,
    nowFn: () => clockMs,
  });

  // Store 4 entries of 1000 bytes each with advancing timestamps.
  await manager.putCacheEntry('chunk_1.bin', makePatternBytes(1000, 1));
  clockMs = 2000;
  await manager.putCacheEntry('chunk_2.bin', makePatternBytes(1000, 2));
  clockMs = 3000;
  await manager.putCacheEntry('chunk_3.bin', makePatternBytes(1000, 3));
  clockMs = 4000;
  await manager.putCacheEntry('chunk_4.bin', makePatternBytes(1000, 4));

  // Access chunk_1 via getCacheEntry (promotes to MRU at t=5000) and touch chunk_2 at t=6000.
  clockMs = 5000;
  const hit1 = await manager.getCacheEntry('chunk_1.bin');
  assert.equal(hit1.hit, true);
  assert.equal(hit1.meta.lastAccessedMs, 5000);

  clockMs = 6000;
  const touched2 = await manager.touchCacheEntry('chunk_2.bin');
  assert.equal(touched2.lastAccessedMs, 6000);

  // Expected LRU order (oldest -> newest): chunk_3 (3000), chunk_4 (4000), chunk_1 (5000), chunk_2 (6000).
  assert.deepEqual(
    manager.listCacheEntries({ sortBy: 'lru' }).map((e) => e.key),
    ['chunk_3.bin', 'chunk_4.bin', 'chunk_1.bin', 'chunk_2.bin']
  );

  // Evict down to 2000 bytes -> must evict chunk_3 then chunk_4 deterministically.
  const ev = await manager.evictToBudget(2000);
  assert.deepEqual(ev, {
    supported: true,
    budgetBytes: 2000,
    beforeBytes: 4000,
    afterBytes: 2000,
    evictedEntries: 2,
    evictedBytes: 2000,
    evictedKeys: ['chunk_3.bin', 'chunk_4.bin'],
    skippedPinnedCount: 0,
    skippedPinnedBytes: 0,
    budgetMet: true,
  });

  // Frozen clock tie-breaking: write 3 entries at the exact same timestamp (t=7000).
  clockMs = 7000;
  await manager.putCacheEntry('tie_z.bin', makePatternBytes(500, 10));
  await manager.putCacheEntry('tie_a.bin', makePatternBytes(500, 11));
  await manager.putCacheEntry('tie_m.bin', makePatternBytes(500, 12));

  // Total is now 2000 + 1500 = 3500. Evict down to 500 -> chunk_1, chunk_2, tie_z, tie_a evicted in exact accessOrder!
  const evTies = await manager.evictToBudget(500);
  assert.deepEqual(evTies.evictedKeys, ['chunk_1.bin', 'chunk_2.bin', 'tie_z.bin', 'tie_a.bin']);
  assert.equal(evTies.afterBytes, 500);
  assert.equal(evTies.budgetMet, true);
  assert.deepEqual(
    manager.listCacheEntries().map((e) => e.key),
    ['tie_m.bin']
  );

  // Invalid maxBytes rejects cleanly with RangeError.
  await assert.rejects(() => manager.evictToBudget(-1), RangeError);
  await assert.rejects(() => manager.evictToBudget(Number.NaN), RangeError);
  await assert.rejects(() => manager.evictToBudget(Number.POSITIVE_INFINITY), RangeError);
});

test('evictToBudget protects pinned boot .pgim and active shader pack even at budget=0 and allows eviction after unpin', async () => {
  let clockMs = 10000;
  const opfsRoot = createMockOpfsDirHandle('');
  const manager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot,
    indexedDBFactory: null,
    nowFn: () => clockMs,
  });

  // Oldest entries are pinned boot PGIM and active shader pack.
  await manager.putCacheEntry('boot/gta6_boot.pgim', makePatternBytes(3000, 1), {
    versionKey: 'pgim-boot-v1',
    pinned: true,
  });
  clockMs += 100;
  await manager.putCacheEntry('shaders/active.wgsl.pack', makePatternBytes(2000, 2), {
    versionKey: 'shader-active-v1',
    pinned: false,
  });
  // Pin shader pack after creation via pinCacheEntry().
  const pinRes = await manager.pinCacheEntry('shaders/active.wgsl.pack');
  assert.equal(pinRes.found, true);
  assert.equal(pinRes.pinned, true);

  // Newer unpinned streaming chunks.
  clockMs += 100;
  await manager.putCacheEntry('chunks/area_a.bin', makePatternBytes(1500, 3));
  clockMs += 100;
  await manager.putCacheEntry('chunks/area_b.bin', makePatternBytes(1500, 4));

  // Total = 8000 (5000 pinned + 3000 unpinned). Request budget=6000 -> evicts both chunks? Only area_a (8000 - 1500 = 6500 > 6000) then area_b (5000 <= 6000).
  const ev1 = await manager.evictToBudget(6000);
  assert.deepEqual(ev1.evictedKeys, ['chunks/area_a.bin', 'chunks/area_b.bin']);
  assert.equal(ev1.skippedPinnedCount, 2);
  assert.equal(ev1.skippedPinnedBytes, 5000);
  assert.equal(ev1.afterBytes, 5000);
  assert.equal(ev1.budgetMet, true);

  // Even at maxBytes=0, pinned entries MUST survive and budgetMet reports false.
  const evZero = await manager.evictToBudget(0);
  assert.equal(evZero.evictedEntries, 0);
  assert.equal(evZero.evictedBytes, 0);
  assert.equal(evZero.skippedPinnedCount, 2);
  assert.equal(evZero.skippedPinnedBytes, 5000);
  assert.equal(evZero.afterBytes, 5000);
  assert.equal(evZero.budgetMet, false);

  // deleteCacheEntry with respectPinned: true also skips pinned entries.
  const blockedDelete = await manager.deleteCacheEntry('boot/gta6_boot.pgim', {
    respectPinned: true,
  });
  assert.equal(blockedDelete.removed, false);
  assert.equal(blockedDelete.skippedPinned, true);

  // Unpin shader pack and evict to 3000 -> shader pack is now evicted, boot .pgim stays!
  const unpinRes = await manager.unpinCacheEntry('shaders/active.wgsl.pack');
  assert.equal(unpinRes.found, true);
  assert.equal(unpinRes.pinned, false);

  const evAfterUnpin = await manager.evictToBudget(3000);
  assert.deepEqual(evAfterUnpin.evictedKeys, ['shaders/active.wgsl.pack']);
  assert.equal(evAfterUnpin.evictedEntries, 1);
  assert.equal(evAfterUnpin.evictedBytes, 2000);
  assert.equal(evAfterUnpin.afterBytes, 3000);
  assert.equal(evAfterUnpin.budgetMet, true);
  assert.deepEqual(
    manager.listCacheEntries().map((e) => e.key),
    ['boot/gta6_boot.pgim']
  );
});

test('quota budget enforcement, maxCacheBytes auto-eviction, QuotaExceededError recovery, and getCacheStats telemetry', async () => {
  let clockMs = 50000;
  let dynamicUsage = 0;
  const totalQuota = 10000;

  const opfsRoot = createMockOpfsDirHandle('');
  const storageObj = {
    async estimate() {
      return { usage: dynamicUsage, quota: totalQuota };
    },
    async persisted() {
      return true;
    },
    async getDirectory() {
      return opfsRoot;
    },
  };

  // Configure maxCacheBytes = 4000 so putCacheEntry auto-evicts older unpinned items.
  const manager = new BrowserStorageManager({
    storageObj,
    indexedDBFactory: null,
    maxCacheBytes: 4000,
    nowFn: () => clockMs,
  });

  await manager.putCacheEntry('boot.pgim', makePatternBytes(1500, 1), {
    versionKey: 'v1',
    pinned: true,
  });
  clockMs += 100;
  await manager.putCacheEntry('old_chunk.bin', makePatternBytes(1500, 2));
  clockMs += 100;
  await manager.putCacheEntry('mid_chunk.bin', makePatternBytes(1000, 3));
  // Total = 4000 (within maxCacheBytes).
  assert.equal(manager.getCacheStats().entryCount, 3);
  assert.equal(manager.getCacheStats().totalBytes, 4000);

  // Writing a 1200-byte entry pushes total to 5200 -> auto-evicts old_chunk.bin (1500B), leaving 3700B <= 4000B.
  clockMs += 100;
  dynamicUsage = 8500; // 85% quota usage -> 'warn'
  const autoRes = await manager.putCacheEntry('new_chunk.bin', makePatternBytes(1200, 4));
  assert.equal(autoRes.stored, true);
  assert.equal(autoRes.evictedEntries, 1);
  assert.equal(autoRes.evictedBytes, 1500);
  assert.equal(manager.getCacheEntryMeta('old_chunk.bin'), null);
  assert.ok(manager.getCacheEntryMeta('boot.pgim'));
  assert.ok(manager.getCacheEntryMeta('new_chunk.bin'));

  const midStats = manager.getCacheStats();
  assert.equal(midStats.entryCount, 3);
  assert.equal(midStats.totalBytes, 3700);
  assert.equal(midStats.pinnedCount, 1);
  assert.equal(midStats.pinnedBytes, 1500);
  assert.equal(midStats.evictedEntries, 1);
  assert.equal(midStats.evictedBytes, 1500);
  assert.equal(midStats.quotaUsagePct, 85);
  assert.equal(midStats.budgetUsagePct, 92.5);
  assert.equal(midStats.quotaPressure, 'warn');

  // Mutating returned getCacheStats() object must not corrupt internal counters.
  midStats.evictedEntries = 9999;
  assert.equal(manager.getCacheStats().evictedEntries, 1);
  assert.equal(JSON.stringify(manager.getCacheStats()).includes('boot.pgim'), false);

  // Quota-aware enforceQuotaBudget(): non-cache usage = 8500 - 3700 = 4800.
  // Target 70% of 10000 = 7000 -> allowed cache budget = 7000 - 4800 = 2200 -> evicts mid_chunk (1000) + new_chunk (1200)?
  // 3700 - 1000 = 2700 > 2200 -> also evicts new_chunk -> 1500 <= 2200 (boot.pgim pinned survives!).
  dynamicUsage = 8500;
  const enforced = await manager.enforceQuotaBudget({
    targetUsageRatio: 0.7,
    onlyUnderPressure: true,
  });
  assert.equal(enforced.supported, true);
  assert.equal(enforced.pressure, 'warn');
  assert.equal(enforced.targetBudgetBytes, 2200);
  assert.deepEqual(enforced.evictedKeys, ['mid_chunk.bin', 'new_chunk.bin']);
  assert.equal(enforced.afterBytes, 1500);
  assert.equal(enforced.budgetMet, true);

  // When pressure is 'ok' and onlyUnderPressure: true, enforceQuotaBudget skips eviction.
  dynamicUsage = 3000; // 30% -> 'ok'
  const skipped = await manager.enforceQuotaBudget({
    targetUsageRatio: 0.1,
    onlyUnderPressure: true,
  });
  assert.equal(skipped.pressure, 'ok');
  assert.equal(skipped.evictedEntries, 0);

  // Verify overall telemetry in getStats() also captured all 3 LRU evictions.
  assert.equal(manager.getStats().evictedEntries, 3);
  assert.equal(manager.getStats().evictedBytes, 3700);
  assert.equal(manager.getCacheStats().evictedEntries, 3);
  assert.equal(manager.getCacheStats().evictedBytes, 3700);
  assert.equal(manager.getCacheStats().quotaUsagePct, 30);

  // Fallback quotaUsagePct uses maxCacheBytes when no browser quota is available.
  const noQuotaMgr = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: createMockOpfsDirHandle(''),
    indexedDBFactory: null,
    maxCacheBytes: 2000,
    nowFn: () => clockMs,
  });
  await noQuotaMgr.putCacheEntry('sample.bin', makePatternBytes(500, 7));
  assert.equal(noQuotaMgr.getCacheStats().quotaUsagePct, 25);
  assert.equal(noQuotaMgr.getCacheStats().budgetUsagePct, 25);

  // QuotaExceededError recovery during putCacheEntry evicts unpinned entries and retries once.
  const flakyCacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR);
  let failOnceOnKey = 'incoming.pgim';
  const origGetFileHandle = flakyCacheDir.getFileHandle.bind(flakyCacheDir);
  flakyCacheDir.getFileHandle = async (fileName, opts) => {
    const fh = await origGetFileHandle(fileName, opts);
    if (fileName === failOnceOnKey) {
      const origCreateWritable = fh.createWritable.bind(fh);
      fh.createWritable = async () => {
        const w = await origCreateWritable();
        const origWrite = w.write.bind(w);
        w.write = async (chunk) => {
          if (failOnceOnKey === fileName) {
            failOnceOnKey = null;
            const err = new Error('QuotaExceededError: OPFS full');
            err.name = 'QuotaExceededError';
            throw err;
          }
          return origWrite(chunk);
        };
        return w;
      };
    }
    return fh;
  };
  const flakyRoot = createMockOpfsDirHandle('', {
    [DEFAULT_OPFS_CACHE_DIR]: flakyCacheDir,
  });
  const recoveryMgr = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: flakyRoot,
    indexedDBFactory: null,
    nowFn: () => (clockMs += 10),
  });
  await recoveryMgr.putCacheEntry('evictable.bin', makePatternBytes(1024, 1));
  await recoveryMgr.putCacheEntry('pinned.pgim', makePatternBytes(1024, 2), { pinned: true });
  const recovered = await recoveryMgr.putCacheEntry('incoming.pgim', makePatternBytes(512, 3), {
    versionKey: 'inc-v1',
  });
  assert.equal(recovered.stored, true);
  assert.equal(recovered.quotaRecovered, true);
  assert.equal(recovered.evictedEntries, 1);
  assert.equal(recoveryMgr.getCacheEntryMeta('evictable.bin'), null);
  assert.ok(recoveryMgr.getCacheEntryMeta('pinned.pgim'));
  assert.ok(recoveryMgr.getCacheEntryMeta('incoming.pgim'));
});

test('cache management discovers pre-existing OPFS chunks, supports registerCacheEntry, and degrades cleanly without OPFS', async () => {
  const chunkName = rangeCacheMod.opfsChunkFileName('x64b.rpf', 0, 262144);
  const opfsRoot = createMockOpfsDirHandle('', {
    [DEFAULT_OPFS_CACHE_DIR]: createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, {
      [chunkName]: makePatternBytes(2048, 5),
      'legacy.bin': makePatternBytes(1024, 6),
    }),
  });

  const manager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot,
    indexedDBFactory: null,
    nowFn: () => 90000,
  });

  // Register metadata on an existing RangeCache file and pin it.
  const reg = await manager.registerCacheEntry(chunkName, {
    etag: '"chunk-e1"',
    versionKey: 'rpf-v1',
    pinned: true,
  });
  assert.equal(reg.supported, true);
  assert.equal(reg.found, true);
  assert.equal(reg.meta.sizeBytes, 2048);
  assert.equal(reg.meta.kind, CACHE_ENTRY_KINDS.RANGE_CHUNK);
  assert.equal(reg.meta.pinned, true);

  // Evict down to 2048 -> removes untracked legacy.bin while keeping pinned chunk.
  const ev = await manager.evictToBudget(2048);
  assert.deepEqual(ev.evictedKeys, ['legacy.bin']);
  assert.equal(ev.afterBytes, 2048);
  assert.equal(ev.skippedPinnedCount, 1);

  // clearOpfsCache clears metadata too and updates cache stats cleanly.
  const cleared = await manager.clearOpfsCache();
  assert.deepEqual(cleared, { supported: true, removedCount: 1, freedBytes: 2048 });
  assert.equal(manager.getCacheStats().entryCount, 0);
  assert.equal(manager.getCacheStats().totalBytes, 0);
  assert.equal(manager.getCacheStats().evictedEntries, 2);
  assert.equal(manager.getCacheStats().evictedBytes, 3072);

  // Unsupported OPFS degrades cleanly across all new cache APIs.
  const unsupportedMgr = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: null,
    indexedDBFactory: null,
  });
  assert.equal((await unsupportedMgr.syncCacheIndex()).supported, false);
  assert.equal((await unsupportedMgr.putCacheEntry('a.pgim', new Uint8Array([1]))).supported, false);
  assert.equal((await unsupportedMgr.getCacheEntry('a.pgim')).supported, false);
  assert.equal((await unsupportedMgr.evictToBudget(1024)).supported, false);
  assert.equal((await unsupportedMgr.enforceQuotaBudget()).supported, false);
  assert.equal(unsupportedMgr.getCacheStats().entryCount, 0);
  assert.equal(unsupportedMgr.getCacheStats().quotaUsagePct, 0);
});
