import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  BrowserStorageManager,
  DEFAULT_CRITICAL_THRESHOLD_RATIO,
  DEFAULT_OPFS_CACHE_DIR,
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
