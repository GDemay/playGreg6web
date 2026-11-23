import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  BrowserStorageManager,
  DEFAULT_OPFS_CACHE_DIR,
  DEFAULT_USERDATA_DB_NAME,
  DEFAULT_USERDATA_STORE_NAME,
  SNAPSHOT_FORMAT,
  SNAPSHOT_VERSION,
  USERDATA_PREFIX,
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
});
