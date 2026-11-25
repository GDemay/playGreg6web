import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const userdataMod = require('../web/workers/userdata.js');

const {
  UserdataStore,
  DEFAULT_DB_NAME,
  DEFAULT_STORE_NAME,
  DEFAULT_RESTORE_TIMEOUT_MS,
  USERDATA_PREFIX,
  ARCHIVE_FORMAT,
  ARCHIVE_VERSION,
  validateUserdataPath,
  validateUserdataPrefix,
  encodeBase64,
  decodeBase64,
} = userdataMod;

function createMemoryBackend(options = {}) {
  const map = new Map();
  let closed = false;

  const backend = {
    map,
    isClosed: () => closed,
    async put(record) {
      map.set(record.path, {
        path: record.path,
        data: record.data instanceof Uint8Array ? record.data.slice() : record.data,
        mtime: record.mtime,
      });
    },
    async delete(key) {
      map.delete(key);
    },
    async getAll() {
      return Array.from(map.values()).map((r) => ({
        ...r,
        data: r.data instanceof Uint8Array ? r.data.slice() : r.data,
      }));
    },
    close() {
      closed = true;
    },
  };

  if (options.includeGet !== false) {
    backend.get = async (key) => {
      const rec = map.get(key);
      if (!rec) return null;
      return {
        ...rec,
        data: rec.data instanceof Uint8Array ? rec.data.slice() : rec.data,
      };
    };
  }

  return backend;
}

function createMockIndexedDB(options = {}) {
  const databases = new Map();
  const upgradeEvents = [];

  function schedule(fn) {
    queueMicrotask(fn);
  }

  function makeStore(recordsMap) {
    return {
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
      get(key) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        schedule(() => {
          const rec = recordsMap.get(key);
          req.result = rec
            ? {
                ...rec,
                data: rec.data instanceof Uint8Array ? rec.data.slice() : rec.data,
              }
            : undefined;
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
        if (options.hangGetAll) {
          return req;
        }
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
  }

  return {
    databases,
    upgradeEvents,
    open(dbName, version = 1) {
      const req = {
        result: null,
        error: null,
        onupgradeneeded: null,
        onsuccess: null,
        onerror: null,
      };

      if (options.hangOpen) {
        return req;
      }

      if (options.failOpen) {
        schedule(() => {
          req.error = new Error('Simulated IndexedDB open failure');
          if (req.onerror) req.onerror({ target: req });
        });
        return req;
      }

      schedule(() => {
        let entry = databases.get(dbName);
        const needsUpgrade = !entry || version > entry.version;
        if (!entry) {
          entry = { version, stores: new Map(), keyPaths: new Map() };
          databases.set(dbName, entry);
        } else if (version > entry.version) {
          entry.version = version;
        }

        const db = {
          name: dbName,
          version: entry.version,
          objectStoreNames: {
            contains: (name) => entry.stores.has(name),
          },
          createObjectStore(name, storeOpts = {}) {
            if (!entry.stores.has(name)) {
              entry.stores.set(name, new Map());
              entry.keyPaths.set(name, storeOpts.keyPath || null);
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
                  if (!options.hangGetAll) {
                    schedule(() => {
                      schedule(() => {
                        if (tx.oncomplete) tx.oncomplete({ target: tx });
                      });
                    });
                  }
                  return innerReq;
                };
                const wrapped = {
                  put: wrap(inner.put),
                  delete: wrap(inner.delete),
                  getAll: wrap(inner.getAll),
                };
                if (options.omitStoreGet !== true) {
                  wrapped.get = wrap(inner.get);
                }
                return wrapped;
              },
            };
            return tx;
          },
          close() {},
        };

        req.result = db;
        if (needsUpgrade && req.onupgradeneeded) {
          upgradeEvents.push({ dbName, version });
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

function createMockFsAdapter(options = {}) {
  const dirs = new Set();
  const files = new Map();
  const mtimes = new Map();

  const adapter = {
    dirs,
    files,
    mtimes,
    writeFile(filePath, data) {
      files.set(filePath, new Uint8Array(data));
    },
    utime(filePath, _atime, mtime) {
      mtimes.set(filePath, mtime);
    },
  };

  if (options.useMkdirTree) {
    adapter.mkdirTree = (dir) => {
      const parts = dir.split('/').filter(Boolean);
      let current = '';
      for (const part of parts) {
        current += `/${part}`;
        dirs.add(current);
      }
    };
  } else {
    adapter.mkdir = (dir) => {
      if (dirs.has(dir)) {
        const err = new Error(`EEXIST: ${dir}`);
        err.errno = 20;
        throw err;
      }
      dirs.add(dir);
    };
  }

  return adapter;
}

// --- Exports & constants ---

test('userdata module exports expected API and UMD globalThis binding', () => {
  assert.equal(DEFAULT_DB_NAME, 'playgta6-userdata');
  assert.equal(DEFAULT_STORE_NAME, 'files');
  assert.equal(DEFAULT_RESTORE_TIMEOUT_MS, 8000);
  assert.equal(USERDATA_PREFIX, '/userdata/');
  assert.equal(ARCHIVE_FORMAT, 'playgta6-userdata-archive-v1');
  assert.equal(ARCHIVE_VERSION, 1);
  assert.equal(typeof validateUserdataPath, 'function');
  assert.equal(typeof validateUserdataPrefix, 'function');
  assert.equal(typeof encodeBase64, 'function');
  assert.equal(typeof decodeBase64, 'function');
  assert.equal(globalThis.PlayGta6Userdata.UserdataStore, UserdataStore);
  assert.equal(globalThis.PlayGta6Userdata.validateUserdataPrefix, validateUserdataPrefix);
  assert.equal(globalThis.PlayGta6Userdata.ARCHIVE_FORMAT, ARCHIVE_FORMAT);
});

// --- Path & prefix validation ---

test('validateUserdataPath enforces /userdata/ confinement and rejects traversal or invalid characters', () => {
  const valid = [
    '/userdata/slot0.sav',
    '/userdata/saves/slot1.sav',
    '/userdata/xgs/profile0/containerA/blob_01.bin',
    '/userdata/.hidden-file',
    '/userdata/dir..name/file.dat',
  ];
  for (const p of valid) {
    assert.equal(validateUserdataPath(p), p);
  }

  const invalid = [
    '',
    null,
    undefined,
    123,
    '/etc/passwd',
    '/userdata',
    '/userdata/',
    '/userdata2/slot0.sav',
    'userdata/slot0.sav',
    '/userdata/../escape.sav',
    '/userdata/saves/../../escape.sav',
    '/userdata/..',
    '/userdata/.',
    '/userdata/./slot0.sav',
    '/userdata/saves//slot0.sav',
    '/userdata\\slot0.sav',
    '/userdata/saves\\slot0.sav',
    '/userdata/slot0.sav/',
    '/userdata/saves/slot0.sav/',
    '/userdata/slot0\0.sav',
  ];
  for (const bad of invalid) {
    assert.throws(
      () => validateUserdataPath(bad),
      `Expected validateUserdataPath to reject ${JSON.stringify(bad)}`,
    );
  }
});

test('validateUserdataPrefix normalizes trailing slash and rejects traversal or invalid segments', () => {
  assert.equal(validateUserdataPrefix('/userdata/'), '/userdata/');
  assert.equal(validateUserdataPrefix('/userdata/saves'), '/userdata/saves/');
  assert.equal(validateUserdataPrefix('/userdata/saves/'), '/userdata/saves/');
  assert.equal(
    validateUserdataPrefix('/userdata/xgs/profile0/slot1'),
    '/userdata/xgs/profile0/slot1/',
  );
  assert.equal(
    validateUserdataPrefix('/userdata/xgs/profile0/slot1/'),
    '/userdata/xgs/profile0/slot1/',
  );

  const invalid = [
    '',
    null,
    undefined,
    42,
    '/userdata',
    '/etc/',
    'userdata/saves/',
    '/userdata2/saves/',
    '/userdata//',
    '/userdata/saves//',
    '/userdata/a//b',
    '/userdata/a//b/',
    '/userdata/..',
    '/userdata/../etc',
    '/userdata/../etc/',
    '/userdata/saves/..',
    '/userdata/saves/../other/',
    '/userdata/.',
    '/userdata/./saves',
    '/userdata/saves/./sub',
    '/userdata/saves\\sub',
    '/userdata\\saves',
    '/userdata/saves\0',
    '/userdata/saves\0/sub',
  ];
  for (const bad of invalid) {
    assert.throws(
      () => validateUserdataPrefix(bad),
      `Expected validateUserdataPrefix to reject ${JSON.stringify(bad)}`,
    );
  }
});

// --- Deterministic Base64 codec ---

test('encodeBase64 and decodeBase64 round-trip arbitrary binary bytes and reject malformed input', () => {
  assert.equal(encodeBase64(new Uint8Array(0)), '');
  assert.deepEqual(decodeBase64(''), new Uint8Array(0));

  // Known RFC 4648 vectors (1, 2, 3 byte lengths).
  assert.equal(encodeBase64(new TextEncoder().encode('f')), 'Zg==');
  assert.equal(encodeBase64(new TextEncoder().encode('fo')), 'Zm8=');
  assert.equal(encodeBase64(new TextEncoder().encode('foo')), 'Zm9v');
  assert.deepEqual(decodeBase64('Zg=='), new TextEncoder().encode('f'));
  assert.deepEqual(decodeBase64('Zm8='), new TextEncoder().encode('fo'));
  assert.deepEqual(decodeBase64('Zm9v'), new TextEncoder().encode('foo'));

  // All 256 byte values + large buffer (16 KiB) to verify no stack overflow.
  const allBytes = new Uint8Array(256);
  for (let i = 0; i < 256; i += 1) {
    allBytes[i] = i;
  }
  const encodedAll = encodeBase64(allBytes);
  assert.deepEqual(decodeBase64(encodedAll), allBytes);

  const large = new Uint8Array(16384);
  for (let i = 0; i < large.length; i += 1) {
    large[i] = (i * 73 + 19) & 0xff;
  }
  assert.deepEqual(decodeBase64(encodeBase64(large)), large);

  // Strict rejection of malformed base64 strings.
  assert.throws(() => decodeBase64(null), TypeError);
  assert.throws(() => decodeBase64('abc'));
  assert.throws(() => decodeBase64('ab$d'));
  assert.throws(() => decodeBase64('===='));
  assert.throws(() => decodeBase64('AB=C'));
  assert.throws(() => decodeBase64('Zg==Zg=='));
});

// --- CRUD on custom backend ---

test('UserdataStore CRUD (saveFile/getFile/deleteFile/listFiles/restoreAll) works with custom backend', async () => {
  const backend = createMemoryBackend();
  const store = new UserdataStore({ backend });

  const saveBytes = new Uint8Array([0xde, 0xad, 0xbe, 0xef, 0x00, 0x7f]);
  const saved = await store.saveFile('/userdata/saves/slot0.sav', saveBytes, 1700000100000);
  assert.deepEqual(saved, {
    path: '/userdata/saves/slot0.sav',
    size: 6,
    mtime: 1700000100000,
  });

  await store.saveFile('/userdata/config.json', '{"lang":"en"}', 1700000200000);

  const fetched = await store.getFile('/userdata/saves/slot0.sav');
  assert.equal(fetched.path, '/userdata/saves/slot0.sav');
  assert.deepEqual(fetched.data, saveBytes);
  assert.equal(fetched.mtime, 1700000100000);

  // Missing file returns null.
  assert.equal(await store.getFile('/userdata/saves/missing.sav'), null);

  // Invalid path rejects on all CRUD methods.
  await assert.rejects(() => store.getFile('/userdata/../etc/passwd'));
  await assert.rejects(() => store.saveFile('/userdata/../etc/passwd', new Uint8Array([1])));
  await assert.rejects(() => store.deleteFile('/userdata/../etc/passwd'));

  const listed = await store.listFiles();
  assert.equal(listed.length, 2);

  // Restore to FS adapter using mkdirTree.
  const fsAdapter = createMockFsAdapter({ useMkdirTree: true });
  const restored = await store.restoreAll(fsAdapter);
  assert.deepEqual(restored, { restored: 2, skipped: 0, timedOut: false, error: null });
  assert.deepEqual(fsAdapter.files.get('/userdata/saves/slot0.sav'), saveBytes);
  assert.equal(fsAdapter.dirs.has('/userdata/saves'), true);
  assert.equal(fsAdapter.mtimes.get('/userdata/saves/slot0.sav'), 1700000100000);

  // Delete and verify.
  const delRes = await store.deleteFile('/userdata/saves/slot0.sav');
  assert.deepEqual(delRes, { path: '/userdata/saves/slot0.sav', deleted: true });
  assert.equal(await store.getFile('/userdata/saves/slot0.sav'), null);
  assert.equal((await store.listFiles()).length, 1);

  store.close();
  assert.equal(backend.isClosed(), true);
});

test('UserdataStore getFile falls back cleanly when custom backend omits get()', async () => {
  const backend = createMemoryBackend({ includeGet: false });
  const store = new UserdataStore({ backend });

  await store.saveFile('/userdata/profile/data.bin', new Uint8Array([1, 2, 3]), 12345);
  const rec = await store.getFile('/userdata/profile/data.bin');
  assert.equal(rec.path, '/userdata/profile/data.bin');
  assert.deepEqual(rec.data, new Uint8Array([1, 2, 3]));
  assert.equal(rec.mtime, 12345);
  assert.equal(await store.getFile('/userdata/profile/absent.bin'), null);
});

// --- CRUD & resilience on mock IndexedDB ---

test('UserdataStore triggers onupgradeneeded, persists across instances, and supports mock IndexedDB CRUD', async () => {
  const mockIdb = createMockIndexedDB();
  const store1 = new UserdataStore({ indexedDB: mockIdb });

  const blobA = new Uint8Array([10, 20, 30, 40]);
  const blobB = new TextEncoder().encode('settings-v1');

  await store1.saveFile('/userdata/xgs/c1/blobA.bin', blobA, 1700001000000);
  await store1.saveFile('/userdata/xgs/c1/blobB.txt', blobB, 1700002000000);

  // Verify onupgradeneeded ran once and created the 'files' store with keyPath 'path'.
  assert.equal(mockIdb.upgradeEvents.length, 1);
  assert.deepEqual(mockIdb.upgradeEvents[0], { dbName: DEFAULT_DB_NAME, version: 1 });
  assert.equal(mockIdb.databases.get(DEFAULT_DB_NAME).keyPaths.get(DEFAULT_STORE_NAME), 'path');

  const fetched = await store1.getFile('/userdata/xgs/c1/blobA.bin');
  assert.deepEqual(fetched.data, blobA);
  assert.equal(fetched.mtime, 1700001000000);
  store1.close();

  // Reopen with a second UserdataStore instance; onupgradeneeded should not fire again.
  const store2 = new UserdataStore({ indexedDB: mockIdb });
  const fetched2 = await store2.getFile('/userdata/xgs/c1/blobB.txt');
  assert.deepEqual(fetched2.data, blobB);
  assert.equal(mockIdb.upgradeEvents.length, 1);

  await store2.deleteFile('/userdata/xgs/c1/blobA.bin');
  assert.equal(await store2.getFile('/userdata/xgs/c1/blobA.bin'), null);

  // Inject an invalid record directly in mock DB to verify restoreAll skips it safely.
  mockIdb.databases
    .get(DEFAULT_DB_NAME)
    .stores.get(DEFAULT_STORE_NAME)
    .set('/etc/shadow', { path: '/etc/shadow', data: new Uint8Array([1]), mtime: 1 });

  const fsAdapter = createMockFsAdapter();
  const restoreRes = await store2.restoreAll(fsAdapter, { timeoutMs: 2000 });
  assert.equal(restoreRes.timedOut, false);
  assert.equal(restoreRes.error, null);
  assert.equal(restoreRes.restored, 1);
  assert.equal(restoreRes.skipped, 1);
  assert.deepEqual(fsAdapter.files.get('/userdata/xgs/c1/blobB.txt'), blobB);
  assert.equal(fsAdapter.files.has('/etc/shadow'), false);
  store2.close();
});

test('UserdataStore getFile works with mock IndexedDB omitting store.get()', async () => {
  const mockIdb = createMockIndexedDB({ omitStoreGet: true });
  const store = new UserdataStore({ indexedDB: mockIdb });

  await store.saveFile('/userdata/legacy/blob.bin', new Uint8Array([7, 8, 9]), 999);
  const rec = await store.getFile('/userdata/legacy/blob.bin');
  assert.deepEqual(rec.data, new Uint8Array([7, 8, 9]));
  assert.equal(rec.mtime, 999);
  store.close();
});

test('UserdataStore restoreAll handles IndexedDB hang timeout, open failure, and unavailability', async () => {
  const hangOpenStore = new UserdataStore({ indexedDB: createMockIndexedDB({ hangOpen: true }) });
  const resOpen = await hangOpenStore.restoreAll(createMockFsAdapter(), { timeoutMs: 40 });
  assert.equal(resOpen.timedOut, true);
  assert.equal(resOpen.restored, 0);
  assert.match(resOpen.error, /timed out/i);

  const hangGetAllStore = new UserdataStore({ indexedDB: createMockIndexedDB({ hangGetAll: true }) });
  const resGetAll = await hangGetAllStore.restoreAll(createMockFsAdapter(), { timeoutMs: 40 });
  assert.equal(resGetAll.timedOut, true);
  assert.equal(resGetAll.restored, 0);
  assert.match(resGetAll.error, /timed out/i);
  hangGetAllStore.close();

  const failOpenStore = new UserdataStore({ indexedDB: createMockIndexedDB({ failOpen: true }) });
  const resFail = await failOpenStore.restoreAll(createMockFsAdapter(), { timeoutMs: 500 });
  assert.equal(resFail.timedOut, false);
  assert.equal(resFail.restored, 0);
  assert.match(resFail.error, /Simulated IndexedDB open failure/);

  const noIdbStore = new UserdataStore({ indexedDB: null });
  const resUnavailable = await noIdbStore.restoreAll(createMockFsAdapter(), { timeoutMs: 100 });
  assert.equal(resUnavailable.timedOut, false);
  assert.equal(resUnavailable.restored, 0);
  assert.match(resUnavailable.error, /unavailable/i);
});

// --- Container prefix operations & usage stats ---

test('listByPrefix returns deterministically sorted paths and avoids sibling prefix collisions', async () => {
  const store = new UserdataStore({ backend: createMemoryBackend() });
  await store.saveFile('/userdata/saves/slot2.sav', new Uint8Array([2]));
  await store.saveFile('/userdata/saves/slot0.sav', new Uint8Array([0]));
  await store.saveFile('/userdata/saves/slot1/meta.bin', new Uint8Array([1]));
  await store.saveFile('/userdata/save/other.sav', new Uint8Array([9]));
  await store.saveFile('/userdata/config.json', new Uint8Array([3]));

  // Trailing slash normalization prevents /userdata/save from matching /userdata/saves/...
  assert.deepEqual(await store.listByPrefix('/userdata/save'), ['/userdata/save/other.sav']);
  assert.deepEqual(await store.listByPrefix('/userdata/save/'), ['/userdata/save/other.sav']);

  assert.deepEqual(await store.listByPrefix('/userdata/saves'), [
    '/userdata/saves/slot0.sav',
    '/userdata/saves/slot1/meta.bin',
    '/userdata/saves/slot2.sav',
  ]);
  assert.deepEqual(await store.listByPrefix('/userdata/saves/slot1'), [
    '/userdata/saves/slot1/meta.bin',
  ]);
  assert.deepEqual(await store.listByPrefix('/userdata/'), [
    '/userdata/config.json',
    '/userdata/save/other.sav',
    '/userdata/saves/slot0.sav',
    '/userdata/saves/slot1/meta.bin',
    '/userdata/saves/slot2.sav',
  ]);
  assert.deepEqual(await store.listByPrefix('/userdata/nonexistent'), []);

  await assert.rejects(() => store.listByPrefix('/userdata/../etc'));
  await assert.rejects(() => store.listByPrefix('/etc/'));
  await assert.rejects(() => store.listByPrefix('/userdata'));
});

test('deletePrefix deletes an entire container cleanly and returns exact deleted count', async () => {
  const store = new UserdataStore({ indexedDB: createMockIndexedDB() });
  await store.saveFile('/userdata/xgs/slot0/a.bin', new Uint8Array([1, 2]));
  await store.saveFile('/userdata/xgs/slot0/b.bin', new Uint8Array([3, 4]));
  await store.saveFile('/userdata/xgs/slot1/a.bin', new Uint8Array([5, 6]));
  await store.saveFile('/userdata/xgs_other/c.bin', new Uint8Array([7]));
  await store.saveFile('/userdata/settings.json', new Uint8Array([8]));

  // Deleting /userdata/xgs/slot0 removes only slot0 blobs.
  const removedSlot0 = await store.deletePrefix('/userdata/xgs/slot0');
  assert.equal(removedSlot0, 2);
  assert.deepEqual(await store.listByPrefix('/userdata/xgs/slot0'), []);
  assert.deepEqual(await store.listByPrefix('/userdata/xgs/'), ['/userdata/xgs/slot1/a.bin']);
  assert.deepEqual(await store.listByPrefix('/userdata/xgs_other'), ['/userdata/xgs_other/c.bin']);

  // Deleting an empty/nonexistent container returns 0.
  assert.equal(await store.deletePrefix('/userdata/xgs/slot0/'), 0);

  // Deleting /userdata/ clears everything.
  const removedAll = await store.deletePrefix('/userdata/');
  assert.equal(removedAll, 3);
  assert.deepEqual(await store.listByPrefix('/userdata/'), []);

  // Rejects invalid/traversal prefixes.
  await assert.rejects(() => store.deletePrefix('/userdata/../'));
  await assert.rejects(() => store.deletePrefix('/userdata/saves\\sub'));
  store.close();
});

test('getUsageStats reports fileCount, totalBytes, and per-top-level-container breakdown without host paths', async () => {
  const store = new UserdataStore({ backend: createMemoryBackend() });
  assert.deepEqual(await store.getUsageStats(), {
    fileCount: 0,
    totalBytes: 0,
    prefixes: {},
  });

  await store.saveFile('/userdata/config.json', new Uint8Array(12));
  await store.saveFile('/userdata/saves/slot0.sav', new Uint8Array(100));
  await store.saveFile('/userdata/saves/nested/blob.bin', new Uint8Array(50));
  await store.saveFile('/userdata/profiles/p0/prefs.bin', new Uint8Array(25));

  const stats = await store.getUsageStats();
  assert.equal(stats.fileCount, 4);
  assert.equal(stats.totalBytes, 187);
  assert.deepEqual(Object.keys(stats.prefixes), [
    '/userdata/',
    '/userdata/profiles/',
    '/userdata/saves/',
  ]);
  assert.deepEqual(stats.prefixes, {
    '/userdata/': { fileCount: 1, totalBytes: 12 },
    '/userdata/profiles/': { fileCount: 1, totalBytes: 25 },
    '/userdata/saves/': { fileCount: 2, totalBytes: 150 },
  });
});

// --- Portable archive export / import ---

test('exportArchive and importArchive round-trip in merge mode and replace mode across stores', async () => {
  const sourceStore = new UserdataStore({ indexedDB: createMockIndexedDB() });
  const binarySave = new Uint8Array([0x00, 0x01, 0xfe, 0xff, 0x42, 0x99]);
  const emptyBlob = new Uint8Array(0);
  const jsonProfile = new TextEncoder().encode('{"slot":1,"difficulty":"normal"}');

  await sourceStore.saveFile('/userdata/saves/slot1.sav', binarySave, 1700000001000);
  await sourceStore.saveFile('/userdata/saves/empty.bin', emptyBlob, 1700000002000);
  await sourceStore.saveFile('/userdata/profile.json', jsonProfile, 1700000003000);

  const archive = await sourceStore.exportArchive();
  assert.equal(archive.format, 'playgta6-userdata-archive-v1');
  assert.equal(archive.version, 1);
  assert.equal(typeof archive.exportedAt, 'string');
  assert.ok(!Number.isNaN(Date.parse(archive.exportedAt)));
  assert.deepEqual(
    archive.files.map((f) => f.path),
    ['/userdata/profile.json', '/userdata/saves/empty.bin', '/userdata/saves/slot1.sav'],
  );
  assert.equal(archive.files[2].size, binarySave.byteLength);
  assert.equal(typeof archive.files[2].base64, 'string');

  // Verify JSON serialization round-trip.
  const serialized = JSON.stringify(archive);

  // 1. Merge mode into a store with pre-existing files.
  const targetStore = new UserdataStore({ backend: createMemoryBackend() });
  await targetStore.saveFile('/userdata/existing.txt', new TextEncoder().encode('keep-me'), 111);
  await targetStore.saveFile('/userdata/profile.json', new TextEncoder().encode('old-profile'), 222);

  const mergeResult = await targetStore.importArchive(serialized);
  assert.deepEqual(mergeResult, {
    importedCount: 3,
    totalBytes: binarySave.byteLength + emptyBlob.byteLength + jsonProfile.byteLength,
  });

  // Unrelated pre-existing file stays intact; colliding path is updated.
  assert.deepEqual(
    (await targetStore.getFile('/userdata/existing.txt')).data,
    new TextEncoder().encode('keep-me'),
  );
  assert.deepEqual((await targetStore.getFile('/userdata/profile.json')).data, jsonProfile);
  assert.deepEqual((await targetStore.getFile('/userdata/saves/slot1.sav')).data, binarySave);
  assert.deepEqual((await targetStore.getFile('/userdata/saves/empty.bin')).data, emptyBlob);
  assert.equal((await targetStore.getFile('/userdata/saves/slot1.sav')).mtime, 1700000001000);

  // 2. Replace mode clears existing files before writing imported ones.
  const replaceResult = await targetStore.importArchive(archive, { replace: true });
  assert.deepEqual(replaceResult, {
    importedCount: 3,
    totalBytes: binarySave.byteLength + emptyBlob.byteLength + jsonProfile.byteLength,
  });
  assert.equal(await targetStore.getFile('/userdata/existing.txt'), null);
  assert.deepEqual(await targetStore.listByPrefix('/userdata/'), [
    '/userdata/profile.json',
    '/userdata/saves/empty.bin',
    '/userdata/saves/slot1.sav',
  ]);

  // 3. Prefix-filtered export works cleanly.
  const savesOnly = await sourceStore.exportArchive({ prefix: '/userdata/saves' });
  assert.deepEqual(
    savesOnly.files.map((f) => f.path),
    ['/userdata/saves/empty.bin', '/userdata/saves/slot1.sav'],
  );

  // 4. Empty archive with replace: true clears the store.
  const emptyImport = await targetStore.importArchive(
    { format: ARCHIVE_FORMAT, version: ARCHIVE_VERSION, exportedAt: new Date().toISOString(), files: [] },
    { replace: true },
  );
  assert.deepEqual(emptyImport, { importedCount: 0, totalBytes: 0 });
  assert.deepEqual(await targetStore.listByPrefix('/userdata/'), []);

  sourceStore.close();
});

test('importArchive atomically rejects invalid format, version, traversal paths, and corrupted entries before mutating state', async () => {
  const store = new UserdataStore({ backend: createMemoryBackend() });
  const original = new TextEncoder().encode('untouched-save');
  await store.saveFile('/userdata/original.sav', original, 5000);

  const badArchives = [
    null,
    [],
    { format: 'wrong-format', version: 1, files: [] },
    { format: ARCHIVE_FORMAT, version: 2, files: [] },
    { format: ARCHIVE_FORMAT, version: 1, files: 'not-an-array' },
    // Valid first entry followed by path-traversal second entry.
    {
      format: ARCHIVE_FORMAT,
      version: 1,
      files: [
        { path: '/userdata/new.sav', size: 1, base64: encodeBase64(new Uint8Array([1])) },
        { path: '/userdata/../escape.sav', size: 1, base64: encodeBase64(new Uint8Array([2])) },
      ],
    },
    // Outside /userdata/.
    {
      format: ARCHIVE_FORMAT,
      version: 1,
      files: [{ path: '/etc/passwd', size: 1, base64: encodeBase64(new Uint8Array([1])) }],
    },
    // Backslash & NUL byte.
    {
      format: ARCHIVE_FORMAT,
      version: 1,
      files: [{ path: '/userdata/sub\\evil.sav', size: 1, base64: encodeBase64(new Uint8Array([1])) }],
    },
    {
      format: ARCHIVE_FORMAT,
      version: 1,
      files: [{ path: '/userdata/evil\0.sav', size: 1, base64: encodeBase64(new Uint8Array([1])) }],
    },
    // Duplicate path.
    {
      format: ARCHIVE_FORMAT,
      version: 1,
      files: [
        { path: '/userdata/dup.sav', size: 1, base64: encodeBase64(new Uint8Array([1])) },
        { path: '/userdata/dup.sav', size: 1, base64: encodeBase64(new Uint8Array([2])) },
      ],
    },
    // Malformed base64.
    {
      format: ARCHIVE_FORMAT,
      version: 1,
      files: [{ path: '/userdata/bad64.sav', base64: '!!!invalid!!!' }],
    },
    // Size mismatch.
    {
      format: ARCHIVE_FORMAT,
      version: 1,
      files: [{ path: '/userdata/mismatch.sav', size: 99, base64: encodeBase64(new Uint8Array([1, 2])) }],
    },
  ];

  for (const bad of badArchives) {
    await assert.rejects(() => store.importArchive(bad, { replace: false }));
    await assert.rejects(() => store.importArchive(bad, { replace: true }));
  }

  // State must remain completely unchanged after every rejected import!
  assert.deepEqual(await store.listByPrefix('/userdata/'), ['/userdata/original.sav']);
  assert.deepEqual((await store.getFile('/userdata/original.sav')).data, original);
});
