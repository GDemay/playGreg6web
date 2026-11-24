import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

import {
  HudController,
  HUD_ACTIONS,
  DEFAULT_FILENAMES,
  BOOT_MODE,
  DEFAULT_MAX_VALIDATION_ERRORS,
  MAX_VALIDATION_ERROR_MESSAGE_LENGTH,
  normalizeBootMode,
  sanitizeValidationErrorMessage,
  parseSideModuleProgressLabel,
} from '../web/ui/hud_controller.js';
import { bootstrapPlayGta6App } from '../web/app_bootstrap.js';
import {
  BrowserStorageManager,
  DEFAULT_OPFS_CACHE_DIR,
  SNAPSHOT_FORMAT,
  SNAPSHOT_VERSION,
} from '../web/storage/storage_manager.js';
import { CrashReporter } from '../web/telemetry/crash_report.js';
import { PerfMonitor } from '../web/telemetry/perf_monitor.js';
import { InputProfileManager } from '../web/input/input_profiles.js';
import {
  RuntimeSessionController,
  SESSION_STATE,
  SESSION_MODE,
} from '../web/runtime/session_controller.js';

const require = createRequire(import.meta.url);
const { GpuFrameRecorder, OPCODES } = require('../web/workers/gpu_recorder.js');
const { GpuBootPanelController, EXPORT_FORMAT } = require('../web/ui/gpu_boot_panel.js');

// --- Lightweight Mocks for Browser APIs (DOM, OPFS, IndexedDB, localStorage, fetch) ---

function createMockHud() {
  const logs = [];
  const statuses = [];
  const metrics = [];
  return {
    logs,
    statuses,
    metrics,
    appendLog(message, level = 'info') {
      logs.push({ message, level });
    },
    setStatus(text, level = 'info') {
      statuses.push({ text, level });
    },
    updateMetrics(payload) {
      metrics.push({ ...payload });
    },
  };
}

function createMockOpfsDirHandle(name = '', initialEntries = null) {
  const entries = new Map();
  if (initialEntries) {
    for (const [k, v] of Object.entries(initialEntries)) {
      entries.set(k, v instanceof Uint8Array ? v.slice() : v);
    }
  }

  return {
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
}

function createMockIndexedDB() {
  const databases = new Map();

  function makeStore(recordsMap) {
    return {
      put(record) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          recordsMap.set(record.path, {
            ...record,
            data: record.data instanceof Uint8Array ? record.data.slice() : record.data,
          });
          req.result = record.path;
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      delete(key) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          recordsMap.delete(key);
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      getAll() {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          req.result = Array.from(recordsMap.values()).map((r) => ({
            ...r,
            data: r.data instanceof Uint8Array ? r.data.slice() : r.data,
          }));
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
      clear() {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          recordsMap.clear();
          if (req.onsuccess) req.onsuccess({ target: req });
        });
        return req;
      },
    };
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
      queueMicrotask(() => {
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
          },
          transaction(storeName) {
            if (!entry.stores.has(storeName)) {
              entry.stores.set(storeName, new Map());
            }
            const store = makeStore(entry.stores.get(storeName));
            return {
              objectStore: () => store,
            };
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

function createMockLocalStorage() {
  const map = new Map();
  return {
    _map: map,
    getItem(key) {
      return map.has(key) ? map.get(key) : null;
    },
    setItem(key, value) {
      map.set(key, String(value));
    },
    removeItem(key) {
      map.delete(key);
    },
  };
}

function createMockElement(initialValue = '') {
  const listeners = new Map();
  const attributes = new Map();
  return {
    value: initialValue,
    dataset: {},
    _listeners: listeners,
    _attributes: attributes,
    addEventListener(type, fn) {
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type).add(fn);
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (set) {
        set.delete(fn);
      }
    },
    setAttribute(name, val) {
      attributes.set(name, String(val));
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
    listenerCount(type) {
      const set = listeners.get(type);
      return set ? set.size : 0;
    },
    dispatch(type, extra = {}) {
      const set = listeners.get(type);
      if (!set) return;
      const event = { type, target: this, ...extra };
      for (const fn of [...set]) {
        fn(event);
      }
    },
    click() {
      this.dispatch('click');
    },
  };
}

function buildPacket(opcode, payloadBytes = new Uint8Array(0), flags = 0) {
  const out = new Uint8Array(8 + payloadBytes.byteLength);
  const view = new DataView(out.buffer);
  view.setUint16(0, opcode, true);
  view.setUint16(2, flags, true);
  view.setUint32(4, payloadBytes.byteLength, true);
  out.set(payloadBytes, 8);
  return out;
}

function buildSampleGpuFrame() {
  const begin = buildPacket(OPCODES.BEGIN_FRAME);
  const clearPayload = new Uint8Array(16);
  const clearView = new DataView(clearPayload.buffer);
  clearView.setFloat32(0, 0.1, true);
  clearView.setFloat32(4, 0.2, true);
  clearView.setFloat32(8, 0.3, true);
  clearView.setFloat32(12, 1.0, true);
  const clear = buildPacket(OPCODES.CLEAR_COLOR, clearPayload);
  const present = buildPacket(OPCODES.PRESENT);

  const frame = new Uint8Array(begin.byteLength + clear.byteLength + present.byteLength);
  frame.set(begin, 0);
  frame.set(clear, begin.byteLength);
  frame.set(present, begin.byteLength + clear.byteLength);
  return frame;
}

function createFullCapabilities() {
  return {
    crossOriginIsolated: true,
    sharedArrayBuffer: true,
    webgpu: true,
    wasm64: true,
    opfs: true,
    gamepad: true,
    audioWorklet: true,
  };
}

function createMockFetch(statusPayload = {}) {
  return async () => ({
    ok: true,
    status: 200,
    async json() {
      return {
        ok: true,
        coopCoep: true,
        dataDirConfigured: true,
        hasManifest: true,
        hasBootset: true,
        hasWasmBinary: false,
        shaderPackCount: 2,
        ...statusPayload,
      };
    },
  });
}

async function flushAsync(ticks = 6) {
  for (let i = 0; i < ticks; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

// --- Tests ---

test('clearOpfsCache clears OPFS entries via real BrowserStorageManager and logs freed bytes', async () => {
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, {
    'chunk_000.bin': new Uint8Array(2048),
    'chunk_001.bin': new Uint8Array(1024),
  });
  const opfsRoot = createMockOpfsDirHandle('', {
    [DEFAULT_OPFS_CACHE_DIR]: cacheDir,
  });

  const storageManager = new BrowserStorageManager({
    storageObj: {
      async estimate() {
        return { usage: 3072, quota: 1024 * 1024 };
      },
      async persisted() {
        return true;
      },
      async getDirectory() {
        return opfsRoot;
      },
    },
    opfsRoot,
    indexedDBFactory: createMockIndexedDB(),
  });

  const hud = createMockHud();
  const controller = new HudController({ hud, storageManager });

  const before = await storageManager.inspectOpfsCache();
  assert.equal(before.entryCount, 2);
  assert.equal(before.totalBytes, 3072);

  const res = await controller.clearOpfsCache();
  assert.equal(res.ok, true);
  assert.equal(res.action, HUD_ACTIONS.CLEAR_OPFS_CACHE);
  assert.equal(res.removedCount, 2);
  assert.equal(res.freedBytes, 3072);
  assert.match(res.summary, /3 KiB/);
  assert.equal(hud.logs.length, 1);
  assert.equal(hud.logs[0].level, 'ok');
  assert.match(hud.logs[0].message, /3072 B/);

  const after = await storageManager.inspectOpfsCache();
  assert.equal(after.entryCount, 0);
  assert.equal(after.totalBytes, 0);

  const state = controller.getState();
  assert.equal(state.lastAction.action, HUD_ACTIONS.CLEAR_OPFS_CACHE);
  assert.equal(state.lastAction.ok, true);
});

test('exportSaves and importSaves round-trip userdata backup through real BrowserStorageManager', async () => {
  const mockIdb = createMockIndexedDB();
  const storageManager = new BrowserStorageManager({
    storageObj: null,
    opfsRoot: null,
    indexedDBFactory: mockIdb,
    now: () => 1700000000000,
  });

  // Seed two /userdata/ save files using importUserdataSnapshot.
  const payloadA = new Uint8Array([1, 2, 3, 4, 5, 6]);
  const payloadB = new TextEncoder().encode('{"slot":1,"progress":42}');
  await storageManager.importUserdataSnapshot({
    format: SNAPSHOT_FORMAT,
    version: SNAPSHOT_VERSION,
    files: [
      {
        path: '/userdata/saves/slot0.sav',
        encoding: 'base64',
        dataBase64: Buffer.from(payloadA).toString('base64'),
        byteLength: payloadA.byteLength,
        updatedAt: 1700000000000,
      },
      {
        path: '/userdata/profile.json',
        encoding: 'base64',
        dataBase64: Buffer.from(payloadB).toString('base64'),
        byteLength: payloadB.byteLength,
        updatedAt: 1700000000000,
      },
    ],
  });

  const downloads = [];
  const hud = createMockHud();
  const controller = new HudController({
    hud,
    storageManager,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });

  // 1. Export saves and confirm JSON backup download.
  const exp = await controller.exportSaves();
  assert.equal(exp.ok, true);
  assert.equal(exp.action, HUD_ACTIONS.EXPORT_SAVES);
  assert.equal(exp.downloaded, true);
  assert.equal(exp.fileCount, 2);
  assert.equal(exp.totalBytes, payloadA.byteLength + payloadB.byteLength);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, DEFAULT_FILENAMES.USERDATA_BACKUP);
  assert.equal(downloads[0].mimeType, 'application/json');
  assert.equal(downloads[0].data, exp.json);

  // 2. Wipe IndexedDB and re-import from the exported JSON string.
  const cleared = await storageManager.clearUserdata();
  assert.equal(cleared.clearedCount, 2);
  assert.equal((await storageManager.listUserdataFiles()).fileCount, 0);

  const imp = await controller.importSaves(exp.json, { overwrite: true });
  assert.equal(imp.ok, true);
  assert.equal(imp.action, HUD_ACTIONS.IMPORT_SAVES);
  assert.equal(imp.importedCount, 2);
  assert.equal(imp.skippedCount, 0);
  assert.equal(imp.importedBytes, payloadA.byteLength + payloadB.byteLength);

  // 3. Re-importing with overwrite: false skips existing files.
  const impSkip = await controller.importSaves(exp.json, { overwrite: false });
  assert.equal(impSkip.ok, true);
  assert.equal(impSkip.importedCount, 0);
  assert.equal(impSkip.skippedCount, 2);

  // 4. Verify restored snapshot content matches original export.
  const reExport = await storageManager.exportUserdataSnapshot();
  assert.deepEqual(reExport.files, exp.snapshot.files);

  // 5. Malformed JSON & path traversal fail gracefully without throwing.
  const badJson = await controller.importSaves('{not valid json');
  assert.equal(badJson.ok, false);
  assert.equal(badJson.action, HUD_ACTIONS.IMPORT_SAVES);
  assert.ok(badJson.error);

  const traversalJson = JSON.stringify({
    format: SNAPSHOT_FORMAT,
    version: SNAPSHOT_VERSION,
    files: [
      {
        path: '/userdata/../etc/passwd',
        encoding: 'base64',
        dataBase64: 'AAAA',
      },
    ],
  });
  const badPath = await controller.importSaves(traversalJson);
  assert.equal(badPath.ok, false);
  assert.match(badPath.error, /Invalid userdata path segment/i);
});

test('toggleGpuRecording and exportGpuTrace record and download .pg6trace via real GpuFrameRecorder', async () => {
  const gpuRecorder = new GpuFrameRecorder({ maxFrames: 8, autoStart: false });
  const downloads = [];
  const hud = createMockHud();

  const controller = new HudController({
    hud,
    gpuRecorder,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });

  assert.equal(controller.getState().isRecording, false);

  // Exporting before any frames are recorded returns ok: false without downloading empty bytes.
  const emptyExp = await controller.exportGpuTrace();
  assert.equal(emptyExp.ok, false);
  assert.equal(emptyExp.byteLength, 0);
  assert.equal(downloads.length, 0);

  // Toggle ON
  const startRes = await controller.toggleGpuRecording();
  assert.equal(startRes.ok, true);
  assert.equal(startRes.action, HUD_ACTIONS.TOGGLE_GPU_RECORDING);
  assert.equal(startRes.isRecording, true);
  assert.equal(gpuRecorder.isRecording(), true);
  assert.equal(controller.getState().isRecording, true);

  // Record 2 complete frames
  const frameBytes = buildSampleGpuFrame();
  gpuRecorder.recordChunk(frameBytes);
  gpuRecorder.recordChunk(frameBytes);

  // Toggle OFF
  const stopRes = await controller.toggleGpuRecording();
  assert.equal(stopRes.ok, true);
  assert.equal(stopRes.isRecording, false);
  assert.equal(gpuRecorder.isRecording(), false);
  assert.match(stopRes.summary, /2 frames/);

  // Export .pg6trace binary stream
  const traceRes = await controller.exportGpuTrace();
  assert.equal(traceRes.ok, true);
  assert.equal(traceRes.action, HUD_ACTIONS.EXPORT_GPU_TRACE);
  assert.equal(traceRes.filename, DEFAULT_FILENAMES.GPU_TRACE);
  assert.equal(traceRes.downloaded, true);
  assert.equal(traceRes.frameCount, 2);
  assert.equal(traceRes.byteLength, frameBytes.byteLength * 2);
  assert.ok(traceRes.bytes instanceof Uint8Array);

  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, 'playgta6-capture.pg6trace');
  assert.equal(downloads[0].mimeType, 'application/octet-stream');
  assert.deepEqual(downloads[0].data.subarray(0, frameBytes.byteLength), frameBytes);
});

test('toggleGpuRecording and exportGpuTrace also work via smokeDemoRunner when gpuRecorder is omitted', async () => {
  const innerRecorder = new GpuFrameRecorder({ maxFrames: 4, autoStart: false });
  const frameBytes = buildSampleGpuFrame();
  const smokeDemoRunner = {
    running: false,
    recorder: innerRecorder,
    start() {
      this.running = true;
      return true;
    },
    stop() {
      this.running = false;
      return true;
    },
    startRecording() {
      innerRecorder.start();
      innerRecorder.recordChunk(frameBytes);
    },
    stopRecording() {
      innerRecorder.stop();
    },
    exportTraceBytes(opts) {
      return innerRecorder.exportTraceBytes(opts);
    },
    exportTraceMetadata() {
      return innerRecorder.exportMetadata();
    },
  };

  const downloads = [];
  const controller = new HudController({
    hud: createMockHud(),
    smokeDemoRunner,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });

  const onRes = await controller.toggleGpuRecording();
  assert.equal(onRes.ok, true);
  assert.equal(onRes.isRecording, true);

  const offRes = await controller.toggleGpuRecording();
  assert.equal(offRes.ok, true);
  assert.equal(offRes.isRecording, false);

  const expRes = await controller.exportGpuTrace();
  assert.equal(expRes.ok, true);
  assert.equal(expRes.frameCount, 1);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, 'playgta6-capture.pg6trace');
});

test('exportCrashReport produces sanitized JSON bundle via real CrashReporter, PerfMonitor, StorageManager, and SessionController', async () => {
  const crashReporter = new CrashReporter({
    appVersion: '0.2.0-test',
    now: () => 1700000123000,
  });
  crashReporter.recordError(
    'io_error',
    new Error('Failed reading C:\\Users\\secretuser\\GTA6\\data.rpf with Bearer super-secret-token-999'),
    { path: '/home/secretuser/private/key.bin' }
  );
  crashReporter.recordLog('warn', 'hud', 'Loaded from file:///Users/secretuser/local.wasm?token=abc123');

  const perfMonitor = new PerfMonitor({ now: () => 1000 });
  perfMonitor.recordFrameTimeMs(16.6);
  perfMonitor.recordGpuBatch({ packets: 12, drawCalls: 2, bytes: 512, durationMs: 16.6 });

  const storageManager = new BrowserStorageManager({
    storageObj: {
      async estimate() {
        return { usage: 4096, quota: 1024 * 1024 };
      },
      async persisted() {
        return false;
      },
    },
    opfsRoot: null,
    indexedDBFactory: createMockIndexedDB(),
  });

  const sessionController = new RuntimeSessionController({
    capabilityProbe: async () => createFullCapabilities(),
    fetchImpl: createMockFetch({ hasWasmBinary: false, shaderPackCount: 5 }),
    startSmokeDemo: async () => ({ stop() {} }),
  });
  await sessionController.start({ preferSmoke: true });

  const downloads = [];
  const clipboardWrites = [];
  const hud = createMockHud();

  const controller = new HudController({
    hud,
    crashReporter,
    perfMonitor,
    storageManager,
    sessionController,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
    copyToClipboard(text) {
      clipboardWrites.push(text);
    },
  });

  const res = await controller.exportCrashReport({ copyToClipboard: true });
  assert.equal(res.ok, true);
  assert.equal(res.action, HUD_ACTIONS.EXPORT_CRASH_REPORT);
  assert.equal(res.filename, DEFAULT_FILENAMES.CRASH_REPORT);
  assert.equal(res.downloaded, true);
  assert.equal(res.copied, true);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, 'playgta6-diagnostic-report.json');
  assert.equal(downloads[0].mimeType, 'application/json');
  assert.equal(clipboardWrites.length, 1);

  // Verify sanitization: zero host usernames, paths, or secrets leaked.
  assert.doesNotMatch(res.json, /secretuser/);
  assert.doesNotMatch(res.json, /super-secret-token-999/);
  assert.doesNotMatch(res.json, /token=abc123/);
  assert.match(res.json, /\[REDACTED_PATH\]/);
  assert.match(res.json, /\[REDACTED\]/);

  const parsed = JSON.parse(res.json);
  assert.equal(parsed.schemaVersion, 1);
  assert.equal(parsed.appVersion, '0.2.0-test');
  assert.equal(parsed.errors.length, 1);
  assert.equal(parsed.perf.gpu.totalGpuPackets, 12);
  assert.equal(parsed.storage.quota.usageBytes, 4096);
  assert.equal(parsed.serverStatus.shaderPackCount, 5);

  await sessionController.stop();
});

test('switchInputProfile switches active profile and persists via real InputProfileManager', async () => {
  const mockStorage = createMockLocalStorage();
  const inputProfiles = new InputProfileManager({ storage: mockStorage });
  const hud = createMockHud();
  const controller = new HudController({ hud, inputProfiles });

  assert.equal(controller.getState().activeProfileId, 'default');

  const okRes = await controller.switchInputProfile('fps');
  assert.equal(okRes.ok, true);
  assert.equal(okRes.action, HUD_ACTIONS.SWITCH_INPUT_PROFILE);
  assert.equal(okRes.profileId, 'fps');
  assert.equal(okRes.persisted, true);
  assert.equal(inputProfiles.getActiveProfile().id, 'fps');
  assert.equal(controller.getState().activeProfileId, 'fps');

  // Verify persistence by loading into a fresh InputProfileManager instance.
  const reloaded = new InputProfileManager({ storage: mockStorage, autoLoad: true });
  assert.equal(reloaded.getActiveProfile().id, 'fps');

  // Unknown profile ID fails gracefully without throwing.
  const badRes = await controller.switchInputProfile('nonexistent-profile');
  assert.equal(badRes.ok, false);
  assert.equal(badRes.action, HUD_ACTIONS.SWITCH_INPUT_PROFILE);
  assert.match(badRes.error, /Unknown input profile/i);
  assert.equal(inputProfiles.getActiveProfile().id, 'fps');
});

test('toggleSmokeDemo and restartSession drive real RuntimeSessionController lifecycle', async () => {
  let wasmStarts = 0;
  let wasmStops = 0;
  let smokeStarts = 0;
  let smokeStops = 0;

  const sessionController = new RuntimeSessionController({
    capabilityProbe: async () => createFullCapabilities(),
    fetchImpl: createMockFetch({ hasWasmBinary: true }),
    startWasmRuntime: async () => {
      wasmStarts += 1;
      return {
        stop() {
          wasmStops += 1;
        },
      };
    },
    startSmokeDemo: async () => {
      smokeStarts += 1;
      return {
        stop() {
          smokeStops += 1;
        },
      };
    },
  });

  const hud = createMockHud();
  const controller = new HudController({ hud, sessionController });

  assert.equal(controller.getState().sessionState, SESSION_STATE.IDLE);

  // 1. Toggle smoke demo ON from idle -> running_smoke.
  const startSmoke = await controller.toggleSmokeDemo();
  assert.equal(startSmoke.ok, true);
  assert.equal(startSmoke.action, HUD_ACTIONS.TOGGLE_SMOKE_DEMO);
  assert.equal(startSmoke.running, true);
  assert.equal(sessionController.getState(), SESSION_STATE.RUNNING_SMOKE);
  assert.equal(controller.getState().sessionState, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(smokeStarts, 1);

  // 2. Toggle smoke demo OFF -> stopped.
  const stopSmoke = await controller.toggleSmokeDemo();
  assert.equal(stopSmoke.ok, true);
  assert.equal(stopSmoke.running, false);
  assert.equal(sessionController.getState(), SESSION_STATE.STOPPED);
  assert.equal(smokeStops, 1);

  // 3. Restart session preferring Wasm64 (preferSmoke: false) -> running_wasm.
  const restartWasm = await controller.restartSession({ preferSmoke: false });
  assert.equal(restartWasm.ok, true);
  assert.equal(restartWasm.action, HUD_ACTIONS.RESTART_SESSION);
  assert.equal(restartWasm.state, SESSION_STATE.RUNNING_WASM);
  assert.equal(restartWasm.mode, SESSION_MODE.WASM);
  assert.equal(wasmStarts, 1);

  // 4. Restart session with preferSmoke: true -> stops Wasm64 and boots smoke demo.
  const restartSmoke = await controller.restartSession({ preferSmoke: true });
  assert.equal(restartSmoke.ok, true);
  assert.equal(restartSmoke.state, SESSION_STATE.RUNNING_SMOKE);
  assert.equal(restartSmoke.mode, SESSION_MODE.SMOKE);
  assert.equal(wasmStops, 1);
  assert.equal(smokeStarts, 2);

  await sessionController.stop();
  assert.equal(smokeStops, 2);
});

test('bindButtons wires DOM controls cleanly, avoids duplicate listeners, and cleans up on destroy', async () => {
  const cacheDir = createMockOpfsDirHandle(DEFAULT_OPFS_CACHE_DIR, {
    'block.bin': new Uint8Array(512),
  });
  const opfsRoot = createMockOpfsDirHandle('', {
    [DEFAULT_OPFS_CACHE_DIR]: cacheDir,
  });
  const storageManager = new BrowserStorageManager({
    storageObj: {
      async estimate() {
        return { usage: 512, quota: 1024 * 1024 };
      },
      async persisted() {
        return false;
      },
      async getDirectory() {
        return opfsRoot;
      },
    },
    opfsRoot,
    indexedDBFactory: createMockIndexedDB(),
  });
  const gpuRecorder = new GpuFrameRecorder({ autoStart: false });
  const crashReporter = new CrashReporter({ now: () => 1700000000000 });
  const inputProfiles = new InputProfileManager({ storage: createMockLocalStorage() });
  const sessionController = new RuntimeSessionController({
    capabilityProbe: async () => createFullCapabilities(),
    fetchImpl: createMockFetch({ hasWasmBinary: false }),
    startSmokeDemo: async () => ({ stop() {} }),
  });

  const downloads = [];
  const hud = createMockHud();
  const controller = new HudController({
    hud,
    storageManager,
    gpuRecorder,
    crashReporter,
    inputProfiles,
    sessionController,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });

  const buttons = {
    clearCacheBtn: createMockElement(),
    exportSavesBtn: createMockElement(),
    recordTraceBtn: createMockElement(),
    exportTraceBtn: createMockElement(),
    exportCrashBtn: createMockElement(),
    restartBtn: createMockElement(),
    profileSelect: createMockElement('default'),
  };

  controller.bindButtons(buttons);
  // Re-binding replaces previous listeners rather than duplicating them.
  controller.bindButtons(buttons);

  assert.equal(controller.getState().boundControlCount, 7);
  assert.equal(buttons.clearCacheBtn.listenerCount('click'), 1);
  assert.equal(buttons.profileSelect.listenerCount('change'), 1);
  assert.equal(buttons.recordTraceBtn.dataset.recording, 'false');
  assert.equal(buttons.recordTraceBtn.getAttribute('aria-pressed'), 'false');

  // Fire each bound control
  buttons.clearCacheBtn.click();
  await flushAsync();
  assert.equal((await storageManager.inspectOpfsCache()).entryCount, 0);

  buttons.exportSavesBtn.click();
  await flushAsync();
  assert.ok(downloads.some((d) => d.filename === DEFAULT_FILENAMES.USERDATA_BACKUP));

  buttons.recordTraceBtn.click();
  await flushAsync();
  assert.equal(gpuRecorder.isRecording(), true);
  assert.equal(buttons.recordTraceBtn.dataset.recording, 'true');
  assert.equal(buttons.recordTraceBtn.getAttribute('aria-pressed'), 'true');

  gpuRecorder.recordChunk(buildSampleGpuFrame());
  buttons.exportTraceBtn.click();
  await flushAsync();
  assert.ok(downloads.some((d) => d.filename === DEFAULT_FILENAMES.GPU_TRACE));

  buttons.exportCrashBtn.click();
  await flushAsync();
  assert.ok(downloads.some((d) => d.filename === DEFAULT_FILENAMES.CRASH_REPORT));

  buttons.restartBtn.click();
  await flushAsync();
  assert.equal(sessionController.getState(), SESSION_STATE.RUNNING_SMOKE);

  buttons.profileSelect.value = 'southpaw';
  buttons.profileSelect.dispatch('change');
  await flushAsync();
  assert.equal(inputProfiles.getActiveProfile().id, 'southpaw');

  const logCountBeforeDestroy = hud.logs.length;
  const downloadCountBeforeDestroy = downloads.length;

  // Destroy cleans up every listener and is idempotent.
  controller.destroy();
  controller.destroy();

  assert.equal(controller.getState().destroyed, true);
  assert.equal(controller.getState().boundControlCount, 0);
  for (const [name, el] of Object.entries(buttons)) {
    const eventType = name === 'profileSelect' ? 'change' : 'click';
    assert.equal(el.listenerCount(eventType), 0, `${name} should have 0 listeners after destroy()`);
  }

  // Firing elements after destroy must be complete no-ops.
  buttons.clearCacheBtn.click();
  buttons.exportSavesBtn.click();
  buttons.recordTraceBtn.click();
  buttons.exportTraceBtn.click();
  buttons.exportCrashBtn.click();
  buttons.restartBtn.click();
  buttons.profileSelect.value = 'default';
  buttons.profileSelect.dispatch('change');
  await flushAsync();

  assert.equal(hud.logs.length, logCountBeforeDestroy);
  assert.equal(downloads.length, downloadCountBeforeDestroy);
  assert.equal(inputProfiles.getActiveProfile().id, 'southpaw');

  await sessionController.stop();
});

test('gracefully handles absent subsystems and failing download/HUD callbacks without unhandled rejections', async () => {
  const controller = new HudController();

  assert.deepEqual(controller.getState(), {
    isRecording: false,
    sessionState: SESSION_STATE.IDLE,
    sessionMode: SESSION_MODE.NONE,
    activeProfileId: null,
    lastAction: null,
    boundControlCount: 0,
    destroyed: false,
  });

  const results = [
    await controller.clearOpfsCache(),
    await controller.exportSaves(),
    await controller.importSaves('{"files":[]}'),
    await controller.toggleGpuRecording(),
    await controller.exportGpuTrace(),
    await controller.exportCrashReport(),
    await controller.switchInputProfile('fps'),
    await controller.toggleSmokeDemo(),
    await controller.restartSession({ preferSmoke: true }),
  ];

  for (const res of results) {
    assert.equal(res.ok, false);
    assert.equal(typeof res.action, 'string');
    assert.equal(typeof res.summary, 'string');
    assert.equal(typeof res.error, 'string');
  }

  // Failing downloadBlob and throwing hud.appendLog never reject.
  const crashReporter = new CrashReporter({ now: () => 1700000000000 });
  const failingController = new HudController({
    hud: {
      appendLog() {
        throw new Error('HUD DOM detached');
      },
    },
    crashReporter,
    downloadBlob() {
      throw new Error('Download blocked by browser');
    },
  });

  const failRes = await failingController.exportCrashReport();
  assert.equal(failRes.ok, false);
  assert.equal(failRes.action, HUD_ACTIONS.EXPORT_CRASH_REPORT);
  assert.match(failRes.error, /Download blocked by browser/);
});

function createPanelMockDocument() {
  function makeNode(tagName) {
    const listeners = new Map();
    const attributes = new Map();
    const node = {
      tagName: tagName.toUpperCase(),
      className: '',
      textContent: '',
      hidden: false,
      style: {},
      children: [],
      parentNode: null,
      setAttribute(name, val) {
        attributes.set(name, String(val));
      },
      getAttribute(name) {
        return attributes.has(name) ? attributes.get(name) : null;
      },
      appendChild(child) {
        if (child && typeof child === 'object') {
          child.parentNode = node;
        }
        node.children.push(child);
        return child;
      },
      removeChild(child) {
        const idx = node.children.indexOf(child);
        if (idx !== -1) {
          node.children.splice(idx, 1);
        }
        if (child && typeof child === 'object') {
          child.parentNode = null;
        }
        return child;
      },
      addEventListener(type, fn) {
        if (!listeners.has(type)) listeners.set(type, new Set());
        listeners.get(type).add(fn);
      },
      removeEventListener(type, fn) {
        listeners.get(type)?.delete(fn);
      },
    };
    return node;
  }

  return {
    head: makeNode('head'),
    body: makeNode('body'),
    createElement(tag) {
      return makeNode(tag);
    },
    querySelector() {
      return null;
    },
  };
}

test('updateGpuBootTelemetry and exportMissingShaders wire cleanly to real GpuBootPanelController and DOM button', async () => {
  const doc = createPanelMockDocument();
  const container = doc.createElement('div');
  const gpuBootPanel = new GpuBootPanelController({
    container,
    documentObj: doc,
    now: () => 1700000555000,
  });

  const downloads = [];
  const clipboardWrites = [];
  const hud = createMockHud();
  const controller = new HudController({
    hud,
    gpuBootPanel,
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
    copyToClipboard(text) {
      clipboardWrites.push(text);
    },
  });

  // Update telemetry via HudController helper.
  const panelState = controller.updateGpuBootTelemetry({
    bootSummary: { stage: 'FIRST_PRESENT', milestonesReached: 9, totalMilestones: 9 },
    frameStats: { framesPresented: 12, droppedFrames: 1 },
    fallbackStats: { totalLookups: 10, packHits: 8, fallbackHits: 2, builtinHits: 0 },
    missingShaders: [
      { hash: '0x0000000000001234', stage: 'vertex', hits: 3, rawPath: 'C:\\Users\\alice\\shader.dxil' },
      { hash: '0x0000000000005678', stage: 'fragment', hits: 7 },
    ],
  });

  assert.ok(panelState);
  assert.equal(panelState.boot.stage, 'FIRST_PRESENT');
  assert.equal(panelState.boot.readyForSplash, true);
  assert.equal(panelState.frameStats.framesPresented, 12);
  assert.equal(panelState.totalMissingShaders, 2);

  // Export missing shaders programmatically with clipboard copy.
  const res = await controller.exportMissingShaders({ copyToClipboard: true });
  assert.equal(res.ok, true);
  assert.equal(res.action, HUD_ACTIONS.EXPORT_MISSING_SHADERS);
  assert.equal(res.filename, DEFAULT_FILENAMES.MISSING_SHADERS);
  assert.equal(res.missingShaderCount, 2);
  assert.equal(res.downloaded, true);
  assert.equal(res.copied, true);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, 'playgta6-missing-shaders.json');
  assert.equal(downloads[0].mimeType, 'application/json');
  assert.equal(downloads[0].data, res.json);
  assert.equal(clipboardWrites.length, 1);
  assert.doesNotMatch(res.json, /alice/i);

  const parsed = JSON.parse(res.json);
  assert.equal(parsed.format, EXPORT_FORMAT);
  assert.equal(parsed.missingShaderCount, 2);
  assert.equal(parsed.missingShaders[0].hash, '0x0000000000005678');
  assert.equal(parsed.missingShaders[0].hits, 7);

  // Bind #btn-export-missing-shaders DOM button and verify click + destroy cleanup.
  const exportMissingBtn = createMockElement();
  controller.bindButtons({ exportMissingShadersBtn: exportMissingBtn });
  assert.equal(controller.getState().boundControlCount, 1);
  assert.equal(exportMissingBtn.listenerCount('click'), 1);

  exportMissingBtn.click();
  await flushAsync();
  assert.equal(downloads.length, 2);

  controller.destroy();
  assert.equal(exportMissingBtn.listenerCount('click'), 0);
  assert.equal(controller.updateGpuBootTelemetry({ frameStats: { framesPresented: 99 } }), null);
  assert.equal(gpuBootPanel.getState().frameStats.framesPresented, 12);

  gpuBootPanel.destroy();
});

test('HudController resolves app.gpuBootPanel fallback and preserves backwards compatibility when absent or failing', async () => {
  // 1. Absent gpuBootPanel: safe no-op for telemetry and graceful failure for export.
  const emptyController = new HudController({ hud: createMockHud() });
  assert.equal(
    emptyController.updateGpuBootTelemetry({
      bootSummary: { stage: 'PGIM_LOAD' },
      missingShaders: [{ hash: '0x1', stage: 'vertex', hits: 1 }],
    }),
    null
  );
  const absentRes = await emptyController.exportMissingShaders();
  assert.equal(absentRes.ok, false);
  assert.equal(absentRes.action, HUD_ACTIONS.EXPORT_MISSING_SHADERS);
  assert.match(absentRes.error, /not configured/i);

  // 2. Fallback via options.app.gpuBootPanel (including late attachment).
  const doc = createPanelMockDocument();
  const appStub = { gpuBootPanel: null };
  const downloads = [];
  const appController = new HudController({
    app: appStub,
    hud: createMockHud(),
    downloadBlob(filename, data, mimeType) {
      downloads.push({ filename, data, mimeType });
    },
  });
  assert.equal(appController.updateGpuBootTelemetry({ bootSummary: { stage: 'PGIM_LOAD' } }), null);

  appStub.gpuBootPanel = new GpuBootPanelController({
    container: doc.createElement('div'),
    documentObj: doc,
    now: () => 1700000999000,
  });

  const updated = appController.updateGpuBootTelemetry({
    bootSummary: { stage: 'BINK_OPEN' },
    missingShaders: [{ hash: '0xabcd', stage: 'compute', hits: 2 }],
  });
  assert.equal(updated.boot.stage, 'BINK_OPEN');
  assert.equal(updated.totalMissingShaders, 1);

  const exp = await appController.exportMissingShaders();
  assert.equal(exp.ok, true);
  assert.equal(exp.missingShaderCount, 1);
  assert.equal(downloads.length, 1);
  assert.equal(downloads[0].filename, DEFAULT_FILENAMES.MISSING_SHADERS);

  // 3. Throwing gpuBootPanel methods never crash caller.
  const throwingController = new HudController({
    hud: createMockHud(),
    gpuBootPanel: {
      update() {
        throw new Error('panel update boom');
      },
      exportMissingShadersJson() {
        throw new Error('panel export boom');
      },
    },
  });
  assert.equal(throwingController.updateGpuBootTelemetry({ bootSummary: { stage: 'PGIM_LOAD' } }), null);
  const errRes = await throwingController.exportMissingShaders();
  assert.equal(errRes.ok, false);
  assert.equal(errRes.action, HUD_ACTIONS.EXPORT_MISSING_SHADERS);
  assert.match(errRes.error, /panel export boom/);

  appStub.gpuBootPanel.destroy();
});

// --- W69: Real-vs-Smoke Boot Mode, Split-Module Progress, and Live WebGPU Validation Errors ---

function createDisplayMockElement(extra = {}) {
  const attributes = new Map();
  return {
    textContent: '',
    title: '',
    hidden: false,
    dataset: {},
    children: [],
    _attributes: attributes,
    setAttribute(name, val) {
      attributes.set(name, String(val));
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
    ...extra,
  };
}

test('W69: normalizeBootMode and bootMode tracking distinguish real-wasm64, smoke, and idle without mislabeling smoke frames', async () => {
  assert.equal(normalizeBootMode('real-wasm64'), BOOT_MODE.REAL_WASM64);
  assert.equal(normalizeBootMode('wasm-guest'), BOOT_MODE.REAL_WASM64);
  assert.equal(normalizeBootMode('running_wasm'), BOOT_MODE.REAL_WASM64);
  assert.equal(normalizeBootMode('smoke'), BOOT_MODE.SMOKE);
  assert.equal(normalizeBootMode('smoke-demo'), BOOT_MODE.SMOKE);
  assert.equal(normalizeBootMode('running_smoke'), BOOT_MODE.SMOKE);
  assert.equal(normalizeBootMode('idle'), BOOT_MODE.IDLE);
  assert.equal(normalizeBootMode('pending'), BOOT_MODE.IDLE);
  assert.equal(normalizeBootMode('unknown-garbage'), BOOT_MODE.IDLE);
  assert.equal(normalizeBootMode(null), BOOT_MODE.IDLE);

  const bootModeBadge = createDisplayMockElement();
  const bootStatusEl = createDisplayMockElement();
  const hud = createMockHud();

  const controller = new HudController({
    hud,
    displayElements: { bootModeBadge, bootStatusEl },
  });

  // Initial state is idle.
  assert.equal(controller.getBootMode(), BOOT_MODE.IDLE);
  assert.equal(bootModeBadge.textContent, 'Idle');
  assert.equal(bootModeBadge.dataset.bootMode, 'idle');
  assert.equal(bootModeBadge.dataset.state, 'info');
  assert.equal(bootStatusEl.textContent, 'Idle');

  // Switch to smoke mode -> unmistakable smoke-demo badge and status text.
  const smokeStatus = controller.setBootMode('smoke');
  assert.equal(smokeStatus.bootMode, 'smoke');
  assert.equal(smokeStatus.isSmoke, true);
  assert.equal(smokeStatus.isRealWasm64, false);
  assert.equal(smokeStatus.badgeText, 'Smoke Demo');
  assert.match(smokeStatus.statusText, /Smoke Demo/i);
  assert.match(smokeStatus.statusText, /Not Real Game/i);
  assert.equal(bootModeBadge.textContent, 'Smoke Demo');
  assert.equal(bootModeBadge.dataset.bootMode, 'smoke');
  assert.equal(bootModeBadge.dataset.state, 'warn');
  assert.equal(bootModeBadge.getAttribute('aria-label'), 'Boot mode: Smoke Demo');
  assert.equal(hud.statuses.at(-1).level, 'warn');

  // Receiving frame telemetry without a real boot transition must keep smoke mode.
  controller.updateGpuBootTelemetry({
    bootSummary: { stage: 'FIRST_PRESENT', milestonesReached: 9, totalMilestones: 9 },
    frameStats: { framesPresented: 120 },
  });
  assert.equal(controller.getBootMode(), 'smoke');
  assert.equal(bootModeBadge.textContent, 'Smoke Demo');

  // Conflicting firstPresentSource='smoke-demo' with bootMode='real-wasm64' must NEVER mislabel as real game.
  const guarded = controller.setBootMode('real-wasm64', { firstPresentSource: 'smoke-demo' });
  assert.equal(guarded.bootMode, 'smoke');
  assert.equal(guarded.isRealWasm64, false);
  assert.equal(bootModeBadge.textContent, 'Smoke Demo');

  controller.updateGpuBootTelemetry({
    bootMode: 'real-wasm64',
    firstPresentSource: 'smoke-demo',
  });
  assert.equal(controller.getBootMode(), 'smoke');

  // Valid transition to real-wasm64 updates badge and status text cleanly.
  const realStatus = controller.setBootMode('real-wasm64');
  assert.equal(realStatus.bootMode, 'real-wasm64');
  assert.equal(realStatus.isRealWasm64, true);
  assert.equal(realStatus.isSmoke, false);
  assert.equal(realStatus.badgeText, 'Real Wasm64');
  assert.equal(realStatus.statusText, 'Real Wasm64 Guest');
  assert.equal(bootModeBadge.textContent, 'Real Wasm64');
  assert.equal(bootModeBadge.dataset.bootMode, 'real-wasm64');
  assert.equal(bootModeBadge.dataset.state, 'ok');
  assert.equal(hud.statuses.at(-1).level, 'ok');
});

test('W69: active RuntimeSessionController smoke mode can never be mislabeled as real-wasm64 and syncs across lifecycle transitions', async () => {
  const sessionController = new RuntimeSessionController({
    capabilityProbe: async () => createFullCapabilities(),
    fetchImpl: createMockFetch({ hasWasmBinary: true }),
    startWasmRuntime: async () => ({ stop() {} }),
    startSmokeDemo: async () => ({ stop() {} }),
  });

  const bootModeBadge = createDisplayMockElement();
  const bootStatusEl = createDisplayMockElement();
  const controller = new HudController({
    hud: createMockHud(),
    sessionController,
    displayElements: { bootModeBadge, bootStatusEl },
  });

  assert.equal(controller.getBootMode(), 'idle');

  // Start smoke demo via HudController.
  await controller.toggleSmokeDemo();
  assert.equal(sessionController.getState(), SESSION_STATE.RUNNING_SMOKE);
  assert.equal(controller.getBootMode(), 'smoke');
  assert.equal(bootModeBadge.textContent, 'Smoke Demo');
  assert.equal(bootModeBadge.dataset.bootMode, 'smoke');

  // Even if caller tries to force real-wasm64 while smoke session is active, guard preserves smoke.
  controller.setBootMode('real-wasm64');
  assert.equal(controller.getBootMode(), 'smoke');
  assert.equal(bootModeBadge.textContent, 'Smoke Demo');
  assert.equal(controller.getBootStatus().isRealWasm64, false);

  // Stop smoke directly on sessionController -> immediately returns to idle.
  await sessionController.stop();
  assert.equal(controller.getBootMode(), 'idle');

  // Restart into Wasm64 runtime -> updates to real-wasm64.
  await controller.restartSession({ preferSmoke: false });
  assert.equal(sessionController.getState(), SESSION_STATE.RUNNING_WASM);
  assert.equal(controller.getBootMode(), 'real-wasm64');
  assert.equal(bootModeBadge.textContent, 'Real Wasm64');
  assert.equal(bootModeBadge.dataset.bootMode, 'real-wasm64');
  assert.equal(bootModeBadge.dataset.state, 'ok');

  // Restart back into smoke demo -> updates to smoke.
  await controller.restartSession({ preferSmoke: true });
  assert.equal(controller.getBootMode(), 'smoke');
  assert.equal(bootModeBadge.textContent, 'Smoke Demo');

  await sessionController.stop();

  // Also verify inference from app.bootOrchestrator when sessionController is absent.
  const orchController = new HudController({
    app: {
      bootOrchestrator: {
        getState() {
          return { bootMode: 'wasm-guest' };
        },
      },
    },
  });
  assert.equal(orchController.getBootMode(), 'real-wasm64');
});

test('W69: split-module (sideModuleCount / guest_part_*.wasm) progress tracks, deduplicates, sanitizes paths, and renders cleanly', () => {
  const bootStatusEl = createDisplayMockElement();
  const sideModuleStatusEl = createDisplayMockElement();
  const hud = createMockHud();

  const controller = new HudController({
    hud,
    bootMode: 'real-wasm64',
    displayElements: { bootStatusEl, sideModuleStatusEl },
  });

  // Initial state: no split modules reported.
  const initial = controller.getSideModuleProgress();
  assert.equal(initial.active, false);
  assert.equal(initial.sideModuleCount, 0);
  assert.equal(initial.loadedCount, 0);
  assert.equal(initial.complete, false);
  assert.equal(sideModuleStatusEl.textContent, 'No split modules');

  // Announce 4 split side modules before any finish loading.
  const announced = controller.setSideModuleProgress({ sideModuleCount: 4, loadedCount: 0 });
  assert.equal(announced.active, true);
  assert.equal(announced.sideModuleCount, 4);
  assert.equal(announced.loadedCount, 0);
  assert.equal(announced.remainingCount, 4);
  assert.equal(announced.progressPct, 0);
  assert.equal(announced.statusText, 'Side modules: 0/4 (0%)');
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 0/4 (0%)');
  assert.equal(sideModuleStatusEl.dataset.state, 'info');
  assert.equal(bootStatusEl.textContent, 'Real Wasm64 Guest — Side modules: 0/4 (0%)');

  // Load first module incrementally + verify duplicate load does not double-count.
  controller.recordSideModuleLoaded('guest_part_0.wasm');
  const dup = controller.recordSideModuleLoaded('./dist/guest_part_0.wasm?v=1');
  assert.equal(dup.loadedCount, 1);
  assert.equal(dup.remainingCount, 3);
  assert.equal(dup.progressPct, 25);
  assert.deepEqual(dup.loadedModules, ['guest_part_0.wasm']);

  // Host path in side module name is stripped to basename (zero host username leakage).
  const second = controller.recordSideModuleLoaded(
    'C:\\Users\\secretuser\\playgta6web\\web\\dist\\guest_part_1.wasm?token=abc'
  );
  assert.equal(second.loadedCount, 2);
  assert.equal(second.progressPct, 50);
  assert.equal(second.currentModule, 'guest_part_1.wasm');
  assert.deepEqual(second.loadedModules, ['guest_part_0.wasm', 'guest_part_1.wasm']);
  assert.doesNotMatch(second.loadedModules.join(','), /secretuser|token/);
  assert.equal(bootStatusEl.textContent, 'Real Wasm64 Guest — Side modules: 2/4 (50%)');

  // Load remaining 2 modules -> complete state.
  controller.recordSideModuleLoaded('/home/alice/dist/guest_part_2.wasm');
  const done = controller.recordSideModuleLoaded('guest_part_3.wasm', { updateHudStatus: true });
  assert.equal(done.complete, true);
  assert.equal(done.loadedCount, 4);
  assert.equal(done.remainingCount, 0);
  assert.equal(done.progressPct, 100);
  assert.equal(done.statusText, 'Side modules: 4/4 loaded');
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 4/4 loaded');
  assert.equal(sideModuleStatusEl.dataset.state, 'ok');
  assert.equal(sideModuleStatusEl.dataset.loaded, '4');
  assert.equal(sideModuleStatusEl.dataset.total, '4');
  assert.equal(bootStatusEl.textContent, 'Real Wasm64 Guest — Side modules: 4/4 loaded');
  assert.equal(hud.statuses.at(-1).text, 'Real Wasm64 Guest — Side modules: 4/4 loaded');

  // Update via updateGpuBootTelemetry payload works seamlessly too.
  controller.updateGpuBootTelemetry({
    side_module_count: 6,
    loadedSideModules: 3,
  });
  assert.equal(controller.getSideModuleProgress().sideModuleCount, 6);
  assert.equal(controller.getSideModuleProgress().loadedCount, 3);
  assert.equal(controller.getSideModuleProgress().progressPct, 50);

  // Invalid/negative numbers degrade safely without throwing, and reset clears state.
  controller.setSideModuleProgress({ sideModuleCount: -10, loadedCount: Number.NaN });
  assert.equal(controller.getSideModuleProgress().sideModuleCount, 0);
  assert.equal(controller.getSideModuleProgress().loadedCount, 0);

  controller.recordSideModuleLoaded('guest_part_0.wasm');
  const reset = controller.resetSideModuleProgress();
  assert.equal(reset.active, false);
  assert.equal(reset.sideModuleCount, 0);
  assert.equal(reset.loadedCount, 0);
  assert.deepEqual(reset.loadedModules, []);
  assert.equal(sideModuleStatusEl.textContent, 'No split modules');
});

test('W69: live WebGPU validation errors are bounded, sanitized, capped in length, and rendered cleanly', () => {
  const bootStatusEl = createDisplayMockElement();
  const validationCountEl = createDisplayMockElement();
  const validationLatestEl = createDisplayMockElement();
  const validationBannerEl = createDisplayMockElement({ hidden: true });
  const hud = createMockHud();

  const controller = new HudController({
    hud,
    bootMode: 'real-wasm64',
    maxValidationErrors: 3,
    maxValidationMessageLength: 96,
    now: () => 1700001000000,
    displayElements: {
      bootStatusEl,
      validationCountEl,
      validationLatestEl,
      validationBannerEl,
    },
  });

  // Initial empty summary.
  const empty = controller.getValidationSummary();
  assert.equal(empty.count, 0);
  assert.equal(empty.historySize, 0);
  assert.equal(empty.maxHistory, 3);
  assert.equal(empty.hasErrors, false);
  assert.equal(empty.latestError, null);
  assert.equal(empty.latestMessage, null);
  assert.deepEqual(empty.errors, []);
  assert.equal(validationCountEl.textContent, '0');
  assert.equal(validationCountEl.dataset.state, 'ok');
  assert.equal(validationLatestEl.textContent, 'None');
  assert.equal(validationBannerEl.hidden, true);

  // Record #1 with Windows host path, POSIX host path, query secret, and control characters.
  const rec1 = controller.recordValidationError({
    type: 'validation',
    message:
      'Invalid shader at C:\\Users\\alice\\GTA6\\shader.wgsl\nand /home/alice/secret.bin?token=xyz Bearer secret-tok',
    timestampMs: 1700001000100,
  });
  assert.ok(rec1);
  assert.equal(rec1.type, 'validation');
  assert.doesNotMatch(rec1.message, /alice|secret-tok|token=xyz|[\r\n]/i);
  assert.ok(rec1.message.length <= 96);
  assert.equal(validationCountEl.textContent, '1');
  assert.equal(validationCountEl.dataset.state, 'err');
  assert.equal(validationLatestEl.textContent, rec1.message);
  assert.equal(validationBannerEl.hidden, false);
  assert.equal(validationBannerEl.dataset.hasErrors, 'true');
  assert.match(bootStatusEl.textContent, /1 WebGPU validation error/);
  assert.equal(controller.getBootStatus().statusLevel, 'error');
  assert.equal(hud.logs.length, 1);
  assert.equal(hud.logs[0].level, 'error');

  // Record #2 (OOM) + #3 (overly long message -> length capped with ellipsis) + #4 (evicts #1 from bounded ring).
  controller.recordValidationError({
    error: { name: 'GPUOutOfMemoryError', message: 'Allocation failed for file:///Users/bob/tex.bin' },
  });
  const longMsg = 'BindGroup layout mismatch at binding 0 in render pass pipeline validation. '.repeat(4);
  const rec3 = controller.recordValidationError(longMsg);
  assert.equal(rec3.message.length, 96);
  assert.ok(rec3.message.endsWith('...'));

  const rec4 = controller.recordValidationError(
    new Error('Pipeline layout incompatible with \\\\fileserver\\private\\dump.rpf')
  );
  assert.doesNotMatch(rec4.message, /fileserver|private/i);

  const summary = controller.getValidationSummary();
  assert.equal(summary.count, 4);
  assert.equal(summary.historySize, 3);
  assert.equal(summary.maxHistory, 3);
  assert.equal(summary.errors.length, 3);
  assert.equal(summary.errors[0].type, 'out-of-memory');
  assert.doesNotMatch(summary.errors[0].message, /bob/i);
  assert.equal(summary.latestMessage, rec4.message);
  assert.equal(validationCountEl.textContent, '4');
  assert.equal(validationLatestEl.textContent, rec4.message);
  assert.match(bootStatusEl.textContent, /4 WebGPU validation errors/);

  // Returned summary is a defensive copy.
  summary.errors[0].message = 'MUTATED';
  summary.latestError.message = 'MUTATED';
  assert.notEqual(controller.getValidationSummary().errors[0].message, 'MUTATED');
  assert.notEqual(controller.getValidationSummary().latestMessage, 'MUTATED');

  // setValidationErrors replaces list cleanly without spamming HUD logs and respects totalCount + ring cap.
  const logCountBeforeBulk = hud.logs.length;
  const bulkSummary = controller.setValidationErrors(
    [
      { seq: 10, type: 'validation', message: 'Error A from /Users/carol/a.wgsl' },
      { seq: 11, type: 'internal', message: 'Error B' },
      { seq: 12, type: 'validation', message: 'Error C' },
      { seq: 13, type: 'validation', message: 'Error D from C:\\Users\\carol\\d.wgsl' },
    ],
    { totalCount: 12 }
  );
  assert.equal(hud.logs.length, logCountBeforeBulk);
  assert.equal(bulkSummary.count, 12);
  assert.equal(bulkSummary.historySize, 3);
  assert.equal(bulkSummary.errors[0].seq, 11);
  assert.equal(bulkSummary.latestError.seq, 13);
  assert.doesNotMatch(bulkSummary.latestMessage, /carol/i);
  assert.equal(validationCountEl.textContent, '12');

  // clearValidationErrors resets counters and hides the banner.
  const cleared = controller.clearValidationErrors();
  assert.equal(cleared.count, 0);
  assert.equal(cleared.historySize, 0);
  assert.equal(cleared.hasErrors, false);
  assert.equal(validationCountEl.textContent, '0');
  assert.equal(validationLatestEl.textContent, 'None');
  assert.equal(validationBannerEl.hidden, true);
  assert.equal(controller.getBootStatus().statusLevel, 'ok');

  // Default constants and direct sanitizeValidationErrorMessage checks.
  assert.equal(DEFAULT_MAX_VALIDATION_ERRORS, 32);
  assert.equal(MAX_VALIDATION_ERROR_MESSAGE_LENGTH, 256);
  const directSanitized = sanitizeValidationErrorMessage(
    'Failed at C:\\Users\\dave\\file.bin data:application/octet-stream;base64,QUFBQUFBQUFBQUFBQUFBQUFBQUFB',
    120
  );
  assert.doesNotMatch(directSanitized, /dave|QUFB/);
  assert.match(directSanitized, /\[REDACTED_PATH\]/);
  assert.match(directSanitized, /\[REDACTED_BUFFER\]/);
});

test('W69: absent, frozen, or throwing optional DOM elements never throw and destroy() stops further mutations', () => {
  // 1. Completely absent DOM elements and HUD.
  const bare = new HudController();
  assert.doesNotThrow(() => {
    bare.setBootMode('real-wasm64');
    bare.setSideModuleProgress({ sideModuleCount: 2, loadedCount: 1 });
    bare.recordSideModuleLoaded('guest_part_1.wasm');
    bare.recordValidationError('Invalid vertex buffer stride');
    bare.setValidationErrors([{ message: 'Blend state mismatch' }]);
    bare.clearValidationErrors();
    bare.renderBootStatus({ updateHudStatus: true });
  });
  assert.equal(bare.getBootMode(), 'real-wasm64');
  assert.equal(bare.getValidationSummary().count, 0);

  // 2. Frozen / throwing DOM elements and throwing hud.setStatus never crash.
  const throwingEl = {
    set textContent(_v) {
      throw new Error('DOM detached');
    },
    set title(_v) {
      throw new Error('DOM detached');
    },
    set hidden(_v) {
      throw new Error('DOM detached');
    },
    get dataset() {
      throw new Error('DOM detached');
    },
    setAttribute() {
      throw new Error('DOM detached');
    },
  };
  const resilient = new HudController({
    hud: {
      appendLog() {
        throw new Error('log detached');
      },
      setStatus() {
        throw new Error('status detached');
      },
    },
    displayElements: {
      bootModeBadge: Object.freeze({}),
      bootStatusEl: throwingEl,
      sideModuleStatusEl: throwingEl,
      validationCountEl: throwingEl,
      validationLatestEl: throwingEl,
      validationBannerEl: throwingEl,
    },
  });

  assert.doesNotThrow(() => {
    resilient.setBootMode('smoke');
    resilient.recordSideModuleLoaded('guest_part_0.wasm');
    resilient.recordValidationError({ message: 'Attachment format mismatch' });
  });
  assert.equal(resilient.getBootMode(), 'smoke');
  assert.equal(resilient.getValidationSummary().count, 1);

  // 3. Container validationBannerEl with child elements preserves child DOM while updating attributes/hidden.
  const childNode = createDisplayMockElement();
  const containerBanner = createDisplayMockElement({ children: [childNode], textContent: 'KEEP' });
  const btn = createMockElement();
  const withBindButtons = new HudController();
  withBindButtons.bindButtons({
    clearCacheBtn: btn,
    validationBannerEl: containerBanner,
  });
  // Display element passed to bindButtons does not inflate boundControlCount.
  assert.equal(withBindButtons.getState().boundControlCount, 1);
  withBindButtons.recordValidationError('Render pass depth format invalid');
  assert.equal(containerBanner.hidden, false);
  assert.equal(containerBanner.dataset.count, '1');
  assert.equal(containerBanner.textContent, 'KEEP');

  // 4. After destroy(), mutators safely no-op without changing state.
  withBindButtons.destroy();
  assert.equal(withBindButtons.recordValidationError('Should be ignored'), null);
  assert.equal(withBindButtons.getValidationSummary().count, 1);
  withBindButtons.setBootMode('real-wasm64');
  assert.equal(withBindButtons.getBootMode(), 'idle');
  withBindButtons.setSideModuleProgress({ sideModuleCount: 5, loadedCount: 5 });
  assert.equal(withBindButtons.getSideModuleProgress().sideModuleCount, 0);
});

// --- W78: Track Loaded Split Wasm Side Modules from Progress Labels ---

test('W78: parseSideModuleProgressLabel parses layout, side-module loads, runtime init, and ignores unrelated/error lines', () => {
  assert.deepEqual(
    parseSideModuleProgressLabel('Split-module Wasm64 layout detected (5 side modules).'),
    { kind: 'layout', sideModuleCount: 5 }
  );
  assert.deepEqual(
    parseSideModuleProgressLabel('Split-module Wasm64 layout detected (1 side module).'),
    { kind: 'layout', sideModuleCount: 1 }
  );
  assert.deepEqual(
    parseSideModuleProgressLabel('Loading Wasm side module (guest_part_0.wasm)...'),
    { kind: 'module', moduleName: 'guest_part_0.wasm' }
  );
  assert.deepEqual(
    parseSideModuleProgressLabel(
      'Loading Wasm side module (C:\\Users\\secretuser\\playgta6web\\web\\dist\\guest_part_2.wasm?token=abc)...'
    ),
    { kind: 'module', moduleName: 'guest_part_2.wasm' }
  );
  assert.deepEqual(
    parseSideModuleProgressLabel('Loaded Wasm side module (guest_part_3.wasm)'),
    { kind: 'module', moduleName: 'guest_part_3.wasm' }
  );
  assert.deepEqual(
    parseSideModuleProgressLabel('Wasm side module loaded (guest_part_4.wasm)'),
    { kind: 'module', moduleName: 'guest_part_4.wasm' }
  );
  assert.deepEqual(
    parseSideModuleProgressLabel('[wasm] playgta6web Wasm64 runtime initialized (ABI v1)'),
    { kind: 'runtime_initialized' }
  );
  assert.deepEqual(parseSideModuleProgressLabel('[boot] XMEM_INIT reached'), {
    kind: 'runtime_initialized',
  });
  assert.deepEqual(
    parseSideModuleProgressLabel('Loading Emscripten Wasm64 runtime (dist/runtime.js)...'),
    { kind: 'runtime_starting' }
  );

  // Pre-init PGIM_LOAD, failure lines, and unrelated messages must return null.
  assert.equal(parseSideModuleProgressLabel('[boot] PGIM_LOAD mapped'), null);
  assert.equal(parseSideModuleProgressLabel('[boot] XMEM_INIT failed: out of memory'), null);
  assert.equal(parseSideModuleProgressLabel('wgpu_worker ready (Adapter)'), null);
  assert.equal(parseSideModuleProgressLabel(''), null);
  assert.equal(parseSideModuleProgressLabel(null), null);
});

test('W78: HudController observes hud.appendLog progress labels, deduplicates pthread replays, caps total, and detaches on destroy', () => {
  const sideModuleStatusEl = createDisplayMockElement();
  const bootStatusEl = createDisplayMockElement();
  const hud = createMockHud();
  const originalAppendLog = hud.appendLog;

  const controller = new HudController({
    hud,
    bootMode: 'real-wasm64',
    displayElements: { sideModuleStatusEl, bootStatusEl },
  });

  // Live Chrome Test #25 progress sequence through hud.appendLog:
  hud.appendLog('Split-module Wasm64 layout detected (5 side modules).', 'info');
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 0/5 (0%)');
  assert.equal(sideModuleStatusEl.dataset.loaded, '0');
  assert.equal(sideModuleStatusEl.dataset.total, '5');
  assert.equal(sideModuleStatusEl.dataset.state, 'info');

  const expectedSteps = [
    ['guest_part_0.wasm', 1, 'Side modules: 1/5 (20%)', 'info'],
    ['guest_part_1.wasm', 2, 'Side modules: 2/5 (40%)', 'info'],
    ['guest_part_2.wasm', 3, 'Side modules: 3/5 (60%)', 'info'],
    ['guest_part_3.wasm', 4, 'Side modules: 4/5 (80%)', 'info'],
    ['guest_part_4.wasm', 5, 'Side modules: 5/5 loaded', 'ok'],
  ];

  for (const [name, expectedLoaded, expectedText, expectedState] of expectedSteps) {
    hud.appendLog(`Loading Wasm side module (${name})...`, 'info');
    const prog = controller.getSideModuleProgress();
    assert.equal(prog.sideModuleCount, 5);
    assert.equal(prog.loadedCount, expectedLoaded);
    assert.equal(sideModuleStatusEl.textContent, expectedText);
    assert.equal(sideModuleStatusEl.dataset.loaded, String(expectedLoaded));
    assert.equal(sideModuleStatusEl.dataset.total, '5');
    assert.equal(sideModuleStatusEl.dataset.state, expectedState);
  }

  // Pthread workers re-emitting the same side module names (including case variants/paths) must not double-count.
  hud.appendLog('Loading Wasm side module (guest_part_0.wasm)...', 'info');
  hud.appendLog('Loading Wasm side module (./dist/GUEST_PART_1.WASM?v=2)...', 'info');
  // Stray extra module beyond announced 5 side modules stays capped at 5/5.
  hud.appendLog('Loading Wasm side module (guest_part_99.wasm)...', 'info');
  hud.appendLog('[wasm] playgta6web Wasm64 runtime initialized (ABI v1)', 'info');

  const finalProg = controller.getSideModuleProgress();
  assert.equal(finalProg.sideModuleCount, 5);
  assert.equal(finalProg.loadedCount, 5);
  assert.equal(finalProg.progressPct, 100);
  assert.equal(finalProg.complete, true);
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 5/5 loaded');
  assert.equal(sideModuleStatusEl.dataset.state, 'ok');

  // All messages still reached the underlying HUD log array untouched.
  assert.equal(hud.logs.length, 10);

  // Subsequent reboot cycle resets progress cleanly on next layout announcement.
  hud.appendLog('Split-module Wasm64 layout detected (5 side modules).', 'info');
  assert.equal(controller.getSideModuleProgress().loadedCount, 0);
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 0/5 (0%)');

  // destroy() restores original hud.appendLog and stops observing future logs.
  controller.destroy();
  assert.equal(hud.appendLog, originalAppendLog);
  hud.appendLog('Loading Wasm side module (guest_part_0.wasm)...', 'info');
  assert.equal(controller.getSideModuleProgress().loadedCount, 0);
  assert.equal(hud.logs.length, 12);
});

test('W78: runtime initialization and post-init milestones mark pending side modules complete without affecting monolithic or smoke builds', () => {
  // 1. Missed side-module log lines complete cleanly on runtime initialization.
  const sideEl = createDisplayMockElement();
  const hud = createMockHud();
  const controller = new HudController({
    hud,
    bootMode: 'real-wasm64',
    displayElements: { sideModuleStatusEl: sideEl },
  });

  controller.setSideModules({ sideModuleCount: 5 });
  assert.equal(sideEl.textContent, 'Side modules: 0/5 (0%)');

  // PGIM_LOAD happens during preload before side modules load -> must stay 0/5.
  controller.updateGpuBootTelemetry({ bootSummary: { stage: 'PGIM_LOAD' } });
  assert.equal(controller.getSideModuleProgress().loadedCount, 0);
  assert.equal(sideEl.textContent, 'Side modules: 0/5 (0%)');

  // Partial load (2/5) followed by runtime init completes to 5/5.
  hud.appendLog('Loading Wasm side module (guest_part_0.wasm)...', 'info');
  hud.appendLog('Loading Wasm side module (guest_part_1.wasm)...', 'info');
  assert.equal(sideEl.textContent, 'Side modules: 2/5 (40%)');

  hud.appendLog('[wasm] playgta6web Wasm64 runtime initialized (ABI v1)', 'info');
  assert.equal(controller.getSideModuleProgress().loadedCount, 5);
  assert.equal(controller.getSideModuleProgress().complete, true);
  assert.equal(sideEl.textContent, 'Side modules: 5/5 loaded');

  // 2. XMEM_INIT via updateGpuBootTelemetry also marks pending side modules complete (e.g. frozen hud).
  const sideEl2 = createDisplayMockElement();
  const frozenHud = Object.freeze({ appendLog() {} });
  const controller2 = new HudController({
    hud: frozenHud,
    bootMode: 'real-wasm64',
    displayElements: { sideModuleStatusEl: sideEl2 },
  });
  controller2.setSideModuleProgress({ sideModuleCount: 5, loadedCount: 0 });
  controller2.updateGpuBootTelemetry({ bootSummary: { stage: 'XMEM_INIT' } });
  assert.equal(controller2.getSideModuleProgress().loadedCount, 5);
  assert.equal(controller2.getSideModuleProgress().complete, true);
  assert.equal(sideEl2.textContent, 'Side modules: 5/5 loaded');

  // 3. Monolithic build (0 side modules) stays inactive after runtime init.
  const monoEl = createDisplayMockElement();
  const monoHud = createMockHud();
  const monoController = new HudController({
    hud: monoHud,
    bootMode: 'real-wasm64',
    displayElements: { sideModuleStatusEl: monoEl },
  });
  monoHud.appendLog('[wasm] playgta6web Wasm64 runtime initialized (ABI v1)', 'info');
  assert.equal(monoController.getSideModuleProgress().active, false);
  assert.equal(monoEl.textContent, 'No split modules');

  // 4. Smoke mode never marks side modules complete from runtime-init logs or FIRST_PRESENT telemetry.
  const smokeEl = createDisplayMockElement();
  const smokeHud = createMockHud();
  const smokeController = new HudController({
    hud: smokeHud,
    bootMode: 'smoke',
    displayElements: { sideModuleStatusEl: smokeEl },
  });
  smokeController.setSideModuleProgress({ sideModuleCount: 5, loadedCount: 0 });
  smokeHud.appendLog('[wasm] playgta6web Wasm64 runtime initialized (ABI v1)', 'info');
  smokeController.updateGpuBootTelemetry({ bootSummary: { stage: 'FIRST_PRESENT' } });
  assert.equal(smokeController.getSideModuleProgress().loadedCount, 0);
  assert.equal(smokeEl.textContent, 'Side modules: 0/5 (0%)');

  controller.destroy();
  controller2.destroy();
  monoController.destroy();
  smokeController.destroy();
});

test('W78: end-to-end bootstrapPlayGta6App + BroadcastChannel game-progress + HudController updates HUD to 5/5 loaded', async () => {
  const channels = [];
  class MockBroadcastChannel {
    constructor(name) {
      this.name = name;
      this.onmessage = null;
      this.closed = false;
      channels.push(this);
    }
    emit(data) {
      if (!this.closed && typeof this.onmessage === 'function') {
        this.onmessage({ data });
      }
    }
    close() {
      this.closed = true;
    }
  }

  const sideModuleStatusEl = createDisplayMockElement();
  const bootModeBadge = createDisplayMockElement();
  let hudController = null;

  const hudApi = {
    logs: [],
    statuses: [],
    metrics: [],
    appendLog(message, level = 'info') {
      this.logs.push({ message, level });
    },
    setStatus(text, level = 'info') {
      this.statuses.push({ text, level });
    },
    updateMetrics(payload) {
      this.metrics.push({ ...payload });
    },
    setBootMode(mode, opts) {
      if (hudController) {
        hudController.setBootMode(mode, { ...opts, updateHudStatus: false });
      }
    },
    updateBootStatus(snapshot) {
      if (hudController && snapshot) {
        if (snapshot.sideModuleCount !== undefined || Array.isArray(snapshot.sideModuleUrls)) {
          hudController.setSideModuleProgress(
            {
              sideModuleCount: snapshot.sideModuleCount,
              sideModules: snapshot.sideModuleUrls,
              loadedCount: snapshot.loadedSideModules ?? snapshot.loadedSideModuleCount,
            },
            { updateHudStatus: false }
          );
        }
        if (snapshot.bootMode !== undefined) {
          hudController.setBootMode(snapshot.bootMode, { updateHudStatus: false });
        }
      }
    },
    setSideModules(progress) {
      if (hudController) {
        hudController.setSideModuleProgress(progress, { updateHudStatus: false });
      }
    },
  };

  const mockWindow = {
    crossOriginIsolated: true,
    SharedArrayBuffer,
    BroadcastChannel: MockBroadcastChannel,
    location: { origin: 'http://127.0.0.1:8080' },
    addEventListener() {},
    removeEventListener() {},
  };

  const sideUrls = [
    '/dist/guest_part_0.wasm',
    '/dist/guest_part_1.wasm',
    '/dist/guest_part_2.wasm',
    '/dist/guest_part_3.wasm',
    '/dist/guest_part_4.wasm',
  ];

  const app = bootstrapPlayGta6App({
    windowObj: mockWindow,
    documentObj: null,
    navigatorObj: { gpu: {} },
    canvas: { width: 640, height: 360 },
    hud: hudApi,
    capabilityProbe: () => createFullCapabilities(),
    fetchImpl: createMockFetch({
      hasWasmBinary: true,
      sideModuleCount: 5,
      sideModuleUrls: sideUrls,
    }),
    createWasmWorker: async () => ({
      postMessage() {},
      addEventListener() {},
      removeEventListener() {},
      terminate() {},
    }),
    attachGlobalErrorHandlers: false,
    attachGlobalHook: false,
    autoResize: false,
  });

  hudController = new HudController({
    app,
    sessionController: app.session,
    hud: hudApi,
    displayElements: { bootModeBadge, sideModuleStatusEl },
  });

  await app.start();
  assert.equal(bootModeBadge.textContent, 'Real Wasm64');
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 0/5 (0%)');

  const progressChannel = channels.find((ch) => ch.name === 'game-progress');
  assert.ok(progressChannel, 'expected game-progress BroadcastChannel to be opened');

  progressChannel.emit({ label: 'Loading Wasm side module (guest_part_0.wasm)...' });
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 1/5 (20%)');

  progressChannel.emit({ label: 'Loading Wasm side module (guest_part_1.wasm)...' });
  progressChannel.emit({ label: 'Loading Wasm side module (guest_part_2.wasm)...' });
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 3/5 (60%)');

  progressChannel.emit({ label: 'Loading Wasm side module (guest_part_3.wasm)...' });
  progressChannel.emit({ label: 'Loading Wasm side module (guest_part_4.wasm)...' });
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 5/5 loaded');
  assert.equal(sideModuleStatusEl.dataset.loaded, '5');
  assert.equal(sideModuleStatusEl.dataset.total, '5');
  assert.equal(sideModuleStatusEl.dataset.state, 'ok');

  progressChannel.emit({ label: '[wasm] playgta6web Wasm64 runtime initialized (ABI v1)' });
  progressChannel.emit({ label: '[boot] XMEM_INIT' });
  assert.equal(hudController.getSideModuleProgress().loadedCount, 5);
  assert.equal(hudController.getSideModuleProgress().progressPct, 100);
  assert.equal(sideModuleStatusEl.textContent, 'Side modules: 5/5 loaded');

  hudController.destroy();
  await app.destroy();
});
