'use strict';

// Main runtime coordinator worker (keeps the browser DOM thread completely free).
(function initPlayGta6Loader() {
if (typeof importScripts === 'function') {
  importScripts('./workers/userdata.js');
  importScripts('./workers/httpfs_mailbox.js');
}

const progress =
  typeof self !== 'undefined' && typeof BroadcastChannel === 'function'
    ? new BroadcastChannel('game-progress')
    : null;

const userdataApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6Userdata) ||
  (typeof require === 'function' ? require('./workers/userdata.js') : null);
const httpfsMailboxApi =
  (typeof globalThis !== 'undefined' && globalThis.PlayGta6HttpfsMailbox) ||
  (typeof require === 'function' ? require('./workers/httpfs_mailbox.js') : null);

const userdataStore = userdataApi ? new userdataApi.UserdataStore() : null;

const utf8Decoder = new TextDecoder('utf-8', { fatal: false });

let gpuWorker = null;
let ioWorker = null;
let audioWorker = null;
let inputSab = null;
let httpfsMailboxSab = null;
let httpfsClient = null;

function report(label) {
  if (progress) {
    progress.postMessage({ label });
  }
}

function ptrToNumber(value, label = 'pointer') {
  if (typeof value === 'bigint') {
    if (value < 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
      throw new RangeError(`${label} out of addressable range: ${value}`);
    }
    return Number(value);
  }
  if (typeof value === 'number') {
    if (!Number.isFinite(value) || value < 0 || !Number.isInteger(value)) {
      throw new RangeError(`${label} must be a non-negative integer: ${value}`);
    }
    return value;
  }
  throw new TypeError(`${label} must be a BigInt or Number`);
}

function resolveBuffer(memoryOrBuffer) {
  if (!memoryOrBuffer) return null;
  if (memoryOrBuffer.buffer) return memoryOrBuffer.buffer;
  if (memoryOrBuffer instanceof ArrayBuffer) return memoryOrBuffer;
  if (typeof SharedArrayBuffer !== 'undefined' && memoryOrBuffer instanceof SharedArrayBuffer) {
    return memoryOrBuffer;
  }
  return null;
}

function copyMemoryBytes(buf, ptr, len) {
  const byteOffset = ptrToNumber(ptr, 'ptr');
  const byteLength = ptrToNumber(len, 'len');
  if (byteOffset + byteLength > buf.byteLength) {
    throw new RangeError(
      `Wasm memory read out of bounds: [${byteOffset}..${byteOffset + byteLength}) > ${buf.byteLength}`,
    );
  }
  const copy = new Uint8Array(byteLength);
  if (byteLength > 0) {
    copy.set(new Uint8Array(buf, byteOffset, byteLength));
  }
  return copy;
}

// Builds Wasm64/Wasm32 `env` host import functions matching docs/wasm_abi.md and runtime/os/os_shim.cpp.
function createHostImports(options = {}) {
  let currentMemory = options.memory || null;
  const getMemory = typeof options.getMemory === 'function' ? options.getMemory : null;
  const logFn = typeof options.log === 'function' ? options.log : (msg) => report(`[wasm] ${msg}`);
  const wgpuSubmitFn = typeof options.wgpuSubmit === 'function' ? options.wgpuSubmit : null;
  const fsClient = options.httpfsClient || null;
  const store = options.userdataStore || null;

  function activeBuffer() {
    const raw = getMemory ? getMemory() || currentMemory : currentMemory;
    return resolveBuffer(raw);
  }

  function host_log(ptr, len) {
    const buf = activeBuffer();
    if (!buf) return;
    const bytes = copyMemoryBytes(buf, ptr, len);
    logFn(utf8Decoder.decode(bytes));
  }

  function host_wgpu_submit(ptr, len) {
    const buf = activeBuffer();
    if (!buf) return;
    const bytes = copyMemoryBytes(buf, ptr, len);
    if (wgpuSubmitFn) {
      wgpuSubmitFn(bytes);
    }
  }

  function host_httpfs_open(ptr, len) {
    if (!fsClient || typeof fsClient.open !== 'function') {
      return -1;
    }
    const buf = activeBuffer();
    if (!buf) return -1;
    try {
      const bytes = copyMemoryBytes(buf, ptr, len);
      const relPath = utf8Decoder.decode(bytes);
      const handle = fsClient.open(relPath);
      return Number.isInteger(handle) ? handle | 0 : -1;
    } catch {
      return -1;
    }
  }

  function host_httpfs_read(handle, offset, dstPtr, len) {
    if (!fsClient || typeof fsClient.read !== 'function') {
      return -1n;
    }
    const buf = activeBuffer();
    if (!buf) return -1n;
    try {
      const h = Number(handle) | 0;
      if (h < 0) return -1n;
      const dstOffset = ptrToNumber(dstPtr, 'dstPtr');
      const byteLength = ptrToNumber(len, 'len');
      if (dstOffset + byteLength > buf.byteLength) {
        return -1n;
      }
      if (byteLength === 0) {
        return 0n;
      }
      const dstView = new Uint8Array(buf, dstOffset, byteLength);
      const n = fsClient.read(h, offset, dstView, byteLength);
      if (typeof n === 'bigint') {
        return n;
      }
      return Number.isFinite(n) ? BigInt(Math.trunc(n)) : -1n;
    } catch {
      return -1n;
    }
  }

  function host_httpfs_close(handle) {
    if (!fsClient || typeof fsClient.close !== 'function') {
      return;
    }
    try {
      fsClient.close(Number(handle) | 0);
    } catch {
      // Close errors are non-fatal for void ABI.
    }
  }

  const env = {
    host_log,
    host_wgpu_submit,
    host_httpfs_open,
    host_httpfs_read,
    host_httpfs_close,
  };

  const userdata = {
    restore: (fsAdapter, opts) => {
      if (!store) {
        return Promise.resolve({
          restored: 0,
          skipped: 0,
          timedOut: false,
          error: 'userdata store unavailable',
        });
      }
      return store.restoreAll(fsAdapter, opts);
    },
    save: (path, bytes, mtime) => {
      if (!store) return Promise.resolve(null);
      return store.saveFile(path, bytes, mtime).catch((err) => {
        report(`userdata save warning (${path}): ${err}`);
        return null;
      });
    },
    delete: (path) => {
      if (!store) return Promise.resolve(null);
      return store.deleteFile(path).catch((err) => {
        report(`userdata delete warning (${path}): ${err}`);
        return null;
      });
    },
  };

  return {
    env,
    userdata,
    setMemory(mem) {
      currentMemory = mem || null;
    },
  };
}

function encodeTestFramePacket(r, g, b, a) {
  // Matches runtime/abi/playgta6_abi.h:
  // BeginFrame (8B) + ClearColor (8B + 16B) + Present (8B) = 40 bytes
  const buf = new ArrayBuffer(40);
  const view = new DataView(buf);
  // BeginFrame (opcode 1, payload 0)
  view.setUint16(0, 1, true);
  view.setUint16(2, 0, true);
  view.setUint32(4, 0, true);
  // ClearColor (opcode 2, payload 16)
  view.setUint16(8, 2, true);
  view.setUint16(10, 0, true);
  view.setUint32(12, 16, true);
  view.setFloat32(16, r, true);
  view.setFloat32(20, g, true);
  view.setFloat32(24, b, true);
  view.setFloat32(28, a, true);
  // Present (opcode 7, payload 0)
  view.setUint16(32, 7, true);
  view.setUint16(34, 0, true);
  view.setUint32(36, 0, true);
  return buf;
}

function createRuntimeModuleOptions(overrides = {}) {
  let boundModule = null;
  const activeHttpfsClient =
    overrides.httpfsClient !== undefined ? overrides.httpfsClient : httpfsClient;
  const activeUserdataStore =
    overrides.userdataStore !== undefined ? overrides.userdataStore : userdataStore;
  const activeGpuWorker = overrides.gpuWorker !== undefined ? overrides.gpuWorker : gpuWorker;

  const wgpuSubmit =
    typeof overrides.wgpuSubmit === 'function'
      ? overrides.wgpuSubmit
      : (bytes) => {
          if (!activeGpuWorker) return;
          const copy = bytes.slice().buffer;
          activeGpuWorker.postMessage({ cmdBuffer: copy }, [copy]);
        };

  const logFn =
    typeof overrides.log === 'function' ? overrides.log : (msg) => report(`[wasm] ${msg}`);

  const hostImports = createHostImports({
    memory: overrides.memory || null,
    getMemory: () =>
      overrides.memory ||
      (boundModule && (boundModule.wasmMemory || (boundModule.HEAPU8 && boundModule.HEAPU8.buffer))) ||
      null,
    log: logFn,
    wgpuSubmit,
    httpfsClient: activeHttpfsClient,
    userdataStore: activeUserdataStore,
  });

  const preRun = [
    (mod) => {
      boundModule = mod || null;
      if (mod && (mod.wasmMemory || (mod.HEAPU8 && mod.HEAPU8.buffer))) {
        hostImports.setMemory(mod.wasmMemory || mod.HEAPU8.buffer);
      }
      if (!activeUserdataStore || !mod || !mod.FS) return;
      const useRunDep =
        typeof mod.addRunDependency === 'function' &&
        typeof mod.removeRunDependency === 'function';
      if (useRunDep) {
        mod.addRunDependency('userdata-restore');
      }
      activeUserdataStore
        .restoreAll(mod.FS, { timeoutMs: 8000 })
        .then((res) => {
          if (res.timedOut) {
            report(`userdata restore timed out (continuing boot): ${res.error}`);
          } else if (res.error) {
            report(`userdata restore skipped: ${res.error}`);
          } else {
            report(`userdata restored (${res.restored} files)`);
          }
        })
        .catch((err) => {
          report(`userdata restore warning: ${err}`);
        })
        .finally(() => {
          if (useRunDep) {
            mod.removeRunDependency('userdata-restore');
          }
        });
    },
  ];

  return {
    locateFile: (p) => `./dist/${p}`,
    preRun,
    env: hostImports.env,
    hostImports,
    httpfsMailboxSab:
      overrides.httpfsMailboxSab !== undefined ? overrides.httpfsMailboxSab : httpfsMailboxSab,
    inputSab: overrides.inputSab !== undefined ? overrides.inputSab : inputSab,
    hostCallbacks: {
      log: logFn,
      wgpuSubmit,
      httpfsOpen: (relPath) => (activeHttpfsClient ? activeHttpfsClient.open(relPath) : -1),
      httpfsRead: (handle, offset, dst, len) =>
        activeHttpfsClient ? activeHttpfsClient.read(handle, offset, dst, len) : -1,
      httpfsClose: (handle) => (activeHttpfsClient ? activeHttpfsClient.close(handle) : false),
      userdataRestore: hostImports.userdata.restore,
      userdataSave: hostImports.userdata.save,
      userdataDelete: hostImports.userdata.delete,
    },
  };
}

async function tryBootWasmRuntime() {
  try {
    const head = await fetch('./dist/runtime.js', { method: 'HEAD' });
    if (!head.ok) {
      report('No web/dist/runtime.js built yet (running WebGPU command-ring self-test).');
      const pkt = encodeTestFramePacket(0.08, 0.14, 0.24, 1.0);
      gpuWorker.postMessage({ cmdBuffer: pkt }, [pkt]);
      return;
    }
    report('Loading Emscripten Wasm64 runtime (dist/runtime.js)...');
    importScripts('./dist/runtime.js');
    if (typeof self.createRuntimeModule === 'function') {
      await self.createRuntimeModule(createRuntimeModuleOptions());
    }
  } catch (err) {
    if (progress) {
      progress.postMessage({ error: `runtime boot: ${err}` });
    }
  }
}

if (typeof self !== 'undefined') {
  self.onmessage = (ev) => {
    const msg = ev.data || {};
    if (msg.resize && gpuWorker) {
      gpuWorker.postMessage({ resize: true, width: msg.width, height: msg.height });
      return;
    }

    if (!msg.canvas) return;
    inputSab = msg.inputSab || null;

    if (
      typeof SharedArrayBuffer !== 'undefined' &&
      httpfsMailboxApi &&
      typeof httpfsMailboxApi.createHttpfsMailboxBuffer === 'function'
    ) {
      httpfsMailboxSab = httpfsMailboxApi.createHttpfsMailboxBuffer();
      httpfsClient = new httpfsMailboxApi.HttpfsMailboxClient(httpfsMailboxSab);
    } else {
      httpfsMailboxSab = null;
      httpfsClient = null;
    }

    ioWorker = new Worker('./workers/io_worker.js');
    gpuWorker = new Worker('./workers/wgpu_worker.js');
    audioWorker = new Worker('./workers/audio_worker.js');

    let readyCount = 0;
    const onWorkerReady = (name, detail) => {
      report(`${name} ready${detail ? ` (${detail})` : ''}`);
      if (++readyCount === 3) {
        tryBootWasmRuntime();
      }
    };

    ioWorker.onmessage = (e) => {
      if (e.data.ready) onWorkerReady('io_worker', e.data.summary);
      else if (e.data.error && progress) progress.postMessage({ error: `io_worker: ${e.data.error}` });
    };
    gpuWorker.onmessage = (e) => {
      if (e.data.ready) onWorkerReady('wgpu_worker', e.data.adapter);
      else if (e.data.frame) report(`WebGPU frame #${e.data.frame} presented`);
      else if (e.data.error && progress) progress.postMessage({ error: `wgpu_worker: ${e.data.error}` });
    };
    audioWorker.onmessage = (e) => {
      if (e.data.ready) onWorkerReady('audio_worker');
    };

    ioWorker.postMessage({ init: true, dataBase: msg.dataBase });
    if (httpfsMailboxSab) {
      ioWorker.postMessage({ attachMailbox: true, sab: httpfsMailboxSab });
    }
    gpuWorker.postMessage(
      { init: true, canvas: msg.canvas, width: msg.width, height: msg.height },
      [msg.canvas],
    );
    audioWorker.postMessage({ init: true });
  };
}

const exportedApi = {
  createHostImports,
  createRuntimeModuleOptions,
  encodeTestFramePacket,
  ptrToNumber,
};

if (typeof globalThis !== 'undefined') {
  globalThis.PlayGta6Loader = exportedApi;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = exportedApi;
}
})();
