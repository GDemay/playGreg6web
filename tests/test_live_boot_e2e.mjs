import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { bootstrapPlayGta6App, PlayGta6AppHandle } from '../web/app_bootstrap.js';
import { HudController, HUD_ACTIONS, DEFAULT_FILENAMES } from '../web/ui/hud_controller.js';
import { SESSION_MODE, SESSION_STATE } from '../web/runtime/session_controller.js';
import { SmokeDemoRunner } from '../web/demo/run_smoke_demo.js';
import {
  validateDataManifest,
  validateBootset,
  computeStreamingReadiness,
  verifyLiveEndpoints,
} from '../web/storage/manifest_verifier.js';
import {
  encodeBase64,
  SNAPSHOT_FORMAT,
  SNAPSHOT_VERSION,
} from '../web/storage/storage_manager.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

const REQUIRED_INDEX_HTML_IDS = Object.freeze([
  // Viewport & HUD frame
  'viewport',
  'hud',
  'hud-status-dot',
  'hud-status',
  'hud-toggle',
  'hud-body',
  'hud-refresh-status',
  'hud-clear-log',
  'hud-log',
  // Capability badges
  'cap-coi',
  'cap-webgpu',
  'cap-wasm64',
  'cap-opfs',
  'cap-gamepad',
  // Server readiness cards
  'srv-manifest',
  'srv-bootset',
  'srv-wasm',
  'srv-shaders',
  // Telemetry cards
  'metric-fps',
  'metric-gpu-packets',
  'metric-opfs-hits',
  'metric-pointer-lock',
  'metric-gamepad',
  // Controls & Diagnostics (W30)
  'btn-smoke-demo',
  'btn-restart-session',
  'btn-record-trace',
  'btn-export-trace',
  'btn-clear-cache',
  'btn-export-saves',
  'btn-export-crash',
  'select-input-profile',
]);

function makePatternBytes(length, seed = 0x4b) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 31) ^ ((i >> 7) * 17) ^ seed) & 0xff;
  }
  return out;
}

function sha256Hex(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

function detectPython3Command() {
  const candidates =
    process.platform === 'win32'
      ? [
          { cmd: 'python', prefixArgs: [] },
          { cmd: 'python3', prefixArgs: [] },
          { cmd: 'py', prefixArgs: ['-3'] },
        ]
      : [
          { cmd: 'python3', prefixArgs: [] },
          { cmd: 'python', prefixArgs: [] },
        ];

  for (const candidate of candidates) {
    const probe = spawnSync(candidate.cmd, [...candidate.prefixArgs, '--version'], {
      stdio: 'pipe',
      encoding: 'utf8',
      timeout: 5000,
      windowsHide: true,
    });
    if (probe.error || probe.status !== 0) {
      continue;
    }
    const output = `${probe.stdout || ''}\n${probe.stderr || ''}`;
    if (/Python\s+3\./i.test(output)) {
      return candidate;
    }
  }

  throw new Error('Could not find a working Python 3 executable to launch server/serve_local.py');
}

function allocateEphemeralPort() {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.unref();
    srv.on('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address();
      const port = typeof addr === 'object' && addr ? addr.port : 0;
      srv.close((err) => {
        if (err) reject(err);
        else resolve(port);
      });
    });
  });
}

async function waitForServerReady(baseUrl, child, getLogs, timeoutMs = 10000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (child.exitCode !== null || child.signalCode !== null) {
      throw new Error(
        `serve_local.py exited prematurely (code=${child.exitCode}, signal=${child.signalCode}):\n${getLogs()}`,
      );
    }
    try {
      const res = await fetch(`${baseUrl}/api/status`);
      if (res.status === 200) {
        await res.arrayBuffer();
        return;
      }
    } catch {
      // Server still starting up.
    }
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error(
    `Timed out waiting for ${baseUrl}/api/status after ${timeoutMs}ms:\n${getLogs()}`,
  );
}

async function waitWithTimeout(promise, ms) {
  let timer = null;
  try {
    return await Promise.race([
      promise.then(() => true),
      new Promise((resolve) => {
        timer = setTimeout(() => resolve(false), ms);
        timer.unref?.();
      }),
    ]);
  } finally {
    if (timer !== null) {
      clearTimeout(timer);
    }
  }
}

async function stopChildProcess(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) {
    return;
  }

  const exitPromise = new Promise((resolve) => {
    child.once('exit', resolve);
  });

  try {
    child.kill();
  } catch {
    // Process already terminated.
  }

  const settled = await waitWithTimeout(exitPromise, 2000);
  if (!settled && child.exitCode === null && child.signalCode === null) {
    try {
      child.kill('SIGKILL');
    } catch {
      // Ignore kill errors during forced teardown.
    }
    await waitWithTimeout(exitPromise, 1000);
  }
}

function createClock(startMs = 10_000) {
  let current = startMs;
  return {
    now: () => current,
    advance(deltaMs) {
      current += deltaMs;
      return current;
    },
  };
}

function createFakeRaf(clock) {
  let nextId = 1;
  const scheduled = new Map();
  return {
    raf(cb) {
      const id = nextId++;
      scheduled.set(id, cb);
      return id;
    },
    cancelRaf(id) {
      scheduled.delete(id);
    },
    flushFrame(deltaMs = 16) {
      if (scheduled.size === 0) return 0;
      const batch = [...scheduled.entries()];
      scheduled.clear();
      const ts = clock.advance(deltaMs);
      for (const [, cb] of batch) {
        cb(ts);
      }
      return batch.length;
    },
    pendingCount() {
      return scheduled.size;
    },
  };
}

function createFakeTimers() {
  let nextId = 1;
  const active = new Map();
  return {
    setInterval(fn, ms) {
      const id = nextId++;
      active.set(id, { fn, ms });
      return id;
    },
    clearInterval(id) {
      active.delete(id);
    },
    tickAll() {
      for (const entry of [...active.values()]) {
        entry.fn();
      }
    },
    activeCount() {
      return active.size;
    },
  };
}

function createMockEventTarget(extra = {}) {
  const listeners = new Map();
  const attributes = new Map();
  return {
    dataset: {},
    value: '',
    textContent: '',
    children: [],
    ...extra,
    setAttribute(name, value) {
      attributes.set(name, String(value));
    },
    getAttribute(name) {
      return attributes.has(name) ? attributes.get(name) : null;
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    addEventListener(type, fn, opts) {
      if (!listeners.has(type)) {
        listeners.set(type, new Set());
      }
      listeners.get(type).add({ fn, once: Boolean(opts && opts.once) });
    },
    removeEventListener(type, fn) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of set) {
        if (entry.fn === fn) {
          set.delete(entry);
        }
      }
    },
    dispatch(type, event = {}) {
      const set = listeners.get(type);
      if (!set) return;
      for (const entry of [...set]) {
        if (entry.once) {
          set.delete(entry);
        }
        entry.fn({ type, target: this, ...event });
      }
    },
    listenerCount(type) {
      if (type) {
        return listeners.get(type)?.size ?? 0;
      }
      let total = 0;
      for (const set of listeners.values()) {
        total += set.size;
      }
      return total;
    },
  };
}

function createMemoryStorage() {
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
    clear() {
      map.clear();
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
    async removeEntry(entryName) {
      entries.delete(entryName);
    },
  };
}

function createMockIndexedDB() {
  const databases = new Map();

  function makeStore(recordsMap, notifyComplete) {
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
          notifyComplete();
        });
        return req;
      },
      delete(key) {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          recordsMap.delete(key);
          if (req.onsuccess) req.onsuccess({ target: req });
          notifyComplete();
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
          notifyComplete();
        });
        return req;
      },
      clear() {
        const req = { result: undefined, error: null, onsuccess: null, onerror: null };
        queueMicrotask(() => {
          recordsMap.clear();
          if (req.onsuccess) req.onsuccess({ target: req });
          notifyComplete();
        });
        return req;
      },
    };
  }

  return {
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
          objectStoreNames: { contains: (name) => entry.stores.has(name) },
          createObjectStore(name) {
            if (!entry.stores.has(name)) {
              entry.stores.set(name, new Map());
            }
            return makeStore(entry.stores.get(name), () => {});
          },
          transaction(storeName) {
            if (!entry.stores.has(storeName)) {
              entry.stores.set(storeName, new Map());
            }
            const tx = {
              error: null,
              oncomplete: null,
              onerror: null,
              onabort: null,
              objectStore(name) {
                const map = entry.stores.get(name);
                return makeStore(map, () => {
                  queueMicrotask(() => {
                    if (tx.oncomplete) tx.oncomplete({ target: tx });
                  });
                });
              },
            };
            return tx;
          },
          close() {},
        };

        req.result = db;
        if (isNew && typeof req.onupgradeneeded === 'function') {
          req.onupgradeneeded({ target: req });
        }
        if (typeof req.onsuccess === 'function') {
          req.onsuccess({ target: req });
        }
      });
      return req;
    },
  };
}

function createMockHud() {
  const logs = [];
  const statuses = [];
  const metrics = [];
  return {
    logs,
    statuses,
    metrics,
    setStatus(text, level = 'info') {
      statuses.push({ text, level });
    },
    appendLog(message, level = 'info') {
      logs.push({ message, level });
    },
    updateMetrics(payload) {
      metrics.push({ ...payload });
    },
    refreshServerStatus() {},
    setCollapsed() {},
  };
}

function createMockWebGpuEnv() {
  let nextHandle = 1;
  let swapchainCounter = 0;
  const gpuCalls = [];

  const context = {
    configure(config) {
      gpuCalls.push({ op: 'context.configure', config: { ...config } });
    },
    unconfigure() {
      gpuCalls.push({ op: 'context.unconfigure' });
    },
    getCurrentTexture() {
      const id = `swapchain-view-${swapchainCounter++}`;
      return {
        createView() {
          return { kind: 'GPUTextureView', id };
        },
      };
    },
  };

  const canvas = createMockEventTarget({
    id: 'viewport',
    width: 640,
    height: 360,
    clientWidth: 640,
    clientHeight: 360,
    getContext(kind) {
      return kind === 'webgpu' ? context : null;
    },
  });

  const device = createMockEventTarget({
    limits: { maxTextureDimension2D: 8192 },
    lost: new Promise(() => {}),
    destroy() {
      gpuCalls.push({ op: 'device.destroy' });
    },
    createBuffer(desc) {
      const buf = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
        },
      };
      gpuCalls.push({ op: 'createBuffer', size: desc.size });
      return buf;
    },
    createTexture(desc) {
      const handle = nextHandle++;
      const tex = {
        kind: 'GPUTexture',
        handle,
        desc: { ...desc, size: { ...desc.size } },
        destroyed: false,
        createView() {
          return { kind: 'GPUTextureView', textureHandle: handle, handle: nextHandle++ };
        },
        destroy() {
          this.destroyed = true;
        },
      };
      gpuCalls.push({ op: 'createTexture', format: desc.format });
      return tex;
    },
    createSampler(desc) {
      gpuCalls.push({ op: 'createSampler' });
      return { kind: 'GPUSampler', handle: nextHandle++, desc: { ...desc } };
    },
    createShaderModule(desc) {
      gpuCalls.push({ op: 'createShaderModule' });
      return { kind: 'GPUShaderModule', handle: nextHandle++, code: desc.code };
    },
    createRenderPipeline(desc) {
      gpuCalls.push({ op: 'createRenderPipeline' });
      return {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index = 0) {
          return { kind: 'GPUBindGroupLayout', pipelineKind: 'render', index, handle: nextHandle++ };
        },
      };
    },
    createComputePipeline(desc) {
      gpuCalls.push({ op: 'createComputePipeline' });
      return {
        kind: 'GPUComputePipeline',
        handle: nextHandle++,
        desc,
        getBindGroupLayout(index = 0) {
          return {
            kind: 'GPUBindGroupLayout',
            pipelineKind: 'compute',
            index,
            handle: nextHandle++,
          };
        },
      };
    },
    createBindGroup(desc) {
      gpuCalls.push({ op: 'createBindGroup', entryCount: desc?.entries?.length ?? 0 });
      return { kind: 'GPUBindGroup', handle: nextHandle++, desc };
    },
    createCommandEncoder() {
      return {
        beginRenderPass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            setVertexBuffer() {},
            setIndexBuffer() {},
            setViewport() {},
            setScissorRect() {},
            draw() {
              gpuCalls.push({ op: 'draw' });
            },
            drawIndexed(indexCount) {
              gpuCalls.push({ op: 'drawIndexed', indexCount });
            },
            end() {},
          };
        },
        beginComputePass() {
          return {
            setPipeline() {},
            setBindGroup() {},
            dispatchWorkgroups(x, y, z) {
              gpuCalls.push({ op: 'dispatchWorkgroups', x, y, z });
            },
            end() {},
          };
        },
        copyBufferToBuffer() {},
        finish() {
          return { kind: 'GPUCommandBuffer', handle: nextHandle++ };
        },
      };
    },
    queue: {
      writeBuffer() {},
      writeTexture() {},
      submit() {
        gpuCalls.push({ op: 'queue.submit' });
      },
    },
  });

  const adapter = {
    async requestDevice() {
      return device;
    },
  };

  return { canvas, device, adapter, context, gpuCalls };
}

function createMockIndexDom(canvas) {
  const elements = {};
  for (const id of REQUIRED_INDEX_HTML_IDS) {
    if (id === 'viewport') {
      elements[id] = canvas;
    } else if (id === 'select-input-profile') {
      elements[id] = createMockEventTarget({ id, value: 'default' });
    } else {
      elements[id] = createMockEventTarget({ id });
    }
  }
  return elements;
}

function createLiveServerFetch(baseUrl) {
  const requests = [];
  const fetchImpl = async (input, init) => {
    const rawUrl = typeof input === 'string' ? input : input?.url || String(input);
    const resolvedUrl = /^https?:\/\//i.test(rawUrl) ? rawUrl : new URL(rawUrl, baseUrl).href;
    const res = await fetch(resolvedUrl, init);
    requests.push({
      requestedUrl: rawUrl,
      resolvedUrl,
      method: (init && init.method) || 'GET',
      status: res.status,
    });
    return res;
  };
  fetchImpl.requests = requests;
  return fetchImpl;
}

function runGpuTraceCli(pythonCmd, tracePath) {
  const scriptPath = path.join(REPO_ROOT, 'tools', 'gpu_trace.py');
  const proc = spawnSync(
    pythonCmd.cmd,
    [...pythonCmd.prefixArgs, scriptPath, tracePath, '--json', '--strict'],
    {
      encoding: 'utf8',
      timeout: 15000,
      windowsHide: true,
    },
  );
  if (proc.error) {
    throw proc.error;
  }
  let parsed = null;
  if (proc.stdout && proc.stdout.trim().length > 0) {
    parsed = JSON.parse(proc.stdout);
  }
  return {
    status: proc.status,
    stdout: proc.stdout || '',
    stderr: proc.stderr || '',
    json: parsed,
  };
}

async function flushMicrotasks(rounds = 12) {
  for (let i = 0; i < rounds; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

async function waitForCondition(predicate, timeoutMs = 3000, stepMs = 10) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) {
      return true;
    }
    await new Promise((resolve) => setTimeout(resolve, stepMs));
  }
  return Boolean(predicate());
}

describe('W33: headless live boot, HUD, and GPU trace E2E verification', () => {
  let tempRootDir = null;
  let tempDataDir = null;
  let tempTraceDir = null;
  let serverProc = null;
  let baseUrl = '';
  let serverLogs = '';
  let exitGuard = null;
  let pythonCmd = null;
  let servedIndexHtml = '';

  const PAYLOAD_A_REL = 'archives/boot_chunk.bin';
  const PAYLOAD_B_REL = 'archives/world_stream.bin';
  const PAYLOAD_A_BYTES = makePatternBytes(16 * 1024, 0x3c);
  const PAYLOAD_B_BYTES = makePatternBytes(8 * 1024 + 123, 0x79);

  before(async () => {
    tempRootDir = fs.mkdtempSync(path.join(os.tmpdir(), 'playgta6web-live-boot-e2e-'));
    tempDataDir = path.join(tempRootDir, 'data');
    tempTraceDir = path.join(tempRootDir, 'traces');
    fs.mkdirSync(path.join(tempDataDir, 'archives'), { recursive: true });
    fs.mkdirSync(tempTraceDir, { recursive: true });

    // Write deterministic synthetic payload files.
    fs.writeFileSync(path.join(tempDataDir, PAYLOAD_A_REL), PAYLOAD_A_BYTES);
    fs.writeFileSync(path.join(tempDataDir, PAYLOAD_B_REL), PAYLOAD_B_BYTES);

    // Write canonical data-manifest.json (abi_version: 1, exact totals and SHA-256 digests).
    const manifest = {
      abi_version: 1,
      file_count: 2,
      total_bytes: PAYLOAD_A_BYTES.byteLength + PAYLOAD_B_BYTES.byteLength,
      files: [
        {
          path: PAYLOAD_A_REL,
          size: PAYLOAD_A_BYTES.byteLength,
          sha256: sha256Hex(PAYLOAD_A_BYTES),
          status: 'complete',
          incomplete: false,
        },
        {
          path: PAYLOAD_B_REL,
          size: PAYLOAD_B_BYTES.byteLength,
          sha256: sha256Hex(PAYLOAD_B_BYTES),
          status: 'complete',
          incomplete: false,
        },
      ],
    };

    // Write canonical bootset.json (non-overlapping in-bounds ranges).
    const bootset = [
      { path: PAYLOAD_A_REL, offset: 0, length: 4096 },
      { path: PAYLOAD_B_REL, offset: 0, length: 2048 },
    ];

    fs.writeFileSync(
      path.join(tempDataDir, 'data-manifest.json'),
      JSON.stringify(manifest, null, 2),
      'utf8',
    );
    fs.writeFileSync(
      path.join(tempDataDir, 'bootset.json'),
      JSON.stringify(bootset, null, 2),
      'utf8',
    );

    pythonCmd = detectPython3Command();
    const port = await allocateEphemeralPort();
    baseUrl = `http://127.0.0.1:${port}`;

    const serverScript = path.join(REPO_ROOT, 'server', 'serve_local.py');
    const webRoot = path.join(REPO_ROOT, 'web');

    serverProc = spawn(
      pythonCmd.cmd,
      [
        ...pythonCmd.prefixArgs,
        serverScript,
        '--host',
        '127.0.0.1',
        '--port',
        String(port),
        '--web',
        webRoot,
        '--data',
        tempDataDir,
      ],
      {
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true,
      },
    );

    const appendLog = (chunk) => {
      if (serverLogs.length < 16384) {
        serverLogs += chunk.toString('utf8');
      }
    };
    serverProc.stdout?.on('data', appendLog);
    serverProc.stderr?.on('data', appendLog);

    exitGuard = () => {
      if (serverProc && serverProc.exitCode === null && serverProc.signalCode === null) {
        try {
          serverProc.kill();
        } catch {
          // Best-effort cleanup on unexpected process exit.
        }
      }
    };
    process.once('exit', exitGuard);

    await waitForServerReady(baseUrl, serverProc, () => serverLogs);
  });

  after(async () => {
    if (exitGuard) {
      process.removeListener('exit', exitGuard);
      exitGuard = null;
    }
    try {
      await stopChildProcess(serverProc);
    } finally {
      serverProc = null;
      if (tempRootDir) {
        fs.rmSync(tempRootDir, { recursive: true, force: true });
        tempRootDir = null;
        tempDataDir = null;
        tempTraceDir = null;
      }
    }
  });

  it('1. serves /, /api/status, and core ES modules with COOP/COEP headers and sanitized status JSON', async () => {
    const endpoints = [
      { urlPath: '/', contentTypeRe: /^text\/html/i },
      { urlPath: '/api/status', contentTypeRe: /^application\/json/i },
      { urlPath: '/app_bootstrap.js', contentTypeRe: /javascript/i },
      { urlPath: '/ui/hud_controller.js', contentTypeRe: /javascript/i },
      { urlPath: '/storage/manifest_verifier.js', contentTypeRe: /javascript/i },
    ];

    const bodies = new Map();

    for (const ep of endpoints) {
      const res = await fetch(`${baseUrl}${ep.urlPath}`);
      assert.equal(res.status, 200, `Expected 200 OK for ${ep.urlPath}`);
      assert.equal(
        res.headers.get('cross-origin-opener-policy'),
        'same-origin',
        `Missing COOP header on ${ep.urlPath}`,
      );
      assert.equal(
        res.headers.get('cross-origin-embedder-policy'),
        'require-corp',
        `Missing COEP header on ${ep.urlPath}`,
      );
      assert.equal(
        res.headers.get('cross-origin-resource-policy'),
        'same-origin',
        `Missing CORP header on ${ep.urlPath}`,
      );
      assert.equal(
        res.headers.get('accept-ranges'),
        'bytes',
        `Missing Accept-Ranges: bytes on ${ep.urlPath}`,
      );
      assert.match(
        res.headers.get('content-type') || '',
        ep.contentTypeRe,
        `Unexpected Content-Type for ${ep.urlPath}`,
      );

      const text = await res.text();
      assert.ok(text.length > 0, `Expected non-empty response body for ${ep.urlPath}`);
      bodies.set(ep.urlPath, text);
    }

    // Verify served web/index.html contains all expected HUD & Controls IDs.
    servedIndexHtml = bodies.get('/');
    assert.match(servedIndexHtml, /<!doctype html>/i);
    for (const id of REQUIRED_INDEX_HTML_IDS) {
      assert.match(
        servedIndexHtml,
        new RegExp(`id="${id}"`),
        `Live-served / (web/index.html) must declare id="${id}"`,
      );
    }

    // Verify served ES modules contain expected exports.
    assert.match(bodies.get('/app_bootstrap.js'), /export function bootstrapPlayGta6App/);
    assert.match(bodies.get('/ui/hud_controller.js'), /export class HudController/);
    assert.match(bodies.get('/storage/manifest_verifier.js'), /export async function verifyLiveEndpoints/);

    // Verify sanitized /api/status JSON payload and zero host filesystem path leaks.
    const rawStatus = bodies.get('/api/status');
    const status = JSON.parse(rawStatus);
    assert.equal(status.ok, true);
    assert.equal(status.coopCoep, true);
    assert.equal(status.dataDirConfigured, true);
    assert.equal(status.hasManifest, true);
    assert.equal(status.hasBootset, true);
    assert.equal(status.hasWasmBinary, false);
    assert.equal(typeof status.shaderPackCount, 'number');
    assert.ok(Number.isInteger(status.shaderPackCount) && status.shaderPackCount >= 0);

    assert.equal(rawStatus.includes(tempRootDir), false, '/api/status must not leak tempRootDir');
    assert.equal(rawStatus.includes(tempDataDir), false, '/api/status must not leak tempDataDir');
    assert.equal(
      rawStatus.includes(path.basename(tempRootDir)),
      false,
      '/api/status must not leak temp dir basename',
    );
    assert.equal(rawStatus.includes(REPO_ROOT), false, '/api/status must not leak REPO_ROOT');
    assert.equal(rawStatus.includes(os.tmpdir()), false, '/api/status must not leak os.tmpdir()');
    assert.equal(/[A-Za-z]:\\/.test(rawStatus), false, '/api/status must not leak Windows drive paths');
    assert.equal(/\/(?:Users|home|tmp|var)\//.test(rawStatus), false, '/api/status must not leak POSIX host paths');
  });

  it('2. validates live data-manifest.json, bootset.json, 100% readiness, and 206 Range + ETag endpoints (W31)', async () => {
    const manifestRes = await fetch(`${baseUrl}/data/data-manifest.json`);
    assert.equal(manifestRes.status, 200);
    assert.equal(manifestRes.headers.get('cross-origin-opener-policy'), 'same-origin');
    assert.equal(manifestRes.headers.get('cross-origin-embedder-policy'), 'require-corp');
    const manifestJson = await manifestRes.json();

    const bootsetRes = await fetch(`${baseUrl}/data/bootset.json`);
    assert.equal(bootsetRes.status, 200);
    assert.equal(bootsetRes.headers.get('cross-origin-opener-policy'), 'same-origin');
    assert.equal(bootsetRes.headers.get('cross-origin-embedder-policy'), 'require-corp');
    const bootsetJson = await bootsetRes.json();

    const manifestValidation = validateDataManifest(manifestJson);
    assert.equal(manifestValidation.ok, true, `Manifest errors: ${manifestValidation.errors.join('; ')}`);
    assert.deepEqual(manifestValidation.errors, []);
    assert.deepEqual(manifestValidation.warnings, []);
    assert.equal(manifestValidation.summary.fileCount, 2);
    assert.equal(manifestValidation.summary.completeFiles, 2);
    assert.equal(manifestValidation.summary.incompleteFiles, 0);

    const bootsetValidation = validateBootset(bootsetJson, manifestValidation);
    assert.equal(bootsetValidation.ok, true, `Bootset errors: ${bootsetValidation.errors.join('; ')}`);
    assert.deepEqual(bootsetValidation.errors, []);
    assert.deepEqual(bootsetValidation.warnings, []);
    assert.deepEqual(bootsetValidation.missingFiles, []);
    assert.deepEqual(bootsetValidation.incompleteFiles, []);
    assert.equal(bootsetValidation.uniqueBytes, 4096 + 2048);

    const readiness = computeStreamingReadiness(manifestValidation, bootsetJson);
    assert.equal(readiness.manifestValid, true);
    assert.equal(readiness.bootsetValid, true);
    assert.equal(readiness.bootsetReady, true);
    assert.equal(readiness.readinessPct, 100);
    assert.equal(readiness.completeFiles, 2);
    assert.equal(readiness.incompleteFiles, 0);
    assert.deepEqual(readiness.issues, []);

    const probe = await verifyLiveEndpoints({
      fetchImpl: fetch,
      dataBase: `${baseUrl}/data/`,
      manifest: manifestValidation,
      probeBytes: 64,
      checkCoopCoep: true,
    });
    assert.equal(probe.ok, true, `Live endpoint probe failed: ${JSON.stringify(probe)}`);
    assert.equal(probe.probed, 2);
    assert.equal(probe.passed, 2);
    assert.equal(probe.failed, 0);
    assert.equal(probe.skippedIncomplete, 0);
    assert.deepEqual(probe.errors, []);

    for (const item of probe.results) {
      assert.equal(item.ok, true, `Probe failed for ${item.path}: ${item.errors.join('; ')}`);
      assert.equal(item.head.status, 200);
      assert.equal(item.range.status, 206);
      assert.equal(item.range.receivedBytes, 64);
      assert.equal(item.range.contentRange, `bytes 0-63/${item.expectedSize}`);
      assert.ok(
        item.etag && item.etag.startsWith('"') && item.etag.endsWith('"'),
        `Expected strong quoted ETag for ${item.path}, got ${item.etag}`,
      );
      assert.equal(item.head.etag, item.range.etag);
      assert.equal(item.coopCoep, true);
    }
  });

  it('3. boots bootstrapPlayGta6App + HudController against live server, verifies G5 .pg6trace via tools/gpu_trace.py, sanitized crash report, HUD controls, and clean teardown', async () => {
    const clock = createClock(20_000);
    const raf = createFakeRaf(clock);
    const timers = createFakeTimers();
    const hud = createMockHud();
    const localStorage = createMemoryStorage();
    const indexedDB = createMockIndexedDB();
    const cacheDirHandle = createMockOpfsDirHandle('playgta6-cache', {
      'chunk_0.bin': makePatternBytes(512, 0x11),
      'chunk_1.bin': makePatternBytes(1024, 0x22),
    });
    const opfsRoot = createMockOpfsDirHandle('', {
      'playgta6-cache': cacheDirHandle,
    });
    const webgpu = createMockWebGpuEnv();
    const mockDom = createMockIndexDom(webgpu.canvas);
    const liveFetch = createLiveServerFetch(baseUrl);
    const downloads = [];

    // Confirm every mocked element ID matches the live-served web/index.html.
    if (!servedIndexHtml) {
      servedIndexHtml = await (await fetch(`${baseUrl}/`)).text();
    }
    for (const id of Object.keys(mockDom)) {
      assert.match(servedIndexHtml, new RegExp(`id="${id}"`), `Expected id="${id}" in live web/index.html`);
    }

    const windowObj = createMockEventTarget({
      crossOriginIsolated: true,
      SharedArrayBuffer,
      devicePixelRatio: 1,
      innerWidth: 640,
      innerHeight: 360,
      location: { origin: baseUrl },
      localStorage,
      indexedDB,
      requestAnimationFrame: raf.raf,
      cancelAnimationFrame: raf.cancelRaf,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });

    const documentObj = createMockEventTarget({
      pointerLockElement: null,
      hidden: false,
      visibilityState: 'visible',
      getElementById(id) {
        return mockDom[id] || null;
      },
      createElement(tag) {
        return createMockEventTarget({ tagName: tag.toUpperCase() });
      },
    });

    const navigatorObj = {
      gpu: {
        requestAdapter: async () => webgpu.adapter,
        getPreferredCanvasFormat: () => 'bgra8unorm',
      },
      storage: {
        getDirectory: async () => opfsRoot,
        estimate: async () => ({ usage: 1536, quota: 64 * 1024 * 1024 }),
        persisted: async () => true,
      },
      getGamepads: () => [],
    };

    const app = bootstrapPlayGta6App({
      windowObj,
      documentObj,
      navigatorObj,
      fetchImpl: liveFetch,
      canvas: webgpu.canvas,
      hud,
      dataBase: `${baseUrl}/data/`,
      now: clock.now,
      epochNow: clock.now,
      raf: raf.raf,
      cancelRaf: raf.cancelRaf,
      setInterval: timers.setInterval,
      clearInterval: timers.clearInterval,
    });

    // Dynamic SmokeDemoRunner proxy matching web/index.html (W30).
    const smokeRunnerProxy = {
      get recorder() {
        return app.getSmokeRunner()?.recorder || null;
      },
      get running() {
        return Boolean(app.getSmokeRunner()?.running);
      },
      start() {
        const runner = app.getSmokeRunner();
        return runner ? runner.start() : false;
      },
      stop() {
        const runner = app.getSmokeRunner();
        return runner ? runner.stop() : false;
      },
      startRecording() {
        const runner = app.getSmokeRunner();
        if (!runner) {
          throw new Error('Smoke demo runner is not active');
        }
        return runner.startRecording();
      },
      stopRecording() {
        const runner = app.getSmokeRunner();
        if (!runner) {
          throw new Error('Smoke demo runner is not active');
        }
        return runner.stopRecording();
      },
      exportTraceBytes(opts) {
        const runner = app.getSmokeRunner();
        return runner ? runner.exportTraceBytes(opts) : new Uint8Array(0);
      },
      exportTraceMetadata() {
        const runner = app.getSmokeRunner();
        return runner ? runner.exportTraceMetadata() : null;
      },
    };

    const hudController = new HudController({
      sessionController: app.session,
      smokeDemoRunner: smokeRunnerProxy,
      storageManager: app.storageManager,
      crashReporter: app.crashReporter,
      perfMonitor: app.perfMonitor,
      inputProfiles: app.inputProfiles,
      hud,
      downloadBlob(filename, data, mimeType) {
        downloads.push({ filename, data, mimeType });
      },
      now: clock.now,
    });

    try {
      assert.ok(app instanceof PlayGta6AppHandle);

      // Seed one /userdata/ save file into IndexedDB so #btn-export-saves exports real data.
      const sampleSavePayload = new TextEncoder().encode('PLAYGTA6-SAVE-SLOT-0');
      const importRes = await app.storageManager.importUserdataSnapshot({
        format: SNAPSHOT_FORMAT,
        version: SNAPSHOT_VERSION,
        files: [
          {
            path: '/userdata/saves/slot0.sav',
            encoding: 'base64',
            dataBase64: encodeBase64(sampleSavePayload),
            updatedAt: clock.now(),
          },
        ],
      });
      assert.equal(importRes.supported, true);
      assert.equal(importRes.importedCount, 1);

      // Populate #select-input-profile options from InputProfileManager.
      const profileSelect = documentObj.getElementById('select-input-profile');
      const availableProfiles = app.inputProfiles.listProfiles();
      assert.ok(availableProfiles.length >= 4, 'Expected built-in input profiles');
      for (const profile of availableProfiles) {
        const opt = documentObj.createElement('option');
        opt.value = profile.id;
        opt.textContent = profile.name;
        profileSelect.appendChild(opt);
      }
      profileSelect.value = app.inputProfiles.getActiveProfile().id;

      hudController.bindButtons({
        smokeDemoBtn: documentObj.getElementById('btn-smoke-demo'),
        restartBtn: documentObj.getElementById('btn-restart-session'),
        recordTraceBtn: documentObj.getElementById('btn-record-trace'),
        exportTraceBtn: documentObj.getElementById('btn-export-trace'),
        clearCacheBtn: documentObj.getElementById('btn-clear-cache'),
        exportSavesBtn: documentObj.getElementById('btn-export-saves'),
        exportCrashBtn: documentObj.getElementById('btn-export-crash'),
        profileSelect,
      });
      assert.equal(hudController.getState().boundControlCount, 8);

      // (a) Boot app -> live /api/status reports hasWasmBinary: false -> auto-fallback to G5 WebGPU Smoke Demo.
      const startResult = await app.start();
      assert.equal(startResult.ok, true);
      assert.equal(startResult.mode, SESSION_MODE.SMOKE);
      assert.equal(startResult.state, SESSION_STATE.RUNNING_SMOKE);
      assert.equal(app.session.getState(), SESSION_STATE.RUNNING_SMOKE);
      assert.equal(app.session.getMode(), SESSION_MODE.SMOKE);

      assert.ok(
        liveFetch.requests.some(
          (req) => req.resolvedUrl === `${baseUrl}/api/status` && req.status === 200,
        ),
        'bootstrapPlayGta6App must query live /api/status endpoint',
      );

      const smokeRunner = app.getSmokeRunner();
      assert.ok(smokeRunner instanceof SmokeDemoRunner);
      assert.equal(smokeRunner.initialized, true);
      assert.equal(smokeRunner.running, true);
      assert.ok(webgpu.gpuCalls.some((c) => c.op === 'createTexture'));
      assert.ok(webgpu.gpuCalls.some((c) => c.op === 'createBindGroup'));
      assert.ok(webgpu.gpuCalls.some((c) => c.op === 'createRenderPipeline'));
      assert.ok(webgpu.gpuCalls.some((c) => c.op === 'createComputePipeline'));

      // (b) Record 4 live frames via #btn-record-trace, export .pg6trace via #btn-export-trace,
      //     and validate with `python tools/gpu_trace.py <trace> --json --strict`.
      const RECORDED_FRAME_COUNT = 4;
      mockDom['btn-record-trace'].dispatch('click');
      await flushMicrotasks();
      assert.equal(hudController.getState().isRecording, true);
      assert.equal(mockDom['btn-record-trace'].dataset.recording, 'true');
      assert.equal(mockDom['btn-record-trace'].getAttribute('aria-pressed'), 'true');

      for (let i = 0; i < RECORDED_FRAME_COUNT; i += 1) {
        assert.ok(raf.flushFrame(16) >= 1, `Expected RAF callback for recorded frame ${i}`);
      }

      mockDom['btn-record-trace'].dispatch('click');
      await flushMicrotasks();
      assert.equal(hudController.getState().isRecording, false);
      assert.equal(mockDom['btn-record-trace'].dataset.recording, 'false');
      assert.equal(mockDom['btn-record-trace'].getAttribute('aria-pressed'), 'false');

      // Step one unrecorded frame to verify recording actually stopped.
      raf.flushFrame(16);
      assert.equal(smokeRunner.frameIndex, RECORDED_FRAME_COUNT + 1);

      mockDom['btn-export-trace'].dispatch('click');
      await flushMicrotasks();
      const traceDownload = downloads.find((d) => d.filename === DEFAULT_FILENAMES.GPU_TRACE);
      assert.ok(traceDownload, 'Clicking #btn-export-trace must download .pg6trace');
      assert.equal(traceDownload.mimeType, 'application/octet-stream');
      assert.ok(traceDownload.data instanceof Uint8Array);
      assert.ok(traceDownload.data.byteLength > 8);

      // Write exported live-frames .pg6trace to temp file and inspect with tools/gpu_trace.py.
      const recordedTracePath = path.join(tempTraceDir, 'live-recorded.pg6trace');
      fs.writeFileSync(recordedTracePath, traceDownload.data);
      const recordedCli = runGpuTraceCli(pythonCmd, recordedTracePath);
      assert.equal(
        recordedCli.status,
        0,
        `tools/gpu_trace.py failed on recorded trace (status=${recordedCli.status}):\n${recordedCli.stderr}`,
      );
      assert.ok(recordedCli.json, 'Expected valid JSON output from tools/gpu_trace.py');
      assert.equal(recordedCli.json.ok, true);
      assert.deepEqual(recordedCli.json.diagnostics, [], 'Expected zero parse errors/warnings');
      assert.equal(recordedCli.json.unknown_opcode_count, 0);
      assert.equal(recordedCli.json.consumed_bytes, recordedCli.json.total_bytes);
      assert.equal(recordedCli.json.frame_count, RECORDED_FRAME_COUNT);
      assert.equal(recordedCli.json.draw_indexed_calls, RECORDED_FRAME_COUNT);
      assert.equal(recordedCli.json.compute_dispatches, RECORDED_FRAME_COUNT);
      assert.equal(recordedCli.json.opcode_histogram.BeginFrame, RECORDED_FRAME_COUNT);
      assert.equal(recordedCli.json.opcode_histogram.DrawIndexed, RECORDED_FRAME_COUNT);
      assert.equal(recordedCli.json.opcode_histogram.DispatchCompute, RECORDED_FRAME_COUNT);
      assert.equal(recordedCli.json.opcode_histogram.Present, RECORDED_FRAME_COUNT);
      assert.equal(recordedCli.json.opcode_histogram.SetBindGroup, RECORDED_FRAME_COUNT * 2);
      for (const frame of recordedCli.json.frames) {
        assert.equal(frame.complete, true);
        assert.equal(frame.has_begin_frame, true);
      }

      // Also verify the full G5 scene stream (init setup buffer + exported recorded frames)
      // through tools/gpu_trace.py so CreateTexture and CreateBindGroup opcodes are decoded too.
      const initBuffer = smokeRunner.scene.buildInitCommandBuffer();
      const fullG5Bytes = new Uint8Array(initBuffer.byteLength + traceDownload.data.byteLength);
      fullG5Bytes.set(initBuffer, 0);
      fullG5Bytes.set(traceDownload.data, initBuffer.byteLength);

      const fullTracePath = path.join(tempTraceDir, 'live-full-g5.pg6trace');
      fs.writeFileSync(fullTracePath, fullG5Bytes);
      const fullCli = runGpuTraceCli(pythonCmd, fullTracePath);
      assert.equal(
        fullCli.status,
        0,
        `tools/gpu_trace.py failed on full G5 trace (status=${fullCli.status}):\n${fullCli.stderr}`,
      );
      assert.equal(fullCli.json.ok, true);
      assert.deepEqual(fullCli.json.diagnostics, [], 'Expected zero parse errors in full G5 trace');
      assert.equal(fullCli.json.unknown_opcode_count, 0);
      assert.equal(fullCli.json.consumed_bytes, fullCli.json.total_bytes);

      const expectedG5Opcodes = [
        'BeginFrame',
        'CreateTexture',
        'CreateBindGroup',
        'DrawIndexed',
        'DispatchCompute',
        'Present',
      ];
      for (const opName of expectedG5Opcodes) {
        assert.ok(
          (fullCli.json.opcode_histogram[opName] || 0) >= 1,
          `Expected G5 opcode "${opName}" in gpu_trace.py histogram`,
        );
      }
      assert.ok(fullCli.json.textures_created >= 1);
      assert.ok(fullCli.json.bind_groups_created >= 2);
      const completeFullFrames = fullCli.json.frames.filter((f) => f.complete && f.has_begin_frame);
      assert.equal(completeFullFrames.length, RECORDED_FRAME_COUNT);

      // (c) Inject synthetic errors with host paths, click #btn-export-crash, and verify sanitization.
      windowObj.dispatch('error', {
        error: new Error(`Simulated failure reading C:\\Users\\tester\\secret\\game.rpf and ${tempDataDir}`),
        filename: 'file:///C:/Users/tester/playgta6web/web/index.html',
        lineno: 101,
        colno: 5,
      });
      windowObj.dispatch('unhandledrejection', {
        reason: new Error('Rejected from /Users/tester/Library/cache and /home/tester/local-data/archive.rpf'),
      });
      await flushMicrotasks();

      mockDom['btn-export-crash'].dispatch('click');
      await flushMicrotasks();
      const crashDownload = downloads.find((d) => d.filename === DEFAULT_FILENAMES.CRASH_REPORT);
      assert.ok(crashDownload, 'Clicking #btn-export-crash must download diagnostic report JSON');
      assert.equal(crashDownload.mimeType, 'application/json');

      const rawReport = crashDownload.data;
      assert.equal(typeof rawReport, 'string');
      const parsedReport = JSON.parse(rawReport);
      assert.equal(typeof parsedReport.schemaVersion, 'number');
      assert.ok(parsedReport.serverStatus, 'Diagnostic report must include serverStatus');
      assert.equal(parsedReport.serverStatus.hasManifest, true);
      assert.equal(parsedReport.serverStatus.hasBootset, true);
      assert.equal(parsedReport.serverStatus.hasWasmBinary, false);
      assert.ok(Array.isArray(parsedReport.errors) && parsedReport.errors.length >= 2);

      // Strictly assert zero host path leaks in the serialized JSON report.
      assert.equal(rawReport.includes('C:\\'), false, 'Report must not leak C:\\ paths');
      assert.equal(/[A-Za-z]:\\/.test(rawReport), false, 'Report must not leak Windows drive paths');
      assert.equal(rawReport.includes('/Users/'), false, 'Report must not leak /Users/ paths');
      assert.equal(rawReport.includes('/home/'), false, 'Report must not leak /home/ paths');
      assert.equal(rawReport.includes(tempRootDir), false, 'Report must not leak tempRootDir');
      assert.equal(rawReport.includes(tempDataDir), false, 'Report must not leak tempDataDir');
      assert.equal(rawReport.includes(REPO_ROOT), false, 'Report must not leak REPO_ROOT');
      assert.ok(
        rawReport.includes('[REDACTED_PATH]'),
        'Injected host paths must be actively redacted with [REDACTED_PATH]',
      );

      // (d) Switch input profiles via #select-input-profile.
      profileSelect.value = 'fps';
      profileSelect.dispatch('change', { target: profileSelect });
      await flushMicrotasks();
      assert.equal(app.inputProfiles.getActiveProfile().id, 'fps');
      assert.equal(hudController.getState().activeProfileId, 'fps');
      assert.equal(hudController.getState().lastAction?.action, HUD_ACTIONS.SWITCH_INPUT_PROFILE);
      assert.equal(hudController.getState().lastAction?.ok, true);
      assert.ok(localStorage._map.size > 0, 'Switching input profile must persist to storage');

      profileSelect.value = 'lefty-kb';
      profileSelect.dispatch('change', { target: profileSelect });
      await flushMicrotasks();
      assert.equal(app.inputProfiles.getActiveProfile().id, 'lefty-kb');
      assert.equal(hudController.getState().activeProfileId, 'lefty-kb');

      // (e) Clear OPFS cache via #btn-clear-cache.
      assert.equal(cacheDirHandle._entries.size, 2);
      mockDom['btn-clear-cache'].dispatch('click');
      await flushMicrotasks();
      assert.equal(hudController.getState().lastAction?.action, HUD_ACTIONS.CLEAR_OPFS_CACHE);
      assert.equal(hudController.getState().lastAction?.ok, true);
      assert.equal(cacheDirHandle._entries.size, 0, 'OPFS cache directory must be empty after clear');

      // (f) Export /userdata/ saves via #btn-export-saves.
      mockDom['btn-export-saves'].dispatch('click');
      await flushMicrotasks();
      const savesDownload = downloads.find((d) => d.filename === DEFAULT_FILENAMES.USERDATA_BACKUP);
      assert.ok(savesDownload, 'Clicking #btn-export-saves must download userdata backup JSON');
      assert.equal(savesDownload.mimeType, 'application/json');
      const parsedSaves = JSON.parse(savesDownload.data);
      assert.equal(parsedSaves.format, SNAPSHOT_FORMAT);
      assert.equal(parsedSaves.supported, true);
      assert.equal(parsedSaves.fileCount, 1);
      assert.equal(parsedSaves.files[0].path, '/userdata/saves/slot0.sav');
      assert.equal(parsedSaves.files[0].byteLength, sampleSavePayload.byteLength);

      // (g) Restart session via #btn-restart-session.
      const runnerBeforeRestart = app.getSmokeRunner();
      const statusCallsBeforeRestart = liveFetch.requests.length;
      mockDom['btn-restart-session'].dispatch('click');
      const restarted = await waitForCondition(
        () =>
          hudController.getState().lastAction?.action === HUD_ACTIONS.RESTART_SESSION &&
          app.getSmokeRunner() !== null &&
          app.getSmokeRunner() !== runnerBeforeRestart,
      );
      assert.equal(restarted, true, 'Clicking #btn-restart-session must complete session restart');
      assert.equal(hudController.getState().lastAction?.ok, true);
      assert.equal(runnerBeforeRestart.destroyed, true);
      assert.equal(app.session.getState(), SESSION_STATE.RUNNING_SMOKE);
      assert.equal(app.session.getMode(), SESSION_MODE.SMOKE);
      assert.ok(
        liveFetch.requests.length > statusCallsBeforeRestart,
        'Restarting session must re-probe live /api/status',
      );
      const runnerAfterRestart = app.getSmokeRunner();
      assert.ok(runnerAfterRestart instanceof SmokeDemoRunner);
      assert.notEqual(runnerAfterRestart, runnerBeforeRestart);
      raf.flushFrame(16);
      assert.equal(runnerAfterRestart.frameIndex, 1);
    } finally {
      // (h) Clean destroy() with zero leaked timers, RAF handles, or event listeners.
      hudController.destroy();
      await app.destroy();
    }

    assert.equal(hudController.getState().destroyed, true);
    assert.equal(hudController.getState().boundControlCount, 0);
    assert.equal(app.destroyed, true);
    assert.equal(app.getSmokeRunner(), null);
    assert.equal(app.getWasmWorker(), null);
    assert.equal(app.session.getState(), SESSION_STATE.STOPPED);
    assert.equal(app.inputBridge.disposed, true);
    assert.equal(app.perfMonitor.isHudAttached, false);
    assert.equal(app.crashReporter.isGlobalHandlersAttached, false);
    assert.equal(app.audioEngine.state, 'closed');

    // Assert zero leaked timers, pending RAF callbacks, or DOM/window/WebGPU listeners.
    assert.equal(timers.activeCount(), 0, 'Expected zero leaked interval timers after destroy()');
    assert.equal(raf.pendingCount(), 0, 'Expected zero pending RAF handles after destroy()');
    assert.equal(windowObj.listenerCount(), 0, 'Expected zero leaked window listeners after destroy()');
    assert.equal(documentObj.listenerCount(), 0, 'Expected zero leaked document listeners after destroy()');
    assert.equal(webgpu.canvas.listenerCount(), 0, 'Expected zero leaked canvas listeners after destroy()');
    assert.equal(webgpu.device.listenerCount(), 0, 'Expected zero leaked GPUDevice listeners after destroy()');
    for (const [id, el] of Object.entries(mockDom)) {
      assert.equal(
        el.listenerCount(),
        0,
        `Expected zero leaked listeners on #${id} after destroy()`,
      );
    }
  });
});
