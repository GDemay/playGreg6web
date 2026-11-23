import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { Worker as NodeWorker } from 'node:worker_threads';

import {
  WGPK_VERSION,
  SHADER_STAGE as WGPK_SHADER_STAGE,
  fnv1a64,
  formatHashHex,
  buildWgslPackBuffer,
  WgslPackCache,
} from '../web/workers/wgsl_pack_loader.js';

import {
  INPUT_STATE_BYTES,
  INPUT_OFFSETS,
  INPUT_BUTTONS,
  InputBridge,
} from '../web/input/input_bridge.js';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

const { RangeCache } = require('../web/workers/range_cache.js');
const {
  createHttpfsMailboxBuffer,
  HttpfsMailboxHost,
} = require('../web/workers/httpfs_mailbox.js');
const { createHostImports } = require('../web/loader.js');
const { UserdataStore } = require('../web/workers/userdata.js');
const { GpuRingBuffer } = require('../web/workers/gpu_ring.js');
const {
  WgpuCommandProcessor,
  OPCODES,
  SHADER_STAGE: WGPU_SHADER_STAGE,
  PRIMITIVE_TOPOLOGY,
  BUFFER_USAGE,
} = require('../web/workers/wgpu_worker.js');
const {
  AudioRingBuffer,
  AudioMixerWorker,
} = require('../web/workers/audio_worker.js');

const VS_WGSL = [
  '@vertex',
  'fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {',
  '  return vec4<f32>(pos, 1.0);',
  '}',
].join('\n');

const FS_WGSL = [
  '@fragment',
  'fn fs_main() -> @location(0) vec4<f32> {',
  '  return vec4<f32>(0.25, 0.55, 0.85, 1.0);',
  '}',
].join('\n');

function makePatternBytes(length, seed = 41) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 37) ^ ((i >> 8) * 19) ^ seed) & 0xff;
  }
  return out;
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
  throw new Error(`Timed out waiting for ${baseUrl}/api/status after ${timeoutMs}ms:\n${getLogs()}`);
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

class BinaryCommandBuilder {
  constructor() {
    this.chunks = [];
  }

  _pushPacket(opcode, payloadBytes) {
    const buf = new ArrayBuffer(8 + payloadBytes.byteLength);
    const view = new DataView(buf);
    view.setUint16(0, opcode, true);
    view.setUint16(2, 0, true);
    view.setUint32(4, payloadBytes.byteLength, true);
    new Uint8Array(buf, 8).set(payloadBytes);
    this.chunks.push(new Uint8Array(buf));
  }

  beginFrame() {
    this._pushPacket(OPCODES.BEGIN_FRAME, new Uint8Array(0));
    return this;
  }

  clearColor(r, g, b, a) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setFloat32(0, r, true);
    view.setFloat32(4, g, true);
    view.setFloat32(8, b, true);
    view.setFloat32(12, a, true);
    this._pushPacket(OPCODES.CLEAR_COLOR, new Uint8Array(buf));
    return this;
  }

  createBuffer(id, usageFlags, byteSize) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, usageFlags, true);
    view.setBigUint64(8, BigInt(byteSize), true);
    this._pushPacket(OPCODES.CREATE_BUFFER, new Uint8Array(buf));
    return this;
  }

  writeBuffer(id, dstOffset, bytes) {
    const src = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = new ArrayBuffer(16 + src.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, dstOffset, true);
    view.setUint32(8, src.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(src);
    this._pushPacket(OPCODES.WRITE_BUFFER, new Uint8Array(buf));
    return this;
  }

  createShaderModule(id, stage, wgslCode) {
    const encoded = new TextEncoder().encode(wgslCode);
    const buf = new ArrayBuffer(16 + encoded.byteLength);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, stage, true);
    view.setUint32(8, encoded.byteLength, true);
    view.setUint32(12, 0, true);
    new Uint8Array(buf, 16).set(encoded);
    this._pushPacket(OPCODES.CREATE_SHADER_MODULE, new Uint8Array(buf));
    return this;
  }

  createRenderPipeline(
    id,
    vsShaderId,
    fsShaderId,
    vertexStride,
    topology = PRIMITIVE_TOPOLOGY.TRIANGLE_LIST,
    flags = 0,
  ) {
    const buf = new ArrayBuffer(24);
    const view = new DataView(buf);
    view.setUint32(0, id, true);
    view.setUint32(4, vsShaderId, true);
    view.setUint32(8, fsShaderId, true);
    view.setUint32(12, vertexStride, true);
    view.setUint32(16, topology, true);
    view.setUint32(20, flags, true);
    this._pushPacket(OPCODES.CREATE_RENDER_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setPipeline(pipelineId) {
    const buf = new ArrayBuffer(4);
    new DataView(buf).setUint32(0, pipelineId, true);
    this._pushPacket(OPCODES.SET_PIPELINE, new Uint8Array(buf));
    return this;
  }

  setVertexBuffer(slot, bufferId, offset = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, slot, true);
    view.setUint32(4, bufferId, true);
    view.setBigUint64(8, BigInt(offset), true);
    this._pushPacket(OPCODES.SET_VERTEX_BUFFER, new Uint8Array(buf));
    return this;
  }

  draw(vertexCount, instanceCount = 1, firstVertex = 0, firstInstance = 0) {
    const buf = new ArrayBuffer(16);
    const view = new DataView(buf);
    view.setUint32(0, vertexCount, true);
    view.setUint32(4, instanceCount, true);
    view.setUint32(8, firstVertex, true);
    view.setUint32(12, firstInstance, true);
    this._pushPacket(OPCODES.DRAW, new Uint8Array(buf));
    return this;
  }

  present() {
    this._pushPacket(OPCODES.PRESENT, new Uint8Array(0));
    return this;
  }

  finish() {
    const total = this.chunks.reduce((sum, c) => sum + c.byteLength, 0);
    const out = new Uint8Array(total);
    let offset = 0;
    for (const chunk of this.chunks) {
      out.set(chunk, offset);
      offset += chunk.byteLength;
    }
    return out;
  }
}

function createMockWebGpu() {
  const calls = [];
  let nextHandle = 1;
  const swapchainView = { kind: 'GPUTextureView', id: 'e2e-swapchain-view-0' };

  const ctx = {
    getCurrentTexture() {
      return {
        createView() {
          return swapchainView;
        },
      };
    },
  };

  const device = {
    createBuffer(desc) {
      const buffer = {
        kind: 'GPUBuffer',
        handle: nextHandle++,
        size: desc.size,
        usage: desc.usage,
        destroyed: false,
        destroy() {
          this.destroyed = true;
          calls.push({ op: 'buffer.destroy', buffer });
        },
      };
      calls.push({ op: 'createBuffer', desc: { ...desc }, result: buffer });
      return buffer;
    },

    createShaderModule(desc) {
      const mod = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        label: desc.label || null,
        code: desc.code,
        async getCompilationInfo() {
          return { messages: [] };
        },
      };
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: mod });
      return mod;
    },

    createRenderPipeline(desc) {
      const pipeline = {
        kind: 'GPURenderPipeline',
        handle: nextHandle++,
        desc,
      };
      calls.push({ op: 'createRenderPipeline', desc, result: pipeline });
      return pipeline;
    },

    createCommandEncoder() {
      const commandBuffer = { kind: 'GPUCommandBuffer', handle: nextHandle++ };
      return {
        beginRenderPass(desc) {
          calls.push({ op: 'beginRenderPass', desc });
          return {
            setPipeline(pipeline) {
              calls.push({ op: 'setPipeline', pipeline });
            },
            setVertexBuffer(slot, buffer, offset) {
              calls.push({ op: 'setVertexBuffer', slot, buffer, offset });
            },
            draw(vertexCount, instanceCount, firstVertex, firstInstance) {
              calls.push({
                op: 'draw',
                vertexCount,
                instanceCount,
                firstVertex,
                firstInstance,
              });
            },
            end() {
              calls.push({ op: 'renderPass.end' });
            },
          };
        },
        finish() {
          calls.push({ op: 'encoder.finish', result: commandBuffer });
          return commandBuffer;
        },
      };
    },

    queue: {
      writeBuffer(buffer, dstOffset, data) {
        calls.push({
          op: 'writeBuffer',
          buffer,
          dstOffset,
          data: new Uint8Array(data),
        });
      },
      submit(commandBuffers) {
        calls.push({ op: 'submit', commandBuffers });
      },
    },
  };

  return { device, ctx, calls, swapchainView };
}

class MockEventTarget {
  constructor() {
    this._handlers = new Map();
  }

  addEventListener(type, handler) {
    if (!this._handlers.has(type)) {
      this._handlers.set(type, new Set());
    }
    this._handlers.get(type).add(handler);
  }

  removeEventListener(type, handler) {
    const set = this._handlers.get(type);
    if (!set) return;
    set.delete(handler);
    if (set.size === 0) this._handlers.delete(type);
  }

  dispatch(type, event = {}) {
    const set = this._handlers.get(type);
    if (!set) return;
    for (const fn of [...set]) {
      fn(event);
    }
  }
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
          objectStoreNames: {
            contains: (name) => entry.stores.has(name),
          },
          createObjectStore(name) {
            if (!entry.stores.has(name)) {
              entry.stores.set(name, new Map());
            }
            return makeStore(entry.stores.get(name), () => {});
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

function createMockFsAdapter() {
  const dirs = new Set();
  const files = new Map();
  const mtimes = new Map();

  return {
    dirs,
    files,
    mtimes,
    mkdirTree(dirPath) {
      const parts = dirPath.split('/').filter(Boolean);
      let current = '';
      for (const part of parts) {
        current += `/${part}`;
        dirs.add(current);
      }
    },
    writeFile(filePath, bytes) {
      files.set(filePath, new Uint8Array(bytes));
    },
    utime(filePath, _atime, mtime) {
      mtimes.set(filePath, mtime);
    },
  };
}

describe('playgta6web headless end-to-end pipeline', () => {
  let tempDataDir = null;
  let serverProc = null;
  let baseUrl = '';
  let serverLogs = '';
  let exitGuard = null;

  const ARCHIVE_REL_PATH = 'archives/level0.bin';
  const ARCHIVE_BYTES = makePatternBytes(20 * 1024 + 333, 0x5a); // >5 chunks at 4 KiB
  const VS_HASH = fnv1a64(VS_WGSL);
  const FS_HASH = fnv1a64(FS_WGSL);
  const SHADER_PACK_REL_PATH = 'shaders/core.wgsl.pack';

  before(async () => {
    tempDataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'playgta6web-e2e-'));
    fs.mkdirSync(path.join(tempDataDir, 'archives'), { recursive: true });
    fs.mkdirSync(path.join(tempDataDir, 'shaders'), { recursive: true });

    // 1. Multi-chunk synthetic archive
    fs.writeFileSync(path.join(tempDataDir, ARCHIVE_REL_PATH), ARCHIVE_BYTES);

    // 2. WGPK shader bundle
    const packBuffer = buildWgslPackBuffer([
      { hash: VS_HASH, stage: WGPK_SHADER_STAGE.VERTEX, wgsl: VS_WGSL },
      { hash: FS_HASH, stage: WGPK_SHADER_STAGE.FRAGMENT, wgsl: FS_WGSL },
    ]);
    fs.writeFileSync(path.join(tempDataDir, SHADER_PACK_REL_PATH), new Uint8Array(packBuffer));

    // 3. data-manifest.json and bootset.json
    const manifest = {
      version: 1,
      files: [
        { path: ARCHIVE_REL_PATH, size: ARCHIVE_BYTES.byteLength },
        { path: SHADER_PACK_REL_PATH, size: packBuffer.byteLength },
      ],
    };
    const bootset = {
      version: 1,
      entries: [
        // Prefetch first 2 chunks (0..8191) of archives/level0.bin
        { path: ARCHIVE_REL_PATH, offset: 0, length: 8192 },
      ],
    };
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

    const { cmd, prefixArgs } = detectPython3Command();
    const port = await allocateEphemeralPort();
    baseUrl = `http://127.0.0.1:${port}`;

    const serverScript = path.join(REPO_ROOT, 'server', 'serve_local.py');
    const webRoot = path.join(REPO_ROOT, 'web');

    serverProc = spawn(
      cmd,
      [
        ...prefixArgs,
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
          // Best-effort cleanup on unexpected exit.
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
      if (tempDataDir) {
        fs.rmSync(tempDataDir, { recursive: true, force: true });
        tempDataDir = null;
      }
    }
  });

  it('1. serves /api/status with COOP/COEP headers, bundle metadata, zero path leaks, and RFC Range/ETag semantics', async () => {
    const statusRes = await fetch(`${baseUrl}/api/status`);
    assert.equal(statusRes.status, 200);
    assert.equal(statusRes.headers.get('cross-origin-opener-policy'), 'same-origin');
    assert.equal(statusRes.headers.get('cross-origin-embedder-policy'), 'require-corp');
    assert.equal(statusRes.headers.get('cross-origin-resource-policy'), 'same-origin');
    assert.equal(statusRes.headers.get('accept-ranges'), 'bytes');
    assert.match(statusRes.headers.get('content-type') || '', /^application\/json/i);

    const rawBody = await statusRes.text();
    const status = JSON.parse(rawBody);
    assert.equal(status.ok, true);
    assert.equal(status.coopCoep, true);
    assert.equal(status.dataDirConfigured, true);
    assert.equal(status.hasManifest, true);
    assert.equal(status.hasBootset, true);
    assert.equal(typeof status.hasWasmBinary, 'boolean');
    assert.ok(
      status.shaderPackCount >= 1,
      `Expected at least 1 .wgsl.pack bundle, got ${status.shaderPackCount}`,
    );

    // Ensure no host filesystem paths leak into /api/status JSON
    assert.equal(rawBody.includes(tempDataDir), false);
    assert.equal(rawBody.includes(path.basename(tempDataDir)), false);
    assert.equal(rawBody.includes(REPO_ROOT), false);
    assert.equal(rawBody.includes(os.tmpdir()), false);
    assert.equal(/[A-Za-z]:\\/.test(rawBody), false);

    // Verify 206 Partial Content + ETag + COOP/COEP on /data/archives/level0.bin
    const rangeRes = await fetch(`${baseUrl}/data/${ARCHIVE_REL_PATH}`, {
      headers: { Range: 'bytes=100-355' },
    });
    assert.equal(rangeRes.status, 206);
    assert.equal(rangeRes.headers.get('cross-origin-opener-policy'), 'same-origin');
    assert.equal(rangeRes.headers.get('cross-origin-embedder-policy'), 'require-corp');
    assert.equal(
      rangeRes.headers.get('content-range'),
      `bytes 100-355/${ARCHIVE_BYTES.byteLength}`,
    );
    const etag = rangeRes.headers.get('etag');
    assert.ok(etag && etag.startsWith('"') && etag.endsWith('"'), `Expected strong ETag, got ${etag}`);
    const rangeBytes = new Uint8Array(await rangeRes.arrayBuffer());
    assert.deepEqual(rangeBytes, ARCHIVE_BYTES.subarray(100, 356));

    // If-None-Match -> 304 Not Modified
    const notModRes = await fetch(`${baseUrl}/data/${ARCHIVE_REL_PATH}`, {
      headers: { 'If-None-Match': etag },
    });
    assert.equal(notModRes.status, 304);

    // If-Range with matching strong ETag -> 206 Partial Content
    const ifRangeHit = await fetch(`${baseUrl}/data/${ARCHIVE_REL_PATH}`, {
      headers: { Range: 'bytes=0-63', 'If-Range': etag },
    });
    assert.equal(ifRangeHit.status, 206);
    assert.deepEqual(new Uint8Array(await ifRangeHit.arrayBuffer()), ARCHIVE_BYTES.subarray(0, 64));

    // If-Range with stale ETag -> 200 OK full archive
    const ifRangeMiss = await fetch(`${baseUrl}/data/${ARCHIVE_REL_PATH}`, {
      headers: { Range: 'bytes=0-63', 'If-Range': '"stale-etag"' },
    });
    assert.equal(ifRangeMiss.status, 200);
    assert.equal((await ifRangeMiss.arrayBuffer()).byteLength, ARCHIVE_BYTES.byteLength);

    // Out-of-bounds Range -> 416 Range Not Satisfiable
    const oobRes = await fetch(`${baseUrl}/data/${ARCHIVE_REL_PATH}`, {
      headers: { Range: `bytes=${ARCHIVE_BYTES.byteLength + 10}-${ARCHIVE_BYTES.byteLength + 20}` },
    });
    assert.equal(oobRes.status, 416);
  });

  it('2. streams real HTTP 206 ranges through RangeCache -> HttpfsMailboxHost/Client -> Wasm64 createHostImports', async () => {
    const manifestRes = await fetch(`${baseUrl}/data/data-manifest.json`);
    assert.equal(manifestRes.status, 200);
    const manifest = await manifestRes.json();

    const bootsetRes = await fetch(`${baseUrl}/data/bootset.json`);
    assert.equal(bootsetRes.status, 200);
    const bootset = await bootsetRes.json();

    const networkRequests = [];
    const trackedFetch = async (url, init) => {
      const res = await fetch(url, init);
      networkRequests.push({
        url,
        range: init?.headers?.Range || null,
        status: res.status,
      });
      return res;
    };

    const rangeCache = new RangeCache({
      dataBase: `${baseUrl}/data/`,
      chunkSize: 4096,
      maxBytes: 256 * 1024,
      fetchImpl: trackedFetch,
    });

    // Prefetch bootset.json (chunks 0 and 1 of archives/level0.bin)
    const prefetchSummary = await rangeCache.prefetchBootset(bootset);
    assert.equal(prefetchSummary.requestedEntries, 1);
    assert.equal(prefetchSummary.uniqueChunks, 2);
    assert.equal(prefetchSummary.prefetched, 2);
    assert.equal(prefetchSummary.failed, 0);
    assert.equal(networkRequests.length, 2);
    assert.ok(networkRequests.every((r) => r.status === 206));

    // Spanning read inside prefetched region (chunks 0 -> 1) must be 100% cache hit
    const prefetchedSlice = await rangeCache.read(ARCHIVE_REL_PATH, 3500, 2000);
    assert.deepEqual(prefetchedSlice, ARCHIVE_BYTES.subarray(3500, 5500));
    assert.equal(networkRequests.length, 2, 'Prefetched spanning slice must not trigger HTTP fetches');
    assert.equal(rangeCache.getStats().hits, 2);

    // Spanning read crossing from cached chunk 1 into uncached chunk 2 triggers exactly 1 HTTP 206 fetch
    const crossSlice = await rangeCache.read(ARCHIVE_REL_PATH, 7000, 3000);
    assert.deepEqual(crossSlice, ARCHIVE_BYTES.subarray(7000, 10000));
    assert.equal(networkRequests.length, 3);
    assert.equal(networkRequests[2].status, 206);

    // Now wire HttpfsMailboxHost over a SharedArrayBuffer (2048B payload so large Wasm64 reads span
    // multiple mailbox round-trips AND multiple 4096B HTTP Range chunks).
    const mailboxSab = createHttpfsMailboxBuffer(2048);
    const host = new HttpfsMailboxHost(mailboxSab, {
      rangeCache,
      manifest,
    });
    const hostLoop = host.start();

    const workerScript = `
      const { workerData, parentPort } = require('node:worker_threads');
      const { HttpfsMailboxClient } = require(workerData.mailboxPath);
      const { createHostImports } = require(workerData.loaderPath);

      const memory = new WebAssembly.Memory({ initial: 2, maximum: 2, shared: true });
      const memU8 = new Uint8Array(memory.buffer);
      const enc = new TextEncoder();
      const logs = [];

      const client = new HttpfsMailboxClient(workerData.sab, { timeoutMs: 8000 });
      const imports = createHostImports({
        memory,
        httpfsClient: client,
        log: (msg) => logs.push(msg),
      });

      // Verify host_log with Wasm64 BigInt pointer/length
      const logBytes = enc.encode('e2e-wasm64-httpfs');
      memU8.set(logBytes, 64);
      imports.env.host_log(64n, BigInt(logBytes.byteLength));

      // Missing file rejected by manifest -> -1
      const badPath = enc.encode('archives/does_not_exist.bin');
      memU8.set(badPath, 256);
      const missingHandle = imports.env.host_httpfs_open(256n, BigInt(badPath.byteLength));

      // Open real archive via Wasm64 BigInt pointer/length
      const relPathBytes = enc.encode(workerData.archiveRelPath);
      memU8.set(relPathBytes, 512);
      const handle = imports.env.host_httpfs_open(512n, BigInt(relPathBytes.byteLength));

      // First read: 10000 bytes starting at offset 7500 (spans chunks 1, 2, 3, 4 and 5 mailbox frames!)
      const firstLen = imports.env.host_httpfs_read(handle, 7500n, 4096n, 10000n);
      const firstCopy = Array.from(memU8.slice(4096, 4096 + Number(firstLen)));

      // Tail read past EOF: 1000 bytes requested from archiveSize - 250 -> exact 250 bytes
      const tailOffset = BigInt(workerData.archiveSize - 250);
      const tailLen = imports.env.host_httpfs_read(handle, tailOffset, 16384n, 1000n);
      const tailCopy = Array.from(memU8.slice(16384, 16384 + Number(tailLen)));

      parentPort.postMessage({
        phase: 'after-cold-reads',
        handle,
        missingHandle,
        firstLen: Number(firstLen),
        firstCopy,
        tailLen: Number(tailLen),
        tailCopy,
        logs,
      });

      // Wait for main thread signal before executing cached repeat read
      parentPort.once('message', () => {
        const repeatLen = imports.env.host_httpfs_read(handle, 7500n, 32768n, 10000n);
        const repeatCopy = Array.from(memU8.slice(32768, 32768 + Number(repeatLen)));
        imports.env.host_httpfs_close(handle);
        const afterCloseLen = imports.env.host_httpfs_read(handle, 0n, 32768n, 16n);

        parentPort.postMessage({
          phase: 'after-repeat-read',
          repeatLen: Number(repeatLen),
          repeatCopy,
          afterCloseLen: Number(afterCloseLen),
        });
      });
    `;

    try {
      const worker = new NodeWorker(workerScript, {
        eval: true,
        workerData: {
          sab: mailboxSab,
          mailboxPath: path.join(REPO_ROOT, 'web', 'workers', 'httpfs_mailbox.js'),
          loaderPath: path.join(REPO_ROOT, 'web', 'loader.js'),
          archiveRelPath: ARCHIVE_REL_PATH,
          archiveSize: ARCHIVE_BYTES.byteLength,
        },
      });

      const coldMsg = await new Promise((resolve, reject) => {
        worker.once('message', resolve);
        worker.once('error', reject);
        worker.once('exit', (code) => {
          if (code !== 0) reject(new Error(`Worker exited early with code ${code}`));
        });
      });

      assert.equal(coldMsg.phase, 'after-cold-reads');
      assert.deepEqual(coldMsg.logs, ['e2e-wasm64-httpfs']);
      assert.equal(coldMsg.missingHandle, -1);
      assert.ok(coldMsg.handle >= 0);
      assert.equal(coldMsg.firstLen, 10000);
      assert.deepEqual(new Uint8Array(coldMsg.firstCopy), ARCHIVE_BYTES.subarray(7500, 17500));
      assert.equal(coldMsg.tailLen, 250);
      assert.deepEqual(
        new Uint8Array(coldMsg.tailCopy),
        ARCHIVE_BYTES.subarray(ARCHIVE_BYTES.byteLength - 250),
      );

      // Snapshot network requests & cache hits before repeat read
      const netCallsBeforeRepeat = networkRequests.length;
      const hitsBeforeRepeat = rangeCache.getStats().hits;
      assert.ok(
        networkRequests.every((r) => r.status === 206),
        'All RangeCache HTTP requests must receive 206 Partial Content',
      );

      // Trigger repeat read of the exact same 10000-byte range over SAB -> Wasm64 memory
      const repeatPromise = new Promise((resolve, reject) => {
        worker.once('message', resolve);
        worker.once('error', reject);
      });
      const exitPromise = new Promise((resolve, reject) => {
        worker.once('exit', (code) => {
          if (code === 0) resolve();
          else reject(new Error(`Worker exited with non-zero code ${code}`));
        });
      });
      worker.postMessage('go-repeat');

      const repeatMsg = await repeatPromise;
      await exitPromise;

      assert.equal(repeatMsg.phase, 'after-repeat-read');
      assert.equal(repeatMsg.repeatLen, 10000);
      assert.deepEqual(new Uint8Array(repeatMsg.repeatCopy), ARCHIVE_BYTES.subarray(7500, 17500));
      assert.ok(repeatMsg.afterCloseLen < 0, 'Read on closed handle must fail');

      assert.equal(
        networkRequests.length,
        netCallsBeforeRepeat,
        'Repeat read through HttpfsMailboxHost must hit RangeCache with zero HTTP fetches',
      );
      assert.ok(
        rangeCache.getStats().hits > hitsBeforeRepeat,
        'RangeCache hit counter must increase on repeat reads',
      );
    } finally {
      await host.stop();
      await hostLoop;
    }
  });

  it('3. fetches .wgsl.pack over HTTP -> WgslPackCache -> GpuRingBuffer SAB -> WgpuCommandProcessor frame execution', async () => {
    const packCache = new WgslPackCache();
    const packUrl = `${baseUrl}/data/${SHADER_PACK_REL_PATH}`;
    const pack = await packCache.loadPackUrl(packUrl);

    assert.equal(pack.version, WGPK_VERSION);
    assert.equal(pack.size, 2);
    assert.equal(packCache.hasShader(VS_HASH), true);
    assert.equal(packCache.hasShader(FS_HASH), true);

    const vsSource = packCache.getShaderSource(VS_HASH);
    const fsSource = packCache.getShaderSource(FS_HASH);
    assert.equal(vsSource, VS_WGSL);
    assert.equal(fsSource, FS_WGSL);
    assert.equal(packCache.getShaderMetadata(VS_HASH)?.hashHex, formatHashHex(VS_HASH));

    // Create mock WebGPU device + swapchain context and verify WgslPackCache prewarm caching too
    const { device, ctx, calls, swapchainView } = createMockWebGpu();
    const prewarm = await packCache.prewarm(device, [VS_HASH, FS_HASH]);
    assert.equal(prewarm.total, 2);
    assert.equal(prewarm.compiled, 2);
    assert.equal(prewarm.cached, 0);
    // Clear prewarm mock calls so we cleanly inspect the GPU ring frame submission below
    calls.length = 0;

    // Producer SAB ring buffer
    const producerRing = GpuRingBuffer.create(65536);

    // Consumer WgpuCommandProcessor attached to the same SharedArrayBuffer
    const processor = new WgpuCommandProcessor(device, ctx, { format: 'bgra8unorm' });
    processor.attachRing(producerRing.sab);

    // Encode full render frame packet stream using WGSL loaded from HTTP .wgsl.pack
    const triangleVertices = new Float32Array([
      0.0, 0.5, 0.0,
      -0.5, -0.5, 0.0,
      0.5, -0.5, 0.0,
    ]);
    const vertexBytes = new Uint8Array(
      triangleVertices.buffer,
      triangleVertices.byteOffset,
      triangleVertices.byteLength,
    );

    const framePackets = new BinaryCommandBuilder()
      .beginFrame()
      .clearColor(0.08, 0.12, 0.2, 1.0)
      .createShaderModule(1, WGPU_SHADER_STAGE.VERTEX, vsSource)
      .createShaderModule(2, WGPU_SHADER_STAGE.FRAGMENT, fsSource)
      .createBuffer(10, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST, vertexBytes.byteLength)
      .writeBuffer(10, 0, vertexBytes)
      .createRenderPipeline(20, 1, 2, 12, PRIMITIVE_TOPOLOGY.TRIANGLE_LIST)
      .setPipeline(20)
      .setVertexBuffer(0, 10, 0)
      .draw(3, 1, 0, 0)
      .present()
      .finish();

    // Write via Wasm host_wgpu_submit -> GpuRingBuffer SAB -> WgpuCommandProcessor.drainRing()
    const wasmMem = new WebAssembly.Memory({ initial: 1, maximum: 1 });
    new Uint8Array(wasmMem.buffer).set(framePackets, 1024);
    const gpuHostImports = createHostImports({
      memory: wasmMem,
      wgpuSubmit: (bytes) => {
        const ok = producerRing.writeBytes(bytes);
        assert.equal(ok, true, 'GpuRingBuffer.writeBytes must accept full frame packet');
      },
    });
    gpuHostImports.env.host_wgpu_submit(1024n, BigInt(framePackets.byteLength));

    assert.equal(producerRing.availableBytes(), framePackets.byteLength);
    assert.equal(producerRing.droppedPackets, 0);

    const drainResult = processor.drainRing();
    assert.equal(drainResult.bytesDrained, framePackets.byteLength);
    assert.equal(drainResult.presents, 1);
    assert.equal(drainResult.frameCount, 1);
    assert.equal(producerRing.availableBytes(), 0);

    // Assert exact WebGPU call order and payloads
    assert.deepEqual(
      calls.map((c) => c.op),
      [
        'createShaderModule',
        'createShaderModule',
        'createBuffer',
        'writeBuffer',
        'createRenderPipeline',
        'beginRenderPass',
        'setPipeline',
        'setVertexBuffer',
        'draw',
        'renderPass.end',
        'encoder.finish',
        'submit',
      ],
    );

    assert.equal(calls[0].desc.code, VS_WGSL);
    assert.equal(calls[1].desc.code, FS_WGSL);
    assert.equal(calls[2].desc.size, vertexBytes.byteLength);
    assert.equal(calls[2].desc.usage, BUFFER_USAGE.VERTEX | BUFFER_USAGE.COPY_DST);
    assert.equal(calls[3].buffer, calls[2].result);
    assert.equal(calls[3].dstOffset, 0);
    assert.deepEqual(calls[3].data, vertexBytes);

    assert.equal(calls[4].desc.vertex.module, calls[0].result);
    assert.equal(calls[4].desc.vertex.entryPoint, 'vs_main');
    assert.equal(calls[4].desc.vertex.buffers[0].arrayStride, 12);
    assert.equal(calls[4].desc.vertex.buffers[0].attributes[0].format, 'float32x3');
    assert.equal(calls[4].desc.fragment.module, calls[1].result);
    assert.equal(calls[4].desc.fragment.entryPoint, 'fs_main');
    assert.equal(calls[4].desc.primitive.topology, 'triangle-list');

    assert.equal(calls[5].desc.colorAttachments[0].view, swapchainView);
    assert.ok(Math.abs(calls[5].desc.colorAttachments[0].clearValue.r - 0.08) < 1e-5);
    assert.equal(calls[6].pipeline, calls[4].result);
    assert.equal(calls[7].slot, 0);
    assert.equal(calls[7].buffer, calls[2].result);
    assert.equal(calls[7].offset, 0);
    assert.equal(calls[8].vertexCount, 3);
    assert.equal(calls[8].instanceCount, 1);
    assert.equal(calls[11].commandBuffers[0], calls[10].result);
  });

  it('4. verifies InputBridge 40B SAB, AudioRingBuffer stereo SAB, and UserdataStore /userdata/slot0.sav round-trip', async () => {
    // --- A. InputBridge 40-byte SAB (keyboard + pointer-locked mouse + gamepad) ---
    const canvas = new MockEventTarget();
    const doc = new MockEventTarget();
    doc.pointerLockElement = null;
    doc.hidden = false;
    doc.visibilityState = 'visible';
    const win = new MockEventTarget();
    let currentGamepads = [];
    const nav = { getGamepads: () => currentGamepads };

    const bridge = new InputBridge({
      canvas,
      windowObj: win,
      documentObj: doc,
      navigatorObj: nav,
      autoStart: false,
    });

    try {
      assert.ok(bridge.sab instanceof SharedArrayBuffer);
      assert.equal(bridge.sab.byteLength, INPUT_STATE_BYTES);
      assert.equal(INPUT_STATE_BYTES, 40);

      // Engage pointer lock
      doc.pointerLockElement = canvas;
      doc.dispatch('pointerlockchange');
      assert.equal(bridge.isPointerLocked(), true);

      // Keyboard: W + Space + ShiftLeft
      win.dispatch('keydown', { code: 'KeyW', repeat: false });
      win.dispatch('keydown', { code: 'Space', repeat: false });
      win.dispatch('keydown', { code: 'ShiftLeft', repeat: false });

      // Mouse movement + left click (RT=1.0)
      win.dispatch('mousemove', { movementX: 14, movementY: -9 });
      win.dispatch('mousemove', { movementX: -2, movementY: 4 });
      win.dispatch('mousedown', { button: 0 });

      // Gamepad: Y pressed (btn 3), LT=0.75 (btn 6), right stick (0.6, -0.5)
      const gpButtons = Array.from({ length: 17 }, () => ({ pressed: false, value: 0 }));
      gpButtons[3] = { pressed: true, value: 1 };
      gpButtons[6] = { pressed: true, value: 0.75 };
      currentGamepads = [
        {
          connected: true,
          mapping: 'standard',
          buttons: gpButtons,
          axes: [0, 0, 0.6, -0.5],
        },
      ];
      assert.equal(bridge.pollGamepads(), true);

      const snap = bridge.getSnapshot();
      assert.ok(snap !== null);
      assert.equal(snap.seq & 1, 0, 'Committed seqlock seq must be even');
      assert.ok(snap.seq >= 2);

      const expectedButtons =
        INPUT_BUTTONS.MOVE_FORWARD |
        INPUT_BUTTONS.JUMP |
        INPUT_BUTTONS.A |
        INPUT_BUTTONS.SPRINT |
        INPUT_BUTTONS.LS |
        INPUT_BUTTONS.Y;
      assert.equal(snap.buttons, expectedButtons);
      assert.equal(snap.mouse_dx, 12);
      assert.equal(snap.mouse_dy, -5);
      assert.equal(snap.lx, 0);
      assert.equal(snap.ly, 1);
      assert.ok(snap.rx > 0.5 && snap.rx < 0.7);
      assert.ok(snap.ry > 0.4 && snap.ry < 0.6); // invertGamepadY=true flips -0.5 -> +0.5
      assert.ok(snap.lt > 0.7 && snap.lt < 0.8);
      assert.equal(snap.rt, 1);

      // Cross-check raw 40-byte SAB offsets directly
      const sabView = new DataView(bridge.sab);
      assert.equal(sabView.getUint32(INPUT_OFFSETS.SEQ, true), snap.seq);
      assert.equal(sabView.getUint32(INPUT_OFFSETS.BUTTONS, true), expectedButtons);
      assert.equal(sabView.getInt32(INPUT_OFFSETS.MOUSE_DX, true), 12);
      assert.equal(sabView.getInt32(INPUT_OFFSETS.MOUSE_DY, true), -5);
      assert.equal(sabView.getFloat32(INPUT_OFFSETS.LEFT_STICK_Y, true), 1);
      assert.equal(sabView.getFloat32(INPUT_OFFSETS.RIGHT_TRIGGER, true), 1);
    } finally {
      bridge.destroy();
    }

    // --- B. AudioRingBuffer stereo SAB wrap-around + AudioMixerWorker ---
    const capacityFrames = 512;
    const channels = 2;
    const audioProducer = AudioRingBuffer.create(capacityFrames, channels);
    const audioConsumer = new AudioRingBuffer(audioProducer.sab);

    const makeStereoBlock = (startFrame, frameCount) => {
      const pcm = new Float32Array(frameCount * channels);
      for (let f = 0; f < frameCount; f += 1) {
        const idx = startFrame + f;
        pcm[f * 2] = Math.fround(Math.sin(idx * 0.05) * 0.8);
        pcm[f * 2 + 1] = Math.fround(Math.cos(idx * 0.03) * 0.6);
      }
      return pcm;
    };

    // Push & pull 3 blocks of 300 frames (900 frames total > 512 capacity -> exercises wrap-around)
    for (let block = 0; block < 3; block += 1) {
      const input = makeStereoBlock(block * 300, 300);
      const written = audioProducer.pushSamples(input);
      assert.equal(written, 300);
      assert.equal(audioConsumer.availableFrames(), 300);

      const output = new Float32Array(300 * channels);
      const pulled = audioConsumer.pullSamples(output);
      assert.equal(pulled, 300);
      assert.deepEqual(output, input);
    }

    assert.equal(audioConsumer.underruns, 0);
    assert.equal(audioConsumer.underrunFrames, 0);
    assert.equal(audioProducer.overflows, 0);
    assert.equal(audioProducer.droppedFrames, 0);
    assert.equal(audioConsumer.availableFrames(), 0);

    // Also verify AudioMixerWorker attached to the same SAB
    const workerMsgs = [];
    const mixer = new AudioMixerWorker((m) => workerMsgs.push(m));
    mixer.handleMessage({ init: true, ringSab: audioProducer.sab, sampleRate: 48000, channels: 2 });
    assert.equal(workerMsgs[0].ready, true);
    assert.equal(workerMsgs[0].capacityFrames, capacityFrames);

    const gainInput = makeStereoBlock(900, 64);
    mixer.handleMessage({ id: 1, setVolume: true, volume: 0.5 });
    mixer.handleMessage({ id: 2, pushPcm: true, samples: gainInput });
    const gainOut = new Float32Array(64 * channels);
    assert.equal(mixer.pullSamples(gainOut), 64);
    for (let i = 0; i < gainOut.length; i += 1) {
      assert.ok(Math.abs(gainOut[i] - gainInput[i] * 0.5) < 1e-6);
    }
    assert.equal(mixer.getStats().underruns, 0);

    // --- C. UserdataStore /userdata/slot0.sav round-trip with in-memory IDB mock ---
    const mockIdb = createMockIndexedDB();
    const saveBytes = makePatternBytes(4096, 0xc3);
    const saveMtime = 1700001234000;
    const settingsBytes = new TextEncoder().encode(JSON.stringify({ masterVolume: 0.5, slot: 0 }));

    const writerStore = new UserdataStore({ indexedDB: mockIdb });
    const writerImports = createHostImports({ userdataStore: writerStore });
    const savedSlot = await writerImports.userdata.save('/userdata/slot0.sav', saveBytes, saveMtime);
    assert.equal(savedSlot.path, '/userdata/slot0.sav');
    assert.equal(savedSlot.size, saveBytes.byteLength);
    await writerStore.saveFile('/userdata/profiles/p0/settings.json', settingsBytes, saveMtime + 500);
    await assert.rejects(() => writerStore.saveFile('/userdata/../escape.sav', new Uint8Array([1])));
    writerStore.close();

    // Fresh store instance simulates reload and restores into mock Emscripten FS
    const readerStore = new UserdataStore({ indexedDB: mockIdb });
    const readerImports = createHostImports({ userdataStore: readerStore });
    const fsAdapter = createMockFsAdapter();
    const restoreRes = await readerImports.userdata.restore(fsAdapter, { timeoutMs: 2000 });

    assert.equal(restoreRes.timedOut, false);
    assert.equal(restoreRes.error, null);
    assert.equal(restoreRes.restored, 2);
    assert.equal(restoreRes.skipped, 0);
    assert.deepEqual(fsAdapter.files.get('/userdata/slot0.sav'), saveBytes);
    assert.deepEqual(fsAdapter.files.get('/userdata/profiles/p0/settings.json'), settingsBytes);
    assert.equal(fsAdapter.mtimes.get('/userdata/slot0.sav'), saveMtime);
    assert.equal(fsAdapter.dirs.has('/userdata'), true);
    assert.equal(fsAdapter.dirs.has('/userdata/profiles/p0'), true);
    readerStore.close();
  });
});
