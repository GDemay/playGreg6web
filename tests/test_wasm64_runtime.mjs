import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');

const {
  createHostImports,
  createRuntimeModuleOptions,
  encodeTestFramePacket,
  ptrToNumber,
} = require('../web/loader.js');

const REQUIRED_HOST_IMPORT_NAMES = [
  'host_log',
  'host_httpfs_open',
  'host_httpfs_read',
  'host_httpfs_close',
  'host_wgpu_submit',
];

const RUNTIME_WASM_PATH = path.join(repoRoot, 'web', 'dist', 'runtime.wasm');
const CHECK_WASM_EXPORTS_SCRIPT = path.join(repoRoot, 'tools', 'check_wasm_exports.py');

// ---------------------------------------------------------------------------
// Minimal zero-dependency Wasm64 (memory64) binary module builder
// ---------------------------------------------------------------------------

const WASM_MAGIC_VERSION = [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00];

const SEC = {
  TYPE: 1,
  IMPORT: 2,
  FUNCTION: 3,
  EXPORT: 7,
  CODE: 10,
};

const VAL = {
  I32: 0x7f,
  I64: 0x7e,
};

const KIND = {
  FUNC: 0x00,
  MEMORY: 0x02,
};

const MEM_FLAG = {
  HAS_MAX: 0x01,
  SHARED: 0x02,
  MEMORY64: 0x04,
};

function encodeUleb128(value) {
  let v = typeof value === 'bigint' ? value : BigInt(value);
  if (v < 0n) {
    throw new RangeError(`ULEB128 requires non-negative value, got ${value}`);
  }
  const out = [];
  do {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    if (v !== 0n) {
      byte |= 0x80;
    }
    out.push(byte);
  } while (v !== 0n);
  return out;
}

function encodeName(str) {
  const bytes = Array.from(new TextEncoder().encode(str));
  return [...encodeUleb128(bytes.length), ...bytes];
}

function encodeSection(id, payload) {
  return [id, ...encodeUleb128(payload.length), ...payload];
}

function encodeFuncType(params, results) {
  return [
    0x60,
    ...encodeUleb128(params.length),
    ...params,
    ...encodeUleb128(results.length),
    ...results,
  ];
}

function encodeCodeBody(localDecls, bodyBytes) {
  const inner = [...encodeUleb128(localDecls.length)];
  for (const [count, valType] of localDecls) {
    inner.push(...encodeUleb128(count), valType);
  }
  inner.push(...bodyBytes, 0x0b); // end
  return [...encodeUleb128(inner.length), ...inner];
}

/**
 * Builds a minimal valid Wasm64 (memory64, i64 memory index) binary module
 * implementing the frozen playgta6 host ABI from docs/wasm_abi.md:
 *
 * Imports:
 *   - env.memory (memory64, flags = 0x05 unshared or 0x07 shared; 0x04 if unbounded)
 *   - env.host_log(i64, i64) -> ()
 *   - env.host_httpfs_open(i64, i64) -> i32
 *   - env.host_httpfs_read(i32, i64, i64, i64) -> i64
 *   - env.host_httpfs_close(i32) -> ()
 *   - env.host_wgpu_submit(i64, i64) -> ()
 *
 * Exports:
 *   - memory
 *   - guest_log_and_submit(logPtr: i64, logLen: i64, pktPtr: i64, pktLen: i64) -> ()
 *   - guest_httpfs_roundtrip(pathPtr: i64, pathLen: i64, offset: i64, dstPtr: i64, len: i64) -> i64
 *   - guest_call_log(ptr: i64, len: i64) -> ()
 *   - guest_call_wgpu_submit(ptr: i64, len: i64) -> ()
 *   - guest_store_u8(addr: i64, val: i32) -> ()
 *   - guest_load_u8(addr: i64) -> i32
 *   - guest_memory_size() -> i64
 *   - guest_memory_grow(delta: i64) -> i64
 */
function buildWasm64AbiModule({ shared = false, minPages = 1n, maxPages = 2n } = {}) {
  const hasMax = maxPages !== null && maxPages !== undefined;
  let memFlags = MEM_FLAG.MEMORY64;
  if (hasMax) memFlags |= MEM_FLAG.HAS_MAX;
  if (shared) memFlags |= MEM_FLAG.SHARED;

  // Type section
  const types = [
    encodeFuncType([VAL.I64, VAL.I64], []), // 0: (i64, i64) -> ()
    encodeFuncType([VAL.I64, VAL.I64], [VAL.I32]), // 1: (i64, i64) -> i32
    encodeFuncType([VAL.I32, VAL.I64, VAL.I64, VAL.I64], [VAL.I64]), // 2: (i32, i64, i64, i64) -> i64
    encodeFuncType([VAL.I32], []), // 3: (i32) -> ()
    encodeFuncType([VAL.I64, VAL.I64, VAL.I64, VAL.I64], []), // 4: guest_log_and_submit
    encodeFuncType([VAL.I64, VAL.I64, VAL.I64, VAL.I64, VAL.I64], [VAL.I64]), // 5: guest_httpfs_roundtrip
    encodeFuncType([VAL.I64, VAL.I32], []), // 6: guest_store_u8
    encodeFuncType([VAL.I64], [VAL.I32]), // 7: guest_load_u8
    encodeFuncType([], [VAL.I64]), // 8: guest_memory_size
    encodeFuncType([VAL.I64], [VAL.I64]), // 9: guest_memory_grow
  ];
  const typePayload = [...encodeUleb128(types.length), ...types.flat()];

  // Import section: env.memory + 5 env.host_* functions (func indices 0..4)
  const memLimits = [
    memFlags,
    ...encodeUleb128(minPages),
    ...(hasMax ? encodeUleb128(maxPages) : []),
  ];
  const importEntries = [
    [...encodeName('env'), ...encodeName('memory'), KIND.MEMORY, ...memLimits],
    [...encodeName('env'), ...encodeName('host_log'), KIND.FUNC, ...encodeUleb128(0)],
    [...encodeName('env'), ...encodeName('host_httpfs_open'), KIND.FUNC, ...encodeUleb128(1)],
    [...encodeName('env'), ...encodeName('host_httpfs_read'), KIND.FUNC, ...encodeUleb128(2)],
    [...encodeName('env'), ...encodeName('host_httpfs_close'), KIND.FUNC, ...encodeUleb128(3)],
    [...encodeName('env'), ...encodeName('host_wgpu_submit'), KIND.FUNC, ...encodeUleb128(0)],
  ];
  const importPayload = [...encodeUleb128(importEntries.length), ...importEntries.flat()];

  // Function section: defined funcs at indices 5..12
  const definedTypeIndices = [4, 5, 0, 0, 6, 7, 8, 9];
  const funcPayload = [
    ...encodeUleb128(definedTypeIndices.length),
    ...definedTypeIndices.flatMap((idx) => encodeUleb128(idx)),
  ];

  // Export section
  const exportEntries = [
    [...encodeName('memory'), KIND.MEMORY, ...encodeUleb128(0)],
    [...encodeName('guest_log_and_submit'), KIND.FUNC, ...encodeUleb128(5)],
    [...encodeName('guest_httpfs_roundtrip'), KIND.FUNC, ...encodeUleb128(6)],
    [...encodeName('guest_call_log'), KIND.FUNC, ...encodeUleb128(7)],
    [...encodeName('guest_call_wgpu_submit'), KIND.FUNC, ...encodeUleb128(8)],
    [...encodeName('guest_store_u8'), KIND.FUNC, ...encodeUleb128(9)],
    [...encodeName('guest_load_u8'), KIND.FUNC, ...encodeUleb128(10)],
    [...encodeName('guest_memory_size'), KIND.FUNC, ...encodeUleb128(11)],
    [...encodeName('guest_memory_grow'), KIND.FUNC, ...encodeUleb128(12)],
  ];
  const exportPayload = [...encodeUleb128(exportEntries.length), ...exportEntries.flat()];

  // Code section
  const codeBodies = [
    // func 5: guest_log_and_submit(logPtr, logLen, pktPtr, pktLen)
    encodeCodeBody([], [
      0x20, 0x00, // local.get 0 (logPtr)
      0x20, 0x01, // local.get 1 (logLen)
      0x10, 0x00, // call 0 (host_log)
      0x20, 0x02, // local.get 2 (pktPtr)
      0x20, 0x03, // local.get 3 (pktLen)
      0x10, 0x04, // call 4 (host_wgpu_submit)
    ]),

    // func 6: guest_httpfs_roundtrip(pathPtr, pathLen, fileOffset, dstPtr, readLen) -> i64
    // local 5: i32 handle, local 6: i64 bytesRead
    encodeCodeBody(
      [
        [1, VAL.I32],
        [1, VAL.I64],
      ],
      [
        0x20, 0x00, // local.get 0 (pathPtr)
        0x20, 0x01, // local.get 1 (pathLen)
        0x10, 0x01, // call 1 (host_httpfs_open)
        0x22, 0x05, // local.tee 5 (handle)
        0x41, 0x00, // i32.const 0
        0x48, // i32.lt_s
        0x04, 0x40, // if (void)
        0x42, 0x7f, // i64.const -1
        0x0f, // return
        0x0b, // end if
        0x20, 0x05, // local.get 5 (handle)
        0x20, 0x02, // local.get 2 (fileOffset)
        0x20, 0x03, // local.get 3 (dstPtr)
        0x20, 0x04, // local.get 4 (readLen)
        0x10, 0x02, // call 2 (host_httpfs_read)
        0x21, 0x06, // local.set 6 (bytesRead)
        0x20, 0x05, // local.get 5 (handle)
        0x10, 0x03, // call 3 (host_httpfs_close)
        0x20, 0x06, // local.get 6 (bytesRead)
      ],
    ),

    // func 7: guest_call_log(ptr, len)
    encodeCodeBody([], [
      0x20, 0x00,
      0x20, 0x01,
      0x10, 0x00,
    ]),

    // func 8: guest_call_wgpu_submit(ptr, len)
    encodeCodeBody([], [
      0x20, 0x00,
      0x20, 0x01,
      0x10, 0x04,
    ]),

    // func 9: guest_store_u8(addr: i64, val: i32)
    encodeCodeBody([], [
      0x20, 0x00, // local.get 0 (addr)
      0x20, 0x01, // local.get 1 (val)
      0x3a, 0x00, 0x00, // i32.store8 align=0 offset=0
    ]),

    // func 10: guest_load_u8(addr: i64) -> i32
    encodeCodeBody([], [
      0x20, 0x00, // local.get 0 (addr)
      0x2d, 0x00, 0x00, // i32.load8_u align=0 offset=0
    ]),

    // func 11: guest_memory_size() -> i64
    encodeCodeBody([], [
      0x3f, 0x00, // memory.size 0
    ]),

    // func 12: guest_memory_grow(delta: i64) -> i64
    encodeCodeBody([], [
      0x20, 0x00, // local.get 0 (delta)
      0x40, 0x00, // memory.grow 0
    ]),
  ];
  const codePayload = [...encodeUleb128(codeBodies.length), ...codeBodies.flat()];

  return new Uint8Array([
    ...WASM_MAGIC_VERSION,
    ...encodeSection(SEC.TYPE, typePayload),
    ...encodeSection(SEC.IMPORT, importPayload),
    ...encodeSection(SEC.FUNCTION, funcPayload),
    ...encodeSection(SEC.EXPORT, exportPayload),
    ...encodeSection(SEC.CODE, codePayload),
  ]);
}

/**
 * Creates a real Wasm64 (i64-indexed) WebAssembly.Memory if the host Node/V8
 * engine supports memory64. Uses standardized `{ address: 'i64' }` first and
 * falls back to legacy `{ index: 'i64' }` for older V8 builds.
 * Returns null if memory64 is unsupported.
 */
function createMemory64({ initial = 1n, maximum = 2n, shared = false } = {}) {
  const base = { initial, maximum };
  if (shared) {
    base.shared = true;
  }
  for (const key of ['address', 'index']) {
    try {
      return new WebAssembly.Memory({ ...base, [key]: 'i64' });
    } catch {
      // Try next descriptor key or report unsupported below.
    }
  }
  return null;
}

function supportsMemory64() {
  return createMemory64({ initial: 1n, maximum: 2n, shared: false }) !== null;
}

function resolvePythonCommand() {
  const candidates = process.platform === 'win32' ? ['python', 'python3'] : ['python3', 'python'];
  for (const cmd of candidates) {
    const probe = spawnSync(cmd, ['--version'], { encoding: 'utf8' });
    if (probe.status === 0) {
      return cmd;
    }
  }
  return null;
}

function makePatternBytes(length, seed = 17) {
  const out = new Uint8Array(length);
  for (let i = 0; i < length; i += 1) {
    out[i] = ((i * 53) + (i >>> 8) * 19 + seed) & 0xff;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Part A: Synthetic Wasm64 Runtime & BigInt64 Host Import Tests (Always CI)
// ---------------------------------------------------------------------------

test('Part A1: synthetic Wasm64 binary validates under WebAssembly and check_wasm_exports.py', async () => {
  const unsharedWasm = buildWasm64AbiModule({ shared: false, minPages: 1n, maxPages: 2n });
  const sharedWasm = buildWasm64AbiModule({ shared: true, minPages: 1n, maxPages: 2n });

  if (supportsMemory64()) {
    assert.equal(WebAssembly.validate(unsharedWasm), true, 'unshared Wasm64 binary must validate');
    assert.equal(WebAssembly.validate(sharedWasm), true, 'shared Wasm64 binary must validate');

    const mod = await WebAssembly.compile(sharedWasm);
    const imports = WebAssembly.Module.imports(mod);
    const exportsList = WebAssembly.Module.exports(mod);

    assert.ok(
      imports.some((i) => i.module === 'env' && i.name === 'memory' && i.kind === 'memory'),
      'Expected env.memory import in compiled Wasm64 module',
    );
    for (const fnName of REQUIRED_HOST_IMPORT_NAMES) {
      assert.ok(
        imports.some((i) => i.module === 'env' && i.name === fnName && i.kind === 'function'),
        `Expected env.${fnName} function import in compiled Wasm64 module`,
      );
    }
    assert.ok(
      exportsList.some((e) => e.name === 'guest_log_and_submit' && e.kind === 'function'),
      'Expected guest_log_and_submit export',
    );
  }

  // Cross-check the exact same Wasm64 binary with tools/check_wasm_exports.py.
  const pythonCmd = resolvePythonCommand();
  if (pythonCmd) {
    const res = spawnSync(
      pythonCmd,
      [
        CHECK_WASM_EXPORTS_SCRIPT,
        '-',
        '--require-memory64',
        '--require-shared-memory',
        '--require-host-imports',
        '--json',
      ],
      { input: Buffer.from(sharedWasm), encoding: 'utf8' },
    );
    assert.equal(res.status, 0, `check_wasm_exports.py failed: ${res.stderr || res.stdout}`);
    const report = JSON.parse(res.stdout);
    assert.equal(report.ok, true);
    assert.equal(report.is_memory64, true);
    assert.equal(report.is_shared_memory, true);
    assert.equal(report.all_host_imports_ok, true);
    assert.equal(report.host_imports.length, 5);
    assert.equal(report.memories[0].flags, 0x07);
  }
});

test('Part A2: real Wasm64 instantiation executes host_log and host_wgpu_submit with BigInt pointers across page growth', async (t) => {
  if (!supportsMemory64()) {
    t.skip('Current Node/V8 runtime does not support WebAssembly memory64');
    return;
  }

  for (const shared of [false, true]) {
    const wasmBytes = buildWasm64AbiModule({ shared, minPages: 1n, maxPages: 2n });
    const memory = createMemory64({ initial: 1n, maximum: 2n, shared });
    assert.ok(memory, `Expected Wasm64 memory (shared=${shared})`);
    if (shared) {
      assert.ok(memory.buffer instanceof SharedArrayBuffer);
    } else {
      assert.ok(memory.buffer instanceof ArrayBuffer);
    }

    const logs = [];
    const packets = [];
    const hostImports = createHostImports({
      memory,
      log: (msg) => logs.push(msg),
      wgpuSubmit: (bytes) => packets.push(bytes),
    });

    const { instance } = await WebAssembly.instantiate(wasmBytes, {
      env: {
        memory,
        ...hostImports.env,
      },
    });
    const ex = instance.exports;

    // Verify true memory64 i64 page count and i64 byte load/store.
    assert.equal(ex.guest_memory_size(), 1n);
    ex.guest_store_u8(64n, 0xab);
    assert.equal(ex.guest_load_u8(64n), 0xab);

    // Stage UTF-8 log message and binary GPU ClearColor frame packet in page 0.
    const enc = new TextEncoder();
    const msg1 = 'Wasm64 memory64 live call — shared=' + shared;
    const msg1Bytes = enc.encode(msg1);
    const pkt1 = new Uint8Array(encodeTestFramePacket(0.12, 0.24, 0.36, 1.0));

    const page0U8 = new Uint8Array(memory.buffer);
    const logPtr1 = 512n;
    const pktPtr1 = 1024n;
    page0U8.set(msg1Bytes, Number(logPtr1));
    page0U8.set(pkt1, Number(pktPtr1));

    ex.guest_log_and_submit(
      logPtr1,
      BigInt(msg1Bytes.byteLength),
      pktPtr1,
      BigInt(pkt1.byteLength),
    );

    assert.equal(logs.length, 1);
    assert.equal(logs[0], msg1);
    assert.equal(packets.length, 1);
    assert.deepEqual(packets[0], pkt1);

    // Overwriting guest linear memory afterwards must not mutate captured GPU packet copy.
    page0U8.fill(0xff, Number(pktPtr1), Number(pktPtr1) + pkt1.byteLength);
    assert.deepEqual(packets[0], pkt1);

    // Grow memory64 from 1 page (64 KiB) to 2 pages (128 KiB) from inside Wasm64.
    const prevPages = ex.guest_memory_grow(1n);
    assert.equal(prevPages, 1n);
    assert.equal(ex.guest_memory_size(), 2n);
    assert.equal(memory.buffer.byteLength, 131072);

    // Write into page 1 (> 65536n) and verify createHostImports tracks the grown buffer.
    const msg2 = 'post-grow page-1 Wasm64 payload @ 70000n';
    const msg2Bytes = enc.encode(msg2);
    const pkt2 = new Uint8Array(encodeTestFramePacket(0.75, 0.5, 0.25, 1.0));

    const grownU8 = new Uint8Array(memory.buffer);
    const logPtr2 = 70000n;
    const pktPtr2 = 72000n;
    grownU8.set(msg2Bytes, Number(logPtr2));
    grownU8.set(pkt2, Number(pktPtr2));

    ex.guest_store_u8(131071n, 0x5c);
    assert.equal(ex.guest_load_u8(131071n), 0x5c);

    ex.guest_log_and_submit(
      logPtr2,
      BigInt(msg2Bytes.byteLength),
      pktPtr2,
      BigInt(pkt2.byteLength),
    );

    assert.equal(logs.length, 2);
    assert.equal(logs[1], msg2);
    assert.equal(packets.length, 2);
    assert.deepEqual(packets[1], pkt2);
  }
});

test('Part A3: real Wasm64 host_httpfs_* round-trip with BigInt pointers and offsets (0n and 65536n)', async (t) => {
  if (!supportsMemory64()) {
    t.skip('Current Node/V8 runtime does not support WebAssembly memory64');
    return;
  }

  // Virtual 70 KiB archive so we exercise both offset 0n and page-boundary offset 65536n.
  const archiveBytes = makePatternBytes(70000, 43);
  const openHandles = new Map();
  const recordedReads = [];
  const closedHandles = [];
  let nextHandle = 7;

  const mockHttpfsClient = {
    open(relPath) {
      if (relPath === 'x64/stream_pack.rpf') {
        const h = nextHandle++;
        openHandles.set(h, archiveBytes);
        return h;
      }
      return -1;
    },
    read(handle, offset, dstView, len) {
      recordedReads.push({ handle, offset, len });
      const data = openHandles.get(handle);
      if (!data) return -1n;
      const offNum = Number(offset);
      if (offNum < 0) return -1n;
      if (offNum >= data.byteLength) return 0n;
      const toCopy = Math.min(len, data.byteLength - offNum);
      dstView.set(data.subarray(offNum, offNum + toCopy), 0);
      return BigInt(toCopy);
    },
    close(handle) {
      closedHandles.push(handle);
      return openHandles.delete(handle);
    },
  };

  const wasmBytes = buildWasm64AbiModule({ shared: true, minPages: 2n, maxPages: 2n });
  const memory = createMemory64({ initial: 2n, maximum: 2n, shared: true });
  const hostImports = createHostImports({
    memory,
    httpfsClient: mockHttpfsClient,
  });

  const { instance } = await WebAssembly.instantiate(wasmBytes, {
    env: {
      memory,
      ...hostImports.env,
    },
  });
  const ex = instance.exports;
  const memU8 = new Uint8Array(memory.buffer);
  const enc = new TextEncoder();

  // Stage valid relative path at 256n.
  const goodPath = enc.encode('x64/stream_pack.rpf');
  memU8.set(goodPath, 256);

  // 1. Read 256 bytes from offset 0n into page 0 (dstPtr = 4096n).
  const n0 = ex.guest_httpfs_roundtrip(256n, BigInt(goodPath.byteLength), 0n, 4096n, 256n);
  assert.equal(n0, 256n);
  assert.deepEqual(memU8.slice(4096, 4096 + 256), archiveBytes.subarray(0, 256));
  assert.equal(recordedReads[0].offset, 0n);

  // 2. Read 4096 bytes from offset 65536n into page 1 (dstPtr = 69632n).
  const n1 = ex.guest_httpfs_roundtrip(256n, BigInt(goodPath.byteLength), 65536n, 69632n, 4096n);
  assert.equal(n1, 4096n);
  assert.deepEqual(memU8.slice(69632, 69632 + 4096), archiveBytes.subarray(65536, 65536 + 4096));
  assert.equal(recordedReads[1].offset, 65536n);

  // 3. EOF-clamped read starting at 65536n requesting 8192n bytes -> 70000 - 65536 = 4464n bytes.
  const n2 = ex.guest_httpfs_roundtrip(256n, BigInt(goodPath.byteLength), 65536n, 81920n, 8192n);
  assert.equal(n2, 4464n);
  assert.deepEqual(memU8.slice(81920, 81920 + 4464), archiveBytes.subarray(65536, 70000));

  // 4. Zero-length read returns 0n without calling mock read().
  const readsBeforeZero = recordedReads.length;
  const nZero = ex.guest_httpfs_roundtrip(256n, BigInt(goodPath.byteLength), 65536n, 4096n, 0n);
  assert.equal(nZero, 0n);
  assert.equal(recordedReads.length, readsBeforeZero);

  // 5. Missing file fails cleanly with -1n and never leaves an open handle.
  const missingPath = enc.encode('x64/does_not_exist.rpf');
  memU8.set(missingPath, 512);
  const nMissing = ex.guest_httpfs_roundtrip(
    512n,
    BigInt(missingPath.byteLength),
    0n,
    4096n,
    64n,
  );
  assert.equal(nMissing, -1n);

  // 6. Out-of-bounds destination pointer from Wasm64 returns -1n safely.
  const oobDst = BigInt(memory.buffer.byteLength - 8);
  const nOob = ex.guest_httpfs_roundtrip(256n, BigInt(goodPath.byteLength), 0n, oobDst, 64n);
  assert.equal(nOob, -1n);

  // Every opened handle must have been closed cleanly.
  assert.equal(openHandles.size, 0);
  assert.equal(closedHandles.length, 5);
});

test('Part A4: BigInt boundary, overflow, and out-of-bounds protection across ptrToNumber and host imports', async () => {
  // Exact MAX_SAFE_INTEGER boundary checks.
  const maxSafeBig = BigInt(Number.MAX_SAFE_INTEGER);
  assert.equal(ptrToNumber(0n), 0);
  assert.equal(ptrToNumber(65536n), 65536);
  assert.equal(ptrToNumber(maxSafeBig), Number.MAX_SAFE_INTEGER);
  assert.throws(() => ptrToNumber(maxSafeBig + 1n), RangeError);
  assert.throws(() => ptrToNumber(1n << 62n), RangeError);
  assert.throws(() => ptrToNumber(-1n), RangeError);
  assert.throws(() => ptrToNumber(-65536n), RangeError);
  assert.throws(() => ptrToNumber(-(1n << 63n)), RangeError);
  assert.throws(() => ptrToNumber(NaN), RangeError);
  assert.throws(() => ptrToNumber(Infinity), RangeError);
  assert.throws(() => ptrToNumber(null), TypeError);
  assert.throws(() => ptrToNumber(undefined), TypeError);

  const memory =
    createMemory64({ initial: 1n, maximum: 1n, shared: false }) ||
    new WebAssembly.Memory({ initial: 1, maximum: 1 });

  const logs = [];
  const packets = [];
  let openCalls = 0;
  let readCalls = 0;
  const hostImports = createHostImports({
    memory,
    log: (msg) => logs.push(msg),
    wgpuSubmit: (bytes) => packets.push(bytes),
    httpfsClient: {
      open() {
        openCalls += 1;
        return 1;
      },
      read() {
        readCalls += 1;
        return 4n;
      },
      close() {},
    },
  });

  const memSize = BigInt(memory.buffer.byteLength);

  // Direct JS-level BigInt overflow & OOB rejection on host_log / host_wgpu_submit.
  for (const [badPtr, badLen] of [
    [-1n, 4n],
    [0n, -1n],
    [maxSafeBig + 1n, 4n],
    [0n, maxSafeBig + 1n],
    [memSize, 1n],
    [memSize - 4n, 8n],
    [memSize + 65536n, 0n],
  ]) {
    assert.throws(() => hostImports.env.host_log(badPtr, badLen), RangeError);
    assert.throws(() => hostImports.env.host_wgpu_submit(badPtr, badLen), RangeError);
  }
  assert.equal(logs.length, 0);
  assert.equal(packets.length, 0);

  // Zero-length slice at exact end of linear memory is valid.
  hostImports.env.host_log(memSize, 0n);
  hostImports.env.host_wgpu_submit(memSize, 0n);
  assert.deepEqual(logs, ['']);
  assert.equal(packets.length, 1);
  assert.equal(packets[0].byteLength, 0);

  // host_httpfs_open and host_httpfs_read swallow invalid BigInt bounds into -1 / -1n.
  for (const [badPtr, badLen] of [
    [-1n, 4n],
    [maxSafeBig + 1n, 4n],
    [memSize - 2n, 8n],
    [0n, -8n],
  ]) {
    assert.equal(hostImports.env.host_httpfs_open(badPtr, badLen), -1);
    assert.equal(hostImports.env.host_httpfs_read(1, 0n, badPtr, badLen), -1n);
  }
  assert.equal(hostImports.env.host_httpfs_read(-1, 0n, 0n, 4n), -1n);
  assert.equal(openCalls, 0);
  assert.equal(readCalls, 0);

  // Verify real Wasm64 -> JS BigInt64 boundary rejects negative (0xffffffffffffffff -> -1n)
  // and > MAX_SAFE_INTEGER (2^53) pointers coming straight out of Wasm64 registers.
  if (supportsMemory64()) {
    const wasmBytes = buildWasm64AbiModule({ shared: false, minPages: 1n, maxPages: 1n });
    const { instance } = await WebAssembly.instantiate(wasmBytes, {
      env: { memory, ...hostImports.env },
    });
    const ex = instance.exports;

    assert.throws(() => ex.guest_call_log(-1n, 4n), RangeError);
    assert.throws(() => ex.guest_call_log(maxSafeBig + 1n, 4n), RangeError);
    assert.throws(() => ex.guest_call_wgpu_submit(memSize - 2n, 16n), RangeError);
    assert.throws(() => ex.guest_call_wgpu_submit(0n, -1n), RangeError);

    // Also verify createRuntimeModuleOptions binds a Wasm64 memory via preRun.
    const optsLogs = [];
    const opts = createRuntimeModuleOptions({
      log: (m) => optsLogs.push(m),
    });
    opts.preRun[0]({ wasmMemory: memory });
    const probeBytes = new TextEncoder().encode('preRun-wasm64-bound');
    new Uint8Array(memory.buffer).set(probeBytes, 128);
    opts.env.host_log(128n, BigInt(probeBytes.byteLength));
    assert.deepEqual(optsLogs, ['preRun-wasm64-bound']);
  }
});

// ---------------------------------------------------------------------------
// Part B: Local Emscripten Wasm64 Artifact Verification (Skips cleanly in CI)
// ---------------------------------------------------------------------------

test('Part B1: local web/dist/runtime.wasm passes check_wasm_exports.py (memory64 + shared + 5 host imports)', (t) => {
  if (!fs.existsSync(RUNTIME_WASM_PATH)) {
    t.skip('web/dist/runtime.wasm not present locally (expected in CI without Emscripten)');
    return;
  }

  const pythonCmd = resolvePythonCommand();
  if (!pythonCmd) {
    t.skip('Python interpreter not available to invoke tools/check_wasm_exports.py');
    return;
  }

  const res = spawnSync(
    pythonCmd,
    [
      CHECK_WASM_EXPORTS_SCRIPT,
      RUNTIME_WASM_PATH,
      '--require-memory64',
      '--require-shared-memory',
      '--require-host-imports',
      '--json',
    ],
    { encoding: 'utf8' },
  );

  assert.equal(
    res.status,
    0,
    `check_wasm_exports.py failed (exit=${res.status}):\n${res.stdout}\n${res.stderr}`,
  );

  const report = JSON.parse(res.stdout);
  assert.equal(report.ok, true, `Expected report.ok === true, errors: ${JSON.stringify(report.errors)}`);
  assert.equal(report.is_memory64, true, 'Expected runtime.wasm to be compiled with -sMEMORY64=1');
  assert.equal(report.is_shared_memory, true, 'Expected runtime.wasm to use shared memory (-pthread)');
  assert.equal(report.all_host_imports_ok, true, 'Expected all 5 env.host_* Wasm64 imports to match ABI');
  assert.equal(report.host_imports.length, 5);
  for (const entry of report.host_imports) {
    assert.equal(entry.ok, true, `Host import ${entry.module}.${entry.name} failed: ${entry.detail}`);
  }
});

test('Part B2: local web/dist/runtime.wasm compiles under WebAssembly and exposes required imports/exports', async (t) => {
  if (!fs.existsSync(RUNTIME_WASM_PATH)) {
    t.skip('web/dist/runtime.wasm not present locally (expected in CI without Emscripten)');
    return;
  }
  if (!supportsMemory64()) {
    t.skip('Current Node/V8 runtime does not support WebAssembly memory64');
    return;
  }

  const wasmBytes = fs.readFileSync(RUNTIME_WASM_PATH);
  const mod = await WebAssembly.compile(wasmBytes);

  const imports = WebAssembly.Module.imports(mod);
  const exportsList = WebAssembly.Module.exports(mod);

  assert.ok(
    imports.some((i) => i.module === 'env' && i.name === 'memory' && i.kind === 'memory'),
    'Expected runtime.wasm to import env.memory',
  );

  for (const fnName of REQUIRED_HOST_IMPORT_NAMES) {
    assert.ok(
      imports.some((i) => i.module === 'env' && i.name === fnName && i.kind === 'function'),
      `Expected runtime.wasm to import env.${fnName} as function`,
    );
  }

  const exportMap = new Map(exportsList.map((e) => [e.name, e.kind]));
  assert.equal(
    exportMap.get('main'),
    'function',
    `Expected runtime.wasm to export function 'main' (exports: ${Array.from(exportMap.keys()).join(', ')})`,
  );
  assert.equal(
    exportMap.get('playgta6_attach_httpfs_mailbox'),
    'function',
    `Expected runtime.wasm to export function 'playgta6_attach_httpfs_mailbox'`,
  );
});
