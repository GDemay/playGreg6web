import test from 'node:test';
import assert from 'node:assert/strict';

import CrashReporterDefault, {
  SCHEMA_VERSION,
  DEFAULT_MAX_LOGS,
  DEFAULT_MAX_ERRORS,
  DEFAULT_APP_VERSION,
  ERROR_CATEGORIES,
  BOOT_ERROR_CODES,
  CrashReporter,
  sanitizeString,
  sanitizeValue,
  classifyError,
  classifyBootError,
  formatGitHubIssueMarkdown,
} from '../web/telemetry/crash_report.js';
import { PerfMonitor } from '../web/telemetry/perf_monitor.js';

function createDeterministicClock(startMs = 1700000000000) {
  let currentMs = startMs;
  return {
    now: () => currentMs,
    advance(deltaMs) {
      currentMs += deltaMs;
      return currentMs;
    },
    set(valueMs) {
      currentMs = valueMs;
      return currentMs;
    },
  };
}

function createMockEventTarget() {
  const listeners = new Map();
  return {
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
    dispatchEvent(event) {
      const set = listeners.get(event.type);
      if (set) {
        for (const fn of [...set]) {
          fn(event);
        }
      }
      return true;
    },
    listenerCount(type) {
      return listeners.get(type)?.size ?? 0;
    },
  };
}

test('default export, constants, and constructor validation', () => {
  assert.equal(CrashReporterDefault, CrashReporter);
  assert.equal(SCHEMA_VERSION, 1);
  assert.equal(DEFAULT_MAX_LOGS, 200);
  assert.equal(DEFAULT_MAX_ERRORS, 32);
  assert.equal(DEFAULT_APP_VERSION, '0.1.0');
  assert.deepEqual(ERROR_CATEGORIES, [
    'wasm_trap',
    'webgpu_validation',
    'webgpu_device_lost',
    'worker_error',
    'io_error',
    'unhandled_rejection',
    'runtime',
  ]);

  const reporter = new CrashReporter();
  assert.equal(reporter.maxLogs, 200);
  assert.equal(reporter.maxErrors, 32);
  assert.equal(reporter.appVersion, '0.1.0');
  assert.equal(reporter.logCount, 0);
  assert.equal(reporter.errorCount, 0);
  assert.equal(reporter.isGlobalHandlersAttached, false);

  assert.throws(() => new CrashReporter(null), TypeError);
  assert.throws(() => new CrashReporter({ maxLogs: 0 }), RangeError);
  assert.throws(() => new CrashReporter({ maxLogs: 1.5 }), RangeError);
  assert.throws(() => new CrashReporter({ maxErrors: -1 }), RangeError);
  assert.throws(() => new CrashReporter({ now: 'not-a-fn' }), TypeError);
});

test('sanitizeString redacts Windows paths, POSIX paths, file:// URLs, query strings, and auth tokens', () => {
  // Windows drive paths (backslashes, forward slashes, quoted with spaces, UNC)
  const winOut = sanitizeString('Failed loading C:\\Users\\alice\\GTA6\\game.exe at startup');
  assert.ok(!winOut.includes('alice'));
  assert.ok(!winOut.includes('GTA6'));
  assert.ok(!winOut.includes('game.exe'));
  assert.equal(winOut, 'Failed loading [REDACTED_PATH] at startup');

  const winQuoted = sanitizeString('Opened "C:\\Users\\Alice Smith\\Documents\\title.xvd"');
  assert.ok(!winQuoted.includes('Alice'));
  assert.ok(!winQuoted.includes('title.xvd'));
  assert.equal(winQuoted, 'Opened "[REDACTED_PATH]"');

  const winForward = sanitizeString('path=D:/Games/GTA6/chunk.rpf');
  assert.equal(winForward, 'path=[REDACTED_PATH]');

  const unc = sanitizeString('Mirror \\\\fileserver\\private\\alice\\dump.bin failed');
  assert.ok(!unc.includes('fileserver'));
  assert.ok(!unc.includes('alice'));
  assert.equal(unc, 'Mirror [REDACTED_PATH] failed');

  // POSIX host paths
  const posixHome = sanitizeString('Cannot open /home/bob/secret for reading');
  assert.ok(!posixHome.includes('bob'));
  assert.ok(!posixHome.includes('secret'));
  assert.equal(posixHome, 'Cannot open [REDACTED_PATH] for reading');

  const posixUsers = sanitizeString('Read error in /Users/charlie/Desktop/game.bin');
  assert.ok(!posixUsers.includes('charlie'));
  assert.equal(posixUsers, 'Read error in [REDACTED_PATH]');

  const tilde = sanitizeString('Config missing at ~/secrets/keys.json');
  assert.ok(!tilde.includes('secrets'));
  assert.equal(tilde, 'Config missing at [REDACTED_PATH]');

  // file:// URLs (preserving stack frame :line:col)
  const fileUrlWin = sanitizeString('at boot (file:///C:/Users/alice/playgta6web/web/loader.js:42:10)');
  assert.ok(!fileUrlWin.includes('file://'));
  assert.ok(!fileUrlWin.includes('alice'));
  assert.equal(fileUrlWin, 'at boot ([REDACTED_FILE_URL]:42:10)');

  const fileUrlPosix = sanitizeString('Imported file:///home/bob/secret.mjs');
  assert.ok(!fileUrlPosix.includes('file://'));
  assert.ok(!fileUrlPosix.includes('bob'));
  assert.equal(fileUrlPosix, 'Imported [REDACTED_FILE_URL]');

  // Query strings (?token=...)
  const urlWithQuery = sanitizeString('GET http://localhost:8080/data/chunk.bin?token=abc&sig=999 failed');
  assert.ok(!urlWithQuery.includes('token=abc'));
  assert.ok(!urlWithQuery.includes('999'));
  assert.equal(urlWithQuery, 'GET http://localhost:8080/data/chunk.bin?[REDACTED] failed');

  const standaloneQuery = sanitizeString('Request params ?token=abc rejected');
  assert.ok(!standaloneQuery.includes('abc'));
  assert.equal(standaloneQuery, 'Request params ?[REDACTED] rejected');

  // Bearer & Basic auth tokens and credential key-value pairs
  const bearer = sanitizeString('Authorization: Bearer xyz');
  assert.ok(!bearer.includes('xyz'));
  assert.equal(bearer, 'Authorization: Bearer [REDACTED]');

  const rawBearer = sanitizeString('Rejected header Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig');
  assert.ok(!rawBearer.includes('eyJhbGci'));
  assert.equal(rawBearer, 'Rejected header Bearer [REDACTED]');

  const basic = sanitizeString('Basic dXNlcjpwYXNz');
  assert.ok(!basic.includes('dXNlcjpwYXNz'));
  assert.equal(basic, 'Basic [REDACTED]');

  const userinfo = sanitizeString('Proxy https://alice:hunter2@proxy.local/api');
  assert.ok(!userinfo.includes('alice'));
  assert.ok(!userinfo.includes('hunter2'));
  assert.equal(userinfo, 'Proxy https://[REDACTED]@proxy.local/api');

  const apiKey = sanitizeString('api_key=sk_live_secret123 password="my-password"');
  assert.ok(!apiKey.includes('sk_live_secret123'));
  assert.ok(!apiKey.includes('my-password'));

  // Non-sensitive app routes and normal prose stay intact
  assert.equal(sanitizeString('GET /api/status 200 OK'), 'GET /api/status 200 OK');
  assert.equal(sanitizeString('WebGPU frame #120 ready? Yes.'), 'WebGPU frame #120 ready? Yes.');

  // Idempotency (including full URLs with query strings)
  const once = sanitizeString(
    'C:\\Users\\alice\\GTA6\\game.exe http://localhost:8080/a?token=abc:14:5 ?token=abc Bearer xyz'
  );
  assert.equal(sanitizeString(once), once);

  // Null / undefined
  assert.equal(sanitizeString(null), '');
  assert.equal(sanitizeString(undefined), '');
});

test('bounded maxLogs and maxErrors ring eviction preserves chronological order', () => {
  const clock = createDeterministicClock(1700000000000);
  const reporter = new CrashReporter({ maxLogs: 3, maxErrors: 2, now: clock.now });

  for (let i = 1; i <= 5; i += 1) {
    clock.advance(10);
    reporter.recordLog('info', `worker/C:\\Users\\alice\\w${i}.js`, `Log #${i} /home/bob/f${i}`);
  }

  assert.equal(reporter.logCount, 3);
  const logs = reporter.getLogs();
  assert.equal(logs.length, 3);
  assert.deepEqual(
    logs.map((l) => l.message),
    ['Log #3 [REDACTED_PATH]', 'Log #4 [REDACTED_PATH]', 'Log #5 [REDACTED_PATH]']
  );
  assert.ok(logs.every((l) => !l.source.includes('alice')));
  assert.ok(logs[0].timestampMs < logs[1].timestampMs && logs[1].timestampMs < logs[2].timestampMs);

  for (let i = 1; i <= 4; i += 1) {
    clock.advance(25);
    reporter.recordError('runtime', new Error(`Boom #${i} ?token=secret${i}`));
  }

  assert.equal(reporter.errorCount, 2);
  const errors = reporter.getErrors();
  assert.equal(errors.length, 2);
  assert.deepEqual(
    errors.map((e) => e.message),
    ['Boom #3 ?[REDACTED]', 'Boom #4 ?[REDACTED]']
  );
  assert.ok(errors[0].timestampMs < errors[1].timestampMs);

  reporter.clear();
  assert.equal(reporter.logCount, 0);
  assert.equal(reporter.errorCount, 0);
});

test('Wasm trap, WebGPU validation/device-lost, worker, IO, and rejection classification', () => {
  const clock = createDeterministicClock(1700000100000);
  const reporter = new CrashReporter({ now: clock.now });

  // Explicit categories
  const e1 = reporter.recordError('wasm_trap', new WebAssembly.RuntimeError('unreachable'));
  assert.equal(e1.category, 'wasm_trap');
  assert.equal(e1.name, 'RuntimeError');
  assert.equal(e1.message, 'unreachable');

  const e2 = reporter.recordError('webgpu_validation', {
    name: 'GPUValidationError',
    message: 'WGSL bind group mismatch at /home/bob/shader.wgsl',
  });
  assert.equal(e2.category, 'webgpu_validation');
  assert.equal(e2.name, 'GPUValidationError');
  assert.ok(!e2.message.includes('bob'));

  const e3 = reporter.recordError('webgpu_device_lost', {
    reason: 'destroyed',
    message: 'Device lost after TDR',
  });
  assert.equal(e3.category, 'webgpu_device_lost');
  assert.equal(e3.name, 'GPUDeviceLostInfo');
  assert.equal(e3.context.reason, 'destroyed');

  const e4 = reporter.recordError('worker_error', 'loader worker crashed');
  assert.equal(e4.category, 'worker_error');

  const e5 = reporter.recordError('io_error', 'OPFS quota exceeded');
  assert.equal(e5.category, 'io_error');

  const e6 = reporter.recordError('unhandled_rejection', new Error('Promise failed'));
  assert.equal(e6.category, 'unhandled_rejection');

  const e7 = reporter.recordError('runtime', 'Generic fault');
  assert.equal(e7.category, 'runtime');

  // Auto-classification when category is omitted or unrecognized
  assert.equal(classifyError(new WebAssembly.RuntimeError('memory access out of bounds')), 'wasm_trap');
  assert.equal(classifyError(new WebAssembly.CompileError('invalid module')), 'wasm_trap');
  assert.equal(classifyError({ name: 'GPUValidationError', message: 'Invalid pipeline' }), 'webgpu_validation');
  assert.equal(classifyError({ reason: 'unknown', message: 'WebGPU device lost' }), 'webgpu_device_lost');
  assert.equal(classifyError('Dedicated worker postMessage failed'), 'worker_error');
  assert.equal(classifyError('HTTPFS range request failed'), 'io_error');
  assert.equal(classifyError('Something unexpected'), 'runtime');

  const autoWasm = reporter.recordError(null, new WebAssembly.RuntimeError('unreachable'));
  assert.equal(autoWasm.category, 'wasm_trap');
});

test('collectCapabilities safely fingerprints environment without leaking UA paths or cookies', () => {
  const mockGlobal = {
    crossOriginIsolated: true,
    SharedArrayBuffer: function FakeSAB() {},
    WebAssembly: {
      Memory: function FakeMemory(desc) {
        if (desc?.index !== 'i64') {
          throw new Error('unexpected');
        }
      },
    },
    AudioWorkletNode: function FakeAudioWorkletNode() {},
    document: { cookie: 'session=super-secret-cookie' },
  };

  const mockNav = {
    gpu: { requestAdapter: () => null },
    storage: { getDirectory: () => Promise.resolve({}) },
    getGamepads: () => [],
    hardwareConcurrency: 16,
    deviceMemory: 8,
    userAgent: 'Mozilla/5.0 (/Users/alice/custom-browser) Bearer secret-ua-token',
  };

  const reporter = new CrashReporter({
    globalObj: mockGlobal,
    navigatorObj: mockNav,
  });

  const caps = reporter.collectCapabilities();
  assert.deepEqual(caps, {
    crossOriginIsolated: true,
    sharedArrayBuffer: true,
    webgpu: true,
    wasm64: true,
    opfs: true,
    gamepad: true,
    audioWorklet: true,
    hardwareConcurrency: 16,
    deviceMemory: 8,
  });

  const serialized = JSON.stringify(caps);
  assert.ok(!serialized.includes('Mozilla'));
  assert.ok(!serialized.includes('alice'));
  assert.ok(!serialized.includes('cookie'));
  assert.ok(!serialized.includes('secret'));

  // Fallback when APIs are missing or throw
  const minimalReporter = new CrashReporter({
    globalObj: {
      crossOriginIsolated: false,
      WebAssembly: {
        Memory() {
          throw new TypeError('memory64 not supported');
        },
      },
    },
    navigatorObj: {},
  });
  assert.deepEqual(minimalReporter.collectCapabilities(), {
    crossOriginIsolated: false,
    sharedArrayBuffer: false,
    webgpu: false,
    wasm64: false,
    opfs: false,
    gamepad: false,
    audioWorklet: false,
    hardwareConcurrency: null,
    deviceMemory: null,
  });
});

test('buildReport and toJson recursively sanitize all fields including nested extra and Error.stack', () => {
  const clock = createDeterministicClock(1700000200000);
  const reporter = new CrashReporter({
    appVersion: '0.2.0-dev',
    now: clock.now,
    globalObj: { crossOriginIsolated: true, SharedArrayBuffer },
    navigatorObj: { hardwareConcurrency: 8 },
  });

  const err = new Error('Trap loading C:\\Users\\alice\\GTA6\\game.exe with Bearer topsecret123');
  err.stack = [
    'RuntimeError: Trap loading C:\\Users\\alice\\GTA6\\game.exe',
    '    at loadWasm (file:///C:/Users/alice/playgta6web/web/loader.js:88:12)',
    '    at fetchChunk (http://localhost:8080/data/boot.rpf?token=abc123:14:5)',
    '    at main (/home/bob/playgta6web/web/index.html:42:7)',
  ].join('\n');

  reporter.recordLog('warn', 'httpfs', 'Retrying /Users/charlie/cache/chunk.bin?token=secret');
  reporter.recordError('wasm_trap', err, {
    localPath: 'C:\\Users\\alice\\GTA6\\game.exe',
    headers: { Authorization: 'Bearer xyz987' },
    rawBytes: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
  });

  const cyclic = { name: 'cycle' };
  cyclic.self = cyclic;

  const nestedErr = new Error('Nested fail in /home/bob/secret');
  nestedErr.stack = 'Error: Nested fail in /home/bob/secret\n    at fn (file:///home/bob/app.js:9:3)';

  const report = reporter.buildReport({
    storageSummary: {
      quotaBytes: 1024 * 1024 * 1024,
      usageBytes: 4096,
      rootDir: '/home/bob/opfs',
    },
    serverStatus: {
      hasManifest: true,
      hasBootset: true,
      hasWasmBinary: false,
      shaderPackCount: 2,
      endpoint: 'http://localhost:8080/api/status?token=leakme',
    },
    extra: {
      deep: {
        winPath: 'C:\\Users\\alice\\GTA6\\game.exe',
        posixPath: '/home/bob/secret',
        fileUrl: 'file:///Users/alice/dump.bin',
        authHeader: 'Bearer ultra-secret',
        token: 'raw-secret-value',
        nestedError: nestedErr,
        bigCounter: 9007199254740993n,
        rawBuffer: new Uint8Array(64),
        list: ['C:\\Users\\alice\\x.bin', '?token=xyz'],
        cyclic,
      },
      'C:\\Users\\alice\\key-path': 'value',
    },
  });

  assert.equal(report.schemaVersion, 1);
  assert.equal(report.generatedAt, new Date(1700000200000).toISOString());
  assert.equal(report.appVersion, '0.2.0-dev');
  assert.equal(typeof report.capabilities, 'object');
  assert.equal(report.errors.length, 1);
  assert.equal(report.errors[0].category, 'wasm_trap');
  assert.equal(report.errors[0].timestamp, new Date(1700000200000).toISOString());
  assert.ok(report.errors[0].stack.includes('[REDACTED_FILE_URL]:88:12'));
  assert.ok(report.errors[0].stack.includes('?[REDACTED]:14:5'));
  assert.ok(report.errors[0].stack.includes('[REDACTED_PATH]:42:7'));
  assert.equal(report.extra.deep.nestedError.name, 'Error');
  assert.ok(report.extra.deep.nestedError.stack.includes('[REDACTED_FILE_URL]:9:3'));
  assert.equal(report.recentLogs.length, 1);
  assert.equal(report.perf, null);
  assert.equal(report.serverStatus.hasManifest, true);
  assert.equal(report.serverStatus.shaderPackCount, 2);
  assert.equal(report.extra.deep.bigCounter, '9007199254740993');
  assert.equal(report.extra.deep.token, '[REDACTED]');
  assert.deepEqual(report.extra.deep.rawBuffer, { type: 'Uint8Array', byteLength: 64 });
  assert.equal(report.extra.deep.cyclic.self, '[Circular]');
  assert.deepEqual(report.errors[0].context.rawBytes, { type: 'Uint8Array', byteLength: 4 });

  const json = reporter.toJson({
    storageSummary: { rootDir: '/home/bob/opfs' },
    serverStatus: { endpoint: 'http://localhost:8080/api/status?token=leakme' },
    extra: { nestedError: nestedErr, path: 'C:\\Users\\alice\\GTA6\\game.exe' },
  });

  // Verify valid JSON and zero leaked secrets/usernames/paths anywhere in the output.
  const parsed = JSON.parse(json);
  assert.equal(parsed.schemaVersion, 1);

  const combinedJson = JSON.stringify(report) + json;
  for (const forbidden of [
    'alice',
    'bob',
    'charlie',
    'GTA6\\game.exe',
    'file://',
    'topsecret123',
    'abc123',
    'xyz987',
    'ultra-secret',
    'raw-secret-value',
    'leakme',
    'key-path',
  ]) {
    assert.ok(!combinedJson.includes(forbidden), `Report leaked forbidden token: ${forbidden}`);
  }

  // Compact JSON option
  const compact = reporter.toJson({ pretty: false });
  assert.ok(!compact.includes('\n'));
});

test('attachGlobalHandlers and detachGlobalHandlers capture error & unhandledrejection events cleanly', () => {
  const clock = createDeterministicClock(1700000300000);
  const reporter = new CrashReporter({ now: clock.now });
  const target = createMockEventTarget();

  assert.throws(() => reporter.attachGlobalHandlers(null), TypeError);

  const detach = reporter.attachGlobalHandlers(target);
  assert.equal(reporter.isGlobalHandlersAttached, true);
  assert.equal(target.listenerCount('error'), 1);
  assert.equal(target.listenerCount('unhandledrejection'), 1);

  // Re-attaching does not duplicate listeners
  reporter.attachGlobalHandlers(target);
  assert.equal(target.listenerCount('error'), 1);
  assert.equal(target.listenerCount('unhandledrejection'), 1);

  const trapErr = new WebAssembly.RuntimeError('unreachable in C:\\Users\\alice\\mod.wasm');
  target.dispatchEvent({
    type: 'error',
    error: trapErr,
    message: trapErr.message,
    filename: 'file:///C:/Users/alice/web/loader.js',
    lineno: 77,
    colno: 14,
  });

  clock.advance(50);
  const rejErr = new Error('Fetch failed ?token=secret-rej');
  rejErr.stack = 'Error: Fetch failed ?token=secret-rej\n    at /home/bob/web/io.js:12:4';
  target.dispatchEvent({
    type: 'unhandledrejection',
    reason: rejErr,
  });

  assert.equal(reporter.errorCount, 2);
  const [e1, e2] = reporter.getErrors();
  assert.equal(e1.category, 'wasm_trap');
  assert.ok(!e1.message.includes('alice'));
  assert.equal(e1.context.filename, '[REDACTED_FILE_URL]');
  assert.equal(e1.context.lineno, 77);
  assert.equal(e1.context.colno, 14);

  assert.equal(e2.category, 'unhandled_rejection');
  assert.ok(!e2.message.includes('secret-rej'));
  assert.ok(!e2.stack.includes('bob'));

  // Detach via returned function and verify idempotency + no further captures
  detach();
  assert.equal(reporter.isGlobalHandlersAttached, false);
  assert.equal(target.listenerCount('error'), 0);
  assert.equal(target.listenerCount('unhandledrejection'), 0);

  target.dispatchEvent({ type: 'error', message: 'should be ignored' });
  target.dispatchEvent({ type: 'unhandledrejection', reason: 'ignored too' });
  assert.equal(reporter.errorCount, 2);

  // Calling detachGlobalHandlers again is a safe no-op
  reporter.detachGlobalHandlers();
  assert.equal(reporter.isGlobalHandlersAttached, false);
});

test('seamless integration with PerfMonitor.getSnapshot()', () => {
  const clock = createDeterministicClock(1000);
  const perf = new PerfMonitor({ windowSize: 10, now: clock.now });

  perf.recordFrameTimeMs(16.6);
  perf.recordFrameTimeMs(16.7);
  perf.recordFrameTimeMs(40.0); // jank frame
  perf.recordGpuBatch({ packets: 25, drawCalls: 12, computeDispatches: 2, bytes: 4096, durationMs: 16.6 });
  perf.recordIoRead({ bytes: 1024 * 1024, opfsHit: true, durationMs: 50 });
  perf.updateAudioTelemetry({ bufferedFrames: 2048, underruns: 1, underrunFrames: 128, activeVoices: 6 });

  const reporter = new CrashReporter({ now: () => 1700000400000 });
  const snap = perf.getSnapshot();
  const reportFromSnap = reporter.buildReport({ perfSnapshot: snap });
  const reportFromInstance = reporter.buildReport({ perfSnapshot: perf });

  assert.equal(reportFromSnap.perf.frames.totalFrames, 3);
  assert.equal(reportFromSnap.perf.frames.jankFrames, 1);
  assert.equal(reportFromSnap.perf.gpu.totalGpuPackets, 25);
  assert.equal(reportFromSnap.perf.gpu.totalDrawCalls, 12);
  assert.equal(reportFromSnap.perf.io.opfsCacheHits, 1);
  assert.equal(reportFromSnap.perf.audio.bufferedFrames, 2048);
  assert.equal(reportFromSnap.perf.audio.activeVoices, 6);

  assert.deepEqual(reportFromInstance.perf, reportFromSnap.perf);
});

test('sanitizeValue handles Maps, Sets, Dates, non-finite numbers, and deeply nested objects', () => {
  const m = new Map([
    ['C:\\Users\\alice\\file.txt', 'ok'],
    ['token', 'should-redact'],
    ['normal', '/home/bob/data'],
  ]);
  const s = new Set(['Bearer xyz', 42, Number.NaN, Number.POSITIVE_INFINITY]);

  const sanitized = sanitizeValue({
    map: m,
    set: s,
    when: new Date(1700000000000),
    invalidDate: new Date('invalid'),
  });

  assert.deepEqual(sanitized.map, {
    '[REDACTED_PATH]': 'ok',
    token: '[REDACTED]',
    normal: '[REDACTED_PATH]',
  });
  assert.deepEqual(sanitized.set, ['Bearer [REDACTED]', 42, null, null]);
  assert.equal(sanitized.when, new Date(1700000000000).toISOString());
  assert.equal(sanitized.invalidDate, null);
});

test('classifyBootError classifies all Wasm64 browser boot failure categories with actionable hints', () => {
  assert.deepEqual(BOOT_ERROR_CODES, {
    WASM_MODULE_TOO_LARGE: 'WASM_MODULE_TOO_LARGE',
    WASM64_UNSUPPORTED: 'WASM64_UNSUPPORTED',
    COI_SAB_MISSING: 'COI_SAB_MISSING',
    WORKER_SCRIPT_COLLISION: 'WORKER_SCRIPT_COLLISION',
    HTTPFS_PGIM_ERROR: 'HTTPFS_PGIM_ERROR',
    UNKNOWN_BOOT_ERROR: 'UNKNOWN_BOOT_ERROR',
  });

  // 1. V8 1 GiB Wasm module size limit (both error variants)
  const tooLarge1 = classifyBootError(
    new WebAssembly.CompileError('WebAssembly.compile(): Buffer source exceeds maximum size of 1073741824')
  );
  assert.equal(tooLarge1.code, 'WASM_MODULE_TOO_LARGE');
  assert.equal(tooLarge1.category, 'wasm_compile');
  assert.ok(tooLarge1.hint.includes('--opt Oz --no-link-opt --strip'));

  const tooLarge2 = classifyBootError('RangeError: buffer source exceeds maximum size');
  assert.equal(tooLarge2.code, 'WASM_MODULE_TOO_LARGE');
  assert.equal(tooLarge2.category, 'wasm_compile');
  assert.ok(tooLarge2.hint.includes('--opt Oz --no-link-opt --strip'));

  // 2. Wasm64 / memory64 unsupported
  const wasm64a = classifyBootError(new TypeError('WebAssembly.Memory(): memory64 not supported'));
  assert.equal(wasm64a.code, 'WASM64_UNSUPPORTED');
  assert.equal(wasm64a.category, 'capability');
  assert.ok(wasm64a.hint.length > 0);

  const wasm64b = classifyBootError('Wasm64 is unsupported in this browser');
  assert.equal(wasm64b.code, 'WASM64_UNSUPPORTED');
  assert.equal(wasm64b.category, 'capability');

  const wasm64c = classifyBootError("TypeError: WebAssembly.Memory(): invalid address type 'i64'");
  assert.equal(wasm64c.code, 'WASM64_UNSUPPORTED');
  assert.equal(wasm64c.category, 'capability');

  // 3. Cross-origin isolation / SharedArrayBuffer missing
  const sab1 = classifyBootError(new ReferenceError('SharedArrayBuffer is not defined'));
  assert.equal(sab1.code, 'COI_SAB_MISSING');
  assert.equal(sab1.category, 'capability');
  assert.ok(sab1.hint.length > 0);

  const sab2 = classifyBootError('crossOriginIsolated is false; COOP/COEP headers missing');
  assert.equal(sab2.code, 'COI_SAB_MISSING');
  assert.equal(sab2.category, 'capability');

  // 4. Classic worker importScripts global identifier collision
  const collision1 = classifyBootError(
    new SyntaxError("Identifier 'OPCODES' has already been declared")
  );
  assert.equal(collision1.code, 'WORKER_SCRIPT_COLLISION');
  assert.equal(collision1.category, 'worker');
  assert.ok(collision1.hint.length > 0);

  const collision2 = classifyBootError('SyntaxError: redeclaration of const HEADER_BYTES');
  assert.equal(collision2.code, 'WORKER_SCRIPT_COLLISION');
  assert.equal(collision2.category, 'worker');

  // 5. HTTPFS / PGIM load error
  const httpfs1 = classifyBootError(new Error('HTTPFS range request failed (HTTP 404)'));
  assert.equal(httpfs1.code, 'HTTPFS_PGIM_ERROR');
  assert.equal(httpfs1.category, 'io');
  assert.ok(httpfs1.hint.length > 0);

  const pgim1 = classifyBootError('PGIM magic mismatch while mapping boot image');
  assert.equal(pgim1.code, 'HTTPFS_PGIM_ERROR');
  assert.equal(pgim1.category, 'io');

  // 6. Generic fallback
  const unknown1 = classifyBootError(new Error('Something completely unexpected happened'));
  assert.equal(unknown1.code, 'UNKNOWN_BOOT_ERROR');
  assert.equal(unknown1.category, 'runtime');
  assert.ok(unknown1.hint.length > 0);

  const unknownNull = classifyBootError(null);
  assert.equal(unknownNull.code, 'UNKNOWN_BOOT_ERROR');
  assert.equal(unknownNull.category, 'runtime');
  assert.equal(unknownNull.message, '');
});

test('classifyBootError strips host paths, drive letters, usernames, and tokens from messages', () => {
  const winTooLarge = classifyBootError(
    new Error(
      'Failed compiling C:\\Users\\alice\\GTA6\\guest.wasm?token=secret123: Buffer source exceeds maximum size of 1073741824 (Bearer eyJsecret)'
    )
  );
  assert.equal(winTooLarge.code, 'WASM_MODULE_TOO_LARGE');
  assert.equal(winTooLarge.category, 'wasm_compile');
  for (const leaked of ['alice', 'C:\\', 'GTA6', 'guest.wasm', 'secret123', 'eyJsecret']) {
    assert.ok(!winTooLarge.message.includes(leaked), `Leaked "${leaked}" in: ${winTooLarge.message}`);
  }
  assert.ok(winTooLarge.message.includes('[REDACTED_PATH]'));
  assert.ok(winTooLarge.message.includes('Bearer [REDACTED]'));

  const posixCollision = classifyBootError(
    "Identifier 'OPCODES' has already been declared in /Users/charlie/work/worker.js and /home/bob/helper.js?api_key=xyz999"
  );
  assert.equal(posixCollision.code, 'WORKER_SCRIPT_COLLISION');
  for (const leaked of ['charlie', 'bob', '/Users/', '/home/', 'xyz999']) {
    assert.ok(!posixCollision.message.includes(leaked), `Leaked "${leaked}" in: ${posixCollision.message}`);
  }

  // Host path ending in .pgim still classifies as HTTPFS_PGIM_ERROR while redacting the path
  const pgimHostPath = classifyBootError(
    'Failed to read D:/Users/alice/GTA6/boot.pgim (Authorization: Bearer supersecret)'
  );
  assert.equal(pgimHostPath.code, 'HTTPFS_PGIM_ERROR');
  assert.equal(pgimHostPath.category, 'io');
  for (const leaked of ['alice', 'D:/', 'GTA6', 'boot.pgim', 'supersecret']) {
    assert.ok(!pgimHostPath.message.includes(leaked), `Leaked "${leaked}" in: ${pgimHostPath.message}`);
  }
});

test('setBootSnapshot and setGpuTelemetry enforce strict allowlisting and export bootDiagnostics in JSON bundles', () => {
  const clock = createDeterministicClock(1700000500000);
  const reporter = new CrashReporter({
    appVersion: '0.3.0-test',
    now: clock.now,
    globalObj: { crossOriginIsolated: true, SharedArrayBuffer },
    navigatorObj: { hardwareConcurrency: 12 },
  });

  // Record multiple boot errors (including a host-path .pgim error and V8 1 GiB limit error)
  reporter.recordError(
    'wasm_trap',
    new WebAssembly.CompileError(
      'WebAssembly.compile() of C:\\Users\\alice\\guest.wasm: Buffer source exceeds maximum size of 1073741824'
    )
  );
  clock.advance(20);
  reporter.recordError(
    'io_error',
    new Error('Failed loading /home/bob/game/boot.pgim?token=leaked-pgim-token')
  );

  // Attach boot snapshot with both allowlisted fields and disallowed host paths / raw bytes / secrets
  const storedBoot = reporter.setBootSnapshot({
    stage: 'PGIM_LOAD',
    mode: 'real-wasm',
    progressPct: 14.5,
    readyForSplash: false,
    elapsedMs: 412.5,
    milestonesReached: 1,
    wasmModuleBytes: 1200000000,
    pgimBytes: 536870912,
    pgimSegments: 4,
    trapCount: 1,
    uniqueTrapCount: 1,
    milestones: [
      {
        milestone: 'PGIM_LOAD',
        index: 0,
        reached: true,
        elapsedMs: 120,
        lastElapsedMs: 120,
        count: 1,
        detail: {
          source: 'boot',
          line: 'Mapped C:\\Users\\alice\\GTA6\\boot.pgim?token=sec',
          disallowedNested: 'should-drop',
        },
        secretBytes: new Uint8Array([1, 2, 3, 4]),
      },
    ],
    lastTrap: {
      seq: 1,
      kind: 'UNRESOLVED_IMPORT',
      guestVa: '0x140001234',
      dll: 'kernel32.dll',
      symbol: 'CreateFileW',
      import: 'kernel32.dll!CreateFileW',
      message: 'Unresolved call from /Users/charlie/GTA6/game.exe Bearer tok999',
      fatal: false,
      elapsedMs: 390,
      stage: 'PGIM_LOAD',
      count: 2,
      rawDump: new Uint8Array([0xde, 0xad, 0xbe, 0xef]),
      hostPath: 'C:\\Users\\alice\\secret.bin',
    },
    unresolvedImports: [
      {
        name: 'kernel32.dll!CreateFileW',
        dll: 'kernel32.dll',
        symbol: 'CreateFileW',
        ordinal: null,
        guestVa: '0x140001234',
        count: 2,
        firstElapsedMs: 380,
        lastElapsedMs: 390,
        extraLeak: 'C:\\Users\\alice\\leak',
      },
    ],
    // Disallowed top-level keys that MUST be dropped completely
    hostPath: 'C:\\Users\\alice\\GTA6',
    localGameDir: '/home/bob/gta6',
    rawBytes: new Uint8Array([0xaa, 0xbb, 0xcc]),
    wasmBinary: new ArrayBuffer(32),
    secretToken: 'top-secret-boot-token',
  });

  assert.equal(storedBoot.stage, 'PGIM_LOAD');
  assert.equal(storedBoot.mode, 'real-wasm');
  assert.equal(storedBoot.progressPct, 14.5);
  assert.equal(storedBoot.wasmModuleBytes, 1200000000);
  assert.equal(storedBoot.pgimSegments, 4);
  assert.equal('hostPath' in storedBoot, false);
  assert.equal('localGameDir' in storedBoot, false);
  assert.equal('rawBytes' in storedBoot, false);
  assert.equal('wasmBinary' in storedBoot, false);
  assert.equal('secretToken' in storedBoot, false);
  assert.equal('rawDump' in storedBoot.lastTrap, false);
  assert.equal('hostPath' in storedBoot.lastTrap, false);
  assert.equal('disallowedNested' in storedBoot.milestones[0].detail, false);
  assert.equal('secretBytes' in storedBoot.milestones[0], false);
  assert.equal('extraLeak' in storedBoot.unresolvedImports[0], false);
  assert.ok(!storedBoot.lastTrap.message.includes('charlie'));
  assert.ok(!storedBoot.lastTrap.message.includes('tok999'));

  // Attach GPU telemetry with both allowlisted fields and disallowed shader source / raw bytes / paths
  const storedGpu = reporter.setGpuTelemetry({
    timestamp: 1700000500020,
    adapter: 'nvidia blackwell /home/bob/driver',
    backend: 'webgpu',
    deviceLost: false,
    validationErrors: 1,
    shaderPackCount: 2,
    frameStats: {
      framesPresented: 3,
      droppedFrames: 0,
      drawCalls: 42,
      computeDispatches: 5,
      packetsProcessed: 120,
      width: 1280,
      height: 720,
      rawFrameBuffer: new Uint8Array([9, 8, 7]),
      secretKey: 'drop-me',
    },
    fallbackStats: {
      packHits: 18,
      fallbackHits: 2,
      builtinHits: 1,
      totalLookups: 21,
      uniqueMissing: 1,
      overflowMissingCount: 0,
      hitRatePct: 90.5,
      leakedPath: 'C:\\Users\\alice\\shaders',
    },
    missingShaders: [
      {
        hash: 0x123456789abcdef0n,
        stage: 'fragment',
        hits: 4,
        firstSeenFrame: 1,
        wgsl: '@fragment fn main() {}',
        rawSpirv: new Uint8Array([0x03, 0x02, 0x23, 0x07]),
        sourcePath: 'C:\\Users\\alice\\shader.hlsl',
      },
    ],
    // Disallowed top-level keys
    wgslSource: 'secret shader code',
    commandBuffer: new ArrayBuffer(128),
    cacheDir: '/Users/charlie/gpu-cache',
  });

  assert.equal(storedGpu.backend, 'webgpu');
  assert.equal(storedGpu.validationErrors, 1);
  assert.ok(!storedGpu.adapter.includes('bob'));
  assert.equal(storedGpu.frameStats.framesPresented, 3);
  assert.equal(storedGpu.frameStats.drawCalls, 42);
  assert.equal('rawFrameBuffer' in storedGpu.frameStats, false);
  assert.equal('secretKey' in storedGpu.frameStats, false);
  assert.equal(storedGpu.fallbackStats.hitRatePct, 90.5);
  assert.equal('leakedPath' in storedGpu.fallbackStats, false);
  assert.deepEqual(storedGpu.missingShaders, [
    { hash: '0x123456789abcdef0', stage: 'fragment', hits: 4, firstSeenFrame: 1 },
  ]);
  assert.equal('wgslSource' in storedGpu, false);
  assert.equal('commandBuffer' in storedGpu, false);
  assert.equal('cacheDir' in storedGpu, false);

  // Verify buildReport() and toJson() include bootDiagnostics while preserving all legacy fields
  const report = reporter.buildReport({
    serverStatus: { hasWasmBinary: true },
  });
  assert.equal(report.schemaVersion, SCHEMA_VERSION);
  assert.equal(report.appVersion, '0.3.0-test');
  assert.equal(report.errors.length, 2);
  assert.equal(report.serverStatus.hasWasmBinary, true);

  assert.ok(report.bootDiagnostics && typeof report.bootDiagnostics === 'object');
  assert.equal(report.bootDiagnostics.classifiedErrors.length, 2);
  assert.equal(report.bootDiagnostics.classifiedErrors[0].code, 'WASM_MODULE_TOO_LARGE');
  assert.equal(report.bootDiagnostics.classifiedErrors[0].category, 'wasm_compile');
  assert.ok(report.bootDiagnostics.classifiedErrors[0].hint.includes('--opt Oz --no-link-opt --strip'));
  assert.equal(report.bootDiagnostics.classifiedErrors[1].code, 'HTTPFS_PGIM_ERROR');
  assert.equal(report.bootDiagnostics.classifiedErrors[1].category, 'io');
  assert.deepEqual(report.bootDiagnostics.errorCodeCounts, {
    WASM_MODULE_TOO_LARGE: 1,
    HTTPFS_PGIM_ERROR: 1,
  });
  assert.equal(report.bootDiagnostics.bootSnapshot.stage, 'PGIM_LOAD');
  assert.equal(report.bootDiagnostics.gpuTelemetry.frameStats.framesPresented, 3);

  const json = reporter.toJson();
  const parsed = JSON.parse(json);
  assert.equal(parsed.bootDiagnostics.classifiedErrors.length, 2);
  assert.equal(parsed.bootDiagnostics.bootSnapshot.stage, 'PGIM_LOAD');
  assert.equal(parsed.bootDiagnostics.gpuTelemetry.missingShaders[0].hash, '0x123456789abcdef0');

  for (const forbidden of [
    'alice',
    'bob',
    'charlie',
    'leaked-pgim-token',
    'top-secret-boot-token',
    'secret shader code',
    '@fragment',
    'tok999',
    'drop-me',
    'should-drop',
    'shader.hlsl',
    'gpu-cache',
  ]) {
    assert.ok(!json.includes(forbidden), `Exported JSON leaked forbidden token: ${forbidden}`);
  }

  // Duck-typed BootTracer instance support via getSummary()
  const tracerLike = {
    getSummary() {
      return {
        stage: 'FIRST_PRESENT',
        progressPct: 100,
        readyForSplash: true,
        hostPath: 'C:\\Users\\alice\\should-be-dropped',
      };
    },
  };
  const fromTracer = reporter.setBootSnapshot(tracerLike);
  assert.equal(fromTracer.stage, 'FIRST_PRESENT');
  assert.equal(fromTracer.readyForSplash, true);
  assert.equal('hostPath' in fromTracer, false);

  // Passing raw binary buffers directly to setters returns null
  assert.equal(reporter.setBootSnapshot(new Uint8Array([1, 2, 3])), null);
  assert.equal(reporter.setGpuTelemetry(new ArrayBuffer(16)), null);
  assert.equal(reporter.getBootSnapshot(), null);
  assert.equal(reporter.getGpuTelemetry(), null);

  // Per-call overrides in buildReport work without mutating stored state
  const overrideReport = reporter.buildReport({
    bootSnapshot: { stage: 'D3D12_CREATE_DEVICE', progressPct: 77, secret: 'nope' },
    gpuTelemetry: { backend: 'webgpu', framesPresented: 1, rawBytes: new Uint8Array(8) },
  });
  assert.equal(overrideReport.bootDiagnostics.bootSnapshot.stage, 'D3D12_CREATE_DEVICE');
  assert.equal('secret' in overrideReport.bootDiagnostics.bootSnapshot, false);
  assert.equal(overrideReport.bootDiagnostics.gpuTelemetry.framesPresented, 1);
  assert.equal('rawBytes' in overrideReport.bootDiagnostics.gpuTelemetry, false);
  assert.equal(reporter.getBootSnapshot(), null);
  assert.equal(reporter.getGpuTelemetry(), null);

  // Clearing via clear() resets attached snapshots too
  reporter.setBootSnapshot({ stage: 'TLS_INIT' });
  reporter.setGpuTelemetry({ backend: 'webgpu' });
  reporter.clear();
  assert.equal(reporter.getBootSnapshot(), null);
  assert.equal(reporter.getGpuTelemetry(), null);
});

test('W152 coalesces consecutive and non-consecutive duplicate (category, message) errors without evicting earlier root causes', () => {
  const clock = createDeterministicClock(1700001000000);
  const reporter = new CrashReporter({ maxErrors: 3, now: clock.now });

  // Initial zero stats
  assert.equal(reporter.recordedErrorCount, 0);
  assert.equal(reporter.coalescedCount, 0);
  assert.deepEqual(reporter.categoryCounts, {});
  assert.deepEqual(reporter.getErrorStats(), {
    recordedCount: 0,
    bufferedCount: 0,
    coalescedCount: 0,
    droppedCount: 0,
    maxErrors: 3,
    categoryCounts: {},
  });

  // 1. Record two early root-cause errors.
  const root1 = reporter.recordError('wasm_trap', new WebAssembly.RuntimeError('null indirect call'), {
    stage: 'TLS_INIT',
  });
  assert.equal(root1.occurrences, 1);
  assert.equal(root1.firstSeenAtMs, 1700001000000);
  assert.equal(root1.lastSeenAtMs, 1700001000000);

  clock.advance(50);
  const root2 = reporter.recordError('io_error', 'HTTPFS range request returned 503', {
    chunk: 4,
  });
  assert.equal(root2.occurrences, 1);
  assert.equal(root2.firstSeenAtMs, 1700001000050);
  assert.equal(root2.lastSeenAtMs, 1700001000050);

  // 2. Flood 250 identical WebGPU validation errors — earlier root causes must survive.
  for (let i = 0; i < 250; i += 1) {
    clock.advance(16);
    reporter.recordError('webgpu_validation', {
      name: 'GPUValidationError',
      message: 'Bind group layout mismatch at slot 0',
    });
  }

  // 3. Non-consecutive repeat of root1 (even via auto-classification) coalesces into existing root1 entry.
  clock.advance(100);
  const root1Repeat = reporter.recordError(null, new WebAssembly.RuntimeError('null indirect call'), {
    stage: 'LATER_STAGE',
  });
  assert.equal(root1Repeat.occurrences, 2);
  assert.equal(root1Repeat.firstSeenAtMs, 1700001000000);
  assert.equal(root1Repeat.lastSeenAtMs, clock.now());
  // Root-cause context from first occurrence is preserved.
  assert.deepEqual(root1Repeat.context, { stage: 'TLS_INIT' });

  assert.equal(reporter.errorCount, 3);
  assert.equal(reporter.recordedErrorCount, 253);
  assert.equal(reporter.coalescedCount, 250);
  assert.deepEqual(reporter.categoryCounts, {
    wasm_trap: 2,
    io_error: 1,
    webgpu_validation: 250,
  });

  const errors = reporter.getErrors();
  assert.equal(errors.length, 3);
  // Chronological first-seen order preserved.
  assert.equal(errors[0].message, 'null indirect call');
  assert.equal(errors[0].occurrences, 2);
  assert.equal(errors[0].timestampMs, 1700001000000);
  assert.equal(errors[0].firstSeenAtMs, 1700001000000);
  assert.equal(errors[0].lastSeenAtMs, clock.now());

  assert.equal(errors[1].message, 'HTTPFS range request returned 503');
  assert.equal(errors[1].occurrences, 1);

  assert.equal(errors[2].category, 'webgpu_validation');
  assert.equal(errors[2].message, 'Bind group layout mismatch at slot 0');
  assert.equal(errors[2].occurrences, 250);
  assert.equal(errors[2].firstSeenAtMs, 1700001000050 + 16);
  assert.equal(errors[2].lastSeenAtMs, 1700001000050 + 16 * 250);

  // Additive stats exposed in buildReport() and toJson() alongside bootDiagnostics.
  const report = reporter.buildReport();
  assert.deepEqual(report.errorStats, {
    recordedCount: 253,
    bufferedCount: 3,
    coalescedCount: 250,
    droppedCount: 0,
    maxErrors: 3,
    categoryCounts: {
      wasm_trap: 2,
      io_error: 1,
      webgpu_validation: 250,
    },
  });
  assert.equal(report.errors.length, 3);
  assert.equal(report.errors[2].occurrences, 250);
  assert.equal(report.bootDiagnostics.classifiedErrors[1].code, 'HTTPFS_PGIM_ERROR');
  assert.equal(report.bootDiagnostics.classifiedErrors[1].occurrences, 1);
  assert.equal(report.bootDiagnostics.errorCodeCounts.HTTPFS_PGIM_ERROR, 1);
  assert.equal(report.bootDiagnostics.errorCodeCounts.UNKNOWN_BOOT_ERROR, 252);

  const parsed = JSON.parse(reporter.toJson());
  assert.deepEqual(parsed.errorStats, report.errorStats);
});

test('W152 separates distinct categories with same message, handles eviction + re-entry, redaction, immutability, and clear()', () => {
  const clock = createDeterministicClock(1700002000000);
  const reporter = new CrashReporter({ maxErrors: 2, now: clock.now });

  // Same message across different categories must NOT coalesce.
  reporter.recordError('io_error', 'request timed out');
  clock.advance(10);
  reporter.recordError('worker_error', 'request timed out');
  assert.equal(reporter.errorCount, 2);
  assert.equal(reporter.recordedErrorCount, 2);
  assert.equal(reporter.coalescedCount, 0);
  assert.deepEqual(reporter.categoryCounts, { io_error: 1, worker_error: 1 });

  // Pushing a 3rd distinct error evicts the oldest ('io_error', 'request timed out').
  clock.advance(10);
  reporter.recordError('runtime', 'Failed opening C:\\Users\\alice\\GTA6\\a.bin?token=sec1');
  assert.equal(reporter.errorCount, 2);
  assert.equal(reporter.getErrorStats().droppedCount, 1);

  // Different raw host path & secret token that sanitize to the exact same message MUST coalesce cleanly without leaking.
  clock.advance(20);
  const coalescedSanitized = reporter.recordError(
    'runtime',
    'Failed opening C:\\Users\\bob\\GTA6\\b.bin?token=sec2',
    { rawDump: new Uint8Array([1, 2, 3, 4]), token: 'must-not-leak' }
  );
  assert.equal(coalescedSanitized.occurrences, 2);
  assert.equal(coalescedSanitized.message, 'Failed opening [REDACTED_PATH]?[REDACTED]');
  assert.equal(reporter.coalescedCount, 1);

  // Re-recording the previously evicted error creates a fresh entry (occurrences: 1) rather than resurrecting evicted state.
  clock.advance(30);
  const reentered = reporter.recordError('io_error', 'request timed out');
  assert.equal(reentered.occurrences, 1);
  assert.equal(reentered.firstSeenAtMs, clock.now());
  assert.equal(reentered.lastSeenAtMs, clock.now());

  const stats = reporter.getErrorStats();
  assert.equal(stats.recordedCount, 5);
  assert.equal(stats.bufferedCount, 2);
  assert.equal(stats.coalescedCount, 1);
  assert.equal(stats.droppedCount, 2);
  assert.equal(stats.bufferedCount + stats.droppedCount + stats.coalescedCount, stats.recordedCount);
  assert.deepEqual(stats.categoryCounts, {
    io_error: 2,
    worker_error: 1,
    runtime: 2,
  });

  // Returned stats and error snapshots are defensively cloned.
  stats.categoryCounts.io_error = 9999;
  const catCopy = reporter.categoryCounts;
  catCopy.runtime = 9999;
  const errSnapshot = reporter.getErrors();
  errSnapshot[0].occurrences = 9999;
  assert.equal(reporter.categoryCounts.io_error, 2);
  assert.equal(reporter.categoryCounts.runtime, 2);
  assert.equal(reporter.getErrors()[0].occurrences, 2);

  // Exported JSON leaks zero paths, usernames, or tokens.
  const json = reporter.toJson();
  for (const forbidden of ['alice', 'bob', 'GTA6', 'sec1', 'sec2', 'must-not-leak']) {
    assert.ok(!json.includes(forbidden), `Leaked forbidden token: ${forbidden}`);
  }

  // clear() resets ring, coalescing index, and all counters.
  reporter.clear();
  assert.equal(reporter.errorCount, 0);
  assert.equal(reporter.recordedErrorCount, 0);
  assert.equal(reporter.coalescedCount, 0);
  assert.deepEqual(reporter.categoryCounts, {});
  assert.deepEqual(reporter.getErrorStats(), {
    recordedCount: 0,
    bufferedCount: 0,
    coalescedCount: 0,
    droppedCount: 0,
    maxErrors: 2,
    categoryCounts: {},
  });

  // After clear(), recording a previously-seen error starts fresh with occurrences === 1.
  clock.advance(10);
  const freshAfterClear = reporter.recordError('io_error', 'request timed out');
  assert.equal(freshAfterClear.occurrences, 1);
  assert.equal(reporter.recordedErrorCount, 1);
  assert.equal(reporter.coalescedCount, 0);

  // coalesceDuplicates option validation & opt-out mode.
  assert.throws(() => new CrashReporter({ coalesceDuplicates: 'yes' }), TypeError);
  const rawReporter = new CrashReporter({ maxErrors: 2, coalesceDuplicates: false, now: clock.now });
  rawReporter.recordError('runtime', 'dup');
  rawReporter.recordError('runtime', 'dup');
  assert.equal(rawReporter.errorCount, 2);
  assert.equal(rawReporter.recordedErrorCount, 2);
  assert.equal(rawReporter.coalescedCount, 0);
  assert.deepEqual(rawReporter.categoryCounts, { runtime: 2 });
});

test('W220 aggressively redacts extended host paths, query/token secrets, and raw binary payloads while preserving VAs, offsets, and hashes', () => {
  // Extended Windows device/UNC paths, MSYS/WSL paths, and unclosed quoted POSIX paths
  const extWin = sanitizeString('Open failed on \\\\?\\C:\\Users\\alice\\GTA6\\game.exe and \\\\.\\D:\\vault\\dump.bin');
  assert.ok(!extWin.includes('alice'));
  assert.ok(!extWin.includes('GTA6'));
  assert.ok(!extWin.includes('vault'));
  assert.equal(extWin, 'Open failed on [REDACTED_PATH] and [REDACTED_PATH]');

  const msysAndPosix = sanitizeString(
    'Failed in /c/Users/alice/GTA6/mod.wasm and "/home/bob/secret.rpf and /cygdrive/d/Games/x.bin'
  );
  for (const leaked of ['alice', 'bob', 'GTA6', 'secret.rpf', 'Games']) {
    assert.ok(!msysAndPosix.includes(leaked), `Leaked "${leaked}" in: ${msysAndPosix}`);
  }

  // Query secrets, JSON key/value secrets, AWS/GitHub/JWT bare tokens
  const jwt = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c';
  const secretsText = sanitizeString(
    `GET https://cdn.local/chunk.bin?X-Amz-Signature=deadbeef&token=tok123 ` +
      `body={"api_key":"sk_live_9999999999999999","session_token":"sess_abc"} ` +
      `bare=${jwt} pat=ghp_1234567890abcdef1234567890abcdef aws=AKIAIOSFODNN7EXAMPLE`
  );
  for (const leaked of [
    'deadbeef',
    'tok123',
    'sk_live_9999999999999999',
    'sess_abc',
    'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9',
    'SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c',
    'ghp_1234567890abcdef1234567890abcdef',
    'AKIAIOSFODNN7EXAMPLE',
  ]) {
    assert.ok(!secretsText.includes(leaked), `Leaked secret "${leaked}" in: ${secretsText}`);
  }

  // Raw binary payloads: data URI base64, multi-byte spaced hex dump, >64-char hex blob, raw Wasm NUL string
  const dataUri = sanitizeString('Image payload data:application/octet-stream;base64,AGFzbQEAAAABBAFgAA== leaked');
  assert.ok(!dataUri.includes('AGFzbQEAAAABBAFgAA=='));
  assert.ok(dataUri.includes('[REDACTED_BUFFER]'));

  const hexDump = sanitizeString(
    'Packet dump: de ad be ef 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff end'
  );
  assert.ok(!hexDump.includes('de ad be ef'));
  assert.equal(hexDump, 'Packet dump: [REDACTED_BUFFER] end');

  const longHexBlob = sanitizeString(`Blob: 0x${'ab'.repeat(40)} done`);
  assert.ok(!longHexBlob.includes('abababababab'));
  assert.equal(longHexBlob, 'Blob: [REDACTED_BUFFER] done');

  const rawWasmBytesStr = sanitizeString('\x00asm\x01\x00\x00\x00\x01\x05\x01\x60\x00\x00');
  assert.equal(rawWasmBytesStr, '[REDACTED_BUFFER]');

  // Exact preservation of trap VAs, instruction offsets, 64-bit & SHA-256 shader hashes, and app routes
  const sha256Hex = '0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef';
  const preserved = sanitizeString(
    `Trap at guestVa=0x140001234 instructionOffset=+0x2c shader=0x123456789abcdef0 sha256=${sha256Hex} route=/game/boot.pgim`
  );
  assert.equal(
    preserved,
    `Trap at guestVa=0x140001234 instructionOffset=+0x2c shader=0x123456789abcdef0 sha256=${sha256Hex} route=/game/boot.pgim`
  );

  // Idempotency on complex mixed strings
  const mixed = sanitizeString(
    `\\\\?\\C:\\Users\\alice\\x.exe /home/bob/y?token=sec ${jwt} data:image/png;base64,iVBORw0KGgoAAAANSUhEUg== guestVa=0x140001234 +0x2c`
  );
  assert.equal(sanitizeString(mixed), mixed);
  assert.ok(mixed.includes('0x140001234'));
  assert.ok(mixed.includes('+0x2c'));

  // sanitizeValue summarizes serialized Node Buffers and Blobs without dumping bytes or paths
  const serializedBuf = { type: 'Buffer', data: [0x00, 0x61, 0x73, 0x6d, 0x01, 0x00] };
  const sanitizedObj = sanitizeValue({
    serializedBuf,
    rawUint8: new Uint8Array([1, 2, 3, 4, 5]),
    session_token: 'super-secret-session',
    private_key: '-----BEGIN KEY-----',
    guestVa: '0x140009990',
    instructionOffset: '+0x18',
  });
  assert.deepEqual(sanitizedObj.serializedBuf, { type: 'Buffer', byteLength: 6 });
  assert.deepEqual(sanitizedObj.rawUint8, { type: 'Uint8Array', byteLength: 5 });
  assert.equal(sanitizedObj.session_token, '[REDACTED]');
  assert.equal(sanitizedObj.private_key, '[REDACTED]');
  assert.equal(sanitizedObj.guestVa, '0x140009990');
  assert.equal(sanitizedObj.instructionOffset, '+0x18');
  if (typeof Blob === 'function') {
    const blobSanitized = sanitizeValue({ payload: new Blob(['secret-game-bytes']) });
    assert.deepEqual(blobSanitized.payload, { type: 'Blob', byteLength: 17 });
  }
});

test('W220 preserves trap VAs, instruction offsets, GPU validation/draw/skip counts, 1%-low FPS, Bink2 telemetry, and missing shader hashes in bundles', () => {
  const clock = createDeterministicClock(1700005000000);
  const perf = new PerfMonitor({ windowSize: 10, now: clock.now });
  perf.recordFrameTimeMs(16.0);
  perf.recordFrameTimeMs(16.5);
  perf.recordFrameTimeMs(17.0);
  perf.recordFrameTimeMs(25.0); // p99 = 25.0ms -> 1%-low FPS = 40
  const rawPerfSnap = perf.getSnapshot();
  assert.equal('onePercentLowFps' in rawPerfSnap.frames, false);

  const reporter = new CrashReporter({
    appVersion: '0.4.0-w220',
    now: clock.now,
    globalObj: { crossOriginIsolated: true, SharedArrayBuffer },
    navigatorObj: {
      gpu: {},
      storage: { getDirectory: () => Promise.resolve({}) },
      hardwareConcurrency: 16,
    },
  });

  reporter.setBootSnapshot({
    stage: 'BINK_OPEN',
    mode: 'real-wasm',
    progressPct: 88.9,
    readyForSplash: false,
    trapCount: 3,
    uniqueTrapCount: 2,
    lastTrap: {
      seq: 3,
      kind: 'UNIMPLEMENTED_OPCODE',
      guestVa: 0x140045678n,
      instructionOffset: '+0x2c',
      rva: '0x45678',
      functionVa: '0x14004564c',
      mnemonic: 'vpgatherdd',
      opcode: '0x62f27d48',
      stage: 'BINK_OPEN',
      count: 2,
      message: 'Unimplemented AVX-512 op while loading C:\\Users\\alice\\GTA6\\bink.bk2?token=secret',
      rawBytes: new Uint8Array([0x62, 0xf2, 0x7d, 0x48]),
      localFile: '/home/bob/dump.bin',
    },
    recentTraps: [
      {
        seq: 1,
        kind: 'INDIRECT_CALL_MISS',
        guestVa: '0x140012000',
        insnOffset: 0x14,
        stage: 'ENTRY_CALL',
        count: 1,
        message: 'Indirect call target unresolved at /Users/charlie/game.exe',
      },
    ],
    unresolvedImports: [
      {
        name: 'Bink2Scarlett.dll!BinkOpen',
        dll: 'Bink2Scarlett.dll',
        symbol: 'BinkOpen',
        guestVa: '0x140020100',
        instructionOffset: '+0x8',
        count: 1,
      },
    ],
  });

  reporter.setGpuTelemetry({
    backend: 'webgpu',
    adapter: 'NVIDIA RTX /home/bob/driver',
    deviceLost: false,
    framesPresented: 120,
    presentCount: 120,
    drawCalls: 450,
    realPackDrawCount: 410,
    fallbackDrawCount: 32,
    skippedDrawCount: 8,
    unboundBindGroupCount: 3,
    computeDispatches: 14,
    validationErrorCount: 2,
    lastValidationErrorCode: 'BIND_GROUP_MISMATCH',
    lastValidationErrorMessage: 'Bind group 1 mismatch from C:\\Users\\alice\\shader.wgsl?api_key=k999',
    shaderFallback: {
      packHits: 92,
      fallbackHits: 6,
      builtinHits: 2,
      totalLookups: 100,
      uniqueMissing: 3,
      hitRatePct: 92.0,
    },
    missingShaderSummary: {
      top: [
        { hash: 0x00000000deadbeefn, stage: 'vertex', hits: 5, firstSeenFrame: 4, wgsl: 'secret-wgsl' },
        { hash: '0x123456789abcdef0', stage: 'fragment', hits: 19, firstSeenFrame: 1, rawSpirv: new Uint8Array(8) },
        { hash: '0xfedcba9876543210', stage: 'compute', hits: 11, firstSeenFrame: 2, sourcePath: '/home/bob/cs.hlsl' },
      ],
    },
    rawCommandBuffer: new ArrayBuffer(256),
  });

  // Duck-typed BinkVideoPresenter / SplashFrameBridge stats provider with raw planes & paths that MUST be dropped
  const mockBinkPresenter = {
    getStats() {
      return {
        pixelFormat: 'yuv420p',
        colorSpace: 'bt709',
        colorRange: 'limited',
        chromaOrder: 'yvu',
        loadOp: 'clear',
        frameWidth: 3840,
        frameHeight: 2160,
        targetFps: 30,
        uploadedFrames: 48,
        presentedFrames: 46,
        droppedFrames: 2,
        skippedFrames: 1,
        hasFrame: true,
        paused: false,
        skipped: false,
        // Disallowed raw buffers & host paths that MUST be stripped
        yPlane: new Uint8Array(64),
        rgba: new Uint8Array(128),
        videoPath: 'C:\\Users\\alice\\GTA6\\movies\\intro.bk2',
        streamUrl: 'http://localhost:8080/video?token=binksecret',
      };
    },
  };

  const storedBink = reporter.setBinkTelemetry(mockBinkPresenter);
  assert.equal(storedBink.pixelFormat, 'yuv420p');
  assert.equal(storedBink.chromaOrder, 'yvu');
  assert.equal(storedBink.frameWidth, 3840);
  assert.equal(storedBink.frameHeight, 2160);
  assert.equal(storedBink.presentedFrames, 46);
  assert.equal(storedBink.droppedFrames, 2);
  assert.equal(storedBink.skippedFrames, 1);
  assert.equal('yPlane' in storedBink, false);
  assert.equal('rgba' in storedBink, false);
  assert.equal('videoPath' in storedBink, false);
  assert.equal('streamUrl' in storedBink, false);

  const report = reporter.buildReport({ perfSnapshot: rawPerfSnap });
  // Caller's perfSnapshot object was not mutated
  assert.equal('onePercentLowFps' in rawPerfSnap.frames, false);
  // Exported bundle carries derived 1%-low FPS (1000 / 25.0 = 40)
  assert.equal(report.perf.frames.onePercentLowFps, 40);

  // Exact trap VA and instruction offset preserved
  assert.equal(report.bootDiagnostics.bootSnapshot.lastTrap.guestVa, '0x140045678');
  assert.equal(report.bootDiagnostics.bootSnapshot.lastTrap.instructionOffset, '+0x2c');
  assert.equal(report.bootDiagnostics.bootSnapshot.lastTrap.rva, '0x45678');
  assert.equal(report.bootDiagnostics.bootSnapshot.lastTrap.functionVa, '0x14004564c');
  assert.equal(report.bootDiagnostics.bootSnapshot.lastTrap.mnemonic, 'vpgatherdd');
  assert.equal(report.bootDiagnostics.bootSnapshot.recentTraps[0].guestVa, '0x140012000');
  assert.equal(report.bootDiagnostics.bootSnapshot.recentTraps[0].insnOffset, 20);
  assert.equal('rawBytes' in report.bootDiagnostics.bootSnapshot.lastTrap, false);
  assert.equal('localFile' in report.bootDiagnostics.bootSnapshot.lastTrap, false);

  // Exact GPU validation/draw/skip counts and shader hashes preserved
  const gpu = report.bootDiagnostics.gpuTelemetry;
  assert.equal(gpu.drawCalls, 450);
  assert.equal(gpu.realPackDrawCount, 410);
  assert.equal(gpu.fallbackDrawCount, 32);
  assert.equal(gpu.skippedDrawCount, 8);
  assert.equal(gpu.unboundBindGroupCount, 3);
  assert.equal(gpu.validationErrorCount, 2);
  assert.equal(gpu.lastValidationErrorCode, 'BIND_GROUP_MISMATCH');
  assert.ok(!gpu.lastValidationErrorMessage.includes('alice'));
  assert.ok(!gpu.lastValidationErrorMessage.includes('k999'));
  assert.equal(gpu.missingShaders.length, 3);
  assert.equal(gpu.missingShaders[0].hash, '0x00000000deadbeef');
  assert.equal(gpu.missingShaders[1].hash, '0x123456789abcdef0');
  assert.equal(gpu.missingShaders[2].hash, '0xfedcba9876543210');
  assert.equal('rawCommandBuffer' in gpu, false);

  // Bink2 telemetry attached in bootDiagnostics
  assert.equal(report.bootDiagnostics.binkTelemetry.pixelFormat, 'yuv420p');
  assert.equal(report.bootDiagnostics.binkTelemetry.presentedFrames, 46);

  // JSON export leaks zero paths, secrets, or shader sources
  const json = reporter.toJson({ perfSnapshot: perf });
  for (const forbidden of ['alice', 'bob', 'charlie', 'secret', 'k999', 'binksecret', 'secret-wgsl', 'intro.bk2']) {
    assert.ok(!json.includes(forbidden), `JSON bundle leaked "${forbidden}"`);
  }

  // clear() resets binkTelemetry too
  reporter.clear();
  assert.equal(reporter.getBinkTelemetry(), null);
});

test('W220 formatGitHubIssueMarkdown produces compact, strictly sanitized Markdown with exact VAs, offsets, GPU/Bink2 counts, 1%-low FPS, and top missing shader hashes', () => {
  const clock = createDeterministicClock(1700006000000);
  const reporter = new CrashReporter({
    appVersion: '0.4.0-w220',
    now: clock.now,
    globalObj: { crossOriginIsolated: true, SharedArrayBuffer },
    navigatorObj: { gpu: {}, storage: { getDirectory: () => Promise.resolve({}) }, hardwareConcurrency: 8 },
  });

  reporter.recordError(
    'wasm_trap',
    new WebAssembly.RuntimeError('Trap at 0x140045678 loading C:\\Users\\alice\\GTA6\\game.exe?token=errtok'),
    { guestVa: '0x140045678', instructionOffset: '+0x2c', rawDump: new Uint8Array([0xde, 0xad]) }
  );

  reporter.setBootSnapshot({
    stage: 'D3D12_CREATE_DEVICE',
    mode: 'real-wasm',
    progressPct: 77.8,
    readyForSplash: false,
    trapCount: 2,
    uniqueTrapCount: 2,
    lastTrap: {
      seq: 2,
      kind: 'UNIMPLEMENTED_OPCODE',
      guestVa: '0x140045678',
      instructionOffset: '+0x2c',
      mnemonic: 'vpgatherdd',
      stage: 'D3D12_CREATE_DEVICE',
      count: 2,
      message: 'Unimplemented op in /home/bob/game.wasm Bearer secret123',
    },
    unresolvedImports: [
      {
        name: 'kernel32.dll!CreateFileW',
        dll: 'kernel32.dll',
        symbol: 'CreateFileW',
        guestVa: '0x140001234',
        instructionOffset: '+0x10',
        count: 3,
      },
    ],
  });

  reporter.setGpuTelemetry({
    backend: 'webgpu',
    adapter: 'NVIDIA Blackwell /Users/charlie/driver',
    framesPresented: 60,
    drawCalls: 320,
    realPackDrawCount: 280,
    fallbackDrawCount: 30,
    skippedDrawCount: 10,
    unboundBindGroupCount: 4,
    computeDispatches: 9,
    validationErrorCount: 3,
    lastValidationErrorCode: 'PIPELINE_LAYOUT_MISMATCH',
    lastValidationErrorMessage: 'Layout mismatch at C:\\Users\\alice\\ps.wgsl?sig=s1',
    fallbackStats: {
      packHits: 90,
      fallbackHits: 10,
      builtinHits: 0,
      totalLookups: 100,
      uniqueMissing: 3,
      hitRatePct: 90,
    },
    missingShaders: [
      { hash: '0x00000000000000aa', stage: 'vertex', hits: 2, firstSeenFrame: 5 },
      { hash: '0x123456789abcdef0', stage: 'fragment', hits: 17, firstSeenFrame: 1 },
      { hash: '0xabcdef0123456789', stage: 'compute', hits: 9, firstSeenFrame: 2 },
    ],
  });

  reporter.setBinkTelemetry({
    pixelFormat: 'yuv420p',
    chromaOrder: 'yvu',
    colorSpace: 'bt709',
    colorRange: 'limited',
    frameWidth: 1920,
    frameHeight: 1080,
    presentedFrames: 30,
    droppedFrames: 1,
    skippedFrames: 0,
    uploadedFrames: 31,
    targetFps: 30,
  });

  const md = reporter.toGitHubIssueMarkdown({
    perfSnapshot: {
      frames: {
        fps: 58.42,
        avgFrameTimeMs: 17.12,
        p95FrameTimeMs: 21.4,
        p99FrameTimeMs: 24.24,
        onePercentLowFps: 41.25,
        totalFrames: 300,
        jankFrames: 2,
      },
    },
  });

  // Verify structure and exact diagnostic tokens preserved
  assert.ok(md.startsWith('### PlayGTA6Web Sanitized Diagnostic Summary'));
  assert.ok(md.includes('`0.4.0-w220`'));
  assert.ok(md.includes('1%-low FPS=`41.25`'));
  assert.ok(md.includes('FPS=`58.42`'));
  assert.ok(md.includes('draws=320'));
  assert.ok(md.includes('realPackDraws=280'));
  assert.ok(md.includes('fallbackDraws=30'));
  assert.ok(md.includes('skippedDraws=10'));
  assert.ok(md.includes('unboundBindGroups=4'));
  assert.ok(md.includes('validationErrors=3'));
  assert.ok(md.includes('`PIPELINE_LAYOUT_MISMATCH`'));
  assert.ok(md.includes('`yuv420p`'));
  assert.ok(md.includes('res=1920x1080'));
  assert.ok(md.includes('chroma=`yvu`'));
  assert.ok(md.includes('presented=30'));
  assert.ok(md.includes('dropped=1'));
  assert.ok(md.includes('`0x140045678`'));
  assert.ok(md.includes('`+0x2c`'));
  assert.ok(md.includes('`kernel32.dll!CreateFileW`'));
  assert.ok(md.includes('`0x140001234`'));
  assert.ok(md.includes('`0x123456789abcdef0`'));
  assert.ok(md.includes('`0xabcdef0123456789`'));

  // Missing shaders table must be sorted by hits descending (17 before 9 before 2)
  const idxTop = md.indexOf('0x123456789abcdef0');
  const idxMid = md.indexOf('0xabcdef0123456789');
  const idxLow = md.indexOf('0x00000000000000aa');
  assert.ok(idxTop > 0 && idxMid > idxTop && idxLow > idxMid);

  // Zero leaked host paths, usernames, or tokens
  for (const forbidden of ['alice', 'bob', 'charlie', 'errtok', 'secret123', 'sig=s1', 'GTA6\\game.exe', 'game.exe']) {
    assert.ok(!md.includes(forbidden), `Markdown leaked "${forbidden}"`);
  }

  // Defense-in-depth: passing a RAW UNSANITIZED object directly to formatGitHubIssueMarkdown()
  const rawDirtyReport = {
    appVersion: '0.4.0 C:\\Users\\alice\\repo',
    schemaVersion: 1,
    bootDiagnostics: {
      bootSnapshot: {
        stage: 'ENTRY_CALL',
        trapCount: 1,
        lastTrap: {
          kind: 'GUEST_TRAP',
          guestVa: '0x140099990',
          instructionOffset: '+0x44',
          message: 'Fault | in /home/bob/secret.rpf?token=rawtok data:application/octet-stream;base64,QUJDREVGR0hJSktMTU5PUA==',
          rawMemory: new Uint8Array([0xca, 0xfe, 0xba, 0xbe]),
        },
        hostPath: 'C:\\Users\\alice\\GTA6',
      },
      gpuTelemetry: {
        backend: 'webgpu',
        validationErrors: 4,
        drawCalls: 100,
        skippedDrawCount: 7,
        missingShaders: [
          { hash: '0x9988776655443322', stage: 'fragment', hits: 12, wgsl: '@fragment fn leak() {}' },
        ],
        bink: {
          pixelFormat: 'nv12',
          frameWidth: 1280,
          frameHeight: 720,
          presentedFrames: 15,
          droppedFrames: 0,
          yPlane: new Uint8Array(32),
          videoPath: '/Users/charlie/Movies/intro.bk2',
        },
        secretBuf: { type: 'Buffer', data: [1, 2, 3, 4] },
      },
    },
    perf: { frames: { fps: 60, p99FrameTimeMs: 20 } },
    errors: [
      {
        category: 'wasm_trap',
        message: 'Crash in C:\\Users\\alice\\bin.wasm | Bearer my-raw-bearer',
        context: { guestVa: '0x140099990', instructionOffset: '+0x44', payload: new Uint8Array(16) },
      },
    ],
    extra: {
      password: 'do-not-render-extra-password',
      localPath: 'C:\\Users\\alice\\never-shown',
    },
  };

  const dirtyMd = formatGitHubIssueMarkdown(rawDirtyReport);
  assert.ok(dirtyMd.includes('`0x140099990`'));
  assert.ok(dirtyMd.includes('`+0x44`'));
  assert.ok(dirtyMd.includes('1%-low FPS=`50`'));
  assert.ok(dirtyMd.includes('validationErrors=4'));
  assert.ok(dirtyMd.includes('skippedDraws=7'));
  assert.ok(dirtyMd.includes('`0x9988776655443322`'));
  assert.ok(dirtyMd.includes('`nv12`'));
  assert.ok(dirtyMd.includes('\\|')); // Pipe in message escaped so Markdown table stays valid
  for (const forbidden of [
    'alice',
    'bob',
    'charlie',
    'rawtok',
    'QUJDREVGR0hJSktMTU5PUA==',
    '@fragment',
    'intro.bk2',
    'my-raw-bearer',
    'do-not-render-extra-password',
    'never-shown',
  ]) {
    assert.ok(!dirtyMd.includes(forbidden), `formatGitHubIssueMarkdown leaked "${forbidden}"`);
  }

  // Graceful handling of null/empty inputs and CrashReporter instance passed directly
  assert.ok(formatGitHubIssueMarkdown(null).startsWith('### PlayGTA6Web Sanitized Diagnostic Summary'));
  assert.ok(formatGitHubIssueMarkdown({}).startsWith('### PlayGTA6Web Sanitized Diagnostic Summary'));
  const fromInstance = formatGitHubIssueMarkdown(reporter, {
    title: '### Custom Heading',
    maxMissingShaders: 1,
  });
  assert.ok(fromInstance.startsWith('### Custom Heading'));
  assert.ok(fromInstance.includes('0x123456789abcdef0'));
  assert.ok(!fromInstance.includes('0xabcdef0123456789'));
});
