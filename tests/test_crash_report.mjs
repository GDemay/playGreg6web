import test from 'node:test';
import assert from 'node:assert/strict';

import CrashReporterDefault, {
  SCHEMA_VERSION,
  DEFAULT_MAX_LOGS,
  DEFAULT_MAX_ERRORS,
  DEFAULT_APP_VERSION,
  ERROR_CATEGORIES,
  CrashReporter,
  sanitizeString,
  sanitizeValue,
  classifyError,
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
