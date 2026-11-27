import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  DEFAULT_MAX_CONCURRENCY,
  DEFAULT_SAMPLE_COUNT,
  INCOMPLETE_DOWNLOAD_SUFFIXES,
  MANIFEST_ABI_VERSION,
  MAX_MAX_CONCURRENCY,
  MIN_MAX_CONCURRENCY,
  MISMATCH_CATEGORIES,
  SCHEMA_DIAGNOSTIC_CODES,
  STREAM_HASH_ALGORITHMS,
  computeDigest,
  computeStreamingReadiness,
  createIncrementalVerifier,
  getFingerprintCacheStats,
  getStats,
  hasIncompleteSuffix,
  isStrongEtag,
  parseContentRange,
  resetStats,
  validateBootset,
  validateDataManifest,
  validateManifestPath,
  verifyLiveEndpoints,
  verifyStream,
} from '../web/storage/manifest_verifier.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..');

function makeValidManifest(overrides = {}) {
  const files = overrides.files || [
    { path: 'MicrosoftGame.Config', size: 576 },
    { path: 'x64/levels.rpf', size: 1048000 },
    { path: 'x64/audio/sfx.awc', size: 262144 },
  ];
  const totalBytes = files.reduce((sum, f) => sum + (typeof f.size === 'number' ? f.size : 0), 0);
  return {
    abi_version: MANIFEST_ABI_VERSION,
    file_count: files.length,
    total_bytes: totalBytes,
    files,
    ...overrides,
  };
}

function makeMockHeaders(map) {
  const lower = new Map();
  for (const [k, v] of Object.entries(map || {})) {
    lower.set(k.toLowerCase(), String(v));
  }
  return {
    get(name) {
      const val = lower.get(String(name).toLowerCase());
      return val === undefined ? null : val;
    },
  };
}

function createMockDataFetch(fileMap, options = {}) {
  const calls = [];
  const basePrefix = options.dataBase || '/data/';
  const coopCoep = options.coopCoep !== false;

  const fetchImpl = async (url, init = {}) => {
    const method = (init.method || 'GET').toUpperCase();
    const reqHeaders = init.headers || {};
    const rangeHeader =
      typeof reqHeaders.get === 'function'
        ? reqHeaders.get('Range')
        : reqHeaders.Range || reqHeaders.range || null;
    calls.push({ url, method, range: rangeHeader });

    const relEncoded = url.startsWith(basePrefix) ? url.slice(basePrefix.length) : url;
    const relPath = relEncoded
      .split('/')
      .map((s) => decodeURIComponent(s))
      .join('/');

    if (options.onRequest) {
      const custom = await options.onRequest({ url, method, rangeHeader, relPath });
      if (custom) return custom;
    }

    const entry = fileMap.get(relPath);
    if (!entry) {
      return {
        status: 404,
        ok: false,
        headers: makeMockHeaders({ 'Content-Length': '0' }),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const bytes = entry.bytes;
    const size = bytes.byteLength;
    const baseHeaders = {
      'Accept-Ranges': 'bytes',
      ETag: entry.etag || `"${size.toString(16)}-12345678"`,
    };
    if (coopCoep) {
      baseHeaders['Cross-Origin-Opener-Policy'] = 'same-origin';
      baseHeaders['Cross-Origin-Embedder-Policy'] = 'require-corp';
    }

    if (method === 'HEAD') {
      return {
        status: 200,
        ok: true,
        headers: makeMockHeaders({
          ...baseHeaders,
          'Content-Length': String(size),
        }),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (rangeHeader) {
      const match = /^bytes=(\d+)-(\d+)$/i.exec(rangeHeader);
      if (!match || size === 0) {
        return {
          status: 416,
          ok: false,
          headers: makeMockHeaders({
            ...baseHeaders,
            'Content-Range': `bytes */${size}`,
            'Content-Length': '0',
          }),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      const start = Number(match[1]);
      const end = Math.min(Number(match[2]), size - 1);
      const slice = bytes.slice(start, end + 1);
      return {
        status: 206,
        ok: true,
        headers: makeMockHeaders({
          ...baseHeaders,
          'Content-Range': `bytes ${start}-${end}/${size}`,
          'Content-Length': String(slice.byteLength),
        }),
        async arrayBuffer() {
          return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
        },
      };
    }

    return {
      status: 200,
      ok: true,
      headers: makeMockHeaders({
        ...baseHeaders,
        'Content-Length': String(size),
      }),
      async arrayBuffer() {
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      },
    };
  };

  return { fetchImpl, calls };
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
    if (probe.error || probe.status !== 0) continue;
    const output = `${probe.stdout || ''}\n${probe.stderr || ''}`;
    if (/Python\s+3\./i.test(output)) {
      return candidate;
    }
  }
  return null;
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
      // Server still starting.
    }
    await new Promise((r) => setTimeout(r, 25));
  }
  throw new Error(`Timed out waiting for ${baseUrl}/api/status:\n${getLogs()}`);
}

async function stopChildProcess(child) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  const exitPromise = new Promise((resolve) => child.once('exit', resolve));
  try {
    child.kill();
  } catch {
    // Already exited.
  }
  const timer = setTimeout(() => {
    try {
      child.kill('SIGKILL');
    } catch {
      // Ignore.
    }
  }, 2000);
  timer.unref?.();
  await exitPromise;
  clearTimeout(timer);
}

describe('validateManifestPath and helpers', () => {
  it('accepts clean relative POSIX paths', () => {
    const valid = [
      'levels.rpf',
      'MicrosoftGame.Config',
      'x64/audio/sfx.awc',
      'dlc/pack 1/file (v2).rpf',
      '100%complete.bin',
    ];
    for (const p of valid) {
      const res = validateManifestPath(p);
      assert.equal(res.ok, true, `expected valid path: ${p}`);
      assert.equal(res.normalized, p);
      assert.equal(res.error, null);
    }
  });

  it('rejects path traversal, leading/trailing slashes, backslashes, drive letters, and NUL', () => {
    const invalid = [
      '',
      null,
      123,
      '../secret.bin',
      'x64/../../etc/passwd',
      'x64/levels.rpf/..',
      '..',
      '.',
      './levels.rpf',
      'x64/./levels.rpf',
      '/levels.rpf',
      '/etc/passwd',
      'x64/',
      'x64//levels.rpf',
      'x64\\levels.rpf',
      '..\\levels.rpf',
      'C:levels.rpf',
      'C:/Windows/win.ini',
      'd:/game/levels.rpf',
      'x64/C:foo.rpf',
      'levels.rpf\0.txt',
      'levels\n.rpf',
      '%2e%2e/secret.rpf',
      'x64/%2e%2e/escape.rpf',
      'x64/a%2fb.rpf',
      'x64/a%5cb.rpf',
    ];
    for (const p of invalid) {
      const res = validateManifestPath(p);
      assert.equal(res.ok, false, `expected invalid path to be rejected: ${JSON.stringify(p)}`);
      assert.equal(res.normalized, null);
      assert.ok(typeof res.error === 'string' && res.error.length > 0);
    }
  });

  it('detects incomplete download suffixes matching tools/src/scan.rs', () => {
    for (const suffix of INCOMPLETE_DOWNLOAD_SUFFIXES) {
      assert.equal(hasIncompleteSuffix(`x64/world.rpf${suffix}`), true);
      assert.equal(hasIncompleteSuffix(`x64/world.rpf${suffix.toUpperCase()}`), true);
    }
    assert.equal(hasIncompleteSuffix('x64/world.rpf'), false);
    assert.equal(hasIncompleteSuffix('MicrosoftGame.Config'), false);
  });

  it('parses valid Content-Range headers and rejects malformed ones', () => {
    assert.deepEqual(parseContentRange('bytes 0-0/1024'), { start: 0, end: 0, total: 1024 });
    assert.deepEqual(parseContentRange('BYTES 10-19/*'), { start: 10, end: 19, total: null });
    assert.equal(parseContentRange('bytes 10-5/1024'), null);
    assert.equal(parseContentRange('bytes 0-1024/1024'), null);
    assert.equal(parseContentRange('bytes */1024'), null);
    assert.equal(parseContentRange('items 0-0/10'), null);
    assert.equal(parseContentRange(null), null);
  });
});

describe('validateDataManifest', () => {
  it('validates canonical docs/io_manifest.md manifest with optional metadata', () => {
    const sha = 'A'.repeat(64);
    const manifest = makeValidManifest({
      files: [
        { path: 'MicrosoftGame.Config', size: 576, sha256: sha, etag: '"240-1000"' },
        { path: 'x64/levels.rpf', size: 1048000, etag: 'W/"ffdc0-2000"', status: 'complete' },
      ],
    });

    const res = validateDataManifest(manifest);
    assert.equal(res.ok, true);
    assert.deepEqual(res.errors, []);
    assert.deepEqual(res.warnings, []);
    assert.equal(res.fileMap.size, 2);
    assert.equal(res.fileMap.get('MicrosoftGame.Config').sha256, 'a'.repeat(64));
    assert.equal(res.fileMap.get('MicrosoftGame.Config').etag, '"240-1000"');
    assert.equal(res.fileMap.get('x64/levels.rpf').incomplete, false);
    assert.deepEqual(res.summary, {
      abiVersion: 1,
      fileCount: 2,
      declaredFileCount: 2,
      totalBytes: 1048576,
      declaredTotalBytes: 1048576,
      completeFiles: 2,
      incompleteFiles: 0,
      completeBytes: 1048576,
      incompleteBytes: 0,
      hashedFiles: 1,
      etagFiles: 2,
      duplicateCount: 0,
    });
  });

  it('accepts JSON string input and rejects malformed JSON or non-object payloads', () => {
    const validJson = JSON.stringify(makeValidManifest());
    assert.equal(validateDataManifest(validJson).ok, true);

    const badJson = validateDataManifest('{not-valid-json');
    assert.equal(badJson.ok, false);
    assert.match(badJson.errors[0], /JSON parse error/);

    assert.equal(validateDataManifest([]).ok, false);
    assert.equal(validateDataManifest(null).ok, false);
  });

  it('validates abi_version and warns on legacy version field', () => {
    const missingVersion = { files: [{ path: 'a.bin', size: 10 }] };
    const rMissing = validateDataManifest(missingVersion);
    assert.equal(rMissing.ok, false);
    assert.ok(rMissing.errors.some((e) => /abi_version/.test(e)));

    const badVersion = makeValidManifest({ abi_version: 99 });
    const rBad = validateDataManifest(badVersion);
    assert.equal(rBad.ok, false);
    assert.ok(rBad.errors.some((e) => /Unsupported manifest "abi_version"/.test(e)));

    const legacyVersion = { version: 1, files: [{ path: 'a.bin', size: 10 }] };
    const rLegacy = validateDataManifest(legacyVersion);
    assert.equal(rLegacy.ok, true);
    assert.equal(rLegacy.summary.abiVersion, 1);
    assert.ok(rLegacy.warnings.some((w) => /canonical "abi_version"/.test(w)));
  });

  it('rejects path traversal, duplicate paths, invalid sizes, bad sha256, and mismatched aggregates', () => {
    const manifest = {
      abi_version: 1,
      file_count: 99,
      total_bytes: 1234567,
      files: [
        { path: 'x64/levels.rpf', size: 1000 },
        { path: 'x64/levels.rpf', size: 1000 }, // duplicate
        { path: '../escape.rpf', size: 500 }, // traversal
        { path: '/abs/path.rpf', size: 500 }, // leading slash
        { path: 'win\\back.rpf', size: 500 }, // backslash
        { path: 'C:/game/drive.rpf', size: 500 }, // drive letter
        { path: 'neg.rpf', size: -1 }, // negative size
        { path: 'float.rpf', size: 12.5 }, // non-integer size
        { path: 'str.rpf', size: '100' }, // string size
        { path: 'badhash.rpf', size: 64, sha256: 'not-a-sha256' }, // bad sha256
        { path: 'badetag.rpf', size: 64, etag: '' }, // empty etag
        { path: 'badflag.rpf', size: 64, incomplete: 'yes' }, // non-boolean incomplete
      ],
    };

    const res = validateDataManifest(manifest);
    assert.equal(res.ok, false);
    assert.equal(res.summary.duplicateCount, 1);
    assert.ok(res.errors.some((e) => /file_count/.test(e)));
    assert.ok(res.errors.some((e) => /duplicates/.test(e)));
    assert.ok(res.errors.some((e) => /traversal/.test(e)));
    assert.ok(res.errors.some((e) => /leading slash/.test(e)));
    assert.ok(res.errors.some((e) => /backslashes/.test(e)));
    assert.ok(res.errors.some((e) => /drive letter/.test(e)));
    assert.ok(res.errors.some((e) => /files\[6\]\.size/.test(e)));
    assert.ok(res.errors.some((e) => /files\[7\]\.size/.test(e)));
    assert.ok(res.errors.some((e) => /files\[8\]\.size/.test(e)));
    assert.ok(res.errors.some((e) => /sha256/.test(e)));
    assert.ok(res.errors.some((e) => /etag/.test(e)));
    assert.ok(res.errors.some((e) => /incomplete must be a boolean/.test(e)));
  });

  it('detects total_bytes mismatch when all file sizes are valid', () => {
    const manifest = {
      abi_version: 1,
      file_count: 2,
      total_bytes: 999,
      files: [
        { path: 'a.bin', size: 100 },
        { path: 'b.bin', size: 200 },
      ],
    };
    const res = validateDataManifest(manifest);
    assert.equal(res.ok, false);
    assert.ok(res.errors.some((e) => /total_bytes/.test(e)));
  });

  it('classifies incomplete flags, status values, temp suffixes, and case collisions', () => {
    const manifest = makeValidManifest({
      files: [
        { path: 'GTA6.exe', size: 1000 },
        { path: 'gta6.exe', size: 1000 }, // case-only collision warning
        { path: 'x64/audio.rpf.!qB', size: 4000 }, // temp suffix auto-incomplete + warning
        { path: 'x64/world.rpf', size: 5000, incomplete: true },
        { path: 'x64/textures.rpf', size: 2000, status: 'downloading' },
      ],
    });
    const res = validateDataManifest(manifest);
    assert.equal(res.ok, true);
    assert.equal(res.summary.completeFiles, 2);
    assert.equal(res.summary.incompleteFiles, 3);
    assert.equal(res.summary.completeBytes, 2000);
    assert.equal(res.summary.incompleteBytes, 11000);
    assert.equal(res.fileMap.get('x64/audio.rpf.!qB').incomplete, true);
    assert.equal(res.fileMap.get('x64/audio.rpf.!qB').incompleteReason, 'temp_suffix');
    assert.ok(res.warnings.some((w) => /case-insensitively/.test(w)));
    assert.ok(res.warnings.some((w) => /temporary download suffix/.test(w)));
  });
});

describe('validateBootset', () => {
  it('validates canonical bootset array and object wrapper against manifest', () => {
    const manifest = makeValidManifest();
    const bootset = [
      { path: 'MicrosoftGame.Config', offset: 0, length: 576 },
      { path: 'x64/levels.rpf', offset: 0, length: 65536 },
      { path: 'x64/levels.rpf', offset: 65536, length: 32768 },
    ];

    const resArray = validateBootset(bootset, manifest);
    assert.equal(resArray.ok, true);
    assert.deepEqual(resArray.errors, []);
    assert.deepEqual(resArray.warnings, []);
    assert.equal(resArray.totalBytes, 576 + 65536 + 32768);
    assert.equal(resArray.uniqueBytes, 576 + 65536 + 32768);
    assert.equal(resArray.overlapBytes, 0);
    assert.equal(resArray.entries.length, 3);

    const resObj = validateBootset({ entries: bootset }, manifest);
    assert.equal(resObj.ok, true);
    assert.equal(resObj.uniqueBytes, resArray.uniqueBytes);
  });

  it('rejects out-of-bounds ranges, negative offset, zero length, missing files, and traversal', () => {
    const manifest = makeValidManifest();
    const bootset = [
      { path: 'MicrosoftGame.Config', offset: 0, length: 577 }, // 1 byte past EOF
      { path: 'x64/levels.rpf', offset: 1048000, length: 1 }, // starts at EOF
      { path: 'x64/levels.rpf', offset: -1, length: 16 }, // negative offset
      { path: 'x64/levels.rpf', offset: 0, length: 0 }, // zero length
      { path: 'x64/missing.rpf', offset: 0, length: 64 }, // missing from manifest
      { path: '../escape.rpf', offset: 0, length: 64 }, // traversal
    ];

    const res = validateBootset(bootset, manifest);
    assert.equal(res.ok, false);
    assert.equal(res.entries.length, 0);
    assert.deepEqual(res.missingFiles, ['x64/missing.rpf']);
    assert.ok(res.errors.some((e) => /exceeds manifest file size 576/.test(e)));
    assert.ok(res.errors.some((e) => /exceeds manifest file size 1048000/.test(e)));
    assert.ok(res.errors.some((e) => /offset/.test(e)));
    assert.ok(res.errors.some((e) => /length/.test(e)));
    assert.ok(res.errors.some((e) => /missing from manifest/.test(e)));
    assert.ok(res.errors.some((e) => /traversal/.test(e)));
  });

  it('deduplicates exact duplicates, detects overlapping ranges, and computes merged byte budget', () => {
    const manifest = makeValidManifest();
    const bootset = [
      { path: 'x64/levels.rpf', offset: 0, length: 1000 },
      { path: 'x64/levels.rpf', offset: 0, length: 1000 }, // exact duplicate
      { path: 'x64/levels.rpf', offset: 500, length: 1000 }, // overlaps [500, 1000) -> 500 overlap bytes
      { path: 'x64/levels.rpf', offset: 2000, length: 500 }, // disjoint [2000, 2500)
      { path: 'MicrosoftGame.Config', offset: 0, length: 200 },
    ];

    const res = validateBootset(bootset, manifest);
    assert.equal(res.ok, true);
    assert.equal(res.entries.length, 5);
    assert.equal(res.deduplicatedEntries.length, 4);
    assert.equal(res.duplicates.length, 1);
    assert.equal(res.overlaps.length, 1);
    assert.equal(res.overlaps[0].overlapBytes, 500);
    assert.equal(res.totalBytes, 1000 + 1000 + 1000 + 500 + 200); // 3700 raw
    assert.equal(res.uniqueBytes, 1500 + 500 + 200); // 2200 merged
    assert.equal(res.overlapBytes, 1500);
    assert.equal(res.mergedRanges.length, 3);
    assert.ok(res.warnings.some((w) => /duplicates bootset range/.test(w)));
    assert.ok(res.warnings.some((w) => /overlap by 500 bytes/.test(w)));
  });

  it('validates bootset standalone without a manifest', () => {
    const bootset = [
      { path: 'a.rpf', offset: 0, length: 4096 },
      { path: 'b.rpf', offset: 128, length: 256 },
    ];
    const res = validateBootset(bootset);
    assert.equal(res.ok, true);
    assert.equal(res.totalBytes, 4352);
    assert.equal(res.uniqueBytes, 4352);
    assert.deepEqual(res.missingFiles, []);
  });
});

describe('computeStreamingReadiness', () => {
  it('computes 100% readiness for complete manifest and satisfied bootset', () => {
    const manifest = makeValidManifest();
    const bootset = [
      { path: 'MicrosoftGame.Config', offset: 0, length: 576 },
      { path: 'x64/levels.rpf', offset: 0, length: 65536 },
    ];

    const readiness = computeStreamingReadiness(manifest, bootset);
    assert.equal(readiness.manifestValid, true);
    assert.equal(readiness.bootsetValid, true);
    assert.equal(readiness.totalFiles, 3);
    assert.equal(readiness.completeFiles, 3);
    assert.equal(readiness.incompleteFiles, 0);
    assert.equal(readiness.totalBytes, 576 + 1048000 + 262144);
    assert.equal(readiness.readyBytes, readiness.totalBytes);
    assert.equal(readiness.incompleteBytes, 0);
    assert.equal(readiness.readinessPct, 100);
    assert.equal(readiness.bootsetReady, true);
    assert.equal(readiness.bootsetBytes, 576 + 65536);
    assert.deepEqual(readiness.missingBootsetFiles, []);
    assert.deepEqual(readiness.incompleteBootsetFiles, []);
    assert.deepEqual(readiness.issues, []);
  });

  it('computes partial-download readiness and flags incomplete/missing bootset dependencies', () => {
    const manifest = makeValidManifest({
      files: [
        { path: 'MicrosoftGame.Config', size: 1000 },
        { path: 'x64/levels.rpf', size: 3000, incomplete: true },
        { path: 'x64/audio.rpf.!qB', size: 6000 },
      ],
    });
    const bootset = [
      { path: 'MicrosoftGame.Config', offset: 0, length: 1000 },
      { path: 'x64/levels.rpf', offset: 0, length: 1024 },
      { path: 'x64/missing.rpf', offset: 0, length: 512 },
    ];

    const readiness = computeStreamingReadiness(manifest, bootset);
    assert.equal(readiness.totalFiles, 3);
    assert.equal(readiness.completeFiles, 1);
    assert.equal(readiness.incompleteFiles, 2);
    assert.equal(readiness.totalBytes, 10000);
    assert.equal(readiness.readyBytes, 1000);
    assert.equal(readiness.incompleteBytes, 9000);
    assert.equal(readiness.readinessPct, 10);
    assert.equal(readiness.bootsetReady, false);
    assert.equal(readiness.bootsetBytes, 2024);
    assert.deepEqual(readiness.missingBootsetFiles, ['x64/missing.rpf']);
    assert.deepEqual(readiness.incompleteBootsetFiles, ['x64/levels.rpf']);
    assert.ok(readiness.issues.some((i) => i.severity === 'error' && i.source === 'bootset'));
    assert.ok(readiness.issues.some((i) => i.severity === 'warning' && i.source === 'bootset'));
  });

  it('marks bootsetReady=false when bootset only references an incomplete file', () => {
    const manifest = makeValidManifest({
      files: [
        { path: 'x64/levels.rpf', size: 4096, status: 'partial' },
      ],
    });
    const bootset = [{ path: 'x64/levels.rpf', offset: 0, length: 1024 }];
    const readiness = computeStreamingReadiness(manifest, bootset);
    assert.equal(readiness.bootsetReady, false);
    assert.deepEqual(readiness.missingBootsetFiles, []);
    assert.deepEqual(readiness.incompleteBootsetFiles, ['x64/levels.rpf']);
    assert.equal(readiness.readinessPct, 0);
  });

  it('handles empty manifest without dividing by zero', () => {
    const manifest = { abi_version: 1, file_count: 0, total_bytes: 0, files: [] };
    const readiness = computeStreamingReadiness(manifest);
    assert.equal(readiness.totalFiles, 0);
    assert.equal(readiness.readinessPct, 0);
    assert.equal(readiness.bootsetReady, false);
  });
});

describe('verifyLiveEndpoints (mocked fetch)', () => {
  it('probes up to sampleCount complete files, skips incomplete files, and verifies HEAD + 206 + ETag', async () => {
    const files = [];
    const backing = new Map();
    for (let i = 0; i < 7; i += 1) {
      const p = `x64/pack_${i}.rpf`;
      const bytes = new Uint8Array(128 + i * 10).fill(i + 1);
      const etag = `"etag-${i}"`;
      files.push({ path: p, size: bytes.byteLength, etag });
      backing.set(p, { bytes, etag });
    }
    // Insert an incomplete file at the start to ensure it is skipped.
    files.unshift({ path: 'x64/partial.rpf', size: 2048, incomplete: true });

    const manifest = makeValidManifest({ files });
    const { fetchImpl, calls } = createMockDataFetch(backing, { dataBase: '/data/' });

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data',
      manifest,
      sampleCount: DEFAULT_SAMPLE_COUNT,
    });

    assert.equal(report.ok, true);
    assert.equal(report.probed, 5);
    assert.equal(report.passed, 5);
    assert.equal(report.failed, 0);
    assert.equal(report.skippedIncomplete, 1);
    assert.equal(report.results.length, 5);
    assert.equal(calls.length, 10); // 5 HEAD + 5 Range GET
    assert.ok(calls.every((c) => !c.url.includes('partial.rpf')));
    for (const r of report.results) {
      assert.equal(r.ok, true);
      assert.equal(r.head.status, 200);
      assert.equal(r.range.status, 206);
      assert.equal(r.range.requested, 'bytes=0-0');
      assert.equal(r.range.receivedBytes, 1);
      assert.equal(r.range.parsedContentRange.total, r.expectedSize);
      assert.equal(r.coopCoep, true);
    }
  });

  it('supports multi-byte Range probes (probeBytes > 1) and 0-byte complete files', async () => {
    const backing = new Map([
      ['small.bin', { bytes: new Uint8Array([1, 2, 3, 4]), etag: '"small-1"' }],
      ['empty.bin', { bytes: new Uint8Array(0), etag: '"empty-0"' }],
    ]);
    const manifest = makeValidManifest({
      files: [
        { path: 'small.bin', size: 4 },
        { path: 'empty.bin', size: 0 },
      ],
    });
    const { fetchImpl } = createMockDataFetch(backing);

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 5,
      probeBytes: 16,
    });

    assert.equal(report.ok, true);
    assert.equal(report.passed, 2);
    assert.equal(report.results[0].range.requested, 'bytes=0-3');
    assert.equal(report.results[0].range.receivedBytes, 4);
    assert.equal(report.results[1].range.skipped, true);
    assert.equal(report.results[1].head.contentLength, 0);
  });

  it('flags 200 instead of 206, missing Accept-Ranges, Content-Range mismatch, missing ETag, and network errors', async () => {
    const manifest = makeValidManifest({
      files: [
        { path: 'no206.bin', size: 100 },
        { path: 'badsize.bin', size: 100 },
        { path: 'noetag.bin', size: 100 },
        { path: 'boom.bin', size: 100 },
      ],
    });

    const backing = new Map([
      ['no206.bin', { bytes: new Uint8Array(100), etag: '"a"' }],
      ['badsize.bin', { bytes: new Uint8Array(100), etag: '"b"' }],
      ['noetag.bin', { bytes: new Uint8Array(100), etag: '"c"' }],
      ['boom.bin', { bytes: new Uint8Array(100), etag: '"d"' }],
    ]);

    const { fetchImpl } = createMockDataFetch(backing, {
      onRequest: async ({ method, relPath }) => {
        if (relPath === 'no206.bin' && method === 'GET') {
          return {
            status: 200,
            ok: true,
            headers: makeMockHeaders({ 'Content-Length': '100', ETag: '"a"' }),
            async arrayBuffer() {
              return new ArrayBuffer(100);
            },
          };
        }
        if (relPath === 'badsize.bin' && method === 'GET') {
          return {
            status: 206,
            ok: true,
            headers: makeMockHeaders({
              'Accept-Ranges': 'bytes',
              'Content-Range': 'bytes 0-0/999',
              'Content-Length': '1',
              ETag: '"b"',
            }),
            async arrayBuffer() {
              return new ArrayBuffer(1);
            },
          };
        }
        if (relPath === 'noetag.bin') {
          return {
            status: method === 'HEAD' ? 200 : 206,
            ok: true,
            headers: makeMockHeaders(
              method === 'HEAD'
                ? { 'Content-Length': '100' } // missing Accept-Ranges + ETag
                : { 'Content-Range': 'bytes 0-0/100', 'Content-Length': '1' },
            ),
            async arrayBuffer() {
              return new ArrayBuffer(method === 'HEAD' ? 0 : 1);
            },
          };
        }
        if (relPath === 'boom.bin') {
          throw new Error('ECONNREFUSED');
        }
        return null;
      },
    });

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
    });

    assert.equal(report.ok, false);
    assert.equal(report.probed, 4);
    assert.equal(report.passed, 0);
    assert.equal(report.failed, 4);

    assert.ok(report.results[0].errors.some((e) => /expected 206/.test(e)));
    assert.ok(report.results[1].errors.some((e) => /Content-Range total \(999\)/.test(e)));
    assert.ok(report.results[2].errors.some((e) => /missing Accept-Ranges/.test(e)));
    assert.ok(report.results[2].errors.some((e) => /missing ETag/.test(e)));
    assert.ok(report.results[3].errors.some((e) => /ECONNREFUSED/.test(e)));
  });

  it('throws TypeError if no fetch implementation is available', async () => {
    await assert.rejects(
      () => verifyLiveEndpoints({ fetchImpl: null, manifest: makeValidManifest() }),
      TypeError,
    );
  });
});

describe('verifyLiveEndpoints against live ephemeral server/serve_local.py', () => {
  const pythonCmd = detectPython3Command();
  let tmpRoot = null;
  let serverProc = null;
  let baseUrl = '';
  let serverLogs = '';

  before(async () => {
    if (!pythonCmd) return;
    tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'playgta6-manifest-verifier-'));
    const dataDir = path.join(tmpRoot, 'data');
    fs.mkdirSync(path.join(dataDir, 'x64', 'audio'), { recursive: true });

    const configBytes = Buffer.from('<Game><Title>Synthetic</Title></Game>', 'utf8');
    const rpfBytes = Buffer.alloc(16384, 0x5a);
    rpfBytes.write('RPF9', 0, 'ascii');
    const audioBytes = Buffer.alloc(4096, 0x33);
    const emptyBytes = Buffer.alloc(0);

    fs.writeFileSync(path.join(dataDir, 'MicrosoftGame.Config'), configBytes);
    fs.writeFileSync(path.join(dataDir, 'x64', 'levels.rpf'), rpfBytes);
    fs.writeFileSync(path.join(dataDir, 'x64', 'audio', 'sfx.awc'), audioBytes);
    fs.writeFileSync(path.join(dataDir, 'empty.meta'), emptyBytes);

    const manifest = {
      abi_version: 1,
      file_count: 4,
      total_bytes: configBytes.length + rpfBytes.length + audioBytes.length + emptyBytes.length,
      files: [
        { path: 'MicrosoftGame.Config', size: configBytes.length },
        { path: 'empty.meta', size: emptyBytes.length },
        { path: 'x64/audio/sfx.awc', size: audioBytes.length },
        { path: 'x64/levels.rpf', size: rpfBytes.length },
      ],
    };
    const bootset = [
      { path: 'MicrosoftGame.Config', offset: 0, length: configBytes.length },
      { path: 'x64/levels.rpf', offset: 0, length: 4096 },
    ];

    fs.writeFileSync(path.join(dataDir, 'data-manifest.json'), JSON.stringify(manifest, null, 2));
    fs.writeFileSync(path.join(dataDir, 'bootset.json'), JSON.stringify(bootset, null, 2));

    const port = await allocateEphemeralPort();
    baseUrl = `http://127.0.0.1:${port}`;
    const scriptPath = path.join(REPO_ROOT, 'server', 'serve_local.py');
    serverProc = spawn(
      pythonCmd.cmd,
      [
        ...pythonCmd.prefixArgs,
        scriptPath,
        '--host',
        '127.0.0.1',
        '--port',
        String(port),
        '--web',
        path.join(REPO_ROOT, 'web'),
        '--data',
        dataDir,
      ],
      { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true },
    );
    serverProc.stdout.on('data', (chunk) => {
      serverLogs += chunk.toString('utf8');
    });
    serverProc.stderr.on('data', (chunk) => {
      serverLogs += chunk.toString('utf8');
    });

    await waitForServerReady(baseUrl, serverProc, () => serverLogs);
  });

  after(async () => {
    await stopChildProcess(serverProc);
    if (tmpRoot) {
      fs.rmSync(tmpRoot, { recursive: true, force: true });
    }
  });

  it('validates fetched manifest/bootset and verifies live HEAD/Range/ETag/COOP-COEP responses', async (t) => {
    if (!pythonCmd) {
      t.skip('Python 3 not available in environment');
      return;
    }

    const manifestRes = await fetch(`${baseUrl}/data/data-manifest.json`);
    assert.equal(manifestRes.status, 200);
    const manifest = await manifestRes.json();

    const bootsetRes = await fetch(`${baseUrl}/data/bootset.json`);
    assert.equal(bootsetRes.status, 200);
    const bootset = await bootsetRes.json();

    const manifestVal = validateDataManifest(manifest);
    assert.equal(manifestVal.ok, true);

    const bootsetVal = validateBootset(bootset, manifestVal);
    assert.equal(bootsetVal.ok, true);

    const readiness = computeStreamingReadiness(manifestVal, bootset);
    assert.equal(readiness.readinessPct, 100);
    assert.equal(readiness.bootsetReady, true);

    const liveReport = await verifyLiveEndpoints({
      fetchImpl: fetch,
      dataBase: `${baseUrl}/data/`,
      manifest,
      sampleCount: 5,
      checkCoopCoep: true,
    });

    assert.equal(liveReport.ok, true, `unexpected errors: ${JSON.stringify(liveReport, null, 2)}`);
    assert.equal(liveReport.probed, 4);
    assert.equal(liveReport.passed, 4);
    assert.equal(liveReport.failed, 0);

    const rpfProbe = liveReport.results.find((r) => r.path === 'x64/levels.rpf');
    assert.ok(rpfProbe);
    assert.equal(rpfProbe.head.status, 200);
    assert.equal(rpfProbe.head.contentLength, 16384);
    assert.equal(rpfProbe.range.status, 206);
    assert.equal(rpfProbe.range.parsedContentRange.total, 16384);
    assert.ok(rpfProbe.etag && rpfProbe.etag.startsWith('"'));
    assert.equal(rpfProbe.coopCoep, true);
  });

  it('detects live size drift when manifest size disagrees with file served by serve_local.py', async (t) => {
    if (!pythonCmd) {
      t.skip('Python 3 not available in environment');
      return;
    }

    const driftedManifest = {
      abi_version: 1,
      file_count: 1,
      total_bytes: 99999,
      files: [{ path: 'x64/levels.rpf', size: 99999 }],
    };

    const report = await verifyLiveEndpoints({
      fetchImpl: fetch,
      dataBase: `${baseUrl}/data/`,
      manifest: driftedManifest,
      sampleCount: 1,
    });

    assert.equal(report.ok, false);
    assert.equal(report.failed, 1);
    assert.ok(report.results[0].errors.some((e) => /Content-Length \(16384\)/.test(e)));
    assert.ok(report.results[0].errors.some((e) => /Content-Range total \(16384\)/.test(e)));
    assert.equal(report.mismatchesByCategory.size_mismatch, 1);
    assert.equal(report.mismatchCount, 1);
  });
});

function sha256Hex(bytes) {
  return crypto.createHash('sha256').update(bytes).digest('hex');
}

describe('verification telemetry, mismatch categories, AbortSignal, and sanitization (W155)', () => {
  it('exports canonical MISMATCH_CATEGORIES constants', () => {
    assert.deepEqual(MISMATCH_CATEGORIES, {
      MISSING: 'missing',
      SIZE_MISMATCH: 'size_mismatch',
      HASH_MISMATCH: 'hash_mismatch',
    });
    assert.equal(Object.isFrozen(MISMATCH_CATEGORIES), true);
  });

  it('produces deterministic timing and mismatch telemetry for sync routines via injectable nowFn', () => {
    let clock = 1000;
    const stepClock = () => {
      clock += 25;
      return clock;
    };

    const manifest = {
      abi_version: 1,
      file_count: 2,
      total_bytes: 999, // intentional aggregate size mismatch vs 300
      files: [
        { path: 'a.bin', size: 100 },
        { path: 'b.bin', size: 200 },
      ],
    };

    const manifestRes = validateDataManifest(manifest, { nowFn: stepClock });
    assert.equal(manifestRes.ok, false);
    assert.equal(manifestRes.aborted, false);
    assert.equal(manifestRes.elapsedMs, 25);
    assert.equal(manifestRes.filesVerified, 2);
    assert.equal(manifestRes.bytesVerified, 300);
    assert.equal(manifestRes.mismatchCount, 1);
    assert.deepEqual(manifestRes.mismatchesByCategory, {
      missing: 0,
      size_mismatch: 1,
      hash_mismatch: 0,
    });
    assert.equal(manifestRes.telemetry.elapsedMs, 25);
    assert.equal(manifestRes.telemetry.bytesVerified, 300);
    assert.equal(manifestRes.telemetry.filesVerified, 2);
    assert.equal(manifestRes.telemetry.bytesPerSecond, 12000);
    assert.equal(manifestRes.telemetry.aborted, false);

    // Backwards-jumping clock must clamp elapsedMs to 0 and bytesPerSecond to 0.
    const ticks = [500, 400];
    let idx = 0;
    const backwardsRes = validateDataManifest(makeValidManifest(), {
      nowFn: () => ticks[Math.min(idx++, ticks.length - 1)],
    });
    assert.equal(backwardsRes.elapsedMs, 0);
    assert.equal(backwardsRes.telemetry.bytesPerSecond, 0);

    // validateBootset records missing + size_mismatch categories and deterministic timing.
    let bootClock = 100;
    const bootRes = validateBootset(
      [
        { path: 'MicrosoftGame.Config', offset: 0, length: 576 },
        { path: 'x64/missing.rpf', offset: 0, length: 64 },
        { path: 'MicrosoftGame.Config', offset: 0, length: 1024 }, // exceeds 576
      ],
      makeValidManifest(),
      {
        nowFn: () => {
          bootClock += 10;
          return bootClock;
        },
      },
    );
    assert.equal(bootRes.ok, false);
    assert.equal(bootRes.aborted, false);
    assert.equal(bootRes.filesVerified, 1);
    assert.equal(bootRes.bytesVerified, 576);
    assert.equal(bootRes.mismatchCount, 2);
    assert.deepEqual(bootRes.mismatchesByCategory, {
      missing: 1,
      size_mismatch: 1,
      hash_mismatch: 0,
    });
    assert.ok(bootRes.elapsedMs >= 10);

    // computeStreamingReadiness aggregates mismatches and timing across manifest + bootset.
    let readyClock = 0;
    const readiness = computeStreamingReadiness(
      manifest,
      [{ path: 'missing.bin', offset: 0, length: 16 }],
      {
        nowFn: () => {
          readyClock += 5;
          return readyClock;
        },
      },
    );
    assert.equal(readiness.aborted, false);
    assert.equal(readiness.mismatchCount, 2);
    assert.deepEqual(readiness.mismatchesByCategory, {
      missing: 1,
      size_mismatch: 1,
      hash_mismatch: 0,
    });
    assert.equal(readiness.filesVerified, 2);
    assert.equal(readiness.bytesVerified, 300);
    assert.ok(readiness.elapsedMs > 0);
  });

  it('tracks live verification timing, throughput, and all mismatch categories (missing, size_mismatch, hash_mismatch)', async () => {
    const okBytes = new Uint8Array(64).fill(0x41);
    const badSizeBytes = new Uint8Array(32).fill(0x42);
    const badHashBytes = new Uint8Array(48).fill(0x43);
    const etagDriftBytes = new Uint8Array(24).fill(0x44);

    const manifest = makeValidManifest({
      files: [
        { path: 'ok.bin', size: okBytes.byteLength, sha256: sha256Hex(okBytes), etag: '"ok-1"' },
        { path: 'missing.bin', size: 128, etag: '"miss-1"' },
        { path: 'size_drift.bin', size: 100, etag: '"size-1"' },
        {
          path: 'hash_drift.bin',
          size: badHashBytes.byteLength,
          sha256: '0'.repeat(64),
          etag: '"hash-1"',
        },
        { path: 'etag_drift.bin', size: etagDriftBytes.byteLength, etag: '"expected-etag"' },
      ],
    });

    const backing = new Map([
      ['ok.bin', { bytes: okBytes, etag: '"ok-1"' }],
      // missing.bin omitted -> 404
      ['size_drift.bin', { bytes: badSizeBytes, etag: '"size-1"' }],
      ['hash_drift.bin', { bytes: badHashBytes, etag: '"hash-1"' }],
      ['etag_drift.bin', { bytes: etagDriftBytes, etag: '"server-etag-other"' }],
    ]);

    let nowMs = 5000;
    const { fetchImpl } = createMockDataFetch(backing, {
      onRequest: async () => {
        nowMs += 20;
        return null;
      },
    });

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      probeBytes: 1,
      verifyHashes: true,
      nowFn: () => nowMs,
    });

    assert.equal(report.ok, false);
    assert.equal(report.aborted, false);
    assert.equal(report.probed, 5);
    assert.equal(report.passed, 1);
    assert.equal(report.failed, 4);
    assert.equal(report.filesVerified, 5);
    // ok.bin (64 full via sha256 GET) + missing.bin (0) + size_drift.bin (1 via 206)
    // + hash_drift.bin (48 full via sha256 GET) + etag_drift.bin (1 via 206) = 114 bytes
    assert.equal(report.bytesVerified, 64 + 0 + 1 + 48 + 1);
    assert.ok(report.elapsedMs > 0);
    assert.equal(report.telemetry.elapsedMs, report.elapsedMs);
    assert.equal(report.telemetry.bytesVerified, report.bytesVerified);
    assert.equal(report.telemetry.filesVerified, 5);
    assert.ok(report.telemetry.bytesPerSecond > 0);
    assert.equal(report.telemetry.aborted, false);

    assert.deepEqual(report.mismatchesByCategory, {
      missing: 1,
      size_mismatch: 1,
      hash_mismatch: 2,
    });
    assert.equal(report.mismatchCount, 4);
    assert.equal(report.mismatches.length, 4);
    assert.deepEqual(report.telemetry.mismatchesByCategory, report.mismatchesByCategory);
    assert.equal(report.telemetry.mismatchCount, 4);

    const byPath = new Map(report.results.map((r) => [r.path, r]));
    assert.equal(byPath.get('ok.bin').ok, true);
    assert.equal(byPath.get('ok.bin').hash.verified, true);
    assert.equal(byPath.get('ok.bin').bytesVerified, 64);
    assert.deepEqual(byPath.get('ok.bin').mismatchCategories, []);

    assert.deepEqual(byPath.get('missing.bin').mismatchCategories, ['missing']);
    assert.equal(byPath.get('missing.bin').bytesVerified, 0);

    assert.deepEqual(byPath.get('size_drift.bin').mismatchCategories, ['size_mismatch']);

    assert.deepEqual(byPath.get('hash_drift.bin').mismatchCategories, ['hash_mismatch']);
    assert.equal(byPath.get('hash_drift.bin').hash.verified, false);
    assert.equal(byPath.get('hash_drift.bin').hash.actual, sha256Hex(badHashBytes));

    assert.deepEqual(byPath.get('etag_drift.bin').mismatchCategories, ['hash_mismatch']);
  });

  it('reuses Range GET bytes for SHA-256 verification when full file fits within probeBytes and supports 0-byte files + custom hashFn', async () => {
    const smallBytes = new Uint8Array([10, 20, 30, 40]);
    const emptyBytes = new Uint8Array(0);
    const backing = new Map([
      ['small.bin', { bytes: smallBytes, etag: '"s1"' }],
      ['empty.bin', { bytes: emptyBytes, etag: '"e0"' }],
    ]);
    const manifest = makeValidManifest({
      files: [
        { path: 'small.bin', size: 4, sha256: sha256Hex(smallBytes), etag: '"s1"' },
        { path: 'empty.bin', size: 0, sha256: sha256Hex(emptyBytes), etag: '"e0"' },
      ],
    });

    const { fetchImpl, calls } = createMockDataFetch(backing);
    let hashInvocations = 0;
    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 5,
      probeBytes: 16,
      verifyHashes: true,
      hashFn: (bytes) => {
        hashInvocations += 1;
        return sha256Hex(bytes);
      },
    });

    assert.equal(report.ok, true);
    assert.equal(report.filesVerified, 2);
    assert.equal(report.bytesVerified, 4);
    assert.equal(report.mismatchCount, 0);
    assert.equal(hashInvocations, 2);
    // small.bin: HEAD + Range GET (no extra full GET needed); empty.bin: HEAD only.
    assert.equal(calls.length, 3);
    assert.equal(report.results[0].hash.verified, true);
    assert.equal(report.results[1].hash.verified, true);
  });

  it('supports pre-aborted AbortSignal across sync and async verification routines without issuing fetches', async () => {
    const controller = new AbortController();
    controller.abort(new Error('user cancelled'));

    const manifest = makeValidManifest();
    const manifestRes = validateDataManifest(manifest, { signal: controller.signal });
    assert.equal(manifestRes.ok, false);
    assert.equal(manifestRes.aborted, true);
    assert.equal(manifestRes.telemetry.aborted, true);
    assert.equal(manifestRes.filesVerified, 0);

    const bootsetRes = validateBootset(
      [{ path: 'MicrosoftGame.Config', offset: 0, length: 16 }],
      manifest,
      { signal: controller.signal },
    );
    assert.equal(bootsetRes.ok, false);
    assert.equal(bootsetRes.aborted, true);
    assert.equal(bootsetRes.telemetry.aborted, true);

    const readiness = computeStreamingReadiness(manifest, null, { signal: controller.signal });
    assert.equal(readiness.aborted, true);
    assert.equal(readiness.bootsetReady, false);
    assert.equal(readiness.telemetry.aborted, true);

    const backing = new Map([
      ['MicrosoftGame.Config', { bytes: new Uint8Array(576), etag: '"a"' }],
    ]);
    const { fetchImpl, calls } = createMockDataFetch(backing);
    const liveReport = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      signal: controller.signal,
    });

    assert.equal(liveReport.ok, false);
    assert.equal(liveReport.aborted, true);
    assert.equal(liveReport.telemetry.aborted, true);
    assert.equal(liveReport.probed, 0);
    assert.equal(liveReport.filesVerified, 0);
    assert.equal(liveReport.bytesVerified, 0);
    assert.equal(calls.length, 0);
  });

  it('cancels verifyLiveEndpoints mid-run and mid-request via AbortSignal and forwards signal to fetchImpl', async () => {
    const backing = new Map([
      ['f0.bin', { bytes: new Uint8Array(64).fill(1), etag: '"e0"' }],
      ['f1.bin', { bytes: new Uint8Array(64).fill(2), etag: '"e1"' }],
      ['f2.bin', { bytes: new Uint8Array(64).fill(3), etag: '"e2"' }],
    ]);
    const manifest = makeValidManifest({
      files: [
        { path: 'f0.bin', size: 64, etag: '"e0"' },
        { path: 'f1.bin', size: 64, etag: '"e1"' },
        { path: 'f2.bin', size: 64, etag: '"e2"' },
      ],
    });

    // Case 1: Abort during f1.bin HEAD request; fetchImpl ignores signal and hangs forever,
    // verifying raceWithAbort cancels promptly and preserves completed f0.bin telemetry.
    const controller = new AbortController();
    const seenSignals = [];
    const seenPaths = [];
    const { fetchImpl: baseFetch } = createMockDataFetch(backing);

    const fetchWithHang = async (url, init = {}) => {
      seenSignals.push(init.signal);
      seenPaths.push(`${init.method || 'GET'} ${url}`);
      if (url.endsWith('f1.bin') && init.method === 'HEAD') {
        setTimeout(() => controller.abort(), 5);
        return new Promise(() => {}); // Never settles on its own
      }
      return baseFetch(url, init);
    };

    const report = await verifyLiveEndpoints({
      fetchImpl: fetchWithHang,
      dataBase: '/data/',
      manifest,
      sampleCount: 3,
      maxConcurrency: 1,
      signal: controller.signal,
    });

    assert.equal(report.ok, false);
    assert.equal(report.aborted, true);
    assert.equal(report.telemetry.aborted, true);
    assert.equal(report.probed, 1);
    assert.equal(report.passed, 1);
    assert.equal(report.failed, 0);
    assert.equal(report.filesVerified, 1);
    assert.equal(report.bytesVerified, 1);
    assert.equal(report.mismatchCount, 0);
    assert.ok(seenSignals.every((s) => s === controller.signal));
    assert.ok(seenPaths.every((p) => !p.includes('f2.bin')));
    assert.ok(seenPaths.every((p) => !(p.startsWith('GET') && p.includes('f1.bin'))));
  });

  it('never leaks host filesystem paths or raw file bytes in verification diagnostics or telemetry', async () => {
    const rawSecretPayload = 'RAW_SECRET_FILE_PAYLOAD_BYTES_DO_NOT_LEAK_987654321';
    const secretBytes = new TextEncoder().encode(rawSecretPayload);
    const backing = new Map([
      ['secret.bin', { bytes: secretBytes, etag: '"sec-1"' }],
      ['win_err.bin', { bytes: new Uint8Array(16), etag: '"w1"' }],
      ['posix_err.bin', { bytes: new Uint8Array(16), etag: '"p1"' }],
    ]);
    const manifest = makeValidManifest({
      files: [
        {
          path: 'secret.bin',
          size: secretBytes.byteLength,
          sha256: 'f'.repeat(64), // intentional hash mismatch
          etag: '"sec-1"',
        },
        { path: 'win_err.bin', size: 16, etag: '"w1"' },
        { path: 'posix_err.bin', size: 16, etag: '"p1"' },
      ],
    });

    const { fetchImpl } = createMockDataFetch(backing, {
      onRequest: async ({ relPath }) => {
        if (relPath === 'win_err.bin') {
          throw new Error(
            "ENOENT: failed reading 'C:\\Users\\Alice Smith\\Games\\GTA6\\x64\\levels.rpf' and \\\\fileserver\\private\\dump.bin\x00\x01",
          );
        }
        if (relPath === 'posix_err.bin') {
          throw new Error(
            'EACCES: open "/Users/bob/private/gta6/levels.rpf" or file:///home/dev/secret.rpf',
          );
        }
        return null;
      },
    });

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 5,
      verifyHashes: true,
    });

    const serialized = JSON.stringify(report);
    assert.equal(serialized.includes(rawSecretPayload), false);
    assert.equal(serialized.includes('C:\\Users'), false);
    assert.equal(serialized.includes('Alice Smith'), false);
    assert.equal(serialized.includes('fileserver'), false);
    assert.equal(serialized.includes('/Users/bob'), false);
    assert.equal(serialized.includes('/home/dev'), false);
    assert.equal(serialized.includes('file://'), false);
    assert.equal(/[\x00-\x1f\x7f]/.test(serialized), false);
    assert.ok(serialized.includes('<redacted-path>'));
    assert.equal(report.mismatchesByCategory.hash_mismatch, 1);
  });
});

function makeSyntheticPayload(byteLength, seed = 0x5a) {
  const out = new Uint8Array(byteLength);
  let state = (seed >>> 0) || 1;
  for (let i = 0; i < byteLength; i += 1) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    out[i] = (state >>> 24) & 0xff;
  }
  return out;
}

function splitIntoChunks(bytes, sizes) {
  const chunks = [];
  let offset = 0;
  let sizeIdx = 0;
  while (offset < bytes.byteLength) {
    const requested = sizes[sizeIdx % sizes.length];
    sizeIdx += 1;
    if (requested === 0) {
      chunks.push(new Uint8Array(0));
      continue;
    }
    const end = Math.min(bytes.byteLength, offset + requested);
    chunks.push(bytes.subarray(offset, end));
    offset = end;
  }
  return chunks;
}

function createChunkReadableStream(chunks, options = {}) {
  let idx = 0;
  let cancelled = false;
  let pulls = 0;
  const stream = new ReadableStream({
    pull(controller) {
      pulls += 1;
      if (options.hangAtIndex !== undefined && idx === options.hangAtIndex) {
        if (options.onHang) options.onHang();
        return new Promise(() => {});
      }
      if (idx >= chunks.length) {
        controller.close();
        return;
      }
      controller.enqueue(chunks[idx]);
      idx += 1;
    },
    cancel() {
      cancelled = true;
    },
  });
  return {
    stream,
    get pulls() {
      return pulls;
    },
    get cancelled() {
      return cancelled;
    },
  };
}

describe('incremental streaming chunk verification & schema diagnostics (W221)', () => {
  it('exports frozen STREAM_HASH_ALGORITHMS and SCHEMA_DIAGNOSTIC_CODES constants', () => {
    assert.deepEqual(STREAM_HASH_ALGORITHMS, {
      SHA256: 'sha256',
      CRC32: 'crc32',
      FNV1A32: 'fnv1a32',
      FNV1A64: 'fnv1a64',
    });
    assert.equal(Object.isFrozen(STREAM_HASH_ALGORITHMS), true);
    assert.equal(Object.isFrozen(SCHEMA_DIAGNOSTIC_CODES), true);
    assert.equal(SCHEMA_DIAGNOSTIC_CODES.TOTAL_BYTES_MISMATCH, 'TOTAL_BYTES_MISMATCH');
    assert.equal(SCHEMA_DIAGNOSTIC_CODES.CHUNK_HASH_COUNT_MISMATCH, 'CHUNK_HASH_COUNT_MISMATCH');
  });

  it('matches known reference vectors and Node crypto SHA-256 across all block boundary sizes', () => {
    const enc = new TextEncoder();

    // Known reference vectors
    assert.equal(computeDigest(new Uint8Array(0), 'sha256'), sha256Hex(new Uint8Array(0)));
    assert.equal(computeDigest(enc.encode('abc'), 'SHA-256'), sha256Hex(enc.encode('abc')));

    assert.equal(computeDigest(new Uint8Array(0), 'crc32'), '00000000');
    assert.equal(computeDigest(enc.encode('123456789'), 'CRC-32'), 'cbf43926');

    assert.equal(computeDigest(new Uint8Array(0), 'fnv1a32'), '811c9dc5');
    assert.equal(computeDigest(enc.encode('a'), 'fnv-1a'), 'e40c292c');
    assert.equal(computeDigest(enc.encode('foobar'), 'fnv1a32'), 'bf9cf968');

    assert.equal(computeDigest(new Uint8Array(0), 'fnv1a64'), 'cbf29ce484222325');
    assert.equal(computeDigest(enc.encode('a'), 'fnv1a-64'), 'af63dc4c8601ec8c');
    assert.equal(computeDigest(enc.encode('foobar'), 'fnv1a64'), '85944171f73967e8');

    // Every critical SHA-256 padding/block boundary: 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, etc.
    const boundarySizes = [0, 1, 2, 54, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 1024, 65537];
    for (const size of boundarySizes) {
      const payload = makeSyntheticPayload(size, size + 17);
      const expected = sha256Hex(payload);
      const actual = computeDigest(payload, 'sha256');
      assert.equal(actual, expected, `SHA-256 mismatch at boundary size=${size}`);
    }
  });

  it('matches independent BigInt FNV64 across carries, subviews and chunk-hasher resets', () => {
    const reference = (bytes) => {
      let h = 0xcbf29ce484222325n;
      for (const byte of bytes) h = ((h ^ BigInt(byte)) * 0x100000001b3n) & 0xffffffffffffffffn;
      return h.toString(16).padStart(16, '0');
    };
    const inputs = [new Uint8Array(0), new Uint8Array(8193).fill(255)];
    for (const size of [1, 7, 255, 256, 4097, 65537]) {
      inputs.push(makeSyntheticPayload(size + 11, size).subarray(3, size + 3));
    }
    for (const payload of inputs) {
      const expected = reference(payload);
      assert.equal(computeDigest(payload, 'fnv1a64'), expected);
      const chunkSize = 257;
      const expectedChunkDigests = [];
      for (let offset = 0; offset < payload.length; offset += chunkSize) {
        expectedChunkDigests.push(reference(payload.subarray(offset, offset + chunkSize)));
      }
      const verifier = createIncrementalVerifier({
        algorithm: 'fnv1a64', expectedDigest: expected, expectedSize: payload.length,
        chunkSize, expectedChunkDigests,
      });
      for (let offset = 0; offset < payload.length; offset += 113) {
        verifier.update(payload.subarray(offset, offset + 113));
      }
      const result = verifier.finalize();
      assert.equal(result.digest, expected);
      assert.equal(result.ok, true);
    }
  });

  it('guarantees single-shot vs multi-chunk streaming equivalence across algorithms, view types, and chunkings', async () => {
    const payload = makeSyntheticPayload(131072 + 373, 0x221); // ~128 KiB synthetic .pgim payload
    const algorithms = [
      STREAM_HASH_ALGORITHMS.SHA256,
      STREAM_HASH_ALGORITHMS.CRC32,
      STREAM_HASH_ALGORITHMS.FNV1A32,
      STREAM_HASH_ALGORITHMS.FNV1A64,
    ];

    for (const algorithm of algorithms) {
      const singleShot = computeDigest(payload, algorithm);
      if (algorithm === STREAM_HASH_ALGORITHMS.SHA256) {
        assert.equal(singleShot, sha256Hex(payload));
      }

      // 1) Irregular prime-sized chunks with interleaved empty 0-byte chunks
      const verifier = createIncrementalVerifier({
        algorithm,
        expectedDigest: singleShot,
        expectedSize: payload.byteLength,
        path: 'guest/image.pgim',
      });
      const unevenChunks = splitIntoChunks(payload, [0, 1, 7, 13, 63, 64, 65, 0, 509, 4099, 16384]);
      for (const chunk of unevenChunks) {
        const step = verifier.update(chunk);
        assert.equal(step.ok, true);
        assert.equal(step.aborted, false);
      }
      const res = verifier.finalize();
      assert.equal(res.ok, true, `uneven chunk verification failed for ${algorithm}`);
      assert.equal(res.digest, singleShot);
      assert.equal(res.expected, singleShot);
      assert.equal(res.actual, singleShot);
      assert.equal(res.checkedBytes, payload.byteLength);
      assert.equal(res.sizeVerified, true);
      assert.equal(res.digestVerified, true);
      assert.equal(res.mismatch, null);
      assert.deepEqual(res.mismatches, []);

      // 2) ReadableStream via verifyStream
      const { stream } = createChunkReadableStream(splitIntoChunks(payload, [257, 1024, 8192]));
      const streamRes = await verifyStream(stream, {
        algorithm,
        expectedDigest: singleShot,
        expectedSize: payload.byteLength,
        path: 'shaders/boot.wgsl.pack',
      });
      assert.equal(streamRes.ok, true);
      assert.equal(streamRes.digest, singleShot);
      assert.equal(streamRes.checkedBytes, payload.byteLength);

      // 3) AsyncIterable yielding mixed ArrayBuffer, DataView (non-zero byteOffset), and SharedArrayBuffer views
      async function* mixedViewStream() {
        const split1 = 100;
        const split2 = 500;
        const split3 = 2000;
        // Raw ArrayBuffer
        yield payload.slice(0, split1).buffer;
        // DataView with non-zero byteOffset
        const padded = new Uint8Array(split2 - split1 + 23);
        padded.set(payload.subarray(split1, split2), 11);
        yield new DataView(padded.buffer, 11, split2 - split1);
        // SharedArrayBuffer (when available)
        if (typeof SharedArrayBuffer !== 'undefined') {
          const sab = new SharedArrayBuffer(split3 - split2);
          new Uint8Array(sab).set(payload.subarray(split2, split3));
          yield sab;
        } else {
          yield payload.subarray(split2, split3);
        }
        // Remaining Uint8Array slice
        yield payload.subarray(split3);
      }

      const asyncRes = await verifyStream(mixedViewStream(), {
        algorithm,
        expectedDigest: singleShot,
        expectedSize: payload.byteLength,
      });
      assert.equal(asyncRes.ok, true);
      assert.equal(asyncRes.digest, singleShot);
      assert.equal(asyncRes.checkedBytes, payload.byteLength);
    }

    // 1-byte-at-a-time streaming on a 300-byte payload (crosses multiple 64-byte SHA-256 blocks)
    const small = makeSyntheticPayload(300, 99);
    const byteChunks = splitIntoChunks(small, [1]);
    const byteRes = await verifyStream(byteChunks, {
      algorithm: 'sha256',
      expectedDigest: sha256Hex(small),
      expectedSize: small.byteLength,
    });
    assert.equal(byteRes.ok, true);
    assert.equal(byteRes.chunkCount, 300);
    assert.equal(byteRes.checkedBytes, 300);
  });

  it('detects corrupted chunks with structured mismatch diagnostics (expected, actual, checkedBytes, elapsedMs) and supports failFast', async () => {
    const chunkSize = 4096;
    const payload = makeSyntheticPayload(chunkSize * 4 + 512, 0x77); // 4 full blocks + 1 trailing partial block
    const expectedFullDigest = computeDigest(payload, 'sha256');
    const expectedBlockDigests = [];
    for (let off = 0; off < payload.byteLength; off += chunkSize) {
      expectedBlockDigests.push(
        computeDigest(payload.subarray(off, Math.min(payload.byteLength, off + chunkSize)), 'sha256'),
      );
    }

    // Corrupt 1 byte inside block #2 (offset 9000)
    const corrupted = payload.slice();
    corrupted[chunkSize * 2 + 808] ^= 0x01;
    const actualFullDigest = computeDigest(corrupted, 'sha256');
    const actualBlock2Digest = computeDigest(
      corrupted.subarray(chunkSize * 2, chunkSize * 3),
      'sha256',
    );

    let clock = 1000;
    const stepClock = () => {
      clock += 10;
      return clock;
    };

    // Stream with unaligned network chunk sizes (e.g., 1500 bytes each) across 4096-byte hash blocks!
    const networkChunks = splitIntoChunks(corrupted, [1500]);
    const verifier = createIncrementalVerifier({
      algorithm: 'sha256',
      expectedDigest: expectedFullDigest,
      expectedSize: payload.byteLength,
      chunkSize,
      expectedChunkDigests: expectedBlockDigests,
      path: 'assets/world.pgim',
      nowFn: stepClock,
    });

    const detectedEarly = [];
    for (const chunk of networkChunks) {
      const updateRes = verifier.update(chunk);
      if (updateRes.mismatch) {
        detectedEarly.push(updateRes.mismatch);
      }
    }
    const report = verifier.finalize();

    assert.equal(report.ok, false);
    assert.equal(report.digestVerified, false);
    assert.equal(report.sizeVerified, true);
    assert.equal(report.checkedBytes, corrupted.byteLength);
    assert.equal(report.expected, expectedFullDigest);
    assert.equal(report.actual, actualFullDigest);
    assert.equal(report.verifiedBlocks, 4); // blocks 0, 1, 3, 4 matched; block 2 failed
    assert.equal(report.chunkMismatches.length, 1);
    assert.equal(detectedEarly.length, 1);

    const chunkErr = report.chunkMismatches[0];
    assert.equal(chunkErr.category, MISMATCH_CATEGORIES.HASH_MISMATCH);
    assert.equal(chunkErr.kind, 'chunk_digest_mismatch');
    assert.equal(chunkErr.path, 'assets/world.pgim');
    assert.equal(chunkErr.chunkIndex, 2);
    assert.equal(chunkErr.offset, chunkSize * 2);
    assert.equal(chunkErr.length, chunkSize);
    assert.equal(chunkErr.expected, expectedBlockDigests[2]);
    assert.equal(chunkErr.actual, actualBlock2Digest);
    assert.ok(chunkErr.checkedBytes >= chunkSize * 3);
    assert.ok(chunkErr.elapsedMs > 0);

    // Final digest mismatch is also recorded with structured fields.
    const digestErr = report.mismatches.find((m) => m.kind === 'digest_mismatch');
    assert.ok(digestErr);
    assert.equal(digestErr.expected, expectedFullDigest);
    assert.equal(digestErr.actual, actualFullDigest);
    assert.equal(digestErr.checkedBytes, corrupted.byteLength);
    assert.ok(digestErr.elapsedMs >= chunkErr.elapsedMs);
    assert.equal(report.mismatchesByCategory.hash_mismatch, 2);

    // failFast=true cancels ReadableStream immediately after block #2 fails without pulling tail chunks
    const fastStream = createChunkReadableStream(
      splitIntoChunks(corrupted, [chunkSize]),
    );
    const fastRes = await verifyStream(fastStream.stream, {
      algorithm: 'sha256',
      expectedDigest: expectedFullDigest,
      expectedSize: payload.byteLength,
      chunkSize,
      expectedChunkDigests: expectedBlockDigests,
      path: 'assets/world.pgim',
      failFast: true,
    });
    assert.equal(fastRes.ok, false);
    assert.equal(fastStream.cancelled, true);
    assert.equal(fastRes.chunkMismatches.length, 1);
    assert.equal(fastRes.chunkMismatches[0].chunkIndex, 2);
    assert.equal(fastRes.checkedBytes, chunkSize * 3);
    assert.equal(fastRes.chunkCount, 3);
    assert.equal(fastRes.digest, null);
    assert.equal(fastRes.digestVerified, false);
  });

  it('detects per-update chunk mismatches, truncated streams, oversized streams, and chunk count drift', () => {
    const chunks = [
      new Uint8Array([1, 2, 3, 4]),
      new Uint8Array([5, 6, 7, 8]),
      new Uint8Array([9, 10, 11, 12]),
    ];
    const digests = chunks.map((c) => computeDigest(c, 'crc32'));

    // Missing trailing chunk + truncated size
    const truncated = createIncrementalVerifier({
      algorithm: 'crc32',
      expectedSize: 12,
      expectedChunkDigests: digests,
      path: 'shaders/pack.wgsl.pack',
    });
    assert.equal(truncated.update(chunks[0]).ok, true);
    assert.equal(truncated.update(chunks[1]).ok, true);
    const truncRes = truncated.finalize();
    assert.equal(truncRes.ok, false);
    assert.equal(truncRes.sizeVerified, false);
    assert.equal(truncRes.checkedBytes, 8);
    assert.ok(truncRes.mismatches.some((m) => m.kind === 'chunk_count_mismatch' && m.expected === 3 && m.actual === 2));
    assert.ok(truncRes.mismatches.some((m) => m.kind === 'size_mismatch' && m.expected === 12 && m.actual === 8));

    // Oversized stream triggers early size_mismatch during update()
    const oversized = createIncrementalVerifier({
      algorithm: 'fnv1a32',
      expectedSize: 6,
      path: 'shaders/pack.wgsl.pack',
    });
    assert.equal(oversized.update(chunks[0]).ok, true);
    const overflowStep = oversized.update(chunks[1]);
    assert.equal(overflowStep.ok, false);
    assert.equal(overflowStep.mismatch.category, MISMATCH_CATEGORIES.SIZE_MISMATCH);
    assert.equal(overflowStep.mismatch.expected, 6);
    assert.equal(overflowStep.mismatch.actual, 8);
    assert.equal(overflowStep.mismatch.checkedBytes, 8);

    const overRes = oversized.finalize();
    assert.equal(overRes.ok, false);
    assert.equal(overRes.sizeVerified, false);
    assert.equal(overRes.expected, 6);
    assert.equal(overRes.actual, 8);
    assert.equal(overRes.mismatchesByCategory.size_mismatch, 1);
  });

  it('handles empty/boundary inputs, finalize idempotency, reset reuse, entry shorthand, and invalid arguments', () => {
    // 0-byte verification with no update() calls
    const emptyVerifier = createIncrementalVerifier({
      algorithm: 'sha256',
      expectedDigest: sha256Hex(new Uint8Array(0)),
      expectedSize: 0,
      chunkSize: 4096,
      expectedChunkDigests: [],
      path: 'empty.bin',
    });
    const res1 = emptyVerifier.finalize();
    const res2 = emptyVerifier.finalize();
    assert.strictEqual(res1, res2); // idempotent cached result
    assert.equal(res1.ok, true);
    assert.equal(res1.checkedBytes, 0);
    assert.equal(res1.chunkCount, 0);
    assert.equal(res1.sizeVerified, true);
    assert.equal(res1.digestVerified, true);

    // Calling update() after finalize() throws; reset() enables reuse
    assert.throws(() => emptyVerifier.update(new Uint8Array([1])), /already been finalized/);
    emptyVerifier.reset();
    assert.equal(emptyVerifier.update(new Uint8Array(0)).ok, true);
    const resAfterReset = emptyVerifier.finalize();
    assert.equal(resAfterReset.ok, true);
    assert.equal(resAfterReset.chunkCount, 1);
    assert.equal(resAfterReset.checkedBytes, 0);

    // Manifest entry shorthand auto-selects algorithm & expectations
    const buf = new TextEncoder().encode('PGIM_TEST_PAYLOAD');
    const crc = computeDigest(buf, 'crc32');
    const manifestCheck = validateDataManifest({
      abi_version: 1,
      file_count: 1,
      total_bytes: buf.byteLength,
      files: [{ path: 'guest/boot.pgim', size: buf.byteLength, crc32: crc }],
    });
    assert.equal(manifestCheck.ok, true);
    const entry = manifestCheck.fileMap.get('guest/boot.pgim');
    const fromEntry = createIncrementalVerifier({ entry });
    assert.equal(fromEntry.algorithm, 'crc32');
    fromEntry.update(buf.subarray(0, 5));
    fromEntry.update(buf.subarray(5));
    const entryRes = fromEntry.finalize();
    assert.equal(entryRes.ok, true);
    assert.equal(entryRes.digest, crc);
    assert.equal(entryRes.path, 'guest/boot.pgim');

    // Validation errors on bad arguments
    assert.throws(() => createIncrementalVerifier({ algorithm: 'md5' }), RangeError);
    assert.throws(() => createIncrementalVerifier({ algorithm: 'sha256', expectedDigest: '1234' }), RangeError);
    assert.throws(() => createIncrementalVerifier({ expectedSize: -5 }), RangeError);
    assert.throws(() => createIncrementalVerifier({ chunkSize: 0 }), RangeError);
    assert.throws(() => createIncrementalVerifier({ expectedChunkDigests: 'not-array' }), TypeError);
    assert.throws(() => createIncrementalVerifier().update('not-a-buffer'), TypeError);
  });

  it('reports deterministic telemetry, progress snapshots, AbortSignal cancellation, and sanitized paths', async () => {
    let clock = 2000;
    const nowFn = () => {
      clock += 25;
      return clock;
    };

    const payload = makeSyntheticPayload(1000, 0x33);
    const expected = computeDigest(payload, 'fnv1a64');
    const seenUpdates = [];

    const verifier = createIncrementalVerifier({
      algorithm: 'fnv1a64',
      expectedDigest: expected,
      expectedSize: 1000,
      path: 'C:\\Users\\Alice\\Private\\image.pgim',
      nowFn,
      onChunk: (u) => seenUpdates.push(u),
    });

    verifier.update(payload.subarray(0, 400));
    const midSnap = verifier.getSnapshot();
    assert.equal(midSnap.checkedBytes, 400);
    assert.equal(midSnap.chunkCount, 1);
    assert.equal(midSnap.finalized, false);
    assert.ok(midSnap.elapsedMs > 0);
    assert.ok(midSnap.bytesPerSecond > 0);
    assert.equal(midSnap.path.includes('Alice'), false);
    assert.ok(midSnap.path.includes('<redacted-path>'));

    verifier.update(payload.subarray(400));
    const finalRes = verifier.finalize();
    assert.equal(finalRes.ok, true);
    assert.equal(finalRes.checkedBytes, 1000);
    assert.equal(finalRes.bytesVerified, 1000);
    assert.equal(finalRes.filesVerified, 1);
    assert.equal(finalRes.chunkCount, 2);
    assert.ok(finalRes.elapsedMs > 0);
    assert.ok(finalRes.bytesPerSecond > 0);
    assert.equal(finalRes.telemetry.checkedBytes, 1000);
    assert.equal(finalRes.telemetry.chunkCount, 2);
    assert.equal(seenUpdates.length, 2);

    // Abort mid-stream on a hanging ReadableStream cancels cleanly
    const controller = new AbortController();
    const abortStream = createChunkReadableStream(
      [payload.subarray(0, 200), payload.subarray(200, 400)],
      {
        hangAtIndex: 1,
        onHang: () => setTimeout(() => controller.abort(new Error('cancelled')), 5),
      },
    );
    const abortedRes = await verifyStream(abortStream.stream, {
      algorithm: 'sha256',
      expectedDigest: sha256Hex(payload),
      expectedSize: 1000,
      signal: controller.signal,
    });
    assert.equal(abortedRes.ok, false);
    assert.equal(abortedRes.aborted, true);
    assert.equal(abortedRes.telemetry.aborted, true);
    assert.equal(abortedRes.checkedBytes, 200);
    assert.equal(abortedRes.filesVerified, 0);
    assert.equal(abortStream.cancelled, true);
  });

  it('streams full response body in verifyLiveEndpoints without calling arrayBuffer() when body.getReader() is available', async () => {
    const payload = makeSyntheticPayload(20000, 0x44);
    const expectedSha = sha256Hex(payload);
    const manifest = makeValidManifest({
      files: [{ path: 'shaders/world.wgsl.pack', size: payload.byteLength, sha256: expectedSha, etag: '"pack-1"' }],
    });

    let fullArrayBufferCalled = false;
    const fetchImpl = async (url, init = {}) => {
      const method = (init.method || 'GET').toUpperCase();
      const rangeHeader = init.headers?.Range || init.headers?.range || null;
      const baseHeaders = {
        'Accept-Ranges': 'bytes',
        ETag: '"pack-1"',
      };
      if (method === 'HEAD') {
        return {
          status: 200,
          ok: true,
          headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(payload.byteLength) }),
        };
      }
      if (rangeHeader) {
        const slice = payload.subarray(0, 1);
        return {
          status: 206,
          ok: true,
          headers: makeMockHeaders({
            ...baseHeaders,
            'Content-Range': `bytes 0-0/${payload.byteLength}`,
            'Content-Length': '1',
          }),
          async arrayBuffer() {
            return slice.buffer.slice(slice.byteOffset, slice.byteOffset + slice.byteLength);
          },
        };
      }
      const { stream } = createChunkReadableStream(splitIntoChunks(payload, [1024, 3000, 4096]));
      return {
        status: 200,
        ok: true,
        headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(payload.byteLength) }),
        body: stream,
        async arrayBuffer() {
          fullArrayBufferCalled = true;
          throw new Error('arrayBuffer() must not be called when streaming body.getReader()');
        },
      };
    };

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      verifyHashes: true,
    });

    assert.equal(fullArrayBufferCalled, false);
    assert.equal(report.ok, true);
    assert.equal(report.passed, 1);
    assert.equal(report.bytesVerified, payload.byteLength);
    assert.equal(report.results[0].hash.verified, true);
    assert.equal(report.results[0].hash.expected, expectedSha);
    assert.equal(report.results[0].hash.actual, expectedSha);
    assert.equal(report.results[0].hash.checkedBytes, payload.byteLength);
    assert.equal(typeof report.results[0].hash.elapsedMs, 'number');
  });

  it('emits structured, sanitized schemaDiagnostics for manifest streaming hash fields and bootset issues', () => {
    const manifestRes = validateDataManifest({
      abi_version: 1,
      file_count: 2,
      total_bytes: 99999,
      files: [
        {
          path: 'guest/image.pgim',
          size: 8192,
          crc32: 'ZZZZZZZZ', // invalid hex
          fnv1a32: '1234', // wrong length
          fnv1a64: 'not-hex-64',
          chunk_size: 4096,
          chunk_hashes: ['a'.repeat(64)], // expects 2 chunks for 8192/4096
        },
        {
          path: 'C:/Users/Alice/secret/levels.rpf', // host path must be redacted in diagnostics
          size: 100,
        },
      ],
    });

    assert.equal(manifestRes.ok, false);
    assert.ok(Array.isArray(manifestRes.schemaDiagnostics));
    const codes = new Set(manifestRes.schemaDiagnostics.map((d) => d.code));
    assert.ok(codes.has(SCHEMA_DIAGNOSTIC_CODES.INVALID_CRC32));
    assert.ok(codes.has(SCHEMA_DIAGNOSTIC_CODES.INVALID_FNV1A32));
    assert.ok(codes.has(SCHEMA_DIAGNOSTIC_CODES.INVALID_FNV1A64));
    assert.ok(codes.has(SCHEMA_DIAGNOSTIC_CODES.CHUNK_HASH_COUNT_MISMATCH));
    assert.ok(codes.has(SCHEMA_DIAGNOSTIC_CODES.INVALID_PATH));
    assert.ok(codes.has(SCHEMA_DIAGNOSTIC_CODES.TOTAL_BYTES_MISMATCH));

    const chunkCountDiag = manifestRes.schemaDiagnostics.find(
      (d) => d.code === SCHEMA_DIAGNOSTIC_CODES.CHUNK_HASH_COUNT_MISMATCH,
    );
    assert.equal(chunkCountDiag.expected, 2);
    assert.equal(chunkCountDiag.actual, 1);
    assert.equal(chunkCountDiag.path, 'guest/image.pgim');

    const serializedDiags = JSON.stringify(manifestRes.schemaDiagnostics);
    assert.equal(serializedDiags.includes('C:/Users/Alice'), false);
    assert.ok(serializedDiags.includes('<redacted-path>'));

    // Bootset structured schemaDiagnostics
    const bootRes = validateBootset(
      [
        { path: 'MicrosoftGame.Config', offset: 0, length: 9999 },
        { path: 'missing.rpf', offset: 0, length: 64 },
      ],
      makeValidManifest(),
    );
    assert.equal(bootRes.ok, false);
    const bootCodes = new Set(bootRes.schemaDiagnostics.map((d) => d.code));
    assert.ok(bootCodes.has(SCHEMA_DIAGNOSTIC_CODES.RANGE_OUT_OF_BOUNDS));
    assert.ok(bootCodes.has(SCHEMA_DIAGNOSTIC_CODES.MISSING_MANIFEST_FILE));
  });
});

describe('bounded concurrency, streaming reader hashing, AbortSignal propagation & telemetry (W278)', () => {
  it('exports concurrency bound constants and clamps maxConcurrency to [1, 16] with default 4', async () => {
    assert.equal(DEFAULT_MAX_CONCURRENCY, 4);
    assert.equal(MIN_MAX_CONCURRENCY, 1);
    assert.equal(MAX_MAX_CONCURRENCY, 16);

    const backing = new Map([['a.bin', { bytes: new Uint8Array(16).fill(1), etag: '"a"' }]]);
    const manifest = makeValidManifest({ files: [{ path: 'a.bin', size: 16, etag: '"a"' }] });
    const { fetchImpl } = createMockDataFetch(backing);

    const cases = [
      { input: undefined, expected: 4 },
      { input: null, expected: 4 },
      { input: Number.NaN, expected: 4 },
      { input: 'invalid', expected: 4 },
      { input: 0, expected: 1 },
      { input: -8, expected: 1 },
      { input: -Infinity, expected: 1 },
      { input: 2.9, expected: 2 },
      { input: 16, expected: 16 },
      { input: 64, expected: 16 },
      { input: Infinity, expected: 16 },
    ];

    for (const tc of cases) {
      const report = await verifyLiveEndpoints({
        fetchImpl,
        dataBase: '/data/',
        manifest,
        sampleCount: 1,
        maxConcurrency: tc.input,
      });
      assert.equal(report.ok, true);
      assert.equal(
        report.maxConcurrency,
        tc.expected,
        `expected maxConcurrency(${String(tc.input)}) -> ${tc.expected}`,
      );
      assert.equal(report.telemetry.maxConcurrency, tc.expected);
    }
  });

  it('enforces bounded concurrency across default (4) and explicit maxConcurrency while preserving manifest order', async () => {
    resetStats();

    const fileCount = 10;
    const files = [];
    const payloads = new Map();
    let expectedTotalBytes = 0;
    for (let i = 0; i < fileCount; i += 1) {
      const p = `x64/archive_${String(i).padStart(2, '0')}.rpf`;
      const bytes = makeSyntheticPayload(4096 + i * 257, 0x100 + i);
      const sha = sha256Hex(bytes);
      const etag = `"arch-${i}"`;
      files.push({ path: p, size: bytes.byteLength, sha256: sha, etag });
      payloads.set(p, { bytes, etag });
      expectedTotalBytes += bytes.byteLength;
    }
    const manifest = makeValidManifest({ files });

    const runWithConcurrency = async (maxConcurrencyOpt) => {
      let inFlight = 0;
      let peakInFlight = 0;
      let arrayBufferCalled = 0;

      const fetchImpl = async (url, init = {}) => {
        const method = (init.method || 'GET').toUpperCase();
        const rangeHeader = init.headers?.Range || init.headers?.range || null;
        const relPath = url.replace(/^\/data\//, '');
        const entry = payloads.get(relPath);
        const baseHeaders = {
          'Accept-Ranges': 'bytes',
          ETag: entry.etag,
        };

        if (method === 'HEAD') {
          return {
            status: 200,
            ok: true,
            headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(entry.bytes.byteLength) }),
          };
        }

        if (rangeHeader) {
          const { stream } = createChunkReadableStream([entry.bytes.subarray(0, 1)]);
          return {
            status: 206,
            ok: true,
            headers: makeMockHeaders({
              ...baseHeaders,
              'Content-Range': `bytes 0-0/${entry.bytes.byteLength}`,
              'Content-Length': '1',
            }),
            body: stream,
            async arrayBuffer() {
              arrayBufferCalled += 1;
              throw new Error('Range probe must stream via body.getReader()');
            },
          };
        }

        // Full GET for SHA-256: track active concurrency across async chunk delivery.
        const chunks = splitIntoChunks(entry.bytes, [1024, 1536, 2048]);
        let idx = 0;
        let activeForThisStream = false;
        const stream = new ReadableStream({
          async pull(controller) {
            if (!activeForThisStream) {
              activeForThisStream = true;
              inFlight += 1;
              if (inFlight > peakInFlight) {
                peakInFlight = inFlight;
              }
            }
            await new Promise((r) => setTimeout(r, 4));
            if (idx >= chunks.length) {
              if (activeForThisStream) {
                activeForThisStream = false;
                inFlight -= 1;
              }
              controller.close();
              return;
            }
            controller.enqueue(chunks[idx]);
            idx += 1;
          },
          cancel() {
            if (activeForThisStream) {
              activeForThisStream = false;
              inFlight -= 1;
            }
          },
        });

        return {
          status: 200,
          ok: true,
          headers: makeMockHeaders({
            ...baseHeaders,
            'Content-Length': String(entry.bytes.byteLength),
          }),
          body: stream,
          async arrayBuffer() {
            arrayBufferCalled += 1;
            throw new Error('Full GET must stream via body.getReader()');
          },
        };
      };

      const opts = {
        fetchImpl,
        dataBase: '/data/',
        manifest,
        sampleCount: fileCount,
        verifyHashes: true,
      };
      if (maxConcurrencyOpt !== undefined) {
        opts.maxConcurrency = maxConcurrencyOpt;
      }

      const report = await verifyLiveEndpoints(opts);
      return { report, peakInFlight, arrayBufferCalled };
    };

    // 1. Default maxConcurrency (4)
    const defaultRun = await runWithConcurrency(undefined);
    assert.equal(defaultRun.report.ok, true);
    assert.equal(defaultRun.report.maxConcurrency, 4);
    assert.equal(defaultRun.report.passed, fileCount);
    assert.equal(defaultRun.report.bytesHashed, expectedTotalBytes);
    assert.equal(defaultRun.arrayBufferCalled, 0);
    assert.ok(
      defaultRun.peakInFlight >= 2 && defaultRun.peakInFlight <= 4,
      `expected default concurrency in [2, 4], got ${defaultRun.peakInFlight}`,
    );
    assert.ok(
      defaultRun.report.peakConcurrentHashes >= 2 && defaultRun.report.peakConcurrentHashes <= 4,
      `expected peakConcurrentHashes in [2, 4], got ${defaultRun.report.peakConcurrentHashes}`,
    );
    assert.deepEqual(
      defaultRun.report.results.map((r) => r.path),
      files.map((f) => f.path),
    );

    // 2. Explicit maxConcurrency = 2
    resetStats();
    const boundedRun = await runWithConcurrency(2);
    assert.equal(boundedRun.report.ok, true);
    assert.equal(boundedRun.report.maxConcurrency, 2);
    assert.equal(boundedRun.arrayBufferCalled, 0);
    assert.equal(boundedRun.peakInFlight, 2);
    assert.equal(boundedRun.report.peakConcurrentHashes, 2);

    const stats = getStats();
    assert.equal(stats.filesVerified, fileCount);
    assert.equal(stats.bytesHashed, expectedTotalBytes);
    assert.equal(stats.mismatchCount, 0);
    assert.equal(stats.abortedCount, 0);
    assert.equal(stats.peakConcurrentHashes, 2);
    assert.equal(stats.activeHashes, 0);
  });

  it('streams Range GET directly via body.getReader() without arrayBuffer() and cancels non-206 bodies', async () => {
    resetStats();
    const smallPayload = makeSyntheticPayload(12000, 0x99);
    const smallSha = sha256Hex(smallPayload);
    const manifest = makeValidManifest({
      files: [
        { path: 'shaders/small.wgsl.pack', size: smallPayload.byteLength, sha256: smallSha, etag: '"s1"' },
        { path: 'x64/no_range_server.rpf', size: 1000000, etag: '"nr1"' },
      ],
    });

    let arrayBufferCalled = 0;
    let fullGetCalledForSmall = false;
    let non206BodyCancelled = false;

    const fetchImpl = async (url, init = {}) => {
      const method = (init.method || 'GET').toUpperCase();
      const rangeHeader = init.headers?.Range || init.headers?.range || null;
      const isSmall = url.endsWith('small.wgsl.pack');

      if (method === 'HEAD') {
        return {
          status: 200,
          ok: true,
          headers: makeMockHeaders({
            'Accept-Ranges': 'bytes',
            ETag: isSmall ? '"s1"' : '"nr1"',
            'Content-Length': String(isSmall ? smallPayload.byteLength : 1000000),
          }),
        };
      }

      if (isSmall) {
        if (!rangeHeader) {
          fullGetCalledForSmall = true;
        }
        const { stream } = createChunkReadableStream(splitIntoChunks(smallPayload, [1024, 2048, 3072]));
        return {
          status: 206,
          ok: true,
          headers: makeMockHeaders({
            'Accept-Ranges': 'bytes',
            ETag: '"s1"',
            'Content-Range': `bytes 0-${smallPayload.byteLength - 1}/${smallPayload.byteLength}`,
            'Content-Length': String(smallPayload.byteLength),
          }),
          body: stream,
          async arrayBuffer() {
            arrayBufferCalled += 1;
            throw new Error('must not call arrayBuffer() when body.getReader() is available');
          },
        };
      }

      // Misconfigured server returns 200 instead of 206 for a Range probe on a large file.
      const stream = new ReadableStream({
        pull(controller) {
          controller.enqueue(new Uint8Array(65536));
        },
        cancel() {
          non206BodyCancelled = true;
        },
      });
      return {
        status: 200,
        ok: true,
        headers: makeMockHeaders({
          'Content-Length': '1000000',
          ETag: '"nr1"',
        }),
        body: stream,
        async arrayBuffer() {
          arrayBufferCalled += 1;
          throw new Error('must not buffer 200 body on failed Range probe');
        },
      };
    };

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 5,
      probeBytes: 16384,
      verifyHashes: true,
      maxConcurrency: 2,
    });

    assert.equal(arrayBufferCalled, 0);
    assert.equal(fullGetCalledForSmall, false);
    assert.equal(non206BodyCancelled, true);
    assert.equal(report.ok, false);
    assert.equal(report.passed, 1);
    assert.equal(report.failed, 1);
    assert.equal(report.results[0].ok, true);
    assert.equal(report.results[0].hash.verified, true);
    assert.equal(report.results[0].hash.actual, smallSha);
    assert.equal(report.bytesHashed, smallPayload.byteLength);

    const stats = getStats();
    assert.equal(stats.filesVerified, 2);
    assert.equal(stats.bytesHashed, smallPayload.byteLength);
    assert.equal(stats.activeHashes, 0);
  });

  it('propagates AbortSignal across all active concurrent workers, cancels streams, and increments abortedCount', async () => {
    resetStats();
    const files = [];
    const payloads = new Map();
    for (let i = 0; i < 8; i += 1) {
      const p = `x64/pack_${i}.rpf`;
      const bytes = makeSyntheticPayload(4096, 0x20 + i);
      files.push({ path: p, size: bytes.byteLength, sha256: sha256Hex(bytes), etag: `"e${i}"` });
      payloads.set(p, bytes);
    }
    const manifest = makeValidManifest({ files });

    const controller = new AbortController();
    const seenSignals = [];
    const requestedFiles = new Set();
    const streamHandles = [];
    let hangingCount = 0;

    const fetchImpl = async (url, init = {}) => {
      seenSignals.push(init.signal);
      const method = (init.method || 'GET').toUpperCase();
      const rangeHeader = init.headers?.Range || init.headers?.range || null;
      const relPath = url.replace(/^\/data\//, '');
      requestedFiles.add(relPath);
      const bytes = payloads.get(relPath);
      const baseHeaders = { 'Accept-Ranges': 'bytes', ETag: `"e${relPath.match(/\d+/)[0]}"` };

      if (method === 'HEAD') {
        return {
          status: 200,
          ok: true,
          headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(bytes.byteLength) }),
        };
      }
      if (rangeHeader) {
        return {
          status: 206,
          ok: true,
          headers: makeMockHeaders({
            ...baseHeaders,
            'Content-Range': `bytes 0-0/${bytes.byteLength}`,
            'Content-Length': '1',
          }),
          body: createChunkReadableStream([bytes.subarray(0, 1)]).stream,
        };
      }

      // Full GET hangs after yielding the first 512 bytes until AbortSignal cancels all 3 workers.
      const handle = createChunkReadableStream([bytes.subarray(0, 512), bytes.subarray(512)], {
        hangAtIndex: 1,
        onHang: () => {
          hangingCount += 1;
          if (hangingCount === 3) {
            setTimeout(() => controller.abort(new Error('boot cancelled')), 5);
          }
        },
      });
      streamHandles.push(handle);
      return {
        status: 200,
        ok: true,
        headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(bytes.byteLength) }),
        body: handle.stream,
      };
    };

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 8,
      maxConcurrency: 3,
      verifyHashes: true,
      signal: controller.signal,
    });

    assert.equal(report.ok, false);
    assert.equal(report.aborted, true);
    assert.equal(report.abortedCount, 3);
    assert.equal(report.telemetry.aborted, true);
    assert.equal(report.telemetry.abortedCount, 3);
    assert.equal(report.probed, 0);
    assert.equal(streamHandles.length, 3);
    assert.ok(streamHandles.every((h) => h.cancelled === true));
    assert.ok(seenSignals.every((s) => s === controller.signal));
    // Remaining queued files (pack_3..pack_7) must never have started.
    assert.equal(requestedFiles.size, 3);
    assert.equal(report.bytesHashed, 512 * 3);

    const stats = getStats();
    assert.equal(stats.abortedCount, 3);
    assert.equal(stats.bytesHashed, 512 * 3);
    assert.equal(stats.peakConcurrentHashes, 3);
    assert.equal(stats.activeHashes, 0);

    // Pre-aborted signal increments abortedCount without launching workers or bumping peakConcurrentHashes.
    resetStats();
    const preAborted = new AbortController();
    preAborted.abort();
    const preReport = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      signal: preAborted.signal,
    });
    assert.equal(preReport.aborted, true);
    assert.equal(preReport.abortedCount, 1);
    assert.equal(getStats().abortedCount, 1);
    assert.equal(getStats().peakConcurrentHashes, 0);
    assert.equal(getStats().activeHashes, 0);
  });

  it('tracks sanitized mismatchCount, bytesHashed, filesVerified across APIs and resets cleanly via resetStats()', async () => {
    resetStats();
    assert.deepEqual(getStats(), {
      filesVerified: 0,
      bytesHashed: 0,
      mismatchCount: 0,
      abortedCount: 0,
      peakConcurrentHashes: 0,
      activeHashes: 0,
    });

    const goodBytes = makeSyntheticPayload(1024, 0xaa);
    const badHashBytes = makeSyntheticPayload(512, 0xbb);
    const sizeDriftBytes = makeSyntheticPayload(256, 0xcc);

    const manifest = makeValidManifest({
      files: [
        { path: 'good.bin', size: goodBytes.byteLength, sha256: sha256Hex(goodBytes), etag: '"g1"' },
        { path: 'bad_hash.bin', size: badHashBytes.byteLength, sha256: '0'.repeat(64), etag: '"bh1"' },
        { path: 'missing.bin', size: 128, etag: '"m1"' },
        { path: 'size_drift.bin', size: 500, etag: '"sd1"' },
      ],
    });

    const backing = new Map([
      ['good.bin', { bytes: goodBytes, etag: '"g1"' }],
      ['bad_hash.bin', { bytes: badHashBytes, etag: '"bh1"' }],
      ['size_drift.bin', { bytes: sizeDriftBytes, etag: '"sd1"' }],
    ]);
    const { fetchImpl } = createMockDataFetch(backing);

    const liveReport = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      verifyHashes: true,
      maxConcurrency: 4,
    });

    assert.equal(liveReport.ok, false);
    assert.equal(liveReport.filesVerified, 4);
    assert.equal(liveReport.mismatchCount, 3); // 1 hash_mismatch + 1 missing + 1 size_mismatch
    assert.equal(liveReport.bytesHashed, 1024 + 512);

    const afterLive = getStats();
    assert.equal(afterLive.filesVerified, 4);
    assert.equal(afterLive.bytesHashed, 1024 + 512);
    assert.equal(afterLive.mismatchCount, 3);
    assert.equal(afterLive.abortedCount, 0);
    assert.ok(afterLive.peakConcurrentHashes >= 1 && afterLive.peakConcurrentHashes <= 4);
    assert.equal(afterLive.activeHashes, 0);

    // Standalone verifyStream and createIncrementalVerifier also accumulate into getStats().
    const streamPayload = makeSyntheticPayload(2048, 0xdd);
    const { stream } = createChunkReadableStream(splitIntoChunks(streamPayload, [512]));
    const streamRes = await verifyStream(stream, {
      algorithm: 'sha256',
      expectedDigest: 'f'.repeat(64), // 1 mismatch
      expectedSize: 2048,
      path: 'C:\\Users\\Alice\\Private\\secret.pgim',
    });
    assert.equal(streamRes.ok, false);
    assert.equal(streamRes.mismatchCount, 1);

    const inc = createIncrementalVerifier({
      algorithm: 'crc32',
      expectedDigest: computeDigest(goodBytes, 'crc32'),
      expectedSize: goodBytes.byteLength,
    });
    inc.update(goodBytes);
    const incRes1 = inc.finalize();
    const incRes2 = inc.finalize(); // idempotent: must NOT double-count
    assert.strictEqual(incRes1, incRes2);

    const afterAll = getStats();
    assert.equal(afterAll.filesVerified, 4 + 1 + 1);
    assert.equal(afterAll.bytesHashed, 1024 + 512 + 2048 + 1024);
    assert.equal(afterAll.mismatchCount, 3 + 1);
    assert.equal(afterAll.abortedCount, 0);
    assert.equal(afterAll.activeHashes, 0);

    // Telemetry snapshot must be strictly sanitized numeric values (no paths or raw bytes).
    const serializedStats = JSON.stringify(afterAll);
    assert.equal(serializedStats.includes('Alice'), false);
    assert.equal(serializedStats.includes('secret.pgim'), false);
    for (const [key, value] of Object.entries(afterAll)) {
      assert.equal(typeof value, 'number', `getStats().${key} must be numeric`);
      assert.ok(Number.isSafeInteger(value) && value >= 0);
    }

    // resetStats() clears all counters cleanly.
    resetStats();
    assert.deepEqual(getStats(), {
      filesVerified: 0,
      bytesHashed: 0,
      mismatchCount: 0,
      abortedCount: 0,
      peakConcurrentHashes: 0,
      activeHashes: 0,
    });
  });
});

describe('warm-boot ETag+size fingerprint skip cache (W328)', () => {
  it('validates strong RFC 9110 ETags via isStrongEtag() and rejects weak, empty, unquoted, or control-char values', () => {
    assert.equal(isStrongEtag('"abc123"'), true);
    assert.equal(isStrongEtag(' "ffdc0-2000" '), true);
    assert.equal(isStrongEtag('"a"'), true);

    assert.equal(isStrongEtag('W/"abc123"'), false);
    assert.equal(isStrongEtag('w/"abc123"'), false);
    assert.equal(isStrongEtag('""'), false);
    assert.equal(isStrongEtag('abc123'), false);
    assert.equal(isStrongEtag(''), false);
    assert.equal(isStrongEtag('   '), false);
    assert.equal(isStrongEtag('"bad\nchar"'), false);
    assert.equal(isStrongEtag(null), false);
    assert.equal(isStrongEtag(undefined), false);
    assert.equal(isStrongEtag(12345), false);
  });

  it('skips GET body fetches and SHA-256 re-hashing on warm boot when HEAD strong ETag + Content-Length match verifiedFingerprints or previousSnapshot', async () => {
    resetStats();

    const rpfBytes = makeSyntheticPayload(16384, 0x31);
    const pgimBytes = makeSyntheticPayload(12288, 0x32);
    const packBytes = makeSyntheticPayload(8192, 0x33);
    const totalAssetBytes = rpfBytes.byteLength + pgimBytes.byteLength + packBytes.byteLength;

    const files = [
      { path: 'x64/levels.rpf', size: rpfBytes.byteLength, sha256: sha256Hex(rpfBytes), etag: '"rpf-v1"' },
      { path: 'guest/boot.pgim', size: pgimBytes.byteLength, sha256: sha256Hex(pgimBytes), etag: '"pgim-v1"' },
      { path: 'shaders/world.wgsl.pack', size: packBytes.byteLength, sha256: sha256Hex(packBytes), etag: '"pack-v1"' },
    ];
    const manifest = makeValidManifest({ files });
    const backing = new Map([
      ['x64/levels.rpf', { bytes: rpfBytes, etag: '"rpf-v1"' }],
      ['guest/boot.pgim', { bytes: pgimBytes, etag: '"pgim-v1"' }],
      ['shaders/world.wgsl.pack', { bytes: packBytes, etag: '"pack-v1"' }],
    ]);

    // 1. Cold boot: hashes all 3 files and emits a sanitized verifiedFingerprints snapshot.
    const coldMock = createMockDataFetch(backing);
    const coldReport = await verifyLiveEndpoints({
      fetchImpl: coldMock.fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      verifyHashes: true,
      checkCoopCoep: true,
    });

    assert.equal(coldReport.ok, true);
    assert.equal(coldReport.passed, 3);
    assert.equal(coldReport.fingerprintSkippedCount, 0);
    assert.equal(coldReport.bytesSkippedByFingerprint, 0);
    assert.equal(coldReport.telemetry.fingerprintSkippedCount, 0);
    assert.equal(coldReport.telemetry.bytesSkippedByFingerprint, 0);
    assert.equal(coldReport.bytesHashed, totalAssetBytes);
    assert.deepEqual(coldReport.verifiedFingerprints, {
      'x64/levels.rpf': { etag: '"rpf-v1"', size: rpfBytes.byteLength, sha256: sha256Hex(rpfBytes) },
      'guest/boot.pgim': { etag: '"pgim-v1"', size: pgimBytes.byteLength, sha256: sha256Hex(pgimBytes) },
      'shaders/world.wgsl.pack': { etag: '"pack-v1"', size: packBytes.byteLength, sha256: sha256Hex(packBytes) },
    });
    assert.ok(coldMock.calls.some((c) => c.method === 'GET'));

    // 2. Warm boot via verifiedFingerprints plain object: ONLY HEAD requests, ZERO GET requests!
    resetStats();
    const warmMock = createMockDataFetch(backing);
    const warmReport = await verifyLiveEndpoints({
      fetchImpl: warmMock.fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      verifyHashes: true,
      checkCoopCoep: true,
      verifiedFingerprints: coldReport.verifiedFingerprints,
    });

    assert.equal(warmReport.ok, true);
    assert.equal(warmReport.probed, 3);
    assert.equal(warmReport.passed, 3);
    assert.equal(warmReport.failed, 0);
    assert.equal(warmReport.filesVerified, 3);
    assert.equal(warmReport.bytesHashed, 0);
    assert.equal(warmReport.bytesVerified, 0);
    assert.equal(warmReport.peakConcurrentHashes, 0);
    assert.equal(warmReport.fingerprintSkippedCount, 3);
    assert.equal(warmReport.bytesSkippedByFingerprint, totalAssetBytes);
    assert.equal(warmReport.telemetry.fingerprintSkippedCount, 3);
    assert.equal(warmReport.telemetry.bytesSkippedByFingerprint, totalAssetBytes);
    assert.equal(warmReport.telemetry.bytesHashed, 0);

    // Strictly 3 HEAD requests and 0 GET requests (neither Range GET nor full body GET).
    assert.equal(warmMock.calls.length, 3);
    assert.ok(warmMock.calls.every((c) => c.method === 'HEAD'));

    for (const r of warmReport.results) {
      assert.equal(r.ok, true);
      assert.equal(r.fingerprintSkipped, true);
      assert.equal(r.bytesSkippedByFingerprint, r.expectedSize);
      assert.equal(r.range.skipped, true);
      assert.equal(r.range.status, null);
      assert.equal(r.hash.verified, true);
      assert.equal(r.hash.skippedByFingerprint, true);
      assert.equal(r.hash.checkedBytes, 0);
      assert.equal(r.hash.actual, r.hash.expected);
      assert.equal(r.coopCoep, true);
    }

    assert.deepEqual(getFingerprintCacheStats(), {
      fingerprintSkippedCount: 3,
      bytesSkippedByFingerprint: totalAssetBytes,
    });

    // 3. Warm boot via previousSnapshot (passing full coldReport directly) and via Map.
    const snapMock = createMockDataFetch(backing);
    const snapReport = await verifyLiveEndpoints({
      fetchImpl: snapMock.fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      verifyHashes: true,
      previousSnapshot: coldReport,
    });
    assert.equal(snapReport.ok, true);
    assert.equal(snapReport.fingerprintSkippedCount, 3);
    assert.equal(snapReport.bytesSkippedByFingerprint, totalAssetBytes);
    assert.equal(snapReport.bytesHashed, 0);
    assert.equal(snapMock.calls.length, 3);
    assert.ok(snapMock.calls.every((c) => c.method === 'HEAD'));

    const mapMock = createMockDataFetch(backing);
    const fpMap = new Map(Object.entries(coldReport.verifiedFingerprints));
    const mapReport = await verifyLiveEndpoints({
      fetchImpl: mapMock.fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      verifyHashes: true,
      verifiedFingerprints: fpMap,
    });
    assert.equal(mapReport.ok, true);
    assert.equal(mapReport.fingerprintSkippedCount, 3);
    assert.equal(mapReport.bytesSkippedByFingerprint, totalAssetBytes);
    assert.equal(mapMock.calls.length, 3);
    assert.ok(mapMock.calls.every((c) => c.method === 'HEAD'));

    // Cumulative fingerprint cache stats tracked across runs and reset cleanly by resetStats().
    assert.deepEqual(getFingerprintCacheStats(), {
      fingerprintSkippedCount: 9,
      bytesSkippedByFingerprint: totalAssetBytes * 3,
    });
    resetStats();
    assert.deepEqual(getFingerprintCacheStats(), {
      fingerprintSkippedCount: 0,
      bytesSkippedByFingerprint: 0,
    });
    // Existing W278 getStats() contract stays 100% intact.
    assert.deepEqual(getStats(), {
      filesVerified: 0,
      bytesHashed: 0,
      mismatchCount: 0,
      abortedCount: 0,
      peakConcurrentHashes: 0,
      activeHashes: 0,
    });
  });

  it('forces full streaming re-hash when ETag changes, updating snapshot on valid payload and catching corruption on bad payload', async () => {
    resetStats();
    const v1Bytes = makeSyntheticPayload(8192, 0x51);
    const v2Bytes = makeSyntheticPayload(8192, 0x52);
    const corruptBytes = v2Bytes.slice();
    corruptBytes[4096] ^= 0xff;

    const previousVerified = {
      'x64/world.rpf': { etag: '"world-v1"', size: v1Bytes.byteLength, sha256: sha256Hex(v1Bytes) },
      'shaders/fx.wgsl.pack': { etag: '"fx-v1"', size: v1Bytes.byteLength, sha256: sha256Hex(v1Bytes) },
    };

    const manifest = makeValidManifest({
      files: [
        { path: 'x64/world.rpf', size: v2Bytes.byteLength, sha256: sha256Hex(v2Bytes) },
        { path: 'shaders/fx.wgsl.pack', size: v2Bytes.byteLength, sha256: sha256Hex(v2Bytes) },
      ],
    });

    const backing = new Map([
      ['x64/world.rpf', { bytes: v2Bytes, etag: '"world-v2"' }],
      ['shaders/fx.wgsl.pack', { bytes: corruptBytes, etag: '"fx-v2"' }],
    ]);

    let arrayBufferCalled = 0;
    const methodsByPath = new Map();
    const fetchImpl = async (url, init = {}) => {
      const method = (init.method || 'GET').toUpperCase();
      const rangeHeader = init.headers?.Range || init.headers?.range || null;
      const relPath = url.replace(/^\/data\//, '');
      const list = methodsByPath.get(relPath) || [];
      list.push(rangeHeader ? `${method}:range` : method);
      methodsByPath.set(relPath, list);

      const entry = backing.get(relPath);
      const baseHeaders = { 'Accept-Ranges': 'bytes', ETag: entry.etag };
      if (method === 'HEAD') {
        return {
          status: 200,
          ok: true,
          headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(entry.bytes.byteLength) }),
        };
      }
      if (rangeHeader) {
        return {
          status: 206,
          ok: true,
          headers: makeMockHeaders({
            ...baseHeaders,
            'Content-Range': `bytes 0-0/${entry.bytes.byteLength}`,
            'Content-Length': '1',
          }),
          body: createChunkReadableStream([entry.bytes.subarray(0, 1)]).stream,
        };
      }
      return {
        status: 200,
        ok: true,
        headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(entry.bytes.byteLength) }),
        body: createChunkReadableStream(splitIntoChunks(entry.bytes, [2048])).stream,
        async arrayBuffer() {
          arrayBufferCalled += 1;
          throw new Error('must stream via body.getReader() on ETag fallback');
        },
      };
    };

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 5,
      verifyHashes: true,
      verifiedFingerprints: previousVerified,
    });

    assert.equal(arrayBufferCalled, 0);
    assert.equal(report.ok, false);
    assert.equal(report.passed, 1);
    assert.equal(report.failed, 1);
    assert.equal(report.fingerprintSkippedCount, 0);
    assert.equal(report.bytesSkippedByFingerprint, 0);
    assert.equal(report.bytesHashed, v2Bytes.byteLength * 2);
    assert.deepEqual(methodsByPath.get('x64/world.rpf'), ['HEAD', 'GET:range', 'GET']);
    assert.deepEqual(methodsByPath.get('shaders/fx.wgsl.pack'), ['HEAD', 'GET:range', 'GET']);

    // Valid updated file replaces old snapshot entry; failed file is excluded from verifiedFingerprints.
    assert.equal(report.results[0].ok, true);
    assert.equal(report.results[0].fingerprintSkipped, false);
    assert.deepEqual(report.verifiedFingerprints, {
      'x64/world.rpf': { etag: '"world-v2"', size: v2Bytes.byteLength, sha256: sha256Hex(v2Bytes) },
    });
    assert.equal(report.results[1].ok, false);
    assert.equal(report.results[1].hash.verified, false);
    assert.deepEqual(report.results[1].mismatchCategories, ['hash_mismatch']);
  });

  it('falls back to full streaming verification on weak ETag, missing ETag, missing Content-Length, size drift, or unhashed snapshot entry', async () => {
    resetStats();
    const payload = makeSyntheticPayload(4096, 0x61);
    const sha = sha256Hex(payload);
    const wrongSizePayload = makeSyntheticPayload(2048, 0x62);

    const manifest = makeValidManifest({
      files: [
        { path: 'weak.rpf', size: payload.byteLength, sha256: sha },
        { path: 'missing_etag.rpf', size: payload.byteLength, sha256: sha },
        { path: 'no_head_len.pgim', size: payload.byteLength, sha256: sha },
        { path: 'size_diff.pack', size: payload.byteLength, sha256: sha },
        { path: 'no_cached_sha.rpf', size: payload.byteLength, sha256: sha },
        { path: 'diff_cached_sha.rpf', size: payload.byteLength, sha256: sha },
      ],
    });

    const verifiedFingerprints = {
      'weak.rpf': { etag: 'W/"weak-1"', size: payload.byteLength, sha256: sha },
      'missing_etag.rpf': { etag: '"was-strong"', size: payload.byteLength, sha256: sha },
      'no_head_len.pgim': { etag: '"no-len-1"', size: payload.byteLength, sha256: sha },
      'size_diff.pack': { etag: '"size-1"', size: payload.byteLength, sha256: sha },
      'no_cached_sha.rpf': { etag: '"nosha-1"', size: payload.byteLength, sha256: null },
      'diff_cached_sha.rpf': { etag: '"diffsha-1"', size: payload.byteLength, sha256: '0'.repeat(64) },
    };

    const getCounts = new Map();
    const fetchImpl = async (url, init = {}) => {
      const method = (init.method || 'GET').toUpperCase();
      const rangeHeader = init.headers?.Range || init.headers?.range || null;
      const relPath = url.replace(/^\/data\//, '');
      if (method === 'GET' && !rangeHeader) {
        getCounts.set(relPath, (getCounts.get(relPath) || 0) + 1);
      }

      const bytes = relPath === 'size_diff.pack' ? wrongSizePayload : payload;
      const etagByPath = {
        'weak.rpf': 'W/"weak-1"',
        'missing_etag.rpf': null,
        'no_head_len.pgim': '"no-len-1"',
        'size_diff.pack': '"size-1"',
        'no_cached_sha.rpf': '"nosha-1"',
        'diff_cached_sha.rpf': '"diffsha-1"',
      };
      const etag = etagByPath[relPath];
      const baseHeaders = { 'Accept-Ranges': 'bytes' };
      if (etag) baseHeaders.ETag = etag;

      if (method === 'HEAD') {
        const headers = { ...baseHeaders };
        if (relPath !== 'no_head_len.pgim') {
          headers['Content-Length'] = String(bytes.byteLength);
        }
        return { status: 200, ok: true, headers: makeMockHeaders(headers) };
      }

      if (rangeHeader) {
        return {
          status: 206,
          ok: true,
          headers: makeMockHeaders({
            ...baseHeaders,
            'Content-Range': `bytes 0-0/${bytes.byteLength}`,
            'Content-Length': '1',
          }),
          body: createChunkReadableStream([bytes.subarray(0, 1)]).stream,
        };
      }

      return {
        status: 200,
        ok: true,
        headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(bytes.byteLength) }),
        body: createChunkReadableStream(splitIntoChunks(bytes, [1024])).stream,
      };
    };

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      verifyHashes: true,
      verifiedFingerprints,
    });

    assert.equal(report.fingerprintSkippedCount, 0);
    assert.equal(report.bytesSkippedByFingerprint, 0);
    assert.equal(report.telemetry.fingerprintSkippedCount, 0);
    assert.equal(report.telemetry.bytesSkippedByFingerprint, 0);

    // Every single entry must have fallen back to a full GET hash verification.
    for (const f of manifest.files) {
      assert.equal(getCounts.get(f.path), 1, `expected full GET fallback for ${f.path}`);
    }
    assert.equal(report.bytesHashed, payload.byteLength * 5 + wrongSizePayload.byteLength);

    // Weak ETag must never be written to verifiedFingerprints; valid strong ETag fallbacks are emitted.
    assert.equal('weak.rpf' in report.verifiedFingerprints, false);
    assert.equal('missing_etag.rpf' in report.verifiedFingerprints, false);
    assert.equal('size_diff.pack' in report.verifiedFingerprints, false);
    assert.equal(report.verifiedFingerprints['no_head_len.pgim'].sha256, sha);
    assert.equal(report.verifiedFingerprints['no_cached_sha.rpf'].sha256, sha);
    assert.equal(report.verifiedFingerprints['diff_cached_sha.rpf'].sha256, sha);
  });

  it('bypasses matching fingerprint cache when forceReverify: true is set', async () => {
    resetStats();
    const bytesA = makeSyntheticPayload(6144, 0x71);
    const bytesB = makeSyntheticPayload(4096, 0x72);
    const manifest = makeValidManifest({
      files: [
        { path: 'x64/a.rpf', size: bytesA.byteLength, sha256: sha256Hex(bytesA), etag: '"a1"' },
        { path: 'guest/b.pgim', size: bytesB.byteLength, sha256: sha256Hex(bytesB), etag: '"b1"' },
      ],
    });
    const backing = new Map([
      ['x64/a.rpf', { bytes: bytesA, etag: '"a1"' }],
      ['guest/b.pgim', { bytes: bytesB, etag: '"b1"' }],
    ]);

    const snapshot = {
      'x64/a.rpf': { etag: '"a1"', size: bytesA.byteLength, sha256: sha256Hex(bytesA) },
      'guest/b.pgim': { etag: '"b1"', size: bytesB.byteLength, sha256: sha256Hex(bytesB) },
    };

    const { fetchImpl, calls } = createMockDataFetch(backing);
    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      verifyHashes: true,
      verifiedFingerprints: snapshot,
      previousSnapshot: snapshot,
      forceReverify: true,
    });

    assert.equal(report.ok, true);
    assert.equal(report.passed, 2);
    assert.equal(report.fingerprintSkippedCount, 0);
    assert.equal(report.bytesSkippedByFingerprint, 0);
    assert.equal(report.telemetry.fingerprintSkippedCount, 0);
    assert.equal(report.telemetry.bytesSkippedByFingerprint, 0);
    assert.equal(report.bytesHashed, bytesA.byteLength + bytesB.byteLength);
    // 2 HEAD + 2 Range GET + 2 Full GET = 6 requests
    assert.equal(calls.length, 6);
    assert.equal(calls.filter((c) => c.method === 'GET' && !c.range).length, 2);
    assert.ok(report.results.every((r) => r.fingerprintSkipped === false));
    assert.deepEqual(getFingerprintCacheStats(), {
      fingerprintSkippedCount: 0,
      bytesSkippedByFingerprint: 0,
    });
  });

  it('preserves W278 bounded concurrency, manifest order, AbortSignal handling, and sanitized telemetry on mixed warm-boot batches', async () => {
    resetStats();
    const entries = [
      { path: 'x64/skip_0.rpf', size: 10240, etag: '"skip-0"', mode: 'skip' },
      { path: 'guest/changed_1.pgim', size: 6144, etag: '"chg-v2"', cachedEtag: '"chg-v1"', mode: 'fallback' },
      { path: 'shaders/skip_2.wgsl.pack', size: 8192, etag: '"skip-2"', mode: 'skip' },
      { path: 'x64/weak_3.rpf', size: 4096, etag: 'W/"weak-3"', mode: 'fallback' },
      { path: 'x64/skip_4.rpf', size: 12288, etag: '"skip-4"', mode: 'skip' },
      { path: 'shaders/new_5.wgsl.pack', size: 5120, etag: '"new-5"', mode: 'fallback' },
    ];

    const files = [];
    const payloads = new Map();
    const verifiedFingerprints = {
      // Malicious / unsafe cache keys must be silently ignored and never leaked.
      '../escape.rpf': { etag: '"bad"', size: 10, sha256: 'a'.repeat(64) },
      'C:/Users/Alice/secret.rpf': { etag: '"bad2"', size: 10, sha256: 'b'.repeat(64) },
    };

    let expectedSkippedBytes = 0;
    let expectedHashedBytes = 0;
    for (let i = 0; i < entries.length; i += 1) {
      const e = entries[i];
      const bytes = makeSyntheticPayload(e.size, 0x80 + i);
      const sha = sha256Hex(bytes);
      files.push({ path: e.path, size: e.size, sha256: sha });
      payloads.set(e.path, { bytes, etag: e.etag });
      if (e.mode === 'skip') {
        expectedSkippedBytes += e.size;
        verifiedFingerprints[e.path] = { etag: e.etag, size: e.size, sha256: sha };
      } else {
        expectedHashedBytes += e.size;
        if (e.cachedEtag) {
          verifiedFingerprints[e.path] = { etag: e.cachedEtag, size: e.size, sha256: sha };
        } else if (e.etag.startsWith('W/')) {
          verifiedFingerprints[e.path] = { etag: e.etag, size: e.size, sha256: sha };
        }
      }
    }

    const manifest = makeValidManifest({ files });
    let inFlightStreams = 0;
    let peakInFlightStreams = 0;
    const getRequestedPaths = new Set();

    const fetchImpl = async (url, init = {}) => {
      const method = (init.method || 'GET').toUpperCase();
      const rangeHeader = init.headers?.Range || init.headers?.range || null;
      const relPath = url.replace(/^\/data\//, '');
      const item = payloads.get(relPath);
      const baseHeaders = { 'Accept-Ranges': 'bytes', ETag: item.etag };

      if (method === 'HEAD') {
        return {
          status: 200,
          ok: true,
          headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(item.bytes.byteLength) }),
        };
      }

      getRequestedPaths.add(relPath);
      if (rangeHeader) {
        return {
          status: 206,
          ok: true,
          headers: makeMockHeaders({
            ...baseHeaders,
            'Content-Range': `bytes 0-0/${item.bytes.byteLength}`,
            'Content-Length': '1',
          }),
          body: createChunkReadableStream([item.bytes.subarray(0, 1)]).stream,
        };
      }

      const chunks = splitIntoChunks(item.bytes, [1024]);
      let idx = 0;
      let active = false;
      const stream = new ReadableStream({
        async pull(controller) {
          if (!active) {
            active = true;
            inFlightStreams += 1;
            if (inFlightStreams > peakInFlightStreams) peakInFlightStreams = inFlightStreams;
          }
          await new Promise((r) => setTimeout(r, 4));
          if (idx >= chunks.length) {
            if (active) {
              active = false;
              inFlightStreams -= 1;
            }
            controller.close();
            return;
          }
          controller.enqueue(chunks[idx]);
          idx += 1;
        },
        cancel() {
          if (active) {
            active = false;
            inFlightStreams -= 1;
          }
        },
      });

      return {
        status: 200,
        ok: true,
        headers: makeMockHeaders({ ...baseHeaders, 'Content-Length': String(item.bytes.byteLength) }),
        body: stream,
        async arrayBuffer() {
          throw new Error('must stream fallback files via body.getReader()');
        },
      };
    };

    const report = await verifyLiveEndpoints({
      fetchImpl,
      dataBase: '/data/',
      manifest,
      sampleCount: 10,
      maxConcurrency: 2,
      verifyHashes: true,
      verifiedFingerprints,
    });

    assert.equal(report.ok, true);
    assert.equal(report.passed, 6);
    assert.equal(report.maxConcurrency, 2);
    assert.equal(report.fingerprintSkippedCount, 3);
    assert.equal(report.bytesSkippedByFingerprint, expectedSkippedBytes);
    assert.equal(report.bytesHashed, expectedHashedBytes);
    assert.ok(peakInFlightStreams >= 1 && peakInFlightStreams <= 2);
    assert.ok(report.peakConcurrentHashes >= 1 && report.peakConcurrentHashes <= 2);

    // Manifest order preserved.
    assert.deepEqual(
      report.results.map((r) => r.path),
      entries.map((e) => e.path),
    );

    // Skipped entries never issued any GET request.
    assert.equal(getRequestedPaths.has('x64/skip_0.rpf'), false);
    assert.equal(getRequestedPaths.has('shaders/skip_2.wgsl.pack'), false);
    assert.equal(getRequestedPaths.has('x64/skip_4.rpf'), false);
    assert.equal(getRequestedPaths.has('guest/changed_1.pgim'), true);
    assert.equal(getRequestedPaths.has('x64/weak_3.rpf'), true);
    assert.equal(getRequestedPaths.has('shaders/new_5.wgsl.pack'), true);

    // Sanitized output must never leak unsafe cache keys or host paths.
    const serialized = JSON.stringify(report);
    assert.equal(serialized.includes('Alice'), false);
    assert.equal(serialized.includes('../escape.rpf'), false);
    assert.equal('x64/weak_3.rpf' in report.verifiedFingerprints, false);
    assert.equal(Object.keys(report.verifiedFingerprints).length, 5);

    // AbortSignal during HEAD does not count aborted entry as fingerprint-skipped.
    resetStats();
    const controller = new AbortController();
    const abortFetch = async (url, init = {}) => {
      if (init.method === 'HEAD') {
        setTimeout(() => controller.abort(new Error('cancelled during HEAD')), 5);
        return new Promise(() => {});
      }
      return fetchImpl(url, init);
    };
    const abortReport = await verifyLiveEndpoints({
      fetchImpl: abortFetch,
      dataBase: '/data/',
      manifest,
      sampleCount: 6,
      maxConcurrency: 2,
      verifyHashes: true,
      verifiedFingerprints,
      signal: controller.signal,
    });
    assert.equal(abortReport.ok, false);
    assert.equal(abortReport.aborted, true);
    assert.equal(abortReport.abortedCount, 2);
    assert.equal(abortReport.fingerprintSkippedCount, 0);
    assert.equal(abortReport.bytesSkippedByFingerprint, 0);
    assert.deepEqual(getFingerprintCacheStats(), {
      fingerprintSkippedCount: 0,
      bytesSkippedByFingerprint: 0,
    });
  });
});
