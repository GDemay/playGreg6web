import { describe, it, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  DEFAULT_SAMPLE_COUNT,
  INCOMPLETE_DOWNLOAD_SUFFIXES,
  MANIFEST_ABI_VERSION,
  computeStreamingReadiness,
  hasIncompleteSuffix,
  parseContentRange,
  validateBootset,
  validateDataManifest,
  validateManifestPath,
  verifyLiveEndpoints,
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
  });
});
