import test from 'node:test';
import assert from 'node:assert/strict';

import {
  WGPK_MAGIC,
  WGPK_VERSION,
  WGPK_HEADER_BYTES,
  WGPK_ENTRY_BYTES,
  DEFAULT_INITIAL_RANGE_BYTES,
  DEFAULT_MAX_CACHED_WGSL_BYTES,
  DEFAULT_MAX_CONCURRENT_RANGE_FETCHES,
  DEFAULT_MAX_RANGE_FETCH_RETRIES,
  DEFAULT_RANGE_RETRY_DELAY_MS,
  DEFAULT_PREFETCH_MAX_GAP_BYTES,
  DEFAULT_PREFETCH_MAX_SPAN_BYTES,
  SHADER_STAGE,
  STAGE_NAMES,
  fnv1a64,
  normalizeHash,
  formatHashHex,
  buildWgslPackBuffer,
  parseWgslPack,
  parseWgslPackIndex,
  loadFromBuffer,
  loadIndexFromUrl,
  loadFromUrl,
  WgslPack,
  WgslPackCache,
  WgslPackLoader,
} from '../web/workers/wgsl_pack_loader.js';

const SAMPLE_VS_WGSL =
  '@vertex fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(pos, 1.0); }';
const SAMPLE_FS_WGSL =
  '@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.25, 0.5, 1.0); }';
const SAMPLE_CS_WGSL = '@compute @workgroup_size(1) fn main() {}';

function createMockGpuDevice(options = {}) {
  const calls = [];
  let nextHandle = 1;
  let activeCompilations = 0;
  let peakCompilations = 0;
  const compileDelayMs = options.compileDelayMs || 0;
  const withCompilationInfo = Boolean(options.withCompilationInfo);

  const device = {
    createShaderModule(desc) {
      const mod = {
        kind: 'GPUShaderModule',
        handle: nextHandle++,
        label: desc.label,
        code: desc.code,
      };
      if (withCompilationInfo) {
        mod.getCompilationInfo = async () => {
          activeCompilations += 1;
          if (activeCompilations > peakCompilations) {
            peakCompilations = activeCompilations;
          }
          await new Promise((resolve) => setTimeout(resolve, compileDelayMs));
          activeCompilations -= 1;
          return { messages: [] };
        };
      }
      calls.push({ op: 'createShaderModule', desc: { ...desc }, result: mod });
      return mod;
    },
  };

  return {
    device,
    calls,
    getPeakCompilations: () => peakCompilations,
  };
}

test('constants match shaders/src/pack.rs and shaders/src/convert.rs', () => {
  assert.equal(WGPK_MAGIC, 'WGPK');
  assert.equal(WGPK_VERSION, 1);
  assert.equal(WGPK_HEADER_BYTES, 16);
  assert.equal(WGPK_ENTRY_BYTES, 20);
  assert.deepEqual(SHADER_STAGE, {
    VERTEX: 1,
    FRAGMENT: 2,
    COMPUTE: 3,
  });
  assert.deepEqual(STAGE_NAMES, {
    1: 'vertex',
    2: 'fragment',
    3: 'compute',
  });
});

test('fnv1a64 and formatHashHex match exact test vectors from shaders/src/convert.rs', () => {
  const enc = new TextEncoder();

  // Exact vectors from shaders/src/convert.rs::tests::fnv1a_64_matches_standard_test_vectors
  assert.equal(fnv1a64(new Uint8Array(0)), 0xcbf29ce484222325n);
  assert.equal(fnv1a64(enc.encode('a')), 0xaf63dc4c8601ec8cn);
  assert.equal(fnv1a64(enc.encode('foobar')), 0x85944171f73967e8n);

  // String, ArrayBuffer, and sliced Uint8Array inputs produce identical hashes
  assert.equal(fnv1a64(''), 0xcbf29ce484222325n);
  assert.equal(fnv1a64('a'), 0xaf63dc4c8601ec8cn);
  assert.equal(fnv1a64('foobar'), 0x85944171f73967e8n);

  const padded = new Uint8Array([0xaa, 0xbb, ...enc.encode('foobar'), 0xcc]);
  const slice = padded.subarray(2, 8);
  assert.equal(fnv1a64(slice), 0x85944171f73967e8n);
  assert.equal(fnv1a64(enc.encode('foobar').buffer), 0x85944171f73967e8n);

  // formatHashHex canonical 0x + 16 lowercase hex digits
  assert.equal(formatHashHex(0xdeadbeefn), '0x00000000deadbeef');
  assert.equal(formatHashHex(0xdeadbeef), '0x00000000deadbeef');
  assert.equal(formatHashHex('0xDEADBEEF'), '0x00000000deadbeef');
  assert.equal(formatHashHex('deadbeef'), '0x00000000deadbeef');
  assert.equal(formatHashHex(0n), '0x0000000000000000');
  assert.equal(formatHashHex(0xffffffffffffffffn), '0xffffffffffffffff');

  assert.equal(normalizeHash('0x00000000deadbeef'), 0xdeadbeefn);
  assert.throws(() => normalizeHash(-1n), RangeError);
  assert.throws(() => normalizeHash(1n << 64n), RangeError);
  assert.throws(() => normalizeHash('0xnothex'), RangeError);
  assert.throws(() => normalizeHash('0x1234567890abcdef0'), RangeError);
});

test('binary fixture matches shaders/src/pack.rs byte-for-byte', () => {
  // Exact byte layout produced by shaders/src/pack.rs::write_pack for:
  // hash = 0x1122_3344_5566_7788, stage = Compute (3), flags = 0,
  // wgsl = "@compute @workgroup_size(1) fn main() {}" (40 ASCII bytes)
  const payloadBytes = new TextEncoder().encode(SAMPLE_CS_WGSL);
  assert.equal(payloadBytes.byteLength, 40);

  const expectedBytes = new Uint8Array([
    // Header (16 bytes): "WGPK", version=1 LE, entry_count=1 LE, flags=0 LE
    0x57, 0x47, 0x50, 0x4b,
    0x01, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00,
    // Entry 0 (20 bytes): hash=0x1122334455667788 LE, stage=3, reserved=[0,0,0], offset=36, length=40
    0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
    0x03, 0x00, 0x00, 0x00,
    0x24, 0x00, 0x00, 0x00,
    0x28, 0x00, 0x00, 0x00,
    // Payload (40 bytes)
    ...payloadBytes,
  ]);
  assert.equal(expectedBytes.byteLength, 76);

  const built = buildWgslPackBuffer([
    {
      hash: 0x1122334455667788n,
      stage: SHADER_STAGE.COMPUTE,
      wgsl: SAMPLE_CS_WGSL,
    },
  ]);
  assert.deepEqual(new Uint8Array(built), expectedBytes);

  const pack = parseWgslPack(expectedBytes);
  assert.ok(pack instanceof WgslPack);
  assert.equal(pack.version, 1);
  assert.equal(pack.flags, 0);
  assert.equal(pack.size, 1);
  assert.ok(pack.entries instanceof Map);
  assert.equal(pack.has(0x1122334455667788n), true);
  assert.equal(pack.has('0x1122334455667788'), true);
  assert.equal(pack.has('1122334455667788'), true);
  assert.equal(pack.has(0x9999n), false);

  const entry = pack.getByHash(0x1122334455667788n);
  assert.deepEqual(entry, {
    hash: 0x1122334455667788n,
    hashHex: '0x1122334455667788',
    stage: 'compute',
    stageCode: 3,
    offset: 36,
    length: 40,
    wgsl: SAMPLE_CS_WGSL,
  });
  assert.equal(pack.getSource('0x1122334455667788'), SAMPLE_CS_WGSL);
  assert.equal(pack.getSource(0xdeadbeefn), undefined);
});

test('parseWgslPack handles multiple stages, UTF-8 multibyte text, non-zero byteOffset, and lazyDecode vs eagerDecode', () => {
  const utf8Fragment = `// 日本語コメント & rocket 🚀\n${SAMPLE_FS_WGSL}`;
  const vsHash = fnv1a64('synthetic-vs-spv');
  const fsHash = fnv1a64('synthetic-fs-spv');
  const csHash = fnv1a64('synthetic-cs-spv');

  const packBuf = buildWgslPackBuffer(
    [
      { hash: vsHash, stage: 'vertex', wgsl: SAMPLE_VS_WGSL },
      { hash: fsHash, stage: SHADER_STAGE.FRAGMENT, wgsl: utf8Fragment },
      { hash: csHash, stage: 'compute', wgsl: SAMPLE_CS_WGSL },
    ],
    { flags: 0x0000_0005 }
  );

  // Set non-zero reserved bytes on entry 0 to confirm forward compatibility (ignored on read).
  const packBytes = new Uint8Array(packBuf);
  packBytes[WGPK_HEADER_BYTES + 9] = 0xab;
  packBytes[WGPK_HEADER_BYTES + 10] = 0xcd;
  packBytes[WGPK_HEADER_BYTES + 11] = 0xef;

  // Embed inside a larger buffer at a non-zero offset to test view slicing.
  const outer = new Uint8Array(32 + packBytes.byteLength + 16);
  outer.set(packBytes, 32);
  const subView = outer.subarray(32, 32 + packBytes.byteLength);

  // 1. Eager decode (default)
  const eagerPack = parseWgslPack(subView);
  assert.equal(eagerPack.flags, 5);
  assert.equal(eagerPack.lazyDecode, false);
  const eagerStats = eagerPack.stats();
  assert.equal(eagerStats.entryCount, 3);
  assert.equal(eagerStats.decodedCount, 3);
  assert.deepEqual(eagerStats.stageCounts, { vertex: 1, fragment: 1, compute: 1 });
  assert.equal(eagerPack.getSource(vsHash), SAMPLE_VS_WGSL);
  assert.equal(eagerPack.getSource(fsHash), utf8Fragment);
  assert.equal(eagerPack.getSource(csHash), SAMPLE_CS_WGSL);

  // Verify UTF-8 byte length (multibyte chars > JS string length)
  const fsEntry = eagerPack.getByHash(fsHash);
  assert.equal(fsEntry.length, new TextEncoder().encode(utf8Fragment).byteLength);
  assert.ok(fsEntry.length > utf8Fragment.length);

  // 2. Lazy decode
  const lazyPack = parseWgslPack(subView, { lazyDecode: true });
  assert.equal(lazyPack.lazyDecode, true);
  assert.equal(lazyPack.stats().decodedCount, 0);

  // Listing entries and checking existence does NOT decode strings
  const listed = lazyPack.listEntries();
  assert.equal(listed.length, 3);
  assert.equal(listed[0].stage, 'vertex');
  assert.equal(listed[1].stage, 'fragment');
  assert.equal(listed[2].stage, 'compute');
  assert.equal(lazyPack.has(fsHash), true);
  assert.equal(lazyPack.stats().decodedCount, 0);

  // Reading one shader decodes only that shader, and caches the string on subsequent reads
  assert.equal(lazyPack.getSource(fsHash), utf8Fragment);
  assert.equal(lazyPack.stats().decodedCount, 1);
  assert.equal(lazyPack.getByHash(fsHash).wgsl, utf8Fragment);
  assert.equal(lazyPack.stats().decodedCount, 1);

  // Reading a second shader increments decodedCount to 2
  assert.equal(lazyPack.getSource(csHash), SAMPLE_CS_WGSL);
  assert.equal(lazyPack.stats().decodedCount, 2);
});

test('parseWgslPack supports empty packs (0 entries), empty payloads, and SharedArrayBuffer', () => {
  const emptyBuf = buildWgslPackBuffer([]);
  assert.equal(emptyBuf.byteLength, WGPK_HEADER_BYTES);
  const emptyPack = parseWgslPack(emptyBuf);
  assert.equal(emptyPack.size, 0);
  assert.deepEqual(emptyPack.listEntries(), []);
  assert.equal(emptyPack.stats().payloadBytes, 0);

  const zeroPayloadBuf = buildWgslPackBuffer([
    { hash: 1n, stage: SHADER_STAGE.VERTEX, wgsl: '' },
    { hash: 2n, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  const zeroPack = parseWgslPack(zeroPayloadBuf);
  assert.equal(zeroPack.getSource(1n), '');
  assert.equal(zeroPack.getSource(2n), SAMPLE_FS_WGSL);

  if (typeof SharedArrayBuffer !== 'undefined') {
    const src = new Uint8Array(zeroPayloadBuf);
    const sab = new SharedArrayBuffer(src.byteLength);
    new Uint8Array(sab).set(src);
    const sabPack = parseWgslPack(sab, { lazyDecode: true });
    assert.equal(sabPack.getSource(2n), SAMPLE_FS_WGSL);
  }
});

test('parseWgslPack rejects malformed, truncated, out-of-bounds, overlapping, and invalid UTF-8 inputs', () => {
  const validBuf = buildWgslPackBuffer([
    { hash: 0x1122334455667788n, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
    { hash: 0x8877665544332211n, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
  ]);
  const validBytes = new Uint8Array(validBuf);

  // Non-buffer input
  assert.throws(() => parseWgslPack(null), TypeError);
  assert.throws(() => parseWgslPack('WGPK'), TypeError);

  // Truncated header
  assert.throws(() => parseWgslPack(validBytes.subarray(0, 8)), /truncated/i);
  assert.throws(() => parseWgslPack(new Uint8Array(0)), /truncated/i);

  // Bad magic
  const badMagic = validBytes.slice();
  badMagic.set(new TextEncoder().encode('NOPE'), 0);
  assert.throws(() => parseWgslPack(badMagic), /invalid wgpk pack magic/i);

  // Unsupported version
  const badVersion = validBytes.slice();
  new DataView(badVersion.buffer).setUint32(4, 99, true);
  assert.throws(() => parseWgslPack(badVersion), /unsupported wgpk pack version 99/i);

  // Truncated entry table (claims 2 entries, only header + 1 entry bytes exist)
  const truncatedTable = validBytes.slice(0, WGPK_HEADER_BYTES + WGPK_ENTRY_BYTES);
  assert.throws(() => parseWgslPack(truncatedTable), /truncated/i);

  // Invalid stage code (0 and 9)
  const badStage0 = validBytes.slice();
  badStage0[WGPK_HEADER_BYTES + 8] = 0;
  assert.throws(() => parseWgslPack(badStage0), /invalid wgpk shader stage value 0/i);

  const badStage9 = validBytes.slice();
  badStage9[WGPK_HEADER_BYTES + 8] = 9;
  assert.throws(() => parseWgslPack(badStage9), /invalid wgpk shader stage value 9/i);

  // Offset pointing inside header/table (< dataStart)
  const headerOverlap = validBytes.slice();
  new DataView(headerOverlap.buffer).setUint32(WGPK_HEADER_BYTES + 12, 4, true);
  assert.throws(() => parseWgslPack(headerOverlap), /out of bounds/i);

  // Out-of-bounds offset
  const oobOffset = validBytes.slice();
  new DataView(oobOffset.buffer).setUint32(WGPK_HEADER_BYTES + 12, 99999, true);
  assert.throws(() => parseWgslPack(oobOffset), /out of bounds/i);

  // Out-of-bounds length (extends past buffer end)
  const oobLength = validBytes.slice();
  new DataView(oobLength.buffer).setUint32(WGPK_HEADER_BYTES + 16, 99999, true);
  assert.throws(() => parseWgslPack(oobLength), /out of bounds/i);

  // uint32 overflow on offset + length
  const overflowRange = validBytes.slice();
  const overflowView = new DataView(overflowRange.buffer);
  overflowView.setUint32(WGPK_HEADER_BYTES + 12, 0xffffffff, true);
  overflowView.setUint32(WGPK_HEADER_BYTES + 16, 16, true);
  assert.throws(() => parseWgslPack(overflowRange), /out of bounds/i);

  // Overlapping payload ranges between entry 0 and entry 1
  const overlapEntries = validBytes.slice();
  const overlapView = new DataView(overlapEntries.buffer);
  const entry0Offset = overlapView.getUint32(WGPK_HEADER_BYTES + 12, true);
  // Point entry 1 into the middle of entry 0's payload
  overlapView.setUint32(WGPK_HEADER_BYTES + WGPK_ENTRY_BYTES + 12, entry0Offset + 8, true);
  overlapView.setUint32(WGPK_HEADER_BYTES + WGPK_ENTRY_BYTES + 16, 16, true);
  assert.throws(() => parseWgslPack(overlapEntries), /overlapping payload/i);

  // Invalid UTF-8 bytes inside payload
  const badUtf8 = validBytes.slice();
  const dataStart = WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES;
  badUtf8[dataStart] = 0xff;
  assert.throws(() => parseWgslPack(badUtf8), /not valid utf-8/i);

  // In lazyDecode mode, parsing succeeds and accessing the corrupted entry throws
  const lazyBadUtf8 = parseWgslPack(badUtf8, { lazyDecode: true });
  assert.equal(lazyBadUtf8.getSource(0x8877665544332211n), SAMPLE_VS_WGSL);
  assert.throws(() => lazyBadUtf8.getSource(0x1122334455667788n), /not valid utf-8/i);
});

test('WgslPackCache deduplicates GPUShaderModule compilation, tracks stats, and isolates devices', () => {
  const hVs = fnv1a64('shader-vs-1');
  const hFs = fnv1a64('shader-fs-1');
  const hCs = fnv1a64('shader-cs-1');

  const pack1 = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  const pack2 = buildWgslPackBuffer([
    { hash: hCs, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
  ]);

  const cache = new WgslPackCache({ lazyDecode: true });
  cache.loadPackBuffer(pack1, 'base.wgsl.pack');
  cache.loadPackBuffer(pack2, 'compute.wgsl.pack');

  assert.equal(cache.hasShader(hVs), true);
  assert.equal(cache.hasShader(formatHashHex(hCs)), true);
  assert.equal(cache.getShaderSource(hFs), SAMPLE_FS_WGSL);
  assert.equal(cache.getShaderSource(0x1234n), undefined);

  const mock1 = createMockGpuDevice();
  const mock2 = createMockGpuDevice();

  // First lookup compiles and caches
  const modVs1 = cache.getOrCreateShaderModule(mock1.device, hVs, 'gta6');
  assert.equal(mock1.calls.length, 1);
  assert.equal(mock1.calls[0].desc.code, SAMPLE_VS_WGSL);
  assert.equal(mock1.calls[0].desc.label, `gta6:vertex:${formatHashHex(hVs)}`);

  // Second lookup (even via hex string) returns identical cached GPUShaderModule
  const modVs2 = cache.getOrCreateShaderModule(mock1.device, formatHashHex(hVs), 'gta6');
  assert.equal(modVs2, modVs1);
  assert.equal(mock1.calls.length, 1);

  // Different device compiles its own module instead of returning mock1's module
  const modVsOtherDevice = cache.getOrCreateShaderModule(mock2.device, hVs);
  assert.notEqual(modVsOtherDevice, modVs1);
  assert.equal(mock2.calls.length, 1);
  assert.equal(mock2.calls[0].desc.label, `wgpk:vertex:${formatHashHex(hVs)}`);

  // Unknown shader throws and counts as a miss without incrementing compiles
  assert.throws(
    () => cache.getOrCreateShaderModule(mock1.device, 0xdeadbeefn),
    /not found in any loaded wgsl pack/i
  );

  const stats = cache.getStats();
  assert.equal(stats.packCount, 2);
  assert.equal(stats.shaderCount, 3);
  assert.equal(stats.cachedModules, 2);
  assert.equal(stats.hits, 1);
  assert.equal(stats.misses, 3);
  assert.equal(stats.compiles, 2);
  assert.equal(stats.evictions, 0);
  assert.equal(stats.hitRate, 1 / 4);
  assert.equal(stats.totalPackBytes, pack1.byteLength + pack2.byteLength);

  // Evict removes cached modules across devices and forces recompilation on next access
  assert.equal(cache.evict(hVs), true);
  assert.equal(cache.evict(hVs), false);
  assert.equal(cache.getStats().cachedModules, 0);
  assert.equal(cache.getStats().evictions, 2);

  const modVsRecompiled = cache.getOrCreateShaderModule(mock1.device, hVs);
  assert.notEqual(modVsRecompiled, modVs1);
  assert.equal(mock1.calls.length, 2);

  // Evict with removeSource drops the pack entry too
  assert.equal(cache.evict(hCs, { removeSource: true }), true);
  assert.equal(cache.hasShader(hCs), false);

  // Clear empties packs, shaders, and cached modules
  cache.clear({ resetStats: true });
  const cleared = cache.getStats();
  assert.equal(cleared.packCount, 0);
  assert.equal(cleared.shaderCount, 0);
  assert.equal(cleared.cachedModules, 0);
  assert.equal(cleared.hits, 0);
  assert.equal(cleared.misses, 0);
  assert.equal(cleared.compiles, 0);
});

test('WgslPackCache.prewarm batch-compiles with bounded concurrency and skips cached shaders', async () => {
  const entries = [];
  for (let i = 0; i < 10; i += 1) {
    const wgsl = `@compute @workgroup_size(${i + 1}) fn main() {}`;
    entries.push({
      hash: fnv1a64(`compute-shader-${i}`),
      stage: SHADER_STAGE.COMPUTE,
      wgsl,
    });
  }

  const cache = new WgslPackCache();
  cache.loadPackBuffer(buildWgslPackBuffer(entries), 'batch.wgsl.pack');

  const mock = createMockGpuDevice({ withCompilationInfo: true, compileDelayMs: 5 });

  // Pre-compile entry 0 first
  const preExisting = cache.getOrCreateShaderModule(mock.device, entries[0].hash);

  // Prewarm first 8 hashes (with a duplicate) at concurrency = 3
  const subset = [
    entries[0].hash,
    entries[1].hash,
    entries[2].hash,
    entries[2].hash, // duplicate should be deduplicated
    entries[3].hash,
    entries[4].hash,
    entries[5].hash,
    entries[6].hash,
    entries[7].hash,
  ];
  const res = await cache.prewarm(mock.device, subset, { concurrency: 3, labelPrefix: 'warm' });
  assert.equal(res.total, 8);
  assert.equal(res.cached, 1);
  assert.equal(res.compiled, 7);
  assert.equal(res.modules.get(entries[0].hash), preExisting);
  assert.equal(mock.calls.length, 8);
  assert.ok(mock.getPeakCompilations() <= 3, `expected peak concurrency <= 3, got ${mock.getPeakCompilations()}`);
  assert.ok(mock.getPeakCompilations() >= 2, `expected concurrent execution, got peak ${mock.getPeakCompilations()}`);

  // Prewarming without an explicit hash list compiles the remaining 2 shaders
  const resAll = await cache.prewarm(mock.device, undefined, { concurrency: 2 });
  assert.equal(resAll.total, 10);
  assert.equal(resAll.cached, 8);
  assert.equal(resAll.compiled, 2);
  assert.equal(mock.calls.length, 10);
});

test('WgslPackCache supports loadPackUrl, pack replacement, and shader-index.json metadata', async () => {
  const hVs = 0x00000000deadbeefn;
  const hFs = 0xfeedfacecafebaben;

  const packV1 = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);

  const shaderIndex = {
    abi_version: 1,
    pack: 'shaders.wgsl.pack',
    shaders: {
      '0x00000000deadbeef': {
        stage: 'vertex',
        wgsl: 'shaders/00000000deadbeef.wgsl',
        entry_point: 'vs_main',
        pack_offset: WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES,
        pack_length: new TextEncoder().encode(SAMPLE_VS_WGSL).byteLength,
      },
      '0xfeedfacecafebabe': {
        stage: 'fragment',
        wgsl: 'shaders/feedfacecafebabe.wgsl',
        entry_point: 'fs_main',
      },
    },
  };

  const fetchedUrls = [];
  const mockFetch = async (url) => {
    fetchedUrls.push(url);
    if (url === '/assets/shaders.wgsl.pack') {
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        async arrayBuffer() {
          return packV1;
        },
      };
    }
    if (url === '/assets/shader-index.json') {
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        async json() {
          return shaderIndex;
        },
      };
    }
    return {
      ok: false,
      status: 404,
      statusText: 'Not Found',
      async arrayBuffer() {
        return new ArrayBuffer(0);
      },
    };
  };

  const cache = new WgslPackCache();
  const loadedPack = await cache.loadPackUrl('/assets/shaders.wgsl.pack', mockFetch);
  assert.equal(loadedPack.size, 2);

  const indexInfo = await cache.loadShaderIndexUrl('/assets/shader-index.json', mockFetch);
  assert.deepEqual(indexInfo, {
    abiVersion: 1,
    pack: 'shaders.wgsl.pack',
    count: 2,
  });
  assert.deepEqual(fetchedUrls, ['/assets/shaders.wgsl.pack', '/assets/shader-index.json']);

  const meta = cache.getShaderMetadata(hVs);
  assert.equal(meta.hash, hVs);
  assert.equal(meta.hashHex, '0x00000000deadbeef');
  assert.equal(meta.stage, 'vertex');
  assert.equal(meta.stageCode, 1);
  assert.equal(meta.entryPoint, 'vs_main');
  assert.equal(meta.wgslPath, 'shaders/00000000deadbeef.wgsl');
  assert.equal(meta.packName, '/assets/shaders.wgsl.pack');

  // HTTP 404 rejects with clear error
  await assert.rejects(
    () => cache.loadPackUrl('/assets/missing.wgsl.pack', mockFetch),
    /HTTP 404 Not Found/
  );

  // Unsupported shader-index.json abi_version rejects
  assert.throws(
    () => cache.loadShaderIndex({ abi_version: 2, pack: null, shaders: {} }),
    /unsupported shader-index\.json abi_version 2/i
  );

  // Reloading the same pack name with updated WGSL invalidates stale compiled modules
  const mock = createMockGpuDevice();
  const modOld = cache.getOrCreateShaderModule(mock.device, hVs);
  const updatedVs = '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const packV2 = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: updatedVs },
  ]);
  cache.loadPackBuffer(packV2, '/assets/shaders.wgsl.pack');
  assert.equal(cache.hasShader(hFs), false);
  assert.equal(cache.getShaderSource(hVs), updatedVs);

  const modNew = cache.getOrCreateShaderModule(mock.device, hVs);
  assert.notEqual(modNew, modOld);
  assert.equal(modNew.code, updatedVs);
});

test('G40 WgslPackCache.registerDynamicShader validates inputs, normalizes hashes, and supports stage checks', () => {
  const cache = new WgslPackCache();
  const hashBig = 0x00000000cafebaben;
  const dynWgsl = '@fragment fn fs_live() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.8, 0.4, 1.0); }';

  // Reject invalid hash, stage, wgslSource, and meta shapes
  assert.throws(() => cache.registerDynamicShader(-1n, 'fragment', dynWgsl), RangeError);
  assert.throws(() => cache.registerDynamicShader(1n << 64n, 'fragment', dynWgsl), RangeError);
  assert.throws(() => cache.registerDynamicShader('0xzzzz', 'fragment', dynWgsl), RangeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'geometry', dynWgsl), RangeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 0, dynWgsl), RangeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 99, dynWgsl), RangeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, null, dynWgsl), TypeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', ''), RangeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', '   \n\t  '), RangeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', null), TypeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', 123), TypeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', dynWgsl, 'bad-meta'), TypeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', dynWgsl, []), TypeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', dynWgsl, { label: 42 }), TypeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', dynWgsl, { entryPoint: 42 }), TypeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', dynWgsl, { source: '' }), TypeError);
  assert.throws(() => cache.registerDynamicShader(hashBig, 'fragment', dynWgsl, { tags: [123] }), TypeError);

  // Register via uppercase unprefixed hex string and uppercase stage name
  const registered = cache.registerDynamicShader('CAFEBABE', 'FRAGMENT', dynWgsl, {
    label: 'rdna2-live',
    entryPoint: 'fs_live',
    source: 'rdna2-translator',
    tags: ['live', 'rdna2'],
  });

  assert.equal(registered.hash, hashBig);
  assert.equal(registered.hashHex, '0x00000000cafebabe');
  assert.equal(registered.stage, 'fragment');
  assert.equal(registered.stageCode, SHADER_STAGE.FRAGMENT);
  assert.equal(registered.wgsl, dynWgsl);
  assert.equal(registered.length, new TextEncoder().encode(dynWgsl).byteLength);
  assert.equal(registered.dynamic, true);
  assert.equal(registered.overridesStatic, false);
  assert.equal(registered.label, 'rdna2-live');
  assert.equal(registered.entryPoint, 'fs_live');
  assert.equal(registered.source, 'rdna2-translator');
  assert.deepEqual(registered.tags, ['live', 'rdna2']);
  assert.equal(registered.revision, 1);
  assert.equal(typeof registered.registeredAt, 'number');

  // Normalized lookups via BigInt, prefixed hex, unprefixed hex, and safe integer
  assert.equal(cache.hasDynamicShader(hashBig), true);
  assert.equal(cache.hasDynamicShader('0x00000000cafebabe', 'fragment'), true);
  assert.equal(cache.hasDynamicShader('cafebabe', SHADER_STAGE.FRAGMENT), true);
  assert.equal(cache.hasDynamicShader(0xcafebabe, 'FRAGMENT'), true);
  assert.equal(cache.hasDynamicShader(hashBig, 'vertex'), false);
  assert.equal(cache.hasDynamicShader(hashBig, SHADER_STAGE.COMPUTE), false);
  assert.equal(cache.hasDynamicShader(0x1234n), false);
  assert.throws(() => cache.hasDynamicShader(hashBig, 'tessellation'), RangeError);

  assert.equal(cache.hasShader(hashBig), true);
  assert.equal(cache.hasShader(hashBig, 'fragment'), true);
  assert.equal(cache.hasShader(hashBig, 'vertex'), false);

  const viaLookup = cache.lookup('0xCAFEBABE', 'fragment');
  assert.equal(viaLookup.wgsl, dynWgsl);
  assert.equal(viaLookup.dynamic, true);
  const viaGet = cache.get(hashBig);
  assert.equal(viaGet.wgsl, dynWgsl);
  assert.equal(cache.get(hashBig, 'vertex'), undefined);

  const stats = cache.getStats();
  assert.equal(stats.dynamicCount, 1);
  assert.equal(stats.staticShaderCount, 0);
  assert.equal(stats.shaderCount, 1);
  assert.equal(stats.dynamicOverrideCount, 0);
  assert.equal(stats.dynamicHits, 2); // viaLookup + viaGet (mismatched stage did not count)
  assert.deepEqual(stats.dynamicStageCounts, { vertex: 0, fragment: 1, compute: 0 });
  assert.deepEqual(stats.stageCounts, { vertex: 0, fragment: 1, compute: 0 });
});

test('G40 WgslPackCache hot-overrides static pack entries safely, evicts stale GPUShaderModules, and restores on clearDynamicShaders', () => {
  const hVs = fnv1a64('static-vs-override-test');
  const hFs = fnv1a64('static-fs-untouched');
  const hDynOnly = fnv1a64('dynamic-only-cs');

  const staticPack = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);

  const cache = new WgslPackCache({ lazyDecode: true });
  cache.loadPackBuffer(staticPack, 'boot.wgsl.pack');

  const mock1 = createMockGpuDevice();
  const mock2 = createMockGpuDevice();

  // Compile static VS on two devices before overriding
  const staticMod1 = cache.getOrCreateShaderModule(mock1.device, hVs, 'gta6');
  const staticMod2 = cache.getOrCreateShaderModule(mock2.device, hVs, 'gta6');
  assert.equal(staticMod1.code, SAMPLE_VS_WGSL);
  assert.equal(staticMod2.code, SAMPLE_VS_WGSL);
  assert.equal(cache.getStats().dynamicHits, 0);

  // Attempting to register over a static shader without { override: true } must throw
  const hotVsV1 =
    '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.5, 0.5, 0.0, 1.0); }';
  assert.throws(
    () => cache.registerDynamicShader(hVs, 'vertex', hotVsV1),
    /already exists in static pack.*override: true/i
  );
  assert.equal(cache.hasDynamicShader(hVs), false);
  assert.equal(cache.getOrCreateShaderModule(mock1.device, hVs), staticMod1);

  // Register dynamic override with { override: true } -> evicts stale modules on both devices
  const overrideV1 = cache.registerDynamicShader(hVs, 'vertex', hotVsV1, {
    override: true,
    label: 'hotfix-v1',
    entryPoint: 'vs_main',
    source: 'hot-patch',
  });
  assert.equal(overrideV1.overridesStatic, true);
  assert.equal(overrideV1.revision, 1);
  assert.equal(cache.getStats().evictions, 2);
  assert.equal(cache.getStats().cachedModules, 0);

  // Also register a purely dynamic compute shader (no static entry, override flag not required)
  const dynCs = '@compute @workgroup_size(8, 8, 1) fn cs_live() {}';
  cache.registerDynamicShader(hDynOnly, SHADER_STAGE.COMPUTE, dynCs, {
    label: 'rdna2-cs',
    source: 'rdna2-translator',
  });

  // Lookups prioritize dynamic over static, while untouched static shaders still resolve from pack
  assert.equal(cache.getShaderSource(hVs), hotVsV1);
  assert.equal(cache.getShaderSource(hFs), SAMPLE_FS_WGSL);
  assert.equal(cache.getShaderSource(hDynOnly), dynCs);

  // getOrCreateShaderModule compiles the hot-patched WGSL and appends dynamic label
  const hotMod1 = cache.getOrCreateShaderModule(mock1.device, hVs, 'gta6');
  assert.notEqual(hotMod1, staticMod1);
  assert.equal(hotMod1.code, hotVsV1);
  assert.equal(hotMod1.label, `gta6:vertex:${formatHashHex(hVs)}:hotfix-v1`);

  // Second call hits cache AND increments dynamicHits
  const hitsBefore = cache.getStats().dynamicHits;
  const hotMod1Cached = cache.getOrCreateShaderModule(mock1.device, hVs, 'gta6');
  assert.equal(hotMod1Cached, hotMod1);
  assert.equal(cache.getStats().dynamicHits, hitsBefore + 1);

  // Hot-patching the dynamic shader again bumps revision and evicts stale dynamic module
  const hotVsV2 =
    '@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.25, 0.75, 0.0, 1.0); }';
  const overrideV2 = cache.registerDynamicShader(hVs, 'vertex', hotVsV2, {
    override: true,
    label: 'hotfix-v2',
  });
  assert.equal(overrideV2.revision, 2);
  const hotMod2 = cache.getOrCreateShaderModule(mock1.device, hVs, 'gta6');
  assert.notEqual(hotMod2, hotMod1);
  assert.equal(hotMod2.code, hotVsV2);
  assert.equal(hotMod2.label, `gta6:vertex:${formatHashHex(hVs)}:hotfix-v2`);

  // Reloading the static pack preserves active dynamic override and its compiled module
  const updatedStaticPack = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  cache.loadPackBuffer(updatedStaticPack, 'boot.wgsl.pack');
  assert.equal(cache.getOrCreateShaderModule(mock1.device, hVs), hotMod2);

  // Verify stats accounting (2 static + 1 new dynamic + 1 overriding dynamic => 3 distinct shaders)
  const midStats = cache.getStats();
  assert.equal(midStats.staticShaderCount, 2);
  assert.equal(midStats.dynamicCount, 2);
  assert.equal(midStats.dynamicOverrideCount, 1);
  assert.equal(midStats.shaderCount, 3);
  assert.equal(
    midStats.dynamicPayloadBytes,
    new TextEncoder().encode(hotVsV2).byteLength + new TextEncoder().encode(dynCs).byteLength
  );
  assert.ok(midStats.dynamicHits >= 5);

  // clearDynamicShaders removes all dynamic shaders, evicts their compiled modules, and restores static entries
  const clearedCount = cache.clearDynamicShaders();
  assert.equal(clearedCount, 2);
  assert.equal(cache.hasDynamicShader(hVs), false);
  assert.equal(cache.hasDynamicShader(hDynOnly), false);
  assert.equal(cache.hasShader(hDynOnly), false);
  assert.equal(cache.hasShader(hVs), true);
  assert.equal(cache.hasShader(hFs), true);
  assert.equal(cache.getShaderSource(hVs), SAMPLE_VS_WGSL);

  // Next compile uses the restored static pack WGSL
  const restoredMod = cache.getOrCreateShaderModule(mock1.device, hVs, 'gta6');
  assert.notEqual(restoredMod, hotMod2);
  assert.equal(restoredMod.code, SAMPLE_VS_WGSL);
  assert.equal(restoredMod.label, `gta6:vertex:${formatHashHex(hVs)}`);

  const afterClearStats = cache.getStats();
  assert.equal(afterClearStats.dynamicCount, 0);
  assert.equal(afterClearStats.dynamicOverrideCount, 0);
  assert.equal(afterClearStats.staticShaderCount, 2);
  assert.equal(afterClearStats.shaderCount, 2);

  // clear({ resetStats: true }) resets dynamicHits too
  cache.clear({ resetStats: true });
  assert.equal(cache.getStats().dynamicHits, 0);
  assert.equal(cache.getStats().dynamicCount, 0);
});

test('G40 WgslPackCache introspection, unregisterDynamicShader, lazy-decode preservation, and prewarm with dynamic shaders', async () => {
  const hVs = fnv1a64('inspect-vs');
  const hFs = fnv1a64('inspect-fs');
  const hDynCs = fnv1a64('inspect-dyn-cs');

  const packBuf = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);

  const cache = new WgslPackCache({ lazyDecode: true });
  const pack = cache.loadPackBuffer(packBuf, 'main.wgsl.pack');
  cache.loadShaderIndex({
    abi_version: 1,
    pack: 'main.wgsl.pack',
    shaders: {
      [formatHashHex(hVs)]: {
        stage: 'vertex',
        wgsl: 'shaders/vs.wgsl',
        entry_point: 'vs_main',
      },
    },
  });

  const hotVs = '@vertex fn vs_hot() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const liveCs = '@compute @workgroup_size(4) fn cs_hot() {}';
  cache.registerDynamicShader(hVs, 'vertex', hotVs, {
    override: true,
    label: 'vs-override',
    entryPoint: 'vs_hot',
    source: 'hot-patch',
  });
  cache.registerDynamicShader(hDynCs, 'compute', liveCs, {
    label: 'cs-live',
    entryPoint: 'cs_hot',
    source: 'rdna2-translator',
  });

  // Introspection must NOT trigger lazy UTF-8 decoding on static pack entries
  assert.equal(pack.stats().decodedCount, 0);
  const dynList = cache.listDynamicShaders();
  assert.equal(dynList.length, 2);
  assert.equal(dynList[0].wgsl, undefined);
  assert.equal(cache.listDynamicShaders({ stage: 'compute', includeSource: true })[0].wgsl, liveCs);

  const allActive = cache.listShaders();
  assert.equal(allActive.length, 3);
  assert.equal(cache.listShaders({ dynamicOnly: true }).length, 2);
  assert.equal(cache.listShaders({ staticOnly: true }).length, 2);
  assert.equal(cache.listShaders({ stage: 'fragment' }).length, 1);

  const staticOnlyVs = cache.listShaders({ staticOnly: true }).find((s) => s.hash === hVs);
  assert.equal(staticOnlyVs.overriddenByDynamic, true);

  const metaVs = cache.getShaderMetadata(hVs);
  assert.equal(metaVs.dynamic, true);
  assert.equal(metaVs.overridesStatic, true);
  assert.equal(metaVs.entryPoint, 'vs_hot');
  assert.equal(metaVs.wgslPath, 'shaders/vs.wgsl');
  assert.equal(metaVs.packName, 'main.wgsl.pack');

  const metaFs = cache.getShaderMetadata(hFs);
  assert.equal(metaFs.dynamic, false);
  assert.equal(metaFs.overridesStatic, false);

  // inspect() produces a JSON-serializable snapshot without decoding static payloads
  const snapshot = cache.inspect();
  assert.doesNotThrow(() => JSON.stringify(snapshot));
  assert.deepEqual(snapshot.packNames, ['main.wgsl.pack']);
  assert.equal(snapshot.shaderIndex.count, 1);
  assert.equal(snapshot.dynamicShaders.length, 2);
  assert.equal(pack.stats().decodedCount, 0);

  // prewarm() without an explicit hash list compiles static + dynamic shaders (deduplicated)
  const mock = createMockGpuDevice();
  const warm = await cache.prewarm(mock.device);
  assert.equal(warm.total, 3);
  assert.equal(warm.compiled, 3);
  assert.equal(warm.modules.get(hVs).code, hotVs);
  assert.equal(warm.modules.get(hFs).code, SAMPLE_FS_WGSL);
  assert.equal(warm.modules.get(hDynCs).code, liveCs);

  // unregisterDynamicShader removes only the targeted override and restores its static fallback
  assert.equal(cache.unregisterDynamicShader(hVs), true);
  assert.equal(cache.unregisterDynamicShader(hVs), false);
  assert.equal(cache.hasDynamicShader(hVs), false);
  assert.equal(cache.hasDynamicShader(hDynCs), true);
  assert.equal(cache.getShaderSource(hVs), SAMPLE_VS_WGSL);
  assert.equal(cache.getOrCreateShaderModule(mock.device, hVs).code, SAMPLE_VS_WGSL);
});

test('G40 WgslPack supports direct dynamic shader registration, overrides, lookup/get, and stats', () => {
  const hVs = 0x1111222233334444n;
  const hNew = 0x5555666677778888n;

  const packBuf = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
  ]);
  const pack = parseWgslPack(packBuf, { lazyDecode: true });

  assert.equal(pack.size, 1);
  assert.equal(pack.getStats().dynamicCount, 0);
  assert.equal(pack.getStats().dynamicHits, 0);

  // Static collision without override throws
  const patchedVs = '@vertex fn vs_patched() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }';
  assert.throws(() => pack.registerDynamicShader(hVs, 'vertex', patchedVs), /override: true/i);

  // Register override + brand-new dynamic shader
  pack.registerDynamicShader(hVs, 'vertex', patchedVs, { override: true, label: 'pack-patch' });
  pack.registerDynamicShader(hNew, 'compute', SAMPLE_CS_WGSL, { source: 'rdna2-translator' });

  assert.equal(pack.size, 2);
  assert.equal(pack.has(hNew), true);
  assert.equal(pack.hasDynamicShader(hVs, 'vertex'), true);
  assert.equal(pack.hasDynamicShader(hVs, 'fragment'), false);
  assert.equal(pack.hasDynamicShader(hNew, SHADER_STAGE.COMPUTE), true);

  assert.equal(pack.lookup(hVs, 'vertex').wgsl, patchedVs);
  assert.equal(pack.get(hNew, 'compute').wgsl, SAMPLE_CS_WGSL);
  assert.equal(pack.get(hNew, 'vertex'), undefined);
  assert.equal(pack.getSource(hVs), patchedVs);
  assert.equal(pack.listDynamicShaders().length, 2);

  // Static pack table stays untouched and un-decoded
  assert.equal(pack.listEntries().length, 1);
  assert.equal(pack.getStats().decodedCount, 0);
  assert.equal(pack.getStats().dynamicCount, 2);
  assert.equal(pack.getStats().dynamicOverrideCount, 1);
  assert.equal(pack.getStats().dynamicHits, 3);

  // Clearing dynamic shaders restores static entry cleanly
  assert.equal(pack.clearDynamicShaders(), 2);
  assert.equal(pack.size, 1);
  assert.equal(pack.has(hNew), false);
  assert.equal(pack.hasDynamicShader(hVs), false);
  assert.equal(pack.getSource(hVs), SAMPLE_VS_WGSL);
  assert.equal(pack.getStats().dynamicCount, 0);
});

// ---------------------------------------------------------------------------
// W109: HTTP 206 Range index-only & on-demand WGSL streaming tests (#737)
// ---------------------------------------------------------------------------

function createRangeMockFetch(fileMap, options = {}) {
  const requests = [];
  const ignoreRange = Boolean(options.ignoreRange);
  const rejectHeadStatus = options.rejectHeadStatus || 0;
  const acceptRangesHeader =
    options.acceptRangesHeader !== undefined ? options.acceptRangesHeader : 'bytes';
  const maxChunkBytes = options.maxChunkBytes || Infinity;
  const beforeResponse = options.beforeResponse || null;

  const readHeader = (headers, name) => {
    if (!headers) return null;
    if (typeof headers.get === 'function') return headers.get(name);
    const target = name.toLowerCase();
    for (const [k, v] of Object.entries(headers)) {
      if (k.toLowerCase() === target) return String(v);
    }
    return null;
  };

  const mockFetch = async (url, init = {}) => {
    const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
    const range = readHeader(init && init.headers, 'Range');

    if (beforeResponse) {
      const override = await beforeResponse({ url, method, range, init, requests });
      if (override) {
        requests.push({
          url,
          method,
          range,
          status: override.status,
          bytesSent: 0,
          start: null,
          end: null,
        });
        return override;
      }
    }

    const raw = fileMap[url];
    if (!raw) {
      requests.push({ url, method, range, status: 404, bytesSent: 0, start: null, end: null });
      return {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        headers: new Headers(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    const bytes = raw instanceof Uint8Array ? raw : new Uint8Array(raw);
    const total = bytes.byteLength;

    if (method === 'HEAD') {
      if (rejectHeadStatus) {
        requests.push({
          url,
          method,
          range,
          status: rejectHeadStatus,
          bytesSent: 0,
          start: null,
          end: null,
        });
        return {
          ok: false,
          status: rejectHeadStatus,
          statusText: 'Method Not Allowed',
          headers: new Headers(),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      const headers = new Headers({
        'Content-Length': String(total),
        'Accept-Ranges': acceptRangesHeader,
      });
      requests.push({ url, method, range: null, status: 200, bytesSent: 0, start: null, end: null });
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        headers,
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (range && !ignoreRange) {
      const match = /^bytes=(\d+)-(\d+)$/i.exec(range.trim());
      if (!match) {
        requests.push({ url, method, range, status: 416, bytesSent: 0, start: null, end: null });
        return {
          ok: false,
          status: 416,
          statusText: 'Range Not Satisfiable',
          headers: new Headers({ 'Content-Range': `bytes */${total}` }),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      const reqStart = Number(match[1]);
      const reqEnd = Number(match[2]);
      if (reqStart >= total || reqEnd < reqStart) {
        requests.push({ url, method, range, status: 416, bytesSent: 0, start: reqStart, end: reqEnd });
        return {
          ok: false,
          status: 416,
          statusText: 'Range Not Satisfiable',
          headers: new Headers({ 'Content-Range': `bytes */${total}` }),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      const clampedEnd = Math.min(reqEnd, total - 1, reqStart + maxChunkBytes - 1);
      const slice = bytes.slice(reqStart, clampedEnd + 1);
      requests.push({
        url,
        method,
        range,
        status: 206,
        bytesSent: slice.byteLength,
        start: reqStart,
        end: clampedEnd,
      });
      return {
        ok: true,
        status: 206,
        statusText: 'Partial Content',
        headers: new Headers({
          'Content-Range': `bytes ${reqStart}-${clampedEnd}/${total}`,
          'Content-Length': String(slice.byteLength),
          'Accept-Ranges': 'bytes',
        }),
        async arrayBuffer() {
          return slice.buffer;
        },
      };
    }

    // Full 200 OK response
    const copy = bytes.slice();
    requests.push({
      url,
      method,
      range,
      status: 200,
      bytesSent: copy.byteLength,
      start: 0,
      end: copy.byteLength - 1,
    });
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: new Headers({
        'Content-Length': String(copy.byteLength),
      }),
      async arrayBuffer() {
        return copy.buffer;
      },
    };
  };

  return { fetch: mockFetch, requests };
}

test('W109 loadIndexFromUrl parses header + table via HTTP 206 Range (single-step and two-step) without fetching payload bytes', async () => {
  assert.equal(DEFAULT_INITIAL_RANGE_BYTES, 64 * 1024);
  assert.equal(DEFAULT_MAX_CACHED_WGSL_BYTES, 32 * 1024 * 1024);

  // Build a pack with 10 entries: tableEnd = 16 + 10 * 20 = 216 bytes, payloads start at byte 216.
  const entries = [];
  for (let i = 0; i < 10; i += 1) {
    const stage = (i % 3) + 1;
    const wgsl = `// shader #${i} ${'x'.repeat(200)}\n@compute @workgroup_size(${i + 1}) fn main_${i}() {}`;
    entries.push({
      hash: fnv1a64(`ranged-shader-${i}`),
      stage,
      wgsl,
    });
  }
  const packBuf = buildWgslPackBuffer(entries, { flags: 0x1234 });
  const tableEnd = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES;
  assert.equal(tableEnd, 216);
  assert.ok(packBuf.byteLength > 2500);

  // 1. Single-step range load: initialRangeBytes exactly covers header + table (216 bytes) -> 1 roundtrip, 0 payload bytes!
  const singleMock = createRangeMockFetch({ '/packs/large.wgsl.pack': packBuf });
  const singlePack = await loadIndexFromUrl('/packs/large.wgsl.pack', {
    fetch: singleMock.fetch,
    initialRangeBytes: tableEnd,
  });

  assert.equal(singlePack.mode, 'ranged');
  assert.equal(singlePack.flags, 0x1234);
  assert.equal(singlePack.size, 10);
  assert.equal(singlePack.byteLength, packBuf.byteLength);
  assert.equal(singlePack.dataStart, tableEnd);
  assert.equal(singleMock.requests.length, 1);
  assert.equal(singleMock.requests[0].range, `bytes=0-${tableEnd - 1}`);
  assert.equal(singleMock.requests[0].bytesSent, tableEnd);
  assert.ok(singleMock.requests[0].end < tableEnd, 'must not fetch any payload bytes');

  const singleStats = singlePack.stats();
  assert.equal(singleStats.mode, 'ranged');
  assert.equal(singleStats.rangeFetches, 1);
  assert.equal(singleStats.cacheHits, 0);
  assert.equal(singleStats.cacheMisses, 0);
  assert.equal(singleStats.cachedWgslBytes, 0);
  assert.equal(singleStats.decodedCount, 0);

  // Metadata inspection and existence checks trigger zero additional HTTP requests
  assert.equal(singlePack.has(entries[0].hash), true);
  assert.equal(singlePack.has(entries[9].hash), true);
  assert.equal(singlePack.has(0x999999n), false);
  assert.equal(singlePack.listEntries().length, 10);
  assert.equal(singleMock.requests.length, 1);

  // Sync getWgsl / getSource on an unfetched ranged entry throws a clear actionable error
  assert.throws(() => singlePack.getWgsl(entries[0].hash), /getWgslAsync/i);

  // 2. Two-step range load: initialRangeBytes = 64 (< tableEnd 216) -> fetches 0-63 then 64-215, zero payload bytes!
  const twoStepMock = createRangeMockFetch({ '/packs/large.wgsl.pack': packBuf });
  const twoStepPack = await WgslPack.loadIndexFromUrl('/packs/large.wgsl.pack', {
    fetch: twoStepMock.fetch,
    initialRangeBytes: 64,
  });

  assert.equal(twoStepPack.mode, 'ranged');
  assert.equal(twoStepPack.size, 10);
  assert.equal(twoStepMock.requests.length, 2);
  assert.equal(twoStepMock.requests[0].range, 'bytes=0-63');
  assert.equal(twoStepMock.requests[1].range, `bytes=64-${tableEnd - 1}`);
  const totalBytesDownloaded = twoStepMock.requests.reduce((acc, r) => acc + r.bytesSent, 0);
  assert.equal(totalBytesDownloaded, tableEnd);
  assert.equal(twoStepPack.stats().rangeFetches, 2);

  // 3. Header-only initial range (initialRangeBytes = 16) also works in two exact steps (0-15, then 16-215)
  const headerOnlyMock = createRangeMockFetch({ '/packs/large.wgsl.pack': packBuf });
  const headerOnlyPack = await loadIndexFromUrl('/packs/large.wgsl.pack', {
    fetch: headerOnlyMock.fetch,
    initialRangeBytes: WGPK_HEADER_BYTES,
  });
  assert.equal(headerOnlyPack.size, 10);
  assert.equal(headerOnlyMock.requests.length, 2);
  assert.equal(headerOnlyMock.requests[0].range, 'bytes=0-15');
  assert.equal(headerOnlyMock.requests[1].range, `bytes=16-${tableEnd - 1}`);
});

test('W109 getWgslAsync fetches exact byte ranges, caches hits, deduplicates concurrent calls, and recovers from errors', async () => {
  const utf8Fs = `// UTF-8 range shader 日本語 🚀\n${SAMPLE_FS_WGSL}`;
  const hVs = fnv1a64('async-vs');
  const hFs = fnv1a64('async-fs');
  const hCs = fnv1a64('async-cs');
  const hEmpty = fnv1a64('async-empty');

  const packBuf = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: utf8Fs },
    { hash: hCs, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
    { hash: hEmpty, stage: SHADER_STAGE.COMPUTE, wgsl: '' },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 4 * WGPK_ENTRY_BYTES;

  // Gate to deterministically hold concurrent requests in flight
  let releaseGate = null;
  let gateActive = false;
  let failNextCs = false;

  const mock = createRangeMockFetch(
    { '/packs/on-demand.wgsl.pack': packBuf },
    {
      beforeResponse: async ({ range }) => {
        if (failNextCs && range && !range.startsWith('bytes=0-')) {
          failNextCs = false;
          return {
            ok: false,
            status: 500,
            statusText: 'Internal Server Error',
            headers: new Headers(),
            async arrayBuffer() {
              return new ArrayBuffer(0);
            },
          };
        }
        if (gateActive && range && !range.startsWith('bytes=0-')) {
          await new Promise((resolve) => {
            releaseGate = resolve;
          });
        }
        return null;
      },
    }
  );

  const pack = await loadIndexFromUrl('/packs/on-demand.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    maxRangeFetchRetries: 0,
  });
  assert.equal(mock.requests.length, 1);

  // 1. Fetch UTF-8 fragment shader by exact byte range
  const fsMeta = pack.listEntries().find((e) => e.hash === hFs);
  const fsWgsl = await pack.getWgslAsync(hFs);
  assert.equal(fsWgsl, utf8Fs);
  assert.equal(mock.requests.length, 2);
  assert.equal(mock.requests[1].range, `bytes=${fsMeta.offset}-${fsMeta.offset + fsMeta.length - 1}`);
  assert.equal(mock.requests[1].bytesSent, fsMeta.length);

  let stats = pack.stats();
  assert.equal(stats.rangeFetches, 2);
  assert.equal(stats.cacheMisses, 1);
  assert.equal(stats.cacheHits, 0);
  assert.equal(stats.cachedWgslBytes, fsMeta.length);
  assert.equal(stats.decodedCount, 1);

  // Once cached, even sync getWgsl() works while resident in LRU
  assert.equal(pack.getWgsl(hFs), utf8Fs);

  // 2. Repeated getWgslAsync (via hex string) hits LRU cache with 0 extra HTTP fetches
  const fsWgslCached = await pack.getWgslAsync(formatHashHex(hFs));
  assert.equal(fsWgslCached, utf8Fs);
  assert.equal(mock.requests.length, 2);
  stats = pack.stats();
  assert.equal(stats.rangeFetches, 2);
  assert.equal(stats.cacheHits, 1);
  assert.equal(stats.cacheMisses, 1);

  // 3. Unknown hash returns null with 0 HTTP fetches
  assert.equal(await pack.getWgslAsync(0xdeadbeefn), null);
  assert.equal(mock.requests.length, 2);

  // 4. Empty 0-byte payload resolves to '' with 0 HTTP fetches
  assert.equal(await pack.getWgslAsync(hEmpty), '');
  assert.equal(mock.requests.length, 2);

  // 5. Concurrent calls for the same hash (mixed BigInt / hex formats) deduplicate to 1 HTTP Range fetch
  gateActive = true;
  const p1 = pack.getWgslAsync(hVs);
  const p2 = pack.getWgslAsync(formatHashHex(hVs));
  const p3 = pack.getWgslAsync(formatHashHex(hVs).slice(2).toUpperCase());
  const p4 = pack.getWgslAsync(hVs);

  // Wait a microtask tick so the single fetch reaches the gate
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(typeof releaseGate, 'function', 'expected exactly one in-flight fetch to hit gate');
  gateActive = false;
  releaseGate();

  const [r1, r2, r3, r4] = await Promise.all([p1, p2, p3, p4]);
  assert.equal(r1, SAMPLE_VS_WGSL);
  assert.equal(r2, SAMPLE_VS_WGSL);
  assert.equal(r3, SAMPLE_VS_WGSL);
  assert.equal(r4, SAMPLE_VS_WGSL);
  assert.equal(mock.requests.length, 3, 'all 4 concurrent calls must share 1 Range request');
  assert.equal(pack.stats().inFlightDeduplications, 3);

  // 6. Failed in-flight fetch cleans up so subsequent retry succeeds
  failNextCs = true;
  await assert.rejects(() => pack.getWgslAsync(hCs), /HTTP 500/);
  const csWgsl = await pack.getWgslAsync(hCs);
  assert.equal(csWgsl, SAMPLE_CS_WGSL);
});

test('W109 ranged LRU cache enforces maxCachedWgslBytes, refreshes recency on hit, and evicts oldest entries', async () => {
  // Four 100-byte shaders
  const makeWgsl = (tag) => {
    const prefix = `@compute @workgroup_size(1) fn fn_${tag}() { /* `;
    const suffix = ' */ }';
    const padLen = 100 - prefix.length - suffix.length;
    return `${prefix}${'a'.repeat(padLen)}${suffix}`;
  };

  const hA = fnv1a64('lru-A');
  const hB = fnv1a64('lru-B');
  const hC = fnv1a64('lru-C');
  const hD = fnv1a64('lru-D');
  const wgslA = makeWgsl('A');
  const wgslB = makeWgsl('B');
  const wgslC = makeWgsl('C');
  const wgslD = makeWgsl('D');
  assert.equal(new TextEncoder().encode(wgslA).byteLength, 100);

  const packBuf = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.COMPUTE, wgsl: wgslA },
    { hash: hB, stage: SHADER_STAGE.COMPUTE, wgsl: wgslB },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, wgsl: wgslC },
    { hash: hD, stage: SHADER_STAGE.COMPUTE, wgsl: wgslD },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 4 * WGPK_ENTRY_BYTES;
  const mock = createRangeMockFetch({ '/packs/lru.wgsl.pack': packBuf });

  // Budget = 220 bytes -> holds at most 2 shaders (200 bytes)
  const pack = await loadIndexFromUrl('/packs/lru.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    maxCachedWgslBytes: 220,
  });

  // Load A and B -> cache: [A, B], 200 bytes
  assert.equal(await pack.getWgslAsync(hA), wgslA);
  assert.equal(await pack.getWgslAsync(hB), wgslB);
  assert.equal(pack.stats().cachedWgslBytes, 200);
  assert.equal(pack.stats().cachedWgslEntries, 2);
  assert.equal(pack.stats().wgslCacheEvictions, 0);
  assert.equal(mock.requests.length, 3); // 1 index + A + B

  // Access A -> cache hit refreshes recency -> cache order becomes [B, A]
  assert.equal(await pack.getWgslAsync(hA), wgslA);
  assert.equal(pack.stats().cacheHits, 1);
  assert.equal(mock.requests.length, 3);

  // Load C -> exceeds 220 -> evicts oldest (B) -> cache: [A, C], 200 bytes
  assert.equal(await pack.getWgslAsync(hC), wgslC);
  assert.equal(pack.stats().cachedWgslBytes, 200);
  assert.equal(pack.stats().wgslCacheEvictions, 1);
  assert.equal(mock.requests.length, 4);

  // A must still be cached (0 extra fetches)
  assert.equal(await pack.getWgslAsync(hA), wgslA);
  assert.equal(pack.stats().cacheHits, 2);
  assert.equal(mock.requests.length, 4);

  // B was evicted, so fetching B issues a new Range request and evicts C (since A was just touched)
  assert.equal(await pack.getWgslAsync(hB), wgslB);
  assert.equal(pack.stats().wgslCacheEvictions, 2);
  assert.equal(pack.stats().cachedWgslBytes, 200);
  assert.equal(mock.requests.length, 5);

  // clearWgslCache resets cached bytes to 0 and forces next access to re-fetch
  assert.equal(pack.clearWgslCache(), 2);
  assert.equal(pack.stats().cachedWgslBytes, 0);
  assert.equal(await pack.getWgslAsync(hD), wgslD);
  assert.equal(pack.stats().cachedWgslBytes, 100);
  assert.equal(mock.requests.length, 6);

  // Single shader larger than maxCachedWgslBytes returns WGSL cleanly without caching or looping
  const tinyMock = createRangeMockFetch({ '/packs/lru.wgsl.pack': packBuf });
  const tinyCachePack = await loadIndexFromUrl('/packs/lru.wgsl.pack', {
    fetch: tinyMock.fetch,
    initialRangeBytes: tableEnd,
    maxCachedWgslBytes: 50,
  });
  assert.equal(await tinyCachePack.getWgslAsync(hA), wgslA);
  assert.equal(tinyCachePack.stats().cachedWgslBytes, 0);
  assert.equal(tinyCachePack.stats().cachedWgslEntries, 0);
  // Second fetch re-fetches because it exceeded cache budget
  assert.equal(await tinyCachePack.getWgslAsync(hA), wgslA);
  assert.equal(tinyMock.requests.length, 3);
});

test('W109 loadIndexFromUrl falls back seamlessly to full buffer mode when server returns 200 OK', async () => {
  const hVs = fnv1a64('fallback-200-vs');
  const hFs = fnv1a64('fallback-200-fs');
  const packBuf = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);

  const mock = createRangeMockFetch({ '/packs/no-range.wgsl.pack': packBuf }, { ignoreRange: true });
  const pack = await loadIndexFromUrl('/packs/no-range.wgsl.pack', { fetch: mock.fetch });

  assert.equal(pack.mode, 'buffer');
  assert.equal(pack.size, 2);
  assert.equal(mock.requests.length, 1);
  assert.equal(mock.requests[0].status, 200);
  assert.equal(pack.stats().mode, 'buffer');
  assert.equal(pack.stats().rangeFetches, 0);

  // Both sync getWgsl and async getWgslAsync resolve immediately with zero extra network calls
  assert.equal(pack.getWgsl(hVs), SAMPLE_VS_WGSL);
  assert.equal(await pack.getWgslAsync(hFs), SAMPLE_FS_WGSL);
  assert.equal(await pack.getWgslAsync(0x12345n), null);
  assert.equal(mock.requests.length, 1);
});

test('W109 loadFromUrl with streamThresholdBytes automatically switches between buffer and ranged modes', async () => {
  const hSmall = fnv1a64('small-pack-shader');
  const smallPackBuf = buildWgslPackBuffer([
    { hash: hSmall, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
  ]);

  const hLarge1 = fnv1a64('large-pack-shader-1');
  const hLarge2 = fnv1a64('large-pack-shader-2');
  const largeWgsl1 = `// ${'A'.repeat(2048)}\n${SAMPLE_VS_WGSL}`;
  const largeWgsl2 = `// ${'B'.repeat(2048)}\n${SAMPLE_FS_WGSL}`;
  const largePackBuf = buildWgslPackBuffer([
    { hash: hLarge1, stage: SHADER_STAGE.VERTEX, wgsl: largeWgsl1 },
    { hash: hLarge2, stage: SHADER_STAGE.FRAGMENT, wgsl: largeWgsl2 },
  ]);
  const largeTableEnd = WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES;

  assert.ok(smallPackBuf.byteLength < 512);
  assert.ok(largePackBuf.byteLength > 4000);

  // 1. Default (no streamThresholdBytes): single GET -> buffer mode (legacy behavior unchanged)
  const defaultMock = createRangeMockFetch({ '/packs/large.wgsl.pack': largePackBuf });
  const defaultPack = await loadFromUrl('/packs/large.wgsl.pack', { fetch: defaultMock.fetch });
  assert.equal(defaultPack.mode, 'buffer');
  assert.equal(defaultMock.requests.length, 1);
  assert.equal(defaultMock.requests[0].method, 'GET');
  assert.equal(defaultMock.requests[0].range, null);

  // 2. Small pack (<= threshold) stays full-buffer via HEAD + GET
  const smallMock = createRangeMockFetch({ '/packs/small.wgsl.pack': smallPackBuf });
  const smallPack = await loadFromUrl('/packs/small.wgsl.pack', {
    fetch: smallMock.fetch,
    streamThresholdBytes: 1024,
  });
  assert.equal(smallPack.mode, 'buffer');
  assert.equal(smallMock.requests[0].method, 'HEAD');
  assert.equal(smallMock.requests[1].method, 'GET');
  assert.equal(smallPack.getWgsl(hSmall), SAMPLE_CS_WGSL);

  // 3. Large pack (> threshold) switches to ranged mode via HEAD + Range (no full download!)
  const largeMock = createRangeMockFetch({ '/packs/large.wgsl.pack': largePackBuf });
  const largePack = await loadFromUrl('/packs/large.wgsl.pack', {
    fetch: largeMock.fetch,
    streamThresholdBytes: 1024,
    initialRangeBytes: largeTableEnd,
  });
  assert.equal(largePack.mode, 'ranged');
  assert.equal(largeMock.requests.length, 2);
  assert.equal(largeMock.requests[0].method, 'HEAD');
  assert.equal(largeMock.requests[1].status, 206);
  assert.equal(largeMock.requests[1].bytesSent, largeTableEnd);
  assert.equal(await largePack.getWgslAsync(hLarge2), largeWgsl2);
  assert.equal(largeMock.requests.length, 3);

  // 4. Server rejecting HEAD with 405 falls back cleanly to initial Range probe (zero duplicate index fetch!)
  const noHeadMock = createRangeMockFetch(
    { '/packs/large.wgsl.pack': largePackBuf, '/packs/small.wgsl.pack': smallPackBuf },
    { rejectHeadStatus: 405 }
  );
  const noHeadLarge = await loadFromUrl('/packs/large.wgsl.pack', {
    fetch: noHeadMock.fetch,
    streamThresholdBytes: 1024,
    initialRangeBytes: largeTableEnd,
  });
  assert.equal(noHeadLarge.mode, 'ranged');
  assert.equal(noHeadMock.requests.length, 2); // HEAD (405) + 1 Range probe reused as index
  assert.equal(await noHeadLarge.getWgslAsync(hLarge1), largeWgsl1);

  const noHeadSmall = await loadFromUrl('/packs/small.wgsl.pack', {
    fetch: noHeadMock.fetch,
    streamThresholdBytes: 1024,
  });
  assert.equal(noHeadSmall.mode, 'buffer');
  assert.equal(noHeadSmall.getWgsl(hSmall), SAMPLE_CS_WGSL);

  // 5. Server advertising Accept-Ranges: none on HEAD falls back to full GET buffer mode
  const noRangeMock = createRangeMockFetch(
    { '/packs/large.wgsl.pack': largePackBuf },
    { acceptRangesHeader: 'none' }
  );
  const noRangePack = await loadFromUrl('/packs/large.wgsl.pack', {
    fetch: noRangeMock.fetch,
    streamThresholdBytes: 1024,
  });
  assert.equal(noRangePack.mode, 'buffer');
  assert.equal(noRangePack.getWgsl(hLarge1), largeWgsl1);
});

test('W109 WgslPackCache integrates ranged packs with getWgslAsync, GPUShaderModule async compile, prewarm, dynamic overrides, and stats', async () => {
  const hRangedVs = fnv1a64('cache-ranged-vs');
  const hRangedFs = fnv1a64('cache-ranged-fs');
  const hBufCs = fnv1a64('cache-buf-cs');

  const rangedBuf = buildWgslPackBuffer([
    { hash: hRangedVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hRangedFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  const bufPack = buildWgslPackBuffer([
    { hash: hBufCs, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
  ]);
  const rangedTableEnd = WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES;

  const mock = createRangeMockFetch({ '/packs/world.wgsl.pack': rangedBuf });
  const cache = new WgslPackCache({
    fetch: mock.fetch,
    maxCachedWgslBytes: 4096,
    initialRangeBytes: rangedTableEnd,
  });

  const rangedPack = await cache.loadIndexFromUrl('/packs/world.wgsl.pack');
  assert.equal(rangedPack.mode, 'ranged');
  assert.equal(cache.has(hRangedVs), true);
  assert.equal(cache.hasShader(hRangedFs, 'fragment'), true);
  assert.equal(cache.getStats().mode, 'ranged');
  assert.equal(cache.getStats().rangeFetches, 1);

  // Load a second full-buffer pack via loadFromBuffer -> mixed mode telemetry
  cache.loadFromBuffer(bufPack, 'boot.wgsl.pack');
  assert.equal(cache.getStats().mode, 'mixed');

  // getWgslAsync works uniformly across ranged pack, buffer pack, stage mismatch, and missing hashes
  assert.equal(await cache.getWgslAsync(hBufCs), SAMPLE_CS_WGSL);
  assert.equal(await cache.getWgslAsync(hRangedVs, 'fragment'), null);
  assert.equal(await cache.getWgslAsync(hRangedVs, 'vertex'), SAMPLE_VS_WGSL);
  assert.equal(await cache.getWgslAsync(0xbeefn), null);
  assert.equal(cache.getStats().rangeFetches, 2);
  assert.equal(cache.getStats().cacheMisses, 1);
  assert.equal(cache.getStats().cacheHits, 1); // buffer pack async hit

  // Dynamic override supersedes ranged static shader with zero HTTP fetches
  const hotFs = '@fragment fn fs_hot() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }';
  cache.registerDynamicShader(hRangedFs, 'fragment', hotFs, { override: true, label: 'live' });
  assert.equal(await cache.getWgslAsync(hRangedFs), hotFs);
  assert.equal(cache.getStats().rangeFetches, 2); // no fetch for overridden ranged shader!

  // Unregister override -> fetches underlying ranged static shader on demand
  cache.unregisterDynamicShader(hRangedFs);
  const gpu = createMockGpuDevice();
  const modFs = await cache.getOrCreateShaderModuleAsync(gpu.device, hRangedFs, 'gta6');
  assert.equal(modFs.code, SAMPLE_FS_WGSL);
  assert.equal(cache.getStats().rangeFetches, 3);

  // Second getOrCreateShaderModuleAsync hits device module cache with zero extra fetches
  const modFs2 = await cache.getOrCreateShaderModuleAsync(gpu.device, hRangedFs, 'gta6');
  assert.equal(modFs2, modFs);
  assert.equal(cache.getStats().rangeFetches, 3);

  // prewarm() works seamlessly across ranged + buffer packs
  const warm = await cache.prewarm(gpu.device, [hRangedVs, hRangedFs, hBufCs]);
  assert.equal(warm.total, 3);
  assert.equal(warm.cached, 1);
  assert.equal(warm.compiled, 2);
  assert.equal(warm.modules.get(hRangedVs).code, SAMPLE_VS_WGSL);
  assert.equal(warm.modules.get(hBufCs).code, SAMPLE_CS_WGSL);

  // loadFromUrl on cache with streamThresholdBytes
  const thresholdPack = await cache.loadFromUrl('/packs/world.wgsl.pack', {
    packName: 'world-auto',
    streamThresholdBytes: 64,
    initialRangeBytes: rangedTableEnd,
  });
  assert.equal(thresholdPack.mode, 'ranged');

  // Module-level loadFromBuffer matches parseWgslPack
  const directBuf = loadFromBuffer(bufPack);
  assert.equal(directBuf.mode, 'buffer');
  assert.equal(directBuf.getWgsl(hBufCs), SAMPLE_CS_WGSL);
});

test('W109 ranged index parser & fetcher validate corrupt headers, bounds, UTF-8, and options', async () => {
  const h1 = fnv1a64('val-1');
  const h2 = fnv1a64('val-2');
  const validBuf = buildWgslPackBuffer([
    { hash: h1, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: h2, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  const validBytes = new Uint8Array(validBuf);
  const tableEnd = WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES;

  // Invalid option ranges
  const mockOk = createRangeMockFetch({ '/ok.wgsl.pack': validBuf });
  await assert.rejects(
    () => loadIndexFromUrl('/ok.wgsl.pack', { fetch: mockOk.fetch, maxCachedWgslBytes: -1 }),
    RangeError
  );
  await assert.rejects(
    () => loadIndexFromUrl('/ok.wgsl.pack', { fetch: mockOk.fetch, initialRangeBytes: 8 }),
    RangeError
  );
  await assert.rejects(
    () => loadFromUrl('/ok.wgsl.pack', { fetch: mockOk.fetch, streamThresholdBytes: -10 }),
    RangeError
  );

  // HTTP 404
  await assert.rejects(
    () => loadIndexFromUrl('/missing.wgsl.pack', { fetch: mockOk.fetch }),
    /HTTP 404/
  );

  // Bad magic via Range
  const badMagic = validBytes.slice();
  badMagic.set(new TextEncoder().encode('BAD!'), 0);
  const badMagicMock = createRangeMockFetch({ '/bad.wgsl.pack': badMagic });
  await assert.rejects(
    () => loadIndexFromUrl('/bad.wgsl.pack', { fetch: badMagicMock.fetch }),
    /invalid wgpk pack magic/i
  );

  // Out-of-bounds entry against totalBytes reported by Content-Range
  const oobEntry = validBytes.slice();
  new DataView(oobEntry.buffer).setUint32(WGPK_HEADER_BYTES + 16, 999999, true);
  const oobMock = createRangeMockFetch({ '/oob.wgsl.pack': oobEntry });
  await assert.rejects(
    () => loadIndexFromUrl('/oob.wgsl.pack', { fetch: oobMock.fetch, initialRangeBytes: tableEnd }),
    /out of bounds/i
  );

  // Invalid UTF-8 payload fetched on demand rejects without caching
  const badUtf8 = validBytes.slice();
  badUtf8[tableEnd] = 0xff;
  const badUtf8Mock = createRangeMockFetch({ '/bad-utf8.wgsl.pack': badUtf8 });
  const badUtf8Pack = await loadIndexFromUrl('/bad-utf8.wgsl.pack', {
    fetch: badUtf8Mock.fetch,
    initialRangeBytes: tableEnd,
  });
  await assert.rejects(() => badUtf8Pack.getWgslAsync(h1), /not valid utf-8/i);
  assert.equal(badUtf8Pack.stats().cachedWgslBytes, 0);
  // Valid sibling entry still loads fine!
  assert.equal(await badUtf8Pack.getWgslAsync(h2), SAMPLE_FS_WGSL);

  // parseWgslPackIndex works directly on an index-only slice (no payload bytes present)
  const indexOnlySlice = validBytes.slice(0, tableEnd);
  const directIndexPack = parseWgslPackIndex(indexOnlySlice, {
    url: '/ok.wgsl.pack',
    totalBytes: validBytes.byteLength,
    fetch: mockOk.fetch,
  });
  assert.equal(directIndexPack.mode, 'ranged');
  assert.equal(await directIndexPack.getWgslAsync(h1), SAMPLE_VS_WGSL);
});

// ---------------------------------------------------------------------------
// W160: Large (1.83 GiB – 4 GiB, 18,504+ shader) ranged packs, unsigned u32
//       offsets, in-flight fetch coalescing, and ranged cache telemetry (#854)
// ---------------------------------------------------------------------------

function buildSyntheticLargeIndexBuffer(entries, options = {}) {
  const flags = options.flags === undefined ? 0 : options.flags >>> 0;
  const tableEnd = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES;
  const buf = new ArrayBuffer(tableEnd);
  const bytes = new Uint8Array(buf);
  const view = new DataView(buf);

  bytes[0] = 0x57; // W
  bytes[1] = 0x47; // G
  bytes[2] = 0x50; // P
  bytes[3] = 0x4b; // K
  view.setUint32(4, WGPK_VERSION, true);
  view.setUint32(8, entries.length >>> 0, true);
  view.setUint32(12, flags, true);

  for (let i = 0; i < entries.length; i += 1) {
    const item = entries[i];
    const base = WGPK_HEADER_BYTES + i * WGPK_ENTRY_BYTES;
    view.setBigUint64(base, normalizeHash(item.hash), true);
    bytes[base + 8] = item.stage;
    bytes[base + 9] = 0;
    bytes[base + 10] = 0;
    bytes[base + 11] = 0;
    view.setUint32(base + 12, item.offset >>> 0, true);
    view.setUint32(base + 16, item.length >>> 0, true);
  }

  return bytes;
}

function createSparseRangeMockFetch({
  url: targetUrl,
  indexBytes,
  totalBytes,
  payloadsByOffset,
  omitTotalInContentRange = false,
  beforeResponse = null,
}) {
  const requests = [];
  const enc = new TextEncoder();
  const payloadMap = new Map();
  for (const [offset, wgsl] of payloadsByOffset.entries()) {
    const bytes = typeof wgsl === 'string' ? enc.encode(wgsl) : wgsl;
    payloadMap.set(offset, bytes);
  }

  const readHeader = (headers, name) => {
    if (!headers) return null;
    if (typeof headers.get === 'function') return headers.get(name);
    const target = name.toLowerCase();
    for (const [k, v] of Object.entries(headers)) {
      if (k.toLowerCase() === target) return String(v);
    }
    return null;
  };

  const mockFetch = async (url, init = {}) => {
    const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
    const range = readHeader(init && init.headers, 'Range');

    if (beforeResponse) {
      const override = await beforeResponse({ url, method, range, init, requests });
      if (override) {
        requests.push({ url, method, range, status: override.status, start: null, end: null, bytesSent: 0 });
        return override;
      }
    }

    if (url !== targetUrl) {
      requests.push({ url, method, range, status: 404, start: null, end: null, bytesSent: 0 });
      return {
        ok: false,
        status: 404,
        statusText: 'Not Found',
        headers: new Headers(),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (method === 'HEAD') {
      requests.push({ url, method, range: null, status: 200, start: null, end: null, bytesSent: 0 });
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        headers: new Headers({
          'Content-Length': String(totalBytes),
          'Accept-Ranges': 'bytes',
        }),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    assert.ok(range, `expected Range header for sparse large pack request to ${url}`);
    const match = /^bytes=(\d+)-(\d+)$/i.exec(range.trim());
    assert.ok(match, `expected valid non-negative Range header, got: ${range}`);
    const start = Number(match[1]);
    const reqEnd = Number(match[2]);
    assert.ok(Number.isSafeInteger(start) && start >= 0, `invalid start offset: ${start}`);
    assert.ok(Number.isSafeInteger(reqEnd) && reqEnd >= start, `invalid end offset: ${reqEnd}`);
    assert.ok(start < totalBytes, `start ${start} >= totalBytes ${totalBytes}`);

    const end = Math.min(reqEnd, totalBytes - 1);
    const len = end - start + 1;
    let slice;

    if (end < indexBytes.byteLength) {
      slice = indexBytes.slice(start, end + 1);
    } else if (start < indexBytes.byteLength && end >= indexBytes.byteLength) {
      slice = new Uint8Array(len);
      const copyLen = indexBytes.byteLength - start;
      slice.set(indexBytes.subarray(start, indexBytes.byteLength), 0);
      // Remaining bytes beyond indexBytes are zero-filled padding unless needed.
      assert.ok(copyLen > 0);
    } else {
      const payload = payloadMap.get(start);
      assert.ok(payload, `unexpected payload range start=${start} (end=${end})`);
      assert.equal(payload.byteLength, len, `payload length mismatch at offset ${start}`);
      slice = payload.slice();
    }

    const totalTag = omitTotalInContentRange ? '*' : String(totalBytes);
    requests.push({ url, method, range, status: 206, start, end, bytesSent: slice.byteLength });
    return {
      ok: true,
      status: 206,
      statusText: 'Partial Content',
      headers: new Headers({
        'Content-Range': `bytes ${start}-${end}/${totalTag}`,
        'Content-Length': String(slice.byteLength),
        'Accept-Ranges': 'bytes',
      }),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  };

  return { fetch: mockFetch, requests };
}

test('W160 ranged TOC parsing and HTTP 206 Range fetches handle 1.83–4 GiB offsets (> 0x80000000 up to 0xFFFFFFFF) without signed 32-bit wraparound', async () => {
  const enc = new TextEncoder();
  const TOTAL_4_GIB = 0x100000000; // 4,294,967,296 bytes (exact 4 GiB)
  const ENTRY_COUNT = 18504;
  const TABLE_END = WGPK_HEADER_BYTES + ENTRY_COUNT * WGPK_ENTRY_BYTES; // 370,096 bytes

  const makePaddedWgsl = (tag, targetBytes) => {
    const prefix = `@compute @workgroup_size(1) fn fn_${tag}() { /* `;
    const suffix = ' */ }';
    const padLen = targetBytes - enc.encode(prefix).byteLength - enc.encode(suffix).byteLength;
    assert.ok(padLen >= 0);
    return `${prefix}${'z'.repeat(padLen)}${suffix}`;
  };

  const wgslLow = makePaddedWgsl('low_0', 128);
  const wgsl183Gib = makePaddedWgsl('at_1_83_gib', 192);
  const wgslCrossing2Gib = makePaddedWgsl('cross_2_gib', 64);
  const wgslExact2Gib = makePaddedWgsl('exact_2_gib', 256);
  const wgslOver3Gib = `// 3 GiB+ UTF-8 🚀 日本語\n${SAMPLE_FS_WGSL}`;
  const wgslCeiling4Gib = makePaddedWgsl('ceiling_4_gib', 256);

  assert.equal(enc.encode(wgslCeiling4Gib).byteLength, 256);

  // Place 18,504 non-overlapping entries spanning from TABLE_END all the way to the exact 4 GiB ceiling.
  const entries = new Array(ENTRY_COUNT);
  const payloadsByOffset = new Map();

  // First 18,498 entries packed densely right after TABLE_END (each 64 bytes -> ~1.18 MiB).
  let cursor = TABLE_END;
  for (let i = 0; i < ENTRY_COUNT - 6; i += 1) {
    const len = i === 0 ? enc.encode(wgslLow).byteLength : 64;
    entries[i] = {
      hash: BigInt(i + 1),
      stage: (i % 3) + 1,
      offset: cursor,
      length: len,
    };
    if (i === 0) {
      payloadsByOffset.set(cursor, wgslLow);
    }
    cursor += len;
  }

  const h183Gib = fnv1a64('w160-1.83gib');
  const hCross2Gib = fnv1a64('w160-cross-2gib');
  const hExact2Gib = fnv1a64('w160-exact-2gib');
  const hOver3Gib = fnv1a64('w160-over-3gib');
  const hCeiling4Gib = fnv1a64('w160-ceiling-4gib');
  const hZeroAtHigh = fnv1a64('w160-zero-high');

  const off183Gib = 0x75000000; // ~1.83 GiB
  const offCross2Gib = 0x7ffffff0; // crosses 0x80000000 (+64 -> 0x80000030)
  const offExact2Gib = 0x80000100; // strictly > 2 GiB (signed i32 would be negative!)
  const offOver3Gib = 0xc0001234; // > 3 GiB
  const offZeroHigh = 0xe0000000; // 0-length entry past 3.5 GiB
  const offCeiling4Gib = 0xffffff00; // 4,294,967,040 + 256 == 4,294,967,296 (exact 4 GiB)

  const highEntries = [
    { hash: h183Gib, stage: SHADER_STAGE.VERTEX, offset: off183Gib, length: enc.encode(wgsl183Gib).byteLength },
    { hash: hCross2Gib, stage: SHADER_STAGE.FRAGMENT, offset: offCross2Gib, length: enc.encode(wgslCrossing2Gib).byteLength },
    { hash: hExact2Gib, stage: SHADER_STAGE.COMPUTE, offset: offExact2Gib, length: enc.encode(wgslExact2Gib).byteLength },
    { hash: hOver3Gib, stage: SHADER_STAGE.FRAGMENT, offset: offOver3Gib, length: enc.encode(wgslOver3Gib).byteLength },
    { hash: hZeroAtHigh, stage: SHADER_STAGE.COMPUTE, offset: offZeroHigh, length: 0 },
    { hash: hCeiling4Gib, stage: SHADER_STAGE.VERTEX, offset: offCeiling4Gib, length: enc.encode(wgslCeiling4Gib).byteLength },
  ];
  for (let i = 0; i < highEntries.length; i += 1) {
    entries[ENTRY_COUNT - 6 + i] = highEntries[i];
  }

  payloadsByOffset.set(off183Gib, wgsl183Gib);
  payloadsByOffset.set(offCross2Gib, wgslCrossing2Gib);
  payloadsByOffset.set(offExact2Gib, wgslExact2Gib);
  payloadsByOffset.set(offOver3Gib, wgslOver3Gib);
  payloadsByOffset.set(offCeiling4Gib, wgslCeiling4Gib);

  const indexBytes = buildSyntheticLargeIndexBuffer(entries, { flags: 0x80000001 });
  const mock = createSparseRangeMockFetch({
    url: '/packs/full-4gib.wgsl.pack',
    indexBytes,
    totalBytes: TOTAL_4_GIB,
    payloadsByOffset,
  });

  // Default 64 KiB initial window requires 2 range requests for the 18,504-entry TOC (370,096 bytes)
  const pack = await loadIndexFromUrl('/packs/full-4gib.wgsl.pack', { fetch: mock.fetch });
  assert.equal(pack.mode, 'ranged');
  assert.equal(pack.size, ENTRY_COUNT);
  assert.equal(pack.flags, 0x80000001);
  assert.equal(pack.byteLength, TOTAL_4_GIB);
  assert.equal(pack.dataStart, TABLE_END);
  assert.equal(mock.requests.length, 2);
  assert.equal(mock.requests[0].range, `bytes=0-${DEFAULT_INITIAL_RANGE_BYTES - 1}`);
  assert.equal(mock.requests[1].range, `bytes=${DEFAULT_INITIAL_RANGE_BYTES}-${TABLE_END - 1}`);

  let stats = pack.getStats();
  assert.equal(stats.rangeFetchCount, 2);
  assert.equal(stats.rangeFetches, 2);
  assert.equal(stats.rangeBytesFetched, TABLE_END);
  assert.equal(stats.inFlightCoalesced, 0);
  assert.equal(stats.lruEvictions, 0);

  // Verify every >2 GiB offset is stored as an unsigned positive JS number and fetches the exact unsigned byte range.
  const fetched183 = await pack.getWgslAsync(h183Gib);
  assert.equal(fetched183, wgsl183Gib);
  assert.equal(mock.requests[2].range, `bytes=${off183Gib}-${off183Gib + 192 - 1}`);

  const fetchedCross = await pack.getWgslAsync(hCross2Gib);
  assert.equal(fetchedCross, wgslCrossing2Gib);
  assert.equal(mock.requests[3].range, `bytes=${offCross2Gib}-${offCross2Gib + 64 - 1}`);
  assert.equal(mock.requests[3].range, 'bytes=2147483632-2147483695');

  const fetched2Gib = await pack.getWgslAsync(hExact2Gib);
  assert.equal(fetched2Gib, wgslExact2Gib);
  assert.equal(mock.requests[4].range, 'bytes=2147483904-2147484159');

  const fetched3Gib = await pack.getWgslAsync(hOver3Gib);
  assert.equal(fetched3Gib, wgslOver3Gib);
  const over3Len = enc.encode(wgslOver3Gib).byteLength;
  assert.equal(mock.requests[5].range, `bytes=${offOver3Gib}-${offOver3Gib + over3Len - 1}`);

  const fetchedCeiling = await pack.getWgslAsync(hCeiling4Gib);
  assert.equal(fetchedCeiling, wgslCeiling4Gib);
  assert.equal(mock.requests[6].range, 'bytes=4294967040-4294967295');

  // Zero-length entry past 3.5 GiB resolves immediately without firing an HTTP request
  assert.equal(await pack.getWgslAsync(hZeroAtHigh), '');
  assert.equal(mock.requests.length, 7);

  stats = pack.getStats();
  assert.equal(stats.rangeFetchCount, 7);
  assert.equal(stats.rangeBytesFetched, TABLE_END + 192 + 64 + 256 + over3Len + 256);

  // loadFromUrl with streamThresholdBytes auto-switches to ranged mode when HEAD Content-Length is 4 GiB
  const autoMock = createSparseRangeMockFetch({
    url: '/packs/full-4gib.wgsl.pack',
    indexBytes,
    totalBytes: TOTAL_4_GIB,
    payloadsByOffset,
  });
  const autoPack = await loadFromUrl('/packs/full-4gib.wgsl.pack', {
    fetch: autoMock.fetch,
    streamThresholdBytes: 1024 * 1024,
  });
  assert.equal(autoPack.mode, 'ranged');
  assert.equal(autoMock.requests[0].method, 'HEAD');
  assert.equal(await autoPack.getWgslAsync(hCeiling4Gib), wgslCeiling4Gib);

  // Unknown total size (Content-Range: bytes 0-N/*) still accepts exact 4 GiB ceiling (end == 0x100000000)
  const smallHighIndex = buildSyntheticLargeIndexBuffer([
    { hash: hExact2Gib, stage: SHADER_STAGE.COMPUTE, offset: offExact2Gib, length: 256 },
    { hash: hCeiling4Gib, stage: SHADER_STAGE.VERTEX, offset: offCeiling4Gib, length: 256 },
  ]);
  const starMock = createSparseRangeMockFetch({
    url: '/packs/star-4gib.wgsl.pack',
    indexBytes: smallHighIndex,
    totalBytes: TOTAL_4_GIB,
    payloadsByOffset,
    omitTotalInContentRange: true,
  });
  const starPack = await loadIndexFromUrl('/packs/star-4gib.wgsl.pack', {
    fetch: starMock.fetch,
    initialRangeBytes: smallHighIndex.byteLength,
  });
  assert.equal(starPack.byteLength, TOTAL_4_GIB);
  assert.equal(await starPack.getWgslAsync(hCeiling4Gib), wgslCeiling4Gib);

  // Corrupt entry overflowing past 4 GiB when totalBytes is unknown is still rejected
  const overflowIndex = buildSyntheticLargeIndexBuffer([
    { hash: hCeiling4Gib, stage: SHADER_STAGE.VERTEX, offset: 0xffffffff, length: 16 },
  ]);
  assert.throws(() => parseWgslPackIndex(overflowIndex), /out of bounds/i);

  // Overlapping ranges above 2 GiB are still detected accurately
  const overlapHighIndex = buildSyntheticLargeIndexBuffer([
    { hash: hExact2Gib, stage: SHADER_STAGE.COMPUTE, offset: 0x80000000, length: 128 },
    { hash: hOver3Gib, stage: SHADER_STAGE.FRAGMENT, offset: 0x80000040, length: 128 },
  ]);
  assert.throws(
    () => parseWgslPackIndex(overlapHighIndex, { totalBytes: TOTAL_4_GIB }),
    /overlapping payload/i
  );
});

test('W160 concurrent getWgslAsync coalesces burst HTTP 206 Range fetches and cleans up inFlightFetches on both resolve and reject', async () => {
  const enc = new TextEncoder();
  const TOTAL_3_GIB = 0xc0000000;
  const hResolve = fnv1a64('w160-coalesce-resolve');
  const hReject = fnv1a64('w160-coalesce-reject');
  const hSibling = fnv1a64('w160-coalesce-sibling');

  const offResolve = 0x80002000;
  const offReject = 0x90004000;
  const offSibling = 0xa0006000;

  const wgslResolve = `// coalesced >2GiB shader\n${SAMPLE_VS_WGSL}`;
  const wgslReject = `// retried >2GiB shader\n${SAMPLE_FS_WGSL}`;
  const wgslSibling = `// parallel >2GiB sibling\n${SAMPLE_CS_WGSL}`;

  const resolveLen = enc.encode(wgslResolve).byteLength;
  const rejectLen = enc.encode(wgslReject).byteLength;
  const siblingLen = enc.encode(wgslSibling).byteLength;

  const indexBytes = buildSyntheticLargeIndexBuffer([
    { hash: hResolve, stage: SHADER_STAGE.VERTEX, offset: offResolve, length: resolveLen },
    { hash: hReject, stage: SHADER_STAGE.FRAGMENT, offset: offReject, length: rejectLen },
    { hash: hSibling, stage: SHADER_STAGE.COMPUTE, offset: offSibling, length: siblingLen },
  ]);

  const gates = new Map();
  let failRejectRangeOnce = true;

  const mock = createSparseRangeMockFetch({
    url: '/packs/coalesce.wgsl.pack',
    indexBytes,
    totalBytes: TOTAL_3_GIB,
    payloadsByOffset: new Map([
      [offResolve, wgslResolve],
      [offReject, wgslReject],
      [offSibling, wgslSibling],
    ]),
    beforeResponse: async ({ range }) => {
      if (!range || range.startsWith('bytes=0-')) {
        return null;
      }
      if (range.startsWith(`bytes=${offResolve}-`) || range.startsWith(`bytes=${offSibling}-`)) {
        await new Promise((resolve) => {
          gates.set(range, resolve);
        });
        return null;
      }
      if (range.startsWith(`bytes=${offReject}-`) && failRejectRangeOnce) {
        await new Promise((resolve) => {
          gates.set(range, resolve);
        });
        failRejectRangeOnce = false;
        return {
          ok: false,
          status: 503,
          statusText: 'Service Unavailable',
          headers: new Headers(),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      return null;
    },
  });

  const pack = await loadIndexFromUrl('/packs/coalesce.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: indexBytes.byteLength,
    maxRangeFetchRetries: 0,
  });
  assert.equal(pack.getStats().inFlightCount, 0);
  assert.equal(pack.getStats().rangeFetchCount, 1);
  assert.equal(pack.getStats().rangeBytesFetched, indexBytes.byteLength);

  // 1. Burst of 8 concurrent calls for hResolve + 2 concurrent calls for hSibling
  const resolvePromises = [
    pack.getWgslAsync(hResolve),
    pack.getWgslAsync(formatHashHex(hResolve)),
    pack.getWgslAsync(formatHashHex(hResolve).slice(2).toUpperCase()),
    pack.getWgslAsync(hResolve),
    pack.getWgslAsync(hResolve),
    pack.getWgslAsync(formatHashHex(hResolve)),
    pack.getWgslAsync(hResolve),
    pack.getWgslAsync(hResolve),
  ];
  const siblingPromises = [
    pack.getWgslAsync(hSibling),
    pack.getWgslAsync(formatHashHex(hSibling)),
  ];

  await new Promise((r) => setTimeout(r, 10));
  assert.equal(gates.size, 2, 'distinct hashes should run concurrently while duplicates coalesce');
  assert.equal(pack.getStats().inFlightCount, 2);
  assert.equal(pack.getStats().inFlightCoalesced, 8); // 7 for hResolve + 1 for hSibling
  assert.equal(pack.getStats().inFlightDeduplications, 8);

  for (const release of gates.values()) {
    release();
  }
  gates.clear();

  const resolvedValues = await Promise.all(resolvePromises);
  const siblingValues = await Promise.all(siblingPromises);
  assert.ok(resolvedValues.every((v) => v === wgslResolve));
  assert.ok(siblingValues.every((v) => v === wgslSibling));

  // Verify cleanup and telemetry after resolve
  let stats = pack.getStats();
  assert.equal(stats.inFlightCount, 0);
  assert.equal(stats.rangeFetchCount, 3); // 1 index + 1 hResolve + 1 hSibling
  assert.equal(stats.rangeBytesFetched, indexBytes.byteLength + resolveLen + siblingLen);
  assert.equal(stats.inFlightCoalesced, 8);

  // Subsequent lookup hits LRU cache with zero extra network calls
  assert.equal(await pack.getWgslAsync(hResolve), wgslResolve);
  assert.equal(pack.getStats().rangeFetchCount, 3);
  assert.equal(pack.getStats().cacheHits, 1);

  // 2. Burst of 4 concurrent calls for hReject when the single HTTP 206 fetch fails with 503
  let retryResultFromCatch = null;
  let inFlightObservedInCatch = -1;

  const pFail1 = pack.getWgslAsync(hReject).catch(async (err) => {
    assert.match(err.message, /HTTP 503/);
    // Must already be cleaned up in finally BEFORE catch handler runs so immediate retry fires cleanly!
    inFlightObservedInCatch = pack.getStats().inFlightCount;
    retryResultFromCatch = await pack.getWgslAsync(hReject);
    return 'caught-and-retried';
  });
  const pFail2 = pack.getWgslAsync(formatHashHex(hReject));
  const pFail3 = pack.getWgslAsync(hReject);
  const pFail4 = pack.getWgslAsync(hReject);

  await new Promise((r) => setTimeout(r, 10));
  assert.equal(gates.size, 1);
  assert.equal(pack.getStats().inFlightCount, 1);
  assert.equal(pack.getStats().inFlightCoalesced, 11); // +3 coalesced onto the failing request

  for (const release of gates.values()) {
    release();
  }
  gates.clear();

  const [s1, s2, s3, s4] = await Promise.allSettled([pFail1, pFail2, pFail3, pFail4]);
  assert.equal(s1.status, 'fulfilled');
  assert.equal(s1.value, 'caught-and-retried');
  assert.equal(inFlightObservedInCatch, 0, 'inFlightFetches must be cleared before caller rejection handlers run');
  assert.equal(retryResultFromCatch, wgslReject);
  assert.equal(s2.status, 'rejected');
  assert.equal(s3.status, 'rejected');
  assert.equal(s4.status, 'rejected');

  stats = pack.getStats();
  assert.equal(stats.inFlightCount, 0);
  assert.equal(stats.rangeFetchCount, 5); // 3 prior + 1 failed 503 + 1 successful retry
  assert.equal(stats.rangeBytesFetched, indexBytes.byteLength + resolveLen + siblingLen + rejectLen);
  assert.equal(stats.inFlightCoalesced, 11);
});

test('W160 ranged cache telemetry (inFlightCoalesced, rangeFetchCount, rangeBytesFetched, lruEvictions) aggregates cleanly across WgslPackCache', async () => {
  const enc = new TextEncoder();
  const TOTAL_4_GIB = 0x100000000;
  const make100ByteWgsl = (tag) => {
    const prefix = `@compute @workgroup_size(1) fn w160_${tag}() { /* `;
    const suffix = ' */ }';
    const padLen = 100 - prefix.length - suffix.length;
    return `${prefix}${'k'.repeat(padLen)}${suffix}`;
  };

  const hA = fnv1a64('w160-telem-A');
  const hB = fnv1a64('w160-telem-B');
  const hC = fnv1a64('w160-telem-C');
  const wgslA = make100ByteWgsl('A');
  const wgslB = make100ByteWgsl('B');
  const wgslC = make100ByteWgsl('C');
  assert.equal(enc.encode(wgslA).byteLength, 100);

  const offA = 0x80000000;
  const offB = 0xa0000000;
  const offC = 0xffffff9c; // 0xFFFFFF9C + 100 == 0x100000000 (exact 4 GiB)

  const indexBytes = buildSyntheticLargeIndexBuffer([
    { hash: hA, stage: SHADER_STAGE.COMPUTE, offset: offA, length: 100 },
    { hash: hB, stage: SHADER_STAGE.COMPUTE, offset: offB, length: 100 },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, offset: offC, length: 100 },
  ]);

  let releaseA = null;
  let holdA = true;
  const mock = createSparseRangeMockFetch({
    url: '/packs/telemetry-4gib.wgsl.pack',
    indexBytes,
    totalBytes: TOTAL_4_GIB,
    payloadsByOffset: new Map([
      [offA, wgslA],
      [offB, wgslB],
      [offC, wgslC],
    ]),
    beforeResponse: async ({ range }) => {
      if (holdA && range === `bytes=${offA}-${offA + 99}`) {
        holdA = false;
        await new Promise((resolve) => {
          releaseA = resolve;
        });
      }
      return null;
    },
  });

  // LRU budget = 220 bytes (fits 2 x 100-byte shaders; 3rd shader triggers 1 LRU eviction)
  const cache = new WgslPackCache({
    fetch: mock.fetch,
    initialRangeBytes: indexBytes.byteLength,
    maxCachedWgslBytes: 220,
  });

  const pack = await cache.loadIndexFromUrl('/packs/telemetry-4gib.wgsl.pack');
  const gpu = createMockGpuDevice();

  // Burst 3 concurrent getOrCreateShaderModuleAsync calls for hA (coalesced into 1 range fetch & 1 shader compile)
  const m1 = cache.getOrCreateShaderModuleAsync(gpu.device, hA);
  const m2 = cache.getOrCreateShaderModuleAsync(gpu.device, formatHashHex(hA));
  const m3 = cache.getWgslAsync(hA);

  await new Promise((r) => setTimeout(r, 10));
  assert.equal(typeof releaseA, 'function');
  assert.equal(cache.getStats().inFlightCount, 1);
  assert.equal(cache.getStats().inFlightCoalesced, 2);
  releaseA();

  const [mod1, mod2, src3] = await Promise.all([m1, m2, m3]);
  assert.equal(mod1, mod2);
  assert.equal(mod1.code, wgslA);
  assert.equal(src3, wgslA);
  assert.equal(gpu.calls.length, 1);

  // Fetch B (fits within 220 B budget) then C (evicts A -> lruEvictions === 1)
  assert.equal(await cache.getWgslAsync(hB), wgslB);
  assert.equal(await cache.getWgslAsync(hC), wgslC);

  const packStats = pack.getStats();
  const cacheStats = cache.getStats();

  for (const s of [packStats, cacheStats]) {
    assert.equal(s.rangeFetchCount, 4); // 1 index + A + B + C
    assert.equal(s.rangeFetches, 4);
    assert.equal(s.rangeBytesFetched, indexBytes.byteLength + 300);
    assert.equal(s.inFlightCoalesced, 2);
    assert.equal(s.inFlightDeduplications, 2);
    assert.equal(s.inFlightCount, 0);
    assert.equal(s.lruEvictions, 1);
    assert.equal(s.wgslCacheEvictions, 1);
    for (const key of [
      'inFlightCoalesced',
      'rangeFetchCount',
      'rangeBytesFetched',
      'lruEvictions',
      'inFlightCount',
    ]) {
      assert.equal(typeof s[key], 'number');
      assert.ok(Number.isSafeInteger(s[key]) && s[key] >= 0, `${key} must be a non-negative safe integer`);
    }
  }

  // inspect() remains fully JSON-serializable with all additive ranged telemetry fields
  const snapshot = cache.inspect();
  assert.doesNotThrow(() => JSON.stringify(snapshot));
  assert.equal(snapshot.stats.rangeFetchCount, 4);
  assert.equal(snapshot.stats.rangeBytesFetched, indexBytes.byteLength + 300);
  assert.equal(snapshot.stats.inFlightCoalesced, 2);
  assert.equal(snapshot.stats.lruEvictions, 1);
});

// ---------------------------------------------------------------------------
// W162: Bounded HTTP 206 Range concurrency semaphore & transient retry (#864)
// ---------------------------------------------------------------------------

test('W162 bounded FIFO semaphore caps concurrent HTTP 206 Range fetches, preserves order, coalesces duplicates before acquiring slots, and supports unlimited (0)', async () => {
  assert.equal(DEFAULT_MAX_CONCURRENT_RANGE_FETCHES, 8);
  assert.equal(DEFAULT_MAX_RANGE_FETCH_RETRIES, 2);
  assert.equal(DEFAULT_RANGE_RETRY_DELAY_MS, 25);

  const count = 20;
  const entries = [];
  for (let i = 0; i < count; i += 1) {
    entries.push({
      hash: fnv1a64(`w162-sem-shader-${i}`),
      stage: SHADER_STAGE.COMPUTE,
      wgsl: `@compute @workgroup_size(${i + 1}) fn sem_${i}() {}`,
    });
  }
  const packBuf = buildWgslPackBuffer(entries);
  const tableEnd = WGPK_HEADER_BYTES + count * WGPK_ENTRY_BYTES;

  // 1. Validate option bounds up front
  const validateMock = createRangeMockFetch({ '/packs/sem.wgsl.pack': packBuf });
  await assert.rejects(
    () => loadIndexFromUrl('/packs/sem.wgsl.pack', { fetch: validateMock.fetch, maxConcurrentRangeFetches: -1 }),
    RangeError
  );
  await assert.rejects(
    () => loadIndexFromUrl('/packs/sem.wgsl.pack', { fetch: validateMock.fetch, maxRangeFetchRetries: -1 }),
    RangeError
  );
  await assert.rejects(
    () => loadIndexFromUrl('/packs/sem.wgsl.pack', { fetch: validateMock.fetch, rangeRetryDelayMs: -5 }),
    RangeError
  );
  await assert.rejects(
    () => loadIndexFromUrl('/packs/sem.wgsl.pack', { fetch: validateMock.fetch, sleepFn: 'not-a-fn' }),
    TypeError
  );

  // 2. Cap = 4 with 20 distinct hashes + duplicate burst -> never exceeds 4 active fetches, strict FIFO order
  let liveConcurrent = 0;
  let observedPeak = 0;
  const startedOrder = [];
  const pendingReleases = [];

  const capMock = createRangeMockFetch(
    { '/packs/sem.wgsl.pack': packBuf },
    {
      beforeResponse: async ({ range }) => {
        if (!range || range.startsWith('bytes=0-')) {
          return null;
        }
        liveConcurrent += 1;
        if (liveConcurrent > observedPeak) {
          observedPeak = liveConcurrent;
        }
        startedOrder.push(range);
        await new Promise((resolve) => {
          pendingReleases.push(() => {
            liveConcurrent -= 1;
            resolve();
          });
        });
        return null;
      },
    }
  );

  const pack = await loadIndexFromUrl('/packs/sem.wgsl.pack', {
    fetch: capMock.fetch,
    initialRangeBytes: tableEnd,
    maxConcurrentRangeFetches: 4,
  });
  assert.equal(pack.getStats().maxConcurrentRangeFetches, 4);
  assert.equal(pack.getStats().activeRangeFetches, 0);
  assert.equal(pack.getStats().peakConcurrentRangeFetches, 0);

  const expectedRangeByIdx = entries.map((e) => {
    const meta = pack.entries.get(e.hash);
    return `bytes=${meta.offset}-${meta.offset + meta.length - 1}`;
  });

  // Launch 20 distinct fetches plus 3 duplicate calls for entry 0 and 2 duplicate calls for entry 10
  // (entry 10 is queued in the FIFO wait queue, so duplicates must coalesce without taking extra slots!).
  const promises = entries.map((e) => pack.getWgslAsync(e.hash));
  const dupPromises = [
    pack.getWgslAsync(formatHashHex(entries[0].hash)),
    pack.getWgslAsync(entries[0].hash),
    pack.getWgslAsync(entries[0].hash),
    pack.getWgslAsync(formatHashHex(entries[10].hash)),
    pack.getWgslAsync(entries[10].hash),
  ];

  // Exactly 4 should be active immediately; 16 distinct queued; 5 duplicates coalesced.
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(liveConcurrent, 4);
  assert.equal(observedPeak, 4);
  assert.equal(pendingReleases.length, 4);
  assert.equal(pack.getStats().activeRangeFetches, 4);
  assert.equal(pack.getStats().peakConcurrentRangeFetches, 4);
  assert.equal(pack.getStats().inFlightCount, 20);
  assert.equal(pack.getStats().inFlightCoalesced, 5);

  // Release requests one by one in FIFO order and verify concurrency never exceeds 4.
  while (startedOrder.length < count || pendingReleases.length > 0) {
    assert.ok(liveConcurrent <= 4, `active fetches exceeded cap 4: ${liveConcurrent}`);
    assert.ok(pack.getStats().activeRangeFetches <= 4);
    const release = pendingReleases.shift();
    if (release) {
      release();
    }
    await new Promise((r) => setTimeout(r, 2));
  }

  const results = await Promise.all(promises);
  const dupResults = await Promise.all(dupPromises);
  for (let i = 0; i < count; i += 1) {
    assert.equal(results[i], entries[i].wgsl);
  }
  assert.equal(dupResults[0], entries[0].wgsl);
  assert.equal(dupResults[3], entries[10].wgsl);

  assert.equal(observedPeak, 4, 'peak concurrent range fetches must be capped at 4');
  assert.deepEqual(startedOrder, expectedRangeByIdx, 'queued range fetches must dispatch in strict FIFO order');
  assert.equal(capMock.requests.length, 1 + count, 'each distinct hash must issue exactly 1 Range request');

  const finalStats = pack.getStats();
  assert.equal(finalStats.activeRangeFetches, 0);
  assert.equal(finalStats.peakConcurrentRangeFetches, 4);
  assert.equal(finalStats.rangeFetchRetries, 0);
  assert.equal(finalStats.inFlightCount, 0);
  assert.equal(finalStats.inFlightCoalesced, 5);

  // 3. maxConcurrentRangeFetches: 0 (unlimited) dispatches all concurrent distinct hashes at once
  let unlimitedLive = 0;
  let unlimitedPeak = 0;
  const unlimitedReleases = [];
  const unlimitedMock = createRangeMockFetch(
    { '/packs/unlimited.wgsl.pack': packBuf },
    {
      beforeResponse: async ({ range }) => {
        if (!range || range.startsWith('bytes=0-')) {
          return null;
        }
        unlimitedLive += 1;
        if (unlimitedLive > unlimitedPeak) {
          unlimitedPeak = unlimitedLive;
        }
        await new Promise((resolve) => {
          unlimitedReleases.push(() => {
            unlimitedLive -= 1;
            resolve();
          });
        });
        return null;
      },
    }
  );
  const unlimitedPack = await loadIndexFromUrl('/packs/unlimited.wgsl.pack', {
    fetch: unlimitedMock.fetch,
    initialRangeBytes: tableEnd,
    maxConcurrentRangeFetches: 0,
  });
  const unlimitedPromises = entries.slice(0, 12).map((e) => unlimitedPack.getWgslAsync(e.hash));
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(unlimitedPeak, 12);
  assert.equal(unlimitedPack.getStats().activeRangeFetches, 12);
  assert.equal(unlimitedPack.getStats().peakConcurrentRangeFetches, 12);
  while (unlimitedReleases.length > 0) {
    unlimitedReleases.shift()();
  }
  await Promise.all(unlimitedPromises);
  assert.equal(unlimitedPack.getStats().activeRangeFetches, 0);
  assert.equal(unlimitedPack.getStats().peakConcurrentRangeFetches, 12);
});

test('W162 transient TypeError: Failed to fetch, HTTP 5xx, and HTTP 429 retry with bounded backoff and recover cleanly', async () => {
  const hNet = fnv1a64('w162-retry-net');
  const h503 = fnv1a64('w162-retry-503');
  const h429 = fnv1a64('w162-retry-429');
  const hExhaust = fnv1a64('w162-retry-exhaust');

  const packBuf = buildWgslPackBuffer([
    { hash: hNet, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: h503, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
    { hash: h429, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
    { hash: hExhaust, stage: SHADER_STAGE.COMPUTE, wgsl: '@compute @workgroup_size(2) fn ex() {}' },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 4 * WGPK_ENTRY_BYTES;

  const sleepCalls = [];
  const sleepFn = async (ms) => {
    sleepCalls.push(ms);
  };

  let indexFailuresRemaining = 1;
  const failuresByStart = new Map();
  let packRef = null;

  const mock = createRangeMockFetch(
    { '/packs/retry.wgsl.pack': packBuf },
    {
      beforeResponse: async ({ range }) => {
        if (range && range.startsWith('bytes=0-')) {
          if (indexFailuresRemaining > 0) {
            indexFailuresRemaining -= 1;
            throw new TypeError('Failed to fetch');
          }
          return null;
        }
        const match = /^bytes=(\d+)-/.exec(range || '');
        if (!match) return null;
        const start = Number(match[1]);
        const spec = failuresByStart.get(start);
        if (!spec || spec.remaining <= 0) {
          return null;
        }
        spec.remaining -= 1;
        if (spec.kind === 'typeerror') {
          throw new TypeError('Failed to fetch');
        }
        return {
          ok: false,
          status: spec.status,
          statusText: spec.statusText,
          headers: new Headers(),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      },
    }
  );

  // Initial index load recovers from 1 transient TypeError: Failed to fetch!
  packRef = await loadIndexFromUrl('/packs/retry.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    sleepFn,
  });
  assert.equal(packRef.getStats().rangeFetchRetries, 1);
  assert.equal(packRef.getStats().rangeFetchCount, 2); // 1 failed + 1 succeeded
  assert.deepEqual(sleepCalls, [25]);

  const offNet = packRef.entries.get(hNet).offset;
  const off503 = packRef.entries.get(h503).offset;
  const off429 = packRef.entries.get(h429).offset;
  const offExhaust = packRef.entries.get(hExhaust).offset;

  // 1. TypeError: Failed to fetch fails twice, succeeds on 3rd attempt (default maxRangeFetchRetries = 2)
  failuresByStart.set(offNet, { kind: 'typeerror', remaining: 2 });
  sleepCalls.length = 0;
  const wgslNet = await packRef.getWgslAsync(hNet);
  assert.equal(wgslNet, SAMPLE_VS_WGSL);
  assert.deepEqual(sleepCalls, [25, 50]);
  assert.equal(packRef.getStats().rangeFetchRetries, 3);
  assert.equal(packRef.getStats().activeRangeFetches, 0);

  // 2. HTTP 503 fails once, recovers on retry #1
  failuresByStart.set(off503, { kind: 'http', status: 503, statusText: 'Service Unavailable', remaining: 1 });
  sleepCalls.length = 0;
  const wgsl503 = await packRef.getWgslAsync(h503);
  assert.equal(wgsl503, SAMPLE_FS_WGSL);
  assert.deepEqual(sleepCalls, [25]);
  assert.equal(packRef.getStats().rangeFetchRetries, 4);

  // 3. HTTP 429 Too Many Requests fails once, recovers on retry #1
  failuresByStart.set(off429, { kind: 'http', status: 429, statusText: 'Too Many Requests', remaining: 1 });
  sleepCalls.length = 0;
  const wgsl429 = await packRef.getWgslAsync(h429);
  assert.equal(wgsl429, SAMPLE_CS_WGSL);
  assert.deepEqual(sleepCalls, [25]);
  assert.equal(packRef.getStats().rangeFetchRetries, 5);

  // 4. Exhausted retries (3 consecutive TypeError: Failed to fetch with maxRangeFetchRetries = 2)
  failuresByStart.set(offExhaust, { kind: 'typeerror', remaining: 5 });
  sleepCalls.length = 0;
  await assert.rejects(() => packRef.getWgslAsync(hExhaust), /Failed to fetch/);
  assert.deepEqual(sleepCalls, [25, 50]);
  assert.equal(packRef.getStats().rangeFetchRetries, 7);
  assert.equal(packRef.getStats().activeRangeFetches, 0);
  assert.equal(packRef.getStats().inFlightCount, 0);
});

test('W162 non-retryable HTTP 404, 416, 400, and AbortError fail fast immediately without wasting retries', async () => {
  const h404 = fnv1a64('w162-failfast-404');
  const h416 = fnv1a64('w162-failfast-416');
  const h400 = fnv1a64('w162-failfast-400');
  const hAbort = fnv1a64('w162-failfast-abort');

  const packBuf = buildWgslPackBuffer([
    { hash: h404, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: h416, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
    { hash: h400, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
    { hash: hAbort, stage: SHADER_STAGE.COMPUTE, wgsl: '@compute @workgroup_size(4) fn ab() {}' },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 4 * WGPK_ENTRY_BYTES;

  const sleepCalls = [];
  const sleepFn = async (ms) => {
    sleepCalls.push(ms);
  };

  let attemptCounts = new Map();
  let modeByStart = new Map();

  const mock = createRangeMockFetch(
    { '/packs/failfast.wgsl.pack': packBuf },
    {
      beforeResponse: async ({ range }) => {
        if (!range || range.startsWith('bytes=0-')) {
          return null;
        }
        const start = Number(/^bytes=(\d+)-/.exec(range)[1]);
        attemptCounts.set(start, (attemptCounts.get(start) || 0) + 1);
        const mode = modeByStart.get(start);
        if (mode === 'abort') {
          const err = new Error('The operation was aborted');
          err.name = 'AbortError';
          throw err;
        }
        if (typeof mode === 'number') {
          return {
            ok: false,
            status: mode,
            statusText: `Status ${mode}`,
            headers: new Headers(),
            async arrayBuffer() {
              return new ArrayBuffer(0);
            },
          };
        }
        return null;
      },
    }
  );

  const pack = await loadIndexFromUrl('/packs/failfast.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    maxRangeFetchRetries: 3,
    rangeRetryDelayMs: 50,
    sleepFn,
  });

  const off404 = pack.entries.get(h404).offset;
  const off416 = pack.entries.get(h416).offset;
  const off400 = pack.entries.get(h400).offset;
  const offAbort = pack.entries.get(hAbort).offset;

  modeByStart.set(off404, 404);
  modeByStart.set(off416, 416);
  modeByStart.set(off400, 400);
  modeByStart.set(offAbort, 'abort');

  await assert.rejects(() => pack.getWgslAsync(h404), /HTTP 404/);
  await assert.rejects(() => pack.getWgslAsync(h416), /HTTP 416/);
  await assert.rejects(() => pack.getWgslAsync(h400), /HTTP 400/);
  await assert.rejects(() => pack.getWgslAsync(hAbort), (err) => err.name === 'AbortError');

  assert.equal(attemptCounts.get(off404), 1, '404 must not retry');
  assert.equal(attemptCounts.get(off416), 1, '416 must not retry');
  assert.equal(attemptCounts.get(off400), 1, '400 must not retry');
  assert.equal(attemptCounts.get(offAbort), 1, 'AbortError must not retry');
  assert.equal(sleepCalls.length, 0, 'sleepFn must never be called for non-retryable errors');

  const stats = pack.getStats();
  assert.equal(stats.rangeFetchRetries, 0);
  assert.equal(stats.activeRangeFetches, 0);
  assert.equal(stats.inFlightCount, 0);
});

test('W162 WgslPackCache forwards concurrency/retry options and aggregates rangeFetchRetries, peakConcurrentRangeFetches, and activeRangeFetches', async () => {
  const entries = [];
  for (let i = 0; i < 6; i += 1) {
    entries.push({
      hash: fnv1a64(`w162-cache-shader-${i}`),
      stage: SHADER_STAGE.COMPUTE,
      wgsl: `@compute @workgroup_size(${i + 1}) fn c_${i}() {}`,
    });
  }
  const packBuf = buildWgslPackBuffer(entries);
  const tableEnd = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES;

  const sleepCalls = [];
  const sleepFn = async (ms) => {
    sleepCalls.push(ms);
  };

  let failFirstPayload = true;
  const releases = [];

  const mock = createRangeMockFetch(
    { '/packs/cache-w162.wgsl.pack': packBuf },
    {
      beforeResponse: async ({ range }) => {
        if (!range || range.startsWith('bytes=0-')) {
          return null;
        }
        if (failFirstPayload) {
          failFirstPayload = false;
          throw new TypeError('Failed to fetch');
        }
        await new Promise((resolve) => {
          releases.push(resolve);
        });
        return null;
      },
    }
  );

  const cache = new WgslPackCache({
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    maxConcurrentRangeFetches: 3,
    maxRangeFetchRetries: 2,
    rangeRetryDelayMs: 10,
    sleepFn,
  });

  const pack = await cache.loadIndexFromUrl('/packs/cache-w162.wgsl.pack');
  assert.equal(pack.getStats().maxConcurrentRangeFetches, 3);
  assert.equal(pack.getStats().maxRangeFetchRetries, 2);

  const promises = entries.map((e) => cache.getWgslAsync(e.hash));
  await new Promise((r) => setTimeout(r, 10));

  // While gated, activeRangeFetches and peakConcurrentRangeFetches must be visible on both pack and cache
  const midCacheStats = cache.getStats();
  assert.equal(midCacheStats.activeRangeFetches, 3);
  assert.equal(midCacheStats.peakConcurrentRangeFetches, 3);
  assert.equal(midCacheStats.rangeFetchRetries, 1);

  while (releases.length > 0 || cache.getStats().activeRangeFetches > 0) {
    const rel = releases.shift();
    if (rel) rel();
    await new Promise((r) => setTimeout(r, 2));
  }

  const results = await Promise.all(promises);
  for (let i = 0; i < entries.length; i += 1) {
    assert.equal(results[i], entries[i].wgsl);
  }

  assert.deepEqual(sleepCalls, [10]);

  const packStats = pack.stats();
  const cacheStats = cache.getStats();
  for (const s of [packStats, cacheStats]) {
    assert.equal(s.rangeFetchRetries, 1);
    assert.equal(s.peakConcurrentRangeFetches, 3);
    assert.equal(s.activeRangeFetches, 0);
    assert.equal(s.inFlightCount, 0);
  }

  const snapshot = cache.inspect();
  assert.doesNotThrow(() => JSON.stringify(snapshot));
  assert.equal(snapshot.stats.rangeFetchRetries, 1);
  assert.equal(snapshot.stats.peakConcurrentRangeFetches, 3);
  assert.equal(snapshot.stats.activeRangeFetches, 0);
});

// ---------------------------------------------------------------------------
// W202: Hot-shader LRU lookup optimization, coalesced ranged prefetchHashes,
//       in-flight batch deduplication, and cache/prefetch telemetry
// ---------------------------------------------------------------------------

test('W202 hot-hash WGSL LRU cache tracks lruHits/lruMisses/lruEvictions/lruBytes and never re-fetches hot intro/menu shaders', async () => {
  assert.equal(DEFAULT_MAX_CACHED_WGSL_BYTES, 32 * 1024 * 1024);
  assert.equal(DEFAULT_PREFETCH_MAX_GAP_BYTES, 16 * 1024);
  assert.equal(DEFAULT_PREFETCH_MAX_SPAN_BYTES, 512 * 1024);

  const enc = new TextEncoder();
  const makeSizedWgsl = (tag, targetBytes) => {
    const prefix = `@compute @workgroup_size(1) fn w202_${tag}() { /* `;
    const suffix = ' */ }';
    const padLen = targetBytes - enc.encode(prefix).byteLength - enc.encode(suffix).byteLength;
    assert.ok(padLen >= 0);
    return `${prefix}${'h'.repeat(padLen)}${suffix}`;
  };

  const hIntro = fnv1a64('w202-hot-intro');
  const hMenu = fnv1a64('w202-hot-menu');
  const hWorld1 = fnv1a64('w202-cold-world-1');
  const hWorld2 = fnv1a64('w202-cold-world-2');

  const wgslIntro = makeSizedWgsl('intro', 120);
  const wgslMenu = makeSizedWgsl('menu', 120);
  const wgslWorld1 = makeSizedWgsl('world1', 120);
  const wgslWorld2 = makeSizedWgsl('world2', 120);

  const packBuf = buildWgslPackBuffer([
    { hash: hIntro, stage: SHADER_STAGE.VERTEX, wgsl: wgslIntro },
    { hash: hMenu, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslMenu },
    { hash: hWorld1, stage: SHADER_STAGE.COMPUTE, wgsl: wgslWorld1 },
    { hash: hWorld2, stage: SHADER_STAGE.COMPUTE, wgsl: wgslWorld2 },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 4 * WGPK_ENTRY_BYTES;
  const mock = createRangeMockFetch({ '/packs/w202-lru.wgsl.pack': packBuf });

  // Budget = 360 bytes -> holds 3 x 120-byte shaders; 4th triggers eviction of LRU tail.
  const pack = await loadIndexFromUrl('/packs/w202-lru.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    maxCachedWgslBytes: 360,
  });

  // Load intro, menu, world1 -> 3 misses, 0 hits, 360 lruBytes
  assert.equal(await pack.getWgslAsync(hIntro), wgslIntro);
  assert.equal(await pack.getWgslAsync(hMenu), wgslMenu);
  assert.equal(await pack.getWgslAsync(hWorld1), wgslWorld1);
  assert.equal(mock.requests.length, 4); // 1 index + 3 shaders

  let s = pack.stats();
  assert.equal(s.lruHits, 0);
  assert.equal(s.lruMisses, 3);
  assert.equal(s.lruEvictions, 0);
  assert.equal(s.lruBytes, 360);
  assert.equal(s.prefetchBatches, 0);
  assert.equal(s.prefetchShaders, 0);

  // Repeated hot lookups of intro (canonical hex string + BigInt) and menu (back-to-back MRU fast-path)
  // must NEVER issue HTTP Range requests and must keep both intro & menu hot!
  const introHex = formatHashHex(hIntro);
  const menuHex = formatHashHex(hMenu);
  for (let i = 0; i < 5; i += 1) {
    assert.equal(await pack.getWgslAsync(introHex), wgslIntro);
    assert.equal(await pack.getWgslAsync(hIntro), wgslIntro);
  }
  for (let i = 0; i < 4; i += 1) {
    assert.equal(await pack.getWgslAsync(menuHex), wgslMenu);
  }
  assert.equal(mock.requests.length, 4, 'hot intro/menu lookups must issue 0 extra Range fetches');

  s = pack.stats();
  assert.equal(s.lruHits, 14);
  assert.equal(s.lruMisses, 3);
  assert.equal(s.lruEvictions, 0);
  assert.equal(s.lruBytes, 360);

  // Now fetch cold world2 -> must evict world1 (the oldest untouched entry), keeping intro & menu hot!
  assert.equal(await pack.getWgslAsync(hWorld2), wgslWorld2);
  assert.equal(mock.requests.length, 5);

  s = pack.stats();
  assert.equal(s.lruEvictions, 1);
  assert.equal(s.lruMisses, 4);
  assert.equal(s.lruBytes, 360);

  // Intro & menu are still resident (0 HTTP fetches), whereas world1 was evicted (1 HTTP fetch).
  assert.equal(await pack.getWgslAsync(introHex), wgslIntro);
  assert.equal(await pack.getWgslAsync(menuHex), wgslMenu);
  assert.equal(mock.requests.length, 5);
  assert.equal(pack.stats().lruHits, 16);

  assert.equal(await pack.getWgslAsync(hWorld1), wgslWorld1);
  assert.equal(mock.requests.length, 6);
  assert.equal(pack.stats().lruEvictions, 2);
  assert.equal(pack.stats().lruMisses, 5);
  assert.equal(pack.stats().lruBytes, 360);
});

test('W202 prefetchHashes coalesces nearby byte ranges by maxGapBytes/maxSpanBytes and skips unrequested gap entries', async () => {
  const enc = new TextEncoder();
  const makeFixed = (tag, size) => {
    const prefix = `@compute @workgroup_size(1) fn pf_${tag}() { /* `;
    const suffix = ' */ }';
    const padLen = size - enc.encode(prefix).byteLength - enc.encode(suffix).byteLength;
    assert.ok(padLen >= 0);
    return `${prefix}${'p'.repeat(padLen)}${suffix}`;
  };

  // Layout in pack:
  //   [0] h0: 100B (requested)
  //   [1] hGapSmall: 60B (NOT requested - small gap <= 128B)
  //   [2] h1: 100B (requested -> coalesces with h0 across 60B gap!)
  //   [3] h2: 100B (requested -> contiguous with h1 -> same batch!)
  //   [4] hGapHuge: 400B (NOT requested - huge gap > 128B -> forces batch split!)
  //   [5] h3: 100B (requested -> batch 2)
  //   [6] h4: 100B (requested -> contiguous with h3 -> batch 2)
  //   [7] hOversized: 600B (requested -> single shader > maxSpanBytes=380 -> own exact batch 3)
  const h0 = fnv1a64('w202-pf-0');
  const hGapSmall = fnv1a64('w202-pf-gap-small');
  const h1 = fnv1a64('w202-pf-1');
  const h2 = fnv1a64('w202-pf-2');
  const hGapHuge = fnv1a64('w202-pf-gap-huge');
  const h3 = fnv1a64('w202-pf-3');
  const h4 = fnv1a64('w202-pf-4');
  const hOversized = fnv1a64('w202-pf-oversized');
  const hEmpty = fnv1a64('w202-pf-empty');
  const hMissing = fnv1a64('w202-pf-missing');

  const entries = [
    { hash: h0, stage: SHADER_STAGE.VERTEX, wgsl: makeFixed('0', 100) },
    { hash: hGapSmall, stage: SHADER_STAGE.FRAGMENT, wgsl: makeFixed('gs', 60) },
    { hash: h1, stage: SHADER_STAGE.FRAGMENT, wgsl: makeFixed('1', 100) },
    { hash: h2, stage: SHADER_STAGE.COMPUTE, wgsl: makeFixed('2', 100) },
    { hash: hGapHuge, stage: SHADER_STAGE.COMPUTE, wgsl: makeFixed('gh', 400) },
    { hash: h3, stage: SHADER_STAGE.VERTEX, wgsl: makeFixed('3', 100) },
    { hash: h4, stage: SHADER_STAGE.FRAGMENT, wgsl: makeFixed('4', 100) },
    { hash: hOversized, stage: SHADER_STAGE.COMPUTE, wgsl: makeFixed('big', 600) },
    { hash: hEmpty, stage: SHADER_STAGE.COMPUTE, wgsl: '' },
  ];

  const packBuf = buildWgslPackBuffer(entries);
  const tableEnd = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES;
  const mock = createRangeMockFetch({ '/packs/w202-pf.wgsl.pack': packBuf });

  const pack = await loadIndexFromUrl('/packs/w202-pf.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    prefetchMaxGapBytes: 128,
    prefetchMaxSpanBytes: 380,
  });

  // Pass hashes deliberately out-of-order, with duplicates, mixed hex/BigInt, empty payload, and unknown hash
  const result = await pack.prefetchHashes([
    h4,
    formatHashHex(h1),
    hOversized,
    h0,
    h1, // duplicate
    h3,
    h2,
    hEmpty,
    hMissing,
  ]);

  // Expected batches:
  //   Batch 1: [h0..h2] span = 100 + 60 + 100 + 100 = 360B (<= 380B, gap 60 <= 128)
  //   Batch 2: [h3..h4] span = 100 + 100 = 200B (gap from h2 is 400B > 128B)
  //   Batch 3: [hOversized] span = 600B (h3..hOversized would be 800B > 380B -> own exact span)
  assert.equal(result.requested, 8); // 6 fetched + 1 empty cached + 1 missing
  assert.equal(result.prefetched, 6);
  assert.equal(result.cached, 1); // hEmpty (0-length)
  assert.equal(result.inFlight, 0);
  assert.equal(result.missing, 1);
  assert.equal(result.batches, 3);
  assert.equal(result.bytesFetched, 360 + 200 + 600);

  // Only 1 index + 3 batched HTTP 206 Range requests instead of 1 + 6!
  assert.equal(mock.requests.length, 4);
  const off0 = pack.entries.get(h0).offset;
  const off2End = pack.entries.get(h2).offset + pack.entries.get(h2).length - 1;
  const off3 = pack.entries.get(h3).offset;
  const off4End = pack.entries.get(h4).offset + pack.entries.get(h4).length - 1;
  const offBig = pack.entries.get(hOversized).offset;
  const offBigEnd = offBig + pack.entries.get(hOversized).length - 1;

  const payloadRanges = mock.requests.slice(1).map((r) => r.range);
  assert.deepEqual(payloadRanges, [
    `bytes=${off0}-${off2End}`,
    `bytes=${off3}-${off4End}`,
    `bytes=${offBig}-${offBigEnd}`,
  ]);

  // Verify telemetry on pack
  const s = pack.stats();
  assert.equal(s.prefetchBatches, 3);
  assert.equal(s.prefetchShaders, 6);
  assert.equal(s.lruBytes, 100 * 5 + 600); // unrequested hGapSmall was NOT cached into LRU!
  assert.equal(s.cachedWgslEntries, 7); // 6 fetched + 1 empty
  assert.equal(s.lruHits, 0);
  assert.equal(s.lruMisses, 0);
  assert.equal(s.prefetchMaxGapBytes, 128);
  assert.equal(s.prefetchMaxSpanBytes, 380);

  // Subsequent getWgslAsync for every prefetched shader hits LRU with ZERO HTTP fetches!
  for (const idx of [0, 2, 3, 5, 6, 7]) {
    assert.equal(await pack.getWgslAsync(entries[idx].hash), entries[idx].wgsl);
  }
  assert.equal(mock.requests.length, 4);
  assert.equal(pack.stats().lruHits, 6);
  assert.equal(pack.stats().lruMisses, 0);

  // Unrequested gap entry hGapSmall is NOT in LRU -> triggers a fresh Range fetch when actually requested
  assert.equal(await pack.getWgslAsync(hGapSmall), entries[1].wgsl);
  assert.equal(mock.requests.length, 5);
  assert.equal(pack.stats().lruMisses, 1);

  // Repeating prefetchHashes on already-cached shaders does 0 HTTP requests
  const repeat = await pack.prefetchHashes([h0, h1, h2]);
  assert.equal(repeat.prefetched, 0);
  assert.equal(repeat.cached, 3);
  assert.equal(repeat.batches, 0);
  assert.equal(repeat.bytesFetched, 0);
  assert.equal(mock.requests.length, 5);
  assert.equal(pack.stats().prefetchBatches, 3);
  assert.equal(pack.stats().prefetchShaders, 6);

  // Per-call maxGapBytes=0 only coalesces strictly contiguous entries (never bridges a gap)
  pack.clearWgslCache();
  assert.equal(pack.stats().lruBytes, 0);
  const strictRes = await pack.prefetchHashes([h0, h1, h2], { maxGapBytes: 0, maxSpanBytes: 1024 });
  assert.equal(strictRes.batches, 2); // [h0] and [h1, h2] (60B gap between h0 & h1 splits)
  assert.equal(strictRes.prefetched, 3);
  assert.equal(strictRes.bytesFetched, 100 + 200);
});

test('W202 prefetchHashes deduplicates in-flight fetches bidirectionally with getWgslAsync and cleans up on errors/retries', async () => {
  const hA = fnv1a64('w202-inflight-A');
  const hB = fnv1a64('w202-inflight-B');
  const hC = fnv1a64('w202-inflight-C');
  const hD = fnv1a64('w202-inflight-D');

  const wgslA = `@vertex fn a_vs() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }`;
  const wgslB = `@fragment fn b_fs() -> @location(0) vec4<f32> { return vec4<f32>(0.5); }`;
  const wgslC = `@compute @workgroup_size(4) fn c_cs() {}`;
  const wgslD = `@compute @workgroup_size(8) fn d_cs() {}`;

  const packBuf = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: wgslA },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslB },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, wgsl: wgslC },
    { hash: hD, stage: SHADER_STAGE.COMPUTE, wgsl: wgslD },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 4 * WGPK_ENTRY_BYTES;

  const gates = [];
  let failNextBatch = false;
  let holdPayloads = false;

  const mock = createRangeMockFetch(
    { '/packs/w202-inflight.wgsl.pack': packBuf },
    {
      beforeResponse: async ({ range }) => {
        if (!range || range.startsWith('bytes=0-')) {
          return null;
        }
        if (holdPayloads) {
          await new Promise((resolve) => {
            gates.push(resolve);
          });
        }
        if (failNextBatch) {
          failNextBatch = false;
          return {
            ok: false,
            status: 503,
            statusText: 'Service Unavailable',
            headers: new Headers(),
            async arrayBuffer() {
              return new ArrayBuffer(0);
            },
          };
        }
        return null;
      },
    }
  );

  const pack = await loadIndexFromUrl('/packs/w202-inflight.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    maxRangeFetchRetries: 0,
  });

  // 1. Start prefetchHashes([hA, hB]) while gated, then concurrently call getWgslAsync(hA), getWgslAsync(hB),
  //    and another prefetchHashes([hB, hC])!
  holdPayloads = true;
  const pPrefetch1 = pack.prefetchHashes([hA, hB]);
  const pGetA = pack.getWgslAsync(formatHashHex(hA));
  const pGetB = pack.getWgslAsync(hB);
  const pPrefetch2 = pack.prefetchHashes([hB, hC]); // hB deduplicates onto pPrefetch1, only hC fetches!

  await new Promise((r) => setTimeout(r, 10));
  assert.equal(gates.length, 2, 'only 2 HTTP 206 requests expected: batch([A,B]) and batch([C])');
  assert.equal(pack.stats().inFlightCount, 3); // hA, hB, hC
  assert.equal(pack.stats().inFlightCoalesced, 3); // pGetA, pGetB, and hB in pPrefetch2

  holdPayloads = false;
  while (gates.length > 0) {
    gates.shift()();
  }

  const [res1, gotA, gotB, res2] = await Promise.all([pPrefetch1, pGetA, pGetB, pPrefetch2]);
  assert.equal(gotA, wgslA);
  assert.equal(gotB, wgslB);
  assert.equal(res1.prefetched, 2);
  assert.equal(res1.batches, 1);
  assert.equal(res2.inFlight, 1); // hB deduplicated onto pPrefetch1
  assert.equal(res2.prefetched, 1); // hC
  assert.equal(res2.batches, 1);
  assert.equal(mock.requests.length, 3); // 1 index + batch([A,B]) + batch([C])
  assert.equal(pack.stats().inFlightCount, 0);

  // 2. Start getWgslAsync(hD) first, then call prefetchHashes([hD]) while hD is in flight -> 0 extra fetches!
  holdPayloads = true;
  const pGetD = pack.getWgslAsync(hD);
  const pPrefetchD = pack.prefetchHashes([hD]);
  await new Promise((r) => setTimeout(r, 10));
  assert.equal(gates.length, 1);
  assert.equal(pack.stats().inFlightCount, 1);
  holdPayloads = false;
  gates.shift()();

  const [gotD, resD] = await Promise.all([pGetD, pPrefetchD]);
  assert.equal(gotD, wgslD);
  assert.equal(resD.inFlight, 1);
  assert.equal(resD.prefetched, 0);
  assert.equal(resD.batches, 0);
  assert.equal(mock.requests.length, 4); // +1 for hD only

  // 3. Batch failure cleans up inFlightFetches BEFORE caller .catch() runs so immediate retry succeeds!
  pack.clearWgslCache();
  failNextBatch = true;
  let inFlightInCatch = -1;
  let retryWgslA = null;
  const pFailPrefetch = pack.prefetchHashes([hA, hB]);
  const pFailGetA = pack.getWgslAsync(hA).catch(async (err) => {
    assert.match(err.message, /HTTP 503/);
    inFlightInCatch = pack.stats().inFlightCount;
    retryWgslA = await pack.getWgslAsync(hA);
    return 'recovered';
  });

  const [settledPrefetch, settledGetA] = await Promise.allSettled([pFailPrefetch, pFailGetA]);
  assert.equal(settledPrefetch.status, 'rejected');
  assert.match(settledPrefetch.reason.message, /HTTP 503/);
  assert.equal(settledGetA.status, 'fulfilled');
  assert.equal(settledGetA.value, 'recovered');
  assert.equal(inFlightInCatch, 0, 'inFlightFetches must clear before rejecting callers');
  assert.equal(retryWgslA, wgslA);
  assert.equal(pack.stats().inFlightCount, 0);
});

test('W202 WgslPackCache.prefetchHashes aggregates multi-pack ranged/buffer/dynamic prefetch and exposes telemetry', async () => {
  const hR1 = fnv1a64('w202-cache-r1');
  const hR2 = fnv1a64('w202-cache-r2');
  const hR3 = fnv1a64('w202-cache-r3');
  const hBuf = fnv1a64('w202-cache-buf');
  const hDyn = fnv1a64('w202-cache-dyn');
  const hUnknown = fnv1a64('w202-cache-unknown');

  const rangedBuf = buildWgslPackBuffer([
    { hash: hR1, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hR2, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
    { hash: hR3, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
  ]);
  const bufferBuf = buildWgslPackBuffer([
    { hash: hBuf, stage: SHADER_STAGE.COMPUTE, wgsl: '@compute @workgroup_size(1) fn buf_main() {}' },
  ]);
  const rangedTableEnd = WGPK_HEADER_BYTES + 3 * WGPK_ENTRY_BYTES;

  const mock = createRangeMockFetch({ '/packs/w202-ranged.wgsl.pack': rangedBuf });
  const cache = new WgslPackCache({
    fetch: mock.fetch,
    initialRangeBytes: rangedTableEnd,
    maxCachedWgslBytes: 4096,
    prefetchMaxGapBytes: 1024,
    prefetchMaxSpanBytes: 8192,
  });

  await cache.loadIndexFromUrl('/packs/w202-ranged.wgsl.pack');
  cache.loadFromBuffer(bufferBuf, 'boot.wgsl.pack');
  const dynWgsl = '@fragment fn dyn_hot() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';
  cache.registerDynamicShader(hDyn, 'fragment', dynWgsl);

  // Coalesce all 3 ranged shaders into 1 single HTTP 206 fetch while handling buffer, dynamic, and missing hashes cleanly
  const summary = await cache.prefetchHashes([hR1, hR2, hR3, hBuf, hDyn, hUnknown, hR1]);
  assert.equal(summary.requested, 6);
  assert.equal(summary.prefetched, 3);
  assert.equal(summary.cached, 2); // hBuf + hDyn
  assert.equal(summary.inFlight, 0);
  assert.equal(summary.missing, 1);
  assert.equal(summary.batches, 1);
  assert.equal(mock.requests.length, 2); // 1 index + 1 coalesced batch

  // Now look up all 3 ranged shaders via cache -> 3 LRU hits, 0 extra HTTP fetches!
  assert.equal(await cache.getWgslAsync(hR1), SAMPLE_VS_WGSL);
  assert.equal(await cache.getWgslAsync(hR2), SAMPLE_FS_WGSL);
  assert.equal(await cache.getWgslAsync(hR3), SAMPLE_CS_WGSL);
  assert.equal(mock.requests.length, 2);

  const stats = cache.getStats();
  assert.equal(stats.lruHits, 3);
  assert.equal(stats.lruMisses, 0);
  assert.equal(stats.lruEvictions, 0);
  assert.equal(
    stats.lruBytes,
    new TextEncoder().encode(SAMPLE_VS_WGSL).byteLength +
      new TextEncoder().encode(SAMPLE_FS_WGSL).byteLength +
      new TextEncoder().encode(SAMPLE_CS_WGSL).byteLength
  );
  assert.equal(stats.prefetchBatches, 1);
  assert.equal(stats.prefetchShaders, 3);

  // inspect() remains JSON-serializable with all W202 telemetry fields
  const snapshot = cache.inspect();
  assert.doesNotThrow(() => JSON.stringify(snapshot));
  assert.equal(snapshot.stats.lruHits, 3);
  assert.equal(snapshot.stats.lruMisses, 0);
  assert.equal(snapshot.stats.lruEvictions, 0);
  assert.equal(snapshot.stats.lruBytes, stats.lruBytes);
  assert.equal(snapshot.stats.prefetchBatches, 1);
  assert.equal(snapshot.stats.prefetchShaders, 3);

  // Null/empty hashes return zero summary; invalid options reject with RangeError/TypeError
  const emptyRes = await cache.prefetchHashes(null);
  assert.deepEqual(emptyRes, {
    requested: 0,
    prefetched: 0,
    cached: 0,
    inFlight: 0,
    missing: 0,
    batches: 0,
    bytesFetched: 0,
  });
  await assert.rejects(() => cache.prefetchHashes(123), TypeError);
  await assert.rejects(() => cache.prefetchHashes([], { maxGapBytes: -1 }), RangeError);
  await assert.rejects(() => cache.prefetchHashes([], { maxSpanBytes: 0 }), RangeError);
  await assert.rejects(
    () => loadIndexFromUrl('/packs/w202-ranged.wgsl.pack', { fetch: mock.fetch, prefetchMaxGapBytes: -5 }),
    RangeError
  );
  await assert.rejects(
    () => loadIndexFromUrl('/packs/w202-ranged.wgsl.pack', { fetch: mock.fetch, prefetchMaxSpanBytes: 0 }),
    RangeError
  );
});

test('W202 prefetchHashes handles 200 OK fallback, partial UTF-8 errors, and >2 GiB unsigned u32 offsets cleanly', async () => {
  const enc = new TextEncoder();
  const h1 = fnv1a64('w202-edge-1');
  const h2 = fnv1a64('w202-edge-2');
  const packBuf = buildWgslPackBuffer([
    { hash: h1, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: h2, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES;

  // 1. Server returns 200 OK on prefetch range request (non-range server mid-stream)
  let serve200AfterIndex = false;
  const mock200 = createRangeMockFetch(
    { '/packs/w202-200.wgsl.pack': packBuf },
    {
      beforeResponse: async ({ range }) => {
        if (serve200AfterIndex && range && !range.startsWith('bytes=0-')) {
          return {
            ok: true,
            status: 200,
            statusText: 'OK',
            headers: new Headers({ 'Content-Length': String(packBuf.byteLength) }),
            async arrayBuffer() {
              return packBuf.slice(0);
            },
          };
        }
        return null;
      },
    }
  );
  const pack200 = await loadIndexFromUrl('/packs/w202-200.wgsl.pack', {
    fetch: mock200.fetch,
    initialRangeBytes: tableEnd,
  });
  serve200AfterIndex = true;
  const res200 = await pack200.prefetchHashes([h1, h2]);
  assert.equal(res200.prefetched, 2);
  assert.equal(res200.batches, 1);
  assert.equal(await pack200.getWgslAsync(h1), SAMPLE_VS_WGSL);
  assert.equal(await pack200.getWgslAsync(h2), SAMPLE_FS_WGSL);
  assert.equal(pack200.stats().lruHits, 2);

  // 2. Corrupt UTF-8 in h1 inside a coalesced [h1, h2] batch rejects h1 while still caching valid h2
  const corruptBytes = new Uint8Array(packBuf).slice();
  corruptBytes[tableEnd] = 0xff;
  const mockBad = createRangeMockFetch({ '/packs/w202-bad.wgsl.pack': corruptBytes });
  const packBad = await loadIndexFromUrl('/packs/w202-bad.wgsl.pack', {
    fetch: mockBad.fetch,
    initialRangeBytes: tableEnd,
  });
  const pBadPrefetch = packBad.prefetchHashes([h1, h2]);
  const pBadGet2 = packBad.getWgslAsync(h2);
  await assert.rejects(() => pBadPrefetch, /not valid utf-8/i);
  assert.equal(await pBadGet2, SAMPLE_FS_WGSL);
  assert.equal(mockBad.requests.length, 2, 'h1 and h2 should share 1 coalesced batch request');
  assert.equal(packBad.stats().inFlightCount, 0);
  assert.equal(await packBad.getWgslAsync(h2), SAMPLE_FS_WGSL);
  assert.equal(packBad.stats().lruHits, 1);

  // 3. >2 GiB unsigned u32 coalesced prefetch (exact 4 GiB ceiling)
  const TOTAL_4_GIB = 0x100000000;
  const wgslHighA = `@vertex fn hi_a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }`;
  const wgslHighB = `@fragment fn hi_b() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }`;
  const lenA = enc.encode(wgslHighA).byteLength;
  const lenB = enc.encode(wgslHighB).byteLength;
  const offB = TOTAL_4_GIB - lenB;
  const offA = offB - lenA - 16; // 16-byte gap above 3.99 GiB!

  const hHighA = fnv1a64('w202-hi-a');
  const hHighB = fnv1a64('w202-hi-b');
  const highIndex = buildSyntheticLargeIndexBuffer([
    { hash: hHighA, stage: SHADER_STAGE.VERTEX, offset: offA, length: lenA },
    { hash: hHighB, stage: SHADER_STAGE.FRAGMENT, offset: offB, length: lenB },
  ]);

  const combinedSpan = new Uint8Array(TOTAL_4_GIB - offA);
  combinedSpan.set(enc.encode(wgslHighA), 0);
  combinedSpan.set(enc.encode(wgslHighB), offB - offA);

  const highRequests = [];
  const highFetch = async (url, init = {}) => {
    const range = init.headers && (init.headers.Range || init.headers.range);
    highRequests.push(range);
    const [, sStr, eStr] = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(sStr);
    const end = Number(eStr);
    const slice = start === 0 ? highIndex.slice(start, end + 1) : combinedSpan.slice();
    return {
      ok: true,
      status: 206,
      statusText: 'Partial Content',
      headers: new Headers({
        'Content-Range': `bytes ${start}-${end}/${TOTAL_4_GIB}`,
        'Content-Length': String(slice.byteLength),
      }),
      async arrayBuffer() {
        return slice.buffer;
      },
    };
  };

  const highPack = await loadIndexFromUrl('/packs/w202-4gib.wgsl.pack', {
    fetch: highFetch,
    initialRangeBytes: highIndex.byteLength,
  });
  const highRes = await highPack.prefetchHashes([hHighB, hHighA]);
  assert.equal(highRes.batches, 1);
  assert.equal(highRes.prefetched, 2);
  assert.equal(highRequests[1], `bytes=${offA}-${TOTAL_4_GIB - 1}`);
  assert.equal(await highPack.getWgslAsync(hHighA), wgslHighA);
  assert.equal(await highPack.getWgslAsync(hHighB), wgslHighB);
  assert.equal(highPack.stats().lruHits, 2);
});

// ---------------------------------------------------------------------------
// W239: Bounded LRU cache (maxCacheBytes / maxCacheEntries), hot-hash pinning
//       (pinHashes / unpinHashes), prefetchHashes({ maxCoalesceGapBytes }),
//       and sanitized getStats() telemetry for 1+ GiB (16,226-entry) packs
// ---------------------------------------------------------------------------

test('W239 bounded LRU cache enforces maxCacheEntries and maxCacheBytes with O(1) touch & eviction, keeping unbounded entry default', async () => {
  assert.equal(WgslPackLoader, WgslPackCache);

  const enc = new TextEncoder();
  const makeSized = (tag, targetBytes) => {
    const prefix = `@compute @workgroup_size(1) fn w239_${tag}() { /* `;
    const suffix = ' */ }';
    const padLen = targetBytes - enc.encode(prefix).byteLength - enc.encode(suffix).byteLength;
    assert.ok(padLen >= 0);
    return `${prefix}${'x'.repeat(padLen)}${suffix}`;
  };

  const hashes = [];
  const entries = [];
  for (let i = 0; i < 6; i += 1) {
    const h = fnv1a64(`w239-lru-entry-${i}`);
    hashes.push(h);
    entries.push({
      hash: h,
      stage: (i % 3) + 1,
      wgsl: makeSized(`e${i}`, 100),
    });
  }
  const packBuf = buildWgslPackBuffer(entries);
  const tableEnd = WGPK_HEADER_BYTES + entries.length * WGPK_ENTRY_BYTES;

  // 1. Default maxCacheEntries is unbounded (null): all 6 entries fit without entry-count eviction
  const defaultMock = createRangeMockFetch({ '/packs/w239-default.wgsl.pack': packBuf });
  const defaultPack = await loadIndexFromUrl('/packs/w239-default.wgsl.pack', {
    fetch: defaultMock.fetch,
    initialRangeBytes: tableEnd,
  });
  assert.equal(defaultPack.getStats().maxCacheEntries, null);
  for (let i = 0; i < entries.length; i += 1) {
    assert.equal(await defaultPack.getWgslAsync(hashes[i]), entries[i].wgsl);
  }
  assert.equal(defaultPack.getStats().cacheEntries, 6);
  assert.equal(defaultPack.getStats().cacheBytes, 600);
  assert.equal(defaultPack.getStats().cacheEvictions, 0);

  // 2. Bounded maxCacheEntries = 3 (with default 32 MiB byte budget): evicts LRU entry on 4th fetch
  const entryCapMock = createRangeMockFetch({ '/packs/w239-entries.wgsl.pack': packBuf });
  const entryCapPack = await loadIndexFromUrl('/packs/w239-entries.wgsl.pack', {
    fetch: entryCapMock.fetch,
    initialRangeBytes: tableEnd,
    maxCacheEntries: 3,
  });
  assert.equal(entryCapPack.getStats().maxCacheEntries, 3);

  assert.equal(await entryCapPack.getWgslAsync(hashes[0]), entries[0].wgsl);
  assert.equal(await entryCapPack.getWgslAsync(hashes[1]), entries[1].wgsl);
  assert.equal(await entryCapPack.getWgslAsync(hashes[2]), entries[2].wgsl);
  assert.equal(entryCapPack.getStats().cacheEntries, 3);
  assert.equal(entryCapPack.getStats().cacheEvictions, 0);
  assert.equal(entryCapMock.requests.length, 4); // 1 index + 3 shaders

  // Touch hashes[0] -> LRU order becomes [1, 2, 0]
  assert.equal(await entryCapPack.getWgslAsync(hashes[0]), entries[0].wgsl);
  assert.equal(entryCapPack.getStats().cacheHits, 1);
  assert.equal(entryCapMock.requests.length, 4);

  // Fetch hashes[3] -> exceeds maxCacheEntries=3 -> evicts hashes[1] -> resident: [2, 0, 3]
  assert.equal(await entryCapPack.getWgslAsync(hashes[3]), entries[3].wgsl);
  assert.equal(entryCapPack.getStats().cacheEntries, 3);
  assert.equal(entryCapPack.getStats().cacheBytes, 300);
  assert.equal(entryCapPack.getStats().cacheEvictions, 1);
  assert.equal(entryCapMock.requests.length, 5);

  // hashes[0] and hashes[2] are still cached (0 extra fetches)
  assert.equal(await entryCapPack.getWgslAsync(hashes[0]), entries[0].wgsl);
  assert.equal(await entryCapPack.getWgslAsync(hashes[2]), entries[2].wgsl);
  assert.equal(entryCapMock.requests.length, 5);
  assert.equal(entryCapPack.getStats().cacheHits, 3);

  // hashes[1] was evicted -> re-fetches and evicts hashes[3] (oldest remaining)
  assert.equal(await entryCapPack.getWgslAsync(hashes[1]), entries[1].wgsl);
  assert.equal(entryCapMock.requests.length, 6);
  assert.equal(entryCapPack.getStats().cacheEvictions, 2);

  // 3. Combined maxCacheBytes = 250 + maxCacheEntries = 5 (byte limit binds first at 2 entries)
  const byteCapMock = createRangeMockFetch({ '/packs/w239-bytes.wgsl.pack': packBuf });
  const byteCapPack = await loadIndexFromUrl('/packs/w239-bytes.wgsl.pack', {
    fetch: byteCapMock.fetch,
    initialRangeBytes: tableEnd,
    maxCacheBytes: 250,
    maxCacheEntries: 5,
  });
  assert.equal(byteCapPack.getStats().maxCacheBytes, 250);
  assert.equal(byteCapPack.getStats().maxCacheEntries, 5);
  assert.equal(await byteCapPack.getWgslAsync(hashes[0]), entries[0].wgsl);
  assert.equal(await byteCapPack.getWgslAsync(hashes[1]), entries[1].wgsl);
  assert.equal(await byteCapPack.getWgslAsync(hashes[2]), entries[2].wgsl);
  assert.equal(byteCapPack.getStats().cacheEntries, 2);
  assert.equal(byteCapPack.getStats().cacheBytes, 200);
  assert.equal(byteCapPack.getStats().cacheEvictions, 1);

  // 4. Option validation rejects negative / non-integer limits
  await assert.rejects(
    () => loadIndexFromUrl('/packs/w239-bytes.wgsl.pack', { fetch: byteCapMock.fetch, maxCacheBytes: -1 }),
    RangeError
  );
  await assert.rejects(
    () => loadIndexFromUrl('/packs/w239-bytes.wgsl.pack', { fetch: byteCapMock.fetch, maxCacheEntries: -1 }),
    RangeError
  );
  await assert.rejects(
    () => loadIndexFromUrl('/packs/w239-bytes.wgsl.pack', { fetch: byteCapMock.fetch, maxCacheEntries: 2.5 }),
    RangeError
  );
  await assert.rejects(
    () => byteCapPack.prefetchHashes([hashes[0]], { maxCoalesceGapBytes: -4 }),
    RangeError
  );
  assert.throws(() => byteCapPack.pinHashes(123), TypeError);
  assert.throws(() => byteCapPack.pinHashes('0x1234'), TypeError);
  assert.throws(() => byteCapPack.unpinHashes(123), TypeError);
});

test('W239 pinHashes and unpinHashes keep hot Bink2 / UI shaders resident across LRU eviction pressure', async () => {
  const enc = new TextEncoder();
  const make100B = (tag) => {
    const prefix = `@fragment fn pin_${tag}() -> @location(0) vec4<f32> { /* `;
    const suffix = ' */ return vec4<f32>(1.0); }';
    const padLen = 100 - enc.encode(prefix).byteLength - enc.encode(suffix).byteLength;
    assert.ok(padLen >= 0);
    return `${prefix}${'b'.repeat(padLen)}${suffix}`;
  };

  const hBink = fnv1a64('w239-hot-bink2');
  const hUi = fnv1a64('w239-hot-ui');
  const hCold1 = fnv1a64('w239-cold-1');
  const hCold2 = fnv1a64('w239-cold-2');
  const hCold3 = fnv1a64('w239-cold-3');
  const hOversized = fnv1a64('w239-oversized-pinned');

  const wgslBink = make100B('bink2');
  const wgslUi = make100B('ui');
  const wgslCold1 = make100B('c1');
  const wgslCold2 = make100B('c2');
  const wgslCold3 = make100B('c3');
  const wgslOversized = make100B('ov') + '/* ' + 'z'.repeat(300) + ' */';

  const packBuf = buildWgslPackBuffer([
    { hash: hBink, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslBink },
    { hash: hUi, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslUi },
    { hash: hCold1, stage: SHADER_STAGE.COMPUTE, wgsl: wgslCold1 },
    { hash: hCold2, stage: SHADER_STAGE.COMPUTE, wgsl: wgslCold2 },
    { hash: hCold3, stage: SHADER_STAGE.COMPUTE, wgsl: wgslCold3 },
    { hash: hOversized, stage: SHADER_STAGE.VERTEX, wgsl: wgslOversized },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 6 * WGPK_ENTRY_BYTES;
  const mock = createRangeMockFetch({ '/packs/w239-pin.wgsl.pack': packBuf });

  // Cache budget fits 3 x 100B entries (maxCacheBytes = 320, maxCacheEntries = 3)
  const pack = await loadIndexFromUrl('/packs/w239-pin.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    maxCacheBytes: 320,
    maxCacheEntries: 3,
  });

  // Pin hBink BEFORE fetching (via hex string + duplicate), and pin hUi AFTER fetching (promotes LRU -> pinned)
  assert.equal(pack.pinHashes([formatHashHex(hBink), hBink]), 1);
  assert.equal(pack.isPinned(hBink), true);
  assert.equal(pack.isPinned(hUi), false);
  assert.equal(pack.getStats().pinnedCount, 1);

  assert.equal(await pack.getWgslAsync(hBink), wgslBink);
  assert.equal(await pack.getWgslAsync(hUi), wgslUi);
  assert.equal(pack.pinHashes([hUi]), 1);
  assert.equal(pack.getStats().pinnedCount, 2);
  assert.equal(pack.getStats().pinnedBytes, 200);
  assert.equal(pack.getStats().pinnedCachedEntries, 2);
  assert.equal(pack.getStats().cacheEntries, 2);
  assert.equal(pack.getStats().cacheBytes, 200);
  assert.deepEqual(new Set(pack.getPinnedHashes()), new Set([hBink, hUi]));

  // Sync getWgsl() also works on pinned entries while resident
  assert.equal(pack.getWgsl(hBink), wgslBink);
  assert.equal(pack.getWgsl(hUi), wgslUi);

  // Stream cold1, cold2, cold3 through the remaining 1-slot / 120B unpinned headroom
  const reqsBeforeChurn = mock.requests.length;
  assert.equal(await pack.getWgslAsync(hCold1), wgslCold1); // fits (3 entries, 300B)
  assert.equal(pack.getStats().cacheEvictions, 0);
  assert.equal(await pack.getWgslAsync(hCold2), wgslCold2); // evicts cold1 (never bink/ui!)
  assert.equal(pack.getStats().cacheEvictions, 1);
  assert.equal(await pack.getWgslAsync(hCold3), wgslCold3); // evicts cold2 (never bink/ui!)
  assert.equal(pack.getStats().cacheEvictions, 2);
  assert.equal(mock.requests.length, reqsBeforeChurn + 3);

  // Hot Bink2 and UI shaders must still be 100% cache hits with ZERO HTTP re-fetches!
  const reqsAfterChurn = mock.requests.length;
  assert.equal(await pack.getWgslAsync(hBink), wgslBink);
  assert.equal(await pack.getWgslAsync(hUi), wgslUi);
  assert.equal(await pack.getWgslAsync(hCold3), wgslCold3);
  assert.equal(mock.requests.length, reqsAfterChurn);
  assert.equal(pack.getStats().cacheHits, 3);

  // clearWgslCache({ keepPinned: true }) drops only unpinned entries (cold3), preserving pinned bink & ui
  assert.equal(pack.clearWgslCache({ keepPinned: true }), 1);
  assert.equal(pack.getStats().cacheEntries, 2);
  assert.equal(pack.getStats().cacheBytes, 200);
  assert.equal(await pack.getWgslAsync(hBink), wgslBink);
  assert.equal(await pack.getWgslAsync(hUi), wgslUi);
  assert.equal(mock.requests.length, reqsAfterChurn);

  // Pinning an oversized shader (> maxCacheBytes) still keeps it resident when fetched,
  // and subsequent unpinned fetches never evict pinned shaders
  assert.equal(pack.pinHashes([hOversized]), 1);
  assert.equal(await pack.getWgslAsync(hOversized), wgslOversized);
  assert.equal(pack.getStats().pinnedCount, 3);
  assert.equal(pack.getStats().pinnedCachedEntries, 3);

  const reqsWithOversized = mock.requests.length;
  // Unpinned cold1 cannot fit alongside pinned entries (> 320B), returns WGSL cleanly without evicting pinned
  assert.equal(await pack.getWgslAsync(hCold1), wgslCold1);
  assert.equal(mock.requests.length, reqsWithOversized + 1);
  assert.equal(await pack.getWgslAsync(hBink), wgslBink);
  assert.equal(await pack.getWgslAsync(hUi), wgslUi);
  assert.equal(await pack.getWgslAsync(hOversized), wgslOversized);
  assert.equal(mock.requests.length, reqsWithOversized + 1);

  // Unpinning the oversized shader immediately drops it (exceeds maxCacheBytes) and increments cacheEvictions
  const evictionsBeforeUnpinOversized = pack.getStats().cacheEvictions;
  assert.equal(pack.unpinHashes([hOversized]), 1);
  assert.equal(pack.isPinned(hOversized), false);
  assert.equal(pack.getStats().pinnedCount, 2);
  assert.equal(pack.getStats().cacheEvictions, evictionsBeforeUnpinOversized + 1);
  assert.equal(pack.getStats().cacheBytes, 200);

  // Unpinning hBink moves it back into the evictable LRU pool so future pressure can evict it
  assert.equal(pack.unpinHashes([formatHashHex(hBink)]), 1);
  assert.equal(pack.unpinHashes([hBink]), 0); // already unpinned
  assert.equal(pack.isPinned(hBink), false);
  assert.equal(pack.getStats().pinnedCount, 1);

  // Load cold1, cold2, cold3 -> cold1 fits (ui pinned + bink + cold1 = 3), cold2 evicts bink!
  assert.equal(await pack.getWgslAsync(hCold1), wgslCold1);
  assert.equal(await pack.getWgslAsync(hCold2), wgslCold2);
  // hUi (still pinned) stays cached, while hBink was evicted and requires a fresh Range fetch
  const reqsBeforeReFetchBink = mock.requests.length;
  assert.equal(await pack.getWgslAsync(hUi), wgslUi);
  assert.equal(mock.requests.length, reqsBeforeReFetchBink);
  assert.equal(await pack.getWgslAsync(hBink), wgslBink);
  assert.equal(mock.requests.length, reqsBeforeReFetchBink + 1);

  // Null/undefined pinHashes and unpinHashes are safe no-ops returning 0
  assert.equal(pack.pinHashes(null), 0);
  assert.equal(pack.unpinHashes(undefined), 0);
});

test('W239 WgslPackLoader handles 1+ GiB (16,226-entry) ranged pack with pinHashes, prefetchHashes({ maxCoalesceGapBytes }), and full sanitized getStats()', async () => {
  const enc = new TextEncoder();
  const ENTRY_COUNT = 16226;
  const TOTAL_BYTES = 0x60000000; // 1.5 GiB (1,610,612,736 bytes)
  const TABLE_END = WGPK_HEADER_BYTES + ENTRY_COUNT * WGPK_ENTRY_BYTES; // 324,536 bytes

  const makeWgsl = (tag, targetBytes) => {
    const prefix = `@compute @workgroup_size(8) fn w239_big_${tag}() { /* `;
    const suffix = ' */ }';
    const padLen = targetBytes - enc.encode(prefix).byteLength - enc.encode(suffix).byteLength;
    assert.ok(padLen >= 0);
    return `${prefix}${'s'.repeat(padLen)}${suffix}`;
  };

  // Build 16,226-entry TOC with 2 clusters of nearby shaders separated by a large gap
  // (simulating real 1+ GiB pack ranged streaming under streamShaders=1).
  const entries = new Array(ENTRY_COUNT);
  let cursor = TABLE_END;
  for (let i = 0; i < ENTRY_COUNT - 6; i += 1) {
    entries[i] = {
      hash: BigInt(i + 1),
      stage: (i % 3) + 1,
      offset: cursor,
      length: 64,
    };
    cursor += 64;
  }

  // High-offset cluster 1 around ~1.0 GiB: hBink (120B), 80B gap, hUi (120B), contiguous hHud (120B)
  // High-offset cluster 2 around ~1.4 GiB (far beyond maxCoalesceGapBytes): hWorldA (120B), 200B gap, hWorldB (120B), 5000B gap, hWorldC (120B)
  const hBink = fnv1a64('w239-16226-bink');
  const hUi = fnv1a64('w239-16226-ui');
  const hHud = fnv1a64('w239-16226-hud');
  const hWorldA = fnv1a64('w239-16226-world-a');
  const hWorldB = fnv1a64('w239-16226-world-b');
  const hWorldC = fnv1a64('w239-16226-world-c');

  const wgslBink = makeWgsl('bink', 120);
  const wgslUi = makeWgsl('ui', 120);
  const wgslHud = makeWgsl('hud', 120);
  const wgslWorldA = makeWgsl('wa', 120);
  const wgslWorldB = makeWgsl('wb', 120);
  const wgslWorldC = makeWgsl('wc', 120);

  const offBink = 0x40000000; // 1.0 GiB
  const offUi = offBink + 120 + 80; // 80B gap
  const offHud = offUi + 120; // contiguous (0B gap)

  const offWorldA = 0x58000000; // ~1.375 GiB
  const offWorldB = offWorldA + 120 + 200; // 200B gap
  const offWorldC = offWorldB + 120 + 5000; // 5000B gap

  const specialEntries = [
    { hash: hBink, stage: SHADER_STAGE.FRAGMENT, offset: offBink, length: 120, wgsl: wgslBink },
    { hash: hUi, stage: SHADER_STAGE.FRAGMENT, offset: offUi, length: 120, wgsl: wgslUi },
    { hash: hHud, stage: SHADER_STAGE.VERTEX, offset: offHud, length: 120, wgsl: wgslHud },
    { hash: hWorldA, stage: SHADER_STAGE.COMPUTE, offset: offWorldA, length: 120, wgsl: wgslWorldA },
    { hash: hWorldB, stage: SHADER_STAGE.COMPUTE, offset: offWorldB, length: 120, wgsl: wgslWorldB },
    { hash: hWorldC, stage: SHADER_STAGE.COMPUTE, offset: offWorldC, length: 120, wgsl: wgslWorldC },
  ];
  for (let i = 0; i < specialEntries.length; i += 1) {
    entries[ENTRY_COUNT - 6 + i] = specialEntries[i];
  }

  const indexBytes = buildSyntheticLargeIndexBuffer(entries);
  const requests = [];
  const mockFetch = async (url, init = {}) => {
    const method = (init.method || 'GET').toUpperCase();
    const range = init.headers && (init.headers.Range || init.headers.range);
    if (method === 'HEAD') {
      requests.push({ method, range: null, status: 200, bytesSent: 0 });
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        headers: new Headers({
          'Content-Length': String(TOTAL_BYTES),
          'Accept-Ranges': 'bytes',
        }),
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }
    const [, sStr, eStr] = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(sStr);
    const end = Math.min(Number(eStr), TOTAL_BYTES - 1);
    const len = end - start + 1;
    const out = new Uint8Array(len);
    if (start < indexBytes.byteLength) {
      const copyEnd = Math.min(end + 1, indexBytes.byteLength);
      out.set(indexBytes.subarray(start, copyEnd), 0);
    }
    for (const sp of specialEntries) {
      const spStart = sp.offset;
      const spEnd = sp.offset + sp.length;
      if (spStart >= start && spEnd <= end + 1) {
        out.set(enc.encode(sp.wgsl), spStart - start);
      }
    }
    requests.push({ method, range, status: 206, start, end, bytesSent: len });
    return {
      ok: true,
      status: 206,
      statusText: 'Partial Content',
      headers: new Headers({
        'Content-Range': `bytes ${start}-${end}/${TOTAL_BYTES}`,
        'Content-Length': String(len),
        'Accept-Ranges': 'bytes',
      }),
      async arrayBuffer() {
        return out.buffer;
      },
    };
  };

  // Use WgslPackLoader with bounded maxCacheBytes + maxCacheEntries, pin BEFORE pack load
  const loader = new WgslPackLoader({
    fetch: mockFetch,
    streamThresholdBytes: 1024 * 1024,
    maxCacheBytes: 480,
    maxCacheEntries: 4,
  });

  // Pin hot Bink2 + UI shaders before loading the pack; _registerPack propagates them automatically!
  assert.equal(loader.pinHashes([hBink, formatHashHex(hUi)]), 2);
  assert.equal(loader.getStats().pinnedCount, 2);

  const pack = await loader.loadFromUrl('/packs/gta6-16226.wgsl.pack');
  assert.equal(pack.mode, 'ranged');
  assert.equal(pack.size, ENTRY_COUNT);
  assert.equal(pack.isPinned(hBink), true);
  assert.equal(pack.isPinned(hUi), true);

  // Coalesced prefetch with maxCoalesceGapBytes: 256 (deliberately unsorted input):
  //   Range 1: [hBink..hHud] (80B + 0B gaps <= 256B) -> span 440B
  //   Range 2: [hWorldA..hWorldB] (200B gap <= 256B) -> span 440B
  //   Range 3: [hWorldC] (5000B gap > 256B -> split!) -> span 120B
  const reqsBeforePrefetch = requests.length;
  const pf = await loader.prefetchHashes(
    [hWorldC, hHud, hWorldB, hBink, hWorldA, hUi],
    { maxCoalesceGapBytes: 256 }
  );
  assert.equal(pf.requested, 6);
  assert.equal(pf.prefetched, 6);
  assert.equal(pf.batches, 3);
  assert.equal(pf.bytesFetched, 440 + 440 + 120);
  assert.equal(requests.length, reqsBeforePrefetch + 3, '6 unsorted shaders must coalesce into 3 HTTP 206 requests');

  // Verify sorted coalesced byte ranges
  const pfRanges = requests.slice(reqsBeforePrefetch).map((r) => r.range).sort();
  assert.deepEqual(
    pfRanges,
    [
      `bytes=${offBink}-${offHud + 119}`,
      `bytes=${offWorldA}-${offWorldB + 119}`,
      `bytes=${offWorldC}-${offWorldC + 119}`,
    ].sort()
  );

  // Even though 6 x 120B = 720B > maxCacheBytes=480 and 6 > maxCacheEntries=4,
  // pinned hBink and hUi MUST remain resident with 0 extra HTTP fetches!
  const reqsAfterPrefetch = requests.length;
  assert.equal(await loader.getWgslAsync(hBink), wgslBink);
  assert.equal(await loader.getWgslAsync(hUi), wgslUi);
  assert.equal(requests.length, reqsAfterPrefetch);

  // Verify every W239 sanitized telemetry field on both pack.getStats() and loader.getStats()
  const packStats = pack.getStats();
  const loaderStats = loader.getStats();
  const expectedRangeBytes = TABLE_END + 440 + 440 + 120;

  for (const s of [packStats, loaderStats]) {
    assert.equal(s.cacheHits, 2);
    assert.equal(s.cacheMisses, 0);
    assert.equal(s.cacheEvictions, 2); // 6 prefetched into 4-entry/480B cap with 2 pinned => 2 unpinned evicted
    assert.equal(s.pinnedCount, 2);
    assert.equal(s.cacheEntries, 4); // 2 pinned + 2 newest unpinned
    assert.equal(s.cacheBytes, 480);
    assert.equal(s.prefetchCoalescedRanges, 3);
    assert.equal(s.rangeBytesFetched, expectedRangeBytes);

    for (const key of [
      'cacheHits',
      'cacheMisses',
      'cacheEvictions',
      'pinnedCount',
      'cacheEntries',
      'cacheBytes',
      'prefetchCoalescedRanges',
      'rangeBytesFetched',
    ]) {
      assert.equal(typeof s[key], 'number', `${key} must be a number`);
      assert.ok(Number.isSafeInteger(s[key]) && s[key] >= 0, `${key} must be a sanitized non-negative integer`);
    }
  }

  // inspect() remains JSON-serializable with all W239 telemetry fields
  const snapshot = loader.inspect();
  assert.doesNotThrow(() => JSON.stringify(snapshot));
  assert.equal(snapshot.stats.pinnedCount, 2);
  assert.equal(snapshot.stats.cacheEntries, 4);
  assert.equal(snapshot.stats.cacheBytes, 480);
  assert.equal(snapshot.stats.cacheEvictions, 2);
  assert.equal(snapshot.stats.prefetchCoalescedRanges, 3);
  assert.equal(snapshot.stats.rangeBytesFetched, expectedRangeBytes);

  // Unpin on loader propagates to underlying pack and updates pinnedCount
  assert.equal(loader.unpinHashes([hBink, hUi]), 2);
  assert.equal(loader.getStats().pinnedCount, 0);
  assert.equal(pack.getStats().pinnedCount, 0);
  assert.equal(loader.isPinned(hBink), false);

  // clear() resets pinned set on loader as well
  loader.pinHashes([hBink]);
  assert.equal(loader.getStats().pinnedCount, 1);
  loader.clear({ resetStats: true });
  assert.equal(loader.getStats().pinnedCount, 0);
});

// ---------------------------------------------------------------------------
// W259: Ranged WGPK ETag / Last-Modified / total-size validation,
//       cache: 'no-store', and bounded 1x stale-index auto-reload (#1120)
// ---------------------------------------------------------------------------

function createMutableRangePackMock(initialState) {
  const state = {
    bytes:
      initialState.bytes instanceof Uint8Array
        ? initialState.bytes
        : new Uint8Array(initialState.bytes),
    etag: initialState.etag ?? null,
    lastModified: initialState.lastModified ?? null,
    includeContentRangeTotal:
      initialState.includeContentRangeTotal !== undefined
        ? Boolean(initialState.includeContentRangeTotal)
        : true,
    beforeRequest: initialState.beforeRequest || null,
  };

  const requests = [];

  const mockFetch = async (url, init = {}) => {
    const method = (init && init.method ? String(init.method) : 'GET').toUpperCase();
    const headersIn = init && init.headers ? { ...init.headers } : {};
    const range = headersIn.Range || headersIn.range || null;
    const cacheMode = init ? init.cache : undefined;

    if (typeof state.beforeRequest === 'function') {
      const override = await state.beforeRequest({
        url,
        method,
        range,
        init,
        cache: cacheMode,
        requests,
        state,
      });
      if (override) {
        requests.push({
          url,
          method,
          range,
          cache: cacheMode,
          headers: headersIn,
          credentials: init ? init.credentials : undefined,
          status: override.status,
        });
        return override;
      }
    }

    const total = state.bytes.byteLength;
    const respHeaders = new Headers({ 'Accept-Ranges': 'bytes' });
    if (state.etag !== null) {
      respHeaders.set('ETag', state.etag);
    }
    if (state.lastModified !== null) {
      respHeaders.set('Last-Modified', state.lastModified);
    }

    if (method === 'HEAD') {
      respHeaders.set('Content-Length', String(total));
      requests.push({
        url,
        method,
        range: null,
        cache: cacheMode,
        headers: headersIn,
        credentials: init ? init.credentials : undefined,
        status: 200,
      });
      return {
        ok: true,
        status: 200,
        statusText: 'OK',
        headers: respHeaders,
        async arrayBuffer() {
          return new ArrayBuffer(0);
        },
      };
    }

    if (range) {
      const match = /^bytes=(\d+)-(\d+)$/i.exec(range.trim());
      const reqStart = Number(match[1]);
      const reqEnd = Number(match[2]);
      if (reqStart >= total || reqEnd < reqStart) {
        respHeaders.set('Content-Range', `bytes */${total}`);
        requests.push({
          url,
          method,
          range,
          cache: cacheMode,
          headers: headersIn,
          credentials: init ? init.credentials : undefined,
          status: 416,
        });
        return {
          ok: false,
          status: 416,
          statusText: 'Range Not Satisfiable',
          headers: respHeaders,
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      const clampedEnd = Math.min(reqEnd, total - 1);
      const slice = state.bytes.slice(reqStart, clampedEnd + 1);
      respHeaders.set(
        'Content-Range',
        `bytes ${reqStart}-${clampedEnd}/${state.includeContentRangeTotal ? total : '*'}`
      );
      respHeaders.set('Content-Length', String(slice.byteLength));
      requests.push({
        url,
        method,
        range,
        cache: cacheMode,
        headers: headersIn,
        credentials: init ? init.credentials : undefined,
        status: 206,
      });
      return {
        ok: true,
        status: 206,
        statusText: 'Partial Content',
        headers: respHeaders,
        async arrayBuffer() {
          return slice.buffer;
        },
      };
    }

    const copy = state.bytes.slice();
    respHeaders.set('Content-Length', String(copy.byteLength));
    requests.push({
      url,
      method,
      range: null,
      cache: cacheMode,
      headers: headersIn,
      credentials: init ? init.credentials : undefined,
      status: 200,
    });
    return {
      ok: true,
      status: 200,
      statusText: 'OK',
      headers: respHeaders,
      async arrayBuffer() {
        return copy.buffer;
      },
    };
  };

  return { fetch: mockFetch, requests, state };
}

test('W259 normal ranged fetch captures ETag/Last-Modified/totalSize, sends cache: no-store, preserves fetchInit, and performs 0 reloads', async () => {
  const hVs = fnv1a64('w259-normal-vs');
  const hFs = fnv1a64('w259-normal-fs');
  const packBuf = buildWgslPackBuffer([
    { hash: hVs, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hFs, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES;

  const mock = createMutableRangePackMock({
    bytes: packBuf,
    etag: '"splash-v1"',
    lastModified: 'Wed, 25 Nov 2026 17:00:00 GMT',
  });

  const controller = new AbortController();
  const pack = await loadIndexFromUrl('/packs/splash.wgsl.pack', {
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
    fetchInit: {
      signal: controller.signal,
      credentials: 'same-origin',
      headers: { 'X-Custom-Token': 'abc123' },
    },
  });

  assert.equal(pack.mode, 'ranged');
  assert.equal(pack.etag, '"splash-v1"');
  assert.equal(pack.lastModified, 'Wed, 25 Nov 2026 17:00:00 GMT');
  assert.equal(pack.totalSize, packBuf.byteLength);
  assert.equal(pack.getStats().staleIndexReloads, 0);
  assert.equal(pack.getStats().etag, '"splash-v1"');
  assert.equal(pack.getStats().lastModified, 'Wed, 25 Nov 2026 17:00:00 GMT');
  assert.equal(pack.getStats().totalSize, packBuf.byteLength);

  assert.equal(await pack.getWgslAsync(hVs), SAMPLE_VS_WGSL);
  assert.equal(await pack.getWgslAsync(hFs), SAMPLE_FS_WGSL);

  assert.equal(mock.requests.length, 3); // 1 header + 2 entry fetches, 0 reloads
  assert.equal(pack.getStats().staleIndexReloads, 0);
  for (const req of mock.requests) {
    assert.equal(req.cache, 'no-store', 'every ranged header and entry fetch must pass cache: no-store');
    assert.equal(req.headers['X-Custom-Token'], 'abc123', 'caller headers must be preserved');
    assert.equal(req.credentials, 'same-origin', 'caller fetchInit fields must be preserved');
  }

  // loadFromUrl with streamThresholdBytes also sends cache: 'no-store' on HEAD + Range probes
  const autoMock = createMutableRangePackMock({
    bytes: packBuf,
    etag: '"splash-v1"',
  });
  const autoPack = await loadFromUrl('/packs/splash.wgsl.pack', {
    fetch: autoMock.fetch,
    streamThresholdBytes: 32,
    initialRangeBytes: tableEnd,
  });
  assert.equal(autoPack.mode, 'ranged');
  assert.equal(autoMock.requests[0].method, 'HEAD');
  assert.equal(autoMock.requests[0].cache, 'no-store');
  assert.equal(autoMock.requests[1].status, 206);
  assert.equal(autoMock.requests[1].cache, 'no-store');
  assert.equal(autoPack.getStats().staleIndexReloads, 0);
});

test('W259 getWgslAsync auto-reloads stale WGPK index on ETag, total-size, mid-shader corruption, and HTTP 416/412 mismatches', async () => {
  const hA = fnv1a64('w259-swap-a');
  const hB = fnv1a64('w259-swap-b');
  const hC = fnv1a64('w259-swap-c');

  const wgslA_v1 = '@vertex fn vs_v1() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
  const wgslB_v1 = '@fragment fn fs_v1() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.0, 0.0, 1.0); }';
  const wgslC_v1 = '@compute @workgroup_size(1) fn cs_v1() {}';

  // In V2, shader A grows so that V1's byte offset for shader B lands right on a function-body
  // `let ...` statement (the exact CreateShaderModule: module-scope 'let' failure!), while
  // shader B shifts to a later offset with updated source.
  const enc = new TextEncoder();
  const wgslA_v1_len = enc.encode(wgslA_v1).byteLength;
  const v2Prefix = '@vertex fn vs_v2() -> @builtin(position) vec4<f32> { ';
  const padToOldBOffset = Math.max(0, wgslA_v1_len - enc.encode(v2Prefix).byteLength);
  const wgslA_v2 =
    `${v2Prefix}${' '.repeat(padToOldBOffset)}` +
    `let spliced_local = vec4<f32>(0.5, 0.5, 0.0, 1.0); return spliced_local; }`;
  const wgslB_v2 = '@fragment fn fs_v2() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.25, 1.0); }';
  const wgslC_v2 = '@compute @workgroup_size(8) fn cs_v2() {}';

  const packV1 = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: wgslA_v1 },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslB_v1 },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, wgsl: wgslC_v1 },
  ]);
  const packV2 = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: wgslA_v2 },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslB_v2 },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, wgsl: wgslC_v2 },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 3 * WGPK_ENTRY_BYTES;

  // Confirm that reading V1's offset for hB directly out of V2 would produce mid-shader `let ...` garbage!
  const parsedV1 = parseWgslPack(packV1);
  const oldBMeta = parsedV1.getByHash(hB);
  const splicedGarbage = new TextDecoder().decode(
    new Uint8Array(packV2).subarray(oldBMeta.offset, oldBMeta.offset + oldBMeta.length)
  );
  assert.ok(splicedGarbage.startsWith('let spliced_local'), `expected spliced 'let' garbage, got: ${splicedGarbage}`);

  // 1. ETag + size mismatch through WgslPackCache (verifies cache sync + GPUShaderModule eviction + pin preservation)
  const mock = createMutableRangePackMock({
    bytes: packV1,
    etag: '"splash-v1"',
    lastModified: 'Wed, 25 Nov 2026 17:00:00 GMT',
  });
  const cache = new WgslPackCache({
    fetch: mock.fetch,
    initialRangeBytes: tableEnd,
  });
  const pack = await cache.loadIndexFromUrl('/packs/splash.wgsl.pack');
  const gpu = createMockGpuDevice();

  // Pin & compile hA from V1 so it lives in both LRU pinned cache and GPUShaderModule cache.
  cache.pinHashes([hA]);
  const modA_v1 = await cache.getOrCreateShaderModuleAsync(gpu.device, hA, 'gta6');
  assert.equal(modA_v1.code, wgslA_v1);
  assert.equal(pack.getStats().staleIndexReloads, 0);

  // Swap splash.wgsl.pack on disk to V2 with a new ETag & shifted offsets!
  mock.state.bytes = new Uint8Array(packV2);
  mock.state.etag = '"splash-v2"';
  mock.state.lastModified = 'Wed, 25 Nov 2026 17:05:00 GMT';

  // Fetching hB detects the ETag/size mismatch, reloads the header ONCE, clears stale LRU,
  // updates baseline metadata, and returns valid wgslB_v2 from the refreshed offset!
  const gotB = await cache.getWgslAsync(hB);
  assert.equal(gotB, wgslB_v2);
  assert.equal(pack.etag, '"splash-v2"');
  assert.equal(pack.lastModified, 'Wed, 25 Nov 2026 17:05:00 GMT');
  assert.equal(pack.totalSize, packV2.byteLength);
  assert.equal(pack.getStats().staleIndexReloads, 1);
  assert.equal(cache.getStats().staleIndexReloads, 1);

  // Old cached hA WGSL and GPUShaderModule were invalidated, while hA stays pinned!
  assert.equal(pack.isPinned(hA), true);
  const modA_v2 = await cache.getOrCreateShaderModuleAsync(gpu.device, hA, 'gta6');
  assert.notEqual(modA_v2, modA_v1);
  assert.equal(modA_v2.code, wgslA_v2);
  assert.equal(pack.getStats().staleIndexReloads, 1, 'subsequent fetches against V2 must perform 0 extra reloads');

  // 2. Total-size mismatch only (no ETag / no Last-Modified headers)
  const sizeOnlyMock = createMutableRangePackMock({
    bytes: packV1,
    etag: null,
    lastModified: null,
  });
  const sizeOnlyPack = await loadIndexFromUrl('/packs/size-only.wgsl.pack', {
    fetch: sizeOnlyMock.fetch,
    initialRangeBytes: tableEnd,
  });
  assert.equal(sizeOnlyPack.etag, null);
  assert.equal(sizeOnlyPack.totalSize, packV1.byteLength);

  sizeOnlyMock.state.bytes = new Uint8Array(packV2);
  assert.equal(await sizeOnlyPack.getWgslAsync(hB), wgslB_v2);
  assert.equal(sizeOnlyPack.totalSize, packV2.byteLength);
  assert.equal(sizeOnlyPack.getStats().staleIndexReloads, 1);

  // 3. Same-size pack swap with NO ETag/Last-Modified/totalSize -> mid-shader `let ...` corruption triggers 1x reload!
  const sameSizePad = enc.encode(wgslA_v2).byteLength - wgslA_v1_len;
  const wgslB_sameSizeV1 = `${wgslB_v1}${' '.repeat(sameSizePad)}`;
  const sameSizePackV1 = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: wgslA_v1 },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslB_sameSizeV1 },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, wgsl: wgslC_v1 },
  ]);
  const sameSizePackV2 = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: wgslA_v2 },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslB_v1 },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, wgsl: wgslC_v1 },
  ]);
  assert.equal(sameSizePackV1.byteLength, sameSizePackV2.byteLength);

  const corruptDetectMock = createMutableRangePackMock({
    bytes: sameSizePackV1,
    etag: null,
    lastModified: null,
    includeContentRangeTotal: false,
  });
  const corruptDetectPack = await loadIndexFromUrl('/packs/same-size.wgsl.pack', {
    fetch: corruptDetectMock.fetch,
    initialRangeBytes: tableEnd,
  });
  corruptDetectMock.state.bytes = new Uint8Array(sameSizePackV2);
  assert.equal(await corruptDetectPack.getWgslAsync(hB), wgslB_v1);
  assert.equal(corruptDetectPack.getStats().staleIndexReloads, 1);

  // 4. Shrunk pack causing HTTP 416 Range Not Satisfiable (with Content-Range: bytes */<newTotal>) recovers via 1x reload
  const bigWgslA = `@vertex fn vs_big() -> @builtin(position) vec4<f32> { /* ${'a'.repeat(400)} */ return vec4<f32>(0.0); }`;
  const bigPack = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: bigWgslA },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslB_v1 },
  ]);
  const smallPack = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: wgslA_v1 },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslB_v2 },
  ]);
  const shrinkMock = createMutableRangePackMock({
    bytes: bigPack,
    etag: '"big-v1"',
  });
  const shrinkPack = await loadIndexFromUrl('/packs/shrink.wgsl.pack', {
    fetch: shrinkMock.fetch,
    initialRangeBytes: WGPK_HEADER_BYTES + 2 * WGPK_ENTRY_BYTES,
  });
  shrinkMock.state.bytes = new Uint8Array(smallPack);
  shrinkMock.state.etag = '"small-v2"';
  // Old offset of hB is past the end of smallPack -> server responds HTTP 416 with bytes */<smallPack.byteLength>
  assert.equal(await shrinkPack.getWgslAsync(hB), wgslB_v2);
  assert.ok(shrinkMock.requests.some((r) => r.status === 416), 'expected initial 416 before reload');
  assert.equal(shrinkPack.getStats().staleIndexReloads, 1);

  // 5. HTTP 412 Precondition Failed triggers 1x reload and recovers cleanly
  let send412Once = true;
  const mock412 = createMutableRangePackMock({
    bytes: packV1,
    etag: '"v1"',
    beforeRequest: ({ range, state }) => {
      if (send412Once && range && !range.startsWith('bytes=0-')) {
        send412Once = false;
        state.bytes = new Uint8Array(packV2);
        state.etag = '"v2"';
        return {
          ok: false,
          status: 412,
          statusText: 'Precondition Failed',
          headers: new Headers({ ETag: '"v2"' }),
          async arrayBuffer() {
            return new ArrayBuffer(0);
          },
        };
      }
      return null;
    },
  });
  const pack412 = await loadIndexFromUrl('/packs/412.wgsl.pack', {
    fetch: mock412.fetch,
    initialRangeBytes: tableEnd,
  });
  assert.equal(await pack412.getWgslAsync(hC), wgslC_v2);
  assert.equal(pack412.getStats().staleIndexReloads, 1);
});

test('W259 guards against infinite reload loops when ETag keeps changing and coalesces concurrent stale reloads', async () => {
  const hA = fnv1a64('w259-loop-a');
  const hB = fnv1a64('w259-loop-b');
  const hC = fnv1a64('w259-loop-c');

  const packV1 = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: SAMPLE_VS_WGSL },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: SAMPLE_FS_WGSL },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, wgsl: SAMPLE_CS_WGSL },
  ]);
  const tableEnd = WGPK_HEADER_BYTES + 3 * WGPK_ENTRY_BYTES;

  // 1. Server mutates ETag on EVERY single request -> must reload at most ONCE and throw (no infinite loop!)
  let etagCounter = 0;
  const flappingMock = createMutableRangePackMock({
    bytes: packV1,
    etag: '"rev-0"',
    beforeRequest: ({ state }) => {
      etagCounter += 1;
      state.etag = `"rev-${etagCounter}"`;
      return null;
    },
  });

  const flappingPack = await loadIndexFromUrl('/packs/flapping.wgsl.pack', {
    fetch: flappingMock.fetch,
    initialRangeBytes: tableEnd,
    maxRangeFetchRetries: 0,
  });
  assert.equal(flappingPack.etag, '"rev-1"');
  assert.equal(flappingMock.requests.length, 1);

  await assert.rejects(
    () => flappingPack.getWgslAsync(hA),
    /stale pack mismatch.*persisted after index reload/i
  );

  // Exactly 4 HTTP calls: initial index ("rev-1") + stale entry ("rev-2") + 1 reload ("rev-3") + 1 retry ("rev-4")
  assert.equal(flappingMock.requests.length, 4);
  assert.equal(flappingPack.getStats().staleIndexReloads, 1);
  assert.equal(flappingPack.getStats().inFlightCount, 0);
  assert.equal(flappingPack.getStats().cachedWgslEntries, 0, 'mismatched payload must never be cached');

  // 2. Concurrent fetches hitting a single pack swap coalesce onto exactly 1 header reload!
  const wgslA_v2 = `@vertex fn vs_v2_conc() -> @builtin(position) vec4<f32> { /* ${'z'.repeat(80)} */ return vec4<f32>(1.0); }`;
  const wgslB_v2 = '@fragment fn fs_v2_conc() -> @location(0) vec4<f32> { return vec4<f32>(0.2, 0.4, 0.6, 1.0); }';
  const wgslC_v2 = '@compute @workgroup_size(4) fn cs_v2_conc() {}';
  const packV2 = buildWgslPackBuffer([
    { hash: hA, stage: SHADER_STAGE.VERTEX, wgsl: wgslA_v2 },
    { hash: hB, stage: SHADER_STAGE.FRAGMENT, wgsl: wgslB_v2 },
    { hash: hC, stage: SHADER_STAGE.COMPUTE, wgsl: wgslC_v2 },
  ]);

  const concMock = createMutableRangePackMock({
    bytes: packV1,
    etag: '"conc-v1"',
  });
  const concPack = await loadIndexFromUrl('/packs/concurrent-swap.wgsl.pack', {
    fetch: concMock.fetch,
    initialRangeBytes: tableEnd,
  });

  // Swap pack on disk before 3 concurrent fetches fire
  concMock.state.bytes = new Uint8Array(packV2);
  concMock.state.etag = '"conc-v2"';

  const [resA, resB, resC] = await Promise.all([
    concPack.getWgslAsync(hA),
    concPack.getWgslAsync(hB),
    concPack.getWgslAsync(hC),
  ]);
  assert.equal(resA, wgslA_v2);
  assert.equal(resB, wgslB_v2);
  assert.equal(resC, wgslC_v2);
  assert.equal(
    concPack.getStats().staleIndexReloads,
    1,
    'concurrent stale fetches must coalesce onto a single index reload'
  );
  assert.equal(concPack.getStats().inFlightCount, 0);
});
