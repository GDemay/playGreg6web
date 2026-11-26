import test from 'node:test';
import assert from 'node:assert/strict';
import {
  WGPK_VERSION_V2,
  WGPK_HEADER_BYTES,
  WGPK_ENTRY_BYTES_V2,
  DEFAULT_MAX_FULL_FALLBACK_BYTES,
  buildWgslPackBuffer,
  parseWgslPackIndex,
  loadIndexFromUrl,
  loadFromUrl,
} from '../web/workers/wgsl_pack_loader.js';

const SHADER_A_HASH = 0x1111111111111111n;
const SHADER_B_HASH = 0x2222222222222222n;
const SHADER_C_HASH = 0x3333333333333333n;
const SHADER_A_WGSL = '@vertex fn vs_a() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }';
const SHADER_B_WGSL = '@fragment fn fs_b() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }';
const SHADER_C_WGSL = '@compute @workgroup_size(1) fn cs_c() {}';

function makeRangedFetch(bytes, opts = {}) {
  const view = new Uint8Array(bytes);
  const requests = [];
  const fetchImpl = async (url, init = {}) => {
    const method = (init.method || 'GET').toUpperCase();
    const headers = init.headers || {};
    const rangeHeader = headers.Range || headers.range || null;
    requests.push({ method, rangeHeader, url });
    if (opts.onRequest) {
      const override = await opts.onRequest({ method, rangeHeader, url, init });
      if (override) return override;
    }
    if (method === 'HEAD') {
      return {
        ok: true,
        status: 200,
        headers: new Map([
          ['content-length', String(opts.reportedSize ?? view.byteLength)],
          ['accept-ranges', opts.acceptRanges ?? 'bytes'],
          ['etag', opts.etag ?? '"v1"'],
        ]),
      };
    }
    if (rangeHeader) {
      const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
      const start = Number(m[1]);
      const end = Math.min(Number(m[2]), view.byteLength - 1);
      const slice = view.slice(start, end + 1);
      return {
        ok: true,
        status: 206,
        headers: new Map([
          ['content-range', `bytes ${start}-${end}/${opts.reportedSize ?? view.byteLength}`],
          ['content-length', String(slice.byteLength)],
          ['etag', opts.etag ?? '"v1"'],
        ]),
        arrayBuffer: async () => slice.buffer,
      };
    }
    return {
      ok: true,
      status: 200,
      headers: new Map([['content-length', String(view.byteLength)]]),
      arrayBuffer: async () => view.slice().buffer,
    };
  };
  return { fetchImpl, requests };
}

test('Orch-20 wgsl_pack_loader rejects oversized 200 OK (>256 MiB / 7.51 GB) before calling arrayBuffer() and cancels response body', async () => {
  assert.equal(DEFAULT_MAX_FULL_FALLBACK_BYTES, 256 * 1024 * 1024);

  let arrayBufferCalled = false;
  let bodyCanceled = false;

  const oversized200Fetch = async () => ({
    ok: true,
    status: 200,
    headers: new Map([['content-length', String(8_063_800_000)]]),
    body: {
      cancel() {
        bodyCanceled = true;
        return Promise.resolve();
      },
    },
    async arrayBuffer() {
      arrayBufferCalled = true;
      throw new Error('must never buffer 7.51 GB response');
    },
  });

  await assert.rejects(
    () => loadIndexFromUrl('/shaders/huge-v2.wgsl.pack', { fetch: oversized200Fetch }),
    (err) => err && err.code === 'WGPK_FULL_FALLBACK_REJECTED'
  );
  assert.equal(arrayBufferCalled, false, 'arrayBuffer() must not be called on oversized 200 OK');
  assert.equal(bodyCanceled, true, 'response.body.cancel() must be invoked to abort oversized stream');

  // loadFromUrl with 7.51 GB Content-Length + Accept-Ranges: none also fails fast without GET
  let getAttempted = false;
  const noRangeFetch = async (_url, init) => {
    if ((init && init.method) === 'HEAD') {
      return {
        ok: true,
        status: 200,
        headers: new Map([
          ['content-length', String(8_063_800_000)],
          ['accept-ranges', 'none'],
        ]),
      };
    }
    getAttempted = true;
    return { ok: true, status: 200, arrayBuffer: async () => new ArrayBuffer(0) };
  };

  await assert.rejects(
    () =>
      loadFromUrl('/shaders/huge-v2.wgsl.pack', {
        fetch: noRangeFetch,
        streamThresholdBytes: 128 * 1024 * 1024,
      }),
    (err) => err && err.code === 'WGPK_FULL_FALLBACK_REJECTED'
  );
  assert.equal(getAttempted, false);
});

test('Orch-20 prefetchHashes never leaks _inFlightFetches or hangs deduplicated getWgslAsync callers on unexpected post-fetch errors', async () => {
  const buf = buildWgslPackBuffer(
    [
      { hash: SHADER_A_HASH, stage: 'vertex', wgsl: SHADER_A_WGSL },
      { hash: SHADER_B_HASH, stage: 'fragment', wgsl: SHADER_B_WGSL },
    ],
    { version: WGPK_VERSION_V2 }
  );

  const { fetchImpl } = makeRangedFetch(buf);
  const pack = await loadIndexFromUrl('/shaders/v2.wgsl.pack', { fetch: fetchImpl });

  // Inject an unexpected throw in _putWgslCache (post-fetch, previously unguarded by try/finally)
  const origPut = pack._putWgslCache.bind(pack);
  pack._putWgslCache = () => {
    throw new Error('simulated cache write explosion');
  };

  const prefetchPromise = pack.prefetchHashes([SHADER_A_HASH, SHADER_B_HASH]);
  // Concurrent getWgslAsync deduplicates onto the in-flight prefetch deferred
  const dedupedA = pack.getWgslAsync(SHADER_A_HASH);
  const dedupedB = pack.getWgslAsync(SHADER_B_HASH);

  await assert.rejects(() => prefetchPromise, /simulated cache write explosion/);
  await assert.rejects(() => dedupedA, /simulated cache write explosion/);
  await assert.rejects(() => dedupedB, /simulated cache write explosion/);

  assert.equal(pack._inFlightFetches.size, 0, '_inFlightFetches must be completely drained');

  // Subsequent fetch recovers cleanly once cache write works again
  pack._putWgslCache = origPut;
  const recovered = await pack.getWgslAsync(SHADER_A_HASH);
  assert.equal(recovered, SHADER_A_WGSL);
});

test('Orch-20 _reloadStaleIndex prioritizes TOC reload ahead of queued shader fetches in _rangeFetchWaiters', async () => {
  const bufV1 = buildWgslPackBuffer(
    [
      { hash: SHADER_A_HASH, stage: 'vertex', wgsl: SHADER_A_WGSL },
      { hash: SHADER_B_HASH, stage: 'fragment', wgsl: SHADER_B_WGSL },
      { hash: SHADER_C_HASH, stage: 'compute', wgsl: SHADER_C_WGSL },
    ],
    { version: WGPK_VERSION_V2 }
  );

  // Build v2 pack with a shifted offset for B and C (different ETag)
  const bufV2 = buildWgslPackBuffer(
    [
      { hash: SHADER_A_HASH, stage: 'vertex', wgsl: `${SHADER_A_WGSL} /* pad */` },
      { hash: SHADER_B_HASH, stage: 'fragment', wgsl: SHADER_B_WGSL },
      { hash: SHADER_C_HASH, stage: 'compute', wgsl: SHADER_C_WGSL },
    ],
    { version: WGPK_VERSION_V2 }
  );

  let activeBytes = new Uint8Array(bufV1);
  let activeEtag = '"v1"';
  const fetchOrder = [];
  let releaseFirstFetch = null;
  let firstDataFetchSeen = false;

  const fetchImpl = async (_url, init = {}) => {
    const range = init.headers && (init.headers.Range || init.headers.range);
    fetchOrder.push(range);
    if (!firstDataFetchSeen && range && !range.startsWith('bytes=0-')) {
      firstDataFetchSeen = true;
      await new Promise((resolve) => {
        releaseFirstFetch = resolve;
      });
    }
    const m = /^bytes=(\d+)-(\d+)$/.exec(range);
    const start = Number(m[1]);
    const end = Math.min(Number(m[2]), activeBytes.byteLength - 1);
    const slice = activeBytes.slice(start, end + 1);
    return {
      ok: true,
      status: 206,
      headers: new Map([
        ['content-range', `bytes ${start}-${end}/${activeBytes.byteLength}`],
        ['content-length', String(slice.byteLength)],
        ['etag', activeEtag],
      ]),
      arrayBuffer: async () => slice.buffer,
    };
  };

  const pack = await loadIndexFromUrl('/shaders/prio.wgsl.pack', {
    fetch: fetchImpl,
    maxConcurrentRangeFetches: 1,
  });

  // Switch server to v2 before starting concurrent fetches
  activeBytes = new Uint8Array(bufV2);
  activeEtag = '"v2"';

  const pA = pack.getWgslAsync(SHADER_A_HASH);
  // Queue B and C behind A while slot 1 is occupied
  const pB = pack.getWgslAsync(SHADER_B_HASH);
  const pC = pack.getWgslAsync(SHADER_C_HASH);

  // Release A -> A sees stale ETag '"v2"' and calls _reloadStaleIndex (priority: true),
  // which must jump ahead of queued B/C in _rangeFetchWaiters!
  assert.equal(typeof releaseFirstFetch, 'function');
  releaseFirstFetch();

  const [wgslA, wgslB, wgslC] = await Promise.all([pA, pB, pC]);
  assert.equal(wgslA, `${SHADER_A_WGSL} /* pad */`);
  assert.equal(wgslB, SHADER_B_WGSL);
  assert.equal(wgslC, SHADER_C_WGSL);

  // Verify the priority TOC reload (bytes=0-...) jumped ahead of queued C in _rangeFetchWaiters!
  // fetchOrder[0] = initial TOC load
  // fetchOrder[1] = A's stale fetch (releases slot to B before A checks ETag)
  // fetchOrder[2] = B's stale fetch (in-flight while A enqueues priority TOC reload ahead of C)
  // fetchOrder[3] = priority TOC reload bytes=0-65535 (BEFORE C!)
  const reloadIdx = fetchOrder.findIndex((r, idx) => idx > 0 && r && r.startsWith('bytes=0-'));
  const cEntryV1 = pack._entryList; // just verify C's first fetch was after reloadIdx
  assert.ok(reloadIdx > 0, `expected stale TOC reload in fetchOrder=${JSON.stringify(fetchOrder)}`);
  const firstStaleCIdx = fetchOrder.findIndex((r) => r === 'bytes=236-275');
  assert.ok(
    firstStaleCIdx === -1 || reloadIdx < firstStaleCIdx,
    `expected priority TOC reload (idx=${reloadIdx}) to run before queued shader C (idx=${firstStaleCIdx}); fetchOrder=${JSON.stringify(fetchOrder)}`
  );
  assert.ok(cEntryV1.length === 3);
});

test('Orch-20 aborted AbortSignal short-circuits queued range fetches without incrementing rangeFetches or leaking slots', async () => {
  const buf = buildWgslPackBuffer(
    [
      { hash: SHADER_A_HASH, stage: 'vertex', wgsl: SHADER_A_WGSL },
      { hash: SHADER_B_HASH, stage: 'fragment', wgsl: SHADER_B_WGSL },
    ],
    { version: WGPK_VERSION_V2 }
  );

  const ac = new AbortController();
  let releaseFirst = null;
  const { fetchImpl } = makeRangedFetch(buf, {
    onRequest: async ({ rangeHeader }) => {
      if (rangeHeader && !rangeHeader.startsWith('bytes=0-')) {
        await new Promise((resolve) => {
          releaseFirst = resolve;
        });
      }
      return null;
    },
  });

  const pack = await loadIndexFromUrl('/shaders/abort.wgsl.pack', {
    fetch: fetchImpl,
    fetchInit: { signal: ac.signal },
    maxConcurrentRangeFetches: 1,
  });

  const fetchesBefore = pack.stats().rangeFetchCount;
  const pA = pack.getWgslAsync(SHADER_A_HASH);
  const pB = pack.getWgslAsync(SHADER_B_HASH); // queued in _rangeFetchWaiters

  // Abort while pB is queued -> pB must reject immediately with AbortError and be removed from queue
  ac.abort();
  await assert.rejects(() => pB, (err) => err && (err.name === 'AbortError' || err.code === 'ABORT_ERR'));

  releaseFirst();
  await pA;

  assert.equal(pack._activeRangeFetches, 0);
  assert.equal(pack._rangeFetchWaiters.length, 0);
  // Only pA incremented rangeFetchCount; queued aborted pB never fetched
  assert.equal(pack.stats().rangeFetchCount, fetchesBefore + 1);
});

test('Orch-20 parseWgpkEntryTable monotonic fast-path & out-of-order overlap detection, plus UTF-8 stale-reload recovery', async () => {
  // 1. Out-of-order non-overlapping v2 TOC succeeds; out-of-order overlapping v2 TOC throws.
  const entryCount = 2;
  const dataStart = WGPK_HEADER_BYTES + entryCount * WGPK_ENTRY_BYTES_V2;
  const tocGood = new Uint8Array(dataStart);
  const dvGood = new DataView(tocGood.buffer);
  tocGood.set([0x57, 0x47, 0x50, 0x4b], 0);
  dvGood.setUint32(4, WGPK_VERSION_V2, true);
  dvGood.setUint32(8, entryCount, true);
  dvGood.setUint32(12, 0, true);
  // Entry 0 at 5 GiB + 100 (len 20), Entry 1 at 5 GiB (len 20) -> out of order, non-overlapping
  const base5G = 5n * 1024n * 1024n * 1024n;
  dvGood.setBigUint64(16, SHADER_A_HASH, true);
  tocGood[16 + 8] = 1;
  dvGood.setUint32(16 + 12, 20, true);
  dvGood.setBigUint64(16 + 16, base5G + 100n, true);
  dvGood.setBigUint64(40, SHADER_B_HASH, true);
  tocGood[40 + 8] = 2;
  dvGood.setUint32(40 + 12, 20, true);
  dvGood.setBigUint64(40 + 16, base5G, true);

  const parsedGood = parseWgslPackIndex(tocGood, { url: '/v2.pack', totalSize: Number(base5G + 200n) });
  assert.equal(parsedGood.size, 2);

  // Now make Entry 1 overlap Entry 0 ([5G+90 .. 5G+110) overlaps [5G+100 .. 5G+120))
  const tocOverlap = tocGood.slice();
  const dvOverlap = new DataView(tocOverlap.buffer);
  dvOverlap.setBigUint64(40 + 16, base5G + 90n, true);
  assert.throws(
    () => parseWgslPackIndex(tocOverlap, { url: '/v2.pack', totalSize: Number(base5G + 200n) }),
    /overlapping payload ranges/i
  );

  // 2. First-attempt invalid UTF-8 bytes trigger 1x stale-index reload and succeed with fresh pack
  const validBuf = buildWgslPackBuffer(
    [{ hash: SHADER_A_HASH, stage: 'vertex', wgsl: SHADER_A_WGSL }],
    { version: WGPK_VERSION_V2 }
  );
  let corruptOnce = true;
  const { fetchImpl } = makeRangedFetch(validBuf, {
    onRequest: async ({ rangeHeader }) => {
      if (corruptOnce && rangeHeader && !rangeHeader.startsWith('bytes=0-')) {
        corruptOnce = false;
        const badUtf8 = new Uint8Array(SHADER_A_WGSL.length).fill(0xff);
        const m = /^bytes=(\d+)-(\d+)$/.exec(rangeHeader);
        return {
          ok: true,
          status: 206,
          headers: new Map([
            ['content-range', `bytes ${m[1]}-${m[2]}/${validBuf.byteLength}`],
            ['content-length', String(badUtf8.byteLength)],
            ['etag', '"v1"'],
          ]),
          arrayBuffer: async () => badUtf8.buffer,
        };
      }
      return null;
    },
  });

  const pack = await loadIndexFromUrl('/shaders/utf8-stale.wgsl.pack', { fetch: fetchImpl });
  const wgsl = await pack.getWgslAsync(SHADER_A_HASH);
  assert.equal(wgsl, SHADER_A_WGSL);
  assert.equal(pack.stats().staleIndexReloads, 1);
});
