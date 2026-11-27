import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const { RangeCache, coalesceAdjacentRanges, normalizeRelPath } = require('../web/workers/range_cache.js');

// Original containment accounting is deliberately independent of the sorted lookup.
function reference(cache, raw, gap, limit) {
  const entries = [];
  for (const entry of raw) {
    if (!entry || typeof entry.path !== 'string') continue;
    const offset = Number.isInteger(entry.offset) && entry.offset >= 0 ? entry.offset : 0;
    const length = Number.isInteger(entry.length) && entry.length > 0 ? entry.length : 0;
    if (!length) continue;
    try { entries.push({ path: normalizeRelPath(entry.path), offset, length }); } catch {}
  }
  return coalesceAdjacentRanges(entries, gap, limit).map((span) => {
    let total = 0;
    for (const e of entries) {
      if (e.path === span.path && e.offset >= span.offset && e.offset + e.length <= span.end) {
        total += cache._chunkAlignedSpanBytes(e.offset, e.length);
      }
    }
    const rounding = Math.max(0, total - cache._chunkAlignedSpanBytes(span.offset, span.length));
    return { ...span, savedBytes: span.mergedCount > 1
      ? Math.max(0, Math.max(span.overlapBytes, rounding) + (span.mergedCount - 1) * 256 - span.gapBytes)
      : 0 };
  });
}

test('coalescing accounting matches containment across oversized overlapping spans and invalid inputs', () => {
  const cache = new RangeCache({ chunkSize: 7, fetchFn: async () => { throw Error('no fetch'); } });
  const raw = [
    { path: '/a', offset: 0, length: 10000 },
    { path: 'a', offset: 8, length: 3 }, { path: 'a', offset: 9, length: 2 },
    { path: 'a', offset: 90, length: 30000 }, { path: 'a', offset: 100, length: 1 },
    { path: 'b', offset: 8, length: 3 }, { path: '../a', offset: 0, length: 4 },
    null, {}, { path: 'a', offset: -1, length: 2 },
    { path: 'a', offset: 2 ** 54, length: 1 }, { path: 'a', offset: 2 ** 54, length: 2 },
    { path: 'a', offset: 0, length: Number.MAX_SAFE_INTEGER },
  ];
  for (const limit of [10, 1000, Number.MAX_SAFE_INTEGER]) {
    assert.deepEqual(cache._planCoalescedSpans(raw, 10, limit), reference(cache, raw, 10, limit));
  }
});

test('coalescing accounting matches the original planner for shuffled randomized manifests', () => {
  let seed = 937;
  const random = (n) => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed % n; };
  const cache = new RangeCache({ chunkSize: 257, fetchFn: async () => { throw Error('no fetch'); } });
  for (let batch = 0; batch < 200; batch += 1) {
    const raw = Array.from({ length: 200 }, () => ({
      path: ['a', '/b', 'c'][random(3)], offset: random(100000), length: random(10000),
    }));
    const gap = random(1000);
    const limit = 1 + random(15000);
    assert.deepEqual(cache._planCoalescedSpans(raw, gap, limit), reference(cache, raw, gap, limit));
  }
});
