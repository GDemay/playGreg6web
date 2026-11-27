#!/usr/bin/env node
'use strict';
// Synthetic CPU benchmark of the production bootset coalescing planner; no I/O or game data.
const assert = require('node:assert/strict');
const path = require('node:path');
const { performance } = require('node:perf_hooks');
const os = require('node:os');
const args = process.argv.slice(2);
function option(name, fallback) {
  const i = args.indexOf(name);
  return i < 0 ? fallback : args[i + 1];
}
const baseline = option('--baseline');
if (!baseline) throw Error('usage: node tools/bench_range_planner.cjs --baseline <baseline range_cache.js> [--samples 9]');
const samples = Number(option('--samples', 9));
if (!Number.isInteger(samples) || samples < 3) throw Error('--samples must be an integer >= 3');
const Before = require(path.resolve(baseline)).RangeCache;
const After = require('../web/workers/range_cache.js').RangeCache;
const create = (Type) => new Type({ fetchFn: async () => { throw Error('unexpected I/O'); } });
const before = create(Before);
const after = create(After);
const results = [];
for (const [name, spans, entriesPerSpan] of [
  ['small paired bootset control', 32, 2],
  ['dispersed paired bootset', 10000, 2],
  ['dispersed singleton bootset', 10000, 1],
]) {
  const entries = [];
  for (let i = 0; i < spans; i += 1) {
    for (let j = 0; j < entriesPerSpan; j += 1) {
      entries.push({ path: 'synthetic/archive', offset: i * 4 * 1024 * 1024 + j * 32768, length: 65536 });
    }
  }
  // Reverse order exercises sorting rather than assuming the manifest is ordered.
  entries.reverse();
  const plan = (cache) => cache._planCoalescedSpans(entries, 65536, 2 * 1024 * 1024);
  assert.deepEqual(plan(after), plan(before));
  for (let i = 0; i < 3; i += 1) { plan(before); plan(after); }
  const times = { before: [], after: [] };
  for (let i = 0; i < samples; i += 1) {
    for (const label of i % 2 ? ['after', 'before'] : ['before', 'after']) {
      const start = performance.now();
      const output = plan(label === 'before' ? before : after);
      times[label].push(performance.now() - start);
      assert.equal(output.length, spans);
    }
  }
  results.push({ workload: name, entries: entries.length, spans, samplesMs: times });
}
console.log(JSON.stringify({
  benchmark: 'synthetic production RangeCache._planCoalescedSpans CPU time',
  environment: { node: process.version, platform: process.platform, arch: process.arch, cpu: os.cpus()[0].model },
  sampling: '3 untimed warmups; alternating before/after order; exact deep equality before timing; one plan per sample',
  unmeasured: 'browser boot time, HTTP/OPFS I/O, native runtime and game workloads',
  results,
}, null, 2));
