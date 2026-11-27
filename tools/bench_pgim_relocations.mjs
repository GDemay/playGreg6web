#!/usr/bin/env node
// Synthetic containers only; no game data, browser or native-runtime timing.
// node tools/bench_pgim_relocations.mjs BASELINE_CHECKOUT PATCH_CHECKOUT
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import assert from 'node:assert/strict';
const require = createRequire(import.meta.url);
const roots = process.argv.slice(2);
if (roots.length !== 2) throw new Error('Provide baseline and patch checkout directories');
const implementations = roots.map((root) => require(resolve(root, 'web/runtime/pgim_preloader.js')).parsePgimLayout);
const rounds = 9;
const warmups = 3;
const loadAverageStart = os.loadavg();
function fixture(count, sectionCount) {
  const sectionSize = Math.max(4096, 16 * Math.ceil(count / sectionCount));
  const size = 64 + sectionCount * 64 + count * 4;
  const bytes = new Uint8Array(size);
  const view = new DataView(bytes.buffer);
  bytes.set([0x50, 0x47, 0x49, 0x4d]);
  for (const [at, value] of [[4, 1], [8, 64], [12, 64], [20, 4096], [32, sectionSize * sectionCount], [36, 4096], [44, sectionCount], [48, Math.ceil(count / 2)], [52, Math.floor(count / 2)]]) view.setUint32(at, value, true);
  view.setBigUint64(24, 0x140000000n, true);
  view.setBigUint64(56, BigInt(size), true);
  for (let i = 0; i < sectionCount; i++) {
    const at = 64 + i * 64;
    bytes.set(new TextEncoder().encode(`.s${i}`), at);
    view.setUint32(at + 16, i * sectionSize, true);
    view.setUint32(at + 20, sectionSize, true);
    view.setUint32(at + 24, 3, true);
    view.setBigUint64(at + 48, 0xcbf29ce484222325n, true);
  }
  let at = 64 + sectionCount * 64;
  for (let i = 0; i < count; i += 2, at += 4) view.setUint32(at, i * 16, true);
  for (let i = 1; i < count; i += 2, at += 4) view.setUint32(at, i * 16, true);
  return bytes;
}
const cases = [];
for (const [count, sections] of [[0, 8], [100000, 8], [1000000, 8], [1000000, 32]]) {
  const bytes = fixture(count, sections);
  const before = implementations[0](bytes);
  const after = implementations[1](bytes);
  assert.deepEqual(after, before);
  const iterationsPerSample = count === 0 ? 1000 : 1;
  const run = (fn) => {
    let result;
    for (let iteration = 0; iteration < iterationsPerSample; iteration++) result = fn(bytes);
    return result;
  };
  const samplesMs = [[], []];
  const cpuMs = [[], []];
  for (const fn of implementations) for (let i = 0; i < warmups; i++) run(fn);
  for (let round = 0; round < rounds; round++) {
    for (const index of round % 2 ? [1, 0] : [0, 1]) {
      const cpuStart = process.cpuUsage();
      const start = performance.now();
      const result = run(implementations[index]);
      const elapsed = performance.now() - start;
      const cpuElapsed = process.cpuUsage(cpuStart);
      assert.equal(result.dir64.length + result.highlow.length, count);
      samplesMs[index].push(elapsed);
      cpuMs[index].push((cpuElapsed.user + cpuElapsed.system) / 1000);
    }
  }
  cases.push({ control: count === 0, iterationsPerSample, relocations: count, sections, containerBytes: bytes.length, checksumVerification: false, correctness: 'deep-equal complete parsed layout', baselineMs: samplesMs[0], patchMs: samplesMs[1], baselineCpuMs: cpuMs[0], patchCpuMs: cpuMs[1] });
}
console.log(JSON.stringify({ benchmark: 'synthetic parsePgimLayout, mixed DIR64/HIGHLOW sites in zero-initialized virtual sections', environment: { node: process.version, platform: process.platform, arch: process.arch, os: os.release(), cpu: os.cpus()[0].model }, commits: roots.map((root) => execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()), rounds, warmups, primaryMetric: 'process CPU time (user + system), ms per sample; includes process threads and GC', loadAverageStart, loadAverageEnd: os.loadavg(), sampling: 'alternating baseline/patch order in one process; fixture creation and correctness comparison outside timing; warm in-memory buffers; wall samples retained; zero-relocation control parses 1000 containers per sample', cases }, null, 2));
