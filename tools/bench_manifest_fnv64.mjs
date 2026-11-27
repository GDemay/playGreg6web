#!/usr/bin/env node
// Synthetic in-memory bytes; this does not measure disk, network, browser or boot.
// node tools/bench_manifest_fnv64.mjs BASELINE_CHECKOUT PATCH_CHECKOUT
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';
import { execFileSync } from 'node:child_process';
import os from 'node:os';
import assert from 'node:assert/strict';
const roots = process.argv.slice(2);
if (roots.length !== 2) throw new Error('Provide baseline and patch checkout directories');
const implementations = await Promise.all(roots.map((root) => import(pathToFileURL(resolve(root, 'web/storage/manifest_verifier.js')))));
const rounds = 9;
const warmups = 3;
const loadAverageStart = os.loadavg();
const cases = [];
for (const { size, algorithm, control } of [
  { size: 1024 * 1024, algorithm: 'sha256', control: true },
  { size: 1024 * 1024, algorithm: 'fnv1a64', control: false },
  { size: 16 * 1024 * 1024, algorithm: 'fnv1a64', control: false },
]) {
  const bytes = new Uint8Array(size);
  let seed = 12345;
  for (let i = 0; i < size; i++) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; bytes[i] = seed >>> 24; }
  const expected = implementations[0].computeDigest(bytes, algorithm);
  for (const mode of control ? ['computeDigest'] : ['computeDigest', 'incremental-64KiB']) {
    const run = (mod) => {
      if (mode === 'computeDigest') return mod.computeDigest(bytes, algorithm);
      const verifier = mod.createIncrementalVerifier({ algorithm, expectedSize: size, expectedDigest: expected });
      for (let at = 0; at < size; at += 65536) verifier.update(bytes.subarray(at, at + 65536));
      const result = verifier.finalize();
      assert.equal(result.ok, true);
      return result.digest;
    };
    const samplesMs = [[], []];
    const cpuMs = [[], []];
    for (const mod of implementations) for (let i = 0; i < warmups; i++) assert.equal(run(mod), expected);
    for (let round = 0; round < rounds; round++) {
      for (const index of round % 2 ? [1, 0] : [0, 1]) {
        const cpuStart = process.cpuUsage();
        const start = performance.now();
        const digest = run(implementations[index]);
        const elapsed = performance.now() - start;
        const cpuElapsed = process.cpuUsage(cpuStart);
        assert.equal(digest, expected);
        samplesMs[index].push(elapsed);
        cpuMs[index].push((cpuElapsed.user + cpuElapsed.system) / 1000);
      }
    }
    cases.push({ control, algorithm, bytes: size, mode, digest: expected, baselineMs: samplesMs[0], patchMs: samplesMs[1], baselineCpuMs: cpuMs[0], patchCpuMs: cpuMs[1] });
  }
}
console.log(JSON.stringify({ benchmark: 'synthetic FNV64-selected manifest verifier APIs', environment: { node: process.version, platform: process.platform, arch: process.arch, os: os.release(), cpu: os.cpus()[0].model }, commits: roots.map((root) => execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()), rounds, warmups, primaryMetric: 'process CPU time (user + system), ms per sample; includes process threads and GC', loadAverageStart, loadAverageEnd: os.loadavg(), sampling: 'alternating baseline/patch order in one process; deterministic fixture generation outside timing; warm in-memory bytes; wall samples retained; unchanged SHA256 1MiB control', cases }, null, 2));
