"use strict";
// Synthetic compactWgslForMetadata comparison. Usage:
// node tools/bench_wgsl_compact.cjs BASELINE_JS PATCH_JS [SAMPLES] [CALLS]
const { performance } = require("node:perf_hooks");
const os = require("node:os");
const path = require("node:path");
const [baselinePath, patchPath, samplesArg = "9", callsArg = "20"] =
  process.argv.slice(2);
if (!baselinePath || !patchPath)
  throw new Error("expected BASELINE_JS PATCH_JS [SAMPLES] [CALLS]");
const samples = Number(samplesArg),
  calls = Number(callsArg);
if (![samples, calls].every((v) => Number.isInteger(v) && v > 0))
  throw new Error("sample and call counts must be positive integers");
const base = require(path.resolve(baselinePath)).compactWgslForMetadata;
const patch = require(path.resolve(patchPath)).compactWgslForMetadata;

function makeSource(bindingCount, targetBytes, referencesAtEnd, variant) {
  const declarations = Array.from(
    { length: bindingCount },
    (_, i) =>
      `@group(0) @binding(${i}) var<storage, read> cb_${i}: array<f32>;`,
  ).join("\n");
  const refs =
    variant === "unused"
      ? ""
      : Array.from(
          { length: bindingCount },
          (_, i) => `acc = acc + cb_${i}[0];`,
        ).join("\n");
  const statement =
    variant === "prefix" ? "acc = acc + cb_0_suffix;\n" : "acc = acc + 1.0;\n";
  const filler = statement.repeat(
    Math.max(
      1,
      Math.floor(
        (targetBytes - declarations.length - refs.length - 100) /
          statement.length,
      ),
    ),
  );
  return `${declarations}\n@fragment fn fs_main() -> @location(0) vec4<f32> {\nvar acc = 0.0;\n${referencesAtEnd ? filler + refs : refs + filler}\nreturn vec4<f32>(acc);\n}`;
}
const workloads = [
  { bindings: 0, targetBytes: 230000, position: "late" },
  { bindings: 4, targetBytes: 230000, position: "late" },
  { bindings: 16, targetBytes: 230000, position: "late" },
  { bindings: 32, targetBytes: 230000, position: "late" },
  { bindings: 16, targetBytes: 230000, position: "early" },
  { bindings: 16, targetBytes: 5000, position: "late" },
  { bindings: 16, targetBytes: 230000, position: "late", variant: "prefix" },
  { bindings: 16, targetBytes: 230000, position: "late", variant: "unused" },
  { bindings: 65, targetBytes: 230000, position: "late" },
];
const result = {
  workload: "Synthetic WGSL metadata compaction; no device or game data",
  environment: {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpu: os.cpus()[0]?.model || "unknown",
    loadStart: os.loadavg(),
  },
  sampling: `${samples} alternating baseline/patch pairs; ${calls} calls per sample; five warmups per variant`,
  cases: [],
};
for (const workload of workloads) {
  const source = makeSource(
    workload.bindings,
    workload.targetBytes,
    workload.position === "late",
    workload.variant,
  );
  const expected = base(source);
  if (patch(source) !== expected)
    throw new Error(`metadata mismatch for ${JSON.stringify(workload)}`);
  for (let i = 0; i < 5; i++) {
    base(source);
    patch(source);
  }
  const rows = [];
  for (let i = 0; i < samples; i++) {
    const row = {};
    for (const [label, fn] of i % 2
      ? [
          ["patch", patch],
          ["base", base],
        ]
      : [
          ["base", base],
          ["patch", patch],
        ]) {
      const beforeCpu = process.cpuUsage();
      const beforeWall = performance.now();
      let out;
      for (let j = 0; j < calls; j++) out = fn(source);
      const cpu = process.cpuUsage(beforeCpu);
      if (out !== expected)
        throw new Error("metadata changed during measurement");
      row[label] = {
        cpuMs: (cpu.user + cpu.system) / 1000,
        wallMs: performance.now() - beforeWall,
      };
    }
    rows.push(row);
  }
  result.cases.push({
    ...workload,
    sourceBytes: source.length,
    stubBytes: expected.length,
    samples: rows,
  });
}
result.environment.loadEnd = os.loadavg();
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
