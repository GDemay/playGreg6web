"use strict";
// Fresh-process alternating samples; workload matches bench_wgsl_compact.cjs.
const { spawnSync } = require("node:child_process");
const crypto = require("node:crypto");
const os = require("node:os");
const path = require("node:path");
const { performance } = require("node:perf_hooks");
if (process.argv[2] === "--child") {
  const [file, count, size, calls] = process.argv.slice(3);
  const n = Number(count),
    target = Number(size);
  const declarations = Array.from(
    { length: n },
    (_, i) =>
      `@group(0) @binding(${i}) var<storage, read> cb_${i}: array<f32>;`,
  ).join("\n");
  const refs = Array.from(
    { length: n },
    (_, i) => `acc = acc + cb_${i}[0];`,
  ).join("\n");
  const statement = "acc = acc + 1.0;\n";
  const filler = statement.repeat(
    Math.max(
      1,
      Math.floor(
        (target - declarations.length - refs.length - 100) / statement.length,
      ),
    ),
  );
  const source = `${declarations}\n@fragment fn fs_main() -> @location(0) vec4<f32> {\nvar acc = 0.0;\n${filler + refs}\nreturn vec4<f32>(acc);\n}`;
  const fn = require(path.resolve(file)).compactWgslForMetadata;
  let output;
  for (let i = 0; i < 20; i++) output = fn(source);
  const cpuBefore = process.cpuUsage(),
    wallBefore = performance.now();
  for (let i = 0; i < Number(calls); i++) output = fn(source);
  const cpu = process.cpuUsage(cpuBefore);
  console.log(
    JSON.stringify({
      cpuMs: (cpu.user + cpu.system) / 1000,
      wallMs: performance.now() - wallBefore,
      sourceBytes: source.length,
      outputSha256: crypto.createHash("sha256").update(output).digest("hex"),
    }),
  );
} else {
  const [baseline, patch, samplesArg = "11", callsArg = "100"] =
    process.argv.slice(2);
  if (!baseline || !patch)
    throw Error("expected BASELINE_JS PATCH_JS [SAMPLES] [CALLS]");
  const samples = Number(samplesArg),
    calls = Number(callsArg);
  const result = {
    environment: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      cpu: os.cpus()[0]?.model,
      loadStart: os.loadavg(),
    },
    sampling: `${samples} alternating pairs of fresh Node processes, 20 warmups, ${calls} timed calls per process`,
    cases: [],
  };
  for (const [bindings, targetBytes] of [
    [0, 230000],
    [16, 5000],
    [16, 230000],
  ]) {
    const rows = [];
    for (let i = 0; i < samples; i++) {
      const row = {};
      for (const [label, file] of i % 2
        ? [
            ["patch", patch],
            ["base", baseline],
          ]
        : [
            ["base", baseline],
            ["patch", patch],
          ]) {
        const p = spawnSync(
          process.execPath,
          [
            __filename,
            "--child",
            path.resolve(file),
            String(bindings),
            String(targetBytes),
            String(calls),
          ],
          { encoding: "utf8" },
        );
        if (p.status !== 0) throw Error(p.stderr);
        row[label] = JSON.parse(p.stdout);
      }
      if (row.base.outputSha256 !== row.patch.outputSha256)
        throw Error("output mismatch");
      rows.push(row);
    }
    result.cases.push({ bindings, targetBytes, samples: rows });
  }
  result.environment.loadEnd = os.loadavg();
  console.log(JSON.stringify(result, null, 2));
}
