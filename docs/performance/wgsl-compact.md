# WGSL metadata compaction benchmark

These measurements compare the production `compactWgslForMetadata` function at baseline commit `52918dfa4d41e9f316990b06b713219dd9135aac` (worker SHA-256 `0fd46059cc39ab91537251d8ae456b368916dfef14f692127ca5820f7a79702d`) with the change in this branch (worker SHA-256 `1c9d7db31ccc695b87cd200f0951a424e044218b02a91c97b1f6d0f49596bc6c`). They use entirely synthetic text: no game shader or game data. The prefix-collision workload deliberately contains an undefined identifier; none of the text is submitted to a WGSL compiler or GPU.

The real worker calls compaction from `_compileShaderModuleEntry` for non-builtin shaders over 2,048 characters; the default compilation limit is 256 KiB. The roughly 230 KB inputs are below that limit, but actual shader-size and binding distributions have not been measured. These are compact-function timings, not end-to-end load, shader compilation, rendering, or frame-rate timings.

The host was an Apple M5, macOS arm64, Node v24.21.0 and Chrome 158.0.8127.89. It was not isolated from normal system processes; raw files include load averages. No builds ran during timing. For each synthetic workload, both variants return exactly the same compact string. A seeded differential check also compares 5,000 generated inputs including duplicate and overlapping identifiers, comments and identifier boundaries; the targeted unit tests check downstream binding and stage metadata. Median times below are milliseconds per 100 calls; ranges are the minimum and maximum of 11 samples. All measured calls run sequentially and variant order alternates by sample.

| Measurement | Baseline median [range] | Change median [range] | Change |
| --- | ---: | ---: | ---: |
| Chrome, 230 KB, 16 late references | 70.750 [69.580–71.595] | 32.175 [31.855–32.485] | −38.575 ms (−54.52%) |
| Chrome, 230 KB, 16 early references | 26.385 [26.205–26.900] | 26.495 [26.325–30.740] | +0.110 ms (+0.42%) |
| Chrome, 5 KB, 16 late references | 1.565 [1.510–1.650] | 1.920 [1.505–2.070] | +0.355 ms (+22.68%, +0.00355 ms/call) |
| Fresh Node processes, 230 KB, 16 late references | 81.794 [78.533–111.627] | 55.069 [51.419–65.560] | −26.725 ms (−32.67%) CPU |
| Fresh Node processes, 5 KB, 16 late references | 3.039 [2.794–3.357] | 3.093 [2.667–3.445] | +0.054 ms (+1.78%) CPU |

Chrome runs two separately loaded page contexts in one browser; the V8 function runs on the page, not in a Web Worker. Each timed invocation has five warmup calls. The baseline-versus-baseline Chrome check showed +1.27% median drift on the 5 KB case. A prior same-patch Chrome run showed 1.590 → 1.970 ms (+23.90%) on this case. The separate-process Node experiment starts a new process and loads the requested source for every sample, with 20 warmup calls before timing; its baseline-versus-baseline check showed 0.00% drift on the 5 KB case and −5.40% drift on the zero-binding case. In the shared-process Node run, the zero-binding median rose 15.72% (32.082 → 37.125 ms/100 calls), with wide overlapping ranges; that drift does not appear in the fresh-process run. Results do not establish the causes of these controls, or their frequency in real workloads.

Reproduce from the repository root with a clean baseline checkout and this branch checkout. Paths below are placeholders; the generated output files contain only synthetic measurements and metadata.

```sh
node tools/verify_wgsl_compact.cjs BASELINE/web/workers/wgpu_worker.js web/workers/wgpu_worker.js
node tools/bench_wgsl_compact.cjs BASELINE/web/workers/wgpu_worker.js web/workers/wgpu_worker.js 11 100 > node-results.json
node tools/bench_wgsl_compact_process.cjs BASELINE/web/workers/wgpu_worker.js web/workers/wgpu_worker.js 11 100 > node-process-results.json
node tools/bench_wgsl_compact_process.cjs BASELINE/web/workers/wgpu_worker.js BASELINE/web/workers/wgpu_worker.js 11 100 > node-process-aa.json
node tools/browser_perf/run_wgsl_compact.cjs BASELINE/web/workers/wgpu_worker.js web/workers/wgpu_worker.js CHROME_EXECUTABLE chrome-results.json 11 100
node tools/browser_perf/run_wgsl_compact.cjs BASELINE/web/workers/wgpu_worker.js BASELINE/web/workers/wgpu_worker.js CHROME_EXECUTABLE chrome-aa.json 11 100
```

Raw data: [Chrome](wgsl-compact-chrome-2026-11-27.json), [Chrome repeat](wgsl-compact-chrome-repeat-2026-11-27.json), [Chrome A/A](wgsl-compact-chrome-aa-2026-11-27.json), [Node](wgsl-compact-node-2026-11-27.json), [Node repeat](wgsl-compact-node-repeat-2026-11-27.json), [Node fresh processes](wgsl-compact-node-process-2026-11-27.json), and [Node fresh processes A/A](wgsl-compact-node-process-aa-2026-11-27.json). The repeat and A/A files predate addition of the prefix and unused cases; their common cases use the same workloads.
