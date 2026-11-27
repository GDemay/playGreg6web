# Activate the existing browser texture-clear consumer

The browser worker already implements opcode 63. Wasm boots now enable its producer path while the guest runs, then restore the prior setting. This lets existing sampled-color texture pools reuse larger textures and clear reused contents without CPU zero-upload packets. There is no wire-ABI change.

The producer rejects depth/stencil address textures and creates single-sample `COPY_DST` textures. The worker's existing copy-based clear handles color formats, compressed blocks, arrays, mip tails and 3D depth reduction. This contribution activates that existing contract; it does not change the worker's clear algorithm.

## Performance

**Software-GPU browser consumer-contract replay**, not an actual Wasm producer or game run. The synthetic workload repeatedly resets a sampled-only RGBA8 texture of the same shape. The baseline contract uploads zeros for a reused 64 KiB texture and recreates a 4 MiB texture because its large pool is disabled. The enabled contract reuses the texture and sends `ClearTexture`.

Every operation also copies the entire texture into the same preallocated GPU sink, using identical consumer commands in both variants. Thus every fresh baseline allocation is observed and initialized, rather than remaining unused/lazy. The sink allocation, fixtures, initial reusable zero-buffer allocation and correctness readbacks are outside timing.

Primary metric: median of each batch's `(encoding wall + completion-fence wait wall) / operations`. This is a serialized consumer workload wall time, not GPU timestamp execution time. Component medians are reported separately and must not be added together to reconstruct the median total.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| 1024×1024 RGBA8 total per reset, bounded rerun | 3.3578125 ms | 0.3446875 ms | −3.0131250 ms (−89.735%) |
| Same workload, encoding component | 0.0593750 ms | 0.0437500 ms | −0.0156250 ms (−26.316%) |
| Same workload, fence-wait component | 3.2850000 ms | 0.2975000 ms | −2.9875000 ms (−90.944%) |
| Unchanged consume-only total, bounded rerun | 0.2000000 ms | 0.1975000 ms | −0.0025000 ms (−1.250%) |
| Unchanged consume-only fence, bounded rerun | 0.1940625 ms | 0.1878125 ms | −0.0062500 ms (−3.221%) |
| 1024×1024 RGBA8 total per reset, initial run | 3.3121875 ms | 0.3518750 ms | −2.9603125 ms (−89.376%) |
| Initial consume-only total | 0.2028125 ms | 0.1737500 ms | −0.0290625 ms (−14.330%) |
| 128×128 RGBA8 total per reset, initial run | 0.1043750 ms | 0.1050000 ms | +0.0006250 ms (+0.599%) |
| Same small workload, encoding component | 0.0862500 ms | 0.0340625 ms | −0.0521875 ms (−60.507%) |
| Same small workload, fence-wait component | 0.0150000 ms | 0.0815625 ms | +0.0665625 ms (+443.750%) |

No small-texture total-time improvement is demonstrated. Its encoding work shifts toward queued work; the fence component increases. Small total sample ranges overlap (before 0.094375–0.276875 ms, after 0.074688–0.255313 ms).

The first large-texture control's fence median moved 0.194375→0.165625 ms (−14.79%, 0.028750 ms), so one bounded matched rerun was performed with only the large workload and unchanged consume-only control. The rerun control total moved −1.25%, while the large reset improvement remained about 90%. Both complete runs are retained separately; no samples were trimmed. Rerun large-total ranges: before 3.153438–4.163125 ms, after 0.290625–0.411875 ms. Control-total ranges: before 0.150313–0.287188 ms, after 0.165625–0.302813 ms.

For each 16-operation large-texture batch, API instrumentation records 16→0 texture creations and 0→16 explicit clear copies. The consumer copies remain 16 in both variants, totaling 67,108,864 bytes per batch. These are application API counts, not GPU-bus traffic measurements. The small baseline sends 16 zero uploads totaling 1,048,576 bytes; the enabled contract sends no CPU zero-upload data. Actual C++ packet-generation cost is unmeasured.

[Initial raw results](texture-clear-initial-2026-11-27.json) and [bounded rerun raw results](texture-clear-rerun-2026-11-27.json) contain every sample and operation count.

Environment: dedicated Linux Codespace, 4 vCPUs, 16,770,469,888 visible RAM bytes, AMD EPYC 7763, Linux 6.8.0-1070-azure, Node v24.22.2, Chrome Headless Shell 158.0.8127.88, Google / SwiftShader **software** adapter. No concurrent benchmark, build or test jobs. Initial host loads: 2.01→2.24; rerun: 1.49→1.73 (one-minute averages). No macOS/Metal timings are used.

Baseline producer: `14d1b86644dbc35fbf32d9522c505972545a7d8a`. Producer activation is the `dda93e78` commit recorded fully in the evidence. The browser's clear consumer itself is unchanged. The replay selects the before/after producer contracts explicitly; it does not execute the Wasm producer. The C++ synthetic-boot test separately verifies the flag is enabled during guest entry and restored afterward.

## Reproduce

Node 24+ and Chrome/Chrome Headless Shell with bundled SwiftShader are required. The runner serves both checkouts on localhost with COOP/COEP headers, loads identical harness source, selects SwiftShader explicitly and rejects other adapters. No npm packages are needed.

```sh
git worktree add ../clear-base 14d1b86644dbc35fbf32d9522c505972545a7d8a
# Initial run: both texture sizes, each with its consume-only control.
node tools/browser_perf/run_texture_clear.cjs \
  ../clear-base . /path/to/chrome-headless-shell clear-initial.json 7 5 16 both
# One bounded rerun of the large workload and its unchanged control.
node tools/browser_perf/run_texture_clear.cjs \
  ../clear-base . /path/to/chrome-headless-shell clear-rerun.json 7 5 16 large
```

Each run uses 7 rounds, alternating baseline/patch order. Each invocation warms 3 batches, records 5 batches of 16 resets, then repeats that procedure for the unchanged consume-only control. This is 35 measured batches / 560 resets per variant and size. Both pages are brought to the foreground and background throttling is disabled. A correctness/runner smoke uses trailing arguments `1 1 1 large`; keep it separate from performance evidence.

## Correctness and limits

Every invocation validates real GPU readbacks for every mip/layer of RGBA8 arrays, 3D depth-shrinking mips, nonrenderable R8 signed textures and BC7 array/tail blocks. A write→clear→partial-write sequence checks ordering. Final benchmark textures are read back as zero. Both measured runs report zero GPU validation errors. Existing unit tests additionally cover unsupported depth/MSAA inputs, promoted/nonpromoted 1D handling, staged-upload ordering and bounded zero-buffer reuse.

Unmeasured: actual Wasm producer cost, real pool hit rates / size distributions, gameplay/FPS, hardware GPU timing, native-executor timing, movie decoding, GPU timestamps and cold zero-buffer allocation. The software-GPU clear result must not be presented as a hardware or game speedup.
