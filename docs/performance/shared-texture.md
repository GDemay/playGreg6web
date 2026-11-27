# Shared texture sources: measured application snapshot removal

Modern WebGPU accepts a shared source for `GPUQueue.writeTexture` and snapshots its contents before the call returns. The handler now uses this contract for nonchunked sources within the existing chunk cap. A synchronous `TypeError` selects the cached legacy snapshot path; other errors propagate. Chunked and oversized uploads retain the previous path.

Specification: [WebGPU §19.2, writeTexture](https://www.w3.org/TR/2026/CRD-webgpu-20261116/#dom-gpuqueue-writetexture), including `AllowSharedBufferSource` and the content-timeline copy.

## Performance

These are **software-GPU browser results**, using real Chrome WebGPU APIs and production packet handling. They do not predict hardware-GPU or gameplay performance. Each measured upload is a synthetic 3840×2160 `r8unorm` Y plane, 8,294,400 bytes, in a `SharedArrayBuffer` packet.

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| Synchronous upload encoding wall time, median per Y upload | 1.101875 ms | 0.644375 ms | −0.457500 ms (−41.52%) |
| Nonshared Y control, median encoding wall time per upload | 0.661250 ms | 0.644375 ms | −0.016875 ms (−2.55%) |
| Queue completion fence wait wall time, median per Y upload | 5.836875 ms | 5.766875 ms | −0.070000 ms (−1.20%) |
| Nonshared control fence wait, median per upload | 5.688125 ms | 5.748125 ms | +0.060000 ms (+1.05%) |
| Application SAB snapshots per batch of 8 Y uploads | 8 | 0 | −8 (−100%) |
| `writeTexture` calls / source bytes per Y upload | 1 / 8,294,400 B | 1 / 8,294,400 B | unchanged |

Encoding sample ranges, normalized per upload: before 1.045000–1.237500 ms; after 0.611875–0.683125 ms. Nonshared control ranges: before 0.615625–0.716875 ms; after 0.620625–0.700000 ms. Fence wait is a wall-clock queue-completion wait, **not a GPU timestamp duration**. The eliminated application snapshot copies 8,294,400 bytes per Y upload; the browser still performs the content-timeline copy required by WebGPU.

Baseline: `14d1b86644dbc35fbf32d9522c505972545a7d8a`. Measured production patch: `4fa5b59ca5d49f806252093ea9ff480918cbbb0f`. Later documentation/runner commits do not change that production code. [Sanitized raw batches](shared-texture-2026-11-27.json) retain every sample and count.

Environment: dedicated Linux Codespace, 4 vCPUs, 16,770,469,888 visible RAM bytes, AMD EPYC 7763, Linux 6.8.0-1070-azure, Node v24.22.2, Chrome Headless Shell 158.0.8127.88, adapter vendor `google`, architecture `swiftshader`. No concurrent benchmark, build or test jobs ran in this lane. Host load averages are preserved in the raw evidence.

Sampling: 7 rounds, alternating baseline/patch page order; each invocation performs 3 warmup batches and 5 measured batches of 8 uploads, then the same procedure for a nonshared aligned Y control. This gives 35 measured batches / 280 measured uploads per variant and workload. The median uses batch wall time divided by 8. Both pages use identical harness code; background throttling is disabled and each page is brought to the foreground before invocation. Deterministic fixture generation and readbacks are outside timing; production upload encoding, source preparation and queue calls are inside.

## Reproduce

Use Node 24+ and Chrome/Chrome Headless Shell with SwiftShader available. The runner uses Node's built-in HTTP server and CDP WebSocket, with no npm dependencies. It serves both checkouts with COOP/COEP headers, using the patch checkout's identical harness for each variant.

```sh
git worktree add ../shared-base 14d1b86644dbc35fbf32d9522c505972545a7d8a
node tools/browser_perf/run_shared_texture.cjs \
  ../shared-base . /path/to/chrome-headless-shell shared-results.json 7 5 8
```

The exact software-adapter flags are in the runner. It rejects a non-SwiftShader adapter. A one-batch correctness/debug invocation uses trailing arguments `1 1 1`; do not mix such samples with the measured run.

For interactive inspection, serve this repository with COOP/COEP headers, open `tools/browser_perf/texture_shared.html`, and run `await runSharedTextureBenchmark()` in DevTools. The automated runner provides matched variants and explicit software-adapter selection.

## Correctness and limits

Every invocation validates an actual GPU readback after overwriting a 512-byte SAB source immediately after `writeTexture` returns: the original GPU pixels must remain. It also compares all 8,294,400 bytes of the final Y-plane texture. Array/partial upload ordering, explicitly staged then direct writes, and BC7 array block readbacks are checked outside timing. The measured lane reported zero GPU validation errors.

Unit coverage checks modern acceptance with no application snapshot, cached legacy fallback including cross-realm `TypeError`, non-`TypeError` propagation, queue replacement, correct snapshot counters even on failed fallback writes, and unchanged snapshot behavior above the chunk cap.

Unmeasured: hardware GPU timing, actual movie decoding, real game/runtime execution, drawing/pacing/FPS, GPU timestamp duration, browser-internal copies, and latency on older browsers. Legacy fallback behavior is unit-tested; the measured browser accepts shared sources.
