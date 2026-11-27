# Sparse staging of fragmented CPU-visible buffers

This changes CPU command production for buffers whose translated-window slots do not have a contiguous host view. A primed submission with at most half its bytes in marked slots stages only selected 4 KiB pages. First, verified, direct/unscanned and denser flushes retain the original bulk allocation/read/mark order. Unreadable pages retain their prior shadow contents.

The [complete native samples](fragmented-buffer-flush-2026-11-27.json) measure `Device::begin_submission`, `flush_cpu_buffers` and `end_submission`, including temporary staging allocation/deallocation and command encoding. GPU execution is absent. On the isolated four-vCPU AMD EPYC 7763 runner, a 64 MiB fragmented buffer with one changed 4 KiB page per flush takes **38.028603 → 0.115659 ms/flush** (37.912945 ms saved, 99.696% lower). Quarter/half-dirty resident controls improve 35–68%; dense/near-dense controls have overlapping ranges. The worst contiguous increase is 4.842 microseconds at 8 MiB (+11.66%, overlapping ranges); these controls are not an intended performance claim.

Baseline: `52918dfa4d41e9f316990b06b713219dd9135aac`. Measured patch: `a49b7c3d489a5423dc3a896e471cd3d1410504c6`. Later test-only changes add verification, direct and unscanned regression assertions without changing the timed loop or production code. Environment: Linux 7.0.0-1020-azure x86-64/glibc 2.43, four logical CPUs, 16,759,422,976 bytes RAM, GCC 15.2, CMake 4.5.1, Release, Opus OFF. Full validation separately uses default Opus ON.

Sampling: seven alternating pairs of separately built binaries, with one separate untimed two-flush invocation per binary/case. Every invocation primes the buffer before timing; resident controls first upload every page. Sparse/contiguous cases time 64 flushes, resident-density cases four, dense-first one, and the unaligned case four. Writes, complete packet verification and command-stream hashing are outside timing. All 308 measured invocations match the baseline's payload bytes, packet IDs/offsets/lengths/order, hashes and compared-byte counts. Raw samples include every value and spread for all 22 workloads.

Full Linux validation on final test head `568eb51fd574185575706121978a7cf210ae8e23` passes: default Opus ON CTest 102/102 (including all three new fixtures and getter tests), Rust 1211 passed/0 failed/1 ignored, Python 1326 tests OK/4 skipped, Node 2512 passed/0 failed/4 skipped plus syntax/hygiene checks. [Validation run](https://github.com/GDemay/playGreg6web/actions/runs/46070929240).

```sh
cmake -S runtime -B build -DCMAKE_BUILD_TYPE=Release \
  -DPLAYGTA6_OPUS=OFF -DPLAYGTA6_PERFORMANCE_BENCHMARKS=ON
cmake --build build -j4 --target bench_fragmented_buffer_flush test_fragmented_buffer_flush test_guest_window
ctest --test-dir build -R '(^fragmented_buffer|playgta6_guest_window_tests$)' --output-on-failure
build/bench_fragmented_buffer_flush 64 64 fragmented
build/bench_fragmented_buffer_flush 8 4 fragmented-dense-resident
build/bench_fragmented_buffer_flush 8 4 fragmented-half
build/bench_fragmented_buffer_flush 8 4 fragmented-nearly-dense
build/bench_fragmented_buffer_flush 2 4 fragmented-unaligned
```

The [immutable matched workflow](https://github.com/GDemay/playGreg6web/blob/e7ea534922694d53d28b969cba9ccffb7d0351a7/.github/workflows/translated-flush-perf.yml) copies the same benchmark source/build registration into baseline, builds both trees sequentially, runs the original XS suite on both plus the patch's focused tests, and then samples without builds. [Primary run](https://github.com/GDemay/playGreg6web/actions/runs/46069917130).

[Rejected controls and phase diagnostics](fragmented-buffer-flush-diagnostics-2026-11-27.json) are retained separately. Earlier eager candidate allocation changed heap/page-fault behavior and regressed an 8 MiB dense control; restoring destruction order alone did not fix it. A near-dense control also regressed before the half-marked limit. The final allocation-free snapshot query preserves bulk ordering for these workloads.

The actual game's affected-buffer sizes, mapping density and frequency are unmeasured. Native XMem shared views can remain contiguous; this fixture deliberately models fragmented translated backing. Browser/Wasm timing, GPU work, game FPS, RSS, allocator internals and persistent resource destruction are unmeasured. The sparse scratch still requests the full logical size; the measured saving comes from avoiding initialization/reads of clean pages, not a measured reduction in allocator-requested bytes.
