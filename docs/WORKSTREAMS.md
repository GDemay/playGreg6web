# Workstreams & Parallel Tasks

Work is split into **5 independent tracks** so many contributors (and AI agents) can work in parallel without merge conflicts. Tracks only communicate through the frozen contracts in `docs/io_manifest.md`, `docs/wasm_abi.md`, and `docs/gpu_commands.md`.

> **Rule for all contributors & AI agents:**
> - If a task needs an ABI change, open a small PR touching `docs/` + `runtime/abi/playgta6_abi.h` first and merge it before dependent feature work.
> - **Never stop just because listed items are claimed:** if every task below is `DONE` or `IN PROGRESS`, inspect `docs/END_GOAL.md` (M1 → M4) + `gh pr list --state open`, define the next non-overlapping numbered task (disjoint files from open PRs), claim it with a GitHub Issue (`gh issue create --assignee "@me"`), add it here, and start immediately!

---

## Track 1: Recompiler (`recomp/`, Rust)
**Goal:** Lift the Xbox Series x86-64 PE executable into Wasm64 (`memory64`).

- [x] **R1 — PE Parser & Import Dumper** — `DONE` (PR #21, closes #1) *(needs `GTA6_GAME_DIR` for real test; unit-testable with synthetic PE)*
  - Parse PE64 headers, sections, exports, TLS callbacks, and import tables (`object` / `goblin` crate).
  - Emit a sanitized symbol-only report (`docs/imports_required.md` — DLL + function names only, zero game bytes).
- [x] **R2 — x86-64 Decoder & Function Discovery** — `DONE` (PR #37, closes #13) *(no game files required for unit tests)*
  - Integrate `iced-x86` to decode `.text` blocks, follow calls/jumps, and build a CFG + relocation map.
- [ ] **R3 — LLVM IR → Wasm64 Lifter Prototype** — `IN PROGRESS` (#14, @Timon-ok) *(no game files required)*
  - Lift small synthetic x86-64 test functions (arithmetic, branches, memory loads/stores, Win64 ABI calls) to LLVM IR and compile to `wasm64`.
- [ ] **R4 — PE64 `.pdata` Exception Directory & Jump-Table Recovery** — `IN PROGRESS` (#38, @Timon-ok) *(no game files required — disjoint file `recomp/src/pdata.rs`)*
  - Parse PE64 `IMAGE_DIRECTORY_ENTRY_EXCEPTION` (`RUNTIME_FUNCTION` begin/end/unwind RVA) and `UNWIND_INFO` to recover exact function boundaries and switch/jump tables without heuristics.
- [ ] **R5 — Import Thunk & Runtime Call-Bridge Generator** — `IN PROGRESS` (#39, @Timon-ok) *(no game files required — disjoint file `recomp/src/stubs.rs`)*
  - Given parsed PE64 imports (`R1`), generate C++/LLVM thunk glue tables binding imported Xbox/Win32 DLL symbols to `pg6_win32_*` / `pg6_gdk_*` / `d3d12_xs` runtime shims (and safe panic/log stubs for unresolved symbols).
- [ ] **R6 — PE Section Image & Relocation Packer for Wasm64 Memory** — `IN PROGRESS` (#40, @Timon-ok) *(no game files required — disjoint file `recomp/src/image.rs`)*
  - Emit a deterministic Wasm64 static data image + base-relocation table (`.rdata`, `.data`, `.bss`, `.reloc`) so recompiled code sees the exact guest virtual address layout inside `VirtualMemoryArena`.

---

## Track 2: Xbox / Win32 OS Shims (`runtime/os/`, C++20 + Emscripten)
**Goal:** Reimplement console OS APIs backed by browser workers, pthreads, and HTTPFS.

- [x] **O1 — Emscripten Wasm64 + Pthreads Smoke Build** — `DONE` (PR #35, closes #23) *(no game files required)*
  - Wire `runtime/main.cpp` JS glue (`web/loader.js` imports for `host_log`, `host_wgpu_submit`, `host_httpfs_*`) and SAB HTTPFS mailbox consumer in `io_worker.js`.
- [x] **O2 — Blocking HTTPFS Bridge (`io_worker.js` ↔ `runtime/os/`)** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Implement synchronous worker-to-worker `Atomics.wait` range reads over a `SharedArrayBuffer` so `host_httpfs_open`/`read`/`close` work from pthreads.
- [x] **O3 — Virtual Memory & Threading/TLS Shims** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Implement Win32/Xbox `VirtualAlloc`/`VirtualProtect` arena mapping, `TlsAlloc`, SRW locks, and `XTaskQueue` on top of Emscripten pthreads.
- [x] **O4 — `GameInput` & `/userdata` Shims** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Read `playgta6::InputState` from the input `SharedArrayBuffer` and map save paths to IndexedDB `/userdata`.
- [x] **O5 — Xbox GDK `xgameruntime` & Win32 Kernel/File/Sync C-ABI Shims** — `DONE` (PR #36, closes #16) *(no game files required)*
  - Expose C-ABI Win32 & Xbox GDK imports (`XGameRuntimeInitialize`, `XTaskQueue*`, `XUser*` offline stub, `VirtualAlloc`/`Protect`/`Free`, `Tls*`, `SRWLock*`, `CriticalSection`/`Event`, `CreateFileW`/`ReadFile`/`SetFilePointerEx`/`GetFileSizeEx`/`CloseHandle`, `QueryPerformanceCounter`/`Frequency`) backed by our OS shims.
- [ ] **O6 — Win32 Synchronization & Thread Pool Shims** — `IN PROGRESS` (#41, @Timon-ok) *(no game files required — disjoint files `runtime/os/win32_sync.{h,cpp}`, `tests/test_win32_sync.cpp`)*
  - Implement `CreateSemaphoreW`/`ReleaseSemaphore`, `CreateMutexW`/`ReleaseMutex`, `WaitForMultipleObjects` (`bWaitAll` true/false), `SleepConditionVariableSRW`/`WakeConditionVariable`/`WakeAllConditionVariable`, and `CreateThread`/`SetThreadAffinityMask`/`GetCurrentThreadId`.
- [ ] **O7 — Win32 Heap, Module Registry & `GetProcAddress` Resolver** — `IN PROGRESS` (#42, @Timon-ok) *(no game files required — disjoint files `runtime/os/win32_loader.{h,cpp}`, `tests/test_win32_loader.cpp`)*
  - Implement `GetProcessHeap`/`HeapCreate`/`HeapAlloc`/`HeapReAlloc`/`HeapFree`/`HeapSize`, `GetModuleHandleW`/`LoadLibraryW`, and a dynamic `GetProcAddress` symbol registry mapping DLL + export names to registered runtime shims.
- [ ] **O8 — Offline Networking, Telemetry & Xbox Live Safe Stubs** — `IN PROGRESS` (#43, @Timon-ok) *(no game files required — disjoint files `runtime/os/xbox_offline.{h,cpp}`, `tests/test_xbox_offline.cpp`)*
  - Implement clean offline-mode stubs for `ws2_32` (`WSAStartup`, `socket`, `getaddrinfo` → offline), `winhttp`, and Xbox Live (`XblInitialize`, `XblContextCreateHandle`, `XStore*`) so single-player boot never blocks on online services.

---

## Track 3: GPU Command Bridge (`runtime/gpu/` + `web/workers/wgpu_worker.js`, C++20 + JS)
**Goal:** Translate `d3d12_xs` command lists to binary packets executed on WebGPU.

- [x] **G1 — Expand Command Ring & Resource Tables** — `DONE` (PR #7, closes #4) *(no game files required)*
  - Implement `CreateBuffer`, `WriteBuffer`, `CreateShaderModule`, `CreatePipeline`, and `Draw` packets across `runtime/abi/playgta6_abi.h`, `runtime/gpu/cmd_encoder.cpp`, and `web/workers/wgpu_worker.js`.
- [x] **G2 — SharedArrayBuffer Zero-Copy GPU Ring** — `DONE` (PR #22, closes #10) *(no game files required)*
  - Replace `postMessage` buffer copies with a lock-free SPSC `SharedArrayBuffer` ring buffer between Wasm64 and `wgpu_worker.js`.
- [x] **G3 — `d3d12_xs` Interface Stubs & Trace Harness** — `DONE` (PR #7, closes #4) *(no game files required)*
  - Implement minimal COM-like `ID3D12Device` / `ID3D12GraphicsCommandList` / `PresentX` shim layer that emits packets matching `docs/gpu_commands.md`.
- [x] **G4 — `WGPK` `.wgsl.pack` Binary Loader & Shader Module Cache** — `DONE` (PR #26, closes #18) *(no game files required)*
  - Parse `.wgsl.pack` binary bundles (`shaders/src/pack.rs`) in JS (`web/workers/wgsl_pack_loader.js`), resolve WGSL by 64-bit FNV-1a hash, and cache compiled `GPUShaderModule`s.
- [x] **G5 — D3D12-XS Textures, Samplers, BindGroups, Indexed Draw & Compute Dispatch** — `DONE` (PR #55, closes #24) *(no game files required)*
  - Extend GPU command ABI, `CommandEncoder`, `d3d12_xs.h`, and `wgpu_worker.js` with textures, samplers, bind groups, indexed drawing, and compute dispatch (for RDNA2 mesh-compute lowering).
- [ ] **G6 — D3D12-XS Descriptor Heaps, Root Signatures & Resource State Tracker** — `IN PROGRESS` (#44, @Timon-ok) *(no game files required — disjoint files `runtime/gpu/descriptor_heap.{h,cpp}`, `tests/test_descriptor_heap.cpp`)*
  - Implement CPU/GPU descriptor heap tables (`CBV_SRV_UAV`, `SAMPLER`, `RTV`, `DSV`), root signature parameter mapping, and dirty-bind-group caching that compiles D3D12 descriptor tables into WebGPU `CreateBindGroup`/`SetBindGroup` calls.
- [ ] **G7 — D3D12 PSO Cache, Render Targets & Viewport/Scissor State** — `IN PROGRESS` (#45, @Timon-ok) *(no game files required — disjoint files `runtime/gpu/pso_cache.{h,cpp}`, `tests/test_pso_cache.cpp`)*
  - Hash D3D12 graphics & compute PSO descriptors (blend/depth/rasterizer/vertex-layout/shader hashes) to deduplicate WebGPU pipeline creation and track active RTV/DSV/viewport/scissor state per command list.
- [x] **G8 — Binary GPU Command Stream Disassembler & Trace Inspector** — `DONE` (PR #49, closes #32) *(no game files required — disjoint files `tools/gpu_trace.py`, `tests/test_gpu_trace.py`)*
  - Parse recorded `playgta6` GPU command ring byte streams into human-readable/JSON frame traces with opcode histograms, byte counts, and malformed-packet diagnostics for debugging M3 frames.
- [x] **G9 — WebGPU `OffscreenCanvas` Swapchain, Depth-Stencil & Resize Manager** — `DONE` (PR #59, closes #54) *(no game files required — disjoint files `web/workers/wgpu_swapchain.js`, `tests/test_wgpu_swapchain.mjs`)*
  - Manage `OffscreenCanvas` WebGPU context config, DPR-aware dynamic resize, `depth24plus`/`depth32float` depth-stencil attachment lifecycle, `device.lost` recovery, and frame-pacing telemetry.
- [x] **G10 — BCn / Texture Format & Mip-Chain Upload Helpers** — `DONE` (PR #66, closes #57) *(no game files required — disjoint files `web/workers/texture_formats.js`, `tests/test_texture_formats.mjs`)*
  - Map DXGI/Xbox texture formats (`BC1`–`BC7`, `RGBA8`, `RGBA16F`, depth/stencil) to WebGPU `GPUTextureFormat`, compute 256-byte-aligned `bytesPerRow` + mip dimensions, and provide CPU fallback BC1/BC3 decompression when WebGPU `texture-compression-bc` is absent.
- [ ] **G11 — WebGPU Pipeline & BindGroupLayout Cache Manager (`web/workers/pipeline_cache.js`)** — `IN PROGRESS` (#62, @MarcBlattmann) *(no game files required — disjoint files `web/workers/pipeline_cache.js`, `tests/test_pipeline_cache.mjs`)*
  - Deduplicate `GPURenderPipeline` and `GPUComputePipeline` creation in JS by descriptor hash, support async `createRenderPipelineAsync` warm-up, and consume shader reflection metadata (`S6`).
- [ ] **G12 — Update `tools/gpu_trace.py` for G5 Opcodes `12..21`** — `IN PROGRESS` (#67, @MarcBlattmann) *(no game files required — disjoint files `tools/gpu_trace.py`, `tests/test_gpu_trace.py`)*
  - Add full typed payload decoding for `CreateTexture`..`DispatchCompute` (`12..21`), compute pass summaries, and texture/bind-group histograms matching the merged `G5` ABI.
- [ ] **G13 — WebGPU Shader/Pipeline Diagnostics & Capture Replay (`web/workers/gpu_recorder.js`)** — `IN PROGRESS` (#72, @MarcBlattmann) *(no game files required — disjoint files `web/workers/gpu_recorder.js`, `tests/test_gpu_recorder.mjs`)*
  - Ring-buffered in-browser frame recorder that captures raw binary GPU command streams (`BeginFrame`..`Present`) and exports downloadable `.pg6trace` blobs compatible with `tools/gpu_trace.py`.
- [ ] **G14 — WebGPU Staging Buffer Pool & Upload Ring (`web/workers/staging_pool.js`)** *(unclaimed; no game files required — disjoint files `web/workers/staging_pool.js`, `tests/test_staging_pool.mjs`)*
  - Sub-allocating ring of `MAP_WRITE | COPY_SRC` / `writeBuffer` staging chunks with fence-driven recycling to eliminate per-frame buffer allocation churn in `wgpu_worker.js`.

---

## Track 4: Shader Pipeline (`shaders/`, Rust)
**Goal:** Convert Xbox RDNA2 shaders (`--mesh-compute`, `--wave64-emulate`) → SPIR-V → WGSL packs.

- [x] **S1 — `naga` SPIR-V → WGSL Batch Converter** — `DONE` (PR #9, closes #5) *(testable with synthetic SPIR-V; real corpus needs local SPIR-V)*
  - Add `naga` to `shaders/Cargo.toml`, convert a directory of `.spv` files to `.wgsl`, emit `shader-index.json`, and output a structured failure-reason report (unsupported capabilities/extensions).
- [x] **S2 — Shader Pack Bundler & WebGPU Pre-Validator** — `DONE` (PR #9, closes #5) *(no game files required)*
  - Pack lowered `.wgsl` files into compact `.wgsl.pack` bundles (like `playgta5`) for fast loading in `wgpu_worker.js`.
- [x] **S3 — WebGPU Compatibility Passes** — `DONE` (PR #25, closes #11) *(needs SPIR-V failure report from S1)*
  - Implement SPIR-V/WGSL rewrite passes for unsupported WebGPU features (bindless indexing limits, subgroup/wave64 fallbacks, mesh-compute lowering).
- [x] **S4 — DXBC/DXIL Container Inspector & Naga IR Push-Constant/BindGroup Lowering** — `DONE` (PR #56, closes #27) *(no game files required)*
  - Add pure-Rust DXBC/DXIL container parser (`shaders/src/dxbc.rs`) and Naga IR lowering passes (`shaders/src/lower.rs`: push constants → uniform buffer, >4 bind-group remapping) + `inspect-dxbc` CLI.
- [ ] **S5 — Wave64/Subgroup Emulation & Mesh-Compute WGSL Lowering Library** — `IN PROGRESS` (#46, @Timon-ok) *(no game files required — wait for S4 or keep disjoint in `shaders/src/wave_emul.rs`)*
  - Provide reusable WGSL workgroup-shared-memory emulation helpers for RDNA2 Wave64 ballot/shuffle/prefix-sum ops and indirect compute-driven meshlet culling/draw emission when WebGPU `subgroups` is unavailable.
- [ ] **S6 — Root-Signature-to-WebGPU `BindGroupLayout` Reflection & Cache Manifest** — `IN PROGRESS` (#47, @Timon-ok) *(no game files required — disjoint file `shaders/src/reflect.rs`)*
  - Reflect parsed `naga::Module` resource bindings, vertex inputs, and workgroup sizes into compact JSON metadata alongside `shader-index.json` so `wgpu_worker.js` can build exact `GPUBindGroupLayout`s without runtime string parsing.

---

## Track 5: Browser Shell, Streaming I/O & Audio (`web/`, `server/`, `tools/`, JS + Python + Rust)
**Goal:** Fast, resilient browser loading, streaming, caching, and audio playback.

- [x] **W1 — OPFS + LRU Range Cache in `io_worker.js`** — `DONE` (PR #15, closes #2) *(no game files required)*
  - Cache fetched HTTP `206` byte chunks in Origin Private File System (OPFS) with a configurable budget + `bootset.json` prefetch on startup.
- [x] **W2 — IndexedDB `/userdata` Save Persistence** — `DONE` (PR #15, closes #2) *(no game files required)*
  - Restore `/userdata/` files into Wasm FS before `main()` and flush writes asynchronously (mirroring `playgta5`'s `loader.js` pattern).
- [x] **W3 — `audio_worker.js` Opus / WebAudio Pipeline** — `DONE` (PR #8, closes #6) *(no game files required)*
  - Implement `AudioWorklet` ring buffer + Ogg Opus stream decoding.
- [x] **W4 — Local Environment & Diagnostics (`tools/`)** — `DONE` (PR #8, closes #6) *(uses `GTA6_GAME_DIR` when available)*
  - Extend `tools/` to generate `bootset.json`, inspect RPF9 archive headers without extracting 100+ GB, and verify local file integrity.
- [x] **W5 — Full `InputBridge` (Gamepad API + Xbox Buttons/Sticks/Triggers + Pointer Lock)** — `DONE` (PR #28, closes #17) *(no game files required)*
  - Populate the full 40-byte `playgta6::InputState` SAB from Pointer Lock mouse, keyboard, and `navigator.getGamepads()` matching `runtime/os/game_input.h`.
- [x] **W6 — Partial-Download Game Folder Scanner & Incremental Manifest CLI (`tools/`)** — `DONE` (PR #30, closes #19) *(works on partial or complete `GTA6_GAME_DIR`)*
  - Add read-only `scan` command detecting completed vs downloading files (`.!qB`/`.part`/sparse) and `--skip-incomplete` for `manifest`/`bootset`.
- [x] **W7 — Local Server Hardening (`/api/status`, ETag/If-Range) & Browser Diagnostics HUD (`server/`, `web/index.html`)** — `DONE` (PR #29, closes #20) *(no game files required)*
  - Add ETag/`If-Range`/`HEAD` + sanitized `/api/status` to `server/serve_local.py` and upgrade `web/index.html` with a live Wasm64/WebGPU/SAB HUD.
- [x] **W8 — Multi-Voice Audio Mixer & PCM/ADPCM Voice Streamer** — `DONE` (PR #51, closes #31) *(no game files required — disjoint files `web/workers/audio_mixer.js`, `tests/test_audio_mixer.mjs`)*
  - Implement a lock-free multi-voice mixer (3D pan/gain, sample-rate resampling, ADPCM/PCM decoding, soft-clip limiter) that feeds `audio_worker.js`'s SAB ring buffer for XAudio2/XMA2 voice emulation.
- [x] **W9 — Headless End-to-End Integration Harness (Local Server ↔ Workers ↔ SAB ↔ WebGPU Mock)** — `DONE` (PR #48, closes #33) *(no game files required — disjoint file `tests/test_e2e_pipeline.mjs`)*
  - Spin up `server/serve_local.py` on an ephemeral port in a Node test, exercise real HTTP `206` Range + `ETag`/`If-Range` reads through `RangeCache` → `WgslPackCache` → `InputBridge` → `GpuRingBuffer` → `WgpuCommandProcessor` end-to-end.
- [x] **W10 — Pure-Rust Xbox XVD/XVC Container Header & Layout Inspector** — `DONE` (PR #50, closes #34) *(works on local `.xvd`/`.xvc` or synthetic unit tests — disjoint files `tools/src/xvd.rs`, `tools/src/main.rs`)*
  - Read-only parser for `msft-xvd` container headers (version, content type, drive/embedded XVD sizes, XVC region presence, encryption flags) wired as `playgta6-manifest xvd <path> [--json]`.
- [x] **W11 — Streaming Browser RPF9 TOC & Virtual File System Reader** — `DONE` (PR #61, closes #52) *(no game files required — disjoint files `web/workers/rpf9_vfs.js`, `tests/test_rpf9_vfs.mjs`)*
  - Fetch only RPF header + TOC + requested sub-file byte ranges via `RangeCache`, parse directory/file entries & names table, resolve virtual paths, and stream sub-file chunks on demand.
- [x] **W12 — `AudioWorklet` SAB Ring Processor & Main-Thread WebAudio Engine** — `DONE` (PR #60, closes #53) *(no game files required — disjoint files `web/workers/audio_worklet_processor.js`, `web/audio/audio_engine.js`, `tests/test_audio_engine.mjs`)*
  - Zero-GC `AudioWorkletProcessor` pulling 128-frame stereo blocks from `AudioRingBuffer` SAB + main-thread `AudioEngine` (`AudioContext` bootstrap, user-gesture unlock, volume/mute, underrun telemetry).
- [x] **W13 — Local Preflight & Milestone Readiness Checker CLI** — `DONE` (PR #65, closes #58) *(no game files required — disjoint files `tools/preflight.py`, `tests/test_preflight.py`)*
  - Single-command `python tools/preflight.py [--json]` verifying toolchains, `.env` / `GTA6_GAME_DIR`, COOP/COEP server sanity, `web/dist/runtime.wasm`, `data-manifest.json`, `bootset.json`, and `.wgsl.pack` readiness.
- [x] **W14 — OPFS Cache Quota Inspector & Browser Storage Manager** — `DONE` (PR #70, closes #63) *(no game files required — disjoint files `web/storage/storage_manager.js`, `tests/test_storage_manager.mjs`)*
  - Query `navigator.storage.estimate()`, request persistent storage, inspect/clear OPFS range cache and IndexedDB `/userdata` slots cleanly from the browser HUD.
- [ ] **W15 — Frame Telemetry Ring & Performance Profiler (`web/telemetry/perf_monitor.js`)** — `IN PROGRESS` (#64, @MarcBlattmann) *(no game files required — disjoint files `web/telemetry/perf_monitor.js`, `tests/test_perf_monitor.mjs`)*
  - Lock-free rolling frame-time, p95/p99 latency, HTTP Range throughput, GPU packet rate, and audio underrun aggregator feeding `window.playgta6Hud.updateMetrics()`.
- [ ] **W16 — Browser WebGPU Smoke Demo & Built-in Intro Splash Test Scene (`web/demo/smoke_scene.js`)** — `IN PROGRESS` (#68, @MarcBlattmann) *(no game files required — disjoint files `web/demo/smoke_scene.js`, `tests/test_smoke_scene.mjs`)*
  - Self-contained binary command stream generator exercising `G5` (`CreateTexture`, `CreateSampler`, `CreateBindGroup`, `SetIndexBuffer`, `DrawIndexed`, `DispatchCompute`) + `WgpuSwapchain` so opening `http://127.0.0.1:8080` renders a live textured 3D/compute frame in Chrome/Edge even before `runtime.wasm` is compiled.
- [ ] **W17 — Emscripten Wasm64 Build Script for Linux/macOS (`runtime/build_wasm.sh`) & CI Artifact Check** — `IN PROGRESS` (#69, @MarcBlattmann) *(no game files required — disjoint files `runtime/build_wasm.sh`, `tests/test_build_scripts.py`)*
  - Cross-platform POSIX shell counterpart to `runtime/build_wasm.ps1` (`--check-only`, `-sMEMORY64=1 -pthread`, `web/dist/runtime.{js,wasm}` output).
- [ ] **W18 — Streaming HTTPFS Priority Scheduler & Prefetch Queue (`web/workers/io_scheduler.js`)** — `IN PROGRESS` (#73, @MarcBlattmann) *(no game files required — disjoint files `web/workers/io_scheduler.js`, `tests/test_io_scheduler.mjs`)*
  - Priority queue (critical boot/sync reads > streaming world cells > background prefetch) with cancellation, retry backoff, and concurrency throttling in front of `RangeCache`.
- [ ] **W19 — Binary Wasm64 Memory64/Threads & ABI Import Validator (`tools/check_wasm_exports.py`)** — `IN PROGRESS` (#74, @MarcBlattmann) *(no game files required — disjoint files `tools/check_wasm_exports.py`, `tests/test_wasm_exports.py`)*
  - Binary `.wasm` section/import/export validator verifying a compiled `runtime.wasm` declares `memory64` (`i64` memory), shared memory (`threads`), and exact `env.host_*` ABI imports from `docs/wasm_abi.md`.
- [ ] **W20 — Gamepad Remapper & Input Profile Persistence (`web/input/input_profiles.js`)** *(unclaimed; no game files required — disjoint files `web/input/input_profiles.js`, `tests/test_input_profiles.mjs`)*
  - Configurable keyboard/mouse/gamepad presets, sensitivity/deadzone curves, and localStorage/IndexedDB profile persistence feeding `InputBridge`.
- [ ] **W21 — Structured Crash Reporter & Runtime Diagnostic Bundle Exporter (`web/telemetry/crash_report.js`)** *(unclaimed; no game files required — disjoint files `web/telemetry/crash_report.js`, `tests/test_crash_report.mjs`)*
  - Captures Wasm traps, WebGPU validation/device-lost errors, recent runtime logs, and browser capability fingerprints into sanitized shareable JSON reports (never including game data or host paths).
