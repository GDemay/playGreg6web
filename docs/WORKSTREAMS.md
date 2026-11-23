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
- [x] **R3 — LLVM IR → Wasm64 Lifter Prototype** — `DONE` (PR #132, closes #14) *(no game files required)*
  - Lift small synthetic x86-64 test functions (arithmetic, branches, memory loads/stores, Win64 ABI calls) to LLVM IR and compile to `wasm64`.
- [x] **R4 — PE64 `.pdata` Exception Directory & Jump-Table Recovery** — `DONE` (PR #85, closes #38) *(no game files required — disjoint file `recomp/src/pdata.rs`)*
  - Parse PE64 `IMAGE_DIRECTORY_ENTRY_EXCEPTION` (`RUNTIME_FUNCTION` begin/end/unwind RVA) and `UNWIND_INFO` to recover exact function boundaries and switch/jump tables without heuristics.
- [x] **R5 — Import Thunk & Runtime Call-Bridge Generator** — `DONE` (PR #93, closes #39) *(no game files required — disjoint file `recomp/src/stubs.rs`)*
  - Given parsed PE64 imports (`R1`), generate C++/LLVM thunk glue tables binding imported Xbox/Win32 DLL symbols to `pg6_win32_*` / `pg6_gdk_*` / `d3d12_xs` runtime shims (and safe panic/log stubs for unresolved symbols).
- [x] **R6 — PE Section Image & Relocation Packer for Wasm64 Memory** — `DONE` (PR #78, closes #40) *(no game files required — disjoint file `recomp/src/image.rs`)*
  - Emit a deterministic Wasm64 static data image + base-relocation table (`.rdata`, `.data`, `.bss`, `.reloc`) so recompiled code sees the exact guest virtual address layout inside `VirtualMemoryArena`.
- [ ] **R7 — Lifter Coverage: String Ops, Bit Ops, Atomics & Remaining Integer Instructions (`recomp/src/lift/`)** — `IN PROGRESS` (#118, @Timon-ok) *(no game files required)*
  - Lift `rep movs/stos/cmps/scas/lods`, `bt*`, `bsf/bsr/tzcnt/lzcnt/popcnt`, `cmpxchg`/`xadd`/`bswap`, `rcl/rcr`, `shld/shrd`, `loop`/`jrcxz`, `leave`/`enter`, and `lock`-prefixed atomics, verified by execution and host-CPU differential tests.
- [ ] **R8 — Lifter Coverage: SSE/SSE2 Scalar & Packed Float and Integer SIMD (`recomp/src/lift/simd.rs`)** — `IN PROGRESS` (#119, @Timon-ok) *(no game files required — disjoint file `recomp/src/lift/simd.rs`)*
  - Lift the SSE/SSE2 (+ commonly emitted SSE3/SSSE3/SSE4.1) instructions MSVC x64 code uses onto the XMM slots of the guest state, verified by execution and host-CPU differential tests.
- [ ] **R9 — End-to-End Recompile Pipeline Driver (`recomp/src/pipeline.rs`)** — `IN PROGRESS` (#120, @Timon-ok) *(no game files required — disjoint file `recomp/src/pipeline.rs`)*
  - One `playgta6-recomp build <pe> --out <dir>` command chaining parse → `.pdata`-seeded discovery → lift → import thunks → packed guest image, with a deterministic build manifest and coverage report.
- [x] **R10 — MSVC x64 Jump-Table Pattern Recovery (`recomp/src/jump_table.rs`)** — `DONE` (PR #178, closes #149) *(driven by M2 discovery: 9,969 unresolved switch tables in `game_scarlett_master.exe`)*
  - Expand `recomp/src/jump_table.rs` to recover MSVC image-base-relative (`lea rBase, [rip+...]; movsxd/mov reg, [rBase + idx*4]; add reg, rBase; jmp reg`) switch tables and bounds checks across the real binary.
- [ ] **R11 — Win64 Exception Handling for Recompiled Code (`recomp/src/eh.rs` + `runtime/guest/`)** — `IN PROGRESS` (#160, @Timon-ok) *(imports `__CxxFrameHandler3/4`, `_CxxThrowException`, `__C_specific_handler`, `RtlVirtualUnwind`, `longjmp`)*
  - Parse C++ EH `FuncInfo` (v3/v4) and `__C_specific_handler` scope tables into the CFG (funclets, catch/continuation targets); implement `RaiseException`/`RtlCaptureContext`/`RtlLookupFunctionEntry`/`RtlVirtualUnwind`/`RtlUnwindEx`/`RtlRestoreContext`/`setjmp`/`longjmp` over guest state and `.pdata`; define the lifter/runtime resume protocol.
- [ ] **R12 — Multi-Module Recompile: Bundled DLLs & Guest-to-Guest Import Binding (`recomp/`)** — `IN PROGRESS` (#161, @Timon-ok) *(starts once R9 is merged)*
  - Build the executable plus the redistributable DLLs shipped next to it (`MSVCP140`, `VCRUNTIME140`, `concrt140`, …) into one guest program: per-module image bases, exports, forwarders, imports bound to another module's exports, shared function registration tables.

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
- [x] **O6 — Win32 Synchronization & Thread Pool Shims** — `DONE` (PR #80, closes #41) *(no game files required — disjoint files `runtime/os/win32_sync.{h,cpp}`, `tests/test_win32_sync.cpp`)*
  - Implement `CreateSemaphoreW`/`ReleaseSemaphore`, `CreateMutexW`/`ReleaseMutex`, `WaitForMultipleObjects` (`bWaitAll` true/false), `SleepConditionVariableSRW`/`WakeConditionVariable`/`WakeAllConditionVariable`, and `CreateThread`/`SetThreadAffinityMask`/`GetCurrentThreadId`.
- [x] **O7 — Win32 Heap, Module Registry & `GetProcAddress` Resolver** — `DONE` (PR #91, closes #42) *(no game files required — disjoint files `runtime/os/win32_loader.{h,cpp}`, `tests/test_win32_loader.cpp`)*
  - Implement `GetProcessHeap`/`HeapCreate`/`HeapAlloc`/`HeapReAlloc`/`HeapFree`/`HeapSize`, `GetModuleHandleW`/`LoadLibraryW`, and a dynamic `GetProcAddress` symbol registry mapping DLL + export names to registered runtime shims.
- [x] **O8 — Offline Networking, Telemetry & Xbox Live Safe Stubs** — `DONE` (PR #77, closes #43) *(no game files required — disjoint files `runtime/os/xbox_offline.{h,cpp}`, `tests/test_xbox_offline.cpp`)*
  - Implement clean offline-mode stubs for `ws2_32` (`WSAStartup`, `socket`, `getaddrinfo` → offline), `winhttp`, and Xbox Live (`XblInitialize`, `XblContextCreateHandle`, `XStore*`) so single-player boot never blocks on online services.
- [x] **O9 — Guest CPU State ABI & Recompiled-Code Dispatcher (`runtime/guest/`)** — `DONE` (PRs #138 + #158, closes #115) *(no game files required — small ABI PR first, then disjoint files `runtime/guest/guest_runtime.{h,cpp}`, `tests/test_guest_runtime.cpp`)*
  - Document the lifter's guest CPU state layout + runtime helper symbols in the shared ABI, then implement `__playgta6_indirect_call`/`__playgta6_trap`/`__playgta6_unimplemented`/`__playgta6_divide_error`, the guest-address → lifted-function table, IAT-slot → import-thunk dispatch, guest stack setup, and entry-point invocation.
- [x] **O10 — `PGIM` Guest Image Loader for `VirtualMemoryArena`** — `DONE` (PR #141, closes #116) *(no game files required — disjoint files `runtime/os/image_loader.{h,cpp}`, `tests/test_image_loader.cpp`)*
  - Load the `PGIM` v1 container from `playgta6-recomp pack-image` (R6): validate, reserve at the preferred base, commit sections, copy payloads, apply DIR64/HIGHLOW relocations when rebased, and set page protections.
- [x] **O11 — Runtime Shim Export Registration Table** — `DONE` (PR #133, closes #117) *(no game files required — disjoint files `runtime/os/shim_exports.{h,cpp}`, `tests/test_shim_exports.cpp`)*
  - Register every `pg6_win32_*` / `pg6_gdk_*` C-ABI shim with the O7 module registry under its Win32/GDK export name so guest `GetProcAddress` and the R5 import bridge can resolve them, with a header-sync test.
- [x] **O12 — Xbox Scarlett `xmem.dll` / `XCRDAPI.dll` / `xg_xs.dll` / `GameInput.dll` Shims (`runtime/os/xbox_scarlett.{h,cpp}`)** — `DONE` (PR #157, closes #150) *(driven by M2 `docs/imports_required.md`)*
  - Implement C-ABI shims for the 12 `xmem.dll` physical/virtual memory APIs, `XCRDAPI.dll!DStorageGetFactory` (DirectStorage backed by HTTPFS), `xg_xs.dll` (`XGCreateTextureComputer`, `XGSuggestSwizzleMode`), and `GameInput.dll` ordinal `#1`.
- [x] **O13 — UCRT (`api-ms-win-crt-*`) & `VCRUNTIME140` C-ABI Bridge (`runtime/os/ucrt_bridge.{h,cpp}`)** — `DONE` (PR #174, closes #151) *(driven by M2 `docs/imports_required.md`: 186 C-runtime imports)*
  - Expose thin Win64/C-ABI wrappers forwarding `api-ms-win-crt-{convert,filesystem,heap,locale,math,runtime,stdio,string,time,utility}` and `VCRUNTIME140` memory/string/exception exports onto Emscripten libc so `playgta6-recomp stubs` binds ~300 imports cleanly.
- [x] **O14 — Xbox Middleware Shims: `libHttpClient.GDK`, `Bink2Scarlett`, `AcpHal`/`xapu`/`chad`, FSR & `MFPlat` (`runtime/os/xbox_middleware.{h,cpp}`)** — `DONE` (PR #165, closes #152) *(driven by M2 `docs/imports_required.md`: 104 middleware imports)*
  - Implement clean offline/non-blocking C-ABI shims for `libHttpClient.GDK.dll` (38), `Bink2Scarlett.dll` (23), `AcpHal`/`xapu`/`chad` (19), `XboxBvhBuilder` (7), `amd_fidelityfx_loader_xbox` (4), and `MFPlat`/`MFReadWrite` (7).
- [ ] **O15 — Remaining Win32 Core (`api-ms-win-core-*`, `ADVAPI32`, `bcrypt`, `ntuser`) & `MSVCP140` C++ Runtime Shims (`runtime/os/win32_compat.{h,cpp}`)** — `IN PROGRESS` (#163, @Timon-ok) *(driven by M2 `docs/imports_required.md`)*
  - Implement the remaining ~55 Win32 core file/process/string/time/registry/bcrypt/window shims + `MSVCP140.dll` mutex/xtime/locale/iostream helpers to hit 100% import resolution for `game_scarlett_master.exe`.
- [ ] **O16 — Guest COM Object Bridge & `xgameruntime.dll` `QueryApiImpl` Provider (`runtime/guest/com_bridge.{h,cpp}`)** — `IN PROGRESS` (#167, @Timon-ok) *(the executable has no static `xgameruntime.dll` imports: the GDK client code loads it and calls `QueryApiImpl`/`InitializeApiImpl`)*
  - Generic guest-memory COM objects (vtables of dispatchable addresses, `IUnknown` lifetime, argument marshaling) plus the `xgameruntime.dll` provider entry points routed onto the existing `pg6_gdk_*` shims; shared by `d3d12_xs` (G24), DirectStorage and GameInput interfaces.
- [ ] **O17 — Unified Boot Import Coverage Wiring (`runtime/os/shim_exports.cpp` + `tools/check_import_coverage.py`)** — `UNCLAIMED` *(wait for O13/O14/O15/G24 merge)*
  - Wire `pg6_xbox_scarlett_register_all`, `pg6_ucrt_register_all`, `pg6_xbox_middleware_register_all`, `pg6_win32_compat_register_all`, and `pg6_d3d12_xs_register_all` into `shim_exports.cpp` and verify 100% coverage of all non-guest-DLL imports in `docs/imports_required.md`.
- [ ] **O18 — Contiguous Guest Address Space: Offset-Mapped `VirtualMemoryArena` Backing for wasm64 (`runtime/os/vmem.{h,cpp}`)** — `IN PROGRESS` (#187, @Timon-ok) *(lifted code reads guest memory directly in linear memory; the arena keeps pages in separate host allocations)*
  - Back the arena with one contiguous region and a single guest-base translation shared by the lifter's `guest_ptr` seam, the import-bridge seam, the guest runtime and the image loader, on host builds and under Emscripten `-sMEMORY64`, keeping commit/protect bookkeeping.

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
- [x] **G6 — D3D12-XS Descriptor Heaps, Root Signatures & Resource State Tracker** — `DONE` (PR #110, closes #44) *(no game files required — disjoint files `runtime/gpu/descriptor_heap.{h,cpp}`, `tests/test_descriptor_heap.cpp`)*
  - Implement CPU/GPU descriptor heap tables (`CBV_SRV_UAV`, `SAMPLER`, `RTV`, `DSV`), root signature parameter mapping, and dirty-bind-group caching that compiles D3D12 descriptor tables into WebGPU `CreateBindGroup`/`SetBindGroup` calls.
- [x] **G7 — D3D12 PSO Cache, Render Targets & Viewport/Scissor State** — `DONE` (PR #109, closes #45) *(no game files required — disjoint files `runtime/gpu/pso_cache.{h,cpp}`, `tests/test_pso_cache.cpp`)*
  - Hash D3D12 graphics & compute PSO descriptors (blend/depth/rasterizer/vertex-layout/shader hashes) to deduplicate WebGPU pipeline creation and track active RTV/DSV/viewport/scissor state per command list.
- [x] **G8 — Binary GPU Command Stream Disassembler & Trace Inspector** — `DONE` (PR #49, closes #32) *(no game files required — disjoint files `tools/gpu_trace.py`, `tests/test_gpu_trace.py`)*
  - Parse recorded `playgta6` GPU command ring byte streams into human-readable/JSON frame traces with opcode histograms, byte counts, and malformed-packet diagnostics for debugging M3 frames.
- [x] **G9 — WebGPU `OffscreenCanvas` Swapchain, Depth-Stencil & Resize Manager** — `DONE` (PR #59, closes #54) *(no game files required — disjoint files `web/workers/wgpu_swapchain.js`, `tests/test_wgpu_swapchain.mjs`)*
  - Manage `OffscreenCanvas` WebGPU context config, DPR-aware dynamic resize, `depth24plus`/`depth32float` depth-stencil attachment lifecycle, `device.lost` recovery, and frame-pacing telemetry.
- [x] **G10 — BCn / Texture Format & Mip-Chain Upload Helpers** — `DONE` (PR #66, closes #57) *(no game files required — disjoint files `web/workers/texture_formats.js`, `tests/test_texture_formats.mjs`)*
  - Map DXGI/Xbox texture formats (`BC1`–`BC7`, `RGBA8`, `RGBA16F`, depth/stencil) to WebGPU `GPUTextureFormat`, compute 256-byte-aligned `bytesPerRow` + mip dimensions, and provide CPU fallback BC1/BC3 decompression when WebGPU `texture-compression-bc` is absent.
- [x] **G11 — WebGPU Pipeline & BindGroupLayout Cache Manager (`web/workers/pipeline_cache.js`)** — `DONE` (PR #75, closes #62) *(no game files required — disjoint files `web/workers/pipeline_cache.js`, `tests/test_pipeline_cache.mjs`)*
  - Deduplicate `GPURenderPipeline` and `GPUComputePipeline` creation in JS by descriptor hash, support async `createRenderPipelineAsync` warm-up, and consume shader reflection metadata (`S6`).
- [x] **G12 — Update `tools/gpu_trace.py` for G5 Opcodes `12..21`** — `DONE` (PR #81, closes #67) *(no game files required — disjoint files `tools/gpu_trace.py`, `tests/test_gpu_trace.py`)*
  - Add full typed payload decoding for `CreateTexture`..`DispatchCompute` (`12..21`), compute pass summaries, and texture/bind-group histograms matching the merged `G5` ABI.
- [x] **G13 — WebGPU Shader/Pipeline Diagnostics & Capture Replay (`web/workers/gpu_recorder.js`)** — `DONE` (PR #86, closes #72) *(no game files required — disjoint files `web/workers/gpu_recorder.js`, `tests/test_gpu_recorder.mjs`)*
  - Ring-buffered in-browser frame recorder that captures raw binary GPU command streams (`BeginFrame`..`Present`) and exports downloadable `.pg6trace` blobs compatible with `tools/gpu_trace.py`.
- [x] **G14 — WebGPU Staging Buffer Pool & Upload Ring (`web/workers/staging_pool.js`)** — `DONE` (PR #98, closes #82) *(no game files required — disjoint files `web/workers/staging_pool.js`, `tests/test_staging_pool.mjs`)*
  - Sub-allocating ring of `MAP_WRITE | COPY_SRC` / `writeBuffer` staging chunks with fence-driven recycling to eliminate per-frame buffer allocation churn in `wgpu_worker.js`.
- [x] **G15 — Live WebGPU Smoke Demo Runner Wiring (`web/demo/run_smoke_demo.js`)** — `DONE` (PR #99, closes #88) *(no game files required — disjoint files `web/demo/run_smoke_demo.js`, `tests/test_run_smoke_demo.mjs`)*
  - Wires `SmokeSceneController` (`W16`) + `GpuRingBuffer` + `WgpuSwapchain` (`G9`) + `WgpuCommandProcessor` + `GpuFrameRecorder` (`G13`) + `PerfMonitor` (`W15`) into a turnkey browser `requestAnimationFrame` driver for live WebGPU testing.
- [x] **G16 — WebGPU Timestamp Query & Pass GPU-Time Profiler (`web/workers/gpu_timer.js`)** — `DONE` (PR #104, closes #94) *(no game files required — disjoint files `web/workers/gpu_timer.js`, `tests/test_gpu_timer.mjs`)*
  - Wraps optional WebGPU `'timestamp-query'` `GPUQuerySet` resolution (gracefully degrading when absent) to measure per-pass render/compute GPU timings in milliseconds.
- [x] **G17 — S6 Reflection Manifest Loader & Auto-BindGroup Builder (`web/workers/reflection_loader.js`)** — `DONE` (PR #126, closes #101) *(no game files required — disjoint files `web/workers/reflection_loader.js`, `tests/test_reflection_loader.mjs`)*
  - Parses `S6` (`shaders/src/reflect.rs`) reflection JSON manifests in JS and builds exact `GPUBindGroupLayout` + `GPUPipelineLayout` descriptors via `WebGpuPipelineCache` (`G11`).
- [x] **G18 — WebGPU Dynamic Uniform Ring & Draw-Call Constant Batcher (`web/workers/uniform_ring.js`)** — `DONE` (PR #124, closes #106) *(no game files required — disjoint files `web/workers/uniform_ring.js`, `tests/test_uniform_ring.mjs`)*
  - 256-byte-aligned dynamic uniform buffer ring (`hasDynamicOffset: true`) for per-draw constant uploads (D3D12 root constants / CBVs) with single-upload per-frame flush.
- [x] **G19 — WebGPU Indirect Draw & Compute-Culling Buffer Helper (`web/workers/indirect_draw.js`)** — `DONE` (PR #134, closes #112) *(no game files required — disjoint files `web/workers/indirect_draw.js`, `tests/test_indirect_draw.mjs`)*
  - Encodes and validates WebGPU `drawIndirect` / `drawIndexedIndirect` / `dispatchWorkgroupsIndirect` command buffers for compute-driven RDNA2 meshlet culling (`S5`).
- [ ] **G20 — Render Pipeline & Render Pass Packet ABI v2 + D3D12 State Bridge (`runtime/gpu/state_bridge`)** — `IN PROGRESS` (#121, @Timon-ok) *(no game files required — small ABI PR first, then disjoint files `runtime/gpu/state_bridge.{h,cpp}`, `tests/test_state_bridge.cpp`)*
  - Add packets for full pipeline keys, render pass begin/end with load/store ops, and viewport/scissor/blend-constant/stencil-ref, then wire the G6 descriptor/root-signature tracker and G7 PSO cache/state tracker into packet emission.
- [x] **G21 — WebGPU Resource Registry & Frame-Deferred Garbage Collector (`web/workers/resource_registry.js`)** — `DONE` (PR #145, closes #127) *(no game files required — disjoint files `web/workers/resource_registry.js`, `tests/test_resource_registry.mjs`)*
  - Tracks GPU buffer/texture/view/sampler/bind-group/pipeline handles with VRAM byte accounting and frame-epoch deferred destruction after `onSubmittedWorkDone` fences.
- [x] **G22 — WebGPU Transient Render-Target Pool & Attachment Aliasing (`web/workers/rt_pool.js`)** — `DONE` (PR #144, closes #128) *(no game files required — disjoint files `web/workers/rt_pool.js`, `tests/test_rt_pool.mjs`)*
  - Pools transient color/depth/shadow/post-FX `GPUTexture` + view attachments by descriptor key with per-frame reuse and idle/budget LRU eviction.
- [x] **G23 — Wire `GpuStagingPool` (`G14`), `DynamicUniformRing` (`G18`), `GpuPassTimer` (`G16`) & `ReflectionLoader` (`G17`) into `web/workers/wgpu_worker.js`** — `DONE` (PR #180, closes #172) *(no game files required — disjoint files `web/workers/wgpu_worker.js`, `tests/test_wgpu_worker_integration.mjs`)*
  - Integrate pooled staging uploads, dynamic uniform batching, optional timestamp profiling, reflected bind-group layouts, resource registry, and render-target pool into the main WebGPU command processor.
- [ ] **G24 — `d3d12_xs.dll` C-ABI Entrypoints & COM Vtable Bridge (`runtime/gpu/d3d12_xs_exports.{h,cpp}`)** — `IN PROGRESS` (#168, @Timon-ok) *(driven by M2 `docs/imports_required.md`: `D3D12XboxCreateDevice`, `D3D12ConfigureVirtualMemoryX`, `D3D12SerializeVersionedRootSignature`)*
  - Expose the exact 3 `d3d12_xs.dll` exports with Win64-ABI COM vtables (`ID3D12Device`, `ID3D12CommandQueue`, `ID3D12GraphicsCommandList`, `ID3D12Resource`, `ID3D12DescriptorHeap`, `ID3D12PipelineState`) bridged to `G6`/`G7`/`G20`.

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
- [x] **S5 — Wave64/Subgroup Emulation & Mesh-Compute WGSL Lowering Library** — `DONE` (PR #137, closes #46) *(no game files required — wait for S4 or keep disjoint in `shaders/src/wave_emul.rs`)*
  - Provide reusable WGSL workgroup-shared-memory emulation helpers for RDNA2 Wave64 ballot/shuffle/prefix-sum ops and indirect compute-driven meshlet culling/draw emission when WebGPU `subgroups` is unavailable.
- [x] **S6 — Root-Signature-to-WebGPU `BindGroupLayout` Reflection & Cache Manifest** — `DONE` (PR #92, closes #47) *(no game files required — disjoint file `shaders/src/reflect.rs`)*
  - Reflect parsed `naga::Module` resource bindings, vertex inputs, and workgroup sizes into compact JSON metadata alongside `shader-index.json` so `wgpu_worker.js` can build exact `GPUBindGroupLayout`s without runtime string parsing.
- [x] **S7 — Offline `WGPK` Shader Pack Inspector, Diff & WebGPU Limit Budget CLI (`tools/inspect_wgsl_pack.py`)** — `DONE` (PR #146, closes #129) *(no game files required — disjoint files `tools/inspect_wgsl_pack.py`, `tests/test_inspect_wgsl_pack.py`)*
  - Inspects, diffs, and audits compiled `.wgsl.pack` bundles against WebGPU group/binding/workgroup limits and feature directives (`f16`, `subgroups`).
- [ ] **S8 — Shader Container Format Scanner & DXBC/DXIL/RDNA2 Blob Classifier (`tools/scan_shaders.py`)** — `IN PROGRESS` (#181, @MarcBlattmann) *(no game files required — disjoint files `tools/scan_shaders.py`, `tests/test_scan_shaders.py`)*
  - Read-only local scanner classifying shader containers (`DXBC`, `DXIL`, `SPIR-V`, or Xbox Scarlett RDNA2 ISA signatures) in a local directory into a sanitized JSON summary for M2/M4 shader planning.

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
- [x] **W15 — Frame Telemetry Ring & Performance Profiler (`web/telemetry/perf_monitor.js`)** — `DONE` (PR #71, closes #64) *(no game files required — disjoint files `web/telemetry/perf_monitor.js`, `tests/test_perf_monitor.mjs`)*
  - Lock-free rolling frame-time, p95/p99 latency, HTTP Range throughput, GPU packet rate, and audio underrun aggregator feeding `window.playgta6Hud.updateMetrics()`.
- [x] **W16 — Browser WebGPU Smoke Demo & Built-in Intro Splash Test Scene (`web/demo/smoke_scene.js`)** — `DONE` (PR #79, closes #68) *(no game files required — disjoint files `web/demo/smoke_scene.js`, `tests/test_smoke_scene.mjs`)*
  - Self-contained binary command stream generator exercising `G5` (`CreateTexture`, `CreateSampler`, `CreateBindGroup`, `SetIndexBuffer`, `DrawIndexed`, `DispatchCompute`) + `WgpuSwapchain` so opening `http://127.0.0.1:8080` renders a live textured 3D/compute frame in Chrome/Edge even before `runtime.wasm` is compiled.
- [x] **W17 — Emscripten Wasm64 Build Script for Linux/macOS (`runtime/build_wasm.sh`) & CI Artifact Check** — `DONE` (PR #76, closes #69) *(no game files required — disjoint files `runtime/build_wasm.sh`, `tests/test_build_scripts.py`)*
  - Cross-platform POSIX shell counterpart to `runtime/build_wasm.ps1` (`--check-only`, `-sMEMORY64=1 -pthread`, `web/dist/runtime.{js,wasm}` output).
- [x] **W18 — Streaming HTTPFS Priority Scheduler & Prefetch Queue (`web/workers/io_scheduler.js`)** — `DONE` (PR #87, closes #73) *(no game files required — disjoint files `web/workers/io_scheduler.js`, `tests/test_io_scheduler.mjs`)*
  - Priority queue (critical boot/sync reads > streaming world cells > background prefetch) with cancellation, retry backoff, and concurrency throttling in front of `RangeCache`.
- [x] **W19 — Binary Wasm64 Memory64/Threads & ABI Import Validator (`tools/check_wasm_exports.py`)** — `DONE` (PR #90, closes #74) *(no game files required — disjoint files `tools/check_wasm_exports.py`, `tests/test_wasm_exports.py`)*
  - Binary `.wasm` section/import/export validator verifying a compiled `runtime.wasm` declares `memory64` (`i64` memory), shared memory (`threads`), and exact `env.host_*` ABI imports from `docs/wasm_abi.md`.
- [x] **W20 — Gamepad Remapper & Input Profile Persistence (`web/input/input_profiles.js`)** — `DONE` (PR #96, closes #83) *(no game files required — disjoint files `web/input/input_profiles.js`, `tests/test_input_profiles.mjs`)*
  - Configurable keyboard/mouse/gamepad presets, sensitivity/deadzone curves, and localStorage/IndexedDB profile persistence feeding `InputBridge`.
- [x] **W21 — Structured Crash Reporter & Runtime Diagnostic Bundle Exporter (`web/telemetry/crash_report.js`)** — `DONE` (PR #97, closes #84) *(no game files required — disjoint files `web/telemetry/crash_report.js`, `tests/test_crash_report.mjs`)*
  - Captures Wasm traps, WebGPU validation/device-lost errors, recent runtime logs, and browser capability fingerprints into sanitized shareable JSON reports (never including game data or host paths).
- [x] **W22 — Browser Emscripten Runtime Session Controller (`web/runtime/session_controller.js`)** — `DONE` (PR #100, closes #89) *(no game files required — disjoint files `web/runtime/session_controller.js`, `tests/test_session_controller.mjs`)*
  - Orchestrates worker startup (`io_worker`, `wgpu_worker`, `audio_engine`), `InputBridge`, `PerfMonitor`, `BrowserStorageManager`, and clean restart/fallback into `SmokeSceneController` when `dist/runtime.wasm` is absent.
- [x] **W23 — Local Asset Pipeline Orchestrator CLI (`tools/prepare_local.py`)** — `DONE` (PR #105, closes #95) *(no game files required — disjoint files `tools/prepare_local.py`, `tests/test_prepare_local.py`)*
  - Single-command dry-run/execute helper that runs `scan` → `xvd`/`rpf9`/`manifest`/`bootset` → `recomp inspect` → `shaders convert` against `GTA6_GAME_DIR` into gitignored `local-data/` & `work/`.
- [x] **W24 — Browser HUD Action Controls & Interactive Diagnostics Wiring (`web/ui/hud_controller.js`)** — `DONE` (PR #122, closes #102) *(no game files required — disjoint files `web/ui/hud_controller.js`, `tests/test_hud_controller.mjs`)*
  - Binds HUD buttons (Clear OPFS Cache, Export Saves, Capture GPU Trace, Export Crash Report, Switch Input Profile, Toggle Smoke Demo) to `BrowserStorageManager`, `GpuFrameRecorder`, `CrashReporter`, and `SessionController`.
- [x] **W25 — LAN / Private-Host CORS & Range Streaming Benchmark CLI (`tools/bench_range_server.py`)** — `DONE` (PR #111, closes #103) *(no game files required — disjoint files `tools/bench_range_server.py`, `tests/test_bench_range_server.py`)*
  - Measures concurrent HTTP `206` Range throughput, latency percentiles, and `If-None-Match` (`304`) overhead against `server/serve_local.py` on synthetic payloads.
- [x] **W26 — Browser Diagnostics & Smoke Demo Bootstrap Glue (`web/app_bootstrap.js`)** — `DONE` (PR #125, closes #107) *(no game files required — disjoint files `web/app_bootstrap.js`, `tests/test_app_bootstrap.mjs`)*
  - Clean ES module entrypoint connecting `RuntimeSessionController`, `SmokeDemoRunner`, `InputProfileManager`, `AudioEngine`, `BrowserStorageManager`, and `CrashReporter` for `web/index.html`.
- [x] **W27 — Multi-Track Integration Check & CI Coverage Summary Script (`tools/repo_health.py`)** — `DONE` (PR #123, closes #108) *(no game files required — disjoint files `tools/repo_health.py`, `tests/test_repo_health.py`)*
  - Verifies zero ABI drift between `runtime/abi/playgta6_abi.h`, `tools/gpu_trace.py`, `web/workers/wgpu_worker.js`, `web/input/input_bridge.js`, and `docs/WORKSTREAMS.md` in one fast stdlib check.
- [x] **W28 — Streaming Audio Bank / AWC Header Inspector (`tools/src/awc.rs`)** — `DONE` (PR #136, closes #113) *(no game files required — disjoint file `tools/src/awc.rs`)*
  - Read-only RAGE `.awc` audio bank container header parser (stream/chunk counts, codec tag PCM/IMA-ADPCM/XMA2/Opus, sample rate, channel count) without extracting or committing audio data.
- [x] **W29 — Web Worker Pool & Offscreen Task Dispatcher (`web/workers/worker_pool.js`)** — `DONE` (PR #135, closes #114) *(no game files required — disjoint files `web/workers/worker_pool.js`, `tests/test_worker_pool.mjs`)*
  - Reusable bounded Web Worker pool supporting transferable `ArrayBuffer` jobs, cancellation, and graceful crash recovery for background decompression & shader pre-parsing.
- [x] **W30 — Wire `web/index.html` to `web/app_bootstrap.js` & `web/ui/hud_controller.js` (`web/index.html`)** — `DONE` (PR #143, closes #130) *(no game files required — disjoint files `web/index.html`, `tests/test_index_html_wiring.mjs`)*
  - Adds compact interactive HUD action controls and replaces the legacy hardcoded worker bootstrap with `bootstrapPlayGta6App` + `HudController` so opening `http://127.0.0.1:8080` renders live WebGPU smoke frames or boots `runtime.wasm`.
- [x] **W31 — Browser Streaming Manifest & Chunk Integrity Verifier (`web/storage/manifest_verifier.js`)** — `DONE` (PR #142, closes #131) *(no game files required — disjoint files `web/storage/manifest_verifier.js`, `tests/test_manifest_verifier.mjs`)*
  - Validates `data-manifest.json` and `bootset.json` schemas, computes partial-download streaming readiness percentages, and probes live HTTP `206` Range / `ETag` consistency.
- [x] **W32 — Wire `IoPriorityScheduler` (`W18`) & `Rpf9Vfs` (`W11`) into `web/workers/io_worker.js`** — `DONE` (PR #166, closes #153) *(no game files required — disjoint files `web/workers/io_worker.js`, `tests/test_io_worker_integration.mjs`)*
  - Connect prioritized Range scheduling and on-demand RPF9 virtual sub-file streaming into the `HttpfsMailbox` worker loop.
- [x] **W33 — Headless Live Boot & HUD E2E Verification Harness (`tests/test_live_boot_e2e.mjs`)** — `DONE` (PR #155, closes #154) *(no game files required — disjoint file `tests/test_live_boot_e2e.mjs`)*
  - End-to-end test spinning up `server/serve_local.py`, fetching `/`, `/api/status`, and JS modules with `COOP`/`COEP` headers, and running `bootstrapPlayGta6App` + `HudController` through smoke-demo frames, trace export, and diagnostic bundle generation.
- [x] **W34 — Real RPF9 Little-Endian `0x52504639` (`"9FPR"`) Magic Support (`tools/src/rpf9.rs`, `web/workers/rpf9_vfs.js`)** — `DONE` (PR #147, closes #140) *(driven by M2 inspection of real Xbox `.rpf` archives)*
  - Accept little-endian `0x52504639` (`39 46 50 52` = `"9FPR"`) alongside legacy big-endian `"RPF9"` headers in both the Rust CLI inspector and browser streaming RPF9 VFS reader.
- [x] **W35 — Real RPF9 Header v9 Field Layout & Encryption-Flag Diagnostic (`tools/src/rpf9.rs`)** — `DONE` (PR #179, closes #171) *(driven by M2 inspection of real `appdata0.rpf`/`common.rpf`: word 4 is a 32-bit hash/flag rather than `names_length`)*
  - Parse the real Xbox Series RPF9 header words cleanly without rejecting valid archives as truncated, and report header layout/entry/encryption metadata cleanly.
- [x] **W36 — Wasm64 Memory64 Runtime & BigInt64 Host Import Integration Tests (`tests/test_wasm64_runtime.mjs`)** — `DONE` (PR #177, closes #173) *(no game files required — disjoint file `tests/test_wasm64_runtime.mjs`)*
  - Verifies synthetic Wasm64 `i64` memory marshaling with `web/loader.js` `createHostImports` and validates local `web/dist/runtime.wasm` ABI exports when present.
- [ ] **W37 — Real-Binary Import Coverage Auditor CLI (`tools/check_import_coverage.py`)** — `IN PROGRESS` (#182, @MarcBlattmann) *(no game files required — disjoint files `tools/check_import_coverage.py`, `tests/test_import_coverage.py`)*
  - Parses all 68 DLLs / 587 symbols in `docs/imports_required.md`, audits live `runtime/` shim & registry coverage vs `R12` guest-recompiled DLLs, and reports exact remaining symbols.
- [ ] **W38 — WebGPU Bink2 / Splash Video Frame Presenter (`web/workers/bink_presenter.js`)** — `IN PROGRESS` (#184, @MarcBlattmann) *(no game files required — disjoint files `web/workers/bink_presenter.js`, `tests/test_bink_presenter.mjs`)*
  - Double-buffered WebGPU YUV420p/YUVA/RGBA8 texture uploader + BT.709/BT.601 WGSL full-screen letterbox presenter for intro/splash frames (`Bink2Scarlett.dll`).
- [ ] **W39 — `MicrosoftGame.Config` & Xbox Package Readiness Inspector CLI (`tools/inspect_game_assets.py`)** — `IN PROGRESS` (#185, @MarcBlattmann) *(no game files required — disjoint files `tools/inspect_game_assets.py`, `tests/test_inspect_game_assets.py`)*
  - Safely parses Xbox GDK `MicrosoftGame.Config` XML and verifies declared executables, bundled DLLs, and `ShellVisuals` splash assets in `GTA6_GAME_DIR` with sanitized JSON output.
