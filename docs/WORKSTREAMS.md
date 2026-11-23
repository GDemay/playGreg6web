# Workstreams & Parallel Tasks

Work is split into **5 independent tracks** so many contributors (and AI agents) can work in parallel without merge conflicts. Tracks only communicate through the frozen contracts in `docs/io_manifest.md`, `docs/wasm_abi.md`, and `docs/gpu_commands.md`.

> If a task needs an ABI change, open a small PR touching `docs/` + `runtime/abi/playgta6_abi.h` first and merge it before dependent feature work.

---

## Track 1: Recompiler (`recomp/`, Rust)
**Goal:** Lift the Xbox Series x86-64 PE executable into Wasm64 (`memory64`).

- [ ] **R1 — PE Parser & Import Dumper** — `IN PROGRESS` (#1, @MarcBlattmann) *(needs `GTA6_GAME_DIR` for real test; unit-testable with synthetic PE)*
  - Parse PE64 headers, sections, exports, TLS callbacks, and import tables (`object` / `goblin` crate).
  - Emit a sanitized symbol-only report (`docs/imports_required.md` — DLL + function names only, zero game bytes).
- [ ] **R2 — x86-64 Decoder & Function Discovery** — `IN PROGRESS` (#13, @Timon-ok) *(no game files required for unit tests)*
  - Integrate `iced-x86` to decode `.text` blocks, follow calls/jumps, and build a CFG + relocation map.
- [ ] **R3 — LLVM IR → Wasm64 Lifter Prototype** — `IN PROGRESS` (#14, @Timon-ok) *(no game files required)*
  - Lift small synthetic x86-64 test functions (arithmetic, branches, memory loads/stores, Win64 ABI calls) to LLVM IR and compile to `wasm64`.

---

## Track 2: Xbox / Win32 OS Shims (`runtime/os/`, C++20 + Emscripten)
**Goal:** Reimplement console OS APIs backed by browser workers, pthreads, and HTTPFS.

- [ ] **O1 — Emscripten Wasm64 + Pthreads Smoke Build** *(no game files required; waits on W1+W2 & G2)*
  - Wire `runtime/main.cpp` JS glue (`web/loader.js` imports for `host_log`, `host_wgpu_submit`) and verify `runtime/build_wasm.ps1` boots in Chrome/Edge.
- [x] **O2 — Blocking HTTPFS Bridge (`io_worker.js` ↔ `runtime/os/`)** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Implement synchronous worker-to-worker `Atomics.wait` range reads over a `SharedArrayBuffer` so `host_httpfs_open`/`read`/`close` work from pthreads.
- [x] **O3 — Virtual Memory & Threading/TLS Shims** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Implement Win32/Xbox `VirtualAlloc`/`VirtualProtect` arena mapping, `TlsAlloc`, SRW locks, and `XTaskQueue` on top of Emscripten pthreads.
- [x] **O4 — `GameInput` & `/userdata` Shims** — `DONE` (PR #12, closes #3) *(no game files required)*
  - Read `playgta6::InputState` from the input `SharedArrayBuffer` and map save paths to IndexedDB `/userdata`.
- [ ] **O5 — Xbox GDK `xgameruntime` & Win32 Kernel/File/Sync C-ABI Shims** — `IN PROGRESS` (#16, @MarcBlattmann) *(no game files required)*
  - Expose C-ABI Win32 & Xbox GDK imports (`XGameRuntimeInitialize`, `XTaskQueue*`, `XUser*` offline stub, `VirtualAlloc`/`Protect`/`Free`, `Tls*`, `SRWLock*`, `CriticalSection`/`Event`, `CreateFileW`/`ReadFile`/`SetFilePointerEx`/`GetFileSizeEx`/`CloseHandle`, `QueryPerformanceCounter`/`Frequency`) backed by our OS shims.

---

## Track 3: GPU Command Bridge (`runtime/gpu/` + `web/workers/wgpu_worker.js`, C++20 + JS)
**Goal:** Translate `d3d12_xs` command lists to binary packets executed on WebGPU.

- [x] **G1 — Expand Command Ring & Resource Tables** — `DONE` (PR #7, closes #4) *(no game files required)*
  - Implement `CreateBuffer`, `WriteBuffer`, `CreateShaderModule`, `CreatePipeline`, and `Draw` packets across `runtime/abi/playgta6_abi.h`, `runtime/gpu/cmd_encoder.cpp`, and `web/workers/wgpu_worker.js`.
- [ ] **G2 — SharedArrayBuffer Zero-Copy GPU Ring** — `IN PROGRESS` (#10, @MarcBlattmann) *(no game files required)*
  - Replace `postMessage` buffer copies with a lock-free SPSC `SharedArrayBuffer` ring buffer between Wasm64 and `wgpu_worker.js`.
- [x] **G3 — `d3d12_xs` Interface Stubs & Trace Harness** — `DONE` (PR #7, closes #4) *(no game files required)*
  - Implement minimal COM-like `ID3D12Device` / `ID3D12GraphicsCommandList` / `PresentX` shim layer that emits packets matching `docs/gpu_commands.md`.
- [ ] **G4 — `WGPK` `.wgsl.pack` Binary Loader & Shader Module Cache** — `IN PROGRESS` (#18, @MarcBlattmann) *(no game files required)*
  - Parse `.wgsl.pack` binary bundles (`shaders/src/pack.rs`) in JS (`web/workers/wgsl_pack_loader.js`), resolve WGSL by 64-bit FNV-1a hash, and cache compiled `GPUShaderModule`s.

---

## Track 4: Shader Pipeline (`shaders/`, Rust)
**Goal:** Convert Xbox RDNA2 shaders (`--mesh-compute`, `--wave64-emulate`) → SPIR-V → WGSL packs.

- [x] **S1 — `naga` SPIR-V → WGSL Batch Converter** — `DONE` (PR #9, closes #5) *(testable with synthetic SPIR-V; real corpus needs local SPIR-V)*
  - Add `naga` to `shaders/Cargo.toml`, convert a directory of `.spv` files to `.wgsl`, emit `shader-index.json`, and output a structured failure-reason report (unsupported capabilities/extensions).
- [x] **S2 — Shader Pack Bundler & WebGPU Pre-Validator** — `DONE` (PR #9, closes #5) *(no game files required)*
  - Pack lowered `.wgsl` files into compact `.wgsl.pack` bundles (like `playgta5`) for fast loading in `wgpu_worker.js`.
- [ ] **S3 — WebGPU Compatibility Passes** — `IN PROGRESS` (#11, @MarcBlattmann) *(needs SPIR-V failure report from S1)*
  - Implement SPIR-V/WGSL rewrite passes for unsupported WebGPU features (bindless indexing limits, subgroup/wave64 fallbacks, mesh-compute lowering).

---

## Track 5: Browser Shell, Streaming I/O & Audio (`web/`, `server/`, `tools/`, JS + Python + Rust)
**Goal:** Fast, resilient browser loading, streaming, caching, and audio playback.

- [ ] **W1 — OPFS + LRU Range Cache in `io_worker.js`** — `IN PROGRESS` (#2, @MarcBlattmann) *(no game files required)*
  - Cache fetched HTTP `206` byte chunks in Origin Private File System (OPFS) with a configurable budget + `bootset.json` prefetch on startup.
- [ ] **W2 — IndexedDB `/userdata` Save Persistence** — `IN PROGRESS` (#2, @MarcBlattmann) *(no game files required)*
  - Restore `/userdata/` files into Wasm FS before `main()` and flush writes asynchronously (mirroring `playgta5`'s `loader.js` pattern).
- [x] **W3 — `audio_worker.js` Opus / WebAudio Pipeline** — `DONE` (PR #8, closes #6) *(no game files required)*
  - Implement `AudioWorklet` ring buffer + Ogg Opus stream decoding.
- [x] **W4 — Local Environment & Diagnostics (`tools/`)** — `DONE` (PR #8, closes #6) *(uses `GTA6_GAME_DIR` when available)*
  - Extend `tools/` to generate `bootset.json`, inspect RPF9 archive headers without extracting 100+ GB, and verify local file integrity.
- [ ] **W5 — Full `InputBridge` (Gamepad API + Xbox Buttons/Sticks/Triggers + Pointer Lock)** — `IN PROGRESS` (#17, @MarcBlattmann) *(no game files required)*
  - Populate the full 40-byte `playgta6::InputState` SAB from Pointer Lock mouse, keyboard, and `navigator.getGamepads()` matching `runtime/os/game_input.h`.
- [ ] **W6 — Partial-Download Game Folder Scanner & Incremental Manifest CLI (`tools/`)** — `IN PROGRESS` (#19, @MarcBlattmann) *(works on partial or complete `GTA6_GAME_DIR`)*
  - Add read-only `scan` command detecting completed vs downloading files (`.!qB`/`.part`/sparse) and `--skip-incomplete` for `manifest`/`bootset`.
- [ ] **W7 — Local Server Hardening (`/api/status`, ETag/If-Range) & Browser Diagnostics HUD (`server/`, `web/index.html`)** — `IN PROGRESS` (#20, @MarcBlattmann) *(no game files required)*
  - Add ETag/`If-Range`/`HEAD` + sanitized `/api/status` to `server/serve_local.py` and upgrade `web/index.html` with a live Wasm64/WebGPU/SAB HUD.
