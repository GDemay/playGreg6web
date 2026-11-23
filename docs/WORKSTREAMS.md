# Workstreams & Parallel Tasks

Work is split into **5 independent tracks** so many contributors (and AI agents) can work in parallel without merge conflicts. Tracks only communicate through the frozen contracts in `docs/io_manifest.md`, `docs/wasm_abi.md`, and `docs/gpu_commands.md`.

> If a task needs an ABI change, open a small PR touching `docs/` + `runtime/abi/playgta6_abi.h` first and merge it before dependent feature work.

---

## Track 1: Recompiler (`recomp/`, Rust)
**Goal:** Lift the Xbox Series x86-64 PE executable into Wasm64 (`memory64`).

- [ ] **R1 — PE Parser & Import Dumper** *(needs `GTA6_GAME_DIR` for real test; unit-testable with synthetic PE)*
  - Parse PE64 headers, sections, exports, TLS callbacks, and import tables (`object` / `goblin` crate).
  - Emit a sanitized symbol-only report (`docs/imports_required.md` — DLL + function names only, zero game bytes).
- [ ] **R2 — x86-64 Decoder & Function Discovery** *(no game files required for unit tests)*
  - Integrate `iced-x86` to decode `.text` blocks, follow calls/jumps, and build a CFG + relocation map.
- [ ] **R3 — LLVM IR → Wasm64 Lifter Prototype** *(no game files required)*
  - Lift small synthetic x86-64 test functions (arithmetic, branches, memory loads/stores, Win64 ABI calls) to LLVM IR and compile to `wasm64`.

---

## Track 2: Xbox / Win32 OS Shims (`runtime/os/`, C++20 + Emscripten)
**Goal:** Reimplement console OS APIs backed by browser workers, pthreads, and HTTPFS.

- [ ] **O1 — Emscripten Wasm64 + Pthreads Smoke Build** *(no game files required)*
  - Wire `runtime/main.cpp` JS glue (`web/loader.js` imports for `host_log`, `host_wgpu_submit`) and verify `runtime/build_wasm.ps1` boots in Chrome/Edge.
- [ ] **O2 — Blocking HTTPFS Bridge (`io_worker.js` ↔ `runtime/os/`)** *(no game files required)*
  - Implement synchronous worker-to-worker `Atomics.wait` range reads over a `SharedArrayBuffer` so `host_httpfs_open`/`read`/`close` work from pthreads.
- [ ] **O3 — Virtual Memory & Threading/TLS Shims** *(no game files required)*
  - Implement Win32/Xbox `VirtualAlloc`/`VirtualProtect` arena mapping, `TlsAlloc`, SRW locks, and `XTaskQueue` on top of Emscripten pthreads.
- [ ] **O4 — `GameInput` & `/userdata` Shims** *(no game files required)*
  - Read `playgta6::InputState` from the input `SharedArrayBuffer` and map save paths to IndexedDB `/userdata`.

---

## Track 3: GPU Command Bridge (`runtime/gpu/` + `web/workers/wgpu_worker.js`, C++20 + JS)
**Goal:** Translate `d3d12_xs` command lists to binary packets executed on WebGPU.

- [ ] **G1 — Expand Command Ring & Resource Tables** *(no game files required)*
  - Implement `CreateBuffer`, `WriteBuffer`, `CreateShaderModule`, `CreatePipeline`, and `Draw` packets across `runtime/abi/playgta6_abi.h`, `runtime/gpu/cmd_encoder.cpp`, and `web/workers/wgpu_worker.js`.
- [ ] **G2 — SharedArrayBuffer Zero-Copy GPU Ring** *(no game files required)*
  - Replace `postMessage` buffer copies with a lock-free SPSC `SharedArrayBuffer` ring buffer between Wasm64 and `wgpu_worker.js`.
- [ ] **G3 — `d3d12_xs` Interface Stubs & Trace Harness** *(no game files required)*
  - Implement minimal COM-like `ID3D12Device` / `ID3D12GraphicsCommandList` / `PresentX` shim layer that emits packets matching `docs/gpu_commands.md`.

---

## Track 4: Shader Pipeline (`shaders/`, Rust)
**Goal:** Convert Xbox RDNA2 shaders (`--mesh-compute`, `--wave64-emulate`) → SPIR-V → WGSL packs.

- [ ] **S1 — `naga` SPIR-V → WGSL Batch Converter** *(testable with synthetic SPIR-V; real corpus needs local SPIR-V)*
  - Add `naga` to `shaders/Cargo.toml`, convert a directory of `.spv` files to `.wgsl`, emit `shader-index.json`, and output a structured failure-reason report (unsupported capabilities/extensions).
- [ ] **S2 — Shader Pack Bundler & WebGPU Pre-Validator** *(no game files required)*
  - Pack lowered `.wgsl` files into compact `.wgsl.pack` bundles (like `playgta5`) for fast loading in `wgpu_worker.js`.
- [ ] **S3 — WebGPU Compatibility Passes** *(needs SPIR-V failure report from S1)*
  - Implement SPIR-V/WGSL rewrite passes for unsupported WebGPU features (bindless indexing limits, subgroup/wave64 fallbacks, mesh-compute lowering).

---

## Track 5: Browser Shell, Streaming I/O & Audio (`web/`, `server/`, `tools/`, JS + Python + Rust)
**Goal:** Fast, resilient browser loading, streaming, caching, and audio playback.

- [ ] **W1 — OPFS + LRU Range Cache in `io_worker.js`** *(no game files required)*
  - Cache fetched HTTP `206` byte chunks in Origin Private File System (OPFS) with a configurable budget + `bootset.json` prefetch on startup.
- [ ] **W2 — IndexedDB `/userdata` Save Persistence** *(no game files required)*
  - Restore `/userdata/` files into Wasm FS before `main()` and flush writes asynchronously (mirroring `playgta5`'s `loader.js` pattern).
- [ ] **W3 — `audio_worker.js` Opus / WebAudio Pipeline** *(no game files required)*
  - Implement `AudioWorklet` ring buffer + Ogg Opus stream decoding.
- [ ] **W4 — Local Environment & Diagnostics (`tools/`)** *(uses `GTA6_GAME_DIR` when available)*
  - Extend `tools/` to generate `bootset.json`, inspect RPF9 archive headers without extracting 100+ GB, and verify local file integrity.
