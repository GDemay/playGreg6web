# End Goal — `playgta6web`

## Vision
Run **Grand Theft Auto VI (Xbox Series)** directly inside a user's modern web browser (Chrome/Edge) via **Wasm64 (`memory64`) + WebGPU + HTTP `206` Range streaming**, hosted locally or on a private family/LAN server — with **zero server-side video streaming** and **zero copyrighted files in this repository**.

## Non-Goals
- **No server-side game streaming** (no WebRTC/Moonlight video host). The client browser executes the recompiled game and renders with WebGPU.
- **No short-term fake sandbox / asset-viewer detour.** Every piece built here moves directly toward booting and running the real game binary.
- **No copyrighted files, keys, DRM bypasses, or game source code** in git, issues, or PRs.

## Big Milestones Tracker (Live Progress)

> Update checkboxes here as major end-to-end milestones are proven (not just unit-tested).

### M0 — Foundation & Multi-Contributor Infrastructure — `DONE`
- [x] Hybrid repo ready: Rust pipelines (`recomp/`, `shaders/`, `tools/`), C++20/Emscripten Wasm64 runtime (`runtime/`), browser worker shell (`web/`), COOP/COEP + HTTP `206` Range server (`server/`).
- [x] Frozen cross-track contracts (`docs/io_manifest.md`, `docs/wasm_abi.md`, `docs/gpu_commands.md`) + multi-agent task-claiming workflow (`AGENTS.md`, `docs/WORKSTREAMS.md`).
- [x] CI guardrails (No Game Files Guard, Rust workspace tests, C++ CTest, Python unittest, JS syntax & `node:test`).

### M1 — Browser Runtime Shell, Streaming I/O & Shader Tooling — `IN PROGRESS`
- [x] COOP/COEP + HTTP `206` Range + `ETag`/`If-Range` + `/api/status` local server (`server/serve_local.py`) & live diagnostics HUD (`web/index.html`).
- [x] OPFS + LRU `RangeCache`, `bootset.json` prefetch, and IndexedDB `/userdata` persistence (`web/workers/`).
- [x] Lock-free `SharedArrayBuffer` GPU command ring (`runtime/gpu/gpu_ring.h` ↔ `web/workers/gpu_ring.js`) & WebGPU command processor (`web/workers/wgpu_worker.js`).
- [x] Full `InputBridge` (Pointer Lock, mouse/keyboard, Gamepad API → 40-byte `InputState` SAB) & SAB audio pipeline (`audio_worker.js`).
- [x] SPIR-V → WGSL converter (`naga`), WebGPU compatibility sanitizer/auditor, `.wgsl.pack` bundler, and JS `WGPK` loader/cache (`wgsl_pack_loader.js`).
- [ ] First Emscripten Wasm64 + Pthreads smoke build (`dist/runtime.wasm`) booting in Chrome/Edge and rendering a real WebGPU clear/triangle frame through the SAB GPU ring (`O1` + `G5`).

### M2 — Real Game Static Analysis & Asset Readiness — `IN PROGRESS`
- [x] Pure-Rust PE64 parser & sanitized import/export/TLS dumper (`playgta6-recomp inspect`).
- [x] Non-extracting RPF9 header inspector, `bootset.json` generator, and partial-download folder scanner (`playgta6-manifest scan/rpf9/bootset`).
- [x] Run `playgta6-manifest scan --from-env` and `playgta6-recomp inspect` on a real local Xbox Series build — verified loose `.rpf` (little-endian `0x52504639` `"9FPR"` magic) + `game_scarlett_master.exe` layout (`355,103` functions, `22.6M` instructions decoded with 0 errors, `68` DLLs / `587` imports) and published sanitized [`docs/imports_required.md`](./imports_required.md) (symbol names & header metadata only, zero game bytes).
- [ ] Run `playgta6-shaders audit` / `convert` on real extracted local shaders to generate a real WebGPU blocker report + `shader-index.json` + `.wgsl.pack` locally.

### M3 — Wasm64 Recompilation & Console OS/GDK Boot Surface — `IN PROGRESS`
- [x] Core OS shims: `VirtualMemoryArena`, `TlsTable`/`SrwLock`/`XTaskQueue`, `GameInputReader`, and `HttpfsMailbox` (`runtime/os/`).
- [ ] Xbox GDK `xgameruntime` + Win32 kernel/file/sync/heap/`GetProcAddress` + offline Xbox Live shim coverage (`O5`–`O8`, driven by real `docs/imports_required.md`).
- [ ] `recomp/` x86-64 decoder + CFG + `.pdata` exception table + LLVM IR → Wasm64 lifter (`R2`–`R6`) compiling synthetic Win64 functions to working Wasm64 objects.
- [ ] Recompiled real game executable links against `runtime/` (`-sMEMORY64=1 -pthread`), passes CRT startup, and reaches its first `XGameRuntimeInitialize` + `D3D12CreateDevice` call in the browser.

### M4 — First Real Frame / Intro Splash in Browser — `UPCOMING`
- [ ] `runtime/gpu/` translates `d3d12_xs` textures, samplers, descriptor heaps, root signatures, PSOs, indexed draws, and compute passes (`G5`–`G7`) to `wgpu_worker.js`.
- [ ] Lowered WGSL shaders execute without WebGPU validation errors on consumer GPUs.
- [ ] **Big Milestone:** First real intro / splash / legal screen frames from the recompiled binary render onto the browser `OffscreenCanvas` via WebGPU.

### M5 — Title Screen, Menus & World Streaming — `UPCOMING`
- [ ] Interactive menu navigation via `InputBridge` (keyboard/mouse + Gamepad API).
- [ ] Streaming multi-GB `.rpf9` world/UI assets smoothly over HTTP `206` Range + OPFS cache without stalling main/render threads.
- [ ] Streaming audio voices (Opus/ADPCM/XMA2) playing through `audio_worker.js` / `AudioWorklet`.
- [ ] **Big Milestone:** Reach the interactive main menu and trigger world load.

### M6 — Playable In-Browser Build — `UPCOMING`
- [ ] Persistent saves in IndexedDB (`/userdata`) survive page reloads.
- [ ] Stable multi-threaded Wasm64 + WebGPU frame pacing on consumer Chrome/Edge GPUs.
- [ ] **Big Milestone:** Playable open-world gameplay running 100% client-side in the browser.
