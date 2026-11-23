# End Goal — `playgta6web`

## Vision
Run **Grand Theft Auto VI (Xbox Series)** directly inside a user's modern web browser (Chrome/Edge) via **Wasm64 (`memory64`) + WebGPU + HTTP `206` Range streaming**, hosted locally or on a private family/LAN server — with **zero server-side video streaming** and **zero copyrighted files in this repository**.

## Non-Goals
- **No server-side game streaming** (no WebRTC/Moonlight video host). The client browser executes the recompiled game and renders with WebGPU.
- **No short-term fake sandbox / asset-viewer detour.** Every piece built here moves directly toward booting and running the real game binary.
- **No copyrighted files, keys, DRM bypasses, or game source code** in git, issues, or PRs.

## Definition of Done (Target Milestones)

### M0 — Foundation *(Current)*
- Hybrid repo ready: Rust pipelines (`recomp/`, `shaders/`, `tools/`), C++20/Emscripten Wasm64 runtime (`runtime/`), browser worker shell (`web/`), COOP/COEP + HTTP `206` Range server (`server/`), shared ABI contracts (`docs/`), and CI.

### M1 — Static Analysis & Asset/Shader Readiness
- `tools/` generates `data-manifest.json` + `bootset.json` from a local `GTA6_GAME_DIR`.
- `recomp/` parses the Xbox Series PE executable, dumps its complete import table (DLLs + symbols) and section layout into `docs/` (symbol names only, never binaries).
- `shaders/` converts RDNA2 → SPIR-V (`--mesh-compute`, `--wave64-emulate`) → WGSL via `naga` and produces a coverage report + `shader-index.json`.

### M2 — First Wasm64 Boot & OS Shims
- `recomp/` lifts x86-64 code sections → LLVM IR → Wasm64 object linked with `runtime/` (`-sMEMORY64=1 -pthread`).
- `runtime/os/` implements the minimal boot surface (`xgameruntime`, `XTaskQueue`, threading/TLS, virtual memory, `httpfs` Range VFS, IndexedDB `/userdata`).
- The recompiled binary initializes inside `web/loader.js` and reaches its first D3D12 device/swapchain creation calls.

### M3 — First WebGPU Frame
- `runtime/gpu/` translates `d3d12_xs` resource creation, descriptor heaps, command lists, and `PresentX` into binary command streams consumed by `web/workers/wgpu_worker.js`.
- First real frames from the game render onto the browser `OffscreenCanvas` using lowered WGSL shaders.

### M4 — Playable In-Browser Build
- Interactive input (`web/input/input_bridge.js`: Pointer Lock, keyboard/mouse, Gamepad API → `GameInput` shim).
- Streaming audio (`web/workers/audio_worker.js`: Opus/XMA2 → WebAudio).
- Persistent saves in IndexedDB (`/userdata`), OPFS range cache in `io_worker.js`, and stable multi-threaded performance on consumer GPUs.
