# Agent Instructions

## Hard Rules
- **No copyrighted game files or game source code in this repo.** Never commit or add game binaries, keys, `.rpf`/`.xvd`/`.ttpf`/`.awc` archives, extracted assets, decrypted dumps, or generated game shaders. All user-supplied game data and generated outputs belong strictly in gitignored local folders (`local-data/`, `work/`).
- **Keep `README.md` small and clean.** Do not bloat it with long checklists, redundant repo trees, or unrequested sections. Only update `README.md` when explicitly asked or when a core architectural fact changes.
- **Don't just add stuff.** Only create files, dependencies, or abstractions that are directly needed for the current task.

## Project Goal
- Run **Grand Theft Auto VI (Xbox Series)** directly inside the **user's browser** via **Wasm64 (`memory64`) + WebGPU + HTTP `206` Range streaming** (self-hosted, client-side execution — no server-side video streaming and no short-term detour).
- Reuse useful research/tools from `shadany7824/playgta5` and `ItsNotRexed/Leonida` where helpful to speed up development, without hard-coupling the repo to them.

## Folder Responsibilities
- `web/` — Browser frontend (`index.html`, `loader.js`), dedicated workers (`web/workers/`), and input bridge (`web/input/`). Keep the main thread non-blocking; require Cross-Origin Isolation (`COOP`/`COEP`) for `SharedArrayBuffer`.
- `recomp/` — **Rust** static `x86-64 -> LLVM IR -> Wasm64` recompilation pipeline.
- `runtime/` — **C++20 + Emscripten (`-sMEMORY64 -pthread`)** runtime: shared ABI (`runtime/abi/`), Xbox/Win32 OS shims (`runtime/os/`), and `d3d12_xs -> WebGPU` command bridge (`runtime/gpu/`).
- `shaders/` — **Rust** offline `RDNA2 -> SPIR-V -> WGSL` (`naga`) translation and shader packing.
- `server/` — Lightweight local HTTP server with `COOP`/`COEP` headers and HTTP `Range` (`206 Partial Content`) support.
- `tools/` — **Rust** local manifest/bootset builders and diagnostics.
- `docs/` — Technical specs, ABI contracts, and WebGPU/Wasm64 findings.
- `tests/` — Automated unit and integration tests.
- `third_party/` — Open-source submodules/dependencies only (never game files).
