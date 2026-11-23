# Agent Instructions

## Hard Rules
- **No copyrighted game files or game source code in this repo.** Never commit, paste, or add game binaries, keys, `.rpf`/`.xvd`/`.xvc`/`.ttpf`/`.awc` archives, extracted assets, decrypted dumps, or generated game shaders. All user-supplied game data and generated outputs belong strictly in gitignored local folders (`local-data/`, `work/`) referenced through `.env`.
- **Keep `README.md` small and clean.** Do not bloat it with long checklists, redundant repo trees, or unrequested sections. Only update `README.md` when explicitly asked or when a core architectural fact changes.
- **Don't just add stuff.** Only create files, dependencies, or abstractions that are directly needed for the current task.

## Start-Here Playbook (When a User Says "Help Work On This")
Follow these steps whenever starting a session with a new or returning contributor:

1. **Read the project context:**
   - [docs/END_GOAL.md](./docs/END_GOAL.md) — what the finished product is (and non-goals).
   - [docs/WORKSTREAMS.md](./docs/WORKSTREAMS.md) — the 5 parallel tracks and open tasks.
   - ABI contracts when touching cross-track boundaries: [docs/io_manifest.md](./docs/io_manifest.md), [docs/wasm_abi.md](./docs/wasm_abi.md), [docs/gpu_commands.md](./docs/gpu_commands.md).
2. **Check the user's local machine & game path:**
   - Run `python tools/local_env.py --check`.
   - If `GTA6_GAME_DIR` is **NOT SET**, ask the user where their local Xbox Series game folder is (or whether they want to work on a task that does not require game files).
   - When the user provides a path, save it into gitignored `.env` via `python tools/local_env.py --set-game "<path>"`. Never copy game files into the repo.
3. **Pick a track & branch cleanly:**
   - Match the user's installed toolchains (`tools/local_env.py --check`) and whether they have local game files to an open task in [docs/WORKSTREAMS.md](./docs/WORKSTREAMS.md).
   - Work on a dedicated branch named `<track>/<task-slug>` (e.g., `recomp/pe-imports`, `shaders/naga-pass`, `web/opfs-cache`) so multiple contributors don't collide on `main`.
   - If a shared ABI in `docs/` or `runtime/abi/playgta6_abi.h` must change, keep that in a small dedicated PR first.
4. **Verify before finishing:**
   - Run `cargo test --workspace`, `python -m unittest discover -s tests -v`, `node --check` on modified JS files, and `ctest` on `runtime/` if CMake is installed.
   - Run `git status` and confirm zero game files, dumps, or `.env` are staged.

## Folder Responsibilities
- `web/` — Browser frontend (`index.html`, `loader.js`), dedicated workers (`web/workers/`), and input bridge (`web/input/`). Keep the main thread non-blocking; require Cross-Origin Isolation (`COOP`/`COEP`) for `SharedArrayBuffer`.
- `recomp/` — **Rust** static `x86-64 -> LLVM IR -> Wasm64` recompilation pipeline.
- `runtime/` — **C++20 + Emscripten (`-sMEMORY64 -pthread`)** runtime: shared ABI (`runtime/abi/`), Xbox/Win32 OS shims (`runtime/os/`), and `d3d12_xs -> WebGPU` command bridge (`runtime/gpu/`).
- `shaders/` — **Rust** offline `RDNA2 -> SPIR-V -> WGSL` (`naga`) translation and shader packing.
- `server/` — Lightweight local HTTP server with `COOP`/`COEP` headers and HTTP `Range` (`206 Partial Content`) support.
- `tools/` — **Rust + Python** local setup (`tools/local_env.py`), manifest/bootset builders, and diagnostics.
- `docs/` — End goal, parallel workstreams, ABI contracts, and WebGPU/Wasm64 findings.
- `tests/` — Automated unit and integration tests.
- `third_party/` — Open-source submodules/dependencies only (never game files).
