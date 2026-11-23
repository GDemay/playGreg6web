# Contributing to `playgta6web`

## 1. Setup Your Machine (1 minute)
Check your installed toolchains and configure your local game folder (stored in gitignored `.env`):

```bash
python tools/local_env.py --check
python tools/local_env.py --set-game "D:\path\to\your\game"   # optional for non-asset tasks
```

Using Claude / Gemini / Codex? Just open the repo and ask it to help you work on a task — `AGENTS.md`, `CLAUDE.md`, and `GEMINI.md` automatically walk it through checking `.env`, asking for your local path if needed, and picking a task from `docs/WORKSTREAMS.md`.

## 2. Pick a Track & Branch
1. Read [docs/END_GOAL.md](./docs/END_GOAL.md) and pick an open task from [docs/WORKSTREAMS.md](./docs/WORKSTREAMS.md).
2. Create a branch named after your track:
   - `recomp/<slug>` (`recomp/`)
   - `os/<slug>` (`runtime/os/`)
   - `gpu/<slug>` (`runtime/gpu/`, `web/workers/wgpu_worker.js`)
   - `shaders/<slug>` (`shaders/`)
   - `web/<slug>` (`web/`, `server/`, `tools/`)
3. Keep PRs scoped to one track. If you need to change a shared ABI (`docs/*.md` or `runtime/abi/playgta6_abi.h`), open a small ABI PR first.

## 3. Run Tests Before Opening a PR
```bash
cargo test --workspace
python -m unittest discover -s tests -v
node --check web/loader.js
```
If you have CMake installed:
```bash
cmake -S runtime -B runtime/build-host
cmake --build runtime/build-host --parallel
ctest --test-dir runtime/build-host --output-on-failure
```

## 4. Non-Negotiable Rules
- **Never commit copyrighted game files, binaries, keys, `.rpf`/`.xvd`/`.ttpf`/`.awc`, extracted assets, or generated game shaders.** CI blocks forbidden extensions, and any PR containing copyrighted material will be rejected.
- **Keep `README.md` small and clean** — put detailed notes in `docs/`.
