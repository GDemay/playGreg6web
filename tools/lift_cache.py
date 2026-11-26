"""Content-keyed cache of `playgta6-recomp build-program` output (shard IR plus metadata).

Shared by `tools/run_native_boot.py` (`--target native`) and `tools/run_browser_boot.py`
(`--target wasm64 --memory-model window-runs --indirect-dispatch shard`). An entry is keyed by
content hashes of the recompiler sources, the exe, the offered DLLs, the shim headers and the
lift options, so the native and the browser lift of the same inputs are separate entries in the
same cache directory (`<object cache>-lift`, or `PLAYGTA6_LIFT_CACHE`; `off` disables). Entries
are hard-linked in and out (copied across volumes); the `keep` most recent are kept. Only hashes
are kept outside the cache entries: never any game bytes.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Sequence

LIFT_CACHE_ENV = "PLAYGTA6_LIFT_CACHE"  # directory, or "off"
LIFT_CACHE_KEEP_ENV = "PLAYGTA6_LIFT_CACHE_KEEP"
LIFT_CACHE_KEEP = 2  # entries are GiBs of IR each
LIFT_CACHE_VERSION = 1
LIFT_STAMP = ".lift-key"
PROGRAM_KEEP = frozenset({"native", LIFT_STAMP})  # the harness build below the program dir is not lift output


def workspace_dirs(repo: Path) -> list[Path]:
    """Crate directories of the cargo workspace (the root Cargo.toml `members`), else `recomp`."""
    manifest = repo / "Cargo.toml"
    members: list[str] = []
    if manifest.is_file():
        text = manifest.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S)
        if block:
            members = re.findall(r'"([^"]+)"', block.group(1))
    return [repo / m for m in members if "*" not in m] or [repo / "recomp"]


def shim_inputs(repo: Path) -> list[Path]:
    """The runtime files `build-program` scans to bind imports (recomp `checkout_shim_headers`):
    `runtime/os/*.h` and `*.cpp` (export tables, `pg6_gthunk_*` registrations),
    `runtime/gpu/d3d12_xs.h`, `runtime/gpu/d3d12_xs_exports.*`, `runtime/guest/guest_callbacks.h`."""
    runtime = repo / "runtime"
    files = sorted((runtime / "os").glob("*.h")) + sorted((runtime / "os").glob("*.cpp"))
    files += [runtime / "gpu" / "d3d12_xs.h", runtime / "guest" / "guest_callbacks.h"]
    files += sorted((runtime / "gpu").glob("d3d12_xs_exports.*"))
    return files


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lift_key(recomp_sources: Mapping[str, str], exe_hash: str, dll_hashes: Mapping[str, str],
             shim_hashes: Mapping[str, str], options: Sequence[str]) -> str:
    """Cache key of a `build-program` output: content hashes of the recompiler sources, the exe,
    the offered DLLs and the shim headers (name -> sha256), plus the lift options (so a native and
    a wasm64 lift of the same inputs never share an entry)."""
    h = hashlib.sha256(f"playgta6-lift v{LIFT_CACHE_VERSION}\n".encode())
    for tag, table in (("src", recomp_sources), ("dll", dll_hashes), ("shim", shim_hashes)):
        for name in sorted(table):
            h.update(f"{tag} {name} {table[name]}\n".encode())
    h.update(f"exe {exe_hash}\n".encode())
    h.update(("opt " + "\0".join(options) + "\n").encode())
    return h.hexdigest()[:32]


def recomp_source_files(repo: Path) -> list[Path]:
    """The files the recompiler binary is built from: every workspace crate's `src/`, manifest and
    build script, and the root Cargo.toml / Cargo.lock."""
    files = [repo / "Cargo.toml", repo / "Cargo.lock"]
    for crate in workspace_dirs(repo):
        files += [crate / "Cargo.toml", crate / "build.rs"]
        for dirpath, dirnames, filenames in os.walk(crate / "src"):
            dirnames.sort()
            files += [Path(dirpath) / n for n in sorted(filenames)]
    return [f for f in files if f.is_file()]


def lift_inputs_key(repo: Path, exe: Path, game_dir: Path | None, options: Sequence[str]) -> str:
    def rel(f: Path) -> str:
        return f.relative_to(repo).as_posix()

    sources = {rel(f): file_digest(f) for f in recomp_source_files(repo)}
    shims = {rel(f): file_digest(f) for f in shim_inputs(repo) + [repo / "runtime" / "abi" / "playgta6_abi.h"]
             if f.is_file()}
    dlls = {f.name.lower(): file_digest(f) for f in sorted(game_dir.iterdir())
            if f.is_file() and f.suffix.lower() == ".dll"} if game_dir is not None and game_dir.is_dir() else {}
    return lift_key(sources, file_digest(exe), dlls, shims, options)


def lift_cache_dir(cache_dir: Path | None, env: Mapping[str, str]) -> Path | None:
    """`PLAYGTA6_LIFT_CACHE` ("off" disables), else `<object cache>-lift` beside the object cache."""
    value = env.get(LIFT_CACHE_ENV)
    if value:
        return None if value.lower() in ("0", "off", "no") else Path(value).expanduser().resolve()
    import build_native_boot  # noqa: PLC0415 (only for the default object cache location)
    obj = (cache_dir if cache_dir is not None else build_native_boot.default_cache_dir(env)).resolve()
    return obj.with_name(obj.name + "-lift")


def lift_cache_keep(env: Mapping[str, str]) -> int:
    return int(env.get(LIFT_CACHE_KEEP_ENV) or LIFT_CACHE_KEEP)


def _link_file(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:  # another volume
        shutil.copy2(src, dst)


def _link_tree(src: Path, dst: Path) -> None:
    """Hard-links (copies across volumes) every file below `src` to `dst`."""
    for dirpath, _, filenames in os.walk(src):
        out = dst / Path(dirpath).relative_to(src)
        out.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            _link_file(Path(dirpath) / name, out / name)


def clear_program(program: Path, keep: Iterable[str] = PROGRAM_KEEP) -> None:
    """Removes the lift output from `program` (keeps the names in `keep`). Its files may be hard
    links into the lift cache, so they are unlinked before a lift, never rewritten in place."""
    if not program.is_dir():
        return
    keep = set(keep) - {LIFT_STAMP}
    for entry in program.iterdir():
        if entry.name in keep:
            continue
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def lift_cache_restore(cache: Path, key: str, program: Path, keep: Iterable[str] = PROGRAM_KEEP) -> None:
    clear_program(program, keep)
    _link_tree(cache / key, program)
    os.utime(cache / key)
    (program / LIFT_STAMP).write_text(key + "\n", encoding="utf-8")


def lift_cache_store(cache: Path, key: str, program: Path, keep: int = LIFT_CACHE_KEEP,
                     skip: Iterable[str] = PROGRAM_KEEP) -> None:
    """Links the fresh lift output (minus `skip`) into `cache/key` and keeps the `keep` most recent entries."""
    skip = set(skip) | {LIFT_STAMP}
    entry = cache / key
    if not entry.is_dir():
        tmp = cache / f"{key}.tmp-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for item in program.iterdir():
            if item.name in skip:
                continue
            if item.is_dir():
                _link_tree(item, tmp / item.name)
            else:
                _link_file(item, tmp / item.name)
        try:
            tmp.rename(entry)
        except OSError:  # another checkout stored the same key meanwhile
            shutil.rmtree(tmp, ignore_errors=True)
    (program / LIFT_STAMP).write_text(key + "\n", encoding="utf-8")
    entries = sorted((e for e in cache.iterdir() if e.is_dir() and ".tmp-" not in e.name),
                     key=lambda e: e.stat().st_mtime, reverse=True)
    for old in entries[max(keep, 1):]:
        if old.name != key:
            shutil.rmtree(old, ignore_errors=True)


def read_stamp(program: Path) -> str | None:
    try:
        return (program / LIFT_STAMP).read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
