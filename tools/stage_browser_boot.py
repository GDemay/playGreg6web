#!/usr/bin/env python3
"""Safe local browser real-boot artifact staging CLI (Track 5 / W51, W66, W119, T46, T70).

Stages allowlisted local recompilation boot artifacts (`program-manifest.json`,
`build-manifest.json`, nested `*.pgim`) from `--program-dir` into `--boot-dir`
(served at `/boot/` by `server/serve_local.py`), validates + atomically stages
`runtime.wasm`, `runtime.js`, optional `guest_manifest.json`, and any sibling
top-level `*.wasm` Emscripten side modules from `--wasm-dir` into `--web-dist`,
validates + atomically stages WGPK v1/v2 `.wgsl.pack` shader bundles plus
adjacent/overridden/merged `<pack>.reflect.json` sidecars and optional
`reflection-manifest.json` into `--shaders-dir`, optionally writes a
schema-validated `boot-options.json` into `--boot-dir`, and optionally verifies
the staged fast-boot bundle (`--verify-staged`).

Safety & privacy invariants:
- Refuses to write boot artifacts or `boot-options.json` into tracked repository
  directories (`--boot-dir` must be under `work/`, `local-data/`, or outside the
  repo).
- Refuses to write Wasm64 runtime artifacts into tracked repository directories
  (`--web-dist` must be `web/dist` or outside the repo).
- Refuses to write shader packs or `.reflect.json` sidecars into tracked
  repository directories (`--shaders-dir` must be under `local-data/`, `work/`,
  or outside the repo).
- Skips any symlinks escaping `--program-dir` / `--wasm-dir`, rejects unsafe
  side-module or shader-pack filenames/traversal, and never copies
  non-allowlisted build artifacts (`.ll`, `.o`, `.cpp`, `.c`, `.h`, `.rsp`, etc.).
- Validates `runtime.wasm` via `tools/check_wasm_exports.py` (`memory64`,
  shared memory, 5/5 `env.host_*` ABI imports, and fast-boot exports
  `_playgta6_audio_ring` / `_playgta6_get_input_state_ptr`), validates WGPK v1/v2
  headers/TOC/bounds, validates `.reflect.json` sidecar synchronization and
  `reflection-manifest.json` consistency, and schema-validates
  `boot-options.json` BEFORE atomically replacing files (`os.replace` via
  same-directory temp file).
- Strictly sanitized JSON and text output: zero Windows/POSIX/UNC host paths,
  zero usernames, and zero raw bytes.
- Pure Python 3.12 standard library only.

Usage:
    python tools/stage_browser_boot.py --program-dir work/recomp_program
        [--wasm-dir work/wasm_guest_dist] [--boot-dir work/recomp_build]
        [--web-dist web/dist] [--shader-pack work/splash.wgsl.pack]
        [--shaders-dir local-data/shaders] [--reflect-sidecar <path>]
        [--reflection-manifest <path>] [--merge-reflect <extra_reflect.json> ...]
        [--require-fast-boot-exports] [--require-sidecar-sync] [--verify-staged]
        [--verify-window-marks] [--dump-pipelines] [--input-script <preset>]
        [--boot-options-json <json-or-file>] [--dry-run] [--json]
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
import re
import struct
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROGRAM_DIR_REL = Path("work") / "recomp_program"
DEFAULT_BOOT_DIR_REL = Path("work") / "recomp_build"
DEFAULT_WEB_DIST_REL = Path("web") / "dist"
DEFAULT_SHADERS_DIR_REL = Path("local-data") / "shaders"

RUNTIME_WASM_NAME = "runtime.wasm"
RUNTIME_JS_NAME = "runtime.js"
GUEST_MANIFEST_NAME = "guest_manifest.json"
GUEST_MANIFEST_MODULE_KEYS: tuple[str, ...] = (
    "guest_dll_modules",
    "guestDllModules",
    "guest_modules",
    "guestModules",
    "modules",
)
PROGRAM_MANIFEST_NAME = "program-manifest.json"
BOOT_MANIFEST_NAME = "boot-manifest.json"
PGIM_MAGIC = b"PGIM"
WASM_MAGIC = b"\x00asm"
WASM_VERSION = 1
WASM_HEADER_BYTES = 8
FAST_BOOT_EXPORT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "playgta6_get_input_state_ptr",
        ("playgta6_get_input_state_ptr", "_playgta6_get_input_state_ptr"),
    ),
    (
        "playgta6_audio_ring",
        ("playgta6_audio_ring", "_playgta6_audio_ring"),
    ),
)
DEFAULT_SHADER_PACK_NAME = "splash.wgsl.pack"
REFLECT_SIDECAR_SUFFIX = ".reflect.json"
REFLECT_SIDECAR_SCHEMA = "playgta6-stage-reflection"
REFLECT_SIDECAR_VERSION = 1
REFLECT_SIDECAR_STAGES: frozenset[str] = frozenset({"vertex", "fragment", "compute"})
REFLECTION_MANIFEST_NAME = "reflection-manifest.json"
BOOT_OPTIONS_NAME = "boot-options.json"
DEFAULT_DUMP_PIPELINES_DIR = "/pg6_dump"
WGPK_MAGIC = b"WGPK"
WGPK_VERSION_V1 = 1
WGPK_VERSION_V2 = 2
WGPK_SUPPORTED_VERSIONS: tuple[int, ...] = (WGPK_VERSION_V1, WGPK_VERSION_V2)
WGPK_HEADER_STRUCT = struct.Struct("<4sIII")
WGPK_HEADER_BYTES = WGPK_HEADER_STRUCT.size  # 16 bytes
WGPK_ENTRY_STRUCT_V1 = struct.Struct("<QB3sII")
WGPK_ENTRY_BYTES_V1 = WGPK_ENTRY_STRUCT_V1.size  # 20 bytes
WGPK_ENTRY_STRUCT_V2 = struct.Struct("<QB3sIQ")
WGPK_ENTRY_BYTES_V2 = WGPK_ENTRY_STRUCT_V2.size  # 24 bytes
WGPK_STAGE_NAMES: dict[int, str] = {1: "vertex", 2: "fragment", 3: "compute"}
WGPK_MAX_U32 = 0xFFFFFFFF
WGPK_MAX_U64 = 0xFFFFFFFFFFFFFFFF
WGPK_MAX_ENTRY_ERRORS = 8
WGPK_MAX_SAMPLE_HASHES = 8
_HASH_HEX_RE = re.compile(r"^(?:0[xX])?([0-9a-fA-F]{1,16})$")

ALLOWED_INPUT_SCRIPT_PRESETS: dict[str, str] = {
    "start-screen": "start-screen",
    "menu-flow": "menu-flow",
    "calibration-skip": "calibration-skip",
    "intro-watch": "intro-watch",
}
_SAFE_INPUT_SCRIPT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_SAFE_SIDE_MODULE_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*\.wasm$", re.IGNORECASE
)
_SAFE_SHADER_PACK_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*\.wgsl\.pack$", re.IGNORECASE
)

BOOT_OPTIONS_UINT_KEYS: frozenset[str] = frozenset(
    {
        "max_wall_ms",
        "max_dispatches",
        "arena_log_limit",
        "import_trace_limit",
    }
)
BOOT_OPTIONS_BOOL_KEYS: frozenset[str] = frozenset(
    {
        "report_to_log",
        "verify_window_marks",
        "present_pacing",
        "fence_publish_lag",
        "defer_gpu_objects",
        "xmem_shared_pages",
        "present_frame_waits",
        "audio_endpoint",
    }
)
BOOT_OPTIONS_STR_KEYS: frozenset[str] = frozenset(
    {
        "dump_pipelines_dir",
        "input_script",
    }
)
BOOT_OPTIONS_ALLOWED_KEYS: frozenset[str] = (
    BOOT_OPTIONS_UINT_KEYS | BOOT_OPTIONS_BOOL_KEYS | BOOT_OPTIONS_STR_KEYS
)
_UINT64_MAX = (1 << 64) - 1

ALLOWLISTED_MANIFEST_NAMES: frozenset[str] = frozenset(
    {
        "program-manifest.json",
        "build-manifest.json",
    }
)
ALLOWLISTED_BOOT_SUFFIXES: tuple[str, ...] = (".pgim",)
COPY_CHUNK_BYTES = 256 * 1024

_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)


class StageBrowserBootError(Exception):
    """Raised for internal staging errors caught and reported cleanly."""


# ---------------------------------------------------------------------------
# Cross-platform sanitization
# ---------------------------------------------------------------------------


def _current_usernames() -> list[str]:
    names: list[str] = []
    candidates: list[str] = [
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
    ]
    try:
        candidates.append(getpass.getuser())
    except (OSError, KeyError, RuntimeError):
        pass
    try:
        candidates.append(Path.home().name)
    except (OSError, RuntimeError):
        pass
    for candidate in candidates:
        cleaned = candidate.strip()
        if len(cleaned) >= 3 and cleaned not in names:
            names.append(cleaned)
    return names


def _home_path_variants() -> list[str]:
    variants: list[str] = []
    try:
        home = Path.home()
        for raw in (str(home), home.as_posix()):
            cleaned = raw.rstrip("/\\")
            if len(cleaned) >= 4 and cleaned not in variants:
                variants.append(cleaned)
    except (OSError, RuntimeError):
        pass
    return variants


def sanitize_text(raw: str | None) -> str:
    """Strip Windows/POSIX/UNC host paths (both `\\` and `/`) and active usernames."""
    if raw is None:
        return ""
    text = str(raw)
    for home_str in _home_path_variants():
        if home_str:
            text = re.sub(re.escape(home_str), "<path>", text, flags=re.IGNORECASE)
    # Strip Windows UNC backslash paths before normalizing separators so
    # `\\server\share\user\file` never survives as `//server/share/...`.
    text = _UNC_BS_PATH_RE.sub("<path>", text)
    text = text.replace("\\", "/")
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = _UNC_FS_PATH_RE.sub("<path>", text)
    text = _POSIX_PATH_RE.sub("<path>", text)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        text = pattern.sub("<user>", text)
    return text


def basename_label(source: str | Path | None) -> str:
    """Basename-only label; splits on both `/` and `\\` regardless of host OS."""
    if source is None:
        return "<unknown>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return sanitize_text(base)


def repo_relative_label(path: Path | None, repo_root: Path) -> str | None:
    """Return a repo-relative POSIX label or a sanitized basename, never a host path."""
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        posix = rel.as_posix()
        return sanitize_text("." if not posix else posix)
    except (OSError, ValueError):
        return basename_label(path)


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings/paths and replace raw bytes with size placeholders."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Path):
        return basename_label(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Sibling tool loader (tools/check_wasm_exports.py)
# ---------------------------------------------------------------------------


def _load_tools_module(module_filename: str, module_name: str) -> Any:
    """Load a sibling tools/*.py module, reusing an already-imported copy when present."""
    short_name = module_filename[:-3] if module_filename.endswith(".py") else module_filename
    for key in (module_name, short_name):
        existing = sys.modules.get(key)
        if existing is not None:
            return existing

    tools_dir = Path(__file__).resolve().parent
    path = tools_dir / module_filename
    if not path.is_file():
        raise FileNotFoundError(f"missing tool module: tools/{module_filename}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load tools/{module_filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


# ---------------------------------------------------------------------------
# Tracked-directory safety guards
# ---------------------------------------------------------------------------


def _resolve_against_repo(raw_path: Path | str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return repo_root / path
    return path


def validate_boot_dir_safety(boot_dir: Path, repo_root: Path) -> tuple[bool, str | None]:
    """Ensure `--boot-dir` is under `work/`, `local-data/`, or outside the repository."""
    resolved_repo = repo_root.resolve()
    resolved_boot = boot_dir.resolve()
    if not resolved_boot.is_relative_to(resolved_repo):
        return True, None

    # Compare against unresolved repo children so a symlink inside `work/` or
    # `local-data/` pointing into a tracked directory cannot bypass the guard.
    allowed_roots = (
        resolved_repo / "work",
        resolved_repo / "local-data",
    )
    if any(resolved_boot.is_relative_to(allowed) for allowed in allowed_roots):
        return True, None

    label = repo_relative_label(resolved_boot, resolved_repo) or "<repo>"
    return (
        False,
        f"refusing to write boot artifacts into tracked repository directory '{label}' "
        "(must be under work/, local-data/, or outside the repository)",
    )


def validate_web_dist_safety(web_dist: Path, repo_root: Path) -> tuple[bool, str | None]:
    """Ensure `--web-dist` is `web/dist` (gitignored) or outside the repository."""
    resolved_repo = repo_root.resolve()
    resolved_dist = web_dist.resolve()
    if not resolved_dist.is_relative_to(resolved_repo):
        return True, None

    allowed_dist = resolved_repo / "web" / "dist"
    if resolved_dist.is_relative_to(allowed_dist):
        return True, None

    label = repo_relative_label(resolved_dist, resolved_repo) or "<repo>"
    return (
        False,
        f"refusing to write Wasm64 runtime into tracked repository directory '{label}' "
        "(must be web/dist or outside the repository)",
    )


def validate_shaders_dir_safety(shaders_dir: Path, repo_root: Path) -> tuple[bool, str | None]:
    """Ensure `--shaders-dir` is under `local-data/`, `work/`, or outside the repository."""
    resolved_repo = repo_root.resolve()
    resolved_shaders = shaders_dir.resolve()
    if not resolved_shaders.is_relative_to(resolved_repo):
        return True, None

    allowed_roots = (
        resolved_repo / "local-data",
        resolved_repo / "work",
    )
    if any(resolved_shaders.is_relative_to(allowed) for allowed in allowed_roots):
        return True, None

    label = repo_relative_label(resolved_shaders, resolved_repo) or "<repo>"
    return (
        False,
        f"refusing to write shader artifacts into tracked repository directory '{label}' "
        "(must be under local-data/, work/, or outside the repository)",
    )


def is_safe_shader_pack_name(filename: str) -> bool:
    """Return True only for safe top-level `*.wgsl.pack` filenames."""
    if not filename:
        return False
    if "/" in filename or "\\" in filename or "\x00" in filename or ".." in filename:
        return False
    return _SAFE_SHADER_PACK_NAME_RE.match(filename) is not None


# ---------------------------------------------------------------------------
# Allowlisted boot artifact discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootArtifact:
    """One allowlisted boot artifact discovered under `--program-dir`."""

    rel_posix: str
    source_path: Path
    kind: str  # "program_manifest" | "build_manifest" | "pgim"


@dataclass
class BootDiscovery:
    """Result of scanning `--program-dir` for allowlisted boot artifacts."""

    artifacts: list[BootArtifact] = field(default_factory=list)
    skipped_escaping_symlinks: int = 0
    skipped_non_allowlisted: int = 0


def is_allowlisted_boot_file(filename: str) -> bool:
    """Return True only for program/build manifests and `.pgim` guest images."""
    lowered = filename.lower()
    if lowered in ALLOWLISTED_MANIFEST_NAMES:
        return True
    return any(lowered.endswith(suffix) for suffix in ALLOWLISTED_BOOT_SUFFIXES)


def _boot_artifact_kind(filename: str) -> str:
    lowered = filename.lower()
    if lowered == "program-manifest.json":
        return "program_manifest"
    if lowered == "build-manifest.json":
        return "build_manifest"
    return "pgim"


def discover_boot_artifacts(program_dir: Path) -> BootDiscovery:
    """Recursively discover allowlisted manifests and `*.pgim` files inside `program_dir`.

    Skips any symlinks escaping `program_dir` (file or directory symlinks) as well
    as non-allowlisted build artifacts such as `.ll`, `.o`, `.cpp`, `.c`, `.h`,
    `.rsp`, or arbitrary `.json` files.
    """
    resolved_root = program_dir.resolve()
    discovery = BootDiscovery()
    seen_rel: set[str] = set()

    for entry in program_dir.rglob("*"):
        if entry.is_symlink():
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                discovery.skipped_escaping_symlinks += 1
                continue
            if not resolved.is_relative_to(resolved_root):
                discovery.skipped_escaping_symlinks += 1
                continue
            if resolved.is_dir():
                continue
            if not resolved.is_file():
                continue
        else:
            if not entry.is_file():
                continue
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_relative_to(resolved_root):
                discovery.skipped_escaping_symlinks += 1
                continue

        if not is_allowlisted_boot_file(entry.name):
            discovery.skipped_non_allowlisted += 1
            continue

        try:
            rel_posix = entry.relative_to(program_dir).as_posix()
        except ValueError:
            continue
        if not rel_posix or ".." in rel_posix.split("/"):
            continue
        if rel_posix in seen_rel:
            continue
        seen_rel.add(rel_posix)
        discovery.artifacts.append(
            BootArtifact(
                rel_posix=rel_posix,
                source_path=resolved,
                kind=_boot_artifact_kind(entry.name),
            )
        )

    discovery.artifacts.sort(key=lambda item: (item.rel_posix.lower(), item.rel_posix))
    return discovery


# ---------------------------------------------------------------------------
# Content comparison & atomic file staging
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Compute hex SHA-256 digest of a file using streaming I/O."""
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


@dataclass(frozen=True)
class FileStagePlan:
    """Incremental copy decision for one source -> destination file pair."""

    needs_copy: bool
    reason: str  # "missing" | "replace_symlink" | "size_changed" | "sha256_changed" | "unchanged"
    size_bytes: int
    sha256: str


def plan_file_stage(src: Path, dst: Path) -> FileStagePlan:
    """Determine whether `src` must be copied to `dst` (missing, size, or SHA-256 changed)."""
    src_stat = src.stat()
    src_size = int(src_stat.st_size)
    src_digest = sha256_file(src)

    if dst.is_symlink():
        return FileStagePlan(
            needs_copy=True,
            reason="replace_symlink",
            size_bytes=src_size,
            sha256=src_digest,
        )

    if not dst.exists():
        return FileStagePlan(
            needs_copy=True,
            reason="missing",
            size_bytes=src_size,
            sha256=src_digest,
        )

    if not dst.is_file():
        raise StageBrowserBootError(
            f"destination path is not a regular file: {basename_label(dst)}"
        )

    try:
        if src.resolve() == dst.resolve():
            return FileStagePlan(
                needs_copy=False,
                reason="unchanged",
                size_bytes=src_size,
                sha256=src_digest,
            )
    except OSError:
        pass

    dst_stat = dst.stat()
    if int(dst_stat.st_size) != src_size:
        return FileStagePlan(
            needs_copy=True,
            reason="size_changed",
            size_bytes=src_size,
            sha256=src_digest,
        )

    dst_digest = sha256_file(dst)
    if dst_digest != src_digest:
        return FileStagePlan(
            needs_copy=True,
            reason="sha256_changed",
            size_bytes=src_size,
            sha256=src_digest,
        )

    return FileStagePlan(
        needs_copy=False,
        reason="unchanged",
        size_bytes=src_size,
        sha256=src_digest,
    )


def _write_temp_copy(src: Path, dst_dir: Path, final_name: str) -> Path:
    """Stream-copy `src` into a new temp file in `dst_dir`, preserving timestamps."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_stat = src.stat()
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{final_name}.",
        suffix=".tmp",
        dir=dst_dir,
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as dst_fh, src.open("rb") as src_fh:
            while True:
                chunk = src_fh.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                dst_fh.write(chunk)
            dst_fh.flush()
            try:
                os.fsync(dst_fh.fileno())
            except OSError:
                pass
        try:
            os.utime(tmp_path, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
        except OSError:
            pass
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy `src` to `dst` via a same-directory temp file and `os.replace`."""
    tmp_path = _write_temp_copy(src, dst.parent, dst.name)
    try:
        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_copy_files(pairs: Sequence[tuple[Path, Path]]) -> None:
    """Prepare all temp files first, then atomically replace each destination."""
    staged: list[tuple[Path, Path]] = []
    replaced: set[Path] = set()
    try:
        for src, dst in pairs:
            tmp_path = _write_temp_copy(src, dst.parent, dst.name)
            staged.append((tmp_path, dst))
        for tmp_path, dst in staged:
            os.replace(tmp_path, dst)
            replaced.add(tmp_path)
    except Exception:
        for tmp_path, _ in staged:
            if tmp_path not in replaced:
                tmp_path.unlink(missing_ok=True)
        raise


def plan_bytes_stage(payload: bytes, dst: Path) -> FileStagePlan:
    """Determine whether raw `payload` bytes must be written to `dst`."""
    payload_size = len(payload)
    payload_digest = hashlib.sha256(payload).hexdigest()

    if dst.is_symlink():
        return FileStagePlan(
            needs_copy=True,
            reason="replace_symlink",
            size_bytes=payload_size,
            sha256=payload_digest,
        )

    if not dst.exists():
        return FileStagePlan(
            needs_copy=True,
            reason="missing",
            size_bytes=payload_size,
            sha256=payload_digest,
        )

    if not dst.is_file():
        raise StageBrowserBootError(
            f"destination path is not a regular file: {basename_label(dst)}"
        )

    dst_stat = dst.stat()
    if int(dst_stat.st_size) != payload_size:
        return FileStagePlan(
            needs_copy=True,
            reason="size_changed",
            size_bytes=payload_size,
            sha256=payload_digest,
        )

    dst_digest = sha256_file(dst)
    if dst_digest != payload_digest:
        return FileStagePlan(
            needs_copy=True,
            reason="sha256_changed",
            size_bytes=payload_size,
            sha256=payload_digest,
        )

    return FileStagePlan(
        needs_copy=False,
        reason="unchanged",
        size_bytes=payload_size,
        sha256=payload_digest,
    )


def _write_temp_bytes(payload: bytes, dst_dir: Path, final_name: str) -> Path:
    """Write `payload` bytes into a new temp file in `dst_dir`."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        prefix=f".{final_name}.",
        suffix=".tmp",
        dir=dst_dir,
    )
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as dst_fh:
            dst_fh.write(payload)
            dst_fh.flush()
            try:
                os.fsync(dst_fh.fileno())
            except OSError:
                pass
        return tmp_path
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_bytes(payload: bytes, dst: Path) -> None:
    """Atomically write `payload` to `dst` via a same-directory temp file and `os.replace`."""
    tmp_path = _write_temp_bytes(payload, dst.parent, dst.name)
    try:
        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_stage_mixed(
    file_pairs: Sequence[tuple[Path, Path]],
    byte_pairs: Sequence[tuple[bytes, Path]],
) -> None:
    """Prepare all file/bytes temp outputs first, then atomically replace each destination."""
    staged: list[tuple[Path, Path]] = []
    replaced: set[Path] = set()
    try:
        for src, dst in file_pairs:
            tmp_path = _write_temp_copy(src, dst.parent, dst.name)
            staged.append((tmp_path, dst))
        for payload, dst in byte_pairs:
            tmp_path = _write_temp_bytes(payload, dst.parent, dst.name)
            staged.append((tmp_path, dst))
        for tmp_path, dst in staged:
            os.replace(tmp_path, dst)
            replaced.add(tmp_path)
    except Exception:
        for tmp_path, _ in staged:
            if tmp_path not in replaced:
                tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Wasm64 validation via tools/check_wasm_exports.py
# ---------------------------------------------------------------------------


def validate_wasm_candidate(
    wasm_path: Path,
    *,
    require_fast_boot_exports: bool = False,
) -> dict[str, Any]:
    """Validate `runtime.wasm` for memory64, shared memory, 5/5 `env.host_*` ABI imports, and fast-boot exports."""
    base_export_fields: dict[str, Any] = {
        "has_main_export": False,
        "main_export_name": None,
        "has_input_state_export": False,
        "input_state_export_name": None,
        "has_audio_ring_export": False,
        "audio_ring_export_name": None,
        "fast_boot_exports_ok": False,
        "all_browser_boot_exports_ok": False,
        "missing_fast_boot_exports": [name for name, _ in FAST_BOOT_EXPORT_SPECS],
        "missing_browser_boot_exports": [],
        "mismatched_browser_boot_exports": [],
    }
    try:
        wasm_mod = _load_tools_module("check_wasm_exports.py", "_playgta6_stage_wasm")
    except Exception as exc:  # noqa: BLE001
        return {
            "checked": True,
            "ok": False,
            "byte_size": 0,
            "is_memory64": False,
            "is_shared_memory": False,
            "all_host_imports_ok": False,
            "host_imports_valid": 0,
            "host_imports_total": 0,
            "missing_host_imports": [],
            "mismatched_host_imports": [],
            "exports_count": 0,
            **base_export_fields,
            "errors": [sanitize_text(f"unable to load check_wasm_exports.py: {exc}")],
            "warnings": [],
        }

    expected_total = len(getattr(wasm_mod, "REQUIRED_HOST_IMPORTS", ()))
    raw: bytes | None = None
    try:
        raw = wasm_path.read_bytes()
        module = wasm_mod.parse_wasm_module(raw)
        report = wasm_mod.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
            require_host_imports=True,
            path=RUNTIME_WASM_NAME,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "checked": True,
            "ok": False,
            "byte_size": len(raw) if raw is not None else 0,
            "is_memory64": False,
            "is_shared_memory": False,
            "all_host_imports_ok": False,
            "host_imports_valid": 0,
            "host_imports_total": expected_total,
            "missing_host_imports": [],
            "mismatched_host_imports": [],
            "exports_count": 0,
            **base_export_fields,
            "errors": [sanitize_text(str(exc))],
            "warnings": [],
        }

    host_valid = sum(1 for item in report.host_imports if item.ok)
    host_total = len(report.host_imports)

    checks_by_name = {
        check.name: check for check in getattr(report, "browser_boot_exports", ())
    }
    main_check = checks_by_name.get("main")
    input_check = checks_by_name.get("playgta6_get_input_state_ptr")
    audio_check = checks_by_name.get("playgta6_audio_ring")

    has_main = bool(main_check and main_check.ok)
    main_name = main_check.matched_name if has_main and main_check is not None else None
    has_input = bool(input_check and input_check.ok)
    input_name = input_check.matched_name if has_input and input_check is not None else None
    has_audio = bool(audio_check and audio_check.ok)
    audio_name = audio_check.matched_name if has_audio and audio_check is not None else None

    missing_fast_boot: list[str] = []
    if not has_input:
        missing_fast_boot.append("playgta6_get_input_state_ptr")
    if not has_audio:
        missing_fast_boot.append("playgta6_audio_ring")
    fast_boot_exports_ok = len(missing_fast_boot) == 0

    val_errors = [sanitize_text(err) for err in report.errors]
    val_warnings = [sanitize_text(warn) for warn in report.warnings]

    expects_linked_exports = has_main or has_input or has_audio
    if not has_input:
        if require_fast_boot_exports:
            detail = (
                sanitize_text(input_check.detail)
                if input_check is not None and input_check.present and not input_check.ok
                else (
                    "runtime.wasm missing required fast-boot function export "
                    "'playgta6_get_input_state_ptr' / '_playgta6_get_input_state_ptr'"
                )
            )
            val_errors.append(detail)
        elif expects_linked_exports:
            val_warnings.append(
                "runtime.wasm missing fast-boot function export "
                "'playgta6_get_input_state_ptr' / '_playgta6_get_input_state_ptr' "
                "(input SAB bridge disabled)"
            )

    if not has_audio:
        if require_fast_boot_exports:
            detail = (
                sanitize_text(audio_check.detail)
                if audio_check is not None and audio_check.present and not audio_check.ok
                else (
                    "runtime.wasm missing required fast-boot function export "
                    "'playgta6_audio_ring' / '_playgta6_audio_ring' (#1028)"
                )
            )
            val_errors.append(detail)
        elif expects_linked_exports:
            val_warnings.append(
                "runtime.wasm missing fast-boot function export "
                "'playgta6_audio_ring' / '_playgta6_audio_ring' "
                "(#1028 guest PCM audio ring disabled)"
            )

    abi_ok = bool(
        report.ok
        and report.is_memory64
        and report.is_shared_memory
        and report.all_host_imports_ok
        and (not require_fast_boot_exports or fast_boot_exports_ok)
        and len(val_errors) == 0
    )
    return {
        "checked": True,
        "ok": abi_ok,
        "byte_size": report.byte_size,
        "wasm_version": report.wasm_version,
        "is_memory64": report.is_memory64,
        "is_shared_memory": report.is_shared_memory,
        "all_host_imports_ok": report.all_host_imports_ok,
        "host_imports_valid": host_valid,
        "host_imports_total": host_total,
        "missing_host_imports": list(report.missing_host_imports),
        "mismatched_host_imports": list(report.mismatched_host_imports),
        "exports_count": report.exports_count,
        "has_main_export": has_main,
        "main_export_name": main_name,
        "has_input_state_export": has_input,
        "input_state_export_name": input_name,
        "has_audio_ring_export": has_audio,
        "audio_ring_export_name": audio_name,
        "fast_boot_exports_ok": fast_boot_exports_ok,
        "all_browser_boot_exports_ok": bool(
            getattr(report, "all_browser_boot_exports_ok", False)
        ),
        "missing_fast_boot_exports": missing_fast_boot,
        "missing_browser_boot_exports": list(
            getattr(report, "missing_browser_boot_exports", ())
        ),
        "mismatched_browser_boot_exports": list(
            getattr(report, "mismatched_browser_boot_exports", ())
        ),
        "errors": val_errors,
        "warnings": val_warnings,
    }


def validate_guest_manifest(
    manifest_path: Path,
) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate a `guest_manifest.json` file (non-empty valid JSON object, sanitized module labels)."""
    label = basename_label(manifest_path)
    try:
        raw_text = manifest_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return (
            False,
            {"name": label, "ok": False, "size_bytes": 0, "guest_modules": []},
            [sanitize_text(f"failed to read {label}: {exc}")],
        )

    size_bytes = len(raw_text.encode("utf-8"))
    if not raw_text.strip():
        return (
            False,
            {"name": label, "ok": False, "size_bytes": size_bytes, "guest_modules": []},
            [f"invalid {label}: file is empty"],
        )

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return (
            False,
            {"name": label, "ok": False, "size_bytes": size_bytes, "guest_modules": []},
            [
                sanitize_text(
                    f"invalid JSON in {label} (line {exc.lineno}, col {exc.colno}): {exc.msg}"
                )
            ],
        )

    if not isinstance(doc, dict):
        return (
            False,
            {"name": label, "ok": False, "size_bytes": size_bytes, "guest_modules": []},
            [f"invalid {label}: top-level JSON value must be an object"],
        )

    errors: list[str] = []
    guest_modules: list[str] = []
    seen_lower: set[str] = set()
    selected_key: str | None = None
    raw_modules: Any = None
    for key in GUEST_MANIFEST_MODULE_KEYS:
        if key in doc and doc[key] is not None:
            selected_key = key
            raw_modules = doc[key]
            break

    if selected_key is not None:
        if not isinstance(raw_modules, list):
            errors.append(f"invalid {label}: '{selected_key}' must be a JSON list")
        else:
            for idx, item in enumerate(raw_modules):
                raw_name: Any = None
                if isinstance(item, str):
                    raw_name = item
                elif isinstance(item, Mapping):
                    for field_key in ("module", "name", "id"):
                        if item.get(field_key) is not None:
                            raw_name = item[field_key]
                            break
                if not isinstance(raw_name, str) or not raw_name.strip():
                    errors.append(
                        f"invalid {label}: '{selected_key}[{idx}]' must be a non-empty module string or object"
                    )
                    continue
                cleaned = (
                    basename_label(raw_name)
                    if ("/" in raw_name or "\\" in raw_name or re.match(r"^[A-Za-z]:", raw_name))
                    else sanitize_text(raw_name.strip())
                )
                if cleaned and cleaned != "<unknown>" and cleaned.lower() not in seen_lower:
                    seen_lower.add(cleaned.lower())
                    guest_modules.append(cleaned)

    side_modules_raw = doc.get("sideModules") or doc.get("side_modules")
    side_module_count = len(side_modules_raw) if isinstance(side_modules_raw, list) else 0

    ok = len(errors) == 0
    return (
        ok,
        {
            "name": label,
            "ok": ok,
            "size_bytes": size_bytes,
            "guest_modules": guest_modules,
            "guest_module_count": len(guest_modules),
            "side_module_count": side_module_count,
        },
        sanitize_value(errors),
    )


# ---------------------------------------------------------------------------
# Sibling Wasm64 side-module discovery & runtime staging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideModuleWasm:
    """One top-level sibling `*.wasm` side module discovered under `--wasm-dir`."""

    name: str
    source_path: Path


@dataclass
class SideModuleDiscovery:
    """Result of scanning `--wasm-dir` for top-level sibling `*.wasm` side modules."""

    modules: list[SideModuleWasm] = field(default_factory=list)
    skipped_escaping_symlinks: int = 0
    rejected_unsafe_names: list[str] = field(default_factory=list)


def _natural_sort_key(name: str) -> tuple[tuple[int, int | str], ...]:
    parts = re.split(r"(\d+)", name.lower())
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts if part)


def is_safe_side_module_name(filename: str) -> bool:
    """Return True only for safe top-level sibling `*.wasm` side-module filenames."""
    if not filename or filename.lower() == RUNTIME_WASM_NAME.lower():
        return False
    if "/" in filename or "\\" in filename or "\x00" in filename or ".." in filename:
        return False
    return _SAFE_SIDE_MODULE_NAME_RE.match(filename) is not None


def discover_side_module_wasms(wasm_dir: Path) -> SideModuleDiscovery:
    """Discover top-level sibling `*.wasm` side modules inside `wasm_dir`.

    Excludes `runtime.wasm`, skips external/broken symlinks, ignores non-`.wasm`
    entries and directories, and records any unsafe `.wasm` filenames or path
    traversal attempts for explicit rejection.
    """
    resolved_root = wasm_dir.resolve()
    discovery = SideModuleDiscovery()
    seen_lower: set[str] = set()

    for entry in wasm_dir.iterdir():
        name = entry.name
        lowered = name.lower()
        if not lowered.endswith(".wasm"):
            continue
        if lowered == RUNTIME_WASM_NAME.lower():
            continue

        if not is_safe_side_module_name(name):
            if entry.is_symlink() or not entry.is_dir():
                discovery.rejected_unsafe_names.append(sanitize_text(name))
            continue

        if entry.is_symlink():
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                discovery.skipped_escaping_symlinks += 1
                continue
            if not resolved.is_relative_to(resolved_root):
                discovery.skipped_escaping_symlinks += 1
                continue
            if not resolved.is_file():
                continue
        else:
            if not entry.is_file():
                continue
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_relative_to(resolved_root):
                discovery.skipped_escaping_symlinks += 1
                continue

        if lowered in seen_lower:
            discovery.rejected_unsafe_names.append(sanitize_text(name))
            continue
        seen_lower.add(lowered)
        discovery.modules.append(SideModuleWasm(name=name, source_path=resolved))

    discovery.modules.sort(key=lambda item: (_natural_sort_key(item.name), item.name))
    discovery.rejected_unsafe_names.sort()
    return discovery


def _empty_wasm_section() -> dict[str, Any]:
    return {
        "status": "skipped",
        "validation": None,
        "guest_manifest": None,
        "copied_count": 0,
        "planned_count": 0,
        "unchanged_count": 0,
        "side_module_count": 0,
        "skipped_escaping_symlinks": 0,
        "files": [],
        "side_modules": [],
    }


def _check_file_within_root(path: Path, root: Path) -> tuple[bool, Path | None]:
    """Verify `path` exists as a regular file whose resolved target stays inside `root`."""
    resolved_root = root.resolve()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False, None
    if not resolved.is_relative_to(resolved_root):
        return False, None
    if not resolved.is_file():
        return False, None
    return True, resolved


def stage_wasm_runtime(
    *,
    wasm_dir: Path | str,
    web_dist: Path | str = DEFAULT_WEB_DIST_REL,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    dry_run: bool = False,
    require_fast_boot_exports: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate `runtime.wasm` and atomically stage runtime + optional `guest_manifest.json` + sibling `*.wasm` side modules."""
    resolved_repo = Path(repo_root).resolve()
    wasm_path = _resolve_against_repo(wasm_dir, resolved_repo)
    web_dist_path = _resolve_against_repo(web_dist, resolved_repo)

    wasm_section = _empty_wasm_section()
    errors: list[str] = []
    warnings: list[str] = []

    web_dist_safe, web_dist_safety_err = validate_web_dist_safety(
        web_dist_path, resolved_repo
    )
    if not web_dist_safe:
        wasm_section["status"] = "refused"
        errors.append(web_dist_safety_err or "refused unsafe --web-dist")
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)

    if not wasm_path.is_dir():
        wasm_section["status"] = "skipped"
        warnings.append("wasm-dir does not exist; skipping Wasm64 staging")
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)

    src_wasm_raw = wasm_path / RUNTIME_WASM_NAME
    src_js_raw = wasm_path / RUNTIME_JS_NAME

    wasm_exists_raw = src_wasm_raw.exists() or src_wasm_raw.is_symlink()
    js_exists_raw = src_js_raw.exists() or src_js_raw.is_symlink()

    if not wasm_exists_raw and not js_exists_raw:
        wasm_section["status"] = "skipped"
        warnings.append(
            "wasm-dir does not contain runtime.wasm + runtime.js; skipping Wasm64 staging"
        )
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)

    wasm_safe_file, resolved_wasm = _check_file_within_root(src_wasm_raw, wasm_path)
    js_safe_file, resolved_js = _check_file_within_root(src_js_raw, wasm_path)

    if wasm_exists_raw and not wasm_safe_file:
        wasm_section["status"] = "refused"
        errors.append(
            "refusing runtime.wasm from wasm-dir (escaping symlink or not a regular file)"
        )
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)
    if js_exists_raw and not js_safe_file:
        wasm_section["status"] = "refused"
        errors.append(
            "refusing runtime.js from wasm-dir (escaping symlink or not a regular file)"
        )
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)
    if not wasm_safe_file:
        wasm_section["status"] = "error"
        errors.append("runtime.wasm missing in wasm-dir alongside runtime.js")
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)
    if not js_safe_file:
        wasm_section["status"] = "error"
        errors.append("runtime.js missing in wasm-dir alongside runtime.wasm")
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)

    assert resolved_wasm is not None and resolved_js is not None
    validation = validate_wasm_candidate(
        resolved_wasm,
        require_fast_boot_exports=require_fast_boot_exports,
    )
    wasm_section["validation"] = validation
    warnings.extend(validation.get("warnings") or [])

    if not validation.get("ok"):
        wasm_section["status"] = "refused"
        val_errors = validation.get("errors") or [
            "runtime.wasm failed Wasm64/shared/host-ABI validation"
        ]
        errors.append(
            "refusing to stage runtime.wasm into web-dist (validation failed): "
            + "; ".join(val_errors)
        )
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)

    # Validate optional sibling `guest_manifest.json` if present inside `wasm_dir`.
    src_gm_raw = wasm_path / GUEST_MANIFEST_NAME
    resolved_gm: Path | None = None
    if src_gm_raw.exists() or src_gm_raw.is_symlink():
        gm_safe, gm_resolved = _check_file_within_root(src_gm_raw, wasm_path)
        if not gm_safe or gm_resolved is None:
            wasm_section["status"] = "refused"
            errors.append(
                f"refusing {GUEST_MANIFEST_NAME} from wasm-dir (escaping symlink or not a regular file)"
            )
            return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)
        gm_ok, gm_info, gm_errors = validate_guest_manifest(gm_resolved)
        wasm_section["guest_manifest"] = gm_info
        if not gm_ok:
            wasm_section["status"] = "refused"
            errors.extend(gm_errors)
            return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)
        resolved_gm = gm_resolved

    # Discover top-level sibling `*.wasm` side modules only after `runtime.wasm`
    # passes ABI validation.
    side_discovery = discover_side_module_wasms(wasm_path)
    wasm_section["skipped_escaping_symlinks"] = side_discovery.skipped_escaping_symlinks
    if side_discovery.skipped_escaping_symlinks > 0:
        warnings.append(
            f"skipped {side_discovery.skipped_escaping_symlinks} symlink(s) escaping wasm-dir"
        )
    if side_discovery.rejected_unsafe_names:
        wasm_section["status"] = "refused"
        errors.append(
            "refusing unsafe side-module filename(s) in wasm-dir: "
            + ", ".join(side_discovery.rejected_unsafe_names)
        )
        return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)

    candidates: list[tuple[str, Path, Path, bool]] = [
        (RUNTIME_WASM_NAME, resolved_wasm, web_dist_path / RUNTIME_WASM_NAME, False),
        (RUNTIME_JS_NAME, resolved_js, web_dist_path / RUNTIME_JS_NAME, False),
    ]
    if resolved_gm is not None:
        candidates.append(
            (GUEST_MANIFEST_NAME, resolved_gm, web_dist_path / GUEST_MANIFEST_NAME, False)
        )
    for side_mod in side_discovery.modules:
        candidates.append(
            (side_mod.name, side_mod.source_path, web_dist_path / side_mod.name, True)
        )

    resolved_web_dist = web_dist_path.resolve()
    for name, _, dst_file, _ in candidates:
        if dst_file.parent.resolve() != resolved_web_dist or dst_file.name != name:
            wasm_section["status"] = "refused"
            errors.append(
                f"refused Wasm64 artifact escaping web-dist: {sanitize_text(name)}"
            )
            return (
                sanitize_value(wasm_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

    plans: list[tuple[str, Path, Path, bool, FileStagePlan]] = []
    for name, src_file, dst_file, is_side_module in candidates:
        try:
            plan = plan_file_stage(src_file, dst_file)
        except (OSError, StageBrowserBootError) as exc:
            wasm_section["status"] = "error"
            errors.append(f"failed to inspect {sanitize_text(name)}: {sanitize_text(str(exc))}")
            return (
                sanitize_value(wasm_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )
        plans.append((name, src_file, dst_file, is_side_module, plan))

    to_copy = [
        (src_file, dst_file)
        for _, src_file, dst_file, _, plan in plans
        if plan.needs_copy
    ]
    if to_copy and not dry_run:
        try:
            atomic_copy_files(to_copy)
        except OSError as exc:
            wasm_section["status"] = "error"
            errors.append(
                f"failed to atomically stage Wasm64 runtime: {sanitize_text(str(exc))}"
            )
            return (
                sanitize_value(wasm_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

    copied = 0
    planned = 0
    unchanged = 0
    file_entries: list[dict[str, Any]] = []
    side_module_entries: list[dict[str, Any]] = []
    for name, _, _, is_side_module, plan in plans:
        if not plan.needs_copy:
            unchanged += 1
            action = "unchanged"
        elif dry_run:
            planned += 1
            action = "plan_copy"
        else:
            copied += 1
            action = "copied"
        entry = {
            "name": sanitize_text(name),
            "action": action,
            "reason": plan.reason,
            "size_bytes": plan.size_bytes,
            "sha256": plan.sha256,
        }
        file_entries.append(entry)
        if is_side_module:
            side_module_entries.append(dict(entry))

    wasm_section["copied_count"] = copied
    wasm_section["planned_count"] = planned
    wasm_section["unchanged_count"] = unchanged
    wasm_section["side_module_count"] = len(side_module_entries)
    wasm_section["files"] = file_entries
    wasm_section["side_modules"] = side_module_entries
    if dry_run:
        wasm_section["status"] = "planned" if planned > 0 else "unchanged"
    elif copied > 0:
        wasm_section["status"] = "staged"
    else:
        wasm_section["status"] = "unchanged"

    return sanitize_value(wasm_section), sanitize_value(errors), sanitize_value(warnings)


# ---------------------------------------------------------------------------
# Shader pack (.wgsl.pack), WGPK v1/v2 validation & .reflect.json sidecar staging (W119, T70)
# ---------------------------------------------------------------------------


def _empty_shaders_section() -> dict[str, Any]:
    return {
        "status": "skipped",
        "pack": None,
        "wgpk_validation": None,
        "sidecar": None,
        "sidecar_sync": None,
        "reflection_manifest": None,
        "sidecar_staged": False,
        "sidecar_bytes": 0,
        "reflection_manifest_staged": False,
        "copied_count": 0,
        "planned_count": 0,
        "unchanged_count": 0,
    }


def _normalize_hash_hex(raw: Any) -> str | None:
    """Normalize a 64-bit hex shader hash string or integer to canonical `0x` + 16 lowercase hex digits."""
    if isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        if 0 <= raw <= WGPK_MAX_U64:
            return f"0x{raw:016x}"
        return None
    if not isinstance(raw, str):
        return None
    match = _HASH_HEX_RE.match(raw.strip())
    if not match:
        return None
    return f"0x{int(match.group(1), 16):016x}"


def _wgpk_entry_stride(version: int) -> int:
    return WGPK_ENTRY_BYTES_V2 if version == WGPK_VERSION_V2 else WGPK_ENTRY_BYTES_V1


def _unpack_wgpk_entry(version: int, buf: bytes | bytearray | memoryview, pos: int) -> tuple[int, int, bytes, int, int]:
    """Unpack one WGPK TOC entry as `(hash, stage_code, reserved, offset, length)` for v1 or v2."""
    if version == WGPK_VERSION_V2:
        shader_hash, stage_raw, reserved, length, offset = WGPK_ENTRY_STRUCT_V2.unpack_from(buf, pos)
        return int(shader_hash), int(stage_raw), reserved, int(offset), int(length)
    shader_hash, stage_raw, reserved, offset, length = WGPK_ENTRY_STRUCT_V1.unpack_from(buf, pos)
    return int(shader_hash), int(stage_raw), reserved, int(offset), int(length)


def validate_wgpk_pack(
    pack_path: Path,
    *,
    label: str | None = None,
) -> tuple[bool, dict[str, Any], dict[str, str], list[str], list[str]]:
    """Validate a binary `.wgsl.pack` (`WGPK` v1 or v2) header, entry table, stage codes, and payload bounds.

    Streams only the 16-byte header and TOC records (20 bytes/entry for v1, 24 bytes/entry
    with `u64` offset for v2) without loading WGSL payload bytes into memory.
    Returns `(ok, summary, entry_stages_by_hash, errors, warnings)`.
    """
    safe_label = sanitize_text(label) if label is not None else basename_label(pack_path)
    errors: list[str] = []
    warnings: list[str] = []
    entry_stages: dict[str, str] = {}
    stage_counts: dict[str, int] = {"vertex": 0, "fragment": 0, "compute": 0}

    size_bytes = 0
    magic_ok = False
    header_ok = False
    table_ok = False
    bounds_ok = False
    version: int | None = None
    entry_stride: int | None = None
    entry_count: int | None = None
    flags: int | None = None
    data_start_offset: int | None = None
    max_payload_end: int | None = None
    out_of_bounds_entries = 0
    invalid_stage_entries = 0
    overlapping_payloads = 0

    try:
        size_bytes = int(pack_path.stat().st_size)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        errors.append(sanitize_text(f"failed to stat WGPK pack {safe_label}: {reason}"))
        return (
            False,
            {
                "name": safe_label,
                "ok": False,
                "size_bytes": 0,
                "magic_ok": False,
                "header_ok": False,
                "table_ok": False,
                "bounds_ok": False,
                "version": None,
                "entry_stride": None,
                "entry_count": None,
                "flags": None,
                "data_start_offset": None,
                "max_payload_end": None,
                "stage_counts": stage_counts,
                "out_of_bounds_entries": 0,
                "invalid_stage_entries": 0,
                "overlapping_payloads": 0,
                "errors": sanitize_value(errors),
            },
            entry_stages,
            sanitize_value(errors),
            sanitize_value(warnings),
        )

    if size_bytes == 0:
        errors.append(f"refusing empty shader-pack: {safe_label}")
    elif size_bytes < WGPK_HEADER_BYTES:
        errors.append(
            f"WGPK pack {safe_label} truncated: {size_bytes} bytes "
            f"(minimum WGPK header is {WGPK_HEADER_BYTES} bytes)"
        )
    else:
        try:
            with pack_path.open("rb") as fh:
                header = fh.read(WGPK_HEADER_BYTES)
                if len(header) < WGPK_HEADER_BYTES:
                    errors.append(f"WGPK pack {safe_label} truncated while reading header")
                else:
                    magic, ver, count, hdr_flags = WGPK_HEADER_STRUCT.unpack_from(header, 0)
                    if magic != WGPK_MAGIC:
                        errors.append(
                            f"WGPK pack {safe_label} has invalid magic (expected 'WGPK')"
                        )
                    else:
                        magic_ok = True
                        version = int(ver)
                        entry_count = int(count)
                        flags = int(hdr_flags)

                        if version not in WGPK_SUPPORTED_VERSIONS:
                            errors.append(
                                f"WGPK pack {safe_label} has unsupported WGPK version {version} "
                                f"(expected 1 or 2)"
                            )
                        else:
                            header_ok = True
                            entry_stride = _wgpk_entry_stride(version)
                            table_bytes = entry_count * entry_stride
                            data_start = WGPK_HEADER_BYTES + table_bytes
                            data_start_offset = data_start
                            max_offset_limit = (
                                WGPK_MAX_U64 if version == WGPK_VERSION_V2 else WGPK_MAX_U32
                            )

                            if data_start > size_bytes:
                                errors.append(
                                    f"WGPK pack {safe_label} truncated entry table: "
                                    f"version={version}, entry_count={entry_count} requires "
                                    f"{data_start} bytes, file is {size_bytes} bytes"
                                )
                            else:
                                table_raw = fh.read(table_bytes)
                                if len(table_raw) < table_bytes:
                                    errors.append(
                                        f"WGPK pack {safe_label} truncated while reading entry table"
                                    )
                                else:
                                    table_ok = True
                                    highest_end = data_start
                                    entry_errors_reported = 0
                                    suppressed_entry_errors = 0
                                    non_empty_ranges: list[tuple[int, int, int]] = []

                                    for idx in range(entry_count):
                                        base = idx * entry_stride
                                        shader_hash, stage_raw, _res, offset, length = (
                                            _unpack_wgpk_entry(version, table_raw, base)
                                        )
                                        stage_name = WGPK_STAGE_NAMES.get(stage_raw)
                                        if stage_name is None:
                                            invalid_stage_entries += 1
                                            if entry_errors_reported < WGPK_MAX_ENTRY_ERRORS:
                                                errors.append(
                                                    f"WGPK pack {safe_label} entry {idx} has "
                                                    f"invalid shader stage {stage_raw}"
                                                )
                                                entry_errors_reported += 1
                                            else:
                                                suppressed_entry_errors += 1
                                        else:
                                            stage_counts[stage_name] += 1
                                            entry_stages[f"0x{shader_hash:016x}"] = stage_name

                                        payload_end = offset + length
                                        if (
                                            offset < data_start
                                            or payload_end > max_offset_limit
                                            or payload_end > size_bytes
                                        ):
                                            out_of_bounds_entries += 1
                                            if entry_errors_reported < WGPK_MAX_ENTRY_ERRORS:
                                                if offset < data_start:
                                                    reason = (
                                                        f"overlaps header/entry table "
                                                        f"(offset {offset} < data_start {data_start})"
                                                    )
                                                elif payload_end > max_offset_limit:
                                                    width = "uint64" if version == WGPK_VERSION_V2 else "uint32"
                                                    reason = (
                                                        f"range overflows {width} "
                                                        f"[{offset}..+{length}]"
                                                    )
                                                else:
                                                    reason = (
                                                        f"exceeds pack bounds "
                                                        f"[{data_start}..{size_bytes}]"
                                                    )
                                                errors.append(
                                                    f"WGPK pack {safe_label} entry {idx} payload "
                                                    f"[{offset}..+{length}] {reason}"
                                                )
                                                entry_errors_reported += 1
                                            else:
                                                suppressed_entry_errors += 1
                                        else:
                                            if payload_end > highest_end:
                                                highest_end = payload_end
                                            if length > 0:
                                                non_empty_ranges.append((offset, payload_end, idx))

                                    if len(non_empty_ranges) > 1:
                                        non_empty_ranges.sort(key=lambda item: (item[0], item[1]))
                                        for prev, curr in zip(non_empty_ranges, non_empty_ranges[1:]):
                                            if curr[0] < prev[1]:
                                                overlapping_payloads += 1
                                                if entry_errors_reported < WGPK_MAX_ENTRY_ERRORS:
                                                    errors.append(
                                                        f"WGPK pack {safe_label} overlapping payload ranges "
                                                        f"between entry {prev[2]} [{prev[0]}..{prev[1]}) "
                                                        f"and entry {curr[2]} [{curr[0]}..{curr[1]})"
                                                    )
                                                    entry_errors_reported += 1
                                                else:
                                                    suppressed_entry_errors += 1

                                    if suppressed_entry_errors > 0:
                                        errors.append(
                                            f"WGPK pack {safe_label} has {suppressed_entry_errors} "
                                            f"additional invalid/out-of-bounds entry error(s)"
                                        )
                                    max_payload_end = highest_end
                                    bounds_ok = (
                                        out_of_bounds_entries == 0 and overlapping_payloads == 0
                                    )
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            errors.append(sanitize_text(f"failed to read WGPK pack {safe_label}: {reason}"))

    ok = len(errors) == 0
    summary = {
        "name": safe_label,
        "ok": ok,
        "size_bytes": size_bytes,
        "magic_ok": magic_ok,
        "header_ok": header_ok,
        "table_ok": table_ok,
        "bounds_ok": bounds_ok,
        "version": version,
        "entry_stride": entry_stride,
        "entry_count": entry_count,
        "flags": flags,
        "data_start_offset": data_start_offset,
        "max_payload_end": max_payload_end,
        "stage_counts": stage_counts,
        "out_of_bounds_entries": out_of_bounds_entries,
        "invalid_stage_entries": invalid_stage_entries,
        "overlapping_payloads": overlapping_payloads,
        "errors": sanitize_value(errors),
    }
    return ok, sanitize_value(summary), entry_stages, sanitize_value(errors), sanitize_value(warnings)


def check_wgpk_sidecar_sync(
    pack_entries: Mapping[str, str],
    sidecar_stages: Mapping[str, Any],
    *,
    pack_name: str,
    sidecar_name: str,
    require_sync: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Verify synchronization between WGPK TOC entries and a `<pack>.reflect.json` stage map."""
    safe_pack = sanitize_text(pack_name)
    safe_sidecar = sanitize_text(sidecar_name)
    errors: list[str] = []
    warnings: list[str] = []

    norm_pack: dict[str, str] = {}
    for raw_hash, stage_name in pack_entries.items():
        norm = _normalize_hash_hex(raw_hash)
        if norm is not None:
            norm_pack[norm] = str(stage_name).strip().lower()

    norm_sidecar: dict[str, str | None] = {}
    for raw_key, stage_obj in sidecar_stages.items():
        norm = _normalize_hash_hex(raw_key)
        key_id = norm if norm is not None else sanitize_text(str(raw_key))
        stage_val = stage_obj.get("stage") if isinstance(stage_obj, Mapping) else None
        norm_sidecar[key_id] = (
            stage_val.strip().lower() if isinstance(stage_val, str) and stage_val.strip() else None
        )

    matched = 0
    missing_in_sidecar: list[str] = []
    stage_mismatches: list[str] = []
    for hash_hex in sorted(norm_pack):
        pack_stage = norm_pack[hash_hex]
        if hash_hex not in norm_sidecar:
            missing_in_sidecar.append(hash_hex)
        else:
            matched += 1
            sc_stage = norm_sidecar[hash_hex]
            if sc_stage is not None and sc_stage != pack_stage:
                stage_mismatches.append(
                    f"{hash_hex} (pack={pack_stage}, sidecar={sc_stage})"
                )

    extra_in_sidecar = sorted(k for k in norm_sidecar if k not in norm_pack)
    pack_count = len(norm_pack)
    sidecar_count = len(sidecar_stages)

    if stage_mismatches:
        errors.append(
            f"sidecar {safe_sidecar} stage mismatch with {safe_pack}: "
            f"{len(stage_mismatches)} shader(s) disagree ({', '.join(stage_mismatches[:4])})"
        )

    if pack_count > 0:
        if sidecar_count == 0:
            errors.append(
                f"sidecar {safe_sidecar} has 0 stages but {safe_pack} contains {pack_count} shader(s)"
            )
        elif missing_in_sidecar:
            msg = (
                f"sidecar {safe_sidecar} missing {len(missing_in_sidecar)}/{pack_count} "
                f"shader hash(es) from {safe_pack}"
            )
            if require_sync:
                errors.append(msg)
            else:
                warnings.append(msg)

    synchronized = len(stage_mismatches) == 0 and len(missing_in_sidecar) == 0
    summary = {
        "pack_name": safe_pack,
        "sidecar_name": safe_sidecar,
        "pack_entry_count": pack_count,
        "sidecar_stage_count": sidecar_count,
        "matched_hash_count": matched,
        "missing_in_sidecar_count": len(missing_in_sidecar),
        "missing_in_sidecar_samples": missing_in_sidecar[:WGPK_MAX_SAMPLE_HASHES],
        "extra_in_sidecar_count": len(extra_in_sidecar),
        "extra_in_sidecar_samples": extra_in_sidecar[:WGPK_MAX_SAMPLE_HASHES],
        "stage_mismatch_count": len(stage_mismatches),
        "stage_mismatch_samples": stage_mismatches[:WGPK_MAX_SAMPLE_HASHES],
        "synchronized": synchronized,
        "ok": len(errors) == 0,
    }
    return sanitize_value(summary), sanitize_value(errors), sanitize_value(warnings)


def validate_reflection_manifest(
    manifest_path: Path,
    *,
    expected_pack_names: Sequence[str] = (),
    pack_entries: Mapping[str, str] | None = None,
) -> tuple[bool, dict[str, Any], list[str], list[str]]:
    """Validate `reflection-manifest.json` structure, pack filename consistency, and shader parity."""
    label = basename_label(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []
    stage_counts: dict[str, int] = {"vertex": 0, "fragment": 0, "compute": 0}

    try:
        raw_text = manifest_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(sanitize_text(f"failed to read {label}: {exc}"))
        return (
            False,
            {"name": label, "ok": False, "size_bytes": 0, "shader_count": 0},
            sanitize_value(errors),
            warnings,
        )

    size_bytes = len(raw_text.encode("utf-8"))
    if not raw_text.strip():
        errors.append(f"invalid {label}: file is empty")
        return (
            False,
            {"name": label, "ok": False, "size_bytes": size_bytes, "shader_count": 0},
            sanitize_value(errors),
            warnings,
        )

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        errors.append(
            sanitize_text(
                f"invalid JSON in {label} (line {exc.lineno}, col {exc.colno}): {exc.msg}"
            )
        )
        return (
            False,
            {"name": label, "ok": False, "size_bytes": size_bytes, "shader_count": 0},
            sanitize_value(errors),
            warnings,
        )

    if not isinstance(doc, dict):
        errors.append(f"invalid {label}: top-level JSON value must be an object")
        return (
            False,
            {"name": label, "ok": False, "size_bytes": size_bytes, "shader_count": 0},
            sanitize_value(errors),
            warnings,
        )

    schema_version = doc.get("schema_version")
    if schema_version is not None and (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 1
    ):
        errors.append(f"invalid {label}: 'schema_version' must be a positive integer")

    declared_pack: str | None = None
    pack_name_matches: bool | None = None
    for pack_key in ("pack", "pack_name", "packName", "shader_pack"):
        if pack_key in doc and doc[pack_key] is not None:
            raw_pack = doc[pack_key]
            if not isinstance(raw_pack, str) or not raw_pack.strip():
                errors.append(f"invalid {label}: '{pack_key}' must be a non-empty string")
            else:
                declared_pack = basename_label(raw_pack)
                if expected_pack_names:
                    expected_lower = {basename_label(n).lower() for n in expected_pack_names if n}
                    pack_name_matches = declared_pack.lower() in expected_lower
                    if not pack_name_matches:
                        expected_list = ", ".join(
                            sorted({basename_label(n) for n in expected_pack_names if n})
                        )
                        errors.append(
                            f"{label} pack filename '{declared_pack}' does not match "
                            f"staged shader pack ({expected_list})"
                        )
            break

    shaders_map = doc.get("shaders")
    norm_manifest_shaders: dict[str, str | None] = {}
    if not isinstance(shaders_map, dict):
        errors.append(f"invalid {label}: missing or non-object 'shaders' map")
    else:
        for raw_key, entry in shaders_map.items():
            if not isinstance(raw_key, str) or not raw_key.strip() or not isinstance(entry, dict):
                errors.append(
                    f"invalid {label}: every entry in 'shaders' must be a JSON object with a non-empty key"
                )
                break
            stage_val = entry.get("stage")
            stage_norm: str | None = None
            if stage_val is not None:
                if not isinstance(stage_val, str) or stage_val.strip().lower() not in REFLECT_SIDECAR_STAGES:
                    errors.append(
                        f"invalid {label}: shader {sanitize_text(raw_key)!r} has invalid 'stage' value"
                    )
                    break
                stage_norm = stage_val.strip().lower()
                stage_counts[stage_norm] += 1
            norm_key = _normalize_hash_hex(raw_key) or sanitize_text(raw_key.strip())
            norm_manifest_shaders[norm_key] = stage_norm

        if len(shaders_map) == 0 and (pack_entries is None or len(pack_entries) > 0):
            errors.append(f"invalid {label}: 'shaders' map is empty")

        for count_key in ("entry_count", "shader_count"):
            if count_key in doc and doc[count_key] is not None:
                declared_count = doc[count_key]
                if isinstance(declared_count, bool) or not isinstance(declared_count, int) or declared_count < 0:
                    errors.append(f"invalid {label}: '{count_key}' must be a non-negative integer")
                elif declared_count != len(shaders_map):
                    errors.append(
                        f"{label} '{count_key}' ({declared_count}) does not match "
                        f"'shaders' map count ({len(shaders_map)})"
                    )

    matched_hashes = 0
    missing_in_manifest: list[str] = []
    stage_mismatches: list[str] = []
    hash_parity_ok: bool | None = None
    if pack_entries is not None and isinstance(shaders_map, dict):
        norm_pack = {
            norm: str(st).strip().lower()
            for k, st in pack_entries.items()
            if (norm := _normalize_hash_hex(k)) is not None
        }
        for hash_hex in sorted(norm_pack):
            pack_stage = norm_pack[hash_hex]
            if hash_hex not in norm_manifest_shaders:
                missing_in_manifest.append(hash_hex)
            else:
                matched_hashes += 1
                manifest_stage = norm_manifest_shaders[hash_hex]
                if manifest_stage is not None and manifest_stage != pack_stage:
                    stage_mismatches.append(
                        f"{hash_hex} (pack={pack_stage}, manifest={manifest_stage})"
                    )
        hash_parity_ok = len(missing_in_manifest) == 0 and len(stage_mismatches) == 0
        if stage_mismatches:
            errors.append(
                f"{label} stage mismatch with WGPK pack: {len(stage_mismatches)} shader(s) disagree "
                f"({', '.join(stage_mismatches[:4])})"
            )
        if missing_in_manifest:
            errors.append(
                f"{label} missing {len(missing_in_manifest)}/{len(norm_pack)} shader hash(es) "
                f"from WGPK pack"
            )

    ok = len(errors) == 0
    summary = {
        "name": label,
        "ok": ok,
        "size_bytes": size_bytes,
        "schema_version": int(schema_version) if isinstance(schema_version, int) and not isinstance(schema_version, bool) else None,
        "pack_name": declared_pack,
        "pack_name_matches": pack_name_matches,
        "shader_count": len(shaders_map) if isinstance(shaders_map, dict) else 0,
        "stage_counts": stage_counts,
        "matched_hash_count": matched_hashes,
        "missing_in_manifest_count": len(missing_in_manifest),
        "missing_in_manifest_samples": missing_in_manifest[:WGPK_MAX_SAMPLE_HASHES],
        "stage_mismatch_count": len(stage_mismatches),
        "stage_mismatch_samples": stage_mismatches[:WGPK_MAX_SAMPLE_HASHES],
        "hash_parity_ok": hash_parity_ok,
        "errors": sanitize_value(errors),
    }
    return ok, sanitize_value(summary), sanitize_value(errors), sanitize_value(warnings)


def _load_reflect_sidecar_with_stages(
    sidecar_path: Path,
) -> tuple[bool, dict[str, Any], dict[str, Any], list[str]]:
    """Validate a `<pack>.reflect.json` sidecar and return `(ok, summary, stages_map, errors)`."""
    label = basename_label(sidecar_path)
    try:
        raw_text = sidecar_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return (
            False,
            {},
            {},
            [sanitize_text(f"failed to read reflection sidecar {label}: {exc}")],
        )

    if not raw_text.strip():
        return (
            False,
            {},
            {},
            [f"invalid reflection sidecar {label}: file is empty"],
        )

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return (
            False,
            {},
            {},
            [
                sanitize_text(
                    f"invalid JSON in reflection sidecar {label} "
                    f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
                )
            ],
        )

    if not isinstance(parsed, dict):
        return (
            False,
            {},
            {},
            [f"invalid reflection sidecar {label}: top-level JSON value must be an object"],
        )

    errors: list[str] = []
    schema_val = parsed.get("schema")
    if schema_val is not None and (not isinstance(schema_val, str) or not schema_val.strip()):
        errors.append(
            f"invalid reflection sidecar {label}: 'schema' must be a non-empty string"
        )

    version_val = parsed.get("version")
    if version_val is not None and (
        isinstance(version_val, bool) or not isinstance(version_val, int) or version_val < 1
    ):
        errors.append(
            f"invalid reflection sidecar {label}: 'version' must be a positive integer"
        )

    stages_val = parsed.get("stages")
    stage_counts: dict[str, int] = {"vertex": 0, "fragment": 0, "compute": 0}
    if not isinstance(stages_val, dict):
        errors.append(
            f"invalid reflection sidecar {label}: missing or non-object 'stages' map"
        )
    else:
        for stage_key, stage_entry in stages_val.items():
            if not isinstance(stage_key, str) or not stage_key.strip() or not isinstance(stage_entry, dict):
                errors.append(
                    f"invalid reflection sidecar {label}: every entry in 'stages' must be a JSON object"
                )
                break
            stage_kind = stage_entry.get("stage")
            if stage_kind is not None:
                if not isinstance(stage_kind, str) or stage_kind.strip().lower() not in REFLECT_SIDECAR_STAGES:
                    errors.append(
                        f"invalid reflection sidecar {label}: stage {sanitize_text(stage_key)!r} has invalid 'stage' value"
                    )
                    break
                stage_counts[stage_kind.strip().lower()] += 1
            bindings_val = stage_entry.get("bindings")
            if bindings_val is not None and not isinstance(bindings_val, list):
                errors.append(
                    f"invalid reflection sidecar {label}: stage {sanitize_text(stage_key)!r} 'bindings' must be a list"
                )
                break

    if errors:
        return False, {}, {}, sanitize_value(errors)

    summary = {
        "schema": sanitize_text(schema_val) if isinstance(schema_val, str) else None,
        "version": int(version_val) if isinstance(version_val, int) else None,
        "stage_count": len(stages_val),
        "stage_counts": stage_counts,
    }
    return True, summary, stages_val, []


def validate_reflect_sidecar(sidecar_path: Path) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate a `<pack>.reflect.json` stage-reflection sidecar file."""
    ok, summary, _stages_map, errors = _load_reflect_sidecar_with_stages(sidecar_path)
    return ok, summary, errors


def load_and_validate_reflect_sidecar_v1(
    sidecar_path: Path,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    """Load and strictly validate a `playgta6-stage-reflection` v1 sidecar JSON file."""
    label = basename_label(sidecar_path)
    try:
        raw_text = sidecar_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return (
            False,
            None,
            [sanitize_text(f"failed to read reflection sidecar {label}: {exc}")],
        )

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return (
            False,
            None,
            [
                sanitize_text(
                    f"invalid JSON in reflection sidecar {label} "
                    f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
                )
            ],
        )

    if not isinstance(parsed, dict):
        return (
            False,
            None,
            [f"invalid reflection sidecar {label}: top-level JSON value must be an object"],
        )

    errors: list[str] = []
    schema_val = parsed.get("schema")
    if schema_val != REFLECT_SIDECAR_SCHEMA:
        errors.append(
            f"invalid reflection sidecar {label}: 'schema' must be {REFLECT_SIDECAR_SCHEMA!r}"
        )

    version_val = parsed.get("version")
    if type(version_val) is not int or version_val != REFLECT_SIDECAR_VERSION:
        errors.append(
            f"invalid reflection sidecar {label}: 'version' must be {REFLECT_SIDECAR_VERSION}"
        )

    stages_val = parsed.get("stages")
    validated_stages: dict[str, Any] = {}
    if not isinstance(stages_val, dict):
        errors.append(
            f"invalid reflection sidecar {label}: missing or non-object 'stages' map"
        )
    else:
        for stage_key, stage_entry in stages_val.items():
            if not isinstance(stage_key, str) or not stage_key.strip():
                errors.append(
                    f"invalid reflection sidecar {label}: every stage key must be a non-empty string"
                )
                break
            if not isinstance(stage_entry, dict):
                errors.append(
                    f"invalid reflection sidecar {label}: stage {sanitize_text(stage_key)!r} must be a JSON object"
                )
                break
            stage_kind = stage_entry.get("stage")
            if not isinstance(stage_kind, str) or stage_kind not in REFLECT_SIDECAR_STAGES:
                errors.append(
                    f"invalid reflection sidecar {label}: stage {sanitize_text(stage_key)!r} has invalid 'stage' value"
                )
                break
            if not isinstance(stage_entry.get("bindings"), list):
                errors.append(
                    f"invalid reflection sidecar {label}: stage {sanitize_text(stage_key)!r} must have a 'bindings' list"
                )
                break
            validated_stages[stage_key] = stage_entry

    if errors:
        return False, None, sanitize_value(errors)

    doc = {
        "schema": REFLECT_SIDECAR_SCHEMA,
        "version": REFLECT_SIDECAR_VERSION,
        "stages": validated_stages,
    }
    return True, doc, []


def merge_reflect_sidecar_stages(
    *,
    existing_stages: Mapping[str, Any] | None = None,
    source_stages: Mapping[str, Any] | None = None,
    extra_stage_maps: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], dict[str, int]]:
    """Merge v1 stage-reflection maps deterministically without losing existing stages."""
    merged: dict[str, Any] = {}
    existing_count = 0
    if existing_stages:
        for k, v in existing_stages.items():
            merged[k] = v
        existing_count = len(existing_stages)

    source_count = 0
    if source_stages:
        for k, v in source_stages.items():
            merged[k] = v
        source_count = len(source_stages)

    base_keys = set(merged.keys())
    added_keys: set[str] = set()
    updated_keys: set[str] = set()
    for extra in extra_stage_maps:
        for k, v in extra.items():
            if k in base_keys:
                updated_keys.add(k)
            else:
                added_keys.add(k)
            merged[k] = v

    ordered_stages = {k: merged[k] for k in sorted(merged)}
    doc = {
        "schema": REFLECT_SIDECAR_SCHEMA,
        "version": REFLECT_SIDECAR_VERSION,
        "stages": ordered_stages,
    }
    stats = {
        "existing_stages": existing_count,
        "source_stages": source_count,
        "added_stages": len(added_keys),
        "updated_stages": len(updated_keys),
        "preserved_stages": len(base_keys - updated_keys),
        "total_stages": len(ordered_stages),
    }
    return doc, stats


def _normalize_merge_reflect_paths(
    merge_reflect: Path | str | Sequence[Path | str] | None,
    resolved_repo: Path,
) -> list[Path]:
    if merge_reflect is None:
        return []
    if isinstance(merge_reflect, (str, Path)):
        raw_items: Sequence[Path | str] = (merge_reflect,)
    else:
        raw_items = merge_reflect
    return [_resolve_against_repo(item, resolved_repo) for item in raw_items]


def stage_shader_pack(
    *,
    shader_pack: Path | str | None,
    shaders_dir: Path | str = DEFAULT_SHADERS_DIR_REL,
    shader_pack_name: str = DEFAULT_SHADER_PACK_NAME,
    reflect_sidecar: Path | str | None = None,
    reflection_manifest: Path | str | None = None,
    merge_reflect: Path | str | Sequence[Path | str] | None = None,
    require_sidecar_sync: bool = False,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate and atomically stage a WGPK v1/v2 `.wgsl.pack` bundle, sidecar, and optional `reflection-manifest.json`."""
    resolved_repo = Path(repo_root).resolve()
    shaders_dir_path = _resolve_against_repo(shaders_dir, resolved_repo)
    merge_paths_raw = _normalize_merge_reflect_paths(merge_reflect, resolved_repo)
    shaders_section = _empty_shaders_section()
    errors: list[str] = []
    warnings: list[str] = []

    shaders_safe, shaders_safety_err = validate_shaders_dir_safety(
        shaders_dir_path, resolved_repo
    )
    if not shaders_safe:
        shaders_section["status"] = "refused"
        errors.append(shaders_safety_err or "refused unsafe --shaders-dir")
        return sanitize_value(shaders_section), sanitize_value(errors), sanitize_value(warnings)

    if shader_pack is None and not merge_paths_raw and reflection_manifest is None:
        if reflect_sidecar is not None:
            shaders_section["status"] = "error"
            errors.append("--reflect-sidecar requires --shader-pack")
        return sanitize_value(shaders_section), sanitize_value(errors), sanitize_value(warnings)

    if not is_safe_shader_pack_name(shader_pack_name):
        shaders_section["status"] = "refused"
        errors.append(
            f"refusing unsafe shader pack destination filename: {sanitize_text(shader_pack_name)}"
        )
        return sanitize_value(shaders_section), sanitize_value(errors), sanitize_value(warnings)

    src_pack: Path | None = None
    src_pack_raw: Path | None = None
    wgpk_entries: dict[str, str] | None = None
    wgpk_summary: dict[str, Any] | None = None
    if shader_pack is not None:
        src_pack_raw = _resolve_against_repo(shader_pack, resolved_repo)
        if not src_pack_raw.name.lower().endswith(".wgsl.pack"):
            shaders_section["status"] = "refused"
            errors.append(
                f"shader-pack must end with .wgsl.pack: {basename_label(src_pack_raw)}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

        try:
            src_pack = src_pack_raw.resolve(strict=True)
        except OSError:
            shaders_section["status"] = "error"
            errors.append(
                f"shader-pack does not exist: {repo_relative_label(src_pack_raw, resolved_repo)}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

        if not src_pack.is_file():
            shaders_section["status"] = "error"
            errors.append(
                f"shader-pack is not a regular file: {repo_relative_label(src_pack_raw, resolved_repo)}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

        wgpk_ok, wgpk_summary, wgpk_entries, wgpk_errors, wgpk_warnings = validate_wgpk_pack(
            src_pack,
            label=basename_label(src_pack_raw),
        )
        shaders_section["wgpk_validation"] = wgpk_summary
        warnings.extend(wgpk_warnings)
        if not wgpk_ok:
            shaders_section["status"] = "refused"
            errors.extend(wgpk_errors)
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

    # Locate optional adjacent `<src_pack>.reflect.json` or explicit `--reflect-sidecar` override.
    sidecar_source: str | None = None
    sidecar_resolved: Path | None = None
    sidecar_meta: dict[str, Any] = {}

    if reflect_sidecar is not None:
        sidecar_source = "override"
        sidecar_raw = _resolve_against_repo(reflect_sidecar, resolved_repo)
        try:
            sidecar_candidate = sidecar_raw.resolve(strict=True)
        except OSError:
            shaders_section["status"] = "error"
            errors.append(
                f"reflect-sidecar does not exist: {repo_relative_label(sidecar_raw, resolved_repo)}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )
        if not sidecar_candidate.is_file():
            shaders_section["status"] = "error"
            errors.append(
                f"reflect-sidecar is not a regular file: "
                f"{repo_relative_label(sidecar_raw, resolved_repo)}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )
        sidecar_resolved = sidecar_candidate
    elif src_pack_raw is not None:
        adjacent_raw = Path(str(src_pack_raw) + REFLECT_SIDECAR_SUFFIX)
        if adjacent_raw.exists() or adjacent_raw.is_symlink():
            sidecar_source = "adjacent"
            try:
                adjacent_resolved = adjacent_raw.resolve(strict=True)
            except OSError:
                shaders_section["status"] = "refused"
                errors.append(
                    f"adjacent reflection sidecar is unreadable or a broken symlink: "
                    f"{basename_label(adjacent_raw)}"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            if not adjacent_resolved.is_file():
                shaders_section["status"] = "refused"
                errors.append(
                    f"adjacent reflection sidecar is not a regular file: "
                    f"{basename_label(adjacent_raw)}"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            sidecar_resolved = adjacent_resolved

    # Locate optional adjacent `reflection-manifest.json` or explicit `--reflection-manifest`.
    manifest_source: str | None = None
    manifest_resolved: Path | None = None
    manifest_meta: dict[str, Any] | None = None
    if reflection_manifest is not None:
        manifest_source = "override"
        manifest_raw = _resolve_against_repo(reflection_manifest, resolved_repo)
        try:
            manifest_candidate = manifest_raw.resolve(strict=True)
        except OSError:
            shaders_section["status"] = "error"
            errors.append(
                f"reflection-manifest does not exist: "
                f"{repo_relative_label(manifest_raw, resolved_repo)}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )
        if not manifest_candidate.is_file():
            shaders_section["status"] = "error"
            errors.append(
                f"reflection-manifest is not a regular file: "
                f"{repo_relative_label(manifest_raw, resolved_repo)}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )
        manifest_resolved = manifest_candidate
    elif src_pack_raw is not None:
        adj_manifest_raw = src_pack_raw.parent / REFLECTION_MANIFEST_NAME
        if adj_manifest_raw.exists() or adj_manifest_raw.is_symlink():
            manifest_source = "adjacent"
            try:
                adj_manifest_resolved = adj_manifest_raw.resolve(strict=True)
            except OSError:
                shaders_section["status"] = "refused"
                errors.append(
                    f"adjacent {REFLECTION_MANIFEST_NAME} is unreadable or a broken symlink"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            if not adj_manifest_resolved.is_file():
                shaders_section["status"] = "refused"
                errors.append(
                    f"adjacent {REFLECTION_MANIFEST_NAME} is not a regular file"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            manifest_resolved = adj_manifest_resolved

    resolved_shaders = shaders_dir_path.resolve()
    dst_pack = shaders_dir_path / shader_pack_name
    dst_sidecar = Path(str(dst_pack) + REFLECT_SIDECAR_SUFFIX)
    sidecar_dest_name = dst_sidecar.name
    dst_manifest = shaders_dir_path / REFLECTION_MANIFEST_NAME

    # Validate destination containment before doing any staging work.
    dest_checks: list[tuple[str, Path]] = []
    if src_pack is not None:
        dest_checks.append((shader_pack_name, dst_pack))
    if sidecar_resolved is not None or merge_paths_raw:
        dest_checks.append((sidecar_dest_name, dst_sidecar))
    if manifest_resolved is not None:
        dest_checks.append((REFLECTION_MANIFEST_NAME, dst_manifest))
    for dest_name, dst_file in dest_checks:
        if dst_file.parent.resolve() != resolved_shaders or dst_file.name != dest_name:
            shaders_section["status"] = "refused"
            errors.append(
                f"refused shader artifact escaping shaders-dir: {sanitize_text(dest_name)}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

    # Validate `reflection-manifest.json` consistency before staging anything.
    if manifest_resolved is not None:
        expected_pack_names: list[str] = [shader_pack_name]
        if src_pack_raw is not None:
            expected_pack_names.append(src_pack_raw.name)
        manifest_pack_entries = wgpk_entries
        if manifest_pack_entries is None and dst_pack.is_file():
            existing_ok, _, existing_entries, _, _ = validate_wgpk_pack(
                dst_pack, label=shader_pack_name
            )
            if existing_ok:
                manifest_pack_entries = existing_entries
        manifest_ok, manifest_meta, manifest_errors, manifest_warnings = (
            validate_reflection_manifest(
                manifest_resolved,
                expected_pack_names=expected_pack_names,
                pack_entries=manifest_pack_entries,
            )
        )
        warnings.extend(manifest_warnings)
        if not manifest_ok:
            shaders_section["status"] = "refused"
            shaders_section["reflection_manifest"] = manifest_meta
            errors.extend(manifest_errors)
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

    # Incremental `--merge-reflect` path: validate existing staged sidecar,
    # optional source sidecar, and every extra v1 sidecar BEFORE touching disk.
    if merge_paths_raw:
        extra_stage_maps: list[Mapping[str, Any]] = []
        merged_labels: list[str] = []
        for merge_raw in merge_paths_raw:
            try:
                merge_resolved = merge_raw.resolve(strict=True)
            except OSError:
                shaders_section["status"] = "error"
                errors.append(
                    f"merge-reflect sidecar does not exist: "
                    f"{repo_relative_label(merge_raw, resolved_repo)}"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            if not merge_resolved.is_file():
                shaders_section["status"] = "error"
                errors.append(
                    f"merge-reflect sidecar is not a regular file: "
                    f"{repo_relative_label(merge_raw, resolved_repo)}"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            extra_ok, extra_doc, extra_errors = load_and_validate_reflect_sidecar_v1(
                merge_resolved
            )
            if not extra_ok or extra_doc is None:
                shaders_section["status"] = "refused"
                errors.extend(extra_errors)
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            extra_stage_maps.append(extra_doc["stages"])
            merged_labels.append(
                repo_relative_label(merge_raw, resolved_repo) or basename_label(merge_raw)
            )

        source_stages: Mapping[str, Any] | None = None
        if sidecar_resolved is not None:
            src_sc_ok, src_sc_doc, src_sc_errors = load_and_validate_reflect_sidecar_v1(
                sidecar_resolved
            )
            if not src_sc_ok or src_sc_doc is None:
                shaders_section["status"] = "refused"
                errors.extend(src_sc_errors)
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            source_stages = src_sc_doc["stages"]

        existing_stages: Mapping[str, Any] | None = None
        if dst_sidecar.exists() or dst_sidecar.is_symlink():
            dst_sc_safe, dst_sc_resolved = _check_file_within_root(
                dst_sidecar, shaders_dir_path
            )
            if not dst_sc_safe or dst_sc_resolved is None:
                shaders_section["status"] = "refused"
                errors.append(
                    f"existing staged reflection sidecar is unreadable, not a regular file, "
                    f"or escapes shaders-dir: {sanitize_text(sidecar_dest_name)}"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            exist_ok, exist_doc, exist_errors = load_and_validate_reflect_sidecar_v1(
                dst_sc_resolved
            )
            if not exist_ok or exist_doc is None:
                shaders_section["status"] = "refused"
                errors.extend(exist_errors)
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )
            existing_stages = exist_doc["stages"]

        merged_doc, merge_stats = merge_reflect_sidecar_stages(
            existing_stages=existing_stages,
            source_stages=source_stages,
            extra_stage_maps=extra_stage_maps,
        )
        if wgpk_entries is not None:
            sync_info, sync_errors, sync_warnings = check_wgpk_sidecar_sync(
                wgpk_entries,
                merged_doc["stages"],
                pack_name=shader_pack_name,
                sidecar_name=sidecar_dest_name,
                require_sync=require_sidecar_sync,
            )
            shaders_section["sidecar_sync"] = sync_info
            warnings.extend(sync_warnings)
            if sync_errors:
                shaders_section["status"] = "refused"
                errors.extend(sync_errors)
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )

        merged_payload = (json.dumps(merged_doc, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )

        pack_plan: FileStagePlan | None = None
        if src_pack is not None:
            try:
                pack_plan = plan_file_stage(src_pack, dst_pack)
            except (OSError, StageBrowserBootError) as exc:
                shaders_section["status"] = "error"
                errors.append(
                    f"failed to inspect shader artifact {sanitize_text(shader_pack_name)}: "
                    f"{sanitize_text(str(exc))}"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )

        manifest_plan: FileStagePlan | None = None
        if manifest_resolved is not None:
            try:
                manifest_plan = plan_file_stage(manifest_resolved, dst_manifest)
            except (OSError, StageBrowserBootError) as exc:
                shaders_section["status"] = "error"
                errors.append(
                    f"failed to inspect shader artifact {REFLECTION_MANIFEST_NAME}: "
                    f"{sanitize_text(str(exc))}"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )

        try:
            sidecar_plan = plan_bytes_stage(merged_payload, dst_sidecar)
        except (OSError, StageBrowserBootError) as exc:
            shaders_section["status"] = "error"
            errors.append(
                f"failed to inspect shader artifact {sanitize_text(sidecar_dest_name)}: "
                f"{sanitize_text(str(exc))}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

        file_pairs: list[tuple[Path, Path]] = []
        byte_pairs: list[tuple[bytes, Path]] = []
        if src_pack is not None and pack_plan is not None and pack_plan.needs_copy:
            file_pairs.append((src_pack, dst_pack))
        if manifest_resolved is not None and manifest_plan is not None and manifest_plan.needs_copy:
            file_pairs.append((manifest_resolved, dst_manifest))
        if sidecar_plan.needs_copy:
            byte_pairs.append((merged_payload, dst_sidecar))

        if (file_pairs or byte_pairs) and not dry_run:
            try:
                _atomic_stage_mixed(file_pairs, byte_pairs)
            except OSError as exc:
                shaders_section["status"] = "error"
                errors.append(
                    f"failed to atomically stage shader artifacts: {sanitize_text(str(exc))}"
                )
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )

        copied = 0
        planned = 0
        unchanged = 0
        if pack_plan is not None:
            if not pack_plan.needs_copy:
                unchanged += 1
                pack_action = "unchanged"
            elif dry_run:
                planned += 1
                pack_action = "plan_copy"
            else:
                copied += 1
                pack_action = "copied"
            pack_entry: dict[str, Any] = {
                "name": sanitize_text(shader_pack_name),
                "action": pack_action,
                "reason": pack_plan.reason,
                "size_bytes": pack_plan.size_bytes,
                "sha256": pack_plan.sha256,
            }
            if wgpk_summary is not None:
                pack_entry["wgpk_version"] = wgpk_summary.get("version")
                pack_entry["entry_count"] = wgpk_summary.get("entry_count")
            shaders_section["pack"] = pack_entry

        if manifest_plan is not None and manifest_meta is not None:
            if not manifest_plan.needs_copy:
                unchanged += 1
                manifest_action = "unchanged"
            elif dry_run:
                planned += 1
                manifest_action = "plan_copy"
            else:
                copied += 1
                manifest_action = "copied"
            shaders_section["reflection_manifest"] = {
                **manifest_meta,
                "action": manifest_action,
                "reason": manifest_plan.reason,
                "size_bytes": manifest_plan.size_bytes,
                "sha256": manifest_plan.sha256,
                "source": manifest_source,
            }
            if manifest_action in ("copied", "unchanged"):
                shaders_section["reflection_manifest_staged"] = True

        if not sidecar_plan.needs_copy:
            unchanged += 1
            sidecar_action = "unchanged"
        elif dry_run:
            planned += 1
            sidecar_action = "plan_copy"
        else:
            copied += 1
            sidecar_action = "copied"

        sidecar_entry: dict[str, Any] = {
            "name": sanitize_text(sidecar_dest_name),
            "action": sidecar_action,
            "reason": sidecar_plan.reason,
            "size_bytes": sidecar_plan.size_bytes,
            "sha256": sidecar_plan.sha256,
            "source": "merged",
            "stage_count": merge_stats["total_stages"],
            "merged": True,
            "merged_files": merged_labels,
            "merge_stats": merge_stats,
        }
        shaders_section["sidecar"] = sidecar_entry
        if sidecar_action in ("copied", "unchanged"):
            shaders_section["sidecar_staged"] = True
            shaders_section["sidecar_bytes"] = sidecar_plan.size_bytes

        shaders_section["copied_count"] = copied
        shaders_section["planned_count"] = planned
        shaders_section["unchanged_count"] = unchanged
        if dry_run:
            shaders_section["status"] = "planned" if planned > 0 else "unchanged"
        elif copied > 0:
            shaders_section["status"] = "staged"
        else:
            shaders_section["status"] = "unchanged"

        return sanitize_value(shaders_section), sanitize_value(errors), sanitize_value(warnings)

    # Validate sidecar JSON and WGPK synchronization BEFORE staging either the
    # pack or the sidecar so invalid/desynced sidecars never clobber existing files.
    if sidecar_resolved is not None:
        sidecar_ok, sidecar_meta, sidecar_stages_map, sidecar_errors = (
            _load_reflect_sidecar_with_stages(sidecar_resolved)
        )
        if not sidecar_ok:
            shaders_section["status"] = "refused"
            errors.extend(sidecar_errors)
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )
        if wgpk_entries is not None:
            sync_info, sync_errors, sync_warnings = check_wgpk_sidecar_sync(
                wgpk_entries,
                sidecar_stages_map,
                pack_name=shader_pack_name,
                sidecar_name=sidecar_dest_name,
                require_sync=require_sidecar_sync,
            )
            shaders_section["sidecar_sync"] = sync_info
            warnings.extend(sync_warnings)
            if sync_errors:
                shaders_section["status"] = "refused"
                errors.extend(sync_errors)
                return (
                    sanitize_value(shaders_section),
                    sanitize_value(errors),
                    sanitize_value(warnings),
                )

    candidates: list[tuple[str, str, Path, Path]] = []
    if src_pack is not None:
        candidates.append(("pack", shader_pack_name, src_pack, dst_pack))
    if sidecar_resolved is not None:
        candidates.append(("sidecar", sidecar_dest_name, sidecar_resolved, dst_sidecar))
    if manifest_resolved is not None:
        candidates.append(("manifest", REFLECTION_MANIFEST_NAME, manifest_resolved, dst_manifest))

    plans: list[tuple[str, str, Path, Path, FileStagePlan]] = []
    for kind, dest_name, src_file, dst_file in candidates:
        try:
            plan = plan_file_stage(src_file, dst_file)
        except (OSError, StageBrowserBootError) as exc:
            shaders_section["status"] = "error"
            errors.append(
                f"failed to inspect shader artifact {sanitize_text(dest_name)}: "
                f"{sanitize_text(str(exc))}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )
        plans.append((kind, dest_name, src_file, dst_file, plan))

    to_copy = [
        (src_file, dst_file)
        for _, _, src_file, dst_file, plan in plans
        if plan.needs_copy
    ]
    if to_copy and not dry_run:
        try:
            atomic_copy_files(to_copy)
        except OSError as exc:
            shaders_section["status"] = "error"
            errors.append(
                f"failed to atomically stage shader artifacts: {sanitize_text(str(exc))}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

    copied = 0
    planned = 0
    unchanged = 0
    for kind, dest_name, _, _, plan in plans:
        if not plan.needs_copy:
            unchanged += 1
            action = "unchanged"
        elif dry_run:
            planned += 1
            action = "plan_copy"
        else:
            copied += 1
            action = "copied"

        entry: dict[str, Any] = {
            "name": sanitize_text(dest_name),
            "action": action,
            "reason": plan.reason,
            "size_bytes": plan.size_bytes,
            "sha256": plan.sha256,
        }
        if kind == "pack":
            if wgpk_summary is not None:
                entry["wgpk_version"] = wgpk_summary.get("version")
                entry["entry_count"] = wgpk_summary.get("entry_count")
            shaders_section["pack"] = entry
        elif kind == "sidecar":
            entry["source"] = sidecar_source
            entry["stage_count"] = int(sidecar_meta.get("stage_count", 0))
            shaders_section["sidecar"] = entry
            if action in ("copied", "unchanged"):
                shaders_section["sidecar_staged"] = True
                shaders_section["sidecar_bytes"] = plan.size_bytes
        else:
            if manifest_meta is not None:
                entry = {**manifest_meta, **entry}
            entry["source"] = manifest_source
            shaders_section["reflection_manifest"] = entry
            if action in ("copied", "unchanged"):
                shaders_section["reflection_manifest_staged"] = True

    shaders_section["copied_count"] = copied
    shaders_section["planned_count"] = planned
    shaders_section["unchanged_count"] = unchanged
    if dry_run:
        shaders_section["status"] = "planned" if planned > 0 else "unchanged"
    elif copied > 0:
        shaders_section["status"] = "staged"
    else:
        shaders_section["status"] = "unchanged"

    return sanitize_value(shaders_section), sanitize_value(errors), sanitize_value(warnings)


# ---------------------------------------------------------------------------
# Validated boot-options.json & input-script preset staging (W119 / O36 / O37 / T46)
# ---------------------------------------------------------------------------


def sanitize_input_script_preset(raw: Any) -> str | None:
    """Sanitize and canonicalize an input-script preset against the allowlist.

    Matches `ALLOWED_INPUT_SCRIPT_PRESETS` in `tools/run_browser_boot.py` and
    `web/input/input_bridge.js` (`start-screen`, `menu-flow`, `calibration-skip`,
    `intro-watch`).
    """
    if not isinstance(raw, str):
        return None
    trimmed = raw.strip()
    if (
        not trimmed
        or len(trimmed) > 64
        or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in trimmed)
    ):
        return None
    lower = trimmed.lower()
    if not _SAFE_INPUT_SCRIPT_RE.match(lower):
        return None
    return ALLOWED_INPUT_SCRIPT_PRESETS.get(lower)


def validate_input_script_preset(raw: Any) -> tuple[bool, str | None, str | None]:
    """Validate an input-script preset; returns `(ok, canonical_or_none, error_or_none)`."""
    if raw is None:
        return True, None, None
    allowed = ", ".join(ALLOWED_INPUT_SCRIPT_PRESETS.values())
    if not isinstance(raw, str):
        return False, None, f"invalid input_script: must be a string (allowed: {allowed})"
    cleaned = sanitize_input_script_preset(raw)
    if cleaned is None:
        return (
            False,
            None,
            sanitize_text(f"invalid input_script {raw!r} (allowed: {allowed})"),
        )
    return True, cleaned, None


def _validate_dump_pipelines_dir(value: Any) -> tuple[bool, str | None, str | None]:
    """Validate `dump_pipelines_dir` as a safe virtual MEMFS path (e.g. `/pg6_dump`)."""
    if not isinstance(value, str):
        return (
            False,
            None,
            "invalid boot-options.json: 'dump_pipelines_dir' must be a non-empty string",
        )
    cleaned = value.strip()
    if (
        not cleaned
        or len(cleaned) > 256
        or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in cleaned)
        or "\\" in cleaned
        or re.match(r"^[A-Za-z]:", cleaned)
        or ".." in cleaned.split("/")
    ):
        return (
            False,
            None,
            "invalid boot-options.json: 'dump_pipelines_dir' must be a safe non-empty virtual path (e.g. '/pg6_dump')",
        )
    return True, cleaned, None


def _empty_boot_options_section() -> dict[str, Any]:
    return {
        "status": "skipped",
        "staged": False,
        "filename": BOOT_OPTIONS_NAME,
        "action": None,
        "reason": None,
        "size_bytes": 0,
        "sha256": None,
        "options": None,
    }


def parse_and_validate_boot_options(
    *,
    boot_options: Mapping[str, Any] | str | Path | None = None,
    verify_window_marks: bool = False,
    dump_pipelines: bool = False,
    input_script: str | None = None,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> tuple[bool, dict[str, Any] | None, list[str]]:
    """Parse and schema-validate `boot-options.json` input and CLI flags."""
    if (
        boot_options is None
        and not verify_window_marks
        and not dump_pipelines
        and input_script is None
    ):
        return True, None, []

    resolved_repo = Path(repo_root).resolve()
    raw_obj: Any = {}

    if boot_options is not None:
        if isinstance(boot_options, Mapping):
            raw_obj = dict(boot_options)
        elif isinstance(boot_options, Path):
            try:
                raw_text = boot_options.read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError) as exc:
                return (
                    False,
                    None,
                    [
                        sanitize_text(
                            f"failed to read boot-options file {basename_label(boot_options)}: {exc}"
                        )
                    ],
                )
            try:
                raw_obj = json.loads(raw_text)
            except json.JSONDecodeError as exc:
                return (
                    False,
                    None,
                    [
                        sanitize_text(
                            f"invalid JSON in boot-options file {basename_label(boot_options)} "
                            f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
                        )
                    ],
                )
        elif isinstance(boot_options, str):
            stripped = boot_options.strip()
            looks_like_json = (
                stripped.startswith(("{", "[", '"'))
                or stripped in ("true", "false", "null")
                or (bool(stripped) and stripped[0] in "-0123456789")
            )
            if looks_like_json:
                try:
                    raw_obj = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    return (
                        False,
                        None,
                        [
                            sanitize_text(
                                f"invalid inline boot-options JSON "
                                f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
                            )
                        ],
                    )
            else:
                options_path = _resolve_against_repo(boot_options, resolved_repo)
                if not options_path.is_file():
                    return (
                        False,
                        None,
                        [
                            f"boot-options JSON file does not exist: "
                            f"{repo_relative_label(options_path, resolved_repo)}"
                        ],
                    )
                try:
                    raw_text = options_path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeDecodeError) as exc:
                    return (
                        False,
                        None,
                        [
                            sanitize_text(
                                f"failed to read boot-options file "
                                f"{basename_label(options_path)}: {exc}"
                            )
                        ],
                    )
                try:
                    raw_obj = json.loads(raw_text)
                except json.JSONDecodeError as exc:
                    return (
                        False,
                        None,
                        [
                            sanitize_text(
                                f"invalid JSON in boot-options file "
                                f"{basename_label(options_path)} "
                                f"(line {exc.lineno}, col {exc.colno}): {exc.msg}"
                            )
                        ],
                    )
        else:
            return (
                False,
                None,
                ["invalid boot-options input: expected JSON object, JSON string, or file path"],
            )

    if not isinstance(raw_obj, dict):
        return (
            False,
            None,
            ["invalid boot-options.json: top-level JSON value must be an object"],
        )

    errors: list[str] = []
    validated: dict[str, Any] = {}
    for raw_key, value in raw_obj.items():
        key = str(raw_key)
        if key not in BOOT_OPTIONS_ALLOWED_KEYS:
            errors.append(
                f"invalid boot-options.json: unsupported key '{sanitize_text(key)}' "
                f"(allowed: {', '.join(sorted(BOOT_OPTIONS_ALLOWED_KEYS))})"
            )
            continue
        if key in BOOT_OPTIONS_UINT_KEYS:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > _UINT64_MAX:
                errors.append(
                    f"invalid boot-options.json: '{key}' must be a non-negative uint64 integer"
                )
                continue
            validated[key] = int(value)
        elif key in BOOT_OPTIONS_BOOL_KEYS:
            if not isinstance(value, bool):
                errors.append(
                    f"invalid boot-options.json: '{key}' must be a boolean"
                )
                continue
            validated[key] = value
        elif key == "dump_pipelines_dir":
            dir_ok, cleaned_dir, dir_err = _validate_dump_pipelines_dir(value)
            if not dir_ok or cleaned_dir is None:
                errors.append(
                    dir_err
                    or "invalid boot-options.json: 'dump_pipelines_dir' must be a safe non-empty virtual path"
                )
                continue
            validated[key] = cleaned_dir
        elif key == "input_script":
            preset_ok, cleaned_preset, preset_err = validate_input_script_preset(value)
            if not preset_ok or cleaned_preset is None:
                errors.append(
                    f"invalid boot-options.json: {preset_err or 'invalid input_script'}"
                )
                continue
            validated[key] = cleaned_preset

    if input_script is not None:
        cli_preset_ok, cli_preset, cli_preset_err = validate_input_script_preset(input_script)
        if not cli_preset_ok or cli_preset is None:
            errors.append(cli_preset_err or "invalid --input-script preset")
        else:
            validated["input_script"] = cli_preset

    if errors:
        return False, None, sanitize_value(errors)

    if verify_window_marks:
        validated["verify_window_marks"] = True

    if dump_pipelines:
        validated["dump_pipelines_dir"] = DEFAULT_DUMP_PIPELINES_DIR

    ordered = {key: validated[key] for key in sorted(validated)}
    return True, ordered, []


def stage_boot_options(
    *,
    boot_dir: Path | str = DEFAULT_BOOT_DIR_REL,
    boot_options: Mapping[str, Any] | str | Path | None = None,
    verify_window_marks: bool = False,
    dump_pipelines: bool = False,
    input_script: str | None = None,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate and atomically stage sanitized `boot-options.json` into `--boot-dir`."""
    resolved_repo = Path(repo_root).resolve()
    boot_path = _resolve_against_repo(boot_dir, resolved_repo)
    section = _empty_boot_options_section()
    errors: list[str] = []
    warnings: list[str] = []

    boot_safe, boot_safety_err = validate_boot_dir_safety(boot_path, resolved_repo)
    if not boot_safe:
        section["status"] = "refused"
        errors.append(boot_safety_err or "refused unsafe --boot-dir")
        return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)

    ok_val, validated, val_errors = parse_and_validate_boot_options(
        boot_options=boot_options,
        verify_window_marks=verify_window_marks,
        dump_pipelines=dump_pipelines,
        input_script=input_script,
        repo_root=resolved_repo,
    )
    if not ok_val:
        section["status"] = "refused"
        errors.extend(val_errors)
        return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)

    if validated is None:
        return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)

    resolved_boot = boot_path.resolve()
    dst_file = boot_path / BOOT_OPTIONS_NAME
    if dst_file.parent.resolve() != resolved_boot or dst_file.name != BOOT_OPTIONS_NAME:
        section["status"] = "refused"
        errors.append("refused boot-options.json escaping boot-dir")
        return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)

    payload = (json.dumps(validated, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        plan = plan_bytes_stage(payload, dst_file)
    except (OSError, StageBrowserBootError) as exc:
        section["status"] = "error"
        errors.append(
            f"failed to inspect {BOOT_OPTIONS_NAME}: {sanitize_text(str(exc))}"
        )
        return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)

    if plan.needs_copy and not dry_run:
        try:
            atomic_write_bytes(payload, dst_file)
        except OSError as exc:
            section["status"] = "error"
            errors.append(
                f"failed to atomically stage {BOOT_OPTIONS_NAME}: {sanitize_text(str(exc))}"
            )
            return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)

    if not plan.needs_copy:
        action = "unchanged"
        status = "unchanged"
        staged = True
    elif dry_run:
        action = "plan_copy"
        status = "planned"
        staged = False
    else:
        action = "copied"
        status = "staged"
        staged = True

    section["status"] = status
    section["staged"] = staged
    section["action"] = action
    section["reason"] = plan.reason
    section["size_bytes"] = plan.size_bytes
    section["sha256"] = plan.sha256
    section["options"] = validated
    return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)


# ---------------------------------------------------------------------------
# Staged fast-boot bundle & WGPK/sidecar verification (T70)
# ---------------------------------------------------------------------------


def _empty_verification_section() -> dict[str, Any]:
    return {
        "checked": False,
        "ok": True,
        "status": "skipped",
        "web_dist": None,
        "boot_dir": None,
        "shaders_dir": None,
        "errors": [],
        "warnings": [],
    }


def _verify_staged_web_dist(
    web_dist: Path,
    *,
    require_fast_boot_exports: bool = True,
    require_guest_manifest: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Verify staged `runtime.js`, `runtime.wasm` (ABI + fast-boot exports), and `guest_manifest.json`."""
    errors: list[str] = []
    warnings: list[str] = []

    runtime_js_info: dict[str, Any] = {"present": False, "ok": False, "size_bytes": 0}
    runtime_wasm_info: dict[str, Any] = {
        "present": False,
        "ok": False,
        "size_bytes": 0,
        "fast_boot_exports_ok": False,
        "has_main_export": False,
        "main_export_name": None,
        "has_input_state_export": False,
        "input_state_export_name": None,
        "has_audio_ring_export": False,
        "audio_ring_export_name": None,
        "missing_fast_boot_exports": [name for name, _ in FAST_BOOT_EXPORT_SPECS],
    }
    guest_manifest_info: dict[str, Any] = {
        "present": False,
        "ok": not require_guest_manifest,
        "size_bytes": 0,
        "guest_modules": [],
        "guest_module_count": 0,
    }

    if not web_dist.is_dir():
        errors.append("staged web-dist directory does not exist or is not a directory")
        return (
            {
                "status": "fail",
                "runtime_js": runtime_js_info,
                "runtime_wasm": runtime_wasm_info,
                "guest_manifest": guest_manifest_info,
                "side_module_count": 0,
            },
            sanitize_value(errors),
            sanitize_value(warnings),
        )

    js_ok, resolved_js = _check_file_within_root(web_dist / RUNTIME_JS_NAME, web_dist)
    if not js_ok or resolved_js is None:
        errors.append(f"staged {RUNTIME_JS_NAME} missing or invalid in web-dist")
    else:
        runtime_js_info["present"] = True
        try:
            js_size = int(resolved_js.stat().st_size)
        except OSError as exc:
            errors.append(sanitize_text(f"failed to stat staged {RUNTIME_JS_NAME}: {exc}"))
        else:
            runtime_js_info["size_bytes"] = js_size
            if js_size <= 0:
                errors.append(f"staged {RUNTIME_JS_NAME} is empty (0 bytes)")
            else:
                runtime_js_info["ok"] = True

    wasm_ok, resolved_wasm = _check_file_within_root(web_dist / RUNTIME_WASM_NAME, web_dist)
    if not wasm_ok or resolved_wasm is None:
        errors.append(f"staged {RUNTIME_WASM_NAME} missing or invalid in web-dist")
    else:
        runtime_wasm_info["present"] = True
        val = validate_wasm_candidate(
            resolved_wasm,
            require_fast_boot_exports=require_fast_boot_exports,
        )
        runtime_wasm_info = {
            "present": True,
            "ok": bool(val.get("ok")),
            "size_bytes": int(val.get("byte_size", 0)),
            "is_memory64": bool(val.get("is_memory64")),
            "is_shared_memory": bool(val.get("is_shared_memory")),
            "all_host_imports_ok": bool(val.get("all_host_imports_ok")),
            "host_imports_valid": int(val.get("host_imports_valid", 0)),
            "host_imports_total": int(val.get("host_imports_total", 0)),
            "has_main_export": bool(val.get("has_main_export")),
            "main_export_name": val.get("main_export_name"),
            "has_input_state_export": bool(val.get("has_input_state_export")),
            "input_state_export_name": val.get("input_state_export_name"),
            "has_audio_ring_export": bool(val.get("has_audio_ring_export")),
            "audio_ring_export_name": val.get("audio_ring_export_name"),
            "fast_boot_exports_ok": bool(val.get("fast_boot_exports_ok")),
            "all_browser_boot_exports_ok": bool(val.get("all_browser_boot_exports_ok")),
            "missing_fast_boot_exports": list(val.get("missing_fast_boot_exports") or ()),
        }
        warnings.extend(val.get("warnings") or [])
        if not val.get("ok"):
            errors.extend(val.get("errors") or ["staged runtime.wasm failed validation"])

    gm_raw = web_dist / GUEST_MANIFEST_NAME
    if gm_raw.exists() or gm_raw.is_symlink():
        gm_safe, resolved_gm = _check_file_within_root(gm_raw, web_dist)
        if not gm_safe or resolved_gm is None:
            guest_manifest_info["ok"] = False
            errors.append(
                f"staged {GUEST_MANIFEST_NAME} is an escaping symlink or not a regular file"
            )
        else:
            gm_ok, gm_summary, gm_errors = validate_guest_manifest(resolved_gm)
            guest_manifest_info = {"present": True, **gm_summary}
            if not gm_ok:
                errors.extend(gm_errors)
    elif require_guest_manifest:
        guest_manifest_info["ok"] = False
        errors.append(f"staged {GUEST_MANIFEST_NAME} missing in web-dist")

    side_scan = discover_side_module_wasms(web_dist)
    if side_scan.rejected_unsafe_names:
        errors.append(
            "unsafe side-module filename(s) in staged web-dist: "
            + ", ".join(side_scan.rejected_unsafe_names)
        )
    if side_scan.skipped_escaping_symlinks > 0:
        warnings.append(
            f"skipped {side_scan.skipped_escaping_symlinks} escaping *.wasm symlink(s) in staged web-dist"
        )
    for mod in side_scan.modules:
        try:
            with mod.source_path.open("rb") as fh:
                hdr = fh.read(WASM_HEADER_BYTES)
        except OSError as exc:
            errors.append(sanitize_text(f"failed to read staged side module {mod.name}: {exc}"))
            continue
        if len(hdr) < WASM_HEADER_BYTES or hdr[:4] != WASM_MAGIC or struct.unpack_from("<I", hdr, 4)[0] != WASM_VERSION:
            errors.append(f"staged side module {sanitize_text(mod.name)} has invalid Wasm header")

    section = {
        "status": "pass" if len(errors) == 0 else "fail",
        "runtime_js": runtime_js_info,
        "runtime_wasm": runtime_wasm_info,
        "guest_manifest": guest_manifest_info,
        "side_module_count": len(side_scan.modules),
    }
    return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)


def _verify_staged_boot_dir(
    boot_dir: Path,
    *,
    require_boot_options: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Verify staged `program-manifest.json`, `.pgim` guest image(s), and optional `boot-options.json`."""
    errors: list[str] = []
    warnings: list[str] = []

    manifest_info: dict[str, Any] = {
        "present": False,
        "ok": False,
        "name": PROGRAM_MANIFEST_NAME,
        "module_count": 0,
    }
    boot_opts_info: dict[str, Any] = {
        "present": False,
        "ok": not require_boot_options,
        "name": BOOT_OPTIONS_NAME,
        "options": None,
    }
    pgim_entries: list[dict[str, Any]] = []
    has_guest_pgim = False

    if not boot_dir.is_dir():
        errors.append("staged boot-dir does not exist or is not a directory")
        return (
            {
                "status": "fail",
                "program_manifest": manifest_info,
                "has_guest_pgim": False,
                "pgim_count": 0,
                "valid_pgim_count": 0,
                "pgims": [],
                "boot_options": boot_opts_info,
            },
            sanitize_value(errors),
            sanitize_value(warnings),
        )

    pm_ok, resolved_pm = _check_file_within_root(boot_dir / PROGRAM_MANIFEST_NAME, boot_dir)
    if not pm_ok or resolved_pm is None:
        errors.append(f"staged {PROGRAM_MANIFEST_NAME} missing or invalid in boot-dir")
    else:
        manifest_info["present"] = True
        try:
            pm_doc = json.loads(resolved_pm.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(
                sanitize_text(f"failed to parse staged {PROGRAM_MANIFEST_NAME}: {type(exc).__name__}")
            )
        else:
            if not isinstance(pm_doc, dict):
                errors.append(f"staged {PROGRAM_MANIFEST_NAME} root must be a JSON object")
            else:
                modules_raw = pm_doc.get("modules")
                if modules_raw is not None and not isinstance(modules_raw, list):
                    errors.append(f"staged {PROGRAM_MANIFEST_NAME} 'modules' must be a JSON list")
                else:
                    manifest_info["module_count"] = len(modules_raw) if isinstance(modules_raw, list) else 0
                    manifest_info["ok"] = True

    discovery = discover_boot_artifacts(boot_dir)
    if discovery.skipped_escaping_symlinks > 0:
        warnings.append(
            f"skipped {discovery.skipped_escaping_symlinks} escaping symlink(s) in staged boot-dir"
        )
    pgim_artifacts = [item for item in discovery.artifacts if item.kind == "pgim"]
    valid_pgim_count = 0
    for artifact in pgim_artifacts:
        pgim_errors: list[str] = []
        size_bytes = 0
        magic_ok = False
        try:
            size_bytes = int(artifact.source_path.stat().st_size)
            with artifact.source_path.open("rb") as fh:
                hdr = fh.read(len(PGIM_MAGIC))
        except OSError as exc:
            pgim_errors.append(
                sanitize_text(f"failed to read staged PGIM {artifact.rel_posix}: {exc}")
            )
            hdr = b""
        if len(hdr) < len(PGIM_MAGIC):
            pgim_errors.append(
                f"staged PGIM '{artifact.rel_posix}' truncated ({len(hdr)} bytes)"
            )
        elif hdr != PGIM_MAGIC:
            pgim_errors.append(
                f"staged PGIM '{artifact.rel_posix}' has invalid magic (expected 'PGIM')"
            )
        else:
            magic_ok = True
        if artifact.source_path.name.lower() == "guest.pgim" and magic_ok:
            has_guest_pgim = True
        item_ok = len(pgim_errors) == 0
        if item_ok:
            valid_pgim_count += 1
        else:
            errors.extend(pgim_errors)
        pgim_entries.append(
            {
                "rel_path": artifact.rel_posix,
                "size_bytes": size_bytes,
                "magic_ok": magic_ok,
                "ok": item_ok,
                "errors": pgim_errors,
            }
        )

    if not pgim_artifacts:
        errors.append("no staged .pgim guest image (e.g. guest.pgim) found in boot-dir")

    opts_raw = boot_dir / BOOT_OPTIONS_NAME
    if opts_raw.exists() or opts_raw.is_symlink():
        opts_safe, resolved_opts = _check_file_within_root(opts_raw, boot_dir)
        if not opts_safe or resolved_opts is None:
            boot_opts_info["ok"] = False
            errors.append(
                f"staged {BOOT_OPTIONS_NAME} is an escaping symlink or not a regular file"
            )
        else:
            boot_opts_info["present"] = True
            opts_ok, opts_parsed, opts_errors = parse_and_validate_boot_options(
                boot_options=resolved_opts
            )
            boot_opts_info["ok"] = opts_ok
            boot_opts_info["options"] = opts_parsed
            if not opts_ok:
                errors.extend(opts_errors)
    elif require_boot_options:
        boot_opts_info["ok"] = False
        errors.append(f"staged {BOOT_OPTIONS_NAME} missing in boot-dir")

    section = {
        "status": "pass" if len(errors) == 0 else "fail",
        "program_manifest": manifest_info,
        "has_guest_pgim": has_guest_pgim,
        "pgim_count": len(pgim_entries),
        "valid_pgim_count": valid_pgim_count,
        "pgims": pgim_entries,
        "boot_options": boot_opts_info,
    }
    return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)


def _verify_staged_shaders_dir(
    shaders_dir: Path,
    *,
    shader_pack_name: str | None = None,
    require_sidecar: bool = False,
    require_sidecar_sync: bool = False,
    require_reflection_manifest: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Verify staged WGPK v1/v2 packs, `<pack>.reflect.json` sidecar sync, and `reflection-manifest.json`."""
    errors: list[str] = []
    warnings: list[str] = []

    if not shaders_dir.is_dir():
        errors.append("staged shaders-dir does not exist or is not a directory")
        return (
            {
                "status": "fail",
                "pack_count": 0,
                "valid_pack_count": 0,
                "packs": [],
                "sidecars": [],
                "reflection_manifest": None,
            },
            sanitize_value(errors),
            sanitize_value(warnings),
        )

    pack_paths: list[tuple[str, Path]] = []
    if shader_pack_name is not None:
        pack_raw = shaders_dir / shader_pack_name
        pack_safe, resolved_pack = _check_file_within_root(pack_raw, shaders_dir)
        if not pack_safe or resolved_pack is None:
            errors.append(
                f"staged shader pack {sanitize_text(shader_pack_name)} missing or invalid in shaders-dir"
            )
        else:
            pack_paths.append((shader_pack_name, resolved_pack))
    else:
        for entry in sorted(shaders_dir.iterdir(), key=lambda p: (_natural_sort_key(p.name), p.name)):
            if not entry.name.lower().endswith(".wgsl.pack"):
                continue
            ok_file, resolved = _check_file_within_root(entry, shaders_dir)
            if not ok_file or resolved is None:
                warnings.append(
                    f"skipped escaping or non-regular shader pack in shaders-dir: {sanitize_text(entry.name)}"
                )
                continue
            pack_paths.append((entry.name, resolved))

    if not pack_paths and not errors:
        errors.append("no staged .wgsl.pack shader bundle found in shaders-dir")

    pack_reports: list[dict[str, Any]] = []
    sidecar_reports: list[dict[str, Any]] = []
    combined_entries: dict[str, str] = {}
    valid_pack_count = 0

    for pack_name, pack_path in pack_paths:
        pk_ok, pk_summary, pk_entries, pk_errors, pk_warnings = validate_wgpk_pack(
            pack_path, label=pack_name
        )
        pack_reports.append(pk_summary)
        warnings.extend(pk_warnings)
        if pk_ok:
            valid_pack_count += 1
            combined_entries.update(pk_entries)
        else:
            errors.extend(pk_errors)

        sc_name = pack_name + REFLECT_SIDECAR_SUFFIX
        sc_raw = shaders_dir / sc_name
        if sc_raw.exists() or sc_raw.is_symlink():
            sc_safe, resolved_sc = _check_file_within_root(sc_raw, shaders_dir)
            if not sc_safe or resolved_sc is None:
                err = f"staged reflection sidecar {sanitize_text(sc_name)} is an escaping symlink or not a regular file"
                errors.append(err)
                sidecar_reports.append({"name": sc_name, "present": True, "ok": False, "errors": [err]})
            else:
                sc_ok, sc_meta, sc_stages_map, sc_errors = _load_reflect_sidecar_with_stages(
                    resolved_sc
                )
                if not sc_ok:
                    errors.extend(sc_errors)
                    sidecar_reports.append(
                        {"name": sc_name, "present": True, "ok": False, "errors": sc_errors}
                    )
                else:
                    sync_info: dict[str, Any] | None = None
                    sync_errors: list[str] = []
                    if pk_ok:
                        sync_info, sync_errors, sync_warnings = check_wgpk_sidecar_sync(
                            pk_entries,
                            sc_stages_map,
                            pack_name=pack_name,
                            sidecar_name=sc_name,
                            require_sync=require_sidecar_sync,
                        )
                        warnings.extend(sync_warnings)
                        errors.extend(sync_errors)
                    sidecar_reports.append(
                        {
                            "name": sc_name,
                            "present": True,
                            "ok": len(sync_errors) == 0,
                            **sc_meta,
                            "sync": sync_info,
                        }
                    )
        elif require_sidecar:
            err = f"staged reflection sidecar {sanitize_text(sc_name)} missing in shaders-dir"
            errors.append(err)
            sidecar_reports.append({"name": sc_name, "present": False, "ok": False, "errors": [err]})

    manifest_report: dict[str, Any] | None = None
    rm_raw = shaders_dir / REFLECTION_MANIFEST_NAME
    if rm_raw.exists() or rm_raw.is_symlink():
        rm_safe, resolved_rm = _check_file_within_root(rm_raw, shaders_dir)
        if not rm_safe or resolved_rm is None:
            err = f"staged {REFLECTION_MANIFEST_NAME} is an escaping symlink or not a regular file"
            errors.append(err)
            manifest_report = {"name": REFLECTION_MANIFEST_NAME, "present": True, "ok": False, "errors": [err]}
        else:
            rm_ok, rm_summary, rm_errors, rm_warnings = validate_reflection_manifest(
                resolved_rm,
                expected_pack_names=[name for name, _ in pack_paths],
                pack_entries=combined_entries if valid_pack_count > 0 else None,
            )
            warnings.extend(rm_warnings)
            if not rm_ok:
                errors.extend(rm_errors)
            manifest_report = {"present": True, **rm_summary}
    elif require_reflection_manifest:
        err = f"staged {REFLECTION_MANIFEST_NAME} missing in shaders-dir"
        errors.append(err)
        manifest_report = {"name": REFLECTION_MANIFEST_NAME, "present": False, "ok": False, "errors": [err]}

    section = {
        "status": "pass" if len(errors) == 0 else "fail",
        "pack_count": len(pack_reports),
        "valid_pack_count": valid_pack_count,
        "packs": pack_reports,
        "sidecars": sidecar_reports,
        "reflection_manifest": manifest_report,
    }
    return sanitize_value(section), sanitize_value(errors), sanitize_value(warnings)


def verify_staged_bundle(
    *,
    web_dist: Path | str | None = None,
    boot_dir: Path | str | None = None,
    shaders_dir: Path | str | None = None,
    shader_pack_name: str | None = None,
    require_fast_boot_exports: bool = True,
    require_guest_manifest: bool = False,
    require_boot_options: bool = False,
    require_sidecar: bool = False,
    require_sidecar_sync: bool = False,
    require_reflection_manifest: bool = False,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
) -> dict[str, Any]:
    """Verify staged browser fast-boot artifacts (`web_dist`, `boot_dir`, `shaders_dir`) and return a sanitized report."""
    resolved_repo = Path(repo_root).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    web_dist_section: dict[str, Any] | None = None
    boot_dir_section: dict[str, Any] | None = None
    shaders_dir_section: dict[str, Any] | None = None

    if web_dist is not None:
        web_dist_path = _resolve_against_repo(web_dist, resolved_repo)
        web_dist_section, wd_errors, wd_warnings = _verify_staged_web_dist(
            web_dist_path,
            require_fast_boot_exports=require_fast_boot_exports,
            require_guest_manifest=require_guest_manifest,
        )
        errors.extend(wd_errors)
        warnings.extend(wd_warnings)

    if boot_dir is not None:
        boot_dir_path = _resolve_against_repo(boot_dir, resolved_repo)
        boot_dir_section, bd_errors, bd_warnings = _verify_staged_boot_dir(
            boot_dir_path,
            require_boot_options=require_boot_options,
        )
        errors.extend(bd_errors)
        warnings.extend(bd_warnings)

    if shaders_dir is not None:
        shaders_dir_path = _resolve_against_repo(shaders_dir, resolved_repo)
        shaders_dir_section, sd_errors, sd_warnings = _verify_staged_shaders_dir(
            shaders_dir_path,
            shader_pack_name=shader_pack_name,
            require_sidecar=require_sidecar,
            require_sidecar_sync=require_sidecar_sync,
            require_reflection_manifest=require_reflection_manifest,
        )
        errors.extend(sd_errors)
        warnings.extend(sd_warnings)

    if web_dist is None and boot_dir is None and shaders_dir is None:
        errors.append("no staged directories provided to verify")

    ok = len(errors) == 0
    return sanitize_value(
        {
            "checked": True,
            "ok": ok,
            "status": "pass" if ok else "fail",
            "web_dist": web_dist_section,
            "boot_dir": boot_dir_section,
            "shaders_dir": shaders_dir_section,
            "errors": errors,
            "warnings": warnings,
        }
    )


# ---------------------------------------------------------------------------
# End-to-end staging pipeline
# ---------------------------------------------------------------------------


def stage_browser_boot(
    *,
    program_dir: Path | str | None = None,
    wasm_dir: Path | str | None = None,
    boot_dir: Path | str = DEFAULT_BOOT_DIR_REL,
    web_dist: Path | str = DEFAULT_WEB_DIST_REL,
    shader_pack: Path | str | None = None,
    shaders_dir: Path | str = DEFAULT_SHADERS_DIR_REL,
    shader_pack_name: str = DEFAULT_SHADER_PACK_NAME,
    reflect_sidecar: Path | str | None = None,
    reflection_manifest: Path | str | None = None,
    merge_reflect: Path | str | Sequence[Path | str] | None = None,
    require_fast_boot_exports: bool = False,
    require_sidecar_sync: bool = False,
    verify_staged: bool = False,
    verify_window_marks: bool = False,
    dump_pipelines: bool = False,
    input_script: str | None = None,
    boot_options: Mapping[str, Any] | str | Path | None = None,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Stage allowlisted boot artifacts, Wasm64 runtime, shader packs + sidecars, and boot-options."""
    resolved_repo = Path(repo_root).resolve()
    program_path = (
        _resolve_against_repo(program_dir, resolved_repo) if program_dir is not None else None
    )
    boot_path = _resolve_against_repo(boot_dir, resolved_repo)
    web_dist_path = _resolve_against_repo(web_dist, resolved_repo)
    wasm_path = (
        _resolve_against_repo(wasm_dir, resolved_repo) if wasm_dir is not None else None
    )
    shader_pack_path = (
        _resolve_against_repo(shader_pack, resolved_repo) if shader_pack is not None else None
    )
    shaders_dir_path = _resolve_against_repo(shaders_dir, resolved_repo)
    reflect_sidecar_path = (
        _resolve_against_repo(reflect_sidecar, resolved_repo)
        if reflect_sidecar is not None
        else None
    )
    reflection_manifest_path = (
        _resolve_against_repo(reflection_manifest, resolved_repo)
        if reflection_manifest is not None
        else None
    )
    merge_reflect_paths = _normalize_merge_reflect_paths(merge_reflect, resolved_repo)

    errors: list[str] = []
    warnings: list[str] = []

    boot_safe, boot_safety_err = validate_boot_dir_safety(boot_path, resolved_repo)
    web_dist_safe, web_dist_safety_err = validate_web_dist_safety(
        web_dist_path, resolved_repo
    )
    shaders_safe, shaders_safety_err = validate_shaders_dir_safety(
        shaders_dir_path, resolved_repo
    )

    boot_section: dict[str, Any] = {
        "status": "pending",
        "discovered_count": 0,
        "copied_count": 0,
        "planned_count": 0,
        "unchanged_count": 0,
        "skipped_non_allowlisted": 0,
        "skipped_escaping_symlinks": 0,
        "total_bytes": 0,
        "artifacts": [],
    }
    wasm_section: dict[str, Any] = _empty_wasm_section()
    shaders_section: dict[str, Any] = _empty_shaders_section()
    boot_options_section: dict[str, Any] = _empty_boot_options_section()
    verification_section: dict[str, Any] = _empty_verification_section()

    wants_boot_options = bool(
        verify_window_marks
        or dump_pipelines
        or input_script is not None
        or boot_options is not None
    )

    if not boot_safe:
        boot_section["status"] = "refused"
        if wants_boot_options:
            boot_options_section["status"] = "refused"
        errors.append(boot_safety_err or "refused unsafe --boot-dir")
    if not web_dist_safe:
        wasm_section["status"] = "refused"
        errors.append(web_dist_safety_err or "refused unsafe --web-dist")
    if not shaders_safe:
        shaders_section["status"] = "refused"
        errors.append(shaders_safety_err or "refused unsafe --shaders-dir")

    # Early exit if any safety guard tripped — never touch filesystem.
    if not boot_safe or not web_dist_safe or not shaders_safe:
        return sanitize_value(
            {
                "schema_version": SCHEMA_VERSION,
                "ok": False,
                "dry_run": dry_run,
                "paths": {
                    "program_dir": repo_relative_label(program_path, resolved_repo),
                    "boot_dir": repo_relative_label(boot_path, resolved_repo),
                    "wasm_dir": repo_relative_label(wasm_path, resolved_repo),
                    "web_dist": repo_relative_label(web_dist_path, resolved_repo),
                    "shader_pack": repo_relative_label(shader_pack_path, resolved_repo),
                    "shaders_dir": repo_relative_label(shaders_dir_path, resolved_repo),
                    "reflect_sidecar": repo_relative_label(reflect_sidecar_path, resolved_repo),
                    "reflection_manifest": repo_relative_label(
                        reflection_manifest_path, resolved_repo
                    ),
                    "merge_reflect": [
                        repo_relative_label(p, resolved_repo) for p in merge_reflect_paths
                    ],
                },
                "safety": {
                    "boot_dir_safe": boot_safe,
                    "web_dist_safe": web_dist_safe,
                    "shaders_dir_safe": shaders_safe,
                },
                "boot": boot_section,
                "wasm": wasm_section,
                "shaders": shaders_section,
                "boot_options": boot_options_section,
                "verification": verification_section,
                "sidecar_staged": False,
                "sidecar_bytes": 0,
                "boot_options_staged": False,
                "errors": errors,
                "warnings": warnings,
            }
        )

    # ---- Stage 1: Allowlisted boot artifacts (--program-dir -> --boot-dir) ----
    if program_path is None:
        boot_section["status"] = "skipped"
    elif not program_path.is_dir():
        boot_section["status"] = "error"
        errors.append(
            f"program-dir does not exist or is not a directory: "
            f"{repo_relative_label(program_path, resolved_repo)}"
        )
    else:
        resolved_program = program_path.resolve()
        resolved_boot = boot_path.resolve()
        if resolved_boot == resolved_program or resolved_boot.is_relative_to(resolved_program):
            boot_section["status"] = "refused"
            errors.append(
                "refusing to stage boot artifacts into program-dir itself or a subdirectory of it"
            )
        else:
            discovery = discover_boot_artifacts(program_path)
            boot_section["discovered_count"] = len(discovery.artifacts)
            boot_section["skipped_non_allowlisted"] = discovery.skipped_non_allowlisted
            boot_section["skipped_escaping_symlinks"] = discovery.skipped_escaping_symlinks

            if discovery.skipped_escaping_symlinks > 0:
                warnings.append(
                    f"skipped {discovery.skipped_escaping_symlinks} symlink(s) escaping program-dir"
                )

            if not discovery.artifacts:
                boot_section["status"] = "empty"
                warnings.append(
                    "no allowlisted boot artifacts (program-manifest.json, build-manifest.json, *.pgim) "
                    "found in program-dir"
                )
            else:
                copied = 0
                planned = 0
                unchanged = 0
                total_bytes = 0
                artifact_entries: list[dict[str, Any]] = []
                boot_failed = False

                for artifact in discovery.artifacts:
                    dst_file = (boot_path / artifact.rel_posix)
                    # Guard against destination path traversal even if rel_posix were strange.
                    if not dst_file.resolve().is_relative_to(resolved_boot):
                        boot_failed = True
                        errors.append(
                            f"refused boot artifact escaping boot-dir: {artifact.rel_posix}"
                        )
                        continue
                    try:
                        plan = plan_file_stage(artifact.source_path, dst_file)
                    except (OSError, StageBrowserBootError) as exc:
                        boot_failed = True
                        errors.append(
                            f"failed to inspect boot artifact {artifact.rel_posix}: "
                            f"{sanitize_text(str(exc))}"
                        )
                        continue

                    total_bytes += plan.size_bytes
                    if not plan.needs_copy:
                        unchanged += 1
                        action = "unchanged"
                    elif dry_run:
                        planned += 1
                        action = "plan_copy"
                    else:
                        try:
                            atomic_copy_file(artifact.source_path, dst_file)
                        except OSError as exc:
                            boot_failed = True
                            errors.append(
                                f"failed to stage boot artifact {artifact.rel_posix}: "
                                f"{sanitize_text(str(exc))}"
                            )
                            continue
                        copied += 1
                        action = "copied"

                    artifact_entries.append(
                        {
                            "rel_path": artifact.rel_posix,
                            "kind": artifact.kind,
                            "action": action,
                            "reason": plan.reason,
                            "size_bytes": plan.size_bytes,
                            "sha256": plan.sha256,
                        }
                    )

                boot_section["copied_count"] = copied
                boot_section["planned_count"] = planned
                boot_section["unchanged_count"] = unchanged
                boot_section["total_bytes"] = total_bytes
                boot_section["artifacts"] = artifact_entries

                if boot_failed:
                    boot_section["status"] = "error"
                elif dry_run:
                    boot_section["status"] = "planned" if planned > 0 else "unchanged"
                elif copied > 0:
                    boot_section["status"] = "staged"
                else:
                    boot_section["status"] = "unchanged"

    # ---- Stage 2: Validated Wasm64 runtime + side modules (--wasm-dir -> --web-dist) ----
    if wasm_path is not None:
        wasm_section, wasm_errors, wasm_warnings = stage_wasm_runtime(
            wasm_dir=wasm_path,
            web_dist=web_dist_path,
            repo_root=resolved_repo,
            dry_run=dry_run,
            require_fast_boot_exports=require_fast_boot_exports,
        )
        errors.extend(wasm_errors)
        warnings.extend(wasm_warnings)

    # ---- Stage 3: Shader pack + adjacent/overridden/merged .reflect.json sidecar ----
    if (
        shader_pack_path is not None
        or reflect_sidecar_path is not None
        or reflection_manifest_path is not None
        or merge_reflect_paths
    ):
        shaders_section, shader_errors, shader_warnings = stage_shader_pack(
            shader_pack=shader_pack_path,
            shaders_dir=shaders_dir_path,
            shader_pack_name=shader_pack_name,
            reflect_sidecar=reflect_sidecar_path,
            reflection_manifest=reflection_manifest_path,
            merge_reflect=merge_reflect_paths,
            require_sidecar_sync=require_sidecar_sync,
            repo_root=resolved_repo,
            dry_run=dry_run,
        )
        errors.extend(shader_errors)
        warnings.extend(shader_warnings)

    # ---- Stage 4: Optional validated boot-options.json (--boot-dir/boot-options.json) ----
    if wants_boot_options:
        if boot_section["status"] == "refused":
            boot_options_section["status"] = "refused"
        else:
            boot_options_section, opts_errors, opts_warnings = stage_boot_options(
                boot_dir=boot_path,
                boot_options=boot_options,
                verify_window_marks=verify_window_marks,
                dump_pipelines=dump_pipelines,
                input_script=input_script,
                repo_root=resolved_repo,
                dry_run=dry_run,
            )
            errors.extend(opts_errors)
            warnings.extend(opts_warnings)

    # ---- Stage 5: Optional post-stage bundle verification (--verify-staged) ----
    if verify_staged:
        if dry_run:
            verification_section["status"] = "skipped_dry_run"
        elif errors:
            verification_section["status"] = "skipped_stage_failed"
        else:
            verify_web: Path | None = (
                web_dist_path
                if (wasm_path is not None or web_dist_path.is_dir())
                else None
            )
            verify_boot: Path | None = (
                boot_path
                if (program_path is not None or wants_boot_options or boot_path.is_dir())
                else None
            )
            verify_shaders: Path | None = (
                shaders_dir_path
                if (
                    shader_pack_path is not None
                    or reflect_sidecar_path is not None
                    or reflection_manifest_path is not None
                    or bool(merge_reflect_paths)
                    or shaders_dir_path.is_dir()
                )
                else None
            )
            verification_section = verify_staged_bundle(
                web_dist=verify_web,
                boot_dir=verify_boot,
                shaders_dir=verify_shaders,
                shader_pack_name=shader_pack_name if shader_pack_path is not None else None,
                require_fast_boot_exports=True,
                require_boot_options=wants_boot_options,
                require_sidecar_sync=require_sidecar_sync,
                repo_root=resolved_repo,
            )
            errors.extend(verification_section.get("errors") or [])
            warnings.extend(verification_section.get("warnings") or [])

    ok = len(errors) == 0
    return sanitize_value(
        {
            "schema_version": SCHEMA_VERSION,
            "ok": ok,
            "dry_run": dry_run,
            "paths": {
                "program_dir": repo_relative_label(program_path, resolved_repo),
                "boot_dir": repo_relative_label(boot_path, resolved_repo),
                "wasm_dir": repo_relative_label(wasm_path, resolved_repo),
                "web_dist": repo_relative_label(web_dist_path, resolved_repo),
                "shader_pack": repo_relative_label(shader_pack_path, resolved_repo),
                "shaders_dir": repo_relative_label(shaders_dir_path, resolved_repo),
                "reflect_sidecar": repo_relative_label(reflect_sidecar_path, resolved_repo),
                "reflection_manifest": repo_relative_label(
                    reflection_manifest_path, resolved_repo
                ),
                "merge_reflect": [
                    repo_relative_label(p, resolved_repo) for p in merge_reflect_paths
                ],
            },
            "safety": {
                "boot_dir_safe": boot_safe,
                "web_dist_safe": web_dist_safe,
                "shaders_dir_safe": shaders_safe,
            },
            "boot": boot_section,
            "wasm": wasm_section,
            "shaders": shaders_section,
            "boot_options": boot_options_section,
            "verification": verification_section,
            "sidecar_staged": bool(shaders_section.get("sidecar_staged", False)),
            "sidecar_bytes": int(shaders_section.get("sidecar_bytes", 0)),
            "boot_options_staged": bool(boot_options_section.get("staged", False)),
            "errors": errors,
            "warnings": warnings,
        }
    )


# ---------------------------------------------------------------------------
# Output formatting & CLI
# ---------------------------------------------------------------------------


def format_report_text(report: Mapping[str, Any]) -> str:
    """Render a concise, strictly sanitized human-readable staging summary."""
    status_label = "PASS" if report.get("ok") else "FAIL"
    mode = "dry-run" if report.get("dry_run") else "stage"
    paths = report.get("paths") or {}
    boot = report.get("boot") or {}
    wasm = report.get("wasm") or {}
    shaders = report.get("shaders") or {}
    boot_options = report.get("boot_options") or {}
    verification = report.get("verification") or {}

    lines: list[str] = [
        f"Browser Real-Boot Staging [{status_label}] mode={mode}",
        f"  Boot: {paths.get('program_dir')} -> {paths.get('boot_dir')} "
        f"[{str(boot.get('status', 'unknown')).upper()}] "
        f"(discovered={boot.get('discovered_count', 0)}, "
        f"copied={boot.get('copied_count', 0)}, "
        f"planned={boot.get('planned_count', 0)}, "
        f"unchanged={boot.get('unchanged_count', 0)}, "
        f"skipped_other={boot.get('skipped_non_allowlisted', 0)}, "
        f"skipped_symlinks={boot.get('skipped_escaping_symlinks', 0)})",
    ]

    artifacts = boot.get("artifacts") or []
    for item in artifacts:
        lines.append(
            f"    [{str(item.get('action', '')).upper()}] "
            f"{item.get('rel_path')} ({item.get('kind')}, {item.get('size_bytes')} bytes, "
            f"{item.get('reason')})"
        )

    wasm_status = str(wasm.get("status", "skipped")).upper()
    wasm_src = paths.get("wasm_dir") or "<none>"
    side_modules = wasm.get("side_modules") or []
    wasm_skipped_symlinks = int(wasm.get("skipped_escaping_symlinks", 0) or 0)
    wasm_meta_parts: list[str] = []
    if side_modules:
        wasm_meta_parts.append(f"side_modules={len(side_modules)}")
    if wasm_skipped_symlinks > 0:
        wasm_meta_parts.append(f"skipped_symlinks={wasm_skipped_symlinks}")
    wasm_meta = f" ({', '.join(wasm_meta_parts)})" if wasm_meta_parts else ""
    lines.append(
        f"  Wasm64: {wasm_src} -> {paths.get('web_dist')} [{wasm_status}]{wasm_meta}"
    )
    validation = wasm.get("validation")
    if isinstance(validation, Mapping):
        val_tag = "PASS" if validation.get("ok") else "FAIL"
        lines.append(
            f"    Validation [{val_tag}]: "
            f"memory64={'YES' if validation.get('is_memory64') else 'NO'}, "
            f"shared={'YES' if validation.get('is_shared_memory') else 'NO'}, "
            f"host_imports={validation.get('host_imports_valid', 0)}/"
            f"{validation.get('host_imports_total', 0)}, "
            f"audio_ring={'YES' if validation.get('has_audio_ring_export') else 'NO'}, "
            f"input_state={'YES' if validation.get('has_input_state_export') else 'NO'}"
        )
    files_list = wasm.get("files") or []
    listed_names: set[str] = set()
    for item in files_list:
        item_name = str(item.get("name", ""))
        listed_names.add(item_name)
        lines.append(
            f"    [{str(item.get('action', '')).upper()}] "
            f"{item_name} ({item.get('size_bytes')} bytes, {item.get('reason')})"
        )
    for item in side_modules:
        item_name = str(item.get("name", ""))
        if item_name in listed_names:
            continue
        lines.append(
            f"    [{str(item.get('action', '')).upper()}] "
            f"{item_name} ({item.get('size_bytes')} bytes, {item.get('reason')})"
        )

    shaders_status = str(shaders.get("status", "skipped")).upper()
    shaders_src = paths.get("shader_pack") or "<none>"
    sidecar_staged = bool(report.get("sidecar_staged", shaders.get("sidecar_staged", False)))
    sidecar_bytes = int(report.get("sidecar_bytes", shaders.get("sidecar_bytes", 0)) or 0)
    lines.append(
        f"  Shaders: {shaders_src} -> {paths.get('shaders_dir') or '<none>'} "
        f"[{shaders_status}] "
        f"(sidecar_staged={'true' if sidecar_staged else 'false'}, "
        f"sidecar_bytes={sidecar_bytes})"
    )
    wgpk_val = shaders.get("wgpk_validation")
    if isinstance(wgpk_val, Mapping):
        wgpk_tag = "PASS" if wgpk_val.get("ok") else "FAIL"
        ver = wgpk_val.get("version")
        ver_label = f"v{ver}" if ver is not None else "unknown"
        lines.append(
            f"    WGPK [{wgpk_tag}]: version={ver_label}, "
            f"entries={wgpk_val.get('entry_count')}, "
            f"bounds={'OK' if wgpk_val.get('bounds_ok') else 'FAIL'}"
        )
    pack_item = shaders.get("pack")
    if isinstance(pack_item, Mapping):
        lines.append(
            f"    [{str(pack_item.get('action', '')).upper()}] "
            f"{pack_item.get('name')} ({pack_item.get('size_bytes')} bytes, "
            f"{pack_item.get('reason')})"
        )
    sidecar_item = shaders.get("sidecar")
    if isinstance(sidecar_item, Mapping):
        merge_suffix = ""
        merge_stats = sidecar_item.get("merge_stats")
        if isinstance(merge_stats, Mapping):
            merge_suffix = (
                f", merged=+{merge_stats.get('added_stages', 0)}"
                f"/~{merge_stats.get('updated_stages', 0)}"
                f"/={merge_stats.get('preserved_stages', 0)}"
            )
        lines.append(
            f"    [{str(sidecar_item.get('action', '')).upper()}] "
            f"{sidecar_item.get('name')} ({sidecar_item.get('size_bytes')} bytes, "
            f"{sidecar_item.get('reason')}, stages={sidecar_item.get('stage_count', 0)}, "
            f"source={sidecar_item.get('source')}{merge_suffix})"
        )
    sidecar_sync = shaders.get("sidecar_sync")
    if isinstance(sidecar_sync, Mapping):
        sync_tag = "OK" if sidecar_sync.get("synchronized") else "DESYNC"
        lines.append(
            f"    SidecarSync [{sync_tag}]: "
            f"matched={sidecar_sync.get('matched_hash_count', 0)}/"
            f"{sidecar_sync.get('pack_entry_count', 0)}, "
            f"stage_mismatches={sidecar_sync.get('stage_mismatch_count', 0)}"
        )
    manifest_item = shaders.get("reflection_manifest")
    if isinstance(manifest_item, Mapping) and manifest_item.get("action"):
        lines.append(
            f"    [{str(manifest_item.get('action', '')).upper()}] "
            f"{manifest_item.get('name', REFLECTION_MANIFEST_NAME)} "
            f"({manifest_item.get('size_bytes', 0)} bytes, "
            f"{manifest_item.get('reason')}, shaders={manifest_item.get('shader_count', 0)}, "
            f"source={manifest_item.get('source')})"
        )

    boot_opts_status = str(boot_options.get("status", "skipped")).upper()
    boot_opts_staged = bool(
        report.get("boot_options_staged", boot_options.get("staged", False))
    )
    lines.append(
        f"  BootOptions: {paths.get('boot_dir')}/{BOOT_OPTIONS_NAME} "
        f"[{boot_opts_status}] "
        f"(boot_options_staged={'true' if boot_opts_staged else 'false'})"
    )
    if boot_options.get("action"):
        opts_detail_parts: list[str] = [
            f"{boot_options.get('size_bytes', 0)} bytes",
            str(boot_options.get("reason")),
        ]
        opts_map = boot_options.get("options")
        if isinstance(opts_map, Mapping):
            if opts_map.get("dump_pipelines_dir"):
                opts_detail_parts.append(
                    f"dump_pipelines_dir={opts_map.get('dump_pipelines_dir')}"
                )
            if opts_map.get("input_script"):
                opts_detail_parts.append(f"input_script={opts_map.get('input_script')}")
        lines.append(
            f"    [{str(boot_options.get('action', '')).upper()}] "
            f"{boot_options.get('filename', BOOT_OPTIONS_NAME)} "
            f"({', '.join(opts_detail_parts)})"
        )

    if isinstance(verification, Mapping) and verification.get("checked"):
        ver_status = str(verification.get("status", "fail")).upper()
        lines.append(f"  Verification [{ver_status}]")
        v_web = verification.get("web_dist")
        if isinstance(v_web, Mapping):
            v_wasm = v_web.get("runtime_wasm") or {}
            lines.append(
                f"    web_dist [{str(v_web.get('status', 'fail')).upper()}]: "
                f"fast_boot_exports={'OK' if v_wasm.get('fast_boot_exports_ok') else 'MISSING'}, "
                f"audio_ring={'YES' if v_wasm.get('has_audio_ring_export') else 'NO'}, "
                f"input_state={'YES' if v_wasm.get('has_input_state_export') else 'NO'}"
            )
        v_boot = verification.get("boot_dir")
        if isinstance(v_boot, Mapping):
            lines.append(
                f"    boot_dir [{str(v_boot.get('status', 'fail')).upper()}]: "
                f"pgims={v_boot.get('valid_pgim_count', 0)}/{v_boot.get('pgim_count', 0)}, "
                f"guest_pgim={'YES' if v_boot.get('has_guest_pgim') else 'NO'}"
            )
        v_shaders = verification.get("shaders_dir")
        if isinstance(v_shaders, Mapping):
            lines.append(
                f"    shaders_dir [{str(v_shaders.get('status', 'fail')).upper()}]: "
                f"packs={v_shaders.get('valid_pack_count', 0)}/{v_shaders.get('pack_count', 0)}"
            )

    warnings = report.get("warnings") or []
    if warnings:
        lines.append("  Warnings:")
        for warn in warnings:
            lines.append(f"    - {warn}")

    errors = report.get("errors") or []
    if errors:
        lines.append("  Errors:")
        for err in errors:
            lines.append(f"    - {err}")

    return sanitize_text("\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely stage allowlisted local recomp boot artifacts (.pgim + manifests), "
            "validated Wasm64 runtime.{wasm,js} plus sibling *.wasm side modules, "
            "WGPK v1/v2 .wgsl.pack shader bundles with .reflect.json sidecars and "
            "reflection-manifest.json, and optional validated boot-options.json for "
            "server/serve_local.py."
        )
    )
    parser.add_argument(
        "--program-dir",
        type=Path,
        default=None,
        help=(
            "Source recomp program directory (default: work/recomp_program unless "
            "staging shaders/boot-options standalone without it)."
        ),
    )
    parser.add_argument(
        "--wasm-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing runtime.wasm, runtime.js, optional "
            "guest_manifest.json, and any sibling *.wasm side modules (e.g. work/wasm_guest_dist)."
        ),
    )
    parser.add_argument(
        "--boot-dir",
        type=Path,
        default=DEFAULT_BOOT_DIR_REL,
        help="Destination boot directory served at /boot/ (default: work/recomp_build).",
    )
    parser.add_argument(
        "--web-dist",
        type=Path,
        default=DEFAULT_WEB_DIST_REL,
        help="Destination Wasm64 web dist directory (default: web/dist).",
    )
    parser.add_argument(
        "--shader-pack",
        type=Path,
        default=None,
        help=(
            "Optional source .wgsl.pack file (WGPK v1 or v2) to validate and stage into "
            "--shaders-dir (auto-stages adjacent <pack>.reflect.json and reflection-manifest.json when present)."
        ),
    )
    parser.add_argument(
        "--shaders-dir",
        type=Path,
        default=DEFAULT_SHADERS_DIR_REL,
        help=(
            "Destination directory for staged .wgsl.pack and .reflect.json sidecars "
            "(default: local-data/shaders)."
        ),
    )
    parser.add_argument(
        "--shader-pack-name",
        type=str,
        default=DEFAULT_SHADER_PACK_NAME,
        help=(
            "Destination filename inside --shaders-dir for --shader-pack "
            "(default: splash.wgsl.pack)."
        ),
    )
    parser.add_argument(
        "--reflect-sidecar",
        type=Path,
        default=None,
        help=(
            "Optional override path for the .reflect.json sidecar staged alongside "
            "--shader-pack."
        ),
    )
    parser.add_argument(
        "--reflection-manifest",
        type=Path,
        default=None,
        help=(
            "Optional override path for reflection-manifest.json validated and staged into "
            "--shaders-dir."
        ),
    )
    parser.add_argument(
        "--merge-reflect",
        action="append",
        type=Path,
        default=None,
        metavar="EXTRA_REFLECT_JSON",
        help=(
            "Merge additional playgta6-stage-reflection v1 sidecar stages into the "
            "staged <pack>.reflect.json without losing existing stages (may be passed "
            "multiple times)."
        ),
    )
    parser.add_argument(
        "--require-fast-boot-exports",
        action="store_true",
        help=(
            "Fail staging unless runtime.wasm exports both _playgta6_audio_ring and "
            "_playgta6_get_input_state_ptr."
        ),
    )
    parser.add_argument(
        "--require-sidecar-sync",
        action="store_true",
        help=(
            "Fail staging if a staged <pack>.reflect.json sidecar is missing any shader "
            "hashes present in the WGPK pack."
        ),
    )
    parser.add_argument(
        "--verify-staged",
        action="store_true",
        help=(
            "Run post-stage verification across staged web-dist, boot-dir, and shaders-dir "
            "(checks fast-boot exports, WGPK v1/v2 packs, sidecar sync, reflection-manifest.json, "
            "program-manifest.json, .pgim magic, and boot-options.json)."
        ),
    )
    parser.add_argument(
        "--verify-window-marks",
        action="store_true",
        help=(
            "Stage validated boot-options.json into --boot-dir with "
            "verify_window_marks=true."
        ),
    )
    parser.add_argument(
        "--dump-pipelines",
        action="store_true",
        help=(
            f'Stage validated boot-options.json into --boot-dir with '
            f'dump_pipelines_dir="{DEFAULT_DUMP_PIPELINES_DIR}".'
        ),
    )
    parser.add_argument(
        "--input-script",
        type=str,
        default=None,
        help=(
            "Optional validated input-script preset to record in boot-options.json "
            f"(allowed: {', '.join(ALLOWED_INPUT_SCRIPT_PRESETS.values())})."
        ),
    )
    parser.add_argument(
        "--boot-options-json",
        "--boot-options",
        dest="boot_options_json",
        type=str,
        default=None,
        help=(
            "Optional inline JSON object or path to a JSON file to validate and stage "
            "as boot-options.json into --boot-dir."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root used for tracked-directory safety checks (default: parent of tools/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report planned copies without creating or modifying files.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit strictly sanitized machine-readable JSON.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    program_dir = args.program_dir
    if program_dir is None:
        resolved_repo = Path(args.repo_root).resolve()
        default_program = _resolve_against_repo(DEFAULT_PROGRAM_DIR_REL, resolved_repo)
        standalone_verify = bool(
            args.verify_staged
            and args.wasm_dir is None
            and args.shader_pack is None
            and args.reflect_sidecar is None
            and args.reflection_manifest is None
            and not bool(args.merge_reflect)
            and not args.verify_window_marks
            and not args.dump_pipelines
            and args.input_script is None
            and args.boot_options_json is None
        )
        has_other_target = bool(
            args.shader_pack is not None
            or args.reflect_sidecar is not None
            or args.reflection_manifest is not None
            or bool(args.merge_reflect)
            or args.verify_window_marks
            or args.dump_pipelines
            or args.input_script is not None
            or args.boot_options_json is not None
            or standalone_verify
        )
        if not standalone_verify and (not has_other_target or default_program.is_dir()):
            program_dir = DEFAULT_PROGRAM_DIR_REL

    report = stage_browser_boot(
        program_dir=program_dir,
        wasm_dir=args.wasm_dir,
        boot_dir=args.boot_dir,
        web_dist=args.web_dist,
        shader_pack=args.shader_pack,
        shaders_dir=args.shaders_dir,
        shader_pack_name=args.shader_pack_name,
        reflect_sidecar=args.reflect_sidecar,
        reflection_manifest=args.reflection_manifest,
        merge_reflect=args.merge_reflect,
        require_fast_boot_exports=args.require_fast_boot_exports,
        require_sidecar_sync=args.require_sidecar_sync,
        verify_staged=args.verify_staged,
        verify_window_marks=args.verify_window_marks,
        dump_pipelines=args.dump_pipelines,
        input_script=args.input_script,
        boot_options=args.boot_options_json,
        repo_root=args.repo_root,
        dry_run=args.dry_run,
    )

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
