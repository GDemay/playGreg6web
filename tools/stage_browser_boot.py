#!/usr/bin/env python3
"""Safe local browser real-boot artifact staging CLI (Track 5 / W51, W66, W119, T46).

Stages allowlisted local recompilation boot artifacts (`program-manifest.json`,
`build-manifest.json`, nested `*.pgim`) from `--program-dir` into `--boot-dir`
(served at `/boot/` by `server/serve_local.py`), validates + atomically stages
`runtime.wasm`, `runtime.js`, and any sibling top-level `*.wasm` Emscripten
side modules from `--wasm-dir` into `--web-dist`, atomically stages `.wgsl.pack`
shader bundles plus adjacent/overridden/merged `<pack>.reflect.json` sidecars
into `--shaders-dir`, and optionally writes a schema-validated `boot-options.json`
into `--boot-dir`.

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
  shared memory, 5/5 `env.host_*` ABI imports), validates `.reflect.json` JSON
  structure, and schema-validates `boot-options.json` BEFORE atomically
  replacing files (`os.replace` via same-directory temp file).
- Strictly sanitized JSON and text output: zero Windows/POSIX/UNC host paths,
  zero usernames, and zero raw bytes.
- Pure Python 3.12 standard library only.

Usage:
    python tools/stage_browser_boot.py --program-dir work/recomp_program
        [--wasm-dir work/wasm_guest_dist] [--boot-dir work/recomp_build]
        [--web-dist web/dist] [--shader-pack work/splash.wgsl.pack]
        [--shaders-dir local-data/shaders] [--reflect-sidecar <path>]
        [--merge-reflect <extra_reflect.json> ...] [--verify-window-marks]
        [--dump-pipelines] [--input-script <preset>]
        [--boot-options-json <json-or-file>] [--dry-run] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
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
DEFAULT_SHADER_PACK_NAME = "splash.wgsl.pack"
REFLECT_SIDECAR_SUFFIX = ".reflect.json"
REFLECT_SIDECAR_SCHEMA = "playgta6-stage-reflection"
REFLECT_SIDECAR_VERSION = 1
REFLECT_SIDECAR_STAGES: frozenset[str] = frozenset({"vertex", "fragment", "compute"})
BOOT_OPTIONS_NAME = "boot-options.json"
DEFAULT_DUMP_PIPELINES_DIR = "/pg6_dump"
WGPK_MAGIC = b"WGPK"

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
    for candidate in (
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
    ):
        cleaned = candidate.strip()
        if len(cleaned) >= 3 and cleaned not in names:
            names.append(cleaned)
    try:
        home_name = Path.home().name.strip()
        if len(home_name) >= 3 and home_name not in names:
            names.append(home_name)
    except (OSError, RuntimeError):
        pass
    return names


def sanitize_text(raw: str | None) -> str:
    """Strip Windows/POSIX/UNC host paths (both `\\` and `/`) and active usernames."""
    if raw is None:
        return ""
    text = str(raw)
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


def validate_wasm_candidate(wasm_path: Path) -> dict[str, Any]:
    """Validate `runtime.wasm` for memory64, shared memory, and 5/5 `env.host_*` ABI imports."""
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
            "errors": [sanitize_text(str(exc))],
            "warnings": [],
        }

    host_valid = sum(1 for item in report.host_imports if item.ok)
    host_total = len(report.host_imports)
    abi_ok = bool(
        report.ok
        and report.is_memory64
        and report.is_shared_memory
        and report.all_host_imports_ok
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
        "errors": [sanitize_text(err) for err in report.errors],
        "warnings": [sanitize_text(warn) for warn in report.warnings],
    }


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
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate `runtime.wasm` and atomically stage runtime + sibling `*.wasm` side modules."""
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
    validation = validate_wasm_candidate(resolved_wasm)
    wasm_section["validation"] = validation

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
# Shader pack (.wgsl.pack) & .reflect.json sidecar staging (W119)
# ---------------------------------------------------------------------------


def _empty_shaders_section() -> dict[str, Any]:
    return {
        "status": "skipped",
        "pack": None,
        "sidecar": None,
        "sidecar_staged": False,
        "sidecar_bytes": 0,
        "copied_count": 0,
        "planned_count": 0,
        "unchanged_count": 0,
    }


def validate_reflect_sidecar(sidecar_path: Path) -> tuple[bool, dict[str, Any], list[str]]:
    """Validate a `<pack>.reflect.json` stage-reflection sidecar file."""
    label = basename_label(sidecar_path)
    try:
        raw_text = sidecar_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return (
            False,
            {},
            [sanitize_text(f"failed to read reflection sidecar {label}: {exc}")],
        )

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return (
            False,
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
    if not isinstance(stages_val, dict):
        errors.append(
            f"invalid reflection sidecar {label}: missing or non-object 'stages' map"
        )
    else:
        for stage_key, stage_entry in stages_val.items():
            if not isinstance(stage_key, str) or not isinstance(stage_entry, dict):
                errors.append(
                    f"invalid reflection sidecar {label}: every entry in 'stages' must be a JSON object"
                )
                break

    if errors:
        return False, {}, sanitize_value(errors)

    summary = {
        "schema": sanitize_text(schema_val) if isinstance(schema_val, str) else None,
        "version": int(version_val) if isinstance(version_val, int) else None,
        "stage_count": len(stages_val),
    }
    return True, summary, []


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
    merge_reflect: Path | str | Sequence[Path | str] | None = None,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    dry_run: bool = False,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Atomically stage a `.wgsl.pack` bundle and its adjacent, overridden, or merged `.reflect.json` sidecar."""
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

    if shader_pack is None and not merge_paths_raw:
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

        try:
            pack_size = int(src_pack.stat().st_size)
            with src_pack.open("rb") as fh:
                pack_magic = fh.read(len(WGPK_MAGIC))
        except OSError as exc:
            shaders_section["status"] = "error"
            errors.append(
                f"failed to read shader-pack {basename_label(src_pack_raw)}: {sanitize_text(str(exc))}"
            )
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

        if pack_size == 0:
            shaders_section["status"] = "refused"
            errors.append(f"refusing empty shader-pack: {basename_label(src_pack_raw)}")
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

        if pack_magic != WGPK_MAGIC:
            warnings.append(
                f"shader-pack {basename_label(src_pack_raw)} does not start with WGPK magic"
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

    resolved_shaders = shaders_dir_path.resolve()
    dst_pack = shaders_dir_path / shader_pack_name
    dst_sidecar = Path(str(dst_pack) + REFLECT_SIDECAR_SUFFIX)
    sidecar_dest_name = dst_sidecar.name

    # Validate destination containment before doing any staging work.
    dest_checks: list[tuple[str, Path]] = []
    if src_pack is not None:
        dest_checks.append((shader_pack_name, dst_pack))
    if sidecar_resolved is not None or merge_paths_raw:
        dest_checks.append((sidecar_dest_name, dst_sidecar))
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
            shaders_section["pack"] = {
                "name": sanitize_text(shader_pack_name),
                "action": pack_action,
                "reason": pack_plan.reason,
                "size_bytes": pack_plan.size_bytes,
                "sha256": pack_plan.sha256,
            }

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

    # Validate sidecar JSON BEFORE staging either the pack or the sidecar so
    # invalid sidecars never clobber existing staged files.
    if sidecar_resolved is not None:
        sidecar_ok, sidecar_meta, sidecar_errors = validate_reflect_sidecar(sidecar_resolved)
        if not sidecar_ok:
            shaders_section["status"] = "refused"
            errors.extend(sidecar_errors)
            return (
                sanitize_value(shaders_section),
                sanitize_value(errors),
                sanitize_value(warnings),
            )

    assert src_pack is not None
    candidates: list[tuple[str, str, Path, Path]] = [
        ("pack", shader_pack_name, src_pack, dst_pack),
    ]
    if sidecar_resolved is not None:
        candidates.append(("sidecar", sidecar_dest_name, sidecar_resolved, dst_sidecar))

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
            shaders_section["pack"] = entry
        else:
            entry["source"] = sidecar_source
            entry["stage_count"] = int(sidecar_meta.get("stage_count", 0))
            shaders_section["sidecar"] = entry
            if action in ("copied", "unchanged"):
                shaders_section["sidecar_staged"] = True
                shaders_section["sidecar_bytes"] = plan.size_bytes

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
    merge_reflect: Path | str | Sequence[Path | str] | None = None,
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
        )
        errors.extend(wasm_errors)
        warnings.extend(wasm_warnings)

    # ---- Stage 3: Shader pack + adjacent/overridden/merged .reflect.json sidecar ----
    if (
        shader_pack_path is not None
        or reflect_sidecar_path is not None
        or merge_reflect_paths
    ):
        shaders_section, shader_errors, shader_warnings = stage_shader_pack(
            shader_pack=shader_pack_path,
            shaders_dir=shaders_dir_path,
            shader_pack_name=shader_pack_name,
            reflect_sidecar=reflect_sidecar_path,
            merge_reflect=merge_reflect_paths,
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
            f"{validation.get('host_imports_total', 0)}"
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
            ".wgsl.pack shader bundles with adjacent .reflect.json sidecars, "
            "and optional validated boot-options.json for server/serve_local.py."
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
            "Optional directory containing runtime.wasm, runtime.js, and any "
            "sibling *.wasm side modules (e.g. work/wasm_guest_dist)."
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
            "Optional source .wgsl.pack file to stage into --shaders-dir "
            "(auto-stages adjacent <pack>.reflect.json when present)."
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
        has_other_target = bool(
            args.shader_pack is not None
            or args.reflect_sidecar is not None
            or bool(args.merge_reflect)
            or args.verify_window_marks
            or args.dump_pipelines
            or args.input_script is not None
            or args.boot_options_json is not None
        )
        if not has_other_target or default_program.is_dir():
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
        merge_reflect=args.merge_reflect,
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
