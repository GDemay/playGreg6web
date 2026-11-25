#!/usr/bin/env python3
"""Sanitized pre-launch browser boot & split-module bundle verifier CLI (Track 5 / W75).

Verifies a staged browser-boot bundle before launching `server/serve_local.py`
and Chrome/Edge:

1. **Wasm64 & V8 1 GiB Cap (`--web-dist`):**
   - Verifies `runtime.js` and `runtime.wasm` exist inside `--web-dist` (regular
     files, no escaping symlinks).
   - Validates `runtime.wasm` via `tools/check_wasm_exports.py` (`memory64`,
     shared memory, 5/5 `env.host_*` ABI imports) and checks
     `size <= 1,073,741,824` bytes (V8's 1 GiB `WebAssembly.instantiate()` cap).
   - Discovers top-level sibling side modules (`guest_part_*.wasm` / `*.wasm`,
     naturally sorted, skipping escaping symlinks), verifies each has valid
     `\\0asm` magic (version 1) and `size <= 1,073,741,824` bytes (`v8_limit_ok`),
     and reports individual and total Wasm byte sizes.
   - When `guest_manifest.json` is present inside `--web-dist`, reports lifted
     guest DLL/module entries and whether `bink2scarlett` / `Bink2Scarlett.dll`
     is present.

2. **PGIM / Program Manifest (`--boot-dir`, optional):**
   - Checks `program-manifest.json` (or `boot-manifest.json` fallback), rejects
     path traversal and escaping symlinks, and verifies every referenced `.pgim`
     file exists inside `--boot-dir` and starts with valid `PGIM` magic.

3. **Shader Pack & Stage-Reflection Sidecar (`--shaders-dir`, optional):**
   - Discovers `*.wgsl.pack` bundles (skipping escaping symlinks), validates
     `WGPK` magic, version 1, entry table bounds, and parses `entry_count` safely
     matching `shaders/src/pack.rs`.
   - When `*.reflect.json` stage-reflection sidecars are present, validates
     `#950` `inline_sampler` entries (`sdwords` must be 4 uint32 integers) and
     reports `inline_sampler_stages`, `inline_sampler_bindings`, and
     `malformed_inline_samplers` (failing `--strict` on any malformed entry).

Security & privacy invariants:
- Pure Python 3.12 standard library only (zero external dependencies).
- Strictly strips all host paths, Windows drive letters, UNC paths, POSIX
  `/home/`/`/Users/` paths, and active OS usernames from both `--json` and text
  reports.
- Exits `0` when all checks pass (or `1` on any failure; `--strict` promotes
  warnings to failures).

Usage:
    python tools/verify_browser_boot_bundle.py --web-dist <dir>
        [--boot-dir <dir>] [--shaders-dir <dir>] [--json] [--strict]
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Constants matching tools/check_wasm_exports.py, tools/stage_browser_boot.py,
# recomp/src/image.rs, web/runtime/pgim_preloader.js, and shaders/src/pack.rs
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1
DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEB_DIST_REL = Path("web") / "dist"

# Chrome/V8 enforces a hard 1 GiB limit on a single WebAssembly module.
V8_MAX_WASM_MODULE_BYTES = 1024 * 1024 * 1024  # 1,073,741,824 bytes

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
BINK2_MODULE_KEY = "bink2scarlett"
WASM_MAGIC = b"\x00asm"
WASM_VERSION = 1
WASM_HEADER_BYTES = 8

PROGRAM_MANIFEST_NAME = "program-manifest.json"
BOOT_MANIFEST_NAME = "boot-manifest.json"
PROGRAM_MANIFEST_SCHEMA = "playgta6-recomp-program"
PGIM_MAGIC = b"PGIM"
PGIM_HEADER_BYTES = 64
PGIM_SECTION_ENTRY_BYTES = 64
PGIM_VERSION = 1

WGPK_MAGIC = b"WGPK"
WGPK_VERSION = 1
WGPK_HEADER_STRUCT = struct.Struct("<4sIII")
WGPK_HEADER_BYTES = WGPK_HEADER_STRUCT.size  # 16 bytes
WGPK_ENTRY_STRUCT = struct.Struct("<QB3sII")
WGPK_ENTRY_BYTES = WGPK_ENTRY_STRUCT.size  # 20 bytes
WGPK_VALID_STAGES: frozenset[int] = frozenset({1, 2, 3})

REFLECT_SIDECAR_SUFFIX = ".reflect.json"
REFLECT_SIDECAR_SCHEMA = "playgta6-stage-reflection"
REFLECT_SIDECAR_VERSION = 1

_SAFE_SIDE_MODULE_NAME_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*\.wasm$", re.IGNORECASE
)
_SAFE_MODULE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_ENCODED_TRAVERSAL_RE = re.compile(r"%(?:2e|2f|5c)", re.IGNORECASE)

_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes|proc|run|srv)/[^\s\"'<>|,;()\[\]]*"
)


# ---------------------------------------------------------------------------
# Cross-platform sanitization
# ---------------------------------------------------------------------------


def _current_usernames() -> list[str]:
    """Return active OS username candidates (length >= 3) for redaction."""
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
    """Return concrete home directory path strings (native + POSIX) for pre-redaction."""
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
    """Strip host paths, drive letters, UNC paths, POSIX roots, and active usernames."""
    if raw is None:
        return ""
    text = str(raw)
    # Redact concrete home directory strings first (handles spaces in home paths).
    for home_str in _home_path_variants():
        if home_str:
            text = re.sub(re.escape(home_str), "<path>", text, flags=re.IGNORECASE)
    # Strip Windows UNC/device backslash paths before normalizing separators.
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
    """Return a repo-relative POSIX label or sanitized basename, never a host path."""
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


def _sanitize_os_error(label: str, exc: OSError) -> str:
    reason = exc.strerror or type(exc).__name__
    return sanitize_text(f"failed to read {label}: {reason}")


# ---------------------------------------------------------------------------
# Sibling tool loader & filesystem helpers
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


def get_file_size(path: Path) -> int:
    """Return the byte size of `path` (isolated helper for clean unit-test mocking)."""
    return int(path.stat().st_size)


def _check_file_within_root(path: Path, root: Path) -> tuple[str, Path | None]:
    """Verify `path` exists as a regular file whose resolved target stays inside `root`.

    Returns `(status, resolved_path)` where `status` is one of:
    - `"ok"`: regular file inside `root`
    - `"missing"`: path does not exist (and is not a dangling symlink)
    - `"escaping_symlink"`: symlink pointing outside `root` or broken symlink
    - `"not_file"`: exists inside `root` but is not a regular file (e.g. directory)
    """
    resolved_root = root.resolve()
    is_link = path.is_symlink()
    exists = path.exists()

    if not exists and not is_link:
        return "missing", None

    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return ("escaping_symlink" if is_link else "missing"), None

    if not resolved.is_relative_to(resolved_root):
        return "escaping_symlink", None

    if not resolved.is_file():
        return "not_file", None

    return "ok", resolved


def _read_header_bytes(path: Path, count: int) -> bytes:
    with path.open("rb") as fh:
        return fh.read(count)


# ---------------------------------------------------------------------------
# Wasm64 & V8 1 GiB cap verification (--web-dist)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SideModuleEntry:
    """Discovered top-level sibling `*.wasm` side module inside `--web-dist`."""

    name: str
    path: Path


@dataclass
class SideModuleScan:
    """Result of scanning `--web-dist` for sibling `*.wasm` side modules."""

    modules: list[SideModuleEntry] = field(default_factory=list)
    skipped_escaping_symlinks: int = 0
    rejected_unsafe_names: list[str] = field(default_factory=list)


def discover_side_modules(web_dist: Path) -> SideModuleScan:
    """Discover top-level sibling `*.wasm` side modules inside `web_dist`.

    Excludes `runtime.wasm`, naturally sorts modules (`guest_part_1.wasm`, ...,
    `guest_part_10.wasm`), skips escaping/broken symlinks, and flags unsafe
    `.wasm` filenames.
    """
    resolved_root = web_dist.resolve()
    scan = SideModuleScan()
    seen_lower: set[str] = set()

    for entry in web_dist.iterdir():
        name = entry.name
        lowered = name.lower()
        if not lowered.endswith(".wasm"):
            continue
        if lowered == RUNTIME_WASM_NAME.lower():
            continue

        if not is_safe_side_module_name(name):
            if entry.is_symlink() or not entry.is_dir():
                scan.rejected_unsafe_names.append(sanitize_text(name))
            continue

        if entry.is_symlink():
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                scan.skipped_escaping_symlinks += 1
                continue
            if not resolved.is_relative_to(resolved_root):
                scan.skipped_escaping_symlinks += 1
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
                scan.skipped_escaping_symlinks += 1
                continue

        if lowered in seen_lower:
            scan.rejected_unsafe_names.append(sanitize_text(name))
            continue
        seen_lower.add(lowered)
        scan.modules.append(SideModuleEntry(name=name, path=resolved))

    scan.modules.sort(key=lambda item: (_natural_sort_key(item.name), item.name))
    scan.rejected_unsafe_names.sort()
    return scan


def inspect_side_module_wasm(
    name: str,
    wasm_path: Path,
    *,
    max_module_bytes: int = V8_MAX_WASM_MODULE_BYTES,
) -> dict[str, Any]:
    """Validate one sibling `*.wasm` side module for `\\0asm` magic and V8 1 GiB cap."""
    errors: list[str] = []
    size_bytes = 0
    v8_limit_ok = False
    magic_ok = False
    wasm_version: int | None = None

    try:
        size_bytes = get_file_size(wasm_path)
    except OSError as exc:
        errors.append(_sanitize_os_error(name, exc))
        return {
            "name": sanitize_text(name),
            "size_bytes": 0,
            "v8_limit_ok": False,
            "magic_ok": False,
            "wasm_version": None,
            "ok": False,
            "errors": errors,
        }

    v8_limit_ok = size_bytes <= max_module_bytes
    if not v8_limit_ok:
        errors.append(
            f"{sanitize_text(name)} size ({size_bytes} bytes) exceeds V8 1 GiB module cap "
            f"({max_module_bytes} bytes)"
        )

    try:
        header = _read_header_bytes(wasm_path, WASM_HEADER_BYTES)
    except OSError as exc:
        errors.append(_sanitize_os_error(name, exc))
        header = b""

    if len(header) < WASM_HEADER_BYTES:
        errors.append(
            f"{sanitize_text(name)} truncated: expected at least {WASM_HEADER_BYTES} Wasm header "
            f"bytes, got {len(header)}"
        )
    else:
        magic = header[:4]
        (version,) = struct.unpack("<I", header[4:8])
        wasm_version = int(version)
        if magic != WASM_MAGIC:
            errors.append(
                f"{sanitize_text(name)} has invalid Wasm magic (expected \\x00asm)"
            )
        elif version != WASM_VERSION:
            errors.append(
                f"{sanitize_text(name)} has unsupported Wasm version {version} (expected {WASM_VERSION})"
            )
        else:
            magic_ok = True

    return {
        "name": sanitize_text(name),
        "size_bytes": size_bytes,
        "v8_limit_ok": v8_limit_ok,
        "magic_ok": magic_ok,
        "wasm_version": wasm_version,
        "ok": len(errors) == 0,
        "errors": errors,
    }


def _is_bink2scarlett_module(name: str) -> bool:
    """Return True if `name` refers to `bink2scarlett` or `Bink2Scarlett.dll` (case-insensitive)."""
    cleaned = name.strip().lower()
    if cleaned.endswith(".dll"):
        cleaned = cleaned[:-4]
    return cleaned == BINK2_MODULE_KEY


def _sanitize_guest_module_label(raw: str) -> str:
    """Sanitize a guest module name without leaking host path segments."""
    cleaned = raw.strip()
    if not cleaned:
        return ""
    if "/" in cleaned or "\\" in cleaned or _WINDOWS_DRIVE_PREFIX_RE.match(cleaned):
        label = basename_label(cleaned)
        return "" if label == "<unknown>" else label
    return sanitize_text(cleaned)


def _empty_guest_manifest_info() -> dict[str, Any]:
    return {
        "present": False,
        "ok": True,
        "guest_modules": [],
        "bink2scarlett_present": False,
        "warnings": [],
    }


def inspect_guest_manifest(web_dist: Path) -> tuple[dict[str, Any], list[str]]:
    """Inspect optional `guest_manifest.json` inside `--web-dist` for lifted guest modules."""
    info = _empty_guest_manifest_info()
    warnings: list[str] = []

    status, resolved = _check_file_within_root(web_dist / GUEST_MANIFEST_NAME, web_dist)
    if status == "missing":
        return info, warnings
    if status == "escaping_symlink":
        msg = f"{GUEST_MANIFEST_NAME} is an escaping or broken symlink outside web-dist"
        info["ok"] = False
        info["warnings"].append(msg)
        warnings.append(msg)
        return info, warnings
    if status != "ok" or resolved is None:
        msg = f"{GUEST_MANIFEST_NAME} in web-dist is not a regular file"
        info["ok"] = False
        info["warnings"].append(msg)
        warnings.append(msg)
        return info, warnings

    info["present"] = True
    try:
        doc = json.loads(resolved.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = f"failed to parse {GUEST_MANIFEST_NAME}: {sanitize_text(type(exc).__name__)}"
        info["ok"] = False
        info["warnings"].append(msg)
        warnings.append(msg)
        return info, warnings

    if not isinstance(doc, Mapping):
        msg = f"{GUEST_MANIFEST_NAME} root must be a JSON object"
        info["ok"] = False
        info["warnings"].append(msg)
        warnings.append(msg)
        return info, warnings

    selected_key: str | None = None
    raw_modules: Any = None
    for key in GUEST_MANIFEST_MODULE_KEYS:
        if key in doc and doc[key] is not None:
            selected_key = key
            raw_modules = doc[key]
            break

    guest_modules: list[str] = []
    seen_lower: set[str] = set()
    if selected_key is not None:
        if not isinstance(raw_modules, list):
            msg = f"{GUEST_MANIFEST_NAME} '{selected_key}' must be a list"
            info["ok"] = False
            info["warnings"].append(msg)
            warnings.append(msg)
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
                    msg = (
                        f"{GUEST_MANIFEST_NAME} '{selected_key}[{idx}]' must be a "
                        f"non-empty module string or object"
                    )
                    info["ok"] = False
                    info["warnings"].append(msg)
                    warnings.append(msg)
                    continue
                cleaned = _sanitize_guest_module_label(raw_name)
                if not cleaned:
                    continue
                lowered = cleaned.lower()
                if lowered not in seen_lower:
                    seen_lower.add(lowered)
                    guest_modules.append(cleaned)

    info["guest_modules"] = guest_modules
    info["bink2scarlett_present"] = any(_is_bink2scarlett_module(m) for m in guest_modules)
    return info, warnings


def verify_web_dist(
    web_dist: Path,
    *,
    max_module_bytes: int = V8_MAX_WASM_MODULE_BYTES,
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Verify `runtime.js`, `runtime.wasm` (ABI + V8 cap), and any sibling side modules."""
    errors: list[str] = []
    warnings: list[str] = []

    runtime_js_info: dict[str, Any] = {
        "present": False,
        "ok": False,
        "size_bytes": 0,
        "errors": [],
    }
    runtime_wasm_info: dict[str, Any] = {
        "present": False,
        "ok": False,
        "size_bytes": 0,
        "v8_limit_ok": False,
        "magic_ok": False,
        "wasm_version": None,
        "is_memory64": False,
        "is_shared_memory": False,
        "all_host_imports_ok": False,
        "host_imports_valid": 0,
        "host_imports_total": 0,
        "missing_host_imports": [],
        "mismatched_host_imports": [],
        "exports_count": 0,
        "errors": [],
        "warnings": [],
    }
    section: dict[str, Any] = {
        "status": "fail",
        "bundle_mode": "unknown",
        "runtime_js": runtime_js_info,
        "runtime_wasm": runtime_wasm_info,
        "side_modules": [],
        "side_module_count": 0,
        "skipped_escaping_symlinks": 0,
        "rejected_unsafe_names": [],
        "runtime_wasm_bytes": 0,
        "side_modules_total_bytes": 0,
        "total_wasm_bytes": 0,
        "all_v8_limits_ok": False,
        "guest_manifest": _empty_guest_manifest_info(),
        "bink2scarlett_present": False,
    }

    if not web_dist.is_dir():
        errors.append("web-dist directory does not exist or is not a directory")
        return section, errors, warnings

    # ---- runtime.js ----
    js_status, resolved_js = _check_file_within_root(web_dist / RUNTIME_JS_NAME, web_dist)
    if js_status == "escaping_symlink":
        msg = "runtime.js is an escaping or broken symlink outside web-dist"
        runtime_js_info["errors"].append(msg)
        errors.append(msg)
    elif js_status != "ok" or resolved_js is None:
        msg = "runtime.js missing in web-dist"
        runtime_js_info["errors"].append(msg)
        errors.append(msg)
    else:
        runtime_js_info["present"] = True
        try:
            js_size = get_file_size(resolved_js)
            runtime_js_info["size_bytes"] = js_size
            if js_size <= 0:
                msg = "runtime.js is empty (0 bytes)"
                runtime_js_info["errors"].append(msg)
                errors.append(msg)
            else:
                runtime_js_info["ok"] = True
        except OSError as exc:
            msg = _sanitize_os_error(RUNTIME_JS_NAME, exc)
            runtime_js_info["errors"].append(msg)
            errors.append(msg)

    # ---- runtime.wasm ----
    wasm_status, resolved_wasm = _check_file_within_root(web_dist / RUNTIME_WASM_NAME, web_dist)
    if wasm_status == "escaping_symlink":
        msg = "runtime.wasm is an escaping or broken symlink outside web-dist"
        runtime_wasm_info["errors"].append(msg)
        errors.append(msg)
    elif wasm_status != "ok" or resolved_wasm is None:
        msg = "runtime.wasm missing in web-dist"
        runtime_wasm_info["errors"].append(msg)
        errors.append(msg)
    else:
        runtime_wasm_info["present"] = True
        try:
            wasm_size = get_file_size(resolved_wasm)
        except OSError as exc:
            msg = _sanitize_os_error(RUNTIME_WASM_NAME, exc)
            runtime_wasm_info["errors"].append(msg)
            errors.append(msg)
            wasm_size = -1

        if wasm_size >= 0:
            runtime_wasm_info["size_bytes"] = wasm_size
            section["runtime_wasm_bytes"] = wasm_size
            v8_ok = wasm_size <= max_module_bytes
            runtime_wasm_info["v8_limit_ok"] = v8_ok
            if not v8_ok:
                msg = (
                    f"runtime.wasm size ({wasm_size} bytes) exceeds V8 1 GiB module cap "
                    f"({max_module_bytes} bytes)"
                )
                runtime_wasm_info["errors"].append(msg)
                errors.append(msg)

            # Avoid reading a >1 GiB file on disk into memory; still check its header.
            actual_disk_size = int(resolved_wasm.stat().st_size)
            if actual_disk_size > V8_MAX_WASM_MODULE_BYTES:
                try:
                    header = _read_header_bytes(resolved_wasm, WASM_HEADER_BYTES)
                    if len(header) >= WASM_HEADER_BYTES:
                        magic = header[:4]
                        (version,) = struct.unpack("<I", header[4:8])
                        runtime_wasm_info["wasm_version"] = int(version)
                        runtime_wasm_info["magic_ok"] = (
                            magic == WASM_MAGIC and version == WASM_VERSION
                        )
                        if not runtime_wasm_info["magic_ok"]:
                            msg = "runtime.wasm has invalid Wasm header magic/version"
                            runtime_wasm_info["errors"].append(msg)
                            errors.append(msg)
                except OSError as exc:
                    msg = _sanitize_os_error(RUNTIME_WASM_NAME, exc)
                    runtime_wasm_info["errors"].append(msg)
                    errors.append(msg)
            else:
                try:
                    wasm_mod = _load_tools_module(
                        "check_wasm_exports.py", "_playgta6_bundle_wasm"
                    )
                    expected_total = len(getattr(wasm_mod, "REQUIRED_HOST_IMPORTS", ()))
                    runtime_wasm_info["host_imports_total"] = expected_total
                    raw_bytes = resolved_wasm.read_bytes()
                    module = wasm_mod.parse_wasm_module(raw_bytes)
                    report = wasm_mod.validate_runtime_wasm(
                        module,
                        require_memory64=True,
                        require_shared_memory=True,
                        require_host_imports=True,
                        path=RUNTIME_WASM_NAME,
                    )
                except Exception as exc:  # noqa: BLE001
                    msg = sanitize_text(f"runtime.wasm validation error: {exc}")
                    runtime_wasm_info["errors"].append(msg)
                    errors.append(msg)
                else:
                    host_valid = sum(1 for item in report.host_imports if item.ok)
                    host_total = len(report.host_imports)
                    runtime_wasm_info["magic_ok"] = True
                    runtime_wasm_info["wasm_version"] = report.wasm_version
                    runtime_wasm_info["is_memory64"] = report.is_memory64
                    runtime_wasm_info["is_shared_memory"] = report.is_shared_memory
                    runtime_wasm_info["all_host_imports_ok"] = report.all_host_imports_ok
                    runtime_wasm_info["host_imports_valid"] = host_valid
                    runtime_wasm_info["host_imports_total"] = host_total
                    runtime_wasm_info["missing_host_imports"] = list(
                        report.missing_host_imports
                    )
                    runtime_wasm_info["mismatched_host_imports"] = list(
                        report.mismatched_host_imports
                    )
                    runtime_wasm_info["exports_count"] = report.exports_count
                    val_errors = [sanitize_text(e) for e in report.errors]
                    val_warnings = [sanitize_text(w) for w in report.warnings]
                    runtime_wasm_info["errors"].extend(val_errors)
                    runtime_wasm_info["warnings"].extend(val_warnings)
                    errors.extend(val_errors)
                    warnings.extend(val_warnings)

            runtime_wasm_info["ok"] = bool(
                runtime_wasm_info["v8_limit_ok"]
                and runtime_wasm_info["magic_ok"]
                and runtime_wasm_info["is_memory64"]
                and runtime_wasm_info["is_shared_memory"]
                and runtime_wasm_info["all_host_imports_ok"]
                and len(runtime_wasm_info["errors"]) == 0
            )

    # ---- Sibling *.wasm side modules ----
    scan = discover_side_modules(web_dist)
    section["skipped_escaping_symlinks"] = scan.skipped_escaping_symlinks
    section["rejected_unsafe_names"] = list(scan.rejected_unsafe_names)

    if scan.skipped_escaping_symlinks > 0:
        warnings.append(
            f"skipped {scan.skipped_escaping_symlinks} escaping/broken *.wasm symlink(s) in web-dist"
        )
    if scan.rejected_unsafe_names:
        errors.append(
            "rejected unsafe side-module filename(s) in web-dist: "
            + ", ".join(scan.rejected_unsafe_names)
        )

    side_entries: list[dict[str, Any]] = []
    side_total_bytes = 0
    for mod in scan.modules:
        info = inspect_side_module_wasm(
            mod.name,
            mod.path,
            max_module_bytes=max_module_bytes,
        )
        side_entries.append(info)
        side_total_bytes += int(info["size_bytes"])
        errors.extend(info["errors"])

    section["side_modules"] = side_entries
    section["side_module_count"] = len(side_entries)
    section["side_modules_total_bytes"] = side_total_bytes
    section["total_wasm_bytes"] = int(section["runtime_wasm_bytes"]) + side_total_bytes
    section["bundle_mode"] = "split" if side_entries else "monolithic"
    section["all_v8_limits_ok"] = bool(
        runtime_wasm_info["present"]
        and runtime_wasm_info["v8_limit_ok"]
        and all(item["v8_limit_ok"] for item in side_entries)
    )

    # ---- Optional guest_manifest.json (lifted guest modules & Bink2Scarlett readiness) ----
    guest_manifest_info, gm_warnings = inspect_guest_manifest(web_dist)
    section["guest_manifest"] = guest_manifest_info
    section["bink2scarlett_present"] = bool(guest_manifest_info["bink2scarlett_present"])
    warnings.extend(gm_warnings)

    section["status"] = "pass" if len(errors) == 0 else "fail"

    return section, errors, warnings


# ---------------------------------------------------------------------------
# PGIM / Program Manifest verification (--boot-dir)
# ---------------------------------------------------------------------------


def _validate_safe_rel_pgim_path(raw_path: Any, *, label: str) -> tuple[str | None, str | None]:
    """Validate a manifest-referenced relative `.pgim` path (strict no-traversal)."""
    if not isinstance(raw_path, str):
        return None, f"{label} must be a string"
    cleaned = raw_path.strip()
    if not cleaned:
        return None, f"{label} must not be empty"
    if _CONTROL_CHAR_RE.search(cleaned):
        return None, f"{label} contains control characters"
    if "\\" in cleaned:
        return None, f"{label} must not contain backslashes"
    if cleaned.startswith("/") or cleaned.startswith("//"):
        return None, f"{label} must be relative to boot-dir (leading '/' rejected)"
    if _WINDOWS_DRIVE_PREFIX_RE.match(cleaned):
        return None, f"{label} must not contain a Windows drive prefix"
    if _SCHEME_RE.match(cleaned):
        return None, f"{label} must not contain a URL scheme"
    if "?" in cleaned or "#" in cleaned:
        return None, f"{label} must not contain query or fragment tokens"
    if _ENCODED_TRAVERSAL_RE.search(cleaned):
        return None, f"{label} contains percent-encoded path traversal tokens"

    segments = cleaned.split("/")
    for seg in segments:
        if not seg or seg in (".", ".."):
            return None, f"{label} contains unsafe path traversal segment"

    last = segments[-1]
    if not last.lower().endswith(".pgim") or len(last) <= len(".pgim"):
        return None, f"{label} must reference a .pgim file"

    return "/".join(segments), None


def _extract_explicit_pgim_field(obj: Mapping[str, Any]) -> Any:
    for key in ("pgimUrl", "pgim_url", "pgimPath", "pgim_path"):
        if obj.get(key) is not None:
            return obj[key]
    pgim_val = obj.get("pgim")
    if pgim_val is not None:
        if isinstance(pgim_val, Mapping):
            return pgim_val.get("path") if pgim_val.get("path") is not None else pgim_val.get("url")
        return pgim_val
    image_val = obj.get("image")
    if image_val is not None:
        if isinstance(image_val, str):
            return image_val
        if isinstance(image_val, Mapping):
            for key in ("path", "url", "pgim"):
                if image_val.get(key) is not None:
                    return image_val[key]
    return None


def _extract_manifest_pgim_files(
    raw_files: Any,
) -> tuple[list[str], list[str]]:
    """Extract validated relative `.pgim` paths from a manifest's `files` array."""
    if not isinstance(raw_files, list):
        return [], []
    paths: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for idx, item in enumerate(raw_files):
        raw_path: Any = None
        if isinstance(item, str):
            raw_path = item
        elif isinstance(item, Mapping) and "path" in item:
            raw_path = item["path"]
        if not isinstance(raw_path, str):
            continue
        if not raw_path.strip().lower().endswith(".pgim"):
            continue
        norm, err = _validate_safe_rel_pgim_path(
            raw_path, label=f"manifest files[{idx}] PGIM path"
        )
        if err is not None:
            errors.append(err)
            continue
        assert norm is not None
        if norm not in seen:
            seen.add(norm)
            paths.append(norm)
    return paths, errors


def _match_module_file_entry(module_name: str, file_paths: Sequence[str]) -> str | None:
    if not file_paths:
        return None
    target = module_name.lower()
    canonical = f"{target}/image/guest.pgim"
    canonical_suffix = f"/{canonical}"
    for rel in file_paths:
        norm = rel.lower()
        if norm == canonical or norm.endswith(canonical_suffix):
            return rel
    prefix = f"{target}/"
    for rel in file_paths:
        if rel.lower().startswith(prefix):
            return rel
    segment = f"/{target}/"
    for rel in file_paths:
        if segment in rel.lower():
            return rel
    return None


def _validate_module_name(raw_name: Any, *, label: str) -> tuple[str | None, str | None]:
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, f"{label} must be a non-empty string"
    cleaned = raw_name.strip()
    if ".." in cleaned or not _SAFE_MODULE_NAME_RE.match(cleaned):
        return None, f"{label} contains unsafe characters"
    return cleaned, None


def _extract_pgim_references(
    doc: Mapping[str, Any],
    *,
    manifest_kind: str,
) -> tuple[list[tuple[str | None, str]], list[str]]:
    """Parse `program-manifest.json` or `boot-manifest.json` into `(module_name, rel_pgim_path)` pairs."""
    refs: list[tuple[str | None, str]] = []
    errors: list[str] = []

    file_pgims, file_errors = _extract_manifest_pgim_files(doc.get("files"))
    errors.extend(file_errors)

    def _parse_modules_array(modules_raw: Any) -> None:
        if not isinstance(modules_raw, list) or len(modules_raw) == 0:
            errors.append('manifest "modules" must be a non-empty list')
            return
        seen_names: set[str] = set()
        for idx, entry in enumerate(modules_raw):
            explicit_pgim: Any = None
            if isinstance(entry, str):
                raw_name: Any = entry
            elif isinstance(entry, Mapping):
                raw_name = (
                    entry.get("module")
                    if entry.get("module") is not None
                    else (
                        entry.get("name")
                        if entry.get("name") is not None
                        else entry.get("id")
                    )
                )
                explicit_pgim = _extract_explicit_pgim_field(entry)
            else:
                errors.append(f"modules[{idx}] must be a string or JSON object")
                continue

            mod_name, name_err = _validate_module_name(
                raw_name, label=f"modules[{idx}] name"
            )
            if name_err is not None:
                errors.append(name_err)
                continue
            assert mod_name is not None
            key = mod_name.lower()
            if key in seen_names:
                errors.append(f"duplicate module '{mod_name}' in manifest")
                continue
            seen_names.add(key)

            if explicit_pgim is not None:
                rel, path_err = _validate_safe_rel_pgim_path(
                    explicit_pgim, label=f"module '{mod_name}' PGIM path"
                )
                if path_err is not None:
                    errors.append(path_err)
                    continue
                assert rel is not None
                refs.append((mod_name, rel))
            else:
                matched = _match_module_file_entry(mod_name, file_pgims)
                if matched is not None:
                    refs.append((mod_name, matched))
                else:
                    refs.append((mod_name, f"{mod_name}/image/guest.pgim"))

    if manifest_kind == "program":
        schema = doc.get("schema")
        if schema is not None and schema != PROGRAM_MANIFEST_SCHEMA:
            errors.append(
                f"unsupported program manifest schema '{sanitize_text(str(schema))}' "
                f"(expected '{PROGRAM_MANIFEST_SCHEMA}')"
            )
        schema_version = doc.get("schema_version")
        if schema_version is not None and schema_version != 1:
            errors.append(
                f"unsupported program manifest schema_version {schema_version!r} (expected 1)"
            )
        _parse_modules_array(doc.get("modules"))
        return refs, errors

    # Fallback: boot-manifest.json
    schema_version = doc.get("schema_version")
    if schema_version is not None and schema_version != 1:
        errors.append(
            f"unsupported boot manifest schema_version {schema_version!r} (expected 1)"
        )

    explicit_top = _extract_explicit_pgim_field(doc)
    if explicit_top is None and isinstance(doc.get("modules"), list) and len(doc["modules"]) > 0:
        _parse_modules_array(doc["modules"])
        return refs, errors

    mod_name: str | None = "main"
    if doc.get("module") is not None:
        mod_name, name_err = _validate_module_name(doc.get("module"), label="boot manifest module")
        if name_err is not None:
            errors.append(name_err)
            return refs, errors
    elif doc.get("name") is not None:
        mod_name, name_err = _validate_module_name(doc.get("name"), label="boot manifest name")
        if name_err is not None:
            errors.append(name_err)
            return refs, errors

    if explicit_top is not None:
        rel, path_err = _validate_safe_rel_pgim_path(
            explicit_top, label="boot manifest PGIM path"
        )
        if path_err is not None:
            errors.append(path_err)
            return refs, errors
        assert rel is not None
        refs.append((mod_name, rel))
    elif file_pgims:
        for rel in file_pgims:
            refs.append((mod_name, rel))
    elif isinstance(doc.get("module"), str):
        default_rel = (
            "image/guest.pgim"
            if doc.get("schema") == "playgta6-recomp-build"
            else f"{mod_name}/image/guest.pgim"
        )
        refs.append((mod_name, default_rel))
    else:
        errors.append("boot-manifest.json does not declare any .pgim image path")

    return refs, errors


def _inspect_pgim_file(
    boot_dir: Path,
    module_name: str | None,
    rel_path: str,
) -> dict[str, Any]:
    """Verify one manifest-referenced `.pgim` file stays inside `boot_dir` and has `PGIM` magic."""
    errors: list[str] = []
    present = False
    magic_ok = False
    size_bytes = 0
    pgim_version: int | None = None

    resolved_boot = boot_dir.resolve()
    parts = rel_path.split("/")
    candidate = boot_dir.joinpath(*parts)

    # Reject any intermediate symlink component escaping boot_dir as well as final target.
    cursor = boot_dir
    escaped = False
    for part in parts:
        cursor = cursor / part
        if cursor.is_symlink():
            try:
                resolved_step = cursor.resolve(strict=True)
            except OSError:
                escaped = True
                break
            if not resolved_step.is_relative_to(resolved_boot):
                escaped = True
                break

    if escaped:
        errors.append(f"referenced PGIM '{rel_path}' escapes boot-dir via symlink")
        return {
            "module": module_name,
            "rel_path": rel_path,
            "present": False,
            "magic_ok": False,
            "size_bytes": 0,
            "pgim_version": None,
            "ok": False,
            "errors": errors,
        }

    status, resolved = _check_file_within_root(candidate, boot_dir)
    if status == "escaping_symlink":
        errors.append(f"referenced PGIM '{rel_path}' escapes boot-dir via symlink")
    elif status != "ok" or resolved is None:
        errors.append(f"referenced PGIM missing in boot-dir: {rel_path}")
    else:
        present = True
        try:
            size_bytes = get_file_size(resolved)
            header = _read_header_bytes(resolved, PGIM_HEADER_BYTES)
        except OSError as exc:
            errors.append(_sanitize_os_error(rel_path, exc))
            header = b""

        if len(header) < len(PGIM_MAGIC):
            errors.append(
                f"PGIM '{rel_path}' truncated: expected at least 4 magic bytes, got {len(header)}"
            )
        elif header[:4] != PGIM_MAGIC:
            errors.append(f"PGIM '{rel_path}' has invalid magic (expected 'PGIM')")
        else:
            magic_ok = True
            if len(header) >= PGIM_HEADER_BYTES:
                version, hdr_bytes, sec_entry_bytes = struct.unpack_from("<III", header, 4)
                if hdr_bytes == PGIM_HEADER_BYTES and sec_entry_bytes == PGIM_SECTION_ENTRY_BYTES:
                    pgim_version = int(version)
                    if version != PGIM_VERSION:
                        magic_ok = False
                        errors.append(
                            f"PGIM '{rel_path}' has unsupported version {version} "
                            f"(expected {PGIM_VERSION})"
                        )

    return {
        "module": module_name,
        "rel_path": rel_path,
        "present": present,
        "magic_ok": magic_ok,
        "size_bytes": size_bytes,
        "pgim_version": pgim_version,
        "ok": len(errors) == 0,
        "errors": errors,
    }


def verify_boot_dir(boot_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Verify `program-manifest.json` (or `boot-manifest.json`) and referenced `.pgim` files."""
    errors: list[str] = []
    warnings: list[str] = []

    section: dict[str, Any] = {
        "checked": True,
        "status": "fail",
        "manifest_name": None,
        "manifest_kind": None,
        "pgim_count": 0,
        "valid_pgim_count": 0,
        "total_pgim_bytes": 0,
        "pgims": [],
        "guest_modules": [],
        "bink2scarlett_present": False,
    }

    if not boot_dir.is_dir():
        errors.append("boot-dir does not exist or is not a directory")
        return section, errors, warnings

    manifest_path: Path | None = None
    manifest_name: str | None = None
    manifest_kind: str | None = None

    prog_status, prog_resolved = _check_file_within_root(
        boot_dir / PROGRAM_MANIFEST_NAME, boot_dir
    )
    if prog_status == "escaping_symlink":
        errors.append(f"{PROGRAM_MANIFEST_NAME} is an escaping or broken symlink outside boot-dir")
        return section, errors, warnings
    if prog_status == "ok" and prog_resolved is not None:
        manifest_path = prog_resolved
        manifest_name = PROGRAM_MANIFEST_NAME
        manifest_kind = "program"
    else:
        boot_status, boot_resolved = _check_file_within_root(
            boot_dir / BOOT_MANIFEST_NAME, boot_dir
        )
        if boot_status == "escaping_symlink":
            errors.append(
                f"{BOOT_MANIFEST_NAME} is an escaping or broken symlink outside boot-dir"
            )
            return section, errors, warnings
        if boot_status == "ok" and boot_resolved is not None:
            manifest_path = boot_resolved
            manifest_name = BOOT_MANIFEST_NAME
            manifest_kind = "boot"

    if manifest_path is None or manifest_name is None or manifest_kind is None:
        errors.append(
            f"boot-dir missing both {PROGRAM_MANIFEST_NAME} and {BOOT_MANIFEST_NAME}"
        )
        return section, errors, warnings

    section["manifest_name"] = manifest_name
    section["manifest_kind"] = manifest_kind

    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"failed to parse {manifest_name}: {sanitize_text(type(exc).__name__)}")
        return section, errors, warnings

    if not isinstance(doc, Mapping):
        errors.append(f"{manifest_name} root must be a JSON object")
        return section, errors, warnings

    refs, ref_errors = _extract_pgim_references(doc, manifest_kind=manifest_kind)
    errors.extend(ref_errors)

    pgim_entries: list[dict[str, Any]] = []
    valid_count = 0
    total_bytes = 0
    for mod_name, rel_path in refs:
        info = _inspect_pgim_file(boot_dir, mod_name, rel_path)
        pgim_entries.append(info)
        total_bytes += int(info["size_bytes"])
        if info["ok"]:
            valid_count += 1
        else:
            errors.extend(info["errors"])

    boot_modules: list[str] = []
    seen_boot_modules: set[str] = set()
    for info in pgim_entries:
        mod_name = info.get("module")
        if isinstance(mod_name, str) and mod_name.strip():
            cleaned_mod = sanitize_text(mod_name.strip())
            if cleaned_mod.lower() not in seen_boot_modules:
                seen_boot_modules.add(cleaned_mod.lower())
                boot_modules.append(cleaned_mod)

    section["pgims"] = pgim_entries
    section["pgim_count"] = len(pgim_entries)
    section["valid_pgim_count"] = valid_count
    section["total_pgim_bytes"] = total_bytes
    section["guest_modules"] = boot_modules
    section["bink2scarlett_present"] = any(_is_bink2scarlett_module(m) for m in boot_modules)
    if not pgim_entries and not errors:
        errors.append(f"{manifest_name} references zero .pgim files")

    section["status"] = "pass" if len(errors) == 0 else "fail"
    return section, errors, warnings


# ---------------------------------------------------------------------------
# Shader Pack verification (--shaders-dir)
# ---------------------------------------------------------------------------


def inspect_wgsl_pack_file(pack_name: str, pack_path: Path) -> tuple[dict[str, Any], list[str]]:
    """Safely validate a `.wgsl.pack` (`WGPK` v1) binary matching `shaders/src/pack.rs`."""
    errors: list[str] = []
    warnings: list[str] = []
    size_bytes = 0
    magic_ok = False
    version: int | None = None
    entry_count: int | None = None
    flags: int | None = None

    try:
        size_bytes = get_file_size(pack_path)
    except OSError as exc:
        errors.append(_sanitize_os_error(pack_name, exc))
        return (
            {
                "name": sanitize_text(pack_name),
                "size_bytes": 0,
                "magic_ok": False,
                "version": None,
                "entry_count": None,
                "flags": None,
                "ok": False,
                "errors": errors,
            },
            warnings,
        )

    if size_bytes < WGPK_HEADER_BYTES:
        errors.append(
            f"{sanitize_text(pack_name)} truncated: {size_bytes} bytes "
            f"(minimum WGPK header is {WGPK_HEADER_BYTES} bytes)"
        )
    else:
        try:
            with pack_path.open("rb") as fh:
                header = fh.read(WGPK_HEADER_BYTES)
                if len(header) < WGPK_HEADER_BYTES:
                    errors.append(
                        f"{sanitize_text(pack_name)} truncated while reading WGPK header"
                    )
                else:
                    magic, ver, count, hdr_flags = WGPK_HEADER_STRUCT.unpack_from(header, 0)
                    if magic != WGPK_MAGIC:
                        errors.append(
                            f"{sanitize_text(pack_name)} has invalid magic (expected 'WGPK')"
                        )
                    else:
                        magic_ok = True
                        version = int(ver)
                        entry_count = int(count)
                        flags = int(hdr_flags)

                        if ver != WGPK_VERSION:
                            errors.append(
                                f"{sanitize_text(pack_name)} has unsupported WGPK version {ver} "
                                f"(expected {WGPK_VERSION})"
                            )
                        else:
                            table_bytes = count * WGPK_ENTRY_BYTES
                            data_start = WGPK_HEADER_BYTES + table_bytes
                            if data_start > size_bytes:
                                errors.append(
                                    f"{sanitize_text(pack_name)} truncated entry table: "
                                    f"entry_count={count} requires {data_start} bytes, "
                                    f"file is {size_bytes} bytes"
                                )
                            else:
                                # Stream-check every 20-byte entry table record without
                                # loading the full WGSL payloads into memory.
                                table_raw = fh.read(table_bytes)
                                if len(table_raw) < table_bytes:
                                    errors.append(
                                        f"{sanitize_text(pack_name)} truncated while reading entry table"
                                    )
                                else:
                                    for idx in range(count):
                                        base = idx * WGPK_ENTRY_BYTES
                                        _h, stage_raw, _res, offset, length = (
                                            WGPK_ENTRY_STRUCT.unpack_from(table_raw, base)
                                        )
                                        if stage_raw not in WGPK_VALID_STAGES:
                                            errors.append(
                                                f"{sanitize_text(pack_name)} entry {idx} has "
                                                f"invalid shader stage {stage_raw}"
                                            )
                                            break
                                        if offset < data_start or (offset + length) > size_bytes:
                                            errors.append(
                                                f"{sanitize_text(pack_name)} entry {idx} payload "
                                                f"[{offset}..+{length}] exceeds pack bounds "
                                                f"[{data_start}..{size_bytes}]"
                                            )
                                            break
                                if count == 0 and not errors:
                                    warnings.append(
                                        f"{sanitize_text(pack_name)} contains 0 shader entries"
                                    )
        except OSError as exc:
            errors.append(_sanitize_os_error(pack_name, exc))

    return (
        {
            "name": sanitize_text(pack_name),
            "size_bytes": size_bytes,
            "magic_ok": magic_ok,
            "version": version,
            "entry_count": entry_count,
            "flags": flags,
            "ok": len(errors) == 0,
            "errors": errors,
        },
        warnings,
    )


def _is_valid_uint32(value: Any) -> bool:
    """Return True only for non-bool integers within `[0, 0xFFFFFFFF]`."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF


def validate_inline_sampler_source(source: Any) -> bool | None:
    """Validate a `#950` `inline_sampler` reflection binding source.

    Returns:
    - `None` when `source` does not declare `"inline_sampler"`.
    - `True` when `source["inline_sampler"]["sdwords"]` is a 4-element list of uint32s.
    - `False` when `"inline_sampler"` is present but malformed.
    """
    if not isinstance(source, Mapping) or "inline_sampler" not in source:
        return None
    sampler = source["inline_sampler"]
    if not isinstance(sampler, Mapping):
        return False
    words = sampler.get("sdwords")
    return (
        isinstance(words, list)
        and len(words) == 4
        and all(_is_valid_uint32(w) for w in words)
    )


def inspect_reflect_sidecar_file(
    sidecar_name: str,
    sidecar_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Inspect a `<pack>.reflect.json` stage-reflection sidecar for `#950` inline samplers."""
    safe_name = sanitize_text(sidecar_name)
    warnings: list[str] = []
    size_bytes = 0
    schema_str: str | None = None
    version_int: int | None = None
    stage_count = 0
    inline_sampler_stages = 0
    inline_sampler_bindings = 0
    malformed_inline_samplers = 0

    try:
        size_bytes = get_file_size(sidecar_path)
    except OSError as exc:
        warnings.append(_sanitize_os_error(sidecar_name, exc))
        return (
            {
                "name": safe_name,
                "size_bytes": 0,
                "schema": None,
                "version": None,
                "stage_count": 0,
                "inline_sampler_stages": 0,
                "inline_sampler_bindings": 0,
                "malformed_inline_samplers": 0,
                "ok": False,
                "warnings": warnings,
            },
            warnings,
        )

    try:
        raw_text = sidecar_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        warnings.append(f"{safe_name}: failed to read ({sanitize_text(type(exc).__name__)})")
        return (
            {
                "name": safe_name,
                "size_bytes": size_bytes,
                "schema": None,
                "version": None,
                "stage_count": 0,
                "inline_sampler_stages": 0,
                "inline_sampler_bindings": 0,
                "malformed_inline_samplers": 0,
                "ok": False,
                "warnings": warnings,
            },
            warnings,
        )

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        warnings.append(f"{safe_name}: invalid JSON ({sanitize_text(type(exc).__name__)})")
        return (
            {
                "name": safe_name,
                "size_bytes": size_bytes,
                "schema": None,
                "version": None,
                "stage_count": 0,
                "inline_sampler_stages": 0,
                "inline_sampler_bindings": 0,
                "malformed_inline_samplers": 0,
                "ok": False,
                "warnings": warnings,
            },
            warnings,
        )

    if not isinstance(doc, Mapping):
        warnings.append(f"{safe_name}: top-level JSON value must be an object")
        return (
            {
                "name": safe_name,
                "size_bytes": size_bytes,
                "schema": None,
                "version": None,
                "stage_count": 0,
                "inline_sampler_stages": 0,
                "inline_sampler_bindings": 0,
                "malformed_inline_samplers": 0,
                "ok": False,
                "warnings": warnings,
            },
            warnings,
        )

    raw_schema = doc.get("schema")
    if raw_schema is not None:
        if isinstance(raw_schema, str) and raw_schema.strip():
            schema_str = sanitize_text(raw_schema.strip())
        else:
            warnings.append(f"{safe_name}: 'schema' must be a non-empty string")

    raw_version = doc.get("version")
    if raw_version is not None:
        if isinstance(raw_version, int) and not isinstance(raw_version, bool) and raw_version >= 1:
            version_int = int(raw_version)
        else:
            warnings.append(f"{safe_name}: 'version' must be a positive integer")

    stages = doc.get("stages")
    if not isinstance(stages, Mapping):
        warnings.append(f"{safe_name}: missing or invalid 'stages' object")
    else:
        stage_count = len(stages)
        for stage_key, stage_entry in stages.items():
            safe_stage_key = sanitize_text(str(stage_key))
            if not isinstance(stage_entry, Mapping):
                warnings.append(f"{safe_name}: stage '{safe_stage_key}' must be a JSON object")
                continue
            bindings = stage_entry.get("bindings")
            if bindings is None:
                continue
            if not isinstance(bindings, list):
                warnings.append(f"{safe_name}: stage '{safe_stage_key}' 'bindings' must be a list")
                continue

            stage_has_inline_sampler = False
            for b_idx, binding in enumerate(bindings):
                if not isinstance(binding, Mapping):
                    continue
                inline_status = validate_inline_sampler_source(binding.get("source"))
                if inline_status is True:
                    inline_sampler_bindings += 1
                    stage_has_inline_sampler = True
                elif inline_status is False:
                    malformed_inline_samplers += 1
                    warnings.append(
                        f"{safe_name}: stage '{safe_stage_key}' binding[{b_idx}] has "
                        f"malformed inline_sampler (sdwords must be 4 uint32 integers)"
                    )
            if stage_has_inline_sampler:
                inline_sampler_stages += 1

    return (
        {
            "name": safe_name,
            "size_bytes": size_bytes,
            "schema": schema_str,
            "version": version_int,
            "stage_count": stage_count,
            "inline_sampler_stages": inline_sampler_stages,
            "inline_sampler_bindings": inline_sampler_bindings,
            "malformed_inline_samplers": malformed_inline_samplers,
            "ok": len(warnings) == 0,
            "warnings": warnings,
        },
        warnings,
    )


def verify_shaders_dir(shaders_dir: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    """Verify `*.wgsl.pack` (`WGPK`) shader bundles and optional `*.reflect.json` sidecars."""
    errors: list[str] = []
    warnings: list[str] = []

    section: dict[str, Any] = {
        "checked": True,
        "status": "fail",
        "pack_count": 0,
        "valid_pack_count": 0,
        "skipped_escaping_symlinks": 0,
        "total_pack_bytes": 0,
        "total_shader_entries": 0,
        "packs": [],
        "reflect_sidecar_count": 0,
        "valid_reflect_sidecar_count": 0,
        "reflect_stage_count": 0,
        "inline_sampler_stages": 0,
        "inline_sampler_bindings": 0,
        "malformed_inline_samplers": 0,
        "reflect_sidecars": [],
    }

    if not shaders_dir.is_dir():
        errors.append("shaders-dir does not exist or is not a directory")
        return section, errors, warnings

    resolved_root = shaders_dir.resolve()
    candidates: list[tuple[str, Path]] = []
    sidecar_candidates: list[tuple[str, Path]] = []
    skipped_pack_symlinks = 0
    skipped_sidecar_symlinks = 0

    for entry in shaders_dir.iterdir():
        name = entry.name
        lowered = name.lower()
        is_pack = lowered.endswith(".wgsl.pack")
        is_sidecar = lowered.endswith(REFLECT_SIDECAR_SUFFIX)
        if not is_pack and not is_sidecar:
            continue
        if entry.is_symlink():
            try:
                resolved = entry.resolve(strict=True)
            except OSError:
                if is_pack:
                    skipped_pack_symlinks += 1
                else:
                    skipped_sidecar_symlinks += 1
                continue
            if not resolved.is_relative_to(resolved_root):
                if is_pack:
                    skipped_pack_symlinks += 1
                else:
                    skipped_sidecar_symlinks += 1
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
                if is_pack:
                    skipped_pack_symlinks += 1
                else:
                    skipped_sidecar_symlinks += 1
                continue

        if is_pack:
            candidates.append((name, resolved))
        else:
            sidecar_candidates.append((name, resolved))

    candidates.sort(key=lambda item: (_natural_sort_key(item[0]), item[0]))
    sidecar_candidates.sort(key=lambda item: (_natural_sort_key(item[0]), item[0]))
    section["skipped_escaping_symlinks"] = skipped_pack_symlinks + skipped_sidecar_symlinks
    if skipped_pack_symlinks > 0:
        warnings.append(
            f"skipped {skipped_pack_symlinks} escaping/broken *.wgsl.pack symlink(s) in shaders-dir"
        )
    if skipped_sidecar_symlinks > 0:
        warnings.append(
            f"skipped {skipped_sidecar_symlinks} escaping/broken *.reflect.json symlink(s) in shaders-dir"
        )

    sidecars: list[dict[str, Any]] = []
    valid_sidecars = 0
    reflect_stages = 0
    inline_sampler_stages = 0
    inline_sampler_bindings = 0
    malformed_inline_samplers = 0
    for sidecar_name, sidecar_path in sidecar_candidates:
        sc_info, sc_warnings = inspect_reflect_sidecar_file(sidecar_name, sidecar_path)
        sidecars.append(sc_info)
        warnings.extend(sc_warnings)
        reflect_stages += int(sc_info["stage_count"])
        inline_sampler_stages += int(sc_info["inline_sampler_stages"])
        inline_sampler_bindings += int(sc_info["inline_sampler_bindings"])
        malformed_inline_samplers += int(sc_info["malformed_inline_samplers"])
        if sc_info["ok"]:
            valid_sidecars += 1

    section["reflect_sidecars"] = sidecars
    section["reflect_sidecar_count"] = len(sidecars)
    section["valid_reflect_sidecar_count"] = valid_sidecars
    section["reflect_stage_count"] = reflect_stages
    section["inline_sampler_stages"] = inline_sampler_stages
    section["inline_sampler_bindings"] = inline_sampler_bindings
    section["malformed_inline_samplers"] = malformed_inline_samplers

    if not candidates:
        errors.append("no .wgsl.pack shader bundles found in shaders-dir")
        return section, errors, warnings

    packs: list[dict[str, Any]] = []
    valid_count = 0
    total_bytes = 0
    total_entries = 0
    for pack_name, pack_path in candidates:
        info, pack_warnings = inspect_wgsl_pack_file(pack_name, pack_path)
        packs.append(info)
        warnings.extend(pack_warnings)
        total_bytes += int(info["size_bytes"])
        if info["ok"]:
            valid_count += 1
            total_entries += int(info["entry_count"] or 0)
        else:
            errors.extend(info["errors"])

    section["packs"] = packs
    section["pack_count"] = len(packs)
    section["valid_pack_count"] = valid_count
    section["total_pack_bytes"] = total_bytes
    section["total_shader_entries"] = total_entries
    section["status"] = "pass" if len(errors) == 0 else "fail"
    return section, errors, warnings


# ---------------------------------------------------------------------------
# End-to-end bundle verifier
# ---------------------------------------------------------------------------


def _empty_boot_section() -> dict[str, Any]:
    return {
        "checked": False,
        "status": "skipped",
        "manifest_name": None,
        "manifest_kind": None,
        "pgim_count": 0,
        "valid_pgim_count": 0,
        "total_pgim_bytes": 0,
        "pgims": [],
        "guest_modules": [],
        "bink2scarlett_present": False,
    }


def _empty_shaders_section() -> dict[str, Any]:
    return {
        "checked": False,
        "status": "skipped",
        "pack_count": 0,
        "valid_pack_count": 0,
        "skipped_escaping_symlinks": 0,
        "total_pack_bytes": 0,
        "total_shader_entries": 0,
        "packs": [],
        "reflect_sidecar_count": 0,
        "valid_reflect_sidecar_count": 0,
        "reflect_stage_count": 0,
        "inline_sampler_stages": 0,
        "inline_sampler_bindings": 0,
        "malformed_inline_samplers": 0,
        "reflect_sidecars": [],
    }


def verify_browser_boot_bundle(
    *,
    web_dist: Path | str,
    boot_dir: Path | str | None = None,
    shaders_dir: Path | str | None = None,
    strict: bool = False,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    max_module_bytes: int = V8_MAX_WASM_MODULE_BYTES,
) -> dict[str, Any]:
    """Verify a staged browser-boot bundle and return a strictly sanitized report."""
    resolved_repo = Path(repo_root).resolve()
    web_dist_path = Path(web_dist).expanduser()
    boot_dir_path = Path(boot_dir).expanduser() if boot_dir is not None else None
    shaders_dir_path = Path(shaders_dir).expanduser() if shaders_dir is not None else None

    errors: list[str] = []
    warnings: list[str] = []

    wasm_section, wasm_errors, wasm_warnings = verify_web_dist(
        web_dist_path,
        max_module_bytes=max_module_bytes,
    )
    errors.extend(wasm_errors)
    warnings.extend(wasm_warnings)

    if boot_dir_path is not None:
        boot_section, boot_errors, boot_warnings = verify_boot_dir(boot_dir_path)
        errors.extend(boot_errors)
        warnings.extend(boot_warnings)
    else:
        boot_section = _empty_boot_section()

    if shaders_dir_path is not None:
        shaders_section, shader_errors, shader_warnings = verify_shaders_dir(shaders_dir_path)
        errors.extend(shader_errors)
        warnings.extend(shader_warnings)
    else:
        shaders_section = _empty_shaders_section()

    ok = len(errors) == 0 and (not strict or len(warnings) == 0)

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "ok": ok,
        "strict": bool(strict),
        "v8_max_wasm_module_bytes": int(max_module_bytes),
        "paths": {
            "web_dist": repo_relative_label(web_dist_path, resolved_repo),
            "boot_dir": repo_relative_label(boot_dir_path, resolved_repo),
            "shaders_dir": repo_relative_label(shaders_dir_path, resolved_repo),
        },
        "wasm": wasm_section,
        "boot": boot_section,
        "shaders": shaders_section,
        "errors": errors,
        "warnings": warnings,
    }
    return sanitize_value(report)


# ---------------------------------------------------------------------------
# Output formatting & CLI
# ---------------------------------------------------------------------------


def _format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def format_report_text(report: Mapping[str, Any]) -> str:
    """Render a concise, strictly sanitized human-readable bundle verification summary."""
    status_label = "PASS" if report.get("ok") else "FAIL"
    strict_tag = " (strict)" if report.get("strict") else ""
    paths = report.get("paths") or {}
    wasm = report.get("wasm") or {}
    rt_js = wasm.get("runtime_js") or {}
    rt_wasm = wasm.get("runtime_wasm") or {}
    side_modules = wasm.get("side_modules") or []
    boot = report.get("boot") or {}
    shaders = report.get("shaders") or {}

    lines: list[str] = [
        f"Browser Boot Bundle Verification [{status_label}]{strict_tag}",
        f"  Wasm64 [{str(wasm.get('status', 'fail')).upper()}]: {paths.get('web_dist')} "
        f"(mode={wasm.get('bundle_mode')}, "
        f"side_modules={wasm.get('side_module_count', 0)}, "
        f"total={wasm.get('total_wasm_bytes', 0)} bytes / "
        f"{_format_mib(int(wasm.get('total_wasm_bytes', 0)))}, "
        f"v8_cap={'OK' if wasm.get('all_v8_limits_ok') else 'EXCEEDED'})",
        f"    runtime.js:   {'OK' if rt_js.get('ok') else 'FAIL'} "
        f"({rt_js.get('size_bytes', 0)} bytes)",
        f"    runtime.wasm: {'OK' if rt_wasm.get('ok') else 'FAIL'} "
        f"({rt_wasm.get('size_bytes', 0)} bytes, "
        f"v8_limit={'OK' if rt_wasm.get('v8_limit_ok') else 'FAIL'}, "
        f"memory64={'YES' if rt_wasm.get('is_memory64') else 'NO'}, "
        f"shared={'YES' if rt_wasm.get('is_shared_memory') else 'NO'}, "
        f"host_imports={rt_wasm.get('host_imports_valid', 0)}/"
        f"{rt_wasm.get('host_imports_total', 0)})",
    ]

    for mod in side_modules:
        lines.append(
            f"    side {mod.get('name')}: {'OK' if mod.get('ok') else 'FAIL'} "
            f"({mod.get('size_bytes', 0)} bytes, "
            f"magic={'OK' if mod.get('magic_ok') else 'FAIL'}, "
            f"v8_limit={'OK' if mod.get('v8_limit_ok') else 'FAIL'})"
        )

    guest_manifest = wasm.get("guest_manifest") or {}
    if guest_manifest.get("present"):
        gm_mods = guest_manifest.get("guest_modules") or []
        lines.append(
            f"    guest_manifest.json: {'OK' if guest_manifest.get('ok') else 'WARN'} "
            f"(guest_modules={len(gm_mods)}, "
            f"bink2scarlett={'YES' if guest_manifest.get('bink2scarlett_present') else 'NO'})"
        )

    if boot.get("checked"):
        lines.append(
            f"  Boot [{str(boot.get('status', 'fail')).upper()}]: {paths.get('boot_dir')} "
            f"(manifest={boot.get('manifest_name') or '<none>'}, "
            f"pgims={boot.get('valid_pgim_count', 0)}/{boot.get('pgim_count', 0)}, "
            f"total={boot.get('total_pgim_bytes', 0)} bytes)"
        )
        for pgim in boot.get("pgims") or []:
            lines.append(
                f"    pgim {pgim.get('rel_path')}: {'OK' if pgim.get('ok') else 'FAIL'} "
                f"({pgim.get('size_bytes', 0)} bytes, "
                f"magic={'PGIM' if pgim.get('magic_ok') else 'BAD'})"
            )
    else:
        lines.append("  Boot [SKIPPED]: --boot-dir not specified")

    if shaders.get("checked"):
        lines.append(
            f"  Shaders [{str(shaders.get('status', 'fail')).upper()}]: {paths.get('shaders_dir')} "
            f"(packs={shaders.get('valid_pack_count', 0)}/{shaders.get('pack_count', 0)}, "
            f"entries={shaders.get('total_shader_entries', 0)}, "
            f"total={shaders.get('total_pack_bytes', 0)} bytes)"
        )
        for pack in shaders.get("packs") or []:
            lines.append(
                f"    pack {pack.get('name')}: {'OK' if pack.get('ok') else 'FAIL'} "
                f"({pack.get('size_bytes', 0)} bytes, "
                f"magic={'WGPK' if pack.get('magic_ok') else 'BAD'}, "
                f"entries={pack.get('entry_count')})"
            )
        if int(shaders.get("reflect_sidecar_count", 0)) > 0:
            lines.append(
                f"    reflect sidecars: "
                f"{shaders.get('valid_reflect_sidecar_count', 0)}/"
                f"{shaders.get('reflect_sidecar_count', 0)} "
                f"(stages={shaders.get('reflect_stage_count', 0)}, "
                f"inline_sampler_stages={shaders.get('inline_sampler_stages', 0)}, "
                f"inline_sampler_bindings={shaders.get('inline_sampler_bindings', 0)}, "
                f"malformed_inline_samplers={shaders.get('malformed_inline_samplers', 0)})"
            )
    else:
        lines.append("  Shaders [SKIPPED]: --shaders-dir not specified")

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
            "Sanitized pre-launch browser boot & split-module Wasm64 bundle verifier "
            "(checks V8 1 GiB cap, Wasm64/shared/host ABI, PGIM manifest, and WGPK shader packs)."
        )
    )
    parser.add_argument(
        "--web-dist",
        type=Path,
        default=DEFAULT_WEB_DIST_REL,
        help="Directory containing runtime.js, runtime.wasm, and optional sibling *.wasm side modules (default: web/dist).",
    )
    parser.add_argument(
        "--boot-dir",
        type=Path,
        default=None,
        help="Optional boot directory containing program-manifest.json (or boot-manifest.json) and referenced *.pgim files.",
    )
    parser.add_argument(
        "--shaders-dir",
        type=Path,
        default=None,
        help="Optional shader directory containing *.wgsl.pack (WGPK) shader bundles.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit strictly sanitized machine-readable JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat any warning (e.g. skipped escaping symlinks or empty shader packs) as a failure.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = verify_browser_boot_bundle(
        web_dist=args.web_dist,
        boot_dir=args.boot_dir,
        shaders_dir=args.shaders_dir,
        strict=args.strict,
    )

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
