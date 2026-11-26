#!/usr/bin/env python3
"""Recompilation coverage and boot-path bottleneck profiler (Track 1 / R15, T52, T68, T83).

Parses exact JSON produced by `playgta6-recomp build` (`build-manifest.json`,
schema `playgta6-recomp-build`), `playgta6-recomp build-program`
(`program-manifest.json`, schema `playgta6-recomp-program`),
`tools/build_guest_wasm.py` (`guest_manifest.json`), and
`playgta6-recomp discover --json` (`render_summary_json` in
`recomp/src/summary.rs`). Also accepts a build output directory (module dir,
multi-module program dir, dist dir, or `--out` root containing `<module>/`),
and supports browser fast-boot Wasm64 staging audits via `--dist-dir <DIR>`.

Computes:
- Function / block / instruction lift coverage and shard size distribution.
- Multi-module program summary (module count, guest DLLs, per-module lifted
  function/shard/import counts, unresolved imports by DLL, and dependency cycles).
- Wasm / `GOT.func` and `wasmTable` footprint telemetry from `guest_manifest.json`
  (or dispatch-mode estimate when only `program-manifest.json` is present).
- Browser fast-boot Wasm64 audit (`--dist-dir`): `runtime.wasm` +
  `guest_part_*.wasm` presence, total bytes, split-balance ratio / dominant-shard
  warning, pure-Python Wasm export section inspection (`_main` / `main`,
  `_playgta6_get_input_state_ptr`, `_playgta6_audio_ring`), and boot manifest
  consistency against staged artifacts.
- Top-N unlifted x86-64 mnemonics and lift-failure categories.
- Entry-point + TLS-callback boot-path BFS, including shortest-path traces
  to `D3D12XboxCreateDevice` / `PresentX` when import/call metadata is present.
- Jump-table resolution summary.
- Prioritized recommendations to unblock splash-screen boot.

Security invariant: stdout, JSON, and Markdown never include host absolute
paths, usernames, or raw binary bytes. Pure Python 3.12 stdlib.

Usage:
    python tools/audit_recomp_build.py <recomp-build-dir-or-json> [--top N] [--boot-depth D] [--fail-on-unresolved [N]] [--dist-dir DIR] [--boot-manifest FILE] [--balance-warn-ratio R] [--markdown] [--json]
    python tools/audit_recomp_build.py --dist-dir <staged-dist-dir> [--boot-manifest FILE] [--balance-warn-ratio R] [--markdown] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence

SCHEMA_BUILD = "playgta6-recomp-build"
SCHEMA_PROGRAM = "playgta6-recomp-program"
SCHEMA_AUDIT = "playgta6-recomp-audit"
SCHEMA_AUDIT_VERSION = 1
MANIFEST_FILE = "build-manifest.json"
PROGRAM_MANIFEST_FILE = "program-manifest.json"
GUEST_MANIFEST_FILE = "guest_manifest.json"
BOOT_MANIFEST_FILE = "boot-manifest.json"
RUNTIME_WASM_FILE = "runtime.wasm"
RUNTIME_JS_FILE = "runtime.js"
REPORT_FILE = "report.txt"
DEFAULT_TOP = 15
EXIT_THRESHOLD_FAILED = 2
BOOT_TARGETS = ("D3D12XboxCreateDevice", "PresentX")

# Browser fast-boot Wasm64 audit constants (T68 / T83).
MIB_BYTES = 1024 * 1024
V8_MAX_WASM_MODULE_BYTES = 1024 * 1024 * 1024  # 1 GiB Chromium/V8 single-module cap
DEFAULT_SPLIT_BALANCE_WARN_RATIO = 1.5
WASM_MAGIC = b"\x00asm"
WASM_VERSION_BYTES = b"\x01\x00\x00\x00"
WASM_SECTION_EXPORT = 7
WASM_MAX_SECTION_ID = 13
WASM_EXPORT_KINDS: dict[int, str] = {
    0x00: "func",
    0x01: "table",
    0x02: "memory",
    0x03: "global",
    0x04: "tag",
}
REQUIRED_RUNTIME_EXPORTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("_main", ("_main", "main")),
    (
        "_playgta6_get_input_state_ptr",
        ("_playgta6_get_input_state_ptr", "playgta6_get_input_state_ptr"),
    ),
    (
        "_playgta6_audio_ring",
        ("_playgta6_audio_ring", "playgta6_audio_ring"),
    ),
)

_GUEST_PART_RE = re.compile(r"^guest_part_([A-Za-z0-9_-]+)\.wasm$", re.IGNORECASE)
_SAFE_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_SCHEME_PREFIX_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_ENCODED_TRAVERSAL_RE = re.compile(r"%(?:2e|2f|5c)", re.IGNORECASE)

DISCOVER_JSON_NAMES = (
    "discover.json",
    "discovery.json",
    "cfg.json",
    "summary.json",
    "lift-summary.json",
    "coverage.json",
)

_FILE_URI_RE = re.compile(r"file://[^\s\"'<>|,;()\[\]]+", re.IGNORECASE)
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_MSYS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/[A-Za-z]/(?:Users|users|home|Windows|Program|tmp|var|opt|mnt)/[^\s\"'<>|,;()\[\]]*"
)
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes|srv|run|media|cygdrive)/[^\s\"'<>|,;()\[\]]*"
)
_TILDE_HOME_RE = re.compile(r"(?<![A-Za-z0-9_])~[A-Za-z0-9._-]*/[^\s\"'<>|,;()\[\]]*")
_REPORT_MNEMONIC_RE = re.compile(r"^\s{2,}([A-Za-z][A-Za-z0-9./]+)\s+(\d+)\s*$")


class AuditError(ValueError):
    """Raised when inputs cannot be loaded or classified."""


class _WasmBinaryError(ValueError):
    """Raised when a Wasm binary is truncated, malformed, or invalid."""


# ---------------------------------------------------------------------------
# Sanitization
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
    for env_path_var in ("USERPROFILE", "HOME"):
        raw_home = os.environ.get(env_path_var, "").strip().rstrip("/\\")
        if not raw_home:
            continue
        parts = [p for p in re.split(r"[/\\]+", raw_home) if p]
        if len(parts) >= 2 and parts[-2].lower() in ("users", "home"):
            leaf = parts[-1].strip()
            if len(leaf) >= 3 and leaf not in names:
                names.append(leaf)
    try:
        home_name = Path.home().name.strip()
        if len(home_name) >= 3 and home_name not in names:
            names.append(home_name)
    except (OSError, RuntimeError):
        pass
    return names


def sanitize_text(raw: str | None) -> str:
    """Strip host absolute paths (both `\\` and `/`) and active usernames."""
    if raw is None:
        return ""
    text = _FILE_URI_RE.sub("<path>", str(raw))
    text = _UNC_PATH_RE.sub("<path>", text)
    text = text.replace("\\", "/")
    sanitized = _WINDOWS_PATH_RE.sub("<path>", text)
    sanitized = _UNC_FS_PATH_RE.sub("<path>", sanitized)
    sanitized = _MSYS_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    sanitized = _TILDE_HOME_RE.sub("<path>", sanitized)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        sanitized = pattern.sub("<user>", sanitized)
    return sanitized


def _is_user_home_leaf(parts: Sequence[str]) -> bool:
    if len(parts) < 2:
        return False
    if parts[-2].lower() not in ("users", "home"):
        return False
    if len(parts) == 2:
        return True
    first = parts[0]
    if not first:
        return True
    return len(first) == 2 and first[1] == ":" and first[0].isalpha()


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
    if _is_user_home_leaf(parts):
        return "<user>"
    return sanitize_text(base)


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings; drop raw bytes and host Path objects from reports."""
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, bytearray):
        return f"<bytes:{len(value)}>"
    if isinstance(value, os.PathLike):
        return basename_label(os.fspath(value))
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_value(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Parsing helpers matching recomp JSON field names
# ---------------------------------------------------------------------------


def parse_va(value: Any) -> int | None:
    """Parse a guest VA from an `0x...` string or integer (summary.rs / pipeline.rs)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text, 16) if text.lower().startswith("0x") else int(text, 0)
        except ValueError:
            try:
                return int(text, 16)
            except ValueError:
                return None
    return None


def format_va(value: Any) -> str:
    parsed = parse_va(value)
    if parsed is not None:
        return f"0x{parsed:x}"
    if isinstance(value, str) and value.strip():
        return sanitize_text(value.strip())
    return sanitize_text(str(value))


def as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return int(text, 0)
        except ValueError:
            parsed = parse_va(text)
            return parsed if parsed is not None else default
    return default


def as_float_pct(part: int, whole: int) -> float:
    if whole <= 0:
        return 0.0
    return round(part * 100.0 / whole, 2)


def median(values: Sequence[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _as_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return default
        try:
            return float(text)
        except ValueError:
            return default
    return default


def _pick(mapping: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    if not isinstance(mapping, Mapping):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def _looks_like_bytes(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray))


def is_build_manifest(doc: Mapping[str, Any]) -> bool:
    if doc.get("schema") == SCHEMA_BUILD:
        return True
    return isinstance(doc.get("lift"), Mapping) and isinstance(doc.get("discovery"), Mapping)


def is_program_manifest(doc: Mapping[str, Any]) -> bool:
    if doc.get("schema") == SCHEMA_PROGRAM:
        return True
    if is_build_manifest(doc) or is_guest_manifest(doc):
        return False
    return isinstance(doc.get("modules"), list) and (
        "init_order" in doc or "external_dlls" in doc or "export_thunks" in doc
    )


def is_guest_manifest(doc: Mapping[str, Any]) -> bool:
    if doc.get("schema") in (SCHEMA_BUILD, SCHEMA_PROGRAM):
        return False
    if isinstance(doc.get("lift"), Mapping) and isinstance(doc.get("discovery"), Mapping):
        return False
    if "function_count" in doc and "entry_points" in doc and isinstance(doc.get("functions"), list):
        return False
    return (
        isinstance(doc.get("wasmMetrics"), Mapping)
        or isinstance(doc.get("wasm_metrics"), Mapping)
        or isinstance(doc.get("sideModules"), list)
        or isinstance(doc.get("sideModuleSizes"), list)
        or isinstance(doc.get("guest_dll_modules"), list)
        or ("runtimeWasm" in doc and "schemaVersion" in doc)
    )


def is_discover_json(doc: Mapping[str, Any]) -> bool:
    if is_build_manifest(doc) or is_program_manifest(doc) or is_guest_manifest(doc):
        return False
    functions = doc.get("functions")
    return (
        "function_count" in doc
        and "entry_points" in doc
        and isinstance(functions, list)
    )


# ---------------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------------


def _read_json_file(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"failed to read {basename_label(path)}: {exc.strerror or type(exc).__name__}") from exc
    except UnicodeDecodeError as exc:
        raise AuditError(f"failed to decode {basename_label(path)} as UTF-8 JSON") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuditError(f"invalid JSON in {basename_label(path)}: {exc.msg}") from exc


def _iter_json_files_in_dir(directory: Path) -> list[Path]:
    found: list[Path] = []
    direct = directory / MANIFEST_FILE
    if direct.is_file():
        found.append(direct)
    else:
        try:
            for child in sorted(directory.iterdir()):
                if child.is_dir() and (child / MANIFEST_FILE).is_file():
                    found.append(child / MANIFEST_FILE)
        except OSError:
            pass
    for name in DISCOVER_JSON_NAMES:
        candidate = directory / name
        if candidate.is_file():
            found.append(candidate)
        try:
            for child in sorted(directory.iterdir()):
                nested = child / name
                if child.is_dir() and nested.is_file() and nested not in found:
                    found.append(nested)
        except OSError:
            pass
    # Deduplicate while preserving order.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in found:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def load_inputs(path: str | Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str, str | None]:
    """Load a build-manifest and/or discover JSON from a file or directory.

    Returns `(manifest, discover, source_label, report_text)`.
    """
    target = Path(path)
    manifest: dict[str, Any] | None = None
    discover: dict[str, Any] | None = None
    report_text: str | None = None
    labels: list[str] = []

    files: list[Path]
    if target.is_dir():
        files = _iter_json_files_in_dir(target)
        report_path = target / REPORT_FILE
        if report_path.is_file():
            try:
                report_text = report_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                report_text = None
        if not files and report_text is None:
            try:
                for child in sorted(target.iterdir()):
                    nested = child / REPORT_FILE
                    if child.is_dir() and nested.is_file():
                        report_text = nested.read_text(encoding="utf-8", errors="replace")
                        files = _iter_json_files_in_dir(child)
                        break
            except OSError:
                pass
        labels.append(basename_label(target))
    elif target.is_file():
        files = [target]
        labels.append(basename_label(target))
        sibling_dir = target.parent
        if target.name != MANIFEST_FILE:
            sibling_manifest = sibling_dir / MANIFEST_FILE
            if sibling_manifest.is_file() and sibling_manifest != target:
                files.append(sibling_manifest)
        for name in DISCOVER_JSON_NAMES:
            sibling = sibling_dir / name
            if sibling.is_file() and sibling not in files:
                files.append(sibling)
        report_path = sibling_dir / REPORT_FILE
        if report_path.is_file():
            try:
                report_text = report_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                report_text = None
    else:
        raise AuditError(f"input not found: {basename_label(path)}")

    for file_path in files:
        doc = _read_json_file(file_path)
        if _looks_like_bytes(doc):
            continue
        if not isinstance(doc, dict):
            raise AuditError(f"{basename_label(file_path)} is not a JSON object")
        if is_build_manifest(doc):
            manifest = doc
            labels.append(basename_label(file_path))
        elif is_discover_json(doc):
            discover = doc
            labels.append(basename_label(file_path))
        elif isinstance(doc.get("lift"), Mapping) and not manifest:
            manifest = doc
            labels.append(basename_label(file_path))
        elif isinstance(doc.get("functions"), list) and not discover:
            discover = doc
            labels.append(basename_label(file_path))

    if manifest is None and discover is None:
        raise AuditError(
            "no playgta6-recomp build-manifest.json or discover JSON found"
        )
    source = labels[0] if len(labels) == 1 else "+".join(dict.fromkeys(labels))
    return manifest, discover, sanitize_text(source), report_text


@dataclass
class ModuleArtifact:
    entry: dict[str, Any]
    manifest: dict[str, Any] | None = None
    discover: dict[str, Any] | None = None
    report_text: str | None = None


@dataclass
class AuditInputs:
    source: str
    manifest: dict[str, Any] | None = None
    discover: dict[str, Any] | None = None
    report_text: str | None = None
    program_manifest: dict[str, Any] | None = None
    guest_manifest: dict[str, Any] | None = None
    module_artifacts: list[ModuleArtifact] = field(default_factory=list)


def _safe_resolve_under(base_dir: Path, raw_rel: Any) -> Path | None:
    """Resolve a manifest-supplied relative path safely inside `base_dir`."""
    if not isinstance(raw_rel, str):
        return None
    cleaned = raw_rel.strip().replace("\\", "/")
    if not cleaned:
        return None
    if cleaned.startswith("/") or re.match(r"^[A-Za-z]:", cleaned):
        return None
    parts = [part for part in cleaned.split("/") if part]
    if not parts or any(part == ".." for part in parts):
        return None
    try:
        root = base_dir.resolve()
        candidate = (root / Path(*parts)).resolve()
        if not candidate.is_relative_to(root):
            return None
        return candidate if candidate.is_file() else None
    except OSError:
        return None


def _read_optional_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _load_module_artifacts(base_dir: Path, entry: Mapping[str, Any]) -> ModuleArtifact:
    """Load optional per-module `build-manifest.json`, discover JSON, and `report.txt`."""
    entry_dict = dict(entry)
    manifest_path: Path | None = None
    bm = entry.get("build_manifest")
    if isinstance(bm, Mapping):
        manifest_path = _safe_resolve_under(base_dir, bm.get("path"))
    if manifest_path is None:
        mod_name = entry.get("module")
        if isinstance(mod_name, str) and mod_name.strip():
            manifest_path = _safe_resolve_under(base_dir, f"{mod_name.strip()}/{MANIFEST_FILE}")

    mod_manifest: dict[str, Any] | None = None
    mod_discover: dict[str, Any] | None = None
    mod_report: str | None = None

    module_dir: Path | None = None
    if manifest_path is not None:
        doc = _read_json_file(manifest_path)
        if isinstance(doc, dict) and not _looks_like_bytes(doc):
            mod_manifest = doc
        module_dir = manifest_path.parent
    else:
        mod_name = entry.get("module")
        if isinstance(mod_name, str) and mod_name.strip():
            cleaned = mod_name.strip().replace("\\", "/")
            if not cleaned.startswith("/") and not re.match(r"^[A-Za-z]:", cleaned) and ".." not in cleaned.split("/"):
                try:
                    root = base_dir.resolve()
                    cand_dir = (root / cleaned).resolve()
                    if cand_dir.is_relative_to(root) and cand_dir.is_dir():
                        module_dir = cand_dir
                except OSError:
                    module_dir = None

    if module_dir is not None:
        for name in DISCOVER_JSON_NAMES:
            cand = module_dir / name
            if cand.is_file():
                doc = _read_json_file(cand)
                if isinstance(doc, dict) and not _looks_like_bytes(doc) and (
                    is_discover_json(doc) or isinstance(doc.get("functions"), list)
                ):
                    mod_discover = doc
                    break
        mod_report = _read_optional_text(module_dir / REPORT_FILE)

    return ModuleArtifact(
        entry=entry_dict,
        manifest=mod_manifest,
        discover=mod_discover,
        report_text=mod_report,
    )


def _find_guest_manifest(base_dir: Path, *, check_dist: bool = True) -> tuple[dict[str, Any] | None, str | None]:
    candidates = [base_dir / GUEST_MANIFEST_FILE]
    if check_dist:
        candidates.append(base_dir / "dist" / GUEST_MANIFEST_FILE)
    for cand in candidates:
        if cand.is_file():
            doc = _read_json_file(cand)
            if not isinstance(doc, dict):
                raise AuditError(f"{basename_label(cand)} is not a JSON object")
            return doc, basename_label(cand)
    return None, None


def _load_optional_single_in_dir(
    directory: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None, list[str]]:
    manifest: dict[str, Any] | None = None
    discover: dict[str, Any] | None = None
    labels: list[str] = []
    bm_path = directory / MANIFEST_FILE
    if bm_path.is_file():
        doc = _read_json_file(bm_path)
        if isinstance(doc, dict) and not _looks_like_bytes(doc):
            manifest = doc
            labels.append(basename_label(bm_path))
    for name in DISCOVER_JSON_NAMES:
        cand = directory / name
        if cand.is_file():
            doc = _read_json_file(cand)
            if isinstance(doc, dict) and not _looks_like_bytes(doc) and (
                is_discover_json(doc) or isinstance(doc.get("functions"), list)
            ):
                discover = doc
                labels.append(basename_label(cand))
                break
    report_text = _read_optional_text(directory / REPORT_FILE)
    return manifest, discover, report_text, labels


def load_audit_inputs(path: str | Path) -> AuditInputs:
    """Load single-module, multi-module (`program-manifest.json`), or `guest_manifest.json` inputs."""
    target = Path(path)
    if not target.exists():
        raise AuditError(f"input not found: {basename_label(path)}")

    if target.is_dir():
        prog_path = target / PROGRAM_MANIFEST_FILE
        if prog_path.is_file():
            prog_doc = _read_json_file(prog_path)
            if not isinstance(prog_doc, dict):
                raise AuditError(f"{basename_label(prog_path)} is not a JSON object")
            guest_doc, guest_label = _find_guest_manifest(target, check_dist=True)
            artifacts = [
                _load_module_artifacts(target, entry)
                for entry in (prog_doc.get("modules") or [])
                if isinstance(entry, Mapping)
            ]
            labels = [basename_label(target), basename_label(prog_path)]
            if guest_label:
                labels.append(guest_label)
            source = "+".join(dict.fromkeys(labels))
            return AuditInputs(
                source=sanitize_text(source),
                program_manifest=prog_doc,
                guest_manifest=guest_doc,
                module_artifacts=artifacts,
            )

        guest_path = target / GUEST_MANIFEST_FILE
        if guest_path.is_file():
            guest_doc = _read_json_file(guest_path)
            if not isinstance(guest_doc, dict):
                raise AuditError(f"{basename_label(guest_path)} is not a JSON object")
            manifest, discover, report_text, extra_labels = _load_optional_single_in_dir(target)
            labels = [basename_label(target), basename_label(guest_path), *extra_labels]
            source = "+".join(dict.fromkeys(labels))
            return AuditInputs(
                source=sanitize_text(source),
                manifest=manifest,
                discover=discover,
                report_text=report_text,
                guest_manifest=guest_doc,
            )

        # Preserve single-module & nested `<module>/build-manifest.json` directory scan.
        manifest, discover, source, report_text = load_inputs(target)
        return AuditInputs(
            source=source,
            manifest=manifest,
            discover=discover,
            report_text=report_text,
        )

    if target.is_file():
        doc = _read_json_file(target)
        if _looks_like_bytes(doc) or not isinstance(doc, dict):
            raise AuditError(f"{basename_label(target)} is not a JSON object")

        if is_program_manifest(doc):
            guest_doc, guest_label = _find_guest_manifest(target.parent, check_dist=True)
            artifacts = [
                _load_module_artifacts(target.parent, entry)
                for entry in (doc.get("modules") or [])
                if isinstance(entry, Mapping)
            ]
            labels = [basename_label(target)]
            if guest_label:
                labels.append(guest_label)
            source = labels[0] if len(labels) == 1 else "+".join(dict.fromkeys(labels))
            return AuditInputs(
                source=sanitize_text(source),
                program_manifest=doc,
                guest_manifest=guest_doc,
                module_artifacts=artifacts,
            )

        if is_guest_manifest(doc):
            sibling_prog = target.parent / PROGRAM_MANIFEST_FILE
            if sibling_prog.is_file():
                prog_doc = _read_json_file(sibling_prog)
                if not isinstance(prog_doc, dict):
                    raise AuditError(f"{basename_label(sibling_prog)} is not a JSON object")
                artifacts = [
                    _load_module_artifacts(target.parent, entry)
                    for entry in (prog_doc.get("modules") or [])
                    if isinstance(entry, Mapping)
                ]
                labels = [basename_label(target), basename_label(sibling_prog)]
                return AuditInputs(
                    source=sanitize_text("+".join(dict.fromkeys(labels))),
                    program_manifest=prog_doc,
                    guest_manifest=doc,
                    module_artifacts=artifacts,
                )
            manifest, discover, report_text, extra_labels = _load_optional_single_in_dir(target.parent)
            labels = [basename_label(target), *extra_labels]
            source = labels[0] if len(labels) == 1 else "+".join(dict.fromkeys(labels))
            return AuditInputs(
                source=sanitize_text(source),
                manifest=manifest,
                discover=discover,
                report_text=report_text,
                guest_manifest=doc,
            )

        manifest, discover, source, report_text = load_inputs(target)
        return AuditInputs(
            source=source,
            manifest=manifest,
            discover=discover,
            report_text=report_text,
        )

    raise AuditError(f"input not found: {basename_label(path)}")


def parse_report_mnemonics(report_text: str | None) -> dict[str, int]:
    """Fallback: parse `report.txt` top unsupported mnemonics (pipeline.rs)."""
    counts: dict[str, int] = {}
    if not report_text:
        return counts
    in_section = False
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("top unsupported mnemonics"):
            in_section = True
            continue
        if in_section:
            if not stripped:
                break
            match = _REPORT_MNEMONIC_RE.match(line)
            if not match:
                if stripped.startswith("imports:") or stripped.endswith(":"):
                    break
                continue
            mnemonic, count = match.group(1), int(match.group(2))
            counts[mnemonic.lower()] = counts.get(mnemonic.lower(), 0) + count
    return counts


# ---------------------------------------------------------------------------
# Function graph (discover --json + optional lift overlays)
# ---------------------------------------------------------------------------


def _name_matches_target(name: str, target: str) -> bool:
    left = name.strip()
    right = target.strip()
    if not left or not right:
        return False
    if left.lower() == right.lower():
        return True
    if left.lower().endswith("!" + right.lower()):
        return True
    return right.lower() in left.lower().split("::")[-1].lower()


class FunctionNode:
    __slots__ = (
        "entry",
        "blocks",
        "instructions",
        "callees",
        "named_callees",
        "indirect_calls",
        "unsupported",
        "lifted",
        "name",
    )

    def __init__(self, entry: str) -> None:
        self.entry = entry
        self.blocks = 0
        self.instructions = 0
        self.callees: list[str] = []
        self.named_callees: list[str] = []
        self.indirect_calls = 0
        self.unsupported: dict[str, int] = {}
        self.lifted = True
        self.name: str | None = None


def _add_callee(node: FunctionNode, raw: Any) -> None:
    if raw is None:
        return
    va = parse_va(raw)
    if va is not None:
        label = format_va(va)
        if label not in node.callees:
            node.callees.append(label)
        return
    if isinstance(raw, str) and raw.strip():
        name = sanitize_text(raw.strip())
        if name not in node.named_callees:
            node.named_callees.append(name)


def _parse_unsupported(raw: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            mnemonic = str(key).strip().lower()
            if mnemonic:
                counts[mnemonic] = counts.get(mnemonic, 0) + as_int(value, 0)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                mnemonic = item.strip().lower()
                if mnemonic:
                    counts[mnemonic] = counts.get(mnemonic, 0) + 1
            elif isinstance(item, Mapping):
                mnemonic = str(item.get("mnemonic") or item.get("kind") or "").strip().lower()
                if mnemonic:
                    counts[mnemonic] = counts.get(mnemonic, 0) + max(as_int(item.get("count"), 1), 1)
    return counts


def ingest_discover_functions(discover: Mapping[str, Any] | None) -> dict[str, FunctionNode]:
    nodes: dict[str, FunctionNode] = {}
    if not discover:
        return nodes
    for item in discover.get("functions") or []:
        if not isinstance(item, Mapping):
            continue
        entry_raw = item.get("entry") or item.get("va") or item.get("address")
        va = parse_va(entry_raw)
        if va is None:
            continue
        entry = format_va(va)
        node = nodes.get(entry) or FunctionNode(entry)
        node.blocks = as_int(item.get("blocks"), node.blocks)
        node.instructions = as_int(item.get("instructions"), node.instructions)
        node.indirect_calls = as_int(item.get("indirect_calls"), node.indirect_calls)
        if item.get("name"):
            node.name = sanitize_text(str(item["name"]))
        for field in ("callees", "tail_calls", "calls"):
            values = item.get(field) or []
            if isinstance(values, list):
                for callee in values:
                    _add_callee(node, callee)
        imports = item.get("imports") or item.get("import_callees") or []
        if isinstance(imports, list):
            for imported in imports:
                _add_callee(node, imported)
        unsupported = _parse_unsupported(
            item.get("unsupported_mnemonics")
            or item.get("unsupported_by_mnemonic")
            or item.get("unsupported")
        )
        for mnemonic, count in unsupported.items():
            node.unsupported[mnemonic] = node.unsupported.get(mnemonic, 0) + count
        if "lifted" in item:
            node.lifted = bool(item["lifted"])
        elif node.unsupported:
            node.lifted = False
        nodes[entry] = node
    return nodes


def ingest_tls_and_roots(
    manifest: Mapping[str, Any] | None,
    discover: Mapping[str, Any] | None,
) -> list[str]:
    roots: list[str] = []

    def push(raw: Any) -> None:
        va = parse_va(raw)
        if va is None:
            return
        label = format_va(va)
        if label not in roots:
            roots.append(label)

    if manifest:
        image = manifest.get("image") or {}
        if isinstance(image, Mapping):
            push(image.get("entry_point"))
        options = manifest.get("options") or {}
        if isinstance(options, Mapping):
            for extra in options.get("extra_entries") or []:
                push(extra)
        refinements = manifest.get("refinements") or {}
        if isinstance(refinements, Mapping):
            code_pointers = refinements.get("code_pointers") or {}
            if isinstance(code_pointers, Mapping):
                for accepted in code_pointers.get("accepted") or []:
                    push(accepted)

    if discover:
        tls = discover.get("tls_callbacks") or discover.get("tls") or []
        if isinstance(tls, Mapping):
            tls = tls.get("callbacks") or tls.get("entry_points") or []
        if isinstance(tls, list):
            for cb in tls:
                if isinstance(cb, Mapping):
                    push(cb.get("va") or cb.get("entry") or cb.get("address"))
                else:
                    push(cb)
        entries = discover.get("entry_points") or []
        if isinstance(entries, list):
            # Discover lists PE entry, exports, TLS callbacks together; keep order.
            for entry in entries:
                push(entry)

    return roots


def _count_function_coverage(
    nodes: Mapping[str, FunctionNode],
    discovered_functions: int,
    lifted_fn_count: int,
    address_taken: int,
    has_lift_data: bool,
) -> tuple[int, int]:
    """Return `(lifted_functions, unlifted_functions)`.

    Per-function `unsupported_mnemonics` mark a function as unlifted even when
    R9 emitted IR for every discovered CFG (traps at run time). Discover-only
    inputs with no lift section count as fully unlifted.
    """
    unlifted_from_nodes = sum(1 for n in nodes.values() if n.unsupported or not n.lifted)
    if nodes and unlifted_from_nodes:
        lifted = max(discovered_functions - unlifted_from_nodes, 0)
        return lifted, unlifted_from_nodes + address_taken
    if has_lift_data:
        lifted = lifted_fn_count
        unlifted = max(discovered_functions - lifted, 0) + address_taken
        return max(lifted, 0), max(unlifted, 0)
    return 0, max(discovered_functions, len(nodes)) + address_taken


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _mnemonic_rows(
    counts: Mapping[str, int],
    affected: Mapping[str, set[str]],
    top: int,
) -> list[dict[str, Any]]:
    rows: list[tuple[str, int, int]] = []
    keys = set(counts) | set(affected)
    for mnemonic in keys:
        count = int(counts.get(mnemonic, 0))
        funcs = len(affected.get(mnemonic, ()))
        if count <= 0 and funcs <= 0:
            continue
        if count <= 0:
            count = funcs
        rows.append((mnemonic, count, funcs))
    rows.sort(key=lambda row: (-row[1], -row[2], row[0]))
    limited = rows[: max(top, 0)]
    return [
        {
            "mnemonic": name,
            "count": count,
            "affected_functions": funcs,
        }
        for name, count, funcs in limited
    ]


def _shard_sizes(manifest: Mapping[str, Any] | None) -> list[int]:
    if not manifest:
        return []
    lift = manifest.get("lift") or {}
    if not isinstance(lift, Mapping):
        return []
    shards = lift.get("shards") or []
    if not isinstance(shards, list) or not shards:
        return []
    sizes: list[int] = []
    for shard in shards:
        if isinstance(shard, Mapping):
            sizes.append(as_int(shard.get("instructions"), 0))
        elif isinstance(shard, (int, float)):
            sizes.append(int(shard))
    return sizes


def compute_shard_balance_summary(shard_sizes_bytes: Sequence[int]) -> dict[str, Any]:
    """Compute deterministic fast-boot shard balance summary metrics (T83).

    Returns a sanitized summary with:
    - `shard_count`: number of shards (`int`, >= 0)
    - `max_shard_mib`: largest shard size in MiB (`float`, 4 decimals)
    - `total_mib`: sum of shard sizes in MiB (`float`, 4 decimals)
    - `balance_ratio`: `max_bytes / mean_bytes` (`float`, 4 decimals;
      0.0 when `shard_count == 0` or `total_bytes == 0`, zero-division safe).
    """
    sizes = [max(as_int(sz, 0), 0) for sz in shard_sizes_bytes]
    count = len(sizes)
    if count == 0:
        return {
            "shard_count": 0,
            "max_shard_mib": 0.0,
            "total_mib": 0.0,
            "balance_ratio": 0.0,
        }
    total_bytes = sum(sizes)
    max_bytes = max(sizes)
    mean_bytes = total_bytes / count
    balance_ratio = round(max_bytes / mean_bytes, 4) if mean_bytes > 0.0 else 0.0
    return {
        "shard_count": count,
        "max_shard_mib": round(max_bytes / MIB_BYTES, 4),
        "total_mib": round(total_bytes / MIB_BYTES, 4),
        "balance_ratio": balance_ratio,
    }


def _shard_stats_from_sizes(sizes: Sequence[int], unit: str = "instructions") -> dict[str, Any]:
    if not sizes:
        return {
            "count": 0,
            "min": 0,
            "median": 0.0,
            "max": 0,
            "balance_ratio": 0.0,
            "unit": unit,
        }
    total = sum(sizes)
    max_val = max(sizes)
    mean_val = total / len(sizes)
    balance_ratio = round(max_val / mean_val, 4) if mean_val > 0.0 else 0.0
    return {
        "count": len(sizes),
        "min": min(sizes),
        "median": median(sizes),
        "max": max_val,
        "balance_ratio": balance_ratio,
        "unit": unit,
    }


def _shard_stats(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    return _shard_stats_from_sizes(_shard_sizes(manifest), unit="instructions")


def _jump_tables(
    manifest: Mapping[str, Any] | None,
    discover: Mapping[str, Any] | None,
) -> dict[str, Any]:
    resolved = 0
    unresolved = 0
    source = None
    if discover and isinstance(discover.get("jump_tables"), Mapping):
        source = discover["jump_tables"]
    elif manifest:
        discovery = manifest.get("discovery") or {}
        if isinstance(discovery, Mapping) and isinstance(discovery.get("jump_tables"), Mapping):
            source = discovery["jump_tables"]
    if isinstance(source, Mapping):
        resolved = as_int(source.get("resolved"), 0)
        unresolved = as_int(source.get("unresolved"), 0)
    total = resolved + unresolved
    return {
        "resolved_jump_tables": resolved,
        "unresolved_indirect_jumps": unresolved,
        "resolution_pct": as_float_pct(resolved, total),
    }


def boot_bfs(
    nodes: Mapping[str, FunctionNode],
    roots: Sequence[str],
    boot_depth: int | None,
) -> tuple[list[str], dict[str, list[str]], int]:
    """BFS over direct/tail callees. `boot_depth` is max edge distance from a root."""
    reachable: list[str] = []
    parents: dict[str, str | None] = {}
    paths: dict[str, list[str]] = {}
    indirect = 0
    queue: deque[tuple[str, int]] = deque()
    seen: set[str] = set()

    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        queue.append((root, 0))
        parents[root] = None
        paths[root] = [root]

    while queue:
        current, depth = queue.popleft()
        reachable.append(current)
        node = nodes.get(current)
        if node is None:
            continue
        indirect += node.indirect_calls
        if boot_depth is not None and depth >= boot_depth:
            continue
        for callee in node.callees:
            if callee in seen:
                continue
            seen.add(callee)
            parents[callee] = current
            paths[callee] = paths[current] + [callee]
            queue.append((callee, depth + 1))
        for name in node.named_callees:
            if name in seen:
                continue
            seen.add(name)
            parents[name] = current
            paths[name] = paths[current] + [name]
            queue.append((name, depth + 1))

    return reachable, paths, indirect


def _shortest_boot_targets(
    nodes: Mapping[str, FunctionNode],
    paths: Mapping[str, list[str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    named_nodes = list(paths.keys())
    for target in BOOT_TARGETS:
        match: str | None = None
        if target in paths:
            match = target
        else:
            for key in named_nodes:
                if _name_matches_target(key, target):
                    match = key
                    break
            if match is None:
                for node in nodes.values():
                    for name in node.named_callees:
                        if _name_matches_target(name, target) and node.entry in paths:
                            match = name
                            path = list(paths[node.entry]) + [name]
                            result[target] = {
                                "reachable": True,
                                "via": name,
                                "shortest_path": path,
                            }
                            break
                    if target in result:
                        break
                if target in result:
                    continue
        if match is not None and match in paths:
            result[target] = {
                "reachable": True,
                "via": match,
                "shortest_path": list(paths[match]),
            }
        else:
            result[target] = {"reachable": False, "via": None, "shortest_path": []}
    return result


def _recommendations(
    coverage: Mapping[str, Any],
    boot: Mapping[str, Any],
    jump: Mapping[str, Any],
    mnemonics: Sequence[Mapping[str, Any]],
    unresolved_imports: Sequence[str],
    address_taken_not_lifted: int,
) -> list[dict[str, Any]]:
    recs: list[dict[str, Any]] = []

    def add(category: str, detail: str) -> None:
        recs.append({"priority": len(recs) + 1, "category": category, "detail": detail})

    boot_pct = float(boot.get("boot_lifted_pct") or 0.0)
    if boot.get("boot_reachable_functions") and boot_pct < 100.0:
        add(
            "boot_lift",
            f"Boot-path lift coverage is {boot_pct:.2f}%; lift remaining reachable functions before splash.",
        )
    boot_mnems = boot.get("boot_unlifted_mnemonics") or []
    if boot_mnems:
        top = boot_mnems[0]
        add(
            "boot_mnemonic",
            f"Implement lifter support for `{top['mnemonic']}` "
            f"({top['count']} hits on the boot path, {top['affected_functions']} functions).",
        )
    missing_targets = [
        name
        for name, info in (boot.get("boot_targets") or {}).items()
        if not (info or {}).get("reachable")
    ]
    if missing_targets:
        add(
            "boot_gpu_imports",
            "No call-graph path to "
            + " / ".join(missing_targets)
            + "; wire import thunks and confirm IAT calls from the boot chain.",
        )
    else:
        reachable_targets = [
            name
            for name, info in (boot.get("boot_targets") or {}).items()
            if (info or {}).get("reachable")
        ]
        if reachable_targets:
            add(
                "boot_gpu_imports",
                "Boot path reaches "
                + " / ".join(reachable_targets)
                + "; keep those d3d12_xs shims complete for PresentX splash.",
            )
    if as_int(boot.get("boot_unresolved_indirect_calls"), 0) > 0:
        add(
            "boot_indirect",
            f"{boot['boot_unresolved_indirect_calls']} unresolved indirect calls on the boot path "
            "(IAT/vcall); resolve slots or add dispatcher shims.",
        )
    if float(jump.get("resolution_pct") or 0.0) < 100.0 and (
        as_int(jump.get("unresolved_indirect_jumps"), 0) > 0
    ):
        add(
            "jump_tables",
            f"{jump['unresolved_indirect_jumps']} unresolved indirect jumps remain "
            f"({jump['resolution_pct']:.2f}% tables resolved); improve switch-table recovery.",
        )
    if mnemonics and not boot_mnems:
        top = mnemonics[0]
        add(
            "lifter_mnemonic",
            f"Highest remaining unlifted mnemonic is `{top['mnemonic']}` "
            f"({top['count']} occurrences, {top['affected_functions']} functions).",
        )
    if unresolved_imports:
        preview = ", ".join(unresolved_imports[:6])
        extra = "" if len(unresolved_imports) <= 6 else f" (+{len(unresolved_imports) - 6} more)"
        add(
            "unresolved_imports",
            f"Bind remaining unresolved imports referenced by discovered code: {preview}{extra}.",
        )
    if address_taken_not_lifted > 0:
        add(
            "address_taken",
            f"{address_taken_not_lifted} address-taken code pointers are not lifted; "
            "seed them as extra `--entry` points.",
        )
    if float(coverage.get("lifted_pct") or 0.0) < 100.0 and not recs:
        add(
            "coverage",
            f"Overall function lift coverage is {coverage['lifted_pct']:.2f}%; "
            "expand the lifter for remaining unlifted functions.",
        )
    if not recs:
        add(
            "ready",
            "No remaining lift/boot blockers in this report; run the guest image against the splash path.",
        )
    return recs


def _unresolved_import_names(manifest: Mapping[str, Any] | None) -> list[str]:
    names: list[str] = []
    if not manifest:
        return names
    refinements = manifest.get("refinements") or {}
    if not isinstance(refinements, Mapping):
        return names
    slots = refinements.get("import_slots") or {}
    if not isinstance(slots, Mapping):
        return names
    for item in slots.get("referenced_unresolved") or []:
        if isinstance(item, Mapping):
            imported = sanitize_text(str(item.get("import") or "").strip())
            raw_dll = str(item.get("dll") or "").strip()
            dll = basename_label(raw_dll) if raw_dll else ""
            if dll == "<unknown>":
                dll = ""
            if imported and dll:
                label = f"{dll}!{imported}"
            else:
                label = imported or dll
            if label and label not in names:
                names.append(sanitize_text(label))
    return names


def _format_program_import_label(item: Mapping[str, Any]) -> str:
    raw_dll = item.get("dll")
    dll = basename_label(raw_dll) if raw_dll else ""
    if dll == "<unknown>":
        dll = ""
    name = sanitize_text(str(item.get("name") or "").strip())
    ordinal = item.get("ordinal")
    has_ord = ordinal is not None and not isinstance(ordinal, bool)
    if name and dll:
        return f"{dll}!{name}"
    if has_ord and dll:
        return f"{dll}!#{as_int(ordinal, 0)}"
    if name:
        return name
    if has_ord:
        return f"#{as_int(ordinal, 0)}"
    return dll


def _extract_program_indirect_dispatch(program_manifest: Mapping[str, Any] | None) -> str | None:
    if not isinstance(program_manifest, Mapping):
        return None
    options = program_manifest.get("options")
    if isinstance(options, Mapping):
        build_opts = options.get("build")
        if isinstance(build_opts, Mapping):
            val = _pick(build_opts, "indirect_dispatch", "indirectDispatch")
            if isinstance(val, str) and val.strip():
                return sanitize_text(val.strip())
        val = _pick(options, "indirect_dispatch", "indirectDispatch")
        if isinstance(val, str) and val.strip():
            return sanitize_text(val.strip())
    top_val = _pick(program_manifest, "indirect_dispatch", "indirectDispatch")
    if isinstance(top_val, str) and top_val.strip():
        return sanitize_text(top_val.strip())
    for mod in program_manifest.get("modules") or []:
        if isinstance(mod, Mapping):
            reg = mod.get("registration")
            if isinstance(reg, Mapping) and reg.get("shard_dispatch_symbol"):
                return "shard"
    return None


def _summarize_wasm_footprint(
    guest_manifest: Mapping[str, Any] | None,
    *,
    program_indirect_dispatch: str | None = None,
    total_shards: int = 0,
    total_lifted_functions: int = 0,
) -> dict[str, Any]:
    """Summarize Wasm module sizes, `wasmTable`, and `GOT.func` footprint."""
    if not isinstance(guest_manifest, Mapping):
        dispatch = program_indirect_dispatch
        estimated = (
            total_shards
            if dispatch == "shard" and total_shards > 0
            else total_lifted_functions
        )
        return {
            "available": False,
            "indirect_dispatch": dispatch,
            "split_modules": 0,
            "split_strategy": None,
            "side_module_count": 0,
            "side_module_shard_count": 0,
            "guest_dll_modules": [],
            "total_wasm_bytes": None,
            "total_size_mib": None,
            "total_func_imports": None,
            "total_got_func_imports": None,
            "total_got_mem_imports": None,
            "total_func_exports": None,
            "total_table_size": None,
            "main_table_size": None,
            "side_table_size": None,
            "shard_dispatch_exports": None,
            "shard_table_exports": None,
            "got_func_sub_imports": None,
            "got_func_shard_dispatch_imports": None,
            "estimated_got_func_entries": estimated,
            "any_exceeds_v8_1gib_limit": False,
            "shard_balance": compute_shard_balance_summary([]),
            "modules": [],
            "side_modules": [],
        }

    wasm_metrics = _pick(guest_manifest, "wasmMetrics", "wasm_metrics")
    wasm_metrics = wasm_metrics if isinstance(wasm_metrics, Mapping) else {}
    totals = wasm_metrics.get("totals")
    totals = totals if isinstance(totals, Mapping) else {}

    raw_dispatch = _pick(
        wasm_metrics,
        "indirectDispatch",
        "indirect_dispatch",
        default=_pick(
            guest_manifest,
            "indirectDispatch",
            "indirect_dispatch",
            default=program_indirect_dispatch,
        ),
    )
    indirect_dispatch = (
        sanitize_text(str(raw_dispatch).strip())
        if isinstance(raw_dispatch, str) and raw_dispatch.strip()
        else None
    )

    raw_strategy = _pick(guest_manifest, "splitStrategy", "split_strategy")
    split_strategy = (
        sanitize_text(str(raw_strategy).strip())
        if isinstance(raw_strategy, str) and raw_strategy.strip()
        else None
    )
    split_modules = as_int(_pick(guest_manifest, "splitModules", "split_modules"), 0)

    guest_dll_modules: list[str] = []
    raw_guest_dlls = _pick(guest_manifest, "guest_dll_modules", "guestDllModules") or []
    if isinstance(raw_guest_dlls, list):
        for item in raw_guest_dlls:
            if isinstance(item, str) and item.strip():
                label = basename_label(item)
                if label != "<unknown>" and label not in guest_dll_modules:
                    guest_dll_modules.append(label)

    side_modules: list[dict[str, Any]] = []
    raw_side_sizes = _pick(guest_manifest, "sideModuleSizes", "side_module_sizes") or []
    if isinstance(raw_side_sizes, list) and raw_side_sizes:
        for idx, item in enumerate(raw_side_sizes):
            if not isinstance(item, Mapping):
                continue
            shards_list = _pick(item, "shards") or []
            default_sc = len(shards_list) if isinstance(shards_list, list) else 0
            wasm_bytes = as_int(_pick(item, "wasmBytes", "wasm_bytes"), 0)
            size_mib = _as_float(
                _pick(item, "sizeMiB", "size_mib"),
                round(wasm_bytes / (1024 * 1024), 4) if wasm_bytes else 0.0,
            )
            side_modules.append(
                {
                    "index": as_int(_pick(item, "index"), idx),
                    "name": basename_label(_pick(item, "name")),
                    "role": sanitize_text(str(_pick(item, "role", default="shards"))),
                    "shard_count": as_int(_pick(item, "shardCount", "shard_count"), default_sc),
                    "wasm_bytes": wasm_bytes,
                    "size_mib": round(size_mib, 4),
                    "exceeds_v8_1gib_limit": bool(
                        _pick(item, "exceedsV8_1GiBLimit", "exceeds_v8_1gib_limit", default=False)
                    ),
                }
            )
    else:
        raw_side_names = _pick(guest_manifest, "sideModules", "side_modules") or []
        if isinstance(raw_side_names, list):
            for idx, item in enumerate(raw_side_names):
                if isinstance(item, str) and item.strip():
                    side_modules.append(
                        {
                            "index": idx,
                            "name": basename_label(item),
                            "role": "shards",
                            "shard_count": 0,
                            "wasm_bytes": 0,
                            "size_mib": 0.0,
                            "exceeds_v8_1gib_limit": False,
                        }
                    )

    wasm_mod_rows: list[dict[str, Any]] = []
    raw_wasm_mods = wasm_metrics.get("modules") or []
    if isinstance(raw_wasm_mods, list):
        for item in raw_wasm_mods:
            if not isinstance(item, Mapping):
                continue
            wasm_bytes = as_int(_pick(item, "wasmBytes", "wasm_bytes"), 0)
            size_mib = _as_float(
                _pick(item, "sizeMiB", "size_mib"),
                round(wasm_bytes / (1024 * 1024), 4) if wasm_bytes else 0.0,
            )
            dylink_raw = _pick(item, "dylinkTableSize", "dylink_table_size")
            wasm_mod_rows.append(
                {
                    "name": basename_label(_pick(item, "name")),
                    "role": sanitize_text(str(_pick(item, "role", default="main"))),
                    "wasm_bytes": wasm_bytes,
                    "size_mib": round(size_mib, 4),
                    "func_imports": as_int(_pick(item, "funcImports", "func_imports"), 0),
                    "got_func_imports": as_int(_pick(item, "gotFuncImports", "got_func_imports"), 0),
                    "got_mem_imports": as_int(_pick(item, "gotMemImports", "got_mem_imports"), 0),
                    "func_exports": as_int(_pick(item, "funcExports", "func_exports"), 0),
                    "table_size": as_int(_pick(item, "tableSize", "table_size"), 0),
                    "dylink_table_size": as_int(dylink_raw, 0) if dylink_raw is not None else None,
                    "shard_dispatch_exports": as_int(
                        _pick(item, "shardDispatchExports", "shard_dispatch_exports"), 0
                    ),
                    "shard_table_exports": as_int(
                        _pick(item, "shardTableExports", "shard_table_exports"), 0
                    ),
                    "got_func_sub_imports": as_int(
                        _pick(item, "gotFuncSubImports", "got_func_sub_imports"), 0
                    ),
                    "got_func_shard_dispatch_imports": as_int(
                        _pick(
                            item,
                            "gotFuncShardDispatchImports",
                            "got_func_shard_dispatch_imports",
                        ),
                        0,
                    ),
                }
            )

    side_module_count = as_int(
        _pick(totals, "sideModuleCount", "side_module_count"),
        sum(1 for m in wasm_mod_rows if m["role"] == "side") or len(side_modules),
    )
    side_module_shard_count = sum(s["shard_count"] for s in side_modules)
    total_wasm_bytes = as_int(
        _pick(totals, "wasmBytes", "wasm_bytes"),
        sum(m["wasm_bytes"] for m in wasm_mod_rows) or sum(s["wasm_bytes"] for s in side_modules),
    )
    total_size_mib = round(
        _as_float(
            _pick(totals, "sizeMiB", "size_mib"),
            total_wasm_bytes / (1024 * 1024) if total_wasm_bytes else 0.0,
        ),
        4,
    )
    total_func_imports = as_int(
        _pick(totals, "funcImports", "func_imports"),
        sum(m["func_imports"] for m in wasm_mod_rows),
    )
    total_got_func_imports = as_int(
        _pick(totals, "gotFuncImports", "got_func_imports"),
        sum(m["got_func_imports"] for m in wasm_mod_rows),
    )
    total_got_mem_imports = as_int(
        _pick(totals, "gotMemImports", "got_mem_imports"),
        sum(m["got_mem_imports"] for m in wasm_mod_rows),
    )
    total_func_exports = as_int(
        _pick(totals, "funcExports", "func_exports"),
        sum(m["func_exports"] for m in wasm_mod_rows),
    )
    main_table_size = as_int(
        _pick(totals, "mainTableSize", "main_table_size"),
        sum(m["table_size"] for m in wasm_mod_rows if m["role"] == "main"),
    )
    side_table_size = as_int(
        _pick(totals, "sideTableSize", "side_table_size"),
        sum(m["table_size"] for m in wasm_mod_rows if m["role"] == "side"),
    )
    total_table_size = as_int(
        _pick(totals, "tableSize", "table_size"),
        main_table_size + side_table_size,
    )
    shard_dispatch_exports = as_int(
        _pick(totals, "shardDispatchExports", "shard_dispatch_exports"),
        sum(m["shard_dispatch_exports"] for m in wasm_mod_rows),
    )
    shard_table_exports = as_int(
        _pick(totals, "shardTableExports", "shard_table_exports"),
        sum(m["shard_table_exports"] for m in wasm_mod_rows),
    )
    got_func_sub_imports = as_int(
        _pick(totals, "gotFuncSubImports", "got_func_sub_imports"),
        sum(m["got_func_sub_imports"] for m in wasm_mod_rows),
    )
    got_func_shard_dispatch_imports = as_int(
        _pick(totals, "gotFuncShardDispatchImports", "got_func_shard_dispatch_imports"),
        sum(m["got_func_shard_dispatch_imports"] for m in wasm_mod_rows),
    )
    any_exceeds_v8 = any(s["exceeds_v8_1gib_limit"] for s in side_modules)
    side_wasm_bytes = [s["wasm_bytes"] for s in side_modules]
    if not any(b > 0 for b in side_wasm_bytes):
        side_row_bytes = [m["wasm_bytes"] for m in wasm_mod_rows if m["role"] == "side"]
        if side_row_bytes:
            side_wasm_bytes = side_row_bytes
    shard_balance = compute_shard_balance_summary(side_wasm_bytes)

    return {
        "available": True,
        "indirect_dispatch": indirect_dispatch,
        "split_modules": split_modules,
        "split_strategy": split_strategy,
        "side_module_count": side_module_count,
        "side_module_shard_count": side_module_shard_count,
        "guest_dll_modules": guest_dll_modules,
        "total_wasm_bytes": total_wasm_bytes,
        "total_size_mib": total_size_mib,
        "total_func_imports": total_func_imports,
        "total_got_func_imports": total_got_func_imports,
        "total_got_mem_imports": total_got_mem_imports,
        "total_func_exports": total_func_exports,
        "total_table_size": total_table_size,
        "main_table_size": main_table_size,
        "side_table_size": side_table_size,
        "shard_dispatch_exports": shard_dispatch_exports,
        "shard_table_exports": shard_table_exports,
        "got_func_sub_imports": got_func_sub_imports,
        "got_func_shard_dispatch_imports": got_func_shard_dispatch_imports,
        "estimated_got_func_entries": total_got_func_imports,
        "any_exceeds_v8_1gib_limit": any_exceeds_v8,
        "shard_balance": shard_balance,
        "modules": wasm_mod_rows,
        "side_modules": side_modules,
    }


def _append_multimodule_recommendations(
    recs: list[dict[str, Any]],
    program: Mapping[str, Any] | None,
    wasm_footprint: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    extra: list[tuple[str, str]] = []
    if wasm_footprint:
        if wasm_footprint.get("any_exceeds_v8_1gib_limit"):
            extra.append(
                (
                    "wasm_size",
                    "One or more Wasm side modules exceed the 1 GiB V8 module limit; "
                    "increase --split-modules or lower --max-side-module-mib.",
                )
            )
        if wasm_footprint.get("available"):
            got_func = as_int(wasm_footprint.get("total_got_func_imports"), 0)
            if got_func > 1000 and wasm_footprint.get("indirect_dispatch") != "shard":
                extra.append(
                    (
                        "wasm_got_func",
                        f"Measured GOT.func footprint is {got_func} imports without shard dispatch; "
                        "enable --indirect-dispatch shard to shrink wasmTable and GOT.func overhead.",
                    )
                )
    if program:
        cycles = program.get("cycles") or []
        if cycles:
            extra.append(
                (
                    "module_cycles",
                    f"Program dependency graph contains {len(cycles)} circular guest DLL edge(s); "
                    "verify init_order and DllMain reentrancy.",
                )
            )
        prog_imports = program.get("imports") or {}
        host_unres = as_int(prog_imports.get("host_unresolved"), 0)
        if host_unres > 0 and not any(r.get("category") == "unresolved_imports" for r in recs):
            extra.append(
                (
                    "unresolved_imports",
                    f"{host_unres} unresolved host import(s) remain across program modules; "
                    "bind shims or record expected exclusions.",
                )
            )
    if not extra:
        return recs
    filtered = [dict(r) for r in recs if r.get("category") != "ready"]
    for category, detail in extra:
        filtered.append({"priority": len(filtered) + 1, "category": category, "detail": detail})
    for idx, rec in enumerate(filtered, start=1):
        rec["priority"] = idx
    return filtered


def _audit_single_module_core(
    manifest: Mapping[str, Any] | None,
    discover: Mapping[str, Any] | None,
    source: str,
    report_text: str | None,
    *,
    top: int,
    boot_depth: int | None,
) -> dict[str, Any]:
    nodes = ingest_discover_functions(discover)
    roots = ingest_tls_and_roots(manifest, discover)

    discovery = (manifest or {}).get("discovery") if manifest else None
    lift = (manifest or {}).get("lift") if manifest else None
    discovery = discovery if isinstance(discovery, Mapping) else {}
    lift = lift if isinstance(lift, Mapping) else {}

    discovered_functions = as_int(
        discovery.get("functions"),
        as_int((discover or {}).get("function_count"), len(nodes)),
    )
    if not discovered_functions:
        discovered_functions = len(nodes)

    lifted_fn_count = as_int(lift.get("functions"), 0)
    address_taken = 0
    refinements = (manifest or {}).get("refinements") if manifest else None
    if isinstance(refinements, Mapping):
        dispatch = refinements.get("dispatch") or {}
        if isinstance(dispatch, Mapping):
            not_lifted = dispatch.get("address_taken_not_lifted") or []
            if isinstance(not_lifted, list):
                address_taken = len(not_lifted)

    for node in nodes.values():
        if node.unsupported:
            node.lifted = False

    lifted_functions, unlifted_functions = _count_function_coverage(
        nodes,
        discovered_functions,
        lifted_fn_count,
        address_taken,
        has_lift_data=bool(lift),
    )

    total_blocks = as_int(
        discovery.get("blocks"),
        as_int((discover or {}).get("block_count"), sum(n.blocks for n in nodes.values())),
    )
    total_instructions = as_int(
        lift.get("instructions"),
        as_int(
            discovery.get("instructions"),
            as_int((discover or {}).get("instruction_count"), sum(n.instructions for n in nodes.values())),
        ),
    )
    unsupported_instructions = as_int(lift.get("unsupported_instructions"), 0)
    if not unsupported_instructions:
        unsupported_instructions = sum(sum(n.unsupported.values()) for n in nodes.values())
    lifted_instructions = max(total_instructions - unsupported_instructions, 0)
    # Blocks: without per-block lift data, treat unsupported functions' blocks as unlifted.
    unlifted_blocks = sum(n.blocks for n in nodes.values() if not n.lifted)
    lifted_blocks = max(total_blocks - unlifted_blocks, 0) if total_blocks else 0

    coverage = {
        "lifted_functions": lifted_functions,
        "unlifted_functions": unlifted_functions,
        "discovered_functions": discovered_functions,
        "lifted_pct": as_float_pct(lifted_functions, discovered_functions),
        "lifted_blocks": lifted_blocks,
        "unlifted_blocks": unlifted_blocks,
        "block_lifted_pct": as_float_pct(lifted_blocks, total_blocks),
        "lifted_instructions": lifted_instructions,
        "unlifted_instructions": unsupported_instructions,
        "instruction_lifted_pct": as_float_pct(lifted_instructions, total_instructions),
        "unsupported_instructions": unsupported_instructions,
    }

    mnemonic_counts: dict[str, int] = {}
    mnemonic_funcs: dict[str, set[str]] = defaultdict(set)
    for node in nodes.values():
        for mnemonic, count in node.unsupported.items():
            mnemonic_counts[mnemonic] = mnemonic_counts.get(mnemonic, 0) + count
            mnemonic_funcs[mnemonic].add(node.entry)
    if not mnemonic_counts:
        by_mnemonic = lift.get("unsupported_by_mnemonic") if lift else None
        if isinstance(by_mnemonic, Mapping):
            for key, value in by_mnemonic.items():
                mnemonic = str(key).strip().lower()
                if mnemonic:
                    mnemonic_counts[mnemonic] = mnemonic_counts.get(mnemonic, 0) + as_int(value, 0)
    if not mnemonic_counts:
        for mnemonic, count in parse_report_mnemonics(report_text).items():
            mnemonic_counts[mnemonic] = mnemonic_counts.get(mnemonic, 0) + count

    mnemonics = _mnemonic_rows(mnemonic_counts, mnemonic_funcs, top)
    shards = _shard_stats(manifest)
    jump = _jump_tables(manifest, discover)

    if not roots and nodes:
        roots = list(nodes.keys())[:1]

    reachable, paths, boot_indirect = boot_bfs(nodes, roots, boot_depth)
    boot_nodes = [nodes[e] for e in reachable if e in nodes]
    if boot_nodes:
        boot_unlifted = sum(1 for n in boot_nodes if not n.lifted)
        boot_lifted = len(boot_nodes) - boot_unlifted
        boot_lifted_pct = as_float_pct(boot_lifted, len(boot_nodes))
    elif reachable:
        boot_lifted_pct = coverage["lifted_pct"]
        boot_lifted = 0
        boot_unlifted = 0
    else:
        boot_lifted_pct = coverage["lifted_pct"]
        boot_lifted = 0
        boot_unlifted = 0

    boot_mnem_counts: dict[str, int] = {}
    boot_mnem_funcs: dict[str, set[str]] = defaultdict(set)
    for node in boot_nodes:
        for mnemonic, count in node.unsupported.items():
            boot_mnem_counts[mnemonic] = boot_mnem_counts.get(mnemonic, 0) + count
            boot_mnem_funcs[mnemonic].add(node.entry)

    boot_targets = _shortest_boot_targets(nodes, paths)
    boot = {
        "roots": roots,
        "boot_depth": boot_depth,
        "boot_reachable_functions": len(reachable) if (nodes or roots) else discovered_functions,
        "boot_lifted_functions": boot_lifted if boot_nodes else lifted_functions,
        "boot_unlifted_functions": boot_unlifted if boot_nodes else unlifted_functions,
        "boot_lifted_pct": boot_lifted_pct if boot_nodes else coverage["lifted_pct"],
        "boot_unlifted_mnemonics": _mnemonic_rows(boot_mnem_counts, boot_mnem_funcs, top),
        "boot_unresolved_indirect_calls": boot_indirect,
        "boot_targets": boot_targets,
        "reachable_entries": reachable,
    }

    unresolved_imports = _unresolved_import_names(manifest)
    recs = _recommendations(
        coverage,
        boot,
        jump,
        mnemonics,
        unresolved_imports,
        address_taken,
    )

    module = ""
    if manifest and isinstance(manifest.get("module"), str):
        raw_module = manifest["module"].strip()
        if raw_module:
            module = (
                basename_label(raw_module)
                if ("/" in raw_module or "\\" in raw_module or re.match(r"^[A-Za-z]:", raw_module))
                else sanitize_text(raw_module)
            )

    return {
        "schema": SCHEMA_AUDIT,
        "schema_version": SCHEMA_AUDIT_VERSION,
        "source": source,
        "module": module,
        "coverage": coverage,
        "shards": shards,
        "unlifted_mnemonics": mnemonics,
        "jump_tables": jump,
        "boot": boot,
        "unresolved_imports": unresolved_imports,
        "address_taken_not_lifted": address_taken,
        "recommendations": recs,
    }


def _audit_program_inputs(
    inputs: AuditInputs,
    *,
    top: int,
    boot_depth: int | None,
) -> dict[str, Any]:
    program_manifest = inputs.program_manifest or {}
    guest_manifest = inputs.guest_manifest

    # Group program-level imports by importer (case-insensitive across dll/file/module)
    # and collect program-wide unresolved import breakdown.
    program_imports = program_manifest.get("imports") or []
    unresolved_by_importer: dict[str, list[str]] = defaultdict(list)
    unresolved_by_dll_map: dict[str, list[str]] = defaultdict(list)
    unresolved_by_dll_counts: dict[str, int] = defaultdict(int)
    program_unresolved_labels: list[str] = []
    program_import_entries_unresolved = 0
    unresolved_recorded = 0
    unresolved_unrecorded = 0

    for item in program_imports:
        if not isinstance(item, Mapping):
            continue
        binding = str(item.get("binding") or "").strip().lower()
        if binding != "unresolved":
            continue
        program_import_entries_unresolved += 1
        if item.get("recorded"):
            unresolved_recorded += 1
        else:
            unresolved_unrecorded += 1
        label = _format_program_import_label(item)
        dll_label = basename_label(item.get("dll")) if item.get("dll") else "<unknown>"
        unresolved_by_dll_counts[dll_label] += 1
        if label:
            if label not in program_unresolved_labels:
                program_unresolved_labels.append(label)
            if label not in unresolved_by_dll_map[dll_label]:
                unresolved_by_dll_map[dll_label].append(label)
            importer_raw = item.get("importer")
            if isinstance(importer_raw, str) and importer_raw.strip():
                imp_key = basename_label(importer_raw).lower()
                if label not in unresolved_by_importer[imp_key]:
                    unresolved_by_importer[imp_key].append(label)

    # Aggregate across every module listed in program-manifest.json.
    module_rows: list[dict[str, Any]] = []
    all_nodes: dict[str, FunctionNode] = {}
    all_roots: list[str] = []
    all_shard_sizes: list[int] = []
    total_discovered_fns = 0
    total_lifted_fns = 0
    total_unlifted_fns = 0
    total_lifted_blocks = 0
    total_unlifted_blocks = 0
    total_blocks = 0
    total_instructions = 0
    total_unsupported_instructions = 0
    total_jump_resolved = 0
    total_jump_unresolved = 0
    total_address_taken = 0
    agg_mnemonic_counts: dict[str, int] = {}
    agg_mnemonic_funcs: dict[str, set[str]] = defaultdict(set)

    sum_imp_total = 0
    sum_imp_guest = 0
    sum_imp_runtime = 0
    sum_imp_host_bound = 0
    sum_imp_host_unresolved = 0

    exe_module_name = ""
    first_module_name = ""
    exe_count = 0
    guest_dll_count = 0
    guest_dlls: list[str] = []

    def push_root(raw: Any) -> None:
        va = parse_va(raw)
        if va is None:
            return
        label = format_va(va)
        if label not in all_roots:
            all_roots.append(label)

    for artifact in inputs.module_artifacts:
        entry = artifact.entry
        mod_manifest = artifact.manifest
        mod_discover = artifact.discover
        mod_report = artifact.report_text

        raw_mod_name = str(entry.get("module") or "").strip()
        if not raw_mod_name and mod_manifest and isinstance(mod_manifest.get("module"), str):
            raw_mod_name = mod_manifest["module"].strip()
        if raw_mod_name:
            mod_name = (
                basename_label(raw_mod_name)
                if ("/" in raw_mod_name or "\\" in raw_mod_name or re.match(r"^[A-Za-z]:", raw_mod_name))
                else sanitize_text(raw_mod_name)
            )
        else:
            mod_name = basename_label(entry.get("dll") or entry.get("file"))
        if not first_module_name:
            first_module_name = mod_name

        dll_name = basename_label(entry.get("dll") or entry.get("file") or mod_name)
        role = sanitize_text(str(entry.get("role") or "dll").strip().lower())
        reason = sanitize_text(str(entry.get("reason") or "").strip())
        if role == "exe":
            exe_count += 1
            if not exe_module_name:
                exe_module_name = mod_name
        else:
            guest_dll_count += 1
            if dll_name not in guest_dlls:
                guest_dlls.append(dll_name)

        # Roots from program entry + TLS callbacks + per-module manifest/discover.
        push_root(entry.get("entry_point"))
        tls_obj = entry.get("tls")
        if isinstance(tls_obj, Mapping):
            for cb in tls_obj.get("callbacks") or []:
                push_root(cb)
        for root in ingest_tls_and_roots(mod_manifest, mod_discover):
            push_root(root)

        # Discover CFG nodes for this module.
        mod_nodes = ingest_discover_functions(mod_discover)
        for node in mod_nodes.values():
            if node.unsupported:
                node.lifted = False
            all_nodes[node.entry] = node

        bm_discovery = (mod_manifest or {}).get("discovery") if mod_manifest else None
        bm_lift = (mod_manifest or {}).get("lift") if mod_manifest else None
        bm_discovery = bm_discovery if isinstance(bm_discovery, Mapping) else {}
        bm_lift = bm_lift if isinstance(bm_lift, Mapping) else {}

        entry_lift = entry.get("lift") if isinstance(entry.get("lift"), Mapping) else {}
        entry_reg = (
            entry.get("registration") if isinstance(entry.get("registration"), Mapping) else {}
        )

        mod_address_taken = 0
        bm_refinements = (mod_manifest or {}).get("refinements") if mod_manifest else None
        if isinstance(bm_refinements, Mapping):
            dispatch = bm_refinements.get("dispatch") or {}
            if isinstance(dispatch, Mapping):
                not_lifted = dispatch.get("address_taken_not_lifted") or []
                if isinstance(not_lifted, list):
                    mod_address_taken = len(not_lifted)
        total_address_taken += mod_address_taken

        mod_lifted_raw = as_int(
            bm_lift.get("functions"),
            as_int(entry_lift.get("functions"), as_int(entry_reg.get("count"), 0)),
        )
        mod_discovered = as_int(
            bm_discovery.get("functions"),
            as_int((mod_discover or {}).get("function_count"), max(len(mod_nodes), mod_lifted_raw)),
        )
        if not mod_discovered:
            mod_discovered = max(len(mod_nodes), mod_lifted_raw)

        has_lift_info = bool(bm_lift or entry_lift or entry_reg)
        mod_lifted, mod_unlifted = _count_function_coverage(
            mod_nodes,
            mod_discovered,
            mod_lifted_raw,
            mod_address_taken,
            has_lift_data=has_lift_info,
        )

        mod_total_blocks = as_int(
            bm_discovery.get("blocks"),
            as_int(
                (mod_discover or {}).get("block_count"),
                sum(n.blocks for n in mod_nodes.values()),
            ),
        )
        mod_unlifted_blocks = sum(n.blocks for n in mod_nodes.values() if not n.lifted)
        mod_lifted_blocks = max(mod_total_blocks - mod_unlifted_blocks, 0) if mod_total_blocks else 0

        mod_instructions = as_int(
            bm_lift.get("instructions"),
            as_int(
                entry_lift.get("instructions"),
                as_int(
                    bm_discovery.get("instructions"),
                    as_int(
                        (mod_discover or {}).get("instruction_count"),
                        sum(n.instructions for n in mod_nodes.values()),
                    ),
                ),
            ),
        )
        mod_unsupported_inst = as_int(
            bm_lift.get("unsupported_instructions"),
            as_int(entry_lift.get("unsupported"), 0),
        )
        if not mod_unsupported_inst:
            mod_unsupported_inst = sum(sum(n.unsupported.values()) for n in mod_nodes.values())
        mod_lifted_inst = max(mod_instructions - mod_unsupported_inst, 0)

        # Mnemonics for this module.
        mod_mnem_counts: dict[str, int] = {}
        for node in mod_nodes.values():
            for mnemonic, count in node.unsupported.items():
                mod_mnem_counts[mnemonic] = mod_mnem_counts.get(mnemonic, 0) + count
                agg_mnemonic_funcs[mnemonic].add(f"{mod_name}:{node.entry}")
        if not mod_mnem_counts:
            by_mnem = bm_lift.get("unsupported_by_mnemonic") if bm_lift else None
            if isinstance(by_mnem, Mapping):
                for key, value in by_mnem.items():
                    mnemonic = str(key).strip().lower()
                    if mnemonic:
                        mod_mnem_counts[mnemonic] = mod_mnem_counts.get(mnemonic, 0) + as_int(
                            value, 0
                        )
        if not mod_mnem_counts:
            for mnemonic, count in parse_report_mnemonics(mod_report).items():
                mod_mnem_counts[mnemonic] = mod_mnem_counts.get(mnemonic, 0) + count
        for mnemonic, count in mod_mnem_counts.items():
            agg_mnemonic_counts[mnemonic] = agg_mnemonic_counts.get(mnemonic, 0) + count

        mod_shard_sizes = _shard_sizes(mod_manifest)
        mod_shards = _shard_stats_from_sizes(mod_shard_sizes, unit="instructions")
        all_shard_sizes.extend(mod_shard_sizes)

        mod_jump = _jump_tables(mod_manifest, mod_discover)
        total_jump_resolved += mod_jump["resolved_jump_tables"]
        total_jump_unresolved += mod_jump["unresolved_indirect_jumps"]

        # Imports summary for this module.
        entry_imp = entry.get("imports") if isinstance(entry.get("imports"), Mapping) else {}
        bm_imp = (
            (mod_manifest or {}).get("imports")
            if isinstance((mod_manifest or {}).get("imports"), Mapping)
            else {}
        )
        imp_total = as_int(entry_imp.get("total"), as_int(bm_imp.get("total"), 0))
        imp_guest = as_int(entry_imp.get("guest"), 0)
        imp_runtime = as_int(entry_imp.get("runtime"), 0)
        imp_host_bound = as_int(entry_imp.get("host_bound"), as_int(bm_imp.get("bound"), 0))
        imp_host_unresolved = as_int(
            entry_imp.get("host_unresolved"), as_int(bm_imp.get("unresolved"), 0)
        )

        # Match per-module unresolved labels from program imports[] + build-manifest referenced_unresolved.
        mod_unresolved_labels: list[str] = []
        for key_cand in (
            dll_name.lower(),
            basename_label(entry.get("file")).lower(),
            mod_name.lower(),
        ):
            for lbl in unresolved_by_importer.get(key_cand, ()):
                if lbl not in mod_unresolved_labels:
                    mod_unresolved_labels.append(lbl)
        bm_ref_unresolved = _unresolved_import_names(mod_manifest)
        for lbl in bm_ref_unresolved:
            if lbl not in mod_unresolved_labels:
                mod_unresolved_labels.append(lbl)
            if lbl not in program_unresolved_labels:
                program_unresolved_labels.append(lbl)

        imp_host_unresolved = max(imp_host_unresolved, len(mod_unresolved_labels))
        if imp_total < imp_guest + imp_runtime + imp_host_bound + imp_host_unresolved:
            imp_total = imp_guest + imp_runtime + imp_host_bound + imp_host_unresolved

        sum_imp_total += imp_total
        sum_imp_guest += imp_guest
        sum_imp_runtime += imp_runtime
        sum_imp_host_bound += imp_host_bound
        sum_imp_host_unresolved += imp_host_unresolved

        entry_exp = entry.get("exports") if isinstance(entry.get("exports"), Mapping) else {}
        mod_exports = {
            "total": as_int(entry_exp.get("total"), 0),
            "functions": as_int(entry_exp.get("functions"), 0),
            "data": as_int(entry_exp.get("data"), 0),
            "forwarders": as_int(entry_exp.get("forwarders"), 0),
            "ordinal_only": as_int(entry_exp.get("ordinal_only"), 0),
        }
        depends_on = [
            basename_label(dep)
            for dep in (entry.get("depends_on") or [])
            if isinstance(dep, str) and dep.strip()
        ]
        iat_prefill = entry.get("iat_prefill")
        iat_prefill_count = len(iat_prefill) if isinstance(iat_prefill, list) else 0

        total_discovered_fns += mod_discovered
        total_lifted_fns += mod_lifted
        total_unlifted_fns += mod_unlifted
        total_blocks += mod_total_blocks
        total_lifted_blocks += mod_lifted_blocks
        total_unlifted_blocks += mod_unlifted_blocks
        total_instructions += mod_instructions
        total_unsupported_instructions += mod_unsupported_inst

        module_rows.append(
            {
                "module": mod_name,
                "dll": dll_name,
                "role": role,
                "reason": reason,
                "base": format_va(entry["base"]) if entry.get("base") is not None else None,
                "size_of_image": (
                    format_va(entry["size_of_image"])
                    if entry.get("size_of_image") is not None
                    else None
                ),
                "rebased": bool(entry.get("rebased", False)),
                "entry_point": (
                    format_va(entry["entry_point"])
                    if entry.get("entry_point") is not None
                    else None
                ),
                "has_build_manifest": mod_manifest is not None,
                "discovered_functions": mod_discovered,
                "lifted_functions": mod_lifted,
                "unlifted_functions": mod_unlifted,
                "lifted_pct": as_float_pct(mod_lifted, mod_discovered),
                "instructions": mod_instructions,
                "unsupported_instructions": mod_unsupported_inst,
                "instruction_lifted_pct": as_float_pct(mod_lifted_inst, mod_instructions),
                "shard_count": mod_shards["count"],
                "shards": mod_shards,
                "imports": {
                    "total": imp_total,
                    "guest": imp_guest,
                    "runtime": imp_runtime,
                    "host_bound": imp_host_bound,
                    "host_unresolved": imp_host_unresolved,
                    "referenced_unresolved": len(bm_ref_unresolved),
                },
                "unresolved_imports": mod_unresolved_labels,
                "exports": mod_exports,
                "depends_on": depends_on,
                "iat_prefill_count": iat_prefill_count,
                "address_taken_not_lifted": mod_address_taken,
            }
        )

    # Reconcile unresolved_by_dll if program_manifest.imports was empty but module manifests had them.
    if not unresolved_by_dll_counts and program_unresolved_labels:
        for lbl in program_unresolved_labels:
            dll_part = lbl.split("!", 1)[0] if "!" in lbl else "<unknown>"
            unresolved_by_dll_counts[dll_part] += 1
            if lbl not in unresolved_by_dll_map[dll_part]:
                unresolved_by_dll_map[dll_part].append(lbl)

    unresolved_by_dll = [
        {
            "dll": dll_name,
            "count": count,
            "imports": unresolved_by_dll_map.get(dll_name, []),
        }
        for dll_name, count in sorted(
            unresolved_by_dll_counts.items(),
            key=lambda kv: (-kv[1], kv[0].lower()),
        )
    ]

    total_host_unresolved = max(
        sum_imp_host_unresolved,
        program_import_entries_unresolved,
        len(program_unresolved_labels),
    )

    program_dispatch = _extract_program_indirect_dispatch(program_manifest)
    total_shards_from_modules = len(all_shard_sizes)
    wasm_footprint = _summarize_wasm_footprint(
        guest_manifest,
        program_indirect_dispatch=program_dispatch,
        total_shards=total_shards_from_modules,
        total_lifted_functions=total_lifted_fns,
    )
    effective_dispatch = program_dispatch or wasm_footprint.get("indirect_dispatch")
    total_shards = total_shards_from_modules or as_int(
        wasm_footprint.get("side_module_shard_count"), 0
    )

    if all_shard_sizes:
        shards = _shard_stats_from_sizes(all_shard_sizes, unit="instructions")
    elif wasm_footprint.get("side_modules"):
        side_counts = [
            as_int(s.get("shard_count"), 0)
            for s in wasm_footprint["side_modules"]
            if as_int(s.get("shard_count"), 0) > 0
        ]
        if side_counts:
            mean_sc = sum(side_counts) / len(side_counts)
            shards = {
                "count": sum(side_counts),
                "min": min(side_counts),
                "median": median(side_counts),
                "max": max(side_counts),
                "balance_ratio": round(max(side_counts) / mean_sc, 4) if mean_sc > 0.0 else 0.0,
                "unit": "shards_per_side_module",
            }
        else:
            shards = _shard_stats_from_sizes([], unit="instructions")
    else:
        shards = _shard_stats_from_sizes([], unit="instructions")

    total_lifted_instructions = max(total_instructions - total_unsupported_instructions, 0)
    coverage = {
        "lifted_functions": total_lifted_fns,
        "unlifted_functions": total_unlifted_fns,
        "discovered_functions": total_discovered_fns,
        "lifted_pct": as_float_pct(total_lifted_fns, total_discovered_fns),
        "lifted_blocks": total_lifted_blocks,
        "unlifted_blocks": total_unlifted_blocks,
        "block_lifted_pct": as_float_pct(total_lifted_blocks, total_blocks),
        "lifted_instructions": total_lifted_instructions,
        "unlifted_instructions": total_unsupported_instructions,
        "instruction_lifted_pct": as_float_pct(total_lifted_instructions, total_instructions),
        "unsupported_instructions": total_unsupported_instructions,
    }

    mnemonics = _mnemonic_rows(agg_mnemonic_counts, agg_mnemonic_funcs, top)
    jump_total = total_jump_resolved + total_jump_unresolved
    jump = {
        "resolved_jump_tables": total_jump_resolved,
        "unresolved_indirect_jumps": total_jump_unresolved,
        "resolution_pct": as_float_pct(total_jump_resolved, jump_total),
    }

    if not all_roots and all_nodes:
        all_roots = list(all_nodes.keys())[:1]
    reachable, paths, boot_indirect = boot_bfs(all_nodes, all_roots, boot_depth)
    boot_nodes = [all_nodes[e] for e in reachable if e in all_nodes]
    if boot_nodes:
        boot_unlifted = sum(1 for n in boot_nodes if not n.lifted)
        boot_lifted = len(boot_nodes) - boot_unlifted
        boot_lifted_pct = as_float_pct(boot_lifted, len(boot_nodes))
    else:
        boot_lifted_pct = coverage["lifted_pct"]
        boot_lifted = 0
        boot_unlifted = 0

    boot_mnem_counts: dict[str, int] = {}
    boot_mnem_funcs: dict[str, set[str]] = defaultdict(set)
    for node in boot_nodes:
        for mnemonic, count in node.unsupported.items():
            boot_mnem_counts[mnemonic] = boot_mnem_counts.get(mnemonic, 0) + count
            boot_mnem_funcs[mnemonic].add(node.entry)

    boot_targets = _shortest_boot_targets(all_nodes, paths)
    boot = {
        "roots": all_roots,
        "boot_depth": boot_depth,
        "boot_reachable_functions": (
            len(reachable) if (all_nodes or all_roots) else total_discovered_fns
        ),
        "boot_lifted_functions": boot_lifted if boot_nodes else total_lifted_fns,
        "boot_unlifted_functions": boot_unlifted if boot_nodes else total_unlifted_fns,
        "boot_lifted_pct": boot_lifted_pct if boot_nodes else coverage["lifted_pct"],
        "boot_unlifted_mnemonics": _mnemonic_rows(boot_mnem_counts, boot_mnem_funcs, top),
        "boot_unresolved_indirect_calls": boot_indirect,
        "boot_targets": boot_targets,
        "reachable_entries": reachable,
    }

    init_order = [
        basename_label(item)
        for item in (program_manifest.get("init_order") or [])
        if isinstance(item, str) and item.strip()
    ]
    external_dlls = []
    for item in program_manifest.get("external_dlls") or []:
        if isinstance(item, Mapping):
            external_dlls.append(
                {
                    "dll": basename_label(item.get("dll")),
                    "status": sanitize_text(str(item.get("status") or "")),
                    "importers": [
                        basename_label(imp)
                        for imp in (item.get("importers") or [])
                        if isinstance(imp, str) and imp.strip()
                    ],
                }
            )
    cycles = []
    for item in program_manifest.get("cycles") or []:
        if isinstance(item, Mapping):
            cycles.append(
                {
                    "from": basename_label(item.get("from")),
                    "to": basename_label(item.get("to")),
                }
            )
    raw_diagnostics = program_manifest.get("diagnostics")
    diagnostics_count = len(raw_diagnostics) if isinstance(raw_diagnostics, list) else 0

    program_summary = {
        "module_count": len(module_rows),
        "exe_count": exe_count,
        "guest_dll_count": guest_dll_count,
        "guest_dlls": guest_dlls,
        "init_order": init_order,
        "indirect_dispatch": effective_dispatch,
        "total_shards": total_shards,
        "total_lifted_functions": total_lifted_fns,
        "total_discovered_functions": total_discovered_fns,
        "total_instructions": total_instructions,
        "total_unsupported_instructions": total_unsupported_instructions,
        "imports": {
            "total": sum_imp_total,
            "guest": sum_imp_guest,
            "runtime": sum_imp_runtime,
            "host_bound": sum_imp_host_bound,
            "host_unresolved": total_host_unresolved,
            "unresolved_recorded": unresolved_recorded,
            "unresolved_unrecorded": unresolved_unrecorded,
        },
        "unresolved_by_dll": unresolved_by_dll,
        "external_dlls": external_dlls,
        "cycles": cycles,
        "diagnostics_count": diagnostics_count,
        "modules": module_rows,
    }

    recs = _recommendations(
        coverage,
        boot,
        jump,
        mnemonics,
        program_unresolved_labels,
        total_address_taken,
    )
    recs = _append_multimodule_recommendations(recs, program_summary, wasm_footprint)

    report = {
        "schema": SCHEMA_AUDIT,
        "schema_version": SCHEMA_AUDIT_VERSION,
        "source": inputs.source,
        "module": exe_module_name or first_module_name,
        "coverage": coverage,
        "shards": shards,
        "unlifted_mnemonics": mnemonics,
        "jump_tables": jump,
        "boot": boot,
        "unresolved_imports": program_unresolved_labels,
        "address_taken_not_lifted": total_address_taken,
        "program": program_summary,
        "wasm_footprint": wasm_footprint,
        "recommendations": recs,
    }
    return sanitize_value(report)


def _program_summary_from_guest_manifest(
    wasm_footprint: Mapping[str, Any],
    single_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    guest_dlls = list(wasm_footprint.get("guest_dll_modules") or [])
    total_shards = as_int(wasm_footprint.get("side_module_shard_count"), 0)
    cov = (single_report or {}).get("coverage") if isinstance(single_report, Mapping) else None
    cov = cov if isinstance(cov, Mapping) else {}
    unres_list = (
        list(single_report.get("unresolved_imports") or [])
        if isinstance(single_report, Mapping)
        else []
    )
    main_mod = (
        sanitize_text(str(single_report.get("module") or "main"))
        if isinstance(single_report, Mapping) and single_report.get("module")
        else "main"
    )
    modules: list[dict[str, Any]] = [
        {
            "module": main_mod,
            "dll": main_mod,
            "role": "exe",
            "reason": "executable",
            "base": None,
            "size_of_image": None,
            "rebased": False,
            "entry_point": None,
            "has_build_manifest": bool(single_report and cov.get("discovered_functions")),
            "discovered_functions": as_int(cov.get("discovered_functions"), 0),
            "lifted_functions": as_int(cov.get("lifted_functions"), 0),
            "unlifted_functions": as_int(cov.get("unlifted_functions"), 0),
            "lifted_pct": _as_float(cov.get("lifted_pct"), 0.0),
            "instructions": as_int(cov.get("lifted_instructions"), 0)
            + as_int(cov.get("unlifted_instructions"), 0),
            "unsupported_instructions": as_int(cov.get("unsupported_instructions"), 0),
            "instruction_lifted_pct": _as_float(cov.get("instruction_lifted_pct"), 0.0),
            "shard_count": as_int(
                ((single_report or {}).get("shards") or {}).get("count"),
                total_shards,
            ),
            "shards": dict((single_report or {}).get("shards") or _shard_stats_from_sizes([])),
            "imports": {
                "total": len(unres_list),
                "guest": 0,
                "runtime": 0,
                "host_bound": 0,
                "host_unresolved": len(unres_list),
                "referenced_unresolved": len(unres_list),
            },
            "unresolved_imports": unres_list,
            "exports": {"total": 0, "functions": 0, "data": 0, "forwarders": 0, "ordinal_only": 0},
            "depends_on": guest_dlls,
            "iat_prefill_count": 0,
            "address_taken_not_lifted": as_int(
                (single_report or {}).get("address_taken_not_lifted"), 0
            ),
        }
    ]
    for dll in guest_dlls:
        modules.append(
            {
                "module": dll,
                "dll": dll,
                "role": "dll",
                "reason": "guest_manifest",
                "base": None,
                "size_of_image": None,
                "rebased": False,
                "entry_point": None,
                "has_build_manifest": False,
                "discovered_functions": 0,
                "lifted_functions": 0,
                "unlifted_functions": 0,
                "lifted_pct": 0.0,
                "instructions": 0,
                "unsupported_instructions": 0,
                "instruction_lifted_pct": 0.0,
                "shard_count": 0,
                "shards": _shard_stats_from_sizes([]),
                "imports": {
                    "total": 0,
                    "guest": 0,
                    "runtime": 0,
                    "host_bound": 0,
                    "host_unresolved": 0,
                    "referenced_unresolved": 0,
                },
                "unresolved_imports": [],
                "exports": {
                    "total": 0,
                    "functions": 0,
                    "data": 0,
                    "forwarders": 0,
                    "ordinal_only": 0,
                },
                "depends_on": [],
                "iat_prefill_count": 0,
                "address_taken_not_lifted": 0,
            }
        )
    return {
        "module_count": len(modules),
        "exe_count": 1,
        "guest_dll_count": len(guest_dlls),
        "guest_dlls": guest_dlls,
        "init_order": [*guest_dlls, main_mod],
        "indirect_dispatch": wasm_footprint.get("indirect_dispatch"),
        "total_shards": max(
            total_shards,
            as_int(((single_report or {}).get("shards") or {}).get("count"), 0),
        ),
        "total_lifted_functions": as_int(cov.get("lifted_functions"), 0),
        "total_discovered_functions": as_int(cov.get("discovered_functions"), 0),
        "total_instructions": as_int(cov.get("lifted_instructions"), 0)
        + as_int(cov.get("unlifted_instructions"), 0),
        "total_unsupported_instructions": as_int(cov.get("unsupported_instructions"), 0),
        "imports": {
            "total": len(unres_list),
            "guest": 0,
            "runtime": 0,
            "host_bound": 0,
            "host_unresolved": len(unres_list),
            "unresolved_recorded": 0,
            "unresolved_unrecorded": len(unres_list),
        },
        "unresolved_by_dll": [],
        "external_dlls": [],
        "cycles": [],
        "diagnostics_count": 0,
        "modules": modules,
    }


def unresolved_import_count(report: Mapping[str, Any]) -> int:
    """Return the effective unresolved import count for threshold gating."""
    listed = len(report.get("unresolved_imports") or [])
    program = report.get("program")
    if isinstance(program, Mapping):
        prog_imports = program.get("imports")
        if isinstance(prog_imports, Mapping):
            return max(as_int(prog_imports.get("host_unresolved"), 0), listed)
    return listed


def check_unresolved_threshold(
    report: Mapping[str, Any],
    max_unresolved: int,
) -> tuple[bool, int]:
    """Check whether unresolved imports are within `max_unresolved`."""
    if max_unresolved < 0:
        raise AuditError("--fail-on-unresolved must be >= 0")
    count = unresolved_import_count(report)
    return count <= max_unresolved, count


# ---------------------------------------------------------------------------
# Browser fast-boot Wasm64 audit (T68)
# ---------------------------------------------------------------------------


def _read_exact(fh: BinaryIO, length: int, pos: int, end: int) -> tuple[bytes, int]:
    if length < 0 or pos + length > end:
        raise _WasmBinaryError("unexpected EOF in Wasm binary")
    data = fh.read(length)
    if len(data) != length:
        raise _WasmBinaryError("short read in Wasm binary")
    return data, pos + length


def _read_uleb128_u32(fh: BinaryIO, pos: int, end: int) -> tuple[int, int]:
    result = 0
    shift = 0
    for idx in range(5):
        raw, pos = _read_exact(fh, 1, pos, end)
        byte = raw[0]
        if idx == 4 and ((byte & 0x80) != 0 or (byte & 0xF0) != 0):
            raise _WasmBinaryError("invalid ULEB128 u32")
        result |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return result, pos
        shift += 7
    raise _WasmBinaryError("overlong ULEB128 u32")


def inspect_wasm_header(path: Path) -> bool:
    """Return True iff `path` begins with valid WebAssembly 1.0 magic + version."""
    try:
        with path.open("rb") as fh:
            header = fh.read(8)
    except OSError:
        return False
    return len(header) == 8 and header[:4] == WASM_MAGIC and header[4:8] == WASM_VERSION_BYTES


def _empty_export_audit(*, inspected: bool, valid_wasm: bool = False) -> dict[str, Any]:
    required_names = [canonical for canonical, _ in REQUIRED_RUNTIME_EXPORTS]
    return {
        "inspected": inspected,
        "valid_wasm": valid_wasm,
        "has_export_section": False,
        "total_export_count": 0,
        "func_export_count": 0,
        "memory_exported": False,
        "required": {name: False for name in required_names},
        "required_present_count": 0,
        "required_total_count": len(required_names),
        "missing_required": list(required_names),
        "all_required_present": False,
    }


def inspect_wasm_exports(path: Path) -> dict[str, Any]:
    """Pure-Python WebAssembly Export section (id=7) inspector for `runtime.wasm`.

    Seeks directly past non-export sections so large Wasm64 binaries are audited
    without loading code/data sections into memory. Emits strictly sanitized
    counts, booleans, and fixed required-symbol names (zero host paths/bytes).
    """
    try:
        file_size = path.stat().st_size
    except OSError:
        return _empty_export_audit(inspected=True, valid_wasm=False)

    if file_size < 8:
        return _empty_export_audit(inspected=True, valid_wasm=False)

    has_export_section = False
    total_exports = 0
    func_exports = 0
    memory_exported = False
    func_export_names: set[str] = set()

    try:
        with path.open("rb") as fh:
            header, pos = _read_exact(fh, 8, 0, file_size)
            if header[:4] != WASM_MAGIC or header[4:8] != WASM_VERSION_BYTES:
                return _empty_export_audit(inspected=True, valid_wasm=False)

            while pos < file_size:
                sec_id_bytes, pos = _read_exact(fh, 1, pos, file_size)
                sec_id = sec_id_bytes[0]
                if sec_id > WASM_MAX_SECTION_ID:
                    raise _WasmBinaryError("unknown Wasm section id")
                sec_size, pos = _read_uleb128_u32(fh, pos, file_size)
                sec_end = pos + sec_size
                if sec_end > file_size:
                    raise _WasmBinaryError("truncated Wasm section")

                if sec_id == WASM_SECTION_EXPORT:
                    if has_export_section:
                        raise _WasmBinaryError("duplicate Wasm export section")
                    has_export_section = True
                    export_count, pos = _read_uleb128_u32(fh, pos, sec_end)
                    for _ in range(export_count):
                        name_len, pos = _read_uleb128_u32(fh, pos, sec_end)
                        raw_name, pos = _read_exact(fh, name_len, pos, sec_end)
                        try:
                            name = raw_name.decode("utf-8")
                        except UnicodeDecodeError as exc:
                            raise _WasmBinaryError("invalid UTF-8 export name") from exc
                        kind_bytes, pos = _read_exact(fh, 1, pos, sec_end)
                        kind_code = kind_bytes[0]
                        if kind_code not in WASM_EXPORT_KINDS:
                            raise _WasmBinaryError("unknown Wasm export kind")
                        _index, pos = _read_uleb128_u32(fh, pos, sec_end)
                        total_exports += 1
                        kind_name = WASM_EXPORT_KINDS[kind_code]
                        if kind_name == "func":
                            func_exports += 1
                            func_export_names.add(name)
                        elif kind_name == "memory":
                            memory_exported = True
                    if pos != sec_end:
                        raise _WasmBinaryError("trailing bytes in Wasm export section")
                else:
                    pos = sec_end
                    fh.seek(pos)
    except (_WasmBinaryError, OSError):
        return _empty_export_audit(inspected=True, valid_wasm=False)

    required_map: dict[str, bool] = {}
    missing_required: list[str] = []
    for canonical, aliases in REQUIRED_RUNTIME_EXPORTS:
        present = any(alias in func_export_names for alias in aliases)
        required_map[canonical] = present
        if not present:
            missing_required.append(canonical)

    present_count = len(REQUIRED_RUNTIME_EXPORTS) - len(missing_required)
    return {
        "inspected": True,
        "valid_wasm": True,
        "has_export_section": has_export_section,
        "total_export_count": total_exports,
        "func_export_count": func_exports,
        "memory_exported": memory_exported,
        "required": required_map,
        "required_present_count": present_count,
        "required_total_count": len(REQUIRED_RUNTIME_EXPORTS),
        "missing_required": missing_required,
        "all_required_present": len(missing_required) == 0,
    }


def _natural_part_sort_key(name: str) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _is_safe_child_file(root_resolved: Path, candidate: Path) -> bool:
    try:
        resolved = candidate.resolve()
        return resolved.is_relative_to(root_resolved) and resolved.is_file()
    except OSError:
        return False


def _validate_manifest_relpath(raw: Any) -> tuple[str, ...] | None:
    """Validate a manifest-supplied relative artifact path (rejects traversal/host paths)."""
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if _SAFE_CONTROL_CHAR_RE.search(cleaned):
        return None
    if _ENCODED_TRAVERSAL_RE.search(cleaned):
        return None
    if cleaned.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", cleaned):
        return None
    if _SCHEME_PREFIX_RE.match(cleaned):
        return None
    normalized = cleaned.replace("\\", "/")
    segments = [seg for seg in normalized.split("/") if seg and seg != "."]
    if not segments or any(seg == ".." for seg in segments):
        return None
    return tuple(segments)


def _resolve_manifest_artifact(base_dirs: Sequence[Path], segments: tuple[str, ...]) -> Path | None:
    for base in base_dirs:
        try:
            root = base.resolve()
            candidate = (root / Path(*segments)).resolve()
            if candidate.is_relative_to(root) and candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


def _compute_split_balance(
    part_sizes: Sequence[int],
    *,
    warn_ratio: float = DEFAULT_SPLIT_BALANCE_WARN_RATIO,
) -> dict[str, Any]:
    summary = compute_shard_balance_summary(part_sizes)
    count = summary["shard_count"]
    if count == 0:
        return {
            "part_count": 0,
            "shard_count": 0,
            "min_bytes": 0,
            "median_bytes": 0.0,
            "mean_bytes": 0.0,
            "max_bytes": 0,
            "total_bytes": 0,
            "max_shard_mib": 0.0,
            "total_mib": 0.0,
            "max_to_mean_ratio": 0.0,
            "max_to_min_ratio": 0.0,
            "max_share_ratio": 0.0,
            "split_balance_ratio": 0.0,
            "balance_ratio": 0.0,
            "warn_threshold_ratio": round(warn_ratio, 4),
            "has_empty_part": False,
            "dominant_shard_warning": False,
            "balanced": False,
        }

    sizes = [max(as_int(sz, 0), 0) for sz in part_sizes]
    total = sum(sizes)
    min_b = min(sizes)
    max_b = max(sizes)
    mean_b = total / count
    med_b = median(sizes)
    has_empty = min_b == 0

    max_to_mean = summary["balance_ratio"]
    if min_b > 0:
        max_to_min: float | None = round(max_b / min_b, 4)
    elif max_b == 0:
        max_to_min = 0.0
    else:
        max_to_min = None
    max_share = round(max_b / total, 4) if total > 0 else 0.0

    dominant_warning = bool(
        count >= 2
        and (
            max_to_mean > warn_ratio
            or (has_empty and max_b > 0)
        )
    )
    balanced = bool(count >= 1 and total > 0 and not has_empty and not dominant_warning)

    return {
        "part_count": count,
        "shard_count": summary["shard_count"],
        "min_bytes": min_b,
        "median_bytes": round(med_b, 2),
        "mean_bytes": round(mean_b, 2),
        "max_bytes": max_b,
        "total_bytes": total,
        "max_shard_mib": summary["max_shard_mib"],
        "total_mib": summary["total_mib"],
        "max_to_mean_ratio": max_to_mean,
        "max_to_min_ratio": max_to_min,
        "max_share_ratio": max_share,
        "split_balance_ratio": max_to_mean,
        "balance_ratio": summary["balance_ratio"],
        "warn_threshold_ratio": round(warn_ratio, 4),
        "has_empty_part": has_empty,
        "dominant_shard_warning": dominant_warning,
        "balanced": balanced,
    }


def _audit_boot_manifest_consistency(
    dist_dir: Path,
    *,
    boot_manifest: str | Path | None,
    runtime_wasm_present: bool,
    runtime_wasm_bytes: int,
    runtime_js_present: bool,
    staged_guest_parts: Mapping[str, int],
) -> dict[str, Any]:
    dist_resolved = dist_dir.resolve()
    manifest_path: Path | None = None
    manifest_kind = "none"

    if boot_manifest is not None:
        explicit = Path(boot_manifest)
        if not explicit.is_file():
            raise AuditError(f"boot manifest not found: {basename_label(boot_manifest)}")
        manifest_path = explicit
        manifest_kind = (
            "guest_manifest"
            if explicit.name.lower() == GUEST_MANIFEST_FILE
            else "boot_manifest"
        )
    else:
        guest_cand = dist_dir / GUEST_MANIFEST_FILE
        boot_cand = dist_dir / BOOT_MANIFEST_FILE
        if _is_safe_child_file(dist_resolved, guest_cand):
            manifest_path = guest_cand
            manifest_kind = "guest_manifest"
        elif _is_safe_child_file(dist_resolved, boot_cand):
            manifest_path = boot_cand
            manifest_kind = "boot_manifest"

    staged_side_count = len(staged_guest_parts)
    if manifest_path is None:
        return {
            "present": False,
            "kind": "none",
            "valid_json": False,
            "declared_side_module_count": 0,
            "staged_side_module_count": staged_side_count,
            "declared_split_modules": None,
            "split_count_matches": None,
            "runtime_wasm_matches": None,
            "runtime_js_matches": None,
            "total_bytes_matches": None,
            "missing_on_disk_count": 0,
            "unlisted_on_disk_count": 0,
            "size_mismatch_count": 0,
            "unsafe_entry_count": 0,
            "checked_artifact_count": 0,
            "consistent": False,
        }

    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {
            "present": True,
            "kind": manifest_kind,
            "valid_json": False,
            "declared_side_module_count": 0,
            "staged_side_module_count": staged_side_count,
            "declared_split_modules": None,
            "split_count_matches": None,
            "runtime_wasm_matches": None,
            "runtime_js_matches": None,
            "total_bytes_matches": None,
            "missing_on_disk_count": 0,
            "unlisted_on_disk_count": 0,
            "size_mismatch_count": 0,
            "unsafe_entry_count": 0,
            "checked_artifact_count": 0,
            "consistent": False,
        }

    if not isinstance(doc, dict):
        return {
            "present": True,
            "kind": manifest_kind,
            "valid_json": False,
            "declared_side_module_count": 0,
            "staged_side_module_count": staged_side_count,
            "declared_split_modules": None,
            "split_count_matches": None,
            "runtime_wasm_matches": None,
            "runtime_js_matches": None,
            "total_bytes_matches": None,
            "missing_on_disk_count": 0,
            "unlisted_on_disk_count": 0,
            "size_mismatch_count": 0,
            "unsafe_entry_count": 0,
            "checked_artifact_count": 0,
            "consistent": False,
        }

    if (
        manifest_path.name.lower() != BOOT_MANIFEST_FILE
        and doc.get("schema") != "boot-manifest"
        and is_guest_manifest(doc)
    ):
        manifest_kind = "guest_manifest"

    base_dirs: list[Path] = [dist_dir]
    if manifest_path.parent.resolve() != dist_resolved:
        base_dirs.append(manifest_path.parent)

    unsafe_entry_count = 0
    missing_keys: set[str] = set()
    size_mismatch_keys: set[str] = set()
    checked_keys: set[str] = set()

    def _check_artifact(
        raw_rel: Any,
        *,
        expected_bytes: int | None = None,
        search_dirs: Sequence[Path] = (dist_dir,),
    ) -> tuple[str | None, int | None]:
        nonlocal unsafe_entry_count
        segs = _validate_manifest_relpath(raw_rel)
        if segs is None:
            unsafe_entry_count += 1
            return None, None
        key = "/".join(segs).lower()
        checked_keys.add(key)
        resolved = _resolve_manifest_artifact(search_dirs, segs)
        if resolved is None:
            missing_keys.add(key)
            return key, None
        try:
            actual_size = resolved.stat().st_size
        except OSError:
            missing_keys.add(key)
            return key, None
        if expected_bytes is not None and actual_size != expected_bytes:
            size_mismatch_keys.add(key)
        return key, actual_size

    # 1. Runtime Wasm / JS declarations.
    runtime_wasm_matches: bool | None = None
    raw_runtime_wasm = _pick(doc, "runtimeWasm", "runtime_wasm")
    if raw_runtime_wasm is not None:
        key, actual = _check_artifact(raw_runtime_wasm, search_dirs=(dist_dir,))
        runtime_wasm_matches = bool(key is not None and actual is not None and runtime_wasm_present)

    runtime_js_matches: bool | None = None
    raw_runtime_js = _pick(doc, "runtimeJs", "runtime_js")
    if raw_runtime_js is not None:
        key, actual = _check_artifact(raw_runtime_js, search_dirs=(dist_dir,))
        runtime_js_matches = bool(key is not None and actual is not None and runtime_js_present)

    # 2. Side modules & declared byte sizes.
    declared_side_keys: set[str] = set()
    declared_side_order: list[str] = []
    declared_side_sizes_map: dict[str, int] = {}

    raw_side_sizes = _pick(doc, "sideModuleSizes", "side_module_sizes")
    if isinstance(raw_side_sizes, list):
        for item in raw_side_sizes:
            if not isinstance(item, Mapping):
                unsafe_entry_count += 1
                continue
            raw_name = _pick(item, "name", "path", "file")
            raw_bytes = _pick(item, "wasmBytes", "wasm_bytes", "size", "bytes")
            expected = (
                as_int(raw_bytes, -1)
                if raw_bytes is not None and not isinstance(raw_bytes, bool)
                else None
            )
            if expected is not None and expected < 0:
                expected = None
            key, actual = _check_artifact(raw_name, expected_bytes=expected, search_dirs=(dist_dir,))
            if key is not None:
                if key not in declared_side_keys:
                    declared_side_keys.add(key)
                    declared_side_order.append(key)
                if actual is not None:
                    declared_side_sizes_map[key] = actual

    raw_side_names = _pick(doc, "sideModules", "side_modules")
    if isinstance(raw_side_names, list):
        for item in raw_side_names:
            key, actual = _check_artifact(item, search_dirs=(dist_dir,))
            if key is not None:
                if key not in declared_side_keys:
                    declared_side_keys.add(key)
                    declared_side_order.append(key)
                if actual is not None and key not in declared_side_sizes_map:
                    declared_side_sizes_map[key] = actual

    # 3. wasmMetrics per-module byte sizes & totals.
    wasm_metrics = _pick(doc, "wasmMetrics", "wasm_metrics")
    total_bytes_matches: bool | None = None
    wasm_metrics_actual_by_key: dict[str, int] = {}
    if isinstance(wasm_metrics, Mapping):
        raw_mods = wasm_metrics.get("modules")
        if isinstance(raw_mods, list):
            for item in raw_mods:
                if not isinstance(item, Mapping):
                    continue
                raw_name = _pick(item, "name", "path", "file")
                raw_bytes = _pick(item, "wasmBytes", "wasm_bytes", "size")
                expected = (
                    as_int(raw_bytes, -1)
                    if raw_bytes is not None and not isinstance(raw_bytes, bool)
                    else None
                )
                if expected is not None and expected < 0:
                    expected = None
                key, actual = _check_artifact(
                    raw_name,
                    expected_bytes=expected,
                    search_dirs=(dist_dir,),
                )
                if key is not None and actual is not None:
                    wasm_metrics_actual_by_key[key] = actual
                    role = str(item.get("role") or "").strip().lower()
                    if role == "side" and key not in declared_side_keys:
                        declared_side_keys.add(key)
                        declared_side_order.append(key)

        totals = wasm_metrics.get("totals")
        if isinstance(totals, Mapping):
            raw_total = _pick(totals, "wasmBytes", "wasm_bytes")
            if raw_total is not None and not isinstance(raw_total, bool):
                declared_total = as_int(raw_total, -1)
                if declared_total >= 0:
                    actual_total = runtime_wasm_bytes + sum(staged_guest_parts.values())
                    # Include any non-guest_part declared side modules (e.g., guest_reg.wasm).
                    for key, sz in declared_side_sizes_map.items():
                        if key not in staged_guest_parts:
                            actual_total += sz
                    total_bytes_matches = declared_total == actual_total

    raw_top_total = _pick(doc, "totalWasmBytes", "total_wasm_bytes", "totalBytes", "total_bytes")
    if total_bytes_matches is None and raw_top_total is not None and not isinstance(raw_top_total, bool):
        declared_total = as_int(raw_top_total, -1)
        if declared_total >= 0:
            actual_total = runtime_wasm_bytes + sum(staged_guest_parts.values())
            for key, sz in declared_side_sizes_map.items():
                if key not in staged_guest_parts:
                    actual_total += sz
            total_bytes_matches = declared_total == actual_total

    # 4. Optional boot-manifest.json image/file references (`pgim`, `image`, `files`, `artifacts`).
    for single_key in ("pgim", "image"):
        val = doc.get(single_key)
        if isinstance(val, str) and val.strip():
            _check_artifact(val, search_dirs=base_dirs)

    for list_key in ("files", "artifacts"):
        raw_list = doc.get(list_key)
        if isinstance(raw_list, list):
            for entry in raw_list:
                if isinstance(entry, str):
                    _check_artifact(entry, search_dirs=base_dirs)
                elif isinstance(entry, Mapping):
                    raw_path = _pick(entry, "path", "name", "file")
                    raw_sz = _pick(entry, "size", "bytes", "wasmBytes", "wasm_bytes")
                    expected = (
                        as_int(raw_sz, -1)
                        if raw_sz is not None and not isinstance(raw_sz, bool)
                        else None
                    )
                    if expected is not None and expected < 0:
                        expected = None
                    _check_artifact(raw_path, expected_bytes=expected, search_dirs=base_dirs)
                else:
                    unsafe_entry_count += 1

    # 5. Split count and unlisted staged `guest_part_*.wasm` checks.
    raw_split_modules = _pick(doc, "splitModules", "split_modules")
    declared_split_modules: int | None = None
    split_count_matches: bool | None = None
    if raw_split_modules is not None and not isinstance(raw_split_modules, bool):
        declared_split_modules = as_int(raw_split_modules, -1)
        if declared_split_modules >= 0:
            split_count_matches = declared_split_modules == staged_side_count
        else:
            declared_split_modules = None

    has_side_section = (
        raw_side_names is not None
        or raw_side_sizes is not None
        or manifest_kind == "guest_manifest"
        or bool(declared_side_keys)
    )
    unlisted_on_disk_count = 0
    if has_side_section:
        for staged_name in staged_guest_parts:
            if staged_name not in declared_side_keys:
                unlisted_on_disk_count += 1

    consistent = bool(
        len(missing_keys) == 0
        and unlisted_on_disk_count == 0
        and len(size_mismatch_keys) == 0
        and unsafe_entry_count == 0
        and runtime_wasm_matches is not False
        and runtime_js_matches is not False
        and split_count_matches is not False
        and total_bytes_matches is not False
    )

    return {
        "present": True,
        "kind": manifest_kind,
        "valid_json": True,
        "declared_side_module_count": len(declared_side_order),
        "staged_side_module_count": staged_side_count,
        "declared_split_modules": declared_split_modules,
        "split_count_matches": split_count_matches,
        "runtime_wasm_matches": runtime_wasm_matches,
        "runtime_js_matches": runtime_js_matches,
        "total_bytes_matches": total_bytes_matches,
        "missing_on_disk_count": len(missing_keys),
        "unlisted_on_disk_count": unlisted_on_disk_count,
        "size_mismatch_count": len(size_mismatch_keys),
        "unsafe_entry_count": unsafe_entry_count,
        "checked_artifact_count": len(checked_keys),
        "consistent": consistent,
    }


def audit_wasm64_fast_boot(
    dist_dir: str | Path,
    *,
    boot_manifest: str | Path | None = None,
    balance_warn_ratio: float = DEFAULT_SPLIT_BALANCE_WARN_RATIO,
) -> dict[str, Any]:
    """Audit a staged browser Wasm64 dist directory for fast-boot readiness (T68).

    Checks:
      1. `runtime.wasm` + `guest_part_*.wasm` presence, total byte size, V8 1 GiB
         module cap, and split-balance ratio (warning when a single shard
         dominates cold startup download/compile time).
      2. Pure-Python Wasm export section inspection on `runtime.wasm` confirming
         `_main` / `main`, `_playgta6_get_input_state_ptr`, and
         `_playgta6_audio_ring`.
      3. Boot manifest (`guest_manifest.json` / `boot-manifest.json`) consistency
         against staged artifacts.

    Emits strictly sanitized counts, ratios, booleans, and fixed symbol/warning
    codes (zero host filesystem paths or usernames).
    """
    if isinstance(balance_warn_ratio, bool) or balance_warn_ratio <= 1.0:
        raise AuditError("--balance-warn-ratio must be > 1.0")

    target = Path(dist_dir)
    if not target.is_dir():
        raise AuditError(f"dist directory not found: {basename_label(dist_dir)}")

    dist_resolved = target.resolve()
    skipped_unsafe_entries = 0

    # 1. Inspect runtime.wasm & runtime.js safely inside dist_dir.
    runtime_wasm_path = target / RUNTIME_WASM_FILE
    runtime_wasm_present = _is_safe_child_file(dist_resolved, runtime_wasm_path)
    if not runtime_wasm_present and runtime_wasm_path.exists():
        skipped_unsafe_entries += 1

    runtime_wasm_bytes = 0
    if runtime_wasm_present:
        try:
            runtime_wasm_bytes = runtime_wasm_path.stat().st_size
        except OSError:
            runtime_wasm_present = False
            runtime_wasm_bytes = 0

    runtime_js_path = target / RUNTIME_JS_FILE
    runtime_js_present = _is_safe_child_file(dist_resolved, runtime_js_path)
    if not runtime_js_present and runtime_js_path.exists():
        skipped_unsafe_entries += 1

    # 2. Pure-Python Wasm export section inspection on runtime.wasm.
    if runtime_wasm_present:
        exports = inspect_wasm_exports(runtime_wasm_path)
    else:
        exports = _empty_export_audit(inspected=False, valid_wasm=False)

    # 3. Discover guest_part_*.wasm split modules (naturally sorted, symlink-safe).
    part_candidates: list[Path] = []
    try:
        for child in target.iterdir():
            if not _GUEST_PART_RE.match(child.name):
                continue
            if not _is_safe_child_file(dist_resolved, child):
                skipped_unsafe_entries += 1
                continue
            part_candidates.append(child)
    except OSError as exc:
        raise AuditError(
            f"failed to scan dist directory {basename_label(dist_dir)}: "
            f"{exc.strerror or type(exc).__name__}"
        ) from exc

    part_candidates.sort(key=lambda p: _natural_part_sort_key(p.name))

    staged_guest_parts: dict[str, int] = {}
    part_sizes: list[int] = []
    guest_parts_valid_count = 0
    guest_parts_invalid_count = 0
    exceeds_v8_count = 0

    if runtime_wasm_present and runtime_wasm_bytes > V8_MAX_WASM_MODULE_BYTES:
        exceeds_v8_count += 1

    for part_path in part_candidates:
        try:
            size = part_path.stat().st_size
        except OSError:
            guest_parts_invalid_count += 1
            continue
        staged_guest_parts[part_path.name.lower()] = size
        part_sizes.append(size)
        if inspect_wasm_header(part_path):
            guest_parts_valid_count += 1
        else:
            guest_parts_invalid_count += 1
        if size > V8_MAX_WASM_MODULE_BYTES:
            exceeds_v8_count += 1

    guest_parts_total_bytes = sum(part_sizes)
    total_wasm_bytes = runtime_wasm_bytes + guest_parts_total_bytes

    artifacts = {
        "runtime_wasm_present": runtime_wasm_present,
        "runtime_wasm_valid": bool(exports["valid_wasm"]),
        "runtime_wasm_bytes": runtime_wasm_bytes,
        "runtime_wasm_mib": round(runtime_wasm_bytes / (1024 * 1024), 4),
        "runtime_js_present": runtime_js_present,
        "guest_part_count": len(part_sizes),
        "guest_parts_present": len(part_sizes) > 0,
        "guest_parts_valid_count": guest_parts_valid_count,
        "guest_parts_invalid_count": guest_parts_invalid_count,
        "guest_parts_total_bytes": guest_parts_total_bytes,
        "guest_parts_total_mib": round(guest_parts_total_bytes / (1024 * 1024), 4),
        "total_wasm_bytes": total_wasm_bytes,
        "total_wasm_mib": round(total_wasm_bytes / (1024 * 1024), 4),
        "v8_1gib_limit_bytes": V8_MAX_WASM_MODULE_BYTES,
        "exceeds_v8_1gib_count": exceeds_v8_count,
        "any_exceeds_v8_1gib_limit": exceeds_v8_count > 0,
        "skipped_unsafe_entries": skipped_unsafe_entries,
    }

    split_balance = _compute_split_balance(part_sizes, warn_ratio=balance_warn_ratio)
    shard_balance = compute_shard_balance_summary(part_sizes)

    boot_manifest_report = _audit_boot_manifest_consistency(
        target,
        boot_manifest=boot_manifest,
        runtime_wasm_present=runtime_wasm_present,
        runtime_wasm_bytes=runtime_wasm_bytes,
        runtime_js_present=runtime_js_present,
        staged_guest_parts=staged_guest_parts,
    )

    warnings: list[str] = []
    if not runtime_wasm_present:
        warnings.append("missing_runtime_wasm")
    elif not exports["valid_wasm"]:
        warnings.append("invalid_runtime_wasm")
    elif not exports["all_required_present"]:
        warnings.append("missing_required_exports")

    if len(part_sizes) == 0:
        warnings.append("missing_guest_parts")
    if guest_parts_invalid_count > 0:
        warnings.append("invalid_guest_part_wasm")
    if exceeds_v8_count > 0:
        warnings.append("v8_1gib_limit_exceeded")
    if split_balance["dominant_shard_warning"]:
        warnings.append("dominant_shard_imbalance")

    if not boot_manifest_report["present"]:
        warnings.append("missing_boot_manifest")
    elif not boot_manifest_report["valid_json"]:
        warnings.append("invalid_boot_manifest_json")
    else:
        if boot_manifest_report["unsafe_entry_count"] > 0:
            warnings.append("boot_manifest_unsafe_entries")
        if not boot_manifest_report["consistent"]:
            warnings.append("boot_manifest_mismatch")

    ready = bool(
        runtime_wasm_present
        and exports["valid_wasm"]
        and exports["all_required_present"]
        and len(part_sizes) > 0
        and guest_parts_invalid_count == 0
        and exceeds_v8_count == 0
        and boot_manifest_report["consistent"]
    )

    return sanitize_value(
        {
            "ready": ready,
            "has_warnings": len(warnings) > 0,
            "warning_count": len(warnings),
            "warnings": warnings,
            "artifacts": artifacts,
            "shard_balance": shard_balance,
            "split_balance": split_balance,
            "exports": exports,
            "boot_manifest": boot_manifest_report,
        }
    )


def _fast_boot_recommendation_items(fast_boot: Mapping[str, Any]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    artifacts = fast_boot.get("artifacts") or {}
    exports = fast_boot.get("exports") or {}
    balance = fast_boot.get("split_balance") or {}
    manifest = fast_boot.get("boot_manifest") or {}

    if not artifacts.get("runtime_wasm_present"):
        items.append(
            (
                "fast_boot_runtime",
                "Stage runtime.wasm into the browser dist directory before launching.",
            )
        )
    elif not exports.get("valid_wasm"):
        items.append(
            (
                "fast_boot_runtime",
                "Staged runtime.wasm failed Wasm binary header/section parsing; rebuild with Emscripten -sMEMORY64=1.",
            )
        )
    elif not exports.get("all_required_present"):
        missing = ", ".join(exports.get("missing_required") or [])
        items.append(
            (
                "fast_boot_exports",
                f"runtime.wasm is missing required browser boot exports ({missing}); "
                "ensure -sEXPORTED_FUNCTIONS=_main,_playgta6_get_input_state_ptr,_playgta6_audio_ring.",
            )
        )

    if not artifacts.get("guest_parts_present"):
        items.append(
            (
                "fast_boot_splits",
                "No guest_part_*.wasm split side modules found; build split modules for parallel browser cold-start streaming.",
            )
        )
    elif as_int(artifacts.get("guest_parts_invalid_count"), 0) > 0:
        items.append(
            (
                "fast_boot_splits",
                f"{artifacts['guest_parts_invalid_count']} guest_part_*.wasm file(s) failed Wasm header validation.",
            )
        )

    if artifacts.get("any_exceeds_v8_1gib_limit"):
        items.append(
            (
                "fast_boot_v8_cap",
                f"{artifacts.get('exceeds_v8_1gib_count', 1)} Wasm module(s) exceed the 1 GiB V8 module cap; "
                "increase --split-modules or lower --max-side-module-mib.",
            )
        )

    if balance.get("dominant_shard_warning"):
        items.append(
            (
                "fast_boot_balance",
                f"Split-module size imbalance detected (max/mean={_as_float(balance.get('split_balance_ratio'), 0.0):.2f}x, "
                f"largest share={_as_float(balance.get('max_share_ratio'), 0.0) * 100.0:.1f}%, "
                f"threshold={_as_float(balance.get('warn_threshold_ratio'), DEFAULT_SPLIT_BALANCE_WARN_RATIO):.2f}x); "
                "rebalance guest_part_*.wasm shards so one shard does not dominate cold startup download/compile time.",
            )
        )

    if not manifest.get("present"):
        items.append(
            (
                "fast_boot_manifest",
                "No guest_manifest.json or boot-manifest.json found in dist directory; stage manifest alongside Wasm artifacts.",
            )
        )
    elif not manifest.get("valid_json"):
        items.append(
            (
                "fast_boot_manifest",
                "Boot manifest JSON could not be parsed; regenerate guest_manifest.json / boot-manifest.json.",
            )
        )
    elif not manifest.get("consistent"):
        items.append(
            (
                "fast_boot_manifest",
                f"Boot manifest drift detected (missing={as_int(manifest.get('missing_on_disk_count'), 0)}, "
                f"unlisted={as_int(manifest.get('unlisted_on_disk_count'), 0)}, "
                f"size_mismatches={as_int(manifest.get('size_mismatch_count'), 0)}, "
                f"unsafe_entries={as_int(manifest.get('unsafe_entry_count'), 0)}); "
                "re-stage dist artifacts and regenerate the manifest.",
            )
        )

    return items


def _append_fast_boot_recommendations(
    recs: list[dict[str, Any]],
    fast_boot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    extra = _fast_boot_recommendation_items(fast_boot)
    if not extra:
        return recs
    filtered = [dict(r) for r in recs if r.get("category") != "ready"]
    for category, detail in extra:
        filtered.append({"priority": len(filtered) + 1, "category": category, "detail": detail})
    for idx, rec in enumerate(filtered, start=1):
        rec["priority"] = idx
    return filtered


def audit_dist_dir(
    dist_dir: str | Path,
    *,
    boot_manifest: str | Path | None = None,
    balance_warn_ratio: float = DEFAULT_SPLIT_BALANCE_WARN_RATIO,
) -> dict[str, Any]:
    """Standalone browser fast-boot Wasm64 audit report for `--dist-dir <DIR>`."""
    fast_boot = audit_wasm64_fast_boot(
        dist_dir,
        boot_manifest=boot_manifest,
        balance_warn_ratio=balance_warn_ratio,
    )
    items = _fast_boot_recommendation_items(fast_boot)
    if items:
        recs = [
            {"priority": idx, "category": cat, "detail": detail}
            for idx, (cat, detail) in enumerate(items, start=1)
        ]
    else:
        recs = [
            {
                "priority": 1,
                "category": "ready",
                "detail": "Wasm64 fast-boot staging checks passed (runtime exports, split balance, and boot manifest aligned).",
            }
        ]
    report = {
        "schema": SCHEMA_AUDIT,
        "schema_version": SCHEMA_AUDIT_VERSION,
        "mode": "wasm64_fast_boot",
        "source": basename_label(dist_dir),
        "wasm64_fast_boot": fast_boot,
        "recommendations": recs,
    }
    return sanitize_value(report)


def audit_recomp_build(
    path: str | Path,
    *,
    top: int = DEFAULT_TOP,
    boot_depth: int | None = None,
    dist_dir: str | Path | None = None,
    boot_manifest: str | Path | None = None,
    balance_warn_ratio: float = DEFAULT_SPLIT_BALANCE_WARN_RATIO,
) -> dict[str, Any]:
    """Audit a recomp build dir, `program-manifest.json`, `guest_manifest.json`, `build-manifest.json`, or discover JSON."""
    if top < 0:
        raise AuditError("--top must be >= 0")
    if boot_depth is not None and boot_depth < 0:
        raise AuditError("--boot-depth must be >= 0")

    inputs = load_audit_inputs(path)

    if inputs.program_manifest is not None:
        report = _audit_program_inputs(inputs, top=top, boot_depth=boot_depth)
    elif inputs.guest_manifest is not None:
        if inputs.manifest is not None or inputs.discover is not None:
            report = _audit_single_module_core(
                inputs.manifest,
                inputs.discover,
                inputs.source,
                inputs.report_text,
                top=top,
                boot_depth=boot_depth,
            )
        else:
            empty_cov = {
                "lifted_functions": 0,
                "unlifted_functions": 0,
                "discovered_functions": 0,
                "lifted_pct": 0.0,
                "lifted_blocks": 0,
                "unlifted_blocks": 0,
                "block_lifted_pct": 0.0,
                "lifted_instructions": 0,
                "unlifted_instructions": 0,
                "instruction_lifted_pct": 0.0,
                "unsupported_instructions": 0,
            }
            empty_boot = {
                "roots": [],
                "boot_depth": boot_depth,
                "boot_reachable_functions": 0,
                "boot_lifted_functions": 0,
                "boot_unlifted_functions": 0,
                "boot_lifted_pct": 0.0,
                "boot_unlifted_mnemonics": [],
                "boot_unresolved_indirect_calls": 0,
                "boot_targets": {
                    t: {"reachable": False, "via": None, "shortest_path": []}
                    for t in BOOT_TARGETS
                },
                "reachable_entries": [],
            }
            report = {
                "schema": SCHEMA_AUDIT,
                "schema_version": SCHEMA_AUDIT_VERSION,
                "source": inputs.source,
                "module": "",
                "coverage": empty_cov,
                "shards": _shard_stats_from_sizes([]),
                "unlifted_mnemonics": [],
                "jump_tables": {
                    "resolved_jump_tables": 0,
                    "unresolved_indirect_jumps": 0,
                    "resolution_pct": 0.0,
                },
                "boot": empty_boot,
                "unresolved_imports": [],
                "address_taken_not_lifted": 0,
                "recommendations": [
                    {
                        "priority": 1,
                        "category": "ready",
                        "detail": "Wasm footprint loaded from guest_manifest.json; pass a recomp build dir for lift coverage.",
                    }
                ],
            }

        wasm_footprint = _summarize_wasm_footprint(
            inputs.guest_manifest,
            total_shards=as_int((report.get("shards") or {}).get("count"), 0),
            total_lifted_functions=as_int(
                (report.get("coverage") or {}).get("lifted_functions"), 0
            ),
        )
        if not report["shards"]["count"] and wasm_footprint.get("side_modules"):
            side_counts = [
                as_int(s.get("shard_count"), 0)
                for s in wasm_footprint["side_modules"]
                if as_int(s.get("shard_count"), 0) > 0
            ]
            if side_counts:
                mean_sc = sum(side_counts) / len(side_counts)
                report["shards"] = {
                    "count": sum(side_counts),
                    "min": min(side_counts),
                    "median": median(side_counts),
                    "max": max(side_counts),
                    "balance_ratio": round(max(side_counts) / mean_sc, 4) if mean_sc > 0.0 else 0.0,
                    "unit": "shards_per_side_module",
                }
        program_summary = _program_summary_from_guest_manifest(wasm_footprint, report)
        report["program"] = program_summary
        report["wasm_footprint"] = wasm_footprint
        report["recommendations"] = _append_multimodule_recommendations(
            report["recommendations"],
            program_summary,
            wasm_footprint,
        )
    else:
        report = _audit_single_module_core(
            inputs.manifest,
            inputs.discover,
            inputs.source,
            inputs.report_text,
            top=top,
            boot_depth=boot_depth,
        )

    if dist_dir is not None:
        fast_boot = audit_wasm64_fast_boot(
            dist_dir,
            boot_manifest=boot_manifest,
            balance_warn_ratio=balance_warn_ratio,
        )
        report["wasm64_fast_boot"] = fast_boot
        report["recommendations"] = _append_fast_boot_recommendations(
            report["recommendations"],
            fast_boot,
        )
    return sanitize_value(report)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _format_fast_boot_text_lines(fast_boot: Mapping[str, Any]) -> list[str]:
    artifacts = fast_boot.get("artifacts") or {}
    balance = fast_boot.get("split_balance") or {}
    shard_bal = fast_boot.get("shard_balance") or {}
    exports = fast_boot.get("exports") or {}
    manifest = fast_boot.get("boot_manifest") or {}
    warnings = fast_boot.get("warnings") or []
    req = exports.get("required") or {}

    shard_count = as_int(
        shard_bal.get("shard_count", balance.get("shard_count", balance.get("part_count"))),
        0,
    )
    max_shard_mib = _as_float(shard_bal.get("max_shard_mib", balance.get("max_shard_mib")), 0.0)
    total_mib = _as_float(shard_bal.get("total_mib", balance.get("total_mib")), 0.0)
    balance_ratio = _as_float(
        shard_bal.get("balance_ratio", balance.get("balance_ratio", balance.get("split_balance_ratio"))),
        0.0,
    )

    max_to_min = balance.get("max_to_min_ratio")
    max_to_min_str = f"{_as_float(max_to_min, 0.0):.2f}x" if max_to_min is not None else "inf"
    lines = [
        "wasm64 fast-boot:",
        f"  ready: {'yes' if fast_boot.get('ready') else 'no'}, "
        f"warnings: {as_int(fast_boot.get('warning_count'), 0)}"
        + (f" ({', '.join(warnings)})" if warnings else ""),
        f"  runtime.wasm: {'present' if artifacts.get('runtime_wasm_present') else 'missing'} "
        f"({as_int(artifacts.get('runtime_wasm_bytes'), 0)} bytes, "
        f"valid={'yes' if artifacts.get('runtime_wasm_valid') else 'no'}), "
        f"runtime.js: {'present' if artifacts.get('runtime_js_present') else 'missing'}",
        f"  guest_part_*.wasm: {as_int(artifacts.get('guest_part_count'), 0)} files "
        f"(valid {as_int(artifacts.get('guest_parts_valid_count'), 0)}, "
        f"invalid {as_int(artifacts.get('guest_parts_invalid_count'), 0)}), "
        f"parts bytes {as_int(artifacts.get('guest_parts_total_bytes'), 0)}, "
        f"total wasm bytes {as_int(artifacts.get('total_wasm_bytes'), 0)} "
        f"({_as_float(artifacts.get('total_wasm_mib'), 0.0):.4f} MiB)",
        f"  shard balance: shards {shard_count}, max_shard_mib {max_shard_mib:.4f} MiB, "
        f"total_mib {total_mib:.4f} MiB, balance_ratio {balance_ratio:.4f}x",
        f"  split balance: ratio {_as_float(balance.get('split_balance_ratio'), 0.0):.4f}x "
        f"(max/min {max_to_min_str}, largest share {_as_float(balance.get('max_share_ratio'), 0.0) * 100.0:.2f}%, "
        f"threshold {_as_float(balance.get('warn_threshold_ratio'), DEFAULT_SPLIT_BALANCE_WARN_RATIO):.2f}x, "
        f"dominant_warning={'yes' if balance.get('dominant_shard_warning') else 'no'})",
        f"  runtime exports: {as_int(exports.get('required_present_count'), 0)}/"
        f"{as_int(exports.get('required_total_count'), len(REQUIRED_RUNTIME_EXPORTS))} required "
        f"(_main={'ok' if req.get('_main') else 'missing'}, "
        f"_playgta6_get_input_state_ptr={'ok' if req.get('_playgta6_get_input_state_ptr') else 'missing'}, "
        f"_playgta6_audio_ring={'ok' if req.get('_playgta6_audio_ring') else 'missing'})",
        f"  boot manifest ({manifest.get('kind') or 'none'}): "
        f"present={'yes' if manifest.get('present') else 'no'}, "
        f"consistent={'yes' if manifest.get('consistent') else 'no'}, "
        f"declared={as_int(manifest.get('declared_side_module_count'), 0)}, "
        f"staged={as_int(manifest.get('staged_side_module_count'), 0)}, "
        f"missing={as_int(manifest.get('missing_on_disk_count'), 0)}, "
        f"unlisted={as_int(manifest.get('unlisted_on_disk_count'), 0)}, "
        f"size_mismatches={as_int(manifest.get('size_mismatch_count'), 0)}, "
        f"unsafe={as_int(manifest.get('unsafe_entry_count'), 0)}",
        "",
    ]
    return lines


def _format_fast_boot_markdown_lines(fast_boot: Mapping[str, Any]) -> list[str]:
    artifacts = fast_boot.get("artifacts") or {}
    balance = fast_boot.get("split_balance") or {}
    shard_bal = fast_boot.get("shard_balance") or {}
    exports = fast_boot.get("exports") or {}
    manifest = fast_boot.get("boot_manifest") or {}
    warnings = fast_boot.get("warnings") or []
    req = exports.get("required") or {}

    shard_count = as_int(
        shard_bal.get("shard_count", balance.get("shard_count", balance.get("part_count"))),
        0,
    )
    max_shard_mib = _as_float(shard_bal.get("max_shard_mib", balance.get("max_shard_mib")), 0.0)
    total_mib = _as_float(shard_bal.get("total_mib", balance.get("total_mib")), 0.0)
    balance_ratio = _as_float(
        shard_bal.get("balance_ratio", balance.get("balance_ratio", balance.get("split_balance_ratio"))),
        0.0,
    )

    max_to_min = balance.get("max_to_min_ratio")
    max_to_min_str = f"{_as_float(max_to_min, 0.0):.2f}x" if max_to_min is not None else "inf"
    lines = [
        "",
        "## Wasm64 fast-boot audit",
        "",
        f"- Ready: **{'yes' if fast_boot.get('ready') else 'no'}** "
        f"(warnings: {as_int(fast_boot.get('warning_count'), 0)}"
        + (f" — {', '.join(f'`{w}`' for w in warnings)})" if warnings else ")"),
        f"- `runtime.wasm`: {'present' if artifacts.get('runtime_wasm_present') else '**missing**'} "
        f"({as_int(artifacts.get('runtime_wasm_bytes'), 0)} bytes, "
        f"valid={bool(artifacts.get('runtime_wasm_valid'))})",
        f"- `guest_part_*.wasm`: **{as_int(artifacts.get('guest_part_count'), 0)}** files "
        f"({as_int(artifacts.get('guest_parts_total_bytes'), 0)} bytes), "
        f"total Wasm: **{as_int(artifacts.get('total_wasm_bytes'), 0)}** bytes "
        f"({_as_float(artifacts.get('total_wasm_mib'), 0.0):.4f} MiB)",
        f"- Shard balance: **{shard_count}** shards, max **{max_shard_mib:.4f} MiB**, "
        f"total **{total_mib:.4f} MiB**, balance ratio **{balance_ratio:.4f}x**",
        f"- Split balance: **{_as_float(balance.get('split_balance_ratio'), 0.0):.4f}x** "
        f"(max/min {max_to_min_str}, largest share {_as_float(balance.get('max_share_ratio'), 0.0) * 100.0:.2f}%, "
        f"dominant shard warning: `{bool(balance.get('dominant_shard_warning'))}`)",
        f"- Required runtime exports ({as_int(exports.get('required_present_count'), 0)}/"
        f"{as_int(exports.get('required_total_count'), len(REQUIRED_RUNTIME_EXPORTS))}): "
        f"`_main`={bool(req.get('_main'))}, "
        f"`_playgta6_get_input_state_ptr`={bool(req.get('_playgta6_get_input_state_ptr'))}, "
        f"`_playgta6_audio_ring`={bool(req.get('_playgta6_audio_ring'))}",
        f"- Boot manifest (`{manifest.get('kind') or 'none'}`): "
        f"present={bool(manifest.get('present'))}, consistent=**{bool(manifest.get('consistent'))}**, "
        f"missing={as_int(manifest.get('missing_on_disk_count'), 0)}, "
        f"unlisted={as_int(manifest.get('unlisted_on_disk_count'), 0)}, "
        f"size_mismatches={as_int(manifest.get('size_mismatch_count'), 0)}, "
        f"unsafe={as_int(manifest.get('unsafe_entry_count'), 0)}",
    ]
    return lines


def format_text_report(report: Mapping[str, Any]) -> str:
    fast_boot = report.get("wasm64_fast_boot")
    if "coverage" not in report and isinstance(fast_boot, Mapping):
        lines = [
            f"playgta6-recomp wasm64 fast-boot audit: {report.get('source') or '<unknown>'}",
            "",
        ]
        lines.extend(_format_fast_boot_text_lines(fast_boot))
        lines.append("recommendations:")
        for rec in report.get("recommendations") or []:
            lines.append(f"  {rec['priority']}. [{rec['category']}] {rec['detail']}")
        return "\n".join(lines) + "\n"

    cov = report["coverage"]
    shards = report["shards"]
    jump = report["jump_tables"]
    boot = report["boot"]
    lines = [
        f"playgta6-recomp coverage audit: {report.get('source') or '<unknown>'}",
        f"module: {report.get('module') or '-'}",
        "",
        "coverage:",
        f"  functions:    {cov['lifted_functions']} lifted / {cov['unlifted_functions']} unlifted "
        f"of {cov['discovered_functions']} ({cov['lifted_pct']:.2f}%)",
        f"  blocks:       {cov['lifted_blocks']} lifted / {cov['unlifted_blocks']} unlifted "
        f"({cov['block_lifted_pct']:.2f}%)",
        f"  instructions: {cov['lifted_instructions']} lifted / {cov['unlifted_instructions']} unlifted "
        f"({cov['instruction_lifted_pct']:.2f}%)",
        "",
        "shards:",
        f"  count {shards['count']}, size ({shards['unit']}) "
        f"min {shards['min']} / median {shards['median']} / max {shards['max']}",
        "",
        "jump tables:",
        f"  resolved {jump['resolved_jump_tables']}, unresolved {jump['unresolved_indirect_jumps']} "
        f"({jump['resolution_pct']:.2f}%)",
        "",
        "boot path:",
        f"  roots: {', '.join(boot.get('roots') or []) or '-'}",
        f"  reachable {boot['boot_reachable_functions']} functions, "
        f"lifted {boot['boot_lifted_pct']:.2f}%, "
        f"unresolved indirect calls {boot['boot_unresolved_indirect_calls']}",
    ]
    depth = boot.get("boot_depth")
    if depth is not None:
        lines.append(f"  depth limit: {depth}")
    for name, info in (boot.get("boot_targets") or {}).items():
        if info.get("reachable"):
            path = " -> ".join(info.get("shortest_path") or [])
            lines.append(f"  {name}: reachable via {path}")
        else:
            lines.append(f"  {name}: not reachable on discovered boot path")
    lines.append("")
    lines.append("top unlifted mnemonics:")
    mnems = report.get("unlifted_mnemonics") or []
    if not mnems:
        lines.append("  (none)")
    else:
        for row in mnems:
            lines.append(
                f"  {row['mnemonic']:<16} {row['count']:>8}  functions {row['affected_functions']}"
            )
    boot_mnems = boot.get("boot_unlifted_mnemonics") or []
    lines.append("boot unlifted mnemonics:")
    if not boot_mnems:
        lines.append("  (none)")
    else:
        for row in boot_mnems:
            lines.append(
                f"  {row['mnemonic']:<16} {row['count']:>8}  functions {row['affected_functions']}"
            )
    lines.append("")
    program = report.get("program")
    if isinstance(program, Mapping):
        prog_imp = program.get("imports") or {}
        lines.append("program modules:")
        lines.append(
            f"  modules: {program.get('module_count', 0)} "
            f"({program.get('exe_count', 0)} exe, {program.get('guest_dll_count', 0)} guest dlls), "
            f"total shards {program.get('total_shards', 0)}"
        )
        if program.get("init_order"):
            lines.append(f"  init order: {', '.join(program['init_order'])}")
        if program.get("indirect_dispatch"):
            lines.append(f"  indirect dispatch: {program['indirect_dispatch']}")
        lines.append(
            f"  program imports: {prog_imp.get('total', 0)} total "
            f"(guest {prog_imp.get('guest', 0)}, runtime {prog_imp.get('runtime', 0)}, "
            f"host bound {prog_imp.get('host_bound', 0)}, "
            f"unresolved {prog_imp.get('host_unresolved', 0)})"
        )
        for mod in program.get("modules") or []:
            mod_imp = mod.get("imports") or {}
            lines.append(
                f"  - {mod['module']} ({mod['role']}, {mod['dll']}): "
                f"{mod['lifted_functions']}/{mod['discovered_functions']} fn ({mod['lifted_pct']:.2f}%), "
                f"shards {mod['shard_count']}, "
                f"imports {mod_imp.get('total', 0)} "
                f"(guest {mod_imp.get('guest', 0)}, host {mod_imp.get('host_bound', 0)}, "
                f"unresolved {mod_imp.get('host_unresolved', 0)})"
            )
        unres_by_dll = program.get("unresolved_by_dll") or []
        if unres_by_dll:
            lines.append("  unresolved imports by dll:")
            for row in unres_by_dll:
                lines.append(f"    {row['dll']}: {row['count']}")
        lines.append("")

    wasm_fp = report.get("wasm_footprint")
    if isinstance(wasm_fp, Mapping):
        lines.append("wasm / GOT.func footprint:")
        if wasm_fp.get("available"):
            lines.append(
                f"  indirect dispatch: {wasm_fp.get('indirect_dispatch') or '-'}, "
                f"split modules: {wasm_fp.get('split_modules', 0)}, "
                f"side modules: {wasm_fp.get('side_module_count', 0)} "
                f"({wasm_fp.get('side_module_shard_count', 0)} shards)"
            )
            lines.append(
                f"  total size: {_as_float(wasm_fp.get('total_size_mib'), 0.0):.2f} MiB "
                f"({as_int(wasm_fp.get('total_wasm_bytes'), 0)} bytes)"
            )
            fp_bal = wasm_fp.get("shard_balance")
            if isinstance(fp_bal, Mapping):
                lines.append(
                    f"  shard balance: shards {as_int(fp_bal.get('shard_count'), 0)}, "
                    f"max_shard_mib {_as_float(fp_bal.get('max_shard_mib'), 0.0):.4f} MiB, "
                    f"total_mib {_as_float(fp_bal.get('total_mib'), 0.0):.4f} MiB, "
                    f"balance_ratio {_as_float(fp_bal.get('balance_ratio'), 0.0):.4f}x"
                )
            lines.append(
                f"  GOT.func imports: {as_int(wasm_fp.get('total_got_func_imports'), 0)} "
                f"(sub_*: {as_int(wasm_fp.get('got_func_sub_imports'), 0)}, "
                f"shard_dispatch: {as_int(wasm_fp.get('got_func_shard_dispatch_imports'), 0)}), "
                f"GOT.mem: {as_int(wasm_fp.get('total_got_mem_imports'), 0)}"
            )
            lines.append(
                f"  wasmTable size: {as_int(wasm_fp.get('total_table_size'), 0)} "
                f"(main: {as_int(wasm_fp.get('main_table_size'), 0)}, "
                f"side: {as_int(wasm_fp.get('side_table_size'), 0)}), "
                f"shard dispatch exports: {as_int(wasm_fp.get('shard_dispatch_exports'), 0)}"
            )
            for mod in wasm_fp.get("modules") or []:
                lines.append(
                    f"  - {mod['name']} ({mod['role']}): "
                    f"{_as_float(mod.get('size_mib'), 0.0):.2f} MiB, "
                    f"GOT.func {mod['got_func_imports']}, "
                    f"table {mod['table_size']}, "
                    f"func imports {mod['func_imports']}, "
                    f"func exports {mod['func_exports']}"
                )
        else:
            lines.append(
                f"  guest_manifest.json not found; "
                f"indirect dispatch: {wasm_fp.get('indirect_dispatch') or 'table'}, "
                f"estimated GOT.func entries: {wasm_fp.get('estimated_got_func_entries', 0)}"
            )
        lines.append("")

    if isinstance(fast_boot, Mapping):
        lines.extend(_format_fast_boot_text_lines(fast_boot))

    lines.append("recommendations:")
    for rec in report.get("recommendations") or []:
        lines.append(f"  {rec['priority']}. [{rec['category']}] {rec['detail']}")
    return "\n".join(lines) + "\n"


def format_markdown_report(report: Mapping[str, Any]) -> str:
    fast_boot = report.get("wasm64_fast_boot")
    if "coverage" not in report and isinstance(fast_boot, Mapping):
        lines = [
            f"# Wasm64 fast-boot audit (`{report.get('source') or 'unknown'}`)",
        ]
        lines.extend(_format_fast_boot_markdown_lines(fast_boot))
        lines.extend(["", "## Recommendations", ""])
        for rec in report.get("recommendations") or []:
            lines.append(f"{rec['priority']}. **{rec['category']}** — {rec['detail']}")
        return "\n".join(lines) + "\n"

    cov = report["coverage"]
    shards = report["shards"]
    jump = report["jump_tables"]
    boot = report["boot"]
    lines = [
        f"# Recomp coverage audit (`{report.get('source') or 'unknown'}`)",
        "",
        f"- Module: `{report.get('module') or '-'}`",
        f"- Functions: **{cov['lifted_functions']}** lifted / **{cov['unlifted_functions']}** unlifted "
        f"of {cov['discovered_functions']} (**{cov['lifted_pct']:.2f}%**)",
        f"- Blocks: {cov['lifted_blocks']} / {cov['unlifted_blocks']} ({cov['block_lifted_pct']:.2f}%)",
        f"- Instructions: {cov['lifted_instructions']} / {cov['unlifted_instructions']} "
        f"({cov['instruction_lifted_pct']:.2f}%)",
        f"- Shards: {shards['count']} "
        f"(min {shards['min']}, median {shards['median']}, max {shards['max']} {shards['unit']})",
        f"- Jump tables: {jump['resolved_jump_tables']} resolved, "
        f"{jump['unresolved_indirect_jumps']} unresolved ({jump['resolution_pct']:.2f}%)",
        "",
        "## Boot path",
        "",
        f"- Reachable functions: {boot['boot_reachable_functions']}",
        f"- Boot lifted: {boot['boot_lifted_pct']:.2f}%",
        f"- Unresolved indirect calls: {boot['boot_unresolved_indirect_calls']}",
        f"- Roots: {', '.join(f'`{r}`' for r in (boot.get('roots') or [])) or '_'}",
        "",
    ]
    for name, info in (boot.get("boot_targets") or {}).items():
        if info.get("reachable"):
            path = " → ".join(f"`{p}`" for p in (info.get("shortest_path") or []))
            lines.append(f"- `{name}`: reachable {path}")
        else:
            lines.append(f"- `{name}`: **not reachable**")
    lines.extend(["", "## Top unlifted mnemonics", ""])
    mnems = report.get("unlifted_mnemonics") or []
    if not mnems:
        lines.append("_None._")
    else:
        lines.append("| Mnemonic | Count | Functions |")
        lines.append("| --- | ---: | ---: |")
        for row in mnems:
            lines.append(f"| `{row['mnemonic']}` | {row['count']} | {row['affected_functions']} |")

    program = report.get("program")
    if isinstance(program, Mapping):
        prog_imp = program.get("imports") or {}
        lines.extend(
            [
                "",
                "## Program modules",
                "",
                f"- Modules: **{program.get('module_count', 0)}** "
                f"({program.get('exe_count', 0)} exe, {program.get('guest_dll_count', 0)} guest DLLs)",
                f"- Total shards: **{program.get('total_shards', 0)}**",
                f"- Program imports: {prog_imp.get('total', 0)} total "
                f"({prog_imp.get('guest', 0)} guest, {prog_imp.get('runtime', 0)} runtime, "
                f"{prog_imp.get('host_bound', 0)} host bound, "
                f"**{prog_imp.get('host_unresolved', 0)}** unresolved)",
            ]
        )
        if program.get("indirect_dispatch"):
            lines.append(f"- Indirect dispatch: `{program['indirect_dispatch']}`")
        if program.get("modules"):
            lines.extend(
                [
                    "",
                    "| Module | Role | DLL | Lifted / Discovered | Shards | Imports (Guest / Host / Unresolved) |",
                    "| --- | --- | --- | ---: | ---: | ---: |",
                ]
            )
            for mod in program["modules"]:
                mod_imp = mod.get("imports") or {}
                lines.append(
                    f"| `{mod['module']}` | `{mod['role']}` | `{mod['dll']}` | "
                    f"{mod['lifted_functions']} / {mod['discovered_functions']} ({mod['lifted_pct']:.2f}%) | "
                    f"{mod['shard_count']} | "
                    f"{mod_imp.get('guest', 0)} / {mod_imp.get('host_bound', 0)} / "
                    f"{mod_imp.get('host_unresolved', 0)} |"
                )
        unres_by_dll = program.get("unresolved_by_dll") or []
        if unres_by_dll:
            lines.extend(["", "### Unresolved imports by DLL", ""])
            for row in unres_by_dll:
                lines.append(f"- `{row['dll']}`: {row['count']}")

    wasm_fp = report.get("wasm_footprint")
    if isinstance(wasm_fp, Mapping):
        lines.extend(["", "## Wasm & GOT.func footprint", ""])
        if wasm_fp.get("available"):
            lines.extend(
                [
                    f"- Indirect dispatch: `{wasm_fp.get('indirect_dispatch') or '-'}`",
                    f"- Side modules: {wasm_fp.get('side_module_count', 0)} "
                    f"({wasm_fp.get('side_module_shard_count', 0)} shards)",
                    f"- Total Wasm size: {_as_float(wasm_fp.get('total_size_mib'), 0.0):.2f} MiB",
                    *(
                        [
                            f"- Shard balance: **{as_int(wasm_fp['shard_balance'].get('shard_count'), 0)}** shards, "
                            f"max **{_as_float(wasm_fp['shard_balance'].get('max_shard_mib'), 0.0):.4f} MiB**, "
                            f"total **{_as_float(wasm_fp['shard_balance'].get('total_mib'), 0.0):.4f} MiB**, "
                            f"balance ratio **{_as_float(wasm_fp['shard_balance'].get('balance_ratio'), 0.0):.4f}x**"
                        ]
                        if isinstance(wasm_fp.get("shard_balance"), Mapping)
                        else []
                    ),
                    f"- `GOT.func` imports: **{as_int(wasm_fp.get('total_got_func_imports'), 0)}** "
                    f"(`sub_*`: {as_int(wasm_fp.get('got_func_sub_imports'), 0)}, "
                    f"shard dispatch: {as_int(wasm_fp.get('got_func_shard_dispatch_imports'), 0)})",
                    f"- `wasmTable` size: **{as_int(wasm_fp.get('total_table_size'), 0)}** "
                    f"(main: {as_int(wasm_fp.get('main_table_size'), 0)}, "
                    f"side: {as_int(wasm_fp.get('side_table_size'), 0)})",
                    f"- Shard dispatch exports: {as_int(wasm_fp.get('shard_dispatch_exports'), 0)}",
                ]
            )
            if wasm_fp.get("modules"):
                lines.extend(
                    [
                        "",
                        "| Wasm Module | Role | Size (MiB) | GOT.func | TableSize | Func Imports | Func Exports |",
                        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for mod in wasm_fp["modules"]:
                    lines.append(
                        f"| `{mod['name']}` | `{mod['role']}` | "
                        f"{_as_float(mod.get('size_mib'), 0.0):.2f} | "
                        f"{mod['got_func_imports']} | {mod['table_size']} | "
                        f"{mod['func_imports']} | {mod['func_exports']} |"
                    )
        else:
            lines.append(
                f"- `guest_manifest.json` not present; indirect dispatch `{wasm_fp.get('indirect_dispatch') or 'table'}`, "
                f"estimated `GOT.func` entries: **{wasm_fp.get('estimated_got_func_entries', 0)}**"
            )

    if isinstance(fast_boot, Mapping):
        lines.extend(_format_fast_boot_markdown_lines(fast_boot))

    lines.extend(["", "## Recommendations", ""])
    for rec in report.get("recommendations") or []:
        lines.append(f"{rec['priority']}. **{rec['category']}** — {rec['detail']}")
    return "\n".join(lines) + "\n"


def format_json_report(report: Mapping[str, Any]) -> str:
    return json.dumps(sanitize_value(report), indent=2, sort_keys=False) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit playgta6-recomp build coverage, multi-module programs, Wasm/GOT.func "
            "footprint, browser fast-boot Wasm64 staging (--dist-dir), and boot-path "
            "lift bottlenecks from build-manifest.json, program-manifest.json, "
            "guest_manifest.json, and/or discover --json."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help=(
            "Path to a recomp build directory, program-manifest.json, guest_manifest.json, "
            "build-manifest.json, or discover JSON file (optional when --dist-dir is passed)."
        ),
    )
    parser.add_argument(
        "--dist-dir",
        default=None,
        metavar="DIR",
        help=(
            "Run browser fast-boot Wasm64 audit on staged dist directory "
            "(runtime.wasm, guest_part_*.wasm, required exports, split-balance, boot manifest)."
        ),
    )
    parser.add_argument(
        "--boot-manifest",
        default=None,
        metavar="FILE",
        help=(
            "Optional explicit boot manifest (guest_manifest.json or boot-manifest.json) "
            "to validate against --dist-dir."
        ),
    )
    parser.add_argument(
        "--balance-warn-ratio",
        type=float,
        default=DEFAULT_SPLIT_BALANCE_WARN_RATIO,
        metavar="R",
        help=(
            "Max-to-mean byte ratio threshold above which a single guest_part_*.wasm "
            f"shard triggers a dominant-shard warning (default {DEFAULT_SPLIT_BALANCE_WARN_RATIO})."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        metavar="N",
        help=f"Top-N unlifted mnemonics to rank (default {DEFAULT_TOP}).",
    )
    parser.add_argument(
        "--boot-depth",
        type=int,
        default=None,
        metavar="D",
        help="Maximum BFS depth from entry/TLS roots (default: unlimited).",
    )
    parser.add_argument(
        "--fail-on-unresolved",
        type=int,
        nargs="?",
        const=0,
        default=None,
        metavar="N",
        help=(
            f"Exit with code {EXIT_THRESHOLD_FAILED} if unresolved imports exceed N "
            "(defaults to 0 when flag is passed without a value)."
        ),
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit a sanitized Markdown report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a sanitized machine-readable JSON report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.fail_on_unresolved is not None and args.fail_on_unresolved < 0:
            raise AuditError("--fail-on-unresolved must be >= 0")
        if args.boot_manifest is not None and args.dist_dir is None:
            raise AuditError("--boot-manifest requires --dist-dir")
        if args.input is None and args.dist_dir is None:
            raise AuditError("either an input path or --dist-dir is required")

        if args.input is None:
            report = audit_dist_dir(
                args.dist_dir,
                boot_manifest=args.boot_manifest,
                balance_warn_ratio=args.balance_warn_ratio,
            )
        else:
            report = audit_recomp_build(
                args.input,
                top=args.top,
                boot_depth=args.boot_depth,
                dist_dir=args.dist_dir,
                boot_manifest=args.boot_manifest,
                balance_warn_ratio=args.balance_warn_ratio,
            )
    except AuditError as exc:
        print(f"Error: {sanitize_text(str(exc))}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(
            f"Error: {sanitize_text(exc.strerror or type(exc).__name__)}",
            file=sys.stderr,
        )
        return 1

    if args.json:
        sys.stdout.write(format_json_report(report))
    elif args.markdown:
        sys.stdout.write(format_markdown_report(report))
    else:
        sys.stdout.write(format_text_report(report))

    if args.fail_on_unresolved is not None:
        passed, count = check_unresolved_threshold(report, args.fail_on_unresolved)
        if not passed:
            print(
                f"Threshold failed: {count} unresolved import(s) exceed "
                f"--fail-on-unresolved={args.fail_on_unresolved}",
                file=sys.stderr,
            )
            return EXIT_THRESHOLD_FAILED
    return 0


if __name__ == "__main__":
    sys.exit(main())
