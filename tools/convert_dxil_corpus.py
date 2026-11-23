#!/usr/bin/env python3
"""Offline DXIL -> SPIR-V -> WGSL conversion harness (Track 4 / S11).

Discovers `.dxbc` / `.dxil` blobs from a local corpus directory (directory glob
plus optional S10 `corpus-manifest.json`), then runs a SHA-256 cached parallel
pipeline:

  1. DXIL -> `.spv` via `dxil-spirv` (records `missing_tool:dxil-spirv` if absent)
  2. optional `spirv-val` when installed
  3. `.spv` -> `.wgsl` + `.wgsl.pack` + S6 `reflection-manifest.json` via
     `playgta6-shaders convert` / `playgta6-shaders reflect`

CLI:
    python tools/convert_dxil_corpus.py --corpus-dir <dir> \\
        [--out-dir local-data/shaders/converted] [--jobs N] \\
        [--dry-run] [--check-only] [--json]

Security invariants:
  - Pure Python 3.12 standard library.
  - Refuses to write into tracked repository source trees (allows gitignored
    `local-data/`, `work/`, or directories outside the repo).
  - JSON + Markdown coverage reports never contain host paths, usernames, or
    raw shader bytes. Path sanitization splits on both `\\` and `/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import inspect_wgsl_pack as wgpk  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = _REPO_ROOT
DEFAULT_OUT_DIR = REPO_ROOT / "local-data" / "shaders" / "converted"
CORPUS_MANIFEST_NAME = "corpus-manifest.json"
COVERAGE_JSON_NAME = "coverage.json"
COVERAGE_MD_NAME = "coverage.md"
REFLECTION_MANIFEST_NAME = "reflection-manifest.json"
LAYOUT_MANIFEST_NAME = "layout-manifest.json"
SHADER_INDEX_NAME = "shader-index.json"
PACK_FILE_NAME = "converted.wgsl.pack"
SPV_FILE_NAME = "converted.spv"
STATUS_FILE_NAME = "status.json"
SCHEMA_VERSION = 1

ALLOWED_REPO_OUTPUT_DIR_NAMES: tuple[str, ...] = ("local-data", "work")
SHADER_EXTENSIONS: frozenset[str] = frozenset({".dxbc", ".dxil"})
PLAYGTA6_SHADERS_BINARIES: tuple[str, ...] = ("playgta6-shaders.exe", "playgta6-shaders")

FAILURE_BUCKETS: tuple[str, ...] = (
    "missing_tool",
    "dxil_parse_error",
    "unsupported_sm",
    "rdna2_precompiled_only",
    "spirv_validation_error",
    "naga_unsupported_cap",
    "bind_group_limit",
    "workgroup_limit",
    "unknown_error",
)

STAGE_DXIL_TO_SPIRV = "dxil_to_spirv"
STAGE_SPIRV_VAL = "spirv_val"
STAGE_SPIRV_TO_WGSL = "spirv_to_wgsl"

WhichFn = Callable[[str], str | None]


# ---------------------------------------------------------------------------
# Sanitization (Windows + POSIX separators on any host)
# ---------------------------------------------------------------------------

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:[\\/]")


class ConvertError(ValueError):
    """Raised when corpus conversion cannot start (bad args, unsafe output)."""

    def __init__(self, message: str, *, code: str = "convert_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class UnsafeOutputPathError(ConvertError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsafe_output_path")


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


def posix_rel_key(raw: str | Path) -> str:
    """Normalize mixed Windows/POSIX path text to a `/`-separated relative key."""
    text = str(raw).strip().replace("\\", "/")
    if _DRIVE_PREFIX_RE.match(text.replace("/", "\\")) or (
        len(text) >= 2 and text[1] == ":" and (len(text) == 2 or text[2] == "/")
    ):
        text = text[2:].lstrip("/")
    text = text.lstrip("/")
    parts = [p for p in text.split("/") if p not in ("", ".", "..")]
    if parts and len(parts[0]) == 2 and parts[0][1] == ":":
        parts = parts[1:]
    return "/".join(parts)


def sanitize_source_label(source: str | Path | None) -> str:
    """Basename-only label; splits on both `\\` and `/` regardless of host OS."""
    if source is None:
        return "<memory>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    if raw.startswith("<") and raw.endswith(">"):
        return raw
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return scrub_sensitive_text(base)


def sanitize_relative_path(path: str | Path, base_dir: Path | None = None) -> str:
    """POSIX relative path under `base_dir`, else basename. Host-OS independent."""
    raw = str(path)
    if base_dir is not None:
        try:
            resolved = Path(path)
            rel = resolved.resolve().relative_to(Path(base_dir).resolve())
            key = posix_rel_key(rel.as_posix())
            if key:
                return scrub_sensitive_text(key)
        except (ValueError, OSError):
            pass
        key = posix_rel_key(raw)
        if key:
            # Drop absolute host prefixes that survived Path() on POSIX
            # (e.g. `C:/Users/alice/corpus/foo.dxil` while tests run on Linux).
            base_name = Path(base_dir).name
            parts = key.split("/")
            if base_name in parts:
                idx = parts.index(base_name)
                tail = "/".join(parts[idx + 1 :])
                if tail:
                    return scrub_sensitive_text(tail)
            return scrub_sensitive_text(key)
    return sanitize_source_label(path)


def scrub_sensitive_text(text: str) -> str:
    """Strip Windows/POSIX host paths and the current username from arbitrary text."""
    sanitized = _WINDOWS_PATH_RE.sub("<path>", text)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        sanitized = pattern.sub("<user>", sanitized)
    return sanitized


def sanitize_message(text: str | None, *, limit: int = 500) -> str:
    """Sanitize a tool stderr/stdout snippet: no host paths, usernames, or raw bytes."""
    if not text:
        return ""
    printable = "".join(ch if (ch.isprintable() or ch in "\n\t") else " " for ch in text)
    collapsed = re.sub(r"[ \t]+", " ", printable).strip()
    collapsed = scrub_sensitive_text(collapsed)
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 3] + "..."
    return collapsed


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings in JSON-able structures."""
    if isinstance(value, str):
        return sanitize_message(value, limit=2000)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): sanitize_value(item) for key, item in value.items()}
    return value


# ---------------------------------------------------------------------------
# Safety guard
# ---------------------------------------------------------------------------


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def is_safe_output_dir(path: Path | str, *, repo_root: Path | str = REPO_ROOT) -> bool:
    """True iff `path` may receive generated conversion artifacts."""
    target = Path(path).expanduser().resolve()
    root = Path(repo_root).resolve()
    if not _is_under(target, root):
        return True
    for name in ALLOWED_REPO_OUTPUT_DIR_NAMES:
        if _is_under(target, root / name):
            return True
    return False


def validate_output_dir(path: Path | str, *, repo_root: Path | str = REPO_ROOT) -> Path:
    target = Path(path).expanduser().resolve()
    if not is_safe_output_dir(target, repo_root=repo_root):
        raise UnsafeOutputPathError(
            "Refusing to write conversion artifacts into a tracked repository "
            "source directory. Point --out-dir at gitignored local-data/, work/, or a temp directory."
        )
    return target


# ---------------------------------------------------------------------------
# Tool detection & command runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


CommandRunner = Callable[..., CommandResult]


@dataclass(frozen=True)
class Toolset:
    dxil_spirv: str | None
    spirv_val: str | None
    playgta6_shaders: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dxil-spirv": {"available": self.dxil_spirv is not None},
            "spirv-val": {"available": self.spirv_val is not None},
            "playgta6-shaders": {"available": self.playgta6_shaders is not None},
        }


def default_command_runner(
    argv: Sequence[str],
    *,
    timeout: float | None = 120.0,
    cwd: str | Path | None = None,
) -> CommandResult:
    try:
        proc = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            check=False,
        )
    except FileNotFoundError as exc:
        name = Path(str(exc.filename or (argv[0] if argv else "tool"))).name
        return CommandResult(returncode=127, stdout="", stderr=f"missing_tool:{name}")
    except subprocess.TimeoutExpired:
        return CommandResult(returncode=124, stdout="", stderr="tool timed out")
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        return CommandResult(returncode=127, stdout="", stderr=f"missing_tool:{reason}")
    return CommandResult(
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _which_with_exe(which_fn: WhichFn, name: str) -> str | None:
    found = which_fn(name)
    if found:
        return found
    if not name.lower().endswith(".exe"):
        return which_fn(f"{name}.exe")
    return None


def find_playgta6_shaders(
    *,
    repo_root: Path | str = REPO_ROOT,
    which_fn: WhichFn = shutil.which,
) -> str | None:
    """Prefer `target/release`, then `target/debug`, then PATH (workspace or crate)."""
    root = Path(repo_root)
    search_dirs = (
        root / "target" / "release",
        root / "target" / "debug",
        root / "shaders" / "target" / "release",
        root / "shaders" / "target" / "debug",
    )
    for directory in search_dirs:
        for binary in PLAYGTA6_SHADERS_BINARIES:
            candidate = directory / binary
            if candidate.is_file():
                return str(candidate)
    return _which_with_exe(which_fn, "playgta6-shaders")


def detect_tools(
    *,
    repo_root: Path | str = REPO_ROOT,
    which_fn: WhichFn = shutil.which,
) -> Toolset:
    """Locate conversion tools. `which_fn` is injectable for unit tests."""
    return Toolset(
        dxil_spirv=_which_with_exe(which_fn, "dxil-spirv"),
        spirv_val=_which_with_exe(which_fn, "spirv-val"),
        playgta6_shaders=find_playgta6_shaders(repo_root=repo_root, which_fn=which_fn),
    )


# ---------------------------------------------------------------------------
# Failure classifier
# ---------------------------------------------------------------------------


def classify_failure(stderr: str, returncode: int, stage: str) -> str:
    """Bucket a tool failure into a structured coverage category.

    Matching is case-insensitive against combined stderr (and the stage name).
    Never inspects raw binary payloads.
    """
    text = f"{stage}\n{stderr or ''}".lower()
    combined = sanitize_message(text, limit=4000).lower()

    if (
        returncode in (127, 126)
        or "missing_tool" in combined
        or "not found" in combined
        or "no such file" in combined
        or "is not recognized" in combined
        or "cannot find the file" in combined
    ):
        return "missing_tool"

    if stage == STAGE_SPIRV_VAL or any(
        token in combined
        for token in ("spirv-val", "spv_error", "spir-v validation", "spirv validation")
    ):
        if returncode != 0 and stage == STAGE_SPIRV_VAL:
            return "spirv_validation_error"

    if any(
        token in combined
        for token in (
            "rdna2",
            "rdna 2",
            "precompiled isa",
            "scarlett isa",
            "gcn isa",
            "shader isa",
            "rdna2_precompiled",
        )
    ):
        return "rdna2_precompiled_only"

    if any(
        token in combined
        for token in (
            "unsupported shader model",
            "unsupported sm",
            "shader model",
            "sm_5",
            "sm5.",
            "sm_4",
            "dxbc sm5",
            "unsupported_sm",
        )
    ):
        return "unsupported_sm"

    if any(
        token in combined
        for token in (
            "dxil parse",
            "dxbc parse",
            "invalid dxbc",
            "invalid dxil",
            "failed to parse dx",
            "not a dxbc",
            "truncated dxbc",
            "dxil_parse",
        )
    ):
        return "dxil_parse_error"

    if any(
        token in combined
        for token in (
            "naga",
            "unsupported capability",
            "unsupported cap",
            "capability not supported",
            "unimplemented cap",
            "opcapability",
        )
    ):
        return "naga_unsupported_cap"

    if any(
        token in combined
        for token in (
            "bind group",
            "bindgroup",
            "maxbindgroups",
            "max_bind_groups",
            "descriptor set",
            "too many bind",
        )
    ):
        return "bind_group_limit"

    if any(
        token in combined
        for token in (
            "workgroup",
            "local_size",
            "localsize",
            "invocations per workgroup",
            "workgroup_size",
        )
    ):
        return "workgroup_limit"

    if any(
        token in combined
        for token in ("spirv-val", "spv_error", "invalid spir-v", "invalid spirv")
    ):
        return "spirv_validation_error"

    if returncode != 0:
        return "unknown_error"
    return "unknown_error"


# ---------------------------------------------------------------------------
# Corpus discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusEntry:
    path: Path
    rel_path: str
    sha256: str | None = None
    source: str = "glob"  # glob | manifest

    def label(self) -> str:
        return self.rel_path or sanitize_source_label(self.path)


def _is_shader_blob(path: Path) -> bool:
    return path.suffix.lower() in SHADER_EXTENSIONS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path_fields(item: Mapping[str, Any]) -> list[str]:
    fields: list[str] = []
    for key in ("path", "relative_path", "rel_path", "file", "source", "name"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            fields.append(value.strip())
    return fields


def _iter_manifest_items(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return list(payload)
    if not isinstance(payload, dict):
        return []
    for key in ("entries", "files", "blobs", "shaders", "corpus"):
        value = payload.get(key)
        if isinstance(value, list):
            return list(value)
    return []


def _resolve_under_corpus(raw: str, corpus_dir: Path) -> Path | None:
    """Resolve a manifest path that may use Windows separators or a drive prefix."""
    key = posix_rel_key(raw)
    if not key:
        return None
    direct = corpus_dir / Path(*key.split("/"))
    try:
        if direct.is_file() and _is_shader_blob(direct):
            return direct
    except OSError:
        return None
    # Basename fallback (absolute host paths recorded by S10 on another OS).
    base = key.split("/")[-1]
    if base and _is_shader_blob(Path(base)):
        fallback = corpus_dir / base
        try:
            if fallback.is_file():
                return fallback
        except OSError:
            return None
    return None


def load_corpus_manifest(manifest_path: Path, corpus_dir: Path) -> list[CorpusEntry]:
    """Parse S10 `corpus-manifest.json` (flexible entries/files list)."""
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []

    entries: list[CorpusEntry] = []
    for item in _iter_manifest_items(payload):
        if isinstance(item, str):
            resolved = _resolve_under_corpus(item, corpus_dir)
            if resolved is None:
                continue
            rel = sanitize_relative_path(resolved, corpus_dir)
            if not rel or rel == "<unknown>":
                rel = posix_rel_key(item) or sanitize_source_label(item)
            entries.append(CorpusEntry(path=resolved, rel_path=rel, source="manifest"))
            continue
        if not isinstance(item, Mapping):
            continue
        sha_raw = item.get("sha256") or item.get("hash")
        sha = str(sha_raw).lower() if isinstance(sha_raw, str) and sha_raw else None
        for field_value in _manifest_path_fields(item):
            resolved = _resolve_under_corpus(field_value, corpus_dir)
            if resolved is None:
                # Keep a record if the file exists as written (POSIX hosts).
                try:
                    as_path = Path(field_value)
                    if as_path.is_file() and _is_shader_blob(as_path):
                        resolved = as_path
                except OSError:
                    resolved = None
            if resolved is None:
                continue
            rel = sanitize_relative_path(resolved, corpus_dir)
            if not rel or rel.startswith("<"):
                rel = posix_rel_key(field_value) or sanitize_source_label(field_value)
            entries.append(
                CorpusEntry(path=resolved, rel_path=rel, sha256=sha, source="manifest")
            )
            break
    return entries


def glob_shader_blobs(corpus_dir: Path) -> list[CorpusEntry]:
    found: list[CorpusEntry] = []
    try:
        stack = [corpus_dir]
        while stack:
            current = stack.pop()
            try:
                children = list(current.iterdir())
            except OSError:
                continue
            for child in children:
                try:
                    if child.is_dir():
                        stack.append(child)
                    elif child.is_file() and _is_shader_blob(child):
                        found.append(
                            CorpusEntry(
                                path=child,
                                rel_path=sanitize_relative_path(child, corpus_dir),
                                source="glob",
                            )
                        )
                except OSError:
                    continue
    except OSError:
        return []
    found.sort(key=lambda entry: entry.rel_path)
    return found


def discover_corpus(corpus_dir: Path | str) -> list[CorpusEntry]:
    """Discover `.dxbc`/`.dxil` blobs via S10 manifest and/or directory globbing."""
    root = Path(corpus_dir)
    by_resolved: dict[str, CorpusEntry] = {}

    manifest = root / CORPUS_MANIFEST_NAME
    if manifest.is_file():
        for entry in load_corpus_manifest(manifest, root):
            key = str(entry.path)
            by_resolved[key] = entry

    for entry in glob_shader_blobs(root):
        key = str(entry.path)
        if key not in by_resolved:
            by_resolved[key] = entry

    discovered = list(by_resolved.values())
    discovered.sort(key=lambda entry: (entry.rel_path, str(entry.path)))
    return discovered


# ---------------------------------------------------------------------------
# Conversion pipeline
# ---------------------------------------------------------------------------


@dataclass
class ShaderResult:
    source: str
    sha256: str
    status: str
    stages: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    failure_code: str | None = None
    failure_stage: str | None = None
    failure_message: str | None = None
    missing_tool: str | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "sha256": self.sha256,
            "status": self.status,
            "stages": list(self.stages),
            "artifacts": list(self.artifacts),
            "cached": self.cached,
        }
        if self.failure_code is not None:
            payload["failure"] = {
                "code": self.failure_code,
                "stage": self.failure_stage,
                "message": self.failure_message or "",
            }
            if self.missing_tool:
                payload["failure"]["tool"] = self.missing_tool
        else:
            payload["failure"] = None
        return payload


@dataclass
class CoverageReport:
    corpus_label: str
    tools: Toolset
    results: list[ShaderResult]
    discovered: int
    dry_run: bool = False
    check_only: bool = False

    @property
    def converted(self) -> int:
        return sum(1 for item in self.results if item.status == "converted")

    @property
    def cached_count(self) -> int:
        return sum(1 for item in self.results if item.status == "cached")

    @property
    def failed(self) -> int:
        return sum(1 for item in self.results if item.status == "failed")

    def buckets(self) -> dict[str, int]:
        counts = {name: 0 for name in FAILURE_BUCKETS}
        for item in self.results:
            if item.failure_code in counts:
                counts[item.failure_code] += 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.failed == 0 or self.dry_run or self.check_only,
            "schema_version": SCHEMA_VERSION,
            "corpus": self.corpus_label,
            "discovered": self.discovered,
            "converted": self.converted,
            "cached": self.cached_count,
            "failed": self.failed,
            "dryRun": self.dry_run,
            "checkOnly": self.check_only,
            "tools": self.tools.to_dict(),
            "buckets": self.buckets(),
            "results": [item.to_dict() for item in self.results],
        }


def _failed(
    entry: CorpusEntry,
    digest: str,
    *,
    stage: str,
    stderr: str,
    returncode: int,
    tool: str | None = None,
    stages: list[str] | None = None,
    artifacts: list[str] | None = None,
) -> ShaderResult:
    code = classify_failure(stderr, returncode, stage)
    missing = None
    if code == "missing_tool":
        missing = tool or stage
        message = f"missing_tool:{missing}"
    else:
        message = sanitize_message(stderr) or code
    return ShaderResult(
        source=entry.label(),
        sha256=digest,
        status="failed",
        stages=list(stages or []),
        artifacts=list(artifacts or []),
        failure_code=code,
        failure_stage=stage,
        failure_message=message,
        missing_tool=missing,
    )


def _cache_dir(out_dir: Path, digest: str) -> Path:
    return out_dir / "cache" / digest[:2] / digest


def _read_cached(entry: CorpusEntry, digest: str, cache_dir: Path) -> ShaderResult | None:
    status_path = cache_dir / STATUS_FILE_NAME
    pack_path = cache_dir / PACK_FILE_NAME
    spv_path = cache_dir / SPV_FILE_NAME
    if not status_path.is_file() or not pack_path.is_file() or not spv_path.is_file():
        return None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "converted":
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = [SPV_FILE_NAME, PACK_FILE_NAME]
    return ShaderResult(
        source=entry.label(),
        sha256=digest,
        status="cached",
        stages=list(payload.get("stages") or [STAGE_DXIL_TO_SPIRV, STAGE_SPIRV_TO_WGSL]),
        artifacts=[sanitize_source_label(str(name)) for name in artifacts],
        cached=True,
    )


def _write_status(cache_dir: Path, result: ShaderResult) -> None:
    payload = {
        "status": result.status,
        "sha256": result.sha256,
        "stages": result.stages,
        "artifacts": result.artifacts,
    }
    (cache_dir / STATUS_FILE_NAME).write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _playgta6_argv_convert(binary: str, spv_dir: Path, out_dir: Path, pack_path: Path) -> list[str]:
    return [
        binary,
        "convert",
        "--spirv",
        str(spv_dir),
        "--out",
        str(out_dir),
        "--pack",
        str(pack_path),
    ]


def _playgta6_argv_reflect(binary: str, spv_dir: Path, out_dir: Path) -> list[str]:
    return [
        binary,
        "reflect",
        "--spirv",
        str(spv_dir),
        "--out",
        str(out_dir),
    ]


def _collect_artifacts(cache_dir: Path) -> list[str]:
    names: list[str] = []
    for name in (
        SPV_FILE_NAME,
        PACK_FILE_NAME,
        SHADER_INDEX_NAME,
        REFLECTION_MANIFEST_NAME,
        LAYOUT_MANIFEST_NAME,
    ):
        if (cache_dir / name).is_file():
            names.append(name)
    for child in sorted(cache_dir.glob("*.wgsl")):
        names.append(child.name)
    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def convert_one(
    entry: CorpusEntry,
    *,
    out_dir: Path,
    tools: Toolset,
    command_runner: CommandRunner,
) -> ShaderResult:
    digest = entry.sha256 or _sha256_file(entry.path)
    cache_dir = _cache_dir(out_dir, digest)
    cached = _read_cached(entry, digest, cache_dir)
    if cached is not None:
        return cached

    cache_dir.mkdir(parents=True, exist_ok=True)
    stages: list[str] = []
    artifacts: list[str] = []

    if tools.dxil_spirv is None:
        return _failed(
            entry,
            digest,
            stage=STAGE_DXIL_TO_SPIRV,
            stderr="missing_tool:dxil-spirv",
            returncode=127,
            tool="dxil-spirv",
        )

    spv_path = cache_dir / SPV_FILE_NAME
    dxil_argv = [tools.dxil_spirv, "-o", str(spv_path), str(entry.path)]
    dxil_result = command_runner(dxil_argv)
    if dxil_result.returncode != 0 or not spv_path.is_file():
        return _failed(
            entry,
            digest,
            stage=STAGE_DXIL_TO_SPIRV,
            stderr=dxil_result.stderr or dxil_result.stdout,
            returncode=dxil_result.returncode,
            tool="dxil-spirv",
        )
    stages.append(STAGE_DXIL_TO_SPIRV)
    artifacts.append(SPV_FILE_NAME)

    if tools.spirv_val is not None:
        val_result = command_runner([tools.spirv_val, str(spv_path)])
        if val_result.returncode != 0:
            return _failed(
                entry,
                digest,
                stage=STAGE_SPIRV_VAL,
                stderr=val_result.stderr or val_result.stdout,
                returncode=val_result.returncode,
                tool="spirv-val",
                stages=stages,
                artifacts=artifacts,
            )
        stages.append(STAGE_SPIRV_VAL)

    if tools.playgta6_shaders is None:
        return _failed(
            entry,
            digest,
            stage=STAGE_SPIRV_TO_WGSL,
            stderr="missing_tool:playgta6-shaders",
            returncode=127,
            tool="playgta6-shaders",
            stages=stages,
            artifacts=artifacts,
        )

    spv_dir = cache_dir / "spirv"
    wgsl_dir = cache_dir / "wgsl"
    spv_dir.mkdir(parents=True, exist_ok=True)
    wgsl_dir.mkdir(parents=True, exist_ok=True)
    staged_spv = spv_dir / "shader.spv"
    if staged_spv.resolve() != spv_path.resolve():
        shutil.copyfile(spv_path, staged_spv)

    pack_path = cache_dir / PACK_FILE_NAME
    convert_argv = _playgta6_argv_convert(
        tools.playgta6_shaders, spv_dir, wgsl_dir, pack_path
    )
    convert_result = command_runner(convert_argv)
    if convert_result.returncode != 0:
        return _failed(
            entry,
            digest,
            stage=STAGE_SPIRV_TO_WGSL,
            stderr=convert_result.stderr or convert_result.stdout,
            returncode=convert_result.returncode,
            tool="playgta6-shaders",
            stages=stages,
            artifacts=artifacts,
        )

    reflect_argv = _playgta6_argv_reflect(tools.playgta6_shaders, spv_dir, wgsl_dir)
    reflect_result = command_runner(reflect_argv)
    if reflect_result.returncode != 0:
        return _failed(
            entry,
            digest,
            stage=STAGE_SPIRV_TO_WGSL,
            stderr=reflect_result.stderr or reflect_result.stdout,
            returncode=reflect_result.returncode,
            tool="playgta6-shaders",
            stages=stages,
            artifacts=artifacts,
        )
    stages.append(STAGE_SPIRV_TO_WGSL)

    # S6 writes layout-manifest.json; coverage reports the S9 alias name too.
    layout_src = wgsl_dir / LAYOUT_MANIFEST_NAME
    reflection_src = wgsl_dir / REFLECTION_MANIFEST_NAME
    if layout_src.is_file() and not reflection_src.is_file():
        shutil.copyfile(layout_src, wgsl_dir / REFLECTION_MANIFEST_NAME)
        shutil.copyfile(layout_src, cache_dir / REFLECTION_MANIFEST_NAME)
    elif reflection_src.is_file():
        shutil.copyfile(reflection_src, cache_dir / REFLECTION_MANIFEST_NAME)

    index_src = wgsl_dir / SHADER_INDEX_NAME
    if index_src.is_file():
        shutil.copyfile(index_src, cache_dir / SHADER_INDEX_NAME)
    if not pack_path.is_file():
        nested_pack = wgsl_dir / PACK_FILE_NAME
        if nested_pack.is_file():
            shutil.copyfile(nested_pack, pack_path)

    for wgsl_file in wgsl_dir.glob("*.wgsl"):
        dest = cache_dir / wgsl_file.name
        if dest.resolve() != wgsl_file.resolve():
            shutil.copyfile(wgsl_file, dest)

    artifacts = _collect_artifacts(cache_dir)
    result = ShaderResult(
        source=entry.label(),
        sha256=digest,
        status="converted",
        stages=stages,
        artifacts=artifacts,
        cached=False,
    )
    _write_status(cache_dir, result)
    return result


def convert_corpus(
    corpus_dir: Path | str,
    out_dir: Path | str,
    *,
    jobs: int = 1,
    dry_run: bool = False,
    check_only: bool = False,
    repo_root: Path | str = REPO_ROOT,
    which_fn: WhichFn = shutil.which,
    command_runner: CommandRunner = default_command_runner,
    write_reports: bool = True,
) -> CoverageReport:
    """Discover and convert a DXIL/DXBC corpus. `which_fn` / `command_runner` are injectable."""
    root = Path(corpus_dir)
    if not root.is_dir():
        raise ConvertError(
            f"corpus directory does not exist: {sanitize_source_label(root)}",
            code="missing_corpus",
        )

    destination = validate_output_dir(out_dir, repo_root=repo_root)
    tools = detect_tools(repo_root=repo_root, which_fn=which_fn)
    discovered = discover_corpus(root)
    corpus_label = sanitize_source_label(root)

    if check_only or dry_run:
        results = [
            ShaderResult(
                source=entry.label(),
                sha256=entry.sha256 or "",
                status="check_only" if check_only else "dry_run",
                stages=[],
                artifacts=[],
            )
            for entry in discovered
        ]
        report = CoverageReport(
            corpus_label=corpus_label,
            tools=tools,
            results=results,
            discovered=len(discovered),
            dry_run=dry_run,
            check_only=check_only,
        )
        return report

    workers = max(1, int(jobs))
    results: list[ShaderResult] = []
    if not discovered:
        report = CoverageReport(
            corpus_label=corpus_label,
            tools=tools,
            results=[],
            discovered=0,
        )
    elif workers == 1:
        for entry in discovered:
            results.append(
                convert_one(
                    entry,
                    out_dir=destination,
                    tools=tools,
                    command_runner=command_runner,
                )
            )
        report = CoverageReport(
            corpus_label=corpus_label,
            tools=tools,
            results=results,
            discovered=len(discovered),
        )
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    convert_one,
                    entry,
                    out_dir=destination,
                    tools=tools,
                    command_runner=command_runner,
                ): entry
                for entry in discovered
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: (item.source, item.sha256))
        report = CoverageReport(
            corpus_label=corpus_label,
            tools=tools,
            results=results,
            discovered=len(discovered),
        )

    if write_reports:
        destination.mkdir(parents=True, exist_ok=True)
        write_coverage_reports(destination, report)
    return report


def format_coverage_markdown(report: CoverageReport) -> str:
    tools = report.tools.to_dict()
    lines = [
        "# DXIL Corpus Conversion Coverage",
        "",
        f"- Corpus: `{report.corpus_label}`",
        f"- Discovered: {report.discovered}",
        f"- Converted: {report.converted}",
        f"- Cached: {report.cached_count}",
        f"- Failed: {report.failed}",
        "",
        "## Tools",
        "",
        f"- `dxil-spirv`: {'available' if tools['dxil-spirv']['available'] else 'missing'}",
        f"- `spirv-val`: {'available' if tools['spirv-val']['available'] else 'missing'}",
        f"- `playgta6-shaders`: {'available' if tools['playgta6-shaders']['available'] else 'missing'}",
        "",
        "## Failure buckets",
        "",
        "| Bucket | Count |",
        "| --- | --- |",
    ]
    for name, count in report.buckets().items():
        lines.append(f"| `{name}` | {count} |")
    lines.extend(["", "## Results", "", "| Source | Status | SHA-256 | Detail |", "| --- | --- | --- | --- |"])
    for item in report.results:
        detail = item.failure_code or ",".join(item.artifacts)
        lines.append(
            f"| `{item.source}` | `{item.status}` | `{item.sha256[:12] if item.sha256 else ''}` | {detail} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_coverage_reports(out_dir: Path, report: CoverageReport) -> None:
    json_text = json.dumps(sanitize_value(report.to_dict()), indent=2) + "\n"
    md_text = format_coverage_markdown(report)
    (out_dir / COVERAGE_JSON_NAME).write_text(json_text, encoding="utf-8", newline="\n")
    (out_dir / COVERAGE_MD_NAME).write_text(md_text, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert_dxil_corpus.py",
        description=(
            "Convert a local DXIL/DXBC corpus to SPIR-V + WGSL packs and emit a "
            "sanitized conversion coverage report."
        ),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        required=True,
        help="Directory of .dxbc/.dxil blobs (optional corpus-manifest.json from S10).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: local-data/shaders/converted).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="ThreadPoolExecutor worker count (default: CPU count).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover blobs and detect tools without running converters.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Same as --dry-run: validate discovery + tool detection only.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit a strictly sanitized JSON coverage report on stdout.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        report = convert_corpus(
            args.corpus_dir,
            args.out_dir,
            jobs=args.jobs,
            dry_run=bool(args.dry_run),
            check_only=bool(args.check_only),
            repo_root=args.repo_root,
            write_reports=not (args.dry_run or args.check_only),
        )
    except UnsafeOutputPathError as exc:
        if args.as_json:
            print(
                json.dumps(
                    {"ok": False, "errorCode": exc.code, "error": exc.message},
                    indent=2,
                )
            )
        else:
            print(f"Error: {exc.message}", file=sys.stderr)
        return 1
    except ConvertError as exc:
        if args.as_json:
            print(
                json.dumps(
                    {"ok": False, "errorCode": exc.code, "error": exc.message},
                    indent=2,
                )
            )
        else:
            print(f"Error: {exc.message}", file=sys.stderr)
        return 1
    except OSError as exc:
        message = sanitize_message(exc.strerror or type(exc).__name__)
        if args.as_json:
            print(
                json.dumps(
                    {"ok": False, "errorCode": "io_error", "error": message},
                    indent=2,
                )
            )
        else:
            print(f"Error: {message}", file=sys.stderr)
        return 1

    payload = sanitize_value(report.to_dict())
    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(format_coverage_markdown(report), end="")
    return 0


# Silence unused import warning for wgpk: pack convention documented via CLI comments.
_ = wgpk.PACK_MAGIC


if __name__ == "__main__":
    raise SystemExit(main())
