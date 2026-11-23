#!/usr/bin/env python3
"""Local DXIL/DXBC corpus extractor & RTS0/PSV0 analyzer CLI (Track 4 / S10).

Structurally parses DXBC / DXIL containers found under a local game directory
(or a single binary file), decodes DXIL / PSV0 / RTS0 / ISG1 / OSG1 / ACCC
summaries, and optionally writes deduplicated container blobs into a
gitignored output folder.

Security & privacy invariants:
  - Default mode is strictly read-only inspection (no writes).
  - `--out-dir` refuses tracked repository directories; only `work/`,
    `local-data/`, or paths outside the repo are allowed.
  - Never emits host absolute paths, drive letters, usernames, or raw shader
    bytes in stdout / JSON / the corpus manifest.
  - Pure Python 3.12 standard library only (zero external dependencies).

CLI usage:
    python tools/extract_dxil_corpus.py (--from-env | --path <dir-or-file>) \\
        [--out-dir <gitignored-dir>] [--inspect-only] \\
        [--max-bytes-per-file <N>] [--json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Constants matching shaders/src/dxbc.rs and tools/scan_shaders.py
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024
DEFAULT_OVERLAP_BYTES = 64 * 1024
MIN_CHUNK_SIZE = 64
DEFAULT_OUT_DIR = REPO_ROOT / "work" / "shaders" / "dxil"
ALLOWED_REPO_OUTPUT_DIR_NAMES: tuple[str, ...] = ("local-data", "work")
MANIFEST_FILE_NAME = "corpus-manifest.json"

DXBC_MAGIC = b"DXBC"
DXBC_HEADER_STRUCT = struct.Struct("<4s16sHHII")
DXBC_HEADER_BYTES = DXBC_HEADER_STRUCT.size  # 32
DXBC_CHUNK_HEADER_STRUCT = struct.Struct("<4sI")
DXBC_CHUNK_HEADER_BYTES = DXBC_CHUNK_HEADER_STRUCT.size  # 8
DXBC_MIN_CHUNKS = 1
DXBC_MAX_CHUNKS = 64
DXBC_MAX_CONTAINER_BYTES = 256 * 1024 * 1024
DXIL_PROGRAM_HEADER_BYTES = 24
LLVM_BITCODE_MAGIC = b"\x42\x43\xc0\xde"
RTS0_HEADER_BYTES = 24
PSV0_MIN_RUNTIME_INFO = 24
SIGNATURE_ELEMENT_MAX = 256
SIGNATURE_NAME_MAX = 256
MAX_SEMANTIC_NAMES = 64

# Shader kind: full name -> profile prefix used in output filenames.
DXIL_SHADER_KIND_NAMES: dict[int, str] = {
    0: "pixel",
    1: "vertex",
    2: "geometry",
    3: "hull",
    4: "domain",
    5: "compute",
    6: "library",
    7: "ray_generation",
    8: "intersection",
    9: "any_hit",
    10: "closest_hit",
    11: "miss",
    12: "callable",
    13: "mesh",
    14: "amplification",
    15: "node",
}

DXIL_SHADER_KIND_PREFIX: dict[int, str] = {
    0: "ps",
    1: "vs",
    2: "gs",
    3: "hs",
    4: "ds",
    5: "cs",
    6: "lib",
    7: "rgs",
    8: "is",
    9: "ahs",
    10: "chs",
    11: "ms_rt",
    12: "call",
    13: "ms",
    14: "as",
    15: "node",
}

KNOWN_DXBC_FOURCCS: frozenset[str] = frozenset(
    {
        "DXIL",
        "ILDB",
        "SHEX",
        "SHDR",
        "ISGN",
        "OSGN",
        "ISG1",
        "OSG1",
        "PSG1",
        "PCSG",
        "RTS0",
        "SFI0",
        "STAT",
        "PSV0",
        "RDEF",
        "HASH",
        "Aon9",
        "SPDB",
        "ILDN",
        "PRIV",
        "ICEC",
        "IFCE",
        "OSG5",
        "RDAT",
        "VERS",
        "ACCC",  # Xbox Scarlett precompiled RDNA2 ISA
    }
)

DXIL_FOURCCS: frozenset[str] = frozenset({"DXIL", "ILDB", "PSV0"})
DXBC_SM5_FOURCCS: frozenset[str] = frozenset({"SHEX", "SHDR"})

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_SIZE_SUFFIX_RE = re.compile(r"^\s*(\d+)\s*([kKmMgG](?:i?[bB])?|[bB])?\s*$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DxbcParseError(ValueError):
    """Raised when a DXBC/DXIL container fails structural validation."""


class UnsafeOutputPathError(ValueError):
    """Raised when `--out-dir` would write into a tracked repository path."""


# ---------------------------------------------------------------------------
# .env resolution & sanitization (matching tools/scan_shaders.py)
# ---------------------------------------------------------------------------


def load_dotenv(env_path: Path = ENV_FILE) -> dict[str, str]:
    """Parse a simple KEY=VALUE `.env` file matching `tools/local_env.py`."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def resolve_game_dir(
    env_path: Path = ENV_FILE,
    *,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve `GTA6_GAME_DIR` from environment variables or `.env`."""
    env_map = os.environ if env is None else env
    dotenv_vals = load_dotenv(env_path)
    raw = (env_map.get("GTA6_GAME_DIR") or dotenv_vals.get("GTA6_GAME_DIR", "")).strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = env_path.resolve().parent / candidate
    return candidate


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


def _scrub_sensitive_text(text: str) -> str:
    """Remove host absolute paths and current OS usernames from arbitrary text."""
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


def sanitize_source_label(source: str | Path | None) -> str:
    """Return a strictly sanitized basename-only label (no host paths or usernames)."""
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
    # Normalize BOTH Windows `\` and POSIX `/` separators before taking basename.
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return _scrub_sensitive_text(base)


def sanitize_relative_path(path: Path, base_dir: Path | None) -> str:
    """Return a strictly sanitized POSIX relative path under `base_dir` or basename."""
    if base_dir is not None:
        try:
            rel = path.resolve().relative_to(base_dir.resolve())
            parts = [p for p in rel.parts if p not in ("", ".", "..")]
            if parts and not (len(parts[0]) == 2 and parts[0][1] == ":"):
                # Always emit POSIX `/` separators regardless of host OS.
                return _scrub_sensitive_text("/".join(parts))
        except (ValueError, OSError):
            pass
    return sanitize_source_label(path)


def sanitize_error_message(source: str | Path, exc: BaseException) -> str:
    """Format an error message without ever leaking host absolute paths or usernames."""
    label = sanitize_source_label(source)
    if isinstance(exc, OSError):
        reason = exc.strerror or type(exc).__name__
    else:
        reason = _scrub_sensitive_text(str(exc)) or type(exc).__name__
    return f"failed to read {label}: {reason}"


def parse_byte_size(raw: str | int) -> int:
    """Parse a positive integer byte count with optional K/M/G/KiB/MiB/GiB suffix."""
    if isinstance(raw, bool):
        raise ValueError("byte size must be an integer, not bool")
    if isinstance(raw, int):
        if raw <= 0:
            raise ValueError("byte size must be > 0")
        return raw
    match = _SIZE_SUFFIX_RE.fullmatch(str(raw))
    if not match:
        raise ValueError(f"invalid byte size: {raw!r}")
    value = int(match.group(1))
    suffix = (match.group(2) or "").lower()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024 * 1024,
        "mb": 1024 * 1024,
        "mib": 1024 * 1024,
        "g": 1024 * 1024 * 1024,
        "gb": 1024 * 1024 * 1024,
        "gib": 1024 * 1024 * 1024,
    }
    result = value * multipliers[suffix]
    if result <= 0:
        raise ValueError("byte size must be > 0")
    return result


# ---------------------------------------------------------------------------
# Output-dir safety (matching tools/gen_fallback_shaders.py)
# ---------------------------------------------------------------------------


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def is_safe_output_dir(path: Path | str, *, repo_root: Path | str = REPO_ROOT) -> bool:
    """Return True iff `path` may receive extracted DXBC corpus blobs.

    Allowed destinations:
      - `<repo>/local-data/**` or `<repo>/work/**` (gitignored)
      - any path outside the repository (temp dirs used by tests)
    """
    target = Path(path).expanduser().resolve()
    root = Path(repo_root).resolve()

    if not _is_under(target, root):
        return True

    for name in ALLOWED_REPO_OUTPUT_DIR_NAMES:
        if _is_under(target, root / name):
            return True
    return False


def validate_output_dir(path: Path | str, *, repo_root: Path | str = REPO_ROOT) -> Path:
    """Resolve and validate an output directory, raising on tracked source paths."""
    target = Path(path).expanduser().resolve()
    if not is_safe_output_dir(target, repo_root=repo_root):
        raise UnsafeOutputPathError(
            "Refusing to write extracted shader blobs into a tracked repository "
            "source directory. Point --out-dir at gitignored local-data/, work/, "
            "or a temp directory outside the repo."
        )
    return target


# ---------------------------------------------------------------------------
# Result data models (strictly sanitized metadata; zero raw binary payloads)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DxilProgramSummary:
    shader_kind: str
    shader_kind_raw: int
    profile_prefix: str
    sm_major: int
    sm_minor: int
    size_in_uint32: int
    dxil_version_major: int
    dxil_version_minor: int
    bitcode_offset: int
    bitcode_size: int
    has_llvm_bitcode_magic: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "shader_kind": self.shader_kind,
            "shader_kind_raw": self.shader_kind_raw,
            "profile_prefix": self.profile_prefix,
            "sm_major": self.sm_major,
            "sm_minor": self.sm_minor,
            "shader_model": f"{self.sm_major}.{self.sm_minor}",
            "size_in_uint32": self.size_in_uint32,
            "dxil_version": f"{self.dxil_version_major}.{self.dxil_version_minor}",
            "bitcode_offset": self.bitcode_offset,
            "bitcode_size": self.bitcode_size,
            "has_llvm_bitcode_magic": self.has_llvm_bitcode_magic,
        }


@dataclass(frozen=True)
class Psv0Summary:
    runtime_info_size: int
    stage: str | None
    stage_raw: int | None
    resource_count: int
    input_element_count: int | None
    output_element_count: int | None
    patch_const_or_prim_element_count: int | None = None
    uses_view_id: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "runtime_info_size": self.runtime_info_size,
            "stage": self.stage,
            "stage_raw": self.stage_raw,
            "resource_count": self.resource_count,
            "input_element_count": self.input_element_count,
            "output_element_count": self.output_element_count,
        }
        if self.patch_const_or_prim_element_count is not None:
            payload["patch_const_or_prim_element_count"] = self.patch_const_or_prim_element_count
        if self.uses_view_id is not None:
            payload["uses_view_id"] = self.uses_view_id
        return payload


@dataclass(frozen=True)
class Rts0Summary:
    version: int
    version_label: str
    root_parameter_count: int
    static_sampler_count: int
    flags: int
    parameters_offset: int
    static_sampler_offset: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "version_label": self.version_label,
            "root_parameter_count": self.root_parameter_count,
            "static_sampler_count": self.static_sampler_count,
            "flags": self.flags,
            "parameters_offset": self.parameters_offset,
            "static_sampler_offset": self.static_sampler_offset,
        }


@dataclass(frozen=True)
class SignatureElementSummary:
    semantic_name: str
    semantic_index: int
    register: int
    stream: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "semantic_name": self.semantic_name,
            "semantic_index": self.semantic_index,
            "register": self.register,
        }
        if self.stream is not None:
            payload["stream"] = self.stream
        return payload


@dataclass(frozen=True)
class SignatureSummary:
    fourcc: str
    element_count: int
    elements: tuple[SignatureElementSummary, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "fourcc": self.fourcc,
            "element_count": self.element_count,
            "elements": [el.to_dict() for el in self.elements],
        }


@dataclass(frozen=True)
class DxbcChunkInfo:
    fourcc: str
    offset: int
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"fourcc": self.fourcc, "offset": self.offset, "size": self.size}


@dataclass(frozen=True)
class DxbcContainerInfo:
    """Parsed DXBC / DXIL container metadata (never retains raw bytecode dumps)."""

    digest_hex: str
    version_major: int
    version_minor: int
    total_size: int
    actual_bytes: int
    chunks: tuple[DxbcChunkInfo, ...]
    format: str
    dxil: DxilProgramSummary | None = None
    psv0: Psv0Summary | None = None
    rts0: Rts0Summary | None = None
    signatures: tuple[SignatureSummary, ...] = ()
    has_accc: bool = False
    has_stat: bool = False
    has_hash: bool = False
    sha256_hex: str = ""

    @property
    def fourccs(self) -> tuple[str, ...]:
        return tuple(c.fourcc for c in self.chunks)

    @property
    def shader_kind(self) -> str | None:
        if self.dxil is not None:
            return self.dxil.shader_kind
        return None

    @property
    def profile_prefix(self) -> str:
        if self.dxil is not None:
            return self.dxil.profile_prefix
        return "unk"

    @property
    def sm_major(self) -> int | None:
        if self.dxil is not None:
            return self.dxil.sm_major
        return None

    @property
    def sm_minor(self) -> int | None:
        if self.dxil is not None:
            return self.dxil.sm_minor
        return None

    def corpus_filename(self) -> str:
        """Deterministic deduplicated blob name: `<sha256[:16]>_<kind>_sm<maj>_<min>.dxbc`."""
        digest16 = (self.sha256_hex or "0" * 16)[:16]
        kind = self.profile_prefix or "unk"
        maj = self.sm_major if self.sm_major is not None else 0
        minor = self.sm_minor if self.sm_minor is not None else 0
        return f"{digest16}_{kind}_sm{maj}_{minor}.dxbc"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "digest": self.digest_hex,
            "sha256": self.sha256_hex,
            "version": f"{self.version_major}.{self.version_minor}",
            "total_size": self.total_size,
            "actual_bytes": self.actual_bytes,
            "format": self.format,
            "chunks": [c.to_dict() for c in self.chunks],
            "fourccs": list(self.fourccs),
            "has_accc": self.has_accc,
            "has_stat": self.has_stat,
            "has_hash": self.has_hash,
            "corpus_filename": self.corpus_filename(),
        }
        if self.dxil is not None:
            payload["dxil"] = self.dxil.to_dict()
        if self.psv0 is not None:
            payload["psv0"] = self.psv0.to_dict()
        if self.rts0 is not None:
            payload["rts0"] = self.rts0.to_dict()
        if self.signatures:
            payload["signatures"] = [s.to_dict() for s in self.signatures]
        return payload


@dataclass(frozen=True)
class ContainerHit:
    """One validated DXBC/DXIL container located inside a scanned file."""

    offset: int
    info: DxbcContainerInfo
    source_path: str

    def to_dict(self) -> dict[str, Any]:
        payload = self.info.to_dict()
        payload["offset"] = self.offset
        payload["source_path"] = self.source_path
        return payload


@dataclass(frozen=True)
class FileExtractResult:
    path: str
    file_size: int
    bytes_scanned: int
    truncated: bool
    containers: tuple[ContainerHit, ...] = ()
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "file_size": self.file_size,
            "bytes_scanned": self.bytes_scanned,
            "truncated": self.truncated,
            "container_count": len(self.containers),
            "containers": [c.to_dict() for c in self.containers],
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass
class CorpusReport:
    schema_version: int
    target_label: str
    target_kind: str
    max_bytes_per_file: int | None
    inspect_only: bool
    out_dir_label: str | None
    files_scanned: int = 0
    files_with_containers: int = 0
    files_errored: int = 0
    total_file_bytes: int = 0
    total_bytes_scanned: int = 0
    containers_found: int = 0
    containers_unique: int = 0
    containers_written: int = 0
    containers_deduplicated: int = 0
    format_counts: dict[str, int] = field(default_factory=dict)
    kind_counts: dict[str, int] = field(default_factory=dict)
    fourcc_counts: dict[str, int] = field(default_factory=dict)
    files: list[FileExtractResult] = field(default_factory=list)
    written_blobs: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target": self.target_label,
            "target_kind": self.target_kind,
            "max_bytes_per_file": self.max_bytes_per_file,
            "inspect_only": self.inspect_only,
            "out_dir": self.out_dir_label,
            "summary": {
                "files_scanned": self.files_scanned,
                "files_with_containers": self.files_with_containers,
                "files_errored": self.files_errored,
                "total_file_bytes": self.total_file_bytes,
                "total_bytes_scanned": self.total_bytes_scanned,
                "containers_found": self.containers_found,
                "containers_unique": self.containers_unique,
                "containers_written": self.containers_written,
                "containers_deduplicated": self.containers_deduplicated,
                "format_counts": dict(self.format_counts),
                "kind_counts": dict(self.kind_counts),
                "fourcc_counts": dict(self.fourcc_counts),
            },
            "files": [f.to_dict() for f in self.files],
            "written_blobs": list(self.written_blobs),
        }


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _is_printable_fourcc(raw: bytes) -> bool:
    return len(raw) == 4 and all(0x20 <= b <= 0x7E for b in raw)


def _read_c_string(payload: bytes, offset: int, max_len: int = SIGNATURE_NAME_MAX) -> str | None:
    if offset <= 0 or offset >= len(payload):
        return None
    end = min(len(payload), offset + max_len)
    chunk = payload[offset:end]
    nul = chunk.find(b"\x00")
    if nul <= 0:
        return None
    try:
        text = chunk[:nul].decode("ascii")
    except UnicodeDecodeError:
        return None
    if not text or any(ord(ch) < 0x20 for ch in text):
        return None
    return text


def _shader_kind_name(raw: int) -> str:
    return DXIL_SHADER_KIND_NAMES.get(raw, f"unknown({raw})")


def _shader_kind_prefix(raw: int) -> str:
    return DXIL_SHADER_KIND_PREFIX.get(raw, "unk")


# ---------------------------------------------------------------------------
# Chunk decoders
# ---------------------------------------------------------------------------


def decode_dxil_program_header(payload: bytes | memoryview) -> DxilProgramSummary | None:
    """Decode a `DXIL` chunk `DxilProgramHeader` (+ LLVM bitcode magic check)."""
    if len(payload) < DXIL_PROGRAM_HEADER_BYTES:
        return None
    program_version, size_u32, dxil_magic, dxil_ver, bitcode_offset, bitcode_size = struct.unpack_from(
        "<6I", payload, 0
    )
    # dxil_magic is stored as u32 little-endian "DXIL"
    magic_bytes = struct.pack("<I", dxil_magic)
    if magic_bytes != b"DXIL":
        # Still accept containers whose magic field is non-standard but structurally sound.
        pass

    kind_raw = (program_version >> 16) & 0xFFFF
    sm_major = (program_version >> 4) & 0xF
    sm_minor = program_version & 0xF
    dxil_major = (dxil_ver >> 8) & 0xFF
    dxil_minor = dxil_ver & 0xFF

    # BitcodeOffset is relative to byte 8 (DxilBitcodeHeader), matching shaders/src/dxbc.rs.
    bc_start = 8 + bitcode_offset
    has_llvm = (
        bitcode_size >= 4
        and bc_start + 4 <= len(payload)
        and bc_start + bitcode_size <= len(payload)
        and bytes(payload[bc_start : bc_start + 4]) == LLVM_BITCODE_MAGIC
    )
    return DxilProgramSummary(
        shader_kind=_shader_kind_name(kind_raw),
        shader_kind_raw=kind_raw,
        profile_prefix=_shader_kind_prefix(kind_raw),
        sm_major=sm_major,
        sm_minor=sm_minor,
        size_in_uint32=size_u32,
        dxil_version_major=dxil_major,
        dxil_version_minor=dxil_minor,
        bitcode_offset=bitcode_offset,
        bitcode_size=bitcode_size,
        has_llvm_bitcode_magic=has_llvm,
    )


def decode_psv0(payload: bytes | memoryview) -> Psv0Summary | None:
    """Decode a `PSV0` Pipeline State Validation summary."""
    data = bytes(payload)
    if len(data) < 4:
        return None
    runtime_info_size = struct.unpack_from("<I", data, 0)[0]
    if runtime_info_size < PSV0_MIN_RUNTIME_INFO:
        return None
    if 4 + runtime_info_size > len(data):
        return None

    stage: str | None = None
    stage_raw: int | None = None
    input_count: int | None = None
    output_count: int | None = None
    patch_count: int | None = None
    uses_view_id: bool | None = None

    if runtime_info_size >= 36 and 4 + 36 <= len(data):
        # PSVRuntimeInfo1 fields start at offset 24 within the RuntimeInfo struct
        # (absolute offset 4+24 = 28 in the chunk payload).
        stage_raw = data[4 + 24]
        uses_view_id = data[4 + 25] != 0
        input_count = data[4 + 28]
        output_count = data[4 + 29]
        patch_count = data[4 + 30]
        stage = _shader_kind_name(stage_raw) if stage_raw in DXIL_SHADER_KIND_NAMES else f"unknown({stage_raw})"

    # Resource count immediately follows RuntimeInfo.
    resource_count = 0
    res_pos = 4 + runtime_info_size
    if res_pos + 4 <= len(data):
        resource_count = struct.unpack_from("<I", data, res_pos)[0]
        # Sanity clamp: refuse absurd counts that would imply OOB.
        if resource_count > 1_000_000:
            resource_count = 0

    return Psv0Summary(
        runtime_info_size=runtime_info_size,
        stage=stage,
        stage_raw=stage_raw,
        resource_count=resource_count,
        input_element_count=input_count,
        output_element_count=output_count,
        patch_const_or_prim_element_count=patch_count,
        uses_view_id=uses_view_id,
    )


def decode_rts0(payload: bytes | memoryview) -> Rts0Summary | None:
    """Decode a `RTS0` root signature v1.0 / v1.1 header summary."""
    data = bytes(payload)
    if len(data) < RTS0_HEADER_BYTES:
        return None
    version, num_params, params_off, num_samplers, sampler_off, flags = struct.unpack_from(
        "<6I", data, 0
    )
    if version == 1:
        version_label = "1.0"
    elif version == 2:
        version_label = "1.1"
    elif version == 3:
        version_label = "1.2"
    else:
        version_label = f"unknown({version})"

    # Bound-check offsets when non-zero / counts non-zero.
    if num_params > 0 and (params_off < RTS0_HEADER_BYTES or params_off >= len(data)):
        return None
    if num_samplers > 0 and (sampler_off < RTS0_HEADER_BYTES or sampler_off >= len(data)):
        return None
    if num_params > 100_000 or num_samplers > 100_000:
        return None

    return Rts0Summary(
        version=version,
        version_label=version_label,
        root_parameter_count=num_params,
        static_sampler_count=num_samplers,
        flags=flags,
        parameters_offset=params_off,
        static_sampler_offset=sampler_off,
    )


def decode_signature_chunk(fourcc: str, payload: bytes | memoryview) -> SignatureSummary | None:
    """Decode `ISG1`/`OSG1` (and legacy `ISGN`/`OSGN`) element count + semantic names."""
    data = bytes(payload)
    if len(data) < 8:
        return None
    element_count, first_element_offset = struct.unpack_from("<II", data, 0)
    if element_count > SIGNATURE_ELEMENT_MAX:
        return None
    if first_element_offset < 8 or first_element_offset > len(data):
        return None

    elements: list[SignatureElementSummary] = []
    if element_count == 0:
        return SignatureSummary(fourcc=fourcc, element_count=0, elements=())

    # Infer element stride from remaining payload (24 = legacy ISGN, 32 = ISG1/OSG1).
    remaining = len(data) - first_element_offset
    if remaining < element_count * 24:
        # Still report element_count even if we cannot safely enumerate names.
        return SignatureSummary(fourcc=fourcc, element_count=element_count, elements=())

    if remaining >= element_count * 32:
        stride = 32
        is_dxil_style = True
    else:
        stride = 24
        is_dxil_style = False

    limit = min(element_count, MAX_SEMANTIC_NAMES)
    for idx in range(limit):
        base = first_element_offset + idx * stride
        if base + stride > len(data):
            break
        if is_dxil_style:
            stream, name_off, sem_index, _sys, _comp, register = struct.unpack_from(
                "<6I", data, base
            )
        else:
            stream = None
            name_off, sem_index, _sys, _comp, register = struct.unpack_from("<5I", data, base)

        name = _read_c_string(data, name_off) or f"<unnamed:{idx}>"
        elements.append(
            SignatureElementSummary(
                semantic_name=name,
                semantic_index=sem_index,
                register=register,
                stream=stream,
            )
        )

    return SignatureSummary(
        fourcc=fourcc,
        element_count=element_count,
        elements=tuple(elements),
    )


# ---------------------------------------------------------------------------
# Structural DXBC / DXIL container parser
# ---------------------------------------------------------------------------


def parse_dxbc_container(data: bytes) -> DxbcContainerInfo:
    """Validate and summarize a DXBC / DXIL container.

    Validates `DXBC` magic, declared total size, chunk count, and the chunk
    offset table. Extracts FourCC chunk list and decodes recognized chunks
    (`DXIL`, `PSV0`, `RTS0`, `ISG1`, `OSG1`, `ACCC`, …).
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes-like, got {type(data).__name__}")
    buf = bytes(data)

    if len(buf) < DXBC_HEADER_BYTES:
        raise DxbcParseError(
            f"DXBC container truncated: {len(buf)} bytes (minimum required: {DXBC_HEADER_BYTES})"
        )

    magic, digest, ver_major, ver_minor, total_size, chunk_count = DXBC_HEADER_STRUCT.unpack_from(
        buf, 0
    )
    if magic != DXBC_MAGIC:
        raise DxbcParseError(f"invalid DXBC magic {magic!r}")

    if total_size < DXBC_HEADER_BYTES or total_size > len(buf):
        raise DxbcParseError(
            f"invalid DXBC total size {total_size} bytes (input buffer has {len(buf)} bytes)"
        )
    if total_size > DXBC_MAX_CONTAINER_BYTES:
        raise DxbcParseError(f"DXBC total size {total_size} exceeds sanity bound")

    if chunk_count < DXBC_MIN_CHUNKS or chunk_count > DXBC_MAX_CHUNKS:
        raise DxbcParseError(f"invalid DXBC chunk count {chunk_count}")

    table_end = DXBC_HEADER_BYTES + chunk_count * 4
    if table_end > total_size:
        raise DxbcParseError(
            f"DXBC chunk offset table ({chunk_count} chunks, end={table_end}) "
            f"exceeds container size {total_size}"
        )

    offsets = struct.unpack_from(f"<{chunk_count}I", buf, DXBC_HEADER_BYTES)
    container = buf[:total_size]

    chunks: list[DxbcChunkInfo] = []
    dxil: DxilProgramSummary | None = None
    psv0: Psv0Summary | None = None
    rts0: Rts0Summary | None = None
    signatures: list[SignatureSummary] = []
    has_accc = False
    has_stat = False
    has_hash = False
    known_tag_count = 0

    seen_offsets: set[int] = set()
    for idx, off in enumerate(offsets):
        if off in seen_offsets:
            raise DxbcParseError(f"duplicate DXBC chunk offset {off}")
        seen_offsets.add(off)

        payload_start = off + DXBC_CHUNK_HEADER_BYTES
        if off < table_end or payload_start > total_size:
            raise DxbcParseError(
                f"DXBC chunk {idx} at offset {off} exceeds container bounds ({total_size} bytes)"
            )

        fourcc_bytes, chunk_size = DXBC_CHUNK_HEADER_STRUCT.unpack_from(container, off)
        if not _is_printable_fourcc(fourcc_bytes):
            raise DxbcParseError(f"DXBC chunk {idx} has non-printable FourCC {fourcc_bytes!r}")

        payload_end = payload_start + chunk_size
        if payload_end > total_size:
            raise DxbcParseError(
                f"DXBC chunk {idx} at offset {off} (+header+{chunk_size} bytes) "
                f"exceeds container bounds ({total_size} bytes)"
            )

        tag = fourcc_bytes.decode("ascii")
        chunks.append(DxbcChunkInfo(fourcc=tag, offset=off, size=chunk_size))
        if tag in KNOWN_DXBC_FOURCCS:
            known_tag_count += 1

        payload = container[payload_start:payload_end]
        if tag == "DXIL" and dxil is None:
            dxil = decode_dxil_program_header(payload)
        elif tag == "PSV0" and psv0 is None:
            psv0 = decode_psv0(payload)
        elif tag == "RTS0" and rts0 is None:
            rts0 = decode_rts0(payload)
        elif tag in ("ISG1", "OSG1", "ISGN", "OSGN", "PSG1", "PCSG"):
            sig = decode_signature_chunk(tag, payload)
            if sig is not None:
                signatures.append(sig)
        elif tag == "ACCC":
            has_accc = True
        elif tag == "STAT":
            has_stat = True
        elif tag == "HASH":
            has_hash = True

    if known_tag_count == 0:
        raise DxbcParseError("DXBC container has no recognized FourCC tags")

    tag_set = {c.fourcc for c in chunks}
    if tag_set & DXIL_FOURCCS:
        container_format = "DXIL_SM6"
    elif tag_set & DXBC_SM5_FOURCCS:
        container_format = "DXBC_SM5"
    else:
        container_format = "DXBC"

    sha256_hex = hashlib.sha256(container).hexdigest()
    digest_hex = digest.hex()

    return DxbcContainerInfo(
        digest_hex=digest_hex,
        version_major=ver_major,
        version_minor=ver_minor,
        total_size=total_size,
        actual_bytes=len(buf),
        chunks=tuple(chunks),
        format=container_format,
        dxil=dxil,
        psv0=psv0,
        rts0=rts0,
        signatures=tuple(signatures),
        has_accc=has_accc,
        has_stat=has_stat,
        has_hash=has_hash,
        sha256_hex=sha256_hex,
    )


def try_parse_dxbc_at(
    buf: bytes,
    rel_offset: int,
    *,
    file_size: int | None = None,
) -> tuple[DxbcContainerInfo, bytes] | None:
    """Validate a candidate DXBC container at `rel_offset` and return (info, blob)."""
    buf_len = len(buf)
    if rel_offset + DXBC_HEADER_BYTES > buf_len:
        return None

    magic, _digest, ver_major, ver_minor, total_size, chunk_count = DXBC_HEADER_STRUCT.unpack_from(
        buf, rel_offset
    )
    if magic != DXBC_MAGIC:
        return None
    if ver_major < 1 or ver_major > 2 or ver_minor > 16:
        return None
    if chunk_count < DXBC_MIN_CHUNKS or chunk_count > DXBC_MAX_CHUNKS:
        return None

    table_end = DXBC_HEADER_BYTES + chunk_count * 4
    min_valid_size = table_end + DXBC_CHUNK_HEADER_BYTES
    if total_size < min_valid_size or total_size > DXBC_MAX_CONTAINER_BYTES:
        return None
    if file_size is not None and rel_offset + total_size > file_size and file_size == buf_len:
        # When scanning an in-memory slice whose length == file_size, enforce bound.
        pass
    if rel_offset + total_size > buf_len:
        return None

    blob = buf[rel_offset : rel_offset + total_size]
    try:
        info = parse_dxbc_container(blob)
    except DxbcParseError:
        return None
    return info, blob


# ---------------------------------------------------------------------------
# Streaming scanner
# ---------------------------------------------------------------------------


def _validate_chunk_params(chunk_size: int, overlap: int) -> tuple[int, int]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be an integer >= {MIN_CHUNK_SIZE}, got {chunk_size!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError(f"overlap must be a non-negative integer, got {overlap!r}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )
    return chunk_size, overlap


def _scan_window_for_dxbc(
    window: bytes,
    *,
    window_file_offset: int,
    scan_limit: int,
    file_size: int | None,
    source_path: str,
    seen_offsets: set[int],
    hits: list[tuple[ContainerHit, bytes]],
) -> None:
    if scan_limit <= 0 or not window:
        return

    pos = window.find(DXBC_MAGIC, 0, scan_limit + len(DXBC_MAGIC) - 1)
    while pos != -1 and pos < scan_limit:
        abs_offset = window_file_offset + pos
        if abs_offset not in seen_offsets:
            parsed = try_parse_dxbc_at(window, pos, file_size=file_size)
            if parsed is not None:
                info, blob = parsed
                # When window is a sliding overlap view, ensure total_size fits.
                if pos + info.total_size <= len(window):
                    seen_offsets.add(abs_offset)
                    hits.append(
                        (
                            ContainerHit(offset=abs_offset, info=info, source_path=source_path),
                            blob,
                        )
                    )
        pos = window.find(DXBC_MAGIC, pos + 1, scan_limit + len(DXBC_MAGIC) - 1)


def scan_bytes_for_dxbc(
    data: bytes | bytearray | memoryview,
    *,
    source_path: str = "<memory>",
    max_bytes: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
) -> list[tuple[ContainerHit, bytes]]:
    """Scan an in-memory buffer for validated DXBC/DXIL containers."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes-like buffer, got {type(data).__name__}")
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0
    ):
        raise ValueError(f"max_bytes must be a positive integer or None, got {max_bytes!r}")

    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    view = memoryview(data)
    total_len = len(view)
    limit = total_len if max_bytes is None else min(total_len, max_bytes)

    hits: list[tuple[ContainerHit, bytes]] = []
    seen: set[int] = set()
    if limit == 0:
        return hits

    # For small buffers, parse in one shot to avoid overlap edge cases.
    if limit <= chunk_size + overlap:
        window = bytes(view[:limit])
        _scan_window_for_dxbc(
            window,
            window_file_offset=0,
            scan_limit=len(window),
            file_size=total_len,
            source_path=source_path,
            seen_offsets=seen,
            hits=hits,
        )
        return hits

    tail = b""
    window_file_offset = 0
    cursor = 0
    while cursor < limit:
        end = min(cursor + chunk_size, limit)
        chunk = bytes(view[cursor:end])
        cursor = end
        is_final = cursor >= limit
        window = tail + chunk if tail else chunk
        scan_limit = len(window) if is_final else max(0, len(window) - overlap)
        _scan_window_for_dxbc(
            window,
            window_file_offset=window_file_offset,
            scan_limit=scan_limit,
            file_size=total_len,
            source_path=source_path,
            seen_offsets=seen,
            hits=hits,
        )
        if is_final:
            break
        if overlap > 0:
            tail = window[scan_limit:]
            window_file_offset += scan_limit
        else:
            tail = b""
            window_file_offset += len(window)
    return hits


def scan_file_for_dxbc(
    path: str | Path,
    *,
    base_dir: Path | None = None,
    max_bytes: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
) -> tuple[FileExtractResult, list[tuple[ContainerHit, bytes]]]:
    """Stream-scan a single file for DXBC/DXIL containers."""
    file_path = Path(path)
    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    rel_label = sanitize_relative_path(file_path, base_dir)

    try:
        st = file_path.stat()
        file_size = st.st_size
    except OSError as exc:
        return (
            FileExtractResult(
                path=rel_label,
                file_size=0,
                bytes_scanned=0,
                truncated=False,
                error=sanitize_error_message(file_path, exc),
            ),
            [],
        )

    hits: list[tuple[ContainerHit, bytes]] = []
    seen: set[int] = set()
    bytes_scanned = 0
    tail = b""
    window_file_offset = 0

    try:
        with file_path.open("rb") as fp:
            while True:
                if max_bytes is not None:
                    remaining = max_bytes - bytes_scanned
                    if remaining <= 0:
                        break
                    to_read = min(chunk_size, remaining)
                else:
                    to_read = chunk_size

                chunk = fp.read(to_read)
                bytes_scanned += len(chunk)
                reached_limit = max_bytes is not None and bytes_scanned >= max_bytes
                eof = len(chunk) < to_read
                is_final = eof or reached_limit

                window = tail + chunk if tail else chunk
                if not window:
                    break

                scan_limit = len(window) if is_final else max(0, len(window) - overlap)
                _scan_window_for_dxbc(
                    window,
                    window_file_offset=window_file_offset,
                    scan_limit=scan_limit,
                    file_size=file_size,
                    source_path=rel_label,
                    seen_offsets=seen,
                    hits=hits,
                )

                if is_final:
                    break
                if overlap > 0:
                    tail = window[scan_limit:]
                    window_file_offset += scan_limit
                else:
                    tail = b""
                    window_file_offset += len(window)
    except OSError as exc:
        return (
            FileExtractResult(
                path=rel_label,
                file_size=file_size,
                bytes_scanned=bytes_scanned,
                truncated=file_size > bytes_scanned,
                containers=tuple(h for h, _ in hits),
                error=sanitize_error_message(file_path, exc),
            ),
            hits,
        )

    result = FileExtractResult(
        path=rel_label,
        file_size=file_size,
        bytes_scanned=bytes_scanned,
        truncated=file_size > bytes_scanned,
        containers=tuple(h for h, _ in hits),
    )
    return result, hits


# ---------------------------------------------------------------------------
# Extraction / corpus building
# ---------------------------------------------------------------------------


def _aggregate_counts(
    hits: Sequence[tuple[ContainerHit, bytes]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    format_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    fourcc_counts: Counter[str] = Counter()
    for hit, _blob in hits:
        format_counts[hit.info.format] += 1
        kind = hit.info.shader_kind or "unknown"
        kind_counts[kind] += 1
        for tag in hit.info.fourccs:
            fourcc_counts[tag] += 1
    return (
        dict(sorted(format_counts.items())),
        dict(sorted(kind_counts.items())),
        dict(sorted(fourcc_counts.items())),
    )


def extract_corpus(
    target: str | Path,
    *,
    out_dir: Path | str | None = None,
    inspect_only: bool = False,
    max_bytes_per_file: int | None = None,
    target_label: str | None = None,
    repo_root: Path | str = REPO_ROOT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
) -> CorpusReport:
    """Scan `target` for DXBC/DXIL containers; optionally write a deduplicated corpus."""
    target_path = Path(target)
    write_enabled = out_dir is not None and not inspect_only
    destination: Path | None = None
    out_dir_label: str | None = None

    if write_enabled:
        assert out_dir is not None
        destination = validate_output_dir(out_dir, repo_root=repo_root)
        # Label is basename-only (never an absolute host path).
        out_dir_label = sanitize_source_label(destination)

    if target_path.is_dir():
        target_kind = "directory"
        label = sanitize_source_label(target_label if target_label is not None else target_path)
        base_dir = target_path
        discovered: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(target_path):
            dirnames.sort()
            filenames.sort()
            base = Path(dirpath)
            for fname in filenames:
                discovered.append(base / fname)
        discovered.sort(key=lambda p: sanitize_relative_path(p, base_dir))
    elif target_path.is_file():
        target_kind = "file"
        label = sanitize_source_label(target_label if target_label is not None else target_path)
        base_dir = None
        discovered = [target_path]
    else:
        raise FileNotFoundError(f"target does not exist: {sanitize_source_label(target_path)}")

    report = CorpusReport(
        schema_version=SCHEMA_VERSION,
        target_label=label,
        target_kind=target_kind,
        max_bytes_per_file=max_bytes_per_file,
        inspect_only=not write_enabled,
        out_dir_label=out_dir_label,
    )

    all_hits: list[tuple[ContainerHit, bytes]] = []
    file_results: list[FileExtractResult] = []

    for fpath in discovered:
        fres, hits = scan_file_for_dxbc(
            fpath,
            base_dir=base_dir,
            max_bytes=max_bytes_per_file,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        file_results.append(fres)
        report.files_scanned += 1
        report.total_file_bytes += fres.file_size
        report.total_bytes_scanned += fres.bytes_scanned
        if fres.error is not None:
            report.files_errored += 1
        if fres.containers:
            report.files_with_containers += 1
            all_hits.extend(hits)

    report.containers_found = len(all_hits)
    report.files = file_results
    fmt, kinds, fourccs = _aggregate_counts(all_hits)
    report.format_counts = fmt
    report.kind_counts = kinds
    report.fourcc_counts = fourccs

    # Deduplicate by SHA-256 of the exact container blob.
    unique: dict[str, tuple[ContainerHit, bytes]] = {}
    for hit, blob in all_hits:
        key = hit.info.sha256_hex
        if key not in unique:
            unique[key] = (hit, blob)
    report.containers_unique = len(unique)
    report.containers_deduplicated = report.containers_found - report.containers_unique

    if write_enabled and destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        written: list[dict[str, Any]] = []
        for sha, (hit, blob) in sorted(unique.items()):
            filename = hit.info.corpus_filename()
            out_path = destination / filename
            # Skip rewrite if identical content already present.
            if out_path.is_file():
                try:
                    existing = out_path.read_bytes()
                    if hashlib.sha256(existing).hexdigest() == sha:
                        written.append(
                            {
                                "filename": filename,
                                "sha256": sha,
                                "size": len(blob),
                                "format": hit.info.format,
                                "shader_kind": hit.info.shader_kind,
                                "shader_model": (
                                    f"{hit.info.sm_major}.{hit.info.sm_minor}"
                                    if hit.info.sm_major is not None
                                    else None
                                ),
                                "has_accc": hit.info.has_accc,
                                "fourccs": list(hit.info.fourccs),
                                "status": "already_present",
                            }
                        )
                        continue
                except OSError:
                    pass
            out_path.write_bytes(blob)
            written.append(
                {
                    "filename": filename,
                    "sha256": sha,
                    "size": len(blob),
                    "format": hit.info.format,
                    "shader_kind": hit.info.shader_kind,
                    "shader_model": (
                        f"{hit.info.sm_major}.{hit.info.sm_minor}"
                        if hit.info.sm_major is not None
                        else None
                    ),
                    "has_accc": hit.info.has_accc,
                    "fourccs": list(hit.info.fourccs),
                    "status": "written",
                }
            )

        report.containers_written = sum(1 for w in written if w["status"] == "written")
        report.written_blobs = written

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "target": label,
            "blob_count": len(written),
            "blobs": written,
            "summary": report.to_dict()["summary"],
        }
        manifest_path = destination / MANIFEST_FILE_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_json(report: CorpusReport) -> str:
    """Render a strictly sanitized deterministic JSON report."""
    return json.dumps(report.to_dict(), indent=2) + "\n"


def render_markdown(report: CorpusReport) -> str:
    """Render a strictly sanitized Markdown summary report."""
    lines: list[str] = [
        f"# DXIL / DXBC Corpus Report: `{_escape_md_cell(report.target_label)}`",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Target kind | `{report.target_kind}` |",
        f"| Inspect only | `{str(report.inspect_only).lower()}` |",
        f"| Files scanned | {report.files_scanned} |",
        f"| Files with containers | {report.files_with_containers} |",
        f"| Files errored | {report.files_errored} |",
        f"| Containers found | {report.containers_found} |",
        f"| Unique containers | {report.containers_unique} |",
        f"| Deduplicated | {report.containers_deduplicated} |",
        f"| Blobs written | {report.containers_written} |",
        "",
    ]

    if report.format_counts:
        lines.extend(["## Formats", "", "| Format | Count |", "| :--- | ---: |"])
        for fmt, count in report.format_counts.items():
            lines.append(f"| `{_escape_md_cell(fmt)}` | {count} |")
        lines.append("")

    if report.kind_counts:
        lines.extend(["## Shader Kinds", "", "| Kind | Count |", "| :--- | ---: |"])
        for kind, count in report.kind_counts.items():
            lines.append(f"| `{_escape_md_cell(kind)}` | {count} |")
        lines.append("")

    if report.fourcc_counts:
        lines.extend(["## Chunk FourCCs", "", "| FourCC | Count |", "| :--- | ---: |"])
        for tag, count in report.fourcc_counts.items():
            lines.append(f"| `{_escape_md_cell(tag)}` | {count} |")
        lines.append("")

    matched = [f for f in report.files if f.containers or f.error]
    if matched:
        lines.extend(
            [
                "## Matched Files",
                "",
                "| File | Containers | Formats | Status |",
                "| :--- | ---: | :--- | :--- |",
            ]
        )
        for entry in matched:
            fmts = ",".join(sorted({c.info.format for c in entry.containers})) or "-"
            status = f"error: {entry.error}" if entry.error else ("truncated" if entry.truncated else "ok")
            lines.append(
                f"| `{_escape_md_cell(entry.path)}` | {len(entry.containers)} "
                f"| `{_escape_md_cell(fmts)}` | {_escape_md_cell(status)} |"
            )
        lines.append("")
    else:
        lines.append("No DXBC / DXIL containers detected.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local DXIL/DXBC corpus extractor & RTS0/PSV0 analyzer "
            "(structural parse of DXIL/PSV0/RTS0/ISG1/OSG1/ACCC)."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--from-env",
        action="store_true",
        help="Scan GTA6_GAME_DIR resolved from environment or .env.",
    )
    source_group.add_argument(
        "--path",
        metavar="DIR_OR_FILE",
        help="Scan a specific local directory or binary file.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="DIR",
        default=None,
        help=(
            "Write deduplicated .dxbc blobs + corpus-manifest.json into a gitignored "
            "directory (work/, local-data/, or outside the repo). Default: inspect-only."
        ),
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Force read-only inspection even when --out-dir is provided.",
    )
    parser.add_argument(
        "--max-bytes-per-file",
        metavar="N",
        type=parse_byte_size,
        default=None,
        help="Maximum bytes to scan per file (e.g. 4194304 or 4M).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit strictly sanitized machine-readable JSON.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    env_path: Path = ENV_FILE,
    env: Mapping[str, str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.from_env:
        target = resolve_game_dir(env_path=env_path, env=env)
        if target is None:
            print(
                'Error: GTA6_GAME_DIR is not set. Run: python tools/local_env.py --set-game "<path>"',
                file=sys.stderr,
            )
            return 1
        target_label = "<GTA6_GAME_DIR>"
    else:
        target = Path(args.path).expanduser()
        target_label = sanitize_source_label(target)

    if not target.exists():
        print(f"Error: scan target does not exist: {target_label}", file=sys.stderr)
        return 1

    try:
        report = extract_corpus(
            target,
            out_dir=args.out_dir,
            inspect_only=args.inspect_only,
            max_bytes_per_file=args.max_bytes_per_file,
            target_label=target_label,
            repo_root=repo_root,
        )
    except UnsafeOutputPathError as exc:
        print(f"Error: {_scrub_sensitive_text(str(exc))}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: {sanitize_error_message(target_label, exc)}", file=sys.stderr)
        return 1

    if args.json:
        sys.stdout.write(render_json(report))
    else:
        sys.stdout.write(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
