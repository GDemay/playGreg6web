#!/usr/bin/env python3
"""Read-only shader container format scanner and classifier (Track 4 / S8).

Scans binary buffers, files, or local game directories in bounded memory using
a streaming sliding-window chunk scanner to detect and structurally validate:

  - DXBC / DXIL containers (`b"DXBC"` + 32-byte header + chunk offset table +
    FourCC tags distinguishing `DXIL_SM6` vs `DXBC_SM5` vs generic `DXBC`)
  - SPIR-V modules (`0x07230203` LE / `0x03022307` BE + version 1.0..1.6 +
    `bound > 0` + `schema == 0` + optional `OpEntryPoint` stage hints)
  - WGPK shader packs (`b"WGPK"` v1 header + entry table matching
    `shaders/src/pack.rs` and `tools/inspect_wgsl_pack.py`)
  - RAGE shader containers (`b"RGSH"`, `b"fxc\\x00"`, and `.fxc` extension hints)

Security & privacy invariants:
  - Strictly read-only (never writes, extracts, or modifies any target files).
  - Never emits host absolute paths, drive letters, usernames, or raw binary
    bytes/dumps in either Markdown or `--json` output.
  - Pure Python 3.12 standard library only (zero external dependencies).

CLI usage:
    python tools/scan_shaders.py (--from-env | --path <dir-or-file>) \\
        [--max-bytes-per-file <N>] [--skip-incomplete] [--json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# Constants matching shaders/src/dxbc.rs, tools/inspect_wgsl_pack.py,
# tools/local_env.py, and tools/src/scan.rs
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB streaming read chunks
DEFAULT_OVERLAP_BYTES = 64 * 1024  # 64 KiB boundary overlap for headers/tables
MIN_CHUNK_SIZE = 64

# DXBC / DXIL container constants (matching shaders/src/dxbc.rs)
DXBC_MAGIC = b"DXBC"
DXBC_HEADER_STRUCT = struct.Struct("<4s16sHHII")
DXBC_HEADER_BYTES = DXBC_HEADER_STRUCT.size  # 32 bytes
DXBC_CHUNK_HEADER_STRUCT = struct.Struct("<4sI")
DXBC_CHUNK_HEADER_BYTES = DXBC_CHUNK_HEADER_STRUCT.size  # 8 bytes
DXBC_MIN_CHUNKS = 1
DXBC_MAX_CHUNKS = 64
DXBC_MAX_CONTAINER_BYTES = 256 * 1024 * 1024  # 256 MiB sanity upper bound
DXIL_PROGRAM_HEADER_BYTES = 24
LLVM_BITCODE_MAGIC = b"\x42\x43\xc0\xde"

DXIL_FOURCCS: frozenset[str] = frozenset({"DXIL", "ILDB", "PSV0"})
DXBC_SM5_FOURCCS: frozenset[str] = frozenset({"SHEX", "SHDR"})
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
    }
)

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

# SPIR-V constants
SPIRV_MAGIC_LE = b"\x03\x02\x23\x07"
SPIRV_MAGIC_BE = b"\x07\x23\x02\x03"
SPIRV_HEADER_LE = struct.Struct("<5I")
SPIRV_HEADER_BE = struct.Struct(">5I")
SPIRV_HEADER_BYTES = 20
SPIRV_MAX_BOUND = 0x0FFF_FFFF
SPIRV_OP_ENTRY_POINT = 15
SPIRV_OP_FUNCTION = 54
SPIRV_MAX_INSPECT_INSTRUCTIONS = 256

SPIRV_EXECUTION_MODELS: dict[int, str] = {
    0: "vertex",
    1: "tessellation_control",
    2: "tessellation_evaluation",
    3: "geometry",
    4: "fragment",
    5: "compute",
    6: "kernel",
    5267: "task",
    5268: "mesh",
    5313: "ray_generation",
    5314: "intersection",
    5315: "any_hit",
    5316: "closest_hit",
    5317: "miss",
    5318: "callable",
    5364: "task",
    5365: "mesh",
}

# WGPK constants (matching shaders/src/pack.rs & tools/inspect_wgsl_pack.py)
WGPK_MAGIC = b"WGPK"
WGPK_VERSION = 1
WGPK_HEADER_STRUCT = struct.Struct("<4sIII")
WGPK_HEADER_BYTES = WGPK_HEADER_STRUCT.size  # 16 bytes
WGPK_ENTRY_STRUCT = struct.Struct("<QB3sII")
WGPK_ENTRY_BYTES = WGPK_ENTRY_STRUCT.size  # 20 bytes
WGPK_MAX_ENTRIES = 1_000_000
WGPK_MAX_INSPECT_ENTRIES = 4096
WGPK_STAGE_NAMES: dict[int, str] = {
    1: "vertex",
    2: "fragment",
    3: "compute",
}

# RAGE shader container signatures
RAGE_RGSH_MAGIC = b"RGSH"
RAGE_FXC_MAGIC = b"fxc\x00"
RAGE_MIN_HEADER_BYTES = 8
RAGE_MAX_EMBEDDED_HEADER_WORD = 0x0010_0000

# Incomplete download heuristics (matching tools/src/scan.rs)
INCOMPLETE_SUFFIXES: tuple[str, ...] = (
    ".!qb",
    ".part",
    ".crdownload",
    ".tmp",
    ".download",
    ".aria2",
    ".bc!",
)
SPARSE_HEADER_PROBE_BYTES = 0x208  # 520 bytes

# Ordered list of binary magics scanned via C-accelerated bytes.find()
_SEARCH_MAGICS: tuple[tuple[bytes, str], ...] = (
    (DXBC_MAGIC, "dxbc"),
    (SPIRV_MAGIC_LE, "spirv_le"),
    (SPIRV_MAGIC_BE, "spirv_be"),
    (WGPK_MAGIC, "wgpk"),
    (RAGE_RGSH_MAGIC, "rage_rgsh"),
    (RAGE_FXC_MAGIC, "rage_fxc"),
)

# Sanitization patterns (matching tools/preflight.py)
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_SIZE_SUFFIX_RE = re.compile(r"^\s*(\d+)\s*([kKmMgG](?:i?[bB])?|[bB])?\s*$")


# ---------------------------------------------------------------------------
# .env resolution & sanitization helpers
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
# Result data models (strictly sanitized metadata; zero raw binary payloads)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShaderHit:
    """Single validated shader container or format signature detected in a stream."""

    format: str
    offset: int
    size: int | None = None
    version: str | None = None
    shader_kind: str | None = None
    shader_model: str | None = None
    stage_hints: tuple[str, ...] = field(default_factory=tuple)
    chunks: tuple[str, ...] = field(default_factory=tuple)
    has_llvm_bitcode: bool | None = None
    endianness: str | None = None
    spirv_bound: int | None = None
    wgpk_entry_count: int | None = None
    wgpk_stage_counts: dict[str, int] | None = None
    confidence: str = "validated"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format": self.format,
            "offset": self.offset,
            "size": self.size,
            "confidence": self.confidence,
            "stage_hints": list(self.stage_hints),
        }
        if self.version is not None:
            payload["version"] = self.version
        if self.shader_kind is not None:
            payload["shader_kind"] = self.shader_kind
        if self.shader_model is not None:
            payload["shader_model"] = self.shader_model
        if self.chunks:
            payload["chunks"] = list(self.chunks)
        if self.has_llvm_bitcode is not None:
            payload["has_llvm_bitcode"] = self.has_llvm_bitcode
        if self.endianness is not None:
            payload["endianness"] = self.endianness
        if self.spirv_bound is not None:
            payload["spirv_bound"] = self.spirv_bound
        if self.wgpk_entry_count is not None:
            payload["wgpk_entry_count"] = self.wgpk_entry_count
        if self.wgpk_stage_counts is not None:
            payload["wgpk_stage_counts"] = dict(self.wgpk_stage_counts)
        return payload


@dataclass(frozen=True)
class SkippedFile:
    """File skipped because `--skip-incomplete` detected a partial/sparse file."""

    path: str
    file_size: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "file_size": self.file_size,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FileScanResult:
    """Read-only shader container scan result for a single file."""

    path: str
    file_size: int
    bytes_scanned: int
    truncated: bool
    hits: tuple[ShaderHit, ...] = field(default_factory=tuple)
    extension_hint: str | None = None
    error: str | None = None
    skipped_reason: str | None = None

    @property
    def format_counts(self) -> dict[str, int]:
        counts = Counter(hit.format for hit in self.hits)
        return dict(sorted(counts.items()))

    @property
    def stage_hints(self) -> tuple[str, ...]:
        stages: set[str] = set()
        for hit in self.hits:
            stages.update(hit.stage_hints)
            if hit.shader_kind and hit.shader_kind != "unknown":
                stages.add(hit.shader_kind)
        return tuple(sorted(stages))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "file_size": self.file_size,
            "bytes_scanned": self.bytes_scanned,
            "truncated": self.truncated,
            "hit_count": len(self.hits),
            "format_counts": self.format_counts,
            "stage_hints": list(self.stage_hints),
            "hits": [hit.to_dict() for hit in self.hits],
        }
        if self.extension_hint is not None:
            payload["extension_hint"] = self.extension_hint
        if self.error is not None:
            payload["error"] = self.error
        if self.skipped_reason is not None:
            payload["skipped_reason"] = self.skipped_reason
        return payload


@dataclass(frozen=True)
class DirectoryScanReport:
    """Aggregated, strictly sanitized shader container scan report."""

    schema_version: int
    target_label: str
    target_kind: str
    max_bytes_per_file: int | None
    skip_incomplete: bool
    files_scanned: int
    files_with_shaders: int
    files_skipped_incomplete: int
    files_errored: int
    total_file_bytes: int
    total_bytes_scanned: int
    total_hits: int
    format_counts: dict[str, int]
    stage_hint_counts: dict[str, int]
    dxbc_chunk_counts: dict[str, int]
    files: tuple[FileScanResult, ...] = field(default_factory=tuple)
    skipped_incomplete: tuple[SkippedFile, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target": self.target_label,
            "target_kind": self.target_kind,
            "max_bytes_per_file": self.max_bytes_per_file,
            "skip_incomplete": self.skip_incomplete,
            "summary": {
                "files_scanned": self.files_scanned,
                "files_with_shaders": self.files_with_shaders,
                "files_skipped_incomplete": self.files_skipped_incomplete,
                "files_errored": self.files_errored,
                "total_file_bytes": self.total_file_bytes,
                "total_bytes_scanned": self.total_bytes_scanned,
                "total_hits": self.total_hits,
                "format_counts": dict(self.format_counts),
                "stage_hint_counts": dict(self.stage_hint_counts),
                "dxbc_chunk_counts": dict(self.dxbc_chunk_counts),
            },
            "files": [entry.to_dict() for entry in self.files],
            "skipped_incomplete": [entry.to_dict() for entry in self.skipped_incomplete],
        }


# ---------------------------------------------------------------------------
# Structural container validators
# ---------------------------------------------------------------------------


def _is_printable_fourcc(raw: bytes) -> bool:
    return len(raw) == 4 and all(0x20 <= b <= 0x7E for b in raw)


def _decode_dxil_program_header(payload: bytes | memoryview) -> tuple[str | None, str | None, bool]:
    """Decode `(shader_kind, shader_model, has_llvm_bitcode)` from a `DXIL` chunk payload."""
    if len(payload) < DXIL_PROGRAM_HEADER_BYTES:
        return None, None, False
    program_version, _size_u32, _dxil_magic, _dxil_ver, bitcode_offset, bitcode_size = (
        struct.unpack_from("<6I", payload, 0)
    )
    kind_raw = (program_version >> 16) & 0xFFFF
    sm_major = (program_version >> 4) & 0xF
    sm_minor = program_version & 0xF
    shader_kind = DXIL_SHADER_KIND_NAMES.get(kind_raw, f"unknown({kind_raw})")
    shader_model = f"{sm_major}.{sm_minor}"

    # BitcodeOffset is relative to byte 8 (DxilBitcodeHeader), matching shaders/src/dxbc.rs.
    bc_start = 8 + bitcode_offset
    has_llvm = (
        bitcode_size >= 4
        and bc_start + 4 <= len(payload)
        and bc_start + bitcode_size <= len(payload)
        and bytes(payload[bc_start : bc_start + 4]) == LLVM_BITCODE_MAGIC
    )
    return shader_kind, shader_model, has_llvm


def _decode_legacy_shader_header(payload: bytes | memoryview) -> tuple[str | None, str | None]:
    """Decode `(shader_kind, shader_model)` from a legacy `SHEX`/`SHDR` chunk payload."""
    if len(payload) < 8:
        return None, None
    version_token, _dword_count = struct.unpack_from("<2I", payload, 0)
    kind_raw = (version_token >> 16) & 0xFFFF
    sm_major = (version_token >> 4) & 0xF
    sm_minor = version_token & 0xF
    shader_kind = DXIL_SHADER_KIND_NAMES.get(kind_raw, f"unknown({kind_raw})")
    shader_model = f"{sm_major}.{sm_minor}"
    return shader_kind, shader_model


def _decode_rdef_shader_model(payload: bytes | memoryview) -> str | None:
    """Decode fallback shader model from an `RDEF` chunk header."""
    if len(payload) < 28:
        return None
    target_minor = payload[16]
    target_major = payload[17]
    if target_major > 15 or target_minor > 15:
        return None
    return f"{target_major}.{target_minor}"


def _validate_dxbc_at(
    buf: bytes,
    rel_offset: int,
    abs_offset: int,
    file_size: int | None,
) -> ShaderHit | None:
    """Validate a candidate DXBC/DXIL container at `rel_offset` in `buf`."""
    buf_len = len(buf)
    if rel_offset + DXBC_HEADER_BYTES > buf_len:
        return None

    magic, _digest, ver_major, ver_minor, total_size, chunk_count = DXBC_HEADER_STRUCT.unpack_from(
        buf, rel_offset
    )
    if magic != DXBC_MAGIC:
        return None

    # Standard DXBC container header uses version 1.0 (allow narrow range 1..2, minor <= 16).
    if ver_major < 1 or ver_major > 2 or ver_minor > 16:
        return None

    if chunk_count < DXBC_MIN_CHUNKS or chunk_count > DXBC_MAX_CHUNKS:
        return None

    table_end = DXBC_HEADER_BYTES + chunk_count * 4
    min_valid_size = table_end + DXBC_CHUNK_HEADER_BYTES
    if total_size < min_valid_size or total_size > DXBC_MAX_CONTAINER_BYTES:
        return None

    if file_size is not None and abs_offset + total_size > file_size:
        return None

    if rel_offset + table_end > buf_len:
        return None

    offsets = struct.unpack_from(f"<{chunk_count}I", buf, rel_offset + DXBC_HEADER_BYTES)

    # Every chunk offset must fall after the offset table and leave room for the 8-byte header.
    seen_chunk_offsets: set[int] = set()
    for off in offsets:
        if off < table_end or off + DXBC_CHUNK_HEADER_BYTES > total_size:
            return None
        if off in seen_chunk_offsets:
            return None
        seen_chunk_offsets.add(off)

    chunk_tags: list[str] = []
    known_tag_count = 0
    readable_chunks = 0

    dxil_kind: str | None = None
    dxil_sm: str | None = None
    has_llvm_bc: bool | None = None
    legacy_kind: str | None = None
    legacy_sm: str | None = None
    rdef_sm: str | None = None

    for off in offsets:
        chunk_pos = rel_offset + off
        if chunk_pos + DXBC_CHUNK_HEADER_BYTES > buf_len:
            continue
        fourcc_bytes, chunk_size = DXBC_CHUNK_HEADER_STRUCT.unpack_from(buf, chunk_pos)
        if not _is_printable_fourcc(fourcc_bytes):
            return None
        if off + DXBC_CHUNK_HEADER_BYTES + chunk_size > total_size:
            return None

        readable_chunks += 1
        tag = fourcc_bytes.decode("ascii")
        chunk_tags.append(tag)
        if tag in KNOWN_DXBC_FOURCCS:
            known_tag_count += 1

        payload_start = chunk_pos + DXBC_CHUNK_HEADER_BYTES
        payload_avail_end = min(payload_start + chunk_size, buf_len)
        payload = buf[payload_start:payload_avail_end]

        if tag == "DXIL" and dxil_kind is None:
            dxil_kind, dxil_sm, has_llvm_bc = _decode_dxil_program_header(payload)
        elif tag in ("SHEX", "SHDR") and legacy_kind is None:
            legacy_kind, legacy_sm = _decode_legacy_shader_header(payload)
        elif tag == "RDEF" and rdef_sm is None:
            rdef_sm = _decode_rdef_shader_model(payload)

    # Require at least one structurally verified, recognized DXBC/DXIL FourCC tag.
    if readable_chunks == 0 or known_tag_count == 0:
        return None

    tag_set = set(chunk_tags)
    if tag_set & DXIL_FOURCCS:
        container_format = "DXIL_SM6"
    elif tag_set & DXBC_SM5_FOURCCS:
        container_format = "DXBC_SM5"
    else:
        container_format = "DXBC"

    shader_kind = dxil_kind if dxil_kind is not None else legacy_kind
    shader_model = (
        dxil_sm if dxil_sm is not None else (legacy_sm if legacy_sm is not None else rdef_sm)
    )
    stage_hints = (shader_kind,) if shader_kind and not shader_kind.startswith("unknown(") else ()

    return ShaderHit(
        format=container_format,
        offset=abs_offset,
        size=total_size,
        version=f"{ver_major}.{ver_minor}",
        shader_kind=shader_kind,
        shader_model=shader_model,
        stage_hints=stage_hints,
        chunks=tuple(chunk_tags),
        has_llvm_bitcode=has_llvm_bc if container_format == "DXIL_SM6" else None,
        confidence="validated",
    )


def _validate_spirv_at(
    buf: bytes,
    rel_offset: int,
    abs_offset: int,
    endian: str,
) -> ShaderHit | None:
    """Validate a candidate SPIR-V module header at `rel_offset` in `buf`."""
    buf_len = len(buf)
    if rel_offset + SPIRV_HEADER_BYTES > buf_len:
        return None

    header_struct = SPIRV_HEADER_LE if endian == "little" else SPIRV_HEADER_BE
    word_struct = struct.Struct("<I" if endian == "little" else ">I")

    magic, version, _generator, bound, schema = header_struct.unpack_from(buf, rel_offset)
    if magic != 0x07230203:
        return None

    # SPIR-V version encoding: 0x00MMmm00 where MM == 1 and 0 <= mm <= 6.
    high_byte = (version >> 24) & 0xFF
    major = (version >> 16) & 0xFF
    minor = (version >> 8) & 0xFF
    low_byte = version & 0xFF
    if high_byte != 0 or low_byte != 0 or major != 1 or minor > 6:
        return None

    if bound == 0 or bound > SPIRV_MAX_BOUND:
        return None

    if schema != 0:
        return None

    # Best-effort scan of header instructions for OpEntryPoint stage hints.
    stages: list[str] = []
    cursor = rel_offset + SPIRV_HEADER_BYTES
    for _ in range(SPIRV_MAX_INSPECT_INSTRUCTIONS):
        if cursor + 4 > buf_len:
            break
        inst = word_struct.unpack_from(buf, cursor)[0]
        word_count = (inst >> 16) & 0xFFFF
        opcode = inst & 0xFFFF
        if word_count == 0:
            break
        next_cursor = cursor + word_count * 4
        if next_cursor > buf_len:
            break
        if opcode == SPIRV_OP_ENTRY_POINT and word_count >= 4:
            exec_model = word_struct.unpack_from(buf, cursor + 4)[0]
            stage_name = SPIRV_EXECUTION_MODELS.get(exec_model)
            if stage_name and stage_name not in stages:
                stages.append(stage_name)
        elif opcode == SPIRV_OP_FUNCTION:
            break
        cursor = next_cursor

    return ShaderHit(
        format="SPIRV",
        offset=abs_offset,
        size=None,
        version=f"{major}.{minor}",
        stage_hints=tuple(stages),
        endianness=endian,
        spirv_bound=bound,
        confidence="validated",
    )


def _validate_wgpk_at(
    buf: bytes,
    rel_offset: int,
    abs_offset: int,
    file_size: int | None,
) -> ShaderHit | None:
    """Validate a candidate WGPK v1 shader pack at `rel_offset` in `buf`."""
    buf_len = len(buf)
    if rel_offset + WGPK_HEADER_BYTES > buf_len:
        return None

    magic, version, entry_count, flags = WGPK_HEADER_STRUCT.unpack_from(buf, rel_offset)
    if magic != WGPK_MAGIC or version != WGPK_VERSION:
        return None
    if entry_count > WGPK_MAX_ENTRIES:
        return None

    data_start = WGPK_HEADER_BYTES + entry_count * WGPK_ENTRY_BYTES
    if file_size is not None and abs_offset + data_start > file_size:
        return None

    stage_counts: dict[str, int] = {"vertex": 0, "fragment": 0, "compute": 0}

    if entry_count == 0:
        # Empty WGPK packs must have zero flags to prevent false positives on random noise.
        if flags != 0:
            return None
        return ShaderHit(
            format="WGPK",
            offset=abs_offset,
            size=WGPK_HEADER_BYTES,
            version=str(version),
            stage_hints=(),
            wgpk_entry_count=0,
            wgpk_stage_counts=stage_counts,
            confidence="validated",
        )

    # Require at least the first entry record to be readable and structurally valid.
    if rel_offset + WGPK_HEADER_BYTES + WGPK_ENTRY_BYTES > buf_len:
        return None

    max_end = data_start
    non_empty_ranges: list[tuple[int, int]] = []
    inspected = 0
    inspect_limit = min(entry_count, WGPK_MAX_INSPECT_ENTRIES)

    for idx in range(inspect_limit):
        entry_pos = rel_offset + WGPK_HEADER_BYTES + idx * WGPK_ENTRY_BYTES
        if entry_pos + WGPK_ENTRY_BYTES > buf_len:
            break
        _shader_hash, stage_raw, _reserved, offset, length = WGPK_ENTRY_STRUCT.unpack_from(
            buf, entry_pos
        )
        stage_name = WGPK_STAGE_NAMES.get(stage_raw)
        if stage_name is None:
            return None
        if offset < data_start:
            return None
        end = offset + length
        if file_size is not None and abs_offset + end > file_size:
            return None
        if end > max_end:
            max_end = end
        if length > 0:
            non_empty_ranges.append((offset, end))
        stage_counts[stage_name] += 1
        inspected += 1

    if inspected == 0:
        return None

    # Reject overlapping non-empty payload ranges (matching inspect_wgsl_pack.py).
    if len(non_empty_ranges) > 1:
        non_empty_ranges.sort()
        for prev, curr in zip(non_empty_ranges, non_empty_ranges[1:]):
            if curr[0] < prev[1]:
                return None

    stage_hints = tuple(name for name in ("vertex", "fragment", "compute") if stage_counts[name] > 0)
    known_size = max_end if inspected == entry_count else None

    return ShaderHit(
        format="WGPK",
        offset=abs_offset,
        size=known_size,
        version=str(version),
        stage_hints=stage_hints,
        wgpk_entry_count=entry_count,
        wgpk_stage_counts=stage_counts if inspected == entry_count else None,
        confidence="validated",
    )


def _validate_rage_at(
    buf: bytes,
    rel_offset: int,
    abs_offset: int,
    magic: bytes,
    extension_hint: str | None,
) -> ShaderHit | None:
    """Validate a candidate RAGE shader signature (`RGSH` or `fxc\\x00`)."""
    if rel_offset + RAGE_MIN_HEADER_BYTES > len(buf):
        return None

    # At offset 0 or inside an `.fxc` file, exact 4-byte magic + 8-byte header is strong.
    # When embedded at an arbitrary offset inside binary blobs, verify plausible header bytes
    # right after the magic to suppress accidental 4-byte collisions in multi-GB archives.
    if abs_offset > 0 and extension_hint != "fxc":
        tail4 = buf[rel_offset + 4 : rel_offset + 8]
        val_le = struct.unpack("<I", tail4)[0]
        val_be = struct.unpack(">I", tail4)[0]
        is_ascii = all(0x20 <= b <= 0x7E for b in tail4)
        if (
            val_le > RAGE_MAX_EMBEDDED_HEADER_WORD
            and val_be > RAGE_MAX_EMBEDDED_HEADER_WORD
            and not is_ascii
        ):
            return None

    fmt = "RAGE_RGSH" if magic == RAGE_RGSH_MAGIC else "RAGE_FXC"
    return ShaderHit(
        format=fmt,
        offset=abs_offset,
        size=None,
        confidence="validated",
    )


# ---------------------------------------------------------------------------
# Streaming chunk scanner core
# ---------------------------------------------------------------------------


def _scan_window(
    window: bytes,
    *,
    window_file_offset: int,
    scan_limit: int,
    file_size: int | None,
    extension_hint: str | None,
    seen: set[tuple[int, str]],
    max_hits: int | None,
    hits: list[ShaderHit],
) -> None:
    """Scan `[0 .. scan_limit)` of `window` for validated shader containers."""
    if scan_limit <= 0 or not window:
        return
    if max_hits is not None and len(hits) >= max_hits:
        return

    candidates: list[tuple[int, str]] = []
    for magic, kind in _SEARCH_MAGICS:
        pos = window.find(magic, 0, scan_limit + len(magic) - 1)
        while pos != -1 and pos < scan_limit:
            candidates.append((pos, kind))
            pos = window.find(magic, pos + 1, scan_limit + len(magic) - 1)

    if not candidates:
        return

    candidates.sort(key=lambda item: item[0])

    for rel_offset, kind in candidates:
        if max_hits is not None and len(hits) >= max_hits:
            break
        abs_offset = window_file_offset + rel_offset
        hit: ShaderHit | None = None

        if kind == "dxbc":
            hit = _validate_dxbc_at(window, rel_offset, abs_offset, file_size)
        elif kind == "spirv_le":
            hit = _validate_spirv_at(window, rel_offset, abs_offset, "little")
        elif kind == "spirv_be":
            hit = _validate_spirv_at(window, rel_offset, abs_offset, "big")
        elif kind == "wgpk":
            hit = _validate_wgpk_at(window, rel_offset, abs_offset, file_size)
        elif kind == "rage_rgsh":
            hit = _validate_rage_at(window, rel_offset, abs_offset, RAGE_RGSH_MAGIC, extension_hint)
        elif kind == "rage_fxc":
            hit = _validate_rage_at(window, rel_offset, abs_offset, RAGE_FXC_MAGIC, extension_hint)

        if hit is not None:
            key = (hit.offset, hit.format)
            if key not in seen:
                seen.add(key)
                hits.append(hit)


def _validate_chunk_params(chunk_size: int, overlap: int) -> tuple[int, int]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be an integer >= {MIN_CHUNK_SIZE}, got {chunk_size!r}")
    if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
        raise ValueError(f"overlap must be a non-negative integer, got {overlap!r}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be strictly smaller than chunk_size ({chunk_size})")
    return chunk_size, overlap


def scan_bytes(
    data: bytes | bytearray | memoryview,
    *,
    max_bytes: int | None = None,
    file_size: int | None = None,
    extension_hint: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
    max_hits: int | None = None,
) -> list[ShaderHit]:
    """Scan an in-memory binary buffer using the same bounded sliding-window engine."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes-like buffer, got {type(data).__name__}")
    if max_bytes is not None and (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0):
        raise ValueError(f"max_bytes must be a positive integer or None, got {max_bytes!r}")

    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)

    view = memoryview(data)
    total_len = len(view)
    effective_file_size = total_len if file_size is None else file_size
    limit = total_len if max_bytes is None else min(total_len, max_bytes)

    hits: list[ShaderHit] = []
    seen: set[tuple[int, str]] = set()

    if limit == 0:
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
        if is_final:
            scan_limit = len(window)
        else:
            scan_limit = max(0, len(window) - overlap)

        _scan_window(
            window,
            window_file_offset=window_file_offset,
            scan_limit=scan_limit,
            file_size=effective_file_size,
            extension_hint=extension_hint,
            seen=seen,
            max_hits=max_hits,
            hits=hits,
        )

        if is_final or (max_hits is not None and len(hits) >= max_hits):
            break

        if overlap > 0:
            tail = window[scan_limit:]
            window_file_offset += scan_limit
        else:
            tail = b""
            window_file_offset += len(window)

    return hits


def check_incomplete_file(path: Path) -> str | None:
    """Return an incomplete-download reason code if `path` looks partial/unallocated, else None."""
    name_lower = path.name.lower()
    for suffix in INCOMPLETE_SUFFIXES:
        if name_lower.endswith(suffix):
            return "temp_suffix"
    try:
        st = path.stat()
    except OSError:
        return None
    if st.st_size == 0:
        return "empty_file"
    if st.st_size >= SPARSE_HEADER_PROBE_BYTES:
        try:
            with path.open("rb") as fp:
                probe = fp.read(SPARSE_HEADER_PROBE_BYTES)
            if len(probe) == SPARSE_HEADER_PROBE_BYTES and not any(probe):
                return "sparse_zero_header"
        except OSError:
            return None
    return None


def scan_file(
    path: str | Path,
    *,
    base_dir: Path | None = None,
    max_bytes: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
    skip_incomplete: bool = False,
    max_hits: int | None = None,
) -> FileScanResult:
    """Stream-scan a single file on disk in bounded memory (`chunk_size + overlap`)."""
    file_path = Path(path)
    if max_bytes is not None and (isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0):
        raise ValueError(f"max_bytes must be a positive integer or None, got {max_bytes!r}")

    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    rel_label = sanitize_relative_path(file_path, base_dir)
    extension_hint = "fxc" if file_path.name.lower().endswith(".fxc") else None

    try:
        st = file_path.stat()
        file_size = st.st_size
    except OSError as exc:
        return FileScanResult(
            path=rel_label,
            file_size=0,
            bytes_scanned=0,
            truncated=False,
            extension_hint=extension_hint,
            error=sanitize_error_message(file_path, exc),
        )

    if skip_incomplete:
        incomplete_reason = check_incomplete_file(file_path)
        if incomplete_reason is not None:
            return FileScanResult(
                path=rel_label,
                file_size=file_size,
                bytes_scanned=0,
                truncated=False,
                extension_hint=extension_hint,
                skipped_reason=incomplete_reason,
            )

    hits: list[ShaderHit] = []
    seen: set[tuple[int, str]] = set()
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

                if is_final:
                    scan_limit = len(window)
                else:
                    scan_limit = max(0, len(window) - overlap)

                _scan_window(
                    window,
                    window_file_offset=window_file_offset,
                    scan_limit=scan_limit,
                    file_size=file_size,
                    extension_hint=extension_hint,
                    seen=seen,
                    max_hits=max_hits,
                    hits=hits,
                )

                if is_final or (max_hits is not None and len(hits) >= max_hits):
                    break

                if overlap > 0:
                    tail = window[scan_limit:]
                    window_file_offset += scan_limit
                else:
                    tail = b""
                    window_file_offset += len(window)
    except OSError as exc:
        return FileScanResult(
            path=rel_label,
            file_size=file_size,
            bytes_scanned=bytes_scanned,
            truncated=file_size > bytes_scanned,
            hits=tuple(hits),
            extension_hint=extension_hint,
            error=sanitize_error_message(file_path, exc),
        )

    # Surface `.fxc` files even if their inner bytecode is compressed or encrypted.
    if extension_hint == "fxc" and not hits:
        hits.append(
            ShaderHit(
                format="RAGE_FXC_HINT",
                offset=0,
                size=file_size,
                confidence="hint",
            )
        )

    return FileScanResult(
        path=rel_label,
        file_size=file_size,
        bytes_scanned=bytes_scanned,
        truncated=file_size > bytes_scanned,
        hits=tuple(hits),
        extension_hint=extension_hint,
    )


def _build_report(
    *,
    target_label: str,
    target_kind: str,
    max_bytes_per_file: int | None,
    skip_incomplete: bool,
    all_results: Sequence[FileScanResult],
    skipped_files: Sequence[SkippedFile],
    include_empty: bool,
) -> DirectoryScanReport:
    format_counts: Counter[str] = Counter()
    stage_hint_counts: Counter[str] = Counter()
    dxbc_chunk_counts: Counter[str] = Counter()

    files_scanned = 0
    files_with_shaders = 0
    files_errored = 0
    total_file_bytes = 0
    total_bytes_scanned = 0
    total_hits = 0
    reported_files: list[FileScanResult] = []

    for res in all_results:
        files_scanned += 1
        total_file_bytes += res.file_size
        total_bytes_scanned += res.bytes_scanned

        if res.error is not None:
            files_errored += 1

        if res.hits:
            files_with_shaders += 1
            total_hits += len(res.hits)
            for hit in res.hits:
                format_counts[hit.format] += 1
                for stage in hit.stage_hints:
                    stage_hint_counts[stage] += 1
                for chunk in hit.chunks:
                    dxbc_chunk_counts[chunk] += 1

        if include_empty or target_kind == "file" or res.hits or res.error is not None:
            reported_files.append(res)

    return DirectoryScanReport(
        schema_version=SCHEMA_VERSION,
        target_label=target_label,
        target_kind=target_kind,
        max_bytes_per_file=max_bytes_per_file,
        skip_incomplete=skip_incomplete,
        files_scanned=files_scanned,
        files_with_shaders=files_with_shaders,
        files_skipped_incomplete=len(skipped_files),
        files_errored=files_errored,
        total_file_bytes=total_file_bytes,
        total_bytes_scanned=total_bytes_scanned,
        total_hits=total_hits,
        format_counts=dict(sorted(format_counts.items())),
        stage_hint_counts=dict(sorted(stage_hint_counts.items())),
        dxbc_chunk_counts=dict(sorted(dxbc_chunk_counts.items())),
        files=tuple(reported_files),
        skipped_incomplete=tuple(skipped_files),
    )


def scan_directory(
    root_dir: str | Path,
    *,
    max_bytes_per_file: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
    skip_incomplete: bool = False,
    target_label: str | None = None,
    include_empty: bool = False,
    max_hits_per_file: int | None = None,
) -> DirectoryScanReport:
    """Recursively scan a directory for shader containers in deterministic sorted order."""
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {sanitize_source_label(root)}")

    if max_bytes_per_file is not None and (
        isinstance(max_bytes_per_file, bool)
        or not isinstance(max_bytes_per_file, int)
        or max_bytes_per_file <= 0
    ):
        raise ValueError(
            f"max_bytes_per_file must be a positive integer or None, got {max_bytes_per_file!r}"
        )
    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)

    label = sanitize_source_label(target_label if target_label is not None else root)

    discovered_files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        base = Path(dirpath)
        for fname in filenames:
            discovered_files.append(base / fname)

    discovered_files.sort(key=lambda p: sanitize_relative_path(p, root))

    all_results: list[FileScanResult] = []
    skipped_files: list[SkippedFile] = []

    for fpath in discovered_files:
        if skip_incomplete:
            reason = check_incomplete_file(fpath)
            if reason is not None:
                try:
                    fsize = fpath.stat().st_size
                except OSError:
                    fsize = 0
                skipped_files.append(
                    SkippedFile(
                        path=sanitize_relative_path(fpath, root),
                        file_size=fsize,
                        reason=reason,
                    )
                )
                continue

        res = scan_file(
            fpath,
            base_dir=root,
            max_bytes=max_bytes_per_file,
            chunk_size=chunk_size,
            overlap=overlap,
            skip_incomplete=False,
            max_hits=max_hits_per_file,
        )
        all_results.append(res)

    return _build_report(
        target_label=label,
        target_kind="directory",
        max_bytes_per_file=max_bytes_per_file,
        skip_incomplete=skip_incomplete,
        all_results=all_results,
        skipped_files=skipped_files,
        include_empty=include_empty,
    )


def scan_path(
    target: str | Path,
    *,
    max_bytes_per_file: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
    skip_incomplete: bool = False,
    target_label: str | None = None,
    include_empty: bool = False,
    max_hits_per_file: int | None = None,
) -> DirectoryScanReport:
    """Scan either a directory or a single file and return a unified `DirectoryScanReport`."""
    target_path = Path(target)
    if target_path.is_dir():
        return scan_directory(
            target_path,
            max_bytes_per_file=max_bytes_per_file,
            chunk_size=chunk_size,
            overlap=overlap,
            skip_incomplete=skip_incomplete,
            target_label=target_label,
            include_empty=include_empty,
            max_hits_per_file=max_hits_per_file,
        )
    if target_path.is_file():
        label = sanitize_source_label(target_label if target_label is not None else target_path)
        if skip_incomplete:
            reason = check_incomplete_file(target_path)
            if reason is not None:
                try:
                    fsize = target_path.stat().st_size
                except OSError:
                    fsize = 0
                return _build_report(
                    target_label=label,
                    target_kind="file",
                    max_bytes_per_file=max_bytes_per_file,
                    skip_incomplete=True,
                    all_results=(),
                    skipped_files=(SkippedFile(path=label, file_size=fsize, reason=reason),),
                    include_empty=include_empty,
                )

        res = scan_file(
            target_path,
            max_bytes=max_bytes_per_file,
            chunk_size=chunk_size,
            overlap=overlap,
            skip_incomplete=False,
            max_hits=max_hits_per_file,
        )
        return _build_report(
            target_label=label,
            target_kind="file",
            max_bytes_per_file=max_bytes_per_file,
            skip_incomplete=skip_incomplete,
            all_results=(res,),
            skipped_files=(),
            include_empty=True,
        )
    raise FileNotFoundError(f"target does not exist: {sanitize_source_label(target_path)}")


# ---------------------------------------------------------------------------
# Output rendering (Markdown & deterministic JSON)
# ---------------------------------------------------------------------------


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_json(report: DirectoryScanReport) -> str:
    """Render a strictly sanitized deterministic JSON report."""
    return json.dumps(report.to_dict(), indent=2) + "\n"


def render_markdown(report: DirectoryScanReport) -> str:
    """Render a strictly sanitized Markdown summary report."""
    lines: list[str] = [
        f"# Shader Container Scan Report: `{_escape_md_cell(report.target_label)}`",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Target kind | `{report.target_kind}` |",
        f"| Files scanned | {report.files_scanned} |",
        f"| Files with shaders | {report.files_with_shaders} |",
        f"| Files skipped (incomplete) | {report.files_skipped_incomplete} |",
        f"| Files errored | {report.files_errored} |",
        f"| Total bytes scanned | {report.total_bytes_scanned} / {report.total_file_bytes} |",
        f"| Total shader hits | {report.total_hits} |",
        "",
    ]

    if report.format_counts:
        lines.extend(
            [
                "## Detected Container Formats",
                "",
                "| Format | Hits |",
                "| :--- | ---: |",
            ]
        )
        for fmt, count in report.format_counts.items():
            lines.append(f"| `{_escape_md_cell(fmt)}` | {count} |")
        lines.append("")

    if report.stage_hint_counts:
        lines.extend(
            [
                "## Stage Hints",
                "",
                "| Stage | Hits |",
                "| :--- | ---: |",
            ]
        )
        for stage, count in report.stage_hint_counts.items():
            lines.append(f"| `{_escape_md_cell(stage)}` | {count} |")
        lines.append("")

    if report.dxbc_chunk_counts:
        lines.extend(
            [
                "## DXBC / DXIL Chunk FourCC Distribution",
                "",
                "| FourCC | Count |",
                "| :--- | ---: |",
            ]
        )
        for tag, count in report.dxbc_chunk_counts.items():
            lines.append(f"| `{_escape_md_cell(tag)}` | {count} |")
        lines.append("")

    matched_or_errored = [f for f in report.files if f.hits or f.error]
    if matched_or_errored:
        lines.extend(
            [
                "## Matched Files",
                "",
                "| File | Scanned | Formats | Stages | Status |",
                "| :--- | ---: | :--- | :--- | :--- |",
            ]
        )
        for entry in matched_or_errored:
            fmts = ", ".join(f"{k}:{v}" for k, v in entry.format_counts.items()) or "-"
            stages = ", ".join(entry.stage_hints) or "-"
            status = (
                f"error: {entry.error}"
                if entry.error
                else ("truncated" if entry.truncated else "ok")
            )
            lines.append(
                f"| `{_escape_md_cell(entry.path)}` | {entry.bytes_scanned}/{entry.file_size} "
                f"| `{_escape_md_cell(fmts)}` | `{_escape_md_cell(stages)}` | {_escape_md_cell(status)} |"
            )
        lines.append("")
    else:
        lines.append("No shader containers detected.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only shader container format scanner & classifier "
            "(DXBC/DXIL, SPIR-V, WGPK, RAGE)."
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
        "--max-bytes-per-file",
        metavar="N",
        type=parse_byte_size,
        default=None,
        help="Maximum bytes to scan per file (e.g. 4194304 or 4M) for fast sampling.",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip partial downloads (.!qB, .part, .tmp, empty, or sparse zero-header files).",
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
        report = scan_path(
            target,
            max_bytes_per_file=args.max_bytes_per_file,
            skip_incomplete=args.skip_incomplete,
            target_label=target_label,
        )
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
