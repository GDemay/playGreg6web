#!/usr/bin/env python3
"""Read-only AMD RDNA2 / GFX10.3 shader binary & pipeline record inspector (Track 4 / S15).

Scans standalone RDNA2/GFX10.3 shader blobs, multi-shader buffers, or large
binary containers/directories in bounded streaming memory (`chunk_size + overlap`)
to detect and classify GFX10.3 shader code regions and console pipeline records:

  - GFX10.3 terminators: `s_endpgm` (`0xBF810000`), `s_endpgm_saved` (`0xBF9B0000`)
  - GFX10.3 `EXP` instructions (`(word0 & 0xFC000000) == 0xF8000000`), decoding
    `target` bits `[9:4]` (`0..7` = `mrt0..mrt7`, `8` = `mrtz`, `9` = `null`,
    `12..16` = `pos0..pos4`, `32..63` = `param0..param31`) and `done` bit `11`
  - GFX10.3 `SMEM` (`0xF4000000`), `VINTRP` (`0xC8000000`), `MIMG` (`0xF0000000`),
    plus `VALU`, `SALU`, `SOPP`, `MUBUF`/`MTBUF`, `DS`, and `FLAT` instruction counts
  - Stage heuristics:
      * `vertex`  if `pos0` export is present (without pixel-only `mrt*`/`v_interp`)
      * `pixel`   if `mrt0..7`/`mrtz` export or `v_interp` is present
      * `compute` if `s_endpgm`/`s_endpgm_saved` + `smem`/`valu` without any `exp`
      * `unknown` otherwise
  - Console pipeline record hints: Xbox Scarlett `ACCC` chunks and `AMDGPU` ELF
    (`e_machine == 224`) code objects
  - Deterministic FNV-1a 64-bit hashes and size / export target histograms

Security & privacy invariants:
  - Strictly read-only (never writes, extracts, or modifies any target files).
  - Never emits host absolute paths (Windows drives, UNC, POSIX `/home/`, `/Users/`,
    etc.), active OS usernames, or raw shader bytes/hex dumps in JSON or text output.
  - Pure Python 3.12 standard library only (zero external dependencies).

CLI usage:
    python tools/inspect_rdna2_shaders.py <path> [--json] [--max-hits N] \\
        [--min-dwords N] [--max-bytes N]
    python tools/inspect_rdna2_shaders.py --from-env [--json] [--max-hits N] \\
        [--min-dwords N] [--max-bytes N]
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
# Constants & GFX10.3 / RDNA2 ISA encodings
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB streaming read chunks
DEFAULT_OVERLAP_BYTES = 256 * 1024  # 256 KiB backward overlap for shader bodies
DEFAULT_MAX_SHADER_BYTES = 256 * 1024  # 256 KiB max backward window per shader
DEFAULT_MIN_DWORDS = 2
MIN_CHUNK_SIZE = 64

# FNV-1a 64-bit constants (matching shaders/src/convert.rs & tools/inspect_wgsl_pack.py)
FNV1A_64_OFFSET_BASIS = 0xCBF29CE484222325
FNV1A_64_PRIME = 0x00000100000001B3
U64_MAX = 0xFFFFFFFFFFFFFFFF

# GFX10.3 / RDNA2 exact SOPP words
S_ENDPGM_WORD = 0xBF810000
S_ENDPGM_SAVED_WORD = 0xBF9B0000
S_CODE_END_WORD = 0xBF9F0000

S_ENDPGM_BYTES = struct.pack("<I", S_ENDPGM_WORD)
S_ENDPGM_SAVED_BYTES = struct.pack("<I", S_ENDPGM_SAVED_WORD)

# GFX10.3 top-bit masks
MASK_6BIT = 0xFC000000
MASK_7BIT = 0xFE000000
MASK_9BIT = 0xFF800000
MASK_2BIT = 0xC0000000
MASK_4BIT = 0xF0000000

ENC_EXP = 0xF8000000
ENC_SMEM = 0xF4000000
ENC_MIMG = 0xF0000000
ENC_MTBUF = 0xE8000000
ENC_MUBUF = 0xE0000000
ENC_FLAT = 0xDC000000
ENC_DS = 0xD8000000
ENC_VOP3 = 0xD4000000
ENC_VOP3P = 0xCC000000
ENC_VINTRP = 0xC8000000

ENC_SOPP = 0xBF800000
ENC_SOPC = 0xBF000000
ENC_SOP1 = 0xBE800000
ENC_SOPK = 0xB0000000
ENC_SOP2 = 0x80000000

ENC_VOP1 = 0x7E000000
ENC_VOPC = 0x7C000000

# VOP2 opcodes that always carry a mandatory 32-bit literal constant in GFX10.3
_VOP2_LITERAL_OPS: frozenset[int] = frozenset({0x1D, 0x1E, 0x2C, 0x2D, 0x36, 0x37})
# SRC0 encodings that append an extra 32-bit word (literal, SDWA, DPP)
_SRC0_EXTRA_DWORD: frozenset[int] = frozenset({0xE9, 0xEA, 0xF9, 0xFA, 0xFF})
# SOPK opcode with mandatory 32-bit literal (s_setreg_imm32_b32)
_SOPK_LITERAL_OP = 21

# Unassigned 6-bit encoding prefixes in GFX10.3 (invalid as word0)
_INVALID_TOP6_PREFIXES: frozenset[int] = frozenset(
    {
        0xC0000000,
        0xC4000000,
        0xD0000000,
        0xE4000000,
        0xEC000000,
        0xFC000000,
    }
)

# Container boundary words (stop backward shader walk if not an operand word)
_CONTAINER_MAGIC_WORDS: frozenset[int] = frozenset(
    {
        struct.unpack("<I", b"DXBC")[0],
        struct.unpack("<I", b"ACCC")[0],
        struct.unpack("<I", b"\x7fELF")[0],
    }
)

# Common padding words between shaders
_PADDING_WORDS: frozenset[int] = frozenset(
    {
        0x00000000,
        S_CODE_END_WORD,
        0xFFFFFFFF,
        0xCCCCCCCC,
    }
)

# Console pipeline record magics
ELF_MAGIC = b"\x7fELF"
ACCC_MAGIC = b"ACCC"
EM_AMDGPU = 224
ACCC_MAX_CHUNK_BYTES = 64 * 1024 * 1024  # 64 MiB sanity bound

AMDGPU_ELF_MACHINES: dict[int, str] = {
    0x33: "gfx1010",
    0x34: "gfx1011",
    0x35: "gfx1012",
    0x36: "gfx1030",
    0x37: "gfx1031",
    0x38: "gfx1032",
    0x39: "gfx1033",
    0x3A: "gfx1034",
    0x3B: "gfx1035",
    0x3C: "gfx1036",
}

# Size histogram bucket definitions (inclusive upper bound in bytes, label)
SIZE_HISTOGRAM_BUCKETS: tuple[tuple[int | None, str], ...] = (
    (64, "0..64B"),
    (256, "65..256B"),
    (1024, "257..1024B"),
    (4096, "1025..4096B"),
    (16384, "4097..16384B"),
    (65536, "16385..65536B"),
    (None, ">65536B"),
)

# Sanitization patterns (matching tools/scan_shaders.py & tools/extract_dxil_corpus.py)
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_SIZE_SUFFIX_RE = re.compile(r"^\s*(\d+)\s*([kKmMgG](?:i?[bB])?|[bB])?\s*$")


# ---------------------------------------------------------------------------
# .env resolution, hashing, and strict sanitization helpers
# ---------------------------------------------------------------------------


def fnv1a_64(data: bytes | bytearray | memoryview) -> int:
    """Compute deterministic 64-bit FNV-1a hash matching `shaders/src/convert.rs`."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes-like buffer, got {type(data).__name__}")
    digest = FNV1A_64_OFFSET_BASIS
    for byte in bytes(data):
        digest ^= byte
        digest = (digest * FNV1A_64_PRIME) & U64_MAX
    return digest


def format_hash_64(digest: int) -> str:
    """Format a 64-bit integer digest as canonical `0x` + 16 lowercase hex digits."""
    return f"0x{digest & U64_MAX:016x}"


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
    """Remove host absolute paths (Windows, UNC, POSIX) and active OS usernames."""
    sanitized = _WINDOWS_PATH_RE.sub("<path>", text)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(username)}(?![A-Za-z0-9])",
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
        return _scrub_sensitive_text(raw)
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
        reason = _scrub_sensitive_text(exc.strerror or type(exc).__name__)
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


def parse_positive_int(raw: str | int) -> int:
    """Parse a strictly positive integer CLI argument."""
    if isinstance(raw, bool):
        raise argparse.ArgumentTypeError("value must be a positive integer")
    try:
        val = int(raw)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {raw!r}") from exc
    if val <= 0:
        raise argparse.ArgumentTypeError(f"value must be > 0, got {val}")
    return val


def size_histogram_bucket(byte_size: int) -> str:
    """Map a shader byte size to its deterministic histogram bucket label."""
    for upper, label in SIZE_HISTOGRAM_BUCKETS:
        if upper is None or byte_size <= upper:
            return label
    return SIZE_HISTOGRAM_BUCKETS[-1][1]


# ---------------------------------------------------------------------------
# Result data models (strictly sanitized metadata; zero raw shader bytes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportInstruction:
    """Decoded GFX10.3 `EXP` instruction metadata (no raw opcode/register bytes)."""

    target_id: int
    target: str
    done: bool
    en_mask: int
    compr: bool
    vm: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "target_id": self.target_id,
            "done": self.done,
            "en_mask": self.en_mask,
            "compr": self.compr,
            "vm": self.vm,
        }


@dataclass(frozen=True)
class Rdna2ShaderHit:
    """Single detected AMD RDNA2 / GFX10.3 shader region (zero raw shader bytes)."""

    offset: int
    end_offset: int
    byte_size: int
    dword_count: int
    instruction_count: int
    fnv1a_64: int
    stage: str
    terminator: str
    exports: tuple[ExportInstruction, ...] = field(default_factory=tuple)
    export_targets: tuple[str, ...] = field(default_factory=tuple)
    has_done_export: bool = False
    exp_count: int = 0
    smem_count: int = 0
    vintrp_count: int = 0
    mimg_count: int = 0
    valu_count: int = 0
    salu_count: int = 0
    sopp_count: int = 0
    vmem_count: int = 0
    ds_count: int = 0
    flat_count: int = 0

    @property
    def hash_hex(self) -> str:
        return format_hash_64(self.fnv1a_64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash_hex,
            "offset": self.offset,
            "end_offset": self.end_offset,
            "byte_size": self.byte_size,
            "dword_count": self.dword_count,
            "instruction_count": self.instruction_count,
            "stage": self.stage,
            "terminator": self.terminator,
            "export_targets": list(self.export_targets),
            "has_done_export": self.has_done_export,
            "exports": [exp.to_dict() for exp in self.exports],
            "instruction_counts": {
                "exp": self.exp_count,
                "smem": self.smem_count,
                "vintrp": self.vintrp_count,
                "mimg": self.mimg_count,
                "valu": self.valu_count,
                "salu": self.salu_count,
                "sopp": self.sopp_count,
                "vmem": self.vmem_count,
                "ds": self.ds_count,
                "flat": self.flat_count,
            },
        }


@dataclass(frozen=True)
class PipelineRecordHit:
    """Console / AMDGPU pipeline container record detected during scanning."""

    kind: str
    offset: int
    size: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "offset": self.offset,
            "size": self.size,
        }
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True)
class FileInspectResult:
    """Read-only RDNA2 shader scan result for a single file or buffer."""

    path: str
    file_size: int
    bytes_scanned: int
    truncated: bool
    hit_limit_reached: bool = False
    unaligned_tail_bytes: int = 0
    shaders: tuple[Rdna2ShaderHit, ...] = field(default_factory=tuple)
    pipeline_records: tuple[PipelineRecordHit, ...] = field(default_factory=tuple)
    error: str | None = None

    @property
    def stage_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(s.stage for s in self.shaders).items()))

    @property
    def export_target_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for shader in self.shaders:
            for exp in shader.exports:
                counts[exp.target] += 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "file_size": self.file_size,
            "bytes_scanned": self.bytes_scanned,
            "truncated": self.truncated,
            "hit_limit_reached": self.hit_limit_reached,
            "unaligned_tail_bytes": self.unaligned_tail_bytes,
            "shader_count": len(self.shaders),
            "pipeline_record_count": len(self.pipeline_records),
            "stage_counts": self.stage_counts,
            "export_target_counts": self.export_target_counts,
            "shaders": [s.to_dict() for s in self.shaders],
            "pipeline_records": [r.to_dict() for r in self.pipeline_records],
        }
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class InspectReport:
    """Aggregated, strictly sanitized RDNA2 / GFX10.3 shader inspection report."""

    schema_version: int
    target_label: str
    target_kind: str
    max_bytes: int | None
    max_hits: int | None
    min_dwords: int
    files_scanned: int
    files_with_shaders: int
    files_errored: int
    total_file_bytes: int
    total_bytes_scanned: int
    total_shaders: int
    unique_shader_hashes: int
    total_pipeline_records: int
    hit_limit_reached: bool
    stage_counts: dict[str, int]
    terminator_counts: dict[str, int]
    export_target_histogram: dict[str, int]
    size_histogram: dict[str, int]
    instruction_totals: dict[str, int]
    pipeline_record_counts: dict[str, int]
    files: tuple[FileInspectResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target": self.target_label,
            "target_kind": self.target_kind,
            "max_bytes": self.max_bytes,
            "max_hits": self.max_hits,
            "min_dwords": self.min_dwords,
            "summary": {
                "files_scanned": self.files_scanned,
                "files_with_shaders": self.files_with_shaders,
                "files_errored": self.files_errored,
                "total_file_bytes": self.total_file_bytes,
                "total_bytes_scanned": self.total_bytes_scanned,
                "total_shaders": self.total_shaders,
                "unique_shader_hashes": self.unique_shader_hashes,
                "total_pipeline_records": self.total_pipeline_records,
                "hit_limit_reached": self.hit_limit_reached,
                "stage_counts": dict(self.stage_counts),
                "terminator_counts": dict(self.terminator_counts),
                "export_target_histogram": dict(self.export_target_histogram),
                "size_histogram": dict(self.size_histogram),
                "instruction_totals": dict(self.instruction_totals),
                "pipeline_record_counts": dict(self.pipeline_record_counts),
            },
            "files": [f.to_dict() for f in self.files],
        }


# ---------------------------------------------------------------------------
# GFX10.3 / RDNA2 instruction decoder & stage heuristics
# ---------------------------------------------------------------------------


def decode_exp_target(target_id: int) -> str:
    """Decode GFX10.3 `EXP` target bits `[9:4]` into a symbolic target name."""
    tid = target_id & 0x3F
    if 0 <= tid <= 7:
        return f"mrt{tid}"
    if tid == 8:
        return "mrtz"
    if tid == 9:
        return "null"
    if 12 <= tid <= 16:
        return f"pos{tid - 12}"
    if 32 <= tid <= 63:
        return f"param{tid - 32}"
    return f"unknown({tid})"


def decode_exp_instruction(word0: int) -> ExportInstruction | None:
    """Decode a GFX10.3 `EXP` `word0` if `(word0 & 0xFC000000) == 0xF8000000`."""
    if (word0 & MASK_6BIT) != ENC_EXP:
        return None
    target_id = (word0 >> 4) & 0x3F
    return ExportInstruction(
        target_id=target_id,
        target=decode_exp_target(target_id),
        done=bool((word0 >> 11) & 1),
        en_mask=word0 & 0xF,
        compr=bool((word0 >> 10) & 1),
        vm=bool((word0 >> 12) & 1),
    )


def _classify_word0(word0: int) -> tuple[str, int]:
    """Classify a GFX10.3 `word0` into `(category, dword_length)`.

    Returns `("invalid", 1)` for unassigned GFX10.3 6-bit encoding prefixes and
    `("zero", 1)` for `0x00000000` so zero-filled buffers are not counted as VALU.
    """
    if word0 == 0:
        return "zero", 1

    top6 = word0 & MASK_6BIT

    # 64-bit / specialized vector & memory encodings (top 6 bits)
    if top6 == ENC_EXP:
        return "exp", 2
    if top6 == ENC_SMEM:
        return "smem", 2
    if top6 == ENC_VINTRP:
        return "vintrp", 1
    if top6 == ENC_MIMG:
        # GFX10.3 MIMG: 2 base dwords (+ NSA extension dwords if nsize != 0)
        nsize = (word0 >> 1) & 0x3
        return "mimg", 2 + nsize
    if top6 == ENC_MUBUF or top6 == ENC_MTBUF:
        return "vmem", 2
    if top6 == ENC_DS:
        return "ds", 2
    if top6 == ENC_FLAT:
        return "flat", 2
    if top6 == ENC_VOP3 or top6 == ENC_VOP3P:
        return "valu", 2
    if top6 in _INVALID_TOP6_PREFIXES:
        return "invalid", 1

    # Scalar encodings (0x80000000 .. 0xBFFFFFFF)
    top9 = word0 & MASK_9BIT
    if top9 == ENC_SOPP:
        return "sopp", 1
    if top9 == ENC_SOPC:
        ssrc0 = word0 & 0xFF
        ssrc1 = (word0 >> 8) & 0xFF
        return "salu", 2 if (ssrc0 == 0xFF or ssrc1 == 0xFF) else 1
    if top9 == ENC_SOP1:
        ssrc0 = word0 & 0xFF
        return "salu", 2 if ssrc0 == 0xFF else 1
    if (word0 & MASK_4BIT) == ENC_SOPK:
        op = (word0 >> 23) & 0x1F
        return "salu", 2 if op == _SOPK_LITERAL_OP else 1
    if (word0 & MASK_2BIT) == ENC_SOP2:
        ssrc0 = word0 & 0xFF
        ssrc1 = (word0 >> 8) & 0xFF
        return "salu", 2 if (ssrc0 == 0xFF or ssrc1 == 0xFF) else 1

    # Vector ALU 32-bit encodings (bit 31 == 0)
    top7 = word0 & MASK_7BIT
    if top7 == ENC_VOP1 or top7 == ENC_VOPC:
        src0 = word0 & 0x1FF
        return "valu", 2 if src0 in _SRC0_EXTRA_DWORD else 1

    # VOP2 (remaining bit 31 == 0 encodings)
    op = (word0 >> 25) & 0x3F
    src0 = word0 & 0x1FF
    if op in _VOP2_LITERAL_OPS or src0 in _SRC0_EXTRA_DWORD:
        return "valu", 2
    return "valu", 1


def _word_takes_extra_dword(word0: int) -> bool:
    """Return True if `word0` is a valid GFX10.3 instruction taking a following dword."""
    cat, length = _classify_word0(word0)
    return cat not in ("invalid", "zero") and length >= 2


def infer_shader_stage(
    *,
    export_targets: Sequence[str],
    exp_count: int,
    vintrp_count: int,
    smem_count: int,
    valu_count: int,
) -> str:
    """Infer GFX10.3 shader stage from decoded exports and instruction mix.

    Rules:
      - `vertex`  if `pos0` export is present (without pixel-only `mrt*`/`v_interp`)
      - `pixel`   if `mrt0..7`/`mrtz` export or `v_interp` is present
      - `compute` if `smem` or `valu` is present with zero `exp` instructions
      - `unknown` otherwise
    """
    target_set = set(export_targets)
    has_pos0 = "pos0" in target_set
    has_mrt = any(t.startswith("mrt") for t in target_set)
    has_vinterp = vintrp_count > 0

    if has_pos0 and not (has_mrt or has_vinterp):
        return "vertex"
    if has_mrt or has_vinterp:
        return "pixel"
    if has_pos0:
        return "vertex"
    if exp_count == 0 and (smem_count > 0 or valu_count > 0):
        return "compute"
    return "unknown"


def _find_shader_start_index(body_words: tuple[int, ...]) -> int:
    """Walk backwards through `body_words` (before terminator) to locate shader start."""
    n = len(body_words)
    if n == 0:
        return 0

    start_idx = 0
    idx = n - 1
    while idx >= 0:
        w = body_words[idx]

        # Hard shader terminators / AMD icache alignment padding (`s_code_end`).
        if w in (S_ENDPGM_WORD, S_ENDPGM_SAVED_WORD, S_CODE_END_WORD):
            start_idx = idx + 1
            break

        # Known container magics (unless consumed as operand/literal of previous opcode).
        if w in _CONTAINER_MAGIC_WORDS:
            if idx > 0 and _word_takes_extra_dword(body_words[idx - 1]):
                idx -= 2
                continue
            start_idx = idx + 1
            break

        # Repeated padding runs (`0x00000000`, `0xFFFFFFFF`, `0xCCCCCCCC`).
        if w in _PADDING_WORDS and idx > 0 and body_words[idx - 1] == w:
            start_idx = idx + 1
            break

        # Unassigned GFX10.3 encoding prefix (unless operand/literal of previous opcode).
        cat, _length = _classify_word0(w)
        if cat == "invalid":
            if idx > 0 and _word_takes_extra_dword(body_words[idx - 1]):
                idx -= 2
                continue
            start_idx = idx + 1
            break

        idx -= 1

    # Strip any remaining leading padding dwords at the start of the shader.
    while start_idx < n and body_words[start_idx] in _PADDING_WORDS:
        start_idx += 1

    return start_idx


def analyze_shader_region(
    raw_bytes: bytes | bytearray | memoryview,
    *,
    abs_offset: int,
    terminator_word: int,
) -> Rdna2ShaderHit:
    """Decode a contiguous GFX10.3 shader region (including trailing terminator)."""
    blob = bytes(raw_bytes)
    dword_count = len(blob) // 4
    words = struct.unpack(f"<{dword_count}I", blob[: dword_count * 4])

    body_len = max(0, dword_count - 1)
    exports: list[ExportInstruction] = []
    counts: dict[str, int] = {
        "exp": 0,
        "smem": 0,
        "vintrp": 0,
        "mimg": 0,
        "valu": 0,
        "salu": 0,
        "sopp": 1,  # terminator itself is SOPP
        "vmem": 0,
        "ds": 0,
        "flat": 0,
    }
    instruction_count = 1  # terminator

    i = 0
    while i < body_len:
        w0 = words[i]
        cat, inst_dwords = _classify_word0(w0)
        if cat == "exp":
            exp = decode_exp_instruction(w0)
            if exp is not None:
                exports.append(exp)
            counts["exp"] += 1
            instruction_count += 1
            step = 2 if (i + 2 <= body_len) else 1
        elif cat == "mimg":
            counts["mimg"] += 1
            instruction_count += 1
            step = inst_dwords if (i + inst_dwords <= body_len) else (2 if i + 2 <= body_len else 1)
        elif cat in counts:
            counts[cat] += 1
            instruction_count += 1
            step = inst_dwords if (i + inst_dwords <= body_len) else 1
        else:
            step = 1
        i += step

    export_targets = tuple(sorted({e.target for e in exports}))
    has_done_export = any(e.done for e in exports)
    stage = infer_shader_stage(
        export_targets=export_targets,
        exp_count=counts["exp"],
        vintrp_count=counts["vintrp"],
        smem_count=counts["smem"],
        valu_count=counts["valu"],
    )
    terminator_name = "s_endpgm" if terminator_word == S_ENDPGM_WORD else "s_endpgm_saved"
    shader_bytes = blob[: dword_count * 4]

    return Rdna2ShaderHit(
        offset=abs_offset,
        end_offset=abs_offset + len(shader_bytes),
        byte_size=len(shader_bytes),
        dword_count=dword_count,
        instruction_count=instruction_count,
        fnv1a_64=fnv1a_64(shader_bytes),
        stage=stage,
        terminator=terminator_name,
        exports=tuple(exports),
        export_targets=export_targets,
        has_done_export=has_done_export,
        exp_count=counts["exp"],
        smem_count=counts["smem"],
        vintrp_count=counts["vintrp"],
        mimg_count=counts["mimg"],
        valu_count=counts["valu"],
        salu_count=counts["salu"],
        sopp_count=counts["sopp"],
        vmem_count=counts["vmem"],
        ds_count=counts["ds"],
        flat_count=counts["flat"],
    )


# ---------------------------------------------------------------------------
# Console pipeline record detection (AMDGPU ELF & Xbox Scarlett ACCC)
# ---------------------------------------------------------------------------


def _detect_pipeline_records_in_window(
    window: bytes,
    *,
    window_file_offset: int,
    search_start: int,
    file_size: int | None,
    seen_records: set[tuple[int, str]],
    records: list[PipelineRecordHit],
) -> None:
    """Detect validated AMDGPU ELF and Xbox Scarlett ACCC pipeline records."""
    win_len = len(window)

    # 1. AMDGPU 64-bit LE ELF (`\x7fELF`, class=2, data=1, ver=1, e_machine=224)
    pos = window.find(ELF_MAGIC, search_start)
    while pos != -1:
        if pos + 64 <= win_len:
            ei_class = window[pos + 4]
            ei_data = window[pos + 5]
            ei_ver = window[pos + 6]
            if ei_class == 2 and ei_data == 1 and ei_ver == 1:
                _e_type, e_machine, e_version = struct.unpack_from("<HHI", window, pos + 16)
                if e_machine == EM_AMDGPU and e_version == 1:
                    e_flags = struct.unpack_from("<I", window, pos + 48)[0]
                    abs_off = window_file_offset + pos
                    key = (abs_off, "AMDGPU_ELF")
                    if key not in seen_records:
                        seen_records.add(key)
                        mach_id = e_flags & 0xFF
                        detail = AMDGPU_ELF_MACHINES.get(mach_id, f"amdgpu(0x{mach_id:02x})")
                        records.append(
                            PipelineRecordHit(
                                kind="AMDGPU_ELF",
                                offset=abs_off,
                                size=None,
                                detail=detail,
                            )
                        )
        pos = window.find(ELF_MAGIC, pos + 1)

    # 2. Xbox Scarlett `ACCC` precompiled RDNA2 chunk (`b"ACCC"` + u32 LE size)
    pos = window.find(ACCC_MAGIC, search_start)
    while pos != -1:
        if pos + 8 <= win_len:
            chunk_size = struct.unpack_from("<I", window, pos + 4)[0]
            abs_off = window_file_offset + pos
            size_ok = 8 <= chunk_size <= ACCC_MAX_CHUNK_BYTES
            if size_ok and file_size is not None and abs_off + 8 + chunk_size > file_size:
                size_ok = False
            if size_ok:
                probe_end = min(win_len, pos + 8 + chunk_size)
                has_endpgm = (
                    window.find(S_ENDPGM_BYTES, pos + 8, probe_end) != -1
                    or window.find(S_ENDPGM_SAVED_BYTES, pos + 8, probe_end) != -1
                )
                if has_endpgm:
                    key = (abs_off, "ACCC")
                    if key not in seen_records:
                        seen_records.add(key)
                        records.append(
                            PipelineRecordHit(
                                kind="ACCC",
                                offset=abs_off,
                                size=8 + chunk_size,
                                detail="scarlett_accc",
                            )
                        )
        pos = window.find(ACCC_MAGIC, pos + 1)


# ---------------------------------------------------------------------------
# Bounded streaming window scanner
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


def _scan_window(
    window: bytes,
    *,
    window_file_offset: int,
    search_start: int,
    file_size: int | None,
    min_dwords: int,
    max_shader_bytes: int,
    max_hits: int | None,
    seen_end_offsets: set[int],
    seen_records: set[tuple[int, str]],
    last_shader_end: int,
    shaders: list[Rdna2ShaderHit],
    records: list[PipelineRecordHit],
) -> tuple[int, bool]:
    """Scan `window` from `search_start` onward for GFX10.3 terminators & pipeline records.

    Returns `(updated_last_shader_end, hit_limit_reached)`.
    """
    if not window or search_start >= len(window):
        return last_shader_end, False

    _detect_pipeline_records_in_window(
        window,
        window_file_offset=window_file_offset,
        search_start=search_start,
        file_size=file_size,
        seen_records=seen_records,
        records=records,
    )

    if max_hits is not None and len(shaders) >= max_hits:
        return last_shader_end, True

    # Locate all `s_endpgm` and `s_endpgm_saved` terminators in sorted order.
    terminators: list[tuple[int, int]] = []
    for needle, word in (
        (S_ENDPGM_BYTES, S_ENDPGM_WORD),
        (S_ENDPGM_SAVED_BYTES, S_ENDPGM_SAVED_WORD),
    ):
        pos = window.find(needle, search_start)
        while pos != -1:
            terminators.append((pos, word))
            pos = window.find(needle, pos + 1)

    if not terminators:
        return last_shader_end, False

    terminators.sort(key=lambda item: item[0])
    hit_limit_reached = False

    for rel_end_pos, term_word in terminators:
        if max_hits is not None and len(shaders) >= max_hits:
            hit_limit_reached = True
            break

        abs_end_pos = window_file_offset + rel_end_pos
        if abs_end_pos in seen_end_offsets:
            continue
        if abs_end_pos < last_shader_end:
            # Overlaps an already emitted shader (can happen only on unaligned collisions).
            continue

        # Compute aligned backward floor within this window.
        min_abs_floor = max(window_file_offset, last_shader_end, abs_end_pos - max_shader_bytes)
        align_delta = (abs_end_pos - min_abs_floor) % 4
        if align_delta != 0:
            min_abs_floor += align_delta
        if min_abs_floor > abs_end_pos:
            min_abs_floor = abs_end_pos

        rel_floor = min_abs_floor - window_file_offset
        body_byte_len = rel_end_pos - rel_floor
        body_dword_count = body_byte_len // 4
        if body_dword_count > 0:
            body_words = struct.unpack_from(f"<{body_dword_count}I", window, rel_floor)
            start_word_idx = _find_shader_start_index(body_words)
        else:
            start_word_idx = 0

        rel_start = rel_floor + start_word_idx * 4
        abs_start = window_file_offset + rel_start
        region = window[rel_start : rel_end_pos + 4]
        dword_count = len(region) // 4

        seen_end_offsets.add(abs_end_pos)

        if dword_count < min_dwords:
            # Still advance last_shader_end so subsequent shaders do not swallow this terminator.
            last_shader_end = max(last_shader_end, abs_end_pos + 4)
            continue

        hit = analyze_shader_region(
            region,
            abs_offset=abs_start,
            terminator_word=term_word,
        )
        shaders.append(hit)
        last_shader_end = max(last_shader_end, hit.end_offset)

        if max_hits is not None and len(shaders) >= max_hits:
            hit_limit_reached = True
            break

    return last_shader_end, hit_limit_reached


def scan_bytes(
    data: bytes | bytearray | memoryview,
    *,
    source_label: str = "<memory>",
    max_bytes: int | None = None,
    max_hits: int | None = None,
    min_dwords: int = DEFAULT_MIN_DWORDS,
    max_shader_bytes: int = DEFAULT_MAX_SHADER_BYTES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
) -> FileInspectResult:
    """Scan an in-memory buffer for AMD RDNA2 / GFX10.3 shaders using bounded chunks."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes-like buffer, got {type(data).__name__}")
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0
    ):
        raise ValueError(f"max_bytes must be a positive integer or None, got {max_bytes!r}")
    if max_hits is not None and (
        isinstance(max_hits, bool) or not isinstance(max_hits, int) or max_hits <= 0
    ):
        raise ValueError(f"max_hits must be a positive integer or None, got {max_hits!r}")
    if isinstance(min_dwords, bool) or not isinstance(min_dwords, int) or min_dwords <= 0:
        raise ValueError(f"min_dwords must be a positive integer, got {min_dwords!r}")
    if (
        isinstance(max_shader_bytes, bool)
        or not isinstance(max_shader_bytes, int)
        or max_shader_bytes < 4
    ):
        raise ValueError(f"max_shader_bytes must be >= 4, got {max_shader_bytes!r}")

    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    safe_label = sanitize_source_label(source_label)

    view = memoryview(data)
    total_len = len(view)
    limit = total_len if max_bytes is None else min(total_len, max_bytes)

    shaders: list[Rdna2ShaderHit] = []
    records: list[PipelineRecordHit] = []
    seen_end_offsets: set[int] = set()
    seen_records: set[tuple[int, str]] = set()
    last_shader_end = 0
    hit_limit_reached = False

    tail = b""
    window_file_offset = 0
    prev_scanned_end = 0
    cursor = 0

    while cursor < limit:
        end = min(cursor + chunk_size, limit)
        chunk = bytes(view[cursor:end])
        cursor = end

        window = tail + chunk if tail else chunk
        if not window:
            break

        if prev_scanned_end == 0:
            search_start = 0
        else:
            # Step back 3 bytes before the previous chunk end to catch split 4-byte magics.
            search_start = max(0, (prev_scanned_end - 3) - window_file_offset)

        last_shader_end, reached = _scan_window(
            window,
            window_file_offset=window_file_offset,
            search_start=search_start,
            file_size=total_len,
            min_dwords=min_dwords,
            max_shader_bytes=max_shader_bytes,
            max_hits=max_hits,
            seen_end_offsets=seen_end_offsets,
            seen_records=seen_records,
            last_shader_end=last_shader_end,
            shaders=shaders,
            records=records,
        )
        prev_scanned_end = window_file_offset + len(window)

        if reached:
            hit_limit_reached = True
            break
        if cursor >= limit:
            break

        if overlap > 0 and len(window) > overlap:
            keep = overlap
            tail = window[-keep:]
            window_file_offset = prev_scanned_end - keep
        elif overlap > 0:
            tail = window
        else:
            tail = b""
            window_file_offset = prev_scanned_end

    bytes_scanned = prev_scanned_end
    return FileInspectResult(
        path=safe_label,
        file_size=total_len,
        bytes_scanned=bytes_scanned,
        truncated=total_len > bytes_scanned,
        hit_limit_reached=hit_limit_reached,
        unaligned_tail_bytes=bytes_scanned % 4,
        shaders=tuple(shaders),
        pipeline_records=tuple(records),
    )


def scan_file(
    path: str | Path,
    *,
    base_dir: Path | None = None,
    max_bytes: int | None = None,
    max_hits: int | None = None,
    min_dwords: int = DEFAULT_MIN_DWORDS,
    max_shader_bytes: int = DEFAULT_MAX_SHADER_BYTES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
) -> FileInspectResult:
    """Stream-scan a single file on disk in bounded memory (`chunk_size + overlap`)."""
    file_path = Path(path)
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0
    ):
        raise ValueError(f"max_bytes must be a positive integer or None, got {max_bytes!r}")
    if max_hits is not None and (
        isinstance(max_hits, bool) or not isinstance(max_hits, int) or max_hits <= 0
    ):
        raise ValueError(f"max_hits must be a positive integer or None, got {max_hits!r}")
    if isinstance(min_dwords, bool) or not isinstance(min_dwords, int) or min_dwords <= 0:
        raise ValueError(f"min_dwords must be a positive integer, got {min_dwords!r}")
    if (
        isinstance(max_shader_bytes, bool)
        or not isinstance(max_shader_bytes, int)
        or max_shader_bytes < 4
    ):
        raise ValueError(f"max_shader_bytes must be >= 4, got {max_shader_bytes!r}")

    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    rel_label = sanitize_relative_path(file_path, base_dir)

    try:
        file_size = file_path.stat().st_size
    except OSError as exc:
        return FileInspectResult(
            path=rel_label,
            file_size=0,
            bytes_scanned=0,
            truncated=False,
            error=sanitize_error_message(file_path, exc),
        )

    shaders: list[Rdna2ShaderHit] = []
    records: list[PipelineRecordHit] = []
    seen_end_offsets: set[int] = set()
    seen_records: set[tuple[int, str]] = set()
    last_shader_end = 0
    hit_limit_reached = False

    bytes_read = 0
    tail = b""
    window_file_offset = 0
    prev_scanned_end = 0

    try:
        with file_path.open("rb") as fp:
            while True:
                if max_bytes is not None:
                    remaining = max_bytes - bytes_read
                    if remaining <= 0:
                        break
                    to_read = min(chunk_size, remaining)
                else:
                    to_read = chunk_size

                chunk = fp.read(to_read)
                if not chunk:
                    break
                bytes_read += len(chunk)

                window = tail + chunk if tail else chunk
                if prev_scanned_end == 0:
                    search_start = 0
                else:
                    search_start = max(0, (prev_scanned_end - 3) - window_file_offset)

                last_shader_end, reached = _scan_window(
                    window,
                    window_file_offset=window_file_offset,
                    search_start=search_start,
                    file_size=file_size,
                    min_dwords=min_dwords,
                    max_shader_bytes=max_shader_bytes,
                    max_hits=max_hits,
                    seen_end_offsets=seen_end_offsets,
                    seen_records=seen_records,
                    last_shader_end=last_shader_end,
                    shaders=shaders,
                    records=records,
                )
                prev_scanned_end = window_file_offset + len(window)

                if reached:
                    hit_limit_reached = True
                    break

                reached_byte_limit = max_bytes is not None and bytes_read >= max_bytes
                eof = len(chunk) < to_read
                if eof or reached_byte_limit:
                    break

                if overlap > 0 and len(window) > overlap:
                    tail = window[-overlap:]
                    window_file_offset = prev_scanned_end - overlap
                elif overlap > 0:
                    tail = window
                else:
                    tail = b""
                    window_file_offset = prev_scanned_end
    except OSError as exc:
        return FileInspectResult(
            path=rel_label,
            file_size=file_size,
            bytes_scanned=bytes_read,
            truncated=file_size > bytes_read,
            hit_limit_reached=hit_limit_reached,
            unaligned_tail_bytes=bytes_read % 4,
            shaders=tuple(shaders),
            pipeline_records=tuple(records),
            error=sanitize_error_message(file_path, exc),
        )

    return FileInspectResult(
        path=rel_label,
        file_size=file_size,
        bytes_scanned=bytes_read,
        truncated=file_size > bytes_read,
        hit_limit_reached=hit_limit_reached,
        unaligned_tail_bytes=bytes_read % 4,
        shaders=tuple(shaders),
        pipeline_records=tuple(records),
    )


def _build_report(
    *,
    target_label: str,
    target_kind: str,
    max_bytes: int | None,
    max_hits: int | None,
    min_dwords: int,
    all_results: Sequence[FileInspectResult],
    hit_limit_reached: bool,
    include_empty: bool,
) -> InspectReport:
    stage_counts: Counter[str] = Counter()
    terminator_counts: Counter[str] = Counter()
    export_target_histogram: Counter[str] = Counter()
    size_hist_raw: Counter[str] = Counter()
    instruction_totals: dict[str, int] = {
        "exp": 0,
        "smem": 0,
        "vintrp": 0,
        "mimg": 0,
        "valu": 0,
        "salu": 0,
        "sopp": 0,
        "vmem": 0,
        "ds": 0,
        "flat": 0,
    }
    pipeline_record_counts: Counter[str] = Counter()
    unique_hashes: set[int] = set()

    files_scanned = 0
    files_with_shaders = 0
    files_errored = 0
    total_file_bytes = 0
    total_bytes_scanned = 0
    total_shaders = 0
    total_pipeline_records = 0
    reported_files: list[FileInspectResult] = []

    for res in all_results:
        files_scanned += 1
        total_file_bytes += res.file_size
        total_bytes_scanned += res.bytes_scanned

        if res.error is not None:
            files_errored += 1

        if res.shaders:
            files_with_shaders += 1
            total_shaders += len(res.shaders)
            for shader in res.shaders:
                unique_hashes.add(shader.fnv1a_64)
                stage_counts[shader.stage] += 1
                terminator_counts[shader.terminator] += 1
                size_hist_raw[size_histogram_bucket(shader.byte_size)] += 1
                for exp in shader.exports:
                    export_target_histogram[exp.target] += 1
                instruction_totals["exp"] += shader.exp_count
                instruction_totals["smem"] += shader.smem_count
                instruction_totals["vintrp"] += shader.vintrp_count
                instruction_totals["mimg"] += shader.mimg_count
                instruction_totals["valu"] += shader.valu_count
                instruction_totals["salu"] += shader.salu_count
                instruction_totals["sopp"] += shader.sopp_count
                instruction_totals["vmem"] += shader.vmem_count
                instruction_totals["ds"] += shader.ds_count
                instruction_totals["flat"] += shader.flat_count

        if res.pipeline_records:
            total_pipeline_records += len(res.pipeline_records)
            for rec in res.pipeline_records:
                pipeline_record_counts[rec.kind] += 1

        if (
            include_empty
            or target_kind == "file"
            or res.shaders
            or res.pipeline_records
            or res.error is not None
        ):
            reported_files.append(res)

    # Preserve canonical bucket order for non-zero size buckets.
    ordered_size_hist: dict[str, int] = {
        label: size_hist_raw[label]
        for _, label in SIZE_HISTOGRAM_BUCKETS
        if size_hist_raw[label] > 0
    }

    return InspectReport(
        schema_version=SCHEMA_VERSION,
        target_label=target_label,
        target_kind=target_kind,
        max_bytes=max_bytes,
        max_hits=max_hits,
        min_dwords=min_dwords,
        files_scanned=files_scanned,
        files_with_shaders=files_with_shaders,
        files_errored=files_errored,
        total_file_bytes=total_file_bytes,
        total_bytes_scanned=total_bytes_scanned,
        total_shaders=total_shaders,
        unique_shader_hashes=len(unique_hashes),
        total_pipeline_records=total_pipeline_records,
        hit_limit_reached=hit_limit_reached,
        stage_counts=dict(sorted(stage_counts.items())),
        terminator_counts=dict(sorted(terminator_counts.items())),
        export_target_histogram=dict(sorted(export_target_histogram.items())),
        size_histogram=ordered_size_hist,
        instruction_totals=instruction_totals,
        pipeline_record_counts=dict(sorted(pipeline_record_counts.items())),
        files=tuple(reported_files),
    )


def scan_directory(
    root_dir: str | Path,
    *,
    max_bytes: int | None = None,
    max_hits: int | None = None,
    min_dwords: int = DEFAULT_MIN_DWORDS,
    max_shader_bytes: int = DEFAULT_MAX_SHADER_BYTES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
    target_label: str | None = None,
    include_empty: bool = False,
) -> InspectReport:
    """Recursively scan a directory for RDNA2 / GFX10.3 shaders in deterministic order."""
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {sanitize_source_label(root)}")

    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    label = sanitize_source_label(target_label if target_label is not None else root)

    discovered: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames.sort()
        base = Path(dirpath)
        for fname in filenames:
            discovered.append(base / fname)
    discovered.sort(key=lambda p: sanitize_relative_path(p, root))

    all_results: list[FileInspectResult] = []
    total_hits = 0
    hit_limit_reached = False

    for fpath in discovered:
        if max_hits is not None and total_hits >= max_hits:
            hit_limit_reached = True
            break
        remaining_hits = None if max_hits is None else (max_hits - total_hits)
        res = scan_file(
            fpath,
            base_dir=root,
            max_bytes=max_bytes,
            max_hits=remaining_hits,
            min_dwords=min_dwords,
            max_shader_bytes=max_shader_bytes,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_results.append(res)
        total_hits += len(res.shaders)
        if res.hit_limit_reached:
            hit_limit_reached = True
            break

    return _build_report(
        target_label=label,
        target_kind="directory",
        max_bytes=max_bytes,
        max_hits=max_hits,
        min_dwords=min_dwords,
        all_results=all_results,
        hit_limit_reached=hit_limit_reached,
        include_empty=include_empty,
    )


def scan_path(
    target: str | Path,
    *,
    max_bytes: int | None = None,
    max_hits: int | None = None,
    min_dwords: int = DEFAULT_MIN_DWORDS,
    max_shader_bytes: int = DEFAULT_MAX_SHADER_BYTES,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP_BYTES,
    target_label: str | None = None,
    include_empty: bool = False,
) -> InspectReport:
    """Scan either a single file or a directory and return a unified `InspectReport`."""
    target_path = Path(target)
    if target_path.is_dir():
        return scan_directory(
            target_path,
            max_bytes=max_bytes,
            max_hits=max_hits,
            min_dwords=min_dwords,
            max_shader_bytes=max_shader_bytes,
            chunk_size=chunk_size,
            overlap=overlap,
            target_label=target_label,
            include_empty=include_empty,
        )
    if target_path.is_file():
        label = sanitize_source_label(target_label if target_label is not None else target_path)
        res = scan_file(
            target_path,
            max_bytes=max_bytes,
            max_hits=max_hits,
            min_dwords=min_dwords,
            max_shader_bytes=max_shader_bytes,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        return _build_report(
            target_label=label,
            target_kind="file",
            max_bytes=max_bytes,
            max_hits=max_hits,
            min_dwords=min_dwords,
            all_results=(res,),
            hit_limit_reached=res.hit_limit_reached,
            include_empty=True,
        )
    raise FileNotFoundError(f"target does not exist: {sanitize_source_label(target_path)}")


# ---------------------------------------------------------------------------
# Output rendering (strictly sanitized Markdown & deterministic JSON)
# ---------------------------------------------------------------------------


def _escape_md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_json(report: InspectReport) -> str:
    """Render a strictly sanitized deterministic JSON report (zero raw shader bytes)."""
    return json.dumps(report.to_dict(), indent=2) + "\n"


def render_markdown(report: InspectReport) -> str:
    """Render a strictly sanitized Markdown summary report (zero raw shader bytes)."""
    lines: list[str] = [
        f"# RDNA2 / GFX10.3 Shader Inspection Report: `{_escape_md_cell(report.target_label)}`",
        "",
        "| Metric | Value |",
        "| :--- | ---: |",
        f"| Target kind | `{report.target_kind}` |",
        f"| Files scanned | {report.files_scanned} |",
        f"| Files with shaders | {report.files_with_shaders} |",
        f"| Files errored | {report.files_errored} |",
        f"| Total bytes scanned | {report.total_bytes_scanned} / {report.total_file_bytes} |",
        f"| Total shaders | {report.total_shaders} |",
        f"| Unique shader hashes (FNV-1a 64) | {report.unique_shader_hashes} |",
        f"| Pipeline records | {report.total_pipeline_records} |",
        f"| Min dwords | {report.min_dwords} |",
        f"| Hit limit reached | `{str(report.hit_limit_reached).lower()}` |",
        "",
    ]

    if report.stage_counts:
        lines.extend(["## Stage Heuristics", "", "| Stage | Shaders |", "| :--- | ---: |"])
        for stage, count in report.stage_counts.items():
            lines.append(f"| `{_escape_md_cell(stage)}` | {count} |")
        lines.append("")

    if report.export_target_histogram:
        lines.extend(["## Export Target Histogram", "", "| Target | Count |", "| :--- | ---: |"])
        for target, count in report.export_target_histogram.items():
            lines.append(f"| `{_escape_md_cell(target)}` | {count} |")
        lines.append("")

    if report.size_histogram:
        lines.extend(["## Shader Size Histogram", "", "| Size Bucket | Shaders |", "| :--- | ---: |"])
        for bucket, count in report.size_histogram.items():
            lines.append(f"| `{_escape_md_cell(bucket)}` | {count} |")
        lines.append("")

    if report.pipeline_record_counts:
        lines.extend(["## Console Pipeline Records", "", "| Kind | Count |", "| :--- | ---: |"])
        for kind, count in report.pipeline_record_counts.items():
            lines.append(f"| `{_escape_md_cell(kind)}` | {count} |")
        lines.append("")

    matched = [f for f in report.files if f.shaders or f.pipeline_records or f.error]
    if matched:
        lines.extend(
            [
                "## Matched Files",
                "",
                "| File | Scanned | Shaders | Stages | Status |",
                "| :--- | ---: | ---: | :--- | :--- |",
            ]
        )
        for entry in matched:
            stages = ", ".join(f"{k}:{v}" for k, v in entry.stage_counts.items()) or "-"
            status = (
                f"error: {entry.error}"
                if entry.error
                else (
                    "hit_limit"
                    if entry.hit_limit_reached
                    else ("truncated" if entry.truncated else "ok")
                )
            )
            lines.append(
                f"| `{_escape_md_cell(entry.path)}` | {entry.bytes_scanned}/{entry.file_size} "
                f"| {len(entry.shaders)} | `{_escape_md_cell(stages)}` | {_escape_md_cell(status)} |"
            )
        lines.append("")
    else:
        lines.append("No RDNA2 / GFX10.3 shaders detected.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only AMD RDNA2 / GFX10.3 shader binary & pipeline record scanner "
            "(strictly sanitized metadata only; zero raw shader bytes or host paths)."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        metavar="PATH",
        help="Local file or directory to inspect (mutually exclusive with --from-env).",
    )
    parser.add_argument(
        "--from-env",
        action="store_true",
        help="Scan GTA6_GAME_DIR resolved from environment or .env.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit strictly sanitized deterministic JSON.",
    )
    parser.add_argument(
        "--max-hits",
        metavar="N",
        type=parse_positive_int,
        default=None,
        help="Stop after detecting at most N shaders.",
    )
    parser.add_argument(
        "--min-dwords",
        metavar="N",
        type=parse_positive_int,
        default=DEFAULT_MIN_DWORDS,
        help=f"Minimum shader length in 32-bit dwords including terminator (default: {DEFAULT_MIN_DWORDS}).",
    )
    parser.add_argument(
        "--max-bytes",
        metavar="N",
        type=parse_byte_size,
        default=None,
        help="Maximum bytes to scan per file (e.g. 4194304 or 4M) for bounded sampling.",
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

    if args.from_env and args.path is not None:
        parser.error("cannot specify both positional PATH and --from-env")
    if not args.from_env and args.path is None:
        parser.error("either PATH or --from-env is required")

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
            max_bytes=args.max_bytes,
            max_hits=args.max_hits,
            min_dwords=args.min_dwords,
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
