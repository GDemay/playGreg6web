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

# Standalone native RDNA2 stage dump files (`playgta6_boot --dump-pipelines` or raw blobs)
STANDALONE_SHADER_EXTENSIONS: frozenset[str] = frozenset({".bin", ".rdna2", ".shader"})
PIPELINE_DUMP_SCHEMA = "playgta6-pipeline-dump"
PIPELINE_DUMP_SCHEMA_VERSION = 1
PIPELINE_DUMP_MANIFEST_MAX_BYTES = 16 * 1024 * 1024  # 16 MiB safety cap
STANDALONE_MANIFEST_MAX_SHADER_BYTES = 4 * 1024 * 1024  # 4 MiB cap for manifest-listed stages

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

# Canonical GFX10.3 / RDNA2 shader stage ordering (T92)
CANONICAL_STAGE_ORDER: tuple[str, ...] = ("vertex", "pixel", "compute", "unknown")
_CANONICAL_STAGE_RANK: dict[str, int] = {
    stage: idx for idx, stage in enumerate(CANONICAL_STAGE_ORDER)
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

# Sanitization patterns (matching tools/scan_shaders.py, tools/extract_dxil_corpus.py & T83/T92)
_FILE_URI_RE = re.compile(r"file://[^\s\"'<>|,;()\[\]]+", re.IGNORECASE)
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_MSYS_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/[A-Za-z]/(?:Users|users|home|Windows|Program|tmp|var|opt|mnt)/[^\s\"'<>|,;()\[\]]*"
)
_POSIX_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_])/(?:Users|users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes|srv|run|media|cygdrive)/[^\s\"'<>|,;()\[\]]*"
)
_TILDE_HOME_RE = re.compile(r"(?<![A-Za-z0-9_])~[A-Za-z0-9._-]*/[^\s\"'<>|,;()\[\]]*")
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


def _parse_exact_hex_u64(token: str) -> int | None:
    """Parse an exact 16-hex-digit 64-bit hash token (with optional `0x` prefix)."""
    raw = token.strip()
    if raw.lower().startswith("0x"):
        raw = raw[2:]
    if len(raw) != 16 or not all(ch in "0123456789abcdefABCDEF" for ch in raw):
        return None
    return int(raw, 16)


def _parse_stage_token(token: str) -> str | None:
    """Parse a stage name/prefix token into canonical S15 stage (`vertex`/`pixel`/`compute`)."""
    low = token.strip().lower()
    if low in ("vs", "vertex", "vert"):
        return "vertex"
    if low in ("ps", "fs", "pixel", "fragment", "frag"):
        return "pixel"
    if low in ("cs", "compute", "comp"):
        return "compute"
    return None


@dataclass(frozen=True)
class FilenameIdentityHint:
    """Optional stage and 64-bit FNV-1a hash hints parsed from a standalone shader filename."""

    stage: str | None = None
    hash_fnv1a64: int | None = None


@dataclass(frozen=True)
class ManifestStageHint:
    """Validated `playgta6-pipeline-dump` v1 `native_stages` entry for a native `.bin` file."""

    code_identity: int
    stage_identities: tuple[int, ...]
    stage: str | None
    code_bytes: int | None
    code_file: str


def parse_filename_identity_hint(source: str | Path) -> FilenameIdentityHint:
    """Parse optional stage and 16-hex-digit 64-bit identity hash hints from a filename.

    Matches `shaders/src/rdna2_pack.rs` (`parse_filename_hint`) and D3D12 guest
    `native/0x<16hex>.bin` naming:
      - `0x<16hex>.bin`, `<16hex>.bin`, `0x<16hex>.rdna2`
      - `vs_<16hex>.rdna2`, `ps_0x<16hex>.bin`, `fs_<16hex>.shader`, `cs_<16hex>.bin`
      - `<16hex>_ps.rdna2`, `vs_splash_<16hex>.bin`
    """
    basename = sanitize_source_label(source)
    if not basename or basename.startswith("<"):
        return FilenameIdentityHint()
    if "." in basename:
        stem, _ext = basename.rsplit(".", 1)
        if not stem:
            return FilenameIdentityHint()
    else:
        stem = basename

    parts = [part for part in re.split(r"[_-]+", stem) if part]
    if not parts:
        return FilenameIdentityHint()

    stage: str | None = None
    remaining = parts
    first_stage = _parse_stage_token(parts[0])
    if first_stage is not None:
        stage = first_stage
        remaining = parts[1:]
    else:
        last_stage = _parse_stage_token(parts[-1])
        if last_stage is not None:
            stage = last_stage
            remaining = parts[:-1]

    hash_hint: int | None = None
    if len(remaining) == 1:
        hash_hint = _parse_exact_hex_u64(remaining[0])
    elif len(remaining) > 1:
        hash_hint = _parse_exact_hex_u64(remaining[-1])
        if hash_hint is None:
            hash_hint = _parse_exact_hex_u64(remaining[0])

    return FilenameIdentityHint(stage=stage, hash_fnv1a64=hash_hint)


def _normalize_manifest_rel_path(raw: str) -> str | None:
    """Normalize a manifest relative `code_file` path to canonical POSIX form."""
    cleaned = raw.strip().replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.lstrip("/")
    if not cleaned:
        return None
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


def load_pipeline_dump_manifest(manifest_path: Path) -> dict[str, ManifestStageHint]:
    """Read an optional `playgta6-pipeline-dump` v1 `manifest.json` (strictly read-only).

    Returns a mapping from normalized relative `code_file` (e.g. `native/0x<16hex>.bin`)
    to `ManifestStageHint`, or `{}` if the file is absent, invalid, or another schema.
    """
    try:
        if not manifest_path.is_file():
            return {}
        if manifest_path.stat().st_size > PIPELINE_DUMP_MANIFEST_MAX_BYTES:
            return {}
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return {}

    if not isinstance(payload, Mapping):
        return {}
    if payload.get("schema") != PIPELINE_DUMP_SCHEMA:
        return {}
    if payload.get("schema_version") != PIPELINE_DUMP_SCHEMA_VERSION:
        return {}

    native_stages = payload.get("native_stages")
    if not isinstance(native_stages, list):
        return {}

    grouped: dict[str, dict[str, Any]] = {}
    for entry in native_stages:
        if not isinstance(entry, Mapping):
            continue
        raw_code_file = entry.get("code_file")
        raw_code_id = entry.get("code_identity")
        if not isinstance(raw_code_file, str) or not isinstance(raw_code_id, (str, int)):
            continue
        rel_file = _normalize_manifest_rel_path(raw_code_file)
        if rel_file is None:
            continue
        if isinstance(raw_code_id, bool):
            continue
        code_id = (
            raw_code_id
            if isinstance(raw_code_id, int) and 0 <= raw_code_id <= U64_MAX
            else _parse_exact_hex_u64(str(raw_code_id))
        )
        if code_id is None:
            continue

        raw_stage_id = entry.get("identity")
        stage_id: int | None = None
        if isinstance(raw_stage_id, int) and not isinstance(raw_stage_id, bool) and 0 <= raw_stage_id <= U64_MAX:
            stage_id = raw_stage_id
        elif isinstance(raw_stage_id, str):
            stage_id = _parse_exact_hex_u64(raw_stage_id)

        raw_stage = entry.get("stage")
        stage = _parse_stage_token(raw_stage) if isinstance(raw_stage, str) else None

        raw_code_bytes = entry.get("code_bytes")
        code_bytes = (
            raw_code_bytes
            if isinstance(raw_code_bytes, int) and not isinstance(raw_code_bytes, bool) and raw_code_bytes >= 0
            else None
        )

        bucket = grouped.get(rel_file)
        if bucket is None:
            grouped[rel_file] = {
                "code_identity": code_id,
                "stage_identities": [stage_id] if stage_id is not None else [],
                "stage": stage,
                "code_bytes": code_bytes,
                "code_file": rel_file,
            }
        else:
            if stage_id is not None and stage_id not in bucket["stage_identities"]:
                bucket["stage_identities"].append(stage_id)
            if bucket["stage"] is None and stage is not None:
                bucket["stage"] = stage
            if bucket["code_bytes"] is None and code_bytes is not None:
                bucket["code_bytes"] = code_bytes

    result: dict[str, ManifestStageHint] = {}
    for rel_file, bucket in grouped.items():
        result[rel_file] = ManifestStageHint(
            code_identity=bucket["code_identity"],
            stage_identities=tuple(sorted(bucket["stage_identities"])),
            stage=bucket["stage"],
            code_bytes=bucket["code_bytes"],
            code_file=rel_file,
        )
    return result


def _find_manifest_for_directory(root: Path) -> tuple[Path, dict[str, ManifestStageHint]] | None:
    """Locate a `playgta6-pipeline-dump` v1 manifest in `root` or `root.parent`."""
    for candidate_dir in (root, root.parent):
        manifest_file = candidate_dir / "manifest.json"
        if manifest_file.is_file():
            entries = load_pipeline_dump_manifest(manifest_file)
            if entries:
                return candidate_dir.resolve(), entries
    return None


def _lookup_manifest_hint(
    file_path: Path,
    manifest_ctx: tuple[Path, Mapping[str, ManifestStageHint]] | None,
) -> ManifestStageHint | None:
    """Resolve a `ManifestStageHint` for `file_path` given `(manifest_dir, manifest_map)`."""
    if manifest_ctx is None:
        return None
    manifest_dir, manifest_map = manifest_ctx
    try:
        rel = file_path.resolve().relative_to(manifest_dir).as_posix()
    except (ValueError, OSError):
        return None
    return manifest_map.get(rel)


def _find_manifest_hint_for_file(file_path: Path) -> ManifestStageHint | None:
    """Check `file_path.parent` and `file_path.parent.parent` for a dump manifest."""
    for candidate_dir in (file_path.parent, file_path.parent.parent):
        manifest_file = candidate_dir / "manifest.json"
        if manifest_file.is_file():
            entries = load_pipeline_dump_manifest(manifest_file)
            if entries:
                hit = _lookup_manifest_hint(file_path, (candidate_dir.resolve(), entries))
                if hit is not None:
                    return hit
    return None


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


def _scrub_sensitive_text(text: str) -> str:
    """Remove host absolute paths (Windows, UNC, POSIX, MSYS, file://, tilde) and active OS usernames."""
    sanitized = _FILE_URI_RE.sub("<path>", text)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _WINDOWS_PATH_RE.sub("<path>", sanitized)
    sanitized = _UNC_FS_PATH_RE.sub("<path>", sanitized)
    sanitized = _MSYS_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    sanitized = _TILDE_HOME_RE.sub("<path>", sanitized)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(username)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        sanitized = pattern.sub("<user>", sanitized)
    return sanitized


def sanitize_text(raw: str | None) -> str:
    """Public text sanitizer stripping host absolute paths and active OS usernames."""
    if raw is None:
        return ""
    return _scrub_sensitive_text(str(raw))


def _is_user_home_leaf(parts: Sequence[str]) -> bool:
    """Return True if path segments terminate at a host user-home directory leaf."""
    if len(parts) < 2:
        return False
    if parts[-2].lower() not in ("users", "home"):
        return False
    if len(parts) == 2:
        return True
    if len(parts) == 3:
        first = parts[0]
        if not first:
            return True
        if len(first) == 2 and first[1] == ":" and first[0].isalpha():
            return True
        if len(first) == 1 and first.isalpha():
            return True
    return False


def _is_host_absolute_path(raw: str) -> bool:
    """Return True if `raw` looks like a host-local absolute path, URI, UNC, or tilde path."""
    stripped = raw.strip()
    if not stripped:
        return False
    if stripped.lower().startswith("file://"):
        return True
    if stripped.startswith(("\\\\", "//", "/", "~")):
        return True
    if len(stripped) >= 2 and stripped[0].isalpha() and stripped[1] == ":":
        return True
    return False


def sanitize_source_label(source: str | Path | None) -> str:
    """Return a strictly sanitized basename-only label (no host paths or usernames)."""
    if source is None:
        return "<memory>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    if raw.startswith("<") and raw.endswith(">"):
        return _scrub_sensitive_text(raw)
    if raw.lower().startswith("file://"):
        raw = raw[7:]
        if raw.lower().startswith("localhost/"):
            raw = raw[len("localhost") :]
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    if stripped.startswith("~") and "/" not in stripped and "\\" not in stripped:
        return "<user>"
    parts = [p for p in re.split(r"[/\\]+", stripped) if p]
    if not parts:
        return "<unknown>"
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    if _is_user_home_leaf(parts):
        return "<user>"
    return _scrub_sensitive_text(base)


def _sanitize_report_path(path: str | Path | None) -> str:
    """Sanitize a report file path: preserve clean POSIX relative paths, strip host absolute paths."""
    if path is None:
        return "<memory>"
    raw = str(path).strip()
    if not raw:
        return "<unknown>"
    if raw.startswith("<") and raw.endswith(">"):
        return _scrub_sensitive_text(raw)
    if _is_host_absolute_path(raw):
        return sanitize_source_label(raw)
    cleaned = raw.replace("\\", "/")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    cleaned = cleaned.strip("/")
    if not cleaned:
        return "<unknown>"
    return _scrub_sensitive_text(cleaned)


def sanitize_json_value(value: Any) -> Any:
    """Recursively sanitize JSON values, stripping host paths, usernames, PathLike, and raw bytes."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, os.PathLike):
        return sanitize_source_label(os.fspath(value))
    if isinstance(value, str):
        return _scrub_sensitive_text(value)
    if isinstance(value, Mapping):
        return {
            _scrub_sensitive_text(str(k)): sanitize_json_value(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]
    return value


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
    hash_scope: str = "region"  # "whole_file" | "region"
    trimmed_hash_fnv1a64: int | None = None
    trimmed_byte_size: int | None = None
    terminator_count: int = 1
    filename_hash_hint: int | None = None
    filename_hash_matches: bool | None = None
    manifest_code_identity: int | None = None
    manifest_identity: int | None = None
    manifest_stage: str | None = None
    manifest_identity_matches: bool | None = None

    def __post_init__(self) -> None:
        if self.trimmed_hash_fnv1a64 is None:
            object.__setattr__(self, "trimmed_hash_fnv1a64", self.fnv1a_64)
        if self.trimmed_byte_size is None:
            object.__setattr__(self, "trimmed_byte_size", self.byte_size)

    @property
    def code_hash_fnv1a64(self) -> int:
        """Canonical FNV-1a 64-bit code identity (whole-file for standalone dumps, region otherwise)."""
        return self.fnv1a_64

    @property
    def hash_hex(self) -> str:
        return format_hash_64(self.fnv1a_64)

    @property
    def code_hash_hex(self) -> str:
        return format_hash_64(self.fnv1a_64)

    @property
    def trimmed_hash_hex(self) -> str:
        assert self.trimmed_hash_fnv1a64 is not None
        return format_hash_64(self.trimmed_hash_fnv1a64)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hash": self.hash_hex,
            "code_hash_fnv1a64": self.code_hash_hex,
            "trimmed_hash_fnv1a64": self.trimmed_hash_hex,
            "hash_scope": self.hash_scope,
            "offset": self.offset,
            "end_offset": self.end_offset,
            "byte_size": self.byte_size,
            "trimmed_byte_size": self.trimmed_byte_size,
            "dword_count": self.dword_count,
            "instruction_count": self.instruction_count,
            "stage": self.stage,
            "terminator": self.terminator,
            "terminator_count": self.terminator_count,
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
        if self.filename_hash_hint is not None:
            payload["filename_hash_hint"] = format_hash_64(self.filename_hash_hint)
            payload["filename_hash_matches"] = self.filename_hash_matches
        if self.manifest_code_identity is not None:
            payload["manifest_code_identity"] = format_hash_64(self.manifest_code_identity)
            payload["manifest_identity_matches"] = self.manifest_identity_matches
        if self.manifest_identity is not None:
            payload["manifest_identity"] = format_hash_64(self.manifest_identity)
        if self.manifest_stage is not None:
            payload["manifest_stage"] = self.manifest_stage
        return payload


def _stage_sort_key(stage: str) -> tuple[int, str]:
    """Deterministic canonical stage sort key (`vertex`, `pixel`, `compute`, `unknown`, then alpha)."""
    return (_CANONICAL_STAGE_RANK.get(stage, len(CANONICAL_STAGE_ORDER)), stage)


def _shader_hit_sort_key(hit: Rdna2ShaderHit) -> tuple[int, int, int, tuple[int, str]]:
    """Deterministic sort key for shaders within a file."""
    return (hit.offset, hit.end_offset, hit.fnv1a_64, _stage_sort_key(hit.stage))


def _pipeline_record_sort_key(rec: PipelineRecordHit) -> tuple[int, str, int, str]:
    """Deterministic sort key for pipeline records within a file."""
    return (rec.offset, rec.kind, rec.size if rec.size is not None else -1, rec.detail or "")


def _non_negative_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(value, 0)


def compute_stage_breakdown(
    stage_counts: Mapping[str, int],
    *,
    total_shaders: int | None = None,
) -> list[dict[str, Any]]:
    """Compute canonical, zero-division-safe stage breakdown counts and percentages (T92).

    Always includes the canonical RDNA2 stages (`vertex`, `pixel`, `compute`, `unknown`)
    in canonical order, followed by any additional stage labels sorted alphabetically.
    When total shaders is 0 (or <= 0), all stage percentages are safely `0.0`.
    """
    normalized: dict[str, int] = {stage: 0 for stage in CANONICAL_STAGE_ORDER}
    for raw_stage, raw_count in stage_counts.items():
        stage = _scrub_sensitive_text(str(raw_stage))
        normalized[stage] = normalized.get(stage, 0) + _non_negative_int(raw_count)

    total = (
        sum(normalized.values())
        if total_shaders is None
        else _non_negative_int(total_shaders)
    )
    breakdown: list[dict[str, Any]] = []
    for stage in sorted(normalized.keys(), key=_stage_sort_key):
        count = normalized[stage]
        pct = round((count / total) * 100.0, 2) if total > 0 else 0.0
        breakdown.append(
            {
                "stage": stage,
                "count": count,
                "percentage": pct,
            }
        )
    return breakdown


def compute_stage_percentages(
    stage_counts: Mapping[str, int],
    *,
    total_shaders: int | None = None,
) -> dict[str, float]:
    """Return a canonical stage-ordered mapping of stage -> zero-division-safe percentage (T92)."""
    return {
        entry["stage"]: entry["percentage"]
        for entry in compute_stage_breakdown(stage_counts, total_shaders=total_shaders)
    }


def build_canonical_shader_summary(
    results: Sequence[FileInspectResult],
) -> list[dict[str, Any]]:
    """Build a deterministically sorted, strictly sanitized canonical shader summary (T92).

    Groups detected shaders by canonical `(stage, fnv1a_64)` identity, aggregates
    occurrence counts and sorted sanitized source paths, and sorts deterministically by
    `(canonical_stage_rank, stage, hash_hex)`.
    """
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for res in results:
        safe_path = _sanitize_report_path(res.path)
        for shader in res.shaders:
            stage = _scrub_sensitive_text(shader.stage)
            key = (stage, shader.fnv1a_64)
            entry = grouped.get(key)
            if entry is None:
                grouped[key] = {
                    "hash": shader.hash_hex,
                    "stage": stage,
                    "byte_size": shader.byte_size,
                    "dword_count": shader.dword_count,
                    "instruction_count": shader.instruction_count,
                    "terminator": _scrub_sensitive_text(shader.terminator),
                    "export_targets": sorted(
                        {_scrub_sensitive_text(t) for t in shader.export_targets}
                    ),
                    "occurrences": 1,
                    "paths": {safe_path},
                }
            else:
                entry["occurrences"] += 1
                entry["paths"].add(safe_path)
                if shader.byte_size < entry["byte_size"]:
                    entry["byte_size"] = shader.byte_size
                    entry["dword_count"] = shader.dword_count
                    entry["instruction_count"] = shader.instruction_count
                entry["export_targets"] = sorted(
                    set(entry["export_targets"])
                    | {_scrub_sensitive_text(t) for t in shader.export_targets}
                )

    summary_rows: list[dict[str, Any]] = []
    for (_stage, _digest), entry in sorted(
        grouped.items(),
        key=lambda item: (_stage_sort_key(item[0][0]), format_hash_64(item[0][1])),
    ):
        summary_rows.append(
            {
                "hash": entry["hash"],
                "stage": entry["stage"],
                "byte_size": entry["byte_size"],
                "dword_count": entry["dword_count"],
                "instruction_count": entry["instruction_count"],
                "terminator": entry["terminator"],
                "export_targets": list(entry["export_targets"]),
                "occurrences": entry["occurrences"],
                "paths": sorted(entry["paths"]),
            }
        )
    return summary_rows


@dataclass(frozen=True)
class PipelineRecordHit:
    """Console / AMDGPU pipeline container record detected during scanning."""

    kind: str
    offset: int
    size: int | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": _scrub_sensitive_text(self.kind),
            "offset": self.offset,
            "size": self.size,
        }
        if self.detail is not None:
            payload["detail"] = _scrub_sensitive_text(self.detail)
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
    hash_scope: str = "region"  # "whole_file" | "region"
    file_hash_fnv1a64: int | None = None

    @property
    def stage_counts(self) -> dict[str, int]:
        counts = Counter(s.stage for s in self.shaders)
        return {stage: counts[stage] for stage in sorted(counts.keys(), key=_stage_sort_key)}

    @property
    def stage_percentages(self) -> dict[str, float]:
        return compute_stage_percentages(self.stage_counts, total_shaders=len(self.shaders))

    @property
    def export_target_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for shader in self.shaders:
            for exp in shader.exports:
                counts[exp.target] += 1
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, Any]:
        sorted_shaders = sorted(self.shaders, key=_shader_hit_sort_key)
        sorted_records = sorted(self.pipeline_records, key=_pipeline_record_sort_key)
        payload: dict[str, Any] = {
            "path": _sanitize_report_path(self.path),
            "file_size": self.file_size,
            "bytes_scanned": self.bytes_scanned,
            "truncated": self.truncated,
            "hit_limit_reached": self.hit_limit_reached,
            "unaligned_tail_bytes": self.unaligned_tail_bytes,
            "hash_scope": self.hash_scope,
            "shader_count": len(sorted_shaders),
            "pipeline_record_count": len(sorted_records),
            "stage_counts": self.stage_counts,
            "stage_percentages": self.stage_percentages,
            "export_target_counts": self.export_target_counts,
            "shaders": [s.to_dict() for s in sorted_shaders],
            "pipeline_records": [r.to_dict() for r in sorted_records],
        }
        if self.file_hash_fnv1a64 is not None:
            payload["file_hash_fnv1a64"] = format_hash_64(self.file_hash_fnv1a64)
        if self.error is not None:
            payload["error"] = _scrub_sensitive_text(self.error)
        return sanitize_json_value(payload)


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
    stage_breakdown: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    stage_percentages: dict[str, float] = field(default_factory=dict)
    shader_summary: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        canonical_stage_counts = {
            stage: self.stage_counts[stage]
            for stage in sorted(self.stage_counts.keys(), key=_stage_sort_key)
        }
        breakdown = (
            list(self.stage_breakdown)
            if self.stage_breakdown
            else compute_stage_breakdown(
                canonical_stage_counts,
                total_shaders=self.total_shaders,
            )
        )
        percentages = (
            {
                stage: self.stage_percentages[stage]
                for stage in sorted(self.stage_percentages.keys(), key=_stage_sort_key)
            }
            if self.stage_percentages
            else {entry["stage"]: entry["percentage"] for entry in breakdown}
        )
        sorted_files = sorted(
            self.files,
            key=lambda f: (_sanitize_report_path(f.path), f.file_size, f.bytes_scanned),
        )
        canonical_shaders = (
            list(self.shader_summary)
            if self.shader_summary
            else build_canonical_shader_summary(sorted_files)
        )
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "target": sanitize_source_label(self.target_label),
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
                "stage_counts": canonical_stage_counts,
                "stage_percentages": percentages,
                "stage_breakdown": breakdown,
                "terminator_counts": dict(sorted(self.terminator_counts.items())),
                "export_target_histogram": dict(sorted(self.export_target_histogram.items())),
                "size_histogram": dict(self.size_histogram),
                "instruction_totals": dict(self.instruction_totals),
                "pipeline_record_counts": dict(sorted(self.pipeline_record_counts.items())),
                "shader_summary": canonical_shaders,
            },
            "files": [f.to_dict() for f in sorted_files],
        }
        return sanitize_json_value(payload)


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
        hash_scope="region",
        trimmed_hash_fnv1a64=fnv1a_64(shader_bytes),
        trimmed_byte_size=len(shader_bytes),
        terminator_count=1,
    )


def _starts_with_container_magic(blob: bytes | bytearray | memoryview) -> bool:
    """Return True if `blob` starts with a known multi-shader container magic."""
    if len(blob) < 4:
        return False
    prefix = bytes(blob[:4])
    return prefix in (b"DXBC", ACCC_MAGIC, ELF_MAGIC)


def _is_standalone_multi_endpgm_stage(shaders: Sequence[Rdna2ShaderHit]) -> bool:
    """Return True if multiple `s_endpgm` hits in a small `.bin`/`.rdna2`/`.shader` file
    represent one standalone native stage (e.g. early-out `s_endpgm` + main body or stub)
    rather than a padded multi-shader bundle.
    """
    if len(shaders) < 2:
        return True
    if shaders[0].offset != 0:
        return False

    graphics_stages = {s.stage for s in shaders if s.stage in ("vertex", "pixel")}
    if len(graphics_stages) > 1:
        # Conflicting vertex + pixel stages in the same file -> multi-shader bundle.
        return False

    for idx in range(1, len(shaders)):
        prev = shaders[idx - 1]
        curr = shaders[idx]
        if curr.offset == prev.end_offset:
            # Immediately contiguous after previous `s_endpgm` (zero padding gap).
            continue
        if curr.stage == "unknown":
            # Trailing `s_nop` + `s_endpgm` epilogue/stub after padding.
            continue
        # Non-contiguous padded shader of a distinct concrete stage -> multi-shader bundle.
        return False

    return True


def analyze_standalone_shader(
    raw_bytes: bytes | bytearray | memoryview,
    *,
    min_dwords: int = DEFAULT_MIN_DWORDS,
    filename_hint: FilenameIdentityHint | None = None,
    manifest_hint: ManifestStageHint | None = None,
) -> Rdna2ShaderHit | None:
    """Analyze a standalone native RDNA2 stage file and hash its **entire** byte range.

    Matches `disasm-rdna2` (S14), `translate-rdna2` (S17), `pack-rdna2` (S18), and
    `runtime/gpu/d3d12_guest_resources.cpp`:
      - `fnv1a_64` / `code_hash_fnv1a64` hashes the FULL file bytes (including any
        trailing alignment padding or second `s_endpgm`).
      - `trimmed_hash_fnv1a64` records the legacy hash up to and including the first
        `s_endpgm` / `s_endpgm_saved` terminator for diagnostics.
      - Instruction/export decoding covers all executable code up to the final
        `s_endpgm` / `s_endpgm_saved`, skipping padding words (`0x00000000`,
        `s_code_end`, `0xFFFFFFFF`, `0xCCCCCCCC`).
    """
    blob = bytes(raw_bytes)
    if len(blob) < 4 or _starts_with_container_magic(blob):
        return None

    dword_count = len(blob) // 4
    words = struct.unpack(f"<{dword_count}I", blob[: dword_count * 4])

    # Pass 1: locate all instruction-aligned `s_endpgm` / `s_endpgm_saved` terminators.
    endpgm_positions: list[tuple[int, int]] = []
    i = 0
    while i < dword_count:
        w0 = words[i]
        if w0 in (S_ENDPGM_WORD, S_ENDPGM_SAVED_WORD):
            endpgm_positions.append((i, w0))
            i += 1
            continue
        if w0 in _PADDING_WORDS:
            i += 1
            continue
        cat, inst_dwords = _classify_word0(w0)
        if cat == "mimg":
            step = inst_dwords if (i + inst_dwords <= dword_count) else (2 if i + 2 <= dword_count else 1)
        elif cat in ("exp", "smem", "vintrp", "valu", "salu", "sopp", "vmem", "ds", "flat"):
            step = inst_dwords if (i + inst_dwords <= dword_count) else 1
        else:
            step = 1
        i += step

    if not endpgm_positions:
        return None

    first_endpgm_idx, _first_term_word = endpgm_positions[0]
    last_endpgm_idx, last_term_word = endpgm_positions[-1]
    if (last_endpgm_idx + 1) < min_dwords:
        return None

    # Pass 2: decode all instructions & exports across `[0 .. last_endpgm_idx]`.
    exports: list[ExportInstruction] = []
    counts: dict[str, int] = {
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
    instruction_count = 0

    i = 0
    code_limit = last_endpgm_idx + 1
    while i < code_limit:
        w0 = words[i]
        if w0 in (S_ENDPGM_WORD, S_ENDPGM_SAVED_WORD):
            counts["sopp"] += 1
            instruction_count += 1
            i += 1
            continue
        if w0 in _PADDING_WORDS:
            i += 1
            continue
        cat, inst_dwords = _classify_word0(w0)
        if cat == "exp":
            exp = decode_exp_instruction(w0)
            if exp is not None:
                exports.append(exp)
            counts["exp"] += 1
            instruction_count += 1
            step = 2 if (i + 2 <= code_limit) else 1
        elif cat == "mimg":
            counts["mimg"] += 1
            instruction_count += 1
            step = inst_dwords if (i + inst_dwords <= code_limit) else (2 if i + 2 <= code_limit else 1)
        elif cat in counts:
            counts[cat] += 1
            instruction_count += 1
            step = inst_dwords if (i + inst_dwords <= code_limit) else 1
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
    if stage == "unknown":
        if manifest_hint is not None and manifest_hint.stage is not None:
            stage = manifest_hint.stage
        elif filename_hint is not None and filename_hint.stage is not None:
            stage = filename_hint.stage

    terminator_name = "s_endpgm" if last_term_word == S_ENDPGM_WORD else "s_endpgm_saved"
    whole_hash = fnv1a_64(blob)
    trimmed_byte_size = (first_endpgm_idx + 1) * 4
    trimmed_hash = fnv1a_64(blob[:trimmed_byte_size])

    filename_hash_hint = filename_hint.hash_fnv1a64 if filename_hint is not None else None
    filename_hash_matches = (
        whole_hash == filename_hash_hint if filename_hash_hint is not None else None
    )
    manifest_code_identity = manifest_hint.code_identity if manifest_hint is not None else None
    manifest_identity = (
        manifest_hint.stage_identities[0]
        if (manifest_hint is not None and manifest_hint.stage_identities)
        else None
    )
    manifest_stage = manifest_hint.stage if manifest_hint is not None else None
    manifest_identity_matches = (
        whole_hash == manifest_code_identity if manifest_code_identity is not None else None
    )

    return Rdna2ShaderHit(
        offset=0,
        end_offset=len(blob),
        byte_size=len(blob),
        dword_count=dword_count,
        instruction_count=instruction_count,
        fnv1a_64=whole_hash,
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
        hash_scope="whole_file",
        trimmed_hash_fnv1a64=trimmed_hash,
        trimmed_byte_size=trimmed_byte_size,
        terminator_count=len(endpgm_positions),
        filename_hash_hint=filename_hash_hint,
        filename_hash_matches=filename_hash_matches,
        manifest_code_identity=manifest_code_identity,
        manifest_identity=manifest_identity,
        manifest_stage=manifest_stage,
        manifest_identity_matches=manifest_identity_matches,
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
    whole_file: bool = False,
    filename_hint: FilenameIdentityHint | None = None,
    manifest_hint: ManifestStageHint | None = None,
) -> FileInspectResult:
    """Scan an in-memory buffer for AMD RDNA2 / GFX10.3 shaders using bounded chunks.

    When `whole_file=True`, treats the entire buffer as a standalone native RDNA2 stage
    blob and computes `fnv1a_64` / `code_hash_fnv1a64` over all bytes in the buffer.
    """
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

    if whole_file and limit == total_len:
        effective_hint = (
            filename_hint
            if filename_hint is not None
            else parse_filename_identity_hint(source_label)
        )
        standalone_hit = analyze_standalone_shader(
            view,
            min_dwords=min_dwords,
            filename_hint=effective_hint,
            manifest_hint=manifest_hint,
        )
        if standalone_hit is not None:
            return FileInspectResult(
                path=safe_label,
                file_size=total_len,
                bytes_scanned=total_len,
                truncated=False,
                hit_limit_reached=False,
                unaligned_tail_bytes=total_len % 4,
                shaders=(standalone_hit,),
                pipeline_records=(),
                hash_scope="whole_file",
                file_hash_fnv1a64=standalone_hit.fnv1a_64,
            )

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
    shaders.sort(key=_shader_hit_sort_key)
    records.sort(key=_pipeline_record_sort_key)
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
    whole_file: bool | None = None,
    manifest_hint: ManifestStageHint | None = None,
) -> FileInspectResult:
    """Stream-scan a single file on disk in bounded memory (`chunk_size + overlap`).

    Standalone native RDNA2 stage dump files (`native/<code-identity>.bin` from
    `playgta6_boot --dump-pipelines`, or standalone `.bin`/`.rdna2`/`.shader` files
    starting at offset 0 without container records) are hashed over the **entire**
    file bytes (`hash_scope="whole_file"`) to match the D3D12 guest code identity,
    `disasm-rdna2`, `translate-rdna2`, and `pack-rdna2`. Embedded multi-shader blobs
    inside larger containers continue to use bounded streaming region hashing.
    """
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

    filename_hint = parse_filename_identity_hint(file_path.name)
    has_standalone_ext = file_path.suffix.lower() in STANDALONE_SHADER_EXTENSIONS
    resolved_manifest_hint = manifest_hint
    if resolved_manifest_hint is None and base_dir is None and has_standalone_ext:
        resolved_manifest_hint = _find_manifest_hint_for_file(file_path)

    # Fast path for explicitly requested whole-file mode or manifest-verified native stages.
    whole_file_cap = max(max_shader_bytes, STANDALONE_MANIFEST_MAX_SHADER_BYTES)
    if (
        whole_file is True or (whole_file is None and resolved_manifest_hint is not None)
    ) and (max_bytes is None or max_bytes >= file_size) and (min_dwords * 4 <= file_size <= whole_file_cap):
        try:
            raw_bytes = file_path.read_bytes()
        except OSError as exc:
            return FileInspectResult(
                path=rel_label,
                file_size=file_size,
                bytes_scanned=0,
                truncated=file_size > 0,
                error=sanitize_error_message(file_path, exc),
            )
        standalone_hit = analyze_standalone_shader(
            raw_bytes,
            min_dwords=min_dwords,
            filename_hint=filename_hint,
            manifest_hint=resolved_manifest_hint,
        )
        if standalone_hit is not None:
            return FileInspectResult(
                path=rel_label,
                file_size=file_size,
                bytes_scanned=len(raw_bytes),
                truncated=False,
                hit_limit_reached=False,
                unaligned_tail_bytes=len(raw_bytes) % 4,
                shaders=(standalone_hit,),
                pipeline_records=(),
                hash_scope="whole_file",
                file_hash_fnv1a64=standalone_hit.fnv1a_64,
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

    truncated = file_size > bytes_read
    # Auto-promote standalone native RDNA2 stage files to whole-file FNV-1a 64 identity.
    if (
        whole_file is None
        and has_standalone_ext
        and not truncated
        and not hit_limit_reached
        and not records
        and shaders
        and file_size <= max_shader_bytes
        and shaders[0].offset == 0
        and (
            filename_hint.hash_fnv1a64 is not None
            or len(shaders) == 1
            or _is_standalone_multi_endpgm_stage(shaders)
        )
    ):
        try:
            raw_bytes = file_path.read_bytes()
        except OSError:
            raw_bytes = b""
        if raw_bytes and not _starts_with_container_magic(raw_bytes):
            standalone_hit = analyze_standalone_shader(
                raw_bytes,
                min_dwords=min_dwords,
                filename_hint=filename_hint,
                manifest_hint=resolved_manifest_hint,
            )
            if standalone_hit is not None:
                return FileInspectResult(
                    path=rel_label,
                    file_size=file_size,
                    bytes_scanned=bytes_read,
                    truncated=False,
                    hit_limit_reached=False,
                    unaligned_tail_bytes=bytes_read % 4,
                    shaders=(standalone_hit,),
                    pipeline_records=(),
                    hash_scope="whole_file",
                    file_hash_fnv1a64=standalone_hit.fnv1a_64,
                )

    shaders.sort(key=_shader_hit_sort_key)
    records.sort(key=_pipeline_record_sort_key)
    return FileInspectResult(
        path=rel_label,
        file_size=file_size,
        bytes_scanned=bytes_read,
        truncated=truncated,
        hit_limit_reached=hit_limit_reached,
        unaligned_tail_bytes=bytes_read % 4,
        shaders=tuple(shaders),
        pipeline_records=tuple(records),
        hash_scope="region",
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
    canonical_stage_counts: dict[str, int] = {
        stage: stage_counts[stage]
        for stage in sorted(stage_counts.keys(), key=_stage_sort_key)
    }
    reported_files.sort(
        key=lambda f: (_sanitize_report_path(f.path), f.file_size, f.bytes_scanned)
    )
    stage_breakdown = tuple(
        compute_stage_breakdown(canonical_stage_counts, total_shaders=total_shaders)
    )
    stage_percentages = {
        entry["stage"]: entry["percentage"] for entry in stage_breakdown
    }
    shader_summary = tuple(build_canonical_shader_summary(all_results))

    return InspectReport(
        schema_version=SCHEMA_VERSION,
        target_label=sanitize_source_label(target_label),
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
        stage_counts=canonical_stage_counts,
        terminator_counts=dict(sorted(terminator_counts.items())),
        export_target_histogram=dict(sorted(export_target_histogram.items())),
        size_histogram=ordered_size_hist,
        instruction_totals=instruction_totals,
        pipeline_record_counts=dict(sorted(pipeline_record_counts.items())),
        files=tuple(reported_files),
        stage_breakdown=stage_breakdown,
        stage_percentages=stage_percentages,
        shader_summary=shader_summary,
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
    whole_file: bool | None = None,
) -> InspectReport:
    """Recursively scan a directory for RDNA2 / GFX10.3 shaders in deterministic order."""
    root = Path(root_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {sanitize_source_label(root)}")

    chunk_size, overlap = _validate_chunk_params(chunk_size, overlap)
    label = sanitize_source_label(target_label if target_label is not None else root)
    manifest_ctx = _find_manifest_for_directory(root)

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
        file_manifest_hint = _lookup_manifest_hint(fpath, manifest_ctx)
        res = scan_file(
            fpath,
            base_dir=root,
            max_bytes=max_bytes,
            max_hits=remaining_hits,
            min_dwords=min_dwords,
            max_shader_bytes=max_shader_bytes,
            chunk_size=chunk_size,
            overlap=overlap,
            whole_file=whole_file,
            manifest_hint=file_manifest_hint,
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
    whole_file: bool | None = None,
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
            whole_file=whole_file,
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
            whole_file=whole_file,
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
    payload = sanitize_json_value(report.to_dict())
    return json.dumps(payload, indent=2, allow_nan=False) + "\n"


def render_markdown(report: InspectReport) -> str:
    """Render a strictly sanitized Markdown summary report (zero raw shader bytes)."""
    safe_target = sanitize_source_label(report.target_label)
    lines: list[str] = [
        f"# RDNA2 / GFX10.3 Shader Inspection Report: `{_escape_md_cell(safe_target)}`",
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
        pct_map = compute_stage_percentages(
            report.stage_counts,
            total_shaders=report.total_shaders,
        )
        lines.extend(
            ["## Stage Heuristics", "", "| Stage | Shaders | Share |", "| :--- | ---: | ---: |"]
        )
        for stage in sorted(report.stage_counts.keys(), key=_stage_sort_key):
            count = report.stage_counts[stage]
            pct = pct_map.get(stage, 0.0)
            lines.append(f"| `{_escape_md_cell(stage)}` | {count} | {pct:.2f}% |")
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

    matched = sorted(
        (f for f in report.files if f.shaders or f.pipeline_records or f.error),
        key=lambda f: (_sanitize_report_path(f.path), f.file_size, f.bytes_scanned),
    )
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
            safe_path = _sanitize_report_path(entry.path)
            stages = ", ".join(f"{k}:{v}" for k, v in entry.stage_counts.items()) or "-"
            status = (
                f"error: {_scrub_sensitive_text(entry.error)}"
                if entry.error
                else (
                    "hit_limit"
                    if entry.hit_limit_reached
                    else ("truncated" if entry.truncated else "ok")
                )
            )
            lines.append(
                f"| `{_escape_md_cell(safe_path)}` | {entry.bytes_scanned}/{entry.file_size} "
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
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--whole-file",
        action="store_true",
        help="Force whole-file FNV-1a 64 identity hashing on every scanned file.",
    )
    mode_group.add_argument(
        "--embedded",
        action="store_true",
        help="Force embedded sub-region scanning (disable standalone whole-file hashing).",
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

    whole_file_mode: bool | None = True if args.whole_file else (False if args.embedded else None)

    try:
        report = scan_path(
            target,
            max_bytes=args.max_bytes,
            max_hits=args.max_hits,
            min_dwords=args.min_dwords,
            target_label=target_label,
            whole_file=whole_file_mode,
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
