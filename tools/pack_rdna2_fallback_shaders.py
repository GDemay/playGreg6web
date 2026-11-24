#!/usr/bin/env python3
"""Sanitized RDNA2-hash -> WGPK (.wgsl.pack) fallback/override shader packer CLI (Track 4 / S16).

Reads discovered RDNA2/GFX10.3 shader hashes and inferred stages from
`tools/inspect_rdna2_shaders.py --json` output and/or explicit `--hashes`
arguments, optionally applies real translated WGSL overrides from `--wgsl-dir`,
and emits a deterministic binary `WGPK` v1 `.wgsl.pack` bundle matching
`shaders/src/pack.rs`, `web/workers/wgsl_pack_loader.js`, and
`tools/inspect_wgsl_pack.py` byte-for-byte.

Binary `WGPK` v1 container layout:
  Header (16 bytes, little-endian `<4sIII`):
    magic[4]       = b"WGPK"
    version        = u32 LE (1)
    entry_count    = u32 LE
    flags          = u32 LE (0)

  Entry table (`entry_count * 20` bytes, little-endian `<QB3sII`):
    hash           = u64 LE (FNV-1a 64-bit shader hash)
    stage          = u8 (1 = vertex, 2 = fragment, 3 = compute)
    reserved[3]    = 3 zero bytes
    offset         = u32 LE (byte offset from start of file)
    length         = u32 LE (UTF-8 WGSL payload byte length)

  Payload region:
    Concatenated, non-overlapping UTF-8 WGSL sources at `[offset .. offset + length]`.

Built-in WebGPU-validated fallback profiles match G34
(`web/workers/shader_fallback_router.js`) byte-for-byte:
  - `default`
  - `fullscreen_triangle_cbv`
  - `textured_quad_24b`

Security & privacy invariants:
  - Pure Python 3.12 standard library only (zero external dependencies).
  - Refuses to write generated `.wgsl.pack` or report files into tracked
    repository directories (must target gitignored `work/`, `local-data/`, or
    a directory outside the repository).
  - Strictly sanitizes `--report-json`, stdout, and stderr: strips Windows,
    UNC, and POSIX host paths plus active OS usernames, and never emits raw
    binary shader bytes.
  - Ignores symlinks in `--wgsl-dir` that escape the override directory.

CLI usage:
    python tools/pack_rdna2_fallback_shaders.py \
        [--scan-json <path>] [--hashes <stage:hex,...>] \
        [--wgsl-dir <dir>] \
        [--profile {fullscreen_triangle_cbv,textured_quad_24b,default}] \
        --out <path.wgsl.pack> [--report-json <path>]
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
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# WGPK v1 binary format constants (matching shaders/src/pack.rs byte-for-byte)
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 1

PACK_MAGIC = b"WGPK"
PACK_VERSION = 1
PACK_HEADER_STRUCT = struct.Struct("<4sIII")
PACK_HEADER_BYTES = PACK_HEADER_STRUCT.size  # 16 bytes

PACK_ENTRY_STRUCT = struct.Struct("<QB3sII")
PACK_ENTRY_BYTES = PACK_ENTRY_STRUCT.size  # 20 bytes

U32_MAX = 0xFFFFFFFF
U64_MAX = 0xFFFFFFFFFFFFFFFF

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWED_REPO_OUTPUT_DIR_NAMES: tuple[str, ...] = ("local-data", "work")


class ShaderStage(IntEnum):
    """WebGPU shader stage codes matching `ShaderStage` in `shaders/src/convert.rs`."""

    VERTEX = 1
    FRAGMENT = 2
    COMPUTE = 3

    def as_str(self) -> str:
        return STAGE_NAMES[self]


STAGE_NAMES: dict[ShaderStage, str] = {
    ShaderStage.VERTEX: "vertex",
    ShaderStage.FRAGMENT: "fragment",
    ShaderStage.COMPUTE: "compute",
}

STAGE_ALIASES: dict[str, ShaderStage] = {
    "vertex": ShaderStage.VERTEX,
    "vs": ShaderStage.VERTEX,
    "vert": ShaderStage.VERTEX,
    "1": ShaderStage.VERTEX,
    "fragment": ShaderStage.FRAGMENT,
    "fs": ShaderStage.FRAGMENT,
    "frag": ShaderStage.FRAGMENT,
    "pixel": ShaderStage.FRAGMENT,
    "ps": ShaderStage.FRAGMENT,
    "2": ShaderStage.FRAGMENT,
    "compute": ShaderStage.COMPUTE,
    "cs": ShaderStage.COMPUTE,
    "comp": ShaderStage.COMPUTE,
    "3": ShaderStage.COMPUTE,
}

STAGE_OVERRIDE_PREFIXES: dict[ShaderStage, tuple[str, ...]] = {
    ShaderStage.VERTEX: ("vertex", "vs"),
    ShaderStage.FRAGMENT: ("fragment", "pixel", "fs", "ps"),
    ShaderStage.COMPUTE: ("compute", "cs"),
}

DEFAULT_ENTRY_POINTS: dict[ShaderStage, str] = {
    ShaderStage.VERTEX: "vs_main",
    ShaderStage.FRAGMENT: "fs_main",
    ShaderStage.COMPUTE: "cs_main",
}

PROFILE_CHOICES: tuple[str, ...] = (
    "fullscreen_triangle_cbv",
    "textured_quad_24b",
    "default",
)

PROFILE_ALIASES: dict[str, str] = {
    "default": "default",
    "minimal": "default",
    "fullscreen_triangle_cbv": "fullscreen_triangle_cbv",
    "fullscreen_cbv": "fullscreen_triangle_cbv",
    "fullscreen-triangle-cbv": "fullscreen_triangle_cbv",
    "fullscreen-cbv": "fullscreen_triangle_cbv",
    "textured_quad_24b": "textured_quad_24b",
    "textured_24b": "textured_quad_24b",
    "textured-quad-24b": "textured_quad_24b",
    "textured-24b": "textured_quad_24b",
}

# ---------------------------------------------------------------------------
# Built-in WebGPU-validated WGSL profiles (matching G34 shader_fallback_router.js)
# ---------------------------------------------------------------------------

_DEFAULT_VERTEX_WGSL = "\n".join(
    [
        "@vertex",
        "fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {",
        "    var positions = array<vec2<f32>, 3>(",
        "        vec2<f32>(-1.0, -1.0),",
        "        vec2<f32>(3.0, -1.0),",
        "        vec2<f32>(-1.0, 3.0),",
        "    );",
        "    let p = positions[vertex_index];",
        "    return vec4<f32>(p, 0.0, 1.0);",
        "}",
        "",
    ]
)

_DEFAULT_FRAGMENT_WGSL = "\n".join(
    [
        "@fragment",
        "fn fs_main() -> @location(0) vec4<f32> {",
        "    return vec4<f32>(0.05, 0.07, 0.11, 1.0);",
        "}",
        "",
    ]
)

_DEFAULT_COMPUTE_WGSL = "\n".join(
    [
        "@compute @workgroup_size(1)",
        "fn cs_main() {}",
        "",
    ]
)

_FULLSCREEN_TRIANGLE_CBV_VERTEX_WGSL = "\n".join(
    [
        "struct FrameCbv {",
        "    data: array<vec4<f32>, 16>,",
        "};",
        "",
        "@group(0) @binding(0) var<uniform> frame_cbv: FrameCbv;",
        "",
        "struct VsOut {",
        "    @builtin(position) position: vec4<f32>,",
        "    @location(0) uv: vec2<f32>,",
        "};",
        "",
        "@vertex",
        "fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {",
        "    var positions = array<vec2<f32>, 3>(",
        "        vec2<f32>(-1.0, -1.0),",
        "        vec2<f32>(3.0, -1.0),",
        "        vec2<f32>(-1.0, 3.0),",
        "    );",
        "    var uvs = array<vec2<f32>, 3>(",
        "        vec2<f32>(0.0, 1.0),",
        "        vec2<f32>(2.0, 1.0),",
        "        vec2<f32>(0.0, -1.0),",
        "    );",
        "    let idx = vid % 3u;",
        "    let cbv_keep = f32(bitcast<u32>(frame_cbv.data[0].x) & 0xffu) * 0.0;",
        "    var out: VsOut;",
        "    out.position = vec4<f32>(positions[idx], cbv_keep, 1.0);",
        "    out.uv = uvs[idx];",
        "    return out;",
        "}",
        "",
    ]
)

_FULLSCREEN_TRIANGLE_CBV_FRAGMENT_WGSL = "\n".join(
    [
        "struct FrameCbv {",
        "    data: array<vec4<f32>, 16>,",
        "};",
        "",
        "@group(0) @binding(0) var<uniform> frame_cbv: FrameCbv;",
        "",
        "struct FsIn {",
        "    @location(0) uv: vec2<f32>,",
        "};",
        "",
        "@fragment",
        "fn fs_main(input: FsIn) -> @location(0) vec4<f32> {",
        "    let uv = clamp(input.uv, vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0));",
        "    let b0 = bitcast<u32>(frame_cbv.data[0].x);",
        "    let b1 = bitcast<u32>(frame_cbv.data[0].y);",
        "    let b2 = bitcast<u32>(frame_cbv.data[0].z) ^ bitcast<u32>(frame_cbv.data[0].w);",
        "    let cbv_mod = vec3<f32>(",
        "        f32(b0 & 0xffu) / 255.0,",
        "        f32(b1 & 0xffu) / 255.0,",
        "        f32(b2 & 0xffu) / 255.0,",
        "    );",
        "    let base = vec3<f32>(",
        "        0.08 + 0.35 * uv.x,",
        "        0.14 + 0.40 * uv.y,",
        "        0.24 + 0.30 * (1.0 - uv.x),",
        "    );",
        "    let color = clamp(",
        "        mix(base, cbv_mod, 0.25),",
        "        vec3<f32>(0.0, 0.0, 0.0),",
        "        vec3<f32>(1.0, 1.0, 1.0),",
        "    );",
        "    return vec4<f32>(color, 1.0);",
        "}",
        "",
    ]
)

_TEXTURED_QUAD_24B_VERTEX_WGSL = "\n".join(
    [
        "struct VsOut {",
        "    @builtin(position) position: vec4<f32>,",
        "    @location(0) uv: vec2<f32>,",
        "};",
        "",
        "@vertex",
        "fn vs_main(",
        "    @location(0) pos: vec4<f32>,",
        "    @location(1) uv: vec2<f32>,",
        ") -> VsOut {",
        "    var out: VsOut;",
        "    let w = select(pos.w, 1.0, pos.w == 0.0);",
        "    out.position = vec4<f32>(pos.xyz, w);",
        "    out.uv = uv;",
        "    return out;",
        "}",
        "",
    ]
)

_TEXTURED_QUAD_24B_FRAGMENT_WGSL = "\n".join(
    [
        "@group(0) @binding(0) var quad_tex: texture_2d<f32>;",
        "@group(0) @binding(1) var quad_sampler: sampler;",
        "",
        "@fragment",
        "fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {",
        "    let texel = textureSample(quad_tex, quad_sampler, uv);",
        "    return texel;",
        "}",
        "",
    ]
)

BUILTIN_PROFILES: dict[str, dict[ShaderStage, str]] = {
    "default": {
        ShaderStage.VERTEX: _DEFAULT_VERTEX_WGSL,
        ShaderStage.FRAGMENT: _DEFAULT_FRAGMENT_WGSL,
        ShaderStage.COMPUTE: _DEFAULT_COMPUTE_WGSL,
    },
    "fullscreen_triangle_cbv": {
        ShaderStage.VERTEX: _FULLSCREEN_TRIANGLE_CBV_VERTEX_WGSL,
        ShaderStage.FRAGMENT: _FULLSCREEN_TRIANGLE_CBV_FRAGMENT_WGSL,
        ShaderStage.COMPUTE: _DEFAULT_COMPUTE_WGSL,
    },
    "textured_quad_24b": {
        ShaderStage.VERTEX: _TEXTURED_QUAD_24B_VERTEX_WGSL,
        ShaderStage.FRAGMENT: _TEXTURED_QUAD_24B_FRAGMENT_WGSL,
        ShaderStage.COMPUTE: _DEFAULT_COMPUTE_WGSL,
    },
}

# ---------------------------------------------------------------------------
# Strict cross-platform path & username sanitization
# ---------------------------------------------------------------------------

_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)


class PackerError(ValueError):
    """Raised when input parsing, validation, or packing fails."""

    def __init__(self, message: str, *, code: str = "packer_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class UnsafeOutputPathError(PackerError):
    """Raised when an output path targets a tracked repository directory."""

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


def sanitize_text(raw: str | None) -> str:
    """Strip Windows/UNC/POSIX host paths and active OS usernames from text."""
    if raw is None:
        return ""
    text = str(raw)
    text = _UNC_BS_PATH_RE.sub("<path>", text)
    text = text.replace("\\", "/")
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = _UNC_FS_PATH_RE.sub("<path>", text)
    text = _POSIX_PATH_RE.sub("<path>", text)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(username)}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
        text = pattern.sub("<user>", text)
    return text


def sanitize_label(source: str | Path | None) -> str:
    """Return a strictly sanitized basename-only label (zero host paths/usernames)."""
    if source is None:
        return "<none>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    if raw.startswith("<") and raw.endswith(">"):
        return sanitize_text(raw)
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return sanitize_text(base)


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings/paths and strip raw binary payloads."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Path):
        return sanitize_label(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]
    return value


def sanitize_os_error(source: str | Path, exc: OSError) -> str:
    """Format an OSError without ever leaking host paths or usernames."""
    label = sanitize_label(source)
    reason = sanitize_text(exc.strerror or type(exc).__name__)
    return f"failed to access {label}: {reason}"


# ---------------------------------------------------------------------------
# Output path safety guard (no generated packs/reports in tracked repo dirs)
# ---------------------------------------------------------------------------


def _is_under(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def is_safe_output_path(path: Path | str, *, repo_root: Path | str = REPO_ROOT) -> bool:
    """Return True iff `path` is under gitignored `work/`/`local-data/` or outside repo."""
    target = Path(path).expanduser().resolve()
    root = Path(repo_root).resolve()
    if not _is_under(target, root):
        return True
    return any(_is_under(target, root / name) for name in ALLOWED_REPO_OUTPUT_DIR_NAMES)


def validate_output_path(
    path: Path | str,
    *,
    flag_name: str = "--out",
    repo_root: Path | str = REPO_ROOT,
) -> Path:
    """Resolve and validate that an output file path is not inside a tracked repo directory."""
    target = Path(path).expanduser().resolve()
    if not is_safe_output_path(target, repo_root=repo_root):
        raise UnsafeOutputPathError(
            f"refusing to write {flag_name} into a tracked repository directory "
            "(must be under gitignored work/, local-data/, or outside the repository)"
        )
    return target


# ---------------------------------------------------------------------------
# Hash, stage, profile, and WGSL parsing helpers
# ---------------------------------------------------------------------------


def normalize_shader_hash(value: int | str) -> int:
    """Normalize an integer or hex string shader hash to unsigned 64-bit int."""
    if isinstance(value, bool):
        raise PackerError(f"invalid shader hash: {value!r}", code="invalid_hash")
    if isinstance(value, int):
        if value < 0 or value > U64_MAX:
            raise PackerError(f"shader hash out of u64 range: {value}", code="invalid_hash")
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        raw_hex = trimmed[2:] if trimmed.lower().startswith("0x") else trimmed
        if not raw_hex or len(raw_hex) > 16 or not re.fullmatch(r"[0-9a-fA-F]+", raw_hex):
            raise PackerError(f"invalid hex shader hash: {sanitize_text(trimmed)!r}", code="invalid_hash")
        return int(raw_hex, 16)
    raise PackerError(f"unsupported shader hash type: {type(value).__name__}", code="invalid_hash")


def format_shader_hash(shader_hash: int) -> str:
    """Format a 64-bit hash as canonical `0x` + 16 lowercase hex digits."""
    return f"0x{normalize_shader_hash(shader_hash):016x}"


def format_hash_bare(shader_hash: int) -> str:
    """Format a 64-bit hash as 16 lowercase hex digits without `0x`."""
    return f"{normalize_shader_hash(shader_hash):016x}"


def normalize_stage(stage: int | str | ShaderStage, *, allow_unknown: bool = False) -> ShaderStage | None:
    """Normalize a stage code or name into `ShaderStage` (or `None` for `'unknown'` when allowed)."""
    if isinstance(stage, ShaderStage):
        return stage
    if isinstance(stage, bool):
        raise PackerError(f"invalid shader stage: {stage!r}", code="invalid_stage")
    if isinstance(stage, int):
        try:
            return ShaderStage(stage)
        except ValueError as exc:
            raise PackerError(f"invalid shader stage code: {stage}", code="invalid_stage") from exc
    if isinstance(stage, str):
        key = stage.strip().lower()
        if allow_unknown and key in ("unknown", ""):
            return None
        resolved = STAGE_ALIASES.get(key)
        if resolved is not None:
            return resolved
        raise PackerError(f"invalid shader stage name: {sanitize_text(stage)!r}", code="invalid_stage")
    raise PackerError(f"unsupported shader stage type: {type(stage).__name__}", code="invalid_stage")


def normalize_profile(profile: str | None) -> str:
    """Normalize a fallback profile name or alias to its canonical profile key."""
    if profile is None:
        return "default"
    if not isinstance(profile, str):
        raise PackerError(f"invalid profile type: {type(profile).__name__}", code="invalid_profile")
    key = profile.strip().lower()
    if not key:
        return "default"
    canonical = PROFILE_ALIASES.get(key)
    if canonical is None:
        raise PackerError(
            f"unknown profile {sanitize_text(profile)!r} (expected one of: {', '.join(PROFILE_CHOICES)})",
            code="invalid_profile",
        )
    return canonical


def strip_wgsl_comments(source: str) -> str:
    """Strip WGSL `//` line comments and nested `/* ... */` block comments."""
    out: list[str] = []
    i = 0
    n = len(source)
    block_depth = 0
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ""
        if block_depth > 0:
            if ch == "/" and nxt == "*":
                block_depth += 1
                out.append("  ")
                i += 2
            elif ch == "*" and nxt == "/":
                block_depth -= 1
                out.append("  ")
                i += 2
            elif ch == "\n":
                out.append("\n")
                i += 1
            else:
                out.append(" ")
                i += 1
            continue
        if ch == "/" and nxt == "/":
            out.append("  ")
            i += 2
            while i < n and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if ch == "/" and nxt == "*":
            block_depth = 1
            out.append("  ")
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def detect_wgsl_stages(wgsl: str) -> tuple[ShaderStage, ...]:
    """Detect `@vertex`, `@fragment`, and `@compute` entry-point stages in WGSL source."""
    stripped = strip_wgsl_comments(wgsl)
    found: list[ShaderStage] = []
    for stage, attr in (
        (ShaderStage.VERTEX, "vertex"),
        (ShaderStage.FRAGMENT, "fragment"),
        (ShaderStage.COMPUTE, "compute"),
    ):
        if re.search(rf"@{attr}\b", stripped):
            found.append(stage)
    return tuple(found)


def extract_entry_point(wgsl: str, stage: ShaderStage) -> str:
    """Extract the WGSL entry-point function name for `stage`, matching G34."""
    stripped = strip_wgsl_comments(wgsl)
    stage_name = stage.as_str()
    match = re.search(rf"@{stage_name}\b[\s\S]*?\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
    if match:
        return match.group(1)
    fallback = re.search(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
    if fallback:
        return fallback.group(1)
    return DEFAULT_ENTRY_POINTS[stage]


def normalize_fragment_entry_point(wgsl: str) -> tuple[str, bool]:
    """Normalize a legacy `@fragment ... fn ps_main` entry point to `fs_main`.

    The WebGPU v2 pipeline path (`web/workers/render_state_packets.js`) and
    `docs/gpu_commands.md` bind fragment modules with the fixed entry point
    `fs_main`. When an override WGSL module defines `@fragment fn ps_main(...)`
    without an existing `fs_main` declaration, rename the entry point declaration
    comment-safely to `fs_main` while leaving custom entry names untouched.
    """
    stripped = strip_wgsl_comments(wgsl)
    if re.search(r"\bfn\s+fs_main\b", stripped):
        return wgsl, False
    match = re.search(r"@fragment\b[\s\S]*?\bfn\s+([A-Za-z_][A-Za-z0-9_]*)", stripped)
    if not match or match.group(1) != "ps_main":
        return wgsl, False
    updated = wgsl[: match.start(1)] + "fs_main" + wgsl[match.end(1) :]
    return updated, True


def _import_rdna2_inspector() -> Any:
    """Import sibling `tools/inspect_rdna2_shaders.py` regardless of invocation style."""
    try:
        from tools import inspect_rdna2_shaders as inspector  # type: ignore[import-not-found]

        return inspector
    except ImportError:
        tools_dir = str(Path(__file__).resolve().parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        import inspect_rdna2_shaders as inspector  # type: ignore[import-not-found]

        return inspector


# ---------------------------------------------------------------------------
# Input ingestion (`--scan-json` & `--hashes`) and `--wgsl-dir` indexing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShaderInputRecord:
    """One raw `(hash, stage)` record ingested from `--scan-json` or `--hashes`."""

    shader_hash: int
    stage: ShaderStage | None  # None represents "unknown" from RDNA2 scan heuristics
    origin: str  # "scan_json" | "cli_hashes"


def parse_hashes_arg(raw_specs: Sequence[str] | str) -> list[ShaderInputRecord]:
    """Parse one or more `<stage:hex,...>` CLI hash specifications."""
    items = [raw_specs] if isinstance(raw_specs, str) else list(raw_specs)
    records: list[ShaderInputRecord] = []

    for spec in items:
        if not isinstance(spec, str):
            raise PackerError("hash specification must be a string", code="invalid_hash_spec")
        tokens = [tok.strip() for tok in spec.split(",")]
        for tok in tokens:
            if not tok:
                continue
            if ":" not in tok:
                raise PackerError(
                    f"invalid --hashes entry {sanitize_text(tok)!r}: expected '<stage>:<hex>'",
                    code="invalid_hash_spec",
                )
            raw_stage, raw_hash = tok.split(":", 1)
            stage = normalize_stage(raw_stage, allow_unknown=False)
            assert stage is not None
            shader_hash = normalize_shader_hash(raw_hash)
            records.append(
                ShaderInputRecord(
                    shader_hash=shader_hash,
                    stage=stage,
                    origin="cli_hashes",
                )
            )

    if items and not records:
        raise PackerError("--hashes did not contain any '<stage>:<hex>' entries", code="invalid_hash_spec")
    return records


def load_scan_json_records(scan_json_path: Path | str) -> list[ShaderInputRecord]:
    """Read `(hash, stage)` entries from an `inspect_rdna2_shaders.py --json` report."""
    path = Path(scan_json_path).expanduser()
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PackerError(sanitize_os_error(path, exc), code="scan_json_read_error") from exc
    except UnicodeDecodeError as exc:
        raise PackerError(
            f"failed to decode {sanitize_label(path)} as UTF-8",
            code="scan_json_decode_error",
        ) from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise PackerError(
            f"invalid JSON in {sanitize_label(path)}: line {exc.lineno} col {exc.colno}",
            code="scan_json_parse_error",
        ) from exc

    return extract_scan_records_from_payload(payload, source_label=sanitize_label(path))


def extract_scan_records_from_payload(
    payload: Any,
    *,
    source_label: str = "<scan-json>",
) -> list[ShaderInputRecord]:
    """Extract `(hash, stage)` records from parsed `inspect_rdna2_shaders.py` JSON."""
    records: list[ShaderInputRecord] = []

    def _ingest_shader_list(items: Any) -> None:
        if not isinstance(items, list):
            raise PackerError(
                f"expected list of shaders in {sanitize_label(source_label)}",
                code="scan_json_schema_error",
            )
        for idx, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise PackerError(
                    f"invalid shader entry at index {idx} in {sanitize_label(source_label)}",
                    code="scan_json_schema_error",
                )
            if "code_hash_fnv1a64" in item:
                shader_hash = normalize_shader_hash(item["code_hash_fnv1a64"])
            elif "hash" in item:
                shader_hash = normalize_shader_hash(item["hash"])
            else:
                raise PackerError(
                    f"shader entry at index {idx} missing 'hash' in {sanitize_label(source_label)}",
                    code="scan_json_schema_error",
                )
            raw_stage = item.get("stage", "unknown")
            stage = normalize_stage(raw_stage, allow_unknown=True)
            records.append(
                ShaderInputRecord(
                    shader_hash=shader_hash,
                    stage=stage,
                    origin="scan_json",
                )
            )

    if isinstance(payload, list):
        _ingest_shader_list(payload)
        return records

    if not isinstance(payload, Mapping):
        raise PackerError(
            f"scan JSON root must be a JSON object or list in {sanitize_label(source_label)}",
            code="scan_json_schema_error",
        )

    handled = False
    if "native_stages" in payload:
        native_stages = payload["native_stages"]
        if not isinstance(native_stages, list):
            raise PackerError(
                f"'native_stages' must be a list in {sanitize_label(source_label)}",
                code="scan_json_schema_error",
            )
        for idx, entry in enumerate(native_stages):
            if not isinstance(entry, Mapping):
                raise PackerError(
                    f"invalid native_stages entry at index {idx} in {sanitize_label(source_label)}",
                    code="scan_json_schema_error",
                )
            if "code_identity" not in entry:
                raise PackerError(
                    f"native_stages entry at index {idx} missing 'code_identity' in {sanitize_label(source_label)}",
                    code="scan_json_schema_error",
                )
            shader_hash = normalize_shader_hash(entry["code_identity"])
            stage = normalize_stage(entry.get("stage", "unknown"), allow_unknown=True)
            records.append(
                ShaderInputRecord(
                    shader_hash=shader_hash,
                    stage=stage,
                    origin="pipeline_dump_manifest",
                )
            )
        handled = True

    if "files" in payload:
        files = payload["files"]
        if not isinstance(files, list):
            raise PackerError(
                f"'files' must be a list in {sanitize_label(source_label)}",
                code="scan_json_schema_error",
            )
        for file_entry in files:
            if not isinstance(file_entry, Mapping):
                raise PackerError(
                    f"invalid file record in {sanitize_label(source_label)}",
                    code="scan_json_schema_error",
                )
            if "shaders" in file_entry:
                _ingest_shader_list(file_entry["shaders"])
        handled = True

    if "shaders" in payload:
        shaders_field = payload["shaders"]
        if isinstance(shaders_field, list):
            _ingest_shader_list(shaders_field)
        elif isinstance(shaders_field, Mapping):
            for hash_key, meta in shaders_field.items():
                shader_hash = normalize_shader_hash(hash_key)
                if isinstance(meta, Mapping):
                    stage = normalize_stage(meta.get("stage", "unknown"), allow_unknown=True)
                elif isinstance(meta, (str, int)):
                    stage = normalize_stage(meta, allow_unknown=True)
                else:
                    raise PackerError(
                        f"invalid shader mapping entry in {sanitize_label(source_label)}",
                        code="scan_json_schema_error",
                    )
                records.append(
                    ShaderInputRecord(
                        shader_hash=shader_hash,
                        stage=stage,
                        origin="scan_json",
                    )
                )
        else:
            raise PackerError(
                f"'shaders' must be a list or object in {sanitize_label(source_label)}",
                code="scan_json_schema_error",
            )
        handled = True

    if not handled:
        raise PackerError(
            f"unrecognized scan JSON schema in {sanitize_label(source_label)} "
            "(expected 'files[*].shaders', 'shaders', or 'native_stages')",
            code="scan_json_schema_error",
        )

    return records


def load_native_dir_records(native_path: Path | str) -> list[ShaderInputRecord]:
    """Scan a native RDNA2 dump file or pipeline dump directory using whole-file FNV-1a 64 hashes.

    Keys every discovered standalone native RDNA2 stage (`native/<code-identity>.bin`,
    `.bin`/`.rdna2`/`.shader`) by its whole-file FNV-1a 64-bit code identity, matching
    the D3D12 guest layer (`playgta6-pipeline-dump` v1) and `pack-rdna2` (S18).
    """
    target = Path(native_path).expanduser()
    if not target.exists():
        raise PackerError(
            f"--native-dir target does not exist: {sanitize_label(target)}",
            code="native_dir_missing",
        )

    inspector = _import_rdna2_inspector()
    try:
        report = inspector.scan_path(
            target,
            whole_file=True if target.is_file() else None,
        )
    except OSError as exc:
        raise PackerError(sanitize_os_error(target, exc), code="native_dir_read_error") from exc

    records: list[ShaderInputRecord] = []
    for file_res in report.files:
        if file_res.error is not None:
            raise PackerError(sanitize_text(file_res.error), code="native_dir_read_error")
        for hit in file_res.shaders:
            stage = normalize_stage(hit.stage, allow_unknown=True)
            records.append(
                ShaderInputRecord(
                    shader_hash=hit.code_hash_fnv1a64,
                    stage=stage,
                    origin="native_dir",
                )
            )
    return records


@dataclass(frozen=True)
class WgslOverrideFile:
    """Validated override `.wgsl` file inside `--wgsl-dir`."""

    filename: str
    path: Path


@dataclass
class WgslOverrideIndex:
    """Safe, case-insensitive index of `.wgsl` files directly inside `--wgsl-dir`."""

    files_by_lower_name: dict[str, WgslOverrideFile]
    skipped_escaping_symlinks: int = 0

    def lookup(self, shader_hash: int, stage: ShaderStage) -> WgslOverrideFile | None:
        """Resolve an override file with precedence: `<stage>_<hash>.wgsl` > `<hash>.wgsl`."""
        bare = format_hash_bare(shader_hash)
        prefixed = format_shader_hash(shader_hash)
        candidates: list[str] = []
        for prefix in STAGE_OVERRIDE_PREFIXES[stage]:
            candidates.append(f"{prefix}_{bare}.wgsl")
            candidates.append(f"{prefix}_{prefixed}.wgsl")
        candidates.append(f"{bare}.wgsl")
        candidates.append(f"{prefixed}.wgsl")

        for name in candidates:
            hit = self.files_by_lower_name.get(name.lower())
            if hit is not None:
                return hit
        return None

    def infer_stage_for_unknown(self, shader_hash: int) -> tuple[ShaderStage, WgslOverrideFile] | None:
        """Attempt to infer a stage for an `'unknown'` scan hit from `--wgsl-dir` overrides."""
        bare = format_hash_bare(shader_hash)
        prefixed = format_shader_hash(shader_hash)

        stage_hits: list[tuple[ShaderStage, WgslOverrideFile]] = []
        for stage in (ShaderStage.VERTEX, ShaderStage.FRAGMENT, ShaderStage.COMPUTE):
            for prefix in STAGE_OVERRIDE_PREFIXES[stage]:
                for candidate in (f"{prefix}_{bare}.wgsl", f"{prefix}_{prefixed}.wgsl"):
                    hit = self.files_by_lower_name.get(candidate.lower())
                    if hit is not None:
                        stage_hits.append((stage, hit))
                        break
                else:
                    continue
                break
        if len(stage_hits) == 1:
            return stage_hits[0]
        if len(stage_hits) > 1:
            return None

        for candidate in (f"{bare}.wgsl", f"{prefixed}.wgsl"):
            hit = self.files_by_lower_name.get(candidate.lower())
            if hit is not None:
                wgsl = read_override_wgsl(hit)
                detected = detect_wgsl_stages(wgsl)
                if len(detected) == 1:
                    return detected[0], hit
                return None
        return None


def index_wgsl_dir(wgsl_dir: Path | str) -> WgslOverrideIndex:
    """Safely index top-level `.wgsl` files inside `--wgsl-dir` (skipping escaping symlinks)."""
    root = Path(wgsl_dir).expanduser()
    if not root.exists():
        raise PackerError(
            f"--wgsl-dir does not exist: {sanitize_label(root)}",
            code="wgsl_dir_missing",
        )
    if not root.is_dir():
        raise PackerError(
            f"--wgsl-dir is not a directory: {sanitize_label(root)}",
            code="wgsl_dir_invalid",
        )

    resolved_root = root.resolve()
    files_by_lower: dict[str, WgslOverrideFile] = {}
    skipped_escaping = 0

    try:
        entries = sorted(root.iterdir(), key=lambda p: (p.name.lower(), p.name))
    except OSError as exc:
        raise PackerError(sanitize_os_error(root, exc), code="wgsl_dir_read_error") from exc

    for entry in entries:
        if not entry.name.lower().endswith(".wgsl"):
            continue
        try:
            resolved = entry.resolve(strict=True)
        except OSError:
            if entry.is_symlink():
                skipped_escaping += 1
            continue
        if not resolved.is_relative_to(resolved_root):
            skipped_escaping += 1
            continue
        if not resolved.is_file():
            continue
        key = entry.name.lower()
        if key not in files_by_lower:
            files_by_lower[key] = WgslOverrideFile(filename=entry.name, path=resolved)

    return WgslOverrideIndex(
        files_by_lower_name=files_by_lower,
        skipped_escaping_symlinks=skipped_escaping,
    )


def read_override_wgsl(override: WgslOverrideFile) -> str:
    """Read and validate a UTF-8 non-empty WGSL override file."""
    try:
        raw = override.path.read_bytes()
    except OSError as exc:
        raise PackerError(sanitize_os_error(override.filename, exc), code="wgsl_read_error") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PackerError(
            f"WGSL override {sanitize_label(override.filename)} is not valid UTF-8",
            code="wgsl_invalid_utf8",
        ) from exc
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise PackerError(
            f"WGSL override {sanitize_label(override.filename)} is empty",
            code="wgsl_empty",
        )
    return text


# ---------------------------------------------------------------------------
# WGPK v1 binary writer & packed entry models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackedShaderEntry:
    """Single resolved, deduplicated, sorted entry placed in a `WGPK` pack."""

    shader_hash: int
    stage: ShaderStage
    wgsl: str
    source: str  # "override" | "profile"
    profile: str | None
    override_file: str | None
    entry_point: str
    pack_offset: int
    pack_length: int

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.shader_hash)

    def to_report_dict(self) -> dict[str, Any]:
        """Sanitized metadata record (never includes raw binary bytes or WGSL source)."""
        return {
            "hash": self.hash_hex,
            "stage": self.stage.as_str(),
            "stage_code": int(self.stage),
            "source": self.source,
            "profile": self.profile,
            "override_file": sanitize_label(self.override_file) if self.override_file else None,
            "entry_point": sanitize_text(self.entry_point),
            "pack_offset": self.pack_offset,
            "pack_length": self.pack_length,
        }


@dataclass(frozen=True)
class PackBuildResult:
    """Result of building a `WGPK` v1 binary pack and its sanitized report."""

    pack_bytes: bytes
    pack_sha256: str
    profile: str
    entries: tuple[PackedShaderEntry, ...]
    input_records_total: int
    duplicate_entries_collapsed: int
    skipped_unknown_stage_count: int
    stage_conflict_hash_count: int
    warnings: tuple[str, ...]

    def to_report_dict(
        self,
        *,
        output_label: str,
        scan_json_labels: Sequence[str] = (),
        native_dir_labels: Sequence[str] = (),
        hash_spec_count: int = 0,
        wgsl_dir_label: str | None = None,
    ) -> dict[str, Any]:
        stage_counts = {
            "vertex": sum(1 for e in self.entries if e.stage == ShaderStage.VERTEX),
            "fragment": sum(1 for e in self.entries if e.stage == ShaderStage.FRAGMENT),
            "compute": sum(1 for e in self.entries if e.stage == ShaderStage.COMPUTE),
        }
        unique_hashes = len({e.shader_hash for e in self.entries})
        override_count = sum(1 for e in self.entries if e.source == "override")
        profile_count = sum(1 for e in self.entries if e.source == "profile")
        total_wgsl_bytes = sum(e.pack_length for e in self.entries)

        inputs_payload: dict[str, Any] = {
            "scan_json_files": [sanitize_label(lbl) for lbl in scan_json_labels],
            "hash_spec_count": int(hash_spec_count),
            "wgsl_dir": sanitize_label(wgsl_dir_label) if wgsl_dir_label else None,
        }
        if native_dir_labels:
            inputs_payload["native_dirs"] = [sanitize_label(lbl) for lbl in native_dir_labels]

        report: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "pack_format": PACK_MAGIC.decode("ascii"),
            "pack_version": PACK_VERSION,
            "pack_flags": 0,
            "profile": self.profile,
            "output_file": sanitize_label(output_label),
            "pack_bytes": len(self.pack_bytes),
            "pack_sha256": self.pack_sha256,
            "inputs": inputs_payload,
            "summary": {
                "input_records_total": self.input_records_total,
                "packed_entry_count": len(self.entries),
                "unique_hash_count": unique_hashes,
                "duplicate_entries_collapsed": self.duplicate_entries_collapsed,
                "skipped_unknown_stage_count": self.skipped_unknown_stage_count,
                "override_entry_count": override_count,
                "profile_entry_count": profile_count,
                "stage_conflict_hash_count": self.stage_conflict_hash_count,
                "stage_counts": stage_counts,
                "total_wgsl_bytes": total_wgsl_bytes,
            },
            "entries": [entry.to_report_dict() for entry in self.entries],
            "warnings": [sanitize_text(w) for w in self.warnings],
        }
        return sanitize_value(report)


def serialize_wgpk(
    entries: Sequence[tuple[int, ShaderStage, str]],
    *,
    flags: int = 0,
) -> tuple[bytes, list[tuple[int, int]]]:
    """Serialize `(hash, stage, wgsl)` tuples into exact `WGPK` v1 binary bytes.

    Returns `(pack_bytes, [(offset, length), ...])` matching `shaders/src/pack.rs`
    and `web/workers/wgsl_pack_loader.js` byte-for-byte.
    """
    if isinstance(flags, bool) or not isinstance(flags, int) or flags < 0 or flags > U32_MAX:
        raise PackerError(f"flags must be an unsigned 32-bit integer, got {flags!r}", code="invalid_flags")
    if len(entries) > U32_MAX:
        raise PackerError("entry count exceeds u32::MAX", code="pack_overflow")

    encoded: list[tuple[int, ShaderStage, bytes]] = []
    total_payload = 0
    for shader_hash, stage, wgsl in entries:
        h = normalize_shader_hash(shader_hash)
        st = normalize_stage(stage, allow_unknown=False)
        assert st is not None
        payload = wgsl.encode("utf-8")
        if len(payload) > U32_MAX:
            raise PackerError("shader WGSL payload exceeds u32::MAX bytes", code="pack_overflow")
        total_payload += len(payload)
        encoded.append((h, st, payload))

    data_start = PACK_HEADER_BYTES + len(encoded) * PACK_ENTRY_BYTES
    total_size = data_start + total_payload
    if total_size > U32_MAX:
        raise PackerError("total WGPK pack size exceeds u32::MAX byte offsets", code="pack_overflow")

    out = bytearray(total_size)
    PACK_HEADER_STRUCT.pack_into(out, 0, PACK_MAGIC, PACK_VERSION, len(encoded), flags)

    ranges: list[tuple[int, int]] = []
    cursor = data_start
    for idx, (shader_hash, stage, payload) in enumerate(encoded):
        base = PACK_HEADER_BYTES + idx * PACK_ENTRY_BYTES
        length = len(payload)
        PACK_ENTRY_STRUCT.pack_into(
            out,
            base,
            shader_hash,
            int(stage),
            b"\x00\x00\x00",
            cursor,
            length,
        )
        out[cursor : cursor + length] = payload
        ranges.append((cursor, length))
        cursor += length

    return bytes(out), ranges


def build_fallback_pack(
    records: Sequence[ShaderInputRecord],
    *,
    profile: str = "default",
    wgsl_dir: Path | str | None = None,
    flags: int = 0,
) -> PackBuildResult:
    """Deduplicate, deterministically sort by `(hash, stage)`, resolve WGSL, and pack."""
    canonical_profile = normalize_profile(profile)
    profile_templates = BUILTIN_PROFILES[canonical_profile]

    warnings: list[str] = []
    override_index: WgslOverrideIndex | None = None
    if wgsl_dir is not None:
        override_index = index_wgsl_dir(wgsl_dir)
        if override_index.skipped_escaping_symlinks > 0:
            warnings.append(
                f"skipped {override_index.skipped_escaping_symlinks} symlink(s) escaping --wgsl-dir"
            )

    known_pairs: set[tuple[int, ShaderStage]] = set()
    known_hashes: set[int] = set()
    unknown_hashes_seen: list[int] = []
    duplicate_collapsed = 0

    for rec in records:
        if rec.stage is not None:
            pair = (rec.shader_hash, rec.stage)
            if pair in known_pairs:
                duplicate_collapsed += 1
            else:
                known_pairs.add(pair)
                known_hashes.add(rec.shader_hash)
        else:
            unknown_hashes_seen.append(rec.shader_hash)

    skipped_unknown_stage = 0
    seen_unresolved_unknown: set[int] = set()
    for shader_hash in unknown_hashes_seen:
        if shader_hash in known_hashes:
            # Resolved by another record with a known stage.
            duplicate_collapsed += 1
            continue
        if override_index is not None:
            inferred = override_index.infer_stage_for_unknown(shader_hash)
            if inferred is not None:
                inferred_stage, _override_file = inferred
                pair = (shader_hash, inferred_stage)
                if pair in known_pairs:
                    duplicate_collapsed += 1
                else:
                    known_pairs.add(pair)
                    known_hashes.add(shader_hash)
                continue
        if shader_hash in seen_unresolved_unknown:
            duplicate_collapsed += 1
        else:
            seen_unresolved_unknown.add(shader_hash)
            skipped_unknown_stage += 1

    if skipped_unknown_stage > 0:
        warnings.append(
            f"skipped {skipped_unknown_stage} unique shader hash(es) with stage 'unknown' "
            "(provide explicit --hashes or a stage-identifiable WGSL override in --wgsl-dir)"
        )

    # Deterministic ordering strictly by (hash, stage_code).
    sorted_pairs = sorted(known_pairs, key=lambda pair: (pair[0], int(pair[1])))

    stages_per_hash: dict[int, set[ShaderStage]] = {}
    for shader_hash, stage in sorted_pairs:
        stages_per_hash.setdefault(shader_hash, set()).add(stage)
    stage_conflict_hashes = sum(1 for stages in stages_per_hash.values() if len(stages) > 1)
    if stage_conflict_hashes > 0:
        warnings.append(
            f"{stage_conflict_hashes} shader hash(es) appear under multiple stages"
        )

    resolved_specs: list[tuple[int, ShaderStage, str, str, str | None, str | None]] = []
    wgsl_cache: dict[tuple[Path, ShaderStage], tuple[str, bool]] = {}

    for shader_hash, stage in sorted_pairs:
        override_hit = override_index.lookup(shader_hash, stage) if override_index is not None else None
        if override_hit is not None:
            cache_key = (override_hit.path, stage)
            cached = wgsl_cache.get(cache_key)
            if cached is None:
                raw_wgsl = read_override_wgsl(override_hit)
                if stage == ShaderStage.FRAGMENT:
                    wgsl, renamed = normalize_fragment_entry_point(raw_wgsl)
                else:
                    wgsl, renamed = raw_wgsl, False
                wgsl_cache[cache_key] = (wgsl, renamed)
            else:
                wgsl, renamed = cached
            if renamed:
                warnings.append(
                    f"normalized fragment entry point 'ps_main' -> 'fs_main' for {format_shader_hash(shader_hash)}"
                )
            resolved_specs.append(
                (shader_hash, stage, wgsl, "override", None, override_hit.filename)
            )
        else:
            wgsl = profile_templates[stage]
            resolved_specs.append(
                (shader_hash, stage, wgsl, "profile", canonical_profile, None)
            )

    pack_bytes, ranges = serialize_wgpk(
        [(h, st, wgsl) for h, st, wgsl, _src, _prof, _ov in resolved_specs],
        flags=flags,
    )
    pack_sha256 = hashlib.sha256(pack_bytes).hexdigest()

    packed_entries: list[PackedShaderEntry] = []
    for (shader_hash, stage, wgsl, source, prof, override_file), (offset, length) in zip(
        resolved_specs, ranges, strict=True
    ):
        packed_entries.append(
            PackedShaderEntry(
                shader_hash=shader_hash,
                stage=stage,
                wgsl=wgsl,
                source=source,
                profile=prof,
                override_file=override_file,
                entry_point=extract_entry_point(wgsl, stage),
                pack_offset=offset,
                pack_length=length,
            )
        )

    return PackBuildResult(
        pack_bytes=pack_bytes,
        pack_sha256=pack_sha256,
        profile=canonical_profile,
        entries=tuple(packed_entries),
        input_records_total=len(records),
        duplicate_entries_collapsed=duplicate_collapsed,
        skipped_unknown_stage_count=skipped_unknown_stage,
        stage_conflict_hash_count=stage_conflict_hashes,
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Atomic file writers
# ---------------------------------------------------------------------------


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Atomically write bytes to `path` via a same-directory temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text to `path`."""
    atomic_write_bytes(path, text.encode("utf-8"))


# ---------------------------------------------------------------------------
# Sanitized text summary & CLI entrypoint
# ---------------------------------------------------------------------------


def render_text_summary(report: Mapping[str, Any]) -> str:
    """Render a concise, strictly sanitized stdout summary."""
    summary = report["summary"]
    stages = summary["stage_counts"]
    lines = [
        f"Packed {summary['packed_entry_count']} shader(s) into {report['output_file']} "
        f"({report['pack_format']} v{report['pack_version']}, {report['pack_bytes']} bytes, "
        f"profile={report['profile']})",
        f"  stages: vertex={stages['vertex']} fragment={stages['fragment']} compute={stages['compute']}",
        f"  sources: override={summary['override_entry_count']} profile={summary['profile_entry_count']} "
        f"(deduped={summary['duplicate_entries_collapsed']}, skipped_unknown={summary['skipped_unknown_stage_count']})",
        f"  sha256: {report['pack_sha256']}",
    ]
    for warn in report.get("warnings", ()):
        lines.append(f"  warning: {warn}")
    return sanitize_text("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pack RDNA2 shader hashes (from inspect_rdna2_shaders.py --json and/or --hashes) "
            "into a deterministic WGPK v1 .wgsl.pack bundle using optional --wgsl-dir overrides "
            "and G34 WebGPU-validated fallback profiles."
        )
    )
    parser.add_argument(
        "--scan-json",
        action="append",
        default=[],
        metavar="PATH",
        help="Path to inspect_rdna2_shaders.py --json output or playgta6-pipeline-dump manifest.json (may be repeated).",
    )
    parser.add_argument(
        "--native-dir",
        action="append",
        default=[],
        metavar="PATH",
        help="Path to a native RDNA2 dump file or pipeline dump directory to hash whole-file (may be repeated).",
    )
    parser.add_argument(
        "--hashes",
        action="append",
        default=[],
        metavar="STAGE:HEX,...",
        help="Comma-separated explicit shader specs like vertex:0x1234,fragment:0x5678 (may be repeated).",
    )
    parser.add_argument(
        "--wgsl-dir",
        default=None,
        metavar="DIR",
        help="Optional directory of translated <stage>_<hash>.wgsl or <hash>.wgsl overrides.",
    )
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="default",
        help="Fallback WGSL profile matching G34 (default: default).",
    )
    parser.add_argument(
        "--out",
        required=True,
        metavar="PATH.wgsl.pack",
        help="Output .wgsl.pack path (must be in gitignored work/, local-data/, or outside repo).",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        metavar="PATH",
        help="Optional path to write a strictly sanitized JSON packing report.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the strictly sanitized JSON report to stdout.",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path | str = REPO_ROOT,
) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.scan_json and not args.hashes and not args.native_dir:
        print(
            "Error: at least one of --scan-json, --native-dir, or --hashes is required",
            file=sys.stderr,
        )
        return 2

    try:
        out_path = validate_output_path(args.out, flag_name="--out", repo_root=repo_root)
        report_path = (
            validate_output_path(args.report_json, flag_name="--report-json", repo_root=repo_root)
            if args.report_json is not None
            else None
        )

        records: list[ShaderInputRecord] = []
        scan_labels: list[str] = []
        for scan_arg in args.scan_json:
            scan_labels.append(sanitize_label(scan_arg))
            records.extend(load_scan_json_records(scan_arg))

        native_labels: list[str] = []
        for native_arg in args.native_dir:
            native_labels.append(sanitize_label(native_arg))
            records.extend(load_native_dir_records(native_arg))

        hash_records: list[ShaderInputRecord] = []
        if args.hashes:
            hash_records = parse_hashes_arg(args.hashes)
            records.extend(hash_records)

        result = build_fallback_pack(
            records,
            profile=args.profile,
            wgsl_dir=args.wgsl_dir,
        )

        report = result.to_report_dict(
            output_label=out_path.name,
            scan_json_labels=scan_labels,
            native_dir_labels=native_labels,
            hash_spec_count=len(hash_records),
            wgsl_dir_label=sanitize_label(args.wgsl_dir) if args.wgsl_dir is not None else None,
        )

        atomic_write_bytes(out_path, result.pack_bytes)
        if report_path is not None:
            atomic_write_text(report_path, json.dumps(report, indent=2) + "\n")

    except PackerError as exc:
        print(f"Error: {sanitize_text(exc.message)}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"Error: {sanitize_os_error(args.out, exc)}", file=sys.stderr)
        return 1

    if args.json:
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    else:
        sys.stdout.write(render_text_summary(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
