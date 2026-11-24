#!/usr/bin/env python3
"""Sanitized offline .pg6trace + .wgsl.pack (WGPK v1) WebGPU replay verifier CLI (Track 5 / W80).

Validates a local binary `.pg6trace` GPU command stream and an optional `WGPK` v1
`.wgsl.pack` archive completely offline and deterministically using Python 3 stdlib only:

1. **GPU Command Stream (`.pg6trace`):**
   - Strictly decodes and validates little-endian `GpuPacketHeader` (`<HHI`, 8 bytes)
     packet framing matching `runtime/abi/playgta6_abi.h`, `docs/gpu_commands.md`,
     `web/workers/wgpu_worker.js`, and `web/workers/render_state_packets.js`
     (v1 opcodes `0..21`, v2 opcodes `22..35`, plus forward-compatible skipping of
     unknown future opcodes `>= 36`).
   - Counts frames (`Present` opcode 7), `BeginFrame`, `ClearColor`,
     `CreateShaderModule`, `CreateRenderPipeline`, `CreateRenderPipeline2`,
     `Draw`, `DrawIndexed`, `WriteBuffer`, `CreateTexture2`, and all other opcodes.
   - Detects truncated headers, truncated payloads, short fixed payloads,
     truncated variable-length payloads, and invalid UTF-8 in `CreateShaderModule`.

2. **Shader Hash Cross-Check & WGSL Entry-Point Validation (`--shader-pack`):**
   - Strictly parses `WGPK` v1 `.wgsl.pack` archives matching `shaders/src/pack.rs`
     and `web/workers/wgsl_pack_loader.js` (header magic `"WGPK"`, version `1`,
     20-byte entry table, bounds/overlap validation, strict UTF-8 payload decoding).
   - Extracts every `CreateShaderModule` hash (`0x%016x` hash-only payload,
     `hash:`/`wgpk:` prefixed reference, pipeline-key identity for omitted payloads,
     or deterministic FNV-1a 64-bit hash of inline WGSL bytes).
   - Cross-checks all referenced shader hashes against the `WGPK` v1 entry table,
     reporting `matched_shader_hashes`, `missing_shader_hashes`, and `stage_counts`.
   - Validates WGSL entry points (with comment stripping) for every resolved shader:
     - `vertex`   -> `@vertex`   `fn vs_main`
     - `fragment` -> `@fragment` `fn fs_main`
     - `compute`  -> `@compute`  `fn cs_main`
     Instantly catches legacy `ps_main` vs `fs_main` mismatches.

3. **Vertex & Draw Readiness:**
   - Tracks GPU buffer allocations, `WriteBuffer` uploads, shader modules, render
     pipelines (`CreateRenderPipeline` / `CreateRenderPipeline2`), pass resets,
     `SetPipeline`, `SetVertexBuffer`, and `SetIndexBuffer`.
   - Confirms that at least one real `Draw` or `DrawIndexed` packet with a resolved
     pipeline, resolved shaders, and non-empty vertex data precedes `Present`.

4. **Strict Sanitization (`--json`, `--check-replay-assets`, and text mode):**
   - ZERO absolute host paths, drive letters, UNC paths, POSIX `/home/`/`/Users/`
     paths, active OS usernames, raw WGSL source, or raw binary capture bytes ever
     appear in stdout/stderr or `--json` output.

Usage:
    python tools/verify_frame1_replay.py <capture.pg6trace> [--shader-pack <pack.wgsl.pack>]
        [--check-replay-assets] [--json] [--strict]
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# ABI & binary format constants (docs/gpu_commands.md, playgta6_abi.h, pack.rs)
# ---------------------------------------------------------------------------

SCHEMA_NAME = "playgta6-frame1-replay-verifier"
SCHEMA_VERSION = 1

# GpuPacketHeader: u16 opcode, u16 flags, u32 payload_bytes
PACKET_HEADER_STRUCT = struct.Struct("<HHI")
PACKET_HEADER_BYTES = PACKET_HEADER_STRUCT.size  # 8 bytes

# v1 payload structs
CLEAR_COLOR_STRUCT = struct.Struct("<ffff")  # 16
CREATE_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16
WRITE_BUFFER_STRUCT = struct.Struct("<IIII")  # 16 + data_len
CREATE_SHADER_MODULE_STRUCT = struct.Struct("<IIII")  # 16 + code_len
DRAW_STRUCT = struct.Struct("<IIII")  # 16
CREATE_RENDER_PIPELINE_STRUCT = struct.Struct("<IIIIII")  # 24
SET_PIPELINE_STRUCT = struct.Struct("<I")  # 4
SET_VERTEX_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16
DESTROY_RESOURCE_STRUCT = struct.Struct("<I")  # 4
CREATE_TEXTURE_STRUCT = struct.Struct("<IIIIIIII")  # 32
WRITE_TEXTURE_STRUCT = struct.Struct("<IIIIIIIIIIII")  # 48 + data_len
CREATE_SAMPLER_STRUCT = struct.Struct("<IIIIIIII")  # 32
CREATE_BIND_GROUP_STRUCT = struct.Struct("<IIII")  # 16 + 32 * entry_count
BIND_GROUP_ENTRY_BYTES = 32
SET_BIND_GROUP_STRUCT = struct.Struct("<II")  # 8
SET_INDEX_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16
DRAW_INDEXED_STRUCT = struct.Struct("<IIIiI")  # 20
CREATE_COMPUTE_PIPELINE_STRUCT = struct.Struct("<IIII")  # 16
SET_COMPUTE_PIPELINE_STRUCT = struct.Struct("<I")  # 4
DISPATCH_COMPUTE_STRUCT = struct.Struct("<III")  # 12

# v2 payload structs
CREATE_BIND_GROUP_LAYOUT_STRUCT = struct.Struct("<IIIIQ")  # 24 + 24 * entry_count
BIND_GROUP_LAYOUT_ENTRY_BYTES = 24
CREATE_PIPELINE_LAYOUT_STRUCT = struct.Struct("<IIIIQ")  # 24 + 4 * count
CREATE_RENDER_PIPELINE2_HEAD_STRUCT = struct.Struct("<IIII")  # 16 + 376-byte key
RENDER_PIPELINE_KEY_BYTES = 376
CREATE_RENDER_PIPELINE2_BYTES = (
    CREATE_RENDER_PIPELINE2_HEAD_STRUCT.size + RENDER_PIPELINE_KEY_BYTES
)  # 392
KEY_HEAD_STRUCT = struct.Struct("<QQII")  # vs_shader, ps_shader, root_sig, color_target_count
KEY_VERTEX_COUNTS_OFFSET = 172
KEY_VERTEX_BUFFERS_OFFSET = 180
KEY_VERTEX_BUFFER_STRUCT = struct.Struct("<IBB2x")  # 8 bytes: stride, step_mode, attr_count
CREATE_COMPUTE_PIPELINE2_STRUCT = struct.Struct("<IIIIQII")  # 32
BEGIN_RENDER_PASS_BYTES = 368
SET_VIEWPORT_BYTES = 24
SET_SCISSOR_RECT_BYTES = 16
SET_BLEND_CONSTANT_BYTES = 16
SET_STENCIL_REFERENCE_BYTES = 4
CREATE_BIND_GROUP2_STRUCT = struct.Struct("<IIII")  # 16 + 48 * entry_count
BIND_GROUP_ENTRY2_BYTES = 48
SET_BIND_GROUP2_STRUCT = struct.Struct("<IIII")  # 16 + 4 * dynamic_offset_count
CREATE_SAMPLER2_BYTES = 24
CREATE_TEXTURE2_STRUCT = struct.Struct("<IIIIIIIBBBB8s")  # 40
CREATE_TEXTURE2_BYTES = CREATE_TEXTURE2_STRUCT.size

MANIFEST_REFERENCE_FLAG = 0x1

# WGPK v1 binary container constants (shaders/src/pack.rs & wgsl_pack_loader.js)
WGPK_MAGIC = b"WGPK"
WGPK_VERSION = 1
WGPK_HEADER_STRUCT = struct.Struct("<4sIII")
WGPK_HEADER_BYTES = WGPK_HEADER_STRUCT.size  # 16 bytes
WGPK_ENTRY_STRUCT = struct.Struct("<QB3sII")
WGPK_ENTRY_BYTES = WGPK_ENTRY_STRUCT.size  # 20 bytes

FNV1A_64_OFFSET_BASIS = 0xCBF29CE484222325
FNV1A_64_PRIME = 0x00000100000001B3
U64_MASK = 0xFFFFFFFFFFFFFFFF
U32_MAX = 0xFFFFFFFF

STAGE_ORDER: tuple[str, ...] = ("vertex", "fragment", "compute")

REQUIRED_ENTRY_POINTS: dict[str, str] = {
    "vertex": "vs_main",
    "fragment": "fs_main",
    "compute": "cs_main",
}

# Packet ABI stage codes (`GpuShaderStage` in playgta6_abi.h: 0/1/2)
PACKET_STAGE_NAMES: dict[int, str] = {
    0: "vertex",
    1: "fragment",
    2: "compute",
}

# WGPK v1 stage codes (`ShaderStage` in shaders/src/convert.rs: 1/2/3)
WGPK_STAGE_NAMES: dict[int, str] = {
    1: "vertex",
    2: "fragment",
    3: "compute",
}

WGPK_STAGE_CODES: dict[str, int] = {name: code for code, name in WGPK_STAGE_NAMES.items()}


class GpuOpcode(IntEnum):
    """Frozen GPU command stream opcodes from runtime/abi/playgta6_abi.h."""

    Nop = 0
    BeginFrame = 1
    ClearColor = 2
    CreateBuffer = 3
    WriteBuffer = 4
    CreateShaderModule = 5
    Draw = 6
    Present = 7
    CreateRenderPipeline = 8
    SetPipeline = 9
    SetVertexBuffer = 10
    DestroyResource = 11
    CreateTexture = 12
    WriteTexture = 13
    CreateSampler = 14
    CreateBindGroup = 15
    SetBindGroup = 16
    SetIndexBuffer = 17
    DrawIndexed = 18
    CreateComputePipeline = 19
    SetComputePipeline = 20
    DispatchCompute = 21
    CreateBindGroupLayout = 22
    CreatePipelineLayout = 23
    CreateRenderPipeline2 = 24
    CreateComputePipeline2 = 25
    BeginRenderPass = 26
    EndRenderPass = 27
    SetViewport = 28
    SetScissorRect = 29
    SetBlendConstant = 30
    SetStencilReference = 31
    CreateBindGroup2 = 32
    SetBindGroup2 = 33
    CreateSampler2 = 34
    CreateTexture2 = 35


OPCODE_NAMES: dict[int, str] = {op.value: op.name for op in GpuOpcode}

KEY_PACKET_COUNT_NAMES: tuple[str, ...] = (
    "BeginFrame",
    "ClearColor",
    "CreateBuffer",
    "WriteBuffer",
    "CreateShaderModule",
    "CreateRenderPipeline",
    "CreateRenderPipeline2",
    "CreateComputePipeline",
    "CreateComputePipeline2",
    "SetPipeline",
    "SetVertexBuffer",
    "SetIndexBuffer",
    "Draw",
    "DrawIndexed",
    "DispatchCompute",
    "CreateTexture",
    "CreateTexture2",
    "WriteTexture",
    "BeginRenderPass",
    "EndRenderPass",
    "Present",
)

# ---------------------------------------------------------------------------
# Sanitization helpers (zero host paths, usernames, or raw payload leaks)
# ---------------------------------------------------------------------------

_FILE_URL_RE = re.compile(r"\bfile://[^\s\"'<>|,;()\[\]]+", re.IGNORECASE)
_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes|proc|run|srv)/[^\s\"'<>|,;()\[\]]*"
)


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
    text = _FILE_URL_RE.sub("<path>", text)
    for home_str in _home_path_variants():
        if home_str:
            text = re.sub(re.escape(home_str), "<path>", text, flags=re.IGNORECASE)
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


def sanitize_source_label(source: str | Path | None) -> str:
    """Return a strictly sanitized basename-only source label (never a host path)."""
    if source is None:
        return "<memory>"
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
    """Recursively sanitize all strings/paths and replace raw bytes with size placeholders."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Path):
        return sanitize_source_label(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]
    return value


def _sanitize_os_error(source: str | Path, exc: OSError) -> str:
    label = sanitize_source_label(source)
    reason = exc.strerror or type(exc).__name__
    return sanitize_text(f"failed to read {label}: {reason}")


# ---------------------------------------------------------------------------
# Hashing & hash-payload helpers
# ---------------------------------------------------------------------------

_PREFIXED_HASH_RE = re.compile(r"^(?:hash|wgpk)\s*:\s*(?:0[xX])?([0-9a-fA-F]{1,16})$", re.IGNORECASE)
_HEX0X_RE = re.compile(r"^0[xX]([0-9a-fA-F]{1,16})$")
_BARE64_HEX_RE = re.compile(r"^([0-9a-fA-F]{16})$")


def fnv1a_64(data: bytes | bytearray | memoryview | str) -> int:
    """Compute unsigned 64-bit FNV-1a hash matching `shaders/src/convert.rs` & `wgpu_worker.js`."""
    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    digest = FNV1A_64_OFFSET_BASIS
    for byte in raw:
        digest ^= byte
        digest = (digest * FNV1A_64_PRIME) & U64_MASK
    return digest


def format_hash_hex(value: int) -> str:
    """Format an unsigned 64-bit integer as canonical `0x` + 16 lowercase hex digits."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > U64_MASK:
        raise ValueError(f"hash value out of u64 range: {value!r}")
    return f"0x{value:016x}"


def parse_hash_only_shader_payload(text: str) -> str | None:
    """Match `parseHashOnlyShaderPayload` in `web/workers/wgpu_worker.js`.

    Recognizes:
    - `0x<1..16 hex digits>`
    - `hash:<0x?hex>` or `wgpk:<0x?hex>`
    - bare 16-digit hex string
    Returns canonical `0x%016x` or None if payload is not a hash-only reference.
    """
    trimmed = text.strip()
    if not trimmed:
        return None
    for pattern in (_PREFIXED_HASH_RE, _HEX0X_RE, _BARE64_HEX_RE):
        match = pattern.match(trimmed)
        if match:
            return format_hash_hex(int(match.group(1), 16))
    return None


# ---------------------------------------------------------------------------
# WGSL comment stripper & entry-point analyzer
# ---------------------------------------------------------------------------

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WS_RE = re.compile(r"\s+")
_BUILTIN_VERTEX_INDEX_RE = re.compile(r"@\s*builtin\s*\(\s*vertex_index\s*\)")
_LOCATION_ATTR_RE = re.compile(r"@\s*location\s*\(")


@dataclass(frozen=True)
class WgslEntryPoint:
    """Single WGSL shader entry point discovered in module-scope code."""

    stage: str  # "vertex" | "fragment" | "compute"
    name: str
    params_text: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"stage": self.stage, "name": self.name}


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


def _skip_ws(text: str, pos: int) -> int:
    match = _WS_RE.match(text, pos)
    return match.end() if match else pos


def _consume_balanced(text: str, pos: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    n = len(text)
    if pos >= n or text[pos] != open_ch:
        return "", pos
    depth = 1
    start = pos + 1
    i = start
    while i < n and depth > 0:
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
        i += 1
    end = i - 1 if depth == 0 else i
    return text[start:end], i


def extract_wgsl_entry_points(source: str) -> list[WgslEntryPoint]:
    """Scan module-scope WGSL declarations for `@vertex`/`@fragment`/`@compute` `fn` entry points."""
    code = strip_wgsl_comments(source)
    entry_points: list[WgslEntryPoint] = []
    pending_attrs: list[tuple[str, str | None]] = []

    pos = 0
    n = len(code)
    while pos < n:
        pos = _skip_ws(code, pos)
        if pos >= n:
            break
        ch = code[pos]

        if ch == "{":
            _, pos = _consume_balanced(code, pos, "{", "}")
            pending_attrs.clear()
            continue

        if ch == ";":
            pending_attrs.clear()
            pos += 1
            continue

        if ch == "@":
            pos += 1
            pos = _skip_ws(code, pos)
            name_match = _IDENT_RE.match(code, pos)
            if not name_match:
                pos += 1
                continue
            attr_name = name_match.group(0)
            pos = _skip_ws(code, name_match.end())
            attr_args: str | None = None
            if pos < n and code[pos] == "(":
                attr_args, pos = _consume_balanced(code, pos, "(", ")")
            pending_attrs.append((attr_name, attr_args))
            continue

        ident_match = _IDENT_RE.match(code, pos)
        if not ident_match:
            pos += 1
            continue

        keyword = ident_match.group(0)
        pos = ident_match.end()

        if keyword == "fn":
            pos = _skip_ws(code, pos)
            fn_match = _IDENT_RE.match(code, pos)
            fn_name = fn_match.group(0) if fn_match else ""
            if fn_match:
                pos = fn_match.end()

            pos = _skip_ws(code, pos)
            params_text = ""
            if pos < n and code[pos] == "(":
                params_text, pos = _consume_balanced(code, pos, "(", ")")

            fn_stages: list[str] = []
            for attr_name, attr_args in pending_attrs:
                if attr_name in REQUIRED_ENTRY_POINTS:
                    if attr_name not in fn_stages:
                        fn_stages.append(attr_name)
                elif attr_name == "stage" and attr_args is not None:
                    legacy = attr_args.strip().lower()
                    if legacy in REQUIRED_ENTRY_POINTS and legacy not in fn_stages:
                        fn_stages.append(legacy)

            if fn_name:
                for st in fn_stages:
                    entry_points.append(
                        WgslEntryPoint(stage=st, name=fn_name, params_text=params_text)
                    )

            pending_attrs.clear()

            # Advance past return type and function body `{ ... }`.
            paren_depth = 0
            while pos < n:
                c = code[pos]
                if c == "(":
                    paren_depth += 1
                    pos += 1
                elif c == ")":
                    paren_depth = max(0, paren_depth - 1)
                    pos += 1
                elif c == "{" and paren_depth == 0:
                    _, pos = _consume_balanced(code, pos, "{", "}")
                    break
                elif c == ";" and paren_depth == 0:
                    pos += 1
                    break
                else:
                    pos += 1
            continue

        if keyword in (
            "var",
            "struct",
            "alias",
            "const",
            "override",
            "enable",
            "requires",
            "diagnostic",
            "const_assert",
        ):
            pending_attrs.clear()
            while pos < n:
                c = code[pos]
                if c == "{":
                    _, pos = _consume_balanced(code, pos, "{", "}")
                    break
                if c == ";":
                    pos += 1
                    break
                pos += 1
            continue

    return entry_points


def _vertex_shader_input_profile(wgsl_source: str) -> tuple[bool, bool]:
    """Return `(uses_vertex_index, requires_location_inputs)` for a vertex WGSL module."""
    stripped = strip_wgsl_comments(wgsl_source)
    eps = [ep for ep in extract_wgsl_entry_points(wgsl_source) if ep.stage == "vertex"]
    if not eps:
        return False, False
    uses_vertex_index = bool(_BUILTIN_VERTEX_INDEX_RE.search(stripped))
    # Check if the vertex entry point parameter list itself declares @location(...)
    # or references an input struct with @location(...) fields (ignoring return type).
    requires_location = any(_LOCATION_ATTR_RE.search(ep.params_text) for ep in eps)
    if not requires_location:
        # Check struct types referenced in vertex entry point parameters.
        struct_locs: dict[str, bool] = {}
        for m in re.finditer(r"\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{", stripped):
            sname = m.group(1)
            body, _ = _consume_balanced(stripped, m.end() - 1, "{", "}")
            struct_locs[sname] = bool(_LOCATION_ATTR_RE.search(body))
        for ep in eps:
            for ident in _IDENT_RE.findall(ep.params_text):
                if struct_locs.get(ident):
                    requires_location = True
                    break
    return uses_vertex_index, requires_location


def check_wgsl_entry_point(
    wgsl_source: str,
    expected_stage: str,
    shader_hash: str,
) -> tuple[bool, dict[str, Any] | None]:
    """Verify `wgsl_source` declares the required entry point for `expected_stage`.

    Returns `(ok, mismatch_dict_or_none)`.
    """
    expected_ep = REQUIRED_ENTRY_POINTS.get(expected_stage)
    if expected_ep is None:
        return False, {
            "hash": shader_hash,
            "stage": expected_stage,
            "expected_entry_point": None,
            "found_entry_points": [],
            "reason": f"unknown shader stage {expected_stage!r} for {shader_hash}",
        }

    found = extract_wgsl_entry_points(wgsl_source)
    stage_matches = [ep for ep in found if ep.stage == expected_stage]
    if any(ep.name == expected_ep for ep in stage_matches):
        return True, None

    found_dicts = [ep.to_dict() for ep in found]
    if stage_matches:
        found_names = ", ".join(ep.name for ep in stage_matches)
        reason = (
            f"{expected_stage} shader {shader_hash} declares @{expected_stage} fn {found_names}, "
            f"expected {expected_ep}"
        )
    elif found:
        other = ", ".join(f"@{ep.stage} fn {ep.name}" for ep in found)
        reason = (
            f"{expected_stage} shader {shader_hash} has no @{expected_stage} entry point "
            f"(found {other}); expected {expected_ep}"
        )
    else:
        reason = (
            f"{expected_stage} shader {shader_hash} declares no WGSL entry point; "
            f"expected @{expected_stage} fn {expected_ep}"
        )

    return False, {
        "hash": shader_hash,
        "stage": expected_stage,
        "expected_entry_point": expected_ep,
        "found_entry_points": found_dicts,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# WGPK v1 (.wgsl.pack) parser & builder (matching shaders/src/pack.rs)
# ---------------------------------------------------------------------------


class WgpkFormatError(ValueError):
    """Raised when a WGPK v1 `.wgsl.pack` archive is corrupt or truncated."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True)
class WgpkEntry:
    """Single decoded entry from a `WGPK` v1 `.wgsl.pack` archive."""

    hash_int: int
    hash_hex: str
    stage: str  # "vertex" | "fragment" | "compute"
    stage_code: int
    offset: int
    length: int
    wgsl: str


@dataclass(frozen=True)
class WgpkArchive:
    """Parsed `WGPK` v1 `.wgsl.pack` archive."""

    version: int
    flags: int
    byte_length: int
    entries: tuple[WgpkEntry, ...]
    by_hash: Mapping[str, WgpkEntry]
    by_hash_stage: Mapping[tuple[str, str], WgpkEntry]
    stage_counts: Mapping[str, int]
    duplicate_hash_count: int = 0

    def lookup(self, hash_hex: str, stage: str | None = None) -> WgpkEntry | None:
        key = hash_hex.lower()
        if stage is not None:
            exact = self.by_hash_stage.get((key, stage))
            if exact is not None:
                return exact
        return self.by_hash.get(key)


def parse_wgpk_bytes(data: bytes | bytearray | memoryview) -> WgpkArchive:
    """Strictly parse and validate a binary `WGPK` v1 `.wgsl.pack` buffer."""
    raw = bytes(data)
    total_len = len(raw)

    if total_len < WGPK_HEADER_BYTES:
        raise WgpkFormatError(
            "wgpk_truncated_header",
            f"WGPK pack truncated: {total_len} bytes (minimum header size: {WGPK_HEADER_BYTES})",
        )

    magic, version, entry_count, flags = WGPK_HEADER_STRUCT.unpack_from(raw, 0)
    if magic != WGPK_MAGIC:
        magic_hex = " ".join(f"0x{b:02x}" for b in magic)
        raise WgpkFormatError(
            "wgpk_invalid_magic",
            f"invalid WGPK pack magic [{magic_hex}] (expected 'WGPK')",
        )

    if version != WGPK_VERSION:
        raise WgpkFormatError(
            "wgpk_unsupported_version",
            f"unsupported WGPK pack version {version} (expected {WGPK_VERSION})",
        )

    data_start = WGPK_HEADER_BYTES + entry_count * WGPK_ENTRY_BYTES
    if total_len < data_start:
        raise WgpkFormatError(
            "wgpk_truncated_table",
            f"WGPK pack truncated: {total_len} bytes (header + {entry_count} entries require {data_start} bytes)",
        )

    entries: list[WgpkEntry] = []
    by_hash: dict[str, WgpkEntry] = {}
    by_hash_stage: dict[tuple[str, str], WgpkEntry] = {}
    stage_counts: dict[str, int] = {stage: 0 for stage in STAGE_ORDER}
    ranges: list[tuple[int, int, int]] = []
    duplicate_hashes = 0

    for idx in range(entry_count):
        base = WGPK_HEADER_BYTES + idx * WGPK_ENTRY_BYTES
        hash_int, stage_code, _reserved, offset, length = WGPK_ENTRY_STRUCT.unpack_from(raw, base)
        stage_name = WGPK_STAGE_NAMES.get(stage_code)
        if stage_name is None:
            raise WgpkFormatError(
                "wgpk_invalid_stage",
                f"invalid WGPK shader stage code {stage_code} at entry {idx} (expected 1..3)",
            )

        end = offset + length
        if offset < data_start or end > total_len:
            raise WgpkFormatError(
                "wgpk_out_of_bounds",
                f"WGPK entry {idx} payload [{offset}..+{length}] is out of bounds "
                f"(valid payload range: [{data_start}..{total_len}])",
            )

        if length > 0:
            ranges.append((offset, end, idx))

        try:
            wgsl_text = raw[offset:end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WgpkFormatError(
                "wgpk_invalid_utf8",
                f"WGPK entry {idx} payload is not valid UTF-8: {exc.reason}",
            ) from exc

        hash_hex = format_hash_hex(hash_int)
        entry = WgpkEntry(
            hash_int=hash_int,
            hash_hex=hash_hex,
            stage=stage_name,
            stage_code=stage_code,
            offset=offset,
            length=length,
            wgsl=wgsl_text,
        )
        entries.append(entry)
        stage_counts[stage_name] += 1
        if hash_hex in by_hash:
            duplicate_hashes += 1
        else:
            by_hash[hash_hex] = entry
        by_hash_stage.setdefault((hash_hex, stage_name), entry)

    if len(ranges) > 1:
        ranges.sort(key=lambda item: (item[0], item[1]))
        for prev, curr in zip(ranges, ranges[1:]):
            if curr[0] < prev[1]:
                raise WgpkFormatError(
                    "wgpk_overlapping_payloads",
                    f"WGPK overlapping payload ranges between entry {prev[2]} "
                    f"[{prev[0]}..{prev[1]}) and entry {curr[2]} [{curr[0]}..{curr[1]})",
                )

    return WgpkArchive(
        version=version,
        flags=flags,
        byte_length=total_len,
        entries=tuple(entries),
        by_hash=by_hash,
        by_hash_stage=by_hash_stage,
        stage_counts=stage_counts,
        duplicate_hash_count=duplicate_hashes,
    )


def build_wgpk_bytes(
    entries: Sequence[tuple[int | str | None, int | str, str | bytes]],
    *,
    flags: int = 0,
) -> bytes:
    """Serialize `(hash_or_none, stage, wgsl)` tuples into a `WGPK` v1 binary buffer."""
    encoded: list[tuple[int, int, bytes]] = []
    for idx, (raw_hash, raw_stage, raw_wgsl) in enumerate(entries):
        if isinstance(raw_stage, str):
            stage_code = WGPK_STAGE_CODES.get(raw_stage.strip().lower())
            if stage_code is None:
                raise ValueError(f"entry {idx}: invalid stage name {raw_stage!r}")
        elif isinstance(raw_stage, int) and raw_stage in WGPK_STAGE_NAMES:
            stage_code = raw_stage
        else:
            raise ValueError(f"entry {idx}: invalid stage {raw_stage!r}")

        payload = raw_wgsl.encode("utf-8") if isinstance(raw_wgsl, str) else bytes(raw_wgsl)
        if raw_hash is None:
            hash_int = fnv1a_64(payload)
        elif isinstance(raw_hash, str):
            cleaned = raw_hash.strip()
            hex_part = cleaned[2:] if cleaned.lower().startswith("0x") else cleaned
            hash_int = int(hex_part, 16)
        else:
            hash_int = int(raw_hash)
        if hash_int < 0 or hash_int > U64_MASK:
            raise ValueError(f"entry {idx}: hash out of u64 range")
        encoded.append((hash_int, stage_code, payload))

    data_start = WGPK_HEADER_BYTES + len(encoded) * WGPK_ENTRY_BYTES
    total_size = data_start + sum(len(p) for _, _, p in encoded)
    if total_size > U32_MAX:
        raise ValueError("WGPK archive exceeds 4 GiB offset limit")

    out = bytearray(total_size)
    WGPK_HEADER_STRUCT.pack_into(out, 0, WGPK_MAGIC, WGPK_VERSION, len(encoded), flags)
    cursor = data_start
    for idx, (hash_int, stage_code, payload) in enumerate(encoded):
        base = WGPK_HEADER_BYTES + idx * WGPK_ENTRY_BYTES
        WGPK_ENTRY_STRUCT.pack_into(
            out, base, hash_int, stage_code, b"\x00\x00\x00", cursor, len(payload)
        )
        out[cursor : cursor + len(payload)] = payload
        cursor += len(payload)
    return bytes(out)


# ---------------------------------------------------------------------------
# Internal GPU state tracking during .pg6trace replay verification
# ---------------------------------------------------------------------------


@dataclass
class _RawPacket:
    index: int
    offset: int
    opcode: int
    flags: int
    payload: memoryview


@dataclass
class _BufferState:
    resource_id: int
    usage_flags: int
    byte_size: int
    bytes_written: int = 0
    write_count: int = 0


@dataclass
class _ShaderModuleState:
    resource_id: int
    stage: str
    hash_hex: str
    hash_source: str  # "hash_ref" | "inline_fnv1a64" | "pipeline_key" | "sidecar" | "empty"
    code_len: int
    resolved: bool = False
    resolved_via: str = "unresolved"  # "pack" | "inline" | "unresolved"
    entry_point_ok: bool = False
    stage_ok: bool = True
    uses_vertex_index: bool = False
    requires_location_inputs: bool = False


@dataclass
class _RenderPipelineState:
    resource_id: int
    vs_shader_id: int
    fs_shader_id: int
    requires_vertex_buffer: bool
    required_vb_slots: tuple[int, ...]
    is_v2: bool


@dataclass
class _DrawCheckRecord:
    packet_index: int
    kind: str  # "Draw" | "DrawIndexed"
    count: int
    instance_count: int
    pipeline_resolved: bool
    shaders_resolved: bool
    entry_points_valid: bool
    has_vertex_data: bool
    procedural_vertex: bool
    issues: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return (
            self.count > 0
            and self.instance_count > 0
            and self.pipeline_resolved
            and self.shaders_resolved
            and self.entry_points_valid
            and self.has_vertex_data
            and len(self.issues) == 0
        )


def _decode_packet_framing(
    data: bytes,
) -> tuple[list[_RawPacket], int, list[str], list[str]]:
    """Decode raw `GpuPacketHeader` framing from `.pg6trace` bytes."""
    mv = memoryview(data)
    total_len = len(mv)
    packets: list[_RawPacket] = []
    errors: list[str] = []
    warnings: list[str] = []

    if total_len == 0:
        errors.append("trace stream is empty (0 bytes)")
        return packets, 0, errors, warnings

    offset = 0
    packet_index = 0
    while offset < total_len:
        remaining = total_len - offset
        if remaining < PACKET_HEADER_BYTES:
            errors.append(
                f"truncated packet header at byte offset {offset}: "
                f"{remaining} trailing byte(s) (need {PACKET_HEADER_BYTES})"
            )
            break

        opcode, flags, payload_bytes = PACKET_HEADER_STRUCT.unpack_from(mv, offset)
        payload_start = offset + PACKET_HEADER_BYTES
        payload_end = payload_start + payload_bytes
        if payload_end > total_len:
            op_name = OPCODE_NAMES.get(opcode, f"Unknown({opcode})")
            errors.append(
                f"truncated {op_name} (opcode {opcode}) payload at packet {packet_index} "
                f"(offset {offset}): header declares {payload_bytes} byte(s), "
                f"only {total_len - payload_start} available"
            )
            break

        packets.append(
            _RawPacket(
                index=packet_index,
                offset=offset,
                opcode=opcode,
                flags=flags,
                payload=mv[payload_start:payload_end],
            )
        )
        offset = payload_end
        packet_index += 1

    return packets, offset, errors, warnings


def _extract_pipeline_key_hashes(packets: Sequence[_RawPacket]) -> dict[int, str]:
    """Pre-scan CreateRenderPipeline2 / CreateComputePipeline2 keys for shader module hashes."""
    module_hashes: dict[int, str] = {}
    for pkt in packets:
        plen = len(pkt.payload)
        if pkt.opcode == GpuOpcode.CreateRenderPipeline2 and plen >= CREATE_RENDER_PIPELINE2_BYTES:
            _pid, vs_id, fs_id, _layout_id = CREATE_RENDER_PIPELINE2_HEAD_STRUCT.unpack_from(
                pkt.payload, 0
            )
            vs_shader, ps_shader, _root_sig, _rt_count = KEY_HEAD_STRUCT.unpack_from(
                pkt.payload, CREATE_RENDER_PIPELINE2_HEAD_STRUCT.size
            )
            if vs_id != 0 and vs_shader != 0:
                module_hashes.setdefault(vs_id, format_hash_hex(vs_shader))
            if fs_id != 0 and ps_shader != 0:
                module_hashes.setdefault(fs_id, format_hash_hex(ps_shader))
        elif (
            pkt.opcode == GpuOpcode.CreateComputePipeline2
            and plen >= CREATE_COMPUTE_PIPELINE2_STRUCT.size
        ):
            _pid, cs_id, _layout_id, _res, cs_shader, _root_sig, _res2 = (
                CREATE_COMPUTE_PIPELINE2_STRUCT.unpack_from(pkt.payload, 0)
            )
            if cs_id != 0 and cs_shader != 0:
                module_hashes.setdefault(cs_id, format_hash_hex(cs_shader))
    return module_hashes


def _parse_sidecar_omitted_hashes(sidecar_data: Mapping[str, Any] | None) -> dict[int, str]:
    """Extract `{packet_index: fnv1a64_hex}` for omitted CreateShaderModule packets from sidecar."""
    if not sidecar_data:
        return {}
    omitted = sidecar_data.get("omitted")
    if not isinstance(omitted, list):
        return {}
    result: dict[int, str] = {}
    for item in omitted:
        if not isinstance(item, Mapping):
            continue
        pkt_idx = item.get("packet")
        opcode = item.get("opcode")
        fnv = item.get("fnv1a64")
        if (
            isinstance(pkt_idx, int)
            and opcode == int(GpuOpcode.CreateShaderModule)
            and isinstance(fnv, str)
        ):
            parsed = parse_hash_only_shader_payload(fnv)
            if parsed is not None:
                result[pkt_idx] = parsed
    return result


def _validate_fixed_payload(
    pkt: _RawPacket,
    expected_bytes: int,
    errors: list[str],
    warnings: list[str],
) -> bool:
    plen = len(pkt.payload)
    op_name = OPCODE_NAMES.get(pkt.opcode, f"Unknown({pkt.opcode})")
    if plen < expected_bytes:
        errors.append(
            f"packet {pkt.index} ({op_name}) payload too short: "
            f"expected {expected_bytes} byte(s), got {plen}"
        )
        return False
    if plen > expected_bytes:
        warnings.append(
            f"packet {pkt.index} ({op_name}) has {plen - expected_bytes} unexpected trailing byte(s)"
        )
    return True


# ---------------------------------------------------------------------------
# Core offline replay verification engine
# ---------------------------------------------------------------------------


def verify_frame1_replay_bytes(
    trace_bytes: bytes | bytearray | memoryview,
    *,
    shader_pack_bytes: bytes | bytearray | memoryview | None = None,
    trace_label: str = "<memory>",
    shader_pack_label: str | None = None,
    sidecar: Mapping[str, Any] | None = None,
    check_replay_assets: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Verify a `.pg6trace` stream and optional `WGPK` v1 `.wgsl.pack` completely offline.

    Returns a strictly sanitized dictionary ready for JSON serialization.
    """
    errors: list[str] = []
    warnings: list[str] = []

    raw_trace = bytes(trace_bytes)
    packets, consumed_bytes, frame_errors, frame_warnings = _decode_packet_framing(raw_trace)
    errors.extend(frame_errors)
    warnings.extend(frame_warnings)

    # Parse optional WGPK v1 shader pack
    pack_provided = shader_pack_bytes is not None
    pack_archive: WgpkArchive | None = None
    pack_valid = False
    if pack_provided:
        try:
            pack_archive = parse_wgpk_bytes(shader_pack_bytes)  # type: ignore[arg-type]
            pack_valid = True
        except WgpkFormatError as exc:
            errors.append(f"shader pack error ({exc.code}): {exc.message}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"shader pack parse error: {exc}")

    # Pre-scan pipeline keys & optional sidecar for empty/omitted CreateShaderModule hashes
    pipeline_key_hashes = _extract_pipeline_key_hashes(packets)
    sidecar_omitted_hashes = _parse_sidecar_omitted_hashes(sidecar)

    # Counters & state tables
    opcode_histogram: Counter[str] = Counter()
    unknown_opcode_count = 0

    buffers: dict[int, _BufferState] = {}
    textures: set[int] = set()
    shader_modules: dict[int, _ShaderModuleState] = {}
    render_pipelines: dict[int, _RenderPipelineState] = {}
    compute_pipelines: dict[int, int] = {}

    # Shader cross-check tracking
    referenced_hashes: list[str] = []
    referenced_hash_set: set[str] = set()
    matched_hashes: set[str] = set()
    missing_hashes: set[str] = set()
    inline_hashes: set[str] = set()
    unresolved_hashes: set[str] = set()
    stage_counts: dict[str, int] = {stage: 0 for stage in STAGE_ORDER}
    entry_point_mismatches: list[dict[str, Any]] = []
    stage_mismatches: list[dict[str, Any]] = []
    checked_entry_point_keys: set[tuple[str, str]] = set()
    entry_point_checks_passed = 0

    # Frame & draw tracking
    present_count = 0
    packets_since_present = 0
    active_pipeline_id: int | None = None
    bound_vertex_buffers: dict[int, tuple[int, int]] = {}  # slot -> (buffer_id, offset)
    bound_index_buffer: tuple[int, int, int] | None = None  # (buffer_id, fmt, offset)

    total_draws = 0
    total_draw_indexed = 0
    pending_frame_draws: list[_DrawCheckRecord] = []
    draws_before_present: list[_DrawCheckRecord] = []
    vertex_buffer_writes = 0
    vertex_bytes_written = 0

    def _reset_pass_bindings() -> None:
        nonlocal active_pipeline_id, bound_index_buffer
        active_pipeline_id = None
        bound_vertex_buffers.clear()
        bound_index_buffer = None

    def _evaluate_draw(
        pkt_index: int,
        kind: str,
        count: int,
        instance_count: int,
    ) -> _DrawCheckRecord:
        draw_issues: list[str] = []
        if count <= 0:
            draw_issues.append(f"packet {pkt_index} ({kind}) has zero vertex/index count")
        if instance_count <= 0:
            draw_issues.append(f"packet {pkt_index} ({kind}) has zero instance_count")

        if active_pipeline_id is None:
            draw_issues.append(f"packet {pkt_index} ({kind}) issued without active SetPipeline")
            return _DrawCheckRecord(
                packet_index=pkt_index,
                kind=kind,
                count=count,
                instance_count=instance_count,
                pipeline_resolved=False,
                shaders_resolved=False,
                entry_points_valid=False,
                has_vertex_data=False,
                procedural_vertex=False,
                issues=draw_issues,
            )

        pipeline = render_pipelines.get(active_pipeline_id)
        if pipeline is None:
            draw_issues.append(
                f"packet {pkt_index} ({kind}) references unknown pipeline id {active_pipeline_id}"
            )
            return _DrawCheckRecord(
                packet_index=pkt_index,
                kind=kind,
                count=count,
                instance_count=instance_count,
                pipeline_resolved=False,
                shaders_resolved=False,
                entry_points_valid=False,
                has_vertex_data=False,
                procedural_vertex=False,
                issues=draw_issues,
            )

        vs_mod = shader_modules.get(pipeline.vs_shader_id)
        fs_mod = shader_modules.get(pipeline.fs_shader_id) if pipeline.fs_shader_id != 0 else None

        shaders_resolved = True
        entry_points_valid = True

        if vs_mod is None:
            shaders_resolved = False
            entry_points_valid = False
            draw_issues.append(
                f"packet {pkt_index} ({kind}) pipeline {pipeline.resource_id} references unknown "
                f"vertex shader module id {pipeline.vs_shader_id}"
            )
        else:
            if not vs_mod.resolved:
                shaders_resolved = False
                draw_issues.append(
                    f"packet {pkt_index} ({kind}) vertex shader {vs_mod.hash_hex} is unresolved"
                )
            if not vs_mod.entry_point_ok or not vs_mod.stage_ok:
                entry_points_valid = False
                draw_issues.append(
                    f"packet {pkt_index} ({kind}) vertex shader {vs_mod.hash_hex} failed "
                    f"entry-point/stage validation"
                )

        if pipeline.fs_shader_id != 0:
            if fs_mod is None:
                shaders_resolved = False
                entry_points_valid = False
                draw_issues.append(
                    f"packet {pkt_index} ({kind}) pipeline {pipeline.resource_id} references "
                    f"unknown fragment shader module id {pipeline.fs_shader_id}"
                )
            else:
                if not fs_mod.resolved:
                    shaders_resolved = False
                    draw_issues.append(
                        f"packet {pkt_index} ({kind}) fragment shader {fs_mod.hash_hex} is unresolved"
                    )
                if not fs_mod.entry_point_ok or not fs_mod.stage_ok:
                    entry_points_valid = False
                    draw_issues.append(
                        f"packet {pkt_index} ({kind}) fragment shader {fs_mod.hash_hex} failed "
                        f"entry-point/stage validation"
                    )

        # Vertex/index buffer readiness
        has_vertex_data = True
        procedural_vertex = False

        if kind == "DrawIndexed":
            if bound_index_buffer is None:
                has_vertex_data = False
                draw_issues.append(
                    f"packet {pkt_index} (DrawIndexed) issued without SetIndexBuffer"
                )
            else:
                ib_id, _ib_fmt, ib_off = bound_index_buffer
                ib = buffers.get(ib_id)
                if ib is None:
                    has_vertex_data = False
                    draw_issues.append(
                        f"packet {pkt_index} (DrawIndexed) references unknown index buffer id {ib_id}"
                    )
                elif ib.byte_size == 0 or ib_off >= ib.byte_size:
                    has_vertex_data = False
                    draw_issues.append(
                        f"packet {pkt_index} (DrawIndexed) index buffer id {ib_id} has invalid size/offset"
                    )
                elif ib.bytes_written == 0:
                    has_vertex_data = False
                    draw_issues.append(
                        f"packet {pkt_index} (DrawIndexed) index buffer id {ib_id} has 0 bytes written"
                    )

        if pipeline.requires_vertex_buffer:
            for slot in pipeline.required_vb_slots:
                vb_binding = bound_vertex_buffers.get(slot)
                if vb_binding is None:
                    has_vertex_data = False
                    draw_issues.append(
                        f"packet {pkt_index} ({kind}) pipeline {pipeline.resource_id} requires "
                        f"vertex buffer at slot {slot}, but none is bound"
                    )
                    continue
                vb_id, vb_off = vb_binding
                vb = buffers.get(vb_id)
                if vb is None:
                    has_vertex_data = False
                    draw_issues.append(
                        f"packet {pkt_index} ({kind}) slot {slot} references unknown vertex buffer id {vb_id}"
                    )
                elif vb.byte_size == 0 or vb_off >= vb.byte_size:
                    has_vertex_data = False
                    draw_issues.append(
                        f"packet {pkt_index} ({kind}) slot {slot} vertex buffer id {vb_id} has "
                        f"empty size or out-of-range offset"
                    )
                elif vb.bytes_written == 0:
                    has_vertex_data = False
                    draw_issues.append(
                        f"packet {pkt_index} ({kind}) slot {slot} vertex buffer id {vb_id} has "
                        f"no uploaded vertex data (0 bytes written)"
                    )
        else:
            # Pipeline declares no vertex buffer layouts (stride == 0 / vertex_buffer_count == 0).
            if bound_vertex_buffers:
                any_populated = False
                for slot, (vb_id, vb_off) in sorted(bound_vertex_buffers.items()):
                    vb = buffers.get(vb_id)
                    if vb is None or vb.byte_size == 0 or vb_off >= vb.byte_size:
                        has_vertex_data = False
                        draw_issues.append(
                            f"packet {pkt_index} ({kind}) bound vertex buffer slot {slot} "
                            f"(id {vb_id}) is missing or out of range"
                        )
                    elif vb.bytes_written == 0:
                        has_vertex_data = False
                        draw_issues.append(
                            f"packet {pkt_index} ({kind}) bound vertex buffer slot {slot} "
                            f"(id {vb_id}) has 0 bytes written"
                        )
                    else:
                        any_populated = True
                if not any_populated and has_vertex_data:
                    has_vertex_data = False
            else:
                if vs_mod is not None and vs_mod.requires_location_inputs:
                    has_vertex_data = False
                    draw_issues.append(
                        f"packet {pkt_index} ({kind}) vertex shader {vs_mod.hash_hex} declares "
                        f"@location inputs but no vertex buffer is bound"
                    )
                elif vs_mod is not None and vs_mod.uses_vertex_index:
                    procedural_vertex = True
                elif vs_mod is not None and vs_mod.resolved:
                    procedural_vertex = True
                else:
                    has_vertex_data = False

        return _DrawCheckRecord(
            packet_index=pkt_index,
            kind=kind,
            count=count,
            instance_count=instance_count,
            pipeline_resolved=True,
            shaders_resolved=shaders_resolved,
            entry_points_valid=entry_points_valid,
            has_vertex_data=has_vertex_data,
            procedural_vertex=procedural_vertex,
            issues=draw_issues,
        )

    # Walk every packet in stream order
    for pkt in packets:
        op = pkt.opcode
        op_name = OPCODE_NAMES.get(op)
        if op_name is None:
            unknown_opcode_count += 1
            opcode_histogram[f"Unknown({op})"] += 1
            warnings.append(
                f"packet {pkt.index} at offset {pkt.offset} uses unknown opcode {op} "
                f"({len(pkt.payload)} payload byte(s))"
            )
            packets_since_present += 1
            continue

        opcode_histogram[op_name] += 1
        packets_since_present += 1
        payload = pkt.payload
        plen = len(payload)

        if op == GpuOpcode.Nop:
            _validate_fixed_payload(pkt, 0, errors, warnings)
        elif op == GpuOpcode.BeginFrame:
            _validate_fixed_payload(pkt, 0, errors, warnings)
            _reset_pass_bindings()
        elif op == GpuOpcode.ClearColor:
            _validate_fixed_payload(pkt, CLEAR_COLOR_STRUCT.size, errors, warnings)
        elif op == GpuOpcode.CreateBuffer:
            if _validate_fixed_payload(pkt, CREATE_BUFFER_STRUCT.size, errors, warnings):
                res_id, usage_flags, byte_size = CREATE_BUFFER_STRUCT.unpack_from(payload, 0)
                buffers[res_id] = _BufferState(
                    resource_id=res_id,
                    usage_flags=usage_flags,
                    byte_size=byte_size,
                )
        elif op == GpuOpcode.WriteBuffer:
            if plen < WRITE_BUFFER_STRUCT.size:
                errors.append(
                    f"packet {pkt.index} (WriteBuffer) payload too short: "
                    f"expected at least {WRITE_BUFFER_STRUCT.size} bytes, got {plen}"
                )
            else:
                res_id, _dst_offset, data_len, _reserved = WRITE_BUFFER_STRUCT.unpack_from(
                    payload, 0
                )
                expected_total = WRITE_BUFFER_STRUCT.size + data_len
                if expected_total > plen:
                    errors.append(
                        f"packet {pkt.index} (WriteBuffer id={res_id}) declares data_len={data_len}, "
                        f"but only {plen - WRITE_BUFFER_STRUCT.size} byte(s) available"
                    )
                else:
                    if expected_total < plen:
                        warnings.append(
                            f"packet {pkt.index} (WriteBuffer id={res_id}) has "
                            f"{plen - expected_total} trailing byte(s)"
                        )
                    buf = buffers.get(res_id)
                    if buf is not None:
                        buf.bytes_written += data_len
                        buf.write_count += 1
                        # GPUBufferUsage.VERTEX == 0x0020
                        if (buf.usage_flags & 0x0020) != 0 and data_len > 0:
                            vertex_buffer_writes += 1
                            vertex_bytes_written += data_len
        elif op == GpuOpcode.CreateShaderModule:
            if plen < CREATE_SHADER_MODULE_STRUCT.size:
                errors.append(
                    f"packet {pkt.index} (CreateShaderModule) payload too short: "
                    f"expected at least {CREATE_SHADER_MODULE_STRUCT.size} bytes, got {plen}"
                )
                continue
            res_id, stage_code, code_len, _reserved = CREATE_SHADER_MODULE_STRUCT.unpack_from(
                payload, 0
            )
            stage_name = PACKET_STAGE_NAMES.get(stage_code)
            if stage_name is None:
                errors.append(
                    f"packet {pkt.index} (CreateShaderModule id={res_id}) has unknown stage code {stage_code}"
                )
                stage_name = f"unknown({stage_code})"
            else:
                stage_counts[stage_name] += 1

            expected_total = CREATE_SHADER_MODULE_STRUCT.size + code_len
            if expected_total > plen:
                errors.append(
                    f"packet {pkt.index} (CreateShaderModule id={res_id}) declares "
                    f"code_len={code_len}, but only {plen - CREATE_SHADER_MODULE_STRUCT.size} byte(s) available"
                )
                continue
            if expected_total < plen:
                warnings.append(
                    f"packet {pkt.index} (CreateShaderModule id={res_id}) has "
                    f"{plen - expected_total} trailing byte(s)"
                )

            code_raw = bytes(payload[CREATE_SHADER_MODULE_STRUCT.size : expected_total])
            try:
                code_text = code_raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                errors.append(
                    f"packet {pkt.index} (CreateShaderModule id={res_id}) contains invalid UTF-8: {exc.reason}"
                )
                continue

            trimmed = code_text.strip()
            hash_only = parse_hash_only_shader_payload(trimmed)
            inline_wgsl: str | None = None

            if hash_only is not None:
                shader_hash = hash_only
                hash_source = "hash_ref"
            elif len(trimmed) > 0:
                shader_hash = format_hash_hex(fnv1a_64(code_raw))
                hash_source = "inline_fnv1a64"
                inline_wgsl = code_text
            elif res_id in pipeline_key_hashes:
                shader_hash = pipeline_key_hashes[res_id]
                hash_source = "pipeline_key"
            elif pkt.index in sidecar_omitted_hashes:
                shader_hash = sidecar_omitted_hashes[pkt.index]
                hash_source = "sidecar"
            else:
                shader_hash = format_hash_hex(fnv1a_64(f"empty:{stage_name}:{res_id}"))
                hash_source = "empty"

            if shader_hash not in referenced_hash_set:
                referenced_hash_set.add(shader_hash)
                referenced_hashes.append(shader_hash)

            resolved = False
            resolved_via = "unresolved"
            wgsl_to_check: str | None = None
            stage_ok = True

            if pack_archive is not None:
                pack_entry = pack_archive.lookup(shader_hash, stage_name)
                if pack_entry is not None:
                    matched_hashes.add(shader_hash)
                    resolved = True
                    resolved_via = "pack"
                    wgsl_to_check = pack_entry.wgsl
                    if stage_name in REQUIRED_ENTRY_POINTS and pack_entry.stage != stage_name:
                        stage_ok = False
                        stage_mismatches.append(
                            {
                                "hash": shader_hash,
                                "trace_stage": stage_name,
                                "pack_stage": pack_entry.stage,
                                "reason": (
                                    f"shader {shader_hash} declared as {stage_name} in trace "
                                    f"but packed as {pack_entry.stage} in WGPK"
                                ),
                            }
                        )
                else:
                    missing_hashes.add(shader_hash)
                    unresolved_hashes.add(shader_hash)
                    if inline_wgsl is not None:
                        inline_hashes.add(shader_hash)
            elif inline_wgsl is not None:
                inline_hashes.add(shader_hash)
                resolved = True
                resolved_via = "inline"
                wgsl_to_check = inline_wgsl
            else:
                unresolved_hashes.add(shader_hash)

            entry_point_ok = False
            uses_vertex_index = False
            requires_location_inputs = False

            if wgsl_to_check is not None and stage_name in REQUIRED_ENTRY_POINTS:
                check_key = (shader_hash, stage_name)
                ep_ok, mismatch = check_wgsl_entry_point(wgsl_to_check, stage_name, shader_hash)
                entry_point_ok = ep_ok
                if check_key not in checked_entry_point_keys:
                    checked_entry_point_keys.add(check_key)
                    if ep_ok:
                        entry_point_checks_passed += 1
                    elif mismatch is not None:
                        entry_point_mismatches.append(mismatch)
                if stage_name == "vertex":
                    uses_vertex_index, requires_location_inputs = _vertex_shader_input_profile(
                        wgsl_to_check
                    )
            elif inline_wgsl is not None and stage_name in REQUIRED_ENTRY_POINTS:
                # Even if missing from pack, inspect inline WGSL for entry-point diagnostics
                ep_ok, mismatch = check_wgsl_entry_point(inline_wgsl, stage_name, shader_hash)
                entry_point_ok = ep_ok
                check_key = (shader_hash, stage_name)
                if check_key not in checked_entry_point_keys:
                    checked_entry_point_keys.add(check_key)
                    if ep_ok:
                        entry_point_checks_passed += 1
                    elif mismatch is not None:
                        entry_point_mismatches.append(mismatch)
                if stage_name == "vertex":
                    uses_vertex_index, requires_location_inputs = _vertex_shader_input_profile(
                        inline_wgsl
                    )

            shader_modules[res_id] = _ShaderModuleState(
                resource_id=res_id,
                stage=stage_name,
                hash_hex=shader_hash,
                hash_source=hash_source,
                code_len=code_len,
                resolved=resolved,
                resolved_via=resolved_via,
                entry_point_ok=entry_point_ok,
                stage_ok=stage_ok,
                uses_vertex_index=uses_vertex_index,
                requires_location_inputs=requires_location_inputs,
            )

        elif op == GpuOpcode.Draw:
            if _validate_fixed_payload(pkt, DRAW_STRUCT.size, errors, warnings):
                vcount, icount, _first_v, _first_i = DRAW_STRUCT.unpack_from(payload, 0)
                total_draws += 1
                pending_frame_draws.append(_evaluate_draw(pkt.index, "Draw", vcount, icount))

        elif op == GpuOpcode.Present:
            _validate_fixed_payload(pkt, 0, errors, warnings)
            present_count += 1
            draws_before_present.extend(pending_frame_draws)
            pending_frame_draws.clear()
            packets_since_present = 0
            _reset_pass_bindings()

        elif op == GpuOpcode.CreateRenderPipeline:
            if _validate_fixed_payload(pkt, CREATE_RENDER_PIPELINE_STRUCT.size, errors, warnings):
                res_id, vs_id, fs_id, vertex_stride, _topology, _pflags = (
                    CREATE_RENDER_PIPELINE_STRUCT.unpack_from(payload, 0)
                )
                req_vb = vertex_stride > 0
                render_pipelines[res_id] = _RenderPipelineState(
                    resource_id=res_id,
                    vs_shader_id=vs_id,
                    fs_shader_id=fs_id,
                    requires_vertex_buffer=req_vb,
                    required_vb_slots=(0,) if req_vb else (),
                    is_v2=False,
                )

        elif op == GpuOpcode.SetPipeline:
            if _validate_fixed_payload(pkt, SET_PIPELINE_STRUCT.size, errors, warnings):
                (pipeline_id,) = SET_PIPELINE_STRUCT.unpack_from(payload, 0)
                active_pipeline_id = pipeline_id

        elif op == GpuOpcode.SetVertexBuffer:
            if _validate_fixed_payload(pkt, SET_VERTEX_BUFFER_STRUCT.size, errors, warnings):
                slot, buffer_id, vb_offset = SET_VERTEX_BUFFER_STRUCT.unpack_from(payload, 0)
                bound_vertex_buffers[slot] = (buffer_id, vb_offset)

        elif op == GpuOpcode.DestroyResource:
            if _validate_fixed_payload(pkt, DESTROY_RESOURCE_STRUCT.size, errors, warnings):
                (res_id,) = DESTROY_RESOURCE_STRUCT.unpack_from(payload, 0)
                buffers.pop(res_id, None)
                textures.discard(res_id)
                shader_modules.pop(res_id, None)
                render_pipelines.pop(res_id, None)
                compute_pipelines.pop(res_id, None)

        elif op == GpuOpcode.CreateTexture:
            if _validate_fixed_payload(pkt, CREATE_TEXTURE_STRUCT.size, errors, warnings):
                (res_id,) = struct.unpack_from("<I", payload, 0)
                textures.add(res_id)

        elif op == GpuOpcode.WriteTexture:
            if plen < WRITE_TEXTURE_STRUCT.size:
                errors.append(
                    f"packet {pkt.index} (WriteTexture) payload too short: "
                    f"expected at least {WRITE_TEXTURE_STRUCT.size} bytes, got {plen}"
                )
            else:
                fields = WRITE_TEXTURE_STRUCT.unpack_from(payload, 0)
                data_len = fields[10]
                expected_total = WRITE_TEXTURE_STRUCT.size + data_len
                if expected_total > plen:
                    errors.append(
                        f"packet {pkt.index} (WriteTexture) declares data_len={data_len}, "
                        f"but only {plen - WRITE_TEXTURE_STRUCT.size} byte(s) available"
                    )
                elif expected_total < plen:
                    warnings.append(
                        f"packet {pkt.index} (WriteTexture) has {plen - expected_total} trailing byte(s)"
                    )

        elif op == GpuOpcode.CreateSampler:
            _validate_fixed_payload(pkt, CREATE_SAMPLER_STRUCT.size, errors, warnings)

        elif op == GpuOpcode.CreateBindGroup:
            if plen < CREATE_BIND_GROUP_STRUCT.size:
                errors.append(
                    f"packet {pkt.index} (CreateBindGroup) payload too short: "
                    f"expected at least {CREATE_BIND_GROUP_STRUCT.size} bytes, got {plen}"
                )
            else:
                _id, _pid, _grp, entry_count = CREATE_BIND_GROUP_STRUCT.unpack_from(payload, 0)
                expected_total = CREATE_BIND_GROUP_STRUCT.size + entry_count * BIND_GROUP_ENTRY_BYTES
                if expected_total > plen:
                    errors.append(
                        f"packet {pkt.index} (CreateBindGroup) declares {entry_count} entries "
                        f"requiring {expected_total} bytes, got {plen}"
                    )
                elif expected_total < plen:
                    warnings.append(
                        f"packet {pkt.index} (CreateBindGroup) has {plen - expected_total} trailing byte(s)"
                    )

        elif op == GpuOpcode.SetBindGroup:
            _validate_fixed_payload(pkt, SET_BIND_GROUP_STRUCT.size, errors, warnings)

        elif op == GpuOpcode.SetIndexBuffer:
            if _validate_fixed_payload(pkt, SET_INDEX_BUFFER_STRUCT.size, errors, warnings):
                buffer_id, index_fmt, ib_offset = SET_INDEX_BUFFER_STRUCT.unpack_from(payload, 0)
                bound_index_buffer = (buffer_id, index_fmt, ib_offset)

        elif op == GpuOpcode.DrawIndexed:
            if _validate_fixed_payload(pkt, DRAW_INDEXED_STRUCT.size, errors, warnings):
                icount, inst_count, _first_i, _base_v, _first_inst = (
                    DRAW_INDEXED_STRUCT.unpack_from(payload, 0)
                )
                total_draw_indexed += 1
                pending_frame_draws.append(
                    _evaluate_draw(pkt.index, "DrawIndexed", icount, inst_count)
                )

        elif op == GpuOpcode.CreateComputePipeline:
            if _validate_fixed_payload(pkt, CREATE_COMPUTE_PIPELINE_STRUCT.size, errors, warnings):
                res_id, cs_id, _pflags, _res = CREATE_COMPUTE_PIPELINE_STRUCT.unpack_from(
                    payload, 0
                )
                compute_pipelines[res_id] = cs_id

        elif op == GpuOpcode.SetComputePipeline:
            _validate_fixed_payload(pkt, SET_COMPUTE_PIPELINE_STRUCT.size, errors, warnings)

        elif op == GpuOpcode.DispatchCompute:
            _validate_fixed_payload(pkt, DISPATCH_COMPUTE_STRUCT.size, errors, warnings)

        elif op == GpuOpcode.CreateBindGroupLayout:
            if plen < CREATE_BIND_GROUP_LAYOUT_STRUCT.size:
                errors.append(
                    f"packet {pkt.index} (CreateBindGroupLayout) payload too short: "
                    f"expected at least {CREATE_BIND_GROUP_LAYOUT_STRUCT.size} bytes, got {plen}"
                )
            else:
                _id, entry_count, lflags, _res, _cid = (
                    CREATE_BIND_GROUP_LAYOUT_STRUCT.unpack_from(payload, 0)
                )
                if (lflags & MANIFEST_REFERENCE_FLAG) == 0:
                    expected_total = (
                        CREATE_BIND_GROUP_LAYOUT_STRUCT.size
                        + entry_count * BIND_GROUP_LAYOUT_ENTRY_BYTES
                    )
                    if expected_total > plen:
                        errors.append(
                            f"packet {pkt.index} (CreateBindGroupLayout) declares {entry_count} "
                            f"entries requiring {expected_total} bytes, got {plen}"
                        )
                    elif expected_total < plen:
                        warnings.append(
                            f"packet {pkt.index} (CreateBindGroupLayout) has "
                            f"{plen - expected_total} trailing byte(s)"
                        )

        elif op == GpuOpcode.CreatePipelineLayout:
            if plen < CREATE_PIPELINE_LAYOUT_STRUCT.size:
                errors.append(
                    f"packet {pkt.index} (CreatePipelineLayout) payload too short: "
                    f"expected at least {CREATE_PIPELINE_LAYOUT_STRUCT.size} bytes, got {plen}"
                )
            else:
                _id, count, lflags, _res, _cid = CREATE_PIPELINE_LAYOUT_STRUCT.unpack_from(
                    payload, 0
                )
                if (lflags & MANIFEST_REFERENCE_FLAG) == 0:
                    expected_total = CREATE_PIPELINE_LAYOUT_STRUCT.size + count * 4
                    if expected_total > plen:
                        errors.append(
                            f"packet {pkt.index} (CreatePipelineLayout) declares {count} layouts "
                            f"requiring {expected_total} bytes, got {plen}"
                        )
                    elif expected_total < plen:
                        warnings.append(
                            f"packet {pkt.index} (CreatePipelineLayout) has "
                            f"{plen - expected_total} trailing byte(s)"
                        )

        elif op == GpuOpcode.CreateRenderPipeline2:
            if _validate_fixed_payload(pkt, CREATE_RENDER_PIPELINE2_BYTES, errors, warnings):
                res_id, vs_id, fs_id, _layout_id = (
                    CREATE_RENDER_PIPELINE2_HEAD_STRUCT.unpack_from(payload, 0)
                )
                key_base = CREATE_RENDER_PIPELINE2_HEAD_STRUCT.size
                vb_count, _va_count = struct.unpack_from(
                    "<II", payload, key_base + KEY_VERTEX_COUNTS_OFFSET
                )
                vb_count = min(vb_count, 8)
                active_slots: list[int] = []
                for slot_idx in range(vb_count):
                    stride, step_mode, _attr_count = KEY_VERTEX_BUFFER_STRUCT.unpack_from(
                        payload,
                        key_base + KEY_VERTEX_BUFFERS_OFFSET + slot_idx * KEY_VERTEX_BUFFER_STRUCT.size,
                    )
                    if step_mode != 0 and stride > 0:
                        active_slots.append(slot_idx)
                render_pipelines[res_id] = _RenderPipelineState(
                    resource_id=res_id,
                    vs_shader_id=vs_id,
                    fs_shader_id=fs_id,
                    requires_vertex_buffer=len(active_slots) > 0,
                    required_vb_slots=tuple(active_slots),
                    is_v2=True,
                )

        elif op == GpuOpcode.CreateComputePipeline2:
            if _validate_fixed_payload(pkt, CREATE_COMPUTE_PIPELINE2_STRUCT.size, errors, warnings):
                res_id, cs_id, _layout_id, _res, _cs_hash, _root_sig, _res2 = (
                    CREATE_COMPUTE_PIPELINE2_STRUCT.unpack_from(payload, 0)
                )
                compute_pipelines[res_id] = cs_id

        elif op == GpuOpcode.BeginRenderPass:
            _validate_fixed_payload(pkt, BEGIN_RENDER_PASS_BYTES, errors, warnings)
            _reset_pass_bindings()

        elif op == GpuOpcode.EndRenderPass:
            _validate_fixed_payload(pkt, 0, errors, warnings)
            _reset_pass_bindings()

        elif op == GpuOpcode.SetViewport:
            _validate_fixed_payload(pkt, SET_VIEWPORT_BYTES, errors, warnings)

        elif op == GpuOpcode.SetScissorRect:
            _validate_fixed_payload(pkt, SET_SCISSOR_RECT_BYTES, errors, warnings)

        elif op == GpuOpcode.SetBlendConstant:
            _validate_fixed_payload(pkt, SET_BLEND_CONSTANT_BYTES, errors, warnings)

        elif op == GpuOpcode.SetStencilReference:
            _validate_fixed_payload(pkt, SET_STENCIL_REFERENCE_BYTES, errors, warnings)

        elif op == GpuOpcode.CreateBindGroup2:
            if plen < CREATE_BIND_GROUP2_STRUCT.size:
                errors.append(
                    f"packet {pkt.index} (CreateBindGroup2) payload too short: "
                    f"expected at least {CREATE_BIND_GROUP2_STRUCT.size} bytes, got {plen}"
                )
            else:
                _id, _lid, entry_count, _res = CREATE_BIND_GROUP2_STRUCT.unpack_from(payload, 0)
                expected_total = (
                    CREATE_BIND_GROUP2_STRUCT.size + entry_count * BIND_GROUP_ENTRY2_BYTES
                )
                if expected_total > plen:
                    errors.append(
                        f"packet {pkt.index} (CreateBindGroup2) declares {entry_count} entries "
                        f"requiring {expected_total} bytes, got {plen}"
                    )
                elif expected_total < plen:
                    warnings.append(
                        f"packet {pkt.index} (CreateBindGroup2) has {plen - expected_total} trailing byte(s)"
                    )

        elif op == GpuOpcode.SetBindGroup2:
            if plen < SET_BIND_GROUP2_STRUCT.size:
                errors.append(
                    f"packet {pkt.index} (SetBindGroup2) payload too short: "
                    f"expected at least {SET_BIND_GROUP2_STRUCT.size} bytes, got {plen}"
                )
            else:
                _grp, _bg_id, dyn_count, _res = SET_BIND_GROUP2_STRUCT.unpack_from(payload, 0)
                expected_total = SET_BIND_GROUP2_STRUCT.size + dyn_count * 4
                if expected_total > plen:
                    errors.append(
                        f"packet {pkt.index} (SetBindGroup2) declares {dyn_count} dynamic offsets "
                        f"requiring {expected_total} bytes, got {plen}"
                    )
                elif expected_total < plen:
                    warnings.append(
                        f"packet {pkt.index} (SetBindGroup2) has {plen - expected_total} trailing byte(s)"
                    )

        elif op == GpuOpcode.CreateSampler2:
            _validate_fixed_payload(pkt, CREATE_SAMPLER2_BYTES, errors, warnings)

        elif op == GpuOpcode.CreateTexture2:
            if _validate_fixed_payload(pkt, CREATE_TEXTURE2_BYTES, errors, warnings):
                (res_id,) = struct.unpack_from("<I", payload, 0)
                textures.add(res_id)

    # Aggregate checks & diagnostics
    trace_parse_ok = len(errors) == 0 if not pack_provided else (
        len([e for e in errors if not e.startswith("shader pack ")]) == 0
    )
    has_present_frame = present_count >= 1
    if not has_present_frame:
        errors.append("capture has 0 Present frames (need >= 1 Present packet)")

    shader_pack_ok = (not pack_provided) or pack_valid
    if pack_provided and pack_valid and missing_hashes:
        errors.append(
            f"{len(missing_hashes)} shader hash(es) referenced by capture are missing from WGPK pack"
        )
    elif not pack_provided and unresolved_hashes:
        errors.append(
            f"{len(unresolved_hashes)} hash-referenced shader(s) require --shader-pack to resolve"
        )

    if entry_point_mismatches:
        for mm in entry_point_mismatches:
            errors.append(f"entry-point mismatch: {mm['reason']}")

    if stage_mismatches:
        for sm in stage_mismatches:
            errors.append(f"shader stage mismatch: {sm['reason']}")

    # Evaluate draw readiness before Present
    resolved_draws_before_present = sum(
        1 for d in draws_before_present if d.pipeline_resolved and d.shaders_resolved
    )
    ready_draws_before_present = sum(1 for d in draws_before_present if d.ready)
    procedural_draws_before_present = sum(
        1 for d in draws_before_present if d.ready and d.procedural_vertex
    )
    draw_issues: list[str] = []
    for d in draws_before_present:
        draw_issues.extend(d.issues)

    draw_ready_before_present = ready_draws_before_present >= 1
    if not draw_ready_before_present:
        if len(draws_before_present) == 0:
            if total_draws + total_draw_indexed > 0:
                msg = "Draw/DrawIndexed packet(s) found only after final Present (none precede Present)"
            else:
                msg = "no Draw or DrawIndexed packet precedes Present"
            draw_issues.append(msg)
            errors.append(msg)
        else:
            errors.append(
                f"0 of {len(draws_before_present)} Draw/DrawIndexed packet(s) before Present "
                f"had a resolved pipeline, valid shaders, and non-empty vertex data"
            )

    all_shaders_resolved = (
        len(missing_hashes) == 0
        and len(unresolved_hashes) == 0
        and shader_pack_ok
    )
    entry_points_valid = len(entry_point_mismatches) == 0 and len(stage_mismatches) == 0

    if check_replay_assets and pack_archive is not None and pack_archive.duplicate_hash_count > 0:
        warnings.append(
            f"WGPK pack contains {pack_archive.duplicate_hash_count} duplicate shader hash entry(ies)"
        )

    if strict and warnings:
        errors.extend(f"strict warning: {w}" for w in warnings)

    checklist = {
        "trace_parse_ok": trace_parse_ok,
        "has_present_frame": has_present_frame,
        "shader_pack_ok": shader_pack_ok,
        "all_shaders_resolved": all_shaders_resolved,
        "entry_points_valid": entry_points_valid,
        "draw_ready_before_present": draw_ready_before_present,
    }
    overall_ok = all(checklist.values()) and len(errors) == 0

    matched_sorted = sorted(matched_hashes)
    missing_sorted = sorted(missing_hashes)
    inline_sorted = sorted(inline_hashes)
    unresolved_sorted = sorted(unresolved_hashes)
    ref_count = len(referenced_hashes)
    if pack_archive is not None:
        match_rate = (len(matched_sorted) / ref_count) if ref_count > 0 else 1.0
    else:
        resolved_count = ref_count - len(unresolved_sorted)
        match_rate = (resolved_count / ref_count) if ref_count > 0 else 1.0

    packet_counts = {name: int(opcode_histogram.get(name, 0)) for name in KEY_PACKET_COUNT_NAMES}

    report: dict[str, Any] = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "ok": overall_ok,
        "status": "pass" if overall_ok else "fail",
        "check_replay_assets": bool(check_replay_assets),
        "strict": bool(strict),
        "inputs": {
            "trace": sanitize_source_label(trace_label),
            "shader_pack": sanitize_source_label(shader_pack_label)
            if pack_provided
            else None,
        },
        "trace": {
            "valid": trace_parse_ok,
            "byte_length": len(raw_trace),
            "consumed_bytes": consumed_bytes,
            "packet_count": len(packets),
            "frame_count": present_count + (1 if packets_since_present > 0 else 0),
            "present_count": present_count,
            "complete_frames": present_count,
            "has_trailing_incomplete_frame": packets_since_present > 0,
            "unknown_opcode_count": unknown_opcode_count,
            "packet_counts": packet_counts,
            "opcode_histogram": dict(sorted(opcode_histogram.items())),
        },
        "shaders": {
            "pack_provided": pack_provided,
            "pack_valid": pack_valid if pack_provided else None,
            "pack_version": pack_archive.version if pack_archive is not None else None,
            "pack_flags": pack_archive.flags if pack_archive is not None else None,
            "pack_entry_count": len(pack_archive.entries) if pack_archive is not None else 0,
            "pack_stage_counts": dict(pack_archive.stage_counts)
            if pack_archive is not None
            else {stage: 0 for stage in STAGE_ORDER},
            "referenced_shader_count": ref_count,
            "referenced_shader_hashes": sorted(referenced_hash_set),
            "matched_shader_hashes": matched_sorted,
            "missing_shader_hashes": missing_sorted,
            "inline_shader_hashes": inline_sorted,
            "unresolved_shader_hashes": unresolved_sorted,
            "stage_counts": dict(stage_counts),
            "match_rate": round(match_rate, 6),
            "entry_point_checks_passed": entry_point_checks_passed,
            "entry_point_mismatch_count": len(entry_point_mismatches),
            "entry_point_mismatches": entry_point_mismatches,
            "stage_mismatch_count": len(stage_mismatches),
            "stage_mismatches": stage_mismatches,
        },
        "draw_readiness": {
            "ready": draw_ready_before_present,
            "total_draws": total_draws,
            "total_draw_indexed": total_draw_indexed,
            "draws_before_present": len(draws_before_present),
            "resolved_draws_before_present": resolved_draws_before_present,
            "draws_with_vertex_data_before_present": ready_draws_before_present,
            "procedural_draws_before_present": procedural_draws_before_present,
            "vertex_buffer_writes": vertex_buffer_writes,
            "vertex_bytes_written": vertex_bytes_written,
            "issues": draw_issues,
        },
        "checklist": checklist,
        "errors": errors,
        "warnings": warnings,
    }
    return sanitize_value(report)


def verify_frame1_replay_files(
    trace_path: str | Path,
    *,
    shader_pack_path: str | Path | None = None,
    sidecar_path: str | Path | None = None,
    check_replay_assets: bool = False,
    strict: bool = False,
) -> dict[str, Any]:
    """Read local `.pg6trace` and optional `.wgsl.pack` files and run offline verification."""
    trace_label = sanitize_source_label(trace_path)
    pack_label = sanitize_source_label(shader_pack_path) if shader_pack_path is not None else None
    io_errors: list[str] = []

    trace_raw: bytes | None = None
    try:
        trace_raw = Path(trace_path).read_bytes()
    except OSError as exc:
        io_errors.append(_sanitize_os_error(trace_path, exc))

    pack_raw: bytes | None = None
    if shader_pack_path is not None:
        try:
            pack_raw = Path(shader_pack_path).read_bytes()
        except OSError as exc:
            io_errors.append(_sanitize_os_error(shader_pack_path, exc))

    sidecar_obj: Mapping[str, Any] | None = None
    if sidecar_path is not None:
        try:
            parsed = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                sidecar_obj = parsed
            else:
                io_errors.append(
                    f"sidecar {sanitize_source_label(sidecar_path)} root must be a JSON object"
                )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            if isinstance(exc, OSError):
                io_errors.append(_sanitize_os_error(sidecar_path, exc))
            else:
                io_errors.append(
                    f"failed to parse sidecar {sanitize_source_label(sidecar_path)}: {exc}"
                )

    if io_errors or trace_raw is None:
        pack_provided = shader_pack_path is not None
        report: dict[str, Any] = {
            "schema": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "status": "fail",
            "check_replay_assets": bool(check_replay_assets),
            "strict": bool(strict),
            "inputs": {
                "trace": trace_label,
                "shader_pack": pack_label,
            },
            "trace": {
                "valid": False,
                "byte_length": len(trace_raw) if trace_raw is not None else 0,
                "consumed_bytes": 0,
                "packet_count": 0,
                "frame_count": 0,
                "present_count": 0,
                "complete_frames": 0,
                "has_trailing_incomplete_frame": False,
                "unknown_opcode_count": 0,
                "packet_counts": {name: 0 for name in KEY_PACKET_COUNT_NAMES},
                "opcode_histogram": {},
            },
            "shaders": {
                "pack_provided": pack_provided,
                "pack_valid": False if pack_provided else None,
                "pack_version": None,
                "pack_flags": None,
                "pack_entry_count": 0,
                "pack_stage_counts": {stage: 0 for stage in STAGE_ORDER},
                "referenced_shader_count": 0,
                "referenced_shader_hashes": [],
                "matched_shader_hashes": [],
                "missing_shader_hashes": [],
                "inline_shader_hashes": [],
                "unresolved_shader_hashes": [],
                "stage_counts": {stage: 0 for stage in STAGE_ORDER},
                "match_rate": 0.0,
                "entry_point_checks_passed": 0,
                "entry_point_mismatch_count": 0,
                "entry_point_mismatches": [],
                "stage_mismatch_count": 0,
                "stage_mismatches": [],
            },
            "draw_readiness": {
                "ready": False,
                "total_draws": 0,
                "total_draw_indexed": 0,
                "draws_before_present": 0,
                "resolved_draws_before_present": 0,
                "draws_with_vertex_data_before_present": 0,
                "procedural_draws_before_present": 0,
                "vertex_buffer_writes": 0,
                "vertex_bytes_written": 0,
                "issues": [],
            },
            "checklist": {
                "trace_parse_ok": False,
                "has_present_frame": False,
                "shader_pack_ok": False if pack_provided else True,
                "all_shaders_resolved": False,
                "entry_points_valid": False,
                "draw_ready_before_present": False,
            },
            "errors": io_errors,
            "warnings": [],
        }
        return sanitize_value(report)

    return verify_frame1_replay_bytes(
        trace_raw,
        shader_pack_bytes=pack_raw,
        trace_label=trace_label,
        shader_pack_label=pack_label,
        sidecar=sidecar_obj,
        check_replay_assets=check_replay_assets,
        strict=strict,
    )


# ---------------------------------------------------------------------------
# Human-readable sanitized report formatting & CLI
# ---------------------------------------------------------------------------


def format_text_report(report: Mapping[str, Any]) -> str:
    """Format a strictly sanitized multi-line human-readable report."""
    inputs = report["inputs"]
    trace = report["trace"]
    shaders = report["shaders"]
    draw = report["draw_readiness"]
    checklist = report["checklist"]
    pcounts = trace["packet_counts"]

    lines: list[str] = [
        f"Frame-1 WebGPU Replay Verifier: {str(report['status']).upper()}",
        f"  Trace: {inputs['trace']} ({trace['byte_length']} bytes, {trace['packet_count']} packets, "
        f"{trace['present_count']} Present frame(s))",
    ]
    if inputs.get("shader_pack"):
        lines.append(
            f"  Shader pack: {inputs['shader_pack']} "
            f"(valid={shaders['pack_valid']}, entries={shaders['pack_entry_count']})"
        )
    else:
        lines.append("  Shader pack: <none>")

    lines.append(
        "  Key packets: "
        f"BeginFrame={pcounts['BeginFrame']}, ClearColor={pcounts['ClearColor']}, "
        f"CreateShaderModule={pcounts['CreateShaderModule']}, "
        f"CreateRenderPipeline={pcounts['CreateRenderPipeline']}, "
        f"CreateRenderPipeline2={pcounts['CreateRenderPipeline2']}, "
        f"Draw={pcounts['Draw']}, DrawIndexed={pcounts['DrawIndexed']}, "
        f"WriteBuffer={pcounts['WriteBuffer']}, CreateTexture2={pcounts['CreateTexture2']}, "
        f"Present={pcounts['Present']}"
    )
    lines.append(
        "  Shaders: "
        f"referenced={shaders['referenced_shader_count']}, "
        f"matched={len(shaders['matched_shader_hashes'])}, "
        f"missing={len(shaders['missing_shader_hashes'])}, "
        f"stages={shaders['stage_counts']}, "
        f"entry_point_mismatches={shaders['entry_point_mismatch_count']}"
    )
    lines.append(
        "  Draw readiness: "
        f"ready={draw['ready']}, draws_before_present={draw['draws_before_present']}, "
        f"resolved={draw['resolved_draws_before_present']}, "
        f"with_vertex_data={draw['draws_with_vertex_data_before_present']}"
    )

    lines.append("  Checklist:")
    for key, passed in checklist.items():
        mark = "PASS" if passed else "FAIL"
        lines.append(f"    [{mark}] {key}")

    if report["warnings"]:
        lines.append("  Warnings:")
        for w in report["warnings"]:
            lines.append(f"    - {w}")

    if report["errors"]:
        lines.append("  Errors:")
        for e in report["errors"]:
            lines.append(f"    - {e}")

    return sanitize_text("\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verify_frame1_replay.py",
        description=(
            "Offline sanitized .pg6trace + .wgsl.pack (WGPK v1) WebGPU Frame-1 replay verifier."
        ),
    )
    parser.add_argument(
        "trace_pos",
        nargs="?",
        metavar="TRACE",
        help="Path to local .pg6trace GPU command stream file.",
    )
    parser.add_argument(
        "--trace",
        dest="trace_opt",
        metavar="PATH",
        help="Path to local .pg6trace GPU command stream file (alternative to positional TRACE).",
    )
    parser.add_argument(
        "--shader-pack",
        "--pack",
        dest="shader_pack",
        metavar="PATH",
        help="Optional path to WGPK v1 .wgsl.pack shader archive.",
    )
    parser.add_argument(
        "--sidecar",
        dest="sidecar",
        metavar="PATH",
        help="Optional path to playgta6-frame-capture .pg6trace.json sidecar.",
    )
    parser.add_argument(
        "--check-replay-assets",
        action="store_true",
        help="Run full replay asset verification checklist mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit strictly sanitized JSON report to stdout (no host paths, usernames, or raw bytes).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Promote warnings (trailing packet bytes, unknown opcodes, duplicate hashes) to failures.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    trace_path = args.trace_opt or args.trace_pos
    if not trace_path:
        parser.error("a .pg6trace path is required (pass positional TRACE or --trace PATH)")

    report = verify_frame1_replay_files(
        trace_path,
        shader_pack_path=args.shader_pack,
        sidecar_path=args.sidecar,
        check_replay_assets=args.check_replay_assets,
        strict=args.strict,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_text_report(report))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
