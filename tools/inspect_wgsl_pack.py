#!/usr/bin/env python3
"""Offline WGPK (.wgsl.pack) shader pack inspector, diff, and WebGPU budget CLI.

Matches the binary `WGPK` v1 container layout implemented in `shaders/src/pack.rs`,
`shaders/src/convert.rs`, and `web/workers/wgsl_pack_loader.js` byte-for-byte:

  Header (16 bytes):
    magic[4]       = b"WGPK"
    version        = u32 LE (1)
    entry_count    = u32 LE
    flags          = u32 LE

  Entry table (`entry_count * 20` bytes):
    hash           = u64 LE (FNV-1a 64-bit hash)
    stage          = u8 (1 = vertex, 2 = fragment, 3 = compute)
    reserved[3]    = 3 zero bytes (ignored on read for forward compatibility)
    offset         = u32 LE (byte offset from start of file)
    length         = u32 LE (byte length of UTF-8 WGSL payload)

  Payload region:
    Concatenated UTF-8 WGSL sources at `[offset .. offset + length]`.

Pure Python 3.12 stdlib only (zero external dependencies). Usable both as an
importable library and as a CLI:

    python tools/inspect_wgsl_pack.py inspect <pack> [--json] [--strict]
    python tools/inspect_wgsl_pack.py inspect <pack> --diff-dump <dir> [--reflect <reflect.json>] [--max-samples N] [--json] [--strict]
    python tools/inspect_wgsl_pack.py diff <packA> <packB> [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Sequence

# ---------------------------------------------------------------------------
# WGPK binary format constants (matching shaders/src/pack.rs byte-for-byte)
# ---------------------------------------------------------------------------

PACK_MAGIC = b"WGPK"
PACK_VERSION = 1
PACK_HEADER_STRUCT = struct.Struct("<4sIII")
PACK_HEADER_BYTES = PACK_HEADER_STRUCT.size  # 16 bytes

PACK_ENTRY_STRUCT = struct.Struct("<QB3sII")
PACK_ENTRY_BYTES = PACK_ENTRY_STRUCT.size  # 20 bytes

U32_MAX = 0xFFFFFFFF
U64_MAX = 0xFFFFFFFFFFFFFFFF

FNV1A_64_OFFSET_BASIS = 0xCBF29CE484222325
FNV1A_64_PRIME = 0x00000100000001B3


class ShaderStage(IntEnum):
    """WebGPU shader stage codes matching `ShaderStage` in `shaders/src/convert.rs`."""

    VERTEX = 1
    FRAGMENT = 2
    COMPUTE = 3

    def as_str(self) -> str:
        return STAGE_NAMES[self]

    @classmethod
    def from_value(cls, value: int | str | ShaderStage) -> ShaderStage:
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ValueError(f"invalid shader stage: {value!r}")
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError as exc:
                raise ValueError(f"invalid shader stage code: {value}") from exc
        if isinstance(value, str):
            key = value.strip().lower()
            if key in STAGE_NAME_TO_ENUM:
                return STAGE_NAME_TO_ENUM[key]
            raise ValueError(f"invalid shader stage name: {value!r}")
        raise TypeError(f"unsupported shader stage type: {type(value).__name__}")


STAGE_NAMES: dict[ShaderStage, str] = {
    ShaderStage.VERTEX: "vertex",
    ShaderStage.FRAGMENT: "fragment",
    ShaderStage.COMPUTE: "compute",
}

STAGE_NAME_TO_ENUM: dict[str, ShaderStage] = {
    name: stage for stage, name in STAGE_NAMES.items()
}

STAGE_ORDER: tuple[str, ...] = ("vertex", "fragment", "compute")

PIPELINE_DUMP_SCHEMA = "playgta6-pipeline-dump"
PIPELINE_DUMP_SCHEMA_VERSION = 1
REFLECT_SIDECAR_SCHEMA = "playgta6-stage-reflection"
REFLECT_SIDECAR_VERSION = 1
DUMP_DIFF_REPORT_SCHEMA = "playgta6-wgpk-dump-diff"
DUMP_DIFF_REPORT_VERSION = 1
DEFAULT_MAX_MISSING_SAMPLES = 25

DUMP_STAGE_ORDER: tuple[str, ...] = ("vertex", "fragment", "compute", "unknown")
DUMP_ORIGIN_ORDER: tuple[str, ...] = ("native", "dxil")

_DUMP_STAGE_ALIASES: dict[str, str] = {
    "vertex": "vertex",
    "vs": "vertex",
    "fragment": "fragment",
    "pixel": "fragment",
    "ps": "fragment",
    "fs": "fragment",
    "compute": "compute",
    "cs": "compute",
}

_DXIL_KIND_TO_STAGE: dict[int, str] = {
    0: "fragment",
    1: "vertex",
    5: "compute",
}

_FILENAME_HEX16_RE = re.compile(r"^(?:0[xX])?([0-9a-fA-F]{16})$")


class WgpkParseError(ValueError):
    """Raised when a binary WGPK shader pack is truncated, malformed, or invalid."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "parse_error",
        entry_index: int | None = None,
        offset: int | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.entry_index = entry_index
        self.offset = offset
        super().__init__(message)


class DumpDiffError(ValueError):
    """Raised when a pipeline dump directory or reflection sidecar cannot be loaded."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "dump_diff_error",
        source: str | None = None,
    ) -> None:
        self.message = message
        self.code = code
        self.source = source
        super().__init__(message)


# ---------------------------------------------------------------------------
# Hashing, formatting, and sanitization helpers
# ---------------------------------------------------------------------------


def fnv1a_64(data: bytes | bytearray | memoryview | str) -> int:
    """Compute deterministic 64-bit FNV-1a hash matching `shaders/src/convert.rs`."""
    if isinstance(data, str):
        raw = data.encode("utf-8")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
    else:
        raise TypeError(f"expected bytes-like or str, got {type(data).__name__}")

    digest = FNV1A_64_OFFSET_BASIS
    for byte in raw:
        digest ^= byte
        digest = (digest * FNV1A_64_PRIME) & U64_MAX
    return digest


def normalize_shader_hash(value: int | str) -> int:
    """Normalize an integer or hex string shader hash to unsigned 64-bit int."""
    if isinstance(value, bool):
        raise ValueError(f"invalid shader hash: {value!r}")
    if isinstance(value, int):
        if value < 0 or value > U64_MAX:
            raise ValueError(f"shader hash out of u64 range: {value}")
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        raw_hex = trimmed[2:] if trimmed.lower().startswith("0x") else trimmed
        if not raw_hex or len(raw_hex) > 16 or not re.fullmatch(r"[0-9a-fA-F]+", raw_hex):
            raise ValueError(f"invalid hex shader hash: {value!r}")
        return int(raw_hex, 16)
    raise TypeError(f"shader hash must be int or hex str, got {type(value).__name__}")


def format_shader_hash(shader_hash: int) -> str:
    """Format a 64-bit hash as canonical `0x` + 16 lowercase hex digits."""
    return f"0x{normalize_shader_hash(shader_hash):016x}"


def sanitize_source_label(source: str | Path | None) -> str:
    """Return a strictly sanitized source label (basename only; no host paths/usernames)."""
    if source is None:
        return "<memory>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    if raw == "-":
        return "<stdin>"
    if raw.startswith("<") and raw.endswith(">"):
        return raw
    # Split on both POSIX and Windows separators regardless of host OS.
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    # Guard against bare drive letters like "C:" after stripping slashes.
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return base


def _sanitize_os_error(source: str | Path, exc: OSError) -> str:
    """Format an OSError message without ever leaking absolute host paths."""
    label = sanitize_source_label(source)
    reason = exc.strerror or type(exc).__name__
    return f"failed to read {label}: {reason}"


# ---------------------------------------------------------------------------
# Binary WGPK structures, parser, and builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WgpkEntry:
    """Single decoded entry from a `.wgsl.pack` (`WGPK`) archive."""

    hash: int
    stage: ShaderStage
    offset: int
    length: int
    wgsl: str

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.hash)

    @property
    def stage_name(self) -> str:
        return self.stage.as_str()


@dataclass(frozen=True)
class WgpkBundle:
    """Decoded binary `.wgsl.pack` (`WGPK`) bundle."""

    version: int
    flags: int
    byte_length: int
    entries: tuple[WgpkEntry, ...]

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def data_start(self) -> int:
        return PACK_HEADER_BYTES + len(self.entries) * PACK_ENTRY_BYTES

    @property
    def total_wgsl_bytes(self) -> int:
        return sum(entry.length for entry in self.entries)

    def stage_counts(self) -> dict[str, int]:
        counts = {stage: 0 for stage in STAGE_ORDER}
        for entry in self.entries:
            counts[entry.stage_name] += 1
        return counts

    def entry_by_hash(self, shader_hash: int | str) -> WgpkEntry | None:
        target = normalize_shader_hash(shader_hash)
        for entry in self.entries:
            if entry.hash == target:
                return entry
        return None


@dataclass(frozen=True)
class WgpkTocEntry:
    """Single TOC entry from a `.wgsl.pack` (`WGPK`) archive (without WGSL source bytes)."""

    hash: int
    stage: ShaderStage
    offset: int
    length: int

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.hash)

    @property
    def stage_name(self) -> str:
        return self.stage.as_str()


@dataclass(frozen=True)
class WgpkToc:
    """Decoded header and entry table of a binary `.wgsl.pack` (`WGPK`) bundle."""

    version: int
    flags: int
    byte_length: int
    entries: tuple[WgpkTocEntry, ...]

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def hashes(self) -> frozenset[int]:
        return frozenset(entry.hash for entry in self.entries)


def parse_wgpk_toc(data: bytes | bytearray | memoryview) -> WgpkToc:
    """Parse and validate a `.wgsl.pack` header and entry table without decoding WGSL payloads."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes-like buffer, got {type(data).__name__}")

    mv = memoryview(data)
    total_len = len(mv)

    if total_len < PACK_HEADER_BYTES:
        raise WgpkParseError(
            f"WGPK pack truncated: {total_len} bytes (minimum header size: {PACK_HEADER_BYTES})",
            code="truncated_header",
            offset=0,
        )

    magic, version, entry_count, flags = PACK_HEADER_STRUCT.unpack_from(mv, 0)
    if magic != PACK_MAGIC:
        raise WgpkParseError(
            f"invalid WGPK magic {magic!r} (expected {PACK_MAGIC!r})",
            code="invalid_magic",
            offset=0,
        )

    if version != PACK_VERSION:
        raise WgpkParseError(
            f"unsupported WGPK version {version} (expected {PACK_VERSION})",
            code="unsupported_version",
            offset=4,
        )

    table_bytes = entry_count * PACK_ENTRY_BYTES
    data_start = PACK_HEADER_BYTES + table_bytes
    if total_len < data_start:
        raise WgpkParseError(
            f"WGPK pack truncated: {total_len} bytes (header + {entry_count} entry records require {data_start} bytes)",
            code="truncated_table",
            offset=PACK_HEADER_BYTES,
        )

    entries: list[WgpkTocEntry] = []
    non_empty_ranges: list[tuple[int, int, int]] = []

    for idx in range(entry_count):
        base = PACK_HEADER_BYTES + idx * PACK_ENTRY_BYTES
        shader_hash, stage_raw, _reserved, offset, length = PACK_ENTRY_STRUCT.unpack_from(mv, base)

        try:
            stage = ShaderStage(stage_raw)
        except ValueError as exc:
            raise WgpkParseError(
                f"invalid shader stage value {stage_raw} at entry {idx}",
                code="invalid_stage",
                entry_index=idx,
                offset=base + 8,
            ) from exc

        end = offset + length
        if offset < data_start or end > total_len:
            raise WgpkParseError(
                f"entry {idx} payload [{offset}..+{length}] exceeds pack bounds "
                f"(valid payload range: [{data_start}..{total_len}])",
                code="out_of_bounds",
                entry_index=idx,
                offset=base + 12,
            )

        if length > 0:
            non_empty_ranges.append((offset, end, idx))

        entries.append(
            WgpkTocEntry(
                hash=shader_hash,
                stage=stage,
                offset=offset,
                length=length,
            )
        )

    if len(non_empty_ranges) > 1:
        non_empty_ranges.sort(key=lambda item: (item[0], item[1]))
        for prev, curr in zip(non_empty_ranges, non_empty_ranges[1:]):
            if curr[0] < prev[1]:
                raise WgpkParseError(
                    f"overlapping payload ranges between entry {prev[2]} [{prev[0]}..{prev[1]}) "
                    f"and entry {curr[2]} [{curr[0]}..{curr[1]})",
                    code="overlapping_payloads",
                    entry_index=curr[2],
                    offset=curr[0],
                )

    return WgpkToc(
        version=version,
        flags=flags,
        byte_length=total_len,
        entries=tuple(entries),
    )


def parse_wgpk(data: bytes | bytearray | memoryview) -> WgpkBundle:
    """Parse and strictly validate a binary `.wgsl.pack` (`WGPK`) buffer."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError(f"expected bytes-like buffer, got {type(data).__name__}")

    raw = bytes(data)
    total_len = len(raw)

    if total_len < PACK_HEADER_BYTES:
        raise WgpkParseError(
            f"WGPK pack truncated: {total_len} bytes (minimum header size: {PACK_HEADER_BYTES})",
            code="truncated_header",
            offset=0,
        )

    magic, version, entry_count, flags = PACK_HEADER_STRUCT.unpack_from(raw, 0)
    if magic != PACK_MAGIC:
        raise WgpkParseError(
            f"invalid WGPK magic {magic!r} (expected {PACK_MAGIC!r})",
            code="invalid_magic",
            offset=0,
        )

    if version != PACK_VERSION:
        raise WgpkParseError(
            f"unsupported WGPK version {version} (expected {PACK_VERSION})",
            code="unsupported_version",
            offset=4,
        )

    table_bytes = entry_count * PACK_ENTRY_BYTES
    data_start = PACK_HEADER_BYTES + table_bytes
    if total_len < data_start:
        raise WgpkParseError(
            f"WGPK pack truncated: {total_len} bytes (header + {entry_count} entry records require {data_start} bytes)",
            code="truncated_table",
            offset=PACK_HEADER_BYTES,
        )

    entries: list[WgpkEntry] = []
    non_empty_ranges: list[tuple[int, int, int]] = []

    for idx in range(entry_count):
        base = PACK_HEADER_BYTES + idx * PACK_ENTRY_BYTES
        shader_hash, stage_raw, _reserved, offset, length = PACK_ENTRY_STRUCT.unpack_from(raw, base)

        try:
            stage = ShaderStage(stage_raw)
        except ValueError as exc:
            raise WgpkParseError(
                f"invalid shader stage value {stage_raw} at entry {idx}",
                code="invalid_stage",
                entry_index=idx,
                offset=base + 8,
            ) from exc

        end = offset + length
        if offset < data_start or end > total_len:
            raise WgpkParseError(
                f"entry {idx} payload [{offset}..+{length}] exceeds pack bounds "
                f"(valid payload range: [{data_start}..{total_len}])",
                code="out_of_bounds",
                entry_index=idx,
                offset=base + 12,
            )

        if length > 0:
            non_empty_ranges.append((offset, end, idx))

        payload_bytes = raw[offset:end]
        try:
            wgsl = payload_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WgpkParseError(
                f"entry {idx} payload is not valid UTF-8: {exc.reason}",
                code="invalid_utf8",
                entry_index=idx,
                offset=offset + exc.start,
            ) from exc

        entries.append(
            WgpkEntry(
                hash=shader_hash,
                stage=stage,
                offset=offset,
                length=length,
                wgsl=wgsl,
            )
        )

    # Match web/workers/wgsl_pack_loader.js: reject overlapping non-empty payload ranges.
    if len(non_empty_ranges) > 1:
        non_empty_ranges.sort(key=lambda item: (item[0], item[1]))
        for prev, curr in zip(non_empty_ranges, non_empty_ranges[1:]):
            if curr[0] < prev[1]:
                raise WgpkParseError(
                    f"overlapping payload ranges between entry {prev[2]} [{prev[0]}..{prev[1]}) "
                    f"and entry {curr[2]} [{curr[0]}..{curr[1]})",
                    code="overlapping_payloads",
                    entry_index=curr[2],
                    offset=curr[0],
                )

    return WgpkBundle(
        version=version,
        flags=flags,
        byte_length=total_len,
        entries=tuple(entries),
    )


def _coerce_build_entry(
    item: Any, index: int
) -> tuple[int, ShaderStage, bytes]:
    """Normalize a user-supplied entry spec for `build_wgpk`."""
    if isinstance(item, WgpkEntry):
        raw_hash: int | str | None = item.hash
        raw_stage: int | str | ShaderStage = item.stage
        raw_wgsl: str | bytes | bytearray | memoryview = item.wgsl
    elif isinstance(item, Mapping):
        raw_hash = item.get("hash")
        if "stage" not in item or "wgsl" not in item:
            raise ValueError(f"entry {index} mapping must include 'stage' and 'wgsl'")
        raw_stage = item["stage"]
        raw_wgsl = item["wgsl"]
    elif isinstance(item, tuple) and len(item) == 3:
        raw_hash, raw_stage, raw_wgsl = item
    else:
        raise TypeError(
            f"entry {index} must be a WgpkEntry, mapping, or (hash, stage, wgsl) tuple"
        )

    stage = ShaderStage.from_value(raw_stage)

    if isinstance(raw_wgsl, str):
        payload = raw_wgsl.encode("utf-8")
    elif isinstance(raw_wgsl, (bytes, bytearray, memoryview)):
        payload = bytes(raw_wgsl)
    else:
        raise TypeError(f"entry {index} wgsl must be str or bytes-like")

    if len(payload) > U32_MAX:
        raise ValueError(f"entry {index} payload exceeds u32::MAX bytes")

    shader_hash = fnv1a_64(payload) if raw_hash is None else normalize_shader_hash(raw_hash)
    return shader_hash, stage, payload


def build_wgpk(
    entries: Sequence[WgpkEntry | Mapping[str, Any] | tuple[int | str | None, int | str | ShaderStage, str | bytes]],
    *,
    flags: int = 0,
) -> bytes:
    """Serialize WGSL entries into a binary `WGPK` v1 bundle matching `shaders/src/pack.rs`."""
    if isinstance(flags, bool) or not isinstance(flags, int) or flags < 0 or flags > U32_MAX:
        raise ValueError(f"flags must be an unsigned 32-bit integer, got {flags!r}")
    if len(entries) > U32_MAX:
        raise ValueError("entry count exceeds u32::MAX")

    normalized = [_coerce_build_entry(item, idx) for idx, item in enumerate(entries)]
    data_start = PACK_HEADER_BYTES + len(normalized) * PACK_ENTRY_BYTES
    total_payload = sum(len(payload) for _, _, payload in normalized)
    total_size = data_start + total_payload
    if total_size > U32_MAX:
        raise ValueError("total WGPK pack size exceeds u32::MAX byte offsets")

    out = bytearray(total_size)
    PACK_HEADER_STRUCT.pack_into(out, 0, PACK_MAGIC, PACK_VERSION, len(normalized), flags)

    cursor = data_start
    for idx, (shader_hash, stage, payload) in enumerate(normalized):
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
        cursor += length

    return bytes(out)


# ---------------------------------------------------------------------------
# WebGPU limits and static WGSL analysis
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebGpuLimits:
    """WebGPU baseline default limits (`GPUSupportedLimits` defaults)."""

    max_bind_groups: int = 4
    max_bindings_per_bind_group: int = 1000
    max_storage_buffers_per_shader_stage: int = 8
    max_uniform_buffers_per_shader_stage: int = 12
    max_sampled_textures_per_shader_stage: int = 16
    max_storage_textures_per_shader_stage: int = 4
    max_samplers_per_shader_stage: int = 16
    max_compute_workgroup_size_x: int = 256
    max_compute_workgroup_size_y: int = 256
    max_compute_workgroup_size_z: int = 64
    max_compute_invocations_per_workgroup: int = 256
    disallow_experimental_features: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "maxBindGroups": self.max_bind_groups,
            "maxBindingsPerBindGroup": self.max_bindings_per_bind_group,
            "maxStorageBuffersPerShaderStage": self.max_storage_buffers_per_shader_stage,
            "maxUniformBuffersPerShaderStage": self.max_uniform_buffers_per_shader_stage,
            "maxSampledTexturesPerShaderStage": self.max_sampled_textures_per_shader_stage,
            "maxStorageTexturesPerShaderStage": self.max_storage_textures_per_shader_stage,
            "maxSamplersPerShaderStage": self.max_samplers_per_shader_stage,
            "maxComputeWorkgroupSizeX": self.max_compute_workgroup_size_x,
            "maxComputeWorkgroupSizeY": self.max_compute_workgroup_size_y,
            "maxComputeWorkgroupSizeZ": self.max_compute_workgroup_size_z,
            "maxComputeInvocationsPerWorkgroup": self.max_compute_invocations_per_workgroup,
            "disallowExperimentalFeatures": self.disallow_experimental_features,
        }


DEFAULT_WEBGPU_LIMITS = WebGpuLimits()


@dataclass(frozen=True)
class ResourceBinding:
    """Single module-scope `@group(g) @binding(b) var` declaration."""

    group: int
    binding: int
    name: str
    kind: str  # uniform_buffer | storage_buffer | sampled_texture | storage_texture | sampler | unknown
    address_space: str | None = None
    access_mode: str | None = None
    type_expr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "binding": self.binding,
            "name": self.name,
            "kind": self.kind,
            "addressSpace": self.address_space,
            "accessMode": self.access_mode,
            "type": self.type_expr,
        }


@dataclass(frozen=True)
class EntryPointInfo:
    """Single WGSL entry point (`@vertex`, `@fragment`, or `@compute`)."""

    name: str
    stage: str
    workgroup_size: tuple[int, int, int] | None = None
    workgroup_invocations: int | None = None
    workgroup_size_raw: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stage": self.stage,
            "workgroupSize": list(self.workgroup_size) if self.workgroup_size is not None else None,
            "workgroupInvocations": self.workgroup_invocations,
            "workgroupSizeRaw": self.workgroup_size_raw,
        }


@dataclass(frozen=True)
class BudgetViolation:
    """Single WebGPU limit or policy violation detected in a shader or pack."""

    shader_hash: int
    code: str
    message: str
    limit: int | str | None = None
    actual: int | str | None = None
    detail: str | None = None

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.shader_hash)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hash": self.hash_hex,
            "code": self.code,
            "message": self.message,
        }
        if self.limit is not None:
            payload["limit"] = self.limit
        if self.actual is not None:
            payload["actual"] = self.actual
        if self.detail is not None:
            payload["detail"] = self.detail
        return payload


@dataclass(frozen=True)
class ShaderAudit:
    """Static WebGPU resource, feature, and limit audit for a single WGSL shader."""

    shader_hash: int
    wgsl_bytes: int
    pack_stage: str | None
    stages: tuple[str, ...]
    entry_points: tuple[EntryPointInfo, ...]
    bindings: tuple[ResourceBinding, ...]
    max_group_index: int | None
    bindings_per_group: dict[int, int]
    uniform_buffer_count: int
    storage_buffer_count: int
    sampled_texture_count: int
    storage_texture_count: int
    texture_count: int
    sampler_count: int
    workgroup_sizes: tuple[tuple[int, int, int], ...]
    max_workgroup_invocations: int | None
    enabled_features: tuple[str, ...]
    experimental_features: tuple[str, ...]
    has_f16: bool
    has_subgroups: bool
    violations: tuple[BudgetViolation, ...] = field(default_factory=tuple)

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.shader_hash)

    @property
    def entry_point_names(self) -> tuple[str, ...]:
        return tuple(ep.name for ep in self.entry_points)

    @property
    def total_bindings(self) -> int:
        return len(self.bindings)

    @property
    def clean(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash_hex,
            "packStage": self.pack_stage,
            "wgslBytes": self.wgsl_bytes,
            "stages": list(self.stages),
            "entryPoints": [ep.to_dict() for ep in self.entry_points],
            "maxGroupIndex": self.max_group_index,
            "bindingsPerGroup": {str(g): count for g, count in sorted(self.bindings_per_group.items())},
            "totalBindings": self.total_bindings,
            "bindingCounts": {
                "uniformBuffers": self.uniform_buffer_count,
                "storageBuffers": self.storage_buffer_count,
                "sampledTextures": self.sampled_texture_count,
                "storageTextures": self.storage_texture_count,
                "textures": self.texture_count,
                "samplers": self.sampler_count,
            },
            "bindings": [b.to_dict() for b in self.bindings],
            "workgroupSizes": [list(wg) for wg in self.workgroup_sizes],
            "maxWorkgroupInvocations": self.max_workgroup_invocations,
            "enabledFeatures": list(self.enabled_features),
            "experimentalFeatures": list(self.experimental_features),
            "hasF16": self.has_f16,
            "hasSubgroups": self.has_subgroups,
            "violations": [v.to_dict() for v in self.violations],
            "clean": self.clean,
        }


# ---------------------------------------------------------------------------
# WGSL comment stripper & static parser helpers
# ---------------------------------------------------------------------------

_ENABLE_RE = re.compile(
    r"\benable\s+([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*,?\s*;"
)

_CONST_INT_RE = re.compile(
    r"\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[A-Za-z0-9_<>]+\s*)?=\s*"
    r"([+-]?(?:0[xX][0-9a-fA-F]+|\d+)[uUiI]?)\s*;"
)

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WS_RE = re.compile(r"\s+")
_INT_LITERAL_RE = re.compile(r"^([+-]?)(0[xX][0-9a-fA-F]+|\d+)[uUiI]?$")


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


def _parse_wgsl_int(expr: str, consts: Mapping[str, int]) -> int | None:
    """Parse a WGSL integer literal or known module-scope integer `const`."""
    cleaned = expr.strip()
    if not cleaned:
        return None
    if cleaned in consts:
        return consts[cleaned]
    match = _INT_LITERAL_RE.fullmatch(cleaned)
    if not match:
        return None
    sign_str, digits = match.groups()
    value = int(digits, 16) if digits.lower().startswith("0x") else int(digits, 10)
    return -value if sign_str == "-" else value


def _skip_ws(text: str, pos: int) -> int:
    match = _WS_RE.match(text, pos)
    return match.end() if match else pos


def _consume_balanced(text: str, pos: int, open_ch: str, close_ch: str) -> tuple[str, int]:
    """Consume a balanced delimiter pair starting at `pos`; return `(inner, next_pos)`."""
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


def _split_top_level_csv(text: str) -> list[str]:
    """Split comma-separated arguments respecting nested `()`, `<>`, `[]`."""
    parts: list[str] = []
    buf: list[str] = []
    paren = 0
    angle = 0
    bracket = 0
    for ch in text:
        if ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)

        if ch == "," and paren == 0 and angle == 0 and bracket == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _classify_resource_kind(
    address_space: str | None,
    type_expr: str,
) -> str:
    """Classify a WGSL `@group/@binding` var into WebGPU resource categories."""
    if address_space == "uniform":
        return "uniform_buffer"
    if address_space == "storage":
        return "storage_buffer"

    # Unwrap optional binding_array<T> / array<T, N> to inspect the element resource type.
    normalized_type = type_expr.strip()
    for wrapper in ("binding_array<", "array<"):
        if normalized_type.startswith(wrapper) and normalized_type.endswith(">"):
            inner = normalized_type[len(wrapper) : -1]
            inner_parts = _split_top_level_csv(inner)
            if inner_parts:
                normalized_type = inner_parts[0].strip()

    base_match = _IDENT_RE.match(normalized_type)
    base_type = base_match.group(0) if base_match else ""

    if base_type.startswith("texture_storage_"):
        return "storage_texture"
    if base_type.startswith("texture_"):
        return "sampled_texture"
    if base_type in ("sampler", "sampler_comparison"):
        return "sampler"
    return "unknown"


def _parse_workgroup_size_attr(
    raw_args: str,
    consts: Mapping[str, int],
) -> tuple[tuple[int, int, int] | None, int | None]:
    """Parse `@workgroup_size(x, y?, z?)` arguments into `(dims, invocations)`."""
    args = _split_top_level_csv(raw_args)
    if not (1 <= len(args) <= 3):
        return None, None
    parsed: list[int] = []
    for arg in args:
        val = _parse_wgsl_int(arg, consts)
        if val is None:
            return None, None
        parsed.append(val)
    while len(parsed) < 3:
        parsed.append(1)
    dims = (parsed[0], parsed[1], parsed[2])
    invocations = dims[0] * dims[1] * dims[2]
    return dims, invocations


def _scan_module_declarations(
    code: str,
    consts: Mapping[str, int],
) -> tuple[list[EntryPointInfo], list[ResourceBinding]]:
    """Scan module-scope WGSL declarations for entry points and resource bindings."""
    entry_points: list[EntryPointInfo] = []
    bindings: list[ResourceBinding] = []

    pending_attrs: list[tuple[str, str | None]] = []
    pos = 0
    n = len(code)

    while pos < n:
        pos = _skip_ws(code, pos)
        if pos >= n:
            break

        ch = code[pos]

        # Skip any unexpected top-level block cleanly.
        if ch == "{":
            _, pos = _consume_balanced(code, pos, "{", "}")
            pending_attrs.clear()
            continue

        if ch == ";":
            pending_attrs.clear()
            pos += 1
            continue

        # Collect module-scope attribute: @name or @name(...)
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

        # Identifier / keyword at module scope
        ident_match = _IDENT_RE.match(code, pos)
        if not ident_match:
            pos += 1
            continue

        keyword = ident_match.group(0)
        pos = ident_match.end()

        if keyword == "fn":
            pos = _skip_ws(code, pos)
            fn_match = _IDENT_RE.match(code, pos)
            fn_name = fn_match.group(0) if fn_match else "<anonymous>"
            if fn_match:
                pos = fn_match.end()

            fn_stages: list[str] = []
            wg_size: tuple[int, int, int] | None = None
            wg_invocations: int | None = None
            wg_raw: str | None = None

            for attr_name, attr_args in pending_attrs:
                if attr_name in STAGE_NAME_TO_ENUM:
                    if attr_name not in fn_stages:
                        fn_stages.append(attr_name)
                elif attr_name == "stage" and attr_args is not None:
                    legacy_stage = attr_args.strip().lower()
                    if legacy_stage in STAGE_NAME_TO_ENUM and legacy_stage not in fn_stages:
                        fn_stages.append(legacy_stage)
                elif attr_name == "workgroup_size" and attr_args is not None:
                    wg_raw = attr_args.strip()
                    wg_size, wg_invocations = _parse_workgroup_size_attr(attr_args, consts)

            for stage_name in fn_stages:
                entry_points.append(
                    EntryPointInfo(
                        name=fn_name,
                        stage=stage_name,
                        workgroup_size=wg_size if stage_name == "compute" else None,
                        workgroup_invocations=wg_invocations if stage_name == "compute" else None,
                        workgroup_size_raw=wg_raw if stage_name == "compute" else None,
                    )
                )

            pending_attrs.clear()

            # Skip function signature up to `{` (or `;` on malformed snippets), then skip body.
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

        if keyword == "var":
            attrs = list(pending_attrs)
            pending_attrs.clear()

            pos = _skip_ws(code, pos)
            address_space: str | None = None
            access_mode: str | None = None
            if pos < n and code[pos] == "<":
                qual_inner, pos = _consume_balanced(code, pos, "<", ">")
                qual_parts = [p.strip() for p in _split_top_level_csv(qual_inner)]
                if qual_parts:
                    address_space = qual_parts[0]
                if len(qual_parts) > 1:
                    access_mode = qual_parts[1]
                pos = _skip_ws(code, pos)

            var_match = _IDENT_RE.match(code, pos)
            var_name = var_match.group(0) if var_match else "<anonymous>"
            if var_match:
                pos = var_match.end()
            pos = _skip_ws(code, pos)

            type_expr = ""
            if pos < n and code[pos] == ":":
                pos += 1
                type_start = pos
                angle = 0
                paren = 0
                bracket = 0
                while pos < n:
                    c = code[pos]
                    if c == "<":
                        angle += 1
                    elif c == ">":
                        angle = max(0, angle - 1)
                    elif c == "(":
                        paren += 1
                    elif c == ")":
                        paren = max(0, paren - 1)
                    elif c == "[":
                        bracket += 1
                    elif c == "]":
                        bracket = max(0, bracket - 1)
                    elif c in ("=", ";") and angle == 0 and paren == 0 and bracket == 0:
                        break
                    pos += 1
                type_expr = re.sub(r"\s+", " ", code[type_start:pos]).strip()

            # Advance to statement terminator `;`.
            paren = 0
            bracket = 0
            while pos < n:
                c = code[pos]
                if c == "(":
                    paren += 1
                    pos += 1
                elif c == ")":
                    paren = max(0, paren - 1)
                    pos += 1
                elif c == "[":
                    bracket += 1
                    pos += 1
                elif c == "]":
                    bracket = max(0, bracket - 1)
                    pos += 1
                elif c == "{":
                    _, pos = _consume_balanced(code, pos, "{", "}")
                elif c == ";" and paren == 0 and bracket == 0:
                    pos += 1
                    break
                else:
                    pos += 1

            group_val: int | None = None
            binding_val: int | None = None
            for attr_name, attr_args in attrs:
                if attr_name == "group" and attr_args is not None:
                    group_val = _parse_wgsl_int(attr_args, consts)
                elif attr_name == "binding" and attr_args is not None:
                    binding_val = _parse_wgsl_int(attr_args, consts)

            if group_val is not None and binding_val is not None:
                kind = _classify_resource_kind(address_space, type_expr)
                bindings.append(
                    ResourceBinding(
                        group=group_val,
                        binding=binding_val,
                        name=var_name,
                        kind=kind,
                        address_space=address_space,
                        access_mode=access_mode,
                        type_expr=type_expr,
                    )
                )
            continue

        # Other module-scope constructs (struct, alias, const, override, enable, diagnostic, etc.)
        # Clear pending_attrs only when we finish a declaration block or statement.
        if keyword in ("struct", "alias", "const", "override", "enable", "requires", "diagnostic", "const_assert"):
            pending_attrs.clear()
            while pos < n:
                c = code[pos]
                if c == "{":
                    _, pos = _consume_balanced(code, pos, "{", "}")
                    # Optional trailing semicolon after struct is consumed on next loop iteration.
                    break
                if c == ";":
                    pos += 1
                    break
                pos += 1
            continue

    return entry_points, bindings


def analyze_wgsl(
    source: str,
    shader_hash: int,
    *,
    limits: WebGpuLimits | None = None,
    pack_stage: str | None = None,
) -> ShaderAudit:
    """Statically analyze a WGSL source string for stages, bindings, features, and WebGPU limits."""
    active_limits = limits if limits is not None else DEFAULT_WEBGPU_LIMITS
    normalized_hash = normalize_shader_hash(shader_hash)
    wgsl_bytes = len(source.encode("utf-8"))

    stripped = strip_wgsl_comments(source)

    # Extract `enable ...;` directives.
    enabled_set: set[str] = set()
    for match in _ENABLE_RE.finditer(stripped):
        for token in match.group(1).split(","):
            name = token.strip()
            if name:
                enabled_set.add(name)
    enabled_features = tuple(sorted(enabled_set))
    experimental_features = tuple(
        feat for feat in enabled_features if feat.startswith("chromium_experimental_")
    )
    has_f16 = "f16" in enabled_set or "subgroups_f16" in enabled_set
    has_subgroups = (
        "subgroups" in enabled_set
        or "subgroups_f16" in enabled_set
        or any(feat.startswith("chromium_experimental_subgroup") for feat in enabled_set)
    )

    # Extract module-scope integer constants for resolving `@workgroup_size` / `@group` / `@binding`.
    consts: dict[str, int] = {}
    for const_match in _CONST_INT_RE.finditer(stripped):
        cname, cval = const_match.groups()
        parsed_val = _parse_wgsl_int(cval, consts)
        if parsed_val is not None:
            consts[cname] = parsed_val

    entry_points, bindings = _scan_module_declarations(stripped, consts)

    seen_stages: list[str] = []
    for ep in entry_points:
        if ep.stage not in seen_stages:
            seen_stages.append(ep.stage)

    bindings_per_group: dict[int, int] = {}
    max_group_index: int | None = None
    uniform_buffer_count = 0
    storage_buffer_count = 0
    sampled_texture_count = 0
    storage_texture_count = 0
    sampler_count = 0

    for binding in bindings:
        bindings_per_group[binding.group] = bindings_per_group.get(binding.group, 0) + 1
        if max_group_index is None or binding.group > max_group_index:
            max_group_index = binding.group
        if binding.kind == "uniform_buffer":
            uniform_buffer_count += 1
        elif binding.kind == "storage_buffer":
            storage_buffer_count += 1
        elif binding.kind == "sampled_texture":
            sampled_texture_count += 1
        elif binding.kind == "storage_texture":
            storage_texture_count += 1
        elif binding.kind == "sampler":
            sampler_count += 1

    texture_count = sampled_texture_count + storage_texture_count

    workgroup_sizes = tuple(
        ep.workgroup_size for ep in entry_points if ep.workgroup_size is not None
    )
    invocations_list = [
        ep.workgroup_invocations
        for ep in entry_points
        if ep.workgroup_invocations is not None
    ]
    max_workgroup_invocations = max(invocations_list) if invocations_list else None

    # Evaluate WebGPU default-limit and policy violations.
    violations: list[BudgetViolation] = []

    # 1. Bind group index and per-group binding count limits + duplicate (group, binding) detection.
    seen_pairs: set[tuple[int, int]] = set()
    reported_bad_groups: set[int] = set()
    for binding in bindings:
        if binding.group < 0 or binding.group >= active_limits.max_bind_groups:
            if binding.group not in reported_bad_groups:
                reported_bad_groups.add(binding.group)
                violations.append(
                    BudgetViolation(
                        shader_hash=normalized_hash,
                        code="bind_group_index_exceeded",
                        message=(
                            f"@group({binding.group}) exceeds WebGPU maxBindGroups "
                            f"({active_limits.max_bind_groups}; valid range 0..{active_limits.max_bind_groups - 1})"
                        ),
                        limit=active_limits.max_bind_groups - 1,
                        actual=binding.group,
                        detail=binding.name,
                    )
                )
        if binding.binding < 0:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="invalid_binding_index",
                    message=f"@binding({binding.binding}) must be non-negative",
                    limit=0,
                    actual=binding.binding,
                    detail=binding.name,
                )
            )
        pair = (binding.group, binding.binding)
        if pair in seen_pairs:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="duplicate_binding",
                    message=f"duplicate resource binding @group({binding.group}) @binding({binding.binding})",
                    detail=binding.name,
                )
            )
        else:
            seen_pairs.add(pair)

    for group_idx, count in sorted(bindings_per_group.items()):
        if count > active_limits.max_bindings_per_bind_group:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="bindings_per_group_exceeded",
                    message=(
                        f"group {group_idx} declares {count} bindings "
                        f"(limit: {active_limits.max_bindings_per_bind_group})"
                    ),
                    limit=active_limits.max_bindings_per_bind_group,
                    actual=count,
                    detail=f"group={group_idx}",
                )
            )

    # 2. Per-stage resource binding count limits.
    if storage_buffer_count > active_limits.max_storage_buffers_per_shader_stage:
        violations.append(
            BudgetViolation(
                shader_hash=normalized_hash,
                code="storage_buffers_per_stage_exceeded",
                message=(
                    f"storage buffers ({storage_buffer_count}) exceed WebGPU default limit "
                    f"({active_limits.max_storage_buffers_per_shader_stage})"
                ),
                limit=active_limits.max_storage_buffers_per_shader_stage,
                actual=storage_buffer_count,
            )
        )

    if uniform_buffer_count > active_limits.max_uniform_buffers_per_shader_stage:
        violations.append(
            BudgetViolation(
                shader_hash=normalized_hash,
                code="uniform_buffers_per_stage_exceeded",
                message=(
                    f"uniform buffers ({uniform_buffer_count}) exceed WebGPU default limit "
                    f"({active_limits.max_uniform_buffers_per_shader_stage})"
                ),
                limit=active_limits.max_uniform_buffers_per_shader_stage,
                actual=uniform_buffer_count,
            )
        )

    if sampled_texture_count > active_limits.max_sampled_textures_per_shader_stage:
        violations.append(
            BudgetViolation(
                shader_hash=normalized_hash,
                code="sampled_textures_per_stage_exceeded",
                message=(
                    f"sampled textures ({sampled_texture_count}) exceed WebGPU default limit "
                    f"({active_limits.max_sampled_textures_per_shader_stage})"
                ),
                limit=active_limits.max_sampled_textures_per_shader_stage,
                actual=sampled_texture_count,
            )
        )

    if storage_texture_count > active_limits.max_storage_textures_per_shader_stage:
        violations.append(
            BudgetViolation(
                shader_hash=normalized_hash,
                code="storage_textures_per_stage_exceeded",
                message=(
                    f"storage textures ({storage_texture_count}) exceed WebGPU default limit "
                    f"({active_limits.max_storage_textures_per_shader_stage})"
                ),
                limit=active_limits.max_storage_textures_per_shader_stage,
                actual=storage_texture_count,
            )
        )

    if sampler_count > active_limits.max_samplers_per_shader_stage:
        violations.append(
            BudgetViolation(
                shader_hash=normalized_hash,
                code="samplers_per_stage_exceeded",
                message=(
                    f"samplers ({sampler_count}) exceed WebGPU default limit "
                    f"({active_limits.max_samplers_per_shader_stage})"
                ),
                limit=active_limits.max_samplers_per_shader_stage,
                actual=sampler_count,
            )
        )

    # 3. Compute workgroup dimension & total invocation limits.
    for ep in entry_points:
        if ep.workgroup_size is None:
            continue
        wx, wy, wz = ep.workgroup_size
        if wx <= 0 or wy <= 0 or wz <= 0:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="workgroup_size_invalid",
                    message=f"entry point '{ep.name}' has non-positive @workgroup_size({wx}, {wy}, {wz})",
                    actual=f"{wx}x{wy}x{wz}",
                    detail=ep.name,
                )
            )
            continue

        if wx > active_limits.max_compute_workgroup_size_x:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="workgroup_size_x_exceeded",
                    message=(
                        f"entry point '{ep.name}' @workgroup_size.x ({wx}) exceeds WebGPU limit "
                        f"({active_limits.max_compute_workgroup_size_x})"
                    ),
                    limit=active_limits.max_compute_workgroup_size_x,
                    actual=wx,
                    detail=ep.name,
                )
            )
        if wy > active_limits.max_compute_workgroup_size_y:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="workgroup_size_y_exceeded",
                    message=(
                        f"entry point '{ep.name}' @workgroup_size.y ({wy}) exceeds WebGPU limit "
                        f"({active_limits.max_compute_workgroup_size_y})"
                    ),
                    limit=active_limits.max_compute_workgroup_size_y,
                    actual=wy,
                    detail=ep.name,
                )
            )
        if wz > active_limits.max_compute_workgroup_size_z:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="workgroup_size_z_exceeded",
                    message=(
                        f"entry point '{ep.name}' @workgroup_size.z ({wz}) exceeds WebGPU limit "
                        f"({active_limits.max_compute_workgroup_size_z})"
                    ),
                    limit=active_limits.max_compute_workgroup_size_z,
                    actual=wz,
                    detail=ep.name,
                )
            )

        invocations = wx * wy * wz
        if invocations > active_limits.max_compute_invocations_per_workgroup:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="workgroup_invocations_exceeded",
                    message=(
                        f"entry point '{ep.name}' @workgroup_size({wx}, {wy}, {wz}) = {invocations} "
                        f"invocations exceeds WebGPU limit ({active_limits.max_compute_invocations_per_workgroup})"
                    ),
                    limit=active_limits.max_compute_invocations_per_workgroup,
                    actual=invocations,
                    detail=ep.name,
                )
            )

    # 4. Non-standard experimental Chromium WGSL extensions.
    if active_limits.disallow_experimental_features:
        for feat in experimental_features:
            violations.append(
                BudgetViolation(
                    shader_hash=normalized_hash,
                    code="experimental_feature",
                    message=f"non-standard experimental WGSL extension enabled: {feat}",
                    actual=feat,
                )
            )

    # 5. Stage consistency against WGPK entry table metadata (if supplied and entry points exist).
    if pack_stage is not None and seen_stages and pack_stage not in seen_stages:
        violations.append(
            BudgetViolation(
                shader_hash=normalized_hash,
                code="stage_mismatch",
                message=(
                    f"WGPK entry declares stage '{pack_stage}' but WGSL defines stages "
                    f"{', '.join(seen_stages)}"
                ),
                limit=pack_stage,
                actual=",".join(seen_stages),
            )
        )

    return ShaderAudit(
        shader_hash=normalized_hash,
        wgsl_bytes=wgsl_bytes,
        pack_stage=pack_stage,
        stages=tuple(seen_stages),
        entry_points=tuple(entry_points),
        bindings=tuple(bindings),
        max_group_index=max_group_index,
        bindings_per_group=dict(sorted(bindings_per_group.items())),
        uniform_buffer_count=uniform_buffer_count,
        storage_buffer_count=storage_buffer_count,
        sampled_texture_count=sampled_texture_count,
        storage_texture_count=storage_texture_count,
        texture_count=texture_count,
        sampler_count=sampler_count,
        workgroup_sizes=workgroup_sizes,
        max_workgroup_invocations=max_workgroup_invocations,
        enabled_features=enabled_features,
        experimental_features=experimental_features,
        has_f16=has_f16,
        has_subgroups=has_subgroups,
        violations=tuple(violations),
    )


# ---------------------------------------------------------------------------
# Pack inspection and pack-to-pack diff
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PackInspectReport:
    """Structured summary of inspecting a binary `.wgsl.pack` bundle."""

    source: str
    version: int
    flags: int
    pack_bytes: int
    entry_count: int
    total_wgsl_bytes: int
    stage_counts: dict[str, int]
    feature_histogram: dict[str, int]
    violations: tuple[BudgetViolation, ...]
    shaders: tuple[ShaderAudit, ...]
    limits: WebGpuLimits

    @property
    def clean(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "version": self.version,
            "flags": self.flags,
            "packBytes": self.pack_bytes,
            "entryCount": self.entry_count,
            "totalWgslBytes": self.total_wgsl_bytes,
            "stageCounts": dict(self.stage_counts),
            "featureHistogram": dict(self.feature_histogram),
            "violationCount": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "shaders": [s.to_dict() for s in self.shaders],
            "limits": self.limits.to_dict(),
            "clean": self.clean,
        }


@dataclass(frozen=True)
class ChangedShaderDiff:
    """Per-shader diff details for a hash present in both packs with differences."""

    shader_hash: int
    stage_before: str
    stage_after: str
    bytes_before: int
    bytes_after: int
    byte_delta: int
    source_changed: bool
    audit_changed: bool
    violations_before: int
    violations_after: int
    added_features: tuple[str, ...]
    removed_features: tuple[str, ...]

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.shader_hash)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash_hex,
            "stageBefore": self.stage_before,
            "stageAfter": self.stage_after,
            "bytesBefore": self.bytes_before,
            "bytesAfter": self.bytes_after,
            "byteDelta": self.byte_delta,
            "sourceChanged": self.source_changed,
            "auditChanged": self.audit_changed,
            "violationsBefore": self.violations_before,
            "violationsAfter": self.violations_after,
            "addedFeatures": list(self.added_features),
            "removedFeatures": list(self.removed_features),
        }


@dataclass(frozen=True)
class PackDiffReport:
    """Structured diff between two binary `.wgsl.pack` bundles."""

    pack_a: str
    pack_b: str
    entry_count_a: int
    entry_count_b: int
    total_wgsl_bytes_a: int
    total_wgsl_bytes_b: int
    pack_bytes_a: int
    pack_bytes_b: int
    added_hashes: tuple[int, ...]
    removed_hashes: tuple[int, ...]
    changed_hashes: tuple[int, ...]
    unchanged_count: int
    byte_delta: int
    pack_byte_delta: int
    changed: tuple[ChangedShaderDiff, ...]
    stage_count_delta: dict[str, int]
    violation_count_a: int
    violation_count_b: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "packA": self.pack_a,
            "packB": self.pack_b,
            "entryCountA": self.entry_count_a,
            "entryCountB": self.entry_count_b,
            "entryCountDelta": self.entry_count_b - self.entry_count_a,
            "totalWgslBytesA": self.total_wgsl_bytes_a,
            "totalWgslBytesB": self.total_wgsl_bytes_b,
            "byteDelta": self.byte_delta,
            "packBytesA": self.pack_bytes_a,
            "packBytesB": self.pack_bytes_b,
            "packByteDelta": self.pack_byte_delta,
            "addedHashes": [format_shader_hash(h) for h in self.added_hashes],
            "removedHashes": [format_shader_hash(h) for h in self.removed_hashes],
            "changedHashes": [format_shader_hash(h) for h in self.changed_hashes],
            "unchangedCount": self.unchanged_count,
            "changed": [item.to_dict() for item in self.changed],
            "stageCountDelta": dict(self.stage_count_delta),
            "violationCountA": self.violation_count_a,
            "violationCountB": self.violation_count_b,
            "violationDelta": self.violation_count_b - self.violation_count_a,
        }


def _load_bundle(
    path_or_bytes: str | Path | bytes | bytearray | memoryview | WgpkBundle,
    *,
    source_label: str | None = None,
) -> tuple[WgpkBundle, str]:
    """Load a `WgpkBundle` from a path, raw bytes, or existing `WgpkBundle`."""
    if isinstance(path_or_bytes, WgpkBundle):
        return path_or_bytes, sanitize_source_label(source_label)
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        return parse_wgpk(path_or_bytes), sanitize_source_label(source_label)
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        label = sanitize_source_label(source_label if source_label is not None else path)
        raw = path.read_bytes()
        return parse_wgpk(raw), label
    raise TypeError(
        f"expected path, bytes-like buffer, or WgpkBundle, got {type(path_or_bytes).__name__}"
    )


def inspect_pack(
    path_or_bytes: str | Path | bytes | bytearray | memoryview | WgpkBundle,
    *,
    limits: WebGpuLimits | None = None,
    source_label: str | None = None,
) -> PackInspectReport:
    """Parse and audit all shaders in a `.wgsl.pack` bundle against WebGPU limits."""
    active_limits = limits if limits is not None else DEFAULT_WEBGPU_LIMITS
    bundle, label = _load_bundle(path_or_bytes, source_label=source_label)

    stage_counts = {stage: 0 for stage in STAGE_ORDER}
    feature_counter: Counter[str] = Counter()
    audits: list[ShaderAudit] = []
    all_violations: list[BudgetViolation] = []
    seen_hashes: set[int] = set()

    for entry in bundle.entries:
        stage_counts[entry.stage_name] += 1
        audit = analyze_wgsl(
            entry.wgsl,
            entry.hash,
            limits=active_limits,
            pack_stage=entry.stage_name,
        )
        audits.append(audit)
        for feat in audit.enabled_features:
            feature_counter[feat] += 1

        if entry.hash in seen_hashes:
            all_violations.append(
                BudgetViolation(
                    shader_hash=entry.hash,
                    code="duplicate_shader_hash",
                    message=f"duplicate shader hash {entry.hash_hex} in WGPK entry table",
                )
            )
        else:
            seen_hashes.add(entry.hash)

        all_violations.extend(audit.violations)

    feature_histogram = dict(sorted(feature_counter.items()))

    return PackInspectReport(
        source=label,
        version=bundle.version,
        flags=bundle.flags,
        pack_bytes=bundle.byte_length,
        entry_count=bundle.entry_count,
        total_wgsl_bytes=bundle.total_wgsl_bytes,
        stage_counts=stage_counts,
        feature_histogram=feature_histogram,
        violations=tuple(all_violations),
        shaders=tuple(audits),
        limits=active_limits,
    )


def diff_packs(
    pack_a: str | Path | bytes | bytearray | memoryview | WgpkBundle,
    pack_b: str | Path | bytes | bytearray | memoryview | WgpkBundle,
    *,
    limits: WebGpuLimits | None = None,
    label_a: str | None = None,
    label_b: str | None = None,
) -> PackDiffReport:
    """Compare two `.wgsl.pack` bundles and compute added/removed/changed shaders and deltas."""
    active_limits = limits if limits is not None else DEFAULT_WEBGPU_LIMITS
    bundle_a, sanitized_a = _load_bundle(pack_a, source_label=label_a)
    bundle_b, sanitized_b = _load_bundle(pack_b, source_label=label_b)

    report_a = inspect_pack(bundle_a, limits=active_limits, source_label=sanitized_a)
    report_b = inspect_pack(bundle_b, limits=active_limits, source_label=sanitized_b)

    map_a: dict[int, tuple[WgpkEntry, ShaderAudit]] = {
        entry.hash: (entry, audit)
        for entry, audit in zip(bundle_a.entries, report_a.shaders)
    }
    map_b: dict[int, tuple[WgpkEntry, ShaderAudit]] = {
        entry.hash: (entry, audit)
        for entry, audit in zip(bundle_b.entries, report_b.shaders)
    }

    hashes_a = set(map_a.keys())
    hashes_b = set(map_b.keys())

    added_hashes = tuple(sorted(hashes_b - hashes_a))
    removed_hashes = tuple(sorted(hashes_a - hashes_b))
    common_hashes = sorted(hashes_a & hashes_b)

    changed_hashes: list[int] = []
    changed_details: list[ChangedShaderDiff] = []
    unchanged_count = 0

    for shader_hash in common_hashes:
        entry_a, audit_a = map_a[shader_hash]
        entry_b, audit_b = map_b[shader_hash]

        source_changed = (
            entry_a.length != entry_b.length
            or entry_a.wgsl != entry_b.wgsl
            or entry_a.stage != entry_b.stage
        )
        audit_changed = audit_a != audit_b

        if source_changed or audit_changed:
            changed_hashes.append(shader_hash)
            feats_a = set(audit_a.enabled_features)
            feats_b = set(audit_b.enabled_features)
            changed_details.append(
                ChangedShaderDiff(
                    shader_hash=shader_hash,
                    stage_before=entry_a.stage_name,
                    stage_after=entry_b.stage_name,
                    bytes_before=entry_a.length,
                    bytes_after=entry_b.length,
                    byte_delta=entry_b.length - entry_a.length,
                    source_changed=source_changed,
                    audit_changed=audit_changed,
                    violations_before=len(audit_a.violations),
                    violations_after=len(audit_b.violations),
                    added_features=tuple(sorted(feats_b - feats_a)),
                    removed_features=tuple(sorted(feats_a - feats_b)),
                )
            )
        else:
            unchanged_count += 1

    stage_count_delta = {
        stage: report_b.stage_counts[stage] - report_a.stage_counts[stage]
        for stage in STAGE_ORDER
    }

    return PackDiffReport(
        pack_a=sanitized_a,
        pack_b=sanitized_b,
        entry_count_a=report_a.entry_count,
        entry_count_b=report_b.entry_count,
        total_wgsl_bytes_a=report_a.total_wgsl_bytes,
        total_wgsl_bytes_b=report_b.total_wgsl_bytes,
        pack_bytes_a=report_a.pack_bytes,
        pack_bytes_b=report_b.pack_bytes,
        added_hashes=added_hashes,
        removed_hashes=removed_hashes,
        changed_hashes=tuple(changed_hashes),
        unchanged_count=unchanged_count,
        byte_delta=report_b.total_wgsl_bytes - report_a.total_wgsl_bytes,
        pack_byte_delta=report_b.pack_bytes - report_a.pack_bytes,
        changed=tuple(changed_details),
        stage_count_delta=stage_count_delta,
        violation_count_a=len(report_a.violations),
        violation_count_b=len(report_b.violations),
    )


# ---------------------------------------------------------------------------
# Pipeline dump & reflection sidecar coverage diff (T45 / #931)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DumpStageIdentity:
    """Single unique shader stage identity extracted from a pipeline dump."""

    hash: int
    stage: str   # vertex | fragment | compute | unknown
    origin: str  # native | dxil

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.hash)


@dataclass(frozen=True)
class PipelineDumpSummary:
    """Unique shader stage identities discovered in a pipeline dump directory."""

    source: str
    manifest_loaded: bool
    stages: tuple[DumpStageIdentity, ...]


@dataclass(frozen=True)
class ReflectSidecarSummary:
    """Validated stage-reflection sidecar (`<pack>.reflect.json`) hash index."""

    source: str
    schema: str | None
    version: int | None
    stage_hashes: frozenset[int]
    stage_map: dict[int, str]


@dataclass(frozen=True)
class StageCoverageCounts:
    """Coverage counts for one shader stage or origin bucket."""

    dump_total: int
    in_pack: int
    missing_from_pack: int
    in_reflect: int
    missing_from_reflect: int

    def to_dict(self) -> dict[str, int]:
        return {
            "dump_total": self.dump_total,
            "in_pack": self.in_pack,
            "missing_from_pack": self.missing_from_pack,
            "in_reflect": self.in_reflect,
            "missing_from_reflect": self.missing_from_reflect,
        }


@dataclass(frozen=True)
class DumpDiffReport:
    """Sanitized coverage diff of a pipeline dump against a `.wgsl.pack` and `.reflect.json`."""

    pack_source: str
    dump_source: str
    reflect_source: str | None
    reflect_loaded: bool
    manifest_loaded: bool
    pack_entry_count: int
    reflect_stage_count: int
    dump_total_stages: int
    in_pack: int
    missing_from_pack: int
    in_reflect: int
    missing_from_reflect: int
    in_pack_missing_from_reflect: int
    by_stage: dict[str, StageCoverageCounts]
    by_origin: dict[str, StageCoverageCounts]
    max_samples: int
    sample_missing_pack_hashes: tuple[str, ...]
    sample_missing_reflect_hashes: tuple[str, ...]

    @property
    def pack_coverage_pct(self) -> float:
        if self.dump_total_stages == 0:
            return 100.0
        return round((self.in_pack / self.dump_total_stages) * 100.0, 2)

    @property
    def reflect_coverage_pct(self) -> float:
        if self.dump_total_stages == 0:
            return 100.0
        return round((self.in_reflect / self.dump_total_stages) * 100.0, 2)

    @property
    def clean(self) -> bool:
        if self.missing_from_pack > 0:
            return False
        if self.reflect_loaded and self.missing_from_reflect > 0:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": DUMP_DIFF_REPORT_SCHEMA,
            "version": DUMP_DIFF_REPORT_VERSION,
            "source": self.pack_source,
            "dump_source": self.dump_source,
            "reflect_source": self.reflect_source,
            "reflect_loaded": self.reflect_loaded,
            "manifest_loaded": self.manifest_loaded,
            "pack_entry_count": self.pack_entry_count,
            "reflect_stage_count": self.reflect_stage_count,
            "dump_total_stages": self.dump_total_stages,
            "in_pack": self.in_pack,
            "missing_from_pack": self.missing_from_pack,
            "in_reflect": self.in_reflect,
            "missing_from_reflect": self.missing_from_reflect,
            "in_pack_missing_from_reflect": self.in_pack_missing_from_reflect,
            "pack_coverage_pct": self.pack_coverage_pct,
            "reflect_coverage_pct": self.reflect_coverage_pct,
            "by_stage": {
                stage: self.by_stage[stage].to_dict()
                for stage in DUMP_STAGE_ORDER
                if stage in self.by_stage
            },
            "by_origin": {
                origin: self.by_origin[origin].to_dict()
                for origin in DUMP_ORIGIN_ORDER
                if origin in self.by_origin
            },
            "max_samples": self.max_samples,
            "sample_missing_pack_hashes": list(self.sample_missing_pack_hashes),
            "sample_missing_reflect_hashes": list(self.sample_missing_reflect_hashes),
            "missing_pack_hashes_truncated": self.missing_from_pack > len(self.sample_missing_pack_hashes),
            "missing_reflect_hashes_truncated": (
                self.missing_from_reflect > len(self.sample_missing_reflect_hashes)
            ),
            "clean": self.clean,
        }


def _normalize_dump_stage(raw: Any) -> str:
    """Normalize a pipeline dump or reflection stage name into vertex|fragment|compute|unknown."""
    if not isinstance(raw, str):
        return "unknown"
    return _DUMP_STAGE_ALIASES.get(raw.strip().lower(), "unknown")


def _parse_dump_filename_identity(filename: str) -> tuple[int | None, str]:
    """Parse a 64-bit hex shader hash and optional stage hint from a dump blob filename.

    Supports canonical `0x<16-hex>.bin`, `<16-hex>.bin`, `0x<16-hex>.dxbc`, as well as
    optional `<stage>_0x<16-hex>.bin` / `0x<16-hex>_<stage>.bin` prefixes/suffixes.
    """
    base = sanitize_source_label(filename)
    if not base or base.startswith("<"):
        return None, "unknown"
    stem = base.rsplit(".", 1)[0] if "." in base else base
    if not stem:
        return None, "unknown"

    direct = _FILENAME_HEX16_RE.fullmatch(stem)
    if direct:
        return int(direct.group(1), 16), "unknown"

    parts = [part for part in re.split(r"[_-]+", stem) if part]
    if len(parts) < 2:
        return None, "unknown"

    stage = "unknown"
    remaining = parts
    first_stage = _DUMP_STAGE_ALIASES.get(parts[0].lower())
    if first_stage is not None:
        stage = first_stage
        remaining = parts[1:]
    else:
        last_stage = _DUMP_STAGE_ALIASES.get(parts[-1].lower())
        if last_stage is not None:
            stage = last_stage
            remaining = parts[:-1]

    if len(remaining) == 1:
        match = _FILENAME_HEX16_RE.fullmatch(remaining[0])
        if match:
            return int(match.group(1), 16), stage
    return None, "unknown"


def _record_dump_stage(
    stages: dict[int, DumpStageIdentity],
    shader_hash: int,
    stage: str,
    origin: str,
) -> None:
    """Insert or refine a unique stage identity (preferring known stage and native origin)."""
    existing = stages.get(shader_hash)
    if existing is None:
        stages[shader_hash] = DumpStageIdentity(hash=shader_hash, stage=stage, origin=origin)
        return
    resolved_stage = existing.stage if existing.stage != "unknown" else stage
    resolved_origin = "native" if existing.origin == "native" or origin == "native" else existing.origin
    if resolved_stage != existing.stage or resolved_origin != existing.origin:
        stages[shader_hash] = DumpStageIdentity(
            hash=shader_hash,
            stage=resolved_stage,
            origin=resolved_origin,
        )


def _scan_dump_blob_dir(
    directory: Path,
    *,
    suffix: str,
    origin: str,
    stages: dict[int, DumpStageIdentity],
    source_label: str,
) -> None:
    """Scan a dump subdirectory for shader blob filenames without reading shader bytes."""
    try:
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        raise DumpDiffError(
            _sanitize_os_error(source_label, exc),
            code="io_error",
            source=source_label,
        ) from exc

    for child in entries:
        if child.suffix.lower() != suffix:
            continue
        try:
            if not child.is_file():
                continue
        except OSError:
            continue
        shader_hash, stage = _parse_dump_filename_identity(child.name)
        if shader_hash is not None:
            _record_dump_stage(stages, shader_hash, stage, origin)


def load_pipeline_dump(
    dump_dir: str | Path,
    *,
    source_label: str | None = None,
) -> PipelineDumpSummary:
    """Read unique native/DXIL stage identities from a `playgta6-pipeline-dump` v1 directory.

    Reads `<dir>/manifest.json` (if present) and scans `<dir>/native/*.bin` and
    `<dir>/dxil/*.dxbc` filenames (or `<dir>/*.bin`/`*.dxbc` directly when pointed at
    a bare blob directory without `manifest.json` or `native/`/`dxil/` subdirs).
    Never reads or returns raw shader binary bytes.
    """
    root = Path(dump_dir)
    label = sanitize_source_label(source_label if source_label is not None else root)

    if not root.exists():
        raise DumpDiffError(
            f"pipeline dump directory not found: {label}",
            code="dump_not_found",
            source=label,
        )
    if not root.is_dir():
        raise DumpDiffError(
            f"pipeline dump path is not a directory: {label}",
            code="invalid_dump_dir",
            source=label,
        )

    stages: dict[int, DumpStageIdentity] = {}
    manifest_loaded = False
    manifest_path = root / "manifest.json"

    if manifest_path.exists():
        manifest_label = sanitize_source_label(manifest_path)
        if not manifest_path.is_file():
            raise DumpDiffError(
                f"invalid pipeline dump manifest {manifest_label}: path is not a file",
                code="invalid_manifest",
                source=label,
            )
        try:
            raw_text = manifest_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise DumpDiffError(
                _sanitize_os_error(manifest_label, exc),
                code="io_error",
                source=label,
            ) from exc
        except UnicodeDecodeError as exc:
            raise DumpDiffError(
                f"invalid UTF-8 in pipeline dump manifest {manifest_label}: {exc.reason}",
                code="invalid_manifest",
                source=label,
            ) from exc

        try:
            doc = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DumpDiffError(
                f"invalid JSON in pipeline dump manifest {manifest_label} "
                f"(line {exc.lineno}, col {exc.colno}): {exc.msg}",
                code="invalid_manifest",
                source=label,
            ) from exc

        if not isinstance(doc, Mapping):
            raise DumpDiffError(
                f"invalid pipeline dump manifest {manifest_label}: top-level value must be a JSON object",
                code="invalid_manifest",
                source=label,
            )

        schema = doc.get("schema")
        if schema != PIPELINE_DUMP_SCHEMA:
            raise DumpDiffError(
                f"unsupported pipeline dump manifest schema in {manifest_label} "
                f"(expected {PIPELINE_DUMP_SCHEMA!r})",
                code="invalid_manifest",
                source=label,
            )

        if "schema_version" in doc:
            ver = doc.get("schema_version")
            if isinstance(ver, bool) or not isinstance(ver, int) or ver != PIPELINE_DUMP_SCHEMA_VERSION:
                raise DumpDiffError(
                    f"unsupported pipeline dump schema_version in {manifest_label} "
                    f"(expected {PIPELINE_DUMP_SCHEMA_VERSION})",
                    code="invalid_manifest",
                    source=label,
                )

        native_stages = doc.get("native_stages")
        if native_stages is not None:
            if not isinstance(native_stages, list):
                raise DumpDiffError(
                    f"invalid pipeline dump manifest {manifest_label}: 'native_stages' must be a list",
                    code="invalid_manifest",
                    source=label,
                )
            for idx, item in enumerate(native_stages):
                if not isinstance(item, Mapping):
                    raise DumpDiffError(
                        f"invalid pipeline dump manifest {manifest_label}: native_stages[{idx}] must be an object",
                        code="invalid_manifest",
                        source=label,
                    )
                raw_id = item.get("code_identity")
                if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
                    raw_id = item.get("identity")
                if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
                    continue
                try:
                    shader_hash = normalize_shader_hash(raw_id)
                except (TypeError, ValueError) as exc:
                    raise DumpDiffError(
                        f"invalid shader identity at native_stages[{idx}] in {manifest_label}",
                        code="invalid_manifest",
                        source=label,
                    ) from exc
                stage = _normalize_dump_stage(item.get("stage"))
                _record_dump_stage(stages, shader_hash, stage, "native")

        dxil_entries = doc.get("dxil")
        if dxil_entries is not None:
            if not isinstance(dxil_entries, list):
                raise DumpDiffError(
                    f"invalid pipeline dump manifest {manifest_label}: 'dxil' must be a list",
                    code="invalid_manifest",
                    source=label,
                )
            for idx, item in enumerate(dxil_entries):
                if not isinstance(item, Mapping):
                    raise DumpDiffError(
                        f"invalid pipeline dump manifest {manifest_label}: dxil[{idx}] must be an object",
                        code="invalid_manifest",
                        source=label,
                    )
                raw_id = item.get("identity")
                if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
                    continue
                try:
                    shader_hash = normalize_shader_hash(raw_id)
                except (TypeError, ValueError) as exc:
                    raise DumpDiffError(
                        f"invalid shader identity at dxil[{idx}] in {manifest_label}",
                        code="invalid_manifest",
                        source=label,
                    ) from exc

                stage = _normalize_dump_stage(item.get("stage"))
                if stage == "unknown":
                    raw_kind = item.get("kind")
                    if isinstance(raw_kind, int) and not isinstance(raw_kind, bool):
                        stage = _DXIL_KIND_TO_STAGE.get(raw_kind, "unknown")
                _record_dump_stage(stages, shader_hash, stage, "dxil")

        manifest_loaded = True

    native_dir = root / "native"
    dxil_dir = root / "dxil"
    has_native_subdir = native_dir.is_dir()
    has_dxil_subdir = dxil_dir.is_dir()

    if has_native_subdir:
        _scan_dump_blob_dir(
            native_dir,
            suffix=".bin",
            origin="native",
            stages=stages,
            source_label=label,
        )
    if has_dxil_subdir:
        _scan_dump_blob_dir(
            dxil_dir,
            suffix=".dxbc",
            origin="dxil",
            stages=stages,
            source_label=label,
        )
    if not manifest_loaded and not has_native_subdir and not has_dxil_subdir:
        _scan_dump_blob_dir(
            root,
            suffix=".bin",
            origin="native",
            stages=stages,
            source_label=label,
        )
        _scan_dump_blob_dir(
            root,
            suffix=".dxbc",
            origin="dxil",
            stages=stages,
            source_label=label,
        )

    ordered = tuple(stages[h] for h in sorted(stages))
    return PipelineDumpSummary(
        source=label,
        manifest_loaded=manifest_loaded,
        stages=ordered,
    )


def resolve_reflect_sidecar_path(
    pack_path: str | Path | None,
    reflect_path: str | Path | None = None,
) -> Path | None:
    """Resolve an explicit `--reflect` path or auto-detect adjacent `<pack>.reflect.json`."""
    if reflect_path is not None:
        return Path(reflect_path)
    if pack_path is None:
        return None
    raw = str(pack_path)
    if raw == "-":
        return None
    base = Path(pack_path)
    primary = Path(f"{raw}.reflect.json")
    if primary.is_file():
        return primary
    alt = base.with_suffix(".reflect.json")
    if alt != primary and alt.is_file():
        return alt
    return None


def load_reflect_sidecar(
    reflect_path: str | Path,
    *,
    source_label: str | None = None,
) -> ReflectSidecarSummary:
    """Load and validate a `<pack>.reflect.json` sidecar file and return its stage hash set."""
    path = Path(reflect_path)
    label = sanitize_source_label(source_label if source_label is not None else path)

    if not path.exists():
        raise DumpDiffError(
            f"reflection sidecar not found: {label}",
            code="reflect_not_found",
            source=label,
        )
    if not path.is_file():
        raise DumpDiffError(
            f"reflection sidecar path is not a file: {label}",
            code="invalid_reflect_sidecar",
            source=label,
        )

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DumpDiffError(
            _sanitize_os_error(label, exc),
            code="io_error",
            source=label,
        ) from exc
    except UnicodeDecodeError as exc:
        raise DumpDiffError(
            f"invalid UTF-8 in reflection sidecar {label}: {exc.reason}",
            code="invalid_reflect_sidecar",
            source=label,
        ) from exc

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DumpDiffError(
            f"invalid JSON in reflection sidecar {label} "
            f"(line {exc.lineno}, col {exc.colno}): {exc.msg}",
            code="invalid_reflect_sidecar",
            source=label,
        ) from exc

    if not isinstance(doc, Mapping):
        raise DumpDiffError(
            f"invalid reflection sidecar {label}: top-level value must be a JSON object",
            code="invalid_reflect_sidecar",
            source=label,
        )

    schema_val = doc.get("schema")
    if schema_val is not None:
        if not isinstance(schema_val, str) or schema_val != REFLECT_SIDECAR_SCHEMA:
            raise DumpDiffError(
                f"unsupported reflection sidecar schema in {label} "
                f"(expected {REFLECT_SIDECAR_SCHEMA!r})",
                code="invalid_reflect_sidecar",
                source=label,
            )

    version_val = doc.get("version")
    if version_val is not None:
        if isinstance(version_val, bool) or not isinstance(version_val, int) or version_val != REFLECT_SIDECAR_VERSION:
            raise DumpDiffError(
                f"unsupported reflection sidecar version in {label} "
                f"(expected {REFLECT_SIDECAR_VERSION})",
                code="invalid_reflect_sidecar",
                source=label,
            )

    stages_obj = doc.get("stages")
    if not isinstance(stages_obj, Mapping):
        raise DumpDiffError(
            f"invalid reflection sidecar {label}: missing or non-object 'stages' map",
            code="invalid_reflect_sidecar",
            source=label,
        )

    stage_map: dict[int, str] = {}
    for raw_key, raw_entry in stages_obj.items():
        if not isinstance(raw_key, str) or not isinstance(raw_entry, Mapping):
            raise DumpDiffError(
                f"invalid reflection sidecar {label}: every 'stages' entry must map a hash string to an object",
                code="invalid_reflect_sidecar",
                source=label,
            )
        try:
            stage_hash = normalize_shader_hash(raw_key)
        except (TypeError, ValueError) as exc:
            raise DumpDiffError(
                f"invalid shader hash key in reflection sidecar {label}",
                code="invalid_reflect_sidecar",
                source=label,
            ) from exc
        stage_map[stage_hash] = _normalize_dump_stage(raw_entry.get("stage"))

    return ReflectSidecarSummary(
        source=label,
        schema=schema_val,
        version=version_val,
        stage_hashes=frozenset(stage_map.keys()),
        stage_map=stage_map,
    )


def _load_toc(
    path_or_bytes: str | Path | bytes | bytearray | memoryview | WgpkBundle | WgpkToc,
    *,
    source_label: str | None = None,
) -> tuple[WgpkToc, str]:
    """Load a `WgpkToc` from a path, raw bytes, `WgpkBundle`, or `WgpkToc`."""
    if isinstance(path_or_bytes, WgpkToc):
        return path_or_bytes, sanitize_source_label(source_label)
    if isinstance(path_or_bytes, WgpkBundle):
        toc = WgpkToc(
            version=path_or_bytes.version,
            flags=path_or_bytes.flags,
            byte_length=path_or_bytes.byte_length,
            entries=tuple(
                WgpkTocEntry(
                    hash=e.hash,
                    stage=e.stage,
                    offset=e.offset,
                    length=e.length,
                )
                for e in path_or_bytes.entries
            ),
        )
        return toc, sanitize_source_label(source_label)
    if isinstance(path_or_bytes, (bytes, bytearray, memoryview)):
        return parse_wgpk_toc(path_or_bytes), sanitize_source_label(source_label)
    if isinstance(path_or_bytes, (str, Path)):
        path = Path(path_or_bytes)
        label = sanitize_source_label(source_label if source_label is not None else path)
        raw = path.read_bytes()
        return parse_wgpk_toc(raw), label
    raise TypeError(
        f"expected path, bytes-like buffer, WgpkBundle, or WgpkToc, got {type(path_or_bytes).__name__}"
    )


def diff_pack_against_dump(
    pack_or_bytes: str | Path | bytes | bytearray | memoryview | WgpkBundle | WgpkToc,
    dump_dir: str | Path | PipelineDumpSummary,
    *,
    reflect_path: str | Path | ReflectSidecarSummary | None = None,
    max_samples: int = DEFAULT_MAX_MISSING_SAMPLES,
    pack_label: str | None = None,
    dump_label: str | None = None,
    reflect_label: str | None = None,
) -> DumpDiffReport:
    """Compare unique native/DXIL stage identities in a pipeline dump against a `.wgsl.pack`
    TOC and optional `.reflect.json` sidecar (auto-detecting `<pack>.reflect.json` when
    `pack_or_bytes` is a filesystem path and `reflect_path` is omitted).
    """
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 0:
        raise DumpDiffError(
            f"max_samples must be a non-negative integer, got {max_samples!r}",
            code="invalid_argument",
        )

    toc, sanitized_pack = _load_toc(pack_or_bytes, source_label=pack_label)

    if isinstance(dump_dir, PipelineDumpSummary):
        dump_summary = dump_dir
    else:
        dump_summary = load_pipeline_dump(dump_dir, source_label=dump_label)

    reflect_summary: ReflectSidecarSummary | None = None
    if isinstance(reflect_path, ReflectSidecarSummary):
        reflect_summary = reflect_path
    else:
        pack_path_hint = pack_or_bytes if isinstance(pack_or_bytes, (str, Path)) else None
        resolved_reflect = resolve_reflect_sidecar_path(pack_path_hint, reflect_path)
        if resolved_reflect is not None:
            reflect_summary = load_reflect_sidecar(resolved_reflect, source_label=reflect_label)

    pack_hashes = toc.hashes
    reflect_hashes = reflect_summary.stage_hashes if reflect_summary is not None else frozenset()

    stage_buckets: dict[str, list[int]] = {
        stage: [0, 0, 0, 0, 0] for stage in DUMP_STAGE_ORDER
    }
    origin_buckets: dict[str, list[int]] = {
        origin: [0, 0, 0, 0, 0] for origin in DUMP_ORIGIN_ORDER
    }

    in_pack = 0
    missing_from_pack = 0
    in_reflect = 0
    missing_from_reflect = 0
    in_pack_missing_from_reflect = 0
    sample_missing_pack: list[str] = []
    sample_missing_reflect: list[str] = []

    for item in dump_summary.stages:
        has_pack = item.hash in pack_hashes
        has_reflect = item.hash in reflect_hashes

        stage_key = item.stage if item.stage in stage_buckets else "unknown"
        origin_key = item.origin if item.origin in origin_buckets else "native"

        sb = stage_buckets[stage_key]
        ob = origin_buckets[origin_key]
        sb[0] += 1
        ob[0] += 1

        if has_pack:
            in_pack += 1
            sb[1] += 1
            ob[1] += 1
        else:
            missing_from_pack += 1
            sb[2] += 1
            ob[2] += 1
            if len(sample_missing_pack) < max_samples:
                sample_missing_pack.append(item.hash_hex)

        if has_reflect:
            in_reflect += 1
            sb[3] += 1
            ob[3] += 1
        else:
            missing_from_reflect += 1
            sb[4] += 1
            ob[4] += 1
            if len(sample_missing_reflect) < max_samples:
                sample_missing_reflect.append(item.hash_hex)
            if has_pack:
                in_pack_missing_from_reflect += 1

    by_stage = {
        stage: StageCoverageCounts(*vals) for stage, vals in stage_buckets.items()
    }
    by_origin = {
        origin: StageCoverageCounts(*vals) for origin, vals in origin_buckets.items()
    }

    return DumpDiffReport(
        pack_source=sanitized_pack,
        dump_source=dump_summary.source,
        reflect_source=reflect_summary.source if reflect_summary is not None else None,
        reflect_loaded=reflect_summary is not None,
        manifest_loaded=dump_summary.manifest_loaded,
        pack_entry_count=toc.entry_count,
        reflect_stage_count=len(reflect_hashes),
        dump_total_stages=len(dump_summary.stages),
        in_pack=in_pack,
        missing_from_pack=missing_from_pack,
        in_reflect=in_reflect,
        missing_from_reflect=missing_from_reflect,
        in_pack_missing_from_reflect=in_pack_missing_from_reflect,
        by_stage=by_stage,
        by_origin=by_origin,
        max_samples=max_samples,
        sample_missing_pack_hashes=tuple(sample_missing_pack),
        sample_missing_reflect_hashes=tuple(sample_missing_reflect),
    )


# ---------------------------------------------------------------------------
# Human-readable text formatters
# ---------------------------------------------------------------------------


def format_inspect_text(report: PackInspectReport) -> str:
    """Format a human-readable summary of a `PackInspectReport`."""
    lines = [
        f"WGPK Shader Pack: {report.source}",
        f"  Version:        {report.version} (flags=0x{report.flags:08x})",
        f"  Pack size:      {report.pack_bytes} bytes ({report.total_wgsl_bytes} WGSL bytes)",
        f"  Entries:        {report.entry_count} "
        f"(vertex={report.stage_counts['vertex']}, "
        f"fragment={report.stage_counts['fragment']}, "
        f"compute={report.stage_counts['compute']})",
    ]
    if report.feature_histogram:
        feat_str = ", ".join(f"{k}={v}" for k, v in report.feature_histogram.items())
        lines.append(f"  Features:       {feat_str}")
    else:
        lines.append("  Features:       (none)")

    lines.append(f"  Violations:     {len(report.violations)}")
    if report.violations:
        lines.append("")
        lines.append("Budget Violations:")
        for v in report.violations:
            lines.append(f"  - [{v.hash_hex}] {v.code}: {v.message}")

    return "\n".join(lines)


def format_diff_text(report: PackDiffReport) -> str:
    """Format a human-readable summary of a `PackDiffReport`."""
    sign = "+" if report.byte_delta >= 0 else ""
    pack_sign = "+" if report.pack_byte_delta >= 0 else ""
    lines = [
        f"WGPK Pack Diff: {report.pack_a} -> {report.pack_b}",
        f"  Entries:        {report.entry_count_a} -> {report.entry_count_b} "
        f"({report.entry_count_b - report.entry_count_a:+d})",
        f"  WGSL bytes:     {report.total_wgsl_bytes_a} -> {report.total_wgsl_bytes_b} "
        f"({sign}{report.byte_delta})",
        f"  Pack bytes:     {report.pack_bytes_a} -> {report.pack_bytes_b} "
        f"({pack_sign}{report.pack_byte_delta})",
        f"  Added:          {len(report.added_hashes)}",
        f"  Removed:        {len(report.removed_hashes)}",
        f"  Changed:        {len(report.changed_hashes)}",
        f"  Unchanged:      {report.unchanged_count}",
        f"  Violations:     {report.violation_count_a} -> {report.violation_count_b}",
    ]
    if report.added_hashes:
        lines.append("  Added hashes:   " + ", ".join(format_shader_hash(h) for h in report.added_hashes))
    if report.removed_hashes:
        lines.append("  Removed hashes: " + ", ".join(format_shader_hash(h) for h in report.removed_hashes))
    if report.changed:
        lines.append("  Changed entries:")
        for item in report.changed:
            lines.append(
                f"    - {item.hash_hex}: {item.bytes_before} -> {item.bytes_after} bytes "
                f"({item.byte_delta:+d}), violations {item.violations_before} -> {item.violations_after}"
            )
    return "\n".join(lines)


def format_dump_diff_text(report: DumpDiffReport) -> str:
    """Format a strictly sanitized human-readable summary of a `DumpDiffReport`."""
    manifest_status = (
        f"loaded ({PIPELINE_DUMP_SCHEMA} v{PIPELINE_DUMP_SCHEMA_VERSION})"
        if report.manifest_loaded
        else "absent (scanned blob filenames)"
    )
    reflect_status = (
        f"{report.reflect_source} ({report.reflect_stage_count} stages)"
        if report.reflect_loaded and report.reflect_source is not None
        else "(none)"
    )
    lines = [
        f"WGPK Pipeline Dump Coverage: {report.pack_source} vs {report.dump_source}",
        f"  Manifest:              {manifest_status}",
        f"  Reflect sidecar:       {reflect_status}",
        f"  Pack TOC entries:      {report.pack_entry_count}",
        f"  Dump total stages:     {report.dump_total_stages}",
        f"  In pack:               {report.in_pack} / {report.dump_total_stages} ({report.pack_coverage_pct:.2f}%)",
        f"  Missing from pack:     {report.missing_from_pack}",
        f"  In reflect:            {report.in_reflect} / {report.dump_total_stages} ({report.reflect_coverage_pct:.2f}%)",
        f"  Missing from reflect:  {report.missing_from_reflect}"
        f" (in pack but missing reflect: {report.in_pack_missing_from_reflect})",
        "  By stage:",
    ]
    for stage in DUMP_STAGE_ORDER:
        counts = report.by_stage[stage]
        lines.append(
            f"    {stage:<9} total={counts.dump_total}, in_pack={counts.in_pack}, "
            f"missing_pack={counts.missing_from_pack}, in_reflect={counts.in_reflect}, "
            f"missing_reflect={counts.missing_from_reflect}"
        )

    lines.append("  By origin:")
    for origin in DUMP_ORIGIN_ORDER:
        counts = report.by_origin[origin]
        lines.append(
            f"    {origin:<9} total={counts.dump_total}, in_pack={counts.in_pack}, "
            f"missing_pack={counts.missing_from_pack}, in_reflect={counts.in_reflect}, "
            f"missing_reflect={counts.missing_from_reflect}"
        )

    if report.sample_missing_pack_hashes:
        shown = len(report.sample_missing_pack_hashes)
        extra = report.missing_from_pack - shown
        suffix = f" (+{extra} more)" if extra > 0 else ""
        lines.append(
            f"  Sample missing from pack ({shown} of {report.missing_from_pack}){suffix}:"
        )
        lines.append("    " + ", ".join(report.sample_missing_pack_hashes))
    elif report.missing_from_pack > 0:
        lines.append(
            f"  Sample missing from pack (0 of {report.missing_from_pack}, max_samples={report.max_samples})"
        )

    if report.sample_missing_reflect_hashes:
        shown = len(report.sample_missing_reflect_hashes)
        extra = report.missing_from_reflect - shown
        suffix = f" (+{extra} more)" if extra > 0 else ""
        lines.append(
            f"  Sample missing from reflect ({shown} of {report.missing_from_reflect}){suffix}:"
        )
        lines.append("    " + ", ".join(report.sample_missing_reflect_hashes))
    elif report.missing_from_reflect > 0:
        lines.append(
            f"  Sample missing from reflect (0 of {report.missing_from_reflect}, max_samples={report.max_samples})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inspect_wgsl_pack.py",
        description=(
            "Inspect, validate WebGPU limits, diff binary WGPK (.wgsl.pack) "
            "shader bundles, and audit coverage against pipeline dumps."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a single .wgsl.pack bundle, audit WebGPU limits, or diff against a pipeline dump.",
    )
    inspect_parser.add_argument(
        "pack",
        help="Path to the .wgsl.pack file (or '-' to read binary bytes from stdin).",
    )
    inspect_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit strictly sanitized JSON output (basenames only; no host paths).",
    )
    inspect_parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit with code 1 if any WebGPU limit violation is detected, or (with --diff-dump) "
            "if any dump stage is missing from the pack or loaded reflection sidecar."
        ),
    )
    inspect_parser.add_argument(
        "--diff-dump",
        dest="diff_dump",
        default=None,
        metavar="DIR",
        help=(
            "Compare a playgta6-pipeline-dump directory (manifest.json and/or native/*.bin) "
            "against this .wgsl.pack TOC and optional .reflect.json sidecar."
        ),
    )
    inspect_parser.add_argument(
        "--reflect",
        dest="reflect",
        default=None,
        metavar="REFLECT_JSON",
        help=(
            "Optional path to a <pack>.reflect.json sidecar for --diff-dump "
            "(auto-detects <pack>.reflect.json when omitted)."
        ),
    )
    inspect_parser.add_argument(
        "--max-samples",
        dest="max_samples",
        type=int,
        default=DEFAULT_MAX_MISSING_SAMPLES,
        metavar="N",
        help=(
            "Maximum number of missing shader hashes per sample list in --diff-dump mode "
            f"(default: {DEFAULT_MAX_MISSING_SAMPLES})."
        ),
    )

    diff_parser = subparsers.add_parser(
        "diff",
        help="Diff two .wgsl.pack bundles by shader hash, payload bytes, and audits.",
    )
    diff_parser.add_argument("pack_a", metavar="packA", help="Baseline .wgsl.pack file.")
    diff_parser.add_argument("pack_b", metavar="packB", help="Updated .wgsl.pack file.")
    diff_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit strictly sanitized JSON output (basenames only; no host paths).",
    )

    return parser


def _emit_error(
    *,
    as_json: bool,
    command: str,
    source_label: str,
    error_code: str,
    message: str,
) -> int:
    if as_json:
        payload = {
            "ok": False,
            "command": command,
            "source": source_label,
            "errorCode": error_code,
            "error": message,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"Error: {message}", file=sys.stderr)
    return 1


def _cmd_inspect_diff_dump(args: argparse.Namespace) -> int:
    source_label = sanitize_source_label(args.pack)
    dump_label = sanitize_source_label(args.diff_dump)
    reflect_label = sanitize_source_label(args.reflect) if args.reflect is not None else None

    if args.max_samples < 0:
        return _emit_error(
            as_json=args.as_json,
            command="inspect",
            source_label=source_label,
            error_code="invalid_argument",
            message=f"--max-samples must be non-negative, got {args.max_samples}",
        )

    try:
        if args.pack == "-":
            raw = sys.stdin.buffer.read()
            report = diff_pack_against_dump(
                raw,
                Path(args.diff_dump),
                reflect_path=Path(args.reflect) if args.reflect is not None else None,
                max_samples=args.max_samples,
                pack_label="<stdin>",
                dump_label=dump_label,
                reflect_label=reflect_label,
            )
        else:
            report = diff_pack_against_dump(
                Path(args.pack),
                Path(args.diff_dump),
                reflect_path=Path(args.reflect) if args.reflect is not None else None,
                max_samples=args.max_samples,
                pack_label=source_label,
                dump_label=dump_label,
                reflect_label=reflect_label,
            )
    except OSError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="inspect",
            source_label=source_label,
            error_code="io_error",
            message=_sanitize_os_error(args.pack, exc),
        )
    except WgpkParseError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="inspect",
            source_label=source_label,
            error_code=exc.code,
            message=exc.message,
        )
    except DumpDiffError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="inspect",
            source_label=exc.source or dump_label,
            error_code=exc.code,
            message=exc.message,
        )

    passed = report.clean if args.strict else True
    if args.as_json:
        payload = report.to_dict()
        payload["ok"] = passed
        payload["strict"] = bool(args.strict)
        print(json.dumps(payload, indent=2))
    else:
        print(format_dump_diff_text(report))

    return 0 if passed else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    if args.diff_dump is not None:
        return _cmd_inspect_diff_dump(args)

    source_label = sanitize_source_label(args.pack)
    if args.reflect is not None:
        return _emit_error(
            as_json=args.as_json,
            command="inspect",
            source_label=source_label,
            error_code="invalid_argument",
            message="--reflect requires --diff-dump",
        )

    try:
        if args.pack == "-":
            raw = sys.stdin.buffer.read()
            report = inspect_pack(raw, source_label="<stdin>")
        else:
            report = inspect_pack(Path(args.pack), source_label=source_label)
    except OSError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="inspect",
            source_label=source_label,
            error_code="io_error",
            message=_sanitize_os_error(args.pack, exc),
        )
    except WgpkParseError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="inspect",
            source_label=source_label,
            error_code=exc.code,
            message=exc.message,
        )

    passed = report.clean if args.strict else True
    if args.as_json:
        payload = report.to_dict()
        payload["ok"] = passed
        payload["strict"] = bool(args.strict)
        print(json.dumps(payload, indent=2))
    else:
        print(format_inspect_text(report))

    return 0 if passed else 1


def _cmd_diff(args: argparse.Namespace) -> int:
    label_a = sanitize_source_label(args.pack_a)
    label_b = sanitize_source_label(args.pack_b)

    try:
        raw_a = Path(args.pack_a).read_bytes()
    except OSError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="diff",
            source_label=label_a,
            error_code="io_error",
            message=_sanitize_os_error(args.pack_a, exc),
        )

    try:
        raw_b = Path(args.pack_b).read_bytes()
    except OSError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="diff",
            source_label=label_b,
            error_code="io_error",
            message=_sanitize_os_error(args.pack_b, exc),
        )

    try:
        bundle_a = parse_wgpk(raw_a)
    except WgpkParseError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="diff",
            source_label=label_a,
            error_code=exc.code,
            message=f"{label_a}: {exc.message}",
        )

    try:
        bundle_b = parse_wgpk(raw_b)
    except WgpkParseError as exc:
        return _emit_error(
            as_json=args.as_json,
            command="diff",
            source_label=label_b,
            error_code=exc.code,
            message=f"{label_b}: {exc.message}",
        )

    report = diff_packs(bundle_a, bundle_b, label_a=label_a, label_b=label_b)
    if args.as_json:
        payload = report.to_dict()
        payload["ok"] = True
        print(json.dumps(payload, indent=2))
    else:
        print(format_diff_text(report))

    return 0


def _normalize_cli_argv(argv: Sequence[str]) -> list[str]:
    """Allow `<pack> --diff-dump <dir>` shorthand in addition to `inspect <pack> --diff-dump <dir>`."""
    tokens = list(argv)
    if not tokens:
        return tokens
    if tokens[0] in ("inspect", "diff", "-h", "--help"):
        return tokens
    if any(tok == "--diff-dump" or tok.startswith("--diff-dump=") for tok in tokens):
        return ["inspect", *tokens]
    return tokens


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(_normalize_cli_argv(raw_argv))
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "diff":
        return _cmd_diff(args)
    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
