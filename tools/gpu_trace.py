#!/usr/bin/env python3
"""Binary GPU command stream disassembler and frame trace inspector for playgta6web.

Matches the frozen ABI v1 packet layout in `runtime/abi/playgta6_abi.h` and
`docs/gpu_commands.md` (opcodes 0..11 plus forward-compatible handling of
unknown/future opcodes >= 12).

Pure Python 3.12 stdlib only (zero external dependencies). Usable both as an
importable library and as a CLI:

    python tools/gpu_trace.py <trace.bin> [--json] [--stats] [--strict]
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import zlib
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Sequence

ABI_VERSION = 1

HEADER_STRUCT = struct.Struct("<HHI")
HEADER_SIZE = HEADER_STRUCT.size  # 8 bytes

CLEAR_COLOR_STRUCT = struct.Struct("<ffff")  # 16 bytes
CREATE_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16 bytes
WRITE_BUFFER_STRUCT = struct.Struct("<IIII")  # 16 bytes + data_len
CREATE_SHADER_MODULE_STRUCT = struct.Struct("<IIII")  # 16 bytes + code_len
DRAW_STRUCT = struct.Struct("<IIII")  # 16 bytes
CREATE_RENDER_PIPELINE_STRUCT = struct.Struct("<IIIIII")  # 24 bytes
SET_PIPELINE_STRUCT = struct.Struct("<I")  # 4 bytes
SET_VERTEX_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16 bytes
DESTROY_RESOURCE_STRUCT = struct.Struct("<I")  # 4 bytes

DEFAULT_PREVIEW_BYTES = 16

FNV1A_64_OFFSET_BASIS = 0xCBF29CE484222325
FNV1A_64_PRIME = 0x100000001B3
FNV1A_64_MASK = 0xFFFFFFFFFFFFFFFF


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


class GpuShaderStage(IntEnum):
    """Shader stage enum from runtime/abi/playgta6_abi.h."""

    Vertex = 0
    Fragment = 1
    Compute = 2


class GpuPrimitiveTopology(IntEnum):
    """Primitive topology enum from runtime/abi/playgta6_abi.h."""

    TriangleList = 0
    TriangleStrip = 1
    LineList = 2
    LineStrip = 3
    PointList = 4


TOPOLOGY_WEBGPU_NAMES: dict[int, str] = {
    GpuPrimitiveTopology.TriangleList: "triangle-list",
    GpuPrimitiveTopology.TriangleStrip: "triangle-strip",
    GpuPrimitiveTopology.LineList: "line-list",
    GpuPrimitiveTopology.LineStrip: "line-strip",
    GpuPrimitiveTopology.PointList: "point-list",
}

SHADER_STAGE_ENTRY_POINTS: dict[int, str] = {
    GpuShaderStage.Vertex: "vs_main",
    GpuShaderStage.Fragment: "fs_main",
    GpuShaderStage.Compute: "cs_main",
}

BUFFER_USAGE_FLAGS: tuple[tuple[int, str], ...] = (
    (0x0001, "MAP_READ"),
    (0x0002, "MAP_WRITE"),
    (0x0004, "COPY_SRC"),
    (0x0008, "COPY_DST"),
    (0x0010, "INDEX"),
    (0x0020, "VERTEX"),
    (0x0040, "UNIFORM"),
    (0x0080, "STORAGE"),
    (0x0100, "INDIRECT"),
    (0x0200, "QUERY_RESOLVE"),
)

BUFFER_USAGE_KNOWN_MASK = 0
for _bit, _name in BUFFER_USAGE_FLAGS:
    BUFFER_USAGE_KNOWN_MASK |= _bit

OPCODE_NAMES: dict[int, str] = {op.value: op.name for op in GpuOpcode}


def fnv1a_64(data: bytes | bytearray | memoryview) -> int:
    """Compute unsigned 64-bit FNV-1a hash over raw bytes."""
    h = FNV1A_64_OFFSET_BASIS
    for byte in data:
        h ^= byte
        h = (h * FNV1A_64_PRIME) & FNV1A_64_MASK
    return h


def crc32_hex(data: bytes | bytearray | memoryview) -> str:
    """Compute unsigned 32-bit CRC32 formatted as lowercase 0x-prefixed hex."""
    return f"0x{zlib.crc32(data) & 0xFFFFFFFF:08x}"


def decode_buffer_usage(usage_flags: int) -> tuple[list[str], int]:
    """Decode GPUBufferUsage bitmask into ordered flag names and unknown bits."""
    names = [name for bit, name in BUFFER_USAGE_FLAGS if usage_flags & bit]
    unknown_bits = usage_flags & ~BUFFER_USAGE_KNOWN_MASK
    if unknown_bits:
        names.append(f"UNKNOWN(0x{unknown_bits:x})")
    return names, unknown_bits


def decode_shader_stage(stage: int) -> tuple[str, bool]:
    """Return human-readable shader stage name and whether it is known."""
    try:
        return GpuShaderStage(stage).name, True
    except ValueError:
        return f"Unknown({stage})", False


def decode_primitive_topology(topology: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for topology value."""
    try:
        enum_val = GpuPrimitiveTopology(topology)
        return enum_val.name, TOPOLOGY_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({topology})", None, False


def _format_preview(data: bytes | memoryview, preview_bytes: int) -> dict[str, Any]:
    raw = bytes(data)
    limit = max(0, preview_bytes)
    head = raw[:limit]
    return {
        "crc32": crc32_hex(raw),
        "preview_hex": head.hex(),
        "preview_bytes": len(head),
        "truncated_preview": len(raw) > len(head),
    }


@dataclass(frozen=True)
class TraceDiagnostic:
    """Structured warning or error emitted while decoding a GPU command stream."""

    severity: str  # "warning" or "error"
    code: str
    offset: int
    packet_index: int | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "offset": self.offset,
            "packet_index": self.packet_index,
            "message": self.message,
        }


@dataclass(frozen=True)
class DecodedPacket:
    """Single decoded GPU command packet from a binary stream."""

    index: int
    offset: int
    opcode: int
    name: str
    flags: int
    payload_bytes: int
    total_bytes: int
    known: bool
    fields: dict[str, Any]
    valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "offset": self.offset,
            "opcode": self.opcode,
            "name": self.name,
            "flags": self.flags,
            "payload_bytes": self.payload_bytes,
            "total_bytes": self.total_bytes,
            "known": self.known,
            "valid": self.valid,
            "fields": self.fields,
        }


@dataclass(frozen=True)
class FrameSummary:
    """Summary of packets grouped into a frame (BeginFrame..Present)."""

    frame_index: int
    start_packet_index: int
    end_packet_index: int
    packet_count: int
    draw_calls: int
    bytes: int
    opcode_histogram: dict[str, int]
    complete: bool
    has_begin_frame: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "start_packet_index": self.start_packet_index,
            "end_packet_index": self.end_packet_index,
            "packet_count": self.packet_count,
            "draw_calls": self.draw_calls,
            "bytes": self.bytes,
            "opcode_histogram": self.opcode_histogram,
            "complete": self.complete,
            "has_begin_frame": self.has_begin_frame,
        }


@dataclass(frozen=True)
class DecodedTrace:
    """Complete decoded representation of a binary GPU command stream."""

    abi_version: int
    total_bytes: int
    consumed_bytes: int
    packets: tuple[DecodedPacket, ...]
    frames: tuple[FrameSummary, ...]
    opcode_histogram: dict[str, int]
    diagnostics: tuple[TraceDiagnostic, ...]
    strict: bool = False

    @property
    def packet_count(self) -> int:
        return len(self.packets)

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def draw_calls(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.Draw and p.valid)

    @property
    def unknown_opcode_count(self) -> int:
        return sum(1 for p in self.packets if not p.known)

    @property
    def has_errors(self) -> bool:
        return any(d.severity == "error" for d in self.diagnostics)

    @property
    def has_warnings(self) -> bool:
        return any(d.severity == "warning" for d in self.diagnostics)

    @property
    def ok(self) -> bool:
        if self.strict:
            return len(self.diagnostics) == 0
        return not self.has_errors

    def to_dict(self, *, stats_only: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "abi_version": self.abi_version,
            "strict": self.strict,
            "ok": self.ok,
            "total_bytes": self.total_bytes,
            "consumed_bytes": self.consumed_bytes,
            "packet_count": self.packet_count,
            "frame_count": self.frame_count,
            "draw_calls": self.draw_calls,
            "unknown_opcode_count": self.unknown_opcode_count,
            "opcode_histogram": dict(self.opcode_histogram),
            "frames": [frame.to_dict() for frame in self.frames],
            "diagnostics": [diag.to_dict() for diag in self.diagnostics],
        }
        if not stats_only:
            result["packets"] = [pkt.to_dict() for pkt in self.packets]
        return result


# ---------------------------------------------------------------------------
# Packet encoding helpers
# ---------------------------------------------------------------------------


def encode_packet(
    opcode: int | GpuOpcode,
    payload: bytes | bytearray | memoryview = b"",
    flags: int = 0,
) -> bytes:
    """Encode a single GpuPacketHeader + payload binary packet."""
    op_int = int(opcode)
    flags_int = int(flags)
    payload_bytes = bytes(payload)
    if not (0 <= op_int <= 0xFFFF):
        raise ValueError(f"opcode out of u16 range: {op_int}")
    if not (0 <= flags_int <= 0xFFFF):
        raise ValueError(f"flags out of u16 range: {flags_int}")
    if len(payload_bytes) > 0xFFFFFFFF:
        raise ValueError(f"payload exceeds u32 length: {len(payload_bytes)}")
    return HEADER_STRUCT.pack(op_int, flags_int, len(payload_bytes)) + payload_bytes


def encode_nop(flags: int = 0) -> bytes:
    return encode_packet(GpuOpcode.Nop, b"", flags=flags)


def encode_begin_frame(flags: int = 0) -> bytes:
    return encode_packet(GpuOpcode.BeginFrame, b"", flags=flags)


def encode_clear_color(r: float, g: float, b: float, a: float, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.ClearColor,
        CLEAR_COLOR_STRUCT.pack(r, g, b, a),
        flags=flags,
    )


def encode_create_buffer(
    resource_id: int,
    usage_flags: int,
    byte_size: int,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateBuffer,
        CREATE_BUFFER_STRUCT.pack(resource_id, usage_flags, byte_size),
        flags=flags,
    )


def encode_write_buffer(
    resource_id: int,
    dst_offset: int,
    data: bytes | bytearray | memoryview,
    *,
    reserved: int = 0,
    flags: int = 0,
) -> bytes:
    raw = bytes(data)
    head = WRITE_BUFFER_STRUCT.pack(resource_id, dst_offset, len(raw), reserved)
    return encode_packet(GpuOpcode.WriteBuffer, head + raw, flags=flags)


def encode_create_shader_module(
    resource_id: int,
    stage: int | GpuShaderStage,
    wgsl_code: str | bytes | bytearray,
    *,
    reserved: int = 0,
    flags: int = 0,
) -> bytes:
    raw = wgsl_code.encode("utf-8") if isinstance(wgsl_code, str) else bytes(wgsl_code)
    head = CREATE_SHADER_MODULE_STRUCT.pack(resource_id, int(stage), len(raw), reserved)
    return encode_packet(GpuOpcode.CreateShaderModule, head + raw, flags=flags)


def encode_draw(
    vertex_count: int,
    instance_count: int = 1,
    first_vertex: int = 0,
    first_instance: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.Draw,
        DRAW_STRUCT.pack(vertex_count, instance_count, first_vertex, first_instance),
        flags=flags,
    )


def encode_present(flags: int = 0) -> bytes:
    return encode_packet(GpuOpcode.Present, b"", flags=flags)


def encode_create_render_pipeline(
    resource_id: int,
    vs_shader_id: int,
    fs_shader_id: int,
    vertex_stride: int,
    topology: int | GpuPrimitiveTopology,
    pipeline_flags: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateRenderPipeline,
        CREATE_RENDER_PIPELINE_STRUCT.pack(
            resource_id,
            vs_shader_id,
            fs_shader_id,
            vertex_stride,
            int(topology),
            pipeline_flags,
        ),
        flags=flags,
    )


def encode_set_pipeline(pipeline_id: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetPipeline,
        SET_PIPELINE_STRUCT.pack(pipeline_id),
        flags=flags,
    )


def encode_set_vertex_buffer(
    slot: int,
    buffer_id: int,
    offset: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.SetVertexBuffer,
        SET_VERTEX_BUFFER_STRUCT.pack(slot, buffer_id, offset),
        flags=flags,
    )


def encode_destroy_resource(resource_id: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.DestroyResource,
        DESTROY_RESOURCE_STRUCT.pack(resource_id),
        flags=flags,
    )


# ---------------------------------------------------------------------------
# Decoding & frame grouping
# ---------------------------------------------------------------------------


def _warn_severity(strict: bool) -> str:
    return "error" if strict else "warning"


def _check_fixed_size(
    *,
    opcode_name: str,
    expected_size: int,
    payload_bytes: int,
    offset: int,
    packet_index: int,
    strict: bool,
    diagnostics: list[TraceDiagnostic],
) -> tuple[bool, bool]:
    """Validate fixed payload size. Returns (can_decode, is_exact_size)."""
    if payload_bytes < expected_size:
        diagnostics.append(
            TraceDiagnostic(
                severity="error",
                code="payload_too_short",
                offset=offset,
                packet_index=packet_index,
                message=(
                    f"{opcode_name} expects {expected_size} payload byte(s), "
                    f"got {payload_bytes}"
                ),
            )
        )
        return False, False
    if payload_bytes > expected_size:
        diagnostics.append(
            TraceDiagnostic(
                severity=_warn_severity(strict),
                code="payload_size_mismatch",
                offset=offset,
                packet_index=packet_index,
                message=(
                    f"{opcode_name} expects {expected_size} payload byte(s), "
                    f"got {payload_bytes} ({payload_bytes - expected_size} extra byte(s))"
                ),
            )
        )
        return True, False
    return True, True


def _decode_known_payload(
    *,
    opcode: GpuOpcode,
    payload: memoryview,
    offset: int,
    packet_index: int,
    strict: bool,
    preview_bytes: int,
    diagnostics: list[TraceDiagnostic],
) -> tuple[dict[str, Any], bool]:
    """Decode payload for a known opcode (0..11). Returns (fields, valid)."""
    payload_len = len(payload)
    name = opcode.name

    if opcode in (GpuOpcode.Nop, GpuOpcode.BeginFrame, GpuOpcode.Present):
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=0,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        fields: dict[str, Any] = {}
        if not exact and payload_len > 0:
            fields["trailing_bytes"] = payload_len
        return fields, can_decode

    if opcode == GpuOpcode.ClearColor:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CLEAR_COLOR_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {"expected_bytes": CLEAR_COLOR_STRUCT.size, "actual_bytes": payload_len}, False
        r, g, b, a = CLEAR_COLOR_STRUCT.unpack_from(payload, 0)
        for comp_name, val in (("r", r), ("g", g), ("b", b), ("a", a)):
            if not math.isfinite(val):
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="non_finite_float",
                        offset=offset,
                        packet_index=packet_index,
                        message=f"ClearColor.{comp_name} is non-finite ({val})",
                    )
                )
        fields = {"r": r, "g": g, "b": b, "a": a}
        if not exact:
            fields["trailing_bytes"] = payload_len - CLEAR_COLOR_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateBuffer:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_BUFFER_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {"expected_bytes": CREATE_BUFFER_STRUCT.size, "actual_bytes": payload_len}, False
        res_id, usage_flags, byte_size = CREATE_BUFFER_STRUCT.unpack_from(payload, 0)
        usage_names, unknown_bits = decode_buffer_usage(usage_flags)
        if unknown_bits:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_buffer_usage_bits",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateBuffer(id={res_id}) has unrecognized usage_flags "
                        f"bits: 0x{unknown_bits:x}"
                    ),
                )
            )
        fields = {
            "id": res_id,
            "usage_flags": usage_flags,
            "usage_names": usage_names,
            "byte_size": byte_size,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_BUFFER_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.WriteBuffer:
        if payload_len < WRITE_BUFFER_STRUCT.size:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="payload_too_short",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"WriteBuffer header expects at least {WRITE_BUFFER_STRUCT.size} byte(s), "
                        f"got {payload_len}"
                    ),
                )
            )
            return {
                "expected_min_bytes": WRITE_BUFFER_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        res_id, dst_offset, data_len, reserved = WRITE_BUFFER_STRUCT.unpack_from(payload, 0)
        expected_total = WRITE_BUFFER_STRUCT.size + data_len
        available = payload_len - WRITE_BUFFER_STRUCT.size
        valid = True
        if expected_total > payload_len:
            valid = False
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="variable_payload_truncated",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"WriteBuffer(id={res_id}) declares data_len={data_len}, "
                        f"but only {available} byte(s) are present in packet payload"
                    ),
                )
            )
            data_slice = payload[WRITE_BUFFER_STRUCT.size :]
        else:
            if expected_total < payload_len:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="payload_size_mismatch",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"WriteBuffer(id={res_id}) expects {expected_total} payload byte(s) "
                            f"(16 + {data_len}), got {payload_len}"
                        ),
                    )
                )
            data_slice = payload[WRITE_BUFFER_STRUCT.size : expected_total]

        preview = _format_preview(data_slice, preview_bytes)
        fields = {
            "id": res_id,
            "dst_offset": dst_offset,
            "data_len": data_len,
            "reserved": reserved,
            **preview,
        }
        if not valid:
            fields["available_bytes"] = available
        elif expected_total < payload_len:
            fields["trailing_bytes"] = payload_len - expected_total
        return fields, valid

    if opcode == GpuOpcode.CreateShaderModule:
        if payload_len < CREATE_SHADER_MODULE_STRUCT.size:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="payload_too_short",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        "CreateShaderModule header expects at least "
                        f"{CREATE_SHADER_MODULE_STRUCT.size} byte(s), got {payload_len}"
                    ),
                )
            )
            return {
                "expected_min_bytes": CREATE_SHADER_MODULE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        res_id, stage, code_len, reserved = CREATE_SHADER_MODULE_STRUCT.unpack_from(payload, 0)
        stage_name, stage_known = decode_shader_stage(stage)
        if not stage_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_shader_stage",
                    offset=offset,
                    packet_index=packet_index,
                    message=f"CreateShaderModule(id={res_id}) has unknown stage={stage}",
                )
            )
        expected_total = CREATE_SHADER_MODULE_STRUCT.size + code_len
        available = payload_len - CREATE_SHADER_MODULE_STRUCT.size
        valid = True
        if expected_total > payload_len:
            valid = False
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="variable_payload_truncated",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateShaderModule(id={res_id}) declares code_len={code_len}, "
                        f"but only {available} byte(s) are present in packet payload"
                    ),
                )
            )
            code_raw = bytes(payload[CREATE_SHADER_MODULE_STRUCT.size :])
        else:
            if expected_total < payload_len:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="payload_size_mismatch",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"CreateShaderModule(id={res_id}) expects {expected_total} payload "
                            f"byte(s) (16 + {code_len}), got {payload_len}"
                        ),
                    )
                )
            code_raw = bytes(payload[CREATE_SHADER_MODULE_STRUCT.size : expected_total])

        utf8_valid = True
        try:
            code_text = code_raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            utf8_valid = False
            code_text = code_raw.decode("utf-8", errors="replace")
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="invalid_utf8_wgsl",
                    offset=offset,
                    packet_index=packet_index,
                    message=f"CreateShaderModule(id={res_id}) contains invalid UTF-8: {exc}",
                )
            )

        line_count = len(code_text.splitlines()) if code_text else 0
        char_count = len(code_text)
        hash_hex = f"0x{fnv1a_64(code_raw):016x}"
        fields = {
            "id": res_id,
            "stage": stage,
            "stage_name": stage_name,
            "code_len": code_len,
            "reserved": reserved,
            "line_count": line_count,
            "char_count": char_count,
            "wgsl_hash_fnv1a64": hash_hex,
            "utf8_valid": utf8_valid,
        }
        if not valid:
            fields["available_bytes"] = available
        elif expected_total < payload_len:
            fields["trailing_bytes"] = payload_len - expected_total
        return fields, valid

    if opcode == GpuOpcode.Draw:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=DRAW_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {"expected_bytes": DRAW_STRUCT.size, "actual_bytes": payload_len}, False
        vertex_count, instance_count, first_vertex, first_instance = DRAW_STRUCT.unpack_from(
            payload, 0
        )
        fields = {
            "vertex_count": vertex_count,
            "instance_count": instance_count,
            "first_vertex": first_vertex,
            "first_instance": first_instance,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - DRAW_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateRenderPipeline:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_RENDER_PIPELINE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": CREATE_RENDER_PIPELINE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            res_id,
            vs_shader_id,
            fs_shader_id,
            vertex_stride,
            topology,
            pipeline_flags,
        ) = CREATE_RENDER_PIPELINE_STRUCT.unpack_from(payload, 0)
        topology_name, topology_webgpu, topo_known = decode_primitive_topology(topology)
        if not topo_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_topology",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateRenderPipeline(id={res_id}) has unknown topology={topology}"
                    ),
                )
            )
        fields = {
            "id": res_id,
            "vs_shader_id": vs_shader_id,
            "fs_shader_id": fs_shader_id,
            "vertex_stride": vertex_stride,
            "topology": topology,
            "topology_name": topology_name,
            "topology_webgpu": topology_webgpu,
            "flags": pipeline_flags,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_RENDER_PIPELINE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.SetPipeline:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_PIPELINE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {"expected_bytes": SET_PIPELINE_STRUCT.size, "actual_bytes": payload_len}, False
        (pipeline_id,) = SET_PIPELINE_STRUCT.unpack_from(payload, 0)
        fields = {"pipeline_id": pipeline_id}
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_PIPELINE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.SetVertexBuffer:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_VERTEX_BUFFER_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": SET_VERTEX_BUFFER_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        slot, buffer_id, vb_offset = SET_VERTEX_BUFFER_STRUCT.unpack_from(payload, 0)
        fields = {
            "slot": slot,
            "buffer_id": buffer_id,
            "offset": vb_offset,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_VERTEX_BUFFER_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.DestroyResource:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=DESTROY_RESOURCE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": DESTROY_RESOURCE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (res_id,) = DESTROY_RESOURCE_STRUCT.unpack_from(payload, 0)
        fields = {"id": res_id}
        if not exact:
            fields["trailing_bytes"] = payload_len - DESTROY_RESOURCE_STRUCT.size
        return fields, True

    raise AssertionError(f"Unhandled known opcode: {opcode}")


def split_frames(packets: Sequence[DecodedPacket]) -> list[FrameSummary]:
    """Group packets by BeginFrame..Present into structured FrameSummary objects.

    Every packet in the sequence is accounted for:
    - Standard frames span from `BeginFrame` through `Present` (`complete=True`).
    - Setup packets appearing before the first `BeginFrame` (or between `Present`
      and the next `BeginFrame`) are preserved as a frame group with
      `has_begin_frame=False`.
    - An unclosed frame at EOF (or interrupted by a new `BeginFrame`) is emitted
      with `complete=False`.
    """
    frames: list[FrameSummary] = []
    current: list[DecodedPacket] = []

    def _flush(*, complete: bool) -> None:
        if not current:
            return
        histogram: Counter[str] = Counter()
        draw_calls = 0
        total_bytes = 0
        for pkt in current:
            histogram[pkt.name] += 1
            total_bytes += pkt.total_bytes
            if pkt.opcode == GpuOpcode.Draw and pkt.valid:
                draw_calls += 1
        frames.append(
            FrameSummary(
                frame_index=len(frames),
                start_packet_index=current[0].index,
                end_packet_index=current[-1].index,
                packet_count=len(current),
                draw_calls=draw_calls,
                bytes=total_bytes,
                opcode_histogram=dict(histogram),
                complete=complete,
                has_begin_frame=(current[0].opcode == GpuOpcode.BeginFrame),
            )
        )
        current.clear()

    for pkt in packets:
        if pkt.opcode == GpuOpcode.BeginFrame and current:
            _flush(complete=False)
        current.append(pkt)
        if pkt.opcode == GpuOpcode.Present:
            _flush(complete=True)

    if current:
        _flush(complete=False)

    return frames


def decode_stream(
    data: bytes | bytearray | memoryview,
    *,
    strict: bool = False,
    preview_bytes: int = DEFAULT_PREVIEW_BYTES,
) -> DecodedTrace:
    """Decode a raw binary GPU command stream into a DecodedTrace."""
    view = memoryview(data)
    total_bytes = len(view)
    offset = 0
    packet_index = 0
    packets: list[DecodedPacket] = []
    diagnostics: list[TraceDiagnostic] = []
    histogram: Counter[str] = Counter()

    while offset < total_bytes:
        remaining = total_bytes - offset
        if remaining < HEADER_SIZE:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="truncated_header",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"Truncated packet header at byte {offset}: "
                        f"need {HEADER_SIZE} bytes, only {remaining} byte(s) remaining"
                    ),
                )
            )
            break

        raw_opcode, flags, payload_bytes = HEADER_STRUCT.unpack_from(view, offset)
        payload_offset = offset + HEADER_SIZE
        packet_end = payload_offset + payload_bytes

        try:
            known_opcode: GpuOpcode | None = GpuOpcode(raw_opcode)
            name = known_opcode.name
        except ValueError:
            known_opcode = None
            name = f"Unknown({raw_opcode})"

        if packet_end > total_bytes:
            available = total_bytes - payload_offset
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="truncated_payload",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"Truncated payload for {name} (opcode {raw_opcode}) at byte {offset}: "
                        f"header declares {payload_bytes} byte(s), only {available} byte(s) remaining"
                    ),
                )
            )
            break

        payload = view[payload_offset:packet_end]
        if known_opcode is not None:
            fields, valid = _decode_known_payload(
                opcode=known_opcode,
                payload=payload,
                offset=offset,
                packet_index=packet_index,
                strict=strict,
                preview_bytes=preview_bytes,
                diagnostics=diagnostics,
            )
            pkt = DecodedPacket(
                index=packet_index,
                offset=offset,
                opcode=raw_opcode,
                name=name,
                flags=flags,
                payload_bytes=payload_bytes,
                total_bytes=HEADER_SIZE + payload_bytes,
                known=True,
                fields=fields,
                valid=valid,
            )
        else:
            if strict:
                diagnostics.append(
                    TraceDiagnostic(
                        severity="error",
                        code="unknown_opcode",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"Unknown opcode {raw_opcode} at byte {offset} "
                            f"({payload_bytes} payload byte(s))"
                        ),
                    )
                )
            preview = _format_preview(payload, preview_bytes)
            pkt = DecodedPacket(
                index=packet_index,
                offset=offset,
                opcode=raw_opcode,
                name=name,
                flags=flags,
                payload_bytes=payload_bytes,
                total_bytes=HEADER_SIZE + payload_bytes,
                known=False,
                fields={
                    "payload_bytes": payload_bytes,
                    **preview,
                },
                valid=not strict,
            )

        packets.append(pkt)
        histogram[pkt.name] += 1
        offset = packet_end
        packet_index += 1

    frames = split_frames(packets)
    return DecodedTrace(
        abi_version=ABI_VERSION,
        total_bytes=total_bytes,
        consumed_bytes=offset,
        packets=tuple(packets),
        frames=tuple(frames),
        opcode_histogram=dict(histogram),
        diagnostics=tuple(diagnostics),
        strict=strict,
    )


# ---------------------------------------------------------------------------
# Formatting & JSON serialization
# ---------------------------------------------------------------------------


def _format_packet_fields(pkt: DecodedPacket) -> str:
    f = pkt.fields
    if not pkt.known:
        preview = f.get("preview_hex", "")
        suffix = "..." if f.get("truncated_preview") else ""
        return (
            f"payload_bytes={pkt.payload_bytes} crc32={f.get('crc32', '')}"
            + (f" preview=0x{preview}{suffix}" if preview else "")
        )

    if not pkt.valid:
        return " ".join(f"{k}={v}" for k, v in f.items())

    op = pkt.opcode
    if op in (GpuOpcode.Nop, GpuOpcode.BeginFrame, GpuOpcode.Present):
        return f"trailing_bytes={f['trailing_bytes']}" if "trailing_bytes" in f else ""
    if op == GpuOpcode.ClearColor:
        return f"rgba=({f['r']:.4f}, {f['g']:.4f}, {f['b']:.4f}, {f['a']:.4f})"
    if op == GpuOpcode.CreateBuffer:
        usage = "|".join(f["usage_names"]) if f["usage_names"] else "NONE"
        return f"id={f['id']} size={f['byte_size']} usage=0x{f['usage_flags']:04x}({usage})"
    if op == GpuOpcode.WriteBuffer:
        preview = f["preview_hex"]
        suffix = "..." if f["truncated_preview"] else ""
        preview_str = f" preview=0x{preview}{suffix}" if preview else ""
        return (
            f"id={f['id']} dst_offset={f['dst_offset']} data_len={f['data_len']} "
            f"crc32={f['crc32']}{preview_str}"
        )
    if op == GpuOpcode.CreateShaderModule:
        return (
            f"id={f['id']} stage={f['stage_name']} code_len={f['code_len']} "
            f"lines={f['line_count']} chars={f['char_count']} fnv1a64={f['wgsl_hash_fnv1a64']}"
        )
    if op == GpuOpcode.Draw:
        return (
            f"vertices={f['vertex_count']} instances={f['instance_count']} "
            f"first_vertex={f['first_vertex']} first_instance={f['first_instance']}"
        )
    if op == GpuOpcode.CreateRenderPipeline:
        topo = f["topology_webgpu"] or f["topology_name"]
        return (
            f"id={f['id']} vs={f['vs_shader_id']} fs={f['fs_shader_id']} "
            f"stride={f['vertex_stride']} topology={topo} flags=0x{f['flags']:x}"
        )
    if op == GpuOpcode.SetPipeline:
        return f"pipeline_id={f['pipeline_id']}"
    if op == GpuOpcode.SetVertexBuffer:
        return f"slot={f['slot']} buffer_id={f['buffer_id']} offset={f['offset']}"
    if op == GpuOpcode.DestroyResource:
        return f"id={f['id']}"
    return " ".join(f"{k}={v}" for k, v in f.items())


def format_trace_text(trace: DecodedTrace, *, stats_only: bool = False) -> str:
    """Format a DecodedTrace as human-readable disassembly and/or statistics."""
    lines: list[str] = [
        f"=== playgta6 GPU Trace (ABI v{trace.abi_version}) ===",
        (
            f"Bytes: {trace.consumed_bytes}/{trace.total_bytes} | "
            f"Packets: {trace.packet_count} | Frames: {trace.frame_count} | "
            f"Draws: {trace.draw_calls} | Unknown: {trace.unknown_opcode_count} | "
            f"Diagnostics: {len(trace.diagnostics)}"
        ),
    ]

    if not stats_only:
        lines.append("")
        lines.append("Packets:")
        if not trace.packets:
            lines.append("  (none)")
        for pkt in trace.packets:
            flag_part = f" flags=0x{pkt.flags:04x}" if pkt.flags else ""
            status_part = "" if pkt.valid else " [INVALID]"
            detail = _format_packet_fields(pkt)
            detail_part = f"  {detail}" if detail else ""
            lines.append(
                f"  #{pkt.index:04d} @0x{pkt.offset:06x}  "
                f"{pkt.name:<22s} ({pkt.payload_bytes:5d} B){flag_part}{status_part}{detail_part}"
            )

    lines.append("")
    lines.append("Frames:")
    if not trace.frames:
        lines.append("  (none)")
    for frame in trace.frames:
        status = "complete" if frame.complete else "incomplete"
        if not frame.has_begin_frame:
            status += ",no-begin"
        hist_str = ", ".join(f"{k}={v}" for k, v in sorted(frame.opcode_histogram.items()))
        lines.append(
            f"  Frame {frame.frame_index:03d} [{status}]: "
            f"packets={frame.packet_count} (#{frame.start_packet_index}..#{frame.end_packet_index}) "
            f"draws={frame.draw_calls} bytes={frame.bytes} | {hist_str}"
        )

    lines.append("")
    lines.append("Opcode Histogram:")
    if not trace.opcode_histogram:
        lines.append("  (none)")
    for op_name, count in sorted(trace.opcode_histogram.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"  {op_name:<22s}: {count}")

    if trace.diagnostics:
        lines.append("")
        lines.append("Diagnostics:")
        for diag in trace.diagnostics:
            pkt_str = f" packet=#{diag.packet_index}" if diag.packet_index is not None else ""
            lines.append(
                f"  [{diag.severity.upper()}] {diag.code} @0x{diag.offset:06x}{pkt_str}: "
                f"{diag.message}"
            )

    return "\n".join(lines)


def trace_to_json(
    trace: DecodedTrace,
    *,
    stats_only: bool = False,
    indent: int = 2,
) -> str:
    """Serialize a DecodedTrace to formatted JSON."""
    return json.dumps(trace.to_dict(stats_only=stats_only), indent=indent)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Disassemble and inspect a playgta6 binary GPU command stream."
    )
    parser.add_argument(
        "trace_file",
        help="Path to raw binary GPU command stream (or '-' to read from stdin).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit structured JSON instead of text output.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Emit summary statistics, frame breakdown, and opcode histogram only.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail with non-zero exit code on unknown opcodes (>= 12), "
            "size mismatches, or malformed/truncated packets."
        ),
    )
    parser.add_argument(
        "--preview-bytes",
        type=int,
        default=DEFAULT_PREVIEW_BYTES,
        metavar="N",
        help=f"Maximum raw payload preview bytes for WriteBuffer/Unknown (default: {DEFAULT_PREVIEW_BYTES}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.trace_file == "-":
        raw_bytes = sys.stdin.buffer.read()
    else:
        path = Path(args.trace_file)
        if not path.is_file():
            print(f"Error: trace file not found: {path}", file=sys.stderr)
            return 2
        raw_bytes = path.read_bytes()

    trace = decode_stream(
        raw_bytes,
        strict=args.strict,
        preview_bytes=args.preview_bytes,
    )

    if args.as_json:
        print(trace_to_json(trace, stats_only=args.stats))
    else:
        print(format_trace_text(trace, stats_only=args.stats))

    if args.strict and not trace.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
