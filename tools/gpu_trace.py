#!/usr/bin/env python3
"""Binary GPU command stream disassembler and frame trace inspector for playgta6web.

Matches the frozen ABI v1 packet layout in `runtime/abi/playgta6_abi.h` and
`docs/gpu_commands.md` (opcodes 0..21 plus forward-compatible handling of
unknown/future opcodes >= 22).

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
CREATE_TEXTURE_STRUCT = struct.Struct("<IIIIIIII")  # 32 bytes
WRITE_TEXTURE_STRUCT = struct.Struct("<IIIIIIIIIIII")  # 48 bytes + data_len
CREATE_SAMPLER_STRUCT = struct.Struct("<IIIIIIII")  # 32 bytes
BIND_GROUP_ENTRY_STRUCT = struct.Struct("<IIIIQQ")  # 32 bytes
CREATE_BIND_GROUP_STRUCT = struct.Struct("<IIII")  # 16 bytes + entry_count * 32
SET_BIND_GROUP_STRUCT = struct.Struct("<II")  # 8 bytes
SET_INDEX_BUFFER_STRUCT = struct.Struct("<IIQ")  # 16 bytes
DRAW_INDEXED_STRUCT = struct.Struct("<IIIiI")  # 20 bytes
CREATE_COMPUTE_PIPELINE_STRUCT = struct.Struct("<IIII")  # 16 bytes
SET_COMPUTE_PIPELINE_STRUCT = struct.Struct("<I")  # 4 bytes
DISPATCH_COMPUTE_STRUCT = struct.Struct("<III")  # 12 bytes

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


class GpuTextureFormat(IntEnum):
    """Texture format enum from runtime/abi/playgta6_abi.h."""

    Rgba8Unorm = 0
    Bgra8Unorm = 1
    Rgba16Float = 2
    Depth24Plus = 3
    Depth32Float = 4
    Bc7RgbaUnorm = 5


class GpuTextureDimension(IntEnum):
    """Texture dimension enum from runtime/abi/playgta6_abi.h."""

    Dim1D = 0
    Dim2D = 1
    Dim3D = 2


class GpuAddressMode(IntEnum):
    """Sampler address mode enum from runtime/abi/playgta6_abi.h."""

    ClampToEdge = 0
    Repeat = 1
    MirrorRepeat = 2


class GpuFilterMode(IntEnum):
    """Sampler filter mode enum from runtime/abi/playgta6_abi.h."""

    Nearest = 0
    Linear = 1


class GpuBindResourceType(IntEnum):
    """Bind group entry resource type enum from runtime/abi/playgta6_abi.h."""

    Buffer = 0
    TextureView = 1
    Sampler = 2


class GpuIndexFormat(IntEnum):
    """Index buffer format enum from runtime/abi/playgta6_abi.h."""

    Uint16 = 0
    Uint32 = 1


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


TEXTURE_FORMAT_WEBGPU_NAMES: dict[int, str] = {
    GpuTextureFormat.Rgba8Unorm: "rgba8unorm",
    GpuTextureFormat.Bgra8Unorm: "bgra8unorm",
    GpuTextureFormat.Rgba16Float: "rgba16float",
    GpuTextureFormat.Depth24Plus: "depth24plus",
    GpuTextureFormat.Depth32Float: "depth32float",
    GpuTextureFormat.Bc7RgbaUnorm: "bc7-rgba-unorm",
}

TEXTURE_DIMENSION_WEBGPU_NAMES: dict[int, str] = {
    GpuTextureDimension.Dim1D: "1d",
    GpuTextureDimension.Dim2D: "2d",
    GpuTextureDimension.Dim3D: "3d",
}

ADDRESS_MODE_WEBGPU_NAMES: dict[int, str] = {
    GpuAddressMode.ClampToEdge: "clamp-to-edge",
    GpuAddressMode.Repeat: "repeat",
    GpuAddressMode.MirrorRepeat: "mirror-repeat",
}

FILTER_MODE_WEBGPU_NAMES: dict[int, str] = {
    GpuFilterMode.Nearest: "nearest",
    GpuFilterMode.Linear: "linear",
}

INDEX_FORMAT_WEBGPU_NAMES: dict[int, str] = {
    GpuIndexFormat.Uint16: "uint16",
    GpuIndexFormat.Uint32: "uint32",
}

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

TEXTURE_USAGE_FLAGS: tuple[tuple[int, str], ...] = (
    (0x0001, "COPY_SRC"),
    (0x0002, "COPY_DST"),
    (0x0004, "TEXTURE_BINDING"),
    (0x0008, "STORAGE_BINDING"),
    (0x0010, "RENDER_ATTACHMENT"),
)

TEXTURE_USAGE_KNOWN_MASK = 0
for _bit, _name in TEXTURE_USAGE_FLAGS:
    TEXTURE_USAGE_KNOWN_MASK |= _bit

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


def decode_texture_usage(usage_flags: int) -> tuple[list[str], int]:
    """Decode GPUTextureUsage bitmask into ordered flag names and unknown bits."""
    names = [name for bit, name in TEXTURE_USAGE_FLAGS if usage_flags & bit]
    unknown_bits = usage_flags & ~TEXTURE_USAGE_KNOWN_MASK
    if unknown_bits:
        names.append(f"UNKNOWN(0x{unknown_bits:x})")
    return names, unknown_bits


def decode_texture_format(texture_format: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuTextureFormat."""
    try:
        enum_val = GpuTextureFormat(texture_format)
        return enum_val.name, TEXTURE_FORMAT_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({texture_format})", None, False


def decode_texture_dimension(dimension: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuTextureDimension."""
    try:
        enum_val = GpuTextureDimension(dimension)
        return enum_val.name, TEXTURE_DIMENSION_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({dimension})", None, False


def decode_address_mode(mode: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuAddressMode."""
    try:
        enum_val = GpuAddressMode(mode)
        return enum_val.name, ADDRESS_MODE_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({mode})", None, False


def decode_filter_mode(mode: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuFilterMode."""
    try:
        enum_val = GpuFilterMode(mode)
        return enum_val.name, FILTER_MODE_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({mode})", None, False


def decode_bind_resource_type(resource_type: int) -> tuple[str, bool]:
    """Return (enum_name, is_known) for GpuBindResourceType."""
    try:
        return GpuBindResourceType(resource_type).name, True
    except ValueError:
        return f"Unknown({resource_type})", False


def decode_index_format(index_format: int) -> tuple[str, str | None, bool]:
    """Return (enum_name, webgpu_name_or_none, is_known) for GpuIndexFormat."""
    try:
        enum_val = GpuIndexFormat(index_format)
        return enum_val.name, INDEX_FORMAT_WEBGPU_NAMES[enum_val], True
    except ValueError:
        return f"Unknown({index_format})", None, False


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
class BindGroupEntry:
    """Single resource binding entry for CreateBindGroup (32 bytes)."""

    binding: int
    resource_type: int | GpuBindResourceType
    resource_id: int
    offset: int = 0
    size: int = 0
    reserved: int = 0


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
    draw_indexed_calls: int = 0
    compute_dispatches: int = 0
    compute_passes: int = 0
    textures_created: int = 0
    bind_groups_created: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "start_packet_index": self.start_packet_index,
            "end_packet_index": self.end_packet_index,
            "packet_count": self.packet_count,
            "draw_calls": self.draw_calls,
            "draw_indexed_calls": self.draw_indexed_calls,
            "compute_dispatches": self.compute_dispatches,
            "compute_passes": self.compute_passes,
            "textures_created": self.textures_created,
            "bind_groups_created": self.bind_groups_created,
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
        return sum(
            1
            for p in self.packets
            if p.opcode in (GpuOpcode.Draw, GpuOpcode.DrawIndexed) and p.valid
        )

    @property
    def draw_indexed_calls(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.DrawIndexed and p.valid)

    @property
    def compute_dispatches(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.DispatchCompute and p.valid)

    @property
    def compute_passes(self) -> int:
        return sum(f.compute_passes for f in self.frames)

    @property
    def textures_created(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.CreateTexture and p.valid)

    @property
    def bind_groups_created(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.CreateBindGroup and p.valid)

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
            "draw_indexed_calls": self.draw_indexed_calls,
            "compute_dispatches": self.compute_dispatches,
            "compute_passes": self.compute_passes,
            "textures_created": self.textures_created,
            "bind_groups_created": self.bind_groups_created,
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


def encode_create_texture(
    resource_id: int,
    width: int,
    height: int,
    depth_or_layers: int = 1,
    mip_levels: int = 1,
    texture_format: int | GpuTextureFormat = GpuTextureFormat.Rgba8Unorm,
    usage_flags: int = 0,
    dimension: int | GpuTextureDimension = GpuTextureDimension.Dim2D,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateTexture,
        CREATE_TEXTURE_STRUCT.pack(
            resource_id,
            width,
            height,
            depth_or_layers,
            mip_levels,
            int(texture_format),
            usage_flags,
            int(dimension),
        ),
        flags=flags,
    )


def encode_write_texture(
    texture_id: int,
    width: int,
    height: int,
    data: bytes | bytearray | memoryview,
    *,
    mip_level: int = 0,
    origin_x: int = 0,
    origin_y: int = 0,
    origin_z: int = 0,
    depth_or_layers: int = 1,
    bytes_per_row: int = 0,
    rows_per_image: int = 0,
    reserved: int = 0,
    flags: int = 0,
) -> bytes:
    raw = bytes(data)
    head = WRITE_TEXTURE_STRUCT.pack(
        texture_id,
        mip_level,
        origin_x,
        origin_y,
        origin_z,
        width,
        height,
        depth_or_layers,
        bytes_per_row,
        rows_per_image,
        len(raw),
        reserved,
    )
    return encode_packet(GpuOpcode.WriteTexture, head + raw, flags=flags)


def encode_create_sampler(
    resource_id: int,
    address_u: int | GpuAddressMode = GpuAddressMode.ClampToEdge,
    address_v: int | GpuAddressMode = GpuAddressMode.ClampToEdge,
    address_w: int | GpuAddressMode = GpuAddressMode.ClampToEdge,
    mag_filter: int | GpuFilterMode = GpuFilterMode.Linear,
    min_filter: int | GpuFilterMode = GpuFilterMode.Linear,
    mipmap_filter: int | GpuFilterMode = GpuFilterMode.Nearest,
    max_anisotropy: int = 1,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateSampler,
        CREATE_SAMPLER_STRUCT.pack(
            resource_id,
            int(address_u),
            int(address_v),
            int(address_w),
            int(mag_filter),
            int(min_filter),
            int(mipmap_filter),
            max_anisotropy,
        ),
        flags=flags,
    )


def encode_bind_group_entry(
    binding: int,
    resource_type: int | GpuBindResourceType,
    resource_id: int,
    offset: int = 0,
    size: int = 0,
    reserved: int = 0,
) -> bytes:
    return BIND_GROUP_ENTRY_STRUCT.pack(
        binding,
        int(resource_type),
        resource_id,
        reserved,
        offset,
        size,
    )


def encode_create_bind_group(
    resource_id: int,
    pipeline_id: int,
    group_index: int,
    entries: Sequence[BindGroupEntry | dict[str, Any] | bytes | bytearray | memoryview],
    *,
    flags: int = 0,
) -> bytes:
    packed_entries: list[bytes] = []
    for entry in entries:
        if isinstance(entry, BindGroupEntry):
            packed_entries.append(
                encode_bind_group_entry(
                    binding=entry.binding,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    offset=entry.offset,
                    size=entry.size,
                    reserved=entry.reserved,
                )
            )
        elif isinstance(entry, dict):
            packed_entries.append(
                encode_bind_group_entry(
                    binding=entry["binding"],
                    resource_type=entry["resource_type"],
                    resource_id=entry["resource_id"],
                    offset=entry.get("offset", 0),
                    size=entry.get("size", 0),
                    reserved=entry.get("reserved", 0),
                )
            )
        else:
            raw_entry = bytes(entry)
            if len(raw_entry) != BIND_GROUP_ENTRY_STRUCT.size:
                raise ValueError(
                    f"Raw bind group entry must be {BIND_GROUP_ENTRY_STRUCT.size} bytes, "
                    f"got {len(raw_entry)}"
                )
            packed_entries.append(raw_entry)
    head = CREATE_BIND_GROUP_STRUCT.pack(resource_id, pipeline_id, group_index, len(packed_entries))
    return encode_packet(GpuOpcode.CreateBindGroup, head + b"".join(packed_entries), flags=flags)


def encode_set_bind_group(
    group_index: int,
    bind_group_id: int,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.SetBindGroup,
        SET_BIND_GROUP_STRUCT.pack(group_index, bind_group_id),
        flags=flags,
    )


def encode_set_index_buffer(
    buffer_id: int,
    index_format: int | GpuIndexFormat = GpuIndexFormat.Uint16,
    offset: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.SetIndexBuffer,
        SET_INDEX_BUFFER_STRUCT.pack(buffer_id, int(index_format), offset),
        flags=flags,
    )


def encode_draw_indexed(
    index_count: int,
    instance_count: int = 1,
    first_index: int = 0,
    base_vertex: int = 0,
    first_instance: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.DrawIndexed,
        DRAW_INDEXED_STRUCT.pack(
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        ),
        flags=flags,
    )


def encode_create_compute_pipeline(
    resource_id: int,
    cs_shader_id: int,
    pipeline_flags: int = 0,
    *,
    reserved: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateComputePipeline,
        CREATE_COMPUTE_PIPELINE_STRUCT.pack(resource_id, cs_shader_id, pipeline_flags, reserved),
        flags=flags,
    )


def encode_set_compute_pipeline(pipeline_id: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetComputePipeline,
        SET_COMPUTE_PIPELINE_STRUCT.pack(pipeline_id),
        flags=flags,
    )


def encode_dispatch_compute(
    workgroup_x: int,
    workgroup_y: int = 1,
    workgroup_z: int = 1,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.DispatchCompute,
        DISPATCH_COMPUTE_STRUCT.pack(workgroup_x, workgroup_y, workgroup_z),
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
    """Decode payload for a known opcode (0..21). Returns (fields, valid)."""
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

    if opcode == GpuOpcode.CreateTexture:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_TEXTURE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": CREATE_TEXTURE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            res_id,
            width,
            height,
            depth_or_layers,
            mip_levels,
            tex_format,
            usage_flags,
            dimension,
        ) = CREATE_TEXTURE_STRUCT.unpack_from(payload, 0)
        format_name, format_webgpu, fmt_known = decode_texture_format(tex_format)
        if not fmt_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_texture_format",
                    offset=offset,
                    packet_index=packet_index,
                    message=f"CreateTexture(id={res_id}) has unknown format={tex_format}",
                )
            )
        usage_names, unknown_bits = decode_texture_usage(usage_flags)
        if unknown_bits:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_texture_usage_bits",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateTexture(id={res_id}) has unrecognized usage_flags "
                        f"bits: 0x{unknown_bits:x}"
                    ),
                )
            )
        dim_name, dim_webgpu, dim_known = decode_texture_dimension(dimension)
        if not dim_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_texture_dimension",
                    offset=offset,
                    packet_index=packet_index,
                    message=f"CreateTexture(id={res_id}) has unknown dimension={dimension}",
                )
            )
        fields = {
            "id": res_id,
            "width": width,
            "height": height,
            "depth_or_layers": depth_or_layers,
            "mip_levels": mip_levels,
            "format": tex_format,
            "format_name": format_name,
            "format_webgpu": format_webgpu,
            "usage_flags": usage_flags,
            "usage_names": usage_names,
            "dimension": dimension,
            "dimension_name": dim_name,
            "dimension_webgpu": dim_webgpu,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_TEXTURE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.WriteTexture:
        if payload_len < WRITE_TEXTURE_STRUCT.size:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="payload_too_short",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"WriteTexture header expects at least {WRITE_TEXTURE_STRUCT.size} byte(s), "
                        f"got {payload_len}"
                    ),
                )
            )
            return {
                "expected_min_bytes": WRITE_TEXTURE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            texture_id,
            mip_level,
            origin_x,
            origin_y,
            origin_z,
            width,
            height,
            depth_or_layers,
            bytes_per_row,
            rows_per_image,
            data_len,
            reserved,
        ) = WRITE_TEXTURE_STRUCT.unpack_from(payload, 0)
        expected_total = WRITE_TEXTURE_STRUCT.size + data_len
        available = payload_len - WRITE_TEXTURE_STRUCT.size
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
                        f"WriteTexture(texture_id={texture_id}) declares data_len={data_len}, "
                        f"but only {available} byte(s) are present in packet payload"
                    ),
                )
            )
            data_slice = payload[WRITE_TEXTURE_STRUCT.size :]
        else:
            if expected_total < payload_len:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="payload_size_mismatch",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"WriteTexture(texture_id={texture_id}) expects {expected_total} "
                            f"payload byte(s) ({WRITE_TEXTURE_STRUCT.size} + {data_len}), "
                            f"got {payload_len}"
                        ),
                    )
                )
            data_slice = payload[WRITE_TEXTURE_STRUCT.size : expected_total]

        preview = _format_preview(data_slice, preview_bytes)
        fields = {
            "texture_id": texture_id,
            "mip_level": mip_level,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "origin_z": origin_z,
            "width": width,
            "height": height,
            "depth_or_layers": depth_or_layers,
            "bytes_per_row": bytes_per_row,
            "rows_per_image": rows_per_image,
            "data_len": data_len,
            "reserved": reserved,
            **preview,
        }
        if not valid:
            fields["available_bytes"] = available
        elif expected_total < payload_len:
            fields["trailing_bytes"] = payload_len - expected_total
        return fields, valid

    if opcode == GpuOpcode.CreateSampler:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_SAMPLER_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": CREATE_SAMPLER_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            res_id,
            address_u,
            address_v,
            address_w,
            mag_filter,
            min_filter,
            mipmap_filter,
            max_anisotropy,
        ) = CREATE_SAMPLER_STRUCT.unpack_from(payload, 0)

        addr_u_name, addr_u_webgpu, addr_u_known = decode_address_mode(address_u)
        addr_v_name, addr_v_webgpu, addr_v_known = decode_address_mode(address_v)
        addr_w_name, addr_w_webgpu, addr_w_known = decode_address_mode(address_w)
        for axis, raw_val, is_known in (
            ("address_u", address_u, addr_u_known),
            ("address_v", address_v, addr_v_known),
            ("address_w", address_w, addr_w_known),
        ):
            if not is_known:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="unknown_address_mode",
                        offset=offset,
                        packet_index=packet_index,
                        message=f"CreateSampler(id={res_id}) has unknown {axis}={raw_val}",
                    )
                )

        mag_name, mag_webgpu, mag_known = decode_filter_mode(mag_filter)
        min_name, min_webgpu, min_known = decode_filter_mode(min_filter)
        mip_name, mip_webgpu, mip_known = decode_filter_mode(mipmap_filter)
        for filt_field, raw_val, is_known in (
            ("mag_filter", mag_filter, mag_known),
            ("min_filter", min_filter, min_known),
            ("mipmap_filter", mipmap_filter, mip_known),
        ):
            if not is_known:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="unknown_filter_mode",
                        offset=offset,
                        packet_index=packet_index,
                        message=f"CreateSampler(id={res_id}) has unknown {filt_field}={raw_val}",
                    )
                )

        fields = {
            "id": res_id,
            "address_u": address_u,
            "address_u_name": addr_u_name,
            "address_u_webgpu": addr_u_webgpu,
            "address_v": address_v,
            "address_v_name": addr_v_name,
            "address_v_webgpu": addr_v_webgpu,
            "address_w": address_w,
            "address_w_name": addr_w_name,
            "address_w_webgpu": addr_w_webgpu,
            "mag_filter": mag_filter,
            "mag_filter_name": mag_name,
            "mag_filter_webgpu": mag_webgpu,
            "min_filter": min_filter,
            "min_filter_name": min_name,
            "min_filter_webgpu": min_webgpu,
            "mipmap_filter": mipmap_filter,
            "mipmap_filter_name": mip_name,
            "mipmap_filter_webgpu": mip_webgpu,
            "max_anisotropy": max_anisotropy,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_SAMPLER_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateBindGroup:
        if payload_len < CREATE_BIND_GROUP_STRUCT.size:
            diagnostics.append(
                TraceDiagnostic(
                    severity="error",
                    code="payload_too_short",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"CreateBindGroup header expects at least "
                        f"{CREATE_BIND_GROUP_STRUCT.size} byte(s), got {payload_len}"
                    ),
                )
            )
            return {
                "expected_min_bytes": CREATE_BIND_GROUP_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        res_id, pipeline_id, group_index, entry_count = CREATE_BIND_GROUP_STRUCT.unpack_from(
            payload, 0
        )
        entry_stride = BIND_GROUP_ENTRY_STRUCT.size
        expected_total = CREATE_BIND_GROUP_STRUCT.size + entry_count * entry_stride
        available = payload_len - CREATE_BIND_GROUP_STRUCT.size
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
                        f"CreateBindGroup(id={res_id}) declares entry_count={entry_count} "
                        f"({entry_count * entry_stride} byte(s)), but only {available} byte(s) "
                        f"are present in packet payload"
                    ),
                )
            )
            decodable_entries = available // entry_stride
        else:
            if expected_total < payload_len:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="payload_size_mismatch",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"CreateBindGroup(id={res_id}) expects {expected_total} payload "
                            f"byte(s) ({CREATE_BIND_GROUP_STRUCT.size} + {entry_count} * "
                            f"{entry_stride}), got {payload_len}"
                        ),
                    )
                )
            decodable_entries = entry_count

        decoded_entries: list[dict[str, Any]] = []
        for idx in range(decodable_entries):
            entry_off = CREATE_BIND_GROUP_STRUCT.size + idx * entry_stride
            (
                binding,
                resource_type,
                resource_id,
                reserved,
                entry_offset,
                entry_size,
            ) = BIND_GROUP_ENTRY_STRUCT.unpack_from(payload, entry_off)
            rtype_name, rtype_known = decode_bind_resource_type(resource_type)
            if not rtype_known:
                diagnostics.append(
                    TraceDiagnostic(
                        severity=_warn_severity(strict),
                        code="unknown_bind_resource_type",
                        offset=offset,
                        packet_index=packet_index,
                        message=(
                            f"CreateBindGroup(id={res_id}) entry[{idx}] (binding={binding}) "
                            f"has unknown resource_type={resource_type}"
                        ),
                    )
                )
            decoded_entries.append(
                {
                    "binding": binding,
                    "resource_type": resource_type,
                    "resource_type_name": rtype_name,
                    "resource_id": resource_id,
                    "reserved": reserved,
                    "offset": entry_offset,
                    "size": entry_size,
                }
            )

        fields = {
            "id": res_id,
            "pipeline_id": pipeline_id,
            "group_index": group_index,
            "entry_count": entry_count,
            "entries": decoded_entries,
        }
        if not valid:
            fields["available_bytes"] = available
        elif expected_total < payload_len:
            fields["trailing_bytes"] = payload_len - expected_total
        return fields, valid

    if opcode == GpuOpcode.SetBindGroup:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_BIND_GROUP_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": SET_BIND_GROUP_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        group_index, bind_group_id = SET_BIND_GROUP_STRUCT.unpack_from(payload, 0)
        fields = {
            "group_index": group_index,
            "bind_group_id": bind_group_id,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_BIND_GROUP_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.SetIndexBuffer:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_INDEX_BUFFER_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": SET_INDEX_BUFFER_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        buffer_id, index_format, ib_offset = SET_INDEX_BUFFER_STRUCT.unpack_from(payload, 0)
        fmt_name, fmt_webgpu, fmt_known = decode_index_format(index_format)
        if not fmt_known:
            diagnostics.append(
                TraceDiagnostic(
                    severity=_warn_severity(strict),
                    code="unknown_index_format",
                    offset=offset,
                    packet_index=packet_index,
                    message=(
                        f"SetIndexBuffer(buffer_id={buffer_id}) has unknown "
                        f"index_format={index_format}"
                    ),
                )
            )
        fields = {
            "buffer_id": buffer_id,
            "index_format": index_format,
            "index_format_name": fmt_name,
            "index_format_webgpu": fmt_webgpu,
            "offset": ib_offset,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_INDEX_BUFFER_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.DrawIndexed:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=DRAW_INDEXED_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": DRAW_INDEXED_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (
            index_count,
            instance_count,
            first_index,
            base_vertex,
            first_instance,
        ) = DRAW_INDEXED_STRUCT.unpack_from(payload, 0)
        fields = {
            "index_count": index_count,
            "instance_count": instance_count,
            "first_index": first_index,
            "base_vertex": base_vertex,
            "first_instance": first_instance,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - DRAW_INDEXED_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.CreateComputePipeline:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=CREATE_COMPUTE_PIPELINE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": CREATE_COMPUTE_PIPELINE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        res_id, cs_shader_id, pipeline_flags, reserved = (
            CREATE_COMPUTE_PIPELINE_STRUCT.unpack_from(payload, 0)
        )
        fields = {
            "id": res_id,
            "cs_shader_id": cs_shader_id,
            "flags": pipeline_flags,
            "reserved": reserved,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - CREATE_COMPUTE_PIPELINE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.SetComputePipeline:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=SET_COMPUTE_PIPELINE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": SET_COMPUTE_PIPELINE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        (pipeline_id,) = SET_COMPUTE_PIPELINE_STRUCT.unpack_from(payload, 0)
        fields = {"pipeline_id": pipeline_id}
        if not exact:
            fields["trailing_bytes"] = payload_len - SET_COMPUTE_PIPELINE_STRUCT.size
        return fields, True

    if opcode == GpuOpcode.DispatchCompute:
        can_decode, exact = _check_fixed_size(
            opcode_name=name,
            expected_size=DISPATCH_COMPUTE_STRUCT.size,
            payload_bytes=payload_len,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )
        if not can_decode:
            return {
                "expected_bytes": DISPATCH_COMPUTE_STRUCT.size,
                "actual_bytes": payload_len,
            }, False
        workgroup_x, workgroup_y, workgroup_z = DISPATCH_COMPUTE_STRUCT.unpack_from(payload, 0)
        fields = {
            "workgroup_x": workgroup_x,
            "workgroup_y": workgroup_y,
            "workgroup_z": workgroup_z,
        }
        if not exact:
            fields["trailing_bytes"] = payload_len - DISPATCH_COMPUTE_STRUCT.size
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

    render_pass_opcodes = {
        GpuOpcode.SetPipeline,
        GpuOpcode.SetVertexBuffer,
        GpuOpcode.SetIndexBuffer,
        GpuOpcode.Draw,
        GpuOpcode.DrawIndexed,
    }
    compute_pass_opcodes = {
        GpuOpcode.SetComputePipeline,
        GpuOpcode.DispatchCompute,
    }

    def _flush(*, complete: bool) -> None:
        if not current:
            return
        histogram: Counter[str] = Counter()
        draw_calls = 0
        draw_indexed_calls = 0
        compute_dispatches = 0
        compute_passes = 0
        textures_created = 0
        bind_groups_created = 0
        total_bytes = 0
        active_pass: str | None = None
        for pkt in current:
            histogram[pkt.name] += 1
            total_bytes += pkt.total_bytes
            if not pkt.valid:
                continue
            if pkt.opcode == GpuOpcode.Draw:
                draw_calls += 1
            elif pkt.opcode == GpuOpcode.DrawIndexed:
                draw_calls += 1
                draw_indexed_calls += 1
            elif pkt.opcode == GpuOpcode.DispatchCompute:
                compute_dispatches += 1
            elif pkt.opcode == GpuOpcode.CreateTexture:
                textures_created += 1
            elif pkt.opcode == GpuOpcode.CreateBindGroup:
                bind_groups_created += 1

            if pkt.opcode in compute_pass_opcodes:
                if active_pass != "compute":
                    compute_passes += 1
                    active_pass = "compute"
            elif pkt.opcode in render_pass_opcodes:
                active_pass = "render"
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
                draw_indexed_calls=draw_indexed_calls,
                compute_dispatches=compute_dispatches,
                compute_passes=compute_passes,
                textures_created=textures_created,
                bind_groups_created=bind_groups_created,
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
    if op == GpuOpcode.CreateTexture:
        usage = "|".join(f["usage_names"]) if f["usage_names"] else "NONE"
        fmt = f["format_webgpu"] or f["format_name"]
        dim = f["dimension_webgpu"] or f["dimension_name"]
        return (
            f"id={f['id']} {f['width']}x{f['height']}x{f['depth_or_layers']} "
            f"mips={f['mip_levels']} dim={dim} fmt={fmt} "
            f"usage=0x{f['usage_flags']:04x}({usage})"
        )
    if op == GpuOpcode.WriteTexture:
        preview = f["preview_hex"]
        suffix = "..." if f["truncated_preview"] else ""
        preview_str = f" preview=0x{preview}{suffix}" if preview else ""
        return (
            f"texture_id={f['texture_id']} mip={f['mip_level']} "
            f"origin=({f['origin_x']},{f['origin_y']},{f['origin_z']}) "
            f"extent=({f['width']}x{f['height']}x{f['depth_or_layers']}) "
            f"bpr={f['bytes_per_row']} rpi={f['rows_per_image']} "
            f"data_len={f['data_len']} crc32={f['crc32']}{preview_str}"
        )
    if op == GpuOpcode.CreateSampler:
        addr_u = f["address_u_webgpu"] or f["address_u_name"]
        addr_v = f["address_v_webgpu"] or f["address_v_name"]
        addr_w = f["address_w_webgpu"] or f["address_w_name"]
        mag = f["mag_filter_webgpu"] or f["mag_filter_name"]
        min_f = f["min_filter_webgpu"] or f["min_filter_name"]
        mip = f["mipmap_filter_webgpu"] or f["mipmap_filter_name"]
        return (
            f"id={f['id']} addr=({addr_u},{addr_v},{addr_w}) "
            f"filter=(mag={mag},min={min_f},mip={mip}) aniso={f['max_anisotropy']}"
        )
    if op == GpuOpcode.CreateBindGroup:
        entry_parts: list[str] = []
        for entry in f["entries"]:
            base = f"@{entry['binding']}:{entry['resource_type_name']}(#{entry['resource_id']})"
            if entry["resource_type"] == GpuBindResourceType.Buffer and (
                entry["offset"] or entry["size"]
            ):
                size_str = str(entry["size"]) if entry["size"] else "end"
                base += f"[{entry['offset']}:{size_str}]"
            entry_parts.append(base)
        entries_str = ", ".join(entry_parts)
        return (
            f"id={f['id']} pipeline={f['pipeline_id']} group={f['group_index']} "
            f"entries={f['entry_count']} [{entries_str}]"
        )
    if op == GpuOpcode.SetBindGroup:
        return f"group={f['group_index']} bind_group_id={f['bind_group_id']}"
    if op == GpuOpcode.SetIndexBuffer:
        fmt = f["index_format_webgpu"] or f["index_format_name"]
        return f"buffer_id={f['buffer_id']} format={fmt} offset={f['offset']}"
    if op == GpuOpcode.DrawIndexed:
        return (
            f"indices={f['index_count']} instances={f['instance_count']} "
            f"first_index={f['first_index']} base_vertex={f['base_vertex']} "
            f"first_instance={f['first_instance']}"
        )
    if op == GpuOpcode.CreateComputePipeline:
        return f"id={f['id']} cs={f['cs_shader_id']} flags=0x{f['flags']:x}"
    if op == GpuOpcode.SetComputePipeline:
        return f"pipeline_id={f['pipeline_id']}"
    if op == GpuOpcode.DispatchCompute:
        return f"workgroups=({f['workgroup_x']}, {f['workgroup_y']}, {f['workgroup_z']})"
    return " ".join(f"{k}={v}" for k, v in f.items())


def format_trace_text(trace: DecodedTrace, *, stats_only: bool = False) -> str:
    """Format a DecodedTrace as human-readable disassembly and/or statistics."""
    lines: list[str] = [
        f"=== playgta6 GPU Trace (ABI v{trace.abi_version}) ===",
        (
            f"Bytes: {trace.consumed_bytes}/{trace.total_bytes} | "
            f"Packets: {trace.packet_count} | Frames: {trace.frame_count} | "
            f"Draws: {trace.draw_calls} | Dispatches: {trace.compute_dispatches} | "
            f"Unknown: {trace.unknown_opcode_count} | "
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
        extras: list[str] = []
        if frame.draw_indexed_calls:
            extras.append(f"indexed={frame.draw_indexed_calls}")
        if frame.compute_dispatches:
            extras.append(f"dispatches={frame.compute_dispatches}")
        if frame.compute_passes:
            extras.append(f"compute_passes={frame.compute_passes}")
        if frame.textures_created:
            extras.append(f"textures={frame.textures_created}")
        if frame.bind_groups_created:
            extras.append(f"bind_groups={frame.bind_groups_created}")
        extra_str = (" " + " ".join(extras)) if extras else ""
        lines.append(
            f"  Frame {frame.frame_index:03d} [{status}]: "
            f"packets={frame.packet_count} (#{frame.start_packet_index}..#{frame.end_packet_index}) "
            f"draws={frame.draw_calls}{extra_str} bytes={frame.bytes} | {hist_str}"
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
            "Fail with non-zero exit code on unknown opcodes (>= 22), "
            "size mismatches, or malformed/truncated packets."
        ),
    )
    parser.add_argument(
        "--preview-bytes",
        type=int,
        default=DEFAULT_PREVIEW_BYTES,
        metavar="N",
        help=(
            "Maximum raw payload preview bytes for WriteBuffer/WriteTexture/Unknown "
            f"(default: {DEFAULT_PREVIEW_BYTES})."
        ),
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
