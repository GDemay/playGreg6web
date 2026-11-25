#!/usr/bin/env python3
"""Binary GPU command stream disassembler and frame trace inspector for playgta6web.

Matches the packet layout in `runtime/abi/playgta6_abi.h` and
`docs/gpu_commands.md` (v1 opcodes 0..21, v2 opcodes 22..34, plus forward-compatible
handling of unknown/future opcodes >= 35).

Pure Python 3.12 stdlib only (zero external dependencies). Usable both as an
importable library and as a CLI:

    python tools/gpu_trace.py <trace.bin|timings.json> [--json] [--stats] [--strict] [--shader-summary] [--binding-audit] [--bink-video-audit] [--start-screen-skips] [--analyze-30fps] [--budget-ms 33.333] [--top-n 10]
"""

from __future__ import annotations

import argparse
import json
import math
import re
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

# Packet ABI v2 (opcodes 22..34).
TEXTURE_VIEW_DESC_STRUCT = struct.Struct("<BBBBHHII")  # 16 bytes
BIND_GROUP_LAYOUT_ENTRY_STRUCT = struct.Struct("<IIBBBBB3xQ")  # 24 bytes
CREATE_BIND_GROUP_LAYOUT_STRUCT = struct.Struct("<IIIIQ")  # 24 bytes + entry_count * 24
CREATE_PIPELINE_LAYOUT_STRUCT = struct.Struct("<IIIIQ")  # 24 bytes + count * 4
RENDER_PIPELINE_KEY_SIZE = 376
CREATE_RENDER_PIPELINE2_HEAD_STRUCT = struct.Struct("<IIII")  # 16 bytes + 376-byte key
CREATE_RENDER_PIPELINE2_SIZE = CREATE_RENDER_PIPELINE2_HEAD_STRUCT.size + RENDER_PIPELINE_KEY_SIZE
KEY_HEAD_STRUCT = struct.Struct("<QQII")  # vs_shader, ps_shader, root_signature_id, color_target_count
KEY_COLOR_TARGET_STRUCT = struct.Struct("<BBBxBBBxBBBx")  # 12 bytes
KEY_DEPTH_STENCIL_STRUCT = struct.Struct("<BBBxBBBBBBBBIIiII")  # 32 bytes
KEY_PRIMITIVE_STRUCT = struct.Struct("<BBBBB3x")  # 8 bytes
KEY_MULTISAMPLE_STRUCT = struct.Struct("<IIB3x")  # 12 bytes
KEY_COUNTS_STRUCT = struct.Struct("<II")
KEY_VERTEX_BUFFER_STRUCT = struct.Struct("<IBB2x")  # 8 bytes
KEY_VERTEX_ATTRIBUTE_STRUCT = struct.Struct("<IBBBx")  # 8 bytes
CREATE_COMPUTE_PIPELINE2_STRUCT = struct.Struct("<IIIIQII")  # 32 bytes
RENDER_PASS_COLOR_ATTACHMENT_STRUCT = struct.Struct("<IBBH16sffff")  # 40 bytes
RENDER_PASS_DEPTH_STENCIL_STRUCT = struct.Struct("<IBBBB16sfIBBHI")  # 40 bytes
BEGIN_RENDER_PASS_HEAD_STRUCT = struct.Struct("<II")
BEGIN_RENDER_PASS_SIZE = 368
SET_VIEWPORT_STRUCT = struct.Struct("<ffffff")  # 24 bytes
SET_SCISSOR_RECT_STRUCT = struct.Struct("<IIII")  # 16 bytes
SET_BLEND_CONSTANT_STRUCT = struct.Struct("<ffff")  # 16 bytes
SET_STENCIL_REFERENCE_STRUCT = struct.Struct("<I")  # 4 bytes
BIND_GROUP_ENTRY2_STRUCT = struct.Struct("<IIIIQQ16s")  # 48 bytes
CREATE_BIND_GROUP2_STRUCT = struct.Struct("<IIII")  # 16 bytes + entry_count * 48
SET_BIND_GROUP2_STRUCT = struct.Struct("<IIII")  # 16 bytes + count * 4
CREATE_SAMPLER2_STRUCT = struct.Struct("<IBBBBBBBxHxxff")  # 24 bytes
CREATE_TEXTURE2_STRUCT = struct.Struct("<IIIIIIIBBBB8s")  # 40 bytes (GpuCreateTexture2Payload)

SWAPCHAIN_TEXTURE_ID = 0xFFFFFFFF
MANIFEST_REFERENCE_FLAG = 0x1

DEFAULT_PREVIEW_BYTES = 16
DEFAULT_30FPS_BUDGET_MS = 33.333
DEFAULT_TOP_N = 10
SPIKE_ANALYSIS_SCHEMA_VERSION = 1

SPIKE_CAUSES: tuple[str, ...] = (
    "pso_compile",
    "staging_upload",
    "draw_burst",
    "rt_alloc",
    "sync_stall",
    "unknown",
)

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

# v2 "WebGPU enum codes" (docs/gpu_commands.md; numbering of pso::Wgpu* in runtime/gpu/pso_cache.h).
WGPU_TEXTURE_FORMAT_NAMES: tuple[str, ...] = (
    "undefined", "r8unorm", "r8snorm", "r8uint", "r8sint", "r16unorm", "r16snorm", "r16uint",
    "r16sint", "r16float", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint", "r32uint", "r32sint",
    "r32float", "rg16unorm", "rg16snorm", "rg16uint", "rg16sint", "rg16float", "rgba8unorm",
    "rgba8unorm-srgb", "rgba8snorm", "rgba8uint", "rgba8sint", "bgra8unorm", "bgra8unorm-srgb",
    "rgb9e5ufloat", "rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat", "rg32uint", "rg32sint",
    "rg32float", "rgba16unorm", "rgba16snorm", "rgba16uint", "rgba16sint", "rgba16float",
    "rgba32uint", "rgba32sint", "rgba32float", "stencil8", "depth16unorm", "depth24plus",
    "depth24plus-stencil8", "depth32float", "depth32float-stencil8", "bc1-rgba-unorm",
    "bc1-rgba-unorm-srgb", "bc2-rgba-unorm", "bc2-rgba-unorm-srgb", "bc3-rgba-unorm",
    "bc3-rgba-unorm-srgb", "bc4-r-unorm", "bc4-r-snorm", "bc5-rg-unorm", "bc5-rg-snorm",
    "bc6h-rgb-ufloat", "bc6h-rgb-float", "bc7-rgba-unorm", "bc7-rgba-unorm-srgb",
)
WGPU_VERTEX_FORMAT_NAMES: tuple[str, ...] = (
    "undefined", "uint8", "uint8x2", "uint8x4", "sint8", "sint8x2", "sint8x4", "unorm8",
    "unorm8x2", "unorm8x4", "snorm8", "snorm8x2", "snorm8x4", "uint16", "uint16x2", "uint16x4",
    "sint16", "sint16x2", "sint16x4", "unorm16", "unorm16x2", "unorm16x4", "snorm16",
    "snorm16x2", "snorm16x4", "float16", "float16x2", "float16x4", "float32", "float32x2",
    "float32x3", "float32x4", "uint32", "uint32x2", "uint32x3", "uint32x4", "sint32",
    "sint32x2", "sint32x3", "sint32x4", "unorm10-10-10-2", "unorm8x4-bgra",
)
WGPU_BLEND_FACTOR_NAMES: tuple[str, ...] = (
    "zero", "one", "src", "one-minus-src", "src-alpha", "one-minus-src-alpha", "dst",
    "one-minus-dst", "dst-alpha", "one-minus-dst-alpha", "src-alpha-saturated", "constant",
    "one-minus-constant", "src1", "one-minus-src1", "src1-alpha", "one-minus-src1-alpha",
)
WGPU_BLEND_OPERATION_NAMES: tuple[str, ...] = ("add", "subtract", "reverse-subtract", "min", "max")
WGPU_COMPARE_FUNCTION_NAMES: tuple[str, ...] = (
    "undefined", "never", "less", "equal", "less-equal", "greater", "not-equal",
    "greater-equal", "always",
)
WGPU_STENCIL_OPERATION_NAMES: tuple[str, ...] = (
    "keep", "zero", "replace", "invert", "increment-clamp", "decrement-clamp",
    "increment-wrap", "decrement-wrap",
)
WGPU_CULL_MODE_NAMES: tuple[str, ...] = ("none", "front", "back")
WGPU_FRONT_FACE_NAMES: tuple[str, ...] = ("ccw", "cw")
WGPU_INDEX_FORMAT_NAMES: tuple[str, ...] = ("undefined", "uint16", "uint32")
WGPU_VERTEX_STEP_MODE_NAMES: tuple[str, ...] = ("unused", "vertex", "instance")
WGPU_LOAD_OP_NAMES: tuple[str, ...] = ("undefined", "load", "clear")
WGPU_STORE_OP_NAMES: tuple[str, ...] = ("undefined", "store", "discard")
WGPU_TEXTURE_ASPECT_NAMES: tuple[str, ...] = ("all", "depth-only", "stencil-only")
WGPU_VIEW_DIMENSION_NAMES: tuple[str, ...] = (
    "undefined", "1d", "2d", "2d-array", "cube", "cube-array", "3d",
)
BINDING_LAYOUT_TYPE_NAMES: tuple[str, ...] = ("buffer", "sampler", "texture", "storageTexture")
BINDING_SUBTYPE_NAMES: dict[int, tuple[str, ...]] = {
    0: ("uniform", "storage", "read-only-storage"),
    1: ("filtering", "non-filtering", "comparison"),
    2: ("float", "unfilterable-float", "depth", "sint", "uint"),
    3: ("write-only", "read-only", "read-write"),
}
SHADER_STAGE_BITS: tuple[tuple[int, str], ...] = ((0x1, "VERTEX"), (0x2, "FRAGMENT"), (0x4, "COMPUTE"))


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
    # Packet ABI v2 summary metrics (opcodes 22..34).
    begin_render_passes: int = 0
    end_render_passes: int = 0
    color_load_ops: dict[str, int] | None = None
    color_store_ops: dict[str, int] | None = None
    depth_load_ops: dict[str, int] | None = None
    depth_store_ops: dict[str, int] | None = None
    bind_group_layouts_created: int = 0
    pipeline_layouts_created: int = 0
    render_pipelines2_created: int = 0
    compute_pipelines2_created: int = 0
    viewport_changes: int = 0
    scissor_changes: int = 0
    blend_constant_changes: int = 0
    stencil_ref_changes: int = 0
    malformed_v2_packets: int = 0

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
            "begin_render_passes": self.begin_render_passes,
            "end_render_passes": self.end_render_passes,
            "color_load_ops": dict(self.color_load_ops or {}),
            "color_store_ops": dict(self.color_store_ops or {}),
            "depth_load_ops": dict(self.depth_load_ops or {}),
            "depth_store_ops": dict(self.depth_store_ops or {}),
            "bind_group_layouts_created": self.bind_group_layouts_created,
            "pipeline_layouts_created": self.pipeline_layouts_created,
            "render_pipelines2_created": self.render_pipelines2_created,
            "compute_pipelines2_created": self.compute_pipelines2_created,
            "viewport_changes": self.viewport_changes,
            "scissor_changes": self.scissor_changes,
            "blend_constant_changes": self.blend_constant_changes,
            "stencil_ref_changes": self.stencil_ref_changes,
            "malformed_v2_packets": self.malformed_v2_packets,
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
    _shader_hash_refs: tuple[tuple[int, str], ...] = ()

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
        return sum(
            1
            for p in self.packets
            if p.opcode in (GpuOpcode.CreateBindGroup, GpuOpcode.CreateBindGroup2) and p.valid
        )

    @property
    def begin_render_passes(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.BeginRenderPass and p.valid)

    @property
    def end_render_passes(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.EndRenderPass and p.valid)

    @property
    def bind_group_layouts_created(self) -> int:
        return sum(
            1 for p in self.packets if p.opcode == GpuOpcode.CreateBindGroupLayout and p.valid
        )

    @property
    def pipeline_layouts_created(self) -> int:
        return sum(
            1 for p in self.packets if p.opcode == GpuOpcode.CreatePipelineLayout and p.valid
        )

    @property
    def render_pipelines2_created(self) -> int:
        return sum(
            1 for p in self.packets if p.opcode == GpuOpcode.CreateRenderPipeline2 and p.valid
        )

    @property
    def compute_pipelines2_created(self) -> int:
        return sum(
            1 for p in self.packets if p.opcode == GpuOpcode.CreateComputePipeline2 and p.valid
        )

    @property
    def viewport_changes(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.SetViewport and p.valid)

    @property
    def scissor_changes(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.SetScissorRect and p.valid)

    @property
    def blend_constant_changes(self) -> int:
        return sum(1 for p in self.packets if p.opcode == GpuOpcode.SetBlendConstant and p.valid)

    @property
    def stencil_ref_changes(self) -> int:
        return sum(
            1 for p in self.packets if p.opcode == GpuOpcode.SetStencilReference and p.valid
        )

    @property
    def color_load_ops(self) -> dict[str, int]:
        return _merge_op_histograms(f.color_load_ops for f in self.frames)

    @property
    def color_store_ops(self) -> dict[str, int]:
        return _merge_op_histograms(f.color_store_ops for f in self.frames)

    @property
    def depth_load_ops(self) -> dict[str, int]:
        return _merge_op_histograms(f.depth_load_ops for f in self.frames)

    @property
    def depth_store_ops(self) -> dict[str, int]:
        return _merge_op_histograms(f.depth_store_ops for f in self.frames)

    @property
    def malformed_v2_packets(self) -> int:
        return sum(1 for p in self.packets if _is_v2_opcode(p.opcode) and not p.valid)

    @property
    def truncated_v2_diagnostics(self) -> int:
        """Count truncated/short-payload diagnostics tied to ABI v2 packets."""
        return sum(
            1
            for d in self.diagnostics
            if d.code in _V2_TRUNCATION_DIAG_CODES and _diag_targets_v2(self, d)
        )

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

    def to_dict(
        self,
        *,
        stats_only: bool = False,
        shader_summary: bool = False,
        binding_audit: bool = False,
        bink_video_audit: bool = False,
        start_screen_skips: bool = False,
        analyze_30fps: bool = False,
        budget_ms: float = DEFAULT_30FPS_BUDGET_MS,
        top_n: int = DEFAULT_TOP_N,
    ) -> dict[str, Any]:
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
            "begin_render_passes": self.begin_render_passes,
            "end_render_passes": self.end_render_passes,
            "color_load_ops": self.color_load_ops,
            "color_store_ops": self.color_store_ops,
            "depth_load_ops": self.depth_load_ops,
            "depth_store_ops": self.depth_store_ops,
            "bind_group_layouts_created": self.bind_group_layouts_created,
            "pipeline_layouts_created": self.pipeline_layouts_created,
            "render_pipelines2_created": self.render_pipelines2_created,
            "compute_pipelines2_created": self.compute_pipelines2_created,
            "viewport_changes": self.viewport_changes,
            "scissor_changes": self.scissor_changes,
            "blend_constant_changes": self.blend_constant_changes,
            "stencil_ref_changes": self.stencil_ref_changes,
            "malformed_v2_packets": self.malformed_v2_packets,
            "truncated_v2_diagnostics": self.truncated_v2_diagnostics,
            "unknown_opcode_count": self.unknown_opcode_count,
            "opcode_histogram": dict(self.opcode_histogram),
            "frames": [frame.to_dict() for frame in self.frames],
            "diagnostics": [diag.to_dict() for diag in self.diagnostics],
        }
        if not stats_only:
            result["packets"] = [pkt.to_dict() for pkt in self.packets]
        if shader_summary:
            result["shader_summary"] = summarize_shaders(self)
        if binding_audit:
            result["binding_audit"] = audit_bindings(self)
        if bink_video_audit:
            result["bink_video_audit"] = audit_bink_video(self)
        if start_screen_skips:
            result["start_screen_skips"] = audit_start_screen_skips(self)
        if analyze_30fps:
            result["fps30_analysis"] = analyze_30fps_trace(
                self,
                budget_ms=budget_ms,
                top_n=top_n,
            )
        return result


_V2_TRUNCATION_DIAG_CODES = frozenset(
    {
        "truncated_header",
        "truncated_payload",
        "payload_too_short",
        "variable_payload_truncated",
    }
)


def _is_v2_opcode(opcode: int) -> bool:
    return GpuOpcode.CreateBindGroupLayout <= opcode <= GpuOpcode.CreateSampler2


def _merge_op_histograms(histograms: Any) -> dict[str, int]:
    merged: Counter[str] = Counter()
    for hist in histograms:
        if hist:
            merged.update(hist)
    return dict(merged)


def _diag_targets_v2(trace: DecodedTrace, diag: TraceDiagnostic) -> bool:
    if diag.packet_index is not None and 0 <= diag.packet_index < len(trace.packets):
        return _is_v2_opcode(trace.packets[diag.packet_index].opcode)
    # Stream-level truncations (no packet object emitted): match "(opcode N)" in the message.
    for op in range(int(GpuOpcode.CreateBindGroupLayout), int(GpuOpcode.CreateSampler2) + 1):
        if f"(opcode {op})" in diag.message:
            return True
    return False


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


def encode_texture_view_desc(
    texture_format: int = 0,
    dimension: int = 0,
    aspect: int = 0,
    base_mip_level: int = 0,
    mip_level_count: int = 0,
    base_array_layer: int = 0,
    array_layer_count: int = 0,
) -> bytes:
    """Encode a 16-byte GpuTextureViewDesc (all zero = default view)."""
    return TEXTURE_VIEW_DESC_STRUCT.pack(
        texture_format, dimension, aspect, 0,
        base_mip_level, mip_level_count, base_array_layer, array_layer_count,
    )


def encode_create_bind_group_layout(
    layout_id: int,
    entries: Sequence[Sequence[int]],
    *,
    content_id: int = 0,
    flags: int = 0,
    packet_flags: int = 0,
) -> bytes:
    """Encode CreateBindGroupLayout. Each entry is (binding, visibility, type, binding_type
    [, view_dimension, entry_flags, storage_format, min_binding_size])."""
    body = bytearray(
        CREATE_BIND_GROUP_LAYOUT_STRUCT.pack(layout_id, len(entries), flags, 0, content_id)
    )
    for entry in entries:
        padded = tuple(entry) + (0,) * (8 - len(entry))
        body += BIND_GROUP_LAYOUT_ENTRY_STRUCT.pack(*padded)
    return encode_packet(GpuOpcode.CreateBindGroupLayout, body, packet_flags)


def encode_create_pipeline_layout(
    layout_id: int,
    bind_group_layout_ids: Sequence[int],
    *,
    content_id: int = 0,
    flags: int = 0,
    packet_flags: int = 0,
) -> bytes:
    """Encode CreatePipelineLayout (group i uses bind_group_layout_ids[i]; 0 = empty layout)."""
    body = CREATE_PIPELINE_LAYOUT_STRUCT.pack(
        layout_id, len(bind_group_layout_ids), flags, 0, content_id
    ) + struct.pack(f"<{len(bind_group_layout_ids)}I", *bind_group_layout_ids)
    return encode_packet(GpuOpcode.CreatePipelineLayout, body, packet_flags)


def encode_create_render_pipeline2(
    pipeline_id: int,
    vs_module_id: int,
    fs_module_id: int,
    pipeline_layout_id: int,
    key: bytes | bytearray,
    flags: int = 0,
) -> bytes:
    """Encode CreateRenderPipeline2 with a raw 376-byte GpuRenderPipelineKey."""
    if len(key) != RENDER_PIPELINE_KEY_SIZE:
        raise ValueError(
            f"render pipeline key must be {RENDER_PIPELINE_KEY_SIZE} bytes, got {len(key)}"
        )
    head = CREATE_RENDER_PIPELINE2_HEAD_STRUCT.pack(
        pipeline_id, vs_module_id, fs_module_id, pipeline_layout_id
    )
    return encode_packet(GpuOpcode.CreateRenderPipeline2, head + bytes(key), flags)


def encode_create_compute_pipeline2(
    pipeline_id: int,
    cs_module_id: int,
    pipeline_layout_id: int,
    cs_shader: int = 0,
    root_signature_id: int = 0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.CreateComputePipeline2,
        CREATE_COMPUTE_PIPELINE2_STRUCT.pack(
            pipeline_id, cs_module_id, pipeline_layout_id, 0, cs_shader, root_signature_id, 0
        ),
        flags,
    )


def encode_begin_render_pass(
    colors: Sequence[dict[str, Any] | None],
    depth_stencil: dict[str, Any] | None = None,
    flags: int = 0,
) -> bytes:
    """Encode BeginRenderPass.

    Color dicts: texture_id, load_op, store_op[, clear_value, view]. The depth-stencil dict:
    texture_id[, depth_load_op, depth_store_op, stencil_load_op, stencil_store_op,
    depth_clear_value, stencil_clear_value, depth_read_only, stencil_read_only, view].
    """
    if len(colors) > 8:
        raise ValueError("at most 8 color attachments")
    body = bytearray(BEGIN_RENDER_PASS_HEAD_STRUCT.pack(len(colors), 0))
    zero_view = encode_texture_view_desc()
    for i in range(8):
        c = colors[i] if i < len(colors) else None
        if c is None:
            body += bytes(RENDER_PASS_COLOR_ATTACHMENT_STRUCT.size)
            continue
        r, g, b, a = c.get("clear_value", (0.0, 0.0, 0.0, 0.0))
        body += RENDER_PASS_COLOR_ATTACHMENT_STRUCT.pack(
            c["texture_id"], c["load_op"], c["store_op"], 0, c.get("view", zero_view), r, g, b, a
        )
    if depth_stencil is None:
        body += bytes(RENDER_PASS_DEPTH_STENCIL_STRUCT.size)
    else:
        ds = depth_stencil
        body += RENDER_PASS_DEPTH_STENCIL_STRUCT.pack(
            ds["texture_id"],
            ds.get("depth_load_op", 0),
            ds.get("depth_store_op", 0),
            ds.get("stencil_load_op", 0),
            ds.get("stencil_store_op", 0),
            ds.get("view", zero_view),
            ds.get("depth_clear_value", 1.0),
            ds.get("stencil_clear_value", 0),
            int(bool(ds.get("depth_read_only", False))),
            int(bool(ds.get("stencil_read_only", False))),
            0,
            0,
        )
    return encode_packet(GpuOpcode.BeginRenderPass, body, flags)


def encode_end_render_pass(flags: int = 0) -> bytes:
    return encode_packet(GpuOpcode.EndRenderPass, b"", flags)


def encode_set_viewport(
    x: float,
    y: float,
    width: float,
    height: float,
    min_depth: float = 0.0,
    max_depth: float = 1.0,
    flags: int = 0,
) -> bytes:
    return encode_packet(
        GpuOpcode.SetViewport,
        SET_VIEWPORT_STRUCT.pack(x, y, width, height, min_depth, max_depth),
        flags,
    )


def encode_set_scissor_rect(x: int, y: int, width: int, height: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetScissorRect, SET_SCISSOR_RECT_STRUCT.pack(x, y, width, height), flags
    )


def encode_set_blend_constant(r: float, g: float, b: float, a: float, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetBlendConstant, SET_BLEND_CONSTANT_STRUCT.pack(r, g, b, a), flags
    )


def encode_set_stencil_reference(reference: int, flags: int = 0) -> bytes:
    return encode_packet(
        GpuOpcode.SetStencilReference, SET_STENCIL_REFERENCE_STRUCT.pack(reference), flags
    )


def encode_create_bind_group2(
    bind_group_id: int,
    layout_id: int,
    entries: Sequence[Sequence[Any]],
    flags: int = 0,
) -> bytes:
    """Encode CreateBindGroup2. Each entry is (binding, resource_type, resource_id
    [, offset, size, view_bytes])."""
    body = bytearray(CREATE_BIND_GROUP2_STRUCT.pack(bind_group_id, layout_id, len(entries), 0))
    for entry in entries:
        binding, rtype, rid = entry[0], int(entry[1]), entry[2]
        e_offset = entry[3] if len(entry) > 3 else 0
        e_size = entry[4] if len(entry) > 4 else 0
        view = entry[5] if len(entry) > 5 else encode_texture_view_desc()
        body += BIND_GROUP_ENTRY2_STRUCT.pack(binding, rtype, rid, 0, e_offset, e_size, view)
    return encode_packet(GpuOpcode.CreateBindGroup2, body, flags)


def encode_set_bind_group2(
    group_index: int,
    bind_group_id: int,
    dynamic_offsets: Sequence[int] = (),
    flags: int = 0,
) -> bytes:
    body = SET_BIND_GROUP2_STRUCT.pack(
        group_index, bind_group_id, len(dynamic_offsets), 0
    ) + struct.pack(f"<{len(dynamic_offsets)}I", *dynamic_offsets)
    return encode_packet(GpuOpcode.SetBindGroup2, body, flags)


def encode_create_sampler2(
    sampler_id: int,
    address: tuple[int, int, int] = (0, 0, 0),
    filters: tuple[int, int, int] = (1, 1, 1),
    compare: int = 0,
    max_anisotropy: int = 1,
    lod_min_clamp: float = 0.0,
    lod_max_clamp: float = 32.0,
    flags: int = 0,
) -> bytes:
    """Encode CreateSampler2 (address = (u, v, w), filters = (mag, min, mipmap))."""
    return encode_packet(
        GpuOpcode.CreateSampler2,
        CREATE_SAMPLER2_STRUCT.pack(
            sampler_id, *address, *filters, compare, max_anisotropy, lod_min_clamp, lod_max_clamp
        ),
        flags,
    )


def encode_create_texture2(
    resource_id: int,
    width: int,
    height: int,
    texture_format: int,
    *,
    depth_or_layers: int = 1,
    mip_levels: int = 1,
    sample_count: int = 1,
    usage_flags: int = 0,
    dimension: int | GpuTextureDimension = GpuTextureDimension.Dim2D,
    view_formats: Sequence[int] = (),
    flags: int = 0,
) -> bytes:
    """Encode CreateTexture2 (opcode 35) using v2 WebGPU texture format codes."""
    if len(view_formats) > 8:
        raise ValueError(f"at most 8 view_formats, got {len(view_formats)}")
    padded_views = bytes(view_formats) + bytes(8 - len(view_formats))
    return encode_packet(
        GpuOpcode.CreateTexture2,
        CREATE_TEXTURE2_STRUCT.pack(
            resource_id,
            width,
            height,
            depth_or_layers,
            mip_levels,
            sample_count,
            usage_flags,
            int(dimension),
            texture_format,
            len(view_formats),
            0,
            padded_views,
        ),
        flags,
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
    """Decode payload for a known opcode (0..34). Returns (fields, valid)."""
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

    if opcode >= GpuOpcode.CreateBindGroupLayout:
        return _decode_v2_payload(
            opcode=opcode,
            payload=payload,
            offset=offset,
            packet_index=packet_index,
            strict=strict,
            diagnostics=diagnostics,
        )

    raise AssertionError(f"Unhandled known opcode: {opcode}")


# ---------------------------------------------------------------------------
# Packet ABI v2 decoding (opcodes 22..34)
# ---------------------------------------------------------------------------


class _V2Context:
    """Shared diagnostics plumbing for one v2 packet."""

    def __init__(
        self,
        name: str,
        offset: int,
        packet_index: int,
        strict: bool,
        diagnostics: list[TraceDiagnostic],
    ) -> None:
        self.name = name
        self.offset = offset
        self.packet_index = packet_index
        self.strict = strict
        self.diagnostics = diagnostics

    def warn(self, code: str, message: str) -> None:
        self.diagnostics.append(
            TraceDiagnostic(
                severity=_warn_severity(self.strict),
                code=code,
                offset=self.offset,
                packet_index=self.packet_index,
                message=f"{self.name}: {message}",
            )
        )

    def error(self, code: str, message: str) -> None:
        self.diagnostics.append(
            TraceDiagnostic(
                severity="error",
                code=code,
                offset=self.offset,
                packet_index=self.packet_index,
                message=f"{self.name}: {message}",
            )
        )

    def code(self, table: Sequence[str], value: int, what: str) -> str:
        """Name for an enum code; unknown codes produce an `unknown_enum_code` diagnostic."""
        if 0 <= value < len(table):
            return table[value]
        self.warn("unknown_enum_code", f"unknown {what} code {value}")
        return f"Unknown({value})"

    def fixed(self, payload_len: int, size: int) -> tuple[bool, bool]:
        return _check_fixed_size(
            opcode_name=self.name,
            expected_size=size,
            payload_bytes=payload_len,
            offset=self.offset,
            packet_index=self.packet_index,
            strict=self.strict,
            diagnostics=self.diagnostics,
        )

    def tail(self, payload_len: int, head_size: int, count: int, stride: int) -> tuple[int, bool, int]:
        """Validate `head + count * stride`; returns (decodable_count, valid, trailing_bytes)."""
        expected = head_size + count * stride
        available = payload_len - head_size
        if expected > payload_len:
            self.error(
                "variable_payload_truncated",
                f"declares {count} element(s) ({count * stride} byte(s)) but only "
                f"{available} byte(s) are present",
            )
            return available // stride, False, 0
        if expected < payload_len:
            self.warn(
                "payload_size_mismatch",
                f"expects {expected} payload byte(s), got {payload_len}",
            )
        return count, True, payload_len - expected


def _f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _decode_view_desc(ctx: _V2Context, raw: bytes | memoryview) -> dict[str, Any]:
    fmt, dim, aspect, _reserved, base_mip, mip_count, base_layer, layer_count = (
        TEXTURE_VIEW_DESC_STRUCT.unpack(bytes(raw))
    )
    return {
        "format": ctx.code(WGPU_TEXTURE_FORMAT_NAMES, fmt, "texture format"),
        "dimension": ctx.code(WGPU_VIEW_DIMENSION_NAMES, dim, "view dimension"),
        "aspect": ctx.code(WGPU_TEXTURE_ASPECT_NAMES, aspect, "texture aspect"),
        "base_mip_level": base_mip,
        "mip_level_count": mip_count,
        "base_array_layer": base_layer,
        "array_layer_count": layer_count,
    }


def _decode_render_pipeline_key(ctx: _V2Context, key: bytes) -> dict[str, Any]:
    vs_shader, ps_shader, root_signature_id, color_target_count = KEY_HEAD_STRUCT.unpack_from(key, 0)
    if color_target_count > 8:
        ctx.error("invalid_count", f"color_target_count {color_target_count} > 8")
    color_targets: list[dict[str, Any] | None] = []
    for i in range(min(color_target_count, 8)):
        (fmt, blend_enabled, write_mask, c_op, c_src, c_dst, a_op, a_src, a_dst) = (
            KEY_COLOR_TARGET_STRUCT.unpack_from(key, 24 + 12 * i)
        )
        if fmt == 0:
            color_targets.append(None)
            continue
        target: dict[str, Any] = {
            "format": ctx.code(WGPU_TEXTURE_FORMAT_NAMES, fmt, "texture format"),
            "write_mask": write_mask,
        }
        if blend_enabled:
            target["blend"] = {
                "color": {
                    "operation": ctx.code(WGPU_BLEND_OPERATION_NAMES, c_op, "blend operation"),
                    "src_factor": ctx.code(WGPU_BLEND_FACTOR_NAMES, c_src, "blend factor"),
                    "dst_factor": ctx.code(WGPU_BLEND_FACTOR_NAMES, c_dst, "blend factor"),
                },
                "alpha": {
                    "operation": ctx.code(WGPU_BLEND_OPERATION_NAMES, a_op, "blend operation"),
                    "src_factor": ctx.code(WGPU_BLEND_FACTOR_NAMES, a_src, "blend factor"),
                    "dst_factor": ctx.code(WGPU_BLEND_FACTOR_NAMES, a_dst, "blend factor"),
                },
            }
        color_targets.append(target)

    (
        ds_format, depth_write, depth_compare,
        f_cmp, f_fail, f_dfail, f_pass, b_cmp, b_fail, b_dfail, b_pass,
        read_mask, write_mask_ds, depth_bias, slope_bits, clamp_bits,
    ) = KEY_DEPTH_STENCIL_STRUCT.unpack_from(key, 120)
    depth_stencil: dict[str, Any] | None = None
    if ds_format != 0:

        def face(cmp: int, fail: int, dfail: int, pss: int) -> dict[str, str]:
            return {
                "compare": ctx.code(WGPU_COMPARE_FUNCTION_NAMES, cmp, "compare function"),
                "fail_op": ctx.code(WGPU_STENCIL_OPERATION_NAMES, fail, "stencil operation"),
                "depth_fail_op": ctx.code(WGPU_STENCIL_OPERATION_NAMES, dfail, "stencil operation"),
                "pass_op": ctx.code(WGPU_STENCIL_OPERATION_NAMES, pss, "stencil operation"),
            }

        depth_stencil = {
            "format": ctx.code(WGPU_TEXTURE_FORMAT_NAMES, ds_format, "texture format"),
            "depth_write_enabled": bool(depth_write),
            "depth_compare": ctx.code(WGPU_COMPARE_FUNCTION_NAMES, depth_compare, "compare function"),
            "stencil_front": face(f_cmp, f_fail, f_dfail, f_pass),
            "stencil_back": face(b_cmp, b_fail, b_dfail, b_pass),
            "stencil_read_mask": read_mask,
            "stencil_write_mask": write_mask_ds,
            "depth_bias": depth_bias,
            "depth_bias_slope_scale": _f32_from_bits(slope_bits),
            "depth_bias_clamp": _f32_from_bits(clamp_bits),
        }

    topology, strip_index_format, front_face, cull_mode, unclipped_depth = (
        KEY_PRIMITIVE_STRUCT.unpack_from(key, 152)
    )
    count, mask, alpha_to_coverage = KEY_MULTISAMPLE_STRUCT.unpack_from(key, 160)
    vertex_buffer_count, vertex_attribute_count = KEY_COUNTS_STRUCT.unpack_from(key, 172)
    if vertex_buffer_count > 8 or vertex_attribute_count > 16:
        ctx.error(
            "invalid_count",
            f"vertex_buffer_count {vertex_buffer_count} / vertex_attribute_count "
            f"{vertex_attribute_count} exceed 8 / 16",
        )
    attributes = []
    for i in range(min(vertex_attribute_count, 16)):
        attr_offset, attr_format, location, buffer_index = KEY_VERTEX_ATTRIBUTE_STRUCT.unpack_from(
            key, 244 + 8 * i
        )
        attributes.append(
            {
                "offset": attr_offset,
                "format": ctx.code(WGPU_VERTEX_FORMAT_NAMES, attr_format, "vertex format"),
                "shader_location": location,
                "buffer_index": buffer_index,
            }
        )
    vertex_buffers: list[dict[str, Any] | None] = []
    for i in range(min(vertex_buffer_count, 8)):
        stride, step_mode, attribute_count = KEY_VERTEX_BUFFER_STRUCT.unpack_from(key, 180 + 8 * i)
        if step_mode == 0:
            vertex_buffers.append(None)
            continue
        vertex_buffers.append(
            {
                "array_stride": stride,
                "step_mode": ctx.code(WGPU_VERTEX_STEP_MODE_NAMES, step_mode, "vertex step mode"),
                "attributes": [a for a in attributes if a["buffer_index"] == i],
            }
        )

    return {
        "vs_shader": f"0x{vs_shader:016x}",
        "ps_shader": f"0x{ps_shader:016x}",
        "root_signature_id": root_signature_id,
        "color_target_count": color_target_count,
        "color_targets": color_targets,
        "depth_stencil": depth_stencil,
        "primitive": {
            "topology": ctx.code(tuple(TOPOLOGY_WEBGPU_NAMES.values()), topology, "topology"),
            "strip_index_format": ctx.code(WGPU_INDEX_FORMAT_NAMES, strip_index_format, "index format"),
            "front_face": ctx.code(WGPU_FRONT_FACE_NAMES, front_face, "front face"),
            "cull_mode": ctx.code(WGPU_CULL_MODE_NAMES, cull_mode, "cull mode"),
            "unclipped_depth": bool(unclipped_depth),
        },
        "multisample": {
            "count": count,
            "mask": mask,
            "alpha_to_coverage_enabled": bool(alpha_to_coverage),
        },
        "vertex_buffer_count": vertex_buffer_count,
        "vertex_attribute_count": vertex_attribute_count,
        "vertex_buffers": vertex_buffers,
        "key_fnv1a64": f"0x{fnv1a_64(key):016x}",
    }


def _decode_v2_payload(
    *,
    opcode: GpuOpcode,
    payload: memoryview,
    offset: int,
    packet_index: int,
    strict: bool,
    diagnostics: list[TraceDiagnostic],
) -> tuple[dict[str, Any], bool]:
    """Decode payload for a v2 opcode (22..34). Returns (fields, valid)."""
    ctx = _V2Context(opcode.name, offset, packet_index, strict, diagnostics)
    payload_len = len(payload)

    def too_short(size: int) -> tuple[dict[str, Any], bool]:
        ctx.error("payload_too_short", f"expects at least {size} payload byte(s), got {payload_len}")
        return {"expected_min_bytes": size, "actual_bytes": payload_len}, False

    def fixed_fields(size: int) -> tuple[bool, dict[str, Any]]:
        can_decode, exact = ctx.fixed(payload_len, size)
        extra: dict[str, Any] = {}
        if not can_decode:
            extra = {"expected_bytes": size, "actual_bytes": payload_len}
        elif not exact:
            extra = {"trailing_bytes": payload_len - size}
        return can_decode, extra

    if opcode in (GpuOpcode.CreateBindGroupLayout, GpuOpcode.CreatePipelineLayout):
        head = CREATE_BIND_GROUP_LAYOUT_STRUCT
        if payload_len < head.size:
            return too_short(head.size)
        res_id, count, flags, _reserved, content_id = head.unpack_from(payload, 0)
        manifest_ref = bool(flags & MANIFEST_REFERENCE_FLAG)
        if manifest_ref and count:
            ctx.warn("manifest_reference_with_entries", f"ManifestReference with {count} element(s)")
        fields: dict[str, Any] = {
            "id": res_id,
            "flags": flags,
            "manifest_reference": manifest_ref,
            "content_id": f"0x{content_id:016x}",
        }
        if opcode == GpuOpcode.CreatePipelineLayout:
            n, valid, trailing = ctx.tail(payload_len, head.size, count, 4)
            fields["bind_group_layout_count"] = count
            fields["bind_group_layout_ids"] = list(
                struct.unpack_from(f"<{n}I", payload, head.size)
            )
        else:
            stride = BIND_GROUP_LAYOUT_ENTRY_STRUCT.size
            n, valid, trailing = ctx.tail(payload_len, head.size, count, stride)
            entries = []
            for i in range(n):
                (binding, visibility, ltype, subtype, view_dim, eflags, storage_format, min_size) = (
                    BIND_GROUP_LAYOUT_ENTRY_STRUCT.unpack_from(payload, head.size + i * stride)
                )
                entry: dict[str, Any] = {
                    "binding": binding,
                    "visibility": visibility,
                    "visibility_names": [n_ for bit, n_ in SHADER_STAGE_BITS if visibility & bit],
                    "type": ctx.code(BINDING_LAYOUT_TYPE_NAMES, ltype, "binding layout type"),
                    "binding_type": ctx.code(
                        BINDING_SUBTYPE_NAMES.get(ltype, ()), subtype, "binding type"
                    ),
                }
                if ltype in (2, 3):
                    entry["view_dimension"] = ctx.code(
                        WGPU_VIEW_DIMENSION_NAMES, view_dim, "view dimension"
                    )
                if ltype == 0:
                    entry["has_dynamic_offset"] = bool(eflags & 0x1)
                    entry["min_binding_size"] = min_size
                if ltype == 2:
                    entry["multisampled"] = bool(eflags & 0x2)
                if ltype == 3:
                    entry["storage_format"] = ctx.code(
                        WGPU_TEXTURE_FORMAT_NAMES, storage_format, "texture format"
                    )
                entries.append(entry)
            fields["entry_count"] = count
            fields["entries"] = entries
        if trailing:
            fields["trailing_bytes"] = trailing
        return fields, valid

    if opcode == GpuOpcode.CreateRenderPipeline2:
        can_decode, extra = fixed_fields(CREATE_RENDER_PIPELINE2_SIZE)
        if not can_decode:
            return extra, False
        res_id, vs_id, fs_id, layout_id = CREATE_RENDER_PIPELINE2_HEAD_STRUCT.unpack_from(payload, 0)
        key = bytes(payload[16:CREATE_RENDER_PIPELINE2_SIZE])
        return {
            "id": res_id,
            "vs_module_id": vs_id,
            "fs_module_id": fs_id,
            "pipeline_layout_id": layout_id,
            "key": _decode_render_pipeline_key(ctx, key),
            **extra,
        }, True

    if opcode == GpuOpcode.CreateComputePipeline2:
        can_decode, extra = fixed_fields(CREATE_COMPUTE_PIPELINE2_STRUCT.size)
        if not can_decode:
            return extra, False
        res_id, cs_id, layout_id, _r, cs_shader, root_sig, _r2 = (
            CREATE_COMPUTE_PIPELINE2_STRUCT.unpack_from(payload, 0)
        )
        return {
            "id": res_id,
            "cs_module_id": cs_id,
            "pipeline_layout_id": layout_id,
            "cs_shader": f"0x{cs_shader:016x}",
            "root_signature_id": root_sig,
            **extra,
        }, True

    if opcode == GpuOpcode.BeginRenderPass:
        can_decode, extra = fixed_fields(BEGIN_RENDER_PASS_SIZE)
        if not can_decode:
            return extra, False
        color_count, _reserved = BEGIN_RENDER_PASS_HEAD_STRUCT.unpack_from(payload, 0)
        valid = True
        if color_count > 8:
            ctx.error("invalid_count", f"color_count {color_count} > 8")
            valid = False
        colors: list[dict[str, Any] | None] = []
        for i in range(min(color_count, 8)):
            (tex, load_op, store_op, _r, view, r, g, b, a) = (
                RENDER_PASS_COLOR_ATTACHMENT_STRUCT.unpack_from(payload, 8 + 40 * i)
            )
            if tex == 0:
                colors.append(None)
                continue
            colors.append(
                {
                    "texture_id": tex,
                    "swapchain": tex == SWAPCHAIN_TEXTURE_ID,
                    "load_op": ctx.code(WGPU_LOAD_OP_NAMES, load_op, "load op"),
                    "store_op": ctx.code(WGPU_STORE_OP_NAMES, store_op, "store op"),
                    "clear_value": [r, g, b, a],
                    "view": _decode_view_desc(ctx, view),
                }
            )
        (
            ds_tex, d_load, d_store, s_load, s_store, ds_view, depth_clear, stencil_clear,
            depth_ro, stencil_ro, _r0, _r1,
        ) = RENDER_PASS_DEPTH_STENCIL_STRUCT.unpack_from(payload, 328)
        depth_stencil: dict[str, Any] | None = None
        if ds_tex != 0:
            depth_stencil = {
                "texture_id": ds_tex,
                "depth_load_op": ctx.code(WGPU_LOAD_OP_NAMES, d_load, "load op"),
                "depth_store_op": ctx.code(WGPU_STORE_OP_NAMES, d_store, "store op"),
                "stencil_load_op": ctx.code(WGPU_LOAD_OP_NAMES, s_load, "load op"),
                "stencil_store_op": ctx.code(WGPU_STORE_OP_NAMES, s_store, "store op"),
                "depth_clear_value": depth_clear,
                "stencil_clear_value": stencil_clear,
                "depth_read_only": bool(depth_ro),
                "stencil_read_only": bool(stencil_ro),
                "view": _decode_view_desc(ctx, ds_view),
            }
        return {
            "color_count": color_count,
            "color_attachments": colors,
            "depth_stencil": depth_stencil,
            **extra,
        }, valid

    if opcode == GpuOpcode.EndRenderPass:
        can_decode, extra = fixed_fields(0)
        return extra, can_decode

    simple: dict[GpuOpcode, tuple[struct.Struct, tuple[str, ...]]] = {
        GpuOpcode.SetViewport: (
            SET_VIEWPORT_STRUCT,
            ("x", "y", "width", "height", "min_depth", "max_depth"),
        ),
        GpuOpcode.SetScissorRect: (SET_SCISSOR_RECT_STRUCT, ("x", "y", "width", "height")),
        GpuOpcode.SetBlendConstant: (SET_BLEND_CONSTANT_STRUCT, ("r", "g", "b", "a")),
        GpuOpcode.SetStencilReference: (SET_STENCIL_REFERENCE_STRUCT, ("reference",)),
    }
    if opcode in simple:
        layout, names = simple[opcode]
        can_decode, extra = fixed_fields(layout.size)
        if not can_decode:
            return extra, False
        values = layout.unpack_from(payload, 0)
        for field_name, value in zip(names, values):
            if isinstance(value, float) and not math.isfinite(value):
                ctx.warn("non_finite_float", f"{field_name} is non-finite ({value})")
        return {**dict(zip(names, values)), **extra}, True

    if opcode == GpuOpcode.CreateBindGroup2:
        head = CREATE_BIND_GROUP2_STRUCT
        if payload_len < head.size:
            return too_short(head.size)
        res_id, layout_id, entry_count, _reserved = head.unpack_from(payload, 0)
        stride = BIND_GROUP_ENTRY2_STRUCT.size
        n, valid, trailing = ctx.tail(payload_len, head.size, entry_count, stride)
        entries = []
        for i in range(n):
            binding, rtype, rid, _r, e_offset, e_size, view = BIND_GROUP_ENTRY2_STRUCT.unpack_from(
                payload, head.size + i * stride
            )
            rtype_name, rtype_known = decode_bind_resource_type(rtype)
            if not rtype_known:
                ctx.warn("unknown_bind_resource_type", f"entry[{i}] has resource_type={rtype}")
            entry = {
                "binding": binding,
                "resource_type": rtype,
                "resource_type_name": rtype_name,
                "resource_id": rid,
                "null_binding": rid == 0,
                "offset": e_offset,
                "size": e_size,
            }
            if rtype == GpuBindResourceType.TextureView:
                entry["view"] = _decode_view_desc(ctx, view)
            entries.append(entry)
        fields = {
            "id": res_id,
            "layout_id": layout_id,
            "entry_count": entry_count,
            "entries": entries,
        }
        if trailing:
            fields["trailing_bytes"] = trailing
        return fields, valid

    if opcode == GpuOpcode.SetBindGroup2:
        head = SET_BIND_GROUP2_STRUCT
        if payload_len < head.size:
            return too_short(head.size)
        group_index, bind_group_id, count, _reserved = head.unpack_from(payload, 0)
        n, valid, trailing = ctx.tail(payload_len, head.size, count, 4)
        fields = {
            "group_index": group_index,
            "bind_group_id": bind_group_id,
            "dynamic_offsets": list(struct.unpack_from(f"<{n}I", payload, head.size)),
        }
        if trailing:
            fields["trailing_bytes"] = trailing
        return fields, valid

    if opcode == GpuOpcode.CreateSampler2:
        can_decode, extra = fixed_fields(CREATE_SAMPLER2_STRUCT.size)
        if not can_decode:
            return extra, False
        (res_id, au, av, aw, mag, min_f, mip, compare, aniso, lod_min, lod_max) = (
            CREATE_SAMPLER2_STRUCT.unpack_from(payload, 0)
        )
        addr_names = tuple(ADDRESS_MODE_WEBGPU_NAMES.values())
        filter_names = tuple(FILTER_MODE_WEBGPU_NAMES.values())
        return {
            "id": res_id,
            "address_u": ctx.code(addr_names, au, "address mode"),
            "address_v": ctx.code(addr_names, av, "address mode"),
            "address_w": ctx.code(addr_names, aw, "address mode"),
            "mag_filter": ctx.code(filter_names, mag, "filter mode"),
            "min_filter": ctx.code(filter_names, min_f, "filter mode"),
            "mipmap_filter": ctx.code(filter_names, mip, "filter mode"),
            "compare": ctx.code(WGPU_COMPARE_FUNCTION_NAMES, compare, "compare function"),
            "max_anisotropy": aniso,
            "lod_min_clamp": lod_min,
            "lod_max_clamp": lod_max,
            **extra,
        }, True

    if opcode == GpuOpcode.CreateTexture2:
        can_decode, extra = fixed_fields(CREATE_TEXTURE2_STRUCT.size)
        if not can_decode:
            return extra, False
        (res_id, width, height, depth_or_layers, mip_levels, sample_count, usage_flags,
         dimension, tex_format, view_format_count, _reserved, view_formats) = (
            CREATE_TEXTURE2_STRUCT.unpack_from(payload, 0)
        )
        usage_names, unknown_bits = decode_texture_usage(usage_flags)
        if unknown_bits:
            ctx.warn("unknown_texture_usage_bits", f"unrecognized usage_flags bits: 0x{unknown_bits:x}")
        if view_format_count > 8:
            ctx.warn("view_format_count", f"view_format_count {view_format_count} exceeds 8")
        dim_name, _dim_webgpu, dim_known = decode_texture_dimension(dimension)
        if not dim_known:
            ctx.warn("unknown_texture_dimension", f"unknown dimension={dimension}")
        return {
            "id": res_id,
            "width": width,
            "height": height,
            "depth_or_layers": depth_or_layers,
            "mip_levels": mip_levels,
            "sample_count": sample_count,
            "usage_flags": usage_flags,
            "usage_names": usage_names,
            "dimension": dimension,
            "dimension_name": dim_name,
            "format": ctx.code(WGPU_TEXTURE_FORMAT_NAMES, tex_format, "texture format"),
            "view_formats": [
                ctx.code(WGPU_TEXTURE_FORMAT_NAMES, f, "texture format")
                for f in view_formats[: min(view_format_count, 8)]
            ],
            **extra,
        }, True

    raise AssertionError(f"Unhandled v2 opcode: {opcode}")


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
        GpuOpcode.BeginRenderPass,
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
        begin_render_passes = 0
        end_render_passes = 0
        color_load_ops: Counter[str] = Counter()
        color_store_ops: Counter[str] = Counter()
        depth_load_ops: Counter[str] = Counter()
        depth_store_ops: Counter[str] = Counter()
        bind_group_layouts_created = 0
        pipeline_layouts_created = 0
        render_pipelines2_created = 0
        compute_pipelines2_created = 0
        viewport_changes = 0
        scissor_changes = 0
        blend_constant_changes = 0
        stencil_ref_changes = 0
        malformed_v2_packets = 0
        total_bytes = 0
        active_pass: str | None = None
        for pkt in current:
            histogram[pkt.name] += 1
            total_bytes += pkt.total_bytes
            if _is_v2_opcode(pkt.opcode) and not pkt.valid:
                malformed_v2_packets += 1
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
            elif pkt.opcode in (GpuOpcode.CreateBindGroup, GpuOpcode.CreateBindGroup2):
                bind_groups_created += 1
            elif pkt.opcode == GpuOpcode.BeginRenderPass:
                begin_render_passes += 1
                for color in pkt.fields.get("color_attachments") or ():
                    if color is None:
                        continue
                    color_load_ops[str(color["load_op"])] += 1
                    color_store_ops[str(color["store_op"])] += 1
                ds = pkt.fields.get("depth_stencil")
                if ds is not None:
                    depth_load_ops[str(ds["depth_load_op"])] += 1
                    depth_store_ops[str(ds["depth_store_op"])] += 1
            elif pkt.opcode == GpuOpcode.EndRenderPass:
                end_render_passes += 1
            elif pkt.opcode == GpuOpcode.CreateBindGroupLayout:
                bind_group_layouts_created += 1
            elif pkt.opcode == GpuOpcode.CreatePipelineLayout:
                pipeline_layouts_created += 1
            elif pkt.opcode == GpuOpcode.CreateRenderPipeline2:
                render_pipelines2_created += 1
            elif pkt.opcode == GpuOpcode.CreateComputePipeline2:
                compute_pipelines2_created += 1
            elif pkt.opcode == GpuOpcode.SetViewport:
                viewport_changes += 1
            elif pkt.opcode == GpuOpcode.SetScissorRect:
                scissor_changes += 1
            elif pkt.opcode == GpuOpcode.SetBlendConstant:
                blend_constant_changes += 1
            elif pkt.opcode == GpuOpcode.SetStencilReference:
                stencil_ref_changes += 1

            if pkt.opcode in compute_pass_opcodes:
                if active_pass != "compute":
                    compute_passes += 1
                    active_pass = "compute"
            elif pkt.opcode in render_pass_opcodes:
                active_pass = "render"
            elif pkt.opcode == GpuOpcode.EndRenderPass:
                active_pass = None
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
                begin_render_passes=begin_render_passes,
                end_render_passes=end_render_passes,
                color_load_ops=dict(color_load_ops),
                color_store_ops=dict(color_store_ops),
                depth_load_ops=dict(depth_load_ops),
                depth_store_ops=dict(depth_store_ops),
                bind_group_layouts_created=bind_group_layouts_created,
                pipeline_layouts_created=pipeline_layouts_created,
                render_pipelines2_created=render_pipelines2_created,
                compute_pipelines2_created=compute_pipelines2_created,
                viewport_changes=viewport_changes,
                scissor_changes=scissor_changes,
                blend_constant_changes=blend_constant_changes,
                stencil_ref_changes=stencil_ref_changes,
                malformed_v2_packets=malformed_v2_packets,
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
    shader_hash_refs: list[tuple[int, str]] = []

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
            if (
                known_opcode == GpuOpcode.CreateShaderModule
                and valid
                and fields.get("utf8_valid")
                and fields.get("code_len", 0) > 0
            ):
                code_end = CREATE_SHADER_MODULE_STRUCT.size + int(fields["code_len"])
                code_text = bytes(payload[CREATE_SHADER_MODULE_STRUCT.size : code_end]).decode(
                    "utf-8", errors="ignore"
                )
                hash_ref = _parse_hash_only_shader_payload(code_text)
                if hash_ref is not None:
                    shader_hash_refs.append((packet_index, hash_ref))
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
        _shader_hash_refs=tuple(shader_hash_refs),
    )


# ---------------------------------------------------------------------------
# Offline diagnostic audits (--shader-summary & --binding-audit)
# ---------------------------------------------------------------------------

_ZERO_HASH_HEX = "0x0000000000000000"
_PREFIXED_HASH_RE = re.compile(r"^(?:hash|wgpk):(?:0x)?([0-9a-fA-F]{1,16})$")
_HEX0X_RE = re.compile(r"^0x([0-9a-fA-F]{1,16})$")
_BARE64_HEX_RE = re.compile(r"^([0-9a-fA-F]{16})$")

_DRAW_PACKET_NAMES = frozenset(
    {"Draw", "DrawIndexed", "DrawIndirect", "DrawIndexedIndirect"}
)
_DISPATCH_PACKET_NAMES = frozenset({"DispatchCompute", "DispatchComputeIndirect"})

_LAYOUT_TYPE_TO_EXPECTED_RESOURCE: dict[str, tuple[GpuBindResourceType, str]] = {
    "buffer": (GpuBindResourceType.Buffer, "Buffer"),
    "sampler": (GpuBindResourceType.Sampler, "Sampler"),
    "texture": (GpuBindResourceType.TextureView, "TextureView"),
    "storageTexture": (GpuBindResourceType.TextureView, "TextureView"),
}


def _parse_hash_only_shader_payload(text: str) -> str | None:
    """Parse a hash-only CreateShaderModule payload (`0x...`, `hash:...`, `wgpk:...`, or 16-hex)."""
    trimmed = text.strip()
    if not trimmed:
        return None
    for pattern in (_PREFIXED_HASH_RE, _HEX0X_RE, _BARE64_HEX_RE):
        match = pattern.match(trimmed)
        if match:
            return f"0x{int(match.group(1), 16):016x}"
    return None


class _ShaderAuditRecord:
    __slots__ = ("shader_hash", "shader_ids", "stage", "pipeline_ids", "draw_count", "dispatch_count")

    def __init__(self, shader_hash: str | None, stage: str) -> None:
        self.shader_hash = shader_hash
        self.shader_ids: set[int] = set()
        self.stage = stage
        self.pipeline_ids: set[int] = set()
        self.draw_count = 0
        self.dispatch_count = 0


def summarize_shaders(trace: DecodedTrace) -> dict[str, Any]:
    """Summarize shader modules/hashes, pipelines, draws, and dispatches in a DecodedTrace."""
    hash_refs: dict[int, str] = dict(trace._shader_hash_refs)

    # Pre-pass: correlate live CreateShaderModule packet indices with non-zero 64-bit
    # shader identity hashes carried in CreateRenderPipeline2 / CreateComputePipeline2 keys.
    active_mod_pkt: dict[int, int] = {}
    mod_pkt_key_hash: dict[int, str] = {}
    for pkt in trace.packets:
        if not pkt.valid:
            continue
        op = pkt.opcode
        f = pkt.fields
        if op == GpuOpcode.CreateShaderModule:
            active_mod_pkt[int(f["id"])] = pkt.index
        elif op == GpuOpcode.DestroyResource:
            active_mod_pkt.pop(int(f["id"]), None)
        elif op == GpuOpcode.CreateRenderPipeline2:
            key = f["key"]
            vs_id = int(f["vs_module_id"])
            fs_id = int(f["fs_module_id"])
            vs_hash = str(key["vs_shader"])
            ps_hash = str(key["ps_shader"])
            if vs_id != 0 and vs_hash != _ZERO_HASH_HEX and vs_id in active_mod_pkt:
                mod_pkt_key_hash.setdefault(active_mod_pkt[vs_id], vs_hash)
            if fs_id != 0 and ps_hash != _ZERO_HASH_HEX and fs_id in active_mod_pkt:
                mod_pkt_key_hash.setdefault(active_mod_pkt[fs_id], ps_hash)
        elif op == GpuOpcode.CreateComputePipeline2:
            cs_id = int(f["cs_module_id"])
            cs_hash = str(f["cs_shader"])
            if cs_id != 0 and cs_hash != _ZERO_HASH_HEX and cs_id in active_mod_pkt:
                mod_pkt_key_hash.setdefault(active_mod_pkt[cs_id], cs_hash)

    records: dict[tuple[Any, ...], _ShaderAuditRecord] = {}
    active_modules: dict[int, tuple[Any, ...]] = {}
    render_pipelines: dict[int, set[tuple[Any, ...]]] = {}
    compute_pipelines: dict[int, set[tuple[Any, ...]]] = {}
    active_render_pipeline_id: int | None = None
    active_compute_pipeline_id: int | None = None
    total_draws = 0
    total_dispatches = 0
    unbound_draws = 0
    unbound_dispatches = 0

    def _get_or_create(
        skey: tuple[Any, ...],
        *,
        shader_hash: str | None,
        stage: str,
        shader_id: int | None = None,
    ) -> _ShaderAuditRecord:
        rec = records.get(skey)
        if rec is None:
            rec = _ShaderAuditRecord(shader_hash=shader_hash, stage=stage)
            records[skey] = rec
        elif (rec.stage == "Unknown" or rec.stage.startswith("Unknown(")) and (
            stage != "Unknown" and not stage.startswith("Unknown(")
        ):
            rec.stage = stage
        if shader_id is not None and shader_id != 0:
            rec.shader_ids.add(shader_id)
        return rec

    def _resolve_stage_shader(
        *,
        module_id: int,
        key_hash: str | None,
        stage: str,
    ) -> tuple[Any, ...] | None:
        if key_hash is not None and key_hash != _ZERO_HASH_HEX:
            skey: tuple[Any, ...] = ("hash", key_hash)
            _get_or_create(skey, shader_hash=key_hash, stage=stage, shader_id=module_id or None)
            return skey
        if module_id == 0:
            return None
        existing = active_modules.get(module_id)
        if existing is not None:
            rec = records[existing]
            if (rec.stage == "Unknown" or rec.stage.startswith("Unknown(")) and stage != "Unknown":
                rec.stage = stage
            return existing
        skey = ("id", module_id, stage)
        _get_or_create(skey, shader_hash=None, stage=stage, shader_id=module_id)
        return skey

    for pkt in trace.packets:
        if not pkt.valid:
            continue
        op = pkt.opcode
        f = pkt.fields

        if op == GpuOpcode.CreateShaderModule:
            mod_id = int(f["id"])
            stage_name = str(f["stage_name"])
            resolved_hash: str | None = mod_pkt_key_hash.get(pkt.index)
            if resolved_hash is None:
                resolved_hash = hash_refs.get(pkt.index)
            if resolved_hash is None and int(f.get("code_len", 0)) > 0:
                resolved_hash = str(f["wgsl_hash_fnv1a64"])
            if resolved_hash is not None:
                skey = ("hash", resolved_hash)
            else:
                skey = ("id", mod_id, stage_name)
            _get_or_create(skey, shader_hash=resolved_hash, stage=stage_name, shader_id=mod_id)
            active_modules[mod_id] = skey

        elif op == GpuOpcode.CreateRenderPipeline:
            pid = int(f["id"])
            attached: set[tuple[Any, ...]] = set()
            vs_key = _resolve_stage_shader(
                module_id=int(f["vs_shader_id"]), key_hash=None, stage="Vertex"
            )
            fs_key = _resolve_stage_shader(
                module_id=int(f["fs_shader_id"]), key_hash=None, stage="Fragment"
            )
            for skey in (vs_key, fs_key):
                if skey is not None:
                    attached.add(skey)
                    records[skey].pipeline_ids.add(pid)
            render_pipelines[pid] = attached

        elif op == GpuOpcode.CreateRenderPipeline2:
            pid = int(f["id"])
            key = f["key"]
            attached = set()
            vs_key = _resolve_stage_shader(
                module_id=int(f["vs_module_id"]),
                key_hash=str(key["vs_shader"]),
                stage="Vertex",
            )
            fs_key = _resolve_stage_shader(
                module_id=int(f["fs_module_id"]),
                key_hash=str(key["ps_shader"]),
                stage="Fragment",
            )
            for skey in (vs_key, fs_key):
                if skey is not None:
                    attached.add(skey)
                    records[skey].pipeline_ids.add(pid)
            render_pipelines[pid] = attached

        elif op == GpuOpcode.CreateComputePipeline:
            pid = int(f["id"])
            attached = set()
            cs_key = _resolve_stage_shader(
                module_id=int(f["cs_shader_id"]), key_hash=None, stage="Compute"
            )
            if cs_key is not None:
                attached.add(cs_key)
                records[cs_key].pipeline_ids.add(pid)
            compute_pipelines[pid] = attached

        elif op == GpuOpcode.CreateComputePipeline2:
            pid = int(f["id"])
            attached = set()
            cs_key = _resolve_stage_shader(
                module_id=int(f["cs_module_id"]),
                key_hash=str(f["cs_shader"]),
                stage="Compute",
            )
            if cs_key is not None:
                attached.add(cs_key)
                records[cs_key].pipeline_ids.add(pid)
            compute_pipelines[pid] = attached

        elif op == GpuOpcode.DestroyResource:
            rid = int(f["id"])
            active_modules.pop(rid, None)
            render_pipelines.pop(rid, None)
            compute_pipelines.pop(rid, None)
            if active_render_pipeline_id == rid:
                active_render_pipeline_id = None
            if active_compute_pipeline_id == rid:
                active_compute_pipeline_id = None

        elif op in (GpuOpcode.BeginFrame, GpuOpcode.Present):
            active_render_pipeline_id = None
            active_compute_pipeline_id = None

        elif op in (GpuOpcode.BeginRenderPass, GpuOpcode.EndRenderPass):
            active_render_pipeline_id = None

        elif op == GpuOpcode.SetPipeline:
            active_render_pipeline_id = int(f["pipeline_id"])

        elif op == GpuOpcode.SetComputePipeline:
            active_compute_pipeline_id = int(f["pipeline_id"])
            active_render_pipeline_id = None

        elif pkt.name in _DRAW_PACKET_NAMES:
            total_draws += 1
            attached_set = (
                render_pipelines.get(active_render_pipeline_id)
                if active_render_pipeline_id is not None
                else None
            )
            if attached_set:
                for skey in attached_set:
                    records[skey].draw_count += 1
            else:
                unbound_draws += 1

        elif pkt.name in _DISPATCH_PACKET_NAMES:
            total_dispatches += 1
            attached_set = (
                compute_pipelines.get(active_compute_pipeline_id)
                if active_compute_pipeline_id is not None
                else None
            )
            if attached_set:
                for skey in attached_set:
                    records[skey].dispatch_count += 1
            else:
                unbound_dispatches += 1

    rows: list[dict[str, Any]] = []
    for rec in records.values():
        sorted_ids = sorted(rec.shader_ids)
        sorted_pipelines = sorted(rec.pipeline_ids)
        total_exec = rec.draw_count + rec.dispatch_count
        rows.append(
            {
                "shader_hash": rec.shader_hash,
                "shader_id": sorted_ids[0] if sorted_ids else None,
                "shader_ids": sorted_ids,
                "stage": rec.stage,
                "pipeline_count": len(sorted_pipelines),
                "pipeline_ids": sorted_pipelines,
                "draw_count": rec.draw_count,
                "dispatch_count": rec.dispatch_count,
                "total_executions": total_exec,
                "active": total_exec > 0,
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row["total_executions"]),
            -int(row["pipeline_count"]),
            str(row["shader_hash"] or ""),
            int(row["shader_id"]) if row["shader_id"] is not None else -1,
        )
    )

    active_count = sum(1 for row in rows if row["active"])
    active_hashes = [
        str(row["shader_hash"])
        for row in rows
        if row["active"] and row["shader_hash"] is not None
    ]
    return {
        "total_shaders": len(rows),
        "active_shaders": active_count,
        "unused_shaders": len(rows) - active_count,
        "total_draws": total_draws,
        "total_dispatches": total_dispatches,
        "unbound_draws": unbound_draws,
        "unbound_dispatches": unbound_dispatches,
        "active_shader_hashes": active_hashes,
        "shaders": rows,
    }


@dataclass(frozen=True)
class _BindGroupLayoutRecord:
    layout_id: int
    manifest_reference: bool
    content_id: str
    entries: dict[int, dict[str, Any]]


@dataclass(frozen=True)
class _ResolvedAttachment:
    texture_id: int
    swapchain: bool
    format: str | None
    format_source: str  # "view" | "texture" | "swapchain" | "unknown"


@dataclass(frozen=True)
class _ActiveRenderPass:
    packet_index: int
    color_attachments: tuple[_ResolvedAttachment | None, ...]
    depth_stencil: _ResolvedAttachment | None


def audit_bindings(trace: DecodedTrace) -> dict[str, Any]:
    """Cross-check CreateBindGroupLayout vs CreateBindGroup2 and BeginRenderPass vs CreateRenderPipeline2."""
    layouts: dict[int, _BindGroupLayoutRecord] = {}
    bind_groups: dict[int, dict[str, Any]] = {}
    textures: dict[int, str] = {}
    render_pipelines2: dict[int, dict[str, Any]] = {}

    bind_group_layouts_seen = 0
    manifest_reference_layouts = 0
    bind_groups_audited = 0
    bind_groups_skipped_manifest_layout = 0
    render_passes_seen = 0
    draws_audited = 0

    bind_group_mismatches: list[dict[str, Any]] = []
    bind_group_mismatch_counts: Counter[str] = Counter()
    render_pass_mismatches: list[dict[str, Any]] = []
    render_pass_mismatch_counts: Counter[str] = Counter()
    swapchain_target_formats: Counter[str] = Counter()
    color_target_formats: Counter[str] = Counter()

    active_render_pass: _ActiveRenderPass | None = None
    active_render_pipeline_id: int | None = None

    def _record_bg_mismatch(entry: dict[str, Any]) -> None:
        bind_group_mismatches.append(entry)
        bind_group_mismatch_counts[str(entry["kind"])] += 1

    def _record_rp_mismatch(entry: dict[str, Any]) -> None:
        render_pass_mismatches.append(entry)
        render_pass_mismatch_counts[str(entry["kind"])] += 1

    for pkt in trace.packets:
        if not pkt.valid:
            continue
        op = pkt.opcode
        f = pkt.fields

        if op == GpuOpcode.CreateBindGroupLayout:
            bind_group_layouts_seen += 1
            layout_id = int(f["id"])
            manifest_ref = bool(f["manifest_reference"])
            if manifest_ref:
                manifest_reference_layouts += 1
            entries_map = {int(e["binding"]): e for e in f.get("entries", ())}
            layouts[layout_id] = _BindGroupLayoutRecord(
                layout_id=layout_id,
                manifest_reference=manifest_ref,
                content_id=str(f["content_id"]),
                entries=entries_map,
            )

        elif op == GpuOpcode.CreateBindGroup2:
            bind_groups_audited += 1
            bg_id = int(f["id"])
            layout_id = int(f["layout_id"])
            bg_entries = list(f.get("entries", ()))
            bind_groups[bg_id] = {"layout_id": layout_id, "entries": bg_entries}

            layout = layouts.get(layout_id)
            if layout is None:
                _record_bg_mismatch(
                    {
                        "kind": "unknown_layout",
                        "packet_index": pkt.index,
                        "bind_group_id": bg_id,
                        "layout_id": layout_id,
                        "binding": None,
                        "expected_layout_type": None,
                        "expected_binding_type": None,
                        "expected_resource_type": None,
                        "actual_resource_type": None,
                        "resource_id": None,
                        "null_binding": None,
                        "message": (
                            f"CreateBindGroup2(id={bg_id}) references undeclared layout_id={layout_id}"
                        ),
                    }
                )
                continue

            if layout.manifest_reference:
                bind_groups_skipped_manifest_layout += 1
                continue

            seen_bindings: set[int] = set()
            for entry in bg_entries:
                binding = int(entry["binding"])
                rtype = int(entry["resource_type"])
                rtype_name = str(entry["resource_type_name"])
                resource_id = int(entry["resource_id"])
                null_binding = bool(entry["null_binding"])

                if binding in seen_bindings:
                    _record_bg_mismatch(
                        {
                            "kind": "duplicate_binding",
                            "packet_index": pkt.index,
                            "bind_group_id": bg_id,
                            "layout_id": layout_id,
                            "binding": binding,
                            "expected_layout_type": None,
                            "expected_binding_type": None,
                            "expected_resource_type": None,
                            "actual_resource_type": rtype_name,
                            "resource_id": resource_id,
                            "null_binding": null_binding,
                            "message": (
                                f"CreateBindGroup2(id={bg_id}, layout={layout_id}) duplicates "
                                f"@binding({binding})"
                            ),
                        }
                    )
                    continue
                seen_bindings.add(binding)

                layout_entry = layout.entries.get(binding)
                if layout_entry is None:
                    _record_bg_mismatch(
                        {
                            "kind": "undeclared_binding",
                            "packet_index": pkt.index,
                            "bind_group_id": bg_id,
                            "layout_id": layout_id,
                            "binding": binding,
                            "expected_layout_type": None,
                            "expected_binding_type": None,
                            "expected_resource_type": None,
                            "actual_resource_type": rtype_name,
                            "resource_id": resource_id,
                            "null_binding": null_binding,
                            "message": (
                                f"CreateBindGroup2(id={bg_id}, layout={layout_id}) binds "
                                f"@binding({binding}) as {rtype_name}(#{resource_id}) which is "
                                f"missing from declared layout {layout_id}"
                            ),
                        }
                    )
                    continue

                expected_layout_type = str(layout_entry["type"])
                expected_binding_type = str(layout_entry["binding_type"])
                expected_pair = _LAYOUT_TYPE_TO_EXPECTED_RESOURCE.get(expected_layout_type)
                if expected_pair is not None and rtype != int(expected_pair[0]):
                    expected_res_name = expected_pair[1]
                    _record_bg_mismatch(
                        {
                            "kind": "resource_kind_mismatch",
                            "packet_index": pkt.index,
                            "bind_group_id": bg_id,
                            "layout_id": layout_id,
                            "binding": binding,
                            "expected_layout_type": expected_layout_type,
                            "expected_binding_type": expected_binding_type,
                            "expected_resource_type": expected_res_name,
                            "actual_resource_type": rtype_name,
                            "resource_id": resource_id,
                            "null_binding": null_binding,
                            "message": (
                                f"CreateBindGroup2(id={bg_id}, layout={layout_id}) @binding({binding}) "
                                f"expects {expected_layout_type}({expected_binding_type}) "
                                f"[{expected_res_name}], got {rtype_name}(#{resource_id})"
                            ),
                        }
                    )

            for binding in sorted(layout.entries):
                if binding not in seen_bindings:
                    layout_entry = layout.entries[binding]
                    expected_layout_type = str(layout_entry["type"])
                    expected_binding_type = str(layout_entry["binding_type"])
                    expected_pair = _LAYOUT_TYPE_TO_EXPECTED_RESOURCE.get(expected_layout_type)
                    expected_res_name = expected_pair[1] if expected_pair is not None else None
                    _record_bg_mismatch(
                        {
                            "kind": "missing_binding",
                            "packet_index": pkt.index,
                            "bind_group_id": bg_id,
                            "layout_id": layout_id,
                            "binding": binding,
                            "expected_layout_type": expected_layout_type,
                            "expected_binding_type": expected_binding_type,
                            "expected_resource_type": expected_res_name,
                            "actual_resource_type": None,
                            "resource_id": None,
                            "null_binding": None,
                            "message": (
                                f"CreateBindGroup2(id={bg_id}, layout={layout_id}) omits "
                                f"@binding({binding}) declared as "
                                f"{expected_layout_type}({expected_binding_type})"
                            ),
                        }
                    )

        elif op == GpuOpcode.CreateTexture:
            fmt = f.get("format_webgpu") or f.get("format_name")
            if fmt:
                textures[int(f["id"])] = str(fmt)

        elif op == GpuOpcode.CreateTexture2:
            fmt = f.get("format")
            if fmt and fmt != "undefined":
                textures[int(f["id"])] = str(fmt)

        elif op == GpuOpcode.CreateRenderPipeline2:
            key = f["key"]
            render_pipelines2[int(f["id"])] = {
                "color_targets": list(key["color_targets"]),
                "depth_stencil": key["depth_stencil"],
            }

        elif op == GpuOpcode.DestroyResource:
            rid = int(f["id"])
            layouts.pop(rid, None)
            bind_groups.pop(rid, None)
            textures.pop(rid, None)
            render_pipelines2.pop(rid, None)
            if active_render_pipeline_id == rid:
                active_render_pipeline_id = None

        elif op in (GpuOpcode.BeginFrame, GpuOpcode.Present):
            active_render_pass = None
            active_render_pipeline_id = None

        elif op == GpuOpcode.BeginRenderPass:
            render_passes_seen += 1
            active_render_pipeline_id = None
            resolved_colors: list[_ResolvedAttachment | None] = []
            for color in f.get("color_attachments", ()):
                if color is None:
                    resolved_colors.append(None)
                    continue
                tex_id = int(color["texture_id"])
                is_swapchain = bool(color["swapchain"])
                view_fmt = str(color["view"]["format"])
                if view_fmt and view_fmt != "undefined":
                    fmt: str | None = view_fmt
                    fmt_src = "view"
                elif is_swapchain:
                    fmt = None
                    fmt_src = "swapchain"
                elif tex_id in textures:
                    fmt = textures[tex_id]
                    fmt_src = "texture"
                else:
                    fmt = None
                    fmt_src = "unknown"
                resolved_colors.append(
                    _ResolvedAttachment(
                        texture_id=tex_id,
                        swapchain=is_swapchain,
                        format=fmt,
                        format_source=fmt_src,
                    )
                )

            resolved_ds: _ResolvedAttachment | None = None
            ds = f.get("depth_stencil")
            if ds is not None:
                ds_tex_id = int(ds["texture_id"])
                ds_view_fmt = str(ds["view"]["format"])
                if ds_view_fmt and ds_view_fmt != "undefined":
                    ds_fmt: str | None = ds_view_fmt
                    ds_src = "view"
                elif ds_tex_id in textures:
                    ds_fmt = textures[ds_tex_id]
                    ds_src = "texture"
                else:
                    ds_fmt = None
                    ds_src = "unknown"
                resolved_ds = _ResolvedAttachment(
                    texture_id=ds_tex_id,
                    swapchain=False,
                    format=ds_fmt,
                    format_source=ds_src,
                )

            active_render_pass = _ActiveRenderPass(
                packet_index=pkt.index,
                color_attachments=tuple(resolved_colors),
                depth_stencil=resolved_ds,
            )

        elif op == GpuOpcode.EndRenderPass:
            active_render_pass = None
            active_render_pipeline_id = None

        elif op == GpuOpcode.SetPipeline:
            active_render_pipeline_id = int(f["pipeline_id"])

        elif op == GpuOpcode.SetComputePipeline or pkt.name in _DISPATCH_PACKET_NAMES:
            active_render_pass = None
            active_render_pipeline_id = None

        elif pkt.name in _DRAW_PACKET_NAMES:
            if (
                active_render_pass is None
                or active_render_pipeline_id is None
                or active_render_pipeline_id not in render_pipelines2
            ):
                continue
            draws_audited += 1
            pipe = render_pipelines2[active_render_pipeline_id]
            pass_colors = active_render_pass.color_attachments
            pipe_colors = pipe["color_targets"]
            max_slots = max(len(pass_colors), len(pipe_colors))
            for slot in range(max_slots):
                pass_att = pass_colors[slot] if slot < len(pass_colors) else None
                pipe_tgt = pipe_colors[slot] if slot < len(pipe_colors) else None
                if pass_att is None and pipe_tgt is None:
                    continue
                if pass_att is None and pipe_tgt is not None:
                    pipe_fmt = str(pipe_tgt["format"])
                    color_target_formats[pipe_fmt] += 1
                    _record_rp_mismatch(
                        {
                            "kind": "missing_render_pass_attachment",
                            "packet_index": pkt.index,
                            "render_pass_packet_index": active_render_pass.packet_index,
                            "pipeline_id": active_render_pipeline_id,
                            "slot": slot,
                            "texture_id": None,
                            "swapchain": False,
                            "render_pass_format": None,
                            "pipeline_format": pipe_fmt,
                            "message": (
                                f"Draw #{pkt.index} (pipeline={active_render_pipeline_id}) expects "
                                f"color target[{slot}] format '{pipe_fmt}', but render pass "
                                f"#{active_render_pass.packet_index} has no attachment at slot {slot}"
                            ),
                        }
                    )
                    continue
                if pass_att is not None and pipe_tgt is None:
                    att_label = "swapchain" if pass_att.swapchain else f"tex#{pass_att.texture_id}"
                    _record_rp_mismatch(
                        {
                            "kind": "missing_pipeline_color_target",
                            "packet_index": pkt.index,
                            "render_pass_packet_index": active_render_pass.packet_index,
                            "pipeline_id": active_render_pipeline_id,
                            "slot": slot,
                            "texture_id": pass_att.texture_id,
                            "swapchain": pass_att.swapchain,
                            "render_pass_format": pass_att.format,
                            "pipeline_format": None,
                            "message": (
                                f"Draw #{pkt.index} (pipeline={active_render_pipeline_id}) has null "
                                f"color target[{slot}], but render pass "
                                f"#{active_render_pass.packet_index} attaches {att_label} "
                                f"(format={pass_att.format or 'unspecified'})"
                            ),
                        }
                    )
                    continue

                assert pass_att is not None and pipe_tgt is not None
                pipe_fmt = str(pipe_tgt["format"])
                color_target_formats[pipe_fmt] += 1
                if pass_att.swapchain:
                    swapchain_target_formats[pipe_fmt] += 1
                if pass_att.format is not None and pass_att.format != pipe_fmt:
                    att_label = "swapchain" if pass_att.swapchain else f"tex#{pass_att.texture_id}"
                    _record_rp_mismatch(
                        {
                            "kind": "color_format_mismatch",
                            "packet_index": pkt.index,
                            "render_pass_packet_index": active_render_pass.packet_index,
                            "pipeline_id": active_render_pipeline_id,
                            "slot": slot,
                            "texture_id": pass_att.texture_id,
                            "swapchain": pass_att.swapchain,
                            "render_pass_format": pass_att.format,
                            "pipeline_format": pipe_fmt,
                            "message": (
                                f"Draw #{pkt.index} (pipeline={active_render_pipeline_id}) color "
                                f"target[{slot}] format '{pipe_fmt}' mismatches render pass "
                                f"#{active_render_pass.packet_index} {att_label} format "
                                f"'{pass_att.format}'"
                            ),
                        }
                    )

            pass_ds = active_render_pass.depth_stencil
            pipe_ds = pipe["depth_stencil"]
            if pass_ds is not None and pipe_ds is None:
                _record_rp_mismatch(
                    {
                        "kind": "depth_stencil_presence_mismatch",
                        "packet_index": pkt.index,
                        "render_pass_packet_index": active_render_pass.packet_index,
                        "pipeline_id": active_render_pipeline_id,
                        "slot": None,
                        "texture_id": pass_ds.texture_id,
                        "swapchain": False,
                        "render_pass_format": pass_ds.format,
                        "pipeline_format": None,
                        "message": (
                            f"Draw #{pkt.index} (pipeline={active_render_pipeline_id}) has no "
                            f"depthStencil state, but render pass #{active_render_pass.packet_index} "
                            f"attaches depth-stencil tex#{pass_ds.texture_id} "
                            f"(format={pass_ds.format or 'unspecified'})"
                        ),
                    }
                )
            elif pass_ds is None and pipe_ds is not None:
                pipe_ds_fmt = str(pipe_ds["format"])
                _record_rp_mismatch(
                    {
                        "kind": "depth_stencil_presence_mismatch",
                        "packet_index": pkt.index,
                        "render_pass_packet_index": active_render_pass.packet_index,
                        "pipeline_id": active_render_pipeline_id,
                        "slot": None,
                        "texture_id": None,
                        "swapchain": False,
                        "render_pass_format": None,
                        "pipeline_format": pipe_ds_fmt,
                        "message": (
                            f"Draw #{pkt.index} (pipeline={active_render_pipeline_id}) expects "
                            f"depthStencil format '{pipe_ds_fmt}', but render pass "
                            f"#{active_render_pass.packet_index} has no depth-stencil attachment"
                        ),
                    }
                )
            elif pass_ds is not None and pipe_ds is not None:
                pipe_ds_fmt = str(pipe_ds["format"])
                if pass_ds.format is not None and pass_ds.format != pipe_ds_fmt:
                    _record_rp_mismatch(
                        {
                            "kind": "depth_stencil_format_mismatch",
                            "packet_index": pkt.index,
                            "render_pass_packet_index": active_render_pass.packet_index,
                            "pipeline_id": active_render_pipeline_id,
                            "slot": None,
                            "texture_id": pass_ds.texture_id,
                            "swapchain": False,
                            "render_pass_format": pass_ds.format,
                            "pipeline_format": pipe_ds_fmt,
                            "message": (
                                f"Draw #{pkt.index} (pipeline={active_render_pipeline_id}) "
                                f"depthStencil format '{pipe_ds_fmt}' mismatches render pass "
                                f"#{active_render_pass.packet_index} tex#{pass_ds.texture_id} "
                                f"format '{pass_ds.format}'"
                            ),
                        }
                    )

    total_mismatches = len(bind_group_mismatches) + len(render_pass_mismatches)
    return {
        "ok": total_mismatches == 0,
        "total_mismatches": total_mismatches,
        "bind_group_layouts_seen": bind_group_layouts_seen,
        "manifest_reference_layouts": manifest_reference_layouts,
        "bind_groups_audited": bind_groups_audited,
        "bind_groups_skipped_manifest_layout": bind_groups_skipped_manifest_layout,
        "bind_group_mismatch_count": len(bind_group_mismatches),
        "bind_group_mismatch_counts": dict(sorted(bind_group_mismatch_counts.items())),
        "bind_group_mismatches": bind_group_mismatches,
        "render_passes_seen": render_passes_seen,
        "draws_audited": draws_audited,
        "render_pass_mismatch_count": len(render_pass_mismatches),
        "render_pass_mismatch_counts": dict(sorted(render_pass_mismatch_counts.items())),
        "render_pass_mismatches": render_pass_mismatches,
        "swapchain_target_formats": dict(sorted(swapchain_target_formats.items())),
        "swapchain_format_conflict": len(swapchain_target_formats) > 1,
        "color_target_formats": dict(sorted(color_target_formats.items())),
    }


def format_shader_summary_text(summary: dict[str, Any]) -> str:
    """Format a summarize_shaders() dictionary as a human-readable table."""
    lines: list[str] = [
        "Shader Summary:",
        (
            f"  Total: {summary['total_shaders']} | Active: {summary['active_shaders']} | "
            f"Unused: {summary['unused_shaders']} | Draws: {summary['total_draws']} | "
            f"Dispatches: {summary['total_dispatches']}"
        ),
    ]
    rows = summary.get("shaders") or []
    if not rows:
        lines.append("  (none)")
        return "\n".join(lines)

    lines.append(
        f"  {'Hash':<18s}  {'ID':<8s}  {'Stage':<10s}  "
        f"{'Pipelines':>9s}  {'Draws':>7s}  {'Dispatches':>10s}  {'Total':>7s}"
    )
    lines.append(
        f"  {'-' * 18}  {'-' * 8}  {'-' * 10}  "
        f"{'-' * 9}  {'-' * 7}  {'-' * 10}  {'-' * 7}"
    )
    for row in rows:
        hash_str = str(row["shader_hash"]) if row["shader_hash"] is not None else "(unresolved)"
        ids = row.get("shader_ids") or []
        id_str = ",".join(str(i) for i in ids) if ids else "-"
        lines.append(
            f"  {hash_str:<18s}  {id_str:<8s}  {str(row['stage']):<10s}  "
            f"{int(row['pipeline_count']):9d}  {int(row['draw_count']):7d}  "
            f"{int(row['dispatch_count']):10d}  {int(row['total_executions']):7d}"
        )
    return "\n".join(lines)


def format_binding_audit_text(audit: dict[str, Any]) -> str:
    """Format an audit_bindings() dictionary as a human-readable report."""
    total = int(audit["total_mismatches"])
    status = "CLEAN (0 mismatches)" if audit["ok"] else f"ISSUES FOUND ({total} mismatch(es))"
    swapchain_hist = _format_op_histogram(audit.get("swapchain_target_formats"))
    if audit.get("swapchain_format_conflict"):
        swapchain_hist += " [CONFLICT]"

    lines: list[str] = [
        "Binding Audit:",
        f"  Status: {status}",
        (
            f"  Layouts: {audit['bind_group_layouts_seen']} "
            f"(manifest-ref={audit['manifest_reference_layouts']}) | "
            f"BindGroups2: {audit['bind_groups_audited']} "
            f"(skipped-manifest={audit['bind_groups_skipped_manifest_layout']}) | "
            f"BG Mismatches: {audit['bind_group_mismatch_count']}"
        ),
        (
            f"  RenderPasses: {audit['render_passes_seen']} | "
            f"Draws Audited: {audit['draws_audited']} | "
            f"RP Mismatches: {audit['render_pass_mismatch_count']}"
        ),
        f"  Swapchain Target Formats: {swapchain_hist}",
    ]

    bg_mismatches = audit.get("bind_group_mismatches") or []
    if bg_mismatches:
        lines.append("  Bind-Group Mismatches:")
        for item in bg_mismatches:
            binding_str = f" @binding={item['binding']}" if item["binding"] is not None else ""
            lines.append(
                f"    [{item['kind']}] packet=#{item['packet_index']:04d} "
                f"bg={item['bind_group_id']} layout={item['layout_id']}{binding_str}: "
                f"{item['message']}"
            )

    rp_mismatches = audit.get("render_pass_mismatches") or []
    if rp_mismatches:
        lines.append("  Render-Pass Format Mismatches:")
        for item in rp_mismatches:
            slot_str = f" slot={item['slot']}" if item["slot"] is not None else " depth-stencil"
            lines.append(
                f"    [{item['kind']}] draw=#{item['packet_index']:04d} "
                f"pass=#{item['render_pass_packet_index']:04d} "
                f"pipeline={item['pipeline_id']}{slot_str}: {item['message']}"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Offline diagnostic audits (--bink-video-audit & --start-screen-skips)
# ---------------------------------------------------------------------------

_BINK_VIDEO_RESOLUTION_HINTS: dict[tuple[int, int], str] = {
    (3840, 2160): "4k_luma",
    (1920, 1080): "4k_chroma_or_1080p_luma",
    (960, 540): "1080p_chroma",
    (2560, 1440): "1440p_luma",
    (1280, 720): "1440p_chroma_or_720p_luma",
    (640, 360): "720p_chroma",
}

_SKIP_EVENT_REASONS = frozenset(
    {
        "set_pipeline_without_draw",
        "set_compute_pipeline_without_dispatch",
        "pipeline_never_executed",
        "missing_shader_module",
        "zero_count_draw",
        "zero_count_dispatch",
        "draw_outside_render_pass",
    }
)

_BINDING_ANOMALY_REASONS = frozenset(
    {
        "unbound_bind_group_slot",
        "bind_group_layout_mismatch",
        "missing_binding",
        "null_binding",
        "resource_kind_mismatch",
        "undeclared_binding",
        "duplicate_binding",
        "unknown_layout",
        "render_pass_format_mismatch",
    }
)


def _build_packet_frame_map(trace: DecodedTrace) -> dict[int, int]:
    """Map packet index -> frame_index from trace.frames."""
    mapping: dict[int, int] = {}
    for frame in trace.frames:
        for idx in range(frame.start_packet_index, frame.end_packet_index + 1):
            mapping[idx] = frame.frame_index
    return mapping


def _is_r8unorm_format(fmt: Any) -> bool:
    return isinstance(fmt, str) and fmt.strip().lower() == "r8unorm"


class _BinkPlaneRecord:
    __slots__ = (
        "texture_id",
        "width",
        "height",
        "depth_or_layers",
        "mip_levels",
        "usage_names",
        "upload_count",
        "upload_bytes",
        "upload_frames",
        "bind_group_count",
        "bind_group_ids",
        "set_bind_group_count",
        "draw_count",
        "dispatch_count",
        "sampled_pipeline_ids",
        "sampled_compute_pipeline_ids",
        "destroyed",
    )

    def __init__(self, texture_id: int) -> None:
        self.texture_id = texture_id
        self.width: int | None = None
        self.height: int | None = None
        self.depth_or_layers: int | None = None
        self.mip_levels: int | None = None
        self.usage_names: list[str] = []
        self.upload_count = 0
        self.upload_bytes = 0
        self.upload_frames: set[int] = set()
        self.bind_group_count = 0
        self.bind_group_ids: set[int] = set()
        self.set_bind_group_count = 0
        self.draw_count = 0
        self.dispatch_count = 0
        self.sampled_pipeline_ids: set[int] = set()
        self.sampled_compute_pipeline_ids: set[int] = set()
        self.destroyed = False


def audit_bink_video(trace: DecodedTrace) -> dict[str, Any]:
    """Audit r8unorm planar Bink2 video texture uploads, bind-group references, and draw sampling.

    Output is strictly sanitized metadata (counts, byte totals, dimensions, and resource IDs):
    zero host paths and zero raw image bytes.
    """
    packet_frames = _build_packet_frame_map(trace)

    # Pre-pass: identify every texture ID created with or viewed as r8unorm.
    planes: dict[int, _BinkPlaneRecord] = {}
    for pkt in trace.packets:
        if not pkt.valid:
            continue
        op = pkt.opcode
        f = pkt.fields
        if op == GpuOpcode.CreateTexture:
            fmt = f.get("format_webgpu") or f.get("format_name")
            if _is_r8unorm_format(fmt):
                tid = int(f["id"])
                rec = planes.setdefault(tid, _BinkPlaneRecord(tid))
                rec.width = int(f["width"])
                rec.height = int(f["height"])
                rec.depth_or_layers = int(f["depth_or_layers"])
                rec.mip_levels = int(f["mip_levels"])
                rec.usage_names = list(f.get("usage_names") or ())
        elif op == GpuOpcode.CreateTexture2:
            if _is_r8unorm_format(f.get("format")):
                tid = int(f["id"])
                rec = planes.setdefault(tid, _BinkPlaneRecord(tid))
                rec.width = int(f["width"])
                rec.height = int(f["height"])
                rec.depth_or_layers = int(f["depth_or_layers"])
                rec.mip_levels = int(f["mip_levels"])
                rec.usage_names = list(f.get("usage_names") or ())
        elif op == GpuOpcode.CreateBindGroup2:
            for entry in f.get("entries") or ():
                if int(entry["resource_type"]) != int(GpuBindResourceType.TextureView):
                    continue
                tid = int(entry["resource_id"])
                if tid == 0:
                    continue
                view = entry.get("view")
                if isinstance(view, dict) and _is_r8unorm_format(view.get("format")):
                    planes.setdefault(tid, _BinkPlaneRecord(tid))

    bind_group_r8_textures: dict[int, set[int]] = {}
    active_bind_groups: dict[int, int] = {}
    active_render_pipeline_id: int | None = None
    active_compute_pipeline_id: int | None = None
    active_pass_kind: str | None = None  # "render" | "compute" | None

    total_draws_sampling_planes = 0
    total_dispatches_sampling_planes = 0

    for pkt in trace.packets:
        if not pkt.valid:
            continue
        op = pkt.opcode
        f = pkt.fields

        if op == GpuOpcode.CreateTexture:
            tid = int(f["id"])
            rec = planes.get(tid)
            if rec is not None and _is_r8unorm_format(f.get("format_webgpu") or f.get("format_name")):
                rec.width = int(f["width"])
                rec.height = int(f["height"])
                rec.depth_or_layers = int(f["depth_or_layers"])
                rec.mip_levels = int(f["mip_levels"])
                rec.usage_names = list(f.get("usage_names") or ())
                rec.destroyed = False

        elif op == GpuOpcode.CreateTexture2:
            tid = int(f["id"])
            rec = planes.get(tid)
            if rec is not None and _is_r8unorm_format(f.get("format")):
                rec.width = int(f["width"])
                rec.height = int(f["height"])
                rec.depth_or_layers = int(f["depth_or_layers"])
                rec.mip_levels = int(f["mip_levels"])
                rec.usage_names = list(f.get("usage_names") or ())
                rec.destroyed = False

        elif op == GpuOpcode.WriteTexture:
            tid = int(f["texture_id"])
            rec = planes.get(tid)
            if rec is not None:
                rec.upload_count += 1
                rec.upload_bytes += int(f["data_len"])
                frame_idx = packet_frames.get(pkt.index, 0)
                rec.upload_frames.add(frame_idx)
                if rec.width is None and int(f.get("mip_level", 0)) == 0:
                    rec.width = int(f["width"])
                    rec.height = int(f["height"])
                    rec.depth_or_layers = int(f["depth_or_layers"])

        elif op in (GpuOpcode.CreateBindGroup, GpuOpcode.CreateBindGroup2):
            bg_id = int(f["id"])
            referenced_r8: set[int] = set()
            for entry in f.get("entries") or ():
                if int(entry["resource_type"]) != int(GpuBindResourceType.TextureView):
                    continue
                tid = int(entry["resource_id"])
                if tid in planes:
                    referenced_r8.add(tid)
            bind_group_r8_textures[bg_id] = referenced_r8
            for tid in referenced_r8:
                rec = planes[tid]
                rec.bind_group_count += 1
                rec.bind_group_ids.add(bg_id)

        elif op in (GpuOpcode.SetBindGroup, GpuOpcode.SetBindGroup2):
            group_idx = int(f["group_index"])
            bg_id = int(f["bind_group_id"])
            active_bind_groups[group_idx] = bg_id
            for tid in bind_group_r8_textures.get(bg_id, ()):
                planes[tid].set_bind_group_count += 1

        elif op == GpuOpcode.DestroyResource:
            rid = int(f["id"])
            if rid in planes:
                planes[rid].destroyed = True
            if rid in bind_group_r8_textures:
                bind_group_r8_textures.pop(rid, None)
                stale_groups = [g for g, bg in active_bind_groups.items() if bg == rid]
                for g in stale_groups:
                    active_bind_groups.pop(g, None)
            if active_render_pipeline_id == rid:
                active_render_pipeline_id = None
            if active_compute_pipeline_id == rid:
                active_compute_pipeline_id = None

        elif op in (GpuOpcode.BeginFrame, GpuOpcode.Present):
            active_bind_groups.clear()
            active_render_pipeline_id = None
            active_compute_pipeline_id = None
            active_pass_kind = None

        elif op == GpuOpcode.BeginRenderPass:
            active_bind_groups.clear()
            active_render_pipeline_id = None
            active_pass_kind = "render"

        elif op == GpuOpcode.EndRenderPass:
            active_bind_groups.clear()
            active_render_pipeline_id = None
            active_pass_kind = None

        elif op == GpuOpcode.SetPipeline:
            if active_pass_kind == "compute":
                active_bind_groups.clear()
            active_render_pipeline_id = int(f["pipeline_id"])
            active_pass_kind = "render"

        elif op == GpuOpcode.SetComputePipeline:
            if active_pass_kind == "render":
                active_bind_groups.clear()
            active_compute_pipeline_id = int(f["pipeline_id"])
            active_render_pipeline_id = None
            active_pass_kind = "compute"

        elif pkt.name in _DRAW_PACKET_NAMES:
            bound_r8: set[int] = set()
            for bg_id in active_bind_groups.values():
                bound_r8.update(bind_group_r8_textures.get(bg_id, ()))
            if bound_r8:
                total_draws_sampling_planes += 1
                for tid in bound_r8:
                    rec = planes[tid]
                    rec.draw_count += 1
                    if active_render_pipeline_id is not None:
                        rec.sampled_pipeline_ids.add(active_render_pipeline_id)

        elif pkt.name in _DISPATCH_PACKET_NAMES:
            bound_r8 = set()
            for bg_id in active_bind_groups.values():
                bound_r8.update(bind_group_r8_textures.get(bg_id, ()))
            if bound_r8:
                total_dispatches_sampling_planes += 1
                for tid in bound_r8:
                    rec = planes[tid]
                    rec.dispatch_count += 1
                    if active_compute_pipeline_id is not None:
                        rec.sampled_compute_pipeline_ids.add(active_compute_pipeline_id)

    # Detect half-resolution Y / Cb / Cr planar texture groups: (W, H) paired with (W//2, H//2).
    dims_to_tids: dict[tuple[int, int], list[int]] = {}
    for rec in planes.values():
        if rec.width is not None and rec.height is not None and rec.width > 0 and rec.height > 0:
            dims_to_tids.setdefault((rec.width, rec.height), []).append(rec.texture_id)

    planar_full_tids: set[int] = set()
    planar_half_tids: set[int] = set()
    planar_groups: list[dict[str, Any]] = []
    for (w, h), luma_tids in sorted(dims_to_tids.items(), key=lambda kv: (-kv[0][0] * kv[0][1], kv[0])):
        if w < 2 or h < 2 or (w % 2) != 0 or (h % 2) != 0:
            continue
        half_dim = (w // 2, h // 2)
        chroma_tids = dims_to_tids.get(half_dim)
        if not chroma_tids:
            continue
        sorted_luma = sorted(luma_tids)
        sorted_chroma = sorted(chroma_tids)
        planar_full_tids.update(sorted_luma)
        planar_half_tids.update(sorted_chroma)
        group_tids = sorted_luma + sorted_chroma
        planar_groups.append(
            {
                "luma_dimensions": f"{w}x{h}",
                "chroma_dimensions": f"{half_dim[0]}x{half_dim[1]}",
                "luma_texture_ids": sorted_luma,
                "chroma_texture_ids": sorted_chroma,
                "total_upload_count": sum(planes[t].upload_count for t in group_tids),
                "total_upload_bytes": sum(planes[t].upload_bytes for t in group_tids),
                "any_sampled_by_draw": any(planes[t].draw_count > 0 for t in group_tids),
                "any_sampled_by_dispatch": any(planes[t].dispatch_count > 0 for t in group_tids),
            }
        )

    plane_rows: list[dict[str, Any]] = []
    for rec in planes.values():
        dims = (rec.width, rec.height) if rec.width is not None and rec.height is not None else None
        is_video_res = dims in _BINK_VIDEO_RESOLUTION_HINTS if dims is not None else False
        upload_frame_count = len(rec.upload_frames)
        per_frame_uploads = upload_frame_count >= 2
        in_planar_group = rec.texture_id in planar_full_tids or rec.texture_id in planar_half_tids
        is_render_attachment = "RENDER_ATTACHMENT" in rec.usage_names
        is_video_candidate = bool(
            is_video_res or per_frame_uploads or (in_planar_group and rec.upload_count > 0)
        )

        if dims is not None and dims in _BINK_VIDEO_RESOLUTION_HINTS:
            plane_hint = _BINK_VIDEO_RESOLUTION_HINTS[dims]
        elif rec.texture_id in planar_full_tids:
            plane_hint = "planar_luma"
        elif rec.texture_id in planar_half_tids:
            plane_hint = "planar_chroma"
        elif per_frame_uploads:
            plane_hint = "r8unorm_per_frame"
        else:
            plane_hint = "r8unorm"

        plane_rows.append(
            {
                "texture_id": rec.texture_id,
                "width": rec.width,
                "height": rec.height,
                "depth_or_layers": rec.depth_or_layers,
                "mip_levels": rec.mip_levels,
                "format": "r8unorm",
                "usage_names": list(rec.usage_names),
                "is_render_attachment": is_render_attachment,
                "plane_hint": plane_hint,
                "is_video_resolution": is_video_res,
                "is_video_candidate": is_video_candidate,
                "upload_count": rec.upload_count,
                "upload_bytes": rec.upload_bytes,
                "upload_frame_count": upload_frame_count,
                "per_frame_uploads": per_frame_uploads,
                "bind_group_count": rec.bind_group_count,
                "bind_group_ids": sorted(rec.bind_group_ids),
                "set_bind_group_count": rec.set_bind_group_count,
                "draw_count": rec.draw_count,
                "dispatch_count": rec.dispatch_count,
                "sampled_by_draw": rec.draw_count > 0,
                "sampled_by_dispatch": rec.dispatch_count > 0,
                "sampled_pipeline_ids": sorted(rec.sampled_pipeline_ids),
                "sampled_compute_pipeline_ids": sorted(rec.sampled_compute_pipeline_ids),
                "destroyed": rec.destroyed,
            }
        )

    plane_rows.sort(
        key=lambda row: (
            not bool(row["is_video_candidate"]),
            -int(row["upload_count"]),
            -int(row["upload_bytes"]),
            -int(row["draw_count"]),
            int(row["texture_id"]),
        )
    )

    video_candidates = [row for row in plane_rows if row["is_video_candidate"]]
    any_r8unorm_sampled_by_draw = any(row["sampled_by_draw"] for row in plane_rows)
    any_video_plane_sampled_by_draw = any(row["sampled_by_draw"] for row in video_candidates)
    any_plane_sampled_by_draw = (
        any_video_plane_sampled_by_draw if video_candidates else any_r8unorm_sampled_by_draw
    )

    return {
        "r8unorm_texture_count": len(plane_rows),
        "video_candidate_count": len(video_candidates),
        "uploaded_plane_count": sum(1 for row in plane_rows if row["upload_count"] > 0),
        "per_frame_uploaded_plane_count": sum(1 for row in plane_rows if row["per_frame_uploads"]),
        "bound_plane_count": sum(1 for row in plane_rows if row["bind_group_count"] > 0),
        "sampled_by_draw_plane_count": sum(1 for row in plane_rows if row["sampled_by_draw"]),
        "sampled_by_dispatch_plane_count": sum(
            1 for row in plane_rows if row["sampled_by_dispatch"]
        ),
        "any_plane_sampled_by_draw": any_plane_sampled_by_draw,
        "any_video_plane_sampled_by_draw": any_video_plane_sampled_by_draw,
        "any_r8unorm_sampled_by_draw": any_r8unorm_sampled_by_draw,
        "total_upload_count": sum(int(row["upload_count"]) for row in plane_rows),
        "total_upload_bytes": sum(int(row["upload_bytes"]) for row in plane_rows),
        "total_bind_group_refs": sum(int(row["bind_group_count"]) for row in plane_rows),
        "total_set_bind_group_refs": sum(int(row["set_bind_group_count"]) for row in plane_rows),
        "total_draws_sampling_planes": total_draws_sampling_planes,
        "total_dispatches_sampling_planes": total_dispatches_sampling_planes,
        "planar_groups": planar_groups,
        "planes": plane_rows,
    }


def format_bink_video_audit_text(audit: dict[str, Any]) -> str:
    """Format an audit_bink_video() dictionary as a sanitized human-readable report."""
    sampled_str = "YES" if audit["any_plane_sampled_by_draw"] else "NO"
    lines: list[str] = [
        "Bink Video Audit (r8unorm planes):",
        (
            f"  Planes: {audit['r8unorm_texture_count']} "
            f"(video-candidates={audit['video_candidate_count']}, "
            f"uploaded={audit['uploaded_plane_count']}, "
            f"per-frame={audit['per_frame_uploaded_plane_count']}, "
            f"bound={audit['bound_plane_count']}, "
            f"sampled-by-draw={audit['sampled_by_draw_plane_count']}, "
            f"sampled-by-dispatch={audit['sampled_by_dispatch_plane_count']})"
        ),
        f"  Any Video Plane Sampled By Draw: {sampled_str}",
        (
            f"  Uploads: {audit['total_upload_count']} ({audit['total_upload_bytes']} B) | "
            f"BindGroup Refs: {audit['total_bind_group_refs']} | "
            f"SetBindGroup Refs: {audit['total_set_bind_group_refs']} | "
            f"Draws Sampling Planes: {audit['total_draws_sampling_planes']} | "
            f"Dispatches Sampling Planes: {audit['total_dispatches_sampling_planes']}"
        ),
    ]

    groups = audit.get("planar_groups") or []
    if groups:
        lines.append("  Planar Groups:")
        for grp in groups:
            luma_ids = ",".join(f"#{t}" for t in grp["luma_texture_ids"])
            chroma_ids = ",".join(f"#{t}" for t in grp["chroma_texture_ids"])
            grp_sampled = "YES" if grp["any_sampled_by_draw"] else "NO"
            lines.append(
                f"    {grp['luma_dimensions']} (Y: {luma_ids}) + "
                f"{grp['chroma_dimensions']} (Cb/Cr: {chroma_ids}) "
                f"[uploads={grp['total_upload_count']}, bytes={grp['total_upload_bytes']}, "
                f"sampled-by-draw={grp_sampled}]"
            )

    planes = audit.get("planes") or []
    if not planes:
        lines.append("  (none)")
        return "\n".join(lines)

    lines.append(
        f"  {'TexID':<8s}  {'Dimensions':<11s}  {'Hint':<24s}  "
        f"{'Uploads':>7s}  {'Bytes':>11s}  {'Frames':>6s}  {'BGs':>4s}  "
        f"{'SetBG':>5s}  {'Draws':>5s}  {'Disp':>4s}  {'Sampled':<7s}"
    )
    lines.append(
        f"  {'-' * 8}  {'-' * 11}  {'-' * 24}  "
        f"{'-' * 7}  {'-' * 11}  {'-' * 6}  {'-' * 4}  "
        f"{'-' * 5}  {'-' * 5}  {'-' * 4}  {'-' * 7}"
    )
    for row in planes:
        dims = (
            f"{row['width']}x{row['height']}"
            if row["width"] is not None and row["height"] is not None
            else "unknown"
        )
        sampled_flag = "draw" if row["sampled_by_draw"] else (
            "compute" if row["sampled_by_dispatch"] else "no"
        )
        lines.append(
            f"  {int(row['texture_id']):<8d}  {dims:<11s}  {str(row['plane_hint']):<24s}  "
            f"{int(row['upload_count']):7d}  {int(row['upload_bytes']):11d}  "
            f"{int(row['upload_frame_count']):6d}  {int(row['bind_group_count']):4d}  "
            f"{int(row['set_bind_group_count']):5d}  {int(row['draw_count']):5d}  "
            f"{int(row['dispatch_count']):4d}  {sampled_flag:<7s}"
        )
    return "\n".join(lines)


class _StartScreenPairRecord:
    __slots__ = (
        "kind",
        "shader_pair",
        "vs_shader",
        "ps_shader",
        "cs_shader",
        "pipeline_ids",
        "set_count",
        "emitted_draws",
        "emitted_dispatches",
        "reason_counts",
        "affected_bind_group_ids",
        "affected_bindings",
    )

    def __init__(
        self,
        *,
        kind: str,
        shader_pair: str,
        vs_shader: str | None = None,
        ps_shader: str | None = None,
        cs_shader: str | None = None,
    ) -> None:
        self.kind = kind
        self.shader_pair = shader_pair
        self.vs_shader = vs_shader
        self.ps_shader = ps_shader
        self.cs_shader = cs_shader
        self.pipeline_ids: set[int] = set()
        self.set_count = 0
        self.emitted_draws = 0
        self.emitted_dispatches = 0
        self.reason_counts: Counter[str] = Counter()
        self.affected_bind_group_ids: set[int] = set()
        self.affected_bindings: set[str] = set()


def _layouts_equivalent(
    expected: _BindGroupLayoutRecord | None,
    actual: _BindGroupLayoutRecord | None,
) -> bool:
    if expected is None or actual is None:
        return False
    if expected.layout_id == actual.layout_id:
        return True
    if (
        expected.content_id != _ZERO_HASH_HEX
        and actual.content_id != _ZERO_HASH_HEX
        and expected.content_id == actual.content_id
    ):
        return True
    if not expected.manifest_reference and not actual.manifest_reference:
        if expected.entries.keys() != actual.entries.keys():
            return False
        for b, e_entry in expected.entries.items():
            a_entry = actual.entries[b]
            if (
                e_entry.get("type") != a_entry.get("type")
                or e_entry.get("binding_type") != a_entry.get("binding_type")
                or e_entry.get("has_dynamic_offset") != a_entry.get("has_dynamic_offset")
            ):
                return False
        return True
    return False


def _render_pass_mismatch_count(
    active_render_pass: _ActiveRenderPass,
    pipe_info: dict[str, Any],
) -> int:
    """Count render-pass attachment vs CreateRenderPipeline2 key mismatches for one draw."""
    mismatches = 0
    pass_colors = active_render_pass.color_attachments
    pipe_colors = pipe_info["color_targets"]
    max_slots = max(len(pass_colors), len(pipe_colors))
    for slot in range(max_slots):
        pass_att = pass_colors[slot] if slot < len(pass_colors) else None
        pipe_tgt = pipe_colors[slot] if slot < len(pipe_colors) else None
        if pass_att is None and pipe_tgt is None:
            continue
        if pass_att is None or pipe_tgt is None:
            mismatches += 1
            continue
        pipe_fmt = str(pipe_tgt["format"])
        if pass_att.format is not None and pass_att.format != pipe_fmt:
            mismatches += 1

    pass_ds = active_render_pass.depth_stencil
    pipe_ds = pipe_info["depth_stencil"]
    if (pass_ds is None) != (pipe_ds is None):
        mismatches += 1
    elif pass_ds is not None and pipe_ds is not None:
        pipe_ds_fmt = str(pipe_ds["format"])
        if pass_ds.format is not None and pass_ds.format != pipe_ds_fmt:
            mismatches += 1
    return mismatches


def audit_start_screen_skips(trace: DecodedTrace) -> dict[str, Any]:
    """Rank top pipelines/shader pairs with skipped/unemitted draws or missing-binding anomalies.

    Output is strictly sanitized (shader hashes, pipeline/group/binding IDs, and counts):
    zero host paths and zero raw image bytes.
    """
    hash_refs: dict[int, str] = dict(trace._shader_hash_refs)

    # Pre-pass: correlate live CreateShaderModule packet indices with v2 pipeline key hashes.
    active_mod_pkt: dict[int, int] = {}
    mod_pkt_key_hash: dict[int, str] = {}
    for pkt in trace.packets:
        if not pkt.valid:
            continue
        op = pkt.opcode
        f = pkt.fields
        if op == GpuOpcode.CreateShaderModule:
            active_mod_pkt[int(f["id"])] = pkt.index
        elif op == GpuOpcode.DestroyResource:
            active_mod_pkt.pop(int(f["id"]), None)
        elif op == GpuOpcode.CreateRenderPipeline2:
            key = f["key"]
            vs_id = int(f["vs_module_id"])
            fs_id = int(f["fs_module_id"])
            vs_hash = str(key["vs_shader"])
            ps_hash = str(key["ps_shader"])
            if vs_id != 0 and vs_hash != _ZERO_HASH_HEX and vs_id in active_mod_pkt:
                mod_pkt_key_hash.setdefault(active_mod_pkt[vs_id], vs_hash)
            if fs_id != 0 and ps_hash != _ZERO_HASH_HEX and fs_id in active_mod_pkt:
                mod_pkt_key_hash.setdefault(active_mod_pkt[fs_id], ps_hash)
        elif op == GpuOpcode.CreateComputePipeline2:
            cs_id = int(f["cs_module_id"])
            cs_hash = str(f["cs_shader"])
            if cs_id != 0 and cs_hash != _ZERO_HASH_HEX and cs_id in active_mod_pkt:
                mod_pkt_key_hash.setdefault(active_mod_pkt[cs_id], cs_hash)

    pair_records: dict[tuple[str, ...], _StartScreenPairRecord] = {}
    active_module_hashes: dict[int, str | None] = {}
    destroyed_module_ids: set[int] = set()

    layouts: dict[int, _BindGroupLayoutRecord] = {}
    pipeline_layouts: dict[int, list[int]] = {}
    bind_groups: dict[int, dict[str, Any]] = {}
    textures: dict[int, str] = {}

    # Track created pipelines (both live and historical execution counts).
    render_pipelines: dict[int, dict[str, Any]] = {}
    compute_pipelines: dict[int, dict[str, Any]] = {}
    created_pipelines: dict[int, dict[str, Any]] = {}

    active_render_pass: _ActiveRenderPass | None = None
    active_bind_groups: dict[int, int] = {}
    active_pass_kind: str | None = None

    active_render_pipeline_id: int | None = None
    pending_render_drew = False
    active_compute_pipeline_id: int | None = None
    pending_compute_dispatched = False

    total_emitted_draws = 0
    total_emitted_dispatches = 0
    unbound_draws = 0
    unbound_dispatches = 0
    uses_render_passes = trace.begin_render_passes > 0

    def _get_pair(
        *,
        kind: str,
        vs_shader: str | None = None,
        ps_shader: str | None = None,
        cs_shader: str | None = None,
        pipeline_id: int | None = None,
    ) -> _StartScreenPairRecord:
        if kind == "render":
            vs_label = vs_shader or "none"
            ps_label = ps_shader or "none"
            pkey = ("render", vs_label, ps_label)
            pair_label = f"{vs_label} / {ps_label}"
            rec = pair_records.get(pkey)
            if rec is None:
                rec = _StartScreenPairRecord(
                    kind="render",
                    shader_pair=pair_label,
                    vs_shader=vs_label,
                    ps_shader=ps_label,
                )
                pair_records[pkey] = rec
        else:
            cs_label = cs_shader or "none"
            pkey = ("compute", cs_label)
            pair_label = f"cs:{cs_label}"
            rec = pair_records.get(pkey)
            if rec is None:
                rec = _StartScreenPairRecord(
                    kind="compute",
                    shader_pair=pair_label,
                    cs_shader=cs_label,
                )
                pair_records[pkey] = rec
        if pipeline_id is not None:
            rec.pipeline_ids.add(pipeline_id)
        return rec

    def _resolve_shader_label(module_id: int, key_hash: str | None) -> str:
        if key_hash is not None and key_hash != _ZERO_HASH_HEX:
            return key_hash
        if module_id == 0:
            return "none"
        mod_hash = active_module_hashes.get(module_id)
        if mod_hash is not None:
            return mod_hash
        return f"id:{module_id}"

    def _check_bound_groups_for_pipeline(
        pair_rec: _StartScreenPairRecord,
        pipe_info: dict[str, Any] | None,
    ) -> None:
        checked_bg_ids: set[int] = set()
        expected_group_set: set[int] | None = None
        if pipe_info is not None:
            pll_id = int(pipe_info.get("pipeline_layout_id", 0))
            if pll_id != 0 and pll_id in pipeline_layouts:
                expected_groups = pipeline_layouts[pll_id]
                expected_group_set = {
                    idx for idx, bgl_id in enumerate(expected_groups) if bgl_id != 0
                }
                for group_idx, expected_bgl_id in enumerate(expected_groups):
                    if expected_bgl_id == 0:
                        continue
                    expected_bgl = layouts.get(expected_bgl_id)
                    requires_bg = (
                        expected_bgl is None
                        or expected_bgl.manifest_reference
                        or len(expected_bgl.entries) > 0
                    )
                    bg_id = active_bind_groups.get(group_idx)
                    if bg_id is None:
                        if requires_bg:
                            pair_rec.reason_counts["unbound_bind_group_slot"] += 1
                            pair_rec.affected_bindings.add(f"@group({group_idx})")
                        continue
                    bg_info = bind_groups.get(bg_id)
                    if bg_info is not None and bg_info["v2"]:
                        actual_bgl_id = int(bg_info["layout_id"])
                        if actual_bgl_id != expected_bgl_id:
                            actual_bgl = layouts.get(actual_bgl_id)
                            if not _layouts_equivalent(expected_bgl, actual_bgl):
                                pair_rec.reason_counts["bind_group_layout_mismatch"] += 1
                                pair_rec.affected_bind_group_ids.add(bg_id)
                                pair_rec.affected_bindings.add(f"@group({group_idx})")

        for group_idx, bg_id in sorted(active_bind_groups.items()):
            if expected_group_set is not None and group_idx not in expected_group_set:
                continue
            if bg_id in checked_bg_ids:
                continue
            checked_bg_ids.add(bg_id)
            bg_info = bind_groups.get(bg_id)
            if bg_info is None:
                continue
            anomalies: list[tuple[str, int | None]] = bg_info["anomalies"]
            for reason, binding in anomalies:
                pair_rec.reason_counts[reason] += 1
                pair_rec.affected_bind_group_ids.add(bg_id)
                if binding is not None:
                    pair_rec.affected_bindings.add(f"@group({group_idx}):@binding({binding})")
                else:
                    pair_rec.affected_bindings.add(f"@group({group_idx})")

    def _flush_pending_render() -> None:
        nonlocal active_render_pipeline_id, pending_render_drew
        if active_render_pipeline_id is not None and not pending_render_drew:
            pid = active_render_pipeline_id
            pipe_info = render_pipelines.get(pid)
            if pipe_info is not None:
                pair_rec = pipe_info["pair"]
            else:
                pair_rec = _get_pair(
                    kind="render",
                    vs_shader=f"pipeline:{pid}",
                    ps_shader="unknown",
                    pipeline_id=pid,
                )
            pair_rec.reason_counts["set_pipeline_without_draw"] += 1
            _check_bound_groups_for_pipeline(pair_rec, pipe_info)
        active_render_pipeline_id = None
        pending_render_drew = False

    def _flush_pending_compute() -> None:
        nonlocal active_compute_pipeline_id, pending_compute_dispatched
        if active_compute_pipeline_id is not None and not pending_compute_dispatched:
            pid = active_compute_pipeline_id
            pipe_info = compute_pipelines.get(pid)
            if pipe_info is not None:
                pair_rec = pipe_info["pair"]
            else:
                pair_rec = _get_pair(
                    kind="compute",
                    cs_shader=f"pipeline:{pid}",
                    pipeline_id=pid,
                )
            pair_rec.reason_counts["set_compute_pipeline_without_dispatch"] += 1
            _check_bound_groups_for_pipeline(pair_rec, pipe_info)
        active_compute_pipeline_id = None
        pending_compute_dispatched = False

    for pkt in trace.packets:
        if not pkt.valid:
            continue
        op = pkt.opcode
        f = pkt.fields

        if op == GpuOpcode.CreateShaderModule:
            mod_id = int(f["id"])
            resolved_hash: str | None = mod_pkt_key_hash.get(pkt.index)
            if resolved_hash is None:
                resolved_hash = hash_refs.get(pkt.index)
            if resolved_hash is None and int(f.get("code_len", 0)) > 0:
                resolved_hash = str(f["wgsl_hash_fnv1a64"])
            active_module_hashes[mod_id] = resolved_hash
            destroyed_module_ids.discard(mod_id)

        elif op == GpuOpcode.CreateBindGroupLayout:
            layout_id = int(f["id"])
            entries_map = {int(e["binding"]): e for e in f.get("entries", ())}
            layouts[layout_id] = _BindGroupLayoutRecord(
                layout_id=layout_id,
                manifest_reference=bool(f["manifest_reference"]),
                content_id=str(f["content_id"]),
                entries=entries_map,
            )

        elif op == GpuOpcode.CreatePipelineLayout:
            pipeline_layouts[int(f["id"])] = [int(x) for x in f.get("bind_group_layout_ids", ())]

        elif op == GpuOpcode.CreateBindGroup:
            bg_id = int(f["id"])
            bind_groups[bg_id] = {
                "v2": False,
                "layout_id": 0,
                "anomalies": [],
            }

        elif op == GpuOpcode.CreateBindGroup2:
            bg_id = int(f["id"])
            layout_id = int(f["layout_id"])
            bg_entries = list(f.get("entries", ()))
            anomalies: list[tuple[str, int | None]] = []

            for entry in bg_entries:
                if bool(entry.get("null_binding")):
                    anomalies.append(("null_binding", int(entry["binding"])))

            layout = layouts.get(layout_id)
            if layout is None:
                anomalies.append(("unknown_layout", None))
            elif not layout.manifest_reference:
                seen_bindings: set[int] = set()
                for entry in bg_entries:
                    binding = int(entry["binding"])
                    rtype = int(entry["resource_type"])
                    if binding in seen_bindings:
                        anomalies.append(("duplicate_binding", binding))
                        continue
                    seen_bindings.add(binding)
                    layout_entry = layout.entries.get(binding)
                    if layout_entry is None:
                        anomalies.append(("undeclared_binding", binding))
                        continue
                    expected_pair = _LAYOUT_TYPE_TO_EXPECTED_RESOURCE.get(str(layout_entry["type"]))
                    if expected_pair is not None and rtype != int(expected_pair[0]):
                        anomalies.append(("resource_kind_mismatch", binding))
                for binding in sorted(layout.entries):
                    if binding not in seen_bindings:
                        anomalies.append(("missing_binding", binding))

            bind_groups[bg_id] = {
                "v2": True,
                "layout_id": layout_id,
                "anomalies": anomalies,
            }

        elif op == GpuOpcode.CreateTexture:
            fmt = f.get("format_webgpu") or f.get("format_name")
            if fmt:
                textures[int(f["id"])] = str(fmt)

        elif op == GpuOpcode.CreateTexture2:
            fmt = f.get("format")
            if fmt and fmt != "undefined":
                textures[int(f["id"])] = str(fmt)

        elif op == GpuOpcode.CreateRenderPipeline:
            pid = int(f["id"])
            vs_id = int(f["vs_shader_id"])
            fs_id = int(f["fs_shader_id"])
            vs_label = _resolve_shader_label(vs_id, None)
            fs_label = _resolve_shader_label(fs_id, None)
            pair_rec = _get_pair(
                kind="render",
                vs_shader=vs_label,
                ps_shader=fs_label,
                pipeline_id=pid,
            )
            missing_mod = (
                vs_id == 0
                or fs_id == 0
                or vs_id in destroyed_module_ids
                or fs_id in destroyed_module_ids
            )
            if missing_mod:
                pair_rec.reason_counts["missing_shader_module"] += 1
            pipe_info = {
                "id": pid,
                "kind": "render",
                "v2": False,
                "pair": pair_rec,
                "pipeline_layout_id": 0,
                "color_targets": None,
                "depth_stencil": None,
                "set_count": 0,
                "exec_count": 0,
                "zero_exec_count": 0,
            }
            render_pipelines[pid] = pipe_info
            created_pipelines[pid] = pipe_info

        elif op == GpuOpcode.CreateRenderPipeline2:
            pid = int(f["id"])
            vs_id = int(f["vs_module_id"])
            fs_id = int(f["fs_module_id"])
            pll_id = int(f["pipeline_layout_id"])
            key = f["key"]
            vs_hash = str(key["vs_shader"])
            ps_hash = str(key["ps_shader"])
            vs_label = _resolve_shader_label(vs_id, vs_hash)
            ps_label = _resolve_shader_label(fs_id, ps_hash)
            pair_rec = _get_pair(
                kind="render",
                vs_shader=vs_label,
                ps_shader=ps_label,
                pipeline_id=pid,
            )
            has_color_targets = any(t is not None for t in key.get("color_targets") or ())
            missing_vs = vs_id == 0 or vs_id in destroyed_module_ids
            missing_fs = (
                (fs_id == 0 and (ps_hash != _ZERO_HASH_HEX or has_color_targets))
                or (fs_id != 0 and fs_id in destroyed_module_ids)
            )
            if missing_vs or missing_fs:
                pair_rec.reason_counts["missing_shader_module"] += 1
            pipe_info = {
                "id": pid,
                "kind": "render",
                "v2": True,
                "pair": pair_rec,
                "pipeline_layout_id": pll_id,
                "color_targets": list(key["color_targets"]),
                "depth_stencil": key["depth_stencil"],
                "set_count": 0,
                "exec_count": 0,
                "zero_exec_count": 0,
            }
            render_pipelines[pid] = pipe_info
            created_pipelines[pid] = pipe_info

        elif op == GpuOpcode.CreateComputePipeline:
            pid = int(f["id"])
            cs_id = int(f["cs_shader_id"])
            cs_label = _resolve_shader_label(cs_id, None)
            pair_rec = _get_pair(kind="compute", cs_shader=cs_label, pipeline_id=pid)
            if cs_id == 0 or cs_id in destroyed_module_ids:
                pair_rec.reason_counts["missing_shader_module"] += 1
            pipe_info = {
                "id": pid,
                "kind": "compute",
                "v2": False,
                "pair": pair_rec,
                "pipeline_layout_id": 0,
                "set_count": 0,
                "exec_count": 0,
                "zero_exec_count": 0,
            }
            compute_pipelines[pid] = pipe_info
            created_pipelines[pid] = pipe_info

        elif op == GpuOpcode.CreateComputePipeline2:
            pid = int(f["id"])
            cs_id = int(f["cs_module_id"])
            pll_id = int(f["pipeline_layout_id"])
            cs_hash = str(f["cs_shader"])
            cs_label = _resolve_shader_label(cs_id, cs_hash)
            pair_rec = _get_pair(kind="compute", cs_shader=cs_label, pipeline_id=pid)
            if cs_id == 0 or cs_id in destroyed_module_ids:
                pair_rec.reason_counts["missing_shader_module"] += 1
            pipe_info = {
                "id": pid,
                "kind": "compute",
                "v2": True,
                "pair": pair_rec,
                "pipeline_layout_id": pll_id,
                "set_count": 0,
                "exec_count": 0,
                "zero_exec_count": 0,
            }
            compute_pipelines[pid] = pipe_info
            created_pipelines[pid] = pipe_info

        elif op == GpuOpcode.DestroyResource:
            rid = int(f["id"])
            if active_render_pipeline_id == rid:
                _flush_pending_render()
            if active_compute_pipeline_id == rid:
                _flush_pending_compute()
            if rid in active_module_hashes:
                active_module_hashes.pop(rid, None)
                destroyed_module_ids.add(rid)
            layouts.pop(rid, None)
            pipeline_layouts.pop(rid, None)
            bind_groups.pop(rid, None)
            textures.pop(rid, None)
            render_pipelines.pop(rid, None)
            compute_pipelines.pop(rid, None)
            stale_groups = [g for g, bg in active_bind_groups.items() if bg == rid]
            for g in stale_groups:
                active_bind_groups.pop(g, None)

        elif op in (GpuOpcode.BeginFrame, GpuOpcode.Present):
            _flush_pending_render()
            _flush_pending_compute()
            active_render_pass = None
            active_bind_groups.clear()
            active_pass_kind = None

        elif op == GpuOpcode.BeginRenderPass:
            _flush_pending_render()
            _flush_pending_compute()
            active_bind_groups.clear()
            active_pass_kind = "render"
            resolved_colors: list[_ResolvedAttachment | None] = []
            for color in f.get("color_attachments", ()):
                if color is None:
                    resolved_colors.append(None)
                    continue
                tex_id = int(color["texture_id"])
                is_swapchain = bool(color["swapchain"])
                view_fmt = str(color["view"]["format"])
                if view_fmt and view_fmt != "undefined":
                    fmt: str | None = view_fmt
                    fmt_src = "view"
                elif is_swapchain:
                    fmt = None
                    fmt_src = "swapchain"
                elif tex_id in textures:
                    fmt = textures[tex_id]
                    fmt_src = "texture"
                else:
                    fmt = None
                    fmt_src = "unknown"
                resolved_colors.append(
                    _ResolvedAttachment(
                        texture_id=tex_id,
                        swapchain=is_swapchain,
                        format=fmt,
                        format_source=fmt_src,
                    )
                )
            resolved_ds: _ResolvedAttachment | None = None
            ds = f.get("depth_stencil")
            if ds is not None:
                ds_tex_id = int(ds["texture_id"])
                ds_view_fmt = str(ds["view"]["format"])
                if ds_view_fmt and ds_view_fmt != "undefined":
                    ds_fmt: str | None = ds_view_fmt
                    ds_src = "view"
                elif ds_tex_id in textures:
                    ds_fmt = textures[ds_tex_id]
                    ds_src = "texture"
                else:
                    ds_fmt = None
                    ds_src = "unknown"
                resolved_ds = _ResolvedAttachment(
                    texture_id=ds_tex_id,
                    swapchain=False,
                    format=ds_fmt,
                    format_source=ds_src,
                )
            active_render_pass = _ActiveRenderPass(
                packet_index=pkt.index,
                color_attachments=tuple(resolved_colors),
                depth_stencil=resolved_ds,
            )

        elif op == GpuOpcode.EndRenderPass:
            _flush_pending_render()
            active_render_pass = None
            active_bind_groups.clear()
            active_pass_kind = None

        elif op in (GpuOpcode.SetBindGroup, GpuOpcode.SetBindGroup2):
            active_bind_groups[int(f["group_index"])] = int(f["bind_group_id"])

        elif op == GpuOpcode.SetPipeline:
            _flush_pending_render()
            _flush_pending_compute()
            if active_pass_kind == "compute":
                active_bind_groups.clear()
            active_pass_kind = "render"
            pid = int(f["pipeline_id"])
            active_render_pipeline_id = pid
            pending_render_drew = False
            pipe_info = render_pipelines.get(pid)
            if pipe_info is not None:
                pipe_info["set_count"] += 1
                pipe_info["pair"].set_count += 1
            else:
                pair_rec = _get_pair(
                    kind="render",
                    vs_shader=f"pipeline:{pid}",
                    ps_shader="unknown",
                    pipeline_id=pid,
                )
                pair_rec.set_count += 1

        elif op == GpuOpcode.SetComputePipeline:
            _flush_pending_render()
            _flush_pending_compute()
            if active_pass_kind == "render":
                active_bind_groups.clear()
            active_render_pass = None
            active_pass_kind = "compute"
            pid = int(f["pipeline_id"])
            active_compute_pipeline_id = pid
            pending_compute_dispatched = False
            pipe_info = compute_pipelines.get(pid)
            if pipe_info is not None:
                pipe_info["set_count"] += 1
                pipe_info["pair"].set_count += 1
            else:
                pair_rec = _get_pair(
                    kind="compute",
                    cs_shader=f"pipeline:{pid}",
                    pipeline_id=pid,
                )
                pair_rec.set_count += 1

        elif pkt.name in _DRAW_PACKET_NAMES:
            if active_render_pipeline_id is None:
                unbound_draws += 1
                continue
            pending_render_drew = True
            pid = active_render_pipeline_id
            pipe_info = render_pipelines.get(pid)
            if pipe_info is not None:
                pair_rec = pipe_info["pair"]
            else:
                pair_rec = _get_pair(
                    kind="render",
                    vs_shader=f"pipeline:{pid}",
                    ps_shader="unknown",
                    pipeline_id=pid,
                )

            is_zero = False
            if op == GpuOpcode.Draw:
                is_zero = int(f["vertex_count"]) == 0 or int(f["instance_count"]) == 0
            elif op == GpuOpcode.DrawIndexed:
                is_zero = int(f["index_count"]) == 0 or int(f["instance_count"]) == 0

            if is_zero:
                pair_rec.reason_counts["zero_count_draw"] += 1
                if pipe_info is not None:
                    pipe_info["zero_exec_count"] += 1
            else:
                total_emitted_draws += 1
                pair_rec.emitted_draws += 1
                if pipe_info is not None:
                    pipe_info["exec_count"] += 1

            if uses_render_passes and active_render_pass is None:
                pair_rec.reason_counts["draw_outside_render_pass"] += 1

            _check_bound_groups_for_pipeline(pair_rec, pipe_info)

            if (
                pipe_info is not None
                and pipe_info["v2"]
                and active_render_pass is not None
            ):
                rp_mismatches = _render_pass_mismatch_count(active_render_pass, pipe_info)
                if rp_mismatches > 0:
                    pair_rec.reason_counts["render_pass_format_mismatch"] += rp_mismatches

        elif pkt.name in _DISPATCH_PACKET_NAMES:
            if active_compute_pipeline_id is None:
                unbound_dispatches += 1
                continue
            pending_compute_dispatched = True
            pid = active_compute_pipeline_id
            pipe_info = compute_pipelines.get(pid)
            if pipe_info is not None:
                pair_rec = pipe_info["pair"]
            else:
                pair_rec = _get_pair(
                    kind="compute",
                    cs_shader=f"pipeline:{pid}",
                    pipeline_id=pid,
                )

            is_zero = False
            if op == GpuOpcode.DispatchCompute:
                is_zero = (
                    int(f["workgroup_x"]) == 0
                    or int(f["workgroup_y"]) == 0
                    or int(f["workgroup_z"]) == 0
                )

            if is_zero:
                pair_rec.reason_counts["zero_count_dispatch"] += 1
                if pipe_info is not None:
                    pipe_info["zero_exec_count"] += 1
            else:
                total_emitted_dispatches += 1
                pair_rec.emitted_dispatches += 1
                if pipe_info is not None:
                    pipe_info["exec_count"] += 1

            _check_bound_groups_for_pipeline(pair_rec, pipe_info)

    # Flush any pending SetPipeline / SetComputePipeline at EOF.
    _flush_pending_render()
    _flush_pending_compute()

    # Flag created pipelines that were never even bound or executed.
    for pipe_info in created_pipelines.values():
        if (
            pipe_info["set_count"] == 0
            and pipe_info["exec_count"] == 0
            and pipe_info["zero_exec_count"] == 0
        ):
            pipe_info["pair"].reason_counts["pipeline_never_executed"] += 1

    reason_totals: Counter[str] = Counter()
    ranked_rows: list[dict[str, Any]] = []
    for rec in pair_records.values():
        skip_events = sum(
            count for reason, count in rec.reason_counts.items() if reason in _SKIP_EVENT_REASONS
        )
        binding_anomalies = sum(
            count
            for reason, count in rec.reason_counts.items()
            if reason in _BINDING_ANOMALY_REASONS
        )
        total_anomalies = skip_events + binding_anomalies
        if total_anomalies == 0:
            continue
        reason_totals.update(rec.reason_counts)
        top_reason = sorted(rec.reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        ranked_rows.append(
            {
                "kind": rec.kind,
                "shader_pair": rec.shader_pair,
                "vs_shader": rec.vs_shader,
                "ps_shader": rec.ps_shader,
                "cs_shader": rec.cs_shader,
                "pipeline_ids": sorted(rec.pipeline_ids),
                "pipeline_count": len(rec.pipeline_ids),
                "set_count": rec.set_count,
                "emitted_draws": rec.emitted_draws,
                "emitted_dispatches": rec.emitted_dispatches,
                "skip_event_count": skip_events,
                "binding_anomaly_count": binding_anomalies,
                "total_anomalies": total_anomalies,
                "top_reason": top_reason,
                "reason_counts": dict(sorted(rec.reason_counts.items())),
                "affected_bind_group_ids": sorted(rec.affected_bind_group_ids),
                "affected_bindings": sorted(rec.affected_bindings),
            }
        )

    ranked_rows.sort(
        key=lambda row: (
            -int(row["total_anomalies"]),
            -int(row["skip_event_count"]),
            -int(row["binding_anomaly_count"]),
            str(row["kind"]),
            str(row["shader_pair"]),
        )
    )
    for rank, row in enumerate(ranked_rows, start=1):
        row["rank"] = rank

    total_skip_events = sum(int(row["skip_event_count"]) for row in ranked_rows)
    total_binding_anomalies = sum(int(row["binding_anomaly_count"]) for row in ranked_rows)
    total_anomalies = total_skip_events + total_binding_anomalies + unbound_draws + unbound_dispatches

    return {
        "ok": total_anomalies == 0,
        "total_anomalies": total_anomalies,
        "total_skip_events": total_skip_events,
        "total_binding_anomalies": total_binding_anomalies,
        "total_pipelines": len(created_pipelines),
        "total_pipeline_pairs": len(pair_records),
        "affected_pipeline_pairs": len(ranked_rows),
        "emitted_draws": total_emitted_draws,
        "emitted_dispatches": total_emitted_dispatches,
        "unbound_draws": unbound_draws,
        "unbound_dispatches": unbound_dispatches,
        "reason_totals": dict(sorted(reason_totals.items())),
        "ranked_pairs": ranked_rows,
    }


def format_start_screen_skips_text(audit: dict[str, Any]) -> str:
    """Format an audit_start_screen_skips() dictionary as a sanitized human-readable report."""
    total = int(audit["total_anomalies"])
    status = (
        "CLEAN (0 anomalies)"
        if audit["ok"]
        else (
            f"ISSUES FOUND ({total} anomaly event(s) across "
            f"{audit['affected_pipeline_pairs']}/{audit['total_pipeline_pairs']} pipeline pair(s))"
        )
    )
    reasons_str = _format_op_histogram(audit.get("reason_totals"))
    lines: list[str] = [
        "Start-Screen Skips Audit:",
        f"  Status: {status}",
        (
            f"  Pipelines: {audit['total_pipelines']} | "
            f"Pairs: {audit['total_pipeline_pairs']} "
            f"(affected={audit['affected_pipeline_pairs']}) | "
            f"Emitted Draws: {audit['emitted_draws']} | "
            f"Emitted Dispatches: {audit['emitted_dispatches']} | "
            f"Unbound Draws: {audit['unbound_draws']} | "
            f"Unbound Dispatches: {audit['unbound_dispatches']}"
        ),
        (
            f"  Skip Events: {audit['total_skip_events']} | "
            f"Binding Anomalies: {audit['total_binding_anomalies']} | "
            f"Reasons: {reasons_str}"
        ),
    ]

    ranked = audit.get("ranked_pairs") or []
    if not ranked:
        lines.append("  (none)")
        return "\n".join(lines)

    lines.append(
        f"  {'Rank':>4s}  {'Kind':<7s}  {'Shader Pair (VS / PS or CS)':<39s}  "
        f"{'Pipes':>5s}  {'Draws':>5s}  {'Disp':>4s}  {'Skips':>5s}  "
        f"{'BindAnom':>8s}  {'Total':>5s}  {'Top Reason'}"
    )
    lines.append(
        f"  {'-' * 4}  {'-' * 7}  {'-' * 39}  "
        f"{'-' * 5}  {'-' * 5}  {'-' * 4}  {'-' * 5}  "
        f"{'-' * 8}  {'-' * 5}  {'-' * 24}"
    )
    for row in ranked:
        lines.append(
            f"  {int(row['rank']):4d}  {str(row['kind']):<7s}  {str(row['shader_pair']):<39s}  "
            f"{int(row['pipeline_count']):5d}  {int(row['emitted_draws']):5d}  "
            f"{int(row['emitted_dispatches']):4d}  {int(row['skip_event_count']):5d}  "
            f"{int(row['binding_anomaly_count']):8d}  {int(row['total_anomalies']):5d}  "
            f"{row['top_reason']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 30 FPS Frame-Budget Spike & Hotspot Analyzer (T67 / #1211)
# ---------------------------------------------------------------------------

_TIMING_LOG_EXTENSIONS: frozenset[str] = frozenset(
    {".json", ".jsonl", ".ndjson", ".log", ".txt"}
)

_EXPLICIT_CAUSE_ALIASES: dict[str, str] = {
    "pso_compile": "pso_compile",
    "pso": "pso_compile",
    "pipeline_compile": "pso_compile",
    "pipeline_creation": "pso_compile",
    "shader_compile": "pso_compile",
    "shader_module": "pso_compile",
    "compile": "pso_compile",
    "staging_upload": "staging_upload",
    "upload": "staging_upload",
    "staging": "staging_upload",
    "texture_upload": "staging_upload",
    "buffer_upload": "staging_upload",
    "write_texture": "staging_upload",
    "write_buffer": "staging_upload",
    "copy_upload": "staging_upload",
    "vram_upload": "staging_upload",
    "draw_burst": "draw_burst",
    "draw": "draw_burst",
    "draws": "draw_burst",
    "draw_calls": "draw_burst",
    "render_pass": "draw_burst",
    "geometry": "draw_burst",
    "compute_burst": "draw_burst",
    "dispatch_burst": "draw_burst",
    "rt_alloc": "rt_alloc",
    "render_target_alloc": "rt_alloc",
    "rt_pool": "rt_alloc",
    "texture_alloc": "rt_alloc",
    "resource_alloc": "rt_alloc",
    "alloc": "rt_alloc",
    "allocation": "rt_alloc",
    "resize": "rt_alloc",
    "swapchain_resize": "rt_alloc",
    "sync_stall": "sync_stall",
    "sync": "sync_stall",
    "stall": "sync_stall",
    "fence": "sync_stall",
    "fence_wait": "sync_stall",
    "readback": "sync_stall",
    "map_async": "sync_stall",
    "wait": "sync_stall",
    "present_wait": "sync_stall",
    "backpressure": "sync_stall",
    "ring_stall": "sync_stall",
    "gpu_wait": "sync_stall",
    "barrier": "sync_stall",
    "unknown": "unknown",
}

_CAUSE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "pso_compile",
        re.compile(
            r"(?:^|_)(?:pso|pipeline|compile|compiled|compiling|shader|wgsl|naga|tint|"
            r"create_render_pipeline\d*|create_compute_pipeline\d*|create_shader_module)(?:$|_)"
        ),
    ),
    (
        "staging_upload",
        re.compile(
            r"(?:^|_)(?:staging|upload|uploads|uploading|write_texture|write_buffer|"
            r"copy_buffer_to_texture|copy_buffer_to_buffer|texel_upload|slab|bink_upload)(?:$|_)"
        ),
    ),
    (
        "rt_alloc",
        re.compile(
            r"(?:^|_)(?:rt_alloc|rt_pool|render_target|create_texture\d*|create_buffer|"
            r"alloc_rt|reallocate|alloc|allocation|resize|depth_target|create_view)(?:$|_)"
        ),
    ),
    (
        "sync_stall",
        re.compile(
            r"(?:^|_)(?:sync|stall|fence|map_async|readback|submitted_work_done|"
            r"present_wait|backpressure|ring_wait|barrier|wait|flush|lock|bind_group_setup)(?:$|_)"
        ),
    ),
    (
        "draw_burst",
        re.compile(
            r"(?:^|_)(?:draw|draws|draw_indexed|dispatch|dispatch_compute|render_pass|"
            r"compute_pass|shadow|gbuffer|lighting|postprocess|post_process|geometry|"
            r"opaque|transparent|particles|terrain|foliage|sky|hud|ui|blit|composite|"
            r"bink_draw|present|pass)(?:$|_)"
        ),
    ),
)

_CAMEL_BOUNDARY_RE = re.compile(r"([a-z0-9])([A-Z])")
_TOKEN_SEP_RE = re.compile(r"[^a-z0-9]+")
_SHADER_SOURCE_RE = re.compile(
    r"(?:@vertex|@fragment|@compute|@group\s*\(|@binding\s*\(|@location\s*\(|"
    r"@builtin\s*\(|\bvar<|\bfn\s+[A-Za-z_]\w*\s*\(|\bstruct\s+[A-Za-z_]\w*\s*\{|"
    r"\bcbuffer\b|\bTexture2D\b|#version\b|#pragma\b|[{};])"
)
_HOST_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|\\\\[^\\]+\\|file://|(?:^|[\s'\"(])/(?:Users|home|var|tmp|etc|opt|mnt|Volumes|root|private|proc)/|"
    r"(?:^|[\s'\"(])(?:Users|home|local-data|work)[\\/]|"
    r"\.(?:wgsl|rpf|xvd|xvc|ttpf|awc|pg6trace|bin|exe|dll|pdb)\b)",
    re.IGNORECASE,
)
_CHROME_FRAME_NAME_RE = re.compile(
    r"^(?:frame|gpu_?frame|render_?frame|main_?frame|present_?frame|begin_?frame)$",
    re.IGNORECASE,
)


def _validate_budget_and_top_n(budget_ms: float, top_n: int) -> tuple[float, int]:
    if isinstance(budget_ms, bool) or not isinstance(budget_ms, (int, float)):
        raise ValueError(f"budget_ms must be a positive finite number, got {budget_ms!r}")
    budget = float(budget_ms)
    if not math.isfinite(budget) or budget <= 0.0:
        raise ValueError(f"budget_ms must be a positive finite number, got {budget_ms!r}")
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError(f"top_n must be a positive integer, got {top_n!r}")
    return budget, top_n


def _as_non_negative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0:
        return None
    return fval


def _as_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0.0 and value.is_integer():
        return int(value)
    return None


def _normalize_identifier_tokens(text: str) -> str:
    spaced = _CAMEL_BOUNDARY_RE.sub(r"\1_\2", text)
    return _TOKEN_SEP_RE.sub("_", spaced.lower()).strip("_")


def _sanitize_marker_name(raw: Any, fallback: str = "unnamed_pass") -> str:
    """Sanitize pass/marker names so host paths and raw shader source can never leak."""
    if not isinstance(raw, str):
        return fallback
    trimmed = raw.strip()
    if not trimmed:
        return fallback
    if _SHADER_SOURCE_RE.search(trimmed):
        return "shader_event_redacted"
    if _HOST_PATH_RE.search(trimmed) or "\\" in trimmed or trimmed.startswith("/"):
        return "path_redacted"
    if ".." in trimmed:
        return "path_redacted"
    cleaned = re.sub(r"[\x00-\x1f\x7f]+", " ", trimmed)
    cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
    cleaned = re.sub(r"[^A-Za-z0-9_.:#-]+", "_", cleaned).strip("_")
    if not cleaned:
        return fallback
    return cleaned[:64]


def _sanitize_phase_name(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    trimmed = raw.strip()
    if not trimmed or _SHADER_SOURCE_RE.search(trimmed) or _HOST_PATH_RE.search(trimmed):
        return None
    if "\\" in trimmed or "/" in trimmed or ".." in trimmed:
        return None
    norm = _TOKEN_SEP_RE.sub("_", trimmed.lower()).strip("_")
    if not norm:
        return None
    return norm[:32]


def _nearest_rank_percentile(sorted_values: Sequence[float], fraction: float) -> float:
    count = len(sorted_values)
    if count <= 0:
        return 0.0
    idx = min(count - 1, max(0, math.ceil(fraction * count) - 1))
    return float(sorted_values[idx])


def classify_spike_cause(
    marker: str | None = None,
    *,
    explicit_cause: str | None = None,
    counters: dict[str, Any] | None = None,
) -> str:
    """Classify a pass/event marker or frame counter set into a canonical spike cause.

    Canonical causes: `pso_compile`, `staging_upload`, `draw_burst`, `rt_alloc`,
    `sync_stall`, `unknown`.
    """
    if isinstance(explicit_cause, str) and explicit_cause.strip():
        norm_explicit = _normalize_identifier_tokens(explicit_cause)
        mapped = _EXPLICIT_CAUSE_ALIASES.get(norm_explicit)
        if mapped is not None and mapped != "unknown":
            return mapped

    if isinstance(marker, str) and marker.strip():
        if _SHADER_SOURCE_RE.search(marker):
            return "pso_compile"
        norm_marker = _normalize_identifier_tokens(marker)
        if norm_marker in _EXPLICIT_CAUSE_ALIASES and _EXPLICIT_CAUSE_ALIASES[norm_marker] != "unknown":
            return _EXPLICIT_CAUSE_ALIASES[norm_marker]
        for cause, pattern in _CAUSE_PATTERNS:
            if pattern.search(norm_marker):
                return cause

    if counters:
        # 1. Explicit per-cause millisecond breakdowns win when provided.
        ms_scores: dict[str, float] = {
            "pso_compile": (
                (_as_non_negative_float(counters.get("pso_compile_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("compile_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("shader_compile_ms")) or 0.0)
            ),
            "staging_upload": (
                (_as_non_negative_float(counters.get("staging_upload_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("upload_ms")) or 0.0)
            ),
            "rt_alloc": (
                (_as_non_negative_float(counters.get("rt_alloc_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("alloc_ms")) or 0.0)
            ),
            "sync_stall": (
                (_as_non_negative_float(counters.get("sync_stall_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("stall_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("fence_wait_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("readback_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("wait_ms")) or 0.0)
            ),
            "draw_burst": (
                (_as_non_negative_float(counters.get("draw_burst_ms")) or 0.0)
                + (_as_non_negative_float(counters.get("draw_ms")) or 0.0)
            ),
        }
        best_ms_cause = max(
            SPIKE_CAUSES[:-1],
            key=lambda c: (ms_scores[c], -SPIKE_CAUSES.index(c)),
        )
        if ms_scores[best_ms_cause] > 0.0:
            return best_ms_cause

        # 2. Weighted counter heuristic fallback.
        pso_compiles = (
            (_as_non_negative_int(counters.get("pso_compiles")) or 0)
            + (_as_non_negative_int(counters.get("pipelines_compiled")) or 0)
            + (_as_non_negative_int(counters.get("pipelines_created")) or 0)
        )
        shaders_compiled = (
            (_as_non_negative_int(counters.get("shaders_compiled")) or 0)
            + (_as_non_negative_int(counters.get("shader_modules_created")) or 0)
        )
        upload_bytes = (
            (_as_non_negative_int(counters.get("upload_bytes")) or 0)
            + (_as_non_negative_int(counters.get("staging_bytes")) or 0)
            + (_as_non_negative_int(counters.get("staging_upload_bytes")) or 0)
        )
        staging_uploads = (
            (_as_non_negative_int(counters.get("staging_uploads")) or 0)
            + (_as_non_negative_int(counters.get("upload_count")) or 0)
        )
        rt_allocs = (
            (_as_non_negative_int(counters.get("rt_allocs")) or 0)
            + (_as_non_negative_int(counters.get("render_targets_created")) or 0)
            + (_as_non_negative_int(counters.get("textures_created")) or 0)
        )
        buffer_allocs = _as_non_negative_int(counters.get("buffer_allocs")) or 0
        sync_stalls = (
            (_as_non_negative_int(counters.get("sync_stalls")) or 0)
            + (_as_non_negative_int(counters.get("fence_waits")) or 0)
        )
        draw_calls = _as_non_negative_int(counters.get("draw_calls")) or 0
        compute_dispatches = _as_non_negative_int(counters.get("compute_dispatches")) or 0

        weighted_scores: dict[str, float] = {
            "pso_compile": pso_compiles * 8.0 + shaders_compiled * 3.5,
            "staging_upload": (upload_bytes / (1024.0 * 1024.0)) * 2.5 + staging_uploads * 0.5,
            "rt_alloc": rt_allocs * 4.5 + buffer_allocs * 0.15,
            "sync_stall": sync_stalls * 5.0,
            "draw_burst": draw_calls * 0.04 + compute_dispatches * 0.15,
        }
        best_cnt_cause = max(
            SPIKE_CAUSES[:-1],
            key=lambda c: (weighted_scores[c], -SPIKE_CAUSES.index(c)),
        )
        if weighted_scores[best_cnt_cause] > 0.0:
            return best_cnt_cause

    return "unknown"


@dataclass(frozen=True)
class _PassSample:
    name: str
    duration_ms: float
    cause: str


@dataclass(frozen=True)
class _FrameSample:
    frame_index: int
    frame_ms: float
    cause: str
    top_pass: str | None
    top_pass_ms: float
    phase: str | None
    draw_calls: int
    compute_dispatches: int
    pso_compiles: int
    upload_bytes: int
    rt_allocs: int
    sync_stall_ms: float
    passes: tuple[_PassSample, ...]


def _extract_duration_ms(obj: dict[str, Any], *, allow_chrome_dur_us: bool = False) -> float | None:
    for key in (
        "frame_ms",
        "frameMs",
        "duration_ms",
        "durationMs",
        "frame_time_ms",
        "frameTimeMs",
        "dt_ms",
        "dtMs",
        "total_ms",
        "totalMs",
        "elapsed_ms",
        "elapsedMs",
        "gpu_ms",
        "gpuMs",
        "time_ms",
        "timeMs",
        "ms",
    ):
        if key in obj and obj[key] is not None:
            return _as_non_negative_float(obj[key])
    for key in ("frame_us", "duration_us", "durationUs", "us"):
        if key in obj and obj[key] is not None:
            us = _as_non_negative_float(obj[key])
            return None if us is None else us / 1000.0
    if allow_chrome_dur_us and "dur" in obj and obj["dur"] is not None:
        us = _as_non_negative_float(obj["dur"])
        return None if us is None else us / 1000.0
    for key in ("frame_ns", "duration_ns", "durationNs", "ns"):
        if key in obj and obj[key] is not None:
            ns = _as_non_negative_float(obj[key])
            return None if ns is None else ns / 1_000_000.0
    for start_k, end_k in (("start_ms", "end_ms"), ("startMs", "endMs")):
        if start_k in obj and end_k in obj:
            s = _as_non_negative_float(obj[start_k])
            e = _as_non_negative_float(obj[end_k])
            if s is not None and e is not None and e >= s:
                return e - s
            return None
    return None


def _extract_frame_counters(obj: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    int_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("draw_calls", ("draw_calls", "drawCalls", "draws")),
        ("compute_dispatches", ("compute_dispatches", "computeDispatches", "dispatches")),
        ("pso_compiles", ("pso_compiles", "psoCompiles", "pipelines_compiled", "pipelinesCompiled", "pipelines_created", "pipelinesCreated")),
        ("shaders_compiled", ("shaders_compiled", "shadersCompiled", "shader_modules_created", "shaderModulesCreated")),
        ("upload_bytes", ("upload_bytes", "uploadBytes", "staging_bytes", "stagingBytes", "staging_upload_bytes", "stagingUploadBytes")),
        ("staging_uploads", ("staging_uploads", "stagingUploads", "upload_count", "uploadCount")),
        ("rt_allocs", ("rt_allocs", "rtAllocs", "render_targets_created", "renderTargetsCreated", "textures_created", "texturesCreated")),
        ("buffer_allocs", ("buffer_allocs", "bufferAllocs", "buffers_created", "buffersCreated")),
        ("sync_stalls", ("sync_stalls", "syncStalls", "fence_waits", "fenceWaits")),
    )
    for canonical, keys in int_aliases:
        for k in keys:
            if k in obj and obj[k] is not None:
                v = _as_non_negative_int(obj[k])
                if v is not None:
                    out[canonical] = v
                    break

    float_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("pso_compile_ms", ("pso_compile_ms", "psoCompileMs", "compile_ms", "compileMs", "shader_compile_ms")),
        ("staging_upload_ms", ("staging_upload_ms", "stagingUploadMs", "upload_ms", "uploadMs")),
        ("rt_alloc_ms", ("rt_alloc_ms", "rtAllocMs", "alloc_ms", "allocMs")),
        ("sync_stall_ms", ("sync_stall_ms", "syncStallMs", "stall_ms", "stallMs", "fence_wait_ms", "fenceWaitMs", "readback_ms", "readbackMs", "wait_ms")),
        ("draw_ms", ("draw_burst_ms", "drawBurstMs", "draw_ms", "drawMs")),
    )
    for canonical, keys in float_aliases:
        for k in keys:
            if k in obj and obj[k] is not None:
                v = _as_non_negative_float(obj[k])
                if v is not None:
                    out[canonical] = v
                    break
    return out


def _parse_pass_entries(
    raw_passes: Any,
    diagnostics: list[dict[str, Any]],
    record_index: int,
) -> tuple[list[_PassSample], int]:
    samples: list[_PassSample] = []
    malformed = 0
    if raw_passes is None:
        return samples, 0

    if isinstance(raw_passes, dict):
        for raw_name, raw_val in raw_passes.items():
            if isinstance(raw_val, dict):
                dur = _extract_duration_ms(raw_val, allow_chrome_dur_us=True)
                if dur is None:
                    # Also accept GpuPassTimer passStats shape ({avgMs, maxMs, lastMs}).
                    for k in ("lastMs", "avgMs", "maxMs", "totalMs"):
                        if k in raw_val:
                            dur = _as_non_negative_float(raw_val[k])
                            if dur is not None:
                                break
                if dur is None:
                    malformed += 1
                    diagnostics.append(
                        {
                            "severity": "warning",
                            "code": "invalid_pass_duration",
                            "record_index": record_index,
                            "message": f"Frame record {record_index} has a pass entry with an invalid duration.",
                        }
                    )
                    continue
                explicit = raw_val.get("cause") or raw_val.get("category") or raw_val.get("bottleneck")
                cause = classify_spike_cause(
                    str(raw_name),
                    explicit_cause=explicit if isinstance(explicit, str) else None,
                    counters=_extract_frame_counters(raw_val),
                )
                samples.append(
                    _PassSample(
                        name=_sanitize_marker_name(str(raw_name)),
                        duration_ms=dur,
                        cause=cause,
                    )
                )
            else:
                dur = _as_non_negative_float(raw_val)
                if dur is None:
                    malformed += 1
                    diagnostics.append(
                        {
                            "severity": "warning",
                            "code": "invalid_pass_duration",
                            "record_index": record_index,
                            "message": f"Frame record {record_index} has a pass entry with an invalid duration.",
                        }
                    )
                    continue
                cause = classify_spike_cause(str(raw_name))
                samples.append(
                    _PassSample(
                        name=_sanitize_marker_name(str(raw_name)),
                        duration_ms=dur,
                        cause=cause,
                    )
                )
        return samples, malformed

    if not isinstance(raw_passes, list):
        return samples, 0

    for item in raw_passes:
        if not isinstance(item, dict):
            malformed += 1
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "malformed_pass_entry",
                    "record_index": record_index,
                    "message": f"Frame record {record_index} contains a non-object pass entry.",
                }
            )
            continue
        dur = _extract_duration_ms(item, allow_chrome_dur_us=True)
        if dur is None:
            malformed += 1
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "invalid_pass_duration",
                    "record_index": record_index,
                    "message": f"Frame record {record_index} contains a pass entry with a missing or invalid duration.",
                }
            )
            continue
        raw_name = (
            item.get("name")
            or item.get("label")
            or item.get("pass")
            or item.get("op")
            or item.get("marker")
            or item.get("event")
            or ""
        )
        raw_name_str = raw_name if isinstance(raw_name, str) else ""
        explicit = item.get("cause") or item.get("category") or item.get("bottleneck") or item.get("cat")
        cause = classify_spike_cause(
            raw_name_str,
            explicit_cause=explicit if isinstance(explicit, str) else None,
            counters=_extract_frame_counters(item),
        )
        samples.append(
            _PassSample(
                name=_sanitize_marker_name(raw_name_str),
                duration_ms=dur,
                cause=cause,
            )
        )
    return samples, malformed


def _build_frame_sample_from_dict(
    obj: dict[str, Any],
    fallback_index: int,
    diagnostics: list[dict[str, Any]],
) -> tuple[_FrameSample | None, int]:
    malformed = 0
    idx_val: int | None = None
    for k in ("frame_index", "frameIndex", "frame", "frame_id", "frameId", "seq", "index"):
        if k in obj and obj[k] is not None:
            idx_val = _as_non_negative_int(obj[k])
            if idx_val is not None:
                break
    frame_index = fallback_index if idx_val is None else idx_val

    raw_passes = None
    for k in ("passes", "events", "markers", "spans", "ops"):
        if k in obj and obj[k] is not None:
            raw_passes = obj[k]
            break
    passes, pass_malformed = _parse_pass_entries(raw_passes, diagnostics, fallback_index)
    malformed += pass_malformed

    frame_ms = _extract_duration_ms(obj, allow_chrome_dur_us=False)
    if frame_ms is None:
        if passes:
            frame_ms = sum(p.duration_ms for p in passes)
        else:
            malformed += 1
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "invalid_frame_duration",
                    "record_index": fallback_index,
                    "message": f"Frame record {fallback_index} is missing a valid non-negative duration.",
                }
            )
            return None, malformed

    counters = _extract_frame_counters(obj)
    explicit = obj.get("cause") or obj.get("category") or obj.get("bottleneck") or obj.get("spike_cause")
    explicit_str = explicit if isinstance(explicit, str) else None
    raw_marker = obj.get("marker") or obj.get("label") or obj.get("name") or obj.get("op")
    marker_str = raw_marker if isinstance(raw_marker, str) else None

    # Determine dominant cause: explicit cause > timed passes > marker/counters > unknown.
    cause = "unknown"
    if explicit_str:
        cause = classify_spike_cause(explicit_cause=explicit_str)
    if cause == "unknown" and passes:
        cause_ms: dict[str, float] = {c: 0.0 for c in SPIKE_CAUSES}
        for p in passes:
            cause_ms[p.cause] += p.duration_ms
        best_pass_cause = max(
            SPIKE_CAUSES[:-1],
            key=lambda c: (cause_ms[c], -SPIKE_CAUSES.index(c)),
        )
        if cause_ms[best_pass_cause] > 0.0:
            cause = best_pass_cause
    if cause == "unknown":
        cause = classify_spike_cause(marker_str, counters=counters)

    top_pass: str | None = None
    top_pass_ms = 0.0
    if passes:
        best_p = max(passes, key=lambda p: (p.duration_ms, -SPIKE_CAUSES.index(p.cause), p.name))
        top_pass = best_p.name
        top_pass_ms = best_p.duration_ms

    return (
        _FrameSample(
            frame_index=frame_index,
            frame_ms=frame_ms,
            cause=cause,
            top_pass=top_pass,
            top_pass_ms=top_pass_ms,
            phase=_sanitize_phase_name(obj.get("phase")),
            draw_calls=int(counters.get("draw_calls", 0)),
            compute_dispatches=int(counters.get("compute_dispatches", 0)),
            pso_compiles=int(counters.get("pso_compiles", 0)),
            upload_bytes=int(counters.get("upload_bytes", 0)),
            rt_allocs=int(counters.get("rt_allocs", 0)),
            sync_stall_ms=float(counters.get("sync_stall_ms", 0.0)),
            passes=tuple(passes),
        ),
        malformed,
    )


def _looks_like_chrome_events(items: list[Any]) -> bool:
    for item in items:
        if isinstance(item, dict) and "ph" in item and ("ts" in item or "dur" in item):
            return True
    return False


def _parse_chrome_trace_events(
    events: list[Any],
    diagnostics: list[dict[str, Any]],
) -> tuple[list[_FrameSample], int]:
    malformed = 0
    completed: list[tuple[float, float, str, str, dict[str, Any]]] = []
    open_stacks: dict[tuple[Any, Any, str], list[tuple[float, str, dict[str, Any]]]] = {}

    for i, ev in enumerate(events):
        if not isinstance(ev, dict):
            malformed += 1
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "malformed_trace_event",
                    "record_index": i,
                    "message": f"Chrome trace event {i} is not a JSON object.",
                }
            )
            continue
        ph = str(ev.get("ph", "")).upper()
        name = ev.get("name")
        name_str = name if isinstance(name, str) else ""
        cat = ev.get("cat")
        cat_str = cat if isinstance(cat, str) else ""
        args = ev.get("args") if isinstance(ev.get("args"), dict) else {}

        if ph == "X":
            ts = _as_non_negative_float(ev.get("ts"))
            dur = _as_non_negative_float(ev.get("dur"))
            if dur is None:
                malformed += 1
                diagnostics.append(
                    {
                        "severity": "warning",
                        "code": "invalid_event_duration",
                        "record_index": i,
                        "message": f"Chrome trace event {i} has an invalid 'dur' value.",
                    }
                )
                continue
            start_ms = 0.0 if ts is None else ts / 1000.0
            completed.append((start_ms, dur / 1000.0, name_str, cat_str, args))
        elif ph == "B":
            ts = _as_non_negative_float(ev.get("ts"))
            if ts is None:
                malformed += 1
                continue
            key = (ev.get("pid", 0), ev.get("tid", 0), name_str)
            open_stacks.setdefault(key, []).append((ts / 1000.0, cat_str, args))
        elif ph == "E":
            ts = _as_non_negative_float(ev.get("ts"))
            if ts is None:
                malformed += 1
                continue
            key = (ev.get("pid", 0), ev.get("tid", 0), name_str)
            stack = open_stacks.get(key)
            if stack:
                start_ms, b_cat, b_args = stack.pop()
                end_ms = ts / 1000.0
                if end_ms >= start_ms:
                    merged_args = dict(b_args)
                    merged_args.update(args)
                    completed.append((start_ms, end_ms - start_ms, name_str, b_cat or cat_str, merged_args))

    if not completed:
        return [], malformed

    frame_spans: list[tuple[int, float, float, str, dict[str, Any]]] = []
    pass_events: list[tuple[float, float, str, str, dict[str, Any]]] = []

    for start_ms, dur_ms, name_str, cat_str, args in completed:
        is_frame = bool(
            args.get("is_frame")
            or _CHROME_FRAME_NAME_RE.match(name_str.strip())
            or "frame" in _normalize_identifier_tokens(cat_str).split("_")
        )
        if is_frame:
            idx = _as_non_negative_int(args.get("frame_index") if "frame_index" in args else args.get("frame"))
            frame_spans.append((len(frame_spans) if idx is None else idx, start_ms, dur_ms, cat_str, args))
        else:
            pass_events.append((start_ms, dur_ms, name_str, cat_str, args))

    frames: list[_FrameSample] = []
    if frame_spans:
        span_passes: list[list[_PassSample]] = [[] for _ in frame_spans]
        index_to_pos = {span[0]: pos for pos, span in enumerate(frame_spans)}
        for start_ms, dur_ms, name_str, cat_str, args in pass_events:
            explicit = args.get("cause") or args.get("category") or cat_str
            cause = classify_spike_cause(
                name_str,
                explicit_cause=explicit if isinstance(explicit, str) else None,
                counters=_extract_frame_counters(args),
            )
            ps = _PassSample(name=_sanitize_marker_name(name_str), duration_ms=dur_ms, cause=cause)
            target_pos: int | None = None
            arg_idx = _as_non_negative_int(args.get("frame_index") if "frame_index" in args else args.get("frame"))
            if arg_idx is not None and arg_idx in index_to_pos:
                target_pos = index_to_pos[arg_idx]
            else:
                for pos, (_, f_start, f_dur, _, _) in enumerate(frame_spans):
                    if f_start <= start_ms <= (f_start + f_dur + 1e-6):
                        target_pos = pos
                        break
            if target_pos is not None:
                span_passes[target_pos].append(ps)

        for pos, (f_idx, _, f_dur, _, args) in enumerate(frame_spans):
            passes = span_passes[pos]
            counters = _extract_frame_counters(args)
            explicit = args.get("cause") or args.get("bottleneck")
            cause = "unknown"
            if isinstance(explicit, str) and explicit.strip():
                cause = classify_spike_cause(explicit_cause=explicit)
            if cause == "unknown" and passes:
                cause_ms: dict[str, float] = {c: 0.0 for c in SPIKE_CAUSES}
                for p in passes:
                    cause_ms[p.cause] += p.duration_ms
                best_c = max(SPIKE_CAUSES[:-1], key=lambda c: (cause_ms[c], -SPIKE_CAUSES.index(c)))
                if cause_ms[best_c] > 0.0:
                    cause = best_c
            if cause == "unknown":
                cause = classify_spike_cause(counters=counters)
            top_pass = None
            top_pass_ms = 0.0
            if passes:
                best_p = max(passes, key=lambda p: (p.duration_ms, -SPIKE_CAUSES.index(p.cause), p.name))
                top_pass = best_p.name
                top_pass_ms = best_p.duration_ms
            frames.append(
                _FrameSample(
                    frame_index=f_idx,
                    frame_ms=f_dur,
                    cause=cause,
                    top_pass=top_pass,
                    top_pass_ms=top_pass_ms,
                    phase=_sanitize_phase_name(args.get("phase")),
                    draw_calls=int(counters.get("draw_calls", 0)),
                    compute_dispatches=int(counters.get("compute_dispatches", 0)),
                    pso_compiles=int(counters.get("pso_compiles", 0)),
                    upload_bytes=int(counters.get("upload_bytes", 0)),
                    rt_allocs=int(counters.get("rt_allocs", 0)),
                    sync_stall_ms=float(counters.get("sync_stall_ms", 0.0)),
                    passes=tuple(passes),
                )
            )
        return frames, malformed

    # No explicit frame events: group by args.frame_index / args.frame if available.
    grouped: dict[int, list[tuple[float, str, str, dict[str, Any]]]] = {}
    ungrouped: list[tuple[float, str, str, dict[str, Any]]] = []
    for _, dur_ms, name_str, cat_str, args in pass_events:
        arg_idx = _as_non_negative_int(args.get("frame_index") if "frame_index" in args else args.get("frame"))
        if arg_idx is not None:
            grouped.setdefault(arg_idx, []).append((dur_ms, name_str, cat_str, args))
        else:
            ungrouped.append((dur_ms, name_str, cat_str, args))

    if grouped:
        for f_idx in sorted(grouped):
            items = grouped[f_idx]
            passes = []
            for dur_ms, name_str, cat_str, args in items:
                explicit = args.get("cause") or args.get("category") or cat_str
                cause = classify_spike_cause(
                    name_str,
                    explicit_cause=explicit if isinstance(explicit, str) else None,
                    counters=_extract_frame_counters(args),
                )
                passes.append(_PassSample(name=_sanitize_marker_name(name_str), duration_ms=dur_ms, cause=cause))
            total_ms = sum(p.duration_ms for p in passes)
            cause_ms = {c: 0.0 for c in SPIKE_CAUSES}
            for p in passes:
                cause_ms[p.cause] += p.duration_ms
            best_c = max(SPIKE_CAUSES[:-1], key=lambda c: (cause_ms[c], -SPIKE_CAUSES.index(c)))
            cause = best_c if cause_ms[best_c] > 0.0 else "unknown"
            best_p = max(passes, key=lambda p: (p.duration_ms, -SPIKE_CAUSES.index(p.cause), p.name))
            frames.append(
                _FrameSample(
                    frame_index=f_idx,
                    frame_ms=total_ms,
                    cause=cause,
                    top_pass=best_p.name,
                    top_pass_ms=best_p.duration_ms,
                    phase=None,
                    draw_calls=0,
                    compute_dispatches=0,
                    pso_compiles=0,
                    upload_bytes=0,
                    rt_allocs=0,
                    sync_stall_ms=0.0,
                    passes=tuple(passes),
                )
            )
        return frames, malformed

    for idx, (dur_ms, name_str, cat_str, args) in enumerate(ungrouped):
        explicit = args.get("cause") or args.get("category") or cat_str
        cause = classify_spike_cause(
            name_str,
            explicit_cause=explicit if isinstance(explicit, str) else None,
            counters=_extract_frame_counters(args),
        )
        sanitized = _sanitize_marker_name(name_str)
        ps = _PassSample(name=sanitized, duration_ms=dur_ms, cause=cause)
        frames.append(
            _FrameSample(
                frame_index=idx,
                frame_ms=dur_ms,
                cause=cause,
                top_pass=sanitized,
                top_pass_ms=dur_ms,
                phase=_sanitize_phase_name(args.get("phase")),
                draw_calls=0,
                compute_dispatches=0,
                pso_compiles=0,
                upload_bytes=0,
                rt_allocs=0,
                sync_stall_ms=0.0,
                passes=(ps,),
            )
        )
    return frames, malformed


def _parse_frame_records_from_obj(
    payload: Any,
    diagnostics: list[dict[str, Any]],
) -> tuple[list[_FrameSample], int]:
    frames: list[_FrameSample] = []
    malformed = 0

    if isinstance(payload, dict):
        if "traceEvents" in payload and isinstance(payload["traceEvents"], list):
            return _parse_chrome_trace_events(payload["traceEvents"], diagnostics)
        for key in ("frames", "frame_timings", "frameTimings", "frame_times_ms", "frameTimesMs", "timings", "samples"):
            if key in payload:
                sub = payload[key]
                if isinstance(sub, list):
                    return _parse_frame_records_from_obj(sub, diagnostics)
                if isinstance(sub, dict):
                    # Support summary-only dict or invalid container gracefully.
                    break
        # Single frame dict?
        if _extract_duration_ms(payload, allow_chrome_dur_us=False) is not None or any(
            k in payload for k in ("passes", "events", "markers", "spans")
        ):
            sample, bad = _build_frame_sample_from_dict(payload, 0, diagnostics)
            malformed += bad
            if sample is not None:
                frames.append(sample)
            return frames, malformed
        if not payload:
            return [], 0
        malformed += 1
        diagnostics.append(
            {
                "severity": "warning",
                "code": "unrecognized_timing_payload",
                "record_index": None,
                "message": "JSON object does not contain a recognized frames or traceEvents collection.",
            }
        )
        return [], malformed

    if isinstance(payload, list):
        if _looks_like_chrome_events(payload):
            return _parse_chrome_trace_events(payload, diagnostics)
        for idx, item in enumerate(payload):
            if isinstance(item, dict):
                sample, bad = _build_frame_sample_from_dict(item, idx, diagnostics)
                malformed += bad
                if sample is not None:
                    frames.append(sample)
            else:
                dur = _as_non_negative_float(item)
                if dur is None:
                    malformed += 1
                    diagnostics.append(
                        {
                            "severity": "warning",
                            "code": "invalid_frame_duration",
                            "record_index": idx,
                            "message": f"Frame entry {idx} is not a valid non-negative duration.",
                        }
                    )
                    continue
                frames.append(
                    _FrameSample(
                        frame_index=idx,
                        frame_ms=dur,
                        cause="unknown",
                        top_pass=None,
                        top_pass_ms=0.0,
                        phase=None,
                        draw_calls=0,
                        compute_dispatches=0,
                        pso_compiles=0,
                        upload_bytes=0,
                        rt_allocs=0,
                        sync_stall_ms=0.0,
                        passes=(),
                    )
                )
        return frames, malformed

    malformed += 1
    diagnostics.append(
        {
            "severity": "error",
            "code": "unsupported_timing_payload",
            "record_index": None,
            "message": "Top-level timing payload must be a JSON object or array.",
        }
    )
    return [], malformed


def _parse_timing_text(
    text: str,
    diagnostics: list[dict[str, Any]],
) -> tuple[list[_FrameSample], int]:
    stripped = text.lstrip("\ufeff").strip()
    if not stripped:
        return [], 0

    try:
        parsed = json.loads(stripped)
        return _parse_frame_records_from_obj(parsed, diagnostics)
    except json.JSONDecodeError:
        pass

    # Fallback: parse as NDJSON / JSON Lines (one JSON object/number per non-empty line).
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    frames: list[_FrameSample] = []
    malformed = 0
    any_valid_json_line = False
    for line_no, line in enumerate(lines, start=1):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            diagnostics.append(
                {
                    "severity": "warning",
                    "code": "malformed_ndjson_line",
                    "record_index": line_no - 1,
                    "message": f"Line {line_no} could not be parsed as JSON.",
                }
            )
            continue
        any_valid_json_line = True
        if isinstance(obj, dict):
            sample, bad = _build_frame_sample_from_dict(obj, len(frames), diagnostics)
            malformed += bad
            if sample is not None:
                frames.append(sample)
        else:
            dur = _as_non_negative_float(obj)
            if dur is None:
                malformed += 1
                diagnostics.append(
                    {
                        "severity": "warning",
                        "code": "invalid_frame_duration",
                        "record_index": line_no - 1,
                        "message": f"Line {line_no} does not contain a valid frame duration.",
                    }
                )
            else:
                frames.append(
                    _FrameSample(
                        frame_index=len(frames),
                        frame_ms=dur,
                        cause="unknown",
                        top_pass=None,
                        top_pass_ms=0.0,
                        phase=None,
                        draw_calls=0,
                        compute_dispatches=0,
                        pso_compiles=0,
                        upload_bytes=0,
                        rt_allocs=0,
                        sync_stall_ms=0.0,
                        passes=(),
                    )
                )

    if not any_valid_json_line:
        diagnostics.clear()
        diagnostics.append(
            {
                "severity": "error",
                "code": "malformed_json",
                "record_index": None,
                "message": "Input could not be parsed as JSON or NDJSON.",
            }
        )
        return [], max(1, malformed)

    return frames, malformed


def _summarize_frame_samples(
    samples: Sequence[_FrameSample],
    *,
    budget_ms: float,
    top_n: int,
    source_kind: str,
    timing_source: str,
    malformed_records: int,
    diagnostics: Sequence[dict[str, Any]],
    strict: bool = False,
) -> dict[str, Any]:
    total_frames = len(samples)
    sorted_ms = sorted(s.frame_ms for s in samples)

    within = sum(1 for s in samples if s.frame_ms <= budget_ms)
    over = total_frames - within
    compliance_rate = round(within / total_frames, 6) if total_frames > 0 else 0.0

    avg_ms = round(sum(sorted_ms) / total_frames, 3) if total_frames > 0 else 0.0
    min_ms = round(sorted_ms[0], 3) if total_frames > 0 else 0.0
    p50_ms = round(_nearest_rank_percentile(sorted_ms, 0.50), 3)
    p95_ms = round(_nearest_rank_percentile(sorted_ms, 0.95), 3)
    p99_ms = round(_nearest_rank_percentile(sorted_ms, 0.99), 3)
    max_ms = round(sorted_ms[-1], 3) if total_frames > 0 else 0.0

    over_excess_values = [max(0.0, s.frame_ms - budget_ms) for s in samples if s.frame_ms > budget_ms]
    total_over_budget_ms = round(sum(over_excess_values), 3)
    max_over_budget_ms = round(max(over_excess_values), 3) if over_excess_values else 0.0

    # Cause counts & breakdown over all over-budget (spike) frames.
    spike_by_cause: dict[str, list[_FrameSample]] = {c: [] for c in SPIKE_CAUSES}
    for s in samples:
        if s.frame_ms > budget_ms:
            c = s.cause if s.cause in spike_by_cause else "unknown"
            spike_by_cause[c].append(s)

    spike_causes: dict[str, int] = {c: len(spike_by_cause[c]) for c in SPIKE_CAUSES}
    cause_breakdown: dict[str, dict[str, Any]] = {}
    for c in SPIKE_CAUSES:
        group = spike_by_cause[c]
        cnt = len(group)
        if cnt > 0:
            group_ms = [s.frame_ms for s in group]
            group_excess = [max(0.0, s.frame_ms - budget_ms) for s in group]
            tot_c_ms = sum(group_ms)
            cause_breakdown[c] = {
                "frame_count": cnt,
                "share_of_spikes": round(cnt / over, 6) if over > 0 else 0.0,
                "total_ms": round(tot_c_ms, 3),
                "avg_ms": round(tot_c_ms / cnt, 3),
                "max_ms": round(max(group_ms), 3),
                "over_budget_ms": round(sum(group_excess), 3),
            }
        else:
            cause_breakdown[c] = {
                "frame_count": 0,
                "share_of_spikes": 0.0,
                "total_ms": 0.0,
                "avg_ms": 0.0,
                "max_ms": 0.0,
                "over_budget_ms": 0.0,
            }

    # Top N slowest frames (deterministic tie-break: frame_index ascending).
    ranked_frames = sorted(samples, key=lambda s: (-s.frame_ms, s.frame_index))[:top_n]
    top_slow_frames: list[dict[str, Any]] = []
    for rank, s in enumerate(ranked_frames, start=1):
        is_over = s.frame_ms > budget_ms
        top_slow_frames.append(
            {
                "rank": rank,
                "frame_index": s.frame_index,
                "frame_ms": round(s.frame_ms, 3),
                "over_budget": is_over,
                "over_budget_ms": round(max(0.0, s.frame_ms - budget_ms), 3),
                "cause": s.cause,
                "top_pass": s.top_pass,
                "top_pass_ms": round(s.top_pass_ms, 3),
                "phase": s.phase,
                "draw_calls": s.draw_calls,
                "compute_dispatches": s.compute_dispatches,
                "pso_compiles": s.pso_compiles,
                "upload_bytes": s.upload_bytes,
                "rt_allocs": s.rt_allocs,
                "sync_stall_ms": round(s.sync_stall_ms, 3),
            }
        )

    # Aggregate pass hotspots across all frames.
    pass_agg: dict[tuple[str, str], dict[str, Any]] = {}
    for s in samples:
        is_over = s.frame_ms > budget_ms
        for p in s.passes:
            key = (p.name, p.cause)
            rec = pass_agg.get(key)
            if rec is None:
                rec = {
                    "name": p.name,
                    "cause": p.cause,
                    "durations": [],
                    "over_budget_occurrences": 0,
                    "max_ms": -1.0,
                    "worst_frame_index": s.frame_index,
                }
                pass_agg[key] = rec
            rec["durations"].append(p.duration_ms)
            if is_over:
                rec["over_budget_occurrences"] += 1
            if p.duration_ms > rec["max_ms"] or (
                p.duration_ms == rec["max_ms"] and s.frame_index < rec["worst_frame_index"]
            ):
                rec["max_ms"] = p.duration_ms
                rec["worst_frame_index"] = s.frame_index

    pass_rows: list[dict[str, Any]] = []
    for rec in pass_agg.values():
        durs = sorted(rec["durations"])
        tot_p_ms = sum(durs)
        occ = len(durs)
        pass_rows.append(
            {
                "name": rec["name"],
                "cause": rec["cause"],
                "occurrences": occ,
                "over_budget_frame_occurrences": rec["over_budget_occurrences"],
                "total_ms": round(tot_p_ms, 3),
                "avg_ms": round(tot_p_ms / occ, 3),
                "p95_ms": round(_nearest_rank_percentile(durs, 0.95), 3),
                "max_ms": round(durs[-1], 3),
                "worst_frame_index": rec["worst_frame_index"],
            }
        )

    pass_rows.sort(
        key=lambda row: (
            -float(row["max_ms"]),
            -float(row["total_ms"]),
            SPIKE_CAUSES.index(row["cause"]) if row["cause"] in SPIKE_CAUSES else len(SPIKE_CAUSES),
            str(row["name"]),
        )
    )
    top_slow_passes: list[dict[str, Any]] = []
    for rank, row in enumerate(pass_rows[:top_n], start=1):
        top_slow_passes.append({"rank": rank, **row})

    has_error = any(d.get("severity") == "error" for d in diagnostics)
    if strict:
        ok = (not has_error) and malformed_records == 0 and len(diagnostics) == 0
    else:
        ok = not has_error

    return {
        "schema_version": SPIKE_ANALYSIS_SCHEMA_VERSION,
        "source_kind": source_kind,
        "timing_source": timing_source,
        "ok": ok,
        "budget_ms": round(budget_ms, 3),
        "target_fps": round(1000.0 / budget_ms, 3),
        "total_frames": total_frames,
        "frames_within_budget": within,
        "frames_over_budget": over,
        "fps30_compliance_rate": compliance_rate,
        "avg_frame_ms": avg_ms,
        "min_frame_ms": min_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "max_frame_ms": max_ms,
        "total_over_budget_ms": total_over_budget_ms,
        "max_over_budget_ms": max_over_budget_ms,
        "top_n": top_n,
        "spike_causes": spike_causes,
        "cause_breakdown": cause_breakdown,
        "top_slow_frames": top_slow_frames,
        "top_slow_passes": top_slow_passes,
        "malformed_records": malformed_records,
        "diagnostics": list(diagnostics),
    }


def analyze_30fps_frames(
    frames_or_payload: Any,
    *,
    budget_ms: float = DEFAULT_30FPS_BUDGET_MS,
    top_n: int = DEFAULT_TOP_N,
    strict: bool = False,
) -> dict[str, Any]:
    """Analyze frame timing logs (JSON/NDJSON/Chrome trace or Python structures) against a frame budget.

    Output is deterministic and strictly sanitized (zero host paths, zero raw shader source).
    """
    budget, limit = _validate_budget_and_top_n(budget_ms, top_n)
    diagnostics: list[dict[str, Any]] = []

    if isinstance(frames_or_payload, (bytes, bytearray, memoryview)):
        try:
            text = bytes(frames_or_payload).decode("utf-8")
        except UnicodeDecodeError:
            diagnostics.append(
                {
                    "severity": "error",
                    "code": "invalid_utf8",
                    "record_index": None,
                    "message": "Input timing log is not valid UTF-8.",
                }
            )
            return _summarize_frame_samples(
                (),
                budget_ms=budget,
                top_n=limit,
                source_kind="timing_log",
                timing_source="measured",
                malformed_records=1,
                diagnostics=diagnostics,
                strict=strict,
            )
        samples, malformed = _parse_timing_text(text, diagnostics)
    elif isinstance(frames_or_payload, str):
        samples, malformed = _parse_timing_text(frames_or_payload, diagnostics)
    else:
        samples, malformed = _parse_frame_records_from_obj(frames_or_payload, diagnostics)

    return _summarize_frame_samples(
        samples,
        budget_ms=budget,
        top_n=limit,
        source_kind="timing_log",
        timing_source="measured",
        malformed_records=malformed,
        diagnostics=diagnostics,
        strict=strict,
    )


def analyze_30fps_trace(
    trace: DecodedTrace,
    *,
    budget_ms: float = DEFAULT_30FPS_BUDGET_MS,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    """Analyze a binary DecodedTrace for 30 FPS frame-budget spikes and GPU command hotspots.

    Binary `.pg6trace` command streams do not carry wall-clock GPU timestamps, so per-frame
    and per-pass durations are derived from a deterministic packet workload model
    (`timing_source="estimated"`). Output is strictly sanitized.
    """
    budget, limit = _validate_budget_and_top_n(budget_ms, top_n)
    samples: list[_FrameSample] = []

    for frame in trace.frames:
        frame_pkts = trace.packets[frame.start_packet_index : frame.end_packet_index + 1]

        pso_compile_ms = 0.0
        pso_compiles = 0
        staging_upload_ms = 0.0
        upload_bytes = 0
        rt_alloc_ms = 0.0
        rt_allocs = 0
        sync_stall_ms = 0.0
        draw_calls = 0
        compute_dispatches = 0
        pass_list: list[_PassSample] = []

        render_pass_idx = 0
        in_render_pass = False
        cur_rp_ms = 0.0
        standalone_draw_ms = 0.0
        compute_pass_ms = 0.0

        def _flush_render_pass() -> None:
            nonlocal in_render_pass, cur_rp_ms, render_pass_idx
            if in_render_pass:
                pass_list.append(
                    _PassSample(
                        name=f"render_pass_{render_pass_idx}",
                        duration_ms=cur_rp_ms,
                        cause="draw_burst",
                    )
                )
                in_render_pass = False
                cur_rp_ms = 0.0

        for pkt in frame_pkts:
            if not pkt.valid:
                continue
            op = pkt.opcode
            f = pkt.fields
            if op == GpuOpcode.CreateShaderModule:
                pso_compiles += 1
                pso_compile_ms += 3.5
            elif op in (GpuOpcode.CreateRenderPipeline, GpuOpcode.CreateRenderPipeline2):
                pso_compiles += 1
                pso_compile_ms += 8.0
            elif op in (GpuOpcode.CreateComputePipeline, GpuOpcode.CreateComputePipeline2):
                pso_compiles += 1
                pso_compile_ms += 6.0
            elif op in (GpuOpcode.WriteBuffer, GpuOpcode.WriteTexture):
                nbytes = int(f.get("data_len", 0))
                upload_bytes += nbytes
                staging_upload_ms += 0.02 + (nbytes / (1024.0 * 1024.0)) * 2.5
            elif op in (GpuOpcode.CreateTexture, GpuOpcode.CreateTexture2):
                usage_names = f.get("usage_names") or ()
                usage_flags = int(f.get("usage_flags", 0))
                is_rt = ("RENDER_ATTACHMENT" in usage_names) or bool(usage_flags & 0x0010)
                if is_rt:
                    rt_allocs += 1
                    rt_alloc_ms += 4.5
                else:
                    rt_allocs += 1
                    rt_alloc_ms += 0.5
            elif op == GpuOpcode.CreateBuffer:
                rt_alloc_ms += 0.15
            elif op in (GpuOpcode.CreateBindGroupLayout, GpuOpcode.CreatePipelineLayout):
                sync_stall_ms += 0.25
            elif op in (GpuOpcode.CreateBindGroup, GpuOpcode.CreateBindGroup2):
                sync_stall_ms += 0.08
            elif op == GpuOpcode.BeginRenderPass:
                _flush_render_pass()
                render_pass_idx += 1
                in_render_pass = True
                cur_rp_ms = 0.35
            elif op == GpuOpcode.EndRenderPass:
                _flush_render_pass()
            elif op in (GpuOpcode.Draw, GpuOpcode.DrawIndexed):
                draw_calls += 1
                if in_render_pass:
                    cur_rp_ms += 0.04
                else:
                    standalone_draw_ms += 0.04
            elif op == GpuOpcode.DispatchCompute:
                compute_dispatches += 1
                compute_pass_ms += 0.15

        _flush_render_pass()

        if standalone_draw_ms > 0.0:
            pass_list.append(_PassSample(name="draw_calls", duration_ms=standalone_draw_ms, cause="draw_burst"))
        if compute_pass_ms > 0.0:
            pass_list.append(_PassSample(name="compute_dispatch", duration_ms=compute_pass_ms, cause="draw_burst"))
        if pso_compile_ms > 0.0:
            pass_list.append(_PassSample(name="pso_compile", duration_ms=pso_compile_ms, cause="pso_compile"))
        if staging_upload_ms > 0.0:
            pass_list.append(_PassSample(name="staging_upload", duration_ms=staging_upload_ms, cause="staging_upload"))
        if rt_alloc_ms > 0.0:
            pass_list.append(_PassSample(name="rt_alloc", duration_ms=rt_alloc_ms, cause="rt_alloc"))
        if sync_stall_ms > 0.0:
            pass_list.append(_PassSample(name="bind_group_setup", duration_ms=sync_stall_ms, cause="sync_stall"))

        base_ms = 0.5 if frame.complete else 0.2
        total_ms = base_ms + sum(p.duration_ms for p in pass_list)

        cause_ms: dict[str, float] = {c: 0.0 for c in SPIKE_CAUSES}
        for p in pass_list:
            cause_ms[p.cause] += p.duration_ms
        best_c = max(SPIKE_CAUSES[:-1], key=lambda c: (cause_ms[c], -SPIKE_CAUSES.index(c)))
        cause = best_c if cause_ms[best_c] > 0.0 else "unknown"

        top_pass: str | None = None
        top_pass_ms = 0.0
        if pass_list:
            best_p = max(pass_list, key=lambda p: (p.duration_ms, -SPIKE_CAUSES.index(p.cause), p.name))
            top_pass = best_p.name
            top_pass_ms = best_p.duration_ms

        samples.append(
            _FrameSample(
                frame_index=frame.frame_index,
                frame_ms=total_ms,
                cause=cause,
                top_pass=top_pass,
                top_pass_ms=top_pass_ms,
                phase=None,
                draw_calls=draw_calls,
                compute_dispatches=compute_dispatches,
                pso_compiles=pso_compiles,
                upload_bytes=upload_bytes,
                rt_allocs=rt_allocs,
                sync_stall_ms=sync_stall_ms,
                passes=tuple(pass_list),
            )
        )

    trace_diags = [
        {
            "severity": d.severity,
            "code": d.code,
            "record_index": d.packet_index,
            "message": d.message,
        }
        for d in trace.diagnostics
    ]
    malformed_pkts = sum(1 for p in trace.packets if not p.valid) + sum(
        1 for d in trace.diagnostics if d.severity == "error" and d.packet_index is None
    )

    return _summarize_frame_samples(
        samples,
        budget_ms=budget,
        top_n=limit,
        source_kind="binary_trace",
        timing_source="estimated",
        malformed_records=malformed_pkts,
        diagnostics=trace_diags,
        strict=trace.strict,
    )


def _is_timing_log_input(raw_bytes: bytes | bytearray | memoryview, source_hint: str | None = None) -> bool:
    if source_hint:
        if Path(source_hint).suffix.lower() in _TIMING_LOG_EXTENSIONS:
            return True
    raw = bytes(raw_bytes)
    if not raw:
        return True
    trimmed = raw.removeprefix(b"\xef\xbb\xbf").lstrip(b" \t\r\n")
    if not trimmed:
        return True
    if trimmed[:1] in (b"{", b"["):
        return True
    if len(raw) >= HEADER_SIZE:
        opcode, _flags, payload_bytes = HEADER_STRUCT.unpack_from(raw, 0)
        if opcode <= GpuOpcode.CreateTexture2 and (HEADER_SIZE + payload_bytes) <= len(raw):
            return False
        if (HEADER_SIZE + payload_bytes) <= len(raw):
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                return False
    try:
        raw.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def analyze_30fps_bytes(
    raw_bytes: bytes | bytearray | memoryview,
    *,
    budget_ms: float = DEFAULT_30FPS_BUDGET_MS,
    top_n: int = DEFAULT_TOP_N,
    strict: bool = False,
    preview_bytes: int = DEFAULT_PREVIEW_BYTES,
    source_hint: str | None = None,
) -> dict[str, Any]:
    """Auto-detect JSON/NDJSON timing log vs binary `.pg6trace` stream and run 30 FPS analysis."""
    budget, limit = _validate_budget_and_top_n(budget_ms, top_n)
    if _is_timing_log_input(raw_bytes, source_hint=source_hint):
        return analyze_30fps_frames(
            raw_bytes,
            budget_ms=budget,
            top_n=limit,
            strict=strict,
        )
    trace = decode_stream(raw_bytes, strict=strict, preview_bytes=preview_bytes)
    return analyze_30fps_trace(trace, budget_ms=budget, top_n=limit)


def format_30fps_analysis_text(report: dict[str, Any]) -> str:
    """Format a 30 FPS spike analysis dictionary as a sanitized human-readable table report."""
    budget_ms = float(report["budget_ms"])
    target_fps = float(report["target_fps"])
    total = int(report["total_frames"])
    within = int(report["frames_within_budget"])
    over = int(report["frames_over_budget"])
    compliance_pct = float(report["fps30_compliance_rate"]) * 100.0
    timing_src = str(report["timing_source"])

    if total == 0:
        status = f"NO FRAMES (0/0 frames | compliance=0.00% | timing={timing_src})"
    elif over == 0:
        status = (
            f"WITHIN BUDGET ({within}/{total} frames within budget | "
            f"compliance={compliance_pct:.2f}% | timing={timing_src})"
        )
    else:
        status = (
            f"OVER BUDGET ({over}/{total} frames over budget | "
            f"compliance={compliance_pct:.2f}% | timing={timing_src})"
        )

    cause_parts: list[str] = []
    spike_causes = report.get("spike_causes") or {}
    for cause in SPIKE_CAUSES:
        cnt = int(spike_causes.get(cause, 0))
        if cnt > 0:
            pct = (cnt / over * 100.0) if over > 0 else 0.0
            cause_parts.append(f"{cause}={cnt} ({pct:.1f}%)")
    causes_str = ", ".join(cause_parts) if cause_parts else "(none)"

    lines: list[str] = [
        f"30 FPS Frame-Budget Spike Analysis (budget={budget_ms:.3f} ms, target={target_fps:.1f} FPS):",
        f"  Status: {status}",
        (
            f"  Frame Time (ms): min={float(report['min_frame_ms']):.3f} | "
            f"avg={float(report['avg_frame_ms']):.3f} | "
            f"p50={float(report['p50_ms']):.3f} | "
            f"p95={float(report['p95_ms']):.3f} | "
            f"p99={float(report['p99_ms']):.3f} | "
            f"max={float(report['max_frame_ms']):.3f}"
        ),
        (
            f"  Budget Excess: total={float(report['total_over_budget_ms']):.3f} ms | "
            f"max={float(report['max_over_budget_ms']):.3f} ms | "
            f"malformed_records={int(report['malformed_records'])}"
        ),
        f"  Spike Causes: {causes_str}",
    ]

    top_frames = report.get("top_slow_frames") or []
    lines.append("")
    if not top_frames:
        lines.append("  Top Slowest Frames: (none)")
    else:
        lines.append(f"  Top Slowest Frames (top {len(top_frames)}):")
        lines.append(
            f"    {'Rank':>4s}  {'Frame':>6s}  {'Time (ms)':>9s}  {'Budget':<6s}  "
            f"{'Excess (ms)':>11s}  {'Cause':<14s}  {'Top Pass'}"
        )
        lines.append(
            f"    {'-' * 4}  {'-' * 6}  {'-' * 9}  {'-' * 6}  "
            f"{'-' * 11}  {'-' * 14}  {'-' * 24}"
        )
        for row in top_frames:
            budget_lbl = "OVER" if row["over_budget"] else "OK"
            top_pass = row["top_pass"] if row["top_pass"] else "-"
            lines.append(
                f"    {int(row['rank']):4d}  {int(row['frame_index']):6d}  "
                f"{float(row['frame_ms']):9.3f}  {budget_lbl:<6s}  "
                f"{float(row['over_budget_ms']):11.3f}  {str(row['cause']):<14s}  {top_pass}"
            )

    top_passes = report.get("top_slow_passes") or []
    lines.append("")
    if not top_passes:
        lines.append("  Top Slowest Passes: (none)")
    else:
        lines.append(f"  Top Slowest Passes (top {len(top_passes)}):")
        lines.append(
            f"    {'Rank':>4s}  {'Pass':<28s}  {'Cause':<14s}  {'Count':>5s}  "
            f"{'Avg (ms)':>9s}  {'P95 (ms)':>9s}  {'Max (ms)':>9s}  {'Total (ms)':>10s}  {'Worst':>6s}"
        )
        lines.append(
            f"    {'-' * 4}  {'-' * 28}  {'-' * 14}  {'-' * 5}  "
            f"{'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 10}  {'-' * 6}"
        )
        for row in top_passes:
            lines.append(
                f"    {int(row['rank']):4d}  {str(row['name'])[:28]:<28s}  "
                f"{str(row['cause']):<14s}  {int(row['occurrences']):5d}  "
                f"{float(row['avg_ms']):9.3f}  {float(row['p95_ms']):9.3f}  "
                f"{float(row['max_ms']):9.3f}  {float(row['total_ms']):10.3f}  "
                f"#{int(row['worst_frame_index']):<5d}"
            )

    diags = report.get("diagnostics") or []
    if diags:
        lines.append("")
        lines.append("  Diagnostics:")
        for d in diags:
            rec_str = f" record=#{d['record_index']}" if d.get("record_index") is not None else ""
            lines.append(f"    [{str(d['severity']).upper()}] {d['code']}{rec_str}: {d['message']}")

    return "\n".join(lines)


def fps30_analysis_to_json(report: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize a 30 FPS spike analysis dictionary to deterministic formatted JSON."""
    return json.dumps(report, indent=indent, allow_nan=False)


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
    if op >= GpuOpcode.CreateBindGroupLayout:
        return _format_v2_fields(op, f)
    return " ".join(f"{k}={v}" for k, v in f.items())


def _format_view(view: dict[str, Any]) -> str:
    parts = []
    if view["format"] != "undefined":
        parts.append(view["format"])
    if view["dimension"] != "undefined":
        parts.append(view["dimension"])
    if view["aspect"] != "all":
        parts.append(view["aspect"])
    if view["base_mip_level"] or view["mip_level_count"]:
        parts.append(f"mip={view['base_mip_level']}+{view['mip_level_count'] or 'all'}")
    if view["base_array_layer"] or view["array_layer_count"]:
        parts.append(f"layer={view['base_array_layer']}+{view['array_layer_count'] or 'all'}")
    return f"<{','.join(parts)}>" if parts else ""


def _format_v2_fields(op: int, f: dict[str, Any]) -> str:
    if op in (GpuOpcode.CreateBindGroupLayout, GpuOpcode.CreatePipelineLayout):
        base = f"id={f['id']} content={f['content_id']}"
        if f["manifest_reference"]:
            return base + " manifest-ref"
        if op == GpuOpcode.CreatePipelineLayout:
            return base + f" groups={f['bind_group_layout_ids']}"
        entries = []
        for e in f["entries"]:
            stages = "|".join(e["visibility_names"]) or "NONE"
            extra = ""
            if e.get("has_dynamic_offset"):
                extra += ",dynamic"
            if "view_dimension" in e and e["view_dimension"] != "undefined":
                extra += f",{e['view_dimension']}"
            if "storage_format" in e:
                extra += f",{e['storage_format']}"
            entries.append(f"@{e['binding']}:{e['type']}({e['binding_type']}{extra})[{stages}]")
        return base + f" entries={f['entry_count']} [{', '.join(entries)}]"
    if op == GpuOpcode.CreateRenderPipeline2:
        k = f["key"]
        targets = ",".join(t["format"] if t else "null" for t in k["color_targets"])
        ds = k["depth_stencil"]["format"] if k["depth_stencil"] else "none"
        return (
            f"id={f['id']} vs={f['vs_module_id']} fs={f['fs_module_id']} "
            f"layout={f['pipeline_layout_id']} topology={k['primitive']['topology']} "
            f"targets=[{targets}] depth={ds} samples={k['multisample']['count']} "
            f"vbufs={k['vertex_buffer_count']} attrs={k['vertex_attribute_count']} "
            f"key={k['key_fnv1a64']}"
        )
    if op == GpuOpcode.CreateComputePipeline2:
        return f"id={f['id']} cs={f['cs_module_id']} layout={f['pipeline_layout_id']}"
    if op == GpuOpcode.BeginRenderPass:
        parts = []
        for i, c in enumerate(f["color_attachments"]):
            if c is None:
                parts.append(f"c{i}=null")
                continue
            tex = "swapchain" if c["swapchain"] else f"#{c['texture_id']}"
            load = c["load_op"]
            if load == "clear":
                load += "(" + ",".join(f"{v:.3g}" for v in c["clear_value"]) + ")"
            parts.append(f"c{i}={tex}{_format_view(c['view'])} {load}/{c['store_op']}")
        ds = f["depth_stencil"]
        if ds is not None:
            depth = f"{ds['depth_load_op']}/{ds['depth_store_op']}"
            if ds["depth_load_op"] == "clear":
                depth += f"({ds['depth_clear_value']:.3g})"
            stencil = f"{ds['stencil_load_op']}/{ds['stencil_store_op']}"
            if ds["stencil_load_op"] == "clear":
                stencil += f"({ds['stencil_clear_value']})"
            ro = ""
            if ds["depth_read_only"] or ds["stencil_read_only"]:
                ro = " ro=" + "".join(
                    flag
                    for flag, on in (("D", ds["depth_read_only"]), ("S", ds["stencil_read_only"]))
                    if on
                )
            parts.append(
                f"ds=#{ds['texture_id']}{_format_view(ds['view'])} depth={depth} stencil={stencil}{ro}"
            )
        return f"colors={f['color_count']} " + " ".join(parts)
    if op == GpuOpcode.EndRenderPass:
        return f"trailing_bytes={f['trailing_bytes']}" if "trailing_bytes" in f else ""
    if op == GpuOpcode.SetViewport:
        return (
            f"rect=({f['x']:g},{f['y']:g} {f['width']:g}x{f['height']:g}) "
            f"depth=[{f['min_depth']:g},{f['max_depth']:g}]"
        )
    if op == GpuOpcode.SetScissorRect:
        return f"rect=({f['x']},{f['y']} {f['width']}x{f['height']})"
    if op == GpuOpcode.SetBlendConstant:
        return f"rgba=({f['r']:.4f}, {f['g']:.4f}, {f['b']:.4f}, {f['a']:.4f})"
    if op == GpuOpcode.SetStencilReference:
        return f"reference={f['reference']}"
    if op == GpuOpcode.CreateBindGroup2:
        entries = []
        for e in f["entries"]:
            res = "null" if e["null_binding"] else f"#{e['resource_id']}"
            base = f"@{e['binding']}:{e['resource_type_name']}({res})"
            if e["resource_type"] == GpuBindResourceType.Buffer and (e["offset"] or e["size"]):
                base += f"[{e['offset']}:{e['size'] or 'end'}]"
            if "view" in e:
                base += _format_view(e["view"])
            entries.append(base)
        return (
            f"id={f['id']} layout={f['layout_id']} entries={f['entry_count']} "
            f"[{', '.join(entries)}]"
        )
    if op == GpuOpcode.SetBindGroup2:
        return (
            f"group={f['group_index']} bind_group_id={f['bind_group_id']} "
            f"dynamic_offsets={f['dynamic_offsets']}"
        )
    if op == GpuOpcode.CreateSampler2:
        cmp = f" compare={f['compare']}" if f["compare"] != "undefined" else ""
        return (
            f"id={f['id']} addr=({f['address_u']},{f['address_v']},{f['address_w']}) "
            f"filter=(mag={f['mag_filter']},min={f['min_filter']},mip={f['mipmap_filter']}) "
            f"aniso={f['max_anisotropy']} lod=[{f['lod_min_clamp']:g},{f['lod_max_clamp']:g}]{cmp}"
        )
    return " ".join(f"{k}={v}" for k, v in f.items())


def _format_op_histogram(hist: dict[str, int] | None) -> str:
    if not hist:
        return "(none)"
    return ", ".join(f"{name}={count}" for name, count in sorted(hist.items()))


def format_trace_text(
    trace: DecodedTrace,
    *,
    stats_only: bool = False,
    shader_summary: bool = False,
    binding_audit: bool = False,
    bink_video_audit: bool = False,
    start_screen_skips: bool = False,
    analyze_30fps: bool = False,
    budget_ms: float = DEFAULT_30FPS_BUDGET_MS,
    top_n: int = DEFAULT_TOP_N,
) -> str:
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
        (
            f"RenderPasses: begin={trace.begin_render_passes} end={trace.end_render_passes} | "
            f"Layouts: bgl={trace.bind_group_layouts_created} "
            f"pll={trace.pipeline_layouts_created} | "
            f"Pipelines2: render={trace.render_pipelines2_created} "
            f"compute={trace.compute_pipelines2_created} | "
            f"Dynamic: vp={trace.viewport_changes} scissor={trace.scissor_changes} "
            f"blend={trace.blend_constant_changes} stencil={trace.stencil_ref_changes} | "
            f"V2Malformed: {trace.malformed_v2_packets} "
            f"V2Truncated: {trace.truncated_v2_diagnostics}"
        ),
        (
            f"ColorLoadOps: {_format_op_histogram(trace.color_load_ops)} | "
            f"ColorStoreOps: {_format_op_histogram(trace.color_store_ops)} | "
            f"DepthLoadOps: {_format_op_histogram(trace.depth_load_ops)} | "
            f"DepthStoreOps: {_format_op_histogram(trace.depth_store_ops)}"
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
        if frame.begin_render_passes or frame.end_render_passes:
            extras.append(f"passes={frame.begin_render_passes}/{frame.end_render_passes}")
        if frame.bind_group_layouts_created:
            extras.append(f"bgl={frame.bind_group_layouts_created}")
        if frame.pipeline_layouts_created:
            extras.append(f"pll={frame.pipeline_layouts_created}")
        if frame.render_pipelines2_created:
            extras.append(f"rp2={frame.render_pipelines2_created}")
        if frame.compute_pipelines2_created:
            extras.append(f"cp2={frame.compute_pipelines2_created}")
        if frame.viewport_changes:
            extras.append(f"viewport={frame.viewport_changes}")
        if frame.scissor_changes:
            extras.append(f"scissor={frame.scissor_changes}")
        if frame.blend_constant_changes:
            extras.append(f"blend={frame.blend_constant_changes}")
        if frame.stencil_ref_changes:
            extras.append(f"stencil={frame.stencil_ref_changes}")
        if frame.malformed_v2_packets:
            extras.append(f"v2_malformed={frame.malformed_v2_packets}")
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

    if shader_summary:
        lines.append("")
        lines.append(format_shader_summary_text(summarize_shaders(trace)))

    if binding_audit:
        lines.append("")
        lines.append(format_binding_audit_text(audit_bindings(trace)))

    if bink_video_audit:
        lines.append("")
        lines.append(format_bink_video_audit_text(audit_bink_video(trace)))

    if start_screen_skips:
        lines.append("")
        lines.append(format_start_screen_skips_text(audit_start_screen_skips(trace)))

    if analyze_30fps:
        lines.append("")
        lines.append(
            format_30fps_analysis_text(
                analyze_30fps_trace(trace, budget_ms=budget_ms, top_n=top_n)
            )
        )

    return "\n".join(lines)


def trace_to_json(
    trace: DecodedTrace,
    *,
    stats_only: bool = False,
    shader_summary: bool = False,
    binding_audit: bool = False,
    bink_video_audit: bool = False,
    start_screen_skips: bool = False,
    analyze_30fps: bool = False,
    budget_ms: float = DEFAULT_30FPS_BUDGET_MS,
    top_n: int = DEFAULT_TOP_N,
    indent: int = 2,
) -> str:
    """Serialize a DecodedTrace to formatted JSON."""
    return json.dumps(
        trace.to_dict(
            stats_only=stats_only,
            shader_summary=shader_summary,
            binding_audit=binding_audit,
            bink_video_audit=bink_video_audit,
            start_screen_skips=start_screen_skips,
            analyze_30fps=analyze_30fps,
            budget_ms=budget_ms,
            top_n=top_n,
        ),
        indent=indent,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Disassemble and inspect a playgta6 binary GPU command stream or frame timing log."
    )
    parser.add_argument(
        "trace_file",
        help="Path to raw binary GPU command stream or JSON/NDJSON timing log (or '-' to read from stdin).",
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
            "Fail with non-zero exit code on unknown opcodes (>= 35), "
            "size mismatches, or malformed/truncated packets or timing records."
        ),
    )
    parser.add_argument(
        "--shader-summary",
        action="store_true",
        help=(
            "Audit shader modules/hashes attached to pipelines and rank them by "
            "executed draw + dispatch counts."
        ),
    )
    parser.add_argument(
        "--binding-audit",
        action="store_true",
        help=(
            "Audit CreateBindGroupLayout vs CreateBindGroup2 resource kinds/bindings "
            "and BeginRenderPass vs CreateRenderPipeline2 attachment formats."
        ),
    )
    parser.add_argument(
        "--bink-video-audit",
        action="store_true",
        help=(
            "Audit r8unorm planar Bink2 video textures (uploads, byte totals, bind groups, "
            "and whether any draw or compute dispatch sampled those video planes)."
        ),
    )
    parser.add_argument(
        "--start-screen-skips",
        action="store_true",
        help=(
            "Rank top pipelines/shader pairs associated with skipped/unemitted draws "
            "or missing-binding anomalies in sanitized text/JSON output."
        ),
    )
    parser.add_argument(
        "--analyze-30fps",
        action="store_true",
        help=(
            "Analyze GPU trace or JSON/NDJSON frame timing logs against a 30 FPS (33.333 ms) "
            "frame budget, reporting p50/p95/p99/max frame times and categorized spike hotspots."
        ),
    )
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=None,
        metavar="MS",
        help=(
            "Frame budget threshold in milliseconds for --analyze-30fps "
            f"(default: {DEFAULT_30FPS_BUDGET_MS}; passing --budget-ms enables analyzer mode)."
        ),
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        metavar="N",
        help=(
            "Maximum number of slowest frames and passes to report in --analyze-30fps mode "
            f"(default: {DEFAULT_TOP_N})."
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

    run_analyze_30fps = bool(args.analyze_30fps or args.budget_ms is not None)
    budget_ms = DEFAULT_30FPS_BUDGET_MS if args.budget_ms is None else float(args.budget_ms)
    if run_analyze_30fps:
        try:
            budget_ms, top_n = _validate_budget_and_top_n(budget_ms, args.top_n)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    else:
        top_n = args.top_n

    if args.trace_file == "-":
        raw_bytes = sys.stdin.buffer.read()
        source_hint: str | None = None
    else:
        path = Path(args.trace_file)
        if not path.is_file():
            print(f"Error: trace file not found: {path}", file=sys.stderr)
            return 2
        raw_bytes = path.read_bytes()
        source_hint = args.trace_file

    if run_analyze_30fps:
        has_binary_flags = bool(
            args.stats
            or args.shader_summary
            or args.binding_audit
            or args.bink_video_audit
            or args.start_screen_skips
        )
        is_timing_log = _is_timing_log_input(raw_bytes, source_hint=source_hint)
        if is_timing_log or not has_binary_flags:
            report = analyze_30fps_bytes(
                raw_bytes,
                budget_ms=budget_ms,
                top_n=top_n,
                strict=args.strict,
                preview_bytes=args.preview_bytes,
                source_hint=source_hint,
            )
            if args.as_json:
                print(fps30_analysis_to_json(report))
            else:
                print(format_30fps_analysis_text(report))
            if args.strict and not report["ok"]:
                return 1
            return 0

    trace = decode_stream(
        raw_bytes,
        strict=args.strict,
        preview_bytes=args.preview_bytes,
    )

    if args.as_json:
        print(
            trace_to_json(
                trace,
                stats_only=args.stats,
                shader_summary=args.shader_summary,
                binding_audit=args.binding_audit,
                bink_video_audit=args.bink_video_audit,
                start_screen_skips=args.start_screen_skips,
                analyze_30fps=run_analyze_30fps,
                budget_ms=budget_ms,
                top_n=top_n,
            )
        )
    else:
        print(
            format_trace_text(
                trace,
                stats_only=args.stats,
                shader_summary=args.shader_summary,
                binding_audit=args.binding_audit,
                bink_video_audit=args.bink_video_audit,
                start_screen_skips=args.start_screen_skips,
                analyze_30fps=run_analyze_30fps,
                budget_ms=budget_ms,
                top_n=top_n,
            )
        )

    if args.strict and not trace.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
