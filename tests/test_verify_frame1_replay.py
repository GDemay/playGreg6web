#!/usr/bin/env python3
"""Unit tests for tools/verify_frame1_replay.py (Track 5 / W80).

Uses synthetic in-memory `.pg6trace` GPU command streams and `WGPK` v1 `.wgsl.pack`
buffers only (zero copyrighted game data).
"""

from __future__ import annotations

import contextlib
import getpass
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import verify_frame1_replay as vfr  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic .pg6trace packet builders (matching runtime/abi/playgta6_abi.h)
# ---------------------------------------------------------------------------

VALID_VS_WGSL = (
    "@vertex\n"
    "fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {\n"
    "    return vec4<f32>(pos, 1.0);\n"
    "}\n"
)

PROCEDURAL_VS_WGSL = (
    "@vertex\n"
    "fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {\n"
    "    return vec4<f32>(f32(vid), 0.0, 0.0, 1.0);\n"
    "}\n"
)

VALID_FS_WGSL = (
    "@fragment\n"
    "fn fs_main() -> @location(0) vec4<f32> {\n"
    "    return vec4<f32>(1.0, 0.25, 0.5, 1.0);\n"
    "}\n"
)

LEGACY_PS_MAIN_WGSL = (
    "// Legacy pixel shader entry point name that must fail WebGPU fs_main check\n"
    "@fragment\n"
    "fn ps_main() -> @location(0) vec4<f32> {\n"
    "    return vec4<f32>(0.2, 0.8, 0.4, 1.0);\n"
    "}\n"
)

COMMENTED_FS_MAIN_WGSL = (
    "// @fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n"
    "/* @fragment fn fs_main() {} */\n"
    "@fragment\n"
    "fn ps_main() -> @location(0) vec4<f32> {\n"
    "    return vec4<f32>(0.0, 0.0, 1.0, 1.0);\n"
    "}\n"
)

VALID_CS_WGSL = (
    "@compute @workgroup_size(64, 1, 1)\n"
    "fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {\n"
    "    let _x = gid.x;\n"
    "}\n"
)

VS_HASH = 0x1111222233334444
FS_HASH = 0x5555666677778888
CS_HASH = 0x9999AAAABBBBCCCC
MISSING_HASH = 0xDEADBEEFCAFEBABE


def _pkt(opcode: int, payload: bytes = b"", flags: int = 0) -> bytes:
    return struct.pack("<HHI", opcode, flags, len(payload)) + payload


def pkt_begin_frame() -> bytes:
    return _pkt(int(vfr.GpuOpcode.BeginFrame))


def pkt_clear_color(r: float = 0.0, g: float = 0.0, b: float = 0.0, a: float = 1.0) -> bytes:
    return _pkt(int(vfr.GpuOpcode.ClearColor), struct.pack("<ffff", r, g, b, a))


def pkt_create_buffer(buf_id: int, usage: int, byte_size: int) -> bytes:
    return _pkt(int(vfr.GpuOpcode.CreateBuffer), struct.pack("<IIQ", buf_id, usage, byte_size))


def pkt_write_buffer(buf_id: int, dst_offset: int, data: bytes) -> bytes:
    head = struct.pack("<IIII", buf_id, dst_offset, len(data), 0)
    return _pkt(int(vfr.GpuOpcode.WriteBuffer), head + data)


def pkt_create_shader_module(mod_id: int, stage: int, code: str | bytes) -> bytes:
    raw = code.encode("utf-8") if isinstance(code, str) else code
    head = struct.pack("<IIII", mod_id, stage, len(raw), 0)
    return _pkt(int(vfr.GpuOpcode.CreateShaderModule), head + raw)


def pkt_create_render_pipeline(
    pipe_id: int,
    vs_id: int,
    fs_id: int,
    vertex_stride: int = 12,
    topology: int = 0,
    flags: int = 0,
) -> bytes:
    return _pkt(
        int(vfr.GpuOpcode.CreateRenderPipeline),
        struct.pack("<IIIIII", pipe_id, vs_id, fs_id, vertex_stride, topology, flags),
    )


def pkt_create_render_pipeline2(
    pipe_id: int,
    vs_id: int,
    fs_id: int,
    *,
    layout_id: int = 0,
    vs_hash: int = 0,
    ps_hash: int = 0,
    vb_stride: int = 12,
    vb_step_mode: int = 1,
) -> bytes:
    key = bytearray(vfr.RENDER_PIPELINE_KEY_BYTES)
    struct.pack_into("<QQII", key, 0, vs_hash, ps_hash, 0, 1)
    if vb_stride > 0 and vb_step_mode != 0:
        struct.pack_into("<II", key, 172, 1, 1)  # 1 vertex buffer, 1 attribute
        struct.pack_into("<IBB2x", key, 180, vb_stride, vb_step_mode, 1)
    head = struct.pack("<IIII", pipe_id, vs_id, fs_id, layout_id)
    return _pkt(int(vfr.GpuOpcode.CreateRenderPipeline2), head + bytes(key))


def pkt_create_texture2(tex_id: int, width: int = 64, height: int = 64) -> bytes:
    payload = struct.pack(
        "<IIIIIIIBBBB8s",
        tex_id,
        width,
        height,
        1,
        1,
        1,
        0x14,  # TEXTURE_BINDING | RENDER_ATTACHMENT
        1,     # 2d
        22,    # rgba8unorm
        0,
        0,
        bytes(8),
    )
    return _pkt(int(vfr.GpuOpcode.CreateTexture2), payload)


def pkt_begin_render_pass() -> bytes:
    return _pkt(int(vfr.GpuOpcode.BeginRenderPass), bytes(vfr.BEGIN_RENDER_PASS_BYTES))


def pkt_end_render_pass() -> bytes:
    return _pkt(int(vfr.GpuOpcode.EndRenderPass))


def pkt_set_pipeline(pipe_id: int) -> bytes:
    return _pkt(int(vfr.GpuOpcode.SetPipeline), struct.pack("<I", pipe_id))


def pkt_set_vertex_buffer(slot: int, buf_id: int, offset: int = 0) -> bytes:
    return _pkt(int(vfr.GpuOpcode.SetVertexBuffer), struct.pack("<IIQ", slot, buf_id, offset))


def pkt_set_index_buffer(buf_id: int, index_format: int = 0, offset: int = 0) -> bytes:
    return _pkt(int(vfr.GpuOpcode.SetIndexBuffer), struct.pack("<IIQ", buf_id, index_format, offset))


def pkt_draw(vertex_count: int = 3, instance_count: int = 1) -> bytes:
    return _pkt(int(vfr.GpuOpcode.Draw), struct.pack("<IIII", vertex_count, instance_count, 0, 0))


def pkt_draw_indexed(index_count: int = 3, instance_count: int = 1) -> bytes:
    return _pkt(
        int(vfr.GpuOpcode.DrawIndexed),
        struct.pack("<IIIiI", index_count, instance_count, 0, 0, 0),
    )


def pkt_present() -> bytes:
    return _pkt(int(vfr.GpuOpcode.Present))


def build_valid_frame1_trace(
    *,
    vs_code: str = f"0x{VS_HASH:016x}",
    fs_code: str = f"0x{FS_HASH:016x}",
    include_write_vb: bool = True,
    vertex_stride: int = 12,
    bind_vb: bool = True,
    include_draw: bool = True,
    include_present: bool = True,
    draw_after_present: bool = False,
) -> bytes:
    """Build a synthetic Frame-1 `.pg6trace` command stream."""
    vb_bytes = struct.pack("<9f", 0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0)
    chunks: list[bytes] = [
        pkt_begin_frame(),
        pkt_clear_color(0.05, 0.07, 0.11, 1.0),
        pkt_create_buffer(1, 0x0020 | 0x0008, len(vb_bytes)),  # VERTEX | COPY_DST
    ]
    if include_write_vb:
        chunks.append(pkt_write_buffer(1, 0, vb_bytes))
    chunks.extend(
        [
            pkt_create_shader_module(10, 0, vs_code),  # Vertex
            pkt_create_shader_module(11, 1, fs_code),  # Fragment
            pkt_create_render_pipeline(20, 10, 11, vertex_stride=vertex_stride),
        ]
    )
    if draw_after_present and include_present:
        chunks.append(pkt_present())
    chunks.append(pkt_set_pipeline(20))
    if bind_vb:
        chunks.append(pkt_set_vertex_buffer(0, 1, 0))
    if include_draw:
        chunks.append(pkt_draw(3, 1))
    if include_present and not draw_after_present:
        chunks.append(pkt_present())
    return b"".join(chunks)


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        code = vfr.main(argv)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------


class TestVerifyFrame1Replay(unittest.TestCase):
    def test_valid_frame1_capture_and_matching_wgpk_pack(self) -> None:
        trace_bytes = build_valid_frame1_trace()
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", VALID_FS_WGSL),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "frame1.pg6trace"
            pack_path = Path(tmp) / "shaders.wgsl.pack"
            trace_path.write_bytes(trace_bytes)
            pack_path.write_bytes(pack_bytes)

            code, out, err = run_cli(
                [str(trace_path), "--shader-pack", str(pack_path), "--check-replay-assets", "--json"]
            )
            self.assertEqual(code, 0, f"expected exit 0, got {code}; out={out}; err={err}")
            self.assertEqual(err, "")

            report = json.loads(out)
            self.assertTrue(report["ok"])
            self.assertEqual(report["status"], "pass")
            self.assertTrue(report["check_replay_assets"])

            # Trace counts
            self.assertEqual(report["trace"]["present_count"], 1)
            self.assertEqual(report["trace"]["packet_counts"]["BeginFrame"], 1)
            self.assertEqual(report["trace"]["packet_counts"]["ClearColor"], 1)
            self.assertEqual(report["trace"]["packet_counts"]["WriteBuffer"], 1)
            self.assertEqual(report["trace"]["packet_counts"]["CreateShaderModule"], 2)
            self.assertEqual(report["trace"]["packet_counts"]["CreateRenderPipeline"], 1)
            self.assertEqual(report["trace"]["packet_counts"]["Draw"], 1)
            self.assertEqual(report["trace"]["packet_counts"]["Present"], 1)

            # Shader match & entry points
            shaders = report["shaders"]
            self.assertTrue(shaders["pack_valid"])
            self.assertEqual(shaders["match_rate"], 1.0)
            self.assertEqual(
                shaders["matched_shader_hashes"],
                [f"0x{VS_HASH:016x}", f"0x{FS_HASH:016x}"],
            )
            self.assertEqual(shaders["missing_shader_hashes"], [])
            self.assertEqual(shaders["stage_counts"], {"vertex": 1, "fragment": 1, "compute": 0})
            self.assertEqual(shaders["entry_point_checks_passed"], 2)
            self.assertEqual(shaders["entry_point_mismatch_count"], 0)
            self.assertEqual(shaders["entry_point_mismatches"], [])

            # Draw readiness
            draw = report["draw_readiness"]
            self.assertTrue(draw["ready"])
            self.assertEqual(draw["draws_before_present"], 1)
            self.assertEqual(draw["resolved_draws_before_present"], 1)
            self.assertEqual(draw["draws_with_vertex_data_before_present"], 1)
            self.assertEqual(draw["vertex_buffer_writes"], 1)
            self.assertEqual(draw["vertex_bytes_written"], 36)
            self.assertEqual(report["errors"], [])

    def test_v2_packets_draw_indexed_create_texture2_and_compute(self) -> None:
        vb_bytes = struct.pack("<9f", 0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0)
        ib_bytes = struct.pack("<3H", 0, 1, 2)
        trace_bytes = b"".join(
            [
                pkt_begin_frame(),
                pkt_create_texture2(50, 1280, 720),
                pkt_create_buffer(1, 0x0020 | 0x0008, len(vb_bytes)),
                pkt_write_buffer(1, 0, vb_bytes),
                pkt_create_buffer(2, 0x0010 | 0x0008, len(ib_bytes)),
                pkt_write_buffer(2, 0, ib_bytes),
                # Omitted code_len=0 VS/FS resolved via CreateRenderPipeline2 key hashes
                pkt_create_shader_module(10, 0, b""),
                pkt_create_shader_module(11, 1, b""),
                pkt_create_shader_module(12, 2, f"wgpk:0x{CS_HASH:016x}"),
                pkt_create_render_pipeline2(
                    20, 10, 11, vs_hash=VS_HASH, ps_hash=FS_HASH, vb_stride=12, vb_step_mode=1
                ),
                pkt_begin_render_pass(),
                pkt_set_pipeline(20),
                pkt_set_vertex_buffer(0, 1, 0),
                pkt_set_index_buffer(2, 0, 0),
                pkt_draw_indexed(3, 1),
                pkt_end_render_pass(),
                pkt_present(),
            ]
        )
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", VALID_FS_WGSL),
                (CS_HASH, "compute", VALID_CS_WGSL),
            ]
        )

        report = vfr.verify_frame1_replay_bytes(
            trace_bytes,
            shader_pack_bytes=pack_bytes,
            trace_label="v2_frame1.pg6trace",
            shader_pack_label="v2.wgsl.pack",
        )
        self.assertTrue(report["ok"], f"unexpected errors: {report['errors']}")
        self.assertEqual(report["trace"]["packet_counts"]["CreateTexture2"], 1)
        self.assertEqual(report["trace"]["packet_counts"]["CreateRenderPipeline2"], 1)
        self.assertEqual(report["trace"]["packet_counts"]["DrawIndexed"], 1)
        self.assertEqual(report["trace"]["packet_counts"]["BeginRenderPass"], 1)
        self.assertEqual(report["trace"]["packet_counts"]["EndRenderPass"], 1)
        self.assertEqual(report["shaders"]["stage_counts"], {"vertex": 1, "fragment": 1, "compute": 1})
        self.assertEqual(len(report["shaders"]["matched_shader_hashes"]), 3)
        self.assertEqual(report["shaders"]["missing_shader_hashes"], [])
        self.assertTrue(report["draw_readiness"]["ready"])
        self.assertEqual(report["draw_readiness"]["total_draw_indexed"], 1)

    def test_inline_wgsl_fnv1a64_cross_check_and_standalone_mode(self) -> None:
        trace_bytes = build_valid_frame1_trace(vs_code=VALID_VS_WGSL, fs_code=VALID_FS_WGSL)
        vs_fnv = vfr.fnv1a_64(VALID_VS_WGSL)
        fs_fnv = vfr.fnv1a_64(VALID_FS_WGSL)
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (vs_fnv, "vertex", VALID_VS_WGSL),
                (fs_fnv, "fragment", VALID_FS_WGSL),
            ]
        )

        # With matching WGPK pack (FNV-1a 64 hash cross-check)
        report_with_pack = vfr.verify_frame1_replay_bytes(
            trace_bytes,
            shader_pack_bytes=pack_bytes,
        )
        self.assertTrue(report_with_pack["ok"], report_with_pack["errors"])
        self.assertEqual(
            report_with_pack["shaders"]["matched_shader_hashes"],
            sorted([vfr.format_hash_hex(vs_fnv), vfr.format_hash_hex(fs_fnv)]),
        )

        # Without WGPK pack (inline WGSL still verified for entry points + draw readiness)
        report_no_pack = vfr.verify_frame1_replay_bytes(trace_bytes)
        self.assertTrue(report_no_pack["ok"], report_no_pack["errors"])
        self.assertEqual(report_no_pack["shaders"]["entry_point_checks_passed"], 2)
        self.assertEqual(report_no_pack["shaders"]["entry_point_mismatch_count"], 0)

    def test_procedural_fullscreen_triangle_draw_ready(self) -> None:
        trace_bytes = build_valid_frame1_trace(
            vs_code=PROCEDURAL_VS_WGSL,
            fs_code=VALID_FS_WGSL,
            include_write_vb=False,
            vertex_stride=0,
            bind_vb=False,
        )
        report = vfr.verify_frame1_replay_bytes(trace_bytes)
        self.assertTrue(report["ok"], report["errors"])
        self.assertTrue(report["draw_readiness"]["ready"])
        self.assertEqual(report["draw_readiness"]["procedural_draws_before_present"], 1)

    def test_fragment_ps_main_vs_fs_main_mismatch_detected(self) -> None:
        trace_bytes = build_valid_frame1_trace()
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", LEGACY_PS_MAIN_WGSL),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "ps_main.pg6trace"
            pack_path = Path(tmp) / "legacy.wgsl.pack"
            trace_path.write_bytes(trace_bytes)
            pack_path.write_bytes(pack_bytes)

            code, out, _err = run_cli(
                ["--trace", str(trace_path), "--shader-pack", str(pack_path), "--json"]
            )
            self.assertNotEqual(code, 0)
            report = json.loads(out)
            self.assertFalse(report["ok"])
            self.assertFalse(report["checklist"]["entry_points_valid"])
            self.assertEqual(report["shaders"]["entry_point_mismatch_count"], 1)
            mismatch = report["shaders"]["entry_point_mismatches"][0]
            self.assertEqual(mismatch["hash"], f"0x{FS_HASH:016x}")
            self.assertEqual(mismatch["stage"], "fragment")
            self.assertEqual(mismatch["expected_entry_point"], "fs_main")
            self.assertEqual(mismatch["found_entry_points"], [{"stage": "fragment", "name": "ps_main"}])
            self.assertIn("ps_main", mismatch["reason"])
            self.assertIn("fs_main", mismatch["reason"])

    def test_commented_out_fs_main_does_not_bypass_entry_point_check(self) -> None:
        trace_bytes = build_valid_frame1_trace()
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", COMMENTED_FS_MAIN_WGSL),
            ]
        )
        report = vfr.verify_frame1_replay_bytes(trace_bytes, shader_pack_bytes=pack_bytes)
        self.assertFalse(report["ok"])
        self.assertEqual(report["shaders"]["entry_point_mismatch_count"], 1)
        self.assertEqual(
            report["shaders"]["entry_point_mismatches"][0]["found_entry_points"],
            [{"stage": "fragment", "name": "ps_main"}],
        )

    def test_missing_shader_hash_in_pack(self) -> None:
        trace_bytes = build_valid_frame1_trace(fs_code=f"0x{MISSING_HASH:016x}")
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", VALID_FS_WGSL),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "missing_fs.pg6trace"
            pack_path = Path(tmp) / "pack.wgsl.pack"
            trace_path.write_bytes(trace_bytes)
            pack_path.write_bytes(pack_bytes)

            code, out, _err = run_cli(
                [str(trace_path), "--shader-pack", str(pack_path), "--json"]
            )
            self.assertNotEqual(code, 0)
            report = json.loads(out)
            self.assertFalse(report["ok"])
            self.assertFalse(report["checklist"]["all_shaders_resolved"])
            self.assertEqual(report["shaders"]["matched_shader_hashes"], [f"0x{VS_HASH:016x}"])
            self.assertEqual(
                report["shaders"]["missing_shader_hashes"],
                [f"0x{MISSING_HASH:016x}"],
            )
            self.assertFalse(report["draw_readiness"]["ready"])

    def test_stage_mismatch_between_trace_and_pack(self) -> None:
        trace_bytes = build_valid_frame1_trace()
        # Pack FS_HASH as vertex stage instead of fragment
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "vertex", VALID_FS_WGSL),
            ]
        )
        report = vfr.verify_frame1_replay_bytes(trace_bytes, shader_pack_bytes=pack_bytes)
        self.assertFalse(report["ok"])
        self.assertEqual(report["shaders"]["stage_mismatch_count"], 1)
        self.assertFalse(report["checklist"]["entry_points_valid"])

    def test_draw_readiness_failures(self) -> None:
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", VALID_FS_WGSL),
            ]
        )

        # 1. Clear-only capture (no Draw)
        no_draw = build_valid_frame1_trace(include_draw=False)
        rep_no_draw = vfr.verify_frame1_replay_bytes(no_draw, shader_pack_bytes=pack_bytes)
        self.assertFalse(rep_no_draw["ok"])
        self.assertFalse(rep_no_draw["draw_readiness"]["ready"])

        # 2. Draw with empty vertex buffer (no WriteBuffer)
        no_vb_data = build_valid_frame1_trace(include_write_vb=False)
        rep_no_vb = vfr.verify_frame1_replay_bytes(no_vb_data, shader_pack_bytes=pack_bytes)
        self.assertFalse(rep_no_vb["ok"])
        self.assertFalse(rep_no_vb["draw_readiness"]["ready"])
        self.assertTrue(
            any("0 bytes written" in msg for msg in rep_no_vb["draw_readiness"]["issues"])
        )

        # 3. Draw only after Present
        after_present = build_valid_frame1_trace(draw_after_present=True)
        rep_after = vfr.verify_frame1_replay_bytes(after_present, shader_pack_bytes=pack_bytes)
        self.assertFalse(rep_after["ok"])
        self.assertFalse(rep_after["draw_readiness"]["ready"])
        self.assertEqual(rep_after["draw_readiness"]["draws_before_present"], 0)
        self.assertEqual(rep_after["draw_readiness"]["total_draws"], 1)

        # 4. No Present frame at all
        no_present = build_valid_frame1_trace(include_present=False)
        rep_no_present = vfr.verify_frame1_replay_bytes(no_present, shader_pack_bytes=pack_bytes)
        self.assertFalse(rep_no_present["ok"])
        self.assertFalse(rep_no_present["checklist"]["has_present_frame"])

    def test_truncated_and_corrupt_trace_and_bad_wgpk_magic(self) -> None:
        valid_trace = build_valid_frame1_trace()
        valid_pack = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", VALID_FS_WGSL),
            ]
        )

        # Empty trace
        rep_empty = vfr.verify_frame1_replay_bytes(b"")
        self.assertFalse(rep_empty["ok"])
        self.assertFalse(rep_empty["trace"]["valid"])

        # Truncated 5-byte packet header
        rep_trunc_hdr = vfr.verify_frame1_replay_bytes(valid_trace + b"\x01\x00\x00\x00\x00")
        self.assertFalse(rep_trunc_hdr["ok"])
        self.assertFalse(rep_trunc_hdr["trace"]["valid"])
        self.assertTrue(any("truncated packet header" in e for e in rep_trunc_hdr["errors"]))

        # Truncated payload (header declares 100 bytes, only 4 bytes follow)
        bad_payload = struct.pack("<HHI", int(vfr.GpuOpcode.WriteBuffer), 0, 100) + b"\x01\x02\x03\x04"
        rep_trunc_pay = vfr.verify_frame1_replay_bytes(bad_payload)
        self.assertFalse(rep_trunc_pay["ok"])
        self.assertTrue(any("truncated WriteBuffer" in e for e in rep_trunc_pay["errors"]))

        # Bad WGPK magic
        bad_magic_pack = b"NOPE" + valid_pack[4:]
        rep_bad_magic = vfr.verify_frame1_replay_bytes(
            valid_trace, shader_pack_bytes=bad_magic_pack
        )
        self.assertFalse(rep_bad_magic["ok"])
        self.assertFalse(rep_bad_magic["shaders"]["pack_valid"])
        self.assertFalse(rep_bad_magic["checklist"]["shader_pack_ok"])
        self.assertTrue(any("wgpk_invalid_magic" in e for e in rep_bad_magic["errors"]))

        # Unsupported WGPK version
        bad_ver_pack = b"WGPK" + struct.pack("<III", 99, 0, 0)
        rep_bad_ver = vfr.verify_frame1_replay_bytes(valid_trace, shader_pack_bytes=bad_ver_pack)
        self.assertFalse(rep_bad_ver["ok"])
        self.assertTrue(any("wgpk_unsupported_version" in e for e in rep_bad_ver["errors"]))

        # Truncated WGPK header
        rep_trunc_pack = vfr.verify_frame1_replay_bytes(valid_trace, shader_pack_bytes=b"WGPK\x01")
        self.assertFalse(rep_trunc_pack["ok"])
        self.assertTrue(any("wgpk_truncated_header" in e for e in rep_trunc_pack["errors"]))

    def test_json_output_never_leaks_host_paths_usernames_or_raw_wgsl(self) -> None:
        trace_bytes = build_valid_frame1_trace(vs_code=VALID_VS_WGSL, fs_code=VALID_FS_WGSL)
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (vfr.fnv1a_64(VALID_VS_WGSL), "vertex", VALID_VS_WGSL),
                (vfr.fnv1a_64(VALID_FS_WGSL), "fragment", VALID_FS_WGSL),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "secret_workspace_dir" / "nested_captures"
            nested.mkdir(parents=True, exist_ok=True)
            trace_path = nested / "frame1_capture.pg6trace"
            pack_path = nested / "bundle.wgsl.pack"
            trace_path.write_bytes(trace_bytes)
            pack_path.write_bytes(pack_bytes)

            # Run via real CLI subprocess to exercise full argument & stdout pipeline
            proc = subprocess.run(
                [
                    sys.executable,
                    str(TOOLS_DIR / "verify_frame1_replay.py"),
                    str(trace_path),
                    "--shader-pack",
                    str(pack_path),
                    "--check-replay-assets",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, f"stderr={proc.stderr}\nstdout={proc.stdout}")
            raw_json = proc.stdout
            parsed = json.loads(raw_json)
            self.assertTrue(parsed["ok"])
            self.assertEqual(parsed["inputs"]["trace"], "frame1_capture.pg6trace")
            self.assertEqual(parsed["inputs"]["shader_pack"], "bundle.wgsl.pack")

            # Must NEVER contain host directory names, absolute paths, usernames, or raw WGSL
            self.assertNotIn("secret_workspace_dir", raw_json)
            self.assertNotIn("nested_captures", raw_json)
            self.assertNotIn(str(tmp), raw_json)
            self.assertNotIn(Path(tmp).as_posix(), raw_json)
            self.assertNotIn("@vertex", raw_json)
            self.assertNotIn("@fragment", raw_json)
            self.assertNotIn("return vec4<f32>", raw_json)

            # Check active OS username candidates do not leak in JSON
            username_candidates = {
                os.environ.get("USERNAME", "").strip(),
                os.environ.get("USER", "").strip(),
                Path.home().name.strip(),
            }
            try:
                username_candidates.add(getpass.getuser().strip())
            except Exception:  # noqa: BLE001
                pass
            for uname in username_candidates:
                if len(uname) >= 3:
                    self.assertNotIn(uname.lower(), raw_json.lower())

            # Also verify missing-file error output does not leak absolute paths
            missing_path = nested / "does_not_exist.pg6trace"
            code_missing, out_missing, _err_missing = run_cli([str(missing_path), "--json"])
            self.assertEqual(code_missing, 1)
            missing_report = json.loads(out_missing)
            self.assertFalse(missing_report["ok"])
            self.assertEqual(missing_report["inputs"]["trace"], "does_not_exist.pg6trace")
            self.assertNotIn("secret_workspace_dir", out_missing)
            self.assertNotIn(str(tmp), out_missing)
            self.assertNotIn(Path(tmp).as_posix(), out_missing)

    def test_text_output_sanitized_and_readable(self) -> None:
        trace_bytes = build_valid_frame1_trace()
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", VALID_FS_WGSL),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "sample.pg6trace"
            pack_path = Path(tmp) / "sample.wgsl.pack"
            trace_path.write_bytes(trace_bytes)
            pack_path.write_bytes(pack_bytes)

            code, out, _err = run_cli([str(trace_path), "--shader-pack", str(pack_path)])
            self.assertEqual(code, 0)
            self.assertIn("PASS", out)
            self.assertIn("sample.pg6trace", out)
            self.assertNotIn(str(tmp), out)
            self.assertNotIn(Path(tmp).as_posix(), out)

    def test_sidecar_omitted_shader_hash_resolution(self) -> None:
        # Simulate --gpu-capture omitted CreateShaderModule payloads resolved via sidecar
        vb_bytes = struct.pack("<9f", 0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.5, -0.5, 0.0)
        trace_bytes = b"".join(
            [
                pkt_begin_frame(),                                # 0
                pkt_create_buffer(1, 0x0020 | 0x0008, len(vb_bytes)),  # 1
                pkt_write_buffer(1, 0, vb_bytes),                 # 2
                pkt_create_shader_module(10, 0, b""),             # 3
                pkt_create_shader_module(11, 1, b""),             # 4
                pkt_create_render_pipeline(20, 10, 11, vertex_stride=12),  # 5
                pkt_set_pipeline(20),                             # 6
                pkt_set_vertex_buffer(0, 1, 0),                   # 7
                pkt_draw(3, 1),                                   # 8
                pkt_present(),                                    # 9
            ]
        )
        sidecar = {
            "schema": "playgta6-frame-capture",
            "schema_version": 1,
            "trace": "C:\\Users\\alice\\capture.pg6trace",
            "payloads": "omitted",
            "omitted": [
                {"packet": 3, "opcode": 5, "length": 18, "fnv1a64": f"0x{VS_HASH:016x}"},
                {"packet": 4, "opcode": 5, "length": 18, "fnv1a64": f"0x{FS_HASH:016x}"},
            ],
        }
        pack_bytes = vfr.build_wgpk_bytes(
            [
                (VS_HASH, "vertex", VALID_VS_WGSL),
                (FS_HASH, "fragment", VALID_FS_WGSL),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            t_path = Path(tmp) / "omitted.pg6trace"
            p_path = Path(tmp) / "pack.wgsl.pack"
            s_path = Path(tmp) / "omitted.pg6trace.json"
            t_path.write_bytes(trace_bytes)
            p_path.write_bytes(pack_bytes)
            s_path.write_text(json.dumps(sidecar), encoding="utf-8")

            code, out, _err = run_cli(
                [str(t_path), "--shader-pack", str(p_path), "--sidecar", str(s_path), "--json"]
            )
            self.assertEqual(code, 0, out)
            report = json.loads(out)
            self.assertTrue(report["ok"])
            self.assertEqual(
                report["shaders"]["matched_shader_hashes"],
                [f"0x{VS_HASH:016x}", f"0x{FS_HASH:016x}"],
            )
            self.assertNotIn("alice", out)


if __name__ == "__main__":
    unittest.main()
