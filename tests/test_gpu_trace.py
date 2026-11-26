import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import gpu_trace  # noqa: E402


SAMPLE_VS_WGSL = """@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(pos, 1.0);
}
"""

SAMPLE_FS_WGSL = """@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.5, 0.25, 1.0);
}
"""


class TestGpuTraceDisassembler(unittest.TestCase):
    def test_abi_struct_sizes_match_header(self) -> None:
        self.assertEqual(gpu_trace.ABI_VERSION, 1)
        self.assertEqual(gpu_trace.HEADER_SIZE, 8)
        self.assertEqual(gpu_trace.CLEAR_COLOR_STRUCT.size, 16)
        self.assertEqual(gpu_trace.CREATE_BUFFER_STRUCT.size, 16)
        self.assertEqual(gpu_trace.WRITE_BUFFER_STRUCT.size, 16)
        self.assertEqual(gpu_trace.CREATE_SHADER_MODULE_STRUCT.size, 16)
        self.assertEqual(gpu_trace.DRAW_STRUCT.size, 16)
        self.assertEqual(gpu_trace.CREATE_RENDER_PIPELINE_STRUCT.size, 24)
        self.assertEqual(gpu_trace.SET_PIPELINE_STRUCT.size, 4)
        self.assertEqual(gpu_trace.SET_VERTEX_BUFFER_STRUCT.size, 16)
        self.assertEqual(gpu_trace.DESTROY_RESOURCE_STRUCT.size, 4)
        self.assertEqual(gpu_trace.CREATE_TEXTURE_STRUCT.size, 32)
        self.assertEqual(gpu_trace.WRITE_TEXTURE_STRUCT.size, 48)
        self.assertEqual(gpu_trace.CREATE_SAMPLER_STRUCT.size, 32)
        self.assertEqual(gpu_trace.BIND_GROUP_ENTRY_STRUCT.size, 32)
        self.assertEqual(gpu_trace.CREATE_BIND_GROUP_STRUCT.size, 16)
        self.assertEqual(gpu_trace.SET_BIND_GROUP_STRUCT.size, 8)
        self.assertEqual(gpu_trace.SET_INDEX_BUFFER_STRUCT.size, 16)
        self.assertEqual(gpu_trace.DRAW_INDEXED_STRUCT.size, 20)
        self.assertEqual(gpu_trace.CREATE_COMPUTE_PIPELINE_STRUCT.size, 16)
        self.assertEqual(gpu_trace.SET_COMPUTE_PIPELINE_STRUCT.size, 4)
        self.assertEqual(gpu_trace.DISPATCH_COMPUTE_STRUCT.size, 12)

    def test_decode_all_frozen_opcodes_0_through_11_roundtrip(self) -> None:
        vertex_blob = bytes(range(64))
        stream = b"".join(
            [
                gpu_trace.encode_nop(flags=0x0003),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_clear_color(0.25, 0.5, 0.75, 1.0),
                gpu_trace.encode_create_buffer(
                    resource_id=10,
                    usage_flags=0x0020 | 0x0008,  # VERTEX | COPY_DST
                    byte_size=1024,
                ),
                gpu_trace.encode_write_buffer(
                    resource_id=10,
                    dst_offset=16,
                    data=vertex_blob,
                    reserved=7,
                ),
                gpu_trace.encode_create_shader_module(
                    resource_id=20,
                    stage=gpu_trace.GpuShaderStage.Vertex,
                    wgsl_code=SAMPLE_VS_WGSL,
                    reserved=0,
                ),
                gpu_trace.encode_create_shader_module(
                    resource_id=21,
                    stage=gpu_trace.GpuShaderStage.Fragment,
                    wgsl_code=SAMPLE_FS_WGSL,
                    reserved=0,
                ),
                gpu_trace.encode_create_render_pipeline(
                    resource_id=30,
                    vs_shader_id=20,
                    fs_shader_id=21,
                    vertex_stride=12,
                    topology=gpu_trace.GpuPrimitiveTopology.TriangleStrip,
                    pipeline_flags=0x1234,
                ),
                gpu_trace.encode_set_pipeline(pipeline_id=30),
                gpu_trace.encode_set_vertex_buffer(slot=1, buffer_id=10, offset=256),
                gpu_trace.encode_draw(
                    vertex_count=36,
                    instance_count=4,
                    first_vertex=3,
                    first_instance=2,
                ),
                gpu_trace.encode_present(),
                gpu_trace.encode_destroy_resource(resource_id=10),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.diagnostics, ())
        self.assertEqual(trace.consumed_bytes, len(stream))
        self.assertEqual(trace.total_bytes, len(stream))
        self.assertEqual(trace.packet_count, 13)
        self.assertEqual(trace.draw_calls, 1)
        self.assertEqual(trace.unknown_opcode_count, 0)

        # 0: Nop
        p0 = trace.packets[0]
        self.assertEqual(p0.opcode, 0)
        self.assertEqual(p0.name, "Nop")
        self.assertEqual(p0.flags, 0x0003)
        self.assertEqual(p0.payload_bytes, 0)
        self.assertEqual(p0.fields, {})

        # 1: BeginFrame
        p1 = trace.packets[1]
        self.assertEqual(p1.opcode, 1)
        self.assertEqual(p1.name, "BeginFrame")
        self.assertEqual(p1.payload_bytes, 0)

        # 2: ClearColor
        p2 = trace.packets[2]
        self.assertEqual(p2.opcode, 2)
        self.assertEqual(p2.name, "ClearColor")
        self.assertAlmostEqual(p2.fields["r"], 0.25, places=6)
        self.assertAlmostEqual(p2.fields["g"], 0.5, places=6)
        self.assertAlmostEqual(p2.fields["b"], 0.75, places=6)
        self.assertAlmostEqual(p2.fields["a"], 1.0, places=6)

        # 3: CreateBuffer
        p3 = trace.packets[3]
        self.assertEqual(p3.opcode, 3)
        self.assertEqual(p3.name, "CreateBuffer")
        self.assertEqual(p3.fields["id"], 10)
        self.assertEqual(p3.fields["usage_flags"], 0x0028)
        self.assertEqual(p3.fields["usage_names"], ["COPY_DST", "VERTEX"])
        self.assertEqual(p3.fields["byte_size"], 1024)

        # 4: WriteBuffer (metadata + preview/CRC32, without dumping full 64B buffer)
        p4 = trace.packets[4]
        self.assertEqual(p4.opcode, 4)
        self.assertEqual(p4.name, "WriteBuffer")
        self.assertEqual(p4.fields["id"], 10)
        self.assertEqual(p4.fields["dst_offset"], 16)
        self.assertEqual(p4.fields["data_len"], 64)
        self.assertEqual(p4.fields["reserved"], 7)
        expected_crc = f"0x{zlib.crc32(vertex_blob) & 0xFFFFFFFF:08x}"
        self.assertEqual(p4.fields["crc32"], expected_crc)
        self.assertEqual(p4.fields["preview_bytes"], gpu_trace.DEFAULT_PREVIEW_BYTES)
        self.assertEqual(
            p4.fields["preview_hex"],
            vertex_blob[: gpu_trace.DEFAULT_PREVIEW_BYTES].hex(),
        )
        self.assertTrue(p4.fields["truncated_preview"])

        # 5: CreateShaderModule (Vertex + Fragment)
        p5 = trace.packets[5]
        self.assertEqual(p5.opcode, 5)
        self.assertEqual(p5.name, "CreateShaderModule")
        self.assertEqual(p5.fields["id"], 20)
        self.assertEqual(p5.fields["stage"], 0)
        self.assertEqual(p5.fields["stage_name"], "Vertex")
        self.assertEqual(p5.fields["code_len"], len(SAMPLE_VS_WGSL.encode("utf-8")))
        self.assertEqual(p5.fields["line_count"], 4)
        self.assertEqual(p5.fields["char_count"], len(SAMPLE_VS_WGSL))
        self.assertEqual(
            p5.fields["wgsl_hash_fnv1a64"],
            f"0x{gpu_trace.fnv1a_64(SAMPLE_VS_WGSL.encode('utf-8')):016x}",
        )
        self.assertTrue(p5.fields["utf8_valid"])

        p6 = trace.packets[6]
        self.assertEqual(p6.fields["id"], 21)
        self.assertEqual(p6.fields["stage_name"], "Fragment")

        # 8: CreateRenderPipeline
        p7 = trace.packets[7]
        self.assertEqual(p7.opcode, 8)
        self.assertEqual(p7.name, "CreateRenderPipeline")
        self.assertEqual(p7.fields["id"], 30)
        self.assertEqual(p7.fields["vs_shader_id"], 20)
        self.assertEqual(p7.fields["fs_shader_id"], 21)
        self.assertEqual(p7.fields["vertex_stride"], 12)
        self.assertEqual(p7.fields["topology"], 1)
        self.assertEqual(p7.fields["topology_name"], "TriangleStrip")
        self.assertEqual(p7.fields["topology_webgpu"], "triangle-strip")
        self.assertEqual(p7.fields["flags"], 0x1234)

        # 9: SetPipeline
        p8 = trace.packets[8]
        self.assertEqual(p8.opcode, 9)
        self.assertEqual(p8.name, "SetPipeline")
        self.assertEqual(p8.fields["pipeline_id"], 30)

        # 10: SetVertexBuffer
        p9 = trace.packets[9]
        self.assertEqual(p9.opcode, 10)
        self.assertEqual(p9.name, "SetVertexBuffer")
        self.assertEqual(p9.fields["slot"], 1)
        self.assertEqual(p9.fields["buffer_id"], 10)
        self.assertEqual(p9.fields["offset"], 256)

        # 6: Draw
        p10 = trace.packets[10]
        self.assertEqual(p10.opcode, 6)
        self.assertEqual(p10.name, "Draw")
        self.assertEqual(p10.fields["vertex_count"], 36)
        self.assertEqual(p10.fields["instance_count"], 4)
        self.assertEqual(p10.fields["first_vertex"], 3)
        self.assertEqual(p10.fields["first_instance"], 2)

        # 7: Present
        p11 = trace.packets[11]
        self.assertEqual(p11.opcode, 7)
        self.assertEqual(p11.name, "Present")
        self.assertEqual(p11.payload_bytes, 0)

        # 11: DestroyResource
        p12 = trace.packets[12]
        self.assertEqual(p12.opcode, 11)
        self.assertEqual(p12.name, "DestroyResource")
        self.assertEqual(p12.fields["id"], 10)

    def test_fnv1a_64_known_vectors(self) -> None:
        # Standard FNV-1a 64-bit reference test vectors.
        self.assertEqual(gpu_trace.fnv1a_64(b""), 0xCBF29CE484222325)
        self.assertEqual(gpu_trace.fnv1a_64(b"a"), 0xAF63DC4C8601EC8C)
        self.assertEqual(gpu_trace.fnv1a_64(b"foobar"), 0x85944171F73967E8)

    def test_all_buffer_usage_flags_and_unknown_bits(self) -> None:
        all_known = 0x03FF
        names, unknown = gpu_trace.decode_buffer_usage(all_known)
        self.assertEqual(unknown, 0)
        self.assertEqual(
            names,
            [
                "MAP_READ",
                "MAP_WRITE",
                "COPY_SRC",
                "COPY_DST",
                "INDEX",
                "VERTEX",
                "UNIFORM",
                "STORAGE",
                "INDIRECT",
                "QUERY_RESOLVE",
            ],
        )

        # Unknown high bit triggers warning in non-strict mode, error in strict mode.
        pkt = gpu_trace.encode_create_buffer(1, 0x8020, 64)
        t_lenient = gpu_trace.decode_stream(pkt, strict=False)
        self.assertTrue(t_lenient.ok)
        self.assertEqual(t_lenient.packets[0].fields["usage_names"], ["VERTEX", "UNKNOWN(0x8000)"])
        self.assertEqual(len(t_lenient.diagnostics), 1)
        self.assertEqual(t_lenient.diagnostics[0].severity, "warning")
        self.assertEqual(t_lenient.diagnostics[0].code, "unknown_buffer_usage_bits")

        t_strict = gpu_trace.decode_stream(pkt, strict=True)
        self.assertFalse(t_strict.ok)
        self.assertEqual(t_strict.diagnostics[0].severity, "error")

    def test_frame_splitting_and_histograms(self) -> None:
        f0 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_clear_color(0.0, 0.0, 0.0, 1.0),
                gpu_trace.encode_draw(3, 1, 0, 0),
                gpu_trace.encode_draw(6, 2, 0, 0),
                gpu_trace.encode_present(),
            ]
        )
        f1 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_set_pipeline(1),
                gpu_trace.encode_draw(12, 1, 0, 0),
                gpu_trace.encode_present(),
            ]
        )
        trailing = gpu_trace.encode_begin_frame() + gpu_trace.encode_nop()

        trace = gpu_trace.decode_stream(f0 + f1 + trailing)
        self.assertEqual(trace.frame_count, 3)

        self.assertEqual(trace.frames[0].frame_index, 0)
        self.assertEqual(trace.frames[0].packet_count, 5)
        self.assertEqual(trace.frames[0].draw_calls, 2)
        self.assertEqual(trace.frames[0].bytes, len(f0))
        self.assertTrue(trace.frames[0].complete)
        self.assertTrue(trace.frames[0].has_begin_frame)
        self.assertEqual(
            trace.frames[0].opcode_histogram,
            {"BeginFrame": 1, "ClearColor": 1, "Draw": 2, "Present": 1},
        )

        self.assertEqual(trace.frames[1].frame_index, 1)
        self.assertEqual(trace.frames[1].packet_count, 4)
        self.assertEqual(trace.frames[1].draw_calls, 1)
        self.assertEqual(trace.frames[1].bytes, len(f1))
        self.assertTrue(trace.frames[1].complete)
        self.assertEqual(
            trace.frames[1].opcode_histogram,
            {"BeginFrame": 1, "SetPipeline": 1, "Draw": 1, "Present": 1},
        )

        self.assertEqual(trace.frames[2].frame_index, 2)
        self.assertEqual(trace.frames[2].packet_count, 2)
        self.assertEqual(trace.frames[2].draw_calls, 0)
        self.assertEqual(trace.frames[2].bytes, len(trailing))
        self.assertFalse(trace.frames[2].complete)
        self.assertTrue(trace.frames[2].has_begin_frame)

        self.assertEqual(
            trace.opcode_histogram,
            {
                "BeginFrame": 3,
                "ClearColor": 1,
                "SetPipeline": 1,
                "Draw": 3,
                "Present": 2,
                "Nop": 1,
            },
        )

    def test_unknown_future_opcodes_forward_compatibility(self) -> None:
        future_payload = b"\xde\xad\xbe\xef" * 10
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_packet(200, future_payload, flags=0x0042),
                gpu_trace.encode_packet(99, b""),
                gpu_trace.encode_present(),
            ]
        )

        # Non-strict mode handles unknown opcodes (>= 35) cleanly without errors.
        lenient = gpu_trace.decode_stream(stream, strict=False)
        self.assertTrue(lenient.ok)
        self.assertEqual(lenient.diagnostics, ())
        self.assertEqual(lenient.packet_count, 4)
        self.assertEqual(lenient.unknown_opcode_count, 2)

        u200 = lenient.packets[1]
        self.assertFalse(u200.known)
        self.assertTrue(u200.valid)
        self.assertEqual(u200.opcode, 200)
        self.assertEqual(u200.name, "Unknown(200)")
        self.assertEqual(u200.flags, 0x0042)
        self.assertEqual(u200.payload_bytes, len(future_payload))
        self.assertEqual(u200.fields["payload_bytes"], len(future_payload))
        self.assertEqual(u200.fields["crc32"], gpu_trace.crc32_hex(future_payload))
        self.assertTrue(u200.fields["truncated_preview"])

        u99 = lenient.packets[2]
        self.assertEqual(u99.name, "Unknown(99)")
        self.assertEqual(u99.payload_bytes, 0)

        # Strict mode flags unknown opcodes as structured errors.
        strict = gpu_trace.decode_stream(stream, strict=True)
        self.assertFalse(strict.ok)
        self.assertTrue(strict.has_errors)
        self.assertEqual([d.code for d in strict.diagnostics], ["unknown_opcode", "unknown_opcode"])
        self.assertFalse(strict.packets[1].valid)

    def test_malformed_and_truncated_stream_diagnostics(self) -> None:
        # 1. Truncated header (5 trailing bytes)
        valid_prefix = gpu_trace.encode_nop()
        trunc_hdr = gpu_trace.decode_stream(valid_prefix + b"\x01\x02\x03\x04\x05")
        self.assertFalse(trunc_hdr.ok)
        self.assertEqual(trunc_hdr.packet_count, 1)
        self.assertEqual(trunc_hdr.consumed_bytes, 8)
        self.assertEqual(len(trunc_hdr.diagnostics), 1)
        self.assertEqual(trunc_hdr.diagnostics[0].code, "truncated_header")
        self.assertEqual(trunc_hdr.diagnostics[0].severity, "error")
        self.assertEqual(trunc_hdr.diagnostics[0].offset, 8)

        # 2. Truncated payload (header declares 100 bytes, only 4 follow)
        bad_header = struct.pack("<HHI", gpu_trace.GpuOpcode.ClearColor, 0, 100) + b"\x00" * 4
        trunc_pay = gpu_trace.decode_stream(bad_header)
        self.assertFalse(trunc_pay.ok)
        self.assertEqual(trunc_pay.packet_count, 0)
        self.assertEqual(trunc_pay.diagnostics[0].code, "truncated_payload")
        self.assertEqual(trunc_pay.diagnostics[0].severity, "error")

        # 3. Fixed payload too short (ClearColor with only 8 bytes) followed by valid Present
        short_clear = gpu_trace.encode_packet(gpu_trace.GpuOpcode.ClearColor, b"\x00" * 8)
        recovered = gpu_trace.decode_stream(short_clear + gpu_trace.encode_present())
        self.assertFalse(recovered.ok)
        self.assertEqual(recovered.packet_count, 2)
        self.assertFalse(recovered.packets[0].valid)
        self.assertTrue(recovered.packets[1].valid)
        self.assertEqual(recovered.packets[1].name, "Present")
        self.assertEqual(recovered.diagnostics[0].code, "payload_too_short")

        # 4. Fixed payload too large (SetPipeline with 8 bytes instead of 4)
        oversize_set_pipe = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.SetPipeline,
            struct.pack("<II", 42, 0xDEADBEEF),
        )
        over = gpu_trace.decode_stream(oversize_set_pipe, strict=False)
        self.assertTrue(over.ok)
        self.assertTrue(over.has_warnings)
        self.assertEqual(over.packets[0].fields["pipeline_id"], 42)
        self.assertEqual(over.packets[0].fields["trailing_bytes"], 4)
        self.assertEqual(over.diagnostics[0].code, "payload_size_mismatch")
        self.assertEqual(over.diagnostics[0].severity, "warning")

        over_strict = gpu_trace.decode_stream(oversize_set_pipe, strict=True)
        self.assertFalse(over_strict.ok)
        self.assertEqual(over_strict.diagnostics[0].severity, "error")

        # 5. Variable payload length mismatch (WriteBuffer claims 32 bytes, only 4 provided)
        bad_wb = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.WriteBuffer,
            struct.pack("<IIII", 5, 0, 32, 0) + b"abcd",
        )
        wb_trace = gpu_trace.decode_stream(bad_wb)
        self.assertFalse(wb_trace.ok)
        self.assertFalse(wb_trace.packets[0].valid)
        self.assertEqual(wb_trace.packets[0].fields["available_bytes"], 4)
        self.assertEqual(wb_trace.diagnostics[0].code, "variable_payload_truncated")

    def test_text_and_json_serialization(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_clear_color(0.1, 0.2, 0.3, 1.0),
                gpu_trace.encode_draw(3, 1, 0, 0),
                gpu_trace.encode_present(),
            ]
        )
        trace = gpu_trace.decode_stream(stream)

        full_text = gpu_trace.format_trace_text(trace)
        self.assertIn("playgta6 GPU Trace (ABI v1)", full_text)
        self.assertIn("BeginFrame", full_text)
        self.assertIn("ClearColor", full_text)
        self.assertIn("Draw", full_text)
        self.assertIn("Present", full_text)
        self.assertIn("Frame 000 [complete]", full_text)

        stats_text = gpu_trace.format_trace_text(trace, stats_only=True)
        self.assertNotIn("Packets:\n", stats_text)
        self.assertIn("Opcode Histogram:", stats_text)

        parsed_full = json.loads(gpu_trace.trace_to_json(trace))
        self.assertEqual(parsed_full["abi_version"], 1)
        self.assertEqual(parsed_full["packet_count"], 4)
        self.assertEqual(len(parsed_full["packets"]), 4)
        self.assertEqual(len(parsed_full["frames"]), 1)
        self.assertEqual(parsed_full["frames"][0]["draw_calls"], 1)

        parsed_stats = json.loads(gpu_trace.trace_to_json(trace, stats_only=True))
        self.assertNotIn("packets", parsed_stats)
        self.assertEqual(parsed_stats["packet_count"], 4)

    def test_cli_execution_modes(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_create_shader_module(
                    1,
                    gpu_trace.GpuShaderStage.Compute,
                    "@compute @workgroup_size(1) fn cs_main() {}",
                ),
                gpu_trace.encode_packet(42, b"\x01\x02\x03"),
                gpu_trace.encode_present(),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "capture.bin"
            trace_path.write_bytes(stream)

            # Default text mode via subprocess
            proc = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "gpu_trace.py"), str(trace_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertIn("CreateShaderModule", proc.stdout)
            self.assertIn("stage=Compute", proc.stdout)
            self.assertIn("Unknown(42)", proc.stdout)

            # --json --stats mode via in-process main()
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = gpu_trace.main([str(trace_path), "--json", "--stats"])
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["unknown_opcode_count"], 1)
            self.assertNotIn("packets", payload)

            # --strict fails on Unknown(42)
            buf_strict = io.StringIO()
            with redirect_stdout(buf_strict):
                rc_strict = gpu_trace.main([str(trace_path), "--json", "--strict"])
            self.assertEqual(rc_strict, 1)
            strict_payload = json.loads(buf_strict.getvalue())
            self.assertFalse(strict_payload["ok"])
            self.assertEqual(strict_payload["diagnostics"][0]["code"], "unknown_opcode")

            # Missing file returns non-zero exit code
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc_missing = gpu_trace.main([str(Path(tmp) / "does_not_exist.bin")])
            self.assertEqual(rc_missing, 2)
            self.assertIn("trace file not found", err_buf.getvalue())

    def test_edge_cases_empty_preamble_and_invalid_enums(self) -> None:
        # Empty stream
        empty = gpu_trace.decode_stream(b"", strict=True)
        self.assertTrue(empty.ok)
        self.assertEqual(empty.packet_count, 0)
        self.assertEqual(empty.frame_count, 0)
        self.assertIn("(none)", gpu_trace.format_trace_text(empty))

        # Setup preamble before BeginFrame + invalid UTF-8 WGSL + unknown stage/topology
        stream = b"".join(
            [
                gpu_trace.encode_create_shader_module(1, 99, b"\xff\xfe@vertex"),
                gpu_trace.encode_create_render_pipeline(2, 1, 0, 12, 77, 0),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_present(),
            ]
        )
        trace = gpu_trace.decode_stream(stream, strict=False)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.frame_count, 2)
        self.assertFalse(trace.frames[0].has_begin_frame)
        self.assertFalse(trace.frames[0].complete)
        self.assertTrue(trace.frames[1].has_begin_frame)
        self.assertTrue(trace.frames[1].complete)
        self.assertEqual(trace.packets[0].fields["stage_name"], "Unknown(99)")
        self.assertFalse(trace.packets[0].fields["utf8_valid"])
        self.assertEqual(trace.packets[1].fields["topology_name"], "Unknown(77)")
        self.assertIsNone(trace.packets[1].fields["topology_webgpu"])
        diag_codes = {d.code for d in trace.diagnostics}
        self.assertEqual(
            diag_codes,
            {"unknown_shader_stage", "invalid_utf8_wgsl", "unknown_topology"},
        )

        # Strict mode promotes those warnings to errors
        strict_trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertFalse(strict_trace.ok)
        self.assertTrue(all(d.severity == "error" for d in strict_trace.diagnostics))

        # encode_packet range validation
        with self.assertRaises(ValueError):
            gpu_trace.encode_packet(0x10000)
        with self.assertRaises(ValueError):
            gpu_trace.encode_packet(0, b"", flags=0x10000)

    def test_decode_all_g5_opcodes_12_through_21_roundtrip(self) -> None:
        texel_blob = bytes((i * 7) & 0xFF for i in range(64))
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_create_texture(
                    resource_id=100,
                    width=512,
                    height=256,
                    depth_or_layers=1,
                    mip_levels=4,
                    texture_format=gpu_trace.GpuTextureFormat.Bc7RgbaUnorm,
                    usage_flags=0x0002 | 0x0004,  # COPY_DST | TEXTURE_BINDING
                    dimension=gpu_trace.GpuTextureDimension.Dim2D,
                ),
                gpu_trace.encode_write_texture(
                    texture_id=100,
                    width=16,
                    height=16,
                    data=texel_blob,
                    mip_level=1,
                    origin_x=4,
                    origin_y=8,
                    origin_z=0,
                    depth_or_layers=1,
                    bytes_per_row=256,
                    rows_per_image=16,
                    reserved=3,
                ),
                gpu_trace.encode_create_sampler(
                    resource_id=101,
                    address_u=gpu_trace.GpuAddressMode.Repeat,
                    address_v=gpu_trace.GpuAddressMode.MirrorRepeat,
                    address_w=gpu_trace.GpuAddressMode.ClampToEdge,
                    mag_filter=gpu_trace.GpuFilterMode.Linear,
                    min_filter=gpu_trace.GpuFilterMode.Linear,
                    mipmap_filter=gpu_trace.GpuFilterMode.Nearest,
                    max_anisotropy=8,
                ),
                gpu_trace.encode_create_bind_group(
                    resource_id=102,
                    pipeline_id=30,
                    group_index=1,
                    entries=[
                        gpu_trace.BindGroupEntry(
                            binding=0,
                            resource_type=gpu_trace.GpuBindResourceType.Buffer,
                            resource_id=10,
                            offset=64,
                            size=256,
                            reserved=0,
                        ),
                        {
                            "binding": 1,
                            "resource_type": gpu_trace.GpuBindResourceType.TextureView,
                            "resource_id": 100,
                        },
                        gpu_trace.encode_bind_group_entry(
                            binding=2,
                            resource_type=gpu_trace.GpuBindResourceType.Sampler,
                            resource_id=101,
                        ),
                    ],
                ),
                gpu_trace.encode_set_bind_group(group_index=1, bind_group_id=102),
                gpu_trace.encode_set_index_buffer(
                    buffer_id=11,
                    index_format=gpu_trace.GpuIndexFormat.Uint32,
                    offset=128,
                ),
                gpu_trace.encode_draw_indexed(
                    index_count=72,
                    instance_count=3,
                    first_index=6,
                    base_vertex=-16,
                    first_instance=2,
                ),
                gpu_trace.encode_create_compute_pipeline(
                    resource_id=200,
                    cs_shader_id=22,
                    pipeline_flags=0x0005,
                    reserved=9,
                ),
                gpu_trace.encode_set_compute_pipeline(pipeline_id=200),
                gpu_trace.encode_dispatch_compute(workgroup_x=16, workgroup_y=8, workgroup_z=2),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.diagnostics, ())
        self.assertEqual(trace.consumed_bytes, len(stream))
        self.assertEqual(trace.packet_count, 12)
        self.assertEqual(trace.unknown_opcode_count, 0)

        # 12: CreateTexture
        p1 = trace.packets[1]
        self.assertEqual(p1.opcode, 12)
        self.assertEqual(p1.name, "CreateTexture")
        self.assertEqual(p1.payload_bytes, 32)
        self.assertEqual(p1.fields["id"], 100)
        self.assertEqual(p1.fields["width"], 512)
        self.assertEqual(p1.fields["height"], 256)
        self.assertEqual(p1.fields["depth_or_layers"], 1)
        self.assertEqual(p1.fields["mip_levels"], 4)
        self.assertEqual(p1.fields["format"], 5)
        self.assertEqual(p1.fields["format_name"], "Bc7RgbaUnorm")
        self.assertEqual(p1.fields["format_webgpu"], "bc7-rgba-unorm")
        self.assertEqual(p1.fields["usage_flags"], 0x0006)
        self.assertEqual(p1.fields["usage_names"], ["COPY_DST", "TEXTURE_BINDING"])
        self.assertEqual(p1.fields["dimension"], 1)
        self.assertEqual(p1.fields["dimension_name"], "Dim2D")
        self.assertEqual(p1.fields["dimension_webgpu"], "2d")

        # 13: WriteTexture
        p2 = trace.packets[2]
        self.assertEqual(p2.opcode, 13)
        self.assertEqual(p2.name, "WriteTexture")
        self.assertEqual(p2.payload_bytes, 48 + len(texel_blob))
        self.assertEqual(p2.fields["texture_id"], 100)
        self.assertEqual(p2.fields["mip_level"], 1)
        self.assertEqual(p2.fields["origin_x"], 4)
        self.assertEqual(p2.fields["origin_y"], 8)
        self.assertEqual(p2.fields["origin_z"], 0)
        self.assertEqual(p2.fields["width"], 16)
        self.assertEqual(p2.fields["height"], 16)
        self.assertEqual(p2.fields["depth_or_layers"], 1)
        self.assertEqual(p2.fields["bytes_per_row"], 256)
        self.assertEqual(p2.fields["rows_per_image"], 16)
        self.assertEqual(p2.fields["data_len"], 64)
        self.assertEqual(p2.fields["reserved"], 3)
        self.assertEqual(p2.fields["crc32"], gpu_trace.crc32_hex(texel_blob))
        self.assertEqual(p2.fields["preview_bytes"], gpu_trace.DEFAULT_PREVIEW_BYTES)
        self.assertEqual(
            p2.fields["preview_hex"],
            texel_blob[: gpu_trace.DEFAULT_PREVIEW_BYTES].hex(),
        )
        self.assertTrue(p2.fields["truncated_preview"])

        # 14: CreateSampler
        p3 = trace.packets[3]
        self.assertEqual(p3.opcode, 14)
        self.assertEqual(p3.name, "CreateSampler")
        self.assertEqual(p3.payload_bytes, 32)
        self.assertEqual(p3.fields["id"], 101)
        self.assertEqual(p3.fields["address_u_name"], "Repeat")
        self.assertEqual(p3.fields["address_u_webgpu"], "repeat")
        self.assertEqual(p3.fields["address_v_name"], "MirrorRepeat")
        self.assertEqual(p3.fields["address_v_webgpu"], "mirror-repeat")
        self.assertEqual(p3.fields["address_w_name"], "ClampToEdge")
        self.assertEqual(p3.fields["address_w_webgpu"], "clamp-to-edge")
        self.assertEqual(p3.fields["mag_filter_name"], "Linear")
        self.assertEqual(p3.fields["min_filter_name"], "Linear")
        self.assertEqual(p3.fields["mipmap_filter_name"], "Nearest")
        self.assertEqual(p3.fields["max_anisotropy"], 8)

        # 15: CreateBindGroup (multi-entry)
        p4 = trace.packets[4]
        self.assertEqual(p4.opcode, 15)
        self.assertEqual(p4.name, "CreateBindGroup")
        self.assertEqual(p4.payload_bytes, 16 + 3 * 32)
        self.assertEqual(p4.fields["id"], 102)
        self.assertEqual(p4.fields["pipeline_id"], 30)
        self.assertEqual(p4.fields["group_index"], 1)
        self.assertEqual(p4.fields["entry_count"], 3)
        self.assertEqual(len(p4.fields["entries"]), 3)
        self.assertEqual(p4.fields["entries"][0]["binding"], 0)
        self.assertEqual(p4.fields["entries"][0]["resource_type_name"], "Buffer")
        self.assertEqual(p4.fields["entries"][0]["resource_id"], 10)
        self.assertEqual(p4.fields["entries"][0]["offset"], 64)
        self.assertEqual(p4.fields["entries"][0]["size"], 256)
        self.assertEqual(p4.fields["entries"][1]["binding"], 1)
        self.assertEqual(p4.fields["entries"][1]["resource_type_name"], "TextureView")
        self.assertEqual(p4.fields["entries"][1]["resource_id"], 100)
        self.assertEqual(p4.fields["entries"][2]["binding"], 2)
        self.assertEqual(p4.fields["entries"][2]["resource_type_name"], "Sampler")
        self.assertEqual(p4.fields["entries"][2]["resource_id"], 101)

        # 16: SetBindGroup
        p5 = trace.packets[5]
        self.assertEqual(p5.opcode, 16)
        self.assertEqual(p5.name, "SetBindGroup")
        self.assertEqual(p5.payload_bytes, 8)
        self.assertEqual(p5.fields["group_index"], 1)
        self.assertEqual(p5.fields["bind_group_id"], 102)

        # 17: SetIndexBuffer
        p6 = trace.packets[6]
        self.assertEqual(p6.opcode, 17)
        self.assertEqual(p6.name, "SetIndexBuffer")
        self.assertEqual(p6.payload_bytes, 16)
        self.assertEqual(p6.fields["buffer_id"], 11)
        self.assertEqual(p6.fields["index_format"], 1)
        self.assertEqual(p6.fields["index_format_name"], "Uint32")
        self.assertEqual(p6.fields["index_format_webgpu"], "uint32")
        self.assertEqual(p6.fields["offset"], 128)

        # 18: DrawIndexed (including negative signed int32 base_vertex)
        p7 = trace.packets[7]
        self.assertEqual(p7.opcode, 18)
        self.assertEqual(p7.name, "DrawIndexed")
        self.assertEqual(p7.payload_bytes, 20)
        self.assertEqual(p7.fields["index_count"], 72)
        self.assertEqual(p7.fields["instance_count"], 3)
        self.assertEqual(p7.fields["first_index"], 6)
        self.assertEqual(p7.fields["base_vertex"], -16)
        self.assertEqual(p7.fields["first_instance"], 2)

        # 19: CreateComputePipeline
        p8 = trace.packets[8]
        self.assertEqual(p8.opcode, 19)
        self.assertEqual(p8.name, "CreateComputePipeline")
        self.assertEqual(p8.payload_bytes, 16)
        self.assertEqual(p8.fields["id"], 200)
        self.assertEqual(p8.fields["cs_shader_id"], 22)
        self.assertEqual(p8.fields["flags"], 0x0005)
        self.assertEqual(p8.fields["reserved"], 9)

        # 20: SetComputePipeline
        p9 = trace.packets[9]
        self.assertEqual(p9.opcode, 20)
        self.assertEqual(p9.name, "SetComputePipeline")
        self.assertEqual(p9.payload_bytes, 4)
        self.assertEqual(p9.fields["pipeline_id"], 200)

        # 21: DispatchCompute
        p10 = trace.packets[10]
        self.assertEqual(p10.opcode, 21)
        self.assertEqual(p10.name, "DispatchCompute")
        self.assertEqual(p10.payload_bytes, 12)
        self.assertEqual(p10.fields["workgroup_x"], 16)
        self.assertEqual(p10.fields["workgroup_y"], 8)
        self.assertEqual(p10.fields["workgroup_z"], 2)

        # Text formatting includes readable G5 summaries
        text = gpu_trace.format_trace_text(trace)
        self.assertIn("CreateTexture", text)
        self.assertIn("fmt=bc7-rgba-unorm", text)
        self.assertIn("COPY_DST|TEXTURE_BINDING", text)
        self.assertIn("CreateBindGroup", text)
        self.assertIn("@0:Buffer(#10)[64:256]", text)
        self.assertIn("@1:TextureView(#100)", text)
        self.assertIn("@2:Sampler(#101)", text)
        self.assertIn("base_vertex=-16", text)
        self.assertIn("workgroups=(16, 8, 2)", text)

    def test_draw_indexed_negative_base_vertex_extremes(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_draw_indexed(3, base_vertex=-1),
                gpu_trace.encode_draw_indexed(6, base_vertex=-2147483648),
                gpu_trace.encode_draw_indexed(9, base_vertex=2147483647),
            ]
        )
        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.packets[0].fields["base_vertex"], -1)
        self.assertEqual(trace.packets[1].fields["base_vertex"], -2147483648)
        self.assertEqual(trace.packets[2].fields["base_vertex"], 2147483647)

    def test_g5_enums_and_invalid_enum_diagnostics(self) -> None:
        # All known texture usage flags
        names, unknown = gpu_trace.decode_texture_usage(0x001F)
        self.assertEqual(unknown, 0)
        self.assertEqual(
            names,
            [
                "COPY_SRC",
                "COPY_DST",
                "TEXTURE_BINDING",
                "STORAGE_BINDING",
                "RENDER_ATTACHMENT",
            ],
        )

        # Unknown texture format, usage bit, dimension, sampler modes, bind resource type, index fmt
        stream = b"".join(
            [
                gpu_trace.encode_create_texture(
                    resource_id=1,
                    width=32,
                    height=32,
                    texture_format=99,
                    usage_flags=0x0044,  # TEXTURE_BINDING | unknown 0x40
                    dimension=7,
                ),
                gpu_trace.encode_create_sampler(
                    resource_id=2,
                    address_u=9,
                    mag_filter=8,
                ),
                gpu_trace.encode_create_bind_group(
                    resource_id=3,
                    pipeline_id=1,
                    group_index=0,
                    entries=[gpu_trace.BindGroupEntry(binding=0, resource_type=42, resource_id=1)],
                ),
                gpu_trace.encode_set_index_buffer(buffer_id=4, index_format=5, offset=0),
            ]
        )

        lenient = gpu_trace.decode_stream(stream, strict=False)
        self.assertTrue(lenient.ok)
        self.assertEqual(lenient.packets[0].fields["format_name"], "Unknown(99)")
        self.assertIsNone(lenient.packets[0].fields["format_webgpu"])
        self.assertEqual(
            lenient.packets[0].fields["usage_names"],
            ["TEXTURE_BINDING", "UNKNOWN(0x40)"],
        )
        self.assertEqual(lenient.packets[0].fields["dimension_name"], "Unknown(7)")
        self.assertEqual(lenient.packets[1].fields["address_u_name"], "Unknown(9)")
        self.assertEqual(lenient.packets[1].fields["mag_filter_name"], "Unknown(8)")
        self.assertEqual(
            lenient.packets[2].fields["entries"][0]["resource_type_name"],
            "Unknown(42)",
        )
        self.assertEqual(lenient.packets[3].fields["index_format_name"], "Unknown(5)")
        self.assertIsNone(lenient.packets[3].fields["index_format_webgpu"])
        self.assertEqual(
            {d.code for d in lenient.diagnostics},
            {
                "unknown_texture_format",
                "unknown_texture_usage_bits",
                "unknown_texture_dimension",
                "unknown_address_mode",
                "unknown_filter_mode",
                "unknown_bind_resource_type",
                "unknown_index_format",
            },
        )

        strict = gpu_trace.decode_stream(stream, strict=True)
        self.assertFalse(strict.ok)
        self.assertTrue(all(d.severity == "error" for d in strict.diagnostics))

    def test_write_texture_and_bind_group_length_mismatch_diagnostics(self) -> None:
        # 1. WriteTexture header too short (< 48 bytes)
        wt_short_hdr = gpu_trace.encode_packet(gpu_trace.GpuOpcode.WriteTexture, b"\x00" * 24)
        t_wt_hdr = gpu_trace.decode_stream(wt_short_hdr)
        self.assertFalse(t_wt_hdr.ok)
        self.assertFalse(t_wt_hdr.packets[0].valid)
        self.assertEqual(t_wt_hdr.diagnostics[0].code, "payload_too_short")

        # 2. WriteTexture truncated texel bytes (claims 64 bytes, only 10 follow)
        wt_head = gpu_trace.WRITE_TEXTURE_STRUCT.pack(7, 0, 0, 0, 0, 4, 4, 1, 16, 4, 64, 0)
        wt_trunc = gpu_trace.encode_packet(gpu_trace.GpuOpcode.WriteTexture, wt_head + b"\xaa" * 10)
        t_wt_trunc = gpu_trace.decode_stream(wt_trunc)
        self.assertFalse(t_wt_trunc.ok)
        self.assertFalse(t_wt_trunc.packets[0].valid)
        self.assertEqual(t_wt_trunc.packets[0].fields["data_len"], 64)
        self.assertEqual(t_wt_trunc.packets[0].fields["available_bytes"], 10)
        self.assertEqual(t_wt_trunc.diagnostics[0].code, "variable_payload_truncated")

        # 3. WriteTexture trailing extra bytes (claims 4 bytes, 8 provided)
        wt_head_small = gpu_trace.WRITE_TEXTURE_STRUCT.pack(7, 0, 0, 0, 0, 1, 1, 1, 4, 1, 4, 0)
        wt_extra = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.WriteTexture, wt_head_small + b"\x01\x02\x03\x04\xff\xff\xff\xff"
        )
        t_wt_extra = gpu_trace.decode_stream(wt_extra, strict=False)
        self.assertTrue(t_wt_extra.ok)
        self.assertTrue(t_wt_extra.has_warnings)
        self.assertEqual(t_wt_extra.packets[0].fields["trailing_bytes"], 4)
        self.assertEqual(t_wt_extra.packets[0].fields["crc32"], gpu_trace.crc32_hex(b"\x01\x02\x03\x04"))
        self.assertEqual(t_wt_extra.diagnostics[0].code, "payload_size_mismatch")
        self.assertFalse(gpu_trace.decode_stream(wt_extra, strict=True).ok)

        # 4. CreateBindGroup truncated entries (claims 2 entries = 64B, only 40B follow -> 1 complete)
        entry0 = gpu_trace.encode_bind_group_entry(
            0, gpu_trace.GpuBindResourceType.Buffer, 55, offset=0, size=128
        )
        bg_trunc = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.CreateBindGroup,
            gpu_trace.CREATE_BIND_GROUP_STRUCT.pack(9, 1, 0, 2) + entry0 + b"\x00" * 8,
        )
        t_bg_trunc = gpu_trace.decode_stream(bg_trunc)
        self.assertFalse(t_bg_trunc.ok)
        self.assertFalse(t_bg_trunc.packets[0].valid)
        self.assertEqual(t_bg_trunc.packets[0].fields["entry_count"], 2)
        self.assertEqual(len(t_bg_trunc.packets[0].fields["entries"]), 1)
        self.assertEqual(t_bg_trunc.packets[0].fields["available_bytes"], 40)
        self.assertEqual(t_bg_trunc.diagnostics[0].code, "variable_payload_truncated")

        # 5. CreateBindGroup trailing extra bytes
        bg_extra = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.CreateBindGroup,
            gpu_trace.CREATE_BIND_GROUP_STRUCT.pack(9, 1, 0, 1) + entry0 + b"\xbe\xef",
        )
        t_bg_extra = gpu_trace.decode_stream(bg_extra, strict=False)
        self.assertTrue(t_bg_extra.ok)
        self.assertEqual(t_bg_extra.packets[0].fields["trailing_bytes"], 2)
        self.assertEqual(t_bg_extra.diagnostics[0].code, "payload_size_mismatch")
        self.assertFalse(gpu_trace.decode_stream(bg_extra, strict=True).ok)

        # 6. Raw bind group entry length validation on encode
        with self.assertRaises(ValueError):
            gpu_trace.encode_create_bind_group(1, 1, 0, [b"\x00" * 16])

    def test_compute_and_textured_indexed_draw_frame_summaries(self) -> None:
        frame0 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_create_texture(10, 64, 64, usage_flags=0x0006),
                gpu_trace.encode_create_texture(11, 64, 64, usage_flags=0x0014),
                gpu_trace.encode_create_sampler(12),
                gpu_trace.encode_create_compute_pipeline(20, cs_shader_id=1),
                gpu_trace.encode_create_bind_group(
                    30,
                    pipeline_id=20,
                    group_index=0,
                    entries=[
                        gpu_trace.BindGroupEntry(0, gpu_trace.GpuBindResourceType.Buffer, 1),
                    ],
                ),
                gpu_trace.encode_create_bind_group(
                    31,
                    pipeline_id=21,
                    group_index=0,
                    entries=[
                        gpu_trace.BindGroupEntry(0, gpu_trace.GpuBindResourceType.TextureView, 10),
                        gpu_trace.BindGroupEntry(1, gpu_trace.GpuBindResourceType.Sampler, 12),
                    ],
                ),
                # Compute pass #1 (two dispatches back-to-back)
                gpu_trace.encode_set_compute_pipeline(20),
                gpu_trace.encode_set_bind_group(0, 30),
                gpu_trace.encode_dispatch_compute(8, 8, 1),
                gpu_trace.encode_dispatch_compute(4, 4, 1),
                # Render pass (1 DrawIndexed + 1 Draw)
                gpu_trace.encode_set_pipeline(21),
                gpu_trace.encode_set_bind_group(0, 31),
                gpu_trace.encode_set_index_buffer(2, gpu_trace.GpuIndexFormat.Uint16, 0),
                gpu_trace.encode_draw_indexed(36, 1, 0, -4, 0),
                gpu_trace.encode_draw(3, 1, 0, 0),
                # Compute pass #2 (after render pass -> increments compute_passes)
                gpu_trace.encode_set_compute_pipeline(20),
                gpu_trace.encode_dispatch_compute(2, 2, 1),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(frame0, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.frame_count, 1)

        f0 = trace.frames[0]
        self.assertTrue(f0.complete)
        self.assertTrue(f0.has_begin_frame)
        self.assertEqual(f0.draw_calls, 2)
        self.assertEqual(f0.draw_indexed_calls, 1)
        self.assertEqual(f0.compute_dispatches, 3)
        self.assertEqual(f0.compute_passes, 2)
        self.assertEqual(f0.textures_created, 2)
        self.assertEqual(f0.bind_groups_created, 2)

        # Trace-level aggregates match frame totals
        self.assertEqual(trace.draw_calls, 2)
        self.assertEqual(trace.draw_indexed_calls, 1)
        self.assertEqual(trace.compute_dispatches, 3)
        self.assertEqual(trace.compute_passes, 2)
        self.assertEqual(trace.textures_created, 2)
        self.assertEqual(trace.bind_groups_created, 2)

        # JSON output includes all new counters alongside existing ones
        parsed = json.loads(gpu_trace.trace_to_json(trace, stats_only=True))
        self.assertEqual(parsed["draw_calls"], 2)
        self.assertEqual(parsed["draw_indexed_calls"], 1)
        self.assertEqual(parsed["compute_dispatches"], 3)
        self.assertEqual(parsed["compute_passes"], 2)
        self.assertEqual(parsed["textures_created"], 2)
        self.assertEqual(parsed["bind_groups_created"], 2)
        self.assertEqual(parsed["frames"][0]["draw_indexed_calls"], 1)
        self.assertEqual(parsed["frames"][0]["compute_dispatches"], 3)
        self.assertEqual(parsed["frames"][0]["compute_passes"], 2)
        self.assertEqual(parsed["frames"][0]["textures_created"], 2)
        self.assertEqual(parsed["frames"][0]["bind_groups_created"], 2)

        # Text summary prints the new frame metrics
        text = gpu_trace.format_trace_text(trace, stats_only=True)
        self.assertIn("Dispatches: 3", text)
        self.assertIn("indexed=1", text)
        self.assertIn("dispatches=3", text)
        self.assertIn("compute_passes=2", text)
        self.assertIn("textures=2", text)
        self.assertIn("bind_groups=2", text)


def build_render_pipeline_key() -> bytes:
    """376-byte GpuRenderPipelineKey: rgba8unorm + blend, depth24plus-stencil8, 2 vertex buffers."""
    key = bytearray(376)
    struct.pack_into("<QQII", key, 0, 0xAAAA, 0xBBBB, 7, 2)
    # color target 0: rgba8unorm (22), blend add src-alpha / one-minus-src-alpha, mask 0xF
    struct.pack_into("<BBBxBBBxBBBx", key, 24, 22, 1, 0xF, 0, 4, 5, 0, 1, 0)
    # color target 1: null hole (all zero)
    # depth-stencil: depth24plus-stencil8 (47), write, less-equal (4), stencil always/keep/replace
    struct.pack_into(
        "<BBBxBBBBBBBBIIiII", key, 120, 47, 1, 4, 8, 0, 0, 2, 8, 0, 0, 2, 0xFF, 0x0F, -3,
        struct.unpack("<I", struct.pack("<f", 1.5))[0], 0,
    )
    struct.pack_into("<BBBBB3x", key, 152, 1, 2, 1, 2, 0)  # strip, uint32, cw, back
    struct.pack_into("<IIB3x", key, 160, 4, 0xF, 1)
    struct.pack_into("<II", key, 172, 2, 3)
    struct.pack_into("<IBB2x", key, 180, 20, 1, 2)  # slot 0: vertex, 2 attributes
    struct.pack_into("<IBB2x", key, 188, 16, 2, 1)  # slot 1: instance, 1 attribute
    struct.pack_into("<IBBBx", key, 244, 0, 30, 0, 0)  # float32x3 @0
    struct.pack_into("<IBBBx", key, 252, 12, 29, 1, 0)  # float32x2 @1
    struct.pack_into("<IBBBx", key, 260, 0, 31, 2, 1)  # float32x4 @2 (buffer 1)
    return bytes(key)


class TestGpuTraceV2Packets(unittest.TestCase):
    def test_v2_struct_sizes_match_header(self) -> None:
        self.assertEqual(gpu_trace.TEXTURE_VIEW_DESC_STRUCT.size, 16)
        self.assertEqual(gpu_trace.BIND_GROUP_LAYOUT_ENTRY_STRUCT.size, 24)
        self.assertEqual(gpu_trace.CREATE_BIND_GROUP_LAYOUT_STRUCT.size, 24)
        self.assertEqual(gpu_trace.CREATE_PIPELINE_LAYOUT_STRUCT.size, 24)
        self.assertEqual(gpu_trace.CREATE_RENDER_PIPELINE2_SIZE, 392)
        self.assertEqual(gpu_trace.KEY_COLOR_TARGET_STRUCT.size, 12)
        self.assertEqual(gpu_trace.KEY_DEPTH_STENCIL_STRUCT.size, 32)
        self.assertEqual(gpu_trace.KEY_PRIMITIVE_STRUCT.size, 8)
        self.assertEqual(gpu_trace.KEY_MULTISAMPLE_STRUCT.size, 12)
        self.assertEqual(gpu_trace.KEY_VERTEX_BUFFER_STRUCT.size, 8)
        self.assertEqual(gpu_trace.KEY_VERTEX_ATTRIBUTE_STRUCT.size, 8)
        self.assertEqual(gpu_trace.CREATE_COMPUTE_PIPELINE2_STRUCT.size, 32)
        self.assertEqual(gpu_trace.RENDER_PASS_COLOR_ATTACHMENT_STRUCT.size, 40)
        self.assertEqual(gpu_trace.RENDER_PASS_DEPTH_STENCIL_STRUCT.size, 40)
        self.assertEqual(gpu_trace.SET_VIEWPORT_STRUCT.size, 24)
        self.assertEqual(gpu_trace.BIND_GROUP_ENTRY2_STRUCT.size, 48)
        self.assertEqual(gpu_trace.SET_BIND_GROUP2_STRUCT.size, 16)
        self.assertEqual(gpu_trace.CREATE_SAMPLER2_STRUCT.size, 24)
        self.assertEqual(len(gpu_trace.encode_begin_render_pass([])), 8 + 368)
        self.assertEqual(len(gpu_trace.WGPU_TEXTURE_FORMAT_NAMES), 64)
        self.assertEqual(len(gpu_trace.WGPU_VERTEX_FORMAT_NAMES), 42)
        self.assertEqual(
            [op.value for op in gpu_trace.GpuOpcode if op.value >= 22], list(range(22, 39))
        )
        self.assertEqual(gpu_trace.INDIRECT_STRUCT.size, 16)
        # G66 ExecuteIndirect packets decode their GpuIndirectPayload.
        stream = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.DrawIndexedIndirect, gpu_trace.INDIRECT_STRUCT.pack(7, 0, 40)
        )
        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.packets[0].fields["buffer_id"], 7)
        self.assertEqual(trace.packets[0].fields["offset"], 40)

    def test_decode_full_v2_frame(self) -> None:
        view_2d_array = gpu_trace.encode_texture_view_desc(40, 3, 0, 1, 2, 4, 1)
        stream = b"".join(
            [
                gpu_trace.encode_create_bind_group_layout(
                    100,
                    [
                        (0, 0x3, 0, 0, 0, 0x1, 0, 256),  # uniform, dynamic, minBindingSize 256
                        (1, 0x2, 2, 0, 2),  # texture float 2d
                        (2, 0x2, 1, 2),  # comparison sampler
                        (3, 0x4, 3, 0, 2, 0, 16),  # storage texture write-only r32float
                    ],
                    content_id=0x472F25790659490F,
                ),
                gpu_trace.encode_create_bind_group_layout(
                    101, [], content_id=0x1234, flags=gpu_trace.MANIFEST_REFERENCE_FLAG
                ),
                gpu_trace.encode_create_pipeline_layout(102, [100, 0, 101], content_id=0x99),
                gpu_trace.encode_create_render_pipeline2(103, 1, 2, 102, build_render_pipeline_key()),
                gpu_trace.encode_create_compute_pipeline2(104, 3, 102, cs_shader=0xC5, root_signature_id=9),
                gpu_trace.encode_create_sampler2(105, (1, 1, 0), (1, 1, 0), compare=4, max_anisotropy=8),
                gpu_trace.encode_create_bind_group2(
                    106,
                    100,
                    [
                        (0, gpu_trace.GpuBindResourceType.Buffer, 50, 256, 64),
                        (1, gpu_trace.GpuBindResourceType.TextureView, 51, 0, 0, view_2d_array),
                        (2, gpu_trace.GpuBindResourceType.Sampler, 105),
                        (3, gpu_trace.GpuBindResourceType.TextureView, 0),
                    ],
                ),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_begin_render_pass(
                    [
                        {
                            "texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID,
                            "load_op": 2,
                            "store_op": 1,
                            "clear_value": (0.25, 0.5, 0.75, 1.0),
                        },
                        None,
                        {"texture_id": 60, "load_op": 1, "store_op": 2},
                    ],
                    {
                        "texture_id": 61,
                        "depth_load_op": 2,
                        "depth_store_op": 1,
                        "depth_clear_value": 0.0,
                        "stencil_read_only": True,
                    },
                ),
                gpu_trace.encode_set_pipeline(103),
                gpu_trace.encode_set_viewport(0.0, 0.0, 1280.0, 720.0, 0.0, 1.0),
                gpu_trace.encode_set_scissor_rect(0, 0, 1280, 720),
                gpu_trace.encode_set_blend_constant(1.0, 0.5, 0.25, 0.0),
                gpu_trace.encode_set_stencil_reference(0x80),
                gpu_trace.encode_set_bind_group2(0, 106, [512]),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                gpu_trace.encode_set_compute_pipeline(104),
                gpu_trace.encode_dispatch_compute(1),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])
        self.assertEqual(trace.unknown_opcode_count, 0)
        self.assertEqual(trace.bind_groups_created, 1)
        p = trace.packets

        bgl = p[0].fields
        self.assertEqual(bgl["id"], 100)
        self.assertEqual(bgl["content_id"], "0x472f25790659490f")
        self.assertFalse(bgl["manifest_reference"])
        e0, e1, e2, e3 = bgl["entries"]
        self.assertEqual(e0["type"], "buffer")
        self.assertEqual(e0["binding_type"], "uniform")
        self.assertTrue(e0["has_dynamic_offset"])
        self.assertEqual(e0["min_binding_size"], 256)
        self.assertEqual(e0["visibility_names"], ["VERTEX", "FRAGMENT"])
        self.assertEqual((e1["type"], e1["binding_type"], e1["view_dimension"]), ("texture", "float", "2d"))
        self.assertFalse(e1["multisampled"])
        self.assertEqual((e2["type"], e2["binding_type"]), ("sampler", "comparison"))
        self.assertEqual(
            (e3["type"], e3["binding_type"], e3["storage_format"]),
            ("storageTexture", "write-only", "r32float"),
        )
        self.assertTrue(p[1].fields["manifest_reference"])
        self.assertEqual(p[2].fields["bind_group_layout_ids"], [100, 0, 101])
        self.assertEqual(p[2].fields["content_id"], "0x0000000000000099")

        rp = p[3].fields
        self.assertEqual(
            (rp["id"], rp["vs_module_id"], rp["fs_module_id"], rp["pipeline_layout_id"]),
            (103, 1, 2, 102),
        )
        key = rp["key"]
        self.assertEqual(key["vs_shader"], "0x000000000000aaaa")
        self.assertEqual(key["root_signature_id"], 7)
        self.assertEqual(key["color_target_count"], 2)
        self.assertEqual(key["color_targets"][0]["format"], "rgba8unorm")
        self.assertEqual(
            key["color_targets"][0]["blend"]["color"],
            {"operation": "add", "src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha"},
        )
        self.assertIsNone(key["color_targets"][1])
        ds = key["depth_stencil"]
        self.assertEqual(ds["format"], "depth24plus-stencil8")
        self.assertEqual(ds["depth_compare"], "less-equal")
        self.assertEqual(ds["stencil_front"]["pass_op"], "replace")
        self.assertEqual(ds["depth_bias"], -3)
        self.assertEqual(ds["depth_bias_slope_scale"], 1.5)
        self.assertEqual(
            key["primitive"],
            {
                "topology": "triangle-strip",
                "strip_index_format": "uint32",
                "front_face": "cw",
                "cull_mode": "back",
                "unclipped_depth": False,
            },
        )
        self.assertEqual(key["multisample"]["count"], 4)
        vb0, vb1 = key["vertex_buffers"]
        self.assertEqual((vb0["array_stride"], vb0["step_mode"]), (20, "vertex"))
        self.assertEqual([a["format"] for a in vb0["attributes"]], ["float32x3", "float32x2"])
        self.assertEqual((vb1["step_mode"], vb1["attributes"][0]["shader_location"]), ("instance", 2))
        self.assertEqual(key["key_fnv1a64"], f"0x{gpu_trace.fnv1a_64(build_render_pipeline_key()):016x}")

        cp = p[4].fields
        self.assertEqual((cp["cs_module_id"], cp["pipeline_layout_id"], cp["root_signature_id"]), (3, 102, 9))
        sampler = p[5].fields
        self.assertEqual((sampler["address_u"], sampler["address_w"]), ("repeat", "clamp-to-edge"))
        self.assertEqual((sampler["compare"], sampler["max_anisotropy"]), ("less-equal", 8))
        bg = p[6].fields
        self.assertEqual(bg["layout_id"], 100)
        self.assertEqual((bg["entries"][0]["offset"], bg["entries"][0]["size"]), (256, 64))
        self.assertEqual(
            bg["entries"][1]["view"],
            {
                "format": "rgba16float",
                "dimension": "2d-array",
                "aspect": "all",
                "base_mip_level": 1,
                "mip_level_count": 2,
                "base_array_layer": 4,
                "array_layer_count": 1,
            },
        )
        self.assertTrue(bg["entries"][3]["null_binding"])

        rpass = p[8].fields
        self.assertEqual(rpass["color_count"], 3)
        c0, c1, c2 = rpass["color_attachments"]
        self.assertTrue(c0["swapchain"])
        self.assertEqual((c0["load_op"], c0["store_op"], c0["clear_value"]), ("clear", "store", [0.25, 0.5, 0.75, 1.0]))
        self.assertIsNone(c1)
        self.assertEqual((c2["texture_id"], c2["load_op"], c2["store_op"]), (60, "load", "discard"))
        dsa = rpass["depth_stencil"]
        self.assertEqual((dsa["texture_id"], dsa["depth_load_op"], dsa["depth_clear_value"]), (61, "clear", 0.0))
        self.assertEqual((dsa["stencil_load_op"], dsa["stencil_read_only"]), ("undefined", True))

        self.assertEqual(p[10].fields["width"], 1280.0)
        self.assertEqual(p[11].fields, {"x": 0, "y": 0, "width": 1280, "height": 720})
        self.assertEqual(p[12].fields["g"], 0.5)
        self.assertEqual(p[13].fields["reference"], 0x80)
        self.assertEqual(p[14].fields["dynamic_offsets"], [512])
        self.assertEqual(p[16].name, "EndRenderPass")

        frame = trace.frames[1]
        self.assertTrue(frame.complete)
        self.assertEqual(frame.draw_calls, 1)
        self.assertEqual(frame.compute_passes, 1)
        self.assertEqual(frame.begin_render_passes, 1)
        self.assertEqual(frame.end_render_passes, 1)
        self.assertEqual(frame.viewport_changes, 1)
        self.assertEqual(frame.scissor_changes, 1)
        self.assertEqual(frame.blend_constant_changes, 1)
        self.assertEqual(frame.stencil_ref_changes, 1)
        self.assertEqual(frame.color_load_ops, {"clear": 1, "load": 1})
        self.assertEqual(frame.color_store_ops, {"store": 1, "discard": 1})
        self.assertEqual(frame.depth_load_ops, {"clear": 1})
        self.assertEqual(frame.depth_store_ops, {"store": 1})

        # Trace-level ABI v2 aggregates (setup frame + present frame).
        self.assertEqual(trace.begin_render_passes, 1)
        self.assertEqual(trace.end_render_passes, 1)
        self.assertEqual(trace.bind_group_layouts_created, 2)
        self.assertEqual(trace.pipeline_layouts_created, 1)
        self.assertEqual(trace.render_pipelines2_created, 1)
        self.assertEqual(trace.compute_pipelines2_created, 1)
        self.assertEqual(trace.viewport_changes, 1)
        self.assertEqual(trace.scissor_changes, 1)
        self.assertEqual(trace.blend_constant_changes, 1)
        self.assertEqual(trace.stencil_ref_changes, 1)
        self.assertEqual(trace.malformed_v2_packets, 0)
        self.assertEqual(trace.truncated_v2_diagnostics, 0)
        self.assertEqual(
            set(p.opcode for p in trace.packets if 22 <= p.opcode <= 34),
            set(range(22, 35)),
        )

        text = gpu_trace.format_trace_text(trace)
        self.assertIn("BeginRenderPass", text)
        self.assertIn("c0=swapchain clear(0.25,0.5,0.75,1)/store", text)
        self.assertIn("c1=null", text)
        self.assertIn("ro=S", text)
        self.assertIn("targets=[rgba8unorm,null] depth=depth24plus-stencil8 samples=4", text)
        self.assertIn("@1:TextureView(#51)<rgba16float,2d-array,mip=1+2,layer=4+1>", text)
        self.assertIn("@3:TextureView(null)", text)
        self.assertIn("manifest-ref", text)
        self.assertIn("dynamic_offsets=[512]", text)
        self.assertIn("compare=less-equal", text)
        self.assertIn("RenderPasses: begin=1 end=1", text)
        self.assertIn("Layouts: bgl=2 pll=1", text)
        self.assertIn("Dynamic: vp=1 scissor=1 blend=1 stencil=1", text)
        self.assertIn("ColorLoadOps: clear=1, load=1", text)
        parsed = json.loads(gpu_trace.trace_to_json(trace, stats_only=True))
        self.assertEqual(parsed["begin_render_passes"], 1)
        self.assertEqual(parsed["render_pipelines2_created"], 1)
        self.assertEqual(parsed["viewport_changes"], 1)
        self.assertEqual(parsed["malformed_v2_packets"], 0)

    def test_v2_malformed_packets(self) -> None:
        # Truncated layout tail, short BeginRenderPass, unknown enum codes, oversize color_count.
        bad_layout = gpu_trace.encode_packet(
            gpu_trace.GpuOpcode.CreateBindGroupLayout,
            gpu_trace.CREATE_BIND_GROUP_LAYOUT_STRUCT.pack(1, 2, 0, 0, 0) + bytes(24),
        )
        short_pass = gpu_trace.encode_packet(gpu_trace.GpuOpcode.BeginRenderPass, bytes(100))
        weird_pass = bytearray(gpu_trace.encode_begin_render_pass([{"texture_id": 5, "load_op": 9, "store_op": 1}]))
        oversize = bytearray(gpu_trace.encode_begin_render_pass([]))
        struct.pack_into("<I", oversize, 8, 9)
        trace = gpu_trace.decode_stream(bad_layout + short_pass + bytes(weird_pass) + bytes(oversize))
        codes = [d.code for d in trace.diagnostics]
        self.assertIn("variable_payload_truncated", codes)
        self.assertIn("payload_too_short", codes)
        self.assertIn("unknown_enum_code", codes)
        self.assertIn("invalid_count", codes)
        self.assertFalse(trace.packets[0].valid)
        self.assertEqual(len(trace.packets[0].fields["entries"]), 1)
        self.assertFalse(trace.packets[1].valid)
        self.assertEqual(trace.packets[2].fields["color_attachments"][0]["load_op"], "Unknown(9)")
        self.assertFalse(trace.packets[3].valid)

        # Extra trailing bytes on a fixed-size v2 packet: warning (error under --strict).
        padded = gpu_trace.encode_packet(gpu_trace.GpuOpcode.SetStencilReference, struct.pack("<II", 1, 0))
        lenient = gpu_trace.decode_stream(padded)
        self.assertTrue(lenient.ok)
        self.assertEqual(lenient.packets[0].fields["trailing_bytes"], 4)
        self.assertFalse(gpu_trace.decode_stream(padded, strict=True).ok)

        self.assertGreaterEqual(trace.malformed_v2_packets, 2)
        self.assertGreaterEqual(trace.truncated_v2_diagnostics, 2)
        self.assertEqual(trace.frames[0].malformed_v2_packets, trace.malformed_v2_packets)

        # Truncated mid-stream v2 header/payload (BeginRenderPass claims 368B, only 8 follow).
        trunc_hdr = struct.pack("<HHI", gpu_trace.GpuOpcode.BeginRenderPass, 0, 368) + b"\x00" * 8
        trunc_trace = gpu_trace.decode_stream(trunc_hdr)
        self.assertFalse(trunc_trace.ok)
        self.assertEqual(trunc_trace.packet_count, 0)
        self.assertEqual(trunc_trace.diagnostics[0].code, "truncated_payload")
        self.assertEqual(trunc_trace.truncated_v2_diagnostics, 1)

    def test_state_bridge_fixture_disassembles_strictly(self) -> None:
        # Packet stream recorded by runtime/gpu/state_bridge (tests/test_state_bridge.cpp).
        fixture = REPO_ROOT / "tests" / "fixtures" / "state_bridge_frame.hex"
        if not fixture.is_file():
            self.skipTest("state_bridge_frame.hex fixture not present")
        hex_text = fixture.read_text(encoding="utf-8")
        trace = gpu_trace.decode_stream(bytes.fromhex("".join(hex_text.split())), strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])
        self.assertEqual(trace.packet_count, 27)
        self.assertEqual(trace.draw_calls, 2)
        self.assertEqual(trace.opcode_histogram["BeginRenderPass"], 2)
        self.assertEqual(trace.opcode_histogram["EndRenderPass"], 2)
        self.assertEqual(trace.begin_render_passes, 2)
        self.assertEqual(trace.end_render_passes, 2)
        self.assertEqual(trace.render_pipelines2_created, 2)
        self.assertGreaterEqual(trace.viewport_changes, 1)
        self.assertGreaterEqual(trace.scissor_changes, 1)
        self.assertEqual(trace.malformed_v2_packets, 0)
        self.assertEqual(trace.truncated_v2_diagnostics, 0)
        first_pass = trace.packets[0].fields
        self.assertTrue(first_pass["color_attachments"][0]["swapchain"])
        self.assertEqual(first_pass["depth_stencil"]["view"]["format"], "depth32float")
        pipelines = [p for p in trace.packets if p.name == "CreateRenderPipeline2"]
        self.assertEqual([p.fields["key"]["depth_stencil"] is None for p in pipelines], [False, True])
        self.assertEqual(pipelines[0].fields["key"]["color_targets"][0]["format"], "bgra8unorm")
        set_groups = [p.fields["dynamic_offsets"] for p in trace.packets if p.name == "SetBindGroup2"]
        self.assertEqual(set_groups, [[0], [256]])
        stats = json.loads(gpu_trace.trace_to_json(trace, stats_only=True))
        self.assertEqual(stats["begin_render_passes"], 2)
        self.assertIn("clear", stats["color_load_ops"] or stats["depth_load_ops"])

    def test_every_v2_opcode_22_through_34_encodes_and_decodes(self) -> None:
        """Synthetic coverage for each ABI v2 opcode payload size and typed fields."""
        key = build_render_pipeline_key()
        packets = {
            22: gpu_trace.encode_create_bind_group_layout(1, [(0, 0x1, 0, 0)], content_id=7),
            23: gpu_trace.encode_create_pipeline_layout(2, [1, 0], content_id=8),
            24: gpu_trace.encode_create_render_pipeline2(3, 10, 11, 2, key),
            25: gpu_trace.encode_create_compute_pipeline2(4, 12, 2, cs_shader=9),
            26: gpu_trace.encode_begin_render_pass(
                [{"texture_id": 1, "load_op": 1, "store_op": 1}],
                {"texture_id": 2, "depth_load_op": 2, "depth_store_op": 1},
            ),
            27: gpu_trace.encode_end_render_pass(),
            28: gpu_trace.encode_set_viewport(0, 0, 64, 64),
            29: gpu_trace.encode_set_scissor_rect(1, 2, 3, 4),
            30: gpu_trace.encode_set_blend_constant(0.1, 0.2, 0.3, 0.4),
            31: gpu_trace.encode_set_stencil_reference(7),
            32: gpu_trace.encode_create_bind_group2(
                5, 1, [(0, gpu_trace.GpuBindResourceType.Buffer, 9, 0, 16)]
            ),
            33: gpu_trace.encode_set_bind_group2(0, 5, [32]),
            34: gpu_trace.encode_create_sampler2(6),
        }
        self.assertEqual(sorted(packets), list(range(22, 35)))
        stream = b"".join(packets[op] for op in range(22, 35))
        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])
        self.assertEqual([p.opcode for p in trace.packets], list(range(22, 35)))
        self.assertEqual(trace.begin_render_passes, 1)
        self.assertEqual(trace.end_render_passes, 1)
        self.assertEqual(trace.bind_group_layouts_created, 1)
        self.assertEqual(trace.pipeline_layouts_created, 1)
        self.assertEqual(trace.render_pipelines2_created, 1)
        self.assertEqual(trace.compute_pipelines2_created, 1)
        self.assertEqual(trace.viewport_changes, 1)
        self.assertEqual(trace.scissor_changes, 1)
        self.assertEqual(trace.blend_constant_changes, 1)
        self.assertEqual(trace.stencil_ref_changes, 1)
        self.assertEqual(trace.packets[11].fields["dynamic_offsets"], [32])
        self.assertEqual(trace.color_load_ops, {"load": 1})
        self.assertEqual(trace.depth_load_ops, {"clear": 1})

    def test_encoder_rejects_wrong_key_size(self) -> None:
        with self.assertRaises(ValueError):
            gpu_trace.encode_create_render_pipeline2(1, 2, 3, 0, bytes(375))

    def test_create_texture2_decodes(self) -> None:
        # GpuCreateTexture2Payload (40 bytes): a 1280x720 rgba16float render target with one view
        # format, then a truncated one.
        self.assertEqual(gpu_trace.CREATE_TEXTURE2_STRUCT.size, 40)
        payload = gpu_trace.CREATE_TEXTURE2_STRUCT.pack(
            7, 1280, 720, 1, 1, 1, 0x1F, 2, 40, 1, 0, bytes([40, 0, 0, 0, 0, 0, 0, 0])
        )
        stream = gpu_trace.encode_packet(gpu_trace.GpuOpcode.CreateTexture2, payload)
        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertEqual(len(trace.packets), 1)
        fields = trace.packets[0].fields
        self.assertTrue(trace.packets[0].valid)
        self.assertEqual((fields["id"], fields["width"], fields["height"]), (7, 1280, 720))
        self.assertEqual(fields["format"], "rgba16float")
        self.assertEqual(fields["view_formats"], ["rgba16float"])
        self.assertEqual(fields["sample_count"], 1)
        self.assertEqual(len(trace.diagnostics), 0)
        short = gpu_trace.encode_packet(gpu_trace.GpuOpcode.CreateTexture2, payload[:20])
        self.assertFalse(gpu_trace.decode_stream(short).packets[0].valid)


def _make_pipeline_key(
    *,
    vs_shader: int = 0,
    ps_shader: int = 0,
    color_formats: list[int | None] = (27,),  # bgra8unorm
    depth_format: int = 0,
) -> bytes:
    """Build a minimal valid 376-byte GpuRenderPipelineKey for audit tests."""
    key = bytearray(gpu_trace.RENDER_PIPELINE_KEY_SIZE)
    struct.pack_into("<QQII", key, 0, vs_shader, ps_shader, 0, len(color_formats))
    for i, fmt in enumerate(color_formats):
        if fmt is not None and fmt != 0:
            struct.pack_into("<BBBxBBBxBBBx", key, 24 + 12 * i, fmt, 0, 0xF, 0, 1, 0, 0, 1, 0)
    if depth_format:
        struct.pack_into("<BBBx", key, 120, depth_format, 1, 4)  # less-equal
    struct.pack_into("<IIB3x", key, 160, 1, 0xFFFFFFFF, 0)  # multisample count=1
    return bytes(key)


class TestGpuTraceAudits(unittest.TestCase):
    def test_shader_summary_ranking_active_vs_unused(self) -> None:
        unused_wgsl = "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }"
        unused_hash = f"0x{gpu_trace.fnv1a_64(unused_wgsl.encode('utf-8')):016x}"
        v1_vs_hash = f"0x{gpu_trace.fnv1a_64(SAMPLE_VS_WGSL.encode('utf-8')):016x}"

        stream = b"".join(
            [
                # Shader modules: v2 (key hashes), v1 inline, v1 hash-ref, and unused modules.
                gpu_trace.encode_create_shader_module(1, gpu_trace.GpuShaderStage.Vertex, SAMPLE_VS_WGSL),
                gpu_trace.encode_create_shader_module(2, gpu_trace.GpuShaderStage.Fragment, SAMPLE_FS_WGSL),
                gpu_trace.encode_create_shader_module(3, gpu_trace.GpuShaderStage.Compute, ""),
                gpu_trace.encode_create_shader_module(4, gpu_trace.GpuShaderStage.Vertex, SAMPLE_VS_WGSL),
                gpu_trace.encode_create_shader_module(
                    5, gpu_trace.GpuShaderStage.Fragment, "wgpk:0x00000000cafe0005"
                ),
                gpu_trace.encode_create_shader_module(
                    6, gpu_trace.GpuShaderStage.Compute, "0x00000000dead0006"
                ),
                gpu_trace.encode_create_shader_module(
                    7, gpu_trace.GpuShaderStage.Fragment, unused_wgsl
                ),
                # Two v2 render pipelines sharing vs_shader 0x1111, different ps_shader.
                gpu_trace.encode_create_render_pipeline2(
                    100,
                    1,
                    2,
                    0,
                    _make_pipeline_key(vs_shader=0x1111, ps_shader=0x2222, color_formats=[27]),
                ),
                gpu_trace.encode_create_render_pipeline2(
                    101,
                    1,
                    2,
                    0,
                    _make_pipeline_key(vs_shader=0x1111, ps_shader=0x2223, color_formats=[27]),
                ),
                # v2 compute pipeline dispatched 5 times (should rank #1).
                gpu_trace.encode_create_compute_pipeline2(102, 3, 0, cs_shader=0x3333),
                # v1 render pipeline drawn once.
                gpu_trace.encode_create_render_pipeline(
                    103, 4, 5, 12, gpu_trace.GpuPrimitiveTopology.TriangleList
                ),
                # v1 compute pipeline created but never dispatched.
                gpu_trace.encode_create_compute_pipeline(104, 6),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_begin_render_pass(
                    [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 2, "store_op": 1}]
                ),
                gpu_trace.encode_set_pipeline(100),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_draw_indexed(6),
                gpu_trace.encode_set_pipeline(101),
                gpu_trace.encode_draw(9),
                gpu_trace.encode_set_pipeline(103),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                gpu_trace.encode_set_compute_pipeline(102),
                gpu_trace.encode_dispatch_compute(1, 1, 1),
                gpu_trace.encode_dispatch_compute(2, 1, 1),
                gpu_trace.encode_dispatch_compute(3, 1, 1),
                gpu_trace.encode_dispatch_compute(4, 1, 1),
                gpu_trace.encode_dispatch_compute(5, 1, 1),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])

        summary = gpu_trace.summarize_shaders(trace)
        self.assertEqual(summary["total_shaders"], 8)
        self.assertEqual(summary["active_shaders"], 6)
        self.assertEqual(summary["unused_shaders"], 2)
        self.assertEqual(summary["total_draws"], 4)
        self.assertEqual(summary["total_dispatches"], 5)
        self.assertEqual(summary["unbound_draws"], 0)
        self.assertEqual(summary["unbound_dispatches"], 0)

        rows = summary["shaders"]
        # #1: compute shader 0x3333 with 5 dispatches
        self.assertEqual(rows[0]["shader_hash"], "0x0000000000003333")
        self.assertEqual(rows[0]["stage"], "Compute")
        self.assertEqual(rows[0]["pipeline_count"], 1)
        self.assertEqual(rows[0]["draw_count"], 0)
        self.assertEqual(rows[0]["dispatch_count"], 5)
        self.assertEqual(rows[0]["total_executions"], 5)
        self.assertTrue(rows[0]["active"])

        # #2: shared vertex shader 0x1111 across 2 pipelines with 3 total draws (2 + 1)
        self.assertEqual(rows[1]["shader_hash"], "0x0000000000001111")
        self.assertEqual(rows[1]["stage"], "Vertex")
        self.assertEqual(rows[1]["pipeline_count"], 2)
        self.assertEqual(rows[1]["pipeline_ids"], [100, 101])
        self.assertEqual(rows[1]["draw_count"], 3)
        self.assertEqual(rows[1]["dispatch_count"], 0)

        # #3: fragment shader 0x2222 with 2 draws
        self.assertEqual(rows[2]["shader_hash"], "0x0000000000002222")
        self.assertEqual(rows[2]["stage"], "Fragment")
        self.assertEqual(rows[2]["draw_count"], 2)

        # Next three have 1 draw each: 0x2223, 0xcafe0005, and v1 inline VS hash
        one_draw_hashes = {r["shader_hash"] for r in rows[3:6]}
        self.assertEqual(
            one_draw_hashes,
            {"0x0000000000002223", "0x00000000cafe0005", v1_vs_hash},
        )
        self.assertTrue(all(r["draw_count"] == 1 and r["active"] for r in rows[3:6]))

        # Last two are unused: 0xdead0006 (attached to 1 pipeline, 0 dispatches) and unused_hash (0 pipelines)
        self.assertEqual(rows[6]["shader_hash"], "0x00000000dead0006")
        self.assertEqual(rows[6]["pipeline_count"], 1)
        self.assertEqual(rows[6]["total_executions"], 0)
        self.assertFalse(rows[6]["active"])

        self.assertEqual(rows[7]["shader_hash"], unused_hash)
        self.assertEqual(rows[7]["pipeline_count"], 0)
        self.assertEqual(rows[7]["total_executions"], 0)
        self.assertFalse(rows[7]["active"])

        # active_shader_hashes only includes executed shaders, ordered by executions descending.
        self.assertEqual(len(summary["active_shader_hashes"]), 6)
        self.assertEqual(summary["active_shader_hashes"][0], "0x0000000000003333")
        self.assertEqual(summary["active_shader_hashes"][1], "0x0000000000001111")
        self.assertNotIn("0x00000000dead0006", summary["active_shader_hashes"])
        self.assertNotIn(unused_hash, summary["active_shader_hashes"])

        # Verify CLI --shader-summary in text and JSON, plus backwards compatibility when omitted.
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "shaders.pg6trace"
            trace_path.write_bytes(stream)

            # Default mode omits shader_summary completely.
            buf_default = io.StringIO()
            with redirect_stdout(buf_default):
                rc = gpu_trace.main([str(trace_path), "--json", "--stats"])
            self.assertEqual(rc, 0)
            default_json = json.loads(buf_default.getvalue())
            self.assertNotIn("shader_summary", default_json)
            self.assertNotIn("binding_audit", default_json)

            # --shader-summary --json populates "shader_summary".
            buf_json = io.StringIO()
            with redirect_stdout(buf_json):
                rc = gpu_trace.main([str(trace_path), "--json", "--stats", "--shader-summary"])
            self.assertEqual(rc, 0)
            with_summary = json.loads(buf_json.getvalue())
            self.assertIn("shader_summary", with_summary)
            self.assertNotIn("binding_audit", with_summary)
            self.assertEqual(with_summary["shader_summary"]["active_shaders"], 6)
            self.assertEqual(
                with_summary["shader_summary"]["shaders"][0]["shader_hash"],
                "0x0000000000003333",
            )

            # --shader-summary text table output.
            buf_text = io.StringIO()
            with redirect_stdout(buf_text):
                rc = gpu_trace.main([str(trace_path), "--stats", "--shader-summary"])
            self.assertEqual(rc, 0)
            text_out = buf_text.getvalue()
            self.assertIn("Shader Summary:", text_out)
            self.assertIn("Total: 8 | Active: 6 | Unused: 2", text_out)
            self.assertIn("0x0000000000003333", text_out)
            self.assertIn("0x0000000000001111", text_out)
            self.assertIn("0x00000000dead0006", text_out)

    def test_binding_audit_bind_group_layout_mismatches(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_create_bind_group_layout(
                    100,
                    [
                        (0, 0x3, 0, 0),          # buffer (uniform)
                        (1, 0x2, 2, 0, 2),       # texture (float, 2d)
                        (2, 0x2, 1, 0),          # sampler (filtering)
                        (3, 0x4, 3, 0, 2, 0, 16),  # storageTexture (write-only, r32float)
                    ],
                ),
                # Manifest-reference layout should be skipped without false positives.
                gpu_trace.encode_create_bind_group_layout(
                    101,
                    [],
                    content_id=0xABCDEF,
                    flags=gpu_trace.MANIFEST_REFERENCE_FLAG,
                ),
                # Bind group 200: every resource kind mismatched + 1 undeclared binding (@7).
                gpu_trace.encode_create_bind_group2(
                    200,
                    100,
                    [
                        (0, gpu_trace.GpuBindResourceType.TextureView, 50),  # expected Buffer
                        (1, gpu_trace.GpuBindResourceType.Sampler, 51),      # expected TextureView
                        (2, gpu_trace.GpuBindResourceType.Buffer, 52),       # expected Sampler
                        (3, gpu_trace.GpuBindResourceType.Buffer, 53),       # expected TextureView (storageTexture)
                        (7, gpu_trace.GpuBindResourceType.Buffer, 54),       # undeclared in layout 100
                    ],
                ),
                # Bind group 201: missing bindings 2 & 3 + duplicate binding 0.
                gpu_trace.encode_create_bind_group2(
                    201,
                    100,
                    [
                        (0, gpu_trace.GpuBindResourceType.Buffer, 52),
                        (0, gpu_trace.GpuBindResourceType.Buffer, 52),
                        (1, gpu_trace.GpuBindResourceType.TextureView, 50),
                    ],
                ),
                # Bind group 202: references unknown layout 999.
                gpu_trace.encode_create_bind_group2(
                    202,
                    999,
                    [(0, gpu_trace.GpuBindResourceType.Buffer, 52)],
                ),
                # Bind group 203: references manifest-ref layout 101 -> skipped cleanly.
                gpu_trace.encode_create_bind_group2(
                    203,
                    101,
                    [(0, gpu_trace.GpuBindResourceType.Buffer, 52)],
                ),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)

        audit = gpu_trace.audit_bindings(trace)
        self.assertFalse(audit["ok"])
        self.assertEqual(audit["bind_group_layouts_seen"], 2)
        self.assertEqual(audit["manifest_reference_layouts"], 1)
        self.assertEqual(audit["bind_groups_audited"], 4)
        self.assertEqual(audit["bind_groups_skipped_manifest_layout"], 1)
        self.assertEqual(audit["render_pass_mismatch_count"], 0)
        self.assertEqual(
            audit["bind_group_mismatch_counts"],
            {
                "duplicate_binding": 1,
                "missing_binding": 2,
                "resource_kind_mismatch": 4,
                "undeclared_binding": 1,
                "unknown_layout": 1,
            },
        )
        self.assertEqual(audit["bind_group_mismatch_count"], 9)
        self.assertEqual(audit["total_mismatches"], 9)

        kind_mismatches = [
            m for m in audit["bind_group_mismatches"] if m["kind"] == "resource_kind_mismatch"
        ]
        self.assertEqual(
            [(m["binding"], m["expected_layout_type"], m["actual_resource_type"]) for m in kind_mismatches],
            [
                (0, "buffer", "TextureView"),
                (1, "texture", "Sampler"),
                (2, "sampler", "Buffer"),
                (3, "storageTexture", "Buffer"),
            ],
        )

        text = gpu_trace.format_trace_text(trace, stats_only=True, binding_audit=True)
        self.assertIn("Binding Audit:", text)
        self.assertIn("ISSUES FOUND (9 mismatch(es))", text)
        self.assertIn("[resource_kind_mismatch]", text)
        self.assertIn("[undeclared_binding]", text)
        self.assertIn("[missing_binding]", text)
        self.assertIn("[unknown_layout]", text)

    def test_binding_audit_render_pass_format_mismatches_and_swapchain_histogram(self) -> None:
        stream = b"".join(
            [
                # v1 texture: rgba16float
                gpu_trace.encode_create_texture(
                    50, 1280, 720, texture_format=gpu_trace.GpuTextureFormat.Rgba16Float, usage_flags=0x10
                ),
                # v2 textures: rgba8unorm (22), depth32float (48), bgra8unorm (27)
                gpu_trace.encode_create_texture2(51, 1280, 720, 22, usage_flags=0x10),
                gpu_trace.encode_create_texture2(52, 1280, 720, 48, usage_flags=0x10),
                gpu_trace.encode_create_texture2(53, 1280, 720, 27, usage_flags=0x10),
                # Pipeline 100: targets [bgra8unorm (27), rgba8unorm (22)], depth depth24plus-stencil8 (47)
                gpu_trace.encode_create_render_pipeline2(
                    100, 1, 2, 0, _make_pipeline_key(color_formats=[27, 22], depth_format=47)
                ),
                # Pipeline 101: targets [rgba16float (40)] only, no depth-stencil
                gpu_trace.encode_create_render_pipeline2(
                    101, 1, 2, 0, _make_pipeline_key(color_formats=[40], depth_format=0)
                ),
                # Pipeline 102: targets [bgra8unorm (27)] for swapchain
                gpu_trace.encode_create_render_pipeline2(
                    102, 1, 2, 0, _make_pipeline_key(color_formats=[27], depth_format=0)
                ),
                # Pipeline 103: targets [rgba8unorm (22)] for swapchain
                gpu_trace.encode_create_render_pipeline2(
                    103, 1, 2, 0, _make_pipeline_key(color_formats=[22], depth_format=0)
                ),
                gpu_trace.encode_begin_frame(),
                # Pass 1: tex#50 (rgba16float), tex#51 (rgba8unorm), ds=tex#52 (depth32float)
                gpu_trace.encode_begin_render_pass(
                    [
                        {"texture_id": 50, "load_op": 2, "store_op": 1},
                        {"texture_id": 51, "load_op": 2, "store_op": 1},
                    ],
                    {"texture_id": 52, "depth_load_op": 2, "depth_store_op": 1},
                ),
                # Draw 1 (pipeline 100): slot 0 mismatch (rgba16float vs bgra8unorm) + depth mismatch (depth32float vs depth24plus-stencil8)
                gpu_trace.encode_set_pipeline(100),
                gpu_trace.encode_draw(3),
                # Draw 2 (pipeline 101): slot 1 missing in pipeline + depth presence mismatch
                gpu_trace.encode_set_pipeline(101),
                gpu_trace.encode_draw_indexed(6),
                gpu_trace.encode_end_render_pass(),
                # Pass 2: tex#53 (bgra8unorm) only, no depth
                gpu_trace.encode_begin_render_pass(
                    [{"texture_id": 53, "load_op": 2, "store_op": 1}]
                ),
                # Draw 3 (pipeline 100): slot 1 missing in render pass + depth presence mismatch
                gpu_trace.encode_set_pipeline(100),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                # Pass 3: swapchain (default view) drawn with bgra8unorm x2 and rgba8unorm x1 -> format histogram conflict
                gpu_trace.encode_begin_render_pass(
                    [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 2, "store_op": 1}]
                ),
                gpu_trace.encode_set_pipeline(102),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_draw(6),
                gpu_trace.encode_set_pipeline(103),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                # Pass 4: swapchain with explicit view format bgra8unorm (27), drawn with pipeline 103 (rgba8unorm)
                gpu_trace.encode_begin_render_pass(
                    [
                        {
                            "texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID,
                            "load_op": 1,
                            "store_op": 1,
                            "view": gpu_trace.encode_texture_view_desc(texture_format=27),
                        }
                    ]
                ),
                gpu_trace.encode_set_pipeline(103),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])

        audit = gpu_trace.audit_bindings(trace)
        self.assertFalse(audit["ok"])
        self.assertEqual(audit["bind_group_mismatch_count"], 0)
        self.assertEqual(audit["render_passes_seen"], 4)
        self.assertEqual(audit["draws_audited"], 7)
        self.assertEqual(
            audit["render_pass_mismatch_counts"],
            {
                "color_format_mismatch": 2,
                "depth_stencil_format_mismatch": 1,
                "depth_stencil_presence_mismatch": 2,
                "missing_pipeline_color_target": 1,
                "missing_render_pass_attachment": 1,
            },
        )
        self.assertEqual(audit["render_pass_mismatch_count"], 7)
        # Swapchain histogram: pipeline 102 (bgra8unorm) x2 + pipeline 103 (rgba8unorm) x2
        self.assertEqual(audit["swapchain_target_formats"], {"bgra8unorm": 2, "rgba8unorm": 2})
        self.assertTrue(audit["swapchain_format_conflict"])

        color_mismatches = [
            m for m in audit["render_pass_mismatches"] if m["kind"] == "color_format_mismatch"
        ]
        self.assertEqual(len(color_mismatches), 2)
        self.assertEqual(
            (color_mismatches[0]["render_pass_format"], color_mismatches[0]["pipeline_format"]),
            ("rgba16float", "bgra8unorm"),
        )
        self.assertTrue(color_mismatches[1]["swapchain"])
        self.assertEqual(
            (color_mismatches[1]["render_pass_format"], color_mismatches[1]["pipeline_format"]),
            ("bgra8unorm", "rgba8unorm"),
        )

        text = gpu_trace.format_trace_text(trace, stats_only=True, binding_audit=True)
        self.assertIn("Swapchain Target Formats: bgra8unorm=2, rgba8unorm=2 [CONFLICT]", text)
        self.assertIn("[color_format_mismatch]", text)
        self.assertIn("[missing_pipeline_color_target]", text)
        self.assertIn("[missing_render_pass_attachment]", text)
        self.assertIn("[depth_stencil_format_mismatch]", text)

    def test_binding_audit_clean_trace_and_cli_combined(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_create_bind_group_layout(
                    10,
                    [
                        (0, 0x3, 0, 0),            # uniform buffer
                        (1, 0x2, 2, 0, 2),         # sampled texture 2d
                        (2, 0x2, 1, 0),            # filtering sampler
                        (3, 0x4, 3, 0, 2, 0, 22),  # storageTexture rgba8unorm
                    ],
                ),
                gpu_trace.encode_create_bind_group2(
                    11,
                    10,
                    [
                        (0, gpu_trace.GpuBindResourceType.Buffer, 20, 0, 256),
                        (1, gpu_trace.GpuBindResourceType.TextureView, 21),
                        (2, gpu_trace.GpuBindResourceType.Sampler, 22),
                        (3, gpu_trace.GpuBindResourceType.TextureView, 23),
                    ],
                ),
                # Offscreen texture rgba8unorm (22) viewed as rgba8unorm-srgb (23) + depth32float
                gpu_trace.encode_create_texture2(
                    30, 640, 360, 22, usage_flags=0x10, view_formats=[23]
                ),
                gpu_trace.encode_create_texture(
                    31, 640, 360, texture_format=gpu_trace.GpuTextureFormat.Depth32Float, usage_flags=0x10
                ),
                gpu_trace.encode_create_render_pipeline2(
                    40, 1, 2, 0,
                    _make_pipeline_key(vs_shader=0xABCD, ps_shader=0xEF01, color_formats=[23], depth_format=48),
                ),
                gpu_trace.encode_create_render_pipeline2(
                    41, 1, 2, 0,
                    _make_pipeline_key(vs_shader=0xABCD, ps_shader=0xEF02, color_formats=[27], depth_format=0),
                ),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_begin_render_pass(
                    [
                        {
                            "texture_id": 30,
                            "load_op": 2,
                            "store_op": 1,
                            "view": gpu_trace.encode_texture_view_desc(texture_format=23),
                        }
                    ],
                    {"texture_id": 31, "depth_load_op": 2, "depth_store_op": 1},
                ),
                gpu_trace.encode_set_pipeline(40),
                gpu_trace.encode_set_bind_group2(0, 11),
                gpu_trace.encode_draw(6),
                gpu_trace.encode_end_render_pass(),
                gpu_trace.encode_begin_render_pass(
                    [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 2, "store_op": 1}]
                ),
                gpu_trace.encode_set_pipeline(41),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                gpu_trace.encode_present(),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "clean.pg6trace"
            trace_path.write_bytes(stream)

            # Combined CLI --shader-summary --binding-audit --json
            buf_json = io.StringIO()
            with redirect_stdout(buf_json):
                rc = gpu_trace.main(
                    [str(trace_path), "--json", "--stats", "--shader-summary", "--binding-audit"]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf_json.getvalue())
            self.assertTrue(payload["ok"])
            self.assertIn("shader_summary", payload)
            self.assertIn("binding_audit", payload)

            audit = payload["binding_audit"]
            self.assertTrue(audit["ok"])
            self.assertEqual(audit["total_mismatches"], 0)
            self.assertEqual(audit["bind_group_mismatch_count"], 0)
            self.assertEqual(audit["render_pass_mismatch_count"], 0)
            self.assertEqual(audit["draws_audited"], 2)
            self.assertEqual(audit["swapchain_target_formats"], {"bgra8unorm": 1})
            self.assertFalse(audit["swapchain_format_conflict"])

            # Combined CLI text output
            buf_text = io.StringIO()
            with redirect_stdout(buf_text):
                rc = gpu_trace.main(
                    [str(trace_path), "--stats", "--shader-summary", "--binding-audit"]
                )
            self.assertEqual(rc, 0)
            text_out = buf_text.getvalue()
            self.assertIn("Shader Summary:", text_out)
            self.assertIn("0x000000000000abcd", text_out)
            self.assertIn("Binding Audit:", text_out)
            self.assertIn("Status: CLEAN (0 mismatches)", text_out)
            self.assertIn("Swapchain Target Formats: bgra8unorm=1", text_out)
            self.assertNotIn("[CONFLICT]", text_out)

    def test_audits_on_state_bridge_fixture(self) -> None:
        fixture = REPO_ROOT / "tests" / "fixtures" / "state_bridge_frame.hex"
        if not fixture.is_file():
            self.skipTest("state_bridge_frame.hex fixture not present")
        raw = bytes.fromhex("".join(fixture.read_text(encoding="utf-8").split()))
        trace = gpu_trace.decode_stream(raw, strict=True)
        audit = gpu_trace.audit_bindings(trace)
        self.assertTrue(audit["ok"], audit)
        self.assertEqual(audit["total_mismatches"], 0)
        self.assertEqual(audit["draws_audited"], 2)
        self.assertEqual(audit["swapchain_target_formats"], {"bgra8unorm": 1})
        summary = gpu_trace.summarize_shaders(trace)
        self.assertEqual(summary["total_draws"], 2)
        self.assertGreaterEqual(summary["active_shaders"], 1)


class TestGpuTraceBinkAndStartScreenAudits(unittest.TestCase):
    def test_bink_video_audit_4k_planar_uploaded_and_sampled_by_draw(self) -> None:
        # Format code 1 in WGPU_TEXTURE_FORMAT_NAMES is "r8unorm"; code 27 is "bgra8unorm".
        y_payload = b"\x11" * 128
        cb_payload = b"\x22" * 64
        cr_payload = b"\x33" * 64
        font_payload = b"\x44" * 32

        chunks: list[bytes] = [
            # 4K Bink2 planar Y (3840x2160) + Cb/Cr (1920x1080) as r8unorm
            gpu_trace.encode_create_texture2(10, 3840, 2160, 1, usage_flags=0x0006),
            gpu_trace.encode_create_texture2(11, 1920, 1080, 1, usage_flags=0x0006),
            gpu_trace.encode_create_texture2(12, 1920, 1080, 1, usage_flags=0x0006),
            # Non-video r8unorm font atlas uploaded once
            gpu_trace.encode_create_texture2(13, 2048, 2048, 1, usage_flags=0x0006),
            # Non-r8unorm texture (must be excluded from Bink video audit)
            gpu_trace.encode_create_texture2(14, 3840, 2160, 27, usage_flags=0x0016),
            gpu_trace.encode_create_sampler2(20),
            gpu_trace.encode_create_bind_group_layout(
                30,
                [
                    (0, 0x2, 2, 0, 2),
                    (1, 0x2, 2, 0, 2),
                    (2, 0x2, 2, 0, 2),
                    (3, 0x2, 1, 0),
                ],
            ),
            gpu_trace.encode_create_pipeline_layout(31, [30]),
            gpu_trace.encode_create_render_pipeline2(
                500,
                1,
                2,
                31,
                _make_pipeline_key(vs_shader=0xB101, ps_shader=0xB102, color_formats=[27]),
            ),
            gpu_trace.encode_create_bind_group2(
                40,
                30,
                [
                    (0, gpu_trace.GpuBindResourceType.TextureView, 10),
                    (1, gpu_trace.GpuBindResourceType.TextureView, 11),
                    (2, gpu_trace.GpuBindResourceType.TextureView, 12),
                    (3, gpu_trace.GpuBindResourceType.Sampler, 20),
                ],
            ),
            # Single one-time upload for font atlas in setup preamble
            gpu_trace.encode_write_texture(13, 8, 4, font_payload),
        ]

        # 3 frames of per-frame Y/Cb/Cr uploads + YCbCr draw sampling them
        for _ in range(3):
            chunks.extend(
                [
                    gpu_trace.encode_begin_frame(),
                    gpu_trace.encode_write_texture(10, 16, 8, y_payload),
                    gpu_trace.encode_write_texture(11, 8, 8, cb_payload),
                    gpu_trace.encode_write_texture(12, 8, 8, cr_payload),
                    gpu_trace.encode_begin_render_pass(
                        [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 2, "store_op": 1}]
                    ),
                    gpu_trace.encode_set_pipeline(500),
                    gpu_trace.encode_set_bind_group2(0, 40),
                    gpu_trace.encode_draw(3),
                    gpu_trace.encode_end_render_pass(),
                    gpu_trace.encode_present(),
                ]
            )

        trace = gpu_trace.decode_stream(b"".join(chunks), strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])

        audit = gpu_trace.audit_bink_video(trace)
        self.assertEqual(audit["r8unorm_texture_count"], 4)
        self.assertEqual(audit["video_candidate_count"], 3)
        self.assertEqual(audit["uploaded_plane_count"], 4)
        self.assertEqual(audit["per_frame_uploaded_plane_count"], 3)
        self.assertEqual(audit["bound_plane_count"], 3)
        self.assertEqual(audit["sampled_by_draw_plane_count"], 3)
        self.assertEqual(audit["sampled_by_dispatch_plane_count"], 0)
        self.assertTrue(audit["any_plane_sampled_by_draw"])
        self.assertTrue(audit["any_video_plane_sampled_by_draw"])
        self.assertTrue(audit["any_r8unorm_sampled_by_draw"])
        self.assertEqual(audit["total_upload_count"], 10)
        self.assertEqual(
            audit["total_upload_bytes"],
            3 * (len(y_payload) + len(cb_payload) + len(cr_payload)) + len(font_payload),
        )
        self.assertEqual(audit["total_bind_group_refs"], 3)
        self.assertEqual(audit["total_set_bind_group_refs"], 9)
        self.assertEqual(audit["total_draws_sampling_planes"], 3)
        self.assertEqual(audit["total_dispatches_sampling_planes"], 0)

        # Planar group pairs 3840x2160 Y (#10) with 1920x1080 Cb/Cr (#11, #12).
        self.assertEqual(len(audit["planar_groups"]), 1)
        grp = audit["planar_groups"][0]
        self.assertEqual(grp["luma_dimensions"], "3840x2160")
        self.assertEqual(grp["chroma_dimensions"], "1920x1080")
        self.assertEqual(grp["luma_texture_ids"], [10])
        self.assertEqual(grp["chroma_texture_ids"], [11, 12])
        self.assertEqual(grp["total_upload_count"], 9)
        self.assertTrue(grp["any_sampled_by_draw"])

        by_id = {row["texture_id"]: row for row in audit["planes"]}
        self.assertNotIn(14, by_id)

        self.assertEqual(by_id[10]["plane_hint"], "4k_luma")
        self.assertTrue(by_id[10]["is_video_resolution"])
        self.assertTrue(by_id[10]["is_video_candidate"])
        self.assertTrue(by_id[10]["per_frame_uploads"])
        self.assertEqual(by_id[10]["upload_count"], 3)
        self.assertEqual(by_id[10]["upload_bytes"], 3 * len(y_payload))
        self.assertEqual(by_id[10]["upload_frame_count"], 3)
        self.assertEqual(by_id[10]["bind_group_count"], 1)
        self.assertEqual(by_id[10]["bind_group_ids"], [40])
        self.assertEqual(by_id[10]["set_bind_group_count"], 3)
        self.assertEqual(by_id[10]["draw_count"], 3)
        self.assertTrue(by_id[10]["sampled_by_draw"])
        self.assertEqual(by_id[10]["sampled_pipeline_ids"], [500])

        for chroma_id in (11, 12):
            self.assertEqual(by_id[chroma_id]["plane_hint"], "4k_chroma_or_1080p_luma")
            self.assertEqual(by_id[chroma_id]["upload_count"], 3)
            self.assertEqual(by_id[chroma_id]["upload_bytes"], 3 * len(cb_payload))
            self.assertEqual(by_id[chroma_id]["draw_count"], 3)
            self.assertTrue(by_id[chroma_id]["sampled_by_draw"])

        # Font atlas (2048x2048, uploaded once, never bound) is not a video candidate.
        self.assertFalse(by_id[13]["is_video_candidate"])
        self.assertFalse(by_id[13]["per_frame_uploads"])
        self.assertFalse(by_id[13]["sampled_by_draw"])

        text = gpu_trace.format_bink_video_audit_text(audit)
        self.assertIn("Bink Video Audit (r8unorm planes):", text)
        self.assertIn("Any Video Plane Sampled By Draw: YES", text)
        self.assertIn("3840x2160 (Y: #10) + 1920x1080 (Cb/Cr: #11,#12)", text)
        self.assertIn("4k_luma", text)

    def test_bink_video_audit_unsampled_planes_compute_fallback_and_partial_capture(self) -> None:
        # Scenario:
        # - 4K Y (#100) + Cb/Cr (#101, #102) r8unorm planes uploaded every frame and bound in BG #200,
        #   but BG #200 is overwritten by BG #201 (non-video font atlas #103) before Draw -> video planes NOT sampled by draw!
        # - Custom non-standard 256x144 / 128x72 r8unorm planes (#104, #105) uploaded every frame and sampled by DispatchCompute.
        # - 3840x2160 r8unorm render attachment (#106) with 0 uploads.
        # - Partial-capture r8unorm texture (#107) without CreateTexture2, inferred from CreateBindGroup2 view format + WriteTexture.
        r8_view = gpu_trace.encode_texture_view_desc(texture_format=1, dimension=2)
        chunks: list[bytes] = [
            gpu_trace.encode_create_texture2(100, 3840, 2160, 1, usage_flags=0x0006),
            gpu_trace.encode_create_texture2(101, 1920, 1080, 1, usage_flags=0x0006),
            gpu_trace.encode_create_texture2(102, 1920, 1080, 1, usage_flags=0x0006),
            gpu_trace.encode_create_texture2(103, 2048, 2048, 1, usage_flags=0x0006),
            gpu_trace.encode_create_texture2(104, 256, 144, 1, usage_flags=0x0006),
            gpu_trace.encode_create_texture2(105, 128, 72, 1, usage_flags=0x0006),
            gpu_trace.encode_create_texture2(106, 3840, 2160, 1, usage_flags=0x0014),  # RENDER_ATTACHMENT
            gpu_trace.encode_create_bind_group_layout(10, [(0, 0x2, 2, 0, 2)]),
            gpu_trace.encode_create_bind_group2(
                200,
                10,
                [
                    (0, gpu_trace.GpuBindResourceType.TextureView, 100),
                    (1, gpu_trace.GpuBindResourceType.TextureView, 101),
                    (2, gpu_trace.GpuBindResourceType.TextureView, 102),
                ],
            ),
            gpu_trace.encode_create_bind_group2(
                201,
                10,
                [(0, gpu_trace.GpuBindResourceType.TextureView, 103)],
            ),
            gpu_trace.encode_create_bind_group(
                202,
                pipeline_id=600,
                group_index=0,
                entries=[
                    gpu_trace.BindGroupEntry(0, gpu_trace.GpuBindResourceType.TextureView, 104),
                    gpu_trace.BindGroupEntry(1, gpu_trace.GpuBindResourceType.TextureView, 105),
                ],
            ),
            # Partial-capture r8unorm view reference for texture #107 (never created in capture)
            gpu_trace.encode_create_bind_group2(
                203,
                10,
                [(0, gpu_trace.GpuBindResourceType.TextureView, 107, 0, 0, r8_view)],
            ),
            gpu_trace.encode_create_render_pipeline2(
                501, 1, 2, 0, _make_pipeline_key(vs_shader=0xA1, ps_shader=0xA2, color_formats=[27])
            ),
            gpu_trace.encode_create_compute_pipeline2(600, 3, 0, cs_shader=0xC1),
        ]

        for frame_i in range(2):
            chunks.extend(
                [
                    gpu_trace.encode_begin_frame(),
                    gpu_trace.encode_write_texture(100, 16, 16, b"\x01" * 40),
                    gpu_trace.encode_write_texture(101, 8, 8, b"\x02" * 20),
                    gpu_trace.encode_write_texture(102, 8, 8, b"\x03" * 20),
                    gpu_trace.encode_write_texture(104, 16, 16, b"\x04" * 16),
                    gpu_trace.encode_write_texture(105, 8, 8, b"\x05" * 8),
                    gpu_trace.encode_write_texture(107, 640, 360, b"\x06" * 12),
                    gpu_trace.encode_begin_render_pass(
                        [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 2, "store_op": 1}]
                    ),
                    gpu_trace.encode_set_pipeline(501),
                    # Bind video planes, then overwrite slot 0 with font atlas before drawing!
                    gpu_trace.encode_set_bind_group2(0, 200),
                    gpu_trace.encode_set_bind_group2(0, 201),
                    gpu_trace.encode_draw(6),
                    gpu_trace.encode_end_render_pass(),
                    # Compute pass samples custom planar pair (#104, #105)
                    gpu_trace.encode_set_compute_pipeline(600),
                    gpu_trace.encode_set_bind_group(0, 202),
                    gpu_trace.encode_dispatch_compute(4, 4, 1),
                    gpu_trace.encode_present(),
                ]
            )
            if frame_i == 0:
                chunks.append(gpu_trace.encode_write_texture(103, 16, 16, b"\x07" * 10))

        trace = gpu_trace.decode_stream(b"".join(chunks), strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])

        audit = gpu_trace.audit_bink_video(trace)
        self.assertEqual(audit["r8unorm_texture_count"], 8)
        # Video candidates: #100, #101, #102 (4K), #104, #105 (per-frame + planar), #106 (4K res), #107 (640x360 + per-frame).
        # Font atlas #103 (2048x2048, uploaded once) is NOT a video candidate.
        self.assertEqual(audit["video_candidate_count"], 7)
        self.assertFalse(audit["any_video_plane_sampled_by_draw"])
        self.assertFalse(audit["any_plane_sampled_by_draw"])
        self.assertTrue(audit["any_r8unorm_sampled_by_draw"])
        self.assertEqual(audit["sampled_by_draw_plane_count"], 1)  # only font #103
        self.assertEqual(audit["sampled_by_dispatch_plane_count"], 2)  # #104 & #105
        self.assertEqual(audit["total_dispatches_sampling_planes"], 2)

        by_id = {row["texture_id"]: row for row in audit["planes"]}
        self.assertEqual(by_id[100]["set_bind_group_count"], 2)
        self.assertEqual(by_id[100]["draw_count"], 0)
        self.assertFalse(by_id[100]["sampled_by_draw"])
        self.assertTrue(by_id[103]["sampled_by_draw"])
        self.assertEqual(by_id[103]["draw_count"], 2)
        self.assertTrue(by_id[104]["sampled_by_dispatch"])
        self.assertEqual(by_id[104]["plane_hint"], "planar_luma")
        self.assertEqual(by_id[105]["plane_hint"], "planar_chroma")
        self.assertEqual(by_id[104]["sampled_compute_pipeline_ids"], [600])
        self.assertTrue(by_id[106]["is_render_attachment"])
        self.assertEqual(by_id[106]["upload_count"], 0)
        # Partial-capture texture #107 inferred dimensions from WriteTexture mip 0.
        self.assertEqual((by_id[107]["width"], by_id[107]["height"]), (640, 360))
        self.assertEqual(by_id[107]["plane_hint"], "720p_chroma")
        self.assertTrue(by_id[107]["per_frame_uploads"])

        text = gpu_trace.format_bink_video_audit_text(audit)
        self.assertIn("Any Video Plane Sampled By Draw: NO", text)
        self.assertIn("compute", text)

    def test_start_screen_skips_ranking_and_anomaly_categories(self) -> None:
        stream = b"".join(
            [
                # Layouts:
                # BGL 10: bindings 0 (uniform buffer) and 1 (sampled texture)
                gpu_trace.encode_create_bind_group_layout(
                    10,
                    [
                        (0, 0x3, 0, 0),
                        (1, 0x2, 2, 0, 2),
                    ],
                ),
                # BGL 11: binding 0 (sampler)
                gpu_trace.encode_create_bind_group_layout(11, [(0, 0x2, 1, 0)]),
                # PLL 20: requires group 0 = BGL 10
                gpu_trace.encode_create_pipeline_layout(20, [10]),
                # PLL 21: group 0 = empty (0), group 1 = BGL 11 (tests cross-pipeline isolation)
                gpu_trace.encode_create_pipeline_layout(21, [0, 11]),
                # Clean BG 30 for BGL 10
                gpu_trace.encode_create_bind_group2(
                    30,
                    10,
                    [
                        (0, gpu_trace.GpuBindResourceType.Buffer, 500, 0, 64),
                        (1, gpu_trace.GpuBindResourceType.TextureView, 501),
                    ],
                ),
                # Broken BG 31 for BGL 10: missing @binding(1) + null @binding(0)
                gpu_trace.encode_create_bind_group2(
                    31,
                    10,
                    [(0, gpu_trace.GpuBindResourceType.Buffer, 0, 0, 64)],
                ),
                # Clean BG 32 for BGL 11
                gpu_trace.encode_create_bind_group2(
                    32,
                    11,
                    [(0, gpu_trace.GpuBindResourceType.Sampler, 502)],
                ),
                # Offscreen rgba16float target (40) + depth32float (48)
                gpu_trace.encode_create_texture2(60, 1280, 720, 40, usage_flags=0x10),
                gpu_trace.encode_create_texture2(61, 1280, 720, 48, usage_flags=0x10),
                # --- Pipelines ---
                # Clean pair (0xAAAA / 0xBBBB): must NOT appear in ranked_pairs.
                gpu_trace.encode_create_render_pipeline2(
                    100, 1, 2, 20, _make_pipeline_key(vs_shader=0xAAAA, ps_shader=0xBBBB, color_formats=[27])
                ),
                # Clean pair with PLL 21 (group 0 empty, group 1 = BGL 11): must NOT inherit dirty BG 31 at group 0!
                gpu_trace.encode_create_render_pipeline2(
                    101, 1, 2, 21, _make_pipeline_key(vs_shader=0xAAAB, ps_shader=0xBBBC, color_formats=[27])
                ),
                # Depth-only pipeline (ps_shader=0, fs_module_id=0, no color targets): must NOT flag missing_shader_module!
                gpu_trace.encode_create_render_pipeline2(
                    102, 1, 0, 0, _make_pipeline_key(vs_shader=0xCCCC, ps_shader=0, color_formats=[], depth_format=48)
                ),
                # Rank #1 candidate: pair (0x1111 / 0x2222) across TWO pipeline IDs (200 & 201):
                #   - Pipeline 200: 2x set_pipeline_without_draw with broken BG 31 (each adds missing_binding + null_binding)
                #   - Pipeline 201: 2x set_pipeline_without_draw with NO group 0 bound (each adds unbound_bind_group_slot)
                #   Total = 4 skips + 6 binding anomalies = 10 anomalies.
                gpu_trace.encode_create_render_pipeline2(
                    200, 1, 2, 20, _make_pipeline_key(vs_shader=0x1111, ps_shader=0x2222, color_formats=[27])
                ),
                gpu_trace.encode_create_render_pipeline2(
                    201, 1, 2, 20, _make_pipeline_key(vs_shader=0x1111, ps_shader=0x2222, color_formats=[22])
                ),
                # Rank #2 candidate: pair (0x3333 / 0x4444), pipeline 202:
                #   - missing_shader_module (fs_module_id=0 with ps_shader=0x4444) -> 1 skip
                #   - zero_count_draw (Draw(0)) -> 1 skip
                #   - render_pass_format_mismatch (pipeline expects bgra8unorm [27], pass has rgba16float [40]) -> 1 binding anomaly
                #   Total = 3 anomalies.
                gpu_trace.encode_create_render_pipeline2(
                    202, 1, 0, 0, _make_pipeline_key(vs_shader=0x3333, ps_shader=0x4444, color_formats=[27])
                ),
                # Rank #3 candidate: compute pair (cs:0x5555), pipeline 203:
                #   - missing_shader_module (cs_module_id=0) -> 1 skip
                #   - set_compute_pipeline_without_dispatch -> 1 skip
                #   - zero_count_dispatch (0, 1, 1) -> 1 skip
                #   Total = 3 anomalies (ties total=3 with #2, has higher skip_event_count=3 vs 2 -> actually ranks #2!).
                gpu_trace.encode_create_compute_pipeline2(203, 0, 0, cs_shader=0x5555),
                # Rank #4 candidate: never-bound pipeline 204 (0x6666 / 0x7777):
                #   - pipeline_never_executed -> 1 skip.
                gpu_trace.encode_create_render_pipeline2(
                    204, 1, 2, 0, _make_pipeline_key(vs_shader=0x6666, ps_shader=0x7777, color_formats=[27])
                ),
                gpu_trace.encode_begin_frame(),
                # Depth-only pass: pipeline 102 draws cleanly.
                gpu_trace.encode_begin_render_pass([], {"texture_id": 61, "depth_load_op": 2, "depth_store_op": 1}),
                gpu_trace.encode_set_pipeline(102),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                # Swapchain pass:
                gpu_trace.encode_begin_render_pass(
                    [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 2, "store_op": 1}]
                ),
                # Clean pipeline 100
                gpu_trace.encode_set_pipeline(100),
                gpu_trace.encode_set_bind_group2(0, 30),
                gpu_trace.encode_draw(6),
                # Pipeline 200 skipped twice with broken BG 31 bound at group 0
                gpu_trace.encode_set_pipeline(200),
                gpu_trace.encode_set_bind_group2(0, 31),
                gpu_trace.encode_set_pipeline(200),
                # Switch to clean pipeline 101 (group 0 unused, group 1 = BG 32): must stay clean!
                gpu_trace.encode_set_pipeline(101),
                gpu_trace.encode_set_bind_group2(1, 32),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_end_render_pass(),
                # Fresh pass (active_bind_groups cleared): pipeline 201 skipped twice with no group 0 bound
                gpu_trace.encode_begin_render_pass(
                    [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 1, "store_op": 1}]
                ),
                gpu_trace.encode_set_pipeline(201),
                gpu_trace.encode_set_pipeline(201),
                gpu_trace.encode_end_render_pass(),
                # Offscreen rgba16float pass: pipeline 202 issues zero-count draw with color format mismatch
                gpu_trace.encode_begin_render_pass([{"texture_id": 60, "load_op": 2, "store_op": 1}]),
                gpu_trace.encode_set_pipeline(202),
                gpu_trace.encode_draw(0),
                gpu_trace.encode_end_render_pass(),
                # Compute pass: pipeline 203 skipped once, then zero-count dispatched once
                gpu_trace.encode_set_compute_pipeline(203),
                gpu_trace.encode_set_compute_pipeline(203),
                gpu_trace.encode_dispatch_compute(0, 1, 1),
                gpu_trace.encode_present(),
            ]
        )

        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok, [d.message for d in trace.diagnostics])

        audit = gpu_trace.audit_start_screen_skips(trace)
        self.assertFalse(audit["ok"])
        self.assertEqual(audit["total_pipelines"], 8)
        self.assertEqual(audit["total_pipeline_pairs"], 7)
        self.assertEqual(audit["affected_pipeline_pairs"], 4)
        self.assertEqual(audit["emitted_draws"], 3)  # pipelines 102, 100, 101
        self.assertEqual(audit["emitted_dispatches"], 0)
        self.assertEqual(audit["unbound_draws"], 0)
        self.assertEqual(audit["unbound_dispatches"], 0)

        ranked = audit["ranked_pairs"]
        self.assertEqual(len(ranked), 4)
        self.assertEqual([r["rank"] for r in ranked], [1, 2, 3, 4])

        # Clean pairs must not appear in ranked_pairs.
        ranked_names = {r["shader_pair"] for r in ranked}
        self.assertNotIn("0x000000000000aaaa / 0x000000000000bbbb", ranked_names)
        self.assertNotIn("0x000000000000aaab / 0x000000000000bbbc", ranked_names)
        self.assertNotIn("0x000000000000cccc / none", ranked_names)

        # #1: 0x1111 / 0x2222 (10 anomalies: 4 skips + 6 binding anomalies across pipelines 200 & 201)
        r1 = ranked[0]
        self.assertEqual(r1["kind"], "render")
        self.assertEqual(r1["shader_pair"], "0x0000000000001111 / 0x0000000000002222")
        self.assertEqual(r1["pipeline_ids"], [200, 201])
        self.assertEqual(r1["pipeline_count"], 2)
        self.assertEqual(r1["skip_event_count"], 4)
        self.assertEqual(r1["binding_anomaly_count"], 6)
        self.assertEqual(r1["total_anomalies"], 10)
        self.assertEqual(r1["top_reason"], "set_pipeline_without_draw")
        self.assertEqual(
            r1["reason_counts"],
            {
                "missing_binding": 2,
                "null_binding": 2,
                "set_pipeline_without_draw": 4,
                "unbound_bind_group_slot": 2,
            },
        )
        self.assertEqual(r1["affected_bind_group_ids"], [31])
        self.assertIn("@group(0):@binding(1)", r1["affected_bindings"])
        self.assertIn("@group(0):@binding(0)", r1["affected_bindings"])
        self.assertIn("@group(0)", r1["affected_bindings"])

        # #2: cs:0x5555 (3 anomalies, 3 skip events)
        r2 = ranked[1]
        self.assertEqual(r2["kind"], "compute")
        self.assertEqual(r2["shader_pair"], "cs:0x0000000000005555")
        self.assertEqual(r2["skip_event_count"], 3)
        self.assertEqual(r2["binding_anomaly_count"], 0)
        self.assertEqual(r2["total_anomalies"], 3)
        self.assertEqual(
            r2["reason_counts"],
            {
                "missing_shader_module": 1,
                "set_compute_pipeline_without_dispatch": 1,
                "zero_count_dispatch": 1,
            },
        )

        # #3: 0x3333 / 0x4444 (3 anomalies: 2 skips + 1 render pass format mismatch)
        r3 = ranked[2]
        self.assertEqual(r3["shader_pair"], "0x0000000000003333 / 0x0000000000004444")
        self.assertEqual(r3["skip_event_count"], 2)
        self.assertEqual(r3["binding_anomaly_count"], 1)
        self.assertEqual(r3["total_anomalies"], 3)
        self.assertEqual(
            r3["reason_counts"],
            {
                "missing_shader_module": 1,
                "render_pass_format_mismatch": 1,
                "zero_count_draw": 1,
            },
        )

        # #4: 0x6666 / 0x7777 (never bound or drawn)
        r4 = ranked[3]
        self.assertEqual(r4["shader_pair"], "0x0000000000006666 / 0x0000000000007777")
        self.assertEqual(r4["reason_counts"], {"pipeline_never_executed": 1})
        self.assertEqual(r4["total_anomalies"], 1)

        text = gpu_trace.format_start_screen_skips_text(audit)
        self.assertIn("Start-Screen Skips Audit:", text)
        self.assertIn("ISSUES FOUND (17 anomaly event(s) across 4/7 pipeline pair(s))", text)
        self.assertIn("0x0000000000001111 / 0x0000000000002222", text)
        self.assertIn("cs:0x0000000000005555", text)
        self.assertIn("set_pipeline_without_draw", text)

    def test_bink_and_start_screen_cli_json_backwards_compat_and_sanitization(self) -> None:
        raw_image_marker = b"\xde\xad\xbe\xef\xca\xfe\xba\xbe" * 4
        stream = b"".join(
            [
                gpu_trace.encode_create_texture2(10, 3840, 2160, 1, usage_flags=0x0006),
                gpu_trace.encode_create_texture2(11, 1920, 1080, 1, usage_flags=0x0006),
                gpu_trace.encode_create_bind_group_layout(20, [(0, 0x2, 2, 0, 2)]),
                gpu_trace.encode_create_pipeline_layout(21, [20]),
                gpu_trace.encode_create_render_pipeline2(
                    30, 1, 0, 21, _make_pipeline_key(vs_shader=0x9001, ps_shader=0x9002, color_formats=[27])
                ),
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_write_texture(10, 8, 4, raw_image_marker),
                gpu_trace.encode_write_texture(11, 4, 2, raw_image_marker[:16]),
                gpu_trace.encode_begin_render_pass(
                    [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 2, "store_op": 1}]
                ),
                gpu_trace.encode_set_pipeline(30),
                gpu_trace.encode_end_render_pass(),
                gpu_trace.encode_present(),
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "secret_host_dir_933"
            nested.mkdir()
            trace_path = nested / "local_capture_secret.pg6trace"
            trace_path.write_bytes(stream)

            # 1. Backwards compatibility: omitting new flags keeps existing JSON & text untouched.
            buf_default_json = io.StringIO()
            with redirect_stdout(buf_default_json):
                rc = gpu_trace.main([str(trace_path), "--json", "--stats"])
            self.assertEqual(rc, 0)
            default_payload = json.loads(buf_default_json.getvalue())
            self.assertNotIn("bink_video_audit", default_payload)
            self.assertNotIn("start_screen_skips", default_payload)

            buf_default_text = io.StringIO()
            with redirect_stdout(buf_default_text):
                rc = gpu_trace.main([str(trace_path), "--stats"])
            self.assertEqual(rc, 0)
            self.assertNotIn("Bink Video Audit", buf_default_text.getvalue())
            self.assertNotIn("Start-Screen Skips Audit", buf_default_text.getvalue())

            # 2. Combined CLI --bink-video-audit --start-screen-skips --stats --json (sanitized).
            buf_json = io.StringIO()
            with redirect_stdout(buf_json):
                rc = gpu_trace.main(
                    [
                        str(trace_path),
                        "--json",
                        "--stats",
                        "--shader-summary",
                        "--binding-audit",
                        "--bink-video-audit",
                        "--start-screen-skips",
                    ]
                )
            self.assertEqual(rc, 0)
            json_raw = buf_json.getvalue()
            payload = json.loads(json_raw)
            self.assertIn("shader_summary", payload)
            self.assertIn("binding_audit", payload)
            self.assertIn("bink_video_audit", payload)
            self.assertIn("start_screen_skips", payload)
            self.assertFalse(payload["bink_video_audit"]["any_plane_sampled_by_draw"])
            self.assertEqual(payload["bink_video_audit"]["video_candidate_count"], 2)
            self.assertFalse(payload["start_screen_skips"]["ok"])
            self.assertEqual(
                payload["start_screen_skips"]["ranked_pairs"][0]["shader_pair"],
                "0x0000000000009001 / 0x0000000000009002",
            )

            # Sanitization checks on JSON output (no host paths, no raw image bytes).
            self.assertNotIn("secret_host_dir_933", json_raw)
            self.assertNotIn("local_capture_secret", json_raw)
            self.assertNotIn("deadbeef", json_raw.lower())
            self.assertNotIn("cafebabe", json_raw.lower())

            # 3. Combined CLI text output (sanitized).
            buf_text = io.StringIO()
            with redirect_stdout(buf_text):
                rc = gpu_trace.main(
                    [str(trace_path), "--stats", "--bink-video-audit", "--start-screen-skips"]
                )
            self.assertEqual(rc, 0)
            text_out = buf_text.getvalue()
            self.assertIn("Bink Video Audit (r8unorm planes):", text_out)
            self.assertIn("Any Video Plane Sampled By Draw: NO", text_out)
            self.assertIn("Start-Screen Skips Audit:", text_out)
            self.assertIn("0x0000000000009001 / 0x0000000000009002", text_out)
            self.assertNotIn("secret_host_dir_933", text_out)
            self.assertNotIn("local_capture_secret", text_out)
            self.assertNotIn("deadbeef", text_out.lower())
            self.assertNotIn("cafebabe", text_out.lower())

        # Empty stream edge case
        empty = gpu_trace.decode_stream(b"", strict=True)
        bink_empty = gpu_trace.audit_bink_video(empty)
        skips_empty = gpu_trace.audit_start_screen_skips(empty)
        self.assertEqual(bink_empty["r8unorm_texture_count"], 0)
        self.assertFalse(bink_empty["any_plane_sampled_by_draw"])
        self.assertTrue(skips_empty["ok"])
        self.assertEqual(skips_empty["ranked_pairs"], [])
        self.assertIn("(none)", gpu_trace.format_bink_video_audit_text(bink_empty))
        self.assertIn("CLEAN (0 anomalies)", gpu_trace.format_start_screen_skips_text(skips_empty))

    def test_bink_and_start_screen_audits_on_state_bridge_fixture(self) -> None:
        fixture = REPO_ROOT / "tests" / "fixtures" / "state_bridge_frame.hex"
        if not fixture.is_file():
            self.skipTest("state_bridge_frame.hex fixture not present")
        raw = bytes.fromhex("".join(fixture.read_text(encoding="utf-8").split()))
        trace = gpu_trace.decode_stream(raw, strict=True)
        bink = gpu_trace.audit_bink_video(trace)
        self.assertEqual(bink["r8unorm_texture_count"], 0)
        self.assertFalse(bink["any_plane_sampled_by_draw"])
        skips = gpu_trace.audit_start_screen_skips(trace)
        self.assertTrue(skips["ok"], skips)
        self.assertEqual(skips["total_anomalies"], 0)
        self.assertEqual(skips["ranked_pairs"], [])


class TestGpuTrace30FpsSpikeAnalyzer(unittest.TestCase):
    EXPECTED_TOP_LEVEL_KEYS = (
        "schema_version",
        "source_kind",
        "timing_source",
        "ok",
        "budget_ms",
        "frame_budget_ms",
        "target_fps",
        "total_frames",
        "frames_within_budget",
        "frames_over_budget",
        "over_budget_frames_30fps",
        "fps30_compliance_rate",
        "present_fps_avg",
        "draws_per_present_avg",
        "avg_frame_ms",
        "min_frame_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "max_frame_ms",
        "total_over_budget_ms",
        "max_over_budget_ms",
        "top_n",
        "spike_causes",
        "cause_breakdown",
        "top_slow_frames",
        "top_slow_passes",
        "malformed_records",
        "diagnostics",
    )

    def test_smooth_30fps_trace_and_boundary_compliance(self) -> None:
        # 20 frames all within 33.333 ms budget, including exact boundary 33.333 ms.
        frames = [
            {
                "frame_index": i,
                "frame_ms": 16.0 + (i * 0.5) if i < 19 else 33.333,
                "phase": "gameplay",
                "draw_calls": 80 + i,
                "passes": [
                    {"name": "shadow_pass", "duration_ms": 4.0},
                    {"name": "gbuffer_draw", "duration_ms": 8.0 + (i * 0.2)},
                    {"name": "ui_composite", "duration_ms": 2.0},
                ],
            }
            for i in range(20)
        ]
        report = gpu_trace.analyze_30fps_frames({"frames": frames}, top_n=5)

        self.assertTrue(report["ok"])
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["source_kind"], "timing_log")
        self.assertEqual(report["timing_source"], "measured")
        self.assertEqual(report["budget_ms"], 33.333)
        self.assertEqual(report["total_frames"], 20)
        self.assertEqual(report["frames_within_budget"], 20)
        self.assertEqual(report["frames_over_budget"], 0)
        self.assertEqual(report["fps30_compliance_rate"], 1.0)
        self.assertEqual(report["min_frame_ms"], 16.0)
        self.assertEqual(report["p50_ms"], 20.5)
        self.assertEqual(report["p95_ms"], 25.0)
        self.assertEqual(report["p99_ms"], 33.333)
        self.assertEqual(report["max_frame_ms"], 33.333)
        self.assertEqual(report["total_over_budget_ms"], 0.0)
        self.assertEqual(report["max_over_budget_ms"], 0.0)
        self.assertEqual(sum(report["spike_causes"].values()), 0)
        self.assertEqual(len(report["top_slow_frames"]), 5)
        self.assertEqual(report["top_slow_frames"][0]["frame_index"], 19)
        self.assertFalse(report["top_slow_frames"][0]["over_budget"])
        self.assertEqual(report["top_slow_frames"][0]["over_budget_ms"], 0.0)
        self.assertEqual(report["top_slow_frames"][0]["cause"], "draw_burst")

        text = gpu_trace.format_30fps_analysis_text(report)
        self.assertIn("WITHIN BUDGET (20/20 frames within budget | compliance=100.00%", text)
        self.assertIn("Spike Causes: (none)", text)
        self.assertIn("gbuffer_draw", text)

    def test_categorized_spikes_pso_upload_draw_rt_sync_and_unknown(self) -> None:
        # 10 smooth frames + 6 over-budget spike frames covering every canonical category.
        frames: list[dict] = [
            {
                "frame_index": i,
                "frame_ms": 15.0 + i * 0.5,
                "phase": "gameplay",
                "draw_calls": 40,
                "passes": [{"name": "main_gbuffer_draw", "duration_ms": 12.0}],
            }
            for i in range(10)
        ]
        frames.extend(
            [
                # 10: PSO compile spike (slowest: 92.5 ms)
                {
                    "frame_index": 10,
                    "frame_ms": 92.5,
                    "phase": "boot",
                    "pso_compiles": 4,
                    "passes": [
                        {"name": "pso_compile_gbuffer", "duration_ms": 74.0},
                        {"name": "main_gbuffer_draw", "duration_ms": 12.5},
                    ],
                },
                # 11: Staging upload spike (68.0 ms)
                {
                    "frame_index": 11,
                    "frame_ms": 68.0,
                    "phase": "loading_assets",
                    "upload_bytes": 16 * 1024 * 1024,
                    "passes": [
                        {"name": "bink_staging_upload", "duration_ms": 51.0},
                        {"name": "ui_composite", "duration_ms": 6.0},
                    ],
                },
                # 12: Sync stall spike (54.2 ms)
                {
                    "frame_index": 12,
                    "frame_ms": 54.2,
                    "phase": "gameplay",
                    "sync_stall_ms": 38.5,
                    "passes": [
                        {"name": "fence_readback_wait", "duration_ms": 38.5},
                        {"name": "main_gbuffer_draw", "duration_ms": 11.0},
                    ],
                },
                # 13: RT alloc spike (48.0 ms)
                {
                    "frame_index": 13,
                    "frame_ms": 48.0,
                    "phase": "menu",
                    "rt_allocs": 5,
                    "passes": [
                        {"name": "rt_pool_alloc_4k_gbuffer", "duration_ms": 32.0},
                        {"name": "main_gbuffer_draw", "duration_ms": 10.0},
                    ],
                },
                # 14: Draw burst spike (44.0 ms)
                {
                    "frame_index": 14,
                    "frame_ms": 44.0,
                    "phase": "gameplay",
                    "draw_calls": 2400,
                    "passes": [
                        {"name": "shadow_cascade_draw", "duration_ms": 29.0},
                        {"name": "main_gbuffer_draw", "duration_ms": 13.0},
                    ],
                },
                # 15: Unknown spike (39.0 ms, opaque marker with no counters)
                {
                    "frame_index": 15,
                    "frame_ms": 39.0,
                    "passes": [
                        {"name": "mystery_op_alpha", "duration_ms": 31.0},
                    ],
                },
            ]
        )

        report = gpu_trace.analyze_30fps_frames({"frames": frames}, top_n=6)
        self.assertTrue(report["ok"])
        self.assertEqual(report["total_frames"], 16)
        self.assertEqual(report["frames_within_budget"], 10)
        self.assertEqual(report["frames_over_budget"], 6)
        self.assertAlmostEqual(report["fps30_compliance_rate"], 10 / 16, places=6)
        self.assertEqual(report["max_frame_ms"], 92.5)
        self.assertEqual(report["p99_ms"], 92.5)
        self.assertGreaterEqual(report["p95_ms"], report["p50_ms"])
        self.assertLessEqual(report["p95_ms"], report["p99_ms"])

        self.assertEqual(
            report["spike_causes"],
            {
                "pso_compile": 1,
                "staging_upload": 1,
                "draw_burst": 1,
                "rt_alloc": 1,
                "sync_stall": 1,
                "unknown": 1,
            },
        )
        self.assertEqual(sum(report["spike_causes"].values()), report["frames_over_budget"])

        top_frames = report["top_slow_frames"]
        self.assertEqual(
            [(r["frame_index"], r["cause"]) for r in top_frames],
            [
                (10, "pso_compile"),
                (11, "staging_upload"),
                (12, "sync_stall"),
                (13, "rt_alloc"),
                (14, "draw_burst"),
                (15, "unknown"),
            ],
        )
        self.assertEqual(top_frames[0]["top_pass"], "pso_compile_gbuffer")
        self.assertEqual(top_frames[0]["top_pass_ms"], 74.0)
        self.assertAlmostEqual(top_frames[0]["over_budget_ms"], 92.5 - 33.333, places=3)

        top_passes = report["top_slow_passes"]
        self.assertEqual(top_passes[0]["name"], "pso_compile_gbuffer")
        self.assertEqual(top_passes[0]["cause"], "pso_compile")
        self.assertEqual(top_passes[0]["worst_frame_index"], 10)
        self.assertEqual(top_passes[1]["name"], "bink_staging_upload")
        self.assertEqual(top_passes[1]["cause"], "staging_upload")
        self.assertEqual(top_passes[2]["name"], "fence_readback_wait")
        self.assertEqual(top_passes[2]["cause"], "sync_stall")

        text = gpu_trace.format_30fps_analysis_text(report)
        self.assertIn("OVER BUDGET (6/16 frames over budget", text)
        self.assertIn("pso_compile=1", text)
        self.assertIn("staging_upload=1", text)
        self.assertIn("Top Slowest Frames (top 6):", text)
        self.assertIn("Top Slowest Passes (top 6):", text)

    def test_custom_budget_ms_and_cli_validation(self) -> None:
        # 4 frames: 12ms, 18ms, 30ms, 45ms
        payload = {
            "frames": [
                {"frame_index": 0, "frame_ms": 12.0, "passes": [{"name": "draw_pass", "duration_ms": 10.0}]},
                {"frame_index": 1, "frame_ms": 18.0, "passes": [{"name": "staging_upload", "duration_ms": 15.0}]},
                {"frame_index": 2, "frame_ms": 30.0, "passes": [{"name": "pso_compile", "duration_ms": 25.0}]},
                {"frame_index": 3, "frame_ms": 45.0, "passes": [{"name": "fence_wait", "duration_ms": 35.0}]},
            ]
        }

        # At 60 FPS (16.667 ms): 1 within, 3 over
        r60 = gpu_trace.analyze_30fps_frames(payload, budget_ms=16.667, top_n=2)
        self.assertEqual(r60["budget_ms"], 16.667)
        self.assertAlmostEqual(r60["target_fps"], 59.999, places=2)
        self.assertEqual(r60["frames_within_budget"], 1)
        self.assertEqual(r60["frames_over_budget"], 3)
        self.assertEqual(r60["fps30_compliance_rate"], 0.25)
        self.assertEqual(len(r60["top_slow_frames"]), 2)
        self.assertEqual(len(r60["top_slow_passes"]), 2)

        # At 20 FPS (50.0 ms): 4 within, 0 over
        r20 = gpu_trace.analyze_30fps_frames(payload, budget_ms=50.0)
        self.assertEqual(r20["frames_within_budget"], 4)
        self.assertEqual(r20["frames_over_budget"], 0)
        self.assertEqual(r20["fps30_compliance_rate"], 1.0)

        # Library validation errors
        with self.assertRaises(ValueError):
            gpu_trace.analyze_30fps_frames(payload, budget_ms=0.0)
        with self.assertRaises(ValueError):
            gpu_trace.analyze_30fps_frames(payload, budget_ms=-10.0)
        with self.assertRaises(ValueError):
            gpu_trace.analyze_30fps_frames(payload, budget_ms=float("nan"))
        with self.assertRaises(ValueError):
            gpu_trace.analyze_30fps_frames(payload, top_n=0)

        # CLI custom --budget-ms (without needing --analyze-30fps explicitly) and invalid args
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "timings.json"
            log_path.write_text(json.dumps(payload), encoding="utf-8")

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = gpu_trace.main([str(log_path), "--budget-ms", "16.667", "--top-n", "2", "--json"])
            self.assertEqual(rc, 0)
            cli_out = json.loads(buf.getvalue())
            self.assertEqual(cli_out["frames_over_budget"], 3)
            self.assertEqual(len(cli_out["top_slow_frames"]), 2)

            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc_bad = gpu_trace.main([str(log_path), "--analyze-30fps", "--budget-ms", "-1"])
            self.assertEqual(rc_bad, 2)
            self.assertIn("budget_ms", err_buf.getvalue())

            err_buf2 = io.StringIO()
            with redirect_stderr(err_buf2):
                rc_bad2 = gpu_trace.main([str(log_path), "--analyze-30fps", "--top-n", "0"])
            self.assertEqual(rc_bad2, 2)
            self.assertIn("top_n", err_buf2.getvalue())

    def test_chrome_trace_events_and_ndjson_logs(self) -> None:
        # Chrome trace event format (microseconds: ts, dur) with X and B/E pairs.
        chrome_trace = {
            "traceEvents": [
                {"ph": "X", "name": "Frame", "cat": "frame", "ts": 0, "dur": 16500, "args": {"frame_index": 0}},
                {"ph": "X", "name": "gbuffer_draw", "cat": "gpu", "ts": 1000, "dur": 12000, "args": {"frame_index": 0}},
                {"ph": "B", "name": "Frame", "cat": "frame", "ts": 20000, "pid": 1, "tid": 1, "args": {"frame_index": 1}},
                {"ph": "X", "name": "CreateRenderPipeline2", "cat": "gpu", "ts": 21000, "dur": 41000, "args": {"frame_index": 1}},
                {"ph": "E", "name": "Frame", "cat": "frame", "ts": 68000, "pid": 1, "tid": 1},
            ]
        }
        r_chrome = gpu_trace.analyze_30fps_frames(chrome_trace)
        self.assertTrue(r_chrome["ok"])
        self.assertEqual(r_chrome["total_frames"], 2)
        self.assertEqual(r_chrome["frames_within_budget"], 1)
        self.assertEqual(r_chrome["frames_over_budget"], 1)
        self.assertEqual(r_chrome["max_frame_ms"], 48.0)
        self.assertEqual(r_chrome["spike_causes"]["pso_compile"], 1)
        self.assertEqual(r_chrome["top_slow_frames"][0]["cause"], "pso_compile")
        self.assertEqual(r_chrome["top_slow_passes"][0]["name"], "CreateRenderPipeline2")
        self.assertEqual(r_chrome["top_slow_passes"][0]["max_ms"], 41.0)

        # NDJSON with one corrupted line in the middle: recovers valid lines cleanly.
        ndjson = "\n".join(
            [
                json.dumps({"frame_index": 0, "frame_ms": 16.0, "passes": [{"name": "draw_main", "duration_ms": 11.0}]}),
                "{corrupted_json_line",
                json.dumps({"frame_index": 1, "frame_ms": 52.0, "passes": [{"name": "write_texture_upload", "duration_ms": 40.0}]}),
            ]
        )
        r_ndjson = gpu_trace.analyze_30fps_frames(ndjson, strict=False)
        self.assertTrue(r_ndjson["ok"])
        self.assertEqual(r_ndjson["total_frames"], 2)
        self.assertEqual(r_ndjson["frames_over_budget"], 1)
        self.assertEqual(r_ndjson["malformed_records"], 1)
        self.assertEqual(r_ndjson["spike_causes"]["staging_upload"], 1)
        self.assertEqual(r_ndjson["diagnostics"][0]["code"], "malformed_ndjson_line")

        r_ndjson_strict = gpu_trace.analyze_30fps_frames(ndjson, strict=True)
        self.assertFalse(r_ndjson_strict["ok"])

    def test_empty_and_malformed_inputs(self) -> None:
        for empty_input in (b"", "", "   \n\t ", [], {}, {"frames": []}, {"traceEvents": []}):
            rep = gpu_trace.analyze_30fps_frames(empty_input, strict=True)
            self.assertTrue(rep["ok"], f"Failed on empty input {empty_input!r}")
            self.assertEqual(rep["total_frames"], 0)
            self.assertEqual(rep["frames_within_budget"], 0)
            self.assertEqual(rep["frames_over_budget"], 0)
            self.assertEqual(rep["fps30_compliance_rate"], 0.0)
            self.assertEqual(rep["p50_ms"], 0.0)
            self.assertEqual(rep["p95_ms"], 0.0)
            self.assertEqual(rep["p99_ms"], 0.0)
            self.assertEqual(rep["max_frame_ms"], 0.0)
            self.assertEqual(rep["top_slow_frames"], [])
            self.assertEqual(rep["top_slow_passes"], [])

        # Completely malformed JSON
        bad = gpu_trace.analyze_30fps_frames("{not valid json", strict=False)
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["total_frames"], 0)
        self.assertGreaterEqual(bad["malformed_records"], 1)
        self.assertEqual(bad["diagnostics"][0]["code"], "malformed_json")

        # Invalid UTF-8 bytes
        bad_utf8 = gpu_trace.analyze_30fps_frames(b"\xff\xfe\x80", strict=False)
        self.assertFalse(bad_utf8["ok"])
        self.assertEqual(bad_utf8["diagnostics"][0]["code"], "invalid_utf8")

        # Mixed malformed entries inside frames array (negative, bool, string, NaN)
        mixed = {
            "frames": [
                {"frame_index": 0, "frame_ms": -5.0},
                {"frame_index": 1, "frame_ms": True},
                {"frame_index": 2, "frame_ms": "fast"},
                {"frame_index": 3, "frame_ms": float("nan")},
                {"frame_index": 4, "frame_ms": 20.0, "passes": ["not_an_object", {"name": "ok_draw", "duration_ms": 12.0}]},
            ]
        }
        rep_mixed = gpu_trace.analyze_30fps_frames(mixed, strict=False)
        self.assertTrue(rep_mixed["ok"])
        self.assertEqual(rep_mixed["total_frames"], 1)
        self.assertEqual(rep_mixed["malformed_records"], 5)
        self.assertEqual(rep_mixed["p50_ms"], 20.0)

        # CLI strict mode returns exit code 1 on malformed JSON
        with tempfile.TemporaryDirectory() as tmp:
            bad_path = Path(tmp) / "broken.json"
            bad_path.write_text("{broken json", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = gpu_trace.main([str(bad_path), "--analyze-30fps", "--strict", "--json"])
            self.assertEqual(rc, 1)
            out = json.loads(buf.getvalue())
            self.assertFalse(out["ok"])
            self.assertEqual(out["diagnostics"][0]["code"], "malformed_json")

    def test_binary_pg6trace_30fps_spike_analysis_and_backwards_compat(self) -> None:
        # Frame 0: smooth draw frame
        f0 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_begin_render_pass(
                    [{"texture_id": gpu_trace.SWAPCHAIN_TEXTURE_ID, "load_op": 2, "store_op": 1}]
                ),
                gpu_trace.encode_draw(6),
                gpu_trace.encode_end_render_pass(),
                gpu_trace.encode_present(),
            ]
        )
        # Frame 1: PSO compile spike (5 render pipelines2 = ~40 ms)
        f1 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                *[
                    gpu_trace.encode_create_render_pipeline2(
                        100 + i,
                        1,
                        2,
                        0,
                        _make_pipeline_key(vs_shader=0x1000 + i, ps_shader=0x2000 + i, color_formats=[27]),
                    )
                    for i in range(5)
                ],
                gpu_trace.encode_present(),
            ]
        )
        # Frame 2: Staging upload spike (16 MiB WriteTexture = ~40 ms)
        # Use 16 * 1 MiB WriteBuffer chunks to keep memory/test fast (~16 MiB total)
        one_mb = b"\xab" * (1024 * 1024)
        f2 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                *[gpu_trace.encode_write_buffer(10, i * 1024, one_mb) for i in range(16)],
                gpu_trace.encode_present(),
            ]
        )
        # Frame 3: RT alloc spike (8 RENDER_ATTACHMENT textures = ~36 ms)
        f3 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                *[gpu_trace.encode_create_texture2(200 + i, 1920, 1080, 27, usage_flags=0x0010) for i in range(8)],
                gpu_trace.encode_present(),
            ]
        )

        stream = f0 + f1 + f2 + f3
        trace = gpu_trace.decode_stream(stream, strict=True)
        self.assertTrue(trace.ok)

        analysis = gpu_trace.analyze_30fps_trace(trace)
        self.assertTrue(analysis["ok"])
        self.assertEqual(analysis["source_kind"], "binary_trace")
        self.assertEqual(analysis["timing_source"], "estimated")
        self.assertEqual(analysis["total_frames"], 4)
        self.assertEqual(analysis["frames_within_budget"], 1)
        self.assertEqual(analysis["frames_over_budget"], 3)
        self.assertEqual(analysis["fps30_compliance_rate"], 0.25)
        self.assertEqual(analysis["spike_causes"]["pso_compile"], 1)
        self.assertEqual(analysis["spike_causes"]["staging_upload"], 1)
        self.assertEqual(analysis["spike_causes"]["rt_alloc"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "capture.pg6trace"
            trace_path.write_bytes(stream)

            # 1. Backwards compatibility: omitting --analyze-30fps leaves output unchanged.
            buf_orig = io.StringIO()
            with redirect_stdout(buf_orig):
                rc0 = gpu_trace.main([str(trace_path), "--json", "--stats"])
            self.assertEqual(rc0, 0)
            orig_payload = json.loads(buf_orig.getvalue())
            self.assertNotIn("fps30_analysis", orig_payload)

            # 2. Standalone --analyze-30fps --json emits top-level 30 FPS schema directly.
            buf_standalone = io.StringIO()
            with redirect_stdout(buf_standalone):
                rc1 = gpu_trace.main([str(trace_path), "--analyze-30fps", "--json"])
            self.assertEqual(rc1, 0)
            standalone_payload = json.loads(buf_standalone.getvalue())
            self.assertEqual(tuple(standalone_payload.keys()), self.EXPECTED_TOP_LEVEL_KEYS)
            self.assertEqual(standalone_payload["source_kind"], "binary_trace")
            self.assertEqual(standalone_payload["frames_over_budget"], 3)

            # 3. Combined --stats --analyze-30fps --json embeds fps30_analysis inside DecodedTrace JSON.
            buf_combined = io.StringIO()
            with redirect_stdout(buf_combined):
                rc2 = gpu_trace.main([str(trace_path), "--stats", "--analyze-30fps", "--json"])
            self.assertEqual(rc2, 0)
            combined_payload = json.loads(buf_combined.getvalue())
            self.assertIn("fps30_analysis", combined_payload)
            self.assertEqual(combined_payload["fps30_analysis"]["frames_over_budget"], 3)

            # 4. Standalone text output prints clean 30 FPS table (not raw packet disassembly).
            buf_text = io.StringIO()
            with redirect_stdout(buf_text):
                rc3 = gpu_trace.main([str(trace_path), "--analyze-30fps"])
            self.assertEqual(rc3, 0)
            text_out = buf_text.getvalue()
            self.assertIn("30 FPS Frame-Budget Spike Analysis", text_out)
            self.assertNotIn("Packets:\n", text_out)

    def test_sanitization_and_deterministic_json_schema_stability(self) -> None:
        secret_win_path = r"C:\Users\secret_alice\gta6_private\shaders\secret_shader.wgsl"
        secret_unix_path = "/home/secret_bob/local-data/game_dump/frame0.rpf"
        secret_unc_path = r"\\fileserver\private_share\secret_capture.pg6trace"
        secret_wgsl = "@vertex fn secret_vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> { return vec4<f32>(p, 1.0); }"

        payload = {
            "host_path": secret_win_path,
            "frames": [
                {
                    "frame_index": 0,
                    "frame_ms": 55.0,
                    "phase": secret_unix_path,
                    "shader_source": secret_wgsl,
                    "host_file": secret_unc_path,
                    "passes": [
                        {"name": secret_wgsl, "duration_ms": 40.0, "wgsl": secret_wgsl},
                        {"name": secret_win_path, "duration_ms": 8.0},
                        {"name": secret_unix_path, "duration_ms": 4.0},
                        {"name": secret_unc_path, "duration_ms": 3.0},
                    ],
                },
                {
                    "frame_index": 1,
                    "frame_ms": 16.5,
                    "phase": "main_menu",
                    "passes": [{"name": "menu_ui_draw", "duration_ms": 12.0}],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            secret_dir = Path(tmp) / "secret_host_folder_t67"
            secret_dir.mkdir()
            log_path = secret_dir / "secret_capture_file_t67.json"
            log_path.write_text(json.dumps(payload), encoding="utf-8")

            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                rc1 = gpu_trace.main([str(log_path), "--analyze-30fps", "--json"])
            self.assertEqual(rc1, 0)
            json_out_1 = buf1.getvalue()

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                rc2 = gpu_trace.main([str(log_path), "--analyze-30fps", "--json"])
            self.assertEqual(rc2, 0)
            json_out_2 = buf2.getvalue()

            # Determinism: identical inputs produce byte-for-byte identical JSON.
            self.assertEqual(json_out_1, json_out_2)

            buf_txt = io.StringIO()
            with redirect_stdout(buf_txt):
                rc_txt = gpu_trace.main([str(log_path), "--analyze-30fps"])
            self.assertEqual(rc_txt, 0)
            txt_out = buf_txt.getvalue()

            # Strict sanitization: zero host paths, zero filenames, zero raw WGSL source.
            forbidden_tokens = (
                "secret_host_folder_t67",
                "secret_capture_file_t67",
                "secret_alice",
                "secret_bob",
                "gta6_private",
                "fileserver",
                "private_share",
                "secret_vs_main",
                "@vertex",
                "vec4<f32>",
                "secret_shader.wgsl",
                "frame0.rpf",
            )
            for token in forbidden_tokens:
                self.assertNotIn(token, json_out_1, f"Leaked {token!r} in JSON output")
                self.assertNotIn(token, txt_out, f"Leaked {token!r} in text output")

            parsed = json.loads(json_out_1)
            # Exact stable top-level JSON schema order and keys.
            self.assertEqual(tuple(parsed.keys()), self.EXPECTED_TOP_LEVEL_KEYS)
            self.assertEqual(tuple(parsed["spike_causes"].keys()), gpu_trace.SPIKE_CAUSES)
            self.assertEqual(tuple(parsed["cause_breakdown"].keys()), gpu_trace.SPIKE_CAUSES)
            # Shader-source pass was classified as pso_compile before redacting its label.
            self.assertEqual(parsed["top_slow_frames"][0]["cause"], "pso_compile")
            self.assertEqual(parsed["top_slow_frames"][0]["top_pass"], "shader_event_redacted")
            self.assertIsNone(parsed["top_slow_frames"][0]["phase"])
            self.assertEqual(parsed["top_slow_frames"][1]["phase"], "main_menu")


class TestGpuTrace30FpsCadenceAndSanitization(unittest.TestCase):
    def test_cadence_calculations_timing_log_and_custom_budget(self) -> None:
        # 4 frames: 16.667 ms (50 draws), 33.333 ms (boundary - within 30 FPS, 100 draws),
        # 33.334 ms (over 30 FPS, 150 draws), 50.0 ms (over 30 FPS, 200 draws).
        payload = {
            "frames": [
                {"frame_index": 0, "frame_ms": 16.667, "draw_calls": 50},
                {"frame_index": 1, "frame_ms": 33.333, "draw_calls": 100},
                {"frame_index": 2, "frame_ms": 33.334, "draw_calls": 150},
                {"frame_index": 3, "frame_ms": 50.0, "draw_calls": 200},
            ]
        }
        report = gpu_trace.analyze_30fps_frames(payload)
        self.assertEqual(report["frame_budget_ms"], 33.333)
        self.assertEqual(report["over_budget_frames_30fps"], 2)
        self.assertEqual(report["frames_over_budget"], 2)
        self.assertEqual(report["draws_per_present_avg"], 125.0)
        # 4 presents / 133.334 ms * 1000 = 29.99985 -> 30.0
        self.assertEqual(report["present_fps_avg"], round(4000.0 / 133.334, 3))

        # Custom 60 FPS budget (16.667 ms) changes frames_over_budget (3),
        # while frame_budget_ms stays 33.333 and over_budget_frames_30fps stays 2.
        report_60 = gpu_trace.analyze_30fps_frames(payload, budget_ms=16.667)
        self.assertEqual(report_60["budget_ms"], 16.667)
        self.assertEqual(report_60["frames_over_budget"], 3)
        self.assertEqual(report_60["frame_budget_ms"], 33.333)
        self.assertEqual(report_60["over_budget_frames_30fps"], 2)
        self.assertEqual(report_60["draws_per_present_avg"], 125.0)
        self.assertEqual(report_60["present_fps_avg"], report["present_fps_avg"])

        text = gpu_trace.format_30fps_analysis_text(report)
        self.assertIn("Cadence (30 FPS):", text)
        self.assertIn("present_fps_avg=30.000", text)
        self.assertIn("draws_per_present_avg=125.000", text)
        self.assertIn("over_budget_frames_30fps=2", text)
        self.assertIn("frame_budget_ms=33.333", text)

        # Determinism: repeated serialization is byte-for-byte identical.
        j1 = gpu_trace.fps30_analysis_to_json(report)
        j2 = gpu_trace.fps30_analysis_to_json(report)
        self.assertEqual(j1, j2)

    def test_cadence_calculations_binary_trace_and_parity(self) -> None:
        # Setup preamble (no Present) + 2 complete frames (frame 1 has 2 draws, frame 2 has 3 draws + PSO spike).
        preamble = gpu_trace.encode_create_buffer(1, 0x0020, 256)
        frame1 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_draw_indexed(6),
                gpu_trace.encode_present(),
            ]
        )
        frame2 = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                *[
                    gpu_trace.encode_create_render_pipeline2(
                        100 + i,
                        1,
                        2,
                        0,
                        _make_pipeline_key(vs_shader=0x10 + i, ps_shader=0x20 + i, color_formats=[27]),
                    )
                    for i in range(5)
                ],
                gpu_trace.encode_draw(3),
                gpu_trace.encode_draw(6),
                gpu_trace.encode_draw_indexed(9),
                gpu_trace.encode_present(),
            ]
        )
        trace = gpu_trace.decode_stream(preamble + frame1 + frame2, strict=True)
        self.assertTrue(trace.ok)
        self.assertEqual(trace.present_count, 2)
        self.assertEqual(trace.draw_calls, 5)

        trace_dict = trace.to_dict(stats_only=True)
        analysis = gpu_trace.analyze_30fps_trace(trace)

        self.assertEqual(trace_dict["frame_budget_ms"], 33.333)
        self.assertEqual(trace_dict["draws_per_present_avg"], 2.5)
        self.assertEqual(trace_dict["over_budget_frames_30fps"], 1)
        self.assertGreater(trace_dict["present_fps_avg"], 0.0)

        # Exact parity between DecodedTrace.to_dict() and analyze_30fps_trace().
        for key in (
            "frame_budget_ms",
            "present_fps_avg",
            "draws_per_present_avg",
            "over_budget_frames_30fps",
        ):
            self.assertEqual(trace_dict[key], analysis[key])

        # Byte-for-byte determinism across repeated trace_to_json() calls.
        self.assertEqual(
            gpu_trace.trace_to_json(trace, stats_only=True),
            gpu_trace.trace_to_json(trace, stats_only=True),
        )

    def test_zero_present_and_zero_span_zero_division_guards(self) -> None:
        # 1. Direct helper: zero presents, zero span, negative/NaN/Inf inputs.
        zero_both = gpu_trace.compute_30fps_cadence(present_count=0, draw_calls=0, frame_durations_ms=[])
        self.assertEqual(
            zero_both,
            {
                "frame_budget_ms": 33.333,
                "present_fps_avg": 0.0,
                "draws_per_present_avg": 0.0,
                "over_budget_frames_30fps": 0,
            },
        )

        zero_presents = gpu_trace.compute_30fps_cadence(
            present_count=0, draw_calls=42, frame_durations_ms=[20.0, 50.0]
        )
        self.assertEqual(zero_presents["present_fps_avg"], 0.0)
        self.assertEqual(zero_presents["draws_per_present_avg"], 0.0)
        self.assertEqual(zero_presents["over_budget_frames_30fps"], 1)

        zero_span = gpu_trace.compute_30fps_cadence(
            present_count=4, draw_calls=10, frame_durations_ms=[0.0, 0.0, 0.0, 0.0]
        )
        self.assertEqual(zero_span["present_fps_avg"], 0.0)
        self.assertEqual(zero_span["draws_per_present_avg"], 2.5)
        self.assertEqual(zero_span["over_budget_frames_30fps"], 0)

        bad_span = gpu_trace.compute_30fps_cadence(
            present_count=3,
            draw_calls=9,
            frame_durations_ms=[float("nan"), float("inf"), -5.0, 40.0],
            total_span_ms=float("nan"),
        )
        self.assertEqual(bad_span["present_fps_avg"], 0.0)
        self.assertEqual(bad_span["draws_per_present_avg"], 3.0)
        self.assertEqual(bad_span["over_budget_frames_30fps"], 1)

        with self.assertRaises(ValueError):
            gpu_trace.compute_30fps_cadence(present_count=1, draw_calls=1, frame_budget_ms=0.0)
        with self.assertRaises(ValueError):
            gpu_trace.compute_30fps_cadence(present_count=1, draw_calls=1, frame_budget_ms=float("nan"))

        # 2. Empty binary stream.
        empty_trace = gpu_trace.decode_stream(b"", strict=True)
        empty_dict = empty_trace.to_dict(stats_only=True)
        self.assertEqual(empty_dict["frame_budget_ms"], 33.333)
        self.assertEqual(empty_dict["present_fps_avg"], 0.0)
        self.assertEqual(empty_dict["draws_per_present_avg"], 0.0)
        self.assertEqual(empty_dict["over_budget_frames_30fps"], 0)

        # 3. Binary stream with draws but zero Present packets.
        no_present_stream = gpu_trace.encode_begin_frame() + gpu_trace.encode_draw(3) + gpu_trace.encode_draw(6)
        no_present_trace = gpu_trace.decode_stream(no_present_stream, strict=True)
        self.assertEqual(no_present_trace.present_count, 0)
        self.assertEqual(no_present_trace.draw_calls, 2)
        np_dict = no_present_trace.to_dict(stats_only=True)
        self.assertEqual(np_dict["present_fps_avg"], 0.0)
        self.assertEqual(np_dict["draws_per_present_avg"], 0.0)
        self.assertEqual(np_dict["over_budget_frames_30fps"], 0)

        # 4. Timing log with zero-duration frames (zero span) and timing log with presents=0.
        zero_span_log = gpu_trace.analyze_30fps_frames(
            {"frames": [{"frame_ms": 0.0, "draw_calls": 6}, {"frame_ms": 0.0, "draw_calls": 4}]}
        )
        self.assertEqual(zero_span_log["present_fps_avg"], 0.0)
        self.assertEqual(zero_span_log["draws_per_present_avg"], 5.0)
        self.assertEqual(zero_span_log["over_budget_frames_30fps"], 0)

        zero_present_log = gpu_trace.analyze_30fps_frames(
            {
                "frames": [
                    {"frame_ms": 45.0, "draw_calls": 12, "presents": 0},
                    {"frame_ms": 15.0, "draw_calls": 8, "presented": False},
                ]
            }
        )
        self.assertEqual(zero_present_log["present_fps_avg"], 0.0)
        self.assertEqual(zero_present_log["draws_per_present_avg"], 0.0)
        self.assertEqual(zero_present_log["over_budget_frames_30fps"], 1)

    def test_host_path_and_username_json_sanitization(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"USERNAME": "secret_user_t86", "USER": "secret_user_t86"}):
            raw_payload = {
                "win_path": r"C:\Users\secret_alice\Documents\gta6\capture.pg6trace",
                "win_spaces": r"loaded C:\Users\Alice Smith\My Captures\trace.pg6trace ok",
                "unix_home": "/home/secret_bob/work/timings.json",
                "macos_user": "/Users/carol_private/Desktop/frame.bin",
                "unc_share": r"\\fileserver\private_share\dave\dump.pg6trace",
                "file_uri": "file:///C:/Users/eve_secret/local-data/game.rpf",
                "rel_user": "../../Users/grace_secret/trace.bin",
                "nested": [
                    Path(r"C:\Users\secret_alice\nested\secret_file.pg6trace"),
                    {"note": "captured by secret_user_t86 on /tmp/run_99/out.json"},
                ],
                "safe_hash": "0x00000000deadbeef",
                "frame_budget_ms": 33.333,
                "ok": True,
            }
            scrubbed = gpu_trace.sanitize_json_payload(raw_payload)
            serialized = json.dumps(scrubbed)

            for leaked in (
                "C:\\",
                "C:/",
                "/home/",
                "/Users/",
                "/tmp/",
                "\\\\fileserver",
                "file://",
                "secret_alice",
                "Alice Smith",
                "My Captures",
                "secret_bob",
                "carol_private",
                "private_share",
                "eve_secret",
                "grace_secret",
                "secret_user_t86",
                "secret_file.pg6trace",
            ):
                self.assertNotIn(leaked, serialized, f"Leaked {leaked!r} in sanitized JSON")

            self.assertEqual(scrubbed["win_spaces"], "loaded <redacted-path> ok")
            self.assertEqual(scrubbed["safe_hash"], "0x00000000deadbeef")
            self.assertEqual(scrubbed["frame_budget_ms"], 33.333)
            self.assertTrue(scrubbed["ok"])

            # Verify fps30_analysis_to_json and trace_to_json both enforce sanitization.
            mutated_report = gpu_trace.analyze_30fps_frames({"frames": [{"frame_ms": 20.0, "draw_calls": 4}]})
            mutated_report["diagnostics"].append(
                {
                    "severity": "warning",
                    "code": "synthetic_note",
                    "record_index": 0,
                    "message": r"seen at C:\Users\secret_alice\run.json by secret_user_t86",
                }
            )
            fps_json = gpu_trace.fps30_analysis_to_json(mutated_report)
            self.assertNotIn("secret_alice", fps_json)
            self.assertNotIn("secret_user_t86", fps_json)
            self.assertIn("<redacted-path>", fps_json)
            self.assertIn("<redacted-user>", fps_json)

            # Verify sanitize_path_label handles host paths and home roots safely.
            self.assertEqual(
                gpu_trace.sanitize_path_label(r"C:\Users\secret_alice\captures\run.pg6trace"),
                "run.pg6trace",
            )
            self.assertEqual(gpu_trace.sanitize_path_label(r"C:\Users\secret_alice"), "<redacted>")
            self.assertEqual(gpu_trace.sanitize_path_label("/home/secret_bob/"), "<redacted>")

        # CLI missing-file error and --json output must not leak host path directories.
        with tempfile.TemporaryDirectory(prefix="playgta6_t86_host_") as tmp:
            secret_dir = Path(tmp) / "Users" / "secret_cli_user" / "captures"
            secret_dir.mkdir(parents=True)
            trace_path = secret_dir / "capture_t86.pg6trace"
            trace_path.write_bytes(
                gpu_trace.encode_begin_frame() + gpu_trace.encode_draw(3) + gpu_trace.encode_present()
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = gpu_trace.main([str(trace_path), "--json", "--stats"])
            self.assertEqual(rc, 0)
            out_json = buf.getvalue()
            for leaked in (str(tmp), "secret_cli_user", "playgta6_t86_host_", "capture_t86.pg6trace"):
                self.assertNotIn(leaked, out_json)
            parsed = json.loads(out_json)
            self.assertEqual(parsed["frame_budget_ms"], 33.333)
            self.assertEqual(parsed["draws_per_present_avg"], 1.0)
            self.assertGreater(parsed["present_fps_avg"], 0.0)
            self.assertEqual(parsed["over_budget_frames_30fps"], 0)

            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc_missing = gpu_trace.main([str(secret_dir / "missing_t86.pg6trace")])
            self.assertEqual(rc_missing, 2)
            self.assertIn("trace file not found", err_buf.getvalue())
            self.assertNotIn("secret_cli_user", err_buf.getvalue())
            self.assertNotIn("playgta6_t86_host_", err_buf.getvalue())

    def test_backwards_compatibility_cli_flags_and_outputs(self) -> None:
        stream = b"".join(
            [
                gpu_trace.encode_begin_frame(),
                gpu_trace.encode_clear_color(0.0, 0.0, 0.0, 1.0),
                gpu_trace.encode_draw(3),
                gpu_trace.encode_present(),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "compat.pg6trace"
            trace_path.write_bytes(stream)

            # Default text mode keeps original header/frame layout untouched.
            buf_txt = io.StringIO()
            with redirect_stdout(buf_txt):
                rc = gpu_trace.main([str(trace_path)])
            self.assertEqual(rc, 0)
            txt = buf_txt.getvalue()
            self.assertIn("playgta6 GPU Trace (ABI v1)", txt)
            self.assertIn("Frame 000 [complete]", txt)
            self.assertNotIn("30 FPS Frame-Budget Spike Analysis", txt)

            # --json --stats omits opt-in audit sections and packets, preserves legacy keys + adds cadence.
            buf_json = io.StringIO()
            with redirect_stdout(buf_json):
                rc = gpu_trace.main([str(trace_path), "--json", "--stats"])
            self.assertEqual(rc, 0)
            payload = json.loads(buf_json.getvalue())
            for legacy_key in (
                "abi_version",
                "ok",
                "packet_count",
                "frame_count",
                "draw_calls",
                "opcode_histogram",
                "frames",
                "diagnostics",
            ):
                self.assertIn(legacy_key, payload)
            for opt_in_key in (
                "packets",
                "shader_summary",
                "binding_audit",
                "bink_video_audit",
                "start_screen_skips",
                "fps30_analysis",
            ):
                self.assertNotIn(opt_in_key, payload)
            for cadence_key in (
                "frame_budget_ms",
                "present_fps_avg",
                "draws_per_present_avg",
                "over_budget_frames_30fps",
            ):
                self.assertIn(cadence_key, payload)


if __name__ == "__main__":
    unittest.main()
