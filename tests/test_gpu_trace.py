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
                gpu_trace.encode_packet(12, future_payload, flags=0x0042),
                gpu_trace.encode_packet(99, b""),
                gpu_trace.encode_present(),
            ]
        )

        # Non-strict mode handles unknown opcodes cleanly without errors.
        lenient = gpu_trace.decode_stream(stream, strict=False)
        self.assertTrue(lenient.ok)
        self.assertEqual(lenient.diagnostics, ())
        self.assertEqual(lenient.packet_count, 4)
        self.assertEqual(lenient.unknown_opcode_count, 2)

        u12 = lenient.packets[1]
        self.assertFalse(u12.known)
        self.assertTrue(u12.valid)
        self.assertEqual(u12.opcode, 12)
        self.assertEqual(u12.name, "Unknown(12)")
        self.assertEqual(u12.flags, 0x0042)
        self.assertEqual(u12.payload_bytes, len(future_payload))
        self.assertEqual(u12.fields["payload_bytes"], len(future_payload))
        self.assertEqual(u12.fields["crc32"], gpu_trace.crc32_hex(future_payload))
        self.assertTrue(u12.fields["truncated_preview"])

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


if __name__ == "__main__":
    unittest.main()
