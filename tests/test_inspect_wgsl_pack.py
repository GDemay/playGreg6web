#!/usr/bin/env python3
"""Unit and CLI tests for tools/inspect_wgsl_pack.py (Track 4 task S7)."""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import inspect_wgsl_pack as wgpk  # noqa: E402


MINIMAL_VERTEX_WGSL = """
struct VsOut {
    @builtin(position) pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: mat4x4<f32>;

@vertex
fn vs_main(@location(0) in_pos: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.pos = camera * vec4<f32>(in_pos, 1.0);
    return out;
}
""".strip()

MINIMAL_FRAGMENT_WGSL = """
@group(0) @binding(0) var albedo_tex: texture_2d<f32>;
@group(0) @binding(1) var albedo_sampler: sampler;

@fragment
fn fs_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(albedo_tex, albedo_sampler, uv);
}
""".strip()

MINIMAL_COMPUTE_WGSL = """
@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    dst[gid.x] = src[gid.x] * 2.0;
}
""".strip()


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        code = wgpk.main(argv)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


class TestWgpkBinaryFormat(unittest.TestCase):
    """Verify exact binary layout parity with shaders/src/pack.rs & wgsl_pack_loader.js."""

    def test_fnv1a_64_matches_reference_vectors(self) -> None:
        self.assertEqual(wgpk.fnv1a_64(b""), 0xCBF29CE484222325)
        self.assertEqual(wgpk.fnv1a_64(b"a"), 0xAF63DC4C8601EC8C)
        self.assertEqual(wgpk.fnv1a_64("foobar"), 0x85944171F73967E8)
        self.assertEqual(wgpk.format_shader_hash(0x1A2B), "0x0000000000001a2b")

    def test_build_and_parse_round_trip_matches_pack_rs_layout(self) -> None:
        h_vs = wgpk.fnv1a_64(MINIMAL_VERTEX_WGSL)
        h_fs = wgpk.fnv1a_64(MINIMAL_FRAGMENT_WGSL)
        h_cs = wgpk.fnv1a_64(MINIMAL_COMPUTE_WGSL)

        raw = wgpk.build_wgpk(
            [
                {"hash": h_vs, "stage": "vertex", "wgsl": MINIMAL_VERTEX_WGSL},
                {"hash": h_fs, "stage": wgpk.ShaderStage.FRAGMENT, "wgsl": MINIMAL_FRAGMENT_WGSL},
                (h_cs, 3, MINIMAL_COMPUTE_WGSL),
            ],
            flags=0x12345678,
        )

        # Verify raw 16-byte header byte-for-byte.
        self.assertEqual(raw[0:4], b"WGPK")
        self.assertEqual(struct.unpack_from("<III", raw, 4), (1, 3, 0x12345678))

        expected_data_start = 16 + 3 * 20
        # Verify first 20-byte entry record directly.
        e0_hash, e0_stage, e0_res, e0_off, e0_len = struct.unpack_from("<QB3sII", raw, 16)
        self.assertEqual(e0_hash, h_vs)
        self.assertEqual(e0_stage, 1)
        self.assertEqual(e0_res, b"\x00\x00\x00")
        self.assertEqual(e0_off, expected_data_start)
        self.assertEqual(e0_len, len(MINIMAL_VERTEX_WGSL.encode("utf-8")))

        bundle = wgpk.parse_wgpk(raw)
        self.assertEqual(bundle.version, 1)
        self.assertEqual(bundle.flags, 0x12345678)
        self.assertEqual(bundle.entry_count, 3)
        self.assertEqual(bundle.byte_length, len(raw))
        self.assertEqual(bundle.stage_counts(), {"vertex": 1, "fragment": 1, "compute": 1})

        self.assertEqual(bundle.entries[0].hash, h_vs)
        self.assertEqual(bundle.entries[0].stage, wgpk.ShaderStage.VERTEX)
        self.assertEqual(bundle.entries[0].wgsl, MINIMAL_VERTEX_WGSL)
        self.assertEqual(bundle.entries[1].stage, wgpk.ShaderStage.FRAGMENT)
        self.assertEqual(bundle.entries[2].stage, wgpk.ShaderStage.COMPUTE)
        self.assertEqual(bundle.entry_by_hash(h_cs), bundle.entries[2])

    def test_empty_pack_round_trip(self) -> None:
        raw = wgpk.build_wgpk([])
        self.assertEqual(len(raw), 16)
        bundle = wgpk.parse_wgpk(raw)
        self.assertEqual(bundle.entry_count, 0)
        self.assertEqual(bundle.total_wgsl_bytes, 0)

    def test_nonzero_reserved_bytes_tolerated_for_forward_compat(self) -> None:
        raw = bytearray(
            wgpk.build_wgpk([(0x1122334455667788, "compute", MINIMAL_COMPUTE_WGSL)])
        )
        raw[16 + 9 : 16 + 12] = b"\xAA\xBB\xCC"
        bundle = wgpk.parse_wgpk(bytes(raw))
        self.assertEqual(bundle.entry_count, 1)
        self.assertEqual(bundle.entries[0].wgsl, MINIMAL_COMPUTE_WGSL)

    def test_parse_errors_corrupted_header_table_bounds_and_utf8(self) -> None:
        valid = wgpk.build_wgpk(
            [
                (0x1111, "compute", "@compute @workgroup_size(1) fn main() {}"),
                (0x2222, "vertex", "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(); }"),
            ]
        )

        # Truncated header (< 16 bytes)
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(valid[:12])
        self.assertEqual(ctx.exception.code, "truncated_header")

        # Invalid magic
        bad_magic = bytearray(valid)
        bad_magic[0:4] = b"NOPE"
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(bad_magic)
        self.assertEqual(ctx.exception.code, "invalid_magic")

        # Unsupported version
        bad_version = bytearray(valid)
        struct.pack_into("<I", bad_version, 4, 99)
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(bad_version)
        self.assertEqual(ctx.exception.code, "unsupported_version")

        # Truncated entry table
        truncated_table = bytearray(valid[: 16 + 10])
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(truncated_table)
        self.assertEqual(ctx.exception.code, "truncated_table")

        # Invalid stage code
        bad_stage = bytearray(valid)
        bad_stage[16 + 8] = 9
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(bad_stage)
        self.assertEqual(ctx.exception.code, "invalid_stage")

        # Offset pointing inside header/entry table
        overlap_header = bytearray(valid)
        struct.pack_into("<I", overlap_header, 16 + 12, 4)
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(overlap_header)
        self.assertEqual(ctx.exception.code, "out_of_bounds")

        # Offset + length exceeding total pack size
        oob = bytearray(valid)
        struct.pack_into("<I", oob, 16 + 12, 99999)
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(oob)
        self.assertEqual(ctx.exception.code, "out_of_bounds")

        # Overlapping payload ranges between entry 0 and entry 1
        overlapping = bytearray(valid)
        e0_offset = struct.unpack_from("<I", overlapping, 16 + 12)[0]
        struct.pack_into("<I", overlapping, 16 + 20 + 12, e0_offset + 4)
        struct.pack_into("<I", overlapping, 16 + 20 + 16, 8)
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(overlapping)
        self.assertEqual(ctx.exception.code, "overlapping_payloads")

        # Invalid UTF-8 payload
        bad_utf8 = bytearray(valid)
        payload_start = 16 + 2 * 20
        bad_utf8[payload_start] = 0xFF
        bad_utf8[payload_start + 1] = 0xFE
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk(bad_utf8)
        self.assertEqual(ctx.exception.code, "invalid_utf8")


class TestWgslStaticAnalysis(unittest.TestCase):
    """Test WGSL stage, binding, workgroup, feature, and budget limit analysis."""

    def test_multi_stage_bindings_and_features(self) -> None:
        source = """
        // Commented-out traps should be ignored:
        // enable chromium_experimental_fake;
        /* nested /* @group(9) @binding(0) var<storage> fake: array<u32>; */ @workgroup_size(999) */
        enable f16;
        enable subgroups;

        const WG_X: u32 = 16u;

        struct Globals {
            @align(16) view_proj: mat4x4<f32>,
        };

        @group(0) @binding(0) var<uniform> globals: Globals;
        @binding(1) @group(0) var<storage, read> instances: array<vec4<f32>>;
        @group(0) @binding(2) var<storage, read_write> indirect: array<u32>;
        @group(1) @binding(0u) var diffuse: texture_2d<f32>;
        @group(1) @binding(1u) var depth_tex: texture_depth_2d;
        @group(1) @binding(2u) var linear_samp: sampler;
        @group(1) @binding(3u) var shadow_samp: sampler_comparison;
        @group(2) @binding(0) var out_img: texture_storage_2d<rgba8unorm, write>;

        @vertex
        fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
            return globals.view_proj * vec4<f32>(p, 1.0);
        }

        @fragment
        fn fs_main() -> @location(0) vec4<f32> {
            return vec4<f32>(1.0);
        }

        @workgroup_size(WG_X, 8, 1)
        @compute
        fn cs_cull() {
            var local_only: u32 = 0u;
        }
        """
        audit = wgpk.analyze_wgsl(source, 0xCAFEBABE)
        self.assertEqual(audit.stages, ("vertex", "fragment", "compute"))
        self.assertEqual(audit.entry_point_names, ("vs_main", "fs_main", "cs_cull"))
        self.assertEqual(audit.max_group_index, 2)
        self.assertEqual(audit.bindings_per_group, {0: 3, 1: 4, 2: 1})
        self.assertEqual(audit.uniform_buffer_count, 1)
        self.assertEqual(audit.storage_buffer_count, 2)
        self.assertEqual(audit.sampled_texture_count, 2)
        self.assertEqual(audit.storage_texture_count, 1)
        self.assertEqual(audit.texture_count, 3)
        self.assertEqual(audit.sampler_count, 2)
        self.assertEqual(audit.workgroup_sizes, ((16, 8, 1),))
        self.assertEqual(audit.max_workgroup_invocations, 128)
        self.assertEqual(audit.enabled_features, ("f16", "subgroups"))
        self.assertTrue(audit.has_f16)
        self.assertTrue(audit.has_subgroups)
        self.assertEqual(audit.experimental_features, ())
        self.assertEqual(audit.violations, ())
        self.assertTrue(audit.clean)

    def test_workgroup_size_defaults_and_limits(self) -> None:
        # Valid boundary: 16 * 16 * 1 = 256 invocations (exactly at WebGPU default limit)
        ok_wgsl = "@compute @workgroup_size(16, 16) fn cs_ok() {}"
        ok_audit = wgpk.analyze_wgsl(ok_wgsl, 1)
        self.assertEqual(ok_audit.workgroup_sizes, ((16, 16, 1),))
        self.assertEqual(ok_audit.max_workgroup_invocations, 256)
        self.assertEqual(ok_audit.violations, ())

        # Exceeds invocations: 16 * 16 * 2 = 512 > 256
        too_many_inv = "@compute @workgroup_size(16, 16, 2) fn cs_big() {}"
        inv_audit = wgpk.analyze_wgsl(too_many_inv, 2)
        codes = {v.code for v in inv_audit.violations}
        self.assertIn("workgroup_invocations_exceeded", codes)
        self.assertNotIn("workgroup_size_x_exceeded", codes)

        # Exceeds z dimension (> 64) even though total invocations = 128 <= 256
        too_deep_z = "@compute @workgroup_size(1, 1, 128) fn cs_z() {}"
        z_audit = wgpk.analyze_wgsl(too_deep_z, 3)
        z_codes = {v.code for v in z_audit.violations}
        self.assertEqual(z_codes, {"workgroup_size_z_exceeded"})

        # Exceeds x dimension (> 256) AND invocations (> 256)
        too_wide_x = "@compute @workgroup_size(512) fn cs_x() {}"
        x_audit = wgpk.analyze_wgsl(too_wide_x, 4)
        x_codes = {v.code for v in x_audit.violations}
        self.assertEqual(x_codes, {"workgroup_size_x_exceeded", "workgroup_invocations_exceeded"})

        # Override-driven workgroup size does not crash
        override_wgsl = "override WG: u32 = 64u; @compute @workgroup_size(WG) fn cs_ov() {}"
        ov_audit = wgpk.analyze_wgsl(override_wgsl, 5)
        self.assertEqual(ov_audit.workgroup_sizes, ())
        self.assertEqual(ov_audit.entry_points[0].workgroup_size_raw, "WG")
        self.assertEqual(ov_audit.violations, ())

    def test_budget_violations_bind_groups_storage_buffers_and_experimental(self) -> None:
        storage_decls = "\n".join(
            f"@group(0) @binding({i}) var<storage, read> buf_{i}: array<u32>;"
            for i in range(9)
        )
        source = f"""
        enable f16, chromium_experimental_push_constant;
        {storage_decls}
        @group(4) @binding(0) var<uniform> extra_group_ubo: vec4<f32>;
        @group(0) @binding(0) var<uniform> dup_binding: vec4<f32>;

        @compute @workgroup_size(32, 16, 1)
        fn cs_heavy() {{}}
        """
        audit = wgpk.analyze_wgsl(source, 0xDEAD, pack_stage="vertex")
        codes = {v.code for v in audit.violations}
        self.assertIn("bind_group_index_exceeded", codes)
        self.assertIn("duplicate_binding", codes)
        self.assertIn("storage_buffers_per_stage_exceeded", codes)
        self.assertIn("workgroup_invocations_exceeded", codes)
        self.assertIn("experimental_feature", codes)
        self.assertIn("stage_mismatch", codes)
        self.assertEqual(audit.experimental_features, ("chromium_experimental_push_constant",))
        self.assertFalse(audit.clean)

        # Relaxed custom limits should clear the raised-limit violations.
        relaxed = wgpk.WebGpuLimits(
            max_bind_groups=8,
            max_storage_buffers_per_shader_stage=16,
            max_compute_invocations_per_workgroup=1024,
            disallow_experimental_features=False,
        )
        relaxed_audit = wgpk.analyze_wgsl(source, 0xDEAD, limits=relaxed, pack_stage="compute")
        relaxed_codes = {v.code for v in relaxed_audit.violations}
        self.assertEqual(relaxed_codes, {"duplicate_binding"})


class TestInspectAndDiff(unittest.TestCase):
    """Test pack-level inspection summaries and pack-to-pack diffs."""

    def test_inspect_pack_summary_and_feature_histogram(self) -> None:
        f16_fs = "enable f16;\n" + MINIMAL_FRAGMENT_WGSL
        sub_cs = "enable f16, subgroups;\n" + MINIMAL_COMPUTE_WGSL
        pack_bytes = wgpk.build_wgpk(
            [
                (0x1001, "vertex", MINIMAL_VERTEX_WGSL),
                (0x1002, "fragment", f16_fs),
                (0x1003, "compute", sub_cs),
            ]
        )
        report = wgpk.inspect_pack(pack_bytes)
        self.assertEqual(report.entry_count, 3)
        self.assertEqual(
            report.total_wgsl_bytes,
            len(MINIMAL_VERTEX_WGSL.encode("utf-8"))
            + len(f16_fs.encode("utf-8"))
            + len(sub_cs.encode("utf-8")),
        )
        self.assertEqual(report.stage_counts, {"vertex": 1, "fragment": 1, "compute": 1})
        self.assertEqual(report.feature_histogram, {"f16": 2, "subgroups": 1})
        self.assertEqual(report.violations, ())
        self.assertTrue(report.clean)

        as_dict = report.to_dict()
        self.assertEqual(as_dict["entryCount"], 3)
        self.assertEqual(as_dict["totalWgslBytes"], report.total_wgsl_bytes)
        self.assertEqual(as_dict["featureHistogram"], {"f16": 2, "subgroups": 1})
        self.assertEqual(len(as_dict["shaders"]), 3)

    def test_diff_packs_added_removed_changed_and_unchanged(self) -> None:
        cs_v1 = MINIMAL_COMPUTE_WGSL
        cs_v2 = "enable f16;\n" + MINIMAL_COMPUTE_WGSL + "\n// optimized\n"

        pack_a = wgpk.build_wgpk(
            [
                (0x0001, "vertex", MINIMAL_VERTEX_WGSL),      # unchanged
                (0x0002, "fragment", MINIMAL_FRAGMENT_WGSL),  # removed in B
                (0x0003, "compute", cs_v1),                   # changed in B
            ]
        )
        pack_b = wgpk.build_wgpk(
            [
                (0x0001, "vertex", MINIMAL_VERTEX_WGSL),      # unchanged
                (0x0003, "compute", cs_v2),                   # changed in B
                (0x0004, "fragment", MINIMAL_FRAGMENT_WGSL),  # added in B
            ]
        )

        diff = wgpk.diff_packs(pack_a, pack_b)
        self.assertEqual(diff.added_hashes, (0x0004,))
        self.assertEqual(diff.removed_hashes, (0x0002,))
        self.assertEqual(diff.changed_hashes, (0x0003,))
        self.assertEqual(diff.unchanged_count, 1)
        expected_delta = len(cs_v2.encode("utf-8")) - len(cs_v1.encode("utf-8"))
        self.assertEqual(diff.byte_delta, expected_delta)
        self.assertEqual(len(diff.changed), 1)
        self.assertEqual(diff.changed[0].added_features, ("f16",))

        d = diff.to_dict()
        self.assertEqual(d["addedHashes"], ["0x0000000000000004"])
        self.assertEqual(d["removedHashes"], ["0x0000000000000002"])
        self.assertEqual(d["changedHashes"], ["0x0000000000000003"])
        self.assertEqual(d["unchangedCount"], 1)
        self.assertEqual(d["byteDelta"], expected_delta)

    def test_diff_identical_packs(self) -> None:
        pack = wgpk.build_wgpk([(0xABCD, "vertex", MINIMAL_VERTEX_WGSL)])
        diff = wgpk.diff_packs(pack, pack)
        self.assertEqual(diff.added_hashes, ())
        self.assertEqual(diff.removed_hashes, ())
        self.assertEqual(diff.changed_hashes, ())
        self.assertEqual(diff.unchanged_count, 1)
        self.assertEqual(diff.byte_delta, 0)
        self.assertEqual(diff.pack_byte_delta, 0)


class TestCliAndSanitization(unittest.TestCase):
    """Test CLI exit codes, JSON schema, and strict host-path/username sanitization."""

    def test_sanitize_source_label_cross_platform(self) -> None:
        self.assertEqual(
            wgpk.sanitize_source_label(r"C:\Users\secret_user\projects\game\shaders.wgsl.pack"),
            "shaders.wgsl.pack",
        )
        self.assertEqual(
            wgpk.sanitize_source_label("/home/secret_user/builds/shaders.wgsl.pack"),
            "shaders.wgsl.pack",
        )
        self.assertEqual(
            wgpk.sanitize_source_label(r"\\fileserver\share\pack_v2.wgsl.pack"),
            "pack_v2.wgsl.pack",
        )
        self.assertEqual(wgpk.sanitize_source_label("-"), "<stdin>")
        self.assertEqual(wgpk.sanitize_source_label(None), "<memory>")

    def test_cli_inspect_strict_and_sanitized_json(self) -> None:
        sensitive_segment = "secret_host_user_9f8e7d"
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / sensitive_segment / "private_dir"
            nested.mkdir(parents=True)

            clean_path = nested / "clean_bundle.wgsl.pack"
            clean_path.write_bytes(
                wgpk.build_wgpk([(0x101, "vertex", MINIMAL_VERTEX_WGSL)])
            )

            bad_budget_wgsl = """
            @group(5) @binding(0) var<uniform> u: vec4<f32>;
            @compute @workgroup_size(512) fn cs_over() {}
            """
            violating_path = nested / "over_budget.wgsl.pack"
            violating_path.write_bytes(
                wgpk.build_wgpk([(0x202, "compute", bad_budget_wgsl)])
            )

            corrupt_path = nested / "corrupt.wgsl.pack"
            corrupt_path.write_bytes(b"BAD!1234567890123456")

            missing_path = nested / "does_not_exist.wgsl.pack"

            # 1. Clean inspect --json --strict -> exit 0
            code, out, err = _run_cli(["inspect", str(clean_path), "--json", "--strict"])
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["clean"])
            self.assertEqual(payload["source"], "clean_bundle.wgsl.pack")
            self.assertEqual(payload["entryCount"], 1)
            self.assertNotIn(sensitive_segment, out)
            self.assertNotIn("private_dir", out)
            self.assertNotIn(tmpdir, out)

            # 2. Over-budget inspect without --strict -> exit 0, clean=False
            code, out, _ = _run_cli(["inspect", str(violating_path), "--json"])
            self.assertEqual(code, 0)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["clean"])
            self.assertGreaterEqual(payload["violationCount"], 2)
            self.assertNotIn(sensitive_segment, out)

            # 3. Over-budget inspect WITH --strict -> exit 1
            code, out, _ = _run_cli(["inspect", str(violating_path), "--json", "--strict"])
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["clean"])
            self.assertTrue(payload["strict"])
            self.assertEqual(payload["source"], "over_budget.wgsl.pack")
            self.assertNotIn(sensitive_segment, out)
            self.assertNotIn(tmpdir, out)

            # 4. Corrupt pack -> exit 1 + sanitized JSON error
            code, out, _ = _run_cli(["inspect", str(corrupt_path), "--json"])
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["errorCode"], "invalid_magic")
            self.assertEqual(payload["source"], "corrupt.wgsl.pack")
            self.assertNotIn(sensitive_segment, out)
            self.assertNotIn(tmpdir, out)

            # 5. Missing file -> exit 1 + sanitized JSON error (no OSError path leak!)
            code, out, _ = _run_cli(["inspect", str(missing_path), "--json"])
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["errorCode"], "io_error")
            self.assertEqual(payload["source"], "does_not_exist.wgsl.pack")
            self.assertNotIn(sensitive_segment, out)
            self.assertNotIn("private_dir", out)
            self.assertNotIn(tmpdir, out)

            # 6. Diff --json -> exit 0 + sanitized packA/packB basenames
            code, out, err = _run_cli(["diff", str(clean_path), str(violating_path), "--json"])
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["packA"], "clean_bundle.wgsl.pack")
            self.assertEqual(payload["packB"], "over_budget.wgsl.pack")
            self.assertEqual(payload["addedHashes"], ["0x0000000000000202"])
            self.assertEqual(payload["removedHashes"], ["0x0000000000000101"])
            self.assertEqual(payload["unchangedCount"], 0)
            self.assertNotIn(sensitive_segment, out)
            self.assertNotIn("private_dir", out)
            self.assertNotIn(tmpdir, out)

            # 7. Human-readable inspect & diff text modes succeed cleanly
            code, text_out, _ = _run_cli(["inspect", str(clean_path)])
            self.assertEqual(code, 0)
            self.assertIn("clean_bundle.wgsl.pack", text_out)
            self.assertNotIn(sensitive_segment, text_out)

            code, diff_text, _ = _run_cli(["diff", str(clean_path), str(violating_path)])
            self.assertEqual(code, 0)
            self.assertIn("clean_bundle.wgsl.pack -> over_budget.wgsl.pack", diff_text)
            self.assertNotIn(sensitive_segment, diff_text)


class TestDiffDumpCoverage(unittest.TestCase):
    """Tests for --diff-dump pipeline dump coverage audit (T45 / #931)."""

    def _write_reflect(self, path: Path, stages: dict[str, str]) -> Path:
        payload = {
            "schema": "playgta6-stage-reflection",
            "version": 1,
            "stages": {
                hash_key: {"stage": stage_name, "bindings": []}
                for hash_key, stage_name in stages.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def test_parse_wgpk_toc_matches_pack_entries_and_errors(self) -> None:
        raw = wgpk.build_wgpk(
            [
                (0x1111, "vertex", MINIMAL_VERTEX_WGSL),
                (0x2222, "fragment", MINIMAL_FRAGMENT_WGSL),
                (0x3333, "compute", MINIMAL_COMPUTE_WGSL),
            ],
            flags=7,
        )
        toc = wgpk.parse_wgpk_toc(raw)
        self.assertEqual(toc.version, 1)
        self.assertEqual(toc.flags, 7)
        self.assertEqual(toc.entry_count, 3)
        self.assertEqual(toc.hashes, frozenset({0x1111, 0x2222, 0x3333}))
        self.assertEqual(toc.entries[0].stage, wgpk.ShaderStage.VERTEX)
        self.assertEqual(toc.entries[1].stage, wgpk.ShaderStage.FRAGMENT)
        self.assertEqual(toc.entries[2].stage, wgpk.ShaderStage.COMPUTE)

        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk_toc(b"BAD!" + raw[4:])
        self.assertEqual(ctx.exception.code, "invalid_magic")

    def test_diff_dump_manifest_native_dxil_dedup_and_auto_reflect(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_path = root / "game.wgsl.pack"
            reflect_path = Path(f"{pack_path}.reflect.json")
            dump_dir = root / "live_dump"
            (dump_dir / "native").mkdir(parents=True)
            (dump_dir / "dxil").mkdir(parents=True)

            h_vs_in_both = 0x1000000000000001
            h_ps_pack_only = 0x1000000000000002
            h_cs_missing_both = 0x1000000000000003
            h_dxil_vs_in_pack = 0x2000000000000001
            h_dxil_ps_missing = 0x2000000000000002
            h_extra_bin_missing = 0x3000000000000001

            pack_path.write_bytes(
                wgpk.build_wgpk(
                    [
                        (h_vs_in_both, "vertex", MINIMAL_VERTEX_WGSL),
                        (h_ps_pack_only, "fragment", MINIMAL_FRAGMENT_WGSL),
                        (h_dxil_vs_in_pack, "vertex", MINIMAL_VERTEX_WGSL),
                        (0x9999999999999999, "compute", MINIMAL_COMPUTE_WGSL),
                    ]
                )
            )
            self._write_reflect(
                reflect_path,
                {
                    wgpk.format_shader_hash(h_vs_in_both): "vertex",
                    "0x9999999999999999": "compute",
                },
            )

            # Manifest with duplicate code_identity for h_vs_in_both (two metadata layouts)
            manifest = {
                "schema": "playgta6-pipeline-dump",
                "schema_version": 1,
                "native_stages": [
                    {
                        "code_identity": wgpk.format_shader_hash(h_vs_in_both),
                        "identity": "0xaaaaaaaaaaaaaaaa",
                        "stage": "vertex",
                        "layout": "A",
                        "code_bytes": 16,
                        "metadata_bytes": 4,
                        "code_file": f"native/{wgpk.format_shader_hash(h_vs_in_both)}.bin",
                        "metadata_file": f"native/{wgpk.format_shader_hash(h_vs_in_both)}.meta",
                    },
                    {
                        "code_identity": wgpk.format_shader_hash(h_vs_in_both),
                        "identity": "0xbbbbbbbbbbbbbbbb",
                        "stage": "vertex",
                        "layout": "B",
                        "code_bytes": 16,
                        "metadata_bytes": 4,
                        "code_file": f"native/{wgpk.format_shader_hash(h_vs_in_both)}.bin",
                        "metadata_file": f"native/{wgpk.format_shader_hash(h_vs_in_both)}-0xbbbbbbbbbbbbbbbb.meta",
                    },
                    {
                        "code_identity": wgpk.format_shader_hash(h_ps_pack_only),
                        "identity": "0xcccccccccccccccc",
                        "stage": "pixel",
                        "layout": "A",
                        "code_bytes": 16,
                        "metadata_bytes": 4,
                        "code_file": f"native/{wgpk.format_shader_hash(h_ps_pack_only)}.bin",
                        "metadata_file": f"native/{wgpk.format_shader_hash(h_ps_pack_only)}.meta",
                    },
                    {
                        "code_identity": wgpk.format_shader_hash(h_cs_missing_both),
                        "identity": "0xdddddddddddddddd",
                        "stage": "compute",
                        "layout": "A",
                        "code_bytes": 16,
                        "metadata_bytes": 4,
                        "code_file": f"native/{wgpk.format_shader_hash(h_cs_missing_both)}.bin",
                        "metadata_file": f"native/{wgpk.format_shader_hash(h_cs_missing_both)}.meta",
                    },
                ],
                "dxil": [
                    {
                        "identity": wgpk.format_shader_hash(h_dxil_vs_in_pack),
                        "hash_digest": "",
                        "kind": 1,
                        "bytes": 32,
                        "file": f"dxil/{wgpk.format_shader_hash(h_dxil_vs_in_pack)}.dxbc",
                    },
                    {
                        "identity": wgpk.format_shader_hash(h_dxil_ps_missing),
                        "hash_digest": "",
                        "kind": 0,
                        "bytes": 32,
                        "file": f"dxil/{wgpk.format_shader_hash(h_dxil_ps_missing)}.dxbc",
                    },
                ],
            }
            (dump_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            # Also create an extra native .bin not yet flushed to manifest.json + an ignored .meta
            (dump_dir / "native" / f"{wgpk.format_shader_hash(h_vs_in_both)}.bin").write_bytes(b"\x00" * 16)
            (dump_dir / "native" / f"{wgpk.format_shader_hash(h_vs_in_both)}.meta").write_bytes(b"\x00" * 4)
            (dump_dir / "native" / f"{wgpk.format_shader_hash(h_extra_bin_missing)}.bin").write_bytes(b"\x00" * 16)
            (dump_dir / "native" / "not_a_hash.bin").write_bytes(b"\x00" * 8)

            report = wgpk.diff_pack_against_dump(pack_path, dump_dir)
            self.assertTrue(report.manifest_loaded)
            self.assertTrue(report.reflect_loaded)
            self.assertEqual(report.reflect_source, "game.wgsl.pack.reflect.json")
            self.assertEqual(report.pack_entry_count, 4)
            self.assertEqual(report.reflect_stage_count, 2)
            self.assertEqual(report.dump_total_stages, 6)
            self.assertEqual(report.in_pack, 3)
            self.assertEqual(report.missing_from_pack, 3)
            self.assertEqual(report.in_reflect, 1)
            self.assertEqual(report.missing_from_reflect, 5)
            self.assertEqual(report.in_pack_missing_from_reflect, 2)
            self.assertFalse(report.clean)

            self.assertEqual(
                report.by_stage["vertex"],
                wgpk.StageCoverageCounts(2, 2, 0, 1, 1),
            )
            self.assertEqual(
                report.by_stage["fragment"],
                wgpk.StageCoverageCounts(2, 1, 1, 0, 2),
            )
            self.assertEqual(
                report.by_stage["compute"],
                wgpk.StageCoverageCounts(1, 0, 1, 0, 1),
            )
            self.assertEqual(
                report.by_stage["unknown"],
                wgpk.StageCoverageCounts(1, 0, 1, 0, 1),
            )
            self.assertEqual(
                report.by_origin["native"],
                wgpk.StageCoverageCounts(4, 2, 2, 1, 3),
            )
            self.assertEqual(
                report.by_origin["dxil"],
                wgpk.StageCoverageCounts(2, 1, 1, 0, 2),
            )
            self.assertEqual(
                report.sample_missing_pack_hashes,
                (
                    wgpk.format_shader_hash(h_cs_missing_both),
                    wgpk.format_shader_hash(h_dxil_ps_missing),
                    wgpk.format_shader_hash(h_extra_bin_missing),
                ),
            )
            self.assertEqual(
                report.sample_missing_reflect_hashes,
                (
                    wgpk.format_shader_hash(h_ps_pack_only),
                    wgpk.format_shader_hash(h_cs_missing_both),
                    wgpk.format_shader_hash(h_dxil_vs_in_pack),
                    wgpk.format_shader_hash(h_dxil_ps_missing),
                    wgpk.format_shader_hash(h_extra_bin_missing),
                ),
            )

    def test_diff_dump_scan_native_bins_without_manifest_and_sample_capping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_path = root / "pack.wgsl.pack"
            dump_dir = root / "raw_dump"
            native_dir = dump_dir / "native"
            native_dir.mkdir(parents=True)

            hashes = [0x0000000000000010 + i for i in range(6)]
            for h in hashes:
                (native_dir / f"{wgpk.format_shader_hash(h)}.bin").write_bytes(b"\x01\x02\x03\x04")
                (native_dir / f"{wgpk.format_shader_hash(h)}.meta").write_bytes(b"\x05\x06")

            # Pack contains only the first 2 hashes; no reflect sidecar present.
            pack_path.write_bytes(
                wgpk.build_wgpk(
                    [
                        (hashes[0], "vertex", MINIMAL_VERTEX_WGSL),
                        (hashes[1], "fragment", MINIMAL_FRAGMENT_WGSL),
                    ]
                )
            )

            report = wgpk.diff_pack_against_dump(pack_path, dump_dir, max_samples=2)
            self.assertFalse(report.manifest_loaded)
            self.assertFalse(report.reflect_loaded)
            self.assertIsNone(report.reflect_source)
            self.assertEqual(report.dump_total_stages, 6)
            self.assertEqual(report.in_pack, 2)
            self.assertEqual(report.missing_from_pack, 4)
            self.assertEqual(report.in_reflect, 0)
            self.assertEqual(report.missing_from_reflect, 6)
            self.assertEqual(report.by_stage["unknown"].dump_total, 6)
            self.assertEqual(
                report.sample_missing_pack_hashes,
                (wgpk.format_shader_hash(hashes[2]), wgpk.format_shader_hash(hashes[3])),
            )
            self.assertEqual(
                report.sample_missing_reflect_hashes,
                (wgpk.format_shader_hash(hashes[0]), wgpk.format_shader_hash(hashes[1])),
            )

            as_dict = report.to_dict()
            self.assertTrue(as_dict["missing_pack_hashes_truncated"])
            self.assertTrue(as_dict["missing_reflect_hashes_truncated"])

            # Direct bare native directory works too (with optional stage prefixes)
            bare_dir = root / "bare_native"
            bare_dir.mkdir()
            (bare_dir / f"vs_{wgpk.format_shader_hash(hashes[0])}.bin").write_bytes(b"\x00")
            (bare_dir / f"ps_{wgpk.format_shader_hash(hashes[1])}.bin").write_bytes(b"\x00")
            bare_report = wgpk.diff_pack_against_dump(pack_path, bare_dir, max_samples=0)
            self.assertEqual(bare_report.dump_total_stages, 2)
            self.assertEqual(bare_report.in_pack, 2)
            self.assertEqual(bare_report.missing_from_pack, 0)
            self.assertTrue(bare_report.clean)
            self.assertEqual(bare_report.by_stage["vertex"].in_pack, 1)
            self.assertEqual(bare_report.by_stage["fragment"].in_pack, 1)
            self.assertEqual(bare_report.sample_missing_reflect_hashes, ())

    def test_cli_diff_dump_json_text_strict_override_and_sanitization(self) -> None:
        sensitive_segment = "secret_host_user_9f8e7d"
        secret_wgsl_marker = "SECRET_WGSL_SOURCE_BYTES_NEVER_LEAK"
        secret_bin_marker = "SECRET_NATIVE_BLOB_BYTES_NEVER_LEAK"

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / sensitive_segment / "private_dir"
            nested.mkdir(parents=True)

            h_vs = 0x000000000000aa01
            h_ps = 0x000000000000aa02
            h_cs = 0x000000000000aa03

            pack_path = nested / "staged.wgsl.pack"
            pack_path.write_bytes(
                wgpk.build_wgpk(
                    [
                        (h_vs, "vertex", f"// {secret_wgsl_marker}\n" + MINIMAL_VERTEX_WGSL),
                        (h_ps, "fragment", MINIMAL_FRAGMENT_WGSL),
                    ]
                )
            )
            auto_reflect = Path(f"{pack_path}.reflect.json")
            self._write_reflect(auto_reflect, {wgpk.format_shader_hash(h_vs): "vertex"})

            override_reflect = nested / "custom_override.reflect.json"
            self._write_reflect(
                override_reflect,
                {
                    wgpk.format_shader_hash(h_vs): "vertex",
                    wgpk.format_shader_hash(h_ps): "fragment",
                },
            )

            dump_dir = nested / "chrome_dump_01"
            (dump_dir / "native").mkdir(parents=True)
            (dump_dir / "native" / f"{wgpk.format_shader_hash(h_vs)}.bin").write_bytes(
                secret_bin_marker.encode("utf-8")
            )
            (dump_dir / "native" / f"{wgpk.format_shader_hash(h_ps)}.bin").write_bytes(
                secret_bin_marker.encode("utf-8")
            )
            (dump_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "playgta6-pipeline-dump",
                        "schema_version": 1,
                        "native_stages": [
                            {"code_identity": wgpk.format_shader_hash(h_vs), "stage": "vertex"},
                            {"code_identity": wgpk.format_shader_hash(h_ps), "stage": "pixel"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            # 1. Auto-detected sidecar: h_ps missing from reflect -> non-strict exits 0, strict exits 1
            code, out, err = _run_cli(
                ["inspect", str(pack_path), "--diff-dump", str(dump_dir), "--json"]
            )
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertFalse(payload["clean"])
            self.assertEqual(payload["source"], "staged.wgsl.pack")
            self.assertEqual(payload["dump_source"], "chrome_dump_01")
            self.assertEqual(payload["reflect_source"], "staged.wgsl.pack.reflect.json")
            self.assertEqual(payload["dump_total_stages"], 2)
            self.assertEqual(payload["in_pack"], 2)
            self.assertEqual(payload["missing_from_pack"], 0)
            self.assertEqual(payload["in_reflect"], 1)
            self.assertEqual(payload["missing_from_reflect"], 1)
            self.assertEqual(payload["sample_missing_pack_hashes"], [])
            self.assertEqual(
                payload["sample_missing_reflect_hashes"],
                [wgpk.format_shader_hash(h_ps)],
            )
            for forbidden in (sensitive_segment, "private_dir", tmpdir, secret_wgsl_marker, secret_bin_marker):
                self.assertNotIn(forbidden, out)

            code, out_strict, _ = _run_cli(
                ["inspect", str(pack_path), "--diff-dump", str(dump_dir), "--json", "--strict"]
            )
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(out_strict)["ok"])

            # 2. Explicit --reflect override covers both stages -> --strict exits 0 (also test shorthand CLI!)
            code, out_override, err = _run_cli(
                [
                    str(pack_path),
                    "--diff-dump",
                    str(dump_dir),
                    "--reflect",
                    str(override_reflect),
                    "--json",
                    "--strict",
                ]
            )
            self.assertEqual(code, 0, msg=err)
            payload_ov = json.loads(out_override)
            self.assertTrue(payload_ov["ok"])
            self.assertTrue(payload_ov["clean"])
            self.assertEqual(payload_ov["reflect_source"], "custom_override.reflect.json")
            self.assertEqual(payload_ov["in_reflect"], 2)
            self.assertEqual(payload_ov["missing_from_reflect"], 0)
            for forbidden in (sensitive_segment, "private_dir", tmpdir, secret_wgsl_marker, secret_bin_marker):
                self.assertNotIn(forbidden, out_override)

            # 3. Human-readable text mode contains key counts & sanitized basenames only
            (dump_dir / "native" / f"{wgpk.format_shader_hash(h_cs)}.bin").write_bytes(b"\x00" * 8)
            code, text_out, err = _run_cli(
                ["inspect", str(pack_path), "--diff-dump", str(dump_dir), "--max-samples", "5"]
            )
            self.assertEqual(code, 0, msg=err)
            self.assertIn("staged.wgsl.pack vs chrome_dump_01", text_out)
            self.assertIn("Dump total stages:     3", text_out)
            self.assertIn("Missing from pack:     1", text_out)
            self.assertIn(wgpk.format_shader_hash(h_cs), text_out)
            for forbidden in (sensitive_segment, "private_dir", tmpdir, secret_wgsl_marker, secret_bin_marker):
                self.assertNotIn(forbidden, text_out)

    def test_cli_diff_dump_errors_are_sanitized(self) -> None:
        sensitive_segment = "secret_host_user_err_42"
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / sensitive_segment / "private_dir"
            nested.mkdir(parents=True)

            pack_path = nested / "ok.wgsl.pack"
            pack_path.write_bytes(wgpk.build_wgpk([(0x1, "vertex", MINIMAL_VERTEX_WGSL)]))

            # 1. Missing dump dir
            missing_dump = nested / "no_such_dump"
            code, out, _ = _run_cli(
                ["inspect", str(pack_path), "--diff-dump", str(missing_dump), "--json"]
            )
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["errorCode"], "dump_not_found")
            self.assertEqual(payload["source"], "no_such_dump")
            self.assertNotIn(sensitive_segment, out)
            self.assertNotIn(tmpdir, out)

            # 2. Invalid manifest schema
            bad_manifest_dir = nested / "bad_dump"
            bad_manifest_dir.mkdir()
            (bad_manifest_dir / "manifest.json").write_text(
                json.dumps({"schema": "other-schema", "schema_version": 1}),
                encoding="utf-8",
            )
            code, out, _ = _run_cli(
                ["inspect", str(pack_path), "--diff-dump", str(bad_manifest_dir), "--json"]
            )
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertEqual(payload["errorCode"], "invalid_manifest")
            self.assertNotIn(sensitive_segment, out)

            # 3. Invalid manifest schema_version
            (bad_manifest_dir / "manifest.json").write_text(
                json.dumps({"schema": "playgta6-pipeline-dump", "schema_version": 99}),
                encoding="utf-8",
            )
            code, out, _ = _run_cli(
                ["inspect", str(pack_path), "--diff-dump", str(bad_manifest_dir), "--json"]
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_manifest")

            # 4. Missing explicit --reflect sidecar
            empty_dump = nested / "empty_dump"
            empty_dump.mkdir()
            missing_reflect = nested / "missing.reflect.json"
            code, out, _ = _run_cli(
                [
                    "inspect",
                    str(pack_path),
                    "--diff-dump",
                    str(empty_dump),
                    "--reflect",
                    str(missing_reflect),
                    "--json",
                ]
            )
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertEqual(payload["errorCode"], "reflect_not_found")
            self.assertEqual(payload["source"], "missing.reflect.json")
            self.assertNotIn(sensitive_segment, out)

            # 5. Corrupt --reflect JSON
            bad_reflect = nested / "bad.reflect.json"
            bad_reflect.write_text("{not json", encoding="utf-8")
            code, out, _ = _run_cli(
                [
                    "inspect",
                    str(pack_path),
                    "--diff-dump",
                    str(empty_dump),
                    "--reflect",
                    str(bad_reflect),
                    "--json",
                ]
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_reflect_sidecar")
            self.assertNotIn(sensitive_segment, out)

            # 6. --reflect without --diff-dump -> error
            code, out, _ = _run_cli(
                ["inspect", str(pack_path), "--reflect", str(bad_reflect), "--json"]
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_argument")

            # 7. Negative --max-samples -> error
            code, out, _ = _run_cli(
                ["inspect", str(pack_path), "--diff-dump", str(empty_dump), "--max-samples", "-1", "--json"]
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_argument")


class TestVerifyWgslSyntax(unittest.TestCase):
    """Tests for --verify-wgsl-syntax fast streaming pre-flight validator (T66 / #1210)."""

    def test_find_module_scope_let_lines_handles_comments_braces_and_identifiers(self) -> None:
        valid_wgsl = """
        // let commented_out = 1u;
        /* let block_commented = 2u; /* { let nested_commented = 3u; */ */
        const palette_size: u32 = 4u;
        alias let_alias = f32;
        struct Outlet {
            let_field: u32,
        };

        @compute @workgroup_size(1)
        fn cs_ok() {
            let local_a = palette_size;
            if (local_a > 0u) {
                let nested_b = local_a + 1u;
            }
            let local_c = 2u;
        }
        """
        self.assertEqual(wgpk.find_module_scope_let_lines(valid_wgsl), ())

        invalid_wgsl = "\n".join(
            [
                "// /* not a block comment",
                "let bad_top_1: u32 = 1u;",
                "struct S { x: u32 }; let bad_after_struct = 2u;",
                "@compute @workgroup_size(1) fn cs() {",
                "    let ok_inside_fn = 3u;",
                "}",
                "/* // */",
                "let /* inline comment */ bad_with_comment = 4u;",
                "let",
                "    bad_multiline = 5u;",
            ]
        )
        self.assertEqual(wgpk.find_module_scope_let_lines(invalid_wgsl), (2, 3, 8, 9))

    def test_valid_pack_and_empty_pack_pass_streaming_verify(self) -> None:
        wgsl_with_local_lets = """
        @vertex
        fn vs_main(@location(0) p: vec3<f32>) -> @builtin(position) vec4<f32> {
            let pos = vec4<f32>(p, 1.0);
            return pos;
        }
        """.strip()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            valid_pack = root / "valid.wgsl.pack"
            valid_pack.write_bytes(
                wgpk.build_wgpk(
                    [
                        (0x1000000000000001, "vertex", wgsl_with_local_lets),
                        (0x1000000000000002, "fragment", MINIMAL_FRAGMENT_WGSL),
                        (0x1000000000000003, "compute", MINIMAL_COMPUTE_WGSL),
                    ],
                    flags=0x00000003,
                )
            )

            report = wgpk.verify_wgsl_pack_syntax(valid_pack)
            self.assertTrue(report.clean)
            self.assertEqual(report.version, 1)
            self.assertEqual(report.flags, 3)
            self.assertEqual(report.entry_count, 3)
            self.assertEqual(report.entries_checked, 3)
            self.assertEqual(report.duplicate_hash_count, 0)
            self.assertEqual(report.module_scope_let_count, 0)
            self.assertEqual(report.violations, ())

            # CLI JSON mode (explicit `inspect`) -> exit 0
            code, out, err = _run_cli(["inspect", str(valid_pack), "--verify-wgsl-syntax", "--json"])
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["clean"])
            self.assertEqual(payload["schema"], wgpk.SYNTAX_VERIFY_REPORT_SCHEMA)
            self.assertEqual(payload["source"], "valid.wgsl.pack")
            self.assertEqual(payload["entriesChecked"], 3)
            self.assertEqual(payload["violationCount"], 0)
            self.assertNotIn(tmpdir, out)

            # CLI shorthand mode (without `inspect`) + text output -> exit 0
            code, text_out, err = _run_cli([str(valid_pack), "--verify-wgsl-syntax"])
            self.assertEqual(code, 0, msg=err)
            self.assertIn("WGPK Syntax Pre-flight [OK]: valid.wgsl.pack", text_out)
            self.assertNotIn(tmpdir, text_out)

            # Valid empty pack (0 entries) -> exit 0
            empty_pack = root / "empty.wgsl.pack"
            empty_pack.write_bytes(wgpk.build_wgpk([]))
            code, out, err = _run_cli(["inspect", str(empty_pack), "--verify-wgsl-syntax", "--json"])
            self.assertEqual(code, 0, msg=err)
            self.assertTrue(json.loads(out)["clean"])

    def test_module_scope_let_pack_fails_nonzero_and_never_leaks_source_or_host_path(self) -> None:
        sensitive_dir = "secret_host_user_t66_verify"
        secret_ident = "SECRET_SHADER_IDENT_NEVER_LEAK_77"
        broken_wgsl = (
            "// valid header comment\n"
            f"let {secret_ident}: f32 = 1.0;\n"
            "@fragment fn fs_main() -> @location(0) vec4<f32> {\n"
            "    let ok_local = 0.5;\n"
            f"    return vec4<f32>({secret_ident}, ok_local, 0.0, 1.0);\n"
            "}\n"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / sensitive_dir / "private_shaders"
            nested.mkdir(parents=True)
            bad_pack = nested / "broken_let.wgsl.pack"
            bad_hash = 0xCAFEBABE00001234
            bad_pack.write_bytes(
                wgpk.build_wgpk(
                    [
                        (0x0000000000000001, "vertex", MINIMAL_VERTEX_WGSL),
                        (bad_hash, "fragment", broken_wgsl),
                    ]
                )
            )

            # JSON mode exits 1 (without requiring --strict)
            code, out, _ = _run_cli(["inspect", str(bad_pack), "--verify-wgsl-syntax", "--json"])
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["clean"])
            self.assertEqual(payload["source"], "broken_let.wgsl.pack")
            self.assertEqual(payload["moduleScopeLetCount"], 1)
            self.assertEqual(payload["violationsByCode"], {"module_scope_let": 1})
            self.assertEqual(len(payload["violations"]), 1)
            v0 = payload["violations"][0]
            self.assertEqual(v0["code"], "module_scope_let")
            self.assertEqual(v0["entryIndex"], 1)
            self.assertEqual(v0["hash"], wgpk.format_shader_hash(bad_hash))
            self.assertEqual(v0["line"], 2)

            for forbidden in (sensitive_dir, "private_shaders", tmpdir, secret_ident, "ok_local"):
                self.assertNotIn(forbidden, out)

            # Text mode also exits 1 and stays strictly sanitized
            code, text_out, _ = _run_cli(["inspect", str(bad_pack), "--verify-wgsl-syntax"])
            self.assertEqual(code, 1)
            self.assertIn("WGPK Syntax Pre-flight [FAILED]: broken_let.wgsl.pack", text_out)
            self.assertIn("module_scope_let", text_out)
            self.assertIn(wgpk.format_shader_hash(bad_hash), text_out)
            for forbidden in (sensitive_dir, "private_shaders", tmpdir, secret_ident, "ok_local"):
                self.assertNotIn(forbidden, text_out)

    def test_header_bounds_overlap_zero_length_stage_and_utf8_violations(self) -> None:
        valid = wgpk.build_wgpk(
            [
                (0x1111000000000001, "vertex", MINIMAL_VERTEX_WGSL),
                (0x1111000000000002, "fragment", MINIMAL_FRAGMENT_WGSL),
                (0x1111000000000003, "compute", MINIMAL_COMPUTE_WGSL),
            ]
        )

        # 1. Truncated header (< 16 bytes)
        rep = wgpk.verify_wgsl_pack_syntax(valid[:10])
        self.assertFalse(rep.clean)
        self.assertEqual(rep.violations_by_code, {"truncated_header": 1})

        # 2. Invalid magic (never leaks raw magic bytes)
        bad_magic = b"SECR" + valid[4:]
        rep = wgpk.verify_wgsl_pack_syntax(bad_magic)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.violations_by_code, {"invalid_magic": 1})
        self.assertNotIn("SECR", json.dumps(rep.to_dict()))

        # 3. Unsupported version
        bad_ver = bytearray(valid)
        struct.pack_into("<I", bad_ver, 4, 7)
        rep = wgpk.verify_wgsl_pack_syntax(bytes(bad_ver))
        self.assertFalse(rep.clean)
        self.assertEqual(rep.violations_by_code, {"unsupported_version": 1})

        # 4. Truncated entry table (entry_count claims millions of entries on tiny file)
        huge_count = bytearray(valid[:24])
        struct.pack_into("<I", huge_count, 8, 5_000_000)
        rep = wgpk.verify_wgsl_pack_syntax(bytes(huge_count))
        self.assertFalse(rep.clean)
        self.assertEqual(rep.violations_by_code, {"truncated_table": 1})

        # 5. Out-of-bounds offset inside header + truncated payload at end of file
        oob = bytearray(valid)
        struct.pack_into("<I", oob, 16 + 12, 8)  # entry 0 offset inside header
        struct.pack_into("<I", oob, 16 + 2 * 20 + 16, 999999)  # entry 2 length past EOF
        rep = wgpk.verify_wgsl_pack_syntax(bytes(oob))
        self.assertFalse(rep.clean)
        self.assertEqual(rep.violations_by_code.get("out_of_bounds"), 2)

        # 6. Overlapping payload ranges (3-way overlap: entry 0 spans entries 1 and 2)
        overlap = bytearray(valid)
        e0_off = struct.unpack_from("<I", overlap, 16 + 12)[0]
        e0_len = struct.unpack_from("<I", overlap, 16 + 16)[0]
        # Make entry 1 start inside entry 0 and entry 2 also start inside entry 0
        struct.pack_into("<I", overlap, 16 + 20 + 12, e0_off + 2)
        struct.pack_into("<I", overlap, 16 + 20 + 16, 8)
        struct.pack_into("<I", overlap, 16 + 40 + 12, e0_off + 12)
        struct.pack_into("<I", overlap, 16 + 40 + 16, min(10, e0_len - 12))
        rep = wgpk.verify_wgsl_pack_syntax(bytes(overlap))
        self.assertFalse(rep.clean)
        self.assertEqual(rep.violations_by_code.get("overlapping_payloads"), 2)

        # 7. Zero-length corrupt entry
        zero_pack = wgpk.build_wgpk(
            [
                (0xAAAA000000000001, "vertex", MINIMAL_VERTEX_WGSL),
                (0xAAAA000000000002, "fragment", ""),
            ]
        )
        rep = wgpk.verify_wgsl_pack_syntax(zero_pack)
        self.assertFalse(rep.clean)
        self.assertEqual(rep.violations_by_code, {"zero_length_payload": 1})
        self.assertEqual(rep.violations[0].entry_index, 1)

        # 8. Invalid stage code + invalid UTF-8 payload
        bad_stage_utf8 = bytearray(valid)
        bad_stage_utf8[16 + 8] = 99
        payload_start = 16 + 3 * 20
        bad_stage_utf8[payload_start] = 0xFF
        bad_stage_utf8[payload_start + 1] = 0xFE
        rep = wgpk.verify_wgsl_pack_syntax(bytes(bad_stage_utf8))
        self.assertFalse(rep.clean)
        self.assertEqual(rep.violations_by_code.get("invalid_stage"), 1)
        self.assertEqual(rep.violations_by_code.get("invalid_utf8"), 1)

    def test_duplicate_hash_pack_fails_verify_and_reports_count(self) -> None:
        dup_hash = 0xDEADBEEF12345678
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "dup_hashes.wgsl.pack"
            pack_path.write_bytes(
                wgpk.build_wgpk(
                    [
                        (dup_hash, "vertex", MINIMAL_VERTEX_WGSL),
                        (0x0000000000000009, "fragment", MINIMAL_FRAGMENT_WGSL),
                        (dup_hash, "compute", MINIMAL_COMPUTE_WGSL),
                    ]
                )
            )
            code, out, _ = _run_cli(["inspect", str(pack_path), "--verify-wgsl-syntax", "--json"])
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["clean"])
            self.assertEqual(payload["duplicateHashCount"], 1)
            self.assertEqual(payload["violationsByCode"], {"duplicate_shader_hash": 1})
            self.assertEqual(payload["violations"][0]["entryIndex"], 2)
            self.assertEqual(payload["violations"][0]["hash"], wgpk.format_shader_hash(dup_hash))

    def test_verify_missing_file_returns_sanitized_io_error(self) -> None:
        sensitive = "secret_host_user_missing_verify"
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / sensitive / "absent.wgsl.pack"
            code, out, _ = _run_cli(["inspect", str(missing), "--verify-wgsl-syntax", "--json"])
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["errorCode"], "io_error")
            self.assertEqual(payload["source"], "absent.wgsl.pack")
            self.assertNotIn(sensitive, out)
            self.assertNotIn(tmpdir, out)


class TestEmitPriorityHashes(unittest.TestCase):
    """Tests for --emit-priority-hashes and --from-boot-log (T66 / #1210)."""

    def test_extract_shader_hashes_from_boot_log_text_patterns_and_exclusions(self) -> None:
        log_text = "\ufeff" + "\n".join(
            [
                "# Sanitized browser/native boot log",
                "1700000000123456 [wgpu] frame=1 dt=16.6ms short=0xdeadbeef",
                "1700000000123457 [wgpu] shader pack miss: 0xFEDCBA9876543210 (stage=vertex)",
                "1700000000123458 [wgpu] shader pack hit: 0x0000000000000042 (stage=fragment)",
                "1700000000123459 [wgpu] shader miss deadbeef00001111 fallback=frame1",
                "1700000000123460 [wgpu] shader hit: 0000000000000099",
                '{"event":"pso_miss","vertexShaderHash":"0x1122334455667788","fragmentShaderHash":"0xFEDCBA9876543210"}',
                "sha256=0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                "aabbccddeeff0011 # standalone priority hash line",
                "0x0000000000000042 # duplicate",
            ]
        )
        extracted = wgpk.extract_shader_hashes_from_log_text(log_text)
        expected = tuple(
            sorted(
                [
                    0xFEDCBA9876543210,
                    0x0000000000000042,
                    0xDEADBEEF00001111,
                    0x0000000000000099,
                    0x1122334455667788,
                    0xAABBCCDDEEFF0011,
                ]
            )
        )
        self.assertEqual(extracted, expected)
        # Ensure 16-digit decimal timestamps and 64-hex SHA-256 slices were NOT extracted.
        self.assertNotIn(int("1700000000123456", 16), extracted)
        self.assertNotIn(int("1700000000123457", 16), extracted)
        self.assertNotIn(0x0123456789ABCDEF, extracted)

    def test_emit_priority_hashes_pack_only_merged_and_log_only(self) -> None:
        sensitive = "secret_host_user_t66_priority"
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / sensitive / "work"
            nested.mkdir(parents=True)

            h_pack_1 = 0x3000000000000001
            h_pack_2 = 0x1000000000000002
            h_shared = 0x2000000000000003
            h_log_miss = 0x000000000000AA01
            h_log_hit = 0xFFFF00000000BB02

            pack_path = nested / "live.wgsl.pack"
            pack_path.write_bytes(
                wgpk.build_wgpk(
                    [
                        (h_pack_1, "vertex", MINIMAL_VERTEX_WGSL),
                        (h_pack_2, "fragment", MINIMAL_FRAGMENT_WGSL),
                        (h_shared, "compute", MINIMAL_COMPUTE_WGSL),
                    ]
                )
            )

            boot_log_a = nested / "chrome_boot_1.log"
            boot_log_a.write_text(
                "\n".join(
                    [
                        f"[wgpu] shader pack miss {wgpk.format_shader_hash(h_log_miss)}",
                        f"[wgpu] shader pack hit {wgpk.format_shader_hash(h_shared).upper().replace('0X', '0x')}",
                    ]
                ),
                encoding="utf-8",
            )
            boot_log_b = nested / "native_boot_2.log"
            boot_log_b.write_text(
                f"shader hit={wgpk.format_shader_hash(h_log_hit)}\n",
                encoding="utf-8",
            )

            # 1. Pack-only export
            out_pack_only = nested / "out" / "pack_only.txt"
            code, out, err = _run_cli(
                [
                    "inspect",
                    str(pack_path),
                    "--emit-priority-hashes",
                    str(out_pack_only),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["schema"], wgpk.PRIORITY_HASH_EXPORT_SCHEMA)
            self.assertEqual(payload["output"], "pack_only.txt")
            self.assertEqual(payload["packSource"], "live.wgsl.pack")
            self.assertEqual(payload["bootLogSources"], [])
            self.assertEqual(payload["packHashCount"], 3)
            self.assertEqual(payload["bootLogHashCount"], 0)
            self.assertEqual(payload["uniqueHashCount"], 3)
            self.assertNotIn(sensitive, out)
            self.assertNotIn(tmpdir, out)
            self.assertEqual(
                out_pack_only.read_text(encoding="utf-8").splitlines(),
                [
                    wgpk.format_shader_hash(h_pack_2),
                    wgpk.format_shader_hash(h_shared),
                    wgpk.format_shader_hash(h_pack_1),
                ],
            )

            # 2. Merged pack + multiple boot logs (deduplicated & sorted)
            out_merged = nested / "merged_priority.txt"
            code, out, err = _run_cli(
                [
                    str(pack_path),
                    "--from-boot-log",
                    str(boot_log_a),
                    "--from-boot-log",
                    str(boot_log_b),
                    "--emit-priority-hashes",
                    str(out_merged),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["packHashCount"], 3)
            self.assertEqual(payload["bootLogHashCount"], 3)
            self.assertEqual(payload["uniqueHashCount"], 5)
            self.assertEqual(payload["bootLogSources"], ["chrome_boot_1.log", "native_boot_2.log"])
            self.assertNotIn(sensitive, out)
            self.assertEqual(
                out_merged.read_text(encoding="utf-8").splitlines(),
                [
                    wgpk.format_shader_hash(h_log_miss),
                    wgpk.format_shader_hash(h_pack_2),
                    wgpk.format_shader_hash(h_shared),
                    wgpk.format_shader_hash(h_pack_1),
                    wgpk.format_shader_hash(h_log_hit),
                ],
            )

            # 3. Boot-log-only export (no pack argument, shorthand CLI + text output)
            out_log_only = nested / "log_only.txt"
            code, text_out, err = _run_cli(
                [
                    "--from-boot-log",
                    str(boot_log_a),
                    "--emit-priority-hashes",
                    str(out_log_only),
                ]
            )
            self.assertEqual(code, 0, msg=err)
            self.assertIn("WGPK Priority Hashes Exported: log_only.txt", text_out)
            self.assertIn("Emitted unique:     2", text_out)
            self.assertNotIn(sensitive, text_out)
            self.assertEqual(
                out_log_only.read_text(encoding="utf-8").splitlines(),
                [
                    wgpk.format_shader_hash(h_log_miss),
                    wgpk.format_shader_hash(h_shared),
                ],
            )

    def test_combined_verify_and_emit_gates_output_on_clean_pack(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bad_pack = root / "bad.wgsl.pack"
            bad_pack.write_bytes(
                wgpk.build_wgpk([(0x1111, "compute", "let bad = 1u;\n@compute @workgroup_size(1) fn cs() {}")])
            )
            out_blocked = root / "should_not_exist.txt"
            code, out, _ = _run_cli(
                [
                    "inspect",
                    str(bad_pack),
                    "--verify-wgsl-syntax",
                    "--emit-priority-hashes",
                    str(out_blocked),
                    "--json",
                ]
            )
            self.assertEqual(code, 1)
            self.assertFalse(json.loads(out)["ok"])
            self.assertFalse(out_blocked.exists())

            good_pack = root / "good.wgsl.pack"
            good_pack.write_bytes(wgpk.build_wgpk([(0x2222, "compute", MINIMAL_COMPUTE_WGSL)]))
            out_ok = root / "ok_hashes.txt"
            code, out, err = _run_cli(
                [
                    "inspect",
                    str(good_pack),
                    "--verify-wgsl-syntax",
                    "--emit-priority-hashes",
                    str(out_ok),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["clean"])
            self.assertEqual(payload["priorityHashes"]["uniqueHashCount"], 1)
            self.assertEqual(
                out_ok.read_text(encoding="utf-8").splitlines(),
                [wgpk.format_shader_hash(0x2222)],
            )

    def test_emit_priority_hashes_cli_errors_are_sanitized(self) -> None:
        sensitive = "secret_host_user_priority_err"
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / sensitive
            nested.mkdir(parents=True)
            pack_path = nested / "ok.wgsl.pack"
            pack_path.write_bytes(wgpk.build_wgpk([(0x1, "vertex", MINIMAL_VERTEX_WGSL)]))
            out_path = nested / "out.txt"
            missing_log = nested / "missing_boot.log"

            # 1. --from-boot-log without --emit-priority-hashes -> error
            code, out, _ = _run_cli(
                ["inspect", str(pack_path), "--from-boot-log", str(missing_log), "--json"]
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_argument")

            # 2. --emit-priority-hashes with neither pack nor boot log -> error
            code, out, _ = _run_cli(
                ["inspect", "--emit-priority-hashes", str(out_path), "--json"]
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_argument")

            # 3. Missing boot log file -> sanitized io_error
            code, out, _ = _run_cli(
                [
                    "inspect",
                    "--from-boot-log",
                    str(missing_log),
                    "--emit-priority-hashes",
                    str(out_path),
                    "--json",
                ]
            )
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertEqual(payload["errorCode"], "io_error")
            self.assertEqual(payload["source"], "missing_boot.log")
            self.assertNotIn(sensitive, out)
            self.assertNotIn(tmpdir, out)

            # 4. --diff-dump combined with --verify-wgsl-syntax -> invalid_argument
            code, out, _ = _run_cli(
                [
                    "inspect",
                    str(pack_path),
                    "--diff-dump",
                    str(nested),
                    "--verify-wgsl-syntax",
                    "--json",
                ]
            )
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_argument")

class WgpkV2LargeOffsetTests(unittest.TestCase):
    ENTRIES = [
        (0x11, "vertex", "fn a() {}"),
        (0x22, "fragment", "fn bb() {}"),
        (0x33, "compute", ""),
    ]

    def test_v2_round_trip_matches_v1_content(self) -> None:
        v1 = wgpk.build_wgpk(self.ENTRIES, version=1)
        v2 = wgpk.build_wgpk(self.ENTRIES, version=2)
        self.assertEqual(struct.unpack_from("<I", v2, 4)[0], 2)
        self.assertEqual(len(v2) - len(v1), 4 * len(self.ENTRIES))  # 24 vs 20 byte records
        b1, b2 = wgpk.parse_wgpk(v1), wgpk.parse_wgpk(v2)
        self.assertEqual(b2.version, 2)
        self.assertEqual(b2.data_start, 16 + 3 * 24)
        self.assertEqual(
            [(e.hash, e.stage_name, e.wgsl) for e in b1.entries],
            [(e.hash, e.stage_name, e.wgsl) for e in b2.entries],
        )
        toc = wgpk.parse_wgpk_toc(v2)
        self.assertEqual(toc.version, 2)
        self.assertEqual([e.length for e in toc.entries], [len("fn a() {}"), len("fn bb() {}"), 0])

    def test_v2_record_layout_and_64bit_offset_bounds(self) -> None:
        v2 = bytearray(wgpk.build_wgpk(self.ENTRIES, version=2))
        # Record 0: hash(8) stage(1) rsvd(3) length(4) offset(8).
        length, offset = struct.unpack_from("<IQ", v2, 16 + 12)
        self.assertEqual(length, len("fn a() {}"))
        self.assertEqual(offset, 16 + 3 * 24)
        struct.pack_into("<Q", v2, 16 + 16, 5 << 30)  # offset beyond 4 GiB, past the buffer
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.parse_wgpk_toc(v2)
        self.assertEqual(ctx.exception.code, "out_of_bounds")

    def test_v2_streaming_toc_and_default_version(self) -> None:
        self.assertEqual(struct.unpack_from("<I", wgpk.build_wgpk(self.ENTRIES), 4)[0], 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "v2.wgsl.pack"
            path.write_bytes(wgpk.build_wgpk(self.ENTRIES, version=2))
            toc, _label = wgpk.load_wgpk_toc_streaming(path)
            self.assertEqual(toc.version, 2)
            self.assertEqual(len(toc.entries), 3)
        with self.assertRaises(ValueError):
            wgpk.build_wgpk(self.ENTRIES, version=3)


class _VirtualSparseWgpkStream:
    """In-memory seekable stream simulating a >4 GiB WGPK v2 pack without allocating GiB of RAM.

    Any attempt to read the entire stream at once (`size < 0` or `size > max_allowed_read`)
    immediately fails the test so true streaming I/O is strictly proven.
    """

    def __init__(
        self,
        total_size: int,
        segments: Sequence[tuple[int, bytes]],
        *,
        max_allowed_read: int = 1024 * 1024,
    ) -> None:
        self.total_size = total_size
        self.segments = sorted(segments, key=lambda item: item[0])
        self.max_allowed_read = max_allowed_read
        self.pos = 0
        self.max_single_read = 0
        self.total_bytes_read = 0
        self.read_ranges: list[tuple[int, int]] = []

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            new_pos = offset
        elif whence == 1:
            new_pos = self.pos + offset
        elif whence == 2:
            new_pos = self.total_size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if new_pos < 0:
            raise ValueError(f"negative seek position: {new_pos}")
        self.pos = new_pos
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            raise AssertionError("streaming reader must never call read(-1) on a >4 GiB pack")
        if size > self.max_allowed_read:
            raise AssertionError(
                f"streaming read size {size} exceeds max_allowed_read {self.max_allowed_read}"
            )
        if size == 0 or self.pos >= self.total_size:
            return b""
        count = min(size, self.total_size - self.pos)
        start = self.pos
        end = start + count
        buf = bytearray(count)
        for seg_off, seg_bytes in self.segments:
            seg_end = seg_off + len(seg_bytes)
            ov_start = max(start, seg_off)
            ov_end = min(end, seg_end)
            if ov_start < ov_end:
                buf[ov_start - start : ov_end - start] = seg_bytes[
                    ov_start - seg_off : ov_end - seg_off
                ]
        self.pos = end
        self.max_single_read = max(self.max_single_read, count)
        self.total_bytes_read += count
        self.read_ranges.append((start, end))
        return bytes(buf)


def _build_virtual_v2_stream(
    entries: Sequence[tuple[int, str, str, int]],
    *,
    flags: int = 0,
    extra_tail: int = 0,
) -> _VirtualSparseWgpkStream:
    """Build a virtual seekable WGPK v2 stream from `(hash, stage, wgsl, offset)` tuples."""
    header = wgpk.PACK_HEADER_STRUCT.pack(
        wgpk.PACK_MAGIC, wgpk.PACK_VERSION_V2, len(entries), flags
    )
    toc = bytearray()
    segments: list[tuple[int, bytes]] = []
    max_end = wgpk.PACK_HEADER_BYTES + len(entries) * wgpk.PACK_ENTRY_BYTES_V2
    for shader_hash, stage_name, wgsl, offset in entries:
        payload = wgsl.encode("utf-8")
        stage = wgpk.ShaderStage.from_value(stage_name)
        toc += wgpk.PACK_ENTRY_STRUCT_V2.pack(
            shader_hash, int(stage), b"\x00\x00\x00", len(payload), offset
        )
        if payload:
            segments.append((offset, payload))
        max_end = max(max_end, offset + len(payload))
    segments.insert(0, (0, header + bytes(toc)))
    return _VirtualSparseWgpkStream(max_end + extra_tail, segments)


class TestWgpkV2StreamingAndReflectVersions(unittest.TestCase):
    """T104 tests: >4 GiB WGPK v2 streaming inspect/diff/toc, reflect v1/v2/v3, native vs DXIL-only."""

    def test_gt4gib_v2_stream_toc_inspect_verify_and_diff_never_load_whole_pack(self) -> None:
        base_gt4g = (1 << 32) + 4096  # > 4 GiB (0x1_0000_1000)
        h_vs = 0xAA00000000000001
        h_fs = 0xAA00000000000002
        h_cs = 0xAA00000000000003
        h_added = 0xAA00000000000004

        stream_a = _build_virtual_v2_stream(
            [
                (h_vs, "vertex", MINIMAL_VERTEX_WGSL, base_gt4g),
                (h_fs, "fragment", MINIMAL_FRAGMENT_WGSL, base_gt4g + 4096),
                (h_cs, "compute", MINIMAL_COMPUTE_WGSL, base_gt4g + 8192),
            ],
            flags=0x00000005,
        )

        # 1. Streaming TOC reads ONLY header + TOC (zero payload bytes read!).
        toc, label = wgpk.load_wgpk_toc_streaming(stream_a, source_label="huge_v2.wgsl.pack")
        self.assertEqual(label, "huge_v2.wgsl.pack")
        self.assertEqual(toc.version, 2)
        self.assertEqual(toc.flags, 5)
        self.assertGreater(toc.byte_length, wgpk.U32_MAX)
        self.assertEqual(toc.entry_count, 3)
        self.assertEqual(toc.entries[0].offset, base_gt4g)
        self.assertEqual(toc.entries[2].offset, base_gt4g + 8192)
        expected_toc_bytes = wgpk.PACK_HEADER_BYTES + 3 * wgpk.PACK_ENTRY_BYTES_V2
        self.assertEqual(stream_a.total_bytes_read, expected_toc_bytes)

        # 2. Streaming inspect_pack audits each >4 GiB shader one entry at a time.
        report = wgpk.inspect_pack(stream_a, source_label="huge_v2.wgsl.pack")
        self.assertEqual(report.version, 2)
        self.assertEqual(report.flags, 5)
        self.assertGreater(report.pack_bytes, wgpk.U32_MAX)
        self.assertEqual(report.entry_count, 3)
        self.assertEqual(report.stage_counts, {"vertex": 1, "fragment": 1, "compute": 1})
        self.assertTrue(report.clean)
        self.assertEqual([s.shader_hash for s in report.shaders], [h_vs, h_fs, h_cs])
        self.assertLess(stream_a.max_single_read, 4096)

        # 3. Streaming syntax pre-flight handles >4 GiB offsets cleanly too.
        verify_rep = wgpk.verify_wgsl_pack_syntax(stream_a, source_label="huge_v2.wgsl.pack")
        self.assertTrue(verify_rep.clean)
        self.assertEqual(verify_rep.version, 2)
        self.assertEqual(verify_rep.entries_checked, 3)

        # 4. Streaming diff_packs between two >4 GiB WGPK v2 streams.
        modified_fs = MINIMAL_FRAGMENT_WGSL + "\n// modified_v2_stream\n"
        stream_b = _build_virtual_v2_stream(
            [
                (h_vs, "vertex", MINIMAL_VERTEX_WGSL, base_gt4g),
                (h_fs, "fragment", modified_fs, base_gt4g + 8192),
                (h_added, "compute", MINIMAL_COMPUTE_WGSL, base_gt4g + 16384),
            ],
            flags=0x00000005,
        )
        diff_rep = wgpk.diff_packs(
            stream_a,
            stream_b,
            label_a="base_v2.wgsl.pack",
            label_b="next_v2.wgsl.pack",
        )
        self.assertEqual(diff_rep.added_hashes, (h_added,))
        self.assertEqual(diff_rep.removed_hashes, (h_cs,))
        self.assertEqual(diff_rep.changed_hashes, (h_fs,))
        self.assertEqual(diff_rep.unchanged_count, 1)
        self.assertGreater(diff_rep.pack_bytes_a, wgpk.U32_MAX)
        self.assertGreater(diff_rep.pack_bytes_b, wgpk.U32_MAX)
        self.assertTrue(diff_rep.changed[0].source_changed)

    def test_v1_u32_overflow_rejected_and_v2_u64_overflow_rejected(self) -> None:
        # v1 entry whose offset + length overflows u32::MAX (even if stream total_size > 4 GiB)
        v1_header = wgpk.PACK_HEADER_STRUCT.pack(wgpk.PACK_MAGIC, wgpk.PACK_VERSION, 1, 0)
        v1_entry = wgpk.PACK_ENTRY_STRUCT.pack(
            0x1234, int(wgpk.ShaderStage.VERTEX), b"\x00\x00\x00", 0xFFFFFFF0, 0x100
        )
        v1_stream = _VirtualSparseWgpkStream(
            (1 << 32) + 4096,
            [(0, v1_header + v1_entry)],
        )
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.load_wgpk_toc_streaming(v1_stream)
        self.assertEqual(ctx.exception.code, "out_of_bounds")

        # v2 entry whose offset + length overflows u64::MAX
        v2_header = wgpk.PACK_HEADER_STRUCT.pack(wgpk.PACK_MAGIC, wgpk.PACK_VERSION_V2, 1, 0)
        v2_entry = wgpk.PACK_ENTRY_STRUCT_V2.pack(
            0x5678, int(wgpk.ShaderStage.COMPUTE), b"\x00\x00\x00", 64, wgpk.U64_MAX - 16
        )
        v2_stream = _VirtualSparseWgpkStream(
            wgpk.U64_MAX,
            [(0, v2_header + v2_entry)],
        )
        with self.assertRaises(wgpk.WgpkParseError) as ctx:
            wgpk.load_wgpk_toc_streaming(v2_stream)
        self.assertEqual(ctx.exception.code, "out_of_bounds")

    def test_filesystem_inspect_diff_and_diff_dump_never_call_read_bytes(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            pack_a = root / "a_v2.wgsl.pack"
            pack_b = root / "b_v2.wgsl.pack"
            dump_dir = root / "dump"
            (dump_dir / "native").mkdir(parents=True)

            h1 = 0x1010101010101001
            h2 = 0x1010101010101002
            pack_a.write_bytes(
                wgpk.build_wgpk([(h1, "vertex", MINIMAL_VERTEX_WGSL)], version=2)
            )
            pack_b.write_bytes(
                wgpk.build_wgpk(
                    [
                        (h1, "vertex", MINIMAL_VERTEX_WGSL),
                        (h2, "fragment", MINIMAL_FRAGMENT_WGSL),
                    ],
                    version=2,
                )
            )
            (dump_dir / "native" / f"{wgpk.format_shader_hash(h1)}.bin").write_bytes(b"\x00" * 8)

            def _forbid_read_bytes(self: Path) -> bytes:
                raise AssertionError(f"Path.read_bytes() must not be called for {self.name}")

            with mock.patch.object(Path, "read_bytes", _forbid_read_bytes):
                insp = wgpk.inspect_pack(pack_a)
                self.assertEqual(insp.version, 2)
                self.assertEqual(insp.entry_count, 1)

                diff_rep = wgpk.diff_packs(pack_a, pack_b)
                self.assertEqual(diff_rep.added_hashes, (h2,))

                dd_rep = wgpk.diff_pack_against_dump(pack_a, dump_dir)
                self.assertEqual(dd_rep.pack_version, 2)
                self.assertTrue(dd_rep.clean)

                code, out, err = _run_cli(["inspect", str(pack_a), "--json"])
                self.assertEqual(code, 0, msg=err)
                self.assertEqual(json.loads(out)["version"], 2)

                code, out, err = _run_cli(["diff", str(pack_a), str(pack_b), "--json"])
                self.assertEqual(code, 0, msg=err)
                self.assertEqual(json.loads(out)["addedHashes"], [wgpk.format_shader_hash(h2)])

                code, out, err = _run_cli(
                    ["inspect", str(pack_a), "--diff-dump", str(dump_dir), "--json"]
                )
                self.assertEqual(code, 0, msg=err)
                self.assertEqual(json.loads(out)["pack_version"], 2)

    def test_reflect_sidecar_versions_1_2_3_supported_and_invalid_versions_rejected(self) -> None:
        h1 = 0x00000000CAFE0001
        h2 = 0x00000000CAFE0002
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for ver in (1, 2, 3):
                reflect_path = root / f"reflect_v{ver}.json"
                reflect_path.write_text(
                    json.dumps(
                        {
                            "schema": wgpk.REFLECT_SIDECAR_SCHEMA,
                            "version": ver,
                            "stages": {
                                wgpk.format_shader_hash(h1): {
                                    "stage": "vertex",
                                    "bvh_ops": 0,
                                    "raw_address": False,
                                },
                                wgpk.format_shader_hash(h2): {
                                    "stage": "compute",
                                    "bvh_ops": 2 if ver >= 3 else 0,
                                    "raw_address": ver >= 3,
                                },
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                summary = wgpk.load_reflect_sidecar(reflect_path)
                self.assertEqual(summary.version, ver)
                self.assertEqual(summary.stage_hashes, frozenset({h1, h2}))

            for bad_ver in (0, 4, 99, True):
                bad_path = root / f"reflect_bad_{bad_ver}.json"
                bad_path.write_text(
                    json.dumps(
                        {
                            "schema": wgpk.REFLECT_SIDECAR_SCHEMA,
                            "version": bad_ver,
                            "stages": {},
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(wgpk.DumpDiffError) as ctx:
                    wgpk.load_reflect_sidecar(bad_path)
                self.assertEqual(ctx.exception.code, "invalid_reflect_sidecar")

    def test_diff_dump_native_vs_dxil_only_breakdown_v2_pack_and_v3_reflect_sanitized(self) -> None:
        sensitive = "secret_host_user_t104_dump"
        secret_wgsl = "SECRET_WGSL_T104_DO_NOT_LEAK"
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / sensitive / "captures"
            nested.mkdir(parents=True)

            h_native_ok = 0x0000000000001001
            h_native_and_dxil_missing = 0x0000000000001002  # present in both native & dxil -> native
            h_native_only_missing = 0x0000000000001003
            h_dxil_only_ok = 0x0000000000002001
            h_dxil_only_miss_1 = 0x0000000000002002
            h_dxil_only_miss_2 = 0x0000000000002003

            pack_path = nested / "scene_3d.wgsl.pack"
            pack_path.write_bytes(
                wgpk.build_wgpk(
                    [
                        (h_native_ok, "vertex", f"// {secret_wgsl}\n" + MINIMAL_VERTEX_WGSL),
                        (h_dxil_only_ok, "fragment", MINIMAL_FRAGMENT_WGSL),
                    ],
                    version=2,
                )
            )
            reflect_path = Path(f"{pack_path}.reflect.json")
            reflect_path.write_text(
                json.dumps(
                    {
                        "schema": wgpk.REFLECT_SIDECAR_SCHEMA,
                        "version": 3,
                        "stages": {
                            wgpk.format_shader_hash(h_native_ok): {"stage": "vertex"},
                            wgpk.format_shader_hash(h_dxil_only_ok): {"stage": "fragment"},
                        },
                    }
                ),
                encoding="utf-8",
            )

            dump_dir = nested / "pipeline_dump_3d"
            dump_dir.mkdir()
            (dump_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": wgpk.PIPELINE_DUMP_SCHEMA,
                        "schema_version": 1,
                        "native_stages": [
                            {"code_identity": wgpk.format_shader_hash(h_native_ok), "stage": "vertex"},
                            {
                                "code_identity": wgpk.format_shader_hash(h_native_and_dxil_missing),
                                "stage": "pixel",
                            },
                            {
                                "code_identity": wgpk.format_shader_hash(h_native_only_missing),
                                "stage": "compute",
                            },
                        ],
                        "dxil": [
                            {
                                "identity": wgpk.format_shader_hash(h_native_and_dxil_missing),
                                "kind": 0,
                            },
                            {"identity": wgpk.format_shader_hash(h_dxil_only_ok), "kind": 0},
                            {"identity": wgpk.format_shader_hash(h_dxil_only_miss_1), "kind": 1},
                            {"identity": wgpk.format_shader_hash(h_dxil_only_miss_2), "kind": 5},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            # Test CLI JSON with max-samples=1 to verify per-origin capping & truncation flags.
            code, out_json, err = _run_cli(
                [
                    "inspect",
                    str(pack_path),
                    "--diff-dump",
                    str(dump_dir),
                    "--max-samples",
                    "1",
                    "--json",
                ]
            )
            self.assertEqual(code, 0, msg=err)
            payload = json.loads(out_json)
            self.assertEqual(payload["pack_version"], 2)
            self.assertEqual(payload["reflect_version"], 3)
            self.assertEqual(payload["dump_total_stages"], 6)
            self.assertEqual(payload["in_pack"], 2)
            self.assertEqual(payload["missing_from_pack"], 4)
            self.assertEqual(payload["missing_native_from_pack"], 2)
            self.assertEqual(payload["missing_dxil_from_pack"], 2)
            self.assertEqual(payload["missing_dxil_only_from_pack"], 2)
            self.assertEqual(payload["missing_native_from_reflect"], 2)
            self.assertEqual(payload["missing_dxil_from_reflect"], 2)
            self.assertEqual(payload["missing_dxil_only_from_reflect"], 2)
            self.assertEqual(
                payload["sample_missing_native_pack_hashes"],
                [wgpk.format_shader_hash(h_native_and_dxil_missing)],
            )
            self.assertEqual(
                payload["sample_missing_dxil_pack_hashes"],
                [wgpk.format_shader_hash(h_dxil_only_miss_1)],
            )
            self.assertTrue(payload["missing_native_pack_hashes_truncated"])
            self.assertTrue(payload["missing_dxil_pack_hashes_truncated"])
            for forbidden in (sensitive, tmpdir, secret_wgsl):
                self.assertNotIn(forbidden, out_json)

            # Test CLI text output highlights WGPK v2 and native vs DXIL-only missing shaders.
            code, text_out, err = _run_cli(
                ["inspect", str(pack_path), "--diff-dump", str(dump_dir), "--max-samples", "5"]
            )
            self.assertEqual(code, 0, msg=err)
            self.assertIn("Pack format:           WGPK v2", text_out)
            self.assertIn("v3, 2 stages", text_out)
            self.assertIn("Missing from pack:     4 (native=2, dxil_only=2)", text_out)
            self.assertIn("Sample missing native (RDNA2) from pack (2 of 2):", text_out)
            self.assertIn("Sample missing DXIL-only from pack (2 of 2):", text_out)
            self.assertIn(wgpk.format_shader_hash(h_native_only_missing), text_out)
            self.assertIn(wgpk.format_shader_hash(h_dxil_only_miss_2), text_out)
            for forbidden in (sensitive, tmpdir, secret_wgsl):
                self.assertNotIn(forbidden, text_out)


if __name__ == "__main__":
    unittest.main()
