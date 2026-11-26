import contextlib
import io
import json
import re
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import inspect_wgsl_pack as wgpk  # noqa: E402
import merge_wgsl_packs as mp  # noqa: E402


def write_pack(d: Path, name: str, entries, stages=None, version=None, reflect_version=3) -> Path:
    p = d / name
    p.write_bytes(wgpk.build_wgpk(entries, version=version))
    if stages is not None:
        (d / (name + ".reflect.json")).write_text(
            json.dumps({"schema": "playgta6-stage-reflection", "version": reflect_version, "stages": stages})
        )
    return p


def stage(n_resolved: int, n_unresolved: int = 0, stage_name: str = "vertex"):
    b = [{"group": 0, "binding": i, "kind": "uniform", "source": {"user_sgpr": 8}} for i in range(n_resolved)]
    b += [{"group": 0, "binding": 10 + i, "kind": "uniform"} for i in range(n_unresolved)]
    return {"stage": stage_name, "bindings": b}


SYNTHETIC_RICH_WGSL = """\
// Top-level file banner comment
enable f16;
diagnostic(off, derivative_uniformity);

/* Outer block comment
   /* Nested block comment with -> >> >= <= // fake line comment */
   still inside outer comment */
struct Uniforms {
    color : vec4<f32>,
    counts : vec4<u32>,
};

override block_size : u32 = 64u;
const SCALE_VAL : f32 = 1.0e-5f;
const HEX_FLOAT : f16 = 0x1.fp+2h;
const HEX_MASK : u32 = 0x00ffu;

@group(0) @binding(0) var<uniform> cb0 : Uniforms;
@group(0) @binding(1) var<storage, read_write> out_buf : array<vec4<f32>>;

fn helper(a : i32, b : i32, ptr : ptr<function, i32>) -> i32 {
    let/*comment_between_tokens*/x : i32 = a - -b;
    let y : i32 = a + +b;
    let z : i32 = a / *ptr;
    let shifted : u32 = (u32(x) >> 2u) << 1u;
    var acc : i32 = x + y + z + i32(shifted);
    if (acc >= 0 && acc <= 100 || acc != -1) {
        acc += 1;
        acc -= 2;
        acc *= 3;
        acc /= 1;
        acc %= 7;
        acc &= 0xff;
        acc |= 1;
        acc ^= 2;
    } else if (acc == -1) {
        acc = 0;
    }
    return acc;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid : vec3<u32>) {
    var local_v : i32 = 1;
    let nested : array<vec4<f32>, 2> = array<vec4<f32>, 2>(cb0.color, vec4<f32>(.5f, 1.0f, SCALE_VAL, f32(HEX_FLOAT)));
    let h = helper(i32(gid.x), 4, &local_v);
    out_buf[gid.x] = nested[0] + vec4<f32>(f32(h), f32(block_size & HEX_MASK), 0.0, 1.0); // trailing comment
}
"""


class MergeTests(unittest.TestCase):
    def test_union_prefers_sidecar_and_resolved(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(3, "vertex", "// a3"), (1, "vertex", "// a1")],
                           {"0x0000000000000003": stage(0, 1), "0x0000000000000001": stage(1)})
            b = write_pack(d, "b.wgsl.pack", [(3, "vertex", "// b3"), (2, "fragment", "// b2"), (1, "vertex", "// b1")],
                           {"0x0000000000000003": stage(2)})
            out = d / "out" / "m.wgsl.pack"
            s = mp.merge([a, b], out)
            self.assertEqual(s["merged_entries"], 3)
            self.assertEqual(s["duplicates_dropped"], 2)
            bundle = wgpk.parse_wgpk(out.read_bytes())
            self.assertEqual([e.hash for e in bundle.entries], [1, 2, 3])
            texts = {e.hash: e.wgsl for e in bundle.entries}
            self.assertEqual(texts, {1: "// a1", 2: "// b2", 3: "// b3"})
            side = json.loads((out.parent / "m.wgsl.pack.reflect.json").read_text())
            self.assertEqual(sorted(side["stages"]), ["0x0000000000000001", "0x0000000000000003"])
            self.assertEqual(len(side["stages"]["0x0000000000000003"]["bindings"]), 2)

    def test_deterministic_and_tie_goes_to_first(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(5, "compute", "// A")])
            b = write_pack(d, "b.wgsl.pack", [(5, "compute", "// B")])
            o1, o2 = d / "o1.wgsl.pack", d / "o2.wgsl.pack"
            mp.merge([a, b], o1)
            mp.merge([a, b], o2)
            self.assertEqual(o1.read_bytes(), o2.read_bytes())
            self.assertEqual(wgpk.parse_wgpk(o1.read_bytes()).entries[0].wgsl, "// A")
            self.assertFalse((d / "o1.wgsl.pack.reflect.json").exists())

    def test_reads_v2_input_and_cli(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(9, "vertex", "// v2")], {"0x0000000000000009": stage(1)}, version=2)
            out = d / "m.wgsl.pack"
            self.assertEqual(mp.main([str(a), "--out", str(out)]), 0)
            bundle = wgpk.parse_wgpk(out.read_bytes())
            self.assertEqual(bundle.version, 1)
            self.assertEqual(bundle.entries[0].wgsl, "// v2")
            self.assertEqual(mp.main([str(a), "--out", str(a)]), 2)

    def test_v2_output_when_over_u32(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(1, "vertex", "// x" * 10)])
            old = mp.U32_MAX
            mp.U32_MAX = 40
            try:
                out = d / "m.wgsl.pack"
                s = mp.merge([a], out)
            finally:
                mp.U32_MAX = old
            self.assertEqual(s["pack_version"], 2)
            self.assertEqual(wgpk.parse_wgpk(out.read_bytes()).entries[0].wgsl, "// x" * 10)

    def test_force_v2_and_verify_flags_and_mixed_v1_v2_sidecar_versions(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(
                d,
                "base_v1.wgsl.pack",
                [(1, "vertex", "// vs1"), (2, "fragment", "// ps2_old")],
                {
                    "0x0000000000000001": stage(1, stage_name="vertex"),
                    "0x0000000000000002": stage(0, 1, stage_name="fragment"),
                },
                version=1,
                reflect_version=2,
            )
            b = write_pack(
                d,
                "delta_v2.wgsl.pack",
                [(2, "fragment", "// ps2_new"), (3, "compute", "// cs3")],
                {
                    "0x0000000000000002": stage(2, stage_name="fragment"),
                    "0x0000000000000003": stage(1, stage_name="compute"),
                },
                version=2,
                reflect_version=3,
            )
            out = d / "merged_v2.wgsl.pack"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = mp.main([str(a), str(b), "--out", str(out), "--force-v2", "--verify", "--json"])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["pack_version"], 2)
            self.assertEqual(summary["merged_entries"], 3)
            self.assertEqual(summary["duplicates_dropped"], 1)
            self.assertTrue(summary["verified"])
            self.assertEqual(summary["sidecar_verification"]["matched_hashes"], 3)
            self.assertEqual(summary["sidecar_verification"]["stage_mismatches"], 0)
            self.assertEqual(summary["sidecar_verification"]["coverage_pct"], 100.0)

            bundle = wgpk.parse_wgpk(out.read_bytes())
            self.assertEqual(bundle.version, 2)
            self.assertEqual({e.hash: e.wgsl for e in bundle.entries}, {1: "// vs1", 2: "// ps2_new", 3: "// cs3"})
            side = json.loads((d / "merged_v2.wgsl.pack.reflect.json").read_text(encoding="utf-8"))
            self.assertEqual(side["version"], 3)

            # Standalone verify_merged_pack succeeds on clean pack and fails on tampered payload/sidecar.
            v_info = mp.verify_merged_pack(out, [a, b])
            self.assertTrue(v_info["verified"])
            self.assertEqual(v_info["entries_verified"], 3)

            tampered = bytearray(out.read_bytes())
            tampered[-1] = ord("X") if tampered[-1] != ord("X") else ord("Y")
            out.write_bytes(bytes(tampered))
            with self.assertRaises(ValueError):
                mp.verify_merged_pack(out, [a, b])

    def test_read_toc_rejects_offset_before_data_start_and_overlapping_ranges(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            valid = wgpk.build_wgpk([(1, "vertex", "// one_payload"), (2, "fragment", "// two_payload")])

            # Offset pointing into header/TOC (< data_start)
            before_start = bytearray(valid)
            struct.pack_into("<I", before_start, 16 + 12, 8)
            p1 = d / "before_start.wgsl.pack"
            p1.write_bytes(bytes(before_start))
            with self.assertRaises(ValueError) as ctx1:
                mp.read_toc(p1)
            self.assertIn("exceeds pack bounds", str(ctx1.exception))
            self.assertNotIn(str(d), str(ctx1.exception))

            # Overlapping non-empty payload ranges
            overlap = bytearray(valid)
            e0_offset = struct.unpack_from("<I", overlap, 16 + 12)[0]
            struct.pack_into("<I", overlap, 16 + 20 + 12, e0_offset + 2)
            struct.pack_into("<I", overlap, 16 + 20 + 16, 6)
            p2 = d / "overlap.wgsl.pack"
            p2.write_bytes(bytes(overlap))
            with self.assertRaises(ValueError) as ctx2:
                mp.read_toc(p2)
            self.assertIn("overlapping payload ranges", str(ctx2.exception))
            self.assertNotIn(str(d), str(ctx2.exception))

    def test_sanitized_json_and_error_output_never_leak_host_paths(self):
        secret_segment = "secret_host_user_merge_7741"
        with tempfile.TemporaryDirectory() as t:
            nested = Path(t) / secret_segment / "private_packs"
            nested.mkdir(parents=True)
            a = write_pack(nested, "a.wgsl.pack", [(1, "vertex", "// a")])
            out = nested / "m.wgsl.pack"

            # Success JSON output contains basenames only.
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = mp.main([str(a), "--out", str(out), "--force-v2", "--verify", "--json"])
            self.assertEqual(rc, 0)
            out_text = stdout.getvalue() + stderr.getvalue()
            self.assertNotIn(secret_segment, out_text)
            self.assertNotIn("private_packs", out_text)
            self.assertNotIn(str(t), out_text)
            self.assertEqual(json.loads(stdout.getvalue())["inputs"], ["a.wgsl.pack"])

            # Missing input file error output does not leak host path.
            missing = nested / "missing_input.wgsl.pack"
            stdout2 = io.StringIO()
            stderr2 = io.StringIO()
            with contextlib.redirect_stdout(stdout2), contextlib.redirect_stderr(stderr2):
                rc2 = mp.main([str(missing), "--out", str(out)])
            self.assertEqual(rc2, 1)
            err_text = stdout2.getvalue() + stderr2.getvalue()
            self.assertIn("missing_input.wgsl.pack", err_text)
            self.assertNotIn(secret_segment, err_text)
            self.assertNotIn("private_packs", err_text)
            self.assertNotIn(str(t), err_text)

    def test_verify_detects_sidecar_stage_mismatch_via_stage_reflection_sidecar(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            # Pack says fragment (2), sidecar says vertex -> verify must reject.
            a = write_pack(
                d,
                "mismatch.wgsl.pack",
                [(0x11, "fragment", "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }")],
                {"0x0000000000000011": stage(1, stage_name="vertex")},
                version=2,
            )
            out = d / "merged.wgsl.pack"
            with self.assertRaises(ValueError) as ctx:
                mp.merge([a], out, force_v2=True, verify=True)
            self.assertIn("stage mismatch", str(ctx.exception))
            # Atomic commit: failed verify must not leave a broken output file behind.
            self.assertFalse(out.exists())
            self.assertFalse((d / "merged.wgsl.pack.reflect.json").exists())

    def test_in_place_incremental_merge_and_stale_sidecar_cleanup(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            base = write_pack(
                d,
                "splash.wgsl.pack",
                [
                    (0x10, "vertex", "    @vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"),
                    (0x20, "fragment", "    @fragment fn fs_old() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n"),
                ],
                {
                    "0x0000000000000010": stage(1, stage_name="vertex"),
                    "0x0000000000000020": stage(0, 1, stage_name="fragment"),
                },
                version=2,
            )
            delta = write_pack(
                d,
                "delta.wgsl.pack",
                [
                    (0x20, "fragment", "    // better\n    @fragment fn fs_new() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n"),
                    (0x30, "compute", "    @compute @workgroup_size(64, 1, 1) fn cs_new() {}\n"),
                ],
                {
                    "0x0000000000000020": stage(2, stage_name="fragment"),
                    "0x0000000000000030": stage(1, stage_name="compute"),
                },
                version=2,
            )

            # Without --in-place, overwriting an input pack exits 2.
            self.assertEqual(mp.main([str(delta), str(base), "--out", str(base)]), 2)

            # With --in-place --force-v2 --minify-wgsl --verify, base is atomically updated & verified.
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = mp.main(
                    [str(delta), str(base), "--out", str(base), "--in-place", "--force-v2", "--minify-wgsl", "--verify", "--json"]
                )
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            self.assertTrue(summary["verified"])
            self.assertEqual(summary["merged_entries"], 3)
            self.assertEqual(summary["pack_version"], 2)
            self.assertGreater(summary["bytes_saved"], 0)
            bundle = wgpk.parse_wgpk(base.read_bytes())
            self.assertEqual([e.hash for e in bundle.entries], [0x10, 0x20, 0x30])
            self.assertIn("fs_new", bundle.entries[1].wgsl)
            self.assertNotIn("// better", bundle.entries[1].wgsl)

            # Merging packs without sidecars into an output that had a stale sidecar deletes the stale sidecar.
            no_sc = write_pack(d, "nosidecar.wgsl.pack", [(0x40, "vertex", "@vertex fn v() {}")])
            stale_out = d / "stale.wgsl.pack"
            stale_side = d / "stale.wgsl.pack.reflect.json"
            stale_side.write_text(json.dumps({"schema": "playgta6-stage-reflection", "version": 3, "stages": {}}))
            mp.merge([no_sc], stale_out, verify=True)
            self.assertFalse(stale_side.exists())

    def test_cross_input_stage_conflict_reporting_and_fail_flag(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(d, "a.wgsl.pack", [(0x99, "vertex", "@vertex fn v() {}")])
            b = write_pack(d, "b.wgsl.pack", [(0x99, "fragment", "@fragment fn f() {}")])
            out = d / "out.wgsl.pack"

            # Default mode reports stage_conflicts cleanly without dropping diagnostic visibility.
            summary = mp.merge([a, b], out, force_v2=True, verify=True)
            self.assertEqual(summary["stage_conflicts"], 1)
            self.assertEqual(summary["stage_conflict_samples"][0]["hash"], "0x0000000000000099")
            self.assertEqual(summary["stage_conflict_samples"][0]["winner_stage"], "vertex")
            self.assertEqual(summary["stage_conflict_samples"][0]["seen_stages"], ["fragment", "vertex"])

            # --fail-on-stage-conflict rejects cross-input stage conflict.
            out2 = d / "out2.wgsl.pack"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = mp.main([str(a), str(b), "--out", str(out2), "--fail-on-stage-conflict"])
            self.assertEqual(rc, 1)
            self.assertIn("stage conflict", stderr.getvalue())
            self.assertFalse(out2.exists())


class WgslMinifyTests(unittest.TestCase):
    def test_minify_preserves_tokens_idempotence_and_downstream_parsers(self):
        src_tokens = mp.tokenize_wgsl(SYNTHETIC_RICH_WGSL)
        # Regex mirrors of downstream consumers in web/workers/render_state_packets.js & draw_executor.js
        group0_var_re = re.compile(
            r"@group\(\s*0\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var(?:\s*<\s*([^>]+)\s*>)?\s+([A-Za-z_][A-Za-z0-9_]*)"
        )
        draw_exec_u0_re = re.compile(r"@group\(0\)\s+@binding\(0\)\s+var<uniform>")
        compute_entry_re = re.compile(r"@compute\s+@workgroup_size\(\s*64\s*,\s*1\s*,\s*1\s*\)\s+fn\s+cs_main")

        prev_len = len(SYNTHETIC_RICH_WGSL)
        for mode in ("line", "compact", "aggressive"):
            out = mp.minify_wgsl(SYNTHETIC_RICH_WGSL, mode=mode)
            # Token-stream equivalence
            self.assertEqual(mp.tokenize_wgsl(out), src_tokens, msg=f"token mismatch in mode={mode}")
            # Idempotence
            self.assertEqual(mp.minify_wgsl(out, mode=mode), out, msg=f"non-idempotent in mode={mode}")
            # Strictly smaller than input and monotonic across line >= compact >= aggressive
            self.assertLessEqual(len(out), prev_len, msg=f"mode={mode} grew payload")
            prev_len = len(out)
            # Comments stripped
            self.assertNotIn("banner comment", out)
            self.assertNotIn("Nested block comment", out)
            self.assertNotIn("trailing comment", out)
            # Hazardous operator boundaries preserved
            self.assertNotIn("a--b", out)
            self.assertNotIn("a++b", out)
            self.assertNotIn("a/*ptr", out)
            self.assertNotIn("letx", out)
            # Downstream parser regex compatibility preserved!
            matches = group0_var_re.findall(out)
            self.assertEqual([(m[0], m[2]) for m in matches], [("0", "cb0"), ("1", "out_buf")])
            self.assertIsNotNone(draw_exec_u0_re.search(out), msg=f"draw_executor regex broke in mode={mode}")
            self.assertIsNotNone(compute_entry_re.search(out), msg=f"compute entry regex broke in mode={mode}")
            # Directives keep trailing newline even in aggressive mode
            self.assertIn("enable f16;\n", out)
            self.assertRegex(out, r"diagnostic\(off,\s*derivative_uniformity\);\n")

    def test_minify_edge_cases_comments_unicode_linebreaks_operators_and_templates(self):
        # Nested block comments, CRLF, Unicode LS/PS/NEL, adjacent > >, trailing-dot float, division-deref.
        src = (
            "let a = 1.0f; // crlf comment\r\n"
            "let b = 2.0f; // ls comment\u2028"
            "let c = 3.0f; // ps comment\u2029"
            "let d = /* l1 /* l2 */ l1 */ 4u;\n"
            "let e = a / /* split */ *p;\n"
            "let f = array<vec4<f32> >();\n"
            "let g = array<vec4<f32>>();\n"
            "let h = 1. + 2.;\n"
        )
        expected_tokens = mp.tokenize_wgsl(src)
        for mode in mp.WGSL_MINIFY_MODES:
            out = mp.minify_wgsl(src, mode=mode)
            self.assertEqual(mp.tokenize_wgsl(out), expected_tokens)
            self.assertEqual(mp.minify_wgsl(out, mode=mode), out)
            self.assertNotIn("comment", out)
            self.assertNotIn("/*", out)
            self.assertNotIn("*/", out)
            self.assertIn("> >", out)
            self.assertIn(">>()", out)

        # Unterminated nested block comment must raise ValueError rather than silently swallowing code.
        with self.assertRaises(ValueError):
            mp.minify_wgsl("let x = 1; /* outer /* inner */ still open")

        # Invalid mode rejected.
        with self.assertRaises(ValueError):
            mp.minify_wgsl("fn f() {}", mode="turbo")

    def test_merge_minify_wgsl_preserves_hashes_and_auto_downgrades_v2_to_v1(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            padded_vs = ("        // heavy indentation\n" + "                " + "@vertex fn vs_main() -> @builtin(position) vec4<f32> {\n" + "                        return vec4<f32>(0.0, 0.0, 0.0, 1.0);\n" + "                }\n")
            padded_ps = ("        // pixel shader\n" + "                " + "@fragment fn fs_main() -> @location(0) vec4<f32> {\n" + "                        return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n" + "                }\n")
            base = write_pack(
                d,
                "base_v2.wgsl.pack",
                [(0x101, "vertex", padded_vs)],
                {"0x0000000000000101": stage(1, stage_name="vertex")},
                version=2,
            )
            delta = write_pack(
                d,
                "delta_v2.wgsl.pack",
                [(0x202, "fragment", padded_ps)],
                {"0x0000000000000202": stage(1, stage_name="fragment")},
                version=2,
            )

            # Force raw total > U32_MAX, while minified total fits <= U32_MAX: exercises temp payload spool + v2->v1 auto-downgrade!
            raw_v1_size = mp.HEADER.size + 2 * mp.ENTRY_V1.size + len(padded_vs) + len(padded_ps)
            min_v1_size = (
                mp.HEADER.size
                + 2 * mp.ENTRY_V1.size
                + len(mp.minify_wgsl(padded_vs, mode="compact").encode("utf-8"))
                + len(mp.minify_wgsl(padded_ps, mode="compact").encode("utf-8"))
            )
            self.assertGreater(raw_v1_size, min_v1_size)

            old_u32 = mp.U32_MAX
            mp.U32_MAX = min_v1_size + 10
            try:
                out = d / "compacted.wgsl.pack"
                summary = mp.merge([base, delta], out, minify_wgsl_mode="compact", verify=True)
            finally:
                mp.U32_MAX = old_u32

            self.assertEqual(summary["pack_version"], 1)
            self.assertTrue(summary["verified"])
            self.assertEqual(summary["minify_mode"], "compact")
            self.assertGreater(summary["bytes_saved"], 0)
            self.assertGreater(summary["reduction_pct"], 20.0)
            bundle = wgpk.parse_wgpk(out.read_bytes())
            self.assertEqual(bundle.version, 1)
            self.assertEqual([e.hash for e in bundle.entries], [0x101, 0x202])
            self.assertEqual(bundle.entries[0].wgsl, mp.minify_wgsl(padded_vs, mode="compact"))
            self.assertEqual(bundle.entries[1].wgsl, mp.minify_wgsl(padded_ps, mode="compact"))

            # Tampering a minified byte fails verify_merged_pack(..., minify_mode="compact").
            raw = bytearray(out.read_bytes())
            raw[-1] = ord(" ")
            out.write_bytes(bytes(raw))
            with self.assertRaises(ValueError):
                mp.verify_merged_pack(out, [base, delta], minify_mode="compact")

    def test_merge_minify_cli_modes_and_force_v2(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            a = write_pack(
                d,
                "a.wgsl.pack",
                [(0xAA, "compute", SYNTHETIC_RICH_WGSL)],
                {"0x00000000000000aa": stage(2, stage_name="compute")},
                version=2,
            )
            out = d / "out_agg.wgsl.pack"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = mp.main([str(a), "--out", str(out), "--minify-mode", "aggressive", "--force-v2", "--verify", "--json"])
            self.assertEqual(rc, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["pack_version"], 2)
            self.assertEqual(summary["minify_mode"], "aggressive")
            self.assertTrue(summary["verified"])
            self.assertGreater(summary["reduction_pct"], 30.0)


class DiffIntegrationTests(unittest.TestCase):
    def test_diff_boot_log_and_pipeline_dump_then_incremental_merge_clears_missing(self):
        secret_dir = "secret_user_diff_9821"
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / secret_dir
            root.mkdir(parents=True)
            base_hash = 0x1111222233334444
            missing_ps = 0xE60F2F99BBB95E26
            missing_vs = 0xAC2DB5AB29DCC4E1

            base_pack = write_pack(
                root,
                "splash.wgsl.pack",
                [(base_hash, "vertex", "@vertex fn vs_base() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }")],
                {f"0x{base_hash:016x}": stage(1, stage_name="vertex")},
                version=2,
            )
            boot_log = root / "boot.log"
            boot_log.write_text(
                f"d3d12_xs: no shader module for DXIL identity 0x{missing_ps:016x} (pixel shader, DXIL kind 0; fallback)\n"
                f"d3d12_xs: no shader module for DXIL identity 0x{base_hash:016x} (vertex shader, DXIL kind 1; ok)\n",
                encoding="utf-8",
            )
            dump_dir = root / "pipeline_dump"
            dump_dir.mkdir()
            (dump_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "playgta6-pipeline-dump",
                        "schema_version": 1,
                        "native_stages": [
                            {"code_identity": f"0x{missing_vs:016x}", "stage": "vertex"},
                            {"code_identity": f"0x{missing_ps:016x}", "stage": "fragment"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            priority_out = root / "priority_missing.txt"

            # 1. Diff via `merge_wgsl_packs.py diff ... --strict --json` reports the 2 missing hashes and exits 1.
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = mp.main(
                    [
                        "diff",
                        "--pack",
                        str(base_pack),
                        "--boot-log",
                        str(boot_log),
                        "--dump-dir",
                        str(dump_dir),
                        "--emit-priority-hashes",
                        str(priority_out),
                        "--strict",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 1)
            combined = stdout.getvalue() + stderr.getvalue()
            self.assertNotIn(secret_dir, combined)
            self.assertNotIn(str(t), combined)

            report = json.loads(stdout.getvalue())
            self.assertFalse(report["ok"])
            self.assertFalse(report["clean"])
            missing_hashes = [item["hash"] for item in report["missing_from_pack"]]
            # `missing_from_pack` is ordered by descending demand count (`missing_ps` appears in both log & dump).
            self.assertEqual(missing_hashes, [f"0x{missing_ps:016x}", f"0x{missing_vs:016x}"])
            self.assertEqual(
                priority_out.read_text(encoding="utf-8").strip().splitlines(),
                sorted([f"0x{missing_vs:016x}", f"0x{missing_ps:016x}"]),
            )

            # 2. Incrementally merge delta pack with `--minify-wgsl --force-v2 --verify --in-place`.
            delta_pack = write_pack(
                root,
                "delta.wgsl.pack",
                [
                    (missing_vs, "vertex", "    // new vs\n    @vertex fn vs_new() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"),
                    (missing_ps, "fragment", "    // new ps\n    @fragment fn fs_new() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n"),
                ],
                {
                    f"0x{missing_vs:016x}": stage(1, stage_name="vertex"),
                    f"0x{missing_ps:016x}": stage(1, stage_name="fragment"),
                },
                version=2,
            )
            rc_merge = mp.main(
                [str(delta_pack), str(base_pack), "--out", str(base_pack), "--in-place", "--force-v2", "--minify-wgsl", "--verify"]
            )
            self.assertEqual(rc_merge, 0)

            # 3. Re-diff via `--diff` flag now reports clean and exits 0 under `--strict`.
            stdout2 = io.StringIO()
            with contextlib.redirect_stdout(stdout2):
                rc2 = mp.main(
                    [
                        "--diff",
                        "--pack",
                        str(base_pack),
                        "--boot-log",
                        str(boot_log),
                        "--dump-dir",
                        str(dump_dir),
                        "--strict",
                        "--json",
                    ]
                )
            self.assertEqual(rc2, 0)
            report2 = json.loads(stdout2.getvalue())
            self.assertTrue(report2["ok"])
            self.assertTrue(report2["clean"])
            self.assertEqual(report2["missing_from_pack"], [])


if __name__ == "__main__":
    unittest.main()
