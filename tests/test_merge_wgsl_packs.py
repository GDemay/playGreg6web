import contextlib
import io
import json
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


def stage(n_resolved: int, n_unresolved: int = 0):
    b = [{"group": 0, "binding": i, "kind": "uniform", "source": {"user_sgpr": 8}} for i in range(n_resolved)]
    b += [{"group": 0, "binding": 10 + i, "kind": "uniform"} for i in range(n_unresolved)]
    return {"stage": "vertex", "bindings": b}


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
                {"0x0000000000000001": stage(1), "0x0000000000000002": stage(0, 1)},
                version=1,
                reflect_version=2,
            )
            b = write_pack(
                d,
                "delta_v2.wgsl.pack",
                [(2, "fragment", "// ps2_new"), (3, "compute", "// cs3")],
                {"0x0000000000000002": stage(2), "0x0000000000000003": stage(1)},
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


if __name__ == "__main__":
    unittest.main()
