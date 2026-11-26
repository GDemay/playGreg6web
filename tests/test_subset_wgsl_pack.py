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
import subset_wgsl_pack as sp  # noqa: E402


class SubsetTest(unittest.TestCase):
    def test_v2_to_v1_cap_budget_priority(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            ents = [(h, 1, b"x" * n) for h, n in ((1, 10), (2, 20), (3, 30), (4, 5000))]
            (d / "a.wgsl.pack").write_bytes(wgpk.build_wgpk(ents, version=2))
            (d / "a.wgsl.pack.reflect.json").write_text(json.dumps({"schema": "s", "version": 3, "stages": {f"0x{h:016x}": {} for h in (1, 2, 3, 4)}}))
            self.assertEqual(sp.main([str(d / "a.wgsl.pack"), "--out", str(d / "o.wgsl.pack"), "--max-wgsl-bytes", "100", "--max-total-bytes", "40", "--priority-hashes", str(d / "p.txt")] if (d / "p.txt").write_text("0x0000000000000003") else []), 0)
            toc = sp.read_toc(d / "o.wgsl.pack")
            self.assertEqual({e[0] for e in toc}, {1, 3})  # 3 is priority (30), then 1 (10); 2 would exceed 40
            self.assertEqual(wgpk.build_wgpk([(1, 1, b"x" * 10), (3, 1, b"x" * 30)], version=1), (d / "o.wgsl.pack").read_bytes())
            self.assertEqual(set(json.loads((d / "o.wgsl.pack.reflect.json").read_text())["stages"]), {f"0x{h:016x}" for h in (1, 3)})

    def test_priority_stages_kept_respects_budget_and_minify_and_parent_dir(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            vs1 = "    // comment\n    @vertex fn vs1() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            vs2 = "    // comment 2\n    @vertex fn vs2() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }\n"
            (d / "src.wgsl.pack").write_bytes(wgpk.build_wgpk([(1, "vertex", vs1), (2, "vertex", vs2)], version=2))
            # Unpadded uppercase key to exercise robust hex normalization.
            (d / "src.wgsl.pack.reflect.json").write_text(
                json.dumps({"schema": "playgta6-stage-reflection", "version": 3, "stages": {"0x1": {"stage": "vertex", "bindings": []}, "0x2": {"stage": "vertex", "bindings": []}}})
            )
            (d / "pri.txt").write_text("0x0000000000000001\n0x0000000000000002\n", encoding="utf-8")

            out = d / "nested" / "sub.wgsl.pack"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = sp.main([
                    str(d / "src.wgsl.pack"),
                    "--out",
                    str(out),
                    "--max-wgsl-bytes",
                    "500",
                    "--max-total-bytes",
                    str(len(vs1) + 5),  # only 1 of the 2 priority stages fits!
                    "--priority-hashes",
                    str(d / "pri.txt"),
                    "--minify-wgsl",
                    "--json",
                ])
            self.assertEqual(rc, 0)
            rep = json.loads(stdout.getvalue())
            self.assertEqual(rep["stages"], 1)
            self.assertEqual(rep["priority_stages_kept"], 1)
            self.assertEqual(rep["minify_mode"], "compact")
            self.assertLess(rep["payload_bytes"], rep["raw_payload_bytes"])
            self.assertEqual(rep["reflect_stages"], 1)

            bundle = wgpk.parse_wgpk(out.read_bytes())
            self.assertEqual(bundle.version, 1)
            self.assertEqual(len(bundle.entries), 1)
            self.assertEqual(bundle.entries[0].wgsl, mp.minify_wgsl(vs1, mode="compact"))
            side = json.loads((out.parent / "sub.wgsl.pack.reflect.json").read_text(encoding="utf-8"))
            self.assertEqual(list(side["stages"].keys()), ["0x0000000000000001"])

    def test_read_toc_rejects_corrupt_header_and_out_of_bounds_entry(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            bad = d / "bad.wgsl.pack"
            bad.write_bytes(b"NOPE")
            with self.assertRaises(ValueError):
                sp.read_toc(bad)

            valid = bytearray(wgpk.build_wgpk([(1, "vertex", "fn v() {}")], version=1))
            struct.pack_into("<I", valid, 16 + 12, 4)  # offset inside header
            oob = d / "oob.wgsl.pack"
            oob.write_bytes(bytes(valid))
            with self.assertRaises(ValueError):
                sp.read_toc(oob)


if __name__ == "__main__":
    unittest.main()
