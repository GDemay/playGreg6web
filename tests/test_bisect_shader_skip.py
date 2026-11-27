import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import bisect_shader_skip as b  # noqa: E402


def run(suspects, culprit):
    st = b.init_state(suspects)
    for _ in range(20):
        skip = b.next_skip(st)
        if st["culprit"]:
            return st
        b.apply_report(st, culprit in skip)
    raise AssertionError("did not converge")


class BisectTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(b.parse_hashes("# c\n0x10 note\n10\nAB\n"), ["0x0000000000000010", "0x00000000000000ab"])
        with self.assertRaises(ValueError):
            b.parse_hashes("zz\n")
        with self.assertRaises(ValueError):
            b.parse_hashes("0\n")

    def test_finds_every_culprit(self):
        hs = [f"0x{i:016x}" for i in range(1, 12)]
        for c in hs:
            self.assertEqual(run(hs, c)["culprit"], c)

    def test_single_and_errors(self):
        self.assertEqual(run(["0x01"], "0x01")["culprit"], "0x01")
        with self.assertRaises(ValueError):
            b.init_state([])
        with self.assertRaises(ValueError):
            b.apply_report(b.init_state(["0x1", "0x2"]), True)


if __name__ == "__main__":
    unittest.main()
