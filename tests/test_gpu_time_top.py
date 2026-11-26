"""Tests for tools/gpu_time_top.py (synthetic boot reports only)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import gpu_time_top as g  # noqa: E402


class GpuTimeTop(unittest.TestCase):
    def test_missing_section(self):
        s = g.summarize({})
        self.assertFalse(s["available"])
        self.assertIn("no gpu_time", g.render(s))

    def test_unavailable_reports_reason_not_numbers(self):
        s = g.summarize({"gpu_time": {"available": False, "support": "unavailable", "reason": "no timestamp-query"}})
        self.assertEqual(g.render(s), "GPU time unavailable: no timestamp-query\n")

    def test_top_sorted_and_share(self):
        rep = {"gpu_time": {"available": True, "total_ns": 2_000_000, "frames_measured": 3, "other": {"passes": 0, "total_ns": 0},
                            "top": [{"hash": "0xa", "kind": "render", "passes": 2, "total_ns": 500_000, "max_ns": 400_000},
                                    {"hash": "0xb", "kind": "compute", "passes": 1, "total_ns": 1_500_000, "max_ns": 1_500_000}]}}
        s = g.summarize(rep, top=1)
        self.assertEqual([r["hash"] for r in s["rows"]], ["0xb"])
        self.assertAlmostEqual(s["rows"][0]["share"], 0.75)
        self.assertAlmostEqual(s["rows"][0]["total_ms"], 1.5)
        self.assertIn("0xb", g.render(s))

    def test_phase_deltas(self):
        def snap(t_ms, presents, total, a_ns, a_n, b_ns, b_n):
            return {"t_ms": t_ms, "presents": presents, "gpu_time": {"available": True, "total_ns": total, "top": [
                {"hash": "0xa", "kind": "render", "passes": a_n, "total_ns": a_ns},
                {"hash": "0xb", "kind": "compute", "passes": b_n, "total_ns": b_ns}]}}
        rows = g.load_log("\n".join(__import__("json").dumps(d) for d in [
            snap(1000, 10, 1_000_000, 600_000, 6, 400_000, 4),
            snap(2000, 30, 5_000_000, 1_600_000, 16, 3_400_000, 24)]) + "\n{broken")
        p = g.phase(rows, 1, 2, top=5)
        self.assertEqual(p["presents"], 20)
        self.assertAlmostEqual(p["total_ms"], 4.0)
        self.assertAlmostEqual(p["gpu_ms_per_frame"], 0.2)
        self.assertEqual([r["hash"] for r in p["rows"]], ["0xb", "0xa"])
        self.assertAlmostEqual(p["rows"][0]["share"], 0.75)
        self.assertTrue(p["dominant"])
        self.assertIn("one shader", g.render_phase("x", p))
        self.assertFalse(g.phase(rows, 0.5, 0.9)["available"])


if __name__ == "__main__":
    unittest.main()
