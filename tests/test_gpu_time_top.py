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


if __name__ == "__main__":
    unittest.main()
