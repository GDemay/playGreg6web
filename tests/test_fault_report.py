import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import fault_report as fr  # noqa: E402

SCRIPT = str(REPO_ROOT / "tools" / "fault_report.py")


def loc(entry, off, sym=None):
    return {"address": hex(entry + off), "entry": hex(entry), "offset": hex(off), "symbol": sym}


def report(kind="unknown_target"):
    rec = {
        "kind": kind,
        "code_miss": True,
        "thread": 3,
        "elapsed_ms": 4242,
        "rip": "0x140001234",
        "target": "0x140001234",
        "detail": "0x140001180",
        "present": {"presents": 2, "draws_per_frame": 17},
        "function": loc(0x140001180, 0xB4, {"module": "game.exe", "name": "Exported", "offset": "0x10"}),
        "source": {"return_address": "0x140001030", "location": loc(0x140001000, 0x30)},
        "registers": {"rax": "0x1", "rcx": "0x2", "rsp": "0x7ff00"},
        "call_stack": {"method": "rtl-virtual-unwind", "frames": [loc(0x140001000, 0x30)]},
        "recent_entries": [
            {"target": "0x140001180", "name": None, "from": "0x140001020"},
            {"target": "0x5047000000000010", "name": "KERNEL32.dll!Sleep", "from": "0x140001030"},
        ],
    }
    if kind == "unknown_resume_target":
        rec["source"] = {"resume_rip": "0x140002000", "location": loc(0x140001F00, 0x100)}
    if kind == "trap":
        rec["source"] = None
    return {
        "status": "Fault",
        "phase": "entry",
        "message": "boom",
        "fault_record": rec,
        "code_misses": {"unknown_target": {"count": 1, "targets": []}, "trap": {"count": 0, "targets": []}},
    }


class FaultReportTests(unittest.TestCase):
    def test_render_contains_core_facts(self):
        out = fr.render(report())
        for needle in (
            "FAULT unknown_target",
            "[recompiler gap]",
            "sub_140001180+0xb4",
            "game.exe!Exported+0x10",
            "draws/frame=17",
            "elapsed: 4242 ms",
            "KERNEL32.dll!Sleep",
            "return address 0x0000000140001030",
            "rtl-virtual-unwind",
            "rsp=",
            "code_misses in this run: unknown_target=1",
        ):
            self.assertIn(needle, out)

    def test_entries_limit(self):
        out = fr.render(report(), entries=1)
        self.assertIn("last 1 of 2", out)
        self.assertNotIn("0x0000000140001180  from", out)

    def test_suggestions_by_kind(self):
        s = fr.suggestions(report()["fault_record"])
        self.assertTrue(any("--entry 0x140001234" in x for x in s))
        self.assertTrue(any("--entry 0x140001000" in x for x in s))
        s = fr.suggestions(report("unknown_resume_target")["fault_record"])
        self.assertTrue(any("--entry 0x140002000" in x for x in s))
        s = fr.suggestions(report("trap")["fault_record"])
        self.assertTrue(any(" lift " in x and "--entry 0x140001180" in x for x in s))

    def test_no_record(self):
        self.assertEqual(fr.render({"status": "Exited"}), "")

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            ok = Path(d) / "ok.json"
            ok.write_text(json.dumps(report()), encoding="utf-8")
            none = Path(d) / "none.json"
            none.write_text("{}", encoding="utf-8")
            bad = Path(d) / "bad.json"
            bad.write_text("not json", encoding="utf-8")
            self.assertEqual(subprocess.run([sys.executable, SCRIPT, str(ok)], capture_output=True).returncode, 0)
            self.assertEqual(subprocess.run([sys.executable, SCRIPT, str(none)], capture_output=True).returncode, 1)
            self.assertEqual(subprocess.run([sys.executable, SCRIPT, str(bad)], capture_output=True).returncode, 2)
            j = subprocess.run([sys.executable, SCRIPT, str(ok), "--json"], capture_output=True, text=True)
            self.assertEqual(json.loads(j.stdout)["kind"], "unknown_target")


if __name__ == "__main__":
    unittest.main()
