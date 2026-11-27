import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import prebuilt_web as pw  # noqa: E402


def make(root: Path, name: str, marker: str) -> Path:
    d = root / name
    d.mkdir()
    (d / "m.txt").write_text(marker)
    return d


class SwapRetryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.target = make(self.root, "web", "old")
        self.new = make(self.root, "web.new", "new")
        self.sleeps = []
        p = mock.patch.object(pw.time, "sleep", self.sleeps.append)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    def flaky_replace(self, fail_times, only_dst=None):
        real = os.replace
        state = {"n": 0}

        def fn(src, dst):
            if only_dst is None or Path(dst) == only_dst:
                if state["n"] < fail_times:
                    state["n"] += 1
                    raise PermissionError(5, "locked")
            return real(src, dst)
        return fn

    def test_clean_swap(self):
        pw.swap(self.new, self.target)
        self.assertEqual((self.target / "m.txt").read_text(), "new")
        self.assertFalse((self.root / "web.old").exists())
        self.assertEqual(self.sleeps, [])

    def test_retries_then_succeeds(self):
        with mock.patch.object(pw.os, "replace", self.flaky_replace(3, self.root / "web.old")):
            pw.swap(self.new, self.target)
        self.assertEqual((self.target / "m.txt").read_text(), "new")
        self.assertEqual(self.sleeps, [0.5, 1.0, 2.0])

    def test_gives_up_after_five_tries_with_clear_message(self):
        with mock.patch.object(pw.os, "replace", self.flaky_replace(99, self.root / "web.old")):
            with self.assertRaises(OSError) as cm:
                pw.swap(self.new, self.target)
        self.assertIn("5 attempts", str(cm.exception))
        self.assertEqual(self.sleeps, [0.5, 1.0, 2.0, 4.0])
        self.assertEqual((self.target / "m.txt").read_text(), "old")

    def test_rolls_back_when_final_rename_fails(self):
        real = os.replace

        def fn(src, dst):
            if Path(src) == self.new:
                raise PermissionError(5, "locked")
            return real(src, dst)
        with mock.patch.object(pw.os, "replace", fn):
            with self.assertRaises(OSError):
                pw.swap(self.new, self.target)
        self.assertEqual((self.target / "m.txt").read_text(), "old")
        self.assertTrue((self.new / "m.txt").exists())
        self.assertFalse((self.root / "web.old").exists())


if __name__ == "__main__":
    unittest.main()
