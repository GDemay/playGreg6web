#!/usr/bin/env python3
"""Tests for tools/emsdk_locator.py using fake directory trees."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_guest_wasm as bgw  # noqa: E402
import emsdk_locator as el  # noqa: E402


def _emsdk(root: Path, rel: str, version: str | None, mtime: float | None = None) -> Path:
    em = root / "work" / rel / "emsdk" / "upstream" / "emscripten"
    em.mkdir(parents=True)
    (em / "emcc").write_text("x")
    if version is not None:
        v = em / "emscripten-version.txt"
        v.write_text(f'"{version}"\n')
        if mtime is not None:
            os.utime(v, (mtime, mtime))
    return em


class EmsdkLocatorTests(unittest.TestCase):
    def test_none(self):
        with tempfile.TemporaryDirectory() as t:
            self.assertIsNone(el.select_work_emsdk(Path(t)))

    def test_nested_found(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            em = _emsdk(root, "real/o36", "7.0.1")
            self.assertEqual(el.select_work_emsdk(root), (em, "7.0.1"))

    def test_prefers_matching_version_over_newer(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            good = _emsdk(root, "a", "7.0.1", 1000)
            _emsdk(root, "b", "4.0.10", 5000)
            self.assertEqual(el.select_work_emsdk(root)[0], good)

    def test_falls_back_to_newest(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            _emsdk(root, "a", "4.0.1", 1000)
            new = _emsdk(root, "b", "4.0.10", 5000)
            self.assertEqual(el.select_work_emsdk(root), (new, "4.0.10"))

    def test_discover_uses_nested_and_prints(self):
        import io
        from contextlib import redirect_stderr
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            em = _emsdk(root, "real/o36", "7.0.1")
            err = io.StringIO()
            with redirect_stderr(err):
                tc = bgw.discover_emscripten(repo_root=root, env={}, which_fn=lambda _n: None)
            self.assertEqual(tc.emcc, em / "emcc")
            self.assertIn("7.0.1", err.getvalue())


if __name__ == "__main__":
    unittest.main()
