import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import find_chrome as fc  # noqa: E402


class FindChromeTests(unittest.TestCase):
    def test_windows_prefers_program_files_chrome(self):
        env = {"ProgramFiles": "C:/PF", "ProgramFiles(x86)": "C:/PF86", "LOCALAPPDATA": "C:/L"}
        want = str(Path("C:/PF") / "Google/Chrome/Application/chrome.exe")
        self.assertEqual(fc.find_chrome(env, "win32", exists=lambda p: True, which=lambda n: None), want)

    def test_windows_localappdata_then_edge(self):
        env = {"ProgramFiles": "C:/PF", "LOCALAPPDATA": "C:/L"}
        local = str(Path("C:/L") / "Google/Chrome/Application/chrome.exe")
        self.assertEqual(fc.find_chrome(env, "win32", exists=lambda p: p == local, which=lambda n: None), local)
        edge = str(Path("C:/PF") / "Microsoft/Edge/Application/msedge.exe")
        self.assertEqual(fc.find_chrome(env, "win32", exists=lambda p: p == edge, which=lambda n: None), edge)

    def test_none_found(self):
        for plat in ("win32", "linux", "darwin"):
            self.assertIsNone(fc.find_chrome({}, plat, exists=lambda p: False, which=lambda n: None))

    def test_mac(self):
        got = fc.find_chrome({}, "darwin", exists=lambda p: "Google Chrome.app" in p, which=lambda n: None)
        self.assertTrue(got.endswith("Google Chrome"))

    def test_linux_path_order(self):
        got = fc.find_chrome({}, "linux", which=lambda n: "/usr/bin/" + n if n == "chromium" else None)
        self.assertEqual(got, "/usr/bin/chromium")

    def test_resolve_order_and_log(self):
        logs = []
        self.assertEqual(fc.resolve_chrome("x", {"PLAYGTA6_CHROME": "y"}, log=logs.append), "x")
        self.assertEqual(fc.resolve_chrome(None, {"PLAYGTA6_CHROME": "y"}, log=logs.append), "y")
        self.assertEqual(logs, [])
        got = fc.resolve_chrome(None, {}, log=logs.append, platform="linux", which=lambda n: "/b/" + n)
        self.assertEqual(got, "/b/google-chrome")
        self.assertIn("auto-detected /b/google-chrome", logs[0])


if __name__ == "__main__":
    unittest.main()
