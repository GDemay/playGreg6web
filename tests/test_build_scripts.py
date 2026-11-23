import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BUILD_WASM_SH = REPO_ROOT / "runtime" / "build_wasm.sh"
BUILD_WASM_PS1 = REPO_ROOT / "runtime" / "build_wasm.ps1"
RUNTIME_CMAKE = REPO_ROOT / "runtime" / "CMakeLists.txt"


def _find_bash() -> tuple[str | None, bool]:
    """Locate a working bash executable and report whether it shares host PATH.

    Returns ``(bash_path, shares_host_env)``. On Windows, prefers Git Bash
    (MSYS2) over ``C:\\Windows\\System32\\bash.exe`` (WSL) so relative paths and
    host environment overrides work consistently.
    """
    candidates: list[tuple[str, bool]] = []
    if sys.platform == "win32":
        for git_bash in (
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            r"C:\Program Files (x86)\Git\bin\bash.exe",
        ):
            if Path(git_bash).is_file():
                candidates.append((git_bash, True))

    on_path = shutil.which("bash")
    if on_path:
        is_wsl = sys.platform == "win32" and Path(on_path).resolve().parent.name.lower() in (
            "system32",
            "windowsapps",
        )
        candidates.append((on_path, not is_wsl))

    seen: set[str] = set()
    for candidate, shares_env in candidates:
        norm = os.path.normcase(os.path.abspath(candidate))
        if norm in seen:
            continue
        seen.add(norm)
        try:
            probe = subprocess.run(
                [candidate, "-c", "exit 0"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if probe.returncode == 0:
                return candidate, shares_env
        except (OSError, subprocess.SubprocessError):
            continue

    return None, False


def _write_executable_script(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8"))
    if sys.platform != "win32":
        mode = path.stat().st_mode
        path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TestBuildScripts(unittest.TestCase):
    def test_build_wasm_sh_exists_and_uses_lf_line_endings(self) -> None:
        self.assertTrue(BUILD_WASM_SH.is_file(), "runtime/build_wasm.sh must exist")
        raw = BUILD_WASM_SH.read_bytes()
        self.assertNotIn(
            b"\r\n",
            raw,
            "runtime/build_wasm.sh must use LF line endings (no CRLF)",
        )
        self.assertNotIn(
            b"\r",
            raw,
            "runtime/build_wasm.sh must not contain stray CR bytes",
        )

    def test_build_wasm_sh_header_flags_and_parity_with_ps1(self) -> None:
        sh_text = BUILD_WASM_SH.read_text(encoding="utf-8")
        ps1_text = BUILD_WASM_PS1.read_text(encoding="utf-8")

        lines = sh_text.splitlines()
        self.assertGreaterEqual(len(lines), 2)
        self.assertEqual(lines[0], "#!/usr/bin/env bash")
        self.assertIn("set -euo pipefail", sh_text)

        # Required Wasm64 flags and CLI options
        for token in (
            "-sMEMORY64=1",
            "-pthread",
            "--check-only",
            "--build-type",
            "--build-dir",
            "--help",
            "Release",
            "Debug",
            "RelWithDebInfo",
            "build-wasm64",
            "emcmake",
            "emcc",
            "emsdk",
            "0061736d",
        ):
            self.assertIn(token, sh_text, f"Missing expected token in build_wasm.sh: {token}")

        # Parity with runtime/build_wasm.ps1 outputs and CMake configuration
        for shared_token in (
            "-sMEMORY64=1",
            "-pthread",
            "runtime.js",
            "runtime.wasm",
            "web/dist/runtime.js + web/dist/runtime.wasm",
            "CMakeLists.txt",
            "CMAKE_BUILD_TYPE",
            "CMAKE_CXX_FLAGS",
            "--parallel",
        ):
            self.assertIn(shared_token, ps1_text, f"Expected token in build_wasm.ps1: {shared_token}")
            self.assertIn(shared_token, sh_text, f"build_wasm.sh must mirror build_wasm.ps1 token: {shared_token}")

    def test_bash_syntax_help_and_check_only(self) -> None:
        bash_exe, _ = _find_bash()
        if not bash_exe:
            self.skipTest("bash is not available in this environment")

        # 1. Syntax check: bash -n runtime/build_wasm.sh
        syntax = subprocess.run(
            [bash_exe, "-n", "runtime/build_wasm.sh"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(
            syntax.returncode,
            0,
            f"bash -n failed:\nstdout={syntax.stdout}\nstderr={syntax.stderr}",
        )

        # 2. Help output: bash runtime/build_wasm.sh --help
        help_res = subprocess.run(
            [bash_exe, "runtime/build_wasm.sh", "--help"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(
            help_res.returncode,
            0,
            f"--help failed:\nstdout={help_res.stdout}\nstderr={help_res.stderr}",
        )
        self.assertIn("--check-only", help_res.stdout)
        self.assertIn("--build-type", help_res.stdout)
        self.assertIn("--build-dir", help_res.stdout)
        self.assertIn("--help", help_res.stdout)

        # 3. Invalid build type is rejected cleanly
        bad_type = subprocess.run(
            [bash_exe, "runtime/build_wasm.sh", "--build-type", "InvalidType"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertNotEqual(bad_type.returncode, 0)
        self.assertIn("invalid --build-type", (bad_type.stdout + bad_type.stderr).lower())

        # 4. --check-only produces clean diagnostics whether emcmake is installed or not
        check_res = subprocess.run(
            [bash_exe, "runtime/build_wasm.sh", "--check-only"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )
        combined = check_res.stdout + check_res.stderr
        if check_res.returncode == 0:
            self.assertIn("Found emcmake:", combined)
            self.assertIn("Found emcc:", combined)
            self.assertIn("-sMEMORY64=1", combined)
            self.assertIn("-pthread", combined)
            self.assertIn("check passed", combined)
        else:
            self.assertEqual(check_res.returncode, 1)
            self.assertIn("not found on PATH", combined)
            self.assertIn("emsdk", combined)
            self.assertIn("emsdk_env.sh", combined)

    def test_mocked_toolchain_check_only_and_wasm_magic_validation(self) -> None:
        bash_exe, shares_env = _find_bash()
        if not bash_exe or not shares_env:
            self.skipTest("host-env bash is not available in this environment")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            tmp_runtime = tmp_root / "runtime"
            tmp_bin = tmp_root / "fake_bin"
            tmp_runtime.mkdir(parents=True)
            tmp_bin.mkdir(parents=True)

            (tmp_runtime / "build_wasm.sh").write_bytes(BUILD_WASM_SH.read_bytes())
            (tmp_runtime / "CMakeLists.txt").write_bytes(RUNTIME_CMAKE.read_bytes())

            # Fake emcc, emcmake, and cmake scripts for isolated end-to-end validation
            _write_executable_script(
                tmp_bin / "emcc",
                "#!/usr/bin/env bash\n"
                "echo 'emcc (Emscripten fake) 3.1.70'\n",
            )
            _write_executable_script(
                tmp_bin / "emcmake",
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "exec \"$@\"\n",
            )
            _write_executable_script(
                tmp_bin / "cmake",
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "if [[ \"${1:-}\" == \"--build\" ]]; then\n"
                "  mkdir -p \"${MOCK_DIST_DIR}\"\n"
                "  echo '// fake runtime.js' > \"${MOCK_DIST_DIR}/runtime.js\"\n"
                "  if [[ \"${MOCK_BAD_WASM:-0}\" == \"1\" ]]; then\n"
                "    printf 'NOPE1234' > \"${MOCK_DIST_DIR}/runtime.wasm\"\n"
                "  else\n"
                "    printf '\\000asm\\001\\000\\000\\000' > \"${MOCK_DIST_DIR}/runtime.wasm\"\n"
                "  fi\n"
                "  exit 0\n"
                "fi\n"
                "echo \"CONFIGURE:$*\" > \"${MOCK_LOG}\"\n",
            )

            env = os.environ.copy()
            env["PATH"] = f"{tmp_bin}{os.pathsep}{env.get('PATH', '')}"
            env["MOCK_DIST_DIR"] = (tmp_root / "web" / "dist").as_posix()
            env["MOCK_LOG"] = (tmp_root / "cmake_args.log").as_posix()

            # --check-only should succeed with mocked emsdk tools and create web/dist
            check_ok = subprocess.run(
                [bash_exe, "runtime/build_wasm.sh", "--check-only"],
                cwd=tmp_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(
                check_ok.returncode,
                0,
                f"mocked --check-only failed:\nstdout={check_ok.stdout}\nstderr={check_ok.stderr}",
            )
            self.assertIn("Verified Wasm64 build flags", check_ok.stdout)
            self.assertIn("check passed (--check-only)", check_ok.stdout)
            self.assertTrue((tmp_root / "web" / "dist").is_dir())

            # Full mocked build with valid \0asm magic bytes should succeed
            build_ok = subprocess.run(
                [
                    bash_exe,
                    "runtime/build_wasm.sh",
                    "--build-type",
                    "RelWithDebInfo",
                    "--build-dir",
                    "custom-build",
                ],
                cwd=tmp_root,
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(
                build_ok.returncode,
                0,
                f"mocked build failed:\nstdout={build_ok.stdout}\nstderr={build_ok.stderr}",
            )
            self.assertIn("Built Wasm64 runtime into web/dist/runtime.js + web/dist/runtime.wasm", build_ok.stdout)
            logged = (tmp_root / "cmake_args.log").read_text(encoding="utf-8")
            self.assertIn("-DCMAKE_BUILD_TYPE=RelWithDebInfo", logged)
            self.assertIn("-DCMAKE_CXX_FLAGS=-sMEMORY64=1 -pthread", logged)
            self.assertIn("-B custom-build", logged)

            # Full mocked build with invalid wasm magic bytes must fail
            bad_env = env.copy()
            bad_env["MOCK_BAD_WASM"] = "1"
            build_bad = subprocess.run(
                [bash_exe, "runtime/build_wasm.sh"],
                cwd=tmp_root,
                env=bad_env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertNotEqual(build_bad.returncode, 0)
            self.assertIn("magic bytes", (build_bad.stdout + build_bad.stderr).lower())


if __name__ == "__main__":
    unittest.main()
