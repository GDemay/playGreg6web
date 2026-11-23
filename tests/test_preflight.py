import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import preflight  # noqa: E402


def _make_fake_toolchains(available_map: dict[str, str | None]):
    def fake_which(bin_name: str) -> str | None:
        if bin_name == "python3":
            bin_name = "python"
        return f"/fake/bin/{bin_name}" if bin_name in available_map else None

    def fake_version(resolved: str) -> str | None:
        name = Path(resolved).name
        return available_map.get(name)

    return fake_which, fake_version


class TestPreflight(unittest.TestCase):
    def _create_repo_skeleton(self, base: Path, *, include_server: bool = True) -> Path:
        repo = base / "repo"
        (repo / "web").mkdir(parents=True)
        (repo / "web" / "index.html").write_text("<h1>ok</h1>", encoding="utf-8")
        if include_server:
            (repo / "server").mkdir(parents=True)
            shutil.copy2(REPO_ROOT / "server" / "serve_local.py", repo / "server" / "serve_local.py")
        return repo

    def test_empty_repo_reports_missing_artifacts_and_milestones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repo_skeleton(Path(tmp), include_server=True)
            which_fn, version_fn = _make_fake_toolchains(
                {
                    "python": "Python 3.12.4",
                    "node": "v22.5.0",
                    "rustc": "rustc 1.85.0",
                    "cargo": "cargo 1.85.0",
                    "cmake": "cmake version 3.30.0",
                }
            )
            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
            )

            self.assertFalse(report["ok"])
            self.assertTrue(report["milestones"]["M0"]["ready"])
            self.assertFalse(report["milestones"]["M1"]["ready"])
            self.assertFalse(report["milestones"]["M2"]["ready"])
            self.assertFalse(report["milestones"]["M3"]["ready"])

            self.assertTrue(report["server"]["ok"])
            self.assertTrue(report["server"]["coopValid"])
            self.assertTrue(report["server"]["coepValid"])
            self.assertTrue(report["server"]["statusEndpointOk"])

            self.assertFalse(report["localEnv"]["gameDirConfigured"])
            self.assertFalse(report["localEnv"]["gameDirExists"])
            self.assertFalse(report["localEnv"]["dataDirConfigured"])
            self.assertFalse(report["localEnv"]["workDirConfigured"])

            self.assertFalse(report["artifacts"]["runtimeJs"]["present"])
            self.assertFalse(report["artifacts"]["runtimeWasm"]["present"])
            self.assertFalse(report["artifacts"]["dataManifest"]["present"])
            self.assertFalse(report["artifacts"]["bootset"]["present"])
            self.assertEqual(report["artifacts"]["shaderPacks"]["count"], 0)

            self.assertIn('python tools/local_env.py --set-game "<path>"', report["nextCommands"])
            self.assertIn("pwsh runtime/build_wasm.ps1", report["nextCommands"])
            self.assertIn("cargo run -p playgta6-tools -- scan --from-env", report["nextCommands"])

    def test_populated_repo_reaches_m0_through_m3_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=True)

            game_dir = base / "secret_xbox_game_folder"
            game_dir.mkdir()
            (game_dir / "MicrosoftGame.Config").write_text("<Game/>", encoding="utf-8")
            (game_dir / "levels.rpf").write_bytes(b"RPF9")
            (game_dir / "game.exe").write_bytes(b"MZ")
            (game_dir / "package.xvc").write_bytes(b"msft-xvd")

            data_dir = repo / "local-data"
            data_dir.mkdir()
            (repo / "work").mkdir()
            (repo / ".env").write_text(
                f'GTA6_GAME_DIR="{game_dir}"\n'
                "GTA6_DATA_DIR=local-data\n"
                "GTA6_WORK_DIR=work\n",
                encoding="utf-8",
            )

            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("export function init() {}\n", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

            manifest = {
                "abi_version": 1,
                "file_count": 2,
                "total_bytes": 1024,
                "files": [
                    {"path": "levels.rpf", "size": 1000},
                    {"path": "MicrosoftGame.Config", "size": 24},
                ],
            }
            (data_dir / "data-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (data_dir / "bootset.json").write_text(
                json.dumps([{"path": "levels.rpf", "offset": 0, "length": 65536}]),
                encoding="utf-8",
            )
            shaders_dir = data_dir / "shaders"
            shaders_dir.mkdir()
            (shaders_dir / "core.wgsl.pack").write_bytes(b"WGPK\x01\x00\x00\x00")
            (data_dir / "extra.wgsl.pack").write_bytes(b"WGPK\x01\x00\x00\x00")

            which_fn, version_fn = _make_fake_toolchains(
                {
                    "python": "Python 3.12.4",
                    "node": "v22.5.0",
                    "rustc": "rustc 1.85.0",
                    "cargo": "cargo 1.85.0",
                    "cmake": "cmake version 3.30.0",
                    "emcc": "emcc (Emscripten) 3.1.74",
                    "emcmake": "emcmake 3.1.74",
                    "gh": "gh version 2.60.0",
                }
            )
            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
            )

            self.assertTrue(report["ok"])
            for key in ("M0", "M1", "M2", "M3"):
                self.assertTrue(report["milestones"][key]["ready"], msg=f"{key} should be ready: {report['milestones'][key]}")

            le = report["localEnv"]
            self.assertTrue(le["envFileExists"])
            self.assertTrue(le["gameDirConfigured"])
            self.assertTrue(le["gameDirExists"])
            self.assertTrue(le["dataDirConfigured"])
            self.assertTrue(le["workDirConfigured"])
            self.assertTrue(le["gameSummary"]["hasMicrosoftGameConfig"])
            self.assertEqual(le["gameSummary"]["rpfCount"], 1)
            self.assertEqual(le["gameSummary"]["exeCount"], 1)
            self.assertEqual(le["gameSummary"]["xvcXvdCount"], 1)

            art = report["artifacts"]
            self.assertTrue(art["runtimeJs"]["nonEmpty"])
            self.assertTrue(art["runtimeWasm"]["nonEmpty"])
            self.assertTrue(art["runtimeWasm"]["validWasmMagic"])
            self.assertTrue(art["dataManifest"]["validJson"])
            self.assertEqual(art["dataManifest"]["fileCount"], 2)
            self.assertEqual(art["dataManifest"]["abiVersion"], 1)
            self.assertTrue(art["bootset"]["validJson"])
            self.assertEqual(art["bootset"]["entryCount"], 1)
            self.assertEqual(art["shaderPacks"]["count"], 2)
            self.assertEqual(art["shaderPacks"]["nonEmptyCount"], 2)

            self.assertIn("python server/serve_local.py", report["nextCommands"])

    def test_dotenv_json_never_leaks_host_paths_or_usernames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=True)
            secret_game_dir = base / "PrivateUser_TopSecret_GTA6_Dump"
            secret_game_dir.mkdir()
            secret_data_dir = base / "CustomLocalData_SecretPath"
            secret_data_dir.mkdir()
            secret_work_dir = base / "CustomWorkDir_SecretPath"
            secret_work_dir.mkdir()

            (repo / ".env").write_text(
                f"GTA6_GAME_DIR={secret_game_dir}\n"
                f"GTA6_DATA_DIR={secret_data_dir}\n"
                f"GTA6_WORK_DIR={secret_work_dir}\n",
                encoding="utf-8",
            )

            # Also verify sanitize_version strips leaked host paths and usernames from tool output.
            which_fn, version_fn = _make_fake_toolchains(
                {
                    "python": r"Python 3.12.4 (C:\Users\AliceSecret\python.exe)",
                    "node": "v22.1.0 (/home/bobsecret/bin/node)",
                    "rustc": "rustc 1.85.0",
                    "cargo": "cargo 1.85.0",
                }
            )
            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
            )
            serialized = json.dumps(report)

            for forbidden in (
                str(base),
                str(repo),
                str(secret_game_dir),
                str(secret_data_dir),
                str(secret_work_dir),
                "PrivateUser_TopSecret_GTA6_Dump",
                "CustomLocalData_SecretPath",
                "CustomWorkDir_SecretPath",
                "AliceSecret",
                "bobsecret",
                os.environ.get("USERNAME", ""),
                os.environ.get("USER", ""),
            ):
                if forbidden and len(forbidden) > 2:
                    self.assertNotIn(forbidden, serialized)

    def test_invalid_json_manifest_and_bootset_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repo_skeleton(Path(tmp), include_server=True)
            data_dir = repo / "local-data"
            data_dir.mkdir()
            (data_dir / "data-manifest.json").write_text("{not valid json", encoding="utf-8")
            (data_dir / "bootset.json").write_text('{"wrong": true}', encoding="utf-8")

            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("console.log('ok');", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

            which_fn, version_fn = _make_fake_toolchains({"python": "Python 3.12.4"})
            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
            )

            manifest = report["artifacts"]["dataManifest"]
            self.assertTrue(manifest["present"])
            self.assertTrue(manifest["nonEmpty"])
            self.assertFalse(manifest["validJson"])
            self.assertEqual(manifest["error"], "invalid JSON")
            self.assertIsNone(manifest["fileCount"])

            bootset = report["artifacts"]["bootset"]
            self.assertTrue(bootset["present"])
            self.assertFalse(bootset["validJson"])
            self.assertEqual(bootset["error"], "expected JSON array")

            # Invalid manifest/bootset must fail M1 strict readiness as well as M2.
            self.assertFalse(report["milestones"]["M1"]["ready"])
            self.assertFalse(report["milestones"]["M2"]["ready"])

    def test_cli_json_roundtrip_and_strict_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=True)

            # Empty repo: normal mode exits 0, --strict exits non-zero.
            env_clean = {k: v for k, v in os.environ.items() if not k.startswith("GTA6_")}
            proc_normal = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "preflight.py"), "--repo-root", str(repo), "--json"],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc_normal.returncode, 0, msg=proc_normal.stderr)
            self.assertEqual(proc_normal.stderr.strip(), "")
            parsed = json.loads(proc_normal.stdout)
            self.assertEqual(parsed["schemaVersion"], 1)
            self.assertFalse(parsed["milestones"]["M1"]["ready"])
            self.assertTrue(parsed["server"]["ok"])
            self.assertNotIn(str(base), proc_normal.stdout)

            proc_strict_fail = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "preflight.py"),
                    "--repo-root",
                    str(repo),
                    "--json",
                    "--strict",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertNotEqual(proc_strict_fail.returncode, 0)
            parsed_fail = json.loads(proc_strict_fail.stdout)
            self.assertFalse(parsed_fail["milestones"]["M1"]["ready"])

            # Populate required M1 artifacts -> --strict should exit 0.
            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("export default {};\n", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

            proc_strict_ok = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "preflight.py"),
                    "--repo-root",
                    str(repo),
                    "--json",
                    "--strict",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc_strict_ok.returncode, 0, msg=proc_strict_ok.stderr)
            parsed_ok = json.loads(proc_strict_ok.stdout)
            self.assertTrue(parsed_ok["ok"])
            self.assertTrue(parsed_ok["milestones"]["M1"]["ready"])

            # Human-readable output smoke check.
            proc_text = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "preflight.py"), "--repo-root", str(repo)],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc_text.returncode, 0, msg=proc_text.stderr)
            self.assertIn("Milestone Readiness:", proc_text.stdout)
            self.assertIn("M1 - Browser Runtime Shell & Wasm64 Smoke Build: READY", proc_text.stdout)

    def test_missing_server_script_and_invalid_wasm_magic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repo_skeleton(Path(tmp), include_server=False)
            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("export {};", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"NOT_WASM")

            which_fn, version_fn = _make_fake_toolchains({"python": "Python 3.12.4"})
            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
            )

            self.assertFalse(report["server"]["serverScriptPresent"])
            self.assertFalse(report["server"]["ok"])
            self.assertTrue(report["artifacts"]["runtimeWasm"]["nonEmpty"])
            self.assertFalse(report["artifacts"]["runtimeWasm"]["validWasmMagic"])
            self.assertFalse(report["milestones"]["M1"]["ready"])

    def test_in_process_server_header_fallback(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "server"))
        try:
            import serve_local  # noqa: E402

            headers = preflight._check_headers_in_process(serve_local.PlayGta6Handler)
            self.assertEqual(headers.get("cross-origin-opener-policy"), "same-origin")
            self.assertEqual(headers.get("cross-origin-embedder-policy"), "require-corp")
            self.assertEqual(headers.get("cross-origin-resource-policy"), "same-origin")
            self.assertEqual(headers.get("accept-ranges"), "bytes")
        finally:
            if str(REPO_ROOT / "server") in sys.path:
                sys.path.remove(str(REPO_ROOT / "server"))


if __name__ == "__main__":
    unittest.main()
