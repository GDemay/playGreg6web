import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from collections import namedtuple
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import preflight  # noqa: E402

_FakeDiskUsage = namedtuple("_FakeDiskUsage", ["total", "used", "free"])


def _build_wgpk_v1_pack(
    entries: list[tuple[int, int, bytes]],
    *,
    magic: bytes = b"WGPK",
    version: int = 1,
    flags: int = 0,
) -> bytes:
    header = struct.pack("<4sIII", magic, version, len(entries), flags)
    payload_offset = len(header) + len(entries) * 20
    table_chunks: list[bytes] = []
    payload_chunks: list[bytes] = []
    cursor = payload_offset
    for shader_hash, stage_code, wgsl_bytes in entries:
        table_chunks.append(struct.pack("<QB3sII", shader_hash, stage_code, b"\x00\x00\x00", cursor, len(wgsl_bytes)))
        payload_chunks.append(wgsl_bytes)
        cursor += len(wgsl_bytes)
    return header + b"".join(table_chunks) + b"".join(payload_chunks)


def _make_valid_sidecar_doc(stage_specs: list[tuple[str, str]]) -> dict:
    stages: dict[str, dict] = {}
    for key, stage in stage_specs:
        stages[key] = {
            "stage": stage,
            "bindings": [
                {
                    "group": 0,
                    "binding": 0,
                    "kind": "uniform",
                    "source": {"user_sgpr": 8, "table_byte_offset": 0},
                }
            ],
        }
    return {
        "schema": "playgta6-stage-reflection",
        "version": 1,
        "stages": stages,
    }


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

    def test_disk_space_threshold_logic_and_non_fatal_warning(self) -> None:
        gib = preflight.BYTES_PER_GIB
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=True)
            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("export default {};\n", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
            custom_work = base / "separate_work_drive"
            custom_work.mkdir()
            which_fn, version_fn = _make_fake_toolchains({"python": "Python 3.12.4"})

            # Exact 5.0 GiB boundary -> OK, no warning.
            exact_disk = preflight.check_disk_space(
                repo,
                work_dir=repo / "nonexistent_work_subdir",
                min_free_gb=5.0,
                disk_usage_fn=lambda _p: _FakeDiskUsage(total=100 * gib, used=95 * gib, free=5 * gib),
            )
            self.assertTrue(exact_disk["ok"])
            self.assertFalse(exact_disk["warning"])
            self.assertFalse(exact_disk["lowSpace"])
            self.assertEqual(exact_disk["freeGb"], 5.0)
            self.assertEqual(exact_disk["minFreeGb"], 5.0)

            # 1 byte below 5.0 GiB -> triggers lowSpace warning (unrounded comparison).
            just_below = preflight.check_disk_space(
                repo,
                work_dir=repo / "work",
                min_free_gb=5.0,
                disk_usage_fn=lambda _p: _FakeDiskUsage(total=100 * gib, used=95 * gib + 1, free=5 * gib - 1),
            )
            self.assertFalse(just_below["ok"])
            self.assertTrue(just_below["warning"])
            self.assertTrue(just_below["lowSpace"])

            # Workspace has 20 GiB free, separate work drive has 2.5 GiB free -> warning on work only,
            # and M1 readiness stays True (disk warning is non-fatal).
            def split_usage(probe_path: Path) -> _FakeDiskUsage:
                if probe_path.resolve() == custom_work.resolve():
                    return _FakeDiskUsage(total=50 * gib, used=int(47.5 * gib), free=int(2.5 * gib))
                return _FakeDiskUsage(total=100 * gib, used=80 * gib, free=20 * gib)

            report = preflight.run_preflight(
                repo,
                env={"GTA6_WORK_DIR": str(custom_work)},
                which_fn=which_fn,
                version_fn=version_fn,
                min_free_gb=5.0,
                disk_usage_fn=split_usage,
            )
            self.assertTrue(report["ok"])
            self.assertTrue(report["milestones"]["M1"]["ready"])
            ds = report["diskSpace"]
            self.assertTrue(ds["warning"])
            self.assertTrue(ds["lowSpace"])
            self.assertFalse(ds["ok"])
            self.assertEqual(ds["freeGb"], 2.5)
            self.assertTrue(ds["workspace"]["ok"])
            self.assertFalse(ds["workspace"]["lowSpace"])
            self.assertEqual(ds["workspace"]["freeGb"], 20.0)
            self.assertFalse(ds["work"]["ok"])
            self.assertTrue(ds["work"]["lowSpace"])
            self.assertEqual(ds["work"]["freeGb"], 2.5)

            rendered = preflight.format_report_text(report)
            self.assertIn("Disk Space", rendered)
            self.assertIn("[WARN] work drive", rendered)
            self.assertNotIn(str(custom_work), rendered)
            self.assertNotIn(str(base), json.dumps(report))

            # OSError from disk_usage is handled cleanly and never leaks host paths.
            def failing_usage(probe_path: Path) -> _FakeDiskUsage:
                raise OSError(f"cannot stat {probe_path}")

            err_disk = preflight.check_disk_space(repo, disk_usage_fn=failing_usage)
            self.assertFalse(err_disk["ok"])
            self.assertFalse(err_disk["warning"])
            self.assertEqual(err_disk["workspace"]["error"], "disk usage unavailable")
            self.assertNotIn(str(base), json.dumps(err_disk))

    def test_cli_min_free_gb_flag_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repo_skeleton(Path(tmp), include_server=True)
            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("export default {};\n", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
            env_clean = {k: v for k, v in os.environ.items() if not k.startswith("GTA6_")}

            # Threshold 0 -> guaranteed no low-space warning.
            proc_zero = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "preflight.py"),
                    "--repo-root",
                    str(repo),
                    "--min-free-gb",
                    "0",
                    "--json",
                    "--strict",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc_zero.returncode, 0, msg=proc_zero.stderr)
            parsed_zero = json.loads(proc_zero.stdout)
            self.assertFalse(parsed_zero["diskSpace"]["warning"])
            self.assertEqual(parsed_zero["diskSpace"]["minFreeGb"], 0.0)

            # Impossibly high threshold -> emits warning, but --strict still exits 0 when M1 is ready.
            proc_high = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "preflight.py"),
                    "--repo-root",
                    str(repo),
                    "--min-free-gb",
                    "9999999",
                    "--json",
                    "--strict",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc_high.returncode, 0, msg=proc_high.stderr)
            parsed_high = json.loads(proc_high.stdout)
            self.assertTrue(parsed_high["diskSpace"]["warning"])
            self.assertTrue(parsed_high["diskSpace"]["lowSpace"])
            self.assertTrue(parsed_high["milestones"]["M1"]["ready"])

            # Negative threshold -> rejected by CLI argument validation.
            proc_neg = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "preflight.py"),
                    "--repo-root",
                    str(repo),
                    "--min-free-gb",
                    "-1",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertNotEqual(proc_neg.returncode, 0)

    def test_shader_sidecar_absent_never_fails_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repo_skeleton(Path(tmp), include_server=True)
            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("export default {};\n", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")
            which_fn, version_fn = _make_fake_toolchains({"python": "Python 3.12.4"})

            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
            )
            sidecars = report["artifacts"]["shaderSidecars"]
            self.assertFalse(sidecars["present"])
            self.assertTrue(sidecars["ok"])
            self.assertEqual(sidecars["count"], 0)
            self.assertEqual(sidecars["validCount"], 0)
            self.assertEqual(sidecars["invalidCount"], 0)
            self.assertEqual(sidecars["stageCount"], 0)
            self.assertEqual(sidecars["stageCounts"], {"vertex": 0, "fragment": 0, "compute": 0})
            self.assertEqual(sidecars["inlineSamplerCount"], 0)
            self.assertEqual(sidecars["inlineSamplerStageCount"], 0)
            self.assertEqual(sidecars["invalidInlineSamplerCount"], 0)
            self.assertEqual(sidecars["errors"], [])
            self.assertTrue(report["milestones"]["M1"]["ready"])

    def test_shader_sidecar_valid_stage_and_inline_sampler_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._create_repo_skeleton(Path(tmp), include_server=True)
            shaders_dir = repo / "local-data" / "shaders"
            shaders_dir.mkdir(parents=True)
            (shaders_dir / "core.wgsl.pack").write_bytes(b"WGPK\x01\x00\x00\x00")
            (shaders_dir / "unpaired.wgsl.pack").write_bytes(b"WGPK\x01\x00\x00\x00")

            sidecar_doc = {
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {
                    "0x00000000000000a1": {
                        "stage": "vertex",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "uniform",
                                "source": {"user_sgpr": 8, "table_byte_offset": 0},
                            }
                        ],
                    },
                    "0x00000000000000b2": {
                        "stage": "fragment",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 1,
                                "kind": "texture",
                                "source": {"user_sgpr": 8, "table_byte_offset": 32},
                            },
                            {
                                "group": 0,
                                "binding": 34,
                                "kind": "sampler",
                                "source": {"inline_sampler": {"sdwords": [0x92, 0x00F00000, 0x08500000, 0]}},
                            },
                            {
                                "group": 0,
                                "binding": 35,
                                "kind": "sampler",
                                "source": {"inline_sampler": {"sdwords": [0, 0, 0, 0xFFFFFFFF]}},
                            },
                        ],
                    },
                    "0x00000000000000c3": {
                        "stage": "pixel",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 2,
                                "kind": "sampler",
                                "source": {"static_sampler": 0},
                            }
                        ],
                    },
                    "0x00000000000000d4": {
                        "stage": "compute",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "storage",
                                "source": {"user_sgpr": 6, "table_byte_offset": 0},
                            },
                            {
                                "group": 0,
                                "binding": 1,
                                "kind": "sampler",
                                "source": {"inline_sampler": {"sdwords": [1, 2, 3, 4]}},
                            },
                        ],
                    },
                },
            }
            (shaders_dir / "core.wgsl.pack.reflect.json").write_text(
                json.dumps(sidecar_doc),
                encoding="utf-8",
            )

            which_fn, version_fn = _make_fake_toolchains({"python": "Python 3.12.4"})
            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
            )

            packs = report["artifacts"]["shaderPacks"]
            self.assertEqual(packs["count"], 2)
            self.assertEqual(packs["nonEmptyCount"], 2)
            self.assertEqual(packs["pairedSidecarCount"], 1)

            sidecars = report["artifacts"]["shaderSidecars"]
            self.assertTrue(sidecars["present"])
            self.assertTrue(sidecars["ok"])
            self.assertEqual(sidecars["count"], 1)
            self.assertEqual(sidecars["validCount"], 1)
            self.assertEqual(sidecars["invalidCount"], 0)
            self.assertEqual(sidecars["pairedPackCount"], 1)
            self.assertEqual(sidecars["unpairedPackCount"], 1)
            self.assertEqual(sidecars["stageCount"], 4)
            self.assertEqual(sidecars["stageCounts"], {"vertex": 1, "fragment": 2, "compute": 1})
            self.assertEqual(sidecars["bindingCount"], 7)
            self.assertEqual(sidecars["inlineSamplerCount"], 3)
            self.assertEqual(sidecars["inlineSamplerStageCount"], 2)
            self.assertEqual(sidecars["invalidInlineSamplerCount"], 0)
            self.assertEqual(sidecars["errors"], [])

            rendered = preflight.format_report_text(report)
            self.assertIn("*.reflect.json sidecars", rendered)
            self.assertIn("stages=4: vs=1, fs=2, cs=1", rendered)
            self.assertIn("inline_samplers=3 in 2 stages", rendered)

    def test_shader_sidecar_malformed_inline_sampler_and_sanitization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=True)
            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("export default {};\n", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

            shaders_dir = repo / "local-data" / "shaders"
            shaders_dir.mkdir(parents=True)
            (shaders_dir / "SecretDumpPack.wgsl.pack").write_bytes(b"WGPK\x01\x00\x00\x00")

            hostile_sidecar = {
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {
                    r"C:\Users\SecretPlayer\0xdeadbeef": {
                        "stage": "fragment",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "sampler",
                                "source": {"inline_sampler": {"sdwords": [1, 2, 3]}},
                            },
                            {
                                "group": 0,
                                "binding": 1,
                                "kind": "sampler",
                                "source": {"inline_sampler": {"sdwords": [1, 2, 3, True]}},
                            },
                            {
                                "group": 0,
                                "binding": 2,
                                "kind": "sampler",
                                "source": {"inline_sampler": {"sdwords": [1, 2, 3, 0x100000000]}},
                            },
                            {
                                "group": 0,
                                "binding": 3,
                                "kind": "sampler",
                                "source": {"inline_sampler": {"sdwords": [10, 20, 30, 40]}},
                            },
                        ],
                    }
                },
            }
            (shaders_dir / "SecretDumpPack.wgsl.pack.reflect.json").write_text(
                json.dumps(hostile_sidecar),
                encoding="utf-8",
            )
            (shaders_dir / "broken_secret_name.reflect.json").write_text(
                "{not valid json",
                encoding="utf-8",
            )

            which_fn, version_fn = _make_fake_toolchains({"python": "Python 3.12.4"})
            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
            )

            sidecars = report["artifacts"]["shaderSidecars"]
            self.assertTrue(sidecars["present"])
            self.assertFalse(sidecars["ok"])
            self.assertEqual(sidecars["count"], 2)
            self.assertEqual(sidecars["validCount"], 0)
            self.assertEqual(sidecars["invalidCount"], 2)
            self.assertEqual(sidecars["stageCount"], 1)
            self.assertEqual(sidecars["inlineSamplerCount"], 1)
            self.assertEqual(sidecars["inlineSamplerStageCount"], 1)
            self.assertEqual(sidecars["invalidInlineSamplerCount"], 3)
            self.assertIn("malformed inline_sampler", sidecars["errors"])
            self.assertIn("invalid JSON", sidecars["errors"])

            # M1 stays ready (sidecar issues don't break M1/--strict), while M2 flags invalid sidecar.
            self.assertTrue(report["milestones"]["M1"]["ready"])
            self.assertIn("shaders:*.reflect.json (invalid)", report["milestones"]["M2"]["missing"])

            serialized = json.dumps(report)
            rendered = preflight.format_report_text(report)
            for forbidden in (
                str(base),
                "SecretPlayer",
                "SecretDumpPack",
                "broken_secret_name",
                "0xdeadbeef",
            ):
                self.assertNotIn(forbidden, serialized)
                self.assertNotIn(forbidden, rendered)

    def test_splash_wgpk_valid_and_sidecar_in_sync(self) -> None:
        gib = preflight.BYTES_PER_GIB
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=True)
            custom_shaders = base / "SecretCustomShadersDir"
            custom_shaders.mkdir()

            pack_bytes = _build_wgpk_v1_pack(
                [
                    (0x1111, 1, b"@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"),
                    (0x2222, 2, b"@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"),
                ]
            )
            pack_path = custom_shaders / "splash.wgsl.pack"
            sidecar_path = custom_shaders / "splash.wgsl.pack.reflect.json"
            pack_path.write_bytes(pack_bytes)
            sidecar_path.write_text(
                json.dumps(_make_valid_sidecar_doc([("0x1111", "vertex"), ("0x2222", "fragment")])),
                encoding="utf-8",
            )
            now = time.time()
            os.utime(pack_path, (now - 5, now - 5))
            os.utime(sidecar_path, (now, now))

            which_fn, version_fn = _make_fake_toolchains(
                {
                    "python": "Python 3.12.4",
                    "chrome": r"Google Chrome 131.0.6778.86 (C:\Users\AliceSecret\chrome.exe)",
                }
            )
            report = preflight.run_preflight(
                repo,
                env={},
                which_fn=which_fn,
                version_fn=version_fn,
                skip_server_check=True,
                shaders_dir=custom_shaders,
                disk_usage_fn=lambda _p: _FakeDiskUsage(total=200 * gib, used=100 * gib, free=100 * gib),
                memory_info_fn=lambda: (32 * gib, 16 * gib),
            )

            sp = report["splashPack"]
            self.assertTrue(sp["shadersDirProvided"])
            self.assertTrue(sp["shadersDirExists"])
            self.assertTrue(sp["present"])
            self.assertEqual(sp["sizeBytes"], len(pack_bytes))
            self.assertTrue(sp["headerValid"])
            self.assertEqual(sp["version"], 1)
            self.assertEqual(sp["entryCount"], 2)
            self.assertEqual(sp["stageCounts"], {"vertex": 1, "fragment": 1, "compute": 0})
            self.assertTrue(sp["sidecarPresent"])
            self.assertTrue(sp["sidecarValid"])
            self.assertEqual(sp["sidecarStageCount"], 2)
            self.assertFalse(sp["sidecarStale"])
            self.assertGreaterEqual(sp["mtimeDeltaSeconds"], 0.0)
            self.assertTrue(sp["countMatch"])
            self.assertTrue(sp["inSync"])
            self.assertTrue(sp["ok"])
            self.assertFalse(sp["warning"])
            self.assertEqual(sp["warnings"], [])
            self.assertEqual(sp["errors"], [])

            fb = report["fastBoot"]
            self.assertTrue(fb["ready"])
            self.assertTrue(fb["splashPackOk"])
            self.assertTrue(fb["sidecarInSync"])
            self.assertTrue(fb["browserAvailable"])
            self.assertEqual(fb["browserFamily"], "chrome")
            self.assertTrue(fb["diskHeadroomOk"])
            self.assertTrue(fb["ramHeadroomOk"])
            self.assertEqual(fb["warnings"], [])

            serialized = json.dumps(report)
            rendered = preflight.format_report_text(report)
            self.assertIn("splash.wgsl.pack", rendered)
            self.assertIn("sidecar in sync", rendered)
            for forbidden in (str(base), str(custom_shaders), "SecretCustomShadersDir", "AliceSecret"):
                self.assertNotIn(forbidden, serialized)
                self.assertNotIn(forbidden, rendered)

    def test_splash_wgpk_missing_stale_mismatch_and_invalid_sidecar_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=False)
            shaders_dir = base / "shaders"
            shaders_dir.mkdir()

            pack_bytes = _build_wgpk_v1_pack(
                [
                    (0x10, 1, b"@vertex fn vs() {}"),
                    (0x20, 2, b"@fragment fn fs() {}"),
                ]
            )
            pack_path = shaders_dir / "splash.wgsl.pack"
            pack_path.write_bytes(pack_bytes)

            # 1) Missing sidecar warning.
            missing_sidecar = preflight.check_splash_wgpk_sync(repo, shaders_dir=shaders_dir)
            self.assertTrue(missing_sidecar["present"])
            self.assertTrue(missing_sidecar["headerValid"])
            self.assertFalse(missing_sidecar["sidecarPresent"])
            self.assertFalse(missing_sidecar["inSync"])
            self.assertFalse(missing_sidecar["ok"])
            self.assertTrue(missing_sidecar["warning"])
            self.assertTrue(any("missing" in w for w in missing_sidecar["warnings"]))

            # 2) Stale sidecar warning (older mtime than pack).
            sidecar_path = shaders_dir / "splash.wgsl.pack.reflect.json"
            sidecar_path.write_text(
                json.dumps(_make_valid_sidecar_doc([("0x10", "vertex"), ("0x20", "fragment")])),
                encoding="utf-8",
            )
            now = time.time()
            os.utime(pack_path, (now, now))
            os.utime(sidecar_path, (now - 120, now - 120))
            stale = preflight.check_splash_wgpk_sync(repo, shaders_dir=shaders_dir)
            self.assertTrue(stale["sidecarPresent"])
            self.assertTrue(stale["sidecarValid"])
            self.assertTrue(stale["sidecarStale"])
            self.assertLess(stale["mtimeDeltaSeconds"], 0.0)
            self.assertTrue(stale["countMatch"])
            self.assertFalse(stale["inSync"])
            self.assertFalse(stale["ok"])
            self.assertTrue(stale["warning"])
            self.assertTrue(any("stale" in w for w in stale["warnings"]))

            # 3) Fresh sidecar with mismatched stage count.
            sidecar_path.write_text(
                json.dumps(_make_valid_sidecar_doc([("0x10", "vertex")])),
                encoding="utf-8",
            )
            os.utime(sidecar_path, (now + 10, now + 10))
            mismatch = preflight.check_splash_wgpk_sync(repo, shaders_dir=shaders_dir)
            self.assertTrue(mismatch["sidecarValid"])
            self.assertFalse(mismatch["sidecarStale"])
            self.assertEqual(mismatch["sidecarStageCount"], 1)
            self.assertEqual(mismatch["entryCount"], 2)
            self.assertFalse(mismatch["countMatch"])
            self.assertFalse(mismatch["inSync"])
            self.assertFalse(mismatch["ok"])
            self.assertTrue(mismatch["warning"])
            self.assertTrue(any("stage count" in w for w in mismatch["warnings"]))

            # 4) Invalid sidecar JSON.
            sidecar_path.write_text("{corrupt", encoding="utf-8")
            os.utime(sidecar_path, (now + 20, now + 20))
            invalid_sc = preflight.check_splash_wgpk_sync(repo, shaders_dir=shaders_dir)
            self.assertTrue(invalid_sc["sidecarPresent"])
            self.assertFalse(invalid_sc["sidecarValid"])
            self.assertFalse(invalid_sc["inSync"])
            self.assertFalse(invalid_sc["ok"])
            self.assertIn("invalid JSON", invalid_sc["errors"])

    def test_splash_wgpk_missing_pack_and_corrupt_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=False)

            # Missing shaders_dir path -> sanitized warning, zero path leakage.
            missing_dir = base / "DoesNotExist_SecretShaders"
            res_no_dir = preflight.check_splash_wgpk_sync(repo, shaders_dir=missing_dir)
            self.assertTrue(res_no_dir["shadersDirProvided"])
            self.assertFalse(res_no_dir["shadersDirExists"])
            self.assertFalse(res_no_dir["present"])
            self.assertFalse(res_no_dir["ok"])
            self.assertTrue(res_no_dir["warning"])
            self.assertIn("splash.wgsl.pack not found", res_no_dir["warnings"])
            self.assertNotIn("DoesNotExist_SecretShaders", json.dumps(res_no_dir))

            # Existing empty shaders_dir -> missing pack warning.
            shaders_dir = base / "empty_shaders"
            shaders_dir.mkdir()
            res_missing = preflight.check_splash_wgpk_sync(repo, shaders_dir=shaders_dir)
            self.assertTrue(res_missing["shadersDirExists"])
            self.assertFalse(res_missing["present"])
            self.assertFalse(res_missing["ok"])
            self.assertTrue(res_missing["warning"])
            self.assertIn("splash.wgsl.pack not found", res_missing["warnings"])

            pack_path = shaders_dir / "splash.wgsl.pack"

            # Truncated (< 16 bytes).
            pack_path.write_bytes(b"WGPK\x01\x00\x00\x00")
            trunc = preflight.inspect_wgpk_header(pack_path)
            self.assertFalse(trunc["headerValid"])
            self.assertTrue(any("truncated" in e for e in trunc["errors"]))

            # Invalid magic.
            pack_path.write_bytes(struct.pack("<4sIII", b"BAD!", 1, 0, 0))
            bad_magic = preflight.inspect_wgpk_header(pack_path)
            self.assertFalse(bad_magic["headerValid"])
            self.assertIn("invalid WGPK magic", bad_magic["errors"])

            # Unsupported version.
            pack_path.write_bytes(struct.pack("<4sIII", b"WGPK", 2, 0, 0))
            bad_ver = preflight.inspect_wgpk_header(pack_path)
            self.assertFalse(bad_ver["headerValid"])
            self.assertIn("unsupported WGPK version 2", bad_ver["errors"])

            # Entry table exceeds file size.
            pack_path.write_bytes(struct.pack("<4sIII", b"WGPK", 1, 100, 0))
            bad_table = preflight.inspect_wgpk_header(pack_path)
            self.assertFalse(bad_table["headerValid"])
            self.assertTrue(any("entry table" in e for e in bad_table["errors"]))

            # Invalid stage code (9).
            pack_path.write_bytes(_build_wgpk_v1_pack([(0x1, 9, b"fn x() {}")]))
            bad_stage = preflight.inspect_wgpk_header(pack_path)
            self.assertFalse(bad_stage["headerValid"])
            self.assertIn("invalid WGPK entry stage", bad_stage["errors"])

            # Out-of-bounds entry offset/length.
            oob_header = struct.pack("<4sIII", b"WGPK", 1, 1, 0)
            oob_entry = struct.pack("<QB3sII", 0x1, 1, b"\x00\x00\x00", 36, 999)
            pack_path.write_bytes(oob_header + oob_entry)
            oob = preflight.inspect_wgpk_header(pack_path)
            self.assertFalse(oob["headerValid"])
            self.assertIn("WGPK entry payload out of bounds", oob["errors"])

            # Valid 0-entry WGPK header -> headerValid=True, but splash sync warns 0 entries.
            pack_path.write_bytes(struct.pack("<4sIII", b"WGPK", 1, 0, 0))
            zero_entries = preflight.check_splash_wgpk_sync(repo, shaders_dir=shaders_dir)
            self.assertTrue(zero_entries["headerValid"])
            self.assertEqual(zero_entries["entryCount"], 0)
            self.assertFalse(zero_entries["ok"])
            self.assertTrue(any("0 shader entries" in w for w in zero_entries["warnings"]))

    def test_browser_detection_families_and_path_sanitization(self) -> None:
        # Chrome detected with path/username in raw version string -> sanitized cleanly.
        w_chrome, v_chrome = _make_fake_toolchains(
            {"google-chrome": r"Google Chrome 131.0.6778.86 (C:\Users\SecretDev\chrome.exe)"}
        )
        res_chrome = preflight.check_browser(which_fn=w_chrome, version_fn=v_chrome)
        self.assertTrue(res_chrome["available"])
        self.assertEqual(res_chrome["family"], "chrome")
        self.assertEqual(res_chrome["version"], "131.0.6778.86")
        self.assertTrue(res_chrome["browsers"]["chrome"]["available"])
        self.assertFalse(res_chrome["browsers"]["edge"]["available"])
        self.assertFalse(res_chrome["warning"])
        self.assertNotIn("SecretDev", json.dumps(res_chrome))
        self.assertNotIn("/fake/bin", json.dumps(res_chrome))

        # Edge fallback when Chrome absent.
        w_edge, v_edge = _make_fake_toolchains(
            {"msedge": "Microsoft Edge 131.0.2903.63 (/home/secretuser/edge/msedge)"}
        )
        res_edge = preflight.check_browser(which_fn=w_edge, version_fn=v_edge)
        self.assertTrue(res_edge["available"])
        self.assertEqual(res_edge["family"], "edge")
        self.assertEqual(res_edge["version"], "131.0.2903.63")
        self.assertNotIn("secretuser", json.dumps(res_edge))

        # Chromium fallback when Chrome & Edge absent.
        w_chromium, v_chromium = _make_fake_toolchains(
            {"chromium": "Chromium 130.0.6723.58"}
        )
        res_chromium = preflight.check_browser(which_fn=w_chromium, version_fn=v_chromium)
        self.assertTrue(res_chromium["available"])
        self.assertEqual(res_chromium["family"], "chromium")
        self.assertEqual(res_chromium["version"], "130.0.6723.58")

        # No browser detected -> warning emitted.
        w_none, v_none = _make_fake_toolchains({})
        res_none = preflight.check_browser(which_fn=w_none, version_fn=v_none)
        self.assertFalse(res_none["available"])
        self.assertIsNone(res_none["family"])
        self.assertIsNone(res_none["version"])
        self.assertTrue(res_none["warning"])
        self.assertEqual(len(res_none["warnings"]), 1)

    def test_memory_headroom_thresholds_and_sanitized_errors(self) -> None:
        gib = preflight.BYTES_PER_GIB

        # Exact 4.0 GiB boundary -> OK.
        exact = preflight.check_memory_headroom(
            min_free_ram_gb=4.0,
            memory_info_fn=lambda: (16 * gib, 4 * gib),
        )
        self.assertTrue(exact["checked"])
        self.assertTrue(exact["ok"])
        self.assertFalse(exact["lowMemory"])
        self.assertFalse(exact["warning"])
        self.assertEqual(exact["availableGb"], 4.0)
        self.assertEqual(exact["totalGb"], 16.0)

        # 1 byte below 4.0 GiB -> lowMemory warning (unrounded comparison).
        below = preflight.check_memory_headroom(
            min_free_ram_gb=4.0,
            memory_info_fn=lambda: (16 * gib, 4 * gib - 1),
        )
        self.assertTrue(below["checked"])
        self.assertFalse(below["ok"])
        self.assertTrue(below["lowMemory"])
        self.assertTrue(below["warning"])

        # Exception containing host path must never leak.
        def failing_mem() -> tuple[int, int] | None:
            raise OSError(r"failed reading C:\Users\TopSecret\meminfo")

        err_mem = preflight.check_memory_headroom(memory_info_fn=failing_mem)
        self.assertFalse(err_mem["checked"])
        self.assertFalse(err_mem["ok"])
        self.assertEqual(err_mem["error"], "memory info unavailable")
        self.assertNotIn("TopSecret", json.dumps(err_mem))

        # Negative threshold raises ValueError.
        with self.assertRaises(ValueError):
            preflight.check_memory_headroom(min_free_ram_gb=-0.5)

    def test_cli_shaders_dir_ram_flag_and_deterministic_sanitized_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = self._create_repo_skeleton(base, include_server=True)
            dist_dir = repo / "web" / "dist"
            dist_dir.mkdir(parents=True)
            (dist_dir / "runtime.js").write_text("export default {};\n", encoding="utf-8")
            (dist_dir / "runtime.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

            secret_shaders = base / "PrivateShadersDir_Secret"
            secret_shaders.mkdir()
            pack_path = secret_shaders / "splash.wgsl.pack"
            sidecar_path = secret_shaders / "splash.wgsl.pack.reflect.json"
            pack_path.write_bytes(_build_wgpk_v1_pack([(0xabc, 1, b"@vertex fn main() {}")]))
            sidecar_path.write_text(
                json.dumps(_make_valid_sidecar_doc([("0xabc", "vertex")])),
                encoding="utf-8",
            )
            now = time.time()
            os.utime(pack_path, (now - 2, now - 2))
            os.utime(sidecar_path, (now, now))

            env_clean = {k: v for k, v in os.environ.items() if not k.startswith("GTA6_")}
            cmd = [
                sys.executable,
                str(REPO_ROOT / "tools" / "preflight.py"),
                "--repo-root",
                str(repo),
                "--shaders-dir",
                str(secret_shaders),
                "--min-free-gb",
                "0",
                "--min-free-ram-gb",
                "0",
                "--skip-server",
                "--json",
            ]
            run1 = subprocess.run(cmd, capture_output=True, text=True, env=env_clean, check=False)
            run2 = subprocess.run(cmd, capture_output=True, text=True, env=env_clean, check=False)
            self.assertEqual(run1.returncode, 0, msg=run1.stderr)
            self.assertEqual(run2.returncode, 0, msg=run2.stderr)

            doc1 = json.loads(run1.stdout)
            doc2 = json.loads(run2.stdout)
            # Deterministic key order across runs.
            self.assertEqual(list(doc1.keys()), list(doc2.keys()))
            self.assertEqual(list(doc1["splashPack"].keys()), list(doc2["splashPack"].keys()))
            self.assertEqual(list(doc1["browser"].keys()), list(doc2["browser"].keys()))
            self.assertEqual(list(doc1["memory"].keys()), list(doc2["memory"].keys()))
            self.assertEqual(list(doc1["fastBoot"].keys()), list(doc2["fastBoot"].keys()))

            self.assertTrue(doc1["splashPack"]["ok"])
            self.assertTrue(doc1["splashPack"]["inSync"])
            self.assertEqual(doc1["splashPack"]["entryCount"], 1)
            self.assertIn("available", doc1["browser"])
            self.assertIn("family", doc1["browser"])
            self.assertIn("version", doc1["browser"])
            self.assertFalse(doc1["memory"]["lowMemory"])

            # Zero host path or username leakage in JSON output.
            for forbidden in (
                str(base),
                str(secret_shaders),
                "PrivateShadersDir_Secret",
                os.environ.get("USERNAME", ""),
                os.environ.get("USER", ""),
            ):
                if forbidden and len(forbidden) > 2:
                    self.assertNotIn(forbidden, run1.stdout)

            # High RAM threshold emits lowMemory warning but --strict still exits 0 when M1 is ready.
            proc_strict = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "preflight.py"),
                    "--repo-root",
                    str(repo),
                    "--shaders-dir",
                    str(secret_shaders),
                    "--min-free-ram-gb",
                    "999999",
                    "--json",
                    "--strict",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc_strict.returncode, 0, msg=proc_strict.stderr)
            strict_doc = json.loads(proc_strict.stdout)
            self.assertTrue(strict_doc["milestones"]["M1"]["ready"])
            if strict_doc["memory"]["checked"]:
                self.assertTrue(strict_doc["memory"]["lowMemory"])
                self.assertTrue(strict_doc["memory"]["warning"])

            # Negative --min-free-ram-gb is rejected by CLI parser.
            proc_neg_ram = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "preflight.py"),
                    "--repo-root",
                    str(repo),
                    "--min-free-ram-gb",
                    "-1",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertNotEqual(proc_neg_ram.returncode, 0)


if __name__ == "__main__":
    unittest.main()
