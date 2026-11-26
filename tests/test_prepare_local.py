import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import inspect_wgsl_pack as wgpk  # noqa: E402
import prepare_local  # noqa: E402


class TestPrepareLocal(unittest.TestCase):
    def _setup_workspace(
        self,
        base: Path,
        *,
        create_game_dir: bool = True,
        write_dotenv: bool = True,
        data_dir_setting: str = "local-data",
        work_dir_setting: str = "work",
        game_dir_name: str = "PrivateUser_TopSecret_GTA6_Dump",
    ) -> tuple[Path, Path]:
        repo = base / "repo"
        repo.mkdir(parents=True, exist_ok=True)
        game_dir = base / game_dir_name
        if create_game_dir:
            game_dir.mkdir(parents=True, exist_ok=True)
            (game_dir / "MicrosoftGame.Config").write_text("<Game/>", encoding="utf-8")
            (game_dir / "GTA6.exe").write_bytes(b"MZ")
            (game_dir / "levels.rpf").write_bytes(b"RPF9")

        if write_dotenv:
            lines = [
                f'GTA6_GAME_DIR="{game_dir}"',
                f"GTA6_DATA_DIR={data_dir_setting}",
                f"GTA6_WORK_DIR={work_dir_setting}",
            ]
            (repo / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")

        return repo, game_dir

    def test_missing_game_dir_returns_clean_error_with_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()

            called = []

            def fail_if_called(cmd, cwd, env):
                called.append(cmd)
                return prepare_local.CommandResult(returncode=0)

            report = prepare_local.run_pipeline(repo, env={}, runner=fail_if_called)

            self.assertFalse(report["ok"])
            self.assertEqual(report["errorCode"], "missing_game_dir")
            self.assertIn('python tools/local_env.py --set-game "<path>"', report["error"])
            self.assertEqual(report["hint"], 'python tools/local_env.py --set-game "<path>"')
            self.assertEqual(report["steps"], [])
            self.assertEqual(called, [])

    def test_nonexistent_game_dir_returns_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, _missing_game = self._setup_workspace(base, create_game_dir=False)

            report = prepare_local.run_pipeline(repo, env={})

            self.assertFalse(report["ok"])
            self.assertEqual(report["errorCode"], "invalid_game_dir")
            self.assertIn('python tools/local_env.py --set-game "<path>"', report["error"])
            self.assertFalse(report["config"]["gameDirExists"])
            self.assertTrue(report["config"]["gameDirConfigured"])
            # Must not leak the raw temp path in error text.
            self.assertNotIn(str(base), json.dumps(report))

    def test_build_plan_default_and_skip_incomplete_and_single_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, game_dir = self._setup_workspace(base)
            cfg = prepare_local.resolve_config(repo, env={})

            plan = prepare_local.build_pipeline_plan(cfg, step="all", skip_incomplete=True)
            self.assertEqual([s.name for s in plan], ["scan", "manifest", "bootset", "recomp", "shaders"])

            by_name = {s.name: s for s in plan}

            # scan uses --dir <game_dir> --json
            self.assertEqual(
                by_name["scan"].command,
                (
                    "cargo",
                    "run",
                    "-p",
                    "playgta6-tools",
                    "--bin",
                    "playgta6-manifest",
                    "--",
                    "scan",
                    "--dir",
                    str(game_dir.resolve()),
                    "--json",
                ),
            )

            # manifest uses --game <game_dir> --out <data_dir>/data-manifest.json --skip-incomplete
            self.assertEqual(
                by_name["manifest"].command,
                (
                    "cargo",
                    "run",
                    "-p",
                    "playgta6-tools",
                    "--bin",
                    "playgta6-manifest",
                    "--",
                    "manifest",
                    "--game",
                    str(game_dir.resolve()),
                    "--out",
                    str((repo / "local-data" / "data-manifest.json").resolve()),
                    "--skip-incomplete",
                ),
            )

            # bootset uses --game <game_dir> --out <data_dir>/bootset.json --skip-incomplete
            self.assertEqual(
                by_name["bootset"].command,
                (
                    "cargo",
                    "run",
                    "-p",
                    "playgta6-tools",
                    "--bin",
                    "playgta6-manifest",
                    "--",
                    "bootset",
                    "--game",
                    str(game_dir.resolve()),
                    "--out",
                    str((repo / "local-data" / "bootset.json").resolve()),
                    "--skip-incomplete",
                ),
            )

            # recomp writes into gitignored work/imports_required.md
            self.assertEqual(
                by_name["recomp"].command,
                (
                    "cargo",
                    "run",
                    "-p",
                    "playgta6-recomp",
                    "--",
                    "inspect",
                    "--from-env",
                    "--markdown",
                    str((repo / "work" / "imports_required.md").resolve()),
                ),
            )

            # Without --skip-incomplete, manifest & bootset omit that flag.
            plan_no_skip = prepare_local.build_pipeline_plan(cfg, step="manifest", skip_incomplete=False)
            self.assertEqual(len(plan_no_skip), 1)
            self.assertEqual(plan_no_skip[0].name, "manifest")
            self.assertNotIn("--skip-incomplete", plan_no_skip[0].command)

            # Single step selection for bootset / recomp / scan
            for step_name in ("scan", "bootset", "recomp", "shaders"):
                single = prepare_local.build_pipeline_plan(cfg, step=step_name)
                self.assertEqual([s.name for s in single], [step_name])

    def test_conditional_shaders_step_skipped_and_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, _game_dir = self._setup_workspace(base)
            cfg = prepare_local.resolve_config(repo, env={})

            # 1) No work/spirv directory -> cleanly skipped
            plan_missing = prepare_local.build_pipeline_plan(cfg, step="shaders")
            self.assertEqual(plan_missing[0].skipped, "no local SPIR-V directory found")
            self.assertEqual(plan_missing[0].command, ())

            # 2) work/spirv exists but has no .spv files -> cleanly skipped
            spirv_dir = repo / "work" / "spirv"
            spirv_dir.mkdir(parents=True)
            (spirv_dir / "README.txt").write_text("empty", encoding="utf-8")
            plan_empty = prepare_local.build_pipeline_plan(cfg, step="shaders")
            self.assertIsNotNone(plan_empty[0].skipped)
            self.assertIn(".spv", plan_empty[0].skipped)
            self.assertEqual(plan_empty[0].command, ())

            # 3) work/spirv contains a nested .spv file -> included with exact convert command
            nested = spirv_dir / "vs"
            nested.mkdir()
            (nested / "triangle.spv").write_bytes(b"\x03\x02\x23\x07")
            plan_ready = prepare_local.build_pipeline_plan(cfg, step="shaders")
            self.assertIsNone(plan_ready[0].skipped)
            self.assertEqual(
                plan_ready[0].command,
                (
                    "cargo",
                    "run",
                    "-p",
                    "playgta6-shaders",
                    "--",
                    "convert",
                    "--spirv",
                    str(spirv_dir.resolve()),
                    "--out",
                    str((repo / "local-data" / "shaders").resolve()),
                ),
            )

    def test_safety_guard_rejects_tracked_and_external_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            external_out = base / "outside_repo_data"
            external_out.mkdir()

            unsafe_cases = [
                ("docs", "work"),
                ("local-data", "docs"),
                ("web/dist", "work"),
                (".", "work"),
                ("local-data/../docs", "work"),
                ("local-data", "work/../tools"),
                (str(external_out), "work"),
                ("local-data", str(external_out)),
            ]

            for idx, (data_setting, work_setting) in enumerate(unsafe_cases):
                case_base = base / f"case_{idx}"
                repo, _game_dir = self._setup_workspace(
                    case_base,
                    data_dir_setting=data_setting,
                    work_dir_setting=work_setting,
                )
                cfg = prepare_local.resolve_config(repo, env={})

                with self.assertRaises(prepare_local.PipelineSafetyError, msg=f"case {(data_setting, work_setting)}"):
                    prepare_local.build_pipeline_plan(cfg)

                called = []
                report = prepare_local.run_pipeline(
                    repo,
                    env={},
                    runner=lambda cmd, cwd, env: called.append(cmd) or prepare_local.CommandResult(0),
                )
                self.assertFalse(report["ok"], msg=f"case {(data_setting, work_setting)}")
                self.assertEqual(report["errorCode"], "unsafe_output_path")
                self.assertEqual(called, [])

            # Nested subdirectories inside local-data/ and work/ are allowed.
            ok_base = base / "case_ok_nested"
            repo_ok, _ = self._setup_workspace(
                ok_base,
                data_dir_setting="local-data/custom",
                work_dir_setting="work/session1",
            )
            cfg_ok = prepare_local.resolve_config(repo_ok, env={})
            plan_ok = prepare_local.build_pipeline_plan(cfg_ok)
            self.assertEqual(len(plan_ok), 5)

    def test_dry_run_makes_zero_subprocess_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, _game_dir = self._setup_workspace(base)
            spirv_dir = repo / "work" / "spirv"
            spirv_dir.mkdir(parents=True)
            (spirv_dir / "shader.spv").write_bytes(b"\x03\x02\x23\x07")

            invocations = []

            def unexpected_runner(cmd, cwd, env):
                invocations.append(cmd)
                raise AssertionError("dry-run must never invoke runner")

            report = prepare_local.run_pipeline(
                repo,
                env={},
                dry_run=True,
                skip_incomplete=True,
                runner=unexpected_runner,
            )

            self.assertTrue(report["ok"])
            self.assertTrue(report["dryRun"])
            self.assertEqual(invocations, [])
            self.assertEqual(report["summary"]["total"], 5)
            self.assertEqual(report["summary"]["planned"], 5)
            self.assertEqual(report["summary"]["succeeded"], 0)
            self.assertEqual(report["summary"]["failed"], 0)
            self.assertTrue(all(s["status"] == "planned" for s in report["steps"]))

    def test_mocked_runner_execution_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, game_dir = self._setup_workspace(base)

            recorded_calls: list[tuple[list[str], Path, str]] = []

            def fake_runner(cmd: list[str], cwd: Path, env: dict[str, str]) -> prepare_local.CommandResult:
                recorded_calls.append((cmd, cwd, env.get("GTA6_GAME_DIR", "")))
                return prepare_local.CommandResult(returncode=0, stdout=f"ok: {' '.join(cmd[:4])}")

            report = prepare_local.run_pipeline(repo, env={}, runner=fake_runner)

            self.assertTrue(report["ok"])
            self.assertIsNone(report["failedStep"])
            # 4 runnable steps executed, shaders skipped because work/spirv does not exist.
            self.assertEqual(len(recorded_calls), 4)
            self.assertEqual(report["summary"]["succeeded"], 4)
            self.assertEqual(report["summary"]["skipped"], 1)
            self.assertEqual(report["summary"]["failed"], 0)

            for _cmd, cwd, env_game in recorded_calls:
                self.assertEqual(cwd, repo.resolve())
                self.assertEqual(env_game, str(game_dir.resolve()))

            shaders_step = next(s for s in report["steps"] if s["name"] == "shaders")
            self.assertEqual(shaders_step["status"], "skipped")
            self.assertEqual(shaders_step["skipped"], "no local SPIR-V directory found")

    def test_mocked_runner_step_failure_stops_pipeline_and_reports_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, game_dir = self._setup_workspace(base)
            spirv_dir = repo / "work" / "spirv"
            spirv_dir.mkdir(parents=True)
            (spirv_dir / "shader.spv").write_bytes(b"\x03\x02\x23\x07")

            called_steps: list[str] = []

            def failing_runner(cmd: list[str], cwd: Path, env: dict[str, str]) -> prepare_local.CommandResult:
                if "scan" in cmd:
                    called_steps.append("scan")
                    return prepare_local.CommandResult(returncode=0, stdout='{"ok": true}')
                if "manifest" in cmd:
                    called_steps.append("manifest")
                    return prepare_local.CommandResult(
                        returncode=2,
                        stderr=f"Manifest generation failed reading {game_dir}\\broken.rpf",
                    )
                called_steps.append("unexpected")
                return prepare_local.CommandResult(returncode=0)

            report = prepare_local.run_pipeline(repo, env={}, runner=failing_runner)

            self.assertFalse(report["ok"])
            self.assertEqual(report["errorCode"], "step_failed")
            self.assertEqual(report["failedStep"], "manifest")
            self.assertEqual(called_steps, ["scan", "manifest"])
            self.assertEqual(report["summary"]["succeeded"], 1)
            self.assertEqual(report["summary"]["failed"], 1)
            self.assertEqual(report["summary"]["notRun"], 3)

            by_name = {s["name"]: s for s in report["steps"]}
            self.assertEqual(by_name["scan"]["status"], "succeeded")
            self.assertEqual(by_name["manifest"]["status"], "failed")
            self.assertEqual(by_name["manifest"]["returncode"], 2)
            self.assertIn("<GTA6_GAME_DIR>/broken.rpf", by_name["manifest"]["stderr"])
            self.assertEqual(by_name["bootset"]["status"], "not_run")
            self.assertEqual(by_name["recomp"]["status"], "not_run")
            self.assertEqual(by_name["shaders"]["status"], "not_run")

    def test_json_redacts_host_paths_and_usernames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, game_dir = self._setup_workspace(
                base,
                game_dir_name="AliceSecret_XboxSeries_GTA6",
            )
            spirv_dir = repo / "work" / "spirv"
            spirv_dir.mkdir(parents=True)
            (spirv_dir / "main.spv").write_bytes(b"\x03\x02\x23\x07")

            fake_env = {
                "USERNAME": "AliceSecret",
                "USER": "AliceSecret",
            }

            def noisy_runner(cmd: list[str], cwd: Path, env: dict[str, str]) -> prepare_local.CommandResult:
                return prepare_local.CommandResult(
                    returncode=0,
                    stdout=(
                        f"Processed {game_dir} -> {repo / 'local-data'} "
                        f"and {repo / 'work'} for user AliceSecret at C:\\Users\\AliceSecret\\secret.bin "
                        f"and /home/AliceSecret/private.txt"
                    ),
                )

            report = prepare_local.run_pipeline(repo, env=fake_env, runner=noisy_runner)
            serialized = json.dumps(report)

            # Expected placeholders present
            self.assertIn("<GTA6_GAME_DIR>", serialized)
            self.assertIn("<GTA6_DATA_DIR>/data-manifest.json", serialized)
            self.assertIn("<GTA6_DATA_DIR>/bootset.json", serialized)
            self.assertIn("<GTA6_WORK_DIR>/imports_required.md", serialized)
            self.assertIn("<GTA6_WORK_DIR>/spirv", serialized)
            self.assertIn("<GTA6_DATA_DIR>/shaders", serialized)

            # Zero leakage of raw paths or usernames
            for forbidden in (
                str(base),
                str(base.resolve()),
                str(repo),
                str(repo.resolve()),
                str(game_dir),
                str(game_dir.resolve()),
                "AliceSecret",
                "AliceSecret_XboxSeries_GTA6",
                os.environ.get("USERNAME", ""),
                os.environ.get("USER", ""),
            ):
                if forbidden and len(forbidden) >= 3:
                    self.assertNotIn(forbidden, serialized)

    def test_cli_dry_run_json_and_missing_game_dir_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, _game_dir = self._setup_workspace(base)
            env_clean = {k: v for k, v in os.environ.items() if not k.startswith("GTA6_")}

            # --dry-run --json exits 0 with clean JSON and zero host-path leakage
            proc_dry = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "prepare_local.py"),
                    "--repo-root",
                    str(repo),
                    "--dry-run",
                    "--skip-incomplete",
                    "--json",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc_dry.returncode, 0, msg=proc_dry.stderr)
            self.assertEqual(proc_dry.stderr.strip(), "")
            parsed = json.loads(proc_dry.stdout)
            self.assertTrue(parsed["ok"])
            self.assertTrue(parsed["dryRun"])
            self.assertEqual(parsed["summary"]["planned"], 4)
            self.assertEqual(parsed["summary"]["skipped"], 1)
            self.assertNotIn(str(base), proc_dry.stdout)
            self.assertNotIn(str(base.resolve()), proc_dry.stdout)

            # Human-readable --dry-run exits 0
            proc_text = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "prepare_local.py"),
                    "--repo-root",
                    str(repo),
                    "--dry-run",
                    "--step",
                    "manifest",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc_text.returncode, 0, msg=proc_text.stderr)
            self.assertIn("DRY RUN", proc_text.stdout)
            self.assertIn("[PLAN] manifest", proc_text.stdout)

            # Unconfigured repo exits non-zero and prints setup hint
            empty_repo = base / "empty_repo"
            empty_repo.mkdir()
            proc_err = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "prepare_local.py"),
                    "--repo-root",
                    str(empty_repo),
                    "--json",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertNotEqual(proc_err.returncode, 0)
            parsed_err = json.loads(proc_err.stdout)
            self.assertFalse(parsed_err["ok"])
            self.assertEqual(parsed_err["errorCode"], "missing_game_dir")
            self.assertIn('python tools/local_env.py --set-game "<path>"', parsed_err["error"])

    def test_pack_version_and_pack_shaders_plan_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, _game_dir = self._setup_workspace(base)
            spirv_dir = repo / "work" / "spirv"
            spirv_dir.mkdir(parents=True)
            (spirv_dir / "main.spv").write_bytes(b"\x03\x02\x23\x07")
            cfg = prepare_local.resolve_config(repo, env={})

            # Default keeps exact legacy command without --pack / --pack-version.
            default_plan = prepare_local.build_pipeline_plan(cfg, step="shaders")
            self.assertNotIn("--pack", default_plan[0].command)
            self.assertNotIn("--pack-version", default_plan[0].command)

            # --pack-version 2 appends --pack and --pack-version 2.
            v2_plan = prepare_local.build_pipeline_plan(cfg, step="shaders", pack_version="2")
            expected_pack = (repo / "local-data" / "shaders" / "shaders.wgsl.pack").resolve()
            self.assertEqual(
                v2_plan[0].command[-4:],
                ("--pack", str(expected_pack), "--pack-version", "2"),
            )
            self.assertIn(expected_pack, v2_plan[0].outputs)

            # --pack-shaders with default auto also appends --pack and --pack-version auto.
            auto_pack_plan = prepare_local.build_pipeline_plan(cfg, step="shaders", pack_shaders=True)
            self.assertEqual(
                auto_pack_plan[0].command[-4:],
                ("--pack", str(expected_pack), "--pack-version", "auto"),
            )

            # Invalid pack_version rejected cleanly.
            with self.assertRaises(prepare_local.PipelineValidationError) as ctx:
                prepare_local.build_pipeline_plan(cfg, step="shaders", pack_version="3")
            self.assertEqual(ctx.exception.code, "invalid_pack_version")

            # CLI --pack-version 2 --dry-run --json includes redacted pack path.
            env_clean = {k: v for k, v in os.environ.items() if not k.startswith("GTA6_")}
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "prepare_local.py"),
                    "--repo-root",
                    str(repo),
                    "--step",
                    "shaders",
                    "--pack-version",
                    "2",
                    "--dry-run",
                    "--json",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            parsed = json.loads(proc.stdout)
            self.assertTrue(parsed["ok"])
            cmd = parsed["steps"][0]["command"]
            self.assertIn("--pack", cmd)
            self.assertIn("<GTA6_DATA_DIR>/shaders/shaders.wgsl.pack", cmd)
            self.assertIn("--pack-version", cmd)
            self.assertIn("2", cmd)
            self.assertNotIn(str(base), proc.stdout)

    def test_validate_shader_pack_header_v1_v2_and_over_4gib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            v1_path = base / "v1.wgsl.pack"
            v1_path.write_bytes(
                wgpk.build_wgpk([(0x1111, wgpk.ShaderStage.VERTEX, "@vertex fn vs() {}")], version=1)
            )
            meta_v1 = prepare_local.validate_shader_pack_header(v1_path)
            self.assertTrue(meta_v1["ok"])
            self.assertEqual(meta_v1["version"], 1)
            self.assertEqual(meta_v1["entry_bytes"], 20)
            self.assertFalse(meta_v1["is_v2"])
            self.assertEqual(meta_v1["entry_count"], 1)

            v2_path = base / "v2.wgsl.pack"
            v2_path.write_bytes(
                wgpk.build_wgpk([(0x2222, wgpk.ShaderStage.FRAGMENT, "@fragment fn fs() {}")], version=2)
            )
            meta_v2 = prepare_local.validate_shader_pack_header(v2_path)
            self.assertTrue(meta_v2["ok"])
            self.assertEqual(meta_v2["version"], 2)
            self.assertEqual(meta_v2["entry_bytes"], 24)
            self.assertTrue(meta_v2["is_v2"])
            self.assertEqual(meta_v2["entry_count"], 1)

            # Simulated >4 GiB v2 pack via mocked get_file_size.
            huge_path = base / "huge_v2.wgsl.pack"
            offset_64 = 0x1_5000_0000
            payload_len = 256
            header = struct.pack("<4sIII", b"WGPK", 2, 1, 0)
            entry = struct.pack("<QB3sIQ", 0xABCD, 3, b"\x00\x00\x00", payload_len, offset_64)
            huge_path.write_bytes(header + entry)
            with mock.patch.object(
                prepare_local, "get_file_size", return_value=offset_64 + payload_len
            ):
                meta_huge = prepare_local.validate_shader_pack_header(huge_path)
            self.assertTrue(meta_huge["ok"])
            self.assertEqual(meta_huge["version"], 2)
            self.assertTrue(meta_huge["is_v2"])
            self.assertEqual(meta_huge["size_bytes"], offset_64 + payload_len)

            # Unsupported version 3 rejected.
            v3_path = base / "v3.wgsl.pack"
            v3_path.write_bytes(struct.pack("<4sIII", b"WGPK", 3, 0, 0))
            meta_v3 = prepare_local.validate_shader_pack_header(v3_path)
            self.assertFalse(meta_v3["ok"])
            self.assertEqual(meta_v3["version"], 3)
            self.assertIsNone(meta_v3["entry_bytes"])

    def test_validate_reflect_sidecar_v1_v2_v3_and_rejects_writable_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            valid_v3 = base / "good.wgsl.pack.reflect.json"
            valid_v3.write_text(
                json.dumps(
                    {
                        "schema": "playgta6-stage-reflection",
                        "version": 3,
                        "stages": {
                            "0x0000000000000001": {
                                "stage": "fragment",
                                "bindings": [
                                    {
                                        "group": 0,
                                        "binding": 0,
                                        "kind": "uniform",
                                        "source": {"user_sgpr": 2, "root_descriptor": True, "table_byte_offset": 32},
                                    },
                                    {
                                        "group": 0,
                                        "binding": 1,
                                        "kind": "read-only-storage",
                                        "source": {"unresolved": True},
                                    },
                                ],
                            },
                            "0x0000000000000002": {
                                "stage": "compute",
                                "mesh_output": {"max_vertices": 64, "max_primitives": 64, "vertex_words": 8},
                                "bindings": [],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            ok_res = prepare_local.validate_reflect_sidecar_file(valid_v3)
            self.assertTrue(ok_res["ok"])
            self.assertEqual(ok_res["version"], 3)
            self.assertEqual(ok_res["stage_count"], 2)
            self.assertEqual(ok_res["unresolved_source_stages"], 1)
            self.assertEqual(ok_res["unresolved_source_bindings"], 1)
            self.assertEqual(ok_res["mesh_output_stages"], 1)

            bad_v3 = base / "bad.wgsl.pack.reflect.json"
            bad_v3.write_text(
                json.dumps(
                    {
                        "schema": "playgta6-stage-reflection",
                        "version": 3,
                        "stages": {
                            "0x0000000000000003": {
                                "stage": "compute",
                                "bindings": [
                                    {
                                        "group": 0,
                                        "binding": 0,
                                        "kind": "storage",
                                        "source": {"unresolved": True},
                                    }
                                ],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            bad_res = prepare_local.validate_reflect_sidecar_file(bad_v3)
            self.assertFalse(bad_res["ok"])
            self.assertTrue(any("unresolved" in e for e in bad_res["errors"]))

    def test_stage_shader_bundle_v2_and_v3_copy_dry_run_safety_and_sanitization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo, _game_dir = self._setup_workspace(base)
            src_dir = base / "AliceSecret_ShaderBuild"
            src_dir.mkdir(parents=True)

            pack_bytes = wgpk.build_wgpk(
                [
                    (0xAAA1, wgpk.ShaderStage.VERTEX, "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"),
                    (0xBBB2, wgpk.ShaderStage.FRAGMENT, "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"),
                ],
                version=2,
            )
            pack_src = src_dir / "game_v2.wgsl.pack"
            pack_src.write_bytes(pack_bytes)

            reflect_doc = {
                "schema": "playgta6-stage-reflection",
                "version": 3,
                "stages": {
                    "0x000000000000aaa1": {
                        "stage": "vertex",
                        "bindings": [
                            {"group": 0, "binding": 0, "kind": "uniform", "source": {"unresolved": True}},
                        ],
                    }
                },
            }
            reflect_src = src_dir / "game_v2.wgsl.pack.reflect.json"
            reflect_src.write_text(json.dumps(reflect_doc), encoding="utf-8")

            fake_env = {"USERNAME": "AliceSecret", "USER": "AliceSecret"}
            cfg = prepare_local.resolve_config(repo, env=fake_env)

            # 1) dry_run=True validates without copying files.
            dry_res = prepare_local.stage_shader_bundle(pack_src, cfg, dry_run=True, env=fake_env)
            self.assertTrue(dry_res["ok"])
            self.assertTrue(dry_res["dryRun"])
            self.assertFalse(dry_res["staged"])
            self.assertEqual(dry_res["pack"]["version"], 2)
            self.assertTrue(dry_res["pack"]["is_v2"])
            self.assertEqual(dry_res["reflect"]["version"], 3)
            self.assertFalse((repo / "local-data" / "shaders" / "game_v2.wgsl.pack").exists())

            # 2) Real atomic copy stages both pack and auto-discovered sibling reflect.json.
            real_res = prepare_local.stage_shader_bundle(pack_src, cfg, dry_run=False, env=fake_env)
            self.assertTrue(real_res["ok"])
            self.assertTrue(real_res["staged"])
            staged_pack = repo / "local-data" / "shaders" / "game_v2.wgsl.pack"
            staged_reflect = repo / "local-data" / "shaders" / "game_v2.wgsl.pack.reflect.json"
            self.assertTrue(staged_pack.is_file())
            self.assertTrue(staged_reflect.is_file())
            self.assertEqual(staged_pack.read_bytes(), pack_bytes)
            self.assertEqual(json.loads(staged_reflect.read_text(encoding="utf-8")), reflect_doc)

            # 3) Strict path/username sanitization in returned dict.
            serialized = json.dumps(real_res)
            self.assertIn("<GTA6_DATA_DIR>/shaders/game_v2.wgsl.pack", serialized)
            self.assertNotIn("AliceSecret", serialized)
            self.assertNotIn(str(base), serialized)
            self.assertNotIn(str(src_dir), serialized)

            # 4) Escaping dest_dir rejected with PipelineSafetyError.
            with self.assertRaises(prepare_local.PipelineSafetyError):
                prepare_local.stage_shader_bundle(pack_src, cfg, dest_dir=repo / "docs")
            with self.assertRaises(prepare_local.PipelineSafetyError):
                prepare_local.stage_shader_bundle(pack_src, cfg, dest_dir=base / "outside")

            # 5) Standalone CLI --stage-pack works even when GTA6_GAME_DIR is NOT set.
            empty_repo = base / "repo_no_game"
            empty_repo.mkdir()
            env_clean = {k: v for k, v in os.environ.items() if not k.startswith("GTA6_")}
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "prepare_local.py"),
                    "--repo-root",
                    str(empty_repo),
                    "--stage-pack",
                    str(pack_src),
                    "--json",
                ],
                capture_output=True,
                text=True,
                env=env_clean,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            cli_out = json.loads(proc.stdout)
            self.assertTrue(cli_out["ok"])
            self.assertTrue(cli_out["stagedShaderBundle"]["staged"])
            self.assertTrue((empty_repo / "local-data" / "shaders" / "game_v2.wgsl.pack").is_file())
            self.assertTrue((empty_repo / "local-data" / "shaders" / "game_v2.wgsl.pack.reflect.json").is_file())
            self.assertNotIn(str(base), proc.stdout)
            self.assertNotIn("AliceSecret", proc.stdout)


if __name__ == "__main__":
    unittest.main()
