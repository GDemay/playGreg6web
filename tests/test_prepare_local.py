import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

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


if __name__ == "__main__":
    unittest.main()
