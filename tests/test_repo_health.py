import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import repo_health  # noqa: E402

REQUIRED_REPO_FILES: tuple[str, ...] = (
    repo_health.GPU_ABI_HEADER_REL,
    repo_health.GPU_TRACE_PY_REL,
    repo_health.WGPU_WORKER_JS_REL,
    repo_health.SMOKE_SCENE_JS_REL,
    repo_health.INPUT_BRIDGE_JS_REL,
    repo_health.HTTPFS_HEADER_REL,
    repo_health.HTTPFS_JS_REL,
    repo_health.WORKSTREAMS_MD_REL,
    repo_health.GITIGNORE_REL,
    repo_health.CARGO_TOML_REL,
    "recomp/Cargo.toml",
    "shaders/Cargo.toml",
    "tools/Cargo.toml",
)


def _copy_minimal_repo(dest: Path) -> Path:
    """Copy the checked files from the real repo into an isolated temp repo."""
    for rel in REQUIRED_REPO_FILES:
        src = REPO_ROOT / rel
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    (dest / repo_health.TESTS_DIR_REL).mkdir(parents=True, exist_ok=True)
    return dest


class TestRepoHealthRealRepo(unittest.TestCase):
    """Run the full verifier against the real repository to catch live ABI drift in CI."""

    def test_real_repo_passes_all_checks(self) -> None:
        report = repo_health.check_repo_health(REPO_ROOT, strict=True)
        self.assertTrue(
            report["ok"],
            msg=f"Real repo health check failed:\n{repo_health.format_human_report(report)}",
        )
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(report["warning_count"], 0)

        checks = report["checks"]
        gpu = checks["gpu_opcode_parity"]
        self.assertTrue(gpu["ok"])
        self.assertEqual(gpu["opcode_count"], 41)
        self.assertEqual(gpu["min_opcode"], 0)
        self.assertEqual(gpu["max_opcode"], 40)
        for target_rel in (
            repo_health.GPU_TRACE_PY_REL,
            repo_health.WGPU_WORKER_JS_REL,
            repo_health.SMOKE_SCENE_JS_REL,
        ):
            self.assertTrue(gpu["targets"][target_rel]["ok"])
            self.assertEqual(gpu["targets"][target_rel]["count"], 41)

        inp = checks["input_state_parity"]
        self.assertTrue(inp["ok"])
        self.assertEqual(inp["cpp_static_assert_bytes"], 40)
        self.assertEqual(inp["cpp_computed_bytes"], 40)
        self.assertEqual(inp["js_input_state_bytes"], 40)
        self.assertEqual(inp["cpp_field_offsets"], inp["js_field_offsets"])

        httpfs = checks["httpfs_mailbox_parity"]
        self.assertTrue(httpfs["ok"])
        self.assertEqual(httpfs["cpp_constants"]["mailbox_version"], 1)
        self.assertEqual(httpfs["cpp_constants"]["header_bytes"], 40)
        self.assertEqual(httpfs["cpp_constants"]["payload_bytes"], 65536)
        self.assertEqual(httpfs["cpp_offsets"], httpfs["js_offsets"])

        hyg = checks["tracked_hygiene"]
        self.assertTrue(hyg["ok"])
        self.assertEqual(hyg["forbidden_files"], [])
        self.assertEqual(hyg["oversized_binaries"], [])
        self.assertGreater(hyg["tracked_file_count"], 0)

        gi = checks["gitignore_hygiene"]
        self.assertTrue(gi["ok"])
        self.assertEqual(gi["missing_patterns"], [])
        self.assertEqual(gi["negated_patterns"], [])
        self.assertEqual(
            gi["protected_patterns"],
            list(repo_health.REQUIRED_GITIGNORE_PATTERNS),
        )
        self.assertGreaterEqual(gi["rule_count"], len(repo_health.REQUIRED_GITIGNORE_PATTERNS))

        inv = checks["test_inventory"]
        self.assertTrue(inv["ok"])
        self.assertGreater(inv["counts"]["mjs"], 0)
        self.assertGreater(inv["counts"]["py"], 0)
        self.assertGreater(inv["counts"]["cpp"], 0)
        self.assertEqual(
            inv["counts"]["total_test_files"],
            inv["counts"]["mjs"] + inv["counts"]["py"] + inv["counts"]["cpp"],
        )
        self.assertGreaterEqual(inv["counts"]["rust_crates"], 3)
        self.assertTrue({"recomp", "shaders", "tools"}.issubset(set(inv["rust_crates"])))

        readiness = report["fast_boot_readiness"]
        self.assertTrue(readiness["ready"])
        self.assertTrue(readiness["abi_parity_ok"])
        self.assertTrue(readiness["disk_hygiene_ok"])
        self.assertTrue(readiness["zero_game_files"])
        self.assertTrue(readiness["gitignore_protected"])
        self.assertTrue(readiness["test_inventory_ok"])
        self.assertEqual(readiness["test_suite_counts"], inv["counts"])

        ws = checks["workstreams_summary"]
        self.assertTrue(ws["ok"])
        self.assertEqual(len(ws["tracks"]), 5)
        self.assertGreater(ws["totals"]["total"], 0)
        self.assertEqual(
            ws["totals"]["done"] + ws["totals"]["in_progress"] + ws["totals"]["unclaimed"],
            ws["totals"]["total"],
        )

    def test_cli_json_is_valid_and_sanitized(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "repo_health.py"), "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        parsed = json.loads(proc.stdout)
        self.assertTrue(parsed["ok"])
        self.assertEqual(parsed["error_count"], 0)
        self.assertIn("gitignore_hygiene", parsed["checks"])
        self.assertIn("test_inventory", parsed["checks"])
        self.assertTrue(parsed["fast_boot_readiness"]["ready"])

        # Ensure zero host filesystem paths or current usernames leaked into JSON output
        raw_json = proc.stdout
        self.assertNotIn(str(REPO_ROOT), raw_json)
        self.assertNotIn(str(REPO_ROOT).replace("\\", "/"), raw_json)
        self.assertNotIn(str(Path.home()), raw_json)
        self.assertNotIn(str(Path.home()).replace("\\", "/"), raw_json)
        for username in repo_health._current_usernames():
            self.assertNotIn(username, raw_json)

    def test_cli_human_and_strict_mode_regressions(self) -> None:
        human_proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "repo_health.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(human_proc.returncode, 0, msg=human_proc.stderr or human_proc.stdout)
        self.assertIn("playgta6web repo health: PASS", human_proc.stdout)
        self.assertIn("GpuOpcode ABI parity", human_proc.stdout)
        self.assertIn(".gitignore hygiene", human_proc.stdout)
        self.assertIn("Test suite inventory", human_proc.stdout)
        self.assertIn("Fast-boot readiness: READY", human_proc.stdout)

        strict_proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "repo_health.py"), "--strict", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(strict_proc.returncode, 0, msg=strict_proc.stderr or strict_proc.stdout)
        parsed = json.loads(strict_proc.stdout)
        self.assertTrue(parsed["ok"])
        self.assertTrue(parsed["strict"])
        self.assertEqual(parsed["warning_count"], 0)


class TestRepoHealthSyntheticDrift(unittest.TestCase):
    """Test every check against synthetic drift & hygiene violation scenarios."""

    def test_mismatched_gpu_opcode_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            worker_path = repo / repo_health.WGPU_WORKER_JS_REL
            original = worker_path.read_text(encoding="utf-8")
            mutated = original.replace("DISPATCH_COMPUTE: 21,", "DISPATCH_COMPUTE: 99,")
            self.assertNotEqual(original, mutated)
            worker_path.write_text(mutated, encoding="utf-8")

            res = repo_health.check_gpu_opcode_parity(repo)
            self.assertFalse(res["ok"])
            self.assertFalse(res["targets"][repo_health.WGPU_WORKER_JS_REL]["ok"])
            self.assertEqual(
                res["targets"][repo_health.WGPU_WORKER_JS_REL]["mismatched"],
                [{"name": "DISPATCH_COMPUTE", "expected": 21, "actual": 99}],
            )
            self.assertTrue(res["targets"][repo_health.GPU_TRACE_PY_REL]["ok"])
            self.assertTrue(res["targets"][repo_health.SMOKE_SCENE_JS_REL]["ok"])

            overall = repo_health.check_repo_health(repo)
            self.assertFalse(overall["ok"])
            self.assertGreaterEqual(overall["error_count"], 1)

    def test_missing_gpu_opcode_in_python_and_smoke_scene_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            trace_path = repo / repo_health.GPU_TRACE_PY_REL
            trace_path.write_text(
                trace_path.read_text(encoding="utf-8").replace("    DispatchCompute = 21\n", ""),
                encoding="utf-8",
            )
            smoke_path = repo / repo_health.SMOKE_SCENE_JS_REL
            smoke_path.write_text(
                smoke_path.read_text(encoding="utf-8").replace("  DRAW_INDEXED: 18,\n", ""),
                encoding="utf-8",
            )

            res = repo_health.check_gpu_opcode_parity(repo)
            self.assertFalse(res["ok"])
            self.assertEqual(
                res["targets"][repo_health.GPU_TRACE_PY_REL]["missing"],
                ["DispatchCompute"],
            )
            self.assertEqual(
                res["targets"][repo_health.SMOKE_SCENE_JS_REL]["missing"],
                ["DRAW_INDEXED"],
            )

    def test_bad_input_state_size_and_offset_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            bridge_path = repo / repo_health.INPUT_BRIDGE_JS_REL
            text = bridge_path.read_text(encoding="utf-8")
            mutated = text.replace(
                "export const INPUT_STATE_BYTES = 40;",
                "export const INPUT_STATE_BYTES = 44;",
            ).replace("RIGHT_TRIGGER: 36,", "RIGHT_TRIGGER: 40,")
            bridge_path.write_text(mutated, encoding="utf-8")

            res = repo_health.check_input_state_parity(repo)
            self.assertFalse(res["ok"])
            self.assertEqual(res["cpp_static_assert_bytes"], 40)
            self.assertEqual(res["js_input_state_bytes"], 44)
            self.assertTrue(any("byte size drift" in e for e in res["errors"]))
            self.assertTrue(any("RIGHT_TRIGGER" in e for e in res["errors"]))

    def test_httpfs_mailbox_constant_and_offset_drift_fails_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            js_path = repo / repo_health.HTTPFS_JS_REL
            text = js_path.read_text(encoding="utf-8")
            mutated = (
                text.replace("const HTTPFS_PAYLOAD_BYTES = 64 * 1024;", "const HTTPFS_PAYLOAD_BYTES = 32 * 1024;")
                .replace("RESULT: 32,", "RESULT: 36,")
                .replace("ResponseReady: 2,", "ResponseReady: 7,")
            )
            js_path.write_text(mutated, encoding="utf-8")

            res = repo_health.check_httpfs_mailbox_parity(repo)
            self.assertFalse(res["ok"])
            self.assertTrue(any("payload_bytes" in e for e in res["errors"]))
            self.assertTrue(any("HttpfsOffsets.RESULT" in e for e in res["errors"]))
            self.assertTrue(any("HttpfsStatus.ResponseReady" in e for e in res["errors"]))

    def test_forbidden_rpf_and_env_tracked_file_fails_hygiene_git_and_walk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            (repo / "update").mkdir(parents=True, exist_ok=True)
            (repo / "update" / "dlc.rpf").write_bytes(b"RPF9dummy")
            (repo / ".env").write_text("GTA6_GAME_DIR=/secret\n", encoding="utf-8")

            # Fallback walk detection (non-git temp dir)
            walk_res = repo_health.check_tracked_hygiene(repo)
            self.assertFalse(walk_res["ok"])
            self.assertEqual(walk_res["source"], "walk")
            self.assertIn("update/dlc.rpf", walk_res["forbidden_files"])
            self.assertIn(".env", walk_res["forbidden_files"])

            # Git ls-files detection when tracked in a real git repo (force-added past .gitignore)
            if shutil.which("git"):
                subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
                subprocess.run(
                    ["git", "-C", str(repo), "add", "-f", "update/dlc.rpf", ".env"],
                    check=True,
                )
                git_res = repo_health.check_tracked_hygiene(repo)
                self.assertFalse(git_res["ok"])
                self.assertEqual(git_res["source"], "git")
                self.assertIn("update/dlc.rpf", git_res["forbidden_files"])
                self.assertIn(".env", git_res["forbidden_files"])

    def test_oversized_binary_blob_fails_hygiene(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            blob = repo / "assets_blob.dat"
            blob.write_bytes(b"\x00\x01\x02\x03" * 512)  # 2048 bytes binary

            res = repo_health.check_tracked_hygiene(repo, max_binary_bytes=1024)
            self.assertFalse(res["ok"])
            self.assertEqual(res["oversized_binaries"], [{"path": "assets_blob.dat", "bytes": 2048}])

            # Large UTF-8 text file over threshold should NOT be flagged as binary blob
            blob.unlink()
            text_file = repo / "large_source.rs"
            text_file.write_text("a" * 2048, encoding="utf-8")
            clean_res = repo_health.check_tracked_hygiene(repo, max_binary_bytes=1024)
            self.assertTrue(clean_res["ok"])

    def test_workstreams_summary_counting_and_strict_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            ws_path = repo / repo_health.WORKSTREAMS_MD_REL
            ws_path.write_text(
                "\n".join(
                    [
                        "# Workstreams",
                        "",
                        "## Track 1: Recompiler",
                        "- [x] **R1 — Done task** — `DONE` (PR #1)",
                        "  - Sub-bullet should not be counted",
                        "- [ ] **R2 — Active task** — `IN PROGRESS` (#2, @alice)",
                        "- [ ] **R3 — Open task** *(unclaimed)*",
                        "",
                        "## Track 2: Runtime",
                        "- [x] **O1 — Done task** — `DONE` (PR #3)",
                        "- [x] **O2 — Also done** — `DONE` (PR #4)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            res = repo_health.summarize_workstreams(repo)
            self.assertTrue(res["ok"])
            self.assertEqual(res["warnings"], [])
            self.assertEqual(len(res["tracks"]), 2)
            self.assertEqual(res["tracks"][0]["done"], 1)
            self.assertEqual(res["tracks"][0]["in_progress"], 1)
            self.assertEqual(res["tracks"][0]["unclaimed"], 1)
            self.assertEqual(res["tracks"][0]["total"], 3)
            self.assertEqual(res["tracks"][1]["done"], 2)
            self.assertEqual(res["tracks"][1]["in_progress"], 0)
            self.assertEqual(res["tracks"][1]["unclaimed"], 0)
            self.assertEqual(res["totals"]["total"], 5)
            self.assertEqual(res["totals"]["done"], 3)
            self.assertEqual(res["totals"]["in_progress"], 1)
            self.assertEqual(res["totals"]["unclaimed"], 1)
            self.assertEqual(res["totals"]["completion_pct"], 60.0)

            # Now introduce a checkbox/status mismatch warning and check --strict behavior
            ws_path.write_text(
                "\n".join(
                    [
                        "## Track 1: Recompiler",
                        "- [x] **R1 — Contradictory** — `IN PROGRESS` (#9)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            non_strict = repo_health.check_repo_health(repo, strict=False)
            self.assertTrue(non_strict["ok"])
            self.assertEqual(non_strict["warning_count"], 1)

            strict_res = repo_health.check_repo_health(repo, strict=True)
            self.assertFalse(strict_res["ok"])
            self.assertEqual(strict_res["warning_count"], 1)

    def test_cli_returns_nonzero_on_drift_and_sanitizes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            (repo / "game.xvd").write_bytes(b"msft-xvd")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "repo_health.py"),
                    "--repo-root",
                    str(repo),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertIn("game.xvd", payload["checks"]["tracked_hygiene"]["forbidden_files"])
            # Confirm temp host path is never leaked in JSON
            self.assertNotIn(str(repo), proc.stdout)
            self.assertNotIn(str(repo).replace("\\", "/"), proc.stdout)

    def test_gitignore_audit_detects_missing_and_commented_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            gi_path = repo / repo_health.GITIGNORE_REL
            gi_path.write_text(
                "\n".join(
                    [
                        "# local-data/",
                        "work/",
                        "*.rpf",
                        "*.wgsl.pack",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            res = repo_health.check_gitignore_hygiene(repo)
            self.assertFalse(res["ok"])
            self.assertEqual(res["missing_patterns"], ["local-data/", ".env", "*.pgim"])
            self.assertEqual(res["negated_patterns"], [])
            self.assertEqual(res["protected_patterns"], ["work/", "*.rpf", "*.wgsl.pack"])
            self.assertEqual(res["rule_count"], 3)

            overall = repo_health.check_repo_health(repo)
            self.assertFalse(overall["ok"])
            self.assertFalse(overall["fast_boot_readiness"]["gitignore_protected"])
            self.assertFalse(overall["fast_boot_readiness"]["disk_hygiene_ok"])
            self.assertFalse(overall["fast_boot_readiness"]["ready"])

    def test_gitignore_audit_detects_negated_root_anchored_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            gi_path = repo / repo_health.GITIGNORE_REL

            # Equivalent valid forms should pass
            gi_path.write_text(
                "\n".join(
                    [
                        "/local-data/",
                        "**/work",
                        "/.env",
                        "**/*.rpf",
                        "*.pgim",
                        "**/*.wgsl.pack",
                    ]
                ),
                encoding="utf-8",
            )
            ok_res = repo_health.check_gitignore_hygiene(repo)
            self.assertTrue(ok_res["ok"], msg=ok_res["errors"])

            # Negating a required pattern or root-anchoring /*.rpf must fail
            gi_path.write_text(
                "\n".join(
                    [
                        "local-data/",
                        "work/",
                        ".env",
                        "!.env",
                        "/*.rpf",
                        "*.pgim",
                        "*.wgsl.pack",
                        "!*.wgsl.pack",
                    ]
                ),
                encoding="utf-8",
            )
            bad_res = repo_health.check_gitignore_hygiene(repo)
            self.assertFalse(bad_res["ok"])
            self.assertEqual(bad_res["missing_patterns"], ["*.rpf"])
            self.assertEqual(bad_res["negated_patterns"], [".env", "*.wgsl.pack"])

            # Missing .gitignore file must fail
            gi_path.unlink()
            missing_res = repo_health.check_gitignore_hygiene(repo)
            self.assertFalse(missing_res["ok"])
            self.assertEqual(
                missing_res["missing_patterns"],
                list(repo_health.REQUIRED_GITIGNORE_PATTERNS),
            )

    def test_test_inventory_counts_suites_and_rust_crates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            tests_dir = repo / repo_health.TESTS_DIR_REL
            for idx in range(3):
                (tests_dir / f"test_web_{idx}.mjs").write_text("", encoding="utf-8")
            for idx in range(2):
                (tests_dir / f"test_tool_{idx}.py").write_text("", encoding="utf-8")
            for idx in range(4):
                (tests_dir / f"test_rt_{idx}.cpp").write_text("", encoding="utf-8")
            # Non-matching files and directories must not be counted
            (tests_dir / "helper.mjs").write_text("", encoding="utf-8")
            (tests_dir / "common.py").write_text("", encoding="utf-8")
            (tests_dir / "test_dir.mjs").mkdir()

            # Replace Cargo.toml with custom 2-crate workspace
            (repo / "crate_alpha").mkdir()
            (repo / "crate_alpha" / "Cargo.toml").write_text(
                '[package]\nname = "crate_alpha"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            (repo / "crate_beta").mkdir()
            (repo / "crate_beta" / "Cargo.toml").write_text(
                '[package]\nname = "crate_beta"\nversion = "0.1.0"\n',
                encoding="utf-8",
            )
            (repo / repo_health.CARGO_TOML_REL).write_text(
                '[workspace]\nmembers = ["crate_beta", "crate_alpha"]\n',
                encoding="utf-8",
            )

            inv = repo_health.summarize_test_inventory(repo)
            self.assertTrue(inv["ok"], msg=inv["errors"])
            self.assertEqual(inv["warnings"], [])
            self.assertEqual(
                inv["counts"],
                {
                    "mjs": 3,
                    "py": 2,
                    "cpp": 4,
                    "total_test_files": 9,
                    "rust_crates": 2,
                },
            )
            self.assertEqual(inv["rust_crates"], ["crate_alpha", "crate_beta"])

    def test_test_inventory_handles_missing_member_invalid_toml_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))

            # Missing workspace member Cargo.toml -> error
            (repo / repo_health.CARGO_TOML_REL).write_text(
                '[workspace]\nmembers = ["recomp", "ghost_crate"]\n',
                encoding="utf-8",
            )
            missing_member = repo_health.summarize_test_inventory(repo)
            self.assertFalse(missing_member["ok"])
            self.assertTrue(any("ghost_crate" in err for err in missing_member["errors"]))

            # Malformed Cargo.toml -> error
            (repo / repo_health.CARGO_TOML_REL).write_text(
                "[workspace\nbroken = true\n",
                encoding="utf-8",
            )
            bad_toml = repo_health.summarize_test_inventory(repo)
            self.assertFalse(bad_toml["ok"])
            self.assertTrue(any("invalid TOML" in err for err in bad_toml["errors"]))

            # Missing tests/ and Cargo.toml -> warnings only (fails --strict)
            (repo / repo_health.CARGO_TOML_REL).unlink()
            shutil.rmtree(repo / repo_health.TESTS_DIR_REL)
            warn_res = repo_health.summarize_test_inventory(repo)
            self.assertTrue(warn_res["ok"])
            self.assertEqual(len(warn_res["warnings"]), 2)
            strict_overall = repo_health.check_repo_health(repo, strict=True)
            self.assertFalse(strict_overall["ok"])
            self.assertGreaterEqual(strict_overall["warning_count"], 2)

    def test_json_sanitization_strips_host_paths_and_usernames(self) -> None:
        fake_user = "zxq_secret_dev_99"
        custom_root = "/srv/private-workspace-99/playgta6web"
        dirty_report = {
            "ok": False,
            "nested": {
                "win": f"C:\\Users\\{fake_user}\\playgta6web\\local-data\\game.rpf",
                "posix": f"/home/{fake_user}/builds/playgta6web/work/out.pgim",
                "unc": f"\\\\fileserver\\{fake_user}\\share\\x.wgsl.pack",
                "custom": f"{custom_root}/tools/repo_health.py failed for {fake_user}",
                "plain_user": f"owned by {fake_user} in CI",
            },
            "list_items": [f"see /Users/{fake_user}/Desktop/log.txt", 42, True, None],
        }

        with mock.patch.dict(os.environ, {"USERNAME": fake_user, "USER": fake_user}):
            cleaned = repo_health.sanitize_report(dirty_report, repo_root=custom_root)

        serialized = json.dumps(cleaned)
        self.assertNotIn(fake_user, serialized)
        self.assertNotIn(custom_root, serialized)
        self.assertNotIn("C:\\\\Users", serialized)
        self.assertNotIn("/home/", serialized)
        self.assertNotIn("/Users/", serialized)
        self.assertNotIn("fileserver", serialized)
        self.assertEqual(cleaned["list_items"][1:], [42, True, None])

        # Also verify end-to-end CLI --json scrubs injected host paths & mocked username
        with tempfile.TemporaryDirectory() as tmp:
            repo = _copy_minimal_repo(Path(tmp))
            ws_path = repo / repo_health.WORKSTREAMS_MD_REL
            ws_path.write_text(
                "\n".join(
                    [
                        f"## Track 1: Owned by {fake_user} at {repo}",
                        f"- [x] **R1 — Leak test** — `IN PROGRESS` ({repo}/secret)",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            env = dict(os.environ)
            env["USERNAME"] = fake_user
            env["USER"] = fake_user
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "repo_health.py"),
                    "--repo-root",
                    str(repo),
                    "--strict",
                    "--json",
                ],
                capture_output=True,
                text=True,
                env=env,
                check=False,
            )
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["warning_count"], 1)
            self.assertNotIn(fake_user, proc.stdout)
            self.assertNotIn(str(repo), proc.stdout)
            self.assertNotIn(str(repo).replace("\\", "/"), proc.stdout)
            self.assertNotIn(str(Path.home()), proc.stdout)
            self.assertNotIn(str(Path.home()).replace("\\", "/"), proc.stdout)


if __name__ == "__main__":
    unittest.main()
