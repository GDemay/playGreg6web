#!/usr/bin/env python3
"""Unit and CLI tests for tools/verify_browser_boot.py (Track 5 / W48)."""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_wasm_exports as wasm  # noqa: E402
import inspect_wgsl_pack as wgpk  # noqa: E402
import verify_browser_boot as vbb  # noqa: E402

SYNTHETIC_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <Identity Name="Synthetic.BootTitle" Publisher="CN=Synthetic" Version="1.0.0.0" />
  <TitleId>AABBCCDD</TitleId>
  <ExecutableList>
    <Executable Name="Game.exe" Id="Game" TargetDeviceFamily="Scarlett" />
  </ExecutableList>
  <ShellVisuals DefaultDisplayName="Synthetic Boot"
                SplashScreenImage="Assets\\xbs_splash.png"
                BackgroundColor="#000000" />
</Game>
"""

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"splash-bytes-0123456789ABCDEF"


def _full_import_coverage(_repo: Path) -> dict:
    return {
        "ok": True,
        "errors": [],
        "totals": {
            "total": 587,
            "shimmed": 580,
            "guest_recompiled": 7,
            "missing": 0,
            "effective_boot_pct": 100.0,
            "shimmed_pct": 98.81,
            "guest_recompiled_pct": 1.19,
            "missing_pct": 0.0,
        },
    }


def _partial_import_coverage(_repo: Path) -> dict:
    return {
        "ok": True,
        "errors": [],
        "totals": {
            "total": 587,
            "shimmed": 500,
            "guest_recompiled": 7,
            "missing": 80,
            "effective_boot_pct": 86.37,
            "shimmed_pct": 85.18,
            "guest_recompiled_pct": 1.19,
            "missing_pct": 13.63,
        },
    }


def _build_shader_artifacts(shaders_dir: Path) -> None:
    shaders_dir.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "hash": None,
            "stage": "vertex",
            "wgsl": "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }",
        },
        {
            "hash": None,
            "stage": "fragment",
            "wgsl": "@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }",
        },
    ]
    pack_bytes = wgpk.build_wgpk(entries)
    bundle = wgpk.parse_wgpk(pack_bytes)
    (shaders_dir / "fallback.wgsl.pack").write_bytes(pack_bytes)
    reflection = {
        "schema_version": 1,
        "shaders": {entry.hash_hex: {"stage": entry.stage_name} for entry in bundle.entries},
    }
    (shaders_dir / "reflection-manifest.json").write_text(
        json.dumps(reflection, indent=2),
        encoding="utf-8",
    )


def _build_recomp_artifacts(recomp_dir: Path, *, with_pgim: bool = True) -> None:
    recomp_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "playgta6-recomp-build",
        "schema_version": 1,
        "module": "main",
        "lift": {
            "functions": 10,
            "instructions": 100,
            "unsupported_instructions": 5,
            "shards": [
                {"index": 0, "functions": 4, "instructions": 40},
                {"index": 1, "functions": 6, "instructions": 60},
            ],
        },
        "discovery": {"functions": 10, "instructions": 100},
    }
    (recomp_dir / "build-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    program = {
        "schema": "playgta6-recomp-program",
        "schema_version": 1,
        "modules": [
            {
                "module": "main",
                "dll": "game.exe",
                "lift": {"functions": 10, "instructions": 100, "unsupported": 5},
            }
        ],
    }
    (recomp_dir / "program-manifest.json").write_text(json.dumps(program), encoding="utf-8")
    if with_pgim:
        (recomp_dir / "guest.pgim").write_bytes(b"PGIM" + b"\x00" * 32)


def _write_runtime_wasm(
    dist_dir: Path,
    *,
    kind: str = "smoke_shell",
) -> None:
    dist_dir.mkdir(parents=True, exist_ok=True)
    raw = wasm.build_valid_runtime_wasm()
    if kind == "linked_guest_runtime":
        # Prefer export-based classification, then pad past 1 MiB for honesty.
        module = wasm.parse_wasm_module(raw)
        exports = [(exp.name, exp.kind, exp.index) for exp in module.exports]
        exports.append(("pg6_guest_boot_run", "func", 0))
        # Rebuild with guest export + pad.
        base = wasm.build_valid_runtime_wasm()
        # Append custom section padding via raw bytes after a valid module is awkward;
        # pad file size beyond 1 MiB while keeping a valid prefix parse...
        # Better: build module with guest export using build_wasm_module helpers.
        types = []
        imports = []
        # memory + host imports
        imports.append(
            {
                "module": "env",
                "name": "memory",
                "kind": "memory",
                "min_pages": 256,
                "max_pages": 65536,
                "is_shared": True,
                "is_memory64": True,
            }
        )
        for spec in wasm.REQUIRED_HOST_IMPORTS:
            type_idx = len(types)
            types.append((spec.params_wasm64, spec.results_wasm64))
            imports.append(
                {
                    "module": spec.module,
                    "name": spec.name,
                    "kind": "func",
                    "type_index": type_idx,
                }
            )
        # local func type for guest boot export
        boot_type = len(types)
        types.append((("i64",), ("i32",)))
        raw = wasm.build_wasm_module(
            types=types,
            imports=imports,
            function_types=[boot_type],
            exports=[
                ("memory", "memory", 0),
                ("pg6_guest_boot_run", "func", len([i for i in imports if i["kind"] == "func"])),
            ],
            target_features=["+memory64", "+atomics", "+bulk-memory"],
        )
        # Ensure code section exists? function section without code may fail some validators
        # but check_wasm_exports only parses type/import/function/memory/export.
        if len(raw) < vbb.SMOKE_SHELL_MAX_BYTES:
            # Pad with a custom section filled with zeros (still valid wasm).
            pad_needed = vbb.SMOKE_SHELL_MAX_BYTES - len(raw) + 16
            name = wasm.encode_name("pad")
            payload = name + (b"\x00" * max(pad_needed - len(name), 1))
            raw = raw + wasm.encode_section(wasm.SECTION_CUSTOM, payload)
    (dist_dir / "runtime.wasm").write_bytes(raw)
    (dist_dir / "runtime.js").write_text("export default {};\n", encoding="utf-8")


def _create_repo_skeleton(base: Path, *, include_server: bool = True) -> Path:
    repo = base / "repo"
    (repo / "web").mkdir(parents=True)
    (repo / "web" / "index.html").write_text("<h1>boot</h1>", encoding="utf-8")
    (repo / "local-data").mkdir()
    (repo / "work").mkdir()
    if include_server:
        (repo / "server").mkdir(parents=True)
        shutil.copy2(REPO_ROOT / "server" / "serve_local.py", repo / "server" / "serve_local.py")
        # serve_local lazily loads inspect_game_assets from tools/; copy it too.
        (repo / "tools").mkdir(parents=True)
        shutil.copy2(
            REPO_ROOT / "tools" / "inspect_game_assets.py",
            repo / "tools" / "inspect_game_assets.py",
        )
    return repo


def _populate_complete_env(
    repo: Path,
    base: Path,
    *,
    wasm_kind: str = "linked_guest_runtime",
    with_game: bool = True,
) -> Path | None:
    game_dir = None
    if with_game:
        game_dir = base / "SyntheticGameInstall"
        game_dir.mkdir()
        (game_dir / "MicrosoftGame.Config").write_text(SYNTHETIC_CONFIG, encoding="utf-8")
        assets = game_dir / "Assets"
        assets.mkdir()
        (assets / "xbs_splash.png").write_bytes(PNG_BYTES)
        (game_dir / "Game.exe").write_bytes(b"MZ" + b"\x90" * 64)
        (game_dir / "helper.dll").write_bytes(b"MZ" + b"\x91" * 64)
        (game_dir / "levels.rpf").write_bytes(b"RPF9" + b"\x01" * 32)
        (repo / ".env").write_text(
            f"GTA6_GAME_DIR={game_dir}\nGTA6_DATA_DIR=local-data\nGTA6_WORK_DIR=work\n",
            encoding="utf-8",
        )

    _build_shader_artifacts(repo / "local-data" / "shaders")
    _build_recomp_artifacts(repo / "work" / "recomp_build")
    _write_runtime_wasm(repo / "web" / "dist", kind=wasm_kind)
    return game_dir


class TestVerifyBrowserBoot(unittest.TestCase):
    def test_missing_environment_fails_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp))
            report = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_partial_import_coverage,
            )
            self.assertFalse(report["ready_for_browser_splash"])
            self.assertFalse(report["ready_for_real_guest_boot"])
            self.assertEqual(report["gates"]["game_assets"]["status"], "skip")
            self.assertEqual(report["gates"]["import_coverage"]["status"], "fail")
            self.assertEqual(report["gates"]["recomp_artifacts"]["status"], "fail")
            self.assertEqual(report["gates"]["shader_pack"]["status"], "fail")
            self.assertEqual(report["gates"]["wasm64_runtime"]["status"], "fail")
            self.assertEqual(report["gates"]["local_server"]["status"], "skip")

    def test_partial_environment_splash_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _build_shader_artifacts(repo / "local-data" / "shaders")
            _write_runtime_wasm(repo / "web" / "dist", kind="smoke_shell")
            report = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_partial_import_coverage,
            )
            self.assertEqual(report["gates"]["shader_pack"]["status"], "pass")
            self.assertEqual(report["gates"]["wasm64_runtime"]["status"], "warn")
            self.assertEqual(report["gates"]["wasm64_runtime"]["details"]["kind"], "smoke_shell")
            self.assertFalse(report["ready_for_browser_splash"])
            self.assertFalse(report["ready_for_real_guest_boot"])

    def test_complete_environment_ready_for_real_guest_boot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _populate_complete_env(repo, base, wasm_kind="linked_guest_runtime")
            report = vbb.run_verification(
                repo,
                env={},
                skip_server=False,
                import_coverage_fn=_full_import_coverage,
            )
            self.assertEqual(report["gates"]["game_assets"]["status"], "pass")
            self.assertEqual(report["gates"]["import_coverage"]["status"], "pass")
            self.assertEqual(report["gates"]["recomp_artifacts"]["status"], "pass")
            self.assertEqual(report["gates"]["shader_pack"]["status"], "pass")
            self.assertEqual(report["gates"]["wasm64_runtime"]["status"], "pass")
            self.assertEqual(report["gates"]["wasm64_runtime"]["details"]["kind"], "linked_guest_runtime")
            self.assertEqual(report["gates"]["local_server"]["status"], "pass")
            self.assertTrue(report["ready_for_browser_splash"])
            self.assertTrue(report["ready_for_real_guest_boot"])
            self.assertIn("python server/serve_local.py", report["next_steps"])

    def test_smoke_shell_vs_linked_guest_distinction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base, include_server=False)
            _populate_complete_env(repo, base, wasm_kind="smoke_shell")
            smoke = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_full_import_coverage,
            )
            self.assertEqual(smoke["gates"]["wasm64_runtime"]["details"]["kind"], "smoke_shell")
            self.assertEqual(smoke["gates"]["wasm64_runtime"]["status"], "warn")
            self.assertTrue(smoke["ready_for_browser_splash"])
            self.assertFalse(smoke["ready_for_real_guest_boot"])

            _write_runtime_wasm(repo / "web" / "dist", kind="linked_guest_runtime")
            linked = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_full_import_coverage,
            )
            self.assertEqual(
                linked["gates"]["wasm64_runtime"]["details"]["kind"],
                "linked_guest_runtime",
            )
            self.assertEqual(linked["gates"]["wasm64_runtime"]["status"], "pass")
            self.assertTrue(linked["ready_for_real_guest_boot"])

    def test_classify_runtime_wasm_kind_helpers(self) -> None:
        self.assertEqual(
            vbb.classify_runtime_wasm_kind(byte_size=1000, export_names=["memory"]),
            "smoke_shell",
        )
        self.assertEqual(
            vbb.classify_runtime_wasm_kind(
                byte_size=1000,
                export_names=["pg6_guest_boot_run"],
            ),
            "linked_guest_runtime",
        )
        self.assertEqual(
            vbb.classify_runtime_wasm_kind(
                byte_size=vbb.SMOKE_SHELL_MAX_BYTES,
                export_names=["memory"],
            ),
            "linked_guest_runtime",
        )

    def test_ephemeral_server_gate_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _populate_complete_env(repo, base)
            gate = vbb.check_local_server_gate(
                repo,
                data_dir=repo / "local-data",
                game_dir=base / "SyntheticGameInstall",
                recomp_dir=repo / "work" / "recomp_build",
                skip_server=False,
            )
            self.assertEqual(gate.status, "pass")
            self.assertTrue(gate.details["coopValid"])
            self.assertTrue(gate.details["coepValid"])
            self.assertTrue(gate.details["range206Ok"])
            self.assertTrue(gate.details["statusEndpointOk"])
            self.assertTrue(gate.details["shadersOk"])
            self.assertTrue(gate.details["bootOk"])
            self.assertTrue(gate.details["gameSplashOk"])
            self.assertTrue(gate.details["http11_keep_alive"])

    def test_sanitization_strips_windows_and_posix_paths_and_usernames(self) -> None:
        samples = [
            r"C:\Users\AliceSecret\game\MicrosoftGame.Config",
            "/home/bobsecret/work/recomp_build/guest.pgim",
            r"\\nas\share\PrivateUser\dump.rpf",
            "D:/Users/CharlieSecret/local-data/shaders/fallback.wgsl.pack",
        ]
        # Force username detection by temporarily setting env USERNAME.
        old_user = os.environ.get("USERNAME")
        old_unix = os.environ.get("USER")
        try:
            os.environ["USERNAME"] = "AliceSecret"
            os.environ["USER"] = "bobsecret"
            for sample in samples:
                cleaned = vbb.sanitize_text(sample)
                self.assertNotIn("AliceSecret", cleaned)
                self.assertNotIn("bobsecret", cleaned)
                self.assertNotIn("CharlieSecret", cleaned)
                self.assertNotRegex(cleaned, r"[A-Za-z]:\\")
                self.assertNotRegex(cleaned, r"[A-Za-z]:/")
                self.assertNotIn("/home/", cleaned)
                self.assertNotIn("/Users/", cleaned)
        finally:
            if old_user is None:
                os.environ.pop("USERNAME", None)
            else:
                os.environ["USERNAME"] = old_user
            if old_unix is None:
                os.environ.pop("USER", None)
            else:
                os.environ["USER"] = old_unix

        # basename_label must never return absolute paths on either separator style.
        self.assertEqual(
            vbb.basename_label(r"C:\Users\AliceSecret\guest.pgim"),
            "guest.pgim",
        )
        self.assertEqual(
            vbb.basename_label("/home/bobsecret/work/guest.pgim"),
            "guest.pgim",
        )

        # Raw bytes never appear in sanitize_value output.
        self.assertEqual(vbb.sanitize_value(b"\x00asm"), "<bytes:4>")

    def test_json_cli_output_is_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            secret_game = base / "Users" / "TopSecretUser" / "PrivateGTA6"
            secret_game.mkdir(parents=True)
            (secret_game / "MicrosoftGame.Config").write_text(SYNTHETIC_CONFIG, encoding="utf-8")
            assets = secret_game / "Assets"
            assets.mkdir()
            (assets / "xbs_splash.png").write_bytes(PNG_BYTES)
            (secret_game / "Game.exe").write_bytes(b"MZ" + b"\x90" * 64)
            (repo / ".env").write_text(
                f"GTA6_GAME_DIR={secret_game}\nGTA6_DATA_DIR=local-data\nGTA6_WORK_DIR=work\n",
                encoding="utf-8",
            )
            _build_shader_artifacts(repo / "local-data" / "shaders")
            _build_recomp_artifacts(repo / "work" / "recomp_build")
            _write_runtime_wasm(repo / "web" / "dist", kind="linked_guest_runtime")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                # Monkeypatch import coverage inside CLI by calling library with injection,
                # then exercise --json formatting on the report.
                report = vbb.run_verification(
                    repo,
                    env={},
                    skip_server=True,
                    import_coverage_fn=_full_import_coverage,
                )
                print(json.dumps(report, indent=2))

            payload_text = stdout.getvalue()
            self.assertNotIn("TopSecretUser", payload_text)
            self.assertNotIn(str(secret_game), payload_text)
            # Absolute path fragments / usernames must not appear.
            self.assertNotRegex(payload_text, r"[A-Za-z]:\\\\Users")
            self.assertNotRegex(payload_text, r"[A-Za-z]:/Users")
            self.assertNotIn("TopSecretUser", payload_text)
            parsed = json.loads(payload_text)
            self.assertIn("gates", parsed)
            self.assertIn("ready_for_browser_splash", parsed)
            self.assertIn("ready_for_real_guest_boot", parsed)
            self.assertNotIn("\x00", payload_text)

    def test_cli_json_flag_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _populate_complete_env(repo, base, wasm_kind="linked_guest_runtime")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                code = vbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--skip-server",
                        "--json",
                    ]
                )
            # Splash artifacts are present; import coverage against the empty temp repo
            # fails, so real-guest boot is false but splash readiness can still be true.
            payload = json.loads(stdout.getvalue())
            self.assertIn("gates", payload)
            self.assertEqual(payload["gates"]["local_server"]["status"], "skip")
            self.assertEqual(payload["gates"]["import_coverage"]["status"], "fail")
            self.assertFalse(payload["ready_for_real_guest_boot"])
            self.assertEqual(code, 0 if payload["ready_for_browser_splash"] else 1)
            blob = stdout.getvalue()
            self.assertNotRegex(blob, r"[A-Za-z]:\\")
            self.assertNotRegex(blob, r"/Users/")
            self.assertNotRegex(blob, r"/home/")

    def test_gate_result_to_dict_shape(self) -> None:
        gate = vbb.GateResult(
            name="shader_pack",
            status="pass",
            summary="ok",
            details={"packNames": ["fallback.wgsl.pack"]},
            next_action=None,
        )
        as_dict = gate.to_dict()
        self.assertEqual(
            set(as_dict.keys()),
            {"name", "status", "summary", "details", "next_action"},
        )

    def test_recomp_artifacts_without_pgim_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            _build_recomp_artifacts(repo / "work" / "recomp_build", with_pgim=False)
            gate = vbb.check_recomp_artifacts_gate(repo, recomp_dir=repo / "work" / "recomp_build")
            self.assertEqual(gate.status, "fail")
            self.assertIn(".pgim", gate.summary)

    def test_recomp_artifacts_discovers_nested_program_pgims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            recomp_dir = repo / "work" / "recomp_program"
            recomp_dir.mkdir(parents=True)
            # Create >= 4 unrelated sibling directories before module dirs to guard
            # against regressions of the old non-recursive directory-count limit.
            for sibling in ("a_exports", "b_stubs", "c_tables", "d_logs", "exports"):
                (recomp_dir / sibling).mkdir()
                (recomp_dir / sibling / "placeholder.txt").write_text("ok\n", encoding="utf-8")

            program_manifest = {
                "schema": "playgta6-recomp-program",
                "schema_version": 1,
                "modules": [
                    {
                        "module": "main",
                        "dll": "game.exe",
                        "lift": {"functions": 120, "instructions": 1000, "unsupported": 10},
                    },
                    {
                        "module": "vcruntime140",
                        "dll": "vcruntime140.dll",
                        "lift": {"functions": 30, "instructions": 200, "unsupported": 0},
                    },
                ],
            }
            (recomp_dir / "program-manifest.json").write_text(
                json.dumps(program_manifest),
                encoding="utf-8",
            )
            for mod_name in ("main", "vcruntime140"):
                img_dir = recomp_dir / mod_name / "image"
                img_dir.mkdir(parents=True)
                (img_dir / "guest.pgim").write_bytes(b"PGIM" + b"\x00" * 32)
                (recomp_dir / mod_name / "build-manifest.json").write_text(
                    json.dumps(
                        {
                            "schema": "playgta6-recomp-build",
                            "schema_version": 1,
                            "module": mod_name,
                            "lift": {
                                "functions": 60,
                                "instructions": 500,
                                "unsupported_instructions": 5,
                                "shards": [{"index": 0}, {"index": 1}],
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            gate = vbb.check_recomp_artifacts_gate(repo, recomp_dir=recomp_dir)
            self.assertEqual(gate.status, "pass")
            self.assertEqual(gate.details["pgimCount"], 2)
            self.assertEqual(
                gate.details["pgimNames"],
                ["main/image/guest.pgim", "vcruntime140/image/guest.pgim"],
            )
            payload = json.dumps(gate.to_dict())
            self.assertNotRegex(payload, r"[A-Za-z]:\\")
            self.assertNotRegex(payload, r"[A-Za-z]:/")
            self.assertNotIn("/Users/", payload)
            self.assertNotIn("/home/", payload)

    def test_count_pgim_files_deterministic_and_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "recomp_out"
            (root / "zeta" / "image").mkdir(parents=True)
            (root / "alpha" / "nested" / "image").mkdir(parents=True)
            (root / "zeta" / "image" / "guest.pgim").write_bytes(b"PGIM")
            (root / "alpha" / "nested" / "image" / "guest.pgim").write_bytes(b"PGIM")
            (root / "root.pgim").write_bytes(b"PGIM")
            count, rels = vbb._count_pgim_files(root)
            self.assertEqual(count, 3)
            self.assertEqual(
                rels,
                [
                    "alpha/nested/image/guest.pgim",
                    "root.pgim",
                    "zeta/image/guest.pgim",
                ],
            )
            for rel in rels:
                self.assertNotIn("\\", rel)
                self.assertFalse(Path(rel).is_absolute())

    def test_import_coverage_100_percent_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            gate = vbb.check_import_coverage_gate(repo, audit_fn=_full_import_coverage)
            self.assertEqual(gate.status, "pass")
            self.assertIn("587/587", gate.summary)

    def test_shader_pack_sidecar_valid_reports_stage_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            shaders_dir = repo / "local-data" / "shaders"
            _build_shader_artifacts(shaders_dir)
            sidecar = {
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {
                    "0x1111111111111111": {"stage": "vertex", "bindings": []},
                    "0x2222222222222222": {"stage": "fragment", "bindings": []},
                    "0x3333333333333333": {"stage": "compute", "bindings": []},
                },
            }
            (shaders_dir / "fallback.wgsl.pack.reflect.json").write_text(
                json.dumps(sidecar, indent=2),
                encoding="utf-8",
            )
            gate = vbb.check_shader_pack_gate(repo, data_dir=repo / "local-data")
            self.assertEqual(gate.status, "pass")
            self.assertTrue(gate.details["sidecar_present"])
            self.assertTrue(gate.details["sidecar_valid"])
            self.assertEqual(gate.details["sidecar_stage_count"], 3)
            payload = json.dumps(gate.to_dict())
            self.assertNotRegex(payload, r"[A-Za-z]:\\")
            self.assertNotIn("/Users/", payload)
            self.assertNotIn("/home/", payload)

    def test_shader_pack_sidecar_missing_stays_passing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            _build_shader_artifacts(repo / "local-data" / "shaders")
            gate = vbb.check_shader_pack_gate(repo, data_dir=repo / "local-data")
            self.assertEqual(gate.status, "pass")
            self.assertFalse(gate.details["sidecar_present"])
            self.assertFalse(gate.details["sidecar_valid"])
            self.assertEqual(gate.details["sidecar_stage_count"], 0)

    def test_shader_pack_sidecar_malformed_reports_invalid_without_failing_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            shaders_dir = repo / "local-data" / "shaders"
            _build_shader_artifacts(shaders_dir)
            sidecar_path = shaders_dir / "fallback.wgsl.pack.reflect.json"

            # Malformed JSON syntax.
            sidecar_path.write_text('{"schema": "playgta6-stage-reflection", ', encoding="utf-8")
            gate = vbb.check_shader_pack_gate(repo, data_dir=repo / "local-data")
            self.assertNotEqual(gate.status, "fail")
            self.assertTrue(gate.details["sidecar_present"])
            self.assertFalse(gate.details["sidecar_valid"])
            self.assertEqual(gate.details["sidecar_stage_count"], 0)
            payload = json.dumps(gate.to_dict())
            self.assertNotRegex(payload, r"[A-Za-z]:\\")

            # Structurally invalid stages field (list instead of dict).
            sidecar_path.write_text(
                json.dumps({"schema": "playgta6-stage-reflection", "version": 1, "stages": []}),
                encoding="utf-8",
            )
            gate_bad_shape = vbb.check_shader_pack_gate(repo, data_dir=repo / "local-data")
            self.assertNotEqual(gate_bad_shape.status, "fail")
            self.assertTrue(gate_bad_shape.details["sidecar_present"])
            self.assertFalse(gate_bad_shape.details["sidecar_valid"])
            self.assertEqual(gate_bad_shape.details["sidecar_stage_count"], 0)

    def test_boot_options_valid_and_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            recomp_dir = repo / "work" / "recomp_build"
            _build_recomp_artifacts(recomp_dir)

            # Absent boot-options.json passes without error.
            gate_absent = vbb.check_recomp_artifacts_gate(repo, recomp_dir=recomp_dir)
            self.assertEqual(gate_absent.status, "pass")
            self.assertFalse(gate_absent.details["boot_options_present"])
            self.assertFalse(gate_absent.details["boot_options_valid"])

            # Valid boot-options.json passes with boot_options_present & boot_options_valid.
            (recomp_dir / "boot-options.json").write_text(
                json.dumps({"verify_window_marks": True, "max_wall_ms": 300000}),
                encoding="utf-8",
            )
            gate_valid = vbb.check_recomp_artifacts_gate(repo, recomp_dir=recomp_dir)
            self.assertEqual(gate_valid.status, "pass")
            self.assertTrue(gate_valid.details["boot_options_present"])
            self.assertTrue(gate_valid.details["boot_options_valid"])

    def test_boot_options_malformed_fails_cleanly(self) -> None:
        cases = [
            "{not valid json",
            json.dumps(["not", "an", "object"]),
            json.dumps({"verify_window_marks": "yes"}),
            json.dumps({"max_wall_ms": -5}),
            json.dumps({"custom_nested": {"bad": True}}),
        ]
        for raw_content in cases:
            with self.subTest(raw_content=raw_content):
                with tempfile.TemporaryDirectory() as tmp:
                    repo = _create_repo_skeleton(Path(tmp), include_server=False)
                    recomp_dir = repo / "work" / "recomp_build"
                    _build_recomp_artifacts(recomp_dir)
                    (recomp_dir / "boot-options.json").write_text(raw_content, encoding="utf-8")

                    gate = vbb.check_recomp_artifacts_gate(repo, recomp_dir=recomp_dir)
                    self.assertEqual(gate.status, "fail")
                    self.assertTrue(gate.details["boot_options_present"])
                    self.assertFalse(gate.details["boot_options_valid"])
                    payload = json.dumps(gate.to_dict())
                    self.assertNotRegex(payload, r"[A-Za-z]:\\")
                    self.assertNotIn("/Users/", payload)
                    self.assertNotIn("/home/", payload)

    def test_http11_keep_alive_probe_detects_non_persistent_server(self) -> None:
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class _ClosingHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, format: str, *args: object) -> None:
                return None

            def do_GET(self) -> None:
                body = b'{"ok":true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)

            def do_HEAD(self) -> None:
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()

        httpd = HTTPServer(("127.0.0.1", 0), _ClosingHandler)
        port = int(httpd.server_address[1])
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertFalse(vbb._probe_http11_keep_alive("127.0.0.1", port))
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)

    # ------------------------------------------------------------------
    # T76: --min-fps & --require-non-black boot-report verification
    # ------------------------------------------------------------------

    def test_min_fps_direct_present_fps_avg_pass_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            report_path = repo / "work" / "boot-report.json"

            # Direct present_fps_avg at exact 30.0 threshold passes.
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "playgta6-boot-report",
                        "schema_version": 2,
                        "present_fps_avg": 30.0,
                        "summary": {"nonBlackCanvasFrames": 5},
                    }
                ),
                encoding="utf-8",
            )
            gate_pass = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                min_fps=30.0,
            )
            self.assertEqual(gate_pass.status, "pass")
            self.assertTrue(gate_pass.details["min_fps_ok"])
            self.assertEqual(gate_pass.details["present_fps_avg"], 30.0)
            self.assertEqual(gate_pass.details["present_fps_source"], "present_fps_avg")
            self.assertEqual(gate_pass.details["bootReportLabel"], "work/boot-report.json")

            # Slightly below threshold (29.996 would round to 30.00 if compared after rounding).
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "playgta6-boot-report",
                        "schema_version": 2,
                        "summary": {"present_fps_avg": 29.996},
                    }
                ),
                encoding="utf-8",
            )
            gate_below = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                min_fps=30.0,
            )
            self.assertEqual(gate_below.status, "fail")
            self.assertFalse(gate_below.details["min_fps_ok"])
            self.assertIn("--min-fps", gate_below.summary)

            # Explicit 0.0 present_fps_avg is treated as 0 fps (not missing) and fails cleanly.
            report_path.write_text(
                json.dumps({"present_fps_avg": 0.0}),
                encoding="utf-8",
            )
            gate_zero = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                min_fps=30.0,
            )
            self.assertEqual(gate_zero.status, "fail")
            self.assertFalse(gate_zero.details["min_fps_ok"])
            self.assertEqual(gate_zero.details["present_fps_avg"], 0.0)
            self.assertEqual(gate_zero.details["present_fps_source"], "present_fps_avg")

    def test_min_fps_legacy_report_computed_from_present_span(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            report_path = repo / "work" / "legacy-report.json"

            # Legacy report without present_fps_avg: 60 presents over 2000ms = 30.0 fps -> pass.
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "playgta6-boot-report",
                        "schema_version": 1,
                        "counters": {"presents": 60},
                        "first_present_ms": 1000.0,
                        "last_present_ms": 3000.0,
                    }
                ),
                encoding="utf-8",
            )
            gate_pass = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                min_fps=30.0,
            )
            self.assertEqual(gate_pass.status, "pass")
            self.assertTrue(gate_pass.details["min_fps_ok"])
            self.assertEqual(gate_pass.details["present_fps_avg"], 30.0)
            self.assertEqual(gate_pass.details["present_fps_source"], "computed_present_span")
            self.assertEqual(gate_pass.details["present_span_ms"], 2000.0)

            # Legacy v2 report with presents list wall_ms samples and low FPS: 20 / 2.0s = 10.0 fps -> fail.
            report_path.write_text(
                json.dumps(
                    {
                        "schema": "playgta6-boot-report",
                        "schema_version": 2,
                        "counters": {"presents": 20},
                        "presents": [
                            {"wall_ms": 500.0},
                            {"wall_ms": 1500.0},
                            {"wall_ms": 2500.0},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            gate_fail = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                min_fps=30.0,
            )
            self.assertEqual(gate_fail.status, "fail")
            self.assertFalse(gate_fail.details["min_fps_ok"])
            self.assertEqual(gate_fail.details["present_fps_avg"], 10.0)
            self.assertEqual(gate_fail.details["present_fps_source"], "computed_present_span")

            # Legacy report with zero span (last_present_ms == first_present_ms) fails cleanly without ZeroDivisionError.
            report_path.write_text(
                json.dumps(
                    {
                        "counters": {"presents": 1},
                        "first_present_ms": 1000.0,
                        "last_present_ms": 1000.0,
                    }
                ),
                encoding="utf-8",
            )
            gate_zero_span = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                min_fps=30.0,
            )
            self.assertEqual(gate_zero_span.status, "fail")
            self.assertFalse(gate_zero_span.details["min_fps_ok"])
            self.assertIsNone(gate_zero_span.details["present_fps_avg"])

            # Legacy report with 0 presents and no timestamps fails cleanly.
            report_path.write_text(
                json.dumps({"counters": {"presents": 0}}),
                encoding="utf-8",
            )
            gate_no_presents = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                min_fps=30.0,
            )
            self.assertEqual(gate_no_presents.status, "fail")
            self.assertFalse(gate_no_presents.details["min_fps_ok"])
            self.assertIn("0 presents", gate_no_presents.summary)

    def test_require_non_black_pass_and_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            report_path = repo / "work" / "boot-report.json"

            # Pass via nonBlackCanvasFrames > 0 (even when first_visible_frame_s is null).
            report_path.write_text(
                json.dumps(
                    {
                        "first_visible_frame_s": None,
                        "summary": {"capturedFrames": 10, "nonBlackCanvasFrames": 4},
                    }
                ),
                encoding="utf-8",
            )
            gate_nb = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                require_non_black=True,
            )
            self.assertEqual(gate_nb.status, "pass")
            self.assertTrue(gate_nb.details["non_black_ok"])
            self.assertEqual(gate_nb.details["nonBlackCanvasFrames"], 4)

            # Pass via first_visible_frame_s is not None (even without nonBlackCanvasFrames).
            report_path.write_text(
                json.dumps({"first_visible_frame_s": 1.425, "summary": {}}),
                encoding="utf-8",
            )
            gate_fvf = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                require_non_black=True,
            )
            self.assertEqual(gate_fvf.status, "pass")
            self.assertTrue(gate_fvf.details["non_black_ok"])
            self.assertEqual(gate_fvf.details["first_visible_frame_s"], 1.425)

            # Fail when all-black capture (nonBlackCanvasFrames == 0 and first_visible_frame_s is None).
            report_path.write_text(
                json.dumps(
                    {
                        "first_visible_frame_s": None,
                        "summary": {"capturedFrames": 12, "nonBlackCanvasFrames": 0},
                    }
                ),
                encoding="utf-8",
            )
            gate_black = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                require_non_black=True,
            )
            self.assertEqual(gate_black.status, "fail")
            self.assertFalse(gate_black.details["non_black_ok"])
            self.assertIn("all-black", gate_black.summary)

            # Fail when both fields are absent.
            report_path.write_text(json.dumps({"summary": {}}), encoding="utf-8")
            gate_missing_fields = vbb.check_boot_report_gate(
                repo,
                boot_report=report_path,
                require_non_black=True,
            )
            self.assertEqual(gate_missing_fields.status, "fail")
            self.assertFalse(gate_missing_fields.details["non_black_ok"])

    def test_boot_report_auto_discovery_and_run_dir_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            run_a = repo / "work" / "browser-boot" / "runs" / "run_old"
            run_b = repo / "work" / "browser-boot" / "runs" / "run_new"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            old_file = run_a / "boot-report.json"
            new_file = run_b / "boot-report.json"
            old_file.write_text(
                json.dumps({"present_fps_avg": 5.0, "summary": {"nonBlackCanvasFrames": 0}}),
                encoding="utf-8",
            )
            new_file.write_text(
                json.dumps({"present_fps_avg": 35.0, "summary": {"nonBlackCanvasFrames": 8}}),
                encoding="utf-8",
            )
            os.utime(old_file, (1_700_000_000, 1_700_000_000))
            os.utime(new_file, (1_700_000_100, 1_700_000_100))

            # Auto-discovery picks newest run under work/browser-boot/runs/.
            gate_auto = vbb.check_boot_report_gate(
                repo,
                min_fps=30.0,
                require_non_black=True,
            )
            self.assertEqual(gate_auto.status, "pass")
            self.assertEqual(gate_auto.details["sourceMode"], "discovered")
            self.assertEqual(
                gate_auto.details["bootReportLabel"],
                "work/browser-boot/runs/run_new/boot-report.json",
            )

            # Passing a run directory resolves <dir>/boot-report.json.
            gate_dir = vbb.check_boot_report_gate(
                repo,
                boot_report=run_a,
                min_fps=30.0,
                require_non_black=True,
            )
            self.assertEqual(gate_dir.status, "fail")
            self.assertFalse(gate_dir.details["min_fps_ok"])
            self.assertFalse(gate_dir.details["non_black_ok"])

    def test_cli_min_fps_and_require_non_black_exit_codes_and_json_sanitization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base)
            _populate_complete_env(repo, base, wasm_kind="linked_guest_runtime")

            # Store boot report outside repo under a secret-looking host path and embed
            # host paths + UNC paths inside the report JSON itself to test strict sanitization.
            secret_dir = base / "Users" / "TopSecretUser" / "PrivateCaptures"
            secret_dir.mkdir(parents=True)
            good_report = secret_dir / "boot-report.json"
            good_report.write_text(
                json.dumps(
                    {
                        "schema": "playgta6-boot-report",
                        "schema_version": 2,
                        "present_fps_avg": 32.5,
                        "first_visible_frame_s": 1.25,
                        "message": r"captured at C:\Users\TopSecretUser\PrivateCaptures and \\nas\share\TopSecretUser\cap",
                        "capture_frames_dir": "/home/TopSecretUser/frames",
                        "summary": {"capturedFrames": 10, "nonBlackCanvasFrames": 7},
                    }
                ),
                encoding="utf-8",
            )

            # 1) Passing boot report -> exit code 0, sanitized JSON with zero host paths.
            stdout_pass = io.StringIO()
            with redirect_stdout(stdout_pass):
                code_pass = vbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--skip-server",
                        "--boot-report",
                        str(good_report),
                        "--min-fps",
                        "30",
                        "--require-non-black",
                        "--json",
                    ]
                )
            blob_pass = stdout_pass.getvalue()
            payload_pass = json.loads(blob_pass)
            self.assertEqual(code_pass, 0)
            self.assertTrue(payload_pass["ok"])
            self.assertTrue(payload_pass["boot_report_ok"])
            self.assertEqual(payload_pass["gates"]["boot_report"]["status"], "pass")
            self.assertEqual(
                payload_pass["gates"]["boot_report"]["details"]["bootReportLabel"],
                "boot-report.json",
            )
            self.assertNotIn("TopSecretUser", blob_pass)
            self.assertNotIn("PrivateCaptures", blob_pass)
            self.assertNotIn(str(secret_dir), blob_pass)
            self.assertNotRegex(blob_pass, r"[A-Za-z]:\\")
            self.assertNotRegex(blob_pass, r"[A-Za-z]:/")
            self.assertNotIn("/Users/", blob_pass)
            self.assertNotIn("/home/", blob_pass)
            self.assertNotIn("//nas/", blob_pass)

            # 2) Failing all-black / low-FPS report -> exit code 1, sanitized JSON.
            bad_report = secret_dir / "bad-boot-report.json"
            bad_report.write_text(
                json.dumps(
                    {
                        "schema": "playgta6-boot-report",
                        "schema_version": 2,
                        "counters": {"presents": 15},
                        "first_present_ms": 1000.0,
                        "last_present_ms": 2000.0,
                        "first_visible_frame_s": None,
                        "summary": {"capturedFrames": 8, "nonBlackCanvasFrames": 0},
                    }
                ),
                encoding="utf-8",
            )
            stdout_fail = io.StringIO()
            with redirect_stdout(stdout_fail):
                code_fail = vbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--skip-server",
                        "--boot-report",
                        str(bad_report),
                        "--min-fps",
                        "30",
                        "--require-non-black",
                        "--json",
                    ]
                )
            blob_fail = stdout_fail.getvalue()
            payload_fail = json.loads(blob_fail)
            self.assertEqual(code_fail, 1)
            self.assertFalse(payload_fail["ok"])
            self.assertFalse(payload_fail["boot_report_ok"])
            self.assertEqual(payload_fail["gates"]["boot_report"]["status"], "fail")
            self.assertFalse(payload_fail["gates"]["boot_report"]["details"]["min_fps_ok"])
            self.assertFalse(payload_fail["gates"]["boot_report"]["details"]["non_black_ok"])
            self.assertNotIn("TopSecretUser", blob_fail)
            self.assertNotIn("PrivateCaptures", blob_fail)
            self.assertNotRegex(blob_fail, r"[A-Za-z]:\\")
            self.assertNotRegex(blob_fail, r"[A-Za-z]:/")
            self.assertNotIn("/Users/", blob_fail)
            self.assertNotIn("/home/", blob_fail)

            # 3) Missing boot report under secret path -> fails cleanly with zero host paths.
            missing_report = secret_dir / "missing-report.json"
            stdout_missing = io.StringIO()
            with redirect_stdout(stdout_missing):
                code_missing = vbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--skip-server",
                        "--boot-report",
                        str(missing_report),
                        "--min-fps",
                        "30",
                        "--json",
                    ]
                )
            blob_missing = stdout_missing.getvalue()
            payload_missing = json.loads(blob_missing)
            self.assertEqual(code_missing, 1)
            self.assertFalse(payload_missing["boot_report_ok"])
            self.assertEqual(payload_missing["gates"]["boot_report"]["status"], "fail")
            self.assertNotIn("TopSecretUser", blob_missing)
            self.assertNotIn("PrivateCaptures", blob_missing)
            self.assertNotRegex(blob_missing, r"[A-Za-z]:\\")
            self.assertNotRegex(blob_missing, r"[A-Za-z]:/")

            # 4) Text output includes boot_report gate & boot_report_ok without host paths.
            text_out = vbb.format_report_text(payload_pass)
            self.assertIn("boot_report", text_out)
            self.assertIn("boot_report_ok", text_out)
            self.assertNotIn("TopSecretUser", text_out)

    def test_existing_default_report_shape_unchanged_without_boot_report_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _create_repo_skeleton(base, include_server=False)
            _populate_complete_env(repo, base, wasm_kind="linked_guest_runtime")
            report = vbb.run_verification(
                repo,
                env={},
                skip_server=True,
                import_coverage_fn=_full_import_coverage,
            )
            self.assertNotIn("boot_report", report["gates"])
            self.assertNotIn("boot_report_ok", report)
            self.assertTrue(report["ok"])

    def test_invalid_min_fps_and_malformed_boot_report_fail_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _create_repo_skeleton(Path(tmp), include_server=False)
            bad_json = repo / "work" / "malformed.json"
            bad_json.write_text("{not valid json", encoding="utf-8")

            gate_bad_json = vbb.check_boot_report_gate(
                repo,
                boot_report=bad_json,
                min_fps=30.0,
            )
            self.assertEqual(gate_bad_json.status, "fail")
            self.assertTrue(gate_bad_json.details["reportPresent"])
            self.assertFalse(gate_bad_json.details["reportValid"])

            for invalid_fps in (-1.0, float("nan"), float("inf")):
                gate_bad_fps = vbb.check_boot_report_gate(
                    repo,
                    boot_report=bad_json,
                    min_fps=invalid_fps,
                )
                self.assertEqual(gate_bad_fps.status, "fail")
                self.assertFalse(gate_bad_fps.details["min_fps_ok"])

            # CLI rejects negative --min-fps with usage exit code 2.
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as ctx:
                    vbb.main(["--min-fps", "-5"])
            self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
