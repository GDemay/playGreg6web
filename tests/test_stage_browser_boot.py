#!/usr/bin/env python3
"""Unit and CLI tests for tools/stage_browser_boot.py (Track 5 / W51, W66, W119)."""

from __future__ import annotations

import http.client
import io
import json
import os
import re
import struct
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "server"))

import check_wasm_exports as wasm  # noqa: E402
import serve_local  # noqa: E402
import stage_browser_boot as sbb  # noqa: E402


def _make_repo_skeleton(base: Path) -> Path:
    repo = base / "repo"
    (repo / "web").mkdir(parents=True)
    (repo / "web" / "index.html").write_text("<h1>boot</h1>", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "runtime").mkdir()
    (repo / "work").mkdir()
    (repo / "local-data").mkdir()
    return repo


def _populate_program_dir(program_dir: Path) -> None:
    """Create a realistic multi-module build-program tree with allowlisted & excluded files."""
    program_dir.mkdir(parents=True, exist_ok=True)
    (program_dir / "program-manifest.json").write_text(
        json.dumps({"schema": "playgta6-recomp-program", "modules": ["main", "helper"]}),
        encoding="utf-8",
    )
    (program_dir / "pg6_program.cpp").write_text("// excluded cpp\n", encoding="utf-8")
    (program_dir / "notes.txt").write_text("excluded text\n", encoding="utf-8")
    (program_dir / "debug.json").write_text('{"excluded": true}\n', encoding="utf-8")

    main_dir = program_dir / "main"
    (main_dir / "image").mkdir(parents=True)
    (main_dir / "shards").mkdir(parents=True)
    (main_dir / "build-manifest.json").write_text(
        json.dumps({"schema": "playgta6-recomp-build", "module": "main"}),
        encoding="utf-8",
    )
    (main_dir / "image" / "guest.pgim").write_bytes(b"PGIM" + b"\x01" * 48)
    (main_dir / "shards" / "shard-000.ll").write_text("; excluded llvm ir\n", encoding="utf-8")
    (main_dir / "shards" / "shard-000.o").write_bytes(b"\x7fELF" + b"\x00" * 16)
    (main_dir / "registration.cpp").write_text("// excluded registration\n", encoding="utf-8")

    helper_dir = program_dir / "helper" / "image"
    helper_dir.mkdir(parents=True)
    (helper_dir / "helper.pgim").write_bytes(b"PGIM" + b"\x02" * 32)
    (program_dir / "helper" / "build-manifest.json").write_text(
        json.dumps({"schema": "playgta6-recomp-build", "module": "helper"}),
        encoding="utf-8",
    )


def _write_wasm_pair(
    wasm_dir: Path,
    wasm_bytes: bytes,
    js_text: str = "export default function createRuntimeModule() {}\n",
) -> None:
    wasm_dir.mkdir(parents=True, exist_ok=True)
    (wasm_dir / "runtime.wasm").write_bytes(wasm_bytes)
    (wasm_dir / "runtime.js").write_text(js_text, encoding="utf-8")


def _build_synthetic_wgpk(seed: int = 1) -> bytes:
    """Build a minimal valid WGPK v1 binary shader pack header + payload."""
    return struct.pack("<4sIII", b"WGPK", 1, 0, seed)


def _sample_reflect_sidecar(stage_count: int = 2) -> dict[str, object]:
    stages = {
        f"0x{idx + 1:016x}": {
            "stage": "vertex" if idx % 2 == 0 else "fragment",
            "bindings": [
                {
                    "group": 0,
                    "binding": 0,
                    "kind": "uniform",
                    "source": {"user_sgpr": 8, "table_byte_offset": 0},
                }
            ],
        }
        for idx in range(stage_count)
    }
    return {
        "schema": "playgta6-stage-reflection",
        "version": 1,
        "stages": stages,
    }


def _can_create_symlinks(tmp_dir: Path) -> bool:
    target = tmp_dir / "_symlink_probe_target"
    link = tmp_dir / "_symlink_probe_link"
    target.write_text("ok", encoding="utf-8")
    try:
        os.symlink(target, link)
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        link.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


class TestBootArtifactStaging(unittest.TestCase):
    def test_allowlisted_staging_preserves_relative_paths_and_skips_ir_obj_cpp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["boot"]["status"], "staged")
            self.assertEqual(report["boot"]["discovered_count"], 5)
            self.assertEqual(report["boot"]["copied_count"], 5)
            self.assertEqual(report["boot"]["unchanged_count"], 0)
            self.assertEqual(report["boot"]["skipped_non_allowlisted"], 6)
            self.assertEqual(report["wasm"]["status"], "skipped")

            expected_files = {
                "program-manifest.json",
                "main/build-manifest.json",
                "main/image/guest.pgim",
                "helper/build-manifest.json",
                "helper/image/helper.pgim",
            }
            staged_files = {
                p.relative_to(boot_dir).as_posix()
                for p in boot_dir.rglob("*")
                if p.is_file()
            }
            self.assertEqual(staged_files, expected_files)
            self.assertEqual(
                (boot_dir / "main" / "image" / "guest.pgim").read_bytes(),
                b"PGIM" + b"\x01" * 48,
            )

    def test_incremental_staging_skips_unchanged_and_updates_on_size_or_sha256_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)

            first = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(first["ok"])
            self.assertEqual(first["boot"]["copied_count"], 5)

            # Second run: everything unchanged.
            second = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(second["ok"])
            self.assertEqual(second["boot"]["status"], "unchanged")
            self.assertEqual(second["boot"]["copied_count"], 0)
            self.assertEqual(second["boot"]["unchanged_count"], 5)

            # Same-size content modification (SHA-256 changed, even if mtime preserved).
            guest_src = program_dir / "main" / "image" / "guest.pgim"
            orig_stat = guest_src.stat()
            guest_src.write_bytes(b"PGIM" + b"\x99" * 48)
            os.utime(guest_src, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))

            # Size modification on helper.pgim.
            helper_src = program_dir / "helper" / "image" / "helper.pgim"
            helper_src.write_bytes(b"PGIM" + b"\x02" * 64)

            third = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(third["ok"])
            self.assertEqual(third["boot"]["copied_count"], 2)
            self.assertEqual(third["boot"]["unchanged_count"], 3)
            by_rel = {a["rel_path"]: a for a in third["boot"]["artifacts"]}
            self.assertEqual(by_rel["main/image/guest.pgim"]["reason"], "sha256_changed")
            self.assertEqual(by_rel["helper/image/helper.pgim"]["reason"], "size_changed")
            self.assertEqual(
                (boot_dir / "main" / "image" / "guest.pgim").read_bytes(),
                b"PGIM" + b"\x99" * 48,
            )

    def test_escaping_symlink_in_program_dir_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            if not _can_create_symlinks(base):
                self.skipTest("symlinks require elevated privileges on this host")
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)

            outside_secret = base / "outside.pgim"
            outside_secret.write_bytes(b"PGIM-SECRET")
            os.symlink(outside_secret, program_dir / "escape.pgim")

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertGreaterEqual(report["boot"]["skipped_escaping_symlinks"], 1)
            self.assertFalse((boot_dir / "escape.pgim").exists())

    def test_escaping_symlink_guard_via_mock_runs_on_all_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)
            escape_entry = program_dir / "escape.pgim"
            escape_entry.write_bytes(b"PGIM-FAKE")
            outside_target = base / "outside_secret.pgim"
            outside_target.write_bytes(b"PGIM-OUTSIDE")

            orig_is_symlink = Path.is_symlink
            orig_resolve = Path.resolve

            def fake_is_symlink(self_path: Path) -> bool:
                if self_path.name == "escape.pgim":
                    return True
                return orig_is_symlink(self_path)

            def fake_resolve(self_path: Path, strict: bool = False) -> Path:
                if self_path.name == "escape.pgim":
                    return orig_resolve(outside_target, strict=strict)
                return orig_resolve(self_path, strict=strict)

            with (
                mock.patch.object(Path, "is_symlink", fake_is_symlink),
                mock.patch.object(Path, "resolve", fake_resolve),
            ):
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    boot_dir=boot_dir,
                    repo_root=repo,
                )

            self.assertTrue(report["ok"])
            self.assertEqual(report["boot"]["skipped_escaping_symlinks"], 1)
            self.assertFalse((boot_dir / "escape.pgim").exists())
            self.assertTrue((boot_dir / "main" / "image" / "guest.pgim").is_file())

    def test_staged_boot_artifacts_served_cleanly_by_serve_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            _populate_program_dir(program_dir)

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])

            httpd = serve_local.create_server(
                "127.0.0.1",
                0,
                repo / "web",
                repo / "local-data",
                boot_root=boot_dir,
            )
            port = int(httpd.server_address[1])
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3.0)
                try:
                    conn.request("GET", "/boot/program-manifest.json")
                    resp = conn.getresponse()
                    manifest_body = resp.read()
                    self.assertEqual(resp.status, 200)
                    self.assertEqual(
                        resp.getheader("Cross-Origin-Opener-Policy"), "same-origin"
                    )
                    self.assertIn(b"playgta6-recomp-program", manifest_body)

                    conn.request("GET", "/boot/main/image/guest.pgim")
                    resp2 = conn.getresponse()
                    pgim_body = resp2.read()
                    self.assertEqual(resp2.status, 200)
                    self.assertEqual(pgim_body, b"PGIM" + b"\x01" * 48)

                    # Excluded .ll was never staged and is blocked by /boot/ allowlist.
                    conn.request("GET", "/boot/main/shards/shard-000.ll")
                    resp3 = conn.getresponse()
                    resp3.read()
                    self.assertEqual(resp3.status, 403)
                finally:
                    conn.close()
            finally:
                httpd.shutdown()
                httpd.server_close()
                thread.join(timeout=2)

    def test_missing_program_dir_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            report = sbb.stage_browser_boot(
                program_dir=repo / "work" / "does_not_exist",
                boot_dir=repo / "work" / "recomp_build",
                repo_root=repo,
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["boot"]["status"], "error")
            self.assertTrue(report["errors"])


class TestTrackedDirectorySafetyGuard(unittest.TestCase):
    def test_refuses_tracked_boot_dirs_and_allows_work_local_data_and_external(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            _populate_program_dir(program_dir)

            for bad_rel in ("web", "tools", "runtime", "web/dist", "."):
                bad_boot = repo / bad_rel
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    boot_dir=bad_boot,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"], f"expected refusal for {bad_rel}")
                self.assertFalse(report["safety"]["boot_dir_safe"])
                self.assertEqual(report["boot"]["status"], "refused")
                self.assertFalse((bad_boot / "program-manifest.json").exists())

            # Allowed inside work/
            ok_work = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=repo / "work" / "custom_boot",
                repo_root=repo,
            )
            self.assertTrue(ok_work["ok"])
            self.assertTrue((repo / "work" / "custom_boot" / "program-manifest.json").is_file())

            # Allowed inside local-data/
            ok_data = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=repo / "local-data" / "boot_cache",
                repo_root=repo,
            )
            self.assertTrue(ok_data["ok"])
            self.assertTrue(
                (repo / "local-data" / "boot_cache" / "program-manifest.json").is_file()
            )

            # Allowed outside repo (temp dir)
            ext_boot = base / "external_boot"
            ok_ext = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=ext_boot,
                repo_root=repo,
            )
            self.assertTrue(ok_ext["ok"])
            self.assertTrue((ext_boot / "program-manifest.json").is_file())

    def test_refuses_tracked_web_dist_and_allows_web_dist_and_external(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())

            for bad_rel in ("web", "tools", "runtime", "work/recomp_build", "."):
                bad_dist = repo / bad_rel
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    wasm_dir=wasm_dir,
                    boot_dir=repo / "work" / "recomp_build",
                    web_dist=bad_dist,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"], f"expected refusal for web_dist={bad_rel}")
                self.assertFalse(report["safety"]["web_dist_safe"])
                self.assertEqual(report["wasm"]["status"], "refused")
                self.assertFalse((bad_dist / "runtime.wasm").exists())

            # Allowed: web/dist
            ok_dist = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=repo / "work" / "recomp_build",
                web_dist=repo / "web" / "dist",
                repo_root=repo,
            )
            self.assertTrue(ok_dist["ok"])
            self.assertTrue((repo / "web" / "dist" / "runtime.wasm").is_file())

            # Allowed: outside repo
            ext_dist = base / "external_dist"
            ok_ext = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=repo / "work" / "recomp_build",
                web_dist=ext_dist,
                repo_root=repo,
            )
            self.assertTrue(ok_ext["ok"])
            self.assertTrue((ext_dist / "runtime.wasm").is_file())

    def test_refuses_boot_dir_equal_to_or_inside_program_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            _populate_program_dir(program_dir)

            for nested in (program_dir, program_dir / "nested_out"):
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    boot_dir=nested,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"])
                self.assertEqual(report["boot"]["status"], "refused")


class TestWasm64ValidatedStaging(unittest.TestCase):
    def test_valid_wasm64_stages_atomically_into_web_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)

            valid_wasm = wasm.build_valid_runtime_wasm()
            js_payload = "export default function createRuntimeModule() { return 64; }\n"
            _write_wasm_pair(wasm_dir, valid_wasm, js_payload)

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["status"], "staged")
            self.assertEqual(report["wasm"]["copied_count"], 2)
            self.assertTrue(report["wasm"]["validation"]["ok"])
            self.assertTrue(report["wasm"]["validation"]["is_memory64"])
            self.assertTrue(report["wasm"]["validation"]["is_shared_memory"])
            self.assertTrue(report["wasm"]["validation"]["all_host_imports_ok"])
            self.assertEqual(report["wasm"]["validation"]["host_imports_valid"], 5)
            self.assertEqual(report["wasm"]["validation"]["host_imports_total"], 5)
            self.assertEqual((web_dist / "runtime.wasm").read_bytes(), valid_wasm)
            self.assertEqual((web_dist / "runtime.js").read_text(encoding="utf-8"), js_payload)
            # Ensure no leftover .tmp files in web/dist.
            self.assertEqual(
                sorted(p.name for p in web_dist.iterdir()),
                ["runtime.js", "runtime.wasm"],
            )

            # Second run with identical valid Wasm64 is a clean no-op.
            second = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(second["ok"])
            self.assertEqual(second["wasm"]["status"], "unchanged")
            self.assertEqual(second["wasm"]["copied_count"], 0)
            self.assertEqual(second["wasm"]["unchanged_count"], 2)

    def test_atomic_replace_failure_cleans_up_temp_files_and_preserves_existing_wasm(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)

            existing_wasm = wasm.build_valid_runtime_wasm(min_pages=512)
            existing_js = "// existing runtime.js\n"
            _write_wasm_pair(web_dist, existing_wasm, existing_js)

            new_wasm = wasm.build_valid_runtime_wasm(min_pages=1024)
            _write_wasm_pair(wasm_dir, new_wasm, "// new runtime.js\n")

            orig_replace = os.replace

            def fail_on_web_dist(src: Path | str, dst: Path | str) -> None:
                if Path(dst).parent.resolve() == web_dist.resolve():
                    raise OSError("simulated replace failure")
                orig_replace(src, dst)

            with mock.patch("stage_browser_boot.os.replace", side_effect=fail_on_web_dist):
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    wasm_dir=wasm_dir,
                    boot_dir=boot_dir,
                    web_dist=web_dist,
                    repo_root=repo,
                )

            self.assertFalse(report["ok"])
            self.assertEqual(report["wasm"]["status"], "error")
            self.assertEqual((web_dist / "runtime.wasm").read_bytes(), existing_wasm)
            self.assertEqual(
                (web_dist / "runtime.js").read_text(encoding="utf-8"),
                existing_js,
            )
            self.assertEqual(
                sorted(p.name for p in web_dist.iterdir()),
                ["runtime.js", "runtime.wasm"],
            )

    def test_invalid_wasm64_refuses_and_preserves_existing_web_dist(self) -> None:
        cases = [
            ("wasm32_not_memory64", wasm.build_valid_runtime_wasm(is_memory64=False)),
            ("unshared_memory", wasm.build_valid_runtime_wasm(is_shared=False)),
            ("wasm32_host_sigs", wasm.build_valid_runtime_wasm(wasm32_host_sigs=True)),
            (
                "missing_host_import",
                wasm.build_valid_runtime_wasm(omit_host_imports=("host_wgpu_submit",)),
            ),
            ("corrupt_bytes", b"NOT_A_WASM_BINARY"),
        ]
        for label, bad_wasm in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                repo = _make_repo_skeleton(Path(tmp))
                program_dir = repo / "work" / "recomp_program"
                wasm_dir = repo / "work" / "wasm_guest_dist"
                boot_dir = repo / "work" / "recomp_build"
                web_dist = repo / "web" / "dist"
                _populate_program_dir(program_dir)

                # Pre-populate existing valid runtime in web/dist.
                existing_wasm = wasm.build_valid_runtime_wasm(min_pages=512)
                existing_js = "// known-good runtime.js\n"
                _write_wasm_pair(web_dist, existing_wasm, existing_js)

                # Candidate contains invalid Wasm64 + new JS.
                _write_wasm_pair(wasm_dir, bad_wasm, "// broken runtime.js\n")

                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    wasm_dir=wasm_dir,
                    boot_dir=boot_dir,
                    web_dist=web_dist,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"])
                self.assertEqual(report["wasm"]["status"], "refused")
                self.assertFalse(report["wasm"]["validation"]["ok"])
                self.assertTrue(report["errors"])
                # Existing web/dist files MUST remain completely untouched!
                self.assertEqual((web_dist / "runtime.wasm").read_bytes(), existing_wasm)
                self.assertEqual(
                    (web_dist / "runtime.js").read_text(encoding="utf-8"),
                    existing_js,
                )

    def test_partial_wasm_bundle_fails_without_touching_web_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)
            wasm_dir.mkdir(parents=True)
            (wasm_dir / "runtime.wasm").write_bytes(wasm.build_valid_runtime_wasm())
            # Deliberately omit runtime.js.

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=repo / "work" / "recomp_build",
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["wasm"]["status"], "error")
            self.assertFalse(web_dist.exists())

    def test_no_side_modules_keeps_backwards_compatible_report_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["status"], "staged")
            self.assertEqual(report["wasm"]["copied_count"], 2)
            self.assertEqual(report["wasm"]["side_module_count"], 0)
            self.assertEqual(report["wasm"]["side_modules"], [])
            rendered = sbb.format_report_text(report)
            self.assertNotIn("side_modules=", rendered)


class TestSideModuleWasmStaging(unittest.TestCase):
    def test_side_modules_dry_run_initial_stage_unchanged_and_incremental_update(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)

            valid_wasm = wasm.build_valid_runtime_wasm()
            js_payload = "export default function createRuntimeModule() { return 64; }\n"
            _write_wasm_pair(wasm_dir, valid_wasm, js_payload)

            part0_bytes = b"\x00asm\x01\x00\x00\x00" + b"\x10" * 40
            part1_bytes = b"\x00asm\x01\x00\x00\x00" + b"\x20" * 64
            (wasm_dir / "guest_part_0.wasm").write_bytes(part0_bytes)
            (wasm_dir / "guest_part_1.wasm").write_bytes(part1_bytes)
            # Non-.wasm siblings and nested .wasm files must NOT be staged.
            (wasm_dir / "guest_part_0.o").write_bytes(b"\x7fELF")
            (wasm_dir / "notes.txt").write_text("ignore me\n", encoding="utf-8")
            (wasm_dir / "nested").mkdir()
            (wasm_dir / "nested" / "ignored_nested.wasm").write_bytes(
                b"\x00asm\x01\x00\x00\x00"
            )

            # 1. Dry-run plans all 4 files (runtime + 2 side modules) without writing.
            dry = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
                dry_run=True,
            )
            self.assertTrue(dry["ok"])
            self.assertTrue(dry["dry_run"])
            self.assertEqual(dry["wasm"]["status"], "planned")
            self.assertEqual(dry["wasm"]["planned_count"], 4)
            self.assertEqual(dry["wasm"]["copied_count"], 0)
            self.assertEqual(dry["wasm"]["side_module_count"], 2)
            self.assertEqual(
                [m["name"] for m in dry["wasm"]["side_modules"]],
                ["guest_part_0.wasm", "guest_part_1.wasm"],
            )
            self.assertEqual(
                [m["action"] for m in dry["wasm"]["side_modules"]],
                ["plan_copy", "plan_copy"],
            )
            self.assertEqual(
                [m["size_bytes"] for m in dry["wasm"]["side_modules"]],
                [len(part0_bytes), len(part1_bytes)],
            )
            self.assertFalse(web_dist.exists())

            # 2. Initial stage copies runtime + both side modules atomically.
            initial = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(initial["ok"])
            self.assertEqual(initial["wasm"]["status"], "staged")
            self.assertEqual(initial["wasm"]["copied_count"], 4)
            self.assertEqual(initial["wasm"]["unchanged_count"], 0)
            self.assertEqual(initial["wasm"]["side_module_count"], 2)
            self.assertEqual(
                [(m["name"], m["size_bytes"], m["action"]) for m in initial["wasm"]["side_modules"]],
                [
                    ("guest_part_0.wasm", len(part0_bytes), "copied"),
                    ("guest_part_1.wasm", len(part1_bytes), "copied"),
                ],
            )
            self.assertEqual(
                sorted(p.name for p in web_dist.iterdir()),
                ["guest_part_0.wasm", "guest_part_1.wasm", "runtime.js", "runtime.wasm"],
            )
            self.assertEqual((web_dist / "guest_part_0.wasm").read_bytes(), part0_bytes)
            self.assertEqual((web_dist / "guest_part_1.wasm").read_bytes(), part1_bytes)

            # 3. Incremental unchanged re-run is a clean no-op.
            unchanged = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(unchanged["ok"])
            self.assertEqual(unchanged["wasm"]["status"], "unchanged")
            self.assertEqual(unchanged["wasm"]["copied_count"], 0)
            self.assertEqual(unchanged["wasm"]["unchanged_count"], 4)
            self.assertEqual(
                [m["action"] for m in unchanged["wasm"]["side_modules"]],
                ["unchanged", "unchanged"],
            )

            # 4a. Same-size SHA-256 modification (even with preserved mtime) updates only part 0.
            part0_src = wasm_dir / "guest_part_0.wasm"
            orig_stat = part0_src.stat()
            part0_updated = b"\x00asm\x01\x00\x00\x00" + b"\x99" * 40
            self.assertEqual(len(part0_updated), len(part0_bytes))
            part0_src.write_bytes(part0_updated)
            os.utime(part0_src, ns=(orig_stat.st_atime_ns, orig_stat.st_mtime_ns))

            sha_update = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(sha_update["ok"])
            self.assertEqual(sha_update["wasm"]["status"], "staged")
            self.assertEqual(sha_update["wasm"]["copied_count"], 1)
            self.assertEqual(sha_update["wasm"]["unchanged_count"], 3)
            by_side = {m["name"]: m for m in sha_update["wasm"]["side_modules"]}
            self.assertEqual(by_side["guest_part_0.wasm"]["action"], "copied")
            self.assertEqual(by_side["guest_part_0.wasm"]["reason"], "sha256_changed")
            self.assertEqual(by_side["guest_part_1.wasm"]["action"], "unchanged")
            self.assertEqual((web_dist / "guest_part_0.wasm").read_bytes(), part0_updated)
            self.assertEqual((web_dist / "guest_part_1.wasm").read_bytes(), part1_bytes)

            # 4b. Size modification updates only part 1.
            part1_updated = b"\x00asm\x01\x00\x00\x00" + b"\x33" * 96
            (wasm_dir / "guest_part_1.wasm").write_bytes(part1_updated)

            size_update = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(size_update["ok"])
            self.assertEqual(size_update["wasm"]["status"], "staged")
            self.assertEqual(size_update["wasm"]["copied_count"], 1)
            self.assertEqual(size_update["wasm"]["unchanged_count"], 3)
            by_side_2 = {m["name"]: m for m in size_update["wasm"]["side_modules"]}
            self.assertEqual(by_side_2["guest_part_0.wasm"]["action"], "unchanged")
            self.assertEqual(by_side_2["guest_part_1.wasm"]["action"], "copied")
            self.assertEqual(by_side_2["guest_part_1.wasm"]["reason"], "size_changed")
            self.assertEqual(by_side_2["guest_part_1.wasm"]["size_bytes"], len(part1_updated))
            self.assertEqual((web_dist / "guest_part_1.wasm").read_bytes(), part1_updated)

    def test_side_modules_natural_order_for_multidigit_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            wasm_dir = repo / "work" / "wasm_guest_dist"
            web_dist = repo / "web" / "dist"
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())
            for idx in (10, 2, 0, 1):
                (wasm_dir / f"guest_part_{idx}.wasm").write_bytes(
                    b"\x00asm\x01\x00\x00\x00" + bytes([idx]) * 8
                )

            wasm_section, errors, warnings = sbb.stage_wasm_runtime(
                wasm_dir=wasm_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertEqual(errors, [])
            self.assertEqual(warnings, [])
            self.assertEqual(wasm_section["status"], "staged")
            self.assertEqual(
                [m["name"] for m in wasm_section["side_modules"]],
                [
                    "guest_part_0.wasm",
                    "guest_part_1.wasm",
                    "guest_part_2.wasm",
                    "guest_part_10.wasm",
                ],
            )

    def test_refuses_to_stage_side_modules_when_runtime_wasm_fails_abi_validation(
        self,
    ) -> None:
        cases = [
            ("wasm32_not_memory64", wasm.build_valid_runtime_wasm(is_memory64=False)),
            ("unshared_memory", wasm.build_valid_runtime_wasm(is_shared=False)),
            (
                "missing_host_import",
                wasm.build_valid_runtime_wasm(omit_host_imports=("host_wgpu_submit",)),
            ),
            ("corrupt_bytes", b"NOT_WASM"),
        ]
        for label, bad_wasm in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                repo = _make_repo_skeleton(Path(tmp))
                program_dir = repo / "work" / "recomp_program"
                wasm_dir = repo / "work" / "wasm_guest_dist"
                boot_dir = repo / "work" / "recomp_build"
                web_dist = repo / "web" / "dist"
                _populate_program_dir(program_dir)

                # Pre-populate known-good runtime + existing side module in web/dist.
                existing_wasm = wasm.build_valid_runtime_wasm(min_pages=512)
                existing_js = "// known-good runtime.js\n"
                existing_part0 = b"\x00asm\x01\x00\x00\x00GOOD_PART0"
                _write_wasm_pair(web_dist, existing_wasm, existing_js)
                (web_dist / "guest_part_0.wasm").write_bytes(existing_part0)

                # Candidate has invalid runtime.wasm + new/updated side modules.
                _write_wasm_pair(wasm_dir, bad_wasm, "// bad runtime.js\n")
                (wasm_dir / "guest_part_0.wasm").write_bytes(
                    b"\x00asm\x01\x00\x00\x00BAD_PART0"
                )
                (wasm_dir / "guest_part_1.wasm").write_bytes(
                    b"\x00asm\x01\x00\x00\x00NEW_PART1"
                )

                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    wasm_dir=wasm_dir,
                    boot_dir=boot_dir,
                    web_dist=web_dist,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"])
                self.assertEqual(report["wasm"]["status"], "refused")
                self.assertFalse(report["wasm"]["validation"]["ok"])
                self.assertEqual(report["wasm"]["side_modules"], [])
                self.assertEqual(report["wasm"]["side_module_count"], 0)
                # Existing web/dist files MUST remain untouched, and guest_part_1.wasm MUST NOT exist.
                self.assertEqual((web_dist / "runtime.wasm").read_bytes(), existing_wasm)
                self.assertEqual(
                    (web_dist / "runtime.js").read_text(encoding="utf-8"),
                    existing_js,
                )
                self.assertEqual(
                    (web_dist / "guest_part_0.wasm").read_bytes(),
                    existing_part0,
                )
                self.assertFalse((web_dist / "guest_part_1.wasm").exists())

    def test_external_side_module_symlink_is_skipped_real_fs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            if not _can_create_symlinks(base):
                self.skipTest("symlinks require elevated privileges on this host")
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())
            (wasm_dir / "guest_part_0.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00OK")

            outside_secret = base / "outside_secret.wasm"
            outside_secret.write_bytes(b"\x00asm\x01\x00\x00\x00SECRET")
            os.symlink(outside_secret, wasm_dir / "escape_side.wasm")

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["status"], "staged")
            self.assertEqual(report["wasm"]["skipped_escaping_symlinks"], 1)
            self.assertEqual(
                [m["name"] for m in report["wasm"]["side_modules"]],
                ["guest_part_0.wasm"],
            )
            self.assertTrue((web_dist / "guest_part_0.wasm").is_file())
            self.assertFalse((web_dist / "escape_side.wasm").exists())

    def test_external_side_module_symlink_guard_via_mock_runs_on_all_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())
            (wasm_dir / "guest_part_0.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00OK")
            (wasm_dir / "escape_side.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00FAKE")

            outside_target = base / "outside_secret.wasm"
            outside_target.write_bytes(b"\x00asm\x01\x00\x00\x00OUTSIDE")

            orig_is_symlink = Path.is_symlink
            orig_resolve = Path.resolve

            def fake_is_symlink(self_path: Path) -> bool:
                if self_path.name == "escape_side.wasm":
                    return True
                return orig_is_symlink(self_path)

            def fake_resolve(self_path: Path, strict: bool = False) -> Path:
                if self_path.name == "escape_side.wasm":
                    return orig_resolve(outside_target, strict=strict)
                return orig_resolve(self_path, strict=strict)

            with (
                mock.patch.object(Path, "is_symlink", fake_is_symlink),
                mock.patch.object(Path, "resolve", fake_resolve),
            ):
                report = sbb.stage_browser_boot(
                    program_dir=program_dir,
                    wasm_dir=wasm_dir,
                    boot_dir=boot_dir,
                    web_dist=web_dist,
                    repo_root=repo,
                )

            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["status"], "staged")
            self.assertEqual(report["wasm"]["skipped_escaping_symlinks"], 1)
            self.assertEqual(
                [m["name"] for m in report["wasm"]["side_modules"]],
                ["guest_part_0.wasm"],
            )
            self.assertTrue((web_dist / "guest_part_0.wasm").is_file())
            self.assertFalse((web_dist / "escape_side.wasm").exists())
            self.assertTrue(
                any("escaping wasm-dir" in w for w in report["warnings"]),
                f"expected escaping symlink warning, got {report['warnings']!r}",
            )

    def test_rejects_unsafe_side_module_names_and_traversal(self) -> None:
        self.assertTrue(sbb.is_safe_side_module_name("guest_part_0.wasm"))
        self.assertTrue(sbb.is_safe_side_module_name("guest-part.v2.wasm"))
        for bad_name in (
            "",
            "runtime.wasm",
            "RUNTIME.WASM",
            "../escape.wasm",
            "..\\escape.wasm",
            "..evil.wasm",
            ".hidden.wasm",
            "guest..part.wasm",
            "sub/part.wasm",
            "bad name.wasm",
            "bad;cmd.wasm",
            "part\x00.wasm",
        ):
            self.assertFalse(
                sbb.is_safe_side_module_name(bad_name),
                f"expected unsafe name rejection for {bad_name!r}",
            )

        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())
            (wasm_dir / "guest_part_0.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00OK")
            (wasm_dir / "..evil.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00EVIL")

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=boot_dir,
                web_dist=web_dist,
                repo_root=repo,
            )
            self.assertFalse(report["ok"])
            self.assertEqual(report["wasm"]["status"], "refused")
            self.assertFalse(web_dist.exists())
            self.assertTrue(
                any("unsafe side-module filename" in err for err in report["errors"]),
                f"expected unsafe filename error, got {report['errors']!r}",
            )

    def test_stage_wasm_runtime_direct_refuses_tracked_web_dist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            wasm_dir = repo / "work" / "wasm_guest_dist"
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())
            (wasm_dir / "guest_part_0.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00")

            wasm_section, errors, _ = sbb.stage_wasm_runtime(
                wasm_dir=wasm_dir,
                web_dist=repo / "web",
                repo_root=repo,
            )
            self.assertEqual(wasm_section["status"], "refused")
            self.assertTrue(errors)
            self.assertFalse((repo / "web" / "guest_part_0.wasm").exists())

    def test_side_module_json_and_text_output_contain_zero_host_paths_or_raw_bytes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="SideModSecretHost_") as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            ext_boot = base / "ext_boot_out"
            ext_dist = base / "ext_dist_out"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())
            (wasm_dir / "guest_part_0.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00PART0")
            (wasm_dir / "guest_part_1.wasm").write_bytes(b"\x00asm\x01\x00\x00\x00PART1")

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=ext_boot,
                web_dist=ext_dist,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(len(report["wasm"]["side_modules"]), 2)
            encoded = json.dumps(report)
            rendered = sbb.format_report_text(report)

            self.assertIn("guest_part_0.wasm", encoded)
            self.assertIn("guest_part_1.wasm", encoded)
            self.assertIn("side_modules=2", rendered)
            self.assertIn("guest_part_0.wasm", rendered)
            self.assertIn("guest_part_1.wasm", rendered)

            for text in (encoded, rendered):
                self.assertNotIn("SideModSecretHost_", text)
                self.assertNotIn(str(base), text)
                self.assertNotIn(str(base).replace("\\", "/"), text)
                self.assertNotIn("PART0", text)
                self.assertNotIn("PART1", text)
                self.assertNotRegex(text, r"[A-Za-z]:[\\/]")
                self.assertNotIn("/Users/", text)
                self.assertNotIn("/home/", text)
                self.assertNotIn("/tmp/", text)
                for username in sbb._current_usernames():
                    self.assertNotRegex(
                        text,
                        re.compile(
                            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
                            re.IGNORECASE,
                        ),
                    )


class TestDryRunAndCli(unittest.TestCase):
    def test_dry_run_plans_without_writing_any_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            boot_dir = repo / "work" / "recomp_build"
            web_dist = repo / "web" / "dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = sbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--program-dir",
                        "work/recomp_program",
                        "--wasm-dir",
                        "work/wasm_guest_dist",
                        "--boot-dir",
                        "work/recomp_build",
                        "--web-dist",
                        "web/dist",
                        "--dry-run",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["boot"]["status"], "planned")
            self.assertEqual(payload["boot"]["planned_count"], 5)
            self.assertEqual(payload["boot"]["copied_count"], 0)
            self.assertEqual(payload["wasm"]["status"], "planned")
            self.assertEqual(payload["wasm"]["planned_count"], 2)
            self.assertEqual(payload["wasm"]["copied_count"], 0)
            self.assertFalse(boot_dir.exists())
            self.assertFalse(web_dist.exists())

    def test_cli_returns_nonzero_when_wasm_validation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm(is_memory64=False))

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = sbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--program-dir",
                        str(program_dir),
                        "--wasm-dir",
                        str(wasm_dir),
                        "--boot-dir",
                        str(repo / "work" / "recomp_build"),
                        "--web-dist",
                        str(repo / "web" / "dist"),
                        "--dry-run",
                    ]
                )
            self.assertEqual(rc, 1)
            output = buf.getvalue()
            self.assertIn("FAIL", output)
            self.assertIn("memory64=NO", output)


class TestCrossPlatformSanitization(unittest.TestCase):
    def test_sanitizes_windows_unc_and_posix_paths_and_usernames(self) -> None:
        samples = [
            r"C:\Users\AliceSecret\work\recomp_program\guest.pgim",
            r"D:/Users/CharlieSecret/local-data/boot/program-manifest.json",
            r"\\fileserver\PrivateShare\BobSecret\runtime.wasm",
            "//fileserver/PrivateShare/BobSecret/runtime.js",
            "/home/bobsecret/src/playgta6web/web/dist/runtime.wasm",
            "/Users/AliceSecret/Library/Caches/guest.pgim",
            "/tmp/AliceSecret-stage/recomp_build",
        ]
        old_username = os.environ.get("USERNAME")
        old_user = os.environ.get("USER")
        try:
            os.environ["USERNAME"] = "AliceSecret"
            os.environ["USER"] = "BobSecret"
            for sample in samples:
                cleaned = sbb.sanitize_text(sample)
                self.assertNotIn("AliceSecret", cleaned)
                self.assertNotIn("BobSecret", cleaned)
                self.assertNotIn("CharlieSecret", cleaned)
                self.assertNotRegex(cleaned, r"[A-Za-z]:\\")
                self.assertNotRegex(cleaned, r"[A-Za-z]:/")
                self.assertNotIn("\\\\", cleaned)
                self.assertNotIn("//fileserver", cleaned)
                self.assertNotIn("/home/", cleaned)
                self.assertNotIn("/Users/", cleaned)
                self.assertNotIn("/tmp/", cleaned)
        finally:
            if old_username is None:
                os.environ.pop("USERNAME", None)
            else:
                os.environ["USERNAME"] = old_username
            if old_user is None:
                os.environ.pop("USER", None)
            else:
                os.environ["USER"] = old_user

    def test_basename_label_handles_both_separators_cross_platform(self) -> None:
        self.assertEqual(
            sbb.basename_label(r"C:\Users\DevUser\work\recomp_program\guest.pgim"),
            "guest.pgim",
        )
        self.assertEqual(
            sbb.basename_label("/home/devuser/work/recomp_program/guest.pgim"),
            "guest.pgim",
        )
        self.assertEqual(sbb.basename_label("C:\\"), "<unknown>")
        self.assertEqual(sbb.basename_label(None), "<unknown>")

    def test_sanitize_value_replaces_raw_bytes(self) -> None:
        sanitized = sbb.sanitize_value(
            {
                "raw": b"\x00asm\x01\x00\x00\x00",
                "nested": [bytearray(b"abc"), memoryview(b"12345")],
                "path": Path(r"C:\Users\SecretName\guest.pgim"),
            }
        )
        self.assertEqual(sanitized["raw"], "<bytes:8>")
        self.assertEqual(sanitized["nested"], ["<bytes:3>", "<bytes:5>"])
        self.assertEqual(sanitized["path"], "guest.pgim")

    def test_json_and_text_reports_contain_zero_host_paths_or_usernames(self) -> None:
        with tempfile.TemporaryDirectory(prefix="StageSecretHost_") as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            program_dir = repo / "work" / "recomp_program"
            wasm_dir = repo / "work" / "wasm_guest_dist"
            ext_boot = base / "ext_boot_out"
            ext_dist = base / "ext_dist_out"
            _populate_program_dir(program_dir)
            _write_wasm_pair(wasm_dir, wasm.build_valid_runtime_wasm())

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                wasm_dir=wasm_dir,
                boot_dir=ext_boot,
                web_dist=ext_dist,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            encoded = json.dumps(report)
            rendered = sbb.format_report_text(report)

            for text in (encoded, rendered):
                self.assertNotIn("StageSecretHost_", text)
                self.assertNotIn(str(base), text)
                self.assertNotIn(str(base).replace("\\", "/"), text)
                self.assertNotRegex(text, r"[A-Za-z]:[\\/]")
                self.assertNotIn("/Users/", text)
                self.assertNotIn("/home/", text)
                self.assertNotIn("/tmp/", text)
                for username in sbb._current_usernames():
                    self.assertNotRegex(
                        text,
                        re.compile(
                            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
                            re.IGNORECASE,
                        ),
                    )


class TestShaderPackAndSidecarStaging(unittest.TestCase):
    def test_stages_wgsl_pack_with_adjacent_reflect_sidecar_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            shaders_dir = repo / "local-data" / "shaders"
            _populate_program_dir(program_dir)

            src_dir = repo / "work" / "shader_build"
            src_dir.mkdir(parents=True)
            src_pack = src_dir / "foo.wgsl.pack"
            src_sidecar = Path(str(src_pack) + ".reflect.json")
            pack_bytes = _build_synthetic_wgpk(seed=7)
            sidecar_text = json.dumps(_sample_reflect_sidecar(stage_count=3), indent=2) + "\n"
            sidecar_bytes = sidecar_text.encode("utf-8")
            src_pack.write_bytes(pack_bytes)
            src_sidecar.write_bytes(sidecar_bytes)

            # Dry-run plans both pack + sidecar without writing files.
            dry = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                shader_pack=src_pack,
                shaders_dir=shaders_dir,
                repo_root=repo,
                dry_run=True,
            )
            self.assertTrue(dry["ok"])
            self.assertEqual(dry["shaders"]["status"], "planned")
            self.assertEqual(dry["shaders"]["planned_count"], 2)
            self.assertFalse(dry["sidecar_staged"])
            self.assertEqual(dry["sidecar_bytes"], 0)
            self.assertFalse(shaders_dir.exists())

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                shader_pack=src_pack,
                shaders_dir=shaders_dir,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertTrue(report["sidecar_staged"])
            self.assertEqual(report["sidecar_bytes"], len(sidecar_bytes))
            self.assertEqual(report["shaders"]["status"], "staged")
            self.assertTrue(report["shaders"]["sidecar_staged"])
            self.assertEqual(report["shaders"]["sidecar_bytes"], len(sidecar_bytes))
            self.assertEqual(report["shaders"]["copied_count"], 2)
            self.assertEqual(report["shaders"]["pack"]["name"], "splash.wgsl.pack")
            self.assertEqual(report["shaders"]["sidecar"]["name"], "splash.wgsl.pack.reflect.json")
            self.assertEqual(report["shaders"]["sidecar"]["stage_count"], 3)
            self.assertEqual(report["shaders"]["sidecar"]["source"], "adjacent")

            dst_pack = shaders_dir / "splash.wgsl.pack"
            dst_sidecar = shaders_dir / "splash.wgsl.pack.reflect.json"
            self.assertEqual(dst_pack.read_bytes(), pack_bytes)
            self.assertEqual(dst_sidecar.read_bytes(), sidecar_bytes)
            self.assertEqual(
                sorted(p.name for p in shaders_dir.iterdir()),
                ["splash.wgsl.pack", "splash.wgsl.pack.reflect.json"],
            )

            # Second run is incremental unchanged no-op, still reporting sidecar_staged=True.
            second = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                shader_pack=src_pack,
                shaders_dir=shaders_dir,
                repo_root=repo,
            )
            self.assertTrue(second["ok"])
            self.assertEqual(second["shaders"]["status"], "unchanged")
            self.assertEqual(second["shaders"]["copied_count"], 0)
            self.assertEqual(second["shaders"]["unchanged_count"], 2)
            self.assertTrue(second["sidecar_staged"])
            self.assertEqual(second["sidecar_bytes"], len(sidecar_bytes))

    def test_stages_wgsl_pack_without_sidecar_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            program_dir = repo / "work" / "recomp_program"
            boot_dir = repo / "work" / "recomp_build"
            shaders_dir = repo / "local-data" / "shaders"
            _populate_program_dir(program_dir)

            src_pack = repo / "work" / "no_sidecar.wgsl.pack"
            pack_bytes = _build_synthetic_wgpk(seed=11)
            src_pack.write_bytes(pack_bytes)

            report = sbb.stage_browser_boot(
                program_dir=program_dir,
                boot_dir=boot_dir,
                shader_pack=src_pack,
                shaders_dir=shaders_dir,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["shaders"]["status"], "staged")
            self.assertFalse(report["sidecar_staged"])
            self.assertEqual(report["sidecar_bytes"], 0)
            self.assertIsNone(report["shaders"]["sidecar"])
            self.assertEqual(report["shaders"]["copied_count"], 1)
            self.assertEqual((shaders_dir / "splash.wgsl.pack").read_bytes(), pack_bytes)
            self.assertFalse((shaders_dir / "splash.wgsl.pack.reflect.json").exists())

    def test_invalid_sidecar_json_fails_cleanly_without_clobbering_existing_files(self) -> None:
        cases = [
            ("malformed_json", "{not-valid-json"),
            ("non_object_json", "[1, 2, 3]"),
            ("missing_stages_map", '{"schema": "playgta6-stage-reflection", "version": 1}'),
            ("non_dict_stages", '{"schema": "playgta6-stage-reflection", "stages": []}'),
            ("bad_version_bool", '{"version": true, "stages": {}}'),
        ]
        for label, bad_sidecar in cases:
            with self.subTest(case=label), tempfile.TemporaryDirectory() as tmp:
                repo = _make_repo_skeleton(Path(tmp))
                shaders_dir = repo / "local-data" / "shaders"
                shaders_dir.mkdir(parents=True)

                # Pre-populate known-good staged pack + sidecar.
                existing_pack = _build_synthetic_wgpk(seed=100)
                existing_sidecar = json.dumps(_sample_reflect_sidecar(stage_count=1)) + "\n"
                (shaders_dir / "splash.wgsl.pack").write_bytes(existing_pack)
                (shaders_dir / "splash.wgsl.pack.reflect.json").write_text(
                    existing_sidecar, encoding="utf-8"
                )

                # New candidate has updated pack + invalid sidecar.
                src_pack = repo / "work" / "broken.wgsl.pack"
                src_pack.write_bytes(_build_synthetic_wgpk(seed=200))
                Path(str(src_pack) + ".reflect.json").write_text(bad_sidecar, encoding="utf-8")

                report = sbb.stage_browser_boot(
                    shader_pack=src_pack,
                    shaders_dir=shaders_dir,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"])
                self.assertEqual(report["shaders"]["status"], "refused")
                self.assertFalse(report["sidecar_staged"])
                self.assertEqual(report["sidecar_bytes"], 0)
                self.assertTrue(report["errors"])
                # Existing staged pack and sidecar MUST remain completely untouched!
                self.assertEqual((shaders_dir / "splash.wgsl.pack").read_bytes(), existing_pack)
                self.assertEqual(
                    (shaders_dir / "splash.wgsl.pack.reflect.json").read_text(encoding="utf-8"),
                    existing_sidecar,
                )

    def test_reflect_sidecar_override_flag_and_missing_override_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            shaders_dir = repo / "local-data" / "shaders"
            src_pack = repo / "work" / "custom.wgsl.pack"
            src_pack.write_bytes(_build_synthetic_wgpk(seed=5))

            override_sidecar = repo / "work" / "override.reflect.json"
            override_bytes = (json.dumps(_sample_reflect_sidecar(stage_count=4)) + "\n").encode(
                "utf-8"
            )
            override_sidecar.write_bytes(override_bytes)

            report = sbb.stage_browser_boot(
                shader_pack=src_pack,
                shaders_dir=shaders_dir,
                shader_pack_name="custom_out.wgsl.pack",
                reflect_sidecar=override_sidecar,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertTrue(report["sidecar_staged"])
            self.assertEqual(report["sidecar_bytes"], len(override_bytes))
            self.assertEqual(report["shaders"]["sidecar"]["source"], "override")
            self.assertEqual(
                report["shaders"]["sidecar"]["name"], "custom_out.wgsl.pack.reflect.json"
            )
            self.assertEqual(
                (shaders_dir / "custom_out.wgsl.pack.reflect.json").read_bytes(),
                override_bytes,
            )

            # Missing explicit override fails cleanly without writing pack.
            missing_report = sbb.stage_browser_boot(
                shader_pack=src_pack,
                shaders_dir=repo / "local-data" / "shaders_missing",
                reflect_sidecar=repo / "work" / "does_not_exist.reflect.json",
                repo_root=repo,
            )
            self.assertFalse(missing_report["ok"])
            self.assertEqual(missing_report["shaders"]["status"], "error")
            self.assertFalse((repo / "local-data" / "shaders_missing").exists())

    def test_refuses_tracked_shaders_dir_and_unsafe_shader_pack_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            src_pack = repo / "work" / "foo.wgsl.pack"
            src_pack.write_bytes(_build_synthetic_wgpk(seed=1))
            Path(str(src_pack) + ".reflect.json").write_text(
                json.dumps(_sample_reflect_sidecar(stage_count=1)), encoding="utf-8"
            )

            for bad_rel in ("web", "tools", "runtime", "web/dist", "."):
                bad_dir = repo / bad_rel
                report = sbb.stage_browser_boot(
                    shader_pack=src_pack,
                    shaders_dir=bad_dir,
                    repo_root=repo,
                )
                self.assertFalse(report["ok"], f"expected refusal for shaders_dir={bad_rel}")
                self.assertFalse(report["safety"]["shaders_dir_safe"])
                self.assertEqual(report["shaders"]["status"], "refused")
                self.assertFalse((bad_dir / "splash.wgsl.pack").exists())
                self.assertFalse((bad_dir / "splash.wgsl.pack.reflect.json").exists())

            bad_name_report = sbb.stage_browser_boot(
                shader_pack=src_pack,
                shaders_dir=repo / "local-data" / "shaders",
                shader_pack_name="../escape.wgsl.pack",
                repo_root=repo,
            )
            self.assertFalse(bad_name_report["ok"])
            self.assertEqual(bad_name_report["shaders"]["status"], "refused")

    def test_atomic_replace_failure_preserves_existing_pack_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            shaders_dir = repo / "local-data" / "shaders"
            shaders_dir.mkdir(parents=True)

            old_pack = _build_synthetic_wgpk(seed=1)
            old_sidecar = json.dumps(_sample_reflect_sidecar(stage_count=1)) + "\n"
            (shaders_dir / "splash.wgsl.pack").write_bytes(old_pack)
            (shaders_dir / "splash.wgsl.pack.reflect.json").write_text(
                old_sidecar, encoding="utf-8"
            )

            src_pack = repo / "work" / "new.wgsl.pack"
            src_pack.write_bytes(_build_synthetic_wgpk(seed=2))
            Path(str(src_pack) + ".reflect.json").write_text(
                json.dumps(_sample_reflect_sidecar(stage_count=2)) + "\n",
                encoding="utf-8",
            )

            orig_replace = os.replace

            def fail_on_shaders_dir(src: Path | str, dst: Path | str) -> None:
                if Path(dst).parent.resolve() == shaders_dir.resolve():
                    raise OSError("simulated replace failure")
                orig_replace(src, dst)

            with mock.patch("stage_browser_boot.os.replace", side_effect=fail_on_shaders_dir):
                report = sbb.stage_browser_boot(
                    shader_pack=src_pack,
                    shaders_dir=shaders_dir,
                    repo_root=repo,
                )

            self.assertFalse(report["ok"])
            self.assertEqual(report["shaders"]["status"], "error")
            self.assertEqual((shaders_dir / "splash.wgsl.pack").read_bytes(), old_pack)
            self.assertEqual(
                (shaders_dir / "splash.wgsl.pack.reflect.json").read_text(encoding="utf-8"),
                old_sidecar,
            )
            self.assertEqual(
                sorted(p.name for p in shaders_dir.iterdir()),
                ["splash.wgsl.pack", "splash.wgsl.pack.reflect.json"],
            )


class TestBootOptionsStaging(unittest.TestCase):
    def test_verify_window_marks_writes_valid_boot_options_json_and_refuses_tracked_dirs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            boot_dir = repo / "work" / "recomp_build"

            report = sbb.stage_browser_boot(
                boot_dir=boot_dir,
                verify_window_marks=True,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertTrue(report["boot_options_staged"])
            self.assertEqual(report["boot_options"]["status"], "staged")
            self.assertEqual(
                report["boot_options"]["options"],
                {"verify_window_marks": True},
            )
            written = json.loads((boot_dir / "boot-options.json").read_text(encoding="utf-8"))
            self.assertEqual(written, {"verify_window_marks": True})

            # Refuses tracked repo directories cleanly without writing boot-options.json.
            for bad_rel in ("web", "tools", "runtime", "web/dist", "."):
                bad_boot = repo / bad_rel
                refused = sbb.stage_browser_boot(
                    boot_dir=bad_boot,
                    verify_window_marks=True,
                    repo_root=repo,
                )
                self.assertFalse(refused["ok"], f"expected refusal for {bad_rel}")
                self.assertFalse(refused["safety"]["boot_dir_safe"])
                self.assertFalse(refused["boot_options_staged"])
                self.assertEqual(refused["boot_options"]["status"], "refused")
                self.assertFalse((bad_boot / "boot-options.json").exists())

    def test_boot_options_json_inline_file_merge_and_schema_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo_skeleton(Path(tmp))
            boot_dir = repo / "work" / "recomp_build"

            # Inline JSON merged with --verify-window-marks.
            report = sbb.stage_browser_boot(
                boot_dir=boot_dir,
                boot_options='{"max_wall_ms": 900000, "report_to_log": true}',
                verify_window_marks=True,
                repo_root=repo,
            )
            self.assertTrue(report["ok"])
            self.assertTrue(report["boot_options_staged"])
            self.assertEqual(
                json.loads((boot_dir / "boot-options.json").read_text(encoding="utf-8")),
                {
                    "max_wall_ms": 900000,
                    "report_to_log": True,
                    "verify_window_marks": True,
                },
            )

            # File input with valid options.
            opts_src = repo / "work" / "input_opts.json"
            opts_src.write_text(
                json.dumps({"max_dispatches": 1000, "verify_window_marks": False}),
                encoding="utf-8",
            )
            report_file = sbb.stage_browser_boot(
                boot_dir=boot_dir,
                boot_options=str(opts_src),
                repo_root=repo,
            )
            self.assertTrue(report_file["ok"])
            self.assertTrue(report_file["boot_options_staged"])
            self.assertEqual(
                json.loads((boot_dir / "boot-options.json").read_text(encoding="utf-8")),
                {"max_dispatches": 1000, "verify_window_marks": False},
            )

            # Invalid options fail without clobbering existing boot-options.json.
            existing_text = (boot_dir / "boot-options.json").read_text(encoding="utf-8")
            bad_cases = [
                "{bad",
                "[]",
                '{"unknown_key": true}',
                '{"verify_window_marks": "true"}',
                '{"max_wall_ms": true}',
                '{"max_wall_ms": -1}',
            ]
            for bad_input in bad_cases:
                with self.subTest(bad_input=bad_input):
                    bad_report = sbb.stage_browser_boot(
                        boot_dir=boot_dir,
                        boot_options=bad_input,
                        repo_root=repo,
                    )
                    self.assertFalse(bad_report["ok"])
                    self.assertFalse(bad_report["boot_options_staged"])
                    self.assertEqual(bad_report["boot_options"]["status"], "refused")
                    self.assertEqual(
                        (boot_dir / "boot-options.json").read_text(encoding="utf-8"),
                        existing_text,
                    )

    def test_cli_sidecar_and_boot_options_json_and_text_output_sanitized(self) -> None:
        with tempfile.TemporaryDirectory(prefix="W119SecretHost_") as tmp:
            base = Path(tmp)
            repo = _make_repo_skeleton(base)
            src_pack = base / "external_pack.wgsl.pack"
            src_pack.write_bytes(_build_synthetic_wgpk(seed=9))
            sidecar_bytes = (json.dumps(_sample_reflect_sidecar(stage_count=2)) + "\n").encode(
                "utf-8"
            )
            Path(str(src_pack) + ".reflect.json").write_bytes(sidecar_bytes)

            ext_boot = base / "ext_boot_dir"
            ext_shaders = base / "ext_shaders_dir"

            # JSON CLI run
            buf_json = io.StringIO()
            with redirect_stdout(buf_json):
                rc_json = sbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--boot-dir",
                        str(ext_boot),
                        "--shader-pack",
                        str(src_pack),
                        "--shaders-dir",
                        str(ext_shaders),
                        "--verify-window-marks",
                        "--json",
                    ]
                )
            self.assertEqual(rc_json, 0)
            payload = json.loads(buf_json.getvalue())
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["sidecar_staged"])
            self.assertEqual(payload["sidecar_bytes"], len(sidecar_bytes))
            self.assertTrue(payload["boot_options_staged"])

            # Text CLI run (incremental unchanged)
            buf_text = io.StringIO()
            with redirect_stdout(buf_text):
                rc_text = sbb.main(
                    [
                        "--repo-root",
                        str(repo),
                        "--boot-dir",
                        str(ext_boot),
                        "--shader-pack",
                        str(src_pack),
                        "--shaders-dir",
                        str(ext_shaders),
                        "--verify-window-marks",
                    ]
                )
            self.assertEqual(rc_text, 0)
            text_out = buf_text.getvalue()
            self.assertIn("sidecar_staged=true", text_out)
            self.assertIn(f"sidecar_bytes={len(sidecar_bytes)}", text_out)
            self.assertIn("boot_options_staged=true", text_out)

            # Verify zero host paths or usernames in both JSON and text outputs.
            for output in (buf_json.getvalue(), text_out):
                self.assertNotIn("W119SecretHost_", output)
                self.assertNotIn(str(base), output)
                self.assertNotIn(str(base).replace("\\", "/"), output)
                self.assertNotRegex(output, r"[A-Za-z]:[\\/]")
                self.assertNotIn("/Users/", output)
                self.assertNotIn("/home/", output)
                self.assertNotIn("/tmp/", output)
                for username in sbb._current_usernames():
                    self.assertNotRegex(
                        output,
                        re.compile(
                            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
                            re.IGNORECASE,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
