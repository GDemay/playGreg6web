"""Unit tests for tools/check_wasm_exports.py browser-boot checks & sanitized JSON (T71, #1234)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_wasm_exports as wasm_check  # noqa: E402

_HOST_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]")
_HOST_UNC_PATH_RE = re.compile(r"\\\\[^\s]+")
_HOST_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/"
)


def _walk_json_keys_and_strings(value: Any) -> tuple[set[str], list[str]]:
    """Collect all dictionary keys and string values recursively from a JSON payload."""
    keys: set[str] = set()
    strings: list[str] = []

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                keys.add(str(k))
                _visit(v)
        elif isinstance(node, list):
            for item in node:
                _visit(item)
        elif isinstance(node, str):
            strings.append(node)

    _visit(value)
    return keys, strings


def _build_minimal_export_wasm(exports: list[tuple[str, str, int]]) -> bytes:
    """Build a minimal synthetic Wasm binary containing only a header + export section."""
    return wasm_check.build_wasm_module(exports=exports)


class TestBrowserBootRequiredExports(unittest.TestCase):
    """Tests for --browser-boot-required and validate_runtime_wasm(require_browser_boot=True)."""

    def test_all_required_underscored_func_exports_pass(self) -> None:
        raw = _build_minimal_export_wasm(
            [
                ("_main", "func", 0),
                ("_playgta6_get_input_state_ptr", "func", 1),
                ("_playgta6_audio_ring", "func", 2),
            ]
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(module, require_browser_boot=True)

        self.assertTrue(report.ok)
        self.assertTrue(report.all_browser_boot_exports_ok)
        self.assertEqual(report.missing_browser_boot_exports, ())
        self.assertEqual(report.mismatched_browser_boot_exports, ())
        self.assertEqual(report.errors, ())
        self.assertTrue(report.required_checks["browser_boot_exports"])

        by_name = {entry.name: entry for entry in report.browser_boot_exports}
        self.assertEqual(
            set(by_name.keys()),
            {"main", "playgta6_get_input_state_ptr", "playgta6_audio_ring"},
        )
        self.assertEqual(by_name["main"].status, "ok")
        self.assertEqual(by_name["main"].matched_name, "_main")
        self.assertEqual(by_name["main"].actual_kind, "func")
        self.assertEqual(by_name["main"].export_index, 0)
        self.assertEqual(
            by_name["playgta6_get_input_state_ptr"].matched_name,
            "_playgta6_get_input_state_ptr",
        )
        self.assertEqual(
            by_name["playgta6_audio_ring"].matched_name,
            "_playgta6_audio_ring",
        )

    def test_all_required_unprefixed_and_mixed_func_exports_pass(self) -> None:
        for export_names in (
            ("main", "playgta6_get_input_state_ptr", "playgta6_audio_ring"),
            ("main", "_playgta6_get_input_state_ptr", "playgta6_audio_ring"),
            ("_main", "playgta6_get_input_state_ptr", "_playgta6_audio_ring"),
        ):
            raw = _build_minimal_export_wasm(
                [(name, "func", idx) for idx, name in enumerate(export_names)]
            )
            module = wasm_check.parse_wasm_module(raw)
            report = wasm_check.validate_runtime_wasm(module, require_browser_boot=True)
            self.assertTrue(report.ok, f"expected pass for {export_names}: {report.errors}")
            self.assertTrue(report.all_browser_boot_exports_ok)
            matched = tuple(entry.matched_name for entry in report.browser_boot_exports)
            self.assertEqual(matched, export_names)

    def test_full_runtime_wasm_passes_all_required_flags_with_zero_warnings(self) -> None:
        raw = wasm_check.build_valid_runtime_wasm(
            is_memory64=True,
            is_shared=True,
            include_browser_boot_exports=True,
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
            require_host_imports=True,
            require_browser_boot=True,
        )

        self.assertTrue(report.ok)
        self.assertTrue(report.is_memory64)
        self.assertTrue(report.is_shared_memory)
        self.assertTrue(report.all_host_imports_ok)
        self.assertTrue(report.all_browser_boot_exports_ok)
        self.assertEqual(report.errors, ())
        self.assertEqual(report.warnings, ())

    def test_missing_audio_ring_fails_browser_boot_required(self) -> None:
        raw = _build_minimal_export_wasm(
            [
                ("_main", "func", 0),
                ("_playgta6_get_input_state_ptr", "func", 1),
            ]
        )
        module = wasm_check.parse_wasm_module(raw)

        strict_report = wasm_check.validate_runtime_wasm(
            module, require_browser_boot=True
        )
        self.assertFalse(strict_report.ok)
        self.assertFalse(strict_report.all_browser_boot_exports_ok)
        self.assertEqual(
            strict_report.missing_browser_boot_exports, ("playgta6_audio_ring",)
        )
        self.assertEqual(strict_report.mismatched_browser_boot_exports, ())
        self.assertEqual(len(strict_report.errors), 1)
        self.assertIn("playgta6_audio_ring", strict_report.errors[0])

        by_name = {entry.name: entry for entry in strict_report.browser_boot_exports}
        self.assertEqual(by_name["main"].status, "ok")
        self.assertEqual(by_name["playgta6_get_input_state_ptr"].status, "ok")
        self.assertEqual(by_name["playgta6_audio_ring"].status, "missing")
        self.assertFalse(by_name["playgta6_audio_ring"].present)
        self.assertIsNone(by_name["playgta6_audio_ring"].matched_name)
        self.assertIsNone(by_name["playgta6_audio_ring"].actual_kind)

        # Without require_browser_boot, report remains ok (existing behavior preserved).
        inspect_report = wasm_check.validate_runtime_wasm(
            module, require_browser_boot=False
        )
        self.assertTrue(inspect_report.ok)
        self.assertFalse(inspect_report.all_browser_boot_exports_ok)
        self.assertEqual(
            inspect_report.missing_browser_boot_exports, ("playgta6_audio_ring",)
        )

    def test_missing_input_state_fails_browser_boot_required(self) -> None:
        raw = _build_minimal_export_wasm(
            [
                ("main", "func", 0),
                ("playgta6_audio_ring", "func", 1),
            ]
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(module, require_browser_boot=True)

        self.assertFalse(report.ok)
        self.assertFalse(report.all_browser_boot_exports_ok)
        self.assertEqual(
            report.missing_browser_boot_exports, ("playgta6_get_input_state_ptr",)
        )
        self.assertEqual(report.mismatched_browser_boot_exports, ())
        self.assertEqual(len(report.errors), 1)
        self.assertIn("playgta6_get_input_state_ptr", report.errors[0])

    def test_missing_main_fails_browser_boot_required(self) -> None:
        raw = _build_minimal_export_wasm(
            [
                ("_playgta6_get_input_state_ptr", "func", 0),
                ("_playgta6_audio_ring", "func", 1),
            ]
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(module, require_browser_boot=True)

        self.assertFalse(report.ok)
        self.assertEqual(report.missing_browser_boot_exports, ("main",))
        self.assertEqual(report.mismatched_browser_boot_exports, ())
        self.assertTrue(any("main" in err for err in report.errors))

    def test_wrong_export_kind_rejected_as_non_callable(self) -> None:
        raw = _build_minimal_export_wasm(
            [
                ("_main", "table", 0),
                ("_playgta6_get_input_state_ptr", "memory", 0),
                ("_playgta6_audio_ring", "global", 0),
            ]
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(module, require_browser_boot=True)

        self.assertFalse(report.ok)
        self.assertFalse(report.all_browser_boot_exports_ok)
        self.assertEqual(report.missing_browser_boot_exports, ())
        self.assertEqual(
            report.mismatched_browser_boot_exports,
            ("main", "playgta6_get_input_state_ptr", "playgta6_audio_ring"),
        )
        self.assertEqual(len(report.errors), 3)

        by_name = {entry.name: entry for entry in report.browser_boot_exports}
        self.assertEqual(by_name["main"].status, "wrong_kind")
        self.assertEqual(by_name["main"].actual_kind, "table")
        self.assertTrue(by_name["main"].present)
        self.assertFalse(by_name["main"].ok)

        self.assertEqual(by_name["playgta6_get_input_state_ptr"].status, "wrong_kind")
        self.assertEqual(by_name["playgta6_get_input_state_ptr"].actual_kind, "memory")

        self.assertEqual(by_name["playgta6_audio_ring"].status, "wrong_kind")
        self.assertEqual(by_name["playgta6_audio_ring"].actual_kind, "global")

    def test_single_wrong_kind_audio_ring_with_other_valid_funcs(self) -> None:
        raw = _build_minimal_export_wasm(
            [
                ("_main", "func", 0),
                ("_playgta6_get_input_state_ptr", "func", 1),
                ("_playgta6_audio_ring", "tag", 0),
            ]
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(module, require_browser_boot=True)

        self.assertFalse(report.ok)
        self.assertEqual(report.missing_browser_boot_exports, ())
        self.assertEqual(report.mismatched_browser_boot_exports, ("playgta6_audio_ring",))
        by_name = {entry.name: entry for entry in report.browser_boot_exports}
        self.assertEqual(by_name["playgta6_audio_ring"].status, "wrong_kind")
        self.assertEqual(by_name["playgta6_audio_ring"].actual_kind, "tag")

    def test_callable_alias_preferred_over_non_callable_alias(self) -> None:
        raw = _build_minimal_export_wasm(
            [
                ("main", "global", 0),
                ("_main", "func", 0),
                ("_playgta6_get_input_state_ptr", "func", 1),
                ("_playgta6_audio_ring", "func", 2),
            ]
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(module, require_browser_boot=True)

        self.assertTrue(report.ok)
        by_name = {entry.name: entry for entry in report.browser_boot_exports}
        self.assertEqual(by_name["main"].status, "ok")
        self.assertEqual(by_name["main"].matched_name, "_main")
        self.assertEqual(by_name["main"].actual_kind, "func")


class TestSanitizationHelpers(unittest.TestCase):
    """Direct unit tests for host-path and username sanitization."""

    def test_sanitize_text_strips_windows_unc_posix_and_home_paths(self) -> None:
        samples = [
            r"failed reading C:\Users\alice\secret_project\runtime.wasm",
            r"unc path \\fileserver\private\game\runtime.wasm failed",
            "posix path /Users/alice/work/runtime.wasm failed",
            "linux path /home/bob/builds/runtime.wasm failed",
            "temp path /tmp/pytest-of-bob/runtime.wasm failed",
            f"home dir {Path.home() / 'secret_dir' / 'runtime.wasm'}",
        ]
        for sample in samples:
            cleaned = wasm_check.sanitize_text(sample)
            self.assertIsNone(_HOST_WINDOWS_PATH_RE.search(cleaned), cleaned)
            self.assertIsNone(_HOST_UNC_PATH_RE.search(cleaned), cleaned)
            self.assertIsNone(_HOST_POSIX_PATH_RE.search(cleaned), cleaned)
            self.assertNotIn("secret_project", cleaned)
            self.assertNotIn("secret_dir", cleaned)
            self.assertNotIn(str(Path.home()), cleaned)

    def test_sanitize_source_label_returns_basename_only(self) -> None:
        self.assertEqual(wasm_check.sanitize_source_label(None), "<memory>")
        self.assertEqual(wasm_check.sanitize_source_label("-"), "<stdin>")
        self.assertEqual(wasm_check.sanitize_source_label("<stdin>"), "<stdin>")
        self.assertEqual(
            wasm_check.sanitize_source_label(r"C:\Users\alice\work\runtime.wasm"),
            "runtime.wasm",
        )
        self.assertEqual(
            wasm_check.sanitize_source_label("/home/bob/private/runtime.wasm"),
            "runtime.wasm",
        )


class TestCheckWasmExportsCliAndJsonSanitization(unittest.TestCase):
    """CLI integration tests covering --browser-boot-required, --json sanitization, and legacy flags."""

    def _run_cli(
        self, args: list[str], *, stdin_bytes: bytes | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "check_wasm_exports.py"), *args],
            input=stdin_bytes,
            capture_output=True,
            check=False,
        )

    def _assert_zero_host_paths_in_json(
        self,
        raw_stdout: str,
        payload: dict[str, Any],
        forbidden_tokens: list[str],
    ) -> None:
        keys, strings = _walk_json_keys_and_strings(payload)
        self.assertNotIn("path", keys, f"'path' key must not appear in JSON: {keys}")

        for token in forbidden_tokens:
            if token and len(token) >= 3:
                self.assertNotIn(
                    token.lower(),
                    raw_stdout.lower(),
                    f"host token {token!r} leaked into --json stdout:\n{raw_stdout}",
                )

        self.assertIsNone(
            _HOST_WINDOWS_PATH_RE.search(raw_stdout),
            f"Windows drive path leaked into --json stdout:\n{raw_stdout}",
        )
        self.assertIsNone(
            _HOST_UNC_PATH_RE.search(raw_stdout),
            f"UNC path leaked into --json stdout:\n{raw_stdout}",
        )
        self.assertIsNone(
            _HOST_POSIX_PATH_RE.search(raw_stdout),
            f"POSIX host path leaked into --json stdout:\n{raw_stdout}",
        )
        for text in strings:
            self.assertIsNone(_HOST_WINDOWS_PATH_RE.search(text), text)
            self.assertIsNone(_HOST_UNC_PATH_RE.search(text), text)
            self.assertIsNone(_HOST_POSIX_PATH_RE.search(text), text)

    def test_cli_browser_boot_required_matrix_and_zero_host_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secret_dir = (
                Path(tmp) / "secret_host_workspace_t71" / "nested_private_dir_t71"
            )
            secret_dir.mkdir(parents=True)

            all_ok_path = secret_dir / "runtime_all_ok.wasm"
            no_audio_path = secret_dir / "runtime_no_audio.wasm"
            no_input_path = secret_dir / "runtime_no_input.wasm"
            wrong_kind_path = secret_dir / "runtime_wrong_kind.wasm"
            corrupt_path = secret_dir / "runtime_corrupt.wasm"
            missing_path = secret_dir / "runtime_missing_t71.wasm"

            all_ok_path.write_bytes(
                wasm_check.build_valid_runtime_wasm(include_browser_boot_exports=True)
            )
            no_audio_path.write_bytes(
                _build_minimal_export_wasm(
                    [
                        ("_main", "func", 0),
                        ("_playgta6_get_input_state_ptr", "func", 1),
                    ]
                )
            )
            no_input_path.write_bytes(
                _build_minimal_export_wasm(
                    [
                        ("main", "func", 0),
                        ("playgta6_audio_ring", "func", 1),
                    ]
                )
            )
            wrong_kind_path.write_bytes(
                _build_minimal_export_wasm(
                    [
                        ("_main", "func", 0),
                        ("_playgta6_get_input_state_ptr", "func", 1),
                        ("_playgta6_audio_ring", "global", 0),
                    ]
                )
            )
            corrupt_path.write_bytes(b"\x00asm\x01\x00\x00\x00\x07\x20\x01")

            forbidden_tokens = [
                str(tmp),
                str(secret_dir),
                secret_dir.as_posix(),
                "secret_host_workspace_t71",
                "nested_private_dir_t71",
                "runtime_all_ok.wasm",
                "runtime_no_audio.wasm",
                "runtime_no_input.wasm",
                "runtime_wrong_kind.wasm",
                "runtime_corrupt.wasm",
                "runtime_missing_t71.wasm",
                str(Path.home()),
                Path.home().as_posix(),
            ]
            # Also ensure OS username is not leaked if present in path.
            username = os.environ.get("USERNAME") or os.environ.get("USER") or ""
            if len(username) >= 4 and username.lower() not in {
                "main",
                "func",
                "wasm",
                "memory",
                "host",
                "root",
                "play",
                "ring",
                "true",
                "false",
                "none",
                "null",
                "name",
                "kind",
                "data",
                "code",
                "type",
                "size",
                "port",
                "user",
            }:
                forbidden_tokens.append(username)

            # 1) All required exports + Wasm64 + shared + host imports pass.
            proc = self._run_cli(
                [
                    str(all_ok_path),
                    "--require-memory64",
                    "--require-shared-memory",
                    "--require-host-imports",
                    "--browser-boot-required",
                    "--json",
                ]
            )
            stdout = proc.stdout.decode("utf-8")
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8"))
            payload = json.loads(stdout)
            self._assert_zero_host_paths_in_json(stdout, payload, forbidden_tokens)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["all_browser_boot_exports_ok"])
            self.assertEqual(payload["browser_boot_exports_valid"], 3)
            self.assertEqual(payload["browser_boot_exports_total"], 3)
            self.assertEqual(payload["missing_browser_boot_exports"], [])
            self.assertEqual(payload["mismatched_browser_boot_exports"], [])
            self.assertTrue(payload["required_checks"]["browser_boot_exports"])
            self.assertEqual(
                [e["name"] for e in payload["exports"]],
                [
                    "memory",
                    "_main",
                    "_playgta6_get_input_state_ptr",
                    "_playgta6_audio_ring",
                ],
            )

            # 2) Missing audio ring fails --browser-boot-required.
            proc = self._run_cli(
                [str(no_audio_path), "--browser-boot-required", "--json"]
            )
            stdout = proc.stdout.decode("utf-8")
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(stdout)
            self._assert_zero_host_paths_in_json(stdout, payload, forbidden_tokens)
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["all_browser_boot_exports_ok"])
            self.assertEqual(payload["browser_boot_exports_valid"], 2)
            self.assertEqual(
                payload["missing_browser_boot_exports"], ["playgta6_audio_ring"]
            )
            self.assertEqual(payload["mismatched_browser_boot_exports"], [])

            # 3) Missing input state fails --browser-boot-required.
            proc = self._run_cli(
                [str(no_input_path), "--browser-boot-required", "--json"]
            )
            stdout = proc.stdout.decode("utf-8")
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(stdout)
            self._assert_zero_host_paths_in_json(stdout, payload, forbidden_tokens)
            self.assertFalse(payload["ok"])
            self.assertEqual(
                payload["missing_browser_boot_exports"],
                ["playgta6_get_input_state_ptr"],
            )

            # 4) Wrong export kind fails --browser-boot-required.
            proc = self._run_cli(
                [str(wrong_kind_path), "--browser-boot-required", "--json"]
            )
            stdout = proc.stdout.decode("utf-8")
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(stdout)
            self._assert_zero_host_paths_in_json(stdout, payload, forbidden_tokens)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["missing_browser_boot_exports"], [])
            self.assertEqual(
                payload["mismatched_browser_boot_exports"], ["playgta6_audio_ring"]
            )
            by_name = {entry["name"]: entry for entry in payload["browser_boot_exports"]}
            self.assertEqual(by_name["playgta6_audio_ring"]["status"], "wrong_kind")
            self.assertEqual(by_name["playgta6_audio_ring"]["actual_kind"], "global")

            # 5) Malformed binary exits 1 with sanitized JSON (no host paths).
            proc = self._run_cli(
                [str(corrupt_path), "--browser-boot-required", "--json"]
            )
            stdout = proc.stdout.decode("utf-8")
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(stdout)
            self._assert_zero_host_paths_in_json(stdout, payload, forbidden_tokens)
            self.assertFalse(payload["ok"])
            self.assertIn("parse_error", payload)

            # 6) Missing file exits 2 with sanitized JSON (no host paths).
            proc = self._run_cli(
                [str(missing_path), "--browser-boot-required", "--json"]
            )
            stdout = proc.stdout.decode("utf-8")
            self.assertEqual(proc.returncode, 2)
            payload = json.loads(stdout)
            self._assert_zero_host_paths_in_json(stdout, payload, forbidden_tokens)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["error"], "wasm file not found")

            # 7) Text output smoke check: shows browser-boot section and never leaks parent dir.
            proc = self._run_cli([str(all_ok_path), "--browser-boot-required"])
            text_out = proc.stdout.decode("utf-8")
            self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8"))
            self.assertIn("Wasm ABI Validation [PASS]: runtime_all_ok.wasm", text_out)
            self.assertIn("Browser Boot:   OK (3/3 callable)", text_out)
            self.assertIn("Browser-boot callable exports:", text_out)
            self.assertNotIn("secret_host_workspace_t71", text_out)
            self.assertNotIn("nested_private_dir_t71", text_out)

    def test_cli_stdin_with_browser_boot_required_and_existing_flags(self) -> None:
        raw_ok = _build_minimal_export_wasm(
            [
                ("main", "func", 0),
                ("playgta6_get_input_state_ptr", "func", 1),
                ("playgta6_audio_ring", "func", 2),
            ]
        )
        proc = self._run_cli(
            ["-", "--browser-boot-required", "--json"], stdin_bytes=raw_ok
        )
        stdout = proc.stdout.decode("utf-8")
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8"))
        payload = json.loads(stdout)
        self.assertNotIn("path", payload)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["all_browser_boot_exports_ok"])

        # Preserved behavior: existing flags still enforce memory64/shared/host imports.
        proc_fail_mem = self._run_cli(
            ["-", "--require-memory64", "--browser-boot-required", "--json"],
            stdin_bytes=raw_ok,
        )
        self.assertEqual(proc_fail_mem.returncode, 1)
        payload_fail = json.loads(proc_fail_mem.stdout.decode("utf-8"))
        self.assertFalse(payload_fail["ok"])
        self.assertFalse(payload_fail["is_memory64"])
        self.assertTrue(payload_fail["all_browser_boot_exports_ok"])

    def test_validate_wasm_file_json_omits_host_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "private_host_folder_t71"
            nested.mkdir()
            wasm_file = nested / "runtime.wasm"
            wasm_file.write_bytes(
                wasm_check.build_valid_runtime_wasm(include_browser_boot_exports=True)
            )

            report = wasm_check.validate_wasm_file(
                wasm_file,
                require_memory64=True,
                require_shared_memory=True,
                require_host_imports=True,
                require_browser_boot=True,
            )
            self.assertTrue(report.ok)
            serialized = wasm_check.report_to_json(report)
            payload = json.loads(serialized)
            self._assert_zero_host_paths_in_json(
                serialized,
                payload,
                [str(tmp), str(nested), "private_host_folder_t71"],
            )


if __name__ == "__main__":
    unittest.main()
