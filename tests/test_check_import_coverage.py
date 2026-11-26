import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_import_coverage as cic  # noqa: E402

SCRIPT_PATH = str(REPO_ROOT / "tools" / "check_import_coverage.py")


def _write_file(root: Path, rel: str, content: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


class TestCoveragePctZeroDivGuard(unittest.TestCase):
    """Verify zero-division-safe coverage percentage math and empty-import edge cases."""

    def test_coverage_pct_zero_negative_overflow_and_non_finite(self) -> None:
        self.assertEqual(cic.coverage_pct(0, 0), 0.0)
        self.assertEqual(cic.coverage_pct(5, 0), 0.0)
        self.assertEqual(cic.coverage_pct(0, -10), 0.0)
        self.assertEqual(cic.coverage_pct(-3, 10), 0.0)
        self.assertEqual(cic.coverage_pct(15, 10), 100.0)
        self.assertEqual(cic.coverage_pct(float("nan"), 10), 0.0)
        self.assertEqual(cic.coverage_pct(10, float("nan")), 0.0)
        self.assertEqual(cic.coverage_pct(float("inf"), 10), 0.0)
        self.assertEqual(cic.coverage_pct(10, float("inf")), 0.0)
        self.assertEqual(cic.coverage_pct(1, 3), 33.33)
        self.assertEqual(cic.coverage_pct(2, 3), 66.67)
        self.assertEqual(cic.coverage_pct(7, 10), 70.0)
        self.assertEqual(cic.coverage_pct(10, 10), 100.0)

    def test_empty_imports_required_is_zero_div_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(
                root,
                "docs/imports_required.md",
                "# PE64 Inspection Report\n\n## Imports\n\n(no imports)\n",
            )
            report = cic.audit_import_coverage(root)
            self.assertFalse(report["ok"])
            self.assertEqual(report["totals"]["dll_count"], 0)
            self.assertEqual(report["totals"]["total"], 0)
            self.assertEqual(report["totals"]["coverage_pct"], 0.0)
            self.assertEqual(report["totals"]["shimmed_pct"], 0.0)
            self.assertEqual(report["totals"]["effective_boot_pct"], 0.0)
            self.assertEqual(report["totals"]["missing_pct"], 0.0)
            self.assertEqual(report["totals"]["complete_dlls"], 0)

            human = cic.format_human_report(report)
            self.assertIn("0/0 (0.00%)", human)

    def test_empty_dll_section_is_zero_div_safe_and_not_counted_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(
                root,
                "docs/imports_required.md",
                """## Imports

### `Empty.dll` (0 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |

### `KERNEL32.dll` (1 symbol)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `CreateFileW` | hint `0x0001` |
""",
            )
            _write_file(
                root,
                "runtime/os/k32.cpp",
                'void init() { register_export("KERNEL32.dll", "CreateFileW", 0, 1); }\n',
            )
            report = cic.audit_import_coverage(root, sort_dlls=True)
            self.assertTrue(report["ok"])
            by_dll = {d["dll"]: d for d in report["dlls"]}
            empty = by_dll["Empty.dll"]
            self.assertEqual(empty["total"], 0)
            self.assertEqual(empty["coverage_pct"], 0.0)
            self.assertEqual(empty["shimmed_pct"], 0.0)
            self.assertEqual(empty["effective_boot_pct"], 0.0)
            self.assertEqual(empty["status"], "missing")
            self.assertEqual(report["totals"]["complete_dlls"], 1)
            self.assertEqual(report["totals"]["coverage_pct"], 100.0)

    def test_empty_and_non_object_bridge_manifests(self) -> None:
        empty = {"format": cic.BRIDGE_FORMAT, "imports": []}
        report = cic.audit_bridge_manifest(empty)
        self.assertTrue(report["ok"])
        self.assertEqual(report["totals"]["total"], 0)
        self.assertEqual(report["totals"]["coverage_pct"], 0.0)

        bad_root = cic.audit_bridge_manifest(["not", "an", "object"])  # type: ignore[arg-type]
        self.assertFalse(bad_root["ok"])
        self.assertTrue(any("JSON object" in e for e in bad_root["errors"]))
        self.assertEqual(bad_root["totals"]["coverage_pct"], 0.0)


class TestCanonicalDeterministicSorting(unittest.TestCase):
    """Verify byte-for-byte deterministic sorting across permuted DLL and symbol inputs."""

    def _build_permuted_repo(self, root: Path, *, reversed_order: bool) -> Path:
        k32_rows = [
            "| name | `WriteFile` | hint `0x0003` |",
            "| name | `CreateFileW` | hint `0x0001` |",
            "| name | `alphaMissing` | hint `0x0004` |",
            "| name | `ZetaMissing` | hint `0x0005` |",
            "| name | `CloseHandle` | hint `0x0002` |",
        ]
        ws2_rows = [
            "| ordinal | `#115` | ordinal `115` |",
            "| ordinal | `#2` | ordinal `2` |",
            "| name | `WSAStartup` | hint `0x0001` |",
            "| ordinal | `#11` | ordinal `11` |",
            "| ordinal | `#111` | ordinal `111` |",
        ]
        msvcp_rows = [
            "| name | `_Mtx_unlock` | hint `0x0002` |",
            "| name | `??0ios_base@std@@IEAA@XZ` | hint `0x0001` |",
            "| name | `_Mtx_lock` | hint `0x0003` |",
        ]
        if reversed_order:
            k32_rows = list(reversed(k32_rows))
            ws2_rows = list(reversed(ws2_rows))
            msvcp_rows = list(reversed(msvcp_rows))

        sections = [
            (
                "### `WS2_32.dll` (5 symbols)\n\n"
                "| Kind | Symbol | Hint / Ordinal |\n| --- | --- | --- |\n"
                + "\n".join(ws2_rows)
            ),
            (
                "### `MSVCP140.dll` (3 symbols)\n\n"
                "| Kind | Symbol | Hint / Ordinal |\n| --- | --- | --- |\n"
                + "\n".join(msvcp_rows)
            ),
            (
                "### `KERNEL32.dll` (5 symbols)\n\n"
                "| Kind | Symbol | Hint / Ordinal |\n| --- | --- | --- |\n"
                + "\n".join(k32_rows)
            ),
        ]
        if reversed_order:
            sections = list(reversed(sections))

        _write_file(
            root,
            "docs/imports_required.md",
            "# PE64 Inspection Report\n\n## Imports\n\n" + "\n\n".join(sections) + "\n",
        )
        _write_file(
            root,
            "runtime/os/shims.cpp",
            """void init() {
    register_export("KERNEL32.dll", "WriteFile", 0, 1);
    register_export("KERNEL32.dll", "CreateFileW", 0, 2);
    register_export("KERNEL32.dll", "CloseHandle", 0, 3);
    register_export("ws2_32.dll", "WSAStartup", 0, 4);
    register_ordinal("ws2_32.dll", 111, 5);
    register_ordinal("ws2_32.dll", 2, 6);
}
""",
        )
        return root

    def test_permuted_imports_produce_identical_canonical_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = self._build_permuted_repo(Path(tmp_a), reversed_order=False)
            repo_b = self._build_permuted_repo(Path(tmp_b), reversed_order=True)

            report_a = cic.audit_import_coverage(repo_a, sort_dlls=True)
            report_b = cic.audit_import_coverage(repo_b, sort_dlls=True)

            self.assertEqual(json.dumps(report_a, indent=2), json.dumps(report_b, indent=2))
            self.assertEqual(
                cic.format_human_report(report_a, missing_only=True),
                cic.format_human_report(report_b, missing_only=True),
            )
            self.assertEqual(
                [d["dll"] for d in report_a["dlls"]],
                ["KERNEL32.dll", "MSVCP140.dll", "WS2_32.dll"],
            )

            by_dll = {d["dll"]: d for d in report_a["dlls"]}
            self.assertEqual(
                by_dll["KERNEL32.dll"]["shimmed_symbols"],
                ["CloseHandle", "CreateFileW", "WriteFile"],
            )
            self.assertEqual(
                by_dll["KERNEL32.dll"]["missing_symbols"],
                ["alphaMissing", "ZetaMissing"],
            )
            # Named symbols come before ordinals; ordinals sort numerically (#2 < #11 < #111 < #115).
            self.assertEqual(
                by_dll["WS2_32.dll"]["shimmed_symbols"],
                ["WSAStartup", "#2", "#111"],
            )
            self.assertEqual(
                by_dll["WS2_32.dll"]["missing_symbols"],
                ["#11", "#115"],
            )

            # canonicalize_report also normalizes reports built without sort_dlls=True.
            unsorted_a = cic.audit_import_coverage(repo_a, sort_dlls=False)
            unsorted_b = cic.audit_import_coverage(repo_b, sort_dlls=False)
            self.assertEqual(
                cic.canonicalize_report(unsorted_a),
                cic.canonicalize_report(unsorted_b),
            )

    def test_cli_sort_dlls_flag_is_deterministic_across_permutations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            repo_a = self._build_permuted_repo(Path(tmp_a), reversed_order=False)
            repo_b = self._build_permuted_repo(Path(tmp_b), reversed_order=True)

            proc_a = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--repo-root", str(repo_a), "--sort-dlls", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            proc_b = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--repo-root", str(repo_b), "--sort-dlls", "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_a.returncode, 0, msg=proc_a.stderr)
            self.assertEqual(proc_b.returncode, 0, msg=proc_b.stderr)
            self.assertEqual(proc_a.stdout, proc_b.stdout)

            data = json.loads(proc_a.stdout)
            self.assertEqual(
                [d["dll"] for d in data["dlls"]],
                ["KERNEL32.dll", "MSVCP140.dll", "WS2_32.dll"],
            )

    def test_bridge_manifest_is_deterministic_across_permuted_imports(self) -> None:
        imports = [
            {"importer": "z_mod.dll", "dll": "WS2_32.dll", "ordinal": 115, "status": "unresolved", "reason": "no_shim"},
            {"importer": "a_mod.dll", "dll": "d3d12_xs.dll", "name": "D3D12CreateDevice", "status": "bound", "shim": "pg6_d3d12_create_device"},
            {"importer": "a_mod.dll", "dll": "ADVAPI32.dll", "name": "RegOpenKeyW", "status": "unresolved", "reason": "denied", "recorded": "denied: policy"},
            {"importer": "z_mod.dll", "dll": "kernel32.dll", "name": "Sleep", "status": "bound", "shim": "pg6_win32_sleep"},
            {"importer": "a_mod.dll", "dll": "KERNEL32.dll", "name": "GetTickCount64", "status": "bound", "shim": "pg6_win32_get_tick_count64"},
            {"importer": "a_mod.dll", "dll": "ADVAPI32.dll", "name": "RegCloseKey", "status": "unresolved", "reason": "no_shim"},
        ]
        manifest_fwd = {"format": cic.BRIDGE_FORMAT, "imports": list(imports)}
        manifest_rev = {"format": cic.BRIDGE_FORMAT, "imports": list(reversed(imports))}
        manifest_shuf = {
            "format": cic.BRIDGE_FORMAT,
            "imports": [imports[3], imports[0], imports[5], imports[2], imports[4], imports[1]],
        }

        rep_fwd = cic.audit_bridge_manifest(manifest_fwd)
        rep_rev = cic.audit_bridge_manifest(manifest_rev)
        rep_shuf = cic.audit_bridge_manifest(manifest_shuf)

        self.assertEqual(json.dumps(rep_fwd, indent=2), json.dumps(rep_rev, indent=2))
        self.assertEqual(json.dumps(rep_fwd, indent=2), json.dumps(rep_shuf, indent=2))
        self.assertEqual(cic.format_bridge_report(rep_fwd), cic.format_bridge_report(rep_rev))
        self.assertEqual(list(rep_fwd["importers"].keys()), ["a_mod.dll", "z_mod.dll"])
        self.assertEqual(
            list(rep_fwd["dlls"].keys()),
            ["ADVAPI32.dll", "d3d12_xs.dll", "KERNEL32.dll", "kernel32.dll", "WS2_32.dll"],
        )
        self.assertEqual(
            [u["import"] for u in rep_fwd["unresolved"]],
            [
                "a_mod.dll: ADVAPI32.dll!RegCloseKey",
                "a_mod.dll: ADVAPI32.dll!RegOpenKeyW",
                "z_mod.dll: WS2_32.dll!#115",
            ],
        )
        self.assertEqual(
            rep_fwd["unrecorded"],
            [
                "a_mod.dll: ADVAPI32.dll!RegCloseKey",
                "z_mod.dll: WS2_32.dll!#115",
            ],
        )
        self.assertEqual(rep_fwd["totals"]["coverage_pct"], 50.0)


class TestHostPathAndUsernameSanitization(unittest.TestCase):
    """Verify strict recursive host-path and username sanitization in JSON outputs."""

    def test_sanitize_message_strips_windows_unc_posix_extra_and_username(self) -> None:
        with mock.patch.dict(os.environ, {"USERNAME": "secretdev", "USER": "secretdev"}, clear=False):
            raw = (
                r"failed at C:\Users\secretdev\proj\file.cpp and "
                r"\\buildbox\share\dump.bin and "
                r"/home/secretdev/src/app.py and "
                r"/custom_mount/worktree/docs/imports_required.md owned by secretdev "
                r"(keep user32.dll and ??0ios_base@std@@IEAA@XZ)"
            )
            cleaned = cic.sanitize_message(raw, extra_paths=("/custom_mount/worktree",))
            self.assertNotIn("secretdev", cleaned.lower())
            self.assertNotIn(r"C:\Users", cleaned)
            self.assertNotIn(r"\\buildbox", cleaned)
            self.assertNotIn("/home/", cleaned)
            self.assertNotIn("/custom_mount/worktree", cleaned)
            self.assertIn("<path>", cleaned)
            self.assertIn("<user>", cleaned)
            self.assertIn("user32.dll", cleaned)
            self.assertIn("??0ios_base@std@@IEAA@XZ", cleaned)

    def test_sanitize_report_recursive_preserves_schema_keys_and_strips_nested_paths(self) -> None:
        with mock.patch.dict(os.environ, {"USERNAME": "host", "USER": "marctestuser"}, clear=False):
            payload = {
                "ok": False,
                "host": 1,
                "importers": {
                    r"C:\Users\marctestuser\game\game_scarlett.exe": {"host": 1, "unresolved": 1},
                    "/Users/marctestuser/other/app.exe": {"host": 2, "unresolved": 0},
                },
                "unresolved": [
                    {
                        "import": r"C:\Users\marctestuser\game\game_scarlett.exe: foo.dll!Bar",
                        "reason": "no_shim",
                        "recorded": "see /home/marctestuser/notes.txt or marctestuser",
                    }
                ],
                "errors": [r"cannot open /custom/repo/root/file.json for marctestuser"],
            }
            sanitized = cic.sanitize_report(payload, repo_root="/custom/repo/root")
            # Fixed schema key "host" is never mangled even if USERNAME == "host".
            self.assertIn("host", sanitized)
            self.assertEqual(sanitized["host"], 1)
            # Colliding sanitized importer keys merge their integer counts cleanly.
            self.assertEqual(sanitized["importers"], {"<path>": {"host": 3, "unresolved": 1}})
            dumped = json.dumps(sanitized)
            self.assertNotIn("marctestuser", dumped.lower())
            self.assertNotIn("/custom/repo/root", dumped)
            self.assertNotIn(r"C:\Users", dumped)
            self.assertNotIn("/Users/", dumped)
            self.assertNotIn("/home/", dumped)

    def test_cli_bridge_json_sanitizes_host_paths_and_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "bridge.json"
            home_str = str(Path.home())
            manifest = {
                "format": cic.BRIDGE_FORMAT,
                "imports": [
                    {
                        "importer": str(tmp_path / "game_scarlett_master.exe"),
                        "dll": "MFPlat.DLL",
                        "name": "MFCreateAttributes",
                        "status": "unresolved",
                        "reason": "denied",
                        "recorded": f"blocked by {home_str}/audit.h and C:\\Users\\alice\\secret.txt",
                    }
                ],
            }
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            proc = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--bridge-manifest", str(manifest_path), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            raw_out = proc.stdout
            self.assertNotIn(str(tmp_path), raw_out)
            self.assertNotIn(str(tmp_path).replace("\\", "/"), raw_out)
            self.assertNotIn(home_str, raw_out)
            self.assertNotIn(home_str.replace("\\", "/"), raw_out)
            self.assertNotIn(r"C:\Users\alice", raw_out)

            # Missing bridge manifest error on stderr must also be sanitized.
            missing_path = tmp_path / "does_not_exist.json"
            proc_err = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--bridge-manifest", str(missing_path), "--json"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_err.returncode, 1)
            self.assertNotIn(str(tmp_path), proc_err.stderr)
            self.assertNotIn(str(tmp_path).replace("\\", "/"), proc_err.stderr)


class TestBackwardsCompatibleCliBehavior(unittest.TestCase):
    """Verify 100% backwards compatibility for existing CLI flags, JSON schema, and exit codes."""

    def test_existing_cli_flags_fields_and_default_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_file(
                root,
                "docs/imports_required.md",
                """## Imports

### `WS2_32.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| ordinal | `#115` | ordinal `115` |
| ordinal | `#111` | ordinal `111` |

### `KERNEL32.dll` (2 symbols)

| Kind | Symbol | Hint / Ordinal |
| --- | --- | --- |
| name | `ZetaApi` | hint `0x0002` |
| name | `AlphaApi` | hint `0x0001` |
""",
            )
            _write_file(
                root,
                "runtime/os/shims.cpp",
                """void init() {
    register_ordinal("ws2_32.dll", 111, 1);
    register_export("KERNEL32.dll", "AlphaApi", 0, 2);
}
""",
            )

            # Default --missing-only --json preserves legacy DLL order while sorting symbols.
            proc = subprocess.run(
                [
                    sys.executable,
                    SCRIPT_PATH,
                    "--repo-root",
                    str(root),
                    "--missing-only",
                    "--min-pct",
                    "50",
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            data = json.loads(proc.stdout)
            self.assertTrue(data["ok"])
            self.assertTrue(data["missing_only"])
            self.assertEqual([d["dll"] for d in data["dlls"]], ["WS2_32.dll", "KERNEL32.dll"])
            self.assertEqual(data["totals"]["effective_boot_pct"], 50.0)
            self.assertEqual(data["totals"]["coverage_pct"], 50.0)
            self.assertEqual(data["dlls"][0]["coverage_pct"], 50.0)
            self.assertNotIn(str(root), proc.stdout)
            self.assertNotIn(str(root).replace("\\", "/"), proc.stdout)

            # --min-pct 51 fails with exit code 1.
            proc_fail = subprocess.run(
                [sys.executable, SCRIPT_PATH, "--repo-root", str(root), "--min-pct", "51"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_fail.returncode, 1)
            self.assertIn("Effective Boot Coverage: 2/4 (50.00%)", proc_fail.stdout)


if __name__ == "__main__":
    unittest.main()
