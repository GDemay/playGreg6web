import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import inspect_game_assets as iga  # noqa: E402

SYNTHETIC_FULL_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <Identity Name="SyntheticStudio.DemoTitle"
            Publisher="CN=Synthetic Studio"
            Version="1.2.3.4" />
  <TitleId>7A1B2C3D</TitleId>
  <StoreId>9NBLGGH4SYNTH</StoreId>
  <MSAAppId>000000004819ABCD</MSAAppId>
  <ExecutableList>
    <Executable Name="DemoGame.exe"
                Id="Game"
                TargetDeviceFamily="Scarlett"
                Alias="DemoGameAlias.exe" />
    <Executable Name="Tools\\CrashHelper.exe"
                Id="CrashHelper"
                TargetDeviceFamily="XboxOne" />
  </ExecutableList>
  <ShellVisuals DefaultDisplayName="Synthetic Open World"
                SplashScreenImage="Assets\\SplashScreen.png"
                StoreLogo="Assets\\StoreLogo.png"
                Square150x150Logo="Assets\\Logo150.png"
                Square44x44Logo="Assets\\Logo44.png"
                Square480x480Logo="Assets\\Logo480.png"
                BackgroundColor="#101820" />
  <Resources>
    <Resource Language="en-US" />
    <Resource Language="de-DE" />
    <Resource Language="en-US" />
  </Resources>
  <VirtualMachine MemorySizeMB="10240">
    <SystemOsMemoryMB>2048</SystemOsMemoryMB>
  </VirtualMachine>
  <DesktopRegistration>
    <ModFolder Name="Mods" ReadOnly="true" />
    <CustomInstallActions />
  </DesktopRegistration>
</Game>
"""

PNG_STUB = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR" + b"\x01" * 32
PE_STUB = b"MZ" + b"\x90\x00\x03\x00" + b"\x04" * 128
RPF9_STUB = b"RPF9" + b"\x01\x00\x00\x00" + b"\x22" * 64


class TestParseMicrosoftGameConfig(unittest.TestCase):
    def test_parses_full_synthetic_xbox_series_config(self) -> None:
        parsed = iga.parse_microsoft_game_config(SYNTHETIC_FULL_CONFIG)

        self.assertEqual(parsed["configVersion"], "1")
        self.assertEqual(
            parsed["identity"],
            {
                "name": "SyntheticStudio.DemoTitle",
                "publisher": "CN=Synthetic Studio",
                "version": "1.2.3.4",
            },
        )
        self.assertEqual(parsed["titleId"], "7A1B2C3D")
        self.assertEqual(parsed["storeId"], "9NBLGGH4SYNTH")
        self.assertEqual(parsed["msaAppId"], "000000004819ABCD")

        self.assertEqual(len(parsed["executables"]), 2)
        self.assertEqual(
            parsed["executables"][0],
            {
                "name": "DemoGame.exe",
                "id": "Game",
                "targetDeviceFamily": "Scarlett",
                "alias": "DemoGameAlias.exe",
            },
        )
        self.assertEqual(
            parsed["executables"][1],
            {
                "name": "Tools/CrashHelper.exe",
                "id": "CrashHelper",
                "targetDeviceFamily": "XboxOne",
                "alias": None,
            },
        )

        self.assertEqual(
            parsed["shellVisuals"],
            {
                "defaultDisplayName": "Synthetic Open World",
                "splashScreenImage": "Assets/SplashScreen.png",
                "storeLogo": "Assets/StoreLogo.png",
                "square150x150Logo": "Assets/Logo150.png",
                "square44x44Logo": "Assets/Logo44.png",
                "square480x480Logo": "Assets/Logo480.png",
                "backgroundColor": "#101820",
            },
        )
        self.assertEqual(parsed["resources"]["languages"], ["en-US", "de-DE"])
        self.assertIsNotNone(parsed["virtualMachine"])
        self.assertTrue(parsed["virtualMachine"]["present"])
        self.assertEqual(parsed["virtualMachine"]["MemorySizeMB"], "10240")
        self.assertEqual(parsed["virtualMachine"]["SystemOsMemoryMB"], "2048")
        self.assertEqual(parsed["modFolder"], {"name": "Mods", "readOnly": True})
        self.assertEqual(
            parsed["desktopRegistration"],
            {
                "present": True,
                "hasModFolder": True,
                "childTags": ["CustomInstallActions", "ModFolder"],
            },
        )

    def test_parses_from_path_with_utf8_bom_and_utf16(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            utf8_path = Path(tmp) / "MicrosoftGame.Config"
            utf8_path.write_bytes(b"\xef\xbb\xbf" + SYNTHETIC_FULL_CONFIG.encode("utf-8"))
            parsed_utf8 = iga.parse_microsoft_game_config(utf8_path)
            self.assertEqual(parsed_utf8["titleId"], "7A1B2C3D")

            utf16_path = Path(tmp) / "MicrosoftGame16.Config"
            utf16_path.write_bytes(SYNTHETIC_FULL_CONFIG.encode("utf-16"))
            parsed_utf16 = iga.parse_microsoft_game_config(utf16_path)
            self.assertEqual(parsed_utf16["identity"]["name"], "SyntheticStudio.DemoTitle")

    def test_rejects_xxe_doctype_and_entity_declarations(self) -> None:
        xxe_doctype = """<?xml version="1.0"?>
<!DOCTYPE Game [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<Game configVersion="1"><TitleId>&xxe;</TitleId></Game>
"""
        with self.assertRaises(iga.UnsafeXmlError):
            iga.parse_microsoft_game_config(xxe_doctype)

        xxe_lowercase = '<!doctype foo><Game configVersion="1"/>'
        with self.assertRaises(iga.UnsafeXmlError):
            iga.parse_microsoft_game_config(xxe_lowercase)

        entity_only = '<!ENTITY secret "leak"><Game configVersion="1"/>'
        with self.assertRaises(iga.UnsafeXmlError):
            iga.parse_microsoft_game_config(entity_only)

    def test_rejects_oversized_config_and_malformed_xml(self) -> None:
        oversized = '<Game configVersion="1">' + ("A" * 2048) + "</Game>"
        with self.assertRaises(iga.ConfigTooLargeError):
            iga.parse_microsoft_game_config(oversized, max_bytes=1024)

        with tempfile.TemporaryDirectory() as tmp:
            big_file = Path(tmp) / "MicrosoftGame.Config"
            big_file.write_bytes(b"<Game>" + (b"X" * 4096) + b"</Game>")
            with self.assertRaises(iga.ConfigTooLargeError):
                iga.parse_microsoft_game_config(big_file, max_bytes=512)

        with self.assertRaises(iga.GameConfigError):
            iga.parse_microsoft_game_config("<Game><Unclosed></Game>")

        with self.assertRaises(iga.GameConfigError):
            iga.parse_microsoft_game_config("<NotAGame configVersion='1'/>")


class TestInspectGameDirectory(unittest.TestCase):
    def test_cross_checks_present_missing_and_zero_filled_partial_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            game_dir = Path(tmp) / "package"
            (game_dir / "Assets").mkdir(parents=True)
            (game_dir / "Tools").mkdir(parents=True)
            (game_dir / "Data").mkdir(parents=True)

            # Lowercase filename verifies case-insensitive MicrosoftGame.Config lookup.
            (game_dir / "microsoftgame.config").write_text(SYNTHETIC_FULL_CONFIG, encoding="utf-8")

            # Present & complete primary executable; second executable intentionally missing.
            (game_dir / "DemoGame.exe").write_bytes(PE_STUB)

            # SplashScreen present and valid PNG; StoreLogo zero-filled (partial download);
            # Logo150 zero-byte empty file; Logo44 and Logo480 missing.
            (game_dir / "Assets" / "SplashScreen.png").write_bytes(PNG_STUB)
            (game_dir / "Assets" / "StoreLogo.png").write_bytes(b"\x00" * 4096)
            (game_dir / "Assets" / "Logo150.png").write_bytes(b"")

            # Bundled DLLs (1 complete, 1 zero-filled partial) and RPFs (1 complete, 1 partial).
            (game_dir / "xgame_runtime.dll").write_bytes(PE_STUB)
            (game_dir / "Tools" / "partial_helper.dll").write_bytes(b"\x00" * 1024)
            (game_dir / "Data" / "common.rpf").write_bytes(RPF9_STUB)
            (game_dir / "Data" / "world_partial.rpf").write_bytes(b"\x00" * 2048)

            report = iga.inspect_game_directory(game_dir)

            self.assertTrue(report["ok"])
            self.assertTrue(report["configValid"])
            self.assertTrue(report["configFound"])
            self.assertEqual(report["configRelativePath"], "microsoftgame.config")

            # Executables: 1 ok, 1 missing
            exes = {item["id"]: item for item in report["executables"]}
            self.assertTrue(exes["Game"]["present"])
            self.assertTrue(exes["Game"]["complete"])
            self.assertTrue(exes["Game"]["nonZeroHeader"])
            self.assertEqual(exes["Game"]["headerFormat"], "pe")
            self.assertEqual(exes["Game"]["status"], "ok")
            self.assertEqual(exes["Game"]["sizeBytes"], len(PE_STUB))

            self.assertFalse(exes["CrashHelper"]["present"])
            self.assertFalse(exes["CrashHelper"]["complete"])
            self.assertEqual(exes["CrashHelper"]["status"], "missing")

            # ShellVisuals: ok, zero-filled partial, empty, missing
            vis = {item["field"]: item for item in report["shellVisuals"]}
            self.assertEqual(vis["splashScreenImage"]["status"], "ok")
            self.assertEqual(vis["splashScreenImage"]["headerFormat"], "png")
            self.assertTrue(vis["splashScreenImage"]["nonZeroHeader"])

            self.assertEqual(vis["storeLogo"]["status"], "incomplete_zero_header")
            self.assertTrue(vis["storeLogo"]["present"])
            self.assertFalse(vis["storeLogo"]["nonZeroHeader"])
            self.assertTrue(vis["storeLogo"]["partialDownloadSuspected"])
            self.assertEqual(vis["storeLogo"]["sizeBytes"], 4096)

            self.assertEqual(vis["square150x150Logo"]["status"], "empty")
            self.assertTrue(vis["square150x150Logo"]["partialDownloadSuspected"])
            self.assertEqual(vis["square44x44Logo"]["status"], "missing")
            self.assertEqual(vis["square480x480Logo"]["status"], "missing")

            # Inventory
            inv = report["inventory"]
            self.assertEqual(inv["dllCount"], 2)
            self.assertEqual(inv["dllCompleteCount"], 1)
            self.assertEqual(inv["dllIncompleteCount"], 1)
            dll_map = {item["basename"]: item for item in inv["dlls"]}
            self.assertEqual(dll_map["xgame_runtime.dll"]["status"], "ok")
            self.assertEqual(dll_map["partial_helper.dll"]["status"], "incomplete_zero_header")
            self.assertEqual(dll_map["partial_helper.dll"]["relativePath"], "Tools/partial_helper.dll")

            self.assertEqual(inv["rpfCount"], 2)
            self.assertEqual(inv["rpfCompleteCount"], 1)
            self.assertEqual(inv["rpfIncompleteCount"], 1)

            # Readiness summary
            rd = report["readiness"]
            self.assertFalse(rd["allDeclaredAssetsComplete"])
            self.assertTrue(rd["hasPartialDownloads"])
            self.assertEqual(rd["declaredExecutableCount"], 2)
            self.assertEqual(rd["completeExecutableCount"], 1)
            self.assertEqual(rd["missingExecutableCount"], 1)
            self.assertEqual(rd["declaredVisualCount"], 5)
            self.assertEqual(rd["completeVisualCount"], 1)
            self.assertEqual(rd["incompleteVisualCount"], 2)
            self.assertEqual(rd["missingVisualCount"], 2)

    def test_locates_config_in_content_subdirectory_and_blocks_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            secret_outside = base / "outside_secret.exe"
            secret_outside.write_bytes(PE_STUB)

            game_dir = base / "xbox_root"
            content_dir = game_dir / "Content"
            content_dir.mkdir(parents=True)

            traversal_xml = """<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <Identity Name="Test.Traversal" Publisher="CN=Test" Version="1.0.0.0" />
  <ExecutableList>
    <Executable Name="Good.exe" Id="Good" TargetDeviceFamily="Scarlett" />
    <Executable Name="..\\..\\outside_secret.exe" Id="Escape" TargetDeviceFamily="Scarlett" />
    <Executable Name="C:\\Users\\AliceSecret\\evil.exe" Id="AbsPath" TargetDeviceFamily="PC" />
  </ExecutableList>
</Game>
"""
            (content_dir / "MicrosoftGame.Config").write_text(traversal_xml, encoding="utf-8")
            (content_dir / "Good.exe").write_bytes(PE_STUB)

            report = iga.inspect_game_directory(game_dir)
            self.assertTrue(report["configValid"])
            self.assertEqual(report["configRelativePath"], "Content/MicrosoftGame.Config")

            exes = {item["id"]: item for item in report["executables"]}
            self.assertEqual(exes["Good"]["status"], "ok")
            self.assertEqual(exes["Good"]["relativePath"], "Content/Good.exe")
            self.assertEqual(exes["Escape"]["status"], "unsafe_path")
            self.assertFalse(exes["Escape"]["present"])
            self.assertEqual(exes["AbsPath"]["status"], "unsafe_path")
            self.assertFalse(exes["AbsPath"]["present"])

            serialized = json.dumps(report)
            self.assertNotIn("AliceSecret", serialized)
            self.assertNotIn("C:\\Users", serialized)


class TestCliAndSanitization(unittest.TestCase):
    def test_cli_json_never_leaks_host_paths_usernames_or_raw_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            repo = base / "PrivateRepo_DoNotLeak"
            repo.mkdir()
            secret_game_dir = base / "TopSecretUser_XboxDump_Folder"
            (secret_game_dir / "Assets").mkdir(parents=True)

            (secret_game_dir / "MicrosoftGame.Config").write_text(
                SYNTHETIC_FULL_CONFIG, encoding="utf-8"
            )
            marker_bytes = b"MZ" + b"SECRET_BINARY_PAYLOAD_DO_NOT_LEAK"
            (secret_game_dir / "DemoGame.exe").write_bytes(marker_bytes)
            (secret_game_dir / "Assets" / "SplashScreen.png").write_bytes(
                b"\x89PNG\r\n\x1a\nSECRET_PNG_BYTES_DO_NOT_LEAK"
            )
            (repo / ".env").write_text(
                f'GTA6_GAME_DIR="{secret_game_dir}"\n',
                encoding="utf-8",
            )

            clean_env = {k: v for k, v in os.environ.items() if k != "GTA6_GAME_DIR"}
            buf = io.StringIO()
            with mock.patch.dict(os.environ, clean_env, clear=True), redirect_stdout(buf):
                rc = iga.main(["--from-env", "--repo-root", str(repo), "--json"])

            self.assertEqual(rc, 0)
            raw_json = buf.getvalue()
            payload = json.loads(raw_json)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["configValid"])

            # Also verify --from-env with unset GTA6_GAME_DIR exits 1 with sanitized JSON.
            empty_repo = base / "UnconfiguredRepo_SecretName"
            empty_repo.mkdir()
            buf_unset = io.StringIO()
            with mock.patch.dict(os.environ, clean_env, clear=True), redirect_stdout(buf_unset):
                rc_unset = iga.main(["--from-env", "--repo-root", str(empty_repo), "--json"])
            self.assertEqual(rc_unset, 1)
            unset_payload = json.loads(buf_unset.getvalue())
            self.assertFalse(unset_payload["ok"])
            self.assertEqual(unset_payload["status"], "env_not_configured")
            self.assertNotIn("UnconfiguredRepo_SecretName", buf_unset.getvalue())

            for forbidden in (
                str(base),
                str(repo),
                str(secret_game_dir),
                "PrivateRepo_DoNotLeak",
                "TopSecretUser_XboxDump_Folder",
                "SECRET_BINARY_PAYLOAD_DO_NOT_LEAK",
                "SECRET_PNG_BYTES_DO_NOT_LEAK",
                os.environ.get("USERNAME", ""),
                os.environ.get("USER", ""),
            ):
                if forbidden and len(forbidden) > 2:
                    self.assertNotIn(forbidden, raw_json)

    def test_cli_exit_codes_for_valid_missing_and_malformed_configs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # 1. Missing MicrosoftGame.Config -> exit 1
            empty_dir = base / "empty_game"
            empty_dir.mkdir()
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_missing = iga.main(["--dir", str(empty_dir), "--json"])
            self.assertEqual(rc_missing, 1)
            missing_data = json.loads(buf.getvalue())
            self.assertFalse(missing_data["configValid"])
            self.assertEqual(missing_data["status"], "missing_config")
            self.assertNotIn(str(empty_dir), buf.getvalue())

            # 2. Malformed / XXE MicrosoftGame.Config -> exit 1
            xxe_dir = base / "xxe_game"
            xxe_dir.mkdir()
            (xxe_dir / "MicrosoftGame.Config").write_text(
                '<!DOCTYPE Game [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><Game/>',
                encoding="utf-8",
            )
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc_xxe = iga.main(["--dir", str(xxe_dir), "--json"])
            self.assertEqual(rc_xxe, 1)
            xxe_data = json.loads(buf.getvalue())
            self.assertFalse(xxe_data["configValid"])
            self.assertEqual(xxe_data["status"], "invalid_config")
            self.assertNotIn(str(xxe_dir), buf.getvalue())

            # 3. Valid config via subprocess CLI (human-readable + --json) -> exit 0
            valid_dir = base / "valid_game"
            valid_dir.mkdir()
            (valid_dir / "MicrosoftGame.Config").write_text(SYNTHETIC_FULL_CONFIG, encoding="utf-8")
            (valid_dir / "DemoGame.exe").write_bytes(PE_STUB)

            proc_text = subprocess.run(
                [sys.executable, str(REPO_ROOT / "tools" / "inspect_game_assets.py"), "--dir", str(valid_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_text.returncode, 0, msg=proc_text.stderr)
            self.assertIn("Synthetic Open World", proc_text.stdout)
            self.assertIn("7A1B2C3D", proc_text.stdout)

            proc_json = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "inspect_game_assets.py"),
                    "--dir",
                    str(valid_dir),
                    "--json",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc_json.returncode, 0, msg=proc_json.stderr)
            cli_payload = json.loads(proc_json.stdout)
            self.assertTrue(cli_payload["ok"])
            self.assertEqual(cli_payload["config"]["titleId"], "7A1B2C3D")
            self.assertNotIn(str(valid_dir), proc_json.stdout)


class TestDeterministicArchiveInventoryAndSummary(unittest.TestCase):
    def test_archive_inventory_sorted_deterministically_independent_of_iterdir_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            game_dir = Path(tmp) / "synth_pkg"
            (game_dir / "Data" / "SubB").mkdir(parents=True)
            (game_dir / "Data" / "SubA").mkdir(parents=True)
            (game_dir / "Bin").mkdir(parents=True)

            minimal_cfg = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<Game configVersion="1">'
                '<Identity Name="Synth.OrderTest" Publisher="CN=Synth" Version="1.0.0.0" />'
                "<TitleId>11223344</TitleId>"
                "</Game>"
            )
            (game_dir / "MicrosoftGame.Config").write_text(minimal_cfg, encoding="utf-8")

            # Synthetic DLLs in multiple directories
            (game_dir / "z_last.dll").write_bytes(PE_STUB)
            (game_dir / "Bin" / "a_first.dll").write_bytes(PE_STUB)

            # Synthetic RPFs with distinct and tied sizes, plus empty & zero-filled partials
            (game_dir / "Data" / "SubB" / "zebra_large_asset.rpf").write_bytes(RPF9_STUB + b"\x33" * 200)
            (game_dir / "Data" / "SubA" / "alpha_tie_asset.rpf").write_bytes(RPF9_STUB)
            (game_dir / "Data" / "SubB" / "beta_tie_asset.rpf").write_bytes(RPF9_STUB)
            (game_dir / "Data" / "partial_zero_asset.rpf").write_bytes(b"\x00" * 128)
            (game_dir / "Data" / "empty_zero_asset.rpf").write_bytes(b"")

            report_normal = iga.inspect_game_directory(game_dir)

            orig_iterdir = Path.iterdir

            def reversed_iterdir(path_self: Path):
                return iter(list(orig_iterdir(path_self))[::-1])

            with mock.patch.object(Path, "iterdir", reversed_iterdir):
                report_reversed = iga.inspect_game_directory(game_dir)

            json_normal = json.dumps(report_normal, indent=2)
            json_reversed = json.dumps(report_reversed, indent=2)
            self.assertEqual(json_normal, json_reversed)

            # Verify zero .rpf filenames leak into serialized JSON output.
            for secret_name in (
                "zebra_large_asset",
                "alpha_tie_asset",
                "beta_tie_asset",
                "partial_zero_asset",
                "empty_zero_asset",
            ):
                self.assertNotIn(secret_name, json_normal)

            inv = report_normal["inventory"]
            # DLLs strictly sorted by relativePath
            self.assertEqual(
                [item["relativePath"] for item in inv["dlls"]],
                ["Bin/a_first.dll", "z_last.dll"],
            )
            self.assertEqual(inv["dllTotalBytes"], 2 * len(PE_STUB))

            # Archives strictly sorted by (-sizeBytes, pathHash)
            archives = inv["archives"]
            self.assertEqual(len(archives), 5)
            sort_keys = [(-item["sizeBytes"], item["pathHash"]) for item in archives]
            self.assertEqual(sort_keys, sorted(sort_keys))
            self.assertEqual(archives[0]["sizeBytes"], len(RPF9_STUB) + 200)
            self.assertEqual(archives[-1]["sizeBytes"], 0)
            self.assertEqual(archives[-1]["status"], "empty")
            for item in archives:
                self.assertEqual(len(item["pathHash"]), 16)
                self.assertEqual(item["extension"], ".rpf")
                self.assertGreaterEqual(item["sizeRatio"], 0.0)
                self.assertLessEqual(item["sizeRatio"], 1.0)

            # Summary metrics match exact counts and ratios
            summary = inv["archiveSummary"]
            expected_complete_bytes = (len(RPF9_STUB) + 200) + 2 * len(RPF9_STUB)
            expected_incomplete_bytes = 128
            expected_total_bytes = expected_complete_bytes + expected_incomplete_bytes
            self.assertEqual(summary["archiveCount"], 5)
            self.assertEqual(summary["completeCount"], 3)
            self.assertEqual(summary["incompleteCount"], 2)
            self.assertEqual(summary["emptyCount"], 1)
            self.assertEqual(summary["zeroHeaderCount"], 1)
            self.assertEqual(summary["unreadableCount"], 0)
            self.assertEqual(summary["totalBytes"], expected_total_bytes)
            self.assertEqual(summary["completeBytes"], expected_complete_bytes)
            self.assertEqual(summary["incompleteBytes"], expected_incomplete_bytes)
            self.assertEqual(summary["minBytes"], 0)
            self.assertEqual(summary["maxBytes"], len(RPF9_STUB) + 200)
            self.assertAlmostEqual(summary["avgBytes"], round(expected_total_bytes / 5, 2))
            self.assertAlmostEqual(summary["completeCountRatio"], 0.6)
            self.assertAlmostEqual(summary["incompleteCountRatio"], 0.4)
            self.assertAlmostEqual(
                summary["completeByteRatio"],
                round(expected_complete_bytes / expected_total_bytes, 6),
            )
            self.assertAlmostEqual(
                summary["incompleteByteRatio"],
                round(expected_incomplete_bytes / expected_total_bytes, 6),
            )
            self.assertEqual(
                summary["statusCounts"],
                {"empty": 1, "incomplete_zero_header": 1, "ok": 3},
            )
            self.assertEqual(
                summary["headerFormatCounts"],
                {"empty": 1, "rpf": 3, "zero_filled": 1},
            )

            rd = report_normal["readiness"]
            self.assertAlmostEqual(rd["dllCompleteRatio"], 1.0)
            self.assertAlmostEqual(rd["archiveCompleteRatio"], 0.6)
            self.assertAlmostEqual(rd["archiveCompleteByteRatio"], summary["completeByteRatio"])

    def test_zero_archive_and_zero_byte_zero_division_guards(self) -> None:
        self.assertEqual(iga.safe_ratio(42, 0), 0.0)
        self.assertEqual(iga.safe_ratio(42, -10), 0.0)
        self.assertEqual(iga.safe_ratio(0, 0), 0.0)

        empty_summary = iga.summarize_archive_inventory([])
        self.assertEqual(empty_summary["archiveCount"], 0)
        self.assertEqual(empty_summary["totalBytes"], 0)
        self.assertEqual(empty_summary["minBytes"], 0)
        self.assertEqual(empty_summary["maxBytes"], 0)
        self.assertEqual(empty_summary["avgBytes"], 0.0)
        self.assertEqual(empty_summary["completeCountRatio"], 0.0)
        self.assertEqual(empty_summary["incompleteCountRatio"], 0.0)
        self.assertEqual(empty_summary["completeByteRatio"], 0.0)
        self.assertEqual(empty_summary["incompleteByteRatio"], 0.0)
        self.assertEqual(empty_summary["largestArchiveByteRatio"], 0.0)
        self.assertEqual(empty_summary["statusCounts"], {})
        self.assertEqual(empty_summary["headerFormatCounts"], {})

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)

            # 1. Missing directory -> zero-division safe
            missing_report = iga.inspect_game_directory(base / "does_not_exist")
            self.assertEqual(missing_report["status"], "missing_game_dir")
            self.assertEqual(missing_report["inventory"]["archiveSummary"]["avgBytes"], 0.0)
            self.assertEqual(missing_report["readiness"]["archiveCompleteRatio"], 0.0)

            # 2. Empty directory (0 files, 0 archives) -> zero-division safe
            empty_dir = base / "empty_dir"
            empty_dir.mkdir()
            empty_report = iga.inspect_game_directory(empty_dir)
            self.assertEqual(empty_report["status"], "missing_config")
            self.assertEqual(empty_report["inventory"]["archives"], [])
            self.assertEqual(empty_report["inventory"]["archiveSummary"]["completeByteRatio"], 0.0)
            self.assertEqual(empty_report["readiness"]["dllCompleteRatio"], 0.0)

            # 3. Valid minimal config with 0 executables, 0 visuals, 0 DLLs, 0 RPFs
            zero_assets_dir = base / "zero_assets"
            zero_assets_dir.mkdir()
            (zero_assets_dir / "MicrosoftGame.Config").write_text(
                '<Game configVersion="1"><Identity Name="Zero.Assets" Publisher="CN=Zero" Version="1.0.0.0" /></Game>',
                encoding="utf-8",
            )
            zero_report = iga.inspect_game_directory(zero_assets_dir)
            self.assertTrue(zero_report["ok"])
            self.assertTrue(zero_report["readiness"]["allDeclaredAssetsComplete"])
            self.assertEqual(zero_report["readiness"]["executableCompleteRatio"], 0.0)
            self.assertEqual(zero_report["readiness"]["visualCompleteRatio"], 0.0)
            self.assertEqual(zero_report["readiness"]["archiveCompleteRatio"], 0.0)
            self.assertEqual(zero_report["readiness"]["archiveCompleteByteRatio"], 0.0)

            # 4. Multiple 0-byte empty archives (archiveCount > 0, totalBytes == 0)
            zero_bytes_dir = base / "zero_byte_archives"
            zero_bytes_dir.mkdir()
            (zero_bytes_dir / "MicrosoftGame.Config").write_text(
                '<Game configVersion="1"><Identity Name="Zero.Bytes" Publisher="CN=Zero" Version="1.0.0.0" /></Game>',
                encoding="utf-8",
            )
            (zero_bytes_dir / "a.rpf").write_bytes(b"")
            (zero_bytes_dir / "b.rpf").write_bytes(b"")
            zb_report = iga.inspect_game_directory(zero_bytes_dir)
            zb_summary = zb_report["inventory"]["archiveSummary"]
            self.assertEqual(zb_summary["archiveCount"], 2)
            self.assertEqual(zb_summary["emptyCount"], 2)
            self.assertEqual(zb_summary["totalBytes"], 0)
            self.assertEqual(zb_summary["avgBytes"], 0.0)
            self.assertEqual(zb_summary["completeCountRatio"], 0.0)
            self.assertEqual(zb_summary["incompleteCountRatio"], 1.0)
            self.assertEqual(zb_summary["completeByteRatio"], 0.0)
            self.assertEqual(zb_summary["incompleteByteRatio"], 0.0)
            self.assertEqual(zb_summary["largestArchiveByteRatio"], 0.0)
            for arch in zb_report["inventory"]["archives"]:
                self.assertEqual(arch["sizeRatio"], 0.0)


class TestStrictHostPathSanitizationAndBackwardsCompat(unittest.TestCase):
    def test_json_sanitizes_windows_posix_unc_file_url_tilde_and_usernames(self) -> None:
        hostile_xml = """<?xml version="1.0" encoding="utf-8"?>
<Game configVersion="1">
  <Identity Name="MockDevUser99.DemoTitle"
            Publisher="CN=Built by mock_unix_user99 at C:\\Users\\AliceSecret\\Build"
            Version="1.0.0.0" />
  <TitleId>DEADBEEF</TitleId>
  <StoreId>file:///C:/Users/AliceSecret/store.txt</StoreId>
  <MSAAppId>//corp-nas02/private_share/BobSecret/app</MSAAppId>
  <ExecutableList>
    <Executable Name="C:\\Users\\AliceSecret\\My Games\\evil.exe" Id="WinAbs" TargetDeviceFamily="PC" />
    <Executable Name="\\\\corp-nas01\\private_share\\AliceSecret\\unc.exe" Id="UncBs" TargetDeviceFamily="PC" />
    <Executable Name="//corp-nas02/private_share/BobSecret/unc2.exe" Id="UncFs" TargetDeviceFamily="PC" />
    <Executable Name="/home/bob_dev/secret_project/linux.exe" Id="PosixHome" TargetDeviceFamily="Scarlett" />
    <Executable Name="~/private_keys/tilde.exe" Id="TildePath" TargetDeviceFamily="Scarlett" />
  </ExecutableList>
  <ShellVisuals DefaultDisplayName="Build from /Users/carol_dev/Desktop/pkg for MockDevUser99"
                SplashScreenImage="file:///C:/Users/AliceSecret/splash.png"
                StoreLogo="D:/XboxDumps/SecretBuild/store.png" />
  <VirtualMachine HostHint="\\\\corp-nas01\\share\\vm.vhdx">
    <ScratchDir>/home/bob_dev/vm_scratch</ScratchDir>
  </VirtualMachine>
  <ModFolder Name="C:\\Users\\AliceSecret\\Mods" ReadOnly="true" />
</Game>
"""
        with tempfile.TemporaryDirectory() as tmp:
            game_dir = Path(tmp) / "HostSecretDumpDir_99"
            (game_dir / "MockDevUser99").mkdir(parents=True)
            (game_dir / "Data").mkdir(parents=True)
            (game_dir / "MicrosoftGame.Config").write_text(hostile_xml, encoding="utf-8")
            (game_dir / "MockDevUser99" / "helper.dll").write_bytes(PE_STUB)
            (game_dir / "Data" / "SecretCopyrightedArchiveName.rpf").write_bytes(RPF9_STUB)

            fake_env = dict(os.environ)
            fake_env["USERNAME"] = "MockDevUser99"
            fake_env["USER"] = "mock_unix_user99"

            buf = io.StringIO()
            with mock.patch.dict(os.environ, fake_env, clear=True), redirect_stdout(buf):
                rc = iga.main(["--dir", str(game_dir), "--json"])

            self.assertEqual(rc, 0)
            raw_json = buf.getvalue()
            payload = json.loads(raw_json)
            self.assertTrue(payload["ok"])

            for forbidden in (
                str(game_dir),
                "HostSecretDumpDir_99",
                "C:\\Users",
                "C:/Users",
                "D:/XboxDumps",
                "corp-nas01",
                "corp-nas02",
                "private_share",
                "/home/bob_dev",
                "/Users/carol_dev",
                "file:///",
                "~/private_keys",
                "AliceSecret",
                "BobSecret",
                "bob_dev",
                "carol_dev",
                "MockDevUser99",
                "mock_unix_user99",
                "My Games",
                "SecretCopyrightedArchiveName",
            ):
                self.assertNotIn(forbidden, raw_json, msg=f"Leaked forbidden token: {forbidden}")

            # Unsafe paths are all rejected cleanly
            for exe in payload["executables"]:
                self.assertEqual(exe["status"], "unsafe_path")
                self.assertEqual(exe["declaredPath"], "<path>")
                self.assertFalse(exe["present"])
            self.assertEqual(payload["config"]["modFolder"], {"name": "<path>", "readOnly": True})

            # Verify recursive sanitize_report handles bytes and Path values cleanly
            sanitized_misc = iga.sanitize_report(
                {
                    "raw": b"\x01\x02\x03",
                    "p": Path("/home/bob_dev/secret.bin"),
                    "msg": "path=\\\\server\\share\\file and /home/bob_dev/x and C:\\Users\\alice\\y",
                }
            )
            self.assertEqual(sanitized_misc["raw"], "<bytes:3>")
            self.assertEqual(sanitized_misc["p"], "secret.bin")
            self.assertNotIn("bob_dev", sanitized_misc["msg"])
            self.assertNotIn("server", sanitized_misc["msg"])
            self.assertNotIn("C:\\Users", sanitized_misc["msg"])

    def test_backwards_compatibility_for_sanitize_text_and_legacy_text_format(self) -> None:
        self.assertIsNone(iga.sanitize_text(None))
        # Relative package paths containing words like "home" or "tmp" are preserved intact.
        self.assertEqual(iga.sanitize_text("Assets/home/logo.png"), "Assets/home/logo.png")
        self.assertEqual(iga.sanitize_text("Tools/tmp/helper.dll"), "Tools/tmp/helper.dll")

        # Legacy report dict without T88 additive keys still renders cleanly via format_report_text.
        legacy_report = {
            "schemaVersion": 1,
            "ok": True,
            "status": "ok",
            "configFound": True,
            "configValid": True,
            "configRelativePath": "MicrosoftGame.Config",
            "error": None,
            "config": {
                "configVersion": "1",
                "identity": {"name": "Legacy.Title", "publisher": "CN=Legacy", "version": "1.0"},
                "titleId": "AABBCCDD",
                "storeId": None,
                "msaAppId": None,
                "executables": [],
                "shellVisuals": {"defaultDisplayName": "Legacy Display"},
                "resources": {"languages": ["en-US"]},
                "virtualMachine": None,
                "modFolder": None,
                "desktopRegistration": None,
            },
            "executables": [],
            "shellVisuals": [],
            "inventory": {
                "dlls": [],
                "dllCount": 0,
                "dllCompleteCount": 0,
                "dllIncompleteCount": 0,
                "rpfCount": 0,
                "rpfCompleteCount": 0,
                "rpfIncompleteCount": 0,
                "rpfTotalBytes": 0,
            },
            "readiness": {
                "allDeclaredAssetsComplete": True,
                "hasPartialDownloads": False,
                "declaredExecutableCount": 0,
                "presentExecutableCount": 0,
                "completeExecutableCount": 0,
                "missingExecutableCount": 0,
                "incompleteExecutableCount": 0,
                "declaredVisualCount": 0,
                "presentVisualCount": 0,
                "completeVisualCount": 0,
                "missingVisualCount": 0,
                "incompleteVisualCount": 0,
            },
        }
        rendered = iga.format_report_text(legacy_report)
        self.assertIn("Legacy Display", rendered)
        self.assertIn("AABBCCDD", rendered)
        self.assertIn("RPFs : 0 total", rendered)


if __name__ == "__main__":
    unittest.main()
