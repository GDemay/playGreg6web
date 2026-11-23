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


if __name__ == "__main__":
    unittest.main()
