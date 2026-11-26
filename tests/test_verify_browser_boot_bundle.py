#!/usr/bin/env python3
"""Unit and CLI tests for tools/verify_browser_boot_bundle.py (Track 5 / W75)."""

from __future__ import annotations

import io
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_wasm_exports as wasm  # noqa: E402
import inspect_wgsl_pack as wgpk  # noqa: E402
import verify_browser_boot_bundle as vbb  # noqa: E402


def _make_minimal_side_wasm(extra_payload: bytes = b"") -> bytes:
    """Build a valid minimal Wasm v1 binary (optionally padded via custom section)."""
    if not extra_payload:
        return wasm.build_wasm_module()
    return wasm.build_wasm_module(custom_sections=[("pad", extra_payload)])


def _make_pgim_v1_header_bytes(
    *,
    version: int = 1,
    section_count: int = 0,
    total_size: int = 64,
) -> bytes:
    """Build a valid 64-byte PGIM v1 header matching recomp/src/image.rs."""
    buf = bytearray(64)
    buf[0:4] = b"PGIM"
    struct.pack_into(
        "<IIIIIQIIIIIIQ",
        buf,
        4,
        version,
        64,  # header_bytes
        64,  # section_entry_bytes
        0,  # flags
        4096,  # payload_alignment
        0x140000000,  # image_base
        0x10000,  # size_of_image
        0x1000,  # section_alignment
        0x1000,  # entry_point_rva
        section_count,
        0,  # dir64_count
        0,  # highlow_count
        total_size,
    )
    return bytes(buf)


def _write_valid_web_dist(
    web_dist: Path,
    *,
    side_modules: dict[str, bytes] | None = None,
    runtime_wasm_bytes: bytes | None = None,
    runtime_js_text: str = "export default function createRuntimeModule() {}\n",
) -> None:
    web_dist.mkdir(parents=True, exist_ok=True)
    (web_dist / "runtime.js").write_text(runtime_js_text, encoding="utf-8")
    (web_dist / "runtime.wasm").write_bytes(
        runtime_wasm_bytes
        if runtime_wasm_bytes is not None
        else wasm.build_valid_runtime_wasm()
    )
    if side_modules:
        for name, data in side_modules.items():
            (web_dist / name).write_bytes(data)


def _write_valid_boot_dir(boot_dir: Path, *, use_boot_manifest: bool = False) -> None:
    boot_dir.mkdir(parents=True, exist_ok=True)
    if use_boot_manifest:
        (boot_dir / "boot-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "boot-manifest",
                    "schema_version": 1,
                    "module": "main",
                    "pgimPath": "image/guest.pgim",
                }
            ),
            encoding="utf-8",
        )
        (boot_dir / "image").mkdir(parents=True, exist_ok=True)
        (boot_dir / "image" / "guest.pgim").write_bytes(_make_pgim_v1_header_bytes())
    else:
        (boot_dir / "program-manifest.json").write_text(
            json.dumps(
                {
                    "schema": "playgta6-recomp-program",
                    "schema_version": 1,
                    "modules": [
                        {"module": "main", "role": "exe"},
                        {"module": "helper", "pgimPath": "helper/image/helper.pgim"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        (boot_dir / "main" / "image").mkdir(parents=True, exist_ok=True)
        (boot_dir / "main" / "image" / "guest.pgim").write_bytes(_make_pgim_v1_header_bytes())
        (boot_dir / "helper" / "image").mkdir(parents=True, exist_ok=True)
        (boot_dir / "helper" / "image" / "helper.pgim").write_bytes(b"PGIM" + b"\x00" * 28)


def _write_valid_shaders_dir(shaders_dir: Path) -> None:
    shaders_dir.mkdir(parents=True, exist_ok=True)
    pack_bytes = wgpk.build_wgpk(
        [
            (0x1111, wgpk.ShaderStage.VERTEX, "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"),
            (0x2222, wgpk.ShaderStage.FRAGMENT, "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"),
            (0x3333, wgpk.ShaderStage.COMPUTE, "@compute @workgroup_size(1) fn cs() {}"),
        ]
    )
    (shaders_dir / "fallback.wgsl.pack").write_bytes(pack_bytes)


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


class TestWasmAndSplitBundleVerification(unittest.TestCase):
    def test_monolithic_bundle_passes_with_boot_and_shaders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            boot_dir = base / "boot"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            _write_valid_boot_dir(boot_dir)
            _write_valid_shaders_dir(shaders_dir)

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist,
                boot_dir=boot_dir,
                shaders_dir=shaders_dir,
                strict=True,
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["status"], "pass")
            self.assertEqual(report["wasm"]["bundle_mode"], "monolithic")
            self.assertTrue(report["wasm"]["runtime_js"]["ok"])
            self.assertTrue(report["wasm"]["runtime_wasm"]["ok"])
            self.assertTrue(report["wasm"]["runtime_wasm"]["v8_limit_ok"])
            self.assertTrue(report["wasm"]["runtime_wasm"]["is_memory64"])
            self.assertTrue(report["wasm"]["runtime_wasm"]["is_shared_memory"])
            self.assertEqual(report["wasm"]["runtime_wasm"]["host_imports_valid"], 5)
            self.assertEqual(report["wasm"]["runtime_wasm"]["host_imports_total"], 5)
            self.assertEqual(report["wasm"]["side_module_count"], 0)
            self.assertEqual(
                report["wasm"]["total_wasm_bytes"],
                report["wasm"]["runtime_wasm_bytes"],
            )
            self.assertTrue(report["wasm"]["all_v8_limits_ok"])

            self.assertTrue(report["boot"]["checked"])
            self.assertEqual(report["boot"]["status"], "pass")
            self.assertEqual(report["boot"]["manifest_name"], "program-manifest.json")
            self.assertEqual(report["boot"]["pgim_count"], 2)
            self.assertEqual(report["boot"]["valid_pgim_count"], 2)

            self.assertTrue(report["shaders"]["checked"])
            self.assertEqual(report["shaders"]["status"], "pass")
            self.assertEqual(report["shaders"]["pack_count"], 1)
            self.assertEqual(report["shaders"]["valid_pack_count"], 1)
            self.assertEqual(report["shaders"]["total_shader_entries"], 3)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["warnings"], [])

    def test_four_way_split_bundle_natural_sort_and_size_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            sides = {
                "guest_part_10.wasm": _make_minimal_side_wasm(b"A" * 40),
                "guest_part_2.wasm": _make_minimal_side_wasm(b"B" * 20),
                "guest_part_1.wasm": _make_minimal_side_wasm(b"C" * 10),
                "guest_part_3.wasm": _make_minimal_side_wasm(b"D" * 30),
            }
            _write_valid_web_dist(web_dist, side_modules=sides)

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["bundle_mode"], "split")
            self.assertEqual(report["wasm"]["side_module_count"], 4)
            names = [m["name"] for m in report["wasm"]["side_modules"]]
            self.assertEqual(
                names,
                [
                    "guest_part_1.wasm",
                    "guest_part_2.wasm",
                    "guest_part_3.wasm",
                    "guest_part_10.wasm",
                ],
            )
            self.assertTrue(all(m["ok"] and m["magic_ok"] and m["v8_limit_ok"] for m in report["wasm"]["side_modules"]))
            expected_side_total = sum(len(data) for data in sides.values())
            self.assertEqual(report["wasm"]["side_modules_total_bytes"], expected_side_total)
            self.assertEqual(
                report["wasm"]["total_wasm_bytes"],
                report["wasm"]["runtime_wasm_bytes"] + expected_side_total,
            )
            self.assertTrue(report["wasm"]["all_v8_limits_ok"])

    def test_oversized_runtime_wasm_over_1gib_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(web_dist)

            orig_get_size = vbb.get_file_size

            def fake_get_size(path: Path) -> int:
                if path.name == "runtime.wasm":
                    return vbb.V8_MAX_WASM_MODULE_BYTES + 1
                return orig_get_size(path)

            with mock.patch.object(vbb, "get_file_size", side_effect=fake_get_size):
                report = vbb.verify_browser_boot_bundle(web_dist=web_dist)

            self.assertFalse(report["ok"])
            self.assertFalse(report["wasm"]["runtime_wasm"]["v8_limit_ok"])
            self.assertFalse(report["wasm"]["runtime_wasm"]["ok"])
            self.assertFalse(report["wasm"]["all_v8_limits_ok"])
            self.assertEqual(
                report["wasm"]["runtime_wasm"]["size_bytes"],
                vbb.V8_MAX_WASM_MODULE_BYTES + 1,
            )
            self.assertTrue(any("1 GiB" in e for e in report["errors"]))

    def test_oversized_side_module_over_1gib_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(
                web_dist,
                side_modules={
                    "guest_part_1.wasm": _make_minimal_side_wasm(),
                    "guest_part_2.wasm": _make_minimal_side_wasm(),
                },
            )

            orig_get_size = vbb.get_file_size

            def fake_get_size(path: Path) -> int:
                if path.name == "guest_part_2.wasm":
                    return vbb.V8_MAX_WASM_MODULE_BYTES + 4096
                return orig_get_size(path)

            with mock.patch.object(vbb, "get_file_size", side_effect=fake_get_size):
                report = vbb.verify_browser_boot_bundle(web_dist=web_dist)

            self.assertFalse(report["ok"])
            self.assertTrue(report["wasm"]["runtime_wasm"]["v8_limit_ok"])
            by_name = {m["name"]: m for m in report["wasm"]["side_modules"]}
            self.assertTrue(by_name["guest_part_1.wasm"]["v8_limit_ok"])
            self.assertFalse(by_name["guest_part_2.wasm"]["v8_limit_ok"])
            self.assertFalse(by_name["guest_part_2.wasm"]["ok"])
            self.assertFalse(report["wasm"]["all_v8_limits_ok"])
            self.assertTrue(any("guest_part_2.wasm" in e and "1 GiB" in e for e in report["errors"]))

    def test_exact_1gib_boundary_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(
                web_dist,
                side_modules={"guest_part_1.wasm": _make_minimal_side_wasm()},
            )

            orig_get_size = vbb.get_file_size

            def fake_get_size(path: Path) -> int:
                if path.name == "guest_part_1.wasm":
                    return vbb.V8_MAX_WASM_MODULE_BYTES
                return orig_get_size(path)

            with mock.patch.object(vbb, "get_file_size", side_effect=fake_get_size):
                report = vbb.verify_browser_boot_bundle(web_dist=web_dist)

            self.assertTrue(report["ok"])
            self.assertTrue(report["wasm"]["all_v8_limits_ok"])

    def test_corrupt_side_module_magic_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(
                web_dist,
                side_modules={"guest_part_1.wasm": b"NOPE\x01\x00\x00\x00"},
            )
            report = vbb.verify_browser_boot_bundle(web_dist=web_dist)
            self.assertFalse(report["ok"])
            self.assertFalse(report["wasm"]["side_modules"][0]["magic_ok"])
            self.assertTrue(any("invalid Wasm magic" in e for e in report["errors"]))

    def test_runtime_wasm_abi_failure_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            bad_wasm = wasm.build_valid_runtime_wasm(
                is_memory64=False,
                omit_host_imports=("host_wgpu_submit",),
            )
            _write_valid_web_dist(web_dist, runtime_wasm_bytes=bad_wasm)

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist)
            self.assertFalse(report["ok"])
            self.assertFalse(report["wasm"]["runtime_wasm"]["is_memory64"])
            self.assertFalse(report["wasm"]["runtime_wasm"]["all_host_imports_ok"])
            self.assertIn("host_wgpu_submit", report["wasm"]["runtime_wasm"]["missing_host_imports"])

    def test_missing_runtime_js_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(web_dist)
            (web_dist / "runtime.js").unlink()

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist)
            self.assertFalse(report["ok"])
            self.assertFalse(report["wasm"]["runtime_js"]["present"])
            self.assertTrue(any("runtime.js missing" in e for e in report["errors"]))


class TestBootDirAndPgimVerification(unittest.TestCase):
    def test_boot_manifest_fallback_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            boot_dir = base / "boot"
            _write_valid_web_dist(web_dist)
            _write_valid_boot_dir(boot_dir, use_boot_manifest=True)

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, boot_dir=boot_dir)
            self.assertTrue(report["ok"])
            self.assertEqual(report["boot"]["manifest_name"], "boot-manifest.json")
            self.assertEqual(report["boot"]["manifest_kind"], "boot")
            self.assertEqual(report["boot"]["valid_pgim_count"], 1)

    def test_missing_pgim_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            boot_dir = base / "boot"
            _write_valid_web_dist(web_dist)
            _write_valid_boot_dir(boot_dir)
            (boot_dir / "helper" / "image" / "helper.pgim").unlink()

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, boot_dir=boot_dir)
            self.assertFalse(report["ok"])
            self.assertEqual(report["boot"]["status"], "fail")
            self.assertEqual(report["boot"]["valid_pgim_count"], 1)
            self.assertTrue(any("helper/image/helper.pgim" in e for e in report["errors"]))

    def test_corrupt_pgim_magic_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            boot_dir = base / "boot"
            _write_valid_web_dist(web_dist)
            _write_valid_boot_dir(boot_dir)
            (boot_dir / "main" / "image" / "guest.pgim").write_bytes(b"BAD!" + b"\x00" * 60)

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, boot_dir=boot_dir)
            self.assertFalse(report["ok"])
            by_rel = {p["rel_path"]: p for p in report["boot"]["pgims"]}
            self.assertFalse(by_rel["main/image/guest.pgim"]["magic_ok"])
            self.assertTrue(any("invalid magic" in e for e in report["errors"]))

    def test_manifest_path_traversal_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            boot_dir = base / "boot"
            _write_valid_web_dist(web_dist)
            boot_dir.mkdir(parents=True)
            (base / "outside.pgim").write_bytes(b"PGIM" + b"\x00" * 60)
            (boot_dir / "program-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "playgta6-recomp-program",
                        "schema_version": 1,
                        "modules": [{"module": "main", "pgimPath": "../outside.pgim"}],
                    }
                ),
                encoding="utf-8",
            )

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, boot_dir=boot_dir)
            self.assertFalse(report["ok"])
            self.assertTrue(any("traversal" in e for e in report["errors"]))


class TestShaderPackVerification(unittest.TestCase):
    def test_corrupt_wgpk_magic_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir()
            (shaders_dir / "broken.wgsl.pack").write_bytes(b"NOPE" + b"\x00" * 32)

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, shaders_dir=shaders_dir)
            self.assertFalse(report["ok"])
            self.assertEqual(report["shaders"]["status"], "fail")
            self.assertFalse(report["shaders"]["packs"][0]["magic_ok"])
            self.assertTrue(any("WGPK" in e for e in report["errors"]))

    def test_truncated_wgpk_entry_count_fails_safely(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir()
            # Claims 1,000,000 entries in a 16-byte file; must fail safely without huge allocation.
            bogus_header = struct.pack("<4sIII", b"WGPK", 1, 1_000_000, 0)
            (shaders_dir / "truncated.wgsl.pack").write_bytes(bogus_header)

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, shaders_dir=shaders_dir)
            self.assertFalse(report["ok"])
            self.assertTrue(report["shaders"]["packs"][0]["magic_ok"])
            self.assertEqual(report["shaders"]["packs"][0]["entry_count"], 1_000_000)
            self.assertFalse(report["shaders"]["packs"][0]["ok"])
            self.assertTrue(any("truncated entry table" in e for e in report["errors"]))

    def test_empty_wgpk_warns_and_fails_in_strict_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir()
            (shaders_dir / "empty.wgsl.pack").write_bytes(wgpk.build_wgpk([]))

            non_strict = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=False
            )
            self.assertTrue(non_strict["ok"])
            self.assertEqual(non_strict["shaders"]["total_shader_entries"], 0)
            self.assertTrue(any("0 shader entries" in w for w in non_strict["warnings"]))

            strict_report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=True
            )
            self.assertFalse(strict_report["ok"])


class TestEscapingSymlinkRejection(unittest.TestCase):
    def test_escaping_side_module_symlink_skipped_via_mock_all_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            _write_valid_web_dist(
                web_dist,
                side_modules={
                    "guest_part_1.wasm": _make_minimal_side_wasm(),
                    "escape_part.wasm": _make_minimal_side_wasm(),
                },
            )
            outside_target = base / "outside_secret.wasm"
            outside_target.write_bytes(_make_minimal_side_wasm())

            orig_is_symlink = Path.is_symlink
            orig_resolve = Path.resolve

            def fake_is_symlink(self_path: Path) -> bool:
                if self_path.name == "escape_part.wasm":
                    return True
                return orig_is_symlink(self_path)

            def fake_resolve(self_path: Path, strict: bool = False) -> Path:
                if self_path.name == "escape_part.wasm":
                    return orig_resolve(outside_target, strict=strict)
                return orig_resolve(self_path, strict=strict)

            with (
                mock.patch.object(Path, "is_symlink", fake_is_symlink),
                mock.patch.object(Path, "resolve", fake_resolve),
            ):
                report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=False)
                strict_report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)

            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["skipped_escaping_symlinks"], 1)
            self.assertEqual([m["name"] for m in report["wasm"]["side_modules"]], ["guest_part_1.wasm"])
            self.assertTrue(any("escaping/broken" in w for w in report["warnings"]))
            self.assertFalse(strict_report["ok"])

    def test_escaping_runtime_wasm_and_pgim_symlinks_rejected_via_mock_all_hosts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            boot_dir = base / "boot"
            _write_valid_web_dist(web_dist)
            _write_valid_boot_dir(boot_dir, use_boot_manifest=True)
            outside_wasm = base / "outside_runtime.wasm"
            outside_wasm.write_bytes(wasm.build_valid_runtime_wasm())
            outside_pgim = base / "outside.pgim"
            outside_pgim.write_bytes(_make_pgim_v1_header_bytes())

            orig_is_symlink = Path.is_symlink
            orig_resolve = Path.resolve

            def fake_is_symlink(self_path: Path) -> bool:
                if self_path.name in ("runtime.wasm", "guest.pgim"):
                    return True
                return orig_is_symlink(self_path)

            def fake_resolve(self_path: Path, strict: bool = False) -> Path:
                if self_path.name == "runtime.wasm":
                    return orig_resolve(outside_wasm, strict=strict)
                if self_path.name == "guest.pgim":
                    return orig_resolve(outside_pgim, strict=strict)
                return orig_resolve(self_path, strict=strict)

            with (
                mock.patch.object(Path, "is_symlink", fake_is_symlink),
                mock.patch.object(Path, "resolve", fake_resolve),
            ):
                report = vbb.verify_browser_boot_bundle(web_dist=web_dist, boot_dir=boot_dir)

            self.assertFalse(report["ok"])
            self.assertTrue(any("runtime.wasm" in e and "symlink" in e for e in report["errors"]))
            self.assertTrue(any("escapes boot-dir via symlink" in e for e in report["errors"]))

    def test_real_os_escaping_symlink_when_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            if not _can_create_symlinks(base):
                self.skipTest("symlinks require elevated privileges on this host")
            web_dist = base / "dist"
            _write_valid_web_dist(web_dist)
            outside = base / "outside.wasm"
            outside.write_bytes(_make_minimal_side_wasm())
            os.symlink(outside, web_dist / "guest_part_9.wasm")

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertFalse(report["ok"])
            self.assertEqual(report["wasm"]["skipped_escaping_symlinks"], 1)
            self.assertEqual(report["wasm"]["side_module_count"], 0)


class TestStrictSanitizationAndCli(unittest.TestCase):
    def test_sanitize_text_strips_windows_unc_posix_and_username(self) -> None:
        with mock.patch.dict(os.environ, {"USERNAME": "SecretPlayer99", "USER": "SecretPlayer99"}):
            raw = (
                r"Failed reading C:\Users\SecretPlayer99\Games\runtime.wasm and "
                r"\\fileserver\share\SecretPlayer99\dump.pgim and "
                r"//nas01/private/SecretPlayer99/pack.wgsl.pack and "
                r"/home/SecretPlayer99/dev/test and /Users/SecretPlayer99/Desktop/a "
                r"owned by SecretPlayer99 on D:/work/file.bin"
            )
            cleaned = vbb.sanitize_text(raw)

        self.assertNotIn("SecretPlayer99", cleaned)
        self.assertNotIn("C:", cleaned)
        self.assertNotIn("D:", cleaned)
        self.assertNotIn("fileserver", cleaned)
        self.assertNotIn("nas01", cleaned)
        self.assertNotIn("/home/", cleaned)
        self.assertNotIn("/Users/", cleaned)
        self.assertIn("<path>", cleaned)
        self.assertIn("<user>", cleaned)

    def test_cli_json_and_text_outputs_contain_no_host_paths_or_usernames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            boot_dir = base / "boot"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(
                web_dist,
                side_modules={"guest_part_1.wasm": _make_minimal_side_wasm()},
            )
            _write_valid_boot_dir(boot_dir)
            _write_valid_shaders_dir(shaders_dir)

            buf_json = io.StringIO()
            with redirect_stdout(buf_json):
                rc_json = vbb.main(
                    [
                        "--web-dist",
                        str(web_dist),
                        "--boot-dir",
                        str(boot_dir),
                        "--shaders-dir",
                        str(shaders_dir),
                        "--json",
                        "--strict",
                    ]
                )
            self.assertEqual(rc_json, 0)
            json_text = buf_json.getvalue()
            parsed = json.loads(json_text)
            self.assertTrue(parsed["ok"])

            buf_text = io.StringIO()
            with redirect_stdout(buf_text):
                rc_text = vbb.main(
                    [
                        "--web-dist",
                        str(web_dist),
                        "--boot-dir",
                        str(boot_dir),
                        "--shaders-dir",
                        str(shaders_dir),
                    ]
                )
            self.assertEqual(rc_text, 0)
            human_text = buf_text.getvalue()

            active_users = vbb._current_usernames()
            for output in (json_text, human_text):
                self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", output), output)
                self.assertNotIn("\\\\", output)
                self.assertNotIn("/home/", output)
                self.assertNotIn("/Users/", output)
                self.assertNotIn(str(base), output)
                for user in active_users:
                    self.assertNotRegex(
                        output,
                        re.compile(rf"(?<![A-Za-z0-9_]){re.escape(user)}(?![A-Za-z0-9_])", re.IGNORECASE),
                    )

    def test_cli_returns_exit_code_1_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "does_not_exist"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = vbb.main(["--web-dist", str(missing_dir), "--json"])
            self.assertEqual(rc, 1)
            data = json.loads(buf.getvalue())
            self.assertFalse(data["ok"])
            self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", buf.getvalue()))


class TestReflectSidecarInlineSamplerValidation(unittest.TestCase):
    def test_valid_inline_sampler_sidecar_passes_strict_and_counts_stages_and_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            _write_valid_shaders_dir(shaders_dir)

            sidecar = {
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {
                    "0x00000000000000d4": {
                        "stage": "fragment",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "uniform",
                                "source": {"user_sgpr": 8, "table_byte_offset": 0},
                            },
                            {
                                "group": 0,
                                "binding": 34,
                                "kind": "sampler",
                                "source": {
                                    "inline_sampler": {
                                        "sdwords": [0x92, 0x00F00000, 0x08500000, 0],
                                    }
                                },
                            },
                            {
                                "group": 0,
                                "binding": 35,
                                "kind": "sampler",
                                "source": {
                                    "inline_sampler": {
                                        "sdwords": [0, 1, 2, 0xFFFFFFFF],
                                    }
                                },
                            },
                        ],
                    },
                    "0x00000000000000e5": {
                        "stage": "compute",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 2,
                                "kind": "sampler",
                                "source": {
                                    "inline_sampler": {
                                        "sdwords": [12288, 15728640, 67108864, 0],
                                    }
                                },
                            },
                        ],
                    },
                    "0x00000000000000f6": {
                        "stage": "vertex",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "sampler",
                                "source": {"static_sampler": 0},
                            },
                        ],
                    },
                },
            }
            (shaders_dir / "fallback.wgsl.pack.reflect.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist,
                shaders_dir=shaders_dir,
                strict=True,
            )

            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["warnings"], [])
            shaders = report["shaders"]
            self.assertEqual(shaders["reflect_sidecar_count"], 1)
            self.assertEqual(shaders["valid_reflect_sidecar_count"], 1)
            self.assertEqual(shaders["reflect_stage_count"], 3)
            self.assertEqual(shaders["inline_sampler_stages"], 2)
            self.assertEqual(shaders["inline_sampler_bindings"], 3)
            self.assertEqual(shaders["malformed_inline_samplers"], 0)
            self.assertEqual(len(shaders["reflect_sidecars"]), 1)
            self.assertTrue(shaders["reflect_sidecars"][0]["ok"])

            text = vbb.format_report_text(report)
            self.assertIn("inline_sampler_stages=2", text)
            self.assertIn("inline_sampler_bindings=3", text)
            self.assertIn("malformed_inline_samplers=0", text)

    def test_malformed_inline_sampler_sdwords_fails_strict_validation(self) -> None:
        bad_sources = [
            {"inline_sampler": {"sdwords": [1, 2, 3]}},  # too short
            {"inline_sampler": {"sdwords": [1, 2, 3, 4, 5]}},  # too long
            {"inline_sampler": {"sdwords": [1, 2, 3, -1]}},  # negative
            {"inline_sampler": {"sdwords": [1, 2, 3, 0x100000000]}},  # > uint32 max
            {"inline_sampler": {"sdwords": [1, 2, 3, "4"]}},  # string
            {"inline_sampler": {"sdwords": [1, 2, 3, True]}},  # bool
            {"inline_sampler": {"sdwords": [1, 2, 3, 4.5]}},  # float
            {"inline_sampler": {}},  # missing sdwords
            {"inline_sampler": [1, 2, 3, 4]},  # non-object inline_sampler
        ]
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            _write_valid_shaders_dir(shaders_dir)

            bindings: list[dict[str, object]] = [
                {
                    "group": 0,
                    "binding": 0,
                    "kind": "sampler",
                    "source": {"inline_sampler": {"sdwords": [146, 15728640, 139460608, 0]}},
                }
            ]
            for idx, src in enumerate(bad_sources, start=1):
                bindings.append(
                    {"group": 0, "binding": idx, "kind": "sampler", "source": src}
                )

            sidecar = {
                "schema": "playgta6-stage-reflection",
                "version": 1,
                "stages": {
                    "0x000000000000bad1": {
                        "stage": "fragment",
                        "bindings": bindings,
                    }
                },
            }
            (shaders_dir / "fallback.wgsl.pack.reflect.json").write_text(
                json.dumps(sidecar), encoding="utf-8"
            )

            non_strict = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=False
            )
            self.assertTrue(non_strict["ok"])
            self.assertEqual(non_strict["shaders"]["inline_sampler_stages"], 1)
            self.assertEqual(non_strict["shaders"]["inline_sampler_bindings"], 1)
            self.assertEqual(
                non_strict["shaders"]["malformed_inline_samplers"], len(bad_sources)
            )
            self.assertEqual(non_strict["shaders"]["valid_reflect_sidecar_count"], 0)
            self.assertFalse(non_strict["shaders"]["reflect_sidecars"][0]["ok"])
            self.assertEqual(len(non_strict["warnings"]), len(bad_sources))

            strict_report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=True
            )
            self.assertFalse(strict_report["ok"])
            self.assertEqual(
                strict_report["shaders"]["malformed_inline_samplers"], len(bad_sources)
            )

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = vbb.main(
                    [
                        "--web-dist",
                        str(web_dist),
                        "--shaders-dir",
                        str(shaders_dir),
                        "--json",
                        "--strict",
                    ]
                )
            self.assertEqual(rc, 1)
            cli_payload = json.loads(buf.getvalue())
            self.assertFalse(cli_payload["ok"])
            self.assertEqual(
                cli_payload["shaders"]["malformed_inline_samplers"], len(bad_sources)
            )

    def test_malformed_reflect_sidecar_json_and_escaping_symlink_fail_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            _write_valid_shaders_dir(shaders_dir)
            (shaders_dir / "broken.wgsl.pack.reflect.json").write_text(
                "{not valid json", encoding="utf-8"
            )

            strict_report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=True
            )
            self.assertFalse(strict_report["ok"])
            self.assertEqual(strict_report["shaders"]["reflect_sidecar_count"], 1)
            self.assertEqual(strict_report["shaders"]["valid_reflect_sidecar_count"], 0)
            self.assertTrue(any("invalid JSON" in w for w in strict_report["warnings"]))


class TestGuestManifestAndBink2Readiness(unittest.TestCase):
    def test_guest_manifest_reports_bink2scarlett_present_both_spellings(self) -> None:
        for modules_list in (
            ["vcruntime140", "bink2scarlett"],
            ["VCRUNTIME140.dll", "Bink2Scarlett.dll"],
        ):
            with self.subTest(modules_list=modules_list):
                with tempfile.TemporaryDirectory() as tmp:
                    web_dist = Path(tmp) / "dist"
                    _write_valid_web_dist(web_dist)
                    (web_dist / "guest_manifest.json").write_text(
                        json.dumps(
                            {
                                "schemaVersion": 1,
                                "runtimeJs": "runtime.js",
                                "runtimeWasm": "runtime.wasm",
                                "indirectDispatch": "shard",
                                "guest_dll_modules": modules_list,
                                "sideModules": [],
                            }
                        ),
                        encoding="utf-8",
                    )

                    report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
                    self.assertTrue(report["ok"])
                    self.assertEqual(report["errors"], [])
                    self.assertEqual(report["warnings"], [])
                    gm = report["wasm"]["guest_manifest"]
                    self.assertTrue(gm["present"])
                    self.assertTrue(gm["ok"])
                    self.assertEqual(gm["guest_modules"], modules_list)
                    self.assertTrue(gm["bink2scarlett_present"])
                    self.assertTrue(report["wasm"]["bink2scarlett_present"])

                    text = vbb.format_report_text(report)
                    self.assertIn("guest_manifest.json: OK", text)
                    self.assertIn("bink2scarlett=YES", text)

    def test_guest_manifest_without_bink2scarlett_reports_false_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(web_dist)
            (web_dist / "guest_manifest.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "guest_dll_modules": ["vcruntime140", "msvcp140"],
                    }
                ),
                encoding="utf-8",
            )

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertTrue(report["ok"])
            gm = report["wasm"]["guest_manifest"]
            self.assertTrue(gm["present"])
            self.assertTrue(gm["ok"])
            self.assertEqual(gm["guest_modules"], ["vcruntime140", "msvcp140"])
            self.assertFalse(gm["bink2scarlett_present"])
            self.assertFalse(report["wasm"]["bink2scarlett_present"])

            text = vbb.format_report_text(report)
            self.assertIn("bink2scarlett=NO", text)

    def test_program_manifest_reports_bink2scarlett_in_boot_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            boot_dir = base / "boot"
            _write_valid_web_dist(web_dist)
            boot_dir.mkdir(parents=True)
            (boot_dir / "program-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "playgta6-recomp-program",
                        "schema_version": 1,
                        "modules": [
                            {"module": "main", "role": "exe"},
                            {
                                "module": "Bink2Scarlett.dll",
                                "pgimPath": "bink2scarlett/image/guest.pgim",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (boot_dir / "main" / "image").mkdir(parents=True)
            (boot_dir / "main" / "image" / "guest.pgim").write_bytes(_make_pgim_v1_header_bytes())
            (boot_dir / "bink2scarlett" / "image").mkdir(parents=True)
            (boot_dir / "bink2scarlett" / "image" / "guest.pgim").write_bytes(
                _make_pgim_v1_header_bytes()
            )

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, boot_dir=boot_dir, strict=True
            )
            self.assertTrue(report["ok"])
            self.assertTrue(report["boot"]["bink2scarlett_present"])
            self.assertEqual(report["boot"]["guest_modules"], ["main", "Bink2Scarlett.dll"])

    def test_malformed_guest_manifest_warns_and_fails_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(web_dist)
            (web_dist / "guest_manifest.json").write_text(
                json.dumps({"guest_dll_modules": "not-a-list"}),
                encoding="utf-8",
            )

            non_strict = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=False)
            self.assertTrue(non_strict["ok"])
            self.assertFalse(non_strict["wasm"]["guest_manifest"]["ok"])
            self.assertTrue(any("guest_dll_modules" in w for w in non_strict["warnings"]))

            strict_report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertFalse(strict_report["ok"])

    def test_sanitization_strips_host_paths_in_guest_manifest_and_sidecar(self) -> None:
        with mock.patch.dict(os.environ, {"USERNAME": "SecretPlayer99", "USER": "SecretPlayer99"}):
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                web_dist = base / "dist"
                shaders_dir = base / "shaders"
                _write_valid_web_dist(web_dist)
                _write_valid_shaders_dir(shaders_dir)

                (web_dist / "guest_manifest.json").write_text(
                    json.dumps(
                        {
                            "guest_dll_modules": [
                                r"C:\Users\SecretPlayer99\Games\Bink2Scarlett.dll"
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                (shaders_dir / "fallback.wgsl.pack.reflect.json").write_text(
                    json.dumps(
                        {
                            "schema": "playgta6-stage-reflection",
                            "version": 1,
                            "stages": {
                                r"C:\Users\SecretPlayer99\dumps\stage0": {
                                    "stage": "fragment",
                                    "bindings": [
                                        {
                                            "group": 0,
                                            "binding": 0,
                                            "kind": "sampler",
                                            "source": {"inline_sampler": {"sdwords": [1, 2, 3]}},
                                        }
                                    ],
                                }
                            },
                        }
                    ),
                    encoding="utf-8",
                )

                report = vbb.verify_browser_boot_bundle(
                    web_dist=web_dist, shaders_dir=shaders_dir, strict=False
                )
                json_text = json.dumps(report)
                human_text = vbb.format_report_text(report)

                self.assertTrue(report["wasm"]["bink2scarlett_present"])
                self.assertEqual(
                    report["wasm"]["guest_manifest"]["guest_modules"], ["Bink2Scarlett.dll"]
                )
                self.assertEqual(report["shaders"]["malformed_inline_samplers"], 1)
                for output in (json_text, human_text):
                    self.assertNotIn("SecretPlayer99", output)
                    self.assertIsNone(re.search(r"[A-Za-z]:[\\/]", output), output)


def _build_runtime_wasm_with_exports(
    extra_exports: list[tuple[str, str, int]],
) -> bytes:
    """Build a valid Wasm64 + shared memory + 5/5 host imports module with custom exports."""
    types: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    imports: list[dict[str, object]] = [
        {
            "module": "env",
            "name": "memory",
            "kind": "memory",
            "min_pages": 256,
            "max_pages": 65536,
            "is_shared": True,
            "is_memory64": True,
        }
    ]
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
    exports: list[tuple[str, str, int]] = [("memory", "memory", 0)]
    exports.extend(extra_exports)
    return wasm.build_wasm_module(
        types=types,
        imports=imports,
        exports=exports,
        target_features=["+memory64", "+atomics", "+bulk-memory"],
    )


class TestRuntimeWasmAudioRingAndInputExports(unittest.TestCase):
    def test_pure_python_export_section_parser_on_minimal_modules(self) -> None:
        # Empty valid 8-byte Wasm binary (no export section at all).
        bare_wasm = wasm.build_wasm_module()
        probe_bare = vbb.inspect_wasm_runtime_exports(bare_wasm)
        self.assertEqual(probe_bare["exports_count"], 0)
        self.assertFalse(probe_bare["has_main_export"])
        self.assertFalse(probe_bare["has_input_state_export"])
        self.assertFalse(probe_bare["has_audio_ring_export"])

        # Canonical un-underscored Wasm function exports (Emscripten output).
        canonical_wasm = wasm.build_wasm_module(
            exports=[
                ("memory", "memory", 0),
                ("main", "func", 0),
                ("playgta6_get_input_state_ptr", "func", 1),
                ("playgta6_audio_ring", "func", 2),
            ]
        )
        probe_canon = vbb.inspect_wasm_runtime_exports(canonical_wasm)
        self.assertEqual(probe_canon["exports_count"], 4)
        self.assertTrue(probe_canon["has_main_export"])
        self.assertTrue(probe_canon["has_input_state_export"])
        self.assertEqual(probe_canon["input_state_export_name"], "playgta6_get_input_state_ptr")
        self.assertTrue(probe_canon["has_audio_ring_export"])
        self.assertEqual(probe_canon["audio_ring_export_name"], "playgta6_audio_ring")

        # Leading-underscore variants also recognized.
        underscored_wasm = wasm.build_wasm_module(
            exports=[
                ("_main", "func", 0),
                ("_playgta6_get_input_state_ptr", "func", 1),
                ("_playgta6_audio_ring", "func", 2),
            ]
        )
        probe_under = vbb.inspect_wasm_runtime_exports(underscored_wasm)
        self.assertTrue(probe_under["has_main_export"])
        self.assertTrue(probe_under["has_input_state_export"])
        self.assertEqual(probe_under["input_state_export_name"], "_playgta6_get_input_state_ptr")
        self.assertTrue(probe_under["has_audio_ring_export"])
        self.assertEqual(probe_under["audio_ring_export_name"], "_playgta6_audio_ring")

        # Non-function exports with matching names must not satisfy function export checks.
        wrong_kind_wasm = wasm.build_wasm_module(
            exports=[
                ("playgta6_get_input_state_ptr", "global", 0),
                ("playgta6_audio_ring", "memory", 0),
            ]
        )
        probe_wrong = vbb.inspect_wasm_runtime_exports(wrong_kind_wasm)
        self.assertFalse(probe_wrong["has_input_state_export"])
        self.assertFalse(probe_wrong["has_audio_ring_export"])

        # Truncated export section raises ValueError cleanly.
        truncated = b"\x00asm\x01\x00\x00\x00\x07\x05\x01\x03abc"
        with self.assertRaises(ValueError):
            vbb.parse_wasm_export_section(truncated)

    def test_linked_runtime_wasm_with_audio_ring_and_input_exports_passes_strict(self) -> None:
        for input_name, audio_name in (
            ("playgta6_get_input_state_ptr", "playgta6_audio_ring"),
            ("_playgta6_get_input_state_ptr", "_playgta6_audio_ring"),
        ):
            with self.subTest(input_name=input_name, audio_name=audio_name):
                with tempfile.TemporaryDirectory() as tmp:
                    web_dist = Path(tmp) / "dist"
                    rt_bytes = _build_runtime_wasm_with_exports(
                        [
                            ("main", "func", 0),
                            (input_name, "func", 1),
                            (audio_name, "func", 2),
                        ]
                    )
                    _write_valid_web_dist(web_dist, runtime_wasm_bytes=rt_bytes)

                    report = vbb.verify_browser_boot_bundle(
                        web_dist=web_dist,
                        strict=True,
                        require_audio_ring=True,
                        require_input_state=True,
                    )
                    self.assertTrue(report["ok"])
                    self.assertEqual(report["errors"], [])
                    self.assertEqual(report["warnings"], [])
                    rt_wasm = report["wasm"]["runtime_wasm"]
                    self.assertTrue(rt_wasm["has_input_state_export"])
                    self.assertEqual(rt_wasm["input_state_export_name"], input_name)
                    self.assertTrue(rt_wasm["has_audio_ring_export"])
                    self.assertEqual(rt_wasm["audio_ring_export_name"], audio_name)

                    text = vbb.format_report_text(report)
                    self.assertIn("input_state=YES", text)
                    self.assertIn("audio_ring=YES", text)

    def test_linked_runtime_missing_audio_ring_warns_and_fails_strict_or_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            rt_bytes = _build_runtime_wasm_with_exports(
                [
                    ("main", "func", 0),
                    ("playgta6_get_input_state_ptr", "func", 1),
                ]
            )
            _write_valid_web_dist(web_dist, runtime_wasm_bytes=rt_bytes)

            non_strict = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=False)
            self.assertTrue(non_strict["ok"])
            self.assertTrue(non_strict["wasm"]["runtime_wasm"]["has_input_state_export"])
            self.assertFalse(non_strict["wasm"]["runtime_wasm"]["has_audio_ring_export"])
            self.assertTrue(any("playgta6_audio_ring" in w for w in non_strict["warnings"]))

            strict_report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertFalse(strict_report["ok"])

            req_report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, strict=False, require_audio_ring=True
            )
            self.assertFalse(req_report["ok"])
            self.assertFalse(req_report["wasm"]["runtime_wasm"]["ok"])
            self.assertTrue(any("playgta6_audio_ring" in e for e in req_report["errors"]))

    def test_smoke_shell_missing_exports_passes_default_and_fails_cli_require_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(web_dist)

            default_report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertTrue(default_report["ok"])
            self.assertFalse(default_report["wasm"]["runtime_wasm"]["has_input_state_export"])
            self.assertFalse(default_report["wasm"]["runtime_wasm"]["has_audio_ring_export"])

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = vbb.main(
                    [
                        "--web-dist",
                        str(web_dist),
                        "--require-audio-ring",
                        "--require-input-state",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 1)
            payload = json.loads(buf.getvalue())
            self.assertFalse(payload["ok"])
            self.assertTrue(any("playgta6_audio_ring" in e for e in payload["errors"]))
            self.assertTrue(
                any("playgta6_get_input_state_ptr" in e for e in payload["errors"])
            )


class TestWgpkEntryBoundsIntegrity(unittest.TestCase):
    def test_valid_splash_wgpk_pack_reports_clean_bounds_and_splash_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir(parents=True)

            pack_bytes = wgpk.build_wgpk(
                [
                    (0xAAA1, wgpk.ShaderStage.VERTEX, "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"),
                    (0xBBB2, wgpk.ShaderStage.FRAGMENT, "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"),
                ]
            )
            (shaders_dir / "splash.wgsl.pack").write_bytes(pack_bytes)

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=True
            )
            self.assertTrue(report["ok"])
            shaders = report["shaders"]
            self.assertTrue(shaders["all_pack_bounds_ok"])
            self.assertEqual(shaders["out_of_bounds_entries"], 0)
            self.assertTrue(shaders["splash_pack_present"])
            self.assertTrue(shaders["splash_pack_ok"])
            pack = shaders["packs"][0]
            self.assertTrue(pack["header_ok"])
            self.assertTrue(pack["table_ok"])
            self.assertTrue(pack["bounds_ok"])
            self.assertEqual(pack["data_start_offset"], 16 + 2 * 20)
            self.assertEqual(pack["max_payload_end"], len(pack_bytes))
            self.assertEqual(pack["out_of_bounds_entries"], 0)
            self.assertEqual(pack["invalid_stage_entries"], 0)

            text = vbb.format_report_text(report)
            self.assertIn("pack splash.wgsl.pack: OK", text)
            self.assertIn("bounds=OK", text)

    def test_wgpk_entry_payload_exceeding_file_size_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir(parents=True)

            valid = wgpk.build_wgpk(
                [(0x1234, wgpk.ShaderStage.FRAGMENT, "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }")]
            )
            # Truncate 8 bytes off the payload so offset + length > file_size.
            (shaders_dir / "splash.wgsl.pack").write_bytes(valid[:-8])

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir
            )
            self.assertFalse(report["ok"])
            shaders = report["shaders"]
            self.assertFalse(shaders["all_pack_bounds_ok"])
            self.assertEqual(shaders["out_of_bounds_entries"], 1)
            self.assertTrue(shaders["splash_pack_present"])
            self.assertFalse(shaders["splash_pack_ok"])
            pack = shaders["packs"][0]
            self.assertTrue(pack["header_ok"])
            self.assertTrue(pack["table_ok"])
            self.assertFalse(pack["bounds_ok"])
            self.assertEqual(pack["out_of_bounds_entries"], 1)
            self.assertTrue(any("exceeds pack bounds" in e for e in report["errors"]))

    def test_wgpk_entry_offset_overlapping_header_or_table_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir(parents=True)

            # 1 entry -> data_start = 16 + 20 = 36, but entry claims offset = 12 (< 36).
            header = struct.pack("<4sIII", b"WGPK", 1, 1, 0)
            entry = struct.pack("<QB3sII", 0xCAFE, 1, b"\x00\x00\x00", 12, 8)
            payload = b"// dummy wgsl payload"
            (shaders_dir / "splash.wgsl.pack").write_bytes(header + entry + payload)

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir
            )
            self.assertFalse(report["ok"])
            pack = report["shaders"]["packs"][0]
            self.assertTrue(pack["header_ok"])
            self.assertTrue(pack["table_ok"])
            self.assertFalse(pack["bounds_ok"])
            self.assertEqual(pack["out_of_bounds_entries"], 1)
            self.assertTrue(any("overlaps header/entry table" in e for e in report["errors"]))

    def test_wgpk_multiple_oob_entries_and_invalid_version_and_stage_caught(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir(parents=True)

            # Pack 1: 10 entries all out of bounds (including u32 overflow) + bad stage.
            count = 10
            header = struct.pack("<4sIII", b"WGPK", 1, count, 0)
            entries = bytearray()
            for i in range(count):
                stage = 9 if i == 0 else 1
                offset = 0xFFFFFFFC if i == 1 else (16 + count * 20 + 500 + i)
                length = 16
                entries.extend(struct.pack("<QB3sII", i + 1, stage, b"\x00\x00\x00", offset, length))
            (shaders_dir / "corrupt_many.wgsl.pack").write_bytes(header + bytes(entries) + b"abcd")

            # Pack 2: unsupported version 3.
            (shaders_dir / "v3.wgsl.pack").write_bytes(struct.pack("<4sIII", b"WGPK", 3, 0, 0))

            # Pack 3: truncated < 16 bytes header.
            (shaders_dir / "short.wgsl.pack").write_bytes(b"WGPK\x01\x00")

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir
            )
            self.assertFalse(report["ok"])
            by_name = {p["name"]: p for p in report["shaders"]["packs"]}

            corrupt = by_name["corrupt_many.wgsl.pack"]
            self.assertFalse(corrupt["ok"])
            self.assertFalse(corrupt["bounds_ok"])
            self.assertEqual(corrupt["out_of_bounds_entries"], 10)
            self.assertEqual(corrupt["invalid_stage_entries"], 1)
            self.assertTrue(any("additional invalid/out-of-bounds" in e for e in corrupt["errors"]))

            v3 = by_name["v3.wgsl.pack"]
            self.assertTrue(v3["magic_ok"])
            self.assertFalse(v3["header_ok"])
            self.assertFalse(v3["bounds_ok"])
            self.assertIsNone(v3["entry_bytes"])
            self.assertFalse(v3["is_v2"])
            self.assertTrue(any("unsupported WGPK version 3" in e for e in v3["errors"]))

            short = by_name["short.wgsl.pack"]
            self.assertFalse(short["magic_ok"])
            self.assertFalse(short["header_ok"])
            self.assertFalse(short["bounds_ok"])


class TestWgpkV2Support(unittest.TestCase):
    def test_valid_wgpk_v2_pack_passes_strict_and_reports_v2_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir(parents=True)

            pack_bytes = wgpk.build_wgpk(
                [
                    (0x1111, wgpk.ShaderStage.VERTEX, "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"),
                    (0x2222, wgpk.ShaderStage.FRAGMENT, "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"),
                    (0x3333, wgpk.ShaderStage.COMPUTE, "@compute @workgroup_size(1) fn cs() {}"),
                ],
                version=2,
            )
            (shaders_dir / "large_shaders.wgsl.pack").write_bytes(pack_bytes)

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=True
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["warnings"], [])
            pack = report["shaders"]["packs"][0]
            self.assertEqual(pack["version"], 2)
            self.assertEqual(pack["entry_bytes"], 24)
            self.assertTrue(pack["is_v2"])
            self.assertTrue(pack["header_ok"])
            self.assertTrue(pack["table_ok"])
            self.assertTrue(pack["bounds_ok"])
            self.assertEqual(pack["entry_count"], 3)
            self.assertEqual(pack["data_start_offset"], 16 + 3 * 24)
            self.assertEqual(pack["max_payload_end"], len(pack_bytes))

            text = vbb.format_report_text(report)
            self.assertIn("pack large_shaders.wgsl.pack: OK", text)
            self.assertIn("v2", text)

    def test_wgpk_v2_over_4gib_64bit_offsets_and_chunked_toc_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir(parents=True)

            # Build a real WGPK v2 header + 4100 TOC entries (exceeding 4096 chunk size)
            # with final entries placed beyond the 4 GiB (u32::MAX) boundary.
            entry_count = 4100
            data_start = 16 + entry_count * 24
            simulated_base_offset = 0x1_2000_0000  # 4.5 GiB
            payload_len = 512
            simulated_total_size = simulated_base_offset + entry_count * payload_len

            toc = bytearray(data_start)
            struct.pack_into("<4sIII", toc, 0, b"WGPK", 2, entry_count, 0)
            for i in range(entry_count):
                offset = simulated_base_offset + i * payload_len
                stage = (i % 3) + 1
                struct.pack_into(
                    "<QB3sIQ",
                    toc,
                    16 + i * 24,
                    0x1000000 + i,
                    stage,
                    b"\x00\x00\x00",
                    payload_len,
                    offset,
                )
            pack_path = shaders_dir / "huge_v2.wgsl.pack"
            pack_path.write_bytes(bytes(toc))

            orig_get_size = vbb.get_file_size

            def fake_get_size(path: Path) -> int:
                if path.name == "huge_v2.wgsl.pack":
                    return simulated_total_size
                return orig_get_size(path)

            with mock.patch.object(vbb, "get_file_size", side_effect=fake_get_size):
                report = vbb.verify_browser_boot_bundle(
                    web_dist=web_dist, shaders_dir=shaders_dir, strict=True
                )

            self.assertTrue(report["ok"])
            pack = report["shaders"]["packs"][0]
            self.assertEqual(pack["version"], 2)
            self.assertEqual(pack["entry_bytes"], 24)
            self.assertTrue(pack["is_v2"])
            self.assertTrue(pack["table_ok"])
            self.assertTrue(pack["bounds_ok"])
            self.assertEqual(pack["entry_count"], entry_count)
            self.assertEqual(pack["max_payload_end"], simulated_total_size)

    def test_wgpk_v2_oob_corrupt_and_invalid_stage_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            shaders_dir.mkdir(parents=True)

            # 3 entries in v2:
            #  0: overlaps TOC (offset 12 < data_start 88)
            #  1: invalid stage 9 + uint64 overflow (offset = u64::MAX - 4, length = 16)
            #  2: exceeds file size
            count = 3
            header = struct.pack("<4sIII", b"WGPK", 2, count, 0)
            e0 = struct.pack("<QB3sIQ", 1, 1, b"\x00\x00\x00", 16, 12)
            e1 = struct.pack("<QB3sIQ", 2, 9, b"\x00\x00\x00", 16, vbb.WGPK_MAX_U64 - 4)
            e2 = struct.pack("<QB3sIQ", 3, 2, b"\x00\x00\x00", 64, 0x1_0000_0000)
            (shaders_dir / "bad_v2.wgsl.pack").write_bytes(header + e0 + e1 + e2 + b"0123456789")

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir
            )
            self.assertFalse(report["ok"])
            pack = report["shaders"]["packs"][0]
            self.assertEqual(pack["version"], 2)
            self.assertEqual(pack["entry_bytes"], 24)
            self.assertTrue(pack["is_v2"])
            self.assertTrue(pack["header_ok"])
            self.assertTrue(pack["table_ok"])
            self.assertFalse(pack["bounds_ok"])
            self.assertEqual(pack["out_of_bounds_entries"], 3)
            self.assertEqual(pack["invalid_stage_entries"], 1)
            self.assertTrue(any("overlaps header/entry table" in e for e in pack["errors"]))
            self.assertTrue(any("overflows uint64" in e for e in pack["errors"]))
            self.assertTrue(any("exceeds pack bounds" in e for e in pack["errors"]))
            self.assertTrue(any("invalid shader stage 9" in e for e in pack["errors"]))


class TestReflectSidecarV2V3Validation(unittest.TestCase):
    def test_valid_v2_and_v3_sidecars_with_root_descriptor_unresolved_and_mesh_output_pass_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            _write_valid_shaders_dir(shaders_dir)

            sidecar_v2 = {
                "schema": "playgta6-stage-reflection",
                "version": 2,
                "stages": {
                    "0x00000000000000a1": {
                        "stage": "vertex",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "uniform",
                                "source": {
                                    "user_sgpr": 4,
                                    "root_descriptor": True,
                                    "table_byte_offset": 64,
                                },
                            },
                        ],
                    }
                },
            }
            sidecar_v3 = {
                "schema": "playgta6-stage-reflection",
                "version": 3,
                "stages": {
                    "0x00000000000000b2": {
                        "stage": "fragment",
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "uniform",
                                "source": {"unresolved": True},
                            },
                            {
                                "group": 0,
                                "binding": 1,
                                "kind": "read-only-storage",
                                "source": {"unresolved": "cbv_dynamic_index"},
                            },
                            {
                                "group": 0,
                                "binding": 2,
                                "kind": "texture_2d",
                                "source": {"unresolved": True},
                            },
                            {
                                "group": 0,
                                "binding": 3,
                                "kind": "sampler_comparison",
                                "source": {"unresolved": "heap_unknown"},
                            },
                        ],
                    },
                    "0x00000000000000c3": {
                        "stage": "compute",
                        "mesh_output": {
                            "max_vertices": 64,
                            "max_primitives": 128,
                            "vertex_words": 16,
                        },
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "read_only_storage",
                                "source": {"unresolved": True},
                            }
                        ],
                    },
                },
            }
            (shaders_dir / "pack_v2.wgsl.pack.reflect.json").write_text(
                json.dumps(sidecar_v2), encoding="utf-8"
            )
            (shaders_dir / "pack_v3.wgsl.pack.reflect.json").write_text(
                json.dumps(sidecar_v3), encoding="utf-8"
            )

            report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=True
            )
            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["warnings"], [])
            shaders = report["shaders"]
            self.assertEqual(shaders["reflect_sidecar_count"], 2)
            self.assertEqual(shaders["valid_reflect_sidecar_count"], 2)
            self.assertEqual(shaders["unresolved_source_stages"], 2)
            self.assertEqual(shaders["unresolved_source_bindings"], 5)
            self.assertEqual(shaders["malformed_v3_sources"], 0)
            self.assertEqual(shaders["mesh_output_stages"], 1)
            self.assertEqual(shaders["malformed_mesh_outputs"], 0)
            self.assertEqual(shaders["malformed_root_descriptors"], 0)

            text = vbb.format_report_text(report)
            self.assertIn("unresolved_source_stages=2", text)
            self.assertIn("unresolved_source_bindings=5", text)
            self.assertIn("malformed_v3_sources=0", text)
            self.assertIn("mesh_output_stages=1", text)

    def test_v3_unresolved_on_writable_kinds_bad_v2_root_descriptor_mesh_output_and_v4_warn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            web_dist = base / "dist"
            shaders_dir = base / "shaders"
            _write_valid_web_dist(web_dist)
            _write_valid_shaders_dir(shaders_dir)

            bad_sidecar = {
                "schema": "playgta6-stage-reflection",
                "version": 4,  # unsupported version -> warning
                "stages": {
                    "0x000000000000bad0": {
                        "stage": "compute",
                        "mesh_output": {
                            "max_vertices": 300,  # > 256
                            "max_primitives": 128,
                            "vertex_words": 6,  # not multiple of 4
                        },
                        "bindings": [
                            {
                                "group": 0,
                                "binding": 0,
                                "kind": "storage",  # writable -> forbidden for unresolved!
                                "source": {"unresolved": True},
                            },
                            {
                                "group": 0,
                                "binding": 1,
                                "kind": "storage_texture",  # writable -> forbidden!
                                "source": {"unresolved": True},
                            },
                            {
                                "group": 0,
                                "binding": 2,
                                "kind": "storage-texture",  # hyphenated writable -> forbidden!
                                "source": {"unresolved": "rw_tex"},
                            },
                            {
                                "group": 0,
                                "binding": 3,
                                "kind": "uniform",
                                "source": {"unresolved": True, "extra": 1},  # len != 1
                            },
                            {
                                "group": 0,
                                "binding": 4,
                                "kind": "uniform",
                                "source": {"unresolved": ""},  # empty string
                            },
                            {
                                "group": 0,
                                "binding": 5,
                                "kind": "uniform",
                                "source": {
                                    "user_sgpr": 2,
                                    "root_descriptor": True,
                                    "table_byte_offset": 40,  # not multiple of 32
                                },
                            },
                        ],
                    }
                },
            }
            (shaders_dir / "bad_v3.wgsl.pack.reflect.json").write_text(
                json.dumps(bad_sidecar), encoding="utf-8"
            )

            non_strict = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=False
            )
            self.assertTrue(non_strict["ok"])
            shaders = non_strict["shaders"]
            self.assertEqual(shaders["valid_reflect_sidecar_count"], 0)
            self.assertEqual(shaders["unresolved_source_bindings"], 0)
            self.assertEqual(shaders["malformed_v3_sources"], 5)
            self.assertEqual(shaders["malformed_root_descriptors"], 1)
            self.assertEqual(shaders["malformed_mesh_outputs"], 1)
            self.assertTrue(any("unsupported reflection sidecar version 4" in w for w in non_strict["warnings"]))

            strict_report = vbb.verify_browser_boot_bundle(
                web_dist=web_dist, shaders_dir=shaders_dir, strict=True
            )
            self.assertFalse(strict_report["ok"])


class TestSingleSideModuleLayoutAndManifestConsistency(unittest.TestCase):
    def test_single_side_module_two_file_layout_with_manifest_passes_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(
                web_dist,
                side_modules={
                    "guest_part_0.wasm": _make_minimal_side_wasm(b"C" * 2048),
                    "guest_part_1.wasm": _make_minimal_side_wasm(b"R" * 32),
                },
            )
            (web_dist / "guest_manifest.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "runtimeJs": "runtime.js",
                        "runtimeWasm": "runtime.wasm",
                        "splitModules": 1,
                        "splitStrategy": "single-side-module",
                        "sideModules": ["guest_part_0.wasm", "guest_part_1.wasm"],
                        "sideModuleSizes": [
                            {"index": 0, "name": "guest_part_0.wasm", "role": "code", "wasmBytes": 2048},
                            {"index": 1, "name": "guest_part_1.wasm", "role": "registration", "wasmBytes": 32},
                        ],
                        "guest_dll_modules": ["bink2scarlett"],
                    }
                ),
                encoding="utf-8",
            )

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertTrue(report["ok"])
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["warnings"], [])
            wasm_sec = report["wasm"]
            self.assertEqual(wasm_sec["bundle_mode"], "split")
            self.assertEqual(wasm_sec["side_module_layout"], "single-side-module")
            self.assertEqual(
                wasm_sec["expected_side_modules"],
                ["guest_part_0.wasm", "guest_part_1.wasm"],
            )
            self.assertEqual(wasm_sec["missing_side_modules"], [])
            self.assertEqual(wasm_sec["stale_side_modules"], [])
            gm = wasm_sec["guest_manifest"]
            self.assertEqual(gm["split_strategy"], "single-side-module")
            self.assertEqual(gm["split_modules"], 1)
            self.assertEqual(len(gm["side_module_sizes"]), 2)

            text = vbb.format_report_text(report)
            self.assertIn("layout=single-side-module", text)

    def test_single_side_module_detected_without_manifest_for_part0_and_small_part1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(
                web_dist,
                side_modules={
                    "guest_part_0.wasm": _make_minimal_side_wasm(b"A" * 1024),
                    "guest_part_1.wasm": _make_minimal_side_wasm(b"B" * 16),
                },
            )
            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertTrue(report["ok"])
            self.assertEqual(report["wasm"]["side_module_layout"], "single-side-module")

    def test_missing_manifest_side_module_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(
                web_dist,
                side_modules={"guest_part_0.wasm": _make_minimal_side_wasm(b"A" * 256)},
            )
            (web_dist / "guest_manifest.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "sideModules": ["guest_part_0.wasm", "guest_part_1.wasm"],
                    }
                ),
                encoding="utf-8",
            )

            report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=False)
            self.assertFalse(report["ok"])
            self.assertEqual(report["wasm"]["missing_side_modules"], ["guest_part_1.wasm"])
            self.assertEqual(report["wasm"]["stale_side_modules"], [])
            self.assertTrue(any("guest_part_1.wasm" in e and "missing" in e for e in report["errors"]))

    def test_stale_guest_part_2_warns_and_fails_strict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            web_dist = Path(tmp) / "dist"
            _write_valid_web_dist(
                web_dist,
                side_modules={
                    "guest_part_0.wasm": _make_minimal_side_wasm(b"A" * 512),
                    "guest_part_1.wasm": _make_minimal_side_wasm(b"B" * 32),
                    "guest_part_2.wasm": _make_minimal_side_wasm(b"STALE" * 64),
                },
            )
            (web_dist / "guest_manifest.json").write_text(
                json.dumps(
                    {
                        "schemaVersion": 1,
                        "sideModules": ["guest_part_0.wasm", "guest_part_1.wasm"],
                    }
                ),
                encoding="utf-8",
            )

            non_strict = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=False)
            self.assertTrue(non_strict["ok"])
            self.assertEqual(non_strict["wasm"]["side_module_layout"], "multi-part")
            self.assertEqual(non_strict["wasm"]["missing_side_modules"], [])
            self.assertEqual(non_strict["wasm"]["stale_side_modules"], ["guest_part_2.wasm"])
            self.assertTrue(any("stale side module" in w and "guest_part_2.wasm" in w for w in non_strict["warnings"]))

            strict_report = vbb.verify_browser_boot_bundle(web_dist=web_dist, strict=True)
            self.assertFalse(strict_report["ok"])


if __name__ == "__main__":
    unittest.main()
