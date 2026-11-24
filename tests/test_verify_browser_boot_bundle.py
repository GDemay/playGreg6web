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


if __name__ == "__main__":
    unittest.main()
