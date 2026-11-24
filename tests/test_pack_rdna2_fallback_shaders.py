#!/usr/bin/env python3
"""Unit tests for tools/pack_rdna2_fallback_shaders.py (Track 4 / S16)."""

from __future__ import annotations

import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import inspect_rdna2_shaders as rdna2  # noqa: E402
from tools import inspect_wgsl_pack as wgpk  # noqa: E402
from tools import pack_rdna2_fallback_shaders as packer  # noqa: E402


def _build_synthetic_rdna2_exp(target_id: int, *, done: bool = True, en_mask: int = 0xF) -> bytes:
    """Build an 8-byte synthetic GFX10.3 EXP instruction (zero game bytes)."""
    word0 = (
        rdna2.ENC_EXP
        | ((1 if done else 0) << 11)
        | ((target_id & 0x3F) << 4)
        | (en_mask & 0xF)
    )
    word1 = 0x00000000
    return struct.pack("<II", word0, word1)


def _build_synthetic_rdna2_blob() -> bytes:
    """Build synthetic vertex, pixel, compute, and unknown RDNA2 test regions."""
    s_endpgm = struct.pack("<I", rdna2.S_ENDPGM_WORD)
    pad = struct.pack("<II", 0, 0)
    # Vertex: pos0 export (target_id=12) + s_endpgm
    vs_blob = _build_synthetic_rdna2_exp(12, done=True) + s_endpgm
    # Pixel: mrt0 export (target_id=0) + s_endpgm
    ps_blob = _build_synthetic_rdna2_exp(0, done=True) + s_endpgm
    # Compute: SMEM (0xF4000000, 2 dwords) + VALU (VOP1 0x7E000000) + s_endpgm
    cs_blob = struct.pack("<III", rdna2.ENC_SMEM, 0x00000000, rdna2.ENC_VOP1) + s_endpgm
    # Unknown: SOPP nop (0xBF800000) + s_endpgm (no exp, no smem/valu)
    unk_blob = struct.pack("<I", rdna2.ENC_SOPP) + s_endpgm
    return pad + vs_blob + pad + ps_blob + pad + cs_blob + pad + unk_blob + pad


class TestBinaryFormatAndInterop(unittest.TestCase):
    """Verify exact WGPK v1 binary framing and cross-tool/JS compatibility."""

    def test_binary_header_entry_table_and_payload_framing(self) -> None:
        records = [
            packer.ShaderInputRecord(0xDEADBEEFCAFEBABE, packer.ShaderStage.COMPUTE, "cli_hashes"),
            packer.ShaderInputRecord(0x0000000000000001, packer.ShaderStage.FRAGMENT, "cli_hashes"),
            packer.ShaderInputRecord(0x0000000000000001, packer.ShaderStage.VERTEX, "cli_hashes"),
        ]
        result = packer.build_fallback_pack(records, profile="fullscreen_triangle_cbv")
        raw = result.pack_bytes

        # 16-byte header: magic, version, entry_count, flags
        magic, version, entry_count, flags = struct.unpack_from("<4sIII", raw, 0)
        self.assertEqual(magic, b"WGPK")
        self.assertEqual(version, 1)
        self.assertEqual(entry_count, 3)
        self.assertEqual(flags, 0)

        data_start = 16 + 3 * 20
        expected_order = [
            (0x0000000000000001, packer.ShaderStage.VERTEX),
            (0x0000000000000001, packer.ShaderStage.FRAGMENT),
            (0xDEADBEEFCAFEBABE, packer.ShaderStage.COMPUTE),
        ]
        cursor = data_start
        for idx, (exp_hash, exp_stage) in enumerate(expected_order):
            base = 16 + idx * 20
            shader_hash, stage_raw, reserved, offset, length = struct.unpack_from("<QB3sII", raw, base)
            self.assertEqual(shader_hash, exp_hash)
            self.assertEqual(stage_raw, int(exp_stage))
            self.assertEqual(reserved, b"\x00\x00\x00")
            self.assertEqual(offset, cursor)
            expected_wgsl = packer.BUILTIN_PROFILES["fullscreen_triangle_cbv"][exp_stage]
            payload = raw[offset : offset + length]
            self.assertEqual(payload.decode("utf-8"), expected_wgsl)
            cursor += length

        self.assertEqual(cursor, len(raw))

        # Byte-for-byte parity with tools/inspect_wgsl_pack.py parser & builder
        parsed = wgpk.parse_wgpk(raw)
        self.assertEqual(parsed.version, 1)
        self.assertEqual(parsed.flags, 0)
        self.assertEqual(parsed.entry_count, 3)
        rebuilt = wgpk.build_wgpk(
            [(e.hash, e.stage, e.wgsl) for e in parsed.entries],
            flags=0,
        )
        self.assertEqual(raw, rebuilt)
        self.assertEqual(result.stage_conflict_hash_count, 1)

        # Every individual entry passes WebGPU default limit budget analysis cleanly
        for entry in parsed.entries:
            shader_audit = wgpk.analyze_wgsl(entry.wgsl, entry.hash, pack_stage=entry.stage_name)
            self.assertTrue(shader_audit.clean, msg=str(shader_audit.violations))

    def test_all_profiles_pass_wgpk_budget_audit(self) -> None:
        records = [
            packer.ShaderInputRecord(0x10, packer.ShaderStage.VERTEX, "cli_hashes"),
            packer.ShaderInputRecord(0x20, packer.ShaderStage.FRAGMENT, "cli_hashes"),
            packer.ShaderInputRecord(0x30, packer.ShaderStage.COMPUTE, "cli_hashes"),
        ]
        for profile in packer.PROFILE_CHOICES:
            with self.subTest(profile=profile):
                res = packer.build_fallback_pack(records, profile=profile)
                bundle = wgpk.parse_wgpk(res.pack_bytes)
                self.assertEqual(bundle.entry_count, 3)
                self.assertEqual(
                    bundle.stage_counts(),
                    {"vertex": 1, "fragment": 1, "compute": 1},
                )
                inspect_rep = wgpk.inspect_pack(bundle)
                self.assertTrue(inspect_rep.clean, msg=str(inspect_rep.violations))
                self.assertEqual(len(inspect_rep.violations), 0)

    def test_cross_check_with_js_wgsl_pack_loader_and_g34_profiles(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("node not installed")

        with tempfile.TemporaryDirectory() as tmp:
            pack_path = Path(tmp) / "cross_check.wgsl.pack"
            rc = packer.main(
                [
                    "--hashes",
                    "vertex:0x1122334455667788,fragment:0x8877665544332211,compute:0xaabbccddeeff0011",
                    "--profile",
                    "textured_quad_24b",
                    "--out",
                    str(pack_path),
                ]
            )
            self.assertEqual(rc, 0)

            loader_url = (REPO_ROOT / "web" / "workers" / "wgsl_pack_loader.js").as_uri()
            router_path = str(REPO_ROOT / "web" / "workers" / "shader_fallback_router.js")
            script = """
import fs from 'node:fs';
import { createRequire } from 'node:module';
const { parseWgslPack } = await import(process.argv[1]);
const require = createRequire(import.meta.url);
const router = require(process.argv[2]);
const buf = fs.readFileSync(process.argv[3]);
const pack = parseWgslPack(buf);
const profiles = router.getBuiltInFallbackProfiles();
const entries = pack.listEntries().map((e) => ({
  hashHex: e.hashHex,
  stage: e.stage,
  stageCode: e.stageCode,
  wgsl: e.wgsl,
}));
console.log(JSON.stringify({ version: pack.version, flags: pack.flags, entries, profiles }));
"""
            proc = subprocess.run(
                [node, "--input-type=module", "-e", script, loader_url, router_path, str(pack_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            js_out = json.loads(proc.stdout)

            self.assertEqual(js_out["version"], 1)
            self.assertEqual(js_out["flags"], 0)
            self.assertEqual(len(js_out["entries"]), 3)
            self.assertEqual(
                [(e["hashHex"], e["stage"], e["stageCode"]) for e in js_out["entries"]],
                [
                    ("0x1122334455667788", "vertex", 1),
                    ("0x8877665544332211", "fragment", 2),
                    ("0xaabbccddeeff0011", "compute", 3),
                ],
            )

            # Verify every built-in profile matches G34 shader_fallback_router.js byte-for-byte
            js_profiles = js_out["profiles"]
            for profile_name in packer.PROFILE_CHOICES:
                for stage in (
                    packer.ShaderStage.VERTEX,
                    packer.ShaderStage.FRAGMENT,
                    packer.ShaderStage.COMPUTE,
                ):
                    self.assertEqual(
                        packer.BUILTIN_PROFILES[profile_name][stage],
                        js_profiles[profile_name][stage.as_str()],
                        msg=f"mismatch for profile={profile_name} stage={stage.as_str()}",
                    )


class TestDeduplicationSortingAndOverrides(unittest.TestCase):
    """Verify deterministic deduplication, ordering, scan-json ingestion, and WGSL overrides."""

    def test_deterministic_dedup_and_order_independence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scan_a = tmp_path / "scan_a.json"
            scan_b = tmp_path / "scan_b.json"

            scan_a.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "files": [
                            {
                                "path": "bundle_a.bin",
                                "shaders": [
                                    {"hash": "0x00000000000000ff", "stage": "pixel"},
                                    {"hash": "0x000000000000000a", "stage": "vertex"},
                                    {"hash": "0x00000000000000ff", "stage": "pixel"},
                                    {"hash": "0x000000000000000a", "stage": "unknown"},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            scan_b.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "files": [
                            {
                                "path": "bundle_b.bin",
                                "shaders": [
                                    {"hash": "0x000000000000000a", "stage": "vertex"},
                                    {"hash": "0x0000000000000010", "stage": "compute"},
                                    {"hash": "0x00000000000000ff", "stage": "pixel"},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_1 = tmp_path / "pack1.wgsl.pack"
            out_2 = tmp_path / "pack2.wgsl.pack"
            rep_1 = tmp_path / "report1.json"

            rc1 = packer.main(
                [
                    "--scan-json",
                    str(scan_a),
                    "--scan-json",
                    str(scan_b),
                    "--hashes",
                    "compute:0x10,vertex:0x0a",
                    "--profile",
                    "fullscreen_triangle_cbv",
                    "--out",
                    str(out_1),
                    "--report-json",
                    str(rep_1),
                ]
            )
            rc2 = packer.main(
                [
                    "--scan-json",
                    str(scan_b),
                    "--scan-json",
                    str(scan_a),
                    "--hashes",
                    "fragment:0xff,compute:0x0000000000000010",
                    "--profile",
                    "fullscreen_triangle_cbv",
                    "--out",
                    str(out_2),
                ]
            )
            self.assertEqual(rc1, 0)
            self.assertEqual(rc2, 0)
            self.assertEqual(out_1.read_bytes(), out_2.read_bytes())

            report = json.loads(rep_1.read_text(encoding="utf-8"))
            self.assertEqual(report["summary"]["packed_entry_count"], 3)
            self.assertEqual(report["summary"]["unique_hash_count"], 3)
            self.assertEqual(report["summary"]["duplicate_entries_collapsed"], 6)
            self.assertEqual(report["summary"]["skipped_unknown_stage_count"], 0)
            self.assertEqual(
                [(e["hash"], e["stage"]) for e in report["entries"]],
                [
                    ("0x000000000000000a", "vertex"),
                    ("0x0000000000000010", "compute"),
                    ("0x00000000000000ff", "fragment"),
                ],
            )

    def test_end_to_end_from_inspect_rdna2_shaders_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rdna2_bin = tmp_path / "synthetic_shaders.bin"
            rdna2_bin.write_bytes(_build_synthetic_rdna2_blob())

            scan_report = rdna2.scan_path(rdna2_bin)
            scan_json = tmp_path / "rdna2_scan.json"
            scan_json.write_text(rdna2.render_json(scan_report), encoding="utf-8")

            out_pack = tmp_path / "from_scan.wgsl.pack"
            out_report = tmp_path / "pack_report.json"

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = packer.main(
                    [
                        "--scan-json",
                        str(scan_json),
                        "--profile",
                        "fullscreen_triangle_cbv",
                        "--out",
                        str(out_pack),
                        "--report-json",
                        str(out_report),
                    ]
                )
            self.assertEqual(rc, 0)

            parsed = wgpk.parse_wgpk(out_pack.read_bytes())
            self.assertEqual(parsed.entry_count, 3)
            self.assertEqual(parsed.stage_counts(), {"vertex": 1, "fragment": 1, "compute": 1})
            inspect_rep = wgpk.inspect_pack(parsed)
            self.assertTrue(inspect_rep.clean, msg=str(inspect_rep.violations))
            self.assertEqual(len(inspect_rep.violations), 0)

            rep = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(rep["summary"]["packed_entry_count"], 3)
            self.assertEqual(rep["summary"]["skipped_unknown_stage_count"], 1)
            self.assertEqual(rep["summary"]["profile_entry_count"], 3)
            self.assertEqual(rep["summary"]["override_entry_count"], 0)

    def test_wgsl_dir_override_precedence_and_unknown_stage_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wgsl_dir = tmp_path / "overrides"
            wgsl_dir.mkdir()

            h1 = 0x0123456789ABCDEF
            h2 = 0xFEDCBA9876543210
            h3 = 0x000000000000BEEF
            h4 = 0x000000000000CAFE

            stage_specific_wgsl = "@vertex\nfn custom_vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            generic_h1_wgsl = "@vertex\nfn generic_vs() -> @builtin(position) vec4<f32> { return vec4<f32>(1.0); }\n"
            generic_h2_wgsl = "@fragment\nfn custom_fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n"
            rescued_cs_wgsl = "@compute @workgroup_size(8)\nfn rescued_cs() {}\n"

            # Both stage-prefixed and hash-only exist for h1 -> stage-prefixed MUST win.
            (wgsl_dir / f"vertex_{h1:016x}.wgsl").write_text(stage_specific_wgsl, encoding="utf-8")
            (wgsl_dir / f"{h1:016x}.wgsl").write_text(generic_h1_wgsl, encoding="utf-8")
            # Uppercase hash-only file for h2 -> case-insensitive lookup MUST match.
            (wgsl_dir / f"{h2:016X}.wgsl").write_text(generic_h2_wgsl, encoding="utf-8")
            # Hash-only compute shader for unknown-stage h3 -> stage inferred as compute.
            (wgsl_dir / f"0x{h3:016x}.wgsl").write_text(rescued_cs_wgsl, encoding="utf-8")

            scan_json = tmp_path / "scan.json"
            scan_json.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "files": [
                            {
                                "path": "test.bin",
                                "shaders": [
                                    {"hash": f"0x{h1:016x}", "stage": "vertex"},
                                    {"hash": f"0x{h2:016x}", "stage": "pixel"},
                                    {"hash": f"0x{h3:016x}", "stage": "unknown"},
                                    {"hash": f"0x{h4:016x}", "stage": "compute"},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            out_pack = tmp_path / "mixed.wgsl.pack"
            out_report = tmp_path / "mixed_report.json"

            rc = packer.main(
                [
                    "--scan-json",
                    str(scan_json),
                    "--wgsl-dir",
                    str(wgsl_dir),
                    "--profile",
                    "default",
                    "--out",
                    str(out_pack),
                    "--report-json",
                    str(out_report),
                ]
            )
            self.assertEqual(rc, 0)

            bundle = wgpk.parse_wgpk(out_pack.read_bytes())
            self.assertEqual(bundle.entry_count, 4)

            e_h3 = bundle.entry_by_hash(h3)
            e_h4 = bundle.entry_by_hash(h4)
            e_h1 = bundle.entry_by_hash(h1)
            e_h2 = bundle.entry_by_hash(h2)
            assert e_h1 and e_h2 and e_h3 and e_h4

            self.assertEqual(e_h1.wgsl, stage_specific_wgsl)
            self.assertEqual(e_h2.wgsl, generic_h2_wgsl)
            self.assertEqual(e_h3.wgsl, rescued_cs_wgsl)
            self.assertEqual(e_h3.stage, wgpk.ShaderStage.COMPUTE)
            self.assertEqual(e_h4.wgsl, packer.BUILTIN_PROFILES["default"][packer.ShaderStage.COMPUTE])

            rep = json.loads(out_report.read_text(encoding="utf-8"))
            self.assertEqual(rep["summary"]["override_entry_count"], 3)
            self.assertEqual(rep["summary"]["profile_entry_count"], 1)
            self.assertEqual(rep["summary"]["skipped_unknown_stage_count"], 0)
            by_hash = {item["hash"]: item for item in rep["entries"]}
            self.assertEqual(by_hash[f"0x{h1:016x}"]["source"], "override")
            self.assertEqual(by_hash[f"0x{h1:016x}"]["override_file"], f"vertex_{h1:016x}.wgsl")
            self.assertEqual(by_hash[f"0x{h1:016x}"]["entry_point"], "custom_vs")
            self.assertEqual(by_hash[f"0x{h4:016x}"]["source"], "profile")
            self.assertEqual(by_hash[f"0x{h4:016x}"]["profile"], "default")
            self.assertIsNone(by_hash[f"0x{h4:016x}"]["override_file"])


class TestSanitizationAndSafetyGuards(unittest.TestCase):
    """Verify strict path/username sanitization, zero raw bytes, and tracked-dir guards."""

    def test_report_json_and_stdout_strip_paths_usernames_and_raw_bytes(self) -> None:
        fake_user = "secret_dev_user"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Include fake_user in directory and file names to verify scrubbing.
            nested = tmp_path / f"dir_{fake_user}"
            nested.mkdir()
            scan_path = nested / f"scan_{fake_user}.json"
            scan_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "files": [
                            {
                                "path": f"C:\\Users\\{fake_user}\\PrivateGame\\shader.bin",
                                "shaders": [
                                    {"hash": "0x0000000000001234", "stage": "vertex"},
                                    {"hash": "0x0000000000005678", "stage": "unknown"},
                                ],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            wgsl_dir = nested / f"wgsl_{fake_user}"
            wgsl_dir.mkdir()
            override_wgsl = "@vertex\nfn main_vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n"
            (wgsl_dir / "0000000000001234.wgsl").write_text(override_wgsl, encoding="utf-8")

            out_pack = nested / f"out_{fake_user}.wgsl.pack"
            out_report = nested / f"report_{fake_user}.json"

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with mock.patch.dict(os.environ, {"USERNAME": fake_user, "USER": fake_user}):
                with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                    rc = packer.main(
                        [
                            "--scan-json",
                            str(scan_path),
                            "--wgsl-dir",
                            str(wgsl_dir),
                            "--profile",
                            "default",
                            "--out",
                            str(out_pack),
                            "--report-json",
                            str(out_report),
                        ]
                    )

            self.assertEqual(rc, 0, msg=stderr_buf.getvalue())
            report_text = out_report.read_text(encoding="utf-8")
            stdout_text = stdout_buf.getvalue()

            for text in (report_text, stdout_text):
                self.assertNotIn(fake_user, text)
                self.assertNotIn(str(tmp_path), text)
                self.assertNotRegex(text, r"[A-Za-z]:[\\/]")
                self.assertNotRegex(text, r"\\\\[^\s\"']+")
                self.assertNotRegex(text, r"/(?:Users|home|tmp|var|private)/")
                # Never dump raw WGSL source or binary shader bytes into report/stdout
                self.assertNotIn("@builtin(position)", text)
                self.assertNotIn("WGPK\\x", text)

            # Confirm sanitize_text scrubs Windows, UNC, POSIX paths, and usernames directly
            with mock.patch.dict(os.environ, {"USERNAME": fake_user, "USER": fake_user}):
                scrubbed = packer.sanitize_text(
                    f"win=C:\\Users\\{fake_user}\\game\\a.bin "
                    f"unc=\\\\filesrv\\share\\{fake_user}\\b.bin "
                    f"posix=/home/{fake_user}/game/c.bin "
                    f"raw={fake_user}"
                )
            self.assertNotIn(fake_user, scrubbed)
            self.assertNotIn("C:", scrubbed)
            self.assertNotIn("filesrv", scrubbed)
            self.assertNotIn("/home/", scrubbed)
            self.assertIn("<path>", scrubbed)
            self.assertIn("<user>", scrubbed)

    def test_refuses_tracked_repo_output_paths_and_allows_work_or_external(self) -> None:
        with tempfile.TemporaryDirectory() as fake_repo_str:
            fake_repo = Path(fake_repo_str)
            (fake_repo / "web").mkdir()
            (fake_repo / "work").mkdir()
            (fake_repo / "local-data").mkdir()

            # Tracked directory -> refused
            self.assertFalse(packer.is_safe_output_path(fake_repo / "web" / "bad.wgsl.pack", repo_root=fake_repo))
            self.assertFalse(packer.is_safe_output_path(fake_repo / "bad.wgsl.pack", repo_root=fake_repo))

            # Gitignored work/ and local-data/ -> allowed
            self.assertTrue(packer.is_safe_output_path(fake_repo / "work" / "ok.wgsl.pack", repo_root=fake_repo))
            self.assertTrue(packer.is_safe_output_path(fake_repo / "local-data" / "ok.wgsl.pack", repo_root=fake_repo))

            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                rc = packer.main(
                    [
                        "--hashes",
                        "vertex:0x1",
                        "--out",
                        str(fake_repo / "web" / "bad.wgsl.pack"),
                    ],
                    repo_root=fake_repo,
                )
            self.assertEqual(rc, 1)
            self.assertIn("tracked repository directory", stderr_buf.getvalue())
            self.assertFalse((fake_repo / "web" / "bad.wgsl.pack").exists())

            # Also refuses --report-json in tracked repo directory
            stderr_buf = io.StringIO()
            with redirect_stderr(stderr_buf):
                rc = packer.main(
                    [
                        "--hashes",
                        "vertex:0x1",
                        "--out",
                        str(fake_repo / "work" / "ok.wgsl.pack"),
                        "--report-json",
                        str(fake_repo / "web" / "report.json"),
                    ],
                    repo_root=fake_repo,
                )
            self.assertEqual(rc, 1)
            self.assertIn("--report-json", stderr_buf.getvalue())
            self.assertFalse((fake_repo / "work" / "ok.wgsl.pack").exists())

    def test_error_handling_and_sanitized_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            out_pack = tmp_path / "err.wgsl.pack"

            # Missing both --scan-json and --hashes
            err = io.StringIO()
            with redirect_stderr(err):
                rc = packer.main(["--out", str(out_pack)])
            self.assertEqual(rc, 2)

            # Missing scan JSON file (must not leak full host path)
            missing = tmp_path / "does_not_exist.json"
            err = io.StringIO()
            with redirect_stderr(err):
                rc = packer.main(["--scan-json", str(missing), "--out", str(out_pack)])
            self.assertEqual(rc, 1)
            self.assertIn("does_not_exist.json", err.getvalue())
            self.assertNotIn(str(tmp_path), err.getvalue())

            # Malformed --hashes spec
            err = io.StringIO()
            with redirect_stderr(err):
                rc = packer.main(["--hashes", "notastage:0x123", "--out", str(out_pack)])
            self.assertEqual(rc, 1)

            # Invalid UTF-8 override file
            wgsl_dir = tmp_path / "bad_wgsl"
            wgsl_dir.mkdir()
            (wgsl_dir / "0000000000000001.wgsl").write_bytes(b"\xff\xfe\xfd")
            err = io.StringIO()
            with redirect_stderr(err):
                rc = packer.main(
                    [
                        "--hashes",
                        "vertex:0x1",
                        "--wgsl-dir",
                        str(wgsl_dir),
                        "--out",
                        str(out_pack),
                    ]
                )
            self.assertEqual(rc, 1)
            self.assertIn("UTF-8", err.getvalue())

    def test_wgsl_dir_skips_escaping_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            outside = tmp_path / "outside.wgsl"
            outside.write_text("@vertex\nfn evil() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }\n", encoding="utf-8")

            wgsl_dir = tmp_path / "overrides"
            wgsl_dir.mkdir()
            link_path = wgsl_dir / "0000000000000001.wgsl"
            try:
                os.symlink(outside, link_path)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks require elevated privileges on this OS")

            res = packer.build_fallback_pack(
                [packer.ShaderInputRecord(1, packer.ShaderStage.VERTEX, "cli_hashes")],
                profile="default",
                wgsl_dir=wgsl_dir,
            )
            self.assertEqual(res.entries[0].source, "profile")
            self.assertTrue(any("symlink" in w for w in res.warnings))


if __name__ == "__main__":
    unittest.main()
