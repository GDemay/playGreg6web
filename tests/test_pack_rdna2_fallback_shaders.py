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


class TestWholeFileNativeDumpPackingAndEntryPoints(unittest.TestCase):
    """Track 4 / S20: whole-file identity keying and `fs_main` fragment entry point."""

    def _build_native_dump_fixture(self, dump_dir: Path) -> dict[str, int]:
        """Create a synthetic `playgta6-pipeline-dump` v1 dir with padded & 2-s_endpgm stages."""
        native_dir = dump_dir / "native"
        native_dir.mkdir(parents=True)
        s_endpgm = struct.pack("<I", rdna2.S_ENDPGM_WORD)
        s_code_end = struct.pack("<I", rdna2.S_CODE_END_WORD)
        s_nop = struct.pack("<I", rdna2.ENC_SOPP)

        # Vertex stage: code up to first s_endpgm + trailing padding (100% of real dump files!)
        vs_trimmed = _build_synthetic_rdna2_exp(12, done=True) + s_endpgm
        vs_full = vs_trimmed + (s_code_end * 4) + (b"\x00" * 16)
        vs_whole_hash = rdna2.fnv1a_64(vs_full)
        vs_trimmed_hash = rdna2.fnv1a_64(vs_trimmed)
        self.assertNotEqual(vs_whole_hash, vs_trimmed_hash)

        # Pixel stage: early-out s_endpgm + main mrt0 export + second s_endpgm + trailing padding
        ps_first = s_nop + s_nop + s_endpgm
        ps_second = _build_synthetic_rdna2_exp(0, done=True) + s_endpgm
        ps_full = ps_first + ps_second + (s_code_end * 4) + (b"\x00" * 16)
        ps_whole_hash = rdna2.fnv1a_64(ps_full)
        ps_trimmed_hash = rdna2.fnv1a_64(ps_first)
        self.assertNotEqual(ps_whole_hash, ps_trimmed_hash)

        vs_rel = f"native/0x{vs_whole_hash:016x}.bin"
        ps_rel = f"native/0x{ps_whole_hash:016x}.bin"
        (dump_dir / vs_rel).write_bytes(vs_full)
        (dump_dir / ps_rel).write_bytes(ps_full)

        manifest = {
            "schema": "playgta6-pipeline-dump",
            "schema_version": 1,
            "native_stages": [
                {
                    "code_identity": f"0x{vs_whole_hash:016x}",
                    "identity": f"0x{vs_whole_hash:016x}",
                    "stage": "vertex",
                    "layout": "A",
                    "code_bytes": len(vs_full),
                    "metadata_bytes": 0,
                    "code_file": vs_rel,
                    "metadata_file": f"native/0x{vs_whole_hash:016x}.meta",
                },
                {
                    "code_identity": f"0x{ps_whole_hash:016x}",
                    "identity": f"0x{ps_whole_hash:016x}",
                    "stage": "pixel",
                    "layout": "A",
                    "code_bytes": len(ps_full),
                    "metadata_bytes": 0,
                    "code_file": ps_rel,
                    "metadata_file": f"native/0x{ps_whole_hash:016x}.meta",
                },
            ],
            "dxil": [],
            "root_signatures": [],
            "pipelines": [],
        }
        (dump_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        return {
            "vs_whole_hash": vs_whole_hash,
            "vs_trimmed_hash": vs_trimmed_hash,
            "ps_whole_hash": ps_whole_hash,
            "ps_trimmed_hash": ps_trimmed_hash,
        }

    def test_pack_keys_native_dump_files_by_whole_file_identity_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dump_dir = tmp_path / "pipeline_dump"
            hashes = self._build_native_dump_fixture(dump_dir)

            # Path 1: S15 scan JSON -> S16 packer
            scan_report = rdna2.scan_path(dump_dir)
            scan_json = tmp_path / "s15_scan.json"
            scan_json.write_text(rdna2.render_json(scan_report), encoding="utf-8")
            pack_from_scan = tmp_path / "from_scan.wgsl.pack"
            report_from_scan = tmp_path / "from_scan_report.json"

            buf = io.StringIO()
            with redirect_stdout(buf):
                rc1 = packer.main(
                    [
                        "--scan-json",
                        str(scan_json),
                        "--profile",
                        "fullscreen_triangle_cbv",
                        "--out",
                        str(pack_from_scan),
                        "--report-json",
                        str(report_from_scan),
                    ]
                )
            self.assertEqual(rc1, 0)

            # Path 2: Direct playgta6-pipeline-dump manifest.json -> S16 packer
            pack_from_manifest = tmp_path / "from_manifest.wgsl.pack"
            with redirect_stdout(io.StringIO()):
                rc2 = packer.main(
                    [
                        "--scan-json",
                        str(dump_dir / "manifest.json"),
                        "--profile",
                        "fullscreen_triangle_cbv",
                        "--out",
                        str(pack_from_manifest),
                    ]
                )
            self.assertEqual(rc2, 0)

            # Path 3: Direct --native-dir -> S16 packer
            pack_from_native_dir = tmp_path / "from_native_dir.wgsl.pack"
            with redirect_stdout(io.StringIO()):
                rc3 = packer.main(
                    [
                        "--native-dir",
                        str(dump_dir),
                        "--profile",
                        "fullscreen_triangle_cbv",
                        "--out",
                        str(pack_from_native_dir),
                    ]
                )
            self.assertEqual(rc3, 0)

            # All three paths must produce byte-identical WGPK packs!
            pack_bytes = pack_from_scan.read_bytes()
            self.assertEqual(pack_bytes, pack_from_manifest.read_bytes())
            self.assertEqual(pack_bytes, pack_from_native_dir.read_bytes())

            bundle = wgpk.parse_wgpk(pack_bytes)
            self.assertEqual(bundle.entry_count, 2)
            self.assertEqual(bundle.stage_counts(), {"vertex": 1, "fragment": 1, "compute": 0})

            vs_entry = bundle.entry_by_hash(hashes["vs_whole_hash"])
            ps_entry = bundle.entry_by_hash(hashes["ps_whole_hash"])
            self.assertIsNotNone(vs_entry)
            self.assertIsNotNone(ps_entry)
            assert vs_entry is not None and ps_entry is not None
            self.assertEqual(vs_entry.stage, wgpk.ShaderStage.VERTEX)
            self.assertEqual(ps_entry.stage, wgpk.ShaderStage.FRAGMENT)
            self.assertIn("fn vs_main", vs_entry.wgsl)
            self.assertIn("fn fs_main", ps_entry.wgsl)

            # Trimmed first-s_endpgm hashes must NOT be used as pack keys!
            self.assertIsNone(bundle.entry_by_hash(hashes["vs_trimmed_hash"]))
            self.assertIsNone(bundle.entry_by_hash(hashes["ps_trimmed_hash"]))

            rep = json.loads(report_from_scan.read_text(encoding="utf-8"))
            by_hash = {e["hash"]: e for e in rep["entries"]}
            self.assertEqual(by_hash[f"0x{hashes['vs_whole_hash']:016x}"]["entry_point"], "vs_main")
            self.assertEqual(by_hash[f"0x{hashes['ps_whole_hash']:016x}"]["entry_point"], "fs_main")
            self.assertTrue(wgpk.inspect_pack(bundle).clean)

    def test_fragment_override_normalizes_ps_main_to_fs_main(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            dump_dir = tmp_path / "pipeline_dump"
            hashes = self._build_native_dump_fixture(dump_dir)

            wgsl_dir = tmp_path / "overrides"
            wgsl_dir.mkdir()
            legacy_ps_wgsl = (
                "// Legacy S17 output mentioning fn ps_main in a comment\n"
                "@fragment\n"
                "fn ps_main() -> @location(0) vec4<f32> {\n"
                "    return vec4<f32>(0.2, 0.4, 0.6, 1.0);\n"
                "}\n"
            )
            (wgsl_dir / f"0x{hashes['ps_whole_hash']:016x}.wgsl").write_text(
                legacy_ps_wgsl,
                encoding="utf-8",
            )

            out_pack = tmp_path / "normalized.wgsl.pack"
            out_report = tmp_path / "normalized_report.json"
            with redirect_stdout(io.StringIO()):
                rc = packer.main(
                    [
                        "--native-dir",
                        str(dump_dir),
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
            ps_entry = bundle.entry_by_hash(hashes["ps_whole_hash"])
            assert ps_entry is not None
            self.assertIn("fn fs_main()", ps_entry.wgsl)
            self.assertIn("// Legacy S17 output mentioning fn ps_main in a comment", ps_entry.wgsl)
            self.assertEqual(packer.extract_entry_point(ps_entry.wgsl, packer.ShaderStage.FRAGMENT), "fs_main")

            rep = json.loads(out_report.read_text(encoding="utf-8"))
            by_hash = {e["hash"]: e for e in rep["entries"]}
            self.assertEqual(by_hash[f"0x{hashes['ps_whole_hash']:016x}"]["entry_point"], "fs_main")
            self.assertTrue(any("ps_main" in w and "fs_main" in w for w in rep["warnings"]))

            # Already-normalized fs_main override does not emit a rename warning
            already_ok_wgsl = "@fragment\nfn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }\n"
            updated, renamed = packer.normalize_fragment_entry_point(already_ok_wgsl)
            self.assertFalse(renamed)
            self.assertEqual(updated, already_ok_wgsl)


class TestWgpkVerificationAndBounds(unittest.TestCase):
    """T91: Verify WGPK v1 header, table/payload bounds, overlaps, UTF-8, and CLI --verify-pack."""

    def _raw_pack(
        self,
        entries: list[tuple[int, int, bytes, int, int]],
        *,
        magic: bytes = packer.PACK_MAGIC,
        version: int = packer.PACK_VERSION,
        entry_count: int | None = None,
        flags: int = 0,
        payload_blob: bytes = b"",
    ) -> bytes:
        count = len(entries) if entry_count is None else entry_count
        header = packer.PACK_HEADER_STRUCT.pack(magic, version, count, flags)
        table = b"".join(
            packer.PACK_ENTRY_STRUCT.pack(h, st, res, off, length)
            for h, st, res, off, length in entries
        )
        return header + table + payload_blob

    def test_verify_wgpk_pack_valid_and_bounds_errors(self) -> None:
        vs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.VERTEX]
        fs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.FRAGMENT]
        valid_bytes, _ = packer.serialize_wgpk(
            [(0x10, packer.ShaderStage.VERTEX, vs), (0x20, packer.ShaderStage.FRAGMENT, fs)],
            flags=7,
        )

        summary = packer.verify_wgpk_pack(valid_bytes, require_canonical_order=True)
        self.assertEqual(summary.version, 1)
        self.assertEqual(summary.flags, 7)
        self.assertEqual(summary.entry_count, 2)
        self.assertEqual(summary.unique_hash_count, 2)
        self.assertEqual(summary.total_wgsl_bytes, len(vs.encode("utf-8")) + len(fs.encode("utf-8")))
        self.assertTrue(summary.canonical_hash_order)
        self.assertTrue(summary.canonical_pair_order)
        self.assertEqual(summary.stage_counts, {"vertex": 1, "fragment": 1, "compute": 0})

        # Truncated header (empty and < 16 bytes)
        for bad_hdr in (b"", b"WGPK", b"WGPK" + b"\x00" * 11):
            with self.assertRaises(packer.PackerError) as ctx:
                packer.verify_wgpk_pack(bad_hdr)
            self.assertEqual(ctx.exception.code, "wgpk_truncated_header")

        # Invalid magic
        bad_magic = self._raw_pack([], magic=b"NOPE")
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(bad_magic)
        self.assertEqual(ctx.exception.code, "wgpk_invalid_magic")

        # Unsupported versions
        for bad_ver in (0, 2, 99):
            with self.assertRaises(packer.PackerError) as ctx:
                packer.verify_wgpk_pack(self._raw_pack([], version=bad_ver))
            self.assertEqual(ctx.exception.code, "wgpk_unsupported_version")

        # Entry count exceeding MAX_PACK_ENTRIES (u32 table overflow)
        overflow_hdr = packer.PACK_HEADER_STRUCT.pack(
            packer.PACK_MAGIC, packer.PACK_VERSION, packer.MAX_PACK_ENTRIES + 1, 0
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(overflow_hdr)
        self.assertEqual(ctx.exception.code, "pack_overflow")

        # Truncated entry table
        truncated_table = self._raw_pack(
            [(0x1, 1, b"\x00\x00\x00", 56, 0)],
            entry_count=2,
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(truncated_table)
        self.assertEqual(ctx.exception.code, "wgpk_truncated_table")

        # Invalid stage code
        bad_stage = self._raw_pack(
            [(0x1, 9, b"\x00\x00\x00", 36, 4)],
            payload_blob=b"abcd",
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(bad_stage)
        self.assertEqual(ctx.exception.code, "wgpk_invalid_stage")

        # Payload offset pointing inside header/entry table (< data_start)
        offset_in_header = self._raw_pack(
            [(0x1, 1, b"\x00\x00\x00", 8, 4)],
            payload_blob=b"abcd",
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(offset_in_header)
        self.assertEqual(ctx.exception.code, "wgpk_out_of_bounds")

        # Payload extending past EOF
        past_eof = self._raw_pack(
            [(0x1, 1, b"\x00\x00\x00", 36, 10)],
            payload_blob=b"abcd",
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(past_eof)
        self.assertEqual(ctx.exception.code, "wgpk_out_of_bounds")

        # Payload offset + length overflowing u32::MAX
        u32_overflow = self._raw_pack(
            [(0x1, 1, b"\x00\x00\x00", 0xFFFFFFF0, 0x20)],
            payload_blob=b"abcd",
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(u32_overflow)
        self.assertEqual(ctx.exception.code, "wgpk_payload_overflow")

        # Overlapping payload ranges
        data_start = packer.PACK_HEADER_BYTES + 2 * packer.PACK_ENTRY_BYTES
        overlapping = self._raw_pack(
            [
                (0x1, 1, b"\x00\x00\x00", data_start, 8),
                (0x2, 2, b"\x00\x00\x00", data_start + 4, 8),
            ],
            payload_blob=b"0123456789ab",
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(overlapping)
        self.assertEqual(ctx.exception.code, "wgpk_overlapping_payloads")

        # Invalid UTF-8 payload
        bad_utf8 = self._raw_pack(
            [(0x1, 1, b"\x00\x00\x00", 36, 3)],
            payload_blob=b"\xff\xfe\xfd",
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(bad_utf8)
        self.assertEqual(ctx.exception.code, "wgpk_invalid_utf8")

    def test_cli_verify_pack_mode_valid_and_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pack_path = tmp_path / "verified.wgsl.pack"
            report_path = tmp_path / "verify_report.json"

            with redirect_stdout(io.StringIO()):
                rc_build = packer.main(
                    [
                        "--hashes",
                        "fragment:0x200,vertex:0x100",
                        "--out",
                        str(pack_path),
                    ]
                )
            self.assertEqual(rc_build, 0)

            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                rc_verify = packer.main(
                    [
                        "--verify-pack",
                        str(pack_path),
                        "--require-canonical-order",
                        "--json",
                        "--report-json",
                        str(report_path),
                    ]
                )
            self.assertEqual(rc_verify, 0)
            payload = json.loads(out_buf.getvalue())
            self.assertEqual(payload["mode"], "verify")
            self.assertTrue(payload["verified"])
            self.assertTrue(payload["canonical_hash_order"])
            self.assertEqual(payload["entry_count"], 2)
            self.assertEqual(payload["input_file"], "verified.wgsl.pack")
            self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), payload)

            # Corrupt one byte in the magic header -> verify fails with rc=1
            corrupt_path = tmp_path / "corrupt.wgsl.pack"
            corrupt_path.write_bytes(b"BAD!" + pack_path.read_bytes()[4:])
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc_bad = packer.main(["--verify-pack", str(corrupt_path)])
            self.assertEqual(rc_bad, 1)
            self.assertIn("invalid WGPK magic", err_buf.getvalue())

            # Mutually exclusive flags -> exit code 2
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc_conflict = packer.main(
                    ["--verify-pack", str(pack_path), "--out", str(tmp_path / "other.pack")]
                )
            self.assertEqual(rc_conflict, 2)


class TestDuplicateHashRejection(unittest.TestCase):
    """T91: Verify duplicate-hash rejection across verifier, serializer, builder, and CLI."""

    def test_verify_wgpk_rejects_duplicate_hashes_and_pairs(self) -> None:
        vs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.VERTEX]
        fs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.FRAGMENT]
        vs_bytes = vs.encode("utf-8")
        fs_bytes = fs.encode("utf-8")
        data_start = packer.PACK_HEADER_BYTES + 2 * packer.PACK_ENTRY_BYTES

        # Raw pack with exact duplicate (hash, stage) pair
        dup_pair_pack = (
            packer.PACK_HEADER_STRUCT.pack(packer.PACK_MAGIC, packer.PACK_VERSION, 2, 0)
            + packer.PACK_ENTRY_STRUCT.pack(0xABC, 1, b"\x00\x00\x00", data_start, len(vs_bytes))
            + packer.PACK_ENTRY_STRUCT.pack(
                0xABC, 1, b"\x00\x00\x00", data_start + len(vs_bytes), len(vs_bytes)
            )
            + vs_bytes
            + vs_bytes
        )
        for reject_flag in (True, False):
            with self.assertRaises(packer.PackerError) as ctx:
                packer.verify_wgpk_pack(dup_pair_pack, reject_duplicate_hashes=reject_flag)
            self.assertEqual(ctx.exception.code, "wgpk_duplicate_hash")

        # Raw pack with duplicate 64-bit hash across distinct stages
        dup_hash_diff_stage = (
            packer.PACK_HEADER_STRUCT.pack(packer.PACK_MAGIC, packer.PACK_VERSION, 2, 0)
            + packer.PACK_ENTRY_STRUCT.pack(0xABC, 1, b"\x00\x00\x00", data_start, len(vs_bytes))
            + packer.PACK_ENTRY_STRUCT.pack(
                0xABC, 2, b"\x00\x00\x00", data_start + len(vs_bytes), len(fs_bytes)
            )
            + vs_bytes
            + fs_bytes
        )
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(dup_hash_diff_stage, reject_duplicate_hashes=True)
        self.assertEqual(ctx.exception.code, "wgpk_duplicate_hash")

        # Legacy stage-conflict inspection still works when opted in
        summary = packer.verify_wgpk_pack(
            dup_hash_diff_stage,
            reject_duplicate_hashes=False,
            require_canonical_order=True,
        )
        self.assertEqual(summary.entry_count, 2)
        self.assertEqual(summary.unique_hash_count, 1)
        self.assertFalse(summary.canonical_hash_order)
        self.assertTrue(summary.canonical_pair_order)

    def test_serializer_and_builder_reject_duplicate_hashes(self) -> None:
        vs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.VERTEX]
        fs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.FRAGMENT]

        with self.assertRaises(packer.PackerError) as ctx:
            packer.serialize_wgpk(
                [(0x42, packer.ShaderStage.VERTEX, vs), (0x42, packer.ShaderStage.VERTEX, vs)]
            )
        self.assertEqual(ctx.exception.code, "duplicate_hash")

        with self.assertRaises(packer.PackerError) as ctx:
            packer.serialize_wgpk(
                [(0x42, packer.ShaderStage.VERTEX, vs), (0x42, packer.ShaderStage.FRAGMENT, fs)],
                reject_duplicate_hashes=True,
            )
        self.assertEqual(ctx.exception.code, "duplicate_hash")

        with self.assertRaises(packer.PackerError) as ctx:
            packer.sort_entries_canonical(
                [(0x42, packer.ShaderStage.VERTEX, vs), (0x42, packer.ShaderStage.FRAGMENT, fs)],
                reject_duplicate_hashes=True,
            )
        self.assertEqual(ctx.exception.code, "duplicate_hash")

        # build_fallback_pack with reject_duplicate_hashes=True rejects same-stage, cross-stage, and unknown dupes
        for bad_records in (
            [
                packer.ShaderInputRecord(0x99, packer.ShaderStage.VERTEX, "cli_hashes"),
                packer.ShaderInputRecord(0x99, packer.ShaderStage.VERTEX, "cli_hashes"),
            ],
            [
                packer.ShaderInputRecord(0x99, packer.ShaderStage.VERTEX, "cli_hashes"),
                packer.ShaderInputRecord(0x99, packer.ShaderStage.FRAGMENT, "cli_hashes"),
            ],
            [
                packer.ShaderInputRecord(0x99, packer.ShaderStage.VERTEX, "cli_hashes"),
                packer.ShaderInputRecord(0x99, None, "scan_json"),
            ],
        ):
            with self.assertRaises(packer.PackerError) as ctx:
                packer.build_fallback_pack(bad_records, reject_duplicate_hashes=True)
            self.assertEqual(ctx.exception.code, "duplicate_hash")

        # CLI --reject-duplicate-hashes fails without writing output
        with tempfile.TemporaryDirectory() as tmp:
            out_pack = Path(tmp) / "dup.wgsl.pack"
            err_buf = io.StringIO()
            with redirect_stderr(err_buf):
                rc = packer.main(
                    [
                        "--hashes",
                        "vertex:0x1234,fragment:0x1234",
                        "--reject-duplicate-hashes",
                        "--out",
                        str(out_pack),
                    ]
                )
            self.assertEqual(rc, 1)
            self.assertIn("duplicate shader hash", err_buf.getvalue())
            self.assertFalse(out_pack.exists())


class TestCanonicalHashSortingDeterminism(unittest.TestCase):
    """T91: Verify deterministic canonical ascending unsigned 64-bit hash sorting."""

    def test_unsigned_u64_canonical_order_and_permutation_determinism(self) -> None:
        vs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.VERTEX]
        fs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.FRAGMENT]
        cs = packer.BUILTIN_PROFILES["default"][packer.ShaderStage.COMPUTE]

        # Include unsigned 64-bit edge values (signed i64 would mis-order 0x8000... / 0xFFFF...)
        entries_a = [
            (0xFFFFFFFFFFFFFFFF, packer.ShaderStage.COMPUTE, cs),
            (0x8000000000000000, packer.ShaderStage.FRAGMENT, fs),
            (0x0000000000000000, packer.ShaderStage.VERTEX, vs),
            (0x7FFFFFFFFFFFFFFF, packer.ShaderStage.VERTEX, vs),
            (0x0000000000000001, packer.ShaderStage.FRAGMENT, fs),
        ]
        entries_b = list(reversed(entries_a))
        entries_c = [entries_a[3], entries_a[0], entries_a[4], entries_a[2], entries_a[1]]

        bytes_a, ranges_a = packer.serialize_wgpk(entries_a, canonical_sort=True, reject_duplicate_hashes=True)
        bytes_b, ranges_b = packer.serialize_wgpk(entries_b, canonical_sort=True, reject_duplicate_hashes=True)
        bytes_c, _ = packer.serialize_wgpk(entries_c, canonical_sort=True, reject_duplicate_hashes=True)

        self.assertEqual(bytes_a, bytes_b)
        self.assertEqual(bytes_a, bytes_c)

        # Returned ranges map accurately back to each input entry's WGSL payload
        for (expected_list, ranges, raw_pack) in ((entries_a, ranges_a, bytes_a), (entries_b, ranges_b, bytes_b)):
            for (_h, _st, wgsl), (off, length) in zip(expected_list, ranges, strict=True):
                self.assertEqual(raw_pack[off : off + length].decode("utf-8"), wgsl)

        summary = packer.verify_wgpk_pack(bytes_a, reject_duplicate_hashes=True, require_canonical_order=True)
        self.assertTrue(summary.canonical_hash_order)
        self.assertEqual(
            [e.shader_hash for e in summary.entries],
            [
                0x0000000000000000,
                0x0000000000000001,
                0x7FFFFFFFFFFFFFFF,
                0x8000000000000000,
                0xFFFFFFFFFFFFFFFF,
            ],
        )

        # Unsorted raw pack fails require_canonical_order=True
        unsorted_bytes, _ = packer.serialize_wgpk(entries_a, canonical_sort=False, reject_duplicate_hashes=True)
        unsorted_summary = packer.verify_wgpk_pack(unsorted_bytes, require_canonical_order=False)
        self.assertFalse(unsorted_summary.canonical_hash_order)
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_wgpk_pack(unsorted_bytes, require_canonical_order=True)
        self.assertEqual(ctx.exception.code, "wgpk_non_canonical_order")

        # Direct verify_canonical_hash_order checks
        packer.verify_canonical_hash_order([0, 1, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF])
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_canonical_hash_order([0x10, 0x10])
        self.assertEqual(ctx.exception.code, "wgpk_duplicate_hash")
        with self.assertRaises(packer.PackerError) as ctx:
            packer.verify_canonical_hash_order([0x20, 0x10])
        self.assertEqual(ctx.exception.code, "wgpk_non_canonical_order")


class TestEmptyInputGuards(unittest.TestCase):
    """T91: Verify empty-input guards and explicit --allow-empty escape hatch."""

    def test_empty_records_and_unknown_only_inputs_rejected_by_default(self) -> None:
        with self.assertRaises(packer.PackerError) as ctx:
            packer.build_fallback_pack([])
        self.assertEqual(ctx.exception.code, "empty_input")

        with self.assertRaises(packer.PackerError) as ctx:
            packer.build_fallback_pack([packer.ShaderInputRecord(0x1234, None, "scan_json")])
        self.assertEqual(ctx.exception.code, "empty_pack")

        # Explicit allow_empty=True produces a valid 16-byte WGPK v1 header
        empty_res = packer.build_fallback_pack([], allow_empty=True)
        self.assertEqual(len(empty_res.pack_bytes), packer.PACK_HEADER_BYTES)
        self.assertEqual(len(empty_res.entries), 0)
        assert empty_res.verification is not None
        self.assertTrue(empty_res.verification.canonical_hash_order)
        self.assertEqual(empty_res.verification.entry_count, 0)

    def test_cli_empty_scan_json_and_allow_empty_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            empty_scan = tmp_path / "empty_scan.json"
            empty_scan.write_text(json.dumps({"schema_version": 1, "files": []}), encoding="utf-8")
            unknown_only_scan = tmp_path / "unknown_only.json"
            unknown_only_scan.write_text(
                json.dumps({"shaders": [{"hash": "0xdeadbeef", "stage": "unknown"}]}),
                encoding="utf-8",
            )
            out_pack = tmp_path / "empty.wgsl.pack"

            # Empty scan JSON refused without --allow-empty
            err = io.StringIO()
            with redirect_stderr(err):
                rc = packer.main(["--scan-json", str(empty_scan), "--out", str(out_pack)])
            self.assertEqual(rc, 1)
            self.assertIn("no shader records", err.getvalue())
            self.assertFalse(out_pack.exists())

            # Unknown-only scan JSON refused without --allow-empty
            err = io.StringIO()
            with redirect_stderr(err):
                rc = packer.main(["--scan-json", str(unknown_only_scan), "--out", str(out_pack)])
            self.assertEqual(rc, 1)
            self.assertIn("no packable shader entries", err.getvalue())
            self.assertFalse(out_pack.exists())

            # Empty --hashes string rejected
            err = io.StringIO()
            with redirect_stderr(err):
                rc = packer.main(["--hashes", " , , ", "--out", str(out_pack)])
            self.assertEqual(rc, 1)

            # Missing --out (and no --verify-pack) returns exit code 2
            err = io.StringIO()
            with redirect_stderr(err):
                rc = packer.main(["--hashes", "vertex:0x1"])
            self.assertEqual(rc, 2)

            # Explicit --allow-empty succeeds and passes WGPK verification
            out_buf = io.StringIO()
            with redirect_stdout(out_buf):
                rc = packer.main(
                    ["--scan-json", str(empty_scan), "--allow-empty", "--out", str(out_pack), "--json"]
                )
            self.assertEqual(rc, 0)
            rep = json.loads(out_buf.getvalue())
            self.assertEqual(rep["summary"]["packed_entry_count"], 0)
            self.assertTrue(rep["verification"]["verified"])
            self.assertEqual(len(out_pack.read_bytes()), packer.PACK_HEADER_BYTES)


class TestHostPathJsonSanitization(unittest.TestCase):
    """T91: Verify host paths, file URIs, tilde paths, usernames, and raw bytes never leak in JSON."""

    def test_render_json_summary_scrubs_all_host_path_variants_and_usernames(self) -> None:
        fake_user = "hidden_builder_99"
        env_patch = {
            "USERNAME": f"CORP\\{fake_user}",
            "USER": fake_user,
            "HOME": f"/home/{fake_user}",
            "USERPROFILE": f"C:\\Users\\{fake_user}",
        }
        raw_payload = {
            "win_path": f"C:\\Users\\{fake_user}\\Private\\game.bin",
            "unc_path": f"\\\\fileserver\\share\\{fake_user}\\pack.wgsl.pack",
            "posix_path": f"/Users/{fake_user}/workspace/pack.wgsl.pack",
            "cygwin_path": f"/cygdrive/c/Users/{fake_user}/shader.bin",
            "file_uri_posix": f"file:///home/{fake_user}/pack.wgsl.pack",
            "file_uri_win": f"file:///C:/Users/{fake_user}/pack.wgsl.pack",
            "tilde_path": f"~/Documents/{fake_user}/pack.wgsl.pack",
            "path_obj": Path(f"/tmp/{fake_user}/nested/out_{fake_user}.wgsl.pack"),
            "raw_bytes": b"WGPK\x01\x00\x00\x00",
            f"key_with_{fake_user}": [f"owned by {fake_user}"],
        }

        with mock.patch.dict(os.environ, env_patch):
            rendered = packer.render_json_summary(raw_payload)

        parsed = json.loads(rendered)
        self.assertNotIn(fake_user, rendered)
        self.assertNotIn("CORP", rendered)
        self.assertNotIn("C:", rendered)
        self.assertNotIn("fileserver", rendered)
        self.assertNotIn("/Users/", rendered)
        self.assertNotIn("/home/", rendered)
        self.assertNotIn("/cygdrive/", rendered)
        self.assertNotIn("file://", rendered)
        self.assertNotIn("~/Documents", rendered)
        self.assertNotIn("WGPK", rendered)
        self.assertEqual(parsed["raw_bytes"], "<bytes:8>")
        self.assertEqual(parsed["path_obj"], "out_<user>.wgsl.pack")
        self.assertIn("key_with_<user>", parsed)

    def test_cli_json_and_verify_json_include_sanitized_verification_summary(self) -> None:
        fake_user = "ci_secret_runner"
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            nested = tmp_path / f"dir_{fake_user}"
            nested.mkdir()
            out_pack = nested / f"bundle_{fake_user}.wgsl.pack"

            pack_stdout = io.StringIO()
            verify_stdout = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"USERNAME": fake_user, "USER": fake_user, "HOME": f"/home/{fake_user}"},
            ):
                with redirect_stdout(pack_stdout):
                    rc_pack = packer.main(
                        [
                            "--hashes",
                            "compute:0x300,vertex:0x100,fragment:0x200",
                            "--reject-duplicate-hashes",
                            "--out",
                            str(out_pack),
                            "--json",
                        ]
                    )
                with redirect_stdout(verify_stdout):
                    rc_verify = packer.main(
                        [
                            "--verify-pack",
                            str(out_pack),
                            "--require-canonical-order",
                            "--json",
                        ]
                    )

            self.assertEqual(rc_pack, 0)
            self.assertEqual(rc_verify, 0)

            for text in (pack_stdout.getvalue(), verify_stdout.getvalue()):
                self.assertNotIn(fake_user, text)
                self.assertNotIn(str(tmp_path), text)
                self.assertNotRegex(text, r"[A-Za-z]:[\\/]")
                self.assertNotRegex(text, r"/(?:Users|home|tmp|var|private)/")
                self.assertNotIn("@vertex", text)

            pack_json = json.loads(pack_stdout.getvalue())
            verify_json = json.loads(verify_stdout.getvalue())
            self.assertTrue(pack_json["summary"]["canonical_hash_order"])
            self.assertTrue(pack_json["verification"]["verified"])
            self.assertEqual(pack_json["verification"]["pack_sha256"], verify_json["pack_sha256"])
            self.assertEqual(pack_json["output_file"], "bundle_<user>.wgsl.pack")
            self.assertEqual(verify_json["input_file"], "bundle_<user>.wgsl.pack")


if __name__ == "__main__":
    unittest.main()
