#!/usr/bin/env python3
"""Unit, CLI, sanitization, and end-to-end tests for tools/diff_missing_shaders.py (T99 / #1673)."""

from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import diff_missing_shaders as dms  # noqa: E402
import inspect_wgsl_pack as wgpk  # noqa: E402
import merge_wgsl_packs as mp  # noqa: E402


SYNTHETIC_VS = "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"
SYNTHETIC_PS = "@fragment fn ps_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"
SYNTHETIC_CS = "@compute @workgroup_size(1, 1, 1) fn cs_main() {}"


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        code = dms.main(argv)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


def _write_pack(
    directory: Path,
    name: str,
    entries: list[tuple[int, str, str]],
    *,
    version: int | None = None,
    reflect_stages: dict[str, dict] | None = None,
    reflect_version: int = 3,
) -> Path:
    pack_path = directory / name
    pack_path.write_bytes(wgpk.build_wgpk(entries, version=version))
    if reflect_stages is not None:
        sidecar = {
            "schema": "playgta6-stage-reflection",
            "version": reflect_version,
            "stages": reflect_stages,
        }
        (directory / (name + ".reflect.json")).write_text(
            json.dumps(sidecar, indent=1),
            encoding="utf-8",
        )
    return pack_path


def _stage_entry(stage: str, n_resolved: int = 1, n_unresolved: int = 0) -> dict:
    bindings = [
        {"group": 0, "binding": i, "kind": "uniform", "source": {"user_sgpr": 8, "table_byte_offset": i * 16}}
        for i in range(n_resolved)
    ]
    bindings += [
        {"group": 0, "binding": 16 + i, "kind": "uniform"}
        for i in range(n_unresolved)
    ]
    return {"stage": stage, "bindings": bindings}


class TestBootLogParser(unittest.TestCase):
    """Test Chrome t99/t101 and native boot console log parsing."""

    def test_parses_chrome_t99_t101_patterns_and_ignores_timestamps(self) -> None:
        log_text = "\n".join([
            "# Priority / boot log header comment",
            "[1712345678901234] pid=1234567890123456 [wgpu] frame=12 swapchain ok",
            "[1712345678901299] [wgpu] missing shader pipeline: VS 0xac2db5ab29dcc4e1 PS 0x47ee3e08b61101df",
            "[1712345678901350] [wgpu] shader miss: vs=0xac2db5ab29dcc4e1",
            "[1712345678901400] [wgpu] missing shader: ps: 0x47ee3e08b61101df",
            "[1712345678901450] [wgpu] dispatch CS 0x0011223344556677 failed",
            "[1712345678901500] [wgpu] fallback FS 0x8899aabbccddeeff",
            "[1712345678901550] [wgpu] missing vertex 0x1000000000000001 and pixel 0x1000000000000002",
            "[1712345678901600] [wgpu] fragment 0x1000000000000003 compute 0x1000000000000004",
            "[1712345678901650] [wgpu] missing shader 0xac2db5ab29dcc4e1",  # unknown stage must not overwrite vertex
            "[1712345678901700] [wgpu] shader miss deadbeefcafe0001",  # bare 16-hex with hex letters
            "0x2000000000000001 # standalone with 0x",
            "200000000000000a # standalone bare hex",
            # Must be ignored: short hex, 17-digit hex, pure decimal timestamp in shader line
            "[1712345678901800] [wgpu] shader addr=0xdeadbeef too_long=0x1234567890abcdef0 ts=1712345678901999",
        ])

        acc = dms.parse_boot_log_text(log_text, source_label="chrome_t99.log")
        by_hash = {item.hash: item for item in acc.items()}

        self.assertEqual(by_hash[0xAC2DB5AB29DCC4E1].stage, "vertex")
        self.assertEqual(by_hash[0xAC2DB5AB29DCC4E1].count, 3)
        self.assertEqual(by_hash[0xAC2DB5AB29DCC4E1].sources, ("chrome_t99.log",))

        self.assertEqual(by_hash[0x47EE3E08B61101DF].stage, "fragment")
        self.assertEqual(by_hash[0x47EE3E08B61101DF].count, 2)

        self.assertEqual(by_hash[0x0011223344556677].stage, "compute")
        self.assertEqual(by_hash[0x0011223344556677].count, 1)

        self.assertEqual(by_hash[0x8899AABBCCDDEEFF].stage, "fragment")
        self.assertEqual(by_hash[0x1000000000000001].stage, "vertex")
        self.assertEqual(by_hash[0x1000000000000002].stage, "fragment")
        self.assertEqual(by_hash[0x1000000000000003].stage, "fragment")
        self.assertEqual(by_hash[0x1000000000000004].stage, "compute")
        self.assertEqual(by_hash[0xDEADBEEFCAFE0001].stage, "unknown")
        self.assertEqual(by_hash[0x2000000000000001].stage, "unknown")
        self.assertEqual(by_hash[0x200000000000000A].stage, "unknown")

        # Decimal timestamps and non-16-hex numbers must never appear.
        self.assertEqual(len(by_hash), 11)
        self.assertNotIn(0xDEADBEEF, by_hash)
        self.assertNotIn(int("1712345678901234", 16), by_hash)
        self.assertNotIn(int("1712345678901999", 16), by_hash)
        self.assertNotIn(int("1234567890123456", 16), by_hash)

    def test_known_stage_beats_unknown_across_lines(self) -> None:
        log_text = "\n".join([
            "[wgpu] missing shader 0xac2db5ab29dcc4e1",
            "[wgpu] pipeline fallback VS 0xac2db5ab29dcc4e1",
            "[wgpu] missing shader 0x47ee3e08b61101df (stage=pixel)",
        ])
        acc = dms.parse_boot_log_text(log_text, source_label="t101.log")
        by_hash = {item.hash: item for item in acc.items()}
        self.assertEqual(by_hash[0xAC2DB5AB29DCC4E1].stage, "vertex")
        self.assertEqual(by_hash[0xAC2DB5AB29DCC4E1].count, 2)
        self.assertEqual(by_hash[0x47EE3E08B61101DF].stage, "fragment")
        self.assertEqual(by_hash[0x47EE3E08B61101DF].count, 1)


class TestMultiSourceDiffAndVersions(unittest.TestCase):
    """Test WGPK v1/v2, reflect v1/v2/v3, missing-json, crash-report, pipeline-dump, and strict CLI."""

    def test_combined_sources_wgpk_v1_and_v2_reflect_v1_v2_v3(self) -> None:
        h_in_both = 0x1000000000000001
        h_in_pack_only = 0x1000000000000002
        h_missing_vs = 0xAC2DB5AB29DCC4E1
        h_missing_ps = 0x47EE3E08B61101DF
        h_missing_cs = 0x3000000000000003
        h_dump_bin_only = 0x4000000000000004

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Test across WGPK v1 and v2, and reflect versions 1, 2, and 3.
            for pack_ver, reflect_ver in ((1, 1), (2, 2), (2, 3)):
                sub = root / f"case_p{pack_ver}_r{reflect_ver}"
                sub.mkdir()
                pack_path = _write_pack(
                    sub,
                    "active.wgsl.pack",
                    [
                        (h_in_both, "vertex", SYNTHETIC_VS),
                        (h_in_pack_only, "fragment", SYNTHETIC_PS),
                    ],
                    version=pack_ver,
                    reflect_stages={
                        dms.format_shader_hash(h_in_both): _stage_entry("vertex", 1),
                    },
                    reflect_version=reflect_ver,
                )

                # Source 1: Chrome boot log
                log_path = sub / "chrome_t99.log"
                log_path.write_text(
                    f"[wgpu] miss VS {dms.format_shader_hash(h_missing_vs)}\n"
                    f"[wgpu] miss PS {dms.format_shader_hash(h_missing_ps)}\n"
                    f"[wgpu] hit  VS {dms.format_shader_hash(h_in_both)}\n",
                    encoding="utf-8",
                )

                # Source 2: Browser playgta6-missing-shaders-v1 JSON + crash report JSON
                missing_json = sub / "missing_panel.json"
                missing_json.write_text(
                    json.dumps({
                        "format": "playgta6-missing-shaders-v1",
                        "schemaVersion": 1,
                        "missingShaders": [
                            {"hash": dms.format_shader_hash(h_missing_vs), "stage": "vs", "count": 4},
                            {"hash": dms.format_shader_hash(h_missing_cs), "stage": "compute", "misses": 2},
                        ],
                    }),
                    encoding="utf-8",
                )

                crash_json = sub / "crash_report.json"
                crash_json.write_text(
                    json.dumps({
                        "schema": "playgta6-crash-report",
                        "gpuTelemetry": {
                            "missingShaders": [
                                {"hash": dms.format_shader_hash(h_missing_ps), "stage": "pixel", "hits": 3},
                                {"hash": dms.format_shader_hash(h_in_pack_only), "stage": "fragment", "count": 1},
                            ]
                        },
                    }),
                    encoding="utf-8",
                )

                # Source 3: playgta6-pipeline-dump v1 dir (with invalid binary bytes that must never be read)
                dump_dir = sub / "pipeline_dump"
                (dump_dir / "native").mkdir(parents=True)
                (dump_dir / "dxil").mkdir(parents=True)
                (dump_dir / "manifest.json").write_text(
                    json.dumps({
                        "schema": "playgta6-pipeline-dump",
                        "schema_version": 1,
                        "native_stages": [
                            {"code_identity": dms.format_shader_hash(h_in_both), "stage": "vertex"},
                            {"code_identity": dms.format_shader_hash(h_in_both), "stage": "vertex"},
                        ],
                        "dxil": [
                            {"identity": dms.format_shader_hash(h_missing_cs), "kind": 5},
                        ],
                    }),
                    encoding="utf-8",
                )
                (dump_dir / "native" / f"{dms.format_shader_hash(h_dump_bin_only)}.bin").write_bytes(b"\xFF\xFE\x00\x01")
                (dump_dir / "dxil" / f"ps_{dms.format_shader_hash(h_missing_ps)}.dxbc").write_bytes(b"\xFF\xFE")

                prio_out = sub / "out" / "priority.txt"

                # Test shorthand CLI (without explicit `diff` subcommand)
                code, out, err = _run_cli([
                    "--pack", str(pack_path),
                    "--boot-log", str(log_path),
                    "--missing-json", str(missing_json),
                    "--missing-json", str(crash_json),
                    "--dump-dir", str(dump_dir),
                    "--emit-priority-hashes", str(prio_out),
                    "--json",
                ])
                self.assertEqual(code, 0, msg=err)
                payload = json.loads(out)
                self.assertTrue(payload["ok"])
                self.assertFalse(payload["clean"])
                self.assertEqual(payload["schema"], "playgta6-missing-shader-diff")
                self.assertEqual(payload["version"], 1)
                self.assertEqual(payload["pack"]["version"], pack_ver)
                self.assertTrue(payload["reflect"]["loaded"])
                self.assertEqual(payload["reflect"]["version"], reflect_ver)

                summary = payload["summary"]
                self.assertEqual(summary["demanded_unique"], 6)
                self.assertEqual(summary["in_pack"], 2)
                self.assertEqual(summary["missing_from_pack"], 4)
                self.assertEqual(summary["in_reflect"], 1)
                self.assertEqual(summary["missing_from_reflect"], 5)
                self.assertEqual(summary["in_pack_missing_from_reflect"], 1)

                missing_map = {m["hash"]: m for m in payload["missing_from_pack"]}
                self.assertEqual(missing_map[dms.format_shader_hash(h_missing_vs)]["stage"], "vertex")
                self.assertEqual(missing_map[dms.format_shader_hash(h_missing_vs)]["count"], 5)
                self.assertEqual(
                    missing_map[dms.format_shader_hash(h_missing_vs)]["sources"],
                    ["chrome_t99.log", "missing_panel.json"],
                )
                self.assertEqual(missing_map[dms.format_shader_hash(h_missing_ps)]["stage"], "fragment")
                self.assertEqual(missing_map[dms.format_shader_hash(h_missing_ps)]["count"], 5)
                self.assertEqual(missing_map[dms.format_shader_hash(h_missing_cs)]["stage"], "compute")
                self.assertEqual(missing_map[dms.format_shader_hash(h_missing_cs)]["count"], 3)
                self.assertEqual(missing_map[dms.format_shader_hash(h_dump_bin_only)]["stage"], "unknown")
                self.assertEqual(missing_map[dms.format_shader_hash(h_dump_bin_only)]["count"], 1)

                # Priority hashes must be deduplicated and canonically sorted ascending.
                emitted_lines = prio_out.read_text(encoding="utf-8").splitlines()
                self.assertEqual(
                    emitted_lines,
                    sorted([
                        dms.format_shader_hash(h_missing_vs),
                        dms.format_shader_hash(h_missing_ps),
                        dms.format_shader_hash(h_missing_cs),
                        dms.format_shader_hash(h_dump_bin_only),
                    ]),
                )

                # --strict must exit 1 when shaders are missing.
                strict_code, strict_out, _ = _run_cli([
                    "diff",
                    "--pack", str(pack_path),
                    "--boot-log", str(log_path),
                    "--strict",
                    "--json",
                ])
                self.assertEqual(strict_code, 1)
                strict_payload = json.loads(strict_out)
                self.assertFalse(strict_payload["ok"])
                self.assertTrue(strict_payload["strict"])

    def test_strict_fails_when_in_pack_but_missing_from_loaded_reflect(self) -> None:
        h1 = 0x00000000000000A1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pack_path = _write_pack(
                root,
                "p.wgsl.pack",
                [(h1, "vertex", SYNTHETIC_VS)],
                reflect_stages={},
                reflect_version=2,
            )
            log_path = root / "boot.log"
            log_path.write_text(f"VS {dms.format_shader_hash(h1)}\n", encoding="utf-8")

            code, out, _ = _run_cli([
                "diff", "--pack", str(pack_path), "--boot-log", str(log_path), "--strict", "--json"
            ])
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertEqual(payload["summary"]["missing_from_pack"], 0)
            self.assertEqual(payload["summary"]["missing_from_reflect"], 1)
            self.assertEqual(payload["summary"]["in_pack_missing_from_reflect"], 1)
            self.assertFalse(payload["clean"])

    def test_corrupt_inputs_fail_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good_pack = _write_pack(root, "good.wgsl.pack", [(1, "vertex", SYNTHETIC_VS)])
            good_log = root / "boot.log"
            good_log.write_text("VS 0x0000000000000001\n", encoding="utf-8")

            # Corrupt pack magic
            bad_pack = root / "bad.wgsl.pack"
            bad_pack.write_bytes(b"NOPE0123456789abcdef")
            code, out, _ = _run_cli(["diff", "--pack", str(bad_pack), "--boot-log", str(good_log), "--json"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_magic")

            # Corrupt manifest schema
            bad_dump = root / "bad_dump"
            bad_dump.mkdir()
            (bad_dump / "manifest.json").write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
            code, out, _ = _run_cli(["diff", "--pack", str(good_pack), "--dump-dir", str(bad_dump), "--json"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_manifest")

            # Unsupported reflect version (4)
            bad_reflect = root / "bad.reflect.json"
            bad_reflect.write_text(
                json.dumps({"schema": "playgta6-stage-reflection", "version": 4, "stages": {}}),
                encoding="utf-8",
            )
            code, out, _ = _run_cli([
                "diff", "--pack", str(good_pack), "--reflect", str(bad_reflect), "--boot-log", str(good_log), "--json"
            ])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_reflect_sidecar")

            # Corrupt missing JSON
            bad_json = root / "bad_missing.json"
            bad_json.write_text("{broken json", encoding="utf-8")
            code, out, _ = _run_cli(["diff", "--pack", str(good_pack), "--missing-json", str(bad_json), "--json"])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["errorCode"], "invalid_missing_json")


class TestSanitizationInvariant(unittest.TestCase):
    """Verify host paths, drive letters, UNC paths, secret usernames, and WGSL never leak."""

    def test_sanitize_source_label_handles_windows_posix_and_unc(self) -> None:
        self.assertEqual(
            dms.sanitize_source_label(r"C:\Users\secret_user_t99\games\pack.wgsl.pack"),
            "pack.wgsl.pack",
        )
        self.assertEqual(
            dms.sanitize_source_label("/home/secret_user_t99/work/chrome.log"),
            "chrome.log",
        )
        self.assertEqual(
            dms.sanitize_source_label(r"\\fileserver\private_share\delta_v2.wgsl.pack"),
            "delta_v2.wgsl.pack",
        )
        self.assertEqual(dms.sanitize_source_label("-"), "<stdin>")
        self.assertEqual(dms.sanitize_source_label(None), "<memory>")
        self.assertEqual(dms.sanitize_source_label("C:\\"), "<unknown>")

    def test_no_host_path_or_wgsl_leak_in_success_and_error_outputs(self) -> None:
        secret_user = "secret_t99_host_user_9981"
        secret_wgsl_marker = "SECRET_WGSL_IDENTIFIER_DO_NOT_LEAK"
        wgsl_with_marker = f"@vertex fn {secret_wgsl_marker}() -> @builtin(position) vec4<f32> {{ return vec4<f32>(0.0); }}"

        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / secret_user / "private_builds"
            nested.mkdir(parents=True)
            pack_path = _write_pack(
                nested,
                "active_v2.wgsl.pack",
                [(0x1111111111111111, "vertex", wgsl_with_marker)],
                version=2,
                reflect_stages={"0x1111111111111111": _stage_entry("vertex", 1)},
            )
            log_path = nested / "chrome_run.log"
            log_path.write_text("VS 0xac2db5ab29dcc4e1\nVS 0x1111111111111111\n", encoding="utf-8")
            missing_file = nested / "does_not_exist.wgsl.pack"

            # 1. Diff JSON & text success paths
            for extra in (["--json"], []):
                code, out, err = _run_cli(["diff", "--pack", str(pack_path), "--boot-log", str(log_path), *extra])
                self.assertEqual(code, 0)
                combined = out + err
                self.assertNotIn(secret_user, combined)
                self.assertNotIn("private_builds", combined)
                self.assertNotIn(tmp, combined)
                self.assertNotIn(secret_wgsl_marker, combined)

            # 2. Error paths (missing pack, missing log, verify-merge missing file)
            for argv in (
                ["diff", "--pack", str(missing_file), "--boot-log", str(log_path), "--json"],
                ["diff", "--pack", str(missing_file), "--boot-log", str(log_path)],
                ["diff", "--pack", str(pack_path), "--boot-log", str(nested / "nope.log"), "--json"],
                ["verify-merge", "--inputs", str(missing_file), "--merged", str(pack_path), "--json"],
                ["verify-merge", "--inputs", str(missing_file), "--merged", str(pack_path)],
            ):
                code, out, err = _run_cli(argv)
                self.assertEqual(code, 1)
                combined = out + err
                self.assertNotIn(secret_user, combined, msg=f"leaked in {argv}")
                self.assertNotIn("private_builds", combined, msg=f"leaked in {argv}")
                self.assertNotIn(tmp, combined, msg=f"leaked in {argv}")
                self.assertNotIn(secret_wgsl_marker, combined, msg=f"leaked in {argv}")


class TestEndToEndIncrementalWgpkV2Workflow(unittest.TestCase):
    """End-to-end: diff missing -> build delta -> merge --force-v2 --verify -> verify-merge -> re-diff clean."""

    def test_incremental_wgpk_v2_workflow_and_tamper_detection(self) -> None:
        h_base_vs = 0x0000000000000101
        h_shared_ps = 0x0000000000000202
        h_new_vs = 0xAC2DB5AB29DCC4E1
        h_new_ps = 0x47EE3E08B61101DF

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            # Step 1: Base WGPK v1 pack with v2 sidecar (h_shared_ps has 0 resolved bindings).
            base_pack = _write_pack(
                root,
                "base.wgsl.pack",
                [
                    (h_base_vs, "vertex", SYNTHETIC_VS),
                    (h_shared_ps, "fragment", "// old base ps\n" + SYNTHETIC_PS),
                ],
                version=1,
                reflect_stages={
                    dms.format_shader_hash(h_base_vs): _stage_entry("vertex", 1),
                    dms.format_shader_hash(h_shared_ps): _stage_entry("fragment", 0, 1),
                },
                reflect_version=2,
            )

            boot_log = root / "chrome_t101.log"
            boot_log.write_text(
                "\n".join([
                    f"[wgpu] hit  VS {dms.format_shader_hash(h_base_vs)}",
                    f"[wgpu] miss VS {dms.format_shader_hash(h_new_vs)}",
                    f"[wgpu] miss PS {dms.format_shader_hash(h_new_ps)}",
                ]),
                encoding="utf-8",
            )

            # Step 2: Diff missing shaders against base pack & emit priority hashes.
            prio_file = root / "priority_missing.txt"
            code, out, err = _run_cli([
                "diff",
                "--pack", str(base_pack),
                "--boot-log", str(boot_log),
                "--emit-priority-hashes", str(prio_file),
                "--json",
            ])
            self.assertEqual(code, 0, msg=err)
            diff1 = json.loads(out)
            self.assertEqual(diff1["summary"]["missing_from_pack"], 2)
            self.assertEqual(
                prio_file.read_text(encoding="utf-8").splitlines(),
                [dms.format_shader_hash(h_new_ps), dms.format_shader_hash(h_new_vs)],
            )

            # Step 3: Build delta WGPK v2 pack with v3 sidecar (h_shared_ps now has 2 resolved bindings -> wins tie!).
            delta_pack = _write_pack(
                root,
                "delta.wgsl.pack",
                [
                    (h_shared_ps, "fragment", "// improved delta ps\n" + SYNTHETIC_PS),
                    (h_new_vs, "vertex", SYNTHETIC_VS),
                    (h_new_ps, "fragment", SYNTHETIC_PS),
                ],
                version=2,
                reflect_stages={
                    dms.format_shader_hash(h_shared_ps): _stage_entry("fragment", 2),
                    dms.format_shader_hash(h_new_vs): _stage_entry("vertex", 1),
                    dms.format_shader_hash(h_new_ps): _stage_entry("fragment", 1),
                },
                reflect_version=3,
            )

            # Step 4: Merge with --force-v2 --verify via merge_wgsl_packs.py.
            merged_pack = root / "merged_v2.wgsl.pack"
            merge_stdout = io.StringIO()
            with contextlib.redirect_stdout(merge_stdout):
                mrc = mp.main([
                    str(base_pack),
                    str(delta_pack),
                    "--out", str(merged_pack),
                    "--force-v2",
                    "--verify",
                    "--json",
                ])
            self.assertEqual(mrc, 0)
            merge_summary = json.loads(merge_stdout.getvalue())
            self.assertEqual(merge_summary["pack_version"], 2)
            self.assertEqual(merge_summary["merged_entries"], 4)
            self.assertTrue(merge_summary["verified"])

            # Step 5: Verify merge integrity with diff_missing_shaders.py verify-merge (JSON & text).
            vcode, vout, verr = _run_cli([
                "verify-merge",
                "--inputs", str(base_pack), str(delta_pack),
                "--merged", str(merged_pack),
                "--json",
            ])
            self.assertEqual(vcode, 0, msg=verr)
            vreport = json.loads(vout)
            self.assertTrue(vreport["ok"])
            self.assertTrue(vreport["passed"])
            self.assertEqual(vreport["schema"], "playgta6-wgpk-merge-verify")
            self.assertEqual(vreport["merged"]["version"], 2)
            self.assertEqual(vreport["payloads_verified"], 4)
            self.assertEqual(vreport["sidecar"]["version"], 3)
            self.assertEqual(vreport["violation_count"], 0)

            vtext_code, vtext, _ = _run_cli([
                "verify-merge",
                "--inputs", str(base_pack), str(delta_pack),
                "--merged", str(merged_pack),
            ])
            self.assertEqual(vtext_code, 0)
            self.assertIn("[PASS]", vtext)

            # Step 6: Re-diff against merged v2 pack -> 0 missing, --strict exits 0!
            code2, out2, err2 = _run_cli([
                "diff",
                "--pack", str(merged_pack),
                "--boot-log", str(boot_log),
                "--strict",
                "--json",
            ])
            self.assertEqual(code2, 0, msg=err2)
            diff2 = json.loads(out2)
            self.assertTrue(diff2["ok"])
            self.assertTrue(diff2["clean"])
            self.assertEqual(diff2["summary"]["missing_from_pack"], 0)
            self.assertEqual(diff2["summary"]["missing_from_reflect"], 0)
            self.assertEqual(diff2["summary"]["pack_coverage_pct"], 100.0)
            self.assertEqual(diff2["summary"]["reflect_coverage_pct"], 100.0)

            # Step 7: Tamper detection checks in verify-merge!
            orig_bytes = merged_pack.read_bytes()
            orig_sidecar = (root / "merged_v2.wgsl.pack.reflect.json").read_text(encoding="utf-8")

            # 7a. Tamper 1 byte of payload (valid UTF-8 change -> SHA-256 mismatch)
            tampered_payload = bytearray(orig_bytes)
            tampered_payload[-2] = ord("9") if tampered_payload[-2] != ord("9") else ord("8")
            tampered_p = root / "tampered_payload.wgsl.pack"
            tampered_p.write_bytes(bytes(tampered_payload))
            (root / "tampered_payload.wgsl.pack.reflect.json").write_text(orig_sidecar, encoding="utf-8")
            tc, tout, _ = _run_cli([
                "verify-merge", "--inputs", str(base_pack), str(delta_pack), "--merged", str(tampered_p), "--json"
            ])
            self.assertEqual(tc, 1)
            codes = {v["code"] for v in json.loads(tout)["violations"]}
            self.assertIn("payload_mismatch", codes)

            # 7b. Missing hash in merged pack
            missing_hash_pack = _write_pack(
                root,
                "missing_hash.wgsl.pack",
                [
                    (h_base_vs, "vertex", SYNTHETIC_VS),
                    (h_shared_ps, "fragment", "// improved delta ps\n" + SYNTHETIC_PS),
                    (h_new_ps, "fragment", SYNTHETIC_PS),
                ],
                version=2,
            )
            (root / "missing_hash.wgsl.pack.reflect.json").write_text(orig_sidecar, encoding="utf-8")
            mc, mout, _ = _run_cli([
                "verify-merge", "--inputs", str(base_pack), str(delta_pack), "--merged", str(missing_hash_pack), "--json"
            ])
            self.assertEqual(mc, 1)
            mcodes = {v["code"] for v in json.loads(mout)["violations"]}
            self.assertIn("missing_hash", mcodes)
            self.assertIn("dangling_sidecar_hash", mcodes)

            # 7c. Dangling sidecar hash on an otherwise valid merged pack
            dangling_pack = root / "dangling_side.wgsl.pack"
            dangling_pack.write_bytes(orig_bytes)
            side_doc = json.loads(orig_sidecar)
            side_doc["stages"]["0xffffffffffffffff"] = _stage_entry("compute", 1)
            (root / "dangling_side.wgsl.pack.reflect.json").write_text(json.dumps(side_doc), encoding="utf-8")
            dc, dout, _ = _run_cli([
                "verify-merge", "--inputs", str(base_pack), str(delta_pack), "--merged", str(dangling_pack), "--json"
            ])
            self.assertEqual(dc, 1)
            dcodes = {v["code"] for v in json.loads(dout)["violations"]}
            self.assertIn("dangling_sidecar_hash", dcodes)

            # 7d. Non-ascending hash order in TOC
            unsorted_pack = _write_pack(
                root,
                "unsorted.wgsl.pack",
                [
                    (h_new_vs, "vertex", SYNTHETIC_VS),
                    (h_base_vs, "vertex", SYNTHETIC_VS),
                    (h_shared_ps, "fragment", "// improved delta ps\n" + SYNTHETIC_PS),
                    (h_new_ps, "fragment", SYNTHETIC_PS),
                ],
                version=2,
            )
            (root / "unsorted.wgsl.pack.reflect.json").write_text(orig_sidecar, encoding="utf-8")
            uc, uout, _ = _run_cli([
                "verify-merge", "--inputs", str(base_pack), str(delta_pack), "--merged", str(unsorted_pack), "--json"
            ])
            self.assertEqual(uc, 1)
            ucodes = {v["code"] for v in json.loads(uout)["violations"]}
            self.assertIn("hash_order_not_ascending", ucodes)


if __name__ == "__main__":
    unittest.main()
