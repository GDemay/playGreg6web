#!/usr/bin/env python3
"""Unit tests for tools/gen_fallback_shaders.py (Track 4 task S9)."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import gen_fallback_shaders as gen  # noqa: E402
from tools import inspect_wgsl_pack as wgpk  # noqa: E402


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        code = gen.main(argv)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


class TestFallbackCatalog(unittest.TestCase):
    def test_builtin_catalog_has_expected_stages_and_unique_hashes(self) -> None:
        shaders = gen.builtin_fallback_shaders()
        self.assertEqual(len(shaders), 7)
        stages = [s.stage for s in shaders]
        self.assertEqual(stages.count("vertex"), 3)
        self.assertEqual(stages.count("fragment"), 3)
        self.assertEqual(stages.count("compute"), 1)

        hashes = [s.hash for s in shaders]
        self.assertEqual(len(hashes), len(set(hashes)))
        for shader in shaders:
            self.assertEqual(shader.hash, wgpk.fnv1a_64(shader.wgsl))
            self.assertTrue(shader.entry_point)


class TestPackAndReflection(unittest.TestCase):
    def test_generated_pack_passes_inspect_strict_with_expected_hashes(self) -> None:
        artifacts = gen.build_fallback_artifacts()
        report = wgpk.inspect_pack(artifacts.pack_bytes, source_label="fallback.wgsl.pack")
        self.assertTrue(report.clean, msg=[v.to_dict() for v in report.violations])
        self.assertEqual(report.version, 1)
        self.assertEqual(report.flags, 0)
        self.assertEqual(report.entry_count, 7)
        self.assertEqual(report.stage_counts["vertex"], 3)
        self.assertEqual(report.stage_counts["fragment"], 3)
        self.assertEqual(report.stage_counts["compute"], 1)

        expected_hashes = {shader.hash for shader in artifacts.shaders}
        actual_hashes = {entry.hash for entry in wgpk.parse_wgpk(artifacts.pack_bytes).entries}
        self.assertEqual(actual_hashes, expected_hashes)

        # CLI inspect --strict path (0 errors).
        with tempfile.TemporaryDirectory() as tmp:
            pack_path = Path(tmp) / "fallback.wgsl.pack"
            pack_path.write_bytes(artifacts.pack_bytes)
            code, stdout, stderr = _run_inspect(["inspect", str(pack_path), "--strict", "--json"])
            self.assertEqual(code, 0, msg=stderr or stdout)
            payload = json.loads(stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["clean"])
            self.assertEqual(payload["violationCount"], 0)
            self.assertEqual(payload["entryCount"], 7)

    def test_reflection_manifest_matches_every_pack_entry(self) -> None:
        artifacts = gen.build_fallback_artifacts()
        manifest = artifacts.reflection_manifest
        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("register_convention", manifest)
        self.assertIn("bind_group_layouts", manifest)
        self.assertIn("pipeline_layouts", manifest)
        self.assertIn("shaders", manifest)
        self.assertIn("pipelines", manifest)

        pack = wgpk.parse_wgpk(artifacts.pack_bytes)
        pack_hashes = {format_hash(entry.hash) for entry in pack.entries}
        manifest_hashes = set(manifest["shaders"].keys())
        self.assertEqual(manifest_hashes, pack_hashes)

        for entry in pack.entries:
            key = format_hash(entry.hash)
            record = manifest["shaders"][key]
            self.assertEqual(record["stage"], entry.stage_name)
            self.assertIsInstance(record["entry_point"], str)
            self.assertTrue(record["entry_point"])
            self.assertTrue(record["layout_complete"])
            self.assertIn(record["pipeline_layout"], manifest["pipeline_layouts"])
            for bgl_id in record["bind_group_layouts"]:
                self.assertIn(bgl_id, manifest["bind_group_layouts"])

            # Content hashes must match reflect.rs algorithm.
            for bgl_id, bgl in manifest["bind_group_layouts"].items():
                recomputed = gen.format_shader_hash(
                    gen.bind_group_layout_content_hash(bgl["entries"])
                )
                self.assertEqual(recomputed, bgl_id)
            for pl_id, pl in manifest["pipeline_layouts"].items():
                recomputed = gen.format_shader_hash(
                    gen.pipeline_layout_content_hash(
                        [wgpk.normalize_shader_hash(i) for i in pl["bindGroupLayouts"]]
                    )
                )
                self.assertEqual(recomputed, pl_id)

        # Three graphics pipelines (splash, debug, unlit) keyed by vs+fs hashes.
        self.assertEqual(len(manifest["pipelines"]), 3)
        for pipeline in manifest["pipelines"].values():
            self.assertIn("vertex", pipeline["stages"])
            self.assertIn("fragment", pipeline["stages"])
            self.assertTrue(pipeline["layout_complete"])

        # shader-index.json covers the same hashes and pack ranges.
        index = artifacts.shader_index
        self.assertEqual(index["abi_version"], 1)
        self.assertEqual(index["pack"], "fallback.wgsl.pack")
        self.assertEqual(set(index["shaders"].keys()), pack_hashes)
        for entry in pack.entries:
            meta = index["shaders"][format_hash(entry.hash)]
            self.assertEqual(meta["stage"], entry.stage_name)
            self.assertEqual(meta["pack_offset"], entry.offset)
            self.assertEqual(meta["pack_length"], entry.length)

    def test_deterministic_byte_identical_output_across_runs(self) -> None:
        first = gen.build_fallback_artifacts()
        second = gen.build_fallback_artifacts()
        self.assertEqual(first.pack_bytes, second.pack_bytes)
        self.assertEqual(first.shader_index_text, second.shader_index_text)
        self.assertEqual(first.reflection_manifest_text, second.reflection_manifest_text)

        with tempfile.TemporaryDirectory() as tmp:
            out_a = Path(tmp) / "a"
            out_b = Path(tmp) / "b"
            gen.write_fallback_artifacts(out_a)
            gen.write_fallback_artifacts(out_b)
            for name in (
                "fallback.wgsl.pack",
                "shader-index.json",
                "reflection-manifest.json",
            ):
                self.assertEqual(
                    (out_a / name).read_bytes(),
                    (out_b / name).read_bytes(),
                    msg=name,
                )


class TestSafetyGuard(unittest.TestCase):
    def test_refuses_tracked_repository_source_dirs(self) -> None:
        for relative in ("web", "tools", "shaders", "docs", "runtime", "."):
            target = REPO_ROOT / relative
            self.assertFalse(
                gen.is_safe_output_dir(target, repo_root=REPO_ROOT),
                msg=relative,
            )
            with self.assertRaises(gen.UnsafeOutputPathError):
                gen.validate_output_dir(target, repo_root=REPO_ROOT)

        # Escape attempts via .. must also fail.
        with self.assertRaises(gen.UnsafeOutputPathError):
            gen.validate_output_dir(REPO_ROOT / "local-data" / ".." / "web", repo_root=REPO_ROOT)
        with self.assertRaises(gen.UnsafeOutputPathError):
            gen.validate_output_dir(REPO_ROOT / "work" / ".." / "tools", repo_root=REPO_ROOT)

    def test_allows_gitignored_and_temp_dirs(self) -> None:
        self.assertTrue(gen.is_safe_output_dir(REPO_ROOT / "local-data" / "shaders", repo_root=REPO_ROOT))
        self.assertTrue(gen.is_safe_output_dir(REPO_ROOT / "work" / "session", repo_root=REPO_ROOT))

        with tempfile.TemporaryDirectory() as tmp:
            temp_out = Path(tmp) / "fallback-out"
            self.assertTrue(gen.is_safe_output_dir(temp_out, repo_root=REPO_ROOT))
            artifacts = gen.write_fallback_artifacts(temp_out)
            self.assertTrue((temp_out / artifacts.pack_name).is_file())
            self.assertTrue((temp_out / "shader-index.json").is_file())
            self.assertTrue((temp_out / "reflection-manifest.json").is_file())

    def test_cli_refuses_tracked_out_dir(self) -> None:
        code, stdout, stderr = _run_cli(
            ["--out-dir", str(REPO_ROOT / "web"), "--json"]
        )
        self.assertEqual(code, 1)
        payload = json.loads(stdout)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["errorCode"], "unsafe_output_path")
        self.assertIn("tracked", payload["error"].lower())


class TestCli(unittest.TestCase):
    def test_cli_writes_artifacts_and_json_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "shaders"
            code, stdout, stderr = _run_cli(
                [
                    "--out-dir",
                    str(out_dir),
                    "--pack-name",
                    "fallback.wgsl.pack",
                    "--json",
                ]
            )
            self.assertEqual(code, 0, msg=stderr or stdout)
            payload = json.loads(stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["entryCount"], 7)
            self.assertEqual(payload["packName"], "fallback.wgsl.pack")
            self.assertEqual(len(payload["shaderHashes"]), 7)

            pack_path = out_dir / "fallback.wgsl.pack"
            self.assertTrue(pack_path.is_file())
            report = wgpk.inspect_pack(pack_path)
            self.assertTrue(report.clean)
            self.assertTrue((out_dir / "reflection-manifest.json").is_file())
            manifest = json.loads((out_dir / "reflection-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["shaders"].keys()), set(payload["shaderHashes"]))


def format_hash(value: int) -> str:
    return wgpk.format_shader_hash(value)


def _run_inspect(argv: list[str]) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        code = wgpk.main(argv)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


if __name__ == "__main__":
    unittest.main()
