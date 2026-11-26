#!/usr/bin/env python3
"""Unit tests for tools/gen_fallback_shaders.py (Track 4 task S9)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


class TestWgslHygieneValidation(unittest.TestCase):
    def test_builtin_shaders_pass_wgsl_hygiene_and_syntax_verify(self) -> None:
        shaders = gen.builtin_fallback_shaders()
        for shader in shaders:
            issues = gen.check_wgsl_hygiene(shader.wgsl)
            self.assertEqual(issues, (), msg=f"{shader.key}: {issues}")
            gen.validate_wgsl_hygiene(shader.wgsl, shader_key=shader.key)

        artifacts = gen.build_fallback_artifacts()
        syntax_report = wgpk.verify_wgsl_pack_syntax(
            artifacts.pack_bytes, source_label="fallback.wgsl.pack"
        )
        self.assertTrue(syntax_report.clean, msg=[v.to_dict() for v in syntax_report.violations])
        self.assertEqual(syntax_report.module_scope_let_count, 0)

    def test_hygiene_accepts_comments_nested_block_comments_and_fn_scope_let(self) -> None:
        valid_wgsl = (
            "// line comment with { unmatched brace, /* unclosed comment, let x = 1;\n"
            "/* outer block comment\n"
            "   /* nested block comment with { and let y = 2; */\n"
            "   still inside outer comment // not a line comment\n"
            "*/\n"
            "/* // single-line block comment containing double slash */\n"
            "const SCALE: f32 = 2.0;\n"
            "struct Palette {\n"
            "    outlet: f32,\n"
            "}\n"
            "@vertex\n"
            "fn delete_me(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {\n"
            "    let val = f32(idx) * SCALE;\n"
            "    return vec4<f32>(val, 0.0, 0.0, 1.0);\n"
            "}\n"
        )
        self.assertEqual(gen.check_wgsl_hygiene(valid_wgsl), ())
        gen.validate_wgsl_hygiene(valid_wgsl, shader_key="valid_comments")

    def test_hygiene_rejects_empty_source(self) -> None:
        for empty in ("", "   \n\t\n"):
            issues = gen.check_wgsl_hygiene(empty)
            self.assertEqual([i.code for i in issues], ["empty_source"])
            with self.assertRaises(gen.WgslHygieneError) as ctx:
                gen.validate_wgsl_hygiene(empty, shader_key="empty_shader")
            self.assertEqual(ctx.exception.code, "wgsl_hygiene_error")

    def test_hygiene_rejects_crlf_and_bare_cr_newlines(self) -> None:
        crlf_wgsl = "@fragment\r\nfn fs_main() -> @location(0) vec4<f32> {\r\n    return vec4<f32>(1.0);\r\n}\r\n"
        issues = gen.check_wgsl_hygiene(crlf_wgsl)
        self.assertEqual([i.code for i in issues], ["non_lf_newline"] * 4)
        self.assertEqual([i.line for i in issues], [1, 2, 3, 4])

        bare_cr_wgsl = "@fragment\rfn fs_main() -> @location(0) vec4<f32> {\n    return vec4<f32>(1.0);\n}\n"
        cr_issues = gen.check_wgsl_hygiene(bare_cr_wgsl)
        self.assertEqual([(i.code, i.line) for i in cr_issues], [("non_lf_newline", 1)])

    def test_hygiene_rejects_unbalanced_braces(self) -> None:
        unclosed = (
            "@vertex\n"
            "fn vs_main() -> @builtin(position) vec4<f32> {\n"
            "    return vec4<f32>(0.0);\n"
        )
        issues = gen.check_wgsl_hygiene(unclosed)
        self.assertEqual([(i.code, i.line) for i in issues], [("unbalanced_braces", 2)])

        extra_close = (
            "@vertex\n"
            "fn vs_main() -> @builtin(position) vec4<f32> {\n"
            "    return vec4<f32>(0.0);\n"
            "}\n"
            "}\n"
        )
        issues_close = gen.check_wgsl_hygiene(extra_close)
        self.assertEqual([(i.code, i.line) for i in issues_close], [("unbalanced_braces", 5)])

    def test_hygiene_rejects_unclosed_and_unmatched_block_comments(self) -> None:
        unclosed = "/* unclosed comment\n@vertex fn vs_main() {}\n"
        self.assertEqual(
            [(i.code, i.line) for i in gen.check_wgsl_hygiene(unclosed)],
            [("unclosed_block_comment", 1)],
        )

        unclosed_nested = "/* outer\n/* inner */\n@vertex fn vs_main() {}\n"
        self.assertEqual(
            [(i.code, i.line) for i in gen.check_wgsl_hygiene(unclosed_nested)],
            [("unclosed_block_comment", 1)],
        )

        slash_star_slash = "/*/\n"
        self.assertEqual(
            [(i.code, i.line) for i in gen.check_wgsl_hygiene(slash_star_slash)],
            [("unclosed_block_comment", 1)],
        )

        unmatched_close = "@vertex\nfn vs_main() {}\n*/\n"
        self.assertEqual(
            [(i.code, i.line) for i in gen.check_wgsl_hygiene(unmatched_close)],
            [("unmatched_block_comment_close", 3)],
        )

    def test_hygiene_rejects_module_scope_let(self) -> None:
        bad_wgsl = (
            "let BAD_CONST: f32 = 1.0;\n"
            "@fragment\n"
            "fn fs_main() -> @location(0) vec4<f32> {\n"
            "    let ok_local = BAD_CONST;\n"
            "    return vec4<f32>(ok_local);\n"
            "}\n"
            "let ALSO_BAD = 2u;\n"
        )
        issues = gen.check_wgsl_hygiene(bad_wgsl)
        self.assertEqual(
            [(i.code, i.line) for i in issues],
            [("module_scope_let", 1), ("module_scope_let", 7)],
        )

    def test_build_fallback_artifacts_rejects_unhygienic_custom_shader(self) -> None:
        bad_shader = gen.FallbackShader(
            key="bad_module_let",
            stage="fragment",
            entry_point="fs_main",
            wgsl="let INVALID: f32 = 1.0;\n@fragment fn fs_main() -> @location(0) vec4<f32> { return vec4<f32>(INVALID); }\n",
            layout_entries=(),
            bindings=(),
        )
        with self.assertRaises(gen.WgslHygieneError) as ctx:
            gen.build_fallback_artifacts(shaders=[bad_shader])
        self.assertEqual(ctx.exception.code, "wgsl_hygiene_error")
        self.assertEqual(ctx.exception.shader_key, "bad_module_let")
        self.assertEqual(ctx.exception.issues[0].code, "module_scope_let")


class TestCanonicalDeterministicGeneration(unittest.TestCase):
    def test_builtin_catalog_and_pack_use_canonical_ascending_hash_order(self) -> None:
        catalog = gen.builtin_fallback_shaders()
        self.assertTrue(gen.is_canonical_shader_order(catalog))
        self.assertEqual([s.hash for s in catalog], sorted(s.hash for s in catalog))

        artifacts = gen.build_fallback_artifacts()
        pack = wgpk.parse_wgpk(artifacts.pack_bytes)
        pack_hashes = [entry.hash for entry in pack.entries]
        self.assertEqual(pack_hashes, sorted(pack_hashes))
        self.assertEqual(pack_hashes, [s.hash for s in artifacts.shaders])

        index_keys = list(artifacts.shader_index["shaders"].keys())
        manifest_keys = list(artifacts.reflection_manifest["shaders"].keys())
        summary = artifacts.summary()
        self.assertEqual(index_keys, [format_hash(h) for h in pack_hashes])
        self.assertEqual(manifest_keys, [format_hash(h) for h in pack_hashes])
        self.assertEqual(summary["shaderHashes"], [format_hash(h) for h in pack_hashes])
        self.assertTrue(summary["canonicalOrder"])
        self.assertTrue(summary["wgslHygieneChecked"])

    def test_permuted_shader_catalog_produces_byte_identical_artifacts(self) -> None:
        canonical = gen.builtin_fallback_shaders()
        reversed_catalog = list(reversed(canonical))
        rotated_catalog = list(canonical[3:]) + list(canonical[:3])

        self.assertFalse(gen.is_canonical_shader_order(reversed_catalog))
        self.assertFalse(gen.is_canonical_shader_order(rotated_catalog))

        baseline = gen.build_fallback_artifacts(shaders=canonical)
        for variant in (reversed_catalog, rotated_catalog):
            built = gen.build_fallback_artifacts(shaders=variant)
            self.assertEqual(built.pack_bytes, baseline.pack_bytes)
            self.assertEqual(built.shader_index_text, baseline.shader_index_text)
            self.assertEqual(built.reflection_manifest_text, baseline.reflection_manifest_text)
            self.assertEqual(built.summary(), baseline.summary())
            self.assertEqual(built.pack_ranges, baseline.pack_ranges)

    def test_reflection_manifest_sorts_bindings_and_vertex_inputs(self) -> None:
        shader_a = gen.FallbackShader(
            key="mesh_a",
            stage="vertex",
            entry_point="vs_main",
            wgsl=gen.UNLIT_MESH_VS_WGSL,
            layout_entries=(
                gen._buffer_entry(0, gen.VISIBILITY_VERTEX, min_binding_size=64),
            ),
            bindings=(
                gen._binding_record(group=0, binding=2, name="b_two", resource="sampler", register_class="s"),
                gen._binding_record(group=0, binding=0, name="b_zero", resource="buffer", register_class="b"),
            ),
            vertex_inputs=(
                {"location": 1, "type": "vec2<f32>", "format": "float32x2"},
                {"location": 0, "type": "vec3<f32>", "format": "float32x3"},
            ),
        )
        shader_b = gen.FallbackShader(
            key="mesh_a",
            stage="vertex",
            entry_point="vs_main",
            wgsl=gen.UNLIT_MESH_VS_WGSL,
            layout_entries=(
                gen._buffer_entry(0, gen.VISIBILITY_VERTEX, min_binding_size=64),
            ),
            bindings=tuple(reversed(shader_a.bindings)),
            vertex_inputs=tuple(reversed(shader_a.vertex_inputs)),
        )
        manifest_a = gen.render_reflection_manifest_json(gen.build_reflection_manifest([shader_a]))
        manifest_b = gen.render_reflection_manifest_json(gen.build_reflection_manifest([shader_b]))
        self.assertEqual(manifest_a, manifest_b)

    def test_duplicate_shader_hash_is_rejected(self) -> None:
        first = gen.builtin_fallback_shaders()[0]
        dup = gen.FallbackShader(
            key="duplicate_copy",
            stage=first.stage,
            entry_point=first.entry_point,
            wgsl=first.wgsl,
            layout_entries=first.layout_entries,
            bindings=first.bindings,
        )
        with self.assertRaises(gen.FallbackShaderError) as ctx:
            gen.build_fallback_artifacts(shaders=[first, dup])
        self.assertEqual(ctx.exception.code, "duplicate_shader_hash")


class TestHostPathSanitization(unittest.TestCase):
    def test_sanitize_text_strips_host_paths_urls_and_usernames(self) -> None:
        with mock.patch.dict(os.environ, {"USERNAME": "secretdev", "USER": "secretdev"}):
            samples = [
                r"failed at C:\Users\alice\work\fallback.wgsl.pack",
                "failed at D:/projects/game/fallback.wgsl.pack",
                "failed at /Users/bob/playgta6web/local-data/shaders",
                "failed at /home/carol/repo/shaders",
                "failed at /tmp/pytest-run/out",
                r"failed at \\fileserver\share\game\fallback.wgsl.pack",
                "failed at file:///Users/dave/shader.wgsl",
                "built by secretdev on host",
            ]
            for raw in samples:
                cleaned = gen.sanitize_text(raw)
                self.assertNotIn("alice", cleaned, msg=raw)
                self.assertNotIn("bob", cleaned, msg=raw)
                self.assertNotIn("carol", cleaned, msg=raw)
                self.assertNotIn("dave", cleaned, msg=raw)
                self.assertNotIn("secretdev", cleaned, msg=raw)
                self.assertNotIn("C:\\", cleaned, msg=raw)
                self.assertNotIn("D:/", cleaned, msg=raw)
                self.assertNotIn("/Users/", cleaned, msg=raw)
                self.assertNotIn("/home/", cleaned, msg=raw)
                self.assertNotIn("file://", cleaned, msg=raw)

            # Relative shader paths and hex hashes must remain untouched.
            preserved = "shaders/0123456789abcdef.wgsl 0x0123456789abcdef"
            self.assertEqual(gen.sanitize_text(preserved), preserved)

    def test_sanitize_source_label_and_value(self) -> None:
        self.assertEqual(
            gen.sanitize_source_label(r"C:\Users\alice\work\fallback.wgsl.pack"),
            "fallback.wgsl.pack",
        )
        self.assertEqual(
            gen.sanitize_source_label("/home/bob/out/custom.wgsl.pack"),
            "custom.wgsl.pack",
        )
        self.assertEqual(gen.sanitize_source_label("C:"), "<unknown>")
        self.assertEqual(gen.sanitize_source_label(None), "<unknown>")

        nested = gen.sanitize_value(
            {
                "path": Path("/Users/alice/secret/fallback.wgsl.pack"),
                "raw": b"1234",
                "items": [r"C:\Users\bob\out.json", ("/tmp/test/a.wgsl",)],
            }
        )
        self.assertEqual(nested["path"], "fallback.wgsl.pack")
        self.assertEqual(nested["raw"], "<bytes:4>")
        self.assertEqual(nested["items"][0], "<path>")
        self.assertEqual(nested["items"][1], ["<path>"])

    def test_generated_json_and_cli_metadata_contain_zero_host_paths_or_usernames(self) -> None:
        with mock.patch.dict(os.environ, {"USERNAME": "secretdev", "USER": "secretdev"}):
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = Path(tmp) / "secretdev_dir" / "shaders"
                code, stdout, stderr = _run_cli(
                    ["--out-dir", str(out_dir), "--pack-name", "fallback.wgsl.pack", "--json"]
                )
                self.assertEqual(code, 0, msg=stderr or stdout)

                index_text = (out_dir / "shader-index.json").read_text(encoding="utf-8")
                manifest_text = (out_dir / "reflection-manifest.json").read_text(encoding="utf-8")

                for label, blob in (
                    ("cli_json", stdout),
                    ("shader_index", index_text),
                    ("reflection_manifest", manifest_text),
                ):
                    self.assertNotIn("\r", blob, msg=label)
                    self.assertNotIn(str(tmp), blob, msg=label)
                    self.assertNotIn("secretdev", blob.lower(), msg=label)
                    self.assertNotRegex(blob, r"[A-Za-z]:[\\/]", msg=label)
                    self.assertNotIn("/Users/", blob, msg=label)
                    self.assertNotIn("/home/", blob, msg=label)

    def test_custom_shader_metadata_and_cli_errors_are_sanitized(self) -> None:
        with mock.patch.dict(os.environ, {"USERNAME": "secretdev", "USER": "secretdev"}):
            leaky_shader = gen.FallbackShader(
                key="leaky",
                stage="vertex",
                entry_point="vs_main",
                wgsl=gen.SPLASH_VS_WGSL,
                layout_entries=(),
                bindings=(
                    gen._binding_record(
                        group=0,
                        binding=0,
                        name=r"C:\Users\secretdev\private\buf",
                        resource="buffer",
                        register_class="b",
                    ),
                ),
            )
            artifacts = gen.build_fallback_artifacts(shaders=[leaky_shader])
            self.assertNotIn("secretdev", artifacts.reflection_manifest_text.lower())
            self.assertNotIn(r"C:\Users", artifacts.reflection_manifest_text)

            rendered_index = gen.render_shader_index_json(
                r"C:\Users\secretdev\fallback.wgsl.pack",
                artifacts.shaders,
                artifacts.pack_ranges,
            )
            self.assertEqual(json.loads(rendered_index)["pack"], "fallback.wgsl.pack")
            self.assertNotIn("secretdev", rendered_index.lower())

            with mock.patch.object(
                gen,
                "write_fallback_artifacts",
                side_effect=gen.FallbackShaderError(
                    r"failed writing C:\Users\secretdev\bad.pack for secretdev",
                    code="test_error",
                ),
            ):
                code, stdout, _ = _run_cli(["--json"])
            self.assertEqual(code, 1)
            err_payload = json.loads(stdout)
            self.assertFalse(err_payload["ok"])
            self.assertNotIn("secretdev", err_payload["error"].lower())
            self.assertNotIn("C:\\", err_payload["error"])


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
