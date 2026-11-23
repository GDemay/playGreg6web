#!/usr/bin/env python3
"""Unit tests for tools/convert_dxil_corpus.py (Track 4 task S11)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import convert_dxil_corpus as conv  # noqa: E402
from tools import inspect_wgsl_pack as wgpk  # noqa: E402


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        code = conv.main(argv)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


def _assert_no_host_leak(testcase: unittest.TestCase, text: str, *paths: Path) -> None:
    testcase.assertNotIn(":\\", text)
    testcase.assertNotRegex(text, r"[A-Za-z]:/")
    for path in paths:
        raw = str(path)
        testcase.assertNotIn(raw, text)
        testcase.assertNotIn(raw.replace("\\", "/"), text)
        testcase.assertNotIn(raw.replace("/", "\\"), text)
    for env_var in ("USERNAME", "USER", "LOGNAME"):
        user = os.environ.get(env_var, "").strip()
        if len(user) >= 3:
            testcase.assertNotIn(user.lower(), text.lower())


class RecordingRunner:
    """Injectable command_runner that records argv and writes mock artifacts."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_stage: str | None = None
        self.fail_stderr: str = "boom"
        self.fail_code: int = 1

    def __call__(self, argv: list[str], **_kwargs: Any) -> conv.CommandResult:
        argv_list = [str(item) for item in argv]
        self.calls.append(argv_list)
        joined = " ".join(argv_list)

        if self.fail_stage == "dxil" and "-o" in argv_list:
            return conv.CommandResult(returncode=self.fail_code, stderr=self.fail_stderr)
        if self.fail_stage == "val" and "spirv-val" in joined:
            return conv.CommandResult(returncode=self.fail_code, stderr=self.fail_stderr)
        if self.fail_stage == "convert" and "convert" in argv_list:
            return conv.CommandResult(returncode=self.fail_code, stderr=self.fail_stderr)
        if self.fail_stage == "reflect" and "reflect" in argv_list:
            return conv.CommandResult(returncode=self.fail_code, stderr=self.fail_stderr)

        if "-o" in argv_list:
            out_idx = argv_list.index("-o") + 1
            Path(argv_list[out_idx]).write_bytes(b"\x03\x02\x23\x07" + b"\x00" * 16)
            return conv.CommandResult(returncode=0, stdout="dxil-spirv ok")

        if argv_list and argv_list[0].endswith("spirv-val") or (
            len(argv_list) == 2 and "spirv-val" in argv_list[0]
        ):
            return conv.CommandResult(returncode=0, stdout="PASS")

        if "convert" in argv_list:
            spirv_dir = Path(argv_list[argv_list.index("--spirv") + 1])
            out_dir = Path(argv_list[argv_list.index("--out") + 1])
            pack_path = Path(argv_list[argv_list.index("--pack") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            wgsl = "@compute @workgroup_size(1) fn main() {}\n"
            (out_dir / "deadbeef.wgsl").write_text(wgsl, encoding="utf-8")
            pack_path.parent.mkdir(parents=True, exist_ok=True)
            pack_path.write_bytes(wgpk.build_wgpk([{"stage": "compute", "wgsl": wgsl}]))
            (out_dir / "shader-index.json").write_text(
                '{"abi_version":1,"pack":"converted.wgsl.pack","shaders":{}}\n',
                encoding="utf-8",
            )
            self._last_spirv_dir = spirv_dir
            return conv.CommandResult(returncode=0, stdout="Converted 1/1")

        if "reflect" in argv_list:
            out_dir = Path(argv_list[argv_list.index("--out") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "layout-manifest.json").write_text(
                '{"schema_version":1,"shaders":{}}\n',
                encoding="utf-8",
            )
            return conv.CommandResult(returncode=0, stdout="Reflected 1/1")

        return conv.CommandResult(returncode=0)


class TestPathSanitization(unittest.TestCase):
    def test_posix_rel_key_normalizes_windows_and_posix(self) -> None:
        self.assertEqual(conv.posix_rel_key(r"nested\dir\foo.dxil"), "nested/dir/foo.dxil")
        self.assertEqual(conv.posix_rel_key("nested/dir/foo.dxil"), "nested/dir/foo.dxil")
        self.assertEqual(
            conv.posix_rel_key(r"C:\Users\alice\corpus\foo.dxil"),
            "Users/alice/corpus/foo.dxil",
        )
        self.assertEqual(
            conv.posix_rel_key("C:/Users/alice/corpus/foo.dxil"),
            "Users/alice/corpus/foo.dxil",
        )

    def test_sanitize_source_label_splits_both_separators(self) -> None:
        self.assertEqual(
            conv.sanitize_source_label(r"C:\Users\alice\game\pixel.dxil"),
            "pixel.dxil",
        )
        self.assertEqual(
            conv.sanitize_source_label("/home/alice/game/pixel.dxil"),
            "pixel.dxil",
        )
        self.assertEqual(conv.sanitize_source_label(r"nested\\mixed/foo.dxbc"), "foo.dxbc")

    def test_windows_paths_redacted_even_on_posix_hosts(self) -> None:
        leaked = r"failed C:\Users\alice\AppData\Local\game\mesh.dxil parse error"
        cleaned = conv.scrub_sensitive_text(leaked)
        self.assertNotIn("alice", cleaned)
        self.assertNotIn(":\\", cleaned)
        self.assertIn("<path>", cleaned)

        posix = "error writing /home/alice/work/out.spv"
        posix_cleaned = conv.scrub_sensitive_text(posix)
        self.assertNotIn("/home/alice", posix_cleaned)
        self.assertIn("<path>", posix_cleaned)

    def test_sanitize_relative_path_uses_corpus_tail_for_windows_abs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "dxil-corpus"
            corpus.mkdir()
            fake_windows = r"C:\Users\bob\dxil-corpus\nested\lit.dxil"
            rel = conv.sanitize_relative_path(fake_windows, corpus)
            self.assertEqual(rel, "nested/lit.dxil")
            self.assertNotIn("bob", rel)
            self.assertNotIn("C:", rel)


class TestFailureClassifier(unittest.TestCase):
    def test_buckets_from_stderr_and_stage(self) -> None:
        cases = [
            ("missing_tool:dxil-spirv", 127, conv.STAGE_DXIL_TO_SPIRV, "missing_tool"),
            ("dxil-spirv: not found", 1, conv.STAGE_DXIL_TO_SPIRV, "missing_tool"),
            ("invalid DXIL container: truncated DXBC", 1, conv.STAGE_DXIL_TO_SPIRV, "dxil_parse_error"),
            ("unsupported shader model sm_5_0", 1, conv.STAGE_DXIL_TO_SPIRV, "unsupported_sm"),
            ("blob is RDNA2 precompiled ISA", 1, conv.STAGE_DXIL_TO_SPIRV, "rdna2_precompiled_only"),
            ("error: line 0: SPV_ERROR_INVALID_BINARY", 1, conv.STAGE_SPIRV_VAL, "spirv_validation_error"),
            ("naga: unsupported capability GroupNonUniform", 1, conv.STAGE_SPIRV_TO_WGSL, "naga_unsupported_cap"),
            ("maxBindGroups exceeded for DescriptorSet 8", 1, conv.STAGE_SPIRV_TO_WGSL, "bind_group_limit"),
            ("@workgroup_size invocations per workgroup exceed limit", 1, conv.STAGE_SPIRV_TO_WGSL, "workgroup_limit"),
            ("??? totally new backend panic", 1, conv.STAGE_SPIRV_TO_WGSL, "unknown_error"),
        ]
        for stderr, code, stage, expected in cases:
            with self.subTest(expected=expected, stderr=stderr):
                self.assertEqual(conv.classify_failure(stderr, code, stage), expected)

    def test_classifier_does_not_echo_raw_bytes_or_host_paths(self) -> None:
        stderr = "C:\\Users\\alice\\secret.dxil\n" + "\x00\x01\xff" * 40
        bucket = conv.classify_failure(stderr, 1, conv.STAGE_DXIL_TO_SPIRV)
        self.assertIn(bucket, conv.FAILURE_BUCKETS)
        message = conv.sanitize_message(stderr)
        self.assertNotIn("alice", message)
        self.assertNotIn("\x00", message)
        self.assertNotIn(":\\", message)


class TestDiscovery(unittest.TestCase):
    def test_glob_and_manifest_with_windows_separators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            nested = corpus / "nested"
            nested.mkdir(parents=True)
            (nested / "pixel.dxil").write_bytes(b"DXBC" + b"\x00" * 32)
            (corpus / "compute.dxbc").write_bytes(b"DXBC" + b"\x00" * 16)
            (corpus / "notes.txt").write_text("ignore", encoding="utf-8")
            (corpus / "corpus-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "entries": [
                            {"path": r"nested\pixel.dxil", "sha256": "ab" * 32},
                            {"relative_path": r"C:\Users\bob\corpus\compute.dxbc"},
                            {"file": "missing.dxil"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            found = conv.discover_corpus(corpus)
            labels = sorted(item.rel_path.replace("\\", "/") for item in found)
            self.assertEqual(labels, ["compute.dxbc", "nested/pixel.dxil"])
            self.assertTrue(any(item.source == "manifest" for item in found))


class TestToolDetection(unittest.TestCase):
    def test_detects_path_tools_and_cargo_binaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "target" / "release"
            release.mkdir(parents=True)
            binary = release / ("playgta6-shaders.exe" if os.name == "nt" else "playgta6-shaders")
            binary.write_bytes(b"fake")

            mapping = {
                "dxil-spirv": r"C:\Tools\dxil-spirv.exe",
                "dxil-spirv.exe": r"C:\Tools\dxil-spirv.exe",
                "spirv-val": "/usr/bin/spirv-val",
                "spirv-val.exe": None,
                "playgta6-shaders": None,
                "playgta6-shaders.exe": None,
            }

            def fake_which(name: str) -> str | None:
                return mapping.get(name)

            tools = conv.detect_tools(repo_root=root, which_fn=fake_which)
            self.assertTrue(tools.dxil_spirv)
            self.assertTrue(tools.spirv_val)
            self.assertTrue(tools.playgta6_shaders)
            self.assertIn("playgta6-shaders", Path(tools.playgta6_shaders).name)

            empty = conv.detect_tools(repo_root=root / "missing", which_fn=lambda _name: None)
            self.assertIsNone(empty.dxil_spirv)
            self.assertIsNone(empty.spirv_val)
            self.assertIsNone(empty.playgta6_shaders)


class TestSafetyGuard(unittest.TestCase):
    def test_refuses_tracked_and_allows_gitignored_or_temp(self) -> None:
        for relative in ("web", "tools", "shaders", "docs", "runtime", "."):
            target = REPO_ROOT / relative
            self.assertFalse(conv.is_safe_output_dir(target, repo_root=REPO_ROOT), msg=relative)
            with self.assertRaises(conv.UnsafeOutputPathError):
                conv.validate_output_dir(target, repo_root=REPO_ROOT)

        with self.assertRaises(conv.UnsafeOutputPathError):
            conv.validate_output_dir(REPO_ROOT / "local-data" / ".." / "web", repo_root=REPO_ROOT)

        self.assertTrue(
            conv.is_safe_output_dir(REPO_ROOT / "local-data" / "shaders" / "converted", repo_root=REPO_ROOT)
        )
        self.assertTrue(conv.is_safe_output_dir(REPO_ROOT / "work" / "dxil", repo_root=REPO_ROOT))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(conv.is_safe_output_dir(Path(tmp) / "out", repo_root=REPO_ROOT))

    def test_cli_refuses_tracked_out_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            code, stdout, stderr = _run_cli(
                [
                    "--corpus-dir",
                    str(corpus),
                    "--out-dir",
                    str(REPO_ROOT / "web"),
                    "--json",
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 1)
            payload = json.loads(stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["errorCode"], "unsafe_output_path")
            _assert_no_host_leak(self, stdout + stderr, REPO_ROOT / "web")


class TestParallelCachedConversion(unittest.TestCase):
    def _setup_corpus(self, tmp: str) -> tuple[Path, Path, Path]:
        corpus = Path(tmp) / "corpus"
        out_dir = Path(tmp) / "out"
        repo = Path(tmp) / "repo"
        (repo / "target" / "release").mkdir(parents=True)
        binary = repo / "target" / "release" / "playgta6-shaders"
        binary.write_bytes(b"fake")
        nested = corpus / "a"
        nested.mkdir(parents=True)
        (nested / "one.dxil").write_bytes(b"DXBCone")
        (corpus / "two.dxbc").write_bytes(b"DXBCtwo")
        return corpus, out_dir, repo

    def test_parallel_conversion_with_mock_runner_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, out_dir, repo = self._setup_corpus(tmp)
            runner = RecordingRunner()
            which_map = {
                "dxil-spirv": "/opt/dxil-spirv",
                "dxil-spirv.exe": "/opt/dxil-spirv",
                "spirv-val": "/opt/spirv-val",
                "spirv-val.exe": "/opt/spirv-val",
            }

            report = conv.convert_corpus(
                corpus,
                out_dir,
                jobs=2,
                repo_root=repo,
                which_fn=lambda name: which_map.get(name),
                command_runner=runner,
            )
            self.assertEqual(report.discovered, 2)
            self.assertEqual(report.converted, 2)
            self.assertEqual(report.failed, 0)
            self.assertTrue((out_dir / "coverage.json").is_file())
            self.assertTrue((out_dir / "coverage.md").is_file())

            convert_calls = [c for c in runner.calls if "convert" in c]
            reflect_calls = [c for c in runner.calls if "reflect" in c]
            self.assertEqual(len(convert_calls), 2)
            self.assertEqual(len(reflect_calls), 2)
            for call in convert_calls:
                self.assertIn("--spirv", call)
                self.assertIn("--out", call)
                self.assertIn("--pack", call)
            for call in reflect_calls:
                self.assertIn("--spirv", call)
                self.assertIn("--out", call)

            for result in report.results:
                self.assertIn("converted.spv", result.artifacts)
                self.assertIn("converted.wgsl.pack", result.artifacts)
                self.assertIn("reflection-manifest.json", result.artifacts)

            coverage = json.loads((out_dir / "coverage.json").read_text(encoding="utf-8"))
            _assert_no_host_leak(self, json.dumps(coverage), corpus, out_dir, Path.home())

            first_calls = len(runner.calls)
            report2 = conv.convert_corpus(
                corpus,
                out_dir,
                jobs=2,
                repo_root=repo,
                which_fn=lambda name: which_map.get(name),
                command_runner=runner,
            )
            self.assertEqual(report2.cached_count, 2)
            self.assertEqual(len(runner.calls), first_calls)

    def test_missing_dxil_spirv_is_recorded_without_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, out_dir, repo = self._setup_corpus(tmp)
            runner = RecordingRunner()
            report = conv.convert_corpus(
                corpus,
                out_dir,
                jobs=1,
                repo_root=repo,
                which_fn=lambda _name: None,
                command_runner=runner,
            )
            self.assertEqual(report.failed, 2)
            self.assertEqual(report.buckets()["missing_tool"], 2)
            self.assertEqual(runner.calls, [])
            self.assertTrue(all(item.missing_tool == "dxil-spirv" for item in report.results))

    def test_missing_playgta6_shaders_after_spv_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus, out_dir, repo = self._setup_corpus(tmp)
            for leftover in (repo / "target" / "release").glob("playgta6-shaders*"):
                leftover.unlink()
            runner = RecordingRunner()
            which_map = {"dxil-spirv": "/opt/dxil-spirv", "dxil-spirv.exe": "/opt/dxil-spirv"}
            report = conv.convert_corpus(
                corpus,
                out_dir,
                jobs=1,
                repo_root=repo,
                which_fn=lambda name: which_map.get(name),
                command_runner=runner,
            )
            self.assertEqual(report.failed, 2)
            self.assertTrue(all(item.missing_tool == "playgta6-shaders" for item in report.results))
            self.assertTrue(all(conv.STAGE_DXIL_TO_SPIRV in item.stages for item in report.results))


class TestCliDryRunJson(unittest.TestCase):
    def test_dry_run_json_lists_discovery_without_running_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "MyPrivateCorpus"
            corpus.mkdir()
            (corpus / "mesh.dxil").write_bytes(b"DXBC")
            out_dir = Path(tmp) / "converted"
            runner = RecordingRunner()

            with mock.patch.object(conv, "default_command_runner", runner):
                code, stdout, stderr = _run_cli(
                    [
                        "--corpus-dir",
                        str(corpus),
                        "--out-dir",
                        str(out_dir),
                        "--dry-run",
                        "--json",
                        "--jobs",
                        "1",
                    ]
                )
            self.assertEqual(code, 0, msg=stderr)
            payload = json.loads(stdout)
            self.assertTrue(payload["ok"])
            self.assertTrue(payload["dryRun"])
            self.assertEqual(payload["discovered"], 1)
            self.assertEqual(payload["results"][0]["source"], "mesh.dxil")
            self.assertEqual(payload["results"][0]["status"], "dry_run")
            self.assertEqual(runner.calls, [])
            self.assertFalse((out_dir / "coverage.json").exists())
            _assert_no_host_leak(self, stdout, corpus, Path.home())
            self.assertNotIn("MyPrivateCorpus", json.dumps(payload["results"]))

    def test_check_only_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "corpus"
            corpus.mkdir()
            (corpus / "a.dxbc").write_bytes(b"DXBC")
            out_dir = Path(tmp) / "out"
            code, stdout, stderr = _run_cli(
                [
                    "--corpus-dir",
                    str(corpus),
                    "--out-dir",
                    str(out_dir),
                    "--check-only",
                    "--json",
                ]
            )
            self.assertEqual(code, 0, msg=stderr)
            payload = json.loads(stdout)
            self.assertTrue(payload["checkOnly"])
            self.assertEqual(payload["discovered"], 1)

            code_md, stdout_md, stderr_md = _run_cli(
                [
                    "--corpus-dir",
                    str(corpus),
                    "--out-dir",
                    str(out_dir),
                    "--check-only",
                ]
            )
            self.assertEqual(code_md, 0, msg=stderr_md)
            self.assertIn("# DXIL Corpus Conversion Coverage", stdout_md)
            self.assertIn("`dxil-spirv`", stdout_md)


if __name__ == "__main__":
    unittest.main()
