#!/usr/bin/env python3
"""Unit and CLI tests for tools/build_guest_wasm.py (Track 1 / R13)."""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import build_guest_wasm as bgw  # noqa: E402


def _write(path: Path, text: str = "unit\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _touch_lib(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"!<arch>\n")
    return path


def _fake_emsdk(repo: Path) -> None:
    em_dir = repo / "work" / "emsdk" / "upstream" / "emscripten"
    em_dir.mkdir(parents=True, exist_ok=True)
    _write(em_dir / "emcc", "#!/bin/sh\nexit 0\n")
    _write(em_dir / "em++", "#!/bin/sh\nexit 0\n")
    _write(em_dir / "emcc.bat", "@echo off\r\n")
    _write(em_dir / "em++.bat", "@echo off\r\n")


def _layout_sources(root: Path) -> Path:
    src = root / "recomp-out"
    _write(src / "shards" / "a.ll", "; ModuleID = 'a'\n")
    _write(src / "shards" / "b.ll", "; ModuleID = 'b'\n")
    _write(src / "pg6_import_thunks.cpp", "void pg6_thunk() {}\n")
    _write(src / "guest_table.cpp", "void guest_table() {}\n")
    _write(src / "import_thunks.cpp", "void import_thunks() {}\n")
    _write(src / "notes.txt", "not a source\n")
    return src


class RecordingRunner:
    """Fake emcc/em++ that writes requested -o outputs and records argv."""

    def __init__(self, *, fail_ids: frozenset[str] | None = None, delay: float = 0.0) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.compile_calls: list[tuple[str, ...]] = []
        self.link_calls: list[tuple[str, ...]] = []
        self.fail_ids = fail_ids or frozenset()
        self.delay = delay
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __call__(self, argv: Sequence[str]) -> bgw.CommandResult:
        tokens = tuple(str(a) for a in argv)
        with self._lock:
            self.calls.append(tokens)
            self.current += 1
            self.peak = max(self.peak, self.current)
        try:
            if self.delay:
                time.sleep(self.delay)
            joined = " ".join(tokens)
            if "check_wasm_exports.py" in joined:
                return bgw.CommandResult(
                    returncode=0,
                    stdout=json.dumps({"ok": True, "errors": []}),
                )
            out_path = _dash_o(tokens)
            is_link = "-lexports.js" in tokens or any(
                Path(t).name in {bgw.RUNTIME_JS, bgw.RUNTIME_WASM} for t in tokens
            )
            source_token = _source_token(tokens)
            if source_token and Path(source_token).name in self.fail_ids:
                return bgw.CommandResult(returncode=1, stderr=f"failed {Path(source_token).name}")
            if is_link and out_path is not None and Path(out_path).suffix.lower() == ".js":
                with self._lock:
                    self.link_calls.append(tokens)
                js = Path(out_path)
                js.parent.mkdir(parents=True, exist_ok=True)
                js.write_text("export default function createRuntimeModule() {}", encoding="utf-8")
                (js.with_suffix(".wasm")).write_bytes(b"\x00asm\x01\x00\x00\x00")
                return bgw.CommandResult(returncode=0)
            if out_path is not None:
                with self._lock:
                    self.compile_calls.append(tokens)
                Path(out_path).parent.mkdir(parents=True, exist_ok=True)
                Path(out_path).write_bytes(b"obj")
                return bgw.CommandResult(returncode=0)
            return bgw.CommandResult(returncode=1, stderr="no -o")
        finally:
            with self._lock:
                self.current -= 1


def _dash_o(argv: Sequence[str]) -> str | None:
    for idx, token in enumerate(argv):
        if token == "-o" and idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def _source_token(argv: Sequence[str]) -> str | None:
    for token in argv:
        suffix = Path(token).suffix.lower()
        if suffix in bgw.SOURCE_SUFFIXES:
            return token
    return None


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = bgw.main(argv)
    return code, stdout.getvalue(), stderr.getvalue()


class TestDiscovery(unittest.TestCase):
    def test_glob_discovers_shards_and_thunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = _layout_sources(Path(tmp))
            units = bgw.discover_sources(src)
            ids = [u.unit_id for u in units]
            self.assertEqual(
                ids,
                [
                    "guest_table.cpp",
                    "import_thunks.cpp",
                    "pg6_import_thunks.cpp",
                    "shards/a.ll",
                    "shards/b.ll",
                ],
            )
            kinds = {u.unit_id: u.kind for u in units}
            self.assertEqual(kinds["shards/a.ll"], "ll")
            self.assertEqual(kinds["pg6_import_thunks.cpp"], "cpp")

    def test_manifest_and_glob_union(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "out"
            _write(src / "shards" / "from_glob.ll", "; glob\n")
            _write(src / "listed.cpp", "int x;\n")
            _write(src / "only_in_manifest.c", "int y;\n")
            (src / "build-manifest.json").write_text(
                json.dumps(
                    {
                        "sources": ["listed.cpp", {"path": "only_in_manifest.c"}],
                        "shards": [{"file": "shards/from_glob.ll"}],
                    }
                ),
                encoding="utf-8",
            )
            ids = [u.unit_id for u in bgw.discover_sources(src)]
            self.assertIn("listed.cpp", ids)
            self.assertIn("only_in_manifest.c", ids)
            self.assertIn("shards/from_glob.ll", ids)

    def test_r9_pipeline_manifest_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "main"
            _write(src / "lift" / "shard-00000.ll", "; shard\n")
            _write(src / "lift" / "functions.ll", "; registration\n")
            _write(src / "stubs" / "pg6_import_thunks.cpp", "void thunk() {}\n")
            _write(src / "stubs" / "pg6_guest_seam.h", "#pragma once\n")
            _write(src / "image" / "guest.pgim.json", "{}\n")
            (src / "build-manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "playgta6-recomp-build",
                        "schema_version": 1,
                        "lift": {
                            "shards": [
                                {
                                    "index": 0,
                                    "ll": "lift/shard-00000.ll",
                                    "obj": None,
                                }
                            ],
                            "registration": {"ll": "lift/functions.ll", "obj": None},
                        },
                        "files": [
                            {"path": "stubs/pg6_import_thunks.cpp", "size": 1},
                            {"path": "stubs/pg6_import_thunks.h", "size": 1},
                            {"path": "image/guest.pgim", "size": 1},
                            {"path": "lift/shard-00000.ll", "size": 1},
                            {"path": "lift/functions.ll", "size": 1},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            ids = [u.unit_id for u in bgw.discover_sources(src)]
            self.assertEqual(
                ids,
                [
                    "lift/functions.ll",
                    "lift/shard-00000.ll",
                    "stubs/pg6_import_thunks.cpp",
                ],
            )


class TestCacheKeys(unittest.TestCase):
    def test_content_and_flag_changes_invalidate(self) -> None:
        flags_o2 = bgw.compile_flags_for(opt="O2", kind="ll")
        flags_o3 = bgw.compile_flags_for(opt="O3", kind="ll")
        key_a = bgw.cache_key_for(b"ir-a", flags_o2, kind="ll")
        key_b = bgw.cache_key_for(b"ir-b", flags_o2, kind="ll")
        key_flag = bgw.cache_key_for(b"ir-a", flags_o3, kind="ll")
        self.assertEqual(len(key_a), 64)
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_flag)
        self.assertEqual(key_a, bgw.cache_key_for(b"ir-a", flags_o2, kind="ll"))


class TestBuildOrchestrator(unittest.TestCase):
    def _repo(self, tmp: Path) -> Path:
        repo = tmp / "repo"
        _fake_emsdk(repo)
        _touch_lib(repo / "runtime" / "build-wasm64" / "libplaygta6_core.a")
        (repo / "runtime").mkdir(exist_ok=True)
        return repo

    def test_cache_hit_skips_recompile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            cache = root / "cache"
            out = root / "dist"
            runner = RecordingRunner()
            first = bgw.run_build(
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out,
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(first.ok, first.errors)
            self.assertEqual(first.cache_misses, 5)
            self.assertEqual(first.cache_hits, 0)
            compiles_first = len(runner.compile_calls)

            runner2 = RecordingRunner()
            second = bgw.run_build(
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out,
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner2,
                which_fn=lambda _name: None,
            )
            self.assertTrue(second.ok, second.errors)
            self.assertEqual(second.cache_hits, 5)
            self.assertEqual(second.cache_misses, 0)
            self.assertEqual(runner2.compile_calls, [])
            self.assertEqual(len(runner2.link_calls), 1)
            self.assertGreater(compiles_first, 0)

            _write(src / "shards" / "a.ll", "; ModuleID = 'a-changed'\n")
            runner3 = RecordingRunner()
            third = bgw.run_build(
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out,
                jobs=2,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner3,
                which_fn=lambda _name: None,
            )
            self.assertTrue(third.ok, third.errors)
            self.assertEqual(third.cache_hits, 4)
            self.assertEqual(third.cache_misses, 1)
            self.assertEqual(len(runner3.compile_calls), 1)

            runner4 = RecordingRunner()
            fourth = bgw.run_build(
                source_dir=src,
                repo_root=repo,
                cache_dir=cache,
                out_dir=out,
                jobs=2,
                opt="O3",
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner4,
                which_fn=lambda _name: None,
            )
            self.assertTrue(fourth.ok, fourth.errors)
            self.assertEqual(fourth.cache_hits, 0)
            self.assertEqual(fourth.cache_misses, 5)
            self.assertEqual(len(runner4.compile_calls), 5)

    def test_parallel_compile_fanout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner(delay=0.05)
            report = bgw.run_build(
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=4,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertGreaterEqual(runner.peak, 2)
            self.assertEqual(len(runner.compile_calls), 5)

    def test_shard_failure_reports_unit_and_skips_success_masking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner(fail_ids=frozenset({"b.ll"}))
            report = bgw.run_build(
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=4,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertFalse(report.ok)
            self.assertTrue(any("shards/b.ll" in e for e in report.errors), report.errors)
            failed = [u for u in report.sources if u.status == "error"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0].unit.unit_id, "shards/b.ll")
            self.assertEqual(runner.link_calls, [])
            compiled_ok = [u for u in report.sources if u.status == "compiled"]
            self.assertGreaterEqual(len(compiled_ok), 1)

    def test_link_command_has_exact_wasm64_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = self._repo(root)
            src = _layout_sources(root)
            runner = RecordingRunner()
            report = bgw.run_build(
                source_dir=src,
                repo_root=repo,
                cache_dir=root / "cache",
                out_dir=root / "dist",
                jobs=1,
                runtime_lib=repo / "runtime" / "build-wasm64" / "libplaygta6_core.a",
                run_command=runner,
                which_fn=lambda _name: None,
            )
            self.assertTrue(report.ok, report.errors)
            self.assertEqual(len(runner.link_calls), 1)
            link = runner.link_calls[0]
            for flag in (
                "-sMEMORY64=1",
                "-pthread",
                "-lexports.js",
                "-sMODULARIZE=1",
                "-sEXPORT_NAME=createRuntimeModule",
                "-sPTHREAD_POOL_SIZE=4",
                "-sALLOW_MEMORY_GROWTH=1",
                "-sENVIRONMENT=worker",
                "-sERROR_ON_UNDEFINED_SYMBOLS=0",
            ):
                self.assertIn(flag, link)
            self.assertIn("libplaygta6_core.a", [Path(t).name for t in link])
            self.assertEqual(report.outputs, (bgw.RUNTIME_JS, bgw.RUNTIME_WASM))
            self.assertTrue((root / "dist" / "runtime.js").is_file())
            self.assertTrue((root / "dist" / "runtime.wasm").is_file())
            self.assertEqual(bgw.LINK_FLAGS, report.link_flags)

            compile_cmd = runner.compile_calls[0]
            self.assertIn("-sMEMORY64=1", compile_cmd)
            self.assertIn("-pthread", compile_cmd)
            self.assertIn("-O2", compile_cmd)
            self.assertIn("-c", compile_cmd)

    def test_compile_argv_builder_and_link_argv_builder(self) -> None:
        argv = bgw.build_compile_argv(
            Path("emcc"),
            Path("shard.ll"),
            Path("shard.o"),
            bgw.compile_flags_for(opt="O2", kind="ll"),
        )
        self.assertEqual(argv[0], "emcc")
        self.assertIn("-sMEMORY64=1", argv)
        self.assertIn("-pthread", argv)
        self.assertIn("-x", argv)
        self.assertIn("ir", argv)

        link = bgw.build_link_argv(
            "em++",
            [Path("a.o"), Path("b.o")],
            output_js=Path("web/dist/runtime.js"),
            runtime_libs=[Path("libplaygta6_core.a")],
        )
        self.assertEqual(link[0], "em++")
        self.assertIn("-sMEMORY64=1", link)
        self.assertIn("-pthread", link)
        self.assertIn("-lexports.js", link)
        self.assertIn("-sMODULARIZE=1", link)
        self.assertEqual(link[-2], "-o")
        self.assertTrue(str(link[-1]).replace("\\", "/").endswith("web/dist/runtime.js"))


class TestCli(unittest.TestCase):
    def test_check_only_discovers_work_emsdk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            _fake_emsdk(repo)
            old_path = os.environ.get("PATH", "")
            old_emsdk = os.environ.pop("EMSDK", None)
            try:
                os.environ["PATH"] = str(repo)
                code, out, err = _run_cli(
                    [
                        "--check-only",
                        "--json",
                        "--repo-root",
                        str(repo),
                    ]
                )
            finally:
                os.environ["PATH"] = old_path
                if old_emsdk is not None:
                    os.environ["EMSDK"] = old_emsdk
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["mode"], "check-only")
            self.assertTrue(payload["toolchain"]["emcc_found"])
            self.assertEqual(payload["toolchain"]["origin"], "work/emsdk")

    def test_check_only_missing_toolchain_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "empty"
            repo.mkdir()
            toolchain = bgw.discover_emscripten(
                repo_root=repo,
                env={},
                which_fn=lambda _name: None,
            )
            self.assertFalse(toolchain.ok)
            old_path = os.environ.get("PATH", "")
            old_emsdk = os.environ.pop("EMSDK", None)
            try:
                os.environ["PATH"] = str(repo)
                code, out, _err = _run_cli(
                    ["--check-only", "--json", "--repo-root", str(repo)]
                )
            finally:
                os.environ["PATH"] = old_path
                if old_emsdk is not None:
                    os.environ["EMSDK"] = old_emsdk
            self.assertEqual(code, 1)
            payload = json.loads(out)
            self.assertFalse(payload["ok"])
            self.assertTrue(payload["errors"])

    def test_dry_run_and_json_are_sanitized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            src = _layout_sources(root)
            user = os.environ.get("USERNAME") or os.environ.get("USER") or "nobody"
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--json",
                    "--jobs",
                    "2",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache"),
                    "--out-dir",
                    str(root / "dist"),
                ]
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["mode"], "dry-run")
            self.assertEqual(payload["source_count"], 5)
            self.assertIn("-sMEMORY64=1", payload["link_flags"])
            self.assertIn("-pthread", payload["link_flags"])
            self.assertIn("-lexports.js", payload["link_flags"])
            self.assertIn("-sMODULARIZE=1", payload["link_flags"])
            dumped = json.dumps(payload)
            self.assertNotIn("\\\\", dumped)
            if len(user) >= 3:
                self.assertNotRegex(dumped, rf"(?<![A-Za-z0-9_]){user}(?![A-Za-z0-9_])")
            for unit in payload["sources"]:
                self.assertIn(unit["status"], {"planned", "hit"})
                self.assertTrue(unit["id"].replace("\\", "/") == unit["id"])

    def test_dry_run_text_lists_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            _fake_emsdk(repo)
            src = _layout_sources(root)
            code, out, err = _run_cli(
                [
                    str(src),
                    "--dry-run",
                    "--sanitize",
                    "--repo-root",
                    str(repo),
                    "--cache-dir",
                    str(root / "cache"),
                    "--out-dir",
                    str(root / "dist"),
                ]
            )
            self.assertEqual(code, 0, err)
            self.assertIn("Guest Wasm64 build [PASS]", out)
            self.assertIn("shards/a.ll", out)
            self.assertIn("-sMEMORY64=1", out)
            self.assertIn("-lexports.js", out)

    def test_sanitize_argv_strips_host_paths(self) -> None:
        argv = [
            r"C:\Users\someone\emsdk\upstream\emscripten\emcc.bat",
            "-sMEMORY64=1",
            "-pthread",
            r"C:\Users\someone\work\shards\a.ll",
            "-o",
            r"C:\Users\someone\work\wasm_obj_cache\aa\a.o",
        ]
        cleaned = bgw.sanitize_argv(argv)
        self.assertEqual(cleaned[0], "emcc")
        self.assertIn("-sMEMORY64=1", cleaned)
        self.assertEqual(cleaned[-1], "a.o")
        joined = " ".join(cleaned)
        self.assertNotIn("someone", joined)
        self.assertNotIn("C:\\", joined)


class TestLinkFlagContract(unittest.TestCase):
    def test_public_constants_match_cmake(self) -> None:
        cmake = (REPO_ROOT / "runtime" / "CMakeLists.txt").read_text(encoding="utf-8")
        for flag in bgw.LINK_FLAGS:
            self.assertIn(flag, cmake, flag)
        for flag in bgw.COMPILE_WASM_FLAGS:
            self.assertIn(flag, cmake, flag)


if __name__ == "__main__":
    unittest.main()
