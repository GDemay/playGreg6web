import base64
import hashlib
import io
import json
import os
import re
import socket
import struct
import sys
import tempfile
import threading
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import compare_boot_reports  # noqa: E402
import run_browser_boot as rbb  # noqa: E402


def touch(path: Path, when: float, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    os.utime(path, (when, when))


def make_sample_probe(
    *,
    boot_mode: str = "smoke-demo",
    session_state: str = "running_smoke",
    stage: str = "FIRST_PRESENT",
    presents: int = 5,
    real_draws: int = 0,
    fallback_draws: int = 5,
    validation_errors: int = 0,
    milestones: list[dict[str, Any]] | None = None,
    last_trap: dict[str, Any] | None = None,
    hud_log: list[str] | None = None,
) -> dict[str, Any]:
    if milestones is None:
        milestones = [
            {"milestone": "PGIM_LOAD", "index": 0, "reached": True, "elapsedMs": 12.5},
            {"milestone": "FIRST_PRESENT", "index": 8, "reached": True, "elapsedMs": 125.0},
        ]
    return {
        "href": "http://127.0.0.1:8123/",
        "readyState": "complete",
        "capabilities": {
            "crossOriginIsolated": True,
            "sharedArrayBuffer": True,
            "webgpu": True,
        },
        "bootState": {
            "bootMode": boot_mode,
            "sessionState": session_state,
            "milestone": stage,
            "presentCount": presents,
            "realPackDrawCount": real_draws,
            "fallbackDrawCount": fallback_draws,
            "unboundBindGroupCount": 0,
            "skippedDrawCount": 0,
            "validationErrorCount": validation_errors,
            "gpu": {
                "presentCount": presents,
                "realPackDrawCount": real_draws,
                "fallbackDrawCount": fallback_draws,
                "unboundBindGroupCount": 0,
                "skippedDrawCount": 0,
                "validationErrorCount": validation_errors,
            },
        },
        "diagnostics": {
            "bootMode": boot_mode,
            "sessionState": session_state,
            "milestone": stage,
            "framesPresented": presents,
            "validationErrorCount": validation_errors,
        },
        "tracerSummary": {
            "stage": stage,
            "milestones": milestones,
            "lastTrap": last_trap,
        },
        "hud": {
            "stage": stage,
            "status": "Running",
            "frames": str(presents),
            "fps": "60",
            "gpuErrors": str(validation_errors),
            "modeBadge": "Smoke Demo" if boot_mode == "smoke-demo" else "Real Wasm64",
            "logLines": hud_log if hud_log is not None else ["[00:00:01] Boot complete"],
        },
    }


def make_fake_session_result(
    *,
    ok: bool = True,
    status: str = "ok",
    exit_code: int = 0,
    probe: dict[str, Any] | None = None,
    console_lines: list[str] | None = None,
    page_errors: list[str] | None = None,
    presents: list[dict[str, Any]] | None = None,
    first_present_ms: float | None = 125.0,
) -> rbb.BrowserSessionResult:
    effective_probe = make_sample_probe() if (probe is None and ok) else (probe or {})
    effective_presents = (
        [
            {
                "index": 0,
                "wall_ms": 125.0,
                "dispatches": 0,
                "import_calls": 0,
                "counters": {
                    "draws_real_pack": 0,
                    "draws_fallback": 5,
                    "draws_skipped": 0,
                    "gpu_validation_errors": 0,
                },
            }
        ]
        if (presents is None and ok)
        else (presents or [])
    )
    return rbb.BrowserSessionResult(
        ok=ok,
        exit_code=exit_code,
        elapsed_ms=250.0,
        status=status,
        message="Running" if ok else "failed",
        crashed=not ok,
        server_ready={"status_code": 200, "coop": "same-origin", "coep": "require-corp", "api_status": {"ok": True}},
        final_probe=effective_probe,
        console_lines=console_lines if console_lines is not None else ["[   0.10s] [console.log] ready"],
        page_errors=page_errors or [],
        presents=effective_presents,
        first_present_ms=first_present_ms if ok else None,
    )


class RepoFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.repo = root / "repo"
        self.build = self.repo / "work" / "browser-boot"
        self.boot_dir = self.repo / "work" / "recomp_build"
        self.out_dir = self.repo / "web" / "dist"
        self.exe = root / "external" / "game.exe"
        self.game_dir = root / "external" / "pkg"
        self.game_dir.mkdir(parents=True)
        self.chrome = root / "external" / ("chrome.exe" if sys.platform == "win32" else "chrome")
        self.base = 1_000_000.0

        touch(self.repo / "Cargo.toml", self.base, '[workspace]\nmembers = ["recomp"]\n')
        touch(self.repo / "Cargo.lock", self.base, "# lock\n")
        touch(self.repo / "recomp" / "Cargo.toml", self.base, '[package]\nname="playgta6-recomp"\n')
        touch(self.repo / "recomp" / "src" / "main.rs", self.base, "fn main() {}\n")
        touch(self.repo / "runtime" / "a.cpp", self.base, "int x;\n")
        touch(self.repo / "tools" / "build_guest_wasm.py", self.base, "# build\n")
        touch(self.repo / "server" / "serve_local.py", self.base, "# server\n")
        touch(self.repo / "web" / "index.html", self.base, "<!doctype html>\n")
        touch(self.exe, self.base, "MZ")
        touch(self.chrome, self.base, "bin")

        self.recomp = self.repo / "target" / "release" / ("playgta6-recomp" + rbb.EXE_SUFFIX)
        self.manifest = self.boot_dir / "program-manifest.json"
        self.cmake_cache = self.build / "runtime-pic" / "CMakeCache.txt"
        self.runtime_lib = self.build / "runtime-pic" / "libplaygta6_core.a"
        self.runtime_wasm = self.out_dir / "runtime.wasm"
        self.runtime_js = self.out_dir / "runtime.js"

        touch(self.recomp, self.base + 10)
        touch(self.manifest, self.base + 20, "{}")
        touch(self.cmake_cache, self.base + 25)
        touch(self.runtime_lib, self.base + 30)
        touch(self.runtime_wasm, self.base + 40)
        touch(self.runtime_js, self.base + 40)

    def run_cli(
        self,
        *argv: str,
        runner=None,
        session_runner=None,
        chrome_discoverer=None,
        env=None,
    ) -> tuple[int, str, str, list[list[str]], list[dict[str, Any]]]:
        cmd_calls: list[list[str]] = []
        session_calls: list[dict[str, Any]] = []

        def default_runner(cmd, cwd, log, timeout):
            cmd_calls.append(list(cmd))
            return 0

        def default_session(**kwargs):
            session_calls.append(kwargs)
            return make_fake_session_result(
                probe=make_sample_probe(
                    boot_mode="smoke-demo" if kwargs.get("smoke") else "real-wasm64",
                    session_state="running_smoke" if kwargs.get("smoke") else "running_wasm",
                )
            )

        def default_chrome(explicit, **kw):
            return explicit if explicit is not None else self.chrome

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = rbb.main(
                list(argv),
                runner=runner or default_runner,
                session_runner=session_runner or default_session,
                chrome_discoverer=chrome_discoverer or default_chrome,
                env={} if env is None else env,
                repo_root=self.repo,
            )
        return code, out.getvalue(), err.getvalue(), cmd_calls, session_calls


class CliAndPresetsTests(unittest.TestCase):
    def test_presets_and_flag_defaults(self):
        self.assertEqual(rbb.PRESETS["smoke"]["duration_s"], 60.0)
        self.assertFalse(rbb.PRESETS["smoke"]["input"])
        self.assertEqual(rbb.PRESETS["start-screen"]["duration_s"], 120.0)
        self.assertTrue(rbb.PRESETS["start-screen"]["input"])
        self.assertEqual(rbb.PRESETS["long"]["duration_s"], 600.0)
        self.assertFalse(rbb.PRESETS["long"]["input"])

        parser = rbb.build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.preset, "smoke")
        self.assertTrue(args.headless)
        self.assertFalse(args.smoke)
        self.assertEqual(args.port, 0)

        args_headed = parser.parse_args(["--headed", "--preset", "start-screen", "--capture-frames", "--mp4"])
        self.assertFalse(args_headed.headless)
        self.assertEqual(args_headed.preset, "start-screen")
        self.assertTrue(args_headed.capture_frames)
        self.assertTrue(args_headed.mp4)

    def test_invalid_flag_combinations_return_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for bad_args in (
                ("--rebuild", "--trust-build"),
                ("--rebuild", "--no-build"),
                ("--headless", "--headed"),
                ("--duration-s", "0"),
                ("--duration-s", "-5"),
                ("--port", "70000"),
                ("--frame-interval-s", "0"),
                ("--name", "../escape"),
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")


class DryRunAndStalenessTests(unittest.TestCase):
    def test_smoke_dry_run_requires_zero_game_files_and_sanitizes_output(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "clean_repo"
            (repo / "web").mkdir(parents=True)
            out, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                code = rbb.main(
                    ["--smoke", "--dry-run", "--name", "smoke-check"],
                    chrome_discoverer=lambda *a, **k: None,
                    env={},
                    repo_root=repo,
                )
            self.assertEqual(code, 0, err.getvalue())
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["mode"], "smoke")
            self.assertEqual(payload["plan"], [])
            self.assertEqual(payload["duration_s"], rbb.SMOKE_MODE_DEFAULT_DURATION_S)
            self.assertFalse(payload["chrome_found"])
            self.assertEqual(payload["url"], "http://127.0.0.1:<ephemeral>/?smoke=1")
            # Ensure zero host paths leak in dry-run JSON
            raw_json = out.getvalue()
            self.assertNotRegex(raw_json, r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]")
            self.assertNotIn(str(Path(d)), raw_json)

    def test_nothing_stale_runs_zero_build_commands(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, cmd_calls, session_calls = fx.run_cli(
                "--exe", str(fx.exe), "--game-dir", str(fx.game_dir), "--name", "r1"
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(cmd_calls, [])
            self.assertEqual(len(session_calls), 1)
            self.assertNotIn("[plan]", err)
            self.assertIn("FIRST_PRESENT", out)

    def test_full_staleness_plan_uses_proven_asymmetric_split_and_pic_flags(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            plan, runtime_lib = rbb.make_build_plan(
                repo=fx.repo,
                build_dir=fx.build,
                boot_dir=fx.boot_dir,
                out_dir=fx.out_dir,
                exe=fx.exe,
                game_dir=fx.game_dir,
                runtime_lib_override=None,
                jobs=4,
                force=True,
                smoke=False,
            )
            self.assertEqual(
                [s["stage"] for s in plan.steps],
                ["recomp", "program", "runtime_configure", "runtime_lib", "guest_wasm"],
            )
            prog_cmd = plan.steps[1]["cmd"]
            self.assertIn("build-program", prog_cmd)
            self.assertIn("--target", prog_cmd)
            self.assertIn("wasm64", prog_cmd)
            self.assertIn("--memory-model", prog_cmd)
            self.assertIn("window-runs", prog_cmd)
            self.assertIn("--indirect-dispatch", prog_cmd)
            self.assertIn("shard", prog_cmd)
            self.assertIn("--dll-dir", prog_cmd)

            cfg_cmd = plan.steps[2]["cmd"]
            self.assertEqual(cfg_cmd[:2], ["emcmake", "cmake"])
            self.assertTrue(any("-fPIC" in tok and "-sMEMORY64=1" in tok for tok in cfg_cmd))

            wasm_cmd = plan.steps[4]["cmd"]
            self.assertTrue(wasm_cmd[1].endswith("build_guest_wasm.py"))
            for expected_flag in (
                "--split-modules",
                "--split-strategy",
                "asymmetric",
                "--link-guest-boot",
                "--browser-boot-exports",
                "--proxy-to-pthread",
                "--strip",
                "--no-link-opt",
                "--runtime-lib",
                str(runtime_lib),
            ):
                self.assertIn(expected_flag, wasm_cmd)

    def test_cascading_staleness_recomp_and_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            # Touching recompiler source rebuilds recomp -> program -> guest_wasm (not runtime_lib).
            touch(fx.repo / "recomp" / "src" / "main.rs", fx.base + 100)
            plan, _ = rbb.make_build_plan(
                repo=fx.repo, build_dir=fx.build, boot_dir=fx.boot_dir, out_dir=fx.out_dir,
                exe=fx.exe, game_dir=fx.game_dir, runtime_lib_override=None, jobs=2, force=False, smoke=False,
            )
            self.assertEqual([s["stage"] for s in plan.steps], ["recomp", "program", "guest_wasm"])

            # Reset and touch runtime source -> rebuilds runtime_lib -> guest_wasm only.
            touch(fx.recomp, fx.base + 200)
            touch(fx.manifest, fx.base + 210, "{}")
            touch(fx.repo / "runtime" / "a.cpp", fx.base + 300)
            plan, _ = rbb.make_build_plan(
                repo=fx.repo, build_dir=fx.build, boot_dir=fx.boot_dir, out_dir=fx.out_dir,
                exe=fx.exe, game_dir=fx.game_dir, runtime_lib_override=None, jobs=2, force=False, smoke=False,
            )
            self.assertEqual([s["stage"] for s in plan.steps], ["runtime_lib", "guest_wasm"])

            # Touch build_guest_wasm.py -> rebuilds guest_wasm only.
            touch(fx.runtime_lib, fx.base + 400)
            touch(fx.runtime_wasm, fx.base + 410)
            touch(fx.runtime_js, fx.base + 410)
            touch(fx.repo / "tools" / "build_guest_wasm.py", fx.base + 500)
            plan, _ = rbb.make_build_plan(
                repo=fx.repo, build_dir=fx.build, boot_dir=fx.boot_dir, out_dir=fx.out_dir,
                exe=fx.exe, game_dir=fx.game_dir, runtime_lib_override=None, jobs=2, force=False, smoke=False,
            )
            self.assertEqual([s["stage"] for s in plan.steps], ["guest_wasm"])

    def test_trust_build_and_no_build_and_build_failure(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            touch(fx.exe, fx.base + 500)
            touch(fx.repo / "runtime" / "a.cpp", fx.base + 600)

            # --no-build fails when stale
            code, _, err, cmds, sessions = fx.run_cli(
                "--exe", str(fx.exe), "--name", "nb1", "--no-build"
            )
            self.assertEqual((code, cmds, sessions), (2, [], []))
            self.assertIn("stale", err)

            # --trust-build skips staleness even with --no-build
            code, _, err, cmds, sessions = fx.run_cli(
                "--exe", str(fx.exe), "--name", "tb1", "--no-build", "--trust-build"
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(cmds, [])
            self.assertEqual(len(sessions), 1)
            self.assertNotIn("[plan]", err)

            # Failing build step stops with exit 2
            code, _, err, _, sessions = fx.run_cli(
                "--exe", str(fx.exe), "--name", "fail1", runner=lambda *a: 1
            )
            self.assertEqual((code, sessions), (2, []))
            self.assertIn("failed", err)

    def test_missing_manifest_without_exe_fails_with_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            fx.manifest.unlink()
            code, _, err, _, _ = fx.run_cli("--name", "no-exe")
            self.assertEqual(code, 2)
            self.assertIn("PLAYGTA6_EXE", err)

    def test_shader_pack_and_reflection_sidecar_staging(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            shader_src = Path(d) / "external_shaders"
            pack = shader_src / "game.wgsl.pack"
            sidecar = shader_src / "game.wgsl.pack.reflect.json"
            refl = shader_src / "reflection-manifest.json"
            touch(pack, fx.base, "WGPK")
            touch(sidecar, fx.base, '{"schema":"playgta6-stage-reflection","stages":{}}')
            touch(refl, fx.base, '{"schema_version":1,"shaders":{}}')

            # Dry-run shows shader staging without copying
            code, out, err, _, _ = fx.run_cli(
                "--smoke", "--dry-run", "--shader-pack", str(pack), "--name", "sdry"
            )
            self.assertEqual(code, 0, err)
            dry = json.loads(out)
            names = [s["name"] for s in dry["shader_staging"]]
            self.assertEqual(names, ["game.wgsl.pack", "game.wgsl.pack.reflect.json", "reflection-manifest.json"])

            # Real run stages all three files and passes --shaders dir to session
            code, _, err, _, sessions = fx.run_cli(
                "--smoke", "--shader-pack", str(pack), "--name", "srun"
            )
            self.assertEqual(code, 0, err)
            staged_dir = fx.build / "staged-shaders"
            self.assertEqual(sessions[0]["shaders_dir"], staged_dir)
            for fname in ("game.wgsl.pack", "game.wgsl.pack.reflect.json", "reflection-manifest.json"):
                self.assertTrue((staged_dir / fname).is_file(), fname)


class ChromeDiscoveryTests(unittest.TestCase):
    def test_discovery_priority_and_fallbacks(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            explicit = root / "explicit-chrome"
            env_chrome = root / "env-chrome"
            which_chrome = root / "which-chrome"
            for p in (explicit, env_chrome, which_chrome):
                p.write_text("x", encoding="utf-8")

            # Explicit wins
            self.assertEqual(
                rbb.discover_chrome(explicit, env={"CHROME_PATH": str(env_chrome)}, which_fn=lambda _: str(which_chrome)),
                explicit,
            )
            # Missing explicit returns None (does not silently fall back)
            self.assertIsNone(
                rbb.discover_chrome(root / "missing", env={"CHROME_PATH": str(env_chrome)}, which_fn=lambda _: str(which_chrome))
            )
            # CHROME_PATH wins over which
            self.assertEqual(
                rbb.discover_chrome(None, env={"CHROME_PATH": str(env_chrome)}, which_fn=lambda _: str(which_chrome)),
                env_chrome,
            )
            # which fallback
            self.assertEqual(
                rbb.discover_chrome(None, env={}, which_fn=lambda n: str(which_chrome) if "chrome" in n else None, platform="linux"),
                which_chrome,
            )
            # Standard path fallback (Linux & Windows)
            self.assertEqual(
                rbb.discover_chrome(
                    None, env={}, which_fn=lambda _: None, platform="linux",
                    exists_fn=lambda p: p.as_posix() == "/usr/bin/chromium",
                ),
                Path("/usr/bin/chromium"),
            )
            win_edge = Path(r"C:\Program Files") / "Microsoft" / "Edge" / "Application" / "msedge.exe"
            self.assertEqual(
                rbb.discover_chrome(
                    None, env={"PROGRAMFILES": r"C:\Program Files"}, which_fn=lambda _: None, platform="win32",
                    exists_fn=lambda p: p == win_edge,
                ),
                win_edge,
            )
            # Not found returns None -> main returns exit code 2
            self.assertIsNone(
                rbb.discover_chrome(None, env={}, which_fn=lambda _: None, platform="linux", exists_fn=lambda _: False)
            )

    def test_missing_chrome_in_live_run_returns_exit_2(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, _, err, _, _ = fx.run_cli(
                "--smoke", "--name", "nochrome", chrome_discoverer=lambda *a, **k: None
            )
            self.assertEqual(code, 2)
            self.assertIn("Chrome", err)


class ReportAndBaselineTests(unittest.TestCase):
    def test_run_artifacts_sanitization_and_compare_boot_reports_parity(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            dirty_session = make_fake_session_result(
                console_lines=[
                    r"[   0.10s] [console.log] loaded C:\Users\secretuser\Games\title.exe",
                    r"[   0.20s] [console.warn] network path \\fileserver\share\dump.bin",
                    "[   0.30s] [console.log] posix /home/alice/local/file.rpf",
                ],
                probe=make_sample_probe(
                    boot_mode="smoke-demo",
                    presents=4,
                    fallback_draws=4,
                    hud_log=[r"opened C:\Users\secretuser\work\guest.pgim"],
                ),
            )
            code, out, err, _, _ = fx.run_cli(
                "--smoke", "--name", "base", session_runner=lambda **kw: dirty_session
            )
            self.assertEqual(code, 0, err)
            run_dir = fx.build / "runs" / "base"
            for fname in ("boot-report.json", "console.log", "telemetry.json", "command.json", "digest.txt"):
                self.assertTrue((run_dir / fname).is_file(), fname)
                text = (run_dir / fname).read_text(encoding="utf-8")
                self.assertNotRegex(text, r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\]", f"host Windows path leaked in {fname}")
                self.assertNotIn("/home/alice", text, f"host POSIX path leaked in {fname}")
                self.assertNotIn("\\\\fileserver", text, f"UNC path leaked in {fname}")

            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], compare_boot_reports.REPORT_SCHEMA)
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual([m["milestone"] for m in report["milestones"]], ["PGIM_LOAD", "FIRST_PRESENT"])
            self.assertIn("FIRST_PRESENT", out)
            self.assertIn("fallback_draws=4", out)

            # Matching baseline returns 0
            code, cmp_out, cmp_err, _, _ = fx.run_cli(
                "--smoke", "--name", "same", "--baseline", "base",
                session_runner=lambda **kw: dirty_session,
            )
            self.assertEqual(code, 0, cmp_err)
            self.assertIn("PASS", cmp_out)

            # Different milestone/fault against baseline returns 1
            fault_session = make_fake_session_result(
                status="fault",
                probe=make_sample_probe(
                    stage="ENTRY_CALL",
                    milestones=[{"milestone": "ENTRY_CALL", "index": 3, "reached": True, "elapsedMs": 10.0}],
                    last_trap={
                        "kind": "UNIMPLEMENTED_OPCODE",
                        "guestVa": "0x140001000",
                        "message": "unimplemented opcode at C:\\Users\\bob\\game.exe",
                        "fatal": True,
                    },
                ),
            )
            code, diff_out, _, _, _ = fx.run_cli(
                "--smoke", "--name", "diff", "--baseline", "base",
                session_runner=lambda **kw: fault_session,
            )
            self.assertEqual(code, 1)
            self.assertIn("FAIL", diff_out)
            self.assertIn("stop.status", diff_out)

            # Missing baseline returns 2
            code, _, miss_err, _, _ = fx.run_cli("--smoke", "--name", "nobase", "--baseline", "does-not-exist")
            self.assertEqual(code, 2)
            self.assertIn("baseline run not found", miss_err)

    def test_existing_run_dir_returns_2_and_missing_report_returns_3(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            crashed = make_fake_session_result(ok=False, status="error", exit_code=1, probe={}, presents=[])
            code, _, err, _, _ = fx.run_cli("--smoke", "--name", "crash1", session_runner=lambda **kw: crashed)
            self.assertEqual(code, 3)
            self.assertIn("wrote no report", err)

            # Reusing same non-empty run name returns 2
            code, _, err, _, _ = fx.run_cli("--smoke", "--name", "crash1")
            self.assertEqual(code, 2)
            self.assertIn("already exists", err)


class ServerAndCdpLifecycleTests(unittest.TestCase):
    def test_wait_for_server_ready_validates_coop_coep_and_premature_exit(self):
        class GoodHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = json.dumps({"ok": True, "coopCoep": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cross-Origin-Opener-Policy", "same-origin")
                self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return

        srv = ThreadingHTTPServer(("127.0.0.1", 0), GoodHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            ready = rbb.wait_for_server_ready("127.0.0.1", port, timeout_s=3.0)
            self.assertEqual(ready["status_code"], 200)
            self.assertEqual(ready["coop"], "same-origin")
            self.assertEqual(ready["coep"], "require-corp")
            self.assertTrue(ready["api_status"]["ok"])
        finally:
            srv.shutdown()
            srv.server_close()

        # Server missing COOP/COEP headers times out cleanly
        class BadHeaderHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                return

        bad_srv = ThreadingHTTPServer(("127.0.0.1", 0), BadHeaderHandler)
        bad_port = bad_srv.server_address[1]
        th2 = threading.Thread(target=bad_srv.serve_forever, daemon=True)
        th2.start()
        try:
            with self.assertRaises(rbb.RunnerError) as ctx:
                rbb.wait_for_server_ready("127.0.0.1", bad_port, timeout_s=0.4)
            self.assertIn("COOP/COEP", str(ctx.exception))
        finally:
            bad_srv.shutdown()
            bad_srv.server_close()

        # Dead subprocess fails fast
        class DeadProc:
            returncode = 17

            def poll(self):
                return 17

        with self.assertRaises(rbb.RunnerError) as ctx:
            rbb.wait_for_server_ready("127.0.0.1", rbb.find_free_port(), timeout_s=3.0, proc=DeadProc())
        self.assertIn("exited prematurely", str(ctx.exception))

    def test_pure_python_cdp_websocket_client_handshake_events_and_ping_pong(self):
        """Exercise real TCP RFC 6455 handshake, masked frames, ping/pong, and CDP events."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]
        received_methods: list[str] = []

        def recv_client_frame(conn: socket.socket) -> tuple[int, bytes]:
            def rx(n: int) -> bytes:
                buf = bytearray()
                while len(buf) < n:
                    chunk = conn.recv(n - len(buf))
                    if not chunk:
                        raise EOFError()
                    buf.extend(chunk)
                return bytes(buf)

            b0, b1 = rx(2)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack("!H", rx(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", rx(8))[0]
            mask_key = rx(4) if masked else b""
            payload = rx(length) if length else b""
            if masked:
                payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
            return opcode, payload

        def mock_ws_server():
            conn, _ = server_sock.accept()
            try:
                raw = b""
                while b"\r\n\r\n" not in raw:
                    raw += conn.recv(4096)
                m = re.search(rb"Sec-WebSocket-Key:\s*(\S+)", raw, re.I)
                assert m is not None
                accept = base64.b64encode(
                    hashlib.sha1(m.group(1) + rbb.WEBSOCKET_GUID.encode("ascii")).digest()
                ).decode("ascii")
                conn.sendall(
                    (
                        "HTTP/1.1 101 Switching Protocols\r\n"
                        "Upgrade: websocket\r\n"
                        "Connection: Upgrade\r\n"
                        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
                    ).encode("ascii")
                )
                while True:
                    try:
                        opcode, payload = recv_client_frame(conn)
                    except EOFError:
                        break
                    if opcode == 0x8:
                        break
                    if opcode == 0xA:
                        continue
                    req = json.loads(payload.decode("utf-8"))
                    received_methods.append(req["method"])
                    # Send a PING + an async console event before the reply
                    conn.sendall(rbb._ws_encode_frame(0x9, b"hi", mask=False))
                    ev = json.dumps(
                        {
                            "method": "Runtime.consoleAPICalled",
                            "params": {"type": "log", "args": [{"value": f"seen {req['method']}"}]},
                        }
                    ).encode("utf-8")
                    conn.sendall(rbb._ws_encode_frame(0x1, ev, mask=False))
                    reply = json.dumps({"id": req["id"], "result": {"ok": True}}).encode("utf-8")
                    conn.sendall(rbb._ws_encode_frame(0x1, reply, mask=False))
            finally:
                conn.close()
                server_sock.close()

        th = threading.Thread(target=mock_ws_server, daemon=True)
        th.start()
        client = rbb.CdpClient(f"ws://127.0.0.1:{port}/devtools/page/1", timeout=5.0)
        try:
            client.connect()
            res = client.send("Runtime.enable")
            self.assertEqual(res, {"ok": True})
            events = client.drain_events(timeout=0.05)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["method"], "Runtime.consoleAPICalled")
        finally:
            client.close()
            th.join(timeout=3.0)
        self.assertEqual(received_methods, ["Runtime.enable"])

    def test_run_browser_session_lifecycle_capture_mp4_and_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "sess1"
            run_dir.mkdir(parents=True)

            procs: list[Any] = []
            created_profiles: list[Path] = []

            class FakeProc:
                def __init__(self, cmd):
                    self.cmd = list(cmd)
                    self.terminated = False
                    self.returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.terminated = True
                    self.returncode = 0

                def wait(self, timeout=None):
                    self.returncode = 0
                    return 0

                def kill(self):
                    self.returncode = -9

            def fake_popen(cmd, **kwargs):
                for tok in cmd:
                    if isinstance(tok, str) and tok.startswith("--user-data-dir="):
                        created_profiles.append(Path(tok.split("=", 1)[1]))
                p = FakeProc(cmd)
                procs.append(p)
                return p

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url
                    self.methods: list[str] = []
                    self.closed = False
                    self.eval_count = 0

                def connect(self):
                    return None

                def close(self):
                    self.closed = True

                def send(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None):
                    self.methods.append(method)
                    if method == "Runtime.evaluate":
                        self.eval_count += 1
                        return {
                            "result": {
                                "value": make_sample_probe(presents=self.eval_count, fallback_draws=self.eval_count)
                            }
                        }
                    if method == "Page.captureScreenshot":
                        return {"data": base64.b64encode(b"\xff\xd8\xff\xd9").decode("ascii")}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    if self.eval_count == 0:
                        return [
                            {
                                "method": "Runtime.consoleAPICalled",
                                "params": {"type": "log", "args": [{"value": "hello from browser"}]},
                            }
                        ]
                    return []

            cdp_instances: list[FakeCdp] = []

            def cdp_factory(ws_url: str):
                inst = FakeCdp(ws_url)
                cdp_instances.append(inst)
                return inst

            ffmpeg_calls: list[list[str]] = []

            def fake_runner(cmd, cwd, log, timeout):
                ffmpeg_calls.append(list(cmd))
                # Write dummy mp4 at last arg
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            res = rbb.run_browser_session(
                repo=fx.repo,
                run_dir=run_dir,
                chrome_path=fx.chrome,
                port=0,
                boot_dir=fx.boot_dir,
                game_dir=None,
                shaders_dir=None,
                duration_s=0.35,
                headless=True,
                smoke=True,
                send_input=False,
                capture_frames=True,
                frame_interval_s=0.1,
                encode_mp4=True,
                runner=fake_runner,
                popen_fn=fake_popen,
                cdp_factory=cdp_factory,
                wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                which_fn=lambda name: "ffmpeg" if name == "ffmpeg" else None,
            )

            self.assertTrue(res.ok)
            self.assertGreaterEqual(len(res.captured_frames), 1)
            self.assertTrue((run_dir / "frames" / "frame_0001.jpg").is_file())
            self.assertEqual(res.mp4_path, "capture.mp4")
            self.assertEqual(len(ffmpeg_calls), 1)
            self.assertIn("Page.navigate", cdp_instances[0].methods)
            self.assertIn("Page.captureScreenshot", cdp_instances[0].methods)
            # Cleanup verified
            self.assertTrue(cdp_instances[0].closed)
            self.assertEqual(len(procs), 2)
            self.assertTrue(all(p.terminated for p in procs))
            self.assertEqual(len(created_profiles), 1)
            self.assertFalse(created_profiles[0].exists())

    def test_run_browser_session_cleans_up_on_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "kb1"
            run_dir.mkdir(parents=True)
            procs: list[Any] = []
            profiles: list[Path] = []

            class FakeProc:
                terminated = False
                returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.terminated = True
                    self.returncode = 0

                def wait(self, timeout=None):
                    return 0

                def kill(self):
                    self.returncode = -9

            def fake_popen(cmd, **kwargs):
                for tok in cmd:
                    if isinstance(tok, str) and tok.startswith("--user-data-dir="):
                        profiles.append(Path(tok.split("=", 1)[1]))
                p = FakeProc()
                procs.append(p)
                return p

            class InterruptCdp:
                closed = False

                def __init__(self, ws_url: str):
                    pass

                def connect(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Page.navigate":
                        raise KeyboardInterrupt()
                    return {}

                def drain_events(self, timeout=0.0):
                    return []

                def close(self):
                    self.closed = True

            cdp_holder: list[InterruptCdp] = []

            def make_cdp(ws_url: str):
                c = InterruptCdp(ws_url)
                cdp_holder.append(c)
                return c

            with self.assertRaises(KeyboardInterrupt):
                rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=5.0,
                    headless=True,
                    smoke=True,
                    send_input=False,
                    capture_frames=False,
                    frame_interval_s=1.0,
                    encode_mp4=False,
                    popen_fn=fake_popen,
                    cdp_factory=make_cdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                )

            self.assertTrue(cdp_holder[0].closed)
            self.assertEqual(len(procs), 2)
            self.assertTrue(all(p.terminated for p in procs))
            self.assertEqual(len(profiles), 1)
            self.assertFalse(profiles[0].exists())

    def test_encode_frames_to_mp4_graceful_when_ffmpeg_missing(self):
        with tempfile.TemporaryDirectory() as d:
            frames_dir = Path(d) / "frames"
            frames_dir.mkdir()
            (frames_dir / "frame_0001.jpg").write_bytes(b"\xff\xd8\xff\xd9")
            ok, warn = rbb.encode_frames_to_mp4(
                frames_dir, Path(d) / "capture.mp4", which_fn=lambda _: None
            )
            self.assertFalse(ok)
            self.assertIn("ffmpeg not found", warn or "")


class BootFlagsAndOptionsTests(unittest.TestCase):
    def test_new_flag_defaults_preserve_backwards_compatibility(self):
        parser = rbb.build_arg_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.input_script)
        self.assertFalse(args.stream_shaders)
        self.assertIsNone(args.stream_threshold_mb)
        self.assertIsNone(args.boot_options)
        self.assertFalse(args.report_to_log)
        self.assertIsNone(args.max_wall_ms)

        self.assertEqual(rbb.build_target_url("127.0.0.1", 8080), "http://127.0.0.1:8080/")
        self.assertEqual(rbb.build_target_url("127.0.0.1", 8080, smoke=True), "http://127.0.0.1:8080/?smoke=1")

    def test_input_script_and_stream_shader_url_building_and_validation(self):
        self.assertEqual(rbb.sanitize_input_script_preset("start-screen"), "start-screen")
        self.assertEqual(rbb.sanitize_input_script_preset("  START-SCREEN  "), "start-screen")
        for bad in ("", "   ", "unknown", "../escape", "start-screen&evil=1", "start-screen\nx", 123, None):
            self.assertIsNone(rbb.sanitize_input_script_preset(bad), f"expected None for {bad!r}")

        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, input_script="START-SCREEN"),
            "http://127.0.0.1:8080/?inputScript=start-screen",
        )
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, stream_shaders=True),
            "http://127.0.0.1:8080/?streamShaders=1",
        )
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, stream_threshold_mb=32.0),
            "http://127.0.0.1:8080/?streamThresholdMb=32",
        )
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, stream_threshold_mb=0.00001),
            "http://127.0.0.1:8080/?streamThresholdMb=0.00001",
        )
        self.assertEqual(
            rbb.build_target_url(
                "127.0.0.1",
                8080,
                smoke=True,
                input_script="start-screen",
                stream_shaders=True,
                stream_threshold_mb=16.5,
            ),
            "http://127.0.0.1:8080/?smoke=1&inputScript=start-screen&streamShaders=1&streamThresholdMb=16.5",
        )

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for bad_args in (
                ("--input-script", ""),
                ("--input-script", "not-allowed"),
                ("--input-script", "../escape"),
                ("--input-script", "start-screen&foo=1"),
                ("--stream-threshold-mb", "0"),
                ("--stream-threshold-mb", "-4.5"),
                ("--stream-threshold-mb", "nan"),
                ("--stream-threshold-mb", "inf"),
                ("--stream-threshold-mb", "-inf"),
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")

    def test_dry_run_includes_url_query_params_and_boot_options_without_writing(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "flags-dry",
                "--input-script",
                "START-SCREEN",
                "--stream-shaders",
                "--stream-threshold-mb",
                "24.5",
                "--boot-options",
                '{"arena_log_limit": 2048}',
                "--report-to-log",
                "--max-wall-ms",
                "45000",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=start-screen&streamShaders=1&streamThresholdMb=24.5",
            )
            self.assertEqual(payload["input_script"], "start-screen")
            self.assertEqual(payload["inputScript"], "start-screen")
            self.assertTrue(payload["stream_shaders"])
            self.assertEqual(payload["stream_threshold_mb"], 24.5)
            self.assertEqual(
                payload["boot_options"],
                {"arena_log_limit": 2048, "max_wall_ms": 45000, "report_to_log": True},
            )
            self.assertEqual(payload["boot_options_staging"]["action"], "plan_ephemeral")
            self.assertFalse(payload["boot_options_staging"]["staged"])
            # Dry-run must never write boot-options.json to boot_dir
            self.assertFalse((fx.boot_dir / "boot-options.json").exists())

    def test_boot_options_file_validation_errors_and_tracked_dir_guard(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            # Load from a UTF-8-BOM JSON file and override with CLI helpers
            opts_file = Path(d) / "external" / "2026-boot-opts.json"
            opts_file.write_text(
                "\ufeff" + json.dumps({"max_wall_ms": 10000, "verify_window_marks": True}),
                encoding="utf-8",
            )
            code, out, err, _, _ = fx.run_cli(
                "--trust-build",
                "--dry-run",
                "--name",
                "opts-file",
                "--boot-options",
                str(opts_file),
                "--report-to-log",
                "--max-wall-ms",
                "90000",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertEqual(
                payload["boot_options"],
                {"max_wall_ms": 90000, "report_to_log": True, "verify_window_marks": True},
            )
            self.assertEqual(payload["boot_options_staging"]["action"], "plan_write")
            self.assertFalse((fx.boot_dir / "boot-options.json").exists())
            # Zero host paths leaked in dry-run JSON
            self.assertNotIn(str(Path(d)), out)

            # Reject malformed / invalid boot-options inputs
            bad_json_file = Path(d) / "external" / "bad.json"
            bad_json_file.write_text("{not valid json", encoding="utf-8")
            for bad_args in (
                ("--boot-options", ""),
                ("--boot-options", "{bad"),
                ("--boot-options", "[1, 2, 3]"),
                ("--boot-options", "123"),
                ("--boot-options", '{"unknown_key": 1}'),
                ("--boot-options", '{"max_wall_ms": true}'),
                ("--boot-options", '{"max_wall_ms": -5}'),
                ("--boot-options", '{"report_to_log": "yes"}'),
                ("--boot-options", str(Path(d) / "external" / "missing.json")),
                ("--boot-options", str(bad_json_file)),
                ("--max-wall-ms", "-1"),
                ("--max-wall-ms", str(1 << 64)),
            ):
                code, _, err_text, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")
                self.assertNotIn(str(Path(d)), err_text)

            # Refuse writing boot-options.json into tracked repository directories (web/, tools/, repo root)
            for tracked_dir in (fx.repo / "web", fx.repo / "tools", fx.repo):
                code, _, err_text, _, _ = fx.run_cli(
                    "--trust-build",
                    "--dry-run",
                    "--boot-dir",
                    str(tracked_dir),
                    "--report-to-log",
                )
                self.assertEqual(code, 2, f"expected refusal for tracked dir {tracked_dir}")
                self.assertIn("tracked repository directory", err_text)
                self.assertFalse((tracked_dir / "boot-options.json").exists())

    def test_live_boot_options_staging_in_gitignored_boot_dir_and_smoke_ephemeral(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            # Real-game run stages boot-options.json into gitignored work/recomp_build
            code, _, err, _, sessions = fx.run_cli(
                "--trust-build",
                "--name",
                "real-opts",
                "--boot-options",
                '{"max_dispatches": 5000}',
                "--report-to-log",
                "--max-wall-ms",
                "60000",
                "--input-script",
                "start-screen",
                "--stream-shaders",
                "--stream-threshold-mb",
                "32",
            )
            self.assertEqual(code, 0, err)
            staged_path = fx.boot_dir / "boot-options.json"
            self.assertTrue(staged_path.is_file())
            self.assertEqual(
                json.loads(staged_path.read_text(encoding="utf-8")),
                {"max_dispatches": 5000, "max_wall_ms": 60000, "report_to_log": True},
            )
            self.assertEqual(sessions[0]["input_script"], "start-screen")
            self.assertTrue(sessions[0]["stream_shaders"])
            self.assertEqual(sessions[0]["stream_threshold_mb"], 32.0)
            self.assertEqual(
                sessions[0]["boot_options"],
                {"max_dispatches": 5000, "max_wall_ms": 60000, "report_to_log": True},
            )

            # Remove staged file and verify smoke run does NOT touch repo boot_dir
            staged_path.unlink()
            run_dir = fx.build / "runs" / "smoke-ephem"
            run_dir.mkdir(parents=True)
            navigated_urls: list[str] = []
            observed_smoke_boot_opts: list[dict[str, Any]] = []

            class FakeProc:
                returncode = None

                def poll(self):
                    return self.returncode

                def terminate(self):
                    self.returncode = 0

                def wait(self, timeout=None):
                    return 0

                def kill(self):
                    self.returncode = -9

            def fake_popen(cmd, **kwargs):
                if "--boot" in cmd:
                    boot_arg = Path(cmd[cmd.index("--boot") + 1])
                    opts_in_boot = boot_arg / "boot-options.json"
                    if opts_in_boot.is_file():
                        observed_smoke_boot_opts.append(json.loads(opts_in_boot.read_text(encoding="utf-8")))
                return FakeProc()

            class FakeCdp:
                def __init__(self, ws_url: str):
                    pass

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Page.navigate" and isinstance(params, dict):
                        navigated_urls.append(str(params.get("url") or ""))
                    if method == "Runtime.evaluate":
                        probe = make_sample_probe(presents=2, real_draws=7, fallback_draws=1, validation_errors=2)
                        probe["bootUrlOptions"] = {"inputScript": "start-screen"}
                        return {"result": {"value": probe}}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    return []

            res = rbb.run_browser_session(
                repo=fx.repo,
                run_dir=run_dir,
                chrome_path=fx.chrome,
                port=0,
                boot_dir=None,
                game_dir=None,
                shaders_dir=None,
                duration_s=0.15,
                headless=True,
                smoke=True,
                send_input=False,
                capture_frames=False,
                frame_interval_s=1.0,
                encode_mp4=False,
                input_script="start-screen",
                stream_shaders=True,
                stream_threshold_mb=48.0,
                boot_options={"report_to_log": True, "max_wall_ms": 15000},
                popen_fn=fake_popen,
                cdp_factory=FakeCdp,
                wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
            )
            self.assertTrue(res.ok)
            self.assertEqual(len(navigated_urls), 1)
            self.assertIn("smoke=1", navigated_urls[0])
            self.assertIn("inputScript=start-screen", navigated_urls[0])
            self.assertIn("streamShaders=1", navigated_urls[0])
            self.assertIn("streamThresholdMb=48", navigated_urls[0])
            self.assertEqual(observed_smoke_boot_opts, [{"max_wall_ms": 15000, "report_to_log": True}])
            # Smoke mode must never touch repo boot_dir
            self.assertFalse(staged_path.exists())

    def test_report_and_telemetry_summary_fields_and_input_script_sanitization(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            probe = make_sample_probe(
                boot_mode="real-wasm64",
                session_state="running_wasm",
                presents=6,
                real_draws=14,
                fallback_draws=2,
                validation_errors=1,
            )
            # Untrusted probe value should be ignored in favor of sanitized allowlisted value
            probe["inputScriptState"] = {"active": True, "preset": r"C:\Users\evil\script"}
            probe["bootUrlOptions"] = {"inputScript": "START-SCREEN"}

            session = make_fake_session_result(probe=probe)
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "summary-check",
                "--input-script",
                "start-screen",
                "--stream-shaders",
                "--stream-threshold-mb",
                "16",
                "--report-to-log",
                session_runner=lambda **kw: session,
            )
            self.assertEqual(code, 0, err)
            run_dir = fx.build / "runs" / "summary-check"
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))

            expected_summary = {
                "realPackDrawCount": 14,
                "fallbackDrawCount": 2,
                "validationErrorCount": 1,
                "inputScript": "start-screen",
                "streamShaders": True,
                "streamThresholdMb": 16.0,
            }
            self.assertEqual(report["summary"], expected_summary)
            self.assertEqual(report["inputScript"], "start-screen")
            self.assertEqual(report["counters"]["real_pack_draws"], 14)
            self.assertEqual(report["counters"]["fallback_draws"], 2)
            self.assertEqual(report["counters"]["gpu_validation_errors"], 1)
            self.assertEqual(report["boot_options"], {"report_to_log": True})

            self.assertEqual(telemetry["summary"], expected_summary)
            self.assertEqual(telemetry["inputScript"], "start-screen")
            self.assertTrue(telemetry["stream_shaders"])
            self.assertEqual(telemetry["stream_threshold_mb"], 16.0)
            self.assertEqual(telemetry["boot_options"], {"report_to_log": True})

            self.assertEqual(cmd_meta["input_script"], "start-screen")
            self.assertEqual(cmd_meta["inputScript"], "start-screen")
            self.assertTrue(cmd_meta["stream_shaders"])
            self.assertEqual(cmd_meta["stream_threshold_mb"], 16.0)
            self.assertEqual(cmd_meta["boot_options"], {"report_to_log": True})
            self.assertIn("input_script: start-screen", out)

            # Verify compare_boot_reports compatibility remains intact
            code2, cmp_out, cmp_err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "summary-check-2",
                "--baseline",
                "summary-check",
                "--input-script",
                "start-screen",
                session_runner=lambda **kw: session,
            )
            self.assertEqual(code2, 0, cmp_err)
            self.assertIn("PASS", cmp_out)

            # Verify untrusted probe inputScript with no CLI fallback resolves cleanly to None
            bad_probe = make_sample_probe(real_draws=3, fallback_draws=0, validation_errors=0)
            bad_probe["href"] = "http://127.0.0.1:8123/?inputScript=../escape"
            bad_probe["bootState"]["inputScript"] = "evil-script"
            rep_none = rbb.build_boot_report(
                make_fake_session_result(probe=bad_probe),
                smoke=True,
                preset="smoke",
            )
            self.assertIsNone(rep_none["inputScript"])
            self.assertIsNone(rep_none["summary"]["inputScript"])
            self.assertEqual(rep_none["summary"]["realPackDrawCount"], 3)
            self.assertEqual(rep_none["summary"]["fallbackDrawCount"], 0)
            self.assertEqual(rep_none["summary"]["validationErrorCount"], 0)


def _make_synthetic_rgb_png(
    width: int,
    height: int,
    pixel_fn,
) -> bytes:
    """Build a valid 8-bit RGB PNG using only Python stdlib (no Pillow required)."""
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
        return struct.pack("!I", len(data)) + chunk_type + data + struct.pack("!I", crc)

    raw_rows = bytearray()
    for y in range(height):
        raw_rows.append(0)  # filter type 0 (None)
        for x in range(width):
            r, g, b = pixel_fn(x, y)
            raw_rows.extend((r & 0xFF, g & 0xFF, b & 0xFF))

    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", zlib.compress(bytes(raw_rows)))
        + _chunk(b"IEND", b"")
    )


class _FakePilImage:
    """Pure-Python PIL.Image stand-in that inspects synthetic PNG payloads without Pillow."""

    def __init__(self, width: int, height: int, pixels: list[tuple[int, int, int]]) -> None:
        self.size = (width, height)
        self._pixels = pixels

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def convert(self, mode: str):
        assert mode == "RGB"
        return self

    def crop(self, box: tuple[int, int, int, int]):
        left, top, right, bottom = box
        w, _h = self.size
        cropped: list[tuple[int, int, int]] = []
        for y in range(top, bottom):
            row_start = y * w
            for x in range(left, right):
                cropped.append(self._pixels[row_start + x])
        return _FakePilImage(right - left, bottom - top, cropped)

    def getextrema(self):
        if not self._pixels:
            return ((0, 0), (0, 0), (0, 0))
        rs = [p[0] for p in self._pixels]
        gs = [p[1] for p in self._pixels]
        bs = [p[2] for p in self._pixels]
        return ((min(rs), max(rs)), (min(gs), max(gs)), (min(bs), max(bs)))


class _FakePilModule:
    """Minimal PIL.Image module replacement decoding uncompressed/zlib 8-bit RGB PNGs."""

    @staticmethod
    def open(buf: io.BytesIO) -> _FakePilImage:
        data = buf.read()
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise OSError("not a PNG")
        pos = 8
        width = height = 0
        idat = bytearray()
        while pos + 8 <= len(data):
            length, chunk_type = struct.unpack("!I4s", data[pos : pos + 8])
            pos += 8
            payload = data[pos : pos + length]
            pos += length + 4  # skip CRC
            if chunk_type == b"IHDR":
                width, height = struct.unpack("!II", payload[:8])
            elif chunk_type == b"IDAT":
                idat.extend(payload)
            elif chunk_type == b"IEND":
                break
        raw = zlib.decompress(bytes(idat))
        stride = 1 + width * 3
        pixels: list[tuple[int, int, int]] = []
        for y in range(height):
            row = raw[y * stride + 1 : (y + 1) * stride]
            for x in range(width):
                off = x * 3
                pixels.append((row[off], row[off + 1], row[off + 2]))
        return _FakePilImage(width, height, pixels)


class CaptureFramesDumpPipelinesAndPresetsTests(unittest.TestCase):
    def test_t43_flag_defaults_and_validation(self):
        parser = rbb.build_arg_parser()
        args = parser.parse_args([])
        self.assertIsNone(args.capture_frames)
        self.assertEqual(args.capture_interval_sec, 2.0)
        self.assertFalse(args.dump_pipelines)

        # Bare --capture-frames preserves legacy boolean behavior; with DIR captures path
        bare = parser.parse_args(["--capture-frames"])
        self.assertIs(bare.capture_frames, True)
        with_dir = parser.parse_args(["--capture-frames", "work/frames", "--capture-interval-sec", "0.5", "--dump-pipelines"])
        self.assertEqual(with_dir.capture_frames, "work/frames")
        self.assertEqual(with_dir.capture_interval_sec, 0.5)
        self.assertTrue(with_dir.dump_pipelines)

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for bad_args in (
                ("--capture-interval-sec", "0"),
                ("--capture-interval-sec", "-2.5"),
                ("--capture-interval-sec", "nan"),
                ("--capture-interval-sec", "inf"),
                ("--capture-interval-sec", "-inf"),
                ("--capture-frames", ""),
                ("--capture-frames", "   "),
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")

            # Tracked repository directories are refused for --capture-frames (same guard as boot-options)
            for tracked_dir in (fx.repo / "web", fx.repo / "tools", fx.repo):
                code, _, err_text, _, _ = fx.run_cli(
                    "--smoke",
                    "--dry-run",
                    "--capture-frames",
                    str(tracked_dir),
                )
                self.assertEqual(code, 2, f"expected tracked dir refusal for {tracked_dir}")
                self.assertIn("tracked repository directory", err_text)

            # Passing a regular file path as --capture-frames fails cleanly
            file_path = fx.repo / "work" / "not-a-dir.txt"
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text("hi", encoding="utf-8")
            code, _, err_text, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--capture-frames",
                str(file_path),
            )
            self.assertEqual(code, 2)
            self.assertIn("not a directory", err_text)

    def test_input_script_accepts_new_presets(self):
        for preset in ("start-screen", "menu-flow", "calibration-skip", "intro-watch"):
            self.assertEqual(rbb.sanitize_input_script_preset(preset), preset)
            self.assertEqual(rbb.sanitize_input_script_preset(f"  {preset.upper()}  "), preset)
            self.assertEqual(
                rbb.build_target_url("127.0.0.1", 8080, input_script=preset),
                f"http://127.0.0.1:8080/?inputScript={preset}",
            )

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for preset in ("menu-flow", "calibration-skip", "intro-watch"):
                code, out, err, _, _ = fx.run_cli(
                    "--smoke",
                    "--dry-run",
                    "--name",
                    f"dry-{preset}",
                    "--input-script",
                    preset,
                )
                self.assertEqual(code, 0, err)
                payload = json.loads(out)
                self.assertEqual(payload["input_script"], preset)
                self.assertEqual(payload["inputScript"], preset)
                self.assertIn(f"inputScript={preset}", payload["url"])

    def test_dump_pipelines_dry_run_live_staging_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            # Dry-run records dumpPipelines: true without writing boot-options.json
            code, out, err, _, _ = fx.run_cli(
                "--trust-build",
                "--dry-run",
                "--name",
                "dump-dry",
                "--dump-pipelines",
                "--report-to-log",
            )
            self.assertEqual(code, 0, err)
            dry = json.loads(out)
            self.assertTrue(dry["dumpPipelines"])
            self.assertTrue(dry["dump_pipelines"])
            self.assertEqual(
                dry["boot_options"],
                {"dump_pipelines_dir": "/pg6_dump", "report_to_log": True},
            )
            self.assertFalse(dry["boot_options_staging"]["staged"])
            self.assertEqual(dry["boot_options_staging"]["action"], "plan_write")
            self.assertFalse((fx.boot_dir / "boot-options.json").exists())

            # Tracked boot-dir is refused when --dump-pipelines is set
            code, _, err_text, _, _ = fx.run_cli(
                "--trust-build",
                "--dry-run",
                "--boot-dir",
                str(fx.repo / "web"),
                "--dump-pipelines",
            )
            self.assertEqual(code, 2)
            self.assertIn("tracked repository directory", err_text)

            # Live real-game run stages dump_pipelines_dir into boot-options.json and records dumpPipelines
            code, _, err, _, sessions = fx.run_cli(
                "--trust-build",
                "--name",
                "dump-live",
                "--dump-pipelines",
            )
            self.assertEqual(code, 0, err)
            staged_path = fx.boot_dir / "boot-options.json"
            self.assertTrue(staged_path.is_file())
            self.assertEqual(
                json.loads(staged_path.read_text(encoding="utf-8")),
                {"dump_pipelines_dir": "/pg6_dump"},
            )
            self.assertEqual(sessions[0]["boot_options"], {"dump_pipelines_dir": "/pg6_dump"})

            run_dir = fx.build / "runs" / "dump-live"
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            self.assertTrue(cmd_meta["dumpPipelines"])
            self.assertTrue(cmd_meta["dump_pipelines"])
            self.assertEqual(cmd_meta["boot_options"], {"dump_pipelines_dir": "/pg6_dump"})
            self.assertEqual(report["boot_options"], {"dump_pipelines_dir": "/pg6_dump"})
            self.assertEqual(telemetry["boot_options"], {"dump_pipelines_dir": "/pg6_dump"})

            # Reject invalid dump_pipelines_dir values passed via --boot-options
            for bad_json in (
                '{"dump_pipelines_dir": ""}',
                '{"dump_pipelines_dir": 123}',
                '{"dump_pipelines_dir": "../escape"}',
                '{"dump_pipelines_dir": "C:\\\\dump"}',
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", "--boot-options", bad_json)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_json}")

    def test_check_canvas_non_black_pure_python_mock_and_optional_pillow(self):
        all_black = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 0, 0))
        sidebar_only = _make_synthetic_rgb_png(420, 4, lambda x, y: (255, 255, 255) if x < 410 else (0, 0, 0))
        canvas_lit = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 200, 50) if x == 410 and y == 1 else (0, 0, 0))
        too_narrow = _make_synthetic_rgb_png(410, 4, lambda x, y: (255, 255, 255))

        # Pure-Python mock PIL module (always runs regardless of whether Pillow is installed)
        self.assertFalse(rbb.check_canvas_non_black(all_black, pil_image_mod=_FakePilModule, _auto_import_pil=False))
        self.assertFalse(rbb.check_canvas_non_black(sidebar_only, pil_image_mod=_FakePilModule, _auto_import_pil=False))
        self.assertTrue(rbb.check_canvas_non_black(canvas_lit, pil_image_mod=_FakePilModule, _auto_import_pil=False))
        self.assertFalse(rbb.check_canvas_non_black(too_narrow, pil_image_mod=_FakePilModule, _auto_import_pil=False))
        self.assertIsNone(rbb.check_canvas_non_black(b"not-a-png", pil_image_mod=_FakePilModule, _auto_import_pil=False))

        # Graceful skip when Pillow is unavailable
        self.assertIsNone(rbb.check_canvas_non_black(canvas_lit, pil_image_mod=None, _auto_import_pil=False))
        with mock.patch.object(rbb, "_import_pil_image", return_value=None):
            self.assertIsNone(rbb.check_canvas_non_black(canvas_lit))

        # If real Pillow is installed on this machine, verify real PIL produces identical results
        real_pil = rbb._import_pil_image()
        if real_pil is not None:
            self.assertFalse(rbb.check_canvas_non_black(all_black, pil_image_mod=real_pil, _auto_import_pil=False))
            self.assertFalse(rbb.check_canvas_non_black(sidebar_only, pil_image_mod=real_pil, _auto_import_pil=False))
            self.assertTrue(rbb.check_canvas_non_black(canvas_lit, pil_image_mod=real_pil, _auto_import_pil=False))
            self.assertFalse(rbb.check_canvas_non_black(too_narrow, pil_image_mod=real_pil, _auto_import_pil=False))
            self.assertIsNone(rbb.check_canvas_non_black(b"\xff\xd8\xff\xd9", pil_image_mod=real_pil, _auto_import_pil=False))

    def test_capture_frames_dir_png_session_with_and_without_pillow(self):
        sidebar_only_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (255, 255, 255) if x < 410 else (0, 0, 0))
        canvas_lit_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (32, 64, 128) if x >= 410 else (0, 0, 0))
        frame_sequence = [sidebar_only_png, canvas_lit_png, canvas_lit_png]

        class FakeProc:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.returncode = -9

        def run_mock_png_session(fx: RepoFixture, run_name: str, capture_dir: Path, pil_return: Any):
            run_dir = fx.build / "runs" / run_name
            run_dir.mkdir(parents=True)
            shot_params: list[dict[str, Any]] = []
            shot_idx = {"n": 0}
            fake_clock = {"t": 1000.0}

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Runtime.evaluate":
                        return {"result": {"value": make_sample_probe(presents=2, real_draws=5, fallback_draws=0)}}
                    if method == "Page.captureScreenshot":
                        shot_params.append(dict(params or {}))
                        payload = frame_sequence[min(shot_idx["n"], len(frame_sequence) - 1)]
                        shot_idx["n"] += 1
                        return {"data": base64.b64encode(payload).decode("ascii")}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    fake_clock["t"] += 0.1
                    return []

            with (
                mock.patch.object(rbb, "_import_pil_image", return_value=pil_return),
                mock.patch.object(rbb.time, "monotonic", side_effect=lambda: fake_clock["t"]),
            ):
                res = rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=0.29,
                    headless=True,
                    smoke=True,
                    send_input=False,
                    capture_frames=True,
                    capture_frames_dir=capture_dir,
                    capture_interval_sec=0.1,
                    frame_interval_s=1.0,
                    encode_mp4=False,
                    input_script="menu-flow",
                    popen_fn=lambda cmd, **kw: FakeProc(),
                    cdp_factory=FakeCdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                )
            return res, shot_params

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))

            # 1. With PIL available (via pure-Python _FakePilModule): captures PNGs and counts non-black x>=410 frames
            cap_dir_pil = fx.repo / "work" / "frames-with-pil"
            res_pil, params_pil = run_mock_png_session(fx, "cap-pil", cap_dir_pil, _FakePilModule)
            self.assertTrue(res_pil.ok)
            self.assertEqual(len(res_pil.captured_frames), 3)
            self.assertTrue(all(p == {"format": "png"} for p in params_pil))
            for fname in res_pil.captured_frames:
                self.assertTrue(fname.endswith(".png"))
                self.assertTrue((cap_dir_pil / fname).is_file())
            # First frame is sidebar-only (black at x>=410), remaining 2 frames are canvas-lit
            self.assertEqual(res_pil.non_black_canvas_frames, 2)
            self.assertEqual(res_pil.capture_frames_dir, "work/frames-with-pil")

            # Verify CLI wiring writes capturedFrames / nonBlackCanvasFrames across report, telemetry, command, digest
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--name",
                "cli-cap-pil",
                "--input-script",
                "menu-flow",
                "--capture-frames",
                str(cap_dir_pil),
                "--capture-interval-sec",
                "0.5",
                "--dump-pipelines",
                session_runner=lambda **kw: res_pil,
            )
            self.assertEqual(code, 0, err)
            run_dir = fx.build / "runs" / "cli-cap-pil"
            report = json.loads((run_dir / "boot-report.json").read_text(encoding="utf-8"))
            telemetry = json.loads((run_dir / "telemetry.json").read_text(encoding="utf-8"))
            cmd_meta = json.loads((run_dir / "command.json").read_text(encoding="utf-8"))

            self.assertEqual(report["summary"]["capturedFrames"], len(res_pil.captured_frames))
            self.assertEqual(report["summary"]["nonBlackCanvasFrames"], len(res_pil.captured_frames) - 1)
            self.assertEqual(report["summary"]["inputScript"], "menu-flow")
            self.assertEqual(telemetry["summary"]["capturedFrames"], len(res_pil.captured_frames))
            self.assertEqual(telemetry["summary"]["nonBlackCanvasFrames"], len(res_pil.captured_frames) - 1)
            self.assertEqual(telemetry["capturedFrames"], len(res_pil.captured_frames))
            self.assertEqual(telemetry["nonBlackCanvasFrames"], len(res_pil.captured_frames) - 1)
            self.assertEqual(telemetry["capture_frames_dir"], "work/frames-with-pil")
            self.assertEqual(cmd_meta["capturedFrames"], len(res_pil.captured_frames))
            self.assertEqual(cmd_meta["nonBlackCanvasFrames"], len(res_pil.captured_frames) - 1)
            self.assertEqual(cmd_meta["capture_frames_dir"], "work/frames-with-pil")
            self.assertEqual(cmd_meta["capture_interval_sec"], 0.5)
            self.assertTrue(cmd_meta["dumpPipelines"])
            self.assertIn(f"captured_frames: {len(res_pil.captured_frames)} (non_black_canvas={len(res_pil.captured_frames) - 1})", out)
            # Zero host path leaks
            for fname in ("boot-report.json", "telemetry.json", "command.json", "digest.txt"):
                self.assertNotIn(str(Path(d)), (run_dir / fname).read_text(encoding="utf-8"))

            # 2. Without Pillow installed: still captures PNGs, gracefully sets nonBlackCanvasFrames to None
            cap_dir_nopil = fx.repo / "work" / "frames-no-pil"
            res_nopil, _ = run_mock_png_session(fx, "cap-nopil", cap_dir_nopil, None)
            self.assertTrue(res_nopil.ok)
            self.assertGreaterEqual(len(res_nopil.captured_frames), 1)
            self.assertIsNone(res_nopil.non_black_canvas_frames)
            self.assertTrue(any("Pillow" in w for w in res_nopil.warnings))

            rep_nopil = rbb.build_boot_report(res_nopil, smoke=True, preset="smoke")
            tel_nopil = rbb.build_telemetry_payload(res_nopil, smoke=True, preset="smoke", headless=True)
            self.assertEqual(rep_nopil["summary"]["capturedFrames"], len(res_nopil.captured_frames))
            self.assertIsNone(rep_nopil["summary"]["nonBlackCanvasFrames"])
            self.assertEqual(tel_nopil["summary"]["capturedFrames"], len(res_nopil.captured_frames))
            self.assertIsNone(tel_nopil["summary"]["nonBlackCanvasFrames"])
            self.assertIn("non_black_canvas=skipped", rbb.digest(rep_nopil, 0))

            # 3. encode_frames_to_mp4 also handles PNG frames
            ffmpeg_cmds: list[list[str]] = []

            def fake_ffmpeg(cmd, cwd, log, timeout):
                ffmpeg_cmds.append(list(cmd))
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            ok_mp4, warn_mp4 = rbb.encode_frames_to_mp4(
                cap_dir_pil,
                fx.repo / "work" / "out.mp4",
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok_mp4, warn_mp4)
            self.assertTrue(any("frame_%04d.png" in tok for tok in ffmpeg_cmds[0]))


class HudMp4AndNewPresetsTests(unittest.TestCase):
    def test_t49_flag_defaults_and_parsing(self):
        parser = rbb.build_arg_parser()
        defaults = parser.parse_args([])
        self.assertFalse(defaults.hide_hud)
        self.assertIsNone(defaults.mp4_fps)
        self.assertFalse(defaults.mp4_skip_black)

        parsed = parser.parse_args(
            [
                "--input-script",
                "legal-notices",
                "--hide-hud",
                "--mp4",
                "--mp4-fps",
                "4.5",
                "--mp4-skip-black",
            ]
        )
        self.assertEqual(parsed.input_script, "legal-notices")
        self.assertTrue(parsed.hide_hud)
        self.assertTrue(parsed.mp4)
        self.assertEqual(parsed.mp4_fps, 4.5)
        self.assertTrue(parsed.mp4_skip_black)

    def test_legal_notices_and_chrome_menu_flow_input_script_presets(self):
        for preset in ("legal-notices", "chrome-menu-flow"):
            self.assertEqual(rbb.sanitize_input_script_preset(preset), preset)
            self.assertEqual(rbb.sanitize_input_script_preset(f"  {preset.upper()}  "), preset)
            self.assertEqual(
                rbb.build_target_url("127.0.0.1", 8080, input_script=preset),
                f"http://127.0.0.1:8080/?inputScript={preset}",
            )

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for preset in ("legal-notices", "chrome-menu-flow"):
                code, out, err, _, _ = fx.run_cli(
                    "--smoke",
                    "--dry-run",
                    "--name",
                    f"dry-{preset}",
                    "--input-script",
                    preset,
                )
                self.assertEqual(code, 0, err)
                payload = json.loads(out)
                self.assertEqual(payload["input_script"], preset)
                self.assertEqual(payload["inputScript"], preset)
                self.assertEqual(
                    payload["url"],
                    f"http://127.0.0.1:<ephemeral>/?smoke=1&inputScript={preset}",
                )

    def test_hide_hud_url_plumbing_and_dry_run_plan(self):
        self.assertEqual(
            rbb.build_target_url("127.0.0.1", 8080, hide_hud=True),
            "http://127.0.0.1:8080/?hud=0",
        )
        self.assertEqual(
            rbb.build_target_url(
                "127.0.0.1",
                8080,
                smoke=True,
                input_script="chrome-menu-flow",
                stream_shaders=True,
                stream_threshold_mb=32.0,
                hide_hud=True,
            ),
            "http://127.0.0.1:8080/?smoke=1&inputScript=chrome-menu-flow&streamShaders=1&streamThresholdMb=32&hud=0",
        )

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            code, out, err, _, _ = fx.run_cli(
                "--smoke",
                "--dry-run",
                "--name",
                "hud-mp4-dry",
                "--input-script",
                "legal-notices",
                "--hide-hud",
                "--mp4",
                "--mp4-fps",
                "0.5",
                "--mp4-skip-black",
            )
            self.assertEqual(code, 0, err)
            payload = json.loads(out)
            self.assertTrue(payload["hide_hud"])
            self.assertTrue(payload["mp4"])
            self.assertEqual(payload["mp4_fps"], 0.5)
            self.assertTrue(payload["mp4_skip_black"])
            self.assertEqual(payload["input_script"], "legal-notices")
            self.assertEqual(
                payload["url"],
                "http://127.0.0.1:<ephemeral>/?smoke=1&inputScript=legal-notices&hud=0",
            )
            self.assertNotIn(str(Path(d)), out)

            # Default dry-run preserves backwards-compatible URL and defaults
            code_def, out_def, err_def, _, _ = fx.run_cli("--smoke", "--dry-run", "--name", "hud-def")
            self.assertEqual(code_def, 0, err_def)
            payload_def = json.loads(out_def)
            self.assertFalse(payload_def["hide_hud"])
            self.assertIsNone(payload_def["mp4_fps"])
            self.assertFalse(payload_def["mp4_skip_black"])
            self.assertNotIn("hud=0", payload_def["url"])

    def test_mp4_fps_validation(self):
        self.assertIsNone(rbb.validate_mp4_fps(None))
        self.assertEqual(rbb.validate_mp4_fps(0.5), 0.5)
        self.assertEqual(rbb.validate_mp4_fps(24), 24.0)
        for bad in (0, -1, -0.25, float("nan"), float("inf"), float("-inf"), True, "5"):
            with self.assertRaises(rbb.RunnerError, msg=f"expected RunnerError for {bad!r}"):
                rbb.validate_mp4_fps(bad)  # type: ignore[arg-type]

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            for bad_args in (
                ("--mp4-fps", "0"),
                ("--mp4-fps", "-2"),
                ("--mp4-fps", "-0.1"),
                ("--mp4-fps", "nan"),
                ("--mp4-fps", "inf"),
                ("--mp4-fps", "-inf"),
            ):
                code, _, _, _, _ = fx.run_cli("--smoke", "--dry-run", *bad_args)
                self.assertEqual(code, 2, f"expected exit 2 for {bad_args}")

    def test_encode_frames_to_mp4_custom_fps_and_skip_leading_black_frames(self):
        all_black_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 0, 0))
        sidebar_only_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (255, 255, 255) if x < 410 else (0, 0, 0))
        canvas_lit_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (10, 120, 240) if x >= 410 else (0, 0, 0))

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            # Frames 1 & 2 have black canvas (x>=410); frame 3 is lit; frame 4 is black again (must NOT be skipped)
            (frames_dir / "frame_0001.png").write_bytes(all_black_png)
            (frames_dir / "frame_0002.png").write_bytes(sidebar_only_png)
            (frames_dir / "frame_0003.png").write_bytes(canvas_lit_png)
            (frames_dir / "frame_0004.png").write_bytes(all_black_png)

            ffmpeg_cmds: list[list[str]] = []

            def fake_ffmpeg(cmd, cwd, log, timeout):
                ffmpeg_cmds.append(list(cmd))
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            # 1. skip_black=True with HUD visible (canvas_min_x=410) skips frames 1..2 -> -start_number 3 and custom fps 0.5
            ok, warn = rbb.encode_frames_to_mp4(
                frames_dir,
                root / "skip_hud.mp4",
                fps=0.5,
                skip_black=True,
                canvas_min_x=rbb.CANVAS_MIN_X,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok, warn)
            cmd1 = ffmpeg_cmds[-1]
            self.assertEqual(cmd1[cmd1.index("-framerate") + 1], "0.5")
            self.assertIn("-start_number", cmd1)
            self.assertEqual(cmd1[cmd1.index("-start_number") + 1], "3")
            self.assertLess(cmd1.index("-start_number"), cmd1.index("-i"))

            # 2. skip_black=True with hide_hud (canvas_min_x=0): frame 2 has lit pixels at x<410 -> -start_number 2
            ok2, warn2 = rbb.encode_frames_to_mp4(
                frames_dir,
                root / "skip_nohud.mp4",
                fps=4.0,
                skip_black=True,
                canvas_min_x=0,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok2, warn2)
            cmd2 = ffmpeg_cmds[-1]
            self.assertEqual(cmd2[cmd2.index("-framerate") + 1], "4.0")
            self.assertEqual(cmd2[cmd2.index("-start_number") + 1], "2")

            # 3. skip_black=False does NOT pass -start_number
            ok3, warn3 = rbb.encode_frames_to_mp4(
                frames_dir,
                root / "noskip.mp4",
                fps=2.5,
                skip_black=False,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok3, warn3)
            cmd3 = ffmpeg_cmds[-1]
            self.assertEqual(cmd3[cmd3.index("-framerate") + 1], "2.5")
            self.assertNotIn("-start_number", cmd3)

            # 4. Graceful fallback when Pillow is unavailable
            ok4, warn4 = rbb.encode_frames_to_mp4(
                frames_dir,
                root / "nopil.mp4",
                fps=2.0,
                skip_black=True,
                pil_image_mod=None,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok4, warn4)
            self.assertNotIn("-start_number", ffmpeg_cmds[-1])

            # 5. All-black frames gracefully fall back to encoding from frame 1 (no -start_number)
            all_black_dir = root / "all_black"
            all_black_dir.mkdir()
            (all_black_dir / "frame_0001.png").write_bytes(all_black_png)
            (all_black_dir / "frame_0002.png").write_bytes(all_black_png)
            ok5, warn5 = rbb.encode_frames_to_mp4(
                all_black_dir,
                root / "all_black.mp4",
                fps=2.0,
                skip_black=True,
                canvas_min_x=0,
                pil_image_mod=_FakePilModule,
                _auto_import_pil=False,
                runner=fake_ffmpeg,
                which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
            )
            self.assertTrue(ok5, warn5)
            self.assertNotIn("-start_number", ffmpeg_cmds[-1])

    def test_run_browser_session_and_cli_plumbing_for_hide_hud_mp4_fps_and_skip_black(self):
        all_black_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (0, 0, 0))
        left_lit_png = _make_synthetic_rgb_png(420, 4, lambda x, y: (200, 100, 50) if x < 410 else (0, 0, 0))
        frame_seq = [all_black_png, left_lit_png, left_lit_png]

        class FakeProc:
            returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                self.returncode = 0

            def wait(self, timeout=None):
                return 0

            def kill(self):
                self.returncode = -9

        with tempfile.TemporaryDirectory() as d:
            fx = RepoFixture(Path(d))
            run_dir = fx.build / "runs" / "t49-session"
            run_dir.mkdir(parents=True)
            cap_dir = fx.repo / "work" / "t49-frames"

            navigated_urls: list[str] = []
            ffmpeg_cmds: list[list[str]] = []
            shot_idx = {"n": 0}
            fake_clock = {"t": 2000.0}

            class FakeCdp:
                def __init__(self, ws_url: str) -> None:
                    self.ws_url = ws_url

                def connect(self):
                    return None

                def close(self):
                    return None

                def send(self, method: str, params=None, timeout=None):
                    if method == "Page.navigate" and isinstance(params, dict):
                        navigated_urls.append(str(params.get("url") or ""))
                    if method == "Runtime.evaluate":
                        return {"result": {"value": make_sample_probe(presents=2, real_draws=3, fallback_draws=0)}}
                    if method == "Page.captureScreenshot":
                        payload = frame_seq[min(shot_idx["n"], len(frame_seq) - 1)]
                        shot_idx["n"] += 1
                        return {"data": base64.b64encode(payload).decode("ascii")}
                    return {}

                def drain_events(self, timeout: float = 0.0):
                    fake_clock["t"] += 0.1
                    return []

            def fake_ffmpeg(cmd, cwd, log, timeout):
                ffmpeg_cmds.append(list(cmd))
                Path(cmd[-1]).write_bytes(b"ftypmp42")
                return 0

            with (
                mock.patch.object(rbb, "_import_pil_image", return_value=_FakePilModule),
                mock.patch.object(rbb.time, "monotonic", side_effect=lambda: fake_clock["t"]),
            ):
                res = rbb.run_browser_session(
                    repo=fx.repo,
                    run_dir=run_dir,
                    chrome_path=fx.chrome,
                    port=0,
                    boot_dir=None,
                    game_dir=None,
                    shaders_dir=None,
                    duration_s=0.29,
                    headless=True,
                    smoke=True,
                    send_input=False,
                    capture_frames=True,
                    capture_frames_dir=cap_dir,
                    capture_interval_sec=0.1,
                    frame_interval_s=1.0,
                    encode_mp4=True,
                    mp4_fps=3.5,
                    mp4_skip_black=True,
                    hide_hud=True,
                    input_script="chrome-menu-flow",
                    runner=fake_ffmpeg,
                    popen_fn=lambda cmd, **kw: FakeProc(),
                    cdp_factory=FakeCdp,
                    wait_server_fn=lambda *a, **k: {"status_code": 200, "coop": "same-origin", "coep": "require-corp"},
                    wait_cdp_fn=lambda *a, **k: "ws://127.0.0.1:9222/devtools/page/1",
                    which_fn=lambda n: "ffmpeg" if n == "ffmpeg" else None,
                )

            self.assertTrue(res.ok)
            self.assertEqual(len(navigated_urls), 1)
            self.assertIn("inputScript=chrome-menu-flow", navigated_urls[0])
            self.assertIn("hud=0", navigated_urls[0])
            # hide_hud=True checks full frame (canvas_min_x=0), so frame 1 is black and frames 2..3 are lit
            self.assertEqual(len(res.captured_frames), 3)
            self.assertEqual(res.non_black_canvas_frames, 2)
            self.assertEqual(res.mp4_path, "capture.mp4")
            self.assertEqual(len(ffmpeg_cmds), 1)
            cmd = ffmpeg_cmds[0]
            self.assertEqual(cmd[cmd.index("-framerate") + 1], "3.5")
            self.assertEqual(cmd[cmd.index("-start_number") + 1], "2")

            # Verify CLI writes new metadata cleanly into telemetry.json and command.json
            code, _, err, _, sessions = fx.run_cli(
                "--smoke",
                "--name",
                "t49-cli-run",
                "--input-script",
                "legal-notices",
                "--hide-hud",
                "--mp4",
                "--mp4-fps",
                "6",
                "--mp4-skip-black",
            )
            self.assertEqual(code, 0, err)
            self.assertEqual(sessions[0]["input_script"], "legal-notices")
            self.assertTrue(sessions[0]["hide_hud"])
            self.assertEqual(sessions[0]["mp4_fps"], 6.0)
            self.assertTrue(sessions[0]["mp4_skip_black"])

            cli_run_dir = fx.build / "runs" / "t49-cli-run"
            telemetry = json.loads((cli_run_dir / "telemetry.json").read_text(encoding="utf-8"))
            cmd_meta = json.loads((cli_run_dir / "command.json").read_text(encoding="utf-8"))
            self.assertTrue(telemetry["hide_hud"])
            self.assertEqual(telemetry["mp4_fps"], 6.0)
            self.assertTrue(telemetry["mp4_skip_black"])
            self.assertEqual(telemetry["inputScript"], "legal-notices")
            self.assertTrue(cmd_meta["hide_hud"])
            self.assertEqual(cmd_meta["mp4_fps"], 6.0)
            self.assertTrue(cmd_meta["mp4_skip_black"])
            self.assertEqual(cmd_meta["input_script"], "legal-notices")


if __name__ == "__main__":
    unittest.main()
