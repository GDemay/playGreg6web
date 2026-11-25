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
from contextlib import redirect_stderr, redirect_stdout
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
            win_edge = Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe")
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


if __name__ == "__main__":
    unittest.main()
