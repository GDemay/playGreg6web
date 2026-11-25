#!/usr/bin/env python3
"""One-command browser boot & test runner: rebuild what is stale, serve locally, drive Chrome via CDP, keep the run.

    python tools/run_browser_boot.py [--preset smoke|start-screen|long] [--name NAME]
        [--smoke] [--boot-dir DIR] [--game-dir DIR] [--shader-pack FILE] [--out-dir DIR]
        [--runs-dir DIR] [--port PORT] [--duration-s N] [--headless | --headed]
        [--chrome PATH] [--trust-build] [--rebuild] [--no-build] [--dry-run]
        [--capture-frames] [--mp4] [--baseline NAME|DIR]

What it does, in order:

1. Staleness & build plan (real-game mode; skipped for `--smoke` and `--trust-build`):
   - `recomp`: `cargo build --release -p playgta6-recomp` when missing or older than
     workspace sources (only when `program` needs to be built or `--exe` is provided).
   - `program`: `playgta6-recomp build-program <exe> --out <boot-dir> --target wasm64
     --memory-model window-runs --indirect-dispatch shard --jobs N` (+ `--dll-dir <game-dir>`)
     when `<boot-dir>/program-manifest.json` is missing or stale.
   - `runtime_configure` / `runtime_lib`: Release PIC `libplaygta6_core.a` built via
     `emcmake cmake` (`-sMEMORY64=1 -pthread -fPIC`) when missing or older than `runtime/`.
   - `guest_wasm`: `python tools/build_guest_wasm.py <boot-dir> --out-dir <out-dir> ...`
     with the proven asymmetric split flags (`--split-modules 2 --split-strategy asymmetric
     --link-guest-boot --browser-boot-exports --proxy-to-pthread --strip --no-link-opt`)
     when `runtime.wasm`/`runtime.js` is missing or stale.
   - Optional `--shader-pack FILE`: atomically stages `<pack>` and any sibling
     `<pack>.reflect.json` / `reflection-manifest.json` sidecars into a gitignored
     staging directory served at `/shaders/`.
2. Local server + Chrome CDP session:
   - Picks a free localhost port (or `--port`), starts `server/serve_local.py` with
     `COOP`/`COEP` headers, and waits until `/api/status` responds.
   - Discovers Chrome/Chromium/Edge automatically (`--chrome`, `CHROME_PATH`, PATH,
     standard OS install paths) and launches it with a clean temporary `--user-data-dir`,
     ephemeral CDP debugging port, `--enable-unsafe-webgpu`, and `--headless=new` (unless
     `--headed`).
   - Connects over pure-Python stdlib CDP WebSocket, collects console logs, page errors,
     HUD/WebGPU/boot telemetry, optional screenshots (`--capture-frames`), optional H.264
     MP4 via `ffmpeg` (`--mp4`), and scripted input on `start-screen`.
   - Always tears down Chrome, temp user-data-dir, and `serve_local.py` in `finally:`.
3. Uniform artifacts under `<runs-dir>/<name>/` (default `work/browser-boot/runs/<name>/`,
   gitignored, zero host paths in shared outputs):
   - `boot-report.json` (`playgta6-boot-report` v2, compatible with `compare_boot_reports.py`)
   - `console.log`
   - `telemetry.json`
   - `command.json`
   - `digest.txt`
   - optional `frames/frame_0001.jpg`... and `capture.mp4`
4. Optional `--baseline NAME|DIR` invokes `tools/compare_boot_reports.py`.

Exit code: 0 ok (and matches baseline if given), 1 baseline differences, 2 usage or build error,
3 the browser session produced no report.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import os
import re
import select
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_boot_reports  # noqa: E402

EXE_SUFFIX = ".exe" if sys.platform == "win32" else ""
SOURCE_SUFFIXES = {".cpp", ".h", ".hpp", ".c", ".cmake", ".txt", ".rs", ".toml"}
REPORT_SCHEMA = "playgta6-boot-report"
REPORT_SCHEMA_VERSION = 2
WEBSOCKET_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

DEFAULT_BUILD_DIR_REL = Path("work") / "browser-boot"
DEFAULT_RUNS_DIR_REL = DEFAULT_BUILD_DIR_REL / "runs"
DEFAULT_BOOT_DIR_REL = Path("work") / "recomp_build"
DEFAULT_OUT_DIR_REL = Path("web") / "dist"
DEFAULT_RUNTIME_PIC_REL = DEFAULT_BUILD_DIR_REL / "runtime-pic" / "libplaygta6_core.a"
DEFAULT_SHADER_STAGE_REL = DEFAULT_BUILD_DIR_REL / "staged-shaders"

PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {"duration_s": 60.0, "input": False},
    "start-screen": {"duration_s": 120.0, "input": True},
    "long": {"duration_s": 600.0, "input": False},
}
SMOKE_MODE_DEFAULT_DURATION_S = 10.0
INPUT_START_S = 5.0
INPUT_PERIOD_S = 1.0
INPUT_HOLD_S = 0.1
DEFAULT_FRAME_INTERVAL_S = 1.0
DEFAULT_POLL_INTERVAL_S = 0.5
SERVER_READY_TIMEOUT_S = 15.0
CDP_READY_TIMEOUT_S = 15.0

COPY_CHUNK_BYTES = 256 * 1024

_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)

CommandRunner = Callable[[list[str], Path, Path | None, float | None], int]
WhichFn = Callable[[str], str | None]


class RunnerError(Exception):
    """Usage, environment, or build failure (exit code 2)."""


class CdpError(Exception):
    """Chrome DevTools Protocol transport or command failure."""


# ---------------------------------------------------------------------------
# Sanitization (zero host paths / usernames in artifacts and dry-run output)
# ---------------------------------------------------------------------------


def _current_usernames() -> list[str]:
    names: list[str] = []
    for candidate in (
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
    ):
        cleaned = candidate.strip()
        if len(cleaned) >= 3 and cleaned not in names:
            names.append(cleaned)
    try:
        home_name = Path.home().name.strip()
        if len(home_name) >= 3 and home_name not in names:
            names.append(home_name)
    except (OSError, RuntimeError):
        pass
    return names


def sanitize_text(raw: str | None) -> str:
    """Strip Windows/POSIX/UNC host paths and active usernames."""
    if raw is None:
        return ""
    text = str(raw)
    text = _UNC_BS_PATH_RE.sub("<path>", text)
    text = text.replace("\\", "/")
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = _UNC_FS_PATH_RE.sub("<path>", text)
    text = _POSIX_PATH_RE.sub("<path>", text)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        text = pattern.sub("<user>", text)
    return text


def basename_label(source: str | Path | None) -> str:
    """Basename-only label; splits on both `/` and `\\` regardless of host OS."""
    if source is None:
        return "<unknown>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return sanitize_text(base)


def repo_relative_label(path: Path | None, repo_root: Path) -> str | None:
    """Return a repo-relative POSIX label or a sanitized basename, never a host path."""
    if path is None:
        return None
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        posix = rel.as_posix()
        return sanitize_text("." if not posix else posix)
    except (OSError, ValueError):
        return basename_label(path)


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings/paths and replace raw bytes with size placeholders."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Path):
        return basename_label(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]
    return value


def sanitize_cmd(cmd: Sequence[str], repo_root: Path) -> list[str]:
    """Sanitize a command argv list so host paths become repo-relative labels or basenames."""
    out: list[str] = []
    resolved_repo = repo_root.resolve()
    for idx, token in enumerate(cmd):
        text = str(token)
        if idx == 0 and (os.sep in text or "/" in text or "\\" in text):
            out.append(basename_label(text))
            continue
        if text.startswith("-"):
            if "=" in text:
                flag, val = text.split("=", 1)
                if _looks_like_host_path(val):
                    out.append(f"{flag}={repo_relative_label(Path(val), resolved_repo) or basename_label(val)}")
                else:
                    out.append(sanitize_text(text))
            else:
                out.append(sanitize_text(text))
            continue
        if _looks_like_host_path(text):
            out.append(repo_relative_label(Path(text), resolved_repo) or basename_label(text))
        else:
            out.append(sanitize_text(text))
    return out


def _looks_like_host_path(text: str) -> bool:
    if not text:
        return False
    if text.startswith(("http://", "https://", "ws://", "wss://")):
        return False
    if re.match(r"^[A-Za-z]:[\\/]", text) or text.startswith("\\\\"):
        return True
    if _POSIX_PATH_RE.match(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Subprocess & filesystem helpers
# ---------------------------------------------------------------------------


def run_command(cmd: list[str], cwd: Path, log: Path | None = None, timeout: float | None = None) -> int:
    """Run `cmd`; with `log`, stdout and stderr go there, otherwise to this console."""
    try:
        if log is None:
            return subprocess.run(cmd, cwd=cwd, timeout=timeout).returncode
        with log.open("wb") as fh:
            return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124
    except OSError as exc:
        raise RunnerError(f"cannot run {cmd[0]}: {exc}") from exc


def load_dotenv_value(repo_root: Path, key: str) -> str | None:
    """Read one key from `<repo_root>/.env` if present."""
    env_file = repo_root / ".env"
    if not env_file.is_file():
        return None
    try:
        content = env_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            val = v.strip().strip('"').strip("'")
            return val or None
    return None


def resolve_against_repo(raw_path: Path | str, repo_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        return (repo_root / path).resolve()
    return path.resolve()


def newest_mtime(root: Path, files: Sequence[Path] = ()) -> float:
    """Newest mtime among build-relevant sources below `root` (and `files`); 0 if none."""
    newest = 0.0
    for path in files:
        if path.is_file():
            try:
                newest = max(newest, path.stat().st_mtime)
            except OSError:
                pass
    if root.is_dir():
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(("build", "target", ".")) and d != "node_modules"
            ]
            for name in filenames:
                if Path(name).suffix in SOURCE_SUFFIXES:
                    try:
                        newest = max(newest, os.stat(os.path.join(dirpath, name)).st_mtime)
                    except OSError:
                        pass
    return newest


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def workspace_dirs(repo: Path) -> list[Path]:
    """Crate directories of the cargo workspace (root Cargo.toml `members`), else `recomp`."""
    manifest = repo / "Cargo.toml"
    members: list[str] = []
    if manifest.is_file():
        text = manifest.read_text(encoding="utf-8", errors="replace")
        block = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S)
        if block:
            members = re.findall(r'"([^"]+)"', block.group(1))
    return [repo / m for m in members if "*" not in m] or [repo / "recomp"]


def atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy `src` to `dst` via a same-directory temp file and `os.replace`."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_stat = src.stat()
    fd, tmp_str = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=dst.parent)
    tmp_path = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as dst_fh, src.open("rb") as src_fh:
            while True:
                chunk = src_fh.read(COPY_CHUNK_BYTES)
                if not chunk:
                    break
                dst_fh.write(chunk)
            dst_fh.flush()
        try:
            os.utime(tmp_path, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
        except OSError:
            pass
        os.replace(tmp_path, dst)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def find_free_port(host: str = "127.0.0.1") -> int:
    """Allocate an ephemeral free TCP port on `host`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def terminate_process(proc: Any, wait_s: float = 5.0) -> int | None:
    """Gracefully terminate a subprocess, escalating to kill() if needed."""
    if proc is None:
        return None
    try:
        if proc.poll() is not None:
            return proc.returncode
        proc.terminate()
        try:
            return proc.wait(timeout=wait_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                return proc.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                return proc.poll()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Staleness & build planning
# ---------------------------------------------------------------------------


class Plan:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def add(self, stage: str, reason: str, cmd: list[str]) -> None:
        self.steps.append({"stage": stage, "reason": reason, "cmd": cmd})

    def has(self, stage: str) -> bool:
        return any(s["stage"] == stage for s in self.steps)

    def sanitized(self, repo_root: Path) -> list[dict[str, Any]]:
        return [
            {
                "stage": s["stage"],
                "reason": sanitize_text(s["reason"]),
                "cmd": sanitize_cmd(s["cmd"], repo_root),
            }
            for s in self.steps
        ]


def make_build_plan(
    *,
    repo: Path,
    build_dir: Path,
    boot_dir: Path,
    out_dir: Path,
    exe: Path | None,
    game_dir: Path | None,
    runtime_lib_override: Path | None,
    jobs: int,
    force: bool,
    smoke: bool,
) -> tuple[Plan, Path]:
    """Compute the staleness build plan for browser Wasm64 artifacts.

    Returns `(plan, runtime_lib_path)`.
    """
    runtime_pic_dir = build_dir / "runtime-pic"
    runtime_lib = runtime_lib_override if runtime_lib_override is not None else (runtime_pic_dir / "libplaygta6_core.a")
    plan = Plan()
    if smoke:
        return plan, runtime_lib

    def stale(target: Path, reasons: list[tuple[str, float]]) -> str | None:
        if force:
            return "forced (--rebuild)"
        t = mtime(target)
        if t is None:
            return f"{target.name} is missing"
        for what, newest in reasons:
            if newest > t:
                return f"{what} is newer than {target.name}"
        return None

    recomp = repo / "target" / "release" / ("playgta6-recomp" + EXE_SUFFIX)
    manifest = boot_dir / "program-manifest.json"
    manifest_time = mtime(manifest)

    # Check recomp + program only when exe is given, manifest is missing, or --rebuild is forced.
    check_program = force or manifest_time is None or exe is not None
    if check_program:
        sources = [(d / "Cargo.toml") for d in workspace_dirs(repo)] + [repo / "Cargo.toml", repo / "Cargo.lock"]
        newest_rs = max(
            [newest_mtime(d / "src") for d in workspace_dirs(repo)]
            + [0.0]
            + [mtime(p) or 0.0 for p in sources]
        )
        recomp_reason = stale(recomp, [("the recompiler sources", newest_rs)])
        if recomp_reason:
            plan.add("recomp", recomp_reason, ["cargo", "build", "--release", "-p", "playgta6-recomp"])

        exe_time = mtime(exe) if exe else 0.0
        prog_reason = stale(
            manifest,
            [("the recompiler", mtime(recomp) or 0.0), ("the exe", exe_time or 0.0)],
        )
        if prog_reason is None and plan.has("recomp"):
            prog_reason = "the recompiler is rebuilt"
        if prog_reason:
            if exe is None:
                raise RunnerError(
                    "program-manifest.json is missing or forced to rebuild, so --exe or PLAYGTA6_EXE "
                    "is required (or pass --smoke / --trust-build)"
                )
            prog_cmd = [
                str(recomp),
                "build-program",
                str(exe),
                "--out",
                str(boot_dir),
                "--target",
                "wasm64",
                "--memory-model",
                "window-runs",
                "--indirect-dispatch",
                "shard",
                "--jobs",
                str(jobs),
            ]
            if game_dir is not None:
                prog_cmd += ["--dll-dir", str(game_dir)]
            plan.add("program", prog_reason, prog_cmd)

    # Stage 2: Release PIC libplaygta6_core.a
    runtime_newest = newest_mtime(repo / "runtime")
    if runtime_lib_override is None:
        cmake_cache = runtime_pic_dir / "CMakeCache.txt"
        cfg_reason = "forced (--rebuild)" if force else (
            "CMakeCache.txt is missing" if mtime(cmake_cache) is None else None
        )
        if cfg_reason:
            plan.add(
                "runtime_configure",
                cfg_reason,
                [
                    "emcmake",
                    "cmake",
                    "-S",
                    str(repo / "runtime"),
                    "-B",
                    str(runtime_pic_dir),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_CXX_FLAGS=-sMEMORY64=1 -pthread -fPIC",
                    "-DCMAKE_C_FLAGS=-sMEMORY64=1 -pthread -fPIC",
                ],
            )
        lib_reason = stale(runtime_lib, [("the runtime sources", runtime_newest)])
        if lib_reason is None and plan.has("runtime_configure"):
            lib_reason = "CMake was reconfigured"
        if lib_reason:
            plan.add(
                "runtime_lib",
                lib_reason,
                [
                    "cmake",
                    "--build",
                    str(runtime_pic_dir),
                    "--target",
                    "playgta6_core",
                    "--parallel",
                    str(jobs),
                ],
            )

    # Stage 3: Guest Wasm64 + asymmetric split side modules via build_guest_wasm.py
    runtime_wasm = out_dir / "runtime.wasm"
    runtime_js = out_dir / "runtime.js"
    build_guest_script = repo / "tools" / "build_guest_wasm.py"
    wasm_inputs = [
        ("program-manifest.json", mtime(manifest) or 0.0),
        ("libplaygta6_core.a", mtime(runtime_lib) or 0.0),
        ("runtime sources or build_guest_wasm.py", newest_mtime(repo / "runtime", [build_guest_script])),
    ]
    wasm_reason = stale(runtime_wasm, wasm_inputs) or stale(runtime_js, wasm_inputs)
    if wasm_reason is None and (plan.has("program") or plan.has("runtime_lib")):
        wasm_reason = "an upstream stage is rebuilt"
    if wasm_reason:
        wasm_cmd = [
            sys.executable,
            str(build_guest_script),
            str(boot_dir),
            "--out-dir",
            str(out_dir),
            "--cache-dir",
            str(build_dir / "wasm_obj_cache"),
            "--runtime-lib",
            str(runtime_lib),
            "--jobs",
            str(jobs),
            "--opt",
            "O2",
            "--link-guest-boot",
            "--browser-boot-exports",
            "--split-modules",
            "2",
            "--split-strategy",
            "asymmetric",
            "--proxy-to-pthread",
            "--strip",
            "--no-link-opt",
        ]
        plan.add("guest_wasm", wasm_reason, wasm_cmd)

    return plan, runtime_lib


# ---------------------------------------------------------------------------
# Shader pack + reflection sidecar staging
# ---------------------------------------------------------------------------


def plan_shader_staging(shader_pack: Path | None, stage_dir: Path) -> list[dict[str, str]]:
    """Plan atomic staging of `--shader-pack` plus `<pack>.reflect.json` and `reflection-manifest.json`."""
    if shader_pack is None:
        return []
    entries: list[dict[str, str]] = [
        {"kind": "shader_pack", "src": str(shader_pack), "dst": str(stage_dir / shader_pack.name), "name": shader_pack.name}
    ]
    # Primary sidecar convention (tools/stage_reflection_sidecar.py): `<pack>.reflect.json`
    sidecar_primary = Path(str(shader_pack) + ".reflect.json")
    if sidecar_primary.is_file():
        entries.append(
            {
                "kind": "stage_reflection",
                "src": str(sidecar_primary),
                "dst": str(stage_dir / sidecar_primary.name),
                "name": sidecar_primary.name,
            }
        )
    else:
        alt = shader_pack.with_suffix(".reflect.json")
        if alt != sidecar_primary and alt.is_file():
            entries.append(
                {
                    "kind": "stage_reflection",
                    "src": str(alt),
                    "dst": str(stage_dir / alt.name),
                    "name": alt.name,
                }
            )
    manifest = shader_pack.parent / "reflection-manifest.json"
    if manifest.is_file():
        entries.append(
            {
                "kind": "reflection_manifest",
                "src": str(manifest),
                "dst": str(stage_dir / manifest.name),
                "name": manifest.name,
            }
        )
    return entries


def stage_shader_pack(shader_pack: Path, stage_dir: Path) -> list[dict[str, Any]]:
    """Stage shader pack and sibling reflection sidecars into `stage_dir`."""
    if not shader_pack.is_file():
        raise RunnerError(f"shader pack not found: {basename_label(shader_pack)}")
    stage_dir.mkdir(parents=True, exist_ok=True)
    planned = plan_shader_staging(shader_pack, stage_dir)
    staged: list[dict[str, Any]] = []
    for item in planned:
        src = Path(item["src"])
        dst = Path(item["dst"])
        needs_copy = True
        if dst.is_file():
            try:
                if src.resolve() == dst.resolve():
                    needs_copy = False
                else:
                    s_st, d_st = src.stat(), dst.stat()
                    if s_st.st_size == d_st.st_size and int(s_st.st_mtime) == int(d_st.st_mtime):
                        needs_copy = False
            except OSError:
                needs_copy = True
        if needs_copy:
            atomic_copy_file(src, dst)
        staged.append({"kind": item["kind"], "name": item["name"], "copied": needs_copy})
    return staged


# ---------------------------------------------------------------------------
# Chrome / Chromium / Edge discovery
# ---------------------------------------------------------------------------


def discover_chrome(
    explicit_path: Path | None = None,
    *,
    env: Mapping[str, str] | None = None,
    which_fn: WhichFn = shutil.which,
    platform: str | None = None,
    exists_fn: Callable[[Path], bool] | None = None,
) -> Path | None:
    """Locate Chrome, Chromium, or Edge across Windows, Linux, and macOS."""
    env_map = os.environ if env is None else env
    plat = sys.platform if platform is None else platform
    is_file = (lambda p: p.is_file()) if exists_fn is None else exists_fn

    if explicit_path is not None:
        p = Path(explicit_path).expanduser()
        return p if is_file(p) else None

    for env_key in ("CHROME_PATH", "PLAYGTA6_CHROME"):
        raw = (env_map.get(env_key) or "").strip()
        if raw:
            p = Path(raw).expanduser()
            if is_file(p):
                return p

    if plat == "win32":
        which_names = ("chrome", "chrome.exe", "msedge", "msedge.exe", "chromium", "chromium.exe")
    elif plat == "darwin":
        which_names = ("google-chrome", "chromium", "microsoft-edge")
    else:
        which_names = (
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
            "microsoft-edge",
            "microsoft-edge-stable",
        )

    for name in which_names:
        found = which_fn(name)
        if found:
            p = Path(found)
            if is_file(p):
                return p

    candidates: list[Path] = []
    if plat == "win32":
        roots = [
            env_map.get("PROGRAMFILES", r"C:\Program Files"),
            env_map.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            env_map.get("LOCALAPPDATA", ""),
        ]
        rel_paths = (
            Path("Google") / "Chrome" / "Application" / "chrome.exe",
            Path("Microsoft") / "Edge" / "Application" / "msedge.exe",
            Path("Chromium") / "Application" / "chrome.exe",
        )
        for root in roots:
            if not root:
                continue
            for rel in rel_paths:
                candidates.append(Path(root) / rel)
    elif plat == "darwin":
        candidates.extend(
            [
                Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                Path("/Applications/Chromium.app/Contents/MacOS/Chromium"),
                Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
            ]
        )
    else:
        candidates.extend(
            [
                Path("/usr/bin/google-chrome"),
                Path("/usr/bin/google-chrome-stable"),
                Path("/usr/bin/chromium"),
                Path("/usr/bin/chromium-browser"),
                Path("/usr/bin/microsoft-edge"),
                Path("/snap/bin/chromium"),
            ]
        )

    for candidate in candidates:
        if is_file(candidate):
            return candidate
    return None


def build_chrome_args(
    chrome_path: Path,
    *,
    user_data_dir: Path,
    cdp_port: int,
    headless: bool,
    initial_url: str = "about:blank",
) -> list[str]:
    """Construct deterministic Chrome launch flags for WebGPU + CDP testing."""
    cmd = [
        str(chrome_path),
        f"--user-data-dir={user_data_dir}",
        f"--remote-debugging-port={cdp_port}",
        "--enable-unsafe-webgpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--autoplay-policy=no-user-gesture-required",
        "--window-size=1280,720",
    ]
    if headless:
        cmd.append("--headless=new")
    cmd.append(initial_url)
    return cmd


# ---------------------------------------------------------------------------
# Local server readiness helper
# ---------------------------------------------------------------------------


def build_server_cmd(
    *,
    repo: Path,
    host: str,
    port: int,
    web_dir: Path,
    data_dir: Path,
    boot_dir: Path | None,
    game_dir: Path | None,
    shaders_dir: Path | None,
) -> list[str]:
    """Build the `server/serve_local.py` launch command."""
    cmd = [
        sys.executable,
        str(repo / "server" / "serve_local.py"),
        "--host",
        host,
        "--port",
        str(port),
        "--web",
        str(web_dir),
        "--data",
        str(data_dir),
    ]
    if boot_dir is not None:
        cmd += ["--boot", str(boot_dir)]
    if game_dir is not None:
        cmd += ["--game", str(game_dir)]
    if shaders_dir is not None:
        cmd += ["--shaders", str(shaders_dir)]
    return cmd


def wait_for_server_ready(
    host: str,
    port: int,
    *,
    timeout_s: float = SERVER_READY_TIMEOUT_S,
    proc: Any = None,
) -> dict[str, Any]:
    """Poll `/api/status` (or `/`) until HTTP 200 with COOP/COEP headers is observed."""
    deadline = time.monotonic() + timeout_s
    last_error = "no response"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RunnerError(f"serve_local.py exited prematurely with code {proc.returncode}")
        for path in ("/api/status", "/"):
            conn: http.client.HTTPConnection | None = None
            try:
                conn = http.client.HTTPConnection(host, port, timeout=1.5)
                conn.request("GET", path)
                resp = conn.getresponse()
                body = resp.read()
                coop = (resp.getheader("Cross-Origin-Opener-Policy") or "").strip().lower()
                coep = (resp.getheader("Cross-Origin-Embedder-Policy") or "").strip().lower()
                if resp.status == 200:
                    if coop != "same-origin" or coep != "require-corp":
                        last_error = f"missing COOP/COEP headers (coop={coop!r}, coep={coep!r})"
                        continue
                    api_status: dict[str, Any] = {}
                    if path == "/api/status":
                        try:
                            parsed = json.loads(body.decode("utf-8", errors="replace"))
                            if isinstance(parsed, dict):
                                api_status = parsed
                        except json.JSONDecodeError:
                            pass
                    return {
                        "status_code": resp.status,
                        "path": path,
                        "coop": coop,
                        "coep": coep,
                        "api_status": api_status,
                    }
                last_error = f"GET {path} returned HTTP {resp.status}"
            except OSError as exc:
                last_error = str(exc)
            finally:
                if conn is not None:
                    try:
                        conn.close()
                    except OSError:
                        pass
        time.sleep(0.1)
    raise RunnerError(f"local server did not become ready within {timeout_s:.1f}s ({sanitize_text(last_error)})")


# ---------------------------------------------------------------------------
# Pure-Python stdlib RFC 6455 WebSocket + CDP client
# ---------------------------------------------------------------------------


def _ws_encode_frame(opcode: int, payload: bytes, mask: bool = True) -> bytes:
    header = bytearray()
    header.append(0x80 | (opcode & 0x0F))
    length = len(payload)
    mask_bit = 0x80 if mask else 0x00
    if length < 126:
        header.append(mask_bit | length)
    elif length <= 0xFFFF:
        header.append(mask_bit | 126)
        header.extend(struct.pack("!H", length))
    else:
        header.append(mask_bit | 127)
        header.extend(struct.pack("!Q", length))
    if mask:
        mask_key = os.urandom(4)
        header.extend(mask_key)
        masked = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))
        return bytes(header) + masked
    return bytes(header) + payload


class CdpClient:
    """Minimal pure-Python stdlib WebSocket client for Chrome DevTools Protocol."""

    def __init__(self, ws_url: str, timeout: float = 15.0) -> None:
        self.ws_url = ws_url
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = bytearray()
        self._next_id = 1
        self._events: list[dict[str, Any]] = []
        self._closed = False

    def connect(self) -> None:
        parsed = urlsplit(self.ws_url)
        if parsed.scheme != "ws":
            raise CdpError(f"unsupported CDP WebSocket scheme: {parsed.scheme!r}")
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
        path = parsed.path or "/"
        if parsed.query:
            path = f"{path}?{parsed.query}"

        sock = socket.create_connection((host, port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        req = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        ).encode("ascii")
        try:
            sock.sendall(req)
            raw = bytearray()
            while b"\r\n\r\n" not in raw:
                chunk = sock.recv(4096)
                if not chunk:
                    raise CdpError("CDP WebSocket closed during handshake")
                raw.extend(chunk)
                if len(raw) > 65536:
                    raise CdpError("CDP WebSocket handshake header too large")
            header_end = raw.index(b"\r\n\r\n") + 4
            header_text = raw[:header_end].decode("latin1")
            leftover = raw[header_end:]
            lines = header_text.split("\r\n")
            if not lines or " 101 " not in lines[0]:
                raise CdpError(f"CDP WebSocket handshake failed: {lines[0] if lines else 'empty'}")
            headers: dict[str, str] = {}
            for line in lines[1:]:
                if ":" in line:
                    k, v = line.split(":", 1)
                    headers[k.strip().lower()] = v.strip()
            expected_accept = base64.b64encode(
                hashlib.sha1((key + WEBSOCKET_GUID).encode("ascii")).digest()
            ).decode("ascii")
            if headers.get("sec-websocket-accept") != expected_accept:
                raise CdpError("CDP WebSocket handshake returned invalid Sec-WebSocket-Accept")
            self._sock = sock
            self._buf = bytearray(leftover)
            self._closed = False
        except Exception:
            try:
                sock.close()
            except OSError:
                pass
            raise

    def close(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is None:
            return
        if not self._closed:
            try:
                sock.sendall(_ws_encode_frame(0x8, b"", mask=True))
            except OSError:
                pass
        self._closed = True
        try:
            sock.close()
        except OSError:
            pass

    def _read_exact(self, n: int, deadline: float | None) -> bytes:
        assert self._sock is not None
        while len(self._buf) < n:
            if deadline is not None:
                rem = deadline - time.monotonic()
                if rem <= 0:
                    raise TimeoutError("CDP WebSocket read timed out")
                self._sock.settimeout(rem)
            else:
                self._sock.settimeout(self.timeout)
            chunk = self._sock.recv(max(4096, n - len(self._buf)))
            if not chunk:
                self._closed = True
                raise CdpError("CDP WebSocket closed unexpectedly")
            self._buf.extend(chunk)
        data = bytes(self._buf[:n])
        del self._buf[:n]
        return data

    def _recv_message(self, timeout: float | None = None) -> str | None:
        if self._sock is None or self._closed:
            return None
        deadline = None if timeout is None else (time.monotonic() + timeout)
        fragments: list[bytes] = []
        message_opcode: int | None = None
        while True:
            if not self._buf and timeout is not None:
                rem = (deadline - time.monotonic()) if deadline is not None else timeout
                if rem <= 0:
                    return None
                ready, _, _ = select.select([self._sock], [], [], rem)
                if not ready:
                    return None
            head = self._read_exact(2, deadline)
            b0, b1 = head[0], head[1]
            fin = bool(b0 & 0x80)
            opcode = b0 & 0x0F
            masked = bool(b1 & 0x80)
            length = b1 & 0x7F
            if length == 126:
                length = struct.unpack("!H", self._read_exact(2, deadline))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._read_exact(8, deadline))[0]
            mask_key = self._read_exact(4, deadline) if masked else b""
            payload = self._read_exact(length, deadline) if length else b""
            if masked and payload:
                payload = bytes(b ^ mask_key[i & 3] for i, b in enumerate(payload))

            if opcode == 0x8:  # close
                self._closed = True
                return None
            if opcode == 0x9:  # ping -> pong
                try:
                    self._sock.sendall(_ws_encode_frame(0xA, payload, mask=True))
                except OSError:
                    pass
                continue
            if opcode == 0xA:  # pong
                continue
            if opcode in (0x1, 0x2):
                message_opcode = opcode
                fragments = [payload]
            elif opcode == 0x0:
                fragments.append(payload)
            if fin and message_opcode is not None:
                full = b"".join(fragments)
                return full.decode("utf-8", errors="replace")

    def send(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> dict[str, Any]:
        if self._sock is None or self._closed:
            raise CdpError("CDP WebSocket is not connected")
        msg_id = self._next_id
        self._next_id += 1
        payload = json.dumps({"id": msg_id, "method": method, "params": params or {}}).encode("utf-8")
        self._sock.sendall(_ws_encode_frame(0x1, payload, mask=True))
        eff_timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + eff_timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                raise CdpError(f"CDP command {method} timed out")
            text = self._recv_message(timeout=rem)
            if text is None:
                raise CdpError(f"CDP WebSocket closed while awaiting {method}")
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("id") == msg_id:
                if "error" in msg:
                    raise CdpError(f"CDP {method} error: {msg['error']}")
                result = msg.get("result")
                return result if isinstance(result, dict) else {}
            if "method" in msg:
                self._events.append(msg)

    def drain_events(self, timeout: float = 0.0) -> list[dict[str, Any]]:
        """Return queued CDP events and optionally read new events for up to `timeout` seconds."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0 and not self._buf:
                break
            try:
                text = self._recv_message(timeout=max(0.0, rem))
            except (TimeoutError, OSError, CdpError):
                break
            if text is None:
                break
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and "method" in msg:
                self._events.append(msg)
            if timeout <= 0 and not self._buf:
                break
        drained = self._events
        self._events = []
        return drained


def wait_for_cdp_page_ws(
    cdp_port: int,
    *,
    timeout_s: float = CDP_READY_TIMEOUT_S,
    proc: Any = None,
) -> str:
    """Wait until Chrome exposes a page target on `http://127.0.0.1:<cdp_port>/json/list`."""
    deadline = time.monotonic() + timeout_s
    last_error = "no CDP response"
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RunnerError(f"Chrome exited prematurely with code {proc.returncode}")
        for endpoint in (f"http://127.0.0.1:{cdp_port}/json/list", f"http://127.0.0.1:{cdp_port}/json"):
            try:
                with urllib.request.urlopen(endpoint, timeout=1.5) as resp:
                    data = json.loads(resp.read().decode("utf-8", errors="replace"))
                if isinstance(data, list):
                    for target in data:
                        if (
                            isinstance(target, dict)
                            and target.get("type") == "page"
                            and isinstance(target.get("webSocketDebuggerUrl"), str)
                        ):
                            return target["webSocketDebuggerUrl"]
                last_error = "no page target yet"
            except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                last_error = str(exc)
        time.sleep(0.15)
    raise RunnerError(f"Chrome CDP endpoint did not become ready within {timeout_s:.1f}s ({sanitize_text(last_error)})")


# ---------------------------------------------------------------------------
# Browser telemetry probe & CDP session execution
# ---------------------------------------------------------------------------

TELEMETRY_PROBE_JS = r"""(() => {
  const safeText = (id) => {
    const el = document.getElementById(id);
    return el ? String(el.textContent || '').trim() : null;
  };
  let bootState = null;
  try {
    if (window.__playgta6BootState && typeof window.__playgta6BootState === 'object') {
      bootState = JSON.parse(JSON.stringify(window.__playgta6BootState));
    } else if (window.playgta6App && typeof window.playgta6App.getBootTelemetry === 'function') {
      bootState = JSON.parse(JSON.stringify(window.playgta6App.getBootTelemetry()));
    }
  } catch {}
  let diagnostics = null;
  try {
    if (window.playgta6App && typeof window.playgta6App.getDiagnosticsSnapshot === 'function') {
      diagnostics = JSON.parse(JSON.stringify(window.playgta6App.getDiagnosticsSnapshot()));
    }
  } catch {}
  let tracerSummary = null;
  try {
    if (window.playgta6App && window.playgta6App.bootTracer && typeof window.playgta6App.bootTracer.getSummary === 'function') {
      tracerSummary = JSON.parse(JSON.stringify(window.playgta6App.bootTracer.getSummary()));
    }
  } catch {}
  const hudLog = [];
  try {
    const logRoot = document.getElementById('hud-log');
    if (logRoot) {
      for (const child of Array.from(logRoot.children).slice(-60)) {
        hudLog.push(String(child.textContent || '').trim());
      }
    }
  } catch {}
  return {
    href: String(location.href || ''),
    readyState: String(document.readyState || ''),
    capabilities: {
      crossOriginIsolated: self.crossOriginIsolated === true,
      sharedArrayBuffer: typeof SharedArrayBuffer === 'function',
      webgpu: Boolean(typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu),
    },
    bootState,
    diagnostics,
    tracerSummary,
    hud: {
      stage: safeText('hud-boot-stage'),
      status: safeText('hud-status'),
      frames: safeText('metric-frames'),
      fps: safeText('metric-fps'),
      gpuErrors: safeText('gpu-validation-errors'),
      modeBadge: safeText('boot-mode-badge'),
      sideModules: safeText('side-module-status'),
      logLines: hudLog,
    },
  };
})()"""

FORCE_SMOKE_JS = r"""(async () => {
  try {
    const app = window.playgta6App;
    if (!app || !app.session) return 'no-app';
    const state = typeof app.session.getState === 'function' ? app.session.getState() : '';
    if (state === 'running_smoke') return 'already-smoke';
    if (typeof app.session.start === 'function') {
      await app.session.start({ preferSmoke: true });
      return 'started-smoke';
    }
    const btn = document.getElementById('btn-smoke-demo');
    if (btn && btn.dataset.active !== 'true') {
      btn.click();
      return 'clicked-smoke';
    }
    return state || 'unknown';
  } catch (err) {
    return 'error:' + String(err && err.message ? err.message : err);
  }
})()"""


@dataclass
class BrowserSessionResult:
    """Raw output of a live (or mocked) browser boot run."""

    ok: bool = True
    exit_code: int = 0
    elapsed_ms: float = 0.0
    status: str = "ok"  # ok | fault | error | timeout
    message: str = ""
    crashed: bool = False
    server_ready: dict[str, Any] = field(default_factory=dict)
    final_probe: dict[str, Any] = field(default_factory=dict)
    console_lines: list[str] = field(default_factory=list)
    page_errors: list[str] = field(default_factory=list)
    presents: list[dict[str, Any]] = field(default_factory=list)
    first_present_ms: float | None = None
    captured_frames: list[str] = field(default_factory=list)
    scripted_inputs_sent: int = 0
    mp4_path: str | None = None
    warnings: list[str] = field(default_factory=list)


def _format_console_event(params: Mapping[str, Any], elapsed_s: float) -> tuple[str, bool]:
    kind = str(params.get("type") or "log")
    parts: list[str] = []
    for arg in params.get("args") or []:
        if not isinstance(arg, Mapping):
            continue
        if "value" in arg and arg["value"] is not None:
            parts.append(str(arg["value"]))
        elif "description" in arg and arg["description"]:
            parts.append(str(arg["description"]))
        elif "type" in arg:
            parts.append(f"<{arg['type']}>")
    text = sanitize_text(" ".join(parts).strip())
    return f"[{elapsed_s:7.2f}s] [console.{kind}] {text}", (kind == "error")


def _format_exception_event(params: Mapping[str, Any], elapsed_s: float) -> str:
    details = params.get("exceptionDetails") or {}
    text = details.get("text") or "Uncaught exception"
    exc_obj = details.get("exception") or {}
    desc = exc_obj.get("description") or exc_obj.get("value") or ""
    msg = f"{text}: {desc}" if desc and str(desc) not in str(text) else str(text)
    return f"[{elapsed_s:7.2f}s] [pageerror] {sanitize_text(msg)}"


def _extract_present_count(probe: Mapping[str, Any]) -> int:
    candidates: list[int] = []
    boot_state = probe.get("bootState")
    if isinstance(boot_state, Mapping):
        for key in ("presentCount",):
            val = boot_state.get(key)
            if isinstance(val, (int, float)) and val >= 0:
                candidates.append(int(val))
        gpu = boot_state.get("gpu")
        if isinstance(gpu, Mapping) and isinstance(gpu.get("presentCount"), (int, float)):
            candidates.append(int(gpu["presentCount"]))
    diag = probe.get("diagnostics")
    if isinstance(diag, Mapping) and isinstance(diag.get("framesPresented"), (int, float)):
        candidates.append(int(diag["framesPresented"]))
    hud = probe.get("hud")
    if isinstance(hud, Mapping) and isinstance(hud.get("frames"), str) and hud["frames"].isdigit():
        candidates.append(int(hud["frames"]))
    return max(candidates) if candidates else 0


def _sample_present_record(probe: Mapping[str, Any], index: int, wall_ms: float) -> dict[str, Any]:
    boot_state = probe.get("bootState") if isinstance(probe.get("bootState"), Mapping) else {}
    gpu = boot_state.get("gpu") if isinstance(boot_state.get("gpu"), Mapping) else boot_state
    return {
        "index": index,
        "wall_ms": round(wall_ms, 2),
        "dispatches": 0,
        "import_calls": 0,
        "counters": {
            "draws_real_pack": int(gpu.get("realPackDrawCount") or 0),
            "draws_fallback": int(gpu.get("fallbackDrawCount") or 0),
            "draws_skipped": int(gpu.get("skippedDrawCount") or 0),
            "gpu_validation_errors": int(gpu.get("validationErrorCount") or 0),
        },
    }


def encode_frames_to_mp4(
    frames_dir: Path,
    out_mp4: Path,
    *,
    fps: float = 1.0,
    runner: CommandRunner = run_command,
    which_fn: WhichFn = shutil.which,
) -> tuple[bool, str | None]:
    """Encode captured `frame_0001.jpg` screenshots into an H.264 MP4 using `ffmpeg`."""
    ffmpeg = which_fn("ffmpeg")
    if not ffmpeg:
        return False, "ffmpeg not found on PATH; skipping MP4 encoding"
    if not frames_dir.is_dir() or not any(frames_dir.glob("frame_*.jpg")):
        return False, "no captured JPEG frames found for MP4 encoding"
    rate = max(1.0, round(fps, 3))
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(rate),
        "-i",
        str(frames_dir / "frame_%04d.jpg"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_mp4),
    ]
    rc = runner(cmd, frames_dir.parent, frames_dir.parent / "ffmpeg.log", 120.0)
    if rc != 0 or not out_mp4.is_file():
        return False, f"ffmpeg exited with code {rc}"
    return True, None


def run_browser_session(
    *,
    repo: Path,
    run_dir: Path,
    chrome_path: Path,
    port: int,
    boot_dir: Path | None,
    game_dir: Path | None,
    shaders_dir: Path | None,
    duration_s: float,
    headless: bool,
    smoke: bool,
    send_input: bool,
    capture_frames: bool,
    frame_interval_s: float,
    encode_mp4: bool,
    runner: CommandRunner = run_command,
    popen_fn: Callable[..., Any] = subprocess.Popen,
    cdp_factory: Callable[[str], CdpClient] = CdpClient,
    wait_server_fn: Callable[..., dict[str, Any]] = wait_for_server_ready,
    wait_cdp_fn: Callable[..., str] = wait_for_cdp_page_ws,
    which_fn: WhichFn = shutil.which,
) -> BrowserSessionResult:
    """Start `serve_local.py` + Chrome, drive CDP for `duration_s`, and always clean up."""
    host = "127.0.0.1"
    server_port = port if port > 0 else find_free_port(host)
    cdp_port = find_free_port(host)
    target_url = f"http://{host}:{server_port}/{'?smoke=1' if smoke else ''}"

    profile_dir = Path(tempfile.mkdtemp(prefix="playgta6-chrome-profile-"))
    smoke_boot_dir: Path | None = None
    if smoke:
        smoke_boot_dir = Path(tempfile.mkdtemp(prefix="playgta6-smoke-boot-"))
        effective_boot_dir: Path | None = smoke_boot_dir
        effective_game_dir: Path | None = None
    else:
        effective_boot_dir = boot_dir
        effective_game_dir = game_dir

    data_dir = repo / "local-data"
    web_dir = repo / "web"
    server_cmd = build_server_cmd(
        repo=repo,
        host=host,
        port=server_port,
        web_dir=web_dir,
        data_dir=data_dir,
        boot_dir=effective_boot_dir,
        game_dir=effective_game_dir,
        shaders_dir=shaders_dir,
    )
    chrome_cmd = build_chrome_args(
        chrome_path,
        user_data_dir=profile_dir,
        cdp_port=cdp_port,
        headless=headless,
        initial_url="about:blank",
    )

    result = BrowserSessionResult()
    server_proc: Any = None
    chrome_proc: Any = None
    cdp: CdpClient | None = None
    server_log_fh = None
    chrome_log_fh = None

    frames_dir = run_dir / "frames"
    want_frames = capture_frames or encode_mp4
    if want_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    start_mono = time.monotonic()
    try:
        server_log_fh = (run_dir / "server.log").open("wb")
        server_proc = popen_fn(
            server_cmd,
            cwd=repo,
            stdout=server_log_fh,
            stderr=subprocess.STDOUT,
        )
        result.server_ready = wait_server_fn(host, server_port, timeout_s=SERVER_READY_TIMEOUT_S, proc=server_proc)

        chrome_log_fh = (run_dir / "chrome.log").open("wb")
        chrome_proc = popen_fn(
            chrome_cmd,
            cwd=repo,
            stdout=chrome_log_fh,
            stderr=subprocess.STDOUT,
        )
        ws_url = wait_cdp_fn(cdp_port, timeout_s=CDP_READY_TIMEOUT_S, proc=chrome_proc)

        cdp = cdp_factory(ws_url)
        try:
            cdp.connect()
            cdp.send("Runtime.enable")
            cdp.send("Log.enable")
            cdp.send("Page.enable")
            cdp.send("Page.navigate", {"url": target_url})
        except CdpError as exc:
            result.ok = False
            result.crashed = True
            result.status = "error"
            result.exit_code = 1
            result.message = sanitize_text(str(exc))
            result.elapsed_ms = round((time.monotonic() - start_mono) * 1000.0, 2)
            return result

        nav_start = time.monotonic()
        end_time = nav_start + duration_s
        next_poll = nav_start + min(DEFAULT_POLL_INTERVAL_S, duration_s)
        next_frame = nav_start + min(frame_interval_s, duration_s) if want_frames else float("inf")
        next_input = nav_start + INPUT_START_S if send_input else float("inf")
        last_present_count = 0
        smoke_triggered = False

        def process_events(events: Sequence[dict[str, Any]]) -> None:
            now_s = max(0.0, time.monotonic() - nav_start)
            for ev in events:
                method = ev.get("method")
                params = ev.get("params") or {}
                if method == "Runtime.consoleAPICalled":
                    line, _is_err = _format_console_event(params, now_s)
                    result.console_lines.append(line)
                elif method == "Runtime.exceptionThrown":
                    line = _format_exception_event(params, now_s)
                    result.console_lines.append(line)
                    result.page_errors.append(line)
                elif method == "Log.entryAdded":
                    entry = params.get("entry") or {}
                    lvl = str(entry.get("level") or "info")
                    txt = sanitize_text(str(entry.get("text") or ""))
                    result.console_lines.append(f"[{now_s:7.2f}s] [log.{lvl}] {txt}")
                elif method == "Inspector.targetCrashed":
                    result.crashed = True
                    result.status = "error"
                    result.message = "browser target crashed"

        def evaluate_probe() -> dict[str, Any]:
            assert cdp is not None
            try:
                resp = cdp.send(
                    "Runtime.evaluate",
                    {"expression": TELEMETRY_PROBE_JS, "returnByValue": True},
                    timeout=5.0,
                )
                val = (resp.get("result") or {}).get("value")
                return val if isinstance(val, dict) else {}
            except CdpError:
                return {}

        while time.monotonic() < end_time and not result.crashed:
            if chrome_proc.poll() is not None:
                result.crashed = True
                result.status = "error"
                result.message = f"Chrome exited early with code {chrome_proc.returncode}"
                break

            now = time.monotonic()
            sleep_for = min(0.15, max(0.0, end_time - now))
            process_events(cdp.drain_events(timeout=sleep_for))
            now = time.monotonic()

            if now >= next_poll:
                probe = evaluate_probe()
                if probe:
                    result.final_probe = probe
                    if smoke and not smoke_triggered and probe.get("readyState") in ("interactive", "complete"):
                        state = (probe.get("bootState") or {}).get("sessionState")
                        if state and state != "running_smoke":
                            try:
                                cdp.send(
                                    "Runtime.evaluate",
                                    {"expression": FORCE_SMOKE_JS, "awaitPromise": True, "returnByValue": True},
                                    timeout=5.0,
                                )
                            except CdpError:
                                pass
                        smoke_triggered = True
                    present_count = _extract_present_count(probe)
                    wall_ms = (now - nav_start) * 1000.0
                    if present_count > last_present_count:
                        if result.first_present_ms is None:
                            result.first_present_ms = round(wall_ms, 2)
                        result.presents.append(_sample_present_record(probe, len(result.presents), wall_ms))
                        last_present_count = present_count
                    tracer = probe.get("tracerSummary") or {}
                    last_trap = tracer.get("lastTrap") if isinstance(tracer, Mapping) else None
                    if isinstance(last_trap, Mapping) and last_trap.get("fatal") is True:
                        result.status = "fault"
                        result.message = sanitize_text(str(last_trap.get("message") or last_trap.get("kind") or "fatal trap"))
                        break
                next_poll = now + DEFAULT_POLL_INTERVAL_S

            if send_input and now >= next_input:
                try:
                    cdp.send(
                        "Input.dispatchKeyEvent",
                        {"type": "rawKeyDown", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
                        timeout=3.0,
                    )
                    time.sleep(min(INPUT_HOLD_S, max(0.0, end_time - time.monotonic())))
                    cdp.send(
                        "Input.dispatchKeyEvent",
                        {"type": "keyUp", "key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13},
                        timeout=3.0,
                    )
                    result.scripted_inputs_sent += 1
                except CdpError:
                    pass
                next_input = now + INPUT_PERIOD_S

            if want_frames and now >= next_frame:
                try:
                    shot = cdp.send("Page.captureScreenshot", {"format": "jpeg", "quality": 85}, timeout=8.0)
                    b64_data = shot.get("data")
                    if isinstance(b64_data, str) and b64_data:
                        idx = len(result.captured_frames) + 1
                        fname = f"frame_{idx:04d}.jpg"
                        (frames_dir / fname).write_bytes(base64.b64decode(b64_data))
                        result.captured_frames.append(fname)
                except (CdpError, ValueError, OSError):
                    pass
                next_frame = now + frame_interval_s

        # Final telemetry sample + drain
        if not result.crashed:
            process_events(cdp.drain_events(timeout=0.05))
            final_probe = evaluate_probe()
            if final_probe:
                result.final_probe = final_probe
                present_count = _extract_present_count(final_probe)
                wall_ms = (time.monotonic() - nav_start) * 1000.0
                if present_count > last_present_count:
                    if result.first_present_ms is None:
                        result.first_present_ms = round(wall_ms, 2)
                    result.presents.append(_sample_present_record(final_probe, len(result.presents), wall_ms))

        result.elapsed_ms = round((time.monotonic() - start_mono) * 1000.0, 2)
        if result.crashed:
            result.ok = False
            result.exit_code = 1
        elif result.status == "fault":
            result.exit_code = 0

        if encode_mp4:
            fps = 1.0 / max(frame_interval_s, 0.1)
            ok_mp4, mp4_warn = encode_frames_to_mp4(
                frames_dir,
                run_dir / "capture.mp4",
                fps=fps,
                runner=runner,
                which_fn=which_fn,
            )
            if ok_mp4:
                result.mp4_path = "capture.mp4"
            elif mp4_warn:
                result.warnings.append(mp4_warn)
                print(f"[warn] {mp4_warn}", file=sys.stderr)

        return result
    finally:
        if cdp is not None:
            try:
                cdp.close()
            except Exception:
                pass
        terminate_process(chrome_proc)
        terminate_process(server_proc)
        for fh in (chrome_log_fh, server_log_fh):
            if fh is not None:
                try:
                    fh.close()
                except OSError:
                    pass
        shutil.rmtree(profile_dir, ignore_errors=True)
        if smoke_boot_dir is not None:
            shutil.rmtree(smoke_boot_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Report synthesis (playgta6-boot-report v2) & digest
# ---------------------------------------------------------------------------


def build_boot_report(
    session: BrowserSessionResult,
    *,
    smoke: bool,
    preset: str,
) -> dict[str, Any]:
    """Build a sanitized `playgta6-boot-report` v2 dict compatible with `compare_boot_reports.py`."""
    probe = session.final_probe if isinstance(session.final_probe, Mapping) else {}
    boot_state = probe.get("bootState") if isinstance(probe.get("bootState"), Mapping) else {}
    gpu = boot_state.get("gpu") if isinstance(boot_state.get("gpu"), Mapping) else boot_state
    tracer = probe.get("tracerSummary") if isinstance(probe.get("tracerSummary"), Mapping) else {}
    hud = probe.get("hud") if isinstance(probe.get("hud"), Mapping) else {}

    milestones: list[dict[str, Any]] = []
    raw_milestones = tracer.get("milestones")
    if isinstance(raw_milestones, list):
        reached = [
            m for m in raw_milestones
            if isinstance(m, Mapping) and m.get("reached") is True and m.get("milestone")
        ]
        reached.sort(key=lambda m: (float(m.get("elapsedMs") or 0.0), int(m.get("index") or 0)))
        for m in reached:
            milestones.append(
                {
                    "milestone": str(m["milestone"]),
                    "elapsed_ms": round(float(m.get("elapsedMs") or 0.0), 2),
                }
            )
    if session.first_present_ms is not None and not any(m["milestone"] == "FIRST_PRESENT" for m in milestones):
        milestones.append({"milestone": "FIRST_PRESENT", "elapsed_ms": round(session.first_present_ms, 2)})

    last_trap = tracer.get("lastTrap") if isinstance(tracer.get("lastTrap"), Mapping) else None
    fault: dict[str, Any] | None = None
    if isinstance(last_trap, Mapping) and last_trap.get("fatal") is True:
        fault = {
            "kind": str(last_trap.get("kind") or "GUEST_TRAP"),
            "trap_kind": str(last_trap.get("kind") or "GUEST_TRAP"),
            "va": last_trap.get("guestVa"),
            "detail": sanitize_text(str(last_trap.get("message") or "")),
            "exception_code": None,
            "function": None,
            "import": last_trap.get("import"),
        }

    boot_mode = str(boot_state.get("bootMode") or ("smoke-demo" if smoke else "idle"))
    phase = str(
        tracer.get("stage")
        or boot_state.get("milestone")
        or hud.get("stage")
        or ("SMOKE_DEMO" if smoke else "NOT_STARTED")
    )
    status = session.status
    if fault is not None:
        status = "fault"
    elif session.crashed:
        status = "error"

    present_count = max(len(session.presents), _extract_present_count(probe))
    first_present: dict[str, Any] | None = None
    if session.first_present_ms is not None or present_count > 0:
        first_present = {
            "import_seq": 0,
            "elapsed_ms": round(session.first_present_ms if session.first_present_ms is not None else session.elapsed_ms, 2),
            "imports": [],
        }

    console_errors = sum(1 for line in session.console_lines if "[console.error]" in line)
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "runner": "browser",
        "preset": preset,
        "mode": boot_mode,
        "smoke": bool(smoke),
        "status": status,
        "phase": phase,
        "exit_code": session.exit_code,
        "message": sanitize_text(session.message or str(hud.get("status") or "")),
        "elapsed_ms": round(float(session.elapsed_ms), 2),
        "milestones": milestones,
        "first_calls": [],
        "fault": fault,
        "backtrace": {"method": "browser", "frames": []},
        "counters": {
            "import_calls": 0,
            "imports": [],
            "presents": present_count,
            "real_pack_draws": int(gpu.get("realPackDrawCount") or 0),
            "fallback_draws": int(gpu.get("fallbackDrawCount") or 0),
            "unbound_bind_groups": int(gpu.get("unboundBindGroupCount") or 0),
            "skipped_draws": int(gpu.get("skippedDrawCount") or 0),
            "gpu_validation_errors": int(gpu.get("validationErrorCount") or 0),
            "page_errors": len(session.page_errors),
            "console_errors": console_errors,
        },
        "address_space": {
            "failures": 0,
            "reserved_peak": 0,
            "events": 0,
            "log": [],
        },
        "presents": list(session.presents),
        "first_present": first_present,
    }
    return sanitize_value(report)


def build_telemetry_payload(
    session: BrowserSessionResult,
    *,
    smoke: bool,
    preset: str,
    headless: bool,
) -> dict[str, Any]:
    """Build sanitized `telemetry.json` artifact."""
    probe = session.final_probe if isinstance(session.final_probe, Mapping) else {}
    return sanitize_value(
        {
            "schema": "playgta6-browser-telemetry",
            "schema_version": 1,
            "preset": preset,
            "smoke": smoke,
            "headless": headless,
            "elapsed_ms": round(float(session.elapsed_ms), 2),
            "status": session.status,
            "crashed": session.crashed,
            "server": session.server_ready,
            "capabilities": probe.get("capabilities") or {},
            "boot_state": probe.get("bootState"),
            "diagnostics": probe.get("diagnostics"),
            "boot_tracer": probe.get("tracerSummary"),
            "hud": probe.get("hud"),
            "presents_sampled": len(session.presents),
            "first_present_ms": session.first_present_ms,
            "scripted_inputs_sent": session.scripted_inputs_sent,
            "captured_frames": list(session.captured_frames),
            "mp4": session.mp4_path,
            "page_error_count": len(session.page_errors),
            "warnings": list(session.warnings),
        }
    )


def load_report(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def digest(report: dict[str, Any], exit_code: int, max_counters: int = 12) -> str:
    """Format a compact human-readable run digest (mirroring `run_native_boot.digest`)."""
    lines = [
        f"status: {report.get('status')} (phase {report.get('phase')}, mode {report.get('mode')}, "
        f"exit_code {report.get('exit_code')}, runner exit {exit_code}), {report.get('elapsed_ms')} ms"
    ]
    if report.get("message"):
        lines.append(f"message: {report['message']}")
    lines.append("milestones:")
    milestones = report.get("milestones") or []
    if not milestones:
        lines.append("  (none)")
    for m in milestones:
        seq = f", import call {m['import_seq']}" if "import_seq" in m else ""
        lines.append(f"  {m.get('milestone')}: {m.get('elapsed_ms')} ms{seq}")
    presents = report.get("presents") or []
    fp = report.get("first_present")
    first = f", first at {fp.get('elapsed_ms')} ms" if isinstance(fp, dict) else ""
    counters = report.get("counters") or {}
    total_presents = counters.get("presents", len(presents))
    lines.append(f"presents: {total_presents} ({len(presents)} samples{first})")
    if presents:
        last = presents[-1]
        lines.append(f"  last: wall {last.get('wall_ms')} ms")
        draw_counters = {
            k: v for k, v in (last.get("counters") or {}).items()
            if re.search(r"draw|dispatch|upload|copy|clear|error", k) and v
        }
        for k, v in list(draw_counters.items())[:max_counters]:
            lines.append(f"  {k}: {v}")
    lines.append(
        "counters: "
        f"real_pack_draws={counters.get('real_pack_draws', 0)}, "
        f"fallback_draws={counters.get('fallback_draws', 0)}, "
        f"gpu_validation_errors={counters.get('gpu_validation_errors', 0)}, "
        f"page_errors={counters.get('page_errors', 0)}"
    )
    fault = report.get("fault")
    if isinstance(fault, dict):
        lines.append(f"stop: {fault.get('kind')} at {fault.get('va')} ({fault.get('detail')})")
    else:
        lines.append("stop: no fault")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------


def compare_with_baseline(args: argparse.Namespace, run_dir: Path, runs_dir: Path) -> int:
    """Compare `<run_dir>/boot-report.json` against `--baseline` via `compare_boot_reports`."""
    base = Path(args.baseline)
    if not base.is_dir():
        base = runs_dir / args.baseline
    if not (base / "boot-report.json").is_file():
        raise RunnerError(f"baseline run not found: {sanitize_text(str(args.baseline))}")
    argv = [str(base / "boot-report.json"), str(run_dir / "boot-report.json")]
    for pattern in args.ignore:
        argv += ["--ignore", pattern]
    if args.time_tolerance is not None:
        argv += ["--time-tolerance", str(args.time_tolerance)]
    if args.count_tolerance is not None:
        argv += ["--count-tolerance", str(args.count_tolerance)]
    print(f"[compare] against {basename_label(base)}")
    code = compare_boot_reports.main(argv)
    return 1 if code == 1 else (2 if code == 2 else 0)


# ---------------------------------------------------------------------------
# CLI & main entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Rebuild stale browser Wasm64 artifacts, start local COOP/COEP server + Chrome CDP, "
            "collect logs/telemetry/boot-report.json, and optionally compare against a baseline."
        ),
        epilog="Exit code: 0 ok, 1 baseline differences, 2 usage or build error, 3 no report.",
    )
    p.add_argument("--preset", choices=sorted(PRESETS), default="smoke")
    p.add_argument("--name", help="run name (default <preset>-<timestamp>)")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run synthetic browser smoke boot with zero game files (skips recomp/Wasm64 build stages)",
    )
    p.add_argument("--exe", type=Path, help="game executable for recompilation (default $PLAYGTA6_EXE)")
    p.add_argument("--boot-dir", type=Path, help="recompiled boot directory served at /boot/ (default work/recomp_build)")
    p.add_argument("--game-dir", type=Path, help="local game directory served at /game/ (default $GTA6_GAME_DIR)")
    p.add_argument("--shader-pack", type=Path, help="WGPK shader pack (+ optional <pack>.reflect.json sidecar)")
    p.add_argument("--out-dir", type=Path, help="Wasm64 output directory (default web/dist)")
    p.add_argument("--build-dir", type=Path, help="browser build cache directory (default work/browser-boot)")
    p.add_argument("--runs-dir", type=Path, help="run directories (default work/browser-boot/runs)")
    p.add_argument("--runtime-lib", type=Path, help="override PIC libplaygta6_core.a (skips runtime CMake build)")
    p.add_argument("--port", type=int, default=0, help="local server port (default 0 = ephemeral free port)")
    p.add_argument("--duration-s", type=float, default=None, help="run duration in seconds (default from preset)")
    p.add_argument(
        "--frame-interval-s",
        type=float,
        default=DEFAULT_FRAME_INTERVAL_S,
        help="seconds between screenshots when --capture-frames or --mp4 is set (default 1.0)",
    )
    head_group = p.add_mutually_exclusive_group()
    head_group.add_argument("--headless", dest="headless", action="store_true", default=True, help="launch Chrome headless (default)")
    head_group.add_argument("--headed", dest="headless", action="store_false", help="launch Chrome with a visible window")
    p.add_argument("--chrome", type=Path, help="Chrome/Chromium/Edge binary (default $CHROME_PATH or auto-discover)")
    p.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel build jobs")
    p.add_argument("--trust-build", action="store_true", help="skip build staleness checks and use existing artifacts")
    p.add_argument("--rebuild", action="store_true", help="force rebuilding all real-game stages")
    p.add_argument("--no-build", action="store_true", help="fail instead of building any stale stage")
    p.add_argument("--dry-run", action="store_true", help="print sanitized build/launch plan as JSON and exit")
    p.add_argument("--capture-frames", action="store_true", help="capture periodic JPEG screenshots into frames/")
    p.add_argument("--mp4", action="store_true", help="encode captured frames to capture.mp4 via ffmpeg")
    p.add_argument("--baseline", help="earlier run (name in runs-dir or directory path) to compare against")
    p.add_argument("--ignore", action="append", default=[], metavar="PATTERN", help="compare_boot_reports --ignore")
    p.add_argument("--time-tolerance", type=float, metavar="REL", help="compare_boot_reports --time-tolerance")
    p.add_argument("--count-tolerance", type=float, metavar="REL", help="compare_boot_reports --count-tolerance")
    return p


def _resolve_game_dir(args: argparse.Namespace, env: Mapping[str, str], repo: Path) -> Path | None:
    if args.game_dir is not None:
        return resolve_against_repo(args.game_dir, repo)
    for key in ("GTA6_GAME_DIR", "PLAYGTA6_GAME_DIR"):
        raw = (env.get(key) or "").strip()
        if raw:
            return resolve_against_repo(raw, repo)
    dotenv_val = load_dotenv_value(repo, "GTA6_GAME_DIR")
    if dotenv_val:
        return resolve_against_repo(dotenv_val, repo)
    return None


def main(
    argv: Sequence[str] | None = None,
    *,
    runner: CommandRunner = run_command,
    session_runner: Callable[..., BrowserSessionResult] | None = None,
    chrome_discoverer: Callable[..., Path | None] = discover_chrome,
    env: Mapping[str, str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> int:
    env_map = os.environ if env is None else env
    try:
        args = build_arg_parser().parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2
    try:
        return _run(
            args,
            runner=runner,
            session_runner=session_runner or run_browser_session,
            chrome_discoverer=chrome_discoverer,
            env=env_map,
            repo=repo_root.resolve(),
        )
    except RunnerError as exc:
        print(f"error: {sanitize_text(str(exc))}", file=sys.stderr)
        return 2


def _run(
    args: argparse.Namespace,
    *,
    runner: CommandRunner,
    session_runner: Callable[..., BrowserSessionResult],
    chrome_discoverer: Callable[..., Path | None],
    env: Mapping[str, str],
    repo: Path,
) -> int:
    if args.rebuild and args.trust_build:
        raise RunnerError("cannot combine --rebuild and --trust-build")
    if args.rebuild and args.no_build:
        raise RunnerError("cannot combine --rebuild and --no-build")
    if args.port < 0 or args.port > 65535:
        raise RunnerError(f"invalid --port {args.port} (expected 0..65535)")
    if args.duration_s is not None and args.duration_s <= 0:
        raise RunnerError("--duration-s must be > 0")
    if args.frame_interval_s <= 0:
        raise RunnerError("--frame-interval-s must be > 0")
    if args.jobs < 1:
        raise RunnerError("--jobs must be >= 1")

    preset_cfg = PRESETS[args.preset]
    if args.duration_s is not None:
        duration_s = float(args.duration_s)
    elif args.smoke and args.preset == "smoke":
        duration_s = SMOKE_MODE_DEFAULT_DURATION_S
    else:
        duration_s = float(preset_cfg["duration_s"])
    send_input = bool(preset_cfg["input"])

    build_dir = resolve_against_repo(args.build_dir or DEFAULT_BUILD_DIR_REL, repo)
    runs_dir = resolve_against_repo(args.runs_dir or (build_dir / "runs"), repo)
    boot_dir = resolve_against_repo(args.boot_dir or DEFAULT_BOOT_DIR_REL, repo)
    out_dir = resolve_against_repo(args.out_dir or DEFAULT_OUT_DIR_REL, repo)
    runtime_lib_override = resolve_against_repo(args.runtime_lib, repo) if args.runtime_lib else None

    exe: Path | None = None
    if args.exe is not None:
        exe = resolve_against_repo(args.exe, repo)
    elif env.get("PLAYGTA6_EXE"):
        exe = resolve_against_repo(env["PLAYGTA6_EXE"], repo)
    game_dir = None if args.smoke else _resolve_game_dir(args, env, repo)
    shader_pack = resolve_against_repo(args.shader_pack, repo) if args.shader_pack else None
    shader_stage_dir = build_dir / "staged-shaders"

    name = args.name or f"{'smoke' if args.smoke and args.preset == 'smoke' else args.preset}-{time.strftime('%Y%m%d-%H%M%S')}"
    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise RunnerError(f"invalid run name {name!r}")
    run_dir = runs_dir / name
    if not args.dry_run and run_dir.exists() and any(run_dir.iterdir()):
        raise RunnerError(f"run directory {basename_label(run_dir)} already exists (pick another --name)")

    if not args.smoke and not args.dry_run:
        if exe is not None and not exe.is_file():
            raise RunnerError(f"exe not found: {basename_label(exe)}")
        if args.game_dir is not None and (game_dir is None or not game_dir.is_dir()):
            raise RunnerError(f"game directory not found: {basename_label(args.game_dir)}")
    if shader_pack is not None and not args.dry_run and not shader_pack.is_file():
        raise RunnerError(f"shader pack not found: {basename_label(shader_pack)}")

    if args.trust_build or args.smoke:
        plan = Plan()
    else:
        plan, _runtime_lib = make_build_plan(
            repo=repo,
            build_dir=build_dir,
            boot_dir=boot_dir,
            out_dir=out_dir,
            exe=exe,
            game_dir=game_dir,
            runtime_lib_override=runtime_lib_override,
            jobs=args.jobs,
            force=args.rebuild,
            smoke=args.smoke,
        )

    shader_plan = plan_shader_staging(shader_pack, shader_stage_dir) if shader_pack is not None else []
    chrome_path = chrome_discoverer(args.chrome, env=env)
    sanitized_plan = plan.sanitized(repo)

    for step in sanitized_plan:
        print(f"[plan] {step['stage']}: {step['reason']}", file=sys.stderr)

    effective_shaders_dir = shader_stage_dir if shader_pack is not None else None
    if args.dry_run:
        preview_port = args.port if args.port > 0 else 0
        server_preview = build_server_cmd(
            repo=repo,
            host="127.0.0.1",
            port=preview_port,
            web_dir=repo / "web",
            data_dir=repo / "local-data",
            boot_dir=None if args.smoke else boot_dir,
            game_dir=game_dir,
            shaders_dir=effective_shaders_dir,
        )
        dry_payload = sanitize_value(
            {
                "mode": "smoke" if args.smoke else "real-game",
                "preset": args.preset,
                "run_name": name,
                "run_dir": repo_relative_label(run_dir, repo),
                "duration_s": duration_s,
                "headless": args.headless,
                "send_input": send_input,
                "capture_frames": bool(args.capture_frames or args.mp4),
                "mp4": bool(args.mp4),
                "trust_build": bool(args.trust_build),
                "port": preview_port,
                "chrome_found": chrome_path is not None,
                "chrome_label": basename_label(chrome_path) if chrome_path else None,
                "plan": sanitized_plan,
                "shader_staging": [
                    {"kind": s["kind"], "name": s["name"]} for s in shader_plan
                ],
                "server_command": sanitize_cmd(server_preview, repo),
                "url": f"http://127.0.0.1:{preview_port or '<ephemeral>'}/{'?smoke=1' if args.smoke else ''}",
            }
        )
        print(json.dumps(dry_payload, indent=2))
        return 0

    if plan.steps and args.no_build:
        raise RunnerError("stale stages with --no-build: " + ", ".join(s["stage"] for s in plan.steps))

    if chrome_path is None:
        raise RunnerError(
            "Chrome/Chromium/Edge executable not found (pass --chrome PATH or set CHROME_PATH)"
        )

    for step in plan.steps:
        sanitized_step_cmd = " ".join(sanitize_cmd(step["cmd"], repo))
        print(f"[build] {step['stage']}: {sanitized_step_cmd}", file=sys.stderr, flush=True)
        if runner(step["cmd"], repo, None, None) != 0:
            raise RunnerError(f"{step['stage']} failed")

    staged_shaders: list[dict[str, Any]] = []
    if shader_pack is not None:
        staged_shaders = stage_shader_pack(shader_pack, shader_stage_dir)

    if not args.smoke and not (out_dir / "runtime.wasm").is_file():
        raise RunnerError(
            f"runtime.wasm not found in {repo_relative_label(out_dir, repo)} after build "
            "(build Wasm64 runtime or pass --smoke)"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {repo_relative_label(run_dir, repo)}", file=sys.stderr, flush=True)

    session = session_runner(
        repo=repo,
        run_dir=run_dir,
        chrome_path=chrome_path,
        port=args.port,
        boot_dir=None if args.smoke else boot_dir,
        game_dir=game_dir,
        shaders_dir=effective_shaders_dir,
        duration_s=duration_s,
        headless=args.headless,
        smoke=args.smoke,
        send_input=send_input,
        capture_frames=bool(args.capture_frames),
        frame_interval_s=float(args.frame_interval_s),
        encode_mp4=bool(args.mp4),
        runner=runner,
    )

    # Always write console.log, telemetry.json, and command.json
    console_lines = list(session.console_lines)
    hud_lines = ((session.final_probe or {}).get("hud") or {}).get("logLines")
    if isinstance(hud_lines, list) and hud_lines:
        console_lines.append("--- HUD LOG ---")
        console_lines.extend(sanitize_text(str(line)) for line in hud_lines)
    (run_dir / "console.log").write_text(
        ("\n".join(sanitize_text(line) for line in console_lines) + "\n") if console_lines else "",
        encoding="utf-8",
    )

    telemetry_payload = build_telemetry_payload(
        session,
        smoke=args.smoke,
        preset=args.preset,
        headless=args.headless,
    )
    (run_dir / "telemetry.json").write_text(json.dumps(telemetry_payload, indent=2) + "\n", encoding="utf-8")

    command_meta = sanitize_value(
        {
            "preset": args.preset,
            "smoke": bool(args.smoke),
            "name": name,
            "duration_s": duration_s,
            "headless": bool(args.headless),
            "send_input": send_input,
            "capture_frames": bool(args.capture_frames or args.mp4),
            "mp4": bool(args.mp4),
            "trust_build": bool(args.trust_build),
            "chrome": basename_label(chrome_path),
            "plan": sanitized_plan,
            "staged_shaders": staged_shaders,
            "exit_code": session.exit_code,
        }
    )
    (run_dir / "command.json").write_text(json.dumps(command_meta, indent=2) + "\n", encoding="utf-8")

    report_path = run_dir / "boot-report.json"
    if session.ok or session.final_probe or session.presents or session.status in ("ok", "fault"):
        report = build_boot_report(session, smoke=args.smoke, preset=args.preset)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    loaded_report = load_report(report_path)
    if loaded_report is None:
        print(
            f"error: browser session (exit {session.exit_code}) wrote no report; "
            f"see {repo_relative_label(run_dir / 'console.log', repo)}",
            file=sys.stderr,
        )
        return 3

    digest_text = digest(loaded_report, session.exit_code)
    (run_dir / "digest.txt").write_text(digest_text + "\n", encoding="utf-8")
    print(digest_text)

    if args.baseline:
        return compare_with_baseline(args, run_dir, runs_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
