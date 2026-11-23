#!/usr/bin/env python3
"""Local preflight and milestone readiness checker for playgta6web (Track 5 / W13).

Checks installed contributor toolchains, `.env` / environment variable configuration,
built Wasm64/asset/shader artifacts, local COOP/COEP server headers, and M0-M3
milestone readiness.

Security & privacy invariant:
- Machine-readable `--json` output NEVER includes raw host filesystem paths or usernames.
- Strictly uses Python 3.12 standard library modules only.
"""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
WASM_MAGIC = b"\x00asm"

TOOLCHAIN_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("python", ("python", "python3")),
    ("node", ("node",)),
    ("rustc", ("rustc",)),
    ("cargo", ("cargo",)),
    ("cmake", ("cmake",)),
    ("emcc", ("emcc",)),
    ("emcmake", ("emcmake",)),
    ("gh", ("gh",)),
)

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)


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


def sanitize_version(raw: str | None) -> str | None:
    """Sanitize a tool `--version` output line so it never leaks paths or usernames."""
    if not raw:
        return None
    first_line = ""
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            first_line = stripped
            break
    if not first_line:
        return None

    sanitized = _WINDOWS_PATH_RE.sub("<path>", first_line)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)

    for username in _current_usernames():
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])", re.IGNORECASE)
        sanitized = pattern.sub("<user>", sanitized)

    sanitized = " ".join(sanitized.split())
    if len(sanitized) > 160:
        sanitized = sanitized[:157].rstrip() + "..."
    return sanitized or None


def _probe_binary_version(resolved_bin: str) -> str | None:
    try:
        proc = subprocess.run(
            [resolved_bin, "--version"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if proc.returncode != 0:
        return None
    output = proc.stdout.strip() or proc.stderr.strip()
    return sanitize_version(output)


def check_toolchains(
    *,
    which_fn: Callable[[str], str | None] = shutil.which,
    version_fn: Callable[[str], str | None] | None = None,
) -> dict[str, dict[str, Any]]:
    """Detect required contributor toolchains and return sanitized availability/version info."""
    use_active_python = which_fn is shutil.which and version_fn is None
    probe_version = version_fn if version_fn is not None else _probe_binary_version
    results: dict[str, dict[str, Any]] = {}

    for tool_name, candidates in TOOLCHAIN_SPECS:
        if tool_name == "python" and use_active_python:
            results["python"] = {
                "available": True,
                "version": sanitize_version(f"Python {platform.python_version()}"),
            }
            continue

        resolved: str | None = None
        for bin_name in candidates:
            found = which_fn(bin_name)
            if found:
                resolved = found
                break

        if not resolved:
            results[tool_name] = {"available": False, "version": None}
            continue

        version = sanitize_version(probe_version(resolved))
        results[tool_name] = {"available": True, "version": version}

    return results


def load_dotenv(env_path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE `.env` file without mutating `os.environ`."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def _resolve_path_setting(raw_value: str, repo_root: Path) -> Path:
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _resolve_env_paths(
    repo_root: Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    env_map = os.environ if env is None else env
    env_file = repo_root / ".env"
    dotenv_vals = load_dotenv(env_file)

    raw_game = (env_map.get("GTA6_GAME_DIR") or dotenv_vals.get("GTA6_GAME_DIR", "")).strip()
    raw_data = (env_map.get("GTA6_DATA_DIR") or dotenv_vals.get("GTA6_DATA_DIR", "")).strip()
    raw_work = (env_map.get("GTA6_WORK_DIR") or dotenv_vals.get("GTA6_WORK_DIR", "")).strip()

    game_path = _resolve_path_setting(raw_game, repo_root) if raw_game else None
    data_path = _resolve_path_setting(raw_data, repo_root) if raw_data else (repo_root / "local-data")
    work_path = _resolve_path_setting(raw_work, repo_root) if raw_work else (repo_root / "work")

    return {
        "env_file_exists": env_file.is_file(),
        "raw_game": raw_game,
        "raw_data": raw_data,
        "raw_work": raw_work,
        "game_path": game_path,
        "data_path": data_path,
        "work_path": work_path,
    }


def _inspect_game_dir_summary(game_path: Path | None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "hasMicrosoftGameConfig": False,
        "rpfCount": 0,
        "exeCount": 0,
        "xvcXvdCount": 0,
        "fileCount": 0,
    }
    if game_path is None or not game_path.is_dir():
        return summary

    def _record_file(path: Path) -> None:
        summary["fileCount"] += 1
        lower = path.name.lower()
        if lower == "microsoftgame.config":
            summary["hasMicrosoftGameConfig"] = True
        elif lower.endswith(".rpf"):
            summary["rpfCount"] += 1
        elif lower.endswith(".exe"):
            summary["exeCount"] += 1
        elif lower.endswith((".xvc", ".xvd")):
            summary["xvcXvdCount"] += 1

    # Shallow scan (root + 1 level) so multi-hundred-GB game folders stay instant.
    try:
        entries = list(game_path.iterdir())
    except OSError:
        return summary

    for entry in entries:
        try:
            if entry.is_file():
                _record_file(entry)
            elif entry.is_dir():
                for sub in entry.iterdir():
                    if sub.is_file():
                        _record_file(sub)
        except OSError:
            continue

    return summary


def check_local_env(
    repo_root: Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Check `.env` and `GTA6_*` env vars, returning sanitized booleans and counts only."""
    resolved = _resolve_env_paths(repo_root, env=env)
    game_path: Path | None = resolved["game_path"]
    data_path: Path = resolved["data_path"]
    work_path: Path = resolved["work_path"]

    game_configured = bool(resolved["raw_game"])
    game_exists = bool(game_path is not None and game_path.is_dir())
    data_exists = data_path.is_dir()
    work_exists = work_path.is_dir()

    return {
        "envFileExists": bool(resolved["env_file_exists"]),
        "gameDirConfigured": game_configured,
        "gameDirExists": game_exists,
        "dataDirConfigured": bool(resolved["raw_data"]) or data_exists,
        "dataDirExists": data_exists,
        "workDirConfigured": bool(resolved["raw_work"]) or work_exists,
        "workDirExists": work_exists,
        "gameSummary": _inspect_game_dir_summary(game_path if game_exists else None),
    }


def _check_binary_artifact(path: Path, *, check_wasm_magic: bool = False) -> dict[str, Any]:
    present = path.is_file()
    size_bytes = 0
    valid_magic = False
    if present:
        try:
            size_bytes = path.stat().st_size
            if check_wasm_magic and size_bytes >= len(WASM_MAGIC):
                with path.open("rb") as fh:
                    valid_magic = fh.read(len(WASM_MAGIC)) == WASM_MAGIC
        except OSError:
            size_bytes = 0
            valid_magic = False

    result: dict[str, Any] = {
        "present": present,
        "nonEmpty": present and size_bytes > 0,
        "sizeBytes": size_bytes,
    }
    if check_wasm_magic:
        result["validWasmMagic"] = valid_magic
    return result


def _check_data_manifest(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "present": False,
        "nonEmpty": False,
        "sizeBytes": 0,
        "validJson": False,
        "fileCount": None,
        "abiVersion": None,
        "error": None,
    }
    if not path.is_file():
        return result

    result["present"] = True
    try:
        size_bytes = path.stat().st_size
    except OSError:
        result["error"] = "unreadable file"
        return result

    result["sizeBytes"] = size_bytes
    result["nonEmpty"] = size_bytes > 0
    if size_bytes == 0:
        result["error"] = "empty file"
        return result

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        result["error"] = "invalid JSON"
        return result
    except (OSError, UnicodeDecodeError):
        result["error"] = "unreadable file"
        return result

    if not isinstance(parsed, dict):
        result["error"] = "expected JSON object"
        return result

    files_field = parsed.get("files")
    if files_field is not None and not isinstance(files_field, list):
        result["error"] = "invalid files field"
        return result

    file_count: int | None = None
    if isinstance(files_field, list):
        file_count = len(files_field)
    else:
        raw_count = parsed.get("file_count")
        if isinstance(raw_count, int) and not isinstance(raw_count, bool) and raw_count >= 0:
            file_count = raw_count

    abi_version: int | None = None
    raw_abi = parsed.get("abi_version")
    if isinstance(raw_abi, int) and not isinstance(raw_abi, bool):
        abi_version = raw_abi

    result["validJson"] = True
    result["fileCount"] = file_count
    result["abiVersion"] = abi_version
    return result


def _check_bootset(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "present": False,
        "nonEmpty": False,
        "sizeBytes": 0,
        "validJson": False,
        "entryCount": None,
        "error": None,
    }
    if not path.is_file():
        return result

    result["present"] = True
    try:
        size_bytes = path.stat().st_size
    except OSError:
        result["error"] = "unreadable file"
        return result

    result["sizeBytes"] = size_bytes
    result["nonEmpty"] = size_bytes > 0
    if size_bytes == 0:
        result["error"] = "empty file"
        return result

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        result["error"] = "invalid JSON"
        return result
    except (OSError, UnicodeDecodeError):
        result["error"] = "unreadable file"
        return result

    if isinstance(parsed, list):
        result["validJson"] = True
        result["entryCount"] = len(parsed)
        return result

    if isinstance(parsed, dict) and isinstance(parsed.get("entries"), list):
        result["validJson"] = True
        result["entryCount"] = len(parsed["entries"])
        return result

    result["error"] = "expected JSON array"
    return result


def _count_shader_packs(web_root: Path, data_dir: Path) -> dict[str, Any]:
    candidate_dirs = (
        data_dir,
        data_dir / "shaders",
        web_root / "dist",
        web_root / "shaders",
    )
    seen: set[Path] = set()
    count = 0
    non_empty_count = 0

    for directory in candidate_dirs:
        if not directory.is_dir():
            continue
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            for entry in directory.iterdir():
                if entry.is_file() and entry.name.lower().endswith(".wgsl.pack"):
                    count += 1
                    try:
                        if entry.stat().st_size > 0:
                            non_empty_count += 1
                    except OSError:
                        pass
        except OSError:
            continue

    return {
        "present": count > 0,
        "count": count,
        "nonEmptyCount": non_empty_count,
    }


def check_artifacts(repo_root: Path, data_dir: Path | None = None) -> dict[str, Any]:
    """Check presence, non-empty status, and validity of build/runtime/asset artifacts."""
    web_root = repo_root / "web"
    resolved_data_dir = data_dir if data_dir is not None else (repo_root / "local-data")
    dist_dir = web_root / "dist"

    return {
        "runtimeJs": _check_binary_artifact(dist_dir / "runtime.js", check_wasm_magic=False),
        "runtimeWasm": _check_binary_artifact(dist_dir / "runtime.wasm", check_wasm_magic=True),
        "dataManifest": _check_data_manifest(resolved_data_dir / "data-manifest.json"),
        "bootset": _check_bootset(resolved_data_dir / "bootset.json"),
        "shaderPacks": _count_shader_packs(web_root, resolved_data_dir),
    }


def _check_headers_in_process(handler_cls: type) -> dict[str, str]:
    """Fallback header probe if binding a local ephemeral socket is disallowed."""
    captured: dict[str, str] = {}

    class _ProbeHandler(handler_cls):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            self.request_version = "HTTP/1.1"
            self._headers_buffer: list[bytes] = []

        def send_header(self, keyword: str, value: str) -> None:
            captured[keyword.lower()] = value

        def flush_headers(self) -> None:
            return None

    probe = _ProbeHandler()
    handler_cls.end_headers(probe)
    return captured


def check_server_headers(
    repo_root: Path,
    data_dir: Path | None = None,
) -> dict[str, Any]:
    """Verify `server/serve_local.py` imports cleanly and emits COOP/COEP isolation headers."""
    result: dict[str, Any] = {
        "checked": True,
        "serverScriptPresent": False,
        "importable": False,
        "mode": None,
        "coopValid": False,
        "coepValid": False,
        "corpValid": False,
        "acceptRangesValid": False,
        "statusEndpointOk": False,
        "ok": False,
        "error": None,
    }

    server_file = repo_root / "server" / "serve_local.py"
    if not server_file.is_file():
        result["error"] = "server/serve_local.py not found"
        return result
    result["serverScriptPresent"] = True

    spec = importlib.util.spec_from_file_location("_playgta6_preflight_serve_local", server_file)
    if spec is None or spec.loader is None:
        result["error"] = "failed to load server module spec"
        return result

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        result["error"] = "server module import failed"
        return result

    handler_cls = getattr(module, "PlayGta6Handler", None)
    if handler_cls is None:
        result["error"] = "PlayGta6Handler not found in server module"
        return result
    result["importable"] = True

    web_root = repo_root / "web"
    resolved_data_dir = data_dir if data_dir is not None else (repo_root / "local-data")

    class _QuietHandler(handler_cls):  # type: ignore[misc,valid-type]
        def log_message(self, format: str, *args: Any) -> None:
            return None

    httpd: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    try:
        handler_factory = partial(_QuietHandler, web_root=web_root, data_root=resolved_data_dir)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            conn.request("GET", "/api/status")
            resp = conn.getresponse()
            body = resp.read()
            headers = {k.lower(): v for k, v in resp.getheaders()}
        finally:
            conn.close()

        result["mode"] = "ephemeral-http"
        result["coopValid"] = headers.get("cross-origin-opener-policy") == "same-origin"
        result["coepValid"] = headers.get("cross-origin-embedder-policy") == "require-corp"
        result["corpValid"] = headers.get("cross-origin-resource-policy") == "same-origin"
        result["acceptRangesValid"] = headers.get("accept-ranges") == "bytes"

        status_ok = False
        if resp.status == 200:
            try:
                payload = json.loads(body.decode("utf-8"))
                status_ok = bool(isinstance(payload, dict) and payload.get("ok") and payload.get("coopCoep"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                status_ok = False
        result["statusEndpointOk"] = status_ok
        result["ok"] = bool(result["coopValid"] and result["coepValid"] and status_ok)
        if not result["ok"]:
            result["error"] = "required COOP/COEP headers or /api/status check failed"
        return result
    except OSError:
        # Socket bind unavailable in restricted environments: fall back to in-process header check.
        try:
            headers = _check_headers_in_process(handler_cls)
            result["mode"] = "in-process"
            result["coopValid"] = headers.get("cross-origin-opener-policy") == "same-origin"
            result["coepValid"] = headers.get("cross-origin-embedder-policy") == "require-corp"
            result["corpValid"] = headers.get("cross-origin-resource-policy") == "same-origin"
            result["acceptRangesValid"] = headers.get("accept-ranges") == "bytes"
            result["statusEndpointOk"] = callable(getattr(module, "build_status_payload", None))
            result["ok"] = bool(result["coopValid"] and result["coepValid"])
            if not result["ok"]:
                result["error"] = "required COOP/COEP headers missing"
        except Exception:
            result["error"] = "server header self-check failed"
        return result
    except Exception:
        result["error"] = "server header self-check failed"
        return result
    finally:
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:
                pass
            try:
                httpd.server_close()
            except Exception:
                pass
        if thread is not None:
            thread.join(timeout=2)


def compute_milestones(
    toolchains: Mapping[str, Mapping[str, Any]],
    local_env: Mapping[str, Any],
    artifacts: Mapping[str, Mapping[str, Any]],
    server: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Compute M0-M3 readiness rollup and ordered actionable next commands."""
    def _has_tool(name: str) -> bool:
        return bool(toolchains.get(name, {}).get("available"))

    # M0: Foundation & contributor toolchains
    m0_missing: list[str] = []
    for tool in ("python", "node", "rustc", "cargo"):
        if not _has_tool(tool):
            m0_missing.append(f"toolchain:{tool}")
    if server.get("checked", True) and not server.get("ok"):
        m0_missing.append("server:coop-coep")
    m0_ready = len(m0_missing) == 0

    # M1: Browser runtime shell & Wasm64 smoke build
    m1_missing: list[str] = []
    if not _has_tool("python"):
        m1_missing.append("toolchain:python")
    if server.get("checked", True) and not server.get("ok"):
        m1_missing.append("server:coop-coep")

    rt_js = artifacts.get("runtimeJs", {})
    rt_wasm = artifacts.get("runtimeWasm", {})
    manifest = artifacts.get("dataManifest", {})
    bootset = artifacts.get("bootset", {})
    shader_packs = artifacts.get("shaderPacks", {})

    if not rt_js.get("nonEmpty"):
        m1_missing.append("web/dist/runtime.js")
    if not rt_wasm.get("nonEmpty") or not rt_wasm.get("validWasmMagic"):
        m1_missing.append("web/dist/runtime.wasm")
    if manifest.get("present") and not manifest.get("validJson"):
        m1_missing.append("local-data/data-manifest.json (invalid JSON)")
    if bootset.get("present") and not bootset.get("validJson"):
        m1_missing.append("local-data/bootset.json (invalid JSON)")
    m1_ready = len(m1_missing) == 0

    # M2: Real game static analysis & local asset readiness
    m2_missing: list[str] = []
    if not local_env.get("gameDirConfigured"):
        m2_missing.append("env:GTA6_GAME_DIR")
    elif not local_env.get("gameDirExists"):
        m2_missing.append("env:GTA6_GAME_DIR (missing on disk)")
    if not manifest.get("validJson"):
        m2_missing.append("local-data/data-manifest.json")
    if not bootset.get("validJson"):
        m2_missing.append("local-data/bootset.json")
    if not shader_packs.get("present"):
        m2_missing.append("shaders:*.wgsl.pack")
    m2_ready = len(m2_missing) == 0

    # M3: Wasm64 recompilation & OS/GDK build surface
    m3_missing: list[str] = []
    if not m1_ready:
        m3_missing.append("milestone:M1")
    if not m2_ready:
        m3_missing.append("milestone:M2")
    for tool in ("rustc", "cargo", "cmake", "emcc", "emcmake"):
        if not _has_tool(tool):
            m3_missing.append(f"toolchain:{tool}")
    m3_ready = len(m3_missing) == 0

    milestones: dict[str, dict[str, Any]] = {
        "M0": {
            "title": "Foundation & Contributor Toolchains",
            "ready": m0_ready,
            "missing": m0_missing,
        },
        "M1": {
            "title": "Browser Runtime Shell & Wasm64 Smoke Build",
            "ready": m1_ready,
            "missing": m1_missing,
        },
        "M2": {
            "title": "Local Game Analysis & Asset Readiness",
            "ready": m2_ready,
            "missing": m2_missing,
        },
        "M3": {
            "title": "Wasm64 Recompilation & OS/GDK Build Surface",
            "ready": m3_ready,
            "missing": m3_missing,
        },
    }

    next_commands: list[str] = []

    def _add_cmd(cmd: str) -> None:
        if cmd not in next_commands:
            next_commands.append(cmd)

    if not local_env.get("gameDirConfigured") or not local_env.get("gameDirExists"):
        _add_cmd('python tools/local_env.py --set-game "<path>"')

    if not (rt_js.get("nonEmpty") and rt_wasm.get("nonEmpty") and rt_wasm.get("validWasmMagic")):
        if not (_has_tool("emcc") and _has_tool("emcmake")):
            _add_cmd("emsdk install latest && emsdk activate latest")
        _add_cmd("pwsh runtime/build_wasm.ps1")

    if not manifest.get("validJson"):
        _add_cmd("cargo run -p playgta6-tools -- scan --from-env")
        _add_cmd("cargo run -p playgta6-tools -- manifest --out local-data/data-manifest.json")

    if not bootset.get("validJson"):
        _add_cmd("cargo run -p playgta6-tools -- bootset --out local-data/bootset.json")

    if not shader_packs.get("present"):
        _add_cmd("cargo run -p playgta6-shaders -- --help")

    if m1_ready:
        _add_cmd("python server/serve_local.py")

    return milestones, next_commands


def run_preflight(
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
    which_fn: Callable[[str], str | None] = shutil.which,
    version_fn: Callable[[str], str | None] | None = None,
    skip_server_check: bool = False,
) -> dict[str, Any]:
    """Run all preflight checks against `repo_root` and return a sanitized JSON-safe dict."""
    root = Path(repo_root).resolve()
    resolved_env = _resolve_env_paths(root, env=env)
    data_dir: Path = resolved_env["data_path"]

    toolchains = check_toolchains(which_fn=which_fn, version_fn=version_fn)
    local_env = check_local_env(root, env=env)
    artifacts = check_artifacts(root, data_dir=data_dir)
    if skip_server_check:
        server: dict[str, Any] = {
            "checked": False,
            "serverScriptPresent": (root / "server" / "serve_local.py").is_file(),
            "importable": False,
            "mode": None,
            "coopValid": False,
            "coepValid": False,
            "corpValid": False,
            "acceptRangesValid": False,
            "statusEndpointOk": False,
            "ok": False,
            "error": None,
        }
    else:
        server = check_server_headers(root, data_dir=data_dir)

    milestones, next_commands = compute_milestones(toolchains, local_env, artifacts, server)

    return {
        "schemaVersion": 1,
        "ok": bool(milestones["M1"]["ready"]),
        "toolchains": toolchains,
        "localEnv": local_env,
        "artifacts": artifacts,
        "server": server,
        "milestones": milestones,
        "nextCommands": next_commands,
    }


def _tag(ok: bool) -> str:
    return "[OK]  " if ok else "[MISS]"


def format_report_text(report: Mapping[str, Any]) -> str:
    """Format a sanitized preflight report as a concise terminal checklist."""
    lines: list[str] = ["=== playgta6web Local Preflight & Milestone Readiness ===", ""]

    lines.append("Toolchains:")
    toolchains = report["toolchains"]
    for name in ("python", "node", "rustc", "cargo", "cmake", "emcc", "emcmake", "gh"):
        info = toolchains.get(name, {})
        avail = bool(info.get("available"))
        ver = info.get("version") or ("available" if avail else "not found")
        lines.append(f"  {_tag(avail)} {name:<8} {ver}")

    lines.append("")
    lines.append("Local Environment (.env / env vars):")
    le = report["localEnv"]
    summary = le.get("gameSummary", {})
    game_detail = "configured & present" if le.get("gameDirExists") else (
        "configured but missing on disk" if le.get("gameDirConfigured") else "not configured"
    )
    if le.get("gameDirExists"):
        game_detail += (
            f" (MicrosoftGame.Config={summary.get('hasMicrosoftGameConfig')}, "
            f"rpf={summary.get('rpfCount', 0)}, exe={summary.get('exeCount', 0)}, "
            f"xvc/xvd={summary.get('xvcXvdCount', 0)})"
        )
    lines.append(f"  {_tag(bool(le.get('gameDirExists')))} GTA6_GAME_DIR  {game_detail}")
    lines.append(
        f"  {_tag(bool(le.get('dataDirConfigured')))} GTA6_DATA_DIR  "
        f"{'exists' if le.get('dataDirExists') else ('configured (not created yet)' if le.get('dataDirConfigured') else 'not configured')}"
    )
    lines.append(
        f"  {_tag(bool(le.get('workDirConfigured')))} GTA6_WORK_DIR  "
        f"{'exists' if le.get('workDirExists') else ('configured (not created yet)' if le.get('workDirConfigured') else 'not configured')}"
    )

    lines.append("")
    lines.append("Build & Runtime Artifacts:")
    art = report["artifacts"]
    rt_js = art["runtimeJs"]
    rt_wasm = art["runtimeWasm"]
    manifest = art["dataManifest"]
    bootset = art["bootset"]
    packs = art["shaderPacks"]

    js_note = f"{rt_js['sizeBytes']} bytes" if rt_js["present"] else "missing"
    lines.append(f"  {_tag(bool(rt_js['nonEmpty']))} web/dist/runtime.js       {js_note}")
    wasm_ok = bool(rt_wasm["nonEmpty"] and rt_wasm["validWasmMagic"])
    wasm_note = (
        f"{rt_wasm['sizeBytes']} bytes (valid wasm)"
        if wasm_ok
        else (
            f"{rt_wasm['sizeBytes']} bytes (invalid/empty wasm)"
            if rt_wasm["present"]
            else "missing"
        )
    )
    lines.append(f"  {_tag(wasm_ok)} web/dist/runtime.wasm     {wasm_note}")

    if manifest["validJson"]:
        manifest_note = f"valid JSON (files={manifest['fileCount']})"
    elif manifest["present"]:
        manifest_note = f"invalid ({manifest['error']})"
    else:
        manifest_note = "missing"
    lines.append(f"  {_tag(bool(manifest['validJson']))} data-manifest.json        {manifest_note}")

    if bootset["validJson"]:
        bootset_note = f"valid JSON (entries={bootset['entryCount']})"
    elif bootset["present"]:
        bootset_note = f"invalid ({bootset['error']})"
    else:
        bootset_note = "missing"
    lines.append(f"  {_tag(bool(bootset['validJson']))} bootset.json              {bootset_note}")

    lines.append(
        f"  {_tag(bool(packs['present']))} *.wgsl.pack bundles       "
        f"{packs['count']} found ({packs['nonEmptyCount']} non-empty)"
    )

    lines.append("")
    lines.append("Server COOP/COEP Self-Check:")
    srv = report["server"]
    if not srv.get("checked"):
        lines.append("  [SKIP] server check skipped")
    elif srv.get("ok"):
        lines.append(
            f"  {_tag(True)} server/serve_local.py     "
            f"COOP=same-origin, COEP=require-corp ({srv.get('mode')})"
        )
    else:
        lines.append(
            f"  {_tag(False)} server/serve_local.py     {srv.get('error') or 'check failed'}"
        )

    lines.append("")
    lines.append("Milestone Readiness:")
    for key in ("M0", "M1", "M2", "M3"):
        ms = report["milestones"][key]
        status_label = "READY" if ms["ready"] else "NOT READY"
        suffix = "" if ms["ready"] else f" (missing: {', '.join(ms['missing'])})"
        lines.append(f"  {_tag(bool(ms['ready']))} {key} - {ms['title']}: {status_label}{suffix}")

    next_cmds = report.get("nextCommands", [])
    if next_cmds:
        lines.append("")
        lines.append("Next Commands:")
        for cmd in next_cmds:
            lines.append(f"  $ {cmd}")

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check local toolchains, environment, artifacts, COOP/COEP server, and M0-M3 readiness."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the playgta6web repository root (defaults to parent of tools/).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable sanitized JSON report (never contains host paths or usernames).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if required M1 checks fail.",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip the local COOP/COEP server header self-check.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = run_preflight(
        repo_root=args.repo_root,
        skip_server_check=args.skip_server,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))

    if args.strict and not report["milestones"]["M1"]["ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
