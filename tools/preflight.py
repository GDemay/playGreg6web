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
import ctypes
import http.client
import importlib.util
import json
import math
import os
import platform
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import threading
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
WASM_MAGIC = b"\x00asm"
WGPK_MAGIC = b"WGPK"
WGPK_VERSION = 1
WGPK_HEADER_STRUCT = struct.Struct("<4sIII")
WGPK_HEADER_BYTES = WGPK_HEADER_STRUCT.size
WGPK_ENTRY_STRUCT = struct.Struct("<QB3sII")
WGPK_ENTRY_BYTES = WGPK_ENTRY_STRUCT.size
WGPK_STAGE_MAP: Mapping[int, str] = {
    1: "vertex",
    2: "fragment",
    3: "compute",
}
SPLASH_PACK_NAME = "splash.wgsl.pack"
DEFAULT_MIN_FREE_GB = 5.0
DEFAULT_MIN_FREE_RAM_GB = 4.0
BYTES_PER_GIB = 1024**3
REFLECT_SIDECAR_SCHEMA = "playgta6-stage-reflection"
REFLECT_STAGE_ALIASES: Mapping[str, str] = {
    "vertex": "vertex",
    "fragment": "fragment",
    "pixel": "fragment",
    "compute": "compute",
}

BROWSER_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("chrome", ("google-chrome", "google-chrome-stable", "chrome")),
    ("edge", ("microsoft-edge", "microsoft-edge-stable", "msedge")),
    ("chromium", ("chromium", "chromium-browser")),
)
_SEMVER_VERSION_RE = re.compile(r"\b(\d+\.\d+(?:\.\d+){1,2})\b")
_QUAD_VERSION_DIR_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")

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


def _nearest_existing_dir(path: Path, fallback: Path) -> Path:
    for candidate in (path, *path.parents):
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return fallback


def _probe_disk_target(
    path: Path,
    fallback: Path,
    *,
    min_free_gb: float,
    disk_usage_fn: Callable[[Path], Any],
) -> dict[str, Any]:
    target = _nearest_existing_dir(path, fallback)
    try:
        usage = disk_usage_fn(target)
        total_bytes = int(usage.total)
        used_bytes = int(usage.used)
        free_bytes = int(usage.free)
    except OSError:
        return {
            "checked": False,
            "totalBytes": None,
            "usedBytes": None,
            "freeBytes": None,
            "freeGb": None,
            "minFreeGb": round(float(min_free_gb), 2),
            "lowSpace": False,
            "ok": False,
            "error": "disk usage unavailable",
        }

    free_gb = free_bytes / BYTES_PER_GIB
    low_space = free_gb < min_free_gb
    return {
        "checked": True,
        "totalBytes": total_bytes,
        "usedBytes": used_bytes,
        "freeBytes": free_bytes,
        "freeGb": round(free_gb, 2),
        "minFreeGb": round(float(min_free_gb), 2),
        "lowSpace": low_space,
        "ok": not low_space,
        "error": None,
    }


def check_disk_space(
    repo_root: Path,
    work_dir: Path | None = None,
    *,
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    disk_usage_fn: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Check free disk space on the workspace and work drives (sanitized; never leaks paths)."""
    if not math.isfinite(min_free_gb) or min_free_gb < 0:
        raise ValueError("min_free_gb must be a finite non-negative number")

    usage_fn = shutil.disk_usage if disk_usage_fn is None else disk_usage_fn
    resolved_work = work_dir if work_dir is not None else (repo_root / "work")

    workspace_info = _probe_disk_target(
        repo_root,
        repo_root,
        min_free_gb=min_free_gb,
        disk_usage_fn=usage_fn,
    )
    work_info = _probe_disk_target(
        resolved_work,
        repo_root,
        min_free_gb=min_free_gb,
        disk_usage_fn=usage_fn,
    )

    checked_free = [
        info["freeBytes"]
        for info in (workspace_info, work_info)
        if info["checked"] and isinstance(info["freeBytes"], int)
    ]
    min_free_bytes = min(checked_free) if checked_free else None
    min_free_gb_val = round(min_free_bytes / BYTES_PER_GIB, 2) if min_free_bytes is not None else None
    low_space = bool(workspace_info["lowSpace"] or work_info["lowSpace"])

    return {
        "minFreeGb": round(float(min_free_gb), 2),
        "ok": bool(workspace_info["ok"] and work_info["ok"]),
        "warning": low_space,
        "lowSpace": low_space,
        "freeBytes": min_free_bytes,
        "freeGb": min_free_gb_val,
        "workspace": workspace_info,
        "work": work_info,
    }


def _probe_system_memory_bytes() -> tuple[int, int] | None:
    """Return `(total_bytes, available_bytes)` using stdlib-only OS probes."""
    if sys.platform == "win32":
        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        try:
            ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        except (AttributeError, OSError):
            return None
        if not ok or status.ullTotalPhys <= 0:
            return None
        return int(status.ullTotalPhys), int(status.ullAvailPhys)

    proc_meminfo = Path("/proc/meminfo")
    if proc_meminfo.is_file():
        try:
            values_kib: dict[str, int] = {}
            for line in proc_meminfo.read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, rest = line.split(":", 1)
                parts = rest.strip().split()
                if parts and parts[0].isdigit():
                    values_kib[key.strip()] = int(parts[0])
            total_kib = values_kib.get("MemTotal")
            if total_kib is None or total_kib <= 0:
                return None
            avail_kib = values_kib.get("MemAvailable")
            if avail_kib is None:
                avail_kib = (
                    values_kib.get("MemFree", 0)
                    + values_kib.get("Buffers", 0)
                    + values_kib.get("Cached", 0)
                )
            return total_kib * 1024, max(0, avail_kib) * 1024
        except (OSError, UnicodeDecodeError, ValueError):
            return None

    if sys.platform == "darwin":
        try:
            total_proc = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
            vm_proc = subprocess.run(
                ["vm_stat"],
                capture_output=True,
                text=True,
                stdin=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
            if total_proc.returncode != 0 or vm_proc.returncode != 0:
                return None
            total_bytes = int(total_proc.stdout.strip())
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages: dict[str, int] = {}
            for line in vm_proc.stdout.splitlines():
                m = re.match(r"^([^:]+):\s+(\d+)\.$", line.strip())
                if m:
                    pages[m.group(1).strip().lower()] = int(m.group(2))
            avail_pages = (
                pages.get("pages free", 0)
                + pages.get("pages inactive", 0)
                + pages.get("pages speculative", 0)
            )
            return total_bytes, max(0, avail_pages * int(page_size))
        except (OSError, subprocess.SubprocessError, ValueError):
            return None

    return None


def check_memory_headroom(
    *,
    min_free_ram_gb: float = DEFAULT_MIN_FREE_RAM_GB,
    memory_info_fn: Callable[[], tuple[int, int] | None] | None = None,
) -> dict[str, Any]:
    """Check available system RAM against a safe browser-boot headroom threshold."""
    if not math.isfinite(min_free_ram_gb) or min_free_ram_gb < 0:
        raise ValueError("min_free_ram_gb must be a finite non-negative number")

    probe_fn = _probe_system_memory_bytes if memory_info_fn is None else memory_info_fn
    threshold_rounded = round(float(min_free_ram_gb), 2)
    try:
        mem = probe_fn()
    except Exception:
        mem = None

    if (
        not isinstance(mem, tuple)
        or len(mem) != 2
        or not isinstance(mem[0], int)
        or isinstance(mem[0], bool)
        or not isinstance(mem[1], int)
        or isinstance(mem[1], bool)
        or mem[0] <= 0
        or mem[1] < 0
    ):
        return {
            "checked": False,
            "totalBytes": None,
            "availableBytes": None,
            "totalGb": None,
            "availableGb": None,
            "minFreeRamGb": threshold_rounded,
            "lowMemory": False,
            "ok": False,
            "warning": False,
            "error": "memory info unavailable",
        }

    total_bytes, available_bytes = mem
    total_gb = total_bytes / BYTES_PER_GIB
    available_gb = available_bytes / BYTES_PER_GIB
    low_memory = available_gb < min_free_ram_gb
    return {
        "checked": True,
        "totalBytes": total_bytes,
        "availableBytes": available_bytes,
        "totalGb": round(total_gb, 2),
        "availableGb": round(available_gb, 2),
        "minFreeRamGb": threshold_rounded,
        "lowMemory": low_memory,
        "ok": not low_memory,
        "warning": low_memory,
        "error": None,
    }


def _sanitize_browser_version(raw: str | None) -> str | None:
    """Extract a sanitized browser version string that never leaks host paths or usernames."""
    cleaned = sanitize_version(raw)
    if not cleaned:
        return None
    match = _SEMVER_VERSION_RE.search(cleaned)
    if match:
        return match.group(1)
    stripped = re.sub(r"\(\s*<path>\s*\)", "", cleaned)
    stripped = stripped.replace("<path>", "").replace("<user>", "")
    stripped = " ".join(stripped.split()).strip(" -:,()")
    return stripped or None


def _default_browser_candidate_paths(family: str) -> list[Path]:
    paths: list[Path] = []
    if sys.platform == "win32":
        roots = [
            os.environ.get("PROGRAMFILES", ""),
            os.environ.get("PROGRAMFILES(X86)", ""),
            os.environ.get("LOCALAPPDATA", ""),
        ]
        rel_map: Mapping[str, tuple[str, ...]] = {
            "chrome": (r"Google\Chrome\Application\chrome.exe",),
            "edge": (r"Microsoft\Edge\Application\msedge.exe",),
            "chromium": (r"Chromium\Application\chrome.exe",),
        }
        for root_str in roots:
            if not root_str:
                continue
            base = Path(root_str)
            for rel in rel_map.get(family, ()):
                paths.append(base / rel)
    elif sys.platform == "darwin":
        mac_map: Mapping[str, tuple[str, ...]] = {
            "chrome": ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",),
            "edge": ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",),
            "chromium": ("/Applications/Chromium.app/Contents/MacOS/Chromium",),
        }
        for raw_path in mac_map.get(family, ()):
            paths.append(Path(raw_path))
    return paths


def _probe_windows_browser_version(family: str, resolved_bin: str) -> str | None:
    """Read Chrome/Edge/Chromium version on Windows without spawning a GUI process."""
    try:
        parent = Path(resolved_bin).parent
        if parent.is_dir():
            versions: list[tuple[tuple[int, ...], str]] = []
            for entry in parent.iterdir():
                if entry.is_dir() and _QUAD_VERSION_DIR_RE.match(entry.name):
                    versions.append((tuple(int(part) for part in entry.name.split(".")), entry.name))
            if versions:
                versions.sort(key=lambda item: item[0], reverse=True)
                return _sanitize_browser_version(versions[0][1])
    except OSError:
        pass

    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return None

    reg_paths: Mapping[str, tuple[str, ...]] = {
        "chrome": (
            r"Software\Google\Chrome\BLBeacon",
            r"Software\WOW6432Node\Google\Chrome\BLBeacon",
        ),
        "edge": (
            r"Software\Microsoft\Edge\BLBeacon",
            r"Software\WOW6432Node\Microsoft\Edge\BLBeacon",
        ),
        "chromium": (r"Software\Chromium\BLBeacon",),
    }
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for subkey in reg_paths.get(family, ()):
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    val, _ = winreg.QueryValueEx(key, "version")
                    if isinstance(val, str):
                        sanitized = _sanitize_browser_version(val)
                        if sanitized:
                            return sanitized
            except OSError:
                continue
    return None


def _probe_browser_version(family: str, resolved_bin: str) -> str | None:
    if sys.platform == "win32":
        return _probe_windows_browser_version(family, resolved_bin)

    if sys.platform == "darwin":
        try:
            plist_path = Path(resolved_bin).parent.parent / "Info.plist"
            if plist_path.is_file():
                with plist_path.open("rb") as fh:
                    info = plistlib.load(fh)
                if isinstance(info, dict):
                    raw_ver = info.get("CFBundleShortVersionString") or info.get("CFBundleVersion")
                    if isinstance(raw_ver, str):
                        sanitized = _sanitize_browser_version(raw_ver)
                        if sanitized:
                            return sanitized
        except (OSError, plistlib.InvalidFileException, ValueError):
            pass

    return _sanitize_browser_version(_probe_binary_version(resolved_bin))


def check_browser(
    *,
    which_fn: Callable[[str], str | None] = shutil.which,
    version_fn: Callable[[str], str | None] | None = None,
    candidate_paths_fn: Callable[[str], Sequence[Path]] | None = None,
) -> dict[str, Any]:
    """Detect local Chrome/Edge/Chromium availability without ever leaking host paths."""
    use_default_paths = candidate_paths_fn is None and which_fn is shutil.which
    browsers: dict[str, dict[str, Any]] = {}
    primary_family: str | None = None
    primary_version: str | None = None

    for family, bin_names in BROWSER_SPECS:
        resolved: str | None = None
        for bin_name in bin_names:
            found = which_fn(bin_name)
            if found:
                resolved = found
                break

        if resolved is None:
            candidates: Sequence[Path] = ()
            if candidate_paths_fn is not None:
                candidates = candidate_paths_fn(family)
            elif use_default_paths:
                candidates = _default_browser_candidate_paths(family)
            for candidate in candidates:
                try:
                    if candidate.is_file():
                        resolved = str(candidate)
                        break
                except OSError:
                    continue

        if resolved is None:
            browsers[family] = {"available": False, "version": None}
            continue

        if version_fn is not None:
            raw_ver = version_fn(resolved)
            version = _sanitize_browser_version(raw_ver)
        else:
            version = _probe_browser_version(family, resolved)

        browsers[family] = {"available": True, "version": version}
        if primary_family is None:
            primary_family = family
            primary_version = version

    available = primary_family is not None
    warnings: list[str] = []
    if not available:
        warnings.append("no Chrome/Edge/Chromium browser detected for WebGPU/Wasm64 boot")

    return {
        "available": available,
        "family": primary_family,
        "version": primary_version,
        "browsers": browsers,
        "warning": not available,
        "warnings": warnings,
    }


def inspect_wgpk_header(path: Path) -> dict[str, Any]:
    """Validate a WGPK v1 binary pack header and entry table (strictly sanitized)."""
    stage_counts: dict[str, int] = {"vertex": 0, "fragment": 0, "compute": 0}
    result: dict[str, Any] = {
        "present": False,
        "sizeBytes": 0,
        "headerValid": False,
        "version": None,
        "entryCount": None,
        "flags": None,
        "stageCounts": stage_counts,
        "errors": [],
    }
    if not path.is_file():
        return result

    result["present"] = True
    errors: list[str] = []

    def _add_error(msg: str) -> None:
        if msg not in errors:
            errors.append(msg)

    try:
        size_bytes = path.stat().st_size
    except OSError:
        result["errors"] = ["unreadable file"]
        return result

    result["sizeBytes"] = size_bytes
    if size_bytes < WGPK_HEADER_BYTES:
        result["errors"] = ["pack truncated (shorter than 16-byte WGPK header)"]
        return result

    try:
        with path.open("rb") as fh:
            header_bytes = fh.read(WGPK_HEADER_BYTES)
            if len(header_bytes) < WGPK_HEADER_BYTES:
                result["errors"] = ["pack truncated (shorter than 16-byte WGPK header)"]
                return result
            magic, version, entry_count, flags = WGPK_HEADER_STRUCT.unpack(header_bytes)
            if magic != WGPK_MAGIC:
                result["errors"] = ["invalid WGPK magic"]
                return result

            result["version"] = int(version)
            result["entryCount"] = int(entry_count)
            result["flags"] = int(flags)

            if version != WGPK_VERSION:
                result["errors"] = [f"unsupported WGPK version {version}"]
                return result

            table_bytes_len = entry_count * WGPK_ENTRY_BYTES
            min_required = WGPK_HEADER_BYTES + table_bytes_len
            if size_bytes < min_required:
                result["errors"] = ["pack truncated (entry table exceeds file size)"]
                return result

            if entry_count > 0:
                table_bytes = fh.read(table_bytes_len)
                if len(table_bytes) < table_bytes_len:
                    result["errors"] = ["pack truncated (entry table exceeds file size)"]
                    return result

                for _hash_u64, stage_u8, _reserved, offset_u32, length_u32 in WGPK_ENTRY_STRUCT.iter_unpack(
                    table_bytes
                ):
                    stage_name = WGPK_STAGE_MAP.get(stage_u8)
                    if stage_name is None:
                        _add_error("invalid WGPK entry stage")
                    else:
                        stage_counts[stage_name] += 1
                    end_offset = offset_u32 + length_u32
                    if end_offset > size_bytes or (length_u32 > 0 and offset_u32 < min_required):
                        _add_error("WGPK entry payload out of bounds")
    except OSError:
        result["errors"] = ["unreadable file"]
        return result

    result["errors"] = errors
    result["headerValid"] = len(errors) == 0
    return result


def check_splash_wgpk_sync(
    repo_root: Path,
    *,
    data_dir: Path | None = None,
    work_dir: Path | None = None,
    shaders_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Validate `splash.wgsl.pack` WGPK v1 header and `.reflect.json` sidecar sync."""
    resolved_data = data_dir if data_dir is not None else (repo_root / "local-data")
    resolved_work = work_dir if work_dir is not None else (repo_root / "work")
    web_root = repo_root / "web"

    shaders_dir_provided = shaders_dir is not None
    shaders_dir_exists = False
    pack_path: Path | None = None
    warnings: list[str] = []

    def _add_warning(msg: str) -> None:
        if msg not in warnings:
            warnings.append(msg)

    if shaders_dir is not None:
        raw_dir = Path(shaders_dir).expanduser()
        target_dir = raw_dir if (raw_dir.is_absolute() or raw_dir.exists()) else (repo_root / raw_dir)
        shaders_dir_exists = target_dir.is_dir()
        if not shaders_dir_exists:
            _add_warning("shaders directory not found")
        else:
            candidate = target_dir / SPLASH_PACK_NAME
            if candidate.is_file():
                pack_path = candidate
    else:
        search_dirs = (
            resolved_data / "shaders",
            resolved_data,
            web_root / "shaders",
            web_root / "dist",
            resolved_work,
        )
        for directory in search_dirs:
            if directory.is_dir():
                shaders_dir_exists = True
                candidate = directory / SPLASH_PACK_NAME
                if candidate.is_file():
                    pack_path = candidate
                    break

    if pack_path is None:
        _add_warning(f"{SPLASH_PACK_NAME} not found")
        return {
            "shadersDirProvided": shaders_dir_provided,
            "shadersDirExists": shaders_dir_exists,
            "present": False,
            "sizeBytes": 0,
            "headerValid": False,
            "version": None,
            "entryCount": None,
            "flags": None,
            "stageCounts": {"vertex": 0, "fragment": 0, "compute": 0},
            "sidecarPresent": False,
            "sidecarValid": False,
            "sidecarStageCount": None,
            "sidecarStale": False,
            "mtimeDeltaSeconds": None,
            "countMatch": None,
            "inSync": False,
            "ok": False,
            "warning": True,
            "warnings": warnings,
            "errors": [],
        }

    pack_info = inspect_wgpk_header(pack_path)
    errors = list(pack_info["errors"])
    header_valid = bool(pack_info["headerValid"])
    entry_count = pack_info["entryCount"]

    if not header_valid:
        _add_warning(f"{SPLASH_PACK_NAME} header validation failed")
    elif entry_count == 0:
        _add_warning(f"{SPLASH_PACK_NAME} has 0 shader entries")

    sidecar_path = _find_adjacent_sidecar(pack_path)
    sidecar_present = sidecar_path is not None
    sidecar_valid = False
    sidecar_stage_count: int | None = None
    sidecar_stale = False
    mtime_delta_seconds: float | None = None
    count_match: bool | None = None

    if not sidecar_present:
        _add_warning(f"{SPLASH_PACK_NAME}.reflect.json sidecar is missing")
    else:
        assert sidecar_path is not None
        sidecar_info = _inspect_reflect_sidecar(sidecar_path)
        sidecar_valid = bool(sidecar_info["valid"])
        sidecar_stage_count = int(sidecar_info["stageCount"])
        if not sidecar_valid:
            _add_warning(f"{SPLASH_PACK_NAME}.reflect.json sidecar is invalid")
            for err in sidecar_info["errors"]:
                if err not in errors:
                    errors.append(err)

        try:
            pack_mtime = pack_path.stat().st_mtime
            sidecar_mtime = sidecar_path.stat().st_mtime
            delta = sidecar_mtime - pack_mtime
            mtime_delta_seconds = round(delta, 3)
            if (pack_mtime - sidecar_mtime) > 1e-3:
                sidecar_stale = True
                _add_warning(f"{SPLASH_PACK_NAME}.reflect.json sidecar is stale (older than {SPLASH_PACK_NAME})")
        except OSError:
            pass

        if header_valid and sidecar_valid and isinstance(entry_count, int):
            count_match = entry_count == sidecar_stage_count
            if not count_match:
                _add_warning(
                    f"sidecar stage count ({sidecar_stage_count}) does not match "
                    f"{SPLASH_PACK_NAME} entry count ({entry_count})"
                )

    in_sync = bool(
        header_valid
        and sidecar_present
        and sidecar_valid
        and not sidecar_stale
        and count_match is True
    )
    ok = bool(header_valid and isinstance(entry_count, int) and entry_count > 0 and in_sync)

    return {
        "shadersDirProvided": shaders_dir_provided,
        "shadersDirExists": shaders_dir_exists,
        "present": True,
        "sizeBytes": int(pack_info["sizeBytes"]),
        "headerValid": header_valid,
        "version": pack_info["version"],
        "entryCount": entry_count,
        "flags": pack_info["flags"],
        "stageCounts": pack_info["stageCounts"],
        "sidecarPresent": sidecar_present,
        "sidecarValid": sidecar_valid,
        "sidecarStageCount": sidecar_stage_count,
        "sidecarStale": sidecar_stale,
        "mtimeDeltaSeconds": mtime_delta_seconds,
        "countMatch": count_match,
        "inSync": in_sync,
        "ok": ok,
        "warning": len(warnings) > 0,
        "warnings": warnings,
        "errors": errors,
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


def _shader_candidate_dirs(web_root: Path, data_dir: Path) -> list[Path]:
    candidate_dirs = (
        data_dir,
        data_dir / "shaders",
        web_root / "dist",
        web_root / "shaders",
    )
    seen: set[Path] = set()
    existing: list[Path] = []
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
        existing.append(directory)
    return existing


def _inline_sampler_ok(source: Mapping[str, Any]) -> bool | None:
    """Return None if not an inline_sampler source, else whether sdwords has 4 u32 ints."""
    if "inline_sampler" not in source:
        return None
    sampler = source["inline_sampler"]
    words = sampler.get("sdwords") if isinstance(sampler, dict) else None
    return (
        isinstance(words, list)
        and len(words) == 4
        and all(
            isinstance(w, int) and not isinstance(w, bool) and 0 <= w <= 0xFFFFFFFF
            for w in words
        )
    )


def _find_adjacent_sidecar(pack_path: Path) -> Path | None:
    primary = pack_path.with_name(f"{pack_path.name}.reflect.json")
    if primary.is_file():
        return primary
    alt = pack_path.with_suffix(".reflect.json")
    if alt.is_file():
        return alt
    return None


def _inspect_reflect_sidecar(path: Path) -> dict[str, Any]:
    """Validate one `.reflect.json` sidecar and return strictly sanitized counts/errors."""
    stage_counts: dict[str, int] = {"vertex": 0, "fragment": 0, "compute": 0}
    errors: list[str] = []
    stage_count = 0
    binding_count = 0
    inline_sampler_count = 0
    inline_sampler_stage_count = 0
    invalid_inline_sampler_count = 0

    def _add_error(msg: str) -> None:
        if msg not in errors:
            errors.append(msg)

    try:
        size_bytes = path.stat().st_size
    except OSError:
        return {
            "valid": False,
            "stageCount": 0,
            "stageCounts": stage_counts,
            "bindingCount": 0,
            "inlineSamplerCount": 0,
            "inlineSamplerStageCount": 0,
            "invalidInlineSamplerCount": 0,
            "errors": ["unreadable file"],
        }

    if size_bytes == 0:
        return {
            "valid": False,
            "stageCount": 0,
            "stageCounts": stage_counts,
            "bindingCount": 0,
            "inlineSamplerCount": 0,
            "inlineSamplerStageCount": 0,
            "invalidInlineSamplerCount": 0,
            "errors": ["empty file"],
        }

    try:
        parsed = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {
            "valid": False,
            "stageCount": 0,
            "stageCounts": stage_counts,
            "bindingCount": 0,
            "inlineSamplerCount": 0,
            "inlineSamplerStageCount": 0,
            "invalidInlineSamplerCount": 0,
            "errors": ["invalid JSON"],
        }
    except (OSError, UnicodeDecodeError):
        return {
            "valid": False,
            "stageCount": 0,
            "stageCounts": stage_counts,
            "bindingCount": 0,
            "inlineSamplerCount": 0,
            "inlineSamplerStageCount": 0,
            "invalidInlineSamplerCount": 0,
            "errors": ["unreadable file"],
        }

    if not isinstance(parsed, dict):
        return {
            "valid": False,
            "stageCount": 0,
            "stageCounts": stage_counts,
            "bindingCount": 0,
            "inlineSamplerCount": 0,
            "inlineSamplerStageCount": 0,
            "invalidInlineSamplerCount": 0,
            "errors": ["expected JSON object"],
        }

    if "schema" in parsed and parsed["schema"] != REFLECT_SIDECAR_SCHEMA:
        _add_error("invalid schema")

    if "version" in parsed:
        ver = parsed["version"]
        if type(ver) is not int or ver < 1:
            _add_error("invalid version")

    stages = parsed.get("stages")
    if not isinstance(stages, dict):
        _add_error("missing or invalid stages object")
        return {
            "valid": False,
            "stageCount": 0,
            "stageCounts": stage_counts,
            "bindingCount": 0,
            "inlineSamplerCount": 0,
            "inlineSamplerStageCount": 0,
            "invalidInlineSamplerCount": 0,
            "errors": errors,
        }

    for stage_key, stage_entry in stages.items():
        if not isinstance(stage_key, str) or not stage_key.strip() or not isinstance(stage_entry, dict):
            _add_error("invalid stage entry")
            continue

        raw_stage = stage_entry.get("stage")
        normalized_stage = REFLECT_STAGE_ALIASES.get(raw_stage) if isinstance(raw_stage, str) else None
        if normalized_stage is None:
            _add_error("invalid stage kind")
            continue

        bindings = stage_entry.get("bindings")
        if not isinstance(bindings, list):
            _add_error("invalid bindings list")
            continue

        stage_count += 1
        stage_counts[normalized_stage] += 1
        stage_has_inline_sampler = False

        for binding in bindings:
            if not isinstance(binding, dict):
                _add_error("invalid binding entry")
                continue
            binding_count += 1
            source = binding.get("source")
            if isinstance(source, dict):
                sampler_state = _inline_sampler_ok(source)
                if sampler_state is True:
                    inline_sampler_count += 1
                    stage_has_inline_sampler = True
                elif sampler_state is False:
                    invalid_inline_sampler_count += 1
                    _add_error("malformed inline_sampler")

        if stage_has_inline_sampler:
            inline_sampler_stage_count += 1

    return {
        "valid": len(errors) == 0,
        "stageCount": stage_count,
        "stageCounts": stage_counts,
        "bindingCount": binding_count,
        "inlineSamplerCount": inline_sampler_count,
        "inlineSamplerStageCount": inline_sampler_stage_count,
        "invalidInlineSamplerCount": invalid_inline_sampler_count,
        "errors": errors,
    }


def _inspect_shader_packs_and_sidecars(
    web_root: Path,
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_dirs = _shader_candidate_dirs(web_root, data_dir)
    pack_paths: list[Path] = []
    sidecar_paths: list[Path] = []
    seen_sidecars: set[Path] = set()
    non_empty_count = 0

    for directory in candidate_dirs:
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for entry in entries:
            try:
                if not entry.is_file():
                    continue
            except OSError:
                continue
            lower_name = entry.name.lower()
            if lower_name.endswith(".wgsl.pack"):
                pack_paths.append(entry)
                try:
                    if entry.stat().st_size > 0:
                        non_empty_count += 1
                except OSError:
                    pass
            elif lower_name.endswith(".reflect.json"):
                try:
                    resolved = entry.resolve()
                except OSError:
                    resolved = entry
                if resolved not in seen_sidecars:
                    seen_sidecars.add(resolved)
                    sidecar_paths.append(entry)

    paired_pack_count = 0
    for pack_path in pack_paths:
        adjacent = _find_adjacent_sidecar(pack_path)
        if adjacent is not None:
            paired_pack_count += 1
            try:
                resolved = adjacent.resolve()
            except OSError:
                resolved = adjacent
            if resolved not in seen_sidecars:
                seen_sidecars.add(resolved)
                sidecar_paths.append(adjacent)

    pack_count = len(pack_paths)
    unpaired_pack_count = pack_count - paired_pack_count

    valid_count = 0
    invalid_count = 0
    stage_count = 0
    stage_counts: dict[str, int] = {"vertex": 0, "fragment": 0, "compute": 0}
    binding_count = 0
    inline_sampler_count = 0
    inline_sampler_stage_count = 0
    invalid_inline_sampler_count = 0
    errors: list[str] = []

    for sidecar_path in sidecar_paths:
        info = _inspect_reflect_sidecar(sidecar_path)
        if info["valid"]:
            valid_count += 1
        else:
            invalid_count += 1
        stage_count += info["stageCount"]
        for stage_kind in ("vertex", "fragment", "compute"):
            stage_counts[stage_kind] += info["stageCounts"][stage_kind]
        binding_count += info["bindingCount"]
        inline_sampler_count += info["inlineSamplerCount"]
        inline_sampler_stage_count += info["inlineSamplerStageCount"]
        invalid_inline_sampler_count += info["invalidInlineSamplerCount"]
        for err in info["errors"]:
            if err not in errors:
                errors.append(err)

    shader_packs: dict[str, Any] = {
        "present": pack_count > 0,
        "count": pack_count,
        "nonEmptyCount": non_empty_count,
        "pairedSidecarCount": paired_pack_count,
    }
    shader_sidecars: dict[str, Any] = {
        "present": len(sidecar_paths) > 0,
        "count": len(sidecar_paths),
        "validCount": valid_count,
        "invalidCount": invalid_count,
        "pairedPackCount": paired_pack_count,
        "unpairedPackCount": unpaired_pack_count,
        "ok": invalid_count == 0,
        "stageCount": stage_count,
        "stageCounts": stage_counts,
        "bindingCount": binding_count,
        "inlineSamplerCount": inline_sampler_count,
        "inlineSamplerStageCount": inline_sampler_stage_count,
        "invalidInlineSamplerCount": invalid_inline_sampler_count,
        "errors": errors,
    }
    return shader_packs, shader_sidecars


def _count_shader_packs(web_root: Path, data_dir: Path) -> dict[str, Any]:
    shader_packs, _ = _inspect_shader_packs_and_sidecars(web_root, data_dir)
    return shader_packs


def check_artifacts(repo_root: Path, data_dir: Path | None = None) -> dict[str, Any]:
    """Check presence, non-empty status, and validity of build/runtime/asset artifacts."""
    web_root = repo_root / "web"
    resolved_data_dir = data_dir if data_dir is not None else (repo_root / "local-data")
    dist_dir = web_root / "dist"
    shader_packs, shader_sidecars = _inspect_shader_packs_and_sidecars(web_root, resolved_data_dir)

    return {
        "runtimeJs": _check_binary_artifact(dist_dir / "runtime.js", check_wasm_magic=False),
        "runtimeWasm": _check_binary_artifact(dist_dir / "runtime.wasm", check_wasm_magic=True),
        "dataManifest": _check_data_manifest(resolved_data_dir / "data-manifest.json"),
        "bootset": _check_bootset(resolved_data_dir / "bootset.json"),
        "shaderPacks": shader_packs,
        "shaderSidecars": shader_sidecars,
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
    shader_sidecars = artifacts.get("shaderSidecars", {})

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
    if shader_sidecars.get("present") and not shader_sidecars.get("ok"):
        m2_missing.append("shaders:*.reflect.json (invalid)")
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
    min_free_gb: float = DEFAULT_MIN_FREE_GB,
    disk_usage_fn: Callable[[Path], Any] | None = None,
    shaders_dir: Path | str | None = None,
    min_free_ram_gb: float = DEFAULT_MIN_FREE_RAM_GB,
    memory_info_fn: Callable[[], tuple[int, int] | None] | None = None,
    browser_which_fn: Callable[[str], str | None] | None = None,
    browser_version_fn: Callable[[str], str | None] | None = None,
    browser_candidate_paths_fn: Callable[[str], Sequence[Path]] | None = None,
) -> dict[str, Any]:
    """Run all preflight checks against `repo_root` and return a sanitized JSON-safe dict."""
    root = Path(repo_root).resolve()
    resolved_env = _resolve_env_paths(root, env=env)
    data_dir: Path = resolved_env["data_path"]
    work_dir: Path = resolved_env["work_path"]

    toolchains = check_toolchains(which_fn=which_fn, version_fn=version_fn)
    local_env = check_local_env(root, env=env)
    disk_space = check_disk_space(
        root,
        work_dir=work_dir,
        min_free_gb=min_free_gb,
        disk_usage_fn=disk_usage_fn,
    )
    memory = check_memory_headroom(
        min_free_ram_gb=min_free_ram_gb,
        memory_info_fn=memory_info_fn,
    )
    eff_browser_which = which_fn if browser_which_fn is None else browser_which_fn
    eff_browser_version = version_fn if browser_version_fn is None else browser_version_fn
    browser = check_browser(
        which_fn=eff_browser_which,
        version_fn=eff_browser_version,
        candidate_paths_fn=browser_candidate_paths_fn,
    )
    artifacts = check_artifacts(root, data_dir=data_dir)
    splash_pack = check_splash_wgpk_sync(
        root,
        data_dir=data_dir,
        work_dir=work_dir,
        shaders_dir=shaders_dir,
    )

    fast_boot_warnings: list[str] = []
    for msg in splash_pack["warnings"]:
        if msg not in fast_boot_warnings:
            fast_boot_warnings.append(msg)
    for msg in browser["warnings"]:
        if msg not in fast_boot_warnings:
            fast_boot_warnings.append(msg)
    if disk_space["lowSpace"]:
        fast_boot_warnings.append(
            f"free disk space ({disk_space['freeGb']:.2f} GB) is below "
            f"{disk_space['minFreeGb']:.2f} GB browser-boot threshold"
        )
    if memory["lowMemory"]:
        fast_boot_warnings.append(
            f"available RAM ({memory['availableGb']:.2f} GB) is below "
            f"{memory['minFreeRamGb']:.2f} GB browser-boot threshold"
        )

    fast_boot: dict[str, Any] = {
        "ready": bool(
            splash_pack["ok"]
            and browser["available"]
            and disk_space["ok"]
            and memory["ok"]
        ),
        "splashPackOk": bool(splash_pack["ok"]),
        "sidecarInSync": bool(splash_pack["inSync"]),
        "browserAvailable": bool(browser["available"]),
        "browserFamily": browser["family"],
        "diskHeadroomOk": bool(disk_space["ok"]),
        "ramHeadroomOk": bool(memory["ok"]),
        "warnings": fast_boot_warnings,
    }

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
        "diskSpace": disk_space,
        "memory": memory,
        "browser": browser,
        "artifacts": artifacts,
        "splashPack": splash_pack,
        "fastBoot": fast_boot,
        "server": server,
        "milestones": milestones,
        "nextCommands": next_commands,
    }


def _tag(ok: bool) -> str:
    return "[OK]  " if ok else "[MISS]"


def _warn_tag(ok: bool) -> str:
    return "[OK]  " if ok else "[WARN]"


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

    disk = report.get("diskSpace")
    if isinstance(disk, Mapping):
        threshold = disk.get("minFreeGb", DEFAULT_MIN_FREE_GB)
        lines.append("")
        lines.append(f"Disk Space (warning threshold: {threshold:.2f} GB):")
        for label, key in (("workspace drive", "workspace"), ("work drive", "work")):
            target = disk.get(key, {})
            if not target.get("checked"):
                lines.append(
                    f"  [WARN] {label:<16} {target.get('error') or 'disk usage unavailable'}"
                )
            elif target.get("lowSpace"):
                lines.append(
                    f"  [WARN] {label:<16} {target.get('freeGb', 0.0):.2f} GB free "
                    f"(below {threshold:.2f} GB warning threshold)"
                )
            else:
                lines.append(
                    f"  {_warn_tag(True)} {label:<16} {target.get('freeGb', 0.0):.2f} GB free "
                    f"(>= {threshold:.2f} GB)"
                )

    memory = report.get("memory")
    if isinstance(memory, Mapping):
        ram_threshold = memory.get("minFreeRamGb", DEFAULT_MIN_FREE_RAM_GB)
        lines.append("")
        lines.append(f"Memory Headroom (warning threshold: {ram_threshold:.2f} GB):")
        if not memory.get("checked"):
            lines.append(
                f"  [WARN] {'system RAM':<16} {memory.get('error') or 'memory info unavailable'}"
            )
        elif memory.get("lowMemory"):
            lines.append(
                f"  [WARN] {'system RAM':<16} {memory.get('availableGb', 0.0):.2f} GB available / "
                f"{memory.get('totalGb', 0.0):.2f} GB total (below {ram_threshold:.2f} GB warning threshold)"
            )
        else:
            lines.append(
                f"  {_warn_tag(True)} {'system RAM':<16} {memory.get('availableGb', 0.0):.2f} GB available / "
                f"{memory.get('totalGb', 0.0):.2f} GB total (>= {ram_threshold:.2f} GB)"
            )

    browser = report.get("browser")
    if isinstance(browser, Mapping):
        lines.append("")
        lines.append("Browser Availability (WebGPU / Wasm64):")
        if browser.get("available"):
            family = browser.get("family") or "chromium"
            ver = browser.get("version") or "detected"
            lines.append(f"  {_warn_tag(True)} {family:<16} {ver}")
        else:
            lines.append(
                f"  [WARN] {'browser':<16} no Chrome/Edge/Chromium browser detected"
            )

    lines.append("")
    lines.append("Build & Runtime Artifacts:")
    art = report["artifacts"]
    rt_js = art["runtimeJs"]
    rt_wasm = art["runtimeWasm"]
    manifest = art["dataManifest"]
    bootset = art["bootset"]
    packs = art["shaderPacks"]
    sidecars = art.get("shaderSidecars")

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

    if isinstance(sidecars, Mapping):
        if not sidecars.get("present"):
            lines.append(
                "  [SKIP] *.reflect.json sidecars   none staged (optional until shaders are packed)"
            )
        elif sidecars.get("ok"):
            sc = sidecars.get("stageCounts", {})
            lines.append(
                f"  {_warn_tag(True)} *.reflect.json sidecars   "
                f"{sidecars['count']} found (stages={sidecars['stageCount']}: "
                f"vs={sc.get('vertex', 0)}, fs={sc.get('fragment', 0)}, cs={sc.get('compute', 0)}; "
                f"inline_samplers={sidecars['inlineSamplerCount']} in "
                f"{sidecars['inlineSamplerStageCount']} stages)"
            )
        else:
            err_summary = ", ".join(sidecars.get("errors", [])) or "invalid sidecar"
            lines.append(
                f"  {_warn_tag(False)} *.reflect.json sidecars   "
                f"{sidecars['count']} found ({sidecars['invalidCount']} invalid, "
                f"invalid_inline_samplers={sidecars['invalidInlineSamplerCount']}: {err_summary})"
            )

    splash = report.get("splashPack")
    if isinstance(splash, Mapping):
        if splash.get("ok"):
            sc = splash.get("stageCounts", {})
            lines.append(
                f"  {_warn_tag(True)} {SPLASH_PACK_NAME:<25} "
                f"WGPK v{splash.get('version')} (entries={splash.get('entryCount')}: "
                f"vs={sc.get('vertex', 0)}, fs={sc.get('fragment', 0)}, cs={sc.get('compute', 0)}; "
                f"sidecar in sync)"
            )
        else:
            warn_summary = "; ".join(splash.get("warnings") or splash.get("errors") or ["not ready"])
            lines.append(f"  {_warn_tag(False)} {SPLASH_PACK_NAME:<25} {warn_summary}")

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
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=DEFAULT_MIN_FREE_GB,
        help=(
            f"Minimum free disk space in GB on the workspace/work drive before emitting "
            f"a low-space warning (default: {DEFAULT_MIN_FREE_GB:g})."
        ),
    )
    parser.add_argument(
        "--min-free-ram-gb",
        type=float,
        default=DEFAULT_MIN_FREE_RAM_GB,
        help=(
            f"Minimum available system RAM in GB before emitting a low-memory "
            f"browser-boot warning (default: {DEFAULT_MIN_FREE_RAM_GB:g})."
        ),
    )
    parser.add_argument(
        "--shaders-dir",
        type=Path,
        default=None,
        help=(
            f"Optional shaders directory to inspect for {SPLASH_PACK_NAME} WGPK v1 "
            f"header sanity and .reflect.json sidecar sync."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.min_free_gb) or args.min_free_gb < 0:
        parser.error("--min-free-gb must be a finite non-negative number")
    if not math.isfinite(args.min_free_ram_gb) or args.min_free_ram_gb < 0:
        parser.error("--min-free-ram-gb must be a finite non-negative number")

    report = run_preflight(
        repo_root=args.repo_root,
        skip_server_check=args.skip_server,
        min_free_gb=args.min_free_gb,
        shaders_dir=args.shaders_dir,
        min_free_ram_gb=args.min_free_ram_gb,
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
