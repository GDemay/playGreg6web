#!/usr/bin/env python3
"""End-to-end browser real-boot readiness verifier CLI (Track 5 / W48).

Validates the local chain required to boot the recompiled game in Chrome/Edge:

1. `game_assets` — MicrosoftGame.Config + ShellVisuals splash + DLL/RPF inventory
2. `import_coverage` — 587/587 (100%) PE import shim coverage
3. `recomp_artifacts` — program/build manifest, `.pgim`, shards, lift coverage %
4. `shader_pack` — valid WGPK `.wgsl.pack` + `reflection-manifest.json`
5. `wasm64_runtime` — `runtime.wasm` memory64/shared/host ABI + smoke vs linked guest
6. `local_server` — ephemeral COOP/COEP server + Range 206 + boot endpoints

Security & privacy invariants:
- Machine-readable `--json` and text output NEVER include host absolute paths
  (Windows or POSIX), usernames, or raw binary bytes.
- Strictly uses Python 3.12 standard library modules only.

Usage:
    python tools/verify_browser_boot.py [--repo-root <dir>] [--recomp-dir <dir>]
        [--skip-server] [--boot-report <path>] [--min-fps <fps>]
        [--require-non-black] [--json]
"""

from __future__ import annotations

import argparse
import http.client
import importlib.util
import json
import math
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
SMOKE_SHELL_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB
DEFAULT_RECOMP_DIR_REL = Path("work") / "recomp_build"
DEFAULT_SHADERS_DIR_REL = Path("local-data") / "shaders"
RUNTIME_WASM_REL = Path("web") / "dist" / "runtime.wasm"

GATE_NAMES: tuple[str, ...] = (
    "game_assets",
    "import_coverage",
    "recomp_artifacts",
    "shader_pack",
    "wasm64_runtime",
    "local_server",
)

GUEST_BOOT_EXPORT_MARKERS: frozenset[str] = frozenset(
    {
        "pg6_guest_boot_run",
        "pg6_guest_boot_reset",
        "pg6_guest_boot_report_json",
        "pg6_guest_boot_summary",
        "pg6_guest_boot_prepare_thread",
        "pg6_program_init",
        "pg6_program_boot",
        "__playgta6_register_lifted_functions",
        "__playgta6_register_lifted_functions_main",
    }
)

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_SLASH_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_DRIVE_SLASH_RE = re.compile(r"[A-Za-z]:/(?:[^\s\"'<>|,;()\[\]]+)")


# ---------------------------------------------------------------------------
# Sanitization
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
    """Strip host absolute paths (both `\\` and `/`) and active usernames."""
    if raw is None:
        return ""
    text = str(raw).replace("\\", "/")
    sanitized = _WINDOWS_PATH_RE.sub("<path>", text)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _UNC_SLASH_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    sanitized = _DRIVE_SLASH_RE.sub("<path>", sanitized)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        sanitized = pattern.sub("<user>", sanitized)
    return sanitized


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


def repo_relative_label(path: Path, repo_root: Path) -> str:
    """Return a repo-relative POSIX label, never an absolute host path."""
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
        return sanitize_text(rel.as_posix())
    except (OSError, ValueError):
        return basename_label(path)


def sanitize_value(value: Any) -> Any:
    """Recursively sanitize strings; replace raw bytes with size placeholders."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {sanitize_text(str(k)): sanitize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_value(v) for v in value]
    if isinstance(value, Path):
        return basename_label(value)
    return value


# ---------------------------------------------------------------------------
# Gate result model
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    """Structured readiness result for one browser-boot gate."""

    name: str
    status: str  # pass | warn | fail | skip
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    next_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "summary": sanitize_text(self.summary),
            "details": sanitize_value(self.details),
            "next_action": sanitize_text(self.next_action) if self.next_action else None,
        }


# ---------------------------------------------------------------------------
# Dynamic sibling-tool loaders
# ---------------------------------------------------------------------------


def _load_tools_module(module_filename: str, module_name: str) -> Any:
    """Load a sibling tools/*.py module, reusing an already-imported copy when present.

    Registers the module in `sys.modules` before `exec_module` so `@dataclass`
    (and similar) can resolve `cls.__module__`.
    """
    short_name = module_filename[:-3] if module_filename.endswith(".py") else module_filename
    for key in (module_name, short_name):
        existing = sys.modules.get(key)
        if existing is not None:
            return existing

    tools_dir = Path(__file__).resolve().parent
    path = tools_dir / module_filename
    if not path.is_file():
        raise FileNotFoundError(f"missing tool module: tools/{module_filename}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load tools/{module_filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_server_module(repo_root: Path) -> Any:
    """Load `server/serve_local.py` from the given repo root (no cross-repo cache)."""
    path = (repo_root / "server" / "serve_local.py").resolve()
    if not path.is_file():
        raise FileNotFoundError("missing server/serve_local.py")
    # Unique module name per path so temp-repo tests cannot reuse another checkout's server.
    safe = re.sub(r"[^A-Za-z0-9_]", "_", str(path))
    module_name = f"_playgta6_verify_serve_local_{safe}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError("failed to load server/serve_local.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _resolve_env_paths(
    repo_root: Path,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve GTA6_* paths without mutating os.environ (mirrors local_env/preflight)."""
    iga = _load_tools_module("inspect_game_assets.py", "_playgta6_verify_iga")
    env_map = os.environ if env is None else env
    dotenv_vals = iga.load_dotenv(repo_root / ".env")

    raw_game = (env_map.get("GTA6_GAME_DIR") or dotenv_vals.get("GTA6_GAME_DIR", "")).strip()
    raw_data = (env_map.get("GTA6_DATA_DIR") or dotenv_vals.get("GTA6_DATA_DIR", "")).strip()
    raw_work = (env_map.get("GTA6_WORK_DIR") or dotenv_vals.get("GTA6_WORK_DIR", "")).strip()

    def _resolve(raw: str, default_rel: str | None) -> Path | None:
        if raw:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            return candidate
        if default_rel is None:
            return None
        return repo_root / default_rel

    return {
        "raw_game": raw_game,
        "raw_data": raw_data,
        "raw_work": raw_work,
        "game_path": _resolve(raw_game, None),
        "data_path": _resolve(raw_data, "local-data"),
        "work_path": _resolve(raw_work, "work"),
    }


# ---------------------------------------------------------------------------
# Individual gates
# ---------------------------------------------------------------------------


def check_game_assets_gate(
    repo_root: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> GateResult:
    """Gate 1: local Xbox package readiness via inspect_game_assets."""
    iga = _load_tools_module("inspect_game_assets.py", "_playgta6_verify_iga")
    resolved = _resolve_env_paths(repo_root, env=env)
    raw_game = resolved["raw_game"]
    game_path: Path | None = resolved["game_path"]

    if not raw_game:
        return GateResult(
            name="game_assets",
            status="skip",
            summary="GTA6_GAME_DIR is not configured",
            details={"configured": False, "exists": False},
            next_action='python tools/local_env.py --set-game "<path>"',
        )

    if game_path is None or not game_path.is_dir():
        return GateResult(
            name="game_assets",
            status="fail",
            summary="GTA6_GAME_DIR is configured but missing on disk",
            details={"configured": True, "exists": False},
            next_action='python tools/local_env.py --set-game "<path>"',
        )

    report = iga.inspect_game_directory(game_path)
    readiness = report.get("readiness") or {}
    inventory = report.get("inventory") or {}
    shell_visuals = report.get("shellVisuals") or []
    splash = next((v for v in shell_visuals if v.get("field") == "splashScreenImage"), None)
    splash_ok = bool(splash and splash.get("complete") and splash.get("headerFormat") == "png")

    details = {
        "configured": True,
        "exists": True,
        "configFound": bool(report.get("configFound")),
        "configValid": bool(report.get("configValid")),
        "status": report.get("status"),
        "splashPresent": bool(splash and splash.get("present")),
        "splashComplete": bool(splash and splash.get("complete")),
        "splashHeaderFormat": (splash or {}).get("headerFormat"),
        "splashRelativePath": (splash or {}).get("relativePath") or (splash or {}).get("declaredPath"),
        "dllCount": inventory.get("dllCount", 0),
        "rpfCount": inventory.get("rpfCount", 0),
        "allDeclaredAssetsComplete": bool(readiness.get("allDeclaredAssetsComplete")),
        "hasPartialDownloads": bool(readiness.get("hasPartialDownloads")),
    }

    if not report.get("configValid"):
        return GateResult(
            name="game_assets",
            status="fail",
            summary=sanitize_text(str(report.get("error") or "MicrosoftGame.Config invalid")),
            details=details,
            next_action="python tools/inspect_game_assets.py --from-env",
        )

    if not splash_ok:
        return GateResult(
            name="game_assets",
            status="fail",
            summary="ShellVisuals splash PNG missing or incomplete",
            details=details,
            next_action="python tools/inspect_game_assets.py --from-env",
        )

    if readiness.get("hasPartialDownloads") or not readiness.get("allDeclaredAssetsComplete"):
        return GateResult(
            name="game_assets",
            status="warn",
            summary=(
                f"Splash OK; package partial "
                f"(dlls={inventory.get('dllCount', 0)}, rpf={inventory.get('rpfCount', 0)})"
            ),
            details=details,
            next_action="python tools/inspect_game_assets.py --from-env",
        )

    return GateResult(
        name="game_assets",
        status="pass",
        summary=(
            f"MicrosoftGame.Config + splash PNG ready "
            f"(dlls={inventory.get('dllCount', 0)}, rpf={inventory.get('rpfCount', 0)})"
        ),
        details=details,
        next_action=None,
    )


def check_import_coverage_gate(
    repo_root: Path,
    *,
    audit_fn: Callable[[Path], Mapping[str, Any]] | None = None,
) -> GateResult:
    """Gate 2: real-binary PE import coverage (587/587 = 100%)."""
    if audit_fn is None:
        cic = _load_tools_module("check_import_coverage.py", "_playgta6_verify_cic")
        report = cic.audit_import_coverage(repo_root)
    else:
        report = dict(audit_fn(repo_root))

    totals = report.get("totals") or {}
    total = int(totals.get("total") or 0)
    covered = int(totals.get("shimmed") or 0) + int(totals.get("guest_recompiled") or 0)
    missing = int(totals.get("missing") or 0)
    pct = float(totals.get("effective_boot_pct") or 0.0)
    errors = list(report.get("errors") or [])

    details = {
        "total": total,
        "covered": covered,
        "missing": missing,
        "effectiveBootPct": pct,
        "shimmed": int(totals.get("shimmed") or 0),
        "guestRecompiled": int(totals.get("guest_recompiled") or 0),
        "errors": [sanitize_text(e) for e in errors],
    }

    if errors or total <= 0:
        return GateResult(
            name="import_coverage",
            status="fail",
            summary=sanitize_text(errors[0] if errors else "import coverage audit produced no symbols"),
            details=details,
            next_action="python tools/check_import_coverage.py --json",
        )

    if missing == 0 and pct >= 100.0:
        return GateResult(
            name="import_coverage",
            status="pass",
            summary=f"{covered}/{total} ({pct:.2f}%) import coverage",
            details=details,
            next_action=None,
        )

    return GateResult(
        name="import_coverage",
        status="fail",
        summary=f"{covered}/{total} ({pct:.2f}%) import coverage; {missing} missing",
        details=details,
        next_action="python tools/check_import_coverage.py --missing-only",
    )


def _find_manifest_files(recomp_dir: Path) -> dict[str, Path | None]:
    program: Path | None = None
    build: Path | None = None
    candidates = [recomp_dir]
    try:
        for child in sorted(recomp_dir.iterdir()):
            if child.is_dir() and not child.is_symlink():
                candidates.append(child)
    except OSError:
        pass

    for directory in candidates:
        prog = directory / "program-manifest.json"
        build_m = directory / "build-manifest.json"
        if program is None and prog.is_file():
            program = prog
        if build is None and build_m.is_file():
            build = build_m
        if program is not None and build is not None:
            break
    return {"program": program, "build": build}


def _count_pgim_files(recomp_dir: Path) -> tuple[int, list[str]]:
    """Recursively discover `.pgim` guest images under `recomp_dir`.

    Returns `(count, sorted_relative_posix_paths)` with host paths stripped so
    nested `playgta6-recomp build-program` outputs (`<module>/image/guest.pgim`)
    and flat layouts (`guest.pgim`) are both discovered deterministically.
    """
    rel_paths: list[str] = []
    try:
        for entry in recomp_dir.rglob("*.pgim"):
            try:
                if entry.is_symlink() or not entry.is_file():
                    continue
                rel = entry.relative_to(recomp_dir).as_posix()
                rel_paths.append(sanitize_text(rel))
            except (OSError, ValueError):
                continue
    except OSError:
        return 0, []
    rel_paths.sort(key=lambda item: (item.lower(), item))
    return len(rel_paths), rel_paths


def _lift_stats_from_manifest(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Extract shard count + lift coverage from program or build manifests."""
    shards = 0
    functions = 0
    instructions = 0
    unsupported = 0
    schema = doc.get("schema")

    if schema == "playgta6-recomp-program" or isinstance(doc.get("modules"), list):
        modules = doc.get("modules") or []
        for mod in modules:
            if not isinstance(mod, Mapping):
                continue
            lift = mod.get("lift") if isinstance(mod.get("lift"), Mapping) else {}
            functions += int(lift.get("functions") or 0)
            instructions += int(lift.get("instructions") or 0)
            unsupported += int(lift.get("unsupported") or lift.get("unsupported_instructions") or 0)
            # Prefer explicit shard lists from nested build manifests when present.
            nested = mod.get("build_manifest") if isinstance(mod.get("build_manifest"), Mapping) else {}
            # program modules may embed lift only; shard count often lives in build-manifest.
        # Fall through: also inspect top-level lift if present.
        lift_top = doc.get("lift") if isinstance(doc.get("lift"), Mapping) else {}
        if lift_top:
            functions = max(functions, int(lift_top.get("functions") or 0))
            instructions = max(instructions, int(lift_top.get("instructions") or 0))
            unsupported = max(
                unsupported,
                int(lift_top.get("unsupported") or lift_top.get("unsupported_instructions") or 0),
            )
            shard_list = lift_top.get("shards")
            if isinstance(shard_list, list):
                shards = max(shards, len(shard_list))

    lift = doc.get("lift") if isinstance(doc.get("lift"), Mapping) else {}
    if lift:
        functions = max(functions, int(lift.get("functions") or 0))
        instructions = max(instructions, int(lift.get("instructions") or 0))
        unsupported = max(
            unsupported,
            int(lift.get("unsupported_instructions") or lift.get("unsupported") or 0),
        )
        shard_list = lift.get("shards")
        if isinstance(shard_list, list):
            shards = max(shards, len(shard_list))

    lifted = max(instructions - unsupported, 0) if instructions else functions
    whole = instructions if instructions else functions
    coverage_pct = round((lifted * 100.0 / whole), 2) if whole > 0 else 0.0
    return {
        "schema": schema,
        "shardCount": shards,
        "liftedFunctions": functions,
        "liftedInstructions": lifted if instructions else functions,
        "totalInstructions": instructions,
        "unsupportedInstructions": unsupported,
        "liftCoveragePct": coverage_pct,
    }


_BOOT_OPTION_BOOL_KEYS: frozenset[str] = frozenset(
    {
        "verify_window_marks",
        "report_to_log",
    }
)
_BOOT_OPTION_UINT_KEYS: frozenset[str] = frozenset(
    {
        "max_wall_ms",
        "max_dispatches",
        "arena_log_limit",
        "import_trace_limit",
    }
)


def _inspect_boot_options(boot_dir: Path) -> dict[str, Any]:
    """Validate optional `boot-options.json` in `boot_dir` (O36/O37)."""
    options_path = boot_dir / "boot-options.json"
    if not options_path.is_file():
        return {
            "boot_options_present": False,
            "boot_options_valid": False,
            "error": None,
            "keys": [],
        }

    try:
        resolved_dir = boot_dir.resolve()
        resolved_file = options_path.resolve(strict=True)
        if not resolved_file.is_relative_to(resolved_dir):
            return {
                "boot_options_present": True,
                "boot_options_valid": False,
                "error": "boot-options.json: path escapes boot directory",
                "keys": [],
            }
        raw_text = resolved_file.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        return {
            "boot_options_present": True,
            "boot_options_valid": False,
            "error": f"boot-options.json: {type(exc).__name__}",
            "keys": [],
        }

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return {
            "boot_options_present": True,
            "boot_options_valid": False,
            "error": f"boot-options.json: {type(exc).__name__}",
            "keys": [],
        }

    if not isinstance(doc, dict):
        return {
            "boot_options_present": True,
            "boot_options_valid": False,
            "error": "boot-options.json: expected JSON object",
            "keys": [],
        }

    keys: list[str] = []
    for raw_key, value in doc.items():
        key = sanitize_text(str(raw_key))
        keys.append(key)
        if raw_key in _BOOT_OPTION_BOOL_KEYS:
            if not isinstance(value, bool):
                return {
                    "boot_options_present": True,
                    "boot_options_valid": False,
                    "error": f"boot-options.json: '{key}' must be boolean",
                    "keys": sorted(keys),
                }
        elif raw_key in _BOOT_OPTION_UINT_KEYS:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return {
                    "boot_options_present": True,
                    "boot_options_valid": False,
                    "error": f"boot-options.json: '{key}' must be non-negative integer",
                    "keys": sorted(keys),
                }
        else:
            is_scalar = (
                value is None
                or isinstance(value, (bool, int, str))
                or (isinstance(value, float) and math.isfinite(value))
            )
            if not is_scalar:
                return {
                    "boot_options_present": True,
                    "boot_options_valid": False,
                    "error": f"boot-options.json: '{key}' must be a scalar value",
                    "keys": sorted(keys),
                }

    return {
        "boot_options_present": True,
        "boot_options_valid": True,
        "error": None,
        "keys": sorted(keys),
    }


def check_recomp_artifacts_gate(
    repo_root: Path,
    recomp_dir: Path | None = None,
) -> GateResult:
    """Gate 3: recompiled program/build artifacts under work/recomp_build."""
    target = recomp_dir if recomp_dir is not None else (repo_root / DEFAULT_RECOMP_DIR_REL)
    try:
        label = repo_relative_label(target, repo_root)
    except Exception:  # noqa: BLE001
        label = "work/recomp_build"

    details: dict[str, Any] = {
        "recompDirLabel": label,
        "dirExists": False,
        "programManifest": False,
        "buildManifest": False,
        "pgimCount": 0,
        "pgimNames": [],
        "shardCount": 0,
        "liftCoveragePct": 0.0,
        "boot_options_present": False,
        "boot_options_valid": False,
    }

    if not target.is_dir():
        return GateResult(
            name="recomp_artifacts",
            status="fail",
            summary=f"recomp directory missing ({label})",
            details=details,
            next_action="cargo run -p playgta6-recomp -- build-program --help",
        )

    details["dirExists"] = True
    manifests = _find_manifest_files(target)
    program_path = manifests["program"]
    build_path = manifests["build"]
    details["programManifest"] = program_path is not None
    details["buildManifest"] = build_path is not None
    details["programManifestName"] = basename_label(program_path) if program_path else None
    details["buildManifestName"] = basename_label(build_path) if build_path else None

    pgim_count, pgim_names = _count_pgim_files(target)
    details["pgimCount"] = pgim_count
    details["pgimNames"] = pgim_names[:16]

    lift_stats: dict[str, Any] = {
        "shardCount": 0,
        "liftCoveragePct": 0.0,
        "liftedFunctions": 0,
        "totalInstructions": 0,
        "unsupportedInstructions": 0,
    }
    parse_errors: list[str] = []
    boot_options_info = _inspect_boot_options(target)
    details["boot_options_present"] = boot_options_info["boot_options_present"]
    details["boot_options_valid"] = boot_options_info["boot_options_valid"]
    if boot_options_info["keys"]:
        details["bootOptionsKeys"] = boot_options_info["keys"]
    if boot_options_info["error"]:
        details["bootOptionsError"] = boot_options_info["error"]
        parse_errors.append(boot_options_info["error"])

    for path in (program_path, build_path):
        if path is None:
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            parse_errors.append(f"{basename_label(path)}: {type(exc).__name__}")
            continue
        if not isinstance(doc, dict):
            parse_errors.append(f"{basename_label(path)}: expected JSON object")
            continue
        stats = _lift_stats_from_manifest(doc)
        # Prefer richer stats (higher instruction counts / shards).
        if stats["totalInstructions"] >= lift_stats["totalInstructions"]:
            lift_stats.update(stats)
        elif stats["shardCount"] > lift_stats["shardCount"]:
            lift_stats["shardCount"] = stats["shardCount"]

    details.update(
        {
            "shardCount": lift_stats.get("shardCount", 0),
            "liftCoveragePct": lift_stats.get("liftCoveragePct", 0.0),
            "liftedFunctions": lift_stats.get("liftedFunctions", 0),
            "totalInstructions": lift_stats.get("totalInstructions", 0),
            "unsupportedInstructions": lift_stats.get("unsupportedInstructions", 0),
            "parseErrors": parse_errors,
        }
    )

    has_manifest = bool(program_path or build_path)
    if not has_manifest:
        return GateResult(
            name="recomp_artifacts",
            status="fail",
            summary="no program-manifest.json or build-manifest.json found",
            details=details,
            next_action="cargo run -p playgta6-recomp -- build-program --help",
        )

    if pgim_count <= 0:
        return GateResult(
            name="recomp_artifacts",
            status="fail",
            summary="manifest present but no .pgim guest image found",
            details=details,
            next_action="cargo run -p playgta6-recomp -- build-program --help",
        )

    if details["boot_options_present"] and not details["boot_options_valid"]:
        return GateResult(
            name="recomp_artifacts",
            status="fail",
            summary=sanitize_text(
                str(details.get("bootOptionsError") or "boot-options.json invalid")
            ),
            details=details,
            next_action="fix or remove boot-options.json in the boot directory",
        )

    coverage = float(lift_stats.get("liftCoveragePct") or 0.0)
    shards = int(lift_stats.get("shardCount") or 0)
    summary = (
        f"manifest + {pgim_count} .pgim; shards={shards}; lift={coverage:.2f}%"
    )
    if coverage <= 0.0 and int(lift_stats.get("liftedFunctions") or 0) <= 0:
        return GateResult(
            name="recomp_artifacts",
            status="warn",
            summary=summary + " (lift coverage unavailable)",
            details=details,
            next_action="python tools/audit_recomp_build.py work/recomp_build --json",
        )

    return GateResult(
        name="recomp_artifacts",
        status="pass",
        summary=summary,
        details=details,
        next_action=None,
    )


def _inspect_shader_sidecars(
    pack_files: Sequence[Path],
    shaders_dir: Path,
) -> dict[str, Any]:
    """Inspect adjacent `<pack>.reflect.json` sidecar files for staged `.wgsl.pack` bundles."""
    present_count = 0
    all_valid = True
    stage_count = 0
    sidecar_names: list[str] = []
    sidecar_errors: list[str] = []

    try:
        shaders_dir_resolved = shaders_dir.resolve()
    except OSError:
        shaders_dir_resolved = None

    for pack_path in pack_files:
        sidecar_path = pack_path.with_name(f"{pack_path.name}.reflect.json")
        if not sidecar_path.is_file():
            continue
        present_count += 1
        label = basename_label(sidecar_path)
        sidecar_names.append(label)

        try:
            resolved_sidecar = sidecar_path.resolve(strict=True)
            if shaders_dir_resolved is not None and not resolved_sidecar.is_relative_to(
                shaders_dir_resolved
            ):
                all_valid = False
                sidecar_errors.append(f"{label}: path escapes shaders directory")
                continue
            raw_text = resolved_sidecar.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            all_valid = False
            sidecar_errors.append(f"{label}: {type(exc).__name__}")
            continue

        try:
            doc = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            all_valid = False
            sidecar_errors.append(f"{label}: {type(exc).__name__}")
            continue

        if not isinstance(doc, dict):
            all_valid = False
            sidecar_errors.append(f"{label}: expected JSON object")
            continue

        if "schema" in doc and not isinstance(doc["schema"], str):
            all_valid = False
            sidecar_errors.append(f"{label}: schema must be string")
            continue

        if "version" in doc and (
            not isinstance(doc["version"], int) or isinstance(doc["version"], bool)
        ):
            all_valid = False
            sidecar_errors.append(f"{label}: version must be integer")
            continue

        stages = doc.get("stages")
        if not isinstance(stages, dict) or not all(
            isinstance(stage_entry, dict) for stage_entry in stages.values()
        ):
            all_valid = False
            sidecar_errors.append(f"{label}: missing or invalid stages object")
            continue

        stage_count += len(stages)

    sidecar_present = present_count > 0
    return {
        "sidecar_present": sidecar_present,
        "sidecar_valid": sidecar_present and all_valid,
        "sidecar_stage_count": stage_count,
        "sidecar_names": sidecar_names,
        "sidecar_errors": sidecar_errors,
    }


def check_shader_pack_gate(
    repo_root: Path,
    *,
    data_dir: Path | None = None,
) -> GateResult:
    """Gate 4: WGPK shader pack + reflection-manifest.json under local-data/shaders/."""
    wgpk = _load_tools_module("inspect_wgsl_pack.py", "_playgta6_verify_wgpk")
    shaders_dir = (
        (data_dir / "shaders")
        if data_dir is not None
        else (repo_root / DEFAULT_SHADERS_DIR_REL)
    )

    details: dict[str, Any] = {
        "shadersDirLabel": "local-data/shaders",
        "dirExists": shaders_dir.is_dir(),
        "packCount": 0,
        "packNames": [],
        "validPackCount": 0,
        "reflectionPresent": False,
        "reflectionValid": False,
        "reflectionShaderCount": 0,
        "hashParityOk": False,
        "sidecar_present": False,
        "sidecar_valid": False,
        "sidecar_stage_count": 0,
    }

    if not shaders_dir.is_dir():
        return GateResult(
            name="shader_pack",
            status="fail",
            summary="local-data/shaders/ directory missing",
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    pack_files: list[Path] = []
    try:
        for entry in sorted(shaders_dir.iterdir(), key=lambda p: p.name.lower()):
            if entry.is_file() and entry.name.lower().endswith(".wgsl.pack"):
                pack_files.append(entry)
    except OSError:
        pack_files = []

    details["packCount"] = len(pack_files)
    details["packNames"] = [p.name for p in pack_files]

    if not pack_files:
        return GateResult(
            name="shader_pack",
            status="fail",
            summary="no .wgsl.pack bundles in local-data/shaders/",
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    pack_hashes: set[str] = set()
    valid_packs = 0
    pack_errors: list[str] = []
    preferred = next((p for p in pack_files if p.name.lower() == "fallback.wgsl.pack"), pack_files[0])

    for pack_path in pack_files:
        try:
            report = wgpk.inspect_pack(pack_path, source_label=pack_path.name)
            valid_packs += 1
            for shader in report.shaders:
                pack_hashes.add(shader.hash_hex)
            if pack_path == preferred:
                details["preferredPack"] = pack_path.name
                details["preferredEntryCount"] = report.entry_count
                details["preferredClean"] = report.clean
        except Exception as exc:  # noqa: BLE001 - surface sanitized parse failures
            pack_errors.append(f"{pack_path.name}: {sanitize_text(str(exc))}")

    details["validPackCount"] = valid_packs
    details["packErrors"] = pack_errors

    sidecar_info = _inspect_shader_sidecars(pack_files, shaders_dir)
    details["sidecar_present"] = sidecar_info["sidecar_present"]
    details["sidecar_valid"] = sidecar_info["sidecar_valid"]
    details["sidecar_stage_count"] = sidecar_info["sidecar_stage_count"]
    if sidecar_info["sidecar_names"]:
        details["sidecarNames"] = sidecar_info["sidecar_names"]
    if sidecar_info["sidecar_errors"]:
        details["sidecarErrors"] = sidecar_info["sidecar_errors"]

    if valid_packs <= 0:
        return GateResult(
            name="shader_pack",
            status="fail",
            summary="no valid WGPK .wgsl.pack bundles found",
            details=details,
            next_action="python tools/inspect_wgsl_pack.py inspect local-data/shaders/fallback.wgsl.pack",
        )

    reflection_path = shaders_dir / "reflection-manifest.json"
    details["reflectionPresent"] = reflection_path.is_file()
    reflection_hashes: set[str] = set()
    if reflection_path.is_file():
        try:
            reflection = json.loads(reflection_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            details["reflectionError"] = sanitize_text(str(exc))
            reflection = None
        if isinstance(reflection, dict):
            shaders_map = reflection.get("shaders")
            if isinstance(shaders_map, dict):
                reflection_hashes = {str(k) for k in shaders_map.keys()}
                details["reflectionValid"] = True
                details["reflectionShaderCount"] = len(reflection_hashes)
                details["schemaVersion"] = reflection.get("schema_version")
            else:
                details["reflectionError"] = "reflection-manifest.json missing shaders object"

    if not details["reflectionPresent"]:
        return GateResult(
            name="shader_pack",
            status="fail",
            summary=f"{valid_packs} valid WGPK pack(s) but reflection-manifest.json missing",
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    if not details["reflectionValid"]:
        return GateResult(
            name="shader_pack",
            status="fail",
            summary="reflection-manifest.json present but invalid",
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    # Parity: every pack hash should appear in the reflection manifest (manifest may be a superset).
    missing_in_reflection = sorted(pack_hashes - reflection_hashes)
    details["hashParityOk"] = len(missing_in_reflection) == 0
    details["packHashCount"] = len(pack_hashes)
    details["missingInReflection"] = missing_in_reflection[:16]

    if missing_in_reflection:
        return GateResult(
            name="shader_pack",
            status="warn",
            summary=(
                f"WGPK packs OK ({valid_packs}); reflection missing "
                f"{len(missing_in_reflection)} pack hash(es)"
            ),
            details=details,
            next_action="python tools/gen_fallback_shaders.py --out-dir local-data/shaders",
        )

    if details["sidecar_present"] and not details["sidecar_valid"]:
        return GateResult(
            name="shader_pack",
            status="warn",
            summary=(
                f"{valid_packs} valid WGPK pack(s); reflection-manifest.json OK "
                f"but adjacent .reflect.json sidecar invalid"
            ),
            details=details,
            next_action="python tools/stage_reflection_sidecar.py --help",
        )

    sidecar_note = (
        f"; sidecar OK ({details['sidecar_stage_count']} stages)"
        if details["sidecar_present"] and details["sidecar_valid"]
        else ""
    )
    return GateResult(
        name="shader_pack",
        status="pass",
        summary=(
            f"{valid_packs} valid WGPK pack(s); reflection-manifest.json "
            f"({details['reflectionShaderCount']} shaders) parity OK{sidecar_note}"
        ),
        details=details,
        next_action=None,
    )


def classify_runtime_wasm_kind(
    *,
    byte_size: int,
    export_names: Sequence[str],
) -> str:
    """Distinguish smoke shell (< 1 MiB) from linked guest runtime."""
    guest_exports = [name for name in export_names if name in GUEST_BOOT_EXPORT_MARKERS]
    if byte_size >= SMOKE_SHELL_MAX_BYTES or guest_exports:
        return "linked_guest_runtime"
    return "smoke_shell"


def check_wasm64_runtime_gate(repo_root: Path) -> GateResult:
    """Gate 5: web/dist/runtime.wasm Wasm64/threads/ABI + smoke vs linked guest."""
    wasm_mod = _load_tools_module("check_wasm_exports.py", "_playgta6_verify_wasm")
    wasm_path = repo_root / RUNTIME_WASM_REL
    details: dict[str, Any] = {
        "pathLabel": "web/dist/runtime.wasm",
        "present": False,
        "byteSize": 0,
        "kind": None,
        "isMemory64": False,
        "isSharedMemory": False,
        "hostImportsOk": False,
        "hostImportsValid": 0,
        "hostImportsTotal": len(getattr(wasm_mod, "REQUIRED_HOST_IMPORTS", ())),
        "guestBootExports": [],
        "exportCount": 0,
    }

    if not wasm_path.is_file():
        return GateResult(
            name="wasm64_runtime",
            status="fail",
            summary="web/dist/runtime.wasm missing",
            details=details,
            next_action="pwsh runtime/build_wasm.ps1",
        )

    details["present"] = True
    try:
        raw = wasm_path.read_bytes()
    except OSError as exc:
        return GateResult(
            name="wasm64_runtime",
            status="fail",
            summary=f"unable to read runtime.wasm ({sanitize_text(exc.strerror or type(exc).__name__)})",
            details=details,
            next_action="pwsh runtime/build_wasm.ps1",
        )

    details["byteSize"] = len(raw)
    try:
        module = wasm_mod.parse_wasm_module(raw)
        report = wasm_mod.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
            require_host_imports=True,
            path="web/dist/runtime.wasm",
        )
    except Exception as exc:  # noqa: BLE001
        return GateResult(
            name="wasm64_runtime",
            status="fail",
            summary=f"runtime.wasm parse/validate failed: {sanitize_text(str(exc))}",
            details=details,
            next_action="python tools/check_wasm_exports.py web/dist/runtime.wasm --json",
        )

    export_names = [exp.name for exp in report.exports]
    guest_exports = sorted(name for name in export_names if name in GUEST_BOOT_EXPORT_MARKERS)
    kind = classify_runtime_wasm_kind(byte_size=report.byte_size, export_names=export_names)
    host_ok_count = sum(1 for h in report.host_imports if h.ok)

    details.update(
        {
            "byteSize": report.byte_size,
            "kind": kind,
            "isMemory64": report.is_memory64,
            "isSharedMemory": report.is_shared_memory,
            "hostImportsOk": report.all_host_imports_ok,
            "hostImportsValid": host_ok_count,
            "hostImportsTotal": len(report.host_imports),
            "guestBootExports": guest_exports,
            "exportCount": report.exports_count,
            "errors": [sanitize_text(e) for e in report.errors],
            "warnings": [sanitize_text(w) for w in report.warnings],
        }
    )

    abi_ok = bool(
        report.is_memory64 and report.is_shared_memory and report.all_host_imports_ok
    )
    if not abi_ok:
        return GateResult(
            name="wasm64_runtime",
            status="fail",
            summary=(
                f"runtime.wasm ABI incomplete "
                f"(memory64={report.is_memory64}, shared={report.is_shared_memory}, "
                f"host={host_ok_count}/{len(report.host_imports)}; kind={kind})"
            ),
            details=details,
            next_action=(
                "python tools/check_wasm_exports.py web/dist/runtime.wasm "
                "--require-memory64 --require-shared-memory --require-host-imports"
            ),
        )

    if kind == "smoke_shell":
        return GateResult(
            name="wasm64_runtime",
            status="warn",
            summary=(
                f"ABI OK ({host_ok_count}/{len(report.host_imports)}) but runtime.wasm is "
                f"smoke_shell ({report.byte_size} bytes < 1 MiB; no guest boot exports)"
            ),
            details=details,
            next_action="python tools/build_guest_wasm.py work/recomp_build --verify",
        )

    return GateResult(
        name="wasm64_runtime",
        status="pass",
        summary=(
            f"linked_guest_runtime ABI OK "
            f"({report.byte_size} bytes; host {host_ok_count}/{len(report.host_imports)}; "
            f"guest exports={len(guest_exports)})"
        ),
        details=details,
        next_action=None,
    )


def _http_get(
    host: str,
    port: int,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout: float = 3.0,
) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path, headers=dict(headers or {}))
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, body
    finally:
        conn.close()


def _probe_http11_keep_alive(
    host: str,
    port: int,
    *,
    timeout: float = 3.0,
) -> bool:
    """Send two sequential requests over one HTTPConnection and verify HTTP/1.1 reuse."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", "/api/status", headers={"Connection": "keep-alive"})
        resp1 = conn.getresponse()
        resp1.read()
        sock1 = conn.sock
        if (
            resp1.status != 200
            or resp1.version != 11
            or resp1.will_close
            or sock1 is None
        ):
            return False
        conn_hdr1 = (resp1.getheader("Connection") or "").lower()
        if "close" in conn_hdr1:
            return False
        try:
            fd1 = sock1.fileno()
            local_addr1 = sock1.getsockname()
        except OSError:
            return False
        if fd1 < 0:
            return False

        conn.request("HEAD", "/", headers={"Connection": "keep-alive"})
        resp2 = conn.getresponse()
        resp2.read()
        sock2 = conn.sock
        if (
            resp2.status != 200
            or resp2.version != 11
            or resp2.will_close
            or sock2 is None
        ):
            return False
        conn_hdr2 = (resp2.getheader("Connection") or "").lower()
        if "close" in conn_hdr2:
            return False
        try:
            fd2 = sock2.fileno()
            local_addr2 = sock2.getsockname()
        except OSError:
            return False
        return sock2 is sock1 and fd2 == fd1 and local_addr2 == local_addr1
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def check_local_server_gate(
    repo_root: Path,
    *,
    data_dir: Path | None = None,
    game_dir: Path | None = None,
    recomp_dir: Path | None = None,
    skip_server: bool = False,
) -> GateResult:
    """Gate 6: ephemeral COOP/COEP server + Range 206 + boot endpoints."""
    if skip_server:
        return GateResult(
            name="local_server",
            status="skip",
            summary="server check skipped (--skip-server)",
            details={"checked": False, "skipped": True},
            next_action=None,
        )

    details: dict[str, Any] = {
        "checked": True,
        "mode": None,
        "coopValid": False,
        "coepValid": False,
        "corpValid": False,
        "acceptRangesValid": False,
        "statusEndpointOk": False,
        "range206Ok": False,
        "rootOk": False,
        "shadersOk": False,
        "bootOk": False,
        "gameSplashOk": None,
        "splashUrl": None,
        "http11_keep_alive": False,
    }

    server_file = repo_root / "server" / "serve_local.py"
    if not server_file.is_file():
        return GateResult(
            name="local_server",
            status="fail",
            summary="server/serve_local.py not found",
            details=details,
            next_action="python server/serve_local.py",
        )

    try:
        serve = _load_server_module(repo_root)
    except Exception:  # noqa: BLE001
        return GateResult(
            name="local_server",
            status="fail",
            summary="server/serve_local.py import failed",
            details=details,
            next_action="python server/serve_local.py",
        )

    web_root = repo_root / "web"
    resolved_data = data_dir if data_dir is not None else (repo_root / "local-data")
    shaders_root = resolved_data / "shaders"
    boot_root = recomp_dir if recomp_dir is not None else (repo_root / DEFAULT_RECOMP_DIR_REL)
    resolved_game = game_dir

    handler_cls = getattr(serve, "PlayGta6Handler", None)
    if handler_cls is None:
        return GateResult(
            name="local_server",
            status="fail",
            summary="PlayGta6Handler not found in server module",
            details=details,
            next_action="python server/serve_local.py",
        )

    class _QuietHandler(handler_cls):  # type: ignore[misc,valid-type]
        def log_message(self, format: str, *args: Any) -> None:
            return None

    httpd: ThreadingHTTPServer | None = None
    thread: threading.Thread | None = None
    try:
        create_server = getattr(serve, "create_server", None)
        if callable(create_server):
            httpd = create_server(
                "127.0.0.1",
                0,
                web_root,
                resolved_data,
                game_root=resolved_game,
                shaders_root=shaders_root,
                boot_root=boot_root,
            )
            # Replace handler logging by wrapping serve_forever requests through QuietHandler
            # is hard after create_server; mute via monkeypatch on instance RequestHandlerClass.
            httpd.RequestHandlerClass = partial(  # type: ignore[method-assign]
                _QuietHandler,
                web_root=web_root,
                data_root=resolved_data,
                game_root=resolved_game,
                shaders_root=shaders_root,
                boot_root=boot_root,
            )
        else:
            handler_factory = partial(
                _QuietHandler,
                web_root=web_root,
                data_root=resolved_data,
                game_root=resolved_game,
                shaders_root=shaders_root,
                boot_root=boot_root,
            )
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler_factory)

        port = int(httpd.server_address[1])
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        details["mode"] = "ephemeral-http"

        # /api/status + isolation headers
        status_code, headers, body = _http_get("127.0.0.1", port, "/api/status")
        details["coopValid"] = headers.get("cross-origin-opener-policy") == "same-origin"
        details["coepValid"] = headers.get("cross-origin-embedder-policy") == "require-corp"
        details["corpValid"] = headers.get("cross-origin-resource-policy") == "same-origin"
        details["acceptRangesValid"] = headers.get("accept-ranges") == "bytes"
        status_ok = False
        splash_url: str | None = None
        shader_pack_url: str | None = None
        pgim_url: str | None = None
        if status_code == 200:
            try:
                payload = json.loads(body.decode("utf-8"))
                status_ok = bool(
                    isinstance(payload, dict) and payload.get("ok") and payload.get("coopCoep")
                )
                if isinstance(payload, dict):
                    splash_url = payload.get("splashImage")
                    shader_pack_url = payload.get("shaderPackUrl")
                    pgim_url = payload.get("pgimUrl")
            except (UnicodeDecodeError, json.JSONDecodeError):
                status_ok = False
        details["statusEndpointOk"] = status_ok
        details["splashUrl"] = splash_url
        details["shaderPackUrl"] = shader_pack_url
        details["pgimUrl"] = pgim_url

        # `/` root
        root_code, _, _ = _http_get("127.0.0.1", port, "/")
        details["rootOk"] = root_code == 200

        # Range 206 against index.html (or status body fallback file)
        range_ok = False
        range_code, range_headers, _ = _http_get(
            "127.0.0.1",
            port,
            "/",
            headers={"Range": "bytes=0-3"},
        )
        if range_code == 206 and range_headers.get("content-range", "").startswith("bytes "):
            range_ok = True
        details["range206Ok"] = range_ok

        # /shaders/
        shaders_ok = False
        if shader_pack_url:
            sc, _, _ = _http_get("127.0.0.1", port, shader_pack_url)
            shaders_ok = sc == 200
        else:
            # Directory may 404 for listing; accept reflection or any pack probe.
            probe = "/shaders/reflection-manifest.json"
            sc, _, _ = _http_get("127.0.0.1", port, probe)
            shaders_ok = sc == 200
        details["shadersOk"] = shaders_ok

        # /boot/
        boot_ok = False
        if pgim_url:
            bc, _, _ = _http_get("127.0.0.1", port, pgim_url)
            boot_ok = bc == 200
        elif boot_root.is_dir():
            # No pgim yet — endpoint mount still considered present if meta.json/boot allowlist 404s cleanly.
            bc, _, _ = _http_get("127.0.0.1", port, "/boot/missing.pgim")
            boot_ok = bc in (404, 403)
        details["bootOk"] = boot_ok

        # /game/<splash>
        game_splash_ok: bool | None = None
        if splash_url:
            gc, _, _ = _http_get("127.0.0.1", port, splash_url)
            game_splash_ok = gc == 200
        elif resolved_game is None:
            game_splash_ok = None
        else:
            game_splash_ok = False
        details["gameSplashOk"] = game_splash_ok

        # HTTP/1.1 persistent keep-alive reuse check (GET /api/status -> HEAD /)
        details["http11_keep_alive"] = _probe_http11_keep_alive("127.0.0.1", port)

        isolation_ok = bool(
            details["coopValid"]
            and details["coepValid"]
            and details["statusEndpointOk"]
            and details["range206Ok"]
            and details["rootOk"]
            and details["http11_keep_alive"]
        )
        endpoints_ok = bool(details["shadersOk"] and details["bootOk"])
        splash_requirement_ok = game_splash_ok is not False

        if isolation_ok and endpoints_ok and splash_requirement_ok:
            splash_note = (
                f"; splash={splash_url}"
                if splash_url
                else "; splash=n/a"
            )
            return GateResult(
                name="local_server",
                status="pass",
                summary=f"COOP/COEP + HTTP/1.1 keep-alive + Range 206 + boot endpoints OK{splash_note}",
                details=details,
                next_action=None,
            )

        missing: list[str] = []
        if not details["coopValid"] or not details["coepValid"]:
            missing.append("isolation-headers")
        if not details["statusEndpointOk"]:
            missing.append("/api/status")
        if not details["range206Ok"]:
            missing.append("range-206")
        if not details["rootOk"]:
            missing.append("/")
        if not details["http11_keep_alive"]:
            missing.append("http11-keep-alive")
        if not details["shadersOk"]:
            missing.append("/shaders/")
        if not details["bootOk"]:
            missing.append("/boot/")
        if game_splash_ok is False:
            missing.append("/game/<splash>")

        return GateResult(
            name="local_server",
            status="fail",
            summary="server self-check failed: " + ", ".join(missing),
            details=details,
            next_action="python server/serve_local.py",
        )
    except OSError:
        # Restricted environments: fall back to in-process header probe.
        try:
            captured: dict[str, str] = {}

            class _ProbeHandler(handler_cls):  # type: ignore[misc,valid-type]
                def __init__(self) -> None:
                    self.request_version = "HTTP/1.1"

                def send_header(self, keyword: str, value: str) -> None:
                    captured[keyword.lower()] = value

                def flush_headers(self) -> None:
                    return None

            probe = _ProbeHandler()
            handler_cls.end_headers(probe)
            details["mode"] = "in-process"
            details["coopValid"] = captured.get("cross-origin-opener-policy") == "same-origin"
            details["coepValid"] = captured.get("cross-origin-embedder-policy") == "require-corp"
            details["corpValid"] = captured.get("cross-origin-resource-policy") == "same-origin"
            details["acceptRangesValid"] = captured.get("accept-ranges") == "bytes"
            details["statusEndpointOk"] = callable(getattr(serve, "build_status_payload", None))
            details["http11_keep_alive"] = (
                getattr(handler_cls, "protocol_version", "") == "HTTP/1.1"
                and captured.get("connection", "").lower() == "keep-alive"
            )
            ok = bool(details["coopValid"] and details["coepValid"])
            return GateResult(
                name="local_server",
                status="pass" if ok else "fail",
                summary=(
                    "in-process COOP/COEP headers OK (socket bind unavailable)"
                    if ok
                    else "in-process COOP/COEP headers missing"
                ),
                details=details,
                next_action=None if ok else "python server/serve_local.py",
            )
        except Exception:  # noqa: BLE001
            return GateResult(
                name="local_server",
                status="fail",
                summary="server header self-check failed",
                details=details,
                next_action="python server/serve_local.py",
            )
    except Exception:  # noqa: BLE001
        return GateResult(
            name="local_server",
            status="fail",
            summary="server self-check failed",
            details=details,
            next_action="python server/serve_local.py",
        )
    finally:
        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception:  # noqa: BLE001
                pass
            try:
                httpd.server_close()
            except Exception:  # noqa: BLE001
                pass
        if thread is not None:
            thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Optional boot-report threshold gate (T76: --min-fps & --require-non-black)
# ---------------------------------------------------------------------------


def _finite_non_negative_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    fval = float(value)
    if not math.isfinite(fval) or fval < 0.0:
        return None
    return fval


def _first_non_negative_float(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_non_negative_float(value)
        if parsed is not None:
            return parsed
    return None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value >= 0.0 and value.is_integer():
        return int(value)
    return None


def _first_non_negative_int(*values: Any) -> int | None:
    for value in values:
        parsed = _non_negative_int(value)
        if parsed is not None:
            return parsed
    return None


def extract_boot_report_metrics(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Extract sanitized FPS and non-black canvas metrics from a boot report dict."""
    summary = doc.get("summary") if isinstance(doc.get("summary"), Mapping) else {}
    counters = doc.get("counters") if isinstance(doc.get("counters"), Mapping) else {}
    first_present_obj = (
        doc.get("first_present") if isinstance(doc.get("first_present"), Mapping) else {}
    )
    milestones = doc.get("milestones") if isinstance(doc.get("milestones"), list) else []

    raw_presents = doc.get("presents")
    presents_list: list[Mapping[str, Any]] = (
        [item for item in raw_presents if isinstance(item, Mapping)]
        if isinstance(raw_presents, list)
        else []
    )
    wall_times: list[float] = []
    for sample in presents_list:
        for key in ("wall_ms", "wallMs", "elapsed_ms", "elapsedMs"):
            val = _finite_non_negative_float(sample.get(key))
            if val is not None:
                wall_times.append(val)
                break

    presents_count = _first_non_negative_int(
        counters.get("presents"),
        summary.get("presents"),
        summary.get("presentCount"),
        raw_presents if not isinstance(raw_presents, list) else None,
        len(presents_list) if presents_list else None,
        doc.get("presents_sampled"),
    )

    milestone_first_present_ms: float | None = None
    milestone_first_visible_ms: float | None = None
    for entry in milestones:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("milestone") or entry.get("name") or "").strip().upper()
        elapsed = _first_non_negative_float(entry.get("elapsed_ms"), entry.get("elapsedMs"))
        if elapsed is None:
            continue
        if name == "FIRST_PRESENT" and milestone_first_present_ms is None:
            milestone_first_present_ms = elapsed
        elif name == "FIRST_VISIBLE_FRAME" and milestone_first_visible_ms is None:
            milestone_first_visible_ms = elapsed

    first_present_ms = _first_non_negative_float(
        doc.get("first_present_ms"),
        doc.get("firstPresentMs"),
        summary.get("first_present_ms"),
        summary.get("firstPresentMs"),
        counters.get("first_present_ms"),
        first_present_obj.get("elapsed_ms"),
        first_present_obj.get("elapsedMs"),
        milestone_first_present_ms,
        wall_times[0] if wall_times else None,
    )

    last_present_ms = _first_non_negative_float(
        doc.get("last_present_ms"),
        doc.get("lastPresentMs"),
        summary.get("last_present_ms"),
        summary.get("lastPresentMs"),
        counters.get("last_present_ms"),
        wall_times[-1] if len(wall_times) >= 2 else None,
    )

    present_span_ms: float | None = None
    if (
        first_present_ms is not None
        and last_present_ms is not None
        and last_present_ms > first_present_ms
    ):
        present_span_ms = round(last_present_ms - first_present_ms, 2)

    direct_fps_avg = _first_non_negative_float(
        doc.get("present_fps_avg"),
        doc.get("presentFpsAvg"),
        summary.get("present_fps_avg"),
        summary.get("presentFpsAvg"),
        counters.get("present_fps_avg"),
    )

    fallback_fps = _first_non_negative_float(
        doc.get("present_fps"),
        doc.get("presentFps"),
        summary.get("present_fps"),
        summary.get("presentFps"),
        counters.get("present_fps"),
    )

    raw_fps: float | None = None
    fps_source: str | None = None

    if direct_fps_avg is not None:
        raw_fps = direct_fps_avg
        fps_source = "present_fps_avg"
    elif (
        presents_count is not None
        and first_present_ms is not None
        and last_present_ms is not None
        and last_present_ms > first_present_ms
    ):
        raw_fps = (float(presents_count) * 1000.0) / (last_present_ms - first_present_ms)
        fps_source = "computed_present_span"
    elif fallback_fps is not None:
        raw_fps = fallback_fps
        fps_source = "present_fps"
    else:
        elapsed_ms = _finite_non_negative_float(doc.get("elapsed_ms"))
        if (
            presents_count is not None
            and presents_count > 0
            and first_present_ms is not None
            and elapsed_ms is not None
            and elapsed_ms > first_present_ms
        ):
            present_span_ms = round(elapsed_ms - first_present_ms, 2)
            raw_fps = (float(presents_count) * 1000.0) / (elapsed_ms - first_present_ms)
            fps_source = "computed_elapsed"

    non_black_canvas_frames = _first_non_negative_int(
        summary.get("nonBlackCanvasFrames"),
        summary.get("non_black_canvas_frames"),
        doc.get("nonBlackCanvasFrames"),
        doc.get("non_black_canvas_frames"),
        counters.get("nonBlackCanvasFrames"),
        counters.get("non_black_canvas_frames"),
    )

    raw_captured = summary.get("capturedFrames")
    if raw_captured is None:
        raw_captured = doc.get("capturedFrames")
    if raw_captured is None:
        raw_captured_list = doc.get("captured_frames")
        if isinstance(raw_captured_list, list):
            raw_captured = len(raw_captured_list)
    captured_frames = _non_negative_int(raw_captured)

    first_visible_frame_s = _first_non_negative_float(
        doc.get("first_visible_frame_s"),
        doc.get("firstVisibleFrameS"),
        summary.get("first_visible_frame_s"),
        summary.get("firstVisibleFrameS"),
    )
    if first_visible_frame_s is None:
        fvf_ms = _first_non_negative_float(
            summary.get("first_visible_frame_ms"),
            summary.get("firstVisibleFrameMs"),
            doc.get("first_visible_frame_ms"),
            doc.get("firstVisibleFrameMs"),
            milestone_first_visible_ms,
        )
        if fvf_ms is not None:
            first_visible_frame_s = round(fvf_ms / 1000.0, 3)

    has_non_black = bool(
        (non_black_canvas_frames is not None and non_black_canvas_frames > 0)
        or (first_visible_frame_s is not None)
    )

    return {
        "presents_count": presents_count,
        "first_present_ms": round(first_present_ms, 2) if first_present_ms is not None else None,
        "last_present_ms": round(last_present_ms, 2) if last_present_ms is not None else None,
        "present_span_ms": present_span_ms,
        "present_fps_avg": round(raw_fps, 2) if raw_fps is not None else None,
        "present_fps_avg_raw": raw_fps,
        "present_fps_source": fps_source,
        "non_black_canvas_frames": non_black_canvas_frames,
        "captured_frames": captured_frames,
        "first_visible_frame_s": (
            round(first_visible_frame_s, 3) if first_visible_frame_s is not None else None
        ),
        "has_non_black_frame": has_non_black,
    }


def _discover_boot_report(
    repo_root: Path,
    *,
    work_dir: Path,
    recomp_dir: Path,
) -> Path | None:
    """Find the most recent boot-report.json under standard local run/work folders."""
    runs_dir = work_dir / "browser-boot" / "runs"
    if runs_dir.is_dir():
        candidates: list[tuple[int, str, Path]] = []
        try:
            runs_resolved = runs_dir.resolve()
            for child in runs_dir.iterdir():
                if child.is_symlink() or not child.is_dir():
                    continue
                candidate = child / "boot-report.json"
                if candidate.is_symlink() or not candidate.is_file():
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                    if not resolved.is_relative_to(runs_resolved):
                        continue
                    mtime_ns = resolved.stat().st_mtime_ns
                except OSError:
                    continue
                candidates.append((mtime_ns, child.name, candidate))
        except OSError:
            candidates = []
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2]

    for fallback in (recomp_dir / "boot-report.json", work_dir / "boot-report.json"):
        if fallback.is_file() and not fallback.is_symlink():
            return fallback
    return None


def check_boot_report_gate(
    repo_root: Path,
    *,
    boot_report: Path | str | None = None,
    work_dir: Path | None = None,
    recomp_dir: Path | None = None,
    min_fps: float | None = None,
    require_non_black: bool = False,
) -> GateResult:
    """Optional boot-report verification gate (`--min-fps` & `--require-non-black`)."""
    resolved_work = work_dir if work_dir is not None else (repo_root / "work")
    resolved_recomp = (
        recomp_dir if recomp_dir is not None else (repo_root / DEFAULT_RECOMP_DIR_REL)
    )

    if min_fps is not None:
        if (
            isinstance(min_fps, bool)
            or not isinstance(min_fps, (int, float))
            or not math.isfinite(float(min_fps))
            or float(min_fps) < 0.0
        ):
            return GateResult(
                name="boot_report",
                status="fail",
                summary="invalid --min-fps threshold (must be a finite non-negative number)",
                details={
                    "bootReportLabel": None,
                    "sourceMode": "explicit" if boot_report is not None else "discovered",
                    "reportPresent": False,
                    "reportValid": False,
                    "min_fps": None,
                    "min_fps_ok": False,
                    "present_fps_avg": None,
                    "present_fps_source": None,
                    "presents": None,
                    "first_present_ms": None,
                    "last_present_ms": None,
                    "present_span_ms": None,
                    "require_non_black": bool(require_non_black),
                    "non_black_ok": None,
                    "nonBlackCanvasFrames": None,
                    "capturedFrames": None,
                    "first_visible_frame_s": None,
                    "failures": ["invalid --min-fps threshold"],
                },
                next_action="pass a finite non-negative value to --min-fps (e.g. --min-fps 30)",
            )
        min_fps = float(min_fps)

    if boot_report is not None:
        candidate = Path(boot_report)
        if not candidate.is_absolute():
            candidate = repo_root / candidate
        report_path: Path | None = (
            (candidate / "boot-report.json") if candidate.is_dir() else candidate
        )
        source_mode = "explicit"
    else:
        report_path = _discover_boot_report(
            repo_root,
            work_dir=resolved_work,
            recomp_dir=resolved_recomp,
        )
        source_mode = "discovered"

    report_label = (
        repo_relative_label(report_path, repo_root) if report_path is not None else None
    )

    details: dict[str, Any] = {
        "bootReportLabel": report_label,
        "sourceMode": source_mode,
        "reportPresent": False,
        "reportValid": False,
        "min_fps": min_fps,
        "min_fps_ok": None,
        "present_fps_avg": None,
        "present_fps_source": None,
        "presents": None,
        "first_present_ms": None,
        "last_present_ms": None,
        "present_span_ms": None,
        "require_non_black": bool(require_non_black),
        "non_black_ok": None,
        "nonBlackCanvasFrames": None,
        "capturedFrames": None,
        "first_visible_frame_s": None,
        "failures": [],
    }

    if report_path is None or not report_path.is_file():
        missing_note = f" ({report_label})" if report_label else ""
        failure_msg = f"boot-report.json not found{missing_note}"
        details["failures"] = [failure_msg]
        if min_fps is not None:
            details["min_fps_ok"] = False
        if require_non_black:
            details["non_black_ok"] = False
        return GateResult(
            name="boot_report",
            status="fail",
            summary=failure_msg,
            details=details,
            next_action="python tools/run_browser_boot.py --preset start-screen --capture-frames",
        )

    details["reportPresent"] = True
    try:
        raw_text = report_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        failure_msg = f"{report_label}: {type(exc).__name__}"
        details["failures"] = [failure_msg]
        if min_fps is not None:
            details["min_fps_ok"] = False
        if require_non_black:
            details["non_black_ok"] = False
        return GateResult(
            name="boot_report",
            status="fail",
            summary=failure_msg,
            details=details,
            next_action="python tools/run_browser_boot.py --preset start-screen --capture-frames",
        )

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        failure_msg = f"{report_label}: {type(exc).__name__}"
        details["failures"] = [failure_msg]
        if min_fps is not None:
            details["min_fps_ok"] = False
        if require_non_black:
            details["non_black_ok"] = False
        return GateResult(
            name="boot_report",
            status="fail",
            summary=failure_msg,
            details=details,
            next_action="python tools/run_browser_boot.py --preset start-screen --capture-frames",
        )

    if not isinstance(doc, dict):
        failure_msg = f"{report_label}: expected JSON object"
        details["failures"] = [failure_msg]
        if min_fps is not None:
            details["min_fps_ok"] = False
        if require_non_black:
            details["non_black_ok"] = False
        return GateResult(
            name="boot_report",
            status="fail",
            summary=failure_msg,
            details=details,
            next_action="python tools/run_browser_boot.py --preset start-screen --capture-frames",
        )

    details["reportValid"] = True
    metrics = extract_boot_report_metrics(doc)
    details.update(
        {
            "present_fps_avg": metrics["present_fps_avg"],
            "present_fps_source": metrics["present_fps_source"],
            "presents": metrics["presents_count"],
            "first_present_ms": metrics["first_present_ms"],
            "last_present_ms": metrics["last_present_ms"],
            "present_span_ms": metrics["present_span_ms"],
            "nonBlackCanvasFrames": metrics["non_black_canvas_frames"],
            "capturedFrames": metrics["captured_frames"],
            "first_visible_frame_s": metrics["first_visible_frame_s"],
        }
    )

    failures: list[str] = []
    if min_fps is not None:
        raw_fps = metrics["present_fps_avg_raw"]
        if raw_fps is None:
            details["min_fps_ok"] = False
            if metrics["presents_count"] == 0:
                failures.append(f"0 presents recorded (required --min-fps {min_fps:.2f})")
            else:
                failures.append(
                    f"present_fps_avg unavailable (required --min-fps {min_fps:.2f})"
                )
        elif raw_fps < min_fps:
            details["min_fps_ok"] = False
            failures.append(
                f"present_fps_avg {metrics['present_fps_avg']:.2f} fps < "
                f"--min-fps {min_fps:.2f} fps"
            )
        else:
            details["min_fps_ok"] = True

    if require_non_black:
        has_non_black = bool(metrics["has_non_black_frame"])
        details["non_black_ok"] = has_non_black
        if not has_non_black:
            nb = metrics["non_black_canvas_frames"]
            nb_desc = (
                f"nonBlackCanvasFrames={nb}" if nb is not None else "nonBlackCanvasFrames=missing"
            )
            failures.append(
                f"all-black boot capture ({nb_desc}, first_visible_frame_s=null)"
            )

    details["failures"] = failures
    if failures:
        return GateResult(
            name="boot_report",
            status="fail",
            summary="; ".join(failures),
            details=details,
            next_action="python tools/run_browser_boot.py --preset start-screen --capture-frames",
        )

    parts: list[str] = []
    if metrics["present_fps_avg"] is not None:
        parts.append(
            f"fps={metrics['present_fps_avg']:.2f} ({metrics['present_fps_source']})"
        )
    else:
        parts.append("fps=n/a")
    if metrics["non_black_canvas_frames"] is not None:
        parts.append(f"nonBlackCanvasFrames={metrics['non_black_canvas_frames']}")
    if metrics["first_visible_frame_s"] is not None:
        parts.append(f"first_visible_frame_s={metrics['first_visible_frame_s']}")
    return GateResult(
        name="boot_report",
        status="pass",
        summary=f"{report_label}: " + ", ".join(parts),
        details=details,
        next_action=None,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _gate_passed(gate: GateResult, *, allow_warn: bool = False) -> bool:
    if gate.status == "pass":
        return True
    if allow_warn and gate.status == "warn":
        return True
    return False


def compute_readiness(gates: Mapping[str, GateResult]) -> dict[str, Any]:
    """Compute splash / real-guest readiness booleans and ordered next steps."""
    game = gates["game_assets"]
    imports = gates["import_coverage"]
    recomp = gates["recomp_artifacts"]
    shaders = gates["shader_pack"]
    wasm = gates["wasm64_runtime"]
    server = gates["local_server"]

    wasm_kind = (wasm.details or {}).get("kind")
    splash_ready = bool(
        _gate_passed(game, allow_warn=True)
        and game.status != "skip"
        and _gate_passed(shaders, allow_warn=True)
        and _gate_passed(wasm, allow_warn=True)
        and (server.status in ("pass", "skip"))
    )

    real_boot_ready = bool(
        splash_ready
        and _gate_passed(imports)
        and _gate_passed(recomp, allow_warn=True)
        and wasm_kind == "linked_guest_runtime"
        and _gate_passed(wasm)
        and server.status in ("pass", "skip")
    )

    next_steps: list[str] = []
    for name in GATE_NAMES:
        gate = gates[name]
        if gate.status in ("fail", "warn", "skip") and gate.next_action:
            if gate.next_action not in next_steps:
                next_steps.append(gate.next_action)

    boot_report_gate = gates.get("boot_report")
    if boot_report_gate is not None:
        if boot_report_gate.status in ("fail", "warn", "skip") and boot_report_gate.next_action:
            if boot_report_gate.next_action not in next_steps:
                next_steps.append(boot_report_gate.next_action)

    if splash_ready and not real_boot_ready:
        if wasm_kind == "smoke_shell":
            cmd = "python tools/build_guest_wasm.py work/recomp_build --verify"
            if cmd not in next_steps:
                next_steps.append(cmd)
        if not _gate_passed(recomp, allow_warn=True):
            cmd = "cargo run -p playgta6-recomp -- build-program --help"
            if cmd not in next_steps:
                next_steps.append(cmd)

    if real_boot_ready and (boot_report_gate is None or boot_report_gate.status == "pass"):
        cmd = "python server/serve_local.py"
        if cmd not in next_steps:
            next_steps.append(cmd)

    return {
        "ready_for_browser_splash": splash_ready,
        "ready_for_real_guest_boot": real_boot_ready,
        "next_steps": next_steps,
        "wasm_kind": wasm_kind,
    }


def run_verification(
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    *,
    recomp_dir: Path | str | None = None,
    skip_server: bool = False,
    env: Mapping[str, str] | None = None,
    import_coverage_fn: Callable[[Path], Mapping[str, Any]] | None = None,
    boot_report: Path | str | None = None,
    min_fps: float | None = None,
    require_non_black: bool = False,
) -> dict[str, Any]:
    """Run all six browser-boot gates (plus optional boot-report checks) and return a sanitized JSON-safe report."""
    root = Path(repo_root).resolve()
    resolved_env = _resolve_env_paths(root, env=env)
    data_path: Path = resolved_env["data_path"]
    game_path: Path | None = resolved_env["game_path"]
    work_path: Path = resolved_env["work_path"]

    if recomp_dir is not None:
        recomp_path = Path(recomp_dir)
        if not recomp_path.is_absolute():
            recomp_path = root / recomp_path
    else:
        recomp_path = work_path / "recomp_build"

    gates_list = [
        check_game_assets_gate(root, env=env),
        check_import_coverage_gate(root, audit_fn=import_coverage_fn),
        check_recomp_artifacts_gate(root, recomp_dir=recomp_path),
        check_shader_pack_gate(root, data_dir=data_path),
        check_wasm64_runtime_gate(root),
        check_local_server_gate(
            root,
            data_dir=data_path,
            game_dir=game_path if (game_path is not None and game_path.is_dir()) else None,
            recomp_dir=recomp_path,
            skip_server=skip_server,
        ),
    ]
    boot_report_requested = (
        boot_report is not None or min_fps is not None or bool(require_non_black)
    )
    if boot_report_requested:
        gates_list.append(
            check_boot_report_gate(
                root,
                boot_report=boot_report,
                work_dir=work_path,
                recomp_dir=recomp_path,
                min_fps=min_fps,
                require_non_black=bool(require_non_black),
            )
        )

    gates = {gate.name: gate for gate in gates_list}
    readiness = compute_readiness(gates)
    gates_payload = {name: gates[name].to_dict() for name in GATE_NAMES}
    overall_ok = bool(readiness["ready_for_browser_splash"])

    payload: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "ok": overall_ok,
        "ready_for_browser_splash": readiness["ready_for_browser_splash"],
        "ready_for_real_guest_boot": readiness["ready_for_real_guest_boot"],
        "gates": gates_payload,
        "next_steps": readiness["next_steps"],
    }
    if boot_report_requested:
        br_gate = gates["boot_report"]
        br_ok = br_gate.status == "pass"
        gates_payload["boot_report"] = br_gate.to_dict()
        payload["boot_report_ok"] = br_ok
        payload["ok"] = bool(overall_ok and br_ok)

    return sanitize_value(payload)


def format_report_text(report: Mapping[str, Any]) -> str:
    """Render a concise terminal checklist for the verification gates."""
    lines: list[str] = [
        "=== playgta6web Browser Real-Boot Readiness ===",
        "",
        "Gates:",
    ]
    status_tag = {
        "pass": "[PASS]",
        "warn": "[WARN]",
        "fail": "[FAIL]",
        "skip": "[SKIP]",
    }
    gates = report.get("gates") or {}
    ordered_names = list(GATE_NAMES)
    for extra_name in gates.keys():
        if extra_name not in ordered_names:
            ordered_names.append(extra_name)
    for name in ordered_names:
        gate = gates.get(name) or {}
        status = gate.get("status", "fail")
        tag = status_tag.get(status, "[????]")
        summary = gate.get("summary") or ""
        lines.append(f"  {tag} {name:<18} {summary}")

    lines.append("")
    lines.append("Readiness:")
    lines.append(
        f"  ready_for_browser_splash : "
        f"{'YES' if report.get('ready_for_browser_splash') else 'NO'}"
    )
    lines.append(
        f"  ready_for_real_guest_boot: "
        f"{'YES' if report.get('ready_for_real_guest_boot') else 'NO'}"
    )
    if "boot_report_ok" in report:
        lines.append(
            f"  boot_report_ok           : "
            f"{'YES' if report.get('boot_report_ok') else 'NO'}"
        )

    next_steps = report.get("next_steps") or []
    if next_steps:
        lines.append("")
        lines.append("Next Steps:")
        for step in next_steps:
            lines.append(f"  $ {step}")

    return "\n".join(lines)


def _parse_min_fps(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --min-fps value: {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError("--min-fps must be a finite non-negative number")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify end-to-end browser real-boot readiness across game assets, "
            "import coverage, recomp artifacts, shader packs, Wasm64 runtime, "
            "and the local COOP/COEP server."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the playgta6web repository root (defaults to parent of tools/).",
    )
    parser.add_argument(
        "--recomp-dir",
        type=Path,
        default=None,
        help="Recomp build directory (default: <work>/recomp_build or work/recomp_build).",
    )
    parser.add_argument(
        "--skip-server",
        action="store_true",
        help="Skip the ephemeral local COOP/COEP server self-check.",
    )
    parser.add_argument(
        "--boot-report",
        type=Path,
        default=None,
        help=(
            "Optional path to a boot-report.json file (or run directory containing it) "
            "for --min-fps / --require-non-black checks."
        ),
    )
    parser.add_argument(
        "--min-fps",
        type=_parse_min_fps,
        default=None,
        metavar="FPS",
        help=(
            "Require boot report present_fps_avg (or computed presents / "
            "(last_present_ms - first_present_ms)) to be >= FPS."
        ),
    )
    parser.add_argument(
        "--require-non-black",
        action="store_true",
        help=(
            "Require boot report nonBlackCanvasFrames > 0 (or first_visible_frame_s "
            "is not null) so all-black captures fail immediately."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit sanitized machine-readable JSON (never includes host paths or usernames).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = run_verification(
        repo_root=args.repo_root,
        recomp_dir=args.recomp_dir,
        skip_server=args.skip_server,
        boot_report=args.boot_report,
        min_fps=args.min_fps,
        require_non_black=args.require_non_black,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))

    if report.get("boot_report_ok") is False:
        return 1
    if report.get("ready_for_real_guest_boot"):
        return 0
    if report.get("ready_for_browser_splash"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
