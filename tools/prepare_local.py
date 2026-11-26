#!/usr/bin/env python3
"""Single-command local asset pipeline orchestrator for playgta6web (Track 5 / W23).

Reads `.env` / `GTA6_*` environment variables, validates repository safety invariants,
builds an inspectable execution plan, and chains the local Rust asset/recomp/shader
CLIs into gitignored `local-data/` and `work/` directories:

  1. scan     -> playgta6-manifest scan --dir <game_dir> --json
  2. manifest -> playgta6-manifest manifest --game <game_dir> --out <data_dir>/data-manifest.json [--skip-incomplete]
  3. bootset  -> playgta6-manifest bootset --game <game_dir> --out <data_dir>/bootset.json [--skip-incomplete]
  4. recomp   -> playgta6-recomp inspect --from-env --markdown <work_dir>/imports_required.md
  5. shaders  -> playgta6-shaders convert --spirv <work_dir>/spirv --out <data_dir>/shaders (if .spv files exist)

Security & privacy invariants:
- Refuses to write generated outputs anywhere outside gitignored `local-data/` or `work/`.
- Machine-readable `--json` output NEVER leaks raw host filesystem paths or usernames.
- Strictly uses Python 3.12 standard library modules only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1

PIPELINE_STEPS: tuple[str, ...] = ("scan", "manifest", "bootset", "recomp", "shaders")
STEP_CHOICES: tuple[str, ...] = (*PIPELINE_STEPS, "all")
PACK_VERSION_CHOICES: tuple[str, ...] = ("auto", "1", "2")
ALLOWED_OUTPUT_DIR_NAMES: tuple[str, ...] = ("local-data", "work")

# WGPK binary pack format constants (matching shaders/src/pack.rs byte-for-byte)
WGPK_MAGIC = b"WGPK"
WGPK_VERSION_V1 = 1
WGPK_VERSION_V2 = 2
WGPK_SUPPORTED_VERSIONS: tuple[int, ...] = (WGPK_VERSION_V1, WGPK_VERSION_V2)
WGPK_HEADER_STRUCT = struct.Struct("<4sIII")
WGPK_HEADER_BYTES = WGPK_HEADER_STRUCT.size  # 16 bytes
WGPK_ENTRY_STRUCT_V1 = struct.Struct("<QB3sII")
WGPK_ENTRY_BYTES_V1 = WGPK_ENTRY_STRUCT_V1.size  # 20 bytes
WGPK_ENTRY_STRUCT_V2 = struct.Struct("<QB3sIQ")
WGPK_ENTRY_BYTES_V2 = WGPK_ENTRY_STRUCT_V2.size  # 24 bytes
WGPK_VALID_STAGES: frozenset[int] = frozenset({1, 2, 3})
WGPK_MAX_U32 = 0xFFFFFFFF
WGPK_MAX_U64 = 0xFFFFFFFFFFFFFFFF
WGPK_TOC_CHUNK_ENTRIES = 4096
WGPK_MAX_ENTRY_ERRORS = 8
_COPY_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming copy buffer for >4 GiB packs

# Reflection sidecar v1/v2/v3 constants (matching tools/stage_reflection_sidecar.py)
REFLECT_SIDECAR_SCHEMA = "playgta6-stage-reflection"
REFLECT_SIDECAR_SUPPORTED_VERSIONS: tuple[int, ...] = (1, 2, 3)
REFLECT_KIND_MAP: dict[str, str] = {
    "uniform": "uniform",
    "storage": "storage",
    "read_only_storage": "read_only_storage",
    "read-only-storage": "read_only_storage",
    "texture": "texture",
    "texture_2d": "texture",
    "storage_texture": "storage_texture",
    "storage-texture": "storage_texture",
    "sampler": "sampler",
    "comparison_sampler": "comparison_sampler",
    "sampler_comparison": "comparison_sampler",
}
REFLECT_READ_ONLY_KINDS: frozenset[str] = frozenset(
    {"uniform", "read_only_storage", "texture", "sampler", "comparison_sampler"}
)
_SAFE_UNRESOLVED_LABEL_RE = re.compile(r"^[A-Za-z0-9_.$<>\[\]-]{1,64}$")
_SAFE_PACK_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_MESH_VERTICES = 256
MAX_MESH_PRIMITIVES = 256
MAX_MESH_VERTEX_WORDS = 64

SET_GAME_HINT = 'python tools/local_env.py --set-game "<path>"'
MAX_OUTPUT_CHARS = 4000

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)
_PLACEHOLDER_SUFFIX_RE = re.compile(
    r"(<(?:GTA6_GAME_DIR|GTA6_DATA_DIR|GTA6_WORK_DIR|REPO_ROOT|HOME)>)(?:[\\/][^\s\"'<>|,;()\[\]]*)"
)


class PipelineValidationError(ValueError):
    """Raised when configuration or step selection fails pre-execution validation."""

    def __init__(self, message: str, *, code: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


class PipelineSafetyError(PipelineValidationError):
    """Raised when a configured or planned output path escapes gitignored local-data/ or work/."""

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message, code="unsafe_output_path", hint=hint)


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved local pipeline configuration derived from `.env` and environment variables."""

    repo_root: Path
    game_dir: Path | None
    data_dir: Path
    work_dir: Path
    raw_game_dir: str = ""
    raw_data_dir: str = ""
    raw_work_dir: str = ""
    env_file_exists: bool = False
    raw_repo_root: Path | None = None


@dataclass(frozen=True)
class PlannedStep:
    """One step in the local asset preparation execution plan."""

    name: str
    description: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...] = field(default_factory=tuple)
    skipped: str | None = None


@dataclass(frozen=True)
class CommandResult:
    """Outcome returned by a step runner callback."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[list[str], Path, dict[str, str]], Any]


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


def _resolve_setting_path(raw_value: str, repo_root: Path) -> Path:
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate.resolve()


def resolve_config(
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
) -> PipelineConfig:
    """Resolve `GTA6_GAME_DIR`, `GTA6_DATA_DIR`, and `GTA6_WORK_DIR` from env or `.env`."""
    raw_root = Path(repo_root)
    resolved_root = raw_root.resolve()
    env_map = os.environ if env is None else env
    env_file = resolved_root / ".env"
    dotenv_vals = load_dotenv(env_file)

    raw_game = (env_map.get("GTA6_GAME_DIR") or dotenv_vals.get("GTA6_GAME_DIR", "")).strip()
    raw_data = (env_map.get("GTA6_DATA_DIR") or dotenv_vals.get("GTA6_DATA_DIR", "")).strip()
    raw_work = (env_map.get("GTA6_WORK_DIR") or dotenv_vals.get("GTA6_WORK_DIR", "")).strip()

    game_dir = _resolve_setting_path(raw_game, resolved_root) if raw_game else None
    data_dir = (
        _resolve_setting_path(raw_data, resolved_root)
        if raw_data
        else (resolved_root / "local-data").resolve()
    )
    work_dir = (
        _resolve_setting_path(raw_work, resolved_root)
        if raw_work
        else (resolved_root / "work").resolve()
    )

    return PipelineConfig(
        repo_root=resolved_root,
        game_dir=game_dir,
        data_dir=data_dir,
        work_dir=work_dir,
        raw_game_dir=raw_game,
        raw_data_dir=raw_data,
        raw_work_dir=raw_work,
        env_file_exists=env_file.is_file(),
        raw_repo_root=raw_root,
    )


def _allowed_output_roots(repo_root: Path) -> tuple[Path, ...]:
    resolved_root = repo_root.resolve()
    roots: list[Path] = []
    for name in ALLOWED_OUTPUT_DIR_NAMES:
        candidate = (resolved_root / name).resolve()
        if candidate.is_relative_to(resolved_root):
            roots.append(candidate)
    return tuple(roots)


def is_safe_output_path(path: Path, repo_root: Path) -> bool:
    """Return True iff `path` resolves inside gitignored `<repo_root>/local-data` or `<repo_root>/work`."""
    resolved_target = path.resolve()
    return any(resolved_target.is_relative_to(root) for root in _allowed_output_roots(repo_root))


def validate_output_path(path: Path, repo_root: Path, *, label: str = "output path") -> Path:
    """Ensure `path` resolves strictly inside gitignored `local-data/` or `work/`."""
    resolved = path.resolve()
    if not is_safe_output_path(resolved, repo_root):
        raise PipelineSafetyError(
            f"Refusing to write generated outputs to {label} outside gitignored local-data/ or work/.",
            hint="Set GTA6_DATA_DIR=local-data and GTA6_WORK_DIR=work (or subdirectories inside them).",
        )
    return resolved


def validate_config(config: PipelineConfig) -> None:
    """Validate safety invariants and `GTA6_GAME_DIR` before building or running any steps."""
    validate_output_path(config.data_dir, config.repo_root, label="GTA6_DATA_DIR")
    validate_output_path(config.work_dir, config.repo_root, label="GTA6_WORK_DIR")

    if config.game_dir is None or not config.raw_game_dir:
        raise PipelineValidationError(
            f"GTA6_GAME_DIR is not set in environment or .env. Run: {SET_GAME_HINT}",
            code="missing_game_dir",
            hint=SET_GAME_HINT,
        )

    if not config.game_dir.is_dir():
        raise PipelineValidationError(
            f"GTA6_GAME_DIR does not exist or is not a directory (<GTA6_GAME_DIR>). Run: {SET_GAME_HINT}",
            code="invalid_game_dir",
            hint=SET_GAME_HINT,
        )


def _find_spirv_files(spirv_dir: Path) -> list[Path]:
    """Recursively locate `.spv` files inside `spirv_dir` (case-insensitive, sorted)."""
    if not spirv_dir.is_dir():
        return []
    found: list[Path] = []
    try:
        for root_str, _dirs, files in os.walk(spirv_dir):
            root_path = Path(root_str)
            for fname in files:
                if fname.lower().endswith(".spv"):
                    found.append(root_path / fname)
    except OSError:
        return []
    found.sort()
    return found


def build_pipeline_plan(
    config: PipelineConfig,
    *,
    step: str = "all",
    skip_incomplete: bool = False,
    pack_version: str = "auto",
    pack_shaders: bool = False,
) -> list[PlannedStep]:
    """Build the validated, inspectable execution plan for the requested pipeline step(s)."""
    if step not in STEP_CHOICES:
        raise PipelineValidationError(
            f"Unknown step '{step}'. Expected one of: {', '.join(STEP_CHOICES)}.",
            code="invalid_step",
        )
    if pack_version not in PACK_VERSION_CHOICES:
        raise PipelineValidationError(
            f"Unknown pack_version '{pack_version}'. Expected one of: {', '.join(PACK_VERSION_CHOICES)}.",
            code="invalid_pack_version",
        )

    validate_config(config)
    assert config.game_dir is not None  # guaranteed by validate_config

    selected = PIPELINE_STEPS if step == "all" else (step,)
    plan: list[PlannedStep] = []

    for step_name in selected:
        if step_name == "scan":
            plan.append(
                PlannedStep(
                    name="scan",
                    description="Scan local game directory and classify files",
                    command=(
                        "cargo",
                        "run",
                        "-p",
                        "playgta6-tools",
                        "--bin",
                        "playgta6-manifest",
                        "--",
                        "scan",
                        "--dir",
                        str(config.game_dir),
                        "--json",
                    ),
                )
            )
        elif step_name == "manifest":
            manifest_out = validate_output_path(
                config.data_dir / "data-manifest.json",
                config.repo_root,
                label="manifest output",
            )
            cmd = [
                "cargo",
                "run",
                "-p",
                "playgta6-tools",
                "--bin",
                "playgta6-manifest",
                "--",
                "manifest",
                "--game",
                str(config.game_dir),
                "--out",
                str(manifest_out),
            ]
            if skip_incomplete:
                cmd.append("--skip-incomplete")
            plan.append(
                PlannedStep(
                    name="manifest",
                    description="Generate data-manifest.json in local-data/",
                    command=tuple(cmd),
                    outputs=(manifest_out,),
                )
            )
        elif step_name == "bootset":
            bootset_out = validate_output_path(
                config.data_dir / "bootset.json",
                config.repo_root,
                label="bootset output",
            )
            cmd = [
                "cargo",
                "run",
                "-p",
                "playgta6-tools",
                "--bin",
                "playgta6-manifest",
                "--",
                "bootset",
                "--game",
                str(config.game_dir),
                "--out",
                str(bootset_out),
            ]
            if skip_incomplete:
                cmd.append("--skip-incomplete")
            plan.append(
                PlannedStep(
                    name="bootset",
                    description="Generate bootset.json prefetch header list in local-data/",
                    command=tuple(cmd),
                    outputs=(bootset_out,),
                )
            )
        elif step_name == "recomp":
            imports_out = validate_output_path(
                config.work_dir / "imports_required.md",
                config.repo_root,
                label="recomp markdown output",
            )
            plan.append(
                PlannedStep(
                    name="recomp",
                    description="Inspect PE64 imports into gitignored work/imports_required.md",
                    command=(
                        "cargo",
                        "run",
                        "-p",
                        "playgta6-recomp",
                        "--",
                        "inspect",
                        "--from-env",
                        "--markdown",
                        str(imports_out),
                    ),
                    outputs=(imports_out,),
                )
            )
        elif step_name == "shaders":
            spirv_dir = config.work_dir / "spirv"
            shaders_out = validate_output_path(
                config.data_dir / "shaders",
                config.repo_root,
                label="shaders output",
            )
            if not spirv_dir.is_dir():
                plan.append(
                    PlannedStep(
                        name="shaders",
                        description="Convert local SPIR-V shaders into WGSL in local-data/shaders/",
                        command=(),
                        outputs=(),
                        skipped="no local SPIR-V directory found",
                    )
                )
            elif not _find_spirv_files(spirv_dir):
                plan.append(
                    PlannedStep(
                        name="shaders",
                        description="Convert local SPIR-V shaders into WGSL in local-data/shaders/",
                        command=(),
                        outputs=(),
                        skipped="no .spv files found in local SPIR-V directory",
                    )
                )
            else:
                shader_cmd = [
                    "cargo",
                    "run",
                    "-p",
                    "playgta6-shaders",
                    "--",
                    "convert",
                    "--spirv",
                    str(spirv_dir),
                    "--out",
                    str(shaders_out),
                ]
                shader_outputs: list[Path] = [shaders_out]
                if pack_shaders or pack_version != "auto":
                    pack_out = validate_output_path(
                        shaders_out / "shaders.wgsl.pack",
                        config.repo_root,
                        label="shader pack output",
                    )
                    shader_cmd.extend(["--pack", str(pack_out), "--pack-version", pack_version])
                    shader_outputs.append(pack_out)
                plan.append(
                    PlannedStep(
                        name="shaders",
                        description="Convert local SPIR-V shaders into WGSL in local-data/shaders/",
                        command=tuple(shader_cmd),
                        outputs=tuple(shader_outputs),
                    )
                )

    return plan


def _current_usernames(env: Mapping[str, str] | None = None) -> list[str]:
    env_map = os.environ if env is None else env
    names: list[str] = []
    for candidate in (
        env_map.get("USERNAME", ""),
        env_map.get("USER", ""),
        env_map.get("LOGNAME", ""),
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


def _path_variants(path: Path | str) -> list[str]:
    raw = str(path).rstrip("\\/")
    if not raw:
        return []
    variants = [raw]
    posix = raw.replace("\\", "/")
    if posix not in variants:
        variants.append(posix)
    win = raw.replace("/", "\\")
    if win not in variants:
        variants.append(win)
    return variants


def _build_redaction_rules(config: PipelineConfig) -> list[tuple[re.Pattern[str], str]]:
    pairs: list[tuple[str, str]] = []

    def _add(path_val: Path | str | None, placeholder: str) -> None:
        if not path_val:
            return
        for variant in _path_variants(path_val):
            # Only match absolute or multi-char configured paths to avoid clobbering relative words.
            if len(variant) >= 2:
                pairs.append((variant, placeholder))

    if config.game_dir is not None:
        _add(config.game_dir, "<GTA6_GAME_DIR>")
    if config.raw_game_dir and Path(config.raw_game_dir).is_absolute():
        _add(config.raw_game_dir, "<GTA6_GAME_DIR>")

    _add(config.data_dir, "<GTA6_DATA_DIR>")
    if config.raw_data_dir and Path(config.raw_data_dir).is_absolute():
        _add(config.raw_data_dir, "<GTA6_DATA_DIR>")

    _add(config.work_dir, "<GTA6_WORK_DIR>")
    if config.raw_work_dir and Path(config.raw_work_dir).is_absolute():
        _add(config.raw_work_dir, "<GTA6_WORK_DIR>")

    _add(config.repo_root, "<REPO_ROOT>")
    if config.raw_repo_root is not None and config.raw_repo_root.is_absolute():
        _add(config.raw_repo_root, "<REPO_ROOT>")

    try:
        home_dir = Path.home()
        _add(home_dir, "<HOME>")
        _add(home_dir.resolve(), "<HOME>")
    except (OSError, RuntimeError):
        pass

    # Replace longest prefixes first so `<repo>/local-data` matches before `<repo>`.
    seen: set[tuple[str, str]] = set()
    unique_pairs: list[tuple[str, str]] = []
    for needle, token in sorted(pairs, key=lambda item: len(item[0]), reverse=True):
        key = (needle.lower(), token)
        if key not in seen:
            seen.add(key)
            unique_pairs.append((needle, token))

    flags = re.IGNORECASE if os.name == "nt" else 0
    return [(re.compile(re.escape(needle), flags), token) for needle, token in unique_pairs]


def sanitize_text(
    text: str,
    config: PipelineConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Redact configured roots, stray host filesystem paths, and usernames from text."""
    if not text:
        return text

    sanitized = text
    for pattern, token in _build_redaction_rules(config):
        sanitized = pattern.sub(token, sanitized)

    # Normalize backslashes in relative suffixes right after known placeholders.
    sanitized = _PLACEHOLDER_SUFFIX_RE.sub(lambda m: m.group(0).replace("\\", "/"), sanitized)

    # Strip any remaining raw host filesystem paths.
    sanitized = _WINDOWS_PATH_RE.sub("<HOST_PATH>", sanitized)
    sanitized = _UNC_PATH_RE.sub("<HOST_PATH>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<HOST_PATH>", sanitized)

    for username in _current_usernames(env=env):
        user_re = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        sanitized = user_re.sub("<USER>", sanitized)

    return sanitized


def sanitize_path(
    path: Path | str,
    config: PipelineConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return a host-independent redacted representation of `path`."""
    return sanitize_text(str(path), config, env=env)


def sanitize_command(
    command: Sequence[str],
    config: PipelineConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> list[str]:
    """Return a command argv list with all host paths and usernames redacted."""
    return [sanitize_text(arg, config, env=env) for arg in command]


def _truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 16].rstrip() + "\n... [truncated]"


def _sanitize_payload(
    value: Any,
    config: PipelineConfig,
    *,
    env: Mapping[str, str] | None = None,
) -> Any:
    if isinstance(value, str):
        return sanitize_text(value, config, env=env)
    if isinstance(value, list):
        return [_sanitize_payload(item, config, env=env) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_payload(item, config, env=env) for item in value]
    if isinstance(value, dict):
        safe_enum_keys = {"name", "status", "step", "errorCode"}
        return {
            key: (
                val
                if key in safe_enum_keys and isinstance(val, str)
                else _sanitize_payload(val, config, env=env)
            )
            for key, val in value.items()
        }
    return value


def _normalize_command_result(raw: Any) -> CommandResult:
    if isinstance(raw, CommandResult):
        return raw
    if isinstance(raw, subprocess.CompletedProcess):
        return CommandResult(
            returncode=int(raw.returncode),
            stdout=str(raw.stdout or ""),
            stderr=str(raw.stderr or ""),
        )
    if isinstance(raw, Mapping):
        return CommandResult(
            returncode=int(raw.get("returncode", 0)),
            stdout=str(raw.get("stdout", "") or ""),
            stderr=str(raw.get("stderr", "") or ""),
        )
    if isinstance(raw, int):
        return CommandResult(returncode=raw)
    raise TypeError(f"Unsupported runner return type: {type(raw).__name__}")


def default_runner(cmd: list[str], cwd: Path, env: dict[str, str]) -> CommandResult:
    """Execute one pipeline command as a subprocess with stdin detached."""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    return CommandResult(
        returncode=proc.returncode,
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
    )


def _build_subprocess_env(
    config: PipelineConfig,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    sub_env = dict(os.environ if base_env is None else base_env)
    if config.game_dir is not None:
        sub_env["GTA6_GAME_DIR"] = str(config.game_dir)
    sub_env["GTA6_DATA_DIR"] = str(config.data_dir)
    sub_env["GTA6_WORK_DIR"] = str(config.work_dir)
    return sub_env


def _config_summary(config: PipelineConfig, *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    return {
        "envFileExists": config.env_file_exists,
        "gameDirConfigured": bool(config.raw_game_dir),
        "gameDirExists": bool(config.game_dir is not None and config.game_dir.is_dir()),
        "gameDir": "<GTA6_GAME_DIR>" if config.game_dir is not None else None,
        "dataDir": sanitize_path(config.data_dir, config, env=env),
        "workDir": sanitize_path(config.work_dir, config, env=env),
    }


def get_file_size(path: Path) -> int:
    """Return the byte size of `path` (isolated helper for clean unit-test mocking)."""
    return int(path.stat().st_size)


def _is_valid_uint32(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF


def _safe_basename(path: Path | str) -> str:
    raw = str(path).strip().rstrip("/\\")
    if not raw:
        return "<unknown>"
    parts = re.split(r"[/\\]+", raw)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return base


def _sanitize_standalone_text(raw: str) -> str:
    """Redact host filesystem paths and active OS usernames without requiring a PipelineConfig."""
    if not raw:
        return raw
    text = str(raw)
    text = _UNC_PATH_RE.sub("<HOST_PATH>", text)
    text = text.replace("\\", "/")
    text = _WINDOWS_PATH_RE.sub("<HOST_PATH>", text)
    text = _POSIX_PATH_RE.sub("<HOST_PATH>", text)
    for username in _current_usernames():
        user_re = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        text = user_re.sub("<USER>", text)
    return text


def validate_shader_pack_header(pack_path: Path) -> dict[str, Any]:
    """Quick-validate a WGPK v1 (20B TOC) or v2 (24B TOC, >4 GiB) pack without loading payloads."""
    path = Path(pack_path)
    label = _sanitize_standalone_text(_safe_basename(path))
    errors: list[str] = []
    version: int | None = None
    entry_bytes: int | None = None
    entry_count: int | None = None
    size_bytes = 0

    if not path.exists():
        errors.append(f"{label}: shader pack file does not exist")
        return {
            "ok": False,
            "version": None,
            "entry_bytes": None,
            "entry_count": None,
            "size_bytes": 0,
            "is_v2": False,
            "errors": errors,
        }
    if not path.is_file():
        errors.append(f"{label}: shader pack path is not a regular file")
        return {
            "ok": False,
            "version": None,
            "entry_bytes": None,
            "entry_count": None,
            "size_bytes": 0,
            "is_v2": False,
            "errors": errors,
        }

    try:
        size_bytes = get_file_size(path)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        errors.append(_sanitize_standalone_text(f"{label}: failed to stat shader pack: {reason}"))
        return {
            "ok": False,
            "version": None,
            "entry_bytes": None,
            "entry_count": None,
            "size_bytes": 0,
            "is_v2": False,
            "errors": errors,
        }

    if size_bytes < WGPK_HEADER_BYTES:
        errors.append(
            f"{label}: truncated header ({size_bytes} bytes, minimum {WGPK_HEADER_BYTES} bytes)"
        )
        return {
            "ok": False,
            "version": None,
            "entry_bytes": None,
            "entry_count": None,
            "size_bytes": size_bytes,
            "is_v2": False,
            "errors": errors,
        }

    try:
        with path.open("rb") as fh:
            header = fh.read(WGPK_HEADER_BYTES)
            if len(header) < WGPK_HEADER_BYTES:
                errors.append(f"{label}: truncated while reading WGPK header")
            else:
                magic, ver, count, _flags = WGPK_HEADER_STRUCT.unpack_from(header, 0)
                if magic != WGPK_MAGIC:
                    errors.append(f"{label}: invalid WGPK magic (expected 'WGPK')")
                else:
                    version = int(ver)
                    entry_count = int(count)
                    if ver not in WGPK_SUPPORTED_VERSIONS:
                        errors.append(
                            f"{label}: unsupported WGPK version {ver} (expected 1 or 2)"
                        )
                    else:
                        is_v2 = ver == WGPK_VERSION_V2
                        entry_bytes = WGPK_ENTRY_BYTES_V2 if is_v2 else WGPK_ENTRY_BYTES_V1
                        max_offset_limit = WGPK_MAX_U64 if is_v2 else WGPK_MAX_U32
                        table_bytes = count * entry_bytes
                        data_start = WGPK_HEADER_BYTES + table_bytes
                        if data_start > size_bytes:
                            errors.append(
                                f"{label}: truncated entry table (entry_count={count} "
                                f"requires {data_start} bytes, file is {size_bytes} bytes)"
                            )
                        elif not is_v2 and data_start > WGPK_MAX_U32:
                            errors.append(
                                f"{label}: WGPK v1 entry table overflows uint32 for {count} entries"
                            )
                        else:
                            processed = 0
                            reported = 0
                            suppressed = 0
                            while processed < count:
                                batch = min(WGPK_TOC_CHUNK_ENTRIES, count - processed)
                                chunk_len = batch * entry_bytes
                                chunk = fh.read(chunk_len)
                                if len(chunk) < chunk_len:
                                    errors.append(f"{label}: truncated while reading entry table")
                                    break
                                for rel_idx in range(batch):
                                    idx = processed + rel_idx
                                    base = rel_idx * entry_bytes
                                    if is_v2:
                                        _h, stage_raw, _res, length, offset = (
                                            WGPK_ENTRY_STRUCT_V2.unpack_from(chunk, base)
                                        )
                                    else:
                                        _h, stage_raw, _res, offset, length = (
                                            WGPK_ENTRY_STRUCT_V1.unpack_from(chunk, base)
                                        )
                                    if stage_raw not in WGPK_VALID_STAGES:
                                        if reported < WGPK_MAX_ENTRY_ERRORS:
                                            errors.append(
                                                f"{label}: entry {idx} has invalid shader stage {stage_raw}"
                                            )
                                            reported += 1
                                        else:
                                            suppressed += 1
                                    payload_end = offset + length
                                    if (
                                        offset < data_start
                                        or payload_end > max_offset_limit
                                        or payload_end > size_bytes
                                    ):
                                        if reported < WGPK_MAX_ENTRY_ERRORS:
                                            errors.append(
                                                f"{label}: entry {idx} payload [{offset}..+{length}] "
                                                f"out of bounds for [{data_start}..{size_bytes}]"
                                            )
                                            reported += 1
                                        else:
                                            suppressed += 1
                                processed += batch
                            if suppressed > 0:
                                errors.append(
                                    f"{label}: {suppressed} additional invalid/out-of-bounds entry error(s)"
                                )
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        errors.append(_sanitize_standalone_text(f"{label}: failed to read shader pack: {reason}"))

    return {
        "ok": len(errors) == 0,
        "version": version,
        "entry_bytes": entry_bytes,
        "entry_count": entry_count,
        "size_bytes": size_bytes,
        "is_v2": bool(version == WGPK_VERSION_V2),
        "errors": errors,
    }


def validate_reflect_sidecar_file(sidecar_path: Path) -> dict[str, Any]:
    """Validate a `playgta6-stage-reflection` sidecar (v1, v2, or v3)."""
    path = Path(sidecar_path)
    label = _sanitize_standalone_text(_safe_basename(path))
    errors: list[str] = []
    schema_str: str | None = None
    version_int: int | None = None
    stage_count = 0
    unresolved_source_stages = 0
    unresolved_source_bindings = 0
    mesh_output_stages = 0
    size_bytes = 0

    if not path.exists():
        errors.append(f"{label}: reflection sidecar file does not exist")
        return {
            "ok": False,
            "schema": None,
            "version": None,
            "stage_count": 0,
            "unresolved_source_stages": 0,
            "unresolved_source_bindings": 0,
            "mesh_output_stages": 0,
            "size_bytes": 0,
            "errors": errors,
        }
    if not path.is_file():
        errors.append(f"{label}: reflection sidecar path is not a regular file")
        return {
            "ok": False,
            "schema": None,
            "version": None,
            "stage_count": 0,
            "unresolved_source_stages": 0,
            "unresolved_source_bindings": 0,
            "mesh_output_stages": 0,
            "size_bytes": 0,
            "errors": errors,
        }

    try:
        size_bytes = get_file_size(path)
        raw_text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"{label}: failed to read reflection sidecar ({type(exc).__name__})")
        return {
            "ok": False,
            "schema": None,
            "version": None,
            "stage_count": 0,
            "unresolved_source_stages": 0,
            "unresolved_source_bindings": 0,
            "mesh_output_stages": 0,
            "size_bytes": size_bytes,
            "errors": errors,
        }

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        errors.append(f"{label}: invalid JSON ({type(exc).__name__})")
        return {
            "ok": False,
            "schema": None,
            "version": None,
            "stage_count": 0,
            "unresolved_source_stages": 0,
            "unresolved_source_bindings": 0,
            "mesh_output_stages": 0,
            "size_bytes": size_bytes,
            "errors": errors,
        }

    if not isinstance(doc, Mapping):
        errors.append(f"{label}: top-level JSON value must be an object")
        return {
            "ok": False,
            "schema": None,
            "version": None,
            "stage_count": 0,
            "unresolved_source_stages": 0,
            "unresolved_source_bindings": 0,
            "mesh_output_stages": 0,
            "size_bytes": size_bytes,
            "errors": errors,
        }

    raw_schema = doc.get("schema")
    if isinstance(raw_schema, str) and raw_schema.strip():
        schema_str = _sanitize_standalone_text(raw_schema.strip())
    if schema_str != REFLECT_SIDECAR_SCHEMA:
        errors.append(
            f"{label}: invalid schema '{schema_str or '<missing>'}' "
            f"(expected '{REFLECT_SIDECAR_SCHEMA}')"
        )

    raw_version = doc.get("version")
    if isinstance(raw_version, int) and not isinstance(raw_version, bool):
        version_int = int(raw_version)
    if version_int not in REFLECT_SIDECAR_SUPPORTED_VERSIONS:
        errors.append(
            f"{label}: unsupported version {raw_version!r} (expected 1, 2, or 3)"
        )

    stages = doc.get("stages")
    if not isinstance(stages, Mapping):
        errors.append(f"{label}: missing or invalid 'stages' object")
    else:
        stage_count = len(stages)
        for stage_key, stage_entry in stages.items():
            safe_stage_key = _sanitize_standalone_text(str(stage_key))
            if not isinstance(stage_entry, Mapping):
                errors.append(f"{label}: stage '{safe_stage_key}' must be an object")
                continue

            if "mesh_output" in stage_entry and stage_entry["mesh_output"] is not None:
                mo = stage_entry["mesh_output"]
                mo_ok = (
                    isinstance(mo, Mapping)
                    and _is_valid_uint32(mo.get("max_vertices"))
                    and 1 <= mo["max_vertices"] <= MAX_MESH_VERTICES
                    and _is_valid_uint32(mo.get("max_primitives"))
                    and 1 <= mo["max_primitives"] <= MAX_MESH_PRIMITIVES
                    and _is_valid_uint32(mo.get("vertex_words"))
                    and 4 <= mo["vertex_words"] <= MAX_MESH_VERTEX_WORDS
                    and mo["vertex_words"] % 4 == 0
                )
                if mo_ok:
                    mesh_output_stages += 1
                else:
                    errors.append(f"{label}: stage '{safe_stage_key}' has malformed mesh_output")

            bindings = stage_entry.get("bindings")
            if bindings is None:
                continue
            if not isinstance(bindings, list):
                errors.append(f"{label}: stage '{safe_stage_key}' 'bindings' must be a list")
                continue

            stage_has_unresolved = False
            for b_idx, binding in enumerate(bindings):
                if not isinstance(binding, Mapping):
                    continue
                source = binding.get("source")
                if not isinstance(source, Mapping):
                    continue

                if "inline_sampler" in source:
                    sampler = source["inline_sampler"]
                    words = sampler.get("sdwords") if isinstance(sampler, Mapping) else None
                    if not (
                        isinstance(words, list)
                        and len(words) == 4
                        and all(_is_valid_uint32(w) for w in words)
                    ):
                        errors.append(
                            f"{label}: stage '{safe_stage_key}' binding[{b_idx}] "
                            f"has malformed inline_sampler"
                        )

                if source.get("root_descriptor") is True and "table_byte_offset" in source:
                    tbo = source["table_byte_offset"]
                    if not (_is_valid_uint32(tbo) and tbo % 32 == 0):
                        errors.append(
                            f"{label}: stage '{safe_stage_key}' binding[{b_idx}] "
                            f"has malformed v2 root_descriptor table_byte_offset"
                        )

                if "unresolved" in source:
                    mark = source["unresolved"]
                    mark_ok = mark is True or (
                        isinstance(mark, str)
                        and bool(mark.strip())
                        and _SAFE_UNRESOLVED_LABEL_RE.match(mark) is not None
                    )
                    raw_kind = binding.get("kind")
                    norm_kind = (
                        REFLECT_KIND_MAP.get(raw_kind.strip().lower())
                        if isinstance(raw_kind, str)
                        else None
                    )
                    if len(source) != 1 or not mark_ok or norm_kind not in REFLECT_READ_ONLY_KINDS:
                        errors.append(
                            f"{label}: stage '{safe_stage_key}' binding[{b_idx}] "
                            f"has invalid v3 unresolved source (must be read-only kind)"
                        )
                    else:
                        unresolved_source_bindings += 1
                        stage_has_unresolved = True

            if stage_has_unresolved:
                unresolved_source_stages += 1

    return {
        "ok": len(errors) == 0,
        "schema": schema_str,
        "version": version_int,
        "stage_count": stage_count,
        "unresolved_source_stages": unresolved_source_stages,
        "unresolved_source_bindings": unresolved_source_bindings,
        "mesh_output_stages": mesh_output_stages,
        "size_bytes": size_bytes,
        "errors": errors,
    }


def _atomic_copy_file(src: Path, dst: Path) -> None:
    """Atomically copy `src` to `dst` using a temp file in `dst.parent` + `os.replace`."""
    resolved_src = src.resolve()
    resolved_dst = dst.resolve()
    if resolved_src == resolved_dst:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp", dir=str(dst.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out_fh, src.open("rb") as in_fh:
            while True:
                chunk = in_fh.read(_COPY_CHUNK_BYTES)
                if not chunk:
                    break
                out_fh.write(chunk)
            out_fh.flush()
        os.replace(tmp_path, dst)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def stage_shader_bundle(
    pack_path: Path,
    config: PipelineConfig,
    *,
    dest_dir: Path | None = None,
    reflect_path: Path | None = None,
    dry_run: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate and atomically stage a WGPK v1/v2 pack + optional v1-v3 `.reflect.json` sidecar."""
    src_pack = Path(pack_path).expanduser()
    target_dir = Path(dest_dir) if dest_dir is not None else (config.data_dir / "shaders")
    resolved_target_dir = validate_output_path(
        target_dir,
        config.repo_root,
        label="shader staging directory",
    )

    pack_name = _safe_basename(src_pack)
    if not _SAFE_PACK_FILENAME_RE.match(pack_name) or ".." in pack_name:
        raise PipelineValidationError(
            "Invalid shader pack filename for staging.",
            code="invalid_pack_filename",
        )
    pack_dest = validate_output_path(
        resolved_target_dir / pack_name,
        config.repo_root,
        label="staged shader pack",
    )
    pack_meta = validate_shader_pack_header(src_pack)

    active_reflect: Path | None = None
    if reflect_path is not None:
        active_reflect = Path(reflect_path).expanduser()
    else:
        sibling_reflect = src_pack.parent / f"{src_pack.name}.reflect.json"
        if sibling_reflect.is_file():
            active_reflect = sibling_reflect

    reflect_meta: dict[str, Any] | None = None
    reflect_dest: Path | None = None
    reflect_name: str | None = None
    if active_reflect is not None:
        reflect_name = _safe_basename(active_reflect)
        if not _SAFE_PACK_FILENAME_RE.match(reflect_name) or ".." in reflect_name:
            raise PipelineValidationError(
                "Invalid reflection sidecar filename for staging.",
                code="invalid_reflect_filename",
            )
        reflect_dest = validate_output_path(
            resolved_target_dir / reflect_name,
            config.repo_root,
            label="staged reflection sidecar",
        )
        reflect_meta = validate_reflect_sidecar_file(active_reflect)

    all_errors: list[str] = list(pack_meta["errors"])
    if reflect_meta is not None:
        all_errors.extend(reflect_meta["errors"])

    ok = len(all_errors) == 0
    staged = False
    if ok and not dry_run:
        try:
            _atomic_copy_file(src_pack, pack_dest)
            if active_reflect is not None and reflect_dest is not None:
                _atomic_copy_file(active_reflect, reflect_dest)
            staged = True
        except OSError as exc:
            reason = exc.strerror or type(exc).__name__
            all_errors.append(f"failed to stage shader bundle: {reason}")
            ok = False

    safe_pack_name = sanitize_text(pack_name, config, env=env)
    pack_summary: dict[str, Any] = {
        "name": safe_pack_name,
        "dest": sanitize_path(pack_dest, config, env=env),
        **pack_meta,
    }
    reflect_summary: dict[str, Any] | None = None
    if reflect_meta is not None and reflect_name is not None and reflect_dest is not None:
        safe_reflect_name = sanitize_text(reflect_name, config, env=env)
        reflect_summary = {
            "name": safe_reflect_name,
            "dest": sanitize_path(reflect_dest, config, env=env),
            **reflect_meta,
        }

    result: dict[str, Any] = {
        "ok": ok,
        "dryRun": bool(dry_run),
        "staged": staged,
        "targetDir": sanitize_path(resolved_target_dir, config, env=env),
        "pack": pack_summary,
        "reflect": reflect_summary,
        "errors": all_errors,
    }
    return _sanitize_payload(result, config, env=env)


def run_pipeline(
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
    step: str = "all",
    skip_incomplete: bool = False,
    dry_run: bool = False,
    runner: Runner | None = None,
    pack_version: str = "auto",
    pack_shaders: bool = False,
    stage_pack: Path | str | None = None,
    stage_reflect: Path | str | None = None,
) -> dict[str, Any]:
    """Validate config, build plan, optionally execute via `runner`, and return a sanitized report."""
    config = resolve_config(repo_root, env=env)
    cfg_summary = _config_summary(config, env=env)

    try:
        plan = build_pipeline_plan(
            config,
            step=step,
            skip_incomplete=skip_incomplete,
            pack_version=pack_version,
            pack_shaders=pack_shaders,
        )
    except PipelineValidationError as exc:
        report: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "ok": False,
            "dryRun": dry_run,
            "step": step,
            "skipIncomplete": skip_incomplete,
            "error": str(exc),
            "errorCode": exc.code,
            "hint": exc.hint,
            "failedStep": None,
            "config": cfg_summary,
            "steps": [],
            "summary": {
                "total": 0,
                "planned": 0,
                "succeeded": 0,
                "failed": 0,
                "skipped": 0,
                "notRun": 0,
            },
        }
        return _sanitize_payload(report, config, env=env)

    active_runner = default_runner if runner is None else runner
    sub_env = _build_subprocess_env(config, base_env=env)

    step_reports: list[dict[str, Any]] = []
    planned_count = 0
    succeeded_count = 0
    failed_count = 0
    skipped_count = 0
    not_run_count = 0
    failed_step: str | None = None
    top_error: str | None = None

    for planned in plan:
        sanitized_cmd = sanitize_command(planned.command, config, env=env)
        sanitized_outputs = [sanitize_path(out, config, env=env) for out in planned.outputs]

        if planned.skipped is not None:
            skipped_count += 1
            step_reports.append(
                {
                    "name": planned.name,
                    "description": planned.description,
                    "status": "skipped",
                    "skipped": planned.skipped,
                    "command": [],
                    "outputs": [],
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                }
            )
            continue

        if dry_run:
            planned_count += 1
            step_reports.append(
                {
                    "name": planned.name,
                    "description": planned.description,
                    "status": "planned",
                    "skipped": None,
                    "command": sanitized_cmd,
                    "outputs": sanitized_outputs,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                }
            )
            continue

        if failed_step is not None:
            not_run_count += 1
            step_reports.append(
                {
                    "name": planned.name,
                    "description": planned.description,
                    "status": "not_run",
                    "skipped": f"not run because earlier step '{failed_step}' failed",
                    "command": sanitized_cmd,
                    "outputs": sanitized_outputs,
                    "returncode": None,
                    "stdout": "",
                    "stderr": "",
                }
            )
            continue

        try:
            cmd_result = _normalize_command_result(
                active_runner(list(planned.command), config.repo_root, sub_env)
            )
        except OSError as exc:
            cmd_result = CommandResult(returncode=127, stdout="", stderr=str(exc))

        stdout_clean = _truncate_output(sanitize_text(cmd_result.stdout, config, env=env))
        stderr_clean = _truncate_output(sanitize_text(cmd_result.stderr, config, env=env))

        if cmd_result.returncode == 0:
            succeeded_count += 1
            step_reports.append(
                {
                    "name": planned.name,
                    "description": planned.description,
                    "status": "succeeded",
                    "skipped": None,
                    "command": sanitized_cmd,
                    "outputs": sanitized_outputs,
                    "returncode": 0,
                    "stdout": stdout_clean,
                    "stderr": stderr_clean,
                }
            )
        else:
            failed_count += 1
            failed_step = planned.name
            top_error = f"Step '{planned.name}' failed with exit code {cmd_result.returncode}."
            step_reports.append(
                {
                    "name": planned.name,
                    "description": planned.description,
                    "status": "failed",
                    "skipped": None,
                    "command": sanitized_cmd,
                    "outputs": sanitized_outputs,
                    "returncode": cmd_result.returncode,
                    "stdout": stdout_clean,
                    "stderr": stderr_clean,
                }
            )

    staged_bundle: dict[str, Any] | None = None
    stage_error: str | None = None
    stage_error_code: str | None = None
    if stage_pack is not None and failed_count == 0:
        try:
            staged_bundle = stage_shader_bundle(
                Path(stage_pack),
                config,
                reflect_path=Path(stage_reflect) if stage_reflect is not None else None,
                dry_run=dry_run,
                env=env,
            )
            if not staged_bundle["ok"]:
                stage_error = (
                    staged_bundle["errors"][0]
                    if staged_bundle["errors"]
                    else "Shader bundle staging failed."
                )
                stage_error_code = "stage_shader_bundle_failed"
        except PipelineValidationError as exc:
            stage_error = str(exc)
            stage_error_code = exc.code

    overall_ok = failed_count == 0 and stage_error is None
    report = {
        "schemaVersion": SCHEMA_VERSION,
        "ok": overall_ok,
        "dryRun": dry_run,
        "step": step,
        "skipIncomplete": skip_incomplete,
        "error": top_error if top_error is not None else stage_error,
        "errorCode": (
            "step_failed"
            if failed_step is not None
            else stage_error_code
        ),
        "hint": None,
        "failedStep": failed_step,
        "config": cfg_summary,
        "steps": step_reports,
        "summary": {
            "total": len(step_reports),
            "planned": planned_count,
            "succeeded": succeeded_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "notRun": not_run_count,
        },
    }
    if staged_bundle is not None:
        report["stagedShaderBundle"] = staged_bundle
    return _sanitize_payload(report, config, env=env)


def run_stage_shader_bundle(
    pack_path: Path | str,
    *,
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    env: Mapping[str, str] | None = None,
    dest_dir: Path | str | None = None,
    reflect_path: Path | str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Standalone wrapper for `--stage-pack` that does not require `GTA6_GAME_DIR`."""
    config = resolve_config(repo_root, env=env)
    cfg_summary = _config_summary(config, env=env)
    try:
        staged = stage_shader_bundle(
            Path(pack_path),
            config,
            dest_dir=Path(dest_dir) if dest_dir is not None else None,
            reflect_path=Path(reflect_path) if reflect_path is not None else None,
            dry_run=dry_run,
            env=env,
        )
    except PipelineValidationError as exc:
        report: dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "ok": False,
            "dryRun": dry_run,
            "step": "stage-pack",
            "error": str(exc),
            "errorCode": exc.code,
            "hint": exc.hint,
            "config": cfg_summary,
            "stagedShaderBundle": None,
        }
        return _sanitize_payload(report, config, env=env)

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "ok": bool(staged["ok"]),
        "dryRun": dry_run,
        "step": "stage-pack",
        "error": staged["errors"][0] if staged["errors"] else None,
        "errorCode": None if staged["ok"] else "stage_shader_bundle_failed",
        "hint": None,
        "config": cfg_summary,
        "stagedShaderBundle": staged,
    }
    return _sanitize_payload(report, config, env=env)


def format_report_text(report: Mapping[str, Any]) -> str:
    """Format a sanitized pipeline report as concise terminal output."""
    mode = " (DRY RUN)" if report.get("dryRun") else ""
    lines: list[str] = [f"=== playgta6web Local Asset Pipeline{mode} ==="]

    if report.get("error") and not report.get("steps") and not report.get("stagedShaderBundle"):
        lines.append(f"ERROR: {report['error']}")
        if report.get("hint"):
            lines.append(f"Hint : {report['hint']}")
        return "\n".join(lines)

    status_tags = {
        "planned": "[PLAN]",
        "succeeded": "[OK]  ",
        "failed": "[FAIL]",
        "skipped": "[SKIP]",
        "not_run": "[WAIT]",
    }

    for entry in report.get("steps", []):
        tag = status_tags.get(str(entry.get("status", "")), "[----]")
        lines.append(f"  {tag} {entry['name']:<10} {entry['description']}")
        if entry.get("skipped"):
            lines.append(f"         reason: {entry['skipped']}")
        elif entry.get("command"):
            lines.append(f"         $ {' '.join(entry['command'])}")
        if entry.get("outputs"):
            lines.append(f"         out   : {', '.join(entry['outputs'])}")
        if entry.get("status") == "failed":
            lines.append(f"         exit  : {entry.get('returncode')}")
            if entry.get("stderr"):
                first_err = str(entry["stderr"]).splitlines()[0]
                lines.append(f"         stderr: {first_err}")

    staged = report.get("stagedShaderBundle")
    if isinstance(staged, Mapping):
        pack = staged.get("pack") or {}
        tag = "[PLAN]" if staged.get("dryRun") else ("[OK]  " if staged.get("ok") else "[FAIL]")
        ver = pack.get("version")
        ver_tag = f"WGPK v{ver}" if ver is not None else "WGPK v?"
        lines.append(
            f"  {tag} stage-pack {pack.get('name', '<pack>')} "
            f"({ver_tag}, entries={pack.get('entry_count')}, dest={pack.get('dest')})"
        )
        reflect = staged.get("reflect")
        if isinstance(reflect, Mapping):
            lines.append(
                f"         reflect: {reflect.get('name', '<reflect>')} "
                f"(v{reflect.get('version')}, stages={reflect.get('stage_count')}, "
                f"dest={reflect.get('dest')})"
            )

    if "summary" in report:
        summary = report.get("summary", {})
        lines.append("")
        lines.append(
            "Summary: "
            f"total={summary.get('total', 0)}, "
            f"planned={summary.get('planned', 0)}, "
            f"succeeded={summary.get('succeeded', 0)}, "
            f"failed={summary.get('failed', 0)}, "
            f"skipped={summary.get('skipped', 0)}, "
            f"notRun={summary.get('notRun', 0)}"
        )
    if report.get("error"):
        lines.append(f"ERROR: {report['error']}")
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Orchestrate local playgta6web asset scanning, manifest/bootset generation, "
            "PE64 import inspection, and optional SPIR-V -> WGSL shader conversion."
        )
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the playgta6web repository root (defaults to parent of tools/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print or return the sanitized execution plan without spawning subprocesses.",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Pass --skip-incomplete to manifest and bootset steps for partial downloads.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON with all host paths and usernames redacted.",
    )
    parser.add_argument(
        "--step",
        choices=STEP_CHOICES,
        default="all",
        help="Run only one pipeline step or 'all' (default: all).",
    )
    parser.add_argument(
        "--pack-version",
        choices=PACK_VERSION_CHOICES,
        default="auto",
        help="WGPK pack version policy for the shaders step: auto, 1, or 2 (default: auto).",
    )
    parser.add_argument(
        "--pack-shaders",
        action="store_true",
        help="Also emit <data_dir>/shaders/shaders.wgsl.pack during the shaders step.",
    )
    parser.add_argument(
        "--stage-pack",
        type=Path,
        default=None,
        help="Validate and atomically stage an existing WGPK v1/v2 .wgsl.pack (+ sibling .reflect.json) into local-data/shaders/.",
    )
    parser.add_argument(
        "--stage-reflect",
        type=Path,
        default=None,
        help="Optional explicit reflection sidecar (.reflect.json, v1/v2/v3) to stage alongside --stage-pack.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_arg_parser()
    args = parser.parse_args(raw_argv)

    explicit_step = any(tok == "--step" or tok.startswith("--step=") for tok in raw_argv)
    if args.stage_pack is not None and not explicit_step:
        report = run_stage_shader_bundle(
            args.stage_pack,
            repo_root=args.repo_root,
            reflect_path=args.stage_reflect,
            dry_run=args.dry_run,
        )
    else:
        report = run_pipeline(
            repo_root=args.repo_root,
            step=args.step,
            skip_incomplete=args.skip_incomplete,
            dry_run=args.dry_run,
            pack_version=args.pack_version,
            pack_shaders=args.pack_shaders,
            stage_pack=args.stage_pack,
            stage_reflect=args.stage_reflect,
        )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        stream = sys.stdout if report["ok"] else sys.stderr
        print(format_report_text(report), file=stream)

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
