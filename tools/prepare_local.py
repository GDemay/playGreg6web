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
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1

PIPELINE_STEPS: tuple[str, ...] = ("scan", "manifest", "bootset", "recomp", "shaders")
STEP_CHOICES: tuple[str, ...] = (*PIPELINE_STEPS, "all")
ALLOWED_OUTPUT_DIR_NAMES: tuple[str, ...] = ("local-data", "work")

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
) -> list[PlannedStep]:
    """Build the validated, inspectable execution plan for the requested pipeline step(s)."""
    if step not in STEP_CHOICES:
        raise PipelineValidationError(
            f"Unknown step '{step}'. Expected one of: {', '.join(STEP_CHOICES)}.",
            code="invalid_step",
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
                plan.append(
                    PlannedStep(
                        name="shaders",
                        description="Convert local SPIR-V shaders into WGSL in local-data/shaders/",
                        command=(
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
                        ),
                        outputs=(shaders_out,),
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


def run_pipeline(
    repo_root: Path | str = DEFAULT_REPO_ROOT,
    *,
    env: Mapping[str, str] | None = None,
    step: str = "all",
    skip_incomplete: bool = False,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    """Validate config, build plan, optionally execute via `runner`, and return a sanitized report."""
    config = resolve_config(repo_root, env=env)
    cfg_summary = _config_summary(config, env=env)

    try:
        plan = build_pipeline_plan(config, step=step, skip_incomplete=skip_incomplete)
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

    report = {
        "schemaVersion": SCHEMA_VERSION,
        "ok": failed_count == 0,
        "dryRun": dry_run,
        "step": step,
        "skipIncomplete": skip_incomplete,
        "error": top_error,
        "errorCode": "step_failed" if failed_step is not None else None,
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
    return _sanitize_payload(report, config, env=env)


def format_report_text(report: Mapping[str, Any]) -> str:
    """Format a sanitized pipeline report as concise terminal output."""
    mode = " (DRY RUN)" if report.get("dryRun") else ""
    lines: list[str] = [f"=== playgta6web Local Asset Pipeline{mode} ==="]

    if report.get("error") and not report.get("steps"):
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    report = run_pipeline(
        repo_root=args.repo_root,
        step=args.step,
        skip_incomplete=args.skip_incomplete,
        dry_run=args.dry_run,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        stream = sys.stdout if report["ok"] else sys.stderr
        print(format_report_text(report), file=stream)

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
