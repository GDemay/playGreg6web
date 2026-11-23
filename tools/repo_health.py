#!/usr/bin/env python3
"""Cross-track ABI drift verifier and repository health checker for playgta6web (Track 5 / W27).

Verifies in one fast pure-stdlib pass:
1. Cross-track `GpuOpcode` ABI parity across `runtime/abi/playgta6_abi.h`,
   `tools/gpu_trace.py`, `web/workers/wgpu_worker.js`, and `web/demo/smoke_scene.js`.
2. `InputState` 40-byte layout & field-offset parity between
   `runtime/abi/playgta6_abi.h` and `web/input/input_bridge.js`.
3. `HttpfsMailbox` SharedArrayBuffer constants, status/op enums, and byte-offset parity
   between `runtime/os/httpfs_mailbox.h` and `web/workers/httpfs_mailbox.js`.
4. Zero-game-files repository hygiene (no tracked `.rpf`, `.xvd`, `.xvc`, `.ttpf`,
   `.awc`, `.dxbc`, `.spv`, `.wgsl.pack`, `.pgim`, `.exe`, `.dll`, `.env`, or
   oversized binary blobs).
5. `docs/WORKSTREAMS.md` task status summary (DONE / IN PROGRESS / UNCLAIMED per track).

Security & privacy invariant:
- Machine-readable `--json` output NEVER includes raw host filesystem paths or usernames.
- Strictly uses Python 3.12 standard library modules only.

Usage:
    python tools/repo_health.py [--repo-root <path>] [--json] [--strict]
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAX_BINARY_BYTES = 512 * 1024  # 512 KiB threshold for oversized binary blobs
BINARY_SNIFF_BYTES = 8192

GPU_ABI_HEADER_REL = "runtime/abi/playgta6_abi.h"
GPU_TRACE_PY_REL = "tools/gpu_trace.py"
WGPU_WORKER_JS_REL = "web/workers/wgpu_worker.js"
SMOKE_SCENE_JS_REL = "web/demo/smoke_scene.js"
INPUT_BRIDGE_JS_REL = "web/input/input_bridge.js"
HTTPFS_HEADER_REL = "runtime/os/httpfs_mailbox.h"
HTTPFS_JS_REL = "web/workers/httpfs_mailbox.js"
WORKSTREAMS_MD_REL = "docs/WORKSTREAMS.md"

EXPECTED_INPUT_STATE_BYTES = 40

# Matches CI No Game Files Guard (.github/workflows/ci.yml) + W27 requirements.
FORBIDDEN_PATH_RE = re.compile(
    r"(^|/)\.env$|\.(rpf|xvd|xvc|ttpf|awc|dxbc|spv|wgsl\.pack|pgim|exe|dll)$",
    re.IGNORECASE,
)

# Directories skipped only when falling back to a filesystem walk outside git.
WALK_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".venv",
        "__pycache__",
        "build",
        "build-host",
        "local-data",
        "node_modules",
        "target",
        "venv",
        "work",
    }
)

# Primitive C++ types used in ABI structs (size, alignment).
_CPP_TYPE_LAYOUT: dict[str, tuple[int, int]] = {
    "int8_t": (1, 1),
    "uint8_t": (1, 1),
    "char": (1, 1),
    "bool": (1, 1),
    "int16_t": (2, 2),
    "uint16_t": (2, 2),
    "int32_t": (4, 4),
    "uint32_t": (4, 4),
    "float": (4, 4),
    "int64_t": (8, 8),
    "uint64_t": (8, 8),
    "double": (8, 8),
}

_WINDOWS_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_POSIX_PATH_RE = re.compile(
    r"/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes)/[^\s\"'<>|,;()\[\]]*"
)

_C_INT_SUFFIX_RE = re.compile(r"\b(0[xX][0-9a-fA-F]+|\d+)[uUlL]+\b")
_CPP_STATIC_CAST_RE = re.compile(r"static_cast\s*<[^>]+>\s*\(([^()]+)\)")
_TRACK_HEADING_RE = re.compile(r"^##\s+Track\s+(\d+)\s*:\s*(.+?)\s*$")
_CHECKBOX_ITEM_RE = re.compile(r"^- \[([ xX])\]\s+(.+?)\s*$")
_TASK_ID_RE = re.compile(r"\*\*([A-Z]\d+)\b")


# ---------------------------------------------------------------------------
# Generic parsing & sanitization helpers
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


def sanitize_message(text: str) -> str:
    """Strip any accidental host paths or usernames from diagnostic messages."""
    sanitized = _WINDOWS_PATH_RE.sub("<path>", text)
    sanitized = _UNC_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    for username in _current_usernames():
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        sanitized = pattern.sub("<user>", sanitized)
    return sanitized


def pascal_to_upper_snake(name: str) -> str:
    """Convert PascalCase identifier to UPPER_SNAKE_CASE."""
    step1 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    step2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", step1)
    return step2.upper()


def _strip_c_comments(text: str) -> str:
    """Remove // and /* */ comments while preserving string literals and newlines."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string: str | None = None
    while i < n:
        ch = text[i]
        if in_string is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in ('"', "'", "`"):
            in_string = ch
            out.append(ch)
            i += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            i += 2
            while i < n and text[i] != "\n":
                i += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                if text[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _extract_brace_block(text: str, open_brace_idx: int) -> str | None:
    """Return substring inside matching `{ ... }` starting at `open_brace_idx`."""
    if open_brace_idx < 0 or open_brace_idx >= len(text) or text[open_brace_idx] != "{":
        return None
    depth = 0
    in_string: str | None = None
    i = open_brace_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string is not None:
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == in_string:
                in_string = None
            i += 1
            continue
        if ch in ('"', "'", "`"):
            in_string = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace_idx + 1 : i]
        i += 1
    return None


def _eval_ast_int(node: ast.AST, symbols: Mapping[str, int]) -> int:
    if isinstance(node, ast.Expression):
        return _eval_ast_int(node.body, symbols)
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if isinstance(node, ast.Name):
        if node.id in symbols:
            return int(symbols[node.id])
        raise ValueError(f"unknown symbol: {node.id}")
    if isinstance(node, ast.UnaryOp):
        val = _eval_ast_int(node.operand, symbols)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.Invert):
            return ~val
        raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _eval_ast_int(node.left, symbols)
        right = _eval_ast_int(node.right, symbols)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Div):
            if right == 0 or left % right != 0:
                raise ValueError("non-exact integer division")
            return left // right
        if isinstance(node.op, ast.LShift):
            return left << right
        if isinstance(node.op, ast.RShift):
            return left >> right
        if isinstance(node.op, ast.BitOr):
            return left | right
        if isinstance(node.op, ast.BitAnd):
            return left & right
        if isinstance(node.op, ast.BitXor):
            return left ^ right
        raise ValueError(f"unsupported binary op: {type(node.op).__name__}")
    raise ValueError(f"unsupported integer expression node: {type(node).__name__}")


def eval_int_expr(expr: str, symbols: Mapping[str, int] | None = None) -> int:
    """Safely evaluate a C++/JS/Python integer constant expression."""
    cleaned = expr.strip()
    # Unwrap simple C++ static_cast<T>(expr)
    while True:
        updated = _CPP_STATIC_CAST_RE.sub(r"(\1)", cleaned)
        if updated == cleaned:
            break
        cleaned = updated
    # Strip C++ integer suffixes like 0x0001u, 64ULL
    cleaned = _C_INT_SUFFIX_RE.sub(r"\1", cleaned)
    # Strip BigInt 'n' suffix if present (e.g., 40n)
    cleaned = re.sub(r"\b(0[xX][0-9a-fA-F]+|\d+)n\b", r"\1", cleaned)
    tree = ast.parse(cleaned, mode="eval")
    return _eval_ast_int(tree, symbols or {})


def _read_repo_text(repo_root: Path, rel_path: str) -> tuple[str | None, str | None]:
    target = repo_root / rel_path
    if not target.is_file():
        return None, f"missing required file: {rel_path}"
    try:
        return target.read_text(encoding="utf-8"), None
    except OSError as exc:
        return None, sanitize_message(f"failed to read {rel_path}: {exc}")


def _parse_cpp_enum(
    text: str,
    enum_name: str,
    *,
    symbols: Mapping[str, int] | None = None,
) -> tuple[dict[str, int] | None, list[str]]:
    cleaned = _strip_c_comments(text)
    pattern = re.compile(
        rf"\benum\s+(?:class\s+|struct\s+)?{re.escape(enum_name)}\b[^{{;]*\{{",
        re.MULTILINE,
    )
    match = pattern.search(cleaned)
    if not match:
        return None, [f"could not find C++ enum `{enum_name}`"]
    block = _extract_brace_block(cleaned, match.end() - 1)
    if block is None:
        return None, [f"unterminated brace block for C++ enum `{enum_name}`"]

    values: dict[str, int] = {}
    errors: list[str] = []
    local_symbols = dict(symbols or {})
    next_val = 0

    for raw_entry in block.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if "=" in entry:
            name_part, expr_part = entry.split("=", 1)
            name = name_part.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                errors.append(f"invalid enumerator name `{name}` in `{enum_name}`")
                continue
            try:
                val = eval_int_expr(expr_part, local_symbols)
            except (SyntaxError, ValueError) as exc:
                errors.append(f"invalid enumerator value for `{enum_name}::{name}`: {exc}")
                continue
        else:
            name = entry
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
                errors.append(f"invalid enumerator name `{name}` in `{enum_name}`")
                continue
            val = next_val
        values[name] = val
        local_symbols[name] = val
        next_val = val + 1

    if not values and not errors:
        errors.append(f"C++ enum `{enum_name}` has no entries")
    return values, errors


def _parse_python_enum_class(
    text: str,
    class_name: str,
) -> tuple[dict[str, int] | None, list[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return None, [f"syntax error parsing Python file: {exc.msg} (line {exc.lineno})"]

    target_cls: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target_cls = node
            break

    if target_cls is None:
        return None, [f"could not find Python class `{class_name}`"]

    values: dict[str, int] = {}
    errors: list[str] = []
    for stmt in target_cls.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name = stmt.targets[0].id
            if name.startswith("_"):
                continue
            try:
                val = _eval_ast_int(stmt.value, values)
            except ValueError as exc:
                errors.append(f"non-integer member `{class_name}.{name}`: {exc}")
                continue
            values[name] = val
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name = stmt.target.id
            if name.startswith("_"):
                continue
            try:
                val = _eval_ast_int(stmt.value, values)
            except ValueError as exc:
                errors.append(f"non-integer member `{class_name}.{name}`: {exc}")
                continue
            values[name] = val

    if not values and not errors:
        errors.append(f"Python enum class `{class_name}` has no entries")
    return values, errors


def _parse_js_object_constants(
    text: str,
    var_name: str,
    *,
    symbols: Mapping[str, int] | None = None,
) -> tuple[dict[str, int] | None, list[str]]:
    cleaned = _strip_c_comments(text)
    pattern = re.compile(
        rf"\b(?:export\s+)?(?:const|let|var)\s+{re.escape(var_name)}\s*=\s*(?:Object\.freeze\s*\(\s*)?\{{",
        re.MULTILINE,
    )
    match = pattern.search(cleaned)
    if not match:
        return None, [f"could not find JS object `{var_name}`"]
    block = _extract_brace_block(cleaned, match.end() - 1)
    if block is None:
        return None, [f"unterminated brace block for JS object `{var_name}`"]

    values: dict[str, int] = {}
    errors: list[str] = []
    local_symbols = dict(symbols or {})

    for raw_entry in block.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            errors.append(f"malformed property `{entry}` in JS object `{var_name}`")
            continue
        key_part, expr_part = entry.split(":", 1)
        key = key_part.strip().strip("'\"")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            errors.append(f"invalid property name `{key}` in JS object `{var_name}`")
            continue
        try:
            val = eval_int_expr(expr_part, local_symbols)
        except (SyntaxError, ValueError) as exc:
            errors.append(f"invalid integer value for `{var_name}.{key}`: {exc}")
            continue
        values[key] = val
        local_symbols[key] = val

    if not values and not errors:
        errors.append(f"JS object `{var_name}` has no entries")
    return values, errors


def _parse_cpp_constexpr_int(
    text: str,
    const_name: str,
    *,
    symbols: Mapping[str, int] | None = None,
) -> tuple[int | None, str | None]:
    cleaned = _strip_c_comments(text)
    pattern = re.compile(
        rf"\b(?:inline\s+)?constexpr\s+[A-Za-z_][A-Za-z0-9_:]*\s+{re.escape(const_name)}\s*=\s*([^;]+);",
        re.MULTILINE,
    )
    match = pattern.search(cleaned)
    if not match:
        return None, f"could not find C++ constexpr `{const_name}`"
    try:
        return eval_int_expr(match.group(1), symbols), None
    except (SyntaxError, ValueError) as exc:
        return None, f"failed to evaluate C++ constexpr `{const_name}`: {exc}"


def _parse_js_const_int(
    text: str,
    const_name: str,
    *,
    symbols: Mapping[str, int] | None = None,
) -> tuple[int | None, str | None]:
    cleaned = _strip_c_comments(text)
    pattern = re.compile(
        rf"\b(?:export\s+)?(?:const|let|var)\s+{re.escape(const_name)}\s*=\s*([^;\n]+)",
        re.MULTILINE,
    )
    match = pattern.search(cleaned)
    if not match:
        return None, f"could not find JS constant `{const_name}`"
    try:
        return eval_int_expr(match.group(1), symbols), None
    except (SyntaxError, ValueError) as exc:
        return None, f"failed to evaluate JS constant `{const_name}`: {exc}"


def _compare_int_maps(
    expected: Mapping[str, int],
    actual: Mapping[str, int],
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    missing = sorted(k for k in expected if k not in actual)
    extra = sorted(k for k in actual if k not in expected)
    mismatched: list[dict[str, Any]] = []
    for k in sorted(set(expected) & set(actual)):
        if expected[k] != actual[k]:
            mismatched.append({"name": k, "expected": expected[k], "actual": actual[k]})
    return missing, extra, mismatched


# ---------------------------------------------------------------------------
# Check 1: Cross-Track GpuOpcode ABI Parity
# ---------------------------------------------------------------------------


def check_gpu_opcode_parity(repo_root: Path | str) -> dict[str, Any]:
    """Verify GpuOpcode numeric parity across C++, Python, and JS tracks."""
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []

    header_text, read_err = _read_repo_text(root, GPU_ABI_HEADER_REL)
    canonical: dict[str, int] = {}
    if read_err is not None or header_text is None:
        errors.append(read_err or f"missing {GPU_ABI_HEADER_REL}")
    else:
        parsed, parse_errs = _parse_cpp_enum(header_text, "GpuOpcode")
        if parse_errs:
            errors.extend(f"{GPU_ABI_HEADER_REL}: {e}" for e in parse_errs)
        if parsed:
            canonical = parsed

    if canonical:
        # Check duplicate numeric values in canonical enum
        seen_vals: dict[int, str] = {}
        for name, val in canonical.items():
            if val in seen_vals:
                errors.append(
                    f"{GPU_ABI_HEADER_REL}: duplicate GpuOpcode value {val} for `{seen_vals[val]}` and `{name}`"
                )
            else:
                seen_vals[val] = name

        sorted_vals = sorted(canonical.values())
        expected_range = list(range(len(canonical)))
        if sorted_vals != expected_range:
            warnings.append(
                f"{GPU_ABI_HEADER_REL}: GpuOpcode values are not contiguous 0..{len(canonical) - 1}"
            )

    canonical_upper = {pascal_to_upper_snake(k): v for k, v in canonical.items()}

    target_specs: tuple[tuple[str, str, str, Mapping[str, int]], ...] = (
        (GPU_TRACE_PY_REL, "python_enum", "GpuOpcode", canonical),
        (WGPU_WORKER_JS_REL, "js_object", "OPCODES", canonical_upper),
        (SMOKE_SCENE_JS_REL, "js_object", "GPU_OPCODES", canonical_upper),
    )

    targets_report: dict[str, dict[str, Any]] = {}
    for rel_path, kind, symbol_name, expected_map in target_specs:
        target_errors: list[str] = []
        target_text, err = _read_repo_text(root, rel_path)
        parsed_map: dict[str, int] = {}

        if err is not None or target_text is None:
            target_errors.append(err or f"missing {rel_path}")
        elif kind == "python_enum":
            parsed, perrs = _parse_python_enum_class(target_text, symbol_name)
            target_errors.extend(perrs)
            if parsed:
                parsed_map = parsed
        else:
            parsed, perrs = _parse_js_object_constants(target_text, symbol_name)
            target_errors.extend(perrs)
            if parsed:
                parsed_map = parsed

        missing: list[str] = []
        extra: list[str] = []
        mismatched: list[dict[str, Any]] = []

        if canonical and not target_errors:
            missing, extra, mismatched = _compare_int_maps(expected_map, parsed_map)
            for name in missing:
                target_errors.append(f"missing opcode `{name}` (expected {expected_map[name]})")
            for name in extra:
                target_errors.append(f"extra opcode `{name}` = {parsed_map[name]}")
            for diff in mismatched:
                target_errors.append(
                    f"opcode `{diff['name']}` drift: expected {diff['expected']}, got {diff['actual']}"
                )

        target_ok = bool(canonical) and not target_errors
        targets_report[rel_path] = {
            "ok": target_ok,
            "symbol": symbol_name,
            "count": len(parsed_map),
            "missing": missing,
            "extra": extra,
            "mismatched": mismatched,
            "errors": target_errors,
        }
        errors.extend(f"{rel_path}: {e}" for e in target_errors)

    ok = bool(canonical) and not errors
    return {
        "name": "gpu_opcode_parity",
        "ok": ok,
        "canonical_source": GPU_ABI_HEADER_REL,
        "opcode_count": len(canonical),
        "min_opcode": min(canonical.values()) if canonical else None,
        "max_opcode": max(canonical.values()) if canonical else None,
        "opcodes": dict(sorted(canonical.items(), key=lambda kv: kv[1])),
        "targets": targets_report,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Check 2: InputState 40-Byte ABI Parity
# ---------------------------------------------------------------------------


def _parse_cpp_struct_layout(
    text: str,
    struct_name: str,
) -> tuple[int | None, dict[str, int] | None, list[str]]:
    """Compute standard-layout byte size & field offsets for a simple C++ ABI struct."""
    cleaned = _strip_c_comments(text)
    pattern = re.compile(rf"\bstruct\s+(?:alignas\([^)]+\)\s+)?{re.escape(struct_name)}\s*\{{")
    match = pattern.search(cleaned)
    if not match:
        return None, None, [f"could not find C++ struct `{struct_name}`"]
    block = _extract_brace_block(cleaned, match.end() - 1)
    if block is None:
        return None, None, [f"unterminated brace block for C++ struct `{struct_name}`"]

    offsets: dict[str, int] = {}
    errors: list[str] = []
    offset = 0
    max_align = 1

    for raw_stmt in block.split(";"):
        stmt = raw_stmt.strip()
        if not stmt:
            continue
        # Remove default member initializer if present (e.g., `= 0`)
        stmt = stmt.split("=", 1)[0].strip()
        field_match = re.fullmatch(
            r"(?:std::)?([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[\s*([^\]]+)\s*\])?",
            stmt,
        )
        if not field_match:
            errors.append(f"unrecognized field declaration `{stmt}` in struct `{struct_name}`")
            continue
        type_name, field_name, array_len_expr = field_match.groups()
        if type_name not in _CPP_TYPE_LAYOUT:
            errors.append(f"unsupported field type `{type_name}` in struct `{struct_name}`")
            continue
        size, align = _CPP_TYPE_LAYOUT[type_name]
        count = 1
        if array_len_expr is not None:
            try:
                count = eval_int_expr(array_len_expr)
            except (SyntaxError, ValueError) as exc:
                errors.append(f"unsupported array bound `{array_len_expr}` on `{field_name}`: {exc}")
                continue
        if offset % align != 0:
            offset += align - (offset % align)
        offsets[field_name] = offset
        offset += size * count
        if align > max_align:
            max_align = align

    if offset % max_align != 0:
        offset += max_align - (offset % max_align)

    if not offsets and not errors:
        errors.append(f"C++ struct `{struct_name}` has no fields")
    return offset, offsets, errors


def check_input_state_parity(repo_root: Path | str) -> dict[str, Any]:
    """Verify InputState 40-byte size & field offsets between C++ ABI and JS InputBridge."""
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []

    header_text, header_err = _read_repo_text(root, GPU_ABI_HEADER_REL)
    js_text, js_err = _read_repo_text(root, INPUT_BRIDGE_JS_REL)

    cpp_assert_bytes: int | None = None
    cpp_computed_bytes: int | None = None
    cpp_offsets: dict[str, int] = {}

    if header_err is not None or header_text is None:
        errors.append(header_err or f"missing {GPU_ABI_HEADER_REL}")
    else:
        cleaned_header = _strip_c_comments(header_text)
        sa_match = re.search(
            r"\bstatic_assert\s*\(\s*sizeof\s*\(\s*InputState\s*\)\s*==\s*([^,\)]+)",
            cleaned_header,
        )
        if not sa_match:
            errors.append(f"{GPU_ABI_HEADER_REL}: missing static_assert(sizeof(InputState) == ...)")
        else:
            try:
                cpp_assert_bytes = eval_int_expr(sa_match.group(1))
            except (SyntaxError, ValueError) as exc:
                errors.append(
                    f"{GPU_ABI_HEADER_REL}: invalid sizeof(InputState) assertion expression: {exc}"
                )

        computed_size, computed_offsets, struct_errs = _parse_cpp_struct_layout(
            header_text, "InputState"
        )
        if struct_errs:
            errors.extend(f"{GPU_ABI_HEADER_REL}: {e}" for e in struct_errs)
        if computed_size is not None and computed_offsets is not None:
            cpp_computed_bytes = computed_size
            cpp_offsets = {k.upper(): v for k, v in computed_offsets.items()}

        if cpp_assert_bytes is not None and cpp_assert_bytes != EXPECTED_INPUT_STATE_BYTES:
            errors.append(
                f"{GPU_ABI_HEADER_REL}: sizeof(InputState) asserted as {cpp_assert_bytes}, "
                f"expected frozen ABI size {EXPECTED_INPUT_STATE_BYTES}"
            )
        if (
            cpp_assert_bytes is not None
            and cpp_computed_bytes is not None
            and cpp_assert_bytes != cpp_computed_bytes
        ):
            errors.append(
                f"{GPU_ABI_HEADER_REL}: struct InputState computed size ({cpp_computed_bytes}) "
                f"does not match static_assert ({cpp_assert_bytes})"
            )

    js_bytes: int | None = None
    js_offsets: dict[str, int] = {}

    if js_err is not None or js_text is None:
        errors.append(js_err or f"missing {INPUT_BRIDGE_JS_REL}")
    else:
        parsed_js_bytes, byte_err = _parse_js_const_int(js_text, "INPUT_STATE_BYTES")
        if byte_err is not None:
            errors.append(f"{INPUT_BRIDGE_JS_REL}: {byte_err}")
        else:
            js_bytes = parsed_js_bytes

        parsed_js_offsets, offset_errs = _parse_js_object_constants(js_text, "INPUT_OFFSETS")
        if offset_errs:
            errors.extend(f"{INPUT_BRIDGE_JS_REL}: {e}" for e in offset_errs)
        elif parsed_js_offsets:
            js_offsets = parsed_js_offsets

    if cpp_assert_bytes is not None and js_bytes is not None and cpp_assert_bytes != js_bytes:
        errors.append(
            f"InputState byte size drift: {GPU_ABI_HEADER_REL} has {cpp_assert_bytes}, "
            f"{INPUT_BRIDGE_JS_REL} has INPUT_STATE_BYTES = {js_bytes}"
        )

    if cpp_offsets and js_offsets:
        missing, extra, mismatched = _compare_int_maps(cpp_offsets, js_offsets)
        for name in missing:
            errors.append(
                f"{INPUT_BRIDGE_JS_REL}: INPUT_OFFSETS missing field `{name}` (expected {cpp_offsets[name]})"
            )
        for name in extra:
            errors.append(
                f"{INPUT_BRIDGE_JS_REL}: INPUT_OFFSETS has extra field `{name}` = {js_offsets[name]}"
            )
        for diff in mismatched:
            errors.append(
                f"{INPUT_BRIDGE_JS_REL}: INPUT_OFFSETS.{diff['name']} drift: "
                f"expected {diff['expected']}, got {diff['actual']}"
            )

    return {
        "name": "input_state_parity",
        "ok": not errors,
        "expected_bytes": EXPECTED_INPUT_STATE_BYTES,
        "cpp_source": GPU_ABI_HEADER_REL,
        "js_source": INPUT_BRIDGE_JS_REL,
        "cpp_static_assert_bytes": cpp_assert_bytes,
        "cpp_computed_bytes": cpp_computed_bytes,
        "js_input_state_bytes": js_bytes,
        "cpp_field_offsets": cpp_offsets,
        "js_field_offsets": js_offsets,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Check 3: HttpfsMailbox SAB Parity
# ---------------------------------------------------------------------------


def _parse_cpp_offsetof_asserts(
    text: str,
    struct_name: str,
    *,
    symbols: Mapping[str, int] | None = None,
) -> tuple[dict[str, int], list[str]]:
    cleaned = _strip_c_comments(text)
    pattern = re.compile(
        rf"\bstatic_assert\s*\(\s*offsetof\s*\(\s*{re.escape(struct_name)}\s*,\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)\s*==\s*([^,\)]+)",
        re.MULTILINE,
    )
    offsets: dict[str, int] = {}
    errors: list[str] = []
    for match in pattern.finditer(cleaned):
        field_name = match.group(1)
        expr = match.group(2)
        try:
            offsets[field_name] = eval_int_expr(expr, symbols)
        except (SyntaxError, ValueError) as exc:
            errors.append(f"failed to evaluate offsetof({struct_name}, {field_name}): {exc}")
    if not offsets and not errors:
        errors.append(f"no static_assert(offsetof({struct_name}, ...)) entries found")
    return offsets, errors


def check_httpfs_mailbox_parity(repo_root: Path | str) -> dict[str, Any]:
    """Verify HttpfsMailbox SAB header/payload constants, enums, and field offsets."""
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []

    cpp_text, cpp_err = _read_repo_text(root, HTTPFS_HEADER_REL)
    js_text, js_err = _read_repo_text(root, HTTPFS_JS_REL)

    cpp_constants: dict[str, int] = {}
    js_constants: dict[str, int] = {}
    cpp_status: dict[str, int] = {}
    js_status: dict[str, int] = {}
    cpp_op: dict[str, int] = {}
    js_op: dict[str, int] = {}
    cpp_offsets: dict[str, int] = {}
    js_offsets: dict[str, int] = {}

    const_pairs: tuple[tuple[str, str, str], ...] = (
        ("mailbox_version", "kHttpfsMailboxVersion", "HTTPFS_MAILBOX_VERSION"),
        ("header_bytes", "kHttpfsHeaderBytes", "HTTPFS_HEADER_BYTES"),
        ("payload_bytes", "kHttpfsPayloadBytes", "HTTPFS_PAYLOAD_BYTES"),
    )

    if cpp_err is not None or cpp_text is None:
        errors.append(cpp_err or f"missing {HTTPFS_HEADER_REL}")
    else:
        cpp_syms: dict[str, int] = {}
        for logical, cpp_name, _ in const_pairs:
            val, err = _parse_cpp_constexpr_int(cpp_text, cpp_name, symbols=cpp_syms)
            if err is not None or val is None:
                errors.append(f"{HTTPFS_HEADER_REL}: {err}")
            else:
                cpp_constants[logical] = val
                cpp_syms[cpp_name] = val

        parsed_status, status_errs = _parse_cpp_enum(cpp_text, "HttpfsStatus", symbols=cpp_syms)
        errors.extend(f"{HTTPFS_HEADER_REL}: {e}" for e in status_errs)
        if parsed_status:
            cpp_status = parsed_status

        parsed_op, op_errs = _parse_cpp_enum(cpp_text, "HttpfsOp", symbols=cpp_syms)
        errors.extend(f"{HTTPFS_HEADER_REL}: {e}" for e in op_errs)
        if parsed_op:
            cpp_op = parsed_op

        parsed_offsets, offset_errs = _parse_cpp_offsetof_asserts(
            cpp_text, "HttpfsMailbox", symbols=cpp_syms
        )
        errors.extend(f"{HTTPFS_HEADER_REL}: {e}" for e in offset_errs)
        cpp_offsets = {k.upper(): v for k, v in parsed_offsets.items()}

    if js_err is not None or js_text is None:
        errors.append(js_err or f"missing {HTTPFS_JS_REL}")
    else:
        js_syms: dict[str, int] = {}
        for logical, _, js_name in const_pairs:
            val, err = _parse_js_const_int(js_text, js_name, symbols=js_syms)
            if err is not None or val is None:
                errors.append(f"{HTTPFS_JS_REL}: {err}")
            else:
                js_constants[logical] = val
                js_syms[js_name] = val

        mb_val, mb_err = _parse_js_const_int(js_text, "HTTPFS_MAILBOX_BYTES", symbols=js_syms)
        if mb_err is None and mb_val is not None:
            js_constants["mailbox_bytes"] = mb_val
            js_syms["HTTPFS_MAILBOX_BYTES"] = mb_val
            if (
                "header_bytes" in js_constants
                and "payload_bytes" in js_constants
                and mb_val != js_constants["header_bytes"] + js_constants["payload_bytes"]
            ):
                errors.append(
                    f"{HTTPFS_JS_REL}: HTTPFS_MAILBOX_BYTES ({mb_val}) != "
                    f"HTTPFS_HEADER_BYTES + HTTPFS_PAYLOAD_BYTES"
                )

        parsed_js_status, js_status_errs = _parse_js_object_constants(
            js_text, "HttpfsStatus", symbols=js_syms
        )
        errors.extend(f"{HTTPFS_JS_REL}: {e}" for e in js_status_errs)
        if parsed_js_status:
            js_status = parsed_js_status

        parsed_js_op, js_op_errs = _parse_js_object_constants(js_text, "HttpfsOp", symbols=js_syms)
        errors.extend(f"{HTTPFS_JS_REL}: {e}" for e in js_op_errs)
        if parsed_js_op:
            js_op = parsed_js_op

        parsed_js_offsets, js_offset_errs = _parse_js_object_constants(
            js_text, "HttpfsOffsets", symbols=js_syms
        )
        errors.extend(f"{HTTPFS_JS_REL}: {e}" for e in js_offset_errs)
        if parsed_js_offsets:
            js_offsets = parsed_js_offsets

    # Compare key constants
    for logical, cpp_name, js_name in const_pairs:
        if logical in cpp_constants and logical in js_constants:
            if cpp_constants[logical] != js_constants[logical]:
                errors.append(
                    f"HttpfsMailbox constant drift for `{logical}`: "
                    f"{cpp_name}={cpp_constants[logical]} vs {js_name}={js_constants[logical]}"
                )

    # Compare HttpfsStatus enum members (and matching UPPER_SNAKE aliases if present)
    if cpp_status and js_status:
        for name, expected_val in cpp_status.items():
            if name not in js_status:
                errors.append(
                    f"{HTTPFS_JS_REL}: HttpfsStatus missing `{name}` (expected {expected_val})"
                )
            elif js_status[name] != expected_val:
                errors.append(
                    f"{HTTPFS_JS_REL}: HttpfsStatus.{name} drift: expected {expected_val}, got {js_status[name]}"
                )
            upper_alias = pascal_to_upper_snake(name)
            if upper_alias in js_status and js_status[upper_alias] != expected_val:
                errors.append(
                    f"{HTTPFS_JS_REL}: HttpfsStatus.{upper_alias} alias drift: "
                    f"expected {expected_val}, got {js_status[upper_alias]}"
                )

    # Compare HttpfsOp enum members (and matching UPPER_SNAKE aliases if present)
    if cpp_op and js_op:
        for name, expected_val in cpp_op.items():
            if name not in js_op:
                errors.append(f"{HTTPFS_JS_REL}: HttpfsOp missing `{name}` (expected {expected_val})")
            elif js_op[name] != expected_val:
                errors.append(
                    f"{HTTPFS_JS_REL}: HttpfsOp.{name} drift: expected {expected_val}, got {js_op[name]}"
                )
            upper_alias = pascal_to_upper_snake(name)
            if upper_alias in js_op and js_op[upper_alias] != expected_val:
                errors.append(
                    f"{HTTPFS_JS_REL}: HttpfsOp.{upper_alias} alias drift: "
                    f"expected {expected_val}, got {js_op[upper_alias]}"
                )

    # Compare HttpfsMailbox field byte offsets
    if cpp_offsets and js_offsets:
        missing, extra, mismatched = _compare_int_maps(cpp_offsets, js_offsets)
        for name in missing:
            errors.append(
                f"{HTTPFS_JS_REL}: HttpfsOffsets missing `{name}` (expected {cpp_offsets[name]})"
            )
        for name in extra:
            errors.append(
                f"{HTTPFS_JS_REL}: HttpfsOffsets has extra field `{name}` = {js_offsets[name]}"
            )
        for diff in mismatched:
            errors.append(
                f"{HTTPFS_JS_REL}: HttpfsOffsets.{diff['name']} drift: "
                f"expected {diff['expected']}, got {diff['actual']}"
            )

    return {
        "name": "httpfs_mailbox_parity",
        "ok": not errors,
        "cpp_source": HTTPFS_HEADER_REL,
        "js_source": HTTPFS_JS_REL,
        "cpp_constants": cpp_constants,
        "js_constants": js_constants,
        "cpp_status": cpp_status,
        "cpp_op": cpp_op,
        "cpp_offsets": cpp_offsets,
        "js_offsets": js_offsets,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Check 4: Zero-Game-Files Repo Hygiene
# ---------------------------------------------------------------------------


def _list_repo_files(repo_root: Path) -> tuple[list[str], str]:
    """Return sorted repo-relative POSIX paths using `git ls-files` or directory walk."""
    resolved_root = repo_root.resolve()
    try:
        top_proc = subprocess.run(
            ["git", "-C", str(resolved_root), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )
        if top_proc.returncode == 0:
            toplevel = Path(top_proc.stdout.strip()).resolve()
            if toplevel == resolved_root:
                ls_proc = subprocess.run(
                    ["git", "-C", str(resolved_root), "ls-files", "-z"],
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                )
                if ls_proc.returncode == 0:
                    raw_entries = ls_proc.stdout.split(b"\x00")
                    paths = [
                        entry.decode("utf-8", errors="surrogateescape").replace("\\", "/")
                        for entry in raw_entries
                        if entry
                    ]
                    return sorted(paths), "git"
    except (OSError, subprocess.SubprocessError):
        pass

    # Fallback: deterministic directory walk skipping build/VCS/local-data directories
    discovered: list[str] = []
    for dirpath, dirnames, filenames in os.walk(resolved_root):
        dirnames[:] = sorted(d for d in dirnames if d not in WALK_SKIP_DIRS)
        rel_dir = Path(dirpath).relative_to(resolved_root)
        for fname in sorted(filenames):
            rel = (rel_dir / fname).as_posix()
            if rel.startswith("./"):
                rel = rel[2:]
            discovered.append(rel)
    return sorted(discovered), "walk"


def _is_binary_file(path: Path) -> bool:
    """Heuristically classify a file as binary by inspecting up to 8 KiB."""
    try:
        with path.open("rb") as fh:
            sample = fh.read(BINARY_SNIFF_BYTES)
    except OSError:
        return False
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        # Handle UTF-8 multi-byte codepoint split cleanly across sample boundary.
        for trim in (1, 2, 3):
            if len(sample) > trim:
                try:
                    sample[:-trim].decode("utf-8")
                    return False
                except UnicodeDecodeError:
                    continue
        return True
    return False


def check_tracked_hygiene(
    repo_root: Path | str,
    *,
    max_binary_bytes: int = DEFAULT_MAX_BINARY_BYTES,
) -> dict[str, Any]:
    """Verify zero copyrighted game files, `.env`, or oversized binary blobs are tracked."""
    root = Path(repo_root)
    files, source = _list_repo_files(root)

    forbidden_files: list[str] = []
    oversized_binaries: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    for rel in files:
        if FORBIDDEN_PATH_RE.search(rel):
            forbidden_files.append(rel)
            errors.append(f"forbidden file tracked: {rel}")

        full_path = root / rel
        if not full_path.is_file():
            continue
        try:
            size = full_path.stat().st_size
        except OSError:
            continue

        if size > max_binary_bytes and _is_binary_file(full_path):
            oversized_binaries.append({"path": rel, "bytes": size})
            errors.append(
                f"oversized binary blob tracked: {rel} ({size} bytes > {max_binary_bytes} limit)"
            )

    return {
        "name": "tracked_hygiene",
        "ok": not errors,
        "source": source,
        "tracked_file_count": len(files),
        "max_binary_bytes": max_binary_bytes,
        "forbidden_files": forbidden_files,
        "oversized_binaries": oversized_binaries,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Check 5: Workstreams Summary
# ---------------------------------------------------------------------------


def summarize_workstreams(repo_root: Path | str) -> dict[str, Any]:
    """Parse docs/WORKSTREAMS.md and count DONE / IN PROGRESS / UNCLAIMED tasks per track."""
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []

    text, read_err = _read_repo_text(root, WORKSTREAMS_MD_REL)
    if read_err is not None or text is None:
        errors.append(read_err or f"missing {WORKSTREAMS_MD_REL}")
        return {
            "name": "workstreams_summary",
            "ok": False,
            "source": WORKSTREAMS_MD_REL,
            "tracks": [],
            "totals": {
                "total": 0,
                "done": 0,
                "in_progress": 0,
                "unclaimed": 0,
                "completion_pct": 0.0,
            },
            "errors": errors,
            "warnings": warnings,
        }

    tracks: list[dict[str, Any]] = []
    current_track: dict[str, Any] | None = None
    seen_task_ids: dict[str, int] = {}

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip()
        track_match = _TRACK_HEADING_RE.match(line)
        if track_match:
            track_num = int(track_match.group(1))
            track_title = track_match.group(2).strip()
            current_track = {
                "track": track_num,
                "title": track_title,
                "total": 0,
                "done": 0,
                "in_progress": 0,
                "unclaimed": 0,
            }
            tracks.append(current_track)
            continue

        item_match = _CHECKBOX_ITEM_RE.match(line)
        if not item_match:
            continue

        if current_track is None:
            warnings.append(
                f"{WORKSTREAMS_MD_REL}:{line_no}: task item found outside any Track section"
            )
            continue

        checked = item_match.group(1).lower() == "x"
        body = item_match.group(2)

        id_match = _TASK_ID_RE.search(body)
        task_id = id_match.group(1) if id_match else None
        if task_id is not None:
            if task_id in seen_task_ids:
                warnings.append(
                    f"{WORKSTREAMS_MD_REL}:{line_no}: duplicate task id `{task_id}` "
                    f"(first seen on line {seen_task_ids[task_id]})"
                )
            else:
                seen_task_ids[task_id] = line_no

        has_in_progress = bool(re.search(r"`IN PROGRESS`|\bIN PROGRESS\b", body, re.IGNORECASE))
        has_done_tag = bool(re.search(r"`DONE`", body))

        current_track["total"] += 1
        if checked:
            current_track["done"] += 1
            if has_in_progress:
                warnings.append(
                    f"{WORKSTREAMS_MD_REL}:{line_no}: checked `[x]` task "
                    f"`{task_id or 'unknown'}` still tagged IN PROGRESS"
                )
        elif has_in_progress:
            current_track["in_progress"] += 1
        else:
            current_track["unclaimed"] += 1
            if has_done_tag:
                warnings.append(
                    f"{WORKSTREAMS_MD_REL}:{line_no}: unchecked `[ ]` task "
                    f"`{task_id or 'unknown'}` tagged DONE"
                )

    total_tasks = sum(t["total"] for t in tracks)
    total_done = sum(t["done"] for t in tracks)
    total_in_progress = sum(t["in_progress"] for t in tracks)
    total_unclaimed = sum(t["unclaimed"] for t in tracks)
    completion_pct = round((total_done / total_tasks) * 100.0, 1) if total_tasks > 0 else 0.0

    if not tracks:
        errors.append(f"{WORKSTREAMS_MD_REL}: no `## Track N:` sections found")
    elif total_tasks == 0:
        errors.append(f"{WORKSTREAMS_MD_REL}: no task checkbox items found")

    return {
        "name": "workstreams_summary",
        "ok": not errors,
        "source": WORKSTREAMS_MD_REL,
        "tracks": tracks,
        "totals": {
            "total": total_tasks,
            "done": total_done,
            "in_progress": total_in_progress,
            "unclaimed": total_unclaimed,
            "completion_pct": completion_pct,
        },
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Top-level aggregator & CLI formatting
# ---------------------------------------------------------------------------


def check_repo_health(
    repo_root: Path | str | None = None,
    *,
    strict: bool = False,
    max_binary_bytes: int = DEFAULT_MAX_BINARY_BYTES,
) -> dict[str, Any]:
    """Run all cross-track ABI parity, repo hygiene, and workstream summary checks."""
    root = Path(repo_root) if repo_root is not None else DEFAULT_REPO_ROOT

    gpu_check = check_gpu_opcode_parity(root)
    input_check = check_input_state_parity(root)
    httpfs_check = check_httpfs_mailbox_parity(root)
    hygiene_check = check_tracked_hygiene(root, max_binary_bytes=max_binary_bytes)
    workstreams_check = summarize_workstreams(root)

    checks: dict[str, dict[str, Any]] = {
        "gpu_opcode_parity": gpu_check,
        "input_state_parity": input_check,
        "httpfs_mailbox_parity": httpfs_check,
        "tracked_hygiene": hygiene_check,
        "workstreams_summary": workstreams_check,
    }

    error_count = sum(len(c["errors"]) for c in checks.values())
    warning_count = sum(len(c["warnings"]) for c in checks.values())
    all_checks_ok = all(bool(c["ok"]) for c in checks.values())
    overall_ok = all_checks_ok and (warning_count == 0 if strict else True)

    return {
        "ok": overall_ok,
        "strict": strict,
        "error_count": error_count,
        "warning_count": warning_count,
        "checks": checks,
    }


def format_human_report(report: Mapping[str, Any]) -> str:
    """Format a concise human-readable multi-track repo health report."""
    lines: list[str] = []
    status_label = "PASS" if report["ok"] else "FAIL"
    mode_suffix = " (strict)" if report.get("strict") else ""
    lines.append(f"playgta6web repo health: {status_label}{mode_suffix}")
    lines.append("=" * 64)

    checks = report["checks"]

    # 1. GpuOpcode parity
    gpu = checks["gpu_opcode_parity"]
    tag = "PASS" if gpu["ok"] else "FAIL"
    op_range = (
        f"{gpu['min_opcode']}..{gpu['max_opcode']}"
        if gpu["min_opcode"] is not None and gpu["max_opcode"] is not None
        else "n/a"
    )
    lines.append(
        f"[{tag}] GpuOpcode ABI parity: {gpu['opcode_count']} opcodes ({op_range}) "
        f"across {len(gpu['targets']) + 1} files"
    )
    for err in gpu["errors"]:
        lines.append(f"       ERROR: {err}")
    for warn in gpu["warnings"]:
        lines.append(f"       WARN:  {warn}")

    # 2. InputState parity
    inp = checks["input_state_parity"]
    tag = "PASS" if inp["ok"] else "FAIL"
    lines.append(
        f"[{tag}] InputState ABI parity: C++={inp['cpp_static_assert_bytes']}B "
        f"(computed {inp['cpp_computed_bytes']}B), JS={inp['js_input_state_bytes']}B, "
        f"{len(inp['cpp_field_offsets'])} field offsets"
    )
    for err in inp["errors"]:
        lines.append(f"       ERROR: {err}")
    for warn in inp["warnings"]:
        lines.append(f"       WARN:  {warn}")

    # 3. HttpfsMailbox parity
    httpfs = checks["httpfs_mailbox_parity"]
    tag = "PASS" if httpfs["ok"] else "FAIL"
    hdr = httpfs["cpp_constants"].get("header_bytes")
    pay = httpfs["cpp_constants"].get("payload_bytes")
    ver = httpfs["cpp_constants"].get("mailbox_version")
    lines.append(
        f"[{tag}] HttpfsMailbox SAB parity: v{ver}, header={hdr}B, payload={pay}B, "
        f"{len(httpfs['cpp_offsets'])} offsets"
    )
    for err in httpfs["errors"]:
        lines.append(f"       ERROR: {err}")
    for warn in httpfs["warnings"]:
        lines.append(f"       WARN:  {warn}")

    # 4. Tracked hygiene
    hyg = checks["tracked_hygiene"]
    tag = "PASS" if hyg["ok"] else "FAIL"
    lines.append(
        f"[{tag}] Repo hygiene ({hyg['source']}): {hyg['tracked_file_count']} files checked, "
        f"{len(hyg['forbidden_files'])} forbidden, {len(hyg['oversized_binaries'])} oversized binaries"
    )
    for err in hyg["errors"]:
        lines.append(f"       ERROR: {err}")
    for warn in hyg["warnings"]:
        lines.append(f"       WARN:  {warn}")

    # 5. Workstreams summary
    ws = checks["workstreams_summary"]
    tag = "PASS" if ws["ok"] else "FAIL"
    totals = ws["totals"]
    lines.append(
        f"[{tag}] Workstreams summary: {totals['done']}/{totals['total']} DONE "
        f"({totals['completion_pct']}%), {totals['in_progress']} IN PROGRESS, "
        f"{totals['unclaimed']} UNCLAIMED"
    )
    for track in ws["tracks"]:
        lines.append(
            f"       Track {track['track']}: {track['done']:2d} done, "
            f"{track['in_progress']:2d} in-progress, {track['unclaimed']:2d} unclaimed "
            f"({track['total']:2d} total) — {track['title']}"
        )
    for err in ws["errors"]:
        lines.append(f"       ERROR: {err}")
    for warn in ws["warnings"]:
        lines.append(f"       WARN:  {warn}")

    lines.append("=" * 64)
    lines.append(
        f"Summary: {report['error_count']} error(s), {report['warning_count']} warning(s)"
    )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify cross-track ABI parity, zero-game-files hygiene, and workstream progress."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Path to the playgta6web repository root (default: auto-detected).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit sanitized machine-readable JSON report (never includes host paths).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with exit code 1 if any warnings are present in addition to errors.",
    )
    parser.add_argument(
        "--max-binary-bytes",
        type=int,
        default=DEFAULT_MAX_BINARY_BYTES,
        help=f"Maximum allowed tracked binary blob size in bytes (default: {DEFAULT_MAX_BINARY_BYTES}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    report = check_repo_health(
        args.repo_root,
        strict=args.strict,
        max_binary_bytes=args.max_binary_bytes,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_human_report(report))

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
