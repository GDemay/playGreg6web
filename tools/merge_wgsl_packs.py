#!/usr/bin/env python3
"""Union, minify, verify, and diff WGPK v1/v2 shader packs and `.reflect.json` sidecars.

Merge / incremental update / minify:
    python tools/merge_wgsl_packs.py a.wgsl.pack b.wgsl.pack --out merged.wgsl.pack \
        [--force-v2] [--minify-wgsl] [--minify-mode {line,compact,aggressive}] \
        [--in-place] [--fail-on-stage-conflict] [--verify] [--json]

Missing-shader diff against an existing WGPK v1/v2 pack + `.reflect.json`:
    python tools/merge_wgsl_packs.py diff --pack existing.wgsl.pack \
        [--reflect existing.wgsl.pack.reflect.json] \
        [--boot-log boot.log ...] [--missing-json missing.json ...] [--dump-dir dump/ ...] \
        [--emit-priority-hashes missing.txt] [--strict] [--json]

Each input `<pack>` may have a `<pack>.reflect.json` next to it (missing is fine). Stages are
deduplicated by 64-bit shader hash (the canonical WGPK TOC and sidecar identity key). When several
inputs carry the same hash, the one whose sidecar has an entry wins, then the one with more
bindings that have a `source`, then the earlier input. Cross-input stage conflicts (same hash with
different stage codes) are tracked and reported, and can fail fast via `--fail-on-stage-conflict`.
Output TOC order is strictly ascending hash, so the result is deterministic.

The output is WGPK v1 when it fits in u32 offsets, else v2 (or v2 explicitly when `--force-v2` is
set). Packs are streamed in bounded chunks (`CHUNK`) and written atomically via sibling temp files,
so multi-GiB WGPK v2 packs are processed in bounded RAM without loading payloads into memory.
With `--minify-wgsl`, each WGSL payload is compacted semantically (preserving W3C WGSL lexical
tokens, nested `/* ... */` block comments, operator/number/identifier boundaries, global directives,
and downstream binding/attribute parser spacing) without altering 64-bit TOC hashes. With
`--verify`, the written pack is verified via streaming UTF-8/SHA-256 parity and
`stage_reflection_sidecar.py` TOC + sidecar cross-checking before atomic commit. Pure stdlib.

The merged pack and sidecar are derived from the title's shaders: write them outside the
repository (e.g. under work/ or local-data/), never commit them.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib
import json
import os
import re
import struct
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

HEADER = struct.Struct("<4sIII")
ENTRY_V1 = struct.Struct("<QB3sII")  # hash, stage, reserved, offset, length
ENTRY_V2 = struct.Struct("<QB3sIQ")  # hash, stage, reserved, length, offset
_REAL_U32_MAX = 0xFFFFFFFF
U32_MAX = _REAL_U32_MAX
CHUNK = 8 * 1024 * 1024
TOC_CHUNK_ENTRIES = 4096
REFLECT_SCHEMA = "playgta6-stage-reflection"
SUPPORTED_REFLECT_VERSIONS = (1, 2, 3)
STAGE_CODE_TO_NAME: dict[int, str] = {1: "vertex", 2: "fragment", 3: "compute"}
MAX_CONFLICT_SAMPLES = 10

WGSL_MINIFY_MODES: tuple[str, ...] = ("line", "compact", "aggressive")

# W3C WGSL §3.1 line breaks: LF, VT, FF, CR, NEL, LS, PS.
_WGSL_LINE_BREAK_CHARS = "\n\r\x0b\x0c\x85\u2028\u2029"
_WGSL_BLOCK_COMMENT_EDGE_RE = re.compile(r"/\*|\*/")

# Single-pass WGSL lexer (W3C WGSL §3). Order is load-bearing:
#   1. comments (`//`, `/*`)
#   2. line breaks & horizontal blankspace
#   3. numeric literals (hex float, hex int, dec float, dec int)
#   4. identifiers / keywords (ASCII + Unicode XID word chars)
#   5. 3-char and 2-char operators
#   6. single-char punctuation
_WGSL_LEX_RE = re.compile(
    r"(?P<lcomment>//[^\n\r\x0b\x0c\x85\u2028\u2029]*)"
    r"|(?P<bcomment>/\*)"
    r"|(?P<nl>\r\n|[\n\r\x0b\x0c\x85\u2028\u2029])"
    r"|(?P<ws>[ \t\u200e\u200f]+)"
    r"|(?P<tok>"
    r"0[xX](?:[0-9a-fA-F]*\.[0-9a-fA-F]+|[0-9a-fA-F]+\.[0-9a-fA-F]*)(?:[pP][+-]?[0-9]+)?[fh]?"
    r"|0[xX][0-9a-fA-F]+[pP][+-]?[0-9]+[fh]?"
    r"|0[xX][0-9a-fA-F]+[iu]?"
    r"|(?:[0-9]*\.[0-9]+|[0-9]+\.[0-9]*)(?:[eE][+-]?[0-9]+)?[fh]?"
    r"|[0-9]+[eE][+-]?[0-9]+[fh]?"
    r"|[0-9]+[iufh]?"
    r"|[^\W\d]\w*"
    r"|>>=|<<=|->|>>|<<|>=|<=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|\+\+|--"
    r"|[^\s]"
    r")"
)

# Adjacent single-char / edge-char pairs that would fuse into a different WGSL operator or comment.
_WGSL_HAZARDOUS_PAIRS: frozenset[str] = frozenset(
    {
        "->",
        ">>",
        "<<",
        ">=",
        "<=",
        "==",
        "!=",
        "&&",
        "||",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        "++",
        "--",
        "/*",
        "//",
        "*/",
    }
)

_WGSL_GLOBAL_DIRECTIVES: frozenset[str] = frozenset({"enable", "requires", "diagnostic"})

# C-speed regexes for semantic-preserving WGSL compaction (W3C WGSL §3).
_WGSL_LINE_COMMENT_RE = re.compile(r"//[^\n\r\x0b\x0c\x85\u2028\u2029]*")
_WGSL_LINE_BREAK_RE = re.compile(r"\r\n|[\n\r\x0b\x0c\x85\u2028\u2029]")
_WGSL_TRIM_LINES_RE = re.compile(r"(?:^[ \t\u200e\u200f]+|[ \t\u200e\u200f]+$)", re.MULTILINE)
_WGSL_MULTI_NL_RE = re.compile(r"\n{2,}")

# Matches a single ASCII space `" "` sitting between two non-whitespace characters (a, b) where
# `_wgsl_tokens_need_space(a, b)` is strictly False. Starting with a literal `" "` triggers CPython
# SRE's `memchr` literal-prefix skip, and every lookbehind is fixed-width 2 chars (O(1) in C).
_WGSL_NON_ASCII_HWS_RE = re.compile(r"[ \t\u200e\u200f]+")
_WGSL_SAFE_DELETE_SPACE_RE = re.compile(
    r" (?:"
    r"(?<=[^\W\d] )(?![\w@\s])"
    r"|(?<=[^\w.\-><=!&|+*/%^)\s] )"
    r"|(?<=\d )(?![\w.@\s])"
    r"|(?<== )(?![=\s])"
    r"|(?<=\) )(?![@]|[^\W\d]|\s)"
    r"|(?<=\+ )(?![+=\s])"
    r"|(?<=- )(?![>=\-\s])"
    r"|(?<=\* )(?![=/\s])"
    r"|(?<=/ )(?![=*/\s])"
    r"|(?<=> )(?![>=]|[^\W\d]|\s)"
    r"|(?<=< )(?![<=\s])"
    r"|(?<=& )(?![&=\s])"
    r"|(?<=\| )(?![|=\s])"
    r"|(?<=! )(?![=\s])"
    r"|(?<=\. )(?![\w\s])"
    r"|(?<=% )(?![=\s])"
    r"|(?<=\^ )(?![=\s])"
    r")"
)
_WGSL_SAFE_DELETE_SPACE_ASCII_RE = re.compile(
    r" (?:"
    r"(?<=[A-Za-z_] )(?![A-Za-z0-9_@\s])"
    r"|(?<=[^A-Za-z0-9_.\-><=!&|+*/%^)\s] )"
    r"|(?<=[0-9] )(?![A-Za-z0-9_.@\s])"
    r"|(?<== )(?![=\s])"
    r"|(?<=\) )(?![@A-Za-z_\s])"
    r"|(?<=\+ )(?![+=\s])"
    r"|(?<=- )(?![>=\-\s])"
    r"|(?<=\* )(?![=/\s])"
    r"|(?<=/ )(?![=*/\s])"
    r"|(?<=> )(?![>=A-Za-z_\s])"
    r"|(?<=< )(?![<=\s])"
    r"|(?<=& )(?![&=\s])"
    r"|(?<=\| )(?![|=\s])"
    r"|(?<=! )(?![=\s])"
    r"|(?<=\. )(?![A-Za-z0-9_\s])"
    r"|(?<=% )(?![=\s])"
    r"|(?<=\^ )(?![=\s])"
    r")",
    re.ASCII,
)
_WGSL_COMMENT_START_RE = re.compile(r"//|/\*")
# Leading global directives at top of a minified module (`enable ...;`, `requires ...;`, `diagnostic(...);`).
_WGSL_LEADING_DIRECTIVE_RE = re.compile(
    r"^(?:enable|requires|diagnostic)\b[^;]*;"
)


def _import_sibling_tool(module_name: str) -> Any:
    """Import a sibling script in `tools/` whether invoked as a script or package."""
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        tools_dir = str(Path(__file__).resolve().parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        return importlib.import_module(module_name)


def _sanitize_label(source: str | Path | None) -> str:
    """Return a strictly sanitized basename label (no host paths, drive letters, or usernames)."""
    if source is None:
        return "<unknown>"
    raw = str(source).strip()
    if not raw:
        return "<unknown>"
    if raw == "-":
        return "<stdin>"
    if raw.startswith("<") and raw.endswith(">"):
        return raw
    stripped = raw.rstrip("/\\")
    if not stripped:
        return "<unknown>"
    parts = re.split(r"[/\\]+", stripped)
    base = parts[-1]
    if not base or (len(base) == 2 and base[1] == ":"):
        return "<unknown>"
    return base


def _format_os_error(exc: OSError, fallback_path: str | Path | None = None) -> str:
    target = exc.filename if exc.filename is not None else fallback_path
    label = _sanitize_label(target)
    reason = exc.strerror or type(exc).__name__
    return f"{label}: {reason}"


# ---------------------------------------------------------------------------
# Semantic-preserving WGSL lexer & minifier (W3C WGSL §3)
# ---------------------------------------------------------------------------


def _skip_nested_block_comment(source: str, start_after_open: int) -> tuple[int, bool]:
    """Advance past a WGSL nested block comment starting just after the opening `/*`.

    Returns `(end_pos, had_line_break)`. Raises `ValueError` if unterminated.
    """
    depth = 1
    pos = start_after_open
    while depth > 0:
        m = _WGSL_BLOCK_COMMENT_EDGE_RE.search(source, pos)
        if m is None:
            raise ValueError("unterminated WGSL block comment")
        if m.group(0) == "/*":
            depth += 1
        else:
            depth -= 1
        pos = m.end()
    body = source[start_after_open : pos - 2]
    had_nl = any(ch in body for ch in _WGSL_LINE_BREAK_CHARS)
    return pos, had_nl


def _is_wgsl_word_char(ch: str) -> bool:
    return ch == "_" or ch.isalnum()


def _wgsl_tokens_need_space(prev: str, curr: str) -> bool:
    """Return True if concatenating WGSL tokens `prev` and `curr` without whitespace is unsafe.

    Enforces both W3C WGSL lexical token separation AND repository-specific downstream parser
    compatibility guards (`var<...> ident`, `@attr(...) @attr(...) ident`, `@compute @workgroup_size`).
    """
    a = prev[-1]
    b = curr[0]

    # 1. Word + Word (identifiers, keywords, numeric literals & suffixes, `_`).
    if _is_wgsl_word_char(a) and _is_wgsl_word_char(b):
        return True

    # 2. Decimal dot boundaries: avoid fusing `1 . 0`, `. 5`, or trailing-dot float `1.` + `f`/`e`/ident.
    if (a == "." and _is_wgsl_word_char(b)) or (a.isdigit() and b == "."):
        return True

    # 3. Multi-character operator / comment formation hazards (`->`, `>>`, `<<`, `--`, `++`, `/*`, `//`, ...).
    if (a + b) in _WGSL_HAZARDOUS_PAIRS:
        return True

    # 4. Repo-compat Guard A: preserve space after `>` before identifier (e.g. `var<uniform> cb0`,
    #    `var<storage,read> buf`) required by render_state_packets.js, wgpu_worker.js, and layout_model.h.
    if a == ">" and (b == "_" or b.isalpha()):
        return True

    # 5. Repo-compat Guard B: preserve space after `)` before `@` or identifier (e.g.
    #    `@group(0) @binding(0) var<uniform>`, `@workgroup_size(64,1,1) fn cs_main`) and between
    #    word attribute and `@` (e.g. `@compute @workgroup_size(...)`) required by draw_executor.js.
    if a == ")" and (b == "@" or b == "_" or b.isalpha()):
        return True
    if _is_wgsl_word_char(a) and b == "@":
        return True

    return False


def tokenize_wgsl(source: str) -> list[str]:
    """Lex WGSL source into its exact sequence of non-whitespace, non-comment tokens."""
    tokens: list[str] = []
    pos = 0
    n = len(source)
    lex_match = _WGSL_LEX_RE.match
    while pos < n:
        m = lex_match(source, pos)
        if m is None:  # pragma: no cover - `_WGSL_LEX_RE` covers every code point
            raise ValueError(f"unexpected character at offset {pos}")
        kind = m.lastgroup
        if kind == "tok":
            tokens.append(m.group(0))
            pos = m.end()
        elif kind == "bcomment":
            pos, _had_nl = _skip_nested_block_comment(source, m.end())
        else:
            pos = m.end()
    return tokens


def _strip_comments_fast(source: str) -> str:
    """Strip WGSL `//` line comments and nested `/* ... */` block comments cleanly.

    Fast path (when `/*` is absent, >99.9% of shaders): single C-engine regex pass.
    Slow path (when `/*` is present): tracks nested block comment depth and inserts a space or
    newline wherever a removed block comment could otherwise fuse adjacent tokens.
    """
    if "/*" not in source:
        if "//" in source:
            return _WGSL_LINE_COMMENT_RE.sub("", source)
        return source

    parts: list[str] = []
    pos = 0
    n = len(source)
    while pos < n:
        m = _WGSL_COMMENT_START_RE.search(source, pos)
        if m is None:
            parts.append(source[pos:])
            break
        start = m.start()
        if start > pos:
            parts.append(source[pos:start])
        if m.group(0) == "//":
            nl_m = _WGSL_LINE_BREAK_RE.search(source, m.end())
            pos = n if nl_m is None else nl_m.start()
        else:
            # Consume one or more consecutive `/* ... */` block comments cleanly.
            end_pos = start
            had_nl = False
            while end_pos + 1 < n and source[end_pos : end_pos + 2] == "/*":
                end_pos, chunk_nl = _skip_nested_block_comment(source, end_pos + 2)
                had_nl = had_nl or chunk_nl
            if had_nl:
                parts.append("\n")
            else:
                prev_ch = parts[-1][-1] if parts and parts[-1] else ""
                next_ch = source[end_pos] if end_pos < n else ""
                if (
                    prev_ch
                    and next_ch
                    and not prev_ch.isspace()
                    and not next_ch.isspace()
                    and _wgsl_tokens_need_space(prev_ch, next_ch)
                ):
                    parts.append(" ")
            pos = end_pos
    return "".join(parts)


def _compact_horizontal_ws(text: str) -> str:
    """Collapse redundant horizontal whitespace in C speed while preserving WGSL & repo token guards."""
    if "  " in text or "\t" in text or "\u200e" in text or "\u200f" in text:
        text = _WGSL_NON_ASCII_HWS_RE.sub(" ", text)
    if text.isascii():
        return _WGSL_SAFE_DELETE_SPACE_ASCII_RE.sub("", text)
    return _WGSL_SAFE_DELETE_SPACE_RE.sub("", text)


def minify_wgsl(source: str, *, mode: str = "compact") -> str:
    """Semantically compact WGSL source while preserving tokens and downstream parser contracts.

    Modes:
      - `line`: strip `//` and nested `/* ... */` comments, leading indentation, trailing
        whitespace, and blank lines; keep internal horizontal spacing and `\\n` line breaks.
      - `compact` (default): `line` plus collapse redundant horizontal whitespace between tokens
        using WGSL lexical + repo parser compatibility guards; keep `\\n` between non-empty lines.
      - `aggressive`: `compact` plus collapse safe `\\n` line breaks, preserving `\\n` after
        top-level global directives (`enable`, `requires`, `diagnostic`) for prelude injection.
    """
    if mode not in WGSL_MINIFY_MODES:
        raise ValueError(f"unsupported WGSL minify mode {mode!r} (expected one of {WGSL_MINIFY_MODES})")
    if not source:
        return ""

    # Step 1: Strip comments (`//` and nested `/* ... */`).
    cleaned = _strip_comments_fast(source)

    # Step 2: Normalize all WGSL Unicode line breaks to `\n`, trim leading/trailing horizontal WS,
    # and collapse blank lines (all in C-engine regexes).
    if "\r" in cleaned or "\x0b" in cleaned or "\x0c" in cleaned or "\x85" in cleaned or "\u2028" in cleaned or "\u2029" in cleaned:
        cleaned = _WGSL_LINE_BREAK_RE.sub("\n", cleaned)
    cleaned = _WGSL_TRIM_LINES_RE.sub("", cleaned)
    cleaned = _WGSL_MULTI_NL_RE.sub("\n", cleaned).strip("\n")
    if not cleaned:
        return ""

    if mode == "line":
        return cleaned

    if mode == "compact":
        return _compact_horizontal_ws(cleaned)

    # `aggressive` mode: collapse `\n` into minimal token separators while keeping `\n` after
    # module-leading global directives (`enable ...;`, `requires ...;`, `diagnostic(...);`).
    flat = _compact_horizontal_ws(cleaned.replace("\n", " "))
    directives: list[str] = []
    rest = flat
    while rest:
        m = _WGSL_LEADING_DIRECTIVE_RE.match(rest)
        if m is None:
            break
        directives.append(m.group(0))
        rest = rest[m.end() :].lstrip(" ")
    if not directives:
        return flat
    if not rest:
        return "\n".join(directives)
    return "\n".join(directives) + "\n" + rest


# ---------------------------------------------------------------------------
# WGPK v1/v2 TOC reading, sidecar loading, and streaming verification
# ---------------------------------------------------------------------------


def _read_exact(fh: Any, size: int) -> bytes:
    """Read up to `size` bytes from `fh`, looping across short reads until EOF or full."""
    if size <= 0:
        return b""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = fh.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_header_and_toc(path: Path) -> tuple[int, int, int, list[tuple[int, int, int, int]]]:
    """Return `(version, total_bytes, data_start, [(hash, stage_code, offset, length)])` in bounded TOC chunks."""
    label = _sanitize_label(path)
    out: list[tuple[int, int, int, int]] = []
    seen: set[int] = set()
    non_empty_ranges: list[tuple[int, int, int]] = []
    try:
        with path.open("rb") as f:
            total = f.seek(0, os.SEEK_END)
            f.seek(0)
            raw = _read_exact(f, HEADER.size)
            if len(raw) < HEADER.size:
                raise ValueError(f"{label}: truncated WGPK header")
            magic, version, count, _flags = HEADER.unpack(raw)
            if magic != b"WGPK" or version not in (1, 2):
                raise ValueError(f"{label}: not a WGPK v1/v2 pack")
            if version == 1 and total > _REAL_U32_MAX:
                raise ValueError(f"{label}: WGPK v1 pack size {total} exceeds u32 max")
            ent = ENTRY_V2 if version == 2 else ENTRY_V1
            table_bytes = count * ent.size
            data_start = HEADER.size + table_bytes
            if version == 1 and data_start > _REAL_U32_MAX:
                raise ValueError(f"{label}: WGPK v1 TOC exceeds u32 max")
            if total < data_start:
                raise ValueError(f"{label}: truncated entry table")

            remaining = count
            idx = 0
            while remaining > 0:
                batch = min(TOC_CHUNK_ENTRIES, remaining)
                want = batch * ent.size
                chunk = _read_exact(f, want)
                if len(chunk) != want:
                    raise ValueError(f"{label}: truncated entry table")
                for local_i in range(batch):
                    if version == 2:
                        h, stage, _r, length, offset = ent.unpack_from(chunk, local_i * ent.size)
                    else:
                        h, stage, _r, offset, length = ent.unpack_from(chunk, local_i * ent.size)
                    if stage not in STAGE_CODE_TO_NAME:
                        raise ValueError(f"{label}: invalid stage code {stage} at entry {idx}")
                    end = offset + length
                    if offset < data_start or end > total:
                        raise ValueError(f"{label}: entry {idx} payload exceeds pack bounds")
                    if h in seen:
                        raise ValueError(f"{label}: duplicate hash 0x{h:016x}")
                    seen.add(h)
                    if length > 0:
                        non_empty_ranges.append((offset, end, idx))
                    out.append((h, stage, offset, length))
                    idx += 1
                remaining -= batch
    except OSError as exc:
        raise ValueError(_format_os_error(exc, path)) from exc

    if len(non_empty_ranges) > 1:
        non_empty_ranges.sort(key=lambda item: (item[0], item[1], item[2]))
        prev_off, prev_end, prev_idx = non_empty_ranges[0]
        for curr_off, curr_end, curr_idx in non_empty_ranges[1:]:
            if curr_off < prev_end:
                raise ValueError(
                    f"{label}: overlapping payload ranges between entry {prev_idx} "
                    f"[{prev_off}..{prev_end}) and entry {curr_idx} [{curr_off}..{curr_end})"
                )
            if curr_end > prev_end:
                prev_off, prev_end, prev_idx = curr_off, curr_end, curr_idx

    return version, total, data_start, out


def read_toc(path: Path) -> list[tuple[int, int, int, int]]:
    """Return [(hash, stage_code, offset, length)] after bounds and overlap checks (header + TOC only)."""
    _version, _total, _data_start, entries = _read_header_and_toc(path)
    return entries


def load_sidecar(pack: Path, *, sidecar_path: Path | None = None) -> dict[str, Any] | None:
    side = sidecar_path if sidecar_path is not None else pack.with_name(pack.name + ".reflect.json")
    if not side.exists():
        return None
    label = _sanitize_label(side)
    try:
        with side.open(encoding="utf-8-sig") as f:
            doc = json.load(f)
    except OSError as exc:
        raise ValueError(_format_os_error(exc, side)) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: invalid JSON (line {exc.lineno}, col {exc.colno}): {exc.msg}") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("stages"), dict):
        raise ValueError(f"{label}: not a stage reflection sidecar")
    schema = doc.get("schema")
    if schema is not None and schema != REFLECT_SCHEMA:
        raise ValueError(f"{label}: unsupported reflection sidecar schema")
    version = doc.get("version")
    if version is not None:
        if isinstance(version, bool) or not isinstance(version, int) or version not in SUPPORTED_REFLECT_VERSIONS:
            raise ValueError(f"{label}: unsupported reflection sidecar version {version!r}")
    return doc


def resolved_bindings(stage_obj: Any) -> int:
    if not isinstance(stage_obj, dict):
        return 0
    return sum(1 for b in stage_obj.get("bindings", []) if isinstance(b, dict) and "source" in b)


def _sidecar_stages_by_hash(doc: dict[str, Any] | None, label: str) -> dict[int, Any]:
    if not doc:
        return {}
    stages = doc.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(f"{label}: not a stage reflection sidecar")
    out: dict[int, Any] = {}
    for k, v in stages.items():
        if not isinstance(k, str):
            raise ValueError(f"{label}: non-string stage hash key in sidecar")
        try:
            h = int(k.strip(), 16)
        except ValueError as exc:
            raise ValueError(f"{label}: invalid stage hash key in sidecar") from exc
        if h < 0 or h > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"{label}: stage hash key out of u64 range")
        out[h] = v
    return out


def _read_slice_utf8(fh: Any, offset: int, length: int, *, label: str, shader_hash: int) -> str:
    fh.seek(offset)
    raw = _read_exact(fh, length)
    if len(raw) != length:
        raise ValueError(f"{label}: short read while reading 0x{shader_hash:016x}")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label}: payload 0x{shader_hash:016x} is not valid UTF-8: {exc.reason}") from exc


def _stream_slice_sha256_and_utf8(
    fh: Any,
    offset: int,
    length: int,
    *,
    label: str,
    shader_hash: int,
    check_utf8: bool,
) -> str:
    fh.seek(offset)
    hasher = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")() if check_utf8 else None
    left = length
    while left > 0:
        chunk = fh.read(min(CHUNK, left))
        if not chunk:
            raise ValueError(f"{label}: short read while verifying 0x{shader_hash:016x}")
        hasher.update(chunk)
        if decoder is not None:
            try:
                decoder.decode(chunk, final=False)
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{label}: payload 0x{shader_hash:016x} is not valid UTF-8: {exc.reason}"
                ) from exc
        left -= len(chunk)
    if decoder is not None:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"{label}: payload 0x{shader_hash:016x} is not valid UTF-8: {exc.reason}"
            ) from exc
    return hasher.hexdigest()


def verify_merged_pack(
    out: Path,
    pack_paths: list[Path] | None = None,
    *,
    minify_mode: str | None = None,
    sidecar_path: Path | None = None,
    display_label: str | None = None,
) -> dict[str, Any]:
    """Run streaming structural, ascending-hash, UTF-8, payload-parity, and sidecar checks on `out`.

    Also cross-checks the written pack and sidecar with `stage_reflection_sidecar.py`
    (`load_wgpk_toc_with_meta` + `verify_sidecar_against_pack`).
    """
    if minify_mode is not None and minify_mode not in WGSL_MINIFY_MODES:
        raise ValueError(f"unsupported WGSL minify mode {minify_mode!r}")

    out_label = _sanitize_label(display_label if display_label is not None else out)
    version, total_bytes, _data_start, toc = _read_header_and_toc(out)

    # Verify strictly ascending hash order.
    for i in range(1, len(toc)):
        if toc[i - 1][0] >= toc[i][0]:
            raise ValueError(
                f"{out_label}: hashes not strictly ascending at entry {i} "
                f"(0x{toc[i - 1][0]:016x} >= 0x{toc[i][0]:016x})"
            )

    # Cross-check binary TOC with stage_reflection_sidecar.py's streaming WGPK v1/v2 validator.
    sidecar_tool = _import_sibling_tool("stage_reflection_sidecar")
    srs_entries, srs_meta = sidecar_tool.load_wgpk_toc_with_meta(out)
    if len(srs_entries) != len(toc):
        raise ValueError(
            f"{out_label}: stage_reflection_sidecar TOC count mismatch "
            f"({len(srs_entries)} vs {len(toc)})"
        )

    # Stream every payload sequentially (by ascending offset) to validate UTF-8 and compute SHA-256.
    merged_hashes: set[int] = set()
    merged_by_hash: dict[int, tuple[int, int, int]] = {}
    read_plan: list[tuple[int, int, int]] = []
    for h, stage, offset, length in toc:
        merged_hashes.add(h)
        merged_by_hash[h] = (stage, offset, length)
        read_plan.append((offset, length, h))
    read_plan.sort(key=lambda item: item[0])

    merged_digests: dict[int, str] = {}
    try:
        with out.open("rb") as fh:
            for offset, length, h in read_plan:
                if minify_mode is not None:
                    text = _read_slice_utf8(fh, offset, length, label=out_label, shader_hash=h)
                    if minify_wgsl(text, mode=minify_mode) != text:
                        raise ValueError(
                            f"{out_label}: minified payload 0x{h:016x} is not idempotent under mode {minify_mode!r}"
                        )
                    merged_digests[h] = hashlib.sha256(text.encode("utf-8")).hexdigest()
                else:
                    merged_digests[h] = _stream_slice_sha256_and_utf8(
                        fh,
                        offset,
                        length,
                        label=out_label,
                        shader_hash=h,
                        check_utf8=True,
                    )
    except OSError as exc:
        raise ValueError(_format_os_error(exc, out)) from exc

    effective_sidecar_path = sidecar_path if sidecar_path is not None else out.with_name(out.name + ".reflect.json")
    out_sidecar = load_sidecar(out, sidecar_path=effective_sidecar_path)
    out_sidecar_label = _sanitize_label(
        f"{out_label}.reflect.json" if display_label is not None else effective_sidecar_path
    )
    out_sidecar_map = _sidecar_stages_by_hash(out_sidecar, out_sidecar_label) if out_sidecar is not None else None
    sidecar_verification: dict[str, Any] | None = None
    if out_sidecar is not None and out_sidecar_map is not None:
        dangling = sorted(set(out_sidecar_map) - merged_hashes)
        if dangling:
            raise ValueError(
                f"{out_sidecar_label}: {len(dangling)} dangling stage hash(es) not in {out_label} "
                f"(first: 0x{dangling[0]:016x})"
            )
        sidecar_verification = sidecar_tool.verify_sidecar_against_pack(
            srs_entries,
            out_sidecar["stages"],
            pack_label=out_label,
            pack_meta=srs_meta,
        )
        if sidecar_verification["stage_mismatches"] > 0:
            first_sample = sidecar_verification["stage_mismatch_samples"][0]
            raise ValueError(
                f"{out_sidecar_label}: {sidecar_verification['stage_mismatches']} stage mismatch(es) "
                f"against {out_label} (first: {first_sample})"
            )
        if sidecar_verification["extra_reflection_hashes"]["count"] > 0:
            first_extra = sidecar_verification["extra_reflection_hashes"]["samples"][0]
            raise ValueError(
                f"{out_sidecar_label}: {sidecar_verification['extra_reflection_hashes']['count']} "
                f"extra reflection hash(es) not in {out_label} (first: {first_extra})"
            )

    inputs_checked = 0
    if pack_paths is not None:
        inputs_checked = len(pack_paths)
        tocs = [read_toc(p) for p in pack_paths]
        sidecars = [load_sidecar(p) for p in pack_paths]
        side_by_hash = [
            _sidecar_stages_by_hash(sc, _sanitize_label(p.with_name(p.name + ".reflect.json")))
            for p, sc in zip(pack_paths, sidecars)
        ]

        expected: dict[int, tuple[tuple[int, int, int], int, int, int, int]] = {}
        for idx, input_toc in enumerate(tocs):
            for h, stage, offset, length in input_toc:
                entry = side_by_hash[idx].get(h)
                rank = (1 if entry is not None else 0, resolved_bindings(entry), -idx)
                cur = expected.get(h)
                if cur is None or rank > cur[0]:
                    expected[h] = (rank, idx, stage, offset, length)

        expected_hashes = set(expected)
        missing = sorted(expected_hashes - merged_hashes)
        if missing:
            raise ValueError(
                f"{out_label}: missing {len(missing)} input hash(es) (first: 0x{missing[0]:016x})"
            )
        extra = sorted(merged_hashes - expected_hashes)
        if extra:
            raise ValueError(
                f"{out_label}: unexpected {len(extra)} extra hash(es) (first: 0x{extra[0]:016x})"
            )

        # Group winning slices by input pack for sequential streaming SHA-256 verification.
        slices_by_input: dict[int, list[tuple[int, int, int]]] = {}
        for h, (_rank, idx, exp_stage, exp_off, exp_len) in expected.items():
            got_stage, _got_off, got_len = merged_by_hash[h]
            if got_stage != exp_stage:
                raise ValueError(
                    f"{out_label}: stage mismatch for 0x{h:016x} (got {got_stage}, expected {exp_stage})"
                )
            if minify_mode is None and got_len != exp_len:
                raise ValueError(
                    f"{out_label}: length mismatch for 0x{h:016x} (got {got_len}, expected {exp_len})"
                )
            slices_by_input.setdefault(idx, []).append((exp_off, exp_len, h))

        for idx, items in slices_by_input.items():
            items.sort(key=lambda item: item[0])
            in_path = pack_paths[idx]
            in_label = _sanitize_label(in_path)
            try:
                with in_path.open("rb") as fh:
                    for offset, length, h in items:
                        if minify_mode is not None:
                            src_text = _read_slice_utf8(fh, offset, length, label=in_label, shader_hash=h)
                            exp_bytes = minify_wgsl(src_text, mode=minify_mode).encode("utf-8")
                            got_len = merged_by_hash[h][2]
                            if got_len != len(exp_bytes):
                                raise ValueError(
                                    f"{out_label}: minified length mismatch for 0x{h:016x} "
                                    f"(got {got_len}, expected {len(exp_bytes)})"
                                )
                            exp_digest = hashlib.sha256(exp_bytes).hexdigest()
                        else:
                            exp_digest = _stream_slice_sha256_and_utf8(
                                fh,
                                offset,
                                length,
                                label=in_label,
                                shader_hash=h,
                                check_utf8=False,
                            )
                        if exp_digest != merged_digests[h]:
                            raise ValueError(f"{out_label}: payload hash mismatch for 0x{h:016x}")
            except OSError as exc:
                raise ValueError(_format_os_error(exc, in_path)) from exc

        docs = [sc for sc in sidecars if sc]
        if docs:
            if out_sidecar is None or out_sidecar_map is None:
                raise ValueError(f"{out_sidecar_label}: missing merged reflection sidecar")
            expected_ver = max(int(d.get("version", 1)) for d in docs)
            got_ver = int(out_sidecar.get("version", 1))
            if got_ver != expected_ver:
                raise ValueError(
                    f"{out_sidecar_label}: sidecar version mismatch (got {got_ver}, expected {expected_ver})"
                )
            expected_sidecar: dict[int, Any] = {}
            for h, (_rank, idx, _st, _off, _ln) in expected.items():
                entry = side_by_hash[idx].get(h)
                if entry is not None:
                    expected_sidecar[h] = entry
            if set(out_sidecar_map) != set(expected_sidecar):
                raise ValueError(
                    f"{out_sidecar_label}: sidecar stage set mismatch "
                    f"(got {len(out_sidecar_map)}, expected {len(expected_sidecar)})"
                )
            for h, exp_entry in expected_sidecar.items():
                if out_sidecar_map[h] != exp_entry:
                    raise ValueError(f"{out_sidecar_label}: sidecar entry mismatch for 0x{h:016x}")

    result: dict[str, Any] = {
        "source": out_label,
        "pack_version": version,
        "pack_bytes": total_bytes,
        "entries_verified": len(toc),
        "sidecar_entries_verified": len(out_sidecar_map) if out_sidecar_map is not None else None,
        "inputs_checked": inputs_checked,
        "verified": True,
    }
    if minify_mode is not None:
        result["minify_mode"] = minify_mode
    if sidecar_verification is not None:
        result["sidecar_verification"] = {
            "pack_version": sidecar_verification["pack_version"],
            "is_wgpk_v2": sidecar_verification["is_wgpk_v2"],
            "matched_hashes": sidecar_verification["matched_hashes"],
            "missing_reflection_count": sidecar_verification["missing_reflection_hashes"]["count"],
            "extra_reflection_count": sidecar_verification["extra_reflection_hashes"]["count"],
            "stage_mismatches": sidecar_verification["stage_mismatches"],
            "coverage_pct": sidecar_verification["coverage_pct"],
        }
    return result


# ---------------------------------------------------------------------------
# Streaming WGPK v1/v2 merge + optional semantic-preserving WGSL minification
# ---------------------------------------------------------------------------


def _safe_unlink(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _pack_toc_entry(version: int, h: int, stage: int, offset: int, length: int) -> bytes:
    if version == 2:
        return ENTRY_V2.pack(h, stage, b"\0\0\0", length, offset)
    return ENTRY_V1.pack(h, stage, b"\0\0\0", offset, length)


def _write_merged_pack_raw(
    pack_paths: list[Path],
    order: list[int],
    best: dict[int, tuple[tuple[int, int, int], int, int, int, int]],
    tmp_pack: Path,
    *,
    force_v2: bool,
) -> tuple[int, int, int]:
    """Stream unminified winning payloads into `tmp_pack`, returning `(version, pack_bytes, payload_bytes)`."""
    total_payload = sum(best[h][4] for h in order)
    v1_size = HEADER.size + len(order) * ENTRY_V1.size + total_payload
    version = 2 if (force_v2 or v1_size > U32_MAX) else 1
    ent = ENTRY_V2 if version == 2 else ENTRY_V1
    data_start = HEADER.size + len(order) * ent.size

    toc_out = bytearray()
    cursor = data_start
    for h in order:
        _rank, _idx, stage, _off, length = best[h]
        toc_out += _pack_toc_entry(version, h, stage, cursor, length)
        cursor += length

    handles: list[Any] = []
    try:
        for p in pack_paths:
            handles.append(p.open("rb"))
        with tmp_pack.open("wb") as w:
            w.write(HEADER.pack(b"WGPK", version, len(order), 0))
            w.write(toc_out)
            for h in order:
                _rank, idx, _stage, offset, length = best[h]
                src = handles[idx]
                src.seek(offset)
                left = length
                while left:
                    chunk = src.read(min(CHUNK, left))
                    if not chunk:
                        raise ValueError(
                            f"{_sanitize_label(pack_paths[idx])}: short read while copying 0x{h:016x}"
                        )
                    w.write(chunk)
                    left -= len(chunk)
    except OSError as exc:
        raise ValueError(_format_os_error(exc, tmp_pack)) from exc
    finally:
        for fh in handles:
            fh.close()

    return version, data_start + total_payload, total_payload


def _write_merged_pack_minified(
    pack_paths: list[Path],
    order: list[int],
    best: dict[int, tuple[tuple[int, int, int], int, int, int, int]],
    tmp_pack: Path,
    *,
    force_v2: bool,
    minify_mode: str,
) -> tuple[int, int, int, int]:
    """Stream-minify winning payloads in bounded O(single shader) RAM.

    Returns `(version, pack_bytes, raw_payload_bytes, minified_payload_bytes)`.
    """
    raw_total_payload = sum(best[h][4] for h in order)
    # Determine whether target WGPK version is known before minifying:
    # - `force_v2=True` -> always v2 (single-pass placeholder TOC seek-back, zero extra disk copy).
    # - Raw v1 size already fits in U32_MAX -> minified is <= raw -> always v1 (single-pass).
    # - Otherwise (raw > U32_MAX and auto-version) -> spool minified bytes to a temp payload file
    #   so a >4 GiB pack that compacts below 4 GiB automatically emits WGPK v1 cleanly!
    known_version: int | None
    if force_v2:
        known_version = 2
    elif HEADER.size + len(order) * ENTRY_V1.size + raw_total_payload <= U32_MAX:
        known_version = 1
    else:
        known_version = None

    handles: list[Any] = []
    tmp_payload: Path | None = None
    try:
        for p in pack_paths:
            handles.append(p.open("rb"))

        if known_version is not None:
            version = known_version
            ent = ENTRY_V2 if version == 2 else ENTRY_V1
            data_start = HEADER.size + len(order) * ent.size
            toc_out = bytearray()
            cursor = data_start
            minified_total = 0

            with tmp_pack.open("wb") as w:
                # Reserve space for header + TOC; patch in place after streaming payloads.
                w.write(b"\0" * data_start)
                for h in order:
                    _rank, idx, stage, offset, length = best[h]
                    in_label = _sanitize_label(pack_paths[idx])
                    src_text = _read_slice_utf8(handles[idx], offset, length, label=in_label, shader_hash=h)
                    min_bytes = minify_wgsl(src_text, mode=minify_mode).encode("utf-8")
                    if len(min_bytes) > _REAL_U32_MAX:
                        raise ValueError(f"{in_label}: minified payload 0x{h:016x} exceeds u32 max length")
                    w.write(min_bytes)
                    toc_out += _pack_toc_entry(version, h, stage, cursor, len(min_bytes))
                    cursor += len(min_bytes)
                    minified_total += len(min_bytes)

                w.seek(0)
                w.write(HEADER.pack(b"WGPK", version, len(order), 0))
                w.write(toc_out)

            return version, data_start + minified_total, raw_total_payload, minified_total

        # Auto-version path for >4 GiB raw inputs: spool minified payloads in bounded RAM.
        tmp_payload = tmp_pack.with_name(tmp_pack.name + ".payload")
        entries_meta: list[tuple[int, int, int]] = []  # (hash, stage, minified_len)
        minified_total = 0
        with tmp_payload.open("wb") as sp:
            for h in order:
                _rank, idx, stage, offset, length = best[h]
                in_label = _sanitize_label(pack_paths[idx])
                src_text = _read_slice_utf8(handles[idx], offset, length, label=in_label, shader_hash=h)
                min_bytes = minify_wgsl(src_text, mode=minify_mode).encode("utf-8")
                if len(min_bytes) > _REAL_U32_MAX:
                    raise ValueError(f"{in_label}: minified payload 0x{h:016x} exceeds u32 max length")
                sp.write(min_bytes)
                entries_meta.append((h, stage, len(min_bytes)))
                minified_total += len(min_bytes)

        v1_size = HEADER.size + len(order) * ENTRY_V1.size + minified_total
        version = 2 if v1_size > U32_MAX else 1
        ent = ENTRY_V2 if version == 2 else ENTRY_V1
        data_start = HEADER.size + len(order) * ent.size
        toc_out = bytearray()
        cursor = data_start
        for h, stage, mlen in entries_meta:
            toc_out += _pack_toc_entry(version, h, stage, cursor, mlen)
            cursor += mlen

        with tmp_payload.open("rb") as sp, tmp_pack.open("wb") as w:
            w.write(HEADER.pack(b"WGPK", version, len(order), 0))
            w.write(toc_out)
            while True:
                chunk = sp.read(CHUNK)
                if not chunk:
                    break
                w.write(chunk)

        return version, data_start + minified_total, raw_total_payload, minified_total
    except OSError as exc:
        raise ValueError(_format_os_error(exc, tmp_pack)) from exc
    finally:
        for fh in handles:
            fh.close()
        _safe_unlink(tmp_payload)


def merge(
    pack_paths: list[Path],
    out: Path,
    *,
    force_v2: bool = False,
    verify: bool = False,
    minify_wgsl_mode: str | None = None,
    fail_on_stage_conflict: bool = False,
) -> dict[str, Any]:
    if not pack_paths:
        raise ValueError("at least one input .wgsl.pack is required")
    if minify_wgsl_mode is not None and minify_wgsl_mode not in WGSL_MINIFY_MODES:
        raise ValueError(
            f"unsupported WGSL minify mode {minify_wgsl_mode!r} (expected one of {WGSL_MINIFY_MODES})"
        )

    tocs = [read_toc(p) for p in pack_paths]
    sidecars = [load_sidecar(p) for p in pack_paths]
    side_by_hash: list[dict[int, Any]] = [
        _sidecar_stages_by_hash(sc, _sanitize_label(p.with_name(p.name + ".reflect.json")))
        for p, sc in zip(pack_paths, sidecars)
    ]

    best: dict[int, tuple[tuple[int, int, int], int, int, int, int]] = {}
    stages_seen_by_hash: dict[int, set[int]] = {}
    seen_total = 0
    for idx, toc in enumerate(tocs):
        for h, stage, offset, length in toc:
            seen_total += 1
            stages_seen_by_hash.setdefault(h, set()).add(stage)
            entry = side_by_hash[idx].get(h)
            rank = (1 if entry is not None else 0, resolved_bindings(entry), -idx)
            cur = best.get(h)
            if cur is None or rank > cur[0]:
                best[h] = (rank, idx, stage, offset, length)

    stage_conflict_hashes = sorted(h for h, st_set in stages_seen_by_hash.items() if len(st_set) > 1)
    if fail_on_stage_conflict and stage_conflict_hashes:
        first_h = stage_conflict_hashes[0]
        names_seen = sorted(STAGE_CODE_TO_NAME[s] for s in stages_seen_by_hash[first_h])
        raise ValueError(
            f"cross-input stage conflict for 0x{first_h:016x} "
            f"(stages: {', '.join(names_seen)}; total conflicts: {len(stage_conflict_hashes)})"
        )

    order = sorted(best)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_pack = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    sidecar_path = out.with_name(out.name + ".reflect.json")
    tmp_sidecar: Path | None = None

    stages_out: dict[str, Any] = {}
    for h in order:
        entry = side_by_hash[best[h][1]].get(h)
        if entry is not None:
            stages_out["0x%016x" % h] = entry
    docs = [sc for sc in sidecars if sc]

    try:
        if minify_wgsl_mode is not None:
            version, pack_bytes, raw_payload_bytes, minified_payload_bytes = _write_merged_pack_minified(
                pack_paths,
                order,
                best,
                tmp_pack,
                force_v2=force_v2,
                minify_mode=minify_wgsl_mode,
            )
        else:
            version, pack_bytes, raw_payload_bytes = _write_merged_pack_raw(
                pack_paths,
                order,
                best,
                tmp_pack,
                force_v2=force_v2,
            )
            minified_payload_bytes = raw_payload_bytes

        if docs:
            merged_doc = {k: v for k, v in docs[0].items() if k != "stages"}
            merged_doc["version"] = max(int(d.get("version", 1)) for d in docs)
            merged_doc["stages"] = stages_out
            tmp_sidecar = out.with_name(f".{out.name}.reflect.json.tmp.{os.getpid()}")
            try:
                with tmp_sidecar.open("w", encoding="utf-8", newline="\n") as f:
                    json.dump(merged_doc, f, indent=1)
                    f.write("\n")
            except OSError as exc:
                raise ValueError(_format_os_error(exc, sidecar_path)) from exc

        verify_info: dict[str, Any] | None = None
        if verify:
            verify_info = verify_merged_pack(
                tmp_pack,
                pack_paths,
                minify_mode=minify_wgsl_mode,
                sidecar_path=tmp_sidecar if tmp_sidecar is not None else Path(f"{tmp_pack}.nonexistent.reflect.json"),
                display_label=_sanitize_label(out),
            )

        # Atomic commit: input handles are already closed, so in-place replacement is Windows-safe.
        try:
            os.replace(tmp_pack, out)
            if tmp_sidecar is not None:
                os.replace(tmp_sidecar, sidecar_path)
            elif sidecar_path.exists():
                # No inputs carried a sidecar: remove stale sidecar so `--verify` and loaders stay consistent.
                sidecar_path.unlink(missing_ok=True)
        except OSError as exc:
            raise ValueError(_format_os_error(exc, out)) from exc
    except BaseException:
        _safe_unlink(tmp_pack)
        _safe_unlink(tmp_sidecar)
        raise

    stage_counts = {"vertex": 0, "fragment": 0, "compute": 0}
    for h in order:
        stage_counts[STAGE_CODE_TO_NAME[best[h][2]]] += 1

    summary: dict[str, Any] = {
        "inputs": [_sanitize_label(p) for p in pack_paths],
        "input_entries": [len(t) for t in tocs],
        "input_entries_total": seen_total,
        "merged_entries": len(order),
        "duplicates_dropped": seen_total - len(order),
        "stage_conflicts": len(stage_conflict_hashes),
        "stage_counts": stage_counts,
        "sidecar_entries": len(stages_out) if docs else None,
        "pack_version": version,
        "pack_bytes": pack_bytes,
    }
    if stage_conflict_hashes:
        summary["stage_conflict_samples"] = [
            {
                "hash": f"0x{h:016x}",
                "winner_stage": STAGE_CODE_TO_NAME[best[h][2]],
                "seen_stages": sorted(STAGE_CODE_TO_NAME[s] for s in stages_seen_by_hash[h]),
            }
            for h in stage_conflict_hashes[:MAX_CONFLICT_SAMPLES]
        ]
    if minify_wgsl_mode is not None:
        bytes_saved = raw_payload_bytes - minified_payload_bytes
        reduction_pct = round((bytes_saved * 100.0) / raw_payload_bytes, 2) if raw_payload_bytes > 0 else 0.0
        summary["minify_mode"] = minify_wgsl_mode
        summary["raw_payload_bytes"] = raw_payload_bytes
        summary["minified_payload_bytes"] = minified_payload_bytes
        summary["bytes_saved"] = bytes_saved
        summary["reduction_pct"] = reduction_pct
    if verify:
        summary["verified"] = True
        if verify_info is not None and "sidecar_verification" in verify_info:
            summary["sidecar_verification"] = verify_info["sidecar_verification"]
    return summary


# ---------------------------------------------------------------------------
# Missing-shader diff integration (delegating to `tools/diff_missing_shaders.py`)
# ---------------------------------------------------------------------------


def diff_pack(
    pack_path: str | Path,
    *,
    reflect_path: str | Path | None = None,
    boot_logs: Sequence[str | Path] = (),
    missing_jsons: Sequence[str | Path] = (),
    dump_dirs: Sequence[str | Path] = (),
    emit_priority_hashes: str | Path | None = None,
) -> dict[str, Any]:
    """Diff demanded shader identities (boot logs, missing JSONs, pipeline dumps) against a WGPK v1/v2 pack."""
    dms = _import_sibling_tool("diff_missing_shaders")
    return dms.diff_missing_shaders(
        pack_path,
        reflect_path=reflect_path,
        boot_logs=boot_logs,
        missing_jsons=missing_jsons,
        dump_dirs=dump_dirs,
        emit_priority_hashes=emit_priority_hashes,
    )


def _main_diff(argv: list[str]) -> int:
    dms = _import_sibling_tool("diff_missing_shaders")
    ap = argparse.ArgumentParser(
        prog="merge_wgsl_packs.py diff",
        description="Diff demanded shaders (boot logs / missing JSON / pipeline dumps) against a WGPK v1/v2 pack.",
    )
    ap.add_argument("--pack", required=True, type=Path, help="Active .wgsl.pack (v1 or v2)")
    ap.add_argument(
        "--reflect",
        default=None,
        type=Path,
        help="Optional explicit .reflect.json sidecar path (auto-detects <pack>.reflect.json)",
    )
    ap.add_argument(
        "--boot-log",
        action="append",
        default=[],
        type=Path,
        dest="boot_logs",
        help="Chrome or native boot console log file (repeatable)",
    )
    ap.add_argument(
        "--missing-json",
        action="append",
        default=[],
        type=Path,
        dest="missing_jsons",
        help="Browser playgta6-missing-shaders-v1 or crash-report JSON file (repeatable)",
    )
    ap.add_argument(
        "--dump-dir",
        action="append",
        default=[],
        type=Path,
        dest="dump_dirs",
        help="playgta6-pipeline-dump v1 directory (repeatable; reads manifest/filenames only)",
    )
    ap.add_argument(
        "--emit-priority-hashes",
        default=None,
        type=Path,
        dest="emit_priority_hashes",
        help="Write canonically sorted missing-from-pack 0x<16-hex> hashes to file",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any demanded shader is missing from the pack or loaded reflection sidecar",
    )
    ap.add_argument("--json", action="store_true", dest="as_json", help="Emit strictly sanitized JSON")
    args = ap.parse_args(argv)

    try:
        report = diff_pack(
            args.pack,
            reflect_path=args.reflect,
            boot_logs=args.boot_logs,
            missing_jsons=args.missing_jsons,
            dump_dirs=args.dump_dirs,
            emit_priority_hashes=args.emit_priority_hashes,
        )
    except dms.DiffMissingShadersError as exc:
        if args.as_json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "schema": dms.DIFF_REPORT_SCHEMA,
                        "version": dms.DIFF_REPORT_VERSION,
                        "errorCode": exc.code,
                        "error": exc.message,
                        "source": exc.source,
                    },
                    indent=1,
                )
            )
        else:
            print(f"error: {exc.message}", file=sys.stderr)
        return 1
    except OSError as exc:
        msg = _format_os_error(exc)
        if args.as_json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "schema": dms.DIFF_REPORT_SCHEMA,
                        "version": dms.DIFF_REPORT_VERSION,
                        "errorCode": "io_error",
                        "error": msg,
                    },
                    indent=1,
                )
            )
        else:
            print(f"error: {msg}", file=sys.stderr)
        return 1

    strict_failed = bool(args.strict and not report["clean"])
    if args.as_json:
        print(json.dumps({"ok": not strict_failed, "strict": bool(args.strict), **report}, indent=1))
    else:
        print(dms.format_diff_text(report))
    return 1 if strict_failed else 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv:
        if raw_argv[0] == "diff":
            return _main_diff(raw_argv[1:])
        if "--diff" in raw_argv:
            filtered = [arg for arg in raw_argv if arg != "--diff"]
            return _main_diff(filtered)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("packs", nargs="+", type=Path, help="input .wgsl.pack files (earlier wins ties)")
    ap.add_argument("--out", required=True, type=Path, help="merged .wgsl.pack to write")
    ap.add_argument("--force-v2", action="store_true", help="emit WGPK v2 (64-bit offsets) explicitly")
    ap.add_argument(
        "--minify-wgsl",
        action="store_true",
        help="semantically minify WGSL payloads in streaming bounded RAM (default mode: compact)",
    )
    ap.add_argument(
        "--minify-mode",
        choices=WGSL_MINIFY_MODES,
        default=None,
        help="WGSL minification aggressiveness (line, compact, aggressive); implies --minify-wgsl",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="allow --out to overwrite an input pack atomically after temp write (and optional --verify)",
    )
    ap.add_argument(
        "--fail-on-stage-conflict",
        action="store_true",
        help="fail if two inputs share a 64-bit hash with different shader stage codes",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="stream-verify the written WGPK pack (bounds, order, UTF-8, payload parity, sidecar)",
    )
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(raw_argv)

    outs = args.out.resolve()
    if not args.in_place and any(p.resolve() == outs for p in args.packs):
        print("error: --out must differ from every input (pass --in-place to update atomically)", file=sys.stderr)
        return 2

    minify_mode: str | None = args.minify_mode if args.minify_mode is not None else ("compact" if args.minify_wgsl else None)

    try:
        summary = merge(
            args.packs,
            args.out,
            force_v2=args.force_v2,
            verify=args.verify,
            minify_wgsl_mode=minify_mode,
            fail_on_stage_conflict=args.fail_on_stage_conflict,
        )
    except OSError as exc:
        print(f"error: {_format_os_error(exc)}", file=sys.stderr)
        return 1
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(summary, indent=1))
    else:
        print(", ".join(f"{k}={v}" for k, v in summary.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
