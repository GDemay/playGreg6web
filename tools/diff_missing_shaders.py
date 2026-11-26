#!/usr/bin/env python3
"""Strictly sanitized missing-shader diff & WGPK v1/v2 incremental merge verification CLI.

Compares demanded shader identities from Chrome/native boot console logs, browser-exported
`playgta6-missing-shaders-v1` / crash-report JSON files, and/or `playgta6-pipeline-dump` v1
directories against an active WGPK v1/v2 `.wgsl.pack` and `.reflect.json` sidecar (v1/v2/v3).
Also verifies WGPK v1/v2 incremental merge integrity (streaming bounds, ascending hash order,
UTF-8 validity, union completeness, tie-break payload SHA-256 parity, and sidecar alignment).

Usage:
    python tools/diff_missing_shaders.py diff --pack <pack> [--reflect <reflect.json>]
        [--boot-log <log> ...] [--missing-json <report.json> ...] [--dump-dir <dir> ...]
        [--emit-priority-hashes <out.txt>] [--strict] [--json]

    python tools/diff_missing_shaders.py --pack <pack> --boot-log <log> [--json] [--strict]

    python tools/diff_missing_shaders.py verify-merge --inputs <packA> <packB> [...]
        --merged <merged.wgsl.pack> [--json]

Pure Python 3.12 stdlib only (zero external dependencies). Never emits host paths, drive
letters, usernames, raw WGSL source code, or binary shader dumps.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import re
import struct
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# WGPK & schema constants
# ---------------------------------------------------------------------------

PACK_MAGIC = b"WGPK"
HEADER_STRUCT = struct.Struct("<4sIII")
HEADER_BYTES = HEADER_STRUCT.size  # 16 bytes
ENTRY_V1_STRUCT = struct.Struct("<QB3sII")  # hash, stage, reserved, offset, length (20B)
ENTRY_V2_STRUCT = struct.Struct("<QB3sIQ")  # hash, stage, reserved, length, offset (24B)
SUPPORTED_PACK_VERSIONS = (1, 2)

U64_MAX = 0xFFFFFFFFFFFFFFFF
STREAM_CHUNK_BYTES = 8 * 1024 * 1024

REFLECT_SCHEMA = "playgta6-stage-reflection"
SUPPORTED_REFLECT_VERSIONS = (1, 2, 3)

PIPELINE_DUMP_SCHEMA = "playgta6-pipeline-dump"
PIPELINE_DUMP_SCHEMA_VERSION = 1

DIFF_REPORT_SCHEMA = "playgta6-missing-shader-diff"
DIFF_REPORT_VERSION = 1

VERIFY_MERGE_SCHEMA = "playgta6-wgpk-merge-verify"
VERIFY_MERGE_VERSION = 1

STAGE_CODE_TO_NAME: dict[int, str] = {
    1: "vertex",
    2: "fragment",
    3: "compute",
}
STAGE_ORDER: tuple[str, ...] = ("vertex", "fragment", "compute")
DEMAND_STAGE_ORDER: tuple[str, ...] = ("vertex", "fragment", "compute", "unknown")

_STAGE_ALIASES: dict[str, str] = {
    "vertex": "vertex",
    "vs": "vertex",
    "vert": "vertex",
    "1": "vertex",
    "fragment": "fragment",
    "pixel": "fragment",
    "ps": "fragment",
    "fs": "fragment",
    "frag": "fragment",
    "2": "fragment",
    "compute": "compute",
    "cs": "compute",
    "comp": "compute",
    "3": "compute",
}

_DXIL_KIND_TO_STAGE: dict[int, str] = {
    0: "fragment",
    1: "vertex",
    5: "compute",
}

_HEX16_RE = re.compile(r"^(?:0[xX])?([0-9a-fA-F]{16})$")
_STAGE_BOUND_0X_RE = re.compile(
    r"(?<![A-Za-z0-9_])(vertex|fragment|pixel|compute|vs|ps|fs|cs)"
    r"(?:\s+shader)?(?:\s*[:=_-]\s*|\s+)0[xX]([0-9a-fA-F]{16})(?![0-9a-fA-F])",
    re.IGNORECASE,
)
_STAGE_BOUND_BARE_RE = re.compile(
    r"(?<![A-Za-z0-9_])(vertex|fragment|pixel|compute|vs|ps|fs|cs)"
    r"(?:\s+shader)?(?:\s*[:=_-]\s*|\s+)([0-9a-fA-F]{16})(?![0-9a-fA-F])",
    re.IGNORECASE,
)
_EXPLICIT_0X16_RE = re.compile(r"(?<![0-9a-fA-F])0[xX]([0-9a-fA-F]{16})(?![0-9a-fA-F])")
_KEYED_BARE_HEX16_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:miss|hit|hash|identity|code_identity|shader)"
    r"(?:\s*[:=_-]\s*|\s+)([0-9a-fA-F]{16})(?![0-9a-fA-F])",
    re.IGNORECASE,
)
_BARE_HEX16_RE = re.compile(r"(?<![0-9a-fA-F])([0-9a-fA-F]{16})(?![0-9a-fA-F])")
_SHADER_CONTEXT_RE = re.compile(
    r"\b(?:miss|misses|missing|hit|hits|shader|shaders|wgsl|wgpk|pack|hash|hashes|"
    r"pipeline|pso|fallback|stage|identity|code_identity|wgpu|vertex|fragment|pixel|"
    r"compute|vs|ps|fs|cs)\b",
    re.IGNORECASE,
)
_STAGE_WORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])(vertex|fragment|pixel|compute|vs|ps|fs|cs)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_TRAILING_STAGE_RE = re.compile(
    r"^\s*(?:[\(\[\{]\s*(?:stage\s*[:=]\s*)?|(?:,\s*)?stage\s*[:=]\s*)"
    r"(vertex|fragment|pixel|compute|vs|ps|fs|cs)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_HOST_PATH_LEAK_RE = re.compile(
    r"(?:[A-Za-z]:[\\/][^\s\"'<>|]+|\\\\[^\s\"'<>|]+|"
    r"/(?:Users|home|root|tmp|var|etc|opt|mnt|Volumes|private|cygdrive)/[^\s\"'<>|]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Errors & sanitization helpers
# ---------------------------------------------------------------------------


class DiffMissingShadersError(ValueError):
    """Sanitized error raised on invalid packs, sidecars, logs, manifests, or arguments."""

    def __init__(self, message: str, *, code: str = "error", source: str | None = None) -> None:
        self.code = code
        self.source = sanitize_source_label(source) if source is not None else None
        self.message = sanitize_error_text(message)
        super().__init__(self.message)


def sanitize_source_label(source: str | Path | None) -> str:
    """Return a strictly sanitized source label (basename only; no host paths/usernames)."""
    if source is None:
        return "<memory>"
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


def sanitize_error_text(text: str) -> str:
    """Scrub any accidental host paths, UNC paths, or drive prefixes from an error string."""
    return _HOST_PATH_LEAK_RE.sub(lambda m: sanitize_source_label(m.group(0)), text)


def _format_os_error(source: str | Path | None, exc: OSError) -> str:
    label = sanitize_source_label(source if source is not None else exc.filename)
    reason = exc.strerror or type(exc).__name__
    return f"failed to read {label}: {reason}"


def normalize_shader_hash(value: int | str) -> int:
    """Normalize an integer or hex string shader hash to unsigned 64-bit int."""
    if isinstance(value, bool):
        raise ValueError("invalid shader hash")
    if isinstance(value, int):
        if value < 0 or value > U64_MAX:
            raise ValueError("shader hash out of u64 range")
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        raw_hex = trimmed[2:] if trimmed.lower().startswith("0x") else trimmed
        if not raw_hex or len(raw_hex) > 16 or not re.fullmatch(r"[0-9a-fA-F]+", raw_hex):
            raise ValueError("invalid hex shader hash")
        return int(raw_hex, 16)
    raise TypeError(f"unsupported shader hash type: {type(value).__name__}")


def format_shader_hash(shader_hash: int) -> str:
    """Format a 64-bit hash as canonical `0x` + 16 lowercase hex digits."""
    return f"0x{normalize_shader_hash(shader_hash):016x}"


def normalize_stage_name(raw: Any) -> str:
    """Normalize a stage code or name into vertex | fragment | compute | unknown."""
    if isinstance(raw, bool) or raw is None:
        return "unknown"
    if isinstance(raw, int):
        return STAGE_CODE_TO_NAME.get(raw, "unknown")
    if isinstance(raw, str):
        return _STAGE_ALIASES.get(raw.strip().lower(), "unknown")
    return "unknown"


# ---------------------------------------------------------------------------
# WGPK v1 & v2 streaming TOC reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WgpkTocEntry:
    hash: int
    stage_code: int
    stage: str
    offset: int
    length: int

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.hash)


@dataclass(frozen=True)
class WgpkToc:
    source: str
    version: int
    flags: int
    byte_length: int
    data_start: int
    entries: tuple[WgpkTocEntry, ...]

    @property
    def entry_count(self) -> int:
        return len(self.entries)

    @property
    def hashes(self) -> frozenset[int]:
        return frozenset(e.hash for e in self.entries)

    def by_hash(self) -> dict[int, WgpkTocEntry]:
        return {e.hash: e for e in self.entries}

    def stage_counts(self) -> dict[str, int]:
        counts = {s: 0 for s in STAGE_ORDER}
        for e in self.entries:
            counts[e.stage] += 1
        return counts


def _read_exact(fh: Any, count: int) -> bytes:
    if count <= 0:
        return b""
    chunks: list[bytes] = []
    left = count
    while left > 0:
        chunk = fh.read(left)
        if not chunk:
            break
        chunks.append(chunk)
        left -= len(chunk)
    if not chunks:
        return b""
    return chunks[0] if len(chunks) == 1 else b"".join(chunks)


def read_wgpk_toc(path: str | Path, *, source_label: str | None = None) -> WgpkToc:
    """Stream and validate only header + TOC of a WGPK v1 or v2 `.wgsl.pack` file."""
    p = Path(path)
    label = sanitize_source_label(source_label if source_label is not None else p)
    if not p.exists():
        raise DiffMissingShadersError(
            f"WGPK pack not found: {label}",
            code="pack_not_found",
            source=label,
        )
    if not p.is_file():
        raise DiffMissingShadersError(
            f"WGPK pack path is not a file: {label}",
            code="invalid_pack",
            source=label,
        )

    try:
        with p.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            file_size = fh.tell()
            fh.seek(0)
            if file_size < HEADER_BYTES:
                raise DiffMissingShadersError(
                    f"{label}: truncated WGPK header ({file_size} bytes)",
                    code="truncated_header",
                    source=label,
                )
            header_raw = _read_exact(fh, HEADER_BYTES)
            if len(header_raw) < HEADER_BYTES:
                raise DiffMissingShadersError(
                    f"{label}: short read on WGPK header",
                    code="truncated_header",
                    source=label,
                )
            magic, version, entry_count, flags = HEADER_STRUCT.unpack(header_raw)
            if magic != PACK_MAGIC:
                raise DiffMissingShadersError(
                    f"{label}: invalid WGPK magic (expected b'WGPK')",
                    code="invalid_magic",
                    source=label,
                )
            if version not in SUPPORTED_PACK_VERSIONS:
                raise DiffMissingShadersError(
                    f"{label}: unsupported WGPK version {version} (expected 1 or 2)",
                    code="unsupported_version",
                    source=label,
                )

            ent_struct = ENTRY_V2_STRUCT if version == 2 else ENTRY_V1_STRUCT
            table_bytes = entry_count * ent_struct.size
            data_start = HEADER_BYTES + table_bytes
            if file_size < data_start:
                raise DiffMissingShadersError(
                    f"{label}: truncated entry table ({file_size} bytes < {data_start} required)",
                    code="truncated_table",
                    source=label,
                )
            table_buf = _read_exact(fh, table_bytes)
            if len(table_buf) < table_bytes:
                raise DiffMissingShadersError(
                    f"{label}: short read on entry table",
                    code="truncated_table",
                    source=label,
                )
    except DiffMissingShadersError:
        raise
    except OSError as exc:
        raise DiffMissingShadersError(_format_os_error(label, exc), code="io_error", source=label) from exc

    entries: list[WgpkTocEntry] = []
    seen_hashes: set[int] = set()
    non_empty_ranges: list[tuple[int, int, int]] = []

    for idx in range(entry_count):
        rec_off = idx * ent_struct.size
        if version == 2:
            shader_hash, stage_code, _reserved, length, offset = ent_struct.unpack_from(table_buf, rec_off)
        else:
            shader_hash, stage_code, _reserved, offset, length = ent_struct.unpack_from(table_buf, rec_off)

        stage_name = STAGE_CODE_TO_NAME.get(stage_code)
        if stage_name is None:
            raise DiffMissingShadersError(
                f"{label}: invalid shader stage code {stage_code} at entry {idx}",
                code="invalid_stage",
                source=label,
            )

        end = offset + length
        if offset < data_start or end > file_size:
            raise DiffMissingShadersError(
                f"{label}: entry {idx} payload [{offset}..+{length}] exceeds pack bounds "
                f"[{data_start}..{file_size}]",
                code="out_of_bounds",
                source=label,
            )

        if shader_hash in seen_hashes:
            raise DiffMissingShadersError(
                f"{label}: duplicate shader hash {format_shader_hash(shader_hash)} at entry {idx}",
                code="duplicate_hash",
                source=label,
            )
        seen_hashes.add(shader_hash)

        if length > 0:
            non_empty_ranges.append((offset, end, idx))

        entries.append(
            WgpkTocEntry(
                hash=shader_hash,
                stage_code=stage_code,
                stage=stage_name,
                offset=offset,
                length=length,
            )
        )

    if len(non_empty_ranges) > 1:
        non_empty_ranges.sort(key=lambda item: (item[0], item[1], item[2]))
        prev_off, prev_end, prev_idx = non_empty_ranges[0]
        for curr_off, curr_end, curr_idx in non_empty_ranges[1:]:
            if curr_off < prev_end:
                raise DiffMissingShadersError(
                    f"{label}: overlapping payload ranges between entry {prev_idx} "
                    f"[{prev_off}..{prev_end}) and entry {curr_idx} [{curr_off}..{curr_end})",
                    code="overlapping_payloads",
                    source=label,
                )
            if curr_end > prev_end:
                prev_off, prev_end, prev_idx = curr_off, curr_end, curr_idx

    return WgpkToc(
        source=label,
        version=version,
        flags=flags,
        byte_length=file_size,
        data_start=data_start,
        entries=tuple(entries),
    )


# ---------------------------------------------------------------------------
# Reflection sidecar loader (versions 1, 2, and 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReflectSidecar:
    source: str
    schema: str | None
    version: int | None
    stage_hashes: frozenset[int]
    stage_map: dict[int, str]
    raw_entries: dict[int, Mapping[str, Any]]


def resolve_reflect_path(
    pack_path: str | Path | None,
    reflect_path: str | Path | None = None,
) -> Path | None:
    """Resolve explicit `--reflect` path or auto-detect adjacent `<pack>.reflect.json`."""
    if reflect_path is not None:
        return Path(reflect_path)
    if pack_path is None:
        return None
    base = Path(pack_path)
    primary = base.with_name(base.name + ".reflect.json")
    if primary.is_file():
        return primary
    alt = base.with_suffix(".reflect.json")
    if alt != primary and alt.is_file():
        return alt
    return None


def load_reflect_sidecar(
    reflect_path: str | Path,
    *,
    source_label: str | None = None,
) -> ReflectSidecar:
    """Load and validate a `playgta6-stage-reflection` v1/v2/v3 sidecar file."""
    path = Path(reflect_path)
    label = sanitize_source_label(source_label if source_label is not None else path)

    if not path.exists():
        raise DiffMissingShadersError(
            f"reflection sidecar not found: {label}",
            code="reflect_not_found",
            source=label,
        )
    if not path.is_file():
        raise DiffMissingShadersError(
            f"reflection sidecar path is not a file: {label}",
            code="invalid_reflect_sidecar",
            source=label,
        )

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DiffMissingShadersError(_format_os_error(label, exc), code="io_error", source=label) from exc
    except UnicodeDecodeError as exc:
        raise DiffMissingShadersError(
            f"{label}: invalid UTF-8 in reflection sidecar: {exc.reason}",
            code="invalid_reflect_sidecar",
            source=label,
        ) from exc

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DiffMissingShadersError(
            f"{label}: invalid JSON in reflection sidecar (line {exc.lineno}, col {exc.colno}): {exc.msg}",
            code="invalid_reflect_sidecar",
            source=label,
        ) from exc

    if not isinstance(doc, Mapping):
        raise DiffMissingShadersError(
            f"{label}: reflection sidecar top-level value must be a JSON object",
            code="invalid_reflect_sidecar",
            source=label,
        )

    schema_val = doc.get("schema")
    if schema_val is not None:
        if not isinstance(schema_val, str) or schema_val != REFLECT_SCHEMA:
            raise DiffMissingShadersError(
                f"{label}: unsupported reflection sidecar schema (expected {REFLECT_SCHEMA!r})",
                code="invalid_reflect_sidecar",
                source=label,
            )

    version_val = doc.get("version")
    if version_val is not None:
        if (
            isinstance(version_val, bool)
            or not isinstance(version_val, int)
            or version_val not in SUPPORTED_REFLECT_VERSIONS
        ):
            raise DiffMissingShadersError(
                f"{label}: unsupported reflection sidecar version {version_val!r} (expected 1, 2, or 3)",
                code="invalid_reflect_sidecar",
                source=label,
            )

    stages_obj = doc.get("stages")
    if not isinstance(stages_obj, Mapping):
        raise DiffMissingShadersError(
            f"{label}: missing or non-object 'stages' map",
            code="invalid_reflect_sidecar",
            source=label,
        )

    stage_map: dict[int, str] = {}
    raw_entries: dict[int, Mapping[str, Any]] = {}
    for raw_key, raw_entry in stages_obj.items():
        if not isinstance(raw_key, str) or not isinstance(raw_entry, Mapping):
            raise DiffMissingShadersError(
                f"{label}: every 'stages' entry must map a hash string to an object",
                code="invalid_reflect_sidecar",
                source=label,
            )
        try:
            stage_hash = normalize_shader_hash(raw_key)
        except (TypeError, ValueError) as exc:
            raise DiffMissingShadersError(
                f"{label}: invalid shader hash key in reflection sidecar",
                code="invalid_reflect_sidecar",
                source=label,
            ) from exc
        stage_map[stage_hash] = normalize_stage_name(raw_entry.get("stage"))
        raw_entries[stage_hash] = raw_entry

    return ReflectSidecar(
        source=label,
        schema=schema_val,
        version=version_val,
        stage_hashes=frozenset(stage_map),
        stage_map=stage_map,
        raw_entries=raw_entries,
    )


# ---------------------------------------------------------------------------
# Multi-source shader demand parser (boot logs, missing JSON, pipeline dumps)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemandedShader:
    hash: int
    stage: str
    count: int
    sources: tuple[str, ...]

    @property
    def hash_hex(self) -> str:
        return format_shader_hash(self.hash)


class DemandAccumulator:
    """Accumulates demanded shader identities across one or more sanitized sources."""

    def __init__(self) -> None:
        self._stages: dict[int, str] = {}
        self._counts: dict[int, int] = {}
        self._sources: dict[int, set[str]] = {}

    def record(self, shader_hash: int, stage: str, source_label: str, *, count: int = 1) -> None:
        h = normalize_shader_hash(shader_hash)
        st = normalize_stage_name(stage)
        inc = count if count > 0 else 1
        existing_stage = self._stages.get(h)
        if existing_stage is None:
            self._stages[h] = st
        elif existing_stage == "unknown" and st != "unknown":
            self._stages[h] = st
        self._counts[h] = self._counts.get(h, 0) + inc
        self._sources.setdefault(h, set()).add(sanitize_source_label(source_label))

    def items(self) -> tuple[DemandedShader, ...]:
        return tuple(
            DemandedShader(
                hash=h,
                stage=self._stages[h],
                count=self._counts[h],
                sources=tuple(sorted(self._sources[h])),
            )
            for h in sorted(self._counts)
        )


def _spans_overlap(start: int, end: int, consumed: list[tuple[int, int]]) -> bool:
    for c_start, c_end in consumed:
        if start < c_end and c_start < end:
            return True
    return False


def _infer_stage_around_span(line: str, span_start: int, span_end: int, prev_end: int) -> str:
    """Infer a shader stage hint immediately after or locally before a hash span."""
    trailing = _TRAILING_STAGE_RE.match(line[span_end:])
    if trailing:
        return normalize_stage_name(trailing.group(1))

    prefix = line[prev_end:span_start]
    stage_matches = list(_STAGE_WORD_RE.finditer(prefix))
    if stage_matches:
        return normalize_stage_name(stage_matches[-1].group(1))
    return "unknown"


def extract_demanded_shaders_from_log_line(line: str) -> list[tuple[int, str]]:
    """Extract (hash_int, stage_name) occurrences from a single console log line."""
    # 1. Standalone priority-hash line: `[0x]<16-hex>` (with optional `#` comment).
    code_part = line.split("#", 1)[0].strip()
    if code_part:
        standalone = _HEX16_RE.fullmatch(code_part)
        if standalone:
            return [(int(standalone.group(1), 16), "unknown")]

    hits: list[tuple[int, int, int, str]] = []  # (hex_start, hex_end, hash_int, stage)
    consumed: list[tuple[int, int]] = []

    # 2. Explicitly stage-bound `0x<16-hex>` tokens (e.g., `VS 0x...`, `ps: 0x...`, `vertex 0x...`).
    for match in _STAGE_BOUND_0X_RE.finditer(line):
        h_start, h_end = match.span(2)
        if _spans_overlap(h_start, h_end, consumed):
            continue
        stage = normalize_stage_name(match.group(1))
        hits.append((h_start, h_end, int(match.group(2), 16), stage))
        consumed.append((h_start, h_end))

    # 3. Stage-bound bare 16-hex tokens (require hex letter or leading 0000 to reject timestamps).
    for match in _STAGE_BOUND_BARE_RE.finditer(line):
        h_start, h_end = match.span(2)
        if _spans_overlap(h_start, h_end, consumed):
            continue
        token = match.group(2)
        if any(ch in "abcdefABCDEF" for ch in token) or token.startswith("0000"):
            stage = normalize_stage_name(match.group(1))
            hits.append((h_start, h_end, int(token, 16), stage))
            consumed.append((h_start, h_end))

    # 4. Other explicit `0x<16-hex>` tokens on the line.
    extra_spans: list[tuple[int, int, int]] = []
    for match in _EXPLICIT_0X16_RE.finditer(line):
        h_start, h_end = match.span(1)
        if _spans_overlap(h_start, h_end, consumed):
            continue
        extra_spans.append((h_start, h_end, int(match.group(1), 16)))
        consumed.append((h_start, h_end))

    # 5. Bare 16-hex tokens on shader-context lines (keyed or containing a hex letter a-f).
    if _SHADER_CONTEXT_RE.search(line):
        for match in _KEYED_BARE_HEX16_RE.finditer(line):
            h_start, h_end = match.span(1)
            if _spans_overlap(h_start, h_end, consumed):
                continue
            token = match.group(1)
            if any(ch in "abcdefABCDEF" for ch in token) or token.startswith("0000"):
                extra_spans.append((h_start, h_end, int(token, 16)))
                consumed.append((h_start, h_end))
        for match in _BARE_HEX16_RE.finditer(line):
            h_start, h_end = match.span(1)
            if _spans_overlap(h_start, h_end, consumed):
                continue
            token = match.group(1)
            if any(ch in "abcdefABCDEF" for ch in token):
                extra_spans.append((h_start, h_end, int(token, 16)))
                consumed.append((h_start, h_end))

    if extra_spans:
        all_occupied = sorted(consumed, key=lambda s: s[0])
        for h_start, h_end, h_val in sorted(extra_spans, key=lambda s: s[0]):
            prev_end = 0
            for occ_start, occ_end in all_occupied:
                if occ_end <= h_start:
                    prev_end = max(prev_end, occ_end)
            stage = _infer_stage_around_span(line, h_start, h_end, prev_end)
            hits.append((h_start, h_end, h_val, stage))

    hits.sort(key=lambda item: item[0])
    return [(h_val, stage) for _s, _e, h_val, stage in hits]


def parse_boot_log_text(
    text: str,
    *,
    source_label: str = "<memory>",
    acc: DemandAccumulator | None = None,
) -> DemandAccumulator:
    """Parse sanitized console log text into a `DemandAccumulator`."""
    target = acc if acc is not None else DemandAccumulator()
    label = sanitize_source_label(source_label)
    for line in text.lstrip("\ufeff").splitlines():
        for shader_hash, stage in extract_demanded_shaders_from_log_line(line):
            target.record(shader_hash, stage, label, count=1)
    return target


def parse_boot_log_file(
    log_path: str | Path,
    *,
    acc: DemandAccumulator | None = None,
    source_label: str | None = None,
) -> DemandAccumulator:
    """Stream a Chrome/native boot log file and record shader demands."""
    path = Path(log_path)
    label = sanitize_source_label(source_label if source_label is not None else path)
    if not path.exists():
        raise DiffMissingShadersError(
            f"boot log not found: {label}",
            code="log_not_found",
            source=label,
        )
    if not path.is_file():
        raise DiffMissingShadersError(
            f"boot log path is not a file: {label}",
            code="invalid_boot_log",
            source=label,
        )
    target = acc if acc is not None else DemandAccumulator()
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
            for line in fh:
                for shader_hash, stage in extract_demanded_shaders_from_log_line(line):
                    target.record(shader_hash, stage, label, count=1)
    except OSError as exc:
        raise DiffMissingShadersError(_format_os_error(label, exc), code="io_error", source=label) from exc
    return target


def _extract_missing_shader_list(doc: Any, label: str) -> list[Any]:
    if isinstance(doc, list):
        return doc
    if not isinstance(doc, Mapping):
        raise DiffMissingShadersError(
            f"{label}: missing-shader JSON must be an object or array",
            code="invalid_missing_json",
            source=label,
        )

    candidates: list[Any] = []
    for key in ("missingShaders", "topMissingShaders"):
        val = doc.get(key)
        if val is not None:
            if not isinstance(val, list):
                raise DiffMissingShadersError(
                    f"{label}: '{key}' must be a JSON array",
                    code="invalid_missing_json",
                    source=label,
                )
            return val

    gpu_tel = doc.get("gpuTelemetry")
    if isinstance(gpu_tel, Mapping):
        for key in ("missingShaders", "topMissingShaders"):
            val = gpu_tel.get(key)
            if val is not None:
                if not isinstance(val, list):
                    raise DiffMissingShadersError(
                        f"{label}: 'gpuTelemetry.{key}' must be a JSON array",
                        code="invalid_missing_json",
                        source=label,
                    )
                return val
        summary = gpu_tel.get("missingShaderSummary")
        if isinstance(summary, Mapping):
            for key in ("top", "entries"):
                val = summary.get(key)
                if isinstance(val, list):
                    return val

    summary = doc.get("missingShaderSummary")
    if isinstance(summary, Mapping):
        for key in ("top", "entries"):
            val = summary.get(key)
            if isinstance(val, list):
                return val

    # Recognize empty valid playgta6-missing-shaders-v1 export if entries/shaders present.
    for key in ("entries", "shaders"):
        val = doc.get(key)
        if isinstance(val, list):
            return val

    fmt = doc.get("format") or doc.get("schema")
    if fmt == "playgta6-missing-shaders-v1":
        return candidates

    raise DiffMissingShadersError(
        f"{label}: JSON does not contain a recognized missingShaders array",
        code="invalid_missing_json",
        source=label,
    )


def _pick_count(entry: Mapping[str, Any]) -> int:
    for key in ("count", "misses", "hits", "occurrences", "missCount", "miss_count"):
        raw = entry.get(key)
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            return raw
    return 1


def parse_missing_json_file(
    json_path: str | Path,
    *,
    acc: DemandAccumulator | None = None,
    source_label: str | None = None,
) -> DemandAccumulator:
    """Parse browser-exported `playgta6-missing-shaders-v1` or crash-report JSON."""
    path = Path(json_path)
    label = sanitize_source_label(source_label if source_label is not None else path)
    if not path.exists():
        raise DiffMissingShadersError(
            f"missing-shader JSON not found: {label}",
            code="missing_json_not_found",
            source=label,
        )
    if not path.is_file():
        raise DiffMissingShadersError(
            f"missing-shader JSON path is not a file: {label}",
            code="invalid_missing_json",
            source=label,
        )

    try:
        raw_text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise DiffMissingShadersError(_format_os_error(label, exc), code="io_error", source=label) from exc
    except UnicodeDecodeError as exc:
        raise DiffMissingShadersError(
            f"{label}: invalid UTF-8 in missing-shader JSON: {exc.reason}",
            code="invalid_missing_json",
            source=label,
        ) from exc

    try:
        doc = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise DiffMissingShadersError(
            f"{label}: invalid JSON (line {exc.lineno}, col {exc.colno}): {exc.msg}",
            code="invalid_missing_json",
            source=label,
        ) from exc

    raw_items = _extract_missing_shader_list(doc, label)
    target = acc if acc is not None else DemandAccumulator()

    for idx, item in enumerate(raw_items):
        if isinstance(item, str):
            try:
                shader_hash = normalize_shader_hash(item)
            except (TypeError, ValueError) as exc:
                raise DiffMissingShadersError(
                    f"{label}: invalid shader hash at missingShaders[{idx}]",
                    code="invalid_missing_json",
                    source=label,
                ) from exc
            target.record(shader_hash, "unknown", label, count=1)
            continue

        if not isinstance(item, Mapping):
            raise DiffMissingShadersError(
                f"{label}: missingShaders[{idx}] must be an object or hash string",
                code="invalid_missing_json",
                source=label,
            )

        raw_hash = None
        for key in ("hash", "shaderHash", "shader_hash", "hash_fnv1a64", "code_identity", "identity"):
            val = item.get(key)
            if val is not None and (not isinstance(val, str) or val.strip()):
                raw_hash = val
                break
        if raw_hash is None:
            raise DiffMissingShadersError(
                f"{label}: missingShaders[{idx}] is missing a shader hash field",
                code="invalid_missing_json",
                source=label,
            )
        try:
            shader_hash = normalize_shader_hash(raw_hash)
        except (TypeError, ValueError) as exc:
            raise DiffMissingShadersError(
                f"{label}: invalid shader hash at missingShaders[{idx}]",
                code="invalid_missing_json",
                source=label,
            ) from exc

        raw_stage = item.get("stage")
        if raw_stage is None:
            raw_stage = item.get("stageName")
        if raw_stage is None:
            raw_stage = item.get("stage_name")
        stage = normalize_stage_name(raw_stage)
        if stage == "unknown" and "kind" in item:
            kind_val = item.get("kind")
            if isinstance(kind_val, int) and not isinstance(kind_val, bool):
                stage = _DXIL_KIND_TO_STAGE.get(kind_val, normalize_stage_name(kind_val))
            else:
                stage = normalize_stage_name(kind_val)

        target.record(shader_hash, stage, label, count=_pick_count(item))

    return target


def _parse_dump_filename(filename: str) -> tuple[int | None, str]:
    base = sanitize_source_label(filename)
    if not base or base.startswith("<"):
        return None, "unknown"
    stem = base.rsplit(".", 1)[0] if "." in base else base
    if not stem:
        return None, "unknown"

    direct = _HEX16_RE.fullmatch(stem)
    if direct:
        return int(direct.group(1), 16), "unknown"

    parts = [part for part in re.split(r"[_-]+", stem) if part]
    if len(parts) < 2:
        return None, "unknown"

    stage = "unknown"
    remaining = parts
    first_stage = _STAGE_ALIASES.get(parts[0].lower())
    if first_stage is not None:
        stage = first_stage
        remaining = parts[1:]
    else:
        last_stage = _STAGE_ALIASES.get(parts[-1].lower())
        if last_stage is not None:
            stage = last_stage
            remaining = parts[:-1]

    if len(remaining) == 1:
        match = _HEX16_RE.fullmatch(remaining[0])
        if match:
            return int(match.group(1), 16), stage
    return None, "unknown"


def _scan_dump_dir_filenames(
    directory: Path,
    *,
    suffix: str,
    seen: dict[int, str],
    label: str,
) -> None:
    try:
        children = sorted(directory.iterdir(), key=lambda p: p.name)
    except OSError as exc:
        raise DiffMissingShadersError(_format_os_error(label, exc), code="io_error", source=label) from exc

    for child in children:
        if child.suffix.lower() != suffix:
            continue
        try:
            if not child.is_file():
                continue
        except OSError:
            continue
        shader_hash, stage = _parse_dump_filename(child.name)
        if shader_hash is not None:
            existing = seen.get(shader_hash)
            if existing is None or (existing == "unknown" and stage != "unknown"):
                seen[shader_hash] = stage


def parse_pipeline_dump_dir(
    dump_dir: str | Path,
    *,
    acc: DemandAccumulator | None = None,
    source_label: str | None = None,
) -> DemandAccumulator:
    """Parse a `playgta6-pipeline-dump` v1 directory from manifest/filenames only (never reads binaries)."""
    root = Path(dump_dir)
    label = sanitize_source_label(source_label if source_label is not None else root)
    if not root.exists():
        raise DiffMissingShadersError(
            f"pipeline dump directory not found: {label}",
            code="dump_not_found",
            source=label,
        )
    if not root.is_dir():
        raise DiffMissingShadersError(
            f"pipeline dump path is not a directory: {label}",
            code="invalid_dump_dir",
            source=label,
        )

    seen: dict[int, str] = {}
    manifest_loaded = False
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest_label = sanitize_source_label(manifest_path)
        if not manifest_path.is_file():
            raise DiffMissingShadersError(
                f"{manifest_label}: pipeline dump manifest path is not a file",
                code="invalid_manifest",
                source=label,
            )
        try:
            raw_text = manifest_path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise DiffMissingShadersError(_format_os_error(manifest_label, exc), code="io_error", source=label) from exc
        except UnicodeDecodeError as exc:
            raise DiffMissingShadersError(
                f"{manifest_label}: invalid UTF-8 in pipeline dump manifest: {exc.reason}",
                code="invalid_manifest",
                source=label,
            ) from exc

        try:
            doc = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise DiffMissingShadersError(
                f"{manifest_label}: invalid JSON in pipeline dump manifest "
                f"(line {exc.lineno}, col {exc.colno}): {exc.msg}",
                code="invalid_manifest",
                source=label,
            ) from exc

        if not isinstance(doc, Mapping):
            raise DiffMissingShadersError(
                f"{manifest_label}: top-level value must be a JSON object",
                code="invalid_manifest",
                source=label,
            )
        if doc.get("schema") != PIPELINE_DUMP_SCHEMA:
            raise DiffMissingShadersError(
                f"{manifest_label}: unsupported pipeline dump schema (expected {PIPELINE_DUMP_SCHEMA!r})",
                code="invalid_manifest",
                source=label,
            )
        if "schema_version" in doc:
            ver = doc.get("schema_version")
            if isinstance(ver, bool) or not isinstance(ver, int) or ver != PIPELINE_DUMP_SCHEMA_VERSION:
                raise DiffMissingShadersError(
                    f"{manifest_label}: unsupported schema_version (expected {PIPELINE_DUMP_SCHEMA_VERSION})",
                    code="invalid_manifest",
                    source=label,
                )

        native_stages = doc.get("native_stages")
        if native_stages is not None:
            if not isinstance(native_stages, list):
                raise DiffMissingShadersError(
                    f"{manifest_label}: 'native_stages' must be a list",
                    code="invalid_manifest",
                    source=label,
                )
            for idx, item in enumerate(native_stages):
                if not isinstance(item, Mapping):
                    raise DiffMissingShadersError(
                        f"{manifest_label}: native_stages[{idx}] must be an object",
                        code="invalid_manifest",
                        source=label,
                    )
                raw_id = item.get("code_identity")
                if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
                    raw_id = item.get("identity")
                if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
                    continue
                try:
                    shader_hash = normalize_shader_hash(raw_id)
                except (TypeError, ValueError) as exc:
                    raise DiffMissingShadersError(
                        f"{manifest_label}: invalid shader identity at native_stages[{idx}]",
                        code="invalid_manifest",
                        source=label,
                    ) from exc
                stage = normalize_stage_name(item.get("stage"))
                existing = seen.get(shader_hash)
                if existing is None or (existing == "unknown" and stage != "unknown"):
                    seen[shader_hash] = stage

        dxil_entries = doc.get("dxil")
        if dxil_entries is not None:
            if not isinstance(dxil_entries, list):
                raise DiffMissingShadersError(
                    f"{manifest_label}: 'dxil' must be a list",
                    code="invalid_manifest",
                    source=label,
                )
            for idx, item in enumerate(dxil_entries):
                if not isinstance(item, Mapping):
                    raise DiffMissingShadersError(
                        f"{manifest_label}: dxil[{idx}] must be an object",
                        code="invalid_manifest",
                        source=label,
                    )
                raw_id = item.get("identity")
                if raw_id is None or (isinstance(raw_id, str) and not raw_id.strip()):
                    continue
                try:
                    shader_hash = normalize_shader_hash(raw_id)
                except (TypeError, ValueError) as exc:
                    raise DiffMissingShadersError(
                        f"{manifest_label}: invalid shader identity at dxil[{idx}]",
                        code="invalid_manifest",
                        source=label,
                    ) from exc
                stage = normalize_stage_name(item.get("stage"))
                if stage == "unknown":
                    raw_kind = item.get("kind")
                    if isinstance(raw_kind, int) and not isinstance(raw_kind, bool):
                        stage = _DXIL_KIND_TO_STAGE.get(raw_kind, "unknown")
                existing = seen.get(shader_hash)
                if existing is None or (existing == "unknown" and stage != "unknown"):
                    seen[shader_hash] = stage

        manifest_loaded = True

    native_dir = root / "native"
    dxil_dir = root / "dxil"
    has_native = native_dir.is_dir()
    has_dxil = dxil_dir.is_dir()

    if has_native:
        _scan_dump_dir_filenames(native_dir, suffix=".bin", seen=seen, label=label)
    if has_dxil:
        _scan_dump_dir_filenames(dxil_dir, suffix=".dxbc", seen=seen, label=label)
    if not manifest_loaded and not has_native and not has_dxil:
        _scan_dump_dir_filenames(root, suffix=".bin", seen=seen, label=label)
        _scan_dump_dir_filenames(root, suffix=".dxbc", seen=seen, label=label)

    target = acc if acc is not None else DemandAccumulator()
    for h in sorted(seen):
        target.record(h, seen[h], label, count=1)
    return target


# ---------------------------------------------------------------------------
# Missing-shader diff report
# ---------------------------------------------------------------------------


def _pct(num: int, den: int) -> float:
    if den <= 0:
        return 100.0
    return round((num / den) * 100.0, 2)


def diff_missing_shaders(
    pack_path: str | Path,
    *,
    reflect_path: str | Path | None = None,
    boot_logs: Sequence[str | Path] = (),
    missing_jsons: Sequence[str | Path] = (),
    dump_dirs: Sequence[str | Path] = (),
    emit_priority_hashes: str | Path | None = None,
) -> dict[str, Any]:
    """Compare multi-source demanded shaders against a WGPK v1/v2 pack and `.reflect.json` sidecar."""
    if not boot_logs and not missing_jsons and not dump_dirs:
        raise DiffMissingShadersError(
            "at least one demand source is required (--boot-log, --missing-json, or --dump-dir)",
            code="missing_demand_source",
        )

    toc = read_wgpk_toc(pack_path)
    resolved_reflect = resolve_reflect_path(pack_path, reflect_path)
    sidecar = load_reflect_sidecar(resolved_reflect) if resolved_reflect is not None else None

    acc = DemandAccumulator()
    log_labels: list[str] = []
    for log_item in boot_logs:
        log_labels.append(sanitize_source_label(log_item))
        parse_boot_log_file(log_item, acc=acc)

    json_labels: list[str] = []
    for json_item in missing_jsons:
        json_labels.append(sanitize_source_label(json_item))
        parse_missing_json_file(json_item, acc=acc)

    dump_labels: list[str] = []
    for dump_item in dump_dirs:
        dump_labels.append(sanitize_source_label(dump_item))
        parse_pipeline_dump_dir(dump_item, acc=acc)

    demanded = acc.items()
    pack_by_hash = toc.by_hash()
    reflect_hashes = sidecar.stage_hashes if sidecar is not None else frozenset()
    reflect_stage_map = sidecar.stage_map if sidecar is not None else {}

    by_stage: dict[str, dict[str, int]] = {
        stage: {
            "demanded": 0,
            "occurrences": 0,
            "in_pack": 0,
            "missing_from_pack": 0,
            "in_reflect": 0,
            "missing_from_reflect": 0,
        }
        for stage in DEMAND_STAGE_ORDER
    }

    in_pack_count = 0
    missing_pack_items: list[dict[str, Any]] = []
    missing_pack_hashes: list[int] = []
    in_reflect_count = 0
    missing_reflect_items: list[dict[str, Any]] = []
    in_pack_missing_reflect_items: list[dict[str, Any]] = []
    total_occurrences = 0

    for item in demanded:
        total_occurrences += item.count
        pack_entry = pack_by_hash.get(item.hash)
        has_pack = pack_entry is not None
        has_reflect = item.hash in reflect_hashes

        resolved_stage = item.stage
        if resolved_stage == "unknown" and pack_entry is not None:
            resolved_stage = pack_entry.stage
        if resolved_stage == "unknown" and item.hash in reflect_stage_map:
            resolved_stage = reflect_stage_map[item.hash]
        stage_bucket = by_stage[resolved_stage if resolved_stage in by_stage else "unknown"]

        stage_bucket["demanded"] += 1
        stage_bucket["occurrences"] += item.count

        record = {
            "hash": item.hash_hex,
            "stage": resolved_stage,
            "count": item.count,
            "in_pack": has_pack,
            "in_reflect": has_reflect,
            "sources": list(item.sources),
        }

        if has_pack:
            in_pack_count += 1
            stage_bucket["in_pack"] += 1
        else:
            stage_bucket["missing_from_pack"] += 1
            missing_pack_items.append(record)
            missing_pack_hashes.append(item.hash)

        if has_reflect:
            in_reflect_count += 1
            stage_bucket["in_reflect"] += 1
        else:
            stage_bucket["missing_from_reflect"] += 1
            missing_reflect_items.append(record)
            if has_pack:
                in_pack_missing_reflect_items.append(record)

    # Sort missing lists by descending occurrence count, then canonical ascending hash.
    missing_pack_items.sort(key=lambda r: (-r["count"], r["hash"]))
    missing_reflect_items.sort(key=lambda r: (-r["count"], r["hash"]))
    in_pack_missing_reflect_items.sort(key=lambda r: (-r["count"], r["hash"]))

    # Emit canonically sorted deduplicated missing-from-pack hashes if requested.
    emitted_info: dict[str, Any] | None = None
    if emit_priority_hashes is not None:
        out_path = Path(emit_priority_hashes)
        out_label = sanitize_source_label(out_path)
        canonical_hashes = sorted(set(missing_pack_hashes))
        try:
            if out_path.parent and str(out_path.parent) not in ("", "."):
                out_path.parent.mkdir(parents=True, exist_ok=True)
            body = "".join(f"{format_shader_hash(h)}\n" for h in canonical_hashes)
            with out_path.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(body)
        except OSError as exc:
            raise DiffMissingShadersError(
                f"failed to write {out_label}: {exc.strerror or type(exc).__name__}",
                code="io_error",
                source=out_label,
            ) from exc
        emitted_info = {
            "output": out_label,
            "count": len(canonical_hashes),
        }

    demanded_unique = len(demanded)
    missing_from_pack = len(missing_pack_items)
    missing_from_reflect = len(missing_reflect_items)
    clean = (missing_from_pack == 0) and (sidecar is None or missing_from_reflect == 0)

    report: dict[str, Any] = {
        "schema": DIFF_REPORT_SCHEMA,
        "version": DIFF_REPORT_VERSION,
        "pack": {
            "source": toc.source,
            "version": toc.version,
            "flags": toc.flags,
            "pack_bytes": toc.byte_length,
            "entry_count": toc.entry_count,
            "stage_counts": toc.stage_counts(),
        },
        "reflect": {
            "loaded": sidecar is not None,
            "source": sidecar.source if sidecar is not None else None,
            "schema": sidecar.schema if sidecar is not None else None,
            "version": sidecar.version if sidecar is not None else None,
            "stage_count": len(reflect_hashes),
        },
        "sources": {
            "boot_logs": log_labels,
            "missing_jsons": json_labels,
            "dump_dirs": dump_labels,
        },
        "summary": {
            "demanded_unique": demanded_unique,
            "demanded_occurrences": total_occurrences,
            "in_pack": in_pack_count,
            "missing_from_pack": missing_from_pack,
            "in_reflect": in_reflect_count,
            "missing_from_reflect": missing_from_reflect,
            "in_pack_missing_from_reflect": len(in_pack_missing_reflect_items),
            "pack_coverage_pct": _pct(in_pack_count, demanded_unique),
            "reflect_coverage_pct": _pct(in_reflect_count, demanded_unique),
            "clean": clean,
        },
        "by_stage": by_stage,
        "missing_from_pack": missing_pack_items,
        "missing_from_reflect": missing_reflect_items,
        "in_pack_missing_from_reflect": in_pack_missing_reflect_items,
        "clean": clean,
    }
    if emitted_info is not None:
        report["priority_hashes_emitted"] = emitted_info
    return report


# ---------------------------------------------------------------------------
# WGPK v1/v2 incremental merge verification (`verify-merge`)
# ---------------------------------------------------------------------------


def _resolved_bindings_count(stage_obj: Any) -> int:
    """Match `tools/merge_wgsl_packs.py`'s exact tie-break binding resolution count."""
    if not isinstance(stage_obj, Mapping):
        return 0
    bindings = stage_obj.get("bindings", [])
    if not isinstance(bindings, list):
        return 0
    return sum(1 for b in bindings if isinstance(b, Mapping) and "source" in b)


def _stream_slice_sha256(
    fh: Any,
    offset: int,
    length: int,
    *,
    check_utf8: bool,
) -> tuple[str | None, str | None]:
    """Stream `[offset .. offset + length)` returning (sha256_hex, error_message)."""
    fh.seek(offset)
    hasher = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")() if check_utf8 else None
    left = length
    while left > 0:
        chunk = fh.read(min(STREAM_CHUNK_BYTES, left))
        if not chunk:
            return None, f"short read ({length - left} of {length} bytes)"
        hasher.update(chunk)
        if decoder is not None:
            try:
                decoder.decode(chunk, final=False)
            except UnicodeDecodeError as exc:
                return None, f"invalid UTF-8: {exc.reason}"
        left -= len(chunk)
    if decoder is not None:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            return None, f"invalid UTF-8: {exc.reason}"
    return hasher.hexdigest(), None


def verify_merge(
    input_paths: Sequence[str | Path],
    merged_path: str | Path,
) -> dict[str, Any]:
    """Verify WGPK v1/v2 merged pack structure, union completeness, tie-break payload parity, and sidecar."""
    if not input_paths:
        raise DiffMissingShadersError(
            "verify-merge requires at least one input pack (--inputs)",
            code="missing_inputs",
        )

    merged_p = Path(merged_path)
    merged_label = sanitize_source_label(merged_p)
    input_ps = [Path(p) for p in input_paths]
    input_labels = [sanitize_source_label(p) for p in input_ps]

    violations: list[dict[str, Any]] = []

    def add_violation(code: str, message: str, *, shader_hash: int | None = None) -> None:
        item: dict[str, Any] = {
            "code": code,
            "message": sanitize_error_text(message),
        }
        if shader_hash is not None:
            item["hash"] = format_shader_hash(shader_hash)
        violations.append(item)

    # 1. Validate merged pack TOC (magic, version 1/2, bounds, 64-bit offsets, overlaps, duplicates).
    merged_toc: WgpkToc | None = None
    try:
        merged_toc = read_wgpk_toc(merged_p)
    except DiffMissingShadersError as exc:
        add_violation(exc.code, exc.message)

    # 2. Load input TOCs and optional adjacent sidecars.
    input_tocs: list[WgpkToc] = []
    input_sidecars: list[ReflectSidecar | None] = []
    for p, label in zip(input_ps, input_labels):
        try:
            input_tocs.append(read_wgpk_toc(p, source_label=label))
        except DiffMissingShadersError as exc:
            add_violation(f"input_{exc.code}", f"input {label}: {exc.message}")
            input_tocs.append(
                WgpkToc(source=label, version=0, flags=0, byte_length=0, data_start=0, entries=())
            )
            input_sidecars.append(None)
            continue

        side_p = p.with_name(p.name + ".reflect.json")
        if side_p.exists():
            try:
                input_sidecars.append(load_reflect_sidecar(side_p))
            except DiffMissingShadersError as exc:
                add_violation(f"input_{exc.code}", f"input sidecar {sanitize_source_label(side_p)}: {exc.message}")
                input_sidecars.append(None)
        else:
            input_sidecars.append(None)

    # Compute expected tie-break winners across inputs (exact `merge_wgsl_packs.py` rule).
    expected_winners: dict[int, tuple[tuple[int, int, int], int, WgpkTocEntry]] = {}
    for idx, toc in enumerate(input_tocs):
        sc = input_sidecars[idx]
        raw_map = sc.raw_entries if sc is not None else {}
        for entry in toc.entries:
            sc_entry = raw_map.get(entry.hash)
            rank = (1 if sc_entry is not None else 0, _resolved_bindings_count(sc_entry), -idx)
            cur = expected_winners.get(entry.hash)
            if cur is None or rank > cur[0]:
                expected_winners[entry.hash] = (rank, idx, entry)

    merged_digests: dict[int, str] = {}
    payloads_verified = 0
    utf8_errors = 0

    if merged_toc is not None:
        # Check strictly ascending hash order.
        for i in range(1, len(merged_toc.entries)):
            prev_h = merged_toc.entries[i - 1].hash
            curr_h = merged_toc.entries[i].hash
            if prev_h >= curr_h:
                add_violation(
                    "hash_order_not_ascending",
                    f"{merged_label}: TOC hashes not strictly ascending at entry {i} "
                    f"({format_shader_hash(prev_h)} >= {format_shader_hash(curr_h)})",
                    shader_hash=curr_h,
                )

        # Stream every merged payload sequentially to validate UTF-8 and compute SHA-256.
        read_plan = sorted(merged_toc.entries, key=lambda e: e.offset)
        try:
            with merged_p.open("rb") as fh:
                for entry in read_plan:
                    digest, err = _stream_slice_sha256(fh, entry.offset, entry.length, check_utf8=True)
                    if err is not None or digest is None:
                        utf8_errors += 1
                        code = "invalid_utf8" if (err and "UTF-8" in err) else "short_read"
                        add_violation(
                            code,
                            f"{merged_label}: entry {entry.hash_hex}: {err}",
                            shader_hash=entry.hash,
                        )
                    else:
                        merged_digests[entry.hash] = digest
        except OSError as exc:
            add_violation("io_error", _format_os_error(merged_label, exc))

        # Union completeness checks.
        merged_by_hash = merged_toc.by_hash()
        merged_hashes = set(merged_by_hash)
        expected_hashes = set(expected_winners)

        for missing_h in sorted(expected_hashes - merged_hashes):
            add_violation(
                "missing_hash",
                f"{merged_label}: missing expected shader hash {format_shader_hash(missing_h)}",
                shader_hash=missing_h,
            )
        for extra_h in sorted(merged_hashes - expected_hashes):
            add_violation(
                "unexpected_extra_hash",
                f"{merged_label}: unexpected extra shader hash {format_shader_hash(extra_h)}",
                shader_hash=extra_h,
            )

        # Tie-break stage, length, and streaming SHA-256 payload parity against winning inputs.
        slices_by_input: dict[int, list[tuple[int, int, int]]] = {}
        for h in sorted(expected_hashes & merged_hashes):
            _rank, win_idx, win_entry = expected_winners[h]
            got_entry = merged_by_hash[h]
            if got_entry.stage_code != win_entry.stage_code:
                add_violation(
                    "stage_mismatch",
                    f"{merged_label}: stage mismatch for {got_entry.hash_hex} "
                    f"(got {got_entry.stage}, expected {win_entry.stage})",
                    shader_hash=h,
                )
            if got_entry.length != win_entry.length:
                add_violation(
                    "length_mismatch",
                    f"{merged_label}: payload length mismatch for {got_entry.hash_hex} "
                    f"(got {got_entry.length}, expected {win_entry.length})",
                    shader_hash=h,
                )
            elif h in merged_digests:
                slices_by_input.setdefault(win_idx, []).append((win_entry.offset, win_entry.length, h))

        for win_idx, items in slices_by_input.items():
            items.sort(key=lambda item: item[0])
            in_p = input_ps[win_idx]
            in_label = input_labels[win_idx]
            try:
                with in_p.open("rb") as fh:
                    for off, length, h in items:
                        exp_digest, err = _stream_slice_sha256(fh, off, length, check_utf8=False)
                        if err is not None or exp_digest is None:
                            add_violation(
                                "input_short_read",
                                f"input {in_label}: failed reading {format_shader_hash(h)}: {err}",
                                shader_hash=h,
                            )
                            continue
                        if exp_digest != merged_digests[h]:
                            add_violation(
                                "payload_mismatch",
                                f"{merged_label}: payload SHA-256 mismatch for {format_shader_hash(h)} "
                                f"against winning input {in_label}",
                                shader_hash=h,
                            )
                        else:
                            payloads_verified += 1
            except OSError as exc:
                add_violation("io_error", _format_os_error(in_label, exc))

    # 3. Cross-check `<merged>.reflect.json` sidecar alignment.
    merged_side_p = merged_p.with_name(merged_p.name + ".reflect.json")
    merged_side_label = sanitize_source_label(merged_side_p)
    loaded_input_sidecars = [sc for sc in input_sidecars if sc is not None]
    any_input_sidecar = len(loaded_input_sidecars) > 0
    merged_sidecar: ReflectSidecar | None = None

    if merged_side_p.exists():
        try:
            merged_sidecar = load_reflect_sidecar(merged_side_p, source_label=merged_side_label)
        except DiffMissingShadersError as exc:
            add_violation(f"sidecar_{exc.code}", exc.message)
    elif any_input_sidecar:
        add_violation(
            "missing_merged_sidecar",
            f"{merged_side_label}: expected merged reflection sidecar because input packs carry sidecars",
        )

    if merged_sidecar is not None and merged_toc is not None:
        dangling = sorted(merged_sidecar.stage_hashes - merged_toc.hashes)
        for h in dangling:
            add_violation(
                "dangling_sidecar_hash",
                f"{merged_side_label}: dangling stage hash {format_shader_hash(h)} not in {merged_label}",
                shader_hash=h,
            )

    if any_input_sidecar and merged_sidecar is not None:
        expected_version = max(
            sc.version if sc.version is not None else 1 for sc in loaded_input_sidecars
        )
        got_version = merged_sidecar.version if merged_sidecar.version is not None else 1
        if got_version != expected_version:
            add_violation(
                "sidecar_version_mismatch",
                f"{merged_side_label}: sidecar version mismatch (got {got_version}, expected {expected_version})",
            )

        expected_sidecar_entries: dict[int, Mapping[str, Any]] = {}
        for h, (_rank, win_idx, _win_entry) in expected_winners.items():
            win_sc = input_sidecars[win_idx]
            if win_sc is not None and h in win_sc.raw_entries:
                expected_sidecar_entries[h] = win_sc.raw_entries[h]

        for h in sorted(set(expected_sidecar_entries) - set(merged_sidecar.raw_entries)):
            add_violation(
                "missing_sidecar_entry",
                f"{merged_side_label}: missing winning sidecar entry for {format_shader_hash(h)}",
                shader_hash=h,
            )
        for h in sorted(set(merged_sidecar.raw_entries) - set(expected_sidecar_entries)):
            # Avoid duplicating if already flagged as dangling outside pack.
            if merged_toc is None or h in merged_toc.hashes:
                add_violation(
                    "unexpected_sidecar_entry",
                    f"{merged_side_label}: unexpected sidecar entry for {format_shader_hash(h)}",
                    shader_hash=h,
                )
        for h in sorted(set(expected_sidecar_entries) & set(merged_sidecar.raw_entries)):
            if merged_sidecar.raw_entries[h] != expected_sidecar_entries[h]:
                add_violation(
                    "sidecar_entry_mismatch",
                    f"{merged_side_label}: sidecar entry mismatch for {format_shader_hash(h)} against winner",
                    shader_hash=h,
                )

    passed = len(violations) == 0
    return {
        "ok": passed,
        "passed": passed,
        "schema": VERIFY_MERGE_SCHEMA,
        "version": VERIFY_MERGE_VERSION,
        "inputs": input_labels,
        "expected_unique_entries": len(expected_winners),
        "merged": {
            "source": merged_label,
            "version": merged_toc.version if merged_toc is not None else None,
            "pack_bytes": merged_toc.byte_length if merged_toc is not None else None,
            "entry_count": merged_toc.entry_count if merged_toc is not None else None,
            "stage_counts": merged_toc.stage_counts() if merged_toc is not None else None,
        },
        "sidecar": {
            "expected": any_input_sidecar,
            "loaded": merged_sidecar is not None,
            "source": merged_side_label if merged_sidecar is not None else None,
            "version": merged_sidecar.version if merged_sidecar is not None else None,
            "stage_count": len(merged_sidecar.stage_hashes) if merged_sidecar is not None else 0,
        },
        "payloads_verified": payloads_verified,
        "utf8_errors": utf8_errors,
        "violation_count": len(violations),
        "violations": violations,
    }


# ---------------------------------------------------------------------------
# Human-readable text formatters
# ---------------------------------------------------------------------------


def format_diff_text(report: Mapping[str, Any]) -> str:
    pack = report["pack"]
    reflect = report["reflect"]
    summary = report["summary"]
    sources = report["sources"]

    reflect_str = (
        f"{reflect['source']} (v{reflect['version']}, {reflect['stage_count']} stages)"
        if reflect["loaded"]
        else "(none)"
    )
    src_parts: list[str] = []
    if sources["boot_logs"]:
        src_parts.append("logs=" + ",".join(sources["boot_logs"]))
    if sources["missing_jsons"]:
        src_parts.append("missing_json=" + ",".join(sources["missing_jsons"]))
    if sources["dump_dirs"]:
        src_parts.append("dump_dirs=" + ",".join(sources["dump_dirs"]))

    lines = [
        f"Missing-Shader Diff: {pack['source']} (WGPK v{pack['version']}, {pack['entry_count']} entries)",
        f"  Reflect sidecar:       {reflect_str}",
        f"  Demand sources:        {'; '.join(src_parts)}",
        f"  Demanded unique:       {summary['demanded_unique']} ({summary['demanded_occurrences']} occurrences)",
        f"  In pack:               {summary['in_pack']} / {summary['demanded_unique']} "
        f"({summary['pack_coverage_pct']:.2f}%)",
        f"  Missing from pack:     {summary['missing_from_pack']}",
        f"  In reflect:            {summary['in_reflect']} / {summary['demanded_unique']} "
        f"({summary['reflect_coverage_pct']:.2f}%)",
        f"  Missing from reflect:  {summary['missing_from_reflect']} "
        f"(in pack, missing reflect: {summary['in_pack_missing_from_reflect']})",
        "  By stage:",
    ]
    for stage in DEMAND_STAGE_ORDER:
        b = report["by_stage"][stage]
        lines.append(
            f"    {stage:<9} demanded={b['demanded']} (occ={b['occurrences']}), "
            f"in_pack={b['in_pack']}, missing_pack={b['missing_from_pack']}, "
            f"in_reflect={b['in_reflect']}, missing_reflect={b['missing_from_reflect']}"
        )

    missing_pack = report["missing_from_pack"]
    if missing_pack:
        lines.append(f"  Missing from pack ({len(missing_pack)}):")
        for item in missing_pack:
            lines.append(
                f"    - {item['hash']}  stage={item['stage']:<8} count={item['count']}  "
                f"sources={','.join(item['sources'])}"
            )

    emitted = report.get("priority_hashes_emitted")
    if emitted is not None:
        lines.append(
            f"  Priority hashes:       wrote {emitted['count']} hashes to {emitted['output']}"
        )
    return "\n".join(lines)


def format_verify_merge_text(report: Mapping[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    merged = report["merged"]
    sidecar = report["sidecar"]
    ver_str = f"v{merged['version']}" if merged["version"] is not None else "(invalid)"
    lines = [
        f"WGPK Merge Verification [{status}]: {merged['source']} ({ver_str})",
        f"  Inputs:                {', '.join(report['inputs'])}",
        f"  Expected unique:       {report['expected_unique_entries']}",
        f"  Merged entries:        {merged['entry_count']} ({merged['pack_bytes']} bytes)",
        f"  Payloads SHA-256 OK:   {report['payloads_verified']}",
        f"  Sidecar:               "
        + (
            f"{sidecar['source']} (v{sidecar['version']}, {sidecar['stage_count']} stages)"
            if sidecar["loaded"]
            else "(none)"
        ),
        f"  Violations:            {report['violation_count']}",
    ]
    for v in report["violations"]:
        prefix = f"[{v['hash']}] " if "hash" in v else ""
        lines.append(f"    - {prefix}{v['code']}: {v['message']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diff_missing_shaders.py",
        description=(
            "Sanitized missing-shader diff CLI and WGPK v1/v2 incremental merge verifier. "
            "Defaults to `diff` when --pack is supplied directly without a subcommand."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    diff_p = subparsers.add_parser(
        "diff",
        help="Diff demanded shaders (boot logs / missing JSON / pipeline dumps) against a WGPK v1/v2 pack.",
    )
    diff_p.add_argument("--pack", required=True, type=Path, help="Path to active .wgsl.pack (v1 or v2)")
    diff_p.add_argument(
        "--reflect",
        default=None,
        type=Path,
        help="Optional explicit .reflect.json sidecar path (auto-detects <pack>.reflect.json)",
    )
    diff_p.add_argument(
        "--boot-log",
        action="append",
        default=[],
        type=Path,
        dest="boot_logs",
        help="Chrome or native boot console log file (repeatable)",
    )
    diff_p.add_argument(
        "--missing-json",
        action="append",
        default=[],
        type=Path,
        dest="missing_jsons",
        help="Browser playgta6-missing-shaders-v1 or crash-report JSON file (repeatable)",
    )
    diff_p.add_argument(
        "--dump-dir",
        action="append",
        default=[],
        type=Path,
        dest="dump_dirs",
        help="playgta6-pipeline-dump v1 directory (repeatable; reads manifest/filenames only)",
    )
    diff_p.add_argument(
        "--emit-priority-hashes",
        default=None,
        type=Path,
        dest="emit_priority_hashes",
        help="Write canonically sorted missing-from-pack 0x<16-hex> hashes to file",
    )
    diff_p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any demanded shader is missing from the pack or loaded reflection sidecar",
    )
    diff_p.add_argument("--json", action="store_true", dest="as_json", help="Emit strictly sanitized JSON")

    verify_p = subparsers.add_parser(
        "verify-merge",
        help="Verify WGPK v1/v2 merged pack structure, union completeness, payload parity, and sidecar.",
    )
    verify_p.add_argument(
        "--inputs",
        required=True,
        action="extend",
        nargs="+",
        type=Path,
        help="Ordered input .wgsl.pack files used to build the merged pack",
    )
    verify_p.add_argument(
        "--merged",
        required=True,
        type=Path,
        help="Merged .wgsl.pack file to verify",
    )
    verify_p.add_argument("--json", action="store_true", dest="as_json", help="Emit strictly sanitized JSON")

    return parser


def _normalize_argv(argv: list[str]) -> list[str]:
    """Support shorthand invocation without subcommand when `--pack` is supplied."""
    if not argv:
        return argv
    first = argv[0]
    if first in ("diff", "verify-merge", "-h", "--help"):
        return argv
    if "--pack" in argv or any(arg.startswith("--pack=") for arg in argv):
        return ["diff", *argv]
    return argv


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    normalized_argv = _normalize_argv(raw_argv)
    wants_json = "--json" in normalized_argv

    parser = build_arg_parser()
    args = parser.parse_args(normalized_argv)

    if args.command == "diff":
        try:
            report = diff_missing_shaders(
                args.pack,
                reflect_path=args.reflect,
                boot_logs=args.boot_logs,
                missing_jsons=args.missing_jsons,
                dump_dirs=args.dump_dirs,
                emit_priority_hashes=args.emit_priority_hashes,
            )
        except DiffMissingShadersError as exc:
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "schema": DIFF_REPORT_SCHEMA,
                            "version": DIFF_REPORT_VERSION,
                            "errorCode": exc.code,
                            "error": exc.message,
                            "source": exc.source,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"error: {exc.message}", file=sys.stderr)
            return 1
        except OSError as exc:
            msg = _format_os_error(None, exc)
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "schema": DIFF_REPORT_SCHEMA,
                            "version": DIFF_REPORT_VERSION,
                            "errorCode": "io_error",
                            "error": msg,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"error: {msg}", file=sys.stderr)
            return 1

        strict_failed = bool(args.strict and not report["clean"])
        out_payload = {"ok": not strict_failed, "strict": bool(args.strict), **report}
        if args.as_json:
            print(json.dumps(out_payload, indent=2))
        else:
            print(format_diff_text(report))
        return 1 if strict_failed else 0

    if args.command == "verify-merge":
        try:
            report = verify_merge(args.inputs, args.merged)
        except DiffMissingShadersError as exc:
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "passed": False,
                            "schema": VERIFY_MERGE_SCHEMA,
                            "version": VERIFY_MERGE_VERSION,
                            "errorCode": exc.code,
                            "error": exc.message,
                            "source": exc.source,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"error: {exc.message}", file=sys.stderr)
            return 1
        except OSError as exc:
            msg = _format_os_error(None, exc)
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "passed": False,
                            "schema": VERIFY_MERGE_SCHEMA,
                            "version": VERIFY_MERGE_VERSION,
                            "errorCode": "io_error",
                            "error": msg,
                        },
                        indent=2,
                    )
                )
            else:
                print(f"error: {msg}", file=sys.stderr)
            return 1

        if args.as_json:
            print(json.dumps(report, indent=2))
        else:
            print(format_verify_merge_text(report))
        return 0 if report["passed"] else 1

    if wants_json:
        print(json.dumps({"ok": False, "errorCode": "unknown_command", "error": "unknown command"}))
    else:
        print("error: unknown command", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
