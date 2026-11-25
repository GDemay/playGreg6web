#!/usr/bin/env python3
"""Write the stage-reflection sidecar (`<pack>.reflect.json`, G62 / #651) from the translator's
JSON output.

Input: one or more JSON files, each either a `playgta6-shaders translate-rdna2 --json` result (one
stage) or a `playgta6-shaders pack-rdna2 --json` summary (its `entries` list). A stage enters the
sidecar only when its reflection carries the join keys the runtime needs:
  - every binding has a `source` ({user_sgpr, table_byte_offset} | {user_sgpr, root_constants} |
    {user_sgpr, root_descriptor} | {static_sampler} | {sampler_heap | resource_heap: {base_user_sgpr,
    stride, index: {user_sgpr, shift}}} (S34: a directly indexed heap, index from a root constant) |
    {inline_sampler: {sdwords: [d0, d1, d2, d3]}} (#482: the four GFX10 S# dwords of a sampler the
    stage builds from constants; the runtime decodes them) |
    {cbv_indexed_resource_heap | cbv_indexed_sampler_heap: {cbv: {user_sgpr, root_descriptor: true} |
    {user_sgpr, table_byte_offset} | {user_sgpr, root_constants: true}, byte_offset, shift,
    bit_offset?, mask?, stride?}} (G62 / #482: SM 6.6 bindless, the heap index is the dword at
    byte_offset of a constant buffer, read by the runtime at draw time));
  - a vertex stage that fetches attributes lists `vertex_inputs` ({location, vb_slot, byte_offset,
    components, component_type, fetch_format?}).
Stages without them are skipped and counted, so the runtime keeps its conventions for them.

    python tools/stage_reflection_sidecar.py <json>... --out <pack>.reflect.json [--merge-existing <reflect.json>] [--diff-baseline <reflect.json>] [--fail-on-regressions] [--report-missing-sources <report.json>] [--json]

The sidecar is derived from the title's shaders: write it under work/, never into the repository.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "playgta6-stage-reflection"
VERSION = 1
REPORT_SCHEMA = "playgta6-stage-reflection-missing-sources"
REPORT_VERSION = 1
DEFAULT_MAX_SAMPLE_HASHES = 5
STAGES = ("vertex", "fragment", "compute")

_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9_.$<>\[\]-]{1,64}$")
_SAFE_HASH_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

KIND_MAP = {
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


def stage_name(stage: str) -> str | None:
    return {"vertex": "vertex", "fragment": "fragment", "pixel": "fragment", "compute": "compute"}.get(stage)


def heap_source_ok(source: dict[str, Any]) -> bool | None:
    """None: not a heap source; else whether it has base_user_sgpr, stride, index {user_sgpr, shift}."""
    heap = source.get("sampler_heap", source.get("resource_heap"))
    if heap is None:
        return None
    if not isinstance(heap, dict) or not isinstance(heap.get("index"), dict):
        return False
    ints = (heap.get("base_user_sgpr"), heap.get("stride"), heap["index"].get("user_sgpr"), heap["index"].get("shift"))
    return all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in ints) and ints[3] < 32


def _u32(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 0xFFFFFFFF


def cbv_heap_source_ok(source: dict[str, Any]) -> bool | None:
    """None: not a CB-indexed heap source; else whether it follows the G62 contract (#482)."""
    heap = source.get("cbv_indexed_resource_heap", source.get("cbv_indexed_sampler_heap"))
    if heap is None:
        return None
    if not isinstance(heap, dict) or not isinstance(heap.get("cbv"), dict):
        return False
    cbv = heap["cbv"]
    where = [cbv.get("root_descriptor") is True, cbv.get("root_constants") is True, "table_byte_offset" in cbv]
    if not _u32(cbv.get("user_sgpr")) or sum(where) != 1:
        return False
    if where[2] and (not _u32(cbv["table_byte_offset"]) or cbv["table_byte_offset"] % 32 != 0):
        return False
    byte_offset, shift = heap.get("byte_offset"), heap.get("shift")
    if not _u32(byte_offset) or byte_offset % 4 != 0 or not _u32(shift) or shift >= 32:
        return False
    bit_offset, mask, stride = heap.get("bit_offset", 0), heap.get("mask", 0xFFFFFFFF), heap.get("stride", 0)
    return _u32(bit_offset) and bit_offset < 32 and _u32(mask) and mask != 0 and _u32(stride)


def inline_sampler_ok(source: dict[str, Any]) -> bool | None:
    """None: not an inline sampler; else whether sdwords is four 32-bit unsigned integers."""
    if "inline_sampler" not in source:
        return None
    sampler = source["inline_sampler"]
    words = sampler.get("sdwords") if isinstance(sampler, dict) else None
    return (isinstance(words, list) and len(words) == 4 and
            all(isinstance(w, int) and not isinstance(w, bool) and 0 <= w <= 0xFFFFFFFF for w in words))


def convert(entry: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None, str]:
    """(key, sidecar stage, reason it was skipped)."""
    key = entry.get("hash_fnv1a64")
    stage = stage_name(str(entry.get("stage", "")))
    if not key or stage is None:
        return None, None, "no hash or stage"
    out: dict[str, Any] = {"stage": stage, "bindings": []}
    for b in entry.get("bindings", []):
        source = b.get("source")
        if not isinstance(source, dict):
            return key, None, "binding without source"
        if heap_source_ok(source) is False:
            return key, None, "malformed heap source"
        if inline_sampler_ok(source) is False:
            return key, None, "malformed inline sampler"
        if cbv_heap_source_ok(source) is False:
            return key, None, "malformed CB-indexed heap source"
        kind = KIND_MAP.get(str(b.get("kind", "")))
        if kind is None:
            return key, None, f"unknown binding kind {b.get('kind')!r}"
        item: dict[str, Any] = {"group": b.get("group", 0), "binding": b["binding"], "kind": kind, "source": source}
        for field in ("view_dimension", "sample_type", "multisampled", "storage_format", "min_binding_size_bytes"):
            if field in b:
                item[field] = b[field]
        out["bindings"].append(item)
    summary = entry.get("binding_summary") or {}
    fetches_attributes = summary.get("vertex_input_mode") == "attributes"
    if stage == "vertex" and fetches_attributes:
        inputs = entry.get("vertex_inputs")
        if not isinstance(inputs, list):
            return key, None, "vertex stage without vertex_inputs"
        out["vertex_inputs"] = inputs
    return key, out, ""


def entries_of(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, dict) and isinstance(doc.get("entries"), list):
        return [e for e in doc["entries"] if isinstance(e, dict)]
    if isinstance(doc, dict) and "hash_fnv1a64" in doc:
        return [doc]
    return []


def _sanitize_label(value: Any, *, fallback: str = "") -> str:
    if not isinstance(value, str):
        return fallback
    text = value.strip()
    if not text:
        return fallback
    if _SAFE_LABEL_RE.match(text):
        return text
    return "<redacted>"


def _sanitize_hash(value: Any) -> str:
    if value is None:
        return "<redacted>"
    text = str(value).strip()
    if text and _SAFE_HASH_RE.match(text):
        return text
    return "<redacted>"


def _load_sidecar(path: Path, *, label: str, allow_missing: bool) -> dict[str, Any]:
    """Load and validate a v1 sidecar JSON file."""
    if not path.exists():
        if allow_missing:
            return {}
        raise ValueError(f"{label} file does not exist")
    if path.is_dir():
        raise ValueError(f"{label} path is a directory")
    try:
        with path.open(encoding="utf-8-sig") as f:
            doc = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {label}: {exc.msg}") from exc
    if not isinstance(doc, dict):
        raise ValueError(f"{label} must be a JSON object")
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{label} schema must be {SCHEMA!r}, got {_sanitize_label(doc.get('schema'), fallback='<invalid>')!r}")
    version = doc.get("version")
    if type(version) is not int or version != VERSION:
        raise ValueError(f"{label} version must be {VERSION}, got {version!r}")
    stages = doc.get("stages")
    if not isinstance(stages, dict):
        raise ValueError(f"{label} 'stages' must be an object")
    validated: dict[str, Any] = {}
    for k, v in stages.items():
        if not isinstance(k, str) or not k:
            raise ValueError(f"{label} stage keys must be non-empty strings")
        safe_k = _sanitize_hash(k)
        if not isinstance(v, dict):
            raise ValueError(f"{label} stage {safe_k!r} must be an object")
        if v.get("stage") not in STAGES:
            raise ValueError(
                f"{label} stage {safe_k!r} has invalid stage "
                f"{_sanitize_label(v.get('stage'), fallback='<invalid>')!r}"
            )
        if not isinstance(v.get("bindings"), list):
            raise ValueError(f"{label} stage {safe_k!r} must have a 'bindings' list")
        validated[k] = v
    return validated


def load_existing_sidecar(path: Path) -> dict[str, Any]:
    """Load and validate an existing v1 sidecar JSON; returns {} if the file does not exist."""
    return _load_sidecar(path, label="existing sidecar", allow_missing=True)


def load_baseline_sidecar(path: Path) -> dict[str, Any]:
    """Load and validate a baseline v1 sidecar JSON; raises ValueError if missing or invalid."""
    return _load_sidecar(path, label="baseline sidecar", allow_missing=False)


def _is_vertex_index_stage(stage_obj: dict[str, Any]) -> bool:
    """Return True when a vertex stage uses vertex_index (no vertex attribute inputs)."""
    if stage_obj.get("stage") != "vertex":
        return False
    raw_mode = stage_obj.get("vertex_input_mode")
    if isinstance(raw_mode, str) and raw_mode.strip():
        return raw_mode.strip().lower() == "vertex_index"
    inputs = stage_obj.get("vertex_inputs")
    return not isinstance(inputs, list) or len(inputs) == 0


def summarize_stages(stages: dict[str, Any]) -> dict[str, Any]:
    """Compute sanitized breakdown telemetry for a sidecar's stages map."""
    by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    inline_sampler_stages = 0
    inline_sampler_bindings = 0
    resource_heap_stages = 0
    resource_heap_bindings = 0
    cbv_indexed_heap_stages = 0
    cbv_indexed_heap_bindings = 0
    vertex_index_stages = 0

    for stage_obj in stages.values():
        if not isinstance(stage_obj, dict):
            continue
        stage = stage_obj.get("stage")
        if stage in by_stage:
            by_stage[stage] += 1

        has_inline_sampler = False
        has_resource_heap = False
        has_cbv_indexed_heap = False
        bindings = stage_obj.get("bindings")
        if isinstance(bindings, list):
            for binding in bindings:
                if not isinstance(binding, dict):
                    continue
                source = binding.get("source")
                if not isinstance(source, dict):
                    continue
                if "inline_sampler" in source:
                    inline_sampler_bindings += 1
                    has_inline_sampler = True
                if "resource_heap" in source:
                    resource_heap_bindings += 1
                    has_resource_heap = True
                if "cbv_indexed_resource_heap" in source or "cbv_indexed_sampler_heap" in source:
                    cbv_indexed_heap_bindings += 1
                    has_cbv_indexed_heap = True

        if has_inline_sampler:
            inline_sampler_stages += 1
        if has_resource_heap:
            resource_heap_stages += 1
        if has_cbv_indexed_heap:
            cbv_indexed_heap_stages += 1
        if _is_vertex_index_stage(stage_obj):
            vertex_index_stages += 1

    return {
        "by_stage": by_stage,
        "inline_sampler_stages": inline_sampler_stages,
        "inline_sampler_bindings": inline_sampler_bindings,
        "resource_heap_stages": resource_heap_stages,
        "resource_heap_bindings": resource_heap_bindings,
        "cbv_indexed_heap_stages": cbv_indexed_heap_stages,
        "cbv_indexed_heap_bindings": cbv_indexed_heap_bindings,
        "vertex_index_stages": vertex_index_stages,
    }


def _stage_canonical_json(stage_obj: dict[str, Any]) -> str:
    return json.dumps(stage_obj, sort_keys=True, separators=(",", ":"))


def _sample_hashes(keys: list[str], *, max_samples: int = DEFAULT_MAX_SAMPLE_HASHES) -> list[str]:
    seen: set[str] = set()
    samples: list[str] = []
    for key in keys:
        safe_hash = _sanitize_hash(key)
        if safe_hash not in seen and len(samples) < max_samples:
            seen.add(safe_hash)
            samples.append(safe_hash)
    return samples


def diff_sidecars(
    baseline_stages: dict[str, Any],
    current_stages: dict[str, Any],
    *,
    max_samples: int = DEFAULT_MAX_SAMPLE_HASHES,
) -> dict[str, Any]:
    """Compare baseline and current sidecar stage maps and return a sanitized diff summary."""
    baseline_keys = set(baseline_stages.keys())
    current_keys = set(current_stages.keys())
    added_keys = sorted(current_keys - baseline_keys)
    removed_keys = sorted(baseline_keys - current_keys)
    common_keys = sorted(baseline_keys & current_keys)

    changed_keys: list[str] = []
    unchanged_count = 0
    for key in common_keys:
        if _stage_canonical_json(baseline_stages[key]) == _stage_canonical_json(current_stages[key]):
            unchanged_count += 1
        else:
            changed_keys.append(key)

    added_by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    for key in added_keys:
        stage = current_stages[key].get("stage")
        if stage in added_by_stage:
            added_by_stage[stage] += 1

    removed_by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    for key in removed_keys:
        stage = baseline_stages[key].get("stage")
        if stage in removed_by_stage:
            removed_by_stage[stage] += 1

    baseline_count = len(baseline_stages)
    current_count = len(current_stages)
    return {
        "baseline_stages": baseline_count,
        "current_stages": current_count,
        "net_delta": current_count - baseline_count,
        "added_count": len(added_keys),
        "removed_count": len(removed_keys),
        "changed_count": len(changed_keys),
        "unchanged_count": unchanged_count,
        "added_by_stage": added_by_stage,
        "removed_by_stage": removed_by_stage,
        "added_sample_hashes": _sample_hashes(added_keys, max_samples=max_samples),
        "removed_sample_hashes": _sample_hashes(removed_keys, max_samples=max_samples),
        "changed_sample_hashes": _sample_hashes(changed_keys, max_samples=max_samples),
    }


def build(docs: list[Any], existing_stages: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, int]]:
    stages: dict[str, Any] = dict(existing_stages) if existing_stages else {}
    skipped: dict[str, int] = {}
    for doc in docs:
        for entry in entries_of(doc):
            key, stage, why = convert(entry)
            if stage is None:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            stages[key] = stage
    return {"schema": SCHEMA, "version": VERSION, "stages": stages}, skipped


def build_missing_sources_report(
    docs: list[Any],
    *,
    max_samples: int = DEFAULT_MAX_SAMPLE_HASHES,
) -> dict[str, Any]:
    """Build a sanitized summary of stages skipped for 'binding without source'."""
    total_skipped = 0
    by_stage: dict[str, int] = {stage: 0 for stage in STAGES}
    counts_by_pattern: dict[tuple[str, str, str], int] = {}
    samples_by_pattern: dict[tuple[str, str, str], list[str]] = {}
    seen_hashes_by_pattern: dict[tuple[str, str, str], set[str]] = {}

    for doc in docs:
        for entry in entries_of(doc):
            key, stage_obj, why = convert(entry)
            if stage_obj is not None or why != "binding without source":
                continue
            stage = stage_name(str(entry.get("stage", "")))
            if stage is None:
                continue
            total_skipped += 1
            by_stage[stage] += 1
            safe_hash = _sanitize_hash(key)
            for b in entry.get("bindings", []):
                if not isinstance(b, dict) or isinstance(b.get("source"), dict):
                    continue
                raw_kind = b.get("kind")
                mapped_kind = KIND_MAP.get(raw_kind) if isinstance(raw_kind, str) else None
                kind = mapped_kind if mapped_kind is not None else _sanitize_label(raw_kind, fallback="unknown")
                name = _sanitize_label(b.get("name"), fallback="")
                pat = (stage, kind, name)
                counts_by_pattern[pat] = counts_by_pattern.get(pat, 0) + 1
                seen = seen_hashes_by_pattern.setdefault(pat, set())
                samples = samples_by_pattern.setdefault(pat, [])
                if safe_hash not in seen and len(samples) < max_samples:
                    seen.add(safe_hash)
                    samples.append(safe_hash)

    top_missing_bindings = [
        {
            "stage": stage,
            "kind": kind,
            "name": name,
            "count": count,
            "sample_hashes": samples_by_pattern[(stage, kind, name)],
        }
        for (stage, kind, name), count in sorted(
            counts_by_pattern.items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2]),
        )
    ]

    return {
        "schema": REPORT_SCHEMA,
        "version": REPORT_VERSION,
        "total_skipped_missing_source": total_skipped,
        "by_stage": by_stage,
        "top_missing_bindings": top_missing_bindings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--merge-existing",
        type=Path,
        default=None,
        help="load an existing v1 sidecar JSON (if present) and merge newly converted stages on top",
    )
    parser.add_argument(
        "--diff-baseline",
        type=Path,
        default=None,
        help="compare the newly generated sidecar against a previous v1 sidecar JSON",
    )
    parser.add_argument(
        "--fail-on-regressions",
        action="store_true",
        help="with --diff-baseline, exit with code 2 if any previously reflected stage is removed",
    )
    parser.add_argument(
        "--report-missing-sources",
        type=Path,
        default=None,
        help="write a sanitized JSON report summarizing skipped 'binding without source' stages",
    )
    parser.add_argument("--json", action="store_true", help="print the counts as JSON")
    args = parser.parse_args(argv)

    if args.fail_on_regressions and args.diff_baseline is None:
        print("error: --fail-on-regressions requires --diff-baseline", file=sys.stderr)
        return 1

    existing_stages: dict[str, Any] = {}
    if args.merge_existing is not None:
        try:
            existing_stages = load_existing_sidecar(args.merge_existing)
        except (OSError, ValueError) as exc:
            print(f"error: failed to load --merge-existing: {exc}", file=sys.stderr)
            return 1

    baseline_stages: dict[str, Any] | None = None
    if args.diff_baseline is not None:
        try:
            baseline_stages = load_baseline_sidecar(args.diff_baseline)
        except (OSError, ValueError) as exc:
            print(f"error: failed to load --diff-baseline: {exc}", file=sys.stderr)
            return 1

    docs = []
    for path in args.inputs:
        with path.open(encoding="utf-8-sig") as f:
            docs.append(json.load(f))

    sidecar, skipped = build(docs, existing_stages=existing_stages if args.merge_existing is not None else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(sidecar, indent=1) + "\n", encoding="utf-8")

    if args.report_missing_sources is not None:
        report = build_missing_sources_report(docs)
        args.report_missing_sources.parent.mkdir(parents=True, exist_ok=True)
        args.report_missing_sources.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")

    counts: dict[str, Any] = {"stages_written": len(sidecar["stages"]), "skipped": skipped}
    if args.merge_existing is not None:
        counts["merged_existing"] = len(existing_stages)
    counts.update(summarize_stages(sidecar["stages"]))

    diff: dict[str, Any] | None = None
    if baseline_stages is not None:
        diff = diff_sidecars(baseline_stages, sidecar["stages"])
        counts["diff"] = diff

    if args.json:
        print(json.dumps(counts))
    else:
        if args.merge_existing is not None:
            line = (
                f"{counts['stages_written']} stage(s) written "
                f"(merged {counts['merged_existing']} existing); skipped: {skipped or 'none'}"
            )
        else:
            line = f"{counts['stages_written']} stage(s) written; skipped: {skipped or 'none'}"
        if diff is not None:
            line += (
                f"; diff: +{diff['added_count']}/-{diff['removed_count']}"
                f"/~{diff['changed_count']}/={diff['unchanged_count']}"
                f" (net {diff['net_delta']:+d})"
            )
        print(line)

    if args.fail_on_regressions and diff is not None and diff["removed_count"] > 0:
        print(
            f"error: regression detected: {diff['removed_count']} stage(s) removed compared to baseline",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
