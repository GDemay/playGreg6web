#!/usr/bin/env python3
"""Union of several WGPK shader packs and their `.reflect.json` stage sidecars.

    python tools/merge_wgsl_packs.py a.wgsl.pack b.wgsl.pack --out merged.wgsl.pack [--force-v2] [--verify] [--json]

Each input `<pack>` may have a `<pack>.reflect.json` next to it (missing is fine). Stages are
deduplicated by stage hash. When several inputs carry the same hash, the one whose sidecar has an
entry wins, then the one with more bindings that have a resolved `source`, then the earlier input.
The payload and the sidecar entry both come from that winner. Output order is ascending hash, so
the result is deterministic. The output is WGPK v1 when it fits in u32 offsets, else v2 (or v2
explicitly when `--force-v2` is set). Packs are streamed (payloads are copied in chunks), so packs
larger than memory are fine. Pure stdlib.

The merged pack and sidecar are derived from the title's shaders: write them outside the
repository (e.g. under work/), never commit them.
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
from pathlib import Path
from typing import Any

HEADER = struct.Struct("<4sIII")
ENTRY_V1 = struct.Struct("<QB3sII")  # hash, stage, reserved, offset, length
ENTRY_V2 = struct.Struct("<QB3sIQ")  # hash, stage, reserved, length, offset
U32_MAX = 0xFFFFFFFF
CHUNK = 8 * 1024 * 1024
REFLECT_SCHEMA = "playgta6-stage-reflection"
SUPPORTED_REFLECT_VERSIONS = (1, 2, 3)


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


def _read_header_and_toc(path: Path) -> tuple[int, int, int, list[tuple[int, int, int, int]]]:
    """Return (version, total_bytes, data_start, [(hash, stage_code, offset, length)])."""
    label = _sanitize_label(path)
    try:
        with path.open("rb") as f:
            total = f.seek(0, os.SEEK_END)
            f.seek(0)
            raw = f.read(HEADER.size)
            if len(raw) < HEADER.size:
                raise ValueError(f"{label}: truncated WGPK header")
            magic, version, count, _flags = HEADER.unpack(raw)
            if magic != b"WGPK" or version not in (1, 2):
                raise ValueError(f"{label}: not a WGPK v1/v2 pack")
            ent = ENTRY_V2 if version == 2 else ENTRY_V1
            table_bytes = count * ent.size
            data_start = HEADER.size + table_bytes
            if total < data_start:
                raise ValueError(f"{label}: truncated entry table")
            table = f.read(table_bytes)
    except OSError as exc:
        raise ValueError(_format_os_error(exc, path)) from exc

    if len(table) != table_bytes:
        raise ValueError(f"{label}: truncated entry table")

    out: list[tuple[int, int, int, int]] = []
    seen: set[int] = set()
    non_empty_ranges: list[tuple[int, int, int]] = []
    for i in range(count):
        if version == 2:
            h, stage, _r, length, offset = ent.unpack_from(table, i * ent.size)
        else:
            h, stage, _r, offset, length = ent.unpack_from(table, i * ent.size)
        if stage not in (1, 2, 3):
            raise ValueError(f"{label}: invalid stage code {stage} at entry {i}")
        end = offset + length
        if offset < data_start or end > total:
            raise ValueError(f"{label}: entry {i} payload exceeds pack bounds")
        if h in seen:
            raise ValueError(f"{label}: duplicate hash 0x{h:016x}")
        seen.add(h)
        if length > 0:
            non_empty_ranges.append((offset, end, i))
        out.append((h, stage, offset, length))

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


def load_sidecar(pack: Path) -> dict[str, Any] | None:
    side = pack.with_name(pack.name + ".reflect.json")
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


def verify_merged_pack(out: Path, pack_paths: list[Path] | None = None) -> dict[str, Any]:
    """Run streaming structural, ascending-hash, UTF-8, payload-parity, and sidecar checks on `out`."""
    out_label = _sanitize_label(out)
    version, total_bytes, _data_start, toc = _read_header_and_toc(out)

    # Verify strictly ascending hash order.
    for i in range(1, len(toc)):
        if toc[i - 1][0] >= toc[i][0]:
            raise ValueError(
                f"{out_label}: hashes not strictly ascending at entry {i} "
                f"(0x{toc[i - 1][0]:016x} >= 0x{toc[i][0]:016x})"
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

    out_sidecar = load_sidecar(out)
    out_sidecar_label = _sanitize_label(out.with_name(out.name + ".reflect.json"))
    out_sidecar_map = _sidecar_stages_by_hash(out_sidecar, out_sidecar_label) if out_sidecar is not None else None
    if out_sidecar_map is not None:
        dangling = sorted(set(out_sidecar_map) - merged_hashes)
        if dangling:
            raise ValueError(
                f"{out_sidecar_label}: {len(dangling)} dangling stage hash(es) not in {out_label} "
                f"(first: 0x{dangling[0]:016x})"
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
            if got_len != exp_len:
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

    return {
        "source": out_label,
        "pack_version": version,
        "pack_bytes": total_bytes,
        "entries_verified": len(toc),
        "sidecar_entries_verified": len(out_sidecar_map) if out_sidecar_map is not None else None,
        "inputs_checked": inputs_checked,
        "verified": True,
    }


def merge(
    pack_paths: list[Path],
    out: Path,
    *,
    force_v2: bool = False,
    verify: bool = False,
) -> dict[str, Any]:
    tocs = [read_toc(p) for p in pack_paths]
    sidecars = [load_sidecar(p) for p in pack_paths]
    side_by_hash: list[dict[int, Any]] = [
        _sidecar_stages_by_hash(sc, _sanitize_label(p.with_name(p.name + ".reflect.json")))
        for p, sc in zip(pack_paths, sidecars)
    ]

    best: dict[int, tuple[tuple[int, int, int], int, int, int, int]] = {}
    seen_total = 0
    for idx, toc in enumerate(tocs):
        for h, stage, offset, length in toc:
            seen_total += 1
            entry = side_by_hash[idx].get(h)
            rank = (1 if entry is not None else 0, resolved_bindings(entry), -idx)
            cur = best.get(h)
            if cur is None or rank > cur[0]:
                best[h] = (rank, idx, stage, offset, length)

    order = sorted(best)
    total_payload = sum(best[h][4] for h in order)
    v1_size = HEADER.size + len(order) * ENTRY_V1.size + total_payload
    version = 2 if (force_v2 or v1_size > U32_MAX) else 1
    ent = ENTRY_V2 if version == 2 else ENTRY_V1
    data_start = HEADER.size + len(order) * ent.size

    toc_out = bytearray()
    cursor = data_start
    for h in order:
        _rank, _idx, stage, _off, length = best[h]
        if version == 2:
            toc_out += ent.pack(h, stage, b"\0\0\0", length, cursor)
        else:
            toc_out += ent.pack(h, stage, b"\0\0\0", cursor, length)
        cursor += length

    out.parent.mkdir(parents=True, exist_ok=True)
    handles: list[Any] = []
    try:
        for p in pack_paths:
            handles.append(p.open("rb"))
        with out.open("wb") as w:
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
        raise ValueError(_format_os_error(exc, out)) from exc
    finally:
        for fh in handles:
            fh.close()

    stages_out: dict[str, Any] = {}
    for h in order:
        entry = side_by_hash[best[h][1]].get(h)
        if entry is not None:
            stages_out["0x%016x" % h] = entry
    docs = [sc for sc in sidecars if sc]
    if docs:
        merged_doc = {k: v for k, v in docs[0].items() if k != "stages"}
        merged_doc["version"] = max(int(d.get("version", 1)) for d in docs)
        merged_doc["stages"] = stages_out
        sidecar_path = out.with_name(out.name + ".reflect.json")
        try:
            with sidecar_path.open("w", encoding="utf-8", newline="\n") as f:
                json.dump(merged_doc, f, indent=1)
                f.write("\n")
        except OSError as exc:
            raise ValueError(_format_os_error(exc, sidecar_path)) from exc

    if verify:
        verify_merged_pack(out, pack_paths)

    names = {1: "vertex", 2: "fragment", 3: "compute"}
    stage_counts = {"vertex": 0, "fragment": 0, "compute": 0}
    for h in order:
        stage_counts[names[best[h][2]]] += 1
    summary: dict[str, Any] = {
        "inputs": [_sanitize_label(p) for p in pack_paths],
        "input_entries": [len(t) for t in tocs],
        "input_entries_total": seen_total,
        "merged_entries": len(order),
        "duplicates_dropped": seen_total - len(order),
        "stage_counts": stage_counts,
        "sidecar_entries": len(stages_out) if docs else None,
        "pack_version": version,
        "pack_bytes": data_start + total_payload,
    }
    if verify:
        summary["verified"] = True
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("packs", nargs="+", type=Path, help="input .wgsl.pack files (earlier wins ties)")
    ap.add_argument("--out", required=True, type=Path, help="merged .wgsl.pack to write")
    ap.add_argument("--force-v2", action="store_true", help="emit WGPK v2 (64-bit offsets) explicitly")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="stream-verify the written WGPK pack (bounds, order, UTF-8, payload parity, sidecar)",
    )
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)
    outs = args.out.resolve()
    if any(p.resolve() == outs for p in args.packs):
        print("error: --out must differ from every input", file=sys.stderr)
        return 2
    try:
        summary = merge(args.packs, args.out, force_v2=args.force_v2, verify=args.verify)
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
