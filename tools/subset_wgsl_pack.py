#!/usr/bin/env python3
"""Cut a browser-loadable WGPK v1 pack (+ `.reflect.json` sidecar) out of a larger (v1 or v2) pack.

    python tools/subset_wgsl_pack.py std.wgsl.pack --out browser.wgsl.pack \
        [--max-wgsl-bytes 262144] [--max-total-bytes 1600000000] \
        [--priority-pack splash.wgsl.pack ...] [--priority-hashes hashes.txt ...] \
        [--minify-wgsl] [--minify-mode {line,compact,aggressive}] [--json]

The web preloader (web/runtime/pgim_preloader.js) only parses WGPK v1 and holds the whole pack in
memory, so the output is always v1 (u32 offsets) and bounded by --max-total-bytes. Stages over
--max-wgsl-bytes are dropped (same cap as `pack-rdna2 --max-wgsl-bytes`). Selection order under the
total budget: stages named by --priority-pack / --priority-hashes first (still subject to the per
shader cap), then all other stages by ascending size. The sidecar keeps only the selected stages.
Output is ascending hash order. Streamed, so the source may be many GiB. Pure stdlib.
Never commit the result: it is derived from the title's shaders.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any

HEADER = struct.Struct("<4sIII")
ENTRY_V1 = struct.Struct("<QB3sII")
ENTRY_V2 = struct.Struct("<QB3sIQ")
U32_MAX = 0xFFFFFFFF
CHUNK = 8 * 1024 * 1024
TOC_CHUNK_ENTRIES = 4096
DEFAULT_MAX_WGSL_BYTES = 262144
DEFAULT_MAX_TOTAL_BYTES = 1_600_000_000
WGSL_MINIFY_MODES: tuple[str, ...] = ("line", "compact", "aggressive")


def _import_merge_tool() -> Any:
    try:
        return importlib.import_module("merge_wgsl_packs")
    except ModuleNotFoundError:
        tools_dir = str(Path(__file__).resolve().parent)
        if tools_dir not in sys.path:
            sys.path.insert(0, tools_dir)
        return importlib.import_module("merge_wgsl_packs")


def read_toc(path: Path) -> list[tuple[int, int, int, int]]:
    """Return [(hash, stage, length, offset)] in file order after bounds checks."""
    with open(path, "rb") as f:
        total = f.seek(0, os.SEEK_END)
        f.seek(0)
        hdr = f.read(HEADER.size)
        if len(hdr) < HEADER.size:
            raise ValueError(f"{path.name}: truncated WGPK header")
        magic, version, count, _flags = HEADER.unpack(hdr)
        if magic != b"WGPK" or version not in (1, 2):
            raise ValueError(f"{path.name}: not a WGPK v1/v2 pack")
        ent = ENTRY_V1 if version == 1 else ENTRY_V2
        data_start = HEADER.size + ent.size * count
        if total < data_start:
            raise ValueError(f"{path.name}: truncated entry table")
        out: list[tuple[int, int, int, int]] = []
        remaining = count
        idx = 0
        while remaining > 0:
            batch = min(TOC_CHUNK_ENTRIES, remaining)
            want = batch * ent.size
            raw = f.read(want)
            if len(raw) != want:
                raise ValueError(f"{path.name}: truncated entry table")
            for i in range(batch):
                if version == 1:
                    h, s, _r, off, ln = ENTRY_V1.unpack_from(raw, i * ent.size)
                else:
                    h, s, _r, ln, off = ENTRY_V2.unpack_from(raw, i * ent.size)
                if s not in (1, 2, 3):
                    raise ValueError(f"{path.name}: invalid stage code {s} at entry {idx}")
                if off < data_start or off + ln > total:
                    raise ValueError(f"{path.name}: entry {idx} payload exceeds pack bounds")
                out.append((h, s, ln, off))
                idx += 1
            remaining -= batch
    return out


def load_hashes(path: Path) -> set[int]:
    return {
        int(m, 16)
        for m in re.findall(r"0x[0-9a-fA-F]{1,16}|\b[0-9a-fA-F]{16}\b", path.read_text(encoding="utf-8"))
    }


def select(
    toc: list[tuple[int, int, int, int]],
    max_wgsl: int,
    max_total: int,
    priority: set[int],
) -> tuple[list[tuple[int, int, int, int]], int, int]:
    cand = [e for e in toc if 0 < e[2] <= max_wgsl]
    pri = sorted((e for e in cand if e[0] in priority), key=lambda e: (e[2], e[0]))
    rest = sorted((e for e in cand if e[0] not in priority), key=lambda e: (e[2], e[0]))
    chosen: list[tuple[int, int, int, int]] = []
    total = 0
    pri_kept = 0
    for is_pri, bucket in ((True, pri), (False, rest)):
        for e in bucket:
            if total + e[2] > max_total:
                continue
            chosen.append(e)
            total += e[2]
            if is_pri:
                pri_kept += 1
    chosen.sort(key=lambda e: e[0])
    return chosen, total, pri_kept


def write_pack(
    src: Path,
    chosen: list[tuple[int, int, int, int]],
    out: Path,
    *,
    minify_mode: str | None = None,
) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(chosen, key=lambda e: e[0])
    table = HEADER.size + ENTRY_V1.size * len(ordered)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")

    if minify_mode is not None:
        mp = _import_merge_tool()
        try:
            with open(src, "rb") as fin, open(tmp, "wb") as fo:
                fo.write(b"\0" * table)
                off = table
                entries: list[bytes] = []
                for h, s, ln, src_off in ordered:
                    fin.seek(src_off)
                    raw = fin.read(ln)
                    if len(raw) != ln:
                        raise ValueError("source pack truncated")
                    min_bytes = mp.minify_wgsl(raw.decode("utf-8"), mode=minify_mode).encode("utf-8")
                    if off + len(min_bytes) > U32_MAX:
                        raise ValueError("selection exceeds 4 GiB; lower --max-total-bytes")
                    fo.write(min_bytes)
                    entries.append(ENTRY_V1.pack(h, s, b"\0\0\0", off, len(min_bytes)))
                    off += len(min_bytes)
                fo.seek(0)
                fo.write(HEADER.pack(b"WGPK", 1, len(ordered), 0))
                fo.write(b"".join(entries))
            tmp.replace(out)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return off

    off, entries = table, []
    for h, s, ln, _o in ordered:
        entries.append(ENTRY_V1.pack(h, s, b"\0\0\0", off, ln))
        off += ln
    if off > U32_MAX:
        raise ValueError("selection exceeds 4 GiB; lower --max-total-bytes")
    try:
        with open(src, "rb") as fin, open(tmp, "wb") as fo:
            fo.write(HEADER.pack(b"WGPK", 1, len(ordered), 0))
            fo.write(b"".join(entries))
            for _h, _s, ln, o in ordered:
                fin.seek(o)
                left = ln
                while left:
                    b = fin.read(min(CHUNK, left))
                    if not b:
                        raise ValueError("source pack truncated")
                    fo.write(b)
                    left -= len(b)
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return off


def _filter_sidecar_stages(stages: dict[str, Any], keep_hashes: set[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in stages.items():
        if not isinstance(k, str):
            continue
        try:
            h = int(k.strip(), 16)
        except ValueError:
            continue
        if h in keep_hashes:
            out[f"0x{h:016x}"] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pack", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-wgsl-bytes", type=int, default=DEFAULT_MAX_WGSL_BYTES)
    ap.add_argument("--max-total-bytes", type=int, default=DEFAULT_MAX_TOTAL_BYTES)
    ap.add_argument("--priority-pack", type=Path, action="append", default=[])
    ap.add_argument("--priority-hashes", type=Path, action="append", default=[])
    ap.add_argument("--minify-wgsl", action="store_true")
    ap.add_argument("--minify-mode", choices=WGSL_MINIFY_MODES, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    minify_mode: str | None = a.minify_mode if a.minify_mode is not None else ("compact" if a.minify_wgsl else None)

    try:
        toc = read_toc(a.pack)
        priority: set[int] = set()
        for p in a.priority_pack:
            priority |= {e[0] for e in read_toc(p)}
        for p in a.priority_hashes:
            priority |= load_hashes(p)
        chosen, total, npri = select(toc, a.max_wgsl_bytes, a.max_total_bytes, priority)
        size = write_pack(a.pack, chosen, a.out, minify_mode=minify_mode)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    keep_hashes = {e[0] for e in chosen}
    side_in = Path(str(a.pack) + ".reflect.json")
    kept_reflect = None
    if side_in.is_file():
        d = json.loads(side_in.read_text(encoding="utf-8-sig"))
        d["stages"] = _filter_sidecar_stages(d.get("stages", {}), keep_hashes)
        kept_reflect = len(d["stages"])
        side_out = Path(str(a.out) + ".reflect.json")
        side_tmp = side_out.with_name(f".{side_out.name}.tmp.{os.getpid()}")
        try:
            side_tmp.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
            side_tmp.replace(side_out)
        except BaseException:
            side_tmp.unlink(missing_ok=True)
            raise

    payload_bytes = size - (HEADER.size + ENTRY_V1.size * len(chosen))
    rep: dict[str, Any] = {
        "source_stages": len(toc),
        "stages": len(chosen),
        "priority_stages_kept": npri,
        "payload_bytes": payload_bytes,
        "pack_bytes": size,
        "max_wgsl_bytes": a.max_wgsl_bytes,
        "reflect_stages": kept_reflect,
        "dropped_over_cap": sum(1 for e in toc if e[2] > a.max_wgsl_bytes),
    }
    if minify_mode is not None:
        rep["minify_mode"] = minify_mode
        rep["raw_payload_bytes"] = total
    print(json.dumps(rep) if a.json else "\n".join(f"{k}: {v}" for k, v in rep.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
