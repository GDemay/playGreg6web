#!/usr/bin/env python3
"""Print the top GPU-time pipelines from a boot report's `gpu_time` section.

The section is filled from WebGPU timestamp queries by whatever executes the command stream; when
the adapter has no `timestamp-query` or no executor fed results, the report says `available: false`
and this tool says so (it never estimates). Hashes match program_census.py / drops_by_program.

    python tools/gpu_time_top.py runs/x/boot-report.json [--top 20] [--json]

A live run with `--gpu-time-log <file>` writes one cumulative snapshot line per second; cut it into
phases by differencing two lines:

    python tools/gpu_time_top.py --log gpu-time.jsonl --phase menus=0-40 --phase bink=40-70 [--top 15]

(`name=START-END`, seconds since the log started; the snapshot at or before each bound is used.)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def summarize(report: dict[str, Any], top: int = 20) -> dict[str, Any]:
    g = report.get("gpu_time")
    if not isinstance(g, dict):
        return {"available": False, "reason": "boot report has no gpu_time section", "rows": []}
    if not g.get("available"):
        return {"available": False, "reason": g.get("reason") or g.get("support") or "unavailable", "rows": []}
    entries = [e for e in g.get("top", []) if isinstance(e, dict)]
    entries.sort(key=lambda e: (-int(e.get("total_ns", 0)), str(e.get("hash", ""))))
    total = int(g.get("total_ns", 0))
    rows = []
    for e in entries[:top]:
        t = int(e.get("total_ns", 0))
        n = int(e.get("passes", 0))
        rows.append({
            "hash": e.get("hash"), "kind": e.get("kind"), "passes": n, "total_ms": t / 1e6,
            "avg_us": (t / n / 1e3) if n else 0.0, "max_us": int(e.get("max_ns", 0)) / 1e3,
            "share": (t / total) if total else 0.0,
        })
    other = g.get("other") or {}
    return {
        "available": True, "total_ms": total / 1e6, "frames_measured": g.get("frames_measured", 0),
        "frames_skipped": g.get("frames_skipped", 0), "passes_skipped": g.get("passes_skipped", 0),
        "other_ms": int(other.get("total_ns", 0)) / 1e6, "other_passes": other.get("passes", 0), "rows": rows,
    }


def load_log(text: str) -> list[dict[str, Any]]:
    """Snapshot lines of --gpu-time-log, ordered by t_ms; broken lines (a cut-off last write) are skipped."""
    rows = []
    for line in text.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and isinstance(d.get("gpu_time"), dict):
            rows.append(d)
    rows.sort(key=lambda d: int(d.get("t_ms", 0)))
    return rows


def _at(rows: list[dict[str, Any]], t_s: float) -> dict[str, Any] | None:
    """The last snapshot at or before t_s seconds (None before the first one)."""
    best = None
    for d in rows:
        if int(d.get("t_ms", 0)) <= t_s * 1000:
            best = d
    return best


def phase(rows: list[dict[str, Any]], t0: float, t1: float, top: int = 15) -> dict[str, Any]:
    """GPU time between two snapshots: per (hash, kind) deltas, ms per presented frame, top shader share."""
    a, b = _at(rows, t0), _at(rows, t1)
    if b is None:
        return {"available": False, "reason": "no snapshot in the window", "rows": []}
    ga = (a or {}).get("gpu_time", {})
    gb = b["gpu_time"]
    if not gb.get("available"):
        return {"available": False, "reason": gb.get("reason") or "unavailable", "rows": []}
    base = {(e.get("hash"), e.get("kind")): e for e in ga.get("top", []) if isinstance(e, dict)}
    deltas = []
    for e in gb.get("top", []):
        if not isinstance(e, dict):
            continue
        o = base.get((e.get("hash"), e.get("kind")), {})
        n = int(e.get("passes", 0)) - int(o.get("passes", 0))
        t = int(e.get("total_ns", 0)) - int(o.get("total_ns", 0))
        if n > 0 or t > 0:
            deltas.append({"hash": e.get("hash"), "kind": e.get("kind"), "passes": n, "total_ns": t})
    deltas.sort(key=lambda d: (-d["total_ns"], str(d["hash"])))
    total = int(gb.get("total_ns", 0)) - int(ga.get("total_ns", 0))
    frames = int(b.get("presents", 0)) - int((a or {}).get("presents", 0))
    secs = (int(b.get("t_ms", 0)) - int((a or {}).get("t_ms", 0))) / 1000.0
    rows_out = [{"hash": d["hash"], "kind": d["kind"], "passes": d["passes"], "total_ms": d["total_ns"] / 1e6,
                 "ms_per_frame": (d["total_ns"] / 1e6 / frames) if frames else 0.0,
                 "share": (d["total_ns"] / total) if total > 0 else 0.0} for d in deltas[:top]]
    return {"available": True, "seconds": secs, "presents": frames, "total_ms": total / 1e6,
            "gpu_ms_per_frame": (total / 1e6 / frames) if frames else 0.0, "rows": rows_out,
            "dominant": bool(rows_out) and rows_out[0]["share"] >= 0.5}


def render_phase(name: str, p: dict[str, Any]) -> str:
    if not p["available"]:
        return "[%s] GPU time unavailable: %s\n" % (name, p["reason"])
    out = ["[%s] %.1f s, %d presents, GPU %.3f ms total, %.3f ms/frame%s" % (
        name, p["seconds"], p["presents"], p["total_ms"], p["gpu_ms_per_frame"],
        "  (one shader >= 50%)" if p["dominant"] else ""),
        "%-20s %-8s %8s %10s %9s %6s" % ("hash", "kind", "passes", "total_ms", "ms/frame", "share")]
    for r in p["rows"]:
        out.append("%-20s %-8s %8d %10.3f %9.4f %5.1f%%" % (
            r["hash"], r["kind"], r["passes"], r["total_ms"], r["ms_per_frame"], r["share"] * 100))
    return "\n".join(out) + "\n"


def render(s: dict[str, Any]) -> str:
    if not s["available"]:
        return "GPU time unavailable: %s\n" % s["reason"]
    out = ["GPU time: %.3f ms over %s frame(s) (%s skipped frames, %s skipped passes)" % (
        s["total_ms"], s["frames_measured"], s["frames_skipped"], s["passes_skipped"]),
        "%-20s %-8s %8s %10s %9s %9s %6s" % ("hash", "kind", "passes", "total_ms", "avg_us", "max_us", "share")]
    for r in s["rows"]:
        out.append("%-20s %-8s %8d %10.3f %9.1f %9.1f %5.1f%%" % (
            r["hash"], r["kind"], r["passes"], r["total_ms"], r["avg_us"], r["max_us"], r["share"] * 100))
    if s["other_passes"]:
        out.append("(untracked: %s passes, %.3f ms)" % (s["other_passes"], s["other_ms"]))
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boot_report", type=Path, nargs="?")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--log", type=Path, help="--gpu-time-log snapshot file (with --phase)")
    ap.add_argument("--phase", action="append", default=[], metavar="NAME=START-END",
                    help="seconds since the log started; repeatable")
    a = ap.parse_args(argv)
    if a.log:
        rows = load_log(a.log.read_text(encoding="utf-8"))
        res = {}
        for spec in a.phase:
            name, _, rng = spec.partition("=")
            lo, _, hi = rng.partition("-")
            res[name] = phase(rows, float(lo), float(hi), a.top)
        sys.stdout.write(json.dumps(res, indent=2) + "\n" if a.json else "".join(render_phase(n, p) for n, p in res.items()))
        return 0
    if not a.boot_report:
        ap.error("boot_report or --log is required")
    s = summarize(json.loads(a.boot_report.read_text(encoding="utf-8")), a.top)
    sys.stdout.write(json.dumps(s, indent=2) + "\n" if a.json else render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
