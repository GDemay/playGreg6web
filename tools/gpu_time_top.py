#!/usr/bin/env python3
"""Print the top GPU-time pipelines from a boot report's `gpu_time` section.

The section is filled from WebGPU timestamp queries by whatever executes the command stream; when
the adapter has no `timestamp-query` or no executor fed results, the report says `available: false`
and this tool says so (it never estimates). Hashes match program_census.py / drops_by_program.

    python tools/gpu_time_top.py runs/x/boot-report.json [--top 20] [--json]
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
    ap.add_argument("boot_report", type=Path)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    s = summarize(json.loads(a.boot_report.read_text(encoding="utf-8")), a.top)
    sys.stdout.write(json.dumps(s, indent=2) + "\n" if a.json else render(s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
