#!/usr/bin/env python3
"""Bisect which shader hash causes a hang by halving a --skip-shader-file list between runs.

    python tools/bisect_shader_skip.py init suspects.txt --state bisect.json
    python tools/bisect_shader_skip.py next --state bisect.json --out skip.txt
    #   run: playgta6_boot ... --skip-shader-file skip.txt   (does the hang / slowness still happen?)
    python tools/bisect_shader_skip.py report fixed|still-bad --state bisect.json
    #   repeat next / report until "culprit: 0x..." is printed

suspects.txt: one hash per line (`<hash> [note]`, '#' comments), e.g. the hashes from the boot report's
drop table or program_census.py. Each step skips the first half of the remaining suspects:
`fixed` (the problem went away) keeps that half, `still-bad` keeps the other half. The result is only
as good as the repro: a hang that needs two shaders together ends as "inconclusive" (see `report`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_hashes(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tok = line.split()[0]
        try:
            v = int(tok, 16)
        except ValueError:
            raise ValueError(f"bad shader hash {tok!r}") from None
        if not 0 < v < (1 << 64):
            raise ValueError(f"bad shader hash {tok!r}")
        h = f"0x{v:016x}"
        if h not in out:
            out.append(h)
    return out


def init_state(suspects: list[str]) -> dict:
    if not suspects:
        raise ValueError("no suspects")
    return {"suspects": suspects, "pending": None, "steps": 0, "culprit": None, "inconclusive": False}


def next_skip(state: dict) -> list[str]:
    """The hashes to skip in the next run (first half of the suspects); records it as pending."""
    s = state["suspects"]
    if state["culprit"] or state["inconclusive"]:
        return []
    if len(s) == 1:
        state["culprit"] = s[0]
        return []
    state["pending"] = s[: len(s) // 2]
    return list(state["pending"])


def apply_report(state: dict, fixed: bool) -> dict:
    p = state.get("pending")
    if not p:
        raise ValueError("no pending run: call `next` first")
    s = state["suspects"]
    state["steps"] += 1
    if fixed:
        state["suspects"] = list(p)
    else:
        rest = s[len(p):]
        # Skipping the first half did not help; if the other half is what remains, continue there.
        state["suspects"] = rest
    state["pending"] = None
    if len(state["suspects"]) == 1:
        state["culprit"] = state["suspects"][0]
    return state


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "next", "report", "status"])
    ap.add_argument("arg", nargs="?", help="init: suspects file; report: fixed | still-bad")
    ap.add_argument("--state", default="bisect_shader_skip.json")
    ap.add_argument("--out", default="skip_shaders.txt", help="next: the --skip-shader-file to write")
    a = ap.parse_args(argv)
    sp = Path(a.state)
    try:
        if a.cmd == "init":
            if not a.arg:
                ap.error("init needs the suspects file")
            st = init_state(parse_hashes(Path(a.arg).read_text(encoding="utf-8")))
            sp.write_text(json.dumps(st, indent=1))
            print(f"{len(st['suspects'])} suspects; run `next`")
            return 0
        st = json.loads(sp.read_text())
        if a.cmd == "next":
            skip = next_skip(st)
            sp.write_text(json.dumps(st, indent=1))
            if st["culprit"]:
                print(f"culprit: {st['culprit']}")
                return 0
            Path(a.out).write_text("".join(f"{h} bisect\n" for h in skip), encoding="utf-8")
            print(f"wrote {a.out}: skipping {len(skip)} of {len(st['suspects'])} suspects")
        elif a.cmd == "report":
            if a.arg not in ("fixed", "still-bad"):
                ap.error("report needs fixed or still-bad")
            apply_report(st, a.arg == "fixed")
            sp.write_text(json.dumps(st, indent=1))
            print(f"culprit: {st['culprit']}" if st["culprit"] else f"{len(st['suspects'])} suspects left; run `next`")
        else:
            print(json.dumps(st, indent=1))
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
