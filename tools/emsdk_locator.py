#!/usr/bin/env python3
"""Shared discovery of repo-local emsdk installs (work/**/emsdk).

An emsdk install is a directory named ``emsdk`` that contains
``upstream/emscripten`` (the dir that holds ``emcc``). The version is read from
``upstream/emscripten/emscripten-version.txt``.
"""

from __future__ import annotations

import os
from pathlib import Path

REQUIRED_EMSCRIPTEN_VERSION = "7.0.1"
MAX_SEARCH_DEPTH = 5  # levels below work/ that are scanned for an `emsdk` directory
_SKIP_DIRS = frozenset({"node_modules", ".git", "target", "upstream", "__pycache__"})


def read_emscripten_version(emscripten_dir: Path) -> str | None:
    try:
        text = (emscripten_dir / "emscripten-version.txt").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = text.strip().strip('"').strip("'").strip()
    return text or None


def find_work_emsdks(repo_root: Path) -> list[Path]:
    """Return every ``work/**/emsdk/upstream/emscripten`` directory (bounded depth)."""
    work = Path(repo_root) / "work"
    found: list[Path] = []
    if not work.is_dir():
        return found
    for dirpath, dirnames, _ in os.walk(work):
        depth = len(Path(dirpath).relative_to(work).parts)
        if "emsdk" in dirnames:
            cand = Path(dirpath) / "emsdk" / "upstream" / "emscripten"
            if cand.is_dir():
                found.append(cand)
        dirnames[:] = [d for d in dirnames if d != "emsdk" and d not in _SKIP_DIRS and depth + 1 < MAX_SEARCH_DEPTH]
    return found


def _mtime(path: Path) -> float:
    for p in (path / "emscripten-version.txt", path):
        try:
            return p.stat().st_mtime
        except OSError:
            continue
    return 0.0


def select_work_emsdk(
    repo_root: Path, required_version: str = REQUIRED_EMSCRIPTEN_VERSION
) -> tuple[Path, str | None] | None:
    """Pick the install matching ``required_version``, else the newest. Returns (dir, version)."""
    cands = [(c, read_emscripten_version(c)) for c in find_work_emsdks(repo_root)]
    if not cands:
        return None
    matching = [c for c in cands if c[1] == required_version]
    pool = matching or cands
    return max(pool, key=lambda c: (_mtime(c[0]), str(c[0])))


def describe(repo_root: Path, choice: tuple[Path, str | None], required_version: str = REQUIRED_EMSCRIPTEN_VERSION) -> str:
    path, version = choice
    try:
        shown = path.relative_to(repo_root)
    except ValueError:
        shown = path
    note = "" if version == required_version else f" (wanted {required_version})"
    return f"emsdk: using {shown} version {version or 'unknown'}{note}"


def main() -> int:
    """CLI: print the chosen work/**/emsdk (exit 0) or exit 1 when none exists."""
    repo = Path(__file__).resolve().parent.parent
    choice = select_work_emsdk(repo)
    if choice is None:
        return 1
    print(describe(repo, choice))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
