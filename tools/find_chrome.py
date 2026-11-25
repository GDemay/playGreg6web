"""Shared Chrome discovery for tools that drive headless Chrome (run_native_boot, replay_capture)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Mapping, Optional

_MAC = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
)
_LINUX = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")


def _windows_candidates(env: Mapping[str, str]) -> list[str]:
    out = []
    for var, sub in (("ProgramFiles", "Google/Chrome/Application/chrome.exe"),
                     ("ProgramFiles(x86)", "Google/Chrome/Application/chrome.exe"),
                     ("LOCALAPPDATA", "Google/Chrome/Application/chrome.exe"),
                     ("ProgramFiles(x86)", "Microsoft/Edge/Application/msedge.exe"),
                     ("ProgramFiles", "Microsoft/Edge/Application/msedge.exe")):
        base = env.get(var)
        if base:
            out.append(str(Path(base) / sub))
    return out


def find_chrome(env: Optional[Mapping[str, str]] = None, platform: Optional[str] = None,
                exists: Callable[[str], bool] = os.path.isfile,
                which: Callable[[str], Optional[str]] = shutil.which) -> Optional[str]:
    """Return a Chrome/Chromium (or Edge on Windows/macOS) executable path, or None."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform.startswith("win"):
        for c in _windows_candidates(env):
            if exists(c):
                return c
        return which("chrome") or which("msedge")
    if platform == "darwin":
        for c in _MAC:
            if exists(c):
                return c
        return None
    for name in _LINUX:
        found = which(name)
        if found:
            return found
    return None


def resolve_chrome(explicit: Optional[str] = None, env: Optional[Mapping[str, str]] = None,
                   log: Callable[[str], None] = print, **kw) -> Optional[str]:
    """--chrome, then $PLAYGTA6_CHROME, then auto-detection; logs which browser was picked."""
    env = os.environ if env is None else env
    if explicit:
        return explicit
    if env.get("PLAYGTA6_CHROME"):
        return env["PLAYGTA6_CHROME"]
    found = find_chrome(env=env, **kw)
    if found:
        log(f"chrome: auto-detected {found}")
    return found
