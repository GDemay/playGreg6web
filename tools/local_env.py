#!/usr/bin/env python3
"""Local workspace setup & toolchain check for playgta6web contributors and AI agents.

Stores user-specific paths in gitignored `.env` so tools/tests can inspect a local
game folder without ever copying or committing copyrighted files into the repo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / ".env"

KNOWN_KEYS = ("GTA6_GAME_DIR", "GTA6_DATA_DIR", "GTA6_WORK_DIR")
TOOLCHAIN_BINS = (
    "python",
    "cargo",
    "rustc",
    "node",
    "cmake",
    "emcc",
    "llvm-mc",
    "spirv-val",
    "naga",
)


def load_dotenv(env_path: Path = ENV_FILE) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def save_dotenv(updates: dict[str, str], env_path: Path = ENV_FILE) -> None:
    current = load_dotenv(env_path)
    current.setdefault("GTA6_DATA_DIR", "local-data")
    current.setdefault("GTA6_WORK_DIR", "work")
    current.update(updates)

    lines = [
        "# Local machine paths for playgta6web (gitignored - never commit)",
    ]
    for key in KNOWN_KEYS:
        val = current.get(key, "")
        lines.append(f"{key}={val}")
    for key, val in current.items():
        if key not in KNOWN_KEYS:
            lines.append(f"{key}={val}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def resolve_config(env_path: Path = ENV_FILE) -> dict[str, str]:
    file_vals = load_dotenv(env_path)
    cfg: dict[str, str] = {}
    for key in KNOWN_KEYS:
        cfg[key] = os.environ.get(key) or file_vals.get(key, "")
    if not cfg["GTA6_DATA_DIR"]:
        cfg["GTA6_DATA_DIR"] = "local-data"
    if not cfg["GTA6_WORK_DIR"]:
        cfg["GTA6_WORK_DIR"] = "work"
    return cfg


def inspect_game_dir(game_dir_str: str) -> dict[str, object]:
    if not game_dir_str:
        return {
            "configured": False,
            "exists": False,
            "path": "",
            "has_microsoft_game_config": False,
            "rpf_count": 0,
            "exe_count": 0,
        }
    p = Path(game_dir_str)
    if not p.is_dir():
        return {
            "configured": True,
            "exists": False,
            "path": str(p),
            "has_microsoft_game_config": False,
            "rpf_count": 0,
            "exe_count": 0,
        }

    has_config = False
    rpf_count = 0
    exe_count = 0
    # Shallow scan (root + 1 level) so huge folders stay instant.
    for entry in p.iterdir():
        name_lower = entry.name.lower()
        if entry.is_file():
            if name_lower == "microsoftgame.config":
                has_config = True
            elif name_lower.endswith(".rpf"):
                rpf_count += 1
            elif name_lower.endswith(".exe"):
                exe_count += 1
        elif entry.is_dir():
            try:
                for sub in entry.iterdir():
                    if not sub.is_file():
                        continue
                    sub_lower = sub.name.lower()
                    if sub_lower == "microsoftgame.config":
                        has_config = True
                    elif sub_lower.endswith(".rpf"):
                        rpf_count += 1
                    elif sub_lower.endswith(".exe"):
                        exe_count += 1
            except OSError:
                pass

    return {
        "configured": True,
        "exists": True,
        "path": str(p.resolve()),
        "has_microsoft_game_config": has_config,
        "rpf_count": rpf_count,
        "exe_count": exe_count,
    }


def check_environment(env_path: Path = ENV_FILE) -> dict[str, object]:
    cfg = resolve_config(env_path)
    game = inspect_game_dir(cfg["GTA6_GAME_DIR"])
    tools = {name: (shutil.which(name) is not None) for name in TOOLCHAIN_BINS}
    return {
        "env_file_exists": env_path.is_file(),
        "config": cfg,
        "game_dir": game,
        "tools": tools,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or configure local playgta6web paths (.env).")
    parser.add_argument("--check", action="store_true", help="Print local config and installed toolchains.")
    parser.add_argument("--json", action="store_true", help="Output JSON (with --check).")
    parser.add_argument("--set-game", metavar="PATH", help="Save GTA6_GAME_DIR into gitignored .env.")
    parser.add_argument("--set-data", metavar="PATH", help="Save GTA6_DATA_DIR into gitignored .env.")
    parser.add_argument("--set-work", metavar="PATH", help="Save GTA6_WORK_DIR into gitignored .env.")
    args = parser.parse_args(argv)

    updates: dict[str, str] = {}
    if args.set_game is not None:
        game_path = Path(args.set_game).expanduser()
        if not game_path.is_dir():
            print(f"Error: game directory does not exist: {game_path}", file=sys.stderr)
            return 1
        updates["GTA6_GAME_DIR"] = str(game_path.resolve())
    if args.set_data is not None:
        updates["GTA6_DATA_DIR"] = args.set_data
    if args.set_work is not None:
        updates["GTA6_WORK_DIR"] = args.set_work

    if updates:
        save_dotenv(updates)
        print(f"Updated {ENV_FILE.name}: {', '.join(sorted(updates))}")

    if args.check or not updates:
        report = check_environment()
        if args.json:
            print(json.dumps(report, indent=2))
            return 0

        game = report["game_dir"]
        tools = report["tools"]
        print("=== playgta6web Local Environment ===")
        if not game["configured"]:
            print("GTA6_GAME_DIR : NOT SET  -> run: python tools/local_env.py --set-game \"<path>\"")
        elif not game["exists"]:
            print(f"GTA6_GAME_DIR : MISSING ON DISK ({game['path']})")
        else:
            print(
                f"GTA6_GAME_DIR : OK ({game['path']}) "
                f"[MicrosoftGame.Config={game['has_microsoft_game_config']}, "
                f"rpf={game['rpf_count']}, exe={game['exe_count']}]"
            )
        print(f"GTA6_DATA_DIR : {report['config']['GTA6_DATA_DIR']}")
        print(f"GTA6_WORK_DIR : {report['config']['GTA6_WORK_DIR']}")
        installed = [k for k, ok in tools.items() if ok]
        missing = [k for k, ok in tools.items() if not ok]
        print(f"Tools ready   : {', '.join(installed) if installed else '(none)'}")
        print(f"Tools missing : {', '.join(missing) if missing else '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
