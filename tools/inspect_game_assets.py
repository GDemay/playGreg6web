#!/usr/bin/env python3
"""MicrosoftGame.Config & Xbox package readiness inspector CLI (Track 5 / W39, T88).

Safely parses Xbox GDK `MicrosoftGame.Config` XML (rejecting DTD/XXE declarations
and capping input size), cross-checks declared executables and `ShellVisuals`
assets for presence, size, and non-zero header completeness (detecting partial
downloads), and catalogs bundled `.dll`s and deterministically sorted `.rpf`
archive inventory & zero-division-safe size/ratio summaries.

Security & privacy invariants:
- Rejects `<!DOCTYPE` / `<!ENTITY` declarations before XML parsing to block XXE.
- Caps `MicrosoftGame.Config` input size before loading into memory.
- Enforces package-root containment (blocks `..` traversal and external symlinks).
- Machine-readable `--json` output NEVER includes host absolute paths, UNC paths,
  usernames, copyrighted `.rpf` archive names, or raw file bytes.
- Strictly uses Python 3.12 standard library modules only.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Mapping, Sequence

DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1

# Cap MicrosoftGame.Config size at 1 MiB (real Xbox GDK configs are typically a few KiB).
MAX_CONFIG_BYTES = 1 * 1024 * 1024

# Probe the first 520 bytes (0x208, matching tools/src/scan.rs) to detect sparse/zero-filled
# partial downloads and classify common executable/image magic bytes without storing raw bytes.
HEADER_PROBE_BYTES = 0x208

# Maximum subdirectory depth when cataloging bundled DLLs and RPFs.
MAX_WALK_DEPTH = 6

SHELL_VISUAL_IMAGE_FIELDS: tuple[tuple[str, str], ...] = (
    ("splashScreenImage", "SplashScreenImage"),
    ("storeLogo", "StoreLogo"),
    ("square150x150Logo", "Square150x150Logo"),
    ("square44x44Logo", "Square44x44Logo"),
    ("square480x480Logo", "Square480x480Logo"),
)

_FORBIDDEN_XML_DECL_RE = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
_WINDOWS_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_FILE_URL_RE = re.compile(r"\bfile://[^\s\"'<>|,;()\[\]]+", re.IGNORECASE)
_UNC_BS_PATH_RE = re.compile(r"\\\\[^\s\"'<>|,;()\[\]]+")
_UNC_FS_PATH_RE = re.compile(r"(?<!:)//[^\s\"'<>|,;()\[\]]+")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s\"'<>|,;()\[\]]*")
_POSIX_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])/(?:Users|home|tmp|var|opt|root|mnt|private|etc|usr|Library|Volumes|proc|run|srv|media|cygdrive)/[^\s\"'<>|,;()\[\]]*"
)
_TILDE_HOME_RE = re.compile(r"(?<![A-Za-z0-9_])~[A-Za-z0-9._-]*/[^\s\"'<>|,;()\[\]]*")


class GameConfigError(ValueError):
    """Raised when `MicrosoftGame.Config` cannot be safely parsed or validated."""


class UnsafeXmlError(GameConfigError):
    """Raised when XML input contains forbidden DTD or entity declarations."""


class ConfigTooLargeError(GameConfigError):
    """Raised when XML input exceeds the maximum allowed byte size."""


def safe_ratio(numerator: int | float, denominator: int | float, *, digits: int = 6) -> float:
    """Compute `numerator / denominator` safely, returning `0.0` when `denominator <= 0`."""
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), digits)


def _current_usernames() -> list[str]:
    names: list[str] = []
    candidates: list[str] = [
        os.environ.get("USERNAME", ""),
        os.environ.get("USER", ""),
        os.environ.get("LOGNAME", ""),
    ]
    for env_home_var in ("USERPROFILE", "HOME"):
        env_home = os.environ.get(env_home_var, "").strip().rstrip("\\/")
        if env_home:
            tail = re.split(r"[/\\]+", env_home)[-1]
            candidates.append(tail)
    try:
        candidates.append(getpass.getuser())
    except (OSError, KeyError, RuntimeError):
        pass
    try:
        candidates.append(Path.home().name)
    except (OSError, RuntimeError):
        pass

    for candidate in candidates:
        cleaned = candidate.strip()
        if len(cleaned) >= 3 and cleaned not in names:
            names.append(cleaned)
    return names


def _sensitive_host_paths(extra_paths: Sequence[Path | str] = ()) -> list[str]:
    """Collect concrete host filesystem path prefixes that must never leak."""
    raw_candidates: list[str] = []
    for env_var in ("USERPROFILE", "HOME"):
        val = os.environ.get(env_var, "").strip()
        if val:
            raw_candidates.append(val)
    try:
        home = Path.home()
        raw_candidates.append(str(home))
        raw_candidates.append(str(home.resolve()))
    except (OSError, RuntimeError):
        pass

    for item in extra_paths:
        if not item:
            continue
        raw_str = str(item)
        raw_candidates.append(raw_str)
        try:
            raw_candidates.append(str(Path(item).resolve()))
        except (OSError, RuntimeError):
            pass

    variants: list[str] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        for text in (raw, raw.replace("\\", "/"), raw.replace("/", "\\")):
            cleaned = text.rstrip("\\/")
            if len(cleaned) < 4:
                continue
            key = cleaned.lower() if (os.name == "nt" or _WINDOWS_DRIVE_PREFIX_RE.match(cleaned)) else cleaned
            if key not in seen:
                seen.add(key)
                variants.append(cleaned)
    variants.sort(key=len, reverse=True)
    return variants


def _compile_host_prefix_patterns(extra_paths: Sequence[Path | str] = ()) -> list[re.Pattern[str]]:
    patterns: list[re.Pattern[str]] = []
    for host_prefix in _sensitive_host_paths(extra_paths):
        flags = re.IGNORECASE if (os.name == "nt" or _WINDOWS_DRIVE_PREFIX_RE.match(host_prefix)) else 0
        patterns.append(
            re.compile(
                rf"{re.escape(host_prefix)}(?:[\\/][^\s\"'<>|,;()\[\]]*)?",
                flags,
            )
        )
    return patterns


def _compile_username_patterns() -> list[re.Pattern[str]]:
    return [
        re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(username)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        for username in _current_usernames()
    ]


def _sanitize_text_with_patterns(
    text: str,
    prefix_patterns: Sequence[re.Pattern[str]],
    username_patterns: Sequence[re.Pattern[str]],
) -> str:
    sanitized = _FILE_URL_RE.sub("<path>", text)
    for prefix_re in prefix_patterns:
        sanitized = prefix_re.sub("<path>", sanitized)
    sanitized = _UNC_BS_PATH_RE.sub("<path>", sanitized)
    sanitized = _WINDOWS_PATH_RE.sub("<path>", sanitized)
    sanitized = _UNC_FS_PATH_RE.sub("<path>", sanitized)
    sanitized = _POSIX_PATH_RE.sub("<path>", sanitized)
    sanitized = _TILDE_HOME_RE.sub("<path>", sanitized)

    for user_re in username_patterns:
        sanitized = user_re.sub("<user>", sanitized)

    return sanitized


def sanitize_text(
    raw: str | None,
    *,
    extra_paths: Sequence[Path | str] = (),
) -> str | None:
    """Strip host filesystem paths, UNC paths, file URLs, and active usernames."""
    if raw is None:
        return None
    return _sanitize_text_with_patterns(
        str(raw),
        _compile_host_prefix_patterns(extra_paths),
        _compile_username_patterns(),
    )


def sanitize_report(
    value: Any,
    *,
    extra_paths: Sequence[Path | str] = (),
) -> Any:
    """Recursively sanitize every string, path, key, and raw byte blob in a report."""
    prefix_patterns = _compile_host_prefix_patterns(extra_paths)
    username_patterns = _compile_username_patterns()

    def _walk(node: Any) -> Any:
        if isinstance(node, (bytes, bytearray, memoryview)):
            return f"<bytes:{len(node)}>"
        if isinstance(node, Path):
            return _sanitize_text_with_patterns(node.name, prefix_patterns, username_patterns)
        if isinstance(node, str):
            return _sanitize_text_with_patterns(node, prefix_patterns, username_patterns)
        if isinstance(node, Mapping):
            return {
                _sanitize_text_with_patterns(str(key), prefix_patterns, username_patterns): _walk(val)
                for key, val in node.items()
            }
        if isinstance(node, (list, tuple)):
            return [_walk(item) for item in node]
        return node

    return _walk(value)


def load_dotenv(env_path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE `.env` file without mutating `os.environ`."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    try:
        content = env_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return values

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            values[key] = val
    return values


def resolve_game_dir_from_env(
    repo_root: Path = DEFAULT_REPO_ROOT,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve `GTA6_GAME_DIR` from environment variables or `<repo_root>/.env`."""
    env_map = os.environ if env is None else env
    dotenv_vals = load_dotenv(repo_root / ".env")
    raw_game = (env_map.get("GTA6_GAME_DIR") or dotenv_vals.get("GTA6_GAME_DIR", "")).strip()
    if not raw_game:
        return None
    candidate = Path(raw_game).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _find_child(elem: ET.Element, name: str) -> ET.Element | None:
    target = name.lower()
    for child in elem:
        if _local_name(child.tag).lower() == target:
            return child
    return None


def _iter_children(elem: ET.Element, name: str) -> list[ET.Element]:
    target = name.lower()
    return [child for child in elem if _local_name(child.tag).lower() == target]


def _get_attr(elem: ET.Element | None, name: str) -> str | None:
    if elem is None:
        return None
    direct = elem.attrib.get(name)
    if direct is not None:
        cleaned = direct.strip()
        return cleaned if cleaned else None
    target = name.lower()
    for key, val in elem.attrib.items():
        if _local_name(key).lower() == target:
            cleaned = val.strip()
            return cleaned if cleaned else None
    return None


def _get_text(elem: ET.Element | None) -> str | None:
    if elem is None or elem.text is None:
        return None
    cleaned = elem.text.strip()
    return cleaned if cleaned else None


def _parse_optional_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    lower = raw.strip().lower()
    if lower in ("true", "1", "yes"):
        return True
    if lower in ("false", "0", "no"):
        return False
    return None


def _is_unsafe_absolute_path(raw: str) -> bool:
    if "\x00" in raw:
        return True
    if _WINDOWS_DRIVE_PREFIX_RE.match(raw):
        return True
    if raw.startswith(("\\\\", "//", "/", "\\")):
        return True
    return False


def _normalize_declared_path(raw: str | None) -> str | None:
    """Normalize a package-relative path declared in `MicrosoftGame.Config`."""
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if (
        _is_unsafe_absolute_path(cleaned)
        or bool(_FILE_URL_RE.match(cleaned))
        or bool(_TILDE_HOME_RE.match(cleaned))
    ):
        return "<path>"

    parts = [part for part in cleaned.replace("\\", "/").split("/") if part and part != "."]
    if not parts:
        return None
    sanitized = sanitize_text("/".join(parts))
    return sanitized if sanitized else "<path>"


def _validate_package_relative_path(raw: str | None) -> tuple[list[str] | None, str | None]:
    """Validate that a declared path is strictly relative and cannot escape package root."""
    if not raw or not raw.strip():
        return None, "empty_path"
    cleaned = raw.strip()
    if (
        _is_unsafe_absolute_path(cleaned)
        or cleaned == "<path>"
        or bool(_FILE_URL_RE.match(cleaned))
        or bool(_TILDE_HOME_RE.match(cleaned))
    ):
        return None, "unsafe_path"
    parts = [part for part in cleaned.replace("\\", "/").split("/") if part and part != "."]
    if not parts:
        return None, "empty_path"
    if any(part == ".." or ":" in part or "\x00" in part for part in parts):
        return None, "unsafe_path"
    return parts, None


def _decode_xml_bytes(raw_bytes: bytes) -> str:
    if raw_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return raw_bytes.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise GameConfigError("MicrosoftGame.Config is not valid UTF-16 XML") from exc
    try:
        return raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GameConfigError("MicrosoftGame.Config is not valid UTF-8 XML") from exc


def _read_config_file_bounded(path: Path, max_bytes: int) -> str:
    if not path.is_file():
        raise FileNotFoundError("MicrosoftGame.Config file not found")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise GameConfigError("Unable to stat MicrosoftGame.Config") from exc
    if size > max_bytes:
        raise ConfigTooLargeError(
            f"MicrosoftGame.Config exceeds maximum allowed size ({max_bytes} bytes)"
        )
    try:
        with path.open("rb") as fh:
            raw_bytes = fh.read(max_bytes + 1)
    except OSError as exc:
        raise GameConfigError("Unable to read MicrosoftGame.Config") from exc
    if len(raw_bytes) > max_bytes:
        raise ConfigTooLargeError(
            f"MicrosoftGame.Config exceeds maximum allowed size ({max_bytes} bytes)"
        )
    return _decode_xml_bytes(raw_bytes)


def _extract_xml_text(xml_text_or_path: str | bytes | bytearray | os.PathLike[str], max_bytes: int) -> str:
    if isinstance(xml_text_or_path, os.PathLike):
        return _read_config_file_bounded(Path(xml_text_or_path), max_bytes)

    if isinstance(xml_text_or_path, (bytes, bytearray)):
        if len(xml_text_or_path) > max_bytes:
            raise ConfigTooLargeError(
                f"MicrosoftGame.Config exceeds maximum allowed size ({max_bytes} bytes)"
            )
        return _decode_xml_bytes(bytes(xml_text_or_path))

    if isinstance(xml_text_or_path, str):
        stripped = xml_text_or_path.lstrip("\ufeff").lstrip()
        if not stripped.startswith("<") and "\n" not in xml_text_or_path and "\r" not in xml_text_or_path and "\x00" not in xml_text_or_path:
            candidate = Path(xml_text_or_path)
            try:
                if candidate.is_file():
                    return _read_config_file_bounded(candidate, max_bytes)
            except OSError:
                pass
        encoded_size = len(xml_text_or_path.encode("utf-8"))
        if encoded_size > max_bytes:
            raise ConfigTooLargeError(
                f"MicrosoftGame.Config exceeds maximum allowed size ({max_bytes} bytes)"
            )
        return xml_text_or_path.lstrip("\ufeff")

    raise TypeError("xml_text_or_path must be a str, bytes, or PathLike object")


def parse_microsoft_game_config(
    xml_text_or_path: str | bytes | bytearray | os.PathLike[str],
    *,
    max_bytes: int = MAX_CONFIG_BYTES,
) -> dict[str, Any]:
    """Safely parse an Xbox GDK `MicrosoftGame.Config` XML string or file path.

    Rejects `<!DOCTYPE` and `<!ENTITY` declarations to block XXE attacks and enforces
    a strict input size cap (`max_bytes`, default 1 MiB).
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")

    xml_text = _extract_xml_text(xml_text_or_path, max_bytes=max_bytes)
    if not xml_text.strip():
        raise GameConfigError("MicrosoftGame.Config is empty")

    if _FORBIDDEN_XML_DECL_RE.search(xml_text):
        raise UnsafeXmlError(
            "Forbidden XML declaration (<!DOCTYPE or <!ENTITY) rejected to prevent XXE"
        )

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise GameConfigError("Malformed XML in MicrosoftGame.Config") from exc

    if _local_name(root.tag).lower() != "game":
        raise GameConfigError("Expected <Game> root element in MicrosoftGame.Config")

    identity_el = _find_child(root, "Identity")
    identity: dict[str, str | None] = {
        "name": sanitize_text(_get_attr(identity_el, "Name")),
        "publisher": sanitize_text(_get_attr(identity_el, "Publisher")),
        "version": sanitize_text(_get_attr(identity_el, "Version")),
    }

    executables: list[dict[str, str | None]] = []
    exe_list_el = _find_child(root, "ExecutableList")
    if exe_list_el is not None:
        for exe_el in _iter_children(exe_list_el, "Executable"):
            executables.append(
                {
                    "name": _normalize_declared_path(_get_attr(exe_el, "Name")),
                    "id": sanitize_text(_get_attr(exe_el, "Id")),
                    "targetDeviceFamily": sanitize_text(_get_attr(exe_el, "TargetDeviceFamily")),
                    "alias": sanitize_text(_get_attr(exe_el, "Alias")),
                }
            )

    sv_el = _find_child(root, "ShellVisuals")
    shell_visuals: dict[str, str | None] = {
        "defaultDisplayName": sanitize_text(_get_attr(sv_el, "DefaultDisplayName")),
        "splashScreenImage": _normalize_declared_path(_get_attr(sv_el, "SplashScreenImage")),
        "storeLogo": _normalize_declared_path(_get_attr(sv_el, "StoreLogo")),
        "square150x150Logo": _normalize_declared_path(_get_attr(sv_el, "Square150x150Logo")),
        "square44x44Logo": _normalize_declared_path(_get_attr(sv_el, "Square44x44Logo")),
        "square480x480Logo": _normalize_declared_path(_get_attr(sv_el, "Square480x480Logo")),
        "backgroundColor": sanitize_text(_get_attr(sv_el, "BackgroundColor")),
    }

    languages: list[str] = []
    resources_el = _find_child(root, "Resources")
    if resources_el is not None:
        for res_el in _iter_children(resources_el, "Resource"):
            lang = sanitize_text(_get_attr(res_el, "Language"))
            if lang and lang not in languages:
                languages.append(lang)

    vm_el = _find_child(root, "VirtualMachine")
    virtual_machine: dict[str, Any] | None = None
    if vm_el is not None:
        virtual_machine = {"present": True}
        for key, val in vm_el.attrib.items():
            cleaned = sanitize_text(val.strip())
            if cleaned is not None and cleaned != "":
                virtual_machine[_local_name(key)] = cleaned
        for child in vm_el:
            txt = sanitize_text(_get_text(child))
            if txt is not None:
                virtual_machine[_local_name(child.tag)] = txt

    dr_el = _find_child(root, "DesktopRegistration")
    dr_mod_folder_el = _find_child(dr_el, "ModFolder") if dr_el is not None else None
    desktop_registration: dict[str, Any] | None = None
    if dr_el is not None:
        desktop_registration = {
            "present": True,
            "hasModFolder": dr_mod_folder_el is not None,
            "childTags": sorted({_local_name(child.tag) for child in dr_el}),
        }

    root_mod_folder_el = _find_child(root, "ModFolder")
    mod_folder_el = root_mod_folder_el if root_mod_folder_el is not None else dr_mod_folder_el
    mod_folder: dict[str, Any] | None = None
    if mod_folder_el is not None:
        mod_folder = {
            "name": _normalize_declared_path(
                _get_attr(mod_folder_el, "Name") or _get_text(mod_folder_el)
            ),
            "readOnly": _parse_optional_bool(_get_attr(mod_folder_el, "ReadOnly")),
        }

    return sanitize_report(
        {
            "configVersion": sanitize_text(_get_attr(root, "configVersion")),
            "identity": identity,
            "titleId": sanitize_text(_get_text(_find_child(root, "TitleId"))),
            "storeId": sanitize_text(_get_text(_find_child(root, "StoreId"))),
            "msaAppId": sanitize_text(_get_text(_find_child(root, "MSAAppId"))),
            "executables": executables,
            "shellVisuals": shell_visuals,
            "resources": {"languages": languages},
            "virtualMachine": virtual_machine,
            "modFolder": mod_folder,
            "desktopRegistration": desktop_registration,
        }
    )


def _locate_microsoft_game_config(game_dir: Path) -> Path | None:
    """Locate `MicrosoftGame.Config` case-insensitively at root or 1 level deep."""
    try:
        root_entries = sorted(game_dir.iterdir(), key=lambda p: (p.name.lower(), p.name))
    except OSError:
        return None

    for entry in root_entries:
        try:
            if entry.is_file() and entry.name.lower() == "microsoftgame.config":
                return entry
        except OSError:
            continue

    for entry in root_entries:
        try:
            if not entry.is_dir() or entry.is_symlink():
                continue
            sub_entries = sorted(entry.iterdir(), key=lambda p: (p.name.lower(), p.name))
        except OSError:
            continue
        for sub in sub_entries:
            try:
                if sub.is_file() and sub.name.lower() == "microsoftgame.config":
                    return sub
            except OSError:
                continue

    return None


def _resolve_case_insensitive_in_root(root: Path, root_resolved: Path, parts: list[str]) -> Path | None:
    """Resolve package-relative path segments case-insensitively without escaping `root`."""
    current = root
    for idx, part in enumerate(parts):
        direct = current / part
        matched: Path | None = None
        try:
            if direct.exists():
                matched = direct
        except OSError:
            matched = None

        if matched is None:
            try:
                target_lower = part.lower()
                for entry in sorted(current.iterdir(), key=lambda p: (p.name.lower(), p.name)):
                    if entry.name.lower() == target_lower:
                        matched = entry
                        break
            except OSError:
                return None

        if matched is None:
            return None

        try:
            resolved_step = matched.resolve()
        except OSError:
            return None
        if not resolved_step.is_relative_to(root_resolved):
            return None

        if idx < len(parts) - 1 and not matched.is_dir():
            return None
        current = matched

    return current


def _classify_header_format(header: bytes, size_bytes: int, non_zero: bool) -> str:
    if size_bytes == 0:
        return "empty"
    if not non_zero:
        return "zero_filled"
    if header.startswith(b"MZ"):
        return "pe"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if header.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if header.startswith(b"BM"):
        return "bmp"
    if header.startswith(b"DDS "):
        return "dds"
    if header.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "webp"
    if header.startswith((b"RPF9", b"9FPR", b"RPF8", b"8FPR", b"RPF7", b"7FPR")):
        return "rpf"
    return "binary"


def _probe_file_completeness(path: Path) -> dict[str, Any]:
    """Inspect a file's size and leading header completeness without returning raw bytes."""
    try:
        if not path.is_file():
            return {
                "present": False,
                "sizeBytes": 0,
                "nonZeroHeader": False,
                "headerFormat": None,
                "partialDownloadSuspected": False,
                "complete": False,
                "status": "missing",
            }
        size_bytes = path.stat().st_size
    except OSError:
        return {
            "present": True,
            "sizeBytes": 0,
            "nonZeroHeader": False,
            "headerFormat": None,
            "partialDownloadSuspected": False,
            "complete": False,
            "status": "unreadable",
        }

    if size_bytes == 0:
        return {
            "present": True,
            "sizeBytes": 0,
            "nonZeroHeader": False,
            "headerFormat": "empty",
            "partialDownloadSuspected": True,
            "complete": False,
            "status": "empty",
        }

    try:
        with path.open("rb") as fh:
            header = fh.read(HEADER_PROBE_BYTES)
    except OSError:
        return {
            "present": True,
            "sizeBytes": size_bytes,
            "nonZeroHeader": False,
            "headerFormat": None,
            "partialDownloadSuspected": False,
            "complete": False,
            "status": "unreadable",
        }

    non_zero = any(byte != 0 for byte in header)
    header_format = _classify_header_format(header, size_bytes, non_zero)
    if not non_zero:
        return {
            "present": True,
            "sizeBytes": size_bytes,
            "nonZeroHeader": False,
            "headerFormat": header_format,
            "partialDownloadSuspected": True,
            "complete": False,
            "status": "incomplete_zero_header",
        }

    return {
        "present": True,
        "sizeBytes": size_bytes,
        "nonZeroHeader": True,
        "headerFormat": header_format,
        "partialDownloadSuspected": False,
        "complete": True,
        "status": "ok",
    }


def _relative_posix(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _check_declared_asset(
    declared_path: str | None,
    *,
    package_root: Path,
    package_root_resolved: Path,
    game_dir_resolved: Path,
) -> dict[str, Any]:
    parts, err = _validate_package_relative_path(declared_path)
    if err is not None or parts is None:
        safe_declared: str | None
        if declared_path is None:
            safe_declared = None
        else:
            stripped_declared = declared_path.strip()
            if (
                _is_unsafe_absolute_path(stripped_declared)
                or bool(_FILE_URL_RE.match(stripped_declared))
                or bool(_TILDE_HOME_RE.match(stripped_declared))
            ):
                safe_declared = "<path>"
            else:
                safe_declared = sanitize_text(declared_path)
        return {
            "declaredPath": safe_declared,
            "relativePath": None,
            "present": False,
            "sizeBytes": 0,
            "nonZeroHeader": False,
            "headerFormat": None,
            "partialDownloadSuspected": False,
            "complete": False,
            "status": err or "unsafe_path",
        }

    resolved_file = _resolve_case_insensitive_in_root(package_root, package_root_resolved, parts)
    if resolved_file is None or not resolved_file.is_file():
        return {
            "declaredPath": "/".join(parts),
            "relativePath": None,
            "present": False,
            "sizeBytes": 0,
            "nonZeroHeader": False,
            "headerFormat": None,
            "partialDownloadSuspected": False,
            "complete": False,
            "status": "missing",
        }

    probe = _probe_file_completeness(resolved_file)
    try:
        rel_path = _relative_posix(resolved_file.resolve(), game_dir_resolved)
    except ValueError:
        return {
            "declaredPath": "/".join(parts),
            "relativePath": None,
            "present": False,
            "sizeBytes": 0,
            "nonZeroHeader": False,
            "headerFormat": None,
            "partialDownloadSuspected": False,
            "complete": False,
            "status": "unsafe_path",
        }

    return {
        "declaredPath": "/".join(parts),
        "relativePath": rel_path,
        **probe,
    }


def _archive_path_hash(rel_posix: str) -> str:
    """Return a deterministic 16-hex-char SHA-256 digest of a normalized relative path."""
    normalized = rel_posix.replace("\\", "/").strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def sort_archive_inventory(archives: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return a deterministically sorted copy of sanitized archive inventory entries.

    Orders largest archives first (`sizeBytes` descending) with deterministic tie-breaking
    on `(pathHash, extension, depth, status)` so ordering is independent of filesystem
    traversal order.
    """
    ordered = [dict(item) for item in archives]
    ordered.sort(
        key=lambda item: (
            -int(item.get("sizeBytes", 0)),
            str(item.get("pathHash", "")),
            str(item.get("extension", "")),
            int(item.get("depth", 0)),
            str(item.get("status", "")),
        )
    )
    return ordered


def summarize_archive_inventory(archives: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute zero-division-safe size and ratio summary metrics for an archive inventory."""
    archive_count = len(archives)
    complete_count = 0
    incomplete_count = 0
    empty_count = 0
    zero_header_count = 0
    unreadable_count = 0
    total_bytes = 0
    complete_bytes = 0
    incomplete_bytes = 0
    sizes: list[int] = []
    status_counts: dict[str, int] = {}
    format_counts: dict[str, int] = {}

    for item in archives:
        size_bytes = max(0, int(item.get("sizeBytes", 0)))
        sizes.append(size_bytes)
        total_bytes += size_bytes

        complete = bool(item.get("complete", False))
        if complete:
            complete_count += 1
            complete_bytes += size_bytes
        else:
            incomplete_count += 1
            incomplete_bytes += size_bytes

        status = str(item.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == "empty":
            empty_count += 1
        elif status == "incomplete_zero_header":
            zero_header_count += 1
        elif status == "unreadable":
            unreadable_count += 1

        header_fmt = str(item.get("headerFormat") or "unknown")
        format_counts[header_fmt] = format_counts.get(header_fmt, 0) + 1

    min_bytes = min(sizes) if sizes else 0
    max_bytes = max(sizes) if sizes else 0

    return {
        "archiveCount": archive_count,
        "completeCount": complete_count,
        "incompleteCount": incomplete_count,
        "emptyCount": empty_count,
        "zeroHeaderCount": zero_header_count,
        "unreadableCount": unreadable_count,
        "totalBytes": total_bytes,
        "completeBytes": complete_bytes,
        "incompleteBytes": incomplete_bytes,
        "minBytes": min_bytes,
        "maxBytes": max_bytes,
        "avgBytes": safe_ratio(total_bytes, archive_count, digits=2),
        "completeCountRatio": safe_ratio(complete_count, archive_count),
        "incompleteCountRatio": safe_ratio(incomplete_count, archive_count),
        "completeByteRatio": safe_ratio(complete_bytes, total_bytes),
        "incompleteByteRatio": safe_ratio(incomplete_bytes, total_bytes),
        "largestArchiveByteRatio": safe_ratio(max_bytes, total_bytes),
        "statusCounts": {key: status_counts[key] for key in sorted(status_counts)},
        "headerFormatCounts": {key: format_counts[key] for key in sorted(format_counts)},
    }


def _empty_inventory() -> dict[str, Any]:
    return {
        "dlls": [],
        "dllCount": 0,
        "dllCompleteCount": 0,
        "dllIncompleteCount": 0,
        "dllTotalBytes": 0,
        "rpfCount": 0,
        "rpfCompleteCount": 0,
        "rpfIncompleteCount": 0,
        "rpfTotalBytes": 0,
        "archives": [],
        "archiveSummary": summarize_archive_inventory(()),
    }


def _empty_readiness() -> dict[str, Any]:
    return {
        "allDeclaredAssetsComplete": False,
        "hasPartialDownloads": False,
        "declaredExecutableCount": 0,
        "presentExecutableCount": 0,
        "completeExecutableCount": 0,
        "missingExecutableCount": 0,
        "incompleteExecutableCount": 0,
        "executableCompleteRatio": 0.0,
        "declaredVisualCount": 0,
        "presentVisualCount": 0,
        "completeVisualCount": 0,
        "missingVisualCount": 0,
        "incompleteVisualCount": 0,
        "visualCompleteRatio": 0.0,
        "dllCompleteRatio": 0.0,
        "archiveCompleteRatio": 0.0,
        "archiveCompleteByteRatio": 0.0,
    }


def _catalog_package_files(game_dir: Path, game_dir_resolved: Path) -> dict[str, Any]:
    """Catalog bundled `.dll` files and sanitized `.rpf` archive inventory/summary."""
    dlls: list[dict[str, Any]] = []
    raw_archives: list[dict[str, Any]] = []
    rpf_count = 0
    rpf_complete_count = 0
    rpf_incomplete_count = 0
    rpf_total_bytes = 0

    stack: list[tuple[Path, int]] = [(game_dir, 0)]
    visited_dirs: set[Path] = {game_dir_resolved}

    while stack:
        current_dir, depth = stack.pop()
        try:
            entries = sorted(current_dir.iterdir(), key=lambda p: (p.name.lower(), p.name))
        except OSError:
            continue

        subdirs: list[tuple[Path, int]] = []
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    if depth < MAX_WALK_DEPTH:
                        resolved_sub = entry.resolve()
                        if resolved_sub.is_relative_to(game_dir_resolved) and resolved_sub not in visited_dirs:
                            visited_dirs.add(resolved_sub)
                            subdirs.append((entry, depth + 1))
                    continue
                if not entry.is_file():
                    continue
            except OSError:
                continue

            lower_name = entry.name.lower()
            if lower_name.endswith(".dll"):
                probe = _probe_file_completeness(entry)
                try:
                    rel_path = _relative_posix(entry.resolve(), game_dir_resolved)
                except (OSError, ValueError):
                    rel_path = entry.name
                dlls.append(
                    {
                        "basename": entry.name,
                        "relativePath": rel_path,
                        "sizeBytes": probe["sizeBytes"],
                        "nonZeroHeader": probe["nonZeroHeader"],
                        "headerFormat": probe["headerFormat"],
                        "partialDownloadSuspected": probe["partialDownloadSuspected"],
                        "complete": probe["complete"],
                        "status": probe["status"],
                    }
                )
            elif lower_name.endswith(".rpf"):
                rpf_count += 1
                probe = _probe_file_completeness(entry)
                size_bytes = int(probe["sizeBytes"])
                rpf_total_bytes += size_bytes
                if probe["complete"]:
                    rpf_complete_count += 1
                else:
                    rpf_incomplete_count += 1
                try:
                    rel_path = _relative_posix(entry.resolve(), game_dir_resolved)
                except (OSError, ValueError):
                    rel_path = entry.name
                raw_archives.append(
                    {
                        "pathHash": _archive_path_hash(rel_path),
                        "extension": ".rpf",
                        "depth": depth,
                        "sizeBytes": size_bytes,
                        "nonZeroHeader": probe["nonZeroHeader"],
                        "headerFormat": probe["headerFormat"],
                        "partialDownloadSuspected": probe["partialDownloadSuspected"],
                        "complete": probe["complete"],
                        "status": probe["status"],
                    }
                )

        # Reverse so stack pop processes directories in lexicographical order.
        for sub in reversed(subdirs):
            stack.append(sub)

    dlls.sort(key=lambda item: (item["relativePath"].lower(), item["relativePath"], item["basename"]))
    archives = sort_archive_inventory(raw_archives)
    for arch in archives:
        arch["sizeRatio"] = safe_ratio(int(arch["sizeBytes"]), rpf_total_bytes)

    archive_summary = summarize_archive_inventory(archives)
    return {
        "dlls": dlls,
        "dllCount": len(dlls),
        "dllCompleteCount": sum(1 for item in dlls if item["complete"]),
        "dllIncompleteCount": sum(1 for item in dlls if not item["complete"]),
        "dllTotalBytes": sum(int(item["sizeBytes"]) for item in dlls),
        "rpfCount": rpf_count,
        "rpfCompleteCount": rpf_complete_count,
        "rpfIncompleteCount": rpf_incomplete_count,
        "rpfTotalBytes": rpf_total_bytes,
        "archives": archives,
        "archiveSummary": archive_summary,
    }


def inspect_game_directory(game_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Inspect a local Xbox Series game directory and return a sanitized readiness report.

    Never emits host absolute paths, UNC paths, usernames, copyrighted archive names,
    or raw file bytes.
    """
    base_path = Path(game_dir).expanduser()
    empty_inventory = _empty_inventory()
    empty_readiness = _empty_readiness()

    if not base_path.is_dir():
        return sanitize_report(
            {
                "schemaVersion": SCHEMA_VERSION,
                "ok": False,
                "status": "missing_game_dir",
                "configFound": False,
                "configValid": False,
                "configRelativePath": None,
                "error": "Game directory does not exist or is not a directory",
                "config": None,
                "executables": [],
                "shellVisuals": [],
                "inventory": empty_inventory,
                "readiness": empty_readiness,
            },
            extra_paths=(base_path,),
        )

    game_dir_resolved = base_path.resolve()
    extra_paths = (base_path, game_dir_resolved)
    inventory = _catalog_package_files(base_path, game_dir_resolved)
    dll_complete_ratio = safe_ratio(inventory["dllCompleteCount"], inventory["dllCount"])
    archive_complete_ratio = safe_ratio(inventory["rpfCompleteCount"], inventory["rpfCount"])
    archive_complete_byte_ratio = inventory["archiveSummary"]["completeByteRatio"]

    config_path = _locate_microsoft_game_config(base_path)
    if config_path is None:
        return sanitize_report(
            {
                "schemaVersion": SCHEMA_VERSION,
                "ok": False,
                "status": "missing_config",
                "configFound": False,
                "configValid": False,
                "configRelativePath": None,
                "error": "MicrosoftGame.Config not found in game directory",
                "config": None,
                "executables": [],
                "shellVisuals": [],
                "inventory": inventory,
                "readiness": {
                    **empty_readiness,
                    "hasPartialDownloads": bool(
                        inventory["dllIncompleteCount"] > 0 or inventory["rpfIncompleteCount"] > 0
                    ),
                    "dllCompleteRatio": dll_complete_ratio,
                    "archiveCompleteRatio": archive_complete_ratio,
                    "archiveCompleteByteRatio": archive_complete_byte_ratio,
                },
            },
            extra_paths=extra_paths,
        )

    try:
        config_rel_path = _relative_posix(config_path.resolve(), game_dir_resolved)
    except (OSError, ValueError):
        config_rel_path = config_path.name

    try:
        parsed_config = parse_microsoft_game_config(config_path)
    except (GameConfigError, OSError) as exc:
        return sanitize_report(
            {
                "schemaVersion": SCHEMA_VERSION,
                "ok": False,
                "status": "invalid_config",
                "configFound": True,
                "configValid": False,
                "configRelativePath": config_rel_path,
                "error": sanitize_text(str(exc), extra_paths=extra_paths) or "Invalid MicrosoftGame.Config",
                "config": None,
                "executables": [],
                "shellVisuals": [],
                "inventory": inventory,
                "readiness": {
                    **empty_readiness,
                    "hasPartialDownloads": bool(
                        inventory["dllIncompleteCount"] > 0 or inventory["rpfIncompleteCount"] > 0
                    ),
                    "dllCompleteRatio": dll_complete_ratio,
                    "archiveCompleteRatio": archive_complete_ratio,
                    "archiveCompleteByteRatio": archive_complete_byte_ratio,
                },
            },
            extra_paths=extra_paths,
        )

    package_root = config_path.parent
    package_root_resolved = package_root.resolve()

    executable_checks: list[dict[str, Any]] = []
    for exe in parsed_config["executables"]:
        asset_check = _check_declared_asset(
            exe.get("name"),
            package_root=package_root,
            package_root_resolved=package_root_resolved,
            game_dir_resolved=game_dir_resolved,
        )
        executable_checks.append(
            {
                "id": exe.get("id"),
                "targetDeviceFamily": exe.get("targetDeviceFamily"),
                "alias": exe.get("alias"),
                **asset_check,
            }
        )

    visual_checks: list[dict[str, Any]] = []
    sv_cfg = parsed_config["shellVisuals"]
    for field_key, xml_attr in SHELL_VISUAL_IMAGE_FIELDS:
        declared = sv_cfg.get(field_key)
        if declared is None:
            continue
        asset_check = _check_declared_asset(
            declared,
            package_root=package_root,
            package_root_resolved=package_root_resolved,
            game_dir_resolved=game_dir_resolved,
        )
        visual_checks.append(
            {
                "field": field_key,
                "attribute": xml_attr,
                **asset_check,
            }
        )

    present_exes = sum(1 for item in executable_checks if item["present"])
    complete_exes = sum(1 for item in executable_checks if item["complete"])
    missing_exes = sum(1 for item in executable_checks if not item["present"])
    incomplete_exes = sum(
        1 for item in executable_checks if item["present"] and not item["complete"]
    )

    present_visuals = sum(1 for item in visual_checks if item["present"])
    complete_visuals = sum(1 for item in visual_checks if item["complete"])
    missing_visuals = sum(1 for item in visual_checks if not item["present"])
    incomplete_visuals = sum(
        1 for item in visual_checks if item["present"] and not item["complete"]
    )

    has_partial_downloads = bool(
        any(item["partialDownloadSuspected"] for item in executable_checks)
        or any(item["partialDownloadSuspected"] for item in visual_checks)
        or inventory["dllIncompleteCount"] > 0
        or inventory["rpfIncompleteCount"] > 0
    )
    all_declared_complete = bool(
        complete_exes == len(executable_checks)
        and complete_visuals == len(visual_checks)
    )

    return sanitize_report(
        {
            "schemaVersion": SCHEMA_VERSION,
            "ok": True,
            "status": "ok",
            "configFound": True,
            "configValid": True,
            "configRelativePath": config_rel_path,
            "error": None,
            "config": parsed_config,
            "executables": executable_checks,
            "shellVisuals": visual_checks,
            "inventory": inventory,
            "readiness": {
                "allDeclaredAssetsComplete": all_declared_complete,
                "hasPartialDownloads": has_partial_downloads,
                "declaredExecutableCount": len(executable_checks),
                "presentExecutableCount": present_exes,
                "completeExecutableCount": complete_exes,
                "missingExecutableCount": missing_exes,
                "incompleteExecutableCount": incomplete_exes,
                "executableCompleteRatio": safe_ratio(complete_exes, len(executable_checks)),
                "declaredVisualCount": len(visual_checks),
                "presentVisualCount": present_visuals,
                "completeVisualCount": complete_visuals,
                "missingVisualCount": missing_visuals,
                "incompleteVisualCount": incomplete_visuals,
                "visualCompleteRatio": safe_ratio(complete_visuals, len(visual_checks)),
                "dllCompleteRatio": dll_complete_ratio,
                "archiveCompleteRatio": archive_complete_ratio,
                "archiveCompleteByteRatio": archive_complete_byte_ratio,
            },
        },
        extra_paths=extra_paths,
    )


def _status_badge(status: str) -> str:
    if status == "ok":
        return "[OK]     "
    if status in ("empty", "incomplete_zero_header"):
        return "[PARTIAL]"
    if status == "missing":
        return "[MISS]   "
    return "[ERR]    "


def format_report_text(report: Mapping[str, Any]) -> str:
    """Render a sanitized inspection report as a human-readable terminal summary."""
    lines: list[str] = ["=== playgta6web Xbox Package & Asset Inspector ===", ""]

    if not report.get("configValid"):
        lines.append(f"Status : ERROR ({report.get('status')})")
        if report.get("configRelativePath"):
            lines.append(f"Config : {report['configRelativePath']}")
        lines.append(f"Detail : {report.get('error') or 'Unknown error'}")
        inv = report.get("inventory") or {}
        if inv.get("dllCount") or inv.get("rpfCount"):
            lines.append(
                f"Scanned: dlls={inv.get('dllCount', 0)}, rpf={inv.get('rpfCount', 0)}"
            )
        return sanitize_text("\n".join(lines)) or ""

    cfg = report["config"]
    ident = cfg["identity"]
    sv = cfg["shellVisuals"]

    lines.append(f"Config          : {report['configRelativePath']} (configVersion={cfg.get('configVersion') or 'unspecified'})")
    lines.append(f"Display Name    : {sv.get('defaultDisplayName') or '(unset)'}")
    lines.append(f"Identity Name   : {ident.get('name') or '(unset)'}")
    lines.append(f"Identity Version: {ident.get('version') or '(unset)'}")
    lines.append(f"Publisher       : {ident.get('publisher') or '(unset)'}")
    lines.append(f"TitleId         : {cfg.get('titleId') or '(unset)'}")
    if cfg.get("storeId"):
        lines.append(f"StoreId         : {cfg['storeId']}")
    if cfg.get("msaAppId"):
        lines.append(f"MSAAppId        : {cfg['msaAppId']}")
    langs = cfg.get("resources", {}).get("languages") or []
    lines.append(f"Languages       : {', '.join(langs) if langs else '(none declared)'}")
    if cfg.get("virtualMachine"):
        lines.append("VirtualMachine  : present")
    if cfg.get("modFolder"):
        mf = cfg["modFolder"]
        lines.append(f"ModFolder       : {mf.get('name') or '(unnamed)'} (readOnly={mf.get('readOnly')})")

    lines.append("")
    lines.append("Declared Executables:")
    exes = report.get("executables") or []
    if not exes:
        lines.append("  (none declared)")
    else:
        for exe in exes:
            badge = _status_badge(exe["status"])
            target = f" [{exe['targetDeviceFamily']}]" if exe.get("targetDeviceFamily") else ""
            alias = f" alias={exe['alias']}" if exe.get("alias") else ""
            detail = (
                f"{exe['sizeBytes']} bytes ({exe['headerFormat']})"
                if exe["present"]
                else exe["status"]
            )
            lines.append(f"  {badge} {exe['declaredPath'] or '(unnamed)'}{target}{alias} — {detail}")

    lines.append("")
    lines.append("Declared ShellVisuals Images:")
    visuals = report.get("shellVisuals") or []
    if not visuals:
        lines.append("  (none declared)")
    else:
        for vis in visuals:
            badge = _status_badge(vis["status"])
            detail = (
                f"{vis['sizeBytes']} bytes ({vis['headerFormat']})"
                if vis["present"]
                else vis["status"]
            )
            lines.append(f"  {badge} {vis['attribute']}: {vis['declaredPath']} — {detail}")

    inv = report["inventory"]
    lines.append("")
    lines.append("Bundled Package Inventory:")
    lines.append(
        f"  DLLs : {inv['dllCount']} total "
        f"({inv['dllCompleteCount']} complete, {inv['dllIncompleteCount']} incomplete)"
    )
    for dll in inv["dlls"]:
        badge = _status_badge(dll["status"])
        lines.append(
            f"    {badge} {dll['relativePath']} — {dll['sizeBytes']} bytes ({dll['headerFormat']})"
        )
    lines.append(
        f"  RPFs : {inv['rpfCount']} total "
        f"({inv['rpfCompleteCount']} complete, {inv['rpfIncompleteCount']} incomplete, "
        f"{inv['rpfTotalBytes']} bytes)"
    )
    arch_summary = inv.get("archiveSummary")
    if isinstance(arch_summary, Mapping):
        lines.append(
            f"         complete_ratio={arch_summary.get('completeCountRatio', 0.0):.2%}, "
            f"complete_bytes_ratio={arch_summary.get('completeByteRatio', 0.0):.2%}, "
            f"avg={arch_summary.get('avgBytes', 0.0):.2f} bytes"
        )

    rd = report["readiness"]
    lines.append("")
    lines.append("Package Readiness:")
    lines.append(
        f"  Declared Executables : {rd['completeExecutableCount']}/{rd['declaredExecutableCount']} complete "
        f"(missing={rd['missingExecutableCount']}, partial={rd['incompleteExecutableCount']})"
    )
    lines.append(
        f"  Declared Visuals     : {rd['completeVisualCount']}/{rd['declaredVisualCount']} complete "
        f"(missing={rd['missingVisualCount']}, partial={rd['incompleteVisualCount']})"
    )
    if rd["hasPartialDownloads"]:
        lines.append("  WARNING: Zero-filled or empty files detected (partial download suspected).")

    return sanitize_text("\n".join(lines)) or ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect an Xbox Series MicrosoftGame.Config and verify declared executables, "
            "ShellVisuals assets, bundled DLLs, and RPF counts."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--from-env",
        action="store_true",
        help="Resolve GTA6_GAME_DIR from environment variables or gitignored .env.",
    )
    source.add_argument(
        "--dir",
        dest="game_dir",
        type=Path,
        metavar="PATH",
        help="Path to the local Xbox Series game directory to inspect.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root used to locate .env when --from-env is specified.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit strictly sanitized JSON output (no host paths, usernames, or raw bytes).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.from_env:
        repo_root_resolved = args.repo_root.resolve()
        target_dir = resolve_game_dir_from_env(repo_root=repo_root_resolved)
        if target_dir is None:
            report: dict[str, Any] = sanitize_report(
                {
                    "schemaVersion": SCHEMA_VERSION,
                    "ok": False,
                    "status": "env_not_configured",
                    "configFound": False,
                    "configValid": False,
                    "configRelativePath": None,
                    "error": 'GTA6_GAME_DIR is not configured (run: python tools/local_env.py --set-game "<path>")',
                    "config": None,
                    "executables": [],
                    "shellVisuals": [],
                    "inventory": _empty_inventory(),
                    "readiness": _empty_readiness(),
                },
                extra_paths=(args.repo_root, repo_root_resolved),
            )
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(format_report_text(report))
            return 1
        extra_paths: tuple[Path, ...] = (args.repo_root, repo_root_resolved, target_dir)
    else:
        target_dir = args.game_dir
        extra_paths = (target_dir,)

    report = sanitize_report(inspect_game_directory(target_dir), extra_paths=extra_paths)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report_text(report))

    return 0 if report["configValid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
