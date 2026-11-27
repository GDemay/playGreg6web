#!/usr/bin/env python3
"""Automated quantitative visual, color, gamma, aspect, and progression frame audit.

Inspects a captured browser or native boot run directory (e.g. `work/browser-boot/runs/<run>/`
or a direct `frames/` directory) and outputs ONLY strictly-sanitized numerical metrics —
never raw pixels, base64 images, or host filesystem paths/usernames.

Capabilities:
  - Pure-Python stdlib PNG decoder (8-bit color types 0/2/3/4/6, filters 0..4) with optional
    Pillow (`PIL.Image`) acceleration for JPEG/WebP/BMP/PNG and optional `numpy` fast path.
  - Per-frame & visible-frame summary Rec.709 luminance percentiles (`p05`, `p50`, `p95`,
    `mean_luma`, `max_luma`), pure-black fraction (`rgb == (0,0,0)`), near-black crush fraction
    (catching double-gamma `~x^4.84` shadow crush), and clipped-white fraction.
  - Channel balance (`mean_r`, `mean_g`, `mean_b`, `rg_ratio`, `bg_ratio`, `rb_ratio`,
    `green_cast_index`), HSV saturation, and Hasler-Susstrunk colorfulness index.
  - Aspect-ratio / letterbox / pillarbox sanity check (16:9 viewport framing vs squashed
    stretch, windowboxing, or off-center framing, plus run-level union active box).
  - Frame-to-frame visual transition timestamps (`MAD` on downsampled 32x18 RGB thumbnails)
    and anchor-based dwell segments across resolution changes.
  - Before/after `--compare <baseline> <candidate>` diff mode (text, `--json`, `--markdown`)
    with gamma-crush, R/B channel-swap, green-cast, desaturation, progression, and framing
    improvement/regression detection.

Usage:
  python tools/visual_frame_audit.py <run_or_frames_dir> [--json | --markdown]
  python tools/visual_frame_audit.py --compare <baseline_run> <candidate_run> [--json | --markdown]
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import getpass
import io
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import sys
from typing import Any, Mapping, Sequence
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_timeline  # noqa: E402

AUDIT_SCHEMA = "playgta6-visual-frame-audit"
COMPARE_SCHEMA = "playgta6-visual-frame-audit-compare"
SCHEMA_VERSION = 1

THUMB_SIZE = (32, 18)
CANVAS_MIN_X = 410
DEFAULT_FRAME_INTERVAL_S = 1.0
DEFAULT_BLACK_LUMA = 2.0
DEFAULT_NEAR_BLACK_LUMA = 16.0
DEFAULT_WHITE_LUMA = 254.0
DEFAULT_DIFF_THRESHOLD = 4.0
DEFAULT_TRANSITION_THRESHOLD = 12.0
DEFAULT_TARGET_ASPECT = 16.0 / 9.0
DEFAULT_ASPECT_TOLERANCE = 0.06
BAR_ROW_MEAN_MAX = 4.0
BAR_ROW_PIXEL_MAX = 16.0

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".webp"})
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_RE_FRAME_NUM = re.compile(
    r"^frame[_-]?(\d+)(?:[_-]t(\d+(?:\.\d+)?)(ms|s)?)?$",
    re.IGNORECASE,
)
_RE_SEG_THUMB = re.compile(
    r"^seg(\d+)_(\d+(?:\.\d+)?)s_(\d+)d$",
    re.IGNORECASE,
)
_RE_WALLCLOCK = re.compile(
    r"^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})-(\d{3})(?:-(\d+))?$"
)
_RE_SWAP = re.compile(r"^(?:.*[_-])?swap(\d+)$", re.IGNORECASE)
_RE_EMBEDDED_T = re.compile(r"(?:^|[_-])t(\d+(?:\.\d+)?)(ms|s)(?:$|[_-])", re.IGNORECASE)
_RE_FIRST_INT = re.compile(r"(\d+)")
_RE_LAST_INT = re.compile(r"(\d+)(?!.*\d)")
_LABEL_SAFE_RE = re.compile(r"[^A-Za-z0-9._:/+#\- ]")
_LABEL_MAX_LEN = 64


# ---------------------------------------------------------------------------
# Sanitization helpers (zero host paths, usernames, or raw pixel bytes)
# ---------------------------------------------------------------------------


def _current_usernames() -> list[str]:
    names: set[str] = set()
    for candidate in (
        os.environ.get("USERNAME"),
        os.environ.get("USER"),
    ):
        if candidate and len(candidate) >= 3:
            names.add(candidate)
    try:
        u = getpass.getuser()
        if u and len(u) >= 3:
            names.add(u)
    except Exception:
        pass
    return sorted(names, key=len, reverse=True)


def sanitize_text(text: str) -> str:
    """Scrub host paths and active OS usernames from arbitrary output text."""
    out = screen_timeline.sanitize_text(text)
    for name in _current_usernames():
        if name in out:
            out = re.sub(
                rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                "<user>",
                out,
                flags=re.IGNORECASE,
            )
    return out


def sanitize_label(raw: str | Path | None, fallback: str = "run") -> str:
    """Produce a safe run or frame label with no paths, usernames, pipes, or control chars."""
    if raw is None:
        return fallback
    text = str(raw).strip()
    if not text:
        return fallback
    if isinstance(raw, Path) or "/" in text or "\\" in text or re.match(r"(?i)^[A-Za-z]:", text):
        text = screen_timeline.sanitize_path_label(text)
    text = sanitize_text(text)
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        text = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in text)
    text = _LABEL_SAFE_RE.sub("", text)
    text = " ".join(text.split())
    if len(text) > _LABEL_MAX_LEN:
        text = text[:_LABEL_MAX_LEN].rstrip()
    if not text or text in {"<redacted>", "<user>"} or not any(ch.isalnum() for ch in text):
        return fallback
    return text


def sanitize_payload(value: Any) -> Any:
    """Recursively sanitize a JSON-serializable payload."""
    cleaned = screen_timeline.sanitize_json_payload(value)
    if isinstance(cleaned, str):
        return sanitize_text(cleaned)
    if isinstance(cleaned, dict):
        return {
            (sanitize_text(k) if isinstance(k, str) else k): sanitize_payload(v)
            for k, v in cleaned.items()
        }
    if isinstance(cleaned, list):
        return [sanitize_payload(item) for item in cleaned]
    return cleaned


# ---------------------------------------------------------------------------
# Optional dependency probes
# ---------------------------------------------------------------------------


def _import_pil_image() -> Any | None:
    try:
        from PIL import Image  # type: ignore[import-not-found]
        return Image
    except Exception:
        return None


def _import_numpy() -> Any | None:
    try:
        import numpy as np  # type: ignore[import-not-found]
        return np
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image decoding (Pillow optional + pure-Python stdlib PNG fallback)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodedImage:
    width: int
    height: int
    rgb: bytes
    decoder: str


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def decode_png_stdlib(data: bytes) -> DecodedImage | None:
    """Decode an 8-bit non-interlaced PNG (color types 0, 2, 3, 4, 6; filters 0..4) using stdlib only."""
    if len(data) < 33 or data[:8] != _PNG_SIGNATURE:
        return None

    pos = 8
    width = height = bit_depth = color_type = compression = filter_method = interlace = -1
    palette: bytes = b""
    idat_chunks: list[bytes] = []
    saw_ihdr = False
    saw_iend = False

    while pos + 8 <= len(data):
        chunk_len, chunk_type = struct.unpack(">I4s", data[pos : pos + 8])
        pos += 8
        if pos + chunk_len + 4 > len(data):
            return None
        chunk_data = data[pos : pos + chunk_len]
        pos += chunk_len + 4  # skip CRC

        if chunk_type == b"IHDR":
            if chunk_len != 13 or saw_ihdr:
                return None
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            saw_ihdr = True
        elif chunk_type == b"PLTE":
            palette = chunk_data
        elif chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            saw_iend = True
            break

    if not saw_ihdr or not saw_iend or width <= 0 or height <= 0:
        return None
    if bit_depth != 8 or compression != 0 or filter_method != 0 or interlace != 0:
        return None

    bpp_map = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    bpp = bpp_map.get(color_type)
    if bpp is None:
        return None
    if color_type == 3 and len(palette) < 3:
        return None

    try:
        raw = zlib.decompress(b"".join(idat_chunks))
    except zlib.error:
        return None

    stride = width * bpp
    expected_len = height * (1 + stride)
    if len(raw) < expected_len:
        return None

    prev_row = bytearray(stride)
    rgb = bytearray(width * height * 3)
    palette_entries = len(palette) // 3 if color_type == 3 else 0

    src_offset = 0
    dst_offset = 0
    for _y in range(height):
        flt = raw[src_offset]
        src_offset += 1
        row = bytearray(raw[src_offset : src_offset + stride])
        src_offset += stride

        if flt == 0:
            pass
        elif flt == 1:
            for i in range(bpp, stride):
                row[i] = (row[i] + row[i - bpp]) & 0xFF
        elif flt == 2:
            for i in range(stride):
                row[i] = (row[i] + prev_row[i]) & 0xFF
        elif flt == 3:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + ((left + prev_row[i]) >> 1)) & 0xFF
        elif flt == 4:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev_row[i]
                up_left = prev_row[i - bpp] if i >= bpp else 0
                row[i] = (row[i] + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            return None

        if color_type == 2:
            rgb[dst_offset : dst_offset + width * 3] = row
            dst_offset += width * 3
        elif color_type == 6:
            for x in range(width):
                si = x * 4
                rgb[dst_offset] = row[si]
                rgb[dst_offset + 1] = row[si + 1]
                rgb[dst_offset + 2] = row[si + 2]
                dst_offset += 3
        elif color_type == 0:
            for x in range(width):
                v = row[x]
                rgb[dst_offset] = v
                rgb[dst_offset + 1] = v
                rgb[dst_offset + 2] = v
                dst_offset += 3
        elif color_type == 4:
            for x in range(width):
                v = row[x * 2]
                rgb[dst_offset] = v
                rgb[dst_offset + 1] = v
                rgb[dst_offset + 2] = v
                dst_offset += 3
        elif color_type == 3:
            max_idx = palette_entries - 1
            for x in range(width):
                idx = row[x]
                if idx > max_idx:
                    idx = max_idx
                pi = idx * 3
                rgb[dst_offset] = palette[pi]
                rgb[dst_offset + 1] = palette[pi + 1]
                rgb[dst_offset + 2] = palette[pi + 2]
                dst_offset += 3

        prev_row = row

    return DecodedImage(width=width, height=height, rgb=bytes(rgb), decoder="stdlib_png")


def _looks_like_pillow_only_format(data: bytes, suffix: str) -> bool:
    if data[:2] == b"\xff\xd8":
        return True
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    if data[:2] == b"BM":
        return True
    return suffix.lower() in {".jpg", ".jpeg", ".webp", ".bmp"}


def decode_image_rgb(
    path: Path,
    *,
    pil_image_mod: Any | None = None,
    _auto_import_pil: bool = True,
) -> tuple[DecodedImage | None, str | None]:
    """Decode an image file into RGB bytes. Never raises on corrupt/unsupported inputs."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None, "read_error"
    if not raw:
        return None, "unsupported_or_corrupt"

    pil_mod = pil_image_mod if pil_image_mod is not None else (_import_pil_image() if _auto_import_pil else None)
    if pil_mod is not None:
        try:
            with pil_mod.open(io.BytesIO(raw)) as img:
                rgb_img = img.convert("RGB")
                w, h = rgb_img.size
                if w <= 0 or h <= 0:
                    return None, "unsupported_or_corrupt"
                return DecodedImage(width=w, height=h, rgb=rgb_img.tobytes(), decoder="pillow"), None
        except Exception:
            if raw[:8] != _PNG_SIGNATURE:
                return None, "unsupported_or_corrupt"

    if raw[:8] == _PNG_SIGNATURE:
        decoded = decode_png_stdlib(raw)
        if decoded is not None:
            return decoded, None
        return None, "unsupported_or_corrupt"

    if pil_mod is None and _looks_like_pillow_only_format(raw, path.suffix):
        return None, "requires_pillow"
    return None, "unsupported_or_corrupt"


def crop_rgb_x(img: DecodedImage, crop_x: int) -> DecodedImage:
    """Crop RGB buffer horizontally to `[crop_x, width)` if `0 < crop_x < width`."""
    if crop_x <= 0 or crop_x >= img.width:
        return img
    new_w = img.width - crop_x
    src_stride = img.width * 3
    x_byte = crop_x * 3
    dst_stride = new_w * 3
    out = bytearray(new_w * img.height * 3)
    src = memoryview(img.rgb)
    for y in range(img.height):
        s0 = y * src_stride + x_byte
        d0 = y * dst_stride
        out[d0 : d0 + dst_stride] = src[s0 : s0 + dst_stride]
    return DecodedImage(width=new_w, height=img.height, rgb=bytes(out), decoder=img.decoder)


def downsample_thumbnail(
    img: DecodedImage,
    size: tuple[int, int] = THUMB_SIZE,
    *,
    pil_image_mod: Any | None = None,
    _auto_import_pil: bool = True,
) -> bytes:
    """Downsample an RGB image to a fixed `size=(tw, th)` RGB thumbnail."""
    tw, th = size
    pil_mod = pil_image_mod if pil_image_mod is not None else (_import_pil_image() if _auto_import_pil else None)
    if pil_mod is not None:
        try:
            pil_img = pil_mod.frombytes("RGB", (img.width, img.height), img.rgb)
            resample = getattr(getattr(pil_mod, "Resampling", pil_mod), "BILINEAR", 2)
            return pil_img.resize((tw, th), resample).tobytes()
        except Exception:
            pass

    w, h = img.width, img.height
    rgb = img.rgb
    out = bytearray(tw * th * 3)
    row_stride = w * 3
    for ty in range(th):
        y0 = (ty * h) // th
        y1 = max(y0 + 1, ((ty + 1) * h) // th)
        y_step = max(1, (y1 - y0) // 4)
        for tx in range(tw):
            x0 = (tx * w) // tw
            x1 = max(x0 + 1, ((tx + 1) * w) // tw)
            x_step = max(1, (x1 - x0) // 4)
            sr = sg = sb = cnt = 0
            for y in range(y0, y1, y_step):
                base = y * row_stride
                for x in range(x0, x1, x_step):
                    idx = base + x * 3
                    sr += rgb[idx]
                    sg += rgb[idx + 1]
                    sb += rgb[idx + 2]
                    cnt += 1
            di = (ty * tw + tx) * 3
            if cnt > 0:
                out[di] = sr // cnt
                out[di + 1] = sg // cnt
                out[di + 2] = sb // cnt
    return bytes(out)


def thumbnail_mad(a: bytes, b: bytes) -> float:
    """Mean absolute channel difference (`0..255`) between two equal-sized RGB thumbnails."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    total = sum(abs(x - y) for x, y in zip(a[:n], b[:n]))
    return round(total / n, 3)


# ---------------------------------------------------------------------------
# Frame discovery, directory resolution & timestamp inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiscoveredFrame:
    path: Path
    name: str
    frame_index: int
    embedded_timestamp_s: float | None
    sort_key: tuple[Any, ...]


def resolve_frames_dir(target: Path) -> tuple[Path, Path | None]:
    """Resolve `<run_dir>` or direct `<frames_dir>` -> `(frames_dir, run_dir)`."""
    if not target.exists() or not target.is_dir():
        raise ValueError(f"directory does not exist: {sanitize_label(target, fallback='<dir>')}")

    frames_sub = target / "frames"
    if frames_sub.is_dir():
        return frames_sub, target

    thumbs_sub = target / "thumbs"
    if thumbs_sub.is_dir() and any(
        p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS for p in thumbs_sub.iterdir()
    ):
        return thumbs_sub, target

    parent = target.parent
    run_dir: Path | None = None
    if (target / "command.json").is_file() or (target / "telemetry.json").is_file():
        run_dir = target
    elif (parent / "command.json").is_file() or (parent / "telemetry.json").is_file():
        run_dir = parent
    return target, run_dir


def parse_frame_filename(path: Path) -> tuple[tuple[Any, ...], int | None, float | None, float | None]:
    """Parse structured frame stem -> `(sort_key, frame_index, embedded_ts_s, wallclock_s)`."""
    stem = path.stem
    m = _RE_FRAME_NUM.match(stem)
    if m:
        idx = int(m.group(1))
        ts: float | None = None
        if m.group(2) is not None:
            val = float(m.group(2))
            unit = (m.group(3) or "s").lower()
            ts = round(val / 1000.0 if unit == "ms" else val, 4)
        return (0, idx, stem.lower()), idx, ts, None

    m = _RE_SEG_THUMB.match(stem)
    if m:
        idx = int(m.group(1))
        ts = round(float(m.group(2)), 4)
        return (0, idx, ts, stem.lower()), idx, ts, None

    m = _RE_WALLCLOCK.match(stem)
    if m:
        yr, mo, dy, hr, mi, sc, ms = (int(m.group(i)) for i in range(1, 8))
        coll = int(m.group(8)) if m.group(8) else 0
        # Monotonic pseudo-epoch seconds within day/calendar for relative delta sorting
        wall_s = ((((yr * 12 + mo) * 31 + dy) * 24 + hr) * 60 + mi) * 60 + sc + (ms / 1000.0)
        return (1, yr, mo, dy, hr, mi, sc, ms, coll, stem.lower()), None, None, wall_s

    m = _RE_SWAP.match(stem)
    if m:
        idx = int(m.group(1))
        return (0, idx, stem.lower()), idx, None, None

    ts_m = _RE_EMBEDDED_T.search(stem)
    if ts_m:
        val = float(ts_m.group(1))
        ts = round(val / 1000.0 if ts_m.group(2).lower() == "ms" else val, 4)
        first_int = _RE_FIRST_INT.search(stem[: ts_m.start()])
        if first_int:
            idx = int(first_int.group(1))
            return (0, idx, ts, stem.lower()), idx, ts, None
        return (2, ts, stem.lower()), None, ts, None

    last_int = _RE_LAST_INT.search(stem)
    if last_int:
        idx = int(last_int.group(1))
        return (2, idx, stem.lower()), idx, None, None

    return (3, stem.lower()), None, None, None


def discover_frames(frames_dir: Path) -> list[DiscoveredFrame]:
    """Discover and deterministically sort image files in `frames_dir`."""
    raw_items: list[tuple[tuple[Any, ...], Path, int | None, float | None, float | None]] = []
    for p in frames_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue
        sk, idx, ts, wall_s = parse_frame_filename(p)
        raw_items.append((sk, p, idx, ts, wall_s))
    raw_items.sort(key=lambda item: item[0])

    wall_vals = [item[4] for item in raw_items if item[4] is not None]
    t0 = min(wall_vals) if wall_vals else None

    out: list[DiscoveredFrame] = []
    for ord_idx, (sk, p, idx, ts, wall_s) in enumerate(raw_items, start=1):
        resolved_idx = idx if idx is not None else ord_idx
        resolved_ts = ts
        if resolved_ts is None and wall_s is not None and t0 is not None:
            resolved_ts = round(wall_s - t0, 3)
        out.append(
            DiscoveredFrame(
                path=p,
                name=sanitize_label(p.name, fallback=f"frame_{ord_idx:04d}"),
                frame_index=resolved_idx,
                embedded_timestamp_s=resolved_ts,
                sort_key=sk,
            )
        )
    return out


def _read_json_optional(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _infer_sidecar_metadata(
    run_dir: Path | None,
    discovered: Sequence[DiscoveredFrame],
) -> tuple[float | None, str, bool | None]:
    """Return `(inferred_interval_s, source, hide_hud)` from optional run sidecars."""
    if run_dir is None:
        return None, "none", None

    cmd = _read_json_optional(run_dir / "command.json")
    tele = _read_json_optional(run_dir / "telemetry.json")
    hide_hud: bool | None = None
    for src in (cmd, tele):
        if isinstance(src.get("hide_hud"), bool):
            hide_hud = src["hide_hud"]
            break

    exts = {f.path.suffix.lower() for f in discovered}
    is_png_capture = bool(cmd.get("capture_frames_dir")) or (".png" in exts and ".jpg" not in exts and ".jpeg" not in exts)
    is_jpg_capture = bool(exts & {".jpg", ".jpeg"})

    cap_iv = cmd.get("capture_interval_sec")
    if is_png_capture and isinstance(cap_iv, (int, float)) and not isinstance(cap_iv, bool) and cap_iv > 0:
        return float(cap_iv), "sidecar", hide_hud

    mp4_fps = cmd.get("mp4_fps")
    if is_jpg_capture and isinstance(mp4_fps, (int, float)) and not isinstance(mp4_fps, bool) and mp4_fps > 0:
        return round(1.0 / float(mp4_fps), 4), "sidecar", hide_hud

    if is_jpg_capture and (tele or cmd):
        return DEFAULT_FRAME_INTERVAL_S, "inferred_default_1s", hide_hud

    if isinstance(cap_iv, (int, float)) and not isinstance(cap_iv, bool) and cap_iv > 0:
        return float(cap_iv), "sidecar", hide_hud

    return None, "none", hide_hud


# ---------------------------------------------------------------------------
# Per-frame visual / color / gamma / aspect analysis
# ---------------------------------------------------------------------------


def _percentile_from_hist(hist: Sequence[int], total: int, q: float) -> float:
    if total <= 0:
        return 0.0
    target = max(1, math.ceil(q * total))
    acc = 0
    for b, count in enumerate(hist):
        acc += count
        if acc >= target:
            return round(b / 4.0, 2)
    return 255.0


def _classify_framing(
    active_w: int,
    active_h: int,
    letterbox_frac: float,
    pillarbox_frac: float,
    bar_asym_v: float,
    bar_asym_h: float,
    target_aspect: float,
    aspect_tolerance: float,
) -> tuple[float | None, str]:
    if active_w <= 0 or active_h <= 0:
        return None, "all_black"
    active_aspect = round(active_w / active_h, 4)
    if bar_asym_v > 0.05 or bar_asym_h > 0.05:
        return active_aspect, "off_center"
    if letterbox_frac >= 0.02 and pillarbox_frac >= 0.02:
        return active_aspect, "windowboxed"
    if abs(active_aspect - target_aspect) / max(target_aspect, 1e-6) > aspect_tolerance:
        return active_aspect, "aspect_mismatch"
    if letterbox_frac >= 0.02:
        return active_aspect, "letterboxed_16_9"
    if pillarbox_frac >= 0.02:
        return active_aspect, "pillarboxed_16_9"
    return active_aspect, "full_16_9"


def _detect_bars_pure(
    rgb: bytes,
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    """Scan inward from all 4 edges to find black letterbox/pillarbox bars."""
    row_stride = w * 3
    row_mean_limit = BAR_ROW_MEAN_MAX * 10000.0 * w
    px_max_limit = BAR_ROW_PIXEL_MAX * 10000.0

    def is_bar_row(y: int) -> bool:
        base = y * row_stride
        s = 0
        for x in range(w):
            idx = base + x * 3
            y_scaled = 2126 * rgb[idx] + 7152 * rgb[idx + 1] + 722 * rgb[idx + 2]
            if y_scaled > px_max_limit:
                return False
            s += y_scaled
        return s <= row_mean_limit

    top_bar = 0
    while top_bar < h and is_bar_row(top_bar):
        top_bar += 1

    if top_bar >= h:
        return h, 0, w, 0

    bottom_bar = 0
    while bottom_bar < (h - top_bar) and is_bar_row(h - 1 - bottom_bar):
        bottom_bar += 1

    y_start = top_bar
    y_end = h - bottom_bar
    col_h = max(1, y_end - y_start)
    col_mean_limit = BAR_ROW_MEAN_MAX * 10000.0 * col_h

    def is_bar_col(x: int) -> bool:
        s = 0
        x3 = x * 3
        for y in range(y_start, y_end):
            idx = y * row_stride + x3
            y_scaled = 2126 * rgb[idx] + 7152 * rgb[idx + 1] + 722 * rgb[idx + 2]
            if y_scaled > px_max_limit:
                return False
            s += y_scaled
        return s <= col_mean_limit

    left_bar = 0
    while left_bar < w and is_bar_col(left_bar):
        left_bar += 1

    right_bar = 0
    while right_bar < (w - left_bar) and is_bar_col(w - 1 - right_bar):
        right_bar += 1

    return top_bar, bottom_bar, left_bar, right_bar


def analyze_frame_rgb(
    img: DecodedImage,
    *,
    sample_stride: int = 1,
    black_luma: float = DEFAULT_BLACK_LUMA,
    near_black_luma: float = DEFAULT_NEAR_BLACK_LUMA,
    white_luma: float = DEFAULT_WHITE_LUMA,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    _use_numpy: bool = True,
) -> dict[str, Any]:
    """Compute exact Rec.709 luminance, gamma-crush, channel-balance, colorfulness, and framing metrics."""
    w, h = img.width, img.height
    stride = max(1, int(sample_stride))
    np = _import_numpy() if _use_numpy else None

    if np is not None:
        arr = np.frombuffer(img.rgb, dtype=np.uint8).reshape((h, w, 3))
        r_full = arr[:, :, 0].astype(np.int32)
        g_full = arr[:, :, 1].astype(np.int32)
        b_full = arr[:, :, 2].astype(np.int32)
        y_scaled_full = 2126 * r_full + 7152 * g_full + 722 * b_full

        row_is_bar = (y_scaled_full.max(axis=1) <= BAR_ROW_PIXEL_MAX * 10000.0) & (
            y_scaled_full.mean(axis=1) <= BAR_ROW_MEAN_MAX * 10000.0
        )
        non_bar_rows = np.flatnonzero(~row_is_bar)
        if non_bar_rows.size == 0:
            top_bar, bottom_bar, left_bar, right_bar = h, 0, w, 0
        else:
            top_bar = int(non_bar_rows[0])
            bottom_bar = int(h - 1 - non_bar_rows[-1])
            mid_y = y_scaled_full[top_bar : h - bottom_bar, :]
            col_is_bar = (mid_y.max(axis=0) <= BAR_ROW_PIXEL_MAX * 10000.0) & (
                mid_y.mean(axis=0) <= BAR_ROW_MEAN_MAX * 10000.0
            )
            non_bar_cols = np.flatnonzero(~col_is_bar)
            if non_bar_cols.size == 0:
                left_bar, right_bar = w, 0
            else:
                left_bar = int(non_bar_cols[0])
                right_bar = int(w - 1 - non_bar_cols[-1])

        sub = arr[::stride, ::stride, :]
        r = sub[:, :, 0].astype(np.float64)
        g = sub[:, :, 1].astype(np.float64)
        b = sub[:, :, 2].astype(np.float64)
        n_px = int(r.size)
        y_scaled = y_scaled_full[::stride, ::stride]
        luma = y_scaled.astype(np.float64) / 10000.0

        bins = (y_scaled + 1250) // 2500
        hist = np.bincount(bins.ravel(), minlength=1021).tolist()

        mean_luma = float(luma.mean())
        max_luma = float(luma.max()) if n_px else 0.0
        luma_p05 = _percentile_from_hist(hist, n_px, 0.05)
        luma_p50 = _percentile_from_hist(hist, n_px, 0.50)
        luma_p95 = _percentile_from_hist(hist, n_px, 0.95)

        pure_black_mask = (sub[:, :, 0] == 0) & (sub[:, :, 1] == 0) & (sub[:, :, 2] == 0)
        pure_black_frac = float(pure_black_mask.mean())
        near_black_frac = float((luma < near_black_luma).mean())
        clipped_white_mask = (
            ((sub[:, :, 0] == 255) & (sub[:, :, 1] == 255) & (sub[:, :, 2] == 255))
            | (luma >= white_luma)
        )
        clipped_white_frac = float(clipped_white_mask.mean())

        mean_r = float(r.mean())
        mean_g = float(g.mean())
        mean_b = float(b.mean())

        cmax = np.maximum(np.maximum(r, g), b)
        cmin = np.minimum(np.minimum(r, g), b)
        non_zero = cmax > 0.0
        sat = np.zeros_like(cmax)
        sat[non_zero] = (cmax[non_zero] - cmin[non_zero]) / cmax[non_zero]
        mean_hsv_sat = float(sat.mean())
        mean_hsv_sat_non_black = float(sat[non_zero].mean()) if bool(non_zero.any()) else 0.0

        rg = r - g
        yb = 0.5 * (r + g) - b
        mu_rg = float(rg.mean())
        mu_yb = float(yb.mean())
        var_rg = max(0.0, float((rg * rg).mean()) - mu_rg * mu_rg)
        var_yb = max(0.0, float((yb * yb).mean()) - mu_yb * mu_yb)
        colorfulness_hs = math.sqrt(var_rg + var_yb) + 0.3 * math.sqrt(mu_rg * mu_rg + mu_yb * mu_yb)

        active_w = max(0, w - left_bar - right_bar)
        active_h = max(0, h - top_bar - bottom_bar)
        sym_v = min(top_bar, bottom_bar)
        sym_h = min(left_bar, right_bar)
        sym_w = max(0, w - 2 * sym_h)
        sym_h_px = max(0, h - 2 * sym_v)
        if active_w > 0 and active_h > 0 and sym_w > 0 and sym_h_px > 0:
            act_sub = arr[sym_v : h - sym_v : stride, sym_h : w - sym_h : stride, :]
            act_y_scaled = y_scaled_full[sym_v : h - sym_v : stride, sym_h : w - sym_h : stride]
            act_n = int(act_y_scaled.size)
            act_luma = act_y_scaled.astype(np.float64) / 10000.0
            act_bins = (act_y_scaled + 1250) // 2500
            act_hist = np.bincount(act_bins.ravel(), minlength=1021).tolist()
            active_pure_black_frac = float(
                ((act_sub[:, :, 0] == 0) & (act_sub[:, :, 1] == 0) & (act_sub[:, :, 2] == 0)).mean()
            )
            active_near_black_frac = float((act_luma < near_black_luma).mean())
            active_luma_p50 = _percentile_from_hist(act_hist, act_n, 0.50)
        else:
            active_pure_black_frac = pure_black_frac
            active_near_black_frac = near_black_frac
            active_luma_p50 = luma_p50
    else:
        rgb = img.rgb
        top_bar, bottom_bar, left_bar, right_bar = _detect_bars_pure(rgb, w, h)
        active_w = max(0, w - left_bar - right_bar)
        active_h = max(0, h - top_bar - bottom_bar)
        sym_v = min(top_bar, bottom_bar)
        sym_h = min(left_bar, right_bar)
        y_end = h - sym_v
        x_end = w - sym_h
        has_sym_active = active_w > 0 and active_h > 0 and (y_end > sym_v) and (x_end > sym_h)

        hist = [0] * 1021
        act_hist = [0] * 1021
        n_px = 0
        act_n = 0
        sum_luma = 0.0
        max_luma = 0.0
        pure_black_cnt = 0
        near_black_cnt = 0
        act_pure_black_cnt = 0
        act_near_black_cnt = 0
        clipped_white_cnt = 0
        sum_r = sum_g = sum_b = 0.0
        sum_sat = 0.0
        non_zero_sat_cnt = 0
        sum_rg = sum_rg2 = 0.0
        sum_yb = sum_yb2 = 0.0
        row_stride = w * 3

        for y in range(0, h, stride):
            base = y * row_stride
            in_y = has_sym_active and sym_v <= y < y_end
            for x in range(0, w, stride):
                idx = base + x * 3
                r_i = rgb[idx]
                g_i = rgb[idx + 1]
                b_i = rgb[idx + 2]
                y_scaled = 2126 * r_i + 7152 * g_i + 722 * b_i
                luma_val = y_scaled / 10000.0
                b_idx = (y_scaled + 1250) // 2500
                hist[b_idx] += 1
                n_px += 1
                sum_luma += luma_val
                if luma_val > max_luma:
                    max_luma = luma_val

                is_pure_black = (r_i == 0 and g_i == 0 and b_i == 0)
                is_near_black = luma_val < near_black_luma
                if is_pure_black:
                    pure_black_cnt += 1
                if is_near_black:
                    near_black_cnt += 1
                if (r_i == 255 and g_i == 255 and b_i == 255) or luma_val >= white_luma:
                    clipped_white_cnt += 1

                if in_y and sym_h <= x < x_end:
                    act_hist[b_idx] += 1
                    act_n += 1
                    if is_pure_black:
                        act_pure_black_cnt += 1
                    if is_near_black:
                        act_near_black_cnt += 1

                sum_r += r_i
                sum_g += g_i
                sum_b += b_i

                cmax = r_i if r_i >= g_i else g_i
                if b_i > cmax:
                    cmax = b_i
                cmin = r_i if r_i <= g_i else g_i
                if b_i < cmin:
                    cmin = b_i
                if cmax > 0:
                    sum_sat += (cmax - cmin) / cmax
                    non_zero_sat_cnt += 1

                rg_v = r_i - g_i
                yb_v = 0.5 * (r_i + g_i) - b_i
                sum_rg += rg_v
                sum_rg2 += rg_v * rg_v
                sum_yb += yb_v
                sum_yb2 += yb_v * yb_v

        inv_n = 1.0 / max(1, n_px)
        mean_luma = sum_luma * inv_n
        luma_p05 = _percentile_from_hist(hist, n_px, 0.05)
        luma_p50 = _percentile_from_hist(hist, n_px, 0.50)
        luma_p95 = _percentile_from_hist(hist, n_px, 0.95)
        pure_black_frac = pure_black_cnt * inv_n
        near_black_frac = near_black_cnt * inv_n
        clipped_white_frac = clipped_white_cnt * inv_n
        mean_r = sum_r * inv_n
        mean_g = sum_g * inv_n
        mean_b = sum_b * inv_n
        mean_hsv_sat = sum_sat * inv_n
        mean_hsv_sat_non_black = (sum_sat / non_zero_sat_cnt) if non_zero_sat_cnt > 0 else 0.0

        mu_rg = sum_rg * inv_n
        mu_yb = sum_yb * inv_n
        var_rg = max(0.0, sum_rg2 * inv_n - mu_rg * mu_rg)
        var_yb = max(0.0, sum_yb2 * inv_n - mu_yb * mu_yb)
        colorfulness_hs = math.sqrt(var_rg + var_yb) + 0.3 * math.sqrt(mu_rg * mu_rg + mu_yb * mu_yb)

        if act_n > 0:
            active_pure_black_frac = act_pure_black_cnt / act_n
            active_near_black_frac = act_near_black_cnt / act_n
            active_luma_p50 = _percentile_from_hist(act_hist, act_n, 0.50)
        else:
            active_pure_black_frac = pure_black_frac
            active_near_black_frac = near_black_frac
            active_luma_p50 = luma_p50

    is_black = mean_luma <= black_luma
    shadow_crush_frac = max(0.0, near_black_frac - pure_black_frac)
    gamma_crush_suspected = (
        (not is_black)
        and (max_luma >= 32.0)
        and (
            active_pure_black_frac >= 0.40
            or (active_near_black_frac >= 0.55 and active_luma_p50 <= 25.0 and luma_p95 >= 120.0)
        )
    )

    rg_ratio = round(mean_r / mean_g, 4) if mean_g >= 1.0 else None
    bg_ratio = round(mean_b / mean_g, 4) if mean_g >= 1.0 else None
    rb_ratio = round(mean_r / mean_b, 4) if mean_b >= 1.0 else None
    rb_log2_ratio = round(math.log2(mean_r / mean_b), 4) if (mean_r >= 1.0 and mean_b >= 1.0) else None
    green_cast_index = mean_g - 0.5 * (mean_r + mean_b)
    green_dominance = mean_g / max(1.0, 0.5 * (mean_r + mean_b))
    green_cast_suspected = (not is_black) and (green_dominance >= 1.25) and (green_cast_index >= 12.0)

    letterbox_frac = (top_bar + bottom_bar) / max(1, h)
    pillarbox_frac = (left_bar + right_bar) / max(1, w)
    bar_asym_v = abs(top_bar - bottom_bar) / max(1, h)
    bar_asym_h = abs(left_bar - right_bar) / max(1, w)
    active_aspect, framing_status = _classify_framing(
        active_w,
        active_h,
        letterbox_frac,
        pillarbox_frac,
        bar_asym_v,
        bar_asym_h,
        target_aspect,
        aspect_tolerance,
    )

    return {
        "width": w,
        "height": h,
        "frame_aspect": round(w / max(1, h), 4),
        "is_black": is_black,
        "mean_luma": round(mean_luma, 2),
        "luma_p05": round(luma_p05, 2),
        "luma_p50": round(luma_p50, 2),
        "luma_p95": round(luma_p95, 2),
        "max_luma": round(max_luma, 2),
        "pure_black_frac": round(pure_black_frac, 4),
        "near_black_frac": round(near_black_frac, 4),
        "shadow_crush_frac": round(shadow_crush_frac, 4),
        "active_pure_black_frac": round(active_pure_black_frac, 4),
        "active_near_black_frac": round(active_near_black_frac, 4),
        "active_luma_p50": round(active_luma_p50, 2),
        "clipped_white_frac": round(clipped_white_frac, 4),
        "gamma_crush_suspected": gamma_crush_suspected,
        "mean_r": round(mean_r, 2),
        "mean_g": round(mean_g, 2),
        "mean_b": round(mean_b, 2),
        "rg_ratio": rg_ratio,
        "bg_ratio": bg_ratio,
        "rb_ratio": rb_ratio,
        "rb_log2_ratio": rb_log2_ratio,
        "green_cast_index": round(green_cast_index, 2),
        "green_dominance": round(green_dominance, 4),
        "green_cast_suspected": green_cast_suspected,
        "mean_hsv_sat": round(mean_hsv_sat, 4),
        "mean_hsv_sat_non_black": round(mean_hsv_sat_non_black, 4),
        "colorfulness_hs": round(colorfulness_hs, 2),
        "top_bar": top_bar,
        "bottom_bar": bottom_bar,
        "left_bar": left_bar,
        "right_bar": right_bar,
        "active_width": active_w,
        "active_height": active_h,
        "active_aspect": active_aspect,
        "letterbox_frac": round(letterbox_frac, 4),
        "pillarbox_frac": round(pillarbox_frac, 4),
        "framing_status": framing_status,
    }


# ---------------------------------------------------------------------------
# Run-level auditing
# ---------------------------------------------------------------------------


def _median_or_none(values: Sequence[float | None], digits: int = 4) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.median(clean), digits)


def audit_directory(
    target: str | Path,
    *,
    interval_sec: float | None = None,
    crop_x: int = 0,
    auto_crop_hud: bool = False,
    black_luma: float = DEFAULT_BLACK_LUMA,
    near_black_luma: float = DEFAULT_NEAR_BLACK_LUMA,
    white_luma: float = DEFAULT_WHITE_LUMA,
    diff_threshold: float = DEFAULT_DIFF_THRESHOLD,
    transition_threshold: float = DEFAULT_TRANSITION_THRESHOLD,
    target_aspect: float = DEFAULT_TARGET_ASPECT,
    aspect_tolerance: float = DEFAULT_ASPECT_TOLERANCE,
    sample_stride: int = 1,
    label: str | None = None,
    include_per_frame: bool = False,
    pil_image_mod: Any | None = None,
    _auto_import_pil: bool = True,
    _use_numpy: bool = True,
) -> dict[str, Any]:
    """Audit a run directory or frames directory and return a sanitized report dict."""
    target_path = Path(target)
    frames_dir, run_dir = resolve_frames_dir(target_path)
    discovered = discover_frames(frames_dir)
    if not discovered:
        raise ValueError("no supported image frames found")

    inferred_interval, inferred_source, hide_hud = _infer_sidecar_metadata(run_dir, discovered)
    if interval_sec is not None:
        effective_interval: float | None = float(interval_sec)
        timestamp_source = "cli"
    elif all(f.embedded_timestamp_s is not None for f in discovered):
        effective_interval = None
        timestamp_source = "filename"
    elif any(f.embedded_timestamp_s is not None for f in discovered):
        effective_interval = inferred_interval
        timestamp_source = "filename"
    else:
        effective_interval = inferred_interval
        timestamp_source = inferred_source

    run_label = sanitize_label(
        label if label is not None else (run_dir.name if run_dir is not None else target_path.name),
        fallback="run",
    )

    decoded_records: list[dict[str, Any]] = []
    thumbnails: list[bytes] = []
    decoders_used: set[str] = set()
    skipped_reasons: Counter[str] = Counter()
    warnings: list[str] = []
    effective_crop_x = max(0, int(crop_x))

    for ord_idx, item in enumerate(discovered):
        img, err = decode_image_rgb(
            item.path,
            pil_image_mod=pil_image_mod,
            _auto_import_pil=_auto_import_pil,
        )
        if img is None:
            skipped_reasons[err or "unsupported_or_corrupt"] += 1
            continue
        decoders_used.add(img.decoder)

        frame_crop_x = effective_crop_x
        if auto_crop_hud and frame_crop_x == 0 and hide_hud is False and img.width > CANVAS_MIN_X:
            frame_crop_x = CANVAS_MIN_X
            effective_crop_x = CANVAS_MIN_X
        if frame_crop_x > 0 and frame_crop_x < img.width:
            img = crop_rgb_x(img, frame_crop_x)

        metrics = analyze_frame_rgb(
            img,
            sample_stride=sample_stride,
            black_luma=black_luma,
            near_black_luma=near_black_luma,
            white_luma=white_luma,
            target_aspect=target_aspect,
            aspect_tolerance=aspect_tolerance,
            _use_numpy=_use_numpy,
        )
        thumb = downsample_thumbnail(
            img,
            THUMB_SIZE,
            pil_image_mod=pil_image_mod,
            _auto_import_pil=_auto_import_pil,
        )
        ts: float | None = item.embedded_timestamp_s
        if ts is None and effective_interval is not None:
            ts = round(ord_idx * effective_interval, 3)

        rec: dict[str, Any] = {
            "name": item.name,
            "frame_index": item.frame_index,
            "timestamp_s": ts,
            **metrics,
        }
        decoded_records.append(rec)
        thumbnails.append(thumb)

    if skipped_reasons.get("requires_pillow", 0) > 0:
        warnings.append(
            f"Pillow not installed; skipped {skipped_reasons['requires_pillow']} non-PNG frame(s)"
        )
    if skipped_reasons.get("unsupported_or_corrupt", 0) > 0:
        warnings.append(
            f"skipped {skipped_reasons['unsupported_or_corrupt']} unsupported or corrupt frame(s)"
        )
    if skipped_reasons.get("read_error", 0) > 0:
        warnings.append(f"failed to read {skipped_reasons['read_error']} frame(s)")

    if not decoded_records:
        raise ValueError(
            "zero decodable image frames"
            + (f" ({'; '.join(warnings)})" if warnings else "")
        )

    # Frame-to-frame MAD transitions and anchor-based dwell segments
    transitions: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    dynamic_frames = 0
    prev_transition_ts: float | None = None

    cur_seg: dict[str, Any] | None = None
    anchor_thumb: bytes | None = None

    for i, rec in enumerate(decoded_records):
        thumb = thumbnails[i]
        step_mad = 0.0 if i == 0 else thumbnail_mad(thumb, thumbnails[i - 1])
        rec["step_mad"] = step_mad

        if i > 0:
            prev = decoded_records[i - 1]
            if (not rec["is_black"]) and step_mad > diff_threshold:
                dynamic_frames += 1

            kind: str | None = None
            if prev["is_black"] and not rec["is_black"]:
                kind = "black_to_visible"
            elif (not prev["is_black"]) and rec["is_black"]:
                kind = "visible_to_black"
            elif step_mad >= transition_threshold:
                kind = "scene_change"

            if kind is not None:
                ts_cur = rec["timestamp_s"]
                delta_prev: float | None = None
                if ts_cur is not None:
                    if prev_transition_ts is not None:
                        delta_prev = round(ts_cur - prev_transition_ts, 3)
                    else:
                        first_ts = decoded_records[0]["timestamp_s"]
                        if first_ts is not None:
                            delta_prev = round(ts_cur - first_ts, 3)
                    prev_transition_ts = ts_cur
                transitions.append(
                    {
                        "index": len(transitions),
                        "from_frame": prev["frame_index"],
                        "to_frame": rec["frame_index"],
                        "timestamp_s": ts_cur,
                        "delta_prev_s": delta_prev,
                        "step_mad": step_mad,
                        "luma_from": prev["mean_luma"],
                        "luma_to": rec["mean_luma"],
                        "kind": kind,
                    }
                )

        # Anchor-based segment grouping
        stays = False
        if cur_seg is not None and anchor_thumb is not None:
            if cur_seg["is_black"] == rec["is_black"]:
                if rec["is_black"] or thumbnail_mad(thumb, anchor_thumb) <= diff_threshold:
                    stays = True

        if stays and cur_seg is not None:
            cur_seg["end_frame"] = rec["frame_index"]
            cur_seg["end_s"] = rec["timestamp_s"]
            cur_seg["frame_count"] += 1
            cur_seg["_luma_sum"] += rec["mean_luma"]
            cur_seg["_mad_sum"] += step_mad
            cur_seg["_mad_cnt"] += 1
        else:
            cur_seg = {
                "index": len(segments),
                "start_frame": rec["frame_index"],
                "end_frame": rec["frame_index"],
                "frame_count": 1,
                "start_s": rec["timestamp_s"],
                "end_s": rec["timestamp_s"],
                "is_black": rec["is_black"],
                "_luma_sum": rec["mean_luma"],
                "_mad_sum": 0.0,
                "_mad_cnt": 0,
            }
            anchor_thumb = thumb
            segments.append(cur_seg)

    for seg in segments:
        cnt = max(1, seg["frame_count"])
        seg["mean_luma"] = round(seg.pop("_luma_sum") / cnt, 2)
        mad_cnt = seg.pop("_mad_cnt")
        mad_sum = seg.pop("_mad_sum")
        seg["mean_step_mad"] = round(mad_sum / mad_cnt, 3) if mad_cnt > 0 else 0.0
        if seg["start_s"] is not None and seg["end_s"] is not None:
            if effective_interval is not None:
                seg["dwell_s"] = round((seg["end_s"] - seg["start_s"]) + effective_interval, 3)
            else:
                seg["dwell_s"] = round(seg["end_s"] - seg["start_s"], 3)
        else:
            seg["dwell_s"] = None

    visible = [r for r in decoded_records if not r["is_black"]]
    black_count = len(decoded_records) - len(visible)
    all_black = len(visible) == 0

    first_visible_rec = visible[0] if visible else None
    first_ts = decoded_records[0]["timestamp_s"]
    last_ts = decoded_records[-1]["timestamp_s"]
    first_visible_s = first_visible_rec["timestamp_s"] if first_visible_rec else None
    black_loading_s: float | None = None
    if first_visible_s is not None and first_ts is not None:
        black_loading_s = round(max(0.0, first_visible_s - first_ts), 3)

    # Luminance & gamma summary (over visible frames, or all frames if all_black)
    luma_pool = visible if visible else decoded_records
    gamma_crush_count = sum(1 for r in visible if r["gamma_crush_suspected"])
    gamma_crush_frac = round(gamma_crush_count / len(visible), 4) if visible else 0.0

    # Color summary
    green_cast_count = sum(1 for r in visible if r["green_cast_suspected"])
    green_cast_suspected = bool(visible) and (green_cast_count / len(visible) >= 0.5)

    # Framing & run-level union active box
    dims = [(r["width"], r["height"]) for r in decoded_records]
    dim_counts = Counter(dims)
    modal_w, modal_h = dim_counts.most_common(1)[0][0]
    resolution_changes = sum(1 for a, b in zip(dims, dims[1:]) if a != b)

    modal_visible = [r for r in visible if (r["width"], r["height"]) == (modal_w, modal_h)]
    if modal_visible:
        u_top = min(r["top_bar"] for r in modal_visible)
        u_bot = min(r["bottom_bar"] for r in modal_visible)
        u_left = min(r["left_bar"] for r in modal_visible)
        u_right = min(r["right_bar"] for r in modal_visible)
        u_w = max(0, modal_w - u_left - u_right)
        u_h = max(0, modal_h - u_top - u_bot)
        u_lb = (u_top + u_bot) / max(1, modal_h)
        u_pb = (u_left + u_right) / max(1, modal_w)
        u_asym_v = abs(u_top - u_bot) / max(1, modal_h)
        u_asym_h = abs(u_left - u_right) / max(1, modal_w)
        u_aspect, u_status = _classify_framing(
            u_w, u_h, u_lb, u_pb, u_asym_v, u_asym_h, target_aspect, aspect_tolerance
        )
    else:
        u_w = u_h = 0
        u_lb = 1.0
        u_pb = 1.0
        u_aspect = None
        u_status = "all_black"

    framing_status_counts = dict(sorted(Counter(r["framing_status"] for r in decoded_records).items()))

    report: dict[str, Any] = {
        "schema": AUDIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "label": run_label,
        "config": {
            "crop_x": effective_crop_x,
            "sample_stride": max(1, int(sample_stride)),
            "black_luma": round(float(black_luma), 3),
            "near_black_luma": round(float(near_black_luma), 3),
            "white_luma": round(float(white_luma), 3),
            "diff_threshold": round(float(diff_threshold), 3),
            "transition_threshold": round(float(transition_threshold), 3),
            "target_aspect": round(float(target_aspect), 4),
            "aspect_tolerance": round(float(aspect_tolerance), 4),
            "interval_sec": round(effective_interval, 4) if effective_interval is not None else None,
            "timestamp_source": timestamp_source,
            "decoders_used": sorted(decoders_used),
        },
        "counts": {
            "discovered_files": len(discovered),
            "decoded_frames": len(decoded_records),
            "skipped_frames": sum(skipped_reasons.values()),
            "skipped_reasons": dict(sorted(skipped_reasons.items())),
            "black_frames": black_count,
            "visible_frames": len(visible),
            "dynamic_frames": dynamic_frames,
            "all_black": all_black,
        },
        "timing": {
            "first_visible_frame": first_visible_rec["frame_index"] if first_visible_rec else None,
            "first_visible_s": first_visible_s,
            "black_loading_s": black_loading_s,
            "last_timestamp_s": last_ts,
        },
        "luminance": {
            "visible_median_mean_luma": _median_or_none([r["mean_luma"] for r in luma_pool], 2),
            "visible_median_p05": _median_or_none([r["luma_p05"] for r in luma_pool], 2),
            "visible_median_p50": _median_or_none([r["luma_p50"] for r in luma_pool], 2),
            "visible_median_p95": _median_or_none([r["luma_p95"] for r in luma_pool], 2),
            "peak_max_luma": round(max(r["max_luma"] for r in decoded_records), 2),
            "visible_median_pure_black_frac": _median_or_none([r["pure_black_frac"] for r in luma_pool], 4),
            "visible_median_near_black_frac": _median_or_none([r["near_black_frac"] for r in luma_pool], 4),
            "visible_median_shadow_crush_frac": _median_or_none([r["shadow_crush_frac"] for r in luma_pool], 4),
            "visible_median_active_pure_black_frac": _median_or_none(
                [r["active_pure_black_frac"] for r in luma_pool], 4
            ),
            "visible_median_active_near_black_frac": _median_or_none(
                [r["active_near_black_frac"] for r in luma_pool], 4
            ),
            "visible_median_clipped_white_frac": _median_or_none([r["clipped_white_frac"] for r in luma_pool], 4),
            "gamma_crush_frame_count": gamma_crush_count,
            "gamma_crush_frame_frac": gamma_crush_frac,
            "gamma_crush_suspected": bool(visible) and (gamma_crush_frac >= 0.50),
        },
        "color": {
            "visible_median_mean_r": _median_or_none([r["mean_r"] for r in luma_pool], 2),
            "visible_median_mean_g": _median_or_none([r["mean_g"] for r in luma_pool], 2),
            "visible_median_mean_b": _median_or_none([r["mean_b"] for r in luma_pool], 2),
            "visible_median_rg_ratio": _median_or_none([r["rg_ratio"] for r in luma_pool], 4),
            "visible_median_bg_ratio": _median_or_none([r["bg_ratio"] for r in luma_pool], 4),
            "visible_median_rb_ratio": _median_or_none([r["rb_ratio"] for r in luma_pool], 4),
            "visible_median_rb_log2_ratio": _median_or_none([r["rb_log2_ratio"] for r in luma_pool], 4),
            "visible_median_green_cast_index": _median_or_none([r["green_cast_index"] for r in luma_pool], 2),
            "green_cast_frame_count": green_cast_count,
            "green_cast_suspected": green_cast_suspected,
            "visible_median_hsv_sat": _median_or_none([r["mean_hsv_sat"] for r in luma_pool], 4),
            "visible_median_hsv_sat_non_black": _median_or_none(
                [r["mean_hsv_sat_non_black"] for r in luma_pool], 4
            ),
            "visible_median_colorfulness_hs": _median_or_none([r["colorfulness_hs"] for r in luma_pool], 2),
        },
        "framing": {
            "modal_width": modal_w,
            "modal_height": modal_h,
            "frame_aspect": round(modal_w / max(1, modal_h), 4),
            "resolution_changes": resolution_changes,
            "union_active_width": u_w,
            "union_active_height": u_h,
            "union_active_aspect": u_aspect,
            "union_letterbox_frac": round(u_lb, 4),
            "union_pillarbox_frac": round(u_pb, 4),
            "union_framing_status": u_status,
            "framing_status_counts": framing_status_counts,
        },
        "progression": {
            "segment_count": len(segments),
            "transition_count": len(transitions),
            "transitions": transitions,
            "segments": segments,
        },
        "warnings": warnings,
    }
    if include_per_frame:
        report["frames"] = decoded_records
    return sanitize_payload(report)


# ---------------------------------------------------------------------------
# Before / after comparison
# ---------------------------------------------------------------------------


_GOOD_FRAMING_STATUSES = frozenset({"full_16_9", "letterboxed_16_9", "pillarboxed_16_9"})


def _metric_delta(a: float | int | None, b: float | int | None, digits: int = 2) -> dict[str, Any]:
    if a is None or b is None:
        return {"a": a, "b": b, "delta": None, "pct": None}
    if isinstance(a, int) and isinstance(b, int) and not isinstance(a, bool) and not isinstance(b, bool):
        delta: int | float = b - a
    else:
        a = round(float(a), digits)
        b = round(float(b), digits)
        delta = round(b - a, digits)
        if delta == 0.0:
            delta = 0.0
    if a != 0:
        pct: float | None = round(((b - a) / a) * 100.0, 2)
    elif b == 0:
        pct = 0.0
    else:
        pct = None
    return {"a": a, "b": b, "delta": delta, "pct": pct}


def compare_audits(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    labels: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Compare two `audit_directory` reports and produce a sanitized diff & verdict."""
    lbl_a = sanitize_label(labels[0] if labels else baseline.get("label"), fallback="baseline")
    lbl_b = sanitize_label(labels[1] if labels else candidate.get("label"), fallback="candidate")

    ca, cb = baseline.get("counts") or {}, candidate.get("counts") or {}
    ta, tb = baseline.get("timing") or {}, candidate.get("timing") or {}
    la, lb = baseline.get("luminance") or {}, candidate.get("luminance") or {}
    cola, colb = baseline.get("color") or {}, candidate.get("color") or {}
    fa, fb = baseline.get("framing") or {}, candidate.get("framing") or {}
    pa, pb = baseline.get("progression") or {}, candidate.get("progression") or {}

    metrics: dict[str, dict[str, Any]] = {
        "visible_frames": _metric_delta(ca.get("visible_frames"), cb.get("visible_frames"), 0),
        "first_visible_s": _metric_delta(ta.get("first_visible_s"), tb.get("first_visible_s"), 3),
        "visible_median_mean_luma": _metric_delta(
            la.get("visible_median_mean_luma"), lb.get("visible_median_mean_luma"), 2
        ),
        "visible_median_p05": _metric_delta(la.get("visible_median_p05"), lb.get("visible_median_p05"), 2),
        "visible_median_p50": _metric_delta(la.get("visible_median_p50"), lb.get("visible_median_p50"), 2),
        "visible_median_p95": _metric_delta(la.get("visible_median_p95"), lb.get("visible_median_p95"), 2),
        "peak_max_luma": _metric_delta(la.get("peak_max_luma"), lb.get("peak_max_luma"), 2),
        "active_pure_black_frac": _metric_delta(
            la.get("visible_median_active_pure_black_frac"),
            lb.get("visible_median_active_pure_black_frac"),
            4,
        ),
        "active_near_black_frac": _metric_delta(
            la.get("visible_median_active_near_black_frac"),
            lb.get("visible_median_active_near_black_frac"),
            4,
        ),
        "clipped_white_frac": _metric_delta(
            la.get("visible_median_clipped_white_frac"), lb.get("visible_median_clipped_white_frac"), 4
        ),
        "mean_r": _metric_delta(cola.get("visible_median_mean_r"), colb.get("visible_median_mean_r"), 2),
        "mean_g": _metric_delta(cola.get("visible_median_mean_g"), colb.get("visible_median_mean_g"), 2),
        "mean_b": _metric_delta(cola.get("visible_median_mean_b"), colb.get("visible_median_mean_b"), 2),
        "rg_ratio": _metric_delta(cola.get("visible_median_rg_ratio"), colb.get("visible_median_rg_ratio"), 4),
        "bg_ratio": _metric_delta(cola.get("visible_median_bg_ratio"), colb.get("visible_median_bg_ratio"), 4),
        "rb_ratio": _metric_delta(cola.get("visible_median_rb_ratio"), colb.get("visible_median_rb_ratio"), 4),
        "green_cast_index": _metric_delta(
            cola.get("visible_median_green_cast_index"), colb.get("visible_median_green_cast_index"), 2
        ),
        "mean_hsv_sat": _metric_delta(
            cola.get("visible_median_hsv_sat"), colb.get("visible_median_hsv_sat"), 4
        ),
        "colorfulness_hs": _metric_delta(
            cola.get("visible_median_colorfulness_hs"), colb.get("visible_median_colorfulness_hs"), 2
        ),
        "transition_count": _metric_delta(pa.get("transition_count"), pb.get("transition_count"), 0),
        "segment_count": _metric_delta(pa.get("segment_count"), pb.get("segment_count"), 0),
        "union_active_aspect": _metric_delta(
            fa.get("union_active_aspect"), fb.get("union_active_aspect"), 4
        ),
        "union_letterbox_frac": _metric_delta(
            fa.get("union_letterbox_frac"), fb.get("union_letterbox_frac"), 4
        ),
        "union_pillarbox_frac": _metric_delta(
            fa.get("union_pillarbox_frac"), fb.get("union_pillarbox_frac"), 4
        ),
    }

    improvements: list[str] = []
    regressions: list[str] = []
    warnings: list[str] = []

    # 1. All-black regression / recovery
    if bool(cb.get("all_black")) and not bool(ca.get("all_black")):
        regressions.append("candidate became all-black (baseline had visible frames)")
    elif bool(ca.get("all_black")) and not bool(cb.get("all_black")):
        improvements.append("candidate recovered visible frames from all-black baseline")

    # 2. Gamma / black-crush detection
    anb_a = la.get("visible_median_active_near_black_frac")
    anb_b = lb.get("visible_median_active_near_black_frac")
    gc_a = bool(la.get("gamma_crush_suspected"))
    gc_b = bool(lb.get("gamma_crush_suspected"))
    if gc_b and not gc_a:
        regressions.append("gamma crush suspected in candidate (double-gamma shadow crush)")
    elif gc_a and not gc_b:
        improvements.append("gamma crush resolved in candidate")

    if isinstance(anb_a, (int, float)) and isinstance(anb_b, (int, float)):
        diff_nb = anb_b - anb_a
        if diff_nb <= -0.10 and anb_a >= 0.40:
            improvements.append(
                f"active near-black fraction reduced {anb_a * 100:.1f}% -> {anb_b * 100:.1f}% ({diff_nb * 100:+.1f} pp)"
            )
        elif diff_nb >= 0.10 and anb_b >= 0.45:
            regressions.append(
                f"active near-black fraction increased {anb_a * 100:.1f}% -> {anb_b * 100:.1f}% ({diff_nb * 100:+.1f} pp)"
            )

    # 3. R/B channel swap & green cast
    rb_log_a = cola.get("visible_median_rb_log2_ratio")
    rb_log_b = colb.get("visible_median_rb_log2_ratio")
    mg_a = cola.get("visible_median_mean_g")
    mg_b = colb.get("visible_median_mean_g")
    rb_swap_suspected = False
    if (
        isinstance(rb_log_a, (int, float))
        and isinstance(rb_log_b, (int, float))
        and isinstance(mg_a, (int, float))
        and isinstance(mg_b, (int, float))
    ):
        if (
            abs(rb_log_a) >= 0.35
            and (rb_log_a * rb_log_b) < 0.0
            and abs(rb_log_a + rb_log_b) <= 0.25
            and abs(mg_b - mg_a) / max(1.0, mg_a) <= 0.15
        ):
            rb_swap_suspected = True
            regressions.append(
                f"R/B channel swap suspected (rb_ratio {cola.get('visible_median_rb_ratio')} -> "
                f"{colb.get('visible_median_rb_ratio')})"
            )

    green_a = bool(cola.get("green_cast_suspected"))
    green_b = bool(colb.get("green_cast_suspected"))
    if green_b and not green_a:
        regressions.append("green tint/cast suspected in candidate")
    elif green_a and not green_b:
        improvements.append("green tint/cast resolved in candidate")

    # 4. Colorfulness / desaturation
    cf_a = cola.get("visible_median_colorfulness_hs")
    cf_b = colb.get("visible_median_colorfulness_hs")
    if isinstance(cf_a, (int, float)) and isinstance(cf_b, (int, float)):
        cf_diff = cf_b - cf_a
        if cf_a >= 15.0 and cf_diff <= -10.0 and (cf_diff / cf_a) <= -0.30:
            regressions.append(
                f"colorfulness dropped sharply ({cf_a:.2f} -> {cf_b:.2f}, {cf_diff:+.2f})"
            )
        elif cf_b >= 15.0 and cf_diff >= 10.0 and cf_a < 15.0:
            improvements.append(
                f"colorfulness restored ({cf_a:.2f} -> {cf_b:.2f}, {cf_diff:+.2f})"
            )

    # 5. Progression transitions
    tc_a = int(pa.get("transition_count") or 0)
    tc_b = int(pb.get("transition_count") or 0)
    if tc_a > 0 and tc_b == 0:
        regressions.append(f"screen transitions collapsed ({tc_a} -> 0)")
    elif tc_a == 0 and tc_b > 0:
        improvements.append(f"screen transitions gained (0 -> {tc_b})")

    # 6. Framing / aspect ratio
    fs_a = str(fa.get("union_framing_status") or "unknown")
    fs_b = str(fb.get("union_framing_status") or "unknown")
    if fs_a in _GOOD_FRAMING_STATUSES and fs_b not in _GOOD_FRAMING_STATUSES:
        regressions.append(f"framing degraded ({fs_a} -> {fs_b})")
    elif fs_a not in _GOOD_FRAMING_STATUSES and fs_b in _GOOD_FRAMING_STATUSES:
        improvements.append(f"framing fixed ({fs_a} -> {fs_b})")

    # Any non-zero numeric change?
    any_changed = (fs_a != fs_b) or any(
        entry.get("delta") not in (None, 0, 0.0) for entry in metrics.values()
    )

    if regressions:
        verdict = "REGRESSED"
    elif improvements:
        verdict = "IMPROVED"
    elif any_changed:
        verdict = "CHANGED"
    else:
        verdict = "UNCHANGED"

    trans_a = [
        {"to_frame": t.get("to_frame"), "timestamp_s": t.get("timestamp_s"), "kind": t.get("kind")}
        for t in (pa.get("transitions") or [])
    ]
    trans_b = [
        {"to_frame": t.get("to_frame"), "timestamp_s": t.get("timestamp_s"), "kind": t.get("kind")}
        for t in (pb.get("transitions") or [])
    ]

    result: dict[str, Any] = {
        "schema": COMPARE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "labels": [lbl_a, lbl_b],
        "verdict": verdict,
        "flags": {
            "rb_swap_suspected": rb_swap_suspected,
            "baseline_gamma_crush": gc_a,
            "candidate_gamma_crush": gc_b,
            "baseline_green_cast": green_a,
            "candidate_green_cast": green_b,
        },
        "framing_status": {"a": fs_a, "b": fs_b},
        "metrics": metrics,
        "transitions": {"a": trans_a, "b": trans_b},
        "improvements": improvements,
        "regressions": regressions,
        "warnings": warnings,
    }
    return sanitize_payload(result)


# ---------------------------------------------------------------------------
# Formatters (plain text & GitHub-flavored Markdown)
# ---------------------------------------------------------------------------


def _fmt_num(v: Any, suffix: str = "", digits: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return f"{v}{suffix}"
    return f"{float(v):.{digits}f}{suffix}"


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "-"
    return f"{float(v) * 100.0:.1f}%"


def format_audit_text(report: Mapping[str, Any]) -> str:
    c = report.get("counts") or {}
    t = report.get("timing") or {}
    l = report.get("luminance") or {}
    col = report.get("color") or {}
    f = report.get("framing") or {}
    p = report.get("progression") or {}
    cfg = report.get("config") or {}

    lines = [
        f"Visual Frame Audit: {report.get('label', 'run')}",
        f"  frames: {c.get('decoded_frames', 0)} decoded / {c.get('discovered_files', 0)} discovered "
        f"({c.get('visible_frames', 0)} visible, {c.get('black_frames', 0)} black, "
        f"{c.get('dynamic_frames', 0)} dynamic, {c.get('skipped_frames', 0)} skipped) "
        f"decoders={','.join(cfg.get('decoders_used') or []) or 'none'} crop_x={cfg.get('crop_x', 0)}",
        f"  timing: first_visible_frame={t.get('first_visible_frame')} "
        f"first_visible_s={_fmt_num(t.get('first_visible_s'), 's')} "
        f"black_loading_s={_fmt_num(t.get('black_loading_s'), 's')} "
        f"source={cfg.get('timestamp_source')}",
        f"  luminance (Rec.709 0..255): p05={_fmt_num(l.get('visible_median_p05'))} "
        f"p50={_fmt_num(l.get('visible_median_p50'))} p95={_fmt_num(l.get('visible_median_p95'))} "
        f"mean={_fmt_num(l.get('visible_median_mean_luma'))} peak_max={_fmt_num(l.get('peak_max_luma'))}",
        f"  black/white: pure_black={_fmt_pct(l.get('visible_median_pure_black_frac'))} "
        f"(active {_fmt_pct(l.get('visible_median_active_pure_black_frac'))}), "
        f"near_black={_fmt_pct(l.get('visible_median_near_black_frac'))} "
        f"(active {_fmt_pct(l.get('visible_median_active_near_black_frac'))}), "
        f"clipped_white={_fmt_pct(l.get('visible_median_clipped_white_frac'))}, "
        f"gamma_crush={'SUSPECTED' if l.get('gamma_crush_suspected') else 'ok'} "
        f"({l.get('gamma_crush_frame_count', 0)} frames)",
        f"  color: RGB=({_fmt_num(col.get('visible_median_mean_r'))}, "
        f"{_fmt_num(col.get('visible_median_mean_g'))}, {_fmt_num(col.get('visible_median_mean_b'))}) "
        f"R/G={_fmt_num(col.get('visible_median_rg_ratio'), digits=3)} "
        f"B/G={_fmt_num(col.get('visible_median_bg_ratio'), digits=3)} "
        f"R/B={_fmt_num(col.get('visible_median_rb_ratio'), digits=3)} "
        f"green_cast={_fmt_num(col.get('visible_median_green_cast_index'))} "
        f"({'SUSPECTED' if col.get('green_cast_suspected') else 'ok'})",
        f"  saturation: hsv_sat={_fmt_num(col.get('visible_median_hsv_sat'), digits=3)} "
        f"(non_black {_fmt_num(col.get('visible_median_hsv_sat_non_black'), digits=3)}), "
        f"colorfulness_hs={_fmt_num(col.get('visible_median_colorfulness_hs'))}",
        f"  framing: {f.get('modal_width')}x{f.get('modal_height')} (aspect {_fmt_num(f.get('frame_aspect'), digits=4)}), "
        f"union_active={f.get('union_active_width')}x{f.get('union_active_height')} "
        f"(aspect {_fmt_num(f.get('union_active_aspect'), digits=4)}), "
        f"letterbox={_fmt_pct(f.get('union_letterbox_frac'))}, "
        f"pillarbox={_fmt_pct(f.get('union_pillarbox_frac'))}, "
        f"status={f.get('union_framing_status')}",
        f"  progression: {p.get('transition_count', 0)} transitions, {p.get('segment_count', 0)} segments",
    ]
    for tr in (p.get("transitions") or [])[:12]:
        ts_str = f"t={tr['timestamp_s']}s" if tr.get("timestamp_s") is not None else "t=-"
        lines.append(
            f"    [{tr['index']}] frame {tr['from_frame']}->{tr['to_frame']} ({ts_str}): "
            f"{tr['kind']} mad={tr['step_mad']} luma {tr['luma_from']}->{tr['luma_to']}"
        )
    for w in report.get("warnings") or []:
        lines.append(f"  WARN: {w}")
    return sanitize_text("\n".join(lines))


def format_audit_markdown(report: Mapping[str, Any]) -> str:
    c = report.get("counts") or {}
    t = report.get("timing") or {}
    l = report.get("luminance") or {}
    col = report.get("color") or {}
    f = report.get("framing") or {}
    p = report.get("progression") or {}

    lines = [
        f"## Visual Frame Audit — `{report.get('label', 'run')}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Decoded / Visible / Black frames | {c.get('decoded_frames', 0)} / {c.get('visible_frames', 0)} / {c.get('black_frames', 0)} |",
        f"| First visible frame / time | {t.get('first_visible_frame')} / {_fmt_num(t.get('first_visible_s'), 's')} |",
        f"| Rec.709 Luma (p05 / p50 / p95 / peak) | {_fmt_num(l.get('visible_median_p05'))} / {_fmt_num(l.get('visible_median_p50'))} / {_fmt_num(l.get('visible_median_p95'))} / {_fmt_num(l.get('peak_max_luma'))} |",
        f"| Active Pure-Black / Near-Black / Clipped-White | {_fmt_pct(l.get('visible_median_active_pure_black_frac'))} / {_fmt_pct(l.get('visible_median_active_near_black_frac'))} / {_fmt_pct(l.get('visible_median_clipped_white_frac'))} |",
        f"| Gamma Crush Suspected | {'YES' if l.get('gamma_crush_suspected') else 'no'} ({l.get('gamma_crush_frame_count', 0)} frames) |",
        f"| Mean RGB | ({_fmt_num(col.get('visible_median_mean_r'))}, {_fmt_num(col.get('visible_median_mean_g'))}, {_fmt_num(col.get('visible_median_mean_b'))}) |",
        f"| Colorfulness (HS) / HSV Sat | {_fmt_num(col.get('visible_median_colorfulness_hs'))} / {_fmt_num(col.get('visible_median_hsv_sat'), digits=3)} |",
        f"| Framing Status / Union Aspect | `{f.get('union_framing_status')}` / {_fmt_num(f.get('union_active_aspect'), digits=4)} |",
        f"| Transitions / Segments | {p.get('transition_count', 0)} / {p.get('segment_count', 0)} |",
    ]
    return sanitize_text("\n".join(lines))


_FRACTION_METRICS = frozenset(
    {"active_pure_black_frac", "active_near_black_frac", "clipped_white_frac", "union_letterbox_frac", "union_pillarbox_frac"}
)


def _fmt_compare_val(key: str, val: Any) -> str:
    if val is None:
        return "-"
    if key in _FRACTION_METRICS:
        return f"{float(val) * 100.0:.1f}%"
    if isinstance(val, int) and not isinstance(val, bool):
        return str(val)
    if key.endswith("_ratio") or key.endswith("_aspect") or key == "mean_hsv_sat":
        return f"{float(val):.4f}"
    return f"{float(val):.2f}"


def _fmt_compare_delta(key: str, item: Mapping[str, Any]) -> str:
    d = item.get("delta")
    if d is None:
        return "-"
    if key in _FRACTION_METRICS:
        return f"{float(d) * 100.0:+.1f} pp"
    if isinstance(d, int) and not isinstance(d, bool):
        return f"{d:+d}" if d != 0 else "0"
    digits = 4 if (key.endswith("_ratio") or key.endswith("_aspect") or key == "mean_hsv_sat") else 2
    return f"{float(d):+.{digits}f}" if d != 0 else f"{0.0:.{digits}f}"


def format_compare_text(diff: Mapping[str, Any]) -> str:
    lbl_a, lbl_b = (diff.get("labels") or ["baseline", "candidate"])[:2]
    fs = diff.get("framing_status") or {}
    lines = [
        f"Visual Frame Audit Compare: {lbl_a} -> {lbl_b} | verdict: {diff.get('verdict')}",
        f"  framing_status: {fs.get('a')} -> {fs.get('b')}",
    ]
    metrics = diff.get("metrics") or {}
    for key, item in metrics.items():
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"  {key:<28} {_fmt_compare_val(key, item.get('a')):>10} -> "
            f"{_fmt_compare_val(key, item.get('b')):>10} ({_fmt_compare_delta(key, item)})"
        )
    if diff.get("improvements"):
        lines.append("[improvements]")
        for msg in diff["improvements"]:
            lines.append(f"  + {msg}")
    if diff.get("regressions"):
        lines.append("[regressions]")
        for msg in diff["regressions"]:
            lines.append(f"  ! {msg}")
    if diff.get("warnings"):
        lines.append("[warnings]")
        for msg in diff["warnings"]:
            lines.append(f"  WARN {msg}")
    return sanitize_text("\n".join(lines))


def format_compare_markdown(diff: Mapping[str, Any]) -> str:
    lbl_a, lbl_b = (diff.get("labels") or ["baseline", "candidate"])[:2]
    fs = diff.get("framing_status") or {}
    lines = [
        f"## Visual Frame Audit Comparison — **{diff.get('verdict')}** (`{lbl_a}` -> `{lbl_b}`)",
        "",
        f"- Framing status: `{fs.get('a')}` -> `{fs.get('b')}`",
    ]
    for msg in diff.get("improvements") or []:
        lines.append(f"- **Improvement:** {msg}")
    for msg in diff.get("regressions") or []:
        lines.append(f"- **Regression:** {msg}")
    for msg in diff.get("warnings") or []:
        lines.append(f"- **Warning:** {msg}")

    lines.extend(
        [
            "",
            f"| Metric | {lbl_a} | {lbl_b} | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for key, item in (diff.get("metrics") or {}).items():
        if not isinstance(item, Mapping):
            continue
        lines.append(
            f"| `{key}` | {_fmt_compare_val(key, item.get('a'))} | "
            f"{_fmt_compare_val(key, item.get('b'))} | {_fmt_compare_delta(key, item)} |"
        )
    return sanitize_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Quantitative console-color, gamma, aspect-ratio, and screen-progression frame audit.",
        epilog="Exit code: 0 ok, 1 regression (with --compare --fail-on-regression), 2 usage/input error.",
    )
    ap.add_argument("target", nargs="?", type=Path, help="run directory or frames directory")
    ap.add_argument(
        "--compare",
        nargs=2,
        type=Path,
        metavar=("BASELINE", "CANDIDATE"),
        help="compare two run or frame directories (baseline vs candidate)",
    )
    ap.add_argument("--label", type=str, default=None, help="sanitized label for single-run mode")
    ap.add_argument("--labels", nargs=2, type=str, default=None, metavar=("BASE", "CAND"), help="sanitized labels for --compare")
    ap.add_argument("--interval-sec", type=float, default=None, help="override frame interval in seconds")
    ap.add_argument("--crop-x", type=int, default=0, help="crop left X pixels (e.g. 410 to skip HUD)")
    ap.add_argument("--auto-crop-hud", action="store_true", help="auto-crop left HUD when sidecar hide_hud=false")
    ap.add_argument("--black-luma", type=float, default=DEFAULT_BLACK_LUMA, help="mean luma <= threshold is black")
    ap.add_argument("--near-black-luma", type=float, default=DEFAULT_NEAR_BLACK_LUMA, help="near-black luma threshold (default 16.0)")
    ap.add_argument("--white-luma", type=float, default=DEFAULT_WHITE_LUMA, help="clipped-white luma threshold (default 254.0)")
    ap.add_argument("--diff-threshold", type=float, default=DEFAULT_DIFF_THRESHOLD, help="segment MAD threshold (default 4.0)")
    ap.add_argument("--transition-threshold", type=float, default=DEFAULT_TRANSITION_THRESHOLD, help="transition MAD threshold (default 12.0)")
    ap.add_argument("--target-aspect", type=float, default=DEFAULT_TARGET_ASPECT, help="target aspect ratio (default 16/9)")
    ap.add_argument("--aspect-tolerance", type=float, default=DEFAULT_ASPECT_TOLERANCE, help="relative aspect tolerance (default 0.06)")
    ap.add_argument("--sample-stride", type=int, default=1, help="pixel sampling stride >= 1 (default 1)")
    ap.add_argument("--per-frame", action="store_true", help="include per-frame numerical metrics in JSON output")
    ap.add_argument("--fail-on-regression", action="store_true", help="exit 1 when --compare verdict is REGRESSED")
    ap.add_argument("--json", dest="as_json", action="store_true", help="output JSON")
    ap.add_argument("--markdown", dest="as_markdown", action="store_true", help="output GitHub Markdown")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_arg_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return 0 if exc.code in (0, None) else 2

    if args.as_json and args.as_markdown:
        print("error: --json and --markdown are mutually exclusive", file=sys.stderr)
        return 2
    if args.compare and args.target is not None:
        print("error: pass either a target directory or --compare BASELINE CANDIDATE", file=sys.stderr)
        return 2
    if not args.compare and args.target is None:
        print("error: target directory or --compare BASELINE CANDIDATE is required", file=sys.stderr)
        return 2
    if args.sample_stride < 1:
        print("error: --sample-stride must be >= 1", file=sys.stderr)
        return 2
    if args.crop_x < 0:
        print("error: --crop-x must be >= 0", file=sys.stderr)
        return 2
    if args.interval_sec is not None and (not math.isfinite(args.interval_sec) or args.interval_sec <= 0):
        print("error: --interval-sec must be positive", file=sys.stderr)
        return 2

    common_kw = dict(
        interval_sec=args.interval_sec,
        crop_x=args.crop_x,
        auto_crop_hud=args.auto_crop_hud,
        black_luma=args.black_luma,
        near_black_luma=args.near_black_luma,
        white_luma=args.white_luma,
        diff_threshold=args.diff_threshold,
        transition_threshold=args.transition_threshold,
        target_aspect=args.target_aspect,
        aspect_tolerance=args.aspect_tolerance,
        sample_stride=args.sample_stride,
        include_per_frame=args.per_frame,
    )

    try:
        if args.compare:
            lbl_a = args.labels[0] if args.labels else None
            lbl_b = args.labels[1] if args.labels else None
            rep_a = audit_directory(args.compare[0], label=lbl_a, **common_kw)
            rep_b = audit_directory(args.compare[1], label=lbl_b, **common_kw)
            diff = compare_audits(
                rep_a,
                rep_b,
                labels=(args.labels[0], args.labels[1]) if args.labels else None,
            )
            if args.as_json:
                print(json.dumps(diff, indent=2))
            elif args.as_markdown:
                print(format_compare_markdown(diff))
            else:
                print(format_compare_text(diff))
            if args.fail_on_regression and diff.get("verdict") == "REGRESSED":
                return 1
            return 0

        report = audit_directory(args.target, label=args.label, **common_kw)
        if args.as_json:
            print(json.dumps(report, indent=2))
        elif args.as_markdown:
            print(format_audit_markdown(report))
        else:
            print(format_audit_text(report))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {sanitize_text(str(exc))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
