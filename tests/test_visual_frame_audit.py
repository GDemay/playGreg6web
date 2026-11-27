import io
import json
import math
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import visual_frame_audit as vfa  # noqa: E402


# ---------------------------------------------------------------------------
# Pure-Python synthetic PNG / JPEG generators (ZERO real game data!)
# ---------------------------------------------------------------------------


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def make_png_bytes(
    width: int,
    height: int,
    rows: list[bytes],
    *,
    color_type: int = 2,
    filter_type: int = 0,
    palette: bytes | None = None,
) -> bytes:
    """Build a valid 8-bit PNG in memory for any color type (0, 2, 3, 4, 6) and filter (0..4)."""
    bpp = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    stride = width * bpp
    assert len(rows) == height and all(len(r) == stride for r in rows)

    prev = bytes(stride)
    raw = bytearray()
    for row in rows:
        raw.append(filter_type)
        if filter_type == 0:
            raw.extend(row)
        elif filter_type == 1:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                raw.append((row[i] - left) & 0xFF)
        elif filter_type == 2:
            for i in range(stride):
                raw.append((row[i] - prev[i]) & 0xFF)
        elif filter_type == 3:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                raw.append((row[i] - ((left + prev[i]) >> 1)) & 0xFF)
        elif filter_type == 4:
            for i in range(stride):
                left = row[i - bpp] if i >= bpp else 0
                up = prev[i]
                up_left = prev[i - bpp] if i >= bpp else 0
                raw.append((row[i] - vfa._paeth_predictor(left, up, up_left)) & 0xFF)
        else:
            raise ValueError(filter_type)
        prev = row

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    parts = [b"\x89PNG\r\n\x1a\n", _png_chunk(b"IHDR", ihdr)]
    if color_type == 3:
        assert palette is not None
        parts.append(_png_chunk(b"PLTE", palette))
    parts.append(_png_chunk(b"IDAT", zlib.compress(bytes(raw))))
    parts.append(_png_chunk(b"IEND", b""))
    return b"".join(parts)


def make_solid_rgb_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    row = bytes(rgb) * width
    return make_png_bytes(width, height, [row] * height, color_type=2, filter_type=0)


def make_framed_rgb_png(
    width: int,
    height: int,
    fill: tuple[int, int, int],
    *,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    bg: tuple[int, int, int] = (0, 0, 0),
) -> bytes:
    rows: list[bytes] = []
    for y in range(height):
        if y < top or y >= height - bottom:
            rows.append(bytes(bg) * width)
        else:
            active_w = max(0, width - left - right)
            rows.append(bytes(bg) * left + bytes(fill) * active_w + bytes(bg) * right)
    return make_png_bytes(width, height, rows, color_type=2, filter_type=0)


def make_ramp_png(width: int, height: int, gamma: float = 1.0) -> bytes:
    """Horizontal grayscale ramp with optional gamma power curve `v^gamma`."""
    row = bytearray()
    for x in range(width):
        t = x / max(1, width - 1)
        val = int(round(255.0 * (t ** gamma)))
        val = max(0, min(255, val))
        row.extend((val, val, val))
    return make_png_bytes(width, height, [bytes(row)] * height, color_type=2, filter_type=0)


# ---------------------------------------------------------------------------
# 1. Pure-Python stdlib PNG decoder + Pillow fallback tests
# ---------------------------------------------------------------------------


class StdlibPngDecoderTests(unittest.TestCase):
    def test_all_png_filters_and_color_types(self):
        w, h = 4, 3
        # Color type 2 (RGB) across all 5 PNG filter types (0..4)
        rows_rgb = [
            bytes([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
            bytes([15, 25, 35, 45, 55, 65, 75, 85, 95, 105, 115, 125]),
            bytes([200, 100, 50, 210, 110, 60, 220, 120, 70, 230, 130, 80]),
        ]
        expected_rgb = b"".join(rows_rgb)
        for flt in range(5):
            png = make_png_bytes(w, h, rows_rgb, color_type=2, filter_type=flt)
            decoded = vfa.decode_png_stdlib(png)
            self.assertIsNotNone(decoded, f"filter {flt} failed")
            assert decoded is not None
            self.assertEqual((decoded.width, decoded.height, decoded.decoder), (w, h, "stdlib_png"))
            self.assertEqual(decoded.rgb, expected_rgb)

        # Color type 0 (Grayscale)
        gray_rows = [bytes([0, 64, 128, 255])] * h
        dec0 = vfa.decode_png_stdlib(make_png_bytes(w, h, gray_rows, color_type=0, filter_type=1))
        assert dec0 is not None
        self.assertEqual(dec0.rgb[:12], bytes([0, 0, 0, 64, 64, 64, 128, 128, 128, 255, 255, 255]))

        # Color type 3 (Indexed palette)
        palette = bytes([10, 20, 30, 200, 150, 100])
        idx_rows = [bytes([0, 1, 0, 1])] * h
        dec3 = vfa.decode_png_stdlib(make_png_bytes(w, h, idx_rows, color_type=3, filter_type=0, palette=palette))
        assert dec3 is not None
        self.assertEqual(dec3.rgb[:6], bytes([10, 20, 30, 200, 150, 100]))

        # Color type 4 (Grayscale + Alpha)
        ga_rows = [bytes([12, 255, 34, 128, 56, 0, 78, 255])] * h
        dec4 = vfa.decode_png_stdlib(make_png_bytes(w, h, ga_rows, color_type=4, filter_type=2))
        assert dec4 is not None
        self.assertEqual(dec4.rgb[:6], bytes([12, 12, 12, 34, 34, 34]))

        # Color type 6 (RGBA)
        rgba_rows = [bytes([1, 2, 3, 255, 4, 5, 6, 0, 7, 8, 9, 128, 10, 11, 12, 255])] * h
        dec6 = vfa.decode_png_stdlib(make_png_bytes(w, h, rgba_rows, color_type=6, filter_type=4))
        assert dec6 is not None
        self.assertEqual(dec6.rgb[:12], bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

    def test_graceful_skip_jpeg_without_pillow_and_decode_with_pillow(self):
        with tempfile.TemporaryDirectory() as td:
            jpg = Path(td) / "frame_0001.jpg"
            png = Path(td) / "frame_0002.png"
            png.write_bytes(make_solid_rgb_png(16, 9, (120, 100, 80)))

            pil = vfa._import_pil_image()
            if pil is not None:
                im = pil.new("RGB", (16, 9), (120, 100, 80))
                im.save(jpg, format="JPEG", quality=90)
                dec_ok, err_ok = vfa.decode_image_rgb(jpg)
                self.assertIsNone(err_ok)
                self.assertIsNotNone(dec_ok)
                assert dec_ok is not None
                self.assertEqual(dec_ok.decoder, "pillow")
            else:
                jpg.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16 + b"\xff\xd9")

            # Force no-Pillow path -> JPEG gracefully reports requires_pillow, PNG uses stdlib_png
            dec_jpg, err_jpg = vfa.decode_image_rgb(jpg, _auto_import_pil=False)
            self.assertIsNone(dec_jpg)
            self.assertEqual(err_jpg, "requires_pillow")

            dec_png, err_png = vfa.decode_image_rgb(png, _auto_import_pil=False)
            self.assertIsNone(err_png)
            assert dec_png is not None
            self.assertEqual(dec_png.decoder, "stdlib_png")

            # Directory audit with mixed JPEG + PNG under no-Pillow warns and succeeds on PNG
            rep = vfa.audit_directory(td, _auto_import_pil=False)
            self.assertEqual(rep["counts"]["decoded_frames"], 1)
            self.assertEqual(rep["counts"]["skipped_frames"], 1)
            self.assertEqual(rep["counts"]["skipped_reasons"], {"requires_pillow": 1})
            self.assertTrue(any("Pillow not installed" in w for w in rep["warnings"]))


# ---------------------------------------------------------------------------
# 2. Metric accuracy (pure-Python & numpy parity, gamma crush, color, aspect)
# ---------------------------------------------------------------------------


class MetricAccuracyTests(unittest.TestCase):
    def test_rec709_luma_colorfulness_and_pure_vs_numpy_parity(self):
        # Solid vivid orange-red (200, 100, 40)
        png = make_solid_rgb_png(32, 18, (200, 100, 40))
        img = vfa.decode_png_stdlib(png)
        assert img is not None

        m_pure = vfa.analyze_frame_rgb(img, _use_numpy=False)
        m_np = vfa.analyze_frame_rgb(img, _use_numpy=True)
        self.assertEqual(m_pure, m_np)

        expected_y = 0.2126 * 200 + 0.7152 * 100 + 0.0722 * 40  # 116.928
        self.assertAlmostEqual(m_pure["mean_luma"], expected_y, places=1)
        self.assertAlmostEqual(m_pure["luma_p50"], round(expected_y * 4) / 4, places=2)
        self.assertAlmostEqual(m_pure["max_luma"], expected_y, places=1)
        self.assertEqual(m_pure["pure_black_frac"], 0.0)
        self.assertEqual(m_pure["clipped_white_frac"], 0.0)
        self.assertEqual((m_pure["mean_r"], m_pure["mean_g"], m_pure["mean_b"]), (200.0, 100.0, 40.0))
        self.assertAlmostEqual(m_pure["rg_ratio"], 2.0, places=3)
        self.assertAlmostEqual(m_pure["rb_ratio"], 5.0, places=3)
        self.assertAlmostEqual(m_pure["mean_hsv_sat"], (200 - 40) / 200, places=4)
        self.assertGreater(m_pure["colorfulness_hs"], 30.0)

        # Pure grayscale has colorfulness == 0 and HSV sat == 0
        gray_img = vfa.decode_png_stdlib(make_solid_rgb_png(16, 9, (128, 128, 128)))
        assert gray_img is not None
        mg = vfa.analyze_frame_rgb(gray_img, _use_numpy=False)
        self.assertEqual(mg["colorfulness_hs"], 0.0)
        self.assertEqual(mg["mean_hsv_sat"], 0.0)

    def test_double_gamma_crush_vs_healthy_ramp(self):
        crushed = vfa.decode_png_stdlib(make_ramp_png(128, 72, gamma=4.84))
        healthy = vfa.decode_png_stdlib(make_ramp_png(128, 72, gamma=1.0))
        assert crushed is not None and healthy is not None

        mc = vfa.analyze_frame_rgb(crushed, _use_numpy=False)
        mh = vfa.analyze_frame_rgb(healthy, _use_numpy=False)
        self.assertTrue(mc["gamma_crush_suspected"])
        self.assertGreater(mc["active_near_black_frac"], 0.55)
        self.assertFalse(mh["gamma_crush_suspected"])
        self.assertLess(mh["active_near_black_frac"], 0.15)

    def test_clipped_white_and_green_cast_detection(self):
        white = vfa.decode_png_stdlib(make_solid_rgb_png(16, 9, (255, 255, 255)))
        green = vfa.decode_png_stdlib(make_solid_rgb_png(16, 9, (40, 160, 40)))
        assert white is not None and green is not None

        mw = vfa.analyze_frame_rgb(white, _use_numpy=False)
        self.assertEqual(mw["clipped_white_frac"], 1.0)
        self.assertEqual(mw["max_luma"], 255.0)

        mg = vfa.analyze_frame_rgb(green, _use_numpy=False)
        self.assertTrue(mg["green_cast_suspected"])
        self.assertAlmostEqual(mg["green_cast_index"], 120.0, places=1)

    def test_aspect_ratio_letterbox_pillarbox_windowbox_offcenter_and_squashed(self):
        # Full 16:9 (64x36)
        full = vfa.decode_png_stdlib(make_solid_rgb_png(64, 36, (90, 90, 90)))
        assert full is not None
        self.assertEqual(vfa.analyze_frame_rgb(full)["framing_status"], "full_16_9")

        # Letterboxed 16:9 inside 64x48 (top=6, bottom=6 -> active 64x36 = 16:9)
        lb = vfa.decode_png_stdlib(make_framed_rgb_png(64, 48, (90, 90, 90), top=6, bottom=6))
        assert lb is not None
        mlb = vfa.analyze_frame_rgb(lb)
        self.assertEqual(mlb["framing_status"], "letterboxed_16_9")
        self.assertAlmostEqual(mlb["letterbox_frac"], 0.25, places=3)
        self.assertAlmostEqual(mlb["active_aspect"], 64 / 36, places=3)

        # Pillarboxed 16:9 inside 80x36 (left=8, right=8 -> active 64x36 = 16:9)
        pb = vfa.decode_png_stdlib(make_framed_rgb_png(80, 36, (90, 90, 90), left=8, right=8))
        assert pb is not None
        mpb = vfa.analyze_frame_rgb(pb)
        self.assertEqual(mpb["framing_status"], "pillarboxed_16_9")
        self.assertAlmostEqual(mpb["pillarbox_frac"], 0.2, places=3)

        # Windowboxed (bars on all 4 sides)
        wb = vfa.decode_png_stdlib(make_framed_rgb_png(64, 36, (90, 90, 90), top=4, bottom=4, left=8, right=8))
        assert wb is not None
        self.assertEqual(vfa.analyze_frame_rgb(wb)["framing_status"], "windowboxed")

        # Off-center (asymmetric bars)
        oc = vfa.decode_png_stdlib(make_framed_rgb_png(64, 36, (90, 90, 90), top=10, bottom=0))
        assert oc is not None
        self.assertEqual(vfa.analyze_frame_rgb(oc)["framing_status"], "off_center")

        # Squashed 4:3 (64x48 full frame -> aspect 1.333 != 1.7778)
        sq = vfa.decode_png_stdlib(make_solid_rgb_png(64, 48, (90, 90, 90)))
        assert sq is not None
        self.assertEqual(vfa.analyze_frame_rgb(sq)["framing_status"], "aspect_mismatch")


# ---------------------------------------------------------------------------
# 3. Filename / timestamp parsing, HUD crop, transitions & compare mode
# ---------------------------------------------------------------------------


class ProgressionAndCompareTests(unittest.TestCase):
    def test_filename_timestamp_parsing_and_sidecar_inference(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "frame_0001_t0.5s.png").write_bytes(make_solid_rgb_png(32, 18, (0, 0, 0)))
            (d / "frame_0002_t1250ms.png").write_bytes(make_solid_rgb_png(32, 18, (120, 120, 120)))
            rep = vfa.audit_directory(d, include_per_frame=True, _auto_import_pil=False)
            self.assertEqual(rep["config"]["timestamp_source"], "filename")
            self.assertEqual([f["timestamp_s"] for f in rep["frames"]], [0.5, 1.25])
            self.assertEqual(rep["timing"]["first_visible_s"], 1.25)
            self.assertAlmostEqual(rep["timing"]["black_loading_s"], 0.75, places=3)

        # Native segment thumbnail & wall-clock parsing
        _, idx, ts, _ = vfa.parse_frame_filename(Path("seg02_14.5s_5d.png"))
        self.assertEqual((idx, ts), (2, 14.5))
        _, _, _, wc = vfa.parse_frame_filename(Path("20260102-153045-250.png"))
        self.assertIsNotNone(wc)

        # Sidecar command.json capture_interval_sec + auto-crop HUD
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "my_run"
            frames = run_dir / "frames"
            frames.mkdir(parents=True)
            (run_dir / "command.json").write_text(
                json.dumps({"capture_interval_sec": 2.5, "hide_hud": False}), encoding="utf-8"
            )
            # Left 410 px bright HUD, right 20 px black canvas -> auto-crop should isolate black canvas
            row = bytes((200, 200, 200)) * 410 + bytes((0, 0, 0)) * 20
            (frames / "frame_0001.png").write_bytes(make_png_bytes(430, 10, [row] * 10))
            (frames / "frame_0002.png").write_bytes(make_solid_rgb_png(430, 10, (100, 110, 120)))
            rep = vfa.audit_directory(run_dir, auto_crop_hud=True, include_per_frame=True, _auto_import_pil=False)
            self.assertEqual(rep["config"]["timestamp_source"], "sidecar")
            self.assertEqual(rep["config"]["interval_sec"], 2.5)
            self.assertEqual(rep["config"]["crop_x"], 410)
            self.assertEqual(rep["counts"]["black_frames"], 1)
            self.assertEqual(rep["timing"]["first_visible_s"], 2.5)

    def test_multi_phase_transitions_across_resolutions_and_union_active_box(self):
        with tempfile.TemporaryDirectory() as td:
            frames = Path(td) / "frames"
            frames.mkdir()
            # Black -> centered logo (windowboxed individually) -> full 16:9 Bink2 (different res) -> bright menu
            (frames / "frame_0001.png").write_bytes(make_solid_rgb_png(64, 36, (0, 0, 0)))
            (frames / "frame_0002.png").write_bytes(
                make_framed_rgb_png(64, 36, (180, 180, 180), top=10, bottom=10, left=16, right=16)
            )
            (frames / "frame_0003.png").write_bytes(make_solid_rgb_png(32, 18, (40, 120, 200)))
            (frames / "frame_0004.png").write_bytes(make_solid_rgb_png(64, 36, (200, 180, 60)))

            rep = vfa.audit_directory(td, interval_sec=1.0, _auto_import_pil=False)
            self.assertEqual(rep["counts"]["visible_frames"], 3)
            self.assertEqual(rep["counts"]["black_frames"], 1)
            self.assertEqual(rep["framing"]["resolution_changes"], 2)
            # Union active box across modal 64x36 frames includes frame_0004 (full 16:9), so union is full_16_9!
            self.assertEqual(rep["framing"]["union_framing_status"], "full_16_9")
            # Transitions: black->logo, logo->blue, blue->yellow
            self.assertEqual(rep["progression"]["transition_count"], 3)
            kinds = [t["kind"] for t in rep["progression"]["transitions"]]
            self.assertEqual(kinds, ["black_to_visible", "scene_change", "scene_change"])
            self.assertEqual([t["timestamp_s"] for t in rep["progression"]["transitions"]], [1.0, 2.0, 3.0])

    def test_compare_gamma_fix_rb_swap_cli_and_sanitization(self):
        with tempfile.TemporaryDirectory() as td:
            base_dir = Path(td) / "baseline_run" / "frames"
            cand_dir = Path(td) / "candidate_run" / "frames"
            swap_dir = Path(td) / "swapped_run" / "frames"
            for d in (base_dir, cand_dir, swap_dir):
                d.mkdir(parents=True)

            # 1. Gamma crush fix: baseline gamma=4.84 -> candidate gamma=1.0 => IMPROVED
            (base_dir / "frame_0001.png").write_bytes(make_ramp_png(128, 72, gamma=4.84))
            (cand_dir / "frame_0001.png").write_bytes(make_ramp_png(128, 72, gamma=1.0))

            out_json, err = io.StringIO(), io.StringIO()
            with redirect_stdout(out_json), redirect_stderr(err):
                code = vfa.main([
                    "--compare", str(base_dir.parent), str(cand_dir.parent),
                    "--labels", "before_gamma", "after_gamma",
                    "--fail-on-regression", "--json",
                ])
            self.assertEqual(code, 0, err.getvalue())
            diff = json.loads(out_json.getvalue())
            self.assertEqual(diff["schema"], vfa.COMPARE_SCHEMA)
            self.assertEqual(diff["verdict"], "IMPROVED")
            self.assertTrue(any("gamma crush resolved" in s for s in diff["improvements"]))
            # Zero host path leakage!
            self.assertNotIn(str(Path(td)), out_json.getvalue())

            # 2. R/B channel swap: (200, 100, 40) vs (40, 100, 200) => REGRESSED + exit 1 with --fail-on-regression
            (base_dir / "frame_0001.png").write_bytes(make_solid_rgb_png(64, 36, (200, 100, 40)))
            (swap_dir / "frame_0001.png").write_bytes(make_solid_rgb_png(64, 36, (40, 100, 200)))

            out_md, err2 = io.StringIO(), io.StringIO()
            with redirect_stdout(out_md), redirect_stderr(err2):
                code2 = vfa.main([
                    "--compare", str(base_dir.parent), str(swap_dir.parent),
                    "--fail-on-regression", "--markdown",
                ])
            self.assertEqual(code2, 1)
            self.assertIn("REGRESSED", out_md.getvalue())
            self.assertIn("R/B channel swap suspected", out_md.getvalue())
            self.assertNotIn(str(Path(td)), out_md.getvalue())

            # 3. Single-run text & markdown CLI + missing dir exit 2
            out_txt = io.StringIO()
            with redirect_stdout(out_txt):
                self.assertEqual(vfa.main([str(cand_dir.parent)]), 0)
            self.assertIn("Visual Frame Audit:", out_txt.getvalue())
            self.assertNotIn(str(Path(td)), out_txt.getvalue())

            out_err = io.StringIO()
            with redirect_stderr(out_err):
                self.assertEqual(vfa.main([str(Path(td) / "missing_run")]), 2)
            self.assertNotIn(str(Path(td)), out_err.getvalue())


if __name__ == "__main__":
    unittest.main()
