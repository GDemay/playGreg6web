#!/usr/bin/env python3
"""Unit tests for tools/inspect_rdna2_shaders.py (Track 4 / S15).

Covers:
  - Synthetic GFX10.3 Vertex Shader (`exp pos0` + `s_endpgm`),
    Pixel Shader (`v_interp` + `mimg` + `exp mrt0 done` + `s_endpgm`),
    Compute Shader (`smem` / `valu` + `s_endpgm` / `s_endpgm_saved`),
    and Unknown stage shaders (`s_nop` / `exp null` + `s_endpgm`)
  - Complete `EXP` target decoding (`mrt0..7`, `mrtz`, `null`, `pos0..4`,
    `param0..31`, `done` bit 11, `en_mask`, `compr`, `vm`)
  - Multi-shader buffers, console pipeline records (`ACCC` & `AMDGPU_ELF`),
    FNV-1a 64-bit hashes, size histograms, and export target histograms
  - Bounded chunked streaming across chunk boundaries (`chunk_size`/`overlap`),
    `--max-bytes`, `--max-hits`, and `--min-dwords`
  - Edge cases: empty, truncated, unaligned, zero-filled, and random noise files
  - CLI (`<path>`, `--from-env`, `--json`, Markdown) and strict host-path
    (Windows `C:\\Users\\...`, UNC `\\\\server\\...`, POSIX `/home/...`, `/Users/...`,
    active OS username) & zero-raw-shader-byte leak assertions
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import random
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import inspect_rdna2_shaders as rdna2  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic GFX10.3 / RDNA2 binary builders (zero copyrighted game data)
# ---------------------------------------------------------------------------


def encode_exp(
    target_id: int,
    *,
    en: int = 0xF,
    compr: bool = False,
    done: bool = False,
    vm: bool = False,
    word1: int = 0x03020100,
) -> tuple[int, int]:
    """Encode a synthetic 64-bit GFX10.3 `EXP` instruction `(word0, word1)`."""
    w0 = (
        0xF8000000
        | (en & 0xF)
        | ((target_id & 0x3F) << 4)
        | ((1 if compr else 0) << 10)
        | ((1 if done else 0) << 11)
        | ((1 if vm else 0) << 12)
    )
    return (w0, word1 & 0xFFFFFFFF)


def encode_smem(op: int = 0, word1: int = 0x00000000) -> tuple[int, int]:
    """Encode a synthetic 64-bit GFX10.3 `SMEM` instruction."""
    w0 = 0xF4000000 | ((op & 0xFF) << 18)
    return (w0, word1 & 0xFFFFFFFF)


def encode_vintrp(attr: int = 0, attrchan: int = 0, vdst: int = 0) -> int:
    """Encode a synthetic 32-bit GFX10.3 `VINTRP` instruction."""
    return 0xC8000000 | ((vdst & 0xFF) << 18) | ((attr & 0x3F) << 10) | ((attrchan & 0x3) << 8)


def encode_mimg(op: int = 0x20, dmask: int = 0xF, word1: int = 0x00000000) -> tuple[int, int]:
    """Encode a synthetic 64-bit GFX10.3 `MIMG` instruction (`nsize=0`)."""
    w0 = 0xF0000000 | ((op & 0xFF) << 18) | ((dmask & 0xF) << 8)
    return (w0, word1 & 0xFFFFFFFF)


def encode_v_mov_b32_imm(vdst: int, literal: int) -> tuple[int, int]:
    """Encode `v_mov_b32 v<vdst>, <literal>` (VOP1 `src0=0xFF` + 32-bit literal)."""
    w0 = 0x7E0002FF | ((vdst & 0xFF) << 17)
    return (w0, literal & 0xFFFFFFFF)


def encode_v_add_f32(vdst: int = 0, vsrc1: int = 1, src0: int = 0x100) -> int:
    """Encode a synthetic 32-bit GFX10.3 VOP2 `v_add_f32` instruction."""
    return (0x03 << 25) | ((vdst & 0xFF) << 17) | ((vsrc1 & 0xFF) << 9) | (src0 & 0x1FF)


def encode_s_nop(simm16: int = 0) -> int:
    """Encode GFX10.3 `s_nop`."""
    return 0xBF800000 | (simm16 & 0xFFFF)


def pack_words(*words: int) -> bytes:
    return struct.pack(f"<{len(words)}I", *(w & 0xFFFFFFFF for w in words))


def build_synthetic_vs(*, extra_param: bool = False, literal_marker: int | None = None) -> bytes:
    """Build a synthetic Vertex Shader (`exp pos0` + `s_endpgm`)."""
    words: list[int] = []
    if literal_marker is not None:
        words.extend(encode_v_mov_b32_imm(0, literal_marker))
    if extra_param:
        words.extend(encode_exp(32, done=False))  # param0
    words.extend(encode_exp(12, done=True))  # pos0, done=1
    words.append(rdna2.S_ENDPGM_WORD)
    return pack_words(*words)


def build_synthetic_ps(*, include_mimg: bool = True, literal_marker: int | None = None) -> bytes:
    """Build a synthetic Pixel Shader (`v_interp` + optional `mimg` + `exp mrt0 done` + `s_endpgm`)."""
    words: list[int] = [encode_vintrp(attr=0, attrchan=0, vdst=0)]
    if literal_marker is not None:
        words.extend(encode_v_mov_b32_imm(1, literal_marker))
    if include_mimg:
        words.extend(encode_mimg())
    words.extend(encode_exp(0, done=True, compr=True, vm=True))  # mrt0, done=1
    words.append(rdna2.S_ENDPGM_WORD)
    return pack_words(*words)


def build_synthetic_cs(
    *,
    saved_terminator: bool = False,
    use_valu_only: bool = False,
    literal_marker: int | None = None,
) -> bytes:
    """Build a synthetic Compute Shader (`smem`/`valu` without `exp` + `s_endpgm[_saved]`)."""
    words: list[int] = []
    if not use_valu_only:
        words.extend(encode_smem(op=2, word1=0x00000000))
    if literal_marker is not None:
        words.extend(encode_v_mov_b32_imm(2, literal_marker))
    elif use_valu_only:
        words.append(encode_v_add_f32(vdst=0, vsrc1=1, src0=0x102))
    words.append(rdna2.S_ENDPGM_SAVED_WORD if saved_terminator else rdna2.S_ENDPGM_WORD)
    return pack_words(*words)


def build_amdgpu_elf_header(mach_flag: int = 0x36) -> bytes:
    """Build a minimal 64-byte AMDGPU 64-bit LE ELF header (`e_machine=224`)."""
    hdr = bytearray(64)
    hdr[0:4] = b"\x7fELF"
    hdr[4] = 2  # ELFCLASS64
    hdr[5] = 1  # ELFDATA2LSB
    hdr[6] = 1  # EV_CURRENT
    struct.pack_into("<HHI", hdr, 16, 3, 224, 1)  # ET_DYN, EM_AMDGPU, EV_CURRENT
    struct.pack_into("<I", hdr, 48, mach_flag)  # e_flags (0x36 = gfx1030)
    return bytes(hdr)


def _run_cli(
    argv: list[str],
    *,
    env_path: Path | None = None,
    env: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    kwargs: dict = {}
    if env_path is not None:
        kwargs["env_path"] = env_path
    if env is not None:
        kwargs["env"] = env
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        code = rdna2.main(argv, **kwargs)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests: GFX10.3 instruction decoding & stage heuristics
# ---------------------------------------------------------------------------


class TestInstructionDecodingAndStages(unittest.TestCase):
    def test_decode_exp_targets_and_flags(self) -> None:
        for i in range(8):
            self.assertEqual(rdna2.decode_exp_target(i), f"mrt{i}")
        self.assertEqual(rdna2.decode_exp_target(8), "mrtz")
        self.assertEqual(rdna2.decode_exp_target(9), "null")
        self.assertEqual(rdna2.decode_exp_target(10), "unknown(10)")
        for i in range(5):
            self.assertEqual(rdna2.decode_exp_target(12 + i), f"pos{i}")
        for i in range(32):
            self.assertEqual(rdna2.decode_exp_target(32 + i), f"param{i}")

        w0, _w1 = encode_exp(12, en=0xF, compr=False, done=True, vm=False)
        exp = rdna2.decode_exp_instruction(w0)
        assert exp is not None
        self.assertEqual(exp.target_id, 12)
        self.assertEqual(exp.target, "pos0")
        self.assertTrue(exp.done)
        self.assertEqual(exp.en_mask, 0xF)
        self.assertFalse(exp.compr)
        self.assertFalse(exp.vm)

        w0_ps, _ = encode_exp(3, en=0x5, compr=True, done=True, vm=True)
        exp_ps = rdna2.decode_exp_instruction(w0_ps)
        assert exp_ps is not None
        self.assertEqual(exp_ps.target, "mrt3")
        self.assertTrue(exp_ps.done)
        self.assertTrue(exp_ps.compr)
        self.assertTrue(exp_ps.vm)
        self.assertEqual(exp_ps.en_mask, 0x5)

        # Non-EXP word returns None
        self.assertIsNone(rdna2.decode_exp_instruction(rdna2.S_ENDPGM_WORD))

    def test_synthetic_vertex_shader_detection(self) -> None:
        vs = build_synthetic_vs(extra_param=True)
        res = rdna2.scan_bytes(vs)
        self.assertEqual(len(res.shaders), 1)

        hit = res.shaders[0]
        self.assertEqual(hit.stage, "vertex")
        self.assertEqual(hit.terminator, "s_endpgm")
        self.assertEqual(hit.offset, 0)
        self.assertEqual(hit.byte_size, len(vs))
        self.assertEqual(hit.dword_count, len(vs) // 4)
        self.assertEqual(hit.exp_count, 2)
        self.assertEqual(hit.export_targets, ("param0", "pos0"))
        self.assertTrue(hit.has_done_export)
        self.assertEqual(hit.fnv1a_64, rdna2.fnv1a_64(vs))
        self.assertEqual(hit.hash_hex, rdna2.format_hash_64(rdna2.fnv1a_64(vs)))

    def test_synthetic_pixel_shader_detection(self) -> None:
        ps = build_synthetic_ps(include_mimg=True)
        res = rdna2.scan_bytes(ps)
        self.assertEqual(len(res.shaders), 1)

        hit = res.shaders[0]
        self.assertEqual(hit.stage, "pixel")
        self.assertEqual(hit.terminator, "s_endpgm")
        self.assertEqual(hit.vintrp_count, 1)
        self.assertEqual(hit.mimg_count, 1)
        self.assertEqual(hit.exp_count, 1)
        self.assertEqual(hit.export_targets, ("mrt0",))
        self.assertTrue(hit.has_done_export)
        self.assertEqual(hit.exports[0].target, "mrt0")
        self.assertTrue(hit.exports[0].done)
        self.assertEqual(hit.fnv1a_64, rdna2.fnv1a_64(ps))

        # Pixel shader with only v_interp (no exp) still infers pixel
        vinterp_only = pack_words(encode_vintrp(), rdna2.S_ENDPGM_WORD)
        res_vi = rdna2.scan_bytes(vinterp_only)
        self.assertEqual(len(res_vi.shaders), 1)
        self.assertEqual(res_vi.shaders[0].stage, "pixel")

        # Pixel shader with mrtz only
        mrtz_only = pack_words(*encode_exp(8, done=True), rdna2.S_ENDPGM_WORD)
        res_mrtz = rdna2.scan_bytes(mrtz_only)
        self.assertEqual(len(res_mrtz.shaders), 1)
        self.assertEqual(res_mrtz.shaders[0].stage, "pixel")
        self.assertEqual(res_mrtz.shaders[0].export_targets, ("mrtz",))

    def test_synthetic_compute_shader_and_saved_terminator(self) -> None:
        cs_smem = build_synthetic_cs(saved_terminator=False, use_valu_only=False)
        res = rdna2.scan_bytes(cs_smem)
        self.assertEqual(len(res.shaders), 1)
        hit = res.shaders[0]
        self.assertEqual(hit.stage, "compute")
        self.assertEqual(hit.terminator, "s_endpgm")
        self.assertEqual(hit.smem_count, 1)
        self.assertEqual(hit.exp_count, 0)
        self.assertEqual(hit.fnv1a_64, rdna2.fnv1a_64(cs_smem))

        cs_valu = build_synthetic_cs(saved_terminator=True, use_valu_only=True)
        res_saved = rdna2.scan_bytes(cs_valu)
        self.assertEqual(len(res_saved.shaders), 1)
        hit_saved = res_saved.shaders[0]
        self.assertEqual(hit_saved.stage, "compute")
        self.assertEqual(hit_saved.terminator, "s_endpgm_saved")
        self.assertEqual(hit_saved.valu_count, 1)
        self.assertEqual(hit_saved.exp_count, 0)

    def test_unknown_stage_heuristics(self) -> None:
        # s_nop + s_endpgm (no exp, no smem, no valu) -> unknown
        nop_blob = pack_words(encode_s_nop(0), encode_s_nop(1), rdna2.S_ENDPGM_WORD)
        res_nop = rdna2.scan_bytes(nop_blob)
        self.assertEqual(len(res_nop.shaders), 1)
        self.assertEqual(res_nop.shaders[0].stage, "unknown")

        # exp null + s_endpgm -> unknown
        null_exp = pack_words(*encode_exp(9, done=True), rdna2.S_ENDPGM_WORD)
        res_null = rdna2.scan_bytes(null_exp)
        self.assertEqual(len(res_null.shaders), 1)
        self.assertEqual(res_null.shaders[0].stage, "unknown")
        self.assertEqual(res_null.shaders[0].export_targets, ("null",))


# ---------------------------------------------------------------------------
# Unit tests: Multi-shader buffers, streaming, limits, and edge cases
# ---------------------------------------------------------------------------


class TestMultiShaderStreamingAndEdgeCases(unittest.TestCase):
    def test_multi_shader_buffer_and_pipeline_records(self) -> None:
        vs = build_synthetic_vs(extra_param=True)
        ps = build_synthetic_ps(include_mimg=True)
        cs = build_synthetic_cs(saved_terminator=True)
        code_end_pad = pack_words(rdna2.S_CODE_END_WORD, rdna2.S_CODE_END_WORD)

        # Wrap PS inside a synthetic Xbox Scarlett ACCC chunk
        accc_chunk = b"ACCC" + struct.pack("<I", len(ps)) + ps
        # Prepend a synthetic AMDGPU ELF header (gfx1030) before VS
        elf_hdr = build_amdgpu_elf_header(0x36)

        container = (
            elf_hdr
            + (b"\x00" * 16)
            + vs
            + (b"\x00" * 24)
            + accc_chunk
            + code_end_pad
            + cs
            + code_end_pad
            + vs  # duplicate VS to verify unique hash counting
        )

        res = rdna2.scan_bytes(container)
        self.assertEqual(len(res.shaders), 4)
        self.assertEqual(
            [s.stage for s in res.shaders],
            ["vertex", "pixel", "compute", "vertex"],
        )
        self.assertEqual(res.shaders[0].fnv1a_64, res.shaders[3].fnv1a_64)
        self.assertEqual(res.stage_counts, {"compute": 1, "pixel": 1, "vertex": 2})
        self.assertEqual(
            res.export_target_counts,
            {"mrt0": 1, "param0": 2, "pos0": 2},
        )
        self.assertEqual(
            [(r.kind, r.detail) for r in res.pipeline_records],
            [("AMDGPU_ELF", "gfx1030"), ("ACCC", "scarlett_accc")],
        )

    def test_streaming_across_chunk_boundaries(self) -> None:
        vs = build_synthetic_vs()  # 12 bytes
        ps = build_synthetic_ps()  # 24 bytes
        cs = build_synthetic_cs()  # 12 bytes

        buf = bytearray(b"\x00" * 512)
        # Place VS so its `s_endpgm` straddles chunk boundary 128 (offset 118..130)
        buf[118 : 118 + len(vs)] = vs
        # Place PS so its body straddles chunk boundary 256 (offset 244..268)
        buf[244 : 244 + len(ps)] = ps
        # Place CS so its `s_endpgm` straddles chunk boundary 384 (offset 374..386)
        buf[374 : 374 + len(cs)] = cs

        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "straddle_rdna2.bin"
            fpath.write_bytes(buf)

            res = rdna2.scan_file(fpath, chunk_size=128, overlap=64)
            self.assertFalse(res.truncated)
            self.assertEqual(res.bytes_scanned, len(buf))
            self.assertEqual(len(res.shaders), 3)
            self.assertEqual([s.stage for s in res.shaders], ["vertex", "pixel", "compute"])
            self.assertEqual([s.offset for s in res.shaders], [118, 244, 374])

    def test_max_bytes_max_hits_and_min_dwords_controls(self) -> None:
        vs = build_synthetic_vs()  # 3 dwords
        ps = build_synthetic_ps()  # 6 dwords
        cs = build_synthetic_cs()  # 3 dwords
        payload = vs + (b"\x00" * 64) + ps + (b"\x00" * 64) + cs

        # max_hits=2 stops after VS and PS
        capped_hits = rdna2.scan_bytes(payload, max_hits=2)
        self.assertTrue(capped_hits.hit_limit_reached)
        self.assertEqual([s.stage for s in capped_hits.shaders], ["vertex", "pixel"])

        # min_dwords=5 keeps only PS (6 dwords), filtering 3-dword VS & CS
        filtered = rdna2.scan_bytes(payload, min_dwords=5)
        self.assertEqual([s.stage for s in filtered.shaders], ["pixel"])

        # max_bytes=32 truncates before PS and CS
        truncated = rdna2.scan_bytes(payload, max_bytes=32)
        self.assertTrue(truncated.truncated)
        self.assertEqual(truncated.bytes_scanned, 32)
        self.assertEqual([s.stage for s in truncated.shaders], ["vertex"])

    def test_empty_truncated_unaligned_zero_and_noise_buffers(self) -> None:
        # Empty buffer
        res_empty = rdna2.scan_bytes(b"")
        self.assertEqual(res_empty.file_size, 0)
        self.assertEqual(res_empty.shaders, ())
        self.assertEqual(res_empty.unaligned_tail_bytes, 0)

        # Truncated (< 4 bytes) and unaligned non-shader bytes
        for size in (1, 2, 3, 7, 15):
            res_short = rdna2.scan_bytes(b"\x11" * size)
            self.assertEqual(res_short.shaders, ())
            self.assertEqual(res_short.unaligned_tail_bytes, size % 4)

        # Valid VS followed by 3 unaligned trailing bytes
        vs = build_synthetic_vs()
        res_unaligned = rdna2.scan_bytes(vs + b"\xaa\xbb\xcc")
        self.assertEqual(len(res_unaligned.shaders), 1)
        self.assertEqual(res_unaligned.shaders[0].stage, "vertex")
        self.assertEqual(res_unaligned.unaligned_tail_bytes, 3)

        # Zero-filled prefix + lone s_endpgm: filtered by default min_dwords=2;
        # with min_dwords=1 it is detected as unknown (never compute!).
        lone = (b"\x00" * 32) + pack_words(rdna2.S_ENDPGM_WORD)
        self.assertEqual(rdna2.scan_bytes(lone).shaders, ())
        res_lone = rdna2.scan_bytes(lone, min_dwords=1)
        self.assertEqual(len(res_lone.shaders), 1)
        self.assertEqual(res_lone.shaders[0].dword_count, 1)
        self.assertEqual(res_lone.shaders[0].stage, "unknown")

        # Deterministic random noise (256 KiB): zero false positives
        rng = random.Random(0x67A6_4D42)
        noise = rng.randbytes(256 * 1024)
        res_noise = rdna2.scan_bytes(noise)
        self.assertEqual(res_noise.shaders, ())
        self.assertEqual(res_noise.pipeline_records, ())

        # Empty & truncated files on disk
        with tempfile.TemporaryDirectory() as tmp:
            empty_file = Path(tmp) / "empty.bin"
            empty_file.write_bytes(b"")
            trunc_file = Path(tmp) / "trunc.bin"
            trunc_file.write_bytes(b"\x00\x00\x81")  # 3 bytes of truncated s_endpgm
            self.assertEqual(rdna2.scan_file(empty_file).shaders, ())
            self.assertEqual(rdna2.scan_file(trunc_file).shaders, ())


# ---------------------------------------------------------------------------
# Unit tests: CLI, JSON schema, histograms, and strict sanitization
# ---------------------------------------------------------------------------


class TestCliJsonAndStrictSanitization(unittest.TestCase):
    def test_sanitize_helpers_strip_windows_unc_posix_and_username(self) -> None:
        self.assertEqual(
            rdna2.sanitize_source_label(r"C:\Users\SecretAlice\XboxGames\GTA6\shaders.bin"),
            "shaders.bin",
        )
        self.assertEqual(
            rdna2.sanitize_source_label("/home/secret_bob/xbox/dumps/ps_blob.bin"),
            "ps_blob.bin",
        )
        self.assertEqual(
            rdna2.sanitize_source_label("/Users/secret_carol/Library/Caches/vs.bin"),
            "vs.bin",
        )
        self.assertEqual(
            rdna2.sanitize_source_label(r"\\corp-nas\private\share\dump.rpf"),
            "dump.rpf",
        )
        self.assertEqual(rdna2.sanitize_source_label("C:\\"), "<unknown>")

        sample_err = (
            r"cannot open C:\Users\SecretAlice\game\a.bin, "
            r"\\server\share\b.bin, /home/secret_bob/c.bin, /Users/secret_carol/d.bin"
        )
        scrubbed = rdna2._scrub_sensitive_text(sample_err)
        for leaked in (
            "C:\\",
            "SecretAlice",
            "\\\\server",
            "/home/",
            "secret_bob",
            "/Users/",
            "secret_carol",
        ):
            self.assertNotIn(leaked, scrubbed)
        self.assertIn("<path>", scrubbed)

        with mock.patch.dict(os.environ, {"USERNAME": "SecretHostUser99"}):
            self.assertNotIn(
                "SecretHostUser99",
                rdna2.sanitize_source_label("SecretHostUser99_shaders.bin"),
            )

    def test_cli_json_and_markdown_output_with_zero_path_or_raw_byte_leaks(self) -> None:
        # Embed distinctive 32-bit literal markers inside synthetic shaders to verify
        # that ZERO raw shader bytes or literal payloads ever appear in output!
        marker_vs = 0xDEADBEEF
        marker_ps = 0xCAFEBABE
        marker_cs = 0x1337C0DE

        vs = build_synthetic_vs(extra_param=True, literal_marker=marker_vs)
        ps = build_synthetic_ps(include_mimg=True, literal_marker=marker_ps)
        cs = build_synthetic_cs(saved_terminator=True, literal_marker=marker_cs)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "MyPrivateXboxDump"
            sub = root / "shaders" / "rdna2"
            sub.mkdir(parents=True)

            (sub / "a_vertex.bin").write_bytes(vs)
            (sub / "b_pixel.bin").write_bytes(ps)
            (sub / "c_compute.bin").write_bytes(cs)
            (root / "empty.bin").write_bytes(b"")

            code, raw_json, stderr = _run_cli(
                [str(root), "--json", "--min-dwords", "2", "--max-bytes", "1M"]
            )
            self.assertEqual(code, 0, stderr)
            data = json.loads(raw_json)

            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["target"], "MyPrivateXboxDump")
            self.assertEqual(data["target_kind"], "directory")
            self.assertEqual(data["min_dwords"], 2)
            self.assertEqual(data["max_bytes"], 1024 * 1024)

            summary = data["summary"]
            self.assertEqual(summary["files_scanned"], 4)
            self.assertEqual(summary["files_with_shaders"], 3)
            self.assertEqual(summary["files_errored"], 0)
            self.assertEqual(summary["total_shaders"], 3)
            self.assertEqual(summary["unique_shader_hashes"], 3)
            self.assertFalse(summary["hit_limit_reached"])
            self.assertEqual(
                summary["stage_counts"],
                {"compute": 1, "pixel": 1, "vertex": 1},
            )
            self.assertEqual(
                summary["terminator_counts"],
                {"s_endpgm": 2, "s_endpgm_saved": 1},
            )
            self.assertEqual(
                summary["export_target_histogram"],
                {"mrt0": 1, "param0": 1, "pos0": 1},
            )
            self.assertEqual(summary["size_histogram"], {"0..64B": 3})
            self.assertEqual(summary["instruction_totals"]["exp"], 3)
            self.assertEqual(summary["instruction_totals"]["smem"], 1)
            self.assertEqual(summary["instruction_totals"]["vintrp"], 1)
            self.assertEqual(summary["instruction_totals"]["mimg"], 1)

            reported_paths = [f["path"] for f in data["files"]]
            self.assertEqual(
                reported_paths,
                [
                    "shaders/rdna2/a_vertex.bin",
                    "shaders/rdna2/b_pixel.bin",
                    "shaders/rdna2/c_compute.bin",
                ],
            )

            # Also test Markdown output mode
            code_md, md_text, stderr_md = _run_cli([str(root)])
            self.assertEqual(code_md, 0, stderr_md)
            self.assertIn("# RDNA2 / GFX10.3 Shader Inspection Report: `MyPrivateXboxDump`", md_text)
            self.assertIn("## Export Target Histogram", md_text)
            self.assertIn("## Shader Size Histogram", md_text)

            # Strict sanitization assertions on BOTH JSON and Markdown outputs
            resolved_tmp = str(Path(tmp).resolve())
            for output in (raw_json, md_text):
                lowered = output.lower()
                # Zero host absolute paths / drive letters / UNC / POSIX home dirs
                self.assertNotIn(resolved_tmp, output)
                self.assertNotIn(resolved_tmp.replace("\\", "/"), output)
                self.assertNotIn(":\\", output)
                self.assertNotIn(":/", output)
                self.assertNotIn("/home/", output)
                self.assertNotIn("/users/", lowered)
                for env_var in ("USERNAME", "USER", "LOGNAME"):
                    user = os.environ.get(env_var, "").strip()
                    if len(user) >= 3:
                        self.assertNotIn(user.lower(), lowered)

                # Zero raw shader bytes, literal constants, or raw opcode words!
                for forbidden_hex in (
                    "deadbeef",
                    "efbeadde",
                    "cafebabe",
                    "bebafeca",
                    "1337c0de",
                    "dec03713",
                    "bf810000",
                    "000081bf",
                    "bf9b0000",
                    "00009bbf",
                    "f80008cf",
                    "f8001c0f",
                ):
                    self.assertNotIn(forbidden_hex, lowered)

    def test_cli_max_hits_and_missing_path_error_sanitization(self) -> None:
        vs = build_synthetic_vs()
        ps = build_synthetic_ps()
        cs = build_synthetic_cs()
        with tempfile.TemporaryDirectory() as tmp:
            blob_path = Path(tmp) / "bundle.bin"
            blob_path.write_bytes(vs + (b"\x00" * 16) + ps + (b"\x00" * 16) + cs)

            code, out, err = _run_cli([str(blob_path), "--json", "--max-hits", "2"])
            self.assertEqual(code, 0, err)
            data = json.loads(out)
            self.assertEqual(data["summary"]["total_shaders"], 2)
            self.assertTrue(data["summary"]["hit_limit_reached"])

        # Missing Windows path and missing POSIX path must NEVER leak into stderr
        for missing_path, secret_token in (
            (r"C:\Users\SecretWinUser\PrivateGames\missing_shader.bin", "SecretWinUser"),
            ("/home/SecretLinuxUser/private_games/missing_shader.bin", "SecretLinuxUser"),
        ):
            code_err, _out_err, err_msg = _run_cli([missing_path])
            self.assertEqual(code_err, 1)
            self.assertIn("missing_shader.bin", err_msg)
            self.assertNotIn(secret_token, err_msg)
            self.assertNotIn(":\\", err_msg)
            self.assertNotIn("/home/", err_msg)

    def test_cli_from_env_redacts_game_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = Path(tmp) / "repo"
            fake_repo.mkdir()
            game_dir = Path(tmp) / "SuperSecretInstallDir"
            game_dir.mkdir()
            (game_dir / "shader.bin").write_bytes(build_synthetic_cs())
            env_file = fake_repo / ".env"
            env_file.write_text(f'GTA6_GAME_DIR="{game_dir}"\n', encoding="utf-8")

            code, out, err = _run_cli(["--from-env", "--json"], env_path=env_file, env={})
            self.assertEqual(code, 0, err)
            data = json.loads(out)
            self.assertEqual(data["target"], "<GTA6_GAME_DIR>")
            self.assertEqual(data["summary"]["stage_counts"], {"compute": 1})
            self.assertNotIn("SuperSecretInstallDir", out)
            self.assertNotIn(str(game_dir), out)


class TestWholeFileStandaloneIdentityAndManifest(unittest.TestCase):
    """Track 4 / S20: whole-file FNV-1a 64 identity hashing & pipeline dump manifest hints."""

    def test_standalone_file_trailing_padding_hashes_whole_file(self) -> None:
        trimmed_vs = build_synthetic_vs(extra_param=True)
        trailing_pad = pack_words(
            rdna2.S_CODE_END_WORD,
            rdna2.S_CODE_END_WORD,
            0x00000000,
            0x00000000,
            rdna2.S_CODE_END_WORD,
            0x00000000,
        )
        full_vs = trimmed_vs + trailing_pad

        expected_whole_hash = rdna2.fnv1a_64(full_vs)
        expected_trimmed_hash = rdna2.fnv1a_64(trimmed_vs)
        self.assertNotEqual(expected_whole_hash, expected_trimmed_hash)

        with tempfile.TemporaryDirectory() as tmp:
            native_dir = Path(tmp) / "native"
            native_dir.mkdir()
            dump_file = native_dir / f"{rdna2.format_hash_64(expected_whole_hash)}.bin"
            dump_file.write_bytes(full_vs)

            plain_file = Path(tmp) / "per_frame_vs.rdna2"
            plain_file.write_bytes(full_vs)

            # 1. native/0x<hash>.bin hashes the ENTIRE file and validates filename hint
            res_dump = rdna2.scan_file(dump_file)
            self.assertEqual(res_dump.hash_scope, "whole_file")
            self.assertEqual(res_dump.file_hash_fnv1a64, expected_whole_hash)
            self.assertEqual(len(res_dump.shaders), 1)
            hit = res_dump.shaders[0]
            self.assertEqual(hit.stage, "vertex")
            self.assertEqual(hit.hash_scope, "whole_file")
            self.assertEqual(hit.fnv1a_64, expected_whole_hash)
            self.assertEqual(hit.code_hash_fnv1a64, expected_whole_hash)
            self.assertEqual(hit.hash_hex, rdna2.format_hash_64(expected_whole_hash))
            self.assertEqual(hit.trimmed_hash_fnv1a64, expected_trimmed_hash)
            self.assertEqual(hit.byte_size, len(full_vs))
            self.assertEqual(hit.trimmed_byte_size, len(trimmed_vs))
            self.assertEqual(hit.terminator_count, 1)
            self.assertEqual(hit.filename_hash_hint, expected_whole_hash)
            self.assertTrue(hit.filename_hash_matches)

            # 2. Unhinted standalone .rdna2 file also hashes the ENTIRE file
            res_plain = rdna2.scan_file(plain_file)
            self.assertEqual(res_plain.hash_scope, "whole_file")
            self.assertEqual(len(res_plain.shaders), 1)
            self.assertEqual(res_plain.shaders[0].fnv1a_64, expected_whole_hash)
            self.assertEqual(res_plain.shaders[0].trimmed_hash_fnv1a64, expected_trimmed_hash)
            self.assertIsNone(res_plain.shaders[0].filename_hash_hint)

            # 3. Embedded mode (--embedded / whole_file=False) preserves trimmed region hash
            res_emb = rdna2.scan_file(dump_file, whole_file=False)
            self.assertEqual(res_emb.hash_scope, "region")
            self.assertEqual(len(res_emb.shaders), 1)
            self.assertEqual(res_emb.shaders[0].fnv1a_64, expected_trimmed_hash)
            self.assertEqual(res_emb.shaders[0].byte_size, len(trimmed_vs))

            code_emb, out_emb, err_emb = _run_cli([str(dump_file), "--json", "--embedded"])
            self.assertEqual(code_emb, 0, err_emb)
            emb_json = json.loads(out_emb)
            self.assertEqual(
                emb_json["files"][0]["shaders"][0]["hash"],
                rdna2.format_hash_64(expected_trimmed_hash),
            )
            self.assertEqual(emb_json["files"][0]["shaders"][0]["hash_scope"], "region")

    def test_standalone_file_with_two_endpgm_terminators(self) -> None:
        # Early-out branch (`s_nop` + `s_endpgm`) followed immediately by main pixel path
        # (`v_interp` + `exp mrt0 done` + `s_endpgm`) and trailing padding.
        first_branch = pack_words(encode_s_nop(0), encode_s_nop(1), rdna2.S_ENDPGM_WORD)
        second_branch = build_synthetic_ps(include_mimg=True)
        trailing_pad = pack_words(rdna2.S_CODE_END_WORD, rdna2.S_CODE_END_WORD, 0, 0)
        full_ps = first_branch + second_branch + trailing_pad

        expected_whole_hash = rdna2.fnv1a_64(full_ps)
        expected_trimmed_hash = rdna2.fnv1a_64(first_branch)
        self.assertNotEqual(expected_whole_hash, expected_trimmed_hash)

        with tempfile.TemporaryDirectory() as tmp:
            named_file = Path(tmp) / f"{rdna2.format_hash_64(expected_whole_hash)}.bin"
            named_file.write_bytes(full_ps)
            plain_file = Path(tmp) / "two_endpgm_ps.bin"
            plain_file.write_bytes(full_ps)

            for target in (named_file, plain_file):
                with self.subTest(target=target.name):
                    res = rdna2.scan_file(target)
                    self.assertEqual(res.hash_scope, "whole_file")
                    self.assertEqual(len(res.shaders), 1)
                    hit = res.shaders[0]
                    self.assertEqual(hit.stage, "pixel")
                    self.assertEqual(hit.terminator_count, 2)
                    self.assertEqual(hit.fnv1a_64, expected_whole_hash)
                    self.assertEqual(hit.code_hash_fnv1a64, expected_whole_hash)
                    self.assertEqual(hit.trimmed_hash_fnv1a64, expected_trimmed_hash)
                    self.assertEqual(hit.byte_size, len(full_ps))
                    self.assertEqual(hit.trimmed_byte_size, len(first_branch))
                    self.assertEqual(hit.vintrp_count, 1)
                    self.assertEqual(hit.mimg_count, 1)
                    self.assertEqual(hit.exp_count, 1)
                    self.assertEqual(hit.export_targets, ("mrt0",))

    def test_pipeline_dump_manifest_v1_identity_and_stage_hints(self) -> None:
        vs_bytes = build_synthetic_vs(extra_param=True) + pack_words(rdna2.S_CODE_END_WORD, 0, 0)
        # Pixel stage with internal padding before a second s_endpgm + trailing padding
        ps_part1 = pack_words(encode_vintrp(), rdna2.S_ENDPGM_WORD)
        ps_part2 = pack_words(*encode_exp(0, done=True), rdna2.S_ENDPGM_WORD)
        ps_bytes = ps_part1 + pack_words(rdna2.S_CODE_END_WORD, rdna2.S_CODE_END_WORD) + ps_part2 + pack_words(0, 0)
        # Stage whose ISA alone is "unknown" (`s_nop` + `s_endpgm`) rescued via manifest
        unk_bytes = pack_words(encode_s_nop(0), encode_s_nop(1), rdna2.S_ENDPGM_WORD, 0, 0)

        vs_hash = rdna2.fnv1a_64(vs_bytes)
        ps_hash = rdna2.fnv1a_64(ps_bytes)
        unk_hash = rdna2.fnv1a_64(unk_bytes)
        vs_stage_id = 0x1111222233334444
        ps_stage_id = 0x5555666677778888
        wrong_hint_hash = 0xDEADBEEFCAFE0001

        with tempfile.TemporaryDirectory() as tmp:
            dump_dir = Path(tmp) / "pipeline_dump"
            native_dir = dump_dir / "native"
            dxil_dir = dump_dir / "dxil"
            native_dir.mkdir(parents=True)
            dxil_dir.mkdir(parents=True)

            vs_rel = f"native/{rdna2.format_hash_64(vs_hash)}.bin"
            ps_rel = f"native/{rdna2.format_hash_64(ps_hash)}.bin"
            unk_rel = f"native/{rdna2.format_hash_64(wrong_hint_hash)}.bin"

            (dump_dir / vs_rel).write_bytes(vs_bytes)
            (dump_dir / ps_rel).write_bytes(ps_bytes)
            (dump_dir / unk_rel).write_bytes(unk_bytes)
            (native_dir / f"{rdna2.format_hash_64(vs_hash)}.meta").write_bytes(b"\x01\x02\x03\x04")
            (dxil_dir / "0x0000000000000001.dxbc").write_bytes(b"DXBC" + b"\x00" * 28)

            manifest = {
                "schema": "playgta6-pipeline-dump",
                "schema_version": 1,
                "native_stages": [
                    {
                        "code_identity": rdna2.format_hash_64(vs_hash),
                        "identity": rdna2.format_hash_64(vs_stage_id),
                        "stage": "vertex",
                        "layout": "A",
                        "code_bytes": len(vs_bytes),
                        "metadata_bytes": 4,
                        "code_file": vs_rel,
                        "metadata_file": f"native/{rdna2.format_hash_64(vs_hash)}.meta",
                    },
                    {
                        "code_identity": rdna2.format_hash_64(ps_hash),
                        "identity": rdna2.format_hash_64(ps_stage_id),
                        "stage": "pixel",
                        "layout": "A",
                        "code_bytes": len(ps_bytes),
                        "metadata_bytes": 0,
                        "code_file": ps_rel,
                        "metadata_file": f"native/{rdna2.format_hash_64(ps_hash)}.meta",
                    },
                    {
                        "code_identity": rdna2.format_hash_64(wrong_hint_hash),
                        "identity": rdna2.format_hash_64(wrong_hint_hash),
                        "stage": "pixel",
                        "layout": "B",
                        "code_bytes": len(unk_bytes),
                        "metadata_bytes": 0,
                        "code_file": unk_rel,
                        "metadata_file": f"native/{rdna2.format_hash_64(wrong_hint_hash)}.meta",
                    },
                ],
                "dxil": [],
                "root_signatures": [],
                "pipelines": [],
            }
            (dump_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            # Scan entire dump directory via CLI --json
            code, raw_json, err = _run_cli([str(dump_dir), "--json"])
            self.assertEqual(code, 0, err)
            data = json.loads(raw_json)
            self.assertEqual(data["summary"]["total_shaders"], 3)
            self.assertEqual(data["summary"]["stage_counts"], {"pixel": 2, "vertex": 1})

            by_path = {f["path"]: f for f in data["files"]}
            vs_hit = by_path[vs_rel]["shaders"][0]
            self.assertEqual(vs_hit["hash"], rdna2.format_hash_64(vs_hash))
            self.assertEqual(vs_hit["code_hash_fnv1a64"], rdna2.format_hash_64(vs_hash))
            self.assertEqual(vs_hit["manifest_code_identity"], rdna2.format_hash_64(vs_hash))
            self.assertEqual(vs_hit["manifest_identity"], rdna2.format_hash_64(vs_stage_id))
            self.assertTrue(vs_hit["manifest_identity_matches"])
            self.assertTrue(vs_hit["filename_hash_matches"])

            ps_hit = by_path[ps_rel]["shaders"][0]
            self.assertEqual(ps_hit["hash"], rdna2.format_hash_64(ps_hash))
            self.assertEqual(ps_hit["stage"], "pixel")
            self.assertEqual(ps_hit["terminator_count"], 2)
            self.assertTrue(ps_hit["manifest_identity_matches"])

            # Mismatched filename/manifest still hashes actual full bytes and flags mismatch!
            unk_hit = by_path[unk_rel]["shaders"][0]
            self.assertEqual(unk_hit["hash"], rdna2.format_hash_64(unk_hash))
            self.assertEqual(unk_hit["stage"], "pixel")  # rescued from "unknown" via manifest
            self.assertEqual(unk_hit["manifest_stage"], "pixel")
            self.assertFalse(unk_hit["filename_hash_matches"])
            self.assertFalse(unk_hit["manifest_identity_matches"])

            # Single-file scan inside native/ automatically discovers ../manifest.json
            single_res = rdna2.scan_file(dump_dir / ps_rel)
            self.assertEqual(len(single_res.shaders), 1)
            self.assertEqual(single_res.shaders[0].manifest_code_identity, ps_hash)
            self.assertTrue(single_res.shaders[0].manifest_identity_matches)

            # Strict sanitization: zero host paths in JSON
            self.assertNotIn(str(Path(tmp).resolve()), raw_json)

    def test_embedded_multi_shader_bundle_preserves_region_hashing(self) -> None:
        vs = build_synthetic_vs()
        ps = build_synthetic_ps()
        cs = build_synthetic_cs()
        bundle = vs + (b"\x00" * 16) + ps + (b"\x00" * 16) + cs

        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "embedded_bundle.bin"
            fpath.write_bytes(bundle)
            res = rdna2.scan_file(fpath)
            self.assertEqual(res.hash_scope, "region")
            self.assertEqual(len(res.shaders), 3)
            self.assertEqual([s.stage for s in res.shaders], ["vertex", "pixel", "compute"])
            self.assertEqual(
                [s.fnv1a_64 for s in res.shaders],
                [rdna2.fnv1a_64(vs), rdna2.fnv1a_64(ps), rdna2.fnv1a_64(cs)],
            )


# ---------------------------------------------------------------------------
# T92: Canonical shader summary sorting, zero-division guards & JSON sanitization
# ---------------------------------------------------------------------------


class TestT92StageSummarySortingAndSanitization(unittest.TestCase):
    def test_canonical_shader_summary_sorting_and_permutation_invariance(self) -> None:
        vs_a = build_synthetic_vs(extra_param=False, literal_marker=0x10101010)
        vs_b = build_synthetic_vs(extra_param=True, literal_marker=0x20202020)
        ps = build_synthetic_ps(include_mimg=True)
        cs = build_synthetic_cs(saved_terminator=True)
        unk = pack_words(encode_s_nop(0), encode_s_nop(1), rdna2.S_ENDPGM_WORD)

        # Build FileInspectResults in scrambled order (unknown, compute, pixel, vertex)
        # including duplicate VS across two files and out-of-order pipeline records (ACCC before ELF).
        res_unk = rdna2.scan_bytes(unk, source_label="z_unknown.bin")
        res_cs = rdna2.scan_bytes(cs, source_label="y_compute.bin")
        res_ps = rdna2.scan_bytes(ps, source_label="m_pixel.bin")
        res_vs_a = rdna2.scan_bytes(vs_a, source_label="b_vertex_a.bin")
        res_vs_b = rdna2.scan_bytes(vs_b, source_label="a_vertex_b.bin")
        res_vs_dup = rdna2.scan_bytes(vs_a, source_label="c_vertex_dup.bin")

        # Also test pipeline record offset sorting when ACCC precedes AMDGPU_ELF in a buffer.
        accc_first = b"ACCC" + struct.pack("<I", len(ps)) + ps
        elf_second = build_amdgpu_elf_header(0x36)
        res_records = rdna2.scan_bytes(
            accc_first + (b"\x00" * 16) + elf_second,
            source_label="d_records.bin",
        )
        self.assertEqual(
            [r.kind for r in res_records.pipeline_records],
            ["ACCC", "AMDGPU_ELF"],
        )
        self.assertLess(
            res_records.pipeline_records[0].offset,
            res_records.pipeline_records[1].offset,
        )

        order_1 = [res_unk, res_cs, res_ps, res_vs_a, res_vs_b, res_vs_dup]
        order_2 = [res_vs_dup, res_ps, res_vs_b, res_unk, res_vs_a, res_cs]

        report_1 = rdna2._build_report(
            target_label="canonical_test",
            target_kind="directory",
            max_bytes=None,
            max_hits=None,
            min_dwords=2,
            all_results=order_1,
            hit_limit_reached=False,
            include_empty=False,
        )
        report_2 = rdna2._build_report(
            target_label="canonical_test",
            target_kind="directory",
            max_bytes=None,
            max_hits=None,
            min_dwords=2,
            all_results=order_2,
            hit_limit_reached=False,
            include_empty=False,
        )

        # Byte-for-byte identical JSON regardless of input order.
        json_1 = rdna2.render_json(report_1)
        json_2 = rdna2.render_json(report_2)
        self.assertEqual(json_1, json_2)

        payload = json.loads(json_1)
        summary = payload["summary"]

        # Canonical stage ordering: vertex -> pixel -> compute -> unknown.
        self.assertEqual(
            list(summary["stage_counts"].keys()),
            ["vertex", "pixel", "compute", "unknown"],
        )
        self.assertEqual(
            list(summary["stage_percentages"].keys()),
            ["vertex", "pixel", "compute", "unknown"],
        )
        self.assertEqual(
            [row["stage"] for row in summary["stage_breakdown"]],
            ["vertex", "pixel", "compute", "unknown"],
        )

        # Stage counts and percentages (3 vertex, 1 pixel, 1 compute, 1 unknown = 6 total).
        self.assertEqual(summary["total_shaders"], 6)
        self.assertEqual(summary["unique_shader_hashes"], 5)
        self.assertEqual(summary["stage_percentages"]["vertex"], 50.0)
        self.assertEqual(summary["stage_percentages"]["pixel"], 16.67)
        self.assertEqual(summary["stage_percentages"]["compute"], 16.67)
        self.assertEqual(summary["stage_percentages"]["unknown"], 16.67)

        # Canonical shader_summary sorted by (canonical stage rank, hash_hex).
        shader_summary = summary["shader_summary"]
        self.assertEqual(len(shader_summary), 5)
        self.assertEqual(
            [row["stage"] for row in shader_summary],
            ["vertex", "vertex", "pixel", "compute", "unknown"],
        )
        vs_hashes = [row["hash"] for row in shader_summary if row["stage"] == "vertex"]
        self.assertEqual(vs_hashes, sorted(vs_hashes))

        # Duplicate vertex shader aggregated with occurrence count & sorted paths.
        vs_a_hash = rdna2.format_hash_64(rdna2.fnv1a_64(vs_a))
        vs_a_row = next(row for row in shader_summary if row["hash"] == vs_a_hash)
        self.assertEqual(vs_a_row["occurrences"], 2)
        self.assertEqual(vs_a_row["paths"], ["b_vertex_a.bin", "c_vertex_dup.bin"])

        # Files list deterministically sorted by path.
        self.assertEqual(
            [f["path"] for f in payload["files"]],
            [
                "a_vertex_b.bin",
                "b_vertex_a.bin",
                "c_vertex_dup.bin",
                "m_pixel.bin",
                "y_compute.bin",
                "z_unknown.bin",
            ],
        )

        # Markdown renders canonical stage order with share percentages.
        md = rdna2.render_markdown(report_1)
        self.assertIn("| Stage | Shaders | Share |", md)
        self.assertIn("| `vertex` | 3 | 50.00% |", md)
        self.assertLess(md.index("| `vertex` |"), md.index("| `pixel` |"))
        self.assertLess(md.index("| `pixel` |"), md.index("| `compute` |"))
        self.assertLess(md.index("| `compute` |"), md.index("| `unknown` |"))

    def test_zero_shader_zero_division_guards(self) -> None:
        # Direct helper calls with zero counts, explicit total=0, negative values, and bools.
        empty_breakdown = rdna2.compute_stage_breakdown({})
        self.assertEqual(
            empty_breakdown,
            [
                {"stage": "vertex", "count": 0, "percentage": 0.0},
                {"stage": "pixel", "count": 0, "percentage": 0.0},
                {"stage": "compute", "count": 0, "percentage": 0.0},
                {"stage": "unknown", "count": 0, "percentage": 0.0},
            ],
        )
        self.assertEqual(
            rdna2.compute_stage_percentages({}),
            {"vertex": 0.0, "pixel": 0.0, "compute": 0.0, "unknown": 0.0},
        )
        self.assertEqual(
            rdna2.compute_stage_percentages({"vertex": 5, "pixel": 2}, total_shaders=0),
            {"vertex": 0.0, "pixel": 0.0, "compute": 0.0, "unknown": 0.0},
        )
        self.assertEqual(
            rdna2.compute_stage_percentages({"vertex": -4, "pixel": True}),
            {"vertex": 0.0, "pixel": 0.0, "compute": 0.0, "unknown": 0.0},
        )

        # End-to-end CLI --json and Markdown on empty dir, empty file, and filtered-out shaders.
        with tempfile.TemporaryDirectory() as tmp:
            empty_dir = Path(tmp) / "empty_shaders"
            empty_dir.mkdir()
            empty_file = Path(tmp) / "zero.bin"
            empty_file.write_bytes(b"")
            filtered_file = Path(tmp) / "short_vs.bin"
            filtered_file.write_bytes(build_synthetic_vs())

            for argv in (
                [str(empty_dir), "--json"],
                [str(empty_file), "--json"],
                [str(filtered_file), "--json", "--min-dwords", "999"],
            ):
                code, raw_json, err = _run_cli(argv)
                self.assertEqual((code, err), (0, ""))
                payload = json.loads(raw_json)
                summary = payload["summary"]
                self.assertEqual(summary["total_shaders"], 0)
                self.assertEqual(summary["stage_counts"], {})
                self.assertEqual(
                    summary["stage_percentages"],
                    {"vertex": 0.0, "pixel": 0.0, "compute": 0.0, "unknown": 0.0},
                )
                self.assertEqual(summary["stage_breakdown"], empty_breakdown)
                self.assertEqual(summary["shader_summary"], [])

            code_md, md_out, err_md = _run_cli([str(empty_dir)])
            self.assertEqual((code_md, err_md), (0, ""))
            self.assertIn("No RDNA2 / GFX10.3 shaders detected.", md_out)

    def test_strict_host_path_and_username_sanitization_in_json_output(self) -> None:
        # Helper sanitization for user-home leaves, file://, MSYS, tilde, UNC, PathLike, and bytes.
        self.assertEqual(rdna2.sanitize_source_label(r"C:\Users\alice"), "<user>")
        self.assertEqual(rdna2.sanitize_source_label("/home/bob/"), "<user>")
        self.assertEqual(rdna2.sanitize_source_label("/Users/carol"), "<user>")
        self.assertEqual(rdna2.sanitize_source_label("/c/Users/dave"), "<user>")
        self.assertEqual(rdna2.sanitize_source_label("~eve"), "<user>")
        self.assertEqual(
            rdna2.sanitize_source_label("file:///C:/Users/frank/dumps/vs.bin"),
            "vs.bin",
        )

        scrubbed = rdna2.sanitize_json_value(
            {
                "win": r"C:\Users\alice\work\vs.bin",
                "posix": "/home/bob/work/ps.bin",
                "posix_lower": "/users/carol/work/cs.bin",
                "msys": "/c/Users/dave/work/shader.bin",
                "uri": "file:///C:/Users/eve/work/dump.bin",
                "tilde": "~frank/work/shader.bin",
                "unc": r"\\fileserver\share\grace\shader.bin",
                "unc_fs": "//fileserver/share/heidi/shader.bin",
                "path_obj": Path("/home/ivan/projects/playgta6web/shader.bin"),
                "raw_bytes": b"\xde\xad\xbe\xef",
            }
        )
        serialized = json.dumps(scrubbed)
        for leak in (
            "alice",
            "bob",
            "carol",
            "dave",
            "eve",
            "frank",
            "grace",
            "heidi",
            "ivan",
            "C:\\",
            "C:/",
            "/home/",
            "/Users/",
            "/users/",
            "file://",
            "fileserver",
            "deadbeef",
        ):
            self.assertNotIn(leak, serialized)
        self.assertEqual(scrubbed["path_obj"], "shader.bin")
        self.assertEqual(scrubbed["raw_bytes"], "<bytes:4>")

        # Directly constructed InspectReport with leaked host paths/usernames in every text field
        # MUST be strictly sanitized by to_dict() and render_json().
        username = "t92secretuser"
        vs = build_synthetic_vs()
        hit = rdna2.analyze_shader_region(vs, abs_offset=0, terminator_word=rdna2.S_ENDPGM_WORD)
        leaked_file = rdna2.FileInspectResult(
            path=rf"C:\Users\{username}\PrivateDump\vs_00.bin",
            file_size=len(vs),
            bytes_scanned=len(vs),
            truncated=False,
            shaders=(hit,),
            pipeline_records=(
                rdna2.PipelineRecordHit(
                    kind="AMDGPU_ELF",
                    offset=0,
                    detail=f"/home/{username}/sdk/gfx1030.elf",
                ),
            ),
            error=rf"read failed at \\nas-box\private\{username}\vs_00.bin",
        )
        leaked_report = rdna2.InspectReport(
            schema_version=rdna2.SCHEMA_VERSION,
            target_label=rf"C:\Users\{username}\PrivateDump",
            target_kind="directory",
            max_bytes=None,
            max_hits=None,
            min_dwords=2,
            files_scanned=1,
            files_with_shaders=1,
            files_errored=1,
            total_file_bytes=len(vs),
            total_bytes_scanned=len(vs),
            total_shaders=1,
            unique_shader_hashes=1,
            total_pipeline_records=1,
            hit_limit_reached=False,
            stage_counts={"vertex": 1},
            terminator_counts={"s_endpgm": 1},
            export_target_histogram={"pos0": 1},
            size_histogram={"0..64B": 1},
            instruction_totals={"exp": 1, "smem": 0, "vintrp": 0, "mimg": 0, "valu": 0, "salu": 0, "sopp": 1, "vmem": 0, "ds": 0, "flat": 0},
            pipeline_record_counts={"AMDGPU_ELF": 1},
            files=(leaked_file,),
        )

        with mock.patch.dict(
            os.environ,
            {
                "USERNAME": username,
                "USER": username,
                "HOME": f"/home/{username}",
                "USERPROFILE": rf"C:\Users\{username}",
            },
        ):
            rendered_json = rdna2.render_json(leaked_report)
            rendered_md = rdna2.render_markdown(leaked_report)

            # Also run CLI end-to-end with username inside directory & filename.
            with tempfile.TemporaryDirectory() as tmp:
                user_dir = Path(tmp) / "Users" / username
                sub = user_dir / f"{username}_shaders"
                sub.mkdir(parents=True)
                (sub / f"{username}_vs.bin").write_bytes(vs)
                cli_code, cli_json, cli_err = _run_cli([str(user_dir), "--json"])

        self.assertEqual((cli_code, cli_err), (0, ""))
        parsed_leaked = json.loads(rendered_json)
        self.assertEqual(parsed_leaked["target"], "PrivateDump")
        self.assertEqual(parsed_leaked["files"][0]["path"], "vs_00.bin")
        self.assertEqual(parsed_leaked["summary"]["shader_summary"][0]["paths"], ["vs_00.bin"])
        self.assertEqual(json.loads(cli_json)["target"], "<user>")

        for output in (rendered_json, rendered_md, cli_json):
            self.assertNotIn(username, output)
            self.assertNotIn("C:\\", output)
            self.assertNotIn("C:/", output)
            self.assertNotIn("/home/", output)
            self.assertNotIn("/Users/", output)
            self.assertNotIn("nas-box", output)
            self.assertNotRegex(output, r"[A-Za-z]:[\\/]")

    def test_backward_compatibility_existing_cli_flags_and_legacy_report(self) -> None:
        vs = build_synthetic_vs(extra_param=True)
        ps = build_synthetic_ps(include_mimg=True)
        cs = build_synthetic_cs(saved_terminator=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "compat_shaders"
            root.mkdir()
            (root / "01_vs.bin").write_bytes(vs)
            (root / "02_ps.bin").write_bytes(ps)
            (root / "03_cs.bin").write_bytes(cs)

            # Verify existing flags (--json, --max-hits, --min-dwords, --max-bytes, --whole-file, --embedded).
            c1, out1, err1 = _run_cli([str(root), "--json", "--max-hits", "2", "--min-dwords", "2"])
            self.assertEqual((c1, err1), (0, ""))
            d1 = json.loads(out1)
            self.assertEqual(d1["schema_version"], 1)
            self.assertEqual(d1["max_hits"], 2)
            self.assertEqual(d1["min_dwords"], 2)
            self.assertTrue(d1["summary"]["hit_limit_reached"])
            self.assertEqual(d1["summary"]["total_shaders"], 2)

            c2, out2, err2 = _run_cli([str(root / "01_vs.bin"), "--json", "--whole-file", "--max-bytes", "64K"])
            self.assertEqual((c2, err2), (0, ""))
            d2 = json.loads(out2)
            self.assertEqual(d2["files"][0]["hash_scope"], "whole_file")
            self.assertEqual(d2["summary"]["stage_percentages"]["vertex"], 100.0)

            c3, out3, err3 = _run_cli([str(root / "01_vs.bin"), "--json", "--embedded"])
            self.assertEqual((c3, err3), (0, ""))
            d3 = json.loads(out3)
            self.assertEqual(d3["files"][0]["hash_scope"], "region")

            # Missing --from-env still returns exit code 1 with actionable message.
            c4, _out4, err4 = _run_cli(["--from-env", "--json"], env_path=Path(tmp) / "missing.env", env={})
            self.assertEqual(c4, 1)
            self.assertIn("GTA6_GAME_DIR", err4)

        # Legacy InspectReport construction without new optional fields populates T92 summary fields automatically.
        legacy_report = rdna2.InspectReport(
            schema_version=1,
            target_label="legacy_target",
            target_kind="file",
            max_bytes=None,
            max_hits=None,
            min_dwords=2,
            files_scanned=1,
            files_with_shaders=1,
            files_errored=0,
            total_file_bytes=len(ps),
            total_bytes_scanned=len(ps),
            total_shaders=1,
            unique_shader_hashes=1,
            total_pipeline_records=0,
            hit_limit_reached=False,
            stage_counts={"pixel": 1},
            terminator_counts={"s_endpgm": 1},
            export_target_histogram={"mrt0": 1},
            size_histogram={"0..64B": 1},
            instruction_totals={"exp": 1, "smem": 0, "vintrp": 1, "mimg": 1, "valu": 0, "salu": 0, "sopp": 1, "vmem": 0, "ds": 0, "flat": 0},
            pipeline_record_counts={},
        )
        legacy_dict = legacy_report.to_dict()
        self.assertEqual(
            legacy_dict["summary"]["stage_percentages"],
            {"vertex": 0.0, "pixel": 100.0, "compute": 0.0, "unknown": 0.0},
        )
        self.assertEqual(len(legacy_dict["summary"]["stage_breakdown"]), 4)
        self.assertEqual(legacy_dict["summary"]["shader_summary"], [])


if __name__ == "__main__":
    unittest.main()
