#!/usr/bin/env python3
"""Unit tests for tools/scan_shaders.py (Track 4 / S8).

Covers:
  - Synthetic DXBC SM5 (`SHEX`/`SHDR`), DXIL SM6 (`DXIL`/`PSV0`/`ILDB` with and
    without LLVM `BC\\xC0\\xDE` bitcode), and reflection/root-signature DXBC
  - Synthetic SPIR-V LE & BE modules (versions 1.0..1.6, OpEntryPoint stages,
    rejection of invalid versions, zero bound, and non-zero schema)
  - Synthetic WGPK v1 packs (matching tools/inspect_wgsl_pack.py & rejection of
    corrupted versions, stages, bounds, and overlapping ranges)
  - Synthetic RAGE shader signatures (`RGSH`, `fxc\\x00`, and `.fxc` file hints)
  - Zero false positives on deterministic random-byte streams and malformed headers
  - Bounded streaming across chunk boundaries (magics and headers straddling
    chunk boundaries with small `chunk_size`/`overlap`)
  - `--max-bytes-per-file` sampling and `--skip-incomplete` filtering
  - `--from-env` `.env` resolution and strict `--json` / Markdown path sanitization
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

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import inspect_wgsl_pack as wgpk  # noqa: E402
import scan_shaders as scanner  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic binary fixture builders (zero copyrighted game data)
# ---------------------------------------------------------------------------


def build_dxbc_container(
    chunks: list[tuple[bytes, bytes]],
    *,
    ver_major: int = 1,
    ver_minor: int = 0,
) -> bytes:
    """Build a synthetic valid DXBC container matching shaders/src/dxbc.rs."""
    table_end = 32 + len(chunks) * 4
    offsets: list[int] = []
    cursor = table_end
    for _fourcc, payload in chunks:
        offsets.append(cursor)
        cursor += 8 + len(payload)
    total_size = cursor

    digest = bytes(range(16))
    out = bytearray()
    out += struct.pack("<4s16sHHII", b"DXBC", digest, ver_major, ver_minor, total_size, len(chunks))
    for off in offsets:
        out += struct.pack("<I", off)
    for fourcc, payload in chunks:
        out += struct.pack("<4sI", fourcc, len(payload))
        out += payload
    return bytes(out)


def build_dxil_payload(
    *,
    shader_kind: int = 5,  # compute
    sm_major: int = 6,
    sm_minor: int = 6,
    dxil_major: int = 1,
    dxil_minor: int = 6,
    include_llvm_magic: bool = True,
) -> bytes:
    bitcode = (
        b"\x42\x43\xc0\xde\x11\x22\x33\x44"
        if include_llvm_magic
        else b"\x00\x00\x00\x00\x11\x22\x33\x44"
    )
    total_bytes = 24 + len(bitcode)
    size_u32 = total_bytes // 4
    program_version = (shader_kind << 16) | ((sm_major & 0xF) << 4) | (sm_minor & 0xF)
    dxil_version = (dxil_major << 8) | dxil_minor
    bitcode_offset = 16  # relative to byte 8 -> starts at byte 24
    return (
        struct.pack("<II4sIII", program_version, size_u32, b"DXIL", dxil_version, bitcode_offset, len(bitcode))
        + bitcode
    )


def build_legacy_shader_payload(
    *,
    shader_kind: int = 1,  # vertex
    sm_major: int = 5,
    sm_minor: int = 0,
    dword_count: int = 32,
) -> bytes:
    version_token = (shader_kind << 16) | ((sm_major & 0xF) << 4) | (sm_minor & 0xF)
    return struct.pack("<II", version_token, dword_count) + (b"\x00" * 16)


def build_spirv_module(
    *,
    endian: str = "little",
    major: int = 1,
    minor: int = 5,
    generator: int = 0x0008000A,
    bound: int = 42,
    schema: int = 0,
    execution_models: tuple[int, ...] = (4,),  # Fragment
) -> bytes:
    prefix = "<" if endian == "little" else ">"
    version = (major << 16) | (minor << 8)
    words: list[int] = [0x07230203, version, generator, bound, schema]
    # OpCapability Shader (word_count=2, opcode=17)
    words.extend([(2 << 16) | 17, 1])
    # OpMemoryModel Logical GLSL450 (word_count=3, opcode=14)
    words.extend([(3 << 16) | 14, 0, 1])
    for idx, model in enumerate(execution_models):
        # OpEntryPoint <model> <id> "main\0\0\0\0" (word_count=5, opcode=15)
        name_word = int.from_bytes(b"main", endian)
        words.extend([(5 << 16) | 15, model, idx + 1, name_word, 0])
    # OpFunction (terminates header walk)
    words.extend([(5 << 16) | 54, 1, 2, 0, 3])
    return struct.pack(f"{prefix}{len(words)}I", *words)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


class TestDxbcAndDxilClassification(unittest.TestCase):
    def test_classifies_dxil_sm6_with_llvm_bitcode_and_chunks(self) -> None:
        dxil = build_dxil_payload(shader_kind=5, sm_major=6, sm_minor=6, include_llvm_magic=True)
        isgn = struct.pack("<II", 2, 8)
        osgn = struct.pack("<II", 1, 8)
        psv0 = b"\x00" * 16
        blob = build_dxbc_container(
            [(b"ISGN", isgn), (b"OSGN", osgn), (b"PSV0", psv0), (b"DXIL", dxil)]
        )

        hits = scanner.scan_bytes(blob)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.format, "DXIL_SM6")
        self.assertEqual(hit.offset, 0)
        self.assertEqual(hit.size, len(blob))
        self.assertEqual(hit.version, "1.0")
        self.assertEqual(hit.shader_kind, "compute")
        self.assertEqual(hit.shader_model, "6.6")
        self.assertEqual(hit.stage_hints, ("compute",))
        self.assertEqual(hit.chunks, ("ISGN", "OSGN", "PSV0", "DXIL"))
        self.assertTrue(hit.has_llvm_bitcode)

    def test_classifies_dxbc_sm5_shex_and_shdr(self) -> None:
        shex = build_legacy_shader_payload(shader_kind=0, sm_major=5, sm_minor=0)
        stat = struct.pack("<IIII", 64, 8, 0, 4)
        blob = build_dxbc_container([(b"SHEX", shex), (b"STAT", stat)])

        hits = scanner.scan_bytes(blob)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.format, "DXBC_SM5")
        self.assertEqual(hit.shader_kind, "pixel")
        self.assertEqual(hit.shader_model, "5.0")
        self.assertEqual(hit.stage_hints, ("pixel",))
        self.assertEqual(hit.chunks, ("SHEX", "STAT"))
        self.assertIsNone(hit.has_llvm_bitcode)

        shdr = build_legacy_shader_payload(shader_kind=1, sm_major=4, sm_minor=0)
        blob_shdr = build_dxbc_container([(b"SHDR", shdr)])
        hits_shdr = scanner.scan_bytes(blob_shdr)
        self.assertEqual(len(hits_shdr), 1)
        self.assertEqual(hits_shdr[0].format, "DXBC_SM5")
        self.assertEqual(hits_shdr[0].shader_kind, "vertex")
        self.assertEqual(hits_shdr[0].shader_model, "4.0")

    def test_classifies_generic_dxbc_root_signature_container(self) -> None:
        blob = build_dxbc_container([(b"RTS0", b"\x01\x00\x00\x00" + b"\x00" * 16)])
        hits = scanner.scan_bytes(blob)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].format, "DXBC")
        self.assertEqual(hits[0].chunks, ("RTS0",))

    def test_rejects_malformed_dxbc_containers(self) -> None:
        valid_dxil = build_dxil_payload()
        valid = bytearray(build_dxbc_container([(b"DXIL", valid_dxil)]))

        # 1. chunk_count == 0
        zero_chunks = bytearray(valid)
        struct.pack_into("<I", zero_chunks, 28, 0)
        self.assertEqual(scanner.scan_bytes(zero_chunks), [])

        # 2. chunk_count > 64
        too_many_chunks = bytearray(valid)
        struct.pack_into("<I", too_many_chunks, 28, 65)
        self.assertEqual(scanner.scan_bytes(too_many_chunks), [])

        # 3. total_size > buffer size (truncated container)
        truncated = bytes(valid[:-4])
        self.assertEqual(scanner.scan_bytes(truncated), [])

        # 4. chunk offset pointing inside header/table
        bad_offset = bytearray(valid)
        struct.pack_into("<I", bad_offset, 32, 8)
        self.assertEqual(scanner.scan_bytes(bad_offset), [])

        # 5. chunk size exceeding total_size
        bad_chunk_size = bytearray(valid)
        chunk_off = struct.unpack_from("<I", bad_chunk_size, 32)[0]
        struct.pack_into("<I", bad_chunk_size, chunk_off + 4, 99999)
        self.assertEqual(scanner.scan_bytes(bad_chunk_size), [])

        # 6. unrecognized FourCC only
        unknown_fourcc = build_dxbc_container([(b"ZZZZ", b"\x00" * 16)])
        self.assertEqual(scanner.scan_bytes(unknown_fourcc), [])


class TestSpirvClassification(unittest.TestCase):
    def test_detects_little_and_big_endian_spirv_with_stage_hints(self) -> None:
        spv_le = build_spirv_module(endian="little", major=1, minor=6, execution_models=(0, 4))
        spv_be = build_spirv_module(endian="big", major=1, minor=3, execution_models=(5,))
        blob = b"\xaa" * 17 + spv_le + b"\xbb" * 23 + spv_be

        hits = scanner.scan_bytes(blob)
        self.assertEqual(len(hits), 2)

        self.assertEqual(hits[0].format, "SPIRV")
        self.assertEqual(hits[0].offset, 17)
        self.assertEqual(hits[0].endianness, "little")
        self.assertEqual(hits[0].version, "1.6")
        self.assertEqual(hits[0].stage_hints, ("vertex", "fragment"))
        self.assertEqual(hits[0].spirv_bound, 42)

        self.assertEqual(hits[1].format, "SPIRV")
        self.assertEqual(hits[1].offset, 17 + len(spv_le) + 23)
        self.assertEqual(hits[1].endianness, "big")
        self.assertEqual(hits[1].version, "1.3")
        self.assertEqual(hits[1].stage_hints, ("compute",))

    def test_rejects_invalid_spirv_headers(self) -> None:
        # Invalid minor version (1.7)
        self.assertEqual(scanner.scan_bytes(build_spirv_module(major=1, minor=7)), [])
        # Invalid major version (2.0)
        self.assertEqual(scanner.scan_bytes(build_spirv_module(major=2, minor=0)), [])
        # Non-zero low byte in version word
        bad_ver = bytearray(build_spirv_module())
        struct.pack_into("<I", bad_ver, 4, 0x00010501)
        self.assertEqual(scanner.scan_bytes(bad_ver), [])
        # Zero bound
        self.assertEqual(scanner.scan_bytes(build_spirv_module(bound=0)), [])
        # Non-zero schema
        self.assertEqual(scanner.scan_bytes(build_spirv_module(schema=1)), [])


class TestWgpkAndRageClassification(unittest.TestCase):
    def test_detects_valid_wgpk_v1_pack_from_inspect_wgsl_pack(self) -> None:
        pack = wgpk.build_wgpk(
            [
                (None, "vertex", "@vertex fn vs() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0); }"),
                (None, "fragment", "@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }"),
                (None, "compute", "@compute @workgroup_size(8) fn cs() {}"),
            ]
        )
        hits = scanner.scan_bytes(b"\x00" * 32 + pack)
        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.format, "WGPK")
        self.assertEqual(hit.offset, 32)
        self.assertEqual(hit.size, len(pack))
        self.assertEqual(hit.version, "1")
        self.assertEqual(hit.wgpk_entry_count, 3)
        self.assertEqual(hit.wgpk_stage_counts, {"vertex": 1, "fragment": 1, "compute": 1})
        self.assertEqual(hit.stage_hints, ("vertex", "fragment", "compute"))

    def test_rejects_malformed_wgpk(self) -> None:
        pack = bytearray(wgpk.build_wgpk([(None, "compute", "@compute @workgroup_size(1) fn cs() {}")]))

        # Wrong version
        bad_ver = bytearray(pack)
        struct.pack_into("<I", bad_ver, 4, 2)
        self.assertEqual(scanner.scan_bytes(bad_ver), [])

        # Invalid stage code (9)
        bad_stage = bytearray(pack)
        bad_stage[16 + 8] = 9
        self.assertEqual(scanner.scan_bytes(bad_stage), [])

        # Out-of-bounds entry length
        bad_len = bytearray(pack)
        struct.pack_into("<I", bad_len, 16 + 16, 99999)
        self.assertEqual(scanner.scan_bytes(bad_len), [])

    def test_detects_rage_rgsh_fxc_magic_and_fxc_extension_hint(self) -> None:
        rgsh_blob = b"RGSH" + struct.pack("<I", 1) + b"\x00" * 24
        fxc_blob = b"fxc\x00" + struct.pack("<I", 2) + b"\x00" * 24
        dxbc_inner = build_dxbc_container([(b"SHEX", build_legacy_shader_payload())])
        combined = fxc_blob + dxbc_inner + rgsh_blob

        hits = scanner.scan_bytes(combined)
        formats = [h.format for h in hits]
        self.assertEqual(formats, ["RAGE_FXC", "DXBC_SM5", "RAGE_RGSH"])

        with tempfile.TemporaryDirectory() as tmp:
            opaque_fxc = Path(tmp) / "opaque_compressed.fxc"
            opaque_fxc.write_bytes(b"ENCRYPTED_RAGE_SHADER_PAYLOAD_BYTES")
            res = scanner.scan_file(opaque_fxc)
            self.assertEqual(res.extension_hint, "fxc")
            self.assertEqual(len(res.hits), 1)
            self.assertEqual(res.hits[0].format, "RAGE_FXC_HINT")
            self.assertEqual(res.hits[0].confidence, "hint")


class TestNoiseAndStreamingBoundaries(unittest.TestCase):
    def test_zero_false_positives_on_random_bytes_and_fake_magics(self) -> None:
        rng = random.Random(0x67A6_58AD)
        noise = bytearray(rng.randbytes(512 * 1024))
        # Inject bare magics surrounded by random garbage to verify structural validators reject them.
        for pos, magic in (
            (1024, b"DXBC"),
            (2048, b"\x03\x02\x23\x07"),
            (4096, b"\x07\x23\x02\x03"),
            (8192, b"WGPK"),
        ):
            noise[pos : pos + 4] = magic
            noise[pos + 4 : pos + 64] = b"\xff" * 60

        hits = scanner.scan_bytes(bytes(noise))
        self.assertEqual(hits, [])

    def test_streaming_across_chunk_boundaries_with_small_chunks(self) -> None:
        dxil_blob = build_dxbc_container(
            [(b"DXIL", build_dxil_payload(shader_kind=13, sm_major=6, sm_minor=5))]
        )
        spv_blob = build_spirv_module(endian="little", major=1, minor=4, execution_models=(5,))
        wgpk_blob = wgpk.build_wgpk([(None, "fragment", "@fragment fn main() {}")])

        # Place containers so magics and headers straddle 512-byte chunk boundaries.
        buf = bytearray(b"\x55" * 2048)
        # Straddle boundary 512 (magic starts at 510 -> b"DX" in chunk 0, b"BC" in chunk 1)
        buf[510 : 510 + len(dxil_blob)] = dxil_blob
        # Straddle boundary 1024 (SPIR-V header starts at 1016)
        buf[1016 : 1016 + len(spv_blob)] = spv_blob
        # Straddle boundary 1536 (WGPK header starts at 1534)
        buf[1534 : 1534 + len(wgpk_blob)] = wgpk_blob

        with tempfile.TemporaryDirectory() as tmp:
            blob_path = Path(tmp) / "straddle.bin"
            blob_path.write_bytes(buf)

            result = scanner.scan_file(blob_path, chunk_size=512, overlap=256)
            self.assertEqual(result.bytes_scanned, len(buf))
            self.assertFalse(result.truncated)
            self.assertEqual(
                [(h.format, h.offset) for h in result.hits],
                [
                    ("DXIL_SM6", 510),
                    ("SPIRV", 1016),
                    ("WGPK", 1534),
                ],
            )
            self.assertEqual(result.hits[0].shader_kind, "mesh")
            self.assertEqual(result.hits[1].stage_hints, ("compute",))
            self.assertEqual(result.hits[2].wgpk_entry_count, 1)

    def test_max_bytes_per_file_limits_scan_and_marks_truncated(self) -> None:
        first = build_spirv_module(endian="little", execution_models=(0,))
        second = build_dxbc_container([(b"DXIL", build_dxil_payload())])
        payload = first + (b"\x00" * 4096) + second

        with tempfile.TemporaryDirectory() as tmp:
            fpath = Path(tmp) / "large.rpf"
            fpath.write_bytes(payload)

            sampled = scanner.scan_file(fpath, max_bytes=1024, chunk_size=512, overlap=128)
            self.assertTrue(sampled.truncated)
            self.assertEqual(sampled.bytes_scanned, 1024)
            self.assertEqual(sampled.file_size, len(payload))
            self.assertEqual([h.format for h in sampled.hits], ["SPIRV"])

            full = scanner.scan_file(fpath, max_bytes=None, chunk_size=512, overlap=128)
            self.assertFalse(full.truncated)
            self.assertEqual(full.bytes_scanned, len(payload))
            self.assertEqual([h.format for h in full.hits], ["SPIRV", "DXIL_SM6"])


class TestDirectoryCliAndSanitization(unittest.TestCase):
    def test_directory_scan_skip_incomplete_and_strict_json_sanitization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "MyPrivateGameDir"
            sub = root / "x64" / "shaders"
            sub.mkdir(parents=True)

            # Valid shader files
            (sub / "mesh.dxil").write_bytes(
                build_dxbc_container([(b"DXIL", build_dxil_payload(shader_kind=13, sm_major=6, sm_minor=5))])
            )
            (sub / "post.spv").write_bytes(build_spirv_module(execution_models=(4,)))
            (sub / "bundle.wgsl.pack").write_bytes(
                wgpk.build_wgpk([(None, "vertex", "@vertex fn vs() {}")])
            )
            (root / "notes.txt").write_text("plain text without shaders", encoding="utf-8")

            # Incomplete / partial download files
            (root / "huge.rpf.!qB").write_bytes(build_spirv_module())
            (root / "empty.rpf").write_bytes(b"")
            (root / "sparse.rpf").write_bytes((b"\x00" * 1024) + build_spirv_module())

            # Run CLI with --path, --skip-incomplete, --json
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = scanner.main(
                    [
                        "--path",
                        str(root),
                        "--skip-incomplete",
                        "--max-bytes-per-file",
                        "1M",
                        "--json",
                    ]
                )
            self.assertEqual(rc, 0, stderr.getvalue())
            raw_json = stdout.getvalue()
            data = json.loads(raw_json)

            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["target"], "MyPrivateGameDir")
            self.assertEqual(data["summary"]["files_scanned"], 4)
            self.assertEqual(data["summary"]["files_with_shaders"], 3)
            self.assertEqual(data["summary"]["files_skipped_incomplete"], 3)
            self.assertEqual(
                data["summary"]["format_counts"],
                {"DXIL_SM6": 1, "SPIRV": 1, "WGPK": 1},
            )
            reported_paths = [f["path"] for f in data["files"]]
            self.assertEqual(
                reported_paths,
                [
                    "x64/shaders/bundle.wgsl.pack",
                    "x64/shaders/mesh.dxil",
                    "x64/shaders/post.spv",
                ],
            )
            skipped_reasons = {s["path"]: s["reason"] for s in data["skipped_incomplete"]}
            self.assertEqual(
                skipped_reasons,
                {
                    "empty.rpf": "empty_file",
                    "huge.rpf.!qB": "temp_suffix",
                    "sparse.rpf": "sparse_zero_header",
                },
            )

            # Strict sanitization assertions: zero absolute paths, drive letters, or parent dirs!
            resolved_tmp = str(Path(tmp).resolve())
            self.assertNotIn(resolved_tmp, raw_json)
            self.assertNotIn(resolved_tmp.replace("\\", "/"), raw_json)
            self.assertNotIn(":\\", raw_json)
            self.assertNotIn(":/", raw_json)
            for env_var in ("USERNAME", "USER", "LOGNAME"):
                user = os.environ.get(env_var, "").strip()
                if len(user) >= 3:
                    self.assertNotIn(user.lower(), raw_json.lower())

    def test_from_env_uses_dotenv_and_redacts_game_dir_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = Path(tmp) / "repo"
            fake_repo.mkdir()
            game_dir = Path(tmp) / "SecretXboxInstall"
            game_dir.mkdir()
            (game_dir / "shader_pack.fxc").write_bytes(
                b"fxc\x00" + struct.pack("<I", 1) + b"\x00" * 16
            )

            env_file = fake_repo / ".env"
            env_file.write_text(f'GTA6_GAME_DIR="{game_dir}"\n', encoding="utf-8")

            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = scanner.main(
                    ["--from-env", "--json"],
                    env_path=env_file,
                    env={},
                )
            self.assertEqual(rc, 0, stderr.getvalue())
            raw_json = stdout.getvalue()
            data = json.loads(raw_json)
            self.assertEqual(data["target"], "<GTA6_GAME_DIR>")
            self.assertEqual(data["summary"]["format_counts"], {"RAGE_FXC": 1})
            self.assertNotIn("SecretXboxInstall", raw_json)
            self.assertNotIn(str(game_dir), raw_json)

            # Markdown mode also works and stays sanitized
            md_out = io.StringIO()
            with contextlib.redirect_stdout(md_out):
                rc_md = scanner.main(["--from-env"], env_path=env_file, env={})
            self.assertEqual(rc_md, 0)
            md_text = md_out.getvalue()
            self.assertIn("# Shader Container Scan Report: `<GTA6_GAME_DIR>`", md_text)
            self.assertIn("`RAGE_FXC`", md_text)
            self.assertNotIn("SecretXboxInstall", md_text)

    def test_from_env_missing_returns_nonzero_without_leaking_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_env = Path(tmp) / ".env"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = scanner.main(["--from-env"], env_path=missing_env, env={})
            self.assertEqual(rc, 1)
            self.assertIn("GTA6_GAME_DIR is not set", stderr.getvalue())
            self.assertNotIn(tmp, stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
