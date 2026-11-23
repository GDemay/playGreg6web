#!/usr/bin/env python3
"""Unit tests for tools/extract_dxil_corpus.py (Track 4 / S10).

Covers synthetic DXBC/DXIL containers with DXIL/PSV0/RTS0/ISG1/OSG1/ACCC,
malformed containers, deduplication, git-safety guard, cross-platform
path sanitization (Windows `\\` and POSIX `/`), and `--json` CLI output.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import extract_dxil_corpus as corpus  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic fixture builders (zero copyrighted game data)
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
        struct.pack(
            "<II4sIII",
            program_version,
            size_u32,
            b"DXIL",
            dxil_version,
            bitcode_offset,
            len(bitcode),
        )
        + bitcode
    )


def build_psv0_payload(
    *,
    stage: int = 5,  # compute
    resource_count: int = 3,
    input_elements: int = 2,
    output_elements: int = 1,
    runtime_info_size: int = 36,
) -> bytes:
    """Build a minimal PSV0 v1 RuntimeInfo + resource count."""
    info = bytearray(runtime_info_size)
    # bytes 0..16: stage union (zeros ok for compute)
    # bytes 16..24: min/max wave lane counts
    struct.pack_into("<II", info, 16, 4, 128)
    if runtime_info_size >= 36:
        info[24] = stage & 0xFF
        info[25] = 0  # UsesViewID
        info[28] = input_elements & 0xFF
        info[29] = output_elements & 0xFF
        info[30] = 0  # patch/prim
        info[31] = 1  # SigInputVectors
        info[32] = 1  # SigOutputVectors[0]
    out = bytearray()
    out += struct.pack("<I", runtime_info_size)
    out += info
    out += struct.pack("<I", resource_count)
    if resource_count > 0:
        bind_size = 16
        out += struct.pack("<I", bind_size)
        for i in range(resource_count):
            # ResType, Space, LowerBound, UpperBound
            out += struct.pack("<IIII", 1, 0, i, i)
    return bytes(out)


def build_rts0_payload(
    *,
    version: int = 2,  # 1.1
    num_params: int = 4,
    num_samplers: int = 2,
    flags: int = 0x1,
) -> bytes:
    """Build a minimal RTS0 root signature header (v1.0 / v1.1)."""
    header_size = 24
    params_off = header_size
    sampler_off = header_size + num_params * 12
    out = bytearray()
    out += struct.pack(
        "<6I",
        version,
        num_params,
        params_off,
        num_samplers,
        sampler_off,
        flags,
    )
    # Parameter headers (type, visibility, offset) — opaque placeholders
    for i in range(num_params):
        out += struct.pack("<III", 2, 0, sampler_off + 52 * num_samplers + i * 12)
    # Static sampler placeholders (52 bytes each for v1.0/v1.1)
    for _ in range(num_samplers):
        out += b"\x00" * 52
    return bytes(out)


def build_isg1_payload(
    elements: list[tuple[str, int, int]],
) -> bytes:
    """Build an ISG1/OSG1 signature chunk with semantic names (32-byte elements)."""
    element_count = len(elements)
    first_offset = 8
    # Strings are placed after the element table.
    string_base = first_offset + element_count * 32
    out = bytearray()
    out += struct.pack("<II", element_count, first_offset)

    string_blob = bytearray()
    name_offsets: list[int] = []
    for name, _sem_idx, _reg in elements:
        name_offsets.append(string_base + len(string_blob))
        string_blob += name.encode("ascii") + b"\x00"

    for idx, (name, sem_idx, reg) in enumerate(elements):
        # Stream, SemanticName, SemanticIndex, SystemValue, CompType, Register,
        # Mask, NeverWrites, Pad, MinPrecision
        out += struct.pack(
            "<IIIIIIBBHI",
            0,  # stream
            name_offsets[idx],
            sem_idx,
            0,  # system value
            3,  # float32
            reg,
            0xF,  # mask
            0,
            0,
            0,
        )
    out += string_blob
    return bytes(out)


def build_full_dxil_container(
    *,
    shader_kind: int = 0,  # pixel
    sm_major: int = 6,
    sm_minor: int = 6,
    include_accc: bool = True,
) -> bytes:
    dxil = build_dxil_payload(shader_kind=shader_kind, sm_major=sm_major, sm_minor=sm_minor)
    psv0 = build_psv0_payload(stage=shader_kind, resource_count=2, input_elements=3, output_elements=2)
    rts0 = build_rts0_payload(version=1, num_params=3, num_samplers=1, flags=0)
    isg1 = build_isg1_payload([("POSITION", 0, 0), ("TEXCOORD", 0, 1), ("NORMAL", 0, 2)])
    osg1 = build_isg1_payload([("SV_Target", 0, 0), ("SV_Target", 1, 1)])
    chunks: list[tuple[bytes, bytes]] = [
        (b"DXIL", dxil),
        (b"PSV0", psv0),
        (b"RTS0", rts0),
        (b"ISG1", isg1),
        (b"OSG1", osg1),
        (b"STAT", struct.pack("<IIII", 100, 8, 0, 4)),
        (b"HASH", struct.pack("<I", 1) + bytes(range(16))),
    ]
    if include_accc:
        chunks.append((b"ACCC", b"\xDE\xAD\xBE\xEF" + b"\x00" * 28))
    return build_dxbc_container(chunks)


# ---------------------------------------------------------------------------
# Parser unit tests
# ---------------------------------------------------------------------------


class TestParseDxbcContainer(unittest.TestCase):
    def test_parses_full_dxil_psv0_rts0_isg1_osg1_accc(self) -> None:
        blob = build_full_dxil_container(shader_kind=0, sm_major=6, sm_minor=6)
        info = corpus.parse_dxbc_container(blob)

        self.assertEqual(info.format, "DXIL_SM6")
        self.assertEqual(info.total_size, len(blob))
        self.assertEqual(
            info.fourccs,
            ("DXIL", "PSV0", "RTS0", "ISG1", "OSG1", "STAT", "HASH", "ACCC"),
        )
        self.assertTrue(info.has_accc)
        self.assertTrue(info.has_stat)
        self.assertTrue(info.has_hash)

        assert info.dxil is not None
        self.assertEqual(info.dxil.shader_kind, "pixel")
        self.assertEqual(info.dxil.profile_prefix, "ps")
        self.assertEqual((info.dxil.sm_major, info.dxil.sm_minor), (6, 6))
        self.assertTrue(info.dxil.has_llvm_bitcode_magic)
        self.assertEqual(info.dxil.bitcode_size, 8)

        assert info.psv0 is not None
        self.assertEqual(info.psv0.runtime_info_size, 36)
        self.assertEqual(info.psv0.stage, "pixel")
        self.assertEqual(info.psv0.resource_count, 2)
        self.assertEqual(info.psv0.input_element_count, 3)
        self.assertEqual(info.psv0.output_element_count, 2)

        assert info.rts0 is not None
        self.assertEqual(info.rts0.version, 1)
        self.assertEqual(info.rts0.version_label, "1.0")
        self.assertEqual(info.rts0.root_parameter_count, 3)
        self.assertEqual(info.rts0.static_sampler_count, 1)

        sig_map = {s.fourcc: s for s in info.signatures}
        self.assertIn("ISG1", sig_map)
        self.assertIn("OSG1", sig_map)
        self.assertEqual(sig_map["ISG1"].element_count, 3)
        self.assertEqual(
            [e.semantic_name for e in sig_map["ISG1"].elements],
            ["POSITION", "TEXCOORD", "NORMAL"],
        )
        self.assertEqual(sig_map["OSG1"].element_count, 2)
        self.assertEqual(sig_map["OSG1"].elements[0].semantic_name, "SV_Target")
        self.assertEqual(sig_map["OSG1"].elements[1].semantic_index, 1)

        self.assertEqual(info.sha256_hex, hashlib.sha256(blob).hexdigest())
        self.assertTrue(info.corpus_filename().startswith(info.sha256_hex[:16]))
        self.assertIn("_ps_sm6_6.dxbc", info.corpus_filename())

    def test_decodes_all_common_shader_kinds_and_rts0_v1_1(self) -> None:
        for raw, name, prefix in (
            (0, "pixel", "ps"),
            (1, "vertex", "vs"),
            (2, "geometry", "gs"),
            (3, "hull", "hs"),
            (4, "domain", "ds"),
            (5, "compute", "cs"),
            (6, "library", "lib"),
            (13, "mesh", "ms"),
            (14, "amplification", "as"),
        ):
            dxil = build_dxil_payload(shader_kind=raw, sm_major=6, sm_minor=5)
            rts0 = build_rts0_payload(version=2, num_params=1, num_samplers=0, flags=0)
            blob = build_dxbc_container([(b"DXIL", dxil), (b"RTS0", rts0)])
            info = corpus.parse_dxbc_container(blob)
            assert info.dxil is not None
            self.assertEqual(info.dxil.shader_kind, name)
            self.assertEqual(info.dxil.profile_prefix, prefix)
            assert info.rts0 is not None
            self.assertEqual(info.rts0.version_label, "1.1")

    def test_rejects_malformed_containers(self) -> None:
        valid = bytearray(build_full_dxil_container())

        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(b"")
        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(b"\x00" * 16)

        bad_magic = bytearray(valid)
        bad_magic[0:4] = b"NOPE"
        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(bytes(bad_magic))

        bad_total = bytearray(valid)
        struct.pack_into("<I", bad_total, 24, len(valid) + 100)
        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(bytes(bad_total))

        zero_chunks = bytearray(valid)
        struct.pack_into("<I", zero_chunks, 28, 0)
        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(bytes(zero_chunks))

        too_many = bytearray(valid)
        struct.pack_into("<I", too_many, 28, 65)
        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(bytes(too_many))

        bad_offset = bytearray(valid)
        struct.pack_into("<I", bad_offset, 32, 8)
        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(bytes(bad_offset))

        bad_size = bytearray(valid)
        chunk_off = struct.unpack_from("<I", bad_size, 32)[0]
        struct.pack_into("<I", bad_size, chunk_off + 4, 999999)
        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(bytes(bad_size))

        unknown_only = build_dxbc_container([(b"ZZZZ", b"\x00" * 16)])
        with self.assertRaises(corpus.DxbcParseError):
            corpus.parse_dxbc_container(unknown_only)


# ---------------------------------------------------------------------------
# Extraction / safety / sanitization
# ---------------------------------------------------------------------------


class TestExtractAndSafety(unittest.TestCase):
    def test_inspect_only_default_writes_nothing(self) -> None:
        blob = build_full_dxil_container()
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "game" / "shaders"
            src.mkdir(parents=True)
            (src / "mesh.dxil").write_bytes(blob)

            report = corpus.extract_corpus(src, inspect_only=True)
            self.assertTrue(report.inspect_only)
            self.assertEqual(report.containers_found, 1)
            self.assertEqual(report.containers_written, 0)
            self.assertEqual(report.written_blobs, [])
            # No files created under source or elsewhere by default.
            self.assertEqual(list(src.glob("*.dxbc")), [])
            self.assertFalse((src / "corpus-manifest.json").exists())

    def test_out_dir_deduplicates_and_writes_manifest(self) -> None:
        blob_a = build_full_dxil_container(shader_kind=5, sm_major=6, sm_minor=6)
        blob_b = build_full_dxil_container(shader_kind=1, sm_major=6, sm_minor=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            game = root / "SecretGameInstall" / "x64"
            game.mkdir(parents=True)
            (game / "a.dll").write_bytes(b"\x00" * 64 + blob_a + b"\xff" * 32 + blob_a)
            (game / "b.bin").write_bytes(blob_b)
            out = root / "corpus_out"

            report = corpus.extract_corpus(game, out_dir=out, inspect_only=False)
            self.assertFalse(report.inspect_only)
            self.assertEqual(report.containers_found, 3)  # blob_a twice + blob_b
            self.assertEqual(report.containers_unique, 2)
            self.assertEqual(report.containers_deduplicated, 1)
            self.assertEqual(report.containers_written, 2)

            manifest_path = out / "corpus-manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["blob_count"], 2)
            self.assertEqual(len(list(out.glob("*.dxbc"))), 2)

            # Filenames follow <sha256[:16]>_<kind>_sm<maj>_<min>.dxbc
            names = sorted(p.name for p in out.glob("*.dxbc"))
            self.assertTrue(any("_cs_sm6_6.dxbc" in n for n in names))
            self.assertTrue(any("_vs_sm6_5.dxbc" in n for n in names))

            # Re-run should mark blobs already_present and not inflate unique count.
            report2 = corpus.extract_corpus(game, out_dir=out, inspect_only=False)
            self.assertEqual(report2.containers_unique, 2)
            statuses = {b["status"] for b in report2.written_blobs}
            self.assertEqual(statuses, {"already_present"})

    def test_git_safety_refuses_tracked_repo_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = Path(tmp) / "repo"
            (fake_repo / "tools").mkdir(parents=True)
            (fake_repo / "web").mkdir()
            (fake_repo / "work").mkdir()
            (fake_repo / "local-data").mkdir()

            self.assertTrue(
                corpus.is_safe_output_dir(fake_repo / "work" / "shaders", repo_root=fake_repo)
            )
            self.assertTrue(
                corpus.is_safe_output_dir(fake_repo / "local-data" / "dxil", repo_root=fake_repo)
            )
            with tempfile.TemporaryDirectory() as outside:
                self.assertTrue(corpus.is_safe_output_dir(outside, repo_root=fake_repo))

            self.assertFalse(corpus.is_safe_output_dir(fake_repo / "tools", repo_root=fake_repo))
            self.assertFalse(corpus.is_safe_output_dir(fake_repo / "web", repo_root=fake_repo))
            with self.assertRaises(corpus.UnsafeOutputPathError):
                corpus.validate_output_dir(fake_repo / "shaders", repo_root=fake_repo)
            with self.assertRaises(corpus.UnsafeOutputPathError):
                corpus.validate_output_dir(
                    fake_repo / "work" / ".." / "tools", repo_root=fake_repo
                )

            # CLI path also refuses tracked dirs.
            blob = build_full_dxil_container()
            src = fake_repo / "input.dxbc"
            src.write_bytes(blob)
            code, _stdout, stderr = _run_cli(
                ["--path", str(src), "--out-dir", str(fake_repo / "tools" / "out")],
                repo_root=fake_repo,
            )
            self.assertEqual(code, 1)
            self.assertIn("Refusing to write", stderr)


class TestSanitizationAndCli(unittest.TestCase):
    def test_cross_platform_path_sanitization_normalizes_separators(self) -> None:
        # Windows-style absolute path
        self.assertEqual(
            corpus.sanitize_source_label(r"C:\Users\SecretUser\game\shaders\mesh.dxil"),
            "mesh.dxil",
        )
        # POSIX-style absolute path
        self.assertEqual(
            corpus.sanitize_source_label("/Users/SecretUser/game/shaders/mesh.dxil"),
            "mesh.dxil",
        )
        # Mixed separators
        self.assertEqual(
            corpus.sanitize_source_label(r"C:/Users\SecretUser\game/foo\bar.dxbc"),
            "bar.dxbc",
        )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "GameRoot"
            nested = base / "x64" / "shaders"
            nested.mkdir(parents=True)
            fpath = nested / "post.dxil"
            fpath.write_bytes(b"x")
            rel = corpus.sanitize_relative_path(fpath, base)
            self.assertEqual(rel, "x64/shaders/post.dxil")
            self.assertNotIn("\\", rel)
            self.assertNotIn(str(tmp), rel)

        scrubbed = corpus._scrub_sensitive_text(
            r"failed C:\Users\SecretUser\game\a.dll and /home/SecretUser/b.bin"
        )
        self.assertNotIn("SecretUser", scrubbed)
        self.assertIn("<path>", scrubbed)

    def test_json_cli_output_is_sanitized(self) -> None:
        blob = build_full_dxil_container(shader_kind=13, sm_major=6, sm_minor=5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "MyPrivateGameDir"
            sub = root / "x64" / "shaders"
            sub.mkdir(parents=True)
            (sub / "mesh.dxil").write_bytes(blob)
            out = Path(tmp) / "safe_out"

            code, stdout, stderr = _run_cli(
                [
                    "--path",
                    str(root),
                    "--out-dir",
                    str(out),
                    "--json",
                ]
            )
            self.assertEqual(code, 0, stderr)
            data = json.loads(stdout)

            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["target"], "MyPrivateGameDir")
            self.assertEqual(data["summary"]["containers_found"], 1)
            self.assertEqual(data["summary"]["containers_unique"], 1)
            self.assertEqual(data["summary"]["containers_written"], 1)
            self.assertEqual(data["summary"]["format_counts"], {"DXIL_SM6": 1})
            self.assertEqual(data["summary"]["kind_counts"], {"mesh": 1})
            self.assertIn("ACCC", data["summary"]["fourcc_counts"])

            container = data["files"][0]["containers"][0]
            self.assertEqual(container["source_path"], "x64/shaders/mesh.dxil")
            self.assertEqual(container["dxil"]["profile_prefix"], "ms")
            self.assertEqual(container["dxil"]["shader_model"], "6.5")
            self.assertTrue(container["has_accc"])
            self.assertEqual(container["psv0"]["stage"], "mesh")
            self.assertEqual(container["rts0"]["version_label"], "1.0")
            self.assertEqual(
                [e["semantic_name"] for e in container["signatures"][0]["elements"]],
                ["POSITION", "TEXCOORD", "NORMAL"],
            )

            # Strict sanitization: zero absolute paths / drive letters / usernames / raw bytes.
            resolved_tmp = str(Path(tmp).resolve())
            self.assertNotIn(resolved_tmp, stdout)
            self.assertNotIn(resolved_tmp.replace("\\", "/"), stdout)
            self.assertNotIn(":\\", stdout)
            self.assertNotIn("MyPrivateGameDir/", stdout)  # only basename target label
            for env_var in ("USERNAME", "USER", "LOGNAME"):
                user = os.environ.get(env_var, "").strip()
                if len(user) >= 3:
                    self.assertNotIn(user.lower(), stdout.lower())
            # Never dump raw shader blob bytes as hex dumps of the ACCC payload etc.
            self.assertNotIn("deadbeef", stdout.lower())

            # Manifest on disk is also sanitized.
            manifest = json.loads((out / "corpus-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["target"], "MyPrivateGameDir")
            self.assertNotIn(resolved_tmp, json.dumps(manifest))

    def test_from_env_and_inspect_only_flag(self) -> None:
        blob = build_full_dxil_container()
        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = Path(tmp) / "repo"
            fake_repo.mkdir()
            game_dir = Path(tmp) / "SecretXboxInstall"
            game_dir.mkdir()
            (game_dir / "pack.dll").write_bytes(blob)
            env_file = fake_repo / ".env"
            env_file.write_text(f'GTA6_GAME_DIR="{game_dir}"\n', encoding="utf-8")
            out = Path(tmp) / "out"

            code, stdout, stderr = _run_cli(
                ["--from-env", "--out-dir", str(out), "--inspect-only", "--json"],
                env_path=env_file,
                env={},
            )
            self.assertEqual(code, 0, stderr)
            data = json.loads(stdout)
            self.assertEqual(data["target"], "<GTA6_GAME_DIR>")
            self.assertTrue(data["inspect_only"])
            self.assertEqual(data["summary"]["containers_written"], 0)
            self.assertFalse(out.exists())
            self.assertNotIn("SecretXboxInstall", stdout)

    def test_from_env_missing_returns_nonzero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_env = Path(tmp) / ".env"
            code, _stdout, stderr = _run_cli(
                ["--from-env"],
                env_path=missing_env,
                env={},
            )
            self.assertEqual(code, 1)
            self.assertIn("GTA6_GAME_DIR is not set", stderr)
            self.assertNotIn(tmp, stderr)


def _run_cli(
    argv: list[str],
    *,
    env_path: Path | None = None,
    env: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> tuple[int, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    kwargs: dict = {}
    if env_path is not None:
        kwargs["env_path"] = env_path
    if env is not None:
        kwargs["env"] = env
    if repo_root is not None:
        kwargs["repo_root"] = repo_root
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        code = corpus.main(argv, **kwargs)
    return code, stdout_buf.getvalue(), stderr_buf.getvalue()


if __name__ == "__main__":
    unittest.main()
