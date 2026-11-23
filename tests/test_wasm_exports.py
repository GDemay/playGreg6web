import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

import check_wasm_exports as wasm_check  # noqa: E402


class TestWasmBinaryParser(unittest.TestCase):
    def test_parses_valid_wasm64_shared_imported_memory_and_host_imports(self) -> None:
        raw = wasm_check.build_valid_runtime_wasm(
            is_memory64=True,
            is_shared=True,
            imported_memory=True,
            min_pages=256,
            max_pages=65536,
        )
        module = wasm_check.parse_wasm_module(raw)

        self.assertEqual(module.version, 1)
        self.assertEqual(module.byte_size, len(raw))
        self.assertEqual(len(module.memories), 0)
        self.assertEqual(len(module.all_memories), 1)

        mem = module.all_memories[0]
        self.assertEqual(mem.source, "import")
        self.assertEqual(mem.module, "env")
        self.assertEqual(mem.name, "memory")
        self.assertTrue(mem.is_memory64)
        self.assertTrue(mem.is_shared)
        self.assertTrue(mem.has_max)
        self.assertEqual(mem.flags, 0x07)
        self.assertEqual(mem.min_pages, 256)
        self.assertEqual(mem.max_pages, 65536)

        self.assertIn("target_features", module.custom_sections)
        self.assertIn("+memory64", module.target_features)
        self.assertIn("+atomics", module.target_features)

        report = wasm_check.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
            require_host_imports=True,
        )
        self.assertTrue(report.ok)
        self.assertTrue(report.is_memory64)
        self.assertTrue(report.is_shared_memory)
        self.assertTrue(report.all_host_imports_ok)
        self.assertEqual(report.missing_host_imports, ())
        self.assertEqual(report.mismatched_host_imports, ())
        self.assertEqual(report.errors, ())
        self.assertEqual(report.warnings, ())

    def test_parses_defined_memory64_with_u64_leb128_limits(self) -> None:
        # Exceeds u32 max (0xFFFF_FFFF) to verify true u64 LEB128 decoding.
        min_pages = 0x1_0000_0001
        max_pages = 0x2_0000_0000
        raw = wasm_check.build_valid_runtime_wasm(
            is_memory64=True,
            is_shared=True,
            imported_memory=False,
            min_pages=min_pages,
            max_pages=max_pages,
        )
        module = wasm_check.parse_wasm_module(raw)

        self.assertEqual(len(module.memories), 1)
        mem = module.memories[0]
        self.assertEqual(mem.source, "defined")
        self.assertTrue(mem.is_memory64)
        self.assertTrue(mem.is_shared)
        self.assertTrue(mem.has_max)
        self.assertEqual(mem.min_pages, min_pages)
        self.assertEqual(mem.max_pages, max_pages)

        report = wasm_check.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
            require_host_imports=True,
        )
        self.assertTrue(report.ok)

    def test_parses_all_valtypes_import_kinds_export_kinds_and_custom_sections(self) -> None:
        raw = wasm_check.build_wasm_module(
            types=[
                (("i32", "i64", "f32", "f64", "v128", "funcref", "externref"), ("i64", "f32")),
                (("i32",), ()),
            ],
            imports=[
                {"module": "env", "name": "fn_all", "kind": "func", "type_index": 0},
                {
                    "module": "env",
                    "name": "tbl64",
                    "kind": "table",
                    "elem_type": "funcref",
                    "min_size": 10,
                    "max_size": 0x1_0000_0000,
                    "is_table64": True,
                },
                {
                    "module": "env",
                    "name": "mem",
                    "kind": "memory",
                    "min_pages": 4,
                    "max_pages": 64,
                    "is_shared": True,
                    "is_memory64": True,
                },
                {
                    "module": "env",
                    "name": "stack_ptr",
                    "kind": "global",
                    "val_type": "i64",
                    "mutable": True,
                },
                {
                    "module": "env",
                    "name": "cpp_exn",
                    "kind": "tag",
                    "attribute": 0,
                    "type_index": 1,
                },
            ],
            function_types=[1],
            memories=[{"min_pages": 2, "max_pages": None, "is_shared": False, "is_memory64": False}],
            exports=[
                ("run", "func", 1),
                ("tbl", "table", 0),
                ("memory", "memory", 0),
                ("sp", "global", 0),
                ("exn", "tag", 0),
            ],
            custom_sections=[
                ("name", b"\x01\x02\x00\x00"),
                ("producers", b"demo"),
            ],
            target_features=["+memory64", "+simd128", "-tail-call"],
        )
        # Append an opaque standard section (e.g., data_count id=12, tag id=13) to verify safe skipping.
        raw += wasm_check.encode_section(wasm_check.SECTION_DATA_COUNT, b"\x00")
        raw += wasm_check.encode_section(wasm_check.SECTION_TAG, b"\x00")

        module = wasm_check.parse_wasm_module(raw)

        self.assertEqual(
            module.types[0].params,
            ("i32", "i64", "f32", "f64", "v128", "funcref", "externref"),
        )
        self.assertEqual(module.types[0].results, ("i64", "f32"))
        self.assertEqual(
            module.types[0].format(),
            "(i32, i64, f32, f64, v128, funcref, externref) -> (i64, f32)",
        )

        by_kind = {imp.kind: imp for imp in module.imports}
        self.assertEqual(set(by_kind.keys()), {"func", "table", "memory", "global", "tag"})
        self.assertEqual(by_kind["func"].func_type, module.types[0])
        self.assertTrue(by_kind["table"].table.is_table64)
        self.assertEqual(by_kind["table"].table.max_size, 0x1_0000_0000)
        self.assertTrue(by_kind["global"].global_type.mutable)
        self.assertEqual(by_kind["global"].global_type.val_type, "i64")
        self.assertEqual(by_kind["tag"].func_type, module.types[1])

        self.assertEqual(len(module.all_memories), 2)
        self.assertEqual(module.all_memories[0].source, "import")
        self.assertEqual(module.all_memories[1].source, "defined")
        self.assertFalse(module.all_memories[1].has_max)

        self.assertEqual(
            [(e.name, e.kind, e.index) for e in module.exports],
            [
                ("run", "func", 1),
                ("tbl", "table", 0),
                ("memory", "memory", 0),
                ("sp", "global", 0),
                ("exn", "tag", 0),
            ],
        )
        self.assertEqual(module.custom_sections, ("name", "producers", "target_features"))
        self.assertEqual(module.target_features, ("+memory64", "+simd128", "-tail-call"))
        self.assertEqual(module.function_type_indices, (1,))


class TestWasmAbiValidation(unittest.TestCase):
    def test_wasm32_only_binary_fails_require_memory64(self) -> None:
        raw = wasm_check.build_valid_runtime_wasm(
            is_memory64=False,
            is_shared=True,
            wasm32_host_sigs=False,
        )
        module = wasm_check.parse_wasm_module(raw)

        strict_report = wasm_check.validate_runtime_wasm(module, require_memory64=True)
        self.assertFalse(strict_report.ok)
        self.assertFalse(strict_report.is_memory64)
        self.assertTrue(strict_report.is_shared_memory)
        self.assertTrue(any("memory64" in err for err in strict_report.errors))

        inspect_report = wasm_check.validate_runtime_wasm(module, require_memory64=False)
        self.assertTrue(inspect_report.ok)
        self.assertFalse(inspect_report.is_memory64)
        self.assertTrue(any("memory64" in w for w in inspect_report.warnings))

    def test_non_shared_binary_fails_require_shared_memory(self) -> None:
        raw = wasm_check.build_valid_runtime_wasm(
            is_memory64=True,
            is_shared=False,
        )
        module = wasm_check.parse_wasm_module(raw)

        report = wasm_check.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
        )
        self.assertFalse(report.ok)
        self.assertTrue(report.is_memory64)
        self.assertFalse(report.is_shared_memory)
        self.assertTrue(any("shared memory" in err for err in report.errors))

    def test_shared_memory_without_max_emits_warning(self) -> None:
        raw = wasm_check.build_valid_runtime_wasm(
            is_memory64=True,
            is_shared=True,
            max_pages=None,
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(
            module,
            require_memory64=True,
            require_shared_memory=True,
            require_host_imports=True,
        )
        self.assertTrue(report.ok)
        self.assertTrue(any("without a maximum page limit" in w for w in report.warnings))

    def test_missing_host_imports_detected(self) -> None:
        raw = wasm_check.build_valid_runtime_wasm(
            omit_host_imports=("host_httpfs_read", "host_wgpu_submit"),
        )
        module = wasm_check.parse_wasm_module(raw)

        report = wasm_check.validate_runtime_wasm(module, require_host_imports=True)
        self.assertFalse(report.ok)
        self.assertFalse(report.all_host_imports_ok)
        self.assertEqual(report.missing_host_imports, ("host_httpfs_read", "host_wgpu_submit"))
        self.assertEqual(report.mismatched_host_imports, ())

        by_name = {h.name: h for h in report.host_imports}
        self.assertEqual(by_name["host_log"].status, "ok")
        self.assertEqual(by_name["host_httpfs_open"].status, "ok")
        self.assertEqual(by_name["host_httpfs_close"].status, "ok")
        self.assertEqual(by_name["host_httpfs_read"].status, "missing")
        self.assertEqual(by_name["host_wgpu_submit"].status, "missing")

    def test_wasm32_pointer_width_mismatch_detected(self) -> None:
        raw = wasm_check.build_valid_runtime_wasm(
            is_memory64=True,
            is_shared=True,
            wasm32_host_sigs=True,
        )
        module = wasm_check.parse_wasm_module(raw)

        report = wasm_check.validate_runtime_wasm(module, require_host_imports=True)
        self.assertFalse(report.ok)
        self.assertFalse(report.all_host_imports_ok)
        self.assertEqual(report.missing_host_imports, ())
        self.assertEqual(
            report.mismatched_host_imports,
            ("host_log", "host_httpfs_open", "host_httpfs_read", "host_wgpu_submit"),
        )

        by_name = {h.name: h for h in report.host_imports}
        for ptr_fn in ("host_log", "host_httpfs_open", "host_httpfs_read", "host_wgpu_submit"):
            self.assertEqual(by_name[ptr_fn].status, "width_mismatch")
            self.assertIn("width mismatch", by_name[ptr_fn].detail)
        # host_httpfs_close takes only (handle: i32) -> () on both Wasm32 and Wasm64.
        self.assertEqual(by_name["host_httpfs_close"].status, "ok")

    def test_signature_mismatch_wrong_kind_and_invalid_type_index(self) -> None:
        raw = wasm_check.build_wasm_module(
            types=[
                (("i64", "i64"), ("i64",)),  # wrong return type for host_httpfs_open (expects i32)
            ],
            imports=[
                {
                    "module": "env",
                    "name": "memory",
                    "kind": "memory",
                    "min_pages": 1,
                    "max_pages": 16,
                    "is_shared": True,
                    "is_memory64": True,
                },
                # Wrong kind: global instead of func
                {
                    "module": "env",
                    "name": "host_log",
                    "kind": "global",
                    "val_type": "i64",
                    "mutable": False,
                },
                # Wrong return type
                {"module": "env", "name": "host_httpfs_open", "kind": "func", "type_index": 0},
                # Out-of-range type index
                {"module": "env", "name": "host_httpfs_close", "kind": "func", "type_index": 99},
            ],
        )
        module = wasm_check.parse_wasm_module(raw)
        report = wasm_check.validate_runtime_wasm(module, require_host_imports=True)

        self.assertFalse(report.ok)
        by_name = {h.name: h for h in report.host_imports}
        self.assertEqual(by_name["host_log"].status, "wrong_kind")
        self.assertEqual(by_name["host_httpfs_open"].status, "signature_mismatch")
        self.assertEqual(by_name["host_httpfs_close"].status, "invalid_type_index")
        self.assertEqual(by_name["host_httpfs_read"].status, "missing")
        self.assertEqual(by_name["host_wgpu_submit"].status, "missing")


class TestWasmMalformedBinaries(unittest.TestCase):
    def test_empty_and_short_headers_raise_parse_error(self) -> None:
        for bad in (b"", b"\x00as", b"\x00asm\x01\x00\x00"):
            with self.assertRaises(wasm_check.WasmParseError):
                wasm_check.parse_wasm_module(bad)

    def test_bad_magic_and_bad_version(self) -> None:
        with self.assertRaisesRegex(wasm_check.WasmParseError, "invalid Wasm magic"):
            wasm_check.parse_wasm_module(b"\x7fELF\x01\x00\x00\x00")

        with self.assertRaisesRegex(wasm_check.WasmParseError, "unsupported Wasm version"):
            wasm_check.parse_wasm_module(b"\x00asm" + struct.pack("<I", 2))

    def test_truncated_section_payload_and_unknown_section_id(self) -> None:
        header = b"\x00asm\x01\x00\x00\x00"
        # Section id=1 claims 50 bytes, but only 2 follow.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "truncated"):
            wasm_check.parse_wasm_module(header + bytes([1, 50, 0, 0]))

        # Unknown section id=99.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "unknown section id"):
            wasm_check.parse_wasm_module(header + bytes([99, 0]))

        # Truncated LEB128 section size (high bit set at EOF).
        with self.assertRaisesRegex(wasm_check.WasmParseError, "LEB128"):
            wasm_check.parse_wasm_module(header + bytes([1, 0x80]))

    def test_overlong_and_overflowing_leb128(self) -> None:
        header = b"\x00asm\x01\x00\x00\x00"
        # Overlong u32 (6 continuation bytes) for section size.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "overlong LEB128"):
            wasm_check.parse_wasm_module(header + bytes([1, 0x80, 0x80, 0x80, 0x80, 0x80, 0x00]))

        # 5th byte of u32 has overflow bit 0x10 set.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "overflows 32 bits"):
            wasm_check.parse_wasm_module(header + bytes([1, 0x80, 0x80, 0x80, 0x80, 0x10]))

        # 10th byte of u64 memory64 min_pages has overflow bit 0x02 set.
        bad_mem64_payload = bytes([1, 0x04]) + bytes([0x80] * 9 + [0x02])
        with self.assertRaisesRegex(wasm_check.WasmParseError, "overflows 64 bits"):
            wasm_check.parse_wasm_module(header + wasm_check.encode_section(5, bad_mem64_payload))

    def test_trailing_bytes_and_invalid_fields_in_sections(self) -> None:
        header = b"\x00asm\x01\x00\x00\x00"
        # Type section count=0 with 1 extra trailing byte.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "trailing"):
            wasm_check.parse_wasm_module(header + wasm_check.encode_section(1, b"\x00\x99"))

        # Unsupported type form 0x5F.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "unsupported type"):
            wasm_check.parse_wasm_module(header + wasm_check.encode_section(1, b"\x01\x5f"))

        # Unknown valtype 0x55.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "unknown"):
            wasm_check.parse_wasm_module(
                header + wasm_check.encode_section(1, b"\x01\x60\x01\x55\x00")
            )

        # Invalid memory limits flags 0x08.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "unknown memory limits flags"):
            wasm_check.parse_wasm_module(header + wasm_check.encode_section(5, b"\x01\x08\x01"))

        # Invalid UTF-8 in custom section name.
        with self.assertRaisesRegex(wasm_check.WasmParseError, "invalid UTF-8"):
            wasm_check.parse_wasm_module(header + wasm_check.encode_section(0, b"\x02\xff\xfe"))

        # Unknown import kind 0x09.
        bad_import = b"\x01" + wasm_check.encode_name("env") + wasm_check.encode_name("x") + b"\x09"
        with self.assertRaisesRegex(wasm_check.WasmParseError, "unknown import"):
            wasm_check.parse_wasm_module(header + wasm_check.encode_section(2, bad_import))

        # Unknown export kind 0x09.
        bad_export = b"\x01" + wasm_check.encode_name("x") + b"\x09\x00"
        with self.assertRaisesRegex(wasm_check.WasmParseError, "unknown export"):
            wasm_check.parse_wasm_module(header + wasm_check.encode_section(7, bad_export))


class TestCheckWasmExportsCli(unittest.TestCase):
    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(REPO_ROOT / "tools" / "check_wasm_exports.py"), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_cli_json_round_trip_valid_and_failing_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            valid_path = tmp_dir / "runtime_valid.wasm"
            wasm32_path = tmp_dir / "runtime_wasm32.wasm"
            unshared_path = tmp_dir / "runtime_unshared.wasm"
            missing_path = tmp_dir / "runtime_missing_imports.wasm"
            malformed_path = tmp_dir / "corrupt.wasm"

            valid_path.write_bytes(wasm_check.build_valid_runtime_wasm())
            wasm32_path.write_bytes(
                wasm_check.build_valid_runtime_wasm(is_memory64=False, wasm32_host_sigs=True)
            )
            unshared_path.write_bytes(wasm_check.build_valid_runtime_wasm(is_shared=False))
            missing_path.write_bytes(
                wasm_check.build_valid_runtime_wasm(omit_host_imports=("host_log",))
            )
            malformed_path.write_bytes(b"\x00asm\x01\x00\x00\x00\x05\x10\x01")

            # 1) Valid Wasm64 + shared + host imports passes all flags.
            proc = self._run_cli(
                [
                    str(valid_path),
                    "--json",
                    "--require-memory64",
                    "--require-shared-memory",
                    "--require-host-imports",
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            payload = json.loads(proc.stdout)
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["abi_version"], 1)
            self.assertTrue(payload["is_memory64"])
            self.assertTrue(payload["is_shared_memory"])
            self.assertTrue(payload["all_host_imports_ok"])
            self.assertEqual(payload["missing_host_imports"], [])
            self.assertEqual(payload["mismatched_host_imports"], [])
            self.assertEqual(len(payload["host_imports"]), 5)

            # 2) Wasm32 fails --require-memory64.
            proc = self._run_cli([str(wasm32_path), "--json", "--require-memory64"])
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["is_memory64"])

            # 3) Non-shared fails --require-shared-memory.
            proc = self._run_cli([str(unshared_path), "--json", "--require-shared-memory"])
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertFalse(payload["is_shared_memory"])

            # 4) Missing host import fails --require-host-imports.
            proc = self._run_cli([str(missing_path), "--json", "--require-host-imports"])
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertEqual(payload["missing_host_imports"], ["host_log"])

            # 5) Malformed binary exits 1 with structured JSON error.
            proc = self._run_cli([str(malformed_path), "--json"])
            self.assertEqual(proc.returncode, 1)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])
            self.assertIn("parse_error", payload)

            # 6) Human-readable text output smoke test.
            proc = self._run_cli(
                [
                    str(valid_path),
                    "--require-memory64",
                    "--require-shared-memory",
                    "--require-host-imports",
                ]
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("Wasm ABI Validation [PASS]", proc.stdout)
            self.assertIn("Memory64:       YES", proc.stdout)
            self.assertIn("Shared Memory:  YES", proc.stdout)
            self.assertIn("env.host_wgpu_submit", proc.stdout)

            # 7) Non-existent file exits non-zero.
            proc = self._run_cli([str(tmp_dir / "does_not_exist.wasm"), "--json"])
            self.assertEqual(proc.returncode, 2)
            payload = json.loads(proc.stdout)
            self.assertFalse(payload["ok"])


if __name__ == "__main__":
    unittest.main()
