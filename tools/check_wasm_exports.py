#!/usr/bin/env python3
"""Pure-Python binary WebAssembly (.wasm) parser and Wasm64/threads/ABI validator.

Validates compiled `runtime.wasm` binaries against the frozen Wasm64 <-> JS host
ABI documented in `docs/wasm_abi.md` and `runtime/os/os_shim.cpp`, without
requiring external tools such as `wasm-objdump` or `wasm-tools`.

Pure Python 3.12 stdlib only. Usable both as an importable library and as a CLI:

    python tools/check_wasm_exports.py <runtime.wasm> [--json]
        [--require-memory64] [--require-shared-memory] [--require-host-imports]
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

ABI_VERSION = 1
WASM_MAGIC = b"\x00asm"
WASM_VERSION = 1
WASM_PAGE_SIZE = 65536

MEMORY_FLAG_HAS_MAX = 0x01
MEMORY_FLAG_SHARED = 0x02
MEMORY_FLAG_MEMORY64 = 0x04
MEMORY_FLAGS_KNOWN_MASK = MEMORY_FLAG_HAS_MAX | MEMORY_FLAG_SHARED | MEMORY_FLAG_MEMORY64

TABLE_FLAG_HAS_MAX = 0x01
TABLE_FLAG_TABLE64 = 0x04
TABLE_FLAGS_KNOWN_MASK = TABLE_FLAG_HAS_MAX | TABLE_FLAG_TABLE64

SECTION_CUSTOM = 0
SECTION_TYPE = 1
SECTION_IMPORT = 2
SECTION_FUNCTION = 3
SECTION_TABLE = 4
SECTION_MEMORY = 5
SECTION_GLOBAL = 6
SECTION_EXPORT = 7
SECTION_START = 8
SECTION_ELEMENT = 9
SECTION_CODE = 10
SECTION_DATA = 11
SECTION_DATA_COUNT = 12
SECTION_TAG = 13

SECTION_NAMES: dict[int, str] = {
    SECTION_CUSTOM: "custom",
    SECTION_TYPE: "type",
    SECTION_IMPORT: "import",
    SECTION_FUNCTION: "function",
    SECTION_TABLE: "table",
    SECTION_MEMORY: "memory",
    SECTION_GLOBAL: "global",
    SECTION_EXPORT: "export",
    SECTION_START: "start",
    SECTION_ELEMENT: "element",
    SECTION_CODE: "code",
    SECTION_DATA: "data",
    SECTION_DATA_COUNT: "data_count",
    SECTION_TAG: "tag",
}

VALTYPE_NAMES: dict[int, str] = {
    0x7F: "i32",
    0x7E: "i64",
    0x7D: "f32",
    0x7C: "f64",
    0x7B: "v128",
    0x70: "funcref",
    0x6F: "externref",
    # Reference/GC/EH types from newer proposals (safe forward compatibility).
    0x73: "nofuncref",
    0x72: "noexternref",
    0x71: "noneref",
    0x6E: "anyref",
    0x6D: "eqref",
    0x6C: "i31ref",
    0x6B: "structref",
    0x6A: "arrayref",
    0x69: "exnref",
}

VALTYPE_CODES: dict[str, int] = {name: code for code, name in VALTYPE_NAMES.items()}

REFTYPE_CODES: set[int] = {
    0x70,  # funcref
    0x6F,  # externref
    0x73,
    0x72,
    0x71,
    0x6E,
    0x6D,
    0x6C,
    0x6B,
    0x6A,
    0x69,
    0x63,  # (ref null ht)
    0x64,  # (ref ht)
}

IMPORT_EXPORT_KINDS: dict[int, str] = {
    0x00: "func",
    0x01: "table",
    0x02: "memory",
    0x03: "global",
    0x04: "tag",
}

KIND_TO_CODE: dict[str, int] = {name: code for code, name in IMPORT_EXPORT_KINDS.items()}


class WasmParseError(ValueError):
    """Raised when a WebAssembly binary is truncated, malformed, or unsupported."""

    def __init__(self, message: str, *, offset: int | None = None) -> None:
        self.message = message
        self.offset = offset
        full = f"{message} (at offset 0x{offset:x})" if offset is not None else message
        super().__init__(full)


# ---------------------------------------------------------------------------
# Parsed Wasm module data structures
# ---------------------------------------------------------------------------


def format_signature(params: Sequence[str], results: Sequence[str]) -> str:
    """Format a Wasm function signature matching `docs/wasm_abi.md` style."""
    param_str = ", ".join(params)
    if not results:
        result_str = "()"
    elif len(results) == 1:
        result_str = results[0]
    else:
        result_str = f"({', '.join(results)})"
    return f"({param_str}) -> {result_str}"


@dataclass(frozen=True)
class FuncType:
    """WebAssembly function type signature."""

    params: tuple[str, ...]
    results: tuple[str, ...]

    def format(self) -> str:
        return format_signature(self.params, self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": list(self.params),
            "results": list(self.results),
            "signature": self.format(),
        }


@dataclass(frozen=True)
class MemoryLimits:
    """Decoded limits for an imported or defined WebAssembly linear memory."""

    flags: int
    is_memory64: bool
    is_shared: bool
    has_max: bool
    min_pages: int
    max_pages: int | None
    source: str  # "import" or "defined"
    module: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "source": self.source,
            "flags": self.flags,
            "is_memory64": self.is_memory64,
            "is_shared": self.is_shared,
            "has_max": self.has_max,
            "min_pages": self.min_pages,
            "max_pages": self.max_pages,
            "min_bytes": self.min_pages * WASM_PAGE_SIZE,
            "max_bytes": (
                self.max_pages * WASM_PAGE_SIZE if self.max_pages is not None else None
            ),
        }
        if self.module is not None:
            out["module"] = self.module
        if self.name is not None:
            out["name"] = self.name
        return out


@dataclass(frozen=True)
class TableLimits:
    """Decoded limits for an imported WebAssembly table."""

    elem_type: str
    flags: int
    is_table64: bool
    has_max: bool
    min_size: int
    max_size: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "elem_type": self.elem_type,
            "flags": self.flags,
            "is_table64": self.is_table64,
            "has_max": self.has_max,
            "min_size": self.min_size,
            "max_size": self.max_size,
        }


@dataclass(frozen=True)
class GlobalType:
    """Decoded type and mutability for an imported WebAssembly global."""

    val_type: str
    mutable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "val_type": self.val_type,
            "mutable": self.mutable,
        }


@dataclass(frozen=True)
class WasmImport:
    """Decoded entry from the Import section (id=2)."""

    module: str
    name: str
    kind: str
    type_index: int | None = None
    func_type: FuncType | None = None
    memory: MemoryLimits | None = None
    table: TableLimits | None = None
    global_type: GlobalType | None = None
    tag_attribute: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "module": self.module,
            "name": self.name,
            "kind": self.kind,
        }
        if self.type_index is not None:
            out["type_index"] = self.type_index
        if self.func_type is not None:
            out["signature"] = self.func_type.format()
            out["params"] = list(self.func_type.params)
            out["results"] = list(self.func_type.results)
        if self.memory is not None:
            out["memory"] = self.memory.to_dict()
        if self.table is not None:
            out["table"] = self.table.to_dict()
        if self.global_type is not None:
            out["global_type"] = self.global_type.to_dict()
        if self.tag_attribute is not None:
            out["tag_attribute"] = self.tag_attribute
        return out


@dataclass(frozen=True)
class WasmExport:
    """Decoded entry from the Export section (id=7)."""

    name: str
    kind: str
    index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "index": self.index,
        }


@dataclass(frozen=True)
class SectionInfo:
    """Metadata for a single WebAssembly section."""

    id: int
    name: str
    offset: int
    size: int
    custom_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "offset": self.offset,
            "size": self.size,
        }
        if self.custom_name is not None:
            out["custom_name"] = self.custom_name
        return out


@dataclass(frozen=True)
class WasmModule:
    """Parsed WebAssembly binary module structure."""

    version: int
    byte_size: int
    types: tuple[FuncType, ...]
    imports: tuple[WasmImport, ...]
    function_type_indices: tuple[int, ...]
    memories: tuple[MemoryLimits, ...]
    exports: tuple[WasmExport, ...]
    custom_sections: tuple[str, ...]
    target_features: tuple[str, ...]
    sections: tuple[SectionInfo, ...]

    @property
    def all_memories(self) -> tuple[MemoryLimits, ...]:
        """All memories in index order: imported memories followed by defined memories."""
        imported = tuple(
            imp.memory for imp in self.imports if imp.kind == "memory" and imp.memory is not None
        )
        return imported + self.memories

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "byte_size": self.byte_size,
            "types": [t.to_dict() for t in self.types],
            "imports": [imp.to_dict() for imp in self.imports],
            "memories": [m.to_dict() for m in self.all_memories],
            "defined_memories": [m.to_dict() for m in self.memories],
            "exports": [exp.to_dict() for exp in self.exports],
            "custom_sections": list(self.custom_sections),
            "target_features": list(self.target_features),
            "sections": [sec.to_dict() for sec in self.sections],
        }


# ---------------------------------------------------------------------------
# Bounds-checked binary reader
# ---------------------------------------------------------------------------


class _BinaryReader:
    """Bounds-checked cursor over a slice of a WebAssembly binary."""

    __slots__ = ("_buf", "_pos", "_end")

    def __init__(self, data: bytes | bytearray | memoryview, start: int = 0, end: int | None = None) -> None:
        self._buf = memoryview(data)
        self._pos = start
        self._end = len(self._buf) if end is None else end

    @property
    def pos(self) -> int:
        return self._pos

    @property
    def end(self) -> int:
        return self._end

    @property
    def remaining(self) -> int:
        return self._end - self._pos

    def subreader(self, length: int, *, context: str) -> _BinaryReader:
        if length < 0 or self._pos + length > self._end:
            raise WasmParseError(
                f"truncated {context}: need {length} bytes, only {self.remaining} available",
                offset=self._pos,
            )
        start = self._pos
        self._pos += length
        return _BinaryReader(self._buf, start, start + length)

    def read_bytes(self, count: int, *, context: str = "data") -> bytes:
        if count < 0 or self._pos + count > self._end:
            raise WasmParseError(
                f"unexpected EOF reading {context}: need {count} bytes, only {self.remaining} available",
                offset=self._pos,
            )
        start = self._pos
        self._pos += count
        return bytes(self._buf[start : self._pos])

    def read_u8(self, *, context: str = "byte") -> int:
        if self._pos >= self._end:
            raise WasmParseError(f"unexpected EOF reading {context}", offset=self._pos)
        value = self._buf[self._pos]
        self._pos += 1
        return value

    def read_uleb128(self, *, max_bits: int, context: str = "u32") -> int:
        start_offset = self._pos
        result = 0
        shift = 0
        max_bytes = (max_bits + 6) // 7

        for byte_index in range(max_bytes):
            if self._pos >= self._end:
                raise WasmParseError(
                    f"unexpected EOF while decoding LEB128 {context}",
                    offset=start_offset,
                )
            byte = self._buf[self._pos]
            self._pos += 1

            is_last_allowed = byte_index == max_bytes - 1
            if is_last_allowed:
                if byte & 0x80:
                    raise WasmParseError(
                        f"overlong LEB128 {context} (exceeds {max_bytes} bytes)",
                        offset=start_offset,
                    )
                remaining_bits = max_bits - shift
                overflow_mask = ~((1 << remaining_bits) - 1) & 0x7F
                if byte & overflow_mask:
                    raise WasmParseError(
                        f"LEB128 {context} overflows {max_bits} bits",
                        offset=start_offset,
                    )

            result |= (byte & 0x7F) << shift
            if (byte & 0x80) == 0:
                return result
            shift += 7

        raise WasmParseError(f"overlong LEB128 {context}", offset=start_offset)

    def read_sleb128(self, *, max_bits: int, context: str = "s33") -> int:
        start_offset = self._pos
        result = 0
        shift = 0
        max_bytes = (max_bits + 6) // 7

        for byte_index in range(max_bytes):
            if self._pos >= self._end:
                raise WasmParseError(
                    f"unexpected EOF while decoding signed LEB128 {context}",
                    offset=start_offset,
                )
            byte = self._buf[self._pos]
            self._pos += 1

            is_last_allowed = byte_index == max_bytes - 1
            if is_last_allowed and (byte & 0x80):
                raise WasmParseError(
                    f"overlong signed LEB128 {context}",
                    offset=start_offset,
                )

            result |= (byte & 0x7F) << shift
            shift += 7
            if (byte & 0x80) == 0:
                if shift < max_bits and (byte & 0x40):
                    result |= -(1 << shift)
                elif byte & 0x40:
                    result -= 1 << shift
                return result

        raise WasmParseError(f"overlong signed LEB128 {context}", offset=start_offset)

    def read_u32(self, *, context: str = "u32") -> int:
        return self.read_uleb128(max_bits=32, context=context)

    def read_u64(self, *, context: str = "u64") -> int:
        return self.read_uleb128(max_bits=64, context=context)

    def read_name(self, *, context: str = "UTF-8 name") -> str:
        start_offset = self._pos
        length = self.read_u32(context=f"{context} length")
        raw = self.read_bytes(length, context=context)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WasmParseError(
                f"invalid UTF-8 in {context}: {exc}",
                offset=start_offset,
            ) from exc

    def read_valtype(self, *, context: str = "valtype") -> str:
        start_offset = self._pos
        code = self.read_u8(context=context)
        if code in VALTYPE_NAMES:
            return VALTYPE_NAMES[code]
        if code in (0x63, 0x64):
            heap_type = self.read_sleb128(max_bits=33, context="heaptype")
            nullable = "null " if code == 0x63 else ""
            if heap_type < 0:
                abs_code = heap_type & 0x7F
                abs_name = VALTYPE_NAMES.get(abs_code, f"0x{abs_code:02x}")
                return f"(ref {nullable}{abs_name})"
            return f"(ref {nullable}{heap_type})"
        raise WasmParseError(f"unknown {context} byte 0x{code:02x}", offset=start_offset)

    def read_reftype(self, *, context: str = "reftype") -> str:
        start_offset = self._pos
        if self._pos >= self._end:
            raise WasmParseError(f"unexpected EOF reading {context}", offset=start_offset)
        peek = self._buf[self._pos]
        if peek not in REFTYPE_CODES:
            raise WasmParseError(f"invalid {context} byte 0x{peek:02x}", offset=start_offset)
        return self.read_valtype(context=context)

    def read_memory_limits(
        self,
        *,
        source: str,
        module: str | None = None,
        name: str | None = None,
    ) -> MemoryLimits:
        start_offset = self._pos
        flags = self.read_u8(context="memory limits flags")
        if flags & ~MEMORY_FLAGS_KNOWN_MASK:
            raise WasmParseError(
                f"unknown memory limits flags 0x{flags:02x}",
                offset=start_offset,
            )
        has_max = bool(flags & MEMORY_FLAG_HAS_MAX)
        is_shared = bool(flags & MEMORY_FLAG_SHARED)
        is_memory64 = bool(flags & MEMORY_FLAG_MEMORY64)

        if is_memory64:
            min_pages = self.read_u64(context="memory64 min_pages")
            max_pages = self.read_u64(context="memory64 max_pages") if has_max else None
        else:
            min_pages = self.read_u32(context="memory32 min_pages")
            max_pages = self.read_u32(context="memory32 max_pages") if has_max else None

        return MemoryLimits(
            flags=flags,
            is_memory64=is_memory64,
            is_shared=is_shared,
            has_max=has_max,
            min_pages=min_pages,
            max_pages=max_pages,
            source=source,
            module=module,
            name=name,
        )

    def read_table_limits(self) -> TableLimits:
        elem_type = self.read_reftype(context="table element type")
        start_offset = self._pos
        flags = self.read_u8(context="table limits flags")
        if flags & ~TABLE_FLAGS_KNOWN_MASK:
            raise WasmParseError(
                f"unknown table limits flags 0x{flags:02x}",
                offset=start_offset,
            )
        has_max = bool(flags & TABLE_FLAG_HAS_MAX)
        is_table64 = bool(flags & TABLE_FLAG_TABLE64)
        if is_table64:
            min_size = self.read_u64(context="table64 min")
            max_size = self.read_u64(context="table64 max") if has_max else None
        else:
            min_size = self.read_u32(context="table32 min")
            max_size = self.read_u32(context="table32 max") if has_max else None

        return TableLimits(
            elem_type=elem_type,
            flags=flags,
            is_table64=is_table64,
            has_max=has_max,
            min_size=min_size,
            max_size=max_size,
        )


# ---------------------------------------------------------------------------
# Section decoders
# ---------------------------------------------------------------------------


def _parse_type_section(reader: _BinaryReader) -> list[FuncType]:
    count = reader.read_u32(context="type count")
    types: list[FuncType] = []
    for idx in range(count):
        form_offset = reader.pos
        form = reader.read_u8(context=f"type[{idx}] form")
        if form != 0x60:
            raise WasmParseError(
                f"unsupported type[{idx}] form 0x{form:02x} (expected func 0x60)",
                offset=form_offset,
            )
        param_count = reader.read_u32(context=f"type[{idx}] param count")
        params = tuple(
            reader.read_valtype(context=f"type[{idx}] param[{p}]") for p in range(param_count)
        )
        result_count = reader.read_u32(context=f"type[{idx}] result count")
        results = tuple(
            reader.read_valtype(context=f"type[{idx}] result[{r}]") for r in range(result_count)
        )
        types.append(FuncType(params=params, results=results))

    if reader.remaining != 0:
        raise WasmParseError(
            f"trailing {reader.remaining} byte(s) in type section",
            offset=reader.pos,
        )
    return types


def _parse_import_section(reader: _BinaryReader) -> list[WasmImport]:
    count = reader.read_u32(context="import count")
    imports: list[WasmImport] = []
    for idx in range(count):
        module = reader.read_name(context=f"import[{idx}] module")
        name = reader.read_name(context=f"import[{idx}] field")
        kind_offset = reader.pos
        kind_code = reader.read_u8(context=f"import[{idx}] kind")
        kind = IMPORT_EXPORT_KINDS.get(kind_code)
        if kind is None:
            raise WasmParseError(
                f"unknown import[{idx}] kind 0x{kind_code:02x}",
                offset=kind_offset,
            )

        if kind == "func":
            type_index = reader.read_u32(context=f"import[{idx}] func type_index")
            imports.append(
                WasmImport(module=module, name=name, kind="func", type_index=type_index)
            )
        elif kind == "table":
            table_limits = reader.read_table_limits()
            imports.append(
                WasmImport(module=module, name=name, kind="table", table=table_limits)
            )
        elif kind == "memory":
            mem_limits = reader.read_memory_limits(source="import", module=module, name=name)
            imports.append(
                WasmImport(module=module, name=name, kind="memory", memory=mem_limits)
            )
        elif kind == "global":
            val_type = reader.read_valtype(context=f"import[{idx}] global valtype")
            mut_offset = reader.pos
            mut_byte = reader.read_u8(context=f"import[{idx}] global mutability")
            if mut_byte not in (0x00, 0x01):
                raise WasmParseError(
                    f"invalid global mutability 0x{mut_byte:02x}",
                    offset=mut_offset,
                )
            imports.append(
                WasmImport(
                    module=module,
                    name=name,
                    kind="global",
                    global_type=GlobalType(val_type=val_type, mutable=bool(mut_byte)),
                )
            )
        elif kind == "tag":
            attr_offset = reader.pos
            attr = reader.read_u8(context=f"import[{idx}] tag attribute")
            if attr != 0x00:
                raise WasmParseError(
                    f"unsupported tag attribute 0x{attr:02x}",
                    offset=attr_offset,
                )
            type_index = reader.read_u32(context=f"import[{idx}] tag type_index")
            imports.append(
                WasmImport(
                    module=module,
                    name=name,
                    kind="tag",
                    type_index=type_index,
                    tag_attribute=attr,
                )
            )

    if reader.remaining != 0:
        raise WasmParseError(
            f"trailing {reader.remaining} byte(s) in import section",
            offset=reader.pos,
        )
    return imports


def _parse_function_section(reader: _BinaryReader) -> list[int]:
    count = reader.read_u32(context="function count")
    indices = [reader.read_u32(context=f"function[{idx}] type_index") for idx in range(count)]
    if reader.remaining != 0:
        raise WasmParseError(
            f"trailing {reader.remaining} byte(s) in function section",
            offset=reader.pos,
        )
    return indices


def _parse_memory_section(reader: _BinaryReader) -> list[MemoryLimits]:
    count = reader.read_u32(context="memory count")
    memories = [reader.read_memory_limits(source="defined") for _ in range(count)]
    if reader.remaining != 0:
        raise WasmParseError(
            f"trailing {reader.remaining} byte(s) in memory section",
            offset=reader.pos,
        )
    return memories


def _parse_export_section(reader: _BinaryReader) -> list[WasmExport]:
    count = reader.read_u32(context="export count")
    exports: list[WasmExport] = []
    for idx in range(count):
        name = reader.read_name(context=f"export[{idx}] name")
        kind_offset = reader.pos
        kind_code = reader.read_u8(context=f"export[{idx}] kind")
        kind = IMPORT_EXPORT_KINDS.get(kind_code)
        if kind is None:
            raise WasmParseError(
                f"unknown export[{idx}] kind 0x{kind_code:02x}",
                offset=kind_offset,
            )
        index = reader.read_u32(context=f"export[{idx}] index")
        exports.append(WasmExport(name=name, kind=kind, index=index))

    if reader.remaining != 0:
        raise WasmParseError(
            f"trailing {reader.remaining} byte(s) in export section",
            offset=reader.pos,
        )
    return exports


def _try_parse_target_features(reader: _BinaryReader) -> list[str]:
    """Best-effort decode of LLVM's `target_features` custom section."""
    try:
        count = reader.read_u32(context="target_features count")
        features: list[str] = []
        for _ in range(count):
            prefix = reader.read_u8(context="target_feature prefix")
            if prefix not in (ord("+"), ord("-"), ord("=")):
                return []
            feat_name = reader.read_name(context="target_feature name")
            features.append(f"{chr(prefix)}{feat_name}")
        if reader.remaining != 0:
            return []
        return features
    except WasmParseError:
        return []


def parse_wasm_module(data: bytes | bytearray | memoryview) -> WasmModule:
    """Parse a binary WebAssembly module without external tools.

    Strictly bounds-checks header magic/version, LEB128 values, and section
    payloads; raises `WasmParseError` on truncated or malformed binaries.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise WasmParseError(f"expected bytes-like input, got {type(data).__name__}")

    total_len = len(data)
    if total_len < 8:
        raise WasmParseError(
            f"Wasm binary too short: expected at least 8 header bytes, got {total_len}",
            offset=0,
        )

    reader = _BinaryReader(data)
    magic = reader.read_bytes(4, context="Wasm magic")
    if magic != WASM_MAGIC:
        raise WasmParseError(
            f"invalid Wasm magic {magic!r} (expected {WASM_MAGIC!r})",
            offset=0,
        )

    (version,) = struct.unpack("<I", reader.read_bytes(4, context="Wasm version"))
    if version != WASM_VERSION:
        raise WasmParseError(
            f"unsupported Wasm version {version} (expected {WASM_VERSION})",
            offset=4,
        )

    types: list[FuncType] = []
    raw_imports: list[WasmImport] = []
    func_type_indices: list[int] = []
    memories: list[MemoryLimits] = []
    exports: list[WasmExport] = []
    custom_sections: list[str] = []
    target_features: list[str] = []
    sections: list[SectionInfo] = []

    while reader.remaining > 0:
        sec_offset = reader.pos
        sec_id = reader.read_u8(context="section id")
        sec_name = SECTION_NAMES.get(sec_id)
        if sec_name is None:
            raise WasmParseError(f"unknown section id {sec_id}", offset=sec_offset)

        sec_size = reader.read_u32(context=f"{sec_name} section size")
        sec_reader = reader.subreader(sec_size, context=f"{sec_name} section (id={sec_id})")

        custom_name: str | None = None
        if sec_id == SECTION_CUSTOM:
            custom_name = sec_reader.read_name(context="custom section name")
            custom_sections.append(custom_name)
            if custom_name == "target_features":
                target_features.extend(_try_parse_target_features(sec_reader))
        elif sec_id == SECTION_TYPE:
            types.extend(_parse_type_section(sec_reader))
        elif sec_id == SECTION_IMPORT:
            raw_imports.extend(_parse_import_section(sec_reader))
        elif sec_id == SECTION_FUNCTION:
            func_type_indices.extend(_parse_function_section(sec_reader))
        elif sec_id == SECTION_MEMORY:
            memories.extend(_parse_memory_section(sec_reader))
        elif sec_id == SECTION_EXPORT:
            exports.extend(_parse_export_section(sec_reader))
        # Other valid standard sections (table, global, start, element, code,
        # data, data_count, tag) have already been bounds-checked via subreader.

        sections.append(
            SectionInfo(
                id=sec_id,
                name=sec_name,
                offset=sec_offset,
                size=sec_size,
                custom_name=custom_name,
            )
        )

    # Resolve function/tag import signatures against the Type section.
    resolved_imports: list[WasmImport] = []
    for imp in raw_imports:
        if imp.kind in ("func", "tag") and imp.type_index is not None:
            resolved_type = types[imp.type_index] if 0 <= imp.type_index < len(types) else None
            resolved_imports.append(
                WasmImport(
                    module=imp.module,
                    name=imp.name,
                    kind=imp.kind,
                    type_index=imp.type_index,
                    func_type=resolved_type,
                    memory=imp.memory,
                    table=imp.table,
                    global_type=imp.global_type,
                    tag_attribute=imp.tag_attribute,
                )
            )
        else:
            resolved_imports.append(imp)

    return WasmModule(
        version=version,
        byte_size=total_len,
        types=tuple(types),
        imports=tuple(resolved_imports),
        function_type_indices=tuple(func_type_indices),
        memories=tuple(memories),
        exports=tuple(exports),
        custom_sections=tuple(custom_sections),
        target_features=tuple(target_features),
        sections=tuple(sections),
    )


# ---------------------------------------------------------------------------
# ABI specification (`docs/wasm_abi.md` + `runtime/os/os_shim.cpp`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostImportSpec:
    """Expected `env.host_*` ABI import signature from `docs/wasm_abi.md`."""

    module: str
    name: str
    params_wasm64: tuple[str, ...]
    results_wasm64: tuple[str, ...]
    params_wasm32: tuple[str, ...]
    results_wasm32: tuple[str, ...]
    pointer_param_indices: tuple[int, ...] = ()
    purpose: str = ""

    @property
    def expected_signature(self) -> str:
        return format_signature(self.params_wasm64, self.results_wasm64)

    @property
    def wasm32_signature(self) -> str:
        return format_signature(self.params_wasm32, self.results_wasm32)


REQUIRED_HOST_IMPORTS: tuple[HostImportSpec, ...] = (
    HostImportSpec(
        module="env",
        name="host_log",
        params_wasm64=("i64", "i64"),
        results_wasm64=(),
        params_wasm32=("i32", "i32"),
        results_wasm32=(),
        pointer_param_indices=(0, 1),
        purpose="UTF-8 log line to console / progress channel",
    ),
    HostImportSpec(
        module="env",
        name="host_httpfs_open",
        params_wasm64=("i64", "i64"),
        results_wasm64=("i32",),
        params_wasm32=("i32", "i32"),
        results_wasm32=("i32",),
        pointer_param_indices=(0, 1),
        purpose="Open relative /data/ path; returns handle >= 0 or -1",
    ),
    HostImportSpec(
        module="env",
        name="host_httpfs_read",
        params_wasm64=("i32", "i64", "i64", "i64"),
        results_wasm64=("i64",),
        params_wasm32=("i32", "i64", "i32", "i32"),
        results_wasm32=("i64",),
        pointer_param_indices=(2, 3),
        purpose="Range read via io_worker.js",
    ),
    HostImportSpec(
        module="env",
        name="host_httpfs_close",
        params_wasm64=("i32",),
        results_wasm64=(),
        params_wasm32=("i32",),
        results_wasm32=(),
        pointer_param_indices=(),
        purpose="Close virtual file handle",
    ),
    HostImportSpec(
        module="env",
        name="host_wgpu_submit",
        params_wasm64=("i64", "i64"),
        results_wasm64=(),
        params_wasm32=("i32", "i32"),
        results_wasm32=(),
        pointer_param_indices=(0, 1),
        purpose="Forward binary command buffer to wgpu_worker.js",
    ),
)


@dataclass(frozen=True)
class HostImportCheck:
    """Validation result for a single required `env.host_*` ABI import."""

    module: str
    name: str
    present: bool
    ok: bool
    status: str  # "ok", "missing", "wrong_kind", "invalid_type_index", "width_mismatch", "signature_mismatch"
    expected_signature: str
    actual_signature: str | None
    actual_kind: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "name": self.name,
            "present": self.present,
            "ok": self.ok,
            "status": self.status,
            "expected_signature": self.expected_signature,
            "actual_signature": self.actual_signature,
            "actual_kind": self.actual_kind,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Structured Wasm64 / threads / host ABI validation report."""

    ok: bool
    abi_version: int
    path: str | None
    byte_size: int
    wasm_version: int
    is_memory64: bool
    is_shared_memory: bool
    memories: tuple[MemoryLimits, ...]
    host_imports: tuple[HostImportCheck, ...]
    missing_host_imports: tuple[str, ...]
    mismatched_host_imports: tuple[str, ...]
    all_host_imports_ok: bool
    imports_count: int
    exports_count: int
    exports: tuple[WasmExport, ...]
    custom_sections: tuple[str, ...]
    target_features: tuple[str, ...]
    required_checks: dict[str, bool]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "abi_version": self.abi_version,
            "path": self.path,
            "byte_size": self.byte_size,
            "wasm_version": self.wasm_version,
            "is_memory64": self.is_memory64,
            "is_shared_memory": self.is_shared_memory,
            "memories": [m.to_dict() for m in self.memories],
            "host_imports": [h.to_dict() for h in self.host_imports],
            "missing_host_imports": list(self.missing_host_imports),
            "mismatched_host_imports": list(self.mismatched_host_imports),
            "all_host_imports_ok": self.all_host_imports_ok,
            "imports_count": self.imports_count,
            "exports_count": self.exports_count,
            "exports": [e.to_dict() for e in self.exports],
            "custom_sections": list(self.custom_sections),
            "target_features": list(self.target_features),
            "required_checks": dict(self.required_checks),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _check_single_host_import(
    spec: HostImportSpec,
    matches: list[WasmImport],
) -> HostImportCheck:
    if not matches:
        return HostImportCheck(
            module=spec.module,
            name=spec.name,
            present=False,
            ok=False,
            status="missing",
            expected_signature=spec.expected_signature,
            actual_signature=None,
            actual_kind=None,
            detail=f"missing required import {spec.module}.{spec.name} {spec.expected_signature}",
        )

    # Prefer a function import if multiple entries share the same name.
    imp = next((m for m in matches if m.kind == "func"), matches[0])
    if imp.kind != "func":
        return HostImportCheck(
            module=spec.module,
            name=spec.name,
            present=True,
            ok=False,
            status="wrong_kind",
            expected_signature=spec.expected_signature,
            actual_signature=None,
            actual_kind=imp.kind,
            detail=(
                f"{spec.module}.{spec.name} imported as {imp.kind!r}, "
                f"expected 'func' {spec.expected_signature}"
            ),
        )

    if imp.func_type is None:
        return HostImportCheck(
            module=spec.module,
            name=spec.name,
            present=True,
            ok=False,
            status="invalid_type_index",
            expected_signature=spec.expected_signature,
            actual_signature=None,
            actual_kind="func",
            detail=(
                f"{spec.module}.{spec.name} references out-of-range type index "
                f"{imp.type_index}"
            ),
        )

    actual = imp.func_type
    actual_sig = actual.format()

    if actual.params == spec.params_wasm64 and actual.results == spec.results_wasm64:
        return HostImportCheck(
            module=spec.module,
            name=spec.name,
            present=True,
            ok=True,
            status="ok",
            expected_signature=spec.expected_signature,
            actual_signature=actual_sig,
            actual_kind="func",
            detail="matches Wasm64 ABI signature",
        )

    # Detect Wasm32 vs Wasm64 pointer/size_t width mismatch (i32 where i64 expected).
    is_exact_wasm32 = (
        spec.params_wasm32 != spec.params_wasm64
        and actual.params == spec.params_wasm32
        and actual.results == spec.results_wasm32
    )
    has_pointer_width_mismatch = (
        len(actual.params) == len(spec.params_wasm64)
        and actual.results == spec.results_wasm64
        and bool(spec.pointer_param_indices)
        and any(
            actual.params[i] == "i32" and spec.params_wasm64[i] == "i64"
            for i in spec.pointer_param_indices
        )
        and all(
            actual.params[i] == spec.params_wasm64[i]
            for i in range(len(spec.params_wasm64))
            if i not in spec.pointer_param_indices
        )
    )

    if is_exact_wasm32 or has_pointer_width_mismatch:
        return HostImportCheck(
            module=spec.module,
            name=spec.name,
            present=True,
            ok=False,
            status="width_mismatch",
            expected_signature=spec.expected_signature,
            actual_signature=actual_sig,
            actual_kind="func",
            detail=(
                f"pointer/length width mismatch for {spec.module}.{spec.name}: "
                f"got Wasm32-width {actual_sig}, expected Wasm64 {spec.expected_signature}"
            ),
        )

    return HostImportCheck(
        module=spec.module,
        name=spec.name,
        present=True,
        ok=False,
        status="signature_mismatch",
        expected_signature=spec.expected_signature,
        actual_signature=actual_sig,
        actual_kind="func",
        detail=(
            f"signature mismatch for {spec.module}.{spec.name}: "
            f"got {actual_sig}, expected {spec.expected_signature}"
        ),
    )


def validate_runtime_wasm(
    module: WasmModule,
    *,
    require_memory64: bool = False,
    require_shared_memory: bool = False,
    require_host_imports: bool = False,
    expected_host_imports: Sequence[HostImportSpec] = REQUIRED_HOST_IMPORTS,
    path: str | None = None,
) -> ValidationReport:
    """Validate a parsed `WasmModule` against Wasm64/threads/host ABI requirements."""
    all_memories = module.all_memories
    is_memory64 = any(m.is_memory64 for m in all_memories)
    is_shared_memory = any(m.is_shared for m in all_memories)

    imports_by_key: dict[tuple[str, str], list[WasmImport]] = {}
    for imp in module.imports:
        imports_by_key.setdefault((imp.module, imp.name), []).append(imp)

    host_checks = tuple(
        _check_single_host_import(spec, imports_by_key.get((spec.module, spec.name), []))
        for spec in expected_host_imports
    )
    missing_host_imports = tuple(c.name for c in host_checks if c.status == "missing")
    mismatched_host_imports = tuple(
        c.name for c in host_checks if c.present and not c.ok
    )
    all_host_imports_ok = all(c.ok for c in host_checks)

    errors: list[str] = []
    warnings: list[str] = []

    if not all_memories:
        msg = "module declares or imports no linear memory"
        if require_memory64 or require_shared_memory:
            errors.append(msg)
        else:
            warnings.append(msg)

    for idx, mem in enumerate(all_memories):
        if mem.is_shared and not mem.has_max:
            warnings.append(
                f"memory[{idx}] ({mem.source}) is shared without a maximum page limit "
                "(required by WebAssembly threads spec)"
            )
        if mem.max_pages is not None and mem.min_pages > mem.max_pages:
            warnings.append(
                f"memory[{idx}] ({mem.source}) has min_pages ({mem.min_pages}) > "
                f"max_pages ({mem.max_pages})"
            )

    if not is_memory64:
        msg = "memory64 not enabled (no imported or defined memory has flag bit 0x04)"
        if require_memory64:
            errors.append(msg)
        else:
            warnings.append(msg)

    if not is_shared_memory:
        msg = "shared memory not enabled (no imported or defined memory has flag bit 0x02; required for -pthread)"
        if require_shared_memory:
            errors.append(msg)
        else:
            warnings.append(msg)

    for check in host_checks:
        if not check.ok:
            if require_host_imports:
                errors.append(check.detail)
            else:
                warnings.append(check.detail)

    required_checks = {
        "memory64": require_memory64,
        "shared_memory": require_shared_memory,
        "host_imports": require_host_imports,
    }

    return ValidationReport(
        ok=len(errors) == 0,
        abi_version=ABI_VERSION,
        path=path,
        byte_size=module.byte_size,
        wasm_version=module.version,
        is_memory64=is_memory64,
        is_shared_memory=is_shared_memory,
        memories=all_memories,
        host_imports=host_checks,
        missing_host_imports=missing_host_imports,
        mismatched_host_imports=mismatched_host_imports,
        all_host_imports_ok=all_host_imports_ok,
        imports_count=len(module.imports),
        exports_count=len(module.exports),
        exports=module.exports,
        custom_sections=module.custom_sections,
        target_features=module.target_features,
        required_checks=required_checks,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def validate_wasm_file(
    path: str | Path,
    *,
    require_memory64: bool = False,
    require_shared_memory: bool = False,
    require_host_imports: bool = False,
) -> ValidationReport:
    """Read, parse, and validate a `.wasm` file from disk."""
    file_path = Path(path)
    data = file_path.read_bytes()
    module = parse_wasm_module(data)
    return validate_runtime_wasm(
        module,
        require_memory64=require_memory64,
        require_shared_memory=require_shared_memory,
        require_host_imports=require_host_imports,
        path=str(file_path),
    )


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------


def format_report_text(report: ValidationReport) -> str:
    """Render a human-readable validation summary."""
    status_label = "PASS" if report.ok else "FAIL"
    header_target = report.path or "<memory>"
    lines: list[str] = [
        f"Wasm ABI Validation [{status_label}]: {header_target}",
        f"  Size:           {report.byte_size} bytes (Wasm v{report.wasm_version}, ABI v{report.abi_version})",
        f"  Memory64:       {'YES' if report.is_memory64 else 'NO'}",
        f"  Shared Memory:  {'YES' if report.is_shared_memory else 'NO'}",
        f"  Host Imports:   {'OK' if report.all_host_imports_ok else 'INCOMPLETE/MISMATCH'} "
        f"({sum(1 for h in report.host_imports if h.ok)}/{len(report.host_imports)} valid)",
        f"  Counts:         {report.imports_count} imports, {report.exports_count} exports, "
        f"{len(report.memories)} memories",
    ]

    if report.memories:
        lines.append("  Memories:")
        for idx, mem in enumerate(report.memories):
            origin = (
                f"import {mem.module}.{mem.name}"
                if mem.source == "import"
                else "defined"
            )
            max_str = str(mem.max_pages) if mem.max_pages is not None else "unbounded"
            addr = "i64 (memory64)" if mem.is_memory64 else "i32 (wasm32)"
            shared = "shared" if mem.is_shared else "unshared"
            lines.append(
                f"    [{idx}] {origin}: {addr}, {shared}, flags=0x{mem.flags:02x}, "
                f"pages={mem.min_pages}..{max_str}"
            )

    lines.append("  Required env.host_* ABI imports:")
    for check in report.host_imports:
        actual = check.actual_signature or (check.actual_kind or "missing")
        tag = check.status.upper()
        lines.append(
            f"    [{tag}] {check.module}.{check.name}: expected {check.expected_signature}, "
            f"actual {actual}"
        )

    if report.custom_sections:
        lines.append(f"  Custom sections: {', '.join(report.custom_sections)}")
    if report.target_features:
        lines.append(f"  Target features: {', '.join(report.target_features)}")

    if report.errors:
        lines.append("  Errors:")
        for err in report.errors:
            lines.append(f"    - {err}")

    if report.warnings:
        lines.append("  Warnings:")
        for warn in report.warnings:
            lines.append(f"    - {warn}")

    return "\n".join(lines)


def report_to_json(report: ValidationReport, *, indent: int = 2) -> str:
    """Serialize a `ValidationReport` to formatted JSON."""
    return json.dumps(report.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Synthetic Wasm module binary builder helpers (for testing & fixtures)
# ---------------------------------------------------------------------------


def encode_uleb128(value: int, *, max_bits: int = 64) -> bytes:
    """Encode a non-negative integer as unsigned LEB128."""
    if value < 0:
        raise ValueError(f"ULEB128 requires non-negative integer, got {value}")
    if value >= (1 << max_bits):
        raise ValueError(f"value {value} exceeds {max_bits} bits")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value != 0:
            byte |= 0x80
        out.append(byte)
        if value == 0:
            break
    return bytes(out)


def encode_name(text: str) -> bytes:
    """Encode a UTF-8 WebAssembly name (`u32` byte length + UTF-8 bytes)."""
    raw = text.encode("utf-8")
    return encode_uleb128(len(raw), max_bits=32) + raw


def encode_section(section_id: int, payload: bytes) -> bytes:
    """Wrap a section payload with its section ID and LEB128 byte length."""
    return bytes([section_id]) + encode_uleb128(len(payload), max_bits=32) + payload


def encode_memory_limits(
    min_pages: int,
    max_pages: int | None = None,
    *,
    is_shared: bool = False,
    is_memory64: bool = False,
) -> bytes:
    """Encode WebAssembly memory limits (supports memory32/memory64 and shared)."""
    flags = 0
    if max_pages is not None:
        flags |= MEMORY_FLAG_HAS_MAX
    if is_shared:
        flags |= MEMORY_FLAG_SHARED
    if is_memory64:
        flags |= MEMORY_FLAG_MEMORY64

    bits = 64 if is_memory64 else 32
    out = bytearray([flags])
    out.extend(encode_uleb128(min_pages, max_bits=bits))
    if max_pages is not None:
        out.extend(encode_uleb128(max_pages, max_bits=bits))
    return bytes(out)


def encode_table_limits(
    elem_type: str,
    min_size: int,
    max_size: int | None = None,
    *,
    is_table64: bool = False,
) -> bytes:
    """Encode WebAssembly table type + limits."""
    elem_code = VALTYPE_CODES[elem_type]
    flags = 0
    if max_size is not None:
        flags |= TABLE_FLAG_HAS_MAX
    if is_table64:
        flags |= TABLE_FLAG_TABLE64
    bits = 64 if is_table64 else 32
    out = bytearray([elem_code, flags])
    out.extend(encode_uleb128(min_size, max_bits=bits))
    if max_size is not None:
        out.extend(encode_uleb128(max_size, max_bits=bits))
    return bytes(out)


def encode_target_features_section(features: Sequence[str]) -> bytes:
    """Encode an LLVM `target_features` custom section."""
    body = bytearray(encode_name("target_features"))
    body.extend(encode_uleb128(len(features), max_bits=32))
    for feat in features:
        if feat and feat[0] in ("+", "-", "="):
            prefix = ord(feat[0])
            name = feat[1:]
        else:
            prefix = ord("+")
            name = feat
        body.append(prefix)
        body.extend(encode_name(name))
    return encode_section(SECTION_CUSTOM, bytes(body))


def build_wasm_module(
    *,
    types: Sequence[tuple[Sequence[str], Sequence[str]]] = (),
    imports: Sequence[dict[str, Any]] = (),
    function_types: Sequence[int] = (),
    memories: Sequence[dict[str, Any]] = (),
    exports: Sequence[tuple[str, str, int]] = (),
    custom_sections: Sequence[tuple[str, bytes]] = (),
    target_features: Sequence[str] = (),
) -> bytes:
    """Build a synthetic binary WebAssembly module for unit testing and fixtures."""
    out = bytearray(WASM_MAGIC + struct.pack("<I", WASM_VERSION))

    if types:
        payload = bytearray(encode_uleb128(len(types), max_bits=32))
        for params, results in types:
            payload.append(0x60)
            payload.extend(encode_uleb128(len(params), max_bits=32))
            for p in params:
                payload.append(VALTYPE_CODES[p])
            payload.extend(encode_uleb128(len(results), max_bits=32))
            for r in results:
                payload.append(VALTYPE_CODES[r])
        out.extend(encode_section(SECTION_TYPE, bytes(payload)))

    if imports:
        payload = bytearray(encode_uleb128(len(imports), max_bits=32))
        for imp in imports:
            payload.extend(encode_name(imp["module"]))
            payload.extend(encode_name(imp["name"]))
            kind = imp["kind"]
            payload.append(KIND_TO_CODE[kind])
            if kind == "func":
                payload.extend(encode_uleb128(imp["type_index"], max_bits=32))
            elif kind == "table":
                payload.extend(
                    encode_table_limits(
                        imp.get("elem_type", "funcref"),
                        imp.get("min_size", 0),
                        imp.get("max_size"),
                        is_table64=bool(imp.get("is_table64", False)),
                    )
                )
            elif kind == "memory":
                payload.extend(
                    encode_memory_limits(
                        imp.get("min_pages", 1),
                        imp.get("max_pages"),
                        is_shared=bool(imp.get("is_shared", False)),
                        is_memory64=bool(imp.get("is_memory64", False)),
                    )
                )
            elif kind == "global":
                payload.append(VALTYPE_CODES[imp["val_type"]])
                payload.append(0x01 if imp.get("mutable", False) else 0x00)
            elif kind == "tag":
                payload.append(int(imp.get("attribute", 0)))
                payload.extend(encode_uleb128(imp["type_index"], max_bits=32))
            else:
                raise ValueError(f"unsupported import kind {kind!r}")
        out.extend(encode_section(SECTION_IMPORT, bytes(payload)))

    if function_types:
        payload = bytearray(encode_uleb128(len(function_types), max_bits=32))
        for type_idx in function_types:
            payload.extend(encode_uleb128(type_idx, max_bits=32))
        out.extend(encode_section(SECTION_FUNCTION, bytes(payload)))

    if memories:
        payload = bytearray(encode_uleb128(len(memories), max_bits=32))
        for mem in memories:
            payload.extend(
                encode_memory_limits(
                    mem.get("min_pages", 1),
                    mem.get("max_pages"),
                    is_shared=bool(mem.get("is_shared", False)),
                    is_memory64=bool(mem.get("is_memory64", False)),
                )
            )
        out.extend(encode_section(SECTION_MEMORY, bytes(payload)))

    if exports:
        payload = bytearray(encode_uleb128(len(exports), max_bits=32))
        for name, kind, index in exports:
            payload.extend(encode_name(name))
            payload.append(KIND_TO_CODE[kind])
            payload.extend(encode_uleb128(index, max_bits=32))
        out.extend(encode_section(SECTION_EXPORT, bytes(payload)))

    for custom_name, custom_body in custom_sections:
        out.extend(encode_section(SECTION_CUSTOM, encode_name(custom_name) + custom_body))

    if target_features:
        out.extend(encode_target_features_section(target_features))

    return bytes(out)


def build_valid_runtime_wasm(
    *,
    is_memory64: bool = True,
    is_shared: bool = True,
    imported_memory: bool = True,
    min_pages: int = 256,
    max_pages: int | None = 65536,
    wasm32_host_sigs: bool = False,
    omit_host_imports: Sequence[str] = (),
    include_target_features: bool = True,
) -> bytes:
    """Build a synthetic `runtime.wasm` fixture matching `docs/wasm_abi.md`."""
    omit_set = set(omit_host_imports)
    types: list[tuple[Sequence[str], Sequence[str]]] = []
    imports: list[dict[str, Any]] = []
    memories: list[dict[str, Any]] = []

    if imported_memory:
        imports.append(
            {
                "module": "env",
                "name": "memory",
                "kind": "memory",
                "min_pages": min_pages,
                "max_pages": max_pages,
                "is_shared": is_shared,
                "is_memory64": is_memory64,
            }
        )
    else:
        memories.append(
            {
                "min_pages": min_pages,
                "max_pages": max_pages,
                "is_shared": is_shared,
                "is_memory64": is_memory64,
            }
        )

    for spec in REQUIRED_HOST_IMPORTS:
        if spec.name in omit_set:
            continue
        params = spec.params_wasm32 if wasm32_host_sigs else spec.params_wasm64
        results = spec.results_wasm32 if wasm32_host_sigs else spec.results_wasm64
        type_idx = len(types)
        types.append((params, results))
        imports.append(
            {
                "module": spec.module,
                "name": spec.name,
                "kind": "func",
                "type_index": type_idx,
            }
        )

    exports = [("memory", "memory", 0)]
    features: list[str] = []
    if include_target_features:
        if is_memory64:
            features.append("+memory64")
        if is_shared:
            features.append("+atomics")
            features.append("+bulk-memory")

    return build_wasm_module(
        types=types,
        imports=imports,
        memories=memories,
        exports=exports,
        target_features=features,
    )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and validate a compiled WebAssembly binary for Wasm64 memory64, "
            "shared memory (-pthread), and playgta6 env.host_* ABI imports."
        )
    )
    parser.add_argument(
        "wasm_file",
        help="Path to the .wasm binary to inspect (or '-' to read from stdin).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit structured JSON output instead of human-readable text.",
    )
    parser.add_argument(
        "--require-memory64",
        action="store_true",
        help="Fail with exit code 1 unless at least one memory has the memory64 flag (0x04).",
    )
    parser.add_argument(
        "--require-shared-memory",
        action="store_true",
        help="Fail with exit code 1 unless at least one memory has the shared flag (0x02).",
    )
    parser.add_argument(
        "--require-host-imports",
        action="store_true",
        help="Fail with exit code 1 unless all required env.host_* Wasm64 ABI imports are present and match.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.wasm_file == "-":
        raw_bytes = sys.stdin.buffer.read()
        source_label = "<stdin>"
    else:
        path = Path(args.wasm_file)
        source_label = str(path)
        if not path.is_file():
            err_msg = f"wasm file not found: {path}"
            if args.as_json:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "abi_version": ABI_VERSION,
                            "path": source_label,
                            "error": err_msg,
                            "errors": [err_msg],
                        },
                        indent=2,
                    )
                )
            else:
                print(f"Error: {err_msg}", file=sys.stderr)
            return 2
        raw_bytes = path.read_bytes()

    try:
        module = parse_wasm_module(raw_bytes)
    except WasmParseError as exc:
        err_msg = str(exc)
        if args.as_json:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "abi_version": ABI_VERSION,
                        "path": source_label,
                        "byte_size": len(raw_bytes),
                        "parse_error": err_msg,
                        "offset": exc.offset,
                        "errors": [err_msg],
                    },
                    indent=2,
                )
            )
        else:
            print(f"Error: failed to parse {source_label}: {err_msg}", file=sys.stderr)
        return 1

    report = validate_runtime_wasm(
        module,
        require_memory64=args.require_memory64,
        require_shared_memory=args.require_shared_memory,
        require_host_imports=args.require_host_imports,
        path=source_label,
    )

    if args.as_json:
        print(report_to_json(report))
    else:
        print(format_report_text(report))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
