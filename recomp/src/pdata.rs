//! PE64 exception directory (`.pdata`) parser.
//!
//! `IMAGE_DIRECTORY_ENTRY_EXCEPTION` is an array of 12-byte `RUNTIME_FUNCTION` records
//! (`BeginAddress`, `EndAddress`, `UnwindInfoAddress`, all RVAs), one per contiguous chunk of
//! every function that has unwind data. Each record points at an `UNWIND_INFO`:
//!
//! ```text
//! u8  Version:3, Flags:5          (EHANDLER = 1, UHANDLER = 2, CHAININFO = 4)
//! u8  SizeOfProlog
//! u8  CountOfCodes                (2-byte slots)
//! u8  FrameRegister:4, FrameOffset:4
//! UNWIND_CODE[CountOfCodes]       (padded to an even count)
//! u32 ExceptionHandler + language-specific data   if EHANDLER | UHANDLER
//! RUNTIME_FUNCTION parent                          if CHAININFO
//! ```
//!
//! A chained record describes a secondary chunk (cold code, shrink-wrapped regions) of the
//! function its parent belongs to; an `UnwindInfoAddress` with bit 0 set points at another
//! `RUNTIME_FUNCTION` (at `rva - 1`) instead. Following either link to a record with plain
//! unwind info yields the chunk's primary function.
//!
//! [`ExceptionDirectory::from_pe`] never fails: problems become [`PdataDiagnostic`]s and the
//! affected record is dropped (bad range) or kept without unwind info (bad unwind data). All
//! reads go through R1's bounds-checked [`Pe64::rva_to_file_offset`]. The kept records are sorted,
//! deduplicated and non-overlapping, so [`ExceptionDirectory::function_table`] can map every
//! chunk to its primary function and answer "which function owns this address" in `O(log n)`.
//! Everything is `O(n log n)` in the number of records.

use std::fmt;

use crate::code_image::{CodeImage, Region};
use crate::pe::{IMAGE_SCN_MEM_EXECUTE, Pe64};

/// Data directory index of the exception table.
pub const DIR_EXCEPTION: usize = 3;
/// Size of one `RUNTIME_FUNCTION` record.
pub const RUNTIME_FUNCTION_LEN: usize = 12;

pub const UNW_FLAG_EHANDLER: u8 = 0x1;
pub const UNW_FLAG_UHANDLER: u8 = 0x2;
pub const UNW_FLAG_CHAININFO: u8 = 0x4;

/// Bit 0 of `UnwindInfoAddress`: the record at `rva - 1` holds the unwind data.
const RUNTIME_FUNCTION_INDIRECT: u32 = 0x1;
/// Longest chain of parent links followed before giving up (guards against cycles).
const MAX_CHAIN_DEPTH: usize = 32;
const UNWIND_HEADER_LEN: usize = 4;

/// One `RUNTIME_FUNCTION` record (RVAs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuntimeFunction {
    pub begin: u32,
    /// One past the last byte of the chunk.
    pub end: u32,
    pub unwind: u32,
}

impl RuntimeFunction {
    /// Decode a record from the first [`RUNTIME_FUNCTION_LEN`] bytes of `bytes`.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        Some(Self {
            begin: le_u32(bytes, 0)?,
            end: le_u32(bytes, 4)?,
            unwind: le_u32(bytes, 8)?,
        })
    }
}

/// Decoded `UNWIND_CODE` operation. Register numbers use the x64 encoding (0 = RAX ... 15 = R15,
/// XMM0 ... XMM15). Far/near variants of the same operation are folded together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnwindOp {
    /// `UWOP_PUSH_NONVOL`.
    PushNonVolatile { register: u8 },
    /// `UWOP_ALLOC_SMALL` / `UWOP_ALLOC_LARGE`: `sub rsp, size`.
    Alloc { size: u32 },
    /// `UWOP_SET_FPREG`: frame register = `rsp + frame_offset`.
    SetFramePointer,
    /// `UWOP_SAVE_NONVOL` / `UWOP_SAVE_NONVOL_FAR`: `mov [rsp+offset], reg`.
    SaveNonVolatile { register: u8, offset: u32 },
    /// `UWOP_SAVE_XMM128` / `UWOP_SAVE_XMM128_FAR`: `movaps [rsp+offset], xmm`.
    SaveXmm128 { register: u8, offset: u32 },
    /// `UWOP_PUSH_MACHFRAME` (hardware interrupt/exception frame).
    PushMachineFrame { error_code: bool },
    /// `UWOP_EPILOG` (version 2 only), raw: the first one carries the epilog size and flags,
    /// the following ones the epilog offsets from the end of the function.
    Epilog { code_offset: u8, info: u8 },
}

/// One decoded unwind code with its prolog offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnwindCode {
    /// Offset from the function start of the end of the prolog instruction it describes.
    pub prolog_offset: u8,
    pub op: UnwindOp,
}

/// Language-specific exception handler of an `UNWIND_INFO`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExceptionHandler {
    /// RVA of the handler routine (code).
    pub rva: u32,
    /// RVA of the language-specific handler data that follows the handler RVA.
    pub data_rva: u32,
}

/// Decoded `UNWIND_INFO`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnwindInfo {
    /// 1 or 2.
    pub version: u8,
    /// `UNW_FLAG_*` bits.
    pub flags: u8,
    pub prolog_size: u8,
    /// Frame register number (0 = none).
    pub frame_register: u8,
    /// Scaled frame register offset in bytes (`FrameOffset * 16`).
    pub frame_offset: u8,
    /// Unwind codes in table order (reverse prolog order).
    pub codes: Vec<UnwindCode>,
    /// Set when `EHANDLER` or `UHANDLER` is set and `CHAININFO` is not.
    pub handler: Option<ExceptionHandler>,
    /// Parent record when `CHAININFO` is set (takes precedence over handler flags).
    pub chained: Option<RuntimeFunction>,
}

/// Why an `UNWIND_INFO` could not be decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnwindError {
    /// The unwind RVA does not map to file bytes.
    Unmapped,
    /// The structure runs past the mapped bytes.
    Truncated,
    /// Version other than 1 or 2.
    UnsupportedVersion(u8),
    /// Reserved opcode, invalid operation info, or a code that overruns `CountOfCodes`.
    InvalidCode { slot: u8, op: u8 },
}

impl UnwindInfo {
    /// Bytes needed by the structure described by the 4-byte header in `bytes`.
    pub fn encoded_len(bytes: &[u8]) -> Result<usize, UnwindError> {
        let header = bytes
            .get(..UNWIND_HEADER_LEN)
            .ok_or(UnwindError::Truncated)?;
        let flags = header[0] >> 3;
        let count = usize::from(header[2]);
        let padded_slots = count.checked_add(count & 1).ok_or(UnwindError::Truncated)?;
        let codes_bytes = padded_slots
            .checked_mul(2)
            .ok_or(UnwindError::Truncated)?;
        let trailer_len = if flags & UNW_FLAG_CHAININFO != 0 {
            RUNTIME_FUNCTION_LEN
        } else if flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) != 0 {
            4
        } else {
            0
        };
        UNWIND_HEADER_LEN
            .checked_add(codes_bytes)
            .and_then(|n| n.checked_add(trailer_len))
            .ok_or(UnwindError::Truncated)
    }

    /// Decode an `UNWIND_INFO` located at `rva` from `bytes` (which may extend past it).
    pub fn parse(bytes: &[u8], rva: u32) -> Result<Self, UnwindError> {
        let len = Self::encoded_len(bytes)?;
        let bytes = bytes.get(..len).ok_or(UnwindError::Truncated)?;
        let version = bytes[0] & 0x7;
        if version != 1 && version != 2 {
            return Err(UnwindError::UnsupportedVersion(version));
        }
        let flags = bytes[0] >> 3;
        let count = usize::from(bytes[2]);
        let codes_end = count
            .checked_mul(2)
            .and_then(|n| UNWIND_HEADER_LEN.checked_add(n))
            .ok_or(UnwindError::Truncated)?;
        let slots = bytes
            .get(UNWIND_HEADER_LEN..codes_end)
            .ok_or(UnwindError::Truncated)?;
        let codes = decode_codes(slots, version)?;

        let padded_slots = count.checked_add(count & 1).ok_or(UnwindError::Truncated)?;
        let trailer = padded_slots
            .checked_mul(2)
            .and_then(|n| UNWIND_HEADER_LEN.checked_add(n))
            .ok_or(UnwindError::Truncated)?;
        let (mut handler, mut chained) = (None, None);
        if flags & UNW_FLAG_CHAININFO != 0 {
            let trailer_bytes = bytes.get(trailer..).ok_or(UnwindError::Truncated)?;
            chained = Some(RuntimeFunction::parse(trailer_bytes).ok_or(UnwindError::Truncated)?);
        } else if flags & (UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER) != 0 {
            let data_rva = u32::try_from(trailer)
                .ok()
                .and_then(|t| t.checked_add(4))
                .and_then(|t| t.checked_add(rva))
                .ok_or(UnwindError::Truncated)?;
            let handler_rva = le_u32(bytes, trailer).ok_or(UnwindError::Truncated)?;
            handler = Some(ExceptionHandler {
                rva: handler_rva,
                data_rva,
            });
        }
        Ok(Self {
            version,
            flags,
            prolog_size: bytes[1],
            frame_register: bytes[3] & 0xf,
            frame_offset: (bytes[3] >> 4).saturating_mul(16),
            codes,
            handler,
            chained,
        })
    }
}

/// Decode `CountOfCodes` slots. `UWOP_EPILOG` (6) only exists in version 2; the obsolete
/// version-1 meanings of 6 and 7 and the reserved 7, 11..15 are rejected.
fn decode_codes(slots: &[u8], version: u8) -> Result<Vec<UnwindCode>, UnwindError> {
    let count = slots.len() / 2;
    let slot = |i: usize| -> Option<u16> {
        let off = i.checked_mul(2)?;
        let pair = slots.get(off..off.checked_add(2)?)?;
        Some(u16::from_le_bytes([pair[0], pair[1]]))
    };
    let mut codes = Vec::with_capacity(count);
    let mut i = 0usize;
    while i < count {
        let off = i.checked_mul(2).ok_or(UnwindError::Truncated)?;
        let pair = slots
            .get(off..off.checked_add(2).ok_or(UnwindError::Truncated)?)
            .ok_or(UnwindError::Truncated)?;
        let (prolog_offset, op_byte) = (pair[0], pair[1]);
        let (op, info) = (op_byte & 0xf, op_byte >> 4);
        let slot_idx = u8::try_from(i).unwrap_or(u8::MAX);
        let invalid = UnwindError::InvalidCode { slot: slot_idx, op };
        let used = match op {
            1 if info == 1 => 3,
            1 if info == 0 => 2,
            4 | 8 => 2,
            5 | 9 => 3,
            0 | 2 | 3 | 10 => 1,
            6 if version == 2 => 1,
            _ => return Err(invalid),
        };
        let next_i = i.checked_add(used).ok_or(invalid)?;
        if next_i > count {
            return Err(invalid);
        }
        let read_slot = |idx: usize| slot(idx).ok_or(invalid);
        let far = || -> Result<u32, UnwindError> {
            let lo = u32::from(read_slot(i + 1)?);
            let hi = u32::from(read_slot(i + 2)?);
            Ok(lo | (hi << 16))
        };
        let op = match op {
            0 => UnwindOp::PushNonVolatile { register: info },
            1 if info == 0 => UnwindOp::Alloc {
                size: u32::from(read_slot(i + 1)?)
                    .checked_mul(8)
                    .ok_or(invalid)?,
            },
            1 => UnwindOp::Alloc { size: far()? },
            2 => UnwindOp::Alloc {
                size: u32::from(info)
                    .checked_mul(8)
                    .and_then(|v| v.checked_add(8))
                    .ok_or(invalid)?,
            },
            3 => UnwindOp::SetFramePointer,
            4 => UnwindOp::SaveNonVolatile {
                register: info,
                offset: u32::from(read_slot(i + 1)?)
                    .checked_mul(8)
                    .ok_or(invalid)?,
            },
            5 => UnwindOp::SaveNonVolatile {
                register: info,
                offset: far()?,
            },
            6 => UnwindOp::Epilog {
                code_offset: prolog_offset,
                info,
            },
            8 => UnwindOp::SaveXmm128 {
                register: info,
                offset: u32::from(read_slot(i + 1)?)
                    .checked_mul(16)
                    .ok_or(invalid)?,
            },
            9 => UnwindOp::SaveXmm128 {
                register: info,
                offset: far()?,
            },
            10 if info <= 1 => UnwindOp::PushMachineFrame {
                error_code: info == 1,
            },
            _ => return Err(invalid),
        };
        codes.push(UnwindCode { prolog_offset, op });
        i = next_i;
    }
    Ok(codes)
}

/// Category of an exception-directory problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PdataDiagnosticKind {
    /// The directory RVA does not map to file bytes; nothing was read.
    DirectoryUnmapped,
    /// The directory runs past its section's file bytes; the records before that were read.
    DirectoryTruncated,
    /// The directory size is not a multiple of 12; the trailing bytes were ignored.
    TrailingBytes,
    /// Records are not sorted by `BeginAddress` (the loader binary-searches them); sorted here.
    Unsorted,
    /// `BeginAddress >= EndAddress`; record dropped.
    EmptyRange,
    /// `EndAddress` is past `SizeOfImage`; record dropped.
    OutsideImage,
    /// `BeginAddress..EndAddress` is not contained inside an executable section; record dropped.
    NotExecutable,
    /// The range overlaps an earlier record; record dropped.
    Overlap,
    /// The record's `UNWIND_INFO` could not be decoded; the range is kept without unwind info.
    Unwind(UnwindError),
    /// A chained/indirect parent could not be read or is not a record of this directory; the
    /// chunk is treated as its own function.
    ChainBroken,
    /// More than 32 chained links (or a cycle); the chunk is treated as its own function.
    ChainTooDeep,
    /// The exception handler RVA is not inside an executable section.
    HandlerNotCode,
}

impl PdataDiagnosticKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::DirectoryUnmapped => "directory_unmapped",
            Self::DirectoryTruncated => "directory_truncated",
            Self::TrailingBytes => "trailing_bytes",
            Self::Unsorted => "unsorted",
            Self::EmptyRange => "empty_range",
            Self::OutsideImage => "outside_image",
            Self::NotExecutable => "not_executable",
            Self::Overlap => "overlap",
            Self::Unwind(UnwindError::Unmapped) => "unwind_unmapped",
            Self::Unwind(UnwindError::Truncated) => "unwind_truncated",
            Self::Unwind(UnwindError::UnsupportedVersion(_)) => "unwind_unsupported_version",
            Self::Unwind(UnwindError::InvalidCode { .. }) => "unwind_invalid_code",
            Self::ChainBroken => "chain_broken",
            Self::ChainTooDeep => "chain_too_deep",
            Self::HandlerNotCode => "handler_not_code",
        }
    }
}

/// A non-fatal exception-directory problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PdataDiagnostic {
    /// `BeginAddress` of the affected record, or the directory RVA for directory-level problems.
    pub rva: u32,
    pub kind: PdataDiagnosticKind,
}

impl fmt::Display for PdataDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at RVA 0x{:08X}", self.kind.as_str(), self.rva)
    }
}

/// A validated `RUNTIME_FUNCTION` record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdataEntry {
    pub function: RuntimeFunction,
    /// `None` for indirect records and when the unwind info is malformed (see diagnostics).
    pub unwind: Option<UnwindInfo>,
    /// `BeginAddress` of the primary function this chunk belongs to (its own begin unless
    /// chained or indirect).
    pub primary: u32,
}

/// Parsed exception directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExceptionDirectory {
    /// Number of records read from the directory (before validation).
    pub record_count: usize,
    /// Valid records sorted by `BeginAddress`, deduplicated and non-overlapping.
    pub entries: Vec<PdataEntry>,
    /// Problems in directory order, then per record.
    pub diagnostics: Vec<PdataDiagnostic>,
}

/// Bounds-checked RVA reads through R1's section table.
struct Reader<'p, 'f> {
    pe: &'p Pe64,
    file: &'f [u8],
}

impl<'f> Reader<'_, 'f> {
    /// Contiguous initialized file bytes mapped from `rva` within a single PE section (or the
    /// PE headers), clamped to the file length and to non-wrapping `u32` RVAs.
    fn mapped_slice_from(&self, rva: u32) -> Option<&'f [u8]> {
        let file_limit = self.file.len().min(self.pe.file_size);
        let max_rva_span = u64::from(u32::MAX - rva) + 1;
        for section in &self.pe.sections {
            let virt_extent = if section.virtual_size == 0 {
                section.raw_size
            } else {
                section.virtual_size
            };
            let mapped_extent = virt_extent.min(section.raw_size);
            if let Some(delta) = rva.checked_sub(section.virtual_address)
                && delta < mapped_extent
            {
                let raw_offset = usize::try_from(section.raw_offset).ok()?;
                let delta_usize = usize::try_from(delta).ok()?;
                let start = raw_offset.checked_add(delta_usize)?;
                if start >= file_limit {
                    return None;
                }
                let remaining =
                    usize::try_from(u64::from(mapped_extent - delta).min(max_rva_span))
                        .unwrap_or(usize::MAX);
                let end = start.saturating_add(remaining).min(file_limit);
                return self.file.get(start..end);
            }
        }
        if rva < self.pe.optional.size_of_headers {
            let start = usize::try_from(rva).ok()?;
            if start >= file_limit {
                return None;
            }
            let remaining =
                usize::try_from(u64::from(self.pe.optional.size_of_headers - rva).min(max_rva_span))
                    .unwrap_or(usize::MAX);
            let end = start.saturating_add(remaining).min(file_limit);
            return self.file.get(start..end);
        }
        None
    }

    /// `len` bytes at `rva`, only if they are contiguous file bytes of one mapping.
    fn read(&self, rva: u32, len: usize) -> Option<&'f [u8]> {
        if len > 0 {
            let last_off = u32::try_from(len.checked_sub(1)?).ok()?;
            rva.checked_add(last_off)?;
        }
        self.mapped_slice_from(rva)?.get(..len)
    }

    fn unwind_info(&self, rva: u32) -> Result<UnwindInfo, UnwindError> {
        if rva == 0 {
            return Err(UnwindError::Unmapped);
        }
        let header = self
            .read(rva, UNWIND_HEADER_LEN)
            .ok_or(UnwindError::Unmapped)?;
        let len = UnwindInfo::encoded_len(header)?;
        let bytes = self.read(rva, len).ok_or(UnwindError::Truncated)?;
        UnwindInfo::parse(bytes, rva)
    }

    /// Record pointed to by an indirect `UnwindInfoAddress`.
    fn indirect(&self, unwind: u32) -> Option<RuntimeFunction> {
        let target = unwind.checked_sub(RUNTIME_FUNCTION_INDIRECT)?;
        if target == 0 {
            return None;
        }
        RuntimeFunction::parse(self.read(target, RUNTIME_FUNCTION_LEN)?)
    }

    fn is_executable(&self, rva: u32) -> bool {
        self.pe.sections.iter().any(|s| {
            let extent = if s.virtual_size == 0 {
                s.raw_size
            } else {
                s.virtual_size
            };
            s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
                && rva
                    .checked_sub(s.virtual_address)
                    .is_some_and(|delta| delta < extent)
        })
    }

    /// Whether `[begin, end)` is non-empty and contained within one executable section.
    fn is_executable_range(&self, begin: u32, end: u32) -> bool {
        let Some(len) = end.checked_sub(begin) else {
            return false;
        };
        if len == 0 {
            return false;
        }
        self.pe.sections.iter().any(|s| {
            let extent = if s.virtual_size == 0 {
                s.raw_size
            } else {
                s.virtual_size
            };
            s.characteristics & IMAGE_SCN_MEM_EXECUTE != 0
                && begin
                    .checked_sub(s.virtual_address)
                    .is_some_and(|delta| delta < extent && len <= extent - delta)
        })
    }

    /// Follow the parent links starting at `link` to the record with plain unwind info.
    fn primary_of(&self, mut link: RuntimeFunction) -> Result<u32, PdataDiagnosticKind> {
        for _ in 0..MAX_CHAIN_DEPTH {
            let next = if link.unwind & RUNTIME_FUNCTION_INDIRECT != 0 {
                self.indirect(link.unwind)
            } else {
                match self.unwind_info(link.unwind) {
                    Ok(info) => match info.chained {
                        Some(parent) => Some(parent),
                        None => return Ok(link.begin),
                    },
                    Err(_) => None,
                }
            };
            let parent = next.ok_or(PdataDiagnosticKind::ChainBroken)?;
            if parent.begin >= parent.end || parent.end > self.pe.optional.size_of_image {
                return Err(PdataDiagnosticKind::ChainBroken);
            }
            link = parent;
        }
        Err(PdataDiagnosticKind::ChainTooDeep)
    }
}

impl ExceptionDirectory {
    /// Parse the exception directory of `pe`; `file` must be the bytes `pe` was parsed from.
    /// Returns an empty directory when the PE has none.
    pub fn from_pe(pe: &Pe64, file: &[u8]) -> Self {
        let mut out = Self::default();
        let Some(dir) = pe.optional.data_directory(DIR_EXCEPTION) else {
            return out;
        };
        let reader = Reader { pe, file };
        let mut diag = |rva, kind| out.diagnostics.push(PdataDiagnostic { rva, kind });

        let size = usize::try_from(dir.size).unwrap_or(usize::MAX);
        if !size.is_multiple_of(RUNTIME_FUNCTION_LEN) {
            diag(dir.virtual_address, PdataDiagnosticKind::TrailingBytes);
        }
        let Some(mapped) = reader.mapped_slice_from(dir.virtual_address) else {
            diag(dir.virtual_address, PdataDiagnosticKind::DirectoryUnmapped);
            return out;
        };
        if mapped.len() < size {
            diag(dir.virtual_address, PdataDiagnosticKind::DirectoryTruncated);
        }
        let readable = mapped.len().min(size);
        let full_records = readable / RUNTIME_FUNCTION_LEN;
        let record_bytes = full_records.saturating_mul(RUNTIME_FUNCTION_LEN);
        let mut records = Vec::with_capacity(full_records);
        for chunk in mapped
            .get(..record_bytes)
            .unwrap_or(&[])
            .chunks_exact(RUNTIME_FUNCTION_LEN)
        {
            if let Some(record) = RuntimeFunction::parse(chunk) {
                records.push(record);
            }
        }
        out.record_count = records.len();
        if records.windows(2).any(|w| w[1].begin < w[0].begin) {
            diag(dir.virtual_address, PdataDiagnosticKind::Unsorted);
        }

        // Range validation, then sort / dedup / drop overlaps.
        let mut valid = Vec::with_capacity(records.len());
        for record in records {
            let kind = if record.begin >= record.end {
                PdataDiagnosticKind::EmptyRange
            } else if record.end > pe.optional.size_of_image {
                PdataDiagnosticKind::OutsideImage
            } else if !reader.is_executable_range(record.begin, record.end) {
                PdataDiagnosticKind::NotExecutable
            } else {
                valid.push(record);
                continue;
            };
            diag(record.begin, kind);
        }
        valid.sort_unstable();
        valid.dedup();
        let mut kept: Vec<RuntimeFunction> = Vec::with_capacity(valid.len());
        for record in valid {
            if kept.last().is_some_and(|prev| record.begin < prev.end) {
                diag(record.begin, PdataDiagnosticKind::Overlap);
            } else {
                kept.push(record);
            }
        }

        // Unwind info and primary function of every kept record.
        let mut entries = Vec::with_capacity(kept.len());
        for function in kept {
            let (unwind, link) = if function.unwind & RUNTIME_FUNCTION_INDIRECT != 0 {
                (None, Some(function))
            } else {
                match reader.unwind_info(function.unwind) {
                    Ok(info) => {
                        if info.handler.is_some_and(|h| !reader.is_executable(h.rva)) {
                            diag(function.begin, PdataDiagnosticKind::HandlerNotCode);
                        }
                        let link = info.chained.is_some().then_some(function);
                        (Some(info), link)
                    }
                    Err(err) => {
                        diag(function.begin, PdataDiagnosticKind::Unwind(err));
                        (None, None)
                    }
                }
            };
            let primary = match link.map(|link| reader.primary_of(link)) {
                None => function.begin,
                Some(Ok(primary)) => primary,
                Some(Err(kind)) => {
                    diag(function.begin, kind);
                    function.begin
                }
            };
            entries.push(PdataEntry {
                function,
                unwind,
                primary,
            });
        }

        // A primary must itself be a record of this directory.
        let begins: Vec<u32> = entries.iter().map(|e| e.function.begin).collect();
        for entry in &mut entries {
            if entry.primary != entry.function.begin
                && begins.binary_search(&entry.primary).is_err()
            {
                diag(entry.function.begin, PdataDiagnosticKind::ChainBroken);
                entry.primary = entry.function.begin;
            }
        }
        out.entries = entries;
        out
    }

    /// Exact function chunk ranges as absolute VAs, each mapped to its primary function.
    /// Entries whose VA arithmetic overflows `u64` are skipped rather than saturated.
    pub fn function_table(&self, image_base: u64) -> FunctionTable {
        let va = |rva: u32| image_base.checked_add(u64::from(rva));
        FunctionTable::new(
            self.entries
                .iter()
                .filter_map(|e| {
                    Some(FunctionRange {
                        start: va(e.function.begin)?,
                        end: va(e.function.end)?,
                        function: va(e.primary)?,
                    })
                })
                .collect(),
        )
    }

    /// RVAs of the exception handlers referenced by unwind info, sorted and deduplicated.
    pub fn handler_rvas(&self) -> Vec<u32> {
        let mut rvas: Vec<u32> = self
            .entries
            .iter()
            .filter_map(|e| e.unwind.as_ref()?.handler.map(|h| h.rva))
            .collect();
        rvas.sort_unstable();
        rvas.dedup();
        rvas
    }

    /// Feed the directory into discovery: install [`Self::function_table`] on `image` and add
    /// every primary function start and exception handler that lies in executable code to its
    /// entry points (kept sorted and deduplicated).
    pub fn seed(&self, image: &mut CodeImage<'_>) {
        let base = image.base();
        image.function_table = self.function_table(base);
        let handlers = self
            .handler_rvas()
            .into_iter()
            .filter_map(|rva| base.checked_add(u64::from(rva)));
        let seeds: Vec<u64> = image
            .function_table
            .function_entries()
            .chain(handlers)
            .filter(|&va| image.region_of(va) == Region::Code)
            .collect();
        let mut entries = std::mem::take(&mut image.entry_points);
        entries.extend(seeds);
        entries.sort_unstable();
        entries.dedup();
        image.entry_points = entries;
    }
}

/// One function chunk `[start, end)` owned by the function whose entry is `function` (VAs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FunctionRange {
    pub start: u64,
    pub end: u64,
    pub function: u64,
}

impl FunctionRange {
    pub fn contains(&self, va: u64) -> bool {
        va >= self.start && va < self.end
    }
}

/// Sorted, non-overlapping function chunk ranges (from `.pdata`), used to bound analysis.
/// Empty when the image has no exception directory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FunctionTable {
    ranges: Vec<FunctionRange>,
}

impl FunctionTable {
    /// Build a table from arbitrary ranges: empty ranges are dropped, the rest sorted and
    /// deduplicated, and a range overlapping an earlier one is dropped.
    pub fn new(mut ranges: Vec<FunctionRange>) -> Self {
        ranges.retain(|r| r.start < r.end);
        ranges.sort_unstable();
        ranges.dedup();
        let mut kept: Vec<FunctionRange> = Vec::with_capacity(ranges.len());
        for range in ranges {
            if kept.last().is_none_or(|prev| range.start >= prev.end) {
                kept.push(range);
            }
        }
        Self { ranges: kept }
    }

    pub fn ranges(&self) -> &[FunctionRange] {
        &self.ranges
    }

    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Chunk containing `va`.
    pub fn range_at(&self, va: u64) -> Option<&FunctionRange> {
        let idx = self.ranges.partition_point(|r| r.start <= va);
        let range = self.ranges.get(idx.checked_sub(1)?)?;
        range.contains(va).then_some(range)
    }

    /// Entry of the function owning `va`, or `None` when no range covers it.
    pub fn function_at(&self, va: u64) -> Option<u64> {
        self.range_at(va).map(|r| r.function)
    }

    /// The gap between two `.pdata` ranges that contains `va` (`[previous end, next start)`), or
    /// `None` when `va` is covered or lies before the first / after the last range. Leaf functions
    /// (no unwind data, so no `.pdata` record) live in such gaps.
    pub fn gap_at(&self, va: u64) -> Option<(u64, u64)> {
        let idx = self.ranges.partition_point(|r| r.start <= va);
        let previous = self.ranges.get(idx.checked_sub(1)?)?;
        let next = self.ranges.get(idx)?;
        (va >= previous.end).then_some((previous.end, next.start))
    }

    /// Primary function entries in ascending order.
    pub fn function_entries(&self) -> impl Iterator<Item = u64> + '_ {
        self.ranges
            .iter()
            .filter(|r| r.start == r.function)
            .map(|r| r.start)
    }

    /// Number of distinct primary functions.
    pub fn function_count(&self) -> usize {
        self.function_entries().count()
    }
}

fn le_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let b = bytes.get(offset..offset.checked_add(4)?)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::pe::test_util::build_synthetic_pe64;

    pub(crate) const BASE: u64 = 0x1_4000_0000;
    /// File offset of the exception data directory entry in the synthetic PE.
    const DIR_ENTRY: usize = 0x108 + DIR_EXCEPTION * 8;
    /// `.pdata` records are placed at this `.rdata` RVA.
    pub(crate) const PDATA_RVA: u32 = 0x2560;

    /// File offset of `rva` in the synthetic PE (`.text` at 0x1000 / file 0x200, `.rdata` at
    /// 0x2000 / file 0x400).
    pub(crate) fn file_offset(rva: u32) -> usize {
        match rva {
            0x1000..0x1200 => rva as usize - 0x1000 + 0x200,
            0x2000..0x2800 => rva as usize - 0x2000 + 0x400,
            _ => panic!("RVA {rva:#x} is not backed by the synthetic PE"),
        }
    }

    pub(crate) fn put(file: &mut [u8], rva: u32, bytes: &[u8]) {
        let at = file_offset(rva);
        file[at..at + bytes.len()].copy_from_slice(bytes);
    }

    /// R1's synthetic PE with `.text` grown to 0x200 bytes, `.rdata` to 0x800, and an exception
    /// directory at [`PDATA_RVA`] holding `records` (`begin, end, unwind` RVAs).
    pub(crate) fn pe_with_pdata(records: &[(u32, u32, u32)]) -> Vec<u8> {
        let mut file = build_synthetic_pe64(false);
        file[0x190..0x194].copy_from_slice(&0x200u32.to_le_bytes()); // .text VirtualSize
        file[0x1b8..0x1bc].copy_from_slice(&0x800u32.to_le_bytes()); // .rdata VirtualSize
        for (i, &(begin, end, unwind)) in records.iter().enumerate() {
            let rva = PDATA_RVA + 12 * i as u32;
            put(&mut file, rva, &begin.to_le_bytes());
            put(&mut file, rva + 4, &end.to_le_bytes());
            put(&mut file, rva + 8, &unwind.to_le_bytes());
        }
        set_pdata_dir(&mut file, PDATA_RVA, 12 * records.len() as u32);
        file
    }

    pub(crate) fn set_pdata_dir(file: &mut [u8], rva: u32, size: u32) {
        file[DIR_ENTRY..DIR_ENTRY + 4].copy_from_slice(&rva.to_le_bytes());
        file[DIR_ENTRY + 4..DIR_ENTRY + 8].copy_from_slice(&size.to_le_bytes());
    }

    /// Minimal version-1 `UNWIND_INFO` (`sub rsp, 0x28`) with the given flags and trailer.
    pub(crate) fn unwind_bytes(flags: u8, trailer: &[u8]) -> Vec<u8> {
        let mut bytes = vec![
            0x01 | (flags << 3),
            0x04,
            0x01,
            0x00,
            0x04,
            0x42,
            0x00,
            0x00,
        ];
        bytes.extend_from_slice(trailer);
        bytes
    }

    fn parse(file: &[u8]) -> ExceptionDirectory {
        let pe = Pe64::parse(file).expect("synthetic PE parses");
        ExceptionDirectory::from_pe(&pe, file)
    }

    fn kinds(dir: &ExceptionDirectory) -> Vec<(u32, PdataDiagnosticKind)> {
        dir.diagnostics.iter().map(|d| (d.rva, d.kind)).collect()
    }

    const UNWIND_A: u32 = 0x2700;
    const UNWIND_B: u32 = 0x2720;
    const UNWIND_C: u32 = 0x2740;
    const UNWIND_D: u32 = 0x2760;

    #[test]
    fn decodes_every_unwind_code() {
        let bytes = [
            0x02 | (UNW_FLAG_EHANDLER << 3), // version 2, EHANDLER
            0x30,                            // prolog size
            0x15,                            // 21 slots
            0x35,                            // frame register RBP, offset 3 * 16
            0x05,
            0x16, // epilog: size 5, flags 1
            0x20,
            0x06, // epilog: offset 0x20
            0x01,
            0x50, // push rbp
            0x05,
            0x32, // alloc small 0x20
            0x0c,
            0x01,
            0x00,
            0x02, // alloc large 0x200 * 8
            0x10,
            0x11,
            0x00,
            0x00,
            0x01,
            0x00, // alloc large far 0x10000
            0x12,
            0x03, // set fpreg
            0x14,
            0x34,
            0x06,
            0x00, // save rbx at 6 * 8
            0x18,
            0x65,
            0x45,
            0x23,
            0x01,
            0x00, // save rsi far at 0x12345
            0x1c,
            0x68,
            0x02,
            0x00, // save xmm6 at 2 * 16
            0x20,
            0x79,
            0x00,
            0x10,
            0x00,
            0x00, // save xmm7 far at 0x1000
            0x22,
            0x1a, // push machframe with error code
            0x00,
            0x00, // padding to an even slot count
            0x00,
            0x15,
            0x00,
            0x00, // handler RVA 0x1500
            0xaa,
            0xbb, // language-specific data (not part of the structure)
        ];
        assert_eq!(UnwindInfo::encoded_len(&bytes), Ok(4 + 44 + 4));
        let info = UnwindInfo::parse(&bytes, 0x2600).expect("valid unwind info");
        assert_eq!(
            (info.version, info.flags, info.prolog_size),
            (2, UNW_FLAG_EHANDLER, 0x30)
        );
        assert_eq!((info.frame_register, info.frame_offset), (5, 0x30));
        let ops: Vec<_> = info.codes.iter().map(|c| (c.prolog_offset, c.op)).collect();
        assert_eq!(
            ops,
            [
                (
                    0x05,
                    UnwindOp::Epilog {
                        code_offset: 5,
                        info: 1
                    }
                ),
                (
                    0x20,
                    UnwindOp::Epilog {
                        code_offset: 0x20,
                        info: 0
                    }
                ),
                (0x01, UnwindOp::PushNonVolatile { register: 5 }),
                (0x05, UnwindOp::Alloc { size: 0x20 }),
                (0x0c, UnwindOp::Alloc { size: 0x1000 }),
                (0x10, UnwindOp::Alloc { size: 0x10000 }),
                (0x12, UnwindOp::SetFramePointer),
                (
                    0x14,
                    UnwindOp::SaveNonVolatile {
                        register: 3,
                        offset: 0x30
                    }
                ),
                (
                    0x18,
                    UnwindOp::SaveNonVolatile {
                        register: 6,
                        offset: 0x12345
                    }
                ),
                (
                    0x1c,
                    UnwindOp::SaveXmm128 {
                        register: 6,
                        offset: 0x20
                    }
                ),
                (
                    0x20,
                    UnwindOp::SaveXmm128 {
                        register: 7,
                        offset: 0x1000
                    }
                ),
                (0x22, UnwindOp::PushMachineFrame { error_code: true }),
            ]
        );
        assert_eq!(
            info.handler,
            Some(ExceptionHandler {
                rva: 0x1500,
                data_rva: 0x2600 + 52,
            })
        );
        assert_eq!(info.chained, None);
    }

    #[test]
    fn rejects_malformed_unwind_info() {
        assert_eq!(
            UnwindInfo::parse(&[0x01, 0x00], 0),
            Err(UnwindError::Truncated)
        );
        assert_eq!(
            UnwindInfo::parse(&[0x03, 0, 0, 0], 0),
            Err(UnwindError::UnsupportedVersion(3))
        );
        // CountOfCodes = 2 but only one slot present.
        assert_eq!(
            UnwindInfo::parse(&[0x01, 0, 2, 0, 0x01, 0x50], 0),
            Err(UnwindError::Truncated)
        );
        // UWOP_ALLOC_LARGE needs 2 slots, only 1 counted.
        assert_eq!(
            UnwindInfo::parse(&[0x01, 0, 1, 0, 0x04, 0x01, 0, 0], 0),
            Err(UnwindError::InvalidCode { slot: 0, op: 1 })
        );
        // UWOP_EPILOG in version 1, reserved opcode 11, machframe info > 1.
        for (op_byte, op) in [(0x06u8, 6u8), (0x0b, 11), (0x2a, 10)] {
            assert_eq!(
                UnwindInfo::parse(&[0x01, 0, 1, 0, 0x00, op_byte, 0, 0], 0),
                Err(UnwindError::InvalidCode { slot: 0, op })
            );
        }
        // CHAININFO set but the parent record is cut off.
        assert_eq!(
            UnwindInfo::parse(&unwind_bytes(UNW_FLAG_CHAININFO, &[0; 8]), 0),
            Err(UnwindError::Truncated)
        );
    }

    #[test]
    fn parses_runtime_functions_into_sorted_ranges() {
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1018, UNWIND_A),
            (0x1060, 0x1070, UNWIND_A), // shares unwind info (linker folding)
            (0x1020, 0x1030, UNWIND_B),
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_UHANDLER, &[0x40, 0x10, 0, 0]),
        );
        let dir = parse(&file);
        assert_eq!(dir.record_count, 3);
        assert_eq!(kinds(&dir), [(PDATA_RVA, PdataDiagnosticKind::Unsorted)]);
        let begins: Vec<u32> = dir.entries.iter().map(|e| e.function.begin).collect();
        assert_eq!(begins, [0x1010, 0x1020, 0x1060]);
        assert!(dir.entries.iter().all(|e| e.primary == e.function.begin));
        let info = dir.entries[0].unwind.as_ref().expect("unwind info");
        assert_eq!(
            info.codes,
            [UnwindCode {
                prolog_offset: 4,
                op: UnwindOp::Alloc { size: 0x28 },
            }]
        );
        assert_eq!(dir.handler_rvas(), [0x1040]);
        assert_eq!(
            dir.entries[1].unwind.as_ref().and_then(|u| u.handler),
            Some(ExceptionHandler {
                rva: 0x1040,
                data_rva: UNWIND_B + 12,
            })
        );

        let table = dir.function_table(BASE);
        assert_eq!(table.function_count(), 3);
        assert_eq!(table.function_at(BASE + 0x1017), Some(BASE + 0x1010));
        assert_eq!(table.function_at(BASE + 0x1018), None);
        assert_eq!(table.function_at(BASE + 0x106f), Some(BASE + 0x1060));
        assert_eq!(table.function_at(BASE + 0x100f), None);
    }

    #[test]
    fn chained_and_indirect_chunks_map_to_their_primary() {
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1020, UNWIND_A),
            (0x1080, 0x1090, UNWIND_B),      // chained to 0x1010
            (0x10a0, 0x10b0, UNWIND_C),      // chained to 0x1080 (two links)
            (0x10c0, 0x10c8, PDATA_RVA + 1), // indirect: record 0 holds the unwind data
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        let parent = |begin: u32, end: u32, unwind: u32| {
            [begin.to_le_bytes(), end.to_le_bytes(), unwind.to_le_bytes()].concat()
        };
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_CHAININFO, &parent(0x1010, 0x1020, UNWIND_A)),
        );
        put(
            &mut file,
            UNWIND_C,
            &unwind_bytes(UNW_FLAG_CHAININFO, &parent(0x1080, 0x1090, UNWIND_B)),
        );
        let dir = parse(&file);
        assert!(dir.diagnostics.is_empty(), "{:?}", dir.diagnostics);
        let primaries: Vec<(u32, u32)> = dir
            .entries
            .iter()
            .map(|e| (e.function.begin, e.primary))
            .collect();
        assert_eq!(
            primaries,
            [
                (0x1010, 0x1010),
                (0x1080, 0x1010),
                (0x10a0, 0x1010),
                (0x10c0, 0x1010)
            ]
        );
        assert_eq!(
            dir.entries[1].unwind.as_ref().and_then(|u| u.chained),
            Some(RuntimeFunction {
                begin: 0x1010,
                end: 0x1020,
                unwind: UNWIND_A,
            })
        );
        assert_eq!(dir.entries[3].unwind, None, "indirect record");

        let table = dir.function_table(BASE);
        assert_eq!(table.function_count(), 1);
        assert_eq!(table.ranges().len(), 4);
        assert_eq!(table.function_at(BASE + 0x10a8), Some(BASE + 0x1010));
        assert_eq!(
            table.function_entries().collect::<Vec<_>>(),
            [BASE + 0x1010]
        );
    }

    #[test]
    fn broken_and_cyclic_chains_fall_back_to_own_function() {
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1020, UNWIND_A), // chained to 0x1030, which chains back to 0x1010
            (0x1030, 0x1040, UNWIND_B),
            (0x1050, 0x1060, UNWIND_C), // chained to a record that is not in the directory
            (0x1070, 0x1078, 0x1fff),   // indirect into nothing (RVA 0x1ffe is unmapped)
        ]);
        let link = |begin: u32, end: u32, unwind: u32| {
            [begin.to_le_bytes(), end.to_le_bytes(), unwind.to_le_bytes()].concat()
        };
        put(
            &mut file,
            UNWIND_A,
            &unwind_bytes(UNW_FLAG_CHAININFO, &link(0x1030, 0x1040, UNWIND_B)),
        );
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_CHAININFO, &link(0x1010, 0x1020, UNWIND_A)),
        );
        put(
            &mut file,
            UNWIND_C,
            &unwind_bytes(UNW_FLAG_CHAININFO, &link(0x1100, 0x1110, UNWIND_D)),
        );
        put(&mut file, UNWIND_D, &unwind_bytes(0, &[]));
        let dir = parse(&file);
        assert_eq!(
            kinds(&dir),
            [
                (0x1010, PdataDiagnosticKind::ChainTooDeep),
                (0x1030, PdataDiagnosticKind::ChainTooDeep),
                (0x1070, PdataDiagnosticKind::ChainBroken),
                (0x1050, PdataDiagnosticKind::ChainBroken),
            ]
        );
        assert!(dir.entries.iter().all(|e| e.primary == e.function.begin));
    }

    #[test]
    fn invalid_records_are_dropped_with_diagnostics() {
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1020, UNWIND_A),
            (0x1010, 0x1020, UNWIND_A), // exact duplicate: silently merged
            (0x1018, 0x1028, UNWIND_A), // overlaps
            (0x1030, 0x1030, UNWIND_A), // empty
            (0x1040, 0x4000, UNWIND_A), // past SizeOfImage
            (0x1050, 0x1058, 0x2ff0),   // unwind info unmapped (past .rdata)
            (0x1060, 0x1068, UNWIND_B), // bad version
            (0x1070, 0x1078, 0x27fe),   // unwind header crosses the end of .rdata
            (0x2010, 0x2020, UNWIND_A), // in .rdata
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        put(&mut file, UNWIND_B, &[0x05, 0, 0, 0]);
        let dir = parse(&file);
        assert_eq!(
            kinds(&dir),
            [
                (0x1030, PdataDiagnosticKind::EmptyRange),
                (0x1040, PdataDiagnosticKind::OutsideImage),
                (0x2010, PdataDiagnosticKind::NotExecutable),
                (0x1018, PdataDiagnosticKind::Overlap),
                (0x1050, PdataDiagnosticKind::Unwind(UnwindError::Unmapped)),
                (
                    0x1060,
                    PdataDiagnosticKind::Unwind(UnwindError::UnsupportedVersion(5))
                ),
                (0x1070, PdataDiagnosticKind::Unwind(UnwindError::Unmapped)),
            ]
        );
        let kept: Vec<(u32, bool)> = dir
            .entries
            .iter()
            .map(|e| (e.function.begin, e.unwind.is_some()))
            .collect();
        assert_eq!(
            kept,
            [
                (0x1010, true),
                (0x1050, false),
                (0x1060, false),
                (0x1070, false)
            ]
        );
    }

    #[test]
    fn malformed_directories_never_panic() {
        // Missing directory.
        let file = build_synthetic_pe64(false);
        assert_eq!(parse(&file), ExceptionDirectory::default());

        // Directory RVA outside every section.
        let mut file = pe_with_pdata(&[]);
        set_pdata_dir(&mut file, 0x9000, 24);
        let dir = parse(&file);
        assert!(dir.entries.is_empty());
        assert_eq!(
            kinds(&dir),
            [(0x9000, PdataDiagnosticKind::DirectoryUnmapped)]
        );

        // Directory running past the end of .rdata, with a size that is not a multiple of 12.
        let mut file = pe_with_pdata(&[(0x1010, 0x1020, UNWIND_A)]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        set_pdata_dir(&mut file, 0x27f4, 12 * 3 + 5);
        put(
            &mut file,
            0x27f4,
            &[0x10, 0x10, 0, 0, 0x20, 0x10, 0, 0, 0x00, 0x27, 0, 0],
        );
        let dir = parse(&file);
        assert_eq!(dir.record_count, 1);
        assert_eq!(
            kinds(&dir),
            [
                (0x27f4, PdataDiagnosticKind::TrailingBytes),
                (0x27f4, PdataDiagnosticKind::DirectoryTruncated),
            ]
        );
        assert_eq!(dir.entries.len(), 1);

        // Truncated file: records past the end of the file are not read.
        let mut file = pe_with_pdata(&[(0x1010, 0x1020, UNWIND_A); 4]);
        let pe = Pe64::parse(&file).expect("parses");
        file.truncate(file_offset(PDATA_RVA) + 18);
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        assert_eq!(dir.record_count, 1);

        // Garbage everywhere (headers parsed before the bytes are scrambled).
        let mut file = pe_with_pdata(&[]);
        set_pdata_dir(&mut file, 0x2000, 0x7f0);
        let pe = Pe64::parse(&file).expect("parses");
        let mut seed = 0x1234_5678u32;
        for rva in 0x2000..0x2800 {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            put(&mut file, rva, &[(seed >> 16) as u8]);
        }
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        assert!(
            dir.entries
                .windows(2)
                .all(|w| w[0].function.end <= w[1].function.begin)
        );
    }

    #[test]
    fn function_table_lookup_and_overlap_handling() {
        let r = |start, end, function| FunctionRange {
            start,
            end,
            function,
        };
        let table = FunctionTable::new(vec![
            r(0x30, 0x40, 0x10),
            r(0x10, 0x20, 0x10),
            r(0x18, 0x28, 0x18), // overlaps: dropped
            r(0x50, 0x50, 0x50), // empty: dropped
            r(0x10, 0x20, 0x10),
        ]);
        assert_eq!(table.ranges(), &[r(0x10, 0x20, 0x10), r(0x30, 0x40, 0x10)]);
        assert_eq!(table.function_count(), 1);
        assert_eq!(table.function_at(0x35), Some(0x10));
        assert_eq!(table.function_at(0x20), None);
        assert_eq!(table.function_at(0x0), None);
        assert!(FunctionTable::default().function_at(0x10).is_none());
    }

    #[test]
    fn seed_adds_pdata_functions_and_handlers_as_entry_points() {
        let mut file = pe_with_pdata(&[
            (0x1060, 0x1070, UNWIND_A),
            (0x1080, 0x1090, UNWIND_B),
            (0x10b0, 0x10b8, UNWIND_C),
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        // Handler at .text+0xa0, and one that points into .rdata (reported, not seeded).
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_EHANDLER, &[0xa0, 0x10, 0, 0]),
        );
        put(
            &mut file,
            UNWIND_C,
            &unwind_bytes(UNW_FLAG_EHANDLER, &[0x10, 0x20, 0, 0]),
        );
        let pe = Pe64::parse(&file).expect("parses");
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        assert_eq!(kinds(&dir), [(0x10b0, PdataDiagnosticKind::HandlerNotCode)]);
        assert_eq!(dir.handler_rvas(), [0x10a0, 0x2010]);
        let mut image = CodeImage::from_pe(&pe, &file);
        dir.seed(&mut image);
        assert_eq!(
            image.entry_points,
            [
                0x1010, 0x1020, 0x1030, 0x1040, 0x1060, 0x1080, 0x10a0, 0x10b0
            ]
            .map(|rva| BASE + rva)
        );
        assert_eq!(image.function_table.function_count(), 3);
    }

    #[test]
    fn parse_is_deterministic() {
        let mut file = pe_with_pdata(&[(0x1080, 0x1090, UNWIND_B), (0x1010, 0x1020, UNWIND_A)]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        put(
            &mut file,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_EHANDLER, &[0x10, 0x10, 0, 0]),
        );
        assert_eq!(parse(&file), parse(&file));
    }

    #[test]
    fn inverted_and_cross_section_ranges_are_dropped() {
        let mut file = pe_with_pdata(&[
            (0x1050, 0x1020, UNWIND_A), // inverted (begin > end)
            (0x1030, 0x1030, UNWIND_A), // empty (begin == end)
            (0x1010, 0x2010, UNWIND_A), // starts in .text, crosses into .rdata (still <= SizeOfImage)
            (0x1060, 0x1070, UNWIND_A), // valid interior function (must not be swallowed)
            (0x11f0, 0x1200, UNWIND_A), // valid function ending exactly at .text end
            (0x11f8, 0x1201, UNWIND_A), // 1 byte past .text end
        ]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        let dir = parse(&file);
        assert_eq!(
            kinds(&dir),
            [
                (PDATA_RVA, PdataDiagnosticKind::Unsorted),
                (0x1050, PdataDiagnosticKind::EmptyRange),
                (0x1030, PdataDiagnosticKind::EmptyRange),
                (0x1010, PdataDiagnosticKind::NotExecutable),
                (0x11f8, PdataDiagnosticKind::NotExecutable),
            ]
        );
        let ranges: Vec<(u32, u32)> = dir
            .entries
            .iter()
            .map(|e| (e.function.begin, e.function.end))
            .collect();
        assert_eq!(ranges, [(0x1060, 0x1070), (0x11f0, 0x1200)]);
    }

    #[test]
    fn unsorted_and_overlapping_permutations_are_deterministic() {
        let sorted = [
            (0x1010, 0x1020, UNWIND_A),
            (0x1010, 0x1020, UNWIND_B), // same range, different unwind -> dropped as overlap
            (0x1010, 0x1030, UNWIND_A), // same begin, wider end -> dropped as overlap
            (0x1018, 0x1028, UNWIND_A), // partial overlap -> dropped
            (0x1020, 0x1040, UNWIND_B), // adjacent -> kept
            (0x1028, 0x1038, UNWIND_A), // nested inside 0x1020..0x1040 -> dropped
            (0x1060, 0x1070, UNWIND_A), // disjoint -> kept
            (0x1060, 0x1070, UNWIND_A), // exact duplicate -> merged
        ];
        let permutations: [Vec<(u32, u32, u32)>; 3] = [
            sorted.to_vec(),
            sorted.iter().copied().rev().collect(),
            vec![
                sorted[6], sorted[2], sorted[4], sorted[0], sorted[7], sorted[5], sorted[1],
                sorted[3],
            ],
        ];

        let mut baseline_entries = None;
        let mut baseline_table = None;
        let mut baseline_handlers = None;
        for (idx, records) in permutations.iter().enumerate() {
            let mut file = pe_with_pdata(records);
            put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
            put(
                &mut file,
                UNWIND_B,
                &unwind_bytes(UNW_FLAG_EHANDLER, &[0x80, 0x10, 0, 0]),
            );
            let dir = parse(&file);
            assert_eq!(
                dir.diagnostics
                    .iter()
                    .any(|d| d.kind == PdataDiagnosticKind::Unsorted),
                idx != 0,
                "permutation {idx}"
            );
            assert!(
                dir.entries
                    .windows(2)
                    .all(|w| w[0].function.end <= w[1].function.begin)
            );
            let kept: Vec<(u32, u32, u32)> = dir
                .entries
                .iter()
                .map(|e| (e.function.begin, e.function.end, e.function.unwind))
                .collect();
            assert_eq!(
                kept,
                [
                    (0x1010, 0x1020, UNWIND_A),
                    (0x1020, 0x1040, UNWIND_B),
                    (0x1060, 0x1070, UNWIND_A),
                ]
            );
            let table = dir.function_table(BASE);
            let handlers = dir.handler_rvas();
            if let Some(ref prev) = baseline_entries {
                assert_eq!(&dir.entries, prev);
                assert_eq!(&table, baseline_table.as_ref().unwrap());
                assert_eq!(&handlers, baseline_handlers.as_ref().unwrap());
            } else {
                baseline_entries = Some(dir.entries);
                baseline_table = Some(table);
                baseline_handlers = Some(handlers);
            }
        }
    }

    #[test]
    fn truncated_directory_and_huge_sizes_never_panic() {
        // Sub-record directory on a mapped RVA (< 12 bytes).
        let mut file = pe_with_pdata(&[]);
        set_pdata_dir(&mut file, PDATA_RVA, 5);
        let dir = parse(&file);
        assert_eq!(dir.record_count, 0);
        assert_eq!(
            kinds(&dir),
            [(PDATA_RVA, PdataDiagnosticKind::TrailingBytes)]
        );

        // Sub-record directory on an unmapped RVA (< 12 bytes).
        let mut file = pe_with_pdata(&[]);
        set_pdata_dir(&mut file, 0x9000, 5);
        let dir = parse(&file);
        assert_eq!(dir.record_count, 0);
        assert_eq!(
            kinds(&dir),
            [
                (0x9000, PdataDiagnosticKind::TrailingBytes),
                (0x9000, PdataDiagnosticKind::DirectoryUnmapped),
            ]
        );

        // Mapped directory RVA with fewer than 12 bytes remaining in .rdata.
        let mut file = pe_with_pdata(&[]);
        set_pdata_dir(&mut file, 0x27fc, 24);
        let dir = parse(&file);
        assert_eq!(dir.record_count, 0);
        assert_eq!(
            kinds(&dir),
            [(0x27fc, PdataDiagnosticKind::DirectoryTruncated)]
        );

        // Full record fits, but trailing bytes run past .rdata end.
        let mut file = pe_with_pdata(&[]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        set_pdata_dir(&mut file, 0x27f4, 15);
        put(
            &mut file,
            0x27f4,
            &[0x10, 0x10, 0, 0, 0x20, 0x10, 0, 0, 0x00, 0x27, 0, 0],
        );
        let dir = parse(&file);
        assert_eq!(dir.record_count, 1);
        assert_eq!(dir.entries.len(), 1);
        assert_eq!(
            kinds(&dir),
            [
                (0x27f4, PdataDiagnosticKind::TrailingBytes),
                (0x27f4, PdataDiagnosticKind::DirectoryTruncated),
            ]
        );

        // Malicious u32::MAX directory size must not OOM or wrap u32.
        let mut file = pe_with_pdata(&[]);
        put(&mut file, UNWIND_A, &unwind_bytes(0, &[]));
        set_pdata_dir(&mut file, 0x27f4, u32::MAX);
        put(
            &mut file,
            0x27f4,
            &[0x10, 0x10, 0, 0, 0x20, 0x10, 0, 0, 0x00, 0x27, 0, 0],
        );
        let dir = parse(&file);
        assert_eq!(dir.record_count, 1);
        assert_eq!(dir.entries.len(), 1);
        assert_eq!(
            kinds(&dir),
            [
                (0x27f4, PdataDiagnosticKind::TrailingBytes),
                (0x27f4, PdataDiagnosticKind::DirectoryTruncated),
            ]
        );

        // Truncating the backing file at every byte around the directory and unwind info never panics.
        let mut full = pe_with_pdata(&[
            (0x1010, 0x1020, UNWIND_A),
            (0x1030, 0x1040, UNWIND_B),
            (0x1050, 0x1060, PDATA_RVA + 1),
        ]);
        put(&mut full, UNWIND_A, &unwind_bytes(0, &[]));
        put(
            &mut full,
            UNWIND_B,
            &unwind_bytes(UNW_FLAG_EHANDLER, &[0x10, 0x10, 0, 0]),
        );
        let pe = Pe64::parse(&full).expect("parses");
        for cut in [
            0usize,
            file_offset(PDATA_RVA),
            file_offset(PDATA_RVA) + 1,
            file_offset(PDATA_RVA) + 11,
            file_offset(PDATA_RVA) + 12,
            file_offset(PDATA_RVA) + 23,
            file_offset(UNWIND_A),
            file_offset(UNWIND_A) + 3,
            file_offset(UNWIND_A) + 4,
            file_offset(UNWIND_B) + 10,
            full.len(),
        ] {
            let dir = ExceptionDirectory::from_pe(&pe, &full[..cut]);
            assert!(
                dir.entries
                    .windows(2)
                    .all(|w| w[0].function.end <= w[1].function.begin),
                "cut {cut}"
            );
        }
    }

    #[test]
    fn bss_gap_cannot_be_bridged_across_sections() {
        // Shrink .rdata raw_size to 0x600 while virtual_size remains 0x800, leaving an unmapped
        // BSS gap at 0x2600..0x2800. A record starting at 0x25fc crosses into that BSS gap and
        // must not read raw bytes past .rdata's initialized extent.
        let mut file = pe_with_pdata(&[(0x1010, 0x1020, 0x25fe)]);
        // .rdata section header starts at 0x1b0; SizeOfRawData is at +16 (0x1c0).
        file[0x1c0..0x1c4].copy_from_slice(&0x600u32.to_le_bytes());
        put(&mut file, 0x25fc, &unwind_bytes(0, &[]));
        let dir = parse(&file);
        assert_eq!(dir.entries.len(), 1);
        assert_eq!(dir.entries[0].unwind, None);
        assert_eq!(
            kinds(&dir),
            [(0x1010, PdataDiagnosticKind::Unwind(UnwindError::Unmapped))]
        );
    }

    #[test]
    fn checked_arithmetic_guards_unwind_chains_and_va_overflow() {
        // Handler data_rva overflow near u32::MAX.
        let bytes = unwind_bytes(UNW_FLAG_EHANDLER, &[0x10, 0x10, 0, 0]);
        assert_eq!(
            UnwindInfo::parse(&bytes, u32::MAX - 4),
            Err(UnwindError::Truncated)
        );
        // Maximum CountOfCodes (255) on a short buffer returns Truncated without overflow.
        assert_eq!(
            UnwindInfo::parse(&[0x01, 0, u8::MAX, 0], 0x2000),
            Err(UnwindError::Truncated)
        );

        // Null unwind RVA (0) and indirect-to-0 (1) must not read the MZ DOS header;
        // chained parent with inverted begin > end must be rejected as ChainBroken.
        let mut file = pe_with_pdata(&[
            (0x1010, 0x1020, 0),        // null unwind RVA
            (0x1030, 0x1040, 1),        // indirect to RVA 0
            (0x1050, 0x1060, UNWIND_A), // chained to inverted parent
            (0x1070, 0x1080, UNWIND_B), // valid primary
        ]);
        let inverted_parent = [
            0x1070u32.to_le_bytes(),
            0x1060u32.to_le_bytes(), // inverted: begin > end
            UNWIND_B.to_le_bytes(),
        ]
        .concat();
        put(
            &mut file,
            UNWIND_A,
            &unwind_bytes(UNW_FLAG_CHAININFO, &inverted_parent),
        );
        put(&mut file, UNWIND_B, &unwind_bytes(0, &[]));
        let dir = parse(&file);
        assert_eq!(
            kinds(&dir),
            [
                (0x1010, PdataDiagnosticKind::Unwind(UnwindError::Unmapped)),
                (0x1030, PdataDiagnosticKind::ChainBroken),
                (0x1050, PdataDiagnosticKind::ChainBroken),
            ]
        );
        assert!(dir.entries.iter().all(|e| e.primary == e.function.begin));

        // Near-u64::MAX image base skips overflowing ranges cleanly instead of saturating.
        let near_max_base = u64::MAX - 0x1038;
        let table = dir.function_table(near_max_base);
        assert_eq!(
            table.ranges(),
            &[FunctionRange {
                start: near_max_base + 0x1010,
                end: near_max_base + 0x1020,
                function: near_max_base + 0x1010,
            }]
        );
    }

    #[test]
    fn function_table_boundaries_and_gap_lookup() {
        let r = |start, end, function| FunctionRange {
            start,
            end,
            function,
        };
        let ranges = vec![
            r(u64::MAX - 0x10, u64::MAX, u64::MAX - 0x10),
            r(0x40, 0x20, 0x40), // inverted: dropped
            r(0x20, 0x20, 0x20), // empty: dropped
            r(0x0, 0x10, 0x0),
            r(0x10, 0x20, 0x0), // adjacent to [0, 0x10)
            r(0x30, 0x40, 0x30),
        ];
        let mut reversed = ranges.clone();
        reversed.reverse();
        let table = FunctionTable::new(ranges);
        assert_eq!(table, FunctionTable::new(reversed));

        assert_eq!(table.function_at(0), Some(0));
        assert_eq!(table.function_at(0x10), Some(0));
        assert_eq!(table.function_at(u64::MAX - 1), Some(u64::MAX - 0x10));
        assert_eq!(table.function_at(u64::MAX), None);

        // Adjacent ranges have no gap; disjoint ranges report exact gap bounds.
        assert_eq!(table.gap_at(0), None);
        assert_eq!(table.gap_at(0x10), None);
        assert_eq!(table.gap_at(0x20), Some((0x20, 0x30)));
        assert_eq!(table.gap_at(0x2f), Some((0x20, 0x30)));
        assert_eq!(table.gap_at(0x30), None);
        assert_eq!(table.gap_at(0x40), Some((0x40, u64::MAX - 0x10)));
        assert_eq!(table.gap_at(u64::MAX), None);
    }
}
