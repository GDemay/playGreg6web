//! Win64 exception-handling metadata: the language-specific data behind `UNWIND_INFO`.
//!
//! A `.pdata` record whose `UNWIND_INFO` has `UNW_FLAG_EHANDLER`/`UNW_FLAG_UHANDLER` names a
//! language handler (an RVA of code) followed by handler-specific data (see [`crate::pdata`]).
//! The handler RVA never points at the handler itself but at code inside the image:
//!
//! - an import thunk `jmp [rip+slot]` whose IAT slot is a known language handler import
//!   (`VCRUNTIME140.dll!__C_specific_handler`, `__CxxFrameHandler3`,
//!   `VCRUNTIME140_1.dll!__CxxFrameHandler4`), or
//! - a static wrapper linked into the image (MSVC's `__GSHandlerCheck_SEH`/`_EH`/`_EH4`) that
//!   checks the stack cookie and then calls such a thunk with the same dispatcher context, so the
//!   data has the inner handler's format; or
//! - `__GSHandlerCheck`, a wrapper with no inner handler (cookie check only, no language data).
//!   It is recognised because it calls the same cookie-check helper as the identified wrappers.
//!
//! [`EhMetadata::from_pe`] identifies every handler that way (bounded decoding of the handler
//! code, no discovery needed) and parses the data of the three supported formats:
//!
//! - `__C_specific_handler`: an inline SCOPE_TABLE, `u32 Count` then `Count` records
//!   `{BeginAddress, EndAddress, HandlerAddress, JumpTarget}` (RVAs). `JumpTarget == 0` is a
//!   `__finally` whose termination handler is `HandlerAddress`; otherwise `JumpTarget` is the
//!   `__except` body inside the function and `HandlerAddress` is the filter funclet, or the
//!   constant `1` (`EXCEPTION_EXECUTE_HANDLER`).
//! - `__CxxFrameHandler3`: the data is the RVA of a `FuncInfo` (`magicNumber:29, bbtFlags:3` with
//!   magic `0x19930520..=0x19930522`, `maxState`, unwind map, try-block map with handler arrays,
//!   IP-to-state map, unwind-help offset, ES type list (>= `..21`), EH flags (>= `..22`); all
//!   tables are RVAs of fixed-size records). Catch funclets return the continuation address in
//!   `rax` (`lea rax, [continuation]`), so FH3 has no continuation addresses in metadata: discovery
//!   finds them as address-taken interior code (see [`crate::discover`]).
//! - `__CxxFrameHandler4`: the data is the RVA of a compressed `FuncInfo4`. A header byte
//!   (`isCatch`, `isSeparated`, `BBT`, `UnwindMap`, `TryBlockMap`, `EHs`, `NoExcept`) says which
//!   fields follow: `bbtFlags` (compressed), `dispUnwindMap`/`dispTryBlockMap` (raw `i32` RVAs),
//!   `dispIPtoStateMap` (raw; for separated code the RVA of a `{segment RVA, map RVA}` table) and,
//!   for catch funclets, `dispFrame` (compressed). The unwind map holds
//!   `nextOffset:30, type:2` entries (`type` 1/2 = destructor RVA + compressed object offset,
//!   3 = cleanup funclet RVA); try blocks hold compressed `tryLow, tryHigh, catchHigh` plus the RVA
//!   of a handler array; each handler has a header byte (`adjectives`, `dispType`,
//!   `dispCatchObj`, `contIsRVA`, `contAddr:2`), optional fields, the catch funclet RVA and 0..2
//!   continuation addresses (function-relative compressed values, or raw RVAs with `contIsRVA`).
//!   IP-to-state entries are compressed `(ip delta, state + 1)` pairs relative to the function
//!   start. Compressed unsigned integers store their length in the low bits of the first byte
//!   (`...0` 1 byte / 7 bits, `..01` 2 / 14, `.011` 3 / 21, `0111` 4 / 28, `1111` 5 bytes, value
//!   in the last four).
//!
//! Every read is bounds-checked against the PE's section mapping and every count is capped;
//! malformed or unrecognised data becomes an [`EhDiagnostic`], never a panic.
//!
//! # What discovery gets
//! [`EhMetadata::seed`] feeds [`CodeImage::eh`]: funclets (catch handlers, cleanup funclets,
//! destructors, filters, termination handlers) become entry points (most already are, through
//! `.pdata`), and **resume targets** — FH4 catch continuations and `__except` bodies, the places
//! where `RtlUnwindEx`/`RtlRestoreContext` resume execution inside a function — are recorded with
//! their owning function (the `.pdata` function containing them) so discovery makes them block
//! leaders and lists them in [`crate::cfg::Function::resume_targets`]. The IAT slots of the
//! `setjmp` family are recorded as well: the instruction after a call through them is a resume
//! target of the caller (`longjmp` returns there).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use iced_x86::{Decoder, DecoderOptions, FlowControl, Instruction, OpKind};

use crate::code_image::{CodeImage, Region};
use crate::pdata::{
    ExceptionDirectory, ExceptionHandler, PdataEntry, RUNTIME_FUNCTION_LEN, RuntimeFunction,
    UnwindInfo,
};
use crate::pe::{ImportedSymbol, Pe64};

/// Language-handler imports and the data format they consume.
const HANDLER_IMPORTS: [(&str, HandlerKind); 6] = [
    ("__C_specific_handler", HandlerKind::CSpecific),
    ("__C_specific_handler_noexcept", HandlerKind::CSpecific),
    ("__CxxFrameHandler", HandlerKind::CxxFrameHandler3),
    ("__CxxFrameHandler2", HandlerKind::CxxFrameHandler3),
    ("__CxxFrameHandler3", HandlerKind::CxxFrameHandler3),
    ("__CxxFrameHandler4", HandlerKind::CxxFrameHandler4),
];

/// Imports whose call returns a second time when `longjmp` restores the saved context.
pub const SETJMP_IMPORTS: [&str; 5] = [
    "_setjmp",
    "_setjmpex",
    "setjmp",
    "__intrinsic_setjmp",
    "__intrinsic_setjmpex",
];

/// `FuncInfo` magic numbers of `__CxxFrameHandler3` (`magicNumber:29`).
const FH3_MAGIC_MIN: u32 = 0x1993_0520;
const FH3_MAGIC_MAX: u32 = 0x1993_0522;
const FH3_MAGIC_MASK: u32 = 0x1FFF_FFFF;
/// `HandlerAddress` of an `__except (EXCEPTION_EXECUTE_HANDLER)` scope.
const EXCEPTION_EXECUTE_HANDLER: u32 = 1;

/// Upper bound for any table count (scope records, states, try blocks, handlers, IP map entries,
/// separated segments). Real functions stay far below; garbage counts are rejected.
const MAX_TABLE_ENTRIES: u32 = 1 << 16;
/// Byte size of the `C_SCOPE_TABLE` record count prefix (`u32 Count`).
const SCOPE_TABLE_HEADER_LEN: u32 = 4;
/// Byte size of one `C_SCOPE_TABLE_ENTRY` (`BeginAddress`, `EndAddress`, `HandlerAddress`,
/// `JumpTarget`).
const SCOPE_RECORD_LEN: usize = 16;
/// Cap initial `Vec::with_capacity` pre-allocations so malformed/truncated tables with large
/// counts cannot trigger large upfront allocations before entries are verified.
const INITIAL_TABLE_CAPACITY: usize = 256;
/// Instructions decoded per handler when identifying it.
const HANDLER_SCAN_LIMIT: usize = 96;
/// Bit 0 of `UnwindInfoAddress`: points at an indirect `RUNTIME_FUNCTION` record at `rva - 1`.
const RUNTIME_FUNCTION_INDIRECT: u32 = 0x1;
/// Longest `UNW_FLAG_CHAININFO` / indirect `RUNTIME_FUNCTION` chain followed before giving up.
const MAX_CHAIN_DEPTH: usize = 32;
/// Byte length of an `UNWIND_INFO` header before its `UNWIND_CODE` slots.
const UNWIND_HEADER_LEN: usize = 4;
/// Maximum direct `jmp rel32`/`rel8` hops followed when identifying an import thunk.
const MAX_THUNK_HOPS: usize = 4;

// FH4 header bits.
const FH4_IS_CATCH: u8 = 0x01;
const FH4_IS_SEPARATED: u8 = 0x02;
const FH4_BBT: u8 = 0x04;
const FH4_UNWIND_MAP: u8 = 0x08;
const FH4_TRY_BLOCK_MAP: u8 = 0x10;
// FH4 handler header bits.
const FH4_HANDLER_ADJECTIVES: u8 = 0x01;
const FH4_HANDLER_DISP_TYPE: u8 = 0x02;
const FH4_HANDLER_DISP_CATCH_OBJ: u8 = 0x04;
const FH4_HANDLER_CONT_IS_RVA: u8 = 0x08;
const FH4_HANDLER_CONT_SHIFT: u8 = 4;

/// Data format of a language handler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HandlerKind {
    /// `__C_specific_handler` (`_noexcept`): SEH scope table.
    CSpecific,
    /// `__CxxFrameHandler3` (and the older `__CxxFrameHandler`/`2`): `FuncInfo` RVA.
    CxxFrameHandler3,
    /// `__CxxFrameHandler4`: compressed `FuncInfo4` RVA.
    CxxFrameHandler4,
    /// `__GSHandlerCheck`: stack-cookie check only, no language data.
    GsCheck,
}

impl HandlerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CSpecific => "c_specific",
            Self::CxxFrameHandler3 => "cxx3",
            Self::CxxFrameHandler4 => "cxx4",
            Self::GsCheck => "gs_check",
        }
    }

    /// Kind of the language handler import `name` (`None` for other imports).
    pub fn from_import(name: &str) -> Option<Self> {
        HANDLER_IMPORTS
            .iter()
            .find(|(import, _)| *import == name)
            .map(|(_, kind)| *kind)
    }
}

/// An identified handler routine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Handler {
    pub kind: HandlerKind,
    /// The routine is a static wrapper in the image (MSVC `__GSHandlerCheck*`) rather than an
    /// import thunk; the data keeps the inner handler's format.
    pub wrapper: bool,
}

/// Category of an EH metadata problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EhDiagnosticKind {
    /// The handler routine is not a recognised language handler (custom handler or unknown
    /// wrapper); its data is not parsed. Reported once per handler RVA.
    UnknownHandler,
    /// The handler routine reaches more than one language handler. Reported once per handler.
    AmbiguousHandler,
    /// The data, or a table it points to, is unmapped or runs past the mapped bytes.
    Truncated,
    /// `FuncInfo` magic number outside `0x19930520..=0x19930522`.
    UnknownMagic,
    /// A table count above the sanity limit (or negative).
    TooLarge,
    /// A funclet, handler or resume-target RVA that is not inside executable code.
    NotCode,
    /// A reserved encoding (FH4 `contAddr == 3`).
    Reserved,
    /// FH4 separated code: no IP-to-state map for this function's segment.
    MissingSegment,
    /// An FH4 unwind-map `nextOffset` that does not land on an entry.
    BadUnwindLink,
}

impl EhDiagnosticKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnknownHandler => "unknown_handler",
            Self::AmbiguousHandler => "ambiguous_handler",
            Self::Truncated => "truncated",
            Self::UnknownMagic => "unknown_magic",
            Self::TooLarge => "too_large",
            Self::NotCode => "not_code",
            Self::Reserved => "reserved",
            Self::MissingSegment => "missing_segment",
            Self::BadUnwindLink => "bad_unwind_link",
        }
    }
}

/// A non-fatal EH metadata problem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EhDiagnostic {
    /// `BeginAddress` of the function, or the handler RVA for handler-level problems.
    pub rva: u32,
    /// The offending RVA (table, funclet, target); equal to `rva` when there is none.
    pub address: u32,
    pub kind: EhDiagnosticKind,
}

impl fmt::Display for EhDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} at RVA 0x{:08X} (address 0x{:08X})",
            self.kind.as_str(),
            self.rva,
            self.address
        )
    }
}

/// Handler of one SEH scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScopeHandler {
    /// `__except (filter)`: RVA of the filter funclet.
    Filter(u32),
    /// `__except (constant)`: the filter is a constant (`1` = `EXCEPTION_EXECUTE_HANDLER`).
    Constant(u32),
    /// `__finally`: RVA of the termination handler.
    Finally(u32),
}

/// One `C_SCOPE_TABLE` record (RVAs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeRecord {
    pub begin: u32,
    pub end: u32,
    pub handler: ScopeHandler,
    /// `__except` body (0 for `__finally`).
    pub jump_target: u32,
}

/// `__CxxFrameHandler3` unwind map entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnwindEntry3 {
    pub to_state: i32,
    /// RVA of the cleanup funclet (0 = none).
    pub action: u32,
}

/// `__CxxFrameHandler3` catch handler (`HandlerType`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CatchHandler3 {
    pub adjectives: u32,
    /// RVA of the type descriptor (0 = `catch (...)`).
    pub type_descriptor: u32,
    /// Frame offset of the catch object.
    pub catch_object: i32,
    /// RVA of the catch funclet.
    pub handler: u32,
    /// Establisher frame offset.
    pub frame: u32,
}

/// `__CxxFrameHandler3` try block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TryBlock3 {
    pub try_low: i32,
    pub try_high: i32,
    pub catch_high: i32,
    pub handlers: Vec<CatchHandler3>,
}

/// Decoded `__CxxFrameHandler3` `FuncInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncInfo3 {
    /// RVA of the structure.
    pub rva: u32,
    pub magic: u32,
    pub bbt_flags: u8,
    pub unwind_map: Vec<UnwindEntry3>,
    pub try_blocks: Vec<TryBlock3>,
    /// `(ip RVA, state)` pairs.
    pub ip_to_state: Vec<(u32, i32)>,
    pub unwind_help: i32,
    /// RVA of the exception-specification type list (0 = none or pre-`..21` magic).
    pub es_type_list: u32,
    pub eh_flags: u32,
}

/// `__CxxFrameHandler4` unwind map entry type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnwindKind4 {
    /// No unwind action.
    None,
    /// Destructor RVA + frame offset of the object.
    DtorWithObj,
    /// Destructor RVA + frame offset of a pointer to the object.
    DtorWithPtrToObj,
    /// RVA of a cleanup funclet.
    Funclet,
}

/// `__CxxFrameHandler4` unwind map entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnwindEntry4 {
    pub kind: UnwindKind4,
    /// State reached after this entry's action (-1 = none).
    pub next_state: i32,
    /// Destructor or cleanup funclet RVA (0 for [`UnwindKind4::None`]).
    pub action: u32,
    /// Frame offset of the object (destructor entries).
    pub object: u32,
}

/// `__CxxFrameHandler4` catch handler (`HandlerType4`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatchHandler4 {
    pub header: u8,
    pub adjectives: u32,
    pub type_descriptor: u32,
    pub catch_object: u32,
    /// RVA of the catch funclet.
    pub handler: u32,
    /// Continuation addresses as RVAs (empty: the funclet returns the continuation in `rax`).
    pub continuations: Vec<u32>,
}

/// `__CxxFrameHandler4` try block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TryBlock4 {
    pub try_low: u32,
    pub try_high: u32,
    pub catch_high: u32,
    /// RVA of the handler array.
    pub handler_array: u32,
    pub handlers: Vec<CatchHandler4>,
}

/// Decoded `__CxxFrameHandler4` `FuncInfo4`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuncInfo4 {
    /// RVA of the compressed structure.
    pub rva: u32,
    pub header: u8,
    pub bbt_flags: u32,
    /// RVAs of the unwind and try-block maps (0 = absent).
    pub unwind_map_rva: u32,
    pub try_block_map_rva: u32,
    /// RVA of the IP-to-state map used for this function (after segment selection).
    pub ip_map_rva: u32,
    /// Establisher frame offset (catch funclets only).
    pub frame: Option<u32>,
    pub unwind_map: Vec<UnwindEntry4>,
    pub try_blocks: Vec<TryBlock4>,
    /// `(ip RVA, state)` pairs.
    pub ip_to_state: Vec<(u32, i32)>,
}

impl FuncInfo4 {
    pub fn is_catch(&self) -> bool {
        self.header & FH4_IS_CATCH != 0
    }

    pub fn is_separated(&self) -> bool {
        self.header & FH4_IS_SEPARATED != 0
    }
}

/// Parsed language-specific data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LanguageData {
    /// No data (`__GSHandlerCheck`), or the data could not be parsed (see diagnostics).
    None,
    ScopeTable(Vec<ScopeRecord>),
    Cxx3(FuncInfo3),
    Cxx4(FuncInfo4),
}

/// Role of a funclet (a separately called piece of code the language handler runs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FuncletKind {
    /// C++ catch block; returns the continuation (address in `rax`, or an index for FH4
    /// handlers with metadata continuations).
    Catch,
    /// C++ unwind action (state cleanup funclet).
    Cleanup,
    /// Destructor called by FH4 for an unwind-map object.
    Destructor,
    /// SEH `__except` filter.
    Filter,
    /// SEH `__finally` termination handler.
    Finally,
}

impl FuncletKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Catch => "catch",
            Self::Cleanup => "cleanup",
            Self::Destructor => "destructor",
            Self::Filter => "filter",
            Self::Finally => "finally",
        }
    }
}

/// Why a resume target exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResumeKind {
    /// C++ catch continuation (FH4 metadata).
    CatchContinuation,
    /// SEH `__except` body.
    ExceptBody,
}

/// One function with language-specific EH data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EhFunction {
    /// `BeginAddress` / `EndAddress` of the `.pdata` record carrying the handler.
    pub begin: u32,
    pub end: u32,
    /// Handler routine RVA.
    pub handler_rva: u32,
    pub handler: Handler,
    /// RVA of the language-specific data (right after the handler RVA in `UNWIND_INFO`).
    pub data_rva: u32,
    pub data: LanguageData,
}

impl EhFunction {
    /// Funclets referenced by the data, `(rva, kind)`, sorted and deduplicated.
    pub fn funclets(&self) -> Vec<(u32, FuncletKind)> {
        let mut out = BTreeSet::new();
        match &self.data {
            LanguageData::None => {}
            LanguageData::ScopeTable(records) => {
                for record in records {
                    match record.handler {
                        ScopeHandler::Filter(rva) => out.insert((rva, FuncletKind::Filter)),
                        ScopeHandler::Finally(rva) => out.insert((rva, FuncletKind::Finally)),
                        ScopeHandler::Constant(_) => false,
                    };
                }
            }
            LanguageData::Cxx3(info) => {
                for entry in info.unwind_map.iter().filter(|e| e.action != 0) {
                    out.insert((entry.action, FuncletKind::Cleanup));
                }
                for handler in info.try_blocks.iter().flat_map(|t| &t.handlers) {
                    out.insert((handler.handler, FuncletKind::Catch));
                }
            }
            LanguageData::Cxx4(info) => {
                for entry in &info.unwind_map {
                    let kind = match entry.kind {
                        UnwindKind4::None => continue,
                        UnwindKind4::Funclet => FuncletKind::Cleanup,
                        UnwindKind4::DtorWithObj | UnwindKind4::DtorWithPtrToObj => {
                            FuncletKind::Destructor
                        }
                    };
                    out.insert((entry.action, kind));
                }
                for handler in info.try_blocks.iter().flat_map(|t| &t.handlers) {
                    out.insert((handler.handler, FuncletKind::Catch));
                }
            }
        }
        out.into_iter().collect()
    }

    /// Resume targets named by the data, `(rva, kind)`, sorted and deduplicated.
    pub fn resume_targets(&self) -> Vec<(u32, ResumeKind)> {
        let mut out = BTreeSet::new();
        match &self.data {
            LanguageData::ScopeTable(records) => {
                for record in records.iter().filter(|r| r.jump_target != 0) {
                    out.insert((record.jump_target, ResumeKind::ExceptBody));
                }
            }
            LanguageData::Cxx4(info) => {
                for handler in info.try_blocks.iter().flat_map(|t| &t.handlers) {
                    for &rva in &handler.continuations {
                        out.insert((rva, ResumeKind::CatchContinuation));
                    }
                }
            }
            LanguageData::None | LanguageData::Cxx3(_) => {}
        }
        out.into_iter().collect()
    }
}

/// Aggregate counts of an [`EhMetadata`] (for the discovery summary).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EhStats {
    /// Functions per handler kind (`as_str` order of [`HandlerKind`]), and with an unrecognised
    /// handler.
    pub c_specific: usize,
    pub cxx3: usize,
    pub cxx4: usize,
    pub gs_check: usize,
    pub unknown: usize,
    /// Functions whose data parsed without diagnostics.
    pub parsed: usize,
    /// Distinct funclets and metadata resume targets.
    pub funclets: usize,
    pub resume_targets: usize,
    pub diagnostics: usize,
}

/// What [`EhMetadata::seed`] hands to discovery (see [`CodeImage::eh`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EhSeeds {
    /// `(owner function entry VA, resume target VA)`, sorted and deduplicated. The owner is the
    /// `.pdata` function containing the target.
    pub resume_targets: Vec<(u64, u64)>,
    /// IAT slot VAs of the `setjmp` family ([`SETJMP_IMPORTS`]), sorted.
    pub setjmp_slots: Vec<u64>,
    pub stats: EhStats,
}

/// EH metadata of an image.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EhMetadata {
    /// Identified handler routines by RVA (unrecognised ones are absent, see diagnostics).
    pub handlers: BTreeMap<u32, Handler>,
    /// Every `.pdata` record with a handler, sorted by `begin`. Records whose handler is
    /// unrecognised are listed with [`LanguageData::None`] in [`Self::unrecognised`] instead.
    pub functions: Vec<EhFunction>,
    /// `BeginAddress` of records whose handler could not be identified.
    pub unrecognised: Vec<u32>,
    /// IAT slot RVAs of `setjmp`-family imports.
    pub setjmp_slots: Vec<u32>,
    pub diagnostics: Vec<EhDiagnostic>,
}

/// Bounds-checked RVA reads.
struct Reader<'p, 'f> {
    pe: &'p Pe64,
    file: &'f [u8],
}

/// Convert a 32-bit RVA to a 64-bit VA with checked addition.
fn rva_to_va(base: u64, rva: u32) -> Option<u64> {
    base.checked_add(u64::from(rva))
}

/// Decode a little-endian `u32` at `offset` within `bytes`, returning `None` if out of bounds.
fn le_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let b = bytes.get(offset..end)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Compute a safe initial `Vec` capacity bounded by both remaining bytes and [`INITIAL_TABLE_CAPACITY`].
fn bounded_capacity(n: usize, remaining_bytes: usize, min_entry_bytes: usize) -> usize {
    let max_fit = if min_entry_bytes == 0 {
        0
    } else {
        remaining_bytes / min_entry_bytes
    };
    n.min(max_fit).min(INITIAL_TABLE_CAPACITY)
}

impl<'f> Reader<'_, 'f> {
    /// Bytes from `rva` to the end of its section's mapped file data (clamped to `SizeOfImage`).
    fn tail(&self, rva: u32) -> Option<&'f [u8]> {
        if rva >= self.pe.optional.size_of_image {
            return None;
        }
        let offset = self.pe.rva_to_file_offset(rva)?;
        let section_extent = |s: &crate::pe::SectionHeader| {
            if s.virtual_size == 0 {
                s.raw_size
            } else {
                s.virtual_size.min(s.raw_size)
            }
        };
        let section = self.pe.sections.iter().find(|s| {
            let extent = section_extent(s);
            rva.checked_sub(s.virtual_address)
                .is_some_and(|delta| delta < extent)
        });
        let (end_offset, max_rva_len) = match section {
            Some(s) => {
                let extent = section_extent(s);
                let raw_end = usize::try_from(s.raw_offset)
                    .ok()?
                    .checked_add(usize::try_from(extent).ok()?)?;
                let delta = rva.checked_sub(s.virtual_address)?;
                let section_rem = extent.checked_sub(delta)?;
                (raw_end, section_rem)
            }
            // Headers.
            None => {
                let hdr_end = usize::try_from(self.pe.optional.size_of_headers).ok()?;
                let hdr_rem = self.pe.optional.size_of_headers.checked_sub(rva)?;
                (hdr_end, hdr_rem)
            }
        };
        let image_rem = self.pe.optional.size_of_image.checked_sub(rva)?;
        let rva_bound = usize::try_from(max_rva_len.min(image_rem)).ok()?;
        let file_end = end_offset.min(self.file.len());
        let clamped_end = offset
            .checked_add(rva_bound)
            .map_or(file_end, |e| e.min(file_end));
        self.file.get(offset..clamped_end)
    }

    fn read(&self, rva: u32, len: usize) -> Option<&'f [u8]> {
        let len_u32 = u32::try_from(len).ok()?;
        let end_rva = rva.checked_add(len_u32)?;
        if end_rva > self.pe.optional.size_of_image {
            return None;
        }
        self.tail(rva)?.get(..len)
    }

    fn u32(&self, rva: u32) -> Option<u32> {
        let b = self.read(rva, 4)?;
        le_u32(b, 0)
    }

    fn unwind_info(&self, rva: u32) -> Option<UnwindInfo> {
        if rva == 0 {
            return None;
        }
        let header = self.read(rva, UNWIND_HEADER_LEN)?;
        let len = UnwindInfo::encoded_len(header).ok()?;
        let bytes = self.read(rva, len)?;
        UnwindInfo::parse(bytes, rva).ok()
    }

    fn indirect_rf(&self, unwind: u32) -> Option<RuntimeFunction> {
        let target = unwind.checked_sub(RUNTIME_FUNCTION_INDIRECT)?;
        if target == 0 {
            return None;
        }
        RuntimeFunction::parse(self.read(target, RUNTIME_FUNCTION_LEN)?)
    }

    /// Resolve the [`ExceptionHandler`] for `entry`, following `UNW_FLAG_CHAININFO` and indirect
    /// `RUNTIME_FUNCTION` links (including inline chained records in `.xdata`) up to
    /// [`MAX_CHAIN_DEPTH`] hops with cycle detection.
    fn resolve_entry_handler(&self, entry: &PdataEntry) -> Option<ExceptionHandler> {
        if let Some(eh) = entry.unwind.as_ref().and_then(|u| u.handler) {
            return Some(eh);
        }

        let mut visited_unwinds: BTreeSet<u32> = BTreeSet::new();
        let mut visited_indirects: BTreeSet<u32> = BTreeSet::new();
        let mut visited_rfs: BTreeSet<RuntimeFunction> = BTreeSet::new();
        visited_rfs.insert(entry.function);

        let mut cur = if let Some(u) = entry.unwind.as_ref() {
            let chained = u.chained?;
            visited_unwinds.insert(entry.function.unwind);
            if chained.begin >= chained.end
                || chained.end > self.pe.optional.size_of_image
                || !visited_rfs.insert(chained)
            {
                return None;
            }
            chained
        } else if entry.function.unwind & RUNTIME_FUNCTION_INDIRECT != 0 {
            entry.function
        } else {
            return None;
        };

        let mut hops = 0usize;
        loop {
            if cur.unwind & RUNTIME_FUNCTION_INDIRECT != 0 {
                if hops >= MAX_CHAIN_DEPTH {
                    return None;
                }
                let target = cur.unwind.checked_sub(RUNTIME_FUNCTION_INDIRECT)?;
                if !visited_indirects.insert(target) {
                    return None;
                }
                let parent = self.indirect_rf(cur.unwind)?;
                if parent.begin >= parent.end
                    || parent.end > self.pe.optional.size_of_image
                    || !visited_rfs.insert(parent)
                {
                    return None;
                }
                hops += 1;
                cur = parent;
            } else {
                if !visited_unwinds.insert(cur.unwind) {
                    return None;
                }
                let info = self.unwind_info(cur.unwind)?;
                if let Some(eh) = info.handler {
                    return Some(eh);
                }
                let parent = info.chained?;
                if hops >= MAX_CHAIN_DEPTH
                    || parent.begin >= parent.end
                    || parent.end > self.pe.optional.size_of_image
                    || !visited_rfs.insert(parent)
                {
                    return None;
                }
                hops += 1;
                cur = parent;
            }
        }
    }
}

/// Sequential reader over the compressed FH4 encoding.
struct Stream<'f> {
    bytes: &'f [u8],
    pos: usize,
}

impl Stream<'_> {
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn u8(&mut self) -> Option<u8> {
        let b = *self.bytes.get(self.pos)?;
        self.pos = self.pos.checked_add(1)?;
        Some(b)
    }

    /// Raw little-endian `i32` (`ReadInt`).
    fn int(&mut self) -> Option<u32> {
        let end = self.pos.checked_add(4)?;
        let b = self.bytes.get(self.pos..end)?;
        self.pos = end;
        le_u32(b, 0)
    }

    /// Compressed unsigned integer (`ReadUnsigned`).
    fn unsigned(&mut self) -> Option<u32> {
        let (value, len) = decode_compressed(self.bytes.get(self.pos..)?)?;
        self.pos = self.pos.checked_add(len)?;
        Some(value)
    }
}

/// Decodes one FH4 compressed unsigned integer at the start of `bytes`: `(value, length)`.
pub fn decode_compressed(bytes: &[u8]) -> Option<(u32, usize)> {
    let first = *bytes.first()?;
    let len = if first & 0x1 == 0 {
        1
    } else if first & 0x3 == 0x1 {
        2
    } else if first & 0x7 == 0x3 {
        3
    } else if first & 0xF == 0x7 {
        4
    } else {
        5
    };
    let raw = bytes.get(..len)?;
    let value = if len == 5 {
        u32::from_le_bytes([raw[1], raw[2], raw[3], raw[4]])
    } else {
        let mut v = 0u32;
        for (i, b) in raw.iter().enumerate() {
            v |= u32::from(*b) << (8 * i);
        }
        v >> len
    };
    Some((value, len))
}

/// Encodes `value` in the FH4 compressed form (shortest encoding). Inverse of
/// [`decode_compressed`]; used to build test fixtures.
pub fn encode_compressed(value: u32) -> Vec<u8> {
    let len = match value {
        0..0x80 => 1,
        0x80..0x4000 => 2,
        0x4000..0x20_0000 => 3,
        0x20_0000..0x1000_0000 => 4,
        _ => 5,
    };
    if len == 5 {
        let mut out = vec![0x0F];
        out.extend_from_slice(&value.to_le_bytes());
        return out;
    }
    let tag = (1u32 << (len - 1)) - 1;
    let packed = (value << len) | tag;
    packed.to_le_bytes()[..len].to_vec()
}

/// Diagnostics collector for one function.
struct Diags<'a> {
    out: &'a mut Vec<EhDiagnostic>,
    rva: u32,
    count: usize,
}

impl Diags<'_> {
    fn push(&mut self, address: u32, kind: EhDiagnosticKind) {
        self.count += 1;
        self.out.push(EhDiagnostic {
            rva: self.rva,
            address,
            kind,
        });
    }
}

/// Validates a table count read from the data.
fn count(value: u32) -> Option<usize> {
    (value <= MAX_TABLE_ENTRIES)
        .then(|| usize::try_from(value).ok())
        .flatten()
}

impl EhMetadata {
    /// Parse the language-specific data of every `.pdata` record of `dir` with a handler. `file`
    /// must be the bytes `pe` was parsed from. Never fails; problems become diagnostics.
    pub fn from_pe(pe: &Pe64, file: &[u8], dir: &ExceptionDirectory) -> Self {
        let image = CodeImage::from_pe(pe, file);
        let reader = Reader { pe, file };
        let base = pe.optional.image_base;
        let slots = import_slots(pe);
        let setjmp_slots: BTreeSet<u32> = slots
            .iter()
            .filter(|(_, name)| SETJMP_IMPORTS.contains(&name.as_str()))
            .filter_map(|(va, _)| va.checked_sub(base).and_then(|d| u32::try_from(d).ok()))
            .collect();
        let mut out = Self {
            setjmp_slots: setjmp_slots.into_iter().collect(),
            ..Self::default()
        };

        // The module that implements the handlers itself (`vcruntime140*.dll`) exports them
        // under their import names and calls them directly instead of through the IAT.
        let mut exported: BTreeMap<u64, HandlerKind> = BTreeMap::new();
        for e in pe.exports.iter().flat_map(|t| t.symbols.iter()) {
            if e.forwarder.is_some() {
                continue;
            }
            let Some(name) = e.name.as_deref() else {
                continue;
            };
            let Some(kind) = HandlerKind::from_import(name) else {
                continue;
            };
            let Some(va) = rva_to_va(base, e.rva) else {
                continue;
            };
            exported
                .entry(va)
                .and_modify(|existing| *existing = (*existing).min(kind))
                .or_insert(kind);
        }

        // Process `.pdata` records in canonical ascending RVA order (deduplicating identical
        // `(begin, end, unwind)` keys) so outputs are invariant to input order or duplicates.
        let mut ordered_entries: BTreeMap<(u32, u32, u32), &PdataEntry> = BTreeMap::new();
        for entry in &dir.entries {
            ordered_entries
                .entry((
                    entry.function.begin,
                    entry.function.end,
                    entry.function.unwind,
                ))
                .or_insert(entry);
        }
        let resolved_entries: Vec<(&PdataEntry, Option<ExceptionHandler>)> = ordered_entries
            .into_values()
            .map(|entry| (entry, reader.resolve_entry_handler(entry)))
            .collect();

        // Identify every handler routine (direct and chained/indirect) once in canonical ascending
        // RVA order.
        let mut handler_rvas: BTreeSet<u32> = dir.handler_rvas().into_iter().collect();
        handler_rvas.extend(
            resolved_entries
                .iter()
                .filter_map(|(_, eh)| eh.map(|h| h.rva)),
        );
        let scans: BTreeMap<u32, HandlerScan> = handler_rvas
            .into_iter()
            .map(|rva| {
                let mut scan = rva_to_va(base, rva)
                    .map(|va| scan_handler(&image, &slots, va))
                    .unwrap_or_default();
                // A direct call to an exported handler counts like a call through its import.
                let mut kinds: BTreeSet<HandlerKind> = scan.kinds.into_iter().collect();
                kinds.extend(scan.direct_calls.iter().filter_map(|c| exported.get(c)));
                scan.kinds = kinds.into_iter().collect();
                scan.direct_calls.retain(|c| !exported.contains_key(c));
                (rva, scan)
            })
            .collect();
        // Cookie-check helpers called first by recognised wrappers identify `__GSHandlerCheck`.
        let gs_helpers: BTreeSet<u64> = scans
            .values()
            .filter(|scan| scan.thunk.is_none() && scan.kinds.len() == 1)
            .filter_map(|scan| scan.direct_calls.first().copied())
            .collect();
        for (&rva, scan) in &scans {
            let handler = if let Some(&kind) = rva_to_va(base, rva).and_then(|va| exported.get(&va))
            {
                Ok(Handler {
                    kind,
                    wrapper: false,
                })
            } else if let Some(kind) = scan.thunk {
                Ok(Handler {
                    kind,
                    wrapper: false,
                })
            } else {
                match scan.kinds.len() {
                    1 => Ok(Handler {
                        kind: scan.kinds[0],
                        wrapper: true,
                    }),
                    0 if !scan.direct_calls.is_empty()
                        && scan.direct_calls.iter().all(|c| gs_helpers.contains(c)) =>
                    {
                        Ok(Handler {
                            kind: HandlerKind::GsCheck,
                            wrapper: true,
                        })
                    }
                    0 => Err(EhDiagnosticKind::UnknownHandler),
                    _ => Err(EhDiagnosticKind::AmbiguousHandler),
                }
            };
            match handler {
                Ok(handler) => {
                    out.handlers.insert(rva, handler);
                }
                Err(kind) => out.diagnostics.push(EhDiagnostic {
                    rva,
                    address: rva,
                    kind,
                }),
            }
        }

        let mut unrecognised = BTreeSet::new();
        let mut parsed_keys: BTreeSet<(u32, u32, u32)> = BTreeSet::new();
        for (entry, resolved_eh) in resolved_entries {
            let Some(eh) = resolved_eh else {
                continue;
            };
            if !parsed_keys.insert((entry.function.begin, eh.rva, eh.data_rva)) {
                continue;
            }
            let Some(&handler) = out.handlers.get(&eh.rva) else {
                unrecognised.insert(entry.function.begin);
                continue;
            };
            let mut diags = Diags {
                out: &mut out.diagnostics,
                rva: entry.function.begin,
                count: 0,
            };
            let data = match handler.kind {
                HandlerKind::GsCheck => LanguageData::None,
                HandlerKind::CSpecific => {
                    parse_scope_table(&reader, &image, eh.data_rva, &mut diags)
                }
                HandlerKind::CxxFrameHandler3 => {
                    parse_fh3(&reader, &image, eh.data_rva, &mut diags)
                }
                HandlerKind::CxxFrameHandler4 => parse_fh4(
                    &reader,
                    &image,
                    eh.data_rva,
                    entry.function.begin,
                    entry.primary,
                    &mut diags,
                ),
            };
            out.functions.push(EhFunction {
                begin: entry.function.begin,
                end: entry.function.end,
                handler_rva: eh.rva,
                handler,
                data_rva: eh.data_rva,
                data,
            });
        }
        out.unrecognised = unrecognised.into_iter().collect();
        out
    }

    /// Distinct funclets of all functions, `(rva, kind)`.
    pub fn funclets(&self) -> BTreeSet<(u32, FuncletKind)> {
        self.functions
            .iter()
            .flat_map(EhFunction::funclets)
            .collect()
    }

    /// Distinct resume targets of all functions, `(rva, kind)`.
    pub fn resume_targets(&self) -> BTreeSet<(u32, ResumeKind)> {
        self.functions
            .iter()
            .flat_map(EhFunction::resume_targets)
            .collect()
    }

    /// Aggregate counts.
    pub fn stats(&self) -> EhStats {
        let mut stats = EhStats {
            unknown: self.unrecognised.len(),
            diagnostics: self.diagnostics.len(),
            ..EhStats::default()
        };
        let failed: BTreeSet<u32> = self.diagnostics.iter().map(|d| d.rva).collect();
        for function in &self.functions {
            match function.handler.kind {
                HandlerKind::CSpecific => stats.c_specific += 1,
                HandlerKind::CxxFrameHandler3 => stats.cxx3 += 1,
                HandlerKind::CxxFrameHandler4 => stats.cxx4 += 1,
                HandlerKind::GsCheck => stats.gs_check += 1,
            }
            if !failed.contains(&function.begin) {
                stats.parsed += 1;
            }
        }
        stats.funclets = self.funclets().len();
        stats.resume_targets = self
            .resume_targets()
            .into_iter()
            .map(|(rva, _)| rva)
            .collect::<BTreeSet<_>>()
            .len();
        stats
    }

    /// Feed the metadata into discovery: funclets in executable code become entry points (kept
    /// sorted and deduplicated), and [`CodeImage::eh`] receives the resume targets with their
    /// owner (the `.pdata` function containing the target; targets outside every `.pdata`
    /// function are owned by the function whose data names them) plus the `setjmp` slots.
    /// Install `.pdata` first ([`ExceptionDirectory::seed`]).
    pub fn seed(&self, image: &mut CodeImage<'_>) {
        let base = image.base();
        let mut entries: BTreeSet<u64> = std::mem::take(&mut image.entry_points)
            .into_iter()
            .collect();
        entries.extend(
            self.handlers
                .keys()
                .copied()
                .chain(self.funclets().into_iter().map(|(rva, _)| rva))
                .filter_map(|rva| rva_to_va(base, rva))
                .filter(|&va| image.region_of(va) == Region::Code),
        );
        image.entry_points = entries.into_iter().collect();

        let mut targets = BTreeSet::new();
        for function in &self.functions {
            for (rva, _) in function.resume_targets() {
                let Some(target) = rva_to_va(base, rva) else {
                    continue;
                };
                if image.region_of(target) != Region::Code {
                    continue;
                }
                let Some(owner) = image
                    .function_table
                    .function_at(target)
                    .or_else(|| {
                        let begin_va = rva_to_va(base, function.begin)?;
                        Some(image.function_table.function_at(begin_va).unwrap_or(begin_va))
                    })
                else {
                    continue;
                };
                targets.insert((owner, target));
            }
        }
        let setjmp_slots: BTreeSet<u64> = self
            .setjmp_slots
            .iter()
            .filter_map(|&rva| rva_to_va(base, rva))
            .collect();
        image.eh = EhSeeds {
            resume_targets: targets.into_iter().collect(),
            setjmp_slots: setjmp_slots.into_iter().collect(),
            stats: self.stats(),
        };
    }
}

/// IAT slot VA -> imported symbol name, for normal and delay-load imports.
fn import_slots(pe: &Pe64) -> BTreeMap<u64, String> {
    let base = pe.optional.image_base;
    let mut slots = BTreeMap::new();
    let modules = pe
        .imports
        .iter()
        .map(|m| (m.iat_rva, &m.symbols))
        .chain(pe.delay_imports.iter().map(|m| (m.iat_rva, &m.symbols)));
    for (iat, symbols) in modules {
        if iat == 0 {
            continue;
        }
        for (i, symbol) in symbols.iter().enumerate() {
            let Some(slot_rva) = u32::try_from(i)
                .ok()
                .and_then(|idx| idx.checked_mul(8))
                .and_then(|off| iat.checked_add(off))
            else {
                break;
            };
            let Some(slot) = rva_to_va(base, slot_rva) else {
                break;
            };
            if let ImportedSymbol::ByName { name, .. } = symbol {
                slots.insert(slot, name.clone());
            }
        }
    }
    slots
}

/// What a bounded walk over a handler routine found.
#[derive(Default)]
struct HandlerScan {
    /// Set when the routine itself is `jmp [rip+slot]` into a language-handler import.
    thunk: Option<HandlerKind>,
    /// Language handlers reached through IAT slots or import thunks (sorted, distinct).
    kinds: Vec<HandlerKind>,
    /// Direct call targets that are not import thunks, in address order.
    direct_calls: Vec<u64>,
}

/// Language-handler kind of the IAT slot read by a RIP-relative `call`/`jmp`.
fn slot_kind(ins: &Instruction, slots: &BTreeMap<u64, String>) -> Option<HandlerKind> {
    (ins.op_count() > 0 && ins.op_kind(0) == OpKind::Memory && ins.is_ip_rel_memory_operand())
        .then(|| slots.get(&ins.ip_rel_memory_address()))
        .flatten()
        .and_then(|name| HandlerKind::from_import(name))
}

/// First instruction at `va`, if it decodes.
fn first_instruction(image: &CodeImage<'_>, va: u64) -> Option<Instruction> {
    let bytes = image.code_bytes(va).ok()?;
    let mut decoder = Decoder::with_ip(64, bytes, va, DecoderOptions::NONE);
    let ins = decoder.decode();
    (!ins.is_invalid()).then_some(ins)
}

/// Whether `va` is an import thunk `jmp [rip+slot]` (possibly reached through up to
/// [`MAX_THUNK_HOPS`] direct `jmp rel32`/`rel8` hops); returns the handler kind of its slot.
fn thunk_kind(
    image: &CodeImage<'_>,
    slots: &BTreeMap<u64, String>,
    mut va: u64,
) -> Option<Option<HandlerKind>> {
    let mut seen = BTreeSet::new();
    for _ in 0..=MAX_THUNK_HOPS {
        if !seen.insert(va) {
            return None;
        }
        let ins = first_instruction(image, va)?;
        if ins.flow_control() == FlowControl::IndirectBranch && ins.is_ip_rel_memory_operand() {
            return Some(slot_kind(&ins, slots));
        }
        if ins.flow_control() == FlowControl::UnconditionalBranch
            && ins.op_count() > 0
            && ins.op_kind(0) == OpKind::NearBranch64
        {
            va = ins.near_branch64();
            continue;
        }
        return None;
    }
    None
}

/// Walks the handler routine at `va` (both sides of branches, at most
/// [`HANDLER_SCAN_LIMIT`] instructions) and records what it calls.
fn scan_handler(image: &CodeImage<'_>, slots: &BTreeMap<u64, String>, va: u64) -> HandlerScan {
    let mut scan = HandlerScan::default();
    if let Some(Some(kind)) = thunk_kind(image, slots, va) {
        scan.thunk = Some(kind);
        return scan;
    }
    let mut kinds = BTreeSet::new();
    let mut calls = BTreeSet::new();
    let mut seen = BTreeSet::new();
    let mut work = vec![va];
    let mut budget = HANDLER_SCAN_LIMIT;
    while let Some(start) = work.pop() {
        if !seen.insert(start) {
            continue;
        }
        let Ok(bytes) = image.code_bytes(start) else {
            continue;
        };
        let mut decoder = Decoder::with_ip(64, bytes, start, DecoderOptions::NONE);
        while budget > 0 && decoder.can_decode() {
            budget -= 1;
            let ins = decoder.decode();
            if ins.is_invalid() {
                break;
            }
            let direct = (ins.op_count() > 0 && ins.op_kind(0) == OpKind::NearBranch64)
                .then(|| ins.near_branch64());
            match ins.flow_control() {
                FlowControl::Call | FlowControl::UnconditionalBranch => {
                    if let Some(target) = direct {
                        match thunk_kind(image, slots, target) {
                            Some(Some(kind)) => {
                                kinds.insert(kind);
                            }
                            Some(None) => {}
                            None if ins.flow_control() == FlowControl::Call => {
                                calls.insert(target);
                            }
                            None => {
                                if !seen.contains(&target) && work.len() < HANDLER_SCAN_LIMIT {
                                    work.push(target);
                                }
                            }
                        }
                    }
                    if ins.flow_control() == FlowControl::UnconditionalBranch {
                        break;
                    }
                }
                FlowControl::IndirectCall => {
                    kinds.extend(slot_kind(&ins, slots));
                }
                FlowControl::IndirectBranch => {
                    kinds.extend(slot_kind(&ins, slots));
                    break;
                }
                FlowControl::ConditionalBranch => {
                    if let Some(target) = direct
                        && !seen.contains(&target)
                        && work.len() < HANDLER_SCAN_LIMIT
                    {
                        work.push(target);
                    }
                }
                FlowControl::Return | FlowControl::Exception | FlowControl::Interrupt => break,
                _ => {}
            }
        }
    }
    scan.kinds = kinds.into_iter().collect();
    scan.direct_calls = calls.into_iter().collect();
    scan
}

/// Records `rva` as code-or-diagnostic.
fn check_code(image: &CodeImage<'_>, rva: u32, diags: &mut Diags<'_>) {
    if !is_code(image, rva) {
        diags.push(rva, EhDiagnosticKind::NotCode);
    }
}

fn is_code(image: &CodeImage<'_>, rva: u32) -> bool {
    rva_to_va(image.base(), rva).is_some_and(|va| image.region_of(va) == Region::Code)
}

/// `__C_specific_handler` scope table at `rva`.
fn parse_scope_table(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    rva: u32,
    diags: &mut Diags<'_>,
) -> LanguageData {
    let Some(raw_count) = reader.u32(rva) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(n) = count(raw_count) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return LanguageData::None;
    };
    if n == 0 {
        return LanguageData::ScopeTable(Vec::new());
    }
    let Some(byte_len) = n.checked_mul(SCOPE_RECORD_LEN) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return LanguageData::None;
    };
    let Some(records_rva) = rva.checked_add(SCOPE_TABLE_HEADER_LEN) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(bytes) = reader.read(records_rva, byte_len) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let size_of_image = reader.pe.optional.size_of_image;
    let mut records = Vec::with_capacity(n.min(INITIAL_TABLE_CAPACITY));
    for chunk in bytes.chunks_exact(SCOPE_RECORD_LEN) {
        let (Some(begin), Some(end), Some(raw_handler), Some(jump_target)) = (
            le_u32(chunk, 0),
            le_u32(chunk, 4),
            le_u32(chunk, 8),
            le_u32(chunk, 12),
        ) else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return LanguageData::None;
        };
        if begin > end || end > size_of_image {
            diags.push(begin, EhDiagnosticKind::NotCode);
        }
        let handler = if jump_target == 0 {
            check_code(image, raw_handler, diags);
            ScopeHandler::Finally(raw_handler)
        } else {
            check_code(image, jump_target, diags);
            if is_code(image, raw_handler) {
                ScopeHandler::Filter(raw_handler)
            } else {
                if raw_handler != EXCEPTION_EXECUTE_HANDLER {
                    diags.push(raw_handler, EhDiagnosticKind::NotCode);
                }
                ScopeHandler::Constant(raw_handler)
            }
        };
        records.push(ScopeRecord {
            begin,
            end,
            handler,
            jump_target,
        });
    }
    LanguageData::ScopeTable(records)
}

/// `n` records of `size` bytes at `rva`, as little-endian `u32` fields.
fn read_records(
    reader: &Reader<'_, '_>,
    rva: u32,
    raw_count: u32,
    size: usize,
    diags: &mut Diags<'_>,
) -> Option<Vec<Vec<u32>>> {
    if size == 0 || !size.is_multiple_of(4) {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return None;
    }
    let Some(n) = count(raw_count) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return None;
    };
    if n == 0 {
        return Some(Vec::new());
    }
    let Some(byte_len) = n.checked_mul(size) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return None;
    };
    let Some(bytes) = reader.read(rva, byte_len) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return None;
    };
    let mut out = Vec::with_capacity(n.min(INITIAL_TABLE_CAPACITY));
    for chunk in bytes.chunks_exact(size) {
        let mut fields = Vec::with_capacity(size / 4);
        for word in chunk.chunks_exact(4) {
            fields.push(le_u32(word, 0)?);
        }
        out.push(fields);
    }
    Some(out)
}

/// `__CxxFrameHandler3`: the data is the RVA of a `FuncInfo`.
fn parse_fh3(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    data_rva: u32,
    diags: &mut Diags<'_>,
) -> LanguageData {
    let Some(rva) = reader.u32(data_rva) else {
        diags.push(data_rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(raw_magic) = reader.u32(rva) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let magic = raw_magic & FH3_MAGIC_MASK;
    if !(FH3_MAGIC_MIN..=FH3_MAGIC_MAX).contains(&magic) {
        diags.push(rva, EhDiagnosticKind::UnknownMagic);
        return LanguageData::None;
    }
    let Some(fields) = magic
        .checked_sub(FH3_MAGIC_MIN)
        .and_then(|d| usize::try_from(d).ok())
        .and_then(|d| 8usize.checked_add(d))
    else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(header_len) = fields.checked_mul(4) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(header) = reader.read(rva, header_len) else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let f = |i: usize| i.checked_mul(4).and_then(|off| le_u32(header, off));
    let (
        Some(max_state),
        Some(unwind_rva),
        Some(n_try),
        Some(try_rva),
        Some(n_ip),
        Some(ip_rva),
        Some(unwind_help),
    ) = (f(1), f(2), f(3), f(4), f(5), f(6), f(7))
    else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let mut info = FuncInfo3 {
        rva,
        magic,
        bbt_flags: (raw_magic >> 29) as u8,
        unwind_map: Vec::new(),
        try_blocks: Vec::new(),
        ip_to_state: Vec::new(),
        unwind_help: unwind_help as i32,
        es_type_list: if fields > 8 { f(8).unwrap_or(0) } else { 0 },
        eh_flags: if fields > 9 { f(9).unwrap_or(0) } else { 0 },
    };

    if let Some(records) = read_records(reader, unwind_rva, max_state, 8, diags) {
        info.unwind_map = records
            .iter()
            .map(|r| UnwindEntry3 {
                to_state: r[0] as i32,
                action: r[1],
            })
            .collect();
        for entry in info.unwind_map.iter().filter(|e| e.action != 0) {
            check_code(image, entry.action, diags);
        }
    }
    if let Some(records) = read_records(reader, try_rva, n_try, 20, diags) {
        for r in records {
            let mut block = TryBlock3 {
                try_low: r[0] as i32,
                try_high: r[1] as i32,
                catch_high: r[2] as i32,
                handlers: Vec::new(),
            };
            if let Some(handlers) = read_records(reader, r[4], r[3], 20, diags) {
                for h in handlers {
                    check_code(image, h[3], diags);
                    block.handlers.push(CatchHandler3 {
                        adjectives: h[0],
                        type_descriptor: h[1],
                        catch_object: h[2] as i32,
                        handler: h[3],
                        frame: h[4],
                    });
                }
            }
            info.try_blocks.push(block);
        }
    }
    if let Some(records) = read_records(reader, ip_rva, n_ip, 8, diags) {
        info.ip_to_state = records.iter().map(|r| (r[0], r[1] as i32)).collect();
    }
    LanguageData::Cxx3(info)
}

/// Compressed stream at `rva`, or a `Truncated` diagnostic.
fn stream<'f>(reader: &Reader<'_, 'f>, rva: u32, diags: &mut Diags<'_>) -> Option<Stream<'f>> {
    match reader.tail(rva) {
        Some(bytes) if !bytes.is_empty() => Some(Stream { bytes, pos: 0 }),
        _ => {
            diags.push(rva, EhDiagnosticKind::Truncated);
            None
        }
    }
}

/// `__CxxFrameHandler4`: the data is the RVA of a compressed `FuncInfo4`. `segment` is the RVA of
/// the `.pdata` chunk and `primary` is its primary function RVA (base of function-relative
/// continuation offsets).
fn parse_fh4(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    data_rva: u32,
    segment: u32,
    primary: u32,
    diags: &mut Diags<'_>,
) -> LanguageData {
    let Some(rva) = reader.u32(data_rva) else {
        diags.push(data_rva, EhDiagnosticKind::Truncated);
        return LanguageData::None;
    };
    let Some(mut s) = stream(reader, rva, diags) else {
        return LanguageData::None;
    };
    let truncated = |diags: &mut Diags<'_>| {
        diags.push(rva, EhDiagnosticKind::Truncated);
        LanguageData::None
    };
    let Some(header) = s.u8() else {
        return truncated(diags);
    };
    let mut info = FuncInfo4 {
        rva,
        header,
        bbt_flags: 0,
        unwind_map_rva: 0,
        try_block_map_rva: 0,
        ip_map_rva: 0,
        frame: None,
        unwind_map: Vec::new(),
        try_blocks: Vec::new(),
        ip_to_state: Vec::new(),
    };
    let fields = (|| {
        if header & FH4_BBT != 0 {
            info.bbt_flags = s.unsigned()?;
        }
        if header & FH4_UNWIND_MAP != 0 {
            info.unwind_map_rva = s.int()?;
        }
        if header & FH4_TRY_BLOCK_MAP != 0 {
            info.try_block_map_rva = s.int()?;
        }
        let ip_map = s.int()?;
        if header & FH4_IS_CATCH != 0 {
            info.frame = Some(s.unsigned()?);
        }
        Some(ip_map)
    })();
    let Some(ip_map) = fields else {
        return truncated(diags);
    };
    let (ip_map_rva, ip_base) = if info.is_separated() {
        separated_ip_map(reader, ip_map, segment, primary, diags).unwrap_or((0, primary))
    } else {
        (ip_map, primary)
    };
    info.ip_map_rva = ip_map_rva;

    if info.unwind_map_rva != 0 {
        info.unwind_map = parse_unwind_map4(reader, image, info.unwind_map_rva, diags);
    }
    if info.try_block_map_rva != 0 {
        info.try_blocks = parse_try_map4(reader, image, info.try_block_map_rva, primary, diags);
    }
    if info.ip_map_rva != 0 {
        info.ip_to_state = parse_ip_map4(reader, info.ip_map_rva, ip_base, diags);
    }
    LanguageData::Cxx4(info)
}

/// Separated code: `{count, [segment RVA (raw), map RVA (raw)]...}`; the map of `segment` (or,
/// for a chained chunk whose segment has no dedicated entry, its `primary` function). Returns
/// `(map_rva, matched_segment_rva)`.
fn separated_ip_map(
    reader: &Reader<'_, '_>,
    rva: u32,
    segment: u32,
    primary: u32,
    diags: &mut Diags<'_>,
) -> Option<(u32, u32)> {
    let mut s = stream(reader, rva, diags)?;
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return None;
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return None;
    };
    let mut primary_match: Option<(u32, u32)> = None;
    for _ in 0..n {
        let (Some(seg), Some(map)) = (s.int(), s.int()) else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return None;
        };
        if seg == segment {
            return Some((map, seg));
        }
        if primary != segment && seg == primary && primary_match.is_none() {
            primary_match = Some((map, seg));
        }
    }
    if let Some(found) = primary_match {
        return Some(found);
    }
    diags.push(rva, EhDiagnosticKind::MissingSegment);
    None
}

fn parse_unwind_map4(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    rva: u32,
    diags: &mut Diags<'_>,
) -> Vec<UnwindEntry4> {
    let Some(mut s) = stream(reader, rva, diags) else {
        return Vec::new();
    };
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return Vec::new();
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return Vec::new();
    };
    // (entry start offset, next offset, entry)
    let mut raw = Vec::with_capacity(bounded_capacity(n, s.remaining(), 1));
    for _ in 0..n {
        let start = s.pos;
        let parsed = (|| {
            let next_and_type = s.unsigned()?;
            let kind = match next_and_type & 0x3 {
                0 => UnwindKind4::None,
                1 => UnwindKind4::DtorWithObj,
                2 => UnwindKind4::DtorWithPtrToObj,
                _ => UnwindKind4::Funclet,
            };
            let (mut action, mut object) = (0, 0);
            match kind {
                UnwindKind4::DtorWithObj | UnwindKind4::DtorWithPtrToObj => {
                    action = s.int()?;
                    object = s.unsigned()?;
                }
                UnwindKind4::Funclet => action = s.int()?,
                UnwindKind4::None => {}
            }
            let next_offset = usize::try_from(next_and_type >> 2).ok()?;
            Some((
                next_offset,
                UnwindEntry4 {
                    kind,
                    next_state: -1,
                    action,
                    object,
                },
            ))
        })();
        let Some((next_offset, entry)) = parsed else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return Vec::new();
        };
        if entry.kind != UnwindKind4::None {
            check_code(image, entry.action, diags);
        }
        raw.push((start, next_offset, entry));
    }
    // `nextOffset` counts bytes back from the start of the entry to the start of the entry of
    // the next state; landing on the start of the map (the entry count) ends the chain
    // (state -1).
    let starts: Vec<usize> = raw.iter().map(|(start, _, _)| *start).collect();
    raw.into_iter()
        .map(|(start, next_offset, mut entry)| {
            let link = match start.checked_sub(next_offset) {
                Some(0) => Some(-1),
                Some(at) if next_offset != 0 => starts
                    .binary_search(&at)
                    .ok()
                    .and_then(|s| i32::try_from(s).ok()),
                _ => None,
            };
            match link {
                Some(state) => entry.next_state = state,
                None => diags.push(rva, EhDiagnosticKind::BadUnwindLink),
            }
            entry
        })
        .collect()
}

fn parse_try_map4(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    rva: u32,
    function: u32,
    diags: &mut Diags<'_>,
) -> Vec<TryBlock4> {
    let Some(mut s) = stream(reader, rva, diags) else {
        return Vec::new();
    };
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return Vec::new();
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return Vec::new();
    };
    let mut blocks = Vec::with_capacity(bounded_capacity(n, s.remaining(), 7));
    for _ in 0..n {
        let parsed = (|| {
            Some(TryBlock4 {
                try_low: s.unsigned()?,
                try_high: s.unsigned()?,
                catch_high: s.unsigned()?,
                handler_array: s.int()?,
                handlers: Vec::new(),
            })
        })();
        let Some(mut block) = parsed else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return blocks;
        };
        block.handlers = parse_handlers4(reader, image, block.handler_array, function, diags);
        blocks.push(block);
    }
    blocks
}

fn parse_handlers4(
    reader: &Reader<'_, '_>,
    image: &CodeImage<'_>,
    rva: u32,
    function: u32,
    diags: &mut Diags<'_>,
) -> Vec<CatchHandler4> {
    let Some(mut s) = stream(reader, rva, diags) else {
        return Vec::new();
    };
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return Vec::new();
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return Vec::new();
    };
    let mut handlers = Vec::with_capacity(bounded_capacity(n, s.remaining(), 5));
    for _ in 0..n {
        let mut reserved = false;
        let parsed = (|| {
            let header = s.u8()?;
            let mut h = CatchHandler4 {
                header,
                adjectives: 0,
                type_descriptor: 0,
                catch_object: 0,
                handler: 0,
                continuations: Vec::new(),
            };
            if header & FH4_HANDLER_ADJECTIVES != 0 {
                h.adjectives = s.unsigned()?;
            }
            if header & FH4_HANDLER_DISP_TYPE != 0 {
                h.type_descriptor = s.int()?;
            }
            if header & FH4_HANDLER_DISP_CATCH_OBJ != 0 {
                h.catch_object = s.unsigned()?;
            }
            h.handler = s.int()?;
            let cont_count = (header >> FH4_HANDLER_CONT_SHIFT) & 0x3;
            reserved = cont_count == 3;
            for _ in 0..cont_count.min(2) {
                let value = if header & FH4_HANDLER_CONT_IS_RVA != 0 {
                    s.int()?
                } else {
                    function.checked_add(s.unsigned()?)?
                };
                h.continuations.push(value);
            }
            Some(h)
        })();
        let Some(handler) = parsed else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return handlers;
        };
        if reserved {
            diags.push(rva, EhDiagnosticKind::Reserved);
        }
        check_code(image, handler.handler, diags);
        for &cont in &handler.continuations {
            check_code(image, cont, diags);
        }
        handlers.push(handler);
    }
    handlers
}

fn parse_ip_map4(
    reader: &Reader<'_, '_>,
    rva: u32,
    function: u32,
    diags: &mut Diags<'_>,
) -> Vec<(u32, i32)> {
    let Some(mut s) = stream(reader, rva, diags) else {
        return Vec::new();
    };
    let Some(n) = s.unsigned() else {
        diags.push(rva, EhDiagnosticKind::Truncated);
        return Vec::new();
    };
    let Some(n) = count(n) else {
        diags.push(rva, EhDiagnosticKind::TooLarge);
        return Vec::new();
    };
    let mut out = Vec::with_capacity(bounded_capacity(n, s.remaining(), 2));
    let mut ip = function;
    for _ in 0..n {
        let (Some(delta), Some(raw_state)) = (s.unsigned(), s.unsigned()) else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return out;
        };
        let Some(next_ip) = ip.checked_add(delta) else {
            diags.push(rva, EhDiagnosticKind::Truncated);
            return out;
        };
        let state = if raw_state == 0 {
            -1
        } else if let Ok(s) = i32::try_from(raw_state - 1) {
            s
        } else {
            diags.push(rva, EhDiagnosticKind::TooLarge);
            return out;
        };
        ip = next_ip;
        out.push((ip, state));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pdata::tests::{BASE, pe_with_pdata, put, unwind_bytes};
    use crate::pdata::{UNW_FLAG_EHANDLER, UNW_FLAG_UHANDLER};

    // ----- fixture layout (RVAs) -------------------------------------------------------------
    //
    // .text 0x1000..0x1200, .rdata 0x2000..0x2800 (see `pdata::tests::pe_with_pdata`).
    const F_SEH: u32 = 0x1080;
    const F_FH3: u32 = 0x10a0;
    const F_FH4: u32 = 0x10c0;
    const F_CATCH: u32 = 0x1100;
    const F_GS_FH3: u32 = 0x1120;
    const F_GS: u32 = 0x1130;
    const F_UNKNOWN: u32 = 0x1140;
    const F_CLEANUP: u32 = 0x1150;

    const GS_HELPER: u32 = 0x1160;
    const THUNK_SEH: u32 = 0x1168;
    const THUNK_FH3: u32 = 0x1170;
    const THUNK_FH4: u32 = 0x1178;
    const THUNK_SETJMP: u32 = 0x1180;
    const GS_WRAPPER_FH3: u32 = 0x1188;
    const GS_CHECK: u32 = 0x1198;
    const UNKNOWN_HANDLER: u32 = 0x11a8;

    const IMPORTS: u32 = 0x2460;
    const UNWIND_SEH: u32 = 0x25c0;
    const UNWIND_FH3: u32 = 0x2600;
    const UNWIND_FH4: u32 = 0x2610;
    const UNWIND_CATCH: u32 = 0x2620;
    const UNWIND_PLAIN: u32 = 0x2630;
    const UNWIND_GS_FH3: u32 = 0x2640;
    const UNWIND_GS: u32 = 0x2658;
    const UNWIND_UNKNOWN: u32 = 0x2668;
    const FH3_INFO: u32 = 0x2680;
    const FH3_UNWIND: u32 = 0x26a8;
    const FH3_TRY: u32 = 0x26b8;
    const FH3_HANDLERS: u32 = 0x26cc;
    const FH3_IP: u32 = 0x26e0;
    const FH4_INFO: u32 = 0x26f0;
    const FH4_UNWIND: u32 = 0x2700;
    const FH4_TRY: u32 = 0x2710;
    const FH4_HANDLERS: u32 = 0x2718;
    const FH4_IP: u32 = 0x2738;
    const FH4_CATCH_INFO: u32 = 0x2740;
    const FH4_SEGMENTS: u32 = 0x2750;
    const FH4_CATCH_IP: u32 = 0x2768;

    // Funclets and targets referenced by the metadata.
    const SEH_FILTER: u32 = 0x1154;
    const SEH_FINALLY: u32 = 0x1158;
    const FH3_CLEANUP: u32 = 0x115c;
    const FH4_CLEANUP: u32 = F_CLEANUP;
    const FH4_DTOR: u32 = 0x115e;
    const FH4_CATCH2: u32 = 0x1110;

    fn c(value: u32) -> Vec<u8> {
        encode_compressed(value)
    }

    fn le(value: u32) -> [u8; 4] {
        value.to_le_bytes()
    }

    /// `jmp [rip+slot]` at `at`.
    fn jmp_slot(file: &mut [u8], at: u32, slot: u32) {
        let disp = slot.wrapping_sub(at + 6);
        put(file, at, &[0xff, 0x25]);
        put(file, at + 2, &le(disp));
    }

    /// `call rel32` at `at`.
    fn call(file: &mut [u8], at: u32, target: u32) {
        put(file, at, &[0xe8]);
        put(file, at + 1, &le(target.wrapping_sub(at + 5)));
    }

    /// Replaces the import directory with `modules` (built at [`IMPORTS`]); returns the IAT
    /// slot RVA of every symbol, per module.
    fn set_imports(file: &mut [u8], modules: &[(&str, &[&str])]) -> Vec<Vec<u32>> {
        let mut at = IMPORTS + 20 * (modules.len() as u32 + 1);
        let mut slots = Vec::new();
        let mut thunks = Vec::new();
        for (_, names) in modules {
            thunks.push(at);
            slots.push((0..names.len() as u32).map(|i| at + 8 * i).collect());
            at += 8 * (names.len() as u32 + 1);
        }
        for (i, (dll, names)) in modules.iter().enumerate() {
            let dll_rva = at;
            put(file, at, dll.as_bytes());
            put(file, at + dll.len() as u32, &[0]);
            at += dll.len() as u32 + 1;
            for (j, name) in names.iter().enumerate() {
                at += at & 1;
                put(file, thunks[i] + 8 * j as u32, &u64::from(at).to_le_bytes());
                put(file, at, &[0, 0]);
                put(file, at + 2, name.as_bytes());
                put(file, at + 2 + name.len() as u32, &[0]);
                at += 3 + name.len() as u32;
            }
            put(file, thunks[i] + 8 * names.len() as u32, &[0; 8]);
            let desc = IMPORTS + 20 * i as u32;
            put(file, desc, &le(thunks[i]));
            put(file, desc + 12, &le(dll_rva));
            put(file, desc + 16, &le(thunks[i]));
        }
        assert!(
            at <= 0x2560,
            "import fixture overflows into .pdata ({at:#x})"
        );
        put(file, IMPORTS + 20 * modules.len() as u32, &[0; 20]);
        // Import data directory (index 1).
        file[0x108 + 8..0x108 + 12].copy_from_slice(&le(IMPORTS));
        file[0x108 + 12..0x108 + 16].copy_from_slice(&le(20 * (modules.len() as u32 + 1)));
        slots
    }

    /// Handler RVA + data, as the `UNWIND_INFO` trailer.
    fn trailer(handler: u32, data: &[u8]) -> Vec<u8> {
        [&le(handler)[..], data].concat()
    }

    struct Fixture {
        file: Vec<u8>,
        setjmp_slot: u32,
    }

    /// Synthetic PE with one function per handler kind (see the layout constants).
    fn fixture() -> Fixture {
        let mut file = pe_with_pdata(&[
            (F_SEH, F_SEH + 0x20, UNWIND_SEH),
            (F_FH3, F_FH3 + 0x20, UNWIND_FH3),
            (F_FH4, F_FH4 + 0x40, UNWIND_FH4),
            (F_CATCH, F_CATCH + 0x20, UNWIND_CATCH),
            (F_GS_FH3, F_GS_FH3 + 0x10, UNWIND_GS_FH3),
            (F_GS, F_GS + 0x10, UNWIND_GS),
            (F_UNKNOWN, F_UNKNOWN + 0x10, UNWIND_UNKNOWN),
            (F_CLEANUP, F_CLEANUP + 0x10, UNWIND_PLAIN),
        ]);
        let slots = set_imports(
            &mut file,
            &[
                (
                    "VCRUNTIME140.dll",
                    &[
                        "__C_specific_handler",
                        "__CxxFrameHandler3",
                        "__intrinsic_setjmp",
                    ],
                ),
                ("VCRUNTIME140_1.dll", &["__CxxFrameHandler4"]),
            ],
        );
        let (seh, fh3, setjmp, fh4) = (slots[0][0], slots[0][1], slots[0][2], slots[1][0]);

        // Handler routines.
        put(&mut file, GS_HELPER, &[0xc3]);
        jmp_slot(&mut file, THUNK_SEH, seh);
        jmp_slot(&mut file, THUNK_FH3, fh3);
        jmp_slot(&mut file, THUNK_FH4, fh4);
        jmp_slot(&mut file, THUNK_SETJMP, setjmp);
        call(&mut file, GS_WRAPPER_FH3, GS_HELPER);
        call(&mut file, GS_WRAPPER_FH3 + 5, THUNK_FH3);
        put(&mut file, GS_WRAPPER_FH3 + 10, &[0xc3]);
        call(&mut file, GS_CHECK, GS_HELPER);
        put(&mut file, GS_CHECK + 5, &[0xb8, 1, 0, 0, 0, 0xc3]);
        put(&mut file, UNKNOWN_HANDLER, &[0x31, 0xc0, 0xc3]);

        // __C_specific_handler: __except(filter), __except(1), __finally.
        let scopes: Vec<u8> = [
            le(3),
            le(F_SEH + 2),
            le(F_SEH + 8),
            le(SEH_FILTER),
            le(F_SEH + 0x10),
            le(F_SEH + 2),
            le(F_SEH + 8),
            le(EXCEPTION_EXECUTE_HANDLER),
            le(F_SEH + 0x14),
            le(F_SEH + 8),
            le(F_SEH + 0xc),
            le(SEH_FINALLY),
            le(0),
        ]
        .concat();
        let flags = UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER;
        put(
            &mut file,
            UNWIND_SEH,
            &unwind_bytes(flags, &trailer(THUNK_SEH, &scopes)),
        );

        // __CxxFrameHandler3 FuncInfo (magic ..22): 2 states, 1 try block, 2 IP map entries.
        put(
            &mut file,
            UNWIND_FH3,
            &unwind_bytes(flags, &trailer(THUNK_FH3, &le(FH3_INFO))),
        );
        let info3: Vec<u8> = [
            le(0x1993_0522),
            le(2),
            le(FH3_UNWIND),
            le(1),
            le(FH3_TRY),
            le(2),
            le(FH3_IP),
            le(0x28),
            le(0),
            le(1),
        ]
        .concat();
        put(&mut file, FH3_INFO, &info3);
        put(
            &mut file,
            FH3_UNWIND,
            &[le(u32::MAX), le(FH3_CLEANUP), le(0), le(0)].concat(),
        );
        put(
            &mut file,
            FH3_TRY,
            &[le(1), le(1), le(1), le(1), le(FH3_HANDLERS)].concat(),
        );
        put(
            &mut file,
            FH3_HANDLERS,
            &[le(0x8), le(0x2700), le(0x30), le(F_CATCH), le(0x38)].concat(),
        );
        put(
            &mut file,
            FH3_IP,
            &[le(F_FH3), le(u32::MAX), le(F_FH3 + 4), le(0)].concat(),
        );

        // __CxxFrameHandler4 parent: unwind map, try map with two handlers, IP map.
        put(
            &mut file,
            UNWIND_FH4,
            &unwind_bytes(flags, &trailer(THUNK_FH4, &le(FH4_INFO))),
        );
        let info4 = [
            &[FH4_UNWIND_MAP | FH4_TRY_BLOCK_MAP][..],
            &le(FH4_UNWIND),
            &le(FH4_TRY),
            &le(FH4_IP),
        ]
        .concat();
        put(&mut file, FH4_INFO, &info4);
        // Entry starts: 1, 6, 12; next states -1, 0, 1.
        let unwind4 = [
            c(3),
            c((1 << 2) | 3),
            le(FH4_CLEANUP).to_vec(),
            c((5 << 2) | 1),
            le(FH4_DTOR).to_vec(),
            c(0x40),
            c(6 << 2),
        ]
        .concat();
        put(&mut file, FH4_UNWIND, &unwind4);
        let try4 = [c(1), c(1), c(1), c(2), le(FH4_HANDLERS).to_vec()].concat();
        put(&mut file, FH4_TRY, &try4);
        let handlers4 = [
            c(2),
            // adjectives + dispType + one function-relative continuation.
            vec![FH4_HANDLER_ADJECTIVES | FH4_HANDLER_DISP_TYPE | (1 << FH4_HANDLER_CONT_SHIFT)],
            c(8),
            le(0x2700).to_vec(),
            le(F_CATCH).to_vec(),
            c(0x30),
            // catch(...) with two RVA continuations.
            vec![FH4_HANDLER_CONT_IS_RVA | (2 << FH4_HANDLER_CONT_SHIFT)],
            le(FH4_CATCH2).to_vec(),
            le(F_FH4 + 0x34).to_vec(),
            le(F_FH4 + 0x38).to_vec(),
        ]
        .concat();
        put(&mut file, FH4_HANDLERS, &handlers4);
        let ip4 = [c(3), c(0), c(0), c(0x10), c(2), c(0x20), c(1)].concat();
        put(&mut file, FH4_IP, &ip4);

        // FH4 catch funclet: separated code, establisher frame offset, its own IP map, the
        // parent's unwind map and no try blocks of its own.
        put(
            &mut file,
            UNWIND_CATCH,
            &unwind_bytes(flags, &trailer(THUNK_FH4, &le(FH4_CATCH_INFO))),
        );
        let catch_info = [
            &[FH4_IS_CATCH | FH4_IS_SEPARATED | FH4_UNWIND_MAP][..],
            &le(FH4_UNWIND),
            &le(FH4_SEGMENTS),
            &c(0x38),
        ]
        .concat();
        put(&mut file, FH4_CATCH_INFO, &catch_info);
        let segments = [
            c(2),
            le(F_FH4).to_vec(),
            le(FH4_IP).to_vec(),
            le(F_CATCH).to_vec(),
            le(FH4_CATCH_IP).to_vec(),
        ]
        .concat();
        put(&mut file, FH4_SEGMENTS, &segments);
        put(&mut file, FH4_CATCH_IP, &[c(1), c(4), c(2)].concat());

        // GS wrapper around FH3 (data: FuncInfo RVA + GS data), GS check, unknown handler.
        put(
            &mut file,
            UNWIND_GS_FH3,
            &unwind_bytes(
                flags,
                &trailer(GS_WRAPPER_FH3, &[le(FH3_INFO), le(0x20)].concat()),
            ),
        );
        put(
            &mut file,
            UNWIND_GS,
            &unwind_bytes(flags, &trailer(GS_CHECK, &le(0x20))),
        );
        put(
            &mut file,
            UNWIND_UNKNOWN,
            &unwind_bytes(UNW_FLAG_EHANDLER, &trailer(UNKNOWN_HANDLER, &le(0))),
        );
        put(&mut file, UNWIND_PLAIN, &unwind_bytes(0, &[]));
        Fixture {
            file,
            setjmp_slot: setjmp,
        }
    }

    fn parse(file: &[u8]) -> EhMetadata {
        let pe = Pe64::parse(file).expect("synthetic PE parses");
        let dir = ExceptionDirectory::from_pe(&pe, file);
        assert!(dir.diagnostics.is_empty(), "{:?}", dir.diagnostics);
        EhMetadata::from_pe(&pe, file, &dir)
    }

    fn function(eh: &EhMetadata, begin: u32) -> &EhFunction {
        eh.functions
            .iter()
            .find(|f| f.begin == begin)
            .unwrap_or_else(|| panic!("no EH function at {begin:#x}"))
    }

    fn diag_kinds(eh: &EhMetadata) -> Vec<(u32, EhDiagnosticKind)> {
        eh.diagnostics.iter().map(|d| (d.rva, d.kind)).collect()
    }

    // ----- compressed integers ---------------------------------------------------------------

    #[test]
    fn compressed_integers_decode_every_length() {
        assert_eq!(decode_compressed(&[0x00]), Some((0, 1)));
        assert_eq!(decode_compressed(&[0xfe]), Some((0x7f, 1)));
        // 2 bytes: low bits 01, value in the upper 14 bits.
        assert_eq!(decode_compressed(&[0x01, 0x02]), Some((0x80, 2)));
        assert_eq!(decode_compressed(&[0xfd, 0xff]), Some((0x3fff, 2)));
        // 3 bytes: low bits 011.
        assert_eq!(decode_compressed(&[0x03, 0x00, 0x02]), Some((0x4000, 3)));
        // 4 bytes: low bits 0111.
        assert_eq!(
            decode_compressed(&[0xf7, 0xff, 0xff, 0xff]),
            Some((0x0fff_ffff, 4))
        );
        // 5 bytes: low nibble 1111, the high nibble is ignored and the value is the next u32.
        assert_eq!(
            decode_compressed(&[0x0f, 0x78, 0x56, 0x34, 0x12]),
            Some((0x1234_5678, 5))
        );
        assert_eq!(
            decode_compressed(&[0xff, 0xff, 0xff, 0xff, 0xff]),
            Some((u32::MAX, 5))
        );
        // Trailing bytes are not consumed.
        assert_eq!(decode_compressed(&[0x04, 0xaa]), Some((2, 1)));
    }

    #[test]
    fn compressed_integers_reject_truncation_and_round_trip() {
        assert_eq!(decode_compressed(&[]), None);
        assert_eq!(decode_compressed(&[0x01]), None);
        assert_eq!(decode_compressed(&[0x03, 0x00]), None);
        assert_eq!(decode_compressed(&[0x07, 0, 0]), None);
        assert_eq!(decode_compressed(&[0x0f, 0, 0, 0]), None);
        let edges = [
            0,
            1,
            0x7f,
            0x80,
            0x3fff,
            0x4000,
            0x1f_ffff,
            0x20_0000,
            0x0fff_ffff,
            0x1000_0000,
            u32::MAX,
        ];
        for value in edges {
            let bytes = encode_compressed(value);
            assert_eq!(
                decode_compressed(&bytes),
                Some((value, bytes.len())),
                "{value:#x}"
            );
        }
        let lengths: Vec<usize> = edges.iter().map(|&v| encode_compressed(v).len()).collect();
        assert_eq!(lengths, [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
    }

    // ----- handlers ----------------------------------------------------------------------------

    /// Regression (real `vcruntime140.dll`): the DLL that implements the handlers exports them
    /// and its own functions use them directly, and through GS wrappers that call them directly.
    #[test]
    fn identifies_exported_handlers_and_their_local_wrappers() {
        const LOCAL_WRAPPER: u32 = 0x11b0;
        const F_LOCAL: u32 = 0x11c0;
        const UNWIND_LOCAL: u32 = 0x2780;
        const UNWIND_EXPORTED: u32 = 0x2790;
        const EXPORTS: u32 = 0x27a0;
        let mut file = pe_with_pdata(&[
            (F_UNKNOWN, F_UNKNOWN + 0x10, UNWIND_EXPORTED),
            (F_LOCAL, F_LOCAL + 0x10, UNWIND_LOCAL),
        ]);
        put(&mut file, GS_HELPER, &[0xc3]);
        put(&mut file, UNKNOWN_HANDLER, &[0x31, 0xc0, 0xc3]);
        call(&mut file, LOCAL_WRAPPER, GS_HELPER);
        call(&mut file, LOCAL_WRAPPER + 5, UNKNOWN_HANDLER);
        put(&mut file, LOCAL_WRAPPER + 10, &[0xc3]);
        let no_scopes = le(0);
        put(
            &mut file,
            UNWIND_EXPORTED,
            &unwind_bytes(UNW_FLAG_EHANDLER, &trailer(UNKNOWN_HANDLER, &no_scopes)),
        );
        put(
            &mut file,
            UNWIND_LOCAL,
            &unwind_bytes(UNW_FLAG_EHANDLER, &trailer(LOCAL_WRAPPER, &no_scopes)),
        );
        // Export directory: `__C_specific_handler` at UNKNOWN_HANDLER.
        let (eat, names, ords, name, dll) = (
            EXPORTS + 40,
            EXPORTS + 44,
            EXPORTS + 48,
            EXPORTS + 52,
            EXPORTS + 80,
        );
        put(&mut file, EXPORTS + 12, &le(dll));
        put(&mut file, EXPORTS + 16, &le(1));
        put(&mut file, EXPORTS + 20, &le(1));
        put(&mut file, EXPORTS + 24, &le(1));
        put(&mut file, EXPORTS + 28, &le(eat));
        put(&mut file, EXPORTS + 32, &le(names));
        put(&mut file, EXPORTS + 36, &le(ords));
        put(&mut file, eat, &le(UNKNOWN_HANDLER));
        put(&mut file, names, &le(name));
        put(&mut file, ords, &[0, 0]);
        put(&mut file, name, b"__C_specific_handler\0");
        put(&mut file, dll, b"vc.dll\0");
        file[0x108..0x10c].copy_from_slice(&le(EXPORTS));
        file[0x10c..0x110].copy_from_slice(&le(0x60));

        let pe = Pe64::parse(&file).unwrap();
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        let eh = EhMetadata::from_pe(&pe, &file, &dir);
        assert_eq!(
            eh.handlers.get(&UNKNOWN_HANDLER),
            Some(&Handler {
                kind: HandlerKind::CSpecific,
                wrapper: false
            })
        );
        assert_eq!(
            eh.handlers.get(&LOCAL_WRAPPER),
            Some(&Handler {
                kind: HandlerKind::CSpecific,
                wrapper: true
            })
        );
        assert!(eh.unrecognised.is_empty(), "{:?}", eh.unrecognised);
        assert_eq!(eh.stats().c_specific, 2);
    }

    #[test]
    fn identifies_handlers_through_thunks_and_wrappers() {
        let fx = fixture();
        let eh = parse(&fx.file);
        let thunk = |kind| Handler {
            kind,
            wrapper: false,
        };
        let wrapper = |kind| Handler {
            kind,
            wrapper: true,
        };
        assert_eq!(
            eh.handlers
                .iter()
                .map(|(r, h)| (*r, *h))
                .collect::<Vec<_>>(),
            [
                (THUNK_SEH, thunk(HandlerKind::CSpecific)),
                (THUNK_FH3, thunk(HandlerKind::CxxFrameHandler3)),
                (THUNK_FH4, thunk(HandlerKind::CxxFrameHandler4)),
                (GS_WRAPPER_FH3, wrapper(HandlerKind::CxxFrameHandler3)),
                (GS_CHECK, wrapper(HandlerKind::GsCheck)),
            ]
        );
        assert_eq!(
            diag_kinds(&eh),
            [(UNKNOWN_HANDLER, EhDiagnosticKind::UnknownHandler)]
        );
        assert_eq!(eh.unrecognised, [F_UNKNOWN]);
        assert_eq!(eh.setjmp_slots, [fx.setjmp_slot]);
        assert_eq!(function(&eh, F_GS).data, LanguageData::None);
        // The GS wrapper passes its data through: it is the FH3 FuncInfo.
        assert!(
            matches!(&function(&eh, F_GS_FH3).data, LanguageData::Cxx3(info) if info.rva == FH3_INFO)
        );
        let stats = eh.stats();
        assert_eq!(
            (
                stats.c_specific,
                stats.cxx3,
                stats.cxx4,
                stats.gs_check,
                stats.unknown
            ),
            (1, 2, 2, 1, 1)
        );
        assert_eq!((stats.parsed, stats.diagnostics), (6, 1));
    }

    #[test]
    fn handler_reaching_two_language_handlers_is_ambiguous() {
        let mut fx = fixture();
        // The unknown handler now calls both the FH3 and the FH4 thunk.
        call(&mut fx.file, UNKNOWN_HANDLER, THUNK_FH3);
        call(&mut fx.file, UNKNOWN_HANDLER + 5, THUNK_FH4);
        put(&mut fx.file, UNKNOWN_HANDLER + 10, &[0xc3]);
        let eh = parse(&fx.file);
        assert_eq!(
            diag_kinds(&eh),
            [(UNKNOWN_HANDLER, EhDiagnosticKind::AmbiguousHandler)]
        );
    }

    // ----- formats -----------------------------------------------------------------------------

    #[test]
    fn parses_scope_tables() {
        let eh = parse(&fixture().file);
        let f = function(&eh, F_SEH);
        assert_eq!(f.data_rva, UNWIND_SEH + 12);
        assert_eq!(
            f.data,
            LanguageData::ScopeTable(vec![
                ScopeRecord {
                    begin: F_SEH + 2,
                    end: F_SEH + 8,
                    handler: ScopeHandler::Filter(SEH_FILTER),
                    jump_target: F_SEH + 0x10,
                },
                ScopeRecord {
                    begin: F_SEH + 2,
                    end: F_SEH + 8,
                    handler: ScopeHandler::Constant(EXCEPTION_EXECUTE_HANDLER),
                    jump_target: F_SEH + 0x14,
                },
                ScopeRecord {
                    begin: F_SEH + 8,
                    end: F_SEH + 0xc,
                    handler: ScopeHandler::Finally(SEH_FINALLY),
                    jump_target: 0,
                },
            ])
        );
        assert_eq!(
            f.funclets(),
            [
                (SEH_FILTER, FuncletKind::Filter),
                (SEH_FINALLY, FuncletKind::Finally)
            ]
        );
        assert_eq!(
            f.resume_targets(),
            [
                (F_SEH + 0x10, ResumeKind::ExceptBody),
                (F_SEH + 0x14, ResumeKind::ExceptBody)
            ]
        );
    }

    #[test]
    fn parses_fh3_func_info() {
        let eh = parse(&fixture().file);
        let LanguageData::Cxx3(info) = &function(&eh, F_FH3).data else {
            panic!("FH3 data");
        };
        assert_eq!(
            (
                info.magic,
                info.bbt_flags,
                info.unwind_help,
                info.es_type_list,
                info.eh_flags
            ),
            (0x1993_0522, 0, 0x28, 0, 1)
        );
        assert_eq!(
            info.unwind_map,
            [
                UnwindEntry3 {
                    to_state: -1,
                    action: FH3_CLEANUP
                },
                UnwindEntry3 {
                    to_state: 0,
                    action: 0
                }
            ]
        );
        assert_eq!(
            info.try_blocks,
            [TryBlock3 {
                try_low: 1,
                try_high: 1,
                catch_high: 1,
                handlers: vec![CatchHandler3 {
                    adjectives: 8,
                    type_descriptor: 0x2700,
                    catch_object: 0x30,
                    handler: F_CATCH,
                    frame: 0x38,
                }],
            }]
        );
        assert_eq!(info.ip_to_state, [(F_FH3, -1), (F_FH3 + 4, 0)]);
        let f = function(&eh, F_FH3);
        assert_eq!(
            f.funclets(),
            [
                (F_CATCH, FuncletKind::Catch),
                (FH3_CLEANUP, FuncletKind::Cleanup)
            ]
        );
        // FH3 catch funclets return the continuation in rax: nothing in the metadata.
        assert!(f.resume_targets().is_empty());
    }

    #[test]
    fn parses_fh4_func_info() {
        let eh = parse(&fixture().file);
        let LanguageData::Cxx4(info) = &function(&eh, F_FH4).data else {
            panic!("FH4 data");
        };
        assert!(!info.is_catch() && !info.is_separated());
        assert_eq!(
            (
                info.unwind_map_rva,
                info.try_block_map_rva,
                info.ip_map_rva,
                info.frame
            ),
            (FH4_UNWIND, FH4_TRY, FH4_IP, None)
        );
        assert_eq!(
            info.unwind_map,
            [
                UnwindEntry4 {
                    kind: UnwindKind4::Funclet,
                    next_state: -1,
                    action: FH4_CLEANUP,
                    object: 0
                },
                UnwindEntry4 {
                    kind: UnwindKind4::DtorWithObj,
                    next_state: 0,
                    action: FH4_DTOR,
                    object: 0x40
                },
                UnwindEntry4 {
                    kind: UnwindKind4::None,
                    next_state: 1,
                    action: 0,
                    object: 0
                },
            ]
        );
        assert_eq!(info.try_blocks.len(), 1);
        let block = &info.try_blocks[0];
        assert_eq!(
            (
                block.try_low,
                block.try_high,
                block.catch_high,
                block.handler_array
            ),
            (1, 1, 2, FH4_HANDLERS)
        );
        assert_eq!(
            block.handlers,
            [
                CatchHandler4 {
                    header: 0x13,
                    adjectives: 8,
                    type_descriptor: 0x2700,
                    catch_object: 0,
                    handler: F_CATCH,
                    continuations: vec![F_FH4 + 0x30],
                },
                CatchHandler4 {
                    header: 0x28,
                    adjectives: 0,
                    type_descriptor: 0,
                    catch_object: 0,
                    handler: FH4_CATCH2,
                    continuations: vec![F_FH4 + 0x34, F_FH4 + 0x38],
                },
            ]
        );
        assert_eq!(
            info.ip_to_state,
            [(F_FH4, -1), (F_FH4 + 0x10, 1), (F_FH4 + 0x30, 0)]
        );
        let f = function(&eh, F_FH4);
        assert_eq!(
            f.funclets(),
            [
                (F_CATCH, FuncletKind::Catch),
                (FH4_CATCH2, FuncletKind::Catch),
                (FH4_CLEANUP, FuncletKind::Cleanup),
                (FH4_DTOR, FuncletKind::Destructor),
            ]
        );
        assert_eq!(
            f.resume_targets(),
            [0x30, 0x34, 0x38].map(|off| (F_FH4 + off, ResumeKind::CatchContinuation))
        );
    }

    #[test]
    fn parses_fh4_separated_catch_funclet() {
        let eh = parse(&fixture().file);
        let LanguageData::Cxx4(info) = &function(&eh, F_CATCH).data else {
            panic!("FH4 data");
        };
        assert!(info.is_catch() && info.is_separated());
        assert_eq!(info.frame, Some(0x38));
        // The segment table entry for this funclet selects its own IP map.
        assert_eq!(info.ip_map_rva, FH4_CATCH_IP);
        assert_eq!(info.ip_to_state, [(F_CATCH + 4, 1)]);
        assert_eq!(info.unwind_map.len(), 3);
        assert!(info.try_blocks.is_empty());
        assert!(function(&eh, F_CATCH).resume_targets().is_empty());
    }

    // ----- malformed input ---------------------------------------------------------------------

    fn single_diag(eh: &EhMetadata, begin: u32) -> Vec<(u32, EhDiagnosticKind)> {
        eh.diagnostics
            .iter()
            .filter(|d| d.rva == begin)
            .map(|d| (d.address, d.kind))
            .collect()
    }

    #[test]
    fn malformed_fh3_is_diagnosed() {
        // Unknown magic.
        let mut fx = fixture();
        put(&mut fx.file, FH3_INFO, &le(0x1993_0523));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [(FH3_INFO, EhDiagnosticKind::UnknownMagic)]
        );
        assert_eq!(function(&eh, F_FH3).data, LanguageData::None);

        // Huge state count and a try map running off the end of .rdata.
        let mut fx = fixture();
        put(&mut fx.file, FH3_INFO + 4, &le(0x8000_0000));
        put(&mut fx.file, FH3_INFO + 16, &le(0x27f0));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [
                (FH3_UNWIND, EhDiagnosticKind::TooLarge),
                (0x27f0, EhDiagnosticKind::Truncated)
            ]
        );

        // FuncInfo RVA outside the image.
        let mut fx = fixture();
        put(&mut fx.file, UNWIND_FH3 + 12, &le(0x9000));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [(0x9000, EhDiagnosticKind::Truncated)]
        );

        // A catch funclet that is not code.
        let mut fx = fixture();
        put(&mut fx.file, FH3_HANDLERS + 12, &le(0x2010));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [(0x2010, EhDiagnosticKind::NotCode)]
        );
    }

    #[test]
    fn malformed_fh4_is_diagnosed() {
        // Reserved continuation count 3 (two values are still read).
        let mut fx = fixture();
        put(
            &mut fx.file,
            FH4_HANDLERS + 12,
            &[FH4_HANDLER_CONT_IS_RVA | (3 << FH4_HANDLER_CONT_SHIFT)],
        );
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(FH4_HANDLERS, EhDiagnosticKind::Reserved)]
        );

        // An unwind link that lands inside an entry.
        let mut fx = fixture();
        put(&mut fx.file, FH4_UNWIND + 6, &c((3 << 2) | 1));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(FH4_UNWIND, EhDiagnosticKind::BadUnwindLink)]
        );

        // Separated funclet without a segment entry.
        let mut fx = fixture();
        put(&mut fx.file, FH4_SEGMENTS + 9, &le(0x1111));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_CATCH),
            [(FH4_SEGMENTS, EhDiagnosticKind::MissingSegment)]
        );

        // Counts above the limit (5-byte compressed integer) and a map at the end of .rdata.
        let mut fx = fixture();
        put(&mut fx.file, FH4_TRY, &[0x0f, 1, 0, 1, 0]);
        put(&mut fx.file, FH4_INFO + 9, &le(0x27ff));
        put(&mut fx.file, 0x27ff, &[0x01]);
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [
                (FH4_TRY, EhDiagnosticKind::TooLarge),
                (0x27ff, EhDiagnosticKind::Truncated)
            ]
        );

        // FuncInfo4 cut off by the end of the section.
        let mut fx = fixture();
        put(&mut fx.file, UNWIND_FH4 + 12, &le(0x27fc));
        put(
            &mut fx.file,
            0x27fc,
            &[FH4_UNWIND_MAP | FH4_TRY_BLOCK_MAP, 0, 0, 0],
        );
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(0x27fc, EhDiagnosticKind::Truncated)]
        );
    }

    #[test]
    fn malformed_scope_tables_are_diagnosed() {
        let mut fx = fixture();
        // Count 0x10001 exceeds the limit.
        put(&mut fx.file, UNWIND_SEH + 12, &le(MAX_TABLE_ENTRIES + 1));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_SEH),
            [(UNWIND_SEH + 12, EhDiagnosticKind::TooLarge)]
        );

        let mut fx = fixture();
        // A filter "constant" that is neither code nor EXCEPTION_EXECUTE_HANDLER, and an
        // __except body in data.
        put(&mut fx.file, UNWIND_SEH + 16 + 8, &le(7));
        put(&mut fx.file, UNWIND_SEH + 16 + 12, &le(0x2004));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_SEH),
            [
                (0x2004, EhDiagnosticKind::NotCode),
                (7, EhDiagnosticKind::NotCode)
            ]
        );
    }

    #[test]
    fn garbage_data_never_panics() {
        let fx = fixture();
        let mut seed = 0x9e37_79b9u32;
        for round in 0..64 {
            let mut file = fx.file.clone();
            // Scramble the EH data area (unwind infos stay intact for the first rounds).
            let from = if round < 32 { FH3_INFO } else { UNWIND_SEH };
            for rva in from..0x2800 {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                put(&mut file, rva, &[(seed >> 16) as u8]);
            }
            let pe = Pe64::parse(&file).expect("parses");
            let dir = ExceptionDirectory::from_pe(&pe, &file);
            let eh = EhMetadata::from_pe(&pe, &file, &dir);
            let _ = eh.stats();
        }
    }

    #[test]
    fn parse_is_deterministic() {
        let fx = fixture();
        assert_eq!(parse(&fx.file), parse(&fx.file));
    }

    // ----- discovery seeds ---------------------------------------------------------------------

    #[test]
    fn seed_adds_funclets_resume_targets_and_setjmp_slots() {
        let fx = fixture();
        let pe = Pe64::parse(&fx.file).expect("parses");
        let dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        let eh = EhMetadata::from_pe(&pe, &fx.file, &dir);
        let mut image = CodeImage::from_pe(&pe, &fx.file);
        dir.seed(&mut image);
        let before: BTreeSet<u64> = image.entry_points.iter().copied().collect();
        eh.seed(&mut image);
        let after: BTreeSet<u64> = image.entry_points.iter().copied().collect();
        let added: Vec<u32> = after
            .difference(&before)
            .map(|va| (va - BASE) as u32)
            .collect();
        // Funclets without their own .pdata record (F_CATCH and F_CLEANUP already have one).
        assert_eq!(
            added,
            [FH4_CATCH2, SEH_FILTER, SEH_FINALLY, FH3_CLEANUP, FH4_DTOR]
        );
        let va = |rva: u32| BASE + u64::from(rva);
        assert_eq!(
            image.eh.resume_targets,
            [
                (va(F_SEH), va(F_SEH + 0x10)),
                (va(F_SEH), va(F_SEH + 0x14)),
                (va(F_FH4), va(F_FH4 + 0x30)),
                (va(F_FH4), va(F_FH4 + 0x34)),
                (va(F_FH4), va(F_FH4 + 0x38)),
            ]
        );
        assert_eq!(image.eh.setjmp_slots, [va(fx.setjmp_slot)]);
        assert_eq!(image.eh.stats, eh.stats());
        assert_eq!(image.eh.stats.resume_targets, 5);
    }

    // ----- R52 hardening: bounds, checked RVA math, OOM prevention & determinism ---------------

    #[test]
    fn scope_table_huge_and_truncated_counts_never_oom_or_panic() {
        // Huge counts that exceed MAX_TABLE_ENTRIES or would overflow `n * 16` in u32.
        for huge in [u32::MAX, 0x1000_0000, MAX_TABLE_ENTRIES + 1] {
            let mut fx = fixture();
            put(&mut fx.file, UNWIND_SEH + 12, &le(huge));
            let eh = parse(&fx.file);
            assert_eq!(
                single_diag(&eh, F_SEH),
                [(UNWIND_SEH + 12, EhDiagnosticKind::TooLarge)],
                "count={huge:#x}"
            );
            assert_eq!(function(&eh, F_SEH).data, LanguageData::None);
        }

        // Count right at MAX_TABLE_ENTRIES (65536): allowed by cap, but truncated in `.rdata`;
        // must fail bounds check without OOM pre-allocation.
        let mut fx = fixture();
        put(&mut fx.file, UNWIND_SEH + 12, &le(MAX_TABLE_ENTRIES));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_SEH),
            [(UNWIND_SEH + 12, EhDiagnosticKind::Truncated)]
        );
        assert_eq!(function(&eh, F_SEH).data, LanguageData::None);

        // Truncated scope table near the end of `.rdata` (0x2800): claims 2 records (32 bytes)
        // when only 20 bytes remain after the count field.
        let mut fx = fixture();
        let pe = Pe64::parse(&fx.file).unwrap();
        let mut dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        put(&mut fx.file, 0x27e8, &le(2));
        for entry in &mut dir.entries {
            if entry.function.begin == F_SEH {
                entry.unwind.as_mut().unwrap().handler.as_mut().unwrap().data_rva = 0x27e8;
            }
        }
        let eh = EhMetadata::from_pe(&pe, &fx.file, &dir);
        assert_eq!(
            single_diag(&eh, F_SEH),
            [(0x27e8, EhDiagnosticKind::Truncated)]
        );
        assert_eq!(function(&eh, F_SEH).data, LanguageData::None);

        // Empty (0-record) scope table sitting right at the last 4 bytes of `.rdata` (`0x27fc`)
        // must succeed without reading past section end.
        let mut fx = fixture();
        put(&mut fx.file, 0x27fc, &le(0));
        let pe = Pe64::parse(&fx.file).unwrap();
        let mut dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        for entry in &mut dir.entries {
            if entry.function.begin == F_SEH {
                entry.unwind.as_mut().unwrap().handler.as_mut().unwrap().data_rva = 0x27fc;
            }
        }
        let eh = EhMetadata::from_pe(&pe, &fx.file, &dir);
        assert!(single_diag(&eh, F_SEH).is_empty());
        assert_eq!(function(&eh, F_SEH).data, LanguageData::ScopeTable(vec![]));
    }

    #[test]
    fn scope_table_overflowing_and_out_of_bounds_rvas_are_diagnosed() {
        // 1) `data_rva` near `u32::MAX` so `rva.checked_add(4)` overflows.
        let fx = fixture();
        let pe = Pe64::parse(&fx.file).unwrap();
        let mut dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        for entry in &mut dir.entries {
            if entry.function.begin == F_SEH {
                entry.unwind.as_mut().unwrap().handler.as_mut().unwrap().data_rva = u32::MAX - 2;
            }
        }
        let eh = EhMetadata::from_pe(&pe, &fx.file, &dir);
        assert_eq!(
            single_diag(&eh, F_SEH),
            [(u32::MAX - 2, EhDiagnosticKind::Truncated)]
        );
        assert_eq!(function(&eh, F_SEH).data, LanguageData::None);

        // 2) Inverted scope range (`begin > end`), out-of-image `end`, and `u32::MAX` handler /
        //    jump_target RVAs.
        let mut fx = fixture();
        let bad_scopes: Vec<u8> = [
            le(2),
            // Inverted range + overflowing jump target + overflowing filter.
            le(F_SEH + 0x10),
            le(F_SEH + 0x04),
            le(u32::MAX),
            le(u32::MAX),
            // Out-of-image end + overflowing finally handler.
            le(F_SEH + 0x02),
            le(u32::MAX),
            le(u32::MAX),
            le(0),
        ]
        .concat();
        put(
            &mut fx.file,
            UNWIND_SEH,
            &unwind_bytes(
                UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER,
                &trailer(THUNK_SEH, &bad_scopes),
            ),
        );
        let pe = Pe64::parse(&fx.file).unwrap();
        let dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        let eh = EhMetadata::from_pe(&pe, &fx.file, &dir);
        let diags = single_diag(&eh, F_SEH);
        assert!(diags.contains(&(F_SEH + 0x10, EhDiagnosticKind::NotCode)));
        assert!(diags.contains(&(F_SEH + 0x02, EhDiagnosticKind::NotCode)));
        assert!(diags.contains(&(u32::MAX, EhDiagnosticKind::NotCode)));
        // Non-code targets must never be seeded into discovery.
        let mut image = CodeImage::from_pe(&pe, &fx.file);
        dir.seed(&mut image);
        eh.seed(&mut image);
        assert!(
            !image
                .eh
                .resume_targets
                .iter()
                .any(|&(owner, _)| owner == BASE + u64::from(F_SEH))
        );

        // 3) Extreme `image_base` near `u64::MAX` where `base + rva` overflows `u64`:
        //    must not panic or clamp to `u64::MAX`.
        let fx = fixture();
        let mut pe = Pe64::parse(&fx.file).unwrap();
        let dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        pe.optional.image_base = u64::MAX - 0x100;
        let eh = EhMetadata::from_pe(&pe, &fx.file, &dir);
        let mut image = CodeImage::from_pe(&pe, &fx.file);
        eh.seed(&mut image);
        assert!(image.eh.resume_targets.is_empty());
        assert!(image.eh.setjmp_slots.is_empty());
    }

    #[test]
    fn scope_table_duplicates_and_permutations_yield_canonical_outputs() {
        let build = |records: &[[u32; 4]]| {
            let mut fx = fixture();
            let mut raw = Vec::new();
            raw.extend_from_slice(&le(records.len() as u32));
            for rec in records {
                for &word in rec {
                    raw.extend_from_slice(&le(word));
                }
            }
            // Place the custom scope table at a roomy `.rdata` slot (`0x2780`) so 6 records fit.
            put(&mut fx.file, 0x2780, &raw);
            let pe = Pe64::parse(&fx.file).unwrap();
            let mut dir = ExceptionDirectory::from_pe(&pe, &fx.file);
            for entry in &mut dir.entries {
                if entry.function.begin == F_SEH {
                    entry.unwind.as_mut().unwrap().handler.as_mut().unwrap().data_rva = 0x2780;
                }
            }
            let eh = EhMetadata::from_pe(&pe, &fx.file, &dir);
            let mut image = CodeImage::from_pe(&pe, &fx.file);
            dir.seed(&mut image);
            eh.seed(&mut image);
            (eh, image.entry_points, image.eh)
        };

        let r0 = [F_SEH + 2, F_SEH + 8, SEH_FILTER, F_SEH + 0x10];
        let r1 = [F_SEH + 2, F_SEH + 8, EXCEPTION_EXECUTE_HANDLER, F_SEH + 0x14];
        let r2 = [F_SEH + 8, F_SEH + 0xc, SEH_FINALLY, 0];

        let (eh_orig, ep_orig, seeds_orig) = build(&[r0, r1, r2]);
        let (eh_perm, ep_perm, seeds_perm) = build(&[r2, r1, r0]);
        let (eh_dups, ep_dups, seeds_dups) = build(&[r2, r0, r2, r1, r0, r1]);

        let expected_funclets = vec![
            (SEH_FILTER, FuncletKind::Filter),
            (SEH_FINALLY, FuncletKind::Finally),
        ];
        let expected_resumes = vec![
            (F_SEH + 0x10, ResumeKind::ExceptBody),
            (F_SEH + 0x14, ResumeKind::ExceptBody),
        ];

        for eh in [&eh_orig, &eh_perm, &eh_dups] {
            let f = function(eh, F_SEH);
            assert_eq!(f.funclets(), expected_funclets);
            assert_eq!(f.resume_targets(), expected_resumes);
            assert_eq!(eh.funclets(), eh_orig.funclets());
            assert_eq!(eh.resume_targets(), eh_orig.resume_targets());
            assert_eq!(eh.stats(), eh_orig.stats());
        }
        assert_eq!(ep_orig, ep_perm);
        assert_eq!(ep_orig, ep_dups);
        assert_eq!(seeds_orig, seeds_perm);
        assert_eq!(seeds_orig, seeds_dups);
    }

    #[test]
    fn permuted_and_duplicate_pdata_entries_yield_identical_eh_metadata() {
        let fx = fixture();
        let pe = Pe64::parse(&fx.file).unwrap();
        let dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        let eh_canon = EhMetadata::from_pe(&pe, &fx.file, &dir);

        // Reverse entry order.
        let mut dir_rev = dir.clone();
        dir_rev.entries.reverse();
        let eh_rev = EhMetadata::from_pe(&pe, &fx.file, &dir_rev);

        // Rotate and duplicate every entry.
        let mut dir_dup = dir.clone();
        dir_dup.entries.rotate_left(3);
        let cloned = dir_dup.entries.clone();
        dir_dup.entries.extend(cloned);
        let eh_dup = EhMetadata::from_pe(&pe, &fx.file, &dir_dup);

        assert_eq!(eh_canon, eh_rev);
        assert_eq!(eh_canon, eh_dup);
        assert!(
            eh_canon
                .functions
                .windows(2)
                .all(|w| w[0].begin < w[1].begin)
        );
        assert!(
            eh_canon
                .unrecognised
                .windows(2)
                .all(|w| w[0] < w[1])
        );
    }

    #[test]
    fn fh4_overflowing_ip_delta_and_huge_truncated_tables_never_oom_or_panic() {
        // 1) IP-to-state delta that overflows `u32` (`function + u32::MAX`).
        let mut fx = fixture();
        put(&mut fx.file, FH4_IP, &[c(1), c(u32::MAX), c(1)].concat());
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(FH4_IP, EhDiagnosticKind::Truncated)]
        );
        let LanguageData::Cxx4(info) = &function(&eh, F_FH4).data else {
            panic!("FH4 data");
        };
        assert!(info.ip_to_state.is_empty());

        // 2) IP-to-state raw state exceeding `i32::MAX + 1`.
        let mut fx = fixture();
        put(&mut fx.file, FH4_IP, &[c(1), c(0), c(u32::MAX)].concat());
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(FH4_IP, EhDiagnosticKind::TooLarge)]
        );

        // 3) Function-relative continuation offset overflowing `u32`.
        let mut fx = fixture();
        let bad_handler = [
            c(1),
            vec![1 << FH4_HANDLER_CONT_SHIFT],
            le(F_CATCH).to_vec(),
            c(u32::MAX),
        ]
        .concat();
        put(&mut fx.file, FH4_HANDLERS, &bad_handler);
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH4),
            [(FH4_HANDLERS, EhDiagnosticKind::Truncated)]
        );

        // 4) Unwind map / try map / handler array / IP map each claiming MAX_TABLE_ENTRIES right
        //    near the end of `.rdata` (only a few bytes left) must truncate cleanly without OOM.
        let mut fx = fixture();
        let near_end = 0x27f8;
        put(&mut fx.file, near_end, &c(MAX_TABLE_ENTRIES));
        put(&mut fx.file, FH4_INFO + 1, &le(near_end));
        put(&mut fx.file, FH4_INFO + 5, &le(near_end));
        put(&mut fx.file, FH4_INFO + 9, &le(near_end));
        let eh = parse(&fx.file);
        let diags = single_diag(&eh, F_FH4);
        assert_eq!(
            diags,
            [
                (near_end, EhDiagnosticKind::Truncated),
                (near_end, EhDiagnosticKind::Truncated),
                (near_end, EhDiagnosticKind::Truncated),
            ]
        );
    }

    #[test]
    fn fh3_overflowing_record_tables_and_truncated_bounds() {
        // 1) `max_state = MAX_TABLE_ENTRIES` at `0x27f8` (truncated, must not OOM) and
        //    `unwind_rva` near `u32::MAX` (RVA + byte_len overflows `u32`).
        let mut fx = fixture();
        put(&mut fx.file, FH3_INFO + 4, &le(MAX_TABLE_ENTRIES));
        put(&mut fx.file, FH3_INFO + 8, &le(0x27f8));
        put(&mut fx.file, FH3_INFO + 12, &le(2));
        put(&mut fx.file, FH3_INFO + 16, &le(u32::MAX - 4));
        // Zero-count IP map with bogus RVA must still succeed with empty `ip_to_state`.
        put(&mut fx.file, FH3_INFO + 20, &le(0));
        put(&mut fx.file, FH3_INFO + 24, &le(u32::MAX));
        let eh = parse(&fx.file);
        assert_eq!(
            single_diag(&eh, F_FH3),
            [
                (0x27f8, EhDiagnosticKind::Truncated),
                (u32::MAX - 4, EhDiagnosticKind::Truncated),
            ]
        );
        let LanguageData::Cxx3(info) = &function(&eh, F_FH3).data else {
            panic!("FH3 data");
        };
        assert!(info.unwind_map.is_empty());
        assert!(info.try_blocks.is_empty());
        assert!(info.ip_to_state.is_empty());
    }

    #[test]
    fn reader_clamps_to_section_virtual_size() {
        let fx = fixture();
        let pe = Pe64::parse(&fx.file).unwrap();
        let dir = ExceptionDirectory::from_pe(&pe, &fx.file);
        // Shrink `.rdata`'s VirtualSize so the SEH scope table (at `UNWIND_SEH + 12 = 0x25cc`,
        // spanning 52 bytes to `0x2600`) crosses the virtual section boundary even though the
        // backing raw bytes still exist in the file buffer.
        let mut pe_shrunk = pe;
        let rdata = pe_shrunk
            .sections
            .iter_mut()
            .find(|s| s.virtual_address == 0x2000)
            .expect(".rdata section");
        rdata.virtual_size = 0x5e0; // ends at RVA 0x25e0, mid-way through the 3 scope records
        let eh = EhMetadata::from_pe(&pe_shrunk, &fx.file, &dir);
        assert_eq!(
            single_diag(&eh, F_SEH),
            [(UNWIND_SEH + 12, EhDiagnosticKind::Truncated)]
        );
        assert_eq!(function(&eh, F_SEH).data, LanguageData::None);
    }

    // ----- R73: chained/indirect unwind, multi-hop thunks & separated FH4 chunks ---------------

    /// `jmp rel32` at `at` targeting `target`.
    fn jmp_rel32(file: &mut [u8], at: u32, target: u32) {
        put(file, at, &[0xe9]);
        put(file, at + 1, &le(target.wrapping_sub(at + 5)));
    }

    /// Encode a 12-byte `RUNTIME_FUNCTION` record.
    fn rf_bytes(begin: u32, end: u32, unwind: u32) -> [u8; 12] {
        let mut out = [0u8; 12];
        out[0..4].copy_from_slice(&le(begin));
        out[4..8].copy_from_slice(&le(end));
        out[8..12].copy_from_slice(&le(unwind));
        out
    }

    #[test]
    fn chained_unwind_and_inline_xdata_parent_inherit_seh_scope_table() {
        use crate::pdata::UNW_FLAG_CHAININFO;

        const PRIMARY: u32 = 0x1020;
        const COLD_CHUNK: u32 = 0x1060;
        const INLINE_CHILD: u32 = 0x10a0;
        const INLINE_PARENT_BEGIN: u32 = 0x10c0;
        const FILTER_FN: u32 = 0x1140;
        const FINALLY_FN: u32 = 0x1148;
        const EXCEPT_BODY: u32 = PRIMARY + 0x18;
        const OUTSIDE_RESUME: u32 = 0x1150; // in .text, outside any .pdata range
        const THUNK: u32 = 0x1170;

        const UW_PRIMARY: u32 = 0x25c0;
        const UW_COLD: u32 = 0x2620;
        const UW_INLINE_PARENT: u32 = 0x2640;
        const UW_INLINE_CHILD: u32 = 0x26a0;

        // Note: INLINE_PARENT_BEGIN is NOT a standalone top-level .pdata entry; it exists only as
        // an inline chained RUNTIME_FUNCTION inside UW_INLINE_CHILD's .xdata!
        let mut file = pe_with_pdata(&[
            (PRIMARY, PRIMARY + 0x30, UW_PRIMARY),
            (COLD_CHUNK, COLD_CHUNK + 0x20, UW_COLD),
            (INLINE_CHILD, INLINE_CHILD + 0x18, UW_INLINE_CHILD),
        ]);
        let slots = set_imports(&mut file, &[("VCRUNTIME140.dll", &["__C_specific_handler"])]);
        jmp_slot(&mut file, THUNK, slots[0][0]);
        put(&mut file, FILTER_FN, &[0xb8, 1, 0, 0, 0, 0xc3]);
        put(&mut file, FINALLY_FN, &[0xc3]);
        put(&mut file, EXCEPT_BODY, &[0xc3]);
        put(&mut file, OUTSIDE_RESUME, &[0xc3]);

        let primary_scopes: Vec<u8> = [
            le(2),
            le(PRIMARY + 2),
            le(PRIMARY + 0x10),
            le(FILTER_FN),
            le(EXCEPT_BODY),
            le(PRIMARY + 0x10),
            le(PRIMARY + 0x18),
            le(FINALLY_FN),
            le(0),
        ]
        .concat();
        let flags = UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER;
        put(
            &mut file,
            UW_PRIMARY,
            &unwind_bytes(flags, &trailer(THUNK, &primary_scopes)),
        );
        // COLD_CHUNK chains to PRIMARY's RUNTIME_FUNCTION.
        put(
            &mut file,
            UW_COLD,
            &unwind_bytes(
                UNW_FLAG_CHAININFO,
                &rf_bytes(PRIMARY, PRIMARY + 0x30, UW_PRIMARY),
            ),
        );

        // Inline parent UNWIND_INFO with a scope whose jump_target lies outside .pdata.
        let inline_scopes: Vec<u8> = [
            le(1),
            le(INLINE_CHILD),
            le(INLINE_CHILD + 0x10),
            le(EXCEPTION_EXECUTE_HANDLER),
            le(OUTSIDE_RESUME),
        ]
        .concat();
        put(
            &mut file,
            UW_INLINE_PARENT,
            &unwind_bytes(flags, &trailer(THUNK, &inline_scopes)),
        );
        put(
            &mut file,
            UW_INLINE_CHILD,
            &unwind_bytes(
                UNW_FLAG_CHAININFO,
                &rf_bytes(
                    INLINE_PARENT_BEGIN,
                    INLINE_PARENT_BEGIN + 0x20,
                    UW_INLINE_PARENT,
                ),
            ),
        );

        let pe = Pe64::parse(&file).expect("PE parses");
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        let eh = EhMetadata::from_pe(&pe, &file, &dir);

        assert!(eh.diagnostics.is_empty(), "{:?}", eh.diagnostics);
        assert_eq!(
            eh.handlers.get(&THUNK),
            Some(&Handler {
                kind: HandlerKind::CSpecific,
                wrapper: false,
            })
        );
        // All three .pdata records resolve their SEH scope table.
        let cold = function(&eh, COLD_CHUNK);
        assert_eq!(cold.handler_rva, THUNK);
        assert_eq!(
            cold.funclets(),
            [
                (FILTER_FN, FuncletKind::Filter),
                (FINALLY_FN, FuncletKind::Finally),
            ]
        );
        assert_eq!(
            cold.resume_targets(),
            [(EXCEPT_BODY, ResumeKind::ExceptBody)]
        );

        let inline_fn = function(&eh, INLINE_CHILD);
        assert_eq!(inline_fn.handler_rva, THUNK);
        assert_eq!(
            inline_fn.resume_targets(),
            [(OUTSIDE_RESUME, ResumeKind::ExceptBody)]
        );

        // Seed into CodeImage: handler routine, funclets, and resume targets are all seeded,
        // and OUTSIDE_RESUME falls back to INLINE_CHILD (its primary in function_table).
        let mut image = CodeImage::from_pe(&pe, &file);
        dir.seed(&mut image);
        eh.seed(&mut image);
        let va = |rva: u32| BASE + u64::from(rva);
        assert!(image.entry_points.contains(&va(THUNK)));
        assert!(image.entry_points.contains(&va(FILTER_FN)));
        assert!(image.entry_points.contains(&va(FINALLY_FN)));
        assert!(image.eh.resume_targets.contains(&(va(PRIMARY), va(EXCEPT_BODY))));
        assert!(
            image
                .eh
                .resume_targets
                .contains(&(va(INLINE_CHILD), va(OUTSIDE_RESUME)))
        );
    }

    #[test]
    fn indirect_runtime_function_resolves_parent_handler() {
        const PRIMARY: u32 = 0x1020;
        const CHUNK: u32 = 0x1060;
        const THUNK: u32 = 0x1170;
        const FINALLY_FN: u32 = 0x1150;
        const UW_PRIMARY: u32 = 0x25c0;
        const RF_INDIRECT_SLOT: u32 = 0x2600;

        // CHUNK uses an indirect UnwindInfoAddress (bit 0 set) pointing at RF_INDIRECT_SLOT.
        let mut file = pe_with_pdata(&[
            (PRIMARY, PRIMARY + 0x20, UW_PRIMARY),
            (CHUNK, CHUNK + 0x10, RF_INDIRECT_SLOT | RUNTIME_FUNCTION_INDIRECT),
        ]);
        let slots = set_imports(&mut file, &[("VCRUNTIME140.dll", &["__C_specific_handler"])]);
        jmp_slot(&mut file, THUNK, slots[0][0]);
        put(&mut file, FINALLY_FN, &[0xc3]);

        let scopes: Vec<u8> = [
            le(1),
            le(PRIMARY),
            le(PRIMARY + 0x10),
            le(FINALLY_FN),
            le(0),
        ]
        .concat();
        put(
            &mut file,
            UW_PRIMARY,
            &unwind_bytes(
                UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER,
                &trailer(THUNK, &scopes),
            ),
        );
        put(
            &mut file,
            RF_INDIRECT_SLOT,
            &rf_bytes(PRIMARY, PRIMARY + 0x20, UW_PRIMARY),
        );

        let pe = Pe64::parse(&file).expect("PE parses");
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        assert!(dir.diagnostics.is_empty(), "{:?}", dir.diagnostics);
        let eh = EhMetadata::from_pe(&pe, &file, &dir);
        assert!(eh.diagnostics.is_empty(), "{:?}", eh.diagnostics);
        let chunk_fn = function(&eh, CHUNK);
        assert_eq!(chunk_fn.handler_rva, THUNK);
        assert_eq!(chunk_fn.funclets(), [(FINALLY_FN, FuncletKind::Finally)]);
    }

    #[test]
    fn multi_hop_handler_thunks_and_gs_wrapper_are_identified() {
        const F_DIRECT: u32 = 0x1020;
        const F_WRAP: u32 = 0x1040;
        const F_LOOP: u32 = 0x1060;
        const THUNK_IAT: u32 = 0x1100;
        const THUNK_HOP1: u32 = 0x1110;
        const THUNK_HOP2: u32 = 0x1120;
        const WRAPPER: u32 = 0x1130;
        const HELPER: u32 = 0x1150;
        const CYCLE_A: u32 = 0x1160;
        const CYCLE_B: u32 = 0x1168;
        const UW_DIRECT: u32 = 0x25c0;
        const UW_WRAP: u32 = 0x25e0;
        const UW_LOOP: u32 = 0x2600;

        let mut file = pe_with_pdata(&[
            (F_DIRECT, F_DIRECT + 0x10, UW_DIRECT),
            (F_WRAP, F_WRAP + 0x10, UW_WRAP),
            (F_LOOP, F_LOOP + 0x10, UW_LOOP),
        ]);
        let slots = set_imports(&mut file, &[("VCRUNTIME140.dll", &["__C_specific_handler"])]);
        // 2-hop thunk: THUNK_HOP2 -> jmp THUNK_HOP1 -> jmp [rip + iat]
        jmp_slot(&mut file, THUNK_IAT, slots[0][0]);
        jmp_rel32(&mut file, THUNK_HOP1, THUNK_IAT);
        jmp_rel32(&mut file, THUNK_HOP2, THUNK_HOP1);
        // __GSHandlerCheck_SEH wrapper calling helper then multi-hop thunk.
        put(&mut file, HELPER, &[0xc3]);
        call(&mut file, WRAPPER, HELPER);
        jmp_rel32(&mut file, WRAPPER + 5, THUNK_HOP2);
        // Cycle in direct jmp thunks must terminate cleanly as UnknownHandler.
        jmp_rel32(&mut file, CYCLE_A, CYCLE_B);
        jmp_rel32(&mut file, CYCLE_B, CYCLE_A);

        let empty_scopes = le(0);
        put(
            &mut file,
            UW_DIRECT,
            &unwind_bytes(UNW_FLAG_EHANDLER, &trailer(THUNK_HOP2, &empty_scopes)),
        );
        put(
            &mut file,
            UW_WRAP,
            &unwind_bytes(UNW_FLAG_EHANDLER, &trailer(WRAPPER, &empty_scopes)),
        );
        put(
            &mut file,
            UW_LOOP,
            &unwind_bytes(UNW_FLAG_EHANDLER, &trailer(CYCLE_A, &empty_scopes)),
        );

        let eh = parse(&file);
        assert_eq!(
            eh.handlers.get(&THUNK_HOP2),
            Some(&Handler {
                kind: HandlerKind::CSpecific,
                wrapper: false,
            })
        );
        assert_eq!(
            eh.handlers.get(&WRAPPER),
            Some(&Handler {
                kind: HandlerKind::CSpecific,
                wrapper: true,
            })
        );
        assert!(!eh.handlers.contains_key(&CYCLE_A));
        assert_eq!(eh.unrecognised, [F_LOOP]);
        assert_eq!(
            diag_kinds(&eh),
            [(CYCLE_A, EhDiagnosticKind::UnknownHandler)]
        );
    }

    #[test]
    fn separated_fh4_chained_chunk_resolves_continuations_against_primary() {
        use crate::pdata::UNW_FLAG_CHAININFO;

        const PRIMARY: u32 = 0x1020;
        const COLD: u32 = 0x1080;
        const CATCH_FN: u32 = 0x1100;
        const THUNK: u32 = 0x1140;

        const UW_PRIMARY: u32 = 0x25c0;
        const UW_COLD: u32 = 0x25e0;
        const INFO4: u32 = 0x2600;
        const TRY4: u32 = 0x2620;
        const HANDLERS4: u32 = 0x2630;
        const SEGMENTS4: u32 = 0x2650;
        const IP_PRIMARY: u32 = 0x2670;
        const IP_COLD: u32 = 0x2680;

        let mut file = pe_with_pdata(&[
            (PRIMARY, PRIMARY + 0x40, UW_PRIMARY),
            (COLD, COLD + 0x20, UW_COLD),
        ]);
        let slots = set_imports(&mut file, &[("VCRUNTIME140_1.dll", &["__CxxFrameHandler4"])]);
        jmp_slot(&mut file, THUNK, slots[0][0]);
        put(&mut file, CATCH_FN, &[0xc3]);

        put(
            &mut file,
            UW_PRIMARY,
            &unwind_bytes(
                UNW_FLAG_EHANDLER | UNW_FLAG_UHANDLER,
                &trailer(THUNK, &le(INFO4)),
            ),
        );
        // COLD chains to PRIMARY so entry.function.begin == COLD while entry.primary == PRIMARY.
        put(
            &mut file,
            UW_COLD,
            &unwind_bytes(
                UNW_FLAG_CHAININFO,
                &rf_bytes(PRIMARY, PRIMARY + 0x40, UW_PRIMARY),
            ),
        );

        let info4 = [
            &[FH4_IS_SEPARATED | FH4_TRY_BLOCK_MAP][..],
            &le(TRY4),
            &le(SEGMENTS4),
        ]
        .concat();
        put(&mut file, INFO4, &info4);
        let try4 = [c(1), c(0), c(0), c(1), le(HANDLERS4).to_vec()].concat();
        put(&mut file, TRY4, &try4);
        // Function-relative continuation offset 0x24 -> PRIMARY + 0x24 (NOT COLD + 0x24!).
        let handlers4 = [
            c(1),
            vec![1 << FH4_HANDLER_CONT_SHIFT],
            le(CATCH_FN).to_vec(),
            c(0x24),
        ]
        .concat();
        put(&mut file, HANDLERS4, &handlers4);
        let segments = [
            c(2),
            le(PRIMARY).to_vec(),
            le(IP_PRIMARY).to_vec(),
            le(COLD).to_vec(),
            le(IP_COLD).to_vec(),
        ]
        .concat();
        put(&mut file, SEGMENTS4, &segments);
        put(&mut file, IP_PRIMARY, &[c(1), c(0x04), c(1)].concat());
        put(&mut file, IP_COLD, &[c(1), c(0x08), c(2)].concat());

        let eh = parse(&file);
        assert!(eh.diagnostics.is_empty(), "{:?}", eh.diagnostics);

        let cold_fn = function(&eh, COLD);
        let LanguageData::Cxx4(cold_info) = &cold_fn.data else {
            panic!("expected FH4 data on chained cold chunk");
        };
        // Cold chunk picks its own segment IP map, with IP base == COLD.
        assert_eq!(cold_info.ip_map_rva, IP_COLD);
        assert_eq!(cold_info.ip_to_state, [(COLD + 0x08, 1)]);
        // Function-relative continuation resolves against PRIMARY!
        assert_eq!(
            cold_fn.resume_targets(),
            [(PRIMARY + 0x24, ResumeKind::CatchContinuation)]
        );
    }

    #[test]
    fn cyclic_and_broken_unwind_chains_are_rejected_without_panicking() {
        use crate::pdata::UNW_FLAG_CHAININFO;

        const F_SELF: u32 = 0x1020;
        const F_MUTUAL_A: u32 = 0x1040;
        const F_MUTUAL_B: u32 = 0x1060;
        const F_IND_SELF: u32 = 0x1080;
        const F_BROKEN: u32 = 0x10a0;

        const UW_SELF: u32 = 0x25c0;
        const UW_MUTUAL_A: u32 = 0x25e0;
        const UW_MUTUAL_B: u32 = 0x2600;
        const RF_IND_SELF: u32 = 0x2620;
        const UW_BROKEN: u32 = 0x2640;

        let mut file = pe_with_pdata(&[
            (F_SELF, F_SELF + 0x10, UW_SELF),
            (F_MUTUAL_A, F_MUTUAL_A + 0x10, UW_MUTUAL_A),
            (F_MUTUAL_B, F_MUTUAL_B + 0x10, UW_MUTUAL_B),
            (F_IND_SELF, F_IND_SELF + 0x10, RF_IND_SELF | RUNTIME_FUNCTION_INDIRECT),
            (F_BROKEN, F_BROKEN + 0x10, UW_BROKEN),
        ]);

        // Self-cycle via CHAININFO.
        put(
            &mut file,
            UW_SELF,
            &unwind_bytes(
                UNW_FLAG_CHAININFO,
                &rf_bytes(F_SELF, F_SELF + 0x10, UW_SELF),
            ),
        );
        // Mutual 2-cycle via CHAININFO.
        put(
            &mut file,
            UW_MUTUAL_A,
            &unwind_bytes(
                UNW_FLAG_CHAININFO,
                &rf_bytes(F_MUTUAL_B, F_MUTUAL_B + 0x10, UW_MUTUAL_B),
            ),
        );
        put(
            &mut file,
            UW_MUTUAL_B,
            &unwind_bytes(
                UNW_FLAG_CHAININFO,
                &rf_bytes(F_MUTUAL_A, F_MUTUAL_A + 0x10, UW_MUTUAL_A),
            ),
        );
        // Indirect self-cycle.
        put(
            &mut file,
            RF_IND_SELF,
            &rf_bytes(
                F_IND_SELF,
                F_IND_SELF + 0x10,
                RF_IND_SELF | RUNTIME_FUNCTION_INDIRECT,
            ),
        );
        // Broken chain pointing to unmapped unwind RVA.
        put(
            &mut file,
            UW_BROKEN,
            &unwind_bytes(
                UNW_FLAG_CHAININFO,
                &rf_bytes(F_BROKEN, F_BROKEN + 0x10, 0x9990),
            ),
        );

        let pe = Pe64::parse(&file).expect("PE parses");
        let dir = ExceptionDirectory::from_pe(&pe, &file);
        let eh = EhMetadata::from_pe(&pe, &file, &dir);
        assert!(eh.functions.is_empty());
        assert!(eh.handlers.is_empty());
        assert!(eh.diagnostics.is_empty());
    }
}
