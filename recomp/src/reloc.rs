//! Relocation map and PE64 base-relocation table parser.
//!
//! This module provides two complementary relocation representations:
//! - [`RelocationMap`]: instruction-level image address references discovered during decoding.
//! - [`BaseRelocTable`]: hardened parser for `IMAGE_DIRECTORY_ENTRY_BASERELOC` (`.reloc`) tables
//!   with checked 4-byte block-size alignment, minimum 8-byte header guards against zero-size
//!   infinite loops, integer-overflow checks, and deterministic sorted output.

use std::fmt;

use iced_x86::{FlowControl, Instruction, Mnemonic, OpKind, Register};

use crate::code_image::{CodeImage, Region};
use crate::pe::Pe64;

/// Data directory index of `IMAGE_DIRECTORY_ENTRY_BASERELOC`.
pub const DIR_BASERELOC: usize = 5;
/// Size in bytes of an `IMAGE_BASE_RELOCATION` block header (`VirtualAddress` + `SizeOfBlock`).
pub const BASE_RELOC_BLOCK_HEADER_LEN: usize = 8;
/// Size in bytes of one 16-bit base-relocation entry (`type << 12 | offset`).
pub const BASE_RELOC_ENTRY_LEN: usize = 2;
/// Required byte alignment of `SizeOfBlock` per the PE/COFF specification (32-bit boundary).
pub const BASE_RELOC_BLOCK_ALIGN: usize = 4;

/// `IMAGE_REL_BASED_ABSOLUTE` (0): padding / no-op entry.
pub const IMAGE_REL_BASED_ABSOLUTE: u8 = 0;
/// `IMAGE_REL_BASED_HIGH` (1).
pub const IMAGE_REL_BASED_HIGH: u8 = 1;
/// `IMAGE_REL_BASED_LOW` (2).
pub const IMAGE_REL_BASED_LOW: u8 = 2;
/// `IMAGE_REL_BASED_HIGHLOW` (3): 32-bit fixup.
pub const IMAGE_REL_BASED_HIGHLOW: u8 = 3;
/// `IMAGE_REL_BASED_HIGHADJ` (4): consumes an extra 16-bit parameter slot in the block.
pub const IMAGE_REL_BASED_HIGHADJ: u8 = 4;
/// `IMAGE_REL_BASED_DIR64` (10): 64-bit fixup.
pub const IMAGE_REL_BASED_DIR64: u8 = 10;

/// Supported PE64 base-relocation fixup types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BaseRelocType {
    /// `IMAGE_REL_BASED_HIGHLOW` (3): 4-byte little-endian fixup.
    HighLow,
    /// `IMAGE_REL_BASED_DIR64` (10): 8-byte little-endian fixup.
    Dir64,
}

impl BaseRelocType {
    /// Width in bytes patched at the relocation site.
    pub const fn width(self) -> u32 {
        match self {
            Self::HighLow => 4,
            Self::Dir64 => 8,
        }
    }

    /// Raw PE `IMAGE_REL_BASED_*` type code.
    pub const fn code(self) -> u8 {
        match self {
            Self::HighLow => IMAGE_REL_BASED_HIGHLOW,
            Self::Dir64 => IMAGE_REL_BASED_DIR64,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HighLow => "highlow",
            Self::Dir64 => "dir64",
        }
    }
}

/// One validated PE64 base-relocation site. Ordered deterministically by `(rva, kind)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BaseRelocSite {
    /// Relative virtual address of the first patched byte.
    pub rva: u32,
    pub kind: BaseRelocType,
}

impl BaseRelocSite {
    /// Checked exclusive end RVA (`rva + width`), or `None` on `u32` overflow.
    pub const fn checked_end_rva(self) -> Option<u32> {
        self.rva.checked_add(self.kind.width())
    }

    /// Checked absolute virtual address (`image_base + rva`), or `None` on `u64` overflow.
    pub const fn checked_va(self, image_base: u64) -> Option<u64> {
        image_base.checked_add(self.rva as u64)
    }

    /// Checked exclusive end virtual address (`image_base + rva + width`), or `None` on `u64` overflow.
    pub const fn checked_end_va(self, image_base: u64) -> Option<u64> {
        match self.checked_va(image_base) {
            Some(va) => va.checked_add(self.kind.width() as u64),
            None => None,
        }
    }
}

/// Fatal structural error while parsing a PE64 base-relocation directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BaseRelocError {
    /// The PE base-relocation directory does not map to contiguous initialized file bytes.
    DirectoryUnmapped { rva: u32, size: u32 },
    /// Fewer than 8 bytes remain for an `IMAGE_BASE_RELOCATION` block header.
    TruncatedBlockHeader { offset: usize, available: usize },
    /// `SizeOfBlock` is smaller than the 8-byte header (including `0`, guarding against infinite loops).
    BlockSizeTooSmall {
        offset: usize,
        page_rva: u32,
        block_size: u32,
    },
    /// `SizeOfBlock` is not aligned to the required 4-byte (`BASE_RELOC_BLOCK_ALIGN`) boundary.
    MisalignedBlockSize {
        offset: usize,
        page_rva: u32,
        block_size: u32,
    },
    /// `SizeOfBlock` extends past the end of the base-relocation directory.
    TruncatedBlock {
        offset: usize,
        needed: usize,
        available: usize,
    },
    /// An `IMAGE_REL_BASED_HIGHADJ` entry lacked its trailing 16-bit parameter slot inside the block.
    TruncatedHighAdjParameter {
        offset: usize,
        page_rva: u32,
        entry_index: usize,
    },
    /// Arithmetic overflow while computing a block offset, site RVA, site end, or virtual address.
    IntegerOverflow {
        context: &'static str,
        offset: usize,
    },
    /// A relocation site (`[rva, rva + width)`) falls outside the image's `SizeOfImage`.
    SiteOutOfImage {
        rva: u32,
        width: u32,
        size_of_image: u32,
    },
    /// Two distinct relocation sites overlap in guest memory.
    OverlappingSites {
        first_rva: u32,
        first_width: u32,
        second_rva: u32,
    },
}

impl fmt::Display for BaseRelocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::DirectoryUnmapped { rva, size } => write!(
                f,
                "base relocation directory [0x{rva:08X}..+0x{size:X}] is not backed by contiguous file bytes"
            ),
            Self::TruncatedBlockHeader { offset, available } => write!(
                f,
                "truncated base relocation block header at offset 0x{offset:X}: needed {BASE_RELOC_BLOCK_HEADER_LEN} bytes, only {available} available"
            ),
            Self::BlockSizeTooSmall {
                offset,
                page_rva,
                block_size,
            } => write!(
                f,
                "base relocation block at offset 0x{offset:X} (page 0x{page_rva:08X}) has SizeOfBlock {block_size} < {BASE_RELOC_BLOCK_HEADER_LEN}"
            ),
            Self::MisalignedBlockSize {
                offset,
                page_rva,
                block_size,
            } => write!(
                f,
                "base relocation block at offset 0x{offset:X} (page 0x{page_rva:08X}) has misaligned SizeOfBlock {block_size} (must be a multiple of {BASE_RELOC_BLOCK_ALIGN})"
            ),
            Self::TruncatedBlock {
                offset,
                needed,
                available,
            } => write!(
                f,
                "truncated base relocation block at offset 0x{offset:X}: needed {needed} bytes, only {available} available"
            ),
            Self::TruncatedHighAdjParameter {
                offset,
                page_rva,
                entry_index,
            } => write!(
                f,
                "truncated IMAGE_REL_BASED_HIGHADJ entry #{entry_index} at block offset 0x{offset:X} (page 0x{page_rva:08X}): missing 16-bit parameter slot"
            ),
            Self::IntegerOverflow { context, offset } => write!(
                f,
                "integer overflow while computing {context} at base relocation offset 0x{offset:X}"
            ),
            Self::SiteOutOfImage {
                rva,
                width,
                size_of_image,
            } => write!(
                f,
                "base relocation site [0x{rva:08X}..+{width}] exceeds SizeOfImage 0x{size_of_image:08X}"
            ),
            Self::OverlappingSites {
                first_rva,
                first_width,
                second_rva,
            } => write!(
                f,
                "base relocation sites [0x{first_rva:08X}..+{first_width}] and 0x{second_rva:08X} overlap"
            ),
        }
    }
}

impl std::error::Error for BaseRelocError {}

/// Non-fatal finding recorded while parsing a PE64 base-relocation directory.
/// Ordered deterministically so output is invariant to block ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BaseRelocDiagnostic {
    /// An unsupported `IMAGE_REL_BASED_*` type code was encountered and skipped.
    UnsupportedType { rva: u32, type_code: u8 },
    /// An identical `(rva, kind)` relocation appeared more than once and was deduplicated.
    DuplicateSite { rva: u32, kind: BaseRelocType },
}

impl fmt::Display for BaseRelocDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::UnsupportedType { rva, type_code } => write!(
                f,
                "unsupported base relocation type {type_code} at RVA 0x{rva:08X}"
            ),
            Self::DuplicateSite { rva, kind } => write!(
                f,
                "duplicate {} base relocation at RVA 0x{rva:08X}",
                kind.as_str()
            ),
        }
    }
}

/// Deterministic, sorted, deduplicated, non-overlapping PE64 base-relocation table.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BaseRelocTable {
    sites: Vec<BaseRelocSite>,
    diagnostics: Vec<BaseRelocDiagnostic>,
    blocks_parsed: usize,
}

impl BaseRelocTable {
    /// Parse raw `IMAGE_DIRECTORY_ENTRY_BASERELOC` bytes with strict bounds, alignment,
    /// overflow, and overlap checks.
    ///
    /// Both [`Self::sites`] and [`Self::diagnostics`] are returned in deterministic canonical
    /// sorted order regardless of block or entry order in `dir`.
    pub fn parse_bytes(dir: &[u8]) -> Result<Self, BaseRelocError> {
        let mut raw_sites = Vec::new();
        let mut diagnostics = Vec::new();
        let mut blocks_parsed = 0usize;
        let mut offset = 0usize;

        while offset < dir.len() {
            let rest = &dir[offset..];
            if rest.iter().all(|&b| b == 0) {
                break;
            }
            if rest.len() < BASE_RELOC_BLOCK_HEADER_LEN {
                return Err(BaseRelocError::TruncatedBlockHeader {
                    offset,
                    available: rest.len(),
                });
            }

            let page_rva = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
            let block_size = u32::from_le_bytes([rest[4], rest[5], rest[6], rest[7]]);

            if (block_size as u64) < BASE_RELOC_BLOCK_HEADER_LEN as u64 {
                return Err(BaseRelocError::BlockSizeTooSmall {
                    offset,
                    page_rva,
                    block_size,
                });
            }
            if !(block_size as u64).is_multiple_of(BASE_RELOC_BLOCK_ALIGN as u64) {
                return Err(BaseRelocError::MisalignedBlockSize {
                    offset,
                    page_rva,
                    block_size,
                });
            }

            let size = usize::try_from(block_size).map_err(|_| BaseRelocError::IntegerOverflow {
                context: "block size",
                offset,
            })?;
            let next_offset =
                offset
                    .checked_add(size)
                    .ok_or(BaseRelocError::IntegerOverflow {
                        context: "block end offset",
                        offset,
                    })?;
            if next_offset > dir.len() {
                return Err(BaseRelocError::TruncatedBlock {
                    offset,
                    needed: size,
                    available: rest.len(),
                });
            }

            let payload = &rest[BASE_RELOC_BLOCK_HEADER_LEN..size];
            let entry_count = payload.len() / BASE_RELOC_ENTRY_LEN;
            let mut idx = 0usize;
            while idx < entry_count {
                let byte_pos = idx * BASE_RELOC_ENTRY_LEN;
                let entry = u16::from_le_bytes([payload[byte_pos], payload[byte_pos + 1]]);
                let type_code = (entry >> 12) as u8;
                let page_offset = u32::from(entry & 0x0FFF);

                if type_code == IMAGE_REL_BASED_ABSOLUTE {
                    idx += 1;
                    continue;
                }

                let rva =
                    page_rva
                        .checked_add(page_offset)
                        .ok_or(BaseRelocError::IntegerOverflow {
                            context: "relocation site RVA",
                            offset,
                        })?;

                match type_code {
                    IMAGE_REL_BASED_HIGHLOW => {
                        let site = BaseRelocSite {
                            rva,
                            kind: BaseRelocType::HighLow,
                        };
                        site.checked_end_rva()
                            .ok_or(BaseRelocError::IntegerOverflow {
                                context: "HIGHLOW site end RVA",
                                offset,
                            })?;
                        raw_sites.push(site);
                        idx += 1;
                    }
                    IMAGE_REL_BASED_DIR64 => {
                        let site = BaseRelocSite {
                            rva,
                            kind: BaseRelocType::Dir64,
                        };
                        site.checked_end_rva()
                            .ok_or(BaseRelocError::IntegerOverflow {
                                context: "DIR64 site end RVA",
                                offset,
                            })?;
                        raw_sites.push(site);
                        idx += 1;
                    }
                    IMAGE_REL_BASED_HIGHADJ => {
                        // PE/COFF spec: HIGHADJ consumes an additional 16-bit parameter slot.
                        if idx + 1 >= entry_count {
                            return Err(BaseRelocError::TruncatedHighAdjParameter {
                                offset,
                                page_rva,
                                entry_index: idx,
                            });
                        }
                        diagnostics.push(BaseRelocDiagnostic::UnsupportedType {
                            rva,
                            type_code: IMAGE_REL_BASED_HIGHADJ,
                        });
                        idx += 2;
                    }
                    other => {
                        diagnostics.push(BaseRelocDiagnostic::UnsupportedType {
                            rva,
                            type_code: other,
                        });
                        idx += 1;
                    }
                }
            }

            blocks_parsed += 1;
            offset = next_offset;
        }

        // Canonical deterministic ordering, deduplication, and overlap validation.
        raw_sites.sort_unstable();
        let mut sites: Vec<BaseRelocSite> = Vec::with_capacity(raw_sites.len());
        for site in raw_sites {
            match sites.last().copied() {
                Some(prev) if prev == site => {
                    diagnostics.push(BaseRelocDiagnostic::DuplicateSite {
                        rva: site.rva,
                        kind: site.kind,
                    });
                }
                Some(prev) => {
                    let prev_end = prev.checked_end_rva().ok_or(BaseRelocError::IntegerOverflow {
                        context: "previous site end RVA",
                        offset: 0,
                    })?;
                    if prev_end > site.rva {
                        return Err(BaseRelocError::OverlappingSites {
                            first_rva: prev.rva,
                            first_width: prev.kind.width(),
                            second_rva: site.rva,
                        });
                    }
                    sites.push(site);
                }
                None => sites.push(site),
            }
        }

        diagnostics.sort_unstable();
        diagnostics.dedup();

        Ok(Self {
            sites,
            diagnostics,
            blocks_parsed,
        })
    }

    /// Parse raw base-relocation directory bytes and verify every site lies within
    /// `[0, size_of_image)` and does not overflow `u64` virtual addresses relative to `image_base`.
    pub fn parse_bytes_bounded(
        dir: &[u8],
        image_base: u64,
        size_of_image: u32,
    ) -> Result<Self, BaseRelocError> {
        let table = Self::parse_bytes(dir)?;
        for &site in &table.sites {
            let end_rva = site
                .checked_end_rva()
                .ok_or(BaseRelocError::IntegerOverflow {
                    context: "site end RVA",
                    offset: 0,
                })?;
            if end_rva > size_of_image {
                return Err(BaseRelocError::SiteOutOfImage {
                    rva: site.rva,
                    width: site.kind.width(),
                    size_of_image,
                });
            }
            site.checked_end_va(image_base)
                .ok_or(BaseRelocError::IntegerOverflow {
                    context: "site virtual address",
                    offset: 0,
                })?;
        }
        Ok(table)
    }

    /// Parse the base-relocation directory of a parsed [`Pe64`]; `file` must be the bytes `pe`
    /// was parsed from. Returns an empty table when the PE has no `DIR_BASERELOC` directory.
    pub fn from_pe(pe: &Pe64, file: &[u8]) -> Result<Self, BaseRelocError> {
        let Some(dir) = pe.optional.data_directory(DIR_BASERELOC) else {
            return Ok(Self::default());
        };
        let bytes = reloc_directory_bytes(pe, file, dir.virtual_address, dir.size)?;
        Self::parse_bytes_bounded(bytes, pe.optional.image_base, pe.optional.size_of_image)
    }

    /// Sorted, deduplicated, non-overlapping base-relocation sites.
    pub fn sites(&self) -> &[BaseRelocSite] {
        &self.sites
    }

    /// Sorted, deduplicated non-fatal diagnostics.
    pub fn diagnostics(&self) -> &[BaseRelocDiagnostic] {
        &self.diagnostics
    }

    /// Number of non-zero base-relocation blocks decoded before termination.
    pub fn blocks_parsed(&self) -> usize {
        self.blocks_parsed
    }

    pub fn len(&self) -> usize {
        self.sites.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sites.is_empty()
    }

    /// Count of `(dir64, highlow)` sites.
    pub fn counts(&self) -> (usize, usize) {
        let mut dir64 = 0usize;
        let mut highlow = 0usize;
        for site in &self.sites {
            match site.kind {
                BaseRelocType::Dir64 => dir64 += 1,
                BaseRelocType::HighLow => highlow += 1,
            }
        }
        (dir64, highlow)
    }

    /// Look up the relocation site starting at `rva` in `O(log n)`.
    pub fn site_at(&self, rva: u32) -> Option<BaseRelocSite> {
        let idx = self.sites.binary_search_by_key(&rva, |s| s.rva).ok()?;
        self.sites.get(idx).copied()
    }

    /// All sites whose starting RVA lies in `[start_rva, end_rva)` in `O(log n)`.
    pub fn sites_in_range(&self, start_rva: u32, end_rva: u32) -> &[BaseRelocSite] {
        if start_rva >= end_rva {
            return &[];
        }
        let lo = self.sites.partition_point(|s| s.rva < start_rva);
        let hi = lo + self.sites[lo..].partition_point(|s| s.rva < end_rva);
        &self.sites[lo..hi]
    }

    /// Checked absolute virtual addresses of all sites in deterministic ascending order.
    pub fn site_vas(&self, image_base: u64) -> Result<Vec<u64>, BaseRelocError> {
        let mut vas = Vec::with_capacity(self.sites.len());
        for &site in &self.sites {
            let va = site
                .checked_va(image_base)
                .ok_or(BaseRelocError::IntegerOverflow {
                    context: "site virtual address",
                    offset: 0,
                })?;
            site.checked_end_va(image_base)
                .ok_or(BaseRelocError::IntegerOverflow {
                    context: "site end virtual address",
                    offset: 0,
                })?;
            vas.push(va);
        }
        Ok(vas)
    }
}

/// Resolve `[rva, rva + size)` to a contiguous initialized slice inside `file`.
fn reloc_directory_bytes<'f>(
    pe: &Pe64,
    file: &'f [u8],
    rva: u32,
    size: u32,
) -> Result<&'f [u8], BaseRelocError> {
    if size == 0 {
        return Ok(&[]);
    }
    let dir_end_rva = rva
        .checked_add(size)
        .ok_or(BaseRelocError::IntegerOverflow {
            context: "base relocation directory end RVA",
            offset: 0,
        })?;
    let len = usize::try_from(size).map_err(|_| BaseRelocError::IntegerOverflow {
        context: "base relocation directory size",
        offset: 0,
    })?;

    let file_offset = pe
        .sections
        .iter()
        .find_map(|s| {
            let memory_size = if s.virtual_size == 0 {
                s.raw_size
            } else {
                s.virtual_size
            };
            let initialized = s.raw_size.min(memory_size);
            let sec_end = u64::from(s.virtual_address) + u64::from(initialized);
            if rva >= s.virtual_address && u64::from(dir_end_rva) <= sec_end {
                let delta = rva - s.virtual_address;
                s.raw_offset.checked_add(delta)
            } else {
                None
            }
        })
        .or_else(|| (dir_end_rva <= pe.optional.size_of_headers).then_some(rva))
        .ok_or(BaseRelocError::DirectoryUnmapped { rva, size })?;

    let start = usize::try_from(file_offset).map_err(|_| BaseRelocError::IntegerOverflow {
        context: "base relocation directory file offset",
        offset: 0,
    })?;
    let end = start
        .checked_add(len)
        .ok_or(BaseRelocError::IntegerOverflow {
            context: "base relocation directory file end",
            offset: 0,
        })?;
    file.get(start..end)
        .ok_or(BaseRelocError::DirectoryUnmapped { rva, size })
}

/// How an instruction references its target address.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelocKind {
    /// Direct `call rel32` target.
    CallTarget,
    /// Direct `jmp` / `jcc` / `loop` / `jrcxz` / `xbegin` target.
    BranchTarget,
    /// RIP-relative memory operand that is read or written as data.
    RipData,
    /// RIP-relative address computed by `lea` (no memory access).
    RipAddress,
    /// Slot read by an indirect `call [rip+x]` (import thunk / IAT entry).
    CallSlot,
    /// Slot read by an indirect `jmp [rip+x]` (import thunk / IAT entry).
    JumpSlot,
    /// Heuristic: 64-bit immediate (`mov r64, imm64`, sign-extended imm32) inside the image.
    AbsoluteImmediate,
    /// Heuristic: absolute memory displacement without base register (`[disp32]`,
    /// `[index*scale+disp32]`, `moffs64`) inside the image.
    AbsoluteDisplacement,
    /// Jump table read by a resolved switch dispatch (`site` is the entry load).
    JumpTable,
    /// Byte index table of a two-level switch (`site` is the `movzx` reading it).
    JumpTableIndex,
    /// One jump-table entry: `site` is the VA of the 32-bit entry itself (data, not an
    /// instruction) and `target` the case it selects.
    JumpTableEntry,
}

impl RelocKind {
    /// Whether this entry is a guess (a constant that merely falls inside the image range).
    pub fn is_possible(self) -> bool {
        matches!(self, Self::AbsoluteImmediate | Self::AbsoluteDisplacement)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::CallTarget => "call_target",
            Self::BranchTarget => "branch_target",
            Self::RipData => "rip_data",
            Self::RipAddress => "rip_address",
            Self::CallSlot => "call_slot",
            Self::JumpSlot => "jump_slot",
            Self::AbsoluteImmediate => "absolute_immediate",
            Self::AbsoluteDisplacement => "absolute_displacement",
            Self::JumpTable => "jump_table",
            Self::JumpTableIndex => "jump_table_index",
            Self::JumpTableEntry => "jump_table_entry",
        }
    }
}

/// One address reference. Ordering is `(site, kind, target)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Relocation {
    /// VA of the referencing instruction (of the table entry for
    /// [`RelocKind::JumpTableEntry`]).
    pub site: u64,
    pub kind: RelocKind,
    /// Absolute referenced VA.
    pub target: u64,
    /// Where `target` lies (code vs data vs outside).
    pub region: Region,
}

/// Sorted, deduplicated relocations.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RelocationMap {
    entries: Vec<Relocation>,
}

impl RelocationMap {
    /// Build a map from arbitrary relocations (sorts and deduplicates).
    pub fn from_entries(mut entries: Vec<Relocation>) -> Self {
        // Stable: discovery hands over long presorted runs, which a merge sort only checks
        // (equal relocations are identical, so the result is the same as an unstable sort).
        entries.sort();
        entries.dedup();
        Self { entries }
    }

    /// Collect relocations from decoded instructions.
    pub fn from_instructions<'i>(
        image: &CodeImage<'_>,
        instructions: impl IntoIterator<Item = &'i Instruction>,
    ) -> Self {
        let mut entries = Vec::new();
        for instr in instructions {
            instruction_relocations(image, instr, &mut entries);
        }
        Self::from_entries(entries)
    }

    pub fn entries(&self) -> &[Relocation] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Relocations of the instruction at `site`.
    pub fn at(&self, site: u64) -> &[Relocation] {
        let lo = self.entries.partition_point(|r| r.site < site);
        let hi = lo + self.entries[lo..].partition_point(|r| r.site == site);
        &self.entries[lo..hi]
    }
}

/// Append the address references made by `instr` to `out` (unsorted).
pub fn instruction_relocations(
    image: &CodeImage<'_>,
    instr: &Instruction,
    out: &mut Vec<Relocation>,
) {
    let site = instr.ip();
    let mut push = |kind: RelocKind, target: u64| {
        out.push(Relocation {
            site,
            kind,
            target,
            region: image.region_of(target),
        });
    };

    for operand in 0..instr.op_count() {
        match instr.op_kind(operand) {
            OpKind::NearBranch64 => {
                let kind = if instr.flow_control() == FlowControl::Call {
                    RelocKind::CallTarget
                } else {
                    RelocKind::BranchTarget
                };
                push(kind, instr.near_branch64());
            }
            OpKind::Memory if instr.is_ip_rel_memory_operand() => {
                let kind = match instr.flow_control() {
                    FlowControl::IndirectCall => RelocKind::CallSlot,
                    FlowControl::IndirectBranch => RelocKind::JumpSlot,
                    _ if instr.mnemonic() == Mnemonic::Lea => RelocKind::RipAddress,
                    _ => RelocKind::RipData,
                };
                push(kind, instr.ip_rel_memory_address());
            }
            OpKind::Memory if instr.memory_base() == Register::None => {
                let target = instr.memory_displacement64();
                if image.contains(target) {
                    push(RelocKind::AbsoluteDisplacement, target);
                }
            }
            OpKind::Immediate64 => {
                let value = instr.immediate64();
                if image.contains(value) {
                    push(RelocKind::AbsoluteImmediate, value);
                }
            }
            OpKind::Immediate32to64 => {
                let value = instr.immediate32to64() as u64;
                if image.contains(value) {
                    push(RelocKind::AbsoluteImmediate, value);
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::code_image::CodeSection;
    use iced_x86::{Decoder, DecoderOptions};

    fn decode_one(bytes: &[u8], ip: u64) -> Instruction {
        let instr = Decoder::with_ip(64, bytes, ip, DecoderOptions::NONE).decode();
        assert!(!instr.is_invalid());
        instr
    }

    #[test]
    fn low_image_jump_table_displacement_and_imm32() {
        // Image loaded below 2 GiB so 32-bit absolutes can reach it.
        let text = [0u8; 0x100];
        let image = CodeImage::new(
            0x40_0000,
            vec![CodeSection::new(0x40_1000, &text, true)],
            vec![],
        );
        let mut out = Vec::new();
        // jmp qword [rax*8+0x401080]
        instruction_relocations(
            &image,
            &decode_one(&[0xff, 0x24, 0xc5, 0x80, 0x10, 0x40, 0x00], 0x40_1000),
            &mut out,
        );
        // push 0x401010 (imm32 sign-extended to 64)
        instruction_relocations(
            &image,
            &decode_one(&[0x68, 0x10, 0x10, 0x40, 0x00], 0x40_1007),
            &mut out,
        );
        // mov eax, [rax+0x401010] has a base register: not a candidate.
        instruction_relocations(
            &image,
            &decode_one(&[0x8b, 0x80, 0x10, 0x10, 0x40, 0x00], 0x40_100c),
            &mut out,
        );
        let map = RelocationMap::from_entries(out);
        assert_eq!(
            map.entries(),
            &[
                Relocation {
                    site: 0x40_1000,
                    kind: RelocKind::AbsoluteDisplacement,
                    target: 0x40_1080,
                    region: Region::Code,
                },
                Relocation {
                    site: 0x40_1007,
                    kind: RelocKind::AbsoluteImmediate,
                    target: 0x40_1010,
                    region: Region::Code,
                },
            ]
        );
        assert!(map.entries().iter().all(|r| r.kind.is_possible()));
    }

    #[test]
    fn map_sorts_dedups_and_looks_up_by_site() {
        let r = |site, target| Relocation {
            site,
            kind: RelocKind::RipData,
            target,
            region: Region::Data,
        };
        let map = RelocationMap::from_entries(vec![r(3, 9), r(1, 5), r(3, 9), r(3, 7)]);
        assert_eq!(map.entries(), &[r(1, 5), r(3, 7), r(3, 9)]);
        assert_eq!(map.at(3), &[r(3, 7), r(3, 9)]);
        assert_eq!(map.at(1), &[r(1, 5)]);
        assert!(map.at(2).is_empty());
        assert!(map.at(4).is_empty());
    }

    fn encode_entry(type_code: u8, offset: u16) -> u16 {
        assert!(offset <= 0x0FFF);
        (u16::from(type_code) << 12) | offset
    }

    /// Encode one 4-byte-aligned PE base-relocation block (automatically appending an
    /// `IMAGE_REL_BASED_ABSOLUTE` pad entry when `raw_entries.len()` is odd).
    fn build_reloc_block(page_rva: u32, raw_entries: &[u16]) -> Vec<u8> {
        let mut entries = raw_entries.to_vec();
        if entries.len() % 2 != 0 {
            entries.push(0); // IMAGE_REL_BASED_ABSOLUTE padding to 4-byte boundary
        }
        let block_size = (BASE_RELOC_BLOCK_HEADER_LEN + entries.len() * BASE_RELOC_ENTRY_LEN) as u32;
        let mut out = Vec::with_capacity(block_size as usize);
        out.extend_from_slice(&page_rva.to_le_bytes());
        out.extend_from_slice(&block_size.to_le_bytes());
        for entry in entries {
            out.extend_from_slice(&entry.to_le_bytes());
        }
        out
    }

    /// Encode a raw block with an explicit (possibly invalid) `block_size`.
    fn build_raw_block(page_rva: u32, block_size: u32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(BASE_RELOC_BLOCK_HEADER_LEN + payload.len());
        out.extend_from_slice(&page_rva.to_le_bytes());
        out.extend_from_slice(&block_size.to_le_bytes());
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn base_reloc_parses_valid_blocks_header_only_and_zero_padding() {
        let mut dir = Vec::new();
        // Header-only block (8 bytes, 0 entries) is valid and 4-byte aligned.
        dir.extend(build_reloc_block(0x1000, &[]));
        // Normal block with DIR64, HIGHLOW, and ABSOLUTE padding.
        dir.extend(build_reloc_block(
            0x2000,
            &[
                encode_entry(IMAGE_REL_BASED_DIR64, 0x020),
                encode_entry(IMAGE_REL_BASED_ABSOLUTE, 0x000),
                encode_entry(IMAGE_REL_BASED_HIGHLOW, 0x010),
            ],
        ));
        // Trailing all-zero padding (e.g. linker section padding) terminates cleanly.
        dir.extend_from_slice(&[0u8; 12]);

        let table = BaseRelocTable::parse_bytes(&dir).expect("valid table parses");
        assert_eq!(table.blocks_parsed(), 2);
        assert_eq!(table.len(), 2);
        assert!(!table.is_empty());
        assert_eq!(table.counts(), (1, 1));
        assert!(table.diagnostics().is_empty());
        assert_eq!(
            table.sites(),
            &[
                BaseRelocSite {
                    rva: 0x2010,
                    kind: BaseRelocType::HighLow,
                },
                BaseRelocSite {
                    rva: 0x2020,
                    kind: BaseRelocType::Dir64,
                },
            ]
        );
        assert_eq!(
            table.site_at(0x2010),
            Some(BaseRelocSite {
                rva: 0x2010,
                kind: BaseRelocType::HighLow,
            })
        );
        assert_eq!(table.site_at(0x2014), None);
        assert_eq!(table.sites_in_range(0x2000, 0x2020).len(), 1);
        assert_eq!(table.sites_in_range(0x2020, 0x2020).len(), 0);
        assert_eq!(
            table.site_vas(0x1_4000_0000).expect("VAs fit u64"),
            vec![0x1_4000_2010, 0x1_4000_2020]
        );
    }

    #[test]
    fn base_reloc_guards_against_zero_size_and_small_block_infinite_loops() {
        // Empty input is valid and empty.
        let empty = BaseRelocTable::parse_bytes(&[]).expect("empty directory");
        assert!(empty.is_empty());
        assert_eq!(empty.blocks_parsed(), 0);

        // Non-zero page_rva with block_size == 0 must error immediately (no infinite loop).
        let zero_size = build_raw_block(0x1000, 0, &[]);
        assert_eq!(
            BaseRelocTable::parse_bytes(&zero_size),
            Err(BaseRelocError::BlockSizeTooSmall {
                offset: 0,
                page_rva: 0x1000,
                block_size: 0,
            })
        );

        // Zero header [0; 8] followed by non-zero data is NOT padding and must reject size == 0.
        let mut hidden_after_zero = vec![0u8; 8];
        hidden_after_zero.extend(build_reloc_block(
            0x1000,
            &[encode_entry(IMAGE_REL_BASED_DIR64, 0x008)],
        ));
        assert_eq!(
            BaseRelocTable::parse_bytes(&hidden_after_zero),
            Err(BaseRelocError::BlockSizeTooSmall {
                offset: 0,
                page_rva: 0,
                block_size: 0,
            })
        );

        // Any block_size in 1..8 is smaller than the 8-byte header.
        for bad_size in 1u32..8 {
            let bytes = build_raw_block(0x1000, bad_size, &[]);
            assert_eq!(
                BaseRelocTable::parse_bytes(&bytes),
                Err(BaseRelocError::BlockSizeTooSmall {
                    offset: 0,
                    page_rva: 0x1000,
                    block_size: bad_size,
                })
            );
        }
    }

    #[test]
    fn base_reloc_rejects_misaligned_block_sizes() {
        // Odd block size (9) and 2-byte-only block sizes (10, 14) violate 4-byte PE alignment.
        for bad_size in [9u32, 10, 11, 13, 14, 15] {
            let payload = vec![0u8; (bad_size as usize) - BASE_RELOC_BLOCK_HEADER_LEN];
            let bytes = build_raw_block(0x1000, bad_size, &payload);
            assert_eq!(
                BaseRelocTable::parse_bytes(&bytes),
                Err(BaseRelocError::MisalignedBlockSize {
                    offset: 0,
                    page_rva: 0x1000,
                    block_size: bad_size,
                })
            );
        }
    }

    #[test]
    fn base_reloc_rejects_truncated_headers_blocks_and_highadj_params() {
        // Non-zero bytes shorter than 8-byte block header.
        for len in 1..BASE_RELOC_BLOCK_HEADER_LEN {
            let mut bytes = vec![0x10u8; len];
            bytes[0] = 0x01;
            assert_eq!(
                BaseRelocTable::parse_bytes(&bytes),
                Err(BaseRelocError::TruncatedBlockHeader {
                    offset: 0,
                    available: len,
                })
            );
        }

        // Valid first block followed by truncated second block payload.
        let mut dir = build_reloc_block(0x1000, &[encode_entry(IMAGE_REL_BASED_DIR64, 0x008)]);
        let second_offset = dir.len();
        dir.extend(build_raw_block(0x2000, 16, &[0u8; 4])); // claims 16 bytes, only 12 present
        assert_eq!(
            BaseRelocTable::parse_bytes(&dir),
            Err(BaseRelocError::TruncatedBlock {
                offset: second_offset,
                needed: 16,
                available: 12,
            })
        );

        // HIGHADJ as the last u16 slot in a block lacks its 16-bit parameter slot.
        let truncated_highadj = build_raw_block(
            0x3000,
            12,
            &[
                0x00, 0x00, // slot 0: ABSOLUTE
                0x10, 0x40, // slot 1: HIGHADJ (type 4, offset 0x010), no slot 2!
            ],
        );
        assert_eq!(
            BaseRelocTable::parse_bytes(&truncated_highadj),
            Err(BaseRelocError::TruncatedHighAdjParameter {
                offset: 0,
                page_rva: 0x3000,
                entry_index: 1,
            })
        );

        // HIGHADJ WITH its parameter slot skips the parameter word instead of decoding it as DIR64.
        let valid_highadj = build_reloc_block(
            0x3000,
            &[
                encode_entry(IMAGE_REL_BASED_HIGHADJ, 0x010),
                encode_entry(IMAGE_REL_BASED_DIR64, 0x020), // parameter slot, MUST be skipped
                encode_entry(IMAGE_REL_BASED_DIR64, 0x040), // real entry after parameter slot
                0,
            ],
        );
        let parsed = BaseRelocTable::parse_bytes(&valid_highadj).expect("valid HIGHADJ skips param");
        assert_eq!(
            parsed.sites(),
            &[BaseRelocSite {
                rva: 0x3040,
                kind: BaseRelocType::Dir64,
            }]
        );
        assert_eq!(
            parsed.diagnostics(),
            &[BaseRelocDiagnostic::UnsupportedType {
                rva: 0x3010,
                type_code: IMAGE_REL_BASED_HIGHADJ,
            }]
        );
    }

    #[test]
    fn base_reloc_guards_against_integer_overflows_and_out_of_image_sites() {
        // Huge 4-byte-aligned SizeOfBlock must never wrap offset + size.
        let huge_block = build_raw_block(0x1000, 0xFFFF_FFFC, &[]);
        assert!(matches!(
            BaseRelocTable::parse_bytes(&huge_block),
            Err(BaseRelocError::TruncatedBlock { .. } | BaseRelocError::IntegerOverflow { .. })
        ));

        // page_rva + page_offset overflows u32.
        let rva_overflow = build_reloc_block(
            0xFFFF_F800,
            &[encode_entry(IMAGE_REL_BASED_DIR64, 0x800)],
        );
        assert_eq!(
            BaseRelocTable::parse_bytes(&rva_overflow),
            Err(BaseRelocError::IntegerOverflow {
                context: "relocation site RVA",
                offset: 0,
            })
        );

        // Site RVA fits in u32, but rva + 8 (DIR64 width) overflows u32.
        let dir64_end_overflow = build_reloc_block(
            0xFFFF_F000,
            &[encode_entry(IMAGE_REL_BASED_DIR64, 0xFF9)],
        );
        assert_eq!(
            BaseRelocTable::parse_bytes(&dir64_end_overflow),
            Err(BaseRelocError::IntegerOverflow {
                context: "DIR64 site end RVA",
                offset: 0,
            })
        );

        // Site RVA fits in u32, but rva + 4 (HIGHLOW width) overflows u32.
        let highlow_end_overflow = build_reloc_block(
            0xFFFF_F000,
            &[encode_entry(IMAGE_REL_BASED_HIGHLOW, 0xFFD)],
        );
        assert_eq!(
            BaseRelocTable::parse_bytes(&highlow_end_overflow),
            Err(BaseRelocError::IntegerOverflow {
                context: "HIGHLOW site end RVA",
                offset: 0,
            })
        );

        // Bounded parsing rejects sites past SizeOfImage and u64 VA overflow.
        let valid_dir = build_reloc_block(0x2000, &[encode_entry(IMAGE_REL_BASED_DIR64, 0xFFC)]);
        assert_eq!(
            BaseRelocTable::parse_bytes_bounded(&valid_dir, 0x1_4000_0000, 0x3000),
            Err(BaseRelocError::SiteOutOfImage {
                rva: 0x2FFC,
                width: 8,
                size_of_image: 0x3000,
            })
        );
        assert_eq!(
            BaseRelocTable::parse_bytes_bounded(&valid_dir, u64::MAX - 0x2FFE, 0x4000),
            Err(BaseRelocError::IntegerOverflow {
                context: "site virtual address",
                offset: 0,
            })
        );
    }

    #[test]
    fn base_reloc_output_is_deterministic_deduped_and_rejects_overlaps() {
        let block_a = build_reloc_block(
            0x1000,
            &[
                encode_entry(IMAGE_REL_BASED_DIR64, 0x040),
                encode_entry(IMAGE_REL_BASED_HIGH, 0x008),
                encode_entry(IMAGE_REL_BASED_HIGHLOW, 0x010),
            ],
        );
        let block_b = build_reloc_block(
            0x2000,
            &[
                encode_entry(IMAGE_REL_BASED_DIR64, 0x000),
                encode_entry(IMAGE_REL_BASED_LOW, 0x020),
            ],
        );
        let block_dup = build_reloc_block(
            0x1000,
            &[
                encode_entry(IMAGE_REL_BASED_DIR64, 0x040), // duplicate of block_a
                encode_entry(IMAGE_REL_BASED_HIGHLOW, 0x010), // duplicate of block_a
            ],
        );

        let mut order_1 = Vec::new();
        order_1.extend(&block_b);
        order_1.extend(&block_dup);
        order_1.extend(&block_a);

        let mut order_2 = Vec::new();
        order_2.extend(&block_a);
        order_2.extend(&block_b);
        order_2.extend(&block_dup);

        let table_1 = BaseRelocTable::parse_bytes(&order_1).expect("order 1 parses");
        let table_2 = BaseRelocTable::parse_bytes(&order_2).expect("order 2 parses");
        assert_eq!(
            table_1, table_2,
            "sites and diagnostics must be bit-for-bit invariant to block permutation"
        );
        assert_eq!(
            table_1.sites(),
            &[
                BaseRelocSite {
                    rva: 0x1010,
                    kind: BaseRelocType::HighLow,
                },
                BaseRelocSite {
                    rva: 0x1040,
                    kind: BaseRelocType::Dir64,
                },
                BaseRelocSite {
                    rva: 0x2000,
                    kind: BaseRelocType::Dir64,
                },
            ]
        );
        assert_eq!(
            table_1.diagnostics(),
            &[
                BaseRelocDiagnostic::UnsupportedType {
                    rva: 0x1008,
                    type_code: IMAGE_REL_BASED_HIGH,
                },
                BaseRelocDiagnostic::UnsupportedType {
                    rva: 0x2020,
                    type_code: IMAGE_REL_BASED_LOW,
                },
                BaseRelocDiagnostic::DuplicateSite {
                    rva: 0x1010,
                    kind: BaseRelocType::HighLow,
                },
                BaseRelocDiagnostic::DuplicateSite {
                    rva: 0x1040,
                    kind: BaseRelocType::Dir64,
                },
            ]
        );

        // Partial overlap: DIR64 [0x1010..0x1018) vs HIGHLOW [0x1014..0x1018).
        let overlap_partial = build_reloc_block(
            0x1000,
            &[
                encode_entry(IMAGE_REL_BASED_DIR64, 0x010),
                encode_entry(IMAGE_REL_BASED_HIGHLOW, 0x014),
            ],
        );
        assert_eq!(
            BaseRelocTable::parse_bytes(&overlap_partial),
            Err(BaseRelocError::OverlappingSites {
                first_rva: 0x1010,
                first_width: 8,
                second_rva: 0x1014,
            })
        );

        // Same RVA with conflicting widths (HIGHLOW + DIR64) also overlaps.
        let overlap_same_rva = build_reloc_block(
            0x1000,
            &[
                encode_entry(IMAGE_REL_BASED_HIGHLOW, 0x010),
                encode_entry(IMAGE_REL_BASED_DIR64, 0x010),
            ],
        );
        assert_eq!(
            BaseRelocTable::parse_bytes(&overlap_same_rva),
            Err(BaseRelocError::OverlappingSites {
                first_rva: 0x1010,
                first_width: 4,
                second_rva: 0x1010,
            })
        );
    }

    #[test]
    fn base_reloc_from_pe_validates_directory_mapping_and_bounds() {
        use crate::pe::test_util::build_synthetic_pe64;

        const DIR_ENTRY_OFF: usize = 0x108 + DIR_BASERELOC * 8;
        const RELOC_RVA: u32 = 0x2600;
        const RELOC_FILE_OFF: usize = 0x400 + (RELOC_RVA as usize - 0x2000);

        // Default synthetic PE has no base-relocation directory -> empty table.
        let base_file = build_synthetic_pe64(false);
        let base_pe = Pe64::parse(&base_file).expect("synthetic PE parses");
        let empty = BaseRelocTable::from_pe(&base_pe, &base_file).expect("no dir is empty");
        assert!(empty.is_empty());

        // Populate a valid multi-block directory inside .rdata.
        let mut file = build_synthetic_pe64(false);
        file[0x1b8..0x1bc].copy_from_slice(&0x800u32.to_le_bytes()); // .rdata VirtualSize
        let mut reloc_bytes = Vec::new();
        reloc_bytes.extend(build_reloc_block(
            0x2000,
            &[encode_entry(IMAGE_REL_BASED_DIR64, 0x010)],
        ));
        reloc_bytes.extend(build_reloc_block(
            0x1000,
            &[encode_entry(IMAGE_REL_BASED_HIGHLOW, 0x020)],
        ));
        file[RELOC_FILE_OFF..RELOC_FILE_OFF + reloc_bytes.len()].copy_from_slice(&reloc_bytes);
        file[DIR_ENTRY_OFF..DIR_ENTRY_OFF + 4].copy_from_slice(&RELOC_RVA.to_le_bytes());
        file[DIR_ENTRY_OFF + 4..DIR_ENTRY_OFF + 8]
            .copy_from_slice(&(reloc_bytes.len() as u32).to_le_bytes());

        let pe = Pe64::parse(&file).expect("synthetic PE with .reloc parses");
        let table = BaseRelocTable::from_pe(&pe, &file).expect("valid PE base relocs parse");
        assert_eq!(
            table.sites(),
            &[
                BaseRelocSite {
                    rva: 0x1020,
                    kind: BaseRelocType::HighLow,
                },
                BaseRelocSite {
                    rva: 0x2010,
                    kind: BaseRelocType::Dir64,
                },
            ]
        );

        // Directory extending past section raw data is rejected as DirectoryUnmapped.
        let mut bad_dir = file.clone();
        bad_dir[DIR_ENTRY_OFF + 4..DIR_ENTRY_OFF + 8].copy_from_slice(&0x400u32.to_le_bytes());
        let bad_pe = Pe64::parse(&bad_dir).expect("PE header still parses");
        assert_eq!(
            BaseRelocTable::from_pe(&bad_pe, &bad_dir),
            Err(BaseRelocError::DirectoryUnmapped {
                rva: RELOC_RVA,
                size: 0x400,
            })
        );

        // Directory RVA + size overflowing u32 is rejected as IntegerOverflow.
        let mut overflow_dir = file;
        overflow_dir[DIR_ENTRY_OFF..DIR_ENTRY_OFF + 4].copy_from_slice(&0xFFFF_F000u32.to_le_bytes());
        overflow_dir[DIR_ENTRY_OFF + 4..DIR_ENTRY_OFF + 8].copy_from_slice(&0x2000u32.to_le_bytes());
        let overflow_pe = Pe64::parse(&overflow_dir).expect("PE header still parses");
        assert_eq!(
            BaseRelocTable::from_pe(&overflow_pe, &overflow_dir),
            Err(BaseRelocError::IntegerOverflow {
                context: "base relocation directory end RVA",
                offset: 0,
            })
        );
    }

    #[test]
    fn base_reloc_parser_never_panics_on_arbitrary_bytes() {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        for len in 0..=256 {
            let bytes: Vec<u8> = (0..len)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 56) as u8
                })
                .collect();
            if let Ok(table) = BaseRelocTable::parse_bytes(&bytes) {
                // Any accepted table must be strictly ascending and non-overlapping.
                for pair in table.sites().windows(2) {
                    let prev_end = pair[0].checked_end_rva().expect("checked end");
                    assert!(prev_end <= pair[1].rva);
                }
            }
        }
    }
}
