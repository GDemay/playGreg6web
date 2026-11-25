//! Pure-Rust PE32+ (64-bit Portable Executable) parser.
//!
//! Extracts headers, sections, RVA-to-file-offset mappings, standard imports,
//! delay-load imports, exports, and TLS callbacks without external crates or
//! retaining raw binary bytes.

use std::fmt;

pub const DOS_MAGIC: [u8; 2] = *b"MZ";
pub const PE_SIGNATURE: [u8; 4] = *b"PE\0\0";
pub const PE32_PLUS_MAGIC: u16 = 0x020B;

pub const DIR_EXPORT: usize = 0;
pub const DIR_IMPORT: usize = 1;
pub const DIR_TLS: usize = 9;
pub const DIR_DELAY_IMPORT: usize = 13;

pub const IMAGE_SCN_MEM_EXECUTE: u32 = 0x2000_0000;
pub const IMAGE_SCN_MEM_READ: u32 = 0x4000_0000;
pub const IMAGE_SCN_MEM_WRITE: u32 = 0x8000_0000;

const DOS_HEADER_MIN_LEN: usize = 0x40;
const COFF_HEADER_LEN: usize = 20;
const OPT_HEADER_MIN_LEN: usize = 112;
const SECTION_HEADER_LEN: usize = 40;
const IMPORT_DESCRIPTOR_LEN: usize = 20;
const DELAY_IMPORT_DESCRIPTOR_LEN: usize = 32;
const EXPORT_DIRECTORY_LEN: usize = 40;
const TLS_DIRECTORY64_LEN: usize = 40;

const IMAGE_ORDINAL_FLAG64: u64 = 1u64 << 63;
const DELAY_ATTR_RVA: u32 = 0x1;

const MAX_CSTR_LEN: usize = 4096;
const MAX_IMPORT_MODULES: usize = 4096;
const MAX_THUNKS_PER_MODULE: usize = 65_536;
const MAX_EXPORTS: usize = 131_072;
const MAX_TLS_CALLBACKS: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeError {
    Truncated {
        context: &'static str,
        needed: usize,
        available: usize,
    },
    InvalidDosMagic([u8; 2]),
    InvalidPeSignature([u8; 4]),
    UnsupportedOptionalMagic(u16),
    InvalidOffset {
        context: &'static str,
        offset: usize,
    },
    UnmappedRva {
        context: &'static str,
        rva: u32,
    },
    InvalidVirtualAddress {
        context: &'static str,
        va: u64,
        image_base: u64,
    },
    UnterminatedString {
        context: &'static str,
        rva: u32,
    },
    TableTooLarge {
        context: &'static str,
        count: usize,
    },
    IntegerOverflow {
        context: &'static str,
    },
}

impl fmt::Display for PeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PeError::Truncated {
                context,
                needed,
                available,
            } => write!(
                f,
                "truncated PE data while reading {context}: needed {needed} bytes, only {available} available"
            ),
            PeError::InvalidDosMagic(m) => {
                write!(f, "invalid DOS magic: expected 'MZ', got 0x{:02X}{:02X}", m[0], m[1])
            }
            PeError::InvalidPeSignature(sig) => write!(
                f,
                "invalid PE signature: expected 'PE\\0\\0', got {:02X?}",
                sig
            ),
            PeError::UnsupportedOptionalMagic(magic) => write!(
                f,
                "unsupported PE optional header magic 0x{magic:04X} (expected PE32+ 0x{PE32_PLUS_MAGIC:04X})"
            ),
            PeError::InvalidOffset { context, offset } => {
                write!(f, "invalid file offset 0x{offset:X} while reading {context}")
            }
            PeError::UnmappedRva { context, rva } => {
                write!(f, "RVA 0x{rva:08X} does not map to any file bytes ({context})")
            }
            PeError::InvalidVirtualAddress {
                context,
                va,
                image_base,
            } => write!(
                f,
                "virtual address 0x{va:016X} cannot be converted to RVA for image base 0x{image_base:016X} ({context})"
            ),
            PeError::UnterminatedString { context, rva } => {
                write!(f, "unterminated C string at RVA 0x{rva:08X} ({context})")
            }
            PeError::TableTooLarge { context, count } => {
                write!(f, "{context} exceeds parser safety limit ({count} entries)")
            }
            PeError::IntegerOverflow { context } => {
                write!(f, "integer overflow while computing {context}")
            }
        }
    }
}

impl std::error::Error for PeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoffHeader {
    pub machine: u16,
    pub number_of_sections: u16,
    pub time_date_stamp: u32,
    pub pointer_to_symbol_table: u32,
    pub number_of_symbols: u32,
    pub size_of_optional_header: u16,
    pub characteristics: u16,
}

impl CoffHeader {
    pub fn machine_name(&self) -> &'static str {
        match self.machine {
            0x8664 => "AMD64",
            0x014C => "I386",
            0xAA64 => "ARM64",
            0x01C4 => "ARMNT",
            0x0000 => "UNKNOWN",
            _ => "OTHER",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataDirectory {
    pub virtual_address: u32,
    pub size: u32,
}

impl DataDirectory {
    pub fn is_present(&self) -> bool {
        self.virtual_address != 0 && self.size != 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptionalHeader64 {
    pub magic: u16,
    pub major_linker_version: u8,
    pub minor_linker_version: u8,
    pub size_of_code: u32,
    pub size_of_initialized_data: u32,
    pub size_of_uninitialized_data: u32,
    pub entry_point_rva: u32,
    pub base_of_code: u32,
    pub image_base: u64,
    pub section_alignment: u32,
    pub file_alignment: u32,
    pub size_of_image: u32,
    pub size_of_headers: u32,
    pub checksum: u32,
    pub subsystem: u16,
    pub dll_characteristics: u16,
    pub number_of_rva_and_sizes: u32,
    pub data_directories: Vec<DataDirectory>,
}

impl OptionalHeader64 {
    pub fn subsystem_name(&self) -> &'static str {
        match self.subsystem {
            0 => "UNKNOWN",
            1 => "NATIVE",
            2 => "WINDOWS_GUI",
            3 => "WINDOWS_CUI",
            5 => "OS2_CUI",
            7 => "POSIX_CUI",
            9 => "WINDOWS_CE_GUI",
            10 => "EFI_APPLICATION",
            11 => "EFI_BOOT_SERVICE_DRIVER",
            12 => "EFI_RUNTIME_DRIVER",
            13 => "EFI_ROM",
            14 => "XBOX",
            16 => "WINDOWS_BOOT_APPLICATION",
            _ => "OTHER",
        }
    }

    pub fn data_directory(&self, index: usize) -> Option<DataDirectory> {
        self.data_directories
            .get(index)
            .copied()
            .filter(DataDirectory::is_present)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionHeader {
    pub name: String,
    pub virtual_size: u32,
    pub virtual_address: u32,
    pub raw_size: u32,
    pub raw_offset: u32,
    pub characteristics: u32,
}

impl SectionHeader {
    /// Effective virtual span in memory: `raw_size` when `virtual_size == 0`, else `virtual_size`.
    pub const fn virtual_extent(&self) -> u32 {
        if self.virtual_size == 0 {
            self.raw_size
        } else {
            self.virtual_size
        }
    }

    /// Number of file-backed bytes visible in the section's virtual address range.
    pub const fn mapped_raw_size(&self) -> u32 {
        let virt = self.virtual_extent();
        if virt < self.raw_size {
            virt
        } else {
            self.raw_size
        }
    }

    /// True when the section occupies zero virtual bytes.
    pub const fn is_empty(&self) -> bool {
        self.virtual_extent() == 0
    }

    /// True when the section has virtual extent but zero file-backed raw bytes (pure BSS).
    pub const fn is_bss(&self) -> bool {
        self.virtual_extent() > 0 && self.raw_size == 0
    }

    /// Checked exclusive virtual end RVA (`virtual_address + virtual_extent`), or `None` on `u32` overflow.
    pub const fn checked_virtual_end(&self) -> Option<u32> {
        self.virtual_address.checked_add(self.virtual_extent())
    }

    /// Checked exclusive raw file end offset (`raw_offset + raw_size`), or `None` on `u32` overflow.
    pub const fn checked_raw_end(&self) -> Option<u32> {
        self.raw_offset.checked_add(self.raw_size)
    }

    /// Checked virtual address range `[start, end)` relative to `image_base`, or `None` on `u64` overflow.
    pub const fn checked_va_range(&self, image_base: u64) -> Option<(u64, u64)> {
        let Some(start) = image_base.checked_add(self.virtual_address as u64) else {
            return None;
        };
        let Some(end) = start.checked_add(self.virtual_extent() as u64) else {
            return None;
        };
        Some((start, end))
    }

    pub fn permissions_str(&self) -> String {
        let r = if self.characteristics & IMAGE_SCN_MEM_READ != 0 {
            'r'
        } else {
            '-'
        };
        let w = if self.characteristics & IMAGE_SCN_MEM_WRITE != 0 {
            'w'
        } else {
            '-'
        };
        let x = if self.characteristics & IMAGE_SCN_MEM_EXECUTE != 0 {
            'x'
        } else {
            '-'
        };
        format!("{r}{w}{x}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportedSymbol {
    ByName { hint: u16, name: String },
    ByOrdinal { ordinal: u16 },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportModule {
    pub dll_name: String,
    /// RVA of the import address table (`FirstThunk`): symbol `i` is bound through the 8-byte
    /// slot at `iat_rva + 8 * i`. 0 when the descriptor has no IAT.
    pub iat_rva: u32,
    pub symbols: Vec<ImportedSymbol>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelayImportModule {
    pub dll_name: String,
    pub attributes: u32,
    pub uses_rva: bool,
    /// RVA of the delay-load import address table (always an RVA, also for VA-mode
    /// descriptors): symbol `i` is bound through the slot at `iat_rva + 8 * i`. 0 when absent.
    pub iat_rva: u32,
    pub symbols: Vec<ImportedSymbol>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportSymbol {
    pub name: Option<String>,
    pub ordinal: u32,
    pub rva: u32,
    pub forwarder: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportTable {
    pub module_name: Option<String>,
    pub ordinal_base: u32,
    pub symbols: Vec<ExportSymbol>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsCallback {
    pub va: u64,
    pub rva: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsDirectory {
    pub start_address_of_raw_data: u64,
    pub end_address_of_raw_data: u64,
    pub address_of_index: u64,
    pub address_of_callbacks: u64,
    pub size_of_zero_fill: u32,
    pub characteristics: u32,
    pub callbacks: Vec<TlsCallback>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pe64 {
    pub file_size: usize,
    pub e_lfanew: u32,
    pub coff: CoffHeader,
    pub optional: OptionalHeader64,
    pub sections: Vec<SectionHeader>,
    pub imports: Vec<ImportModule>,
    pub delay_imports: Vec<DelayImportModule>,
    pub exports: Option<ExportTable>,
    pub tls: Option<TlsDirectory>,
}

impl Pe64 {
    pub fn parse(data: &[u8]) -> Result<Self, PeError> {
        if data.len() < DOS_HEADER_MIN_LEN {
            return Err(PeError::Truncated {
                context: "DOS header",
                needed: DOS_HEADER_MIN_LEN,
                available: data.len(),
            });
        }

        let dos_magic = [data[0], data[1]];
        if dos_magic != DOS_MAGIC {
            return Err(PeError::InvalidDosMagic(dos_magic));
        }

        let e_lfanew = read_u32(data, 0x3C, "DOS e_lfanew")?;
        let pe_offset = usize::try_from(e_lfanew).map_err(|_| PeError::IntegerOverflow {
            context: "DOS e_lfanew",
        })?;
        let coff_offset_u32 = e_lfanew.checked_add(4).ok_or(PeError::IntegerOverflow {
            context: "PE signature offset",
        })?;
        let sig_bytes = read_slice(data, pe_offset, 4, "PE signature")?;
        let sig = [sig_bytes[0], sig_bytes[1], sig_bytes[2], sig_bytes[3]];
        if sig != PE_SIGNATURE {
            return Err(PeError::InvalidPeSignature(sig));
        }

        let coff_offset = pe_offset
            .checked_add(4)
            .ok_or(PeError::IntegerOverflow {
                context: "COFF header offset",
            })?;
        let coff = parse_coff_header(data, coff_offset)?;

        let opt_offset_u32 = coff_offset_u32
            .checked_add(COFF_HEADER_LEN as u32)
            .ok_or(PeError::IntegerOverflow {
                context: "optional header offset",
            })?;
        let opt_offset = coff_offset
            .checked_add(COFF_HEADER_LEN)
            .ok_or(PeError::IntegerOverflow {
                context: "optional header offset",
            })?;
        let optional = parse_optional_header(data, opt_offset, coff.size_of_optional_header)?;

        let sections_offset_u32 = opt_offset_u32
            .checked_add(u32::from(coff.size_of_optional_header))
            .ok_or(PeError::IntegerOverflow {
                context: "section table offset",
            })?;
        let sections_offset = opt_offset
            .checked_add(usize::from(coff.size_of_optional_header))
            .ok_or(PeError::IntegerOverflow {
                context: "section table offset",
            })?;
        let sections = parse_sections(
            data,
            sections_offset,
            sections_offset_u32,
            &coff,
            optional.image_base,
        )?;

        let imports = if let Some(dir) = optional.data_directory(DIR_IMPORT) {
            parse_imports(data, &sections, optional.size_of_headers, dir)?
        } else {
            Vec::new()
        };

        let delay_imports = if let Some(dir) = optional.data_directory(DIR_DELAY_IMPORT) {
            parse_delay_imports(
                data,
                &sections,
                optional.size_of_headers,
                optional.image_base,
                dir,
            )?
        } else {
            Vec::new()
        };

        let exports = if let Some(dir) = optional.data_directory(DIR_EXPORT) {
            Some(parse_exports(data, &sections, optional.size_of_headers, dir)?)
        } else {
            None
        };

        let tls = if let Some(dir) = optional.data_directory(DIR_TLS) {
            Some(parse_tls(
                data,
                &sections,
                optional.size_of_headers,
                optional.image_base,
                dir,
            )?)
        } else {
            None
        };

        Ok(Self {
            file_size: data.len(),
            e_lfanew,
            coff,
            optional,
            sections,
            imports,
            delay_imports,
            exports,
            tls,
        })
    }

    /// Maps an RVA (Relative Virtual Address) to a file byte offset within the PE image.
    pub fn rva_to_file_offset(&self, rva: u32) -> Option<usize> {
        rva_to_file_offset_raw(
            &self.sections,
            self.optional.size_of_headers,
            self.file_size,
            rva,
        )
    }

    /// Maps a contiguous RVA byte range `[rva, rva + len)` to a file byte offset, ensuring
    /// zero integer wraparound and that the entire range fits within one mapped section (or header)
    /// and within `self.file_size`.
    pub fn rva_range_to_file_offset(&self, rva: u32, len: usize) -> Option<usize> {
        let len_u32 = u32::try_from(len).ok()?;
        rva.checked_add(len_u32)?;
        let (offset, avail) = rva_to_file_span_raw(
            &self.sections,
            self.optional.size_of_headers,
            self.file_size,
            rva,
        )?;
        if len <= avail {
            Some(offset)
        } else {
            None
        }
    }

    /// Converts an RVA to a 64-bit virtual address using checked `u64` addition.
    pub const fn rva_to_va(&self, rva: u32) -> Option<u64> {
        self.optional.image_base.checked_add(rva as u64)
    }

    /// Converts a 64-bit virtual address to a 32-bit RVA using checked subtraction and narrowing.
    pub fn va_to_rva(&self, va: u64) -> Option<u32> {
        va.checked_sub(self.optional.image_base)
            .and_then(|delta| u32::try_from(delta).ok())
    }

    /// Returns sections in canonical deterministic order by `(virtual_address, raw_offset, name)`.
    pub fn sections_by_rva(&self) -> Vec<&SectionHeader> {
        let mut out: Vec<&SectionHeader> = self.sections.iter().collect();
        out.sort_by(|a, b| {
            (a.virtual_address, a.raw_offset, a.name.as_str()).cmp(&(
                b.virtual_address,
                b.raw_offset,
                b.name.as_str(),
            ))
        });
        out
    }

    /// Returns standard import modules in canonical deterministic order by case-folded DLL name,
    /// exact DLL name, and IAT RVA.
    pub fn imports_sorted(&self) -> Vec<&ImportModule> {
        let mut out: Vec<&ImportModule> = self.imports.iter().collect();
        out.sort_by(|a, b| {
            (
                a.dll_name.to_ascii_lowercase(),
                a.dll_name.as_str(),
                a.iat_rva,
            )
                .cmp(&(
                    b.dll_name.to_ascii_lowercase(),
                    b.dll_name.as_str(),
                    b.iat_rva,
                ))
        });
        out
    }

    /// Returns delay-load import modules in canonical deterministic order by case-folded DLL name,
    /// exact DLL name, and IAT RVA.
    pub fn delay_imports_sorted(&self) -> Vec<&DelayImportModule> {
        let mut out: Vec<&DelayImportModule> = self.delay_imports.iter().collect();
        out.sort_by(|a, b| {
            (
                a.dll_name.to_ascii_lowercase(),
                a.dll_name.as_str(),
                a.iat_rva,
            )
                .cmp(&(
                    b.dll_name.to_ascii_lowercase(),
                    b.dll_name.as_str(),
                    b.iat_rva,
                ))
        });
        out
    }
}

fn rva_to_file_offset_raw(
    sections: &[SectionHeader],
    size_of_headers: u32,
    file_size: usize,
    rva: u32,
) -> Option<usize> {
    rva_to_file_span_raw(sections, size_of_headers, file_size, rva).map(|(offset, _avail)| offset)
}

/// Maps `rva` to `(file_offset, contiguous_available_bytes)` within the owning section's
/// initialized raw data (or the PE headers), strictly clamped to `file_size`.
fn rva_to_file_span_raw(
    sections: &[SectionHeader],
    size_of_headers: u32,
    file_size: usize,
    rva: u32,
) -> Option<(usize, usize)> {
    for section in sections {
        let virt_extent = section.virtual_extent();
        if virt_extent == 0 {
            continue;
        }
        // Guard against overflowing section definitions (even on manually constructed `Pe64`).
        if section.checked_virtual_end().is_none() || section.checked_raw_end().is_none() {
            continue;
        }
        let Some(delta) = rva.checked_sub(section.virtual_address) else {
            continue;
        };
        if delta >= virt_extent {
            continue;
        }
        // RVA falls inside this section's virtual range: first matching section owns it
        // deterministically, and uninitialized BSS / out-of-bounds raw regions never fall
        // through to header bytes or later sections.
        let mapped_raw = section.mapped_raw_size();
        if delta >= mapped_raw {
            return None;
        }
        let raw_offset = usize::try_from(section.raw_offset).ok()?;
        let delta_usize = usize::try_from(delta).ok()?;
        let file_offset = raw_offset.checked_add(delta_usize)?;
        if file_offset >= file_size {
            return None;
        }
        let section_remaining = usize::try_from(mapped_raw - delta).ok()?;
        let file_remaining = file_size - file_offset;
        return Some((file_offset, section_remaining.min(file_remaining)));
    }

    if rva < size_of_headers {
        let rva_usize = usize::try_from(rva).ok()?;
        if rva_usize < file_size {
            let hdr_end = usize::try_from(size_of_headers)
                .unwrap_or(file_size)
                .min(file_size);
            let avail = hdr_end.checked_sub(rva_usize)?;
            if avail > 0 {
                return Some((rva_usize, avail));
            }
        }
    }

    None
}

fn read_rva_slice<'a>(
    data: &'a [u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    rva: u32,
    len: usize,
    context: &'static str,
) -> Result<&'a [u8], PeError> {
    let len_u32 = u32::try_from(len).map_err(|_| PeError::IntegerOverflow { context })?;
    rva.checked_add(len_u32)
        .ok_or(PeError::IntegerOverflow { context })?;
    let (offset, avail) = rva_to_file_span_raw(sections, size_of_headers, data.len(), rva)
        .ok_or(PeError::UnmappedRva { context, rva })?;
    if len > avail {
        return Err(PeError::Truncated {
            context,
            needed: len,
            available: avail,
        });
    }
    read_slice(data, offset, len, context)
}

fn parse_coff_header(data: &[u8], offset: usize) -> Result<CoffHeader, PeError> {
    let bytes = read_slice(data, offset, COFF_HEADER_LEN, "COFF header")?;
    Ok(CoffHeader {
        machine: le_u16(&bytes[0..2]),
        number_of_sections: le_u16(&bytes[2..4]),
        time_date_stamp: le_u32(&bytes[4..8]),
        pointer_to_symbol_table: le_u32(&bytes[8..12]),
        number_of_symbols: le_u32(&bytes[12..16]),
        size_of_optional_header: le_u16(&bytes[16..18]),
        characteristics: le_u16(&bytes[18..20]),
    })
}

fn parse_optional_header(
    data: &[u8],
    offset: usize,
    size_of_optional_header: u16,
) -> Result<OptionalHeader64, PeError> {
    let opt_size = usize::from(size_of_optional_header);
    if opt_size < 2 {
        return Err(PeError::Truncated {
            context: "optional header magic",
            needed: 2,
            available: opt_size,
        });
    }

    let opt = read_slice(data, offset, opt_size, "optional header")?;
    let magic = le_u16(&opt[0..2]);
    if magic != PE32_PLUS_MAGIC {
        return Err(PeError::UnsupportedOptionalMagic(magic));
    }

    if opt_size < OPT_HEADER_MIN_LEN {
        return Err(PeError::Truncated {
            context: "PE32+ optional header",
            needed: OPT_HEADER_MIN_LEN,
            available: opt_size,
        });
    }

    let entry_point_rva = le_u32(&opt[16..20]);
    let image_base = le_u64(&opt[24..32]);
    let size_of_image = le_u32(&opt[56..60]);
    image_base
        .checked_add(u64::from(size_of_image))
        .ok_or(PeError::IntegerOverflow {
            context: "optional header image_base + size_of_image",
        })?;
    image_base
        .checked_add(u64::from(entry_point_rva))
        .ok_or(PeError::IntegerOverflow {
            context: "optional header entry point VA",
        })?;

    let number_of_rva_and_sizes = le_u32(&opt[108..112]);
    let dir_bytes = opt_size
        .checked_sub(OPT_HEADER_MIN_LEN)
        .ok_or(PeError::IntegerOverflow {
            context: "optional header data directories size",
        })?;
    let max_dirs_in_header = dir_bytes / 8;
    let dir_count = usize::try_from(number_of_rva_and_sizes)
        .unwrap_or(usize::MAX)
        .min(max_dirs_in_header);

    let mut data_directories = Vec::with_capacity(dir_count);
    for chunk in opt[OPT_HEADER_MIN_LEN..].chunks_exact(8).take(dir_count) {
        data_directories.push(DataDirectory {
            virtual_address: le_u32(&chunk[0..4]),
            size: le_u32(&chunk[4..8]),
        });
    }

    Ok(OptionalHeader64 {
        magic,
        major_linker_version: opt[2],
        minor_linker_version: opt[3],
        size_of_code: le_u32(&opt[4..8]),
        size_of_initialized_data: le_u32(&opt[8..12]),
        size_of_uninitialized_data: le_u32(&opt[12..16]),
        entry_point_rva,
        base_of_code: le_u32(&opt[20..24]),
        image_base,
        section_alignment: le_u32(&opt[32..36]),
        file_alignment: le_u32(&opt[36..40]),
        size_of_image,
        size_of_headers: le_u32(&opt[60..64]),
        checksum: le_u32(&opt[64..68]),
        subsystem: le_u16(&opt[68..70]),
        dll_characteristics: le_u16(&opt[70..72]),
        number_of_rva_and_sizes,
        data_directories,
    })
}

fn parse_sections(
    data: &[u8],
    offset: usize,
    offset_u32: u32,
    coff: &CoffHeader,
    image_base: u64,
) -> Result<Vec<SectionHeader>, PeError> {
    let count = usize::from(coff.number_of_sections);
    let table_bytes_len = count
        .checked_mul(SECTION_HEADER_LEN)
        .ok_or(PeError::IntegerOverflow {
            context: "section table size",
        })?;
    let table_bytes_len_u32 =
        u32::try_from(table_bytes_len).map_err(|_| PeError::IntegerOverflow {
            context: "section table size",
        })?;
    offset_u32
        .checked_add(table_bytes_len_u32)
        .ok_or(PeError::IntegerOverflow {
            context: "section table end offset",
        })?;
    let table_bytes = read_slice(data, offset, table_bytes_len, "section table")?;

    let mut sections = Vec::with_capacity(count);
    for entry in table_bytes.chunks_exact(SECTION_HEADER_LEN) {
        let raw_name = &entry[0..8];
        let trimmed_len = raw_name
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(raw_name.len());
        let mut name = String::from_utf8_lossy(&raw_name[..trimmed_len]).into_owned();

        // Optional COFF string table lookup for "/<offset>" long section names.
        if let Some(str_off_str) = name.strip_prefix('/')
            && let Ok(str_off) = str_off_str.parse::<usize>()
            && let Some(resolved) = resolve_coff_string(data, coff, str_off)
        {
            name = resolved;
        }

        let section = SectionHeader {
            name,
            virtual_size: le_u32(&entry[8..12]),
            virtual_address: le_u32(&entry[12..16]),
            raw_size: le_u32(&entry[16..20]),
            raw_offset: le_u32(&entry[20..24]),
            characteristics: le_u32(&entry[36..40]),
        };

        if section.checked_virtual_end().is_none() {
            return Err(PeError::IntegerOverflow {
                context: "section virtual range",
            });
        }
        if section.checked_raw_end().is_none() {
            return Err(PeError::IntegerOverflow {
                context: "section raw range",
            });
        }
        if section.checked_va_range(image_base).is_none() {
            return Err(PeError::IntegerOverflow {
                context: "section virtual address range",
            });
        }

        sections.push(section);
    }

    Ok(sections)
}

fn resolve_coff_string(data: &[u8], coff: &CoffHeader, string_offset: usize) -> Option<String> {
    if coff.pointer_to_symbol_table == 0 || coff.number_of_symbols == 0 || string_offset < 4 {
        return None;
    }
    let num_symbols = usize::try_from(coff.number_of_symbols).ok()?;
    let ptr_symtab = usize::try_from(coff.pointer_to_symbol_table).ok()?;
    let sym_table_bytes = num_symbols.checked_mul(18)?;
    let str_table_start = ptr_symtab.checked_add(sym_table_bytes)?;
    let table_len = usize::try_from(read_u32(data, str_table_start, "COFF string table size").ok()?).ok()?;
    if table_len < 4 || string_offset >= table_len {
        return None;
    }
    let table_end = str_table_start.checked_add(table_len)?.min(data.len());
    let start = str_table_start.checked_add(string_offset)?;
    if start >= table_end {
        return None;
    }
    let end = table_end.min(start.saturating_add(MAX_CSTR_LEN));
    let tail = &data[start..end];
    let nul = tail.iter().position(|&b| b == 0)?;
    Some(String::from_utf8_lossy(&tail[..nul]).into_owned())
}

fn parse_imports(
    data: &[u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    dir: DataDirectory,
) -> Result<Vec<ImportModule>, PeError> {
    dir.virtual_address
        .checked_add(dir.size)
        .ok_or(PeError::IntegerOverflow {
            context: "import directory range",
        })?;
    let (base_offset, avail) =
        rva_to_file_span_raw(sections, size_of_headers, data.len(), dir.virtual_address).ok_or(
            PeError::UnmappedRva {
                context: "import directory",
                rva: dir.virtual_address,
            },
        )?;

    let mut modules = Vec::new();
    let mut index = 0usize;

    loop {
        if index >= MAX_IMPORT_MODULES {
            return Err(PeError::TableTooLarge {
                context: "import descriptor table",
                count: index,
            });
        }

        let rel = index
            .checked_mul(IMPORT_DESCRIPTOR_LEN)
            .ok_or(PeError::IntegerOverflow {
                context: "import descriptor offset",
            })?;
        let rel_end = rel
            .checked_add(IMPORT_DESCRIPTOR_LEN)
            .ok_or(PeError::IntegerOverflow {
                context: "import descriptor offset",
            })?;
        if rel_end > avail {
            return Err(PeError::Truncated {
                context: "import descriptor",
                needed: IMPORT_DESCRIPTOR_LEN,
                available: avail.saturating_sub(rel),
            });
        }
        let desc_off = base_offset
            .checked_add(rel)
            .ok_or(PeError::IntegerOverflow {
                context: "import descriptor offset",
            })?;
        let desc = read_slice(data, desc_off, IMPORT_DESCRIPTOR_LEN, "import descriptor")?;

        let original_first_thunk = le_u32(&desc[0..4]);
        let time_date_stamp = le_u32(&desc[4..8]);
        let forwarder_chain = le_u32(&desc[8..12]);
        let name_rva = le_u32(&desc[12..16]);
        let first_thunk = le_u32(&desc[16..20]);

        if original_first_thunk == 0
            && time_date_stamp == 0
            && forwarder_chain == 0
            && name_rva == 0
            && first_thunk == 0
        {
            break;
        }
        if name_rva == 0 && original_first_thunk == 0 && first_thunk == 0 {
            break;
        }

        let dll_name = read_c_string_rva(
            data,
            sections,
            size_of_headers,
            name_rva,
            "import DLL name",
        )?;

        let thunk_rva = if original_first_thunk != 0 {
            original_first_thunk
        } else {
            first_thunk
        };

        let symbols = if thunk_rva != 0 {
            parse_thunk_table_rva(data, sections, size_of_headers, thunk_rva, "import thunk")?
        } else {
            Vec::new()
        };

        modules.push(ImportModule {
            dll_name,
            iat_rva: first_thunk,
            symbols,
        });
        index += 1;
    }

    Ok(modules)
}

fn parse_delay_imports(
    data: &[u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    image_base: u64,
    dir: DataDirectory,
) -> Result<Vec<DelayImportModule>, PeError> {
    dir.virtual_address
        .checked_add(dir.size)
        .ok_or(PeError::IntegerOverflow {
            context: "delay-load import directory range",
        })?;
    let (base_offset, avail) =
        rva_to_file_span_raw(sections, size_of_headers, data.len(), dir.virtual_address).ok_or(
            PeError::UnmappedRva {
                context: "delay-load import directory",
                rva: dir.virtual_address,
            },
        )?;

    let mut modules = Vec::new();
    let mut index = 0usize;

    loop {
        if index >= MAX_IMPORT_MODULES {
            return Err(PeError::TableTooLarge {
                context: "delay-load import descriptor table",
                count: index,
            });
        }

        let rel = index
            .checked_mul(DELAY_IMPORT_DESCRIPTOR_LEN)
            .ok_or(PeError::IntegerOverflow {
                context: "delay-load descriptor offset",
            })?;
        let rel_end = rel
            .checked_add(DELAY_IMPORT_DESCRIPTOR_LEN)
            .ok_or(PeError::IntegerOverflow {
                context: "delay-load descriptor offset",
            })?;
        if rel_end > avail {
            return Err(PeError::Truncated {
                context: "delay-load import descriptor",
                needed: DELAY_IMPORT_DESCRIPTOR_LEN,
                available: avail.saturating_sub(rel),
            });
        }
        let desc_off = base_offset
            .checked_add(rel)
            .ok_or(PeError::IntegerOverflow {
                context: "delay-load descriptor offset",
            })?;
        let desc = read_slice(
            data,
            desc_off,
            DELAY_IMPORT_DESCRIPTOR_LEN,
            "delay-load import descriptor",
        )?;

        let attributes = le_u32(&desc[0..4]);
        let dll_name_addr = le_u32(&desc[4..8]);
        let iat_addr = le_u32(&desc[12..16]);
        let int_addr = le_u32(&desc[16..20]);

        if dll_name_addr == 0 {
            break;
        }

        let uses_rva = (attributes & DELAY_ATTR_RVA) != 0;
        let dll_name_rva =
            delay_addr32_to_rva(dll_name_addr, uses_rva, image_base, "delay-load DLL name")?;
        let dll_name = read_c_string_rva(
            data,
            sections,
            size_of_headers,
            dll_name_rva,
            "delay-load DLL name",
        )?;

        let iat_rva = if iat_addr != 0 {
            delay_addr32_to_rva(iat_addr, uses_rva, image_base, "delay-load IAT")?
        } else {
            0
        };
        let thunk_addr = if int_addr != 0 { int_addr } else { iat_addr };
        let symbols = if thunk_addr != 0 {
            let thunk_rva = delay_addr32_to_rva(
                thunk_addr,
                uses_rva,
                image_base,
                "delay-load import thunk table",
            )?;
            if uses_rva {
                parse_thunk_table_rva(
                    data,
                    sections,
                    size_of_headers,
                    thunk_rva,
                    "delay-load import thunk",
                )?
            } else {
                parse_thunk_table_va(
                    data,
                    sections,
                    size_of_headers,
                    image_base,
                    thunk_rva,
                    "delay-load import thunk (VA)",
                )?
            }
        } else {
            Vec::new()
        };

        modules.push(DelayImportModule {
            dll_name,
            attributes,
            uses_rva,
            iat_rva,
            symbols,
        });
        index += 1;
    }

    Ok(modules)
}

fn delay_addr32_to_rva(
    addr: u32,
    uses_rva: bool,
    image_base: u64,
    context: &'static str,
) -> Result<u32, PeError> {
    if uses_rva {
        Ok(addr)
    } else {
        va_to_rva(addr as u64, image_base, context)
    }
}

fn va_to_rva(va: u64, image_base: u64, context: &'static str) -> Result<u32, PeError> {
    let delta = va
        .checked_sub(image_base)
        .ok_or(PeError::InvalidVirtualAddress {
            context,
            va,
            image_base,
        })?;
    u32::try_from(delta).map_err(|_| PeError::InvalidVirtualAddress {
        context,
        va,
        image_base,
    })
}

fn parse_thunk_table_rva(
    data: &[u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    thunk_rva: u32,
    context: &'static str,
) -> Result<Vec<ImportedSymbol>, PeError> {
    let (base_offset, avail) =
        rva_to_file_span_raw(sections, size_of_headers, data.len(), thunk_rva).ok_or(
            PeError::UnmappedRva {
                context,
                rva: thunk_rva,
            },
        )?;

    let mut symbols = Vec::new();
    for i in 0..=MAX_THUNKS_PER_MODULE {
        if i == MAX_THUNKS_PER_MODULE {
            return Err(PeError::TableTooLarge { context, count: i });
        }
        let rel = i
            .checked_mul(8)
            .ok_or(PeError::IntegerOverflow { context })?;
        let rel_end = rel
            .checked_add(8)
            .ok_or(PeError::IntegerOverflow { context })?;
        if rel_end > avail {
            return Err(PeError::Truncated {
                context,
                needed: 8,
                available: avail.saturating_sub(rel),
            });
        }
        let off = base_offset
            .checked_add(rel)
            .ok_or(PeError::IntegerOverflow { context })?;
        let thunk = read_u64(data, off, context)?;
        if thunk == 0 {
            break;
        }
        if (thunk & IMAGE_ORDINAL_FLAG64) != 0 {
            symbols.push(ImportedSymbol::ByOrdinal {
                ordinal: (thunk & 0xFFFF) as u16,
            });
        } else {
            let hint_name_rva = u32::try_from(thunk).map_err(|_| PeError::InvalidOffset {
                context,
                offset: usize::try_from(thunk).unwrap_or(usize::MAX),
            })?;
            symbols.push(read_hint_name(
                data,
                sections,
                size_of_headers,
                hint_name_rva,
                context,
            )?);
        }
    }
    Ok(symbols)
}

fn parse_thunk_table_va(
    data: &[u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    image_base: u64,
    thunk_rva: u32,
    context: &'static str,
) -> Result<Vec<ImportedSymbol>, PeError> {
    let (base_offset, avail) =
        rva_to_file_span_raw(sections, size_of_headers, data.len(), thunk_rva).ok_or(
            PeError::UnmappedRva {
                context,
                rva: thunk_rva,
            },
        )?;

    let mut symbols = Vec::new();
    for i in 0..=MAX_THUNKS_PER_MODULE {
        if i == MAX_THUNKS_PER_MODULE {
            return Err(PeError::TableTooLarge { context, count: i });
        }
        let rel = i
            .checked_mul(8)
            .ok_or(PeError::IntegerOverflow { context })?;
        let rel_end = rel
            .checked_add(8)
            .ok_or(PeError::IntegerOverflow { context })?;
        if rel_end > avail {
            return Err(PeError::Truncated {
                context,
                needed: 8,
                available: avail.saturating_sub(rel),
            });
        }
        let off = base_offset
            .checked_add(rel)
            .ok_or(PeError::IntegerOverflow { context })?;
        let thunk = read_u64(data, off, context)?;
        if thunk == 0 {
            break;
        }
        if (thunk & IMAGE_ORDINAL_FLAG64) != 0 {
            symbols.push(ImportedSymbol::ByOrdinal {
                ordinal: (thunk & 0xFFFF) as u16,
            });
        } else {
            let hint_name_rva = va_to_rva(thunk, image_base, context)?;
            symbols.push(read_hint_name(
                data,
                sections,
                size_of_headers,
                hint_name_rva,
                context,
            )?);
        }
    }
    Ok(symbols)
}

fn read_hint_name(
    data: &[u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    hint_name_rva: u32,
    context: &'static str,
) -> Result<ImportedSymbol, PeError> {
    let hint_bytes = read_rva_slice(data, sections, size_of_headers, hint_name_rva, 2, context)?;
    let hint = le_u16(hint_bytes);
    let name_rva = hint_name_rva
        .checked_add(2)
        .ok_or(PeError::IntegerOverflow {
            context: "import symbol name RVA",
        })?;
    let name = read_c_string_rva(data, sections, size_of_headers, name_rva, "import symbol name")?;
    Ok(ImportedSymbol::ByName { hint, name })
}

fn parse_exports(
    data: &[u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    dir: DataDirectory,
) -> Result<ExportTable, PeError> {
    let dir_end = dir
        .virtual_address
        .checked_add(dir.size)
        .ok_or(PeError::IntegerOverflow {
            context: "export directory range",
        })?;
    let hdr = read_rva_slice(
        data,
        sections,
        size_of_headers,
        dir.virtual_address,
        EXPORT_DIRECTORY_LEN,
        "export directory",
    )?;

    let name_rva = le_u32(&hdr[12..16]);
    let ordinal_base = le_u32(&hdr[16..20]);
    let num_functions =
        usize::try_from(le_u32(&hdr[20..24])).map_err(|_| PeError::TableTooLarge {
            context: "export address table",
            count: usize::MAX,
        })?;
    let num_names =
        usize::try_from(le_u32(&hdr[24..28])).map_err(|_| PeError::TableTooLarge {
            context: "export name table",
            count: usize::MAX,
        })?;
    let addr_functions_rva = le_u32(&hdr[28..32]);
    let addr_names_rva = le_u32(&hdr[32..36]);
    let addr_ordinals_rva = le_u32(&hdr[36..40]);

    if num_functions > MAX_EXPORTS {
        return Err(PeError::TableTooLarge {
            context: "export address table",
            count: num_functions,
        });
    }
    if num_names > MAX_EXPORTS {
        return Err(PeError::TableTooLarge {
            context: "export name table",
            count: num_names,
        });
    }

    let module_name = if name_rva != 0 {
        Some(read_c_string_rva(
            data,
            sections,
            size_of_headers,
            name_rva,
            "export module name",
        )?)
    } else {
        None
    };

    if num_functions == 0 {
        return Ok(ExportTable {
            module_name,
            ordinal_base,
            symbols: Vec::new(),
        });
    }

    let eat_len = num_functions
        .checked_mul(4)
        .ok_or(PeError::IntegerOverflow {
            context: "export address table size",
        })?;
    let eat_bytes = read_rva_slice(
        data,
        sections,
        size_of_headers,
        addr_functions_rva,
        eat_len,
        "export address table",
    )?;

    let mut names_by_index: Vec<Vec<String>> = vec![Vec::new(); num_functions];
    if num_names > 0 {
        let name_ptrs_len = num_names.checked_mul(4).ok_or(PeError::IntegerOverflow {
            context: "export name pointer table size",
        })?;
        let ord_bytes_len = num_names.checked_mul(2).ok_or(PeError::IntegerOverflow {
            context: "export ordinal table size",
        })?;

        let name_ptrs = read_rva_slice(
            data,
            sections,
            size_of_headers,
            addr_names_rva,
            name_ptrs_len,
            "export name pointer table",
        )?;
        let ord_bytes = read_rva_slice(
            data,
            sections,
            size_of_headers,
            addr_ordinals_rva,
            ord_bytes_len,
            "export ordinal table",
        )?;

        for (name_chunk, ord_chunk) in name_ptrs.chunks_exact(4).zip(ord_bytes.chunks_exact(2)) {
            let sym_name_rva = le_u32(name_chunk);
            let eat_idx = usize::from(le_u16(ord_chunk));
            if eat_idx >= num_functions {
                return Err(PeError::InvalidOffset {
                    context: "export ordinal index out of range",
                    offset: eat_idx,
                });
            }
            let sym_name = read_c_string_rva(
                data,
                sections,
                size_of_headers,
                sym_name_rva,
                "export symbol name",
            )?;
            names_by_index[eat_idx].push(sym_name);
        }
    }

    let mut symbols = Vec::new();
    for (eat_idx, func_chunk) in eat_bytes.chunks_exact(4).enumerate() {
        let func_rva = le_u32(func_chunk);
        if func_rva == 0 {
            continue;
        }
        let eat_idx_u32 = u32::try_from(eat_idx).map_err(|_| PeError::IntegerOverflow {
            context: "export ordinal",
        })?;
        let ordinal = ordinal_base
            .checked_add(eat_idx_u32)
            .ok_or(PeError::IntegerOverflow {
                context: "export ordinal",
            })?;
        let forwarder = if func_rva >= dir.virtual_address && func_rva < dir_end {
            Some(read_c_string_rva(
                data,
                sections,
                size_of_headers,
                func_rva,
                "export forwarder",
            )?)
        } else {
            None
        };

        if names_by_index[eat_idx].is_empty() {
            symbols.push(ExportSymbol {
                name: None,
                ordinal,
                rva: func_rva,
                forwarder,
            });
        } else {
            names_by_index[eat_idx].sort();
            for sym_name in &names_by_index[eat_idx] {
                symbols.push(ExportSymbol {
                    name: Some(sym_name.clone()),
                    ordinal,
                    rva: func_rva,
                    forwarder: forwarder.clone(),
                });
            }
        }
    }

    Ok(ExportTable {
        module_name,
        ordinal_base,
        symbols,
    })
}

fn parse_tls(
    data: &[u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    image_base: u64,
    dir: DataDirectory,
) -> Result<TlsDirectory, PeError> {
    dir.virtual_address
        .checked_add(dir.size)
        .ok_or(PeError::IntegerOverflow {
            context: "TLS directory range",
        })?;
    let bytes = read_rva_slice(
        data,
        sections,
        size_of_headers,
        dir.virtual_address,
        TLS_DIRECTORY64_LEN,
        "TLS directory",
    )?;

    let start_address_of_raw_data = le_u64(&bytes[0..8]);
    let end_address_of_raw_data = le_u64(&bytes[8..16]);
    let address_of_index = le_u64(&bytes[16..24]);
    let address_of_callbacks = le_u64(&bytes[24..32]);
    let size_of_zero_fill = le_u32(&bytes[32..36]);
    let characteristics = le_u32(&bytes[36..40]);

    let mut callbacks = Vec::new();
    if address_of_callbacks != 0 {
        let cb_array_rva = va_to_rva(address_of_callbacks, image_base, "TLS callbacks array")?;
        let (cb_array_off, avail) =
            rva_to_file_span_raw(sections, size_of_headers, data.len(), cb_array_rva).ok_or(
                PeError::UnmappedRva {
                    context: "TLS callbacks array",
                    rva: cb_array_rva,
                },
            )?;

        for i in 0..=MAX_TLS_CALLBACKS {
            if i == MAX_TLS_CALLBACKS {
                return Err(PeError::TableTooLarge {
                    context: "TLS callbacks array",
                    count: i,
                });
            }
            let rel = i.checked_mul(8).ok_or(PeError::IntegerOverflow {
                context: "TLS callback offset",
            })?;
            let rel_end = rel.checked_add(8).ok_or(PeError::IntegerOverflow {
                context: "TLS callback offset",
            })?;
            if rel_end > avail {
                return Err(PeError::Truncated {
                    context: "TLS callback entry",
                    needed: 8,
                    available: avail.saturating_sub(rel),
                });
            }
            let off = cb_array_off
                .checked_add(rel)
                .ok_or(PeError::IntegerOverflow {
                    context: "TLS callback offset",
                })?;
            let cb_va = read_u64(data, off, "TLS callback entry")?;
            if cb_va == 0 {
                break;
            }
            let cb_rva = va_to_rva(cb_va, image_base, "TLS callback entry")?;
            callbacks.push(TlsCallback {
                va: cb_va,
                rva: cb_rva,
            });
        }
    }

    Ok(TlsDirectory {
        start_address_of_raw_data,
        end_address_of_raw_data,
        address_of_index,
        address_of_callbacks,
        size_of_zero_fill,
        characteristics,
        callbacks,
    })
}

fn read_c_string_rva(
    data: &[u8],
    sections: &[SectionHeader],
    size_of_headers: u32,
    rva: u32,
    context: &'static str,
) -> Result<String, PeError> {
    let (offset, avail) = rva_to_file_span_raw(sections, size_of_headers, data.len(), rva)
        .ok_or(PeError::UnmappedRva { context, rva })?;
    let scan_len = avail.min(MAX_CSTR_LEN);
    let slice = read_slice(data, offset, scan_len, context)?;
    let nul_pos = slice
        .iter()
        .position(|&b| b == 0)
        .ok_or(PeError::UnterminatedString { context, rva })?;
    Ok(String::from_utf8_lossy(&slice[..nul_pos]).into_owned())
}

fn read_slice<'a>(
    data: &'a [u8],
    offset: usize,
    len: usize,
    context: &'static str,
) -> Result<&'a [u8], PeError> {
    let end = offset.checked_add(len).ok_or(PeError::IntegerOverflow { context })?;
    if end > data.len() {
        return Err(PeError::Truncated {
            context,
            needed: len,
            available: data.len().saturating_sub(offset),
        });
    }
    Ok(&data[offset..end])
}

fn read_u32(data: &[u8], offset: usize, context: &'static str) -> Result<u32, PeError> {
    Ok(le_u32(read_slice(data, offset, 4, context)?))
}

fn read_u64(data: &[u8], offset: usize, context: &'static str) -> Result<u64, PeError> {
    Ok(le_u64(read_slice(data, offset, 8, context)?))
}

fn le_u16(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn le_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn le_u64(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

#[doc(hidden)]
pub mod test_util {
    use super::*;

    /// Builds a valid synthetic PE32+ binary in memory with:
    /// - `.text` (code + TLS callback targets + exported function stubs)
    /// - `.rdata` (Import Directory with named + ordinal imports, Delay-Load Imports in RVA
    ///   and VA modes, Export Directory with named + ordinal-only + forwarded exports, and
    ///   TLS Directory with 2 callbacks + 0 terminator)
    pub fn build_synthetic_pe64(delay_use_va: bool) -> Vec<u8> {
        let image_base: u64 = if delay_use_va {
            0x0040_0000
        } else {
            0x0000_0001_4000_0000
        };
        let file_alignment: u32 = 0x200;
        let section_alignment: u32 = 0x1000;

        let text_rva: u32 = 0x1000;
        let text_raw_off: usize = 0x200;
        let text_raw_size: usize = 0x200;

        let rdata_rva: u32 = 0x2000;
        let rdata_raw_off: usize = 0x400;
        let rdata_raw_size: usize = 0x800;

        let total_size = rdata_raw_off + rdata_raw_size;
        let mut buf = vec![0u8; total_size];

        // --- DOS header ---
        buf[0..2].copy_from_slice(&DOS_MAGIC);
        let e_lfanew: u32 = 0x80;
        write_u32(&mut buf, 0x3C, e_lfanew);

        // --- PE signature ---
        let pe_off = e_lfanew as usize;
        buf[pe_off..pe_off + 4].copy_from_slice(&PE_SIGNATURE);

        // --- COFF header ---
        let coff_off = pe_off + 4;
        let size_of_opt: u16 = 240; // 112 + 16 * 8
        write_u16(&mut buf, coff_off, 0x8664); // AMD64
        write_u16(&mut buf, coff_off + 2, 2); // 2 sections: .text, .rdata
        write_u32(&mut buf, coff_off + 4, 0x6700_1234); // timestamp
        write_u16(&mut buf, coff_off + 16, size_of_opt);
        write_u16(&mut buf, coff_off + 18, 0x0022); // EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE

        // --- PE32+ Optional header ---
        let opt_off = coff_off + COFF_HEADER_LEN;
        write_u16(&mut buf, opt_off, PE32_PLUS_MAGIC);
        buf[opt_off + 2] = 14;
        buf[opt_off + 3] = 38;
        write_u32(&mut buf, opt_off + 4, text_raw_size as u32);
        write_u32(&mut buf, opt_off + 8, rdata_raw_size as u32);
        write_u32(&mut buf, opt_off + 16, text_rva + 0x10); // entry point
        write_u32(&mut buf, opt_off + 20, text_rva); // base of code
        write_u64(&mut buf, opt_off + 24, image_base);
        write_u32(&mut buf, opt_off + 32, section_alignment);
        write_u32(&mut buf, opt_off + 36, file_alignment);
        write_u32(&mut buf, opt_off + 56, 0x3000); // SizeOfImage
        write_u32(&mut buf, opt_off + 60, 0x200); // SizeOfHeaders
        write_u32(&mut buf, opt_off + 64, 0xDEAD_BEEF); // CheckSum
        write_u16(&mut buf, opt_off + 68, 14); // Subsystem: XBOX
        write_u16(&mut buf, opt_off + 70, 0x8160); // DllCharacteristics
        write_u32(&mut buf, opt_off + 108, 16); // NumberOfRvaAndSizes

        let set_dir = |buf: &mut [u8], idx: usize, rva: u32, size: u32| {
            let base = opt_off + OPT_HEADER_MIN_LEN + idx * 8;
            write_u32(buf, base, rva);
            write_u32(buf, base + 4, size);
        };

        // --- Section headers ---
        let sec_off = opt_off + size_of_opt as usize;
        // .text
        buf[sec_off..sec_off + 5].copy_from_slice(b".text");
        write_u32(&mut buf, sec_off + 8, 0x120); // VirtualSize
        write_u32(&mut buf, sec_off + 12, text_rva);
        write_u32(&mut buf, sec_off + 16, text_raw_size as u32);
        write_u32(&mut buf, sec_off + 20, text_raw_off as u32);
        write_u32(
            &mut buf,
            sec_off + 36,
            IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE | 0x0000_0020,
        );

        // .rdata
        let sec2_off = sec_off + SECTION_HEADER_LEN;
        buf[sec2_off..sec2_off + 6].copy_from_slice(b".rdata");
        write_u32(&mut buf, sec2_off + 8, 0x700); // VirtualSize
        write_u32(&mut buf, sec2_off + 12, rdata_rva);
        write_u32(&mut buf, sec2_off + 16, rdata_raw_size as u32);
        write_u32(&mut buf, sec2_off + 20, rdata_raw_off as u32);
        write_u32(&mut buf, sec2_off + 36, IMAGE_SCN_MEM_READ | 0x0000_0040);

        // Put a few NOP/RET bytes in .text
        buf[text_raw_off + 0x10] = 0xC3;
        buf[text_raw_off + 0x20] = 0xC3;
        buf[text_raw_off + 0x30] = 0xC3;
        buf[text_raw_off + 0x40] = 0xC3;

        // Helper to write into .rdata at relative offset `rel`
        let rdata_rva_at = |rel: usize| -> u32 { rdata_rva + rel as u32 };
        let rdata_off_at = |rel: usize| -> usize { rdata_raw_off + rel };

        // ============================================================
        // 1. Standard Import Directory at .rdata + 0x000
        //    2 DLLs + null terminator (3 * 20 = 60 bytes)
        // ============================================================
        let imp_dir_rel = 0x000;
        let k32_name_rel = 0x080;
        let xgr_name_rel = 0x0A0;
        let k32_ilt_rel = 0x0C0; // 3 * 8 = 24 bytes (2 symbols + 0)
        let xgr_ilt_rel = 0x0E0; // 2 * 8 = 16 bytes (1 symbol + 0)
        let hn_virtual_alloc_rel = 0x100;
        let hn_xtask_create_rel = 0x120;

        write_cstr(&mut buf, rdata_off_at(k32_name_rel), "KERNEL32.dll");
        write_cstr(&mut buf, rdata_off_at(xgr_name_rel), "xgameruntime.dll");
        write_hint_name(&mut buf, rdata_off_at(hn_virtual_alloc_rel), 0x0123, "VirtualAlloc");
        write_hint_name(
            &mut buf,
            rdata_off_at(hn_xtask_create_rel),
            0x0007,
            "XTaskQueueCreate",
        );

        // KERNEL32.dll ILT: named VirtualAlloc + ordinal 42 + terminator
        write_u64(
            &mut buf,
            rdata_off_at(k32_ilt_rel),
            rdata_rva_at(hn_virtual_alloc_rel) as u64,
        );
        write_u64(
            &mut buf,
            rdata_off_at(k32_ilt_rel + 8),
            IMAGE_ORDINAL_FLAG64 | 42,
        );
        write_u64(&mut buf, rdata_off_at(k32_ilt_rel + 16), 0);

        // xgameruntime.dll ILT: named XTaskQueueCreate + terminator
        write_u64(
            &mut buf,
            rdata_off_at(xgr_ilt_rel),
            rdata_rva_at(hn_xtask_create_rel) as u64,
        );
        write_u64(&mut buf, rdata_off_at(xgr_ilt_rel + 8), 0);

        // Descriptor 0: KERNEL32.dll (uses OriginalFirstThunk)
        write_u32(&mut buf, rdata_off_at(imp_dir_rel), rdata_rva_at(k32_ilt_rel));
        write_u32(
            &mut buf,
            rdata_off_at(imp_dir_rel + 12),
            rdata_rva_at(k32_name_rel),
        );
        write_u32(
            &mut buf,
            rdata_off_at(imp_dir_rel + 16),
            rdata_rva_at(k32_ilt_rel),
        );

        // Descriptor 1: xgameruntime.dll (OriginalFirstThunk == 0, fallback to FirstThunk)
        let d1 = imp_dir_rel + IMPORT_DESCRIPTOR_LEN;
        write_u32(&mut buf, rdata_off_at(d1), 0);
        write_u32(&mut buf, rdata_off_at(d1 + 12), rdata_rva_at(xgr_name_rel));
        write_u32(&mut buf, rdata_off_at(d1 + 16), rdata_rva_at(xgr_ilt_rel));

        // Descriptor 2: zero terminator (already zeroed)
        set_dir(&mut buf, DIR_IMPORT, rdata_rva_at(imp_dir_rel), 60);

        // ============================================================
        // 2. Delay-Load Import Directory at .rdata + 0x180
        //    1 DLL + null terminator (2 * 32 = 64 bytes)
        // ============================================================
        let delay_dir_rel = 0x180;
        let delay_dll_rel = 0x1E0;
        let delay_int_rel = 0x200; // 3 * 8 = 24 bytes
        let hn_d3d_create_rel = 0x220;

        write_cstr(&mut buf, rdata_off_at(delay_dll_rel), "d3d12_xs.dll");
        write_hint_name(
            &mut buf,
            rdata_off_at(hn_d3d_create_rel),
            0x0015,
            "D3D12CreateDevice",
        );

        let encode_delay_addr32 = |rva: u32| -> u32 {
            if delay_use_va {
                (image_base as u32) + rva
            } else {
                rva
            }
        };
        let encode_delay_thunk_name = |rva: u32| -> u64 {
            if delay_use_va {
                image_base + rva as u64
            } else {
                rva as u64
            }
        };

        write_u64(
            &mut buf,
            rdata_off_at(delay_int_rel),
            encode_delay_thunk_name(rdata_rva_at(hn_d3d_create_rel)),
        );
        write_u64(
            &mut buf,
            rdata_off_at(delay_int_rel + 8),
            IMAGE_ORDINAL_FLAG64 | 99,
        );
        write_u64(&mut buf, rdata_off_at(delay_int_rel + 16), 0);

        let delay_attrs: u32 = if delay_use_va { 0 } else { DELAY_ATTR_RVA };
        write_u32(&mut buf, rdata_off_at(delay_dir_rel), delay_attrs);
        write_u32(
            &mut buf,
            rdata_off_at(delay_dir_rel + 4),
            encode_delay_addr32(rdata_rva_at(delay_dll_rel)),
        );
        write_u32(
            &mut buf,
            rdata_off_at(delay_dir_rel + 16),
            encode_delay_addr32(rdata_rva_at(delay_int_rel)),
        );

        set_dir(&mut buf, DIR_DELAY_IMPORT, rdata_rva_at(delay_dir_rel), 64);

        // ============================================================
        // 3. Export Directory at .rdata + 0x280
        //    Includes 1 named export, 1 ordinal-only export, 1 forwarded export
        // ============================================================
        let exp_dir_rel = 0x280;
        let exp_eat_rel = 0x2B0; // 3 * 4 = 12 bytes
        let exp_names_rel = 0x2C0; // 2 * 4 = 8 bytes
        let exp_ords_rel = 0x2D0; // 2 * 2 = 4 bytes
        let exp_mod_name_rel = 0x2E0;
        let exp_sym1_name_rel = 0x300;
        let exp_sym3_name_rel = 0x320;
        let exp_fwd_str_rel = 0x340;
        let exp_dir_size: u32 = 0x100; // covers 0x280..0x380 so forwarder is inside export dir

        write_cstr(&mut buf, rdata_off_at(exp_mod_name_rel), "SyntheticGame.exe");
        write_cstr(&mut buf, rdata_off_at(exp_sym1_name_rel), "GameMain");
        write_cstr(&mut buf, rdata_off_at(exp_sym3_name_rel), "ForwardedLog");
        write_cstr(&mut buf, rdata_off_at(exp_fwd_str_rel), "KERNEL32.OutputDebugStringA");

        // EAT: [0] = .text+0x10 (GameMain), [1] = .text+0x20 (ordinal-only), [2] = forwarder RVA
        write_u32(&mut buf, rdata_off_at(exp_eat_rel), text_rva + 0x10);
        write_u32(&mut buf, rdata_off_at(exp_eat_rel + 4), text_rva + 0x20);
        write_u32(
            &mut buf,
            rdata_off_at(exp_eat_rel + 8),
            rdata_rva_at(exp_fwd_str_rel),
        );

        // Name Pointer Table
        write_u32(
            &mut buf,
            rdata_off_at(exp_names_rel),
            rdata_rva_at(exp_sym1_name_rel),
        );
        write_u32(
            &mut buf,
            rdata_off_at(exp_names_rel + 4),
            rdata_rva_at(exp_sym3_name_rel),
        );

        // Name Ordinals Table (indices into EAT: 0 and 2)
        write_u16(&mut buf, rdata_off_at(exp_ords_rel), 0);
        write_u16(&mut buf, rdata_off_at(exp_ords_rel + 2), 2);

        // IMAGE_EXPORT_DIRECTORY
        write_u32(
            &mut buf,
            rdata_off_at(exp_dir_rel + 12),
            rdata_rva_at(exp_mod_name_rel),
        );
        write_u32(&mut buf, rdata_off_at(exp_dir_rel + 16), 10); // ordinal base = 10
        write_u32(&mut buf, rdata_off_at(exp_dir_rel + 20), 3); // NumberOfFunctions
        write_u32(&mut buf, rdata_off_at(exp_dir_rel + 24), 2); // NumberOfNames
        write_u32(
            &mut buf,
            rdata_off_at(exp_dir_rel + 28),
            rdata_rva_at(exp_eat_rel),
        );
        write_u32(
            &mut buf,
            rdata_off_at(exp_dir_rel + 32),
            rdata_rva_at(exp_names_rel),
        );
        write_u32(
            &mut buf,
            rdata_off_at(exp_dir_rel + 36),
            rdata_rva_at(exp_ords_rel),
        );

        set_dir(&mut buf, DIR_EXPORT, rdata_rva_at(exp_dir_rel), exp_dir_size);

        // ============================================================
        // 4. TLS Directory 64 at .rdata + 0x400
        //    2 callback VAs (.text+0x30, .text+0x40) + 0 terminator
        // ============================================================
        let tls_dir_rel = 0x400;
        let tls_cbs_rel = 0x440;

        write_u64(
            &mut buf,
            rdata_off_at(tls_dir_rel),
            image_base + rdata_rva_at(0x500) as u64,
        );
        write_u64(
            &mut buf,
            rdata_off_at(tls_dir_rel + 8),
            image_base + rdata_rva_at(0x520) as u64,
        );
        write_u64(
            &mut buf,
            rdata_off_at(tls_dir_rel + 16),
            image_base + rdata_rva_at(0x540) as u64,
        );
        write_u64(
            &mut buf,
            rdata_off_at(tls_dir_rel + 24),
            image_base + rdata_rva_at(tls_cbs_rel) as u64,
        );
        write_u32(&mut buf, rdata_off_at(tls_dir_rel + 32), 16);
        write_u32(&mut buf, rdata_off_at(tls_dir_rel + 36), 0x0010_0000);

        write_u64(
            &mut buf,
            rdata_off_at(tls_cbs_rel),
            image_base + (text_rva + 0x30) as u64,
        );
        write_u64(
            &mut buf,
            rdata_off_at(tls_cbs_rel + 8),
            image_base + (text_rva + 0x40) as u64,
        );
        write_u64(&mut buf, rdata_off_at(tls_cbs_rel + 16), 0);

        set_dir(
            &mut buf,
            DIR_TLS,
            rdata_rva_at(tls_dir_rel),
            TLS_DIRECTORY64_LEN as u32,
        );

        buf
    }

    fn write_u16(buf: &mut [u8], off: usize, v: u16) {
        buf[off..off + 2].copy_from_slice(&v.to_le_bytes());
    }

    fn write_u32(buf: &mut [u8], off: usize, v: u32) {
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn write_u64(buf: &mut [u8], off: usize, v: u64) {
        buf[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }

    fn write_cstr(buf: &mut [u8], off: usize, s: &str) {
        let bytes = s.as_bytes();
        buf[off..off + bytes.len()].copy_from_slice(bytes);
        buf[off + bytes.len()] = 0;
    }

    fn write_hint_name(buf: &mut [u8], off: usize, hint: u16, name: &str) {
        write_u16(buf, off, hint);
        write_cstr(buf, off + 2, name);
    }
}

#[cfg(test)]
mod tests {
    use super::test_util::build_synthetic_pe64;
    use super::*;

    #[test]
    fn parses_synthetic_pe64_rva_delay_imports() {
        let raw = build_synthetic_pe64(false);
        let pe = Pe64::parse(&raw).expect("synthetic PE64 should parse cleanly");

        assert_eq!(pe.e_lfanew, 0x80);
        assert_eq!(pe.coff.machine, 0x8664);
        assert_eq!(pe.coff.machine_name(), "AMD64");
        assert_eq!(pe.coff.number_of_sections, 2);
        assert_eq!(pe.coff.time_date_stamp, 0x6700_1234);
        assert_eq!(pe.coff.characteristics, 0x0022);

        assert_eq!(pe.optional.magic, PE32_PLUS_MAGIC);
        assert_eq!(pe.optional.entry_point_rva, 0x1010);
        assert_eq!(pe.optional.image_base, 0x0000_0001_4000_0000);
        assert_eq!(pe.optional.size_of_image, 0x3000);
        assert_eq!(pe.optional.subsystem, 14);
        assert_eq!(pe.optional.subsystem_name(), "XBOX");
        assert_eq!(pe.optional.data_directories.len(), 16);

        assert_eq!(pe.sections.len(), 2);
        assert_eq!(pe.sections[0].name, ".text");
        assert_eq!(pe.sections[0].permissions_str(), "r-x");
        assert_eq!(pe.sections[1].name, ".rdata");
        assert_eq!(pe.sections[1].permissions_str(), "r--");

        // RVA to file offset mapping
        assert_eq!(pe.rva_to_file_offset(0x1010), Some(0x210));
        assert_eq!(pe.rva_to_file_offset(0x2000), Some(0x400));
        assert_eq!(pe.rva_to_file_offset(0x1900), None); // outside .text virtual_size (0x120)
        assert_eq!(pe.rva_to_file_offset(0x9999), None);

        // Standard imports
        assert_eq!(pe.imports.len(), 2);
        assert_eq!(pe.imports[0].dll_name, "KERNEL32.dll");
        assert_eq!(
            pe.imports[0].symbols,
            vec![
                ImportedSymbol::ByName {
                    hint: 0x0123,
                    name: "VirtualAlloc".to_string(),
                },
                ImportedSymbol::ByOrdinal { ordinal: 42 },
            ]
        );
        assert_eq!(pe.imports[1].dll_name, "xgameruntime.dll");
        assert_eq!(
            pe.imports[1].symbols,
            vec![ImportedSymbol::ByName {
                hint: 0x0007,
                name: "XTaskQueueCreate".to_string(),
            }]
        );

        // Delay-load imports (RVA mode)
        assert_eq!(pe.delay_imports.len(), 1);
        assert_eq!(pe.delay_imports[0].dll_name, "d3d12_xs.dll");
        assert!(pe.delay_imports[0].uses_rva);
        assert_eq!(
            pe.delay_imports[0].symbols,
            vec![
                ImportedSymbol::ByName {
                    hint: 0x0015,
                    name: "D3D12CreateDevice".to_string(),
                },
                ImportedSymbol::ByOrdinal { ordinal: 99 },
            ]
        );

        // Exports
        let exports = pe.exports.as_ref().expect("export table should exist");
        assert_eq!(exports.module_name.as_deref(), Some("SyntheticGame.exe"));
        assert_eq!(exports.ordinal_base, 10);
        assert_eq!(exports.symbols.len(), 3);
        assert_eq!(
            exports.symbols[0],
            ExportSymbol {
                name: Some("GameMain".to_string()),
                ordinal: 10,
                rva: 0x1010,
                forwarder: None,
            }
        );
        assert_eq!(
            exports.symbols[1],
            ExportSymbol {
                name: None,
                ordinal: 11,
                rva: 0x1020,
                forwarder: None,
            }
        );
        assert_eq!(
            exports.symbols[2],
            ExportSymbol {
                name: Some("ForwardedLog".to_string()),
                ordinal: 12,
                rva: 0x2340,
                forwarder: Some("KERNEL32.OutputDebugStringA".to_string()),
            }
        );

        // TLS callbacks
        let tls = pe.tls.as_ref().expect("TLS directory should exist");
        assert_eq!(tls.size_of_zero_fill, 16);
        assert_eq!(
            tls.callbacks,
            vec![
                TlsCallback {
                    va: 0x0000_0001_4000_1030,
                    rva: 0x1030,
                },
                TlsCallback {
                    va: 0x0000_0001_4000_1040,
                    rva: 0x1040,
                },
            ]
        );
    }

    #[test]
    fn parses_delay_imports_in_va_mode() {
        let raw = build_synthetic_pe64(true);
        let pe = Pe64::parse(&raw).expect("synthetic PE64 with VA delay-imports should parse");

        assert_eq!(pe.delay_imports.len(), 1);
        assert!(!pe.delay_imports[0].uses_rva);
        assert_eq!(pe.delay_imports[0].attributes, 0);
        assert_eq!(pe.delay_imports[0].dll_name, "d3d12_xs.dll");
        assert_eq!(
            pe.delay_imports[0].symbols,
            vec![
                ImportedSymbol::ByName {
                    hint: 0x0015,
                    name: "D3D12CreateDevice".to_string(),
                },
                ImportedSymbol::ByOrdinal { ordinal: 99 },
            ]
        );
    }

    #[test]
    fn rejects_empty_and_truncated_inputs() {
        assert!(matches!(Pe64::parse(&[]), Err(PeError::Truncated { .. })));
        assert!(matches!(
            Pe64::parse(&[b'M', b'Z', 0, 0]),
            Err(PeError::Truncated { .. })
        ));

        let raw = build_synthetic_pe64(false);
        // Truncate right inside optional header
        assert!(matches!(
            Pe64::parse(&raw[..0xB0]),
            Err(PeError::Truncated { .. })
        ));
        // Truncate inside .rdata
        assert!(matches!(
            Pe64::parse(&raw[..0x420]),
            Err(PeError::Truncated { .. } | PeError::UnmappedRva { .. })
        ));
    }

    #[test]
    fn rejects_bad_magics_and_pe32() {
        let mut raw = build_synthetic_pe64(false);
        raw[0] = b'X';
        assert_eq!(Pe64::parse(&raw), Err(PeError::InvalidDosMagic([b'X', b'Z'])));

        let mut raw = build_synthetic_pe64(false);
        raw[0x80] = b'N';
        assert_eq!(
            Pe64::parse(&raw),
            Err(PeError::InvalidPeSignature([b'N', b'E', 0, 0]))
        );

        let mut raw = build_synthetic_pe64(false);
        // Optional header magic is at 0x80 + 4 + 20 = 0x98
        raw[0x98..0x9A].copy_from_slice(&0x010Bu16.to_le_bytes()); // PE32 (32-bit)
        assert_eq!(Pe64::parse(&raw), Err(PeError::UnsupportedOptionalMagic(0x010B)));
    }

    #[test]
    fn rejects_corrupt_offsets_and_huge_tables_without_panicking() {
        // Corrupt e_lfanew pointing beyond EOF
        let mut raw = build_synthetic_pe64(false);
        raw[0x3C..0x40].copy_from_slice(&0xFFFF_FF00u32.to_le_bytes());
        assert!(matches!(Pe64::parse(&raw), Err(PeError::Truncated { .. })));

        // Corrupt export NumberOfFunctions to huge value (no OOM panic)
        let mut raw = build_synthetic_pe64(false);
        let exp_num_funcs_off = 0x400 + 0x280 + 20;
        raw[exp_num_funcs_off..exp_num_funcs_off + 4]
            .copy_from_slice(&0x7FFF_FFFFu32.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&raw),
            Err(PeError::TableTooLarge { .. })
        ));

        // Corrupt TLS callback VA below image_base
        let mut raw = build_synthetic_pe64(false);
        let tls_cb0_off = 0x400 + 0x440;
        raw[tls_cb0_off..tls_cb0_off + 8].copy_from_slice(&0x0000_0000_0000_1234u64.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&raw),
            Err(PeError::InvalidVirtualAddress { .. })
        ));
    }

    /// Minimal PE64 builder with configurable headers and sections for edge-case unit tests.
    fn build_minimal_pe64(
        image_base: u64,
        size_of_image: u32,
        size_of_headers: u32,
        sections: &[(&[u8], u32, u32, u32, u32, u32)],
        file_len: usize,
    ) -> Vec<u8> {
        let e_lfanew: usize = 0x40;
        let coff_off = e_lfanew + 4;
        let opt_off = coff_off + COFF_HEADER_LEN;
        let size_of_opt: u16 = 240; // 112 + 16 * 8
        let sec_off = opt_off + usize::from(size_of_opt);
        let min_len = sec_off + sections.len() * SECTION_HEADER_LEN;
        let mut buf = vec![0u8; file_len.max(min_len)];

        buf[0..2].copy_from_slice(&DOS_MAGIC);
        buf[0x3C..0x40].copy_from_slice(&(e_lfanew as u32).to_le_bytes());
        buf[e_lfanew..e_lfanew + 4].copy_from_slice(&PE_SIGNATURE);

        // COFF header
        buf[coff_off..coff_off + 2].copy_from_slice(&0x8664u16.to_le_bytes());
        buf[coff_off + 2..coff_off + 4].copy_from_slice(&(sections.len() as u16).to_le_bytes());
        buf[coff_off + 16..coff_off + 18].copy_from_slice(&size_of_opt.to_le_bytes());
        buf[coff_off + 18..coff_off + 20].copy_from_slice(&0x0022u16.to_le_bytes());

        // PE32+ Optional header
        buf[opt_off..opt_off + 2].copy_from_slice(&PE32_PLUS_MAGIC.to_le_bytes());
        buf[opt_off + 24..opt_off + 32].copy_from_slice(&image_base.to_le_bytes());
        buf[opt_off + 32..opt_off + 36].copy_from_slice(&0x1000u32.to_le_bytes());
        buf[opt_off + 36..opt_off + 40].copy_from_slice(&0x200u32.to_le_bytes());
        buf[opt_off + 56..opt_off + 60].copy_from_slice(&size_of_image.to_le_bytes());
        buf[opt_off + 60..opt_off + 64].copy_from_slice(&size_of_headers.to_le_bytes());
        buf[opt_off + 108..opt_off + 112].copy_from_slice(&16u32.to_le_bytes());

        for (i, &(name, vsize, va, rsize, roff, chars)) in sections.iter().enumerate() {
            let off = sec_off + i * SECTION_HEADER_LEN;
            let n = name.len().min(8);
            buf[off..off + n].copy_from_slice(&name[..n]);
            buf[off + 8..off + 12].copy_from_slice(&vsize.to_le_bytes());
            buf[off + 12..off + 16].copy_from_slice(&va.to_le_bytes());
            buf[off + 16..off + 20].copy_from_slice(&rsize.to_le_bytes());
            buf[off + 20..off + 24].copy_from_slice(&roff.to_le_bytes());
            buf[off + 36..off + 40].copy_from_slice(&chars.to_le_bytes());
        }

        buf
    }

    #[test]
    fn rejects_truncated_dos_pe_coff_optional_and_section_headers() {
        let base = build_synthetic_pe64(false);
        // 1 byte short of DOS header
        assert!(matches!(
            Pe64::parse(&base[..DOS_HEADER_MIN_LEN - 1]),
            Err(PeError::Truncated {
                context: "DOS header",
                ..
            })
        ));
        // Truncated inside PE signature (e_lfanew = 0x80)
        assert!(matches!(
            Pe64::parse(&base[..0x83]),
            Err(PeError::Truncated {
                context: "PE signature",
                ..
            })
        ));
        // Truncated inside COFF header (0x84..0x98)
        assert!(matches!(
            Pe64::parse(&base[..0x97]),
            Err(PeError::Truncated {
                context: "COFF header",
                ..
            })
        ));
        // Truncated inside optional header (0x98..0x188)
        assert!(matches!(
            Pe64::parse(&base[..0x187]),
            Err(PeError::Truncated {
                context: "optional header",
                ..
            })
        ));
        // Truncated inside section table (0x188..0x1D8)
        assert!(matches!(
            Pe64::parse(&base[..0x1D7]),
            Err(PeError::Truncated {
                context: "section table",
                ..
            })
        ));

        // size_of_optional_header < 2
        let mut small_opt = base.clone();
        small_opt[0x84 + 16..0x84 + 18].copy_from_slice(&1u16.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&small_opt),
            Err(PeError::Truncated {
                context: "optional header magic",
                ..
            })
        ));

        // size_of_optional_header >= 2 with PE32+ magic but < OPT_HEADER_MIN_LEN
        let mut short_opt = base.clone();
        short_opt[0x84 + 16..0x84 + 18].copy_from_slice(&111u16.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&short_opt),
            Err(PeError::Truncated {
                context: "PE32+ optional header",
                ..
            })
        ));

        // Huge number_of_sections on small buffer must fail Truncated without OOM
        let mut huge_secs = base;
        huge_secs[0x84 + 2..0x84 + 4].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&huge_secs),
            Err(PeError::Truncated {
                context: "section table",
                ..
            })
        ));
    }

    #[test]
    fn rejects_u32_and_u64_header_and_section_overflow_boundaries() {
        // e_lfanew overflowing u32 when adding 4-byte PE signature
        for &bad_lfanew in &[u32::MAX, u32::MAX - 2, u32::MAX - 3] {
            let mut raw = build_synthetic_pe64(false);
            raw[0x3C..0x40].copy_from_slice(&bad_lfanew.to_le_bytes());
            assert!(matches!(
                Pe64::parse(&raw),
                Err(PeError::IntegerOverflow {
                    context: "PE signature offset",
                })
            ));
        }

        // image_base + size_of_image overflowing u64
        let bad_img = build_minimal_pe64(u64::MAX - 0x100, 0x3000, 0x200, &[], 0x400);
        assert!(matches!(
            Pe64::parse(&bad_img),
            Err(PeError::IntegerOverflow {
                context: "optional header image_base + size_of_image",
            })
        ));

        // Section virtual_address + virtual_size overflowing u32
        let bad_sec_va = build_minimal_pe64(
            0x1_4000_0000,
            0x3000,
            0x200,
            &[(b".ovfva", 0x2000, 0xFFFF_F000, 0x200, 0x200, IMAGE_SCN_MEM_READ)],
            0x400,
        );
        assert!(matches!(
            Pe64::parse(&bad_sec_va),
            Err(PeError::IntegerOverflow {
                context: "section virtual range",
            })
        ));

        // Section raw_offset + raw_size overflowing u32
        let bad_sec_raw = build_minimal_pe64(
            0x1_4000_0000,
            0x3000,
            0x200,
            &[(b".ovfraw", 0x200, 0x1000, 0x2000, 0xFFFF_F000, IMAGE_SCN_MEM_READ)],
            0x400,
        );
        assert!(matches!(
            Pe64::parse(&bad_sec_raw),
            Err(PeError::IntegerOverflow {
                context: "section raw range",
            })
        ));

        // Section image_base + virtual_address + virtual_extent overflowing u64 (with size_of_image = 0)
        let bad_sec_u64 = build_minimal_pe64(
            u64::MAX - 0x1000,
            0,
            0x200,
            &[(b".hi64", 0x200, 0x1000, 0x200, 0x200, IMAGE_SCN_MEM_READ)],
            0x400,
        );
        assert!(matches!(
            Pe64::parse(&bad_sec_u64),
            Err(PeError::IntegerOverflow {
                context: "section virtual address range",
            })
        ));

        // Export directory virtual_address + size overflowing u32
        let mut bad_exp_dir = build_synthetic_pe64(false);
        let exp_dir_entry = 0x98 + OPT_HEADER_MIN_LEN + DIR_EXPORT * 8;
        bad_exp_dir[exp_dir_entry..exp_dir_entry + 4]
            .copy_from_slice(&0xFFFF_F000u32.to_le_bytes());
        bad_exp_dir[exp_dir_entry + 4..exp_dir_entry + 8]
            .copy_from_slice(&0x2000u32.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&bad_exp_dir),
            Err(PeError::IntegerOverflow {
                context: "export directory range",
            })
        ));

        // Export ordinal_base + eat_idx overflowing u32
        let mut bad_exp_ord = build_synthetic_pe64(false);
        let exp_ord_base_off = 0x400 + 0x280 + 16;
        bad_exp_ord[exp_ord_base_off..exp_ord_base_off + 4]
            .copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&bad_exp_ord),
            Err(PeError::IntegerOverflow {
                context: "export ordinal",
            })
        ));

        // Huge number_of_rva_and_sizes is clamped safely to the optional header length
        let mut huge_dirs = build_synthetic_pe64(false);
        huge_dirs[0x98 + 108..0x98 + 112].copy_from_slice(&u32::MAX.to_le_bytes());
        let pe = Pe64::parse(&huge_dirs).expect("clamped data directories should still parse");
        assert_eq!(pe.optional.data_directories.len(), 16);
    }

    #[test]
    fn handles_zero_size_bss_and_out_of_bounds_sections_safely() {
        // Note: size_of_headers is intentionally larger than .bss VA (0x1000) to confirm
        // BSS RVAs do NOT fall through to header bytes.
        let raw = build_minimal_pe64(
            0x1_4000_0000,
            0x8000,
            0x1800,
            &[
                (b".empty", 0, 0x0800, 0, 0, IMAGE_SCN_MEM_READ),
                (b".bss", 0x1000, 0x1000, 0, 0, IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE),
                (b".rawonly", 0, 0x2000, 0x200, 0x200, IMAGE_SCN_MEM_READ),
                (b".partial", 0x1000, 0x3000, 0x100, 0x400, IMAGE_SCN_MEM_READ),
                (b".oob", 0x200, 0x4000, 0x200, 0x9000, IMAGE_SCN_MEM_READ),
            ],
            0x600,
        );
        let pe = Pe64::parse(&raw).expect("synthetic edge-case sections should parse");

        assert!(pe.sections[0].is_empty());
        assert!(!pe.sections[0].is_bss());
        assert_eq!(pe.sections[0].mapped_raw_size(), 0);

        assert!(!pe.sections[1].is_empty());
        assert!(pe.sections[1].is_bss());
        assert_eq!(pe.sections[1].virtual_extent(), 0x1000);
        assert_eq!(pe.sections[1].mapped_raw_size(), 0);
        // .bss must not map to file bytes and must NOT fall through to size_of_headers (0x1800)
        assert_eq!(pe.rva_to_file_offset(0x1000), None);
        assert_eq!(pe.rva_to_file_offset(0x1500), None);

        // .rawonly has virtual_size == 0 -> virtual_extent falls back to raw_size (0x200)
        assert_eq!(pe.sections[2].virtual_extent(), 0x200);
        assert_eq!(pe.rva_to_file_offset(0x2000), Some(0x200));
        assert_eq!(pe.rva_to_file_offset(0x21FF), Some(0x3FF));
        assert_eq!(pe.rva_to_file_offset(0x2200), None);

        // .partial has 0x100 initialized bytes then 0xF00 BSS tail
        assert_eq!(pe.sections[3].mapped_raw_size(), 0x100);
        assert_eq!(pe.rva_to_file_offset(0x3000), Some(0x400));
        assert_eq!(pe.rva_to_file_offset(0x30FF), Some(0x4FF));
        assert_eq!(pe.rva_to_file_offset(0x3100), None);
        assert_eq!(pe.rva_range_to_file_offset(0x3000, 0x100), Some(0x400));
        assert_eq!(pe.rva_range_to_file_offset(0x3000, 0x101), None);

        // .oob has raw_offset beyond file_size (0x600)
        assert_eq!(pe.rva_to_file_offset(0x4000), None);
        assert_eq!(pe.rva_range_to_file_offset(0x4000, 4), None);

        // Header RVA outside all sections maps cleanly within file_size
        assert_eq!(pe.rva_to_file_offset(0), Some(0));
        assert_eq!(pe.rva_to_file_offset(0x3F), Some(0x3F));
        assert_eq!(pe.rva_to_file_offset(0x600), None); // beyond file_size even though < size_of_headers
    }

    #[test]
    fn rva_translation_and_range_edge_cases() {
        // Overlapping sections: first section in table order owns the overlapping RVA range.
        let raw = build_minimal_pe64(
            0xFFFF_FFFF_0000_0000,
            0x4000,
            0x200,
            &[
                (b".first", 0x1000, 0x1000, 0x100, 0x200, IMAGE_SCN_MEM_READ),
                (b".second", 0x1000, 0x1000, 0x200, 0x400, IMAGE_SCN_MEM_READ),
            ],
            0x600,
        );
        let mut pe = Pe64::parse(&raw).expect("overlapping sections parse without panic");

        // RVA 0x1010 maps via .first (raw_offset 0x200), not .second (0x400)
        assert_eq!(pe.rva_to_file_offset(0x1010), Some(0x210));
        // RVA 0x1150 is in .first's BSS tail (0x100..0x1000) -> None (no fallthrough to .second)
        assert_eq!(pe.rva_to_file_offset(0x1150), None);

        // RVA range overflow & boundary checks
        assert_eq!(pe.rva_to_file_offset(u32::MAX), None);
        assert_eq!(pe.rva_range_to_file_offset(u32::MAX, 1), None);
        assert_eq!(pe.rva_range_to_file_offset(0xFFFF_FFF0, 0x20), None);
        assert_eq!(pe.rva_range_to_file_offset(0x10F8, 8), Some(0x2F8));
        assert_eq!(pe.rva_range_to_file_offset(0x10F8, 9), None);

        // Checked rva_to_va / va_to_rva
        assert_eq!(pe.rva_to_va(0x1000), Some(0xFFFF_FFFF_0000_1000));
        assert_eq!(pe.va_to_rva(0xFFFF_FFFF_0000_1000), Some(0x1000));
        assert_eq!(pe.va_to_rva(0xFFFF_FFFE_FFFF_FFFF), None); // below image_base

        // Manually injected overflowing section in Pe64 struct still cannot panic or wrap
        pe.sections.push(SectionHeader {
            name: ".bad".to_string(),
            virtual_size: 0x2000,
            virtual_address: 0xFFFF_F000,
            raw_size: 0x2000,
            raw_offset: 0x200,
            characteristics: IMAGE_SCN_MEM_READ,
        });
        assert_eq!(pe.rva_to_file_offset(0xFFFF_F100), None);
        assert_eq!(pe.rva_range_to_file_offset(0xFFFF_F100, 4), None);
    }

    #[test]
    fn prevents_cross_section_table_and_cstring_bleed() {
        // Truncate .rdata's virtual_size right in the middle of the import directory (which
        // starts at .rdata+0x000 and needs 60 bytes for 2 descriptors + terminator). The file
        // itself is still full length, so without section-bounded reads this would silently
        // bleed past .rdata's virtual_size.
        let mut raw = build_synthetic_pe64(false);
        let rdata_vsize_off = 0x188 + SECTION_HEADER_LEN + 8;
        raw[rdata_vsize_off..rdata_vsize_off + 4].copy_from_slice(&24u32.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&raw),
            Err(PeError::UnmappedRva { .. } | PeError::Truncated { .. })
        ));

        // Truncate .rdata virtual_size right before the NUL terminator of "KERNEL32.dll"
        // (at .rdata+0x080, length 12, NUL at 0x08C -> set virtual_size to 0x08C).
        let mut raw_cstr = build_synthetic_pe64(false);
        raw_cstr[rdata_vsize_off..rdata_vsize_off + 4].copy_from_slice(&0x08Cu32.to_le_bytes());
        assert!(matches!(
            Pe64::parse(&raw_cstr),
            Err(PeError::UnterminatedString {
                context: "import DLL name",
                ..
            })
        ));
    }

    #[test]
    fn coff_long_section_name_bounds_and_overflow_safety() {
        // Valid COFF long section name "/4"
        let mut raw = build_minimal_pe64(
            0x1_4000_0000,
            0x3000,
            0x200,
            &[(b"/4", 0x100, 0x1000, 0x100, 0x200, IMAGE_SCN_MEM_READ)],
            0x400,
        );
        let coff_off = 0x44;
        let symtab_off: u32 = 0x300;
        let num_syms: u32 = 1;
        let strtab_off = (symtab_off as usize) + (num_syms as usize) * 18;
        raw[coff_off + 8..coff_off + 12].copy_from_slice(&symtab_off.to_le_bytes());
        raw[coff_off + 12..coff_off + 16].copy_from_slice(&num_syms.to_le_bytes());
        let long_name = b".long_debug_section\0";
        let strtab_len = (4 + long_name.len()) as u32;
        raw[strtab_off..strtab_off + 4].copy_from_slice(&strtab_len.to_le_bytes());
        raw[strtab_off + 4..strtab_off + 4 + long_name.len()].copy_from_slice(long_name);

        let pe = Pe64::parse(&raw).expect("valid COFF long section name should resolve");
        assert_eq!(pe.sections[0].name, ".long_debug_section");

        // Overflowing symbol table pointer/count must never panic, falling back to raw "/4"
        let mut overflow_sym = raw.clone();
        overflow_sym[coff_off + 8..coff_off + 12].copy_from_slice(&u32::MAX.to_le_bytes());
        overflow_sym[coff_off + 12..coff_off + 16].copy_from_slice(&u32::MAX.to_le_bytes());
        let pe_ovf = Pe64::parse(&overflow_sym).expect("overflowing COFF symtab falls back cleanly");
        assert_eq!(pe_ovf.sections[0].name, "/4");

        // String offset past COFF string table length falls back cleanly
        let mut oob_str = raw;
        let sec_name_off = 0x40 + 4 + COFF_HEADER_LEN + 240;
        oob_str[sec_name_off..sec_name_off + 8].copy_from_slice(b"/99999\0\0");
        let pe_oob = Pe64::parse(&oob_str).expect("OOB COFF string offset falls back cleanly");
        assert_eq!(pe_oob.sections[0].name, "/99999");
    }

    #[test]
    fn deterministic_parsing_and_canonical_listings() {
        // Repeated parsing yields identical output
        let raw = build_synthetic_pe64(false);
        let first = Pe64::parse(&raw).expect("first parse succeeds");
        for _ in 0..8 {
            assert_eq!(Pe64::parse(&raw).expect("repeat parse succeeds"), first);
        }

        // Canonical sorted import listing
        let sorted_imports: Vec<&str> = first
            .imports_sorted()
            .into_iter()
            .map(|m| m.dll_name.as_str())
            .collect();
        assert_eq!(sorted_imports, vec!["KERNEL32.dll", "xgameruntime.dll"]);
        let sorted_delay: Vec<&str> = first
            .delay_imports_sorted()
            .into_iter()
            .map(|m| m.dll_name.as_str())
            .collect();
        assert_eq!(sorted_delay, vec!["d3d12_xs.dll"]);

        // Out-of-order sections in input still produce identical sections_by_rva() order
        let fwd = build_minimal_pe64(
            0x1_4000_0000,
            0x4000,
            0x200,
            &[
                (b".text", 0x100, 0x1000, 0x200, 0x200, IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE),
                (b".rdata", 0x100, 0x2000, 0x200, 0x400, IMAGE_SCN_MEM_READ),
                (b".data", 0x100, 0x3000, 0x200, 0x600, IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE),
            ],
            0x800,
        );
        let rev = build_minimal_pe64(
            0x1_4000_0000,
            0x4000,
            0x200,
            &[
                (b".data", 0x100, 0x3000, 0x200, 0x600, IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE),
                (b".text", 0x100, 0x1000, 0x200, 0x200, IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_EXECUTE),
                (b".rdata", 0x100, 0x2000, 0x200, 0x400, IMAGE_SCN_MEM_READ),
            ],
            0x800,
        );
        let pe_fwd = Pe64::parse(&fwd).expect("fwd parses");
        let pe_rev = Pe64::parse(&rev).expect("rev parses");
        let names_fwd: Vec<&str> = pe_fwd
            .sections_by_rva()
            .into_iter()
            .map(|s| s.name.as_str())
            .collect();
        let names_rev: Vec<&str> = pe_rev
            .sections_by_rva()
            .into_iter()
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(names_fwd, vec![".text", ".rdata", ".data"]);
        assert_eq!(names_fwd, names_rev);

        // Unsorted export alias names pointing to the same EAT index are sorted deterministically
        let mut aliased = build_synthetic_pe64(false);
        // Point both export name pointers (at .rdata+0x2C0) to EAT index 0 in reverse lexical order
        let rdata_off = 0x400;
        // Name 0 = "ForwardedLog" (0x2320), Name 1 = "GameMain" (0x2300) -> both ordinal index 0
        aliased[rdata_off + 0x2C0..rdata_off + 0x2C4].copy_from_slice(&0x2320u32.to_le_bytes());
        aliased[rdata_off + 0x2C4..rdata_off + 0x2C8].copy_from_slice(&0x2300u32.to_le_bytes());
        aliased[rdata_off + 0x2D0..rdata_off + 0x2D2].copy_from_slice(&0u16.to_le_bytes());
        aliased[rdata_off + 0x2D2..rdata_off + 0x2D4].copy_from_slice(&0u16.to_le_bytes());
        let pe_alias = Pe64::parse(&aliased).expect("aliased exports should parse");
        let exports = pe_alias.exports.expect("exports present");
        let ord10_names: Vec<&str> = exports
            .symbols
            .iter()
            .filter(|s| s.ordinal == 10)
            .filter_map(|s| s.name.as_deref())
            .collect();
        assert_eq!(ord10_names, vec!["ForwardedLog", "GameMain"]);
    }

    #[test]
    fn never_panics_on_arbitrary_and_mutated_bytes() {
        let mut state = 0x517C_C1B7_2722_0A95u64;
        let mut next_byte = || -> u8 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 56) as u8
        };

        // Arbitrary byte streams of lengths 0..=256
        for len in 0..=256 {
            let bytes: Vec<u8> = (0..len).map(|_| next_byte()).collect();
            let _ = Pe64::parse(&bytes);
        }

        // Targeted 0xFF and single-byte mutations across synthetic PE64s (both RVA and VA delay)
        for delay_va in [false, true] {
            let base = build_synthetic_pe64(delay_va);
            for stride in (0..base.len()).step_by(17) {
                let mut mutated = base.clone();
                let end = (stride + 8).min(mutated.len());
                for b in &mut mutated[stride..end] {
                    *b = 0xFF;
                }
                if let Ok(pe) = Pe64::parse(&mutated) {
                    let _ = pe.rva_to_file_offset(0);
                    let _ = pe.rva_to_file_offset(u32::MAX);
                    let _ = pe.rva_range_to_file_offset(u32::MAX, 8);
                }
            }
        }
    }
}
