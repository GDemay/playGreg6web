use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

/// Big-endian / synthetic ASCII FourCC magics (`"RPF9"` / `"RPF8"` / `"RPF7"`).
pub const RPF9_MAGIC: [u8; 4] = *b"RPF9";
pub const RPF8_MAGIC: [u8; 4] = *b"RPF8";
pub const RPF7_MAGIC: [u8; 4] = *b"RPF7";

/// Numeric RAGE RPF9 FourCC value (`u32::from_le_bytes(*b"9FPR") == 0x5250_4639`).
pub const RPF9_MAGIC_U32: u32 = 0x5250_4639;

/// Real RAGE little-endian FourCC magics on disk (`0x52504639` LE = `[0x39, 0x46, 0x50, 0x52]` = `"9FPR"`).
pub const RPF9_MAGIC_LE: [u8; 4] = RPF9_MAGIC_U32.to_le_bytes();
pub const RPF8_MAGIC_LE: [u8; 4] = *b"8FPR";
pub const RPF7_MAGIC_LE: [u8; 4] = *b"7FPR";

/// Minimum header size for RPF9 archives inspected by this tool (24 bytes):
/// - Classic/synthetic layout:
///   - `0x00..0x04`: magic (`"9FPR"` / `0x52504639` LE, or synthetic `"RPF9"` BE)
///   - `0x04..0x08`: `entry_count` (`u32` LE)
///   - `0x08..0x0C`: `names_size` (`u32` LE)
///   - `0x0C..0x10`: `encryption_tag` (`u32` LE)
///   - `0x10..0x14`: `metadata_size` (`u32` LE)
///   - `0x14..0x18`: `flags` (`u32` LE)
/// - Real Xbox Series RPF9 v9 layout (`"9FPR"` / `0x52504639` LE):
///   - `0x00..0x04`: magic (`0x52504639` LE)
///   - `0x04..0x10`: header / TOC byte offsets or sizes (`raw_words[1..=3]`)
///   - `0x10..0x14`: 32-bit encryption / hash / signature tag (`raw_words[4]`)
///   - `0x14..0x18`: 32-bit auxiliary tag / flags (`raw_words[5]`)
pub const RPF9_HEADER_SIZE: usize = 24;

/// Heuristic threshold for word 4 (`0x10..0x14`) on large little-endian `"9FPR"` archives
/// whose word 3 is not a known classic encryption tag: values >= 16 MiB indicate a 32-bit
/// encryption/hash/signature tag in the Xbox Series v9 header layout rather than a classic
/// `metadata_size` byte count.
const XBOX_V9_HASH_TAG_THRESHOLD: u32 = 0x0100_0000;

/// Conservative minimum size (in bytes) per TOC entry in RPF7/8/9 archives.
pub const RPF9_ENTRY_STRIDE: u64 = 16;

/// Sector granularity (in bytes) when [`RPF9_ENTRY_FLAG_OFFSET_SHIFT9`] is set on a file entry.
#[allow(dead_code)]
pub const RPF9_SECTOR_SIZE: u64 = 512;

/// TOC entry bit flags stored in bits `24..31` of entry word 0 (matching `web/workers/rpf9_vfs.js`).
#[allow(dead_code)]
pub const RPF9_ENTRY_FLAG_DIRECTORY: u8 = 0x80;
#[allow(dead_code)]
pub const RPF9_ENTRY_FLAG_COMPRESSED: u8 = 0x40;
#[allow(dead_code)]
pub const RPF9_ENTRY_FLAG_RESOURCE: u8 = 0x20;
#[allow(dead_code)]
pub const RPF9_ENTRY_FLAG_ENCRYPTED: u8 = 0x10;
#[allow(dead_code)]
pub const RPF9_ENTRY_FLAG_OFFSET_SHIFT9: u8 = 0x08;

/// Safety cap on classic RPF9 TOC entry count to prevent OOM pre-allocation on crafted headers
/// (262,144 entries * 16 bytes = 4 MiB TOC table).
pub const RPF9_MAX_ENTRY_COUNT: u32 = 262_144;

/// Safety cap on classic RPF9 name table byte length (64 MiB) to prevent OOM pre-allocation.
pub const RPF9_MAX_NAMES_SIZE: u32 = 64 * 1024 * 1024;

/// Safety cap on classic RPF9 metadata section byte length (64 MiB).
pub const RPF9_MAX_METADATA_SIZE: u32 = 64 * 1024 * 1024;

/// Maximum byte length allowed for a single NUL-terminated entry name inside the name table.
#[allow(dead_code)]
pub const RPF9_MAX_NAME_LEN: usize = 512;

/// Maximum directory nesting depth permitted when walking a classic RPF9 TOC tree.
#[allow(dead_code)]
pub const RPF9_MAX_TREE_DEPTH: usize = 128;

/// Default prefix byte length prefetched per archive/config in `bootset.json`.
pub const DEFAULT_BOOTSET_HEAD_BYTES: u64 = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpfMagicEndian {
    /// Real RAGE little-endian FourCC on disk (`39 46 50 52` = `"9FPR"`, `0x52504639` LE).
    Little,
    /// Synthetic / big-endian ASCII FourCC on disk (`52 50 46 39` = `"RPF9"`).
    Big,
}

impl fmt::Display for RpfMagicEndian {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpfMagicEndian::Little => write!(f, "little-endian"),
            RpfMagicEndian::Big => write!(f, "big-endian"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpfVersion {
    Rpf7,
    Rpf8,
    Rpf9,
    Other(u8),
}

impl fmt::Display for RpfVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpfVersion::Rpf7 => write!(f, "RPF7"),
            RpfVersion::Rpf8 => write!(f, "RPF8"),
            RpfVersion::Rpf9 => write!(f, "RPF9"),
            RpfVersion::Other(d) => write!(f, "RPF{}", *d as char),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rpf9HeaderLayout {
    /// Synthetic / classic layout where words 1..5 encode
    /// `(entry_count, names_size, encryption_tag, metadata_size, flags)` and
    /// `24 + entry_count * 16 + names_size + metadata_size <= total_size`.
    Classic,
    /// Real Xbox Series RPF9 v9 little-endian `"9FPR"` layout where words 1..3
    /// hold header/TOC offsets or sizes and word 4 (`0x10..0x14`) is a 32-bit
    /// encryption/hash/signature tag.
    XboxSeriesV9,
}

impl fmt::Display for Rpf9HeaderLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Rpf9HeaderLayout::Classic => write!(f, "classic/synthetic"),
            Rpf9HeaderLayout::XboxSeriesV9 => write!(f, "xbox-series-v9"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpfHeader {
    pub version: RpfVersion,
    pub magic_bytes: [u8; 4],
    pub magic_endian: RpfMagicEndian,
    pub header_layout: Rpf9HeaderLayout,
    pub raw_words: [u32; 6],
    pub entry_count: u32,
    pub names_size: u32,
    pub encryption_tag: u32,
    pub metadata_size: u32,
    pub flags: u32,
    pub total_size: u64,
    pub min_index_bytes: u64,
}

impl RpfHeader {
    /// Human-readable classification for known RAGE/RPF encryption tags.
    pub fn encryption_label(&self) -> &'static str {
        match self.encryption_tag {
            0x0000_0000 => "none",
            0x4E45_504F => "open",
            0x0FFF_FFF9 => "aes",
            0x0FEF_FFFF => "ng",
            _ => "custom",
        }
    }

    /// Formats a sanitized, metadata-only summary without ever dumping archive bytes.
    pub fn format_summary(&self) -> String {
        let magic_ascii: String = self
            .magic_bytes
            .iter()
            .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
            .collect();
        match self.header_layout {
            Rpf9HeaderLayout::Classic => format!(
                concat!(
                    "{} Archive Header\n",
                    "  Version:          {}\n",
                    "  Magic:            \"{}\" ({:02X} {:02X} {:02X} {:02X}, {})\n",
                    "  Header Layout:    {}\n",
                    "  File Size:        {} bytes\n",
                    "  Entry Count:      {}\n",
                    "  Names Size:       {} bytes\n",
                    "  Metadata Size:    {} bytes\n",
                    "  Min Index Bytes:  {} bytes\n",
                    "  Flags:            0x{:08X}\n",
                    "  Encryption Tag:   0x{:08X} ({})"
                ),
                self.version,
                self.version,
                magic_ascii,
                self.magic_bytes[0],
                self.magic_bytes[1],
                self.magic_bytes[2],
                self.magic_bytes[3],
                self.magic_endian,
                self.header_layout,
                self.total_size,
                self.entry_count,
                self.names_size,
                self.metadata_size,
                self.min_index_bytes,
                self.flags,
                self.encryption_tag,
                self.encryption_label()
            ),
            Rpf9HeaderLayout::XboxSeriesV9 => format!(
                concat!(
                    "{} Archive Header\n",
                    "  Version:             {}\n",
                    "  Magic:               \"{}\" ({:02X} {:02X} {:02X} {:02X}, {})\n",
                    "  Header Layout:       {}\n",
                    "  File Size:           {} bytes\n",
                    "  Header Words:        0x{:08X} 0x{:08X} 0x{:08X} 0x{:08X} 0x{:08X} 0x{:08X}\n",
                    "  Layout Words 1-3:    0x{:08X} ({}), 0x{:08X} ({}), 0x{:08X} ({})\n",
                    "  Min Index Bytes:     {} bytes\n",
                    "  Aux Tag / Flags:     0x{:08X}\n",
                    "  Encryption/Hash Tag: 0x{:08X} ({})"
                ),
                self.version,
                self.version,
                magic_ascii,
                self.magic_bytes[0],
                self.magic_bytes[1],
                self.magic_bytes[2],
                self.magic_bytes[3],
                self.magic_endian,
                self.header_layout,
                self.total_size,
                self.raw_words[0],
                self.raw_words[1],
                self.raw_words[2],
                self.raw_words[3],
                self.raw_words[4],
                self.raw_words[5],
                self.raw_words[1],
                self.raw_words[1],
                self.raw_words[2],
                self.raw_words[2],
                self.raw_words[3],
                self.raw_words[3],
                self.min_index_bytes,
                self.flags,
                self.encryption_tag,
                self.encryption_label()
            ),
        }
    }
}

impl fmt::Display for RpfHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.format_summary())
    }
}

#[derive(Debug)]
pub enum RpfError {
    Io(io::Error),
    TruncatedHeader { actual: usize, expected: usize },
    UnsupportedVersion { found: RpfVersion, expected: RpfVersion },
    InvalidMagic([u8; 4]),
    SizeOverflow,
    TruncatedArchive { required_bytes: u64, actual_bytes: u64 },
    EntryCountExceedsLimit { entry_count: u32, max_entries: u32 },
    NamesSizeExceedsLimit { names_size: u32, max_names_size: u32 },
    MetadataSizeExceedsLimit { metadata_size: u32, max_metadata_size: u32 },
    #[allow(dead_code)]
    UnsupportedHeaderLayout { layout: Rpf9HeaderLayout },
    #[allow(dead_code)]
    SliceOutOfBounds {
        section: &'static str,
        offset: u64,
        length: u64,
        available: u64,
    },
    #[allow(dead_code)]
    NameOffsetOutOfBounds {
        entry_index: usize,
        name_offset: u32,
        names_size: usize,
    },
    #[allow(dead_code)]
    UnterminatedName {
        entry_index: usize,
        name_offset: u32,
        max_len: usize,
    },
    #[allow(dead_code)]
    InvalidEntryName {
        entry_index: usize,
        reason: &'static str,
    },
    #[allow(dead_code)]
    InvalidTocRoot { entry_index: usize },
    #[allow(dead_code)]
    ChildRangeOutOfBounds {
        entry_index: usize,
        first_child: u32,
        child_count: u32,
        entry_count: usize,
    },
    #[allow(dead_code)]
    TocCycleOrDuplicateRef { entry_index: usize },
    #[allow(dead_code)]
    DuplicateCanonicalPath {
        path: String,
        first_index: usize,
        duplicate_index: usize,
    },
    #[allow(dead_code)]
    EntryDataOutOfBounds {
        entry_index: usize,
        offset: u64,
        length: u64,
        archive_size: u64,
    },
    #[allow(dead_code)]
    TreeDepthExceeded { entry_index: usize, max_depth: usize },
}

impl fmt::Display for RpfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RpfError::Io(err) => write!(f, "I/O error reading RPF archive: {err}"),
            RpfError::TruncatedHeader { actual, expected } => write!(
                f,
                "truncated RPF header: got {actual} bytes, need at least {expected} bytes"
            ),
            RpfError::UnsupportedVersion { found, expected } => write!(
                f,
                "unsupported RPF archive version {found} (expected {expected})"
            ),
            RpfError::InvalidMagic(magic) => write!(
                f,
                "invalid RPF magic bytes: {:02X} {:02X} {:02X} {:02X} (expected 9FPR / RPF9)",
                magic[0], magic[1], magic[2], magic[3]
            ),
            RpfError::SizeOverflow => {
                write!(f, "RPF header metadata sizes overflow 64-bit bounds")
            }
            RpfError::TruncatedArchive {
                required_bytes,
                actual_bytes,
            } => write!(
                f,
                "truncated RPF9 archive: header requires at least {required_bytes} bytes, but file is {actual_bytes} bytes"
            ),
            RpfError::EntryCountExceedsLimit {
                entry_count,
                max_entries,
            } => write!(
                f,
                "RPF9 entry count {entry_count} exceeds safety limit ({max_entries})"
            ),
            RpfError::NamesSizeExceedsLimit {
                names_size,
                max_names_size,
            } => write!(
                f,
                "RPF9 names table size {names_size} bytes exceeds safety limit ({max_names_size} bytes)"
            ),
            RpfError::MetadataSizeExceedsLimit {
                metadata_size,
                max_metadata_size,
            } => write!(
                f,
                "RPF9 metadata size {metadata_size} bytes exceeds safety limit ({max_metadata_size} bytes)"
            ),
            RpfError::UnsupportedHeaderLayout { layout } => write!(
                f,
                "RPF9 classic TOC parsing does not support header layout {layout}"
            ),
            RpfError::SliceOutOfBounds {
                section,
                offset,
                length,
                available,
            } => write!(
                f,
                "RPF9 {section} slice [{offset}..+{length}) exceeds available buffer length {available}"
            ),
            RpfError::NameOffsetOutOfBounds {
                entry_index,
                name_offset,
                names_size,
            } => write!(
                f,
                "RPF9 TOC entry {entry_index} name offset {name_offset} exceeds names table size {names_size}"
            ),
            RpfError::UnterminatedName {
                entry_index,
                name_offset,
                max_len,
            } => write!(
                f,
                "RPF9 TOC entry {entry_index} name at offset {name_offset} is missing NUL terminator within {max_len} bytes"
            ),
            RpfError::InvalidEntryName {
                entry_index,
                reason,
            } => write!(
                f,
                "RPF9 TOC entry {entry_index} has invalid name: {reason}"
            ),
            RpfError::InvalidTocRoot { entry_index } => write!(
                f,
                "RPF9 TOC root entry {entry_index} must be a directory entry"
            ),
            RpfError::ChildRangeOutOfBounds {
                entry_index,
                first_child,
                child_count,
                entry_count,
            } => write!(
                f,
                "RPF9 directory entry {entry_index} child range [{first_child}..+{child_count}) exceeds entry count {entry_count}"
            ),
            RpfError::TocCycleOrDuplicateRef { entry_index } => write!(
                f,
                "RPF9 TOC cycle or duplicate parent reference detected at entry {entry_index}"
            ),
            RpfError::DuplicateCanonicalPath {
                path,
                first_index,
                duplicate_index,
            } => write!(
                f,
                "RPF9 TOC contains duplicate canonical path \"{path}\" (entries {first_index} and {duplicate_index})"
            ),
            RpfError::EntryDataOutOfBounds {
                entry_index,
                offset,
                length,
                archive_size,
            } => write!(
                f,
                "RPF9 file entry {entry_index} payload [{offset}..+{length}) is out of archive bounds ({archive_size} bytes)"
            ),
            RpfError::TreeDepthExceeded {
                entry_index,
                max_depth,
            } => write!(
                f,
                "RPF9 TOC directory depth at entry {entry_index} exceeds maximum depth {max_depth}"
            ),
        }
    }
}

impl std::error::Error for RpfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            RpfError::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for RpfError {
    fn from(value: io::Error) -> Self {
        RpfError::Io(value)
    }
}

fn detect_rpf_version(magic: [u8; 4]) -> Result<(RpfVersion, RpfMagicEndian), RpfError> {
    if magic == RPF9_MAGIC_LE {
        Ok((RpfVersion::Rpf9, RpfMagicEndian::Little))
    } else if magic == RPF8_MAGIC_LE {
        Ok((RpfVersion::Rpf8, RpfMagicEndian::Little))
    } else if magic == RPF7_MAGIC_LE {
        Ok((RpfVersion::Rpf7, RpfMagicEndian::Little))
    } else if magic[1..4] == *b"FPR" && magic[0].is_ascii_alphanumeric() {
        Ok((RpfVersion::Other(magic[0]), RpfMagicEndian::Little))
    } else if magic == RPF9_MAGIC {
        Ok((RpfVersion::Rpf9, RpfMagicEndian::Big))
    } else if magic == RPF8_MAGIC {
        Ok((RpfVersion::Rpf8, RpfMagicEndian::Big))
    } else if magic == RPF7_MAGIC {
        Ok((RpfVersion::Rpf7, RpfMagicEndian::Big))
    } else if magic[0..3] == *b"RPF" && magic[3].is_ascii_alphanumeric() {
        Ok((RpfVersion::Other(magic[3]), RpfMagicEndian::Big))
    } else {
        Err(RpfError::InvalidMagic(magic))
    }
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Result<u32, RpfError> {
    let end = offset.checked_add(4).ok_or(RpfError::SizeOverflow)?;
    let slice = bytes.get(offset..end).ok_or(RpfError::TruncatedHeader {
        actual: bytes.len(),
        expected: end,
    })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

#[allow(dead_code)]
fn read_u32_le_section(
    bytes: &[u8],
    section: &'static str,
    offset: usize,
) -> Result<u32, RpfError> {
    let end = offset.checked_add(4).ok_or(RpfError::SizeOverflow)?;
    let slice = bytes.get(offset..end).ok_or(RpfError::SliceOutOfBounds {
        section,
        offset: offset as u64,
        length: 4,
        available: bytes.len() as u64,
    })?;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Checked end-offset calculation (`offset + length`) preventing 64-bit wraparound.
#[allow(dead_code)]
pub fn checked_range_end(offset: u64, length: u64) -> Result<u64, RpfError> {
    offset.checked_add(length).ok_or(RpfError::SizeOverflow)
}

/// Bounds-checked subslice extraction using checked 64-bit arithmetic and `usize` conversion.
#[allow(dead_code)]
fn checked_subslice<'a>(
    bytes: &'a [u8],
    section: &'static str,
    offset: u64,
    length: u64,
) -> Result<&'a [u8], RpfError> {
    let end_u64 = checked_range_end(offset, length)?;
    let available = bytes.len() as u64;
    if end_u64 > available {
        return Err(RpfError::SliceOutOfBounds {
            section,
            offset,
            length,
            available,
        });
    }
    let start = usize::try_from(offset).map_err(|_| RpfError::SizeOverflow)?;
    let end = usize::try_from(end_u64).map_err(|_| RpfError::SizeOverflow)?;
    bytes.get(start..end).ok_or(RpfError::SliceOutOfBounds {
        section,
        offset,
        length,
        available,
    })
}

fn is_known_classic_encryption_tag(tag: u32) -> bool {
    matches!(tag, 0x0000_0000 | 0x4E45_504F | 0x0FFF_FFF9 | 0x0FEF_FFFF)
}

fn classic_min_index_bytes(entry_count: u32, names_size: u32, metadata_size: u32) -> Option<u64> {
    let entries_bytes = u64::from(entry_count).checked_mul(RPF9_ENTRY_STRIDE)?;
    (RPF9_HEADER_SIZE as u64)
        .checked_add(entries_bytes)?
        .checked_add(u64::from(names_size))?
        .checked_add(u64::from(metadata_size))
}

/// Validated byte offsets and sizes for a classic RPF9 archive's header, entry table,
/// name table, and metadata region.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RpfTableLayout {
    pub entries_offset: u64,
    pub entries_size: u64,
    pub names_offset: u64,
    pub names_size: u64,
    pub metadata_offset: u64,
    pub metadata_size: u64,
    pub data_offset: u64,
    pub total_size: u64,
}

/// Computes and bounds-checks the classic RPF9 table layout without performing any allocation.
pub fn compute_classic_table_layout(
    entry_count: u32,
    names_size: u32,
    metadata_size: u32,
    total_size: u64,
) -> Result<RpfTableLayout, RpfError> {
    let entries_offset = RPF9_HEADER_SIZE as u64;
    let entries_size = u64::from(entry_count)
        .checked_mul(RPF9_ENTRY_STRIDE)
        .ok_or(RpfError::SizeOverflow)?;
    let names_offset = entries_offset
        .checked_add(entries_size)
        .ok_or(RpfError::SizeOverflow)?;
    let names_size_u64 = u64::from(names_size);
    let metadata_offset = names_offset
        .checked_add(names_size_u64)
        .ok_or(RpfError::SizeOverflow)?;
    let metadata_size_u64 = u64::from(metadata_size);
    let data_offset = metadata_offset
        .checked_add(metadata_size_u64)
        .ok_or(RpfError::SizeOverflow)?;

    if data_offset > total_size {
        return Err(RpfError::TruncatedArchive {
            required_bytes: data_offset,
            actual_bytes: total_size,
        });
    }

    if entry_count > RPF9_MAX_ENTRY_COUNT {
        return Err(RpfError::EntryCountExceedsLimit {
            entry_count,
            max_entries: RPF9_MAX_ENTRY_COUNT,
        });
    }
    if names_size > RPF9_MAX_NAMES_SIZE {
        return Err(RpfError::NamesSizeExceedsLimit {
            names_size,
            max_names_size: RPF9_MAX_NAMES_SIZE,
        });
    }
    if metadata_size > RPF9_MAX_METADATA_SIZE {
        return Err(RpfError::MetadataSizeExceedsLimit {
            metadata_size,
            max_metadata_size: RPF9_MAX_METADATA_SIZE,
        });
    }

    Ok(RpfTableLayout {
        entries_offset,
        entries_size,
        names_offset,
        names_size: names_size_u64,
        metadata_offset,
        metadata_size: metadata_size_u64,
        data_offset,
        total_size,
    })
}

impl RpfHeader {
    /// Returns the checked classic table layout for this header, rejecting unsupported layouts
    /// (such as [`Rpf9HeaderLayout::XboxSeriesV9`]) and out-of-bounds/oversized tables.
    #[allow(dead_code)]
    pub fn table_layout(&self) -> Result<RpfTableLayout, RpfError> {
        match self.header_layout {
            Rpf9HeaderLayout::Classic => compute_classic_table_layout(
                self.entry_count,
                self.names_size,
                self.metadata_size,
                self.total_size,
            ),
            layout => Err(RpfError::UnsupportedHeaderLayout { layout }),
        }
    }
}

/// Inspects an RPF9 header from an in-memory prefix slice and known total archive size.
///
/// Validates the magic version (`9FPR` / `0x52504639` LE and synthetic `RPF9` BE only;
/// `RPF8`/`8FPR`, `RPF7`/`7FPR`, and other `RPF*`/`*FPR` return [`RpfError::UnsupportedVersion`]).
///
/// Big-endian `"RPF9"` archives are validated using the classic synthetic layout.
/// Little-endian `"9FPR"` archives support both classic synthetic headers (where TOC/name/metadata
/// sizes fit within `total_size`) and real Xbox Series RPF9 v9 headers (where word 4 at offset
/// `0x10..0x14` is a 32-bit encryption/hash/signature tag and words 1..3 hold layout offsets/sizes).
pub fn inspect_rpf_bytes(header_bytes: &[u8], total_size: u64) -> Result<RpfHeader, RpfError> {
    let magic_slice = header_bytes.get(0..4).ok_or(RpfError::TruncatedHeader {
        actual: header_bytes.len(),
        expected: RPF9_HEADER_SIZE,
    })?;

    let magic = [
        magic_slice[0],
        magic_slice[1],
        magic_slice[2],
        magic_slice[3],
    ];
    let (version, magic_endian) = detect_rpf_version(magic)?;
    if version != RpfVersion::Rpf9 {
        return Err(RpfError::UnsupportedVersion {
            found: version,
            expected: RpfVersion::Rpf9,
        });
    }

    if header_bytes.len() < RPF9_HEADER_SIZE {
        return Err(RpfError::TruncatedHeader {
            actual: header_bytes.len(),
            expected: RPF9_HEADER_SIZE,
        });
    }

    if total_size < RPF9_HEADER_SIZE as u64 {
        return Err(RpfError::TruncatedArchive {
            required_bytes: RPF9_HEADER_SIZE as u64,
            actual_bytes: total_size,
        });
    }

    let raw_words = [
        read_u32_le(header_bytes, 0)?,
        read_u32_le(header_bytes, 4)?,
        read_u32_le(header_bytes, 8)?,
        read_u32_le(header_bytes, 12)?,
        read_u32_le(header_bytes, 16)?,
        read_u32_le(header_bytes, 20)?,
    ];

    let [_, w1, w2, w3, w4, w5] = raw_words;
    let classic_min = classic_min_index_bytes(w1, w2, w4);

    if magic_endian == RpfMagicEndian::Little {
        let is_xbox_series_v9 = match classic_min {
            None => true,
            Some(min_bytes) if min_bytes > total_size => true,
            Some(_) => !is_known_classic_encryption_tag(w3) && w4 >= XBOX_V9_HASH_TAG_THRESHOLD,
        };

        if is_xbox_series_v9 {
            return Ok(RpfHeader {
                version,
                magic_bytes: magic,
                magic_endian,
                header_layout: Rpf9HeaderLayout::XboxSeriesV9,
                raw_words,
                entry_count: 0,
                names_size: 0,
                encryption_tag: w4,
                metadata_size: 0,
                flags: w5,
                total_size,
                min_index_bytes: RPF9_HEADER_SIZE as u64,
            });
        }
    }

    let layout = compute_classic_table_layout(w1, w2, w4, total_size)?;

    Ok(RpfHeader {
        version,
        magic_bytes: magic,
        magic_endian,
        header_layout: Rpf9HeaderLayout::Classic,
        raw_words,
        entry_count: w1,
        names_size: w2,
        encryption_tag: w3,
        metadata_size: w4,
        flags: w5,
        total_size,
        min_index_bytes: layout.data_offset,
    })
}

/// Non-extracting RPF9 header inspector for local files of any size (including 100+ GB archives).
///
/// Only reads at most [`RPF9_HEADER_SIZE`] bytes from offset 0 after querying file length.
pub fn inspect_rpf_file(path: &Path) -> Result<RpfHeader, RpfError> {
    let mut file = fs::File::open(path)?;
    let total_size = file.metadata()?.len();
    file.seek(SeekFrom::Start(0))?;

    let to_read =
        usize::try_from(total_size.min(RPF9_HEADER_SIZE as u64)).unwrap_or(RPF9_HEADER_SIZE);
    let mut buf = [0u8; RPF9_HEADER_SIZE];
    file.read_exact(&mut buf[..to_read])?;

    inspect_rpf_bytes(&buf[..to_read], total_size)
}

/// Entry-type-specific metadata for a validated classic RPF9 TOC entry.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RpfEntryKind {
    Directory {
        first_child_index: u32,
        child_count: u32,
        reserved: u32,
    },
    File {
        data_offset: u64,
        uncompressed_size: u32,
        compressed_size: u32,
        stored_size: u32,
        is_compressed: bool,
        is_resource: bool,
        is_encrypted: bool,
        is_sector_offset: bool,
    },
}

/// Validated classic RPF9 TOC entry bound to its canonical lowercase virtual path.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpfTocEntry {
    pub index: usize,
    pub name: String,
    pub canonical_path: String,
    pub name_offset: u32,
    pub flags: u8,
    pub kind: RpfEntryKind,
}

/// Parsed classic RPF9 Table of Contents with canonical deterministic ordering.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RpfToc {
    pub header: RpfHeader,
    pub layout: RpfTableLayout,
    /// Canonical lowercase virtual path -> entry (deterministic ascending `BTreeMap` order).
    pub entries_by_path: BTreeMap<String, RpfTocEntry>,
    /// Canonical lowercase directory path -> sorted set of direct child canonical paths.
    pub children_by_dir: BTreeMap<String, BTreeSet<String>>,
}

#[allow(dead_code)]
impl RpfToc {
    /// Returns all non-root canonical virtual paths in deterministic ascending order.
    pub fn list_paths(&self) -> Vec<&str> {
        self.entries_by_path
            .keys()
            .filter(|k| !k.is_empty())
            .map(String::as_str)
            .collect()
    }

    /// Returns direct child canonical paths for `dir_path` (normalized) in deterministic order.
    pub fn list_dir(&self, dir_path: &str) -> Vec<&str> {
        let key = normalize_canonical_lookup_path(dir_path);
        self.children_by_dir
            .get(&key)
            .map(|children| children.iter().map(String::as_str).collect())
            .unwrap_or_default()
    }

    /// Looks up a TOC entry by virtual path (case-insensitive, normalized slashes).
    pub fn get(&self, path: &str) -> Option<&RpfTocEntry> {
        let key = normalize_canonical_lookup_path(path);
        self.entries_by_path.get(&key)
    }

    /// Number of indexed entries (including root if present).
    pub fn entry_count(&self) -> usize {
        self.entries_by_path.len()
    }
}

#[allow(dead_code)]
fn normalize_canonical_lookup_path(path: &str) -> String {
    path.replace('\\', "/")
        .split('/')
        .filter(|seg| !seg.is_empty() && *seg != ".")
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join("/")
}

#[allow(dead_code)]
fn read_checked_name(
    names_bytes: &[u8],
    entry_index: usize,
    name_offset: u32,
) -> Result<String, RpfError> {
    if names_bytes.is_empty() {
        if name_offset == 0 {
            return Ok(String::new());
        }
        return Err(RpfError::NameOffsetOutOfBounds {
            entry_index,
            name_offset,
            names_size: 0,
        });
    }

    let start = usize::try_from(name_offset).map_err(|_| RpfError::SizeOverflow)?;
    if start >= names_bytes.len() {
        return Err(RpfError::NameOffsetOutOfBounds {
            entry_index,
            name_offset,
            names_size: names_bytes.len(),
        });
    }
    let tail = &names_bytes[start..];

    let scan_limit = tail.len().min(RPF9_MAX_NAME_LEN.saturating_add(1));
    let nul_pos = tail[..scan_limit]
        .iter()
        .position(|&b| b == 0)
        .ok_or(RpfError::UnterminatedName {
            entry_index,
            name_offset,
            max_len: RPF9_MAX_NAME_LEN,
        })?;

    if nul_pos > RPF9_MAX_NAME_LEN {
        return Err(RpfError::UnterminatedName {
            entry_index,
            name_offset,
            max_len: RPF9_MAX_NAME_LEN,
        });
    }

    let name_bytes = &tail[..nul_pos];
    let name = std::str::from_utf8(name_bytes).map_err(|_| RpfError::InvalidEntryName {
        entry_index,
        reason: "name is not valid UTF-8",
    })?;
    Ok(name.to_string())
}

#[allow(dead_code)]
fn validate_child_entry_name(entry_index: usize, name: &str) -> Result<(), RpfError> {
    if name.is_empty() {
        return Err(RpfError::InvalidEntryName {
            entry_index,
            reason: "child entry name is empty",
        });
    }
    if name == "." || name == ".." {
        return Err(RpfError::InvalidEntryName {
            entry_index,
            reason: "path traversal segment is not allowed",
        });
    }
    if name.contains('/') || name.contains('\\') {
        return Err(RpfError::InvalidEntryName {
            entry_index,
            reason: "entry name contains path separator",
        });
    }
    if name.bytes().any(|b| b < 0x20 || b == 0x7F) {
        return Err(RpfError::InvalidEntryName {
            entry_index,
            reason: "entry name contains control character",
        });
    }
    Ok(())
}

#[allow(dead_code)]
struct RawTocEntry {
    index: usize,
    name: String,
    name_offset: u32,
    flags: u8,
    kind: RpfEntryKind,
}

/// Parses and validates a classic RPF9 archive's TOC and name table from an in-memory buffer
/// (either the full archive or a prefix covering header + entry table + name table) with
/// checked offset/size arithmetic, bounded allocation, cycle detection, and canonical
/// deterministic `BTreeMap`/`BTreeSet` ordering.
#[allow(dead_code)]
pub fn parse_rpf_toc(archive_bytes: &[u8], total_size: u64) -> Result<RpfToc, RpfError> {
    let header = inspect_rpf_bytes(archive_bytes, total_size)?;
    let layout = header.table_layout()?;

    // Bounds-check and slice entry + name tables BEFORE any allocation.
    let entries_slice = checked_subslice(
        archive_bytes,
        "entry table",
        layout.entries_offset,
        layout.entries_size,
    )?;
    let names_slice = checked_subslice(
        archive_bytes,
        "names table",
        layout.names_offset,
        layout.names_size,
    )?;

    let entry_count = usize::try_from(header.entry_count).map_err(|_| RpfError::SizeOverflow)?;
    let mut raw_entries = Vec::with_capacity(entry_count);

    for idx in 0..entry_count {
        let base = idx
            .checked_mul(RPF9_ENTRY_STRIDE as usize)
            .ok_or(RpfError::SizeOverflow)?;
        let w0 = read_u32_le_section(entries_slice, "entry table", base)?;
        let w1 = read_u32_le_section(
            entries_slice,
            "entry table",
            base.checked_add(4).ok_or(RpfError::SizeOverflow)?,
        )?;
        let w2 = read_u32_le_section(
            entries_slice,
            "entry table",
            base.checked_add(8).ok_or(RpfError::SizeOverflow)?,
        )?;
        let w3 = read_u32_le_section(
            entries_slice,
            "entry table",
            base.checked_add(12).ok_or(RpfError::SizeOverflow)?,
        )?;

        let name_offset = w0 & 0x00FF_FFFF;
        let flags = ((w0 >> 24) & 0xFF) as u8;
        let name = read_checked_name(names_slice, idx, name_offset)?;

        let kind = if (flags & RPF9_ENTRY_FLAG_DIRECTORY) != 0 {
            let child_end = w1.checked_add(w2).ok_or(RpfError::SizeOverflow)?;
            let child_end_usize = usize::try_from(child_end).map_err(|_| RpfError::SizeOverflow)?;
            if child_end_usize > entry_count {
                return Err(RpfError::ChildRangeOutOfBounds {
                    entry_index: idx,
                    first_child: w1,
                    child_count: w2,
                    entry_count,
                });
            }
            RpfEntryKind::Directory {
                first_child_index: w1,
                child_count: w2,
                reserved: w3,
            }
        } else {
            let is_compressed = (flags & RPF9_ENTRY_FLAG_COMPRESSED) != 0;
            let is_resource = (flags & RPF9_ENTRY_FLAG_RESOURCE) != 0;
            let is_encrypted = (flags & RPF9_ENTRY_FLAG_ENCRYPTED) != 0;
            let is_sector_offset = (flags & RPF9_ENTRY_FLAG_OFFSET_SHIFT9) != 0;

            let data_offset = if is_sector_offset {
                u64::from(w1)
                    .checked_mul(RPF9_SECTOR_SIZE)
                    .ok_or(RpfError::SizeOverflow)?
            } else {
                u64::from(w1)
            };
            let uncompressed_size = w2;
            let compressed_size = w3;
            let stored_size = if is_compressed {
                compressed_size
            } else {
                uncompressed_size
            };
            let data_end = checked_range_end(data_offset, u64::from(stored_size))?;

            if stored_size > 0 {
                if data_offset < layout.data_offset || data_end > total_size {
                    return Err(RpfError::EntryDataOutOfBounds {
                        entry_index: idx,
                        offset: data_offset,
                        length: u64::from(stored_size),
                        archive_size: total_size,
                    });
                }
            } else if data_offset > total_size {
                return Err(RpfError::EntryDataOutOfBounds {
                    entry_index: idx,
                    offset: data_offset,
                    length: 0,
                    archive_size: total_size,
                });
            }

            RpfEntryKind::File {
                data_offset,
                uncompressed_size,
                compressed_size,
                stored_size,
                is_compressed,
                is_resource,
                is_encrypted,
                is_sector_offset,
            }
        };

        raw_entries.push(RawTocEntry {
            index: idx,
            name,
            name_offset,
            flags,
            kind,
        });
    }

    let mut entries_by_path: BTreeMap<String, RpfTocEntry> = BTreeMap::new();
    let mut children_by_dir: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    if raw_entries.is_empty() {
        children_by_dir.insert(String::new(), BTreeSet::new());
        return Ok(RpfToc {
            header,
            layout,
            entries_by_path,
            children_by_dir,
        });
    }

    if !matches!(raw_entries[0].kind, RpfEntryKind::Directory { .. }) {
        return Err(RpfError::InvalidTocRoot { entry_index: 0 });
    }

    let mut visited = vec![false; entry_count];
    visited[0] = true;

    let root_raw = &raw_entries[0];
    entries_by_path.insert(
        String::new(),
        RpfTocEntry {
            index: root_raw.index,
            name: String::new(),
            canonical_path: String::new(),
            name_offset: root_raw.name_offset,
            flags: root_raw.flags,
            kind: root_raw.kind,
        },
    );
    children_by_dir.insert(String::new(), BTreeSet::new());

    // Iterative DFS traversal avoids recursion stack overflow on deeply nested archives.
    let mut stack: Vec<(usize, String, usize)> = vec![(0, String::new(), 0)];
    while let Some((dir_idx, dir_path, depth)) = stack.pop() {
        let (first_child_index, child_count) = match raw_entries[dir_idx].kind {
            RpfEntryKind::Directory {
                first_child_index,
                child_count,
                ..
            } => (first_child_index, child_count),
            RpfEntryKind::File { .. } => continue,
        };

        let next_depth = depth.checked_add(1).ok_or(RpfError::SizeOverflow)?;
        let start = usize::try_from(first_child_index).map_err(|_| RpfError::SizeOverflow)?;
        let count = usize::try_from(child_count).map_err(|_| RpfError::SizeOverflow)?;
        let end = start.checked_add(count).ok_or(RpfError::SizeOverflow)?;

        for child_idx in start..end {
            if next_depth > RPF9_MAX_TREE_DEPTH {
                return Err(RpfError::TreeDepthExceeded {
                    entry_index: child_idx,
                    max_depth: RPF9_MAX_TREE_DEPTH,
                });
            }
            if visited[child_idx] {
                return Err(RpfError::TocCycleOrDuplicateRef {
                    entry_index: child_idx,
                });
            }
            visited[child_idx] = true;

            let child = &raw_entries[child_idx];
            validate_child_entry_name(child_idx, &child.name)?;

            let segment = child.name.to_lowercase();
            let canonical_path = if dir_path.is_empty() {
                segment
            } else {
                format!("{dir_path}/{segment}")
            };

            if let Some(existing) = entries_by_path.get(&canonical_path) {
                return Err(RpfError::DuplicateCanonicalPath {
                    path: canonical_path,
                    first_index: existing.index,
                    duplicate_index: child_idx,
                });
            }

            entries_by_path.insert(
                canonical_path.clone(),
                RpfTocEntry {
                    index: child.index,
                    name: child.name.clone(),
                    canonical_path: canonical_path.clone(),
                    name_offset: child.name_offset,
                    flags: child.flags,
                    kind: child.kind,
                },
            );
            children_by_dir
                .entry(dir_path.clone())
                .or_default()
                .insert(canonical_path.clone());

            if matches!(child.kind, RpfEntryKind::Directory { .. }) {
                children_by_dir.entry(canonical_path.clone()).or_default();
                stack.push((child_idx, canonical_path, next_depth));
            }
        }
    }

    Ok(RpfToc {
        header,
        layout,
        entries_by_path,
        children_by_dir,
    })
}

/// Bounds-checked slice reader for a file entry's stored payload bytes within `archive_bytes`.
#[allow(dead_code)]
pub fn read_rpf_entry_bytes<'a>(
    archive_bytes: &'a [u8],
    entry: &RpfTocEntry,
) -> Result<&'a [u8], RpfError> {
    match entry.kind {
        RpfEntryKind::Directory { .. } => Ok(&[]),
        RpfEntryKind::File {
            data_offset,
            stored_size,
            ..
        } => checked_subslice(
            archive_bytes,
            "entry payload",
            data_offset,
            u64::from(stored_size),
        ),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootsetEntry {
    pub path: String,
    pub offset: u64,
    pub length: u64,
}

fn is_bootset_candidate(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("rpf") || ext.eq_ignore_ascii_case("config"))
}

fn collect_bootset_entries(
    root: &Path,
    dir: &Path,
    max_header_bytes: u64,
    out: &mut BTreeMap<String, BootsetEntry>,
) -> io::Result<()> {
    let mut subdirs = BTreeSet::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            subdirs.insert(path);
        } else if meta.is_file() && is_bootset_candidate(&path) {
            let clamped_len = meta.len().min(max_header_bytes);
            if clamped_len == 0 {
                continue;
            }
            let rel = path
                .strip_prefix(root)
                .map_err(io::Error::other)?
                .to_string_lossy()
                .replace('\\', "/");
            out.insert(
                rel.clone(),
                BootsetEntry {
                    path: rel,
                    offset: 0,
                    length: clamped_len,
                },
            );
        }
    }
    for subdir in subdirs {
        collect_bootset_entries(root, &subdir, max_header_bytes, out)?;
    }
    Ok(())
}

/// Scans `game_dir` for `.rpf` and `.config` files (case-insensitive) and returns canonically
/// ordered `bootset.json` range entries with each `length` clamped to the file's actual size.
pub fn generate_bootset(game_dir: &Path, max_header_bytes: u64) -> io::Result<Vec<BootsetEntry>> {
    let mut entries = BTreeMap::new();
    collect_bootset_entries(game_dir, game_dir, max_header_bytes, &mut entries)?;
    Ok(entries.into_values().collect())
}

pub(crate) fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Renders `bootset.json` matching `docs/io_manifest.md`.
pub fn render_bootset_json(entries: &[BootsetEntry]) -> String {
    if entries.is_empty() {
        return "[]\n".to_string();
    }
    let mut out = String::from("[\n");
    for (idx, entry) in entries.iter().enumerate() {
        let comma = if idx + 1 == entries.len() { "" } else { "," };
        out.push_str(&format!(
            "  {{\"path\": \"{}\", \"offset\": {}, \"length\": {}}}{}\n",
            json_escape(&entry.path),
            entry.offset,
            entry.length,
            comma
        ));
    }
    out.push_str("]\n");
    out
}

/// Generates and writes `bootset.json` for `game_dir` to `out_file`.
pub fn write_bootset(
    game_dir: &Path,
    out_file: &Path,
    max_header_bytes: u64,
) -> io::Result<usize> {
    let entries = generate_bootset(game_dir, max_header_bytes)?;
    if let Some(parent) = out_file.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let json = render_bootset_json(&entries);
    fs::write(out_file, json)?;
    Ok(entries.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> std::path::PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = env::temp_dir().join(format!(
            "playgta6_rpf9_{label}_{}_{}",
            std::process::id(),
            n
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn make_rpf_header(
        magic: [u8; 4],
        entry_count: u32,
        names_size: u32,
        encryption_tag: u32,
        metadata_size: u32,
        flags: u32,
    ) -> [u8; RPF9_HEADER_SIZE] {
        let mut buf = [0u8; RPF9_HEADER_SIZE];
        buf[0..4].copy_from_slice(&magic);
        buf[4..8].copy_from_slice(&entry_count.to_le_bytes());
        buf[8..12].copy_from_slice(&names_size.to_le_bytes());
        buf[12..16].copy_from_slice(&encryption_tag.to_le_bytes());
        buf[16..20].copy_from_slice(&metadata_size.to_le_bytes());
        buf[20..24].copy_from_slice(&flags.to_le_bytes());
        buf
    }

    #[test]
    fn parses_valid_synthetic_rpf9_header_and_summary() {
        let header = make_rpf_header(RPF9_MAGIC, 128, 4096, 0x4E45_504F, 1024, 0x0000_0003);
        let total_size = 10 * 1024 * 1024;
        let parsed = inspect_rpf_bytes(&header, total_size).unwrap();

        assert_eq!(parsed.version, RpfVersion::Rpf9);
        assert_eq!(parsed.magic_bytes, RPF9_MAGIC);
        assert_eq!(parsed.magic_endian, RpfMagicEndian::Big);
        assert_eq!(parsed.entry_count, 128);
        assert_eq!(parsed.names_size, 4096);
        assert_eq!(parsed.encryption_tag, 0x4E45_504F);
        assert_eq!(parsed.encryption_label(), "open");
        assert_eq!(parsed.metadata_size, 1024);
        assert_eq!(parsed.flags, 0x0000_0003);
        assert_eq!(parsed.total_size, total_size);
        assert_eq!(parsed.min_index_bytes, 24 + 128 * 16 + 4096 + 1024);

        let summary = parsed.format_summary();
        assert!(summary.contains("RPF9"));
        assert!(summary.contains("52 50 46 39, big-endian"));
        assert!(summary.contains("128"));
        assert!(summary.contains("4096 bytes"));
        assert!(summary.contains("1024 bytes"));
        assert!(summary.contains("0x00000003"));
        assert!(summary.contains("0x4E45504F (open)"));
    }

    #[test]
    fn parses_real_little_endian_9fpr_rpf9_header_and_summary() {
        assert_eq!(RPF9_MAGIC_LE, [0x39, 0x46, 0x50, 0x52]);
        assert_eq!(u32::from_le_bytes(RPF9_MAGIC_LE), RPF9_MAGIC_U32);
        assert_eq!(RPF9_MAGIC_U32.to_le_bytes(), *b"9FPR");

        let header = make_rpf_header(RPF9_MAGIC_LE, 64, 2048, 0x0FEF_FFFF, 512, 0x0000_0007);
        let total_size = 8 * 1024 * 1024;
        let parsed = inspect_rpf_bytes(&header, total_size).unwrap();

        assert_eq!(parsed.version, RpfVersion::Rpf9);
        assert_eq!(parsed.magic_bytes, [0x39, 0x46, 0x50, 0x52]);
        assert_eq!(parsed.magic_endian, RpfMagicEndian::Little);
        assert_eq!(parsed.entry_count, 64);
        assert_eq!(parsed.names_size, 2048);
        assert_eq!(parsed.encryption_tag, 0x0FEF_FFFF);
        assert_eq!(parsed.encryption_label(), "ng");
        assert_eq!(parsed.metadata_size, 512);
        assert_eq!(parsed.flags, 0x0000_0007);
        assert_eq!(parsed.total_size, total_size);
        assert_eq!(parsed.min_index_bytes, 24 + 64 * 16 + 2048 + 512);

        let summary = parsed.format_summary();
        assert!(summary.contains("RPF9 Archive Header"));
        assert!(summary.contains("\"9FPR\" (39 46 50 52, little-endian)"));
        assert!(summary.contains("0x0FEFFFFF (ng)"));
    }

    #[test]
    fn inspects_little_endian_9fpr_rpf_file_without_reading_payload() {
        let dir = unique_temp_dir("inspect_file_le");
        let rpf_path = dir.join("appdata_synthetic.rpf");

        let header = make_rpf_header(
            RPF9_MAGIC_U32.to_le_bytes(),
            4,
            128,
            0x4E45_504F,
            64,
            0x0000_0001,
        );
        let mut content = Vec::with_capacity(4096);
        content.extend_from_slice(&header);
        content.extend_from_slice(&vec![0xBB; 4096 - RPF9_HEADER_SIZE]);
        fs::write(&rpf_path, &content).unwrap();

        let parsed = inspect_rpf_file(&rpf_path).unwrap();
        assert_eq!(parsed.version, RpfVersion::Rpf9);
        assert_eq!(parsed.magic_bytes, RPF9_MAGIC_LE);
        assert_eq!(parsed.magic_endian, RpfMagicEndian::Little);
        assert_eq!(parsed.entry_count, 4);
        assert_eq!(parsed.encryption_label(), "open");
        assert_eq!(parsed.total_size, 4096);
        assert!(!parsed.format_summary().contains("BBBB"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_rpf8_and_rpf7_with_clear_version_error() {
        for magic8 in [RPF8_MAGIC, RPF8_MAGIC_LE] {
            let rpf8 = make_rpf_header(magic8, 10, 128, 0, 0, 0);
            let err8 = inspect_rpf_bytes(&rpf8, 4096).unwrap_err();
            match err8 {
                RpfError::UnsupportedVersion { found, expected } => {
                    assert_eq!(found, RpfVersion::Rpf8);
                    assert_eq!(expected, RpfVersion::Rpf9);
                }
                other => panic!("expected UnsupportedVersion(Rpf8), got {other:?}"),
            }
            assert!(err8.to_string().contains("RPF8"));
            assert!(err8.to_string().contains("RPF9"));
        }

        for magic7 in [RPF7_MAGIC, RPF7_MAGIC_LE] {
            let rpf7 = make_rpf_header(magic7, 10, 128, 0, 0, 0);
            let err7 = inspect_rpf_bytes(&rpf7, 4096).unwrap_err();
            match err7 {
                RpfError::UnsupportedVersion { found, expected } => {
                    assert_eq!(found, RpfVersion::Rpf7);
                    assert_eq!(expected, RpfVersion::Rpf9);
                }
                other => panic!("expected UnsupportedVersion(Rpf7), got {other:?}"),
            }
            assert!(err7.to_string().contains("RPF7"));
        }

        let rpf6_le = make_rpf_header(*b"6FPR", 10, 128, 0, 0, 0);
        let err6 = inspect_rpf_bytes(&rpf6_le, 4096).unwrap_err();
        assert!(matches!(
            err6,
            RpfError::UnsupportedVersion {
                found: RpfVersion::Other(b'6'),
                expected: RpfVersion::Rpf9,
            }
        ));
    }

    #[test]
    fn rejects_bad_magic() {
        let bad = make_rpf_header(*b"PK\x03\x04", 0, 0, 0, 0, 0);
        let err = inspect_rpf_bytes(&bad, 1024).unwrap_err();
        assert!(matches!(err, RpfError::InvalidMagic([0x50, 0x4B, 0x03, 0x04])));
        assert!(err.to_string().contains("invalid RPF magic"));

        let near_miss = make_rpf_header(*b"9FPX", 0, 0, 0, 0, 0);
        let near_err = inspect_rpf_bytes(&near_miss, 1024).unwrap_err();
        assert!(matches!(near_err, RpfError::InvalidMagic([0x39, 0x46, 0x50, 0x58])));
    }

    #[test]
    fn rejects_truncated_header_and_truncated_archive() {
        let short_err = inspect_rpf_bytes(b"RPF9\x01\x00\x00\x00", 8).unwrap_err();
        assert!(matches!(
            short_err,
            RpfError::TruncatedHeader {
                actual: 8,
                expected: RPF9_HEADER_SIZE
            }
        ));

        let short_le_err = inspect_rpf_bytes(b"9FPR\x01\x00\x00\x00", 8).unwrap_err();
        assert!(matches!(
            short_le_err,
            RpfError::TruncatedHeader {
                actual: 8,
                expected: RPF9_HEADER_SIZE
            }
        ));

        // Header declares 100 entries (1600 B) + 512 B names + 24 B header = 2136 B,
        // but archive size is only 1000 B.
        let header = make_rpf_header(RPF9_MAGIC, 100, 512, 0, 0, 0);
        let trunc_err = inspect_rpf_bytes(&header, 1000).unwrap_err();
        assert!(matches!(
            trunc_err,
            RpfError::TruncatedArchive {
                required_bytes: 2136,
                actual_bytes: 1000
            }
        ));
    }

    #[test]
    fn inspects_rpf_file_without_reading_payload() {
        let dir = unique_temp_dir("inspect_file");
        let rpf_path = dir.join("test_archive.rpf");

        let header = make_rpf_header(RPF9_MAGIC, 2, 64, 0x0FFF_FFF9, 32, 0x10);
        let mut content = Vec::with_capacity(2048);
        content.extend_from_slice(&header);
        content.extend_from_slice(&vec![0xAA; 2048 - RPF9_HEADER_SIZE]);
        fs::write(&rpf_path, &content).unwrap();

        let parsed = inspect_rpf_file(&rpf_path).unwrap();
        assert_eq!(parsed.version, RpfVersion::Rpf9);
        assert_eq!(parsed.entry_count, 2);
        assert_eq!(parsed.encryption_label(), "aes");
        assert_eq!(parsed.total_size, 2048);

        // Summary must not leak payload bytes.
        assert!(!parsed.format_summary().contains("AAAA"));

        let _ = fs::remove_dir_all(&dir);
    }

    fn make_raw_words_header(words: [u32; 6]) -> [u8; RPF9_HEADER_SIZE] {
        let mut buf = [0u8; RPF9_HEADER_SIZE];
        for (i, word) in words.iter().enumerate() {
            let start = i * 4;
            buf[start..start + 4].copy_from_slice(&word.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parses_xbox_series_v9_appdata_header_and_summary() {
        let words = [
            RPF9_MAGIC_U32,
            0x0000_0360,
            0x0000_0340,
            0x0000_0040,
            0x24E6_3799,
            0xB766_DF72,
        ];
        let header = make_raw_words_header(words);
        let parsed = inspect_rpf_bytes(&header, 8_880).unwrap();

        assert_eq!(parsed.version, RpfVersion::Rpf9);
        assert_eq!(parsed.magic_bytes, RPF9_MAGIC_LE);
        assert_eq!(parsed.magic_endian, RpfMagicEndian::Little);
        assert_eq!(parsed.header_layout, Rpf9HeaderLayout::XboxSeriesV9);
        assert_eq!(parsed.raw_words, words);
        assert_eq!(parsed.entry_count, 0);
        assert_eq!(parsed.names_size, 0);
        assert_eq!(parsed.metadata_size, 0);
        assert_eq!(parsed.encryption_tag, 0x24E6_3799);
        assert_eq!(parsed.encryption_label(), "custom");
        assert_eq!(parsed.flags, 0xB766_DF72);
        assert_eq!(parsed.total_size, 8_880);
        assert_eq!(parsed.min_index_bytes, RPF9_HEADER_SIZE as u64);

        let summary = parsed.format_summary();
        assert!(summary.contains("RPF9 Archive Header"));
        assert!(summary.contains("\"9FPR\" (39 46 50 52, little-endian)"));
        assert!(summary.contains("xbox-series-v9"));
        assert!(summary.contains("8880 bytes"));
        assert!(summary.contains("0x52504639 0x00000360 0x00000340 0x00000040 0x24E63799 0xB766DF72"));
        assert!(summary.contains("0x00000360 (864), 0x00000340 (832), 0x00000040 (64)"));
        assert!(summary.contains("0xB766DF72"));
        assert!(summary.contains("0x24E63799 (custom)"));
    }

    #[test]
    fn parses_xbox_series_v9_large_archives_without_misclassifying_hash_tag() {
        let words = [
            RPF9_MAGIC_U32,
            0x0000_0360,
            0x0000_0340,
            0x0000_0040,
            0x24E6_3799,
            0xB766_DF72,
        ];
        let header = make_raw_words_header(words);

        // 6.6 MiB archive (like common.rpf) and 4 GiB archive (where 0x24E63799 < total_size).
        for total_size in [6_642_192u64, 4_294_967_296u64] {
            let parsed = inspect_rpf_bytes(&header, total_size).unwrap();
            assert_eq!(parsed.header_layout, Rpf9HeaderLayout::XboxSeriesV9);
            assert_eq!(parsed.raw_words, words);
            assert_eq!(parsed.encryption_tag, 0x24E6_3799);
            assert_eq!(parsed.flags, 0xB766_DF72);
            assert_eq!(parsed.total_size, total_size);
        }
    }

    #[test]
    fn inspects_xbox_series_v9_rpf_file_without_reading_payload() {
        let dir = unique_temp_dir("inspect_file_v9");
        let rpf_path = dir.join("appdata0_synthetic.rpf");

        let words = [
            RPF9_MAGIC_U32,
            0x0000_0360,
            0x0000_0340,
            0x0000_0040,
            0x24E6_3799,
            0xB766_DF72,
        ];
        let header = make_raw_words_header(words);
        let mut content = Vec::with_capacity(8_880);
        content.extend_from_slice(&header);
        content.extend_from_slice(&vec![0xCC; 8_880 - RPF9_HEADER_SIZE]);
        fs::write(&rpf_path, &content).unwrap();

        let parsed = inspect_rpf_file(&rpf_path).unwrap();
        assert_eq!(parsed.header_layout, Rpf9HeaderLayout::XboxSeriesV9);
        assert_eq!(parsed.encryption_tag, 0x24E6_3799);
        assert_eq!(parsed.flags, 0xB766_DF72);
        assert_eq!(parsed.total_size, 8_880);
        assert!(!parsed.format_summary().contains("CCCC"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_genuinely_truncated_le_9fpr_files_under_header_size() {
        for truncated_len in [0usize, 3, 4, 12, 15, 16, 20, 23] {
            let mut buf = make_raw_words_header([
                RPF9_MAGIC_U32,
                0x360,
                0x340,
                0x40,
                0x24E6_3799,
                0xB766_DF72,
            ])
            .to_vec();
            buf.truncate(truncated_len);
            let err = inspect_rpf_bytes(&buf, truncated_len as u64).unwrap_err();
            assert!(
                matches!(
                    err,
                    RpfError::TruncatedHeader {
                        actual,
                        expected: RPF9_HEADER_SIZE
                    } if actual == truncated_len
                ),
                "len={truncated_len} gave unexpected err: {err:?}"
            );
        }

        // Full 24-byte header slice but reported total_size < 24 is still rejected.
        let header = make_raw_words_header([
            RPF9_MAGIC_U32,
            0x360,
            0x340,
            0x40,
            0x24E6_3799,
            0xB766_DF72,
        ]);
        let err = inspect_rpf_bytes(&header, 20).unwrap_err();
        assert!(matches!(
            err,
            RpfError::TruncatedArchive {
                required_bytes: 24,
                actual_bytes: 20
            }
        ));
    }

    #[test]
    fn generates_bootset_clamping_file_sizes() {
        let dir = unique_temp_dir("bootset");
        let game = dir.join("game");
        let sub = game.join("x64").join("audio");
        fs::create_dir_all(&sub).unwrap();

        // Larger than head_bytes (clamped down to 65536)
        fs::write(game.join("levels.rpf"), vec![0u8; 100_000]).unwrap();
        // Smaller than head_bytes (clamped to real size 576, mixed-case .Config)
        fs::write(game.join("MicrosoftGame.Config"), vec![b'x'; 576]).unwrap();
        // Nested uppercase .RPF smaller than head_bytes
        fs::write(sub.join("sfx.RPF"), vec![1u8; 4096]).unwrap();
        // Empty .rpf should be skipped (0-byte range is invalid)
        fs::write(game.join("empty.rpf"), b"").unwrap();
        // Non-matching files should be ignored
        fs::write(game.join("game.exe"), vec![0u8; 8192]).unwrap();
        fs::write(game.join("readme.txt"), b"hello").unwrap();

        let entries = generate_bootset(&game, DEFAULT_BOOTSET_HEAD_BYTES).unwrap();
        assert_eq!(
            entries,
            vec![
                BootsetEntry {
                    path: "MicrosoftGame.Config".to_string(),
                    offset: 0,
                    length: 576,
                },
                BootsetEntry {
                    path: "levels.rpf".to_string(),
                    offset: 0,
                    length: 65_536,
                },
                BootsetEntry {
                    path: "x64/audio/sfx.RPF".to_string(),
                    offset: 0,
                    length: 4096,
                },
            ]
        );

        let out_file = dir.join("out").join("bootset.json");
        let count = write_bootset(&game, &out_file, DEFAULT_BOOTSET_HEAD_BYTES).unwrap();
        assert_eq!(count, 3);

        let json = fs::read_to_string(&out_file).unwrap();
        assert!(json.starts_with("[\n"));
        assert!(json.ends_with("]\n"));
        assert!(json.contains("{\"path\": \"MicrosoftGame.Config\", \"offset\": 0, \"length\": 576},"));
        assert!(json.contains("{\"path\": \"levels.rpf\", \"offset\": 0, \"length\": 65536},"));
        assert!(json.contains("{\"path\": \"x64/audio/sfx.RPF\", \"offset\": 0, \"length\": 4096}\n"));

        let _ = fs::remove_dir_all(&dir);
    }

    fn encode_dir_entry(
        name_offset: u32,
        first_child: u32,
        child_count: u32,
        reserved: u32,
    ) -> [u8; 16] {
        let mut out = [0u8; 16];
        let w0 = (u32::from(RPF9_ENTRY_FLAG_DIRECTORY) << 24) | (name_offset & 0x00FF_FFFF);
        out[0..4].copy_from_slice(&w0.to_le_bytes());
        out[4..8].copy_from_slice(&first_child.to_le_bytes());
        out[8..12].copy_from_slice(&child_count.to_le_bytes());
        out[12..16].copy_from_slice(&reserved.to_le_bytes());
        out
    }

    fn encode_file_entry(
        name_offset: u32,
        flags: u8,
        data_offset: u32,
        uncompressed_size: u32,
        compressed_size: u32,
    ) -> [u8; 16] {
        let mut out = [0u8; 16];
        let w0 = (u32::from(flags) << 24) | (name_offset & 0x00FF_FFFF);
        out[0..4].copy_from_slice(&w0.to_le_bytes());
        out[4..8].copy_from_slice(&data_offset.to_le_bytes());
        out[8..12].copy_from_slice(&uncompressed_size.to_le_bytes());
        out[12..16].copy_from_slice(&compressed_size.to_le_bytes());
        out
    }

    fn build_synthetic_rpf9_archive(
        magic: [u8; 4],
        entries: &[[u8; 16]],
        names_table: &[u8],
        metadata: &[u8],
        payload: &[u8],
    ) -> Vec<u8> {
        let header = make_rpf_header(
            magic,
            entries.len() as u32,
            names_table.len() as u32,
            0x4E45_504F,
            metadata.len() as u32,
            0,
        );
        let mut out = Vec::with_capacity(
            RPF9_HEADER_SIZE
                + entries.len() * RPF9_ENTRY_STRIDE as usize
                + names_table.len()
                + metadata.len()
                + payload.len(),
        );
        out.extend_from_slice(&header);
        for entry in entries {
            out.extend_from_slice(entry);
        }
        out.extend_from_slice(names_table);
        out.extend_from_slice(metadata);
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn rejects_truncated_headers_and_truncated_toc_buffers() {
        for magic in [RPF9_MAGIC, RPF9_MAGIC_LE] {
            let full_header = make_rpf_header(magic, 2, 16, 0x4E45_504F, 0, 0);
            for len in 0..RPF9_HEADER_SIZE {
                let err = inspect_rpf_bytes(&full_header[..len], len as u64).unwrap_err();
                assert!(
                    matches!(
                        err,
                        RpfError::TruncatedHeader {
                            actual,
                            expected: RPF9_HEADER_SIZE
                        } if actual == len
                    ),
                    "magic={magic:?} len={len} gave unexpected err: {err:?}"
                );
            }
        }

        // Valid header claiming 2 entries (32 bytes) + 16-byte names table in a 1024-byte archive,
        // but only the 24-byte header slice is passed to parse_rpf_toc.
        let header_only = make_rpf_header(RPF9_MAGIC, 2, 16, 0x4E45_504F, 0, 0);
        let err = parse_rpf_toc(&header_only, 1024).unwrap_err();
        assert!(matches!(
            err,
            RpfError::SliceOutOfBounds {
                section: "entry table",
                offset: 24,
                length: 32,
                available: 24
            }
        ));

        // Buffer has header + full entry table, but is truncated before the end of the names table.
        let mut short_names = Vec::new();
        short_names.extend_from_slice(&header_only);
        short_names.extend_from_slice(&[0u8; 32]); // 2 entries
        short_names.extend_from_slice(&[0u8; 8]); // only 8 of 16 name bytes
        let err = parse_rpf_toc(&short_names, 1024).unwrap_err();
        assert!(matches!(
            err,
            RpfError::SliceOutOfBounds {
                section: "names table",
                offset: 56,
                length: 16,
                available: 64
            }
        ));
    }

    #[test]
    fn rejects_overflowing_and_excessive_entry_counts_without_oom() {
        // 0x1000_0000 * 16 would wrap to 0 in unchecked u32 arithmetic.
        // Checked 64-bit arithmetic must report 4_294_967_320 required bytes instead of wrapping.
        let wrap_u32 = make_rpf_header(RPF9_MAGIC, 0x1000_0000, 0, 0x4E45_504F, 0, 0);
        let err = inspect_rpf_bytes(&wrap_u32, 1024).unwrap_err();
        assert!(matches!(
            err,
            RpfError::TruncatedArchive {
                required_bytes: 4_294_967_320,
                actual_bytes: 1024
            }
        ));

        // Even if total_size is spoofed to u64::MAX, huge entry counts must be rejected before
        // any allocation happens.
        let huge_entries = make_rpf_header(RPF9_MAGIC, u32::MAX, 0, 0x4E45_504F, 0, 0);
        let err = inspect_rpf_bytes(&huge_entries, u64::MAX).unwrap_err();
        assert!(matches!(
            err,
            RpfError::EntryCountExceedsLimit {
                entry_count: u32::MAX,
                max_entries: RPF9_MAX_ENTRY_COUNT
            }
        ));

        let just_over_entries = make_rpf_header(
            RPF9_MAGIC,
            RPF9_MAX_ENTRY_COUNT + 1,
            0,
            0x4E45_504F,
            0,
            0,
        );
        let err = parse_rpf_toc(&just_over_entries, u64::MAX).unwrap_err();
        assert!(matches!(
            err,
            RpfError::EntryCountExceedsLimit {
                entry_count,
                max_entries: RPF9_MAX_ENTRY_COUNT
            } if entry_count == RPF9_MAX_ENTRY_COUNT + 1
        ));

        // Oversized names table and metadata sections are rejected cleanly too.
        let huge_names = make_rpf_header(RPF9_MAGIC, 0, RPF9_MAX_NAMES_SIZE + 1, 0, 0, 0);
        let err = inspect_rpf_bytes(&huge_names, u64::MAX).unwrap_err();
        assert!(matches!(
            err,
            RpfError::NamesSizeExceedsLimit {
                names_size,
                max_names_size: RPF9_MAX_NAMES_SIZE
            } if names_size == RPF9_MAX_NAMES_SIZE + 1
        ));

        let huge_meta = make_rpf_header(RPF9_MAGIC, 0, 0, 0, RPF9_MAX_METADATA_SIZE + 1, 0);
        let err = inspect_rpf_bytes(&huge_meta, u64::MAX).unwrap_err();
        assert!(matches!(
            err,
            RpfError::MetadataSizeExceedsLimit {
                metadata_size,
                max_metadata_size: RPF9_MAX_METADATA_SIZE
            } if metadata_size == RPF9_MAX_METADATA_SIZE + 1
        ));

        // Direct checked_range_end overflow test.
        assert!(matches!(
            checked_range_end(u64::MAX, 1),
            Err(RpfError::SizeOverflow)
        ));
    }

    #[test]
    fn rejects_overflowing_directory_child_ranges_and_cycles() {
        let names = b"\0dir\0";

        // Directory child range first_child + child_count overflows u32.
        let overflow_entries = [encode_dir_entry(0, u32::MAX, 2, 0)];
        let archive = build_synthetic_rpf9_archive(RPF9_MAGIC, &overflow_entries, names, &[], &[]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(err, RpfError::SizeOverflow));

        // Directory child range exceeds entry_count.
        let oob_entries = [encode_dir_entry(0, 1, 4, 0), encode_dir_entry(1, 0, 0, 0)];
        let archive = build_synthetic_rpf9_archive(RPF9_MAGIC, &oob_entries, names, &[], &[]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::ChildRangeOutOfBounds {
                entry_index: 0,
                first_child: 1,
                child_count: 4,
                entry_count: 2
            }
        ));

        // Self-referencing directory cycle (dir 1 lists entry 0 as its child).
        let cycle_entries = [encode_dir_entry(0, 1, 1, 0), encode_dir_entry(1, 0, 1, 0)];
        let archive = build_synthetic_rpf9_archive(RPF9_MAGIC, &cycle_entries, names, &[], &[]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::TocCycleOrDuplicateRef { entry_index: 0 }
        ));
    }

    #[test]
    fn rejects_out_of_bounds_unterminated_and_invalid_names() {
        // Out-of-bounds name_offset at exact end of names table and strictly past end.
        let names = b"\0ok.bin\0";
        for bad_offset in [names.len() as u32, (names.len() as u32) + 100] {
            let oob_name_entries = [
                encode_dir_entry(0, 1, 1, 0),
                encode_file_entry(bad_offset, 0, 0, 0, 0),
            ];
            let archive =
                build_synthetic_rpf9_archive(RPF9_MAGIC, &oob_name_entries, names, &[], &[]);
            let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
            assert!(matches!(
                err,
                RpfError::NameOffsetOutOfBounds {
                    entry_index: 1,
                    name_offset,
                    names_size: 8
                } if name_offset == bad_offset
            ));
        }

        // Overlong name exceeding RPF9_MAX_NAME_LEN must be rejected.
        let mut overlong_names = vec![0u8];
        overlong_names.extend_from_slice(&vec![b'a'; RPF9_MAX_NAME_LEN + 2]);
        overlong_names.push(0);
        let overlong_entries = [
            encode_dir_entry(0, 1, 1, 0),
            encode_file_entry(1, 0, 0, 0, 0),
        ];
        let archive =
            build_synthetic_rpf9_archive(RPF9_MAGIC, &overlong_entries, &overlong_names, &[], &[]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::UnterminatedName {
                entry_index: 1,
                name_offset: 1,
                max_len: RPF9_MAX_NAME_LEN
            }
        ));

        // Missing NUL terminator at end of names table.
        let unterminated = b"\0no_nul_here";
        let entries = [
            encode_dir_entry(0, 1, 1, 0),
            encode_file_entry(1, 0, 0, 0, 0),
        ];
        let archive =
            build_synthetic_rpf9_archive(RPF9_MAGIC, &entries, unterminated, &[], &[]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::UnterminatedName {
                entry_index: 1,
                name_offset: 1,
                max_len: RPF9_MAX_NAME_LEN
            }
        ));

        // Path traversal segment ".." must be rejected.
        let traversal = b"\0..\0";
        let archive = build_synthetic_rpf9_archive(RPF9_MAGIC, &entries, traversal, &[], &[]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::InvalidEntryName {
                entry_index: 1,
                ..
            }
        ));

        // Embedded path separator in a single entry name must be rejected.
        let slash = b"\0sub/evil.bin\0";
        let archive = build_synthetic_rpf9_archive(RPF9_MAGIC, &entries, slash, &[], &[]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::InvalidEntryName {
                entry_index: 1,
                ..
            }
        ));
    }

    #[test]
    fn rejects_out_of_bounds_and_header_overlapping_file_payloads() {
        let names = b"\0a.bin\0";
        let data_offset = (RPF9_HEADER_SIZE + 2 * RPF9_ENTRY_STRIDE as usize + names.len()) as u32;

        // Payload extends past end of archive.
        let oob_entries = [
            encode_dir_entry(0, 1, 1, 0),
            encode_file_entry(1, 0, data_offset, 32, 0),
        ];
        let archive =
            build_synthetic_rpf9_archive(RPF9_MAGIC, &oob_entries, names, &[], &[0xAA; 16]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::EntryDataOutOfBounds {
                entry_index: 1,
                length: 32,
                ..
            }
        ));

        // Non-empty file payload pointing into header/TOC region (< layout.data_offset).
        let overlap_entries = [
            encode_dir_entry(0, 1, 1, 0),
            encode_file_entry(1, 0, 8, 4, 0),
        ];
        let archive =
            build_synthetic_rpf9_archive(RPF9_MAGIC, &overlap_entries, names, &[], &[0xBB; 16]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::EntryDataOutOfBounds {
                entry_index: 1,
                offset: 8,
                length: 4,
                ..
            }
        ));

        // Sector-shifted offset exceeding archive size.
        let sector_oob = [
            encode_dir_entry(0, 1, 1, 0),
            encode_file_entry(1, RPF9_ENTRY_FLAG_OFFSET_SHIFT9, 100, 16, 0),
        ];
        let archive =
            build_synthetic_rpf9_archive(RPF9_MAGIC, &sector_oob, names, &[], &[0xCC; 16]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::EntryDataOutOfBounds {
                entry_index: 1,
                offset: 51_200,
                length: 16,
                ..
            }
        ));
    }

    #[test]
    fn deterministic_canonical_toc_listing_and_checked_payload_reads() {
        // Archive A: entries and names in unsorted order (z.bin before a_dir, m.dat before b.dat).
        let names_a = b"\0z.bin\0A_Dir\0M.dat\0b.dat\0";
        // offsets: ""=0, "z.bin"=1, "A_Dir"=7, "M.dat"=13, "b.dat"=19
        let table_end_a = RPF9_HEADER_SIZE + 5 * RPF9_ENTRY_STRIDE as usize + names_a.len();
        let pad_to_sector = (512 - (table_end_a % 512)) % 512;
        let payload_z_off = table_end_a as u32;
        let payload_m_off = payload_z_off + 5;
        let sector_b_idx = ((table_end_a + pad_to_sector) / 512) as u32;

        let mut payload_a = vec![0u8; pad_to_sector + 4];
        payload_a[0..5].copy_from_slice(b"ZZZZZ");
        payload_a[5..8].copy_from_slice(b"MMM");
        payload_a[pad_to_sector..pad_to_sector + 4].copy_from_slice(b"BBBB");

        let entries_a = [
            encode_dir_entry(0, 1, 2, 0),          // root: [z.bin, A_Dir]
            encode_file_entry(1, 0, payload_z_off, 5, 0), // z.bin
            encode_dir_entry(7, 3, 2, 0),          // A_Dir: [M.dat, b.dat]
            encode_file_entry(13, 0, payload_m_off, 3, 0), // M.dat
            encode_file_entry(19, RPF9_ENTRY_FLAG_OFFSET_SHIFT9, sector_b_idx, 4, 0), // b.dat
        ];
        let archive_a =
            build_synthetic_rpf9_archive(RPF9_MAGIC_LE, &entries_a, names_a, &[], &payload_a);

        // Archive B: exact same logical tree, completely permuted raw TOC & names order.
        let names_b = b"\0b.dat\0M.dat\0A_Dir\0z.bin\0";
        // offsets: ""=0, "b.dat"=1, "M.dat"=7, "A_Dir"=13, "z.bin"=19
        let table_end_b = RPF9_HEADER_SIZE + 5 * RPF9_ENTRY_STRIDE as usize + names_b.len();
        let payload_b = b"BBBBMMMZZZZZ";
        let off_b = table_end_b as u32;
        let off_m = off_b + 4;
        let off_z = off_m + 3;

        let entries_b = [
            encode_dir_entry(0, 1, 2, 0),          // root: [A_Dir, z.bin]
            encode_dir_entry(13, 3, 2, 0),         // A_Dir: [b.dat, M.dat]
            encode_file_entry(19, 0, off_z, 5, 0), // z.bin
            encode_file_entry(1, 0, off_b, 4, 0),  // b.dat
            encode_file_entry(7, 0, off_m, 3, 0),  // M.dat
        ];
        let archive_b =
            build_synthetic_rpf9_archive(RPF9_MAGIC, &entries_b, names_b, &[], payload_b);

        let toc_a = parse_rpf_toc(&archive_a, archive_a.len() as u64).unwrap();
        let toc_b = parse_rpf_toc(&archive_b, archive_b.len() as u64).unwrap();

        let expected_paths = vec!["a_dir", "a_dir/b.dat", "a_dir/m.dat", "z.bin"];
        assert_eq!(toc_a.list_paths(), expected_paths);
        assert_eq!(toc_b.list_paths(), expected_paths);

        assert_eq!(toc_a.list_dir(""), vec!["a_dir", "z.bin"]);
        assert_eq!(toc_b.list_dir(""), vec!["a_dir", "z.bin"]);
        assert_eq!(toc_a.list_dir("A_Dir"), vec!["a_dir/b.dat", "a_dir/m.dat"]);
        assert_eq!(toc_b.list_dir("a_dir\\"), vec!["a_dir/b.dat", "a_dir/m.dat"]);

        // Bounds-checked payload reads match across both raw layouts.
        assert_eq!(
            read_rpf_entry_bytes(&archive_a, toc_a.get("Z.BIN").unwrap()).unwrap(),
            b"ZZZZZ"
        );
        assert_eq!(
            read_rpf_entry_bytes(&archive_a, toc_a.get("a_dir/b.dat").unwrap()).unwrap(),
            b"BBBB"
        );
        assert_eq!(
            read_rpf_entry_bytes(&archive_b, toc_b.get("a_dir/m.dat").unwrap()).unwrap(),
            b"MMM"
        );

        // Reading payload from a truncated buffer returns SliceOutOfBounds instead of panicking.
        let truncated_err =
            read_rpf_entry_bytes(&archive_a[..table_end_a], toc_a.get("z.bin").unwrap())
                .unwrap_err();
        assert!(matches!(
            truncated_err,
            RpfError::SliceOutOfBounds {
                section: "entry payload",
                ..
            }
        ));
    }

    #[test]
    fn rejects_duplicate_canonical_paths_and_xbox_v9_classic_toc_parse() {
        // Case-insensitive sibling collision ("Data.bin" vs "data.bin").
        let names = b"\0Data.bin\0data.bin\0";
        let entries = [
            encode_dir_entry(0, 1, 2, 0),
            encode_file_entry(1, 0, 0, 0, 0),
            encode_file_entry(10, 0, 0, 0, 0),
        ];
        let archive = build_synthetic_rpf9_archive(RPF9_MAGIC, &entries, names, &[], &[]);
        let err = parse_rpf_toc(&archive, archive.len() as u64).unwrap_err();
        assert!(matches!(
            err,
            RpfError::DuplicateCanonicalPath {
                ref path,
                first_index: 1,
                duplicate_index: 2
            } if path == "data.bin"
        ));

        // XboxSeriesV9 header is cleanly rejected by classic TOC parser without panicking.
        let v9_header = make_raw_words_header([
            RPF9_MAGIC_U32,
            0x0000_0360,
            0x0000_0340,
            0x0000_0040,
            0x24E6_3799,
            0xB766_DF72,
        ]);
        let err = parse_rpf_toc(&v9_header, 8_880).unwrap_err();
        assert!(matches!(
            err,
            RpfError::UnsupportedHeaderLayout {
                layout: Rpf9HeaderLayout::XboxSeriesV9
            }
        ));
    }
}
