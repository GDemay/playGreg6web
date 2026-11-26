use std::fmt;
use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

use crate::rpf9::json_escape;
use crate::scan::format_bytes;

const ABI_VERSION: u32 = 1;

/// Standard 8-byte ASCII magic identifying an Xbox Virtual Disk (`XVD` / `XVC`) header.
pub const XVD_MAGIC: &[u8; 8] = b"msft-xvd";

/// Standard byte offset of the `msft-xvd` header following the 512-byte signature block.
pub const XVD_STANDARD_MAGIC_OFFSET: usize = 0x200;

/// Raw byte offset when an XVD header starts directly at `msft-xvd` without a signature block.
pub const XVD_RAW_MAGIC_OFFSET: usize = 0x0;

/// Full documented XVD header region size (3 KiB = `0xC00` bytes).
pub const XVD_HEADER_SIZE: usize = 0xC00;

/// Standard XVD page / sector alignment in bytes (4 KiB).
pub const XVD_PAGE_SIZE: u64 = 0x1000;

/// Legacy XVD sector alignment in bytes when [`FLAG_LEGACY_SECTOR_SIZE`] is set (512 bytes).
pub const XVD_LEGACY_SECTOR_SIZE: u64 = 0x200;

/// Minimum supported XVD block size in bytes (512-byte sector).
pub const XVD_MIN_BLOCK_SIZE: u32 = 512;

/// Maximum supported XVD block size in bytes (64 MiB safety ceiling).
pub const XVD_MAX_BLOCK_SIZE: u32 = 64 * 1024 * 1024;

/// Minimum bytes required starting from `magic_offset` to parse all documented header fields
/// through `block_size` (`+0x80..+0x84`).
pub const XVD_MIN_FIELDS_BYTES: usize = 0x84;

/// Minimum bytes required to parse a standard XVD header (`0x200` signature + `0x84` fields).
pub const XVD_STANDARD_MIN_BYTES: usize = XVD_STANDARD_MAGIC_OFFSET + XVD_MIN_FIELDS_BYTES;

/// Relative field offsets from `magic_offset` (so standard offsets are `0x200 + rel`).
const OFF_VOLUME_FLAGS: usize = 0x08; // 0x208
const OFF_FORMAT_VERSION: usize = 0x0C; // 0x20C
const OFF_FILE_TIME: usize = 0x10; // 0x210
const OFF_DRIVE_SIZE: usize = 0x18; // 0x218
const OFF_VD_UID: usize = 0x20; // 0x220
const OFF_UD_UID: usize = 0x30; // 0x230
const OFF_CONTENT_TYPE: usize = 0x6C; // 0x26C
const OFF_EMBEDDED_XVD_LENGTH: usize = 0x70; // 0x270
const OFF_USER_DATA_LENGTH: usize = 0x74; // 0x274
const OFF_XVC_DATA_LENGTH: usize = 0x78; // 0x278
const OFF_DYNAMIC_HEADER_LENGTH: usize = 0x7C; // 0x27C
const OFF_BLOCK_SIZE: usize = 0x80; // 0x280

/// Documented XVD `volume_flags` bitmasks (`u32` LE at offset `0x208`).
pub const FLAG_READ_ONLY: u32 = 1 << 0;
pub const FLAG_ENCRYPTION_DISABLED: u32 = 1 << 1;
pub const FLAG_DATA_INTEGRITY_DISABLED: u32 = 1 << 2;
pub const FLAG_LEGACY_SECTOR_SIZE: u32 = 1 << 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VolumeFlags {
    pub raw: u32,
    pub read_only: bool,
    pub encryption_disabled: bool,
    pub data_integrity_disabled: bool,
    pub legacy_sector_size: bool,
}

impl VolumeFlags {
    pub fn from_raw(raw: u32) -> Self {
        Self {
            raw,
            read_only: (raw & FLAG_READ_ONLY) != 0,
            encryption_disabled: (raw & FLAG_ENCRYPTION_DISABLED) != 0,
            data_integrity_disabled: (raw & FLAG_DATA_INTEGRITY_DISABLED) != 0,
            legacy_sector_size: (raw & FLAG_LEGACY_SECTOR_SIZE) != 0,
        }
    }

    pub fn active_names(&self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.read_only {
            names.push("ReadOnly");
        }
        if self.encryption_disabled {
            names.push("EncryptionDisabled");
        }
        if self.data_integrity_disabled {
            names.push("DataIntegrityDisabled");
        }
        if self.legacy_sector_size {
            names.push("LegacySectorSize");
        }
        names
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    Data,
    Title,
    SystemOS,
    EraOS,
    Scratch,
    ResetData,
    Application,
    HostOS,
    X360STFS,
    X360FATX,
    X360GDFX,
    Updater,
    OfflineUpdater,
    Template,
    MteHost,
    MteApp,
    MteTitle,
    MteEraOS,
    AppCompat,
    Unknown(u32),
}

impl ContentType {
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::Data,
            1 => Self::Title,
            2 => Self::SystemOS,
            3 => Self::EraOS,
            4 => Self::Scratch,
            5 => Self::ResetData,
            6 => Self::Application,
            7 => Self::HostOS,
            8 => Self::X360STFS,
            9 => Self::X360FATX,
            10 => Self::X360GDFX,
            11 => Self::Updater,
            12 => Self::OfflineUpdater,
            13 => Self::Template,
            14 => Self::MteHost,
            15 => Self::MteApp,
            16 => Self::MteTitle,
            17 => Self::MteEraOS,
            18 => Self::AppCompat,
            other => Self::Unknown(other),
        }
    }

    pub fn raw_id(self) -> u32 {
        match self {
            Self::Data => 0,
            Self::Title => 1,
            Self::SystemOS => 2,
            Self::EraOS => 3,
            Self::Scratch => 4,
            Self::ResetData => 5,
            Self::Application => 6,
            Self::HostOS => 7,
            Self::X360STFS => 8,
            Self::X360FATX => 9,
            Self::X360GDFX => 10,
            Self::Updater => 11,
            Self::OfflineUpdater => 12,
            Self::Template => 13,
            Self::MteHost => 14,
            Self::MteApp => 15,
            Self::MteTitle => 16,
            Self::MteEraOS => 17,
            Self::AppCompat => 18,
            Self::Unknown(n) => n,
        }
    }
}

impl fmt::Display for ContentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data => f.write_str("Data"),
            Self::Title => f.write_str("Title"),
            Self::SystemOS => f.write_str("SystemOS"),
            Self::EraOS => f.write_str("EraOS"),
            Self::Scratch => f.write_str("Scratch"),
            Self::ResetData => f.write_str("ResetData"),
            Self::Application => f.write_str("Application"),
            Self::HostOS => f.write_str("HostOS"),
            Self::X360STFS => f.write_str("X360STFS"),
            Self::X360FATX => f.write_str("X360FATX"),
            Self::X360GDFX => f.write_str("X360GDFX"),
            Self::Updater => f.write_str("Updater"),
            Self::OfflineUpdater => f.write_str("OfflineUpdater"),
            Self::Template => f.write_str("Template"),
            Self::MteHost => f.write_str("MteHost"),
            Self::MteApp => f.write_str("MteApp"),
            Self::MteTitle => f.write_str("MteTitle"),
            Self::MteEraOS => f.write_str("MteEraOS"),
            Self::AppCompat => f.write_str("AppCompat"),
            Self::Unknown(n) => write!(f, "Unknown({n})"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SizeStatus {
    Complete,
    PartialOrTruncated,
    Undeclared,
}

impl SizeStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Complete => "complete",
            Self::PartialOrTruncated => "partial_or_truncated",
            Self::Undeclared => "undeclared",
        }
    }
}

impl fmt::Display for SizeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Canonical segment kinds present in an Xbox `XVD`/`XVC` package layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum XvdSegmentKind {
    Header,
    EmbeddedXvd,
    XvcData,
    DynamicHeader,
    UserData,
    DriveData,
}

impl XvdSegmentKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Header => "header",
            Self::EmbeddedXvd => "embedded_xvd",
            Self::XvcData => "xvc_data",
            Self::DynamicHeader => "dynamic_header",
            Self::UserData => "user_data",
            Self::DriveData => "drive_data",
        }
    }
}

impl fmt::Display for XvdSegmentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Bounds-checked byte and block span of a single XVD/XVC container segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XvdSegment {
    pub kind: XvdSegmentKind,
    pub offset: u64,
    pub length: u64,
    pub aligned_length: u64,
    pub end: u64,
    pub aligned_end: u64,
    pub block_count: u64,
    pub within_file: bool,
}

/// Checked, monotonically ordered, non-overlapping segment layout derived from an XVD header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XvdSegmentLayout {
    pub page_size: u64,
    pub effective_block_size: u32,
    pub metadata_end: u64,
    pub total_declared_bytes: u64,
    pub drive_block_count: u64,
    pub segments: Vec<XvdSegment>,
}

impl XvdSegmentLayout {
    /// Returns the segment descriptor for `kind`, if declared with a non-zero length.
    #[allow(dead_code)]
    pub fn segment(&self, kind: XvdSegmentKind) -> Option<&XvdSegment> {
        self.segments.iter().find(|seg| seg.kind == kind)
    }

    /// Returns `true` if every declared segment fits within the inspected file size.
    pub fn all_segments_within_file(&self) -> bool {
        self.segments.iter().all(|seg| seg.within_file)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XvdReport {
    pub file_name: String,
    pub file_size: u64,
    pub magic_offset: usize,
    pub volume_flags: VolumeFlags,
    pub is_encrypted: bool,
    pub format_version: u32,
    pub file_time: u64,
    pub drive_size: u64,
    pub vd_uid: [u8; 16],
    pub ud_uid: [u8; 16],
    pub content_type: ContentType,
    pub embedded_xvd_length: u32,
    pub user_data_length: u32,
    pub xvc_data_length: u32,
    pub dynamic_header_length: u32,
    pub block_size: u32,
    pub size_status: SizeStatus,
    pub layout: XvdSegmentLayout,
}

#[derive(Debug)]
pub enum XvdError {
    Io(io::Error),
    TruncatedHeader {
        actual: usize,
        expected: usize,
    },
    InvalidMagic {
        found_at_0x200: [u8; 8],
    },
    InvalidBlockSize {
        block_size: u32,
        reason: &'static str,
    },
    ArithmeticOverflow {
        context: &'static str,
    },
    #[allow(dead_code)]
    SegmentOutOfBounds {
        segment: XvdSegmentKind,
        offset: u64,
        length: u64,
        end: u64,
        limit: u64,
    },
    #[allow(dead_code)]
    BlockOutOfBounds {
        block_index: u64,
        block_count: u64,
        total_blocks: u64,
    },
    #[allow(dead_code)]
    InvalidBufferRange {
        offset: usize,
        length: usize,
        buffer_len: usize,
    },
}

fn io_error_kind_label(kind: io::ErrorKind) -> &'static str {
    match kind {
        io::ErrorKind::NotFound => "not_found",
        io::ErrorKind::PermissionDenied => "permission_denied",
        io::ErrorKind::ConnectionRefused => "connection_refused",
        io::ErrorKind::ConnectionReset => "connection_reset",
        io::ErrorKind::ConnectionAborted => "connection_aborted",
        io::ErrorKind::NotConnected => "not_connected",
        io::ErrorKind::BrokenPipe => "broken_pipe",
        io::ErrorKind::AlreadyExists => "already_exists",
        io::ErrorKind::WouldBlock => "would_block",
        io::ErrorKind::InvalidInput => "invalid_input",
        io::ErrorKind::InvalidData => "invalid_data",
        io::ErrorKind::TimedOut => "timed_out",
        io::ErrorKind::WriteZero => "write_zero",
        io::ErrorKind::Interrupted => "interrupted",
        io::ErrorKind::Unsupported => "unsupported",
        io::ErrorKind::UnexpectedEof => "unexpected_eof",
        io::ErrorKind::OutOfMemory => "out_of_memory",
        _ => "io_error",
    }
}

impl fmt::Display for XvdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Never forward raw OS error strings in Display so host paths cannot leak.
            Self::Io(err) => write!(
                f,
                "I/O error reading XVD/XVC container ({})",
                io_error_kind_label(err.kind())
            ),
            Self::TruncatedHeader { actual, expected } => write!(
                f,
                "truncated XVD/XVC header: got {actual} bytes, need at least {expected} bytes"
            ),
            Self::InvalidMagic { found_at_0x200 } => write!(
                f,
                "invalid XVD/XVC magic at offset 0x200: {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} {:02X} (expected \"msft-xvd\")",
                found_at_0x200[0],
                found_at_0x200[1],
                found_at_0x200[2],
                found_at_0x200[3],
                found_at_0x200[4],
                found_at_0x200[5],
                found_at_0x200[6],
                found_at_0x200[7]
            ),
            Self::InvalidBlockSize { block_size, reason } => write!(
                f,
                "invalid XVD/XVC block size {block_size}: {reason}"
            ),
            Self::ArithmeticOverflow { context } => write!(
                f,
                "XVD/XVC arithmetic overflow while computing {context}"
            ),
            Self::SegmentOutOfBounds {
                segment,
                offset,
                length,
                end,
                limit,
            } => write!(
                f,
                "XVD/XVC segment '{segment}' [0x{offset:X}..0x{end:X}) ({length} bytes) exceeds bounds ({limit} bytes)"
            ),
            Self::BlockOutOfBounds {
                block_index,
                block_count,
                total_blocks,
            } => write!(
                f,
                "XVD/XVC block range [{block_index}..+{block_count}) exceeds total block count ({total_blocks})"
            ),
            Self::InvalidBufferRange {
                offset,
                length,
                buffer_len,
            } => write!(
                f,
                "XVD/XVC buffer range [0x{offset:X}..+{length}) exceeds buffer length ({buffer_len} bytes)"
            ),
        }
    }
}

impl std::error::Error for XvdError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}

impl From<io::Error> for XvdError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

/// Formats a 16-byte Microsoft/Xbox GUID (mixed-endian `Data1`/`Data2`/`Data3` LE, `Data4` raw)
/// as a standard lowercase RFC 4122 UUID string (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
pub fn format_uuid(bytes: &[u8; 16]) -> String {
    let d1 = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let d2 = u16::from_le_bytes([bytes[4], bytes[5]]);
    let d3 = u16::from_le_bytes([bytes[6], bytes[7]]);
    format!(
        "{d1:08x}-{d2:04x}-{d3:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}

fn read_slice<'a>(
    bytes: &'a [u8],
    base: usize,
    rel: usize,
    len: usize,
) -> Result<&'a [u8], XvdError> {
    let start = base.checked_add(rel).ok_or(XvdError::ArithmeticOverflow {
        context: "header_field_offset",
    })?;
    let end = start.checked_add(len).ok_or(XvdError::ArithmeticOverflow {
        context: "header_field_end",
    })?;
    bytes.get(start..end).ok_or(XvdError::TruncatedHeader {
        actual: bytes.len(),
        expected: end,
    })
}

fn read_u32_le(bytes: &[u8], base: usize, rel: usize) -> Result<u32, XvdError> {
    let slice = read_slice(bytes, base, rel, 4)?;
    let arr: [u8; 4] = slice.try_into().map_err(|_| XvdError::TruncatedHeader {
        actual: bytes.len(),
        expected: base.saturating_add(rel).saturating_add(4),
    })?;
    Ok(u32::from_le_bytes(arr))
}

fn read_u64_le(bytes: &[u8], base: usize, rel: usize) -> Result<u64, XvdError> {
    let slice = read_slice(bytes, base, rel, 8)?;
    let arr: [u8; 8] = slice.try_into().map_err(|_| XvdError::TruncatedHeader {
        actual: bytes.len(),
        expected: base.saturating_add(rel).saturating_add(8),
    })?;
    Ok(u64::from_le_bytes(arr))
}

fn read_array_8(bytes: &[u8], base: usize, rel: usize) -> Result<[u8; 8], XvdError> {
    let slice = read_slice(bytes, base, rel, 8)?;
    slice.try_into().map_err(|_| XvdError::TruncatedHeader {
        actual: bytes.len(),
        expected: base.saturating_add(rel).saturating_add(8),
    })
}

fn read_array_16(bytes: &[u8], base: usize, rel: usize) -> Result<[u8; 16], XvdError> {
    let slice = read_slice(bytes, base, rel, 16)?;
    slice.try_into().map_err(|_| XvdError::TruncatedHeader {
        actual: bytes.len(),
        expected: base.saturating_add(rel).saturating_add(16),
    })
}

/// Validates that `block_size` is a non-zero power of two within `[XVD_MIN_BLOCK_SIZE, XVD_MAX_BLOCK_SIZE]`.
pub fn validate_block_size(block_size: u32) -> Result<u32, XvdError> {
    if block_size == 0 {
        return Err(XvdError::InvalidBlockSize {
            block_size,
            reason: "block size must be non-zero",
        });
    }
    if !block_size.is_power_of_two() {
        return Err(XvdError::InvalidBlockSize {
            block_size,
            reason: "block size must be a power of two",
        });
    }
    if block_size < XVD_MIN_BLOCK_SIZE {
        return Err(XvdError::InvalidBlockSize {
            block_size,
            reason: "block size is below minimum sector size (512 bytes)",
        });
    }
    if block_size > XVD_MAX_BLOCK_SIZE {
        return Err(XvdError::InvalidBlockSize {
            block_size,
            reason: "block size exceeds maximum supported size (64 MiB)",
        });
    }
    Ok(block_size)
}

/// Aligns `value` upward to a power-of-two `alignment`, returning an overflow error instead of wrapping.
pub fn align_up_checked(value: u64, alignment: u64) -> Result<u64, XvdError> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(XvdError::InvalidBlockSize {
            block_size: u32::try_from(alignment).unwrap_or(u32::MAX),
            reason: "alignment must be a non-zero power of two",
        });
    }
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|aligned| aligned & !mask)
        .ok_or(XvdError::ArithmeticOverflow {
            context: "align_up",
        })
}

/// Computes the checked block count (`ceil(total_bytes / block_size)`) while ensuring the aligned
/// byte span (`block_count * block_size`) does not overflow `u64`.
pub fn checked_block_count(total_bytes: u64, block_size: u32) -> Result<u64, XvdError> {
    let bs = u64::from(validate_block_size(block_size)?);
    let aligned = align_up_checked(total_bytes, bs).map_err(|err| match err {
        XvdError::ArithmeticOverflow { .. } => XvdError::ArithmeticOverflow {
            context: "block_count_alignment",
        },
        other => other,
    })?;
    Ok(aligned / bs)
}

/// Computes the checked `(byte_offset, byte_length)` span for `[block_index..block_index + block_count)`
/// anchored at `base_offset`, guarding against index/multiplication/addition overflow and out-of-range blocks.
#[allow(dead_code)]
pub fn checked_block_byte_range(
    base_offset: u64,
    block_index: u64,
    block_count: u64,
    block_size: u32,
    total_blocks: u64,
) -> Result<(u64, u64), XvdError> {
    let bs = u64::from(validate_block_size(block_size)?);
    let end_block = block_index
        .checked_add(block_count)
        .ok_or(XvdError::ArithmeticOverflow {
            context: "block_index_range",
        })?;
    if end_block > total_blocks {
        return Err(XvdError::BlockOutOfBounds {
            block_index,
            block_count,
            total_blocks,
        });
    }
    let rel_offset = block_index
        .checked_mul(bs)
        .ok_or(XvdError::ArithmeticOverflow {
            context: "block_byte_offset",
        })?;
    let byte_offset = base_offset
        .checked_add(rel_offset)
        .ok_or(XvdError::ArithmeticOverflow {
            context: "block_absolute_offset",
        })?;
    let byte_length = block_count
        .checked_mul(bs)
        .ok_or(XvdError::ArithmeticOverflow {
            context: "block_byte_length",
        })?;
    let _byte_end = byte_offset
        .checked_add(byte_length)
        .ok_or(XvdError::ArithmeticOverflow {
            context: "block_byte_end",
        })?;
    Ok((byte_offset, byte_length))
}

/// Zero-panic slice accessor backed by checked `u64 -> usize` conversion and bounds checking.
#[allow(dead_code)]
pub fn checked_slice_range(buf: &[u8], offset: u64, length: u64) -> Result<&[u8], XvdError> {
    let start = usize::try_from(offset).map_err(|_| XvdError::ArithmeticOverflow {
        context: "buffer_slice_offset",
    })?;
    let len = usize::try_from(length).map_err(|_| XvdError::ArithmeticOverflow {
        context: "buffer_slice_length",
    })?;
    let end = start.checked_add(len).ok_or(XvdError::ArithmeticOverflow {
        context: "buffer_slice_end",
    })?;
    buf.get(start..end).ok_or(XvdError::InvalidBufferRange {
        offset: start,
        length: len,
        buffer_len: buf.len(),
    })
}

#[allow(clippy::too_many_arguments)]
fn compute_segment_layout(
    magic_offset: usize,
    volume_flags: VolumeFlags,
    raw_block_size: u32,
    embedded_xvd_length: u32,
    xvc_data_length: u32,
    dynamic_header_length: u32,
    user_data_length: u32,
    drive_size: u64,
    file_size: u64,
) -> Result<XvdSegmentLayout, XvdError> {
    let page_size = if volume_flags.legacy_sector_size {
        XVD_LEGACY_SECTOR_SIZE
    } else {
        XVD_PAGE_SIZE
    };
    let effective_block_size = if raw_block_size == 0 {
        u32::try_from(page_size).map_err(|_| XvdError::ArithmeticOverflow {
            context: "effective_block_size",
        })?
    } else {
        validate_block_size(raw_block_size)?
    };

    let header_len = if magic_offset == XVD_STANDARD_MAGIC_OFFSET {
        XVD_HEADER_SIZE as u64
    } else {
        XVD_HEADER_SIZE
            .checked_sub(XVD_STANDARD_MAGIC_OFFSET)
            .ok_or(XvdError::ArithmeticOverflow {
                context: "raw_header_length",
            })? as u64
    };

    let mut segments = Vec::with_capacity(6);
    let header_aligned_len = align_up_checked(header_len, page_size)?;
    let header_blocks = checked_block_count(header_len, effective_block_size)?;
    segments.push(XvdSegment {
        kind: XvdSegmentKind::Header,
        offset: 0,
        length: header_len,
        aligned_length: header_aligned_len,
        end: header_len,
        aligned_end: header_aligned_len,
        block_count: header_blocks,
        within_file: header_len <= file_size,
    });

    let mut cursor = header_aligned_len;
    let optional_segments: &[(XvdSegmentKind, u32)] = &[
        (XvdSegmentKind::EmbeddedXvd, embedded_xvd_length),
        (XvdSegmentKind::XvcData, xvc_data_length),
        (XvdSegmentKind::DynamicHeader, dynamic_header_length),
        (XvdSegmentKind::UserData, user_data_length),
    ];

    for &(kind, raw_len) in optional_segments {
        if raw_len == 0 {
            continue;
        }
        let length = u64::from(raw_len);
        let aligned_length = align_up_checked(length, page_size)?;
        let end = cursor
            .checked_add(length)
            .ok_or(XvdError::ArithmeticOverflow {
                context: "segment_end",
            })?;
        let aligned_end = cursor
            .checked_add(aligned_length)
            .ok_or(XvdError::ArithmeticOverflow {
                context: "segment_aligned_end",
            })?;
        let block_count = checked_block_count(length, effective_block_size)?;
        segments.push(XvdSegment {
            kind,
            offset: cursor,
            length,
            aligned_length,
            end,
            aligned_end,
            block_count,
            within_file: end <= file_size,
        });
        cursor = aligned_end;
    }

    let metadata_end = cursor;
    let drive_block_count = checked_block_count(drive_size, effective_block_size)?;
    let total_declared_bytes = if drive_size > 0 {
        let drive_aligned_len =
            align_up_checked(drive_size, u64::from(effective_block_size)).map_err(|err| {
                match err {
                    XvdError::ArithmeticOverflow { .. } => XvdError::ArithmeticOverflow {
                        context: "drive_aligned_length",
                    },
                    other => other,
                }
            })?;
        let drive_end = cursor
            .checked_add(drive_size)
            .ok_or(XvdError::ArithmeticOverflow {
                context: "drive_data_end",
            })?;
        let drive_aligned_end =
            cursor
                .checked_add(drive_aligned_len)
                .ok_or(XvdError::ArithmeticOverflow {
                    context: "drive_data_aligned_end",
                })?;
        segments.push(XvdSegment {
            kind: XvdSegmentKind::DriveData,
            offset: cursor,
            length: drive_size,
            aligned_length: drive_aligned_len,
            end: drive_end,
            aligned_end: drive_aligned_end,
            block_count: drive_block_count,
            within_file: drive_end <= file_size,
        });
        drive_end
    } else {
        metadata_end
    };

    Ok(XvdSegmentLayout {
        page_size,
        effective_block_size,
        metadata_end,
        total_declared_bytes,
        drive_block_count,
        segments,
    })
}

fn sanitize_file_name(path: &Path) -> String {
    let raw = path.to_string_lossy();
    let tail = raw
        .rsplit(['/', '\\'])
        .find(|seg| !seg.is_empty())
        .unwrap_or("");
    let without_drive = match tail.as_bytes() {
        [drive, b':', ..] if drive.is_ascii_alphabetic() => tail.get(2..).unwrap_or(""),
        _ => tail,
    };
    let cleaned: String = without_drive.chars().filter(|c| !c.is_control()).collect();
    if cleaned.is_empty() || cleaned == "." || cleaned == ".." {
        "unknown.xvd".to_string()
    } else {
        cleaned
    }
}

fn detect_magic_offset(bytes: &[u8]) -> Result<usize, XvdError> {
    let std_magic_end = XVD_STANDARD_MAGIC_OFFSET
        .checked_add(XVD_MAGIC.len())
        .ok_or(XvdError::ArithmeticOverflow {
            context: "standard_magic_end",
        })?;

    if let Some(std_slice) = bytes.get(XVD_STANDARD_MAGIC_OFFSET..std_magic_end)
        && std_slice == XVD_MAGIC
    {
        return Ok(XVD_STANDARD_MAGIC_OFFSET);
    }

    if let Some(raw_slice) = bytes.get(..XVD_MAGIC.len())
        && raw_slice == XVD_MAGIC
    {
        return Ok(XVD_RAW_MAGIC_OFFSET);
    }

    if bytes.len() < std_magic_end {
        return Err(XvdError::TruncatedHeader {
            actual: bytes.len(),
            expected: XVD_STANDARD_MIN_BYTES,
        });
    }

    let found = read_array_8(bytes, XVD_STANDARD_MAGIC_OFFSET, 0)?;
    Err(XvdError::InvalidMagic {
        found_at_0x200: found,
    })
}

/// Parses an Xbox `XVD`/`XVC` header from an in-memory prefix slice and known on-disk file size.
///
/// Strictly read-only metadata inspection: never decrypts, never reads key material, and never
/// dumps signature or payload bytes. Supports both standard containers (`msft-xvd` at offset
/// `0x200`) and raw headers (`msft-xvd` at offset `0x0`).
pub fn parse_xvd_header(bytes: &[u8], file_size: u64) -> Result<XvdReport, XvdError> {
    let magic_offset = detect_magic_offset(bytes)?;
    let required_bytes =
        magic_offset
            .checked_add(XVD_MIN_FIELDS_BYTES)
            .ok_or(XvdError::ArithmeticOverflow {
                context: "header_required_bytes",
            })?;
    if bytes.len() < required_bytes {
        return Err(XvdError::TruncatedHeader {
            actual: bytes.len(),
            expected: required_bytes,
        });
    }

    let volume_flags = VolumeFlags::from_raw(read_u32_le(bytes, magic_offset, OFF_VOLUME_FLAGS)?);
    let is_encrypted = !volume_flags.encryption_disabled;
    let format_version = read_u32_le(bytes, magic_offset, OFF_FORMAT_VERSION)?;
    let file_time = read_u64_le(bytes, magic_offset, OFF_FILE_TIME)?;
    let drive_size = read_u64_le(bytes, magic_offset, OFF_DRIVE_SIZE)?;
    let vd_uid = read_array_16(bytes, magic_offset, OFF_VD_UID)?;
    let ud_uid = read_array_16(bytes, magic_offset, OFF_UD_UID)?;
    let content_type = ContentType::from_u32(read_u32_le(bytes, magic_offset, OFF_CONTENT_TYPE)?);
    let embedded_xvd_length = read_u32_le(bytes, magic_offset, OFF_EMBEDDED_XVD_LENGTH)?;
    let user_data_length = read_u32_le(bytes, magic_offset, OFF_USER_DATA_LENGTH)?;
    let xvc_data_length = read_u32_le(bytes, magic_offset, OFF_XVC_DATA_LENGTH)?;
    let dynamic_header_length = read_u32_le(bytes, magic_offset, OFF_DYNAMIC_HEADER_LENGTH)?;
    let block_size = read_u32_le(bytes, magic_offset, OFF_BLOCK_SIZE)?;

    let layout = compute_segment_layout(
        magic_offset,
        volume_flags,
        block_size,
        embedded_xvd_length,
        xvc_data_length,
        dynamic_header_length,
        user_data_length,
        drive_size,
        file_size,
    )?;

    let size_status = if drive_size == 0 {
        SizeStatus::Undeclared
    } else if file_size < drive_size {
        SizeStatus::PartialOrTruncated
    } else {
        SizeStatus::Complete
    };

    Ok(XvdReport {
        file_name: "buffer.xvd".to_string(),
        file_size,
        magic_offset,
        volume_flags,
        is_encrypted,
        format_version,
        file_time,
        drive_size,
        vd_uid,
        ud_uid,
        content_type,
        embedded_xvd_length,
        user_data_length,
        xvc_data_length,
        dynamic_header_length,
        block_size,
        size_status,
        layout,
    })
}

/// Strictly read-only inspector for local `.xvd`/`.xvc` files of any size (including 100+ GiB).
///
/// Reads at most [`XVD_HEADER_SIZE`] (3 KiB / `0xC00`) bytes from offset 0 after querying file
/// metadata, and stores only the sanitized basename in [`XvdReport::file_name`].
pub fn inspect_xvd_file(path: &Path) -> Result<XvdReport, XvdError> {
    let mut file = fs::File::open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() {
        return Err(XvdError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            "XVD path is not a regular file",
        )));
    }
    let file_size = meta.len();
    file.seek(SeekFrom::Start(0))?;

    let to_read = usize::try_from(file_size.min(XVD_HEADER_SIZE as u64)).unwrap_or(XVD_HEADER_SIZE);
    let mut buf = vec![0u8; to_read];
    let mut read_total = 0usize;
    while read_total < to_read {
        let Some(dst) = buf.get_mut(read_total..) else {
            break;
        };
        match file.read(dst) {
            Ok(0) => break,
            Ok(n) => {
                read_total = read_total
                    .checked_add(n)
                    .ok_or(XvdError::ArithmeticOverflow {
                        context: "header_read_total",
                    })?
                    .min(to_read);
            }
            Err(err) if err.kind() == io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(XvdError::Io(err)),
        }
    }
    buf.truncate(read_total);

    let mut report = parse_xvd_header(&buf, file_size)?;
    report.file_name = sanitize_file_name(path);
    Ok(report)
}

impl XvdReport {
    pub fn container_kind(&self) -> &'static str {
        if self.has_xvc_region() {
            "XVC"
        } else {
            "XVD"
        }
    }

    pub fn has_xvc_region(&self) -> bool {
        self.xvc_data_length > 0
    }

    pub fn has_embedded_xvd(&self) -> bool {
        self.embedded_xvd_length > 0
    }

    pub fn has_user_data(&self) -> bool {
        self.user_data_length > 0
    }

    pub fn has_dynamic_header(&self) -> bool {
        self.dynamic_header_length > 0
    }

    pub fn is_partial_or_truncated(&self) -> bool {
        self.size_status == SizeStatus::PartialOrTruncated
    }

    pub fn missing_bytes(&self) -> u64 {
        if self.is_partial_or_truncated() {
            self.drive_size.saturating_sub(self.file_size)
        } else {
            0
        }
    }

    pub fn effective_block_size(&self) -> u32 {
        self.layout.effective_block_size
    }

    pub fn drive_block_count(&self) -> u64 {
        self.layout.drive_block_count
    }

    #[allow(dead_code)]
    pub fn segment(&self, kind: XvdSegmentKind) -> Option<&XvdSegment> {
        self.layout.segment(kind)
    }

    /// Strictly validates that every declared header/metadata/drive segment fits within `self.file_size`.
    #[allow(dead_code)]
    pub fn validate_segments_within_file(&self) -> Result<(), XvdError> {
        for seg in &self.layout.segments {
            if seg.end > self.file_size {
                return Err(XvdError::SegmentOutOfBounds {
                    segment: seg.kind,
                    offset: seg.offset,
                    length: seg.length,
                    end: seg.end,
                    limit: self.file_size,
                });
            }
        }
        Ok(())
    }

    /// Computes the checked `(byte_offset, byte_length)` span for a drive block range.
    #[allow(dead_code)]
    pub fn checked_drive_block_range(
        &self,
        block_index: u64,
        block_count: u64,
    ) -> Result<(u64, u64), XvdError> {
        let base_offset = self
            .segment(XvdSegmentKind::DriveData)
            .map_or(self.layout.metadata_end, |seg| seg.offset);
        checked_block_byte_range(
            base_offset,
            block_index,
            block_count,
            self.layout.effective_block_size,
            self.layout.drive_block_count,
        )
    }

    /// Computes a checked drive block byte range and verifies it fits within `self.file_size`.
    #[allow(dead_code)]
    pub fn checked_drive_block_range_in_file(
        &self,
        block_index: u64,
        block_count: u64,
    ) -> Result<(u64, u64), XvdError> {
        let (offset, length) = self.checked_drive_block_range(block_index, block_count)?;
        let end = offset
            .checked_add(length)
            .ok_or(XvdError::ArithmeticOverflow {
                context: "drive_block_end",
            })?;
        if end > self.file_size {
            return Err(XvdError::SegmentOutOfBounds {
                segment: XvdSegmentKind::DriveData,
                offset,
                length,
                end,
                limit: self.file_size,
            });
        }
        Ok((offset, length))
    }

    /// Zero-panic slice accessor for a declared segment inside an in-memory container buffer.
    #[allow(dead_code)]
    pub fn slice_segment<'a>(
        &self,
        container_bytes: &'a [u8],
        kind: XvdSegmentKind,
    ) -> Result<&'a [u8], XvdError> {
        let seg = self.segment(kind).ok_or(XvdError::SegmentOutOfBounds {
            segment: kind,
            offset: 0,
            length: 0,
            end: 0,
            limit: container_bytes.len() as u64,
        })?;
        checked_slice_range(container_bytes, seg.offset, seg.length)
    }

    /// Zero-panic slice accessor for a drive block range inside an in-memory container buffer.
    #[allow(dead_code)]
    pub fn slice_drive_blocks<'a>(
        &self,
        container_bytes: &'a [u8],
        block_index: u64,
        block_count: u64,
    ) -> Result<&'a [u8], XvdError> {
        let (offset, length) = self.checked_drive_block_range(block_index, block_count)?;
        checked_slice_range(container_bytes, offset, length)
    }

    pub fn vd_uid_string(&self) -> String {
        format_uuid(&self.vd_uid)
    }

    pub fn ud_uid_string(&self) -> String {
        format_uuid(&self.ud_uid)
    }

    /// Formats a sanitized JSON report containing only header metadata and relative filename
    /// (never absolute host paths, signatures, key material, or payload bytes).
    pub fn to_json(&self) -> String {
        let active_flags = self.volume_flags.active_names();
        let mut flags_json = String::from("[");
        for (idx, flag) in active_flags.iter().enumerate() {
            if idx > 0 {
                flags_json.push_str(", ");
            }
            flags_json.push_str(&format!("\"{flag}\""));
        }
        flags_json.push(']');

        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"abi_version\": {ABI_VERSION},\n"));
        out.push_str(&format!(
            "  \"file_name\": \"{}\",\n",
            json_escape(&self.file_name)
        ));
        out.push_str(&format!(
            "  \"container_kind\": \"{}\",\n",
            self.container_kind()
        ));
        out.push_str("  \"magic\": \"msft-xvd\",\n");
        out.push_str(&format!("  \"magic_offset\": {},\n", self.magic_offset));
        out.push_str(&format!("  \"format_version\": {},\n", self.format_version));
        out.push_str(&format!(
            "  \"content_type\": \"{}\",\n",
            json_escape(&self.content_type.to_string())
        ));
        out.push_str(&format!(
            "  \"content_type_id\": {},\n",
            self.content_type.raw_id()
        ));
        out.push_str(&format!("  \"is_encrypted\": {},\n", self.is_encrypted));
        out.push_str("  \"volume_flags\": {\n");
        out.push_str(&format!("    \"raw\": {},\n", self.volume_flags.raw));
        out.push_str(&format!(
            "    \"hex\": \"0x{:08X}\",\n",
            self.volume_flags.raw
        ));
        out.push_str(&format!(
            "    \"read_only\": {},\n",
            self.volume_flags.read_only
        ));
        out.push_str(&format!(
            "    \"encryption_disabled\": {},\n",
            self.volume_flags.encryption_disabled
        ));
        out.push_str(&format!(
            "    \"data_integrity_disabled\": {},\n",
            self.volume_flags.data_integrity_disabled
        ));
        out.push_str(&format!(
            "    \"legacy_sector_size\": {},\n",
            self.volume_flags.legacy_sector_size
        ));
        out.push_str(&format!("    \"decoded\": {flags_json}\n"));
        out.push_str("  },\n");
        out.push_str(&format!("  \"file_time\": {},\n", self.file_time));
        out.push_str(&format!("  \"vd_uid\": \"{}\",\n", self.vd_uid_string()));
        out.push_str(&format!("  \"ud_uid\": \"{}\",\n", self.ud_uid_string()));
        out.push_str("  \"layout\": {\n");
        out.push_str(&format!("    \"file_size\": {},\n", self.file_size));
        out.push_str(&format!("    \"drive_size\": {},\n", self.drive_size));
        out.push_str(&format!("    \"size_status\": \"{}\",\n", self.size_status));
        out.push_str(&format!(
            "    \"is_partial_or_truncated\": {},\n",
            self.is_partial_or_truncated()
        ));
        out.push_str(&format!(
            "    \"missing_bytes\": {},\n",
            self.missing_bytes()
        ));
        out.push_str(&format!("    \"block_size\": {},\n", self.block_size));
        out.push_str(&format!(
            "    \"embedded_xvd_length\": {},\n",
            self.embedded_xvd_length
        ));
        out.push_str(&format!(
            "    \"user_data_length\": {},\n",
            self.user_data_length
        ));
        out.push_str(&format!(
            "    \"xvc_data_length\": {},\n",
            self.xvc_data_length
        ));
        out.push_str(&format!(
            "    \"dynamic_header_length\": {},\n",
            self.dynamic_header_length
        ));
        out.push_str(&format!(
            "    \"has_xvc_region\": {},\n",
            self.has_xvc_region()
        ));
        out.push_str(&format!(
            "    \"has_embedded_xvd\": {},\n",
            self.has_embedded_xvd()
        ));
        out.push_str(&format!(
            "    \"has_user_data\": {},\n",
            self.has_user_data()
        ));
        out.push_str(&format!(
            "    \"has_dynamic_header\": {},\n",
            self.has_dynamic_header()
        ));
        out.push_str(&format!(
            "    \"page_size\": {},\n",
            self.layout.page_size
        ));
        out.push_str(&format!(
            "    \"effective_block_size\": {},\n",
            self.effective_block_size()
        ));
        out.push_str(&format!(
            "    \"drive_block_count\": {},\n",
            self.drive_block_count()
        ));
        out.push_str(&format!(
            "    \"metadata_end\": {},\n",
            self.layout.metadata_end
        ));
        out.push_str(&format!(
            "    \"total_declared_bytes\": {},\n",
            self.layout.total_declared_bytes
        ));
        out.push_str(&format!(
            "    \"all_segments_within_file\": {},\n",
            self.layout.all_segments_within_file()
        ));
        out.push_str("    \"segments\": [\n");
        for (idx, seg) in self.layout.segments.iter().enumerate() {
            let comma = if idx + 1 == self.layout.segments.len() {
                ""
            } else {
                ","
            };
            out.push_str(&format!(
                "      {{\"kind\": \"{}\", \"offset\": {}, \"length\": {}, \"aligned_length\": {}, \"end\": {}, \"aligned_end\": {}, \"block_count\": {}, \"within_file\": {}}}{}\n",
                seg.kind,
                seg.offset,
                seg.length,
                seg.aligned_length,
                seg.end,
                seg.aligned_end,
                seg.block_count,
                seg.within_file,
                comma
            ));
        }
        out.push_str("    ]\n");
        out.push_str("  }\n");
        out.push_str("}\n");
        out
    }

    /// Formats a clean human-readable terminal summary using only sanitized metadata.
    pub fn format_human(&self) -> String {
        let active_flags = self.volume_flags.active_names();
        let flags_label = if active_flags.is_empty() {
            "none".to_string()
        } else {
            active_flags.join(", ")
        };

        let encryption_line = if self.is_encrypted {
            "ENCRYPTED (EncryptionDisabled flag is NOT set)"
        } else {
            "UNENCRYPTED (EncryptionDisabled flag is set)"
        };

        let size_status_line = match self.size_status {
            SizeStatus::Complete => "complete (file size >= declared drive size)".to_string(),
            SizeStatus::Undeclared => "undeclared (declared drive size is 0)".to_string(),
            SizeStatus::PartialOrTruncated => {
                let missing = self.missing_bytes();
                let pct = if self.drive_size > 0 {
                    ((self.file_size as f64 / self.drive_size as f64) * 100.0).clamp(0.0, 100.0)
                } else {
                    0.0
                };
                if self.has_dynamic_header() {
                    format!(
                        "PARTIAL / TRUNCATED ({pct:.1}% on disk, missing {} / {missing} bytes; may be incomplete download or sparse dynamic XVD)",
                        format_bytes(missing)
                    )
                } else {
                    format!(
                        "PARTIAL / TRUNCATED ({pct:.1}% on disk, missing {} / {missing} bytes — likely incomplete download)",
                        format_bytes(missing)
                    )
                }
            }
        };

        let xvc_suffix = if self.has_xvc_region() {
            " (XVC metadata region present)"
        } else {
            " (none)"
        };

        let mut out = String::new();
        out.push_str("=== playgta6web Xbox XVD/XVC Header ===\n");
        out.push_str(&format!("File             : {}\n", self.file_name));
        out.push_str(&format!(
            "Container Kind   : {} (magic \"msft-xvd\" at 0x{:04X})\n",
            self.container_kind(),
            self.magic_offset
        ));
        out.push_str(&format!("Format Version   : {}\n", self.format_version));
        out.push_str(&format!(
            "Content Type     : {} ({})\n",
            self.content_type,
            self.content_type.raw_id()
        ));
        out.push_str(&format!("Encryption       : {encryption_line}\n"));
        out.push_str(&format!(
            "Volume Flags     : 0x{:08X} ({flags_label})\n",
            self.volume_flags.raw
        ));
        out.push_str(&format!("File Time        : {}\n", self.file_time));
        out.push_str(&format!("VD UID           : {}\n", self.vd_uid_string()));
        out.push_str(&format!("UD UID           : {}\n", self.ud_uid_string()));
        out.push_str("\nLayout & Sizes:\n");
        out.push_str(&format!(
            "  File Size on Disk     : {} ({} bytes)\n",
            format_bytes(self.file_size),
            self.file_size
        ));
        out.push_str(&format!(
            "  Declared Drive Size   : {} ({} bytes)\n",
            format_bytes(self.drive_size),
            self.drive_size
        ));
        out.push_str(&format!("  Size Status           : {size_status_line}\n"));
        out.push_str(&format!(
            "  Block Size            : {} bytes (effective: {} bytes, {} drive blocks)\n",
            self.block_size, self.layout.effective_block_size, self.layout.drive_block_count
        ));
        out.push_str(&format!(
            "  Embedded XVD Length   : {} bytes\n",
            self.embedded_xvd_length
        ));
        out.push_str(&format!(
            "  User Data Length      : {} bytes\n",
            self.user_data_length
        ));
        out.push_str(&format!(
            "  XVC Data Length       : {} bytes{xvc_suffix}\n",
            self.xvc_data_length
        ));
        out.push_str(&format!(
            "  Dynamic Header Length : {} bytes\n",
            self.dynamic_header_length
        ));
        out.push_str(&format!(
            "  Declared Layout End   : {} ({} bytes)\n",
            format_bytes(self.layout.total_declared_bytes),
            self.layout.total_declared_bytes
        ));
        out.push_str("\nSegments:\n");
        for seg in &self.layout.segments {
            let status = if seg.within_file {
                "in_bounds"
            } else {
                "truncated"
            };
            out.push_str(&format!(
                "  {:<15} : [0x{:08X}..0x{:08X}) {} bytes ({} blocks, {})\n",
                seg.kind.as_str(),
                seg.offset,
                seg.end,
                seg.length,
                seg.block_count,
                status
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn unique_temp_dir(label: &str) -> PathBuf {
        let n = TEST_COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir =
            env::temp_dir().join(format!("playgta6_xvd_{label}_{}_{}", std::process::id(), n));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    struct SyntheticXvdSpec {
        magic_offset: usize,
        total_buf_len: usize,
        volume_flags: u32,
        format_version: u32,
        file_time: u64,
        drive_size: u64,
        vd_uid: [u8; 16],
        ud_uid: [u8; 16],
        content_type: u32,
        embedded_xvd_length: u32,
        user_data_length: u32,
        xvc_data_length: u32,
        dynamic_header_length: u32,
        block_size: u32,
    }

    impl Default for SyntheticXvdSpec {
        fn default() -> Self {
            Self {
                magic_offset: XVD_STANDARD_MAGIC_OFFSET,
                total_buf_len: XVD_HEADER_SIZE,
                volume_flags: FLAG_READ_ONLY,
                format_version: 3,
                file_time: 133_500_000_000_000_000,
                drive_size: 4 * 1024 * 1024 * 1024, // 4 GiB
                vd_uid: [
                    0x78, 0x56, 0x34, 0x12, // Data1 LE -> 12345678
                    0xBC, 0x9A, // Data2 LE -> 9abc
                    0xF0, 0xDE, // Data3 LE -> def0
                    0x11, 0x22, // Data4[0..2] -> 1122
                    0x33, 0x44, 0x55, 0x66, 0x77, 0x88, // Data4[2..8] -> 334455667788
                ],
                ud_uid: [
                    0x04, 0x03, 0x02, 0x01, // 01020304
                    0x06, 0x05, // 0506
                    0x08, 0x07, // 0708
                    0x09, 0x0A, // 090a
                    0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, // 0b0c0d0e0f10
                ],
                content_type: 1, // Title
                embedded_xvd_length: 0,
                user_data_length: 4096,
                xvc_data_length: 65_536,
                dynamic_header_length: 0,
                block_size: 4096,
            }
        }
    }

    fn build_synthetic_xvd(spec: &SyntheticXvdSpec) -> Vec<u8> {
        let mut buf = vec![0u8; spec.total_buf_len];
        let base = spec.magic_offset;
        buf[base..base + 8].copy_from_slice(XVD_MAGIC);
        buf[base + OFF_VOLUME_FLAGS..base + OFF_VOLUME_FLAGS + 4]
            .copy_from_slice(&spec.volume_flags.to_le_bytes());
        buf[base + OFF_FORMAT_VERSION..base + OFF_FORMAT_VERSION + 4]
            .copy_from_slice(&spec.format_version.to_le_bytes());
        buf[base + OFF_FILE_TIME..base + OFF_FILE_TIME + 8]
            .copy_from_slice(&spec.file_time.to_le_bytes());
        buf[base + OFF_DRIVE_SIZE..base + OFF_DRIVE_SIZE + 8]
            .copy_from_slice(&spec.drive_size.to_le_bytes());
        buf[base + OFF_VD_UID..base + OFF_VD_UID + 16].copy_from_slice(&spec.vd_uid);
        buf[base + OFF_UD_UID..base + OFF_UD_UID + 16].copy_from_slice(&spec.ud_uid);
        buf[base + OFF_CONTENT_TYPE..base + OFF_CONTENT_TYPE + 4]
            .copy_from_slice(&spec.content_type.to_le_bytes());
        buf[base + OFF_EMBEDDED_XVD_LENGTH..base + OFF_EMBEDDED_XVD_LENGTH + 4]
            .copy_from_slice(&spec.embedded_xvd_length.to_le_bytes());
        buf[base + OFF_USER_DATA_LENGTH..base + OFF_USER_DATA_LENGTH + 4]
            .copy_from_slice(&spec.user_data_length.to_le_bytes());
        buf[base + OFF_XVC_DATA_LENGTH..base + OFF_XVC_DATA_LENGTH + 4]
            .copy_from_slice(&spec.xvc_data_length.to_le_bytes());
        buf[base + OFF_DYNAMIC_HEADER_LENGTH..base + OFF_DYNAMIC_HEADER_LENGTH + 4]
            .copy_from_slice(&spec.dynamic_header_length.to_le_bytes());
        buf[base + OFF_BLOCK_SIZE..base + OFF_BLOCK_SIZE + 4]
            .copy_from_slice(&spec.block_size.to_le_bytes());
        buf
    }

    #[test]
    fn parses_standard_encrypted_xvc_header_and_formats_uuids() {
        let spec = SyntheticXvdSpec::default();
        let buf = build_synthetic_xvd(&spec);
        let file_size = spec.drive_size + 0x10_0000;

        let report = parse_xvd_header(&buf, file_size).unwrap();
        assert_eq!(report.magic_offset, 0x200);
        assert_eq!(report.container_kind(), "XVC");
        assert!(report.is_encrypted);
        assert!(report.volume_flags.read_only);
        assert!(!report.volume_flags.encryption_disabled);
        assert!(!report.volume_flags.data_integrity_disabled);
        assert!(!report.volume_flags.legacy_sector_size);
        assert_eq!(report.format_version, 3);
        assert_eq!(report.file_time, 133_500_000_000_000_000);
        assert_eq!(report.drive_size, 4 * 1024 * 1024 * 1024);
        assert_eq!(
            report.vd_uid_string(),
            "12345678-9abc-def0-1122-334455667788"
        );
        assert_eq!(
            report.ud_uid_string(),
            "01020304-0506-0708-090a-0b0c0d0e0f10"
        );
        assert_eq!(report.content_type, ContentType::Title);
        assert_eq!(report.embedded_xvd_length, 0);
        assert_eq!(report.user_data_length, 4096);
        assert_eq!(report.xvc_data_length, 65_536);
        assert_eq!(report.dynamic_header_length, 0);
        assert_eq!(report.block_size, 4096);
        assert!(report.has_xvc_region());
        assert!(!report.has_embedded_xvd());
        assert!(report.has_user_data());
        assert!(!report.has_dynamic_header());
        assert_eq!(report.size_status, SizeStatus::Complete);
        assert!(!report.is_partial_or_truncated());
        assert_eq!(report.missing_bytes(), 0);

        let human = report.format_human();
        assert!(human.contains("Container Kind   : XVC (magic \"msft-xvd\" at 0x0200)"));
        assert!(human.contains("ENCRYPTED (EncryptionDisabled flag is NOT set)"));
        assert!(human.contains("12345678-9abc-def0-1122-334455667788"));
        assert!(human.contains("Title (1)"));
        assert!(human.contains("XVC metadata region present"));
    }

    #[test]
    fn parses_unencrypted_xvd_with_all_flags_and_raw_offset_zero() {
        let spec = SyntheticXvdSpec {
            magic_offset: XVD_RAW_MAGIC_OFFSET,
            total_buf_len: 0x200,
            volume_flags: FLAG_READ_ONLY
                | FLAG_ENCRYPTION_DISABLED
                | FLAG_DATA_INTEGRITY_DISABLED
                | FLAG_LEGACY_SECTOR_SIZE,
            content_type: 6, // Application
            embedded_xvd_length: 131_072,
            xvc_data_length: 0,
            dynamic_header_length: 8192,
            drive_size: 1_048_576,
            ..SyntheticXvdSpec::default()
        };
        let buf = build_synthetic_xvd(&spec);
        let report = parse_xvd_header(&buf, 2_097_152).unwrap();

        assert_eq!(report.magic_offset, 0x0);
        assert_eq!(report.container_kind(), "XVD");
        assert!(!report.is_encrypted);
        assert!(report.volume_flags.read_only);
        assert!(report.volume_flags.encryption_disabled);
        assert!(report.volume_flags.data_integrity_disabled);
        assert!(report.volume_flags.legacy_sector_size);
        assert_eq!(
            report.volume_flags.active_names(),
            vec![
                "ReadOnly",
                "EncryptionDisabled",
                "DataIntegrityDisabled",
                "LegacySectorSize",
            ]
        );
        assert_eq!(report.content_type, ContentType::Application);
        assert!(report.has_embedded_xvd());
        assert!(!report.has_xvc_region());
        assert!(report.has_dynamic_header());

        let json = report.to_json();
        assert!(json.contains("\"magic_offset\": 0,"));
        assert!(json.contains("\"container_kind\": \"XVD\","));
        assert!(json.contains("\"is_encrypted\": false,"));
        assert!(json.contains("\"encryption_disabled\": true,"));
        assert!(json.contains("\"has_xvc_region\": false,"));
        assert!(json.contains(
            "\"decoded\": [\"ReadOnly\", \"EncryptionDisabled\", \"DataIntegrityDisabled\", \"LegacySectorSize\"]"
        ));
    }

    #[test]
    fn detects_partial_or_truncated_download_against_declared_drive_size() {
        let spec = SyntheticXvdSpec {
            drive_size: 100_000_000_000, // 100 GB declared
            ..SyntheticXvdSpec::default()
        };
        let buf = build_synthetic_xvd(&spec);
        let partial_size = 5_000_000_000; // 5 GB downloaded so far

        let report = parse_xvd_header(&buf, partial_size).unwrap();
        assert_eq!(report.size_status, SizeStatus::PartialOrTruncated);
        assert!(report.is_partial_or_truncated());
        assert_eq!(report.missing_bytes(), 95_000_000_000);

        let json = report.to_json();
        assert!(json.contains("\"size_status\": \"partial_or_truncated\","));
        assert!(json.contains("\"is_partial_or_truncated\": true,"));
        assert!(json.contains("\"missing_bytes\": 95000000000,"));

        let human = report.format_human();
        assert!(human.contains("PARTIAL / TRUNCATED"));
        assert!(human.contains("likely incomplete download"));
    }

    #[test]
    fn maps_all_known_and_unknown_content_types() {
        let expected: &[(u32, ContentType, &str)] = &[
            (0, ContentType::Data, "Data"),
            (1, ContentType::Title, "Title"),
            (2, ContentType::SystemOS, "SystemOS"),
            (3, ContentType::EraOS, "EraOS"),
            (4, ContentType::Scratch, "Scratch"),
            (5, ContentType::ResetData, "ResetData"),
            (6, ContentType::Application, "Application"),
            (7, ContentType::HostOS, "HostOS"),
            (8, ContentType::X360STFS, "X360STFS"),
            (9, ContentType::X360FATX, "X360FATX"),
            (10, ContentType::X360GDFX, "X360GDFX"),
            (11, ContentType::Updater, "Updater"),
            (12, ContentType::OfflineUpdater, "OfflineUpdater"),
            (13, ContentType::Template, "Template"),
            (14, ContentType::MteHost, "MteHost"),
            (15, ContentType::MteApp, "MteApp"),
            (16, ContentType::MteTitle, "MteTitle"),
            (17, ContentType::MteEraOS, "MteEraOS"),
            (18, ContentType::AppCompat, "AppCompat"),
            (99, ContentType::Unknown(99), "Unknown(99)"),
        ];

        for &(id, variant, label) in expected {
            let parsed = ContentType::from_u32(id);
            assert_eq!(parsed, variant);
            assert_eq!(parsed.raw_id(), id);
            assert_eq!(parsed.to_string(), label);
        }
    }

    #[test]
    fn rejects_invalid_magic_and_too_short_buffers() {
        // Empty / too short before offset 0x200
        let empty_err = parse_xvd_header(&[], 0).unwrap_err();
        assert!(matches!(
            empty_err,
            XvdError::TruncatedHeader {
                actual: 0,
                expected: XVD_STANDARD_MIN_BYTES
            }
        ));

        let short_err = parse_xvd_header(&[0u8; 64], 64).unwrap_err();
        assert!(matches!(
            short_err,
            XvdError::TruncatedHeader {
                actual: 64,
                expected: XVD_STANDARD_MIN_BYTES
            }
        ));

        // Magic present at 0x200, but truncated before block_size (0x284)
        let mut trunc_std = vec![0u8; 0x240];
        trunc_std[0x200..0x208].copy_from_slice(XVD_MAGIC);
        let trunc_std_err = parse_xvd_header(&trunc_std, 0x240).unwrap_err();
        assert!(matches!(
            trunc_std_err,
            XvdError::TruncatedHeader {
                actual: 0x240,
                expected: XVD_STANDARD_MIN_BYTES
            }
        ));

        // Magic present at 0x0, but truncated before 0x84
        let mut trunc_raw = vec![0u8; 0x40];
        trunc_raw[0..8].copy_from_slice(XVD_MAGIC);
        let trunc_raw_err = parse_xvd_header(&trunc_raw, 0x40).unwrap_err();
        assert!(matches!(
            trunc_raw_err,
            XvdError::TruncatedHeader {
                actual: 0x40,
                expected: XVD_MIN_FIELDS_BYTES
            }
        ));

        // Full 3 KiB buffer with invalid magic at 0x200
        let mut bad_magic = vec![0u8; XVD_HEADER_SIZE];
        bad_magic[0x200..0x208].copy_from_slice(b"NOT_XVD!");
        let magic_err = parse_xvd_header(&bad_magic, XVD_HEADER_SIZE as u64).unwrap_err();
        assert!(matches!(
            magic_err,
            XvdError::InvalidMagic {
                found_at_0x200: [b'N', b'O', b'T', b'_', b'X', b'V', b'D', b'!']
            }
        ));
        assert!(magic_err.to_string().contains("invalid XVD/XVC magic"));
    }

    #[test]
    fn inspects_xvd_file_sanitizing_host_path_and_never_leaking_payload() {
        let dir = unique_temp_dir("inspect_file");
        let nested = dir.join("secret_user_folder");
        fs::create_dir_all(&nested).unwrap();
        let file_path = nested.join("title_pkg.xvc");

        let spec = SyntheticXvdSpec {
            volume_flags: FLAG_READ_ONLY | FLAG_ENCRYPTION_DISABLED,
            drive_size: 4096,
            ..SyntheticXvdSpec::default()
        };
        let mut content = build_synthetic_xvd(&spec);
        // Append dummy payload bytes past the 3 KiB header to ensure they are never read/dumped.
        content.extend_from_slice(&vec![0xDE; 2048]);
        fs::write(&file_path, &content).unwrap();

        let report = inspect_xvd_file(&file_path).unwrap();
        assert_eq!(report.file_name, "title_pkg.xvc");
        assert_eq!(report.file_size, content.len() as u64);
        assert!(!report.is_encrypted);
        assert_eq!(report.size_status, SizeStatus::Complete);

        let abs_dir = nested.to_string_lossy().to_string();
        let json = report.to_json();
        let human = report.format_human();

        assert!(!json.contains(&abs_dir));
        assert!(!json.contains(&abs_dir.replace('\\', "/")));
        assert!(!json.contains("secret_user_folder"));
        assert!(!human.contains(&abs_dir));
        assert!(!human.contains("secret_user_folder"));
        assert!(!json.contains("DEDEDE"));
        assert!(!human.contains("DEDEDE"));

        assert!(json.contains("\"abi_version\": 1,"));
        assert!(json.contains("\"file_name\": \"title_pkg.xvc\","));
        assert!(json.contains("\"container_kind\": \"XVC\","));
        assert!(json.contains("\"magic\": \"msft-xvd\","));
        assert!(json.contains("\"magic_offset\": 512,"));
        assert!(json.contains("\"is_encrypted\": false,"));
        assert!(json.contains("\"vd_uid\": \"12345678-9abc-def0-1122-334455667788\","));
        assert!(json.contains("\"ud_uid\": \"01020304-0506-0708-090a-0b0c0d0e0f10\","));
        assert!(json.contains("\"has_xvc_region\": true,"));
        assert!(json.contains("\"size_status\": \"complete\","));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_empty_and_boundary_truncated_headers_without_panicking() {
        // Sweep every small boundary length up to 0x283 without magic.
        for len in [0usize, 1, 7, 8, 0x83, 0x84, 0x1FF, 0x200, 0x207] {
            let buf = vec![0u8; len];
            let err = parse_xvd_header(&buf, len as u64).unwrap_err();
            assert!(
                matches!(
                    err,
                    XvdError::TruncatedHeader {
                        actual,
                        expected: XVD_STANDARD_MIN_BYTES
                    } if actual == len
                ),
                "unexpected error for zero buffer of len {len}: {err:?}"
            );
        }

        // Raw magic present at offset 0x0, truncated 1 byte short of XVD_MIN_FIELDS_BYTES (0x83).
        let mut raw_short = vec![0u8; XVD_MIN_FIELDS_BYTES - 1];
        raw_short[0..8].copy_from_slice(XVD_MAGIC);
        let raw_short_err = parse_xvd_header(&raw_short, raw_short.len() as u64).unwrap_err();
        assert!(matches!(
            raw_short_err,
            XvdError::TruncatedHeader {
                actual: 0x83,
                expected: XVD_MIN_FIELDS_BYTES
            }
        ));

        // Exact raw minimum (0x84 bytes) succeeds with zero panics.
        let raw_min_spec = SyntheticXvdSpec {
            magic_offset: XVD_RAW_MAGIC_OFFSET,
            total_buf_len: XVD_MIN_FIELDS_BYTES,
            ..SyntheticXvdSpec::default()
        };
        let raw_min_buf = build_synthetic_xvd(&raw_min_spec);
        let raw_report = parse_xvd_header(&raw_min_buf, raw_min_spec.drive_size).unwrap();
        assert_eq!(raw_report.magic_offset, XVD_RAW_MAGIC_OFFSET);
        assert_eq!(raw_report.block_size, 4096);

        // Standard magic present at 0x200, truncated 1 byte short of XVD_STANDARD_MIN_BYTES (0x283).
        let mut std_short = vec![0u8; XVD_STANDARD_MIN_BYTES - 1];
        std_short[XVD_STANDARD_MAGIC_OFFSET..XVD_STANDARD_MAGIC_OFFSET + 8]
            .copy_from_slice(XVD_MAGIC);
        let std_short_err = parse_xvd_header(&std_short, std_short.len() as u64).unwrap_err();
        assert!(matches!(
            std_short_err,
            XvdError::TruncatedHeader {
                actual: 0x283,
                expected: XVD_STANDARD_MIN_BYTES
            }
        ));

        // Exact standard minimum (0x284 bytes) succeeds with zero panics.
        let std_min_spec = SyntheticXvdSpec {
            magic_offset: XVD_STANDARD_MAGIC_OFFSET,
            total_buf_len: XVD_STANDARD_MIN_BYTES,
            ..SyntheticXvdSpec::default()
        };
        let std_min_buf = build_synthetic_xvd(&std_min_spec);
        let std_report = parse_xvd_header(&std_min_buf, std_min_spec.drive_size).unwrap();
        assert_eq!(std_report.magic_offset, XVD_STANDARD_MAGIC_OFFSET);
        assert_eq!(std_report.block_size, 4096);
    }

    #[test]
    fn rejects_invalid_block_sizes_and_overflowing_block_counts() {
        // Explicit block_size == 0 is rejected by validate_block_size / checked_block_count.
        assert!(matches!(
            validate_block_size(0),
            Err(XvdError::InvalidBlockSize { block_size: 0, .. })
        ));
        assert!(matches!(
            checked_block_count(4096, 0),
            Err(XvdError::InvalidBlockSize { block_size: 0, .. })
        ));

        // In header parsing, block_size == 0 falls back safely to sector/page size (no div-by-zero).
        let zero_bs_std = SyntheticXvdSpec {
            block_size: 0,
            drive_size: 8192,
            ..SyntheticXvdSpec::default()
        };
        let report_std = parse_xvd_header(&build_synthetic_xvd(&zero_bs_std), 1_000_000).unwrap();
        assert_eq!(report_std.effective_block_size(), 4096);
        assert_eq!(report_std.drive_block_count(), 2);

        let zero_bs_legacy = SyntheticXvdSpec {
            volume_flags: FLAG_READ_ONLY | FLAG_LEGACY_SECTOR_SIZE,
            block_size: 0,
            drive_size: 2048,
            ..SyntheticXvdSpec::default()
        };
        let report_legacy =
            parse_xvd_header(&build_synthetic_xvd(&zero_bs_legacy), 1_000_000).unwrap();
        assert_eq!(report_legacy.effective_block_size(), 512);
        assert_eq!(report_legacy.drive_block_count(), 4);

        // Non-zero invalid block sizes (below min, non-power-of-two, above max) are rejected.
        for bad_bs in [1u32, 256, 511, 3000, 4097, XVD_MAX_BLOCK_SIZE * 2, u32::MAX] {
            let spec = SyntheticXvdSpec {
                block_size: bad_bs,
                ..SyntheticXvdSpec::default()
            };
            let err = parse_xvd_header(&build_synthetic_xvd(&spec), 1_000_000).unwrap_err();
            assert!(
                matches!(err, XvdError::InvalidBlockSize { block_size, .. } if block_size == bad_bs),
                "expected InvalidBlockSize for {bad_bs}, got {err:?}"
            );
        }

        // checked_block_count near u64::MAX boundary.
        let max_aligned = u64::MAX - 4095;
        assert_eq!(
            checked_block_count(max_aligned, 4096).unwrap(),
            max_aligned / 4096
        );
        assert!(matches!(
            checked_block_count(max_aligned + 1, 4096),
            Err(XvdError::ArithmeticOverflow { .. })
        ));
        assert!(matches!(
            checked_block_count(u64::MAX, 4096),
            Err(XvdError::ArithmeticOverflow { .. })
        ));

        // Overflowing drive_size in header is rejected cleanly instead of wrapping.
        for huge_drive in [u64::MAX, u64::MAX - 0x1000, u64::MAX - 0x10_000] {
            let spec = SyntheticXvdSpec {
                drive_size: huge_drive,
                ..SyntheticXvdSpec::default()
            };
            let err = parse_xvd_header(&build_synthetic_xvd(&spec), u64::MAX).unwrap_err();
            assert!(
                matches!(err, XvdError::ArithmeticOverflow { .. }),
                "expected ArithmeticOverflow for drive_size=0x{huge_drive:X}, got {err:?}"
            );
        }
    }

    #[test]
    fn guards_overflowing_block_offsets_and_out_of_bounds_block_ranges() {
        // Overflow in block_index + block_count.
        assert!(matches!(
            checked_block_byte_range(0x1000, u64::MAX, 1, 4096, u64::MAX),
            Err(XvdError::ArithmeticOverflow {
                context: "block_index_range"
            })
        ));

        // Overflow in block_index * block_size.
        let overflow_idx = (u64::MAX / 4096) + 1;
        assert!(matches!(
            checked_block_byte_range(0, overflow_idx, 1, 4096, u64::MAX),
            Err(XvdError::ArithmeticOverflow {
                context: "block_byte_offset"
            })
        ));

        // Overflow in base_offset + rel_offset.
        assert!(matches!(
            checked_block_byte_range(u64::MAX - 2048, 1, 1, 4096, 16),
            Err(XvdError::ArithmeticOverflow {
                context: "block_absolute_offset"
            })
        ));

        // Overflow in byte_offset + byte_length.
        assert!(matches!(
            checked_block_byte_range(u64::MAX - 4095, 0, 1, 4096, 16),
            Err(XvdError::ArithmeticOverflow {
                context: "block_byte_end"
            })
        ));

        // Out-of-bounds block range on a valid report.
        let spec = SyntheticXvdSpec {
            embedded_xvd_length: 0,
            xvc_data_length: 0,
            dynamic_header_length: 0,
            user_data_length: 0,
            drive_size: 4 * 4096,
            block_size: 4096,
            ..SyntheticXvdSpec::default()
        };
        let buf = build_synthetic_xvd(&spec);
        // Header aligned to 0x1000; 4 drive blocks sit at [0x1000..0x5000).
        // Make file truncated after 2 drive blocks (0x3000 bytes).
        let report = parse_xvd_header(&buf, 0x3000).unwrap();
        assert_eq!(report.drive_block_count(), 4);
        assert_eq!(report.checked_drive_block_range(0, 4).unwrap(), (0x1000, 0x4000));
        assert_eq!(report.checked_drive_block_range(3, 1).unwrap(), (0x4000, 0x1000));
        assert!(matches!(
            report.checked_drive_block_range(3, 2),
            Err(XvdError::BlockOutOfBounds {
                block_index: 3,
                block_count: 2,
                total_blocks: 4
            })
        ));

        // File-bounded drive block check allows blocks 0..2 and rejects block 2..3.
        assert_eq!(
            report.checked_drive_block_range_in_file(0, 2).unwrap(),
            (0x1000, 0x2000)
        );
        assert!(matches!(
            report.checked_drive_block_range_in_file(2, 1),
            Err(XvdError::SegmentOutOfBounds {
                segment: XvdSegmentKind::DriveData,
                offset: 0x3000,
                length: 0x1000,
                end: 0x4000,
                limit: 0x3000
            })
        ));
    }

    #[test]
    fn validates_boundary_segment_sizes_and_non_overlapping_layout() {
        // Exercise boundary lengths: 1 byte, page-1, exact page, page+1, and u32::MAX (in 64-bit math).
        let spec = SyntheticXvdSpec {
            embedded_xvd_length: 1,
            xvc_data_length: 4095,
            dynamic_header_length: 4097,
            user_data_length: u32::MAX,
            drive_size: 4097,
            block_size: 4096,
            ..SyntheticXvdSpec::default()
        };
        let buf = build_synthetic_xvd(&spec);

        // Compute expected total declared bytes:
        // Header: 0..0xC00 (aligned 0x1000)
        // EmbeddedXvd: 0x1000..0x1001 (aligned 0x2000)
        // XvcData: 0x2000..0x2FFF (aligned 0x3000)
        // DynamicHeader: 0x3000..0x4001 (aligned 0x5000)
        // UserData: 0x5000..(0x5000 + 0xFFFF_FFFF) (aligned 0x1_0000_5000)
        // DriveData: 0x1_0000_5000..(0x1_0000_5000 + 4097)
        let expected_user_aligned: u64 = 0x1_0000_0000;
        let expected_drive_offset = 0x5000 + expected_user_aligned;
        let expected_total = expected_drive_offset + 4097;

        let report_exact = parse_xvd_header(&buf, expected_total).unwrap();
        assert_eq!(report_exact.layout.total_declared_bytes, expected_total);
        assert_eq!(report_exact.layout.metadata_end, expected_drive_offset);
        assert_eq!(report_exact.drive_block_count(), 2);
        assert!(report_exact.layout.all_segments_within_file());
        assert!(report_exact.validate_segments_within_file().is_ok());

        // Verify strict monotonic non-overlapping invariant across all segments.
        for pair in report_exact.layout.segments.windows(2) {
            assert!(
                pair[0].end <= pair[0].aligned_end && pair[0].aligned_end <= pair[1].offset,
                "overlapping segments: {:?} vs {:?}",
                pair[0],
                pair[1]
            );
        }

        // One byte short of total declared bytes triggers SegmentOutOfBounds on DriveData.
        let report_short = parse_xvd_header(&buf, expected_total - 1).unwrap();
        assert!(!report_short.layout.all_segments_within_file());
        assert!(matches!(
            report_short.validate_segments_within_file(),
            Err(XvdError::SegmentOutOfBounds {
                segment: XvdSegmentKind::DriveData,
                end,
                limit,
                ..
            }) if end == expected_total && limit == expected_total - 1
        ));
    }

    #[test]
    fn slices_segments_and_blocks_without_panicking_on_truncation_or_overflow() {
        let spec = SyntheticXvdSpec {
            embedded_xvd_length: 0,
            xvc_data_length: 64,
            dynamic_header_length: 0,
            user_data_length: 0,
            drive_size: 2 * 4096,
            block_size: 4096,
            ..SyntheticXvdSpec::default()
        };
        let mut container = build_synthetic_xvd(&spec);
        // Pad to full layout: Header(0x1000) + XvcData(0x1000) + DriveData(0x2000) = 0x4000.
        container.resize(0x4000, 0);
        container[0x1000..0x1004].copy_from_slice(b"XVCT");
        container[0x2000..0x2004].copy_from_slice(b"BLK0");
        container[0x3000..0x3004].copy_from_slice(b"BLK1");

        let report = parse_xvd_header(&container, container.len() as u64).unwrap();
        let xvc_slice = report
            .slice_segment(&container, XvdSegmentKind::XvcData)
            .unwrap();
        assert_eq!(xvc_slice.len(), 64);
        assert_eq!(&xvc_slice[..4], b"XVCT");

        let blk1 = report.slice_drive_blocks(&container, 1, 1).unwrap();
        assert_eq!(blk1.len(), 4096);
        assert_eq!(&blk1[..4], b"BLK1");

        // Absent segment returns SegmentOutOfBounds.
        assert!(matches!(
            report.slice_segment(&container, XvdSegmentKind::EmbeddedXvd),
            Err(XvdError::SegmentOutOfBounds {
                segment: XvdSegmentKind::EmbeddedXvd,
                ..
            })
        ));

        // Truncated buffer returns InvalidBufferRange without panicking.
        let truncated = &container[..0x2500];
        assert!(matches!(
            report.slice_drive_blocks(truncated, 0, 1),
            Err(XvdError::InvalidBufferRange {
                offset: 0x2000,
                length: 4096,
                buffer_len: 0x2500
            })
        ));

        // Direct checked_slice_range guards overflow.
        assert!(matches!(
            checked_slice_range(&container, u64::MAX, 16),
            Err(XvdError::ArithmeticOverflow { .. })
        ));
    }

    #[test]
    fn produces_deterministic_metadata_json_and_human_output() {
        let spec = SyntheticXvdSpec {
            embedded_xvd_length: 8192,
            xvc_data_length: 4096,
            dynamic_header_length: 4096,
            user_data_length: 2048,
            drive_size: 16_384,
            block_size: 4096,
            ..SyntheticXvdSpec::default()
        };
        let buf = build_synthetic_xvd(&spec);
        let r1 = parse_xvd_header(&buf, 1_048_576).unwrap();
        let r2 = parse_xvd_header(&buf, 1_048_576).unwrap();

        let json1 = r1.to_json();
        let json2 = r2.to_json();
        assert_eq!(json1, json2);

        let human1 = r1.format_human();
        let human2 = r2.format_human();
        assert_eq!(human1, human2);

        // Verify canonical segment ordering in both JSON and human output.
        let expected_order = [
            "\"kind\": \"header\"",
            "\"kind\": \"embedded_xvd\"",
            "\"kind\": \"xvc_data\"",
            "\"kind\": \"dynamic_header\"",
            "\"kind\": \"user_data\"",
            "\"kind\": \"drive_data\"",
        ];
        let mut last_pos = 0usize;
        for marker in expected_order {
            let pos = json1
                .find(marker)
                .unwrap_or_else(|| panic!("missing marker {marker} in JSON:\n{json1}"));
            assert!(
                pos >= last_pos,
                "segment marker {marker} out of canonical order"
            );
            last_pos = pos;
        }

        assert!(json1.contains("\"effective_block_size\": 4096,"));
        assert!(json1.contains("\"drive_block_count\": 4,"));
        assert!(json1.contains("\"all_segments_within_file\": true,"));
        assert!(human1.contains("Segments:\n"));
        assert!(human1.contains("drive_data"));
    }

    #[test]
    fn sanitizes_error_diagnostics_and_cross_platform_file_names() {
        // XvdError::Io must never leak embedded host paths in Display output.
        let win_io = XvdError::Io(io::Error::new(
            io::ErrorKind::NotFound,
            r"C:\Users\Alice\Private\GTA6\secret_pkg.xvc not found",
        ));
        let win_msg = win_io.to_string();
        assert!(win_msg.contains("not_found"));
        assert!(!win_msg.contains("Alice"));
        assert!(!win_msg.contains("C:\\"));
        assert!(!win_msg.contains("secret_pkg.xvc"));

        let posix_io = XvdError::Io(io::Error::new(
            io::ErrorKind::PermissionDenied,
            "/Users/alice/private/game.xvd permission denied",
        ));
        let posix_msg = posix_io.to_string();
        assert!(posix_msg.contains("permission_denied"));
        assert!(!posix_msg.contains("/Users/alice"));
        assert!(!posix_msg.contains("game.xvd"));

        // Missing file inspection returns sanitized error without leaking temp/host path.
        let dir = unique_temp_dir("missing_secret_host");
        let missing = dir.join("missing_title.xvc");
        let missing_err = inspect_xvd_file(&missing).unwrap_err();
        let missing_msg = missing_err.to_string();
        assert!(missing_msg.contains("not_found"));
        assert!(!missing_msg.contains("missing_secret_host"));
        assert!(!missing_msg.contains("missing_title.xvc"));

        // Directory inspection fails cleanly with sanitized invalid_input error.
        let dir_err = inspect_xvd_file(&dir).unwrap_err();
        assert!(!dir_err.to_string().contains("missing_secret_host"));
        let _ = fs::remove_dir_all(&dir);

        // Cross-platform basename sanitization strips Windows, UNC, POSIX, drive-relative, and control chars.
        assert_eq!(
            sanitize_file_name(Path::new(r"C:\Users\Alice\Games\pkg.xvc")),
            "pkg.xvc"
        );
        assert_eq!(
            sanitize_file_name(Path::new(r"\\server\share\dir\pkg.xvd")),
            "pkg.xvd"
        );
        assert_eq!(
            sanitize_file_name(Path::new("/home/alice/games/pkg.xvc")),
            "pkg.xvc"
        );
        assert_eq!(sanitize_file_name(Path::new("D:bare_pkg.xvd")), "bare_pkg.xvd");
        assert_eq!(
            sanitize_file_name(Path::new("dir/clean\x00\nname.xvd")),
            "cleanname.xvd"
        );
        assert_eq!(sanitize_file_name(Path::new("..")), "unknown.xvd");
        assert_eq!(sanitize_file_name(Path::new("/")), "unknown.xvd");

        // Verify sanitized Display formatting for all bounds/overflow error variants.
        assert!(
            XvdError::InvalidBlockSize {
                block_size: 3000,
                reason: "block size must be a power of two"
            }
            .to_string()
            .contains("3000")
        );
        assert!(
            XvdError::ArithmeticOverflow {
                context: "block_byte_offset"
            }
            .to_string()
            .contains("block_byte_offset")
        );
        assert!(
            XvdError::SegmentOutOfBounds {
                segment: XvdSegmentKind::DriveData,
                offset: 0x1000,
                length: 0x2000,
                end: 0x3000,
                limit: 0x2000
            }
            .to_string()
            .contains("drive_data")
        );
        assert!(
            XvdError::BlockOutOfBounds {
                block_index: 4,
                block_count: 2,
                total_blocks: 5
            }
            .to_string()
            .contains("[4..+2)")
        );
        assert!(
            XvdError::InvalidBufferRange {
                offset: 0x100,
                length: 64,
                buffer_len: 32
            }
            .to_string()
            .contains("buffer length (32 bytes)")
        );
    }
}
