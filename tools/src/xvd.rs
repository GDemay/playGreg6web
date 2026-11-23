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
}

#[derive(Debug)]
pub enum XvdError {
    Io(io::Error),
    TruncatedHeader { actual: usize, expected: usize },
    InvalidMagic { found_at_0x200: [u8; 8] },
}

impl fmt::Display for XvdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "I/O error reading XVD/XVC container: {err}"),
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

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ])
}

fn read_array_16(bytes: &[u8], offset: usize) -> [u8; 16] {
    let mut out = [0u8; 16];
    out.copy_from_slice(&bytes[offset..offset + 16]);
    out
}

fn sanitize_file_name(path: &Path) -> String {
    path.file_name()
        .map(|name| name.to_string_lossy().replace('\\', "/"))
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown.xvd".to_string())
}

fn detect_magic_offset(bytes: &[u8]) -> Result<usize, XvdError> {
    if bytes.len() >= XVD_STANDARD_MAGIC_OFFSET + XVD_MAGIC.len()
        && &bytes[XVD_STANDARD_MAGIC_OFFSET..XVD_STANDARD_MAGIC_OFFSET + XVD_MAGIC.len()]
            == XVD_MAGIC
    {
        return Ok(XVD_STANDARD_MAGIC_OFFSET);
    }

    if bytes.len() >= XVD_MAGIC.len() && &bytes[..XVD_MAGIC.len()] == XVD_MAGIC {
        return Ok(XVD_RAW_MAGIC_OFFSET);
    }

    if bytes.len() < XVD_STANDARD_MAGIC_OFFSET + XVD_MAGIC.len() {
        return Err(XvdError::TruncatedHeader {
            actual: bytes.len(),
            expected: XVD_STANDARD_MIN_BYTES,
        });
    }

    let mut found = [0u8; 8];
    found.copy_from_slice(&bytes[XVD_STANDARD_MAGIC_OFFSET..XVD_STANDARD_MAGIC_OFFSET + 8]);
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
    let required_bytes = magic_offset + XVD_MIN_FIELDS_BYTES;
    if bytes.len() < required_bytes {
        return Err(XvdError::TruncatedHeader {
            actual: bytes.len(),
            expected: required_bytes,
        });
    }

    let volume_flags = VolumeFlags::from_raw(read_u32_le(bytes, magic_offset + OFF_VOLUME_FLAGS));
    let is_encrypted = !volume_flags.encryption_disabled;
    let format_version = read_u32_le(bytes, magic_offset + OFF_FORMAT_VERSION);
    let file_time = read_u64_le(bytes, magic_offset + OFF_FILE_TIME);
    let drive_size = read_u64_le(bytes, magic_offset + OFF_DRIVE_SIZE);
    let vd_uid = read_array_16(bytes, magic_offset + OFF_VD_UID);
    let ud_uid = read_array_16(bytes, magic_offset + OFF_UD_UID);
    let content_type = ContentType::from_u32(read_u32_le(bytes, magic_offset + OFF_CONTENT_TYPE));
    let embedded_xvd_length = read_u32_le(bytes, magic_offset + OFF_EMBEDDED_XVD_LENGTH);
    let user_data_length = read_u32_le(bytes, magic_offset + OFF_USER_DATA_LENGTH);
    let xvc_data_length = read_u32_le(bytes, magic_offset + OFF_XVC_DATA_LENGTH);
    let dynamic_header_length = read_u32_le(bytes, magic_offset + OFF_DYNAMIC_HEADER_LENGTH);
    let block_size = read_u32_le(bytes, magic_offset + OFF_BLOCK_SIZE);

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
    })
}

/// Strictly read-only inspector for local `.xvd`/`.xvc` files of any size (including 100+ GiB).
///
/// Reads at most [`XVD_HEADER_SIZE`] (3 KiB / `0xC00`) bytes from offset 0 after querying file
/// metadata, and stores only the sanitized basename in [`XvdReport::file_name`].
pub fn inspect_xvd_file(path: &Path) -> Result<XvdReport, XvdError> {
    let mut file = fs::File::open(path)?;
    let file_size = file.metadata()?.len();
    file.seek(SeekFrom::Start(0))?;

    let to_read = usize::try_from(file_size.min(XVD_HEADER_SIZE as u64)).unwrap_or(XVD_HEADER_SIZE);
    let mut buf = vec![0u8; to_read];
    let mut read_total = 0;
    while read_total < to_read {
        match file.read(&mut buf[read_total..]) {
            Ok(0) => break,
            Ok(n) => read_total += n,
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
            "    \"has_dynamic_header\": {}\n",
            self.has_dynamic_header()
        ));
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
                    (self.file_size as f64 / self.drive_size as f64) * 100.0
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
            "  Block Size            : {} bytes\n",
            self.block_size
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
}
