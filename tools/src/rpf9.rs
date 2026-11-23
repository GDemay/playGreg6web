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

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
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
    if header_bytes.len() < 4 {
        return Err(RpfError::TruncatedHeader {
            actual: header_bytes.len(),
            expected: RPF9_HEADER_SIZE,
        });
    }

    let magic = [
        header_bytes[0],
        header_bytes[1],
        header_bytes[2],
        header_bytes[3],
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
        read_u32_le(header_bytes, 0),
        read_u32_le(header_bytes, 4),
        read_u32_le(header_bytes, 8),
        read_u32_le(header_bytes, 12),
        read_u32_le(header_bytes, 16),
        read_u32_le(header_bytes, 20),
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

    let min_index_bytes = classic_min.ok_or(RpfError::SizeOverflow)?;
    if min_index_bytes > total_size {
        return Err(RpfError::TruncatedArchive {
            required_bytes: min_index_bytes,
            actual_bytes: total_size,
        });
    }

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
        min_index_bytes,
    })
}

/// Non-extracting RPF9 header inspector for local files of any size (including 100+ GB archives).
///
/// Only reads at most [`RPF9_HEADER_SIZE`] bytes from offset 0 after querying file length.
pub fn inspect_rpf_file(path: &Path) -> Result<RpfHeader, RpfError> {
    let mut file = fs::File::open(path)?;
    let total_size = file.metadata()?.len();
    file.seek(SeekFrom::Start(0))?;

    let to_read = usize::try_from(total_size.min(RPF9_HEADER_SIZE as u64)).unwrap_or(RPF9_HEADER_SIZE);
    let mut buf = vec![0u8; to_read];
    file.read_exact(&mut buf)?;

    inspect_rpf_bytes(&buf, total_size)
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
    out: &mut Vec<BootsetEntry>,
) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let meta = entry.metadata()?;
        if meta.is_dir() {
            collect_bootset_entries(root, &path, max_header_bytes, out)?;
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
            out.push(BootsetEntry {
                path: rel,
                offset: 0,
                length: clamped_len,
            });
        }
    }
    Ok(())
}

/// Scans `game_dir` for `.rpf` and `.config` files (case-insensitive) and returns sorted
/// `bootset.json` range entries with each `length` clamped to the file's actual size.
pub fn generate_bootset(game_dir: &Path, max_header_bytes: u64) -> io::Result<Vec<BootsetEntry>> {
    let mut entries = Vec::new();
    collect_bootset_entries(game_dir, game_dir, max_header_bytes, &mut entries)?;
    entries.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(entries)
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
}
