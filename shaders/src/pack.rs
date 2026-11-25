#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use crate::convert::{ConversionError, ConversionErrorKind, ShaderStage, format_shader_hash};

/// Binary pack file magic (`"WGPK"`).
pub const PACK_MAGIC: [u8; 4] = *b"WGPK";

/// Binary pack format version.
pub const PACK_VERSION: u32 = 1;

/// Fixed pack header size in bytes: magic(4) + version(4) + entry_count(4) + flags(4).
pub const PACK_HEADER_BYTES: usize = 16;

/// Fixed entry table record size in bytes: hash(8) + stage(1) + reserved(3) + offset(4) + length(4).
pub const PACK_ENTRY_BYTES: usize = 20;

/// Maximum addressable WGPK v1 archive size in bytes (`u32::MAX`).
pub const MAX_PACK_BYTES: u32 = u32::MAX;

/// Maximum entry count whose header + entry table fits within [`MAX_PACK_BYTES`] (`u32::MAX`).
pub const MAX_PACK_ENTRIES: u32 = (MAX_PACK_BYTES - PACK_HEADER_BYTES as u32) / PACK_ENTRY_BYTES as u32;

/// Shader index ABI version matching `docs/gpu_commands.md`.
pub const SHADER_INDEX_ABI_VERSION: u32 = 1;

/// Input/output entry in a `.wgsl.pack` archive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackEntry {
    pub hash: u64,
    pub stage: ShaderStage,
    pub wgsl: String,
}

/// Decoded entry metadata plus payload from `read_pack`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedPackEntry {
    pub hash: u64,
    pub stage: ShaderStage,
    pub offset: u32,
    pub length: u32,
    pub wgsl: String,
}

/// Decoded `.wgsl.pack` container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderPack {
    pub version: u32,
    pub flags: u32,
    pub entries: Vec<DecodedPackEntry>,
}

impl ShaderPack {
    /// Returns `true` if entries are ordered by strictly ascending 64-bit hash.
    pub fn is_canonical_hash_order(&self) -> bool {
        self.verify_canonical_hash_order().is_ok()
    }

    /// Verify that all decoded entries are in strictly ascending 64-bit hash order.
    pub fn verify_canonical_hash_order(&self) -> Result<(), PackError> {
        verify_canonical_hash_order(self.entries.iter().map(|entry| entry.hash))
    }
}

/// Metadata recorded per shader for `shader-index.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShaderIndexEntry {
    pub hash: u64,
    pub stage: ShaderStage,
    pub wgsl_path: String,
    pub entry_point: String,
    pub pack_offset: Option<u32>,
    pub pack_length: Option<u32>,
}

/// Record of a single shader that failed SPIR-V -> WGSL conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailedShaderRecord {
    pub source_path: String,
    pub error: ConversionError,
}

/// Summary statistics for `render_blocker_report_md`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockerReportInput {
    pub total_files: usize,
    pub converted_count: usize,
    pub vertex_count: usize,
    pub fragment_count: usize,
    pub compute_count: usize,
    pub failures: Vec<FailedShaderRecord>,
}

/// Errors produced when reading or writing `.wgsl.pack` bundles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackError {
    TooShort { actual: usize, expected_min: usize },
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    InvalidStage { entry_index: usize, stage: u8 },
    OutOfBounds {
        entry_index: usize,
        offset: u32,
        length: u32,
        total_len: usize,
    },
    OverlappingPayload {
        first_index: usize,
        first_offset: u32,
        first_length: u32,
        second_index: usize,
        second_offset: u32,
        second_length: u32,
    },
    DuplicateHash {
        first_index: usize,
        second_index: usize,
        hash: u64,
    },
    NonCanonicalOrder {
        entry_index: usize,
        previous_hash: u64,
        hash: u64,
    },
    InvalidUtf8 { entry_index: usize, detail: String },
    Overflow(String),
}

impl fmt::Display for PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort {
                actual,
                expected_min,
            } => write!(
                f,
                "pack truncated: {actual} bytes (minimum required: {expected_min})"
            ),
            Self::InvalidMagic(magic) => write!(
                f,
                "invalid pack magic {:?} (expected {:?})",
                magic, PACK_MAGIC
            ),
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported pack version {v} (expected {PACK_VERSION})")
            }
            Self::InvalidStage { entry_index, stage } => {
                write!(f, "invalid stage value {stage} at entry {entry_index}")
            }
            Self::OutOfBounds {
                entry_index,
                offset,
                length,
                total_len,
            } => write!(
                f,
                "entry {entry_index} payload [{offset}..+{length}] exceeds pack bounds (total {total_len} bytes)"
            ),
            Self::OverlappingPayload {
                first_index,
                first_offset,
                first_length,
                second_index,
                second_offset,
                second_length,
            } => write!(
                f,
                "entry {first_index} payload [{first_offset}..+{first_length}] overlaps entry {second_index} payload [{second_offset}..+{second_length}]"
            ),
            Self::DuplicateHash {
                first_index,
                second_index,
                hash,
            } => write!(
                f,
                "duplicate shader hash {} at entries {first_index} and {second_index}",
                format_shader_hash(*hash)
            ),
            Self::NonCanonicalOrder {
                entry_index,
                previous_hash,
                hash,
            } => write!(
                f,
                "non-canonical hash order at entry {entry_index}: {} must be strictly greater than previous {}",
                format_shader_hash(*hash),
                format_shader_hash(*previous_hash)
            ),
            Self::InvalidUtf8 {
                entry_index,
                detail,
            } => write!(f, "entry {entry_index} payload is not valid UTF-8: {detail}"),
            Self::Overflow(msg) => write!(f, "pack size overflow: {msg}"),
        }
    }
}

impl std::error::Error for PackError {}

/// Compute the checked `u32` byte offset where WGSL payloads begin (`header + entry_count * 20`).
pub fn checked_table_data_start(entry_count: usize) -> Result<u32, PackError> {
    let count_u32 = u32::try_from(entry_count)
        .map_err(|_| PackError::Overflow("entry count exceeds u32::MAX".to_string()))?;
    let table_bytes = count_u32
        .checked_mul(PACK_ENTRY_BYTES as u32)
        .ok_or_else(|| {
            PackError::Overflow(format!(
                "entry table byte length overflow for {count_u32} entries"
            ))
        })?;
    let data_start = (PACK_HEADER_BYTES as u32)
        .checked_add(table_bytes)
        .ok_or_else(|| {
            PackError::Overflow(format!(
                "header + entry table size exceeds u32::MAX for {count_u32} entries"
            ))
        })?;
    let _ = usize::try_from(data_start).map_err(|_| {
        PackError::Overflow("header + entry table size exceeds platform usize".to_string())
    })?;
    Ok(data_start)
}

/// Validate that a WGPK v1 pack with `entry_count` entries and `total_payload_bytes` fits within
/// `u32::MAX` byte offsets, returning `(data_start, total_size)`.
pub fn checked_pack_total_size(
    entry_count: usize,
    total_payload_bytes: u64,
) -> Result<(u32, u32), PackError> {
    let data_start = checked_table_data_start(entry_count)?;
    let total_u64 = u64::from(data_start)
        .checked_add(total_payload_bytes)
        .ok_or_else(|| PackError::Overflow("total pack size overflow".to_string()))?;
    let total_u32 = u32::try_from(total_u64).map_err(|_| {
        PackError::Overflow(format!(
            "pack size {total_u64} exceeds u32::MAX ({MAX_PACK_BYTES}) byte offsets"
        ))
    })?;
    let _ = usize::try_from(total_u32)
        .map_err(|_| PackError::Overflow("total pack size exceeds platform usize".to_string()))?;
    Ok((data_start, total_u32))
}

/// Validate one WGPK v1 entry's `[offset..offset+length)` range against `u32` overflow,
/// header/table `data_start`, and `total_len`, returning checked `(start, end)` `usize` bounds.
pub fn checked_entry_bounds(
    entry_index: usize,
    offset: u32,
    length: u32,
    data_start: u32,
    total_len: usize,
) -> Result<(usize, usize), PackError> {
    let end_u32 = offset.checked_add(length).ok_or_else(|| {
        PackError::Overflow(format!(
            "entry {entry_index} payload range [{offset}..+{length}] overflows u32"
        ))
    })?;
    if offset < data_start {
        return Err(PackError::OutOfBounds {
            entry_index,
            offset,
            length,
            total_len,
        });
    }
    let start = usize::try_from(offset).map_err(|_| {
        PackError::Overflow(format!(
            "entry {entry_index} offset {offset} exceeds platform usize"
        ))
    })?;
    let end = usize::try_from(end_u32).map_err(|_| {
        PackError::Overflow(format!(
            "entry {entry_index} end offset {end_u32} exceeds platform usize"
        ))
    })?;
    if end > total_len {
        return Err(PackError::OutOfBounds {
            entry_index,
            offset,
            length,
            total_len,
        });
    }
    Ok((start, end))
}

/// Verify that a sequence of 64-bit shader hashes is in strictly ascending canonical order
/// (no duplicates and no out-of-order entries).
pub fn verify_canonical_hash_order<I>(hashes: I) -> Result<(), PackError>
where
    I: IntoIterator<Item = u64>,
{
    let mut prev: Option<u64> = None;
    for (idx, hash) in hashes.into_iter().enumerate() {
        if let Some(previous_hash) = prev {
            if hash == previous_hash {
                return Err(PackError::DuplicateHash {
                    first_index: idx - 1,
                    second_index: idx,
                    hash,
                });
            }
            if hash < previous_hash {
                return Err(PackError::NonCanonicalOrder {
                    entry_index: idx,
                    previous_hash,
                    hash,
                });
            }
        }
        prev = Some(hash);
    }
    Ok(())
}

fn validate_unique_hashes(entries: &[PackEntry]) -> Result<(), PackError> {
    let mut seen: BTreeMap<u64, usize> = BTreeMap::new();
    for (idx, entry) in entries.iter().enumerate() {
        if let Some(first_index) = seen.insert(entry.hash, idx) {
            return Err(PackError::DuplicateHash {
                first_index,
                second_index: idx,
                hash: entry.hash,
            });
        }
    }
    Ok(())
}

/// Layout result returned by [`write_pack_with_layout`]: raw binary bytes plus `(offset, length)`
/// per input entry in the same order as `entries`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrittenPack {
    pub bytes: Vec<u8>,
    pub ranges: Vec<(u32, u32)>,
}

fn write_pack_indexed(
    entries: &[PackEntry],
    order: &[usize],
    flags: u32,
) -> Result<WrittenPack, PackError> {
    debug_assert_eq!(entries.len(), order.len());

    let mut total_payload_bytes: u64 = 0;
    for &orig_idx in order {
        let len = entries[orig_idx].wgsl.len();
        let len_u32 = u32::try_from(len).map_err(|_| {
            PackError::Overflow(format!(
                "entry {orig_idx} payload length {len} exceeds u32::MAX"
            ))
        })?;
        total_payload_bytes = total_payload_bytes
            .checked_add(u64::from(len_u32))
            .ok_or_else(|| PackError::Overflow("total payload size overflow".to_string()))?;
    }

    let (data_start_u32, total_size_u32) =
        checked_pack_total_size(entries.len(), total_payload_bytes)?;
    let entry_count_u32 = u32::try_from(entries.len())
        .map_err(|_| PackError::Overflow("entry count exceeds u32::MAX".to_string()))?;
    let total_size = total_size_u32 as usize;

    let mut out = Vec::with_capacity(total_size);
    // Header (16 bytes)
    out.extend_from_slice(&PACK_MAGIC);
    out.extend_from_slice(&PACK_VERSION.to_le_bytes());
    out.extend_from_slice(&entry_count_u32.to_le_bytes());
    out.extend_from_slice(&flags.to_le_bytes());

    // Entry table (20 bytes per entry) in `order`, recording `(offset, length)` back at each input index.
    let mut ranges = vec![(0u32, 0u32); entries.len()];
    let mut cursor = data_start_u32;
    for &orig_idx in order {
        let entry = &entries[orig_idx];
        let offset = cursor;
        let length = u32::try_from(entry.wgsl.len()).map_err(|_| {
            PackError::Overflow(format!("entry {orig_idx} payload exceeds u32::MAX"))
        })?;
        let next_cursor = cursor.checked_add(length).ok_or_else(|| {
            PackError::Overflow(format!(
                "entry {orig_idx} payload range [{offset}..+{length}] overflows u32"
            ))
        })?;

        out.extend_from_slice(&entry.hash.to_le_bytes());
        out.push(entry.stage.to_u8());
        out.extend_from_slice(&[0u8; 3]); // reserved
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&length.to_le_bytes());

        ranges[orig_idx] = (offset, length);
        cursor = next_cursor;
    }

    // Concatenated UTF-8 WGSL payloads in `order`
    for &orig_idx in order {
        out.extend_from_slice(entries[orig_idx].wgsl.as_bytes());
    }

    debug_assert_eq!(cursor, total_size_u32);
    debug_assert_eq!(out.len(), total_size);
    Ok(WrittenPack { bytes: out, ranges })
}

/// Serialize WGSL shader entries into the compact binary `.wgsl.pack` format.
///
/// Rejects duplicate hashes and `u32` layout/offset overflows.
pub fn write_pack(entries: &[PackEntry], flags: u32) -> Result<Vec<u8>, PackError> {
    Ok(write_pack_with_layout(entries, flags)?.bytes)
}

/// Serialize WGSL shader entries into `.wgsl.pack` and also return each entry's `(offset, length)`.
///
/// Rejects duplicate hashes and `u32` layout/offset overflows.
pub fn write_pack_with_layout(entries: &[PackEntry], flags: u32) -> Result<WrittenPack, PackError> {
    validate_unique_hashes(entries)?;
    let order: Vec<usize> = (0..entries.len()).collect();
    write_pack_indexed(entries, &order, flags)
}

/// Serialize WGSL shader entries into `.wgsl.pack`, requiring inputs to already be in strictly
/// ascending 64-bit hash order.
pub fn write_pack_strict(entries: &[PackEntry], flags: u32) -> Result<Vec<u8>, PackError> {
    Ok(write_pack_with_layout_strict(entries, flags)?.bytes)
}

/// Serialize WGSL shader entries into `.wgsl.pack` with layout ranges, requiring inputs to already
/// be in strictly ascending 64-bit hash order.
pub fn write_pack_with_layout_strict(
    entries: &[PackEntry],
    flags: u32,
) -> Result<WrittenPack, PackError> {
    verify_canonical_hash_order(entries.iter().map(|entry| entry.hash))?;
    let order: Vec<usize> = (0..entries.len()).collect();
    write_pack_indexed(entries, &order, flags)
}

/// Serialize WGSL shader entries into a deterministic canonical `.wgsl.pack` sorted by strictly
/// ascending 64-bit hash (rejecting duplicate hashes).
pub fn write_pack_canonical(entries: &[PackEntry], flags: u32) -> Result<Vec<u8>, PackError> {
    Ok(write_pack_canonical_with_layout(entries, flags)?.bytes)
}

/// Serialize WGSL shader entries into a deterministic canonical `.wgsl.pack` sorted by strictly
/// ascending 64-bit hash, returning `ranges[i]` corresponding to input `entries[i]`.
pub fn write_pack_canonical_with_layout(
    entries: &[PackEntry],
    flags: u32,
) -> Result<WrittenPack, PackError> {
    validate_unique_hashes(entries)?;
    let mut order: Vec<usize> = (0..entries.len()).collect();
    order.sort_unstable_by_key(|&idx| entries[idx].hash);
    write_pack_indexed(entries, &order, flags)
}

struct ParsedTocRecord {
    hash: u64,
    stage: ShaderStage,
    offset: u32,
    length: u32,
    start: usize,
    end: usize,
}

fn read_pack_internal(bytes: &[u8], require_canonical_order: bool) -> Result<ShaderPack, PackError> {
    if bytes.len() < PACK_HEADER_BYTES {
        return Err(PackError::TooShort {
            actual: bytes.len(),
            expected_min: PACK_HEADER_BYTES,
        });
    }

    let magic: [u8; 4] = bytes[0..4].try_into().unwrap();
    if magic != PACK_MAGIC {
        return Err(PackError::InvalidMagic(magic));
    }

    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != PACK_VERSION {
        return Err(PackError::UnsupportedVersion(version));
    }

    let entry_count_u32 = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
    let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());

    if u32::try_from(bytes.len()).is_err() {
        return Err(PackError::Overflow(format!(
            "pack size {} exceeds u32::MAX ({MAX_PACK_BYTES}) byte offsets",
            bytes.len()
        )));
    }

    let entry_count = usize::try_from(entry_count_u32)
        .map_err(|_| PackError::Overflow("entry count exceeds platform usize".to_string()))?;
    let data_start_u32 = checked_table_data_start(entry_count)?;
    let data_start = data_start_u32 as usize;

    if bytes.len() < data_start {
        return Err(PackError::TooShort {
            actual: bytes.len(),
            expected_min: data_start,
        });
    }

    // Phase 1: Validate all TOC metadata, u32 range bounds, hash uniqueness/order, and
    // non-overlapping payload spans BEFORE allocating or UTF-8 decoding any WGSL strings.
    let mut toc = Vec::with_capacity(entry_count);
    let mut spans: Vec<(u32, u32, usize, u32)> = Vec::with_capacity(entry_count);
    let mut seen_hashes: BTreeMap<u64, usize> = BTreeMap::new();
    let mut prev_hash: Option<u64> = None;

    for idx in 0..entry_count {
        let base = idx
            .checked_mul(PACK_ENTRY_BYTES)
            .and_then(|offset| PACK_HEADER_BYTES.checked_add(offset))
            .ok_or_else(|| PackError::Overflow("entry table offset overflow".to_string()))?;
        let hash = u64::from_le_bytes(bytes[base..base + 8].try_into().unwrap());
        let stage_raw = bytes[base + 8];
        let stage = ShaderStage::from_u8(stage_raw).ok_or(PackError::InvalidStage {
            entry_index: idx,
            stage: stage_raw,
        })?;
        // bytes[base + 9..base + 12] are reserved for forward compatibility.
        let offset = u32::from_le_bytes(bytes[base + 12..base + 16].try_into().unwrap());
        let length = u32::from_le_bytes(bytes[base + 16..base + 20].try_into().unwrap());

        let (start, end) = checked_entry_bounds(idx, offset, length, data_start_u32, bytes.len())?;

        if require_canonical_order {
            if let Some(previous_hash) = prev_hash {
                if hash == previous_hash {
                    return Err(PackError::DuplicateHash {
                        first_index: idx - 1,
                        second_index: idx,
                        hash,
                    });
                }
                if hash < previous_hash {
                    return Err(PackError::NonCanonicalOrder {
                        entry_index: idx,
                        previous_hash,
                        hash,
                    });
                }
            }
            prev_hash = Some(hash);
        } else if let Some(first_index) = seen_hashes.insert(hash, idx) {
            return Err(PackError::DuplicateHash {
                first_index,
                second_index: idx,
                hash,
            });
        }

        // `end` is guaranteed to fit in `u32` because `checked_entry_bounds` validated `offset + length <= u32::MAX`.
        let end_u32 = offset + length;
        spans.push((offset, end_u32, idx, length));
        toc.push(ParsedTocRecord {
            hash,
            stage,
            offset,
            length,
            start,
            end,
        });
    }

    // Reject overlapping payload ranges (including identical non-empty ranges, partial overlaps,
    // containment, and zero-length entries placed strictly inside a non-empty payload).
    if spans.len() > 1 {
        spans.sort_unstable_by_key(|&(start, end, idx, _len)| (start, end, idx));
        for pair in spans.windows(2) {
            let (first_offset, first_end, first_index, first_length) = pair[0];
            let (second_offset, _second_end, second_index, second_length) = pair[1];
            if second_offset < first_end {
                return Err(PackError::OverlappingPayload {
                    first_index,
                    first_offset,
                    first_length,
                    second_index,
                    second_offset,
                    second_length,
                });
            }
        }
    }

    // Phase 2: Decode validated, non-overlapping slices as strict UTF-8 WGSL.
    let mut entries = Vec::with_capacity(entry_count);
    for (idx, rec) in toc.into_iter().enumerate() {
        let wgsl = std::str::from_utf8(&bytes[rec.start..rec.end])
            .map_err(|err| PackError::InvalidUtf8 {
                entry_index: idx,
                detail: err.to_string(),
            })?
            .to_string();

        entries.push(DecodedPackEntry {
            hash: rec.hash,
            stage: rec.stage,
            offset: rec.offset,
            length: rec.length,
            wgsl,
        });
    }

    Ok(ShaderPack {
        version,
        flags,
        entries,
    })
}

/// Deserialize and strictly validate a binary `.wgsl.pack` buffer (checked header/table bounds,
/// `u32` range overflow, duplicate-hash rejection, non-overlapping payloads, and strict UTF-8).
pub fn read_pack(bytes: &[u8]) -> Result<ShaderPack, PackError> {
    read_pack_internal(bytes, false)
}

/// Deserialize and strictly validate a canonical binary `.wgsl.pack` buffer, additionally
/// requiring entries to be ordered by strictly ascending 64-bit hash.
pub fn read_pack_canonical(bytes: &[u8]) -> Result<ShaderPack, PackError> {
    read_pack_internal(bytes, true)
}

/// Render `shader-index.json` matching `docs/gpu_commands.md` (`abi_version: 1`, `pack`, `shaders`).
pub fn render_shader_index_json(pack_path: Option<&str>, entries: &[ShaderIndexEntry]) -> String {
    // Deterministic ordering by canonical hash key.
    let mut sorted: Vec<&ShaderIndexEntry> = entries.iter().collect();
    sorted.sort_by_key(|e| e.hash);

    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"abi_version\": {SHADER_INDEX_ABI_VERSION},");
    match pack_path {
        Some(path) => {
            out.push_str("  \"pack\": ");
            write_json_string(&mut out, path);
            out.push_str(",\n");
        }
        None => {
            out.push_str("  \"pack\": null,\n");
        }
    }

    if sorted.is_empty() {
        out.push_str("  \"shaders\": {}\n");
    } else {
        out.push_str("  \"shaders\": {\n");
        for (i, entry) in sorted.iter().enumerate() {
            let key = format_shader_hash(entry.hash);
            out.push_str("    ");
            write_json_string(&mut out, &key);
            out.push_str(": {\n");

            out.push_str("      \"stage\": ");
            write_json_string(&mut out, entry.stage.as_str());
            out.push_str(",\n");

            out.push_str("      \"wgsl\": ");
            write_json_string(&mut out, &entry.wgsl_path);
            out.push_str(",\n");

            out.push_str("      \"entry_point\": ");
            write_json_string(&mut out, &entry.entry_point);

            if let (Some(offset), Some(length)) = (entry.pack_offset, entry.pack_length) {
                out.push_str(",\n");
                let _ = writeln!(out, "      \"pack_offset\": {offset},");
                let _ = writeln!(out, "      \"pack_length\": {length}");
            } else {
                out.push('\n');
            }

            if i + 1 < sorted.len() {
                out.push_str("    },\n");
            } else {
                out.push_str("    }\n");
            }
        }
        out.push_str("  }\n");
    }

    out.push_str("}\n");
    out
}

/// Write a properly escaped JSON string literal (including quotes) into `out`.
pub fn write_json_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Render a structured Markdown WebGPU blocker report summarizing converted vs failed shaders.
pub fn render_blocker_report_md(input: &BlockerReportInput) -> String {
    let failed_count = input.failures.len();
    let total = input.total_files.max(input.converted_count + failed_count);
    let pass_rate = if total == 0 {
        0.0
    } else {
        (input.converted_count as f64 / total as f64) * 100.0
    };

    let mut out = String::new();
    out.push_str("# SPIR-V -> WGSL WebGPU Blocker Report\n\n");
    out.push_str("## Summary\n\n");
    out.push_str("| Metric | Count |\n");
    out.push_str("| :--- | ---: |\n");
    let _ = writeln!(out, "| Total `.spv` files | {total} |");
    let _ = writeln!(
        out,
        "| Converted to WGSL | {} ({pass_rate:.1}%) |",
        input.converted_count
    );
    let _ = writeln!(out, "| Failed | {failed_count} |");
    let _ = writeln!(out, "| Vertex shaders | {} |", input.vertex_count);
    let _ = writeln!(out, "| Fragment shaders | {} |", input.fragment_count);
    let _ = writeln!(out, "| Compute shaders | {} |", input.compute_count);
    out.push('\n');

    if input.failures.is_empty() {
        out.push_str("## Blockers\n\n");
        out.push_str("No conversion failures detected.\n");
        return out;
    }

    // Group by high-level error kind.
    let mut by_kind: BTreeMap<ConversionErrorKind, usize> = BTreeMap::new();
    for fail in &input.failures {
        *by_kind.entry(fail.error.kind).or_insert(0) += 1;
    }

    out.push_str("## Failures by Error Category\n\n");
    out.push_str("| Category | Count |\n");
    out.push_str("| :--- | ---: |\n");
    for (kind, count) in &by_kind {
        let _ = writeln!(out, "| `{kind}` | {count} |");
    }
    out.push('\n');

    // Group by normalized blocker key, sorted by descending count then alphabetically.
    struct BlockerGroup<'a> {
        kind: ConversionErrorKind,
        count: usize,
        keywords: Vec<String>,
        samples: Vec<&'a FailedShaderRecord>,
    }

    let mut grouped: BTreeMap<&str, BlockerGroup<'_>> = BTreeMap::new();
    for fail in &input.failures {
        let entry = grouped
            .entry(fail.error.blocker_key.as_str())
            .or_insert_with(|| BlockerGroup {
                kind: fail.error.kind,
                count: 0,
                keywords: Vec::new(),
                samples: Vec::new(),
            });
        entry.count += 1;
        for kw in &fail.error.keywords {
            if !entry.keywords.iter().any(|existing| existing == kw) {
                entry.keywords.push(kw.clone());
            }
        }
        if entry.samples.len() < 5 {
            entry.samples.push(fail);
        }
    }

    let mut ranked: Vec<(&str, BlockerGroup<'_>)> = grouped.into_iter().collect();
    ranked.sort_by(|(k1, g1), (k2, g2)| g2.count.cmp(&g1.count).then_with(|| k1.cmp(k2)));

    out.push_str("## Grouped WebGPU Blockers\n\n");
    out.push_str("| Blocker | Category | Count | Keywords | Sample Shaders |\n");
    out.push_str("| :--- | :--- | ---: | :--- | :--- |\n");
    for (key, group) in &ranked {
        let kw_cell = if group.keywords.is_empty() {
            "-".to_string()
        } else {
            group
                .keywords
                .iter()
                .map(|k| format!("`{}`", escape_md_cell(k)))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let samples_cell = group
            .samples
            .iter()
            .map(|s| format!("`{}`", escape_md_cell(&s.source_path)))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(
            out,
            "| `{}` | `{}` | {} | {} | {} |",
            escape_md_cell(key),
            group.kind,
            group.count,
            kw_cell,
            samples_cell
        );
    }
    out.push('\n');

    out.push_str("## Failure Details\n\n");
    for fail in &input.failures {
        let first_line = fail
            .error
            .detail
            .lines()
            .find(|l| !l.trim().is_empty())
            .unwrap_or(&fail.error.detail)
            .trim();
        let _ = writeln!(
            out,
            "- `{}` — `{}` (`{}`): {}",
            escape_md_cell(&fail.source_path),
            fail.error.kind,
            escape_md_cell(&fail.error.blocker_key),
            first_line
        );
    }

    out
}

fn escape_md_cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::{
        convert_spirv_bytes, format_wgsl_filename,
        test_spirv::{
            build_minimal_compute_spv, build_minimal_fragment_spv, build_minimal_vertex_spv,
            build_unsupported_capability_spv, build_unsupported_extension_spv,
        },
    };

    #[test]
    fn pack_round_trip_with_converted_spirv_shaders() {
        let compute = convert_spirv_bytes(&build_minimal_compute_spv()).unwrap();
        let vertex = convert_spirv_bytes(&build_minimal_vertex_spv()).unwrap();
        let fragment = convert_spirv_bytes(&build_minimal_fragment_spv()).unwrap();

        let pack_entries = vec![
            PackEntry {
                hash: compute.hash,
                stage: compute.stage,
                wgsl: compute.wgsl.clone(),
            },
            PackEntry {
                hash: vertex.hash,
                stage: vertex.stage,
                wgsl: vertex.wgsl.clone(),
            },
            PackEntry {
                hash: fragment.hash,
                stage: fragment.stage,
                wgsl: fragment.wgsl.clone(),
            },
        ];

        let written = write_pack_with_layout(&pack_entries, 0).expect("pack write should succeed");
        assert_eq!(&written.bytes[0..4], b"WGPK");
        assert_eq!(
            u32::from_le_bytes(written.bytes[4..8].try_into().unwrap()),
            1
        );
        assert_eq!(
            u32::from_le_bytes(written.bytes[8..12].try_into().unwrap()),
            3
        );
        assert_eq!(
            u32::from_le_bytes(written.bytes[12..16].try_into().unwrap()),
            0
        );

        let expected_data_start = (PACK_HEADER_BYTES + 3 * PACK_ENTRY_BYTES) as u32;
        assert_eq!(written.ranges[0].0, expected_data_start);

        let decoded = read_pack(&written.bytes).expect("pack read should succeed");
        assert_eq!(decoded.version, PACK_VERSION);
        assert_eq!(decoded.flags, 0);
        assert_eq!(decoded.entries.len(), 3);

        for (i, original) in pack_entries.iter().enumerate() {
            let entry = &decoded.entries[i];
            assert_eq!(entry.hash, original.hash);
            assert_eq!(entry.stage, original.stage);
            assert_eq!(entry.wgsl, original.wgsl);
            assert_eq!((entry.offset, entry.length), written.ranges[i]);
        }

        // Verify shader-index.json output matches docs/gpu_commands.md schema.
        let index_entries = vec![
            ShaderIndexEntry {
                hash: compute.hash,
                stage: compute.stage,
                wgsl_path: format!("shaders/{}", format_wgsl_filename(compute.hash)),
                entry_point: compute.entry_point.clone(),
                pack_offset: Some(written.ranges[0].0),
                pack_length: Some(written.ranges[0].1),
            },
            ShaderIndexEntry {
                hash: vertex.hash,
                stage: vertex.stage,
                wgsl_path: format!("shaders/{}", format_wgsl_filename(vertex.hash)),
                entry_point: vertex.entry_point.clone(),
                pack_offset: Some(written.ranges[1].0),
                pack_length: Some(written.ranges[1].1),
            },
        ];

        let json = render_shader_index_json(Some("shaders.wgsl.pack"), &index_entries);
        assert!(json.contains("\"abi_version\": 1"));
        assert!(json.contains("\"pack\": \"shaders.wgsl.pack\""));
        assert!(json.contains(&format_shader_hash(compute.hash)));
        assert!(json.contains(&format_shader_hash(vertex.hash)));
        assert!(json.contains("\"stage\": \"compute\""));
        assert!(json.contains("\"stage\": \"vertex\""));
        assert!(json.contains("\"pack_offset\":"));
        assert!(json.contains("\"pack_length\":"));
    }

    #[test]
    fn read_pack_validates_magic_version_bounds_and_utf8() {
        let valid = write_pack(
            &[PackEntry {
                hash: 0x1122_3344_5566_7788,
                stage: ShaderStage::Compute,
                wgsl: "@compute @workgroup_size(1) fn main() {}".to_string(),
            }],
            0,
        )
        .unwrap();

        // Too short
        assert!(matches!(
            read_pack(&valid[..8]),
            Err(PackError::TooShort { .. })
        ));

        // Bad magic
        let mut bad_magic = valid.clone();
        bad_magic[0..4].copy_from_slice(b"NOPE");
        assert!(matches!(
            read_pack(&bad_magic),
            Err(PackError::InvalidMagic(_))
        ));

        // Bad version
        let mut bad_version = valid.clone();
        bad_version[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert_eq!(
            read_pack(&bad_version),
            Err(PackError::UnsupportedVersion(99))
        );

        // Invalid stage
        let mut bad_stage = valid.clone();
        bad_stage[PACK_HEADER_BYTES + 8] = 9;
        assert_eq!(
            read_pack(&bad_stage),
            Err(PackError::InvalidStage {
                entry_index: 0,
                stage: 9
            })
        );

        // Out-of-bounds offset
        let mut oob = valid.clone();
        oob[PACK_HEADER_BYTES + 12..PACK_HEADER_BYTES + 16]
            .copy_from_slice(&9999u32.to_le_bytes());
        assert!(matches!(read_pack(&oob), Err(PackError::OutOfBounds { .. })));

        // Offset pointing inside header/table
        let mut overlap = valid.clone();
        overlap[PACK_HEADER_BYTES + 12..PACK_HEADER_BYTES + 16]
            .copy_from_slice(&4u32.to_le_bytes());
        assert!(matches!(
            read_pack(&overlap),
            Err(PackError::OutOfBounds { .. })
        ));

        // Invalid UTF-8 payload
        let mut bad_utf8 = valid.clone();
        let payload_start = PACK_HEADER_BYTES + PACK_ENTRY_BYTES;
        bad_utf8[payload_start] = 0xff;
        assert!(matches!(
            read_pack(&bad_utf8),
            Err(PackError::InvalidUtf8 { entry_index: 0, .. })
        ));
    }

    #[test]
    fn json_string_escaping_handles_special_chars() {
        let mut buf = String::new();
        write_json_string(&mut buf, "a\"b\\c\nd\te\u{0007}");
        assert_eq!(buf, "\"a\\\"b\\\\c\\nd\\te\\u0007\"");

        let json = render_shader_index_json(None, &[]);
        assert!(json.contains("\"abi_version\": 1"));
        assert!(json.contains("\"pack\": null"));
        assert!(json.contains("\"shaders\": {}"));
    }

    #[test]
    fn blocker_report_groups_failures_and_summarizes_counts() {
        let err_ext = convert_spirv_bytes(&build_unsupported_extension_spv()).unwrap_err();
        let err_cap = convert_spirv_bytes(&build_unsupported_capability_spv(true)).unwrap_err();
        let err_hdr = convert_spirv_bytes(&[1, 2, 3]).unwrap_err();

        let report = render_blocker_report_md(&BlockerReportInput {
            total_files: 5,
            converted_count: 2,
            vertex_count: 1,
            fragment_count: 0,
            compute_count: 1,
            failures: vec![
                FailedShaderRecord {
                    source_path: "rt/ray1.spv".to_string(),
                    error: err_ext.clone(),
                },
                FailedShaderRecord {
                    source_path: "rt/ray2.spv".to_string(),
                    error: err_ext,
                },
                FailedShaderRecord {
                    source_path: "rt/query.spv".to_string(),
                    error: err_cap,
                },
                FailedShaderRecord {
                    source_path: "corrupt.spv".to_string(),
                    error: err_hdr,
                },
            ],
        });

        assert!(report.contains("# SPIR-V -> WGSL WebGPU Blocker Report"));
        assert!(report.contains("SPV_KHR_ray_tracing"));
        assert!(report.contains("InvalidSpirvHeader"));
        assert!(report.contains("SpirvParseError"));
        assert!(report.contains("rt/ray1.spv"));
    }

    fn build_raw_wgpk(
        flags: u32,
        records: &[(u64, u8, u32, u32)],
        payload: &[u8],
    ) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            PACK_HEADER_BYTES + records.len() * PACK_ENTRY_BYTES + payload.len(),
        );
        out.extend_from_slice(&PACK_MAGIC);
        out.extend_from_slice(&PACK_VERSION.to_le_bytes());
        out.extend_from_slice(&(records.len() as u32).to_le_bytes());
        out.extend_from_slice(&flags.to_le_bytes());
        for &(hash, stage, offset, length) in records {
            out.extend_from_slice(&hash.to_le_bytes());
            out.push(stage);
            out.extend_from_slice(&[0u8; 3]);
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&length.to_le_bytes());
        }
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn empty_pack_round_trips_and_is_canonical() {
        let flags = 0x1234_5678;
        let bytes = write_pack(&[], flags).expect("empty write_pack succeeds");
        assert_eq!(bytes.len(), PACK_HEADER_BYTES);
        assert_eq!(write_pack_strict(&[], flags).unwrap(), bytes);
        assert_eq!(write_pack_canonical(&[], flags).unwrap(), bytes);

        let decoded = read_pack(&bytes).expect("empty read_pack succeeds");
        assert_eq!(decoded.version, PACK_VERSION);
        assert_eq!(decoded.flags, flags);
        assert!(decoded.entries.is_empty());
        assert!(decoded.is_canonical_hash_order());
        assert_eq!(decoded.verify_canonical_hash_order(), Ok(()));

        let canonical = read_pack_canonical(&bytes).expect("empty read_pack_canonical succeeds");
        assert_eq!(canonical, decoded);
    }

    #[test]
    fn checked_layout_and_entry_bounds_u32_overflow_boundaries() {
        assert_eq!(checked_table_data_start(0), Ok(PACK_HEADER_BYTES as u32));
        assert_eq!(
            checked_table_data_start(MAX_PACK_ENTRIES as usize),
            Ok(4_294_967_276)
        );
        assert!(matches!(
            checked_table_data_start(MAX_PACK_ENTRIES as usize + 1),
            Err(PackError::Overflow(_))
        ));
        assert!(matches!(
            checked_table_data_start(u32::MAX as usize),
            Err(PackError::Overflow(_))
        ));

        // Exact u32::MAX total pack size succeeds; +1 byte overflows.
        let max_payload_at_max_entries = u64::from(u32::MAX - 4_294_967_276);
        assert_eq!(max_payload_at_max_entries, 19);
        assert_eq!(
            checked_pack_total_size(MAX_PACK_ENTRIES as usize, max_payload_at_max_entries),
            Ok((4_294_967_276, u32::MAX))
        );
        assert!(matches!(
            checked_pack_total_size(MAX_PACK_ENTRIES as usize, max_payload_at_max_entries + 1),
            Err(PackError::Overflow(_))
        ));
        assert!(matches!(
            checked_pack_total_size(0, u64::MAX),
            Err(PackError::Overflow(_))
        ));

        // Exact u32::MAX entry range bounds vs u32 overflow.
        assert_eq!(
            checked_entry_bounds(0, 36, 10, 36, 46),
            Ok((36, 46))
        );
        assert_eq!(
            checked_entry_bounds(0, 46, 0, 36, 46),
            Ok((46, 46))
        );
        assert!(matches!(
            checked_entry_bounds(0, u32::MAX - 4, 5, 36, 100),
            Err(PackError::Overflow(_))
        ));
        assert!(matches!(
            checked_entry_bounds(0, u32::MAX, 1, 36, 100),
            Err(PackError::Overflow(_))
        ));
        assert!(matches!(
            checked_entry_bounds(0, u32::MAX, 0, 36, 100),
            Err(PackError::OutOfBounds { entry_index: 0, offset: u32::MAX, length: 0, total_len: 100 })
        ));
        assert!(matches!(
            checked_entry_bounds(2, 35, 0, 36, 100),
            Err(PackError::OutOfBounds { entry_index: 2, offset: 35, length: 0, total_len: 100 })
        ));
    }

    #[test]
    fn read_pack_rejects_u32_table_and_entry_range_overflow() {
        // Header claiming MAX_PACK_ENTRIES + 1 overflows u32 data_start before any allocation.
        let mut hdr_add_overflow = Vec::with_capacity(PACK_HEADER_BYTES);
        hdr_add_overflow.extend_from_slice(&PACK_MAGIC);
        hdr_add_overflow.extend_from_slice(&PACK_VERSION.to_le_bytes());
        hdr_add_overflow.extend_from_slice(&(MAX_PACK_ENTRIES + 1).to_le_bytes());
        hdr_add_overflow.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            read_pack(&hdr_add_overflow),
            Err(PackError::Overflow(_))
        ));
        assert!(matches!(
            read_pack_canonical(&hdr_add_overflow),
            Err(PackError::Overflow(_))
        ));

        // Header claiming u32::MAX entries overflows u32 table multiplication.
        let mut hdr_mul_overflow = hdr_add_overflow.clone();
        hdr_mul_overflow[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            read_pack(&hdr_mul_overflow),
            Err(PackError::Overflow(_))
        ));

        // Header claiming MAX_PACK_ENTRIES on a 16-byte buffer fails with TooShort (no OOM).
        let mut hdr_truncated = hdr_add_overflow.clone();
        hdr_truncated[8..12].copy_from_slice(&MAX_PACK_ENTRIES.to_le_bytes());
        assert_eq!(
            read_pack(&hdr_truncated),
            Err(PackError::TooShort {
                actual: PACK_HEADER_BYTES,
                expected_min: 4_294_967_276,
            })
        );

        // Single entry whose offset + length overflows u32.
        let data_start = (PACK_HEADER_BYTES + PACK_ENTRY_BYTES) as u32;
        let wrap_offset = build_raw_wgpk(
            0,
            &[(0x01, ShaderStage::Compute.to_u8(), u32::MAX - 3, 8)],
            b"fn f() {}",
        );
        assert!(matches!(
            read_pack(&wrap_offset),
            Err(PackError::Overflow(_))
        ));

        let wrap_len = build_raw_wgpk(
            0,
            &[(0x01, ShaderStage::Compute.to_u8(), data_start, u32::MAX)],
            b"fn f() {}",
        );
        assert!(matches!(read_pack(&wrap_len), Err(PackError::Overflow(_))));
    }

    #[test]
    fn read_pack_rejects_overlapping_payload_ranges() {
        let data_start = (PACK_HEADER_BYTES + 2 * PACK_ENTRY_BYTES) as u32;
        let payload = b"@vertex fn vs() {}@fragment fn fs() {}";

        // 1. Partial overlap: [data_start..+10) and [data_start+9..+10)
        let partial = build_raw_wgpk(
            0,
            &[
                (0x10, ShaderStage::Vertex.to_u8(), data_start, 10),
                (0x20, ShaderStage::Fragment.to_u8(), data_start + 9, 10),
            ],
            payload,
        );
        assert_eq!(
            read_pack(&partial),
            Err(PackError::OverlappingPayload {
                first_index: 0,
                first_offset: data_start,
                first_length: 10,
                second_index: 1,
                second_offset: data_start + 9,
                second_length: 10,
            })
        );

        // 2. Identical non-empty range (aliased payloads)
        let identical = build_raw_wgpk(
            0,
            &[
                (0x10, ShaderStage::Vertex.to_u8(), data_start, 18),
                (0x20, ShaderStage::Fragment.to_u8(), data_start, 18),
            ],
            payload,
        );
        assert_eq!(
            read_pack(&identical),
            Err(PackError::OverlappingPayload {
                first_index: 0,
                first_offset: data_start,
                first_length: 18,
                second_index: 1,
                second_offset: data_start,
                second_length: 18,
            })
        );

        // 3. Nested/contained range with out-of-order physical offsets
        let nested = build_raw_wgpk(
            0,
            &[
                (0x10, ShaderStage::Vertex.to_u8(), data_start + 4, 8),
                (0x20, ShaderStage::Fragment.to_u8(), data_start, 18),
            ],
            payload,
        );
        assert_eq!(
            read_pack(&nested),
            Err(PackError::OverlappingPayload {
                first_index: 1,
                first_offset: data_start,
                first_length: 18,
                second_index: 0,
                second_offset: data_start + 4,
                second_length: 8,
            })
        );

        // 4. Zero-length entry positioned strictly inside another non-empty payload
        let zero_inside = build_raw_wgpk(
            0,
            &[
                (0x10, ShaderStage::Vertex.to_u8(), data_start, 18),
                (0x20, ShaderStage::Fragment.to_u8(), data_start + 5, 0),
            ],
            payload,
        );
        assert_eq!(
            read_pack(&zero_inside),
            Err(PackError::OverlappingPayload {
                first_index: 0,
                first_offset: data_start,
                first_length: 18,
                second_index: 1,
                second_offset: data_start + 5,
                second_length: 0,
            })
        );

        // 5. Phase-1 short-circuit: overlap is rejected before UTF-8 decoding invalid bytes
        let invalid_utf8_aliased = build_raw_wgpk(
            0,
            &[
                (0x10, ShaderStage::Compute.to_u8(), data_start, 4),
                (0x20, ShaderStage::Compute.to_u8(), data_start, 4),
            ],
            &[0xff, 0xfe, 0xfd, 0xfc],
        );
        assert!(matches!(
            read_pack(&invalid_utf8_aliased),
            Err(PackError::OverlappingPayload { .. })
        ));
    }

    #[test]
    fn read_pack_accepts_non_overlapping_reordered_and_boundary_empty_payloads() {
        // Empty WGSL entries at start, middle, and exact EOF round-trip through writer & reader.
        let entries = vec![
            PackEntry {
                hash: 0x01,
                stage: ShaderStage::Vertex,
                wgsl: String::new(),
            },
            PackEntry {
                hash: 0x02,
                stage: ShaderStage::Fragment,
                wgsl: "@fragment fn fs_main() {}".to_string(),
            },
            PackEntry {
                hash: 0x03,
                stage: ShaderStage::Compute,
                wgsl: String::new(),
            },
            PackEntry {
                hash: 0x04,
                stage: ShaderStage::Compute,
                wgsl: String::new(),
            },
        ];

        let bytes = write_pack_strict(&entries, 0).expect("strict write with empty entries succeeds");
        let decoded = read_pack_canonical(&bytes).expect("canonical read with empty entries succeeds");
        assert_eq!(decoded.entries.len(), 4);
        assert_eq!(decoded.entries[0].wgsl, "");
        assert_eq!(decoded.entries[1].wgsl, "@fragment fn fs_main() {}");
        assert_eq!(decoded.entries[2].wgsl, "");
        assert_eq!(decoded.entries[3].wgsl, "");
        assert_eq!(decoded.entries[2].offset, decoded.entries[3].offset);
        assert_eq!(decoded.entries[3].offset as usize, bytes.len());

        // Non-overlapping payloads stored in reverse physical order with padding bytes between them.
        let data_start = (PACK_HEADER_BYTES + 2 * PACK_ENTRY_BYTES) as u32;
        let payload = b"SECOND____FIRST";
        let reordered = build_raw_wgpk(
            7,
            &[
                (0x10, ShaderStage::Vertex.to_u8(), data_start + 10, 5),
                (0x20, ShaderStage::Fragment.to_u8(), data_start, 6),
            ],
            payload,
        );
        let parsed = read_pack_canonical(&reordered).expect("non-overlapping physical order succeeds");
        assert_eq!(parsed.flags, 7);
        assert_eq!(parsed.entries[0].wgsl, "FIRST");
        assert_eq!(parsed.entries[1].wgsl, "SECOND");
    }

    #[test]
    fn writer_and_reader_reject_duplicate_hashes() {
        let dup_entries = vec![
            PackEntry {
                hash: 0xaaaa,
                stage: ShaderStage::Vertex,
                wgsl: "@vertex fn vs() {}".to_string(),
            },
            PackEntry {
                hash: 0xbbbb,
                stage: ShaderStage::Fragment,
                wgsl: "@fragment fn fs() {}".to_string(),
            },
            PackEntry {
                hash: 0xaaaa,
                stage: ShaderStage::Compute,
                wgsl: "@compute @workgroup_size(1) fn cs() {}".to_string(),
            },
        ];

        let expected_err = PackError::DuplicateHash {
            first_index: 0,
            second_index: 2,
            hash: 0xaaaa,
        };
        assert_eq!(write_pack(&dup_entries, 0), Err(expected_err.clone()));
        assert_eq!(
            write_pack_with_layout(&dup_entries, 0),
            Err(expected_err.clone())
        );
        assert_eq!(
            write_pack_canonical(&dup_entries, 0),
            Err(expected_err.clone())
        );
        assert_eq!(
            write_pack_canonical_with_layout(&dup_entries, 0),
            Err(expected_err.clone())
        );

        // Non-adjacent duplicate in raw bytes is rejected by read_pack before UTF-8 decode.
        let data_start = (PACK_HEADER_BYTES + 3 * PACK_ENTRY_BYTES) as u32;
        let raw_non_adj_dup = build_raw_wgpk(
            0,
            &[
                (0x10, ShaderStage::Vertex.to_u8(), data_start, 2),
                (0x20, ShaderStage::Fragment.to_u8(), data_start + 2, 2),
                (0x10, ShaderStage::Compute.to_u8(), data_start + 4, 2),
            ],
            &[0xff, 0xff, 0xff, 0xff, 0xff, 0xff],
        );
        assert_eq!(
            read_pack(&raw_non_adj_dup),
            Err(PackError::DuplicateHash {
                first_index: 0,
                second_index: 2,
                hash: 0x10,
            })
        );

        // Adjacent duplicate in raw bytes is rejected by read_pack_canonical with DuplicateHash.
        let data_start_2 = (PACK_HEADER_BYTES + 2 * PACK_ENTRY_BYTES) as u32;
        let raw_adj_dup = build_raw_wgpk(
            0,
            &[
                (0x42, ShaderStage::Vertex.to_u8(), data_start_2, 2),
                (0x42, ShaderStage::Fragment.to_u8(), data_start_2 + 2, 2),
            ],
            b"abcd",
        );
        assert_eq!(
            read_pack_canonical(&raw_adj_dup),
            Err(PackError::DuplicateHash {
                first_index: 0,
                second_index: 1,
                hash: 0x42,
            })
        );
    }

    #[test]
    fn canonical_ascending_hash_order_enforcement_and_determinism() {
        let a = PackEntry {
            hash: 0x0000_0000_0000_0001,
            stage: ShaderStage::Vertex,
            wgsl: "@vertex fn vs_main() {}".to_string(),
        };
        let b = PackEntry {
            hash: 0x0000_0000_0000_0050,
            stage: ShaderStage::Fragment,
            wgsl: "@fragment fn fs_main() {}".to_string(),
        };
        let c = PackEntry {
            hash: u64::MAX - 1,
            stage: ShaderStage::Compute,
            wgsl: "@compute @workgroup_size(1) fn cs_main() {}".to_string(),
        };

        // Canonical writer produces byte-for-byte identical packs across input permutations.
        let perm1 = vec![c.clone(), a.clone(), b.clone()];
        let perm2 = vec![b.clone(), c.clone(), a.clone()];
        let sorted = vec![a.clone(), b.clone(), c.clone()];

        let canon1 = write_pack_canonical_with_layout(&perm1, 3).unwrap();
        let canon2 = write_pack_canonical_with_layout(&perm2, 3).unwrap();
        let strict = write_pack_with_layout_strict(&sorted, 3).unwrap();
        assert_eq!(canon1.bytes, canon2.bytes);
        assert_eq!(canon1.bytes, strict.bytes);

        // `canon1.ranges[i]` matches input `perm1[i]`'s exact bytes inside the canonical pack.
        for (idx, entry) in perm1.iter().enumerate() {
            let (off, len) = canon1.ranges[idx];
            let slice = &canon1.bytes[off as usize..(off + len) as usize];
            assert_eq!(slice, entry.wgsl.as_bytes());
        }

        let decoded_canon = read_pack_canonical(&canon1.bytes).unwrap();
        assert!(decoded_canon.is_canonical_hash_order());
        assert_eq!(decoded_canon.verify_canonical_hash_order(), Ok(()));
        assert_eq!(
            decoded_canon.entries.iter().map(|e| e.hash).collect::<Vec<_>>(),
            vec![a.hash, b.hash, c.hash]
        );

        // Strict writer rejects unsorted entries.
        assert_eq!(
            write_pack_strict(&perm1, 0),
            Err(PackError::NonCanonicalOrder {
                entry_index: 1,
                previous_hash: c.hash,
                hash: a.hash,
            })
        );

        // Unsorted pack written with `write_pack` parses via `read_pack` (backwards-compatible),
        // reports non-canonical via `ShaderPack`, and is rejected by `read_pack_canonical`.
        let unsorted_bytes = write_pack(&perm1, 0).unwrap();
        let decoded_unsorted = read_pack(&unsorted_bytes).unwrap();
        assert!(!decoded_unsorted.is_canonical_hash_order());
        assert_eq!(
            decoded_unsorted.verify_canonical_hash_order(),
            Err(PackError::NonCanonicalOrder {
                entry_index: 1,
                previous_hash: c.hash,
                hash: a.hash,
            })
        );
        assert_eq!(
            read_pack_canonical(&unsorted_bytes),
            Err(PackError::NonCanonicalOrder {
                entry_index: 1,
                previous_hash: c.hash,
                hash: a.hash,
            })
        );

        // Direct `verify_canonical_hash_order` edge cases & Display formatting.
        assert_eq!(verify_canonical_hash_order([]), Ok(()));
        assert_eq!(verify_canonical_hash_order([0]), Ok(()));
        assert_eq!(verify_canonical_hash_order([0, u64::MAX]), Ok(()));
        let dup_err = verify_canonical_hash_order([5, 5]).unwrap_err();
        assert!(dup_err.to_string().contains("duplicate shader hash"));
        let order_err = verify_canonical_hash_order([10, 9]).unwrap_err();
        assert!(order_err.to_string().contains("non-canonical hash order"));
        let overlap_err = PackError::OverlappingPayload {
            first_index: 0,
            first_offset: 56,
            first_length: 10,
            second_index: 1,
            second_offset: 60,
            second_length: 8,
        };
        assert!(overlap_err.to_string().contains("overlaps"));
    }
}
