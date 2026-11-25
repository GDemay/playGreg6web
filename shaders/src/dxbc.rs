use std::collections::BTreeMap;
use std::fmt::{self, Write as _};

use crate::convert::{fnv1a_64, format_shader_hash};
use crate::pack::write_json_string;

/// DirectX shader container magic (`"DXBC"`).
pub const DXBC_MAGIC: [u8; 4] = *b"DXBC";

/// Fixed DXBC container header size in bytes:
/// magic(4) + digest(16) + version(4) + total_size(4) + chunk_count(4).
pub const DXBC_HEADER_BYTES: usize = 32;

/// Fixed per-chunk header size in bytes: fourcc(4) + payload_size(4).
pub const DXBC_CHUNK_HEADER_BYTES: usize = 8;

/// Upper bound on chunk count accepted in a single DXBC container
/// (defends against crafted containers with pathological offset tables).
pub const DXBC_MAX_CHUNKS: u32 = 256;

/// LLVM bitcode magic bytes (`BC\xC0\xDE`) found inside `DXIL` program payloads.
pub const LLVM_BITCODE_MAGIC: [u8; 4] = [0x42, 0x43, 0xC0, 0xDE];

/// Minimum `DxilProgramHeader` size in bytes (6 x 32-bit words).
const DXIL_PROGRAM_HEADER_BYTES: usize = 24;

/// Minimum `RDEF` (SM4) fixed header size in bytes before variable-length tables and strings.
const RDEF_MIN_HEADER_BYTES: usize = 28;

/// Errors produced when parsing a DXBC / DXIL shader container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DxbcError {
    TooShort {
        actual: usize,
        expected_min: usize,
    },
    InvalidMagic([u8; 4]),
    InvalidTotalSize {
        declared: u32,
        actual: usize,
    },
    Overflow(String),
    ChunkTableOutOfBounds {
        chunk_count: u32,
        table_end: usize,
        container_size: usize,
    },
    TooManyChunks {
        chunk_count: u32,
        max_chunks: u32,
    },
    ChunkOutOfBounds {
        chunk_index: usize,
        offset: u32,
        size: u32,
        container_size: usize,
    },
    InvalidChunkTag {
        chunk_index: usize,
        offset: u32,
        fourcc: [u8; 4],
    },
    DuplicateChunkTag {
        fourcc: [u8; 4],
        first_index: usize,
        duplicate_index: usize,
    },
    OverlappingChunks {
        first_index: usize,
        first_offset: u32,
        first_end: u32,
        second_index: usize,
        second_offset: u32,
    },
}

impl fmt::Display for DxbcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooShort {
                actual,
                expected_min,
            } => write!(
                f,
                "DXBC container truncated: {actual} bytes (minimum required: {expected_min})"
            ),
            Self::InvalidMagic(magic) => write!(
                f,
                "invalid DXBC magic {:?} (expected {:?})",
                format_fourcc(*magic),
                "DXBC"
            ),
            Self::InvalidTotalSize { declared, actual } => write!(
                f,
                "invalid DXBC total size {declared} bytes (input buffer has {actual} bytes)"
            ),
            Self::Overflow(msg) => write!(f, "DXBC offset/size overflow: {msg}"),
            Self::ChunkTableOutOfBounds {
                chunk_count,
                table_end,
                container_size,
            } => write!(
                f,
                "DXBC chunk offset table ({chunk_count} chunks, end={table_end}) exceeds container size {container_size}"
            ),
            Self::TooManyChunks {
                chunk_count,
                max_chunks,
            } => write!(
                f,
                "DXBC chunk count {chunk_count} exceeds safety limit ({max_chunks})"
            ),
            Self::ChunkOutOfBounds {
                chunk_index,
                offset,
                size,
                container_size,
            } => write!(
                f,
                "DXBC chunk {chunk_index} at offset {offset} (+header+{size} bytes) exceeds container bounds ({container_size} bytes)"
            ),
            Self::InvalidChunkTag {
                chunk_index,
                offset,
                fourcc,
            } => write!(
                f,
                "DXBC chunk {chunk_index} at offset {offset} has invalid FourCC tag {:?}",
                format_fourcc(*fourcc)
            ),
            Self::DuplicateChunkTag {
                fourcc,
                first_index,
                duplicate_index,
            } => write!(
                f,
                "duplicate DXBC chunk tag {:?} (chunk indices {first_index} and {duplicate_index})",
                format_fourcc(*fourcc)
            ),
            Self::OverlappingChunks {
                first_index,
                first_offset,
                first_end,
                second_index,
                second_offset,
            } => write!(
                f,
                "overlapping DXBC chunks: chunk {first_index} [{first_offset}..{first_end}) overlaps chunk {second_index} at offset {second_offset}"
            ),
        }
    }
}

impl std::error::Error for DxbcError {}

/// DirectX shader stage kind encoded in `DXIL`, `SHDR`, and `SHEX` headers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DxilShaderKind {
    Pixel,
    Vertex,
    Geometry,
    Hull,
    Domain,
    Compute,
    Library,
    RayGeneration,
    Intersection,
    AnyHit,
    ClosestHit,
    Miss,
    Callable,
    Mesh,
    Amplification,
    Node,
    Unknown(u16),
}

impl DxilShaderKind {
    /// Decode the 16-bit program type field from `DxilProgramHeader::ProgramVersion` or SM4/5 version token.
    pub const fn from_u16(raw: u16) -> Self {
        match raw {
            0 => Self::Pixel,
            1 => Self::Vertex,
            2 => Self::Geometry,
            3 => Self::Hull,
            4 => Self::Domain,
            5 => Self::Compute,
            6 => Self::Library,
            7 => Self::RayGeneration,
            8 => Self::Intersection,
            9 => Self::AnyHit,
            10 => Self::ClosestHit,
            11 => Self::Miss,
            12 => Self::Callable,
            13 => Self::Mesh,
            14 => Self::Amplification,
            15 => Self::Node,
            other => Self::Unknown(other),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pixel => "pixel",
            Self::Vertex => "vertex",
            Self::Geometry => "geometry",
            Self::Hull => "hull",
            Self::Domain => "domain",
            Self::Compute => "compute",
            Self::Library => "library",
            Self::RayGeneration => "ray_generation",
            Self::Intersection => "intersection",
            Self::AnyHit => "any_hit",
            Self::ClosestHit => "closest_hit",
            Self::Miss => "miss",
            Self::Callable => "callable",
            Self::Mesh => "mesh",
            Self::Amplification => "amplification",
            Self::Node => "node",
            Self::Unknown(_) => "unknown",
        }
    }

    pub const fn profile_prefix(self) -> &'static str {
        match self {
            Self::Pixel => "ps",
            Self::Vertex => "vs",
            Self::Geometry => "gs",
            Self::Hull => "hs",
            Self::Domain => "ds",
            Self::Compute => "cs",
            Self::Library => "lib",
            Self::RayGeneration => "rgs",
            Self::Intersection => "is",
            Self::AnyHit => "ahs",
            Self::ClosestHit => "chs",
            Self::Miss => "ms_rt",
            Self::Callable => "call",
            Self::Mesh => "ms",
            Self::Amplification => "as",
            Self::Node => "node",
            Self::Unknown(_) => "unk",
        }
    }
}

impl fmt::Display for DxilShaderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(raw) => write!(f, "unknown({raw})"),
            other => f.write_str(other.as_str()),
        }
    }
}

/// Summary extracted from a `DXIL` chunk's `DxilProgramHeader`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxilHeaderSummary {
    pub shader_kind: DxilShaderKind,
    pub shader_model_major: u8,
    pub shader_model_minor: u8,
    pub size_in_uint32: u32,
    pub dxil_magic: [u8; 4],
    pub dxil_version_major: u8,
    pub dxil_version_minor: u8,
    pub bitcode_offset: u32,
    pub bitcode_size: u32,
    pub has_llvm_bitcode_magic: bool,
}

/// Summary extracted from a legacy `SHDR` or `SHEX` (SM4/SM5) chunk header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LegacyShaderHeaderSummary {
    pub shader_kind: DxilShaderKind,
    pub shader_model_major: u8,
    pub shader_model_minor: u8,
    pub dword_count: u32,
}

/// Summary extracted from an `RDEF` (resource definition) chunk header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RdefSummary {
    pub constant_buffer_count: u32,
    pub resource_binding_count: u32,
    pub target_major: u8,
    pub target_minor: u8,
    pub program_type: u16,
    pub flags: u32,
    pub is_rd11: bool,
    pub creator: Option<String>,
}

/// Summary extracted from an `ISGN`, `OSGN`, `ISG1`, `OSG1`, `PSG1`, or `PCSG` signature chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureSummary {
    pub element_count: u32,
}

/// Summary extracted from a `STAT` shader statistics chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatSummary {
    pub instruction_count: u32,
    pub temp_register_count: u32,
    pub def_count: Option<u32>,
    pub dcl_count: Option<u32>,
    pub float_instruction_count: Option<u32>,
    pub int_instruction_count: Option<u32>,
    pub uint_instruction_count: Option<u32>,
}

/// Summary extracted from a `HASH` shader hash chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HashChunkSummary {
    pub flags: u32,
    pub digest_hex: String,
}

/// Summary extracted from an `SFI0` shader feature info chunk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FeatureInfoSummary {
    pub flags: u64,
    pub features: Vec<&'static str>,
}

/// Decoded metadata for a recognized DXBC chunk (never stores raw bytecode dumps).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DxbcChunkDetail {
    Dxil(DxilHeaderSummary),
    LegacyShader(LegacyShaderHeaderSummary),
    Rdef(RdefSummary),
    Signature(SignatureSummary),
    Stat(StatSummary),
    Hash(HashChunkSummary),
    FeatureInfo(FeatureInfoSummary),
    Other,
}

/// Summarized metadata for a single chunk inside a DXBC / DXIL container.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxbcChunkSummary {
    pub fourcc: [u8; 4],
    pub offset: u32,
    pub size: u32,
    pub detail: DxbcChunkDetail,
}

impl DxbcChunkSummary {
    /// Human-readable FourCC tag string (printable ASCII or escaped hex).
    pub fn fourcc_str(&self) -> String {
        format_fourcc(self.fourcc)
    }

    /// Concise single-line human summary for this chunk's decoded metadata.
    pub fn detail_summary(&self) -> String {
        match &self.detail {
            DxbcChunkDetail::Dxil(dxil) => format!(
                "{}_{}_{} (kind={}, dxil={}.{}, bitcode={}B, llvm_bc={})",
                dxil.shader_kind.profile_prefix(),
                dxil.shader_model_major,
                dxil.shader_model_minor,
                dxil.shader_kind,
                dxil.dxil_version_major,
                dxil.dxil_version_minor,
                dxil.bitcode_size,
                dxil.has_llvm_bitcode_magic
            ),
            DxbcChunkDetail::LegacyShader(sh) => format!(
                "{}_{}_{} (kind={}, {} dwords)",
                sh.shader_kind.profile_prefix(),
                sh.shader_model_major,
                sh.shader_model_minor,
                sh.shader_kind,
                sh.dword_count
            ),
            DxbcChunkDetail::Rdef(rdef) => {
                let mut s = format!(
                    "cbuffers={}, bindings={}, sm={}.{}{}",
                    rdef.constant_buffer_count,
                    rdef.resource_binding_count,
                    rdef.target_major,
                    rdef.target_minor,
                    if rdef.is_rd11 { ", RD11" } else { "" }
                );
                if let Some(creator) = &rdef.creator {
                    let _ = write!(s, ", creator=\"{creator}\"");
                }
                s
            }
            DxbcChunkDetail::Signature(sig) => format!("{} element(s)", sig.element_count),
            DxbcChunkDetail::Stat(stat) => {
                let mut s = format!(
                    "instructions={}, temp_regs={}",
                    stat.instruction_count, stat.temp_register_count
                );
                if let Some(dcl) = stat.dcl_count {
                    let _ = write!(s, ", dcls={dcl}");
                }
                s
            }
            DxbcChunkDetail::Hash(h) => {
                format!("flags=0x{:08x}, digest={}", h.flags, h.digest_hex)
            }
            DxbcChunkDetail::FeatureInfo(sfi) => {
                if sfi.features.is_empty() {
                    format!("flags=0x{:016x}", sfi.flags)
                } else {
                    format!("flags=0x{:016x} ({})", sfi.flags, sfi.features.join(", "))
                }
            }
            DxbcChunkDetail::Other => format!("{} bytes", self.size),
        }
    }
}

/// Zero-copy borrowed reference to a single DXBC chunk payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DxbcChunkRef<'a> {
    pub fourcc: [u8; 4],
    pub offset: u32,
    pub payload: &'a [u8],
}

/// Parsed DXBC / DXIL container metadata and chunk summaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxbcInfo {
    pub fnv1a_hash: u64,
    pub digest: [u8; 16],
    pub version_major: u16,
    pub version_minor: u16,
    pub total_size: u32,
    pub actual_bytes: usize,
    pub chunks: Vec<DxbcChunkSummary>,
}

impl DxbcInfo {
    /// Hex-encoded 16-byte DXBC container digest.
    pub fn digest_hex(&self) -> String {
        encode_hex_lower(&self.digest)
    }

    /// Primary shader kind detected from `DXIL` (preferred) or `SHDR`/`SHEX`.
    pub fn shader_kind(&self) -> Option<DxilShaderKind> {
        for chunk in &self.chunks {
            if let DxbcChunkDetail::Dxil(dxil) = &chunk.detail {
                return Some(dxil.shader_kind);
            }
        }
        for chunk in &self.chunks {
            if let DxbcChunkDetail::LegacyShader(sh) = &chunk.detail {
                return Some(sh.shader_kind);
            }
        }
        None
    }

    /// Shader model `(major, minor)` detected from `DXIL`, `SHDR`/`SHEX`, or `RDEF`.
    pub fn shader_model(&self) -> Option<(u8, u8)> {
        for chunk in &self.chunks {
            if let DxbcChunkDetail::Dxil(dxil) = &chunk.detail {
                return Some((dxil.shader_model_major, dxil.shader_model_minor));
            }
        }
        for chunk in &self.chunks {
            if let DxbcChunkDetail::LegacyShader(sh) = &chunk.detail {
                return Some((sh.shader_model_major, sh.shader_model_minor));
            }
        }
        for chunk in &self.chunks {
            if let DxbcChunkDetail::Rdef(rdef) = &chunk.detail {
                return Some((rdef.target_major, rdef.target_minor));
            }
        }
        None
    }

    /// Whether this container holds a `DXIL` program chunk.
    pub fn is_dxil(&self) -> bool {
        self.chunks
            .iter()
            .any(|c| matches!(c.detail, DxbcChunkDetail::Dxil(_)))
    }

    /// Whether a `DXIL` chunk inside this container passed the LLVM bitcode `BC\xC0\xDE` magic check.
    pub fn has_llvm_bitcode(&self) -> bool {
        self.chunks.iter().any(|c| {
            matches!(
                c.detail,
                DxbcChunkDetail::Dxil(DxilHeaderSummary {
                    has_llvm_bitcode_magic: true,
                    ..
                })
            )
        })
    }

    /// Find the first chunk matching `fourcc`.
    pub fn find_chunk(&self, fourcc: &[u8; 4]) -> Option<&DxbcChunkSummary> {
        self.chunks.iter().find(|c| &c.fourcc == fourcc)
    }
}

/// Entry representing one inspected file in batch Markdown/JSON output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DxbcFileEntry {
    pub path: String,
    pub result: Result<DxbcInfo, String>,
}

/// Return `true` if `fourcc` is a valid printable ASCII DXBC chunk FourCC tag:
/// the first byte must be printable non-space ASCII (`0x21..=0x7E`) and the
/// remaining three bytes must be printable ASCII (`0x20..=0x7E`).
pub const fn is_valid_fourcc(fourcc: [u8; 4]) -> bool {
    if !fourcc[0].is_ascii_graphic() {
        return false;
    }
    let mut i = 1;
    while i < 4 {
        let b = fourcc[i];
        if !(b.is_ascii_graphic() || b == b' ') {
            return false;
        }
        i += 1;
    }
    true
}

struct ParsedHeader {
    digest: [u8; 16],
    version_major: u16,
    version_minor: u16,
    total_size: u32,
    chunk_count: u32,
}

struct ParsedChunkEntry<'a> {
    table_index: usize,
    start: u32,
    end: u32,
    chunk: DxbcChunkRef<'a>,
}

/// Validate a DXBC container and return zero-copy chunk payload slices in
/// deterministic canonical ascending-offset order.
///
/// Fully bounds-checked and overflow-checked: guaranteed panic-free on arbitrary,
/// empty, or truncated bytes, and rejects invalid FourCC tags, duplicate FourCC
/// tags, overlapping/aliased chunk byte ranges, and excessive chunk counts.
pub fn parse_dxbc_chunks(bytes: &[u8]) -> Result<Vec<DxbcChunkRef<'_>>, DxbcError> {
    let header = parse_dxbc_header(bytes)?;
    parse_dxbc_chunks_with_header(bytes, header.total_size, header.chunk_count)
}

fn parse_dxbc_chunks_with_header(
    bytes: &[u8],
    total_size: u32,
    chunk_count: u32,
) -> Result<Vec<DxbcChunkRef<'_>>, DxbcError> {
    let container_size = usize::try_from(total_size)
        .map_err(|_| DxbcError::Overflow("DXBC total_size exceeds platform usize".to_string()))?;
    let container = bytes
        .get(..container_size)
        .ok_or(DxbcError::InvalidTotalSize {
            declared: total_size,
            actual: bytes.len(),
        })?;

    let table_bytes_u32 = chunk_count
        .checked_mul(4)
        .ok_or_else(|| DxbcError::Overflow("chunk offset table byte length overflow".to_string()))?;
    let table_end_u32 = (DXBC_HEADER_BYTES as u32)
        .checked_add(table_bytes_u32)
        .ok_or_else(|| {
            DxbcError::Overflow("header + chunk offset table size overflow".to_string())
        })?;
    let table_end = usize::try_from(table_end_u32)
        .map_err(|_| DxbcError::Overflow("chunk offset table end exceeds usize".to_string()))?;

    if table_end > container_size {
        return Err(DxbcError::ChunkTableOutOfBounds {
            chunk_count,
            table_end,
            container_size,
        });
    }

    if chunk_count > DXBC_MAX_CHUNKS {
        return Err(DxbcError::TooManyChunks {
            chunk_count,
            max_chunks: DXBC_MAX_CHUNKS,
        });
    }

    let chunk_count_usize = usize::try_from(chunk_count)
        .map_err(|_| DxbcError::Overflow("chunk count exceeds platform usize".to_string()))?;
    let mut parsed = Vec::with_capacity(chunk_count_usize);

    for idx in 0..chunk_count_usize {
        let entry_rel = idx
            .checked_mul(4)
            .ok_or_else(|| DxbcError::Overflow("chunk offset table index overflow".to_string()))?;
        let entry_pos = DXBC_HEADER_BYTES
            .checked_add(entry_rel)
            .ok_or_else(|| DxbcError::Overflow("chunk offset table entry overflow".to_string()))?;
        let offset = read_u32_le(container, entry_pos).ok_or_else(|| {
            DxbcError::Overflow("chunk offset table index out of bounds".to_string())
        })?;

        let payload_start_u32 = offset
            .checked_add(DXBC_CHUNK_HEADER_BYTES as u32)
            .ok_or_else(|| DxbcError::Overflow(format!("chunk {idx} header offset overflow")))?;
        let offset_usize = usize::try_from(offset)
            .map_err(|_| DxbcError::Overflow(format!("chunk {idx} offset exceeds usize")))?;
        let payload_start = usize::try_from(payload_start_u32).map_err(|_| {
            DxbcError::Overflow(format!("chunk {idx} payload start exceeds usize"))
        })?;

        if offset_usize < table_end || payload_start > container_size {
            return Err(DxbcError::ChunkOutOfBounds {
                chunk_index: idx,
                offset,
                size: 0,
                container_size,
            });
        }

        let fourcc_end = offset_usize
            .checked_add(4)
            .ok_or_else(|| DxbcError::Overflow(format!("chunk {idx} fourcc offset overflow")))?;
        let fourcc_slice =
            container
                .get(offset_usize..fourcc_end)
                .ok_or(DxbcError::ChunkOutOfBounds {
                    chunk_index: idx,
                    offset,
                    size: 0,
                    container_size,
                })?;
        let fourcc: [u8; 4] = [
            fourcc_slice[0],
            fourcc_slice[1],
            fourcc_slice[2],
            fourcc_slice[3],
        ];
        if !is_valid_fourcc(fourcc) {
            return Err(DxbcError::InvalidChunkTag {
                chunk_index: idx,
                offset,
                fourcc,
            });
        }

        let size = read_u32_le(container, fourcc_end)
            .ok_or_else(|| DxbcError::Overflow(format!("chunk {idx} size read out of bounds")))?;

        let payload_end_u32 = payload_start_u32
            .checked_add(size)
            .ok_or_else(|| DxbcError::Overflow(format!("chunk {idx} payload length overflow")))?;
        let payload_end = usize::try_from(payload_end_u32)
            .map_err(|_| DxbcError::Overflow(format!("chunk {idx} payload end exceeds usize")))?;

        if payload_end > container_size {
            return Err(DxbcError::ChunkOutOfBounds {
                chunk_index: idx,
                offset,
                size,
                container_size,
            });
        }

        let payload =
            container
                .get(payload_start..payload_end)
                .ok_or(DxbcError::ChunkOutOfBounds {
                    chunk_index: idx,
                    offset,
                    size,
                    container_size,
                })?;

        parsed.push(ParsedChunkEntry {
            table_index: idx,
            start: offset,
            end: payload_end_u32,
            chunk: DxbcChunkRef {
                fourcc,
                offset,
                payload,
            },
        });
    }

    // Canonicalize enumeration order by physical container offset, then verify
    // non-overlapping byte ranges and unique FourCC tags deterministically.
    parsed.sort_by_key(|entry| (entry.start, entry.end, entry.table_index));

    for pair in parsed.windows(2) {
        let prev = &pair[0];
        let next = &pair[1];
        if next.start < prev.end {
            return Err(DxbcError::OverlappingChunks {
                first_index: prev.table_index,
                first_offset: prev.start,
                first_end: prev.end,
                second_index: next.table_index,
                second_offset: next.start,
            });
        }
    }

    let mut seen_tags: BTreeMap<[u8; 4], usize> = BTreeMap::new();
    for entry in &parsed {
        if let Some(&prev_idx) = seen_tags.get(&entry.chunk.fourcc) {
            let (first_index, duplicate_index) = if prev_idx <= entry.table_index {
                (prev_idx, entry.table_index)
            } else {
                (entry.table_index, prev_idx)
            };
            return Err(DxbcError::DuplicateChunkTag {
                fourcc: entry.chunk.fourcc,
                first_index,
                duplicate_index,
            });
        }
        seen_tags.insert(entry.chunk.fourcc, entry.table_index);
    }

    Ok(parsed.into_iter().map(|entry| entry.chunk).collect())
}

/// Parse a DXBC / DXIL shader container binary into a structured [`DxbcInfo`] summary.
///
/// Computes the deterministic 64-bit FNV-1a hash over `bytes` and summarizes recognized chunks
/// (`DXIL`, `SHDR`, `SHEX`, `RDEF`, `ISGN`/`OSGN`/`ISG1`/`OSG1`/`PSG1`/`PCSG`, `STAT`, `HASH`,
/// `SFI0`) in deterministic canonical ascending-offset order without retaining raw shader bytecode.
pub fn parse_dxbc(bytes: &[u8]) -> Result<DxbcInfo, DxbcError> {
    let header = parse_dxbc_header(bytes)?;
    let chunk_refs = parse_dxbc_chunks_with_header(bytes, header.total_size, header.chunk_count)?;

    let mut chunks = Vec::with_capacity(chunk_refs.len());
    for chunk in chunk_refs {
        let size = u32::try_from(chunk.payload.len())
            .map_err(|_| DxbcError::Overflow("chunk payload exceeds u32::MAX".to_string()))?;
        let detail = summarize_chunk(chunk.fourcc, chunk.payload);
        chunks.push(DxbcChunkSummary {
            fourcc: chunk.fourcc,
            offset: chunk.offset,
            size,
            detail,
        });
    }

    Ok(DxbcInfo {
        fnv1a_hash: fnv1a_64(bytes),
        digest: header.digest,
        version_major: header.version_major,
        version_minor: header.version_minor,
        total_size: header.total_size,
        actual_bytes: bytes.len(),
        chunks,
    })
}

fn parse_dxbc_header(bytes: &[u8]) -> Result<ParsedHeader, DxbcError> {
    if bytes.len() < DXBC_HEADER_BYTES {
        return Err(DxbcError::TooShort {
            actual: bytes.len(),
            expected_min: DXBC_HEADER_BYTES,
        });
    }

    let magic: [u8; 4] = [bytes[0], bytes[1], bytes[2], bytes[3]];
    if magic != DXBC_MAGIC {
        return Err(DxbcError::InvalidMagic(magic));
    }

    let mut digest = [0u8; 16];
    digest.copy_from_slice(&bytes[4..20]);

    let version_major = u16::from_le_bytes([bytes[20], bytes[21]]);
    let version_minor = u16::from_le_bytes([bytes[22], bytes[23]]);
    let total_size = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
    let chunk_count = u32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);

    let container_size = usize::try_from(total_size)
        .map_err(|_| DxbcError::Overflow("DXBC total_size exceeds platform usize".to_string()))?;
    if container_size < DXBC_HEADER_BYTES || container_size > bytes.len() {
        return Err(DxbcError::InvalidTotalSize {
            declared: total_size,
            actual: bytes.len(),
        });
    }

    Ok(ParsedHeader {
        digest,
        version_major,
        version_minor,
        total_size,
        chunk_count,
    })
}

fn summarize_chunk(fourcc: [u8; 4], payload: &[u8]) -> DxbcChunkDetail {
    match &fourcc {
        b"DXIL" => parse_dxil_chunk(payload)
            .map(DxbcChunkDetail::Dxil)
            .unwrap_or(DxbcChunkDetail::Other),
        b"SHDR" | b"SHEX" => parse_legacy_shader_chunk(payload)
            .map(DxbcChunkDetail::LegacyShader)
            .unwrap_or(DxbcChunkDetail::Other),
        b"RDEF" => parse_rdef_chunk(payload)
            .map(DxbcChunkDetail::Rdef)
            .unwrap_or(DxbcChunkDetail::Other),
        b"ISGN" | b"OSGN" | b"ISG1" | b"OSG1" | b"PSG1" | b"PCSG" => parse_signature_chunk(payload)
            .map(DxbcChunkDetail::Signature)
            .unwrap_or(DxbcChunkDetail::Other),
        b"STAT" => parse_stat_chunk(payload)
            .map(DxbcChunkDetail::Stat)
            .unwrap_or(DxbcChunkDetail::Other),
        b"HASH" => parse_hash_chunk(payload)
            .map(DxbcChunkDetail::Hash)
            .unwrap_or(DxbcChunkDetail::Other),
        b"SFI0" => parse_sfi0_chunk(payload)
            .map(DxbcChunkDetail::FeatureInfo)
            .unwrap_or(DxbcChunkDetail::Other),
        _ => DxbcChunkDetail::Other,
    }
}

fn parse_dxil_chunk(payload: &[u8]) -> Option<DxilHeaderSummary> {
    if payload.len() < DXIL_PROGRAM_HEADER_BYTES {
        return None;
    }

    let program_version = read_u32_le(payload, 0)?;
    let size_in_uint32 = read_u32_le(payload, 4)?;
    let dxil_magic_slice = payload.get(8..12)?;
    let dxil_magic = [
        dxil_magic_slice[0],
        dxil_magic_slice[1],
        dxil_magic_slice[2],
        dxil_magic_slice[3],
    ];
    let dxil_version = read_u32_le(payload, 12)?;
    let bitcode_offset = read_u32_le(payload, 16)?;
    let bitcode_size = read_u32_le(payload, 20)?;

    let shader_kind = DxilShaderKind::from_u16((program_version >> 16) as u16);
    let shader_model_major = ((program_version >> 4) & 0xF) as u8;
    let shader_model_minor = (program_version & 0xF) as u8;
    let dxil_version_major = ((dxil_version >> 8) & 0xFF) as u8;
    let dxil_version_minor = (dxil_version & 0xFF) as u8;

    // DxilBitcodeHeader starts at byte 8 of the DXIL chunk; BitcodeOffset is relative to it
    // and must point at or past the end of the 24-byte DxilProgramHeader (bitcode_offset >= 16).
    let has_llvm_bitcode_magic = 8u32
        .checked_add(bitcode_offset)
        .and_then(|start_u32| {
            let end_bc_u32 = start_u32.checked_add(bitcode_size)?;
            let start = usize::try_from(start_u32).ok()?;
            let end_bc = usize::try_from(end_bc_u32).ok()?;
            let end_magic = start.checked_add(4)?;
            if start >= DXIL_PROGRAM_HEADER_BYTES
                && bitcode_size >= 4
                && end_magic <= payload.len()
                && end_bc <= payload.len()
            {
                Some(payload.get(start..end_magic) == Some(LLVM_BITCODE_MAGIC.as_slice()))
            } else {
                Some(false)
            }
        })
        .unwrap_or(false);

    Some(DxilHeaderSummary {
        shader_kind,
        shader_model_major,
        shader_model_minor,
        size_in_uint32,
        dxil_magic,
        dxil_version_major,
        dxil_version_minor,
        bitcode_offset,
        bitcode_size,
        has_llvm_bitcode_magic,
    })
}

fn parse_legacy_shader_chunk(payload: &[u8]) -> Option<LegacyShaderHeaderSummary> {
    if payload.len() < 8 {
        return None;
    }
    let version_token = read_u32_le(payload, 0)?;
    let dword_count = read_u32_le(payload, 4)?;
    let shader_kind = DxilShaderKind::from_u16((version_token >> 16) as u16);
    let shader_model_major = ((version_token >> 4) & 0xF) as u8;
    let shader_model_minor = (version_token & 0xF) as u8;

    Some(LegacyShaderHeaderSummary {
        shader_kind,
        shader_model_major,
        shader_model_minor,
        dword_count,
    })
}

fn parse_rdef_chunk(payload: &[u8]) -> Option<RdefSummary> {
    if payload.len() < RDEF_MIN_HEADER_BYTES {
        return None;
    }
    let constant_buffer_count = read_u32_le(payload, 0)?;
    let resource_binding_count = read_u32_le(payload, 8)?;
    let target_minor = *payload.get(16)?;
    let target_major = *payload.get(17)?;
    let program_type = u16::from_le_bytes([*payload.get(18)?, *payload.get(19)?]);
    let flags = read_u32_le(payload, 20)?;
    let creator_offset_u32 = read_u32_le(payload, 24)?;

    let is_rd11 = payload.get(28..32) == Some(b"RD11".as_slice());
    let creator = usize::try_from(creator_offset_u32)
        .ok()
        .filter(|&off| off >= RDEF_MIN_HEADER_BYTES)
        .and_then(|off| read_c_string(payload, off, 256));

    Some(RdefSummary {
        constant_buffer_count,
        resource_binding_count,
        target_major,
        target_minor,
        program_type,
        flags,
        is_rd11,
        creator,
    })
}

fn parse_signature_chunk(payload: &[u8]) -> Option<SignatureSummary> {
    if payload.len() < 8 {
        return None;
    }
    let element_count = read_u32_le(payload, 0)?;
    Some(SignatureSummary { element_count })
}

fn parse_stat_chunk(payload: &[u8]) -> Option<StatSummary> {
    if payload.len() < 8 {
        return None;
    }
    let instruction_count = read_u32_le(payload, 0)?;
    let temp_register_count = read_u32_le(payload, 4)?;
    let def_count = read_u32_le(payload, 8);
    let dcl_count = read_u32_le(payload, 12);
    let float_instruction_count = read_u32_le(payload, 16);
    let int_instruction_count = read_u32_le(payload, 20);
    let uint_instruction_count = read_u32_le(payload, 24);

    Some(StatSummary {
        instruction_count,
        temp_register_count,
        def_count,
        dcl_count,
        float_instruction_count,
        int_instruction_count,
        uint_instruction_count,
    })
}

fn parse_hash_chunk(payload: &[u8]) -> Option<HashChunkSummary> {
    if payload.len() < 20 {
        return None;
    }
    let flags = read_u32_le(payload, 0)?;
    let digest_hex = encode_hex_lower(payload.get(4..20)?);
    Some(HashChunkSummary { flags, digest_hex })
}

fn parse_sfi0_chunk(payload: &[u8]) -> Option<FeatureInfoSummary> {
    if payload.len() < 8 {
        return None;
    }
    let lo = read_u32_le(payload, 0)? as u64;
    let hi = read_u32_le(payload, 4)? as u64;
    let flags = lo | (hi << 32);

    const FEATURE_BITS: &[(u64, &str)] = &[
        (1 << 0, "Doubles"),
        (1 << 1, "ComputeShadersPlusRawAndStructuredBuffers"),
        (1 << 2, "UavsAtEveryStage"),
        (1 << 3, "64Uavs"),
        (1 << 4, "MinimumPrecision"),
        (1 << 5, "11_1_DoubleExtensions"),
        (1 << 6, "11_1_ShaderExtensions"),
        (1 << 8, "TiledResources"),
        (1 << 9, "StencilRef"),
        (1 << 10, "InnerCoverage"),
        (1 << 11, "TypedUavLoadAdditionalFormats"),
        (1 << 12, "Rovs"),
        (1 << 14, "WaveOps"),
        (1 << 15, "Int64Ops"),
        (1 << 16, "ViewId"),
        (1 << 17, "Barycentrics"),
        (1 << 18, "Native16BitOps"),
        (1 << 19, "ShadingRate"),
        (1 << 20, "RaytracingTier1_1"),
        (1 << 21, "SamplerFeedback"),
        (1 << 22, "AtomicInt64OnTypedResource"),
        (1 << 23, "AtomicInt64OnGroupShared"),
        (1 << 24, "DerivativesInMeshAndAmpShaders"),
        (1 << 25, "ResourceDescriptorHeapIndexing"),
        (1 << 26, "SamplerDescriptorHeapIndexing"),
        (1 << 27, "AtomicInt64OnHeapResource"),
        (1 << 28, "AdvancedTextureOps"),
        (1 << 29, "WriteableMsaaTextures"),
    ];

    let mut features = Vec::new();
    for &(mask, name) in FEATURE_BITS {
        if (flags & mask) != 0 {
            features.push(name);
        }
    }

    Some(FeatureInfoSummary { flags, features })
}

fn read_u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let slice = bytes.get(offset..end)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_c_string(bytes: &[u8], offset: usize, max_len: usize) -> Option<String> {
    if offset == 0 || max_len == 0 {
        return None;
    }
    let tail = bytes.get(offset..)?;
    let limit = tail.len().min(max_len);
    let window = tail.get(..limit)?;
    let nul_pos = window.iter().position(|&b| b == 0)?;
    if nul_pos == 0 {
        return None;
    }
    let text = std::str::from_utf8(window.get(..nul_pos)?).ok()?;
    if text.chars().all(|c| !c.is_control()) {
        Some(text.to_string())
    } else {
        None
    }
}

fn encode_hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Format a 4-byte FourCC code cleanly as ASCII (escaping non-printable bytes).
pub fn format_fourcc(fourcc: [u8; 4]) -> String {
    if fourcc.iter().all(|b| b.is_ascii_graphic() || *b == b' ') {
        String::from_utf8_lossy(&fourcc).into_owned()
    } else {
        format!(
            "0x{:02x}{:02x}{:02x}{:02x}",
            fourcc[0], fourcc[1], fourcc[2], fourcc[3]
        )
    }
}

fn escape_md_cell(text: &str) -> String {
    text.replace('|', "\\|").replace(['\r', '\n'], " ")
}

/// Render a single parsed [`DxbcInfo`] as a clean Markdown summary (no raw bytecode).
pub fn render_dxbc_info_md(path: Option<&str>, info: &DxbcInfo) -> String {
    let mut out = String::new();
    match path {
        Some(p) => {
            let _ = writeln!(out, "# DXBC / DXIL Container Summary: `{}`\n", escape_md_cell(p));
        }
        None => {
            out.push_str("# DXBC / DXIL Container Summary\n\n");
        }
    }

    out.push_str("| Property | Value |\n");
    out.push_str("| :--- | :--- |\n");
    let _ = writeln!(out, "| FNV-1a Hash | `{}` |", format_shader_hash(info.fnv1a_hash));
    let _ = writeln!(out, "| Container Digest | `{}` |", info.digest_hex());
    let _ = writeln!(
        out,
        "| Version | `{}.{}` |",
        info.version_major, info.version_minor
    );
    let _ = writeln!(
        out,
        "| Size | `{} bytes` (input `{}` bytes) |",
        info.total_size, info.actual_bytes
    );
    let _ = writeln!(
        out,
        "| Format | `{}` |",
        if info.is_dxil() { "DXIL" } else { "DXBC" }
    );
    if let Some(kind) = info.shader_kind() {
        let _ = writeln!(out, "| Shader Kind | `{kind}` |");
    }
    if let Some((maj, min)) = info.shader_model() {
        let _ = writeln!(out, "| Shader Model | `{maj}.{min}` |");
    }
    if info.is_dxil() {
        let _ = writeln!(
            out,
            "| LLVM Bitcode Magic (`BC\\xC0\\xDE`) | `{}` |",
            info.has_llvm_bitcode()
        );
    }
    let _ = writeln!(out, "| Chunks | `{}` |", info.chunks.len());
    out.push('\n');

    out.push_str("## Chunks\n\n");
    if info.chunks.is_empty() {
        out.push_str("No chunks present.\n");
    } else {
        out.push_str("| Index | FourCC | Offset | Size | Details |\n");
        out.push_str("| ---: | :--- | ---: | ---: | :--- |\n");
        for (idx, chunk) in info.chunks.iter().enumerate() {
            let _ = writeln!(
                out,
                "| {idx} | `{}` | {} | {} | {} |",
                escape_md_cell(&chunk.fourcc_str()),
                chunk.offset,
                chunk.size,
                escape_md_cell(&chunk.detail_summary())
            );
        }
    }

    out
}

/// Render a batch of inspected DXBC/DXIL files as a clean Markdown summary table.
pub fn render_dxbc_batch_md(entries: &[DxbcFileEntry]) -> String {
    let ok_count = entries.iter().filter(|e| e.result.is_ok()).count();
    let err_count = entries.len() - ok_count;
    let dxil_count = entries
        .iter()
        .filter(|e| e.result.as_ref().is_ok_and(DxbcInfo::is_dxil))
        .count();

    let mut out = String::new();
    out.push_str("# DXBC / DXIL Container Inspection Report\n\n");
    out.push_str("| Metric | Count |\n");
    out.push_str("| :--- | ---: |\n");
    let _ = writeln!(out, "| Total containers | {} |", entries.len());
    let _ = writeln!(out, "| Parsed | {ok_count} |");
    let _ = writeln!(out, "| DXIL containers | {dxil_count} |");
    let _ = writeln!(out, "| Errors | {err_count} |");
    out.push('\n');

    if entries.is_empty() {
        out.push_str("No `.dxbc`, `.dxil`, or `.cso` files found.\n");
        return out;
    }

    out.push_str("| File | Hash | Kind | SM | Format | Chunks | Status |\n");
    out.push_str("| :--- | :--- | :--- | :--- | :--- | ---: | :--- |\n");
    for entry in entries {
        match &entry.result {
            Ok(info) => {
                let kind = info
                    .shader_kind()
                    .map(|k| k.to_string())
                    .unwrap_or_else(|| "-".to_string());
                let sm = info
                    .shader_model()
                    .map(|(maj, min)| format!("{maj}.{min}"))
                    .unwrap_or_else(|| "-".to_string());
                let fmt = if info.is_dxil() {
                    if info.has_llvm_bitcode() {
                        "DXIL (BC)"
                    } else {
                        "DXIL"
                    }
                } else {
                    "DXBC"
                };
                let _ = writeln!(
                    out,
                    "| `{}` | `{}` | `{}` | `{}` | `{}` | {} | ok |",
                    escape_md_cell(&entry.path),
                    format_shader_hash(info.fnv1a_hash),
                    escape_md_cell(&kind),
                    sm,
                    fmt,
                    info.chunks.len()
                );
            }
            Err(err) => {
                let _ = writeln!(
                    out,
                    "| `{}` | - | - | - | - | 0 | error: {} |",
                    escape_md_cell(&entry.path),
                    escape_md_cell(err)
                );
            }
        }
    }

    out
}

/// Render a single parsed [`DxbcInfo`] as deterministic pretty-printed JSON (no raw bytecode).
pub fn render_dxbc_info_json(path: Option<&str>, info: &DxbcInfo) -> String {
    let mut out = String::new();
    write_dxbc_info_json_obj(&mut out, path, info, 0);
    out.push('\n');
    out
}

/// Render a batch of inspected DXBC/DXIL files as deterministic pretty-printed JSON.
pub fn render_dxbc_batch_json(entries: &[DxbcFileEntry]) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    let _ = writeln!(out, "  \"total\": {},", entries.len());
    out.push_str("  \"files\": [\n");
    for (idx, entry) in entries.iter().enumerate() {
        match &entry.result {
            Ok(info) => {
                write_dxbc_info_json_obj(&mut out, Some(&entry.path), info, 4);
            }
            Err(err) => {
                out.push_str("    {\n");
                out.push_str("      \"path\": ");
                write_json_string(&mut out, &entry.path);
                out.push_str(",\n");
                out.push_str("      \"error\": ");
                write_json_string(&mut out, err);
                out.push_str("\n    }");
            }
        }
        if idx + 1 < entries.len() {
            out.push_str(",\n");
        } else {
            out.push('\n');
        }
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

fn write_dxbc_info_json_obj(out: &mut String, path: Option<&str>, info: &DxbcInfo, indent: usize) {
    let pad = " ".repeat(indent);
    let pad2 = " ".repeat(indent + 2);
    let pad4 = " ".repeat(indent + 4);
    let pad6 = " ".repeat(indent + 6);

    let _ = writeln!(out, "{pad}{{");
    if let Some(p) = path {
        out.push_str(&pad2);
        out.push_str("\"path\": ");
        write_json_string(out, p);
        out.push_str(",\n");
    }
    out.push_str(&pad2);
    out.push_str("\"fnv1a_hash\": ");
    write_json_string(out, &format_shader_hash(info.fnv1a_hash));
    out.push_str(",\n");

    out.push_str(&pad2);
    out.push_str("\"digest\": ");
    write_json_string(out, &info.digest_hex());
    out.push_str(",\n");

    let _ = writeln!(out, "{pad2}\"version_major\": {},", info.version_major);
    let _ = writeln!(out, "{pad2}\"version_minor\": {},", info.version_minor);
    let _ = writeln!(out, "{pad2}\"total_size\": {},", info.total_size);
    let _ = writeln!(out, "{pad2}\"actual_bytes\": {},", info.actual_bytes);
    let _ = writeln!(out, "{pad2}\"is_dxil\": {},", info.is_dxil());
    let _ = writeln!(
        out,
        "{pad2}\"has_llvm_bitcode\": {},",
        info.has_llvm_bitcode()
    );

    out.push_str(&pad2);
    out.push_str("\"shader_kind\": ");
    match info.shader_kind() {
        Some(kind) => write_json_string(out, &kind.to_string()),
        None => out.push_str("null"),
    }
    out.push_str(",\n");

    out.push_str(&pad2);
    out.push_str("\"shader_model\": ");
    match info.shader_model() {
        Some((maj, min)) => write_json_string(out, &format!("{maj}.{min}")),
        None => out.push_str("null"),
    }
    out.push_str(",\n");

    if info.chunks.is_empty() {
        let _ = writeln!(out, "{pad2}\"chunks\": []");
    } else {
        let _ = writeln!(out, "{pad2}\"chunks\": [");
        for (i, chunk) in info.chunks.iter().enumerate() {
            let _ = writeln!(out, "{pad4}{{");
            out.push_str(&pad6);
            out.push_str("\"fourcc\": ");
            write_json_string(out, &chunk.fourcc_str());
            out.push_str(",\n");
            let _ = writeln!(out, "{pad6}\"offset\": {},", chunk.offset);
            let _ = writeln!(out, "{pad6}\"size\": {},", chunk.size);
            out.push_str(&pad6);
            out.push_str("\"detail\": ");
            write_chunk_detail_json(out, &chunk.detail);
            out.push('\n');
            if i + 1 < info.chunks.len() {
                let _ = writeln!(out, "{pad4}}},");
            } else {
                let _ = writeln!(out, "{pad4}}}");
            }
        }
        let _ = writeln!(out, "{pad2}]");
    }

    let _ = write!(out, "{pad}}}");
}

fn write_chunk_detail_json(out: &mut String, detail: &DxbcChunkDetail) {
    match detail {
        DxbcChunkDetail::Dxil(dxil) => {
            out.push_str("{\"kind\": \"dxil\", \"shader_kind\": ");
            write_json_string(out, &dxil.shader_kind.to_string());
            let _ = write!(
                out,
                ", \"shader_model_major\": {}, \"shader_model_minor\": {}, \"size_in_uint32\": {}, \"dxil_version\": \"{}.{}\", \"bitcode_offset\": {}, \"bitcode_size\": {}, \"has_llvm_bitcode_magic\": {}}}",
                dxil.shader_model_major,
                dxil.shader_model_minor,
                dxil.size_in_uint32,
                dxil.dxil_version_major,
                dxil.dxil_version_minor,
                dxil.bitcode_offset,
                dxil.bitcode_size,
                dxil.has_llvm_bitcode_magic
            );
        }
        DxbcChunkDetail::LegacyShader(sh) => {
            out.push_str("{\"kind\": \"legacy_shader\", \"shader_kind\": ");
            write_json_string(out, &sh.shader_kind.to_string());
            let _ = write!(
                out,
                ", \"shader_model_major\": {}, \"shader_model_minor\": {}, \"dword_count\": {}}}",
                sh.shader_model_major, sh.shader_model_minor, sh.dword_count
            );
        }
        DxbcChunkDetail::Rdef(rdef) => {
            let _ = write!(
                out,
                "{{\"kind\": \"rdef\", \"constant_buffer_count\": {}, \"resource_binding_count\": {}, \"target_major\": {}, \"target_minor\": {}, \"program_type\": {}, \"flags\": {}, \"is_rd11\": {}, \"creator\": ",
                rdef.constant_buffer_count,
                rdef.resource_binding_count,
                rdef.target_major,
                rdef.target_minor,
                rdef.program_type,
                rdef.flags,
                rdef.is_rd11
            );
            match &rdef.creator {
                Some(c) => write_json_string(out, c),
                None => out.push_str("null"),
            }
            out.push('}');
        }
        DxbcChunkDetail::Signature(sig) => {
            let _ = write!(
                out,
                "{{\"kind\": \"signature\", \"element_count\": {}}}",
                sig.element_count
            );
        }
        DxbcChunkDetail::Stat(stat) => {
            let _ = write!(
                out,
                "{{\"kind\": \"stat\", \"instruction_count\": {}, \"temp_register_count\": {}}}",
                stat.instruction_count, stat.temp_register_count
            );
        }
        DxbcChunkDetail::Hash(h) => {
            let _ = write!(out, "{{\"kind\": \"hash\", \"flags\": {}, \"digest\": ", h.flags);
            write_json_string(out, &h.digest_hex);
            out.push('}');
        }
        DxbcChunkDetail::FeatureInfo(sfi) => {
            let _ = write!(
                out,
                "{{\"kind\": \"feature_info\", \"flags\": {}, \"features\": [",
                sfi.flags
            );
            for (i, feat) in sfi.features.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_json_string(out, feat);
            }
            out.push_str("]}");
        }
        DxbcChunkDetail::Other => {
            out.push_str("{\"kind\": \"other\"}");
        }
    }
}

#[cfg(test)]
pub(crate) mod test_dxbc {
    use super::*;

    /// Build a synthetic valid DXBC container from `(fourcc, payload)` pairs.
    pub(crate) fn build_dxbc_container(chunks: &[([u8; 4], &[u8])]) -> Vec<u8> {
        let header_and_table = DXBC_HEADER_BYTES + chunks.len() * 4;
        let mut offsets = Vec::with_capacity(chunks.len());
        let mut cursor = header_and_table;
        for (_, payload) in chunks {
            offsets.push(cursor as u32);
            cursor += DXBC_CHUNK_HEADER_BYTES + payload.len();
        }
        let total_size = cursor as u32;

        let mut out = Vec::with_capacity(cursor);
        out.extend_from_slice(&DXBC_MAGIC);
        // Deterministic synthetic 16-byte digest
        out.extend_from_slice(&[
            0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54,
            0x32, 0x10,
        ]);
        out.extend_from_slice(&1u16.to_le_bytes()); // major
        out.extend_from_slice(&0u16.to_le_bytes()); // minor
        out.extend_from_slice(&total_size.to_le_bytes());
        out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
        for off in offsets {
            out.extend_from_slice(&off.to_le_bytes());
        }
        for (fourcc, payload) in chunks {
            out.extend_from_slice(fourcc);
            out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            out.extend_from_slice(payload);
        }
        out
    }

    /// Build a synthetic `DXIL` chunk payload containing a valid `DxilProgramHeader` and `BC\xC0\xDE` magic.
    pub(crate) fn build_dxil_chunk_payload(
        kind: u16,
        sm_major: u8,
        sm_minor: u8,
        dxil_major: u8,
        dxil_minor: u8,
        include_llvm_magic: bool,
    ) -> Vec<u8> {
        let bitcode: &[u8] = if include_llvm_magic {
            &[0x42, 0x43, 0xC0, 0xDE, 0x11, 0x22, 0x33, 0x44]
        } else {
            &[0x00, 0x00, 0x00, 0x00, 0x11, 0x22, 0x33, 0x44]
        };
        let total_bytes = DXIL_PROGRAM_HEADER_BYTES + bitcode.len();
        let size_in_uint32 = (total_bytes / 4) as u32;
        let program_version = (u32::from(kind) << 16)
            | (u32::from(sm_major & 0xF) << 4)
            | u32::from(sm_minor & 0xF);
        let dxil_version = (u32::from(dxil_major) << 8) | u32::from(dxil_minor);
        let bitcode_offset = 16u32; // relative to byte 8 -> byte 24

        let mut payload = Vec::with_capacity(total_bytes);
        payload.extend_from_slice(&program_version.to_le_bytes());
        payload.extend_from_slice(&size_in_uint32.to_le_bytes());
        payload.extend_from_slice(b"DXIL");
        payload.extend_from_slice(&dxil_version.to_le_bytes());
        payload.extend_from_slice(&bitcode_offset.to_le_bytes());
        payload.extend_from_slice(&(bitcode.len() as u32).to_le_bytes());
        payload.extend_from_slice(bitcode);
        payload
    }

    /// Build a synthetic `SHDR`/`SHEX` payload.
    pub(crate) fn build_shdr_chunk_payload(kind: u16, sm_major: u8, sm_minor: u8, dwords: u32) -> Vec<u8> {
        let token = (u32::from(kind) << 16)
            | (u32::from(sm_major & 0xF) << 4)
            | u32::from(sm_minor & 0xF);
        let mut payload = Vec::new();
        payload.extend_from_slice(&token.to_le_bytes());
        payload.extend_from_slice(&dwords.to_le_bytes());
        payload
    }

    /// Build a synthetic `ISGN`/`OSGN`/`PSG1` signature payload.
    pub(crate) fn build_signature_chunk_payload(element_count: u32) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&element_count.to_le_bytes());
        payload.extend_from_slice(&8u32.to_le_bytes());
        payload
    }

    /// Build a synthetic `STAT` payload.
    pub(crate) fn build_stat_chunk_payload(instructions: u32, temp_regs: u32, dcls: u32) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&instructions.to_le_bytes());
        payload.extend_from_slice(&temp_regs.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes()); // def_count
        payload.extend_from_slice(&dcls.to_le_bytes());
        payload
    }

    /// Build a synthetic `RDEF` payload with optional RD11 marker and creator string.
    pub(crate) fn build_rdef_chunk_payload(
        cbuffers: u32,
        bindings: u32,
        sm_major: u8,
        sm_minor: u8,
        creator: &str,
    ) -> Vec<u8> {
        let creator_offset = 32u32;
        let mut payload = Vec::new();
        payload.extend_from_slice(&cbuffers.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&bindings.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.push(sm_minor);
        payload.push(sm_major);
        payload.extend_from_slice(&0x4353u16.to_le_bytes()); // Compute
        payload.extend_from_slice(&0u32.to_le_bytes()); // flags
        payload.extend_from_slice(&creator_offset.to_le_bytes());
        payload.extend_from_slice(b"RD11");
        payload.extend_from_slice(creator.as_bytes());
        payload.push(0);
        payload
    }
}

#[cfg(test)]
mod tests {
    use super::test_dxbc::*;
    use super::*;

    #[test]
    fn parses_synthetic_dxbc_with_dxil_shdr_rdef_signatures_stat_hash_and_sfi0() {
        let dxil = build_dxil_chunk_payload(5, 6, 6, 1, 6, true); // cs_6_6
        let shdr = build_shdr_chunk_payload(1, 5, 0, 42); // vs_5_0
        let rdef = build_rdef_chunk_payload(2, 5, 6, 6, "dxc 1.8");
        let isgn = build_signature_chunk_payload(3);
        let osgn = build_signature_chunk_payload(2);
        let stat = build_stat_chunk_payload(128, 12, 7);

        let mut hash_payload = Vec::new();
        hash_payload.extend_from_slice(&1u32.to_le_bytes());
        hash_payload.extend_from_slice(&[0xaa; 16]);

        let sfi0_flags: u64 = (1 << 0) | (1 << 14) | (1 << 15); // Doubles | WaveOps | Int64Ops
        let sfi0_payload = sfi0_flags.to_le_bytes();

        let container = build_dxbc_container(&[
            (*b"RDEF", &rdef),
            (*b"ISGN", &isgn),
            (*b"OSGN", &osgn),
            (*b"STAT", &stat),
            (*b"SHDR", &shdr),
            (*b"DXIL", &dxil),
            (*b"HASH", &hash_payload),
            (*b"SFI0", &sfi0_payload),
        ]);

        let info = parse_dxbc(&container).expect("valid synthetic DXBC container should parse");
        assert_eq!(info.fnv1a_hash, fnv1a_64(&container));
        assert_eq!(info.version_major, 1);
        assert_eq!(info.version_minor, 0);
        assert_eq!(info.total_size as usize, container.len());
        assert_eq!(info.chunks.len(), 8);
        assert!(info.is_dxil());
        assert!(info.has_llvm_bitcode());
        assert_eq!(info.shader_kind(), Some(DxilShaderKind::Compute));
        assert_eq!(info.shader_model(), Some((6, 6)));

        match &info.find_chunk(b"DXIL").unwrap().detail {
            DxbcChunkDetail::Dxil(d) => {
                assert_eq!(d.shader_kind, DxilShaderKind::Compute);
                assert_eq!((d.shader_model_major, d.shader_model_minor), (6, 6));
                assert_eq!((d.dxil_version_major, d.dxil_version_minor), (1, 6));
                assert!(d.has_llvm_bitcode_magic);
                assert_eq!(d.bitcode_size, 8);
            }
            other => panic!("expected Dxil detail, got {other:?}"),
        }

        match &info.find_chunk(b"SHDR").unwrap().detail {
            DxbcChunkDetail::LegacyShader(s) => {
                assert_eq!(s.shader_kind, DxilShaderKind::Vertex);
                assert_eq!((s.shader_model_major, s.shader_model_minor), (5, 0));
                assert_eq!(s.dword_count, 42);
            }
            other => panic!("expected LegacyShader detail, got {other:?}"),
        }

        match &info.find_chunk(b"RDEF").unwrap().detail {
            DxbcChunkDetail::Rdef(r) => {
                assert_eq!(r.constant_buffer_count, 2);
                assert_eq!(r.resource_binding_count, 5);
                assert!(r.is_rd11);
                assert_eq!(r.creator.as_deref(), Some("dxc 1.8"));
            }
            other => panic!("expected Rdef detail, got {other:?}"),
        }

        assert_eq!(
            info.find_chunk(b"ISGN").unwrap().detail,
            DxbcChunkDetail::Signature(SignatureSummary { element_count: 3 })
        );
        assert_eq!(
            info.find_chunk(b"OSGN").unwrap().detail,
            DxbcChunkDetail::Signature(SignatureSummary { element_count: 2 })
        );

        match &info.find_chunk(b"STAT").unwrap().detail {
            DxbcChunkDetail::Stat(st) => {
                assert_eq!(st.instruction_count, 128);
                assert_eq!(st.temp_register_count, 12);
                assert_eq!(st.dcl_count, Some(7));
            }
            other => panic!("expected Stat detail, got {other:?}"),
        }

        match &info.find_chunk(b"SFI0").unwrap().detail {
            DxbcChunkDetail::FeatureInfo(sfi) => {
                assert_eq!(sfi.flags, sfi0_flags);
                assert!(sfi.features.contains(&"Doubles"));
                assert!(sfi.features.contains(&"WaveOps"));
                assert!(sfi.features.contains(&"Int64Ops"));
            }
            other => panic!("expected FeatureInfo detail, got {other:?}"),
        }
    }

    #[test]
    fn recognizes_all_dxil_shader_kinds_and_missing_llvm_magic() {
        for (raw, expected) in [
            (0, DxilShaderKind::Pixel),
            (1, DxilShaderKind::Vertex),
            (2, DxilShaderKind::Geometry),
            (3, DxilShaderKind::Hull),
            (4, DxilShaderKind::Domain),
            (5, DxilShaderKind::Compute),
            (6, DxilShaderKind::Library),
            (13, DxilShaderKind::Mesh),
            (14, DxilShaderKind::Amplification),
        ] {
            let dxil = build_dxil_chunk_payload(raw, 6, 5, 1, 5, false);
            let container = build_dxbc_container(&[(*b"DXIL", &dxil)]);
            let info = parse_dxbc(&container).unwrap();
            assert_eq!(info.shader_kind(), Some(expected));
            assert!(info.is_dxil());
            assert!(!info.has_llvm_bitcode());
        }
    }

    #[test]
    fn rejects_truncated_bad_magic_and_overflowing_dxbc_containers_without_panicking() {
        // Empty / truncated header
        assert!(matches!(
            parse_dxbc(&[]),
            Err(DxbcError::TooShort {
                actual: 0,
                expected_min: 32
            })
        ));
        assert!(matches!(
            parse_dxbc(&[0u8; 16]),
            Err(DxbcError::TooShort { .. })
        ));

        // Bad magic
        let mut bad_magic = build_dxbc_container(&[]);
        bad_magic[0..4].copy_from_slice(b"NOPE");
        assert!(matches!(
            parse_dxbc(&bad_magic),
            Err(DxbcError::InvalidMagic(_))
        ));

        // Declared total_size larger than actual buffer
        let mut bad_total = build_dxbc_container(&[]);
        bad_total[24..28].copy_from_slice(&9999u32.to_le_bytes());
        assert!(matches!(
            parse_dxbc(&bad_total),
            Err(DxbcError::InvalidTotalSize { .. })
        ));

        // Huge chunk_count causing chunk table overflow / out-of-bounds
        let mut huge_count = build_dxbc_container(&[]);
        huge_count[28..32].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            parse_dxbc(&huge_count),
            Err(DxbcError::ChunkTableOutOfBounds { .. } | DxbcError::Overflow(_))
        ));

        // Chunk offset overflowing u32::MAX
        let dxil = build_dxil_chunk_payload(0, 6, 0, 1, 0, true);
        let mut oob_offset = build_dxbc_container(&[(*b"DXIL", &dxil)]);
        oob_offset[32..36].copy_from_slice(&(u32::MAX - 2).to_le_bytes());
        assert!(matches!(
            parse_dxbc(&oob_offset),
            Err(DxbcError::ChunkOutOfBounds { .. } | DxbcError::Overflow(_))
        ));

        // Chunk offset pointing inside header/table
        let mut overlap_offset = build_dxbc_container(&[(*b"DXIL", &dxil)]);
        overlap_offset[32..36].copy_from_slice(&4u32.to_le_bytes());
        assert!(matches!(
            parse_dxbc(&overlap_offset),
            Err(DxbcError::ChunkOutOfBounds { .. })
        ));

        // Chunk payload length exceeding container bounds
        let mut oob_size = build_dxbc_container(&[(*b"DXIL", &dxil)]);
        let chunk_offset = u32::from_le_bytes(oob_size[32..36].try_into().unwrap()) as usize;
        oob_size[chunk_offset + 4..chunk_offset + 8].copy_from_slice(&99999u32.to_le_bytes());
        assert!(matches!(
            parse_dxbc(&oob_size),
            Err(DxbcError::ChunkOutOfBounds { .. })
        ));
    }

    #[test]
    fn renders_clean_markdown_and_json_summaries_without_raw_bytecode() {
        let dxil = build_dxil_chunk_payload(13, 6, 5, 1, 5, true); // Mesh shader ms_6_5
        let isgn = build_signature_chunk_payload(4);
        let container = build_dxbc_container(&[(*b"DXIL", &dxil), (*b"ISGN", &isgn)]);
        let info = parse_dxbc(&container).unwrap();

        let md = render_dxbc_info_md(Some("shaders/mesh.dxil"), &info);
        assert!(md.contains("# DXBC / DXIL Container Summary: `shaders/mesh.dxil`"));
        assert!(md.contains(&format_shader_hash(info.fnv1a_hash)));
        assert!(md.contains("`mesh`"));
        assert!(md.contains("`6.5`"));
        assert!(md.contains("| `DXIL` |"));
        assert!(md.contains("| `ISGN` |"));
        assert!(md.contains("4 element(s)"));

        let json = render_dxbc_info_json(Some("shaders/mesh.dxil"), &info);
        assert!(json.contains("\"path\": \"shaders/mesh.dxil\""));
        assert!(json.contains("\"is_dxil\": true"));
        assert!(json.contains("\"has_llvm_bitcode\": true"));
        assert!(json.contains("\"shader_kind\": \"mesh\""));
        assert!(json.contains("\"shader_model\": \"6.5\""));
        assert!(json.contains("\"fourcc\": \"DXIL\""));
        assert!(json.contains("\"element_count\": 4"));

        let batch = vec![
            DxbcFileEntry {
                path: "ok.dxil".to_string(),
                result: Ok(info),
            },
            DxbcFileEntry {
                path: "broken.dxbc".to_string(),
                result: Err("invalid DXBC magic".to_string()),
            },
        ];
        let batch_md = render_dxbc_batch_md(&batch);
        assert!(batch_md.contains("| Total containers | 2 |"));
        assert!(batch_md.contains("`ok.dxil`"));
        assert!(batch_md.contains("`broken.dxbc`"));

        let batch_json = render_dxbc_batch_json(&batch);
        assert!(batch_json.contains("\"total\": 2"));
        assert!(batch_json.contains("\"error\": \"invalid DXBC magic\""));
    }

    #[test]
    fn rejects_every_truncated_header_length_and_truncated_tables_or_chunks() {
        let dxil = build_dxil_chunk_payload(0, 6, 6, 1, 6, true);
        let isgn = build_signature_chunk_payload(2);
        let valid = build_dxbc_container(&[(*b"DXIL", &dxil), (*b"ISGN", &isgn)]);

        // Every truncated header prefix 0..32 bytes must fail with TooShort.
        for len in 0..DXBC_HEADER_BYTES {
            let err = parse_dxbc(&valid[..len]).unwrap_err();
            assert_eq!(
                err,
                DxbcError::TooShort {
                    actual: len,
                    expected_min: DXBC_HEADER_BYTES,
                }
            );
            assert_eq!(parse_dxbc_chunks(&valid[..len]).unwrap_err(), err);
        }

        // Declared total_size == 0 or < DXBC_HEADER_BYTES fails with InvalidTotalSize.
        for declared in [0u32, 1, 31] {
            let mut buf = build_dxbc_container(&[]);
            buf[24..28].copy_from_slice(&declared.to_le_bytes());
            assert_eq!(
                parse_dxbc(&buf).unwrap_err(),
                DxbcError::InvalidTotalSize {
                    declared,
                    actual: buf.len(),
                }
            );
        }

        // Truncated container body (1 byte short of total_size).
        assert_eq!(
            parse_dxbc(&valid[..valid.len() - 1]).unwrap_err(),
            DxbcError::InvalidTotalSize {
                declared: valid.len() as u32,
                actual: valid.len() - 1,
            }
        );

        // Truncated chunk offset table: total_size covers header + 1 offset, but chunk_count == 2.
        let mut trunc_table = build_dxbc_container(&[]);
        trunc_table.extend_from_slice(&40u32.to_le_bytes());
        let len = trunc_table.len() as u32;
        trunc_table[24..28].copy_from_slice(&len.to_le_bytes());
        trunc_table[28..32].copy_from_slice(&2u32.to_le_bytes());
        assert_eq!(
            parse_dxbc_chunks(&trunc_table).unwrap_err(),
            DxbcError::ChunkTableOutOfBounds {
                chunk_count: 2,
                table_end: DXBC_HEADER_BYTES + 8,
                container_size: trunc_table.len(),
            }
        );

        // Truncated 8-byte chunk header (only 7 bytes after offset table).
        let mut trunc_chunk_hdr = build_dxbc_container(&[]);
        trunc_chunk_hdr.extend_from_slice(&36u32.to_le_bytes()); // offset[0] = 36
        trunc_chunk_hdr.extend_from_slice(b"DXIL\x00\x00\x00"); // 7 bytes, missing last size byte
        let len = trunc_chunk_hdr.len() as u32;
        trunc_chunk_hdr[24..28].copy_from_slice(&len.to_le_bytes());
        trunc_chunk_hdr[28..32].copy_from_slice(&1u32.to_le_bytes());
        assert_eq!(
            parse_dxbc_chunks(&trunc_chunk_hdr).unwrap_err(),
            DxbcError::ChunkOutOfBounds {
                chunk_index: 0,
                offset: 36,
                size: 0,
                container_size: trunc_chunk_hdr.len(),
            }
        );

        // Trailing bytes after total_size are ignored for valid containers,
        // and chunks must never be allowed to read past total_size into trailing bytes.
        let mut with_trailing = valid.clone();
        with_trailing.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        let parsed = parse_dxbc(&with_trailing).expect("valid container with trailing bytes");
        assert_eq!(parsed.total_size as usize, valid.len());
        assert_eq!(parsed.actual_bytes, with_trailing.len());
        assert_eq!(parsed.chunks.len(), 2);

        // Inflate second chunk size by 2 bytes so it stays inside with_trailing.len()
        // but exceeds declared total_size: must still be rejected as ChunkOutOfBounds.
        let second_off = u32::from_le_bytes(with_trailing[36..40].try_into().unwrap()) as usize;
        let orig_size =
            u32::from_le_bytes(with_trailing[second_off + 4..second_off + 8].try_into().unwrap());
        with_trailing[second_off + 4..second_off + 8]
            .copy_from_slice(&(orig_size + 2).to_le_bytes());
        assert_eq!(
            parse_dxbc_chunks(&with_trailing).unwrap_err(),
            DxbcError::ChunkOutOfBounds {
                chunk_index: 1,
                offset: second_off as u32,
                size: orig_size + 2,
                container_size: valid.len(),
            }
        );
    }

    #[test]
    fn handles_zero_length_slices_empty_containers_and_zero_length_chunk_payloads() {
        // Zero-length input slice on all entry points and sub-parsers must never panic.
        assert!(matches!(
            parse_dxbc(&[]),
            Err(DxbcError::TooShort {
                actual: 0,
                expected_min: 32
            })
        ));
        assert!(matches!(
            parse_dxbc_chunks(&[]),
            Err(DxbcError::TooShort {
                actual: 0,
                expected_min: 32
            })
        ));
        assert_eq!(parse_dxil_chunk(&[]), None);
        assert_eq!(parse_legacy_shader_chunk(&[]), None);
        assert_eq!(parse_rdef_chunk(&[]), None);
        assert_eq!(parse_signature_chunk(&[]), None);
        assert_eq!(parse_stat_chunk(&[]), None);
        assert_eq!(parse_hash_chunk(&[]), None);
        assert_eq!(parse_sfi0_chunk(&[]), None);
        assert_eq!(read_u32_le(&[], 0), None);
        assert_eq!(read_c_string(&[], 0, 16), None);
        assert_eq!(read_c_string(&[], 1, 16), None);
        assert_eq!(read_c_string(b"abc\0", 1, 0), None);

        // Valid zero-chunk container.
        let empty_container = build_dxbc_container(&[]);
        let empty_info = parse_dxbc(&empty_container).expect("zero-chunk container should parse");
        assert!(empty_info.chunks.is_empty());
        assert!(!empty_info.is_dxil());
        assert_eq!(empty_info.shader_kind(), None);
        assert_eq!(empty_info.shader_model(), None);
        assert!(render_dxbc_info_md(None, &empty_info).contains("No chunks present."));
        assert!(render_dxbc_info_json(None, &empty_info).contains("\"chunks\": []"));

        // Adjacent zero-length chunk payloads (recognized + custom tags) must parse
        // cleanly without panicking and degrade recognized empty chunks to Detail::Other.
        let zero_payload_container = build_dxbc_container(&[
            (*b"DXIL", &[]),
            (*b"SHDR", &[]),
            (*b"RDEF", &[]),
            (*b"ISGN", &[]),
            (*b"STAT", &[]),
            (*b"HASH", &[]),
            (*b"SFI0", &[]),
            (*b"PRIV", &[]),
        ]);
        let refs = parse_dxbc_chunks(&zero_payload_container)
            .expect("zero-length chunk payloads should parse");
        assert_eq!(refs.len(), 8);
        assert!(refs.iter().all(|c| c.payload.is_empty()));

        let info = parse_dxbc(&zero_payload_container).expect("zero-length chunks summarize safely");
        assert_eq!(info.chunks.len(), 8);
        assert!(info.chunks.iter().all(|c| c.size == 0 && c.detail == DxbcChunkDetail::Other));
        assert!(!info.is_dxil());
        assert!(!info.has_llvm_bitcode());
    }

    #[test]
    fn rejects_overflowing_chunk_counts_offsets_sizes_and_subparser_overflows() {
        // 1. chunk_count * 4 u32 multiplication overflow (0x4000_0000 and u32::MAX)
        for bad_count in [0x4000_0000u32, u32::MAX] {
            let mut buf = build_dxbc_container(&[]);
            buf[28..32].copy_from_slice(&bad_count.to_le_bytes());
            assert!(matches!(
                parse_dxbc_chunks(&buf),
                Err(DxbcError::Overflow(_))
            ));
        }

        // 2. DXBC_HEADER_BYTES + chunk_count * 4 u32 addition overflow (0x3fff_ffff * 4 = 0xffff_fffc)
        let mut add_ovf = build_dxbc_container(&[]);
        add_ovf[28..32].copy_from_slice(&0x3fff_ffffu32.to_le_bytes());
        assert!(matches!(
            parse_dxbc_chunks(&add_ovf),
            Err(DxbcError::Overflow(_))
        ));

        // 3. chunk_count exceeding DXBC_MAX_CHUNKS when table fits inside container_size
        let over_cap = DXBC_MAX_CHUNKS + 1;
        let total = DXBC_HEADER_BYTES + (over_cap as usize) * 4;
        let mut too_many = vec![0u8; total];
        too_many[0..4].copy_from_slice(&DXBC_MAGIC);
        too_many[24..28].copy_from_slice(&(total as u32).to_le_bytes());
        too_many[28..32].copy_from_slice(&over_cap.to_le_bytes());
        assert_eq!(
            parse_dxbc_chunks(&too_many).unwrap_err(),
            DxbcError::TooManyChunks {
                chunk_count: over_cap,
                max_chunks: DXBC_MAX_CHUNKS,
            }
        );

        // 4. Chunk offset + 8 u32 overflow (u32::MAX - 7 ..= u32::MAX)
        for bad_offset in [u32::MAX - 7, u32::MAX - 1, u32::MAX] {
            let mut buf = build_dxbc_container(&[(*b"PRIV", &[1, 2, 3, 4])]);
            buf[32..36].copy_from_slice(&bad_offset.to_le_bytes());
            assert!(matches!(
                parse_dxbc_chunks(&buf),
                Err(DxbcError::Overflow(_))
            ));
        }

        // 5. Chunk payload_start + size u32 overflow
        let mut size_ovf = build_dxbc_container(&[(*b"PRIV", &[1, 2, 3, 4])]);
        let off = u32::from_le_bytes(size_ovf[32..36].try_into().unwrap()) as usize;
        for bad_size in [u32::MAX - (off as u32) - 7, u32::MAX] {
            size_ovf[off + 4..off + 8].copy_from_slice(&bad_size.to_le_bytes());
            assert!(matches!(
                parse_dxbc_chunks(&size_ovf),
                Err(DxbcError::Overflow(_))
            ));
        }

        // 6. DXIL bitcode_offset / bitcode_size overflow & header-overlapping bitcode_offset < 16
        let mut dxil_ovf = build_dxil_chunk_payload(5, 6, 6, 1, 6, true);
        dxil_ovf[16..20].copy_from_slice(&u32::MAX.to_le_bytes()); // bitcode_offset overflow
        let parsed = parse_dxil_chunk(&dxil_ovf).expect("DXIL header still decodes");
        assert!(!parsed.has_llvm_bitcode_magic);

        let mut dxil_size_ovf = build_dxil_chunk_payload(5, 6, 6, 1, 6, true);
        dxil_size_ovf[20..24].copy_from_slice(&u32::MAX.to_le_bytes()); // bitcode_size overflow
        let parsed = parse_dxil_chunk(&dxil_size_ovf).expect("DXIL header still decodes");
        assert!(!parsed.has_llvm_bitcode_magic);

        // Craft a DXIL payload whose DxilMagic bytes happen to be LLVM_BITCODE_MAGIC and
        // bitcode_offset = 0 (aliasing bytes 8..12 inside DxilProgramHeader): must be rejected!
        let mut dxil_alias = build_dxil_chunk_payload(5, 6, 6, 1, 6, false);
        dxil_alias[8..12].copy_from_slice(&LLVM_BITCODE_MAGIC);
        dxil_alias[16..20].copy_from_slice(&0u32.to_le_bytes());
        dxil_alias[20..24].copy_from_slice(&4u32.to_le_bytes());
        let parsed = parse_dxil_chunk(&dxil_alias).expect("DXIL header still decodes");
        assert!(!parsed.has_llvm_bitcode_magic);

        // 7. RDEF creator_offset overflowing, pointing inside RDEF header (< 28), or unterminated
        let mut rdef = build_rdef_chunk_payload(1, 2, 6, 6, "dxc");
        rdef[24..28].copy_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(parse_rdef_chunk(&rdef).unwrap().creator, None);

        rdef[24..28].copy_from_slice(&4u32.to_le_bytes()); // inside fixed header
        assert_eq!(parse_rdef_chunk(&rdef).unwrap().creator, None);

        let mut rdef_unterm = build_rdef_chunk_payload(1, 2, 6, 6, "dxc");
        *rdef_unterm.last_mut().unwrap() = b'X'; // overwrite NUL terminator
        assert_eq!(parse_rdef_chunk(&rdef_unterm).unwrap().creator, None);
    }

    #[test]
    fn rejects_invalid_duplicate_and_overlapping_chunk_tags_and_ranges() {
        // Invalid FourCC tags: NULs, control bytes, DEL, non-ASCII, all-space, leading space.
        let invalid_tags: [[u8; 4]; 7] = [
            [0, 0, 0, 0],
            *b"DX\x00L",
            *b"DX\nL",
            *b"DX\x7fL",
            [0x80, b'A', b'B', b'C'],
            *b"    ",
            *b" DXI",
        ];
        for bad_tag in invalid_tags {
            assert!(!is_valid_fourcc(bad_tag));
            let container = build_dxbc_container(&[(bad_tag, &[1, 2, 3, 4])]);
            assert_eq!(
                parse_dxbc_chunks(&container).unwrap_err(),
                DxbcError::InvalidChunkTag {
                    chunk_index: 0,
                    offset: (DXBC_HEADER_BYTES + 4) as u32,
                    fourcc: bad_tag,
                }
            );
        }

        // Valid trailing-space FourCC tag is accepted.
        assert!(is_valid_fourcc(*b"FOO "));
        let trailing_space = build_dxbc_container(&[(*b"FOO ", &[0xaa])]);
        let chunks = parse_dxbc_chunks(&trailing_space).expect("trailing-space FourCC is valid");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].fourcc, *b"FOO ");

        // Duplicate FourCC tags across distinct non-overlapping chunks must be rejected.
        let isgn_a = build_signature_chunk_payload(1);
        let isgn_b = build_signature_chunk_payload(2);
        let dup_tag = build_dxbc_container(&[(*b"ISGN", &isgn_a), (*b"ISGN", &isgn_b)]);
        assert_eq!(
            parse_dxbc_chunks(&dup_tag).unwrap_err(),
            DxbcError::DuplicateChunkTag {
                fourcc: *b"ISGN",
                first_index: 0,
                duplicate_index: 1,
            }
        );
        assert_eq!(
            parse_dxbc(&dup_tag).unwrap_err(),
            DxbcError::DuplicateChunkTag {
                fourcc: *b"ISGN",
                first_index: 0,
                duplicate_index: 1,
            }
        );

        // Duplicate chunk offset (two offset-table slots aliasing the exact same chunk).
        let mut dup_offset = build_dxbc_container(&[(*b"ISGN", &isgn_a), (*b"OSGN", &isgn_b)]);
        let off0 = u32::from_le_bytes(dup_offset[32..36].try_into().unwrap());
        dup_offset[36..40].copy_from_slice(&off0.to_le_bytes());
        assert_eq!(
            parse_dxbc_chunks(&dup_offset).unwrap_err(),
            DxbcError::OverlappingChunks {
                first_index: 0,
                first_offset: off0,
                first_end: off0 + (DXBC_CHUNK_HEADER_BYTES + isgn_a.len()) as u32,
                second_index: 1,
                second_offset: off0,
            }
        );

        // Partially overlapping chunk ranges (chunk 1 starts inside chunk 0's payload).
        // Craft payload0 so bytes at relative offset 8 form a valid inner chunk header ("OSGN", size 0).
        let mut payload0 = vec![0u8; 16];
        payload0[8..12].copy_from_slice(b"OSGN");
        payload0[12..16].copy_from_slice(&0u32.to_le_bytes());
        let mut partial_overlap = build_dxbc_container(&[(*b"PRIV", &payload0), (*b"STAT", &[])]);
        let off0 = u32::from_le_bytes(partial_overlap[32..36].try_into().unwrap());
        let overlapping_off1 = off0 + (DXBC_CHUNK_HEADER_BYTES as u32) + 8;
        partial_overlap[36..40].copy_from_slice(&overlapping_off1.to_le_bytes());
        assert_eq!(
            parse_dxbc_chunks(&partial_overlap).unwrap_err(),
            DxbcError::OverlappingChunks {
                first_index: 0,
                first_offset: off0,
                first_end: off0 + (DXBC_CHUNK_HEADER_BYTES + payload0.len()) as u32,
                second_index: 1,
                second_offset: overlapping_off1,
            }
        );
    }

    #[test]
    fn enumerates_chunks_deterministically_in_canonical_ascending_offset_order() {
        let dxil = build_dxil_chunk_payload(5, 6, 6, 1, 6, true);
        let isgn = build_signature_chunk_payload(3);
        let stat = build_stat_chunk_payload(64, 8, 4);

        let canonical = build_dxbc_container(&[
            (*b"ISGN", &isgn),
            (*b"STAT", &stat),
            (*b"DXIL", &dxil),
        ]);

        // Clone the container and permute its offset table to [off_2, off_0, off_1].
        let mut permuted = canonical.clone();
        let off0: [u8; 4] = canonical[32..36].try_into().unwrap();
        let off1: [u8; 4] = canonical[36..40].try_into().unwrap();
        let off2: [u8; 4] = canonical[40..44].try_into().unwrap();
        permuted[32..36].copy_from_slice(&off2);
        permuted[36..40].copy_from_slice(&off0);
        permuted[40..44].copy_from_slice(&off1);

        let chunks_canonical = parse_dxbc_chunks(&canonical).expect("canonical container");
        let chunks_permuted = parse_dxbc_chunks(&permuted).expect("permuted offset-table container");
        assert_eq!(chunks_canonical, chunks_permuted);
        assert!(chunks_permuted.windows(2).all(|w| w[0].offset < w[1].offset));
        assert_eq!(
            chunks_permuted.iter().map(|c| c.fourcc).collect::<Vec<_>>(),
            vec![*b"ISGN", *b"STAT", *b"DXIL"]
        );

        let info_canonical = parse_dxbc(&canonical).expect("parse canonical");
        let info_permuted = parse_dxbc(&permuted).expect("parse permuted");
        assert_eq!(info_canonical.chunks, info_permuted.chunks);
        assert_eq!(info_canonical.shader_kind(), info_permuted.shader_kind());
        assert_eq!(info_canonical.shader_model(), info_permuted.shader_model());

        // Repeated runs produce bit-for-bit identical summaries, Markdown, and JSON.
        let info_repeat = parse_dxbc(&permuted).expect("repeat parse");
        assert_eq!(info_permuted, info_repeat);
        assert_eq!(
            render_dxbc_info_md(Some("cs.dxil"), &info_permuted),
            render_dxbc_info_md(Some("cs.dxil"), &info_repeat)
        );
        assert_eq!(
            render_dxbc_info_json(Some("cs.dxil"), &info_permuted),
            render_dxbc_info_json(Some("cs.dxil"), &info_repeat)
        );
    }

    #[test]
    fn formats_all_hardened_dxbc_error_variants_cleanly() {
        let samples = [
            DxbcError::TooManyChunks {
                chunk_count: 300,
                max_chunks: DXBC_MAX_CHUNKS,
            },
            DxbcError::InvalidChunkTag {
                chunk_index: 1,
                offset: 44,
                fourcc: [0x00, 0xff, 0x0a, 0x7f],
            },
            DxbcError::DuplicateChunkTag {
                fourcc: *b"DXIL",
                first_index: 0,
                duplicate_index: 2,
            },
            DxbcError::OverlappingChunks {
                first_index: 0,
                first_offset: 40,
                first_end: 64,
                second_index: 1,
                second_offset: 52,
            },
        ];

        for err in samples {
            let msg = err.to_string();
            assert!(!msg.is_empty());
            assert!(msg.chars().all(|c| !c.is_control()));
        }
    }
}
