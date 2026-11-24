#![forbid(unsafe_code)]

//! Batch AMD RDNA2 (GFX10.3 / Xbox Series Scarlett) -> Naga-validated WGSL -> `WGPK` v1
//! `.wgsl.pack` archive builder.
//!
//! Reads raw RDNA2 shader blobs (`*.bin`, `*.rdna2`, `*.shader`), parses optional filename
//! stage/hash hints (e.g. `vs_<16hex>.rdna2`, `ps_<16hex>.rdna2`, `cs_<16hex>.rdna2`, or
//! `<16hex>.bin`), translates every shader unconditionally through
//! [`crate::rdna2_to_wgsl::translate_rdna2_to_wgsl`] (Naga-validated against baseline WebGPU),
//! and packs all translated shaders deterministically (sorted by ascending `u64` hash) into a
//! binary `WGPK` v1 `.wgsl.pack` via [`crate::pack::write_pack`].
//!
//! Guarantees:
//! * Zero `unsafe` code.
//! * Panic-free across empty, truncated, corrupt, or non-matching directory contents.
//! * Ignores non-files and symlinks that resolve outside the input directory.
//! * Strictly sanitized JSON and text summaries (zero host paths, zero usernames, zero raw bytes).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::fs;
use std::path::{Path, PathBuf};

use crate::convert::{ShaderStage, format_shader_hash};
use crate::dxil_reflect::sanitize_source_path;
use crate::pack::{PackEntry, PackError, write_json_string, write_pack};
use crate::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, WgslBindingInfo, translate_rdna2_to_wgsl,
    write_wgsl_bindings_json,
};

/// File extensions recognized as raw RDNA2 shader blobs (ASCII case-insensitive).
pub const SUPPORTED_RDNA2_EXTENSIONS: &[&str] = &["bin", "rdna2", "shader"];

/// Minimum byte length of a valid RDNA2 shader blob (one 32-bit instruction dword).
pub const MIN_RDNA2_BLOB_BYTES: usize = 4;

/// Optional stage and 64-bit hash hints extracted from a shader blob filename.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rdna2FilenameHint {
    pub stage: Rdna2StageOverride,
    pub hash: Option<u64>,
}

/// A single in-memory raw RDNA2 shader blob passed to [`pack_rdna2_blobs`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2BlobInput {
    /// Sanitized label/filename used solely for optional stage/hash hints.
    pub name: String,
    /// Raw little-endian RDNA2 (GFX10.3) machine-code bytes.
    pub bytes: Vec<u8>,
    /// Explicit stage override; when `Auto`, filename hints and ISA inference apply.
    pub stage_override: Rdna2StageOverride,
    /// Explicit 64-bit hash override; when `None`, filename hash hints or FNV-1a apply.
    pub hash_override: Option<u64>,
}

impl Rdna2BlobInput {
    /// Create a blob input with default `Auto` stage and hash resolution from filename/bytes.
    pub fn new(name: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            bytes,
            stage_override: Rdna2StageOverride::Auto,
            hash_override: None,
        }
    }

    /// Set an explicit stage override on this blob input.
    pub fn with_stage(mut self, stage: Rdna2StageOverride) -> Self {
        self.stage_override = stage;
        self
    }

    /// Set an explicit 64-bit hash override on this blob input.
    pub fn with_hash(mut self, hash: u64) -> Self {
        self.hash_override = Some(hash);
        self
    }
}

/// Options controlling batch RDNA2 -> WGSL -> `WGPK` packing.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rdna2PackOptions {
    /// Default RDNA2 -> WGSL translation options (per-blob overrides/hints take precedence over `Auto`).
    pub translate: Rdna2TranslateOptions,
    /// `WGPK` v1 header flags word (default `0`).
    pub pack_flags: u32,
}

/// Strictly sanitized per-shader entry summary for [`Rdna2PackSummary`]
/// (zero host paths, zero filenames, zero raw bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2PackEntrySummary {
    pub hash_fnv1a64: u64,
    pub stage: ShaderStage,
    pub entry_point: String,
    pub translated_op_count: usize,
    pub unsupported_op_count: usize,
    pub wgsl_byte_len: usize,
    pub bindings: Vec<WgslBindingInfo>,
}

/// Strictly sanitized summary of a batch RDNA2 -> WGSL -> `WGPK` pack operation.
///
/// Contains only deterministic aggregate and per-entry metrics: zero host paths,
/// zero usernames, zero filenames, and zero raw shader bytes.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rdna2PackSummary {
    pub total_blobs: usize,
    pub translated_ok: usize,
    pub skipped_or_failed: usize,
    pub vertex_count: usize,
    pub pixel_count: usize,
    pub compute_count: usize,
    pub pack_bytes: usize,
    pub entries: Vec<Rdna2PackEntrySummary>,
}

impl Rdna2PackSummary {
    /// Render a strictly sanitized deterministic JSON summary.
    pub fn to_json(&self) -> String {
        let mut out = String::with_capacity(512);
        out.push_str("{\n");
        out.push_str("  \"format\": \"wgpk_v1\",\n");
        out.push_str("  \"isa\": \"rdna2_gfx10_3\",\n");
        let _ = writeln!(out, "  \"total_blobs\": {},", self.total_blobs);
        let _ = writeln!(out, "  \"translated_ok\": {},", self.translated_ok);
        let _ = writeln!(out, "  \"skipped_or_failed\": {},", self.skipped_or_failed);
        let _ = writeln!(out, "  \"vertex_count\": {},", self.vertex_count);
        let _ = writeln!(out, "  \"pixel_count\": {},", self.pixel_count);
        let _ = writeln!(out, "  \"compute_count\": {},", self.compute_count);
        let _ = writeln!(out, "  \"pack_bytes\": {},", self.pack_bytes);
        if self.entries.is_empty() {
            out.push_str("  \"entries\": []\n");
        } else {
            out.push_str("  \"entries\": [\n");
            for (i, entry) in self.entries.iter().enumerate() {
                out.push_str("    {\n");
                out.push_str("      \"hash_fnv1a64\": ");
                write_json_string(&mut out, &format_shader_hash(entry.hash_fnv1a64));
                out.push_str(",\n");
                out.push_str("      \"stage\": ");
                write_json_string(&mut out, entry.stage.as_str());
                out.push_str(",\n");
                out.push_str("      \"entry_point\": ");
                write_json_string(&mut out, &entry.entry_point);
                out.push_str(",\n");
                let _ = writeln!(
                    out,
                    "      \"translated_op_count\": {},",
                    entry.translated_op_count
                );
                let _ = writeln!(
                    out,
                    "      \"unsupported_op_count\": {},",
                    entry.unsupported_op_count
                );
                let _ = writeln!(out, "      \"wgsl_byte_len\": {},", entry.wgsl_byte_len);
                out.push_str("      \"bindings\": ");
                write_wgsl_bindings_json(&mut out, &entry.bindings, "      ");
                out.push('\n');
                if i + 1 < self.entries.len() {
                    out.push_str("    },\n");
                } else {
                    out.push_str("    }\n");
                }
            }
            out.push_str("  ]\n");
        }
        out.push_str("}\n");
        out
    }

    /// Render a strictly sanitized deterministic plain-text summary.
    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(256);
        out.push_str("WGPK v1 RDNA2 shader pack summary\n");
        let _ = writeln!(out, "  total_blobs:       {}", self.total_blobs);
        let _ = writeln!(out, "  translated_ok:     {}", self.translated_ok);
        let _ = writeln!(out, "  skipped_or_failed: {}", self.skipped_or_failed);
        let _ = writeln!(out, "  vertex_count:      {}", self.vertex_count);
        let _ = writeln!(out, "  pixel_count:       {}", self.pixel_count);
        let _ = writeln!(out, "  compute_count:     {}", self.compute_count);
        let _ = writeln!(out, "  pack_bytes:        {}", self.pack_bytes);
        out
    }
}

/// Complete output of [`pack_rdna2_blobs`] or [`pack_rdna2_directory`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2PackResult {
    /// Serialized `WGPK` v1 binary `.wgsl.pack` bytes.
    pub pack_bytes: Vec<u8>,
    /// Packed WGSL entries in deterministic ascending hash order.
    pub entries: Vec<PackEntry>,
    /// Strictly sanitized summary metrics.
    pub summary: Rdna2PackSummary,
}

/// Errors produced when packing RDNA2 shader directories or blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rdna2PackError {
    InvalidInputDir(String),
    Io(String),
    Pack(PackError),
}

impl fmt::Display for Rdna2PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInputDir(msg) => write!(f, "invalid RDNA2 input directory: {msg}"),
            Self::Io(msg) => write!(f, "I/O error while packing RDNA2 shaders: {msg}"),
            Self::Pack(err) => write!(f, "failed to serialize WGPK pack: {err}"),
        }
    }
}

impl std::error::Error for Rdna2PackError {}

/// Returns `true` if `path` has a supported RDNA2 shader blob extension
/// (`bin`, `rdna2`, or `shader`, ASCII case-insensitive).
pub fn is_supported_rdna2_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            SUPPORTED_RDNA2_EXTENSIONS
                .iter()
                .any(|allowed| ext.eq_ignore_ascii_case(allowed))
        })
}

fn parse_stage_token(token: &str) -> Option<Rdna2StageOverride> {
    if token.eq_ignore_ascii_case("vs") || token.eq_ignore_ascii_case("vertex") {
        Some(Rdna2StageOverride::Vertex)
    } else if token.eq_ignore_ascii_case("ps")
        || token.eq_ignore_ascii_case("fs")
        || token.eq_ignore_ascii_case("pixel")
        || token.eq_ignore_ascii_case("fragment")
    {
        Some(Rdna2StageOverride::Pixel)
    } else if token.eq_ignore_ascii_case("cs") || token.eq_ignore_ascii_case("compute") {
        Some(Rdna2StageOverride::Compute)
    } else {
        None
    }
}

fn parse_exact_hex_u64(token: &str) -> Option<u64> {
    let hex = token
        .strip_prefix("0x")
        .or_else(|| token.strip_prefix("0X"))
        .unwrap_or(token);
    if hex.len() != 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

/// Parse optional pipeline stage and 64-bit hex hash hints from a shader blob filename.
///
/// Supported patterns include:
/// * `vs_<16hex>.rdna2`, `ps_<16hex>.rdna2`, `cs_<16hex>.rdna2`
/// * `vs_0x<16hex>.bin`, `fs_<16hex>.shader`, `vertex-<16hex>.bin`
/// * `<16hex>.bin`, `0x<16hex>.rdna2`
/// * `<16hex>_ps.rdna2`, `splash_vs.bin`, `vs_splash_<16hex>.bin`
///
/// Any unrecognized name gracefully returns `Rdna2StageOverride::Auto` and `hash: None`.
pub fn parse_filename_hint(name: &str) -> Rdna2FilenameHint {
    let basename = name
        .rsplit(['/', '\\'])
        .find(|part| !part.is_empty())
        .unwrap_or("");
    if basename.is_empty() {
        return Rdna2FilenameHint::default();
    }

    let stem = match basename.rsplit_once('.') {
        Some((prefix, _ext)) if !prefix.is_empty() => prefix,
        Some(_) => return Rdna2FilenameHint::default(),
        None => basename,
    };

    let parts: Vec<&str> = stem
        .split(['_', '-'])
        .filter(|part| !part.is_empty())
        .collect();
    if parts.is_empty() {
        return Rdna2FilenameHint::default();
    }

    let (stage, remaining): (Rdna2StageOverride, &[&str]) =
        if let Some(s) = parse_stage_token(parts[0]) {
            (s, &parts[1..])
        } else if let Some(s) = parse_stage_token(parts[parts.len() - 1]) {
            (s, &parts[..parts.len() - 1])
        } else {
            (Rdna2StageOverride::Auto, &parts[..])
        };

    let hash = match remaining {
        [] => None,
        [single] => parse_exact_hex_u64(single),
        many => many
            .last()
            .and_then(|tok| parse_exact_hex_u64(tok))
            .or_else(|| many.first().and_then(|tok| parse_exact_hex_u64(tok))),
    };

    Rdna2FilenameHint { stage, hash }
}

/// Translate and deterministically pack an in-memory slice of raw RDNA2 shader blobs into a
/// `WGPK` v1 `.wgsl.pack` archive.
///
/// Every shader is unconditionally translated and validated by Naga via
/// [`translate_rdna2_to_wgsl`]. Blobs shorter than [`MIN_RDNA2_BLOB_BYTES`], blobs whose
/// translation fails, or blobs colliding on an already-packed hash are counted in
/// `summary.skipped_or_failed` without panicking.
///
/// Entries in the resulting `WGPK` binary are deterministically sorted by ascending `u64` hash.
pub fn pack_rdna2_blobs(
    blobs: &[Rdna2BlobInput],
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let mut packed_by_hash: BTreeMap<u64, (PackEntry, Rdna2PackEntrySummary)> = BTreeMap::new();
    let mut skipped_or_failed = 0usize;

    // Sort a view of indices deterministically by blob name then raw content length/bytes
    // so duplicate-hash tie-breaking is independent of caller slice order.
    let mut order: Vec<usize> = (0..blobs.len()).collect();
    order.sort_by(|&a, &b| {
        blobs[a]
            .name
            .cmp(&blobs[b].name)
            .then_with(|| blobs[a].bytes.len().cmp(&blobs[b].bytes.len()))
            .then_with(|| blobs[a].bytes.cmp(&blobs[b].bytes))
    });

    for idx in order {
        let blob = &blobs[idx];
        if blob.bytes.len() < MIN_RDNA2_BLOB_BYTES {
            skipped_or_failed += 1;
            continue;
        }

        let hint = parse_filename_hint(&blob.name);
        let stage = if blob.stage_override != Rdna2StageOverride::Auto {
            blob.stage_override
        } else if hint.stage != Rdna2StageOverride::Auto {
            hint.stage
        } else {
            options.translate.stage
        };

        let translate_opts = Rdna2TranslateOptions {
            stage,
            entry_point: options.translate.entry_point.clone(),
            min_cbv_vec4_count: options.translate.min_cbv_vec4_count,
            workgroup_size: options.translate.workgroup_size,
            workgroup_id_sgpr_base: options.translate.workgroup_id_sgpr_base,
        };

        let output = match translate_rdna2_to_wgsl(&blob.bytes, &translate_opts) {
            Ok(out) if out.naga_validated => out,
            _ => {
                skipped_or_failed += 1;
                continue;
            }
        };

        let final_hash = blob
            .hash_override
            .or(hint.hash)
            .unwrap_or(output.hash_fnv1a64);

        if packed_by_hash.contains_key(&final_hash) {
            skipped_or_failed += 1;
            continue;
        }

        let wgsl_byte_len = output.wgsl.len();
        packed_by_hash.insert(
            final_hash,
            (
                PackEntry {
                    hash: final_hash,
                    stage: output.stage,
                    wgsl: output.wgsl,
                },
                Rdna2PackEntrySummary {
                    hash_fnv1a64: final_hash,
                    stage: output.stage,
                    entry_point: output.entry_point,
                    translated_op_count: output.translated_op_count,
                    unsupported_op_count: output.unsupported_op_count,
                    wgsl_byte_len,
                    bindings: output.bindings,
                },
            ),
        );
    }

    let mut entries: Vec<PackEntry> = Vec::with_capacity(packed_by_hash.len());
    let mut entry_summaries: Vec<Rdna2PackEntrySummary> = Vec::with_capacity(packed_by_hash.len());
    for (entry, entry_summary) in packed_by_hash.into_values() {
        entries.push(entry);
        entry_summaries.push(entry_summary);
    }
    let mut vertex_count = 0usize;
    let mut pixel_count = 0usize;
    let mut compute_count = 0usize;
    for entry in &entries {
        match entry.stage {
            ShaderStage::Vertex => vertex_count += 1,
            ShaderStage::Fragment => pixel_count += 1,
            ShaderStage::Compute => compute_count += 1,
        }
    }

    let pack_bytes = write_pack(&entries, options.pack_flags).map_err(Rdna2PackError::Pack)?;
    let summary = Rdna2PackSummary {
        total_blobs: blobs.len(),
        translated_ok: entries.len(),
        skipped_or_failed,
        vertex_count,
        pixel_count,
        compute_count,
        pack_bytes: pack_bytes.len(),
        entries: entry_summaries,
    };

    Ok(Rdna2PackResult {
        pack_bytes,
        entries,
        summary,
    })
}

/// Collect regular RDNA2 blob files (`*.bin`, `*.rdna2`, `*.shader`) inside `input_dir`,
/// ignoring non-files and any symlinks that resolve outside `input_dir`.
fn collect_rdna2_blob_paths(input_dir: &Path) -> Result<Vec<(String, PathBuf)>, Rdna2PackError> {
    let meta = fs::metadata(input_dir).map_err(|_| {
        Rdna2PackError::InvalidInputDir(format!(
            "directory '{}' does not exist or is unreadable",
            sanitize_source_path(input_dir, None)
        ))
    })?;
    if !meta.is_dir() {
        return Err(Rdna2PackError::InvalidInputDir(format!(
            "'{}' is not a directory",
            sanitize_source_path(input_dir, None)
        )));
    }

    let canonical_root = fs::canonicalize(input_dir).map_err(|_| {
        Rdna2PackError::InvalidInputDir(format!(
            "failed to resolve directory '{}'",
            sanitize_source_path(input_dir, None)
        ))
    })?;

    let mut visited_dirs: BTreeSet<PathBuf> = BTreeSet::new();
    visited_dirs.insert(canonical_root.clone());

    let mut dir_stack: Vec<PathBuf> = vec![canonical_root.clone()];
    let mut files: Vec<(String, PathBuf)> = Vec::new();

    while let Some(dir) = dir_stack.pop() {
        let read_dir = fs::read_dir(&dir).map_err(|err| {
            Rdna2PackError::Io(format!(
                "failed to list directory '{}': {}",
                sanitize_source_path(&dir, Some(&canonical_root)),
                err.kind()
            ))
        })?;

        for entry_res in read_dir {
            let Ok(entry) = entry_res else {
                continue;
            };
            let path = entry.path();
            let Ok(sym_meta) = fs::symlink_metadata(&path) else {
                continue;
            };

            if sym_meta.file_type().is_symlink() {
                let Ok(target) = fs::canonicalize(&path) else {
                    // Ignore broken symlinks.
                    continue;
                };
                if !target.starts_with(&canonical_root) {
                    // Ignore symlinks escaping the input directory.
                    continue;
                }
                let Ok(target_meta) = fs::metadata(&target) else {
                    continue;
                };
                if target_meta.is_dir() {
                    if visited_dirs.insert(target.clone()) {
                        dir_stack.push(target);
                    }
                } else if target_meta.is_file() && is_supported_rdna2_extension(&path) {
                    let rel_key = sanitize_source_path(&path, Some(&canonical_root));
                    files.push((rel_key, target));
                }
                continue;
            }

            if sym_meta.is_dir() {
                if let Ok(canon_dir) = fs::canonicalize(&path)
                    && canon_dir.starts_with(&canonical_root)
                    && visited_dirs.insert(canon_dir.clone())
                {
                    dir_stack.push(canon_dir);
                }
                continue;
            }

            if sym_meta.is_file() && is_supported_rdna2_extension(&path) {
                let rel_key = sanitize_source_path(&path, Some(&canonical_root));
                files.push((rel_key, path));
            }
        }
    }

    files.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(files)
}

/// Scan `input_dir` for raw RDNA2 shader blobs (`*.bin`, `*.rdna2`, `*.shader`),
/// translate every blob via [`translate_rdna2_to_wgsl`], and build a deterministic
/// `WGPK` v1 `.wgsl.pack` archive in memory.
///
/// Non-files and symlinks resolving outside `input_dir` are safely ignored.
pub fn pack_rdna2_directory(
    input_dir: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let file_paths = collect_rdna2_blob_paths(input_dir)?;

    let mut blobs = Vec::with_capacity(file_paths.len());
    let mut read_failures = 0usize;

    for (rel_key, path) in &file_paths {
        match fs::read(path) {
            Ok(bytes) => {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| rel_key.clone());
                blobs.push(Rdna2BlobInput::new(name, bytes));
            }
            Err(_) => {
                read_failures += 1;
            }
        }
    }

    let mut result = pack_rdna2_blobs(&blobs, options)?;
    if read_failures > 0 {
        result.summary.total_blobs += read_failures;
        result.summary.skipped_or_failed += read_failures;
    }
    Ok(result)
}

/// Convenience helper that runs [`pack_rdna2_directory`] and writes the resulting
/// `WGPK` v1 `.wgsl.pack` bytes to `out_path` (creating parent directories if needed).
pub fn pack_rdna2_directory_to_file(
    input_dir: &Path,
    out_path: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let result = pack_rdna2_directory(input_dir, options)?;

    let out_label = sanitize_source_path(out_path, None);
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            Rdna2PackError::Io(format!(
                "failed to create output directory for '{out_label}': {}",
                err.kind()
            ))
        })?;
    }
    fs::write(out_path, &result.pack_bytes).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to write WGPK pack '{out_label}': {}",
            err.kind()
        ))
    })?;

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    use naga::valid::{Capabilities, ValidationFlags, Validator};

    use crate::convert::fnv1a_64;
    use crate::pack::{PACK_MAGIC, PACK_VERSION, read_pack};

    fn make_temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("{prefix}_{}_{nanos}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(words.len() * 4);
        for &w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn enc_sopp(op: u8, simm16: u16) -> u32 {
        (0b1_0111_1111_u32 << 23) | (u32::from(op & 0x7F) << 16) | u32::from(simm16)
    }

    fn enc_vop1(op: u8, vdst: u8, src0: u16) -> u32 {
        (0b011_1111_u32 << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(op) << 9)
            | u32::from(src0 & 0x1FF)
    }

    fn enc_vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
        (u32::from(op & 0x3F) << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(vsrc1) << 9)
            | u32::from(src0 & 0x1FF)
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_vop3(
        op: u16,
        vdst: u8,
        src0: u16,
        src1: u16,
        src2: u16,
        clamp: bool,
        omod: u8,
        abs: u8,
        neg: u8,
    ) -> [u32; 2] {
        let w0 = (0b110101u32 << 26)
            | (u32::from(op & 0x3FF) << 16)
            | (u32::from(clamp) << 15)
            | (u32::from(abs & 0x7) << 8)
            | u32::from(vdst);
        let w1 = (u32::from(neg & 0x7) << 29)
            | (u32::from(omod & 0x3) << 27)
            | (u32::from(src2 & 0x1FF) << 18)
            | (u32::from(src1 & 0x1FF) << 9)
            | u32::from(src0 & 0x1FF);
        [w0, w1]
    }

    fn enc_vintrp(op: u8, vdst: u8, attr: u8, attrchan: u8, vsrc: u8) -> u32 {
        (0b110010u32 << 26)
            | (u32::from(vdst) << 18)
            | (u32::from(op & 0x3) << 16)
            | (u32::from(attr & 0x3F) << 10)
            | (u32::from(attrchan & 0x3) << 8)
            | u32::from(vsrc)
    }

    fn enc_exp(
        target: u8,
        en: u8,
        compr: bool,
        done: bool,
        vm: bool,
        vsrc: [u8; 4],
    ) -> [u32; 2] {
        let w0 = (0b111110u32 << 26)
            | (u32::from(vm) << 12)
            | (u32::from(done) << 11)
            | (u32::from(compr) << 10)
            | (u32::from(target & 0x3F) << 4)
            | u32::from(en & 0xF);
        let w1 = u32::from(vsrc[0])
            | (u32::from(vsrc[1]) << 8)
            | (u32::from(vsrc[2]) << 16)
            | (u32::from(vsrc[3]) << 24);
        [w0, w1]
    }

    fn enc_mimg(
        op: u8,
        dmask: u8,
        dim: u8,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        ssamp: u8,
    ) -> [u32; 2] {
        let w0 = (0b111100u32 << 26)
            | (u32::from(op & 0x7F) << 18)
            | (u32::from(dmask & 0xF) << 8)
            | (u32::from(dim & 0x7) << 3)
            | u32::from((op >> 7) & 0x1);
        let w1 = (u32::from(ssamp & 0x1F) << 21)
            | (u32::from(srsrc & 0x1F) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    /// Synthetic RDNA2 fullscreen-triangle vertex shader (`vertex_index` -> clip pos + UV).
    fn build_fullscreen_triangle_vs_rdna2() -> Vec<u8> {
        let mut words = vec![
            enc_vop2(0x18, 1, 128 + 1, 0), // v_lshlrev_b32 v1, 1, v0
            enc_vop2(0x19, 1, 128 + 2, 1), // v_and_b32 v1, 2, v1
            enc_vop2(0x19, 2, 128 + 2, 0), // v_and_b32 v2, 2, v0
            enc_vop1(0x06, 1, 256 + 1),    // v_cvt_f32_u32 v1, v1
            enc_vop1(0x06, 2, 256 + 2),    // v_cvt_f32_u32 v2, v2
        ];
        words.extend_from_slice(&enc_vop3(0x14B, 3, 256 + 1, 244, 243, false, 0, 0, 0));
        words.extend_from_slice(&enc_vop3(0x14B, 4, 256 + 2, 244, 242, false, 0, 0, 0b001));
        words.push(enc_vop1(0x01, 5, 128)); // v_mov_b32 v5, 0
        words.push(enc_vop1(0x01, 6, 242)); // v_mov_b32 v6, 1.0
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [3, 4, 5, 6]));
        words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [1, 2, 0, 0]));
        words.push(enc_sopp(0x01, 0)); // s_endpgm
        words_to_bytes(&words)
    }

    /// Synthetic RDNA2 textured splash pixel shader (`VINTRP` UV -> `image_sample` -> `exp mrt0`).
    fn build_textured_splash_ps_rdna2() -> Vec<u8> {
        let mut words = vec![
            enc_vintrp(0, 2, 0, 0, 0),
            enc_vintrp(1, 2, 0, 0, 1),
            enc_vintrp(0, 3, 0, 1, 0),
            enc_vintrp(1, 3, 0, 1, 1),
        ];
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 2, 1));
        words.push(enc_sopp(0x0C, 0x3F70)); // s_waitcnt vmcnt(0)
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0)); // s_endpgm
        words_to_bytes(&words)
    }

    /// Synthetic RDNA2 compute shader (`s_barrier` + `s_endpgm`).
    fn build_minimal_compute_rdna2() -> Vec<u8> {
        words_to_bytes(&[
            enc_vop1(0x01, 1, 242), // v_mov_b32 v1, 1.0
            enc_sopp(0x0A, 0),      // s_barrier
            enc_sopp(0x01, 0),      // s_endpgm
        ])
    }

    fn assert_naga_wgsl_valid(wgsl: &str) {
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\nWGSL:\n{wgsl}", e.emit_to_string(wgsl)));
        let mut validator = Validator::new(
            ValidationFlags::all(),
            Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        );
        validator.validate(&module).unwrap_or_else(|e| {
            panic!(
                "WGSL validation failed:\n{}\nWGSL:\n{wgsl}",
                e.emit_to_string(wgsl)
            )
        });
    }

    #[test]
    fn filename_hint_parsing_covers_stage_hash_and_fallbacks() {
        assert_eq!(
            parse_filename_hint("vs_0123456789abcdef.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Vertex,
                hash: Some(0x0123_4567_89ab_cdef),
            }
        );
        assert_eq!(
            parse_filename_hint("ps_FEDCBA9876543210.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Pixel,
                hash: Some(0xfedc_ba98_7654_3210),
            }
        );
        assert_eq!(
            parse_filename_hint("cs_0000000000000042.shader"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Compute,
                hash: Some(0x42),
            }
        );
        assert_eq!(
            parse_filename_hint("0011223344556677.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Auto,
                hash: Some(0x0011_2233_4455_6677),
            }
        );
        assert_eq!(
            parse_filename_hint("vs_0x0011223344556677.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Vertex,
                hash: Some(0x0011_2233_4455_6677),
            }
        );
        assert_eq!(
            parse_filename_hint("aabbccddeeff0011_fs.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Pixel,
                hash: Some(0xaabb_ccdd_eeff_0011),
            }
        );
        assert_eq!(
            parse_filename_hint("splash_vs.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Vertex,
                hash: None,
            }
        );
        assert_eq!(
            parse_filename_hint("arbitrary_shader.bin"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Auto,
                hash: None,
            }
        );
        assert_eq!(
            parse_filename_hint("dir/nested/vs_deadbeefcafebabe.rdna2"),
            Rdna2FilenameHint {
                stage: Rdna2StageOverride::Vertex,
                hash: Some(0xdead_beef_cafe_babe),
            }
        );
        assert_eq!(parse_filename_hint(".bin"), Rdna2FilenameHint::default());
        assert_eq!(parse_filename_hint(""), Rdna2FilenameHint::default());
    }

    #[test]
    fn packs_synthetic_fullscreen_vs_and_textured_splash_ps_round_trip_wgpk() {
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let cs_bytes = build_minimal_compute_rdna2();
        let cs_expected_hash = fnv1a_64(&cs_bytes);

        let blobs = vec![
            Rdna2BlobInput::new("ps_2000000000000002.rdna2", ps_bytes),
            Rdna2BlobInput::new("unlabeled_compute.bin", cs_bytes),
            Rdna2BlobInput::new("vs_1000000000000001.rdna2", vs_bytes),
        ];

        let result = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
            .expect("pack_rdna2_blobs should succeed");

        assert_eq!(result.summary.total_blobs, 3);
        assert_eq!(result.summary.translated_ok, 3);
        assert_eq!(result.summary.skipped_or_failed, 0);
        assert_eq!(result.summary.vertex_count, 1);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);
        assert_eq!(result.summary.pack_bytes, result.pack_bytes.len());
        assert_eq!(&result.pack_bytes[0..4], &PACK_MAGIC);

        let decoded = read_pack(&result.pack_bytes).expect("WGPK round-trip read should succeed");
        assert_eq!(decoded.version, PACK_VERSION);
        assert_eq!(decoded.flags, 0);
        assert_eq!(decoded.entries.len(), 3);

        // Entries must be strictly sorted by ascending u64 hash.
        for pair in decoded.entries.windows(2) {
            assert!(
                pair[0].hash < pair[1].hash,
                "WGPK entries must be deterministically sorted by hash"
            );
        }

        let vs_entry = decoded
            .entries
            .iter()
            .find(|e| e.hash == 0x1000_0000_0000_0001)
            .expect("VS entry must exist");
        assert_eq!(vs_entry.stage, ShaderStage::Vertex);
        assert!(vs_entry.wgsl.contains("@vertex"));
        assert!(vs_entry.wgsl.contains("@builtin(vertex_index)"));
        assert!(vs_entry.wgsl.contains("@builtin(position)"));
        assert_naga_wgsl_valid(&vs_entry.wgsl);

        let ps_entry = decoded
            .entries
            .iter()
            .find(|e| e.hash == 0x2000_0000_0000_0002)
            .expect("PS entry must exist");
        assert_eq!(ps_entry.stage, ShaderStage::Fragment);
        assert!(ps_entry.wgsl.contains("@fragment"));
        assert!(ps_entry.wgsl.contains("fn fs_main("));
        assert!(ps_entry.wgsl.contains("texture_2d<f32>"));
        assert!(ps_entry.wgsl.contains("textureSample("));
        assert_naga_wgsl_valid(&ps_entry.wgsl);

        // Per-entry summary + explicit bindings in pack summary JSON.
        assert_eq!(result.summary.entries.len(), 3);
        let ps_summary = result
            .summary
            .entries
            .iter()
            .find(|e| e.hash_fnv1a64 == 0x2000_0000_0000_0002)
            .expect("PS summary must exist");
        assert_eq!(ps_summary.entry_point, "fs_main");
        assert_eq!(ps_summary.bindings.len(), 2);
        let summary_json = result.summary.to_json();
        assert!(summary_json.contains("\"entries\": ["), "JSON:\n{summary_json}");
        assert!(summary_json.contains("\"entry_point\": \"fs_main\""), "JSON:\n{summary_json}");
        assert!(
            summary_json.contains("{\"group\": 0, \"binding\": 1, \"kind\": \"texture_2d\", \"name\": \"tex0\"}"),
            "JSON:\n{summary_json}"
        );

        let cs_entry = decoded
            .entries
            .iter()
            .find(|e| e.hash == cs_expected_hash)
            .expect("auto-hashed CS entry must exist");
        assert_eq!(cs_entry.stage, ShaderStage::Compute);
        assert!(cs_entry.wgsl.contains("@compute"));
        assert!(cs_entry.wgsl.contains("workgroupBarrier()"));
        assert_naga_wgsl_valid(&cs_entry.wgsl);
    }

    #[test]
    fn pack_rdna2_blobs_skips_short_and_duplicate_hashes_deterministically() {
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();

        let blobs_a = vec![
            Rdna2BlobInput::new("empty.bin", vec![]),
            Rdna2BlobInput::new("too_short.rdna2", vec![0xAA, 0xBB, 0xCC]),
            Rdna2BlobInput::new("vs_00000000000000aa.rdna2", vs_bytes.clone()),
            Rdna2BlobInput::new("vs_dup_00000000000000aa.rdna2", vs_bytes.clone()),
            Rdna2BlobInput::new("ps_00000000000000bb.shader", ps_bytes.clone()),
        ];
        let mut blobs_b = blobs_a.clone();
        blobs_b.reverse();

        let res_a = pack_rdna2_blobs(&blobs_a, &Rdna2PackOptions::default()).unwrap();
        let res_b = pack_rdna2_blobs(&blobs_b, &Rdna2PackOptions::default()).unwrap();

        assert_eq!(res_a.pack_bytes, res_b.pack_bytes);
        assert_eq!(res_a.summary, res_b.summary);
        assert_eq!(res_a.summary.total_blobs, 5);
        assert_eq!(res_a.summary.translated_ok, 2);
        assert_eq!(res_a.summary.skipped_or_failed, 3);
        assert_eq!(res_a.summary.vertex_count, 1);
        assert_eq!(res_a.summary.pixel_count, 1);
        assert_eq!(res_a.summary.compute_count, 0);
    }

    #[test]
    fn pack_rdna2_directory_walks_subdirs_ignores_non_matching_and_escaping_symlinks() {
        let root = make_temp_dir("playgta6_rdna2_pack_dir");
        let input_dir = root.join("shaders_in");
        let nested_dir = input_dir.join("splash");
        let outside_dir = root.join("outside");
        fs::create_dir_all(&nested_dir).unwrap();
        fs::create_dir_all(&outside_dir).unwrap();

        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();
        let cs_bytes = build_minimal_compute_rdna2();

        fs::write(nested_dir.join("vs_0000000000000001.rdna2"), &vs_bytes).unwrap();
        fs::write(nested_dir.join("ps_0000000000000002.shader"), &ps_bytes).unwrap();
        fs::write(input_dir.join("0000000000000003.bin"), &cs_bytes).unwrap();
        // Empty blob -> counted as skipped_or_failed.
        fs::write(input_dir.join("empty.bin"), b"").unwrap();
        // Non-matching extensions -> ignored completely.
        fs::write(input_dir.join("notes.txt"), b"ignore me").unwrap();
        fs::write(input_dir.join("already.wgsl"), b"// ignore").unwrap();

        // Outside file + symlink escaping input_dir (best-effort on Windows without symlink privileges).
        let outside_blob = outside_dir.join("vs_9999999999999999.rdna2");
        fs::write(&outside_blob, &vs_bytes).unwrap();
        let escape_link = input_dir.join("vs_9999999999999999.rdna2");
        #[cfg(unix)]
        {
            let _ = std::os::unix::fs::symlink(&outside_blob, &escape_link);
        }
        #[cfg(windows)]
        {
            let _ = std::os::windows::fs::symlink_file(&outside_blob, &escape_link);
        }

        let result = pack_rdna2_directory(&input_dir, &Rdna2PackOptions::default())
            .expect("pack_rdna2_directory should succeed");

        assert_eq!(result.summary.total_blobs, 4);
        assert_eq!(result.summary.translated_ok, 3);
        assert_eq!(result.summary.skipped_or_failed, 1);
        assert_eq!(result.summary.vertex_count, 1);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);

        let decoded = read_pack(&result.pack_bytes).unwrap();
        assert_eq!(decoded.entries.len(), 3);
        assert_eq!(decoded.entries[0].hash, 1);
        assert_eq!(decoded.entries[0].stage, ShaderStage::Vertex);
        assert_eq!(decoded.entries[1].hash, 2);
        assert_eq!(decoded.entries[1].stage, ShaderStage::Fragment);
        assert_eq!(decoded.entries[2].hash, 3);
        assert_eq!(decoded.entries[2].stage, ShaderStage::Compute);
        // Escaping symlink hash 0x9999999999999999 must never be packed.
        assert!(
            decoded
                .entries
                .iter()
                .all(|e| e.hash != 0x9999_9999_9999_9999)
        );

        // Strictly sanitized JSON and text summaries: zero host paths, zero usernames.
        let json = result.summary.to_json();
        let text = result.summary.to_text();
        let root_str = root.to_string_lossy().replace('\\', "/");
        for rendered in [&json, &text] {
            assert!(rendered.contains("total_blobs"));
            assert!(rendered.contains("translated_ok"));
            assert!(rendered.contains("skipped_or_failed"));
            assert!(rendered.contains("vertex_count"));
            assert!(rendered.contains("pixel_count"));
            assert!(rendered.contains("compute_count"));
            assert!(rendered.contains("pack_bytes"));
            assert!(!rendered.contains(&root_str));
            assert!(!rendered.contains("shaders_in"));
            assert!(!rendered.contains("outside"));
        }

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_directory_rejects_missing_or_file_input_without_leaking_host_path() {
        let root = make_temp_dir("playgta6_rdna2_pack_err");
        let missing = root.join("does_not_exist");
        let file = root.join("single.bin");
        fs::write(&file, build_minimal_compute_rdna2()).unwrap();

        let root_str = root.to_string_lossy().replace('\\', "/");

        let err_missing = pack_rdna2_directory(&missing, &Rdna2PackOptions::default())
            .unwrap_err()
            .to_string();
        assert!(!err_missing.contains(&root_str));

        let err_file = pack_rdna2_directory(&file, &Rdna2PackOptions::default())
            .unwrap_err()
            .to_string();
        assert!(!err_file.contains(&root_str));

        let _ = fs::remove_dir_all(&root);
    }

    fn enc_smem(
        op: u8,
        sdata: u8,
        sbase_pair: u8,
        soffset: u8,
        offset: i32,
        glc: bool,
        dlc: bool,
    ) -> [u32; 2] {
        let w0 = (0b111101u32 << 26)
            | (u32::from(op) << 18)
            | (u32::from(glc) << 16)
            | (u32::from(dlc) << 14)
            | (u32::from(sdata & 0x7F) << 6)
            | u32::from(sbase_pair & 0x3F);
        let w1 = (u32::from(soffset & 0x7F) << 25) | ((offset as u32) & 0x001F_FFFF);
        [w0, w1]
    }

    #[test]
    fn pack_rdna2_blobs_emits_tight_cbv_sizes_and_binding_metadata_in_json() {
        // 1) Small CBV shader reading cb[0..2] (offsets 0..44) -> 48 B (3 vec4s)
        let mut small_words = Vec::new();
        small_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x00, false, false));
        small_words.extend_from_slice(&enc_smem(0x08, 8, 0, 0x7D, 0x2C, false, false));
        small_words.push(enc_vop1(0x01, 0, 4));
        small_words.push(enc_vop1(0x01, 1, 8));
        small_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        small_words.push(enc_sopp(0x01, 0));

        // 2) Large CBV shader reading cb[20] (offset 0x140) -> 336 B (21 vec4s)
        let mut large_words = Vec::new();
        large_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x140, false, false));
        large_words.push(enc_vop1(0x01, 0, 4));
        large_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 0, 0, 0]));
        large_words.push(enc_sopp(0x01, 0));

        // 3) Multi-CBV shader: cb0 reading vec4 1 (32 B) and cb1 reading vec4 5 (96 B)
        let mut multi_words = Vec::new();
        multi_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x10, false, false));
        multi_words.extend_from_slice(&enc_smem(0x0A, 8, 2, 0x7D, 0x50, false, false));
        multi_words.push(enc_vop1(0x01, 0, 4));
        multi_words.push(enc_vop1(0x01, 1, 8));
        multi_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        multi_words.push(enc_sopp(0x01, 0));

        let blobs = vec![
            Rdna2BlobInput::new("ps_0000000000000101.rdna2", words_to_bytes(&small_words)),
            Rdna2BlobInput::new("ps_0000000000000102.rdna2", words_to_bytes(&large_words)),
            Rdna2BlobInput::new("ps_0000000000000103.rdna2", words_to_bytes(&multi_words)),
        ];

        let result = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
            .expect("pack_rdna2_blobs should succeed");
        assert_eq!(result.summary.translated_ok, 3);

        let decoded = read_pack(&result.pack_bytes).expect("WGPK read should succeed");
        assert_eq!(decoded.entries.len(), 3);
        assert!(decoded.entries[0].wgsl.contains("array<vec4<u32>, 3>"));
        assert!(decoded.entries[1].wgsl.contains("array<vec4<u32>, 21>"));
        assert!(decoded.entries[2].wgsl.contains("array<vec4<u32>, 2>"));
        assert!(decoded.entries[2].wgsl.contains("array<vec4<u32>, 6>"));
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        // Verify per-entry binding metadata in summary structs and JSON.
        assert_eq!(result.summary.entries[0].bindings[0].min_binding_size_bytes, Some(48));
        assert_eq!(result.summary.entries[0].bindings[0].highest_vec4_offset, Some(2));
        assert_eq!(result.summary.entries[1].bindings[0].min_binding_size_bytes, Some(336));
        assert_eq!(result.summary.entries[1].bindings[0].highest_vec4_offset, Some(20));
        assert_eq!(result.summary.entries[2].bindings.len(), 2);
        assert_eq!(result.summary.entries[2].bindings[0].min_binding_size_bytes, Some(32));
        assert_eq!(result.summary.entries[2].bindings[0].highest_vec4_offset, Some(1));
        assert_eq!(result.summary.entries[2].bindings[1].min_binding_size_bytes, Some(96));
        assert_eq!(result.summary.entries[2].bindings[1].highest_vec4_offset, Some(5));

        let json = result.summary.to_json();
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 48, \"highest_vec4_offset\": 2}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 336, \"highest_vec4_offset\": 20}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 32, \"highest_vec4_offset\": 1}"
            ),
            "JSON:\n{json}"
        );
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 3, \"kind\": \"uniform\", \"name\": \"cb1\", \"min_binding_size_bytes\": 96, \"highest_vec4_offset\": 5}"
            ),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn pack_rdna2_blobs_packs_mixed_vs_fs_and_compute_storage_lds_shader_into_wgpk() {
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let ps_bytes = build_textured_splash_ps_rdna2();

        // Full compute shader using storage buffer load/store, LDS + s_barrier, and image_store:
        let mut cs_words = Vec::new();
        cs_words.push(enc_vop2(0x18, 3, 128 + 2, 0)); // v_lshlrev_b32 v3, 2, v0
        // buffer_load_dword v4, v3, s[4:7], 0 offen (srsrc=1 -> read-only-storage)
        cs_words.extend_from_slice(&[
            (0b111000u32 << 26) | (0x0Cu32 << 18) | (1 << 12),
            (128u32 << 24) | (1u32 << 16) | (4u32 << 8) | 3,
        ]);
        // ds_write_b32 v0, v3, v4
        cs_words.extend_from_slice(&[
            (0b110110u32 << 26) | (0x0Du32 << 18),
            (4u32 << 8) | 3,
        ]);
        cs_words.push(enc_sopp(0x0A, 0)); // s_barrier
        // ds_read_b32 v5, v3, v0
        cs_words.extend_from_slice(&[
            (0b110110u32 << 26) | (0x36u32 << 18),
            (5u32 << 24) | 3,
        ]);
        // buffer_store_dword v5, v3, s[8:11], 0 offen (srsrc=2 -> storage)
        cs_words.extend_from_slice(&[
            (0b111000u32 << 26) | (0x1Cu32 << 18) | (1 << 12),
            (128u32 << 24) | (2u32 << 16) | (5u32 << 8) | 3,
        ]);
        // image_store v[5:8], v[0:1], s[12:19] dmask:0xf dim:1 (srsrc=3 -> storage-texture)
        cs_words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 5, 3, 0));
        cs_words.push(enc_sopp(0x01, 0)); // s_endpgm
        let cs_bytes = words_to_bytes(&cs_words);

        let blobs = vec![
            Rdna2BlobInput::new("vs_0000000000000301.rdna2", vs_bytes),
            Rdna2BlobInput::new("ps_0000000000000302.rdna2", ps_bytes),
            Rdna2BlobInput::new("cs_0000000000000303.rdna2", cs_bytes),
        ];

        let result = pack_rdna2_blobs(&blobs, &Rdna2PackOptions::default())
            .expect("mixed vs+fs+cs WGPK pack should succeed");
        assert_eq!(result.summary.total_blobs, 3);
        assert_eq!(result.summary.translated_ok, 3);
        assert_eq!(result.summary.skipped_or_failed, 0);
        assert_eq!(result.summary.vertex_count, 1);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);

        let decoded = read_pack(&result.pack_bytes).expect("WGPK round-trip should succeed");
        assert_eq!(decoded.entries.len(), 3);
        assert_eq!(decoded.entries[0].stage, ShaderStage::Vertex);
        assert_eq!(decoded.entries[1].stage, ShaderStage::Fragment);
        assert_eq!(decoded.entries[2].stage, ShaderStage::Compute);
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        let cs_entry = &decoded.entries[2];
        assert!(cs_entry.wgsl.contains("fn cs_main("));
        assert!(cs_entry.wgsl.contains("var<workgroup> lds: array<u32, 4096>;"));
        assert!(cs_entry.wgsl.contains("workgroupBarrier();"));
        assert!(cs_entry.wgsl.contains("textureStore("));
        assert!(!cs_entry.wgsl.contains("enable f16"));

        let cs_summary = &result.summary.entries[2];
        assert_eq!(cs_summary.entry_point, "cs_main");
        assert_eq!(cs_summary.unsupported_op_count, 0);
        assert_eq!(cs_summary.bindings.len(), 3);
        assert_eq!(cs_summary.bindings[0].kind, "read-only-storage");
        assert_eq!(cs_summary.bindings[1].kind, "storage");
        assert_eq!(cs_summary.bindings[2].kind, "storage-texture");

        let json = result.summary.to_json();
        assert!(json.contains("\"kind\": \"read-only-storage\""), "JSON:\n{json}");
        assert!(json.contains("\"kind\": \"storage\""), "JSON:\n{json}");
        assert!(json.contains("\"kind\": \"storage-texture\""), "JSON:\n{json}");
    }
}
