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
use std::path::{Component, Path, PathBuf};

use crate::convert::{ShaderStage, fnv1a_64, format_shader_hash};
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

/// Expected `"schema"` identifier in a `playgta6-pipeline-dump` `manifest.json`.
pub const PIPELINE_DUMP_SCHEMA: &str = "playgta6-pipeline-dump";

/// Supported `"schema_version"` in a `playgta6-pipeline-dump` `manifest.json`.
pub const PIPELINE_DUMP_SCHEMA_VERSION: u64 = 1;

/// Maximum nesting depth permitted by the internal zero-dependency JSON parser.
const MAX_JSON_DEPTH: usize = 64;

/// One entry in `manifest.json`'s `"native_stages"` array.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineDumpNativeStage {
    pub code_identity: u64,
    pub identity: Option<u64>,
    pub stage: ShaderStage,
    pub code_file: String,
    pub code_bytes: Option<u64>,
}

/// Parsed `playgta6-pipeline-dump` `manifest.json` (fields needed for RDNA2 packing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineDumpManifest {
    pub schema: String,
    pub schema_version: u64,
    pub native_stages: Vec<PipelineDumpNativeStage>,
}

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
    /// When `true`, skip individual shaders that fail I/O, hash verification, size checks, or
    /// translation/Naga validation and record them in `summary.failures` instead of aborting.
    pub best_effort: bool,
}

/// Strictly sanitized per-shader entry summary for [`Rdna2PackSummary`]
/// (zero host paths, zero filenames, zero raw bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2PackEntrySummary {
    pub hash_fnv1a64: u64,
    pub stage: ShaderStage,
    pub entry_point: String,
    pub naga_validated: bool,
    pub translated_op_count: usize,
    pub unsupported_op_count: usize,
    pub wgsl_byte_len: usize,
    pub bindings: Vec<WgslBindingInfo>,
}

/// Strictly sanitized record of an individual shader that failed packing (zero host paths,
/// zero usernames, zero raw bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2PackFailureSummary {
    pub code_identity: Option<u64>,
    pub stage: Option<ShaderStage>,
    pub reason: String,
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
    pub failed: usize,
    pub manifest_duplicates_skipped: usize,
    pub vertex_count: usize,
    pub pixel_count: usize,
    pub compute_count: usize,
    pub vertex_packed: usize,
    pub fragment_packed: usize,
    pub compute_packed: usize,
    pub pack_bytes: usize,
    pub entries: Vec<Rdna2PackEntrySummary>,
    pub failures: Vec<Rdna2PackFailureSummary>,
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
        let _ = writeln!(out, "  \"failed\": {},", self.failed);
        let _ = writeln!(
            out,
            "  \"manifest_duplicates_skipped\": {},",
            self.manifest_duplicates_skipped
        );
        let _ = writeln!(out, "  \"vertex_count\": {},", self.vertex_count);
        let _ = writeln!(out, "  \"pixel_count\": {},", self.pixel_count);
        let _ = writeln!(out, "  \"compute_count\": {},", self.compute_count);
        let _ = writeln!(out, "  \"vertex_packed\": {},", self.vertex_packed);
        let _ = writeln!(out, "  \"fragment_packed\": {},", self.fragment_packed);
        let _ = writeln!(out, "  \"compute_packed\": {},", self.compute_packed);
        let _ = writeln!(out, "  \"pack_bytes\": {},", self.pack_bytes);
        if self.entries.is_empty() {
            out.push_str("  \"entries\": [],\n");
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
                let _ = writeln!(out, "      \"naga_validated\": {},", entry.naga_validated);
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
            out.push_str("  ],\n");
        }
        if self.failures.is_empty() {
            out.push_str("  \"failures\": []\n");
        } else {
            out.push_str("  \"failures\": [\n");
            for (i, failure) in self.failures.iter().enumerate() {
                out.push_str("    {");
                if let Some(id) = failure.code_identity {
                    out.push_str("\"code_identity\": ");
                    write_json_string(&mut out, &format_shader_hash(id));
                    out.push_str(", ");
                }
                if let Some(stage) = failure.stage {
                    out.push_str("\"stage\": ");
                    write_json_string(&mut out, stage.as_str());
                    out.push_str(", ");
                }
                out.push_str("\"reason\": ");
                write_json_string(&mut out, &failure.reason);
                if i + 1 < self.failures.len() {
                    out.push_str("},\n");
                } else {
                    out.push_str("}\n");
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
        let _ = writeln!(out, "  failed:            {}", self.failed);
        if self.manifest_duplicates_skipped > 0 {
            let _ = writeln!(
                out,
                "  manifest_dups:     {}",
                self.manifest_duplicates_skipped
            );
        }
        let _ = writeln!(out, "  vertex_count:      {}", self.vertex_count);
        let _ = writeln!(out, "  pixel_count:       {}", self.pixel_count);
        let _ = writeln!(out, "  compute_count:     {}", self.compute_count);
        let _ = writeln!(out, "  vertex_packed:     {}", self.vertex_packed);
        let _ = writeln!(out, "  fragment_packed:   {}", self.fragment_packed);
        let _ = writeln!(out, "  compute_packed:    {}", self.compute_packed);
        let _ = writeln!(out, "  pack_bytes:        {}", self.pack_bytes);
        if !self.failures.is_empty() {
            let _ = writeln!(out, "  failures ({}):", self.failures.len());
            for failure in &self.failures {
                let id_str = failure
                    .code_identity
                    .map(format_shader_hash)
                    .unwrap_or_else(|| "<unknown>".to_string());
                let stage_str = failure
                    .stage
                    .map(ShaderStage::as_str)
                    .unwrap_or("unknown");
                let _ = writeln!(out, "    - {id_str} ({stage_str}): {}", failure.reason);
            }
        }
        out
    }
}

/// Complete output of [`pack_rdna2_blobs`], [`pack_rdna2_directory`], or [`pack_rdna2_manifest`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2PackResult {
    /// Serialized `WGPK` v1 binary `.wgsl.pack` bytes.
    pub pack_bytes: Vec<u8>,
    /// Packed WGSL entries in deterministic ascending hash order.
    pub entries: Vec<PackEntry>,
    /// Strictly sanitized summary metrics.
    pub summary: Rdna2PackSummary,
}

/// Errors produced when packing RDNA2 shader directories, manifests, or blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rdna2PackError {
    InvalidInputDir(String),
    InvalidManifest(String),
    HashMismatch {
        expected: u64,
        actual: u64,
        stage: ShaderStage,
    },
    ShaderFailed {
        code_identity: Option<u64>,
        stage: Option<ShaderStage>,
        reason: String,
    },
    Io(String),
    Pack(PackError),
}

impl fmt::Display for Rdna2PackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInputDir(msg) => write!(f, "invalid RDNA2 input directory: {msg}"),
            Self::InvalidManifest(msg) => write!(f, "invalid pipeline-dump manifest: {msg}"),
            Self::HashMismatch {
                expected,
                actual,
                stage,
            } => write!(
                f,
                "RDNA2 {} shader FNV-1a 64 hash mismatch: expected {}, actual {}",
                stage.as_str(),
                format_shader_hash(*expected),
                format_shader_hash(*actual)
            ),
            Self::ShaderFailed {
                code_identity,
                stage,
                reason,
            } => {
                let id_str = code_identity
                    .map(format_shader_hash)
                    .unwrap_or_else(|| "<unknown>".to_string());
                let stage_str = stage.map(ShaderStage::as_str).unwrap_or("unknown");
                write!(f, "RDNA2 shader {id_str} ({stage_str}) failed: {reason}")
            }
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

fn stage_to_override(stage: ShaderStage) -> Rdna2StageOverride {
    match stage {
        ShaderStage::Vertex => Rdna2StageOverride::Vertex,
        ShaderStage::Fragment => Rdna2StageOverride::Pixel,
        ShaderStage::Compute => Rdna2StageOverride::Compute,
    }
}

fn override_to_stage(stage: Rdna2StageOverride) -> Option<ShaderStage> {
    match stage {
        Rdna2StageOverride::Auto => None,
        Rdna2StageOverride::Vertex => Some(ShaderStage::Vertex),
        Rdna2StageOverride::Pixel => Some(ShaderStage::Fragment),
        Rdna2StageOverride::Compute => Some(ShaderStage::Compute),
    }
}

fn parse_manifest_stage(value: &str) -> Option<ShaderStage> {
    if value.eq_ignore_ascii_case("vertex") {
        Some(ShaderStage::Vertex)
    } else if value.eq_ignore_ascii_case("pixel") || value.eq_ignore_ascii_case("fragment") {
        Some(ShaderStage::Fragment)
    } else if value.eq_ignore_ascii_case("compute") {
        Some(ShaderStage::Compute)
    } else {
        None
    }
}

fn parse_manifest_hex_u64(token: &str) -> Option<u64> {
    let hex = token
        .strip_prefix("0x")
        .or_else(|| token.strip_prefix("0X"))?;
    if hex.is_empty() || hex.len() > 16 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(hex, 16).ok()
}

/// Validate that a `code_file` path from `manifest.json` is strictly relative and contains no
/// traversal (`..`), root/drive prefixes, or NUL bytes, returning a normalized relative [`PathBuf`].
fn validate_manifest_relative_path(raw: &str) -> Result<PathBuf, Rdna2PackError> {
    if raw.is_empty() || raw.bytes().any(|b| b == 0) {
        return Err(Rdna2PackError::InvalidManifest(
            "native_stages entry has empty or invalid code_file".to_string(),
        ));
    }
    // Reject explicit leading slashes or Windows drive prefixes like `C:...` on all platforms.
    if raw.starts_with('/') || raw.starts_with('\\') {
        return Err(Rdna2PackError::InvalidManifest(
            "native_stages code_file must be a relative path".to_string(),
        ));
    }
    if raw.len() >= 2 && raw.as_bytes()[0].is_ascii_alphabetic() && raw.as_bytes()[1] == b':' {
        return Err(Rdna2PackError::InvalidManifest(
            "native_stages code_file must not contain a drive prefix".to_string(),
        ));
    }

    let mut rel = PathBuf::new();
    let mut normal_segments = 0usize;
    for segment in raw.split(['/', '\\']) {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".." {
            return Err(Rdna2PackError::InvalidManifest(
                "native_stages code_file must not contain parent directory traversal ('..')"
                    .to_string(),
            ));
        }
        rel.push(segment);
        normal_segments += 1;
    }
    if normal_segments == 0 {
        return Err(Rdna2PackError::InvalidManifest(
            "native_stages code_file must name a file".to_string(),
        ));
    }
    for comp in rel.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(Rdna2PackError::InvalidManifest(
                    "native_stages code_file must remain within the manifest directory".to_string(),
                ));
            }
        }
    }
    Ok(rel)
}

// ---------------------------------------------------------------------------
// Minimal zero-dependency RFC 8259 JSON parser (for `manifest.json`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            Self::Object(entries) => Some(entries),
            _ => None,
        }
    }

    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Number(raw) => raw.parse::<u64>().ok(),
            _ => None,
        }
    }

    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()?
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        let trimmed = input.strip_prefix('\u{feff}').unwrap_or(input);
        Self {
            bytes: trimmed.as_bytes(),
            pos: 0,
        }
    }

    fn parse_root(mut self) -> Result<JsonValue, String> {
        self.skip_ws();
        let val = self.parse_value(0)?;
        self.skip_ws();
        if self.pos != self.bytes.len() {
            return Err(format!("unexpected trailing character at byte {}", self.pos));
        }
        Ok(val)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect_bytes(&mut self, lit: &[u8]) -> Result<(), String> {
        let end = self.pos.saturating_add(lit.len());
        if self.bytes.get(self.pos..end) == Some(lit) {
            self.pos = end;
            Ok(())
        } else {
            Err(format!("unexpected token at byte {}", self.pos))
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, String> {
        if depth > MAX_JSON_DEPTH {
            return Err("JSON nesting depth exceeds safety limit".to_string());
        }
        self.skip_ws();
        let b = self
            .peek()
            .ok_or_else(|| format!("unexpected end of JSON at byte {}", self.pos))?;
        match b {
            b'{' => self.parse_object(depth + 1),
            b'[' => self.parse_array(depth + 1),
            b'"' => self.parse_string().map(JsonValue::String),
            b't' => {
                self.expect_bytes(b"true")?;
                Ok(JsonValue::Bool(true))
            }
            b'f' => {
                self.expect_bytes(b"false")?;
                Ok(JsonValue::Bool(false))
            }
            b'n' => {
                self.expect_bytes(b"null")?;
                Ok(JsonValue::Null)
            }
            b'-' | b'0'..=b'9' => self.parse_number().map(JsonValue::Number),
            _ => Err(format!("unexpected byte 0x{b:02x} at byte {}", self.pos)),
        }
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.pos += 1; // '{'
        self.skip_ws();
        let mut entries = Vec::new();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(format!("expected object key string at byte {}", self.pos));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(format!("expected ':' after object key at byte {}", self.pos));
            }
            self.pos += 1;
            let value = self.parse_value(depth)?;
            entries.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(entries));
                }
                _ => {
                    return Err(format!(
                        "expected ',' or '}}' in object at byte {}",
                        self.pos
                    ));
                }
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, String> {
        self.pos += 1; // '['
        self.skip_ws();
        let mut items = Vec::new();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            let value = self.parse_value(depth)?;
            items.push(value);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b']') => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => {
                    return Err(format!(
                        "expected ',' or ']' in array at byte {}",
                        self.pos
                    ));
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, String> {
        let end = self.pos.saturating_add(4);
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or_else(|| "truncated \\uXXXX escape in JSON string".to_string())?;
        let text = std::str::from_utf8(slice)
            .map_err(|_| "invalid UTF-8 in \\uXXXX escape".to_string())?;
        let code = u16::from_str_radix(text, 16)
            .map_err(|_| format!("invalid hex digits in \\u escape at byte {}", self.pos))?;
        self.pos = end;
        Ok(code)
    }

    fn parse_string(&mut self) -> Result<String, String> {
        if self.peek() != Some(b'"') {
            return Err(format!("expected '\"' at byte {}", self.pos));
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(b) = self.peek() {
            if b == b'"' {
                self.pos += 1;
                return Ok(out);
            }
            if b == b'\\' {
                self.pos += 1;
                let esc = self
                    .peek()
                    .ok_or_else(|| "unterminated escape sequence in JSON string".to_string())?;
                self.pos += 1;
                match esc {
                    b'"' => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/' => out.push('/'),
                    b'b' => out.push('\u{0008}'),
                    b'f' => out.push('\u{000c}'),
                    b'n' => out.push('\n'),
                    b'r' => out.push('\r'),
                    b't' => out.push('\t'),
                    b'u' => {
                        let first = self.parse_hex4()?;
                        if (0xD800..=0xDBFF).contains(&first) {
                            self.expect_bytes(b"\\u")?;
                            let second = self.parse_hex4()?;
                            if !(0xDC00..=0xDFFF).contains(&second) {
                                return Err("invalid UTF-16 low surrogate in JSON string".to_string());
                            }
                            let cp = 0x10000
                                + (((u32::from(first) - 0xD800) << 10)
                                    | (u32::from(second) - 0xDC00));
                            let ch = char::from_u32(cp)
                                .ok_or_else(|| "invalid Unicode scalar in JSON string".to_string())?;
                            out.push(ch);
                        } else if (0xDC00..=0xDFFF).contains(&first) {
                            return Err("unpaired UTF-16 low surrogate in JSON string".to_string());
                        } else {
                            let ch = char::from_u32(u32::from(first))
                                .ok_or_else(|| "invalid Unicode scalar in JSON string".to_string())?;
                            out.push(ch);
                        }
                    }
                    _ => {
                        return Err(format!(
                            "invalid escape sequence '\\{}' at byte {}",
                            esc as char, self.pos
                        ));
                    }
                }
                continue;
            }
            if b < 0x20 {
                return Err(format!("unescaped control character in JSON string at byte {}", self.pos));
            }
            // Copy contiguous safe UTF-8 bytes up to the next '"' or '\' or control char.
            let start = self.pos;
            while let Some(next) = self.peek() {
                if next == b'"' || next == b'\\' || next < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            let chunk = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| "invalid UTF-8 inside JSON string".to_string())?;
            out.push_str(chunk);
        }
        Err("unterminated JSON string".to_string())
    }

    fn parse_number(&mut self) -> Result<String, String> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
            }
            Some(b'1'..=b'9') => {
                self.pos += 1;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(format!("invalid JSON number at byte {}", self.pos)),
        }
        if self.peek() == Some(b'.') {
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("invalid fraction in JSON number at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(format!("invalid exponent in JSON number at byte {}", self.pos));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "invalid UTF-8 in JSON number".to_string())?;
        Ok(text.to_string())
    }
}

/// Parse a `playgta6-pipeline-dump` `manifest.json` document (schema version 1) with zero
/// external dependencies.
///
/// Validates `"schema": "playgta6-pipeline-dump"`, `"schema_version": 1` (if present), and
/// extracts every entry in `"native_stages"` (`code_identity`, `stage`, `code_file`, and optional
/// `identity` / `code_bytes`).
pub fn parse_pipeline_dump_manifest(
    json_text: &str,
) -> Result<PipelineDumpManifest, Rdna2PackError> {
    let root = JsonParser::new(json_text)
        .parse_root()
        .map_err(Rdna2PackError::InvalidManifest)?;

    if root.as_object().is_none() {
        return Err(Rdna2PackError::InvalidManifest(
            "manifest root must be a JSON object".to_string(),
        ));
    }

    let schema = root
        .get("schema")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            Rdna2PackError::InvalidManifest(
                "missing string field \"schema\" (expected \"playgta6-pipeline-dump\")".to_string(),
            )
        })?;
    if schema != PIPELINE_DUMP_SCHEMA {
        return Err(Rdna2PackError::InvalidManifest(format!(
            "unsupported manifest schema '{schema}' (expected '{PIPELINE_DUMP_SCHEMA}')"
        )));
    }

    let schema_version = match root.get("schema_version") {
        Some(v) => {
            let ver = v.as_u64().ok_or_else(|| {
                Rdna2PackError::InvalidManifest(
                    "\"schema_version\" must be an unsigned integer".to_string(),
                )
            })?;
            if ver != PIPELINE_DUMP_SCHEMA_VERSION {
                return Err(Rdna2PackError::InvalidManifest(format!(
                    "unsupported manifest schema_version {ver} (expected {PIPELINE_DUMP_SCHEMA_VERSION})"
                )));
            }
            ver
        }
        None => PIPELINE_DUMP_SCHEMA_VERSION,
    };

    let native_stages_arr = root
        .get("native_stages")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            Rdna2PackError::InvalidManifest("missing array field \"native_stages\"".to_string())
        })?;

    let mut native_stages = Vec::with_capacity(native_stages_arr.len());
    for (idx, item) in native_stages_arr.iter().enumerate() {
        if item.as_object().is_none() {
            return Err(Rdna2PackError::InvalidManifest(format!(
                "native_stages[{idx}] must be a JSON object"
            )));
        }

        let code_id_raw = item
            .get("code_identity")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] missing string field \"code_identity\""
                ))
            })?;
        let code_identity = parse_manifest_hex_u64(code_id_raw).ok_or_else(|| {
            Rdna2PackError::InvalidManifest(format!(
                "native_stages[{idx}] has invalid hex code_identity '{code_id_raw}'"
            ))
        })?;

        let identity = match item.get("identity").and_then(JsonValue::as_str) {
            Some(s) if !s.is_empty() => Some(parse_manifest_hex_u64(s).ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] has invalid hex identity '{s}'"
                ))
            })?),
            _ => None,
        };

        let stage_raw = item
            .get("stage")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] missing string field \"stage\""
                ))
            })?;
        let stage = parse_manifest_stage(stage_raw).ok_or_else(|| {
            Rdna2PackError::InvalidManifest(format!(
                "native_stages[{idx}] has unsupported stage '{stage_raw}' (expected 'vertex', 'pixel', or 'compute')"
            ))
        })?;

        let code_file = item
            .get("code_file")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] missing string field \"code_file\""
                ))
            })?;
        let _ = validate_manifest_relative_path(code_file)?;

        let code_bytes = match item.get("code_bytes") {
            Some(v) => Some(v.as_u64().ok_or_else(|| {
                Rdna2PackError::InvalidManifest(format!(
                    "native_stages[{idx}] field \"code_bytes\" must be an unsigned integer"
                ))
            })?),
            None => None,
        };

        native_stages.push(PipelineDumpNativeStage {
            code_identity,
            identity,
            stage,
            code_file: code_file.to_string(),
            code_bytes,
        });
    }

    Ok(PipelineDumpManifest {
        schema: schema.to_string(),
        schema_version,
        native_stages,
    })
}

fn build_pack_result(
    packed_by_hash: BTreeMap<u64, (PackEntry, Rdna2PackEntrySummary)>,
    total_blobs: usize,
    skipped_or_failed: usize,
    failed: usize,
    manifest_duplicates_skipped: usize,
    failures: Vec<Rdna2PackFailureSummary>,
    pack_flags: u32,
) -> Result<Rdna2PackResult, Rdna2PackError> {
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

    let pack_bytes = write_pack(&entries, pack_flags).map_err(Rdna2PackError::Pack)?;
    let summary = Rdna2PackSummary {
        total_blobs,
        translated_ok: entries.len(),
        skipped_or_failed,
        failed,
        manifest_duplicates_skipped,
        vertex_count,
        pixel_count,
        compute_count,
        vertex_packed: vertex_count,
        fragment_packed: pixel_count,
        compute_packed: compute_count,
        pack_bytes: pack_bytes.len(),
        entries: entry_summaries,
        failures,
    };

    Ok(Rdna2PackResult {
        pack_bytes,
        entries,
        summary,
    })
}

fn write_pack_bytes_to_file(out_path: &Path, pack_bytes: &[u8]) -> Result<(), Rdna2PackError> {
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
    fs::write(out_path, pack_bytes).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to write WGPK pack '{out_label}': {}",
            err.kind()
        ))
    })?;
    Ok(())
}

fn pack_blobs_into_map(
    blobs: &[Rdna2BlobInput],
    options: &Rdna2PackOptions,
    packed_by_hash: &mut BTreeMap<u64, (PackEntry, Rdna2PackEntrySummary)>,
    skipped_or_failed: &mut usize,
    failed: &mut usize,
    failures: &mut Vec<Rdna2PackFailureSummary>,
) {
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
        let hint = parse_filename_hint(&blob.name);
        let stage = if blob.stage_override != Rdna2StageOverride::Auto {
            blob.stage_override
        } else if hint.stage != Rdna2StageOverride::Auto {
            hint.stage
        } else {
            options.translate.stage
        };

        if blob.bytes.len() < MIN_RDNA2_BLOB_BYTES {
            *skipped_or_failed += 1;
            *failed += 1;
            failures.push(Rdna2PackFailureSummary {
                code_identity: blob.hash_override.or(hint.hash),
                stage: override_to_stage(stage),
                reason: format!(
                    "RDNA2 blob too short ({} bytes, minimum {MIN_RDNA2_BLOB_BYTES})",
                    blob.bytes.len()
                ),
            });
            continue;
        }

        let translate_opts = Rdna2TranslateOptions {
            stage,
            entry_point: options.translate.entry_point.clone(),
            min_cbv_vec4_count: options.translate.min_cbv_vec4_count,
            workgroup_size: options.translate.workgroup_size,
            workgroup_id_sgpr_base: options.translate.workgroup_id_sgpr_base,
        };

        let output = match translate_rdna2_to_wgsl(&blob.bytes, &translate_opts) {
            Ok(out) if out.naga_validated => out,
            Ok(_) => {
                *skipped_or_failed += 1;
                *failed += 1;
                failures.push(Rdna2PackFailureSummary {
                    code_identity: blob.hash_override.or(hint.hash).or_else(|| Some(fnv1a_64(&blob.bytes))),
                    stage: override_to_stage(stage),
                    reason: "translated WGSL did not pass Naga validation".to_string(),
                });
                continue;
            }
            Err(err) => {
                *skipped_or_failed += 1;
                *failed += 1;
                failures.push(Rdna2PackFailureSummary {
                    code_identity: blob.hash_override.or(hint.hash).or_else(|| Some(fnv1a_64(&blob.bytes))),
                    stage: override_to_stage(stage),
                    reason: format!("RDNA2->WGSL translation failed: {err}"),
                });
                continue;
            }
        };

        let final_hash = blob
            .hash_override
            .or(hint.hash)
            .unwrap_or(output.hash_fnv1a64);

        if packed_by_hash.contains_key(&final_hash) {
            *skipped_or_failed += 1;
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
                    naga_validated: output.naga_validated,
                    translated_op_count: output.translated_op_count,
                    unsupported_op_count: output.unsupported_op_count,
                    wgsl_byte_len,
                    bindings: output.bindings,
                },
            ),
        );
    }
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
    let mut failed = 0usize;
    let mut failures = Vec::new();

    pack_blobs_into_map(
        blobs,
        options,
        &mut packed_by_hash,
        &mut skipped_or_failed,
        &mut failed,
        &mut failures,
    );

    build_pack_result(
        packed_by_hash,
        blobs.len(),
        skipped_or_failed,
        failed,
        0,
        failures,
        options.pack_flags,
    )
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

fn collect_directory_blobs(
    input_dir: &Path,
) -> Result<(Vec<Rdna2BlobInput>, usize), Rdna2PackError> {
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

    Ok((blobs, read_failures))
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
    let (blobs, read_failures) = collect_directory_blobs(input_dir)?;

    let mut result = pack_rdna2_blobs(&blobs, options)?;
    if read_failures > 0 {
        result.summary.total_blobs += read_failures;
        result.summary.skipped_or_failed += read_failures;
        result.summary.failed += read_failures;
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
    write_pack_bytes_to_file(out_path, &result.pack_bytes)?;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn pack_manifest_into_map(
    manifest_path: &Path,
    options: &Rdna2PackOptions,
    packed_by_hash: &mut BTreeMap<u64, (PackEntry, Rdna2PackEntrySummary)>,
    total_blobs: &mut usize,
    skipped_or_failed: &mut usize,
    failed: &mut usize,
    manifest_duplicates_skipped: &mut usize,
    failures: &mut Vec<Rdna2PackFailureSummary>,
) -> Result<(), Rdna2PackError> {
    let manifest_label = sanitize_source_path(manifest_path, None);
    let meta = fs::metadata(manifest_path).map_err(|_| {
        Rdna2PackError::InvalidManifest(format!(
            "manifest '{}' does not exist or is unreadable",
            manifest_label
        ))
    })?;
    if !meta.is_file() {
        return Err(Rdna2PackError::InvalidManifest(format!(
            "'{}' is not a regular file",
            manifest_label
        )));
    }

    let json_text = fs::read_to_string(manifest_path).map_err(|err| {
        Rdna2PackError::Io(format!(
            "failed to read manifest '{manifest_label}': {}",
            err.kind()
        ))
    })?;
    let manifest = parse_pipeline_dump_manifest(&json_text)?;

    let parent_dir = match manifest_path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let canonical_parent = fs::canonicalize(parent_dir).map_err(|_| {
        Rdna2PackError::Io(format!(
            "failed to resolve parent directory of manifest '{manifest_label}'"
        ))
    })?;

    // Deduplicate native_stages by `code_identity` (multiple PSO metadata blobs may share one
    // code blob in `playgta6-pipeline-dump`) and sort deterministically by ascending code_identity.
    let mut seen_code_ids: BTreeSet<u64> = BTreeSet::new();
    let mut unique_stages: Vec<PipelineDumpNativeStage> =
        Vec::with_capacity(manifest.native_stages.len());
    for entry in manifest.native_stages {
        if !seen_code_ids.insert(entry.code_identity) {
            *manifest_duplicates_skipped += 1;
            continue;
        }
        unique_stages.push(entry);
    }
    unique_stages.sort_by_key(|entry| entry.code_identity);

    *total_blobs += unique_stages.len();

    for entry in unique_stages {
        let rel_path = match validate_manifest_relative_path(&entry.code_file) {
            Ok(p) => p,
            Err(err) => {
                if options.best_effort {
                    *skipped_or_failed += 1;
                    *failed += 1;
                    failures.push(Rdna2PackFailureSummary {
                        code_identity: Some(entry.code_identity),
                        stage: Some(entry.stage),
                        reason: err.to_string(),
                    });
                    continue;
                }
                return Err(err);
            }
        };

        let joined = canonical_parent.join(&rel_path);
        let canonical_file = match fs::canonicalize(&joined) {
            Ok(target) => target,
            Err(err) => {
                let reason = format!(
                    "failed to locate code_file '{}': {}",
                    sanitize_source_path(&rel_path, None),
                    err.kind()
                );
                if options.best_effort {
                    *skipped_or_failed += 1;
                    *failed += 1;
                    failures.push(Rdna2PackFailureSummary {
                        code_identity: Some(entry.code_identity),
                        stage: Some(entry.stage),
                        reason,
                    });
                    continue;
                }
                return Err(Rdna2PackError::ShaderFailed {
                    code_identity: Some(entry.code_identity),
                    stage: Some(entry.stage),
                    reason,
                });
            }
        };

        if !canonical_file.starts_with(&canonical_parent) {
            let reason = "code_file resolves outside the manifest directory".to_string();
            if options.best_effort {
                *skipped_or_failed += 1;
                *failed += 1;
                failures.push(Rdna2PackFailureSummary {
                    code_identity: Some(entry.code_identity),
                    stage: Some(entry.stage),
                    reason,
                });
                continue;
            }
            return Err(Rdna2PackError::ShaderFailed {
                code_identity: Some(entry.code_identity),
                stage: Some(entry.stage),
                reason,
            });
        }

        let file_meta = match fs::metadata(&canonical_file) {
            Ok(m) if m.is_file() => m,
            _ => {
                let reason = "code_file is not a regular file".to_string();
                if options.best_effort {
                    *skipped_or_failed += 1;
                    *failed += 1;
                    failures.push(Rdna2PackFailureSummary {
                        code_identity: Some(entry.code_identity),
                        stage: Some(entry.stage),
                        reason,
                    });
                    continue;
                }
                return Err(Rdna2PackError::ShaderFailed {
                    code_identity: Some(entry.code_identity),
                    stage: Some(entry.stage),
                    reason,
                });
            }
        };
        let _ = file_meta;

        let bytes = match fs::read(&canonical_file) {
            Ok(b) => b,
            Err(err) => {
                let reason = format!("failed to read code_file: {}", err.kind());
                if options.best_effort {
                    *skipped_or_failed += 1;
                    *failed += 1;
                    failures.push(Rdna2PackFailureSummary {
                        code_identity: Some(entry.code_identity),
                        stage: Some(entry.stage),
                        reason,
                    });
                    continue;
                }
                return Err(Rdna2PackError::ShaderFailed {
                    code_identity: Some(entry.code_identity),
                    stage: Some(entry.stage),
                    reason,
                });
            }
        };

        let actual_hash = fnv1a_64(&bytes);
        if actual_hash != entry.code_identity {
            if options.best_effort {
                *skipped_or_failed += 1;
                *failed += 1;
                failures.push(Rdna2PackFailureSummary {
                    code_identity: Some(entry.code_identity),
                    stage: Some(entry.stage),
                    reason: format!(
                        "FNV-1a 64 hash mismatch: expected {}, actual {}",
                        format_shader_hash(entry.code_identity),
                        format_shader_hash(actual_hash)
                    ),
                });
                continue;
            }
            return Err(Rdna2PackError::HashMismatch {
                expected: entry.code_identity,
                actual: actual_hash,
                stage: entry.stage,
            });
        }

        if bytes.len() < MIN_RDNA2_BLOB_BYTES {
            let reason = format!(
                "RDNA2 blob too short ({} bytes, minimum {MIN_RDNA2_BLOB_BYTES})",
                bytes.len()
            );
            if options.best_effort {
                *skipped_or_failed += 1;
                *failed += 1;
                failures.push(Rdna2PackFailureSummary {
                    code_identity: Some(entry.code_identity),
                    stage: Some(entry.stage),
                    reason,
                });
                continue;
            }
            return Err(Rdna2PackError::ShaderFailed {
                code_identity: Some(entry.code_identity),
                stage: Some(entry.stage),
                reason,
            });
        }

        let translate_opts = Rdna2TranslateOptions {
            stage: stage_to_override(entry.stage),
            entry_point: options.translate.entry_point.clone(),
            min_cbv_vec4_count: options.translate.min_cbv_vec4_count,
            workgroup_size: options.translate.workgroup_size,
            workgroup_id_sgpr_base: options.translate.workgroup_id_sgpr_base,
        };

        let output = match translate_rdna2_to_wgsl(&bytes, &translate_opts) {
            Ok(out) if out.naga_validated => out,
            Ok(_) => {
                let reason = "translated WGSL did not pass Naga validation".to_string();
                if options.best_effort {
                    *skipped_or_failed += 1;
                    *failed += 1;
                    failures.push(Rdna2PackFailureSummary {
                        code_identity: Some(entry.code_identity),
                        stage: Some(entry.stage),
                        reason,
                    });
                    continue;
                }
                return Err(Rdna2PackError::ShaderFailed {
                    code_identity: Some(entry.code_identity),
                    stage: Some(entry.stage),
                    reason,
                });
            }
            Err(err) => {
                let reason = format!("RDNA2->WGSL translation failed: {err}");
                if options.best_effort {
                    *skipped_or_failed += 1;
                    *failed += 1;
                    failures.push(Rdna2PackFailureSummary {
                        code_identity: Some(entry.code_identity),
                        stage: Some(entry.stage),
                        reason,
                    });
                    continue;
                }
                return Err(Rdna2PackError::ShaderFailed {
                    code_identity: Some(entry.code_identity),
                    stage: Some(entry.stage),
                    reason,
                });
            }
        };

        let wgsl_byte_len = output.wgsl.len();
        packed_by_hash.insert(
            entry.code_identity,
            (
                PackEntry {
                    hash: entry.code_identity,
                    stage: output.stage,
                    wgsl: output.wgsl,
                },
                Rdna2PackEntrySummary {
                    hash_fnv1a64: entry.code_identity,
                    stage: output.stage,
                    entry_point: output.entry_point,
                    naga_validated: output.naga_validated,
                    translated_op_count: output.translated_op_count,
                    unsupported_op_count: output.unsupported_op_count,
                    wgsl_byte_len,
                    bindings: output.bindings,
                },
            ),
        );
    }

    Ok(())
}

/// Pack RDNA2 shaders directly from a `playgta6-pipeline-dump` `manifest.json` (`native_stages`).
///
/// Deduplicates entries by `code_identity`, verifies whole-file FNV-1a 64 hashes against
/// `code_identity`, passes explicit `ShaderStage` (`Vertex` / `Fragment` / `Compute`) into
/// [`translate_rdna2_to_wgsl`], and builds a deterministic `WGPK` v1 `.wgsl.pack` archive.
///
/// In strict mode (`options.best_effort == false`), any missing file, escaping symlink, FNV-1a
/// mismatch, short blob, or translation/Naga failure returns an error. When
/// `options.best_effort == true`, individual failing shaders are skipped and recorded in
/// `summary.failures`.
pub fn pack_rdna2_manifest(
    manifest_path: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    pack_rdna2_sources(None, Some(manifest_path), options)
}

/// Convenience helper that runs [`pack_rdna2_manifest`] and writes the resulting
/// `WGPK` v1 `.wgsl.pack` bytes to `out_path` (creating parent directories if needed).
pub fn pack_rdna2_manifest_to_file(
    manifest_path: &Path,
    out_path: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    pack_rdna2_sources_to_file(None, Some(manifest_path), out_path, options)
}

/// Pack RDNA2 shaders from an optional `playgta6-pipeline-dump` `manifest.json`, an optional
/// raw blob directory, or both combined (manifest entries take precedence on matching hashes).
pub fn pack_rdna2_sources(
    input_dir: Option<&Path>,
    manifest_path: Option<&Path>,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    if input_dir.is_none() && manifest_path.is_none() {
        return Err(Rdna2PackError::InvalidInputDir(
            "at least one input directory or manifest is required".to_string(),
        ));
    }

    let mut packed_by_hash: BTreeMap<u64, (PackEntry, Rdna2PackEntrySummary)> = BTreeMap::new();
    let mut total_blobs = 0usize;
    let mut skipped_or_failed = 0usize;
    let mut failed = 0usize;
    let mut manifest_duplicates_skipped = 0usize;
    let mut failures = Vec::new();

    if let Some(manifest) = manifest_path {
        pack_manifest_into_map(
            manifest,
            options,
            &mut packed_by_hash,
            &mut total_blobs,
            &mut skipped_or_failed,
            &mut failed,
            &mut manifest_duplicates_skipped,
            &mut failures,
        )?;
    }

    if let Some(dir) = input_dir {
        let (blobs, read_failures) = collect_directory_blobs(dir)?;
        total_blobs += blobs.len() + read_failures;
        skipped_or_failed += read_failures;
        failed += read_failures;
        pack_blobs_into_map(
            &blobs,
            options,
            &mut packed_by_hash,
            &mut skipped_or_failed,
            &mut failed,
            &mut failures,
        );
    }

    build_pack_result(
        packed_by_hash,
        total_blobs,
        skipped_or_failed,
        failed,
        manifest_duplicates_skipped,
        failures,
        options.pack_flags,
    )
}

/// Convenience helper that runs [`pack_rdna2_sources`] and writes the resulting
/// `WGPK` v1 `.wgsl.pack` bytes to `out_path` (creating parent directories if needed).
pub fn pack_rdna2_sources_to_file(
    input_dir: Option<&Path>,
    manifest_path: Option<&Path>,
    out_path: &Path,
    options: &Rdna2PackOptions,
) -> Result<Rdna2PackResult, Rdna2PackError> {
    let result = pack_rdna2_sources(input_dir, manifest_path, options)?;
    write_pack_bytes_to_file(out_path, &result.pack_bytes)?;
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
        assert!(json.contains("\"vertex_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"fragment_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"compute_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"failed\": 0"), "JSON:\n{json}");
        assert!(json.contains("\"naga_validated\": true"), "JSON:\n{json}");
    }

    #[test]
    fn parse_pipeline_dump_manifest_parses_stages_and_rejects_malformed_inputs() {
        let valid_json = r#"{
            "schema": "playgta6-pipeline-dump",
            "schema_version": 1,
            "native_stages": [
                {
                    "code_identity": "0x0123456789abcdef",
                    "identity": "0xfedcba9876543210",
                    "stage": "vertex",
                    "layout": "A",
                    "code_bytes": 64,
                    "metadata_bytes": 128,
                    "code_file": "native/0x0123456789abcdef.bin",
                    "metadata_file": "native/0x0123456789abcdef.meta"
                },
                {
                    "code_identity": "0x1111222233334444",
                    "stage": "pixel",
                    "code_file": "native/0x1111222233334444.bin"
                },
                {
                    "code_identity": "0x5555666677778888",
                    "stage": "compute",
                    "code_file": "native/0x5555666677778888.bin"
                }
            ],
            "dxil": [{"identity": "0x0000000000000001", "hash_digest": "", "kind": -1, "bytes": 0, "file": "dxil/0x0000000000000001.dxbc"}],
            "root_signatures": [],
            "pipelines": [{"index": 0, "vs": null, "ps": null, "depth_bias": -1, "slope": 1.25e-2, "rect_list": false}]
        }"#;

        let parsed =
            parse_pipeline_dump_manifest(valid_json).expect("valid manifest should parse cleanly");
        assert_eq!(parsed.schema, PIPELINE_DUMP_SCHEMA);
        assert_eq!(parsed.schema_version, 1);
        assert_eq!(parsed.native_stages.len(), 3);
        assert_eq!(parsed.native_stages[0].code_identity, 0x0123_4567_89ab_cdef);
        assert_eq!(parsed.native_stages[0].identity, Some(0xfedc_ba98_7654_3210));
        assert_eq!(parsed.native_stages[0].stage, ShaderStage::Vertex);
        assert_eq!(parsed.native_stages[0].code_file, "native/0x0123456789abcdef.bin");
        assert_eq!(parsed.native_stages[0].code_bytes, Some(64));
        assert_eq!(parsed.native_stages[1].stage, ShaderStage::Fragment);
        assert_eq!(parsed.native_stages[2].stage, ShaderStage::Compute);

        // Reject wrong schema, unsupported version, unknown stage, bad hex, path traversal, and invalid JSON.
        assert!(parse_pipeline_dump_manifest(r#"{"schema":"other","native_stages":[]}"#).is_err());
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","schema_version":2,"native_stages":[]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"0x10","stage":"geometry","code_file":"native/a.bin"}]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"not_hex","stage":"vertex","code_file":"native/a.bin"}]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"0x10","stage":"vertex","code_file":"../secret.bin"}]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"0x10","stage":"vertex","code_file":"/etc/passwd"}]}"#
            )
            .is_err()
        );
        assert!(
            parse_pipeline_dump_manifest(
                r#"{"schema":"playgta6-pipeline-dump","native_stages":[{"code_identity":"0x10","stage":"vertex","code_file":"C:\\Windows\\win.ini"}]}"#
            )
            .is_err()
        );
        assert!(parse_pipeline_dump_manifest("{truncated").is_err());
        // Deeply nested JSON must return a clean error rather than overflowing the stack.
        let deep = format!("{}{}", "[".repeat(128), "]".repeat(128));
        assert!(parse_pipeline_dump_manifest(&deep).is_err());
    }

    #[test]
    fn pack_rdna2_manifest_packs_vs_ps_cs_verifies_fnv1a_deduplicates_and_passes_naga() {
        let root = make_temp_dir("playgta6_rdna2_pack_manifest");
        let native_dir = root.join("native");
        fs::create_dir_all(&native_dir).unwrap();

        // 1) Synthetic VS (fullscreen triangle)
        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let vs_hash = fnv1a_64(&vs_bytes);
        let vs_id = format_shader_hash(vs_hash);

        // 2) Synthetic PS reading cb0[0..2] (48 B tight CBV) + exp mrt0 -> exercises min_binding_size_bytes
        let mut ps_words = Vec::new();
        ps_words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x00, false, false));
        ps_words.extend_from_slice(&enc_smem(0x08, 8, 0, 0x7D, 0x2C, false, false));
        ps_words.push(enc_vop1(0x01, 0, 4));
        ps_words.push(enc_vop1(0x01, 1, 8));
        ps_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        ps_words.push(enc_sopp(0x01, 0));
        let ps_bytes = words_to_bytes(&ps_words);
        let ps_hash = fnv1a_64(&ps_bytes);
        let ps_id = format_shader_hash(ps_hash);

        // 3) Synthetic Compute shader (s_barrier + s_endpgm)
        let cs_bytes = build_minimal_compute_rdna2();
        let cs_hash = fnv1a_64(&cs_bytes);
        let cs_id = format_shader_hash(cs_hash);

        // 4) Ambiguous bare shader (no EXP / no VINTRP / no s_barrier) declared explicitly as "vertex"
        //    in manifest.json -> proves explicit stage propagation with zero stage-inference errors!
        let bare_vs_bytes = words_to_bytes(&[
            enc_vop1(0x01, 0, 242), // v_mov_b32 v0, 1.0
            enc_sopp(0x01, 0),      // s_endpgm
        ]);
        let bare_vs_hash = fnv1a_64(&bare_vs_bytes);
        let bare_vs_id = format_shader_hash(bare_vs_hash);

        fs::write(native_dir.join(format!("{vs_id}.bin")), &vs_bytes).unwrap();
        fs::write(native_dir.join(format!("{ps_id}.bin")), &ps_bytes).unwrap();
        fs::write(native_dir.join(format!("{cs_id}.bin")), &cs_bytes).unwrap();
        fs::write(native_dir.join(format!("{bare_vs_id}.bin")), &bare_vs_bytes).unwrap();

        // Include a duplicate `code_identity` entry for `vs_id` (different `identity` & `.meta` file,
        // matching `PipelineDump::native_stage` when two metadata variants share one code blob).
        let manifest_json = format!(
            r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{ps_id}", "identity": "0xaaaaaaaaaaaaaaaa", "stage": "pixel", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{ps_id}.bin", "metadata_file": "native/{ps_id}.meta"}},
    {{"code_identity": "{vs_id}", "identity": "0xbbbbbbbbbbbbbbbb", "stage": "vertex", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{vs_id}.bin", "metadata_file": "native/{vs_id}.meta"}},
    {{"code_identity": "{vs_id}", "identity": "0xcccccccccccccccc", "stage": "vertex", "layout": "B", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{vs_id}.bin", "metadata_file": "native/{vs_id}-0xcccccccccccccccc.meta"}},
    {{"code_identity": "{cs_id}", "identity": "0xdddddddddddddddd", "stage": "compute", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{cs_id}.bin", "metadata_file": "native/{cs_id}.meta"}},
    {{"code_identity": "{bare_vs_id}", "identity": "0xeeeeeeeeeeeeeeee", "stage": "vertex", "layout": "A", "code_bytes": {}, "metadata_bytes": 0, "code_file": "native/{bare_vs_id}.bin", "metadata_file": "native/{bare_vs_id}.meta"}}
  ],
  "dxil": [],
  "root_signatures": [],
  "pipelines": []
}}"#,
            ps_bytes.len(),
            vs_bytes.len(),
            vs_bytes.len(),
            cs_bytes.len(),
            bare_vs_bytes.len()
        );
        let manifest_path = root.join("manifest.json");
        fs::write(&manifest_path, &manifest_json).unwrap();

        let result = pack_rdna2_manifest(&manifest_path, &Rdna2PackOptions::default())
            .expect("pack_rdna2_manifest should succeed");

        assert_eq!(result.summary.total_blobs, 4);
        assert_eq!(result.summary.translated_ok, 4);
        assert_eq!(result.summary.skipped_or_failed, 0);
        assert_eq!(result.summary.failed, 0);
        assert_eq!(result.summary.manifest_duplicates_skipped, 1);
        assert_eq!(result.summary.vertex_packed, 2);
        assert_eq!(result.summary.fragment_packed, 1);
        assert_eq!(result.summary.compute_packed, 1);
        assert_eq!(result.summary.vertex_count, 2);
        assert_eq!(result.summary.pixel_count, 1);
        assert_eq!(result.summary.compute_count, 1);
        assert!(result.summary.failures.is_empty());

        let decoded = read_pack(&result.pack_bytes).expect("WGPK pack must decode cleanly");
        assert_eq!(decoded.version, PACK_VERSION);
        assert_eq!(decoded.entries.len(), 4);
        for entry in &decoded.entries {
            assert_naga_wgsl_valid(&entry.wgsl);
        }

        let vs_packed = decoded.entries.iter().find(|e| e.hash == vs_hash).unwrap();
        assert_eq!(vs_packed.stage, ShaderStage::Vertex);
        assert!(vs_packed.wgsl.contains("fn vs_main("));

        let bare_vs_packed = decoded.entries.iter().find(|e| e.hash == bare_vs_hash).unwrap();
        assert_eq!(bare_vs_packed.stage, ShaderStage::Vertex);
        assert!(bare_vs_packed.wgsl.contains("fn vs_main("));

        let ps_packed = decoded.entries.iter().find(|e| e.hash == ps_hash).unwrap();
        assert_eq!(ps_packed.stage, ShaderStage::Fragment);
        assert!(ps_packed.wgsl.contains("fn fs_main("));

        let cs_packed = decoded.entries.iter().find(|e| e.hash == cs_hash).unwrap();
        assert_eq!(cs_packed.stage, ShaderStage::Compute);
        assert!(cs_packed.wgsl.contains("fn cs_main("));

        let json = result.summary.to_json();
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(json.contains("\"vertex_packed\": 2"), "JSON:\n{json}");
        assert!(json.contains("\"fragment_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"compute_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"failed\": 0"), "JSON:\n{json}");
        assert!(json.contains("\"manifest_duplicates_skipped\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"naga_validated\": true"), "JSON:\n{json}");
        assert!(json.contains("\"entry_point\": \"vs_main\""), "JSON:\n{json}");
        assert!(json.contains("\"entry_point\": \"fs_main\""), "JSON:\n{json}");
        assert!(json.contains("\"entry_point\": \"cs_main\""), "JSON:\n{json}");
        assert!(json.contains("\"min_binding_size_bytes\": 48"), "JSON:\n{json}");
        assert!(json.contains("\"failures\": []"), "JSON:\n{json}");
        assert!(!json.contains(&root_str), "JSON must not leak host path");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_manifest_strict_fails_on_bad_entries_and_best_effort_skips_cleanly() {
        let root = make_temp_dir("playgta6_rdna2_manifest_best_effort");
        let native_dir = root.join("native");
        fs::create_dir_all(&native_dir).unwrap();

        let vs_bytes = build_fullscreen_triangle_vs_rdna2();
        let vs_hash = fnv1a_64(&vs_bytes);
        let vs_id = format_shader_hash(vs_hash);
        fs::write(native_dir.join(format!("{vs_id}.bin")), &vs_bytes).unwrap();

        // Corrupt/tampered blob whose actual FNV-1a does NOT match declared code_identity.
        let bad_hash_id = "0xdeadbeef00000001";
        fs::write(native_dir.join(format!("{bad_hash_id}.bin")), build_minimal_compute_rdna2())
            .unwrap();

        // Truncated 2-byte blob whose FNV-1a matches its declared code_identity, but fails MIN_RDNA2_BLOB_BYTES.
        let short_bytes = [0x01u8, 0x02u8];
        let short_hash = fnv1a_64(&short_bytes);
        let short_id = format_shader_hash(short_hash);
        fs::write(native_dir.join(format!("{short_id}.bin")), short_bytes).unwrap();

        // Missing file entry.
        let missing_id = "0xdeadbeef00000002";

        // 1) Hash mismatch manifest -> strict returns HashMismatch, best-effort skips & reports.
        let mismatch_manifest = root.join("mismatch_manifest.json");
        fs::write(
            &mismatch_manifest,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{vs_id}", "stage": "vertex", "code_file": "native/{vs_id}.bin"}},
    {{"code_identity": "{bad_hash_id}", "stage": "compute", "code_file": "native/{bad_hash_id}.bin"}}
  ]
}}"#
            ),
        )
        .unwrap();

        let strict_err = pack_rdna2_manifest(&mismatch_manifest, &Rdna2PackOptions::default())
            .expect_err("strict manifest mode must fail on FNV-1a hash mismatch");
        assert!(matches!(
            strict_err,
            Rdna2PackError::HashMismatch {
                expected: 0xdead_beef_0000_0001,
                stage: ShaderStage::Compute,
                ..
            }
        ));

        // 2) Mixed manifest with 1 good VS + 3 bad entries (hash mismatch, truncated, missing).
        let mixed_manifest = root.join("mixed_manifest.json");
        fs::write(
            &mixed_manifest,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{vs_id}", "stage": "vertex", "code_file": "native/{vs_id}.bin"}},
    {{"code_identity": "{bad_hash_id}", "stage": "compute", "code_file": "native/{bad_hash_id}.bin"}},
    {{"code_identity": "{short_id}", "stage": "pixel", "code_file": "native/{short_id}.bin"}},
    {{"code_identity": "{missing_id}", "stage": "pixel", "code_file": "native/{missing_id}.bin"}}
  ]
}}"#
            ),
        )
        .unwrap();

        let best_effort_opts = Rdna2PackOptions {
            best_effort: true,
            ..Rdna2PackOptions::default()
        };
        let res = pack_rdna2_manifest(&mixed_manifest, &best_effort_opts)
            .expect("--best-effort manifest mode should pack valid shaders and skip bad ones");

        assert_eq!(res.summary.total_blobs, 4);
        assert_eq!(res.summary.translated_ok, 1);
        assert_eq!(res.summary.vertex_packed, 1);
        assert_eq!(res.summary.fragment_packed, 0);
        assert_eq!(res.summary.compute_packed, 0);
        assert_eq!(res.summary.failed, 3);
        assert_eq!(res.summary.skipped_or_failed, 3);
        assert_eq!(res.summary.failures.len(), 3);

        let decoded = read_pack(&res.pack_bytes).unwrap();
        assert_eq!(decoded.entries.len(), 1);
        assert_eq!(decoded.entries[0].hash, vs_hash);
        assert_naga_wgsl_valid(&decoded.entries[0].wgsl);

        let json = res.summary.to_json();
        let text = res.summary.to_text();
        let root_str = root.to_string_lossy().replace('\\', "/");
        assert!(json.contains("\"vertex_packed\": 1"), "JSON:\n{json}");
        assert!(json.contains("\"failed\": 3"), "JSON:\n{json}");
        assert!(json.contains("\"failures\": ["), "JSON:\n{json}");
        assert!(json.contains("FNV-1a 64 hash mismatch"), "JSON:\n{json}");
        assert!(!json.contains(&root_str), "JSON must never leak host path");
        assert!(!text.contains(&root_str), "text must never leak host path");

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn pack_rdna2_sources_combines_manifest_and_directory_with_manifest_precedence() {
        let root = make_temp_dir("playgta6_rdna2_sources_combined");
        let dump_dir = root.join("dump");
        let native_dir = dump_dir.join("native");
        let extra_dir = root.join("extra");
        fs::create_dir_all(&native_dir).unwrap();
        fs::create_dir_all(&extra_dir).unwrap();

        // Bare shader with no EXP/barrier: manifest explicitly labels it Vertex, while directory
        // scan without hints would infer Compute. Manifest MUST take precedence!
        let shared_bytes = words_to_bytes(&[enc_vop1(0x01, 0, 242), enc_sopp(0x01, 0)]);
        let shared_hash = fnv1a_64(&shared_bytes);
        let shared_id = format_shader_hash(shared_hash);
        fs::write(native_dir.join(format!("{shared_id}.bin")), &shared_bytes).unwrap();
        fs::write(extra_dir.join(format!("{shared_id}.bin")), &shared_bytes).unwrap();

        // Extra PS only in positional directory
        let ps_bytes = build_textured_splash_ps_rdna2();
        let ps_hash = fnv1a_64(&ps_bytes);
        fs::write(extra_dir.join("splash_ps.bin"), &ps_bytes).unwrap();

        let manifest_path = dump_dir.join("manifest.json");
        fs::write(
            &manifest_path,
            format!(
                r#"{{
  "schema": "playgta6-pipeline-dump",
  "schema_version": 1,
  "native_stages": [
    {{"code_identity": "{shared_id}", "stage": "vertex", "code_file": "native/{shared_id}.bin"}}
  ]
}}"#
            ),
        )
        .unwrap();

        let res = pack_rdna2_sources(
            Some(&extra_dir),
            Some(&manifest_path),
            &Rdna2PackOptions::default(),
        )
        .expect("combined manifest + directory pack should succeed");

        assert_eq!(res.summary.translated_ok, 2);
        assert_eq!(res.summary.vertex_packed, 1);
        assert_eq!(res.summary.fragment_packed, 1);
        assert_eq!(res.summary.compute_packed, 0);
        assert_eq!(res.summary.failed, 0);
        assert_eq!(res.summary.skipped_or_failed, 1); // duplicate hash from extra_dir skipped

        let decoded = read_pack(&res.pack_bytes).unwrap();
        let shared_entry = decoded.entries.iter().find(|e| e.hash == shared_hash).unwrap();
        assert_eq!(shared_entry.stage, ShaderStage::Vertex);
        assert!(shared_entry.wgsl.contains("fn vs_main("));
        let ps_entry = decoded.entries.iter().find(|e| e.hash == ps_hash).unwrap();
        assert_eq!(ps_entry.stage, ShaderStage::Fragment);
        assert!(ps_entry.wgsl.contains("fn fs_main("));

        let _ = fs::remove_dir_all(&root);
    }
}
