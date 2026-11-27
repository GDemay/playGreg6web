use std::fmt;

use naga::back::wgsl;
use naga::front::spv;
use naga::valid::{Capabilities, ValidationFlags, Validator};

use crate::compat::{
    CompatAudit, SanitizeOptions, ShaderAuditRecord, SpirvStreamError, audit_naga_module,
    sanitize_spirv_bytes,
};
use crate::lower::{LoweringReport, lower_module_for_webgpu};

/// SPIR-V little-endian magic number (`0x07230203`).
pub const SPIRV_MAGIC: u32 = 0x0723_0203;

/// Minimum SPIR-V header size in bytes (5 x 32-bit words).
pub const SPIRV_HEADER_BYTES: usize = 20;

const FNV1A_64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Maximum character length retained in a sanitized [`ConversionError::detail`].
const MAX_ERROR_DETAIL_CHARS: usize = 4096;

/// Maximum character length retained in a sanitized [`ConversionError::blocker_key`].
const MAX_BLOCKER_KEY_CHARS: usize = 160;

/// Maximum character length retained in each keyword in [`ConversionError::keywords`].
const MAX_KEYWORD_CHARS: usize = 128;

/// Maximum number of keywords attached to a single [`ConversionError`].
const MAX_KEYWORDS: usize = 32;

/// Placeholder emitted in place of redacted host filesystem paths in diagnostic strings.
const REDACTED_PATH_TOKEN: &str = "<path>";

/// WebGPU-compatible shader pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ShaderStage {
    Vertex = 1,
    Fragment = 2,
    Compute = 3,
}

impl ShaderStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Fragment => "fragment",
            Self::Compute => "compute",
        }
    }

    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Vertex),
            2 => Some(Self::Fragment),
            3 => Some(Self::Compute),
            _ => None,
        }
    }
}

impl fmt::Display for ShaderStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Options controlling SPIR-V compatibility pre-passes, Naga IR lowering, and WebGPU limit auditing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvertOptions {
    pub strip_float_controls: bool,
    pub strip_non_semantic: bool,
    pub audit_webgpu_limits: bool,
    pub lower_for_webgpu: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            strip_float_controls: true,
            strip_non_semantic: true,
            audit_webgpu_limits: true,
            lower_for_webgpu: true,
        }
    }
}

impl ConvertOptions {
    /// Disable SPIR-V sanitization rewrites while keeping WebGPU limit auditing and lowering enabled.
    pub const fn without_sanitization() -> Self {
        Self {
            strip_float_controls: false,
            strip_non_semantic: false,
            audit_webgpu_limits: true,
            lower_for_webgpu: true,
        }
    }

    /// Return a copy of these options with Naga IR WebGPU lowering disabled.
    pub const fn without_lowering(mut self) -> Self {
        self.lower_for_webgpu = false;
        self
    }

    pub const fn sanitize_options(self) -> SanitizeOptions {
        SanitizeOptions {
            strip_float_controls: self.strip_float_controls,
            strip_non_semantic: self.strip_non_semantic,
        }
    }
}

/// Successful SPIR-V -> WGSL lowering output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedShader {
    pub hash: u64,
    pub stage: ShaderStage,
    pub entry_point: String,
    pub wgsl: String,
    pub spirv_bytes: usize,
    pub audit: CompatAudit,
    pub lowering: LoweringReport,
}

/// High-level error bucket for batch blocker reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConversionErrorKind {
    InvalidSpirvHeader,
    SpirvParseError,
    ValidationError,
    WgslWriteError,
}

impl ConversionErrorKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidSpirvHeader => "InvalidSpirvHeader",
            Self::SpirvParseError => "SpirvParseError",
            Self::ValidationError => "ValidationError",
            Self::WgslWriteError => "WgslWriteError",
        }
    }
}

impl fmt::Display for ConversionErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Structured shader conversion error with normalized blocker category and extracted keywords.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionError {
    pub kind: ConversionErrorKind,
    pub blocker_key: String,
    pub keywords: Vec<String>,
    pub detail: String,
}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.kind, self.detail)
    }
}

impl std::error::Error for ConversionError {}

/// Compute a deterministic 64-bit FNV-1a hash over shader bytes.
pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = FNV1A_64_OFFSET_BASIS;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV1A_64_PRIME);
    }
    hash
}

/// Format a shader hash in canonical `0x` + 16 lowercase hex digits form.
pub fn format_shader_hash(hash: u64) -> String {
    format!("0x{hash:016x}")
}

/// Format the deterministic WGSL file name for a shader hash.
pub fn format_wgsl_filename(hash: u64) -> String {
    format!("{hash:016x}.wgsl")
}

/// Convert raw SPIR-V binary bytes into validated WebGPU-baseline WGSL using default browser
/// sanitization and WebGPU limit auditing options.
pub fn convert_spirv_bytes(spirv_bytes: &[u8]) -> Result<ConvertedShader, ConversionError> {
    convert_spirv_bytes_with_options(spirv_bytes, &ConvertOptions::default())
}

/// Convert raw SPIR-V binary bytes into validated WebGPU-baseline WGSL with explicit options.
pub fn convert_spirv_bytes_with_options(
    spirv_bytes: &[u8],
    options: &ConvertOptions,
) -> Result<ConvertedShader, ConversionError> {
    validate_spirv_header(spirv_bytes)?;

    let hash = fnv1a_64(spirv_bytes);
    let (sanitized_bytes, mut audit) =
        sanitize_spirv_bytes(spirv_bytes, &options.sanitize_options())
            .map_err(classify_stream_error)?;

    let spv_options = spv::Options {
        adjust_coordinate_space: true,
        strict_capabilities: true,
        block_ctx_dump_prefix: None,
    };

    let mut module =
        spv::parse_u8_slice(&sanitized_bytes, &spv_options).map_err(classify_spv_error)?;

    let (stage, entry_point) = detect_entry_point(&module)?;

    let lowering = if options.lower_for_webgpu {
        lower_module_for_webgpu(&mut module)
    } else {
        LoweringReport::default()
    };

    if options.audit_webgpu_limits {
        audit.merge(audit_naga_module(&module));
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let info = validator
        .validate(&module)
        .map_err(|err| classify_validation_error(&err.emit_to_string("")))?;

    let wgsl_source = normalize_wgsl_newlines(
        wgsl::write_string(&module, &info, wgsl::WriterFlags::empty())
            .map_err(classify_wgsl_error)?,
    );
    validate_emitted_wgsl(&wgsl_source)?;

    Ok(ConvertedShader {
        hash,
        stage,
        entry_point,
        wgsl: wgsl_source,
        spirv_bytes: spirv_bytes.len(),
        audit,
        lowering,
    })
}

/// Run the full SPIR-V pre-pass + Naga WebGPU limits audit + baseline lowering check for a shader.
pub fn audit_spirv_file(
    source_path: &str,
    spirv_bytes: &[u8],
    options: &ConvertOptions,
) -> ShaderAuditRecord {
    let safe_source_path = sanitize_audit_source_path(source_path);
    let (sanitized_bytes, mut audit) =
        match sanitize_spirv_bytes(spirv_bytes, &options.sanitize_options()) {
            Ok(pair) => pair,
            Err(err) => {
                return ShaderAuditRecord {
                    source_path: safe_source_path,
                    stage: None,
                    lowered_to_wgsl: false,
                    stream_error: Some(sanitize_diagnostic_text(&err.to_string())),
                    naga_error: None,
                    audit: CompatAudit::default(),
                };
            }
        };

    let spv_options = spv::Options {
        adjust_coordinate_space: true,
        strict_capabilities: true,
        block_ctx_dump_prefix: None,
    };

    let mut module = match spv::parse_u8_slice(&sanitized_bytes, &spv_options) {
        Ok(module) => module,
        Err(err) => {
            let conv_err = classify_spv_error(err);
            return ShaderAuditRecord {
                source_path: safe_source_path,
                stage: None,
                lowered_to_wgsl: false,
                stream_error: None,
                naga_error: Some(conv_err.to_string()),
                audit,
            };
        }
    };

    if options.lower_for_webgpu {
        let _ = lower_module_for_webgpu(&mut module);
    }

    if options.audit_webgpu_limits {
        audit.merge(audit_naga_module(&module));
    }

    let stage_name = module
        .entry_points
        .first()
        .map(|ep| format!("{:?}", ep.stage).to_ascii_lowercase());

    if let Err(err) = detect_entry_point(&module) {
        return ShaderAuditRecord {
            source_path: safe_source_path,
            stage: stage_name,
            lowered_to_wgsl: false,
            stream_error: None,
            naga_error: Some(err.to_string()),
            audit,
        };
    }

    let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
    let info = match validator.validate(&module) {
        Ok(info) => info,
        Err(err) => {
            let conv_err = classify_validation_error(&err.emit_to_string(""));
            return ShaderAuditRecord {
                source_path: safe_source_path,
                stage: stage_name,
                lowered_to_wgsl: false,
                stream_error: None,
                naga_error: Some(conv_err.to_string()),
                audit,
            };
        }
    };

    match wgsl::write_string(&module, &info, wgsl::WriterFlags::empty()) {
        Ok(wgsl_source) => {
            let normalized = normalize_wgsl_newlines(wgsl_source);
            match validate_emitted_wgsl(&normalized) {
                Ok(()) => ShaderAuditRecord {
                    source_path: safe_source_path,
                    stage: stage_name,
                    lowered_to_wgsl: true,
                    stream_error: None,
                    naga_error: None,
                    audit,
                },
                Err(conv_err) => ShaderAuditRecord {
                    source_path: safe_source_path,
                    stage: stage_name,
                    lowered_to_wgsl: false,
                    stream_error: None,
                    naga_error: Some(conv_err.to_string()),
                    audit,
                },
            }
        }
        Err(err) => {
            let conv_err = classify_wgsl_error(err);
            ShaderAuditRecord {
                source_path: safe_source_path,
                stage: stage_name,
                lowered_to_wgsl: false,
                stream_error: None,
                naga_error: Some(conv_err.to_string()),
                audit,
            }
        }
    }
}

/// Normalize emitted WGSL line endings to `\n`, strip Naga's leading/trailing horizontal
/// indentation, and drop blank lines while preserving intra-line spacing and comments.
fn normalize_wgsl_newlines(wgsl: String) -> String {
    crate::rdna2_to_wgsl::compact_wgsl_whitespace(&wgsl)
}

/// Validate that emitted WGSL is non-empty, lexically well-formed (balanced braces and closed
/// comments), and contains zero illegal module-scope `let` declarations.
///
/// In WGSL, `let` is a reserved keyword valid only inside function/block scope (`brace_depth > 0`);
/// module-scope constants/globals must use `const`, `override`, or `var`.
pub fn validate_emitted_wgsl(wgsl: &str) -> Result<(), ConversionError> {
    let bytes = wgsl.as_bytes();
    let len = bytes.len();
    let mut i = 0usize;
    let mut line = 1usize;
    let mut brace_depth = 0usize;
    let mut block_comment_depth = 0usize;
    let mut has_code_token = false;

    while i < len {
        let b = bytes[i];

        if b == b'\r' {
            if i + 1 < len && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            line += 1;
            continue;
        }
        if b == b'\n' {
            line += 1;
            i += 1;
            continue;
        }

        // Handle nested WGSL block comments (`/* ... /* ... */ ... */`).
        if block_comment_depth > 0 {
            if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
                block_comment_depth += 1;
                i += 2;
            } else if b == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                block_comment_depth -= 1;
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }

        // Enter line comment (`// ...`).
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            i += 2;
            while i < len && bytes[i] != b'\n' && bytes[i] != b'\r' {
                i += 1;
            }
            continue;
        }

        // Enter block comment (`/* ... */`).
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            block_comment_depth = 1;
            i += 2;
            continue;
        }

        if b.is_ascii_whitespace() {
            i += 1;
            continue;
        }

        has_code_token = true;

        match b {
            b'{' => {
                brace_depth += 1;
                i += 1;
            }
            b'}' => {
                if brace_depth == 0 {
                    return Err(sanitize_conversion_error(ConversionError {
                        kind: ConversionErrorKind::WgslWriteError,
                        blocker_key: "WgslValidation: UnbalancedBraces".to_string(),
                        keywords: vec!["UnbalancedBraces".to_string()],
                        detail: format!("unmatched closing brace `}}` on WGSL line {line}"),
                    }));
                }
                brace_depth -= 1;
                i += 1;
            }
            c if is_wgsl_ident_start(c) => {
                let start = i;
                i += 1;
                while i < len && is_wgsl_ident_continue(bytes[i]) {
                    i += 1;
                }
                if brace_depth == 0 && &bytes[start..i] == b"let" {
                    return Err(sanitize_conversion_error(ConversionError {
                        kind: ConversionErrorKind::WgslWriteError,
                        blocker_key: "WgslValidation: ModuleScopeLet".to_string(),
                        keywords: vec!["ModuleScopeLet".to_string()],
                        detail: format!(
                            "invalid module-scope `let` declaration on WGSL line {line}"
                        ),
                    }));
                }
            }
            _ => {
                i += 1;
            }
        }
    }

    if block_comment_depth > 0 {
        return Err(sanitize_conversion_error(ConversionError {
            kind: ConversionErrorKind::WgslWriteError,
            blocker_key: "WgslValidation: UnterminatedComment".to_string(),
            keywords: vec!["UnterminatedComment".to_string()],
            detail: "emitted WGSL ends with an unterminated block comment".to_string(),
        }));
    }

    if brace_depth != 0 {
        return Err(sanitize_conversion_error(ConversionError {
            kind: ConversionErrorKind::WgslWriteError,
            blocker_key: "WgslValidation: UnbalancedBraces".to_string(),
            keywords: vec!["UnbalancedBraces".to_string()],
            detail: format!(
                "emitted WGSL has {brace_depth} unclosed brace(s) at end of module"
            ),
        }));
    }

    if !has_code_token {
        return Err(sanitize_conversion_error(ConversionError {
            kind: ConversionErrorKind::WgslWriteError,
            blocker_key: "WgslValidation: EmptyOutput".to_string(),
            keywords: vec!["EmptyWgslOutput".to_string()],
            detail: "emitted WGSL output is empty".to_string(),
        }));
    }

    Ok(())
}

const fn is_wgsl_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

const fn is_wgsl_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn classify_stream_error(err: SpirvStreamError) -> ConversionError {
    let detail = err.to_string();
    let raw = match err {
        SpirvStreamError::TooShort { .. } => ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: truncated".to_string(),
            keywords: vec!["TruncatedHeader".to_string()],
            detail,
        },
        SpirvStreamError::UnalignedByteLength { .. } => ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: unaligned length".to_string(),
            keywords: vec!["UnalignedWordStream".to_string()],
            detail,
        },
        SpirvStreamError::InvalidMagic { is_big_endian, .. } => {
            let keyword = if is_big_endian {
                "BigEndianSpirv"
            } else {
                "BadMagic"
            };
            ConversionError {
                kind: ConversionErrorKind::InvalidSpirvHeader,
                blocker_key: format!("InvalidSpirvHeader: {keyword}"),
                keywords: vec![keyword.to_string()],
                detail,
            }
        }
        SpirvStreamError::InvalidWordCount { .. }
        | SpirvStreamError::TruncatedInstruction { .. } => ConversionError {
            kind: ConversionErrorKind::SpirvParseError,
            blocker_key: "SpirvParseError: TruncatedOrMalformedStream".to_string(),
            keywords: vec!["IncompleteData".to_string()],
            detail,
        },
    };
    sanitize_conversion_error(raw)
}

fn validate_spirv_header(bytes: &[u8]) -> Result<(), ConversionError> {
    if bytes.len() < SPIRV_HEADER_BYTES {
        return Err(sanitize_conversion_error(ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: truncated".to_string(),
            keywords: vec!["TruncatedHeader".to_string()],
            detail: format!(
                "SPIR-V input too short: {} bytes (minimum {SPIRV_HEADER_BYTES})",
                bytes.len()
            ),
        }));
    }

    if !bytes.len().is_multiple_of(4) {
        return Err(sanitize_conversion_error(ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: unaligned length".to_string(),
            keywords: vec!["UnalignedWordStream".to_string()],
            detail: format!(
                "SPIR-V input length {} is not a multiple of 4 bytes",
                bytes.len()
            ),
        }));
    }

    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    if magic != SPIRV_MAGIC {
        let keyword = if magic == SPIRV_MAGIC.swap_bytes() {
            "BigEndianSpirv"
        } else {
            "BadMagic"
        };
        return Err(sanitize_conversion_error(ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: format!("InvalidSpirvHeader: {keyword}"),
            keywords: vec![keyword.to_string()],
            detail: format!(
                "invalid SPIR-V magic 0x{magic:08x} (expected 0x{SPIRV_MAGIC:08x})"
            ),
        }));
    }

    Ok(())
}

fn detect_entry_point(module: &naga::Module) -> Result<(ShaderStage, String), ConversionError> {
    let Some(first_entry) = module.entry_points.first() else {
        return Err(sanitize_conversion_error(ConversionError {
            kind: ConversionErrorKind::ValidationError,
            blocker_key: "ValidationError: MissingEntryPoint".to_string(),
            keywords: vec!["MissingEntryPoint".to_string()],
            detail: "SPIR-V module contains no entry points".to_string(),
        }));
    };

    let mut primary_stage = None;
    for entry in &module.entry_points {
        let stage = match entry.stage {
            naga::ShaderStage::Vertex => ShaderStage::Vertex,
            naga::ShaderStage::Fragment => ShaderStage::Fragment,
            naga::ShaderStage::Compute => ShaderStage::Compute,
            other => {
                let stage_name = format!("{other:?}");
                return Err(sanitize_conversion_error(ConversionError {
                    kind: ConversionErrorKind::ValidationError,
                    blocker_key: format!("UnsupportedStage({stage_name})"),
                    keywords: vec![format!("Stage::{stage_name}")],
                    detail: format!("shader stage {stage_name} is not supported in WebGPU"),
                }));
            }
        };
        if primary_stage.is_none() {
            primary_stage = Some(stage);
        }
    }

    Ok((
        primary_stage.expect("non-empty entry_points always sets primary_stage"),
        first_entry.name.clone(),
    ))
}

fn classify_spv_error(err: spv::Error) -> ConversionError {
    let detail = err.to_string();
    let raw = match err {
        spv::Error::InvalidHeader => ConversionError {
            kind: ConversionErrorKind::InvalidSpirvHeader,
            blocker_key: "InvalidSpirvHeader: naga rejected header".to_string(),
            keywords: vec!["InvalidHeader".to_string()],
            detail,
        },
        spv::Error::UnsupportedCapability(cap) => {
            let kw = format!("{cap:?}");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedCapability({kw})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnknownCapability(word) => {
            let kw = format!("UnknownCapability({word})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: kw.clone(),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedExtension(ext) => {
            let safe_ext = sanitize_single_line(&ext, MAX_KEYWORD_CHARS);
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedExtension({safe_ext})"),
                keywords: vec![safe_ext],
                detail,
            }
        }
        spv::Error::UnsupportedExtSet(ext) => {
            let safe_ext = sanitize_single_line(&ext, MAX_KEYWORD_CHARS);
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedExtSet({safe_ext})"),
                keywords: vec![safe_ext],
                detail,
            }
        }
        spv::Error::UnsupportedInstruction(_, op) => {
            let kw = format!("{op:?}");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedInstruction({kw})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedExecutionModel(model) => {
            let kw = format!("ExecutionModel({model})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedExecutionModel({model})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedExecutionMode(mode) => {
            let kw = format!("ExecutionMode({mode})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedExecutionMode({mode})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedBuiltIn(builtin) => {
            let kw = format!("BuiltIn({builtin})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedBuiltIn({builtin})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedStorageClass(sc) => {
            let kw = format!("StorageClass({sc})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedStorageClass({sc})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedImageDim(dim) => {
            let kw = format!("ImageDim({dim})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedImageDim({dim})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::UnsupportedImageFormat(fmt_id) => {
            let kw = format!("ImageFormat({fmt_id})");
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("UnsupportedImageFormat({fmt_id})"),
                keywords: vec![kw],
                detail,
            }
        }
        spv::Error::IncompleteData | spv::Error::InvalidWordCount => ConversionError {
            kind: ConversionErrorKind::SpirvParseError,
            blocker_key: "SpirvParseError: TruncatedOrMalformedStream".to_string(),
            keywords: vec!["IncompleteData".to_string()],
            detail,
        },
        other => {
            let variant_name = debug_variant_name(&other);
            let mut keywords = extract_feature_keywords(&detail);
            if keywords.is_empty() {
                keywords.push(variant_name.clone());
            }
            ConversionError {
                kind: ConversionErrorKind::SpirvParseError,
                blocker_key: format!("SpirvParseError: {variant_name}"),
                keywords,
                detail,
            }
        }
    };
    sanitize_conversion_error(raw)
}

fn classify_validation_error(message: &str) -> ConversionError {
    let sanitized = sanitize_diagnostic_text(message);
    let first_line = sanitized
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("validation failed")
        .trim();

    let keywords = extract_feature_keywords(&sanitized);
    let blocker_key = if !keywords.is_empty() {
        format!("ValidationError: {}", keywords.join(", "))
    } else {
        format!("ValidationError: {}", truncate_summary(first_line, 80))
    };

    sanitize_conversion_error(ConversionError {
        kind: ConversionErrorKind::ValidationError,
        blocker_key,
        keywords,
        detail: sanitized,
    })
}

fn classify_wgsl_error(err: wgsl::Error) -> ConversionError {
    let detail = err.to_string();
    let raw = match err {
        wgsl::Error::Unsupported { kind, value } => {
            let kw = format!("{kind}:{value}");
            ConversionError {
                kind: ConversionErrorKind::WgslWriteError,
                blocker_key: format!("WgslUnsupported({kw})"),
                keywords: vec![kw],
                detail,
            }
        }
        wgsl::Error::UnsupportedRelationalFunction(func) => {
            let kw = format!("RelationalFunction::{func:?}");
            ConversionError {
                kind: ConversionErrorKind::WgslWriteError,
                blocker_key: format!("WgslUnsupported({kw})"),
                keywords: vec![kw],
                detail,
            }
        }
        other => {
            let variant_name = debug_variant_name(&other);
            let mut keywords = extract_feature_keywords(&detail);
            if keywords.is_empty() {
                keywords.push(variant_name.clone());
            }
            ConversionError {
                kind: ConversionErrorKind::WgslWriteError,
                blocker_key: format!("WgslWriteError: {variant_name}"),
                keywords,
                detail,
            }
        }
    };
    sanitize_conversion_error(raw)
}

fn sanitize_conversion_error(err: ConversionError) -> ConversionError {
    let detail = sanitize_diagnostic_text(&err.detail);
    let blocker_key = sanitize_single_line(&err.blocker_key, MAX_BLOCKER_KEY_CHARS);
    let mut keywords = Vec::with_capacity(err.keywords.len().min(MAX_KEYWORDS));
    for kw in err.keywords {
        if keywords.len() >= MAX_KEYWORDS {
            break;
        }
        let cleaned = sanitize_single_line(&kw, MAX_KEYWORD_CHARS);
        if !cleaned.is_empty() {
            push_unique(&mut keywords, cleaned);
        }
    }
    ConversionError {
        kind: err.kind,
        blocker_key,
        keywords,
        detail,
    }
}

fn sanitize_single_line(text: &str, max_chars: usize) -> String {
    let sanitized = sanitize_diagnostic_text(text);
    let collapsed = sanitized
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    truncate_summary(&collapsed, max_chars)
}

/// Sanitize a diagnostic or error string so it is deterministic, single-standard-newline (`\n`),
/// free of ANSI escape sequences and control characters, bounded in length, and stripped of all
/// host filesystem paths (Windows drive paths, UNC paths, Unix absolute paths, home-relative
/// `~/` paths, parent traversal `../` paths, and `file://` URIs).
pub fn sanitize_diagnostic_text(text: &str) -> String {
    let cleaned = strip_ansi_and_control_chars(text);
    let redacted = redact_host_paths(&cleaned);
    let trimmed = redacted.trim();
    truncate_summary(trimmed, MAX_ERROR_DETAIL_CHARS)
}

/// Sanitize a source path label recorded by [`audit_spirv_file`] so absolute host paths, UNC
/// paths, `file://` URIs, home paths, and `..` traversals are reduced to basename-only while clean
/// relative forward-slash paths (e.g. `sub/shader.spv`) are preserved deterministically.
pub fn sanitize_audit_source_path(source_path: &str) -> String {
    let cleaned = strip_ansi_and_control_chars(source_path);
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        return "unknown.spv".to_string();
    }

    let mut norm = trimmed.replace('\\', "/");
    let lower = norm.to_ascii_lowercase();
    let is_host_or_traversal = lower.starts_with("file://")
        || norm.starts_with('/')
        || norm.starts_with("~/")
        || norm == "~"
        || has_windows_drive_prefix(&norm)
        || norm.split('/').any(|seg| seg == "..");

    if is_host_or_traversal {
        let stripped_uri = if lower.starts_with("file://") {
            norm[7..].trim_start_matches('/')
        } else {
            norm.as_str()
        };
        if stripped_uri.is_empty() || stripped_uri.ends_with('/') {
            return "unknown.spv".to_string();
        }
        let candidate = stripped_uri.rsplit('/').next().unwrap_or("");
        if candidate.is_empty()
            || candidate == "."
            || candidate == ".."
            || candidate == "~"
            || has_windows_drive_prefix(candidate)
        {
            return "unknown.spv".to_string();
        }
        return candidate.to_string();
    }

    // Clean relative path: collapse redundant slashes and strip leading `./` segments.
    let mut segments = Vec::new();
    for seg in norm.split('/') {
        if seg.is_empty() || seg == "." {
            continue;
        }
        segments.push(seg);
    }
    if segments.is_empty() {
        norm.clear();
        "unknown.spv".to_string()
    } else {
        segments.join("/")
    }
}

fn has_windows_drive_prefix(s: &str) -> bool {
    let bytes = s.as_bytes();
    bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

fn strip_ansi_and_control_chars(text: &str) -> String {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;

    while i < len {
        let b = bytes[i];
        // Strip ANSI CSI escape sequences (`ESC [ ... final_byte`).
        if b == 0x1b {
            if i + 1 < len && bytes[i + 1] == b'[' {
                i += 2;
                while i < len {
                    let c = bytes[i];
                    i += 1;
                    if (0x40..=0x7e).contains(&c) {
                        break;
                    }
                }
            } else {
                i += 1;
            }
            continue;
        }
        if b == b'\r' {
            if i + 1 < len && bytes[i + 1] == b'\n' {
                i += 2;
            } else {
                i += 1;
            }
            out.push('\n');
            continue;
        }
        if b == b'\n' || b == b'\t' {
            out.push(b as char);
            i += 1;
            continue;
        }
        if b < 0x20 || b == 0x7f {
            i += 1;
            continue;
        }
        let ch = text[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn redact_host_paths(text: &str) -> String {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0usize;

    while i < len {
        let at_boundary = i == 0 || is_path_boundary_before(bytes[i - 1]);
        if at_boundary {
            // Preserve non-file web URLs (`http://`, `https://`) without redacting their slashes.
            if starts_with_ascii_ignore_case(&bytes[i..], b"http://")
                || starts_with_ascii_ignore_case(&bytes[i..], b"https://")
            {
                let end = scan_unquoted_token_end(bytes, i);
                out.push_str(&text[i..end]);
                i = end;
                continue;
            }

            let opening_quote = if i > 0 && (bytes[i - 1] == b'"' || bytes[i - 1] == b'\'') {
                Some(bytes[i - 1])
            } else {
                None
            };

            if let Some((path_end, suffix_start)) = match_host_path(bytes, i, opening_quote) {
                out.push_str(REDACTED_PATH_TOKEN);
                if suffix_start < path_end {
                    out.push_str(&text[suffix_start..path_end]);
                }
                i = path_end;
                continue;
            }
        }

        let ch = text[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

const fn is_path_boundary_before(b: u8) -> bool {
    matches!(
        b,
        b' ' | b'\t'
            | b'\n'
            | b'"'
            | b'\''
            | b'`'
            | b'('
            | b'['
            | b'{'
            | b'<'
            | b'='
            | b','
            | b';'
            | b':'
    )
}

const fn is_unquoted_path_stop(b: u8) -> bool {
    b.is_ascii_whitespace()
        || matches!(
            b,
            b'"' | b'\'' | b'`' | b'(' | b')' | b'[' | b']' | b'{' | b'}' | b'<' | b'>' | b','
                | b';'
        )
}

fn starts_with_ascii_ignore_case(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.len() >= needle.len()
        && haystack[..needle.len()]
            .iter()
            .zip(needle.iter())
            .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
}

fn scan_unquoted_token_end(bytes: &[u8], start: usize) -> usize {
    let mut end = start;
    while end < bytes.len() && !is_unquoted_path_stop(bytes[end]) {
        end += 1;
    }
    end
}

fn scan_path_token_end(bytes: &[u8], start: usize, opening_quote: Option<u8>) -> usize {
    if let Some(q) = opening_quote {
        let mut cursor = start;
        while cursor < bytes.len() && bytes[cursor] != b'\n' {
            if bytes[cursor] == q {
                return cursor;
            }
            cursor += 1;
        }
    }
    let mut end = scan_unquoted_token_end(bytes, start);
    // Keep sentence-ending trailing `.` outside the redacted token unless it's `..` or `/.`.
    while end > start + 2
        && bytes[end - 1] == b'.'
        && bytes[end - 2] != b'.'
        && bytes[end - 2] != b'/'
        && bytes[end - 2] != b'\\'
    {
        end -= 1;
    }
    end
}

/// If `bytes[start..]` begins a host filesystem path, returns `(token_end, suffix_start)`
/// where `bytes[start..suffix_start]` is replaced with `<path>` and `bytes[suffix_start..token_end]`
/// (such as `:42:10`) is preserved.
fn match_host_path(
    bytes: &[u8],
    start: usize,
    opening_quote: Option<u8>,
) -> Option<(usize, usize)> {
    let rem = &bytes[start..];

    // 1. `file://...` URI.
    if starts_with_ascii_ignore_case(rem, b"file://") {
        let end = scan_path_token_end(bytes, start, opening_quote);
        if end > start + 7 {
            let suffix = find_line_col_suffix(bytes, start + 7, end);
            return Some((end, suffix));
        }
        return None;
    }

    // 2. UNC or Windows extended-length path (`\\server\share\...`, `\\?\C:\...`, `\\.\...`).
    if rem.len() >= 3
        && rem[0] == b'\\'
        && rem[1] == b'\\'
        && (rem[2].is_ascii_alphanumeric() || matches!(rem[2], b'?' | b'.' | b'_'))
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let suffix = find_line_col_suffix(bytes, start + 2, end);
        return Some((end, suffix));
    }

    // 3. Windows drive path (`C:\...` or `C:/...`).
    if rem.len() >= 3
        && rem[0].is_ascii_alphabetic()
        && rem[1] == b':'
        && (rem[2] == b'\\' || rem[2] == b'/')
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let suffix = find_line_col_suffix(bytes, start + 3, end);
        return Some((end, suffix));
    }

    // 4. Home-relative path (`~/...` or `~\...`).
    if rem.len() >= 3
        && rem[0] == b'~'
        && (rem[1] == b'/' || rem[1] == b'\\')
        && !is_unquoted_path_stop(rem[2])
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let suffix = find_line_col_suffix(bytes, start + 2, end);
        return Some((end, suffix));
    }

    // 5. Parent-directory traversal (`../...` or `..\...`).
    if rem.len() >= 4
        && rem[0] == b'.'
        && rem[1] == b'.'
        && (rem[2] == b'/' || rem[2] == b'\\')
        && !is_unquoted_path_stop(rem[3])
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let suffix = find_line_col_suffix(bytes, start + 3, end);
        return Some((end, suffix));
    }

    // 6. Unix absolute path (`/Users/...`, `/home/...`, `/tmp/...`, or any multi-segment `/a/b`).
    if rem.len() >= 2
        && rem[0] == b'/'
        && (rem[1].is_ascii_alphanumeric() || matches!(rem[1], b'.' | b'_' | b'@' | b'-'))
    {
        let end = scan_path_token_end(bytes, start, opening_quote);
        let token = &bytes[start..end];
        let has_second_slash = token[1..].contains(&b'/');
        let is_known_unix_root = matches!(
            token,
            b"/Users"
                | b"/home"
                | b"/tmp"
                | b"/var"
                | b"/etc"
                | b"/opt"
                | b"/mnt"
                | b"/root"
                | b"/private"
                | b"/Volumes"
                | b"/workspace"
        );
        if has_second_slash || is_known_unix_root {
            let suffix = find_line_col_suffix(bytes, start + 1, end);
            return Some((end, suffix));
        }
    }

    None
}

/// Detect optional trailing `:line` or `:line:col` suffix inside `bytes[min_prefix_end..token_end]`.
fn find_line_col_suffix(bytes: &[u8], min_prefix_end: usize, token_end: usize) -> usize {
    let mut split = token_end;
    for _ in 0..2 {
        let mut cursor = split;
        while cursor > min_prefix_end && bytes[cursor - 1].is_ascii_digit() {
            cursor -= 1;
        }
        if cursor < split && cursor > min_prefix_end && bytes[cursor - 1] == b':' {
            split = cursor - 1;
        } else {
            break;
        }
    }
    split
}

fn debug_variant_name(value: &impl fmt::Debug) -> String {
    let dbg = format!("{value:?}");
    dbg.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("Unknown")
        .to_string()
}

fn truncate_summary(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let collected: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{collected}...")
    } else {
        collected
    }
}

/// Extract capability / extension / WebGPU blocker keywords from diagnostic text.
pub fn extract_feature_keywords(text: &str) -> Vec<String> {
    let sanitized = sanitize_diagnostic_text(text);
    let mut found = Vec::new();

    // Extract Capabilities(...) tokens emitted by Naga validation errors.
    let mut rest = sanitized.as_str();
    while let Some(idx) = rest.find("Capabilities(") {
        let after = &rest[idx + "Capabilities(".len()..];
        if let Some(end) = after.find(')') {
            for token in after[..end].split('|') {
                let cleaned = token.trim();
                if !cleaned.is_empty() {
                    push_unique(&mut found, format!("Capability::{cleaned}"));
                }
            }
            rest = &after[end + 1..];
        } else {
            break;
        }
    }

    // Extract SPIR-V extension identifiers (e.g. SPV_KHR_ray_tracing).
    for word in sanitized.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if word.starts_with("SPV_") && word.len() > 4 {
            push_unique(&mut found, word.to_string());
        }
    }

    // Match well-known WebGPU blocker feature terms when present in diagnostics.
    let lower = sanitized.to_ascii_lowercase();
    const KNOWN_BLOCKERS: &[(&str, &str)] = &[
        ("push_constant", "PushConstants"),
        ("immediate", "ImmediateData"),
        ("float64", "Float64"),
        ("f64", "Float64"),
        ("int64", "Int64"),
        ("i64", "Int64"),
        ("u64", "Int64"),
        ("subgroup", "Subgroup"),
        ("ray_query", "RayQuery"),
        ("rayquery", "RayQuery"),
        ("mesh", "MeshShader"),
        ("task", "TaskShader"),
        ("binding_array", "BindingArray"),
        ("bindless", "Bindless"),
        ("dual_source", "DualSourceBlending"),
        ("clip_distance", "ClipDistances"),
        ("cull_distance", "CullDistance"),
        ("primitive_index", "PrimitiveIndex"),
        ("barycentric", "Barycentric"),
    ];

    for &(needle, label) in KNOWN_BLOCKERS {
        if lower.contains(needle) {
            push_unique(&mut found, label.to_string());
        }
    }

    found
}

fn push_unique(list: &mut Vec<String>, item: String) {
    if !list.iter().any(|existing| existing == &item) {
        list.push(item);
    }
}

#[cfg(test)]
pub(crate) mod test_spirv {
    use super::SPIRV_MAGIC;

    /// Minimal SPIR-V word stream builder for deterministic in-memory unit tests.
    pub(crate) struct SpirvBuilder {
        version: u32,
        next_id: u32,
        words: Vec<u32>,
    }

    impl SpirvBuilder {
        pub(crate) fn new() -> Self {
            Self {
                version: 0x0001_0300, // SPIR-V 1.3
                next_id: 1,
                words: Vec::new(),
            }
        }

        pub(crate) fn id(&mut self) -> u32 {
            let id = self.next_id;
            self.next_id += 1;
            id
        }

        pub(crate) fn inst(&mut self, opcode: u16, operands: &[u32]) {
            let word_count = (1 + operands.len()) as u32;
            self.words.push((word_count << 16) | u32::from(opcode));
            self.words.extend_from_slice(operands);
        }

        pub(crate) fn inst_with_string(
            &mut self,
            opcode: u16,
            prefix: &[u32],
            text: &str,
            suffix: &[u32],
        ) {
            let mut str_words = encode_spirv_string(text);
            let mut operands = Vec::with_capacity(prefix.len() + str_words.len() + suffix.len());
            operands.extend_from_slice(prefix);
            operands.append(&mut str_words);
            operands.extend_from_slice(suffix);
            self.inst(opcode, &operands);
        }

        pub(crate) fn finish_bytes(self) -> Vec<u8> {
            let mut full = Vec::with_capacity(5 + self.words.len());
            full.push(SPIRV_MAGIC);
            full.push(self.version);
            full.push(0); // generator
            full.push(self.next_id); // id bound
            full.push(0); // schema
            full.extend_from_slice(&self.words);

            let mut bytes = Vec::with_capacity(full.len() * 4);
            for w in full {
                bytes.extend_from_slice(&w.to_le_bytes());
            }
            bytes
        }
    }

    fn encode_spirv_string(s: &str) -> Vec<u32> {
        let mut raw = s.as_bytes().to_vec();
        raw.push(0); // NUL terminator
        while !raw.len().is_multiple_of(4) {
            raw.push(0);
        }
        raw.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // SPIR-V opcodes used by synthetic shader tests
    const OP_EXTENSION: u16 = 10;
    const OP_MEMORY_MODEL: u16 = 14;
    const OP_ENTRY_POINT: u16 = 15;
    const OP_EXECUTION_MODE: u16 = 16;
    const OP_CAPABILITY: u16 = 17;
    const OP_TYPE_VOID: u16 = 19;
    const OP_TYPE_FLOAT: u16 = 22;
    const OP_TYPE_VECTOR: u16 = 23;
    const OP_TYPE_POINTER: u16 = 32;
    const OP_TYPE_FUNCTION: u16 = 33;
    const OP_CONSTANT: u16 = 43;
    const OP_CONSTANT_COMPOSITE: u16 = 44;
    const OP_FUNCTION: u16 = 54;
    const OP_FUNCTION_END: u16 = 56;
    const OP_VARIABLE: u16 = 59;
    const OP_STORE: u16 = 62;
    const OP_DECORATE: u16 = 71;
    const OP_LABEL: u16 = 248;
    const OP_RETURN: u16 = 253;

    // SPIR-V enum constants
    const CAP_SHADER: u32 = 1;
    const CAP_FLOAT64: u32 = 10;
    const CAP_RAY_QUERY_KHR: u32 = 4472;
    const ADDRESSING_LOGICAL: u32 = 0;
    const MEMORY_GLSL450: u32 = 1;
    const EXEC_MODEL_VERTEX: u32 = 0;
    const EXEC_MODEL_FRAGMENT: u32 = 4;
    const EXEC_MODEL_GL_COMPUTE: u32 = 5;
    const EXEC_MODE_ORIGIN_UPPER_LEFT: u32 = 7;
    const EXEC_MODE_LOCAL_SIZE: u32 = 17;
    const STORAGE_CLASS_OUTPUT: u32 = 3;
    const DECORATION_BUILTIN: u32 = 11;
    const DECORATION_LOCATION: u32 = 30;
    const BUILTIN_POSITION: u32 = 0;

    /// Build a valid minimal SPIR-V compute shader (`@compute @workgroup_size(1, 1, 1)`).
    pub(crate) fn build_minimal_compute_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_GL_COMPUTE, main_fn], "main", &[]);
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    /// Build a valid minimal SPIR-V vertex shader writing `@builtin(position) = vec4(0, 0, 0, 1)`.
    pub(crate) fn build_minimal_vertex_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let f32_ty = b.id();
        let vec4_ty = b.id();
        let ptr_out_vec4 = b.id();
        let pos_var = b.id();
        let c_zero = b.id();
        let c_one = b.id();
        let c_pos = b.id();
        let vs_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_VERTEX, vs_fn],
            "vs_main",
            &[pos_var],
        );
        b.inst(OP_DECORATE, &[pos_var, DECORATION_BUILTIN, BUILTIN_POSITION]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_TYPE_FLOAT, &[f32_ty, 32]);
        b.inst(OP_TYPE_VECTOR, &[vec4_ty, f32_ty, 4]);
        b.inst(OP_TYPE_POINTER, &[ptr_out_vec4, STORAGE_CLASS_OUTPUT, vec4_ty]);
        b.inst(OP_VARIABLE, &[ptr_out_vec4, pos_var, STORAGE_CLASS_OUTPUT]);
        b.inst(OP_CONSTANT, &[f32_ty, c_zero, 0.0f32.to_bits()]);
        b.inst(OP_CONSTANT, &[f32_ty, c_one, 1.0f32.to_bits()]);
        b.inst(OP_CONSTANT_COMPOSITE, &[vec4_ty, c_pos, c_zero, c_zero, c_zero, c_one]);
        b.inst(OP_FUNCTION, &[void_ty, vs_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_STORE, &[pos_var, c_pos]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    /// Build a valid minimal SPIR-V fragment shader writing `@location(0) = vec4(1, 0, 0, 1)`.
    pub(crate) fn build_minimal_fragment_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let f32_ty = b.id();
        let vec4_ty = b.id();
        let ptr_out_vec4 = b.id();
        let color_var = b.id();
        let c_zero = b.id();
        let c_one = b.id();
        let c_red = b.id();
        let fs_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(
            OP_ENTRY_POINT,
            &[EXEC_MODEL_FRAGMENT, fs_fn],
            "fs_main",
            &[color_var],
        );
        b.inst(OP_EXECUTION_MODE, &[fs_fn, EXEC_MODE_ORIGIN_UPPER_LEFT]);
        b.inst(OP_DECORATE, &[color_var, DECORATION_LOCATION, 0]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_TYPE_FLOAT, &[f32_ty, 32]);
        b.inst(OP_TYPE_VECTOR, &[vec4_ty, f32_ty, 4]);
        b.inst(OP_TYPE_POINTER, &[ptr_out_vec4, STORAGE_CLASS_OUTPUT, vec4_ty]);
        b.inst(OP_VARIABLE, &[ptr_out_vec4, color_var, STORAGE_CLASS_OUTPUT]);
        b.inst(OP_CONSTANT, &[f32_ty, c_zero, 0.0f32.to_bits()]);
        b.inst(OP_CONSTANT, &[f32_ty, c_one, 1.0f32.to_bits()]);
        b.inst(OP_CONSTANT_COMPOSITE, &[vec4_ty, c_red, c_one, c_zero, c_zero, c_one]);
        b.inst(OP_FUNCTION, &[void_ty, fs_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_STORE, &[color_var, c_red]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    /// Build a SPIR-V shader declaring an unsupported extension (`SPV_KHR_ray_tracing`).
    pub(crate) fn build_unsupported_extension_spv() -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst_with_string(OP_EXTENSION, &[], "SPV_KHR_ray_tracing", &[]);
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_GL_COMPUTE, main_fn], "main", &[]);
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }

    /// Build a SPIR-V shader declaring a non-WebGPU capability (`Float64` or `RayQueryKHR`).
    pub(crate) fn build_unsupported_capability_spv(use_ray_query: bool) -> Vec<u8> {
        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(OP_CAPABILITY, &[CAP_SHADER]);
        b.inst(
            OP_CAPABILITY,
            &[if use_ray_query {
                CAP_RAY_QUERY_KHR
            } else {
                CAP_FLOAT64
            }],
        );
        b.inst(OP_MEMORY_MODEL, &[ADDRESSING_LOGICAL, MEMORY_GLSL450]);
        b.inst_with_string(OP_ENTRY_POINT, &[EXEC_MODEL_GL_COMPUTE, main_fn], "main", &[]);
        b.inst(OP_EXECUTION_MODE, &[main_fn, EXEC_MODE_LOCAL_SIZE, 1, 1, 1]);
        b.inst(OP_TYPE_VOID, &[void_ty]);
        b.inst(OP_TYPE_FUNCTION, &[fn_ty, void_ty]);
        b.inst(OP_FUNCTION, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(OP_LABEL, &[entry_label]);
        b.inst(OP_RETURN, &[]);
        b.inst(OP_FUNCTION_END, &[]);
        b.finish_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::test_spirv::*;
    use super::*;

    #[test]
    fn fnv1a_64_matches_standard_test_vectors() {
        assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
        assert_eq!(format_shader_hash(0xdead_beef), "0x00000000deadbeef");
    }

    #[test]
    fn converts_minimal_compute_spirv_to_wgsl() {
        let spv = build_minimal_compute_spv();
        let converted = convert_spirv_bytes(&spv).expect("minimal compute SPIR-V should convert");

        assert_eq!(converted.stage, ShaderStage::Compute);
        assert_eq!(converted.entry_point, "main");
        assert_eq!(converted.hash, fnv1a_64(&spv));
        assert_eq!(converted.spirv_bytes, spv.len());
        assert!(
            converted.wgsl.contains("@compute"),
            "expected @compute in WGSL:\n{}",
            converted.wgsl
        );
        assert!(
            converted.wgsl.contains("@workgroup_size(1, 1, 1)"),
            "expected workgroup_size in WGSL:\n{}",
            converted.wgsl
        );
    }

    #[test]
    fn converts_minimal_vertex_spirv_to_wgsl() {
        let spv = build_minimal_vertex_spv();
        let converted = convert_spirv_bytes(&spv).expect("minimal vertex SPIR-V should convert");

        assert_eq!(converted.stage, ShaderStage::Vertex);
        assert_eq!(converted.entry_point, "vs_main");
        assert_eq!(converted.hash, fnv1a_64(&spv));
        assert!(
            converted.wgsl.contains("@vertex"),
            "expected @vertex in WGSL:\n{}",
            converted.wgsl
        );
        assert!(
            converted.wgsl.contains("@builtin(position)"),
            "expected @builtin(position) in WGSL:\n{}",
            converted.wgsl
        );
    }

    #[test]
    fn converts_minimal_fragment_spirv_to_wgsl() {
        let spv = build_minimal_fragment_spv();
        let converted = convert_spirv_bytes(&spv).expect("minimal fragment SPIR-V should convert");

        assert_eq!(converted.stage, ShaderStage::Fragment);
        assert_eq!(converted.entry_point, "fs_main");
        assert!(
            converted.wgsl.contains("@fragment"),
            "expected @fragment in WGSL:\n{}",
            converted.wgsl
        );
        assert!(
            converted.wgsl.contains("@location(0)"),
            "expected @location(0) in WGSL:\n{}",
            converted.wgsl
        );
    }

    #[test]
    fn rejects_invalid_and_truncated_spirv_headers() {
        let err_short = convert_spirv_bytes(&[0x03, 0x02, 0x23, 0x07]).unwrap_err();
        assert_eq!(err_short.kind, ConversionErrorKind::InvalidSpirvHeader);
        assert!(err_short.blocker_key.contains("truncated"));

        let mut bad_magic = vec![0u8; 24];
        bad_magic[0..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        let err_magic = convert_spirv_bytes(&bad_magic).unwrap_err();
        assert_eq!(err_magic.kind, ConversionErrorKind::InvalidSpirvHeader);
        assert!(err_magic.keywords.iter().any(|k| k == "BadMagic"));

        let unaligned = vec![0u8; 22];
        let err_unaligned = convert_spirv_bytes(&unaligned).unwrap_err();
        assert_eq!(err_unaligned.kind, ConversionErrorKind::InvalidSpirvHeader);
        assert!(err_unaligned.keywords.iter().any(|k| k == "UnalignedWordStream"));

        // Valid header, truncated instruction payload.
        let mut valid_compute = build_minimal_compute_spv();
        valid_compute.truncate(valid_compute.len() - 8);
        let err_trunc = convert_spirv_bytes(&valid_compute).unwrap_err();
        assert_eq!(err_trunc.kind, ConversionErrorKind::SpirvParseError);
    }

    #[test]
    fn categorizes_unsupported_extension_and_capability() {
        let ext_spv = build_unsupported_extension_spv();
        let err = convert_spirv_bytes(&ext_spv).unwrap_err();
        assert_eq!(err.kind, ConversionErrorKind::SpirvParseError);
        assert!(
            err.keywords.iter().any(|k| k == "SPV_KHR_ray_tracing"),
            "expected SPV_KHR_ray_tracing keyword, got {:?}",
            err
        );

        let cap_spv = build_unsupported_capability_spv(true);
        let cap_err = convert_spirv_bytes(&cap_spv).unwrap_err();
        assert!(
            matches!(
                cap_err.kind,
                ConversionErrorKind::SpirvParseError | ConversionErrorKind::ValidationError
            ),
            "unexpected error kind: {:?}",
            cap_err
        );
        assert!(
            !cap_err.keywords.is_empty(),
            "expected extracted capability keyword: {:?}",
            cap_err
        );
    }

    #[test]
    fn categorizes_validation_error_for_missing_entry_point_and_extracts_keywords() {
        let mut b = SpirvBuilder::new();
        b.inst(17, &[1]); // OpCapability Shader
        b.inst(14, &[0, 1]); // OpMemoryModel Logical GLSL450
        let no_entry_spv = b.finish_bytes();

        let err = convert_spirv_bytes(&no_entry_spv).unwrap_err();
        assert_eq!(err.kind, ConversionErrorKind::ValidationError);
        assert!(err.keywords.iter().any(|k| k == "MissingEntryPoint"));

        let kws = extract_feature_keywords(
            "Required Capabilities(FLOAT64 | SUBGROUP) via SPV_KHR_shader_subgroup_extended_types",
        );
        assert!(kws.iter().any(|k| k == "Capability::FLOAT64"));
        assert!(kws.iter().any(|k| k == "Capability::SUBGROUP"));
        assert!(kws
            .iter()
            .any(|k| k == "SPV_KHR_shader_subgroup_extended_types"));
    }

    #[test]
    fn converts_spirv_with_push_constant_when_lowering_enabled_and_rejects_when_disabled() {
        // Minimal compute shader reading from a PushConstant block and binding a uniform in set 5.
        const OP_TYPE_STRUCT: u16 = 30;
        const OP_MEMBER_DECORATE: u16 = 72;
        const DECORATION_BLOCK: u32 = 2;
        const DECORATION_BINDING: u32 = 33;
        const DECORATION_DESCRIPTOR_SET: u32 = 34;
        const DECORATION_OFFSET: u32 = 35;
        const STORAGE_CLASS_UNIFORM: u32 = 2;
        const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;

        let mut b = SpirvBuilder::new();
        let void_ty = b.id();
        let fn_ty = b.id();
        let f32_ty = b.id();
        let vec4_ty = b.id();
        let struct_ty = b.id();
        let ptr_pc_ty = b.id();
        let ptr_ubo_ty = b.id();
        let pc_var = b.id();
        let ubo_var = b.id();
        let main_fn = b.id();
        let entry_label = b.id();

        b.inst(17, &[1]); // OpCapability Shader
        b.inst(14, &[0, 1]); // OpMemoryModel Logical GLSL450
        b.inst_with_string(15, &[5, main_fn], "main", &[]); // OpEntryPoint GLCompute
        b.inst(16, &[main_fn, 17, 1, 1, 1]); // OpExecutionMode LocalSize 1 1 1
        b.inst(71, &[struct_ty, DECORATION_BLOCK]);
        b.inst(OP_MEMBER_DECORATE, &[struct_ty, 0, DECORATION_OFFSET, 0]);
        b.inst(71, &[ubo_var, DECORATION_DESCRIPTOR_SET, 5]);
        b.inst(71, &[ubo_var, DECORATION_BINDING, 1]);
        b.inst(19, &[void_ty]);
        b.inst(33, &[fn_ty, void_ty]);
        b.inst(22, &[f32_ty, 32]);
        b.inst(23, &[vec4_ty, f32_ty, 4]);
        b.inst(OP_TYPE_STRUCT, &[struct_ty, vec4_ty]);
        b.inst(32, &[ptr_pc_ty, STORAGE_CLASS_PUSH_CONSTANT, struct_ty]);
        b.inst(32, &[ptr_ubo_ty, STORAGE_CLASS_UNIFORM, struct_ty]);
        b.inst(59, &[ptr_pc_ty, pc_var, STORAGE_CLASS_PUSH_CONSTANT]);
        b.inst(59, &[ptr_ubo_ty, ubo_var, STORAGE_CLASS_UNIFORM]);
        b.inst(54, &[void_ty, main_fn, 0, fn_ty]);
        b.inst(248, &[entry_label]);
        b.inst(253, &[]);
        b.inst(56, &[]);
        let spv = b.finish_bytes();

        // With default options (lower_for_webgpu = true), both push constants and set 5 lower cleanly.
        let lowered = convert_spirv_bytes(&spv)
            .expect("push constant + set 5 SPIR-V should convert when lowering is enabled");
        assert_eq!(lowered.lowering.push_constant_remaps.len(), 1);
        assert_eq!(lowered.lowering.push_constant_remaps[0].new_group, 3);
        assert_eq!(lowered.lowering.push_constant_remaps[0].new_binding, 0);
        assert_eq!(lowered.lowering.bind_group_remaps.len(), 1);
        assert_eq!(lowered.lowering.bind_group_remaps[0].old_group, 5);
        assert_eq!(lowered.lowering.bind_group_remaps[0].old_binding, 1);
        assert!(lowered.lowering.bind_group_remaps[0].new_group < 4);
        assert!(lowered.wgsl.contains("@group(3) @binding(0)"));
        assert!(!lowered.wgsl.contains("var<immediate>"));
        assert!(!lowered.wgsl.contains("@group(5)"));

        // With lowering disabled, Naga baseline validation rejects AddressSpace::Immediate.
        let err = convert_spirv_bytes_with_options(&spv, &ConvertOptions::default().without_lowering())
            .expect_err("push constant SPIR-V must fail baseline validation when lowering is disabled");
        assert_eq!(err.kind, ConversionErrorKind::ValidationError);
    }

    #[test]
    fn validate_emitted_wgsl_rejects_module_scope_let_variants() {
        let cases: &[(&str, usize)] = &[
            ("let x: u32 = 1u;\nfn main() {}\n", 1),
            ("const a: u32 = 1u;\n   let b: f32 = 0.0;\n", 2),
            ("const a: u32 = 1u;\nlet\tb: u32 = 2u;\n", 2),
            ("const a: u32 = 1u; let b: u32 = 2u;\n", 1),
            ("fn helper() {\n    let ok = 1u;\n}\nlet bad = 2u;\n", 4),
            ("fn helper() { let ok = 1u; } let bad = 2u;\n", 1),
            ("/* comment */ let x = 1u;\n", 1),
            ("/* outer /* inner */ still comment */ let x = 1u;\n", 1),
            ("const a = 1u;\nlet\n    x = 2u;\n", 2),
        ];

        for &(src, expected_line) in cases {
            let err = validate_emitted_wgsl(src)
                .expect_err(&format!("expected module-scope `let` error for:\n{src}"));
            assert_eq!(err.kind, ConversionErrorKind::WgslWriteError);
            assert_eq!(err.blocker_key, "WgslValidation: ModuleScopeLet");
            assert_eq!(err.keywords, vec!["ModuleScopeLet".to_string()]);
            assert!(
                err.detail.contains(&format!("line {expected_line}")),
                "expected line {expected_line} in detail '{}'",
                err.detail
            );
        }
    }

    #[test]
    fn validate_emitted_wgsl_accepts_function_scope_let_comments_and_ident_prefixes() {
        let valid_wgsl = r#"
// let fake_line_comment = 1u;
/* let fake_block_comment = 2u; { unmatched brace in comment */
/* outer /* inner let nested = 3u; } */ still comment */
const letter: u32 = 1u;
var<private> let_value: f32 = 0.0;
alias let0 = f32;
struct let_struct {
    let_field: u32,
}

fn let_helper() -> u32 {
    let inner: u32 = letter;
    return inner;
}

@compute @workgroup_size(1, 1, 1)
fn main() {
    let local_val: u32 = let_helper();
    _ = local_val;
}
"#;
        validate_emitted_wgsl(valid_wgsl)
            .expect("valid WGSL with function-scope let and let-prefixed identifiers should pass");
    }

    #[test]
    fn validate_emitted_wgsl_rejects_empty_unbalanced_braces_and_unterminated_comments() {
        for empty in ["", "   \n\t\r\n", "// comment only\n/* block */\n"] {
            let err = validate_emitted_wgsl(empty).expect_err("empty WGSL must be rejected");
            assert_eq!(err.kind, ConversionErrorKind::WgslWriteError);
            assert_eq!(err.blocker_key, "WgslValidation: EmptyOutput");
            assert_eq!(err.keywords, vec!["EmptyWgslOutput".to_string()]);
        }

        let stray_close = validate_emitted_wgsl("fn f() {}\n}\n")
            .expect_err("stray closing brace must be rejected");
        assert_eq!(stray_close.kind, ConversionErrorKind::WgslWriteError);
        assert_eq!(stray_close.blocker_key, "WgslValidation: UnbalancedBraces");

        let unclosed_open = validate_emitted_wgsl("fn f() {\n    let x = 1u;\n")
            .expect_err("unclosed opening brace must be rejected");
        assert_eq!(unclosed_open.kind, ConversionErrorKind::WgslWriteError);
        assert_eq!(unclosed_open.blocker_key, "WgslValidation: UnbalancedBraces");

        let unterminated_comment = validate_emitted_wgsl("/* outer /* inner */\nfn f() {}\n")
            .expect_err("unterminated nested block comment must be rejected");
        assert_eq!(unterminated_comment.kind, ConversionErrorKind::WgslWriteError);
        assert_eq!(
            unterminated_comment.blocker_key,
            "WgslValidation: UnterminatedComment"
        );
    }

    #[test]
    fn converted_spirv_wgsl_round_trips_through_naga_wgsl_frontend_with_zero_module_scope_let() {
        let stages = [
            ("compute", build_minimal_compute_spv()),
            ("vertex", build_minimal_vertex_spv()),
            ("fragment", build_minimal_fragment_spv()),
        ];

        for (label, spv) in stages {
            let converted =
                convert_spirv_bytes(&spv).unwrap_or_else(|e| panic!("{label} failed: {e}"));
            assert!(
                !converted.wgsl.contains('\r'),
                "{label}: emitted WGSL must use normalized LF newlines"
            );
            validate_emitted_wgsl(&converted.wgsl)
                .unwrap_or_else(|e| panic!("{label}: emitted WGSL failed post-validation: {e}"));

            // Round-trip parse & validate through Naga's WGSL front-end to verify zero WGSL
            // grammar violations (including module-scope `let`).
            let reparsed = naga::front::wgsl::parse_str(&converted.wgsl)
                .unwrap_or_else(|e| panic!("{label}: WGSL re-parse failed: {}", e.emit_to_string(&converted.wgsl)));
            let mut validator = Validator::new(ValidationFlags::all(), Capabilities::empty());
            validator
                .validate(&reparsed)
                .unwrap_or_else(|e| panic!("{label}: WGSL re-validation failed: {}", e.emit_to_string(&converted.wgsl)));
        }
    }

    #[test]
    fn conversion_output_and_errors_are_strictly_deterministic() {
        let valid_inputs = [
            build_minimal_compute_spv(),
            build_minimal_vertex_spv(),
            build_minimal_fragment_spv(),
        ];

        for spv in &valid_inputs {
            let baseline = convert_spirv_bytes(spv).expect("valid SPIR-V must convert");
            let audit_baseline = audit_spirv_file("sub/shader.spv", spv, &ConvertOptions::default());
            for _ in 0..5 {
                let again = convert_spirv_bytes(spv).expect("repeat conversion must succeed");
                assert_eq!(again, baseline);
                let audit_again =
                    audit_spirv_file("sub/shader.spv", spv, &ConvertOptions::default());
                assert_eq!(audit_again, audit_baseline);
            }
        }

        let err_inputs = [
            vec![0x03, 0x02, 0x23, 0x07],
            build_unsupported_extension_spv(),
            build_unsupported_capability_spv(true),
        ];
        for bad in &err_inputs {
            let baseline_err = convert_spirv_bytes(bad).expect_err("invalid SPIR-V must fail");
            for _ in 0..5 {
                let again_err = convert_spirv_bytes(bad).expect_err("repeat must fail identically");
                assert_eq!(again_err, baseline_err);
            }
        }
    }

    #[test]
    fn sanitize_diagnostic_text_strips_host_paths_ansi_control_chars_and_bounds_length() {
        // Windows drive paths (unquoted, quoted with spaces, forward slash, line:col, Rust Debug).
        assert_eq!(
            sanitize_diagnostic_text(r"failed to parse C:\Users\alice\Games\Secret\shader.spv"),
            "failed to parse <path>"
        );
        assert_eq!(
            sanitize_diagnostic_text(r#"error at "C:\Program Files\My Game\sky.hlsl":42:10"#),
            r#"error at "<path>":42:10"#
        );
        assert_eq!(
            sanitize_diagnostic_text("error at D:/Work/Repo/src/a.hlsl:12:5 in main"),
            "error at <path>:12:5 in main"
        );
        assert_eq!(
            sanitize_diagnostic_text(r#"UnsupportedExtension("C:\\Users\\alice\\ext.spv")"#),
            r#"UnsupportedExtension("<path>")"#
        );

        // UNC and extended-length Windows paths.
        assert_eq!(
            sanitize_diagnostic_text(r"cannot open \\fileserver\share\private\a.spv"),
            "cannot open <path>"
        );
        assert_eq!(
            sanitize_diagnostic_text(r"cannot open \\?\C:\Users\alice\a.spv"),
            "cannot open <path>"
        );

        // Unix absolute paths, file:// URIs, home paths, and parent traversals.
        assert_eq!(
            sanitize_diagnostic_text("error in /Users/alice/projects/game/shader.spv:8:1"),
            "error in <path>:8:1"
        );
        assert_eq!(
            sanitize_diagnostic_text("error in '/Users/alice/My Game/shader.spv'"),
            "error in '<path>'"
        );
        assert_eq!(
            sanitize_diagnostic_text("source file:///C:/Users/alice/game/a.spv."),
            "source <path>."
        );
        assert_eq!(
            sanitize_diagnostic_text("loaded ~/local-data/game/a.spv and ../../secret/b.spv"),
            "loaded <path> and <path>"
        );

        // ANSI escape sequences, CR/LF normalization, and NUL control char removal.
        assert_eq!(
            sanitize_diagnostic_text("\x1b[31merror\x1b[0m: /home/bob/game/a.spv\r\n\0line2"),
            "error: <path>\nline2"
        );

        // Clean compiler diagnostics and web URLs are preserved verbatim.
        let clean = "Required Capabilities(FLOAT64 | SUBGROUP) via SPV_KHR_ray_tracing (read/write, shader.wgsl:10:5, https://gpuweb.github.io/gpuweb/wgsl/)";
        assert_eq!(sanitize_diagnostic_text(clean), clean);

        // Unbounded diagnostic length is deterministically capped.
        let huge = "a".repeat(MAX_ERROR_DETAIL_CHARS + 500);
        let capped = sanitize_diagnostic_text(&huge);
        assert_eq!(capped.chars().count(), MAX_ERROR_DETAIL_CHARS + 3);
        assert!(capped.ends_with("..."));
    }

    #[test]
    fn spirv_conversion_and_audit_errors_redact_embedded_host_paths() {
        // 1. Host path embedded inside OpExtension.
        let mut b_ext = SpirvBuilder::new();
        let void_ty = b_ext.id();
        let fn_ty = b_ext.id();
        let main_fn = b_ext.id();
        let entry_label = b_ext.id();
        b_ext.inst(17, &[1]); // OpCapability Shader
        b_ext.inst_with_string(10, &[], r"C:\Users\alice\secret\custom_ext", &[]); // OpExtension
        b_ext.inst(14, &[0, 1]); // OpMemoryModel
        b_ext.inst_with_string(15, &[5, main_fn], "main", &[]);
        b_ext.inst(16, &[main_fn, 17, 1, 1, 1]);
        b_ext.inst(19, &[void_ty]);
        b_ext.inst(33, &[fn_ty, void_ty]);
        b_ext.inst(54, &[void_ty, main_fn, 0, fn_ty]);
        b_ext.inst(248, &[entry_label]);
        b_ext.inst(253, &[]);
        b_ext.inst(56, &[]);
        let ext_spv = b_ext.finish_bytes();

        let ext_err = convert_spirv_bytes(&ext_spv).expect_err("bogus extension must fail");
        assert_eq!(ext_err.kind, ConversionErrorKind::SpirvParseError);
        for field in [
            ext_err.detail.as_str(),
            ext_err.blocker_key.as_str(),
            &ext_err.to_string(),
        ] {
            assert!(!field.contains("C:\\"), "leaked Windows drive in: {field}");
            assert!(!field.contains("alice"), "leaked username in: {field}");
            assert!(!field.contains("secret"), "leaked directory in: {field}");
            assert!(field.contains("<path>"), "expected redacted token in: {field}");
        }
        assert!(ext_err.keywords.iter().all(|kw| !kw.contains("alice")));

        // 2. Host path embedded inside OpExtInstImport.
        let mut b_imp = SpirvBuilder::new();
        let ext_id = b_imp.id();
        let void_ty = b_imp.id();
        let fn_ty = b_imp.id();
        let main_fn = b_imp.id();
        let entry_label = b_imp.id();
        b_imp.inst(17, &[1]); // OpCapability Shader
        b_imp.inst_with_string(11, &[ext_id], "/Users/bob/private/ext_set", &[]); // OpExtInstImport
        b_imp.inst(14, &[0, 1]);
        b_imp.inst_with_string(15, &[5, main_fn], "main", &[]);
        b_imp.inst(16, &[main_fn, 17, 1, 1, 1]);
        b_imp.inst(19, &[void_ty]);
        b_imp.inst(33, &[fn_ty, void_ty]);
        b_imp.inst(54, &[void_ty, main_fn, 0, fn_ty]);
        b_imp.inst(248, &[entry_label]);
        b_imp.inst(253, &[]);
        b_imp.inst(56, &[]);
        let imp_spv = b_imp.finish_bytes();

        let imp_err = convert_spirv_bytes(&imp_spv).expect_err("bogus ext set must fail");
        assert!(!imp_err.detail.contains("/Users/bob"));
        assert!(!imp_err.blocker_key.contains("private"));
        assert!(imp_err.detail.contains("<path>"));

        // 3. Host path embedded inside OpName on a failing validation target (push constant without lowering).
        let mut b_name = SpirvBuilder::new();
        let void_ty = b_name.id();
        let fn_ty = b_name.id();
        let f32_ty = b_name.id();
        let vec4_ty = b_name.id();
        let struct_ty = b_name.id();
        let ptr_pc_ty = b_name.id();
        let pc_var = b_name.id();
        let main_fn = b_name.id();
        let entry_label = b_name.id();
        b_name.inst(17, &[1]); // OpCapability Shader
        b_name.inst(14, &[0, 1]); // OpMemoryModel
        b_name.inst_with_string(15, &[5, main_fn], "main", &[]);
        b_name.inst(16, &[main_fn, 17, 1, 1, 1]);
        b_name.inst_with_string(5, &[pc_var], r"C:\Users\alice\secret\pc.hlsl", &[]); // OpName
        b_name.inst(71, &[struct_ty, 2]); // Block
        b_name.inst(72, &[struct_ty, 0, 35, 0]); // MemberDecorate Offset 0
        b_name.inst(19, &[void_ty]);
        b_name.inst(33, &[fn_ty, void_ty]);
        b_name.inst(22, &[f32_ty, 32]);
        b_name.inst(23, &[vec4_ty, f32_ty, 4]);
        b_name.inst(30, &[struct_ty, vec4_ty]);
        b_name.inst(32, &[ptr_pc_ty, 9, struct_ty]);
        b_name.inst(59, &[ptr_pc_ty, pc_var, 9]);
        b_name.inst(54, &[void_ty, main_fn, 0, fn_ty]);
        b_name.inst(248, &[entry_label]);
        b_name.inst(253, &[]);
        b_name.inst(56, &[]);
        let name_spv = b_name.finish_bytes();

        let val_err = convert_spirv_bytes_with_options(
            &name_spv,
            &ConvertOptions::default().without_lowering(),
        )
        .expect_err("push constant without lowering must fail validation");
        assert_eq!(val_err.kind, ConversionErrorKind::ValidationError);
        assert!(!val_err.detail.contains("alice"), "leaked host path: {}", val_err.detail);
        assert!(!val_err.detail.contains("C:\\"), "leaked drive letter: {}", val_err.detail);

        // 4. audit_spirv_file sanitizes host source_path and error payloads.
        let host_paths = [
            r"C:\Users\alice\Games\Secret\foo.spv",
            "/Users/alice/secret/foo.spv",
            r"\\fileserver\share\dir\foo.spv",
            "file:///C:/Users/alice/foo.spv",
            "../../local-data/secret/foo.spv",
            "~/secret/foo.spv",
        ];
        for host in host_paths {
            let rec = audit_spirv_file(host, &ext_spv, &ConvertOptions::default());
            assert_eq!(rec.source_path, "foo.spv", "host path leaked from '{host}'");
            assert!(!rec.lowered_to_wgsl);
            let naga_err = rec.naga_error.as_deref().unwrap_or("");
            assert!(!naga_err.contains("alice"), "leaked in audit naga_error: {naga_err}");
        }

        // Clean relative paths stay relative and normalized.
        assert_eq!(sanitize_audit_source_path("sub/dir/foo.spv"), "sub/dir/foo.spv");
        assert_eq!(sanitize_audit_source_path("./sub//foo.spv"), "sub/foo.spv");
        assert_eq!(sanitize_audit_source_path(r"sub\dir\foo.spv"), "sub/dir/foo.spv");
        assert_eq!(sanitize_audit_source_path(r"C:\Users\alice\"), "unknown.spv");
        assert_eq!(sanitize_audit_source_path(""), "unknown.spv");
    }

    #[test]
    fn extract_feature_keywords_ignores_host_path_segments() {
        // Host path contains words like `task`, `mesh`, `f64`, `subgroup`, `SPV_secret` that
        // must NOT pollute extracted WebGPU blocker keywords.
        let spurious = extract_feature_keywords(
            "failed to read /Users/task/mesh_build/f64/subgroup/SPV_secret_shader.spv: generic error",
        );
        assert!(
            spurious.is_empty(),
            "host path segments must not trigger keywords, got: {spurious:?}"
        );

        let mixed = extract_feature_keywords(
            r"error in C:\Users\alice\task\shader.spv: Required Capabilities(FLOAT64) via SPV_KHR_ray_tracing",
        );
        assert!(mixed.iter().any(|k| k == "Capability::FLOAT64"));
        assert!(mixed.iter().any(|k| k == "SPV_KHR_ray_tracing"));
        assert!(mixed.iter().any(|k| k == "Float64"));
        assert!(!mixed.iter().any(|k| k == "TaskShader"));
    }
}
