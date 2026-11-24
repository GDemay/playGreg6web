//! Pure-Rust AMD RDNA2 (GFX10.3 / Xbox Series Scarlett) -> WebGPU WGSL shader translator
//! with built-in Naga validation.
//!
//! Decodes RDNA2 machine-code instruction streams via [`crate::rdna2`], lowers the
//! register-machine ISA (`VGPR`, `SGPR`, `VCC`, `SCC`, `SMEM`, `VINTRP`, `MIMG`, `EXP`,
//! `VALU`, `SALU`) into a baseline-WebGPU [`naga::Module`], validates it unconditionally
//! with [`naga::valid::Validator`] (`Capabilities::empty()`), and emits clean WGSL text
//! via [`naga::back::wgsl::write_string`].
//!
//! Guarantees:
//! * Zero `unsafe` code.
//! * Panic-free on empty, misaligned, truncated, or arbitrary fuzz inputs.
//! * Strictly sanitized outputs (no host paths, no raw shader byte dumps).
//! * Graceful fallback for unsupported opcodes: emits sanitized WGSL comments and
//!   increments `unsupported_op_count` while keeping the generated WGSL valid under Naga.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::num::NonZeroU32;

use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    AddressSpace, ArraySize, BinaryOperator, Binding, BuiltIn, EntryPoint, Expression, Function,
    FunctionArgument, FunctionResult, GlobalVariable, Handle, ImageClass, ImageDimension,
    Interpolation, Literal, LocalVariable, MathFunction, MemoryDecorations, Module, Range,
    ResourceBinding, SampleLevel, Sampling, Scalar, ScalarKind, ShaderStage as NagaStage, Span,
    Statement, StructMember, Type, TypeInner, UnaryOperator, VectorSize,
};

use crate::convert::{ShaderStage, format_shader_hash};
use crate::pack::write_json_string;
use crate::rdna2::{
    Rdna2Detail, Rdna2ExportTarget, Rdna2InferredStage, Rdna2Instruction, Rdna2ProgramSummary,
    analyze_instructions, decode_program,
};

/// Default minimum number of `vec4<u32>` slots in the synthesized uniform CBV (`cb0`).
pub const DEFAULT_MIN_CBV_VEC4_COUNT: u32 = 16;

/// Upper bound on `vec4<u32>` slots in `cb0` (1024 * 16B = 16 KiB, within WebGPU 64 KiB baseline).
pub const MAX_CBV_VEC4_COUNT: u32 = 1024;

/// Maximum inter-stage location count under WebGPU baseline limits.
const MAX_WEBGPU_LOCATIONS: u8 = 16;

/// Maximum color attachment count under WebGPU baseline limits.
const MAX_WEBGPU_MRTS: u8 = 8;

/// Stage override option for [`translate_rdna2_to_wgsl`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Rdna2StageOverride {
    #[default]
    Auto,
    Vertex,
    Pixel,
    Compute,
}

impl Rdna2StageOverride {
    /// Parse a CLI stage flag value (`auto`, `vertex`, `pixel`, `fragment`, `compute`).
    pub fn parse_str(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "vertex" | "vs" => Ok(Self::Vertex),
            "pixel" | "fragment" | "ps" | "fs" => Ok(Self::Pixel),
            "compute" | "cs" => Ok(Self::Compute),
            other => Err(format!(
                "invalid --stage value '{other}' (expected 'auto', 'vertex', 'pixel', or 'compute')"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Vertex => "vertex",
            Self::Pixel => "pixel",
            Self::Compute => "compute",
        }
    }
}

impl fmt::Display for Rdna2StageOverride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Options controlling RDNA2 -> WGSL translation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2TranslateOptions {
    /// Pipeline stage selection (`Auto` infers from `EXP`/`VINTRP`/memory instructions).
    pub stage: Rdna2StageOverride,
    /// Optional explicit WGSL entry-point name (defaults to `vs_main`/`fs_main`/`cs_main`).
    pub entry_point: Option<String>,
    /// Minimum number of `vec4<u32>` entries emitted when `cb0` uniform buffer is needed.
    pub min_cbv_vec4_count: u32,
}

impl Default for Rdna2TranslateOptions {
    fn default() -> Self {
        Self {
            stage: Rdna2StageOverride::Auto,
            entry_point: None,
            min_cbv_vec4_count: DEFAULT_MIN_CBV_VEC4_COUNT,
        }
    }
}

/// Explicit WebGPU `@group(group) @binding(binding)` resource declaration emitted in a
/// translated WGSL module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WgslBindingInfo {
    pub group: u32,
    pub binding: u32,
    pub kind: &'static str,
    pub name: String,
}

/// Write a JSON array of [`WgslBindingInfo`] entries with the given leading indentation.
pub(crate) fn write_wgsl_bindings_json(
    out: &mut String,
    bindings: &[WgslBindingInfo],
    indent: &str,
) {
    if bindings.is_empty() {
        out.push_str("[]");
        return;
    }
    out.push_str("[\n");
    for (i, b) in bindings.iter().enumerate() {
        let _ = write!(
            out,
            "{indent}  {{\"group\": {}, \"binding\": {}, \"kind\": ",
            b.group, b.binding
        );
        write_json_string(out, b.kind);
        out.push_str(", \"name\": ");
        write_json_string(out, &b.name);
        out.push('}');
        if i + 1 < bindings.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str(indent);
    out.push(']');
}

/// Summary of WebGPU bindings and stage I/O synthesized for a translated RDNA2 shader.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Rdna2BindingSummary {
    pub has_cbv: bool,
    pub cbv_vec4_count: u32,
    pub has_texture_2d: bool,
    pub has_sampler: bool,
    pub vertex_input_mode: String,
    pub fragment_input_locations: Vec<u32>,
    pub output_locations: Vec<u32>,
    pub exports_position: bool,
    pub exports_depth: bool,
}

/// Result of translating an RDNA2 binary shader into validated WebGPU WGSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2WgslOutput {
    pub wgsl: String,
    pub stage: ShaderStage,
    pub entry_point: String,
    pub hash_fnv1a64: u64,
    pub translated_op_count: usize,
    pub unsupported_op_count: usize,
    pub naga_validated: bool,
    pub bindings: Vec<WgslBindingInfo>,
    pub binding_summary: Rdna2BindingSummary,
}

impl Rdna2WgslOutput {
    /// Render a strictly sanitized deterministic JSON summary (zero host paths, zero raw bytes).
    pub fn to_json(&self) -> String {
        self.to_json_with_source(None)
    }

    /// Render a strictly sanitized deterministic JSON summary with an optional sanitized source basename.
    pub fn to_json_with_source(&self, source_name: Option<&str>) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str("  \"isa\": \"rdna2_gfx10_3\",\n");
        if let Some(src) = source_name {
            out.push_str("  \"source\": ");
            write_json_string(&mut out, src);
            out.push_str(",\n");
        }
        out.push_str("  \"hash_fnv1a64\": ");
        write_json_string(&mut out, &format_shader_hash(self.hash_fnv1a64));
        out.push_str(",\n");
        out.push_str("  \"stage\": ");
        write_json_string(&mut out, self.stage.as_str());
        out.push_str(",\n");
        out.push_str("  \"entry_point\": ");
        write_json_string(&mut out, &self.entry_point);
        out.push_str(",\n");
        let _ = writeln!(
            out,
            "  \"translated_op_count\": {},",
            self.translated_op_count
        );
        let _ = writeln!(
            out,
            "  \"unsupported_op_count\": {},",
            self.unsupported_op_count
        );
        let _ = writeln!(out, "  \"naga_validated\": {},", self.naga_validated);
        let _ = writeln!(out, "  \"wgsl_byte_len\": {},", self.wgsl.len());

        // Explicit WGSL @group/@binding map
        out.push_str("  \"bindings\": ");
        write_wgsl_bindings_json(&mut out, &self.bindings, "  ");
        out.push_str(",\n");

        // binding_summary
        let bs = &self.binding_summary;
        out.push_str("  \"binding_summary\": {\n");
        let _ = writeln!(out, "    \"has_cbv\": {},", bs.has_cbv);
        let _ = writeln!(out, "    \"cbv_vec4_count\": {},", bs.cbv_vec4_count);
        let _ = writeln!(out, "    \"has_texture_2d\": {},", bs.has_texture_2d);
        let _ = writeln!(out, "    \"has_sampler\": {},", bs.has_sampler);
        out.push_str("    \"vertex_input_mode\": ");
        write_json_string(&mut out, &bs.vertex_input_mode);
        out.push_str(",\n");

        out.push_str("    \"fragment_input_locations\": [");
        for (i, loc) in bs.fragment_input_locations.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{loc}");
        }
        out.push_str("],\n");

        out.push_str("    \"output_locations\": [");
        for (i, loc) in bs.output_locations.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            let _ = write!(out, "{loc}");
        }
        out.push_str("],\n");

        let _ = writeln!(out, "    \"exports_position\": {},", bs.exports_position);
        let _ = writeln!(out, "    \"exports_depth\": {}", bs.exports_depth);
        out.push_str("  }\n");
        out.push_str("}\n");
        out
    }
}

/// Errors produced during RDNA2 -> WGSL translation or Naga verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rdna2TranslateError {
    NagaValidation(String),
    WgslEmit(String),
    WgslParse(String),
}

impl fmt::Display for Rdna2TranslateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NagaValidation(msg) => write!(f, "Naga WGSL validation failed: {msg}"),
            Self::WgslEmit(msg) => write!(f, "Naga WGSL emission failed: {msg}"),
            Self::WgslParse(msg) => write!(f, "Naga WGSL parse check failed: {msg}"),
        }
    }
}

impl std::error::Error for Rdna2TranslateError {}

// ---------------------------------------------------------------------------
// Pre-pass program planning
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VertexInputMode {
    None,
    VertexIndex,
    Attributes,
}

impl VertexInputMode {
    const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::VertexIndex => "vertex_index",
            Self::Attributes => "attributes",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct VertexInputAttr {
    /// Byte offset inside the vertex buffer for MUBUF/MTBUF fetches (`None` for legacy synthetic attributes).
    byte_offset: Option<u16>,
    location: u32,
    name: String,
    /// Vector component width (`1` -> `f32`, `2` -> `vec2<f32>`, `3` -> `vec3<f32>`, `4` -> `vec4<f32>`).
    components: u8,
}

struct ProgramPlan {
    stage: ShaderStage,
    entry_point: String,
    active_inst_count: usize,
    vertex_input_mode: VertexInputMode,
    vertex_attrs: Vec<VertexInputAttr>,
    has_buffer_fetch: bool,
    legacy_uv_attr_index: Option<usize>,
    uses_cbv: bool,
    cbv_vec4_count: u32,
    uses_mimg: bool,
    frag_attrs: Vec<u8>,
    vs_params: Vec<u8>,
    ps_mrts: Vec<u8>,
    ps_exports_depth: bool,
    vgprs: BTreeSet<u8>,
    sgprs: BTreeSet<u8>,
    needs_vcc: bool,
    needs_scc: bool,
}

fn resolve_stage(inferred: Rdna2InferredStage, override_stage: Rdna2StageOverride) -> ShaderStage {
    match override_stage {
        Rdna2StageOverride::Vertex => ShaderStage::Vertex,
        Rdna2StageOverride::Pixel => ShaderStage::Fragment,
        Rdna2StageOverride::Compute => ShaderStage::Compute,
        Rdna2StageOverride::Auto => match inferred {
            Rdna2InferredStage::Vertex => ShaderStage::Vertex,
            Rdna2InferredStage::Pixel => ShaderStage::Fragment,
            Rdna2InferredStage::Compute | Rdna2InferredStage::Unknown => ShaderStage::Compute,
        },
    }
}

fn default_entry_point(stage: ShaderStage) -> &'static str {
    match stage {
        ShaderStage::Vertex => "vs_main",
        ShaderStage::Fragment => "fs_main",
        ShaderStage::Compute => "cs_main",
    }
}

fn sanitize_wgsl_ident(raw: &str, fallback: &str) -> String {
    let mut out = String::new();
    for (i, ch) in raw.chars().enumerate() {
        let valid = if i == 0 {
            ch.is_ascii_alphabetic() || ch == '_'
        } else {
            ch.is_ascii_alphanumeric() || ch == '_'
        };
        if valid {
            out.push(ch);
        } else if !out.is_empty() {
            out.push('_');
        }
    }
    if out.is_empty() || out == "_" {
        fallback.to_string()
    } else {
        out
    }
}

fn smem_load_dword_count(op: u8) -> Option<u8> {
    match op {
        0x00 | 0x08 => Some(1),
        0x01 | 0x09 => Some(2),
        0x02 | 0x0A => Some(4),
        0x03 | 0x0B => Some(8),
        0x04 | 0x0C => Some(16),
        _ => None,
    }
}

fn is_supported_mimg_op(op: u8) -> bool {
    matches!(
        op,
        0x00 | 0x01 | 0x20 | 0x21 | 0x22 | 0x24 | 0x25 | 0x27 | 0x30 | 0x34 | 0x37
    )
}

fn mimg_addr_vgpr_count(op: u8) -> u8 {
    match op {
        0x01 | 0x24 | 0x34 => 3, // u, v, lod/mip
        _ => 2,                  // u, v
    }
}

fn mubuf_load_components(op: u8) -> Option<u8> {
    match op {
        0x00 | 0x0C => Some(1),
        0x01 | 0x0D => Some(2),
        0x02 | 0x0F => Some(3),
        0x03 | 0x0E => Some(4),
        _ => None,
    }
}

fn mtbuf_load_components(op: u8) -> Option<u8> {
    match op {
        0x00 => Some(1),
        0x01 => Some(2),
        0x02 => Some(3),
        0x03 => Some(4),
        _ => None,
    }
}

fn record_src_usage(
    src: u16,
    vgprs: &mut BTreeSet<u8>,
    sgprs: &mut BTreeSet<u8>,
    written_vgprs: &BTreeSet<u8>,
    read_before_write_vgprs: &mut BTreeSet<u8>,
    needs_vcc: &mut bool,
    needs_scc: &mut bool,
) {
    match src {
        0..=105 => {
            sgprs.insert(src as u8);
        }
        106 => {
            *needs_vcc = true;
        }
        251 => {
            *needs_vcc = true;
        }
        253 => {
            *needs_scc = true;
        }
        256..=511 => {
            let v = (src - 256) as u8;
            vgprs.insert(v);
            if !written_vgprs.contains(&v) {
                read_before_write_vgprs.insert(v);
            }
        }
        _ => {}
    }
}

fn plan_program(
    summary: &Rdna2ProgramSummary,
    instructions: &[Rdna2Instruction],
    options: &Rdna2TranslateOptions,
) -> ProgramPlan {
    let stage = resolve_stage(summary.inferred_stage, options.stage);
    let entry_point = match &options.entry_point {
        Some(custom) => sanitize_wgsl_ident(custom, default_entry_point(stage)),
        None => default_entry_point(stage).to_string(),
    };

    // Determine effective instruction slice (stop at first s_endpgm if no branch targets past it).
    let max_branch_target = summary
        .branches
        .iter()
        .map(|b| b.target_offset)
        .max()
        .unwrap_or(-1);
    let mut active_inst_count = instructions.len();
    for (idx, inst) in instructions.iter().enumerate() {
        if let Rdna2Detail::Sopp { op, .. } = inst.detail
            && matches!(op, 0x01 | 0x1B | 0x1E)
            && max_branch_target <= inst.offset as i64
        {
            active_inst_count = idx + 1;
            break;
        }
    }

    let active = &instructions[..active_inst_count];

    let mut uses_cbv = false;
    let mut max_cbv_dword: u32 = 0;
    let mut uses_mimg = false;
    let mut has_buffer_fetch = false;
    let mut vertex_buf_fetches: BTreeMap<u16, u8> = BTreeMap::new();
    let mut frag_attr_set: BTreeSet<u8> = BTreeSet::new();
    let mut vs_param_set: BTreeSet<u8> = BTreeSet::new();
    let mut ps_mrt_set: BTreeSet<u8> = BTreeSet::new();
    let mut ps_exports_depth = false;

    let mut vgprs: BTreeSet<u8> = BTreeSet::new();
    let mut sgprs: BTreeSet<u8> = BTreeSet::new();
    let mut written_vgprs: BTreeSet<u8> = BTreeSet::new();
    let mut read_before_write_vgprs: BTreeSet<u8> = BTreeSet::new();
    let mut needs_vcc = false;
    let mut needs_scc = false;

    for inst in active {
        if inst.truncated {
            continue;
        }
        match &inst.detail {
            Rdna2Detail::Sopc { ssrc0, ssrc1, .. } => {
                record_src_usage(
                    u16::from(*ssrc0),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    u16::from(*ssrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                needs_scc = true;
            }
            Rdna2Detail::Sop1 {
                op, sdst, ssrc0, ..
            } => {
                record_src_usage(
                    u16::from(*ssrc0),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if *op == 0x04 {
                    // s_mov_b64: 64-bit SGPR pair source and destination
                    if *ssrc0 <= 104 {
                        sgprs.insert(*ssrc0 + 1);
                    }
                    if *sdst <= 104 {
                        sgprs.insert(*sdst);
                        sgprs.insert(*sdst + 1);
                    } else if *sdst == 105 {
                        sgprs.insert(105);
                    } else if *sdst == 106 {
                        needs_vcc = true;
                    }
                } else {
                    if *op == 0x05 {
                        needs_scc = true;
                    }
                    if *sdst <= 105 {
                        sgprs.insert(*sdst);
                    }
                }
            }
            Rdna2Detail::Sopk { sdst, .. } => {
                if *sdst <= 105 {
                    sgprs.insert(*sdst);
                }
            }
            Rdna2Detail::Sop2 {
                op,
                sdst,
                ssrc0,
                ssrc1,
                ..
            } => {
                record_src_usage(
                    u16::from(*ssrc0),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    u16::from(*ssrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if *op == 0x0A {
                    needs_scc = true;
                }
                if *sdst <= 105 {
                    sgprs.insert(*sdst);
                }
            }
            Rdna2Detail::Smem {
                op,
                sdata,
                soffset,
                offset,
                ..
            } => {
                if let Some(count) = smem_load_dword_count(*op) {
                    uses_cbv = true;
                    if *soffset != 0x7D && *soffset <= 105 {
                        sgprs.insert(*soffset);
                    }
                    let base_word = if *offset > 0 {
                        (*offset as u32) / 4
                    } else {
                        0
                    };
                    max_cbv_dword = max_cbv_dword.max(base_word.saturating_add(u32::from(count)));
                    for k in 0..count {
                        let reg = sdata.saturating_add(k);
                        if reg <= 105 {
                            sgprs.insert(reg);
                        }
                    }
                }
            }
            Rdna2Detail::Vop1 { op, vdst, src0, .. } => {
                if !matches!(op, 0x00 | 0x1B | 0x41) {
                    record_src_usage(
                        *src0,
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                    if *op == 0x02 {
                        let sd = *vdst & 0x7F;
                        if sd <= 105 {
                            sgprs.insert(sd);
                        }
                    } else {
                        vgprs.insert(*vdst);
                        written_vgprs.insert(*vdst);
                    }
                }
            }
            Rdna2Detail::Vopc { vsrc1, src0, .. } => {
                record_src_usage(
                    *src0,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    256 + u16::from(*vsrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                needs_vcc = true;
            }
            Rdna2Detail::Vop2 {
                op,
                vdst,
                vsrc1,
                src0,
                ..
            } => {
                record_src_usage(
                    *src0,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    256 + u16::from(*vsrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if matches!(op, 0x06 | 0x2B | 0x36) {
                    // v_fmac* reads vdst as accumulator
                    record_src_usage(
                        256 + u16::from(*vdst),
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
                if matches!(op, 0x00 | 0x01) {
                    needs_vcc = true;
                }
                vgprs.insert(*vdst);
                written_vgprs.insert(*vdst);
            }
            Rdna2Detail::Vop3 {
                op,
                vdst,
                src0,
                src1,
                src2,
                ..
            } => {
                record_src_usage(
                    *src0,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                let is_unary = (0x180..=0x1FF).contains(op);
                if !is_unary {
                    record_src_usage(
                        *src1,
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
                let is_ternary = matches!(
                    *op,
                    0x100
                        | 0x101
                        | 0x140..=0x165
                        | 0x16D..=0x171
                        | 0x340..=0x35F
                        | 0x36D..=0x370
                );
                if is_ternary {
                    record_src_usage(
                        *src2,
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
                if *op <= 0xFF {
                    let sd = *vdst & 0x7F;
                    if sd <= 105 {
                        sgprs.insert(sd);
                    } else if sd == 106 {
                        needs_vcc = true;
                    }
                } else {
                    vgprs.insert(*vdst);
                    written_vgprs.insert(*vdst);
                }
            }
            Rdna2Detail::Vop3P {
                vdst,
                src0,
                src1,
                src2,
                ..
            } => {
                record_src_usage(
                    *src0,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    *src1,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    *src2,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                vgprs.insert(*vdst);
                written_vgprs.insert(*vdst);
            }
            Rdna2Detail::Vintrp { vdst, attr, .. } => {
                if stage == ShaderStage::Fragment {
                    frag_attr_set.insert(*attr % MAX_WEBGPU_LOCATIONS);
                }
                vgprs.insert(*vdst);
                written_vgprs.insert(*vdst);
            }
            Rdna2Detail::Mimg {
                op,
                dmask,
                vaddr,
                vdata,
                ..
            } => {
                if is_supported_mimg_op(*op) {
                    uses_mimg = true;
                    let addr_cnt = mimg_addr_vgpr_count(*op);
                    for a in 0..addr_cnt {
                        let reg = vaddr.saturating_add(a);
                        vgprs.insert(reg);
                        if !written_vgprs.contains(&reg) {
                            read_before_write_vgprs.insert(reg);
                        }
                    }
                    let eff_mask = if *dmask == 0 { 0xF } else { *dmask };
                    let out_cnt = eff_mask.count_ones() as u8;
                    for d in 0..out_cnt {
                        let reg = vdata.saturating_add(d);
                        vgprs.insert(reg);
                        written_vgprs.insert(reg);
                    }
                }
            }
            Rdna2Detail::Mubuf {
                op, offset, vdata, ..
            } => {
                if let Some(comps) = mubuf_load_components(*op) {
                    has_buffer_fetch = true;
                    vertex_buf_fetches
                        .entry(*offset)
                        .and_modify(|c| *c = (*c).max(comps))
                        .or_insert(comps);
                    for d in 0..comps {
                        let reg = vdata.saturating_add(d);
                        vgprs.insert(reg);
                        written_vgprs.insert(reg);
                    }
                }
            }
            Rdna2Detail::Mtbuf {
                op, offset, vdata, ..
            } => {
                if let Some(comps) = mtbuf_load_components(*op) {
                    has_buffer_fetch = true;
                    vertex_buf_fetches
                        .entry(*offset)
                        .and_modify(|c| *c = (*c).max(comps))
                        .or_insert(comps);
                    for d in 0..comps {
                        let reg = vdata.saturating_add(d);
                        vgprs.insert(reg);
                        written_vgprs.insert(reg);
                    }
                }
            }
            Rdna2Detail::Exp {
                target,
                en,
                compr,
                vsrc,
                ..
            } => {
                if *compr {
                    if (*en & 0b0011) != 0 {
                        vgprs.insert(vsrc[0]);
                        if !written_vgprs.contains(&vsrc[0]) {
                            read_before_write_vgprs.insert(vsrc[0]);
                        }
                    }
                    if (*en & 0b1100) != 0 {
                        vgprs.insert(vsrc[1]);
                        if !written_vgprs.contains(&vsrc[1]) {
                            read_before_write_vgprs.insert(vsrc[1]);
                        }
                    }
                } else {
                    for (idx, &reg) in vsrc.iter().enumerate() {
                        if (*en & (1 << idx)) != 0 {
                            vgprs.insert(reg);
                            if !written_vgprs.contains(&reg) {
                                read_before_write_vgprs.insert(reg);
                            }
                        }
                    }
                }
                match target {
                    Rdna2ExportTarget::Param(n) if stage == ShaderStage::Vertex => {
                        // Reserve @builtin(position) + up to 15 param locations.
                        vs_param_set.insert(*n % (MAX_WEBGPU_LOCATIONS - 1));
                    }
                    Rdna2ExportTarget::Mrt(n) if stage == ShaderStage::Fragment => {
                        if *n < MAX_WEBGPU_MRTS {
                            ps_mrt_set.insert(*n);
                        }
                    }
                    Rdna2ExportTarget::MrtZ if stage == ShaderStage::Fragment => {
                        ps_exports_depth = true;
                    }
                    _ => {}
                }
            }
            Rdna2Detail::Sopp { .. }
            | Rdna2Detail::Flat { .. }
            | Rdna2Detail::Ds { .. }
            | Rdna2Detail::Unknown { .. } => {}
        }
    }

    let mut vertex_attrs: Vec<VertexInputAttr> = Vec::new();
    let mut legacy_uv_attr_index: Option<usize> = None;
    let vertex_input_mode = if stage == ShaderStage::Vertex {
        let multi_vgpr_input = read_before_write_vgprs.iter().any(|&v| v >= 1);
        if has_buffer_fetch || multi_vgpr_input {
            if vertex_buf_fetches.is_empty() {
                // Pure multi-VGPR attribute input without MUBUF/MTBUF fetches.
                vertex_attrs.push(VertexInputAttr {
                    byte_offset: None,
                    location: 0,
                    name: "position".to_string(),
                    components: 4,
                });
                legacy_uv_attr_index = Some(vertex_attrs.len());
                vertex_attrs.push(VertexInputAttr {
                    byte_offset: None,
                    location: 1,
                    name: "uv".to_string(),
                    components: 2,
                });
            } else {
                // Map MUBUF/MTBUF vertex fetches deterministically by ascending byte offset:
                //   offset 0  -> @location(0) position: vec3<f32> (3-comp) or vec4<f32>
                //   offset 12 -> @location(1) color: vec4<f32> (R8G8B8A8_UNORM at +12B)
                //   offset 16 -> @location(2) uv: vec2<f32> (R32G32_FLOAT at +16B)
                //   any other offset -> deterministic sequential @location(N) attr_<offset>
                let mut next_loc: u32 = 0;
                for (&byte_offset, &max_comps) in &vertex_buf_fetches {
                    let (location, name, components) = match byte_offset {
                        0 => {
                            let loc = next_loc;
                            next_loc = loc + 1;
                            let comps = if max_comps == 3 { 3 } else { 4 };
                            (loc, "position".to_string(), comps)
                        }
                        12 => {
                            let loc = next_loc.max(1);
                            next_loc = loc + 1;
                            (loc, "color".to_string(), 4)
                        }
                        16 => {
                            let loc = next_loc.max(2);
                            next_loc = loc + 1;
                            (loc, "uv".to_string(), 2)
                        }
                        other => {
                            let loc = next_loc;
                            next_loc = loc + 1;
                            (loc, format!("attr_{other}"), max_comps.clamp(1, 4))
                        }
                    };
                    if location < u32::from(MAX_WEBGPU_LOCATIONS) {
                        vertex_attrs.push(VertexInputAttr {
                            byte_offset: Some(byte_offset),
                            location,
                            name,
                            components,
                        });
                    }
                }
                // Preserve legacy multi-VGPR v4/v5 UV input when the shader reads v4 or v5
                // before writing and has no explicit offset-16 UV buffer fetch.
                let reads_legacy_uv =
                    read_before_write_vgprs.contains(&4) || read_before_write_vgprs.contains(&5);
                if reads_legacy_uv
                    && !vertex_buf_fetches.contains_key(&16)
                    && next_loc < u32::from(MAX_WEBGPU_LOCATIONS)
                {
                    legacy_uv_attr_index = Some(vertex_attrs.len());
                    vertex_attrs.push(VertexInputAttr {
                        byte_offset: None,
                        location: next_loc,
                        name: "uv".to_string(),
                        components: 2,
                    });
                }
            }
            VertexInputMode::Attributes
        } else {
            vgprs.insert(0);
            VertexInputMode::VertexIndex
        }
    } else {
        VertexInputMode::None
    };

    if stage == ShaderStage::Compute {
        vgprs.insert(0);
    }

    let mut ps_mrts: Vec<u8> = ps_mrt_set.into_iter().collect();
    if stage == ShaderStage::Fragment && ps_mrts.is_empty() {
        ps_mrts.push(0);
    }

    let min_vec4s = options.min_cbv_vec4_count.clamp(1, MAX_CBV_VEC4_COUNT);
    let needed_vec4s = max_cbv_dword.div_ceil(4).max(min_vec4s);
    let cbv_vec4_count = if uses_cbv {
        needed_vec4s.min(MAX_CBV_VEC4_COUNT)
    } else {
        0
    };

    ProgramPlan {
        stage,
        entry_point,
        active_inst_count,
        vertex_input_mode,
        vertex_attrs,
        has_buffer_fetch,
        legacy_uv_attr_index,
        uses_cbv,
        cbv_vec4_count,
        uses_mimg,
        frag_attrs: frag_attr_set.into_iter().collect(),
        vs_params: vs_param_set.into_iter().collect(),
        ps_mrts,
        ps_exports_depth,
        vgprs,
        sgprs,
        needs_vcc,
        needs_scc,
    }
}

// ---------------------------------------------------------------------------
// Naga IR builder
// ---------------------------------------------------------------------------

struct CommonTypes {
    vec2_f32_ty: Handle<Type>,
    vec2_i32_ty: Handle<Type>,
    vec4_f32_ty: Handle<Type>,
}

struct IrEmitter {
    module: Module,
    types: CommonTypes,
    cb0_global: Option<Handle<GlobalVariable>>,
    cb0_vec4_count: u32,
    tex0_global: Option<Handle<GlobalVariable>>,
    samp0_global: Option<Handle<GlobalVariable>>,
    stage: ShaderStage,
    func: Function,
    vgpr_locals: BTreeMap<u8, Handle<LocalVariable>>,
    sgpr_locals: BTreeMap<u8, Handle<LocalVariable>>,
    vcc_local: Option<Handle<LocalVariable>>,
    scc_local: Option<Handle<LocalVariable>>,
    vs_out_local: Option<Handle<LocalVariable>>,
    vs_param_member_index: BTreeMap<u8, u32>,
    ps_out_local: Option<Handle<LocalVariable>>,
    ps_mrt_member_index: BTreeMap<u8, u32>,
    ps_depth_member_index: Option<u32>,
    frag_in_arg: Option<u32>,
    frag_attr_member_index: BTreeMap<u8, u32>,
    vs_in_arg: Option<u32>,
    /// Map from vertex buffer byte offset -> `(VsIn member_index, attr_components)`.
    vs_attr_by_offset: BTreeMap<u16, (u32, u8)>,
    has_buffer_fetch: bool,
    legacy_uv_member_index: Option<u32>,
    vertex_input_mode: VertexInputMode,
    translated_op_count: usize,
    unsupported_op_count: usize,
    unsupported_comments: Vec<String>,
}

fn align_up(value: u32, align: u32) -> u32 {
    if align <= 1 {
        return value;
    }
    (value + (align - 1)) & !(align - 1)
}

fn insert_type(module: &mut Module, name: Option<String>, inner: TypeInner) -> Handle<Type> {
    module
        .types
        .insert(Type { name, inner }, Span::UNDEFINED)
}

impl IrEmitter {
    fn new(plan: &ProgramPlan) -> Self {
        let mut module = Module::default();
        let f32_ty = insert_type(&mut module, None, TypeInner::Scalar(Scalar::F32));
        let u32_ty = insert_type(&mut module, None, TypeInner::Scalar(Scalar::U32));
        let bool_ty = insert_type(&mut module, None, TypeInner::Scalar(Scalar::BOOL));
        let vec2_f32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::F32,
            },
        );
        let vec2_i32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::I32,
            },
        );
        let vec3_f32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::F32,
            },
        );
        let vec3_u32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::U32,
            },
        );
        let vec4_f32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::F32,
            },
        );
        let vec4_u32_ty = insert_type(
            &mut module,
            None,
            TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::U32,
            },
        );

        let types = CommonTypes {
            vec2_f32_ty,
            vec2_i32_ty,
            vec4_f32_ty,
        };

        // Optional @group(0) @binding(0) var<uniform> cb0: Rdna2Cbv;
        let cb0_global = if plan.uses_cbv {
            let count = plan.cbv_vec4_count.max(1);
            let arr_ty = insert_type(
                &mut module,
                None,
                TypeInner::Array {
                    base: vec4_u32_ty,
                    size: ArraySize::Constant(
                        NonZeroU32::new(count).unwrap_or(NonZeroU32::MIN),
                    ),
                    stride: 16,
                },
            );
            let cbv_struct_ty = insert_type(
                &mut module,
                Some("Rdna2Cbv".to_string()),
                TypeInner::Struct {
                    members: vec![StructMember {
                        name: Some("data".to_string()),
                        ty: arr_ty,
                        binding: None,
                        offset: 0,
                    }],
                    span: count.saturating_mul(16),
                },
            );
            Some(module.global_variables.append(
                GlobalVariable {
                    name: Some("cb0".to_string()),
                    space: AddressSpace::Uniform,
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 0,
                    }),
                    ty: cbv_struct_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            ))
        } else {
            None
        };

        // Optional @group(0) @binding(1) tex0 + @group(0) @binding(2) samp0
        let (tex0_global, samp0_global) = if plan.uses_mimg {
            let tex_ty = insert_type(
                &mut module,
                None,
                TypeInner::Image {
                    dim: ImageDimension::D2,
                    arrayed: false,
                    class: ImageClass::Sampled {
                        kind: ScalarKind::Float,
                        multi: false,
                    },
                },
            );
            let samp_ty = insert_type(
                &mut module,
                None,
                TypeInner::Sampler { comparison: false },
            );
            let tex_g = module.global_variables.append(
                GlobalVariable {
                    name: Some("tex0".to_string()),
                    space: AddressSpace::Handle,
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 1,
                    }),
                    ty: tex_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            );
            let samp_g = module.global_variables.append(
                GlobalVariable {
                    name: Some("samp0".to_string()),
                    space: AddressSpace::Handle,
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: 2,
                    }),
                    ty: samp_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            );
            (Some(tex_g), Some(samp_g))
        } else {
            (None, None)
        };

        let mut func = Function {
            name: Some(plan.entry_point.clone()),
            ..Function::default()
        };

        let mut vs_in_arg = None;
        let mut vs_attr_by_offset: BTreeMap<u16, (u32, u8)> = BTreeMap::new();
        let mut legacy_uv_member_index: Option<u32> = None;
        let mut frag_in_arg = None;
        let mut frag_attr_member_index = BTreeMap::new();
        let mut vs_out_local = None;
        let mut vs_param_member_index = BTreeMap::new();
        let mut ps_out_local = None;
        let mut ps_mrt_member_index = BTreeMap::new();
        let mut ps_depth_member_index = None;

        // Stage inputs and outputs
        match plan.stage {
            ShaderStage::Vertex => {
                match plan.vertex_input_mode {
                    VertexInputMode::VertexIndex => {
                        func.arguments.push(FunctionArgument {
                            name: Some("vertex_index".to_string()),
                            ty: u32_ty,
                            binding: Some(Binding::BuiltIn(BuiltIn::VertexIndex)),
                        });
                        vs_in_arg = Some(0);
                    }
                    VertexInputMode::Attributes => {
                        let mut members = vec![StructMember {
                            name: Some("vertex_index".to_string()),
                            ty: u32_ty,
                            binding: Some(Binding::BuiltIn(BuiltIn::VertexIndex)),
                            offset: 0,
                        }];
                        let mut cur_offset = 4u32;
                        let mut max_align = 4u32;
                        for (attr_idx, attr) in plan.vertex_attrs.iter().enumerate() {
                            let (ty, align, size) = match attr.components {
                                1 => (f32_ty, 4u32, 4u32),
                                2 => (vec2_f32_ty, 8u32, 8u32),
                                3 => (vec3_f32_ty, 16u32, 12u32),
                                _ => (vec4_f32_ty, 16u32, 16u32),
                            };
                            cur_offset = align_up(cur_offset, align);
                            max_align = max_align.max(align);
                            let member_idx = members.len() as u32;
                            if let Some(byte_off) = attr.byte_offset {
                                vs_attr_by_offset.insert(byte_off, (member_idx, attr.components));
                            }
                            if plan.legacy_uv_attr_index == Some(attr_idx) {
                                legacy_uv_member_index = Some(member_idx);
                            }
                            members.push(StructMember {
                                name: Some(attr.name.clone()),
                                ty,
                                binding: Some(Binding::Location {
                                    location: attr.location,
                                    interpolation: None,
                                    sampling: None,
                                    blend_src: None,
                                    per_primitive: false,
                                }),
                                offset: cur_offset,
                            });
                            cur_offset += size;
                        }
                        let span = align_up(cur_offset, max_align);
                        let vs_in_ty = insert_type(
                            &mut module,
                            Some("VsIn".to_string()),
                            TypeInner::Struct { members, span },
                        );
                        func.arguments.push(FunctionArgument {
                            name: Some("vs_in".to_string()),
                            ty: vs_in_ty,
                            binding: None,
                        });
                        vs_in_arg = Some(0);
                    }
                    VertexInputMode::None => {}
                }

                let mut out_members = vec![StructMember {
                    name: Some("position".to_string()),
                    ty: vec4_f32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::Position { invariant: false })),
                    offset: 0,
                }];
                let mut offset = 16u32;
                for &param_idx in &plan.vs_params {
                    let member_idx = out_members.len() as u32;
                    vs_param_member_index.insert(param_idx, member_idx);
                    out_members.push(StructMember {
                        name: Some(format!("param_{param_idx}")),
                        ty: vec4_f32_ty,
                        binding: Some(Binding::Location {
                            location: u32::from(param_idx),
                            interpolation: Some(Interpolation::Perspective),
                            sampling: Some(Sampling::Center),
                            blend_src: None,
                            per_primitive: false,
                        }),
                        offset,
                    });
                    offset += 16;
                }
                let vs_out_ty = insert_type(
                    &mut module,
                    Some("VsOut".to_string()),
                    TypeInner::Struct {
                        members: out_members,
                        span: offset,
                    },
                );
                func.result = Some(FunctionResult {
                    ty: vs_out_ty,
                    binding: None,
                });
                let out_local = func.local_variables.append(
                    LocalVariable {
                        name: Some("vs_out".to_string()),
                        ty: vs_out_ty,
                        init: None,
                    },
                    Span::UNDEFINED,
                );
                vs_out_local = Some(out_local);
            }
            ShaderStage::Fragment => {
                if !plan.frag_attrs.is_empty() {
                    let mut in_members = Vec::with_capacity(plan.frag_attrs.len());
                    let mut offset = 0u32;
                    for &attr in &plan.frag_attrs {
                        let member_idx = in_members.len() as u32;
                        frag_attr_member_index.insert(attr, member_idx);
                        in_members.push(StructMember {
                            name: Some(format!("attr_{attr}")),
                            ty: vec4_f32_ty,
                            binding: Some(Binding::Location {
                                location: u32::from(attr),
                                interpolation: Some(Interpolation::Perspective),
                                sampling: Some(Sampling::Center),
                                blend_src: None,
                                per_primitive: false,
                            }),
                            offset,
                        });
                        offset += 16;
                    }
                    let frag_in_ty = insert_type(
                        &mut module,
                        Some("FragIn".to_string()),
                        TypeInner::Struct {
                            members: in_members,
                            span: offset,
                        },
                    );
                    func.arguments.push(FunctionArgument {
                        name: Some("frag_in".to_string()),
                        ty: frag_in_ty,
                        binding: None,
                    });
                    frag_in_arg = Some(0);
                }

                let mut out_members = Vec::new();
                let mut offset = 0u32;
                for &mrt in &plan.ps_mrts {
                    let member_idx = out_members.len() as u32;
                    ps_mrt_member_index.insert(mrt, member_idx);
                    out_members.push(StructMember {
                        name: Some(format!("mrt_{mrt}")),
                        ty: vec4_f32_ty,
                        binding: Some(Binding::Location {
                            location: u32::from(mrt),
                            interpolation: None,
                            sampling: None,
                            blend_src: None,
                            per_primitive: false,
                        }),
                        offset,
                    });
                    offset += 16;
                }
                if plan.ps_exports_depth {
                    let member_idx = out_members.len() as u32;
                    ps_depth_member_index = Some(member_idx);
                    out_members.push(StructMember {
                        name: Some("frag_depth".to_string()),
                        ty: f32_ty,
                        binding: Some(Binding::BuiltIn(BuiltIn::FragDepth)),
                        offset,
                    });
                    offset += 16;
                }
                let ps_out_ty = insert_type(
                    &mut module,
                    Some("PsOut".to_string()),
                    TypeInner::Struct {
                        members: out_members,
                        span: offset,
                    },
                );
                func.result = Some(FunctionResult {
                    ty: ps_out_ty,
                    binding: None,
                });
                let out_local = func.local_variables.append(
                    LocalVariable {
                        name: Some("ps_out".to_string()),
                        ty: ps_out_ty,
                        init: None,
                    },
                    Span::UNDEFINED,
                );
                ps_out_local = Some(out_local);
            }
            ShaderStage::Compute => {
                func.arguments.push(FunctionArgument {
                    name: Some("gid".to_string()),
                    ty: vec3_u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::GlobalInvocationId)),
                });
            }
        }

        // Allocate VGPR locals (`var v_i: f32`) and SGPR locals (`var s_i: u32`).
        let mut vgpr_locals = BTreeMap::new();
        for &v in &plan.vgprs {
            let local = func.local_variables.append(
                LocalVariable {
                    name: Some(format!("v_{v}")),
                    ty: f32_ty,
                    init: None,
                },
                Span::UNDEFINED,
            );
            vgpr_locals.insert(v, local);
        }

        let mut sgpr_locals = BTreeMap::new();
        for &s in &plan.sgprs {
            let local = func.local_variables.append(
                LocalVariable {
                    name: Some(format!("s_{s}")),
                    ty: u32_ty,
                    init: None,
                },
                Span::UNDEFINED,
            );
            sgpr_locals.insert(s, local);
        }

        let vcc_local = if plan.needs_vcc {
            Some(func.local_variables.append(
                LocalVariable {
                    name: Some("vcc".to_string()),
                    ty: bool_ty,
                    init: None,
                },
                Span::UNDEFINED,
            ))
        } else {
            None
        };

        let scc_local = if plan.needs_scc {
            Some(func.local_variables.append(
                LocalVariable {
                    name: Some("scc".to_string()),
                    ty: bool_ty,
                    init: None,
                },
                Span::UNDEFINED,
            ))
        } else {
            None
        };

        let mut emitter = Self {
            module,
            types,
            cb0_global,
            cb0_vec4_count: plan.cbv_vec4_count,
            tex0_global,
            samp0_global,
            stage: plan.stage,
            func,
            vgpr_locals,
            sgpr_locals,
            vcc_local,
            scc_local,
            vs_out_local,
            vs_param_member_index,
            ps_out_local,
            ps_mrt_member_index,
            ps_depth_member_index,
            frag_in_arg,
            frag_attr_member_index,
            vs_in_arg,
            vs_attr_by_offset,
            has_buffer_fetch: plan.has_buffer_fetch,
            legacy_uv_member_index,
            vertex_input_mode: plan.vertex_input_mode,
            translated_op_count: 0,
            unsupported_op_count: 0,
            unsupported_comments: Vec::new(),
        };

        emitter.emit_prologue();
        emitter
    }

    // -- Expression & statement helpers --

    fn raw_expr(&mut self, expr: Expression) -> Handle<Expression> {
        self.func.expressions.append(expr, Span::UNDEFINED)
    }

    fn emit_expr(&mut self, expr: Expression) -> Handle<Expression> {
        let h = self.func.expressions.append(expr, Span::UNDEFINED);
        self.func
            .body
            .push(Statement::Emit(Range::new_from_bounds(h, h)), Span::UNDEFINED);
        h
    }

    fn lit_f32(&mut self, value: f32) -> Handle<Expression> {
        if value.is_finite() {
            self.raw_expr(Expression::Literal(Literal::F32(value)))
        } else {
            let bits = self.lit_u32(value.to_bits());
            self.bitcast_u32_to_f32(bits)
        }
    }

    fn lit_f32_from_bits(&mut self, bits: u32) -> Handle<Expression> {
        let f = f32::from_bits(bits);
        if bits == 0 || (f.is_finite() && f.is_normal()) {
            self.raw_expr(Expression::Literal(Literal::F32(f)))
        } else {
            let u = self.lit_u32(bits);
            self.bitcast_u32_to_f32(u)
        }
    }

    fn lit_u32(&mut self, value: u32) -> Handle<Expression> {
        self.raw_expr(Expression::Literal(Literal::U32(value)))
    }

    fn lit_i32(&mut self, value: i32) -> Handle<Expression> {
        self.raw_expr(Expression::Literal(Literal::I32(value)))
    }

    fn lit_bool(&mut self, value: bool) -> Handle<Expression> {
        self.raw_expr(Expression::Literal(Literal::Bool(value)))
    }

    fn bitcast_f32_to_u32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Uint,
            convert: None,
        })
    }

    fn bitcast_f32_to_i32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Sint,
            convert: None,
        })
    }

    fn bitcast_u32_to_f32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Float,
            convert: None,
        })
    }

    fn bitcast_u32_to_i32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Sint,
            convert: None,
        })
    }

    fn bitcast_i32_to_u32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Uint,
            convert: None,
        })
    }

    fn bitcast_i32_to_f32(&mut self, expr: Handle<Expression>) -> Handle<Expression> {
        self.emit_expr(Expression::As {
            expr,
            kind: ScalarKind::Float,
            convert: None,
        })
    }

    fn store_local(&mut self, local: Handle<LocalVariable>, value: Handle<Expression>) {
        let ptr = self.raw_expr(Expression::LocalVariable(local));
        self.func
            .body
            .push(Statement::Store { pointer: ptr, value }, Span::UNDEFINED);
    }

    fn load_local(&mut self, local: Handle<LocalVariable>) -> Handle<Expression> {
        let ptr = self.raw_expr(Expression::LocalVariable(local));
        self.emit_expr(Expression::Load { pointer: ptr })
    }

    fn load_vgpr_f32(&mut self, reg: u8) -> Handle<Expression> {
        if let Some(&local) = self.vgpr_locals.get(&reg) {
            self.load_local(local)
        } else {
            self.lit_f32(0.0)
        }
    }

    fn store_vgpr_f32(&mut self, reg: u8, val: Handle<Expression>) {
        if let Some(&local) = self.vgpr_locals.get(&reg) {
            self.store_local(local, val);
        }
    }

    fn store_vgpr_u32(&mut self, reg: u8, val: Handle<Expression>) {
        let f = self.bitcast_u32_to_f32(val);
        self.store_vgpr_f32(reg, f);
    }

    fn store_vgpr_i32(&mut self, reg: u8, val: Handle<Expression>) {
        let f = self.bitcast_i32_to_f32(val);
        self.store_vgpr_f32(reg, f);
    }

    /// Pack an `f32` value into the low 16 bits (`[15:0]`) of `v[reg]` as IEEE-754 `f16`
    /// using core WGSL `pack2x16float`, preserving the upper 16 bits (`[31:16]`).
    fn store_vgpr_f16_lo(&mut self, reg: u8, val_f32: Handle<Expression>) {
        let zero = self.lit_f32(0.0);
        let pair = self.emit_expr(Expression::Compose {
            ty: self.types.vec2_f32_ty,
            components: vec![val_f32, zero],
        });
        let packed = self.emit_expr(Expression::Math {
            fun: MathFunction::Pack2x16float,
            arg: pair,
            arg1: None,
            arg2: None,
            arg3: None,
        });
        let mask_lo = self.lit_u32(0x0000_FFFF);
        let new_lo = self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: packed,
            right: mask_lo,
        });
        let old_f = self.load_vgpr_f32(reg);
        let old_u = self.bitcast_f32_to_u32(old_f);
        let mask_hi = self.lit_u32(0xFFFF_0000);
        let preserved_hi = self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: old_u,
            right: mask_hi,
        });
        let combined = self.emit_expr(Expression::Binary {
            op: BinaryOperator::InclusiveOr,
            left: preserved_hi,
            right: new_lo,
        });
        self.store_vgpr_u32(reg, combined);
    }

    /// Pack an `f32` value into the high 16 bits (`[31:16]`) of `v[reg]` as IEEE-754 `f16`
    /// using core WGSL `pack2x16float`, preserving the lower 16 bits (`[15:0]`).
    fn store_vgpr_f16_hi(&mut self, reg: u8, val_f32: Handle<Expression>) {
        let zero = self.lit_f32(0.0);
        let pair = self.emit_expr(Expression::Compose {
            ty: self.types.vec2_f32_ty,
            components: vec![zero, val_f32],
        });
        let packed = self.emit_expr(Expression::Math {
            fun: MathFunction::Pack2x16float,
            arg: pair,
            arg1: None,
            arg2: None,
            arg3: None,
        });
        let mask_hi = self.lit_u32(0xFFFF_0000);
        let new_hi = self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: packed,
            right: mask_hi,
        });
        let old_f = self.load_vgpr_f32(reg);
        let old_u = self.bitcast_f32_to_u32(old_f);
        let mask_lo = self.lit_u32(0x0000_FFFF);
        let preserved_lo = self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: old_u,
            right: mask_lo,
        });
        let combined = self.emit_expr(Expression::Binary {
            op: BinaryOperator::InclusiveOr,
            left: preserved_lo,
            right: new_hi,
        });
        self.store_vgpr_u32(reg, combined);
    }

    fn load_sgpr_u32(&mut self, reg: u8) -> Handle<Expression> {
        if let Some(&local) = self.sgpr_locals.get(&reg) {
            self.load_local(local)
        } else {
            self.lit_u32(0)
        }
    }

    fn store_sgpr_u32(&mut self, reg: u8, val: Handle<Expression>) {
        if let Some(&local) = self.sgpr_locals.get(&reg) {
            self.store_local(local, val);
        }
    }

    fn load_vcc(&mut self) -> Handle<Expression> {
        if let Some(local) = self.vcc_local {
            self.load_local(local)
        } else {
            self.lit_bool(false)
        }
    }

    fn store_vcc(&mut self, val: Handle<Expression>) {
        if let Some(local) = self.vcc_local {
            self.store_local(local, val);
        }
    }

    fn load_scc(&mut self) -> Handle<Expression> {
        if let Some(local) = self.scc_local {
            self.load_local(local)
        } else {
            self.lit_bool(false)
        }
    }

    fn store_scc(&mut self, val: Handle<Expression>) {
        if let Some(local) = self.scc_local {
            self.store_local(local, val);
        }
    }

    fn emit_prologue(&mut self) {
        // Zero-initialize all declared VGPRs, SGPRs, VCC, SCC, and stage output structs.
        let vgpr_list: Vec<(u8, Handle<LocalVariable>)> =
            self.vgpr_locals.iter().map(|(&k, &v)| (k, v)).collect();
        for (_, local) in &vgpr_list {
            let z = self.lit_f32(0.0);
            self.store_local(*local, z);
        }
        let sgpr_list: Vec<(u8, Handle<LocalVariable>)> =
            self.sgpr_locals.iter().map(|(&k, &v)| (k, v)).collect();
        for (_, local) in &sgpr_list {
            let z = self.lit_u32(0);
            self.store_local(*local, z);
        }
        if let Some(vcc) = self.vcc_local {
            let f = self.lit_bool(false);
            self.store_local(vcc, f);
        }
        if let Some(scc) = self.scc_local {
            let f = self.lit_bool(false);
            self.store_local(scc, f);
        }

        // Stage-specific prologue bindings into VGPRs & default output struct values.
        match self.stage {
            ShaderStage::Vertex => {
                if let Some(out_local) = self.vs_out_local {
                    let out_ty = self.func.local_variables[out_local].ty;
                    let zero_struct = self.raw_expr(Expression::ZeroValue(out_ty));
                    self.store_local(out_local, zero_struct);
                    // Default position = vec4<f32>(0.0, 0.0, 0.0, 1.0)
                    let c0 = self.lit_f32(0.0);
                    let c1 = self.lit_f32(0.0);
                    let c2 = self.lit_f32(0.0);
                    let c3 = self.lit_f32(1.0);
                    let default_pos = self.emit_expr(Expression::Compose {
                        ty: self.types.vec4_f32_ty,
                        components: vec![c0, c1, c2, c3],
                    });
                    let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                    let pos_ptr = self.emit_expr(Expression::AccessIndex {
                        base: base_ptr,
                        index: 0,
                    });
                    self.func.body.push(
                        Statement::Store {
                            pointer: pos_ptr,
                            value: default_pos,
                        },
                        Span::UNDEFINED,
                    );
                }

                if let Some(arg_idx) = self.vs_in_arg {
                    match self.vertex_input_mode {
                        VertexInputMode::VertexIndex => {
                            let vi = self.raw_expr(Expression::FunctionArgument(arg_idx));
                            self.store_vgpr_u32(0, vi);
                        }
                        VertexInputMode::Attributes => {
                            let vs_in = self.raw_expr(Expression::FunctionArgument(arg_idx));
                            if self.has_buffer_fetch {
                                // Provide SV_VertexID in v0 prior to buffer attribute fetches.
                                if self.vgpr_locals.contains_key(&0) {
                                    let vi = self.emit_expr(Expression::AccessIndex {
                                        base: vs_in,
                                        index: 0,
                                    });
                                    self.store_vgpr_u32(0, vi);
                                }
                            } else {
                                // Legacy multi-VGPR attribute input without MUBUF/MTBUF fetches.
                                let pos = self.emit_expr(Expression::AccessIndex {
                                    base: vs_in,
                                    index: 1,
                                });
                                for c in 0..4u32 {
                                    let reg = c as u8;
                                    if self.vgpr_locals.contains_key(&reg) {
                                        let comp = self.emit_expr(Expression::AccessIndex {
                                            base: pos,
                                            index: c,
                                        });
                                        self.store_vgpr_f32(reg, comp);
                                    }
                                }
                            }
                            if let Some(uv_member_idx) = self.legacy_uv_member_index {
                                let uv = self.emit_expr(Expression::AccessIndex {
                                    base: vs_in,
                                    index: uv_member_idx,
                                });
                                for c in 0..2u32 {
                                    let reg = 4 + c as u8;
                                    if self.vgpr_locals.contains_key(&reg) {
                                        let comp = self.emit_expr(Expression::AccessIndex {
                                            base: uv,
                                            index: c,
                                        });
                                        self.store_vgpr_f32(reg, comp);
                                    }
                                }
                            }
                        }
                        VertexInputMode::None => {}
                    }
                }
            }
            ShaderStage::Fragment => {
                if let Some(out_local) = self.ps_out_local {
                    let out_ty = self.func.local_variables[out_local].ty;
                    let zero_struct = self.raw_expr(Expression::ZeroValue(out_ty));
                    self.store_local(out_local, zero_struct);
                }
            }
            ShaderStage::Compute => {
                let gid = self.raw_expr(Expression::FunctionArgument(0));
                for c in 0..3u32 {
                    let reg = c as u8;
                    if self.vgpr_locals.contains_key(&reg) {
                        let comp = self.emit_expr(Expression::AccessIndex {
                            base: gid,
                            index: c,
                        });
                        self.store_vgpr_u32(reg, comp);
                    }
                }
            }
        }
    }

    // -- Operand readers --

    fn inline_float_value(src: u16) -> Option<f32> {
        match src {
            240 => Some(0.5),
            241 => Some(-0.5),
            242 => Some(1.0),
            243 => Some(-1.0),
            244 => Some(2.0),
            245 => Some(-2.0),
            246 => Some(4.0),
            247 => Some(-4.0),
            248 => Some(0.159_154_94),
            _ => None,
        }
    }

    fn read_src_u32(&mut self, src: u16, literal: Option<u32>) -> Handle<Expression> {
        match src {
            0..=105 => self.load_sgpr_u32(src as u8),
            106 => {
                let cond = self.load_vcc();
                let t = self.lit_u32(1);
                let f = self.lit_u32(0);
                self.emit_expr(Expression::Select {
                    condition: cond,
                    accept: t,
                    reject: f,
                })
            }
            126 | 127 => self.lit_u32(0xFFFF_FFFF),
            128..=192 => self.lit_u32(u32::from(src - 128)),
            193..=208 => {
                let neg = -(i32::from(src) - 192);
                self.lit_u32(neg as u32)
            }
            240..=248 => {
                let f = Self::inline_float_value(src).unwrap_or(0.0);
                self.lit_u32(f.to_bits())
            }
            251 => {
                let cond = self.load_vcc();
                let t = self.lit_u32(0);
                let f = self.lit_u32(1);
                self.emit_expr(Expression::Select {
                    condition: cond,
                    accept: t,
                    reject: f,
                })
            }
            253 => {
                let cond = self.load_scc();
                let t = self.lit_u32(1);
                let f = self.lit_u32(0);
                self.emit_expr(Expression::Select {
                    condition: cond,
                    accept: t,
                    reject: f,
                })
            }
            255 => self.lit_u32(literal.unwrap_or(0)),
            256..=511 => {
                let vf = self.load_vgpr_f32((src - 256) as u8);
                self.bitcast_f32_to_u32(vf)
            }
            _ => self.lit_u32(0),
        }
    }

    fn read_src_i32(&mut self, src: u16, literal: Option<u32>) -> Handle<Expression> {
        match src {
            128..=192 => self.lit_i32(i32::from(src - 128)),
            193..=208 => self.lit_i32(-(i32::from(src) - 192)),
            255 => self.lit_i32(literal.unwrap_or(0) as i32),
            _ => {
                let u = self.read_src_u32(src, literal);
                self.bitcast_u32_to_i32(u)
            }
        }
    }

    fn read_src_f32(
        &mut self,
        src: u16,
        literal: Option<u32>,
        abs_mod: bool,
        neg_mod: bool,
    ) -> Handle<Expression> {
        let mut val = match src {
            0..=105 => {
                let u = self.load_sgpr_u32(src as u8);
                self.bitcast_u32_to_f32(u)
            }
            128 => self.lit_f32(0.0),
            129..=208 => {
                let u = self.read_src_u32(src, literal);
                self.bitcast_u32_to_f32(u)
            }
            240..=248 => self.lit_f32(Self::inline_float_value(src).unwrap_or(0.0)),
            255 => self.lit_f32_from_bits(literal.unwrap_or(0)),
            256..=511 => self.load_vgpr_f32((src - 256) as u8),
            _ => {
                let u = self.read_src_u32(src, literal);
                self.bitcast_u32_to_f32(u)
            }
        };

        if abs_mod {
            val = self.emit_expr(Expression::Math {
                fun: MathFunction::Abs,
                arg: val,
                arg1: None,
                arg2: None,
                arg3: None,
            });
        }
        if neg_mod {
            val = self.emit_expr(Expression::Unary {
                op: UnaryOperator::Negate,
                expr: val,
            });
        }
        val
    }

    /// Read an `f16` (low or high 16-bit lane via `unpack2x16float`) or `f32` operand as `f32`,
    /// applying optional `abs` and `neg` modifiers (used by `v_fma_mix*` and scalar `*_f16` ops).
    fn read_src_mixed_f16_or_f32(
        &mut self,
        src: u16,
        literal: Option<u32>,
        is_f16: bool,
        hi_half: bool,
        abs_mod: bool,
        neg_mod: bool,
    ) -> Handle<Expression> {
        let mut val = if !is_f16 {
            self.read_src_f32(src, literal, false, false)
        } else {
            match src {
                128 => self.lit_f32(0.0),
                240..=248 => self.lit_f32(Self::inline_float_value(src).unwrap_or(0.0)),
                _ => {
                    let u = self.read_src_u32(src, literal);
                    let pair = self.emit_expr(Expression::Math {
                        fun: MathFunction::Unpack2x16float,
                        arg: u,
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    });
                    self.emit_expr(Expression::AccessIndex {
                        base: pair,
                        index: if hi_half { 1 } else { 0 },
                    })
                }
            }
        };
        if abs_mod {
            val = self.emit_expr(Expression::Math {
                fun: MathFunction::Abs,
                arg: val,
                arg1: None,
                arg2: None,
                arg3: None,
            });
        }
        if neg_mod {
            val = self.emit_expr(Expression::Unary {
                op: UnaryOperator::Negate,
                expr: val,
            });
        }
        val
    }

    /// Unpack a 16-bit IEEE-754 `f16` literal stored in the low 16 bits of a 32-bit word into `f32`.
    fn lit_f16_from_u32_lo(&mut self, bits: u32) -> Handle<Expression> {
        let u = self.lit_u32(bits & 0xFFFF);
        let pair = self.emit_expr(Expression::Math {
            fun: MathFunction::Unpack2x16float,
            arg: u,
            arg1: None,
            arg2: None,
            arg3: None,
        });
        self.emit_expr(Expression::AccessIndex {
            base: pair,
            index: 0,
        })
    }

    fn apply_float_out_mods(
        &mut self,
        mut val: Handle<Expression>,
        omod: u8,
        clamp: bool,
    ) -> Handle<Expression> {
        let scale = match omod {
            1 => Some(2.0f32),
            2 => Some(4.0f32),
            3 => Some(0.5f32),
            _ => None,
        };
        if let Some(s) = scale {
            let factor = self.lit_f32(s);
            val = self.emit_expr(Expression::Binary {
                op: BinaryOperator::Multiply,
                left: val,
                right: factor,
            });
        }
        if clamp {
            let lo = self.lit_f32(0.0);
            let hi = self.lit_f32(1.0);
            val = self.emit_expr(Expression::Math {
                fun: MathFunction::Clamp,
                arg: val,
                arg1: Some(lo),
                arg2: Some(hi),
                arg3: None,
            });
        }
        val
    }

    fn record_unsupported(&mut self, inst: &Rdna2Instruction) {
        self.unsupported_op_count += 1;
        // Strictly sanitized: instruction offset, encoding family, and mnemonic only.
        let suffix = if inst.truncated { " (truncated)" } else { "" };
        self.unsupported_comments.push(format!(
            "// unsupported_op [0x{:04x}] {}: {}{}",
            inst.offset,
            inst.encoding.as_str(),
            inst.mnemonic,
            suffix
        ));
    }

    // -- Instruction dispatch --

    fn translate_instruction(&mut self, inst: &Rdna2Instruction) {
        if inst.truncated {
            self.record_unsupported(inst);
            return;
        }

        let handled = match &inst.detail {
            Rdna2Detail::Sopp { op, .. } => self.translate_sopp(*op),
            Rdna2Detail::Sopc {
                op,
                ssrc0,
                ssrc1,
                literal,
            } => self.translate_sopc(*op, *ssrc0, *ssrc1, *literal),
            Rdna2Detail::Sop1 {
                op,
                sdst,
                ssrc0,
                literal,
            } => self.translate_sop1(*op, *sdst, *ssrc0, *literal),
            Rdna2Detail::Sopk {
                op, sdst, simm16, ..
            } => self.translate_sopk(*op, *sdst, *simm16),
            Rdna2Detail::Sop2 {
                op,
                sdst,
                ssrc0,
                ssrc1,
                literal,
            } => self.translate_sop2(*op, *sdst, *ssrc0, *ssrc1, *literal),
            Rdna2Detail::Smem {
                op,
                sdata,
                soffset,
                offset,
                ..
            } => self.translate_smem(*op, *sdata, *soffset, *offset),
            Rdna2Detail::Vop1 {
                op,
                vdst,
                src0,
                literal,
            } => self.translate_vop1(*op, *vdst, *src0, *literal, 0, false, false, false),
            Rdna2Detail::Vopc {
                op,
                vsrc1,
                src0,
                literal,
            } => self.translate_vopc(*op, *src0, 256 + u16::from(*vsrc1), *literal, 0, 0, None),
            Rdna2Detail::Vop2 {
                op,
                vdst,
                vsrc1,
                src0,
                literal,
            } => self.translate_vop2(
                *op,
                *vdst,
                *src0,
                256 + u16::from(*vsrc1),
                None,
                *literal,
                0,
                false,
                0,
                0,
            ),
            Rdna2Detail::Vop3 {
                op,
                vdst,
                src0,
                src1,
                src2,
                clamp,
                omod,
                abs,
                neg,
                literal,
                ..
            } => self.translate_vop3(
                *op, *vdst, *src0, *src1, *src2, *clamp, *omod, *abs, *neg, *literal,
            ),
            Rdna2Detail::Vop3P {
                op,
                vdst,
                src0,
                src1,
                src2,
                clamp,
                op_sel,
                op_sel_hi,
                neg_lo,
                neg_hi,
                literal,
            } => self.translate_vop3p(
                *op,
                *vdst,
                *src0,
                *src1,
                *src2,
                *clamp,
                *op_sel,
                *op_sel_hi,
                *neg_lo,
                *neg_hi,
                *literal,
            ),
            Rdna2Detail::Vintrp {
                op,
                vdst,
                attr,
                attrchan,
                ..
            } => self.translate_vintrp(*op, *vdst, *attr, *attrchan),
            Rdna2Detail::Mimg {
                op,
                dmask,
                vaddr,
                vdata,
                ..
            } => self.translate_mimg(*op, *dmask, *vaddr, *vdata),
            Rdna2Detail::Mubuf {
                op, offset, vdata, ..
            } => self.translate_vertex_buf_fetch(false, *op, *offset, *vdata),
            Rdna2Detail::Mtbuf {
                op, offset, vdata, ..
            } => self.translate_vertex_buf_fetch(true, *op, *offset, *vdata),
            Rdna2Detail::Exp {
                target,
                en,
                compr,
                vsrc,
                ..
            } => self.translate_exp(*target, *en, *compr, *vsrc),
            Rdna2Detail::Flat { .. }
            | Rdna2Detail::Ds { .. }
            | Rdna2Detail::Unknown { .. } => false,
        };

        if handled {
            self.translated_op_count += 1;
        } else {
            self.record_unsupported(inst);
        }
    }

    // -- SOPP / SOPC / SOP1 / SOPK / SOP2 / SMEM --

    fn translate_sopp(&mut self, op: u8) -> bool {
        match op {
            // s_nop, s_endpgm, s_waitcnt, s_setprio (scheduling priority hint),
            // s_endpgm_saved, s_endpgm_ordered_ps_done, s_code_end,
            // s_inst_prefetch, s_clause, s_wait_idle, s_waitcnt_depctr
            0x00 | 0x01 | 0x0C | 0x0F | 0x1B | 0x1E | 0x1F | 0x20 | 0x21 | 0x22 | 0x23 => true,
            // s_barrier
            0x0A => {
                if self.stage == ShaderStage::Compute {
                    self.func.body.push(
                        Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
                        Span::UNDEFINED,
                    );
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn translate_sopc(&mut self, op: u8, ssrc0: u8, ssrc1: u8, literal: Option<u32>) -> bool {
        let bin_op = match op {
            0x00 | 0x06 => BinaryOperator::Equal,
            0x01 | 0x07 => BinaryOperator::NotEqual,
            0x02 | 0x08 => BinaryOperator::Greater,
            0x03 | 0x09 => BinaryOperator::GreaterEqual,
            0x04 | 0x0A => BinaryOperator::Less,
            0x05 | 0x0B => BinaryOperator::LessEqual,
            _ => return false,
        };
        let is_signed = op <= 0x05;
        let (left, right) = if is_signed {
            (
                self.read_src_i32(u16::from(ssrc0), literal),
                self.read_src_i32(u16::from(ssrc1), literal),
            )
        } else {
            (
                self.read_src_u32(u16::from(ssrc0), literal),
                self.read_src_u32(u16::from(ssrc1), literal),
            )
        };
        let cmp = self.emit_expr(Expression::Binary {
            op: bin_op,
            left,
            right,
        });
        self.store_scc(cmp);
        true
    }

    fn translate_sop1(&mut self, op: u8, sdst: u8, ssrc0: u8, literal: Option<u32>) -> bool {
        match op {
            // s_mov_b32
            0x03 => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                self.store_sgpr_u32(sdst, v);
                true
            }
            // s_mov_b64: 64-bit scalar register-pair move (evaluate both halves before storing)
            0x04 => {
                let (lo, hi) = match ssrc0 {
                    0..=104 => (self.load_sgpr_u32(ssrc0), self.load_sgpr_u32(ssrc0 + 1)),
                    105 => (self.load_sgpr_u32(105), self.lit_u32(0)),
                    106 => (self.read_src_u32(106, None), self.lit_u32(0)),
                    126 => (self.lit_u32(0xFFFF_FFFF), self.lit_u32(0xFFFF_FFFF)),
                    128..=192 => (self.lit_u32(u32::from(ssrc0 - 128)), self.lit_u32(0)),
                    193..=208 => {
                        let neg = -(i32::from(ssrc0) - 192);
                        (self.lit_u32(neg as u32), self.lit_u32(0xFFFF_FFFF))
                    }
                    240..=248 => {
                        let f = f64::from(Self::inline_float_value(u16::from(ssrc0)).unwrap_or(0.0));
                        let bits = f.to_bits();
                        (
                            self.lit_u32((bits & 0xFFFF_FFFF) as u32),
                            self.lit_u32((bits >> 32) as u32),
                        )
                    }
                    255 => (self.lit_u32(literal.unwrap_or(0)), self.lit_u32(0)),
                    _ => (self.read_src_u32(u16::from(ssrc0), literal), self.lit_u32(0)),
                };
                match sdst {
                    0..=104 => {
                        self.store_sgpr_u32(sdst, lo);
                        self.store_sgpr_u32(sdst + 1, hi);
                    }
                    105 => {
                        self.store_sgpr_u32(105, lo);
                    }
                    106 => {
                        let combined = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::InclusiveOr,
                            left: lo,
                            right: hi,
                        });
                        let zero = self.lit_u32(0);
                        let cond = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::NotEqual,
                            left: combined,
                            right: zero,
                        });
                        self.store_vcc(cond);
                    }
                    _ => {}
                }
                true
            }
            // s_cmov_b32
            0x05 => {
                let cond = self.load_scc();
                let accept = self.read_src_u32(u16::from(ssrc0), literal);
                let reject = self.load_sgpr_u32(sdst);
                let res = self.emit_expr(Expression::Select {
                    condition: cond,
                    accept,
                    reject,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_not_b32
            0x07 => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let res = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: v,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_abs_i32
            0x34 => {
                let v = self.read_src_i32(u16::from(ssrc0), literal);
                let a = self.emit_expr(Expression::Math {
                    fun: MathFunction::Abs,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let u = self.bitcast_i32_to_u32(a);
                self.store_sgpr_u32(sdst, u);
                true
            }
            _ => false,
        }
    }

    fn translate_sopk(&mut self, op: u8, sdst: u8, simm16: u16) -> bool {
        let imm_u32 = (simm16 as i16 as i32) as u32;
        match op {
            // s_movk_i32
            0x00 => {
                let v = self.lit_u32(imm_u32);
                self.store_sgpr_u32(sdst, v);
                true
            }
            // s_addk_i32
            0x0F => {
                let lhs = self.load_sgpr_u32(sdst);
                let rhs = self.lit_u32(imm_u32);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: lhs,
                    right: rhs,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_mulk_i32
            0x10 => {
                let lhs = self.load_sgpr_u32(sdst);
                let rhs = self.lit_u32(imm_u32);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: lhs,
                    right: rhs,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_waitcnt_vscnt / vmcnt / expcnt / lgkmcnt
            0x17..=0x1A => true,
            _ => false,
        }
    }

    fn translate_sop2(
        &mut self,
        op: u8,
        sdst: u8,
        ssrc0: u8,
        ssrc1: u8,
        literal: Option<u32>,
    ) -> bool {
        let s0_u = || u16::from(ssrc0);
        let s1_u = || u16::from(ssrc1);
        match op {
            // s_add_u32 / s_add_i32
            0x00 | 0x02 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: l,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_sub_u32 / s_sub_i32
            0x01 | 0x03 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: l,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_min_i32 / s_max_i32
            0x06 | 0x08 => {
                let l = self.read_src_i32(s0_u(), literal);
                let r = self.read_src_i32(s1_u(), literal);
                let fun = if op == 0x06 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                let u = self.bitcast_i32_to_u32(res);
                self.store_sgpr_u32(sdst, u);
                true
            }
            // s_min_u32 / s_max_u32
            0x07 | 0x09 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let fun = if op == 0x07 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_cselect_b32: scc ? ssrc0 : ssrc1
            0x0A => {
                let cond = self.load_scc();
                let accept = self.read_src_u32(s0_u(), literal);
                let reject = self.read_src_u32(s1_u(), literal);
                let res = self.emit_expr(Expression::Select {
                    condition: cond,
                    accept,
                    reject,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_and_b32 / s_or_b32 / s_xor_b32
            0x0E | 0x10 | 0x12 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let bin = match op {
                    0x0E => BinaryOperator::And,
                    0x10 => BinaryOperator::InclusiveOr,
                    _ => BinaryOperator::ExclusiveOr,
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_andn2_b32: s0 & ~s1
            0x14 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let nr = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: r,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: l,
                    right: nr,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_lshl_b32 / s_lshr_b32
            0x1E | 0x20 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let mask = self.lit_u32(31);
                let shift = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: r,
                    right: mask,
                });
                let bin = if op == 0x1E {
                    BinaryOperator::ShiftLeft
                } else {
                    BinaryOperator::ShiftRight
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: shift,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_ashr_i32
            0x22 => {
                let l = self.read_src_i32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let mask = self.lit_u32(31);
                let shift = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: r,
                    right: mask,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: l,
                    right: shift,
                });
                let u = self.bitcast_i32_to_u32(res);
                self.store_sgpr_u32(sdst, u);
                true
            }
            // s_mul_i32
            0x26 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: l,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_bfe_u32: unsigned bitfield extract
            //   offset = src1[4:0], width = src1[22:16]
            //   sdst = (width == 0) ? 0 : ((src0 >> offset) & mask(width))
            0x27 => {
                let s0 = self.read_src_u32(s0_u(), literal);
                let s1 = self.read_src_u32(s1_u(), literal);
                let mask31 = self.lit_u32(31);
                let offset = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: s1,
                    right: mask31,
                });
                let shift16 = self.lit_u32(16);
                let s1_hi = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: s1,
                    right: shift16,
                });
                let mask7f = self.lit_u32(0x7F);
                let width = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: s1_hi,
                    right: mask7f,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: s0,
                    right: offset,
                });
                let max_shift = self.lit_u32(31);
                let safe_w = self.emit_expr(Expression::Math {
                    fun: MathFunction::Min,
                    arg: width,
                    arg1: Some(max_shift),
                    arg2: None,
                    arg3: None,
                });
                let one = self.lit_u32(1);
                let one_shl = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: one,
                    right: safe_w,
                });
                let narrow_mask = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: one_shl,
                    right: one,
                });
                let lit32 = self.lit_u32(32);
                let wide = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::GreaterEqual,
                    left: width,
                    right: lit32,
                });
                let all_ones = self.lit_u32(0xFFFF_FFFF);
                let eff_mask = self.emit_expr(Expression::Select {
                    condition: wide,
                    accept: all_ones,
                    reject: narrow_mask,
                });
                let extracted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: shifted,
                    right: eff_mask,
                });
                let zero = self.lit_u32(0);
                let is_zero_w = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Equal,
                    left: width,
                    right: zero,
                });
                let res = self.emit_expr(Expression::Select {
                    condition: is_zero_w,
                    accept: zero,
                    reject: extracted,
                });
                self.store_sgpr_u32(sdst, res);
                let nonzero = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::NotEqual,
                    left: res,
                    right: zero,
                });
                self.store_scc(nonzero);
                true
            }
            // s_lshl{1,2,3,4}_add_u32
            0x2E..=0x31 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let sh = self.lit_u32(u32::from(op - 0x2D));
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: l,
                    right: sh,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: shifted,
                    right: r,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            _ => false,
        }
    }

    fn translate_smem(&mut self, op: u8, sdata: u8, soffset: u8, byte_offset: i32) -> bool {
        let Some(count) = smem_load_dword_count(op) else {
            return false;
        };
        let Some(cb0) = self.cb0_global else {
            return false;
        };
        let total_dwords = self.cb0_vec4_count.max(1).saturating_mul(4);

        let cb0_ptr = self.raw_expr(Expression::GlobalVariable(cb0));
        let data_ptr = self.emit_expr(Expression::AccessIndex {
            base: cb0_ptr,
            index: 0,
        });

        let has_dynamic_offset = soffset != 0x7D && soffset <= 105;
        let base_word_dyn = if has_dynamic_offset {
            let s_off = self.load_sgpr_u32(soffset);
            let imm_off = self.lit_u32(byte_offset as u32);
            let byte_sum = self.emit_expr(Expression::Binary {
                op: BinaryOperator::Add,
                left: s_off,
                right: imm_off,
            });
            let two = self.lit_u32(2);
            Some(self.emit_expr(Expression::Binary {
                op: BinaryOperator::ShiftRight,
                left: byte_sum,
                right: two,
            }))
        } else {
            None
        };

        let static_base_word = if byte_offset > 0 {
            (byte_offset as u32) / 4
        } else {
            0
        };

        for k in 0..count {
            let dst_reg = sdata.saturating_add(k);
            if dst_reg > 105 {
                break;
            }
            let elem_val = if let Some(base_w) = base_word_dyn {
                let k_lit = self.lit_u32(u32::from(k));
                let raw_word = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: base_w,
                    right: k_lit,
                });
                let total_lit = self.lit_u32(total_dwords);
                let bounded_word = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Modulo,
                    left: raw_word,
                    right: total_lit,
                });
                let two = self.lit_u32(2);
                let vec_idx = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: bounded_word,
                    right: two,
                });
                let three = self.lit_u32(3);
                let comp_idx = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: bounded_word,
                    right: three,
                });
                let vec_ptr = self.emit_expr(Expression::Access {
                    base: data_ptr,
                    index: vec_idx,
                });
                let vec_val = self.emit_expr(Expression::Load { pointer: vec_ptr });
                self.emit_expr(Expression::Access {
                    base: vec_val,
                    index: comp_idx,
                })
            } else {
                let word_idx = (static_base_word.saturating_add(u32::from(k))) % total_dwords;
                let vec_idx = word_idx / 4;
                let comp_idx = word_idx % 4;
                let vec_ptr = self.emit_expr(Expression::AccessIndex {
                    base: data_ptr,
                    index: vec_idx,
                });
                let vec_val = self.emit_expr(Expression::Load { pointer: vec_ptr });
                self.emit_expr(Expression::AccessIndex {
                    base: vec_val,
                    index: comp_idx,
                })
            };

            self.store_sgpr_u32(dst_reg, elem_val);
        }

        true
    }

    // -- VALU: VOP1 / VOPC / VOP2 / VOP3 / VOP3P --

    #[allow(clippy::too_many_arguments)]
    fn translate_vop1(
        &mut self,
        op: u8,
        vdst: u8,
        src0: u16,
        literal: Option<u32>,
        omod: u8,
        clamp: bool,
        abs0: bool,
        neg0: bool,
    ) -> bool {
        match op {
            // v_nop / v_pipeflush / v_clrexcp
            0x00 | 0x1B | 0x41 => true,
            // v_mov_b32
            0x01 => {
                let val = if abs0 || neg0 || omod != 0 || clamp {
                    let f = self.read_src_f32(src0, literal, abs0, neg0);
                    self.apply_float_out_mods(f, omod, clamp)
                } else {
                    self.read_src_f32(src0, literal, false, false)
                };
                self.store_vgpr_f32(vdst, val);
                true
            }
            // v_readfirstlane_b32 sdst, vsrc0
            0x02 => {
                let u = self.read_src_u32(src0, literal);
                self.store_sgpr_u32(vdst & 0x7F, u);
                true
            }
            // v_cvt_f32_i32
            0x05 => {
                let s = self.read_src_i32(src0, literal);
                let f = self.emit_expr(Expression::As {
                    expr: s,
                    kind: ScalarKind::Float,
                    convert: Some(4),
                });
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_cvt_f32_u32
            0x06 => {
                let u = self.read_src_u32(src0, literal);
                let f = self.emit_expr(Expression::As {
                    expr: u,
                    kind: ScalarKind::Float,
                    convert: Some(4),
                });
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_cvt_u32_f32
            0x07 => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let u = self.emit_expr(Expression::As {
                    expr: f,
                    kind: ScalarKind::Uint,
                    convert: Some(4),
                });
                self.store_vgpr_u32(vdst, u);
                true
            }
            // v_cvt_i32_f32
            0x08 => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let i = self.emit_expr(Expression::As {
                    expr: f,
                    kind: ScalarKind::Sint,
                    convert: Some(4),
                });
                self.store_vgpr_i32(vdst, i);
                true
            }
            // v_cvt_f16_f32: convert f32 -> f16 stored in low 16 bits of vdst
            0x0A => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_cvt_f32_f16: convert low 16 bits f16 -> f32
            0x0B => {
                let f = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let out = self.apply_float_out_mods(f, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_fract_f32 / v_trunc_f32 / v_ceil_f32 / v_rndne_f32 / v_floor_f32 /
            // v_exp_f32 / v_log_f32 / v_rsq_f32 / v_sqrt_f32
            0x20 | 0x21 | 0x22 | 0x23 | 0x24 | 0x25 | 0x27 | 0x2E | 0x33 => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let fun = match op {
                    0x20 => MathFunction::Fract,
                    0x21 => MathFunction::Trunc,
                    0x22 => MathFunction::Ceil,
                    0x23 => MathFunction::Round,
                    0x24 => MathFunction::Floor,
                    0x25 => MathFunction::Exp2,
                    0x27 => MathFunction::Log2,
                    0x2E => MathFunction::InverseSqrt,
                    _ => MathFunction::Sqrt,
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: f,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_rcp_f32 / v_rcp_iflag_f32
            0x2A | 0x2B => {
                let one = self.lit_f32(1.0);
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: one,
                    right: f,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_sin_f32 / v_cos_f32 (input is in turns: angle = x * 2*pi)
            0x35 | 0x36 => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let tau = self.lit_f32(std::f32::consts::TAU);
                let rad = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: f,
                    right: tau,
                });
                let fun = if op == 0x35 {
                    MathFunction::Sin
                } else {
                    MathFunction::Cos
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: rad,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_not_b32
            0x37 => {
                let u = self.read_src_u32(src0, literal);
                let res = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: u,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_vopc(
        &mut self,
        op: u8,
        src0: u16,
        src1: u16,
        literal: Option<u32>,
        abs: u8,
        neg: u8,
        sdst_override: Option<u8>,
    ) -> bool {
        let cmp = match op {
            // f32 comparisons: 0x00..=0x0F (v_cmp_*_f32) and 0x10..=0x1F (v_cmpx_*_f32)
            0x00..=0x1F => {
                let cc = op & 0xF;
                let l = self.read_src_f32(src0, literal, (abs & 1) != 0, (neg & 1) != 0);
                let r = self.read_src_f32(src1, literal, (abs & 2) != 0, (neg & 2) != 0);
                match cc {
                    0 => self.lit_bool(false),
                    1 => self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Less,
                        left: l,
                        right: r,
                    }),
                    2 => self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Equal,
                        left: l,
                        right: r,
                    }),
                    3 => self.emit_expr(Expression::Binary {
                        op: BinaryOperator::LessEqual,
                        left: l,
                        right: r,
                    }),
                    4 => self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Greater,
                        left: l,
                        right: r,
                    }),
                    5 | 13 => self.emit_expr(Expression::Binary {
                        op: BinaryOperator::NotEqual,
                        left: l,
                        right: r,
                    }),
                    6 => self.emit_expr(Expression::Binary {
                        op: BinaryOperator::GreaterEqual,
                        left: l,
                        right: r,
                    }),
                    15 => self.lit_bool(true),
                    _ => return false,
                }
            }
            // i32 comparisons: 0x80..=0x87 / 0x90..=0x97
            0x80..=0x87 | 0x90..=0x97 => {
                let cc = op & 0x7;
                let l = self.read_src_i32(src0, literal);
                let r = self.read_src_i32(src1, literal);
                self.emit_int_cmp(cc, l, r)
            }
            // u32 comparisons: 0xC0..=0xC7 / 0xD0..=0xD7
            0xC0..=0xC7 | 0xD0..=0xD7 => {
                let cc = op & 0x7;
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                self.emit_int_cmp(cc, l, r)
            }
            _ => return false,
        };

        if let Some(sd) = sdst_override {
            let sd7 = sd & 0x7F;
            if sd7 == 106 {
                self.store_vcc(cmp);
            } else if sd7 <= 105 {
                let t = self.lit_u32(1);
                let f = self.lit_u32(0);
                let mask_u = self.emit_expr(Expression::Select {
                    condition: cmp,
                    accept: t,
                    reject: f,
                });
                self.store_sgpr_u32(sd7, mask_u);
            }
        } else {
            self.store_vcc(cmp);
        }
        true
    }

    fn emit_int_cmp(
        &mut self,
        cc: u8,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        match cc {
            0 => self.lit_bool(false),
            1 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::Less,
                left,
                right,
            }),
            2 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::Equal,
                left,
                right,
            }),
            3 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::LessEqual,
                left,
                right,
            }),
            4 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::Greater,
                left,
                right,
            }),
            5 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::NotEqual,
                left,
                right,
            }),
            6 => self.emit_expr(Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left,
                right,
            }),
            _ => self.lit_bool(true),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_vop2(
        &mut self,
        op: u8,
        vdst: u8,
        src0: u16,
        src1: u16,
        cond_src: Option<u16>,
        literal: Option<u32>,
        omod: u8,
        clamp: bool,
        abs: u8,
        neg: u8,
    ) -> bool {
        let abs0 = (abs & 1) != 0;
        let neg0 = (neg & 1) != 0;
        let abs1 = (abs & 2) != 0;
        let neg1 = (neg & 2) != 0;

        match op {
            // v_cndmask_b32 vdst, src0, src1, cond (cond ? src1 : src0)
            // GFX10.3 VOP2 opcode 0x00 and 0x01 (and VOP3 0x100 / 0x101).
            0x00 | 0x01 => {
                let reject = self.read_src_f32(src0, literal, abs0, neg0);
                let accept = self.read_src_f32(src1, literal, abs1, neg1);
                let cond = match cond_src {
                    Some(106) | None => self.load_vcc(),
                    Some(csrc) => {
                        let cu = self.read_src_u32(csrc, literal);
                        let zero = self.lit_u32(0);
                        self.emit_expr(Expression::Binary {
                            op: BinaryOperator::NotEqual,
                            left: cu,
                            right: zero,
                        })
                    }
                };
                let res = self.emit_expr(Expression::Select {
                    condition: cond,
                    accept,
                    reject,
                });
                self.store_vgpr_f32(vdst, res);
                true
            }
            // v_add_f32
            0x03 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: l,
                    right: r,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_sub_f32
            0x04 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: l,
                    right: r,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_subrev_f32: src1 - src0
            0x05 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: r,
                    right: l,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_fmac_legacy_f32 / v_fmac_f32: vdst = fma(src0, src1, vdst)
            0x06 | 0x2B => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.load_vgpr_f32(vdst);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_mul_legacy_f32 / v_mul_f32
            0x07 | 0x08 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let prod = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: l,
                    right: r,
                });
                let res = if op == 0x07 {
                    let zero = self.lit_f32(0.0);
                    let l_zero = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Equal,
                        left: l,
                        right: zero,
                    });
                    let r_zero = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Equal,
                        left: r,
                        right: zero,
                    });
                    let either_zero = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::LogicalOr,
                        left: l_zero,
                        right: r_zero,
                    });
                    self.emit_expr(Expression::Select {
                        condition: either_zero,
                        accept: zero,
                        reject: prod,
                    })
                } else {
                    prod
                };
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_min_f32 / v_max_f32
            0x0F | 0x10 => {
                let l = self.read_src_f32(src0, literal, abs0, neg0);
                let r = self.read_src_f32(src1, literal, abs1, neg1);
                let fun = if op == 0x0F {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_min_i32 / v_max_i32
            0x11 | 0x12 => {
                let l = self.read_src_i32(src0, literal);
                let r = self.read_src_i32(src1, literal);
                let fun = if op == 0x11 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_min_u32 / v_max_u32
            0x13 | 0x14 => {
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                let fun = if op == 0x13 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: l,
                    arg1: Some(r),
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_lshrrev_b32 / v_lshlrev_b32: src1 >>/<< (src0 & 31)
            0x16 | 0x18 => {
                let sh_raw = self.read_src_u32(src0, literal);
                let val = self.read_src_u32(src1, literal);
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: sh_raw,
                    right: mask,
                });
                let bin = if op == 0x16 {
                    BinaryOperator::ShiftRight
                } else {
                    BinaryOperator::ShiftLeft
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: val,
                    right: sh,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_ashrrev_i32: i32(src1) >> (src0 & 31)
            0x17 => {
                let sh_raw = self.read_src_u32(src0, literal);
                let val = self.read_src_i32(src1, literal);
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: sh_raw,
                    right: mask,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: val,
                    right: sh,
                });
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_and_b32 / v_or_b32 / v_xor_b32
            0x19 | 0x1A | 0x1B => {
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                let bin = match op {
                    0x19 => BinaryOperator::And,
                    0x1A => BinaryOperator::InclusiveOr,
                    _ => BinaryOperator::ExclusiveOr,
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: r,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_add_nc_u32 / v_sub_nc_u32 / v_subrev_nc_u32
            0x25 | 0x26 | 0x27 => {
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                let (left, right, bin) = match op {
                    0x25 => (l, r, BinaryOperator::Add),
                    0x26 => (l, r, BinaryOperator::Subtract),
                    _ => (r, l, BinaryOperator::Subtract),
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left,
                    right,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_fmamk_f32: vdst = fma(src0, K, src1)
            0x2C => {
                let a = self.read_src_f32(src0, None, abs0, neg0);
                let k = self.lit_f32_from_bits(literal.unwrap_or(0));
                let c = self.read_src_f32(src1, None, abs1, neg1);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(k),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_fmaak_f32: vdst = fma(src0, src1, K)
            0x2D => {
                let a = self.read_src_f32(src0, None, abs0, neg0);
                let b = self.read_src_f32(src1, None, abs1, neg1);
                let k = self.lit_f32_from_bits(literal.unwrap_or(0));
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(k),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_cvt_pkrtz_f16_f32: pack2x16float(vec2<f32>(src0, src1))
            0x2E => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let pair = self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![a, b],
                });
                let packed = self.emit_expr(Expression::Math {
                    fun: MathFunction::Pack2x16float,
                    arg: pair,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_u32(vdst, packed);
                true
            }
            // v_add_f16 / v_sub_f16 / v_subrev_f16 / v_mul_f16 (f32 math + pack2x16float, no `enable f16;`)
            0x32..=0x35 => {
                let a = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, literal, true, false, abs1, neg1);
                let (left, right, bin) = match op {
                    0x32 => (a, b, BinaryOperator::Add),
                    0x33 => (a, b, BinaryOperator::Subtract),
                    0x34 => (b, a, BinaryOperator::Subtract),
                    _ => (a, b, BinaryOperator::Multiply),
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left,
                    right,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_fmac_f16: vdst.lo = fma(src0.lo, src1.lo, vdst.lo)
            0x36 => {
                let a = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, literal, true, false, abs1, neg1);
                let c = self.read_src_mixed_f16_or_f32(
                    256 + u16::from(vdst),
                    None,
                    true,
                    false,
                    false,
                    false,
                );
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_fmamk_f16: vdst.lo = fma(src0.lo, K_f16, src1.lo)
            0x37 => {
                let a = self.read_src_mixed_f16_or_f32(src0, None, true, false, abs0, neg0);
                let k = self.lit_f16_from_u32_lo(literal.unwrap_or(0));
                let c = self.read_src_mixed_f16_or_f32(src1, None, true, false, abs1, neg1);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(k),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_fmaak_f16: vdst.lo = fma(src0.lo, src1.lo, K_f16)
            0x38 => {
                let a = self.read_src_mixed_f16_or_f32(src0, None, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, None, true, false, abs1, neg1);
                let k = self.lit_f16_from_u32_lo(literal.unwrap_or(0));
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(k),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_max_f16 / v_min_f16
            0x39 | 0x3A => {
                let a = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, literal, true, false, abs1, neg1);
                let fun = if op == 0x39 {
                    MathFunction::Max
                } else {
                    MathFunction::Min
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_vop3(
        &mut self,
        op: u16,
        vdst: u8,
        src0: u16,
        src1: u16,
        src2: u16,
        clamp: bool,
        omod: u8,
        abs: u8,
        neg: u8,
        literal: Option<u32>,
    ) -> bool {
        // VOPC promoted to VOP3 (0x000..=0x0FF)
        if op <= 0x0FF {
            return self.translate_vopc(op as u8, src0, src1, literal, abs, neg, Some(vdst));
        }
        // VOP2 promoted to VOP3 (0x100..=0x13F)
        if (0x100..=0x13F).contains(&op) {
            let v2_op = (op - 0x100) as u8;
            let cond = if matches!(v2_op, 0x00 | 0x01) {
                Some(src2)
            } else {
                None
            };
            return self.translate_vop2(
                v2_op, vdst, src0, src1, cond, literal, omod, clamp, abs, neg,
            );
        }
        // VOP1 promoted to VOP3 (0x180..=0x1FF)
        if (0x180..=0x1FF).contains(&op) {
            let v1_op = (op - 0x180) as u8;
            return self.translate_vop1(
                v1_op,
                vdst,
                src0,
                literal,
                omod,
                clamp,
                (abs & 1) != 0,
                (neg & 1) != 0,
            );
        }

        let abs0 = (abs & 1) != 0;
        let neg0 = (neg & 1) != 0;
        let abs1 = (abs & 2) != 0;
        let neg1 = (neg & 2) != 0;
        let abs2 = (abs & 4) != 0;
        let neg2 = (neg & 4) != 0;

        match op {
            // v_mad_legacy_f32 / v_mad_f32 / v_fma_f32
            0x140 | 0x141 | 0x14B => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.read_src_f32(src2, literal, abs2, neg2);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_min3_f32 / v_max3_f32
            0x151 | 0x154 => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.read_src_f32(src2, literal, abs2, neg2);
                let fun = if op == 0x151 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let ab = self.emit_expr(Expression::Math {
                    fun,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: ab,
                    arg1: Some(c),
                    arg2: None,
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_med3_f32: clamp(a, min(b, c), max(b, c))
            0x157 => {
                let a = self.read_src_f32(src0, literal, abs0, neg0);
                let b = self.read_src_f32(src1, literal, abs1, neg1);
                let c = self.read_src_f32(src2, literal, abs2, neg2);
                let lo = self.emit_expr(Expression::Math {
                    fun: MathFunction::Min,
                    arg: b,
                    arg1: Some(c),
                    arg2: None,
                    arg3: None,
                });
                let hi = self.emit_expr(Expression::Math {
                    fun: MathFunction::Max,
                    arg: b,
                    arg1: Some(c),
                    arg2: None,
                    arg3: None,
                });
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Clamp,
                    arg: a,
                    arg1: Some(lo),
                    arg2: Some(hi),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_mul_lo_u32
            0x169 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: a,
                    right: b,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_lshl_add_u32: (src0 << (src1 & 31)) + src2
            0x346 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: b,
                    right: mask,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: a,
                    right: sh,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: shifted,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_add_lshl_u32: (src0 + src1) << (src2 & 31)
            0x347 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let sum = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: a,
                    right: b,
                });
                let mask = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: c,
                    right: mask,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: sum,
                    right: sh,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_fma_f16: vdst.lo = fma(src0.lo, src1.lo, src2.lo)
            0x34B => {
                let a = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let b = self.read_src_mixed_f16_or_f32(src1, literal, true, false, abs1, neg1);
                let c = self.read_src_mixed_f16_or_f32(src2, literal, true, false, abs2, neg2);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_add3_u32
            0x36D => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let ab = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: ab,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_and_or_b32: (src0 & src1) | src2
            0x36F => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let ab = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: ab,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            _ => false,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_vop3p(
        &mut self,
        op: u8,
        vdst: u8,
        src0: u16,
        src1: u16,
        src2: u16,
        clamp: bool,
        op_sel: u8,
        op_sel_hi: u8,
        neg_lo: u8,
        neg_hi: u8,
        literal: Option<u32>,
    ) -> bool {
        let unpack = |this: &mut Self, s: u16| {
            let u = this.read_src_u32(s, literal);
            this.emit_expr(Expression::Math {
                fun: MathFunction::Unpack2x16float,
                arg: u,
                arg1: None,
                arg2: None,
                arg3: None,
            })
        };
        let pack_and_store = |this: &mut Self, mut vec2_val: Handle<Expression>| {
            if clamp {
                let z = this.lit_f32(0.0);
                let o = this.lit_f32(1.0);
                let lo = this.emit_expr(Expression::Splat {
                    size: VectorSize::Bi,
                    value: z,
                });
                let hi = this.emit_expr(Expression::Splat {
                    size: VectorSize::Bi,
                    value: o,
                });
                vec2_val = this.emit_expr(Expression::Math {
                    fun: MathFunction::Clamp,
                    arg: vec2_val,
                    arg1: Some(lo),
                    arg2: Some(hi),
                    arg3: None,
                });
            }
            let packed = this.emit_expr(Expression::Math {
                fun: MathFunction::Pack2x16float,
                arg: vec2_val,
                arg1: None,
                arg2: None,
                arg3: None,
            });
            this.store_vgpr_u32(vdst, packed);
        };

        match op {
            // v_pk_fma_f16
            0x0E => {
                let a = unpack(self, src0);
                let b = unpack(self, src1);
                let c = unpack(self, src2);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                pack_and_store(self, res);
                true
            }
            // v_pk_add_f16 / v_pk_mul_f16
            0x0F | 0x10 => {
                let a = unpack(self, src0);
                let b = unpack(self, src1);
                let bin = if op == 0x0F {
                    BinaryOperator::Add
                } else {
                    BinaryOperator::Multiply
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: a,
                    right: b,
                });
                pack_and_store(self, res);
                true
            }
            // v_pk_min_f16 / v_pk_max_f16
            0x11 | 0x12 => {
                let a = unpack(self, src0);
                let b = unpack(self, src1);
                let fun = if op == 0x11 {
                    MathFunction::Min
                } else {
                    MathFunction::Max
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                pack_and_store(self, res);
                true
            }
            // v_dot2_f32_f16: dot(unpack(src0), unpack(src1)) + src2
            0x13 => {
                let a = unpack(self, src0);
                let b = unpack(self, src1);
                let c = self.read_src_f32(src2, literal, false, false);
                let d = self.emit_expr(Expression::Math {
                    fun: MathFunction::Dot,
                    arg: a,
                    arg1: Some(b),
                    arg2: None,
                    arg3: None,
                });
                let sum = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: d,
                    right: c,
                });
                let out = self.apply_float_out_mods(sum, 0, clamp);
                self.store_vgpr_f32(vdst, out);
                true
            }
            // v_fma_mix_f32 (0x20), v_fma_mixlo_f16 (0x21), v_fma_mixhi_f16 (0x22):
            //   op_sel_hi[i]: 0 = f32 source, 1 = f16 source
            //   op_sel[i]:    when f16 source, 0 = lo 16 bits, 1 = hi 16 bits
            //   neg_hi[i]:    abs modifier
            //   neg_lo[i]:    neg modifier
            0x20..=0x22 => {
                let a = self.read_src_mixed_f16_or_f32(
                    src0,
                    literal,
                    (op_sel_hi & 1) != 0,
                    (op_sel & 1) != 0,
                    (neg_hi & 1) != 0,
                    (neg_lo & 1) != 0,
                );
                let b = self.read_src_mixed_f16_or_f32(
                    src1,
                    literal,
                    (op_sel_hi & 2) != 0,
                    (op_sel & 2) != 0,
                    (neg_hi & 2) != 0,
                    (neg_lo & 2) != 0,
                );
                let c = self.read_src_mixed_f16_or_f32(
                    src2,
                    literal,
                    (op_sel_hi & 4) != 0,
                    (op_sel & 4) != 0,
                    (neg_hi & 4) != 0,
                    (neg_lo & 4) != 0,
                );
                let fma_res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                let out = self.apply_float_out_mods(fma_res, 0, clamp);
                match op {
                    0x20 => self.store_vgpr_f32(vdst, out),
                    0x21 => self.store_vgpr_f16_lo(vdst, out),
                    _ => self.store_vgpr_f16_hi(vdst, out),
                }
                true
            }
            _ => false,
        }
    }

    // -- VINTRP (Fragment attribute interpolation) --

    fn translate_vintrp(&mut self, op: u8, vdst: u8, attr: u8, attrchan: u8) -> bool {
        if !matches!(op, 0..=2) || self.stage != ShaderStage::Fragment {
            return false;
        }
        let Some(arg_idx) = self.frag_in_arg else {
            return false;
        };
        let clamped_attr = attr % MAX_WEBGPU_LOCATIONS;
        let Some(&member_idx) = self.frag_attr_member_index.get(&clamped_attr) else {
            return false;
        };
        let frag_in = self.raw_expr(Expression::FunctionArgument(arg_idx));
        let attr_vec = self.emit_expr(Expression::AccessIndex {
            base: frag_in,
            index: member_idx,
        });
        let comp = self.emit_expr(Expression::AccessIndex {
            base: attr_vec,
            index: u32::from(attrchan & 0x3),
        });
        self.store_vgpr_f32(vdst, comp);
        true
    }

    // -- MIMG (2D Texture sampling & loading) --

    fn translate_mimg(&mut self, op: u8, dmask: u8, vaddr: u8, vdata: u8) -> bool {
        if !is_supported_mimg_op(op) {
            return false;
        }
        let Some(tex0) = self.tex0_global else {
            return false;
        };
        let image = self.raw_expr(Expression::GlobalVariable(tex0));

        let sampled = match op {
            // image_load / image_load_mip
            0x00 | 0x01 => {
                let ux = self.load_vgpr_f32(vaddr);
                let uy = self.load_vgpr_f32(vaddr.saturating_add(1));
                let ix = self.bitcast_f32_to_i32(ux);
                let iy = self.bitcast_f32_to_i32(uy);
                let coord = self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_i32_ty,
                    components: vec![ix, iy],
                });
                let level = if op == 0x01 {
                    let ml = self.load_vgpr_f32(vaddr.saturating_add(2));
                    Some(self.bitcast_f32_to_i32(ml))
                } else {
                    Some(self.lit_i32(0))
                };
                self.emit_expr(Expression::ImageLoad {
                    image,
                    coordinate: coord,
                    array_index: None,
                    sample: None,
                    level,
                })
            }
            // image_sample* (sampled 2D texture)
            _ => {
                let Some(samp0) = self.samp0_global else {
                    return false;
                };
                let sampler = self.raw_expr(Expression::GlobalVariable(samp0));
                let u = self.load_vgpr_f32(vaddr);
                let v = self.load_vgpr_f32(vaddr.saturating_add(1));
                let coord = self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![u, v],
                });

                // WGSL only permits implicit-derivative `textureSample` in Fragment shaders;
                // Vertex/Compute stages must use explicit `textureSampleLevel`.
                let level = match op {
                    0x27 | 0x37 => SampleLevel::Exact(self.lit_f32(0.0)),
                    0x24 | 0x34 => {
                        let lod = self.load_vgpr_f32(vaddr.saturating_add(2));
                        SampleLevel::Exact(lod)
                    }
                    _ if self.stage == ShaderStage::Fragment => SampleLevel::Auto,
                    _ => SampleLevel::Exact(self.lit_f32(0.0)),
                };

                self.emit_expr(Expression::ImageSample {
                    image,
                    sampler,
                    gather: None,
                    coordinate: coord,
                    array_index: None,
                    offset: None,
                    level,
                    depth_ref: None,
                    clamp_to_edge: false,
                })
            }
        };

        let eff_mask = if dmask == 0 { 0xF } else { dmask };
        let mut dst_offset = 0u8;
        for ch in 0..4u32 {
            if (eff_mask & (1 << ch)) != 0 {
                let comp = self.emit_expr(Expression::AccessIndex {
                    base: sampled,
                    index: ch,
                });
                self.store_vgpr_f32(vdata.saturating_add(dst_offset), comp);
                dst_offset = dst_offset.saturating_add(1);
            }
        }
        true
    }

    // -- MUBUF / MTBUF vertex attribute fetch helper --

    fn translate_vertex_buf_fetch(
        &mut self,
        is_mtbuf: bool,
        op: u8,
        byte_offset: u16,
        vdata: u8,
    ) -> bool {
        if self.stage != ShaderStage::Vertex
            || self.vertex_input_mode != VertexInputMode::Attributes
        {
            return false;
        }
        let Some(arg_idx) = self.vs_in_arg else {
            return false;
        };
        let Some(comps) = (if is_mtbuf {
            mtbuf_load_components(op)
        } else {
            mubuf_load_components(op)
        }) else {
            return false;
        };
        let Some(&(member_idx, attr_comps)) = self.vs_attr_by_offset.get(&byte_offset) else {
            return false;
        };
        let vs_in = self.raw_expr(Expression::FunctionArgument(arg_idx));
        let attr_val = self.emit_expr(Expression::AccessIndex {
            base: vs_in,
            index: member_idx,
        });
        for c in 0..comps {
            let val = if attr_comps == 1 {
                if c == 0 {
                    attr_val
                } else if c == 3 {
                    self.lit_f32(1.0)
                } else {
                    self.lit_f32(0.0)
                }
            } else if c < attr_comps {
                self.emit_expr(Expression::AccessIndex {
                    base: attr_val,
                    index: u32::from(c),
                })
            } else if c == 3 {
                self.lit_f32(1.0)
            } else {
                self.lit_f32(0.0)
            };
            self.store_vgpr_f32(vdata.saturating_add(c), val);
        }
        true
    }

    // -- EXP (Vertex position/param & Pixel MRT/depth exports) --

    fn build_export_vec4(
        &mut self,
        en: u8,
        compr: bool,
        vsrc: [u8; 4],
        default_w: f32,
    ) -> Handle<Expression> {
        if compr {
            let lo_vec = if (en & 0b0011) != 0 {
                let vf = self.load_vgpr_f32(vsrc[0]);
                let vu = self.bitcast_f32_to_u32(vf);
                self.emit_expr(Expression::Math {
                    fun: MathFunction::Unpack2x16float,
                    arg: vu,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            } else {
                let z0 = self.lit_f32(0.0);
                let z1 = self.lit_f32(0.0);
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![z0, z1],
                })
            };
            let hi_vec = if (en & 0b1100) != 0 {
                let vf = self.load_vgpr_f32(vsrc[1]);
                let vu = self.bitcast_f32_to_u32(vf);
                self.emit_expr(Expression::Math {
                    fun: MathFunction::Unpack2x16float,
                    arg: vu,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            } else {
                let z2 = self.lit_f32(0.0);
                let w3 = self.lit_f32(default_w);
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![z2, w3],
                })
            };

            let c0 = if (en & 0b0001) != 0 {
                self.emit_expr(Expression::AccessIndex {
                    base: lo_vec,
                    index: 0,
                })
            } else {
                self.lit_f32(0.0)
            };
            let c1 = if (en & 0b0010) != 0 {
                self.emit_expr(Expression::AccessIndex {
                    base: lo_vec,
                    index: 1,
                })
            } else {
                self.lit_f32(0.0)
            };
            let c2 = if (en & 0b0100) != 0 {
                self.emit_expr(Expression::AccessIndex {
                    base: hi_vec,
                    index: 0,
                })
            } else {
                self.lit_f32(0.0)
            };
            let c3 = if (en & 0b1000) != 0 {
                self.emit_expr(Expression::AccessIndex {
                    base: hi_vec,
                    index: 1,
                })
            } else {
                self.lit_f32(default_w)
            };
            self.emit_expr(Expression::Compose {
                ty: self.types.vec4_f32_ty,
                components: vec![c0, c1, c2, c3],
            })
        } else {
            let mut comps = Vec::with_capacity(4);
            for i in 0..4usize {
                if (en & (1 << i)) != 0 {
                    comps.push(self.load_vgpr_f32(vsrc[i]));
                } else {
                    comps.push(self.lit_f32(if i == 3 { default_w } else { 0.0 }));
                }
            }
            self.emit_expr(Expression::Compose {
                ty: self.types.vec4_f32_ty,
                components: comps,
            })
        }
    }

    fn translate_exp(
        &mut self,
        target: Rdna2ExportTarget,
        en: u8,
        compr: bool,
        vsrc: [u8; 4],
    ) -> bool {
        match (self.stage, target) {
            (_, Rdna2ExportTarget::Null) => true,
            (ShaderStage::Vertex, Rdna2ExportTarget::Pos(0)) => {
                let Some(out_local) = self.vs_out_local else {
                    return false;
                };
                let val = self.build_export_vec4(en, compr, vsrc, 1.0);
                let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                let field_ptr = self.emit_expr(Expression::AccessIndex {
                    base: base_ptr,
                    index: 0,
                });
                self.func.body.push(
                    Statement::Store {
                        pointer: field_ptr,
                        value: val,
                    },
                    Span::UNDEFINED,
                );
                true
            }
            (ShaderStage::Vertex, Rdna2ExportTarget::Param(n)) => {
                let Some(out_local) = self.vs_out_local else {
                    return false;
                };
                let clamped = n % (MAX_WEBGPU_LOCATIONS - 1);
                let Some(&member_idx) = self.vs_param_member_index.get(&clamped) else {
                    return false;
                };
                let val = self.build_export_vec4(en, compr, vsrc, 0.0);
                let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                let field_ptr = self.emit_expr(Expression::AccessIndex {
                    base: base_ptr,
                    index: member_idx,
                });
                self.func.body.push(
                    Statement::Store {
                        pointer: field_ptr,
                        value: val,
                    },
                    Span::UNDEFINED,
                );
                true
            }
            (ShaderStage::Fragment, Rdna2ExportTarget::Mrt(n)) => {
                let Some(out_local) = self.ps_out_local else {
                    return false;
                };
                let Some(&member_idx) = self.ps_mrt_member_index.get(&n) else {
                    return false;
                };
                let val = self.build_export_vec4(en, compr, vsrc, 1.0);
                let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                let field_ptr = self.emit_expr(Expression::AccessIndex {
                    base: base_ptr,
                    index: member_idx,
                });
                self.func.body.push(
                    Statement::Store {
                        pointer: field_ptr,
                        value: val,
                    },
                    Span::UNDEFINED,
                );
                true
            }
            (ShaderStage::Fragment, Rdna2ExportTarget::MrtZ) => {
                let Some(out_local) = self.ps_out_local else {
                    return false;
                };
                let Some(member_idx) = self.ps_depth_member_index else {
                    return false;
                };
                let depth_val = if (en & 1) != 0 {
                    self.load_vgpr_f32(vsrc[0])
                } else {
                    self.lit_f32(0.0)
                };
                let base_ptr = self.raw_expr(Expression::LocalVariable(out_local));
                let field_ptr = self.emit_expr(Expression::AccessIndex {
                    base: base_ptr,
                    index: member_idx,
                });
                self.func.body.push(
                    Statement::Store {
                        pointer: field_ptr,
                        value: depth_val,
                    },
                    Span::UNDEFINED,
                );
                true
            }
            _ => false,
        }
    }

    // -- Finalize function & module --

    fn finish(mut self, entry_name: String) -> (Module, usize, usize, Vec<String>) {
        // Emit stage return statement.
        match self.stage {
            ShaderStage::Vertex => {
                if let Some(out_local) = self.vs_out_local {
                    let val = self.load_local(out_local);
                    self.func
                        .body
                        .push(Statement::Return { value: Some(val) }, Span::UNDEFINED);
                }
            }
            ShaderStage::Fragment => {
                if let Some(out_local) = self.ps_out_local {
                    let val = self.load_local(out_local);
                    self.func
                        .body
                        .push(Statement::Return { value: Some(val) }, Span::UNDEFINED);
                }
            }
            ShaderStage::Compute => {
                self.func
                    .body
                    .push(Statement::Return { value: None }, Span::UNDEFINED);
            }
        }

        let (naga_stage, workgroup_size) = match self.stage {
            ShaderStage::Vertex => (NagaStage::Vertex, [0, 0, 0]),
            ShaderStage::Fragment => (NagaStage::Fragment, [0, 0, 0]),
            ShaderStage::Compute => (NagaStage::Compute, [64, 1, 1]),
        };

        self.module.entry_points.push(EntryPoint {
            name: entry_name,
            stage: naga_stage,
            early_depth_test: None,
            workgroup_size,
            workgroup_size_overrides: None,
            function: self.func,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        (
            self.module,
            self.translated_op_count,
            self.unsupported_op_count,
            self.unsupported_comments,
        )
    }
}

/// Baseline WebGPU capabilities permitted during Naga validation.
/// `SHADER_FLOAT16_IN_FLOAT32` enables core WGSL `pack2x16float`/`unpack2x16float`
/// built-ins (no `enable f16;` extension required in WebGPU).
const BASELINE_WEBGPU_CAPABILITIES: Capabilities = Capabilities::SHADER_FLOAT16_IN_FLOAT32;

/// Translate a raw little-endian AMD RDNA2 (GFX10.3 / Scarlett) shader binary into
/// a complete, Naga-validated WebGPU WGSL module.
///
/// Never panics on empty, truncated, misaligned, or arbitrary input byte slices.
pub fn translate_rdna2_to_wgsl(
    bytes: &[u8],
    options: &Rdna2TranslateOptions,
) -> Result<Rdna2WgslOutput, Rdna2TranslateError> {
    let instructions = decode_program(bytes);
    let summary = analyze_instructions(bytes, &instructions);
    let plan = plan_program(&summary, &instructions, options);

    let mut emitter = IrEmitter::new(&plan);
    for inst in &instructions[..plan.active_inst_count] {
        emitter.translate_instruction(inst);
    }

    let entry_point = plan.entry_point.clone();
    let (module, translated_op_count, unsupported_op_count, unsupported_comments) =
        emitter.finish(entry_point.clone());

    // Unconditional Naga validation against baseline WebGPU capabilities.
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    let module_info = validator
        .validate(&module)
        .map_err(|err| Rdna2TranslateError::NagaValidation(err.emit_to_string("")))?;

    let wgsl_body = naga::back::wgsl::write_string(
        &module,
        &module_info,
        naga::back::wgsl::WriterFlags::empty(),
    )
    .map_err(|err| Rdna2TranslateError::WgslEmit(err.to_string()))?;

    // Prepend strictly sanitized WGSL header & graceful fallback opcode comments.
    let mut wgsl = String::with_capacity(wgsl_body.len() + 256);
    let _ = writeln!(
        wgsl,
        "// Translated from AMD RDNA2 (GFX10.3) by playgta6-shaders"
    );
    let _ = writeln!(
        wgsl,
        "// hash: {} | stage: {} | translated_ops: {} | unsupported_ops: {}",
        format_shader_hash(summary.fnv1a_hash),
        plan.stage.as_str(),
        translated_op_count,
        unsupported_op_count
    );
    for comment in &unsupported_comments {
        let _ = writeln!(wgsl, "{comment}");
    }
    wgsl.push('\n');
    wgsl.push_str(&wgsl_body);

    // When `naga/wgsl-in` is active (unit tests), additionally round-trip parse & validate the
    // final WGSL text (including comments) via `naga::front::wgsl::parse_str`.
    #[cfg(test)]
    {
        let reparsed = naga::front::wgsl::parse_str(&wgsl)
            .map_err(|err| Rdna2TranslateError::WgslParse(err.emit_to_string(&wgsl)))?;
        let mut revalidator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
        revalidator
            .validate(&reparsed)
            .map_err(|err| Rdna2TranslateError::NagaValidation(err.emit_to_string(&wgsl)))?;
    }

    let output_locations: Vec<u32> = match plan.stage {
        ShaderStage::Vertex => plan.vs_params.iter().map(|&p| u32::from(p)).collect(),
        ShaderStage::Fragment => plan.ps_mrts.iter().map(|&m| u32::from(m)).collect(),
        ShaderStage::Compute => Vec::new(),
    };

    let mut bindings = Vec::new();
    if plan.uses_cbv {
        bindings.push(WgslBindingInfo {
            group: 0,
            binding: 0,
            kind: "uniform",
            name: "cb0".to_string(),
        });
    }
    if plan.uses_mimg {
        bindings.push(WgslBindingInfo {
            group: 0,
            binding: 1,
            kind: "texture_2d",
            name: "tex0".to_string(),
        });
        bindings.push(WgslBindingInfo {
            group: 0,
            binding: 2,
            kind: "sampler",
            name: "samp0".to_string(),
        });
    }

    let binding_summary = Rdna2BindingSummary {
        has_cbv: plan.uses_cbv,
        cbv_vec4_count: plan.cbv_vec4_count,
        has_texture_2d: plan.uses_mimg,
        has_sampler: plan.uses_mimg,
        vertex_input_mode: plan.vertex_input_mode.as_str().to_string(),
        fragment_input_locations: plan.frag_attrs.iter().map(|&a| u32::from(a)).collect(),
        output_locations,
        exports_position: plan.stage == ShaderStage::Vertex,
        exports_depth: plan.stage == ShaderStage::Fragment && plan.ps_exports_depth,
    };

    Ok(Rdna2WgslOutput {
        wgsl,
        stage: plan.stage,
        entry_point,
        hash_fnv1a64: summary.fnv1a_hash,
        translated_op_count,
        unsupported_op_count,
        naga_validated: true,
        bindings,
        binding_summary,
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::fnv1a_64;

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(words.len() * 4);
        for &w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    fn enc_sopp(op: u8, simm16: u16) -> u32 {
        (0b1011_1111_1u32 << 23) | (u32::from(op & 0x7F) << 16) | u32::from(simm16)
    }

    fn enc_sop1(op: u8, sdst: u8, ssrc0: u8) -> u32 {
        (0b1011_1110_1u32 << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | (u32::from(op) << 8)
            | u32::from(ssrc0)
    }

    fn enc_sop2(op: u8, sdst: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b10u32 << 30)
            | (u32::from(op & 0x7F) << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | (u32::from(ssrc1) << 8)
            | u32::from(ssrc0)
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

    fn enc_vop1(op: u8, vdst: u8, src0: u16) -> u32 {
        (0b0111_111u32 << 25)
            | (u32::from(vdst) << 17)
            | (u32::from(op) << 9)
            | u32::from(src0 & 0x1FF)
    }

    fn enc_vopc(op: u8, src0: u16, vsrc1: u8) -> u32 {
        (0b0111_110u32 << 25)
            | (u32::from(op) << 17)
            | (u32::from(vsrc1) << 9)
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

    #[allow(clippy::too_many_arguments)]
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

    fn assert_naga_wgsl_valid(wgsl: &str) {
        let module = naga::front::wgsl::parse_str(wgsl)
            .unwrap_or_else(|e| panic!("WGSL parse failed:\n{}\nWGSL:\n{wgsl}", e.emit_to_string(wgsl)));
        let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("WGSL validation failed:\n{}\nWGSL:\n{wgsl}", e.emit_to_string(wgsl)));
    }

    #[test]
    fn translates_fullscreen_triangle_vertex_shader_and_validates_with_naga() {
        // Real procedural fullscreen-triangle RDNA2 VS:
        //   v_lshlrev_b32 v1, 1, v0           ; (vid << 1)
        //   v_and_b32     v1, 2, v1           ; u_int = (vid << 1) & 2
        //   v_and_b32     v2, 2, v0           ; v_int = vid & 2
        //   v_cvt_f32_u32 v1, v1              ; u = f32(u_int)
        //   v_cvt_f32_u32 v2, v2              ; v = f32(v_int)
        //   v_fma_f32     v3, v1, 2.0, -1.0   ; pos.x = u * 2.0 - 1.0
        //   v_fma_f32     v4, -v2, 2.0, 1.0   ; pos.y = 1.0 - v * 2.0
        //   v_mov_b32     v5, 0.0             ; pos.z = 0.0
        //   v_mov_b32     v6, 1.0             ; pos.w = 1.0
        //   exp pos0 v3, v4, v5, v6
        //   exp param0 v1, v2, off, off done
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vop2(0x18, 1, 128 + 1, 0)); // v_lshlrev_b32 v1, 1, v0
        words.push(enc_vop2(0x19, 1, 128 + 2, 1)); // v_and_b32 v1, 2, v1
        words.push(enc_vop2(0x19, 2, 128 + 2, 0)); // v_and_b32 v2, 2, v0
        words.push(enc_vop1(0x06, 1, 256 + 1)); // v_cvt_f32_u32 v1, v1
        words.push(enc_vop1(0x06, 2, 256 + 2)); // v_cvt_f32_u32 v2, v2
        words.extend_from_slice(&enc_vop3(0x14B, 3, 256 + 1, 244, 243, false, 0, 0, 0)); // fma(v1, 2.0, -1.0)
        words.extend_from_slice(&enc_vop3(0x14B, 4, 256 + 2, 244, 242, false, 0, 0, 0b001)); // fma(-v2, 2.0, 1.0)
        words.push(enc_vop1(0x01, 5, 128)); // v_mov_b32 v5, 0
        words.push(enc_vop1(0x01, 6, 242)); // v_mov_b32 v6, 1.0
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [3, 4, 5, 6])); // pos0
        words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [1, 2, 0, 0])); // param0 done
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("fullscreen-triangle VS translation should succeed");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.entry_point, "vs_main");
        assert_eq!(out.hash_fnv1a64, fnv1a_64(&bytes));
        assert_eq!(out.translated_op_count, 12);
        assert_eq!(out.unsupported_op_count, 0);
        assert_eq!(out.binding_summary.vertex_input_mode, "vertex_index");
        assert!(out.binding_summary.exports_position);
        assert_eq!(out.binding_summary.output_locations, vec![0]);

        assert!(out.wgsl.contains("@vertex"));
        assert!(out.wgsl.contains("@builtin(vertex_index)"));
        assert!(out.wgsl.contains("@builtin(position)"));
        assert!(out.wgsl.contains("@location(0)"));
        assert!(out.wgsl.contains("struct VsOut"));
        assert!(out.wgsl.contains("fma("));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_textured_splash_blit_pixel_shader_with_vintrp_mimg_exp() {
        // Textured splash-blit PS:
        //   v_interp_p1_f32 v2, v0, attr0.x
        //   v_interp_p2_f32 v2, v1, attr0.x
        //   v_interp_p1_f32 v3, v0, attr0.y
        //   v_interp_p2_f32 v3, v1, attr0.y
        //   image_sample v4, v2, s[8:15], s[4:7] dmask:0xf dim:1
        //   s_waitcnt vmcnt(0)
        //   exp mrt0 v4, v5, v6, v7 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 2, 1));
        words.push(enc_sopp(0x0C, 0x3F70)); // s_waitcnt vmcnt(0)
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("textured splash-blit PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.entry_point, "fs_main");
        assert_eq!(out.translated_op_count, 8);
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.binding_summary.has_sampler);
        assert!(!out.binding_summary.has_cbv);
        assert_eq!(out.binding_summary.fragment_input_locations, vec![0]);
        assert_eq!(out.binding_summary.output_locations, vec![0]);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 1,
                    kind: "texture_2d",
                    name: "tex0".to_string(),
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                },
            ]
        );

        assert!(out.wgsl.contains("@fragment"));
        assert!(out.wgsl.contains("fn fs_main("));
        assert!(!out.wgsl.contains("fn ps_main("));
        assert!(out.wgsl.contains("struct FragIn"));
        assert!(out.wgsl.contains("struct PsOut"));
        assert!(out.wgsl.contains("texture_2d<f32>"));
        assert!(out.wgsl.contains("sampler"));
        assert!(
            out.wgsl.contains("textureSample("),
            "expected textureSample in WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("tex0") && out.wgsl.contains("samp0"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_smem_cbv_modulated_pixel_shader() {
        // CBV-modulated PS:
        //   s_load_dwordx4 s[4:7], s[0:1], 0x10 glc
        //   s_waitcnt lgkmcnt(0)
        //   v_interp_p1_f32 v2, v0, attr0.x
        //   v_interp_p2_f32 v2, v1, attr0.x
        //   v_interp_p1_f32 v3, v0, attr0.y
        //   v_interp_p2_f32 v3, v1, attr0.y
        //   image_sample_lz v4, v2, s[8:15], s[12:15] dmask:0xf dim:1
        //   v_mul_f32 v4, s4, v4
        //   v_mul_f32 v5, s5, v5
        //   v_mul_f32 v6, s6, v6
        //   v_mul_f32 v7, s7, v7
        //   exp mrt0 v4, v5, v6, v7 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x02, 4, 0, 0x7D, 0x10, true, false));
        words.push(enc_sopp(0x0C, 0xC07F));
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 4, 2, 3)); // image_sample_lz
        words.push(enc_vop2(0x08, 4, 4, 4)); // v_mul_f32 v4, s4, v4
        words.push(enc_vop2(0x08, 5, 5, 5)); // v_mul_f32 v5, s5, v5
        words.push(enc_vop2(0x08, 6, 6, 6)); // v_mul_f32 v6, s6, v6
        words.push(enc_vop2(0x08, 7, 7, 7)); // v_mul_f32 v7, s7, v7
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("CBV-modulated PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert!(out.binding_summary.has_cbv);
        assert!(out.binding_summary.cbv_vec4_count >= DEFAULT_MIN_CBV_VEC4_COUNT);
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.binding_summary.has_sampler);
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.wgsl.contains("struct Rdna2Cbv"));
        assert!(out.wgsl.contains("@group(0) @binding(0)"));
        assert!(
            out.wgsl.contains("var<uniform>") && out.wgsl.contains("cb0") && out.wgsl.contains("Rdna2Cbv"),
            "expected uniform cb0 binding in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("textureSampleLevel("),
            "expected textureSampleLevel in WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"has_cbv\": true"));
        assert!(json.contains("\"naga_validated\": true"));
    }

    #[test]
    fn translates_compressed_f16_export_pixel_shader_with_unpack2x16float() {
        // Packed f16 output PS:
        //   v_mov_b32 v0, 1.0
        //   v_mov_b32 v1, 0.5
        //   v_cvt_pkrtz_f16_f32 v2, v0, v1
        //   v_cvt_pkrtz_f16_f32 v3, v1, v0
        //   exp mrt0 v2, v3, off, off done compr vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vop1(0x01, 0, 242)); // 1.0
        words.push(enc_vop1(0x01, 1, 240)); // 0.5
        words.push(enc_vop2(0x2E, 2, 256 + 0, 1)); // v_cvt_pkrtz_f16_f32 v2, v0, v1
        words.push(enc_vop2(0x2E, 3, 256 + 1, 0)); // v_cvt_pkrtz_f16_f32 v3, v1, v0
        words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [2, 3, 0, 0])); // compr
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("compressed f16 export PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.wgsl.contains("pack2x16float"));
        assert!(out.wgsl.contains("unpack2x16float"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_valu_salu_modifiers_and_gracefully_handles_unsupported_ops() {
        // Exercise VOP3 abs/neg/clamp/omod, VOPC + v_cndmask_b32, SALU, math builtins,
        // and an unsupported DS instruction to verify graceful comment fallback.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 2, 128 + 7)); // s_mov_b32 s2, 7
        words.push(enc_sop2(0x00, 3, 2, 128 + 3)); // s_add_u32 s3, s2, 3
        words.push(enc_vop1(0x06, 0, 3)); // v_cvt_f32_u32 v0, s3
        words.push(enc_vop1(0x2A, 1, 256 + 0)); // v_rcp_f32 v1, v0
        words.push(enc_vop1(0x2E, 2, 256 + 1)); // v_rsq_f32 v2, v1
        words.push(enc_vop1(0x33, 3, 256 + 2)); // v_sqrt_f32 v3, v2
        words.push(enc_vop1(0x24, 4, 256 + 3)); // v_floor_f32 v4, v3
        words.push(enc_vop1(0x20, 5, 256 + 3)); // v_fract_f32 v5, v3
        words.push(enc_vopc(0x01, 256 + 4, 5)); // v_cmp_lt_f32 vcc, v4, v5
        words.push(enc_vop2(0x01, 6, 256 + 4, 5)); // v_cndmask_b32 v6, v4, v5, vcc
        words.extend_from_slice(&enc_vop3(
            0x141,
            7,
            256 + 6,
            242,
            240,
            true,  // clamp
            1,     // mul:2
            0b001, // |v6|
            0b010, // -1.0
        ));
        // Unsupported DS instruction: ds_swizzle_b32 v8, v0, v1
        let ds_w0 = (0b110110u32 << 26) | (0x3Du32 << 18);
        let ds_w1 = (8u32 << 24) | (1u32 << 8);
        words.extend_from_slice(&[ds_w0, ds_w1]);
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [6, 7, 4, 5]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("program with unsupported DS op must still produce valid WGSL");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 1);
        assert!(out.translated_op_count >= 12);
        assert!(out.wgsl.contains("// unsupported_op"));
        assert!(out.wgsl.contains("ds_swizzle_b32"));
        assert!(out.wgsl.contains("inverseSqrt("));
        assert!(out.wgsl.contains("sqrt("));
        assert!(out.wgsl.contains("floor("));
        assert!(out.wgsl.contains("fract("));
        assert!(out.wgsl.contains("abs("));
        assert!(out.wgsl.contains("clamp("));
        assert!(out.wgsl.contains("select("));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn handles_empty_truncated_misaligned_and_fuzz_inputs_without_panicking() {
        // Empty and sub-word buffers across all stage overrides
        for len in 0..=3 {
            let buf = vec![0x5Au8; len];
            for stage in [
                Rdna2StageOverride::Auto,
                Rdna2StageOverride::Vertex,
                Rdna2StageOverride::Pixel,
                Rdna2StageOverride::Compute,
            ] {
                let opts = Rdna2TranslateOptions {
                    stage,
                    ..Rdna2TranslateOptions::default()
                };
                let out = translate_rdna2_to_wgsl(&buf, &opts)
                    .expect("empty/sub-word buffer must produce valid WGSL");
                assert!(out.naga_validated);
                assert_naga_wgsl_valid(&out.wgsl);
            }
        }

        // Truncated 64-bit instructions (missing second word)
        let truncated_words = [
            enc_smem(0x02, 0, 0, 0x7D, 0, false, false)[0],
            enc_exp(12, 0xF, false, true, false, [0, 1, 2, 3])[0],
            enc_vop3(0x14B, 0, 0, 0, 0, false, 0, 0, 0)[0],
            enc_mimg(0x20, 0xF, 1, 0, 0, 0, 0)[0],
            enc_vop2(0x2C, 0, 256, 1),
        ];
        for w0 in truncated_words {
            let mut buf = w0.to_le_bytes().to_vec();
            buf.push(0xCC);
            let out = translate_rdna2_to_wgsl(&buf, &Rdna2TranslateOptions::default())
                .expect("truncated instruction must degrade gracefully to valid WGSL");
            assert!(out.naga_validated);
            assert!(out.unsupported_op_count >= 1);
            assert!(out.wgsl.contains("(truncated)"));
            assert_naga_wgsl_valid(&out.wgsl);
        }

        // Deterministic pseudo-random fuzz buffers across all stage overrides
        let mut state: u64 = 0xCAFE_BABE_1337_4242;
        for sample_len in [16usize, 64, 131, 256] {
            let mut fuzz = Vec::with_capacity(sample_len);
            for _ in 0..sample_len {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                fuzz.push((state & 0xFF) as u8);
            }
            for stage in [
                Rdna2StageOverride::Auto,
                Rdna2StageOverride::Vertex,
                Rdna2StageOverride::Pixel,
                Rdna2StageOverride::Compute,
            ] {
                let opts = Rdna2TranslateOptions {
                    stage,
                    ..Rdna2TranslateOptions::default()
                };
                let out = translate_rdna2_to_wgsl(&fuzz, &opts)
                    .expect("arbitrary fuzz bytes must still produce Naga-valid WGSL");
                assert!(out.naga_validated);
                assert_naga_wgsl_valid(&out.wgsl);
            }
        }
    }

    #[test]
    fn translates_attribute_driven_vertex_shader_with_mubuf_and_multi_vgpr_inputs() {
        // Attribute-driven VS using MUBUF buffer_load_format_xyzw + multi-VGPR uv inputs:
        //   buffer_load_format_xyzw v0, v0, s[4:7], 0 idxen
        //   exp pos0 v0, v1, v2, v3
        //   exp param0 v4, v5, off, off done
        //   s_endpgm
        let mubuf_w0 = (0b111000u32 << 26) | (0x03u32 << 18) | (1 << 13);
        let mubuf_w1 = (128u32 << 24) | (1u32 << 16);
        let mut words = Vec::new();
        words.extend_from_slice(&[mubuf_w0, mubuf_w1]);
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [0, 1, 2, 3]));
        words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [4, 5, 0, 0]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("attribute-driven VS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.binding_summary.vertex_input_mode, "attributes");
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.wgsl.contains("struct VsIn"));
        assert!(out.wgsl.contains("@location(0)"));
        assert!(out.wgsl.contains("@location(1)"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_compute_shader_with_barrier_image_load_dynamic_smem_and_vop3p() {
        // Compute shader exercising:
        //   s_mov_b32 s2, 16
        //   s_load_dwordx2 s[4:5], s[0:1], s2, 0x4
        //   image_load v4, v0, s[8:15] dmask:0xf dim:1
        //   v_pk_add_f16 v6, v4, v5, v0
        //   v_dot2_f32_f16 v7, v4, v5, v6 clamp
        //   s_barrier
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 2, 128 + 16)); // s_mov_b32 s2, 16
        words.extend_from_slice(&enc_smem(0x01, 4, 0, 2, 4, false, false)); // dynamic soffset=s2
        words.extend_from_slice(&enc_mimg(0x00, 0xF, 1, 0, 4, 2, 0)); // image_load
        // VOP3P: prefix 0b110011, op 0x0F (v_pk_add_f16), vdst=6, src0=v4, src1=v5, src2=v0
        let vop3p_add_w0 = (0b110011u32 << 26) | (0x0Fu32 << 16) | 6;
        let vop3p_add_w1 = ((256u32 + 0) << 18) | ((256u32 + 5) << 9) | (256u32 + 4);
        words.extend_from_slice(&[vop3p_add_w0, vop3p_add_w1]);
        // VOP3P: op 0x13 (v_dot2_f32_f16), clamp=1, vdst=7, src0=v4, src1=v5, src2=v6
        let vop3p_dot_w0 = (0b110011u32 << 26) | (0x13u32 << 16) | (1 << 15) | 7;
        let vop3p_dot_w1 = ((256u32 + 6) << 18) | ((256u32 + 5) << 9) | (256u32 + 4);
        words.extend_from_slice(&[vop3p_dot_w0, vop3p_dot_w1]);
        words.push(enc_sopp(0x0A, 0)); // s_barrier
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("compute shader should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.entry_point, "cs_main");
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.binding_summary.has_cbv);
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.wgsl.contains("@compute"));
        assert!(out.wgsl.contains("workgroupBarrier()"));
        assert!(out.wgsl.contains("textureLoad("));
        assert!(out.wgsl.contains("dot("));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_multi_mrt_and_depth_export_pixel_shader_with_explicit_lod() {
        // Multi-MRT + MRTZ PS with image_sample_l:
        //   v_interp_mov_f32 v2, p0, attr0.x
        //   v_interp_mov_f32 v3, p0, attr0.y
        //   v_mov_b32 v4, 2.0
        //   image_sample_l v5, v2, s[8:15], s[4:7] dmask:0xf dim:1
        //   exp mrt0 v5, v6, v7, v8
        //   exp mrt1 v8, v7, v6, v5
        //   exp mrtz v4, off, off, off done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vintrp(2, 2, 0, 0, 2)); // v_interp_mov_f32 v2, p0, attr0.x
        words.push(enc_vintrp(2, 3, 0, 1, 2)); // v_interp_mov_f32 v3, p0, attr0.y
        words.push(enc_vop1(0x01, 4, 244)); // v_mov_b32 v4, 2.0
        words.extend_from_slice(&enc_mimg(0x24, 0xF, 1, 2, 5, 2, 1)); // image_sample_l
        words.extend_from_slice(&enc_exp(0, 0xF, false, false, false, [5, 6, 7, 8])); // mrt0
        words.extend_from_slice(&enc_exp(1, 0xF, false, false, false, [8, 7, 6, 5])); // mrt1
        words.extend_from_slice(&enc_exp(8, 0x1, false, true, true, [4, 0, 0, 0])); // mrtz
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("multi-MRT + MRTZ PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0);
        assert_eq!(out.binding_summary.output_locations, vec![0, 1]);
        assert!(out.binding_summary.exports_depth);
        assert!(out.wgsl.contains("@builtin(frag_depth)"));
        assert!(out.wgsl.contains("textureSampleLevel("));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_vop3p(
        op: u8,
        vdst: u8,
        src0: u16,
        src1: u16,
        src2: u16,
        clamp: bool,
        op_sel: u8,
        op_sel_hi: u8,
        neg_lo: u8,
        neg_hi: u8,
    ) -> [u32; 2] {
        let w0 = (0b110011u32 << 26)
            | (u32::from(op & 0x7F) << 16)
            | (u32::from(clamp) << 15)
            | (u32::from((op_sel_hi >> 2) & 0x1) << 14)
            | (u32::from(op_sel & 0x7) << 11)
            | (u32::from(neg_hi & 0x7) << 8)
            | u32::from(vdst);
        let w1 = (u32::from(neg_lo & 0x7) << 29)
            | (u32::from(op_sel_hi & 0x3) << 27)
            | (u32::from(src2 & 0x1FF) << 18)
            | (u32::from(src1 & 0x1FF) << 9)
            | u32::from(src0 & 0x1FF);
        [w0, w1]
    }

    fn enc_mubuf(op: u8, offset: u16, vdata: u8, vaddr: u8, srsrc: u8, idxen: bool) -> [u32; 2] {
        let w0 = (0b111000u32 << 26)
            | (u32::from(op & 0x7F) << 18)
            | (u32::from(idxen) << 13)
            | u32::from(offset & 0xFFF);
        let w1 = (128u32 << 24)
            | (u32::from(srsrc & 0x1F) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    fn enc_mtbuf(
        op: u8,
        format: u8,
        offset: u16,
        vdata: u8,
        vaddr: u8,
        srsrc: u8,
        idxen: bool,
    ) -> [u32; 2] {
        let w0 = (0b111010u32 << 26)
            | (u32::from(format & 0x7F) << 19)
            | (u32::from(op & 0x7) << 16)
            | (u32::from(idxen) << 13)
            | u32::from(offset & 0xFFF);
        let w1 = (128u32 << 24)
            | (u32::from((op >> 3) & 0x1) << 21)
            | (u32::from(srsrc & 0x1F) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    #[test]
    fn translates_frame1_vertex_shader_with_24b_offsets_and_all_new_opcodes() {
        // Synthetic Frame-1 vertex shader exercising:
        //   - s_setprio 0
        //   - s_mov_b64 s[6:7], s[2:3]
        //   - s_bfe_u32 s8, s6, 0x00080004 (offset=4, width=8)
        //   - s_load_dwordx4 s[12:15], s[0:1], 0x0
        //   - buffer_load_dwordx3 v[1:3], v0, s[4:7], 0 idxen offset:0       -> @location(0) position: vec3<f32>
        //   - tbuffer_load_format_xyzw v[4:7], v0, s[4:7], 0 idxen offset:12 -> @location(1) color: vec4<f32>
        //   - buffer_load_dwordx2 v[8:9], v0, s[4:7], 0 idxen offset:16      -> @location(2) uv: vec2<f32>
        //   - v_fmaak_f16 v10, v8, v9, 0x00003c00 (f16 1.0)
        //   - v_cmp_gt_f32 vcc, v1, 0.0
        //   - v_cndmask_b32 v11, v2, v3, vcc (VOP2 opcode 0x00!)
        //   - v_cndmask_b32 v12, v4, v5, s8  (VOP3 opcode 0x100!)
        //   - exp pos0 v1, v11, v3, v7
        //   - exp param0 v12, v5, v6, v7
        //   - exp param1 v8, v9, v10, off done
        //   - s_endpgm
        let mut words = vec![
            enc_sopp(0x0F, 0),            // s_setprio 0
            enc_sop1(0x04, 6, 2),         // s_mov_b64 s[6:7], s[2:3]
            enc_sop2(0x27, 8, 6, 0xFF),   // s_bfe_u32 s8, s6, 0x00080004
            0x0008_0004,
        ];
        words.extend_from_slice(&enc_smem(0x02, 12, 0, 0x7D, 0, false, false)); // s_load_dwordx4 -> cb0
        words.extend_from_slice(&enc_mubuf(0x0F, 0, 1, 0, 1, true)); // buffer_load_dwordx3 @ offset 0
        words.extend_from_slice(&enc_mtbuf(0x03, 4, 12, 4, 0, 1, true)); // tbuffer_load_format_xyzw @ offset 12
        words.extend_from_slice(&enc_mubuf(0x0D, 16, 8, 0, 1, true)); // buffer_load_dwordx2 @ offset 16
        words.push(enc_vop2(0x38, 10, 256 + 8, 9)); // v_fmaak_f16 v10, v8, v9, 0x3c00
        words.push(0x0000_3C00); // f16 1.0
        words.push(enc_vopc(0x04, 256 + 1, 1)); // v_cmp_gt_f32 vcc, v1, v1
        words.push(enc_vop2(0x00, 11, 256 + 2, 3)); // v_cndmask_b32 (VOP2 0x00)
        words.extend_from_slice(&enc_vop3(
            0x100,
            12,
            256 + 4,
            256 + 5,
            8,
            false,
            0,
            0,
            0,
        )); // v_cndmask_b32 (VOP3 0x100)
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [1, 11, 3, 7]));
        words.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [12, 5, 6, 7]));
        words.extend_from_slice(&enc_exp(33, 0x7, false, true, false, [8, 9, 10, 0]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("Frame-1 synthetic VS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.entry_point, "vs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.vertex_input_mode, "attributes");
        assert!(out.binding_summary.has_cbv);
        assert_eq!(
            out.bindings,
            vec![WgslBindingInfo {
                group: 0,
                binding: 0,
                kind: "uniform",
                name: "cb0".to_string(),
            }]
        );

        // Distinct @location(0), @location(1), @location(2) with exact types & no `enable f16;`.
        assert!(out.wgsl.contains("@location(0) position: vec3<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(1) color: vec4<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(2) uv: vec2<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("vs_in.position"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("vs_in.color"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("vs_in.uv"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("select("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"), "must not require shader-f16");
        assert!(!out.wgsl.contains("<f16>"), "must not emit f16 type");
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"bindings\": ["), "JSON:\n{json}");
        assert!(
            json.contains("{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\"}"),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn maps_arbitrary_vertex_fetch_offsets_deterministically_without_collisions() {
        // Fetches at arbitrary non-canonical offsets 8 (vec2), 24 (vec4), 40 (f32).
        let mut words = Vec::new();
        words.extend_from_slice(&enc_mubuf(0x0D, 24, 4, 0, 1, true)); // offset 24 (2 comps)
        words.extend_from_slice(&enc_mubuf(0x0E, 40, 8, 0, 1, true)); // offset 40 (4 comps)
        words.extend_from_slice(&enc_mtbuf(0x00, 4, 8, 1, 0, 1, true)); // offset 8 (1 comp)
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [1, 4, 8, 11]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("arbitrary-offset VS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0);
        // Sorted by ascending byte offset: 8 -> loc 0 (f32), 24 -> loc 1 (vec2), 40 -> loc 2 (vec4).
        // (Naga's WGSL namer appends `_` to identifiers ending in digits.)
        assert!(out.wgsl.contains("@location(0) attr_8_: f32"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(1) attr_24_: vec2<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@location(2) attr_40_: vec4<f32>"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_frame1_pixel_shader_with_v_cndmask_and_v_fma_mix_feeding_compr_mrt0() {
        // Synthetic Frame-1 pixel shader exercising:
        //   - v_interp_p1/p2 attr0.xyzw (color/uv input)
        //   - v_cmp_gt_f32 vcc, v2, 0.5
        //   - v_cndmask_b32 v6, v2, 1.0, vcc (VOP2 opcode 0x00!)
        //   - v_fma_mixlo_f16 v8, v6, 0.5, 0.25 clamp  (low 16 bits -> R)
        //   - v_fma_mixhi_f16 v8, v3, 1.0, 0.0         (high 16 bits -> G, preserving R!)
        //   - v_fma_mixlo_f16 v9, v4, 1.0, 0.0         (low 16 bits -> B)
        //   - v_fma_mixhi_f16 v9, 1.0, 1.0, 0.0        (high 16 bits -> A, preserving B!)
        //   - exp mrt0 v8, v9, off, off done compr vm
        //   - s_endpgm
        let mut words = vec![
            enc_vintrp(0, 2, 0, 0, 0),
            enc_vintrp(1, 2, 0, 0, 1),
            enc_vintrp(0, 3, 0, 1, 0),
            enc_vintrp(1, 3, 0, 1, 1),
            enc_vintrp(0, 4, 0, 2, 0),
            enc_vintrp(1, 4, 0, 2, 1),
            enc_vopc(0x04, 240, 2), // v_cmp_gt_f32 vcc, 0.5, v2
            enc_vop2(0x00, 6, 242, 2), // v_cndmask_b32 v6, 1.0, v2, vcc (VOP2 0x00)
        ];
        // v_fma_mixlo_f16 v8, v6, 0.5, 0x3e800000 (0.25f) clamp (f32 sources: op_sel_hi = 0)
        words.extend_from_slice(&enc_vop3p(
            0x21,
            8,
            256 + 6,
            240,
            0xFF,
            true,
            0,
            0,
            0,
            0,
        ));
        words.push(0.25f32.to_bits());
        // v_fma_mixhi_f16 v8, v3, 1.0, 0.0 (op_sel_hi = 0, neg_hi = 0b001 abs on src0)
        words.extend_from_slice(&enc_vop3p(
            0x22,
            8,
            256 + 3,
            242,
            128,
            false,
            0,
            0,
            0,
            0b001,
        ));
        // v_fma_mixlo_f16 v9, v8.lo (f16 source! op_sel_hi = 0b001, op_sel = 0b000), v4, 0.0
        words.extend_from_slice(&enc_vop3p(
            0x21,
            9,
            256 + 8,
            256 + 4,
            128,
            false,
            0b000,
            0b001,
            0,
            0,
        ));
        // v_fma_mixhi_f16 v9, 1.0, 1.0, 0.0
        words.extend_from_slice(&enc_vop3p(
            0x22,
            9,
            242,
            242,
            128,
            false,
            0,
            0,
            0,
            0,
        ));
        words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [8, 9, 0, 0])); // compr vm done
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("Frame-1 synthetic PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.entry_point, "fs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.bindings.is_empty());

        // Entry point must be fs_main, packing + unpacking v8/v9 with real fma computations.
        assert!(out.wgsl.contains("fn fs_main("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("fma("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("select("), "WGSL:\n{}", out.wgsl);
        // Verify bit-preserving masks for mixlo (0xFFFFu / 0xFFFF0000u) and mixhi.
        assert!(
            out.wgsl.contains("65535u") || out.wgsl.contains("0xffffu"),
            "expected low-16 mask in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("4294901760u") || out.wgsl.contains("0xffff0000u"),
            "expected high-16 mask in WGSL:\n{}",
            out.wgsl
        );
        // Must NOT hardcode zero mrt_0 export and must NOT emit f16 type.
        assert!(!out.wgsl.contains("enable f16"), "must not require shader-f16");
        assert!(!out.wgsl.contains("<f16>"), "must not emit f16 type");
        assert!(out.wgsl.contains("ps_out.mrt_0_ = vec4<f32>("), "WGSL:\n{}", out.wgsl);
        assert!(
            !out.wgsl.contains("ps_out.mrt_0_ = vec4<f32>(0f, 0f, 0f, 0f)"),
            "compressed mrt0 export must unpack computed v_fma_mix* halves:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"entry_point\": \"fs_main\""), "JSON:\n{json}");
        assert!(json.contains("\"bindings\": []"), "JSON:\n{json}");
    }

    #[test]
    fn bindings_json_accurately_lists_all_emitted_group_binding_declarations() {
        // CBV + MIMG shader -> group 0 binding 0 (cb0), binding 1 (tex0), binding 2 (samp0).
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x02, 4, 0, 0x7D, 0, false, false));
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 2, 1));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default()).unwrap();
        assert_eq!(out.bindings.len(), 3);
        assert_eq!(out.bindings[0].group, 0);
        assert_eq!(out.bindings[0].binding, 0);
        assert_eq!(out.bindings[0].kind, "uniform");
        assert_eq!(out.bindings[0].name, "cb0");
        assert_eq!(out.bindings[1].group, 0);
        assert_eq!(out.bindings[1].binding, 1);
        assert_eq!(out.bindings[1].kind, "texture_2d");
        assert_eq!(out.bindings[1].name, "tex0");
        assert_eq!(out.bindings[2].group, 0);
        assert_eq!(out.bindings[2].binding, 2);
        assert_eq!(out.bindings[2].kind, "sampler");
        assert_eq!(out.bindings[2].name, "samp0");

        // Every declared binding in `out.bindings` must match a real `@group(g) @binding(b)` in WGSL.
        for b in &out.bindings {
            let needle = format!("@group({}) @binding({})", b.group, b.binding);
            assert!(out.wgsl.contains(&needle), "missing {needle} in WGSL:\n{}", out.wgsl);
        }

        let json = out.to_json();
        assert!(
            json.contains("{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\"}"),
            "JSON:\n{json}"
        );
        assert!(
            json.contains("{\"group\": 0, \"binding\": 1, \"kind\": \"texture_2d\", \"name\": \"tex0\"}"),
            "JSON:\n{json}"
        );
        assert!(
            json.contains("{\"group\": 0, \"binding\": 2, \"kind\": \"sampler\", \"name\": \"samp0\"}"),
            "JSON:\n{json}"
        );
    }
}
