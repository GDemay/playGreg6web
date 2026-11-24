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
    Statement, StorageAccess, StorageFormat, StructMember, SwizzleComponent, Type, TypeInner,
    UnaryOperator, VectorSize,
};

use crate::convert::{ShaderStage, format_shader_hash};
use crate::pack::write_json_string;
use crate::rdna2::{
    Rdna2Detail, Rdna2ExportTarget, Rdna2InferredStage, Rdna2Instruction, Rdna2ProgramSummary,
    analyze_instructions, decode_program,
};

/// Default minimum number of `vec4<u32>` slots in a statically-sized uniform CBV (`cb0`, `cb1`, ...).
/// Tightly sized to `1` (`16` bytes) so small Xbox Series CBVs (e.g. 48 B / 96 B) are never
/// over-declared as 256 B uniform blocks in WebGPU.
pub const DEFAULT_MIN_CBV_VEC4_COUNT: u32 = 1;

/// Minimum number of `vec4<u32>` slots (`256` bytes) emitted for a CBV that uses a dynamic/unknown
/// `soffset` where no upper bound is statically known.
pub const DEFAULT_DYNAMIC_CBV_VEC4_COUNT: u32 = 16;

/// Upper bound on `vec4<u32>` slots in a single CBV (1024 * 16B = 16 KiB, within WebGPU 64 KiB baseline).
pub const MAX_CBV_VEC4_COUNT: u32 = 1024;

/// Maximum inter-stage location count under WebGPU baseline limits.
const MAX_WEBGPU_LOCATIONS: u8 = 16;

/// Maximum color attachment count under WebGPU baseline limits.
const MAX_WEBGPU_MRTS: u8 = 8;

/// Returns `true` iff `f` can be emitted directly as a WGSL `Literal::F32` without
/// risking Naga decimal rounding overflow past `f32::MAX` (which Chrome/Dawn Tint
/// rejects as unrepresentable), losing `-0.0`, or emitting subnormals / NaN / Inf.
///
/// Naga prints `f32::MAX` (`0x7f7fffff`, `3.4028234663852886e38`) rounded up to
/// `340282350000000000000000000000000000000f` (`3.4028235e38 > f32::MAX`), which
/// Tint rejects. Any finite float with `|f| >= 3.4e38` (including `f32::MAX`,
/// `-f32::MAX`, `0x7f7ffffe`, etc.) routes through `pg6_u32_bits_to_float(bits)`
/// for an exact IEEE-754 bitcast that Tint accepts unconditionally.
#[inline]
fn is_safe_wgsl_f32_literal(f: f32) -> bool {
    f.to_bits() == 0 || (f.is_finite() && f.is_normal() && f.abs() < 3.4e38_f32)
}

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

/// Default compute workgroup dimensions (`@workgroup_size(64, 1, 1)`).
pub const DEFAULT_COMPUTE_WORKGROUP_SIZE: [u32; 3] = [64, 1, 1];

/// LDS workgroup shared memory size in 32-bit dwords (`4096 * 4 = 16 KiB`).
const LDS_WORKGROUP_DWORDS: u32 = 4096;

/// Mask applied to LDS dword indices (`4095 = 0x0FFF`) so workgroup memory accesses never trap.
const LDS_WORKGROUP_INDEX_MASK: u32 = LDS_WORKGROUP_DWORDS - 1;

/// Options controlling RDNA2 -> WGSL translation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2TranslateOptions {
    /// Pipeline stage selection (`Auto` infers from `EXP`/`VINTRP`/memory instructions).
    pub stage: Rdna2StageOverride,
    /// Optional explicit WGSL entry-point name (defaults to `vs_main`/`fs_main`/`cs_main`).
    pub entry_point: Option<String>,
    /// Minimum number of `vec4<u32>` entries emitted when `cb0` uniform buffer is needed.
    pub min_cbv_vec4_count: u32,
    /// Workgroup size `(X, Y, Z)` for `ShaderStage::Compute` (defaults to `[64, 1, 1]`).
    pub workgroup_size: [u32; 3],
    /// Optional base SGPR index for seeding `(wg_id.x, wg_id.y, wg_id.z)` in compute shaders.
    /// When `None`, defaults to `s0`/`s1`/`s2` for SGPRs read before being written.
    pub workgroup_id_sgpr_base: Option<u8>,
}

impl Default for Rdna2TranslateOptions {
    fn default() -> Self {
        Self {
            stage: Rdna2StageOverride::Auto,
            entry_point: None,
            min_cbv_vec4_count: DEFAULT_MIN_CBV_VEC4_COUNT,
            workgroup_size: DEFAULT_COMPUTE_WORKGROUP_SIZE,
            workgroup_id_sgpr_base: None,
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
    /// Minimum WebGPU uniform buffer binding size in bytes (`Some(N * 16)` for `"uniform"`
    /// bindings, `None` for non-buffer bindings such as textures and samplers).
    pub min_binding_size_bytes: Option<u32>,
    /// Highest `vec4` index statically read from this uniform CBV (`Some(max_vec4_index)` for
    /// `"uniform"` bindings, `None` for non-buffer bindings).
    pub highest_vec4_offset: Option<u32>,
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
        if let Some(min_bytes) = b.min_binding_size_bytes {
            let _ = write!(out, ", \"min_binding_size_bytes\": {min_bytes}");
        }
        if let Some(highest_vec4) = b.highest_vec4_offset {
            let _ = write!(out, ", \"highest_vec4_offset\": {highest_vec4}");
        }
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
    pub has_storage_buffer: bool,
    pub has_storage_texture: bool,
    pub has_lds: bool,
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
        let _ = writeln!(out, "    \"has_storage_buffer\": {},", bs.has_storage_buffer);
        let _ = writeln!(
            out,
            "    \"has_storage_texture\": {},",
            bs.has_storage_texture
        );
        let _ = writeln!(out, "    \"has_lds\": {},", bs.has_lds);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CbvKey {
    /// Base SGPR pair index (root descriptor table `sbase` for coalesced V# loads,
    /// or direct SMEM `sbase` for direct CBV loads).
    base_sbase: u8,
    /// `Some(table_offset)` for a coalesced descriptor-table V# entry, `None` for direct SMEM.
    table_offset: Option<i32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CbvUsage {
    max_vec4_index: u32,
    has_dynamic_offset: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CbvSlotPlan {
    key: CbvKey,
    vec4_count: u32,
    highest_vec4_offset: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct StorageBufSlotPlan {
    srsrc: u8,
    writable: bool,
}

struct ProgramPlan {
    stage: ShaderStage,
    entry_point: String,
    workgroup_size: [u32; 3],
    compute_seed_wg_sgprs: Vec<(u8, u32)>,
    active_inst_count: usize,
    vertex_input_mode: VertexInputMode,
    vertex_attrs: Vec<VertexInputAttr>,
    has_buffer_fetch: bool,
    legacy_uv_attr_index: Option<usize>,
    vsharp_fetch_insts: BTreeSet<usize>,
    indirect_smem_cbv_keys: BTreeMap<usize, CbvKey>,
    cbv_slots: Vec<CbvSlotPlan>,
    cbv_vec4_count: u32,
    tex_slots: Vec<u8>,
    samp_slots: Vec<u8>,
    storage_buf_slots: Vec<StorageBufSlotPlan>,
    storage_tex_slots: Vec<u8>,
    uses_lds: bool,
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

#[derive(Debug, Clone, Copy)]
struct MimgAddrLayout {
    total_vgprs: u8,
    bias_off: Option<u8>,
    compare_off: Option<u8>,
    grad_off: Option<u8>,
    uv_off: u8,
    lod_off: Option<u8>,
    is_zero_lod: bool,
    is_load: bool,
    is_store: bool,
}

/// Decode GFX10.3 2D MIMG address VGPR layout (`[offset], [bias], [z_compare], [grad], u, v, [lod/clamp]`).
fn mimg_addr_layout(op: u8) -> Option<MimgAddrLayout> {
    let base = MimgAddrLayout {
        total_vgprs: 2,
        bias_off: None,
        compare_off: None,
        grad_off: None,
        uv_off: 0,
        lod_off: None,
        is_zero_lod: false,
        is_load: false,
        is_store: false,
    };
    match op {
        // image_load: (x, y)
        0x00 => Some(MimgAddrLayout {
            is_load: true,
            ..base
        }),
        // image_load_mip: (x, y, mip)
        0x01 => Some(MimgAddrLayout {
            total_vgprs: 3,
            lod_off: Some(2),
            is_load: true,
            ..base
        }),
        // image_store / image_store_pck: (x, y)
        0x08 | 0x0A => Some(MimgAddrLayout {
            is_store: true,
            ..base
        }),
        // image_store_mip / image_store_mip_pck: (x, y, mip)
        0x09 | 0x0B => Some(MimgAddrLayout {
            total_vgprs: 3,
            lod_off: Some(2),
            is_store: true,
            ..base
        }),
        // image_sample: (u, v)
        0x20 => Some(base),
        // image_sample_cl: (u, v, clamp)
        0x21 => Some(MimgAddrLayout {
            total_vgprs: 3,
            ..base
        }),
        // image_sample_d: (dudx, dvdx, dudy, dvdy, u, v)
        0x22 => Some(MimgAddrLayout {
            total_vgprs: 6,
            grad_off: Some(0),
            uv_off: 4,
            ..base
        }),
        // image_sample_l: (u, v, lod)
        0x24 => Some(MimgAddrLayout {
            total_vgprs: 3,
            lod_off: Some(2),
            ..base
        }),
        // image_sample_b: (bias, u, v)
        0x25 => Some(MimgAddrLayout {
            total_vgprs: 3,
            bias_off: Some(0),
            uv_off: 1,
            ..base
        }),
        // image_sample_lz: (u, v)
        0x27 => Some(MimgAddrLayout {
            is_zero_lod: true,
            ..base
        }),
        // image_sample_c: (z_ref, u, v)
        0x28 => Some(MimgAddrLayout {
            total_vgprs: 3,
            compare_off: Some(0),
            uv_off: 1,
            ..base
        }),
        // image_sample_c_l: (z_ref, u, v, lod)
        0x2C => Some(MimgAddrLayout {
            total_vgprs: 4,
            compare_off: Some(0),
            uv_off: 1,
            lod_off: Some(3),
            ..base
        }),
        // image_sample_c_b: (bias, z_ref, u, v)
        0x2D => Some(MimgAddrLayout {
            total_vgprs: 4,
            bias_off: Some(0),
            compare_off: Some(1),
            uv_off: 2,
            ..base
        }),
        // image_sample_c_lz: (z_ref, u, v)
        0x2F => Some(MimgAddrLayout {
            total_vgprs: 3,
            compare_off: Some(0),
            uv_off: 1,
            is_zero_lod: true,
            ..base
        }),
        // image_sample_o: (offset, u, v)
        0x30 => Some(MimgAddrLayout {
            total_vgprs: 3,
            uv_off: 1,
            ..base
        }),
        // image_sample_l_o: (offset, u, v, lod)
        0x34 => Some(MimgAddrLayout {
            total_vgprs: 4,
            uv_off: 1,
            lod_off: Some(3),
            ..base
        }),
        // image_sample_b_o: (offset, bias, u, v)
        0x35 => Some(MimgAddrLayout {
            total_vgprs: 4,
            bias_off: Some(1),
            uv_off: 2,
            ..base
        }),
        // image_sample_lz_o: (offset, u, v)
        0x37 => Some(MimgAddrLayout {
            total_vgprs: 3,
            uv_off: 1,
            is_zero_lod: true,
            ..base
        }),
        // image_sample_c_o: (offset, z_ref, u, v)
        0x38 => Some(MimgAddrLayout {
            total_vgprs: 4,
            compare_off: Some(1),
            uv_off: 2,
            ..base
        }),
        // image_sample_c_lz_o: (offset, z_ref, u, v)
        0x3F => Some(MimgAddrLayout {
            total_vgprs: 4,
            compare_off: Some(1),
            uv_off: 2,
            is_zero_lod: true,
            ..base
        }),
        _ => None,
    }
}

fn vop3p_is_ternary(op: u8) -> bool {
    !matches!(op, 0x01..=0x08 | 0x0A..=0x0D | 0x0F..=0x12)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StorageBufOpKind {
    LoadUbyte,
    LoadSbyte,
    LoadUshort,
    LoadSshort,
    LoadDwords(u8),
    StoreDwords(u8),
}

impl StorageBufOpKind {
    const fn is_store(self) -> bool {
        matches!(self, Self::StoreDwords(_))
    }

    const fn vdata_count(self) -> u8 {
        match self {
            Self::LoadUbyte | Self::LoadSbyte | Self::LoadUshort | Self::LoadSshort => 1,
            Self::LoadDwords(n) | Self::StoreDwords(n) => n,
        }
    }

    const fn elem_stride_bytes(self) -> u32 {
        match self {
            Self::LoadUbyte | Self::LoadSbyte => 1,
            Self::LoadUshort | Self::LoadSshort => 2,
            Self::LoadDwords(n) | Self::StoreDwords(n) => (n as u32) * 4,
        }
    }
}

fn mubuf_storage_op(op: u8) -> Option<StorageBufOpKind> {
    match op {
        0x00 | 0x0C => Some(StorageBufOpKind::LoadDwords(1)),
        0x01 | 0x0D => Some(StorageBufOpKind::LoadDwords(2)),
        0x02 | 0x0F => Some(StorageBufOpKind::LoadDwords(3)),
        0x03 | 0x0E => Some(StorageBufOpKind::LoadDwords(4)),
        0x08 => Some(StorageBufOpKind::LoadUbyte),
        0x09 => Some(StorageBufOpKind::LoadSbyte),
        0x0A => Some(StorageBufOpKind::LoadUshort),
        0x0B => Some(StorageBufOpKind::LoadSshort),
        0x04 | 0x1C => Some(StorageBufOpKind::StoreDwords(1)),
        0x05 | 0x1D => Some(StorageBufOpKind::StoreDwords(2)),
        0x06 | 0x1F => Some(StorageBufOpKind::StoreDwords(3)),
        0x07 | 0x1E => Some(StorageBufOpKind::StoreDwords(4)),
        _ => None,
    }
}

const fn mubuf_is_atomic(op: u8) -> bool {
    matches!(op, 0x32..=0x3A)
}

fn mtbuf_storage_op(op: u8) -> Option<StorageBufOpKind> {
    match op {
        0x00 => Some(StorageBufOpKind::LoadDwords(1)),
        0x01 => Some(StorageBufOpKind::LoadDwords(2)),
        0x02 => Some(StorageBufOpKind::LoadDwords(3)),
        0x03 => Some(StorageBufOpKind::LoadDwords(4)),
        0x04 => Some(StorageBufOpKind::StoreDwords(1)),
        0x05 => Some(StorageBufOpKind::StoreDwords(2)),
        0x06 => Some(StorageBufOpKind::StoreDwords(3)),
        0x07 => Some(StorageBufOpKind::StoreDwords(4)),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DsLdsOp {
    ReadContiguous(u8),
    WriteContiguous(u8),
    Read2 { stride_bytes: u32 },
    Write2 { stride_bytes: u32 },
}

fn ds_lds_op(op: u8) -> Option<DsLdsOp> {
    match op {
        0x0D => Some(DsLdsOp::WriteContiguous(1)),
        0x0E => Some(DsLdsOp::Write2 { stride_bytes: 4 }),
        0x0F => Some(DsLdsOp::Write2 { stride_bytes: 256 }),
        0x36 => Some(DsLdsOp::ReadContiguous(1)),
        0x37 => Some(DsLdsOp::Read2 { stride_bytes: 4 }),
        0x38 => Some(DsLdsOp::Read2 { stride_bytes: 256 }),
        0x4D => Some(DsLdsOp::WriteContiguous(2)),
        0x76 => Some(DsLdsOp::ReadContiguous(2)),
        0xDE => Some(DsLdsOp::WriteContiguous(3)),
        0xDF => Some(DsLdsOp::WriteContiguous(4)),
        0xFE => Some(DsLdsOp::ReadContiguous(3)),
        0xFF => Some(DsLdsOp::ReadContiguous(4)),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn record_src_usage(
    src: u16,
    vgprs: &mut BTreeSet<u8>,
    sgprs: &mut BTreeSet<u8>,
    written_vgprs: &BTreeSet<u8>,
    read_before_write_vgprs: &mut BTreeSet<u8>,
    written_sgprs: &BTreeSet<u8>,
    read_before_write_sgprs: &mut BTreeSet<u8>,
    needs_vcc: &mut bool,
    needs_scc: &mut bool,
) {
    match src {
        0..=105 => {
            let s = src as u8;
            sgprs.insert(s);
            if !written_sgprs.contains(&s) {
                read_before_write_sgprs.insert(s);
            }
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

#[derive(Debug, Clone)]
struct VSharpCandidate {
    def_inst_idx: usize,
    dst_sgpr: u8,
    table_sbase: u8,
    table_offset: i32,
    buffer_load_users: Vec<usize>,
    tainted_by_scalar_read: bool,
    /// Vertex-shader `idxen` MUBUF/MTBUF attribute fetches using this V# as `srsrc`.
    vertex_fetch_users: usize,
    /// Consumed as `srsrc` by a non-vertex-attribute buffer access (storage, non-`idxen`).
    used_as_other_buffer: bool,
}

fn sgpr_ranges_overlap(a_start: u8, a_len: u8, b_start: u8, b_len: u8) -> bool {
    let a_end = a_start.saturating_add(a_len);
    let b_end = b_start.saturating_add(b_len);
    a_start < b_end && b_start < a_end
}

fn taint_vsharp_if_sgpr_read(live: &mut [VSharpCandidate], src: u16) {
    if src <= 105 {
        let r = src as u8;
        for cand in live.iter_mut() {
            if r >= cand.dst_sgpr && r < cand.dst_sgpr.saturating_add(4) {
                cand.tainted_by_scalar_read = true;
            }
        }
    }
}

fn retire_overwritten_vsharps(
    live: &mut Vec<VSharpCandidate>,
    finished: &mut Vec<VSharpCandidate>,
    start: u8,
    len: u8,
) {
    if start > 105 || len == 0 {
        return;
    }
    let mut i = 0;
    while i < live.len() {
        if sgpr_ranges_overlap(live[i].dst_sgpr, 4, start, len) {
            finished.push(live.swap_remove(i));
        } else {
            i += 1;
        }
    }
}

/// Detect 2-step descriptor-table V# indirections:
///   1. `s_load_dwordx4 s[D:D+3], s[table_ptr], static_offset` (`op == 0x02`)
///   2. `s_buffer_load_* s[...], s[D:D+3], cbv_offset` (`op in 0x08..=0x0C`)
///
/// Matches only when `s[D:D+3]` is NOT consumed as ordinary scalar arithmetic/color data during its live range.
///
/// Returns `(vsharp_fetch_insts, indirect_smem_cbv_keys)` so the V# fetch itself does not
/// allocate a spurious uniform buffer binding and the downstream `s_buffer_load_*` binds
/// directly to the descriptor-table CBV slot (e.g. `cb0` at `@group(0) @binding(0)`).
///
/// In vertex shaders, a `s_load_dwordx4` V# that is consumed ONLY as the `srsrc` of `idxen`
/// MUBUF/MTBUF attribute fetches (vertex-buffer descriptor, lowered to `VsIn` inputs) is
/// also reported in `vsharp_fetch_insts`, so it never allocates a bogus `cbN_` binding.
fn analyze_vsharp_indirections(
    active: &[Rdna2Instruction],
    stage: ShaderStage,
) -> (BTreeSet<usize>, BTreeMap<usize, CbvKey>) {
    let mut live: Vec<VSharpCandidate> = Vec::new();
    let mut finished: Vec<VSharpCandidate> = Vec::new();

    for (inst_idx, inst) in active.iter().enumerate() {
        if inst.truncated {
            continue;
        }
        match &inst.detail {
            Rdna2Detail::Sopc { ssrc0, ssrc1, .. } => {
                taint_vsharp_if_sgpr_read(&mut live, u16::from(*ssrc0));
                taint_vsharp_if_sgpr_read(&mut live, u16::from(*ssrc1));
            }
            Rdna2Detail::Sop1 {
                op, sdst, ssrc0, ..
            } => {
                taint_vsharp_if_sgpr_read(&mut live, u16::from(*ssrc0));
                if matches!(*op, 0x04 | 0x0A) {
                    if *ssrc0 <= 104 {
                        taint_vsharp_if_sgpr_read(&mut live, u16::from(*ssrc0 + 1));
                    }
                    if *sdst <= 104 {
                        retire_overwritten_vsharps(&mut live, &mut finished, *sdst, 2);
                    } else if *sdst == 105 {
                        retire_overwritten_vsharps(&mut live, &mut finished, 105, 1);
                    }
                } else {
                    if *op == 0x05 {
                        taint_vsharp_if_sgpr_read(&mut live, u16::from(*sdst));
                    }
                    if *sdst <= 105 {
                        retire_overwritten_vsharps(&mut live, &mut finished, *sdst, 1);
                    }
                }
            }
            Rdna2Detail::Sopk { op, sdst, .. } => {
                if matches!(*op, 0x0F | 0x10) {
                    taint_vsharp_if_sgpr_read(&mut live, u16::from(*sdst));
                }
                if !matches!(*op, 0x17..=0x1A) && *sdst <= 105 {
                    retire_overwritten_vsharps(&mut live, &mut finished, *sdst, 1);
                }
            }
            Rdna2Detail::Sop2 {
                sdst, ssrc0, ssrc1, ..
            } => {
                taint_vsharp_if_sgpr_read(&mut live, u16::from(*ssrc0));
                taint_vsharp_if_sgpr_read(&mut live, u16::from(*ssrc1));
                if *sdst <= 105 {
                    retire_overwritten_vsharps(&mut live, &mut finished, *sdst, 1);
                }
            }
            Rdna2Detail::Smem {
                op,
                sdata,
                sbase,
                soffset,
                offset,
                ..
            } => {
                let is_dynamic = *soffset != 0x7D && *soffset <= 105;
                if is_dynamic {
                    taint_vsharp_if_sgpr_read(&mut live, u16::from(*soffset));
                }
                let sbase_sgpr = sbase.saturating_mul(2);
                if matches!(*op, 0x08..=0x0C) {
                    for cand in live.iter_mut() {
                        if cand.dst_sgpr == sbase_sgpr {
                            cand.buffer_load_users.push(inst_idx);
                        } else if sgpr_ranges_overlap(cand.dst_sgpr, 4, sbase_sgpr, 4) {
                            cand.tainted_by_scalar_read = true;
                        }
                    }
                } else if smem_load_dword_count(*op).is_some() {
                    for cand in live.iter_mut() {
                        if sgpr_ranges_overlap(cand.dst_sgpr, 4, sbase_sgpr, 2) {
                            cand.tainted_by_scalar_read = true;
                        }
                    }
                }
                if let Some(count) = smem_load_dword_count(*op) {
                    retire_overwritten_vsharps(&mut live, &mut finished, *sdata, count);
                }
                if *op == 0x02 && !is_dynamic && *sdata <= 102 && (*sdata % 2 == 0) {
                    let inline_off = if (128..=192).contains(soffset) {
                        i32::from(*soffset - 128)
                    } else {
                        0
                    };
                    live.push(VSharpCandidate {
                        def_inst_idx: inst_idx,
                        dst_sgpr: *sdata,
                        table_sbase: *sbase,
                        table_offset: offset.saturating_add(inline_off),
                        buffer_load_users: Vec::new(),
                        tainted_by_scalar_read: false,
                        vertex_fetch_users: 0,
                        used_as_other_buffer: false,
                    });
                }
            }
            Rdna2Detail::Vop1 { op, vdst, src0, .. } => {
                if !matches!(op, 0x00 | 0x1B | 0x41) {
                    taint_vsharp_if_sgpr_read(&mut live, *src0);
                    if *op == 0x02 {
                        let sd = *vdst & 0x7F;
                        if sd <= 105 {
                            retire_overwritten_vsharps(&mut live, &mut finished, sd, 1);
                        }
                    }
                }
            }
            Rdna2Detail::Vopc { src0, .. } | Rdna2Detail::Vop2 { src0, .. } => {
                taint_vsharp_if_sgpr_read(&mut live, *src0);
            }
            Rdna2Detail::Vop3 {
                op,
                vdst,
                sdst,
                src0,
                src1,
                src2,
                ..
            } => {
                taint_vsharp_if_sgpr_read(&mut live, *src0);
                let is_unary = (0x180..=0x1FF).contains(op);
                if !is_unary {
                    taint_vsharp_if_sgpr_read(&mut live, *src1);
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
                    taint_vsharp_if_sgpr_read(&mut live, *src2);
                }
                if let Some(sd) = sdst {
                    let sd7 = *sd & 0x7F;
                    if sd7 <= 105 {
                        retire_overwritten_vsharps(&mut live, &mut finished, sd7, 1);
                    }
                }
                if *op <= 0xFF {
                    let sd = *vdst & 0x7F;
                    if sd <= 105 {
                        retire_overwritten_vsharps(&mut live, &mut finished, sd, 1);
                    }
                }
            }
            Rdna2Detail::Vop3P {
                op,
                src0,
                src1,
                src2,
                ..
            } => {
                taint_vsharp_if_sgpr_read(&mut live, *src0);
                taint_vsharp_if_sgpr_read(&mut live, *src1);
                if vop3p_is_ternary(*op) {
                    taint_vsharp_if_sgpr_read(&mut live, *src2);
                }
            }
            Rdna2Detail::Mubuf {
                op,
                idxen,
                srsrc,
                soffset,
                ..
            } => {
                taint_vsharp_if_sgpr_read(&mut live, u16::from(*soffset));
                let is_vertex_fetch =
                    stage == ShaderStage::Vertex && *idxen && mubuf_load_components(*op).is_some();
                record_vsharp_buffer_use(&mut live, *srsrc, is_vertex_fetch);
            }
            Rdna2Detail::Mtbuf {
                op,
                idxen,
                srsrc,
                soffset,
                ..
            } => {
                taint_vsharp_if_sgpr_read(&mut live, u16::from(*soffset));
                let is_vertex_fetch =
                    stage == ShaderStage::Vertex && *idxen && mtbuf_load_components(*op).is_some();
                record_vsharp_buffer_use(&mut live, *srsrc, is_vertex_fetch);
            }
            Rdna2Detail::Flat { saddr, .. } => {
                taint_vsharp_if_sgpr_read(&mut live, u16::from(*saddr));
                if *saddr <= 104 {
                    taint_vsharp_if_sgpr_read(&mut live, u16::from(*saddr + 1));
                }
            }
            Rdna2Detail::Sopp { .. }
            | Rdna2Detail::Vintrp { .. }
            | Rdna2Detail::Mimg { .. }
            | Rdna2Detail::Ds { .. }
            | Rdna2Detail::Exp { .. }
            | Rdna2Detail::Unknown { .. } => {}
        }
    }

    finished.extend(live);

    let mut vsharp_fetch_insts = BTreeSet::new();
    let mut indirect_smem_cbv_keys = BTreeMap::new();
    for cand in finished {
        if !cand.tainted_by_scalar_read && !cand.buffer_load_users.is_empty() {
            vsharp_fetch_insts.insert(cand.def_inst_idx);
            let key = CbvKey {
                base_sbase: cand.table_sbase,
                table_offset: Some(cand.table_offset),
            };
            for user_idx in cand.buffer_load_users {
                indirect_smem_cbv_keys.insert(user_idx, key);
            }
        } else if !cand.tainted_by_scalar_read
            && cand.buffer_load_users.is_empty()
            && cand.vertex_fetch_users > 0
            && !cand.used_as_other_buffer
        {
            // Vertex-buffer V#: attribute fetches lower to `VsIn`, no uniform binding needed.
            vsharp_fetch_insts.insert(cand.def_inst_idx);
        }
    }

    (vsharp_fetch_insts, indirect_smem_cbv_keys)
}

/// Record a MUBUF/MTBUF `srsrc` (in units of 4 SGPRs) consuming a live V# candidate.
fn record_vsharp_buffer_use(live: &mut [VSharpCandidate], srsrc: u8, is_vertex_fetch: bool) {
    let rsrc_sgpr = srsrc.saturating_mul(4);
    for cand in live.iter_mut() {
        if cand.dst_sgpr == rsrc_sgpr && is_vertex_fetch {
            cand.vertex_fetch_users += 1;
        } else if sgpr_ranges_overlap(cand.dst_sgpr, 4, rsrc_sgpr, 4) {
            cand.used_as_other_buffer = true;
        }
    }
}

/// Map a vertex-shader MUBUF/MTBUF attribute fetch to `(attribute_byte_offset, component_shift)`.
///
/// A 1-component 8-bit `tbuffer_load_format_x` (`BUF_FMT_8_*`, formats 1..=6) at a byte offset
/// that is not dword aligned reads a single channel of a packed 4x8-bit attribute (e.g.
/// `offset:7` on an `R8G8B8A8_UNORM` element at `+4` reads `.w`). Such fetches are folded into
/// the dword-aligned `vec4<f32>` attribute (matching WebGPU `unorm8x4`) with a component shift.
fn vertex_fetch_slot(is_mtbuf: bool, op: u8, format: u8, byte_offset: u16) -> (u16, u8) {
    let sub_dword = (byte_offset % 4) as u8;
    if is_mtbuf && op == 0x00 && (1..=6).contains(&format) && sub_dword != 0 {
        (byte_offset - u16::from(sub_dword), sub_dword)
    } else {
        (byte_offset, 0)
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
    let (vsharp_fetch_insts, indirect_smem_cbv_keys) = analyze_vsharp_indirections(active, stage);

    let mut cbv_usage: BTreeMap<CbvKey, CbvUsage> = BTreeMap::new();
    let mut tex_rsrc_set: BTreeSet<u8> = BTreeSet::new();
    let mut samp_rsrc_set: BTreeSet<u8> = BTreeSet::new();
    let mut storage_buf_usage: BTreeMap<u8, bool> = BTreeMap::new();
    let mut storage_tex_rsrc_set: BTreeSet<u8> = BTreeSet::new();
    let mut uses_lds = false;
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
    let mut written_sgprs: BTreeSet<u8> = BTreeSet::new();
    let mut read_before_write_sgprs: BTreeSet<u8> = BTreeSet::new();
    let mut needs_vcc = false;
    let mut needs_scc = false;

    for (inst_idx, inst) in active.iter().enumerate() {
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
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    u16::from(*ssrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
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
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if matches!(*op, 0x04 | 0x0A) {
                    // s_mov_b64 / s_wqm_b64: 64-bit SGPR pair source and destination
                    if *ssrc0 <= 104 {
                        let hi = *ssrc0 + 1;
                        sgprs.insert(hi);
                        if !written_sgprs.contains(&hi) {
                            read_before_write_sgprs.insert(hi);
                        }
                    }
                    if *sdst <= 104 {
                        sgprs.insert(*sdst);
                        sgprs.insert(*sdst + 1);
                        written_sgprs.insert(*sdst);
                        written_sgprs.insert(*sdst + 1);
                    } else if *sdst == 105 {
                        sgprs.insert(105);
                        written_sgprs.insert(105);
                    } else if *sdst == 106 {
                        needs_vcc = true;
                    }
                } else {
                    if *op == 0x05 {
                        needs_scc = true;
                        if *sdst <= 105 {
                            sgprs.insert(*sdst);
                            if !written_sgprs.contains(sdst) {
                                read_before_write_sgprs.insert(*sdst);
                            }
                        }
                    }
                    if *sdst <= 105 {
                        sgprs.insert(*sdst);
                        written_sgprs.insert(*sdst);
                    }
                }
            }
            Rdna2Detail::Sopk { op, sdst, .. } => {
                if *sdst <= 105 {
                    if matches!(*op, 0x0F | 0x10) && !written_sgprs.contains(sdst) {
                        read_before_write_sgprs.insert(*sdst);
                    }
                    sgprs.insert(*sdst);
                    written_sgprs.insert(*sdst);
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
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    u16::from(*ssrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if *op == 0x0A {
                    needs_scc = true;
                }
                if *sdst <= 105 {
                    sgprs.insert(*sdst);
                    written_sgprs.insert(*sdst);
                }
            }
            Rdna2Detail::Smem {
                op,
                sdata,
                sbase,
                soffset,
                offset,
                ..
            } => {
                if vsharp_fetch_insts.contains(&inst_idx) {
                    continue;
                }
                if let Some(count) = smem_load_dword_count(*op) {
                    let is_dynamic = *soffset != 0x7D && *soffset <= 105;
                    if is_dynamic {
                        sgprs.insert(*soffset);
                        if !written_sgprs.contains(soffset) {
                            read_before_write_sgprs.insert(*soffset);
                        }
                    }
                    let base_word = if *offset > 0 {
                        (*offset as u32) / 4
                    } else {
                        0
                    };
                    let loaded_dwords = if *sdata <= 105 {
                        u32::from(count).min(u32::from(106 - *sdata))
                    } else {
                        u32::from(count)
                    }
                    .max(1);
                    let highest_dword = base_word.saturating_add(loaded_dwords - 1);
                    let highest_vec4 = highest_dword / 4;
                    let cbv_key = indirect_smem_cbv_keys
                        .get(&inst_idx)
                        .copied()
                        .unwrap_or(CbvKey {
                            base_sbase: *sbase,
                            table_offset: None,
                        });
                    cbv_usage
                        .entry(cbv_key)
                        .and_modify(|usage| {
                            usage.max_vec4_index = usage.max_vec4_index.max(highest_vec4);
                            usage.has_dynamic_offset |= is_dynamic;
                        })
                        .or_insert(CbvUsage {
                            max_vec4_index: highest_vec4,
                            has_dynamic_offset: is_dynamic,
                        });
                    for k in 0..count {
                        let reg = sdata.saturating_add(k);
                        if reg <= 105 {
                            sgprs.insert(reg);
                            written_sgprs.insert(reg);
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
                        &written_sgprs,
                        &mut read_before_write_sgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                    if *op == 0x02 {
                        let sd = *vdst & 0x7F;
                        if sd <= 105 {
                            sgprs.insert(sd);
                            written_sgprs.insert(sd);
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
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    256 + u16::from(*vsrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
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
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    256 + u16::from(*vsrc1),
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if matches!(op, 0x02 | 0x06 | 0x2B | 0x36 | 0x3C) {
                    // v_dot2c_f32_f16 / v_fmac* / v_pk_fmac_f16 read vdst as accumulator
                    record_src_usage(
                        256 + u16::from(*vdst),
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &written_sgprs,
                        &mut read_before_write_sgprs,
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
                sdst,
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
                    &written_sgprs,
                    &mut read_before_write_sgprs,
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
                        &written_sgprs,
                        &mut read_before_write_sgprs,
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
                        &written_sgprs,
                        &mut read_before_write_sgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
                if let Some(sd) = sdst {
                    let sd7 = *sd & 0x7F;
                    if sd7 <= 105 {
                        sgprs.insert(sd7);
                        written_sgprs.insert(sd7);
                    } else if sd7 == 106 {
                        needs_vcc = true;
                    }
                }
                if *op <= 0xFF {
                    let sd = *vdst & 0x7F;
                    if sd <= 105 {
                        sgprs.insert(sd);
                        written_sgprs.insert(sd);
                    } else if sd == 106 {
                        needs_vcc = true;
                    }
                } else {
                    vgprs.insert(*vdst);
                    written_vgprs.insert(*vdst);
                }
            }
            Rdna2Detail::Vop3P {
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
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                record_src_usage(
                    *src1,
                    &mut vgprs,
                    &mut sgprs,
                    &written_vgprs,
                    &mut read_before_write_vgprs,
                    &written_sgprs,
                    &mut read_before_write_sgprs,
                    &mut needs_vcc,
                    &mut needs_scc,
                );
                if vop3p_is_ternary(*op) {
                    record_src_usage(
                        *src2,
                        &mut vgprs,
                        &mut sgprs,
                        &written_vgprs,
                        &mut read_before_write_vgprs,
                        &written_sgprs,
                        &mut read_before_write_sgprs,
                        &mut needs_vcc,
                        &mut needs_scc,
                    );
                }
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
                srsrc,
                ssamp,
                ..
            } => {
                if let Some(layout) = mimg_addr_layout(*op) {
                    if layout.is_store {
                        if stage != ShaderStage::Vertex {
                            storage_tex_rsrc_set.insert(*srsrc);
                            for a in 0..layout.total_vgprs {
                                let reg = vaddr.saturating_add(a);
                                vgprs.insert(reg);
                                if !written_vgprs.contains(&reg) {
                                    read_before_write_vgprs.insert(reg);
                                }
                            }
                            let eff_mask = if *dmask == 0 { 0xF } else { *dmask };
                            let in_cnt = eff_mask.count_ones() as u8;
                            for d in 0..in_cnt {
                                let reg = vdata.saturating_add(d);
                                vgprs.insert(reg);
                                if !written_vgprs.contains(&reg) {
                                    read_before_write_vgprs.insert(reg);
                                }
                            }
                        }
                    } else {
                        tex_rsrc_set.insert(*srsrc);
                        if !layout.is_load {
                            samp_rsrc_set.insert(*ssamp);
                        }
                        for a in 0..layout.total_vgprs {
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
            }
            Rdna2Detail::Mubuf {
                op,
                offset,
                offen,
                idxen,
                vaddr,
                vdata,
                srsrc,
                soffset,
                ..
            } => {
                if stage == ShaderStage::Vertex {
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
                } else if let Some(buf_op) = mubuf_storage_op(*op) {
                    storage_buf_usage
                        .entry(*srsrc)
                        .and_modify(|w| *w |= buf_op.is_store())
                        .or_insert(buf_op.is_store());
                    if *offen || *idxen {
                        vgprs.insert(*vaddr);
                        if !written_vgprs.contains(vaddr) {
                            read_before_write_vgprs.insert(*vaddr);
                        }
                    }
                    if *offen && *idxen {
                        let v1 = vaddr.saturating_add(1);
                        vgprs.insert(v1);
                        if !written_vgprs.contains(&v1) {
                            read_before_write_vgprs.insert(v1);
                        }
                    }
                    if *soffset <= 105 {
                        sgprs.insert(*soffset);
                        if !written_sgprs.contains(soffset) {
                            read_before_write_sgprs.insert(*soffset);
                        }
                    }
                    let cnt = buf_op.vdata_count();
                    if buf_op.is_store() {
                        for d in 0..cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            if !written_vgprs.contains(&reg) {
                                read_before_write_vgprs.insert(reg);
                            }
                        }
                    } else {
                        for d in 0..cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                    }
                } else if mubuf_is_atomic(*op) {
                    storage_buf_usage.insert(*srsrc, true);
                }
            }
            Rdna2Detail::Mtbuf {
                op,
                format,
                offset,
                offen,
                idxen,
                vaddr,
                vdata,
                srsrc,
                soffset,
                ..
            } => {
                if stage == ShaderStage::Vertex {
                    if let Some(comps) = mtbuf_load_components(*op) {
                        has_buffer_fetch = true;
                        let (attr_offset, shift) = vertex_fetch_slot(true, *op, *format, *offset);
                        let attr_comps = if shift != 0 { 4 } else { comps };
                        vertex_buf_fetches
                            .entry(attr_offset)
                            .and_modify(|c| *c = (*c).max(attr_comps))
                            .or_insert(attr_comps);
                        for d in 0..comps {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                    }
                } else if let Some(buf_op) = mtbuf_storage_op(*op) {
                    storage_buf_usage
                        .entry(*srsrc)
                        .and_modify(|w| *w |= buf_op.is_store())
                        .or_insert(buf_op.is_store());
                    if *offen || *idxen {
                        vgprs.insert(*vaddr);
                        if !written_vgprs.contains(vaddr) {
                            read_before_write_vgprs.insert(*vaddr);
                        }
                    }
                    if *offen && *idxen {
                        let v1 = vaddr.saturating_add(1);
                        vgprs.insert(v1);
                        if !written_vgprs.contains(&v1) {
                            read_before_write_vgprs.insert(v1);
                        }
                    }
                    if *soffset <= 105 {
                        sgprs.insert(*soffset);
                        if !written_sgprs.contains(soffset) {
                            read_before_write_sgprs.insert(*soffset);
                        }
                    }
                    let cnt = buf_op.vdata_count();
                    if buf_op.is_store() {
                        for d in 0..cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            if !written_vgprs.contains(&reg) {
                                read_before_write_vgprs.insert(reg);
                            }
                        }
                    } else {
                        for d in 0..cnt {
                            let reg = vdata.saturating_add(d);
                            vgprs.insert(reg);
                            written_vgprs.insert(reg);
                        }
                    }
                }
            }
            Rdna2Detail::Ds {
                op,
                addr,
                data0,
                data1,
                vdst,
                ..
            } => {
                if stage == ShaderStage::Compute
                    && let Some(lds_op) = ds_lds_op(*op)
                {
                    uses_lds = true;
                    vgprs.insert(*addr);
                    if !written_vgprs.contains(addr) {
                        read_before_write_vgprs.insert(*addr);
                    }
                    match lds_op {
                        DsLdsOp::ReadContiguous(n) => {
                            for k in 0..n {
                                let reg = vdst.saturating_add(k);
                                vgprs.insert(reg);
                                written_vgprs.insert(reg);
                            }
                        }
                        DsLdsOp::Read2 { .. } => {
                            for k in 0..2u8 {
                                let reg = vdst.saturating_add(k);
                                vgprs.insert(reg);
                                written_vgprs.insert(reg);
                            }
                        }
                        DsLdsOp::WriteContiguous(n) => {
                            for k in 0..n {
                                let reg = data0.saturating_add(k);
                                vgprs.insert(reg);
                                if !written_vgprs.contains(&reg) {
                                    read_before_write_vgprs.insert(reg);
                                }
                            }
                        }
                        DsLdsOp::Write2 { .. } => {
                            for &reg in &[data0, data1] {
                                vgprs.insert(*reg);
                                if !written_vgprs.contains(reg) {
                                    read_before_write_vgprs.insert(*reg);
                                }
                            }
                        }
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

    let workgroup_size = [
        options.workgroup_size[0].max(1),
        options.workgroup_size[1].max(1),
        options.workgroup_size[2].max(1),
    ];
    let mut compute_seed_wg_sgprs: Vec<(u8, u32)> = Vec::new();
    if stage == ShaderStage::Compute {
        vgprs.insert(0);
        let wg_base = options.workgroup_id_sgpr_base.unwrap_or(0);
        for c in 0..3u8 {
            let reg = wg_base.saturating_add(c);
            if reg <= 105
                && sgprs.contains(&reg)
                && (options.workgroup_id_sgpr_base.is_some()
                    || read_before_write_sgprs.contains(&reg))
            {
                compute_seed_wg_sgprs.push((reg, u32::from(c)));
            }
        }
    }

    let mut ps_mrts: Vec<u8> = ps_mrt_set.into_iter().collect();
    if stage == ShaderStage::Fragment && ps_mrts.is_empty() {
        ps_mrts.push(0);
    }

    if !tex_rsrc_set.is_empty() && samp_rsrc_set.is_empty() {
        samp_rsrc_set.insert(0);
    }

    let min_vec4s = options.min_cbv_vec4_count.clamp(1, MAX_CBV_VEC4_COUNT);
    let cbv_slots: Vec<CbvSlotPlan> = cbv_usage
        .into_iter()
        .map(|(key, usage)| {
            let static_needed = usage.max_vec4_index.saturating_add(1);
            let floor = if usage.has_dynamic_offset {
                min_vec4s.max(DEFAULT_DYNAMIC_CBV_VEC4_COUNT)
            } else {
                min_vec4s
            };
            let vec4_count = static_needed.max(floor).clamp(1, MAX_CBV_VEC4_COUNT);
            let highest_vec4_offset = usage.max_vec4_index.min(vec4_count - 1);
            CbvSlotPlan {
                key,
                vec4_count,
                highest_vec4_offset,
            }
        })
        .collect();
    let tex_slots: Vec<u8> = tex_rsrc_set.into_iter().collect();
    let samp_slots: Vec<u8> = samp_rsrc_set.into_iter().collect();
    let storage_buf_slots: Vec<StorageBufSlotPlan> = storage_buf_usage
        .into_iter()
        .map(|(srsrc, writable)| StorageBufSlotPlan { srsrc, writable })
        .collect();
    let storage_tex_slots: Vec<u8> = storage_tex_rsrc_set.into_iter().collect();

    let cbv_vec4_count = cbv_slots
        .iter()
        .map(|slot| slot.vec4_count)
        .max()
        .unwrap_or(0);

    ProgramPlan {
        stage,
        entry_point,
        workgroup_size,
        compute_seed_wg_sgprs,
        active_inst_count,
        vertex_input_mode,
        vertex_attrs,
        has_buffer_fetch,
        legacy_uv_attr_index,
        vsharp_fetch_insts,
        indirect_smem_cbv_keys,
        cbv_slots,
        cbv_vec4_count,
        tex_slots,
        samp_slots,
        storage_buf_slots,
        storage_tex_slots,
        uses_lds,
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
    f32_ty: Handle<Type>,
    u32_ty: Handle<Type>,
    i32_ty: Handle<Type>,
    vec2_f32_ty: Handle<Type>,
    vec2_i32_ty: Handle<Type>,
    vec4_f32_ty: Handle<Type>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConstExprType {
    F32,
    U32,
    I32,
    Vec2F32,
}

struct IrEmitter {
    module: Module,
    types: CommonTypes,
    vsharp_fetch_insts: BTreeSet<usize>,
    indirect_smem_cbv_keys: BTreeMap<usize, CbvKey>,
    cb_globals: BTreeMap<CbvKey, (Handle<GlobalVariable>, u32)>,
    tex_globals: BTreeMap<u8, Handle<GlobalVariable>>,
    samp_globals: BTreeMap<u8, Handle<GlobalVariable>>,
    storage_buf_globals: BTreeMap<u8, (Handle<GlobalVariable>, bool)>,
    storage_tex_globals: BTreeMap<u8, Handle<GlobalVariable>>,
    lds_global: Option<Handle<GlobalVariable>>,
    emitted_bindings: Vec<WgslBindingInfo>,
    stage: ShaderStage,
    workgroup_size: [u32; 3],
    compute_seed_wg_sgprs: Vec<(u8, u32)>,
    func: Function,
    helper_u32_to_f32: Option<Handle<Function>>,
    helper_unpack2x16float: Option<Handle<Function>>,
    helper_pack2x16float: Option<Handle<Function>>,
    helper_rt_f32: Option<Handle<Function>>,
    helper_rt_u32: Option<Handle<Function>>,
    helper_rt_i32: Option<Handle<Function>>,
    helper_rt_vec2_f32: Option<Handle<Function>>,
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
        let i32_ty = insert_type(&mut module, None, TypeInner::Scalar(Scalar::I32));
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
            f32_ty,
            u32_ty,
            i32_ty,
            vec2_f32_ty,
            vec2_i32_ty,
            vec4_f32_ty,
        };

        // Uniform constant buffers:
        //   cb0 -> @group(0) @binding(0)
        //   tex0 -> @group(0) @binding(1)
        //   samp0 -> @group(0) @binding(2)
        //   Additional cb1../tex1../samp1.. -> @group(0) @binding(3+)
        let mut cb_globals: BTreeMap<CbvKey, (Handle<GlobalVariable>, u32)> = BTreeMap::new();
        let mut tex_globals: BTreeMap<u8, Handle<GlobalVariable>> = BTreeMap::new();
        let mut samp_globals: BTreeMap<u8, Handle<GlobalVariable>> = BTreeMap::new();
        let mut storage_buf_globals: BTreeMap<u8, (Handle<GlobalVariable>, bool)> = BTreeMap::new();
        let mut storage_tex_globals: BTreeMap<u8, Handle<GlobalVariable>> = BTreeMap::new();
        let mut emitted_bindings: Vec<WgslBindingInfo> = Vec::new();
        let mut next_extra_binding: u32 = 3;

        if !plan.cbv_slots.is_empty() {
            let mut cbv_struct_by_count: BTreeMap<u32, Handle<Type>> = BTreeMap::new();
            for (idx, slot) in plan.cbv_slots.iter().enumerate() {
                let count = slot.vec4_count.max(1);
                let cbv_struct_ty = *cbv_struct_by_count.entry(count).or_insert_with(|| {
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
                    let struct_name = if idx == 0 {
                        "Rdna2Cbv".to_string()
                    } else {
                        format!("Rdna2Cbv{idx}")
                    };
                    insert_type(
                        &mut module,
                        Some(struct_name),
                        TypeInner::Struct {
                            members: vec![StructMember {
                                name: Some("data".to_string()),
                                ty: arr_ty,
                                binding: None,
                                offset: 0,
                            }],
                            span: count.saturating_mul(16),
                        },
                    )
                });
                let binding_num = if idx == 0 {
                    0
                } else {
                    let b = next_extra_binding;
                    next_extra_binding += 1;
                    b
                };
                let name = format!("cb{idx}");
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Uniform,
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: cbv_struct_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                cb_globals.insert(slot.key, (g, count));
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind: "uniform",
                    name,
                    min_binding_size_bytes: Some(count.saturating_mul(16)),
                    highest_vec4_offset: Some(slot.highest_vec4_offset),
                });
            }
        }

        if !plan.tex_slots.is_empty() {
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
            for (idx, &srsrc) in plan.tex_slots.iter().enumerate() {
                let binding_num = if idx == 0 {
                    1
                } else {
                    let b = next_extra_binding;
                    next_extra_binding += 1;
                    b
                };
                let name = format!("tex{idx}");
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Handle,
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: tex_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                tex_globals.insert(srsrc, g);
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind: "texture_2d",
                    name,
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                });
            }
            for (idx, &ssamp) in plan.samp_slots.iter().enumerate() {
                let binding_num = if idx == 0 {
                    2
                } else {
                    let b = next_extra_binding;
                    next_extra_binding += 1;
                    b
                };
                let name = format!("samp{idx}");
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Handle,
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: samp_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                samp_globals.insert(ssamp, g);
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind: "sampler",
                    name,
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                });
            }
        }

        if !plan.storage_buf_slots.is_empty() {
            let u32_runtime_arr_ty = insert_type(
                &mut module,
                None,
                TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            );
            let mut srv_idx = 0u32;
            let mut uav_idx = 0u32;
            for slot in &plan.storage_buf_slots {
                let binding_num = next_extra_binding;
                next_extra_binding += 1;
                let (name, access, kind) = if slot.writable {
                    let n = format!("uav_buf{uav_idx}");
                    uav_idx += 1;
                    (
                        n,
                        StorageAccess::LOAD | StorageAccess::STORE,
                        "storage",
                    )
                } else {
                    let n = format!("sbuf{srv_idx}");
                    srv_idx += 1;
                    (n, StorageAccess::LOAD, "read-only-storage")
                };
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Storage { access },
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: u32_runtime_arr_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                storage_buf_globals.insert(slot.srsrc, (g, slot.writable));
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind,
                    name,
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                });
            }
        }

        if !plan.storage_tex_slots.is_empty() {
            let storage_tex_ty = insert_type(
                &mut module,
                None,
                TypeInner::Image {
                    dim: ImageDimension::D2,
                    arrayed: false,
                    class: ImageClass::Storage {
                        format: StorageFormat::Rgba8Unorm,
                        access: StorageAccess::STORE,
                    },
                },
            );
            for (idx, &srsrc) in plan.storage_tex_slots.iter().enumerate() {
                let binding_num = next_extra_binding;
                next_extra_binding += 1;
                let name = format!("uav_tex{idx}");
                let g = module.global_variables.append(
                    GlobalVariable {
                        name: Some(name.clone()),
                        space: AddressSpace::Handle,
                        binding: Some(ResourceBinding {
                            group: 0,
                            binding: binding_num,
                        }),
                        ty: storage_tex_ty,
                        init: None,
                        memory_decorations: MemoryDecorations::empty(),
                    },
                    Span::UNDEFINED,
                );
                storage_tex_globals.insert(srsrc, g);
                emitted_bindings.push(WgslBindingInfo {
                    group: 0,
                    binding: binding_num,
                    kind: "storage-texture",
                    name,
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                });
            }
        }

        let lds_global = if plan.uses_lds && plan.stage == ShaderStage::Compute {
            let lds_arr_ty = insert_type(
                &mut module,
                None,
                TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(
                        NonZeroU32::new(LDS_WORKGROUP_DWORDS).unwrap_or(NonZeroU32::MIN),
                    ),
                    stride: 4,
                },
            );
            Some(module.global_variables.append(
                GlobalVariable {
                    name: Some("lds".to_string()),
                    space: AddressSpace::WorkGroup,
                    binding: None,
                    ty: lds_arr_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::UNDEFINED,
            ))
        } else {
            None
        };

        emitted_bindings.sort_by_key(|b| (b.group, b.binding));

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
                    name: Some("local_id".to_string()),
                    ty: vec3_u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationId)),
                });
                func.arguments.push(FunctionArgument {
                    name: Some("wg_id".to_string()),
                    ty: vec3_u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
                });
                func.arguments.push(FunctionArgument {
                    name: Some("global_id".to_string()),
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
            vsharp_fetch_insts: plan.vsharp_fetch_insts.clone(),
            indirect_smem_cbv_keys: plan.indirect_smem_cbv_keys.clone(),
            cb_globals,
            tex_globals,
            samp_globals,
            storage_buf_globals,
            storage_tex_globals,
            lds_global,
            emitted_bindings,
            stage: plan.stage,
            workgroup_size: plan.workgroup_size,
            compute_seed_wg_sgprs: plan.compute_seed_wg_sgprs.clone(),
            func,
            helper_u32_to_f32: None,
            helper_unpack2x16float: None,
            helper_pack2x16float: None,
            helper_rt_f32: None,
            helper_rt_u32: None,
            helper_rt_i32: None,
            helper_rt_vec2_f32: None,
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

    fn emit_expr_raw(&mut self, expr: Expression) -> Handle<Expression> {
        let h = self.func.expressions.append(expr, Span::UNDEFINED);
        self.func
            .body
            .push(Statement::Emit(Range::new_from_bounds(h, h)), Span::UNDEFINED);
        h
    }

    fn is_const_expr(&self, h: Handle<Expression>) -> bool {
        match self.func.expressions[h] {
            Expression::Literal(_) | Expression::ZeroValue(_) => true,
            Expression::Unary { expr, .. } | Expression::As { expr, .. } => {
                self.is_const_expr(expr)
            }
            Expression::Binary { left, right, .. } => {
                self.is_const_expr(left) && self.is_const_expr(right)
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                self.is_const_expr(condition)
                    && self.is_const_expr(accept)
                    && self.is_const_expr(reject)
            }
            Expression::Splat { value, .. } => self.is_const_expr(value),
            Expression::Swizzle { vector, .. } => self.is_const_expr(vector),
            Expression::AccessIndex { base, .. } => self.is_const_expr(base),
            Expression::Compose { ref components, .. } => {
                components.iter().all(|&c| self.is_const_expr(c))
            }
            Expression::Math {
                arg,
                arg1,
                arg2,
                arg3,
                ..
            } => {
                self.is_const_expr(arg)
                    && arg1.is_none_or(|a| self.is_const_expr(a))
                    && arg2.is_none_or(|a| self.is_const_expr(a))
                    && arg3.is_none_or(|a| self.is_const_expr(a))
            }
            _ => false,
        }
    }

    fn const_expr_type(&self, h: Handle<Expression>) -> Option<ConstExprType> {
        match self.func.expressions[h] {
            Expression::Literal(Literal::F32(_)) => Some(ConstExprType::F32),
            Expression::Literal(Literal::U32(_)) => Some(ConstExprType::U32),
            Expression::Literal(Literal::I32(_)) => Some(ConstExprType::I32),
            Expression::As { kind, .. } => match kind {
                ScalarKind::Float => Some(ConstExprType::F32),
                ScalarKind::Uint => Some(ConstExprType::U32),
                ScalarKind::Sint => Some(ConstExprType::I32),
                _ => None,
            },
            Expression::Compose { ty, .. } if ty == self.types.vec2_f32_ty => {
                Some(ConstExprType::Vec2F32)
            }
            Expression::Splat {
                size: VectorSize::Bi,
                value,
            } if self.const_expr_type(value) == Some(ConstExprType::F32) => {
                Some(ConstExprType::Vec2F32)
            }
            Expression::Swizzle {
                size: VectorSize::Bi,
                vector,
                ..
            } if self.const_expr_type(vector) == Some(ConstExprType::Vec2F32) => {
                Some(ConstExprType::Vec2F32)
            }
            Expression::AccessIndex { base, .. }
                if self.const_expr_type(base) == Some(ConstExprType::Vec2F32) =>
            {
                Some(ConstExprType::F32)
            }
            Expression::Unary { expr, .. } => self.const_expr_type(expr),
            Expression::Binary { op, left, .. } => match op {
                BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterEqual
                | BinaryOperator::LogicalAnd
                | BinaryOperator::LogicalOr => None,
                _ => self.const_expr_type(left),
            },
            Expression::Select { accept, .. } => self.const_expr_type(accept),
            Expression::Math { fun, arg, .. } => match fun {
                MathFunction::Unpack2x16float => Some(ConstExprType::Vec2F32),
                MathFunction::Pack2x16float => Some(ConstExprType::U32),
                MathFunction::Dot => Some(ConstExprType::F32),
                _ => self.const_expr_type(arg),
            },
            _ => None,
        }
    }

    fn call_helper_1(
        &mut self,
        func_handle: Handle<Function>,
        arg: Handle<Expression>,
    ) -> Handle<Expression> {
        let res = self
            .func
            .expressions
            .append(Expression::CallResult(func_handle), Span::UNDEFINED);
        self.func.body.push(
            Statement::Call {
                function: func_handle,
                arguments: vec![arg],
                result: Some(res),
            },
            Span::UNDEFINED,
        );
        res
    }

    fn ensure_helper_u32_to_f32(&mut self) -> Handle<Function> {
        if let Some(h) = self.helper_u32_to_f32 {
            return h;
        }
        let mut f = Function {
            name: Some("pg6_u32_bits_to_float".to_string()),
            arguments: vec![FunctionArgument {
                name: Some("x".to_string()),
                ty: self.types.u32_ty,
                binding: None,
            }],
            result: Some(FunctionResult {
                ty: self.types.f32_ty,
                binding: None,
            }),
            ..Function::default()
        };
        let arg = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        let cast = f.expressions.append(
            Expression::As {
                expr: arg,
                kind: ScalarKind::Float,
                convert: None,
            },
            Span::UNDEFINED,
        );
        f.body
            .push(Statement::Emit(Range::new_from_bounds(cast, cast)), Span::UNDEFINED);
        f.body
            .push(Statement::Return { value: Some(cast) }, Span::UNDEFINED);
        let h = self.module.functions.append(f, Span::UNDEFINED);
        self.helper_u32_to_f32 = Some(h);
        h
    }

    fn call_pg6_u32_to_f32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = self.ensure_helper_u32_to_f32();
        self.call_helper_1(f, arg)
    }

    fn ensure_helper_unpack2x16float(&mut self) -> Handle<Function> {
        if let Some(h) = self.helper_unpack2x16float {
            return h;
        }
        let mut f = Function {
            name: Some("pg6_unpack_half_pair".to_string()),
            arguments: vec![FunctionArgument {
                name: Some("x".to_string()),
                ty: self.types.u32_ty,
                binding: None,
            }],
            result: Some(FunctionResult {
                ty: self.types.vec2_f32_ty,
                binding: None,
            }),
            ..Function::default()
        };
        let arg = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        let unpacked = f.expressions.append(
            Expression::Math {
                fun: MathFunction::Unpack2x16float,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Emit(Range::new_from_bounds(unpacked, unpacked)),
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Return {
                value: Some(unpacked),
            },
            Span::UNDEFINED,
        );
        let h = self.module.functions.append(f, Span::UNDEFINED);
        self.helper_unpack2x16float = Some(h);
        h
    }

    fn call_pg6_unpack2x16float(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = self.ensure_helper_unpack2x16float();
        self.call_helper_1(f, arg)
    }

    fn ensure_helper_pack2x16float(&mut self) -> Handle<Function> {
        if let Some(h) = self.helper_pack2x16float {
            return h;
        }
        let mut f = Function {
            name: Some("pg6_pack_half_pair".to_string()),
            arguments: vec![FunctionArgument {
                name: Some("v".to_string()),
                ty: self.types.vec2_f32_ty,
                binding: None,
            }],
            result: Some(FunctionResult {
                ty: self.types.u32_ty,
                binding: None,
            }),
            ..Function::default()
        };
        let arg = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        let packed = f.expressions.append(
            Expression::Math {
                fun: MathFunction::Pack2x16float,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Emit(Range::new_from_bounds(packed, packed)),
            Span::UNDEFINED,
        );
        f.body.push(
            Statement::Return {
                value: Some(packed),
            },
            Span::UNDEFINED,
        );
        let h = self.module.functions.append(f, Span::UNDEFINED);
        self.helper_pack2x16float = Some(h);
        h
    }

    fn call_pg6_pack2x16float(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = self.ensure_helper_pack2x16float();
        self.call_helper_1(f, arg)
    }

    fn ensure_identity_helper(
        module: &mut Module,
        slot: &mut Option<Handle<Function>>,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<Function> {
        if let Some(h) = *slot {
            return h;
        }
        let mut f = Function {
            name: Some(name.to_string()),
            arguments: vec![FunctionArgument {
                name: Some("x".to_string()),
                ty,
                binding: None,
            }],
            result: Some(FunctionResult { ty, binding: None }),
            ..Function::default()
        };
        let arg = f
            .expressions
            .append(Expression::FunctionArgument(0), Span::UNDEFINED);
        f.body
            .push(Statement::Return { value: Some(arg) }, Span::UNDEFINED);
        let h = module.functions.append(f, Span::UNDEFINED);
        *slot = Some(h);
        h
    }

    fn call_pg6_rt_f32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = Self::ensure_identity_helper(
            &mut self.module,
            &mut self.helper_rt_f32,
            "pg6_runtime_float",
            self.types.f32_ty,
        );
        self.call_helper_1(f, arg)
    }

    fn call_pg6_rt_u32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = Self::ensure_identity_helper(
            &mut self.module,
            &mut self.helper_rt_u32,
            "pg6_runtime_uint",
            self.types.u32_ty,
        );
        self.call_helper_1(f, arg)
    }

    fn call_pg6_rt_i32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = Self::ensure_identity_helper(
            &mut self.module,
            &mut self.helper_rt_i32,
            "pg6_runtime_sint",
            self.types.i32_ty,
        );
        self.call_helper_1(f, arg)
    }

    fn call_pg6_rt_vec2_f32(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        let f = Self::ensure_identity_helper(
            &mut self.module,
            &mut self.helper_rt_vec2_f32,
            "pg6_runtime_vec2f",
            self.types.vec2_f32_ty,
        );
        self.call_helper_1(f, arg)
    }

    fn wrap_const_runtime(&mut self, arg: Handle<Expression>) -> Handle<Expression> {
        match self.const_expr_type(arg) {
            Some(ConstExprType::F32) => self.call_pg6_rt_f32(arg),
            Some(ConstExprType::U32) => self.call_pg6_rt_u32(arg),
            Some(ConstExprType::I32) => self.call_pg6_rt_i32(arg),
            Some(ConstExprType::Vec2F32) => self.call_pg6_rt_vec2_f32(arg),
            None => arg,
        }
    }

    fn emit_expr(&mut self, expr: Expression) -> Handle<Expression> {
        match expr {
            // Fold integer/boolean constant binary ops in Rust (wrapping semantics) so Tint
            // WGSL const-eval never rejects shift/add/sub/mul overflow on literal operands,
            // and guard non-finite float const-eval operations through runtime helpers.
            Expression::Binary { op, left, right } => {
                match (&self.func.expressions[left], &self.func.expressions[right]) {
                    (
                        Expression::Literal(Literal::U32(a)),
                        Expression::Literal(Literal::U32(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinaryOperator::Add => return self.lit_u32(a.wrapping_add(b)),
                            BinaryOperator::Subtract => return self.lit_u32(a.wrapping_sub(b)),
                            BinaryOperator::Multiply => return self.lit_u32(a.wrapping_mul(b)),
                            BinaryOperator::Divide => {
                                return self.lit_u32(if b == 0 { 0 } else { a / b });
                            }
                            BinaryOperator::Modulo => {
                                return self.lit_u32(if b == 0 { 0 } else { a % b });
                            }
                            BinaryOperator::And => return self.lit_u32(a & b),
                            BinaryOperator::InclusiveOr => return self.lit_u32(a | b),
                            BinaryOperator::ExclusiveOr => return self.lit_u32(a ^ b),
                            BinaryOperator::ShiftLeft => {
                                return self.lit_u32(a.wrapping_shl(b & 31));
                            }
                            BinaryOperator::ShiftRight => {
                                return self.lit_u32(a.wrapping_shr(b & 31));
                            }
                            BinaryOperator::Equal => return self.lit_bool(a == b),
                            BinaryOperator::NotEqual => return self.lit_bool(a != b),
                            BinaryOperator::Less => return self.lit_bool(a < b),
                            BinaryOperator::LessEqual => return self.lit_bool(a <= b),
                            BinaryOperator::Greater => return self.lit_bool(a > b),
                            BinaryOperator::GreaterEqual => return self.lit_bool(a >= b),
                            _ => {}
                        }
                    }
                    (
                        Expression::Literal(Literal::I32(a)),
                        Expression::Literal(Literal::U32(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinaryOperator::ShiftLeft => {
                                return self.lit_i32(a.wrapping_shl(b & 31));
                            }
                            BinaryOperator::ShiftRight => {
                                return self.lit_i32(a.wrapping_shr(b & 31));
                            }
                            _ => {}
                        }
                    }
                    (
                        Expression::Literal(Literal::I32(a)),
                        Expression::Literal(Literal::I32(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinaryOperator::Add => return self.lit_i32(a.wrapping_add(b)),
                            BinaryOperator::Subtract => return self.lit_i32(a.wrapping_sub(b)),
                            BinaryOperator::Multiply => return self.lit_i32(a.wrapping_mul(b)),
                            BinaryOperator::Divide => {
                                return self.lit_i32(if b == 0 { 0 } else { a.wrapping_div(b) });
                            }
                            BinaryOperator::Modulo => {
                                return self.lit_i32(if b == 0 { 0 } else { a.wrapping_rem(b) });
                            }
                            BinaryOperator::And => return self.lit_i32(a & b),
                            BinaryOperator::InclusiveOr => return self.lit_i32(a | b),
                            BinaryOperator::ExclusiveOr => return self.lit_i32(a ^ b),
                            BinaryOperator::Equal => return self.lit_bool(a == b),
                            BinaryOperator::NotEqual => return self.lit_bool(a != b),
                            BinaryOperator::Less => return self.lit_bool(a < b),
                            BinaryOperator::LessEqual => return self.lit_bool(a <= b),
                            BinaryOperator::Greater => return self.lit_bool(a > b),
                            BinaryOperator::GreaterEqual => return self.lit_bool(a >= b),
                            _ => {}
                        }
                    }
                    (
                        Expression::Literal(Literal::Bool(a)),
                        Expression::Literal(Literal::Bool(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        match op {
                            BinaryOperator::LogicalAnd => return self.lit_bool(a && b),
                            BinaryOperator::LogicalOr => return self.lit_bool(a || b),
                            BinaryOperator::Equal => return self.lit_bool(a == b),
                            BinaryOperator::NotEqual => return self.lit_bool(a != b),
                            _ => {}
                        }
                    }
                    (
                        Expression::Literal(Literal::F32(a)),
                        Expression::Literal(Literal::F32(b)),
                    ) => {
                        let (a, b) = (*a, *b);
                        let unsafe_const = match op {
                            BinaryOperator::Add => !(a + b).is_finite(),
                            BinaryOperator::Subtract => !(a - b).is_finite(),
                            BinaryOperator::Multiply => !(a * b).is_finite(),
                            BinaryOperator::Divide => b == 0.0 || !(a / b).is_finite(),
                            _ => false,
                        };
                        if unsafe_const {
                            let rt_left = self.call_pg6_rt_f32(left);
                            return self.emit_expr_raw(Expression::Binary {
                                op,
                                left: rt_left,
                                right,
                            });
                        }
                    }
                    _ => {
                        if matches!(
                            op,
                            BinaryOperator::Add
                                | BinaryOperator::Subtract
                                | BinaryOperator::Multiply
                                | BinaryOperator::Divide
                                | BinaryOperator::Modulo
                                | BinaryOperator::ShiftLeft
                                | BinaryOperator::ShiftRight
                        ) && self.is_const_expr(left)
                            && self.is_const_expr(right)
                        {
                            let rt_left = self.wrap_const_runtime(left);
                            return self.emit_expr_raw(Expression::Binary {
                                op,
                                left: rt_left,
                                right,
                            });
                        }
                    }
                }
                self.emit_expr_raw(Expression::Binary { op, left, right })
            }
            Expression::Unary { op, expr: inner } => {
                match (op, &self.func.expressions[inner]) {
                    (UnaryOperator::BitwiseNot, Expression::Literal(Literal::U32(a))) => {
                        return self.lit_u32(!*a);
                    }
                    (UnaryOperator::BitwiseNot, Expression::Literal(Literal::I32(a))) => {
                        return self.lit_i32(!*a);
                    }
                    (UnaryOperator::Negate, Expression::Literal(Literal::I32(a))) => {
                        return self.lit_i32((*a).wrapping_neg());
                    }
                    (UnaryOperator::LogicalNot, Expression::Literal(Literal::Bool(a))) => {
                        return self.lit_bool(!*a);
                    }
                    _ => {}
                }
                self.emit_expr_raw(Expression::Unary { op, expr: inner })
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => {
                if let Expression::Literal(Literal::Bool(c)) = self.func.expressions[condition] {
                    return if c { accept } else { reject };
                }
                self.emit_expr_raw(Expression::Select {
                    condition,
                    accept,
                    reject,
                })
            }
            Expression::As {
                expr: inner,
                kind,
                convert,
            } => {
                match convert {
                    None => {
                        match (&self.func.expressions[inner], kind) {
                            (Expression::Literal(Literal::U32(bits)), ScalarKind::Float) => {
                                let bits = *bits;
                                let f = f32::from_bits(bits);
                                if is_safe_wgsl_f32_literal(f) {
                                    return self.raw_expr(Expression::Literal(Literal::F32(f)));
                                }
                                return self.call_pg6_u32_to_f32(inner);
                            }
                            (Expression::Literal(Literal::I32(val)), ScalarKind::Float) => {
                                let bits = *val as u32;
                                let f = f32::from_bits(bits);
                                if is_safe_wgsl_f32_literal(f) {
                                    return self.raw_expr(Expression::Literal(Literal::F32(f)));
                                }
                                let u = self.lit_u32(bits);
                                return self.call_pg6_u32_to_f32(u);
                            }
                            (Expression::Literal(Literal::U32(val)), ScalarKind::Sint) => {
                                return self.lit_i32(*val as i32);
                            }
                            (Expression::Literal(Literal::I32(val)), ScalarKind::Uint) => {
                                return self.lit_u32(*val as u32);
                            }
                            (Expression::Literal(Literal::F32(f)), ScalarKind::Uint) => {
                                return self.lit_u32(f.to_bits());
                            }
                            (Expression::Literal(Literal::F32(f)), ScalarKind::Sint) => {
                                return self.lit_i32(f.to_bits() as i32);
                            }
                            _ if kind == ScalarKind::Float && self.is_const_expr(inner) => {
                                let u_arg =
                                    if self.const_expr_type(inner) == Some(ConstExprType::I32) {
                                        self.emit_expr_raw(Expression::As {
                                            expr: inner,
                                            kind: ScalarKind::Uint,
                                            convert: None,
                                        })
                                    } else {
                                        inner
                                    };
                                return self.call_pg6_u32_to_f32(u_arg);
                            }
                            _ => {}
                        }
                    }
                    Some(4) => match (&self.func.expressions[inner], kind) {
                        (Expression::Literal(Literal::F32(f)), ScalarKind::Uint)
                            if !(0.0..4_294_967_296.0).contains(f) =>
                        {
                            let rt = self.call_pg6_rt_f32(inner);
                            return self.emit_expr_raw(Expression::As {
                                expr: rt,
                                kind,
                                convert,
                            });
                        }
                        (Expression::Literal(Literal::F32(f)), ScalarKind::Sint)
                            if !(-2_147_483_648.0..2_147_483_648.0).contains(f) =>
                        {
                            let rt = self.call_pg6_rt_f32(inner);
                            return self.emit_expr_raw(Expression::As {
                                expr: rt,
                                kind,
                                convert,
                            });
                        }
                        _ if matches!(kind, ScalarKind::Uint | ScalarKind::Sint)
                            && self.is_const_expr(inner)
                            && self.const_expr_type(inner) == Some(ConstExprType::F32) =>
                        {
                            let rt = self.call_pg6_rt_f32(inner);
                            return self.emit_expr_raw(Expression::As {
                                expr: rt,
                                kind,
                                convert,
                            });
                        }
                        _ => {}
                    },
                    _ => {}
                }
                self.emit_expr_raw(Expression::As {
                    expr: inner,
                    kind,
                    convert,
                })
            }
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                match fun {
                    MathFunction::Unpack2x16float => {
                        if let Expression::Literal(Literal::U32(bits)) =
                            self.func.expressions[arg]
                        {
                            let lo_non_finite = (bits & 0x7C00) == 0x7C00;
                            let hi_non_finite = ((bits >> 16) & 0x7C00) == 0x7C00;
                            if lo_non_finite || hi_non_finite {
                                return self.call_pg6_unpack2x16float(arg);
                            }
                        } else if self.is_const_expr(arg) {
                            return self.call_pg6_unpack2x16float(arg);
                        }
                    }
                    MathFunction::Pack2x16float => {
                        if self.is_const_expr(arg) {
                            return self.call_pg6_pack2x16float(arg);
                        }
                    }
                    MathFunction::ReverseBits => {
                        if let Expression::Literal(Literal::U32(a)) = self.func.expressions[arg] {
                            return self.lit_u32(a.reverse_bits());
                        }
                    }
                    MathFunction::CountOneBits => {
                        if let Expression::Literal(Literal::U32(a)) = self.func.expressions[arg] {
                            return self.lit_u32(a.count_ones());
                        }
                    }
                    MathFunction::Abs => {
                        if let Expression::Literal(Literal::I32(a)) = self.func.expressions[arg] {
                            return self.lit_i32(a.wrapping_abs());
                        }
                    }
                    MathFunction::Min | MathFunction::Max => {
                        if let Some(a1) = arg1 {
                            match (&self.func.expressions[arg], &self.func.expressions[a1]) {
                                (
                                    Expression::Literal(Literal::U32(a)),
                                    Expression::Literal(Literal::U32(b)),
                                ) => {
                                    let res = if fun == MathFunction::Min {
                                        (*a).min(*b)
                                    } else {
                                        (*a).max(*b)
                                    };
                                    return self.lit_u32(res);
                                }
                                (
                                    Expression::Literal(Literal::I32(a)),
                                    Expression::Literal(Literal::I32(b)),
                                ) => {
                                    let res = if fun == MathFunction::Min {
                                        (*a).min(*b)
                                    } else {
                                        (*a).max(*b)
                                    };
                                    return self.lit_i32(res);
                                }
                                _ => {}
                            }
                        }
                    }
                    MathFunction::Clamp => {
                        if let (Some(lo_h), Some(hi_h)) = (arg1, arg2) {
                            match (
                                &self.func.expressions[arg],
                                &self.func.expressions[lo_h],
                                &self.func.expressions[hi_h],
                            ) {
                                (
                                    Expression::Literal(Literal::U32(a)),
                                    Expression::Literal(Literal::U32(lo)),
                                    Expression::Literal(Literal::U32(hi)),
                                ) => {
                                    return self.lit_u32((*a).max(*lo).min(*hi));
                                }
                                (
                                    Expression::Literal(Literal::I32(a)),
                                    Expression::Literal(Literal::I32(lo)),
                                    Expression::Literal(Literal::I32(hi)),
                                ) => {
                                    return self.lit_i32((*a).max(*lo).min(*hi));
                                }
                                (
                                    Expression::Literal(Literal::F32(_)),
                                    Expression::Literal(Literal::F32(lo)),
                                    Expression::Literal(Literal::F32(hi)),
                                ) => {
                                    if *lo > *hi || !lo.is_finite() || !hi.is_finite() {
                                        let rt = self.call_pg6_rt_f32(arg);
                                        return self.emit_expr_raw(Expression::Math {
                                            fun,
                                            arg: rt,
                                            arg1,
                                            arg2,
                                            arg3,
                                        });
                                    }
                                }
                                _ if self.is_const_expr(arg)
                                    && self.is_const_expr(lo_h)
                                    && self.is_const_expr(hi_h) =>
                                {
                                    let rt = self.wrap_const_runtime(arg);
                                    return self.emit_expr_raw(Expression::Math {
                                        fun,
                                        arg: rt,
                                        arg1,
                                        arg2,
                                        arg3,
                                    });
                                }
                                _ => {}
                            }
                        }
                    }
                    MathFunction::InverseSqrt | MathFunction::Log2 => {
                        let needs_rt = match self.func.expressions[arg] {
                            Expression::Literal(Literal::F32(x)) => x <= 0.0 || !x.is_finite(),
                            _ => self.is_const_expr(arg),
                        };
                        if needs_rt {
                            let rt = self.wrap_const_runtime(arg);
                            return self.emit_expr_raw(Expression::Math {
                                fun,
                                arg: rt,
                                arg1,
                                arg2,
                                arg3,
                            });
                        }
                    }
                    MathFunction::Sqrt => {
                        let needs_rt = match self.func.expressions[arg] {
                            Expression::Literal(Literal::F32(x)) => x < 0.0 || !x.is_finite(),
                            _ => self.is_const_expr(arg),
                        };
                        if needs_rt {
                            let rt = self.wrap_const_runtime(arg);
                            return self.emit_expr_raw(Expression::Math {
                                fun,
                                arg: rt,
                                arg1,
                                arg2,
                                arg3,
                            });
                        }
                    }
                    MathFunction::Exp2 => {
                        let needs_rt = match self.func.expressions[arg] {
                            Expression::Literal(Literal::F32(x)) => x >= 128.0 || !x.is_finite(),
                            _ => self.is_const_expr(arg),
                        };
                        if needs_rt {
                            let rt = self.wrap_const_runtime(arg);
                            return self.emit_expr_raw(Expression::Math {
                                fun,
                                arg: rt,
                                arg1,
                                arg2,
                                arg3,
                            });
                        }
                    }
                    MathFunction::Fma => {
                        if let (Some(a1), Some(a2)) = (arg1, arg2)
                            && self.is_const_expr(arg)
                            && self.is_const_expr(a1)
                            && self.is_const_expr(a2)
                        {
                            let needs_rt = match (
                                &self.func.expressions[arg],
                                &self.func.expressions[a1],
                                &self.func.expressions[a2],
                            ) {
                                (
                                    Expression::Literal(Literal::F32(a)),
                                    Expression::Literal(Literal::F32(b)),
                                    Expression::Literal(Literal::F32(c)),
                                ) => !a.mul_add(*b, *c).is_finite(),
                                _ => true,
                            };
                            if needs_rt {
                                let rt = self.wrap_const_runtime(arg);
                                return self.emit_expr_raw(Expression::Math {
                                    fun,
                                    arg: rt,
                                    arg1,
                                    arg2,
                                    arg3,
                                });
                            }
                        }
                    }
                    _ => {}
                }
                self.emit_expr_raw(Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2,
                    arg3,
                })
            }
            other => self.emit_expr_raw(other),
        }
    }

    fn lit_f32(&mut self, value: f32) -> Handle<Expression> {
        if is_safe_wgsl_f32_literal(value) {
            self.raw_expr(Expression::Literal(Literal::F32(value)))
        } else {
            let bits = self.lit_u32(value.to_bits());
            self.call_pg6_u32_to_f32(bits)
        }
    }

    fn lit_f32_from_bits(&mut self, bits: u32) -> Handle<Expression> {
        let f = f32::from_bits(bits);
        if is_safe_wgsl_f32_literal(f) {
            self.raw_expr(Expression::Literal(Literal::F32(f)))
        } else {
            let u = self.lit_u32(bits);
            self.call_pg6_u32_to_f32(u)
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
                // Seed GFX10.3 compute thread ID VGPRs: v0 = local_id.x, v1 = local_id.y, v2 = local_id.z.
                // (For 1D workgroups `(X, 1, 1)`, `local_id.x` is identical to the packed 1D thread ID.)
                let local_id = self.raw_expr(Expression::FunctionArgument(0));
                for c in 0..3u32 {
                    let reg = c as u8;
                    if self.vgpr_locals.contains_key(&reg) {
                        let comp = self.emit_expr(Expression::AccessIndex {
                            base: local_id,
                            index: c,
                        });
                        self.store_vgpr_u32(reg, comp);
                    }
                }
                // Seed compute workgroup ID SGPRs (`s0`/`s1`/`s2` or configured base) when not overwritten.
                if !self.compute_seed_wg_sgprs.is_empty() {
                    let wg_id = self.raw_expr(Expression::FunctionArgument(1));
                    let seed_list = self.compute_seed_wg_sgprs.clone();
                    for (reg, comp_idx) in seed_list {
                        if self.sgpr_locals.contains_key(&reg) {
                            let comp = self.emit_expr(Expression::AccessIndex {
                                base: wg_id,
                                index: comp_idx,
                            });
                            self.store_sgpr_u32(reg, comp);
                        }
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

    /// Read a packed 2x16-bit `f16` VOP3P operand as `vec2<f32>` via core WGSL `unpack2x16float`
    /// (zero `enable f16;` required), applying exact GFX10.3 `op_sel`, `op_sel_hi`, `neg_lo`,
    /// and `neg_hi` modifiers.
    fn read_src_pk_f16(
        &mut self,
        src: u16,
        literal: Option<u32>,
        op_sel_lo: bool,
        op_sel_hi: bool,
        neg_lo: bool,
        neg_hi: bool,
    ) -> Handle<Expression> {
        let raw = match src {
            128 => {
                let z = self.lit_f32(0.0);
                self.emit_expr(Expression::Splat {
                    size: VectorSize::Bi,
                    value: z,
                })
            }
            240..=248 => {
                // GFX10.3 inline float constants populate the low 16-bit f16 half (0.0 in hi).
                let lo = self.lit_f32(Self::inline_float_value(src).unwrap_or(0.0));
                let hi = self.lit_f32(0.0);
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![lo, hi],
                })
            }
            _ => {
                let u = self.read_src_u32(src, literal);
                self.emit_expr(Expression::Math {
                    fun: MathFunction::Unpack2x16float,
                    arg: u,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                })
            }
        };

        let selected = match (op_sel_lo, op_sel_hi) {
            (false, true) => raw,
            (false, false) => self.emit_expr(Expression::Swizzle {
                size: VectorSize::Bi,
                vector: raw,
                pattern: [
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                ],
            }),
            (true, true) => self.emit_expr(Expression::Swizzle {
                size: VectorSize::Bi,
                vector: raw,
                pattern: [
                    SwizzleComponent::Y,
                    SwizzleComponent::Y,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                ],
            }),
            (true, false) => self.emit_expr(Expression::Swizzle {
                size: VectorSize::Bi,
                vector: raw,
                pattern: [
                    SwizzleComponent::Y,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                    SwizzleComponent::X,
                ],
            }),
        };

        match (neg_lo, neg_hi) {
            (false, false) => selected,
            (true, true) => self.emit_expr(Expression::Unary {
                op: UnaryOperator::Negate,
                expr: selected,
            }),
            (true, false) => {
                let x = self.emit_expr(Expression::AccessIndex {
                    base: selected,
                    index: 0,
                });
                let nx = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: x,
                });
                let y = self.emit_expr(Expression::AccessIndex {
                    base: selected,
                    index: 1,
                });
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![nx, y],
                })
            }
            (false, true) => {
                let x = self.emit_expr(Expression::AccessIndex {
                    base: selected,
                    index: 0,
                });
                let y = self.emit_expr(Expression::AccessIndex {
                    base: selected,
                    index: 1,
                });
                let ny = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::Negate,
                    expr: y,
                });
                self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![x, ny],
                })
            }
        }
    }

    fn update_scc_nonzero(&mut self, val_u32: Handle<Expression>) {
        if self.scc_local.is_some() {
            let zero = self.lit_u32(0);
            let nonzero = self.emit_expr(Expression::Binary {
                op: BinaryOperator::NotEqual,
                left: val_u32,
                right: zero,
            });
            self.store_scc(nonzero);
        }
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

    fn translate_instruction(&mut self, inst_idx: usize, inst: &Rdna2Instruction) {
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
                sbase,
                soffset,
                offset,
                ..
            } => self.translate_smem(inst_idx, *op, *sdata, *sbase, *soffset, *offset),
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
                sdst,
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
                *op, *vdst, *sdst, *src0, *src1, *src2, *clamp, *omod, *abs, *neg, *literal,
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
                srsrc,
                ssamp,
                ..
            } => self.translate_mimg(*op, *dmask, *vaddr, *vdata, *srsrc, *ssamp),
            Rdna2Detail::Mubuf {
                op,
                offset,
                offen,
                idxen,
                vaddr,
                vdata,
                srsrc,
                soffset,
                ..
            } => self.translate_mubuf_or_mtbuf(
                false, *op, 0, *offset, *offen, *idxen, *vaddr, *vdata, *srsrc, *soffset,
            ),
            Rdna2Detail::Mtbuf {
                op,
                format,
                offset,
                offen,
                idxen,
                vaddr,
                vdata,
                srsrc,
                soffset,
                ..
            } => self.translate_mubuf_or_mtbuf(
                true, *op, *format, *offset, *offen, *idxen, *vaddr, *vdata, *srsrc, *soffset,
            ),
            Rdna2Detail::Ds {
                op,
                offset0,
                offset1,
                addr,
                data0,
                data1,
                vdst,
                ..
            } => self.translate_ds(*op, *offset0, *offset1, *addr, *data0, *data1, *vdst),
            Rdna2Detail::Exp {
                target,
                en,
                compr,
                vsrc,
                ..
            } => self.translate_exp(*target, *en, *compr, *vsrc),
            Rdna2Detail::Flat { .. } | Rdna2Detail::Unknown { .. } => false,
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
            // s_mov_b32 / s_wqm_b32 (whole-quad-mode exec mask setup is identity in WGSL)
            0x03 | 0x09 => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                self.store_sgpr_u32(sdst, v);
                if op == 0x09 {
                    self.update_scc_nonzero(v);
                }
                true
            }
            // s_mov_b64 / s_wqm_b64: 64-bit scalar register-pair move (evaluate both halves before storing)
            0x04 | 0x0A => {
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
                self.update_scc_nonzero(res);
                true
            }
            // s_brev_b32
            0x0B => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::ReverseBits,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            // s_bcnt1_i32_b32
            0x0F => {
                let v = self.read_src_u32(u16::from(ssrc0), literal);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::CountOneBits,
                    arg: v,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
                true
            }
            // s_sext_i32_i8 / s_sext_i32_i16
            0x19 | 0x1A => {
                let v = self.read_src_i32(u16::from(ssrc0), literal);
                let shift_bits = if op == 0x19 { 24u32 } else { 16u32 };
                let sh = self.lit_u32(shift_bits);
                let left = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: v,
                    right: sh,
                });
                let sext = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left,
                    right: sh,
                });
                let u = self.bitcast_i32_to_u32(sext);
                self.store_sgpr_u32(sdst, u);
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
                self.update_scc_nonzero(res);
                true
            }
            // s_andn2_b32 / s_orn2_b32: s0 & ~s1 / s0 | ~s1
            0x14 | 0x16 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let nr = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: r,
                });
                let bin = if op == 0x14 {
                    BinaryOperator::And
                } else {
                    BinaryOperator::InclusiveOr
                };
                let res = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: nr,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
                true
            }
            // s_nand_b32 / s_nor_b32 / s_xnor_b32: ~(s0 op s1)
            0x18 | 0x1A | 0x1C => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let bin = match op {
                    0x18 => BinaryOperator::And,
                    0x1A => BinaryOperator::InclusiveOr,
                    _ => BinaryOperator::ExclusiveOr,
                };
                let inner = self.emit_expr(Expression::Binary {
                    op: bin,
                    left: l,
                    right: r,
                });
                let res = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: inner,
                });
                self.store_sgpr_u32(sdst, res);
                self.update_scc_nonzero(res);
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
                self.update_scc_nonzero(res);
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
                self.update_scc_nonzero(u);
                true
            }
            // s_bfm_b32: ((1u << (s0 & 31)) - 1u) << (s1 & 31)
            0x24 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let mask31 = self.lit_u32(31);
                let width = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: l,
                    right: mask31,
                });
                let offset = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: r,
                    right: mask31,
                });
                let one = self.lit_u32(1);
                let sh_w = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: one,
                    right: width,
                });
                let base_mask = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: sh_w,
                    right: one,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: base_mask,
                    right: offset,
                });
                self.store_sgpr_u32(sdst, res);
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
            // s_pack_ll_b32_b16 / s_pack_lh_b32_b16 / s_pack_hh_b32_b16
            0x32..=0x34 => {
                let l = self.read_src_u32(s0_u(), literal);
                let r = self.read_src_u32(s1_u(), literal);
                let mask_lo = self.lit_u32(0x0000_FFFF);
                let mask_hi = self.lit_u32(0xFFFF_0000);
                let sh16 = self.lit_u32(16);
                let lo_half = if op == 0x34 {
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftRight,
                        left: l,
                        right: sh16,
                    })
                } else {
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: l,
                        right: mask_lo,
                    })
                };
                let hi_half = if op == 0x32 {
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: r,
                        right: sh16,
                    })
                } else {
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: r,
                        right: mask_hi,
                    })
                };
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: lo_half,
                    right: hi_half,
                });
                self.store_sgpr_u32(sdst, res);
                true
            }
            _ => false,
        }
    }

    fn translate_smem(
        &mut self,
        inst_idx: usize,
        op: u8,
        sdata: u8,
        sbase: u8,
        soffset: u8,
        byte_offset: i32,
    ) -> bool {
        if self.vsharp_fetch_insts.contains(&inst_idx) {
            return true;
        }
        let Some(count) = smem_load_dword_count(op) else {
            return false;
        };
        let cbv_key = self
            .indirect_smem_cbv_keys
            .get(&inst_idx)
            .copied()
            .unwrap_or(CbvKey {
                base_sbase: sbase,
                table_offset: None,
            });
        let Some(&(cb_global, cb_vec4_count)) = self
            .cb_globals
            .get(&cbv_key)
            .or_else(|| self.cb_globals.values().next())
        else {
            return false;
        };
        let total_dwords = cb_vec4_count.max(1).saturating_mul(4);

        let cb_ptr = self.raw_expr(Expression::GlobalVariable(cb_global));
        let data_ptr = self.emit_expr(Expression::AccessIndex {
            base: cb_ptr,
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
        let mut cached_static_vec: Option<(u32, Handle<Expression>)> = None;

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
                let vec_val = match cached_static_vec {
                    Some((cached_idx, handle)) if cached_idx == vec_idx => handle,
                    _ => {
                        let vec_ptr = self.emit_expr(Expression::AccessIndex {
                            base: data_ptr,
                            index: vec_idx,
                        });
                        let loaded = self.emit_expr(Expression::Load { pointer: vec_ptr });
                        cached_static_vec = Some((vec_idx, loaded));
                        loaded
                    }
                };
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
            // v_cvt_rpi_i32_f32 / v_cvt_flr_i32_f32
            0x0C | 0x0D => {
                let f = self.read_src_f32(src0, literal, abs0, neg0);
                let shifted = if op == 0x0C {
                    let half = self.lit_f32(0.5);
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::Add,
                        left: f,
                        right: half,
                    })
                } else {
                    f
                };
                let floored = self.emit_expr(Expression::Math {
                    fun: MathFunction::Floor,
                    arg: shifted,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                let i = self.emit_expr(Expression::As {
                    expr: floored,
                    kind: ScalarKind::Sint,
                    convert: Some(4),
                });
                self.store_vgpr_i32(vdst, i);
                true
            }
            // v_cvt_f32_ubyte0..3: extract byte k as f32
            0x11..=0x14 => {
                let u = self.read_src_u32(src0, literal);
                let shift_bits = u32::from(op - 0x11) * 8;
                let shifted = if shift_bits > 0 {
                    let sh = self.lit_u32(shift_bits);
                    self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftRight,
                        left: u,
                        right: sh,
                    })
                } else {
                    u
                };
                let mask = self.lit_u32(0xFF);
                let byte_u = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: shifted,
                    right: mask,
                });
                let f = self.emit_expr(Expression::As {
                    expr: byte_u,
                    kind: ScalarKind::Float,
                    convert: Some(4),
                });
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
            // v_bfrev_b32
            0x38 => {
                let u = self.read_src_u32(src0, literal);
                let res = self.emit_expr(Expression::Math {
                    fun: MathFunction::ReverseBits,
                    arg: u,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_rcp_f16 (emulated via f32 + pack2x16float, zero `enable f16;`)
            0x54 => {
                let one = self.lit_f32(1.0);
                let f = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: one,
                    right: f,
                });
                let out = self.apply_float_out_mods(res, omod, clamp);
                self.store_vgpr_f16_lo(vdst, out);
                true
            }
            // v_sqrt_f16 / v_rsq_f16 / v_log_f16 / v_exp_f16 / v_floor_f16 / v_ceil_f16 /
            // v_trunc_f16 / v_rndne_f16 / v_fract_f16
            0x55 | 0x56 | 0x57 | 0x58 | 0x5B | 0x5C | 0x5D | 0x5E | 0x5F => {
                let f = self.read_src_mixed_f16_or_f32(src0, literal, true, false, abs0, neg0);
                let fun = match op {
                    0x55 => MathFunction::Sqrt,
                    0x56 => MathFunction::InverseSqrt,
                    0x57 => MathFunction::Log2,
                    0x58 => MathFunction::Exp2,
                    0x5B => MathFunction::Floor,
                    0x5C => MathFunction::Ceil,
                    0x5D => MathFunction::Trunc,
                    0x5E => MathFunction::Round,
                    _ => MathFunction::Fract,
                };
                let res = self.emit_expr(Expression::Math {
                    fun,
                    arg: f,
                    arg1: None,
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
            // v_dot2c_f32_f16: vdst = dot(unpack(src0), unpack(src1)) + vdst
            0x02 => {
                let a = self.read_src_pk_f16(src0, literal, false, true, neg0, neg0);
                let b = self.read_src_pk_f16(src1, literal, false, true, neg1, neg1);
                let acc = self.load_vgpr_f32(vdst);
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
                    right: acc,
                });
                let out = self.apply_float_out_mods(sum, omod, clamp);
                self.store_vgpr_f32(vdst, out);
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
            // v_mul_i32_i24: sign-extend low 24 bits and multiply
            0x09 => {
                let l_raw = self.read_src_i32(src0, literal);
                let r_raw = self.read_src_i32(src1, literal);
                let sh8 = self.lit_u32(8);
                let l_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: l_raw,
                    right: sh8,
                });
                let l = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: l_sh,
                    right: sh8,
                });
                let r_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: r_raw,
                    right: sh8,
                });
                let r = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: r_sh,
                    right: sh8,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: l,
                    right: r,
                });
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_mul_u32_u24: mask low 24 bits and multiply
            0x0B => {
                let l_raw = self.read_src_u32(src0, literal);
                let r_raw = self.read_src_u32(src1, literal);
                let mask24 = self.lit_u32(0x00FF_FFFF);
                let l = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: l_raw,
                    right: mask24,
                });
                let r = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: r_raw,
                    right: mask24,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: l,
                    right: r,
                });
                self.store_vgpr_u32(vdst, res);
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
            // v_xnor_b32: ~(src0 ^ src1)
            0x1C => {
                let l = self.read_src_u32(src0, literal);
                let r = self.read_src_u32(src1, literal);
                let x = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ExclusiveOr,
                    left: l,
                    right: r,
                });
                let res = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: x,
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
                let a_raw = self.read_src_f32(src0, literal, abs0, neg0);
                let b_raw = self.read_src_f32(src1, literal, abs1, neg1);
                let a = self.apply_float_out_mods(a_raw, omod, clamp);
                let b = self.apply_float_out_mods(b_raw, omod, clamp);
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
            // v_pk_fmac_f16: packed 2xf16 fma(src0, src1, vdst) via unpack2x16float/pack2x16float
            0x3C => {
                let a = self.read_src_pk_f16(src0, literal, false, true, neg0, neg0);
                let b = self.read_src_pk_f16(src1, literal, false, true, neg1, neg1);
                let c = self.read_src_pk_f16(
                    256 + u16::from(vdst),
                    None,
                    false,
                    true,
                    false,
                    false,
                );
                let mut res = self.emit_expr(Expression::Math {
                    fun: MathFunction::Fma,
                    arg: a,
                    arg1: Some(b),
                    arg2: Some(c),
                    arg3: None,
                });
                if clamp {
                    let z = self.lit_f32(0.0);
                    let o = self.lit_f32(1.0);
                    let lo = self.emit_expr(Expression::Splat {
                        size: VectorSize::Bi,
                        value: z,
                    });
                    let hi = self.emit_expr(Expression::Splat {
                        size: VectorSize::Bi,
                        value: o,
                    });
                    res = self.emit_expr(Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: res,
                        arg1: Some(lo),
                        arg2: Some(hi),
                        arg3: None,
                    });
                }
                let packed = self.emit_expr(Expression::Math {
                    fun: MathFunction::Pack2x16float,
                    arg: res,
                    arg1: None,
                    arg2: None,
                    arg3: None,
                });
                self.store_vgpr_u32(vdst, packed);
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
        sdst: Option<u8>,
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
            // v_mad_i32_i24: sext24(src0) * sext24(src1) + src2
            0x142 => {
                let a_raw = self.read_src_i32(src0, literal);
                let b_raw = self.read_src_i32(src1, literal);
                let c = self.read_src_i32(src2, literal);
                let sh8 = self.lit_u32(8);
                let a_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: a_raw,
                    right: sh8,
                });
                let a = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: a_sh,
                    right: sh8,
                });
                let b_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: b_raw,
                    right: sh8,
                });
                let b = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: b_sh,
                    right: sh8,
                });
                let prod = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: prod,
                    right: c,
                });
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_mad_u32_u24: (src0 & 0xFFFFFF) * (src1 & 0xFFFFFF) + src2
            0x143 => {
                let a_raw = self.read_src_u32(src0, literal);
                let b_raw = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let mask24 = self.lit_u32(0x00FF_FFFF);
                let a = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: a_raw,
                    right: mask24,
                });
                let b = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: b_raw,
                    right: mask24,
                });
                let prod = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: a,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: prod,
                    right: c,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_bfe_u32: unsigned bitfield extract (offset = src1[4:0], width = src2[4:0])
            0x148 => {
                let s0 = self.read_src_u32(src0, literal);
                let s1 = self.read_src_u32(src1, literal);
                let s2 = self.read_src_u32(src2, literal);
                let mask31 = self.lit_u32(31);
                let offset = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: s1,
                    right: mask31,
                });
                let width = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: s2,
                    right: mask31,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: s0,
                    right: offset,
                });
                let one = self.lit_u32(1);
                let one_shl = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: one,
                    right: width,
                });
                let bit_mask = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: one_shl,
                    right: one,
                });
                let extracted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: shifted,
                    right: bit_mask,
                });
                let zero = self.lit_u32(0);
                let is_zero = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Equal,
                    left: width,
                    right: zero,
                });
                let res = self.emit_expr(Expression::Select {
                    condition: is_zero,
                    accept: zero,
                    reject: extracted,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_bfi_b32: (src0 & src1) | (~src0 & src2)
            0x14A => {
                let mask = self.read_src_u32(src0, literal);
                let a = self.read_src_u32(src1, literal);
                let b = self.read_src_u32(src2, literal);
                let nmask = self.emit_expr(Expression::Unary {
                    op: UnaryOperator::BitwiseNot,
                    expr: mask,
                });
                let lhs = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: mask,
                    right: a,
                });
                let rhs = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: nmask,
                    right: b,
                });
                let res = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: lhs,
                    right: rhs,
                });
                self.store_vgpr_u32(vdst, res);
                true
            }
            // v_alignbit_b32: funnel shift right of (src0:src1) by (src2 & 31)
            0x14E => {
                let hi = self.read_src_u32(src0, literal);
                let lo = self.read_src_u32(src1, literal);
                let sh_raw = self.read_src_u32(src2, literal);
                let mask31 = self.lit_u32(31);
                let sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: sh_raw,
                    right: mask31,
                });
                let zero = self.lit_u32(0);
                let is_zero = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Equal,
                    left: sh,
                    right: zero,
                });
                let c32 = self.lit_u32(32);
                let inv_sh_raw = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: c32,
                    right: sh,
                });
                let inv_sh = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: inv_sh_raw,
                    right: mask31,
                });
                let hi_part = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: hi,
                    right: inv_sh,
                });
                let lo_part = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: lo,
                    right: sh,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
                    left: hi_part,
                    right: lo_part,
                });
                let res = self.emit_expr(Expression::Select {
                    condition: is_zero,
                    accept: lo,
                    reject: shifted,
                });
                self.store_vgpr_u32(vdst, res);
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
            // v_min3_i32 / v_max3_i32 / v_med3_i32
            0x152 | 0x155 | 0x158 => {
                let a = self.read_src_i32(src0, literal);
                let b = self.read_src_i32(src1, literal);
                let c = self.read_src_i32(src2, literal);
                let res = if op == 0x158 {
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
                    self.emit_expr(Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: a,
                        arg1: Some(lo),
                        arg2: Some(hi),
                        arg3: None,
                    })
                } else {
                    let fun = if op == 0x152 {
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
                    self.emit_expr(Expression::Math {
                        fun,
                        arg: ab,
                        arg1: Some(c),
                        arg2: None,
                        arg3: None,
                    })
                };
                self.store_vgpr_i32(vdst, res);
                true
            }
            // v_min3_u32 / v_max3_u32 / v_med3_u32
            0x153 | 0x156 | 0x159 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let res = if op == 0x159 {
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
                    self.emit_expr(Expression::Math {
                        fun: MathFunction::Clamp,
                        arg: a,
                        arg1: Some(lo),
                        arg2: Some(hi),
                        arg3: None,
                    })
                } else {
                    let fun = if op == 0x153 {
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
                    self.emit_expr(Expression::Math {
                        fun,
                        arg: ab,
                        arg1: Some(c),
                        arg2: None,
                        arg3: None,
                    })
                };
                self.store_vgpr_u32(vdst, res);
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
            // v_add_co_u32 / v_sub_co_u32 / v_subrev_co_u32 (VOP3B with carry-out in sdst)
            0x30F | 0x310 | 0x319 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let (res, carry) = match op {
                    0x30F => {
                        let sum = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Add,
                            left: a,
                            right: b,
                        });
                        let c = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Less,
                            left: sum,
                            right: a,
                        });
                        (sum, c)
                    }
                    0x310 => {
                        let diff = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Subtract,
                            left: a,
                            right: b,
                        });
                        let c = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Less,
                            left: a,
                            right: b,
                        });
                        (diff, c)
                    }
                    _ => {
                        let diff = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Subtract,
                            left: b,
                            right: a,
                        });
                        let c = self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Less,
                            left: b,
                            right: a,
                        });
                        (diff, c)
                    }
                };
                self.store_vgpr_u32(vdst, res);
                if let Some(sd) = sdst {
                    let sd7 = sd & 0x7F;
                    if sd7 == 106 {
                        self.store_vcc(carry);
                    } else if sd7 <= 105 {
                        let t = self.lit_u32(1);
                        let f = self.lit_u32(0);
                        let mask_u = self.emit_expr(Expression::Select {
                            condition: carry,
                            accept: t,
                            reject: f,
                        });
                        self.store_sgpr_u32(sd7, mask_u);
                    }
                }
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
            // v_lshl_or_b32: (src0 << (src1 & 31)) | src2
            0x36E => {
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
                    op: BinaryOperator::InclusiveOr,
                    left: shifted,
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
            // v_or3_b32: src0 | src1 | src2
            0x370 => {
                let a = self.read_src_u32(src0, literal);
                let b = self.read_src_u32(src1, literal);
                let c = self.read_src_u32(src2, literal);
                let ab = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::InclusiveOr,
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
        let read_pk = |this: &mut Self, s: u16, idx: u8| {
            let bit = 1u8 << idx;
            this.read_src_pk_f16(
                s,
                literal,
                (op_sel & bit) != 0,
                (op_sel_hi & bit) != 0,
                (neg_lo & bit) != 0,
                (neg_hi & bit) != 0,
            )
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
                let a = read_pk(self, src0, 0);
                let b = read_pk(self, src1, 1);
                let c = read_pk(self, src2, 2);
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
                let a = read_pk(self, src0, 0);
                let b = read_pk(self, src1, 1);
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
                let a = read_pk(self, src0, 0);
                let b = read_pk(self, src1, 1);
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
                let a = read_pk(self, src0, 0);
                let b = read_pk(self, src1, 1);
                let c = self.read_src_f32(src2, literal, false, (neg_lo & 4) != 0);
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

    fn translate_mimg(
        &mut self,
        op: u8,
        dmask: u8,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        ssamp: u8,
    ) -> bool {
        let Some(layout) = mimg_addr_layout(op) else {
            return false;
        };

        if layout.is_store {
            if self.stage == ShaderStage::Vertex {
                return false;
            }
            let Some(&uav_tex_global) = self
                .storage_tex_globals
                .get(&srsrc)
                .or_else(|| self.storage_tex_globals.values().next())
            else {
                return false;
            };
            let image = self.raw_expr(Expression::GlobalVariable(uav_tex_global));
            let ux = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off));
            let uy = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 1));
            let ix = self.bitcast_f32_to_i32(ux);
            let iy = self.bitcast_f32_to_i32(uy);
            let coord = self.emit_expr(Expression::Compose {
                ty: self.types.vec2_i32_ty,
                components: vec![ix, iy],
            });

            let eff_mask = if dmask == 0 { 0xF } else { dmask };
            let mut src_offset = 0u8;
            let mut comps = Vec::with_capacity(4);
            for ch in 0..4u32 {
                if (eff_mask & (1 << ch)) != 0 {
                    comps.push(self.load_vgpr_f32(vdata.saturating_add(src_offset)));
                    src_offset = src_offset.saturating_add(1);
                } else {
                    comps.push(self.lit_f32(if ch == 3 { 1.0 } else { 0.0 }));
                }
            }
            let val = self.emit_expr(Expression::Compose {
                ty: self.types.vec4_f32_ty,
                components: comps,
            });
            self.func.body.push(
                Statement::ImageStore {
                    image,
                    coordinate: coord,
                    array_index: None,
                    value: val,
                },
                Span::UNDEFINED,
            );
            return true;
        }

        let Some(&tex_global) = self
            .tex_globals
            .get(&srsrc)
            .or_else(|| self.tex_globals.values().next())
        else {
            return false;
        };
        let image = self.raw_expr(Expression::GlobalVariable(tex_global));

        let sampled = if layout.is_load {
            // image_load / image_load_mip
            let ux = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off));
            let uy = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 1));
            let ix = self.bitcast_f32_to_i32(ux);
            let iy = self.bitcast_f32_to_i32(uy);
            let coord = self.emit_expr(Expression::Compose {
                ty: self.types.vec2_i32_ty,
                components: vec![ix, iy],
            });
            let level = if let Some(lod_off) = layout.lod_off {
                let ml = self.load_vgpr_f32(vaddr.saturating_add(lod_off));
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
        } else {
            // image_sample* (sampled 2D texture)
            let Some(&samp_global) = self
                .samp_globals
                .get(&ssamp)
                .or_else(|| self.samp_globals.values().next())
            else {
                return false;
            };
            let sampler = self.raw_expr(Expression::GlobalVariable(samp_global));
            let u = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off));
            let v = self.load_vgpr_f32(vaddr.saturating_add(layout.uv_off + 1));
            let coord = self.emit_expr(Expression::Compose {
                ty: self.types.vec2_f32_ty,
                components: vec![u, v],
            });

            // WGSL only permits implicit-derivative `textureSample` / `textureSampleBias`
            // in Fragment shaders; Vertex/Compute stages must use explicit `textureSampleLevel`
            // or `textureSampleGrad`.
            let level = if layout.is_zero_lod {
                SampleLevel::Exact(self.lit_f32(0.0))
            } else if let Some(lod_off) = layout.lod_off {
                let lod = self.load_vgpr_f32(vaddr.saturating_add(lod_off));
                SampleLevel::Exact(lod)
            } else if let Some(grad_off) = layout.grad_off {
                let dudx = self.load_vgpr_f32(vaddr.saturating_add(grad_off));
                let dvdx = self.load_vgpr_f32(vaddr.saturating_add(grad_off + 1));
                let dudy = self.load_vgpr_f32(vaddr.saturating_add(grad_off + 2));
                let dvdy = self.load_vgpr_f32(vaddr.saturating_add(grad_off + 3));
                let x = self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![dudx, dvdx],
                });
                let y = self.emit_expr(Expression::Compose {
                    ty: self.types.vec2_f32_ty,
                    components: vec![dudy, dvdy],
                });
                SampleLevel::Gradient { x, y }
            } else if let Some(bias_off) = layout.bias_off {
                if self.stage == ShaderStage::Fragment {
                    let bias = self.load_vgpr_f32(vaddr.saturating_add(bias_off));
                    SampleLevel::Bias(bias)
                } else {
                    SampleLevel::Exact(self.lit_f32(0.0))
                }
            } else if self.stage == ShaderStage::Fragment {
                SampleLevel::Auto
            } else {
                SampleLevel::Exact(self.lit_f32(0.0))
            };

            let raw_sample = self.emit_expr(Expression::ImageSample {
                image,
                sampler,
                gather: None,
                coordinate: coord,
                array_index: None,
                offset: None,
                level,
                depth_ref: None,
                clamp_to_edge: false,
            });

            if let Some(cmp_off) = layout.compare_off {
                // Depth/shadow comparison on baseline `texture_2d<f32>` + `sampler`:
                // returns 1.0 when `z_ref <= tex.x`, else 0.0, replicated across channels.
                let z_ref = self.load_vgpr_f32(vaddr.saturating_add(cmp_off));
                let depth_val = self.emit_expr(Expression::AccessIndex {
                    base: raw_sample,
                    index: 0,
                });
                let passed = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::LessEqual,
                    left: z_ref,
                    right: depth_val,
                });
                let one = self.lit_f32(1.0);
                let zero = self.lit_f32(0.0);
                let cmp_f = self.emit_expr(Expression::Select {
                    condition: passed,
                    accept: one,
                    reject: zero,
                });
                self.emit_expr(Expression::Splat {
                    size: VectorSize::Quad,
                    value: cmp_f,
                })
            } else {
                raw_sample
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

    // -- MUBUF / MTBUF vertex attribute fetch & storage buffer load/store helpers --

    #[allow(clippy::too_many_arguments)]
    fn translate_mubuf_or_mtbuf(
        &mut self,
        is_mtbuf: bool,
        op: u8,
        format: u8,
        byte_offset: u16,
        offen: bool,
        idxen: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
    ) -> bool {
        if self.stage == ShaderStage::Vertex {
            return self.translate_vertex_buf_fetch(is_mtbuf, op, format, byte_offset, vdata);
        }
        self.translate_storage_buf_op(
            is_mtbuf,
            op,
            byte_offset,
            offen,
            idxen,
            vaddr,
            vdata,
            srsrc,
            soffset,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_storage_buf_op(
        &mut self,
        is_mtbuf: bool,
        op: u8,
        byte_offset: u16,
        offen: bool,
        idxen: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
    ) -> bool {
        let Some(buf_op) = (if is_mtbuf {
            mtbuf_storage_op(op)
        } else {
            mubuf_storage_op(op)
        }) else {
            return false;
        };
        let Some(&(buf_global, writable)) = self
            .storage_buf_globals
            .get(&srsrc)
            .or_else(|| self.storage_buf_globals.values().next())
        else {
            return false;
        };
        if buf_op.is_store() && !writable {
            return false;
        }

        // Compute effective byte address: vaddr_contrib + soffset + imm_offset
        let mut byte_addr: Option<Handle<Expression>> = None;
        let stride = buf_op.elem_stride_bytes();
        if offen && idxen {
            let idx_f = self.load_vgpr_f32(vaddr);
            let idx_u = self.bitcast_f32_to_u32(idx_f);
            let stride_lit = self.lit_u32(stride);
            let idx_bytes = self.emit_expr(Expression::Binary {
                op: BinaryOperator::Multiply,
                left: idx_u,
                right: stride_lit,
            });
            let off_f = self.load_vgpr_f32(vaddr.saturating_add(1));
            let off_u = self.bitcast_f32_to_u32(off_f);
            let sum = self.emit_expr(Expression::Binary {
                op: BinaryOperator::Add,
                left: idx_bytes,
                right: off_u,
            });
            byte_addr = Some(sum);
        } else if offen {
            let off_f = self.load_vgpr_f32(vaddr);
            byte_addr = Some(self.bitcast_f32_to_u32(off_f));
        } else if idxen {
            let idx_f = self.load_vgpr_f32(vaddr);
            let idx_u = self.bitcast_f32_to_u32(idx_f);
            if stride == 1 {
                byte_addr = Some(idx_u);
            } else {
                let stride_lit = self.lit_u32(stride);
                byte_addr = Some(self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Multiply,
                    left: idx_u,
                    right: stride_lit,
                }));
            }
        }

        let soff_expr = if soffset <= 105 {
            Some(self.load_sgpr_u32(soffset))
        } else if (129..=208).contains(&soffset) {
            Some(self.read_src_u32(u16::from(soffset), None))
        } else {
            None
        };
        if let Some(se) = soff_expr {
            byte_addr = Some(match byte_addr {
                Some(cur) => self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: cur,
                    right: se,
                }),
                None => se,
            });
        }

        if byte_offset != 0 {
            let imm = self.lit_u32(u32::from(byte_offset));
            byte_addr = Some(match byte_addr {
                Some(cur) => self.emit_expr(Expression::Binary {
                    op: BinaryOperator::Add,
                    left: cur,
                    right: imm,
                }),
                None => imm,
            });
        }

        let eff_byte_addr = byte_addr.unwrap_or_else(|| self.lit_u32(0));
        let two = self.lit_u32(2);
        let base_dword = self.emit_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: eff_byte_addr,
            right: two,
        });

        let buf_ptr = self.raw_expr(Expression::GlobalVariable(buf_global));
        let buf_len = self.emit_expr(Expression::ArrayLength(buf_ptr));
        let one = self.lit_u32(1);
        let clamped_len = self.emit_expr(Expression::Math {
            fun: MathFunction::Max,
            arg: buf_len,
            arg1: Some(one),
            arg2: None,
            arg3: None,
        });
        let max_idx = self.emit_expr(Expression::Binary {
            op: BinaryOperator::Subtract,
            left: clamped_len,
            right: one,
        });

        match buf_op {
            StorageBufOpKind::LoadUbyte | StorageBufOpKind::LoadSbyte => {
                let safe_idx = self.emit_expr(Expression::Math {
                    fun: MathFunction::Min,
                    arg: base_dword,
                    arg1: Some(max_idx),
                    arg2: None,
                    arg3: None,
                });
                let elem_ptr = self.emit_expr(Expression::Access {
                    base: buf_ptr,
                    index: safe_idx,
                });
                let word = self.emit_expr(Expression::Load { pointer: elem_ptr });
                let mask3 = self.lit_u32(3);
                let byte_lane = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: eff_byte_addr,
                    right: mask3,
                });
                let three = self.lit_u32(3);
                let bit_shift = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: byte_lane,
                    right: three,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: word,
                    right: bit_shift,
                });
                if buf_op == StorageBufOpKind::LoadUbyte {
                    let mask_ff = self.lit_u32(0xFF);
                    let val = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: shifted,
                        right: mask_ff,
                    });
                    self.store_vgpr_u32(vdata, val);
                } else {
                    let s_raw = self.bitcast_u32_to_i32(shifted);
                    let sh24 = self.lit_u32(24);
                    let shl = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: s_raw,
                        right: sh24,
                    });
                    let sext = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftRight,
                        left: shl,
                        right: sh24,
                    });
                    self.store_vgpr_i32(vdata, sext);
                }
                true
            }
            StorageBufOpKind::LoadUshort | StorageBufOpKind::LoadSshort => {
                let safe_idx = self.emit_expr(Expression::Math {
                    fun: MathFunction::Min,
                    arg: base_dword,
                    arg1: Some(max_idx),
                    arg2: None,
                    arg3: None,
                });
                let elem_ptr = self.emit_expr(Expression::Access {
                    base: buf_ptr,
                    index: safe_idx,
                });
                let word = self.emit_expr(Expression::Load { pointer: elem_ptr });
                let mask2 = self.lit_u32(2);
                let half_lane = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::And,
                    left: eff_byte_addr,
                    right: mask2,
                });
                let three = self.lit_u32(3);
                let bit_shift = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftLeft,
                    left: half_lane,
                    right: three,
                });
                let shifted = self.emit_expr(Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: word,
                    right: bit_shift,
                });
                if buf_op == StorageBufOpKind::LoadUshort {
                    let mask_ffff = self.lit_u32(0xFFFF);
                    let val = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::And,
                        left: shifted,
                        right: mask_ffff,
                    });
                    self.store_vgpr_u32(vdata, val);
                } else {
                    let s_raw = self.bitcast_u32_to_i32(shifted);
                    let sh16 = self.lit_u32(16);
                    let shl = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftLeft,
                        left: s_raw,
                        right: sh16,
                    });
                    let sext = self.emit_expr(Expression::Binary {
                        op: BinaryOperator::ShiftRight,
                        left: shl,
                        right: sh16,
                    });
                    self.store_vgpr_i32(vdata, sext);
                }
                true
            }
            StorageBufOpKind::LoadDwords(comps) => {
                let mut loaded = Vec::with_capacity(usize::from(comps));
                for k in 0..comps {
                    let idx_k = if k == 0 {
                        base_dword
                    } else {
                        let k_lit = self.lit_u32(u32::from(k));
                        self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Add,
                            left: base_dword,
                            right: k_lit,
                        })
                    };
                    let safe_idx = self.emit_expr(Expression::Math {
                        fun: MathFunction::Min,
                        arg: idx_k,
                        arg1: Some(max_idx),
                        arg2: None,
                        arg3: None,
                    });
                    let elem_ptr = self.emit_expr(Expression::Access {
                        base: buf_ptr,
                        index: safe_idx,
                    });
                    loaded.push(self.emit_expr(Expression::Load { pointer: elem_ptr }));
                }
                for (k, word) in loaded.into_iter().enumerate() {
                    self.store_vgpr_u32(vdata.saturating_add(k as u8), word);
                }
                true
            }
            StorageBufOpKind::StoreDwords(comps) => {
                for k in 0..comps {
                    let idx_k = if k == 0 {
                        base_dword
                    } else {
                        let k_lit = self.lit_u32(u32::from(k));
                        self.emit_expr(Expression::Binary {
                            op: BinaryOperator::Add,
                            left: base_dword,
                            right: k_lit,
                        })
                    };
                    let safe_idx = self.emit_expr(Expression::Math {
                        fun: MathFunction::Min,
                        arg: idx_k,
                        arg1: Some(max_idx),
                        arg2: None,
                        arg3: None,
                    });
                    let elem_ptr = self.emit_expr(Expression::Access {
                        base: buf_ptr,
                        index: safe_idx,
                    });
                    let vf = self.load_vgpr_f32(vdata.saturating_add(k));
                    let vu = self.bitcast_f32_to_u32(vf);
                    self.func.body.push(
                        Statement::Store {
                            pointer: elem_ptr,
                            value: vu,
                        },
                        Span::UNDEFINED,
                    );
                }
                true
            }
        }
    }

    // -- DS (LDS Workgroup Shared Memory) --

    fn lds_masked_dword_index(
        &mut self,
        base_addr_u32: Handle<Expression>,
        byte_offset: u32,
    ) -> Handle<Expression> {
        let eff_addr = if byte_offset == 0 {
            base_addr_u32
        } else {
            let off_lit = self.lit_u32(byte_offset);
            self.emit_expr(Expression::Binary {
                op: BinaryOperator::Add,
                left: base_addr_u32,
                right: off_lit,
            })
        };
        let two = self.lit_u32(2);
        let dword_idx = self.emit_expr(Expression::Binary {
            op: BinaryOperator::ShiftRight,
            left: eff_addr,
            right: two,
        });
        let mask = self.lit_u32(LDS_WORKGROUP_INDEX_MASK);
        self.emit_expr(Expression::Binary {
            op: BinaryOperator::And,
            left: dword_idx,
            right: mask,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn translate_ds(
        &mut self,
        op: u8,
        offset0: u8,
        offset1: u8,
        addr: u8,
        data0: u8,
        data1: u8,
        vdst: u8,
    ) -> bool {
        if self.stage != ShaderStage::Compute {
            return false;
        }
        let Some(lds_global) = self.lds_global else {
            return false;
        };
        let Some(lds_op) = ds_lds_op(op) else {
            return false;
        };

        let lds_ptr = self.raw_expr(Expression::GlobalVariable(lds_global));
        let addr_f = self.load_vgpr_f32(addr);
        let addr_u = self.bitcast_f32_to_u32(addr_f);

        match lds_op {
            DsLdsOp::ReadContiguous(n) => {
                let imm16 = u32::from(offset0) | (u32::from(offset1) << 8);
                let mut loaded = Vec::with_capacity(usize::from(n));
                for k in 0..n {
                    let idx = self.lds_masked_dword_index(addr_u, imm16.saturating_add(u32::from(k) * 4));
                    let elem_ptr = self.emit_expr(Expression::Access {
                        base: lds_ptr,
                        index: idx,
                    });
                    loaded.push(self.emit_expr(Expression::Load { pointer: elem_ptr }));
                }
                for (k, word) in loaded.into_iter().enumerate() {
                    self.store_vgpr_u32(vdst.saturating_add(k as u8), word);
                }
                true
            }
            DsLdsOp::WriteContiguous(n) => {
                let imm16 = u32::from(offset0) | (u32::from(offset1) << 8);
                for k in 0..n {
                    let idx = self.lds_masked_dword_index(addr_u, imm16.saturating_add(u32::from(k) * 4));
                    let elem_ptr = self.emit_expr(Expression::Access {
                        base: lds_ptr,
                        index: idx,
                    });
                    let vf = self.load_vgpr_f32(data0.saturating_add(k));
                    let vu = self.bitcast_f32_to_u32(vf);
                    self.func.body.push(
                        Statement::Store {
                            pointer: elem_ptr,
                            value: vu,
                        },
                        Span::UNDEFINED,
                    );
                }
                true
            }
            DsLdsOp::Read2 { stride_bytes } => {
                let idx0 = self.lds_masked_dword_index(addr_u, u32::from(offset0) * stride_bytes);
                let ptr0 = self.emit_expr(Expression::Access {
                    base: lds_ptr,
                    index: idx0,
                });
                let w0 = self.emit_expr(Expression::Load { pointer: ptr0 });

                let idx1 = self.lds_masked_dword_index(addr_u, u32::from(offset1) * stride_bytes);
                let ptr1 = self.emit_expr(Expression::Access {
                    base: lds_ptr,
                    index: idx1,
                });
                let w1 = self.emit_expr(Expression::Load { pointer: ptr1 });

                self.store_vgpr_u32(vdst, w0);
                self.store_vgpr_u32(vdst.saturating_add(1), w1);
                true
            }
            DsLdsOp::Write2 { stride_bytes } => {
                let idx0 = self.lds_masked_dword_index(addr_u, u32::from(offset0) * stride_bytes);
                let ptr0 = self.emit_expr(Expression::Access {
                    base: lds_ptr,
                    index: idx0,
                });
                let vf0 = self.load_vgpr_f32(data0);
                let vu0 = self.bitcast_f32_to_u32(vf0);
                self.func.body.push(
                    Statement::Store {
                        pointer: ptr0,
                        value: vu0,
                    },
                    Span::UNDEFINED,
                );

                let idx1 = self.lds_masked_dword_index(addr_u, u32::from(offset1) * stride_bytes);
                let ptr1 = self.emit_expr(Expression::Access {
                    base: lds_ptr,
                    index: idx1,
                });
                let vf1 = self.load_vgpr_f32(data1);
                let vu1 = self.bitcast_f32_to_u32(vf1);
                self.func.body.push(
                    Statement::Store {
                        pointer: ptr1,
                        value: vu1,
                    },
                    Span::UNDEFINED,
                );
                true
            }
        }
    }

    fn translate_vertex_buf_fetch(
        &mut self,
        is_mtbuf: bool,
        op: u8,
        format: u8,
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
        let (attr_offset, shift) = vertex_fetch_slot(is_mtbuf, op, format, byte_offset);
        let Some(&(member_idx, attr_comps)) = self.vs_attr_by_offset.get(&attr_offset) else {
            return false;
        };
        let vs_in = self.raw_expr(Expression::FunctionArgument(arg_idx));
        let attr_val = self.emit_expr(Expression::AccessIndex {
            base: vs_in,
            index: member_idx,
        });
        for c in 0..comps {
            let src_c = c.saturating_add(shift);
            let val = if attr_comps == 1 {
                if src_c == 0 {
                    attr_val
                } else if c == 3 {
                    self.lit_f32(1.0)
                } else {
                    self.lit_f32(0.0)
                }
            } else if src_c < attr_comps {
                self.emit_expr(Expression::AccessIndex {
                    base: attr_val,
                    index: u32::from(src_c),
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

    fn finish(
        mut self,
        entry_name: String,
    ) -> (Module, usize, usize, Vec<String>, Vec<WgslBindingInfo>) {
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
            ShaderStage::Compute => (NagaStage::Compute, self.workgroup_size),
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
            self.emitted_bindings,
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
    for (inst_idx, inst) in instructions[..plan.active_inst_count].iter().enumerate() {
        emitter.translate_instruction(inst_idx, inst);
    }

    let entry_point = plan.entry_point.clone();
    let (module, translated_op_count, unsupported_op_count, unsupported_comments, bindings) =
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

    let binding_summary = Rdna2BindingSummary {
        has_cbv: !plan.cbv_slots.is_empty(),
        cbv_vec4_count: plan.cbv_vec4_count,
        has_texture_2d: !plan.tex_slots.is_empty(),
        has_sampler: !plan.samp_slots.is_empty(),
        has_storage_buffer: !plan.storage_buf_slots.is_empty(),
        has_storage_texture: !plan.storage_tex_slots.is_empty(),
        has_lds: plan.uses_lds,
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
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
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
        // s_load_dwordx4 at offset 0x10 reads dwords 4..=7 -> vec4 index 1 -> 2 vec4s (32 bytes).
        assert_eq!(out.binding_summary.cbv_vec4_count, 2);
        assert_eq!(out.bindings[0].min_binding_size_bytes, Some(32));
        assert_eq!(out.bindings[0].highest_vec4_offset, Some(1));
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.binding_summary.has_sampler);
        assert_eq!(out.unsupported_op_count, 0);
        assert!(out.wgsl.contains("struct Rdna2Cbv"));
        assert!(out.wgsl.contains("array<vec4<u32>, 2>"), "WGSL:\n{}", out.wgsl);
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
        // Dynamic soffset falls back cleanly to at least 16 vec4s (256 bytes).
        assert_eq!(out.binding_summary.cbv_vec4_count, DEFAULT_DYNAMIC_CBV_VEC4_COUNT);
        assert_eq!(out.bindings[0].min_binding_size_bytes, Some(256));
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
                min_binding_size_bytes: Some(16),
                highest_vec4_offset: Some(0),
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
            json.contains("{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 16, \"highest_vec4_offset\": 0}"),
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
        assert_eq!(out.bindings[0].min_binding_size_bytes, Some(16));
        assert_eq!(out.bindings[0].highest_vec4_offset, Some(0));
        assert_eq!(out.bindings[1].group, 0);
        assert_eq!(out.bindings[1].binding, 1);
        assert_eq!(out.bindings[1].kind, "texture_2d");
        assert_eq!(out.bindings[1].name, "tex0");
        assert_eq!(out.bindings[1].min_binding_size_bytes, None);
        assert_eq!(out.bindings[1].highest_vec4_offset, None);
        assert_eq!(out.bindings[2].group, 0);
        assert_eq!(out.bindings[2].binding, 2);
        assert_eq!(out.bindings[2].kind, "sampler");
        assert_eq!(out.bindings[2].name, "samp0");
        assert_eq!(out.bindings[2].min_binding_size_bytes, None);
        assert_eq!(out.bindings[2].highest_vec4_offset, None);

        // Every declared binding in `out.bindings` must match a real `@group(g) @binding(b)` in WGSL.
        for b in &out.bindings {
            let needle = format!("@group({}) @binding({})", b.group, b.binding);
            assert!(out.wgsl.contains(&needle), "missing {needle} in WGSL:\n{}", out.wgsl);
        }

        let json = out.to_json();
        assert!(
            json.contains("{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 16, \"highest_vec4_offset\": 0}"),
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

    #[test]
    fn translates_textured_splash_ui_ps_with_cbv_image_sample_lz_v_pk_fma_f16_and_pkrtz_compr() {
        // Synthetic textured splash/UI pixel shader exercising:
        //   s_wqm_b64 exec, exec
        //   v_interp_p1_f32 / v_interp_p2_f32 attr0.xy (UV)
        //   s_buffer_load_dwordx4 s[4:7], s[0:3], 0x0 (tint/bias CBV)
        //   s_waitcnt lgkmcnt(0)
        //   image_sample_lz v4, v2, s[8:15], s[12:15] dmask:0xf dim:1
        //   s_waitcnt vmcnt(0)
        //   v_cvt_pkrtz_f16_f32 v8, v4, v5
        //   v_cvt_pkrtz_f16_f32 v9, v6, v7
        //   v_pk_fma_f16 v8, v8, s4, s5 op_sel:[0,1,0] op_sel_hi:[1,0,1] neg_lo:[0,0,1] neg_hi:[1,0,0] clamp
        //   v_pk_fma_f16 v9, v9, s6, 1.0 op_sel:[0,0,0] op_sel_hi:[1,1,0] clamp
        //   exp mrt0 v8, v9, off, off done compr vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_sop1(0x0A, 126, 126)); // s_wqm_b64 exec, exec
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x0, false, false)); // s_buffer_load_dwordx4
        words.push(enc_sopp(0x0C, 0xC07F)); // s_waitcnt lgkmcnt(0)
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 4, 2, 3)); // image_sample_lz
        words.push(enc_sopp(0x0C, 0x3F70)); // s_waitcnt vmcnt(0)
        words.push(enc_vop2(0x2E, 8, 256 + 4, 5)); // v_cvt_pkrtz_f16_f32 v8, v4, v5
        words.push(enc_vop2(0x2E, 9, 256 + 6, 7)); // v_cvt_pkrtz_f16_f32 v9, v6, v7
        words.extend_from_slice(&enc_vop3p(
            0x0E,
            8,
            256 + 8,
            4, // s4
            5, // s5
            true,
            0b010, // op_sel: src1 uses hi for lo lane
            0b101, // op_sel_hi: src1 uses lo for hi lane (.yx swap)
            0b100, // neg_lo on src2
            0b001, // neg_hi on src0
        ));
        words.extend_from_slice(&enc_vop3p(
            0x0E,
            9,
            256 + 9,
            6,   // s6
            242, // inline 1.0 broadcast via op_sel_hi bit 2 = 0
            true,
            0b000,
            0b011,
            0b000,
            0b000,
        ));
        words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [8, 9, 0, 0])); // compr done vm
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("textured splash/UI PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.entry_point, "fs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.translated_op_count, 15);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 0,
                    kind: "uniform",
                    name: "cb0".to_string(),
                    min_binding_size_bytes: Some(16),
                    highest_vec4_offset: Some(0),
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 1,
                    kind: "texture_2d",
                    name: "tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
            ]
        );
        assert!(out.wgsl.contains("textureSampleLevel("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("fma("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("clamp("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"), "must not require shader-f16");
        assert!(!out.wgsl.contains("<f16>"), "must not emit f16 type");
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_vs_and_fs_with_v_pk_ops_v_med3_f32_image_sample_l_and_s_buffer_load_x8_x16() {
        // 1) Synthetic Vertex Shader exercising s_buffer_load_dwordx8 + x16 across two CBVs (cb0, cb1),
        //    v_pk_mul_f16, v_pk_add_f16, v_pk_min_f16, v_pk_max_f16, v_pk_fmac_f16, v_med3_f32,
        //    and image_sample_l.
        let mut vs_words = Vec::new();
        // s_buffer_load_dwordx8 s[4:11], s[0:3], 0x20 (cb0 words 8..15)
        vs_words.extend_from_slice(&enc_smem(0x0B, 4, 0, 0x7D, 0x20, false, false));
        // s_buffer_load_dwordx16 s[16:31], s[4:7], 0x40 (cb1 words 16..31)
        vs_words.extend_from_slice(&enc_smem(0x0C, 16, 2, 0x7D, 0x40, false, false));
        vs_words.push(enc_sopp(0x0C, 0xC07F));
        vs_words.push(enc_vop1(0x06, 1, 256 + 0)); // v_cvt_f32_u32 v1, v0
        vs_words.push(enc_vop1(0x01, 2, 240)); // v_mov_b32 v2, 0.5
        // v_med3_f32 v2, v1, s4, s5
        vs_words.extend_from_slice(&enc_vop3(0x157, 2, 256 + 1, 4, 5, true, 0, 0, 0));
        // v_pk_mul_f16 v3, v1, s6 op_sel:[0,0] op_sel_hi:[1,1]
        vs_words.extend_from_slice(&enc_vop3p(0x10, 3, 256 + 1, 6, 0, false, 0, 0b011, 0, 0));
        // v_pk_add_f16 v3, v3, s7 op_sel:[0,1] op_sel_hi:[1,0] neg_lo:[0,1] clamp
        vs_words.extend_from_slice(&enc_vop3p(
            0x0F,
            3,
            256 + 3,
            7,
            0,
            true,
            0b010,
            0b001,
            0b010,
            0,
        ));
        // v_pk_min_f16 v4, v3, s8
        vs_words.extend_from_slice(&enc_vop3p(0x11, 4, 256 + 3, 8, 0, false, 0, 0b011, 0, 0));
        // v_pk_max_f16 v4, v4, s16
        vs_words.extend_from_slice(&enc_vop3p(0x12, 4, 256 + 4, 16, 0, false, 0, 0b011, 0, 0));
        // v_pk_fmac_f16 v4, v3, s31 (VOP2 0x3C)
        vs_words.push(enc_vop2(0x3C, 4, 31, 3));
        // image_sample_l v5, v0, s[8:15], s[12:15] dmask:0xf dim:1
        vs_words.extend_from_slice(&enc_mimg(0x24, 0xF, 1, 0, 5, 2, 3));
        vs_words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [5, 6, 7, 8])); // pos0
        vs_words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [2, 4, 0, 0])); // param0 done
        vs_words.push(enc_sopp(0x01, 0));

        let vs_bytes = words_to_bytes(&vs_words);
        let vs_out = translate_rdna2_to_wgsl(&vs_bytes, &Rdna2TranslateOptions::default())
            .expect("VS with v_pk_*_f16, v_med3_f32, image_sample_l, and s_buffer_load_x8/x16 must succeed");
        assert!(vs_out.naga_validated);
        assert_eq!(vs_out.stage, ShaderStage::Vertex);
        assert_eq!(vs_out.unsupported_op_count, 0, "WGSL:\n{}", vs_out.wgsl);
        assert_eq!(vs_out.bindings.len(), 4); // cb0@0, tex0@1, samp0@2, cb1@3
        assert_eq!(vs_out.bindings[0].name, "cb0");
        assert_eq!(vs_out.bindings[0].min_binding_size_bytes, Some(64));
        assert_eq!(vs_out.bindings[0].highest_vec4_offset, Some(3));
        assert_eq!(vs_out.bindings[1].name, "tex0");
        assert_eq!(vs_out.bindings[2].name, "samp0");
        assert_eq!(vs_out.bindings[3].name, "cb1");
        assert_eq!(vs_out.bindings[3].binding, 3);
        assert_eq!(vs_out.bindings[3].min_binding_size_bytes, Some(128));
        assert_eq!(vs_out.bindings[3].highest_vec4_offset, Some(7));
        assert!(vs_out.wgsl.contains("textureSampleLevel("));
        assert!(
            vs_out.wgsl.contains("cb0_.data[2]") && vs_out.wgsl.contains("cb0_.data[3]"),
            "expected cb0_.data[2..4] in WGSL:\n{}",
            vs_out.wgsl
        );
        assert!(
            vs_out.wgsl.contains("cb1_.data[4]") && vs_out.wgsl.contains("cb1_.data[7]"),
            "expected cb1_.data[4..8] in WGSL:\n{}",
            vs_out.wgsl
        );
        assert!(!vs_out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&vs_out.wgsl);

        // 2) Synthetic Fragment Shader exercising s_buffer_load_dwordx8, image_sample_l,
        //    v_med3_f32, v_pk_mul_f16, v_pk_add_f16, and VOP3 v_cvt_pkrtz_f16_f32 clamp.
        let mut fs_words = Vec::new();
        fs_words.extend_from_slice(&enc_smem(0x0B, 4, 0, 0x7D, 0x0, false, false)); // s_buffer_load_dwordx8
        fs_words.push(enc_vintrp(0, 2, 0, 0, 0));
        fs_words.push(enc_vintrp(1, 2, 0, 0, 1));
        fs_words.push(enc_vintrp(0, 3, 0, 1, 0));
        fs_words.push(enc_vintrp(1, 3, 0, 1, 1));
        fs_words.push(enc_vop1(0x01, 4, 242)); // lod = 1.0 in v4
        fs_words.extend_from_slice(&enc_mimg(0x24, 0xF, 1, 2, 5, 2, 1)); // image_sample_l v[5:8], v[2:4]
        fs_words.extend_from_slice(&enc_vop3(0x157, 5, 256 + 5, 4, 5, false, 0, 0, 0)); // v_med3_f32
        // VOP3 v_cvt_pkrtz_f16_f32 (0x12E) with clamp
        fs_words.extend_from_slice(&enc_vop3(0x12E, 9, 256 + 5, 256 + 6, 0, true, 0, 0, 0));
        fs_words.extend_from_slice(&enc_vop3(0x12E, 10, 256 + 7, 256 + 8, 0, true, 0, 0, 0));
        fs_words.extend_from_slice(&enc_vop3p(0x10, 9, 256 + 9, 6, 0, false, 0, 0b011, 0, 0)); // v_pk_mul_f16
        fs_words.extend_from_slice(&enc_vop3p(0x0F, 9, 256 + 9, 7, 0, true, 0, 0b011, 0, 0)); // v_pk_add_f16
        fs_words.extend_from_slice(&enc_exp(0, 0xF, true, true, true, [9, 10, 0, 0]));
        fs_words.push(enc_sopp(0x01, 0));

        let fs_bytes = words_to_bytes(&fs_words);
        let fs_out = translate_rdna2_to_wgsl(&fs_bytes, &Rdna2TranslateOptions::default())
            .expect("FS with v_pk_mul/add_f16, v_med3_f32, image_sample_l, and s_buffer_load_x8 must succeed");
        assert!(fs_out.naga_validated);
        assert_eq!(fs_out.stage, ShaderStage::Fragment);
        assert_eq!(fs_out.unsupported_op_count, 0, "WGSL:\n{}", fs_out.wgsl);
        assert!(fs_out.wgsl.contains("textureSampleLevel("));
        assert!(fs_out.wgsl.contains("pack2x16float("));
        assert!(fs_out.wgsl.contains("unpack2x16float("));
        assert!(!fs_out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&fs_out.wgsl);
    }

    #[test]
    fn translates_all_mimg_sampling_and_load_variants_with_dmask_and_multi_texture_bindings() {
        // Fragment shader exercising all 7 required MIMG variants + partial dmasks + 2 textures:
        //   image_sample      (0x20) dmask:0xf on tex0 (srsrc=2)
        //   image_sample_lz   (0x27) dmask:0x3 (RG compacted to v10..v11) on tex0
        //   image_sample_l    (0x24) dmask:0x9 (RA compacted to v12..v13) on tex1 (srsrc=4)
        //   image_sample_b    (0x25) dmask:0xf (bias in v1, uv in v2..v3) on tex0
        //   image_sample_c    (0x28) dmask:0x1 (z_ref in v1, uv in v2..v3 -> shadow factor in v18) on tex1
        //   image_load        (0x00) dmask:0x3 on tex0
        //   image_load_mip    (0x01) dmask:0x5 (RB compacted to v21..v22) on tex1
        let mut words = Vec::new();
        words.push(enc_vop1(0x01, 1, 240)); // v1 = 0.5 (bias / z_ref)
        words.push(enc_vintrp(0, 2, 0, 0, 0)); // v2 = uv.x
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0)); // v3 = uv.y
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.push(enc_vop1(0x01, 4, 244)); // v4 = 2.0 (lod / mip)
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 6, 2, 1)); // image_sample -> v6..v9
        words.extend_from_slice(&enc_mimg(0x27, 0x3, 1, 2, 10, 2, 1)); // image_sample_lz (RG) -> v10..v11
        words.extend_from_slice(&enc_mimg(0x24, 0x9, 1, 2, 12, 4, 1)); // image_sample_l (RA) on tex1 -> v12..v13
        words.extend_from_slice(&enc_mimg(0x25, 0xF, 1, 1, 14, 2, 1)); // image_sample_b -> v14..v17
        words.extend_from_slice(&enc_mimg(0x28, 0x1, 1, 1, 18, 4, 1)); // image_sample_c on tex1 -> v18
        words.extend_from_slice(&enc_mimg(0x00, 0x3, 1, 2, 19, 2, 0)); // image_load -> v19..v20
        words.extend_from_slice(&enc_mimg(0x01, 0x5, 1, 2, 21, 4, 0)); // image_load_mip on tex1 -> v21..v22
        // Combine into mrt0
        words.push(enc_vop2(0x03, 6, 256 + 10, 6));
        words.push(enc_vop2(0x03, 7, 256 + 12, 14));
        words.push(enc_vop2(0x08, 8, 256 + 18, 19));
        words.push(enc_vop2(0x03, 9, 256 + 21, 9));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [6, 7, 8, 9]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("all MIMG sampling & load variants must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 1,
                    kind: "texture_2d",
                    name: "tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 3,
                    kind: "texture_2d",
                    name: "tex1".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
            ]
        );
        assert!(out.wgsl.contains("textureSample("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("textureSampleLevel("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("textureSampleBias("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("textureLoad("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("tex0") && out.wgsl.contains("tex1"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn translates_common_gfx10_3_scalar_and_vector_alu_ops_with_zero_unsupported() {
        // Exercises all common UI/splash scalar & vector ALU ops:
        //   s_lshl_b32, s_lshr_b32, s_and_b32, s_or_b32, s_xor_b32, s_pack_ll/lh/hh_b32_b16,
        //   v_med3_f32, v_floor_f32, v_fract_f32, v_rcp_f32, v_rsq_f32, v_sqrt_f32,
        //   v_exp_f32, v_log_f32, v_cvt_f32_ubyte0..3, v_bfi_b32, v_lshl_or_b32, v_or3_b32,
        //   v_med3_i32, v_med3_u32, v_rcp_f16, v_sqrt_f16
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 2, 128 + 15)); // s_mov_b32 s2, 15
        words.push(enc_sop2(0x1E, 3, 2, 128 + 2)); // s_lshl_b32 s3, s2, 2
        words.push(enc_sop2(0x20, 4, 3, 128 + 1)); // s_lshr_b32 s4, s3, 1
        words.push(enc_sop2(0x0E, 5, 4, 128 + 7)); // s_and_b32 s5, s4, 7
        words.push(enc_sop2(0x10, 6, 5, 128 + 8)); // s_or_b32 s6, s5, 8
        words.push(enc_sop2(0x12, 7, 6, 128 + 3)); // s_xor_b32 s7, s6, 3
        words.push(enc_sop2(0x32, 8, 5, 6)); // s_pack_ll_b32_b16 s8, s5, s6
        words.push(enc_sop2(0x33, 9, 8, 7)); // s_pack_lh_b32_b16 s9, s8, s7
        words.push(enc_sop2(0x34, 10, 9, 8)); // s_pack_hh_b32_b16 s10, s9, s8
        words.push(enc_vop1(0x06, 0, 7)); // v_cvt_f32_u32 v0, s7
        words.push(enc_vop1(0x24, 1, 256 + 0)); // v_floor_f32 v1, v0
        words.push(enc_vop1(0x20, 2, 256 + 0)); // v_fract_f32 v2, v0
        words.push(enc_vop1(0x2A, 3, 256 + 0)); // v_rcp_f32 v3, v0
        words.push(enc_vop1(0x2E, 4, 256 + 3)); // v_rsq_f32 v4, v3
        words.push(enc_vop1(0x33, 5, 256 + 4)); // v_sqrt_f32 v5, v4
        words.push(enc_vop1(0x25, 6, 256 + 5)); // v_exp_f32 v6, v5
        words.push(enc_vop1(0x27, 7, 256 + 6)); // v_log_f32 v7, v6
        words.extend_from_slice(&enc_vop3(0x157, 8, 256 + 7, 256 + 1, 256 + 2, true, 0, 0, 0)); // v_med3_f32
        words.push(enc_vop1(0x11, 9, 10)); // v_cvt_f32_ubyte0 v9, s10
        words.push(enc_vop1(0x12, 10, 10)); // v_cvt_f32_ubyte1 v10, s10
        words.push(enc_vop1(0x13, 11, 10)); // v_cvt_f32_ubyte2 v11, s10
        words.push(enc_vop1(0x14, 12, 10)); // v_cvt_f32_ubyte3 v12, s10
        words.extend_from_slice(&enc_vop3(0x14A, 13, 10, 256 + 9, 256 + 10, false, 0, 0, 0)); // v_bfi_b32
        words.extend_from_slice(&enc_vop3(0x36E, 14, 256 + 13, 128 + 2, 256 + 11, false, 0, 0, 0)); // v_lshl_or_b32
        words.extend_from_slice(&enc_vop3(0x370, 15, 256 + 14, 256 + 12, 7, false, 0, 0, 0)); // v_or3_b32
        words.extend_from_slice(&enc_vop3(0x158, 16, 256 + 13, 256 + 14, 256 + 15, false, 0, 0, 0)); // v_med3_i32
        words.extend_from_slice(&enc_vop3(0x159, 17, 256 + 13, 256 + 14, 256 + 15, false, 0, 0, 0)); // v_med3_u32
        words.push(enc_vop1(0x54, 18, 256 + 8)); // v_rcp_f16
        words.push(enc_vop1(0x55, 18, 256 + 18)); // v_sqrt_f16
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [8, 9, 17, 18]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("common GFX10.3 ALU ops must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("floor("));
        assert!(out.wgsl.contains("fract("));
        assert!(out.wgsl.contains("inverseSqrt("));
        assert!(out.wgsl.contains("sqrt("));
        assert!(out.wgsl.contains("exp2("));
        assert!(out.wgsl.contains("log2("));
        assert!(out.wgsl.contains("clamp("));
        assert!(!out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn tight_cbv_sizing_48_byte_frame1_cbv_avoids_256_byte_overdeclaration() {
        // Synthetic shader reading only cb[0]..cb[2] (byte offsets 0..44):
        //   s_buffer_load_dwordx4 s[4:7],  s[0:3], 0x00  (cb[0]: bytes 0..15)
        //   s_buffer_load_dwordx4 s[8:11], s[0:3], 0x10  (cb[1]: bytes 16..31)
        //   s_buffer_load_dword   s12,     s[0:3], 0x2C  (cb[2].w: bytes 44..47)
        //   v_mov_b32 v0, s4
        //   v_add_f32 v1, s8, s12
        //   exp mrt0 v0, v1, v0, v1 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x00, false, false));
        words.extend_from_slice(&enc_smem(0x0A, 8, 0, 0x7D, 0x10, false, false));
        words.extend_from_slice(&enc_smem(0x08, 12, 0, 0x7D, 0x2C, false, false));
        words.push(enc_vop1(0x01, 0, 4));
        words.push(enc_vop2(0x03, 1, 8, 0));
        words.push(enc_vop2(0x03, 1, 12, 1));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("small 48-byte CBV shader should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.cbv_vec4_count, 3);
        assert_eq!(
            out.bindings,
            vec![WgslBindingInfo {
                group: 0,
                binding: 0,
                kind: "uniform",
                name: "cb0".to_string(),
                min_binding_size_bytes: Some(48),
                highest_vec4_offset: Some(2),
            }]
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 3>"),
            "expected tightly sized 3-vec4 (48 B) CBV array in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("array<vec4<u32>, 16>"),
            "must NOT over-declare 256-byte CBV for 48-byte read:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("cb0_.data[2]"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 48, \"highest_vec4_offset\": 2}"
            ),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn tight_cbv_sizing_large_cbv_reads_vec4_20_without_clamping() {
        // Synthetic shader reading cb[20] (byte offset 20 * 16 = 320 = 0x140):
        //   s_buffer_load_dwordx4 s[4:7], s[0:3], 0x140
        //   v_mov_b32 v0, s4
        //   v_mov_b32 v1, s7
        //   exp mrt0 v0, v1, v0, v1 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x140, false, false));
        words.push(enc_vop1(0x01, 0, 4));
        words.push(enc_vop1(0x01, 1, 7));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("large 336-byte CBV shader should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.cbv_vec4_count, 21);
        assert_eq!(
            out.bindings,
            vec![WgslBindingInfo {
                group: 0,
                binding: 0,
                kind: "uniform",
                name: "cb0".to_string(),
                min_binding_size_bytes: Some(336),
                highest_vec4_offset: Some(20),
            }]
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 21>"),
            "expected 21-vec4 (336 B) CBV array in WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("cb0_.data[20]"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(
            json.contains(
                "{\"group\": 0, \"binding\": 0, \"kind\": \"uniform\", \"name\": \"cb0\", \"min_binding_size_bytes\": 336, \"highest_vec4_offset\": 20}"
            ),
            "JSON:\n{json}"
        );
    }

    #[test]
    fn tight_cbv_sizing_multiple_cbvs_sized_independently() {
        // Synthetic shader with two CBVs:
        //   cb0 (sbase=0): reads vec4 1 (offset 0x10) -> N=2 (32 bytes), highest_vec4_offset=1
        //   cb1 (sbase=2): reads vec4 5 (offset 0x50) -> N=6 (96 bytes), highest_vec4_offset=5
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x0A, 4, 0, 0x7D, 0x10, false, false));
        words.extend_from_slice(&enc_smem(0x0A, 8, 2, 0x7D, 0x50, false, false));
        words.push(enc_vop1(0x01, 0, 4));
        words.push(enc_vop1(0x01, 1, 8));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("multi-CBV shader should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(out.binding_summary.cbv_vec4_count, 6);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 0,
                    kind: "uniform",
                    name: "cb0".to_string(),
                    min_binding_size_bytes: Some(32),
                    highest_vec4_offset: Some(1),
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 3,
                    kind: "uniform",
                    name: "cb1".to_string(),
                    min_binding_size_bytes: Some(96),
                    highest_vec4_offset: Some(5),
                },
            ]
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 2>"),
            "expected cb0 sized to 2 vec4s (32 B):\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 6>"),
            "expected cb1 sized to 6 vec4s (96 B):\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("cb0_.data[1]"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("cb1_.data[5]"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
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
    fn tight_cbv_sizing_dynamic_soffset_falls_back_to_max_static_or_16() {
        // Case A: Dynamic soffset with small static offset (0x04) -> falls back to 16 vec4s (256 B).
        let mut words_small = Vec::new();
        words_small.push(enc_sop1(0x03, 2, 128 + 16)); // s_mov_b32 s2, 16
        words_small.extend_from_slice(&enc_smem(0x09, 4, 0, 2, 0x04, false, false)); // dynamic soffset=s2
        words_small.push(enc_vop1(0x01, 0, 4));
        words_small.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 0, 0, 0]));
        words_small.push(enc_sopp(0x01, 0));

        let out_small = translate_rdna2_to_wgsl(
            &words_to_bytes(&words_small),
            &Rdna2TranslateOptions::default(),
        )
        .expect("dynamic CBV shader should translate cleanly");
        assert!(out_small.naga_validated);
        assert_eq!(out_small.bindings[0].min_binding_size_bytes, Some(256));
        assert_eq!(out_small.bindings[0].highest_vec4_offset, Some(0));
        assert!(
            out_small.wgsl.contains("array<vec4<u32>, 16>"),
            "WGSL:\n{}",
            out_small.wgsl
        );
        assert_naga_wgsl_valid(&out_small.wgsl);

        // Case B: Dynamic soffset + static read at cb[20] -> max(20 + 1, 16) = 21 vec4s (336 B).
        let mut words_large = Vec::new();
        words_large.push(enc_sop1(0x03, 2, 128 + 16));
        words_large.extend_from_slice(&enc_smem(0x09, 4, 0, 2, 0x04, false, false));
        words_large.extend_from_slice(&enc_smem(0x0A, 8, 0, 0x7D, 0x140, false, false));
        words_large.push(enc_vop1(0x01, 0, 4));
        words_large.push(enc_vop1(0x01, 1, 8));
        words_large.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 0, 1]));
        words_large.push(enc_sopp(0x01, 0));

        let out_large = translate_rdna2_to_wgsl(
            &words_to_bytes(&words_large),
            &Rdna2TranslateOptions::default(),
        )
        .expect("dynamic + large static CBV shader should translate cleanly");
        assert!(out_large.naga_validated);
        assert_eq!(out_large.bindings[0].min_binding_size_bytes, Some(336));
        assert_eq!(out_large.bindings[0].highest_vec4_offset, Some(20));
        assert!(
            out_large.wgsl.contains("array<vec4<u32>, 21>"),
            "WGSL:\n{}",
            out_large.wgsl
        );
        assert_naga_wgsl_valid(&out_large.wgsl);
    }

    #[allow(clippy::too_many_arguments)]
    fn enc_mubuf_full(
        op: u8,
        offset: u16,
        offen: bool,
        idxen: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
    ) -> [u32; 2] {
        let w0 = (0b111000u32 << 26)
            | (u32::from(op & 0x7F) << 18)
            | (u32::from(idxen) << 13)
            | (u32::from(offen) << 12)
            | u32::from(offset & 0xFFF);
        let w1 = (u32::from(soffset) << 24)
            | (u32::from(srsrc & 0x1F) << 16)
            | (u32::from(vdata) << 8)
            | u32::from(vaddr);
        [w0, w1]
    }

    fn enc_ds(op: u8, offset0: u8, offset1: u8, addr: u8, data0: u8, data1: u8, vdst: u8) -> [u32; 2] {
        let w0 = (0b110110u32 << 26)
            | (u32::from(op) << 18)
            | (u32::from(offset1) << 8)
            | u32::from(offset0);
        let w1 = (u32::from(vdst) << 24)
            | (u32::from(data1) << 16)
            | (u32::from(data0) << 8)
            | u32::from(addr);
        [w0, w1]
    }

    #[test]
    fn translates_compute_buffer_copy_transform_with_storage_bindings_and_bounds_clamping() {
        // Compute shader: global_idx = (wg_id.x << 6) + local_id.x, byte_off = global_idx << 4
        //   v_lshl_add_u32 v3, s0, 6, v0          ; s0 seeded from wg_id.x, v0 from local_id.x
        //   v_lshlrev_b32  v3, 4, v3              ; byte offset = global_idx * 16
        //   buffer_load_dwordx4 v[4:7], v3, s[4:7], 0 offen          (srsrc=1 -> read-only-storage)
        //   buffer_load_ubyte   v8,     v3, s[4:7], 0 offen offset:2 (srsrc=1 -> read-only-storage)
        //   buffer_load_ushort  v9,     v3, s[4:7], 0 offen offset:4 (srsrc=1 -> read-only-storage)
        //   v_add_f32 v4, v4, 1.0
        //   v_mul_f32 v5, v5, 2.0
        //   v_add_nc_u32 v6, v6, v8
        //   v_add_nc_u32 v7, v7, v9
        //   buffer_store_dwordx4 v[4:7], v3, s[8:11], 0 offen        (srsrc=2 -> storage)
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_vop3(0x346, 3, 0, 128 + 6, 256 + 0, false, 0, 0, 0)); // v_lshl_add_u32 v3, s0, 6, v0
        words.push(enc_vop2(0x18, 3, 128 + 4, 3)); // v_lshlrev_b32 v3, 4, v3
        words.extend_from_slice(&enc_mubuf_full(0x0E, 0, true, false, 3, 4, 1, 128)); // buffer_load_dwordx4
        words.extend_from_slice(&enc_mubuf_full(0x08, 2, true, false, 3, 8, 1, 128)); // buffer_load_ubyte
        words.extend_from_slice(&enc_mubuf_full(0x0A, 4, true, false, 3, 9, 1, 128)); // buffer_load_ushort
        words.push(enc_vop2(0x03, 4, 242, 4)); // v_add_f32 v4, 1.0, v4
        words.push(enc_vop2(0x08, 5, 244, 5)); // v_mul_f32 v5, 2.0, v5
        words.push(enc_vop2(0x25, 6, 256 + 8, 6)); // v_add_nc_u32 v6, v8, v6
        words.push(enc_vop2(0x25, 7, 256 + 9, 7)); // v_add_nc_u32 v7, v9, v7
        words.extend_from_slice(&enc_mubuf_full(0x1E, 0, true, false, 3, 4, 2, 128)); // buffer_store_dwordx4
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("compute buffer copy/transform shader must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.entry_point, "cs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.binding_summary.has_storage_buffer);
        assert!(!out.binding_summary.has_storage_texture);
        assert!(!out.binding_summary.has_lds);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 3,
                    kind: "read-only-storage",
                    name: "sbuf0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 4,
                    kind: "storage",
                    name: "uav_buf0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
            ]
        );

        assert!(out.wgsl.contains("@compute @workgroup_size(64, 1, 1)"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@builtin(local_invocation_id) local_id: vec3<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@builtin(workgroup_id) wg_id: vec3<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("@builtin(global_invocation_id) global_id: vec3<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var<storage> sbuf0_: array<u32>") || out.wgsl.contains("var<storage, read> sbuf0_: array<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("var<storage, read_write> uav_buf0_: array<u32>"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("arrayLength("), "expected bounds clamping via arrayLength:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("var<workgroup>"), "must NOT emit LDS when no DS ops exist:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"), "must not emit enable f16");
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"kind\": \"read-only-storage\""), "JSON:\n{json}");
        assert!(json.contains("\"kind\": \"storage\""), "JSON:\n{json}");
        assert!(json.contains("\"has_storage_buffer\": true"), "JSON:\n{json}");
    }

    #[test]
    fn translates_compute_texture_filter_with_image_sample_lz_and_image_store() {
        // Compute 2D texture filter (workgroup_size 8x8x1):
        //   v_cvt_f32_u32 v4, v0   ; u = f32(local_id.x)
        //   v_cvt_f32_u32 v5, v1   ; v = f32(local_id.y)
        //   image_sample_lz v[6:9], v[4:5], s[8:15], s[4:7] dmask:0xf dim:1
        //   s_waitcnt vmcnt(0)
        //   v_mul_f32 v6, v6, 0.5
        //   image_store v[6:9], v[0:1], s[16:23] dmask:0xf dim:1
        //   image_store_mip v[6:9], v[0:2], s[16:23] dmask:0xf dim:1
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vop1(0x06, 4, 256 + 0));
        words.push(enc_vop1(0x06, 5, 256 + 1));
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 4, 6, 2, 1)); // image_sample_lz -> tex0@1, samp0@2
        words.push(enc_sopp(0x0C, 0x3F70));
        words.push(enc_vop2(0x08, 6, 240, 6));
        words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 6, 4, 0)); // image_store -> uav_tex0@3
        words.extend_from_slice(&enc_mimg(0x09, 0xF, 1, 0, 6, 4, 0)); // image_store_mip -> uav_tex0@3
        words.push(enc_sopp(0x01, 0));

        let opts = Rdna2TranslateOptions {
            workgroup_size: [8, 8, 1],
            ..Rdna2TranslateOptions::default()
        };
        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &opts)
            .expect("compute texture filter shader must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.entry_point, "cs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.binding_summary.has_texture_2d);
        assert!(out.binding_summary.has_sampler);
        assert!(out.binding_summary.has_storage_texture);
        assert!(!out.binding_summary.has_lds);
        assert_eq!(
            out.bindings,
            vec![
                WgslBindingInfo {
                    group: 0,
                    binding: 1,
                    kind: "texture_2d",
                    name: "tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 2,
                    kind: "sampler",
                    name: "samp0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
                WgslBindingInfo {
                    group: 0,
                    binding: 3,
                    kind: "storage-texture",
                    name: "uav_tex0".to_string(),
                    min_binding_size_bytes: None,
                    highest_vec4_offset: None,
                },
            ]
        );

        assert!(out.wgsl.contains("@compute @workgroup_size(8, 8, 1)"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("texture_2d<f32>"), "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("texture_storage_2d<rgba8unorm,write>")
                || out.wgsl.contains("texture_storage_2d<rgba8unorm, write>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(out.wgsl.contains("textureSampleLevel("), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("textureStore("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&out.wgsl);

        let json = out.to_json();
        assert!(json.contains("\"kind\": \"storage-texture\""), "JSON:\n{json}");
        assert!(json.contains("\"has_storage_texture\": true"), "JSON:\n{json}");
    }

    #[test]
    fn translates_compute_lds_reduction_exchange_with_barrier_and_buffer_store() {
        // Compute LDS reduction/exchange covering ds_write_b32, ds_write2_b32, ds_write_b64,
        // ds_write_b128, s_barrier, ds_read_b32, ds_read2_b32, ds_read_b64, ds_read_b128,
        // and buffer_store_dword{,x2,x3,x4}:
        let mut words = Vec::new();
        words.push(enc_vop2(0x18, 3, 128 + 2, 0)); // v_lshlrev_b32 v3, 2, v0 (byte addr = local_id.x * 4)
        words.push(enc_vop1(0x01, 4, 242)); // v4 = 1.0
        words.push(enc_vop1(0x01, 5, 244)); // v5 = 2.0
        words.push(enc_vop1(0x01, 6, 240)); // v6 = 0.5
        words.push(enc_vop1(0x01, 7, 246)); // v7 = 4.0
        words.extend_from_slice(&enc_ds(0x0D, 0, 0, 3, 4, 0, 0)); // ds_write_b32
        words.extend_from_slice(&enc_ds(0x0E, 1, 2, 3, 4, 5, 0)); // ds_write2_b32
        words.extend_from_slice(&enc_ds(0x4D, 16, 0, 3, 4, 0, 0)); // ds_write_b64
        words.extend_from_slice(&enc_ds(0xDF, 32, 0, 3, 4, 0, 0)); // ds_write_b128
        words.push(enc_sopp(0x0A, 0)); // s_barrier
        words.extend_from_slice(&enc_ds(0x36, 0, 0, 3, 0, 0, 8)); // ds_read_b32 -> v8
        words.extend_from_slice(&enc_ds(0x37, 1, 2, 3, 0, 0, 9)); // ds_read2_b32 -> v9..v10
        words.extend_from_slice(&enc_ds(0x76, 16, 0, 3, 0, 0, 11)); // ds_read_b64 -> v11..v12
        words.extend_from_slice(&enc_ds(0xFF, 32, 0, 3, 0, 0, 13)); // ds_read_b128 -> v13..v16
        words.push(enc_vop2(0x03, 8, 256 + 9, 8)); // v_add_f32 v8, v9, v8
        words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 3, 8, 1, 128)); // buffer_store_dword
        words.extend_from_slice(&enc_mubuf_full(0x1D, 16, true, false, 3, 11, 1, 128)); // buffer_store_dwordx2
        words.extend_from_slice(&enc_mubuf_full(0x1F, 32, true, false, 3, 13, 1, 128)); // buffer_store_dwordx3
        words.extend_from_slice(&enc_mubuf_full(0x0C, 0, true, false, 3, 17, 1, 128)); // buffer_load_dword on same UAV
        words.extend_from_slice(&enc_mubuf_full(0x0D, 8, true, false, 3, 18, 2, 128)); // buffer_load_dwordx2 on SRV
        words.extend_from_slice(&enc_mubuf_full(0x0F, 16, true, false, 3, 20, 2, 128)); // buffer_load_dwordx3 on SRV
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("compute LDS reduction shader must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Compute);
        assert_eq!(out.entry_point, "cs_main");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert!(out.binding_summary.has_lds);
        assert!(out.binding_summary.has_storage_buffer);
        // srsrc=1 has both loads and stores -> merged into a single read_write `storage` binding.
        // srsrc=2 has only loads -> `read-only-storage` binding.
        assert_eq!(out.bindings.len(), 2);
        assert_eq!(out.bindings[0].kind, "storage");
        assert_eq!(out.bindings[1].kind, "read-only-storage");

        assert!(out.wgsl.contains("var<workgroup> lds: array<u32, 4096>;"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("workgroupBarrier();"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("4095u"), "expected safe 4095u LDS index mask:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("enable f16"));
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn compute_workgroup_id_sgprs_respect_user_data_overwrite_and_custom_base() {
        // Case A: s0 overwritten by s_mov_b32 before read -> must NOT seed s_0 = wg_id.x,
        // while s1 is read before write -> MUST seed s_1 = wg_id.y.
        let mut words = Vec::new();
        words.push(enc_sop1(0x03, 0, 128 + 42)); // s_mov_b32 s0, 42 (overwrites s0)
        words.push(enc_sop2(0x00, 3, 0, 1)); // s_add_u32 s3, s0, s1 (reads s1 before write)
        words.push(enc_vop1(0x01, 4, 3)); // v_mov_b32 v4, s3
        words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 4, 1, 128));
        words.push(enc_sopp(0x01, 0));

        let out_a = translate_rdna2_to_wgsl(&words_to_bytes(&words), &Rdna2TranslateOptions::default())
            .expect("compute shader should translate cleanly");
        assert!(out_a.naga_validated);
        assert!(!out_a.wgsl.contains("s_0_ = wg_id.x"), "WGSL:\n{}", out_a.wgsl);
        assert!(out_a.wgsl.contains("s_1_ = wg_id.y"), "WGSL:\n{}", out_a.wgsl);
        assert_naga_wgsl_valid(&out_a.wgsl);

        // Case B: Custom workgroup_id_sgpr_base = Some(4) -> seeds s4 = wg_id.x.
        let mut words_b = Vec::new();
        words_b.push(enc_vop2(0x25, 4, 4, 0)); // v_add_nc_u32 v4, s4, v0
        words_b.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 4, 1, 128));
        words_b.push(enc_sopp(0x01, 0));

        let opts_b = Rdna2TranslateOptions {
            workgroup_id_sgpr_base: Some(4),
            ..Rdna2TranslateOptions::default()
        };
        let out_b = translate_rdna2_to_wgsl(&words_to_bytes(&words_b), &opts_b)
            .expect("custom wg_id sgpr base compute shader should translate cleanly");
        assert!(out_b.naga_validated);
        assert!(out_b.wgsl.contains("s_4_ = wg_id.x"), "WGSL:\n{}", out_b.wgsl);
        assert_naga_wgsl_valid(&out_b.wgsl);
    }

    #[test]
    fn tint_const_eval_safe_nan_and_shift_overflow() {
        // Exercises all live Chrome/Dawn Tint const-eval rejection patterns from Test #38:
        //   1. bitcast<f32>(4294967295u) / bitcast<f32>(4294967288u) (inline -1 / -8 as f32)
        //   2. unpack2x16float(32505u) / unpack2x16float(4294967295u) (f16 NaN in v_fmaak_f16 & v_pk_add_f16)
        //   3. 4294967295u << 16u (s_pack_ll_b32_b16), 3204448256u << 3u (s_lshl3_add_u32),
        //      4294967295u << 1u (s_lshl1_add_u32), and u32 const add/sub/mul overflow.
        let mut words = vec![
            // s_mov_b32 s2, 7
            enc_sop1(0x03, 2, 128 + 7),
            // s_pack_ll_b32_b16 s3, s2, -1  (src1 = 193 -> 4294967295u << 16u)
            enc_sop2(0x32, 3, 2, 193),
            // s_lshl3_add_u32 s4, -0.5, s2  (src0 = 241 -> 3204448256u << 3u)
            enc_sop2(0x30, 4, 241, 2),
            // s_lshl1_add_u32 s5, -1, s2    (src0 = 193 -> 4294967295u << 1u)
            enc_sop2(0x2E, 5, 193, 2),
            // s_add_u32 s6, -1, 16          (4294967295u + 16u const overflow -> folds to 15u)
            enc_sop2(0x00, 6, 193, 128 + 16),
            // v_mov_b32 v0, -1              (src0 = 193 -> bitcast<f32>(4294967295u))
            enc_vop1(0x01, 0, 193),
            // v_add_f32 v1, -8, v0          (src0 = 200 -> bitcast<f32>(4294967288u))
            enc_vop2(0x03, 1, 200, 0),
            // v_fmaak_f16 v2, v0, v1, 0x00007ef9 (32505u -> f16 NaN in unpack2x16float)
            enc_vop2(0x38, 2, 256, 1),
            0x0000_7EF9,
        ];
        // v_pk_add_f16 v3, -1, v2       (src0 = 193 -> unpack2x16float(4294967295u))
        words.extend_from_slice(&enc_vop3p(0x0F, 3, 193, 256 + 2, 0, false, 0, 0b011, 0, 0));
        // v_rcp_f32 v4, 0.0             (1.0 / 0.0 const-eval guard)
        words.push(enc_vop1(0x2A, 4, 128));
        // v_add_f32 v4, s3, v4
        words.push(enc_vop2(0x03, 4, 3, 4));
        // v_add_f32 v4, s4, v4
        words.push(enc_vop2(0x03, 4, 4, 4));
        // v_add_f32 v4, s5, v4
        words.push(enc_vop2(0x03, 4, 5, 4));
        // v_add_f32 v4, s6, v4
        words.push(enc_vop2(0x03, 4, 6, 4));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 4]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("shader with NaN bitcasts, f16 NaN unpacks, and shift overflows must validate");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // Must NEVER emit raw const-eval NaN bitcast<f32>, unpack2x16float, or overflowing shifts.
        assert!(
            !out.wgsl.contains("bitcast<f32>(4294967295u)"),
            "raw NaN bitcast<f32>(4294967295u) must not appear:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("bitcast<f32>(4294967288u)"),
            "raw NaN bitcast<f32>(4294967288u) must not appear:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("unpack2x16float(4294967295u)"),
            "raw NaN unpack2x16float(4294967295u) must not appear:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("unpack2x16float(32505u)"),
            "raw NaN unpack2x16float(32505u) must not appear:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("4294967295u << 16u"),
            "overflowing const shift 4294967295u << 16u must be folded:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("3204448256u << 3u"),
            "overflowing const shift 3204448256u << 3u must be folded:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("4294967295u << 1u"),
            "overflowing const shift 4294967295u << 1u must be folded:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("fn pg6_u32_bits_to_float("),
            "expected pg6_u32_bits_to_float runtime helper in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("fn pg6_unpack_half_pair("),
            "expected pg6_unpack_half_pair runtime helper in WGSL:\n{}",
            out.wgsl
        );
    }

    #[test]
    fn coalesces_descriptor_table_s_load_vsharp_into_s_buffer_load_cb0() {
        // Real PSO #50450 pattern:
        //   s_load_dwordx4        s[12:15], s[0:1], 0x0   ; fetch V# descriptor from root table
        //   s_waitcnt             lgkmcnt(0)
        //   s_buffer_load_dwordx4 s[8:11], s[12:15], 0x20 ; read actual CBV data[2] via V#
        //   s_waitcnt             lgkmcnt(0)
        //   v_mov_b32             v0, s8
        //   v_mov_b32             v1, s9
        //   v_mov_b32             v2, s10
        //   v_mov_b32             v3, s11
        //   exp mrt0              v0, v1, v2, v3 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.extend_from_slice(&enc_smem(0x02, 12, 0, 0x7D, 0x0, false, false)); // s_load_dwordx4 s[12:15], s[0:1], 0x0
        words.push(enc_sopp(0x0C, 0xC07F)); // s_waitcnt lgkmcnt(0)
        words.extend_from_slice(&enc_smem(0x0A, 8, 6, 0x7D, 0x20, false, false)); // s_buffer_load_dwordx4 s[8:11], s[12:15], 0x20 (sbase=6 -> s12)
        words.push(enc_sopp(0x0C, 0xC07F));
        words.push(enc_vop1(0x01, 0, 8));
        words.push(enc_vop1(0x01, 1, 9));
        words.push(enc_vop1(0x01, 2, 10));
        words.push(enc_vop1(0x01, 3, 11));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("descriptor-table V# -> s_buffer_load PS should translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Fragment);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_eq!(
            out.bindings,
            vec![WgslBindingInfo {
                group: 0,
                binding: 0,
                kind: "uniform",
                name: "cb0".to_string(),
                min_binding_size_bytes: Some(48),
                highest_vec4_offset: Some(2),
            }]
        );
        assert_eq!(out.binding_summary.cbv_vec4_count, 3);
        assert!(
            out.wgsl.contains("@group(0) @binding(0)"),
            "expected @group(0) @binding(0) in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("@binding(3)"),
            "must NOT emit spurious @binding(3) for coalesced descriptor-table CBV:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("array<vec4<u32>, 3>"),
            "expected tightly sized array<vec4<u32>, 3> (48 B) in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("cb0_.data[2]"),
            "expected direct cb0_.data[2] read in WGSL:\n{}",
            out.wgsl
        );
        assert_naga_wgsl_valid(&out.wgsl);

        // Also verify in-place overwrite: s_buffer_load_dwordx4 s[12:15], s[12:15], 0x20
        // followed by VALU consuming s12..s15 still coalesces to a single cb0 @binding(0)!
        let mut inplace_words = Vec::new();
        inplace_words.extend_from_slice(&enc_smem(0x02, 12, 0, 0x7D, 0x0, false, false));
        inplace_words.push(enc_sopp(0x0C, 0xC07F));
        inplace_words.extend_from_slice(&enc_smem(0x0A, 12, 6, 0x7D, 0x20, false, false));
        inplace_words.push(enc_sopp(0x0C, 0xC07F));
        inplace_words.push(enc_vop1(0x01, 0, 12));
        inplace_words.push(enc_vop1(0x01, 1, 13));
        inplace_words.push(enc_vop1(0x01, 2, 14));
        inplace_words.push(enc_vop1(0x01, 3, 15));
        inplace_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        inplace_words.push(enc_sopp(0x01, 0));

        let inplace_out = translate_rdna2_to_wgsl(
            &words_to_bytes(&inplace_words),
            &Rdna2TranslateOptions::default(),
        )
        .expect("in-place V# overwrite s_buffer_load PS should translate cleanly");
        assert!(inplace_out.naga_validated);
        assert_eq!(inplace_out.bindings.len(), 1);
        assert_eq!(inplace_out.bindings[0].binding, 0);
        assert_eq!(inplace_out.bindings[0].min_binding_size_bytes, Some(48));
        assert!(!inplace_out.wgsl.contains("@binding(3)"));
        assert!(inplace_out.wgsl.contains("cb0_.data[2]"));
        assert_naga_wgsl_valid(&inplace_out.wgsl);
    }

    #[test]
    fn is_safe_wgsl_f32_literal_predicate_bounds() {
        // Safe normal floats and +0.0 emit directly as WGSL f32 literals.
        assert!(is_safe_wgsl_f32_literal(0.0_f32));
        assert!(is_safe_wgsl_f32_literal(1.0_f32));
        assert!(is_safe_wgsl_f32_literal(-1.0_f32));
        assert!(is_safe_wgsl_f32_literal(0.5_f32));
        assert!(is_safe_wgsl_f32_literal(f32::MIN_POSITIVE));
        assert!(is_safe_wgsl_f32_literal(-f32::MIN_POSITIVE));
        assert!(is_safe_wgsl_f32_literal(3.39e38_f32));
        assert!(is_safe_wgsl_f32_literal(-3.39e38_f32));

        // Near-f32::MAX values (where Naga decimal printing can round up past f32::MAX)
        // must route through pg6_u32_bits_to_float(bits).
        assert!(!is_safe_wgsl_f32_literal(f32::MAX));
        assert!(!is_safe_wgsl_f32_literal(-f32::MAX));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x7F7F_FFFF)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0xFF7F_FFFF)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x7F7F_FFFE)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0xFF7F_FFFE)));
        assert!(!is_safe_wgsl_f32_literal(3.4e38_f32));
        assert!(!is_safe_wgsl_f32_literal(-3.4e38_f32));

        // -0.0, subnormals, infinities, and NaNs also route through pg6_u32_bits_to_float.
        assert!(!is_safe_wgsl_f32_literal(-0.0_f32));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x0000_0001)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x007F_FFFF)));
        assert!(!is_safe_wgsl_f32_literal(f32::from_bits(0x8000_0001)));
        assert!(!is_safe_wgsl_f32_literal(f32::INFINITY));
        assert!(!is_safe_wgsl_f32_literal(f32::NEG_INFINITY));
        assert!(!is_safe_wgsl_f32_literal(f32::NAN));
    }

    #[test]
    fn f32_max_and_neg_f32_max_literals_route_through_bitcast_helper() {
        // Reproduces Chrome/Dawn Tint parse rejection from Test #42:
        //   v_min3_f32 / v_max3_f32 / v_min_f32 / v_max_f32 / v_mov_b32 with
        //   0x7f7fffff (f32::MAX), 0xff7fffff (-f32::MAX), and 0x7f7ffffe.
        let mut words = Vec::new();
        // v_min3_f32 v0, 0x7f7fffff, v1, v2
        words.extend_from_slice(&enc_vop3(0x151, 0, 255, 256 + 1, 256 + 2, false, 0, 0, 0));
        words.push(0x7F7F_FFFF);
        // v_max3_f32 v1, 0xff7fffff, v0, v2
        words.extend_from_slice(&enc_vop3(0x154, 1, 255, 256, 256 + 2, false, 0, 0, 0));
        words.push(0xFF7F_FFFF);
        // v_min_f32 v2, 0x7f7fffff, v0
        words.push(enc_vop2(0x0F, 2, 255, 0));
        words.push(0x7F7F_FFFF);
        // v_max_f32 v3, 0xff7fffff, v1
        words.push(enc_vop2(0x10, 3, 255, 1));
        words.push(0xFF7F_FFFF);
        // v_mov_b32 v4, 0x7f7ffffe (near-max, also routed via bitcast helper)
        words.push(enc_vop1(0x01, 4, 255));
        words.push(0x7F7F_FFFE);
        // v_add_f32 v3, v3, v4
        words.push(enc_vop2(0x03, 3, 256 + 3, 4));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [0, 1, 2, 3]));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("shader with f32::MAX / -f32::MAX literals must translate cleanly");

        assert!(out.naga_validated);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        assert!(
            !out.wgsl.contains("340282350000000000000000000000000000000f"),
            "Naga rounded-up f32::MAX decimal literal must NOT appear in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            !out.wgsl.contains("340282330000000000000000000000000000000f"),
            "near-f32::MAX literal 0x7f7ffffe must NOT appear as raw decimal in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("fn pg6_u32_bits_to_float("),
            "expected pg6_u32_bits_to_float runtime helper in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("pg6_u32_bits_to_float(2139095039u)"),
            "expected exact 0x7f7fffff (2139095039u) bitcast call in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("pg6_u32_bits_to_float(4286578687u)"),
            "expected exact 0xff7fffff (4286578687u) bitcast call in WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("pg6_u32_bits_to_float(2139095038u)"),
            "expected exact 0x7f7ffffe (2139095038u) bitcast call in WGSL:\n{}",
            out.wgsl
        );
    }

    /// Synthetic Xbox Series preamble (36 bytes): 6 arbitrary header dwords + the fixed tail
    /// `[0xd5800000, 0x03fe0100, 0x00000ac0]` (the last decodes as `v_cndmask_b32 v0, 64, v5, vcc`).
    fn xbox_preamble_words() -> Vec<u32> {
        vec![
            0x0000_0001,
            0x0000_0024,
            0x1234_5678,
            0x0000_0000,
            0xCAFE_F00D,
            0x0000_0100,
            0xd580_0000,
            0x03fe_0100,
            0x0000_0ac0,
        ]
    }

    /// PSO #12114-shaped UI vertex shader body (starting at the real entry point):
    ///   s_inst_prefetch 0x3
    ///   s_mov_b32 s9, 0
    ///   s_mov_b64 s[12:13], s[0:1]
    ///   s_bfe_u32 s13, s13, 0x00080000
    ///   s_load_dwordx4 s[8:11], s[8:9], 0x0      ; CBV V# table
    ///   s_load_dwordx4 s[12:15], s[12:13], 0x0   ; vertex-buffer V# table
    ///   v_add_nc_u32 v0, s2, v0
    ///   tbuffer_load_format_xyzw v[4:7], v0, s[12:15], 0 format:56 idxen          ; COLOR0
    ///   tbuffer_load_format_x    v1,     v0, s[12:15], 0 format:1  idxen offset:7 ; COLOR1.w
    ///   buffer_load_dwordx2      v[2:3], v0, s[12:15], 0 idxen offset:8           ; POSITION0
    ///   s_buffer_load_dwordx16 s[8:23], s[8:11], 0x0
    ///   v_mul_f32 v8, s8, v2 ; v_mul_f32 v9, s9, v3
    ///   exp pos0 v8, v9, v2, v3 done ; exp param0 v4, v5, v6, v1 ; s_endpgm
    fn pso_12114_vs_body_words() -> Vec<u32> {
        let mut words = vec![
            0xbfa0_0003,                  // s_inst_prefetch 0x3
            enc_sop1(0x03, 9, 128),       // s_mov_b32 s9, 0
            enc_sop1(0x04, 12, 0),        // s_mov_b64 s[12:13], s[0:1]
            enc_sop2(0x27, 13, 13, 0xFF), // s_bfe_u32 s13, s13, 0x00080000
            0x0008_0000,
        ];
        words.extend_from_slice(&enc_smem(0x02, 8, 4, 0x7D, 0, false, false));
        words.extend_from_slice(&enc_smem(0x02, 12, 6, 0x7D, 0, false, false));
        words.push(enc_vop2(0x25, 0, 2, 0)); // v_add_nc_u32 v0, s2, v0
        words.extend_from_slice(&enc_mtbuf(0x03, 56, 0, 4, 0, 3, true));
        words.extend_from_slice(&enc_mtbuf(0x00, 1, 7, 1, 0, 3, true));
        words.extend_from_slice(&enc_mubuf(0x0D, 8, 2, 0, 3, true));
        words.extend_from_slice(&enc_smem(0x0C, 8, 4, 0x7D, 0, false, false));
        words.push(enc_vop2(0x08, 8, 8, 2)); // v_mul_f32 v8, s8, v2
        words.push(enc_vop2(0x08, 9, 9, 3)); // v_mul_f32 v9, s9, v3
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [8, 9, 2, 3]));
        words.extend_from_slice(&enc_exp(32, 0xF, false, false, false, [4, 5, 6, 1]));
        words.push(enc_sopp(0x01, 0));
        words
    }

    /// WGSL text without the leading `// Translated ...` / `// hash: ...` header lines.
    fn wgsl_without_hash_header(wgsl: &str) -> String {
        wgsl.lines()
            .filter(|l| !l.starts_with("// hash:"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn skips_xbox_preamble_preserves_full_blob_hash_and_never_clobbers_v0() {
        let body = pso_12114_vs_body_words();
        let mut words = xbox_preamble_words();
        words.extend_from_slice(&body);
        let full = words_to_bytes(&words);
        let bare = words_to_bytes(&body);
        assert_eq!(&full[0x24..], &bare[..]);

        let decoded = decode_program(&full);
        assert_eq!(decoded[0].offset, 0x24);
        assert_eq!(decoded[0].mnemonic, "s_inst_prefetch");
        assert!(decoded.iter().all(|i| i.mnemonic != "v_cndmask_b32"));

        let out = translate_rdna2_to_wgsl(&full, &Rdna2TranslateOptions::default())
            .expect("preamble-prefixed VS should translate");
        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        // Shader identity key: FNV-1a 64 over the FULL blob (matches d3d12_xs native_code_identity).
        assert_eq!(out.hash_fnv1a64, fnv1a_64(&full));
        assert_ne!(out.hash_fnv1a64, fnv1a_64(&bare));
        assert!(
            out.wgsl.contains(&format_shader_hash(fnv1a_64(&full))),
            "WGSL:\n{}",
            out.wgsl
        );
        // No bogus `v_cndmask_b32 v0, 64, v5, vcc` (v0 clobber / v5 pre-read -> legacy uv).
        assert!(!out.wgsl.contains("select("), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("64f"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains(" uv:"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("@location(3)"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("unknown_enc"), "WGSL:\n{}", out.wgsl);
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);

        // The stripped body translates identically apart from the hash header line.
        let bare_out = translate_rdna2_to_wgsl(&bare, &Rdna2TranslateOptions::default())
            .expect("bare VS body should translate");
        assert_eq!(bare_out.hash_fnv1a64, fnv1a_64(&bare));
        assert_eq!(
            wgsl_without_hash_header(&out.wgsl),
            wgsl_without_hash_header(&bare_out.wgsl)
        );
        assert_eq!(out.bindings, bare_out.bindings);
        assert_eq!(out.translated_op_count, bare_out.translated_op_count);
    }

    #[test]
    fn pso_12114_vertex_buffer_vsharp_emits_no_bogus_cbv_and_three_vs_inputs() {
        let mut words = xbox_preamble_words();
        words.extend_from_slice(&pso_12114_vs_body_words());
        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("PSO #12114-shaped VS should translate");

        assert!(out.naga_validated);
        assert_eq!(out.stage, ShaderStage::Vertex);
        assert_eq!(out.binding_summary.vertex_input_mode, "attributes");
        assert_eq!(out.unsupported_op_count, 0, "WGSL:\n{}", out.wgsl);

        // Only the CBV descriptor-table slot `cb0` at @binding(0); no `cb1` for the VB V#.
        assert_eq!(out.bindings.len(), 1, "bindings: {:?}", out.bindings);
        assert_eq!(out.bindings[0].group, 0);
        assert_eq!(out.bindings[0].binding, 0);
        assert_eq!(out.bindings[0].kind, "uniform");
        assert_eq!(out.bindings[0].name, "cb0");
        assert_eq!(out.bindings[0].highest_vec4_offset, Some(3));
        assert!(
            out.wgsl.contains("var<uniform> cb0_"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(!out.wgsl.contains("cb1"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("@binding(3)"), "WGSL:\n{}", out.wgsl);

        // VsIn: exactly locations 0 (COLOR0 unorm8x4), 1 (COLOR1 unorm8x4, `.w` read by the
        // offset:7 8-bit fetch) and 2 (POSITION0 float2).
        assert!(out.wgsl.contains("struct VsIn"), "WGSL:\n{}", out.wgsl);
        assert!(
            out.wgsl.contains("@location(0) position: vec4<f32>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("@location(1) attr_4_: vec4<f32>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(
            out.wgsl.contains("@location(2) attr_8_: vec2<f32>"),
            "WGSL:\n{}",
            out.wgsl
        );
        assert!(!out.wgsl.contains("@location(3)"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("attr_7"), "WGSL:\n{}", out.wgsl);
        assert!(out.wgsl.contains("vs_in.attr_4_.w"), "WGSL:\n{}", out.wgsl);
        assert!(!out.wgsl.contains("vs_in.attr_4_.x"), "WGSL:\n{}", out.wgsl);
        assert_naga_wgsl_valid(&out.wgsl);
    }

    #[test]
    fn vertex_buffer_vsharp_also_read_by_s_buffer_load_still_binds_cbv() {
        // A V# consumed by both an idxen vertex fetch and `s_buffer_load` is a real CBV.
        let mut words = vec![];
        words.extend_from_slice(&enc_smem(0x02, 12, 0, 0x7D, 0, false, false)); // s_load_dwordx4 s[12:15], s[0:1]
        words.extend_from_slice(&enc_mubuf(0x0D, 0, 2, 0, 3, true)); // buffer_load_dwordx2 v[2:3] idxen
        words.extend_from_slice(&enc_smem(0x08, 16, 6, 0x7D, 0, false, false)); // s_buffer_load_dword s16, s[12:15]
        words.push(enc_vop2(0x08, 8, 16, 2)); // v_mul_f32 v8, s16, v2
        words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [8, 3, 2, 3]));
        words.push(enc_sopp(0x01, 0));
        let bytes = words_to_bytes(&words);
        let out = translate_rdna2_to_wgsl(&bytes, &Rdna2TranslateOptions::default())
            .expect("VS should translate");
        assert_eq!(out.bindings.len(), 1, "bindings: {:?}", out.bindings);
        assert_eq!(out.bindings[0].name, "cb0");
        assert_naga_wgsl_valid(&out.wgsl);
    }
}
