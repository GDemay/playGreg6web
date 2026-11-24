//! Pure-Rust AMD RDNA2 (GFX10.3 / Xbox Series Scarlett) ISA decoder and
//! control-flow / export / resource analyzer.
//!
//! Decodes standard little-endian 32-bit and 64-bit (plus 96-bit literal/NSA)
//! RDNA2 machine-code instruction streams without invoking any external tools,
//! with zero `unsafe` code and guaranteed panic-free behavior on arbitrary,
//! truncated, or misaligned inputs.
//!
//! Primary purpose (Track 4 / S14):
//! * Decode console-compiled RDNA2 shader binaries directly into structured
//!   instructions and human-readable disassembly.
//! * Classify shader stage (`Vertex`, `Pixel`, `Compute`, `Unknown`) from
//!   `EXP` (`pos*`/`param*` vs `mrt*`/`mrtz`), `VINTRP`, and memory usage.
//! * Extract `EXP` targets, `SMEM` descriptor base SGPR pairs, `MIMG`/`MUBUF`/
//!   `MTBUF` resource references, control-flow branch targets, register
//!   high-water marks, and deterministic FNV-1a 64-bit hashes.
//! * Emit strictly sanitized text and JSON summaries (never host paths or raw
//!   bytecode dumps).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};

use crate::convert::{fnv1a_64, format_shader_hash};
use crate::pack::write_json_string;

/// AMD RDNA2 (GFX10.3) instruction encoding family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rdna2Encoding {
    Sopp,
    Sopc,
    Sop1,
    Sopk,
    Sop2,
    Smem,
    Vop1,
    Vopc,
    Vop2,
    Vop3,
    Vop3P,
    Vintrp,
    Exp,
    Mubuf,
    Mtbuf,
    Mimg,
    Flat,
    Ds,
    Unknown,
}

impl Rdna2Encoding {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Sopp => "SOPP",
            Self::Sopc => "SOPC",
            Self::Sop1 => "SOP1",
            Self::Sopk => "SOPK",
            Self::Sop2 => "SOP2",
            Self::Smem => "SMEM",
            Self::Vop1 => "VOP1",
            Self::Vopc => "VOPC",
            Self::Vop2 => "VOP2",
            Self::Vop3 => "VOP3",
            Self::Vop3P => "VOP3P",
            Self::Vintrp => "VINTRP",
            Self::Exp => "EXP",
            Self::Mubuf => "MUBUF",
            Self::Mtbuf => "MTBUF",
            Self::Mimg => "MIMG",
            Self::Flat => "FLAT",
            Self::Ds => "DS",
            Self::Unknown => "UNKNOWN",
        }
    }
}

impl fmt::Display for Rdna2Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Shader pipeline stage inferred heuristically from decoded RDNA2 instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rdna2InferredStage {
    Vertex,
    Pixel,
    Compute,
    Unknown,
}

impl Rdna2InferredStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Vertex => "vertex",
            Self::Pixel => "pixel",
            Self::Compute => "compute",
            Self::Unknown => "unknown",
        }
    }
}

impl fmt::Display for Rdna2InferredStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Decoded `EXP` instruction target slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Rdna2ExportTarget {
    Mrt(u8),
    MrtZ,
    Null,
    Pos(u8),
    Prim,
    Param(u8),
    Unknown(u8),
}

impl Rdna2ExportTarget {
    /// Decode the 6-bit `EXP.target` field (`w0[9:4]`).
    pub const fn from_raw(raw: u8) -> Self {
        match raw {
            0..=7 => Self::Mrt(raw),
            8 => Self::MrtZ,
            9 => Self::Null,
            12..=16 => Self::Pos(raw - 12),
            20 => Self::Prim,
            32..=63 => Self::Param(raw - 32),
            other => Self::Unknown(other),
        }
    }

    pub const fn is_pixel_output(self) -> bool {
        matches!(self, Self::Mrt(_) | Self::MrtZ)
    }

    pub const fn is_vertex_output(self) -> bool {
        matches!(self, Self::Pos(_) | Self::Param(_) | Self::Prim)
    }

    pub fn name(self) -> String {
        match self {
            Self::Mrt(idx) => format!("mrt{idx}"),
            Self::MrtZ => "mrtz".to_string(),
            Self::Null => "null".to_string(),
            Self::Pos(idx) => format!("pos{idx}"),
            Self::Prim => "prim".to_string(),
            Self::Param(idx) => format!("param{idx}"),
            Self::Unknown(raw) => format!("target_{raw}"),
        }
    }
}

impl fmt::Display for Rdna2ExportTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// Structured encoding-specific operand metadata for a decoded RDNA2 instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rdna2Detail {
    Sopp {
        op: u8,
        simm16: u16,
        branch_target: Option<i64>,
    },
    Sopc {
        op: u8,
        ssrc0: u8,
        ssrc1: u8,
        literal: Option<u32>,
    },
    Sop1 {
        op: u8,
        sdst: u8,
        ssrc0: u8,
        literal: Option<u32>,
    },
    Sopk {
        op: u8,
        sdst: u8,
        simm16: u16,
        literal: Option<u32>,
        branch_target: Option<i64>,
    },
    Sop2 {
        op: u8,
        sdst: u8,
        ssrc0: u8,
        ssrc1: u8,
        literal: Option<u32>,
    },
    Smem {
        op: u8,
        sdata: u8,
        sbase: u8,
        soffset: u8,
        offset: i32,
        glc: bool,
        dlc: bool,
    },
    Vop1 {
        op: u8,
        vdst: u8,
        src0: u16,
        literal: Option<u32>,
    },
    Vopc {
        op: u8,
        vsrc1: u8,
        src0: u16,
        literal: Option<u32>,
    },
    Vop2 {
        op: u8,
        vdst: u8,
        vsrc1: u8,
        src0: u16,
        literal: Option<u32>,
    },
    Vop3 {
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
        op_sel: u8,
        literal: Option<u32>,
    },
    Vop3P {
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
    },
    Vintrp {
        op: u8,
        vdst: u8,
        attr: u8,
        attrchan: u8,
        vsrc: u8,
    },
    Exp {
        target: Rdna2ExportTarget,
        en: u8,
        compr: bool,
        done: bool,
        vm: bool,
        vsrc: [u8; 4],
    },
    Mubuf {
        op: u8,
        offset: u16,
        offen: bool,
        idxen: bool,
        glc: bool,
        dlc: bool,
        slc: bool,
        lds: bool,
        tfe: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
    },
    Mtbuf {
        op: u8,
        format: u8,
        offset: u16,
        offen: bool,
        idxen: bool,
        glc: bool,
        dlc: bool,
        slc: bool,
        tfe: bool,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        soffset: u8,
    },
    Mimg {
        op: u8,
        dmask: u8,
        dim: u8,
        unrm: bool,
        glc: bool,
        dlc: bool,
        slc: bool,
        r128: bool,
        tfe: bool,
        lwe: bool,
        a16: bool,
        d16: bool,
        nsize: u8,
        vaddr: u8,
        vdata: u8,
        srsrc: u8,
        ssamp: u8,
    },
    Flat {
        seg: u8,
        op: u8,
        offset: i16,
        glc: bool,
        slc: bool,
        dlc: bool,
        lds: bool,
        addr: u8,
        data: u8,
        saddr: u8,
        vdst: u8,
    },
    Ds {
        op: u8,
        gds: bool,
        offset0: u8,
        offset1: u8,
        addr: u8,
        data0: u8,
        data1: u8,
        vdst: u8,
    },
    Unknown {
        prefix: u8,
    },
}

/// One decoded RDNA2 instruction at byte `offset` with total length `size` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2Instruction {
    pub offset: usize,
    pub size: usize,
    pub encoding: Rdna2Encoding,
    pub mnemonic: String,
    pub operands: String,
    pub truncated: bool,
    pub detail: Rdna2Detail,
}

impl Rdna2Instruction {
    /// Format this instruction as sanitized assembly text (`mnemonic operands`).
    pub fn format_text(&self) -> String {
        let mut out = if self.operands.is_empty() {
            self.mnemonic.clone()
        } else {
            format!("{} {}", self.mnemonic, self.operands)
        };
        if self.truncated {
            out.push_str(" ; <truncated>");
        }
        out
    }
}

/// Extracted `EXP` instruction summary entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2ExportSummary {
    pub offset: usize,
    pub target: Rdna2ExportTarget,
    pub en: u8,
    pub compr: bool,
    pub done: bool,
    pub vm: bool,
    pub vsrc: [u8; 4],
}

/// Extracted `SMEM` scalar memory reference summary entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2SmemRef {
    pub offset: usize,
    pub mnemonic: String,
    /// Actual base SGPR index (`sbase * 2`).
    pub sbase_sgpr: u8,
    pub sdata_sgpr: u8,
    pub byte_offset: i32,
    pub is_buffer: bool,
    pub glc: bool,
    pub dlc: bool,
}

/// Extracted `MIMG` / `MUBUF` / `MTBUF` descriptor resource reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2ResourceRef {
    pub offset: usize,
    pub encoding: Rdna2Encoding,
    pub mnemonic: String,
    /// Actual resource descriptor base SGPR index (`srsrc * 4`).
    pub srsrc_sgpr: u8,
    /// Actual sampler descriptor base SGPR index (`ssamp * 4`), if sampled `MIMG`.
    pub ssamp_sgpr: Option<u8>,
    pub dmask: Option<u8>,
    pub vaddr: u8,
    pub vdata: u8,
}

/// Extracted control-flow branch reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2BranchRef {
    pub offset: usize,
    pub mnemonic: String,
    pub target_offset: i64,
    pub conditional: bool,
}

/// Complete structural, control-flow, export, and resource summary for an RDNA2 program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rdna2ProgramSummary {
    pub fnv1a_hash: u64,
    pub byte_len: usize,
    pub trailing_bytes: usize,
    pub instruction_count: usize,
    pub truncated_instructions: usize,
    pub unknown_instructions: usize,
    pub endpgm_count: usize,
    pub first_endpgm_offset: Option<usize>,
    pub inferred_stage: Rdna2InferredStage,
    pub encoding_counts: BTreeMap<Rdna2Encoding, usize>,
    pub exports: Vec<Rdna2ExportSummary>,
    pub smem_refs: Vec<Rdna2SmemRef>,
    pub smem_base_sgprs: Vec<u8>,
    pub resource_refs: Vec<Rdna2ResourceRef>,
    pub branches: Vec<Rdna2BranchRef>,
    pub max_sgpr: Option<u8>,
    pub max_vgpr: Option<u8>,
    pub has_vintrp: bool,
    pub has_barriers: bool,
    pub has_lds: bool,
}

impl Rdna2ProgramSummary {
    /// Render this summary as strictly sanitized deterministic JSON (no host paths, no raw bytes).
    pub fn to_json(&self) -> String {
        to_json(self)
    }
}

// ---------------------------------------------------------------------------
// Low-level decoding helpers
// ---------------------------------------------------------------------------

fn read_u32_le(bytes: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    let slice = bytes.get(offset..end)?;
    Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

/// Compute relative SOPP/SOPK branch target byte offset: `pc + 4 + sext(simm16) * 4`.
fn compute_branch_target(offset: usize, simm16: u16) -> i64 {
    let rel_dwords = i64::from(simm16 as i16);
    (offset as i64).saturating_add(4).saturating_add(rel_dwords.saturating_mul(4))
}

/// Sign-extend a 21-bit signed SMEM offset to `i32`.
fn sign_extend_21(raw: u32) -> i32 {
    let masked = raw & 0x001F_FFFF;
    if (masked & 0x0010_0000) != 0 {
        (masked | 0xFFE0_0000) as i32
    } else {
        masked as i32
    }
}

/// Sign-extend a 12-bit signed FLAT/GLOBAL/SCRATCH offset to `i16`.
fn sign_extend_12(raw: u16) -> i16 {
    let masked = raw & 0x0FFF;
    if (masked & 0x0800) != 0 {
        (masked | 0xF000) as i16
    } else {
        masked as i16
    }
}

/// Whether a 9-bit vector/scalar source operand consumes an extra 32-bit word
/// (`0xFF` literal constant, `0xF9` SDWA, `0xFA` DPP16, `0xE9` DPP8, `0xEA` DPP8FI).
const fn src_uses_extra_word(src: u16) -> bool {
    matches!(src, 0xFF | 0xF9 | 0xFA | 0xE9 | 0xEA)
}

/// Format a 7-bit scalar destination register (`sdst`).
fn format_sdst(sdst: u8) -> String {
    match sdst {
        0..=105 => format!("s{sdst}"),
        106 => "vcc_lo".to_string(),
        107 => "vcc_hi".to_string(),
        108..=123 => format!("ttmp{}", sdst - 108),
        124 => "m0".to_string(),
        125 => "null".to_string(),
        126 => "exec_lo".to_string(),
        127 => "exec_hi".to_string(),
        other => format!("s_unk_{other}"),
    }
}

/// Format a 64-bit SGPR pair destination (`s[lo:hi]` or special 64-bit pair).
fn format_sdst_pair(sdst: u8) -> String {
    match sdst {
        0..=104 => format!("s[{}:{}]", sdst, sdst + 1),
        106 => "vcc".to_string(),
        125 => "null".to_string(),
        126 => "exec".to_string(),
        other => format_sdst(other),
    }
}

/// Format an SGPR range `s[base:base+count-1]` (or single `sN` when `count == 1`).
fn format_sgpr_range(base: u8, count: u8) -> String {
    if count <= 1 {
        format_sdst(base)
    } else if base <= 105 && base.saturating_add(count - 1) <= 105 {
        format!("s[{}:{}]", base, base + count - 1)
    } else if base == 106 && count == 2 {
        "vcc".to_string()
    } else if base == 126 && count == 2 {
        "exec".to_string()
    } else if base == 125 {
        "null".to_string()
    } else {
        format!("s[{}:{}]", base, base.saturating_add(count - 1))
    }
}

/// Format a 9-bit source operand (covers 8-bit `ssrc` as well as `v0..v255` at `256..511`).
fn format_src_operand(src: u16, literal: Option<u32>) -> String {
    match src {
        0..=105 => format!("s{src}"),
        106 => "vcc_lo".to_string(),
        107 => "vcc_hi".to_string(),
        108..=123 => format!("ttmp{}", src - 108),
        124 => "m0".to_string(),
        125 => "null".to_string(),
        126 => "exec_lo".to_string(),
        127 => "exec_hi".to_string(),
        128..=192 => format!("{}", src - 128),
        193..=208 => format!("-{}", src - 192),
        240 => "0.5".to_string(),
        241 => "-0.5".to_string(),
        242 => "1.0".to_string(),
        243 => "-1.0".to_string(),
        244 => "2.0".to_string(),
        245 => "-2.0".to_string(),
        246 => "4.0".to_string(),
        247 => "-4.0".to_string(),
        248 => "0.15915494".to_string(),
        251 => "src_vccz".to_string(),
        252 => "src_execz".to_string(),
        253 => "src_scc".to_string(),
        254 => "src_lds_direct".to_string(),
        255 => match literal {
            Some(lit) => format!("0x{lit:08x}"),
            None => "<lit>".to_string(),
        },
        0xF9 => "sdwa".to_string(),
        0xFA => "dpp16".to_string(),
        0xE9 => "dpp8".to_string(),
        0xEA => "dpp8fi".to_string(),
        256..=511 => format!("v{}", src - 256),
        other => format!("src_{other}"),
    }
}

/// Format a VOP3 source operand with optional `abs` (`|x|`) and `neg` (`-x`) modifiers.
fn format_vop3_src(src: u16, literal: Option<u32>, abs: bool, neg: bool) -> String {
    let base = format_src_operand(src, literal);
    match (neg, abs) {
        (true, true) => format!("-|{base}|"),
        (true, false) => format!("-{base}"),
        (false, true) => format!("|{base}|"),
        (false, false) => base,
    }
}

fn format_waitcnt(simm16: u16) -> String {
    let vmcnt = (simm16 & 0xF) | (((simm16 >> 14) & 0x3) << 4);
    let expcnt = (simm16 >> 4) & 0x7;
    let lgkmcnt = (simm16 >> 8) & 0x3F;
    let mut parts = Vec::new();
    if vmcnt != 0x3F {
        parts.push(format!("vmcnt({vmcnt})"));
    }
    if expcnt != 0x7 {
        parts.push(format!("expcnt({expcnt})"));
    }
    if lgkmcnt != 0x3F {
        parts.push(format!("lgkmcnt({lgkmcnt})"));
    }
    if parts.is_empty() {
        "0".to_string()
    } else {
        parts.join(" ")
    }
}

// ---------------------------------------------------------------------------
// Opcode mnemonic lookup tables (AMD RDNA2 / GFX10.3 ISA)
// ---------------------------------------------------------------------------

fn sopp_mnemonic(op: u8) -> (String, bool, bool) {
    // Returns (mnemonic, is_branch, is_conditional_branch)
    match op {
        0x00 => ("s_nop".to_string(), false, false),
        0x01 => ("s_endpgm".to_string(), false, false),
        0x02 => ("s_branch".to_string(), true, false),
        0x03 => ("s_wakeup".to_string(), false, false),
        0x04 => ("s_cbranch_scc0".to_string(), true, true),
        0x05 => ("s_cbranch_scc1".to_string(), true, true),
        0x06 => ("s_cbranch_vccz".to_string(), true, true),
        0x07 => ("s_cbranch_vccnz".to_string(), true, true),
        0x08 => ("s_cbranch_execz".to_string(), true, true),
        0x09 => ("s_cbranch_execnz".to_string(), true, true),
        0x0A => ("s_barrier".to_string(), false, false),
        0x0B => ("s_setkill".to_string(), false, false),
        0x0C => ("s_waitcnt".to_string(), false, false),
        0x0D => ("s_sethalt".to_string(), false, false),
        0x0E => ("s_sleep".to_string(), false, false),
        0x0F => ("s_setprio".to_string(), false, false),
        0x10 => ("s_sendmsg".to_string(), false, false),
        0x11 => ("s_sendmsghalt".to_string(), false, false),
        0x12 => ("s_trap".to_string(), false, false),
        0x13 => ("s_icache_inv".to_string(), false, false),
        0x14 => ("s_incperflevel".to_string(), false, false),
        0x15 => ("s_decperflevel".to_string(), false, false),
        0x16 => ("s_ttracedata".to_string(), false, false),
        0x17 => ("s_cbranch_cdbgsys".to_string(), true, true),
        0x18 => ("s_cbranch_cdbguser".to_string(), true, true),
        0x19 => ("s_cbranch_cdbgsys_or_user".to_string(), true, true),
        0x1A => ("s_cbranch_cdbgsys_and_user".to_string(), true, true),
        0x1B => ("s_endpgm_saved".to_string(), false, false),
        0x1C => ("s_set_gpr_idx_off".to_string(), false, false),
        0x1D => ("s_set_gpr_idx_mode".to_string(), false, false),
        0x1E => ("s_endpgm_ordered_ps_done".to_string(), false, false),
        0x1F => ("s_code_end".to_string(), false, false),
        0x20 => ("s_inst_prefetch".to_string(), false, false),
        0x21 => ("s_clause".to_string(), false, false),
        0x22 => ("s_wait_idle".to_string(), false, false),
        0x23 => ("s_waitcnt_depctr".to_string(), false, false),
        0x24 => ("s_round_mode".to_string(), false, false),
        0x25 => ("s_denorm_mode".to_string(), false, false),
        0x28 => ("s_ttracedata_imm".to_string(), false, false),
        other => (format!("sopp_op_{other:#04x}"), false, false),
    }
}

fn sopc_mnemonic(op: u8) -> String {
    match op {
        0x00 => "s_cmp_eq_i32",
        0x01 => "s_cmp_lg_i32",
        0x02 => "s_cmp_gt_i32",
        0x03 => "s_cmp_ge_i32",
        0x04 => "s_cmp_lt_i32",
        0x05 => "s_cmp_le_i32",
        0x06 => "s_cmp_eq_u32",
        0x07 => "s_cmp_lg_u32",
        0x08 => "s_cmp_gt_u32",
        0x09 => "s_cmp_ge_u32",
        0x0A => "s_cmp_lt_u32",
        0x0B => "s_cmp_le_u32",
        0x0C => "s_bitcmp0_b32",
        0x0D => "s_bitcmp1_b32",
        0x0E => "s_bitcmp0_b64",
        0x0F => "s_bitcmp1_b64",
        0x10 => "s_cmp_eq_u64",
        0x11 => "s_cmp_lg_u64",
        other => return format!("sopc_op_{other:#04x}"),
    }
    .to_string()
}

fn sop1_mnemonic(op: u8) -> (String, bool) {
    // Returns (mnemonic, is_64bit_dst)
    match op {
        0x03 => ("s_mov_b32".to_string(), false),
        0x04 => ("s_mov_b64".to_string(), true),
        0x05 => ("s_cmov_b32".to_string(), false),
        0x06 => ("s_cmov_b64".to_string(), true),
        0x07 => ("s_not_b32".to_string(), false),
        0x08 => ("s_not_b64".to_string(), true),
        0x09 => ("s_wqm_b32".to_string(), false),
        0x0A => ("s_wqm_b64".to_string(), true),
        0x0B => ("s_brev_b32".to_string(), false),
        0x0C => ("s_brev_b64".to_string(), true),
        0x0D => ("s_bcnt0_i32_b32".to_string(), false),
        0x0E => ("s_bcnt0_i32_b64".to_string(), false),
        0x0F => ("s_bcnt1_i32_b32".to_string(), false),
        0x10 => ("s_bcnt1_i32_b64".to_string(), false),
        0x11 => ("s_ff0_i32_b32".to_string(), false),
        0x12 => ("s_ff0_i32_b64".to_string(), false),
        0x13 => ("s_ff1_i32_b32".to_string(), false),
        0x14 => ("s_ff1_i32_b64".to_string(), false),
        0x15 => ("s_flbit_i32_b32".to_string(), false),
        0x16 => ("s_flbit_i32_b64".to_string(), false),
        0x17 => ("s_flbit_i32".to_string(), false),
        0x18 => ("s_flbit_i32_i64".to_string(), false),
        0x19 => ("s_sext_i32_i8".to_string(), false),
        0x1A => ("s_sext_i32_i16".to_string(), false),
        0x1B => ("s_bitset0_b32".to_string(), false),
        0x1C => ("s_bitset0_b64".to_string(), true),
        0x1D => ("s_bitset1_b32".to_string(), false),
        0x1E => ("s_bitset1_b64".to_string(), true),
        0x1F => ("s_getpc_b64".to_string(), true),
        0x20 => ("s_setpc_b64".to_string(), false),
        0x21 => ("s_swappc_b64".to_string(), true),
        0x22 => ("s_rfe_b64".to_string(), false),
        0x24 => ("s_and_saveexec_b64".to_string(), true),
        0x25 => ("s_or_saveexec_b64".to_string(), true),
        0x26 => ("s_xor_saveexec_b64".to_string(), true),
        0x27 => ("s_andn2_saveexec_b64".to_string(), true),
        0x28 => ("s_orn2_saveexec_b64".to_string(), true),
        0x29 => ("s_nand_saveexec_b64".to_string(), true),
        0x2A => ("s_nor_saveexec_b64".to_string(), true),
        0x2B => ("s_xnor_saveexec_b64".to_string(), true),
        0x2C => ("s_quadmask_b32".to_string(), false),
        0x2D => ("s_quadmask_b64".to_string(), true),
        0x2E => ("s_movrels_b32".to_string(), false),
        0x2F => ("s_movrels_b64".to_string(), true),
        0x30 => ("s_movreld_b32".to_string(), false),
        0x31 => ("s_movreld_b64".to_string(), true),
        0x34 => ("s_abs_i32".to_string(), false),
        0x37 => ("s_andn1_saveexec_b64".to_string(), true),
        0x38 => ("s_orn1_saveexec_b64".to_string(), true),
        0x39 => ("s_andn1_wrexec_b64".to_string(), true),
        0x3A => ("s_andn2_wrexec_b64".to_string(), true),
        0x3B => ("s_bitreplicate_b64_b32".to_string(), true),
        0x3C => ("s_and_saveexec_b32".to_string(), false),
        0x3D => ("s_or_saveexec_b32".to_string(), false),
        0x3E => ("s_xor_saveexec_b32".to_string(), false),
        0x3F => ("s_andn2_saveexec_b32".to_string(), false),
        0x40 => ("s_orn2_saveexec_b32".to_string(), false),
        0x41 => ("s_nand_saveexec_b32".to_string(), false),
        0x42 => ("s_nor_saveexec_b32".to_string(), false),
        0x43 => ("s_xnor_saveexec_b32".to_string(), false),
        0x44 => ("s_andn1_saveexec_b32".to_string(), false),
        0x45 => ("s_orn1_saveexec_b32".to_string(), false),
        0x46 => ("s_andn1_wrexec_b32".to_string(), false),
        0x47 => ("s_andn2_wrexec_b32".to_string(), false),
        0x49 => ("s_movrelsd_2_b32".to_string(), false),
        other => (format!("sop1_op_{other:#04x}"), false),
    }
}

fn sopk_mnemonic(op: u8) -> (String, bool, bool) {
    // Returns (mnemonic, has_imm32, is_branch)
    match op {
        0x00 => ("s_movk_i32".to_string(), false, false),
        0x01 => ("s_version".to_string(), false, false),
        0x02 => ("s_cmovk_i32".to_string(), false, false),
        0x03 => ("s_cmpk_eq_i32".to_string(), false, false),
        0x04 => ("s_cmpk_lg_i32".to_string(), false, false),
        0x05 => ("s_cmpk_gt_i32".to_string(), false, false),
        0x06 => ("s_cmpk_ge_i32".to_string(), false, false),
        0x07 => ("s_cmpk_lt_i32".to_string(), false, false),
        0x08 => ("s_cmpk_le_i32".to_string(), false, false),
        0x09 => ("s_cmpk_eq_u32".to_string(), false, false),
        0x0A => ("s_cmpk_lg_u32".to_string(), false, false),
        0x0B => ("s_cmpk_gt_u32".to_string(), false, false),
        0x0C => ("s_cmpk_ge_u32".to_string(), false, false),
        0x0D => ("s_cmpk_lt_u32".to_string(), false, false),
        0x0E => ("s_cmpk_le_u32".to_string(), false, false),
        0x0F => ("s_addk_i32".to_string(), false, false),
        0x10 => ("s_mulk_i32".to_string(), false, false),
        0x11 => ("s_getreg_b32".to_string(), false, false),
        0x12 => ("s_setreg_b32".to_string(), false, false),
        0x15 => ("s_setreg_imm32_b32".to_string(), true, false),
        0x16 => ("s_call_b64".to_string(), false, true),
        0x17 => ("s_waitcnt_vscnt".to_string(), false, false),
        0x18 => ("s_waitcnt_vmcnt".to_string(), false, false),
        0x19 => ("s_waitcnt_expcnt".to_string(), false, false),
        0x1A => ("s_waitcnt_lgkmcnt".to_string(), false, false),
        0x1B => ("s_subvector_loop_begin".to_string(), false, true),
        0x1C => ("s_subvector_loop_end".to_string(), false, true),
        other => (format!("sopk_op_{other:#04x}"), false, false),
    }
}

fn sop2_mnemonic(op: u8) -> (String, bool) {
    // Returns (mnemonic, is_64bit_dst)
    match op {
        0x00 => ("s_add_u32".to_string(), false),
        0x01 => ("s_sub_u32".to_string(), false),
        0x02 => ("s_add_i32".to_string(), false),
        0x03 => ("s_sub_i32".to_string(), false),
        0x04 => ("s_addc_u32".to_string(), false),
        0x05 => ("s_subb_u32".to_string(), false),
        0x06 => ("s_min_i32".to_string(), false),
        0x07 => ("s_min_u32".to_string(), false),
        0x08 => ("s_max_i32".to_string(), false),
        0x09 => ("s_max_u32".to_string(), false),
        0x0A => ("s_cselect_b32".to_string(), false),
        0x0B => ("s_cselect_b64".to_string(), true),
        0x0E => ("s_and_b32".to_string(), false),
        0x0F => ("s_and_b64".to_string(), true),
        0x10 => ("s_or_b32".to_string(), false),
        0x11 => ("s_or_b64".to_string(), true),
        0x12 => ("s_xor_b32".to_string(), false),
        0x13 => ("s_xor_b64".to_string(), true),
        0x14 => ("s_andn2_b32".to_string(), false),
        0x15 => ("s_andn2_b64".to_string(), true),
        0x16 => ("s_orn2_b32".to_string(), false),
        0x17 => ("s_orn2_b64".to_string(), true),
        0x18 => ("s_nand_b32".to_string(), false),
        0x19 => ("s_nand_b64".to_string(), true),
        0x1A => ("s_nor_b32".to_string(), false),
        0x1B => ("s_nor_b64".to_string(), true),
        0x1C => ("s_xnor_b32".to_string(), false),
        0x1D => ("s_xnor_b64".to_string(), true),
        0x1E => ("s_lshl_b32".to_string(), false),
        0x1F => ("s_lshl_b64".to_string(), true),
        0x20 => ("s_lshr_b32".to_string(), false),
        0x21 => ("s_lshr_b64".to_string(), true),
        0x22 => ("s_ashr_i32".to_string(), false),
        0x23 => ("s_ashr_i64".to_string(), true),
        0x24 => ("s_bfm_b32".to_string(), false),
        0x25 => ("s_bfm_b64".to_string(), true),
        0x26 => ("s_mul_i32".to_string(), false),
        0x27 => ("s_bfe_u32".to_string(), false),
        0x28 => ("s_bfe_i32".to_string(), false),
        0x29 => ("s_bfe_u64".to_string(), true),
        0x2A => ("s_bfe_i64".to_string(), true),
        0x2B => ("s_cbranch_g_fork".to_string(), false),
        0x2C => ("s_absdiff_i32".to_string(), false),
        0x2E => ("s_lshl1_add_u32".to_string(), false),
        0x2F => ("s_lshl2_add_u32".to_string(), false),
        0x30 => ("s_lshl3_add_u32".to_string(), false),
        0x31 => ("s_lshl4_add_u32".to_string(), false),
        0x32 => ("s_pack_ll_b32_b16".to_string(), false),
        0x33 => ("s_pack_lh_b32_b16".to_string(), false),
        0x34 => ("s_pack_hh_b32_b16".to_string(), false),
        0x35 => ("s_mul_hi_u32".to_string(), false),
        0x36 => ("s_mul_hi_i32".to_string(), false),
        other => (format!("sop2_op_{other:#04x}"), false),
    }
}

fn smem_info(op: u8) -> (String, u8, bool) {
    // Returns (mnemonic, sdata_reg_count, is_buffer)
    match op {
        0x00 => ("s_load_dword".to_string(), 1, false),
        0x01 => ("s_load_dwordx2".to_string(), 2, false),
        0x02 => ("s_load_dwordx4".to_string(), 4, false),
        0x03 => ("s_load_dwordx8".to_string(), 8, false),
        0x04 => ("s_load_dwordx16".to_string(), 16, false),
        0x08 => ("s_buffer_load_dword".to_string(), 1, true),
        0x09 => ("s_buffer_load_dwordx2".to_string(), 2, true),
        0x0A => ("s_buffer_load_dwordx4".to_string(), 4, true),
        0x0B => ("s_buffer_load_dwordx8".to_string(), 8, true),
        0x0C => ("s_buffer_load_dwordx16".to_string(), 16, true),
        0x20 => ("s_gl1_inv".to_string(), 0, false),
        0x21 => ("s_dcache_inv".to_string(), 0, false),
        0x24 => ("s_memtime".to_string(), 2, false),
        0x25 => ("s_memrealtime".to_string(), 2, false),
        0x26 => ("s_atc_probe".to_string(), 0, false),
        0x27 => ("s_atc_probe_buffer".to_string(), 0, true),
        0x28 => ("s_get_waveid_in_workgroup".to_string(), 1, false),
        other => (format!("smem_op_{other:#04x}"), 1, false),
    }
}

fn vop1_mnemonic(op: u8) -> String {
    match op {
        0x00 => "v_nop",
        0x01 => "v_mov_b32",
        0x02 => "v_readfirstlane_b32",
        0x03 => "v_cvt_i32_f64",
        0x04 => "v_cvt_f64_i32",
        0x05 => "v_cvt_f32_i32",
        0x06 => "v_cvt_f32_u32",
        0x07 => "v_cvt_u32_f32",
        0x08 => "v_cvt_i32_f32",
        0x0A => "v_cvt_f16_f32",
        0x0B => "v_cvt_f32_f16",
        0x0C => "v_cvt_rpi_i32_f32",
        0x0D => "v_cvt_flr_i32_f32",
        0x0E => "v_cvt_off_f32_i4",
        0x0F => "v_cvt_f32_f64",
        0x10 => "v_cvt_f64_f32",
        0x11 => "v_cvt_f32_ubyte0",
        0x12 => "v_cvt_f32_ubyte1",
        0x13 => "v_cvt_f32_ubyte2",
        0x14 => "v_cvt_f32_ubyte3",
        0x15 => "v_cvt_u32_f64",
        0x16 => "v_cvt_f64_u32",
        0x17 => "v_trunc_f64",
        0x18 => "v_ceil_f64",
        0x19 => "v_rndne_f64",
        0x1A => "v_floor_f64",
        0x1B => "v_pipeflush",
        0x20 => "v_fract_f32",
        0x21 => "v_trunc_f32",
        0x22 => "v_ceil_f32",
        0x23 => "v_rndne_f32",
        0x24 => "v_floor_f32",
        0x25 => "v_exp_f32",
        0x27 => "v_log_f32",
        0x2A => "v_rcp_f32",
        0x2B => "v_rcp_iflag_f32",
        0x2E => "v_rsq_f32",
        0x2F => "v_rcp_f64",
        0x31 => "v_rsq_f64",
        0x33 => "v_sqrt_f32",
        0x34 => "v_sqrt_f64",
        0x35 => "v_sin_f32",
        0x36 => "v_cos_f32",
        0x37 => "v_not_b32",
        0x38 => "v_bfrev_b32",
        0x39 => "v_ffbh_u32",
        0x3A => "v_ffbl_b32",
        0x3B => "v_ffbh_i32",
        0x3C => "v_frexp_exp_i32_f64",
        0x3D => "v_frexp_mant_f64",
        0x3E => "v_fract_f64",
        0x3F => "v_frexp_exp_i32_f32",
        0x40 => "v_frexp_mant_f32",
        0x41 => "v_clrexcp",
        0x42 => "v_movreld_b32",
        0x43 => "v_movrels_b32",
        0x44 => "v_movrelsd_b32",
        0x48 => "v_movrelsd_2_b32",
        0x50 => "v_cvt_f16_u16",
        0x51 => "v_cvt_f16_i16",
        0x52 => "v_cvt_u16_f16",
        0x53 => "v_cvt_i16_f16",
        0x54 => "v_rcp_f16",
        0x55 => "v_sqrt_f16",
        0x56 => "v_rsq_f16",
        0x57 => "v_log_f16",
        0x58 => "v_exp_f16",
        0x59 => "v_frexp_mant_f16",
        0x5A => "v_frexp_exp_i16_f16",
        0x5B => "v_floor_f16",
        0x5C => "v_ceil_f16",
        0x5D => "v_trunc_f16",
        0x5E => "v_rndne_f16",
        0x5F => "v_fract_f16",
        0x60 => "v_sin_f16",
        0x61 => "v_cos_f16",
        0x62 => "v_sat_pk_u8_i16",
        0x63 => "v_cvt_norm_i16_f16",
        0x64 => "v_cvt_norm_u16_f16",
        0x65 => "v_swap_b32",
        0x66 => "v_swaprel_b32",
        other => return format!("vop1_op_{other:#04x}"),
    }
    .to_string()
}

fn vopc_mnemonic(op: u8) -> String {
    const FLOAT_CC: [&str; 16] = [
        "f", "lt", "eq", "le", "gt", "lg", "ge", "o", "u", "nge", "nlg", "ngt", "nle", "neq",
        "nlt", "tru",
    ];
    const INT_CC: [&str; 8] = ["f", "lt", "eq", "le", "gt", "ne", "ge", "t"];

    match op {
        0x00..=0x0F => format!("v_cmp_{}_f32", FLOAT_CC[(op & 0xF) as usize]),
        0x10..=0x1F => format!("v_cmpx_{}_f32", FLOAT_CC[(op & 0xF) as usize]),
        0x20..=0x2F => format!("v_cmp_{}_f64", FLOAT_CC[(op & 0xF) as usize]),
        0x30..=0x3F => format!("v_cmpx_{}_f64", FLOAT_CC[(op & 0xF) as usize]),
        0x40..=0x4F => format!("v_cmp_{}_f16", FLOAT_CC[(op & 0xF) as usize]),
        0x50..=0x5F => format!("v_cmpx_{}_f16", FLOAT_CC[(op & 0xF) as usize]),
        0x60..=0x67 => format!("v_cmp_{}_i16", INT_CC[(op & 0x7) as usize]),
        0x68..=0x6F => format!("v_cmp_{}_u16", INT_CC[(op & 0x7) as usize]),
        0x70..=0x77 => format!("v_cmpx_{}_i16", INT_CC[(op & 0x7) as usize]),
        0x78..=0x7F => format!("v_cmpx_{}_u16", INT_CC[(op & 0x7) as usize]),
        0x80..=0x87 => format!("v_cmp_{}_i32", INT_CC[(op & 0x7) as usize]),
        0x88 => "v_cmp_class_f32".to_string(),
        0x8F => "v_cmp_class_f16".to_string(),
        0x90..=0x97 => format!("v_cmpx_{}_i32", INT_CC[(op & 0x7) as usize]),
        0x98 => "v_cmpx_class_f32".to_string(),
        0x9F => "v_cmpx_class_f16".to_string(),
        0xA0..=0xA7 => format!("v_cmp_{}_i64", INT_CC[(op & 0x7) as usize]),
        0xA8 => "v_cmp_class_f64".to_string(),
        0xB0..=0xB7 => format!("v_cmpx_{}_i64", INT_CC[(op & 0x7) as usize]),
        0xB8 => "v_cmpx_class_f64".to_string(),
        0xC0..=0xC7 => format!("v_cmp_{}_u32", INT_CC[(op & 0x7) as usize]),
        0xD0..=0xD7 => format!("v_cmpx_{}_u32", INT_CC[(op & 0x7) as usize]),
        0xE0..=0xE7 => format!("v_cmp_{}_u64", INT_CC[(op & 0x7) as usize]),
        0xF0..=0xF7 => format!("v_cmpx_{}_u64", INT_CC[(op & 0x7) as usize]),
        other => format!("vopc_op_{other:#04x}"),
    }
}

fn vop2_info(op: u8) -> (String, bool) {
    // Returns (mnemonic, always_has_literal_word)
    match op {
        0x00 | 0x01 => ("v_cndmask_b32".to_string(), false),
        0x02 => ("v_dot2c_f32_f16".to_string(), false),
        0x03 => ("v_add_f32".to_string(), false),
        0x04 => ("v_sub_f32".to_string(), false),
        0x05 => ("v_subrev_f32".to_string(), false),
        0x06 => ("v_fmac_legacy_f32".to_string(), false),
        0x07 => ("v_mul_legacy_f32".to_string(), false),
        0x08 => ("v_mul_f32".to_string(), false),
        0x09 => ("v_mul_i32_i24".to_string(), false),
        0x0A => ("v_mul_hi_i32_i24".to_string(), false),
        0x0B => ("v_mul_u32_u24".to_string(), false),
        0x0C => ("v_mul_hi_u32_u24".to_string(), false),
        0x0D => ("v_dot4c_i32_i8".to_string(), false),
        0x0E => ("v_dot8c_i32_i4".to_string(), false),
        0x0F => ("v_min_f32".to_string(), false),
        0x10 => ("v_max_f32".to_string(), false),
        0x11 => ("v_min_i32".to_string(), false),
        0x12 => ("v_max_i32".to_string(), false),
        0x13 => ("v_min_u32".to_string(), false),
        0x14 => ("v_max_u32".to_string(), false),
        0x16 => ("v_lshrrev_b32".to_string(), false),
        0x17 => ("v_ashrrev_i32".to_string(), false),
        0x18 => ("v_lshlrev_b32".to_string(), false),
        0x19 => ("v_and_b32".to_string(), false),
        0x1A => ("v_or_b32".to_string(), false),
        0x1B => ("v_xor_b32".to_string(), false),
        0x1C => ("v_xnor_b32".to_string(), false),
        0x20 => ("v_add_co_ci_u32".to_string(), false),
        0x21 => ("v_sub_co_ci_u32".to_string(), false),
        0x22 => ("v_subrev_co_ci_u32".to_string(), false),
        0x25 => ("v_add_nc_u32".to_string(), false),
        0x26 => ("v_sub_nc_u32".to_string(), false),
        0x27 => ("v_subrev_nc_u32".to_string(), false),
        0x2B => ("v_fmac_f32".to_string(), false),
        0x2C => ("v_fmamk_f32".to_string(), true),
        0x2D => ("v_fmaak_f32".to_string(), true),
        0x2E => ("v_cvt_pkrtz_f16_f32".to_string(), false),
        0x32 => ("v_add_f16".to_string(), false),
        0x33 => ("v_sub_f16".to_string(), false),
        0x34 => ("v_subrev_f16".to_string(), false),
        0x35 => ("v_mul_f16".to_string(), false),
        0x36 => ("v_fmac_f16".to_string(), false),
        0x37 => ("v_fmamk_f16".to_string(), true),
        0x38 => ("v_fmaak_f16".to_string(), true),
        0x39 => ("v_max_f16".to_string(), false),
        0x3A => ("v_min_f16".to_string(), false),
        0x3B => ("v_ldexp_f16".to_string(), false),
        0x3C => ("v_pk_fmac_f16".to_string(), false),
        other => (format!("vop2_op_{other:#04x}"), false),
    }
}

fn vop3_info(op: u16) -> (String, u8, bool) {
    // Returns (mnemonic, num_srcs, is_vop3b)
    match op {
        0x000..=0x0FF => (vopc_mnemonic(op as u8), 2, false),
        0x100..=0x13F => {
            let v2_op = (op - 0x100) as u8;
            let (name, _) = vop2_info(v2_op);
            let srcs = if matches!(v2_op, 0x00 | 0x01) { 3 } else { 2 };
            let is_b = matches!(v2_op, 0x20..=0x22);
            (name, srcs, is_b)
        }
        0x140 => ("v_mad_legacy_f32".to_string(), 3, false),
        0x141 => ("v_mad_f32".to_string(), 3, false),
        0x142 => ("v_mad_i32_i24".to_string(), 3, false),
        0x143 => ("v_mad_u32_u24".to_string(), 3, false),
        0x144 => ("v_cubeid_f32".to_string(), 3, false),
        0x145 => ("v_cubesc_f32".to_string(), 3, false),
        0x146 => ("v_cubetc_f32".to_string(), 3, false),
        0x147 => ("v_cubema_f32".to_string(), 3, false),
        0x148 => ("v_bfe_u32".to_string(), 3, false),
        0x149 => ("v_bfe_i32".to_string(), 3, false),
        0x14A => ("v_bfi_b32".to_string(), 3, false),
        0x14B => ("v_fma_f32".to_string(), 3, false),
        0x14C => ("v_fma_f64".to_string(), 3, false),
        0x14D => ("v_lerp_u8".to_string(), 3, false),
        0x14E => ("v_alignbit_b32".to_string(), 3, false),
        0x14F => ("v_alignbyte_b32".to_string(), 3, false),
        0x150 => ("v_mullit_f32".to_string(), 3, false),
        0x151 => ("v_min3_f32".to_string(), 3, false),
        0x152 => ("v_min3_i32".to_string(), 3, false),
        0x153 => ("v_min3_u32".to_string(), 3, false),
        0x154 => ("v_max3_f32".to_string(), 3, false),
        0x155 => ("v_max3_i32".to_string(), 3, false),
        0x156 => ("v_max3_u32".to_string(), 3, false),
        0x157 => ("v_med3_f32".to_string(), 3, false),
        0x158 => ("v_med3_i32".to_string(), 3, false),
        0x159 => ("v_med3_u32".to_string(), 3, false),
        0x15A => ("v_sad_u8".to_string(), 3, false),
        0x15B => ("v_sad_hi_u8".to_string(), 3, false),
        0x15C => ("v_sad_u16".to_string(), 3, false),
        0x15D => ("v_sad_u32".to_string(), 3, false),
        0x15E => ("v_cvt_pk_u8_f32".to_string(), 3, false),
        0x15F => ("v_div_fixup_f32".to_string(), 3, false),
        0x160 => ("v_div_fixup_f64".to_string(), 3, false),
        0x163 => ("v_div_fmas_f32".to_string(), 3, false),
        0x164 => ("v_div_fmas_f64".to_string(), 3, false),
        0x165 => ("v_msad_u8".to_string(), 3, false),
        0x169 => ("v_mul_lo_u32".to_string(), 2, false),
        0x16A => ("v_mul_hi_u32".to_string(), 2, false),
        0x16C => ("v_mul_hi_i32".to_string(), 2, false),
        0x16D => ("v_div_scale_f32".to_string(), 3, true),
        0x16E => ("v_div_scale_f64".to_string(), 3, true),
        0x170 => ("v_mad_u64_u32".to_string(), 3, true),
        0x171 => ("v_mad_i64_i32".to_string(), 3, true),
        0x180..=0x1FF => (vop1_mnemonic((op - 0x180) as u8), 1, false),
        0x2FF => ("v_lshlrev_b64".to_string(), 2, false),
        0x300 => ("v_lshrrev_b64".to_string(), 2, false),
        0x301 => ("v_ashrrev_i64".to_string(), 2, false),
        0x30F => ("v_add_co_u32".to_string(), 2, true),
        0x310 => ("v_sub_co_u32".to_string(), 2, true),
        0x319 => ("v_subrev_co_u32".to_string(), 2, true),
        0x340 => ("v_mad_u16".to_string(), 3, false),
        0x344 => ("v_perm_b32".to_string(), 3, false),
        0x345 => ("v_xad_u32".to_string(), 3, false),
        0x346 => ("v_lshl_add_u32".to_string(), 3, false),
        0x347 => ("v_add_lshl_u32".to_string(), 3, false),
        0x34B => ("v_fma_f16".to_string(), 3, false),
        0x35E => ("v_mad_i16".to_string(), 3, false),
        0x360 => ("v_readlane_b32".to_string(), 2, false),
        0x361 => ("v_writelane_b32".to_string(), 2, false),
        0x364 => ("v_add_f64".to_string(), 2, false),
        0x365 => ("v_mul_f64".to_string(), 2, false),
        0x366 => ("v_min_f64".to_string(), 2, false),
        0x367 => ("v_max_f64".to_string(), 2, false),
        0x368 => ("v_ldexp_f64".to_string(), 2, false),
        0x36D => ("v_add3_u32".to_string(), 3, false),
        0x36E => ("v_lshl_or_b32".to_string(), 3, false),
        0x36F => ("v_and_or_b32".to_string(), 3, false),
        0x370 => ("v_or3_b32".to_string(), 3, false),
        other => (format!("vop3_op_{other:#05x}"), 3, false),
    }
}

fn vop3p_num_srcs(op: u8) -> u8 {
    match op {
        0x01..=0x08 | 0x0A..=0x0D | 0x0F..=0x12 => 2,
        _ => 3,
    }
}

fn vop3p_mnemonic(op: u8) -> String {
    match op {
        0x00 => "v_pk_mad_i16",
        0x01 => "v_pk_mul_lo_u16",
        0x02 => "v_pk_add_i16",
        0x03 => "v_pk_sub_i16",
        0x04 => "v_pk_lshlrev_b16",
        0x05 => "v_pk_lshrrev_b16",
        0x06 => "v_pk_ashrrev_i16",
        0x07 => "v_pk_max_i16",
        0x08 => "v_pk_min_i16",
        0x09 => "v_pk_mad_u16",
        0x0A => "v_pk_add_u16",
        0x0B => "v_pk_sub_u16",
        0x0C => "v_pk_max_u16",
        0x0D => "v_pk_min_u16",
        0x0E => "v_pk_fma_f16",
        0x0F => "v_pk_add_f16",
        0x10 => "v_pk_mul_f16",
        0x11 => "v_pk_min_f16",
        0x12 => "v_pk_max_f16",
        0x13 => "v_dot2_f32_f16",
        0x14 => "v_dot2_i32_i16",
        0x15 => "v_dot2_u32_u16",
        0x16 => "v_dot4_i32_i8",
        0x17 => "v_dot4_u32_u8",
        0x18 => "v_dot8_i32_i4",
        0x19 => "v_dot8_u32_u4",
        0x20 => "v_fma_mix_f32",
        0x21 => "v_fma_mixlo_f16",
        0x22 => "v_fma_mixhi_f16",
        other => return format!("vop3p_op_{other:#04x}"),
    }
    .to_string()
}

fn mubuf_mnemonic(op: u8) -> String {
    match op {
        0x00 => "buffer_load_format_x",
        0x01 => "buffer_load_format_xy",
        0x02 => "buffer_load_format_xyz",
        0x03 => "buffer_load_format_xyzw",
        0x04 => "buffer_store_format_x",
        0x05 => "buffer_store_format_xy",
        0x06 => "buffer_store_format_xyz",
        0x07 => "buffer_store_format_xyzw",
        0x08 => "buffer_load_ubyte",
        0x09 => "buffer_load_sbyte",
        0x0A => "buffer_load_ushort",
        0x0B => "buffer_load_sshort",
        0x0C => "buffer_load_dword",
        0x0D => "buffer_load_dwordx2",
        0x0E => "buffer_load_dwordx4",
        0x0F => "buffer_load_dwordx3",
        0x18 => "buffer_store_byte",
        0x1A => "buffer_store_short",
        0x1C => "buffer_store_dword",
        0x1D => "buffer_store_dwordx2",
        0x1E => "buffer_store_dwordx4",
        0x1F => "buffer_store_dwordx3",
        0x32 => "buffer_atomic_swap",
        0x33 => "buffer_atomic_cmpswap",
        0x34 => "buffer_atomic_add",
        0x35 => "buffer_atomic_sub",
        0x38 => "buffer_atomic_and",
        0x39 => "buffer_atomic_or",
        0x3A => "buffer_atomic_xor",
        0x70 => "buffer_gl0_inv",
        0x71 => "buffer_gl1_inv",
        other => return format!("mubuf_op_{other:#04x}"),
    }
    .to_string()
}

fn mtbuf_mnemonic(op: u8) -> String {
    match op {
        0x00 => "tbuffer_load_format_x",
        0x01 => "tbuffer_load_format_xy",
        0x02 => "tbuffer_load_format_xyz",
        0x03 => "tbuffer_load_format_xyzw",
        0x04 => "tbuffer_store_format_x",
        0x05 => "tbuffer_store_format_xy",
        0x06 => "tbuffer_store_format_xyz",
        0x07 => "tbuffer_store_format_xyzw",
        0x08 => "tbuffer_load_format_d16_x",
        0x09 => "tbuffer_load_format_d16_xy",
        0x0A => "tbuffer_load_format_d16_xyz",
        0x0B => "tbuffer_load_format_d16_xyzw",
        0x0C => "tbuffer_store_format_d16_x",
        0x0D => "tbuffer_store_format_d16_xy",
        0x0E => "tbuffer_store_format_d16_xyz",
        0x0F => "tbuffer_store_format_d16_xyzw",
        other => return format!("mtbuf_op_{other:#04x}"),
    }
    .to_string()
}

fn mimg_info(op: u8) -> (String, bool) {
    // Returns (mnemonic, uses_sampler)
    let uses_sampler = matches!(op, 0x20..=0x6F);
    let name = match op {
        0x00 => "image_load",
        0x01 => "image_load_mip",
        0x02 => "image_load_pck",
        0x03 => "image_load_pck_sgn",
        0x04 => "image_load_mip_pck",
        0x05 => "image_load_mip_pck_sgn",
        0x08 => "image_store",
        0x09 => "image_store_mip",
        0x0A => "image_store_pck",
        0x0B => "image_store_mip_pck",
        0x0E => "image_get_resinfo",
        0x10 => "image_atomic_swap",
        0x11 => "image_atomic_cmpswap",
        0x12 => "image_atomic_add",
        0x13 => "image_atomic_sub",
        0x20 => "image_sample",
        0x21 => "image_sample_cl",
        0x22 => "image_sample_d",
        0x23 => "image_sample_d_cl",
        0x24 => "image_sample_l",
        0x25 => "image_sample_b",
        0x26 => "image_sample_b_cl",
        0x27 => "image_sample_lz",
        0x28 => "image_sample_c",
        0x29 => "image_sample_c_cl",
        0x2A => "image_sample_c_d",
        0x2B => "image_sample_c_d_cl",
        0x2C => "image_sample_c_l",
        0x2D => "image_sample_c_b",
        0x2E => "image_sample_c_b_cl",
        0x2F => "image_sample_c_lz",
        0x30 => "image_sample_o",
        0x34 => "image_sample_l_o",
        0x35 => "image_sample_b_o",
        0x37 => "image_sample_lz_o",
        0x38 => "image_sample_c_o",
        0x3F => "image_sample_c_lz_o",
        0x40 => "image_gather4",
        0x44 => "image_gather4_l",
        0x47 => "image_gather4_lz",
        0x48 => "image_gather4_c",
        0x4F => "image_gather4_c_lz",
        0x60 => "image_get_lod",
        0x68 => "image_sample_cd",
        0xE0 => "image_bvh_intersect_ray",
        0xE1 => "image_bvh64_intersect_ray",
        other => return (format!("mimg_op_{other:#04x}"), uses_sampler),
    };
    (name.to_string(), uses_sampler)
}

fn flat_mnemonic(seg: u8, op: u8) -> String {
    let prefix = match seg {
        0 => "flat",
        1 => "scratch",
        2 => "global",
        _ => "flat_unk",
    };
    let suffix = match op {
        0x08 => "load_ubyte",
        0x09 => "load_sbyte",
        0x0A => "load_ushort",
        0x0B => "load_sshort",
        0x0C => "load_dword",
        0x0D => "load_dwordx2",
        0x0E => "load_dwordx4",
        0x0F => "load_dwordx3",
        0x18 => "store_byte",
        0x1A => "store_short",
        0x1C => "store_dword",
        0x1D => "store_dwordx2",
        0x1E => "store_dwordx4",
        0x1F => "store_dwordx3",
        0x30 => "atomic_swap",
        0x31 => "atomic_cmpswap",
        0x32 => "atomic_add",
        0x33 => "atomic_sub",
        other => return format!("{prefix}_op_{other:#04x}"),
    };
    format!("{prefix}_{suffix}")
}

fn ds_mnemonic(op: u8) -> String {
    match op {
        0x0D => "ds_write_b32",
        0x0E => "ds_write2_b32",
        0x0F => "ds_write2st64_b32",
        0x1E => "ds_write_b8",
        0x1F => "ds_write_b16",
        0x36 => "ds_read_b32",
        0x37 => "ds_read2_b32",
        0x38 => "ds_read2st64_b32",
        0x39 => "ds_read_i8",
        0x3A => "ds_read_u8",
        0x3B => "ds_read_i16",
        0x3C => "ds_read_u16",
        0x3D => "ds_swizzle_b32",
        0x4D => "ds_write_b64",
        0x76 => "ds_read_b64",
        0xB5 => "ds_bpermute_b32",
        0xB6 => "ds_permute_b32",
        0xDE => "ds_write_b96",
        0xDF => "ds_write_b128",
        0xFE => "ds_read_b96",
        0xFF => "ds_read_b128",
        other => return format!("ds_op_{other:#04x}"),
    }
    .to_string()
}

// ---------------------------------------------------------------------------
// Core single-instruction and stream decoders
// ---------------------------------------------------------------------------

/// Decode a single RDNA2 (GFX10.3) instruction starting at byte `offset`.
///
/// Returns `None` if fewer than 4 bytes remain at `offset`.
/// Never panics on arbitrary, truncated, or misaligned byte slices.
pub fn decode_instruction(bytes: &[u8], offset: usize) -> Option<Rdna2Instruction> {
    let w0 = read_u32_le(bytes, offset)?;

    // 1. Bit 31 == 0: VOP1, VOPC, VOP2 (check 7-bit VOP1/VOPC before 1-bit VOP2)
    if (w0 >> 31) == 0 {
        let top7 = w0 >> 25;
        if top7 == 0b0111_111 {
            return Some(decode_vop1(bytes, offset, w0));
        }
        if top7 == 0b0111_110 {
            return Some(decode_vopc(bytes, offset, w0));
        }
        return Some(decode_vop2(bytes, offset, w0));
    }

    // 2. Bits [31:30] == 0b10: SOPP, SOPC, SOP1, SOPK, SOP2
    if (w0 >> 30) == 0b10 {
        let top9 = w0 >> 23;
        if top9 == 0b1011_1111_1 {
            return Some(decode_sopp(offset, w0));
        }
        if top9 == 0b1011_1111_0 {
            return Some(decode_sopc(bytes, offset, w0));
        }
        if top9 == 0b1011_1110_1 {
            return Some(decode_sop1(bytes, offset, w0));
        }
        if (w0 >> 28) == 0b1011 {
            return Some(decode_sopk(bytes, offset, w0));
        }
        return Some(decode_sop2(bytes, offset, w0));
    }

    // 3. Bits [31:30] == 0b11: 6-bit prefixed encodings
    let prefix6 = (w0 >> 26) as u8;
    match prefix6 {
        0b110010 => Some(decode_vintrp(offset, w0)),
        0b110011 => Some(decode_vop3p(bytes, offset, w0)),
        0b110101 => Some(decode_vop3(bytes, offset, w0)),
        0b110110 => Some(decode_ds(bytes, offset, w0)),
        0b110111 => Some(decode_flat(bytes, offset, w0)),
        0b111000 => Some(decode_mubuf(bytes, offset, w0)),
        0b111010 => Some(decode_mtbuf(bytes, offset, w0)),
        0b111100 => Some(decode_mimg(bytes, offset, w0)),
        0b111101 => Some(decode_smem(bytes, offset, w0)),
        0b111110 => Some(decode_exp(bytes, offset, w0)),
        other => Some(Rdna2Instruction {
            offset,
            size: 4,
            encoding: Rdna2Encoding::Unknown,
            mnemonic: format!("unknown_enc_{other:#04x}"),
            operands: String::new(),
            truncated: false,
            detail: Rdna2Detail::Unknown { prefix: other },
        }),
    }
}

fn decode_sopp(offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 16) & 0x7F) as u8;
    let simm16 = (w0 & 0xFFFF) as u16;
    let (mnemonic, is_branch, _) = sopp_mnemonic(op);
    let branch_target = if is_branch {
        Some(compute_branch_target(offset, simm16))
    } else {
        None
    };

    let operands = if let Some(target) = branch_target {
        let rel = simm16 as i16;
        if target >= 0 {
            format!("0x{target:04x} ({rel:+})")
        } else {
            format!("{target} ({rel:+})")
        }
    } else if op == 0x0C {
        format_waitcnt(simm16)
    } else if matches!(op, 0x01 | 0x03 | 0x0A | 0x13 | 0x1B | 0x1C | 0x1E | 0x1F | 0x22) {
        String::new()
    } else {
        format!("0x{simm16:04x}")
    };

    Rdna2Instruction {
        offset,
        size: 4,
        encoding: Rdna2Encoding::Sopp,
        mnemonic,
        operands,
        truncated: false,
        detail: Rdna2Detail::Sopp {
            op,
            simm16,
            branch_target,
        },
    }
}

fn decode_sopc(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 16) & 0x7F) as u8;
    let ssrc1 = ((w0 >> 8) & 0xFF) as u8;
    let ssrc0 = (w0 & 0xFF) as u8;
    let needs_lit = ssrc0 == 0xFF || ssrc1 == 0xFF;
    let (literal, size, truncated) = if needs_lit {
        match read_u32_le(bytes, offset.saturating_add(4)) {
            Some(lit) => (Some(lit), 8, false),
            None => (None, 4, true),
        }
    } else {
        (None, 4, false)
    };

    let mnemonic = sopc_mnemonic(op);
    let operands = format!(
        "{}, {}",
        format_src_operand(u16::from(ssrc0), literal),
        format_src_operand(u16::from(ssrc1), literal)
    );

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Sopc,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Sopc {
            op,
            ssrc0,
            ssrc1,
            literal,
        },
    }
}

fn decode_sop1(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let sdst = ((w0 >> 16) & 0x7F) as u8;
    let op = ((w0 >> 8) & 0xFF) as u8;
    let ssrc0 = (w0 & 0xFF) as u8;
    let needs_lit = ssrc0 == 0xFF;
    let (literal, size, truncated) = if needs_lit {
        match read_u32_le(bytes, offset.saturating_add(4)) {
            Some(lit) => (Some(lit), 8, false),
            None => (None, 4, true),
        }
    } else {
        (None, 4, false)
    };

    let (mnemonic, is_64) = sop1_mnemonic(op);
    let dst_str = if is_64 {
        format_sdst_pair(sdst)
    } else {
        format_sdst(sdst)
    };
    let operands = if op == 0x1F {
        // s_getpc_b64 sdst
        dst_str
    } else if op == 0x20 || op == 0x22 {
        // s_setpc_b64 ssrc0 / s_rfe_b64 ssrc0
        format_src_operand(u16::from(ssrc0), literal)
    } else {
        format!("{dst_str}, {}", format_src_operand(u16::from(ssrc0), literal))
    };

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Sop1,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Sop1 {
            op,
            sdst,
            ssrc0,
            literal,
        },
    }
}

fn decode_sopk(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 23) & 0x1F) as u8;
    let sdst = ((w0 >> 16) & 0x7F) as u8;
    let simm16 = (w0 & 0xFFFF) as u16;
    let (mnemonic, has_imm32, is_branch) = sopk_mnemonic(op);

    let (literal, size, truncated) = if has_imm32 {
        match read_u32_le(bytes, offset.saturating_add(4)) {
            Some(lit) => (Some(lit), 8, false),
            None => (None, 4, true),
        }
    } else {
        (None, 4, false)
    };

    let branch_target = if is_branch {
        Some(compute_branch_target(offset, simm16))
    } else {
        None
    };

    let dst_str = if op == 0x16 {
        format_sdst_pair(sdst)
    } else {
        format_sdst(sdst)
    };

    let operands = if let Some(lit) = literal {
        format!("hwreg(0x{simm16:04x}), 0x{lit:08x}")
    } else if let Some(target) = branch_target {
        let rel = simm16 as i16;
        if target >= 0 {
            format!("{dst_str}, 0x{target:04x} ({rel:+})")
        } else {
            format!("{dst_str}, {target} ({rel:+})")
        }
    } else if matches!(op, 0x17..=0x1A) {
        format!("{dst_str}, 0x{simm16:04x}")
    } else {
        format!("{dst_str}, {}", simm16 as i16)
    };

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Sopk,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Sopk {
            op,
            sdst,
            simm16,
            literal,
            branch_target,
        },
    }
}

fn decode_sop2(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 23) & 0x7F) as u8;
    let sdst = ((w0 >> 16) & 0x7F) as u8;
    let ssrc1 = ((w0 >> 8) & 0xFF) as u8;
    let ssrc0 = (w0 & 0xFF) as u8;
    let needs_lit = ssrc0 == 0xFF || ssrc1 == 0xFF;
    let (literal, size, truncated) = if needs_lit {
        match read_u32_le(bytes, offset.saturating_add(4)) {
            Some(lit) => (Some(lit), 8, false),
            None => (None, 4, true),
        }
    } else {
        (None, 4, false)
    };

    let (mnemonic, is_64) = sop2_mnemonic(op);
    let dst_str = if is_64 {
        format_sdst_pair(sdst)
    } else {
        format_sdst(sdst)
    };
    let operands = format!(
        "{dst_str}, {}, {}",
        format_src_operand(u16::from(ssrc0), literal),
        format_src_operand(u16::from(ssrc1), literal)
    );

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Sop2,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Sop2 {
            op,
            sdst,
            ssrc0,
            ssrc1,
            literal,
        },
    }
}

fn decode_smem(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 18) & 0xFF) as u8;
    let glc = ((w0 >> 16) & 0x1) != 0;
    let dlc = ((w0 >> 14) & 0x1) != 0;
    let sdata = ((w0 >> 6) & 0x7F) as u8;
    let sbase = (w0 & 0x3F) as u8;
    let (mnemonic, sdata_count, is_buffer) = smem_info(op);

    let (w1, size, truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    let soffset = ((w1 >> 25) & 0x7F) as u8;
    let byte_offset = sign_extend_21(w1 & 0x001F_FFFF);
    let base_reg = sbase.saturating_mul(2);
    let base_count = if is_buffer { 4 } else { 2 };

    let operands = if sdata_count == 0 {
        String::new()
    } else {
        let mut s = format!(
            "{}, {}",
            format_sgpr_range(sdata, sdata_count),
            format_sgpr_range(base_reg, base_count)
        );
        if soffset != 0x7D && !truncated {
            let _ = write!(s, ", {}", format_sdst(soffset));
        }
        if byte_offset != 0 || soffset == 0x7D {
            let _ = write!(s, ", {byte_offset:#x}");
        }
        if glc {
            s.push_str(" glc");
        }
        if dlc {
            s.push_str(" dlc");
        }
        s
    };

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Smem,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Smem {
            op,
            sdata,
            sbase,
            soffset,
            offset: byte_offset,
            glc,
            dlc,
        },
    }
}

fn decode_vop1(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let vdst = ((w0 >> 17) & 0xFF) as u8;
    let op = ((w0 >> 9) & 0xFF) as u8;
    let src0 = (w0 & 0x1FF) as u16;
    let needs_extra = src_uses_extra_word(src0);
    let (literal, size, truncated) = if needs_extra {
        match read_u32_le(bytes, offset.saturating_add(4)) {
            Some(lit) => (Some(lit), 8, false),
            None => (None, 4, true),
        }
    } else {
        (None, 4, false)
    };

    let mnemonic = vop1_mnemonic(op);
    let operands = if op == 0x00 || op == 0x1B || op == 0x41 {
        String::new()
    } else if op == 0x02 {
        // v_readfirstlane_b32 sdst, vsrc0
        format!("{}, {}", format_sdst(vdst & 0x7F), format_src_operand(src0, literal))
    } else {
        format!("v{vdst}, {}", format_src_operand(src0, literal))
    };

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Vop1,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Vop1 {
            op,
            vdst,
            src0,
            literal,
        },
    }
}

fn decode_vopc(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 17) & 0xFF) as u8;
    let vsrc1 = ((w0 >> 9) & 0xFF) as u8;
    let src0 = (w0 & 0x1FF) as u16;
    let needs_extra = src_uses_extra_word(src0);
    let (literal, size, truncated) = if needs_extra {
        match read_u32_le(bytes, offset.saturating_add(4)) {
            Some(lit) => (Some(lit), 8, false),
            None => (None, 4, true),
        }
    } else {
        (None, 4, false)
    };

    let mnemonic = vopc_mnemonic(op);
    let operands = format!("vcc, {}, v{vsrc1}", format_src_operand(src0, literal));

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Vopc,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Vopc {
            op,
            vsrc1,
            src0,
            literal,
        },
    }
}

fn decode_vop2(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 25) & 0x3F) as u8;
    let vdst = ((w0 >> 17) & 0xFF) as u8;
    let vsrc1 = ((w0 >> 9) & 0xFF) as u8;
    let src0 = (w0 & 0x1FF) as u16;
    let (mnemonic, always_lit) = vop2_info(op);
    let needs_extra = always_lit || src_uses_extra_word(src0);

    let (literal, size, truncated) = if needs_extra {
        match read_u32_le(bytes, offset.saturating_add(4)) {
            Some(lit) => (Some(lit), 8, false),
            None => (None, 4, true),
        }
    } else {
        (None, 4, false)
    };

    let operands = if always_lit {
        let lit_str = match literal {
            Some(lit) => format!("0x{lit:08x}"),
            None => "<lit>".to_string(),
        };
        if matches!(op, 0x2C | 0x37) {
            // v_fmamk_f32 / v_fmamk_f16: vdst, src0, K, vsrc1
            format!(
                "v{vdst}, {}, {lit_str}, v{vsrc1}",
                format_src_operand(src0, None)
            )
        } else {
            // v_fmaak_f32 / v_fmaak_f16: vdst, src0, vsrc1, K
            format!(
                "v{vdst}, {}, v{vsrc1}, {lit_str}",
                format_src_operand(src0, None)
            )
        }
    } else if matches!(op, 0x00 | 0x01) {
        // v_cndmask_b32 vdst, src0, vsrc1, vcc
        format!(
            "v{vdst}, {}, v{vsrc1}, vcc",
            format_src_operand(src0, literal)
        )
    } else {
        format!("v{vdst}, {}, v{vsrc1}", format_src_operand(src0, literal))
    };

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Vop2,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Vop2 {
            op,
            vdst,
            vsrc1,
            src0,
            literal,
        },
    }
}

fn decode_vop3(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 16) & 0x3FF) as u16;
    let clamp = ((w0 >> 15) & 0x1) != 0;
    let op_sel = ((w0 >> 11) & 0xF) as u8;
    let abs_or_sdst_raw = ((w0 >> 8) & 0x7F) as u8;
    let vdst = (w0 & 0xFF) as u8;

    let (mnemonic, num_srcs, is_vop3b) = vop3_info(op);
    let (abs, sdst) = if is_vop3b {
        (0u8, Some(abs_or_sdst_raw))
    } else {
        (abs_or_sdst_raw & 0x7, None)
    };

    let (w1, mut size, mut truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    let neg = ((w1 >> 29) & 0x7) as u8;
    let omod = ((w1 >> 27) & 0x3) as u8;
    let src2 = ((w1 >> 18) & 0x1FF) as u16;
    let src1 = ((w1 >> 9) & 0x1FF) as u16;
    let src0 = (w1 & 0x1FF) as u16;

    let uses_lit = !truncated
        && (src0 == 0xFF
            || (num_srcs >= 2 && src1 == 0xFF)
            || (num_srcs >= 3 && src2 == 0xFF));
    let literal = if uses_lit {
        match read_u32_le(bytes, offset.saturating_add(8)) {
            Some(lit) => {
                size = 12;
                Some(lit)
            }
            None => {
                truncated = true;
                None
            }
        }
    } else {
        None
    };

    let mut operands = if op <= 0xFF {
        // VOPC promoted to VOP3 writes SGPR mask in vdst[6:0]
        format_sdst(vdst & 0x7F)
    } else if op == 0x360 {
        // v_readlane_b32 sdst, vsrc0, ssrc1
        format_sdst(vdst & 0x7F)
    } else if let Some(sd) = sdst {
        format!("v{vdst}, {}", format_sdst(sd))
    } else {
        format!("v{vdst}")
    };

    if !truncated {
        let s0 = format_vop3_src(src0, literal, (abs & 1) != 0, (neg & 1) != 0);
        let _ = write!(operands, ", {s0}");
        if num_srcs >= 2 {
            let s1 = format_vop3_src(src1, literal, (abs & 2) != 0, (neg & 2) != 0);
            let _ = write!(operands, ", {s1}");
        }
        if num_srcs >= 3 {
            let s2 = format_vop3_src(src2, literal, (abs & 4) != 0, (neg & 4) != 0);
            let _ = write!(operands, ", {s2}");
        }
        if clamp {
            operands.push_str(" clamp");
        }
        match omod {
            1 => operands.push_str(" mul:2"),
            2 => operands.push_str(" mul:4"),
            3 => operands.push_str(" div:2"),
            _ => {}
        }
    }

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Vop3,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Vop3 {
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
            op_sel,
            literal,
        },
    }
}

fn decode_vop3p(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 16) & 0x7F) as u8;
    let clamp = ((w0 >> 15) & 0x1) != 0;
    let op_sel = ((w0 >> 11) & 0x7) as u8;
    let neg_hi = ((w0 >> 8) & 0x7) as u8;
    let vdst = (w0 & 0xFF) as u8;
    let mnemonic = vop3p_mnemonic(op);

    let (w1, mut size, mut truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    let neg_lo = ((w1 >> 29) & 0x7) as u8;
    let op_sel_hi = (((w1 >> 27) & 0x3) | (((w0 >> 14) & 0x1) << 2)) as u8;
    let src2 = ((w1 >> 18) & 0x1FF) as u16;
    let src1 = ((w1 >> 9) & 0x1FF) as u16;
    let src0 = (w1 & 0x1FF) as u16;

    let num_srcs = vop3p_num_srcs(op);
    let uses_lit = !truncated && (src0 == 0xFF || src1 == 0xFF || (num_srcs >= 3 && src2 == 0xFF));
    let literal = if uses_lit {
        match read_u32_le(bytes, offset.saturating_add(8)) {
            Some(lit) => {
                size = 12;
                Some(lit)
            }
            None => {
                truncated = true;
                None
            }
        }
    } else {
        None
    };

    let mut operands = format!("v{vdst}");
    if !truncated {
        let _ = write!(
            operands,
            ", {}, {}",
            format_src_operand(src0, literal),
            format_src_operand(src1, literal),
        );
        if num_srcs >= 3 {
            let _ = write!(operands, ", {}", format_src_operand(src2, literal));
        }
        let src_mask = if num_srcs >= 3 { 0x7u8 } else { 0x3u8 };
        let eff_op_sel = op_sel & src_mask;
        if eff_op_sel != 0 {
            if num_srcs >= 3 {
                let _ = write!(
                    operands,
                    " op_sel:[{},{},{}]",
                    eff_op_sel & 1,
                    (eff_op_sel >> 1) & 1,
                    (eff_op_sel >> 2) & 1
                );
            } else {
                let _ = write!(
                    operands,
                    " op_sel:[{},{}]",
                    eff_op_sel & 1,
                    (eff_op_sel >> 1) & 1
                );
            }
        }
        let eff_op_sel_hi = op_sel_hi & src_mask;
        let is_fma_mix = matches!(op, 0x20..=0x22);
        let default_hi = if is_fma_mix { 0 } else { src_mask };
        if eff_op_sel_hi != default_hi {
            if num_srcs >= 3 {
                let _ = write!(
                    operands,
                    " op_sel_hi:[{},{},{}]",
                    eff_op_sel_hi & 1,
                    (eff_op_sel_hi >> 1) & 1,
                    (eff_op_sel_hi >> 2) & 1
                );
            } else {
                let _ = write!(
                    operands,
                    " op_sel_hi:[{},{}]",
                    eff_op_sel_hi & 1,
                    (eff_op_sel_hi >> 1) & 1
                );
            }
        }
        let eff_neg_lo = neg_lo & src_mask;
        if eff_neg_lo != 0 {
            if num_srcs >= 3 {
                let _ = write!(
                    operands,
                    " neg_lo:[{},{},{}]",
                    eff_neg_lo & 1,
                    (eff_neg_lo >> 1) & 1,
                    (eff_neg_lo >> 2) & 1
                );
            } else {
                let _ = write!(
                    operands,
                    " neg_lo:[{},{}]",
                    eff_neg_lo & 1,
                    (eff_neg_lo >> 1) & 1
                );
            }
        }
        let eff_neg_hi = neg_hi & src_mask;
        if eff_neg_hi != 0 {
            if num_srcs >= 3 {
                let _ = write!(
                    operands,
                    " neg_hi:[{},{},{}]",
                    eff_neg_hi & 1,
                    (eff_neg_hi >> 1) & 1,
                    (eff_neg_hi >> 2) & 1
                );
            } else {
                let _ = write!(
                    operands,
                    " neg_hi:[{},{}]",
                    eff_neg_hi & 1,
                    (eff_neg_hi >> 1) & 1
                );
            }
        }
        if clamp {
            operands.push_str(" clamp");
        }
    }

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Vop3P,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Vop3P {
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
        },
    }
}

fn decode_vintrp(offset: usize, w0: u32) -> Rdna2Instruction {
    let vdst = ((w0 >> 18) & 0xFF) as u8;
    let op = ((w0 >> 16) & 0x3) as u8;
    let attr = ((w0 >> 10) & 0x3F) as u8;
    let attrchan = ((w0 >> 8) & 0x3) as u8;
    let vsrc = (w0 & 0xFF) as u8;

    let chan = match attrchan {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        _ => 'w',
    };
    let (mnemonic, operands) = match op {
        0 => (
            "v_interp_p1_f32".to_string(),
            format!("v{vdst}, v{vsrc}, attr{attr}.{chan}"),
        ),
        1 => (
            "v_interp_p2_f32".to_string(),
            format!("v{vdst}, v{vsrc}, attr{attr}.{chan}"),
        ),
        2 => {
            let param = match vsrc {
                0 => "p10".to_string(),
                1 => "p20".to_string(),
                2 => "p0".to_string(),
                other => format!("p{other}"),
            };
            (
                "v_interp_mov_f32".to_string(),
                format!("v{vdst}, {param}, attr{attr}.{chan}"),
            )
        }
        other => (
            format!("vintrp_op_{other}"),
            format!("v{vdst}, v{vsrc}, attr{attr}.{chan}"),
        ),
    };

    Rdna2Instruction {
        offset,
        size: 4,
        encoding: Rdna2Encoding::Vintrp,
        mnemonic,
        operands,
        truncated: false,
        detail: Rdna2Detail::Vintrp {
            op,
            vdst,
            attr,
            attrchan,
            vsrc,
        },
    }
}

fn decode_exp(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let vm = ((w0 >> 12) & 0x1) != 0;
    let done = ((w0 >> 11) & 0x1) != 0;
    let compr = ((w0 >> 10) & 0x1) != 0;
    let target_raw = ((w0 >> 4) & 0x3F) as u8;
    let en = (w0 & 0xF) as u8;
    let target = Rdna2ExportTarget::from_raw(target_raw);

    let (w1, size, truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    let vsrc = [
        (w1 & 0xFF) as u8,
        ((w1 >> 8) & 0xFF) as u8,
        ((w1 >> 16) & 0xFF) as u8,
        ((w1 >> 24) & 0xFF) as u8,
    ];

    let mut operands = target.name();
    if !truncated {
        if compr {
            let s0 = if (en & 0b0011) != 0 {
                format!("v{}", vsrc[0])
            } else {
                "off".to_string()
            };
            let s1 = if (en & 0b1100) != 0 {
                format!("v{}", vsrc[1])
            } else {
                "off".to_string()
            };
            let _ = write!(operands, " {s0}, {s1}, off, off");
        } else {
            for i in 0..4 {
                let sep = if i == 0 { " " } else { ", " };
                if (en & (1 << i)) != 0 {
                    let _ = write!(operands, "{sep}v{}", vsrc[i]);
                } else {
                    let _ = write!(operands, "{sep}off");
                }
            }
        }
        if done {
            operands.push_str(" done");
        }
        if compr {
            operands.push_str(" compr");
        }
        if vm {
            operands.push_str(" vm");
        }
    }

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Exp,
        mnemonic: "exp".to_string(),
        operands,
        truncated,
        detail: Rdna2Detail::Exp {
            target,
            en,
            compr,
            done,
            vm,
            vsrc,
        },
    }
}

fn decode_mubuf(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 18) & 0x7F) as u8;
    let lds = ((w0 >> 16) & 0x1) != 0;
    let dlc = ((w0 >> 15) & 0x1) != 0;
    let glc = ((w0 >> 14) & 0x1) != 0;
    let idxen = ((w0 >> 13) & 0x1) != 0;
    let offen = ((w0 >> 12) & 0x1) != 0;
    let inst_offset = (w0 & 0xFFF) as u16;
    let mnemonic = mubuf_mnemonic(op);

    let (w1, size, truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    let soffset = ((w1 >> 24) & 0xFF) as u8;
    let tfe = ((w1 >> 23) & 0x1) != 0;
    let slc = ((w1 >> 22) & 0x1) != 0;
    let srsrc = ((w1 >> 16) & 0x1F) as u8;
    let vdata = ((w1 >> 8) & 0xFF) as u8;
    let vaddr = (w1 & 0xFF) as u8;

    let mut operands = String::new();
    if !truncated {
        let srsrc_reg = srsrc.saturating_mul(4);
        let addr_str = if offen || idxen {
            format!("v{vaddr}")
        } else {
            "off".to_string()
        };
        let _ = write!(
            operands,
            "v{vdata}, {addr_str}, {}, {}",
            format_sgpr_range(srsrc_reg, 4),
            format_src_operand(u16::from(soffset), None)
        );
        if offen {
            operands.push_str(" offen");
        }
        if idxen {
            operands.push_str(" idxen");
        }
        if inst_offset != 0 {
            let _ = write!(operands, " offset:{inst_offset}");
        }
        if glc {
            operands.push_str(" glc");
        }
        if slc {
            operands.push_str(" slc");
        }
        if dlc {
            operands.push_str(" dlc");
        }
        if lds {
            operands.push_str(" lds");
        }
        if tfe {
            operands.push_str(" tfe");
        }
    }

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Mubuf,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Mubuf {
            op,
            offset: inst_offset,
            offen,
            idxen,
            glc,
            dlc,
            slc,
            lds,
            tfe,
            vaddr,
            vdata,
            srsrc,
            soffset,
        },
    }
}

fn decode_mtbuf(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let format = ((w0 >> 19) & 0x7F) as u8;
    let op_lo = ((w0 >> 16) & 0x7) as u8;
    let dlc = ((w0 >> 15) & 0x1) != 0;
    let glc = ((w0 >> 14) & 0x1) != 0;
    let idxen = ((w0 >> 13) & 0x1) != 0;
    let offen = ((w0 >> 12) & 0x1) != 0;
    let inst_offset = (w0 & 0xFFF) as u16;

    let (w1, size, truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    let soffset = ((w1 >> 24) & 0xFF) as u8;
    let tfe = ((w1 >> 23) & 0x1) != 0;
    let slc = ((w1 >> 22) & 0x1) != 0;
    let op_hi = ((w1 >> 21) & 0x1) as u8;
    let op = op_lo | (op_hi << 3);
    let srsrc = ((w1 >> 16) & 0x1F) as u8;
    let vdata = ((w1 >> 8) & 0xFF) as u8;
    let vaddr = (w1 & 0xFF) as u8;
    let mnemonic = mtbuf_mnemonic(op);

    let mut operands = String::new();
    if !truncated {
        let srsrc_reg = srsrc.saturating_mul(4);
        let addr_str = if offen || idxen {
            format!("v{vaddr}")
        } else {
            "off".to_string()
        };
        let _ = write!(
            operands,
            "v{vdata}, {addr_str}, {}, {} format:{format}",
            format_sgpr_range(srsrc_reg, 4),
            format_src_operand(u16::from(soffset), None)
        );
        if offen {
            operands.push_str(" offen");
        }
        if idxen {
            operands.push_str(" idxen");
        }
        if inst_offset != 0 {
            let _ = write!(operands, " offset:{inst_offset}");
        }
        if glc {
            operands.push_str(" glc");
        }
        if slc {
            operands.push_str(" slc");
        }
        if dlc {
            operands.push_str(" dlc");
        }
        if tfe {
            operands.push_str(" tfe");
        }
    }

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Mtbuf,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Mtbuf {
            op,
            format,
            offset: inst_offset,
            offen,
            idxen,
            glc,
            dlc,
            slc,
            tfe,
            vaddr,
            vdata,
            srsrc,
            soffset,
        },
    }
}

fn decode_mimg(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let slc = ((w0 >> 25) & 0x1) != 0;
    let op = ((((w0 >> 18) & 0x7F) | ((w0 & 0x1) << 7)) & 0xFF) as u8;
    let lwe = ((w0 >> 17) & 0x1) != 0;
    let tfe = ((w0 >> 16) & 0x1) != 0;
    let r128 = ((w0 >> 15) & 0x1) != 0;
    let glc = ((w0 >> 13) & 0x1) != 0;
    let unrm = ((w0 >> 12) & 0x1) != 0;
    let dmask = ((w0 >> 8) & 0xF) as u8;
    let dlc = ((w0 >> 7) & 0x1) != 0;
    let dim = ((w0 >> 3) & 0x7) as u8;
    let nsize = ((w0 >> 1) & 0x3) as u8;

    let (mnemonic, uses_sampler) = mimg_info(op);

    let (w1, mut size, mut truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    // Consume GFX10 NSA extra address words if present (nsize in 1..=3).
    if !truncated && nsize > 0 {
        let extra_bytes = usize::from(nsize) * 4;
        if offset
            .checked_add(8)
            .and_then(|end| end.checked_add(extra_bytes))
            .is_some_and(|total_end| total_end <= bytes.len())
        {
            size += extra_bytes;
        } else {
            // Advance through any complete remaining words without going out of bounds.
            let avail_words = bytes.len().saturating_sub(offset.saturating_add(8)) / 4;
            size += avail_words.min(usize::from(nsize)) * 4;
            truncated = true;
        }
    }

    let d16 = ((w1 >> 31) & 0x1) != 0;
    let a16 = ((w1 >> 30) & 0x1) != 0;
    let ssamp = ((w1 >> 21) & 0x1F) as u8;
    let srsrc = ((w1 >> 16) & 0x1F) as u8;
    let vdata = ((w1 >> 8) & 0xFF) as u8;
    let vaddr = (w1 & 0xFF) as u8;

    let mut operands = String::new();
    if !truncated {
        let srsrc_reg = srsrc.saturating_mul(4);
        let srsrc_count = if r128 { 4 } else { 8 };
        let _ = write!(
            operands,
            "v{vdata}, v{vaddr}, {}",
            format_sgpr_range(srsrc_reg, srsrc_count)
        );
        if uses_sampler {
            let ssamp_reg = ssamp.saturating_mul(4);
            let _ = write!(operands, ", {}", format_sgpr_range(ssamp_reg, 4));
        }
        let _ = write!(operands, " dmask:{dmask:#x} dim:{dim}");
        if unrm {
            operands.push_str(" unorm");
        }
        if glc {
            operands.push_str(" glc");
        }
        if slc {
            operands.push_str(" slc");
        }
        if dlc {
            operands.push_str(" dlc");
        }
        if r128 {
            operands.push_str(" r128");
        }
        if a16 {
            operands.push_str(" a16");
        }
        if d16 {
            operands.push_str(" d16");
        }
        if lwe {
            operands.push_str(" lwe");
        }
        if tfe {
            operands.push_str(" tfe");
        }
    }

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Mimg,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Mimg {
            op,
            dmask,
            dim,
            unrm,
            glc,
            dlc,
            slc,
            r128,
            tfe,
            lwe,
            a16,
            d16,
            nsize,
            vaddr,
            vdata,
            srsrc,
            ssamp,
        },
    }
}

fn decode_flat(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 18) & 0x7F) as u8;
    let slc = ((w0 >> 17) & 0x1) != 0;
    let glc = ((w0 >> 16) & 0x1) != 0;
    let seg = ((w0 >> 14) & 0x3) as u8;
    let lds = ((w0 >> 13) & 0x1) != 0;
    let dlc = ((w0 >> 12) & 0x1) != 0;
    let raw_off = (w0 & 0xFFF) as u16;
    let inst_offset = if seg == 0 {
        raw_off as i16
    } else {
        sign_extend_12(raw_off)
    };
    let mnemonic = flat_mnemonic(seg, op);

    let (w1, size, truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    let vdst = ((w1 >> 24) & 0xFF) as u8;
    let saddr = ((w1 >> 16) & 0x7F) as u8;
    let data = ((w1 >> 8) & 0xFF) as u8;
    let addr = (w1 & 0xFF) as u8;

    let mut operands = String::new();
    if !truncated {
        let is_store = matches!(op, 0x18..=0x1F);
        if is_store {
            let _ = write!(operands, "v{addr}, v{data}");
        } else {
            let _ = write!(operands, "v{vdst}, v{addr}");
        }
        if seg != 0 {
            if saddr == 0x7F {
                operands.push_str(", off");
            } else {
                let _ = write!(operands, ", {}", format_sdst_pair(saddr));
            }
        }
        if inst_offset != 0 {
            let _ = write!(operands, " offset:{inst_offset}");
        }
        if glc {
            operands.push_str(" glc");
        }
        if slc {
            operands.push_str(" slc");
        }
        if dlc {
            operands.push_str(" dlc");
        }
        if lds {
            operands.push_str(" lds");
        }
    }

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Flat,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Flat {
            seg,
            op,
            offset: inst_offset,
            glc,
            slc,
            dlc,
            lds,
            addr,
            data,
            saddr,
            vdst,
        },
    }
}

fn decode_ds(bytes: &[u8], offset: usize, w0: u32) -> Rdna2Instruction {
    let op = ((w0 >> 18) & 0xFF) as u8;
    let gds = ((w0 >> 17) & 0x1) != 0;
    let offset1 = ((w0 >> 8) & 0xFF) as u8;
    let offset0 = (w0 & 0xFF) as u8;
    let mnemonic = ds_mnemonic(op);

    let (w1, size, truncated) = match read_u32_le(bytes, offset.saturating_add(4)) {
        Some(w1) => (w1, 8, false),
        None => (0, 4, true),
    };

    let vdst = ((w1 >> 24) & 0xFF) as u8;
    let data1 = ((w1 >> 16) & 0xFF) as u8;
    let data0 = ((w1 >> 8) & 0xFF) as u8;
    let addr = (w1 & 0xFF) as u8;

    let mut operands = String::new();
    if !truncated {
        if matches!(op, 0x0E | 0x0F) {
            let _ = write!(operands, "v{vdst}, v{addr}, v{data0}, v{data1}");
        } else {
            let _ = write!(operands, "v{vdst}, v{addr}, v{data0}");
        }
        if offset0 != 0 {
            let _ = write!(operands, " offset0:{offset0}");
        }
        if offset1 != 0 {
            let _ = write!(operands, " offset1:{offset1}");
        }
        if gds {
            operands.push_str(" gds");
        }
    }

    Rdna2Instruction {
        offset,
        size,
        encoding: Rdna2Encoding::Ds,
        mnemonic,
        operands,
        truncated,
        detail: Rdna2Detail::Ds {
            op,
            gds,
            offset0,
            offset1,
            addr,
            data0,
            data1,
            vdst,
        },
    }
}

/// Decode all RDNA2 instructions in `bytes` up to `max_instructions` (or the entire buffer).
pub fn decode_program_limited(
    bytes: &[u8],
    max_instructions: Option<usize>,
) -> Vec<Rdna2Instruction> {
    let mut out = Vec::new();
    let mut offset = 0usize;
    let limit = max_instructions.unwrap_or(usize::MAX);

    while out.len() < limit {
        let Some(inst) = decode_instruction(bytes, offset) else {
            break;
        };
        // Guarantee strict forward progress (never stalls even on malformed inputs).
        let step = inst.size.max(4);
        out.push(inst);
        match offset.checked_add(step) {
            Some(next) if next <= bytes.len() => offset = next,
            _ => break,
        }
    }

    out
}

/// Decode all RDNA2 instructions in `bytes`.
pub fn decode_program(bytes: &[u8]) -> Vec<Rdna2Instruction> {
    decode_program_limited(bytes, None)
}

// ---------------------------------------------------------------------------
// Register tracking & program analysis
// ---------------------------------------------------------------------------

fn record_sgpr(max_sgpr: &mut Option<u8>, reg: u8) {
    if reg <= 105 {
        *max_sgpr = Some(max_sgpr.map_or(reg, |cur| cur.max(reg)));
    }
}

fn record_sgpr_range(max_sgpr: &mut Option<u8>, base: u8, count: u8) {
    if base <= 105 && count > 0 {
        let hi = base.saturating_add(count - 1).min(105);
        record_sgpr(max_sgpr, hi);
    }
}

fn record_vgpr(max_vgpr: &mut Option<u8>, reg: u8) {
    *max_vgpr = Some(max_vgpr.map_or(reg, |cur| cur.max(reg)));
}

fn record_vgpr_range(max_vgpr: &mut Option<u8>, base: u8, count: u8) {
    if count > 0 {
        let hi = base.saturating_add(count - 1);
        record_vgpr(max_vgpr, hi);
    }
}

fn record_src_reg(max_sgpr: &mut Option<u8>, max_vgpr: &mut Option<u8>, src: u16) {
    if src <= 105 {
        record_sgpr(max_sgpr, src as u8);
    } else if (256..=511).contains(&src) {
        record_vgpr(max_vgpr, (src - 256) as u8);
    }
}

/// Analyze a complete RDNA2 machine-code program into a structured [`Rdna2ProgramSummary`].
///
/// Decodes every instruction cleanly to the end of `bytes`, tracks encoding
/// counts, `s_endpgm` positions, inferred pipeline stage, `EXP` targets,
/// `SMEM` descriptor base SGPR pairs, `MIMG`/`MUBUF`/`MTBUF` resource
/// references, control-flow branch targets, max SGPR/VGPR used, and the
/// deterministic FNV-1a 64-bit hash.
pub fn analyze_program(bytes: &[u8]) -> Rdna2ProgramSummary {
    let instructions = decode_program(bytes);
    analyze_instructions(bytes, &instructions)
}

/// Analyze a pre-decoded instruction list alongside the raw `bytes` (used to avoid
/// decoding twice when producing both disassembly and summary).
pub fn analyze_instructions(
    bytes: &[u8],
    instructions: &[Rdna2Instruction],
) -> Rdna2ProgramSummary {
    let mut encoding_counts: BTreeMap<Rdna2Encoding, usize> = BTreeMap::new();
    let mut exports = Vec::new();
    let mut smem_refs = Vec::new();
    let mut smem_base_set: BTreeSet<u8> = BTreeSet::new();
    let mut resource_refs = Vec::new();
    let mut branches = Vec::new();

    let mut truncated_instructions = 0usize;
    let mut unknown_instructions = 0usize;
    let mut endpgm_count = 0usize;
    let mut first_endpgm_offset: Option<usize> = None;
    let mut max_sgpr: Option<u8> = None;
    let mut max_vgpr: Option<u8> = None;

    let mut has_vintrp = false;
    let mut has_barriers = false;
    let mut has_lds = false;
    let mut has_pixel_export = false;
    let mut has_vertex_export = false;
    let mut has_any_export = false;
    let mut has_compute_mem = false;

    for inst in instructions {
        *encoding_counts.entry(inst.encoding).or_insert(0) += 1;
        if inst.truncated {
            truncated_instructions += 1;
        }
        if inst.encoding == Rdna2Encoding::Unknown {
            unknown_instructions += 1;
        }

        match &inst.detail {
            Rdna2Detail::Sopp {
                op, branch_target, ..
            } => {
                if matches!(op, 0x01 | 0x1B | 0x1E) {
                    endpgm_count += 1;
                    if first_endpgm_offset.is_none() {
                        first_endpgm_offset = Some(inst.offset);
                    }
                }
                if *op == 0x0A {
                    has_barriers = true;
                }
                if let Some(target_offset) = branch_target {
                    let (_, _, conditional) = sopp_mnemonic(*op);
                    branches.push(Rdna2BranchRef {
                        offset: inst.offset,
                        mnemonic: inst.mnemonic.clone(),
                        target_offset: *target_offset,
                        conditional,
                    });
                }
            }
            Rdna2Detail::Sopc { ssrc0, ssrc1, .. } => {
                record_sgpr(&mut max_sgpr, *ssrc0);
                record_sgpr(&mut max_sgpr, *ssrc1);
            }
            Rdna2Detail::Sop1 {
                op, sdst, ssrc0, ..
            } => {
                let (_, is_64) = sop1_mnemonic(*op);
                if *op != 0x20 && *op != 0x22 {
                    record_sgpr_range(&mut max_sgpr, *sdst, if is_64 { 2 } else { 1 });
                }
                if *op != 0x1F {
                    record_sgpr_range(&mut max_sgpr, *ssrc0, if is_64 { 2 } else { 1 });
                }
            }
            Rdna2Detail::Sopk {
                op,
                sdst,
                branch_target,
                ..
            } => {
                let width = if *op == 0x16 { 2 } else { 1 };
                record_sgpr_range(&mut max_sgpr, *sdst, width);
                if let Some(target_offset) = branch_target {
                    branches.push(Rdna2BranchRef {
                        offset: inst.offset,
                        mnemonic: inst.mnemonic.clone(),
                        target_offset: *target_offset,
                        conditional: *op != 0x16,
                    });
                }
            }
            Rdna2Detail::Sop2 {
                op,
                sdst,
                ssrc0,
                ssrc1,
                ..
            } => {
                let (_, is_64) = sop2_mnemonic(*op);
                let width = if is_64 { 2 } else { 1 };
                record_sgpr_range(&mut max_sgpr, *sdst, width);
                record_sgpr_range(&mut max_sgpr, *ssrc0, width);
                record_sgpr_range(&mut max_sgpr, *ssrc1, width);
            }
            Rdna2Detail::Smem {
                op,
                sdata,
                sbase,
                soffset,
                offset,
                glc,
                dlc,
            } => {
                let (_, sdata_count, is_buffer) = smem_info(*op);
                let sbase_sgpr = sbase.saturating_mul(2);
                if sdata_count > 0 {
                    has_compute_mem = true;
                    record_sgpr_range(&mut max_sgpr, *sdata, sdata_count);
                    record_sgpr_range(&mut max_sgpr, sbase_sgpr, if is_buffer { 4 } else { 2 });
                    if !inst.truncated && *soffset != 0x7D {
                        record_sgpr(&mut max_sgpr, *soffset);
                    }
                    smem_base_set.insert(sbase_sgpr);
                    smem_refs.push(Rdna2SmemRef {
                        offset: inst.offset,
                        mnemonic: inst.mnemonic.clone(),
                        sbase_sgpr,
                        sdata_sgpr: *sdata,
                        byte_offset: *offset,
                        is_buffer,
                        glc: *glc,
                        dlc: *dlc,
                    });
                }
            }
            Rdna2Detail::Vop1 { op, vdst, src0, .. } => {
                if !matches!(op, 0x00 | 0x1B | 0x41) {
                    if *op == 0x02 {
                        record_sgpr(&mut max_sgpr, *vdst & 0x7F);
                    } else {
                        record_vgpr(&mut max_vgpr, *vdst);
                    }
                    record_src_reg(&mut max_sgpr, &mut max_vgpr, *src0);
                }
            }
            Rdna2Detail::Vopc { vsrc1, src0, .. } => {
                record_vgpr(&mut max_vgpr, *vsrc1);
                record_src_reg(&mut max_sgpr, &mut max_vgpr, *src0);
            }
            Rdna2Detail::Vop2 {
                vdst, vsrc1, src0, ..
            } => {
                record_vgpr(&mut max_vgpr, *vdst);
                record_vgpr(&mut max_vgpr, *vsrc1);
                record_src_reg(&mut max_sgpr, &mut max_vgpr, *src0);
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
                let (_, num_srcs, _) = vop3_info(*op);
                if *op <= 0xFF || *op == 0x360 {
                    record_sgpr(&mut max_sgpr, *vdst & 0x7F);
                } else {
                    record_vgpr(&mut max_vgpr, *vdst);
                }
                if let Some(sd) = sdst {
                    record_sgpr(&mut max_sgpr, *sd);
                }
                if !inst.truncated {
                    record_src_reg(&mut max_sgpr, &mut max_vgpr, *src0);
                    if num_srcs >= 2 {
                        record_src_reg(&mut max_sgpr, &mut max_vgpr, *src1);
                    }
                    if num_srcs >= 3 {
                        record_src_reg(&mut max_sgpr, &mut max_vgpr, *src2);
                    }
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
                record_vgpr(&mut max_vgpr, *vdst);
                if !inst.truncated {
                    record_src_reg(&mut max_sgpr, &mut max_vgpr, *src0);
                    record_src_reg(&mut max_sgpr, &mut max_vgpr, *src1);
                    if vop3p_num_srcs(*op) >= 3 {
                        record_src_reg(&mut max_sgpr, &mut max_vgpr, *src2);
                    }
                }
            }
            Rdna2Detail::Vintrp { op, vdst, vsrc, .. } => {
                has_vintrp = true;
                record_vgpr(&mut max_vgpr, *vdst);
                if *op != 2 {
                    record_vgpr(&mut max_vgpr, *vsrc);
                }
            }
            Rdna2Detail::Exp {
                target,
                en,
                compr,
                done,
                vm,
                vsrc,
            } => {
                has_any_export = true;
                if target.is_pixel_output() {
                    has_pixel_export = true;
                }
                if target.is_vertex_output() {
                    has_vertex_export = true;
                }
                if !inst.truncated {
                    if *compr {
                        if (*en & 0b0011) != 0 {
                            record_vgpr(&mut max_vgpr, vsrc[0]);
                        }
                        if (*en & 0b1100) != 0 {
                            record_vgpr(&mut max_vgpr, vsrc[1]);
                        }
                    } else {
                        for (idx, &reg) in vsrc.iter().enumerate() {
                            if (*en & (1 << idx)) != 0 {
                                record_vgpr(&mut max_vgpr, reg);
                            }
                        }
                    }
                }
                exports.push(Rdna2ExportSummary {
                    offset: inst.offset,
                    target: *target,
                    en: *en,
                    compr: *compr,
                    done: *done,
                    vm: *vm,
                    vsrc: *vsrc,
                });
            }
            Rdna2Detail::Mubuf {
                op,
                vaddr,
                vdata,
                srsrc,
                soffset,
                offen,
                idxen,
                lds,
                ..
            } => {
                has_compute_mem = true;
                if *lds {
                    has_lds = true;
                }
                if !inst.truncated {
                    let srsrc_sgpr = srsrc.saturating_mul(4);
                    record_sgpr_range(&mut max_sgpr, srsrc_sgpr, 4);
                    record_sgpr(&mut max_sgpr, *soffset);
                    let data_count = match *op {
                        0x01 | 0x05 | 0x0D | 0x1D => 2,
                        0x02 | 0x06 | 0x0F | 0x1F => 3,
                        0x03 | 0x07 | 0x0E | 0x1E => 4,
                        _ => 1,
                    };
                    record_vgpr_range(&mut max_vgpr, *vdata, data_count);
                    if *offen && *idxen {
                        record_vgpr_range(&mut max_vgpr, *vaddr, 2);
                    } else if *offen || *idxen {
                        record_vgpr(&mut max_vgpr, *vaddr);
                    }
                    resource_refs.push(Rdna2ResourceRef {
                        offset: inst.offset,
                        encoding: Rdna2Encoding::Mubuf,
                        mnemonic: inst.mnemonic.clone(),
                        srsrc_sgpr,
                        ssamp_sgpr: None,
                        dmask: None,
                        vaddr: *vaddr,
                        vdata: *vdata,
                    });
                }
            }
            Rdna2Detail::Mtbuf {
                vaddr,
                vdata,
                srsrc,
                soffset,
                offen,
                idxen,
                ..
            } => {
                has_compute_mem = true;
                if !inst.truncated {
                    let srsrc_sgpr = srsrc.saturating_mul(4);
                    record_sgpr_range(&mut max_sgpr, srsrc_sgpr, 4);
                    record_sgpr(&mut max_sgpr, *soffset);
                    record_vgpr(&mut max_vgpr, *vdata);
                    if *offen || *idxen {
                        record_vgpr(&mut max_vgpr, *vaddr);
                    }
                    resource_refs.push(Rdna2ResourceRef {
                        offset: inst.offset,
                        encoding: Rdna2Encoding::Mtbuf,
                        mnemonic: inst.mnemonic.clone(),
                        srsrc_sgpr,
                        ssamp_sgpr: None,
                        dmask: None,
                        vaddr: *vaddr,
                        vdata: *vdata,
                    });
                }
            }
            Rdna2Detail::Mimg {
                op,
                dmask,
                r128,
                vaddr,
                vdata,
                srsrc,
                ssamp,
                ..
            } => {
                if matches!(*op, 0x08..=0x0B | 0x10..=0x13) {
                    has_compute_mem = true;
                }
                if !inst.truncated {
                    let (_, uses_sampler) = mimg_info(*op);
                    let srsrc_sgpr = srsrc.saturating_mul(4);
                    let srsrc_count = if *r128 { 4 } else { 8 };
                    record_sgpr_range(&mut max_sgpr, srsrc_sgpr, srsrc_count);
                    let ssamp_sgpr = if uses_sampler {
                        let reg = ssamp.saturating_mul(4);
                        record_sgpr_range(&mut max_sgpr, reg, 4);
                        Some(reg)
                    } else {
                        None
                    };
                    let data_count = dmask.count_ones().max(1) as u8;
                    record_vgpr_range(&mut max_vgpr, *vdata, data_count);
                    record_vgpr(&mut max_vgpr, *vaddr);
                    resource_refs.push(Rdna2ResourceRef {
                        offset: inst.offset,
                        encoding: Rdna2Encoding::Mimg,
                        mnemonic: inst.mnemonic.clone(),
                        srsrc_sgpr,
                        ssamp_sgpr,
                        dmask: Some(*dmask),
                        vaddr: *vaddr,
                        vdata: *vdata,
                    });
                }
            }
            Rdna2Detail::Flat {
                seg,
                op,
                lds,
                addr,
                data,
                saddr,
                vdst,
                ..
            } => {
                has_compute_mem = true;
                if *lds {
                    has_lds = true;
                }
                if !inst.truncated {
                    record_vgpr(&mut max_vgpr, *addr);
                    if matches!(op, 0x18..=0x1F) {
                        record_vgpr(&mut max_vgpr, *data);
                    } else {
                        record_vgpr(&mut max_vgpr, *vdst);
                    }
                    if *seg != 0 && *saddr != 0x7F {
                        record_sgpr_range(&mut max_sgpr, *saddr, 2);
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
                has_lds = true;
                has_compute_mem = true;
                if !inst.truncated {
                    record_vgpr(&mut max_vgpr, *addr);
                    record_vgpr(&mut max_vgpr, *data0);
                    if matches!(*op, 0x0E | 0x0F) {
                        record_vgpr(&mut max_vgpr, *data1);
                    }
                    record_vgpr(&mut max_vgpr, *vdst);
                }
            }
            Rdna2Detail::Unknown { .. } => {}
        }
    }

    let inferred_stage = if has_pixel_export || has_vintrp {
        Rdna2InferredStage::Pixel
    } else if has_vertex_export {
        Rdna2InferredStage::Vertex
    } else if !has_any_export && has_compute_mem {
        Rdna2InferredStage::Compute
    } else {
        Rdna2InferredStage::Unknown
    };

    Rdna2ProgramSummary {
        fnv1a_hash: fnv1a_64(bytes),
        byte_len: bytes.len(),
        trailing_bytes: bytes.len() % 4,
        instruction_count: instructions.len(),
        truncated_instructions,
        unknown_instructions,
        endpgm_count,
        first_endpgm_offset,
        inferred_stage,
        encoding_counts,
        exports,
        smem_refs,
        smem_base_sgprs: smem_base_set.into_iter().collect(),
        resource_refs,
        branches,
        max_sgpr,
        max_vgpr,
        has_vintrp,
        has_barriers,
        has_lds,
    }
}

// ---------------------------------------------------------------------------
// Sanitized human-readable disassembly & JSON formatters
// ---------------------------------------------------------------------------

/// Format a decoded RDNA2 program as strictly sanitized human-readable disassembly text
/// (zero host paths, zero raw bytecode dumps).
pub fn format_disasm(bytes: &[u8], max_instructions: Option<usize>) -> String {
    let all_instructions = decode_program(bytes);
    let summary = analyze_instructions(bytes, &all_instructions);
    format_disasm_with_summary(&summary, &all_instructions, max_instructions)
}

/// Format disassembly from a precomputed summary and decoded instruction slice.
pub fn format_disasm_with_summary(
    summary: &Rdna2ProgramSummary,
    instructions: &[Rdna2Instruction],
    max_instructions: Option<usize>,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "; RDNA2 (GFX10.3) Disassembly Summary");
    let _ = writeln!(out, "; hash:         {}", format_shader_hash(summary.fnv1a_hash));
    let _ = writeln!(out, "; stage:        {}", summary.inferred_stage);
    let _ = writeln!(
        out,
        "; bytes:        {} (trailing: {})",
        summary.byte_len, summary.trailing_bytes
    );
    let _ = writeln!(
        out,
        "; instructions: {} (endpgm: {}, truncated: {}, unknown: {})",
        summary.instruction_count,
        summary.endpgm_count,
        summary.truncated_instructions,
        summary.unknown_instructions
    );
    let sgpr_label = summary
        .max_sgpr
        .map_or_else(|| "none".to_string(), |r| format!("s0..s{r}"));
    let vgpr_label = summary
        .max_vgpr
        .map_or_else(|| "none".to_string(), |r| format!("v0..v{r}"));
    let _ = writeln!(out, "; registers:    sgpr={sgpr_label}, vgpr={vgpr_label}");

    if !summary.exports.is_empty() {
        let targets: Vec<String> = summary
            .exports
            .iter()
            .map(|e| {
                if e.done {
                    format!("{}(done)", e.target)
                } else {
                    e.target.name()
                }
            })
            .collect();
        let _ = writeln!(out, "; exports:      {}", targets.join(", "));
    }

    if !summary.smem_base_sgprs.is_empty() {
        let bases: Vec<String> = summary
            .smem_base_sgprs
            .iter()
            .map(|b| format!("s[{}:{}]", b, b.saturating_add(1)))
            .collect();
        let _ = writeln!(out, "; smem_bases:   {}", bases.join(", "));
    }

    if !summary.resource_refs.is_empty() {
        let _ = writeln!(
            out,
            "; resources:    {} ref(s)",
            summary.resource_refs.len()
        );
    }

    out.push('\n');

    let limit = max_instructions.unwrap_or(instructions.len());
    let shown = instructions.len().min(limit);
    for inst in &instructions[..shown] {
        let _ = writeln!(
            out,
            "  0x{:04x}  [{:<6}]  {}",
            inst.offset,
            inst.encoding.as_str(),
            inst.format_text()
        );
    }
    if shown < instructions.len() {
        let _ = writeln!(
            out,
            "  ; ... ({} more instruction(s) omitted by --max-instructions)",
            instructions.len() - shown
        );
    }

    out
}

/// Serialize an [`Rdna2ProgramSummary`] to strictly sanitized deterministic JSON
/// (zero host paths, zero raw byte dumps).
pub fn to_json(summary: &Rdna2ProgramSummary) -> String {
    to_json_inner(summary, None, None)
}

/// Serialize an [`Rdna2ProgramSummary`] together with sanitized disassembled instruction
/// entries (optionally capped by `max_instructions`) to deterministic JSON.
pub fn to_json_with_disasm(
    summary: &Rdna2ProgramSummary,
    instructions: &[Rdna2Instruction],
    max_instructions: Option<usize>,
) -> String {
    to_json_inner(summary, Some(instructions), max_instructions)
}

fn to_json_inner(
    summary: &Rdna2ProgramSummary,
    instructions: Option<&[Rdna2Instruction]>,
    max_instructions: Option<usize>,
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"isa\": \"rdna2_gfx10_3\",\n");
    out.push_str("  \"fnv1a_hash\": ");
    write_json_string(&mut out, &format_shader_hash(summary.fnv1a_hash));
    out.push_str(",\n");
    let _ = writeln!(out, "  \"byte_len\": {},", summary.byte_len);
    let _ = writeln!(out, "  \"trailing_bytes\": {},", summary.trailing_bytes);
    let _ = writeln!(out, "  \"instruction_count\": {},", summary.instruction_count);
    let _ = writeln!(
        out,
        "  \"truncated_instructions\": {},",
        summary.truncated_instructions
    );
    let _ = writeln!(
        out,
        "  \"unknown_instructions\": {},",
        summary.unknown_instructions
    );
    let _ = writeln!(out, "  \"endpgm_count\": {},", summary.endpgm_count);
    match summary.first_endpgm_offset {
        Some(off) => {
            let _ = writeln!(out, "  \"first_endpgm_offset\": {off},");
        }
        None => {
            out.push_str("  \"first_endpgm_offset\": null,\n");
        }
    }
    out.push_str("  \"inferred_stage\": ");
    write_json_string(&mut out, summary.inferred_stage.as_str());
    out.push_str(",\n");

    match summary.max_sgpr {
        Some(r) => {
            let _ = writeln!(out, "  \"max_sgpr\": {r},");
        }
        None => {
            out.push_str("  \"max_sgpr\": null,\n");
        }
    }
    match summary.max_vgpr {
        Some(r) => {
            let _ = writeln!(out, "  \"max_vgpr\": {r},");
        }
        None => {
            out.push_str("  \"max_vgpr\": null,\n");
        }
    }
    let _ = writeln!(out, "  \"has_vintrp\": {},", summary.has_vintrp);
    let _ = writeln!(out, "  \"has_barriers\": {},", summary.has_barriers);
    let _ = writeln!(out, "  \"has_lds\": {},", summary.has_lds);

    // encoding_counts
    if summary.encoding_counts.is_empty() {
        out.push_str("  \"encoding_counts\": {},\n");
    } else {
        out.push_str("  \"encoding_counts\": {\n");
        for (i, (enc, count)) in summary.encoding_counts.iter().enumerate() {
            out.push_str("    ");
            write_json_string(&mut out, enc.as_str());
            let comma = if i + 1 < summary.encoding_counts.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(out, ": {count}{comma}");
        }
        out.push_str("  },\n");
    }

    // exports
    if summary.exports.is_empty() {
        out.push_str("  \"exports\": [],\n");
    } else {
        out.push_str("  \"exports\": [\n");
        for (i, exp) in summary.exports.iter().enumerate() {
            out.push_str("    {\"offset\": ");
            let _ = write!(out, "{}, \"target\": ", exp.offset);
            write_json_string(&mut out, &exp.target.name());
            let comma = if i + 1 < summary.exports.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                out,
                ", \"en\": {}, \"compr\": {}, \"done\": {}, \"vm\": {}, \"vsrc\": [{}, {}, {}, {}]}}{comma}",
                exp.en, exp.compr, exp.done, exp.vm, exp.vsrc[0], exp.vsrc[1], exp.vsrc[2], exp.vsrc[3]
            );
        }
        out.push_str("  ],\n");
    }

    // smem_base_sgprs
    out.push_str("  \"smem_base_sgprs\": [");
    for (i, sgpr) in summary.smem_base_sgprs.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let _ = write!(out, "{sgpr}");
    }
    out.push_str("],\n");

    // smem_refs
    if summary.smem_refs.is_empty() {
        out.push_str("  \"smem_refs\": [],\n");
    } else {
        out.push_str("  \"smem_refs\": [\n");
        for (i, sref) in summary.smem_refs.iter().enumerate() {
            out.push_str("    {\"offset\": ");
            let _ = write!(out, "{}, \"mnemonic\": ", sref.offset);
            write_json_string(&mut out, &sref.mnemonic);
            let comma = if i + 1 < summary.smem_refs.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                out,
                ", \"sbase_sgpr\": {}, \"sdata_sgpr\": {}, \"byte_offset\": {}, \"is_buffer\": {}, \"glc\": {}, \"dlc\": {}}}{comma}",
                sref.sbase_sgpr, sref.sdata_sgpr, sref.byte_offset, sref.is_buffer, sref.glc, sref.dlc
            );
        }
        out.push_str("  ],\n");
    }

    // resource_refs
    if summary.resource_refs.is_empty() {
        out.push_str("  \"resource_refs\": [],\n");
    } else {
        out.push_str("  \"resource_refs\": [\n");
        for (i, rref) in summary.resource_refs.iter().enumerate() {
            out.push_str("    {\"offset\": ");
            let _ = write!(out, "{}, \"encoding\": ", rref.offset);
            write_json_string(&mut out, rref.encoding.as_str());
            out.push_str(", \"mnemonic\": ");
            write_json_string(&mut out, &rref.mnemonic);
            let _ = write!(out, ", \"srsrc_sgpr\": {}, \"ssamp_sgpr\": ", rref.srsrc_sgpr);
            match rref.ssamp_sgpr {
                Some(s) => {
                    let _ = write!(out, "{s}");
                }
                None => out.push_str("null"),
            }
            out.push_str(", \"dmask\": ");
            match rref.dmask {
                Some(dm) => {
                    let _ = write!(out, "{dm}");
                }
                None => out.push_str("null"),
            }
            let comma = if i + 1 < summary.resource_refs.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                out,
                ", \"vaddr\": {}, \"vdata\": {}}}{comma}",
                rref.vaddr, rref.vdata
            );
        }
        out.push_str("  ],\n");
    }

    // branches
    let branches_comma = if instructions.is_some() { "," } else { "" };
    if summary.branches.is_empty() {
        let _ = writeln!(out, "  \"branches\": []{branches_comma}");
    } else {
        out.push_str("  \"branches\": [\n");
        for (i, br) in summary.branches.iter().enumerate() {
            out.push_str("    {\"offset\": ");
            let _ = write!(out, "{}, \"mnemonic\": ", br.offset);
            write_json_string(&mut out, &br.mnemonic);
            let comma = if i + 1 < summary.branches.len() {
                ","
            } else {
                ""
            };
            let _ = writeln!(
                out,
                ", \"target_offset\": {}, \"conditional\": {}}}{comma}",
                br.target_offset, br.conditional
            );
        }
        let _ = writeln!(out, "  ]{branches_comma}");
    }

    // Optional sanitized disassembly array
    if let Some(insts) = instructions {
        let limit = max_instructions.unwrap_or(insts.len());
        let shown = insts.len().min(limit);
        let _ = writeln!(out, "  \"shown_instructions\": {shown},");
        if shown == 0 {
            out.push_str("  \"instructions\": []\n");
        } else {
            out.push_str("  \"instructions\": [\n");
            for (i, inst) in insts[..shown].iter().enumerate() {
                out.push_str("    {\"offset\": ");
                let _ = write!(out, "{}, \"size\": {}, \"encoding\": ", inst.offset, inst.size);
                write_json_string(&mut out, inst.encoding.as_str());
                out.push_str(", \"mnemonic\": ");
                write_json_string(&mut out, &inst.mnemonic);
                out.push_str(", \"text\": ");
                write_json_string(&mut out, &inst.format_text());
                let comma = if i + 1 < shown { "," } else { "" };
                let _ = writeln!(out, ", \"truncated\": {}}}{comma}", inst.truncated);
            }
            out.push_str("  ]\n");
        }
    }

    out.push_str("}\n");
    out
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(words.len() * 4);
        for &w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    }

    // Synthetic RDNA2 (GFX10.3) instruction encoders for unit tests:
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

    fn enc_sopk(op: u8, sdst: u8, simm16: u16) -> u32 {
        (0b1011u32 << 28)
            | (u32::from(op & 0x1F) << 23)
            | (u32::from(sdst & 0x7F) << 16)
            | u32::from(simm16)
    }

    fn enc_sopc(op: u8, ssrc0: u8, ssrc1: u8) -> u32 {
        (0b1011_1111_0u32 << 23)
            | (u32::from(op & 0x7F) << 16)
            | (u32::from(ssrc1) << 8)
            | u32::from(ssrc0)
    }

    fn enc_smem(op: u8, sdata: u8, sbase_pair: u8, soffset: u8, offset: i32, glc: bool, dlc: bool) -> [u32; 2] {
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

    fn enc_exp(target: u8, en: u8, compr: bool, done: bool, vm: bool, vsrc: [u8; 4]) -> [u32; 2] {
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
    fn enc_mimg(op: u8, dmask: u8, dim: u8, vaddr: u8, vdata: u8, srsrc: u8, ssamp: u8) -> [u32; 2] {
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

    #[test]
    fn decodes_synthetic_vs_fullscreen_triangle_program() {
        // Synthetic fullscreen-triangle VS:
        //   v_mov_b32 v1, 0.0 (inline 0 -> 128)
        //   v_mov_b32 v2, 1.0 (inline 1.0 -> 242)
        //   v_add_f32 v3, v0, v1
        //   exp pos0 v0, v1, v2, v3 done
        //   exp param0 v0, v1, off, off
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vop1(0x01, 1, 128)); // v_mov_b32 v1, 0
        words.push(enc_vop1(0x01, 2, 242)); // v_mov_b32 v2, 1.0
        words.push(enc_vop2(0x03, 3, 256 + 0, 1)); // v_add_f32 v3, v0, v1
        words.extend_from_slice(&enc_exp(12, 0xF, false, false, false, [0, 1, 2, 3])); // pos0
        words.extend_from_slice(&enc_exp(32, 0x3, false, true, false, [0, 1, 0, 0])); // param0 done
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let bytes = words_to_bytes(&words);
        let insts = decode_program(&bytes);
        assert_eq!(insts.len(), 6);
        assert_eq!(insts[0].format_text(), "v_mov_b32 v1, 0");
        assert_eq!(insts[1].format_text(), "v_mov_b32 v2, 1.0");
        assert_eq!(insts[2].format_text(), "v_add_f32 v3, v0, v1");
        assert_eq!(insts[3].format_text(), "exp pos0 v0, v1, v2, v3");
        assert_eq!(insts[4].format_text(), "exp param0 v0, v1, off, off done");
        assert_eq!(insts[5].format_text(), "s_endpgm");

        let summary = analyze_program(&bytes);
        assert_eq!(summary.inferred_stage, Rdna2InferredStage::Vertex);
        assert_eq!(summary.instruction_count, 6);
        assert_eq!(summary.endpgm_count, 1);
        assert_eq!(summary.first_endpgm_offset, Some(28));
        assert_eq!(summary.exports.len(), 2);
        assert_eq!(summary.exports[0].target, Rdna2ExportTarget::Pos(0));
        assert_eq!(summary.exports[1].target, Rdna2ExportTarget::Param(0));
        assert!(summary.exports[1].done);
        assert_eq!(summary.max_vgpr, Some(3));
        assert_eq!(summary.max_sgpr, None);
        assert_eq!(summary.fnv1a_hash, fnv1a_64(&bytes));

        let disasm = format_disasm(&bytes, None);
        assert!(disasm.contains("; stage:        vertex"));
        assert!(disasm.contains("exp pos0 v0, v1, v2, v3"));
        assert!(disasm.contains("exp param0 v0, v1, off, off done"));

        let json = summary.to_json();
        assert!(json.contains("\"inferred_stage\": \"vertex\""));
        assert!(json.contains("\"target\": \"pos0\""));
        assert!(json.contains("\"target\": \"param0\""));
    }

    #[test]
    fn decodes_synthetic_ps_splash_blit_program() {
        // Synthetic splash-blit PS:
        //   v_interp_p1_f32 v2, v0, attr0.x
        //   v_interp_p2_f32 v2, v1, attr0.x
        //   v_interp_p1_f32 v3, v0, attr0.y
        //   v_interp_p2_f32 v3, v1, attr0.y
        //   s_load_dwordx4 s[4:7], s[0:1], 0x10 glc
        //   s_waitcnt lgkmcnt(0)
        //   image_sample v4, v2, s[8:15], s[4:7] dmask:0xf dim:1
        //   s_waitcnt vmcnt(0)
        //   exp mrt0 v4, v5, v6, v7 done vm
        //   s_endpgm
        let mut words = Vec::new();
        words.push(enc_vintrp(0, 2, 0, 0, 0));
        words.push(enc_vintrp(1, 2, 0, 0, 1));
        words.push(enc_vintrp(0, 3, 0, 1, 0));
        words.push(enc_vintrp(1, 3, 0, 1, 1));
        words.extend_from_slice(&enc_smem(0x02, 4, 0, 0x7D, 0x10, true, false)); // s_load_dwordx4 s[4:7], s[0:1], 0x10 glc
        // waitcnt lgkmcnt(0): vmcnt=0x3F (bits 3:0=0xF, bits 15:14=0x3), expcnt=7 (bits 6:4=7), lgkmcnt=0 (bits 13:8=0)
        let wait_lgkm0: u16 = 0xC07F;
        words.push(enc_sopp(0x0C, wait_lgkm0));
        // image_sample op=0x20, dmask=0xF, dim=1 (2D), vaddr=v2, vdata=v4, srsrc=2 (s[8:15]), ssamp=1 (s[4:7])
        words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 4, 2, 1));
        // waitcnt vmcnt(0): vmcnt=0, expcnt=7, lgkmcnt=0x3F
        let wait_vm0: u16 = 0x3F70;
        words.push(enc_sopp(0x0C, wait_vm0));
        words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7])); // mrt0 done vm
        words.push(enc_sopp(0x01, 0)); // s_endpgm

        let bytes = words_to_bytes(&words);
        let insts = decode_program(&bytes);
        assert_eq!(insts.len(), 10);
        assert_eq!(insts[0].format_text(), "v_interp_p1_f32 v2, v0, attr0.x");
        assert_eq!(insts[1].format_text(), "v_interp_p2_f32 v2, v1, attr0.x");
        assert_eq!(
            insts[4].format_text(),
            "s_load_dwordx4 s[4:7], s[0:1], 0x10 glc"
        );
        assert_eq!(insts[5].format_text(), "s_waitcnt lgkmcnt(0)");
        assert_eq!(
            insts[6].format_text(),
            "image_sample v4, v2, s[8:15], s[4:7] dmask:0xf dim:1"
        );
        assert_eq!(insts[7].format_text(), "s_waitcnt vmcnt(0)");
        assert_eq!(insts[8].format_text(), "exp mrt0 v4, v5, v6, v7 done vm");
        assert_eq!(insts[9].format_text(), "s_endpgm");

        let summary = analyze_program(&bytes);
        assert_eq!(summary.inferred_stage, Rdna2InferredStage::Pixel);
        assert!(summary.has_vintrp);
        assert_eq!(summary.exports.len(), 1);
        assert_eq!(summary.exports[0].target, Rdna2ExportTarget::Mrt(0));
        assert!(summary.exports[0].done);
        assert!(summary.exports[0].vm);
        assert_eq!(summary.smem_base_sgprs, vec![0]);
        assert_eq!(summary.smem_refs.len(), 1);
        assert_eq!(summary.smem_refs[0].byte_offset, 0x10);
        assert!(summary.smem_refs[0].glc);
        assert_eq!(summary.resource_refs.len(), 1);
        assert_eq!(summary.resource_refs[0].encoding, Rdna2Encoding::Mimg);
        assert_eq!(summary.resource_refs[0].srsrc_sgpr, 8);
        assert_eq!(summary.resource_refs[0].ssamp_sgpr, Some(4));
        assert_eq!(summary.resource_refs[0].dmask, Some(0xF));
        assert_eq!(summary.max_sgpr, Some(15));
        assert_eq!(summary.max_vgpr, Some(7));

        let json = to_json_with_disasm(&summary, &insts, Some(4));
        assert!(json.contains("\"inferred_stage\": \"pixel\""));
        assert!(json.contains("\"srsrc_sgpr\": 8, \"ssamp_sgpr\": 4"));
        assert!(json.contains("\"shown_instructions\": 4"));
    }

    #[test]
    fn decodes_64bit_literals_vop3_branches_and_compute_stage() {
        let mut words = Vec::new();
        // 0x00: s_mov_b32 s2, 0xdeadbeef (SOP1 with src0 == 0xFF literal -> 8 bytes)
        words.push(enc_sop1(0x03, 2, 0xFF));
        words.push(0xDEAD_BEEF);
        // 0x08: s_add_u32 s3, s2, 0x00000100 (SOP2 with src1 == 0xFF literal -> 8 bytes)
        words.push(enc_sop2(0x00, 3, 2, 0xFF));
        words.push(0x0000_0100);
        // 0x10: s_cmp_eq_u32 s3, 0x00000042 (SOPC with src1 == 0xFF literal -> 8 bytes)
        words.push(enc_sopc(0x06, 3, 0xFF));
        words.push(0x0000_0042);
        // 0x18: s_cbranch_scc1 +4 dwords -> target = 0x18 + 4 + 16 = 0x2c
        words.push(enc_sopp(0x05, 4u16));
        // 0x1c: v_fmamk_f32 v0, v1, 0x3f000000, v2 (always 64-bit!)
        words.push(enc_vop2(0x2C, 0, 256 + 1, 2));
        words.push(0x3F00_0000);
        // 0x24: v_fma_f32 v4, |v0|, -v1, v2 clamp mul:2 (VOP3 -> 8 bytes)
        words.extend_from_slice(&enc_vop3(0x14B, 4, 256 + 0, 256 + 1, 256 + 2, true, 1, 0b001, 0b010));
        // 0x2c: s_load_dword s8, s[0:1], 0x0 (SMEM -> 8 bytes)
        words.extend_from_slice(&enc_smem(0x00, 8, 0, 0x7D, 0, false, true));
        // 0x34: s_setreg_imm32_b32 hwreg(0x0001), 0x12345678 (SOPK 64-bit!)
        words.push(enc_sopk(0x15, 0, 0x0001));
        words.push(0x1234_5678);
        // 0x3c: s_branch -2 dwords -> target = 0x3c + 4 - 8 = 0x38
        words.push(enc_sopp(0x02, (-2i16) as u16));
        // 0x40: v_cmp_lt_f32 vcc, 0x40000000, v4 (VOPC with literal -> 8 bytes)
        words.push(enc_vopc(0x01, 0xFF, 4));
        words.push(0x4000_0000);
        // 0x48: s_waitcnt_vscnt null, 0x0000 (SOPK 0x17)
        words.push(enc_sopk(0x17, 125, 0));
        // 0x4c: s_endpgm
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let insts = decode_program(&bytes);
        assert_eq!(insts.len(), 12);

        assert_eq!(insts[0].size, 8);
        assert_eq!(insts[0].format_text(), "s_mov_b32 s2, 0xdeadbeef");
        assert_eq!(insts[1].size, 8);
        assert_eq!(insts[1].format_text(), "s_add_u32 s3, s2, 0x00000100");
        assert_eq!(insts[2].size, 8);
        assert_eq!(insts[2].format_text(), "s_cmp_eq_u32 s3, 0x00000042");
        assert_eq!(insts[3].offset, 0x18);
        assert_eq!(
            insts[3].detail,
            Rdna2Detail::Sopp {
                op: 0x05,
                simm16: 4,
                branch_target: Some(0x2C),
            }
        );
        assert_eq!(insts[4].size, 8);
        assert_eq!(insts[4].format_text(), "v_fmamk_f32 v0, v1, 0x3f000000, v2");
        assert_eq!(insts[5].size, 8);
        assert_eq!(
            insts[5].format_text(),
            "v_fma_f32 v4, |v0|, -v1, v2 clamp mul:2"
        );
        assert_eq!(insts[6].offset, 0x2C);
        assert_eq!(insts[6].format_text(), "s_load_dword s8, s[0:1], 0x0 dlc");
        assert_eq!(insts[7].size, 8);
        assert_eq!(
            insts[7].format_text(),
            "s_setreg_imm32_b32 hwreg(0x0001), 0x12345678"
        );
        assert_eq!(insts[8].offset, 0x3C);
        assert_eq!(
            insts[8].detail,
            Rdna2Detail::Sopp {
                op: 0x02,
                simm16: (-2i16) as u16,
                branch_target: Some(0x38),
            }
        );
        assert_eq!(insts[9].size, 8);
        assert_eq!(insts[9].format_text(), "v_cmp_lt_f32 vcc, 0x40000000, v4");
        assert_eq!(insts[10].format_text(), "s_waitcnt_vscnt null, 0x0000");
        assert_eq!(insts[11].format_text(), "s_endpgm");

        let summary = analyze_program(&bytes);
        assert_eq!(summary.inferred_stage, Rdna2InferredStage::Compute);
        assert_eq!(summary.branches.len(), 2);
        assert_eq!(summary.branches[0].target_offset, 0x2C);
        assert!(summary.branches[0].conditional);
        assert_eq!(summary.branches[1].target_offset, 0x38);
        assert!(!summary.branches[1].conditional);
        assert_eq!(summary.truncated_instructions, 0);
    }

    #[test]
    fn decodes_mubuf_mtbuf_flat_ds_and_unknown_encodings() {
        let mut words = Vec::new();
        // MUBUF buffer_load_dword v0, v1, s[4:7], s0 offen offset:16 glc
        let mubuf_w0 = (0b111000u32 << 26) | (0x0Cu32 << 18) | (1 << 14) | (1 << 12) | 16;
        let mubuf_w1 = (1u32 << 16) | 1; // srsrc=1 (s[4:7]), vdata=0, vaddr=1, soffset=s0
        words.extend_from_slice(&[mubuf_w0, mubuf_w1]);

        // MTBUF tbuffer_load_format_xyzw v2, off, s[8:11], 0 format:4
        let mtbuf_w0 = (0b111010u32 << 26) | (4u32 << 19) | (3u32 << 16);
        let mtbuf_w1 = (128u32 << 24) | (2u32 << 16) | (2u32 << 8);
        words.extend_from_slice(&[mtbuf_w0, mtbuf_w1]);

        // FLAT global_load_dword v3, v4, s[12:13] offset:-4
        let flat_w0 = (0b110111u32 << 26) | (0x0Cu32 << 18) | (2u32 << 14) | ((-4i16 as u16 as u32) & 0xFFF);
        let flat_w1 = (3u32 << 24) | (12u32 << 16) | 4;
        words.extend_from_slice(&[flat_w0, flat_w1]);

        // DS ds_read_b32 v5, v6, v0 offset0:8
        let ds_w0 = (0b110110u32 << 26) | (0x36u32 << 18) | 8;
        let ds_w1 = (5u32 << 24) | 6;
        words.extend_from_slice(&[ds_w0, ds_w1]);

        // Unknown 6-bit prefix 0b111111 (0x3F)
        words.push(0xFFFF_FFFF);

        let bytes = words_to_bytes(&words);
        let insts = decode_program(&bytes);
        assert_eq!(insts.len(), 5);
        assert_eq!(insts[0].encoding, Rdna2Encoding::Mubuf);
        assert_eq!(
            insts[0].format_text(),
            "buffer_load_dword v0, v1, s[4:7], s0 offen offset:16 glc"
        );
        assert_eq!(insts[1].encoding, Rdna2Encoding::Mtbuf);
        assert_eq!(
            insts[1].format_text(),
            "tbuffer_load_format_xyzw v2, off, s[8:11], 0 format:4"
        );
        assert_eq!(insts[2].encoding, Rdna2Encoding::Flat);
        assert_eq!(
            insts[2].format_text(),
            "global_load_dword v3, v4, s[12:13] offset:-4"
        );
        assert_eq!(insts[3].encoding, Rdna2Encoding::Ds);
        assert_eq!(insts[3].format_text(), "ds_read_b32 v5, v6, v0 offset0:8");
        assert_eq!(insts[4].encoding, Rdna2Encoding::Unknown);

        let summary = analyze_program(&bytes);
        assert!(summary.has_lds);
        assert_eq!(summary.unknown_instructions, 1);
        assert_eq!(summary.resource_refs.len(), 2);
        assert_eq!(summary.resource_refs[0].srsrc_sgpr, 4);
        assert_eq!(summary.resource_refs[1].srsrc_sgpr, 8);
    }

    #[test]
    fn handles_empty_misaligned_truncated_and_arbitrary_buffers_without_panicking() {
        // Empty and sub-word buffers
        for len in 0..=3 {
            let buf = vec![0xAAu8; len];
            let summary = analyze_program(&buf);
            assert_eq!(summary.instruction_count, 0);
            assert_eq!(summary.trailing_bytes, len);
            assert_eq!(summary.inferred_stage, Rdna2InferredStage::Unknown);
            let _ = format_disasm(&buf, None);
            let _ = summary.to_json();
        }

        // Truncated 64-bit instructions (only first 4 bytes present)
        let truncated_words = [
            enc_smem(0x02, 0, 0, 0x7D, 0, false, false)[0], // SMEM missing w1
            enc_exp(12, 0xF, false, true, false, [0, 1, 2, 3])[0], // EXP missing w1
            enc_vop3(0x14B, 0, 0, 0, 0, false, 0, 0, 0)[0], // VOP3 missing w1
            enc_vop2(0x2C, 0, 256, 1),                      // v_fmamk_f32 missing literal
            enc_sop1(0x03, 0, 0xFF),                        // s_mov_b32 literal missing
            enc_sopk(0x15, 0, 0x0001),                      // s_setreg_imm32_b32 missing literal
            enc_mimg(0x20, 0xF, 1, 0, 0, 0, 0)[0],          // MIMG missing w1
        ];
        for w0 in truncated_words {
            let mut buf = w0.to_le_bytes().to_vec();
            buf.push(0x99); // misaligned +1 byte
            let insts = decode_program(&buf);
            assert_eq!(insts.len(), 1);
            assert!(insts[0].truncated);
            assert_eq!(insts[0].size, 4);
            let summary = analyze_program(&buf);
            assert_eq!(summary.truncated_instructions, 1);
            assert_eq!(summary.trailing_bytes, 1);
        }

        // Sweep pseudo-random deterministic byte streams of varying lengths
        let mut state: u64 = 0x1234_5678_9ABC_DEF0;
        let mut fuzz_buf = Vec::with_capacity(515);
        for _ in 0..515 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            fuzz_buf.push((state & 0xFF) as u8);
        }
        let summary = analyze_program(&fuzz_buf);
        assert_eq!(summary.byte_len, 515);
        assert_eq!(summary.trailing_bytes, 3);
        let _ = format_disasm(&fuzz_buf, Some(32));
        let _ = summary.to_json();
    }

    #[test]
    fn decodes_v_cndmask_b32_gfx10_vop2_0x00_and_vop3_0x100_encodings() {
        let mut words = Vec::new();
        words.push(enc_vop2(0x00, 3, 256 + 1, 2)); // v_cndmask_b32 v3, v1, v2, vcc
        words.extend_from_slice(&enc_vop3(0x100, 4, 256 + 1, 256 + 2, 10, false, 0, 0, 0)); // v_cndmask_b32 v4, v1, v2, s10
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let insts = decode_program(&bytes);
        assert_eq!(insts.len(), 3);
        assert_eq!(insts[0].mnemonic, "v_cndmask_b32");
        assert_eq!(insts[0].format_text(), "v_cndmask_b32 v3, v1, v2, vcc");
        assert_eq!(insts[1].mnemonic, "v_cndmask_b32");
        assert_eq!(insts[1].format_text(), "v_cndmask_b32 v4, v1, v2, s10");
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

    #[test]
    fn decodes_vop3p_packed_f16_s_buffer_load_and_mimg_sample_variants() {
        let mut words = Vec::new();
        // v_pk_fma_f16 v4, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[0,1,0] clamp
        words.extend_from_slice(&enc_vop3p(
            0x0E,
            4,
            256 + 1,
            256 + 2,
            256 + 3,
            true,
            0b001,
            0b101,
            0b001,
            0b010,
        ));
        // v_pk_mul_f16 v5, v1, v2 (2-source VOP3P with default op_sel_hi=0b11; unused src2=0xFF must NOT trigger literal!)
        words.extend_from_slice(&enc_vop3p(
            0x10,
            5,
            256 + 1,
            256 + 2,
            0xFF,
            false,
            0,
            0b011,
            0,
            0,
        ));
        // v_pk_add_f16 v6, v4, v5
        words.extend_from_slice(&enc_vop3p(
            0x0F,
            6,
            256 + 4,
            256 + 5,
            0,
            false,
            0,
            0b011,
            0,
            0,
        ));
        // v_pk_min_f16 / v_pk_max_f16
        words.extend_from_slice(&enc_vop3p(0x11, 7, 256 + 4, 256 + 5, 0, false, 0, 0b011, 0, 0));
        words.extend_from_slice(&enc_vop3p(0x12, 8, 256 + 6, 256 + 7, 0, false, 0, 0b011, 0, 0));
        // v_cvt_pkrtz_f16_f32 (VOP2 0x2E and VOP3 0x12E)
        words.push(enc_vop2(0x2E, 9, 256 + 1, 2));
        words.extend_from_slice(&enc_vop3(0x12E, 10, 256 + 3, 256 + 4, 0, true, 0, 0, 0));
        // s_buffer_load_dword / x2 / x4 / x8 / x16
        words.extend_from_slice(&enc_smem(0x08, 4, 0, 0x7D, 0x0, false, false));
        words.extend_from_slice(&enc_smem(0x09, 6, 0, 0x7D, 0x10, false, false));
        words.extend_from_slice(&enc_smem(0x0A, 8, 0, 0x7D, 0x20, false, false));
        words.extend_from_slice(&enc_smem(0x0B, 16, 0, 0x7D, 0x40, false, false));
        words.extend_from_slice(&enc_smem(0x0C, 32, 0, 0x7D, 0x80, false, false));
        // MIMG variants: image_sample_lz (0x27), image_sample_l (0x24), image_sample_b (0x25),
        // image_sample_c (0x28), image_load (0x00), image_load_mip (0x01)
        words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 2, 12, 2, 1));
        words.extend_from_slice(&enc_mimg(0x24, 0x7, 1, 2, 16, 2, 1));
        words.extend_from_slice(&enc_mimg(0x25, 0xF, 1, 2, 20, 2, 1));
        words.extend_from_slice(&enc_mimg(0x28, 0x1, 1, 2, 24, 2, 1));
        words.extend_from_slice(&enc_mimg(0x00, 0xF, 1, 2, 25, 2, 0));
        words.extend_from_slice(&enc_mimg(0x01, 0x3, 1, 2, 29, 2, 0));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let insts = decode_program(&bytes);
        assert_eq!(insts.len(), 19);
        assert_eq!(
            insts[0].format_text(),
            "v_pk_fma_f16 v4, v1, v2, v3 op_sel:[1,0,0] op_sel_hi:[1,0,1] neg_lo:[1,0,0] neg_hi:[0,1,0] clamp"
        );
        assert_eq!(insts[1].size, 8);
        assert_eq!(insts[1].format_text(), "v_pk_mul_f16 v5, v1, v2");
        assert_eq!(insts[2].format_text(), "v_pk_add_f16 v6, v4, v5");
        assert_eq!(insts[3].format_text(), "v_pk_min_f16 v7, v4, v5");
        assert_eq!(insts[4].format_text(), "v_pk_max_f16 v8, v6, v7");
        assert_eq!(insts[5].format_text(), "v_cvt_pkrtz_f16_f32 v9, v1, v2");
        assert_eq!(
            insts[6].format_text(),
            "v_cvt_pkrtz_f16_f32 v10, v3, v4 clamp"
        );
        assert_eq!(
            insts[7].format_text(),
            "s_buffer_load_dword s4, s[0:3], 0x0"
        );
        assert_eq!(
            insts[8].format_text(),
            "s_buffer_load_dwordx2 s[6:7], s[0:3], 0x10"
        );
        assert_eq!(
            insts[9].format_text(),
            "s_buffer_load_dwordx4 s[8:11], s[0:3], 0x20"
        );
        assert_eq!(
            insts[10].format_text(),
            "s_buffer_load_dwordx8 s[16:23], s[0:3], 0x40"
        );
        assert_eq!(
            insts[11].format_text(),
            "s_buffer_load_dwordx16 s[32:47], s[0:3], 0x80"
        );
        assert_eq!(insts[12].mnemonic, "image_sample_lz");
        assert_eq!(insts[13].mnemonic, "image_sample_l");
        assert_eq!(insts[14].mnemonic, "image_sample_b");
        assert_eq!(insts[15].mnemonic, "image_sample_c");
        assert_eq!(insts[16].mnemonic, "image_load");
        assert_eq!(insts[17].mnemonic, "image_load_mip");
        assert_eq!(insts[18].mnemonic, "s_endpgm");

        let summary = analyze_program(&bytes);
        assert_eq!(summary.truncated_instructions, 0);
        assert_eq!(summary.smem_refs.len(), 5);
        assert!(summary.smem_refs.iter().all(|r| r.is_buffer));
        assert_eq!(summary.resource_refs.len(), 6);
    }

    #[test]
    fn decodes_compute_storage_buffer_image_store_lds_and_barrier_ops() {
        let mut words = Vec::new();
        // buffer_load_dwordx4 v4, v0, s[4:7], 0 offen offset:16
        let bld_w0 = (0b111000u32 << 26) | (0x0Eu32 << 18) | (1 << 12) | 16;
        let bld_w1 = (128u32 << 24) | (1u32 << 16) | (4u32 << 8) | 0;
        words.extend_from_slice(&[bld_w0, bld_w1]);
        // buffer_load_ubyte v8, v0, s[4:7], 0 offen
        let bub_w0 = (0b111000u32 << 26) | (0x08u32 << 18) | (1 << 12);
        let bub_w1 = (128u32 << 24) | (1u32 << 16) | (8u32 << 8) | 0;
        words.extend_from_slice(&[bub_w0, bub_w1]);
        // buffer_load_ushort v9, v0, s[4:7], 0 offen
        let bus_w0 = (0b111000u32 << 26) | (0x0Au32 << 18) | (1 << 12);
        let bus_w1 = (128u32 << 24) | (1u32 << 16) | (9u32 << 8) | 0;
        words.extend_from_slice(&[bus_w0, bus_w1]);
        // ds_write2_b32 v0, v1, v4, v5 offset0:0 offset1:4
        let dsw2_w0 = (0b110110u32 << 26) | (0x0Eu32 << 18) | (4u32 << 8);
        let dsw2_w1 = (5u32 << 16) | (4u32 << 8) | 1;
        words.extend_from_slice(&[dsw2_w0, dsw2_w1]);
        // s_barrier
        words.push(enc_sopp(0x0A, 0));
        // ds_read_b128 v10, v1, v0 offset0:0
        let dsr128_w0 = (0b110110u32 << 26) | (0xFFu32 << 18);
        let dsr128_w1 = (10u32 << 24) | 1;
        words.extend_from_slice(&[dsr128_w0, dsr128_w1]);
        // buffer_store_dwordx4 v10, v0, s[8:11], 0 offen
        let bst_w0 = (0b111000u32 << 26) | (0x1Eu32 << 18) | (1 << 12);
        let bst_w1 = (128u32 << 24) | (2u32 << 16) | (10u32 << 8) | 0;
        words.extend_from_slice(&[bst_w0, bst_w1]);
        // image_store v10, v0, s[12:19] dmask:0xf dim:1
        words.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 10, 3, 0));
        // image_store_mip v10, v0, s[12:19] dmask:0xf dim:1
        words.extend_from_slice(&enc_mimg(0x09, 0xF, 1, 0, 10, 3, 0));
        words.push(enc_sopp(0x01, 0));

        let bytes = words_to_bytes(&words);
        let insts = decode_program(&bytes);
        assert_eq!(insts.len(), 10);
        assert_eq!(insts[0].mnemonic, "buffer_load_dwordx4");
        assert_eq!(insts[1].mnemonic, "buffer_load_ubyte");
        assert_eq!(insts[2].mnemonic, "buffer_load_ushort");
        assert_eq!(insts[3].mnemonic, "ds_write2_b32");
        assert!(insts[3].format_text().contains("v4, v5"));
        assert_eq!(insts[4].mnemonic, "s_barrier");
        assert_eq!(insts[5].mnemonic, "ds_read_b128");
        assert_eq!(insts[6].mnemonic, "buffer_store_dwordx4");
        assert_eq!(insts[7].mnemonic, "image_store");
        assert_eq!(insts[8].mnemonic, "image_store_mip");

        let summary = analyze_program(&bytes);
        assert_eq!(summary.inferred_stage, Rdna2InferredStage::Compute);
        assert!(summary.has_barriers);
        assert!(summary.has_lds);
        assert_eq!(summary.truncated_instructions, 0);

        // Standalone image_store-only program must also infer Compute stage.
        let mut img_only = Vec::new();
        img_only.extend_from_slice(&enc_mimg(0x08, 0xF, 1, 0, 4, 2, 0));
        img_only.push(enc_sopp(0x01, 0));
        let img_summary = analyze_program(&words_to_bytes(&img_only));
        assert_eq!(img_summary.inferred_stage, Rdna2InferredStage::Compute);
    }
}
