//! Integration tests for S69 (#1740): 3D compute, mesh-shader, atomics, LDS, and wave/subgroup
//! decoding, CFG analysis, and WGSL wave emulation in `playgta6-shaders`.

use naga::valid::{Capabilities, ShaderStages, SubgroupOperationSet, ValidationFlags, Validator};
use playgta6_shaders::rdna2::{
    Rdna2AtomicDomain, Rdna2AtomicOp, Rdna2DsOpKind, Rdna2Encoding,
    Rdna2ExportTarget, Rdna2InferredStage, Rdna2SwizzleMode, Rdna2WaveOpKind,
    SENDMSG_ID_GS_ALLOC_REQ, SENDMSG_ID_GS_DONE, analyze_instructions, analyze_program,
    decode_ds_swizzle_offset, decode_program, decode_sendmsg, extract_mesh_metadata,
};
use playgta6_shaders::rdna2_cfg::{
    ControlFlowGraph, MeshAllocGuardKind, analyze_compute_mesh_cfg, structure_cfg,
    summarize_compute_mesh_cfg,
};
use playgta6_shaders::wave_emul::{
    RdnaBitmaskHelper, RdnaComputeWaveEmulConfig, RdnaWaveHelper, U64Helper, WaveSize,
    rdna_bitmask_helpers_wgsl, rdna_compute_wave_emul_wgsl, rdna_gs_alloc_req_wgsl,
    rdna_native_subgroup_wave_wgsl, rdna_wave_helpers_wgsl, u64_helpers_wgsl,
};

// ---------------------------------------------------------------------------
// Synthetic GFX10.3 / RDNA2 instruction encoders
// ---------------------------------------------------------------------------

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

fn enc_sopk(op: u8, sdst: u8, simm16: u16) -> u32 {
    (0b1011u32 << 28)
        | (u32::from(op & 0x1F) << 23)
        | (u32::from(sdst & 0x7F) << 16)
        | u32::from(simm16)
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

fn enc_vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    let w0 = (0b110101u32 << 26) | (u32::from(op & 0x3FF) << 16) | u32::from(vdst);
    let w1 = (u32::from(src2 & 0x1FF) << 18)
        | (u32::from(src1 & 0x1FF) << 9)
        | u32::from(src0 & 0x1FF);
    [w0, w1]
}

#[allow(clippy::too_many_arguments)]
fn enc_mubuf(
    op: u8,
    offset: u16,
    offen: bool,
    idxen: bool,
    glc: bool,
    vaddr: u8,
    vdata: u8,
    srsrc: u8,
    soffset: u8,
) -> [u32; 2] {
    let w0 = (0b111000u32 << 26)
        | (u32::from(op & 0x7F) << 18)
        | (u32::from(glc) << 14)
        | (u32::from(idxen) << 13)
        | (u32::from(offen) << 12)
        | u32::from(offset & 0xFFF);
    let w1 = (u32::from(soffset) << 24)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
    [w0, w1]
}

#[allow(clippy::too_many_arguments)]
fn enc_mimg(
    op: u8,
    dmask: u8,
    dim: u8,
    glc: bool,
    vaddr: u8,
    vdata: u8,
    srsrc: u8,
    ssamp: u8,
) -> [u32; 2] {
    let op_lo = u32::from(op & 0x7F);
    let op_hi = u32::from((op >> 7) & 0x1);
    let w0 = (0b111100u32 << 26)
        | (op_lo << 18)
        | (u32::from(glc) << 13)
        | (u32::from(dmask & 0xF) << 8)
        | (u32::from(dim & 0x7) << 3)
        | op_hi;
    let w1 = (u32::from(ssamp & 0x1F) << 21)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
    [w0, w1]
}

#[allow(clippy::too_many_arguments)]
fn enc_flat(
    seg: u8,
    op: u8,
    offset: i16,
    glc: bool,
    addr: u8,
    data: u8,
    saddr: u8,
    vdst: u8,
) -> [u32; 2] {
    let w0 = (0b110111u32 << 26)
        | (u32::from(op & 0x7F) << 18)
        | (u32::from(glc) << 16)
        | (u32::from(seg & 0x3) << 14)
        | u32::from((offset as u16) & 0xFFF);
    let w1 = (u32::from(vdst) << 24)
        | (u32::from(saddr & 0x7F) << 16)
        | (u32::from(data) << 8)
        | u32::from(addr);
    [w0, w1]
}

fn enc_ds(
    op: u8,
    offset0: u8,
    offset1: u8,
    addr: u8,
    data0: u8,
    data1: u8,
    vdst: u8,
) -> [u32; 2] {
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

fn enc_exp(target: u8, en: u8, done: bool, vsrc: [u8; 4]) -> [u32; 2] {
    let w0 = (0b111110u32 << 26)
        | (u32::from(done) << 11)
        | (u32::from(target & 0x3F) << 4)
        | u32::from(en & 0xF);
    let w1 = u32::from(vsrc[0])
        | (u32::from(vsrc[1]) << 8)
        | (u32::from(vsrc[2]) << 16)
        | (u32::from(vsrc[3]) << 24);
    [w0, w1]
}

fn validate_wgsl(src: &str, caps: Capabilities) {
    let module = naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|err| panic!("WGSL parse failed:\n{}\n--- source ---\n{src}", err.emit_to_string(src)));
    let mut validator = Validator::new(ValidationFlags::all(), caps);
    if caps.contains(Capabilities::SUBGROUP) {
        validator
            .subgroup_stages(ShaderStages::COMPUTE | ShaderStages::FRAGMENT)
            .subgroup_operations(SubgroupOperationSet::all());
    }
    if let Err(err) = validator.validate(&module) {
        panic!(
            "WGSL validation failed (caps={caps:?}):\n{}\n--- source ---\n{src}",
            err.emit_to_string(src)
        );
    }
}

// ---------------------------------------------------------------------------
// 1. RDNA2 Decoder: image_atomic_* & buffer_atomic_* & global/flat atomics
// ---------------------------------------------------------------------------

#[test]
fn decodes_all_gfx10_image_atomic_variants_with_metadata() {
    let cases: &[(u8, &str, Rdna2AtomicOp, u8, bool, bool)] = &[
        (0x0F, "image_atomic_swap", Rdna2AtomicOp::Swap, 0x1, false, false),
        (0x10, "image_atomic_cmpswap", Rdna2AtomicOp::CmpSwap, 0x3, false, false),
        (0x11, "image_atomic_add", Rdna2AtomicOp::Add, 0x1, false, false),
        (0x12, "image_atomic_sub", Rdna2AtomicOp::Sub, 0x1, false, false),
        (0x14, "image_atomic_smin", Rdna2AtomicOp::SMin, 0x1, false, false),
        (0x15, "image_atomic_umin", Rdna2AtomicOp::UMin, 0x1, false, false),
        (0x16, "image_atomic_smax", Rdna2AtomicOp::SMax, 0x1, false, false),
        (0x17, "image_atomic_umax", Rdna2AtomicOp::UMax, 0x1, false, false),
        (0x18, "image_atomic_and", Rdna2AtomicOp::And, 0x1, false, false),
        (0x19, "image_atomic_or", Rdna2AtomicOp::Or, 0x1, false, false),
        (0x1A, "image_atomic_xor", Rdna2AtomicOp::Xor, 0x1, false, false),
        (0x1B, "image_atomic_inc", Rdna2AtomicOp::Inc, 0x1, false, false),
        (0x1C, "image_atomic_dec", Rdna2AtomicOp::Dec, 0x1, false, false),
        (0x1D, "image_atomic_fcmpswap", Rdna2AtomicOp::FCmpSwap, 0x3, false, true),
        (0x1E, "image_atomic_fmin", Rdna2AtomicOp::FMin, 0x1, false, true),
        (0x1F, "image_atomic_fmax", Rdna2AtomicOp::FMax, 0x1, false, true),
        // 64-bit image atomic (dmask=0x3 for non-cmpswap, dmask=0xF for cmpswap)
        (0x11, "image_atomic_add", Rdna2AtomicOp::Add, 0x3, true, false),
        (0x10, "image_atomic_cmpswap", Rdna2AtomicOp::CmpSwap, 0xF, true, false),
    ];

    let mut words = Vec::new();
    for &(op, _, _, dmask, _, _) in cases {
        words.extend_from_slice(&enc_mimg(op, dmask, 1, true, 2, 4, 2, 0));
    }
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let insts = decode_program(&bytes);
    assert_eq!(insts.len(), cases.len() + 1);

    for (idx, &(op, expected_mnem, expected_op, dmask, expected_64, expected_float)) in
        cases.iter().enumerate()
    {
        let inst = &insts[idx];
        assert_eq!(inst.encoding, Rdna2Encoding::Mimg);
        assert_eq!(inst.mnemonic, expected_mnem, "mimg op {op:#x}");
        assert!(inst.is_atomic());
        let info = inst.atomic_info().expect("atomic info");
        assert_eq!(info.domain, Rdna2AtomicDomain::Image);
        assert_eq!(info.op, expected_op);
        assert_eq!(info.is_64bit, expected_64, "op {op:#x} dmask {dmask:#x}");
        assert_eq!(info.is_float, expected_float);
        assert!(info.returns_value);
        assert_eq!(info.data_vgpr_count, dmask.count_ones() as u8);
    }

    let summary = analyze_instructions(&bytes, &insts);
    assert_eq!(summary.inferred_stage, Rdna2InferredStage::Compute);
    assert_eq!(summary.resource_refs.len(), cases.len());
    assert!(summary.resource_refs.iter().all(|r| r.ssamp_sgpr.is_none()));
}

#[test]
fn decodes_all_gfx10_buffer_atomic_32_and_64_variants() {
    let cases_32: &[(u8, &str, Rdna2AtomicOp, bool, u8)] = &[
        (0x30, "buffer_atomic_swap", Rdna2AtomicOp::Swap, false, 1),
        (0x31, "buffer_atomic_cmpswap", Rdna2AtomicOp::CmpSwap, false, 2),
        (0x32, "buffer_atomic_add", Rdna2AtomicOp::Add, false, 1),
        (0x33, "buffer_atomic_sub", Rdna2AtomicOp::Sub, false, 1),
        (0x34, "buffer_atomic_csub", Rdna2AtomicOp::CSub, false, 1),
        (0x35, "buffer_atomic_smin", Rdna2AtomicOp::SMin, false, 1),
        (0x36, "buffer_atomic_umin", Rdna2AtomicOp::UMin, false, 1),
        (0x37, "buffer_atomic_smax", Rdna2AtomicOp::SMax, false, 1),
        (0x38, "buffer_atomic_umax", Rdna2AtomicOp::UMax, false, 1),
        (0x39, "buffer_atomic_and", Rdna2AtomicOp::And, false, 1),
        (0x3A, "buffer_atomic_or", Rdna2AtomicOp::Or, false, 1),
        (0x3B, "buffer_atomic_xor", Rdna2AtomicOp::Xor, false, 1),
        (0x3C, "buffer_atomic_inc", Rdna2AtomicOp::Inc, false, 1),
        (0x3D, "buffer_atomic_dec", Rdna2AtomicOp::Dec, false, 1),
        (0x3E, "buffer_atomic_fcmpswap", Rdna2AtomicOp::FCmpSwap, true, 2),
        (0x3F, "buffer_atomic_fmin", Rdna2AtomicOp::FMin, true, 1),
        (0x40, "buffer_atomic_fmax", Rdna2AtomicOp::FMax, true, 1),
    ];

    let cases_64: &[(u8, &str, Rdna2AtomicOp, bool, u8)] = &[
        (0x50, "buffer_atomic_swap_x2", Rdna2AtomicOp::Swap, false, 2),
        (0x51, "buffer_atomic_cmpswap_x2", Rdna2AtomicOp::CmpSwap, false, 4),
        (0x52, "buffer_atomic_add_x2", Rdna2AtomicOp::Add, false, 2),
        (0x53, "buffer_atomic_sub_x2", Rdna2AtomicOp::Sub, false, 2),
        (0x55, "buffer_atomic_smin_x2", Rdna2AtomicOp::SMin, false, 2),
        (0x56, "buffer_atomic_umin_x2", Rdna2AtomicOp::UMin, false, 2),
        (0x57, "buffer_atomic_smax_x2", Rdna2AtomicOp::SMax, false, 2),
        (0x58, "buffer_atomic_umax_x2", Rdna2AtomicOp::UMax, false, 2),
        (0x59, "buffer_atomic_and_x2", Rdna2AtomicOp::And, false, 2),
        (0x5A, "buffer_atomic_or_x2", Rdna2AtomicOp::Or, false, 2),
        (0x5B, "buffer_atomic_xor_x2", Rdna2AtomicOp::Xor, false, 2),
        (0x5C, "buffer_atomic_inc_x2", Rdna2AtomicOp::Inc, false, 2),
        (0x5D, "buffer_atomic_dec_x2", Rdna2AtomicOp::Dec, false, 2),
        (0x5E, "buffer_atomic_fcmpswap_x2", Rdna2AtomicOp::FCmpSwap, true, 4),
        (0x5F, "buffer_atomic_fmin_x2", Rdna2AtomicOp::FMin, true, 2),
        (0x60, "buffer_atomic_fmax_x2", Rdna2AtomicOp::FMax, true, 2),
    ];

    let mut words = Vec::new();
    for &(op, _, _, _, _) in cases_32 {
        words.extend_from_slice(&enc_mubuf(op, 16, true, false, true, 1, 10, 2, 128));
    }
    for &(op, _, _, _, _) in cases_64 {
        words.extend_from_slice(&enc_mubuf(op, 0, true, false, false, 1, 20, 2, 128));
    }
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let insts = decode_program(&bytes);
    assert_eq!(insts.len(), cases_32.len() + cases_64.len() + 1);

    for (i, &(op, mnem, kind, is_flt, regs)) in cases_32.iter().enumerate() {
        let inst = &insts[i];
        assert_eq!(inst.encoding, Rdna2Encoding::Mubuf);
        assert_eq!(inst.mnemonic, mnem, "mubuf32 op {op:#x}");
        let info = inst.atomic_info().expect("atomic info");
        assert_eq!(info.domain, Rdna2AtomicDomain::Buffer);
        assert_eq!(info.op, kind);
        assert!(!info.is_64bit);
        assert_eq!(info.is_float, is_flt);
        assert!(info.returns_value);
        assert_eq!(info.data_vgpr_count, regs);
    }

    let base = cases_32.len();
    for (i, &(op, mnem, kind, is_flt, regs)) in cases_64.iter().enumerate() {
        let inst = &insts[base + i];
        assert_eq!(inst.encoding, Rdna2Encoding::Mubuf);
        assert_eq!(inst.mnemonic, mnem, "mubuf64 op {op:#x}");
        let info = inst.atomic_info().expect("atomic info");
        assert_eq!(info.domain, Rdna2AtomicDomain::Buffer);
        assert_eq!(info.op, kind);
        assert!(info.is_64bit);
        assert_eq!(info.is_float, is_flt);
        assert!(!info.returns_value);
        assert_eq!(info.data_vgpr_count, regs);
    }

    // Verify 64-bit cmpswap_x2 v20..v23 raises max_vgpr to 23.
    let summary = analyze_instructions(&bytes, &insts);
    assert_eq!(summary.max_vgpr, Some(23));
    assert_eq!(summary.inferred_stage, Rdna2InferredStage::Compute);
}

#[test]
fn decodes_global_and_lds_atomics_and_wide_lds_wave_ops() {
    let mut words = Vec::new();
    // Wide LDS ops: ds_write_b96, ds_write_b128, ds_read_b96, ds_read_b128
    words.extend_from_slice(&enc_ds(0xDE, 16, 0, 1, 4, 0, 0)); // ds_write_b96 v1, v[4:6] offset:16
    words.extend_from_slice(&enc_ds(0xDF, 32, 0, 1, 8, 0, 0)); // ds_write_b128 v1, v[8:11] offset:32
    words.extend_from_slice(&enc_ds(0xFE, 16, 0, 1, 0, 0, 12)); // ds_read_b96 v[12:14], v1 offset:16
    words.extend_from_slice(&enc_ds(0xFF, 32, 0, 1, 0, 0, 16)); // ds_read_b128 v[16:19], v1 offset:32
    // Wave-routing DS ops (canonical GFX10.3 opcodes):
    // ds_swizzle_b32 QDMode reverse quad: offset16 = 0x801B (lanes [3,2,1,0])
    words.extend_from_slice(&enc_ds(0x35, 0x1B, 0x80, 2, 0, 0, 20));
    // ds_swizzle_b32 BitMode butterfly XOR 4: and=0x1F, or=0, xor=4 -> (4 << 10) | 0x1F = 0x101F
    words.extend_from_slice(&enc_ds(0x35, 0x1F, 0x10, 2, 0, 0, 21));
    // ds_permute_b32 (0xB2) & ds_bpermute_b32 (0xB3)
    words.extend_from_slice(&enc_ds(0xB2, 0, 0, 3, 20, 0, 22));
    words.extend_from_slice(&enc_ds(0xB3, 0, 0, 3, 21, 0, 23));
    // LDS atomic & GLOBAL atomic
    words.extend_from_slice(&enc_ds(0x20, 4, 0, 1, 2, 0, 24)); // ds_add_rtn_u32
    words.extend_from_slice(&enc_flat(2, 0x35, 0, true, 0, 2, 0x7F, 25)); // global_atomic_smin
    words.extend_from_slice(&enc_flat(2, 0x58, 0, false, 0, 26, 0x7F, 0)); // global_atomic_umax_x2
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let insts = decode_program(&bytes);
    assert_eq!(insts.len(), 12);

    assert_eq!(insts[0].mnemonic, "ds_write_b96");
    assert_eq!(insts[0].format_text(), "ds_write_b96 v1, v[4:6] offset:16");
    let ds0 = insts[0].ds_op_info().expect("ds info");
    assert_eq!(ds0.kind, Rdna2DsOpKind::Write);
    assert_eq!(ds0.data_vgprs, 3);
    assert_eq!(ds0.offset16, 16);

    assert_eq!(insts[1].mnemonic, "ds_write_b128");
    assert_eq!(insts[1].format_text(), "ds_write_b128 v1, v[8:11] offset:32");
    assert_eq!(insts[1].ds_op_info().unwrap().data_vgprs, 4);

    assert_eq!(insts[2].mnemonic, "ds_read_b96");
    assert_eq!(insts[2].format_text(), "ds_read_b96 v[12:14], v1 offset:16");
    assert_eq!(insts[2].ds_op_info().unwrap().kind, Rdna2DsOpKind::Read);
    assert_eq!(insts[2].ds_op_info().unwrap().data_vgprs, 3);

    assert_eq!(insts[3].mnemonic, "ds_read_b128");
    assert_eq!(insts[3].format_text(), "ds_read_b128 v[16:19], v1 offset:32");
    assert_eq!(insts[3].ds_op_info().unwrap().data_vgprs, 4);

    // QDMode swizzle
    assert_eq!(insts[4].mnemonic, "ds_swizzle_b32");
    let ds4 = insts[4].ds_op_info().unwrap();
    assert_eq!(ds4.kind, Rdna2DsOpKind::Swizzle);
    assert!(ds4.is_wave_cross_lane);
    assert_eq!(
        ds4.swizzle_mode,
        Some(Rdna2SwizzleMode::QuadPerm {
            lanes: [3, 2, 1, 0]
        })
    );
    assert_eq!(ds4.swizzle_mode.unwrap().source_lane(4), 7);
    assert_eq!(ds4.swizzle_mode.unwrap().source_lane(7), 4);
    assert_eq!(
        insts[4].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::DsSwizzle
    );

    // BitMode butterfly swizzle
    let ds5 = insts[5].ds_op_info().unwrap();
    assert_eq!(
        ds5.swizzle_mode,
        Some(Rdna2SwizzleMode::BitMode {
            and_mask: 0x1F,
            or_mask: 0,
            xor_mask: 4,
        })
    );
    assert_eq!(ds5.swizzle_mode.unwrap().source_lane(0), 4);
    assert_eq!(ds5.swizzle_mode.unwrap().source_lane(37), 33);

    // ds_permute_b32 & ds_bpermute_b32
    assert_eq!(insts[6].mnemonic, "ds_permute_b32");
    assert_eq!(insts[6].ds_op_info().unwrap().kind, Rdna2DsOpKind::Permute);
    assert_eq!(
        insts[6].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::DsPermute
    );
    assert_eq!(insts[7].mnemonic, "ds_bpermute_b32");
    assert_eq!(insts[7].ds_op_info().unwrap().kind, Rdna2DsOpKind::BPermute);
    assert_eq!(
        insts[7].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::DsBPermute
    );

    // LDS atomic & GLOBAL atomics
    assert_eq!(insts[8].mnemonic, "ds_add_rtn_u32");
    let lds_at = insts[8].atomic_info().expect("lds atomic");
    assert_eq!(lds_at.domain, Rdna2AtomicDomain::Lds);
    assert_eq!(lds_at.op, Rdna2AtomicOp::Add);
    assert!(lds_at.returns_value);

    assert_eq!(insts[9].mnemonic, "global_atomic_smin");
    let g_at = insts[9].atomic_info().expect("global atomic");
    assert_eq!(g_at.domain, Rdna2AtomicDomain::Global);
    assert_eq!(g_at.op, Rdna2AtomicOp::SMin);
    assert!(g_at.returns_value);

    assert_eq!(insts[10].mnemonic, "global_atomic_umax_x2");
    let g64_at = insts[10].atomic_info().expect("global64 atomic");
    assert_eq!(g64_at.domain, Rdna2AtomicDomain::Global);
    assert_eq!(g64_at.op, Rdna2AtomicOp::UMax);
    assert!(g64_at.is_64bit);

    let summary = analyze_instructions(&bytes, &insts);
    assert!(summary.has_lds);
    assert_eq!(summary.max_vgpr, Some(27)); // global_atomic_umax_x2 reads v[26:27]
}

#[test]
fn decodes_wave_mbcnt_bitmask_readlane_and_sendmsg_gs_alloc_req() {
    // Also verify exact LLVM GFX10 MC bytes for v_mbcnt_lo/hi_u32_b32:
    // v_mbcnt_lo_u32_b32 v5, v1, v2 -> [0x05,0x00,0x65,0xd7,0x01,0x05,0x02,0x02]
    // v_mbcnt_hi_u32_b32 v5, v1, v2 -> [0x05,0x00,0x66,0xd7,0x01,0x05,0x02,0x02]
    let llvm_mbcnt_bytes: [u8; 16] = [
        0x05, 0x00, 0x65, 0xd7, 0x01, 0x05, 0x02, 0x02, // v_mbcnt_lo_u32_b32 v5, v1, v2
        0x05, 0x00, 0x66, 0xd7, 0x01, 0x05, 0x02, 0x02, // v_mbcnt_hi_u32_b32 v5, v1, v2
    ];
    let mbcnt_insts = decode_program(&llvm_mbcnt_bytes);
    assert_eq!(mbcnt_insts.len(), 2);
    assert_eq!(mbcnt_insts[0].format_text(), "v_mbcnt_lo_u32_b32 v5, v1, v2");
    assert_eq!(mbcnt_insts[1].format_text(), "v_mbcnt_hi_u32_b32 v5, v1, v2");
    assert_eq!(
        mbcnt_insts[0].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::MbcntLo
    );
    assert_eq!(
        mbcnt_insts[1].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::MbcntHi
    );
    assert!(mbcnt_insts[1].wave_op_info().unwrap().is_wave64_mask);
    // v_mbcnt writes a single 32-bit VGPR (v5, not v[5:6])
    let mbcnt_summary = analyze_instructions(&llvm_mbcnt_bytes, &mbcnt_insts);
    assert_eq!(mbcnt_summary.max_vgpr, Some(5));

    let mut words = Vec::new();
    // SOP1 wave bitmask ops
    words.push(enc_sop1(0x10, 2, 10)); // s_bcnt1_i32_b64 s2, s[10:11]
    words.push(enc_sop1(0x11, 3, 12)); // s_ff0_i32_b32 s3, s12
    words.push(enc_sop1(0x12, 4, 14)); // s_ff0_i32_b64 s4, s[14:15]
    words.push(enc_sop1(0x13, 5, 16)); // s_ff1_i32_b32 s5, s16
    words.push(enc_sop1(0x14, 6, 18)); // s_ff1_i32_b64 s6, s[18:19]
    words.push(enc_sop1(0x15, 7, 20)); // s_flbit_i32_b32 s7, s20
    words.push(enc_sop1(0x16, 8, 22)); // s_flbit_i32_b64 s8, s[22:23]
    // Lane broadcast/read ops
    words.push(enc_vop1(0x02, 24, 256 + 3)); // v_readfirstlane_b32 s24, v3
    words.extend_from_slice(&enc_vop3(0x182, 25, 256 + 4, 0, 0)); // v_readfirstlane_b32 (VOP3) s25, v4
    words.extend_from_slice(&enc_vop3(0x360, 26, 256 + 5, 128 + 7, 0)); // v_readlane_b32 s26, v5, 7
    // s_sendmsg(MSG_GS_ALLOC_REQ) + mesh exports + s_sendmsg(MSG_GS_DONE)
    words.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_ALLOC_REQ)));
    words.extend_from_slice(&enc_exp(12, 0xF, false, [0, 1, 2, 3])); // exp pos0
    words.extend_from_slice(&enc_exp(32, 0xF, false, [4, 5, 6, 7])); // exp param0
    words.extend_from_slice(&enc_exp(20, 0x1, true, [8, 0, 0, 0])); // exp prim done
    words.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_DONE) | (2 << 4) | (1 << 8)));
    words.push(enc_sopp(0x01, 0));

    let bytes = words_to_bytes(&words);
    let insts = decode_program(&bytes);
    assert_eq!(insts[0].format_text(), "s_bcnt1_i32_b64 s2, s[10:11]");
    assert_eq!(insts[1].format_text(), "s_ff0_i32_b32 s3, s12");
    assert_eq!(insts[2].format_text(), "s_ff0_i32_b64 s4, s[14:15]");
    assert_eq!(insts[3].format_text(), "s_ff1_i32_b32 s5, s16");
    assert_eq!(insts[4].format_text(), "s_ff1_i32_b64 s6, s[18:19]");
    assert_eq!(insts[5].format_text(), "s_flbit_i32_b32 s7, s20");
    assert_eq!(insts[6].format_text(), "s_flbit_i32_b64 s8, s[22:23]");
    assert_eq!(insts[7].format_text(), "v_readfirstlane_b32 s24, v3");
    assert_eq!(insts[8].format_text(), "v_readfirstlane_b32 s25, v4");
    assert_eq!(insts[9].format_text(), "v_readlane_b32 s26, v5, 7");

    assert_eq!(
        insts[0].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::Bcnt1B64
    );
    assert_eq!(
        insts[2].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::Ff0B64
    );
    assert_eq!(
        insts[4].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::Ff1B64
    );
    assert_eq!(
        insts[6].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::FlbitB64
    );
    assert_eq!(
        insts[8].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::ReadFirstLane
    );
    assert_eq!(
        insts[9].wave_op_info().unwrap().kind,
        Rdna2WaveOpKind::ReadLane
    );

    // Verify sendmsg decoding (msg_id, op, stream, GS_ALLOC_REQ, GS_DONE)
    let alloc_msg = insts[10].sendmsg_info().expect("sendmsg alloc");
    assert!(alloc_msg.is_gs_alloc_req);
    assert!(!alloc_msg.is_halt);
    assert_eq!(alloc_msg.msg_id, SENDMSG_ID_GS_ALLOC_REQ);

    let done_msg = insts[14].sendmsg_info().expect("sendmsg done");
    assert!(done_msg.is_gs_done);
    assert_eq!(done_msg.op, 2);
    assert_eq!(done_msg.stream, 1);
    assert_eq!(decode_sendmsg(u16::from(SENDMSG_ID_GS_DONE) | (2 << 4) | (1 << 8), true).is_halt, true);

    // Verify mesh metadata extraction
    let mesh = extract_mesh_metadata(&insts);
    assert!(mesh.is_mesh_shader);
    assert_eq!(mesh.gs_alloc_req_offsets.len(), 1);
    assert_eq!(mesh.gs_done_offsets.len(), 1);
    assert_eq!(mesh.prim_export_offsets.len(), 1);
    assert_eq!(mesh.pos_export_slots, vec![0]);
    assert_eq!(mesh.param_export_slots, vec![0]);
    assert_eq!(mesh.done_export_offsets.len(), 1);
    assert_eq!(mesh.wave_op_count, 10);

    // Verify SGPR high-water tracking captures s23 from s_flbit_i32_b64 s8, s[22:23] and s26 from v_readlane
    let summary = analyze_program(&bytes);
    assert_eq!(summary.max_sgpr, Some(26));
    assert_eq!(summary.exports[2].target, Rdna2ExportTarget::Prim);
}

// ---------------------------------------------------------------------------
// 2. CFG & Structural Analysis: Mesh shaders, barriers, waitcnts, atomics
// ---------------------------------------------------------------------------

#[test]
fn cfg_analyzes_mesh_shader_with_single_lane_gs_alloc_req_dominating_exports() {
    // Classic AMD NGG / mesh shader pattern:
    //   bb0: v_cmp_eq_u32 vcc, 0, v0
    //        s_and_saveexec_b32 s0, vcc_lo
    //        s_cbranch_execz -> bb2 (+2 dwords: skip bb1 = 2 dwords: s_mov_b32 + s_sendmsg)
    //   bb1: s_mov_b32 m0, s4
    //        s_sendmsg(GS_ALLOC_REQ)
    //   bb2: s_or_b32 exec_lo, exec_lo, s0
    //        s_barrier
    //        exp pos0 ...
    //        exp param0 ...
    //        exp prim ... done
    //        s_endpgm
    let mut words = Vec::new();
    words.push(enc_vopc(0xC2, 128, 0)); // 0: v_cmp_eq_u32 vcc, 0, v0
    words.push(enc_sop1(0x3C, 0, 106)); // 1: s_and_saveexec_b32 s0, vcc_lo
    words.push(enc_sopp(0x08, 2)); // 2: s_cbranch_execz -> dword 5 (offset 20)
    words.push(enc_sop1(0x03, 124, 4)); // 3 (bb1): s_mov_b32 m0, s4
    words.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_ALLOC_REQ))); // 4 (bb1): s_sendmsg(GS_ALLOC_REQ)
    words.push(enc_sop1(0x03, 126, 0)); // 5 (bb2): s_mov_b32 exec_lo, s0 (restore)
    words.push(enc_sopp(0x0A, 0)); // 6 (bb2): s_barrier
    words.extend_from_slice(&enc_exp(12, 0xF, false, [0, 1, 2, 3])); // 7..8: exp pos0
    words.extend_from_slice(&enc_exp(32, 0xF, false, [4, 5, 6, 7])); // 9..10: exp param0
    words.extend_from_slice(&enc_exp(20, 0x1, true, [8, 0, 0, 0])); // 11..12: exp prim done
    words.push(enc_sopp(0x01, 0)); // 13: s_endpgm

    let bytes = words_to_bytes(&words);
    let instrs = decode_program(&bytes);
    let cfg = ControlFlowGraph::from_instructions(&instrs);
    assert!(cfg.diagnostics.is_empty());
    assert_eq!(cfg.blocks.len(), 3);

    let doms = cfg.dominators();
    assert!(cfg.dominates(&doms, 0, 1));
    assert!(cfg.dominates(&doms, 0, 2));
    assert!(!cfg.dominates(&doms, 1, 2));

    let analysis = analyze_compute_mesh_cfg(&cfg, &instrs);
    assert!(analysis.is_mesh_shader);
    assert_eq!(analysis.mesh_alloc_points.len(), 1);
    let alloc = &analysis.mesh_alloc_points[0];
    assert_eq!(alloc.block_id, 1);
    assert_eq!(alloc.guard_kind, MeshAllocGuardKind::SingleLaneExecGuard);
    assert!(alloc.dominates_all_exports);
    assert!(!alloc.inside_loop);
    assert!(analysis.all_exports_dominated_by_alloc);
    assert!(analysis.unallocated_export_blocks.is_empty());

    // Barrier is in bb2 (post-merge, after EXEC restore), so it is uniform!
    assert_eq!(analysis.barrier_regions.len(), 1);
    assert!(analysis.barrier_regions[0].in_uniform_control_flow);
    assert!(!analysis.has_non_uniform_barrier);

    let structured = structure_cfg(&cfg);
    assert!(structured.is_fully_structured());
    let summary = summarize_compute_mesh_cfg(&analysis);
    assert!(summary.contains("mesh=true"), "{summary}");
    assert!(summary.contains("exports_dom=true"), "{summary}");
}

#[test]
fn cfg_detects_unallocated_mesh_exports_and_non_uniform_barriers_in_compute_loop() {
    // 1. Mesh shader with an export BEFORE GS_ALLOC_REQ -> unallocated export detected!
    let mut bad_mesh = Vec::new();
    bad_mesh.extend_from_slice(&enc_exp(12, 0xF, false, [0, 1, 2, 3])); // exp pos0 BEFORE alloc!
    bad_mesh.push(enc_sopp(0x10, u16::from(SENDMSG_ID_GS_ALLOC_REQ)));
    bad_mesh.extend_from_slice(&enc_exp(20, 0x1, true, [4, 0, 0, 0])); // exp prim AFTER alloc
    bad_mesh.push(enc_sopp(0x01, 0));

    let bad_instrs = decode_program(&words_to_bytes(&bad_mesh));
    let bad_cfg = ControlFlowGraph::from_instructions(&bad_instrs);
    let bad_analysis = analyze_compute_mesh_cfg(&bad_cfg, &bad_instrs);
    assert!(bad_analysis.is_mesh_shader);
    assert!(!bad_analysis.mesh_alloc_points[0].dominates_all_exports);
    assert!(!bad_analysis.all_exports_dominated_by_alloc);

    // 2. 3D compute culling + lighting loop with atomics, LDS, waitcnts, uniform & non-uniform barriers:
    //   bb0 (preheader): buffer_atomic_add, image_atomic_umax, s_waitcnt vmcnt(0), s_waitcnt_vscnt
    //   bb1 (loop hdr):  ds_write_b128, s_waitcnt lgkmcnt(0), s_barrier (uniform, inside loop),
    //                    s_cbranch_execz -> bb3 (skip divergent bb2)
    //   bb2 (divergent): ds_swizzle_b32, s_barrier (NON-UNIFORM!), ds_add_rtn_u32
    //   bb3 (latch):     global_atomic_add, s_cbranch_scc1 -> bb1 (back-edge)
    //   bb4 (exit):      s_endpgm
    let mut words = Vec::new();
    // bb0 (dwords 0..6):
    words.extend_from_slice(&enc_mubuf(0x32, 0, true, false, true, 0, 1, 1, 128)); // 0..1: buffer_atomic_add
    words.extend_from_slice(&enc_mimg(0x17, 0x1, 1, true, 2, 3, 2, 0)); // 2..3: image_atomic_umax
    words.push(enc_sopp(0x0C, 0x0F70)); // 4: s_waitcnt vmcnt(0)
    words.push(enc_sopk(0x17, 125, 0)); // 5: s_waitcnt_vscnt null, 0
    // bb1 (dwords 6..11, offset 24):
    let bb1_dword = words.len() as i16;
    words.extend_from_slice(&enc_ds(0xDF, 0, 0, 0, 4, 0, 0)); // 6..7: ds_write_b128
    words.push(enc_sopp(0x0C, 0xC07F)); // 8: s_waitcnt lgkmcnt(0)
    words.push(enc_sopp(0x0A, 0)); // 9: s_barrier (uniform, inside loop)
    let cbr_idx = words.len();
    words.push(0); // 10: placeholder for s_cbranch_execz -> bb3
    // bb2 (dwords 11..16):
    words.extend_from_slice(&enc_ds(0x35, 0x1B, 0x80, 1, 0, 0, 8)); // 11..12: ds_swizzle_b32
    words.push(enc_sopp(0x0A, 0)); // 13: s_barrier (NON-UNIFORM, inside execz-guarded bb2!)
    words.extend_from_slice(&enc_ds(0x20, 0, 0, 1, 8, 0, 9)); // 14..15: ds_add_rtn_u32
    // bb3 (dwords 16..19):
    let bb3_dword = words.len() as i16;
    words[cbr_idx] = enc_sopp(0x08, (bb3_dword - (cbr_idx as i16 + 1)) as u16);
    words.extend_from_slice(&enc_flat(2, 0x32, 0, false, 0, 9, 0x7F, 0)); // 16..17: global_atomic_add
    let latch_idx = words.len() as i16;
    words.push(enc_sopp(0x05, (bb1_dword - (latch_idx + 1)) as u16)); // 18: s_cbranch_scc1 -> bb1
    // bb4 (dword 19):
    words.push(enc_sopp(0x01, 0)); // 19: s_endpgm

    let cs_bytes = words_to_bytes(&words);
    let cs_instrs = decode_program(&cs_bytes);
    let cs_cfg = ControlFlowGraph::from_instructions(&cs_instrs);
    assert!(cs_cfg.diagnostics.is_empty());

    let cs_analysis = analyze_compute_mesh_cfg(&cs_cfg, &cs_instrs);
    assert!(!cs_analysis.is_mesh_shader);
    assert_eq!(cs_analysis.total_buffer_atomics, 1);
    assert_eq!(cs_analysis.total_image_atomics, 1);
    assert_eq!(cs_analysis.total_lds_atomics, 1);
    assert_eq!(cs_analysis.total_global_atomics, 1);
    assert_eq!(cs_analysis.total_barriers, 2);
    assert_eq!(cs_analysis.total_waitcnts, 3);
    assert!(cs_analysis.total_lds_ops >= 3);
    assert_eq!(cs_analysis.total_wave_ops, 1);
    assert_eq!(cs_analysis.atomic_blocks.len(), 3); // bb0, bb2, bb3

    assert_eq!(cs_analysis.barrier_regions.len(), 2);
    // Barrier in bb1: inside loop, uniform control flow
    assert!(cs_analysis.barrier_regions[0].inside_loop);
    assert!(cs_analysis.barrier_regions[0].in_uniform_control_flow);
    // Barrier in bb2: inside loop, NON-UNIFORM control flow (guarded by s_cbranch_execz)
    assert!(cs_analysis.barrier_regions[1].inside_loop);
    assert!(!cs_analysis.barrier_regions[1].in_uniform_control_flow);
    assert!(cs_analysis.has_non_uniform_barrier);
}

// ---------------------------------------------------------------------------
// 3. Wave / Subgroup WGSL Emulation & Naga Validation
// ---------------------------------------------------------------------------

#[test]
fn wgsl_bitmask_and_pure_wave_helpers_validate_in_compute_and_fragment() {
    let helpers = format!(
        "{}\n{}\n{}",
        u64_helpers_wgsl("s69_", &U64Helper::ALL),
        rdna_wave_helpers_wgsl("s69_", &RdnaWaveHelper::ALL),
        rdna_bitmask_helpers_wgsl("s69_", &RdnaBitmaskHelper::ALL),
    );
    let body = r#"
    let m64 = vec2<u32>(seed, seed ^ 0x55aa55aau);
    let b1_32 = s69_s_bcnt1_i32_b32(seed);
    let b1_64 = s69_s_bcnt1_i32_b64(m64);
    let ff0_32 = s69_s_ff0_i32_b32(seed);
    let ff0_64 = s69_s_ff0_i32_b64(m64);
    let ff1_32 = s69_s_ff1_i32_b32(seed);
    let ff1_64 = s69_s_ff1_i32_b64(m64);
    let flb_32 = s69_s_flbit_i32_b32(seed);
    let flb_64 = s69_s_flbit_i32_b64(m64);
    let flb_i32 = s69_s_flbit_i32(seed);
    let flb_i64 = s69_s_flbit_i32_i64(m64);
    let mb_lo = s69_mbcnt_lo(m64.x, 0u, seed);
    let mb_hi = s69_mbcnt_hi(m64.y, mb_lo, seed);
    let swz = s69_swizzle_lane(seed, 0x801bu) ^ s69_swizzle_lane(seed, 0x101fu);
    let prm = s69_permute_lane(seed * 4u, 63u);
    let acc = b1_32 ^ b1_64 ^ ff0_32 ^ ff0_64 ^ ff1_32 ^ ff1_64 ^ flb_32 ^ flb_64 ^ flb_i32 ^ flb_i64 ^ mb_hi ^ swz ^ prm;
"#;

    let compute_src = format!(
        "{helpers}\n@group(0) @binding(0) var<storage, read_write> out: array<u32, 64>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(local_invocation_index) seed: u32) {{\n{body}    out[seed] = acc;\n}}\n"
    );
    validate_wgsl(&compute_src, Capabilities::empty());

    let fragment_src = format!(
        "{helpers}\n@fragment\nfn main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {{\n    let seed = u32(pos.x) + u32(pos.y);\n{body}    return vec4<f32>(f32(acc & 255u) / 255.0, 0.0, 0.0, 1.0);\n}}\n"
    );
    validate_wgsl(&fragment_src, Capabilities::empty());
}

#[test]
fn wgsl_compute_wave_emul_library_validates_for_wave32_and_wave64() {
    for (wave_size, wg_size) in [
        (WaveSize::Wave32, 32u32),
        (WaveSize::Wave32, 64u32),
        (WaveSize::Wave64, 64u32),
        (WaveSize::Wave64, 128u32),
    ] {
        let cfg = RdnaComputeWaveEmulConfig::new(wave_size, wg_size).with_prefix("w69_");
        let lib = rdna_compute_wave_emul_wgsl(&cfg);
        let shader = format!(
            "{lib}
@group(0) @binding(0) var<storage, read_write> out: array<u32, {wg_size}>;

@compute @workgroup_size({wg_size})
fn main(@builtin(local_invocation_index) li: u32) {{
    let pred = (li & 3u) == 0u;
    let b32 = w69_wave_ballot32(li, pred);
    let b64 = w69_wave_ballot64(li, pred);
    let cnt64 = w69_s_bcnt1_i32_b64(b64);
    let f0 = w69_s_ff0_i32_b64(b64) ^ w69_s_ff0_i32_b32(b32);
    let f1 = w69_s_ff1_i32_b64(b64) ^ w69_s_ff1_i32_b32(b32);
    let fl = w69_s_flbit_i32_b64(b64) ^ w69_s_flbit_i32_b32(b32);
    let mb_lo = w69_v_mbcnt_lo_u32_b32(b64.x, 0u, li);
    let mb_hi = w69_v_mbcnt_hi_u32_b32(b64.y, mb_lo, li);
    let mb_all = w69_v_mbcnt_u64(b64, li);
    let v = li * 7u + 1u;
    let rfirst = w69_v_readfirstlane_b32(li, v);
    let rlane = w69_v_readlane_b32(li, v, 3u);
    let shuf = w69_wave_shuffle_b32(li, v, (li + 1u) & 31u);
    let shuf_xor = w69_wave_shuffle_xor_b32(li, v, 1u);
    let swz_quad = w69_ds_swizzle_b32(li, v, 0x801bu);
    let swz_fft = w69_ds_swizzle_b32(li, v, 0x101fu);
    let bperm = w69_ds_bpermute_b32(li, ((li ^ 3u) << 2u), v);
    let perm = w69_ds_permute_b32(li, ((li ^ 1u) << 2u), v);
    out[li] = cnt64 ^ f0 ^ f1 ^ fl ^ mb_hi ^ mb_all ^ rfirst ^ rlane ^ shuf ^ shuf_xor ^ swz_quad ^ swz_fft ^ bperm ^ perm;
}}
"
        );
        validate_wgsl(&shader, Capabilities::empty());
    }
}

#[test]
fn wgsl_native_subgroup_and_gs_alloc_req_helpers_validate_with_naga() {
    // 1. Native WebGPU `enable subgroups;` wave library
    let sg_lib = rdna_native_subgroup_wave_wgsl("sg69_");
    let sg_shader = format!(
        "{sg_lib}
@group(0) @binding(0) var<storage, read_write> out: array<u32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(local_invocation_index) li: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {{
    let b64 = sg69_wave_ballot64((li & 1u) == 0u);
    let b32 = sg69_wave_ballot32((li & 3u) == 0u);
    let mb_lo = sg69_v_mbcnt_lo_u32_b32(b64.x, 0u, sg_lane);
    let mb_hi = sg69_v_mbcnt_hi_u32_b32(b64.y, mb_lo, sg_lane);
    let rfirst = sg69_v_readfirstlane_b32(li);
    let rlane = sg69_v_readlane_b32(li, 0u);
    let shuf = sg69_wave_shuffle_b32(li, sg_lane ^ 1u);
    let shuf_x = sg69_wave_shuffle_xor_b32(li, 2u);
    let swz = sg69_ds_swizzle_b32(sg_lane, li, 0x801bu);
    let bperm = sg69_ds_bpermute_b32((sg_lane ^ 1u) << 2u, li, 63u);
    let bits = sg69_s_bcnt1_i32_b64(b64) ^ sg69_s_ff1_i32_b64(b64) ^ sg69_s_flbit_i32_b32(b32);
    out[li] = mb_hi ^ rfirst ^ rlane ^ shuf ^ shuf_x ^ swz ^ bperm ^ bits;
}}
"
    );
    validate_wgsl(&sg_shader, Capabilities::SUBGROUP);

    // 2. s_sendmsg(GS_ALLOC_REQ) meshlet allocation WGSL prelude (baseline Capabilities::empty())
    let gs_lib = rdna_gs_alloc_req_wgsl("ms69_", 0, 0, 64, 126, 3);
    let gs_shader = format!(
        "{gs_lib}
@group(0) @binding(1) var<storage, read_write> out_meta: array<u32, 3>;

@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) li: u32) {{
    // Pack m0: 48 vertices in [10:0], 64 primitives in [22:12]
    let m0 = 48u | (64u << 12u);
    let alloc = ms69_gs_alloc_req(li, m0);
    if (li == 0u) {{
        out_meta[0] = alloc.vertex_count;
        out_meta[1] = alloc.primitive_count;
        out_meta[2] = alloc.primitive_base;
    }}
}}
"
    );
    validate_wgsl(&gs_shader, Capabilities::empty());
}

#[test]
fn swizzle_offset_decoder_matches_quad_butterfly_fft_and_broadcast_modes() {
    // Quad permute: swap adjacent pairs [1, 0, 3, 2] -> 0x8000 | (1) | (0<<2) | (3<<4) | (2<<6) = 0x80B1
    let quad_swap = decode_ds_swizzle_offset(0x80B1);
    assert_eq!(
        quad_swap,
        Rdna2SwizzleMode::QuadPerm {
            lanes: [1, 0, 3, 2]
        }
    );
    for lane in 0..64u32 {
        assert_eq!(quad_swap.source_lane(lane), lane ^ 1);
    }

    // Broadcast lane 0 within each 32-lane half: and_mask=0, or_mask=0, xor_mask=0 -> offset16 = 0x0000
    let bcast0 = decode_ds_swizzle_offset(0x0000);
    for lane in 0..32u32 {
        assert_eq!(bcast0.source_lane(lane), 0);
    }
    for lane in 32..64u32 {
        assert_eq!(bcast0.source_lane(lane), 32);
    }

    // FFT / butterfly XOR 16: and_mask=31 (0x1F), or_mask=0, xor_mask=16 -> (16 << 10) | 0x1F = 0x401F
    let fft16 = decode_ds_swizzle_offset(0x401F);
    for lane in 0..64u32 {
        assert_eq!(fft16.source_lane(lane), lane ^ 16);
    }
}
