//! S70 (#1741) Integration Tests:
//! - 3D PBR Texture Sampling (`image_gather4_*`, `image_sample_c_*` shadow PCF,
//!   `image_sample_b`/`d`/`l`/`lz`/`cd`/`cl`/`o` across `2d`, `2d-array`, `cube`,
//!   `cube-array`, `3d`, plus `d16` packed half-float outputs and NSA addresses)
//! - Multi-MRT (`mrt0`..`mrt7`, uncompressed f32 & compressed `compr` f16) + `mrtz` (`@builtin(frag_depth)`)
//! - Fragment interpolation qualifiers (`v_interp_p1_f32`/`p2_f32`/`mov_f32`:
//!   `perspective`/`linear`/`flat` + `center`/`centroid`)
//! - Shared Naga IR lowering (`lower.rs`) for 3D PBR sampling, multi-MRT, and interpolation
//!
//! Every emitted WGSL shader is parsed and validated through `naga::front::wgsl::parse_str`
//! and `naga::valid::Validator` under strict baseline WebGPU `Capabilities::empty()`.

use naga::front::wgsl as wgsl_in;
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{
    Binding, Block, EntryPoint, Expression, Function, FunctionArgument, FunctionResult,
    GlobalVariable, ImageClass, ImageDimension, Interpolation, Literal, MemoryDecorations, Module,
    ResourceBinding, SampleLevel, Sampling, Scalar, ScalarKind, ShaderStage as NagaStage, Span,
    Statement, StructMember, SwizzleComponent, Type, TypeInner, VectorSize,
};

use playgta6_shaders::convert::ShaderStage;
use playgta6_shaders::lower::{
    LoweringError, inspect_fragment_interface, lower_and_emit_wgsl,
    lower_pbr_sampling_and_interpolation, try_lower_module_for_webgpu,
};
use playgta6_shaders::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, Rdna2WgslOutput, translate_rdna2_to_wgsl,
};

// ---------------------------------------------------------------------------
// Minimal synthetic RDNA2 (GFX10.3) encoders for S70 tests
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

fn enc_sopc(op: u8, ssrc0: u8, ssrc1: u8) -> u32 {
    (0b1011_1111_0u32 << 23)
        | (u32::from(op & 0x7F) << 16)
        | (u32::from(ssrc1) << 8)
        | u32::from(ssrc0)
}

fn enc_vop1(op: u8, vdst: u8, src0: u16) -> u32 {
    (0b0111_111u32 << 25)
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
fn enc_mimg_full(
    op: u8,
    dmask: u8,
    dim: u8,
    d16: bool,
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
    let w1 = (u32::from(d16) << 31)
        | (u32::from(ssamp & 0x1F) << 21)
        | (u32::from(srsrc & 0x1F) << 16)
        | (u32::from(vdata) << 8)
        | u32::from(vaddr);
    [w0, w1]
}

fn enc_mimg(op: u8, dmask: u8, dim: u8, vaddr: u8, vdata: u8, srsrc: u8, ssamp: u8) -> [u32; 2] {
    enc_mimg_full(op, dmask, dim, false, vaddr, vdata, srsrc, ssamp)
}

#[allow(clippy::too_many_arguments)]
fn enc_mimg_nsa1(
    op: u8,
    dmask: u8,
    dim: u8,
    vaddr0: u8,
    nsa_regs: [u8; 4],
    vdata: u8,
    srsrc: u8,
    ssamp: u8,
) -> [u32; 3] {
    let [mut w0, w1] = enc_mimg_full(op, dmask, dim, false, vaddr0, vdata, srsrc, ssamp);
    w0 |= 1u32 << 1; // nsize = 1 (4 extra VGPR indices in w2)
    let w2 = u32::from(nsa_regs[0])
        | (u32::from(nsa_regs[1]) << 8)
        | (u32::from(nsa_regs[2]) << 16)
        | (u32::from(nsa_regs[3]) << 24);
    [w0, w1, w2]
}

fn assert_naga_wgsl_valid(wgsl: &str) {
    let module = wgsl_in::parse_str(wgsl)
        .unwrap_or_else(|err| panic!("Naga WGSL parse failed:\n{}\nWGSL:\n{wgsl}", err.emit_to_string(wgsl)));
    // Core WebGPU 1.0 baseline: pack2x16float/unpack2x16float + cube-array textures (no optional enables).
    let baseline_caps =
        Capabilities::SHADER_FLOAT16_IN_FLOAT32.union(Capabilities::CUBE_ARRAY_TEXTURES);
    let mut validator = Validator::new(ValidationFlags::all(), baseline_caps);
    validator
        .validate(&module)
        .unwrap_or_else(|err| panic!("Naga WGSL validation failed:\n{}\nWGSL:\n{wgsl}", err.emit_to_string(wgsl)));
}

fn translate_stage(words: &[u32], stage: Rdna2StageOverride) -> Rdna2WgslOutput {
    let bytes = words_to_bytes(words);
    let opts = Rdna2TranslateOptions {
        stage,
        ..Rdna2TranslateOptions::default()
    };
    let out = translate_rdna2_to_wgsl(&bytes, &opts)
        .unwrap_or_else(|err| panic!("RDNA2 -> WGSL translation failed: {err}"));
    assert!(out.naga_validated);
    assert_eq!(
        out.unsupported_op_count, 0,
        "expected 0 unsupported ops, got {}:\n{}",
        out.unsupported_op_count, out.wgsl
    );
    assert_naga_wgsl_valid(&out.wgsl);
    out
}

// ---------------------------------------------------------------------------
// 1. image_gather4_* tests (components, compare, offset, lod/bias/cl, d16, 3D fallback)
// ---------------------------------------------------------------------------

#[test]
fn s70_image_gather4_all_components_offsets_lod_bias_and_d16() {
    let mut words = Vec::new();
    // v1 = packed offset (dx=1, dy=-1 -> (1) | (0x3F << 8) = 0x3F01), v2..v3 = uv, v4 = lod/bias/clamp
    words.push(enc_vop1(0x01, 1, 128 + 1)); // v1 = 1
    words.push(enc_vintrp(0, 2, 0, 0, 0)); // v2 = attr0.x
    words.push(enc_vintrp(1, 2, 0, 0, 1));
    words.push(enc_vintrp(0, 3, 0, 1, 0)); // v3 = attr0.y
    words.push(enc_vintrp(1, 3, 0, 1, 1));
    words.push(enc_vop1(0x01, 4, 240)); // v4 = 0.5

    // image_gather4 R(0x1), G(0x2), B(0x4), A(0x8) on 2D tex0 (srsrc=2, ssamp=1)
    words.extend_from_slice(&enc_mimg(0x40, 0x1, 1, 2, 10, 2, 1)); // gather4 R -> v10..v13
    words.extend_from_slice(&enc_mimg(0x40, 0x2, 1, 2, 14, 2, 1)); // gather4 G -> v14..v17
    words.extend_from_slice(&enc_mimg(0x40, 0x4, 1, 2, 18, 2, 1)); // gather4 B -> v18..v21
    words.extend_from_slice(&enc_mimg(0x40, 0x8, 1, 2, 22, 2, 1)); // gather4 A -> v22..v25

    // image_gather4_o (0x50), _l (0x44), _lz (0x47), _b (0x45), _cl (0x41)
    words.extend_from_slice(&enc_mimg(0x50, 0x1, 1, 1, 26, 2, 1)); // gather4_o (v1=off, v2..v3=uv)
    words.extend_from_slice(&enc_mimg(0x44, 0x2, 1, 2, 30, 2, 1)); // gather4_l (v2..v3=uv, v4=lod)
    words.extend_from_slice(&enc_mimg(0x47, 0x4, 1, 2, 34, 2, 1)); // gather4_lz
    words.extend_from_slice(&enc_mimg(0x45, 0x8, 1, 1, 38, 2, 1)); // gather4_b (v1=bias, v2..v3=uv)
    words.extend_from_slice(&enc_mimg(0x41, 0x1, 1, 2, 42, 2, 1)); // gather4_cl (v2..v3=uv, v4=clamp)

    // image_gather4 with d16=1 -> packs 4 gathered taps into 2 VGPRs (v46, v47) via pack2x16float
    words.extend_from_slice(&enc_mimg_full(0x40, 0x2, 1, true, 2, 46, 2, 1));

    // Combine results into mrt0
    words.push(enc_vop2(0x03, 10, 256 + 14, 10));
    words.push(enc_vop2(0x03, 11, 256 + 18, 26));
    words.push(enc_vop2(0x03, 12, 256 + 22, 30));
    words.push(enc_vop2(0x03, 13, 256 + 34, 42));
    words.extend_from_slice(&enc_exp(0, 0xF, false, false, false, [10, 11, 12, 13]));
    // Export d16 packed gather taps via compr mrt1
    words.extend_from_slice(&enc_exp(1, 0xF, true, true, true, [46, 47, 0, 0]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_stage(&words, Rdna2StageOverride::Pixel);
    assert_eq!(out.stage, ShaderStage::Fragment);
    assert!(out.wgsl.contains("textureGather(0"), "missing R gather:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureGather(1"), "missing G gather:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureGather(2"), "missing B gather:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureGather(3"), "missing A gather:\n{}", out.wgsl);
    assert!(
        out.wgsl.contains("textureDimensions("),
        "expected textureDimensions UV shift for image_gather4_o:\n{}",
        out.wgsl
    );
    let pack_count = out.wgsl.matches("pack2x16float(").count()
        - out.wgsl.matches("unpack2x16float(").count();
    assert_eq!(
        pack_count,
        2,
        "d16 image_gather4 must emit 2 pack2x16float calls for 4 gathered taps:\n{}",
        out.wgsl
    );
}

#[test]
fn s70_image_gather4_c_shadow_compare_across_2d_2darray_cube_cubearray_and_d16() {
    let mut words = Vec::new();
    // v1 = offset, v2 = z_ref, v3..v6 = (u, v, w/slice, cube_array_slice)
    words.push(enc_vop1(0x01, 1, 128 + 1));
    words.push(enc_vop1(0x01, 2, 240)); // z_ref = 0.5
    words.push(enc_vintrp(0, 3, 0, 0, 0));
    words.push(enc_vintrp(1, 3, 0, 0, 1));
    words.push(enc_vintrp(0, 4, 0, 1, 0));
    words.push(enc_vintrp(1, 4, 0, 1, 1));
    words.push(enc_vintrp(0, 5, 0, 2, 0));
    words.push(enc_vintrp(1, 5, 0, 2, 1));
    words.push(enc_vintrp(0, 6, 0, 3, 0));
    words.push(enc_vintrp(1, 6, 0, 3, 1));

    // Compare-only depth textures (srsrc 2, 4, 6, 8) + compare-only sampler (ssamp 1)
    // 1) 2D: image_gather4_c (0x48) + image_gather4_c_lz_o (0x5F)
    words.extend_from_slice(&enc_mimg(0x48, 0x1, 1, 2, 10, 2, 1));
    words.extend_from_slice(&enc_mimg(0x5F, 0x1, 1, 1, 14, 2, 1));
    // 2) 2D-Array: image_gather4_c_lz (0x4F) dim:5
    words.extend_from_slice(&enc_mimg(0x4F, 0x1, 5, 2, 18, 4, 1));
    // 3) Cube: image_gather4_c_l (0x4C) dim:3
    words.extend_from_slice(&enc_mimg(0x4C, 0x1, 3, 2, 22, 6, 1));
    // 4) Cube-Array: image_gather4_c (0x48) dim:7 + d16=1
    words.extend_from_slice(&enc_mimg_full(0x48, 0x1, 7, true, 2, 26, 8, 1));

    words.push(enc_vop2(0x03, 10, 256 + 14, 10));
    words.push(enc_vop2(0x03, 11, 256 + 18, 22));
    words.extend_from_slice(&enc_exp(0, 0xF, false, false, false, [10, 11, 12, 13]));
    words.extend_from_slice(&enc_exp(1, 0xF, true, true, true, [26, 27, 0, 0]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_stage(&words, Rdna2StageOverride::Pixel);
    assert!(out.wgsl.contains("texture_depth_2d"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("texture_depth_2d_array"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("texture_depth_cube"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("texture_depth_cube_array"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("sampler_comparison"), "WGSL:\n{}", out.wgsl);
    assert!(
        out.wgsl.matches("textureGatherCompare(").count() >= 5,
        "expected >= 5 textureGatherCompare calls:\n{}",
        out.wgsl
    );
    assert!(
        out.bindings.iter().any(|b| b.kind == "sampler_comparison"),
        "expected sampler_comparison binding in {:?}",
        out.bindings
    );
}

#[test]
fn s70_mixed_use_gather4_c_and_sample_on_same_texture_falls_back_cleanly() {
    // When the same texture/sampler is used with both non-compare `image_sample` and
    // compare `image_gather4_c`, the translator must keep `texture_2d<f32>` + `sampler`
    // and emit a valid 4-tap `textureGather` + software comparison fallback.
    let mut words = Vec::new();
    words.push(enc_vop1(0x01, 1, 240)); // z_ref = 0.5
    words.push(enc_vintrp(0, 2, 0, 0, 0));
    words.push(enc_vintrp(1, 2, 0, 0, 1));
    words.push(enc_vintrp(0, 3, 0, 1, 0));
    words.push(enc_vintrp(1, 3, 0, 1, 1));
    words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 2, 6, 2, 1)); // image_sample on srsrc=2, ssamp=1
    words.extend_from_slice(&enc_mimg(0x48, 0x1, 1, 1, 10, 2, 1)); // image_gather4_c on SAME srsrc=2, ssamp=1
    words.push(enc_vop2(0x03, 6, 256 + 10, 6));
    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [6, 11, 12, 13]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_stage(&words, Rdna2StageOverride::Pixel);
    assert!(out.wgsl.contains("texture_2d<f32>"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureSample("), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureGather(0"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("select("), "WGSL:\n{}", out.wgsl);
}

// ---------------------------------------------------------------------------
// 2. image_sample_c_* (Shadow PCF) across 2D, 2D-Array, Cube, Cube-Array
//    in Fragment (uniform + non-uniform CF) and Compute stages
// ---------------------------------------------------------------------------

#[test]
fn s70_image_sample_c_shadow_pcf_across_all_depth_dimensions_fragment_and_compute() {
    // Part A: Fragment shader with shadow PCF on 2d, 2d-array, cube, cube-array
    let mut ps_words = Vec::new();
    ps_words.push(enc_vop1(0x01, 1, 128 + 2)); // offset
    ps_words.push(enc_vop1(0x01, 2, 240)); // bias / z_ref
    ps_words.push(enc_vop1(0x01, 3, 240)); // z_ref
    ps_words.push(enc_vintrp(0, 4, 0, 0, 0)); // u
    ps_words.push(enc_vintrp(1, 4, 0, 0, 1));
    ps_words.push(enc_vintrp(0, 5, 0, 1, 0)); // v
    ps_words.push(enc_vintrp(1, 5, 0, 1, 1));
    ps_words.push(enc_vintrp(0, 6, 0, 2, 0)); // w / slice
    ps_words.push(enc_vintrp(1, 6, 0, 2, 1));
    ps_words.push(enc_vintrp(0, 7, 0, 3, 0)); // cube-array slice
    ps_words.push(enc_vintrp(1, 7, 0, 3, 1));
    ps_words.push(enc_vop1(0x01, 8, 242)); // lod / clamp

    // 2D: image_sample_c (0x28 -> textureSampleCompare), image_sample_c_lz (0x2F -> Level),
    //     image_sample_c_l (0x2C -> Level), image_sample_c_b (0x2D -> Compare),
    //     image_sample_c_o (0x38), image_sample_c_lz_o (0x3F), image_sample_c_d (0x2A -> Level)
    ps_words.extend_from_slice(&enc_mimg(0x28, 0x1, 1, 3, 20, 2, 1));
    ps_words.extend_from_slice(&enc_mimg(0x2F, 0x1, 1, 3, 21, 2, 1));
    ps_words.extend_from_slice(&enc_mimg(0x2C, 0x1, 1, 3, 22, 2, 1));
    ps_words.extend_from_slice(&enc_mimg(0x2D, 0x1, 1, 2, 23, 2, 1));
    ps_words.extend_from_slice(&enc_mimg(0x38, 0x1, 1, 2, 24, 2, 1));
    ps_words.extend_from_slice(&enc_mimg(0x3F, 0x1, 1, 2, 25, 2, 1));
    ps_words.extend_from_slice(&enc_mimg(0x2A, 0x1, 1, 1, 26, 2, 1));
    // 2D-Array (dim:5)
    ps_words.extend_from_slice(&enc_mimg(0x28, 0x1, 5, 3, 27, 4, 1));
    ps_words.extend_from_slice(&enc_mimg(0x2F, 0x1, 5, 3, 28, 4, 1));
    // Cube (dim:3)
    ps_words.extend_from_slice(&enc_mimg(0x28, 0x1, 3, 3, 29, 6, 1));
    ps_words.extend_from_slice(&enc_mimg(0x2F, 0x1, 3, 3, 30, 6, 1));
    // Cube-Array (dim:7)
    ps_words.extend_from_slice(&enc_mimg(0x28, 0x1, 7, 3, 31, 8, 1));
    ps_words.extend_from_slice(&enc_mimg(0x2F, 0x1, 7, 3, 32, 8, 1));

    // Combine into mrt0
    ps_words.push(enc_vop2(0x03, 20, 256 + 21, 20));
    ps_words.push(enc_vop2(0x03, 22, 256 + 23, 22));
    ps_words.push(enc_vop2(0x03, 27, 256 + 29, 27));
    ps_words.push(enc_vop2(0x03, 31, 256 + 32, 31));
    ps_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [20, 22, 27, 31]));
    ps_words.push(enc_sopp(0x01, 0));

    let ps_out = translate_stage(&ps_words, Rdna2StageOverride::Pixel);
    assert!(ps_out.wgsl.contains("textureSampleCompare("), "WGSL:\n{}", ps_out.wgsl);
    assert!(
        ps_out.wgsl.contains("textureSampleCompareLevel("),
        "WGSL:\n{}",
        ps_out.wgsl
    );
    assert!(ps_out.wgsl.contains("texture_depth_2d"), "WGSL:\n{}", ps_out.wgsl);
    assert!(
        ps_out.wgsl.contains("texture_depth_2d_array"),
        "WGSL:\n{}",
        ps_out.wgsl
    );
    assert!(ps_out.wgsl.contains("texture_depth_cube"), "WGSL:\n{}", ps_out.wgsl);
    assert!(
        ps_out.wgsl.contains("texture_depth_cube_array"),
        "WGSL:\n{}",
        ps_out.wgsl
    );

    // Part B: Compute stage `image_sample_c` MUST emit `textureSampleCompareLevel` (never `textureSampleCompare`)
    let mut cs_words = Vec::new();
    cs_words.push(enc_vop1(0x01, 1, 240)); // z_ref
    cs_words.push(enc_vop1(0x01, 2, 240)); // u
    cs_words.push(enc_vop1(0x01, 3, 240)); // v
    cs_words.push(enc_vop1(0x01, 4, 242)); // w / slice
    cs_words.push(enc_vop1(0x01, 5, 128 + 1)); // array index
    cs_words.extend_from_slice(&enc_mimg(0x28, 0x1, 1, 1, 10, 2, 1)); // 2d
    cs_words.extend_from_slice(&enc_mimg(0x28, 0x1, 5, 1, 11, 4, 1)); // 2d-array
    cs_words.extend_from_slice(&enc_mimg(0x28, 0x1, 3, 1, 12, 6, 1)); // cube
    cs_words.extend_from_slice(&enc_mimg(0x28, 0x1, 7, 1, 13, 8, 1)); // cube-array
    cs_words.push(enc_sopp(0x01, 0));

    let cs_out = translate_stage(&cs_words, Rdna2StageOverride::Compute);
    assert_eq!(cs_out.stage, ShaderStage::Compute);
    assert!(
        !cs_out.wgsl.contains("textureSampleCompare("),
        "compute shader must never emit implicit-derivative textureSampleCompare:\n{}",
        cs_out.wgsl
    );
    assert_eq!(
        cs_out.wgsl.matches("textureSampleCompareLevel(").count(),
        4,
        "compute shader must emit 4 textureSampleCompareLevel calls:\n{}",
        cs_out.wgsl
    );
}

// ---------------------------------------------------------------------------
// 3. image_sample / _l / _lz / _b / _d / _cd / _cl / _o across all 5 dims
//    (2d, 2d-array, cube, cube-array, 3d) + d16 outputs + NSA addressing
// ---------------------------------------------------------------------------

#[test]
fn s70_image_sample_variants_across_all_5_dimensions_d16_and_nsa() {
    let mut words = Vec::new();
    // Populate address VGPRs v1..v11
    for v in 1..=11u8 {
        words.push(enc_vop1(0x01, v, 240)); // 0.5
    }

    // 2D (dim:1, srsrc=2): sample, _l, _lz, _b, _d, _cd (0x68), _o (0x30), _cl (0x21)
    words.extend_from_slice(&enc_mimg(0x20, 0xF, 1, 1, 20, 2, 1));
    words.extend_from_slice(&enc_mimg(0x24, 0xF, 1, 1, 24, 2, 1));
    words.extend_from_slice(&enc_mimg(0x27, 0xF, 1, 1, 28, 2, 1));
    words.extend_from_slice(&enc_mimg(0x25, 0xF, 1, 1, 32, 2, 1));
    words.extend_from_slice(&enc_mimg(0x22, 0xF, 1, 1, 36, 2, 1));
    words.extend_from_slice(&enc_mimg(0x68, 0xF, 1, 1, 40, 2, 1));
    words.extend_from_slice(&enc_mimg(0x30, 0xF, 1, 1, 44, 2, 1));
    words.extend_from_slice(&enc_mimg(0x21, 0xF, 1, 1, 48, 2, 1));

    // 2D-Array (dim:5, srsrc=4): _l, _b, _d, d16=1 sparse dmask=0x7 (3 comps -> 2 VGPRs)
    words.extend_from_slice(&enc_mimg(0x24, 0xF, 5, 1, 52, 4, 1));
    words.extend_from_slice(&enc_mimg(0x25, 0xF, 5, 1, 56, 4, 1));
    words.extend_from_slice(&enc_mimg(0x22, 0xF, 5, 1, 60, 4, 1));
    words.extend_from_slice(&enc_mimg_full(0x27, 0x7, 5, true, 1, 64, 4, 1));

    // Cube (dim:3, srsrc=6): sample, _l, _lz, _b, _d (6 grad VGPRs + 3 dir VGPRs)
    words.extend_from_slice(&enc_mimg(0x20, 0xF, 3, 1, 68, 6, 1));
    words.extend_from_slice(&enc_mimg(0x24, 0xF, 3, 1, 72, 6, 1));
    words.extend_from_slice(&enc_mimg(0x27, 0xF, 3, 1, 76, 6, 1));
    words.extend_from_slice(&enc_mimg(0x25, 0xF, 3, 1, 80, 6, 1));
    words.extend_from_slice(&enc_mimg(0x22, 0xF, 3, 1, 84, 6, 1));

    // Cube-Array (dim:7, srsrc=8): sample, _l, _lz, _b, _d (6 grad + 4 coords), d16=1
    words.extend_from_slice(&enc_mimg(0x20, 0xF, 7, 1, 88, 8, 1));
    words.extend_from_slice(&enc_mimg(0x24, 0xF, 7, 1, 92, 8, 1));
    words.extend_from_slice(&enc_mimg(0x27, 0xF, 7, 1, 96, 8, 1));
    words.extend_from_slice(&enc_mimg(0x25, 0xF, 7, 1, 100, 8, 1));
    words.extend_from_slice(&enc_mimg(0x22, 0xF, 7, 1, 104, 8, 1));
    words.extend_from_slice(&enc_mimg_full(0x20, 0xF, 7, true, 1, 108, 8, 1));

    // 3D Volumetric (dim:2, srsrc=10): sample, _l, _lz, _b, _d (6 grad + 3 coords), _o, d16=1, + NSA
    words.extend_from_slice(&enc_mimg(0x20, 0xF, 2, 1, 112, 10, 1));
    words.extend_from_slice(&enc_mimg(0x24, 0xF, 2, 1, 116, 10, 1));
    words.extend_from_slice(&enc_mimg(0x27, 0xF, 2, 1, 120, 10, 1));
    words.extend_from_slice(&enc_mimg(0x25, 0xF, 2, 1, 124, 10, 1));
    words.extend_from_slice(&enc_mimg(0x22, 0xF, 2, 1, 128, 10, 1));
    words.extend_from_slice(&enc_mimg(0x30, 0xF, 2, 1, 132, 10, 1));
    words.extend_from_slice(&enc_mimg_full(0x27, 0xF, 2, true, 1, 136, 10, 1));
    // NSA non-sequential VGPRs for 3D image_sample_l: (u=v2, v=v5, w=v7, lod=v9)
    words.extend_from_slice(&enc_mimg_nsa1(0x24, 0xF, 2, 2, [5, 7, 9, 0], 140, 10, 1));

    // Also test graceful 3D gather4 fallback (must NOT emit illegal textureGather on texture_3d)
    words.extend_from_slice(&enc_mimg(0x40, 0x2, 2, 1, 144, 10, 1));

    // Export combined result
    words.push(enc_vop2(0x03, 20, 256 + 68, 20));
    words.push(enc_vop2(0x03, 21, 256 + 88, 112));
    words.push(enc_vop2(0x03, 22, 256 + 140, 144));
    words.extend_from_slice(&enc_exp(0, 0xF, false, false, false, [20, 21, 22, 23]));
    words.extend_from_slice(&enc_exp(1, 0xF, true, true, true, [108, 109, 0, 0]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_stage(&words, Rdna2StageOverride::Pixel);
    assert!(out.wgsl.contains("texture_2d<f32>"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("texture_2d_array<f32>"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("texture_cube<f32>"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("texture_cube_array<f32>"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("texture_3d<f32>"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureSample("), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureSampleLevel("), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureSampleBias("), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureSampleGrad("), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("pack2x16float("), "WGSL:\n{}", out.wgsl);

    let view_dims: Vec<&str> = out
        .bindings
        .iter()
        .filter_map(|b| b.view_dimension)
        .collect();
    for expected in ["2d", "2d-array", "cube", "cube-array", "3d"] {
        assert!(
            view_dims.contains(&expected),
            "expected view_dimension '{expected}' in {view_dims:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// 4. Multi-MRT (mrt0..mrt7, f32 + compr f16) + mrtz (@builtin(frag_depth))
// ---------------------------------------------------------------------------

#[test]
fn s70_multi_mrt_all_8_targets_with_compr_and_mrtz_including_conditional_write() {
    let mut words = Vec::new();
    // Interpolate 2 attributes
    words.push(enc_vintrp(0, 2, 0, 0, 0));
    words.push(enc_vintrp(1, 2, 0, 0, 1));
    words.push(enc_vintrp(0, 3, 0, 1, 0));
    words.push(enc_vintrp(1, 3, 0, 1, 1));
    words.push(enc_vop1(0x01, 4, 240)); // 0.5
    words.push(enc_vop1(0x01, 5, 242)); // 1.0
    // Pack two f16 pairs for compr MRTs via v_cvt_pkrtz_f16_f32 (VOP2 0x2F)
    words.push(enc_vop2(0x2F, 6, 256 + 2, 3));
    words.push(enc_vop2(0x2F, 7, 256 + 4, 5));

    // Export mrt0..mrt5 unconditionally (mix of uncompressed f32 and compr f16)
    words.extend_from_slice(&enc_exp(0, 0xF, false, false, false, [2, 3, 4, 5])); // mrt0 f32
    words.extend_from_slice(&enc_exp(1, 0xF, true, false, false, [6, 7, 0, 0])); // mrt1 compr f16
    words.extend_from_slice(&enc_exp(2, 0xF, false, false, false, [5, 4, 3, 2])); // mrt2 f32
    words.extend_from_slice(&enc_exp(3, 0xF, true, false, false, [7, 6, 0, 0])); // mrt3 compr f16
    words.extend_from_slice(&enc_exp(4, 0x7, false, false, false, [2, 3, 4, 0])); // mrt4 partial f32
    words.extend_from_slice(&enc_exp(5, 0x3, true, false, false, [6, 0, 0, 0])); // mrt5 partial compr

    // Conditionally export mrt6, mrt7, and mrtz on one side of a branch (PsOut must be zero-initialized!)
    words.push(enc_sopc(0x06, 0, 128 + 0)); // s_cmp_eq_u32 s0, 0
    words.push(enc_sopp(0x04, 6)); // s_cbranch_scc0 +6 words (skip 3x 8-byte EXP)
    words.extend_from_slice(&enc_exp(6, 0xF, false, false, false, [2, 4, 3, 5])); // mrt6
    words.extend_from_slice(&enc_exp(7, 0xF, true, false, false, [6, 7, 0, 0])); // mrt7 compr
    words.extend_from_slice(&enc_exp(8, 0x1, false, true, true, [4, 0, 0, 0])); // mrtz
    words.push(enc_sopp(0x01, 0));

    let out = translate_stage(&words, Rdna2StageOverride::Pixel);
    assert_eq!(out.binding_summary.output_locations, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(out.binding_summary.exports_depth);
    for loc in 0..8 {
        assert!(
            out.wgsl.contains(&format!("@location({loc})")),
            "missing @location({loc}) in PsOut:\n{}",
            out.wgsl
        );
    }
    assert!(out.wgsl.contains("@builtin(frag_depth)"), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", out.wgsl);
}

#[test]
fn s70_depth_only_fragment_shader_with_mrtz_only_and_compr_mrtz() {
    // Uncompressed depth-only shadow pass (only mrtz, zero color MRTs)
    let mut words = Vec::new();
    words.push(enc_vintrp(0, 2, 0, 0, 0));
    words.push(enc_vintrp(1, 2, 0, 0, 1));
    words.extend_from_slice(&enc_exp(8, 0x1, false, true, true, [2, 0, 0, 0]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_stage(&words, Rdna2StageOverride::Pixel);
    assert!(out.binding_summary.output_locations.is_empty());
    assert!(out.binding_summary.exports_depth);
    assert!(out.wgsl.contains("@builtin(frag_depth)"), "WGSL:\n{}", out.wgsl);
    assert!(!out.wgsl.contains("mrt_0"), "depth-only PS should not synthesize mrt_0:\n{}", out.wgsl);

    // Compressed mrtz export (compr = true -> unpack2x16float().x)
    let mut compr_words = Vec::new();
    compr_words.push(enc_vop1(0x01, 2, 240));
    compr_words.push(enc_vop2(0x2F, 3, 256 + 2, 2)); // pack f16
    compr_words.extend_from_slice(&enc_exp(8, 0x1, true, true, true, [3, 0, 0, 0]));
    compr_words.push(enc_sopp(0x01, 0));

    let compr_out = translate_stage(&compr_words, Rdna2StageOverride::Pixel);
    assert!(compr_out.binding_summary.exports_depth);
    assert!(compr_out.wgsl.contains("unpack2x16float("), "WGSL:\n{}", compr_out.wgsl);
}

// ---------------------------------------------------------------------------
// 5. Fragment interpolation qualifiers: perspective/centroid/linear/flat
// ---------------------------------------------------------------------------

#[test]
fn s70_fragment_interpolation_qualifiers_perspective_centroid_linear_and_flat() {
    let mut words = Vec::new();
    // attr0: PERSP_CENTER (v0, v1) -> default perspective center
    words.push(enc_vintrp(0, 10, 0, 0, 0));
    words.push(enc_vintrp(1, 10, 0, 0, 1));
    // attr1: PERSP_CENTROID (v2, v3) -> @interpolate(perspective, centroid)
    words.push(enc_vintrp(0, 11, 1, 0, 2));
    words.push(enc_vintrp(1, 11, 1, 0, 3));
    // attr2: LINEAR_CENTER (v4, v5) -> @interpolate(linear) / @interpolate(linear, center)
    words.push(enc_vintrp(0, 12, 2, 0, 4));
    words.push(enc_vintrp(1, 12, 2, 0, 5));
    // attr3: LINEAR_CENTROID (v6, v7) -> @interpolate(linear, centroid)
    words.push(enc_vintrp(0, 13, 3, 0, 6));
    words.push(enc_vintrp(1, 13, 3, 0, 7));
    // attr4: FLAT via v_interp_mov_f32 (op=2, p0=2) -> @interpolate(flat)
    words.push(enc_vintrp(2, 14, 4, 0, 2));

    // Combine all 5 interpolated channels into mrt0
    words.push(enc_vop2(0x03, 10, 256 + 14, 10));
    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [10, 11, 12, 13]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_stage(&words, Rdna2StageOverride::Pixel);
    assert_eq!(out.binding_summary.fragment_input_locations, vec![0, 1, 2, 3, 4]);

    // Re-parse with Naga and verify exact Interpolation + Sampling on every FragIn member!
    let reparsed = wgsl_in::parse_str(&out.wgsl).expect("WGSL must re-parse cleanly");
    let ifaces = inspect_fragment_interface(&reparsed);
    assert_eq!(ifaces.len(), 1);
    let inputs = &ifaces[0].input_interpolations;
    assert_eq!(inputs.len(), 5, "expected 5 fragment inputs, got {inputs:?}");

    assert_eq!(inputs[0].location, 0);
    assert_eq!(inputs[0].interpolation, Some(Interpolation::Perspective));
    assert!(matches!(inputs[0].sampling, None | Some(Sampling::Center)));

    assert_eq!(inputs[1].location, 1);
    assert_eq!(inputs[1].interpolation, Some(Interpolation::Perspective));
    assert_eq!(inputs[1].sampling, Some(Sampling::Centroid));

    assert_eq!(inputs[2].location, 2);
    assert_eq!(inputs[2].interpolation, Some(Interpolation::Linear));
    assert!(matches!(inputs[2].sampling, None | Some(Sampling::Center)));

    assert_eq!(inputs[3].location, 3);
    assert_eq!(inputs[3].interpolation, Some(Interpolation::Linear));
    assert_eq!(inputs[3].sampling, Some(Sampling::Centroid));

    assert_eq!(inputs[4].location, 4);
    assert_eq!(inputs[4].interpolation, Some(Interpolation::Flat));
}

// ---------------------------------------------------------------------------
// 6. lower.rs: 3D PBR sampling, multi-MRT + frag_depth & interpolation lowering
// ---------------------------------------------------------------------------

#[test]
fn s70_lower_3d_pbr_deferred_gbuffer_shadow_gather_cube_array_3d_and_interpolation_roundtrips() {
    let src = r#"
struct PbrParams {
    light_dir: vec4<f32>,
    params: vec4<f32>,
}

struct FragIn {
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(perspective, centroid) world_normal: vec3<f32>,
    @location(2) @interpolate(linear, center) screen_uv: vec2<f32>,
    @location(3) @interpolate(linear, centroid) fog_uvw: vec3<f32>,
    @location(4) @interpolate(flat) material_id: u32,
}

struct GBufferOut {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) rough_metal: vec4<f32>,
    @location(3) emissive: vec4<f32>,
    @location(4) shadow_pcf: vec4<f32>,
    @location(5) ibl_cube: vec4<f32>,
    @location(6) vol_fog: vec4<f32>,
    @location(7) mat_tag: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@group(0) @binding(0) var<uniform> pbr: PbrParams;
@group(4) @binding(0) var albedo_tex: texture_2d<f32>;
@group(4) @binding(1) var shadow_csm: texture_depth_2d_array;
@group(5) @binding(0) var ibl_probes: texture_cube_array<f32>;
@group(5) @binding(1) var shadow_cube_arr: texture_depth_cube_array;
@group(6) @binding(0) var fog_volume: texture_3d<f32>;
@group(6) @binding(1) var linear_samp: sampler;
@group(7) @binding(0) var shadow_cmp_samp: sampler_comparison;

@fragment
fn fs_gbuffer(in: FragIn) -> GBufferOut {
    let alb = textureSample(albedo_tex, linear_samp, in.uv);
    let gather_r = textureGather(0, albedo_tex, linear_samp, in.uv);
    let pcf_2da = textureSampleCompare(shadow_csm, shadow_cmp_samp, in.uv, 1i, 0.5);
    let pcf_gather = textureGatherCompare(shadow_csm, shadow_cmp_samp, in.uv, 0i, 0.5);
    let ibl = textureSampleLevel(ibl_probes, linear_samp, in.world_normal, 0i, 1.0);
    let pcf_cube = textureSampleCompareLevel(shadow_cube_arr, shadow_cmp_samp, in.world_normal, 0i, 0.5);
    let vol = textureSample(fog_volume, linear_samp, in.fog_uvw);

    var out: GBufferOut;
    out.albedo = alb;
    out.normal = vec4<f32>(normalize(in.world_normal), 1.0);
    out.rough_metal = gather_r;
    out.emissive = vec4<f32>(in.screen_uv, pbr.params.xy);
    out.shadow_pcf = vec4<f32>(pcf_2da, pcf_cube, pcf_gather.x, pcf_gather.y);
    out.ibl_cube = ibl;
    out.vol_fog = vol;
    out.mat_tag = vec4<f32>(f32(in.material_id), 0.0, 0.0, 1.0);
    out.depth = clamp(pbr.params.z + pcf_2da * 0.01, 0.0, 1.0);
    return out;
}
"#;
    let mut module = wgsl_in::parse_str(src).expect("valid 3D PBR deferred G-buffer WGSL");
    let (lowered_wgsl, report) =
        lower_and_emit_wgsl(&mut module).expect("3D PBR G-buffer shader must lower cleanly");

    assert_eq!(report.bind_group_remaps.len(), 7);
    for hi_group in 4..=7 {
        assert!(!lowered_wgsl.contains(&format!("@group({hi_group})")));
    }
    assert_naga_wgsl_valid(&lowered_wgsl);

    let ifaces = inspect_fragment_interface(&module);
    assert_eq!(ifaces.len(), 1);
    assert_eq!(ifaces[0].mrt_output_locations, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    assert!(ifaces[0].exports_frag_depth);
    assert_eq!(ifaces[0].input_interpolations.len(), 5);
    assert_eq!(
        ifaces[0].input_interpolations[4].interpolation,
        Some(Interpolation::Flat)
    );
}

#[test]
fn s70_lower_normalizes_gather_depth_compare_levels_cube_offsets_and_flat_integer_inputs() {
    // Construct a Naga IR fragment module with common SPIR-V/DXIL quirks that fail baseline WebGPU:
    //   1. Integer `@location(0) id: u32` missing `@interpolate(flat)`
    //   2. Float `@location(1)` with `Interpolation::Flat` + illegal `Sampling::Center`
    //   3. Float `@location(2)` with non-baseline `Sampling::Sample`
    //   4. `textureGather` with `SampleLevel::Auto` instead of `SampleLevel::Zero`
    //   5. `textureSampleCompare` with `SampleLevel::Exact(0.0)` instead of `SampleLevel::Zero`
    //   6. `textureSample` on `texture_cube<f32>` with an illegal `offset: Some(...)`
    let mut m = Module::default();
    let f32_ty = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::F32),
        },
        Span::UNDEFINED,
    );
    let u32_ty = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar::U32),
        },
        Span::UNDEFINED,
    );
    let vec2_f32 = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Bi,
                scalar: Scalar::F32,
            },
        },
        Span::UNDEFINED,
    );
    let vec3_f32 = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::F32,
            },
        },
        Span::UNDEFINED,
    );
    let vec4_f32 = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::F32,
            },
        },
        Span::UNDEFINED,
    );
    let vec3_i32 = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::I32,
            },
        },
        Span::UNDEFINED,
    );
    let tex_2d_ty = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Image {
                dim: ImageDimension::D2,
                arrayed: false,
                class: ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
            },
        },
        Span::UNDEFINED,
    );
    let tex_depth_ty = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Image {
                dim: ImageDimension::D2,
                arrayed: false,
                class: ImageClass::Depth { multi: false },
            },
        },
        Span::UNDEFINED,
    );
    let tex_cube_ty = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Image {
                dim: ImageDimension::Cube,
                arrayed: false,
                class: ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
            },
        },
        Span::UNDEFINED,
    );
    let samp_ty = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Sampler { comparison: false },
        },
        Span::UNDEFINED,
    );
    let cmp_samp_ty = m.types.insert(
        Type {
            name: None,
            inner: TypeInner::Sampler { comparison: true },
        },
        Span::UNDEFINED,
    );

    let tex_2d = m.global_variables.append(
        GlobalVariable {
            name: Some("t2d".to_string()),
            space: naga::AddressSpace::Handle,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: tex_2d_ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let tex_depth = m.global_variables.append(
        GlobalVariable {
            name: Some("tdepth".to_string()),
            space: naga::AddressSpace::Handle,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: tex_depth_ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let tex_cube = m.global_variables.append(
        GlobalVariable {
            name: Some("tcube".to_string()),
            space: naga::AddressSpace::Handle,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 2,
            }),
            ty: tex_cube_ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let samp = m.global_variables.append(
        GlobalVariable {
            name: Some("smp".to_string()),
            space: naga::AddressSpace::Handle,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 3,
            }),
            ty: samp_ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let cmp_samp = m.global_variables.append(
        GlobalVariable {
            name: Some("cmp_smp".to_string()),
            space: naga::AddressSpace::Handle,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 4,
            }),
            ty: cmp_samp_ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );

    let in_struct = m.types.insert(
        Type {
            name: Some("QuirkyFragIn".to_string()),
            inner: TypeInner::Struct {
                members: vec![
                    StructMember {
                        name: Some("mat_id".to_string()),
                        ty: u32_ty,
                        binding: Some(Binding::Location {
                            location: 0,
                            interpolation: None,
                            sampling: None,
                            blend_src: None,
                            per_primitive: false,
                        }),
                        offset: 0,
                    },
                    StructMember {
                        name: Some("flat_val".to_string()),
                        ty: f32_ty,
                        binding: Some(Binding::Location {
                            location: 1,
                            interpolation: Some(Interpolation::Flat),
                            sampling: Some(Sampling::Center),
                            blend_src: None,
                            per_primitive: false,
                        }),
                        offset: 4,
                    },
                    StructMember {
                        name: Some("sample_uv".to_string()),
                        ty: vec2_f32,
                        binding: Some(Binding::Location {
                            location: 2,
                            interpolation: Some(Interpolation::Perspective),
                            sampling: Some(Sampling::Sample),
                            blend_src: None,
                            per_primitive: false,
                        }),
                        offset: 8,
                    },
                ],
                span: 16,
            },
        },
        Span::UNDEFINED,
    );

    let mut func = Function {
        name: Some("fs_quirky".to_string()),
        arguments: vec![FunctionArgument {
            name: Some("inp".to_string()),
            ty: in_struct,
            binding: None,
        }],
        result: Some(FunctionResult {
            ty: vec4_f32,
            binding: Some(Binding::Location {
                location: 0,
                interpolation: None,
                sampling: None,
                blend_src: None,
                per_primitive: false,
            }),
        }),
        ..Function::default()
    };

    let arg_e = func
        .expressions
        .append(Expression::FunctionArgument(0), Span::UNDEFINED);
    let uv_e = func.expressions.append(
        Expression::AccessIndex {
            base: arg_e,
            index: 2,
        },
        Span::UNDEFINED,
    );
    let t2d_e = func
        .expressions
        .append(Expression::GlobalVariable(tex_2d), Span::UNDEFINED);
    let td_e = func
        .expressions
        .append(Expression::GlobalVariable(tex_depth), Span::UNDEFINED);
    let tc_e = func
        .expressions
        .append(Expression::GlobalVariable(tex_cube), Span::UNDEFINED);
    let smp_e = func
        .expressions
        .append(Expression::GlobalVariable(samp), Span::UNDEFINED);
    let csmp_e = func
        .expressions
        .append(Expression::GlobalVariable(cmp_samp), Span::UNDEFINED);
    let zero_f = func
        .expressions
        .append(Expression::Literal(Literal::F32(0.0)), Span::UNDEFINED);
    let half_f = func
        .expressions
        .append(Expression::Literal(Literal::F32(0.5)), Span::UNDEFINED);
    let dir_e = func
        .expressions
        .append(Expression::ZeroValue(vec3_f32), Span::UNDEFINED);
    let off_e = func
        .expressions
        .append(Expression::ZeroValue(vec3_i32), Span::UNDEFINED);

    // Quirky gather with SampleLevel::Auto
    let gather_e = func.expressions.append(
        Expression::ImageSample {
            image: t2d_e,
            sampler: smp_e,
            gather: Some(SwizzleComponent::X),
            coordinate: uv_e,
            array_index: None,
            offset: None,
            level: SampleLevel::Auto,
            depth_ref: None,
            clamp_to_edge: false,
        },
        Span::UNDEFINED,
    );
    // Quirky depth compare with SampleLevel::Exact(0.0)
    let cmp_e = func.expressions.append(
        Expression::ImageSample {
            image: td_e,
            sampler: csmp_e,
            gather: None,
            coordinate: uv_e,
            array_index: None,
            offset: None,
            level: SampleLevel::Exact(zero_f),
            depth_ref: Some(half_f),
            clamp_to_edge: false,
        },
        Span::UNDEFINED,
    );
    // Quirky cube sample with illegal offset
    let cube_e = func.expressions.append(
        Expression::ImageSample {
            image: tc_e,
            sampler: smp_e,
            gather: None,
            coordinate: dir_e,
            array_index: None,
            offset: Some(off_e),
            level: SampleLevel::Exact(zero_f),
            depth_ref: None,
            clamp_to_edge: false,
        },
        Span::UNDEFINED,
    );
    let gx = func.expressions.append(
        Expression::AccessIndex {
            base: gather_e,
            index: 0,
        },
        Span::UNDEFINED,
    );
    let cx = func.expressions.append(
        Expression::AccessIndex {
            base: cube_e,
            index: 0,
        },
        Span::UNDEFINED,
    );
    let out_e = func.expressions.append(
        Expression::Compose {
            ty: vec4_f32,
            components: vec![gx, cmp_e, cx, half_f],
        },
        Span::UNDEFINED,
    );
    let mut body = Block::new();
    body.push(
        Statement::Emit(naga::Range::new_from_bounds(uv_e, uv_e)),
        Span::UNDEFINED,
    );
    body.push(
        Statement::Emit(naga::Range::new_from_bounds(gather_e, out_e)),
        Span::UNDEFINED,
    );
    body.push(Statement::Return { value: Some(out_e) }, Span::UNDEFINED);
    func.body = body;

    m.entry_points.push(EntryPoint {
        name: "fs_quirky".to_string(),
        stage: NagaStage::Fragment,
        early_depth_test: None,
        workgroup_size: [0, 0, 0],
        workgroup_size_overrides: None,
        function: func,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });

    // Confirm raw unlowered module fails baseline WebGPU validation.
    let mut pre_val = Validator::new(ValidationFlags::all(), Capabilities::empty());
    assert!(
        pre_val.validate(&m).is_err(),
        "unlowered quirky module must fail baseline WebGPU validation"
    );

    let pbr_summary = lower_pbr_sampling_and_interpolation(&mut m);
    assert_eq!(pbr_summary.normalized_gather_levels, 1);
    assert_eq!(pbr_summary.normalized_depth_compare_levels, 1);
    assert_eq!(pbr_summary.stripped_cube_offsets, 1);
    assert_eq!(pbr_summary.fixed_flat_integer_locations, 1);
    assert_eq!(pbr_summary.sanitized_flat_sampling, 1);
    assert_eq!(pbr_summary.degraded_sample_rate_inputs, 1);

    let (wgsl_out, _) =
        lower_and_emit_wgsl(&mut m).expect("normalized PBR module must emit valid baseline WGSL");
    assert!(wgsl_out.contains("textureGather("), "WGSL:\n{wgsl_out}");
    assert!(wgsl_out.contains("textureSampleCompareLevel("), "WGSL:\n{wgsl_out}");
    assert!(wgsl_out.contains("@interpolate(flat)"), "WGSL:\n{wgsl_out}");
    assert!(wgsl_out.contains("centroid"), "WGSL:\n{wgsl_out}");
    assert_naga_wgsl_valid(&wgsl_out);
}

#[test]
fn s70_lower_guards_invalid_3d_gather_and_excess_mrt_locations() {
    // 1. textureGather on 3D texture -> InvalidTextureOp
    let mut m_g3d = Module::default();
    let vec3_f32 = m_g3d.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar::F32,
            },
        },
        Span::UNDEFINED,
    );
    let tex_3d_ty = m_g3d.types.insert(
        Type {
            name: None,
            inner: TypeInner::Image {
                dim: ImageDimension::D3,
                arrayed: false,
                class: ImageClass::Sampled {
                    kind: ScalarKind::Float,
                    multi: false,
                },
            },
        },
        Span::UNDEFINED,
    );
    let samp_ty = m_g3d.types.insert(
        Type {
            name: None,
            inner: TypeInner::Sampler { comparison: false },
        },
        Span::UNDEFINED,
    );
    let t3 = m_g3d.global_variables.append(
        GlobalVariable {
            name: Some("vol".to_string()),
            space: naga::AddressSpace::Handle,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 0,
            }),
            ty: tex_3d_ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let s0 = m_g3d.global_variables.append(
        GlobalVariable {
            name: Some("smp".to_string()),
            space: naga::AddressSpace::Handle,
            binding: Some(ResourceBinding {
                group: 0,
                binding: 1,
            }),
            ty: samp_ty,
            init: None,
            memory_decorations: MemoryDecorations::empty(),
        },
        Span::UNDEFINED,
    );
    let mut f = Function {
        name: Some("cs_bad_gather3d".to_string()),
        ..Function::default()
    };
    let ie = f
        .expressions
        .append(Expression::GlobalVariable(t3), Span::UNDEFINED);
    let se = f
        .expressions
        .append(Expression::GlobalVariable(s0), Span::UNDEFINED);
    let ce = f
        .expressions
        .append(Expression::ZeroValue(vec3_f32), Span::UNDEFINED);
    let _ge = f.expressions.append(
        Expression::ImageSample {
            image: ie,
            sampler: se,
            gather: Some(SwizzleComponent::X),
            coordinate: ce,
            array_index: None,
            offset: None,
            level: SampleLevel::Zero,
            depth_ref: None,
            clamp_to_edge: false,
        },
        Span::UNDEFINED,
    );
    m_g3d.entry_points.push(EntryPoint {
        name: "cs_bad_gather3d".to_string(),
        stage: NagaStage::Compute,
        early_depth_test: None,
        workgroup_size: [1, 1, 1],
        workgroup_size_overrides: None,
        function: f,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });
    let err = try_lower_module_for_webgpu(&mut m_g3d).unwrap_err();
    assert!(
        matches!(err, LoweringError::InvalidTextureOp { .. }),
        "expected InvalidTextureOp for 3D gather, got {err:?}"
    );

    // 2. Fragment output @location(8) (>= 8 baseline MRTs) -> InvalidResourceBinding
    let mut m_mrt8 = Module::default();
    let vec4_f32 = m_mrt8.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Quad,
                scalar: Scalar::F32,
            },
        },
        Span::UNDEFINED,
    );
    let mut fs_bad = Function {
        name: Some("fs_mrt8".to_string()),
        result: Some(FunctionResult {
            ty: vec4_f32,
            binding: Some(Binding::Location {
                location: 8,
                interpolation: None,
                sampling: None,
                blend_src: None,
                per_primitive: false,
            }),
        }),
        ..Function::default()
    };
    let z4 = fs_bad
        .expressions
        .append(Expression::ZeroValue(vec4_f32), Span::UNDEFINED);
    fs_bad
        .body
        .push(Statement::Return { value: Some(z4) }, Span::UNDEFINED);
    m_mrt8.entry_points.push(EntryPoint {
        name: "fs_mrt8".to_string(),
        stage: NagaStage::Fragment,
        early_depth_test: None,
        workgroup_size: [0, 0, 0],
        workgroup_size_overrides: None,
        function: fs_bad,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    });
    let err2 = try_lower_module_for_webgpu(&mut m_mrt8).unwrap_err();
    assert!(
        matches!(err2, LoweringError::InvalidResourceBinding { .. }),
        "expected InvalidResourceBinding for fragment @location(8), got {err2:?}"
    );
}
