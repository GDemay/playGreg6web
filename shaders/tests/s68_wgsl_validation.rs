//! S68 (#1209 / #482): End-to-end Naga WGSL validation tests for 3D Main Menu / in-game
//! Vertex, Fragment, and Compute shaders covering MIMG dimensions/d16/dmask/VGPR bounds,
//! vertex stage storage restrictions, non-uniform control flow sampling, and multi-binding
//! deduplication & 16-byte uniform alignment.

use std::collections::BTreeSet;

use naga::valid::{Capabilities, ValidationFlags, Validator};
use playgta6_shaders::convert::ShaderStage;
use playgta6_shaders::rdna2_to_wgsl::{
    Rdna2StageOverride, Rdna2TranslateOptions, Rdna2WgslOutput, translate_rdna2_to_wgsl,
};

const BASELINE_WEBGPU_CAPABILITIES: Capabilities = Capabilities::SHADER_FLOAT16_IN_FLOAT32;

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
fn enc_mimg_d16(
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

fn translate_and_validate(words: &[u32], stage: Rdna2StageOverride) -> Rdna2WgslOutput {
    let bytes = words_to_bytes(words);
    let opts = Rdna2TranslateOptions {
        stage,
        ..Rdna2TranslateOptions::default()
    };
    let out = translate_rdna2_to_wgsl(&bytes, &opts)
        .unwrap_or_else(|e| panic!("translation failed for stage {stage}: {e}"));
    assert!(out.naga_validated, "translator must set naga_validated=true");
    assert!(!out.wgsl.contains('\r'), "WGSL must use LF-only newlines");
    assert!(
        !out.wgsl.contains("enable f16"),
        "baseline WebGPU WGSL must not emit `enable f16;`\nWGSL:\n{}",
        out.wgsl
    );

    // Round-trip parse and validate under Naga baseline WebGPU capabilities.
    let module = naga::front::wgsl::parse_str(&out.wgsl).unwrap_or_else(|e| {
        panic!(
            "Naga WGSL parse failed:\n{}\nWGSL:\n{}",
            e.emit_to_string(&out.wgsl),
            out.wgsl
        )
    });
    let mut validator = Validator::new(ValidationFlags::all(), BASELINE_WEBGPU_CAPABILITIES);
    validator.validate(&module).unwrap_or_else(|e| {
        panic!(
            "Naga WGSL validation failed:\n{}\nWGSL:\n{}",
            e.emit_to_string(&out.wgsl),
            out.wgsl
        )
    });

    // Verify binding metadata invariants: unique (group, binding), group < 4, uniform alignment.
    let mut seen_pairs = BTreeSet::new();
    for b in &out.bindings {
        assert!(b.group < 4, "group {} exceeds WebGPU baseline maxBindGroups", b.group);
        assert!(
            seen_pairs.insert((b.group, b.binding)),
            "duplicate reflection binding @group({}) @binding({}): {:?}",
            b.group,
            b.binding,
            out.bindings
        );
        if b.kind == "uniform" {
            let min_bytes = b
                .min_binding_size_bytes
                .expect("uniform binding must report min_binding_size_bytes");
            assert!(
                min_bytes >= 16 && min_bytes % 16 == 0,
                "uniform binding {} size {} must be non-zero multiple of 16",
                b.name,
                min_bytes
            );
            let highest = b
                .highest_vec4_offset
                .expect("uniform binding must report highest_vec4_offset");
            assert!(
                highest < min_bytes / 16,
                "uniform binding {} highest_vec4_offset {} >= vec4_count {}",
                b.name,
                highest,
                min_bytes / 16
            );
        }
        if out.stage == ShaderStage::Vertex {
            assert!(
                b.kind != "storage" && b.kind != "storage-texture",
                "vertex stage must never emit writable {} binding: {:?}",
                b.kind,
                b
            );
        }
    }

    // Verify no module-scope `let` appears outside comments/functions.
    let mut brace_depth = 0usize;
    for line in out.wgsl.lines() {
        let code = line.split("//").next().unwrap_or("").trim();
        if brace_depth == 0 {
            assert!(
                !code.starts_with("let "),
                "module-scope `let` found in WGSL line `{line}`\nWGSL:\n{}",
                out.wgsl
            );
        }
        brace_depth += code.matches('{').count();
        brace_depth = brace_depth.saturating_sub(code.matches('}').count());
    }

    out
}

#[test]
fn s68_mimg_dimensions_d16_dmask_mixed_dim_and_vdata_boundary() {
    // Compute shader exercising:
    // - image_load (0x00) & image_load_mip (0x01) on D2 (dim=1), D2Array (dim=5), D3 (dim=2)
    // - image_store (0x08) on D2, D2Array, D3 with d16=false and d16=true across dmask 0, 1, 3, 7, 0xF
    // - image_sample (0x20), image_sample_d (0x22), image_sample_l (0x24), image_sample_b (0x25),
    //   image_sample_lz (0x27), image_sample_c (0x28), image_sample_c_lz (0x2F)
    // - Mixed dim on the SAME texture slot (srsrc=2 used with dim=1 (D2) AND dim=2 (D3);
    //   srsrc=4 used with dim=5 (D2Array) AND dim=2 (D3))
    // - Boundary VGPR vdata=254 with dmask=0xF (both d16=false and d16=true) and vaddr=254
    let mut words = Vec::new();
    // Seed v0..v7 with finite values
    for r in 0..8u8 {
        words.push(enc_vop1(0x01, r, 242)); // v[r] = 1.0
    }
    words.push(enc_vop1(0x01, 254, 242));
    words.push(enc_vop1(0x01, 255, 242));

    // image_load & image_load_mip across D2, D2Array, D3
    words.extend_from_slice(&enc_mimg_d16(0x00, 0xF, 1, false, 0, 8, 1, 0)); // D2 load
    words.extend_from_slice(&enc_mimg_d16(0x01, 0x7, 5, false, 0, 12, 3, 0)); // D2Array load_mip
    words.extend_from_slice(&enc_mimg_d16(0x01, 0x3, 2, true, 0, 16, 2, 0)); // D3 load_mip d16

    // Mixed dim on srsrc=2 (D2 + D3 -> merged D3): sample_d with D2 inst_dim and D3 inst_dim
    words.extend_from_slice(&enc_mimg_d16(0x22, 0xF, 1, false, 0, 20, 2, 1)); // inst=D2 on tex=D3
    words.extend_from_slice(&enc_mimg_d16(0x22, 0xF, 2, false, 0, 24, 2, 1)); // inst=D3 on tex=D3
    words.extend_from_slice(&enc_mimg_d16(0x25, 0x1, 1, true, 0, 28, 2, 1)); // sample_b in compute -> SampleLevel::Exact

    // Mixed dim on srsrc=4 (D2Array + D3 -> merged D2Array): sample_d with D3 inst_dim on D2Array tex
    words.extend_from_slice(&enc_mimg_d16(0x20, 0xF, 5, false, 0, 32, 4, 1)); // D2Array sample
    words.extend_from_slice(&enc_mimg_d16(0x22, 0xF, 2, false, 0, 36, 4, 1)); // inst=D3 on tex=D2Array
    words.extend_from_slice(&enc_mimg_d16(0x24, 0x0, 5, false, 0, 40, 4, 1)); // dmask=0 -> 0xF, sample_l
    words.extend_from_slice(&enc_mimg_d16(0x28, 0xF, 1, false, 0, 44, 1, 1)); // sample_c on D2
    words.extend_from_slice(&enc_mimg_d16(0x2F, 0x7, 5, true, 0, 48, 4, 1)); // sample_c_lz d16 on D2Array

    // Boundary vdata=254 and vaddr=254 with dmask=0xF (must never panic or write OOB VGPRs)
    words.extend_from_slice(&enc_mimg_d16(0x27, 0xF, 1, false, 254, 254, 1, 1));
    words.extend_from_slice(&enc_mimg_d16(0x27, 0xF, 2, true, 254, 254, 2, 1));

    // image_store across D2, D2Array, D3 with d16=false and d16=true and dmask 0, 1, 3, 7, 0xF
    words.extend_from_slice(&enc_mimg_d16(0x08, 0x0, 1, false, 0, 8, 6, 0)); // D2 store dmask=0
    words.extend_from_slice(&enc_mimg_d16(0x08, 0x1, 1, true, 0, 8, 6, 0)); // D2 store d16 1ch
    words.extend_from_slice(&enc_mimg_d16(0x08, 0x3, 5, true, 0, 8, 7, 0)); // D2Array store d16 2ch
    words.extend_from_slice(&enc_mimg_d16(0x08, 0x7, 2, true, 0, 8, 8, 0)); // D3 store d16 3ch
    words.extend_from_slice(&enc_mimg_d16(0x08, 0xF, 2, true, 254, 254, 8, 0)); // D3 store d16 boundary
    words.push(enc_sopp(0x01, 0)); // s_endpgm

    let out = translate_and_validate(&words, Rdna2StageOverride::Compute);
    assert_eq!(out.stage, ShaderStage::Compute);
    assert!(out.wgsl.contains("textureLoad("), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureStore("), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureSampleGrad("), "WGSL:\n{}", out.wgsl);
    assert!(out.wgsl.contains("textureSampleLevel("), "WGSL:\n{}", out.wgsl);
    // Compute stage must NEVER emit implicit-derivative textureSample( or textureSampleBias(
    assert!(
        !out.wgsl.contains("textureSample("),
        "compute shader must not emit implicit textureSample:\n{}",
        out.wgsl
    );
    assert!(
        !out.wgsl.contains("textureSampleBias("),
        "compute shader must not emit implicit textureSampleBias:\n{}",
        out.wgsl
    );
}

#[test]
fn s68_vertex_shader_forbids_writable_storage_and_validates_with_naga() {
    // Vertex shader attempting:
    // - MIMG image_store (0x08) on a texture
    // - MIMG image_sample (0x20) & image_sample_b (0x25) (must lower to textureSampleLevel in VS!)
    // - MUBUF buffer_store_dword (0x1C) on a storage buffer
    // - Position export
    let mut words = Vec::new();
    words.push(enc_vop1(0x01, 1, 242)); // v1 = 1.0
    words.push(enc_vop1(0x01, 2, 242)); // v2 = 1.0
    words.push(enc_vop1(0x01, 3, 242)); // v3 = 1.0
    words.push(enc_vop1(0x01, 4, 242)); // v4 = 1.0
    // Attempted image_store in Vertex stage -> gracefully rejected, zero storage-texture binding!
    words.extend_from_slice(&enc_mimg_d16(0x08, 0xF, 1, false, 1, 1, 5, 0));
    // Attempted buffer_store_dword in Vertex stage -> zero read_write storage buffer binding!
    words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 1, 4, 0x80));
    // Sample + bias sample in Vertex stage -> must use textureSampleLevel (no implicit derivatives!)
    words.extend_from_slice(&enc_mimg_d16(0x20, 0xF, 1, false, 1, 5, 2, 1));
    words.extend_from_slice(&enc_mimg_d16(0x25, 0xF, 5, false, 1, 9, 3, 1));
    words.extend_from_slice(&enc_exp(12, 0xF, false, true, false, [5, 6, 7, 8]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_and_validate(&words, Rdna2StageOverride::Vertex);
    assert_eq!(out.stage, ShaderStage::Vertex);
    assert!(
        !out.wgsl.contains("read_write"),
        "vertex shader must never emit read_write storage:\n{}",
        out.wgsl
    );
    assert!(
        !out.wgsl.contains("texture_storage_"),
        "vertex shader must never emit storage texture:\n{}",
        out.wgsl
    );
    assert!(
        !out.wgsl.contains("textureSample(") && !out.wgsl.contains("textureSampleBias("),
        "vertex shader must never emit implicit-derivative sampling:\n{}",
        out.wgsl
    );
}

#[test]
fn s68_fragment_nonuniform_cf_lowers_sample_and_sample_bias_to_explicit_level() {
    // Fragment shader with:
    // 1. Uniform-CF `image_sample_b` before any branch (allowed to emit `textureSampleBias` with clamped bias)
    // 2. Divergent conditional branch (`v_cmp_lt_f32` -> `s_cbranch_vccz`) containing BOTH
    //    `image_sample` (0x20) and `image_sample_b` (0x25) on D2 and D3 textures.
    //    Inside non-uniform control flow, both MUST lower to `textureSampleLevel` so WebGPU/Naga
    //    uniformity analysis succeeds!
    let mut words = Vec::new();
    // Interpolate UV into v2, v3
    words.push(enc_vintrp(0, 2, 0, 0, 0));
    words.push(enc_vintrp(1, 2, 0, 0, 1));
    words.push(enc_vintrp(0, 3, 0, 1, 0));
    words.push(enc_vintrp(1, 3, 0, 1, 1));
    // Set v1 = 0.5 (bias) and v4 = 0.25 (z coord)
    words.push(enc_vop1(0x01, 1, 240)); // v1 = 0.5
    words.push(enc_vop1(0x01, 4, 240)); // v4 = 0.5
    // Uniform-CF image_sample_b (bias=v1, u=v2, v=v3) -> v[8:11]
    words.extend_from_slice(&enc_mimg_d16(0x25, 0xF, 1, false, 1, 8, 2, 1));
    // Divergent branch: v_cmp_lt_f32 vcc, v2, v3
    words.push(enc_vopc(0x01, 256 + 2, 3));
    // s_cbranch_vccz +4 dwords (skips two 2-dword MIMG instructions below)
    words.push(enc_sopp(0x06, 4));
    // Inside non-uniform branch: image_sample (0x20, D2) and image_sample_b (0x25, D3)
    words.extend_from_slice(&enc_mimg_d16(0x20, 0xF, 1, false, 2, 8, 2, 1));
    words.extend_from_slice(&enc_mimg_d16(0x25, 0xF, 2, false, 1, 12, 3, 1));
    // Combine and export MRT0
    words.push(enc_vop2(0x03, 8, 256 + 8, 12));
    words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [8, 9, 10, 11]));
    words.push(enc_sopp(0x01, 0));

    let out = translate_and_validate(&words, Rdna2StageOverride::Pixel);
    assert_eq!(out.stage, ShaderStage::Fragment);
    assert!(
        out.wgsl.contains("textureSampleBias("),
        "uniform-CF image_sample_b should emit textureSampleBias:\n{}",
        out.wgsl
    );
    assert!(
        out.wgsl.contains("textureSampleLevel("),
        "non-uniform-CF image_sample / image_sample_b must fall back to textureSampleLevel:\n{}",
        out.wgsl
    );
}

#[test]
fn s68_multi_cbv_texture_sampler_uav_compute_and_fragment_unique_bindings() {
    // 1. Compute shader with 3 CBVs, 2 textures (D2 + D3), 2 samplers, 2 storage buffers (RO + RW),
    //    and 2 storage textures (D2 + D2Array).
    let mut cs_words = Vec::new();
    // 3 distinct CBVs via SMEM s_load_dwordx4 from s[4:5], s[6:7], s[8:9] at distinct offsets
    cs_words.extend_from_slice(&enc_smem(0x02, 16, 2, 0x7D, 0, false, false)); // cb0 offset 0
    cs_words.extend_from_slice(&enc_smem(0x02, 20, 3, 0x7D, 48, false, false)); // cb1 offset 48 (vec4 #3)
    cs_words.extend_from_slice(&enc_smem(0x02, 24, 4, 0x7D, 112, false, false)); // cb2 offset 112 (vec4 #7)
    // Consume CBV outputs in VALU
    cs_words.push(enc_vop2(0x03, 1, 16, 0));
    cs_words.push(enc_vop2(0x03, 2, 20, 1));
    cs_words.push(enc_vop2(0x03, 3, 24, 2));
    // 2 textures + 2 samplers
    cs_words.extend_from_slice(&enc_mimg_d16(0x27, 0xF, 1, false, 1, 4, 1, 1));
    cs_words.extend_from_slice(&enc_mimg_d16(0x27, 0xF, 2, false, 1, 8, 2, 2));
    // 2 storage buffers: RO load on srsrc=10, RW store on srsrc=11
    cs_words.extend_from_slice(&enc_mubuf_full(0x0C, 0, true, false, 0, 12, 10, 0x80));
    cs_words.push(enc_vop2(0x03, 12, 256 + 4, 12));
    cs_words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 0, 12, 11, 0x80));
    // 2 storage textures: D2 on srsrc=12, D2Array on srsrc=13
    cs_words.extend_from_slice(&enc_mimg_d16(0x08, 0xF, 1, false, 1, 4, 12, 0));
    cs_words.extend_from_slice(&enc_mimg_d16(0x08, 0xF, 5, false, 1, 8, 13, 0));
    cs_words.push(enc_sopp(0x01, 0));

    let cs_out = translate_and_validate(&cs_words, Rdna2StageOverride::Compute);
    let cs_cbvs: Vec<_> = cs_out.bindings.iter().filter(|b| b.kind == "uniform").collect();
    assert_eq!(cs_cbvs.len(), 3, "expected 3 CBVs: {:?}", cs_out.bindings);
    assert!(cs_out.bindings.iter().any(|b| b.kind == "texture_2d"));
    assert!(cs_out.bindings.iter().any(|b| b.kind == "sampler"));
    assert!(cs_out.bindings.iter().any(|b| b.kind == "read-only-storage"));
    assert!(cs_out.bindings.iter().any(|b| b.kind == "storage"));
    assert!(cs_out.bindings.iter().any(|b| b.kind == "storage-texture"));

    // 2. Fragment shader with 2 CBVs, 2 textures, 2 samplers, RO + RW storage buffers, and UAV texture.
    let mut ps_words = Vec::new();
    ps_words.push(enc_vintrp(0, 2, 0, 0, 0));
    ps_words.push(enc_vintrp(1, 2, 0, 0, 1));
    ps_words.push(enc_vintrp(0, 3, 0, 1, 0));
    ps_words.push(enc_vintrp(1, 3, 0, 1, 1));
    ps_words.extend_from_slice(&enc_smem(0x02, 16, 2, 0x7D, 16, false, false)); // cb0
    ps_words.extend_from_slice(&enc_smem(0x02, 20, 3, 0x7D, 64, false, false)); // cb1
    ps_words.push(enc_vop2(0x03, 2, 16, 2));
    ps_words.push(enc_vop2(0x03, 3, 20, 3));
    ps_words.extend_from_slice(&enc_mimg_d16(0x20, 0xF, 1, false, 2, 4, 1, 1));
    ps_words.extend_from_slice(&enc_mimg_d16(0x20, 0xF, 5, false, 2, 8, 2, 2));
    ps_words.extend_from_slice(&enc_mubuf_full(0x0C, 0, true, false, 2, 12, 10, 0x80));
    ps_words.extend_from_slice(&enc_mubuf_full(0x1C, 0, true, false, 2, 4, 11, 0x80));
    ps_words.extend_from_slice(&enc_mimg_d16(0x08, 0xF, 1, false, 2, 8, 12, 0));
    ps_words.push(enc_vop2(0x03, 4, 256 + 4, 12));
    ps_words.extend_from_slice(&enc_exp(0, 0xF, false, true, true, [4, 5, 6, 7]));
    ps_words.push(enc_sopp(0x01, 0));

    let ps_out = translate_and_validate(&ps_words, Rdna2StageOverride::Pixel);
    let ps_cbvs: Vec<_> = ps_out.bindings.iter().filter(|b| b.kind == "uniform").collect();
    assert_eq!(ps_cbvs.len(), 2, "expected 2 fragment CBVs: {:?}", ps_out.bindings);
    assert!(ps_out.bindings.len() >= 9, "expected all fragment resources: {:?}", ps_out.bindings);
}
