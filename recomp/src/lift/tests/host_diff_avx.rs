//! Host-CPU differential tests for 256-bit AVX/AVX2 forms, cross-lane AVX instructions, FMA3 and
//! F16C (see [`simd_diff`](super::simd_diff) for the harness).

use iced_x86::code_asm::*;

use super::simd_diff::{Check, Feature, Kind, Op, check_ops, op};

const EXACT: Check = Check::Exact;
const N32: Check = Check::Nan(32);
const N64: Check = Check::Nan(64);
const AVX: Feature = Feature::Avx;
const AVX2: Feature = Feature::Avx2;
const F32: Kind = Kind::F32;
const F64: Kind = Kind::F64;

/// `vins ymm1, ymm3, ymm2` (pair coverage) and `vins ymm1, ymm3, [rbx]`.
macro_rules! ymm3 {
    ($ops:ident, $k:expr, $c:expr, $vex:ident) => {{
        $ops.push(op(AVX, $k, $c, |a| a.$vex(ymm1, ymm3, ymm2)).pairs(false));
        $ops.push(op(AVX, $k, $c, |a| a.$vex(ymm1, ymm3, ymmword_ptr(rbx))));
    }};
}

/// `vins ymm1, ymm2` (pair coverage) and `vins ymm1, [rbx]`.
macro_rules! ymm2 {
    ($ops:ident, $k:expr, $c:expr, $vex:ident) => {{
        $ops.push(op(AVX, $k, $c, |a| a.$vex(ymm1, ymm2)).pairs(false));
        $ops.push(op(AVX, $k, $c, |a| a.$vex(ymm1, ymmword_ptr(rbx))));
    }};
}

#[test]
fn avx_256_bit_float_forms_match_the_host_cpu() {
    let mut ops: Vec<Op> = Vec::new();
    ymm3!(ops, F32, N32, vaddps);
    ymm3!(ops, F64, N64, vaddpd);
    ymm3!(ops, F32, N32, vsubps);
    ymm3!(ops, F64, N64, vsubpd);
    ymm3!(ops, F32, N32, vmulps);
    ymm3!(ops, F64, N64, vmulpd);
    ymm3!(ops, F32, N32, vdivps);
    ymm3!(ops, F64, N64, vdivpd);
    ymm3!(ops, F32, EXACT, vminps);
    ymm3!(ops, F64, EXACT, vmaxpd);
    ymm3!(ops, F32, EXACT, vandps);
    ymm3!(ops, F64, EXACT, vandnpd);
    ymm3!(ops, F32, EXACT, vorps);
    ymm3!(ops, F32, EXACT, vxorps);
    ymm3!(ops, F32, N32, vhaddps);
    ymm3!(ops, F64, N64, vhsubpd);
    ymm3!(ops, F32, N32, vaddsubps);
    ymm3!(ops, F32, EXACT, vunpcklps);
    ymm3!(ops, F32, EXACT, vunpckhps);
    ymm3!(ops, F64, EXACT, vunpcklpd);
    ymm3!(ops, F64, EXACT, vunpckhpd);
    ymm2!(ops, F32, N32, vsqrtps);
    ymm2!(ops, F64, N64, vsqrtpd);
    ymm2!(ops, F32, Check::Approx, vrcpps);
    ymm2!(ops, F32, Check::Approx, vrsqrtps);
    ymm2!(ops, Kind::I32, N32, vcvtdq2ps);
    ymm2!(ops, F32, EXACT, vcvtps2dq);
    ymm2!(ops, F32, EXACT, vcvttps2dq);
    ymm2!(ops, F32, EXACT, vmovaps);
    ymm2!(ops, F32, EXACT, vmovups);
    ymm2!(ops, F64, EXACT, vmovapd);
    ymm2!(ops, Kind::I8, EXACT, vmovdqa);
    ymm2!(ops, Kind::I8, EXACT, vmovdqu);
    ymm2!(ops, F32, EXACT, vmovsldup);
    ymm2!(ops, F32, EXACT, vmovshdup);
    ymm2!(ops, F64, EXACT, vmovddup);
    ops.push(op(AVX, F32, EXACT, |a| a.vmovaps(ymmword_ptr(rbx), ymm2)));
    ops.push(op(AVX, F32, EXACT, |a| a.vmovups(ymmword_ptr(rbx), ymm2)));
    ops.push(op(AVX, F32, EXACT, |a| a.vmovntps(ymmword_ptr(rbx), ymm2)));
    ops.push(op(AVX2, F32, EXACT, |a| {
        a.vmovntdqa(ymm1, ymmword_ptr(rbx))
    }));
    ops.push(op(AVX, F32, EXACT, |a| a.vlddqu(ymm1, ymmword_ptr(rbx))));
    for p in 0..32u32 {
        ops.push(op(AVX, F32, EXACT, |a| a.vcmpps(ymm1, ymm3, ymm2, p)).cases(40));
        ops.push(
            op(AVX, F64, EXACT, |a| {
                a.vcmppd(ymm1, ymm3, ymmword_ptr(rbx), p)
            })
            .cases(20),
        );
    }
    for i in 0..16u32 {
        ops.push(op(AVX, F32, N32, |a| a.vroundps(ymm1, ymm2, i)).cases(40));
        ops.push(op(AVX, F64, N64, |a| a.vroundpd(ymm1, ymmword_ptr(rbx), i)).cases(20));
        ops.push(op(AVX, F64, EXACT, |a| a.vshufpd(ymm1, ymm3, ymm2, i)).cases(4));
        ops.push(op(AVX, F64, EXACT, |a| a.vblendpd(ymm1, ymm3, ymm2, i)).cases(4));
        ops.push(op(AVX, F64, EXACT, |a| a.vpermilpd(ymm1, ymm2, i)).cases(4));
    }
    for i in 0..256u32 {
        ops.push(op(AVX, F32, EXACT, |a| a.vshufps(ymm1, ymm3, ymm2, i)).cases(2));
        ops.push(op(AVX, F32, EXACT, |a| a.vblendps(ymm1, ymm3, ymm2, i)).cases(2));
        ops.push(op(AVX, F32, EXACT, |a| a.vpermilps(ymm1, ymm2, i)).cases(2));
        ops.push(op(AVX, F32, N32, |a| a.vdpps(ymm1, ymm3, ymm2, i)).cases(4));
    }
    for i in 0..4u32 {
        ops.push(op(AVX, F64, EXACT, |a| a.vpermilpd(xmm1, xmm2, i)).cases(4));
    }
    for k in [Kind::I32, F32] {
        ops.push(op(AVX, k, EXACT, |a| a.vpermilps(xmm1, xmm3, xmm2)).cases(64));
        ops.push(op(AVX, k, EXACT, |a| a.vpermilps(ymm1, ymm3, ymmword_ptr(rbx))).cases(64));
    }
    for k in [Kind::I64, F64] {
        ops.push(op(AVX, k, EXACT, |a| a.vpermilpd(xmm1, xmm3, xmmword_ptr(rbx))).cases(64));
        ops.push(op(AVX, k, EXACT, |a| a.vpermilpd(ymm1, ymm3, ymm2)).cases(64));
    }
    ops.push(op(AVX, F32, EXACT, |a| a.vblendvps(ymm1, ymm3, ymm2, ymm0)));
    ops.push(op(AVX, F64, EXACT, |a| {
        a.vblendvpd(ymm1, ymm3, ymmword_ptr(rbx), ymm0)
    }));
    // Width-changing conversions.
    ops.push(op(AVX, F32, N64, |a| a.vcvtps2pd(ymm1, xmm2)).pairs(false));
    ops.push(op(AVX, F32, N64, |a| a.vcvtps2pd(ymm1, xmmword_ptr(rbx))));
    ops.push(op(AVX, Kind::I32, EXACT, |a| a.vcvtdq2pd(ymm1, xmm2)));
    ops.push(op(AVX, Kind::I32, EXACT, |a| {
        a.vcvtdq2pd(ymm1, xmmword_ptr(rbx))
    }));
    ops.push(op(AVX, F64, N32, |a| a.vcvtpd2ps(xmm1, ymm2)).pairs(false));
    ops.push(op(AVX, F64, N32, |a| a.vcvtpd2ps(xmm1, ymmword_ptr(rbx))));
    ops.push(op(AVX, F64, EXACT, |a| a.vcvtpd2dq(xmm1, ymm2)).pairs(false));
    ops.push(op(AVX, F64, EXACT, |a| a.vcvttpd2dq(xmm1, ymm2)).pairs(false));
    ops.push(op(AVX, F64, EXACT, |a| {
        a.vcvttpd2dq(xmm1, ymmword_ptr(rbx))
    }));
    // The zero idiom and vzeroupper/vzeroall.
    ops.push(op(AVX, F32, EXACT, |a| a.vxorps(ymm1, ymm1, ymm1)));
    ops.push(op(AVX, F32, EXACT, |a| a.vzeroupper()));
    ops.push(op(AVX, F32, EXACT, |a| a.vzeroall()));
    check_ops("avx_256_bit_float_forms_match_the_host_cpu", ops);
}

#[test]
#[allow(clippy::vec_init_then_push)] // one form per line reads best
fn avx_cross_lane_instructions_match_the_host_cpu() {
    use Kind::{I8, I16, I32, I64};
    let mut ops: Vec<Op> = Vec::new();
    // Broadcasts.
    ops.push(op(AVX, F32, EXACT, |a| {
        a.vbroadcastss(xmm1, dword_ptr(rbx))
    }));
    ops.push(op(AVX, F32, EXACT, |a| {
        a.vbroadcastss(ymm1, dword_ptr(rbx))
    }));
    ops.push(op(AVX2, F32, EXACT, |a| a.vbroadcastss(xmm1, xmm2)));
    ops.push(op(AVX2, F32, EXACT, |a| a.vbroadcastss(ymm1, xmm2)));
    ops.push(op(AVX, F64, EXACT, |a| {
        a.vbroadcastsd(ymm1, qword_ptr(rbx))
    }));
    ops.push(op(AVX2, F64, EXACT, |a| a.vbroadcastsd(ymm1, xmm2)));
    ops.push(op(AVX, F64, EXACT, |a| {
        a.vbroadcastf128(ymm1, xmmword_ptr(rbx))
    }));
    ops.push(op(AVX2, I64, EXACT, |a| {
        a.vbroadcasti128(ymm1, xmmword_ptr(rbx))
    }));
    macro_rules! pbroadcast {
        ($k:expr, $ins:ident, $mem:ident) => {{
            ops.push(op(AVX2, $k, EXACT, |a| a.$ins(xmm1, xmm2)));
            ops.push(op(AVX2, $k, EXACT, |a| a.$ins(ymm1, xmm2)));
            ops.push(op(AVX2, $k, EXACT, |a| a.$ins(xmm1, $mem(rbx))));
            ops.push(op(AVX2, $k, EXACT, |a| a.$ins(ymm1, $mem(rbx))));
        }};
    }
    pbroadcast!(I8, vpbroadcastb, byte_ptr);
    pbroadcast!(I16, vpbroadcastw, word_ptr);
    pbroadcast!(I32, vpbroadcastd, dword_ptr);
    pbroadcast!(I64, vpbroadcastq, qword_ptr);
    // 128-bit inserts, extracts and permutes.
    for i in 0..2u32 {
        ops.push(op(AVX, I64, EXACT, |a| a.vinsertf128(ymm1, ymm3, xmm2, i)));
        ops.push(op(AVX, I64, EXACT, |a| {
            a.vinsertf128(ymm1, ymm3, xmmword_ptr(rbx), i)
        }));
        ops.push(op(AVX2, I64, EXACT, |a| a.vinserti128(ymm1, ymm3, xmm2, i)));
        ops.push(op(AVX, I64, EXACT, |a| a.vextractf128(xmm1, ymm2, i)));
        ops.push(op(AVX, I64, EXACT, |a| {
            a.vextractf128(xmmword_ptr(rbx), ymm2, i)
        }));
        ops.push(op(AVX2, I64, EXACT, |a| a.vextracti128(xmm1, ymm2, i)));
        ops.push(op(AVX2, I64, EXACT, |a| {
            a.vextracti128(xmmword_ptr(rbx), ymm2, i)
        }));
    }
    for i in 0..256u32 {
        ops.push(op(AVX, I64, EXACT, |a| a.vperm2f128(ymm1, ymm3, ymm2, i)).cases(2));
        ops.push(op(AVX2, I64, EXACT, |a| a.vpermq(ymm1, ymm2, i)).cases(2));
    }
    for i in [0x1Bu32, 0x4E, 0xE4, 0x00, 0x31] {
        ops.push(op(AVX2, I64, EXACT, |a| {
            a.vperm2i128(ymm1, ymm3, ymmword_ptr(rbx), i)
        }));
        ops.push(op(AVX2, F64, EXACT, |a| {
            a.vpermpd(ymm1, ymmword_ptr(rbx), i)
        }));
    }
    for k in [I32, I8] {
        ops.push(op(AVX2, k, EXACT, |a| a.vpermd(ymm1, ymm3, ymm2)).cases(96));
        ops.push(op(AVX2, k, EXACT, |a| a.vpermps(ymm1, ymm3, ymmword_ptr(rbx))).cases(96));
    }
    // Masked loads and stores (mask in ymm3/xmm3; masked-out elements are not touched).
    for k in [I32, I8] {
        ops.push(
            op(AVX, k, EXACT, |a| {
                a.vmaskmovps(xmm1, xmm3, xmmword_ptr(rbx))
            })
            .cases(64),
        );
        ops.push(
            op(AVX, k, EXACT, |a| {
                a.vmaskmovps(ymm1, ymm3, ymmword_ptr(rbx))
            })
            .cases(64),
        );
        ops.push(
            op(AVX, k, EXACT, |a| {
                a.vmaskmovps(xmmword_ptr(rbx), xmm3, xmm2)
            })
            .cases(64),
        );
        ops.push(
            op(AVX, k, EXACT, |a| {
                a.vmaskmovpd(ymmword_ptr(rbx), ymm3, ymm2)
            })
            .cases(64),
        );
        ops.push(
            op(AVX, k, EXACT, |a| {
                a.vmaskmovpd(xmm1, xmm3, xmmword_ptr(rbx))
            })
            .cases(64),
        );
        ops.push(
            op(AVX2, k, EXACT, |a| {
                a.vpmaskmovd(ymm1, ymm3, ymmword_ptr(rbx))
            })
            .cases(64),
        );
        ops.push(
            op(AVX2, k, EXACT, |a| {
                a.vpmaskmovq(xmmword_ptr(rbx), xmm3, xmm2)
            })
            .cases(64),
        );
    }
    check_ops("avx_cross_lane_instructions_match_the_host_cpu", ops);
}

#[test]
fn fma3_matches_the_host_cpu() {
    let fma = Feature::Fma;
    let mut ops: Vec<Op> = Vec::new();
    macro_rules! fma {
        ($($name:ident),*) => {$(
            ops.push(op(fma, F32, N32, |a| a.$name(xmm1, xmm3, xmm2)).pairs(false));
            ops.push(op(fma, F32, N32, |a| a.$name(xmm1, xmm3, xmmword_ptr(rbx))));
            ops.push(op(fma, F32, N32, |a| a.$name(ymm1, ymm3, ymm2)).pairs(false));
            ops.push(op(fma, F32, N32, |a| a.$name(ymm1, ymm3, ymmword_ptr(rbx))));
        )*};
    }
    macro_rules! fma_scalar {
        ($($name:ident),*) => {$(
            ops.push(op(fma, F32, N32, |a| a.$name(xmm1, xmm3, xmm2)).pairs(true));
            ops.push(op(fma, F32, N32, |a| a.$name(xmm1, xmm3, dword_ptr(rbx))));
        )*};
    }
    fma!(
        vfmadd132ps,
        vfmadd213ps,
        vfmadd231ps,
        vfmsub132ps,
        vfmsub213ps,
        vfmsub231ps
    );
    fma!(
        vfnmadd132ps,
        vfnmadd213ps,
        vfnmadd231ps,
        vfnmsub132ps,
        vfnmsub213ps,
        vfnmsub231ps
    );
    fma!(
        vfmaddsub132ps,
        vfmaddsub213ps,
        vfmaddsub231ps,
        vfmsubadd132ps,
        vfmsubadd213ps
    );
    fma!(vfmsubadd231ps);
    fma_scalar!(
        vfmadd132ss,
        vfmadd213ss,
        vfmadd231ss,
        vfmsub132ss,
        vfmsub213ss,
        vfmsub231ss
    );
    fma_scalar!(
        vfnmadd132ss,
        vfnmadd213ss,
        vfnmadd231ss,
        vfnmsub132ss,
        vfnmsub213ss
    );
    fma_scalar!(vfnmsub231ss);
    // Random operands of similar magnitude stress the single rounding (cancellation, ties).
    let near = Kind::F32Near;
    ops.push(op(fma, near, N32, |a| a.vfmadd231ps(ymm1, ymm3, ymm2)).cases(4000));
    ops.push(op(fma, near, N32, |a| a.vfmsub213ps(ymm1, ymm3, ymm2)).cases(4000));
    ops.push(op(fma, near, N32, |a| a.vfnmadd132ss(xmm1, xmm3, xmm2)).cases(4000));
    ops.push(op(fma, near, N32, |a| a.vfmsub213ss(xmm1, xmm3, xmm2)).cases(4000));
    check_ops("fma3_matches_the_host_cpu", ops);
}

#[test]
#[allow(clippy::vec_init_then_push)] // one form per line reads best
fn f16c_conversions_match_the_host_cpu() {
    let f16c = Feature::F16c;
    let mut ops: Vec<Op> = Vec::new();
    ops.push(op(f16c, Kind::F16, EXACT, |a| a.vcvtph2ps(xmm1, xmm2)).pairs(false));
    ops.push(op(f16c, Kind::F16, EXACT, |a| a.vcvtph2ps(ymm1, xmm2)).pairs(false));
    ops.push(op(f16c, Kind::F16, EXACT, |a| {
        a.vcvtph2ps(xmm1, qword_ptr(rbx))
    }));
    ops.push(op(f16c, Kind::F16, EXACT, |a| {
        a.vcvtph2ps(ymm1, xmmword_ptr(rbx))
    }));
    for i in 0..8u32 {
        ops.push(op(f16c, F32, EXACT, |a| a.vcvtps2ph(xmm1, xmm2, i)).pairs(false));
        ops.push(op(f16c, F32, EXACT, |a| a.vcvtps2ph(xmm1, ymm2, i)).pairs(false));
        ops.push(op(f16c, F32, EXACT, |a| {
            a.vcvtps2ph(qword_ptr(rbx), xmm2, i)
        }));
        ops.push(op(f16c, F32, EXACT, |a| {
            a.vcvtps2ph(xmmword_ptr(rbx), ymm2, i)
        }));
    }
    check_ops("f16c_conversions_match_the_host_cpu", ops);
}
