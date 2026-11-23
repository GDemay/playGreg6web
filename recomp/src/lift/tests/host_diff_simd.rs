//! Host-CPU differential tests for SSE/AVX floating point, conversions and data movement (see
//! [`simd_diff`](super::simd_diff) for the harness and the comparison rules).

use iced_x86::code_asm::*;

use super::simd_diff::{Check, Feature, Kind, Op, check_ops, op};

const F32: Kind = Kind::F32;
const F64: Kind = Kind::F64;
const N32: Check = Check::Nan(32);
const N64: Check = Check::Nan(64);
const EXACT: Check = Check::Exact;

/// Legacy `ins xmm1, xmm2` / `ins xmm1, mem` and VEX `vins xmm1, xmm3, xmm2` / `vins xmm1, xmm3,
/// mem`. The legacy register form gets full pair coverage.
macro_rules! two_src {
    ($ops:ident, $f:expr, $k:expr, $c:expr, $scalar:expr, $leg:ident, $vex:ident, $mem:ident) => {{
        $ops.push(op($f, $k, $c, |a| a.$leg(xmm1, xmm2)).pairs($scalar));
        $ops.push(op($f, $k, $c, |a| a.$leg(xmm1, $mem(rbx))));
        $ops.push(op(Feature::Avx, $k, $c, |a| a.$vex(xmm1, xmm3, xmm2)).cases(96));
        $ops.push(op(Feature::Avx, $k, $c, |a| a.$vex(xmm1, xmm3, $mem(rbx))));
    }};
}

/// `ins xmm1, xmm2` / `ins xmm1, mem` in both encodings (unary or move-like forms).
macro_rules! one_src {
    ($ops:ident, $f:expr, $k:expr, $c:expr, $leg:ident, $vex:ident, $mem:ident) => {{
        $ops.push(op($f, $k, $c, |a| a.$leg(xmm1, xmm2)).pairs(false));
        $ops.push(op($f, $k, $c, |a| a.$leg(xmm1, $mem(rbx))));
        $ops.push(op(Feature::Avx, $k, $c, |a| a.$vex(xmm1, xmm2)).cases(96));
        $ops.push(op(Feature::Avx, $k, $c, |a| a.$vex(xmm1, $mem(rbx))));
    }};
}

#[test]
fn sse_float_arithmetic_matches_the_host_cpu() {
    let mut ops: Vec<Op> = Vec::new();
    let s2 = Feature::Sse2;
    two_src!(ops, s2, F32, N32, false, addps, vaddps, xmmword_ptr);
    two_src!(ops, s2, F64, N64, false, addpd, vaddpd, xmmword_ptr);
    two_src!(ops, s2, F32, N32, true, addss, vaddss, dword_ptr);
    two_src!(ops, s2, F64, N64, true, addsd, vaddsd, qword_ptr);
    two_src!(ops, s2, F32, N32, false, subps, vsubps, xmmword_ptr);
    two_src!(ops, s2, F64, N64, false, subpd, vsubpd, xmmword_ptr);
    two_src!(ops, s2, F32, N32, true, subss, vsubss, dword_ptr);
    two_src!(ops, s2, F64, N64, true, subsd, vsubsd, qword_ptr);
    two_src!(ops, s2, F32, N32, false, mulps, vmulps, xmmword_ptr);
    two_src!(ops, s2, F64, N64, false, mulpd, vmulpd, xmmword_ptr);
    two_src!(ops, s2, F32, N32, true, mulss, vmulss, dword_ptr);
    two_src!(ops, s2, F64, N64, true, mulsd, vmulsd, qword_ptr);
    two_src!(ops, s2, F32, N32, false, divps, vdivps, xmmword_ptr);
    two_src!(ops, s2, F64, N64, false, divpd, vdivpd, xmmword_ptr);
    two_src!(ops, s2, F32, N32, true, divss, vdivss, dword_ptr);
    two_src!(ops, s2, F64, N64, true, divsd, vdivsd, qword_ptr);
    // min/max select one of the operands: bit-exact even for NaNs and signed zeros.
    two_src!(ops, s2, F32, EXACT, false, minps, vminps, xmmword_ptr);
    two_src!(ops, s2, F64, EXACT, false, minpd, vminpd, xmmword_ptr);
    two_src!(ops, s2, F32, EXACT, true, minss, vminss, dword_ptr);
    two_src!(ops, s2, F64, EXACT, true, minsd, vminsd, qword_ptr);
    two_src!(ops, s2, F32, EXACT, false, maxps, vmaxps, xmmword_ptr);
    two_src!(ops, s2, F64, EXACT, false, maxpd, vmaxpd, xmmword_ptr);
    two_src!(ops, s2, F32, EXACT, true, maxss, vmaxss, dword_ptr);
    two_src!(ops, s2, F64, EXACT, true, maxsd, vmaxsd, qword_ptr);
    // Scalar unary ops merge into the first source like binary ones.
    two_src!(ops, s2, F32, N32, true, sqrtss, vsqrtss, dword_ptr);
    two_src!(ops, s2, F64, N64, true, sqrtsd, vsqrtsd, qword_ptr);
    one_src!(ops, s2, F32, N32, sqrtps, vsqrtps, xmmword_ptr);
    one_src!(ops, s2, F64, N64, sqrtpd, vsqrtpd, xmmword_ptr);
    two_src!(ops, s2, F32, Check::Approx, true, rcpss, vrcpss, dword_ptr);
    two_src!(
        ops,
        s2,
        F32,
        Check::Approx,
        true,
        rsqrtss,
        vrsqrtss,
        dword_ptr
    );
    one_src!(ops, s2, F32, Check::Approx, rcpps, vrcpps, xmmword_ptr);
    one_src!(ops, s2, F32, Check::Approx, rsqrtps, vrsqrtps, xmmword_ptr);
    two_src!(ops, s2, F32, EXACT, false, andps, vandps, xmmword_ptr);
    two_src!(ops, s2, F64, EXACT, false, andpd, vandpd, xmmword_ptr);
    two_src!(ops, s2, F32, EXACT, false, andnps, vandnps, xmmword_ptr);
    two_src!(ops, s2, F64, EXACT, false, andnpd, vandnpd, xmmword_ptr);
    two_src!(ops, s2, F32, EXACT, false, orps, vorps, xmmword_ptr);
    two_src!(ops, s2, F64, EXACT, false, orpd, vorpd, xmmword_ptr);
    two_src!(ops, s2, F32, EXACT, false, xorps, vxorps, xmmword_ptr);
    two_src!(ops, s2, F64, EXACT, false, xorpd, vxorpd, xmmword_ptr);
    let s3 = Feature::Sse3;
    two_src!(ops, s3, F32, N32, false, haddps, vhaddps, xmmword_ptr);
    two_src!(ops, s3, F64, N64, false, haddpd, vhaddpd, xmmword_ptr);
    two_src!(ops, s3, F32, N32, false, hsubps, vhsubps, xmmword_ptr);
    two_src!(ops, s3, F64, N64, false, hsubpd, vhsubpd, xmmword_ptr);
    two_src!(ops, s3, F32, N32, false, addsubps, vaddsubps, xmmword_ptr);
    two_src!(ops, s3, F64, N64, false, addsubpd, vaddsubpd, xmmword_ptr);
    // Zeroing idioms.
    ops.push(op(s2, F32, EXACT, |a| a.xorps(xmm1, xmm1)));
    ops.push(op(Feature::Avx, F32, EXACT, |a| a.vxorps(xmm1, xmm1, xmm1)));
    check_ops("sse_float_arithmetic_matches_the_host_cpu", ops);
}

#[test]
fn sse_float_compares_match_the_host_cpu() {
    let mut ops: Vec<Op> = Vec::new();
    let (s2, avx) = (Feature::Sse2, Feature::Avx);
    for p in 0..8u32 {
        ops.push(op(s2, F32, EXACT, |a| a.cmpps(xmm1, xmm2, p)).pairs(false));
        ops.push(op(s2, F64, EXACT, |a| a.cmppd(xmm1, xmm2, p)).pairs(false));
        ops.push(op(s2, F32, EXACT, |a| a.cmpss(xmm1, xmm2, p)).pairs(true));
        ops.push(op(s2, F64, EXACT, |a| a.cmpsd_3(xmm1, xmm2, p)).pairs(true));
        ops.push(op(s2, F32, EXACT, |a| a.cmpps(xmm1, xmmword_ptr(rbx), p)));
        ops.push(op(s2, F64, EXACT, |a| a.cmpsd_3(xmm1, qword_ptr(rbx), p)));
    }
    for p in 0..32u32 {
        ops.push(op(avx, F32, EXACT, |a| a.vcmpps(xmm1, xmm3, xmm2, p)).pairs(false));
        ops.push(op(avx, F64, EXACT, |a| a.vcmppd(xmm1, xmm3, xmm2, p)));
        ops.push(op(avx, F32, EXACT, |a| a.vcmpss(xmm1, xmm3, xmm2, p)));
        ops.push(op(avx, F64, EXACT, |a| {
            a.vcmpsd(xmm1, xmm3, qword_ptr(rbx), p)
        }));
    }
    ops.push(op(s2, F32, EXACT, |a| a.comiss(xmm1, xmm2)).pairs(true));
    ops.push(op(s2, F32, EXACT, |a| a.ucomiss(xmm1, xmm2)).pairs(true));
    ops.push(op(s2, F64, EXACT, |a| a.comisd(xmm1, xmm2)).pairs(true));
    ops.push(op(s2, F64, EXACT, |a| a.ucomisd(xmm1, xmm2)).pairs(true));
    ops.push(op(s2, F32, EXACT, |a| a.comiss(xmm1, dword_ptr(rbx))));
    ops.push(op(s2, F64, EXACT, |a| a.ucomisd(xmm1, qword_ptr(rbx))));
    ops.push(op(avx, F32, EXACT, |a| a.vcomiss(xmm1, xmm2)).cases(200));
    ops.push(op(avx, F32, EXACT, |a| a.vucomiss(xmm1, dword_ptr(rbx))));
    ops.push(op(avx, F64, EXACT, |a| a.vcomisd(xmm1, qword_ptr(rbx))));
    ops.push(op(avx, F64, EXACT, |a| a.vucomisd(xmm1, xmm2)).cases(200));
    check_ops("sse_float_compares_match_the_host_cpu", ops);
}

#[test]
fn sse_conversions_match_the_host_cpu() {
    let mut ops: Vec<Op> = Vec::new();
    let (s2, avx) = (Feature::Sse2, Feature::Avx);
    let (i32k, i64k) = (Kind::I32, Kind::I64);
    // int -> float (GPR and memory sources), the other lanes come from the first source.
    for (k, g) in [(F32, i32k), (F32, i64k), (F64, i32k), (F64, i64k)] {
        let n = if k == F32 { N32 } else { N64 };
        let cases = 4 * g.pool_len();
        match (k == F32, g == i32k) {
            (true, true) => {
                ops.push(op(s2, k, n, |a| a.cvtsi2ss(xmm1, eax)).gpr(g).cases(cases));
                ops.push(op(s2, i32k, n, |a| a.cvtsi2ss(xmm1, dword_ptr(rbx))).cases(cases));
                ops.push(op(avx, k, n, |a| a.vcvtsi2ss(xmm1, xmm3, eax)).gpr(g));
                ops.push(op(avx, i32k, n, |a| {
                    a.vcvtsi2ss(xmm1, xmm3, dword_ptr(rbx))
                }));
            }
            (true, false) => {
                ops.push(op(s2, k, n, |a| a.cvtsi2ss(xmm1, rax)).gpr(g).cases(cases));
                ops.push(op(s2, i64k, n, |a| a.cvtsi2ss(xmm1, qword_ptr(rbx))).cases(cases));
                ops.push(op(avx, k, n, |a| a.vcvtsi2ss(xmm1, xmm3, rax)).gpr(g));
            }
            (false, true) => {
                ops.push(op(s2, k, n, |a| a.cvtsi2sd(xmm1, eax)).gpr(g).cases(cases));
                ops.push(op(s2, i32k, n, |a| a.cvtsi2sd(xmm1, dword_ptr(rbx))).cases(cases));
                ops.push(op(avx, k, n, |a| a.vcvtsi2sd(xmm1, xmm3, eax)).gpr(g));
            }
            (false, false) => {
                ops.push(op(s2, k, n, |a| a.cvtsi2sd(xmm1, rax)).gpr(g).cases(cases));
                ops.push(op(s2, i64k, n, |a| a.cvtsi2sd(xmm1, qword_ptr(rbx))).cases(cases));
                ops.push(op(avx, k, n, |a| a.vcvtsi2sd(xmm1, xmm3, qword_ptr(rbx))));
            }
        }
    }
    // float <-> double scalars.
    two_src!(ops, s2, F32, N64, true, cvtss2sd, vcvtss2sd, dword_ptr);
    two_src!(ops, s2, F64, N32, true, cvtsd2ss, vcvtsd2ss, qword_ptr);
    // float -> int scalars, rounding and truncating, 32- and 64-bit destinations.
    let pf = F32_POOL_CASES;
    let pd = F64_POOL_CASES;
    ops.push(op(s2, F32, EXACT, |a| a.cvtss2si(eax, xmm2)).cases(pf));
    ops.push(op(s2, F32, EXACT, |a| a.cvtss2si(rax, xmm2)).cases(pf));
    ops.push(op(s2, F32, EXACT, |a| a.cvttss2si(eax, xmm2)).cases(pf));
    ops.push(op(s2, F32, EXACT, |a| a.cvttss2si(rax, xmm2)).cases(pf));
    ops.push(op(s2, F64, EXACT, |a| a.cvtsd2si(eax, xmm2)).cases(pd));
    ops.push(op(s2, F64, EXACT, |a| a.cvtsd2si(rax, xmm2)).cases(pd));
    ops.push(op(s2, F64, EXACT, |a| a.cvttsd2si(eax, xmm2)).cases(pd));
    ops.push(op(s2, F64, EXACT, |a| a.cvttsd2si(rax, xmm2)).cases(pd));
    ops.push(op(s2, F32, EXACT, |a| a.cvtss2si(eax, dword_ptr(rbx))));
    ops.push(op(s2, F32, EXACT, |a| a.cvttss2si(rax, dword_ptr(rbx))));
    ops.push(op(s2, F64, EXACT, |a| a.cvtsd2si(rax, qword_ptr(rbx))));
    ops.push(op(s2, F64, EXACT, |a| a.cvttsd2si(eax, qword_ptr(rbx))));
    ops.push(op(avx, F32, EXACT, |a| a.vcvtss2si(eax, xmm2)));
    ops.push(op(avx, F32, EXACT, |a| a.vcvttss2si(rax, xmm2)));
    ops.push(op(avx, F64, EXACT, |a| a.vcvtsd2si(rax, xmm2)));
    ops.push(op(avx, F64, EXACT, |a| a.vcvttsd2si(eax, qword_ptr(rbx))));
    // Packed conversions.
    one_src!(ops, s2, Kind::I32, N32, cvtdq2ps, vcvtdq2ps, xmmword_ptr);
    one_src!(ops, s2, F32, EXACT, cvtps2dq, vcvtps2dq, xmmword_ptr);
    one_src!(ops, s2, F32, EXACT, cvttps2dq, vcvttps2dq, xmmword_ptr);
    one_src!(ops, s2, Kind::I32, N64, cvtdq2pd, vcvtdq2pd, qword_ptr);
    one_src!(ops, s2, F64, EXACT, cvtpd2dq, vcvtpd2dq, xmmword_ptr);
    one_src!(ops, s2, F64, EXACT, cvttpd2dq, vcvttpd2dq, xmmword_ptr);
    one_src!(ops, s2, F32, N64, cvtps2pd, vcvtps2pd, qword_ptr);
    one_src!(ops, s2, F64, N32, cvtpd2ps, vcvtpd2ps, xmmword_ptr);
    check_ops("sse_conversions_match_the_host_cpu", ops);
}

/// Enough cases to put every float/double special into lane 0 of the second source twice.
const F32_POOL_CASES: usize = 2 * 55;
const F64_POOL_CASES: usize = 2 * 46;

#[test]
fn sse_rounding_and_dot_products_match_the_host_cpu() {
    let mut ops: Vec<Op> = Vec::new();
    let (s41, avx) = (Feature::Sse41, Feature::Avx);
    for i in 0..16u32 {
        ops.push(op(s41, F32, N32, |a| a.roundps(xmm1, xmm2, i)).pairs(false));
        ops.push(op(s41, F64, N64, |a| a.roundpd(xmm1, xmm2, i)).pairs(false));
        ops.push(op(s41, F32, N32, |a| a.roundss(xmm1, xmm2, i)).cases(F32_POOL_CASES));
        ops.push(op(s41, F64, N64, |a| a.roundsd(xmm1, xmm2, i)).cases(F64_POOL_CASES));
    }
    for i in [0u32, 1, 2, 3, 4, 9, 10, 11] {
        ops.push(op(s41, F32, N32, |a| a.roundps(xmm1, xmmword_ptr(rbx), i)));
        ops.push(op(s41, F64, N64, |a| a.roundsd(xmm1, qword_ptr(rbx), i)));
        ops.push(op(avx, F32, N32, |a| a.vroundps(xmm1, xmm2, i)));
        ops.push(op(avx, F64, N64, |a| a.vroundpd(xmm1, xmmword_ptr(rbx), i)));
        ops.push(op(avx, F32, N32, |a| a.vroundss(xmm1, xmm3, xmm2, i)));
        ops.push(op(avx, F64, N64, |a| {
            a.vroundsd(xmm1, xmm3, qword_ptr(rbx), i)
        }));
    }
    for i in 0..256u32 {
        ops.push(op(s41, F32, N32, |a| a.dpps(xmm1, xmm2, i)).cases(24));
        ops.push(op(s41, F64, N64, |a| a.dppd(xmm1, xmm2, i)).cases(12));
    }
    for i in [0xFFu32, 0xF1, 0x7F, 0x35, 0x31, 0x00] {
        ops.push(op(s41, F32, N32, |a| a.dpps(xmm1, xmmword_ptr(rbx), i)));
        ops.push(op(avx, F32, N32, |a| a.vdpps(xmm1, xmm3, xmm2, i)));
        ops.push(op(avx, F64, N64, |a| {
            a.vdppd(xmm1, xmm3, xmmword_ptr(rbx), i)
        }));
    }
    check_ops("sse_rounding_and_dot_products_match_the_host_cpu", ops);
}

#[test]
fn sse_moves_and_shuffles_match_the_host_cpu() {
    let mut ops: Vec<Op> = Vec::new();
    let (s2, s3, s41, avx) = (Feature::Sse2, Feature::Sse3, Feature::Sse41, Feature::Avx);
    let k = Kind::I32;
    macro_rules! full_move {
        ($leg:ident, $vex:ident) => {{
            ops.push(op(s2, k, EXACT, |a| a.$leg(xmm1, xmm2)));
            ops.push(op(s2, k, EXACT, |a| a.$leg(xmm1, xmmword_ptr(rbx))));
            ops.push(op(s2, k, EXACT, |a| a.$leg(xmmword_ptr(rbx), xmm2)));
            ops.push(op(avx, k, EXACT, |a| a.$vex(xmm1, xmm2)));
            ops.push(op(avx, k, EXACT, |a| a.$vex(xmm1, xmmword_ptr(rbx))));
            ops.push(op(avx, k, EXACT, |a| a.$vex(xmmword_ptr(rbx), xmm2)));
        }};
    }
    full_move!(movaps, vmovaps);
    full_move!(movups, vmovups);
    full_move!(movapd, vmovapd);
    full_move!(movupd, vmovupd);
    full_move!(movdqa, vmovdqa);
    full_move!(movdqu, vmovdqu);
    ops.push(op(s3, k, EXACT, |a| a.lddqu(xmm1, xmmword_ptr(rbx))));
    ops.push(op(avx, k, EXACT, |a| a.vlddqu(xmm1, xmmword_ptr(rbx))));
    ops.push(op(s2, k, EXACT, |a| a.movntps(xmmword_ptr(rbx), xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movntpd(xmmword_ptr(rbx), xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movntdq(xmmword_ptr(rbx), xmm2)));
    ops.push(op(s41, k, EXACT, |a| a.movntdqa(xmm1, xmmword_ptr(rbx))));
    ops.push(op(avx, k, EXACT, |a| a.vmovntps(xmmword_ptr(rbx), xmm2)));
    ops.push(op(avx, k, EXACT, |a| a.vmovntdqa(xmm1, xmmword_ptr(rbx))));
    ops.push(op(s2, k, EXACT, |a| a.movnti(dword_ptr(rbx), eax)));
    ops.push(op(s2, k, EXACT, |a| a.movnti(qword_ptr(rbx), rax)));
    // Scalar moves (NaN patterns included: moves never touch the bits).
    for (kind, mem) in [(F32, 4), (F64, 8)] {
        if mem == 4 {
            ops.push(op(s2, kind, EXACT, |a| a.movss(xmm1, xmm2)));
            ops.push(op(s2, kind, EXACT, |a| a.movss(xmm1, dword_ptr(rbx))));
            ops.push(op(s2, kind, EXACT, |a| a.movss(dword_ptr(rbx), xmm2)));
            ops.push(op(avx, kind, EXACT, |a| a.vmovss_3(xmm1, xmm3, xmm2)));
            ops.push(op(avx, kind, EXACT, |a| a.vmovss(xmm1, dword_ptr(rbx))));
            ops.push(op(avx, kind, EXACT, |a| a.vmovss(dword_ptr(rbx), xmm2)));
        } else {
            ops.push(op(s2, kind, EXACT, |a| a.movsd_2(xmm1, xmm2)));
            ops.push(op(s2, kind, EXACT, |a| a.movsd_2(xmm1, qword_ptr(rbx))));
            ops.push(op(s2, kind, EXACT, |a| a.movsd_2(qword_ptr(rbx), xmm2)));
            ops.push(op(avx, kind, EXACT, |a| a.vmovsd_3(xmm1, xmm3, xmm2)));
            ops.push(op(avx, kind, EXACT, |a| a.vmovsd(xmm1, qword_ptr(rbx))));
            ops.push(op(avx, kind, EXACT, |a| a.vmovsd(qword_ptr(rbx), xmm2)));
        }
    }
    // GPR <-> XMM.
    ops.push(op(s2, k, EXACT, |a| a.movd(xmm1, eax)));
    ops.push(op(s2, k, EXACT, |a| a.movd(xmm1, dword_ptr(rbx))));
    ops.push(op(s2, k, EXACT, |a| a.movd(eax, xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movd(dword_ptr(rbx), xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movq(xmm1, rax)));
    ops.push(op(s2, k, EXACT, |a| a.movq(rax, xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movq(xmm1, xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movq(xmm1, qword_ptr(rbx))));
    ops.push(op(s2, k, EXACT, |a| a.movq(qword_ptr(rbx), xmm2)));
    ops.push(op(avx, k, EXACT, |a| a.vmovd(xmm1, eax)));
    ops.push(op(avx, k, EXACT, |a| a.vmovd(eax, xmm2)));
    ops.push(op(avx, k, EXACT, |a| a.vmovq(xmm1, rax)));
    ops.push(op(avx, k, EXACT, |a| a.vmovq(rax, xmm2)));
    ops.push(op(avx, k, EXACT, |a| a.vmovq(xmm1, xmm2)));
    ops.push(op(avx, k, EXACT, |a| a.vmovq(xmm1, qword_ptr(rbx))));
    ops.push(op(avx, k, EXACT, |a| a.vmovq(qword_ptr(rbx), xmm2)));
    // Half-register moves.
    ops.push(op(s2, k, EXACT, |a| a.movlps(xmm1, qword_ptr(rbx))));
    ops.push(op(s2, k, EXACT, |a| a.movlps(qword_ptr(rbx), xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movhps(xmm1, qword_ptr(rbx))));
    ops.push(op(s2, k, EXACT, |a| a.movhps(qword_ptr(rbx), xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movlpd(xmm1, qword_ptr(rbx))));
    ops.push(op(s2, k, EXACT, |a| a.movlpd(qword_ptr(rbx), xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movhpd(xmm1, qword_ptr(rbx))));
    ops.push(op(s2, k, EXACT, |a| a.movhpd(qword_ptr(rbx), xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movlhps(xmm1, xmm2)));
    ops.push(op(s2, k, EXACT, |a| a.movhlps(xmm1, xmm2)));
    ops.push(op(avx, k, EXACT, |a| {
        a.vmovlps_3(xmm1, xmm3, qword_ptr(rbx))
    }));
    ops.push(op(avx, k, EXACT, |a| {
        a.vmovhps_3(xmm1, xmm3, qword_ptr(rbx))
    }));
    ops.push(op(avx, k, EXACT, |a| a.vmovlpd(qword_ptr(rbx), xmm2)));
    ops.push(op(avx, k, EXACT, |a| a.vmovhpd(qword_ptr(rbx), xmm2)));
    ops.push(op(avx, k, EXACT, |a| a.vmovlhps(xmm1, xmm3, xmm2)));
    ops.push(op(avx, k, EXACT, |a| a.vmovhlps(xmm1, xmm3, xmm2)));
    // Sign masks.
    ops.push(op(s2, F32, EXACT, |a| a.movmskps(eax, xmm2)));
    ops.push(op(s2, F32, EXACT, |a| a.movmskps(rax, xmm2)));
    ops.push(op(s2, F64, EXACT, |a| a.movmskpd(eax, xmm2)));
    ops.push(op(s2, Kind::I8, EXACT, |a| a.pmovmskb(eax, xmm2)));
    ops.push(op(s2, Kind::I8, EXACT, |a| a.pmovmskb(rax, xmm2)));
    ops.push(op(avx, F32, EXACT, |a| a.vmovmskps(eax, xmm2)));
    ops.push(op(avx, F64, EXACT, |a| a.vmovmskpd(rax, xmm2)));
    ops.push(op(avx, Kind::I8, EXACT, |a| a.vpmovmskb(eax, xmm2)));
    // Duplicating moves.
    one_src!(ops, s3, k, EXACT, movsldup, vmovsldup, xmmword_ptr);
    one_src!(ops, s3, k, EXACT, movshdup, vmovshdup, xmmword_ptr);
    one_src!(ops, s3, k, EXACT, movddup, vmovddup, qword_ptr);
    // Shuffles, unpacks, blends, inserts.
    for i in 0..256u32 {
        ops.push(op(s2, k, EXACT, |a| a.shufps(xmm1, xmm2, i)).cases(4));
        ops.push(op(s41, k, EXACT, |a| a.insertps(xmm1, xmm2, i)).cases(4));
    }
    for i in [0u32, 0x1B, 0xE4, 0x4E, 0xB1] {
        ops.push(op(s2, k, EXACT, |a| a.shufps(xmm1, xmmword_ptr(rbx), i)).cases(8));
        ops.push(op(avx, k, EXACT, |a| a.vshufps(xmm1, xmm3, xmm2, i)).cases(8));
        ops.push(op(s41, k, EXACT, |a| a.insertps(xmm1, dword_ptr(rbx), i)).cases(8));
        ops.push(op(avx, k, EXACT, |a| a.vinsertps(xmm1, xmm3, xmm2, i)).cases(8));
    }
    for i in 0..4u32 {
        ops.push(op(s2, k, EXACT, |a| a.shufpd(xmm1, xmm2, i)).cases(8));
        ops.push(
            op(avx, k, EXACT, |a| {
                a.vshufpd(xmm1, xmm3, xmmword_ptr(rbx), i)
            })
            .cases(8),
        );
        ops.push(op(s41, k, EXACT, |a| a.blendpd(xmm1, xmm2, i)).cases(8));
        ops.push(op(avx, k, EXACT, |a| a.vblendpd(xmm1, xmm3, xmm2, i)).cases(8));
        ops.push(op(s41, k, EXACT, |a| a.extractps(eax, xmm2, i)).cases(8));
        ops.push(op(s41, k, EXACT, |a| a.extractps(dword_ptr(rbx), xmm2, i)).cases(8));
        ops.push(op(avx, k, EXACT, |a| a.vextractps(eax, xmm2, i)).cases(8));
    }
    for i in 0..16u32 {
        ops.push(op(s41, k, EXACT, |a| a.blendps(xmm1, xmm2, i)).cases(8));
        ops.push(
            op(avx, k, EXACT, |a| {
                a.vblendps(xmm1, xmm3, xmmword_ptr(rbx), i)
            })
            .cases(8),
        );
    }
    two_src!(ops, s2, k, EXACT, false, unpcklps, vunpcklps, xmmword_ptr);
    two_src!(ops, s2, k, EXACT, false, unpckhps, vunpckhps, xmmword_ptr);
    two_src!(ops, s2, k, EXACT, false, unpcklpd, vunpcklpd, xmmword_ptr);
    two_src!(ops, s2, k, EXACT, false, unpckhpd, vunpckhpd, xmmword_ptr);
    ops.push(op(s41, k, EXACT, |a| a.blendvps(xmm1, xmm2)));
    ops.push(op(s41, k, EXACT, |a| a.blendvpd(xmm1, xmmword_ptr(rbx))));
    ops.push(op(avx, k, EXACT, |a| a.vblendvps(xmm1, xmm3, xmm2, xmm0)));
    ops.push(op(avx, k, EXACT, |a| {
        a.vblendvpd(xmm1, xmm3, xmmword_ptr(rbx), xmm0)
    }));
    ops.push(op(s2, k, EXACT, |a| a.stmxcsr(dword_ptr(rbx))));
    ops.push(op(avx, k, EXACT, |a| a.vstmxcsr(dword_ptr(rbx))));
    check_ops("sse_moves_and_shuffles_match_the_host_cpu", ops);
}
