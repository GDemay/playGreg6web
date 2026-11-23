//! Host-CPU differential tests for integer SIMD in its legacy-SSE, VEX-128 and VEX-256 (AVX2)
//! encodings (see [`simd_diff`](super::simd_diff) for the harness).

use iced_x86::code_asm::*;

use super::simd_diff::{Check, Feature, Kind, Op, check_ops, op};

const EXACT: Check = Check::Exact;
const AVX: Feature = Feature::Avx;
const AVX2: Feature = Feature::Avx2;

/// `ins xmm1, xmm2` (pair coverage), `ins xmm1, [rbx]`, `vins xmm1, xmm3, xmm2`,
/// `vins ymm1, ymm3, ymm2` (pair coverage), `vins ymm1, ymm3, [rbx]`.
macro_rules! int3 {
    ($ops:ident, $f:expr, $k:expr, $leg:ident, $vex:ident) => {{
        $ops.push(op($f, $k, EXACT, |a| a.$leg(xmm1, xmm2)).pairs(false));
        $ops.push(op($f, $k, EXACT, |a| a.$leg(xmm1, xmmword_ptr(rbx))));
        $ops.push(op(AVX, $k, EXACT, |a| a.$vex(xmm1, xmm3, xmm2)));
        $ops.push(op(AVX2, $k, EXACT, |a| a.$vex(ymm1, ymm3, ymm2)).pairs(false));
        $ops.push(op(AVX2, $k, EXACT, |a| {
            a.$vex(ymm1, ymm3, ymmword_ptr(rbx))
        }));
    }};
}

/// Unary forms `ins xmm1, xmm2/m128`, `vins xmm1, xmm2`, `vins ymm1, ymm2/m256`.
macro_rules! int_unary {
    ($ops:ident, $f:expr, $k:expr, $leg:ident, $vex:ident) => {{
        $ops.push(op($f, $k, EXACT, |a| a.$leg(xmm1, xmm2)).pairs(false));
        $ops.push(op($f, $k, EXACT, |a| a.$leg(xmm1, xmmword_ptr(rbx))));
        $ops.push(op(AVX, $k, EXACT, |a| a.$vex(xmm1, xmm2)));
        $ops.push(op(AVX2, $k, EXACT, |a| a.$vex(ymm1, ymm2)));
        $ops.push(op(AVX2, $k, EXACT, |a| a.$vex(ymm1, ymmword_ptr(rbx))));
    }};
}

#[test]
fn integer_simd_arithmetic_matches_the_host_cpu() {
    use Kind::{I8, I16, I32, I64};
    let mut ops: Vec<Op> = Vec::new();
    let (s2, s3, s41, s42) = (
        Feature::Sse2,
        Feature::Ssse3,
        Feature::Sse41,
        Feature::Sse42,
    );
    int3!(ops, s2, I8, paddb, vpaddb);
    int3!(ops, s2, I16, paddw, vpaddw);
    int3!(ops, s2, I32, paddd, vpaddd);
    int3!(ops, s2, I64, paddq, vpaddq);
    int3!(ops, s2, I8, psubb, vpsubb);
    int3!(ops, s2, I16, psubw, vpsubw);
    int3!(ops, s2, I32, psubd, vpsubd);
    int3!(ops, s2, I64, psubq, vpsubq);
    int3!(ops, s2, I8, paddsb, vpaddsb);
    int3!(ops, s2, I16, paddsw, vpaddsw);
    int3!(ops, s2, I8, paddusb, vpaddusb);
    int3!(ops, s2, I16, paddusw, vpaddusw);
    int3!(ops, s2, I8, psubsb, vpsubsb);
    int3!(ops, s2, I16, psubsw, vpsubsw);
    int3!(ops, s2, I8, psubusb, vpsubusb);
    int3!(ops, s2, I16, psubusw, vpsubusw);
    int3!(ops, s2, I16, pmullw, vpmullw);
    int3!(ops, s2, I16, pmulhw, vpmulhw);
    int3!(ops, s2, I16, pmulhuw, vpmulhuw);
    int3!(ops, s2, I64, pmuludq, vpmuludq);
    int3!(ops, s2, I32, pmuludq, vpmuludq);
    int3!(ops, s2, I16, pmaddwd, vpmaddwd);
    int3!(ops, s2, I8, pavgb, vpavgb);
    int3!(ops, s2, I16, pavgw, vpavgw);
    int3!(ops, s2, I8, psadbw, vpsadbw);
    int3!(ops, s2, I8, pminub, vpminub);
    int3!(ops, s2, I8, pmaxub, vpmaxub);
    int3!(ops, s2, I16, pminsw, vpminsw);
    int3!(ops, s2, I16, pmaxsw, vpmaxsw);
    int3!(ops, s2, I8, pcmpeqb, vpcmpeqb);
    int3!(ops, s2, I16, pcmpeqw, vpcmpeqw);
    int3!(ops, s2, I32, pcmpeqd, vpcmpeqd);
    int3!(ops, s2, I8, pcmpgtb, vpcmpgtb);
    int3!(ops, s2, I16, pcmpgtw, vpcmpgtw);
    int3!(ops, s2, I32, pcmpgtd, vpcmpgtd);
    int3!(ops, s2, I64, pand, vpand);
    int3!(ops, s2, I64, pandn, vpandn);
    int3!(ops, s2, I64, por, vpor);
    int3!(ops, s2, I64, pxor, vpxor);
    int3!(ops, s3, I8, pmaddubsw, vpmaddubsw);
    int3!(ops, s3, I16, pmulhrsw, vpmulhrsw);
    int3!(ops, s3, I8, psignb, vpsignb);
    int3!(ops, s3, I16, psignw, vpsignw);
    int3!(ops, s3, I32, psignd, vpsignd);
    int3!(ops, s3, I16, phaddw, vphaddw);
    int3!(ops, s3, I32, phaddd, vphaddd);
    int3!(ops, s3, I16, phaddsw, vphaddsw);
    int3!(ops, s3, I16, phsubw, vphsubw);
    int3!(ops, s3, I32, phsubd, vphsubd);
    int3!(ops, s3, I16, phsubsw, vphsubsw);
    int_unary!(ops, s3, I8, pabsb, vpabsb);
    int_unary!(ops, s3, I16, pabsw, vpabsw);
    int_unary!(ops, s3, I32, pabsd, vpabsd);
    int3!(ops, s41, I32, pmulld, vpmulld);
    int3!(ops, s41, I64, pmuldq, vpmuldq);
    int3!(ops, s41, I32, pmuldq, vpmuldq);
    int3!(ops, s41, I8, pminsb, vpminsb);
    int3!(ops, s41, I32, pminsd, vpminsd);
    int3!(ops, s41, I16, pminuw, vpminuw);
    int3!(ops, s41, I32, pminud, vpminud);
    int3!(ops, s41, I8, pmaxsb, vpmaxsb);
    int3!(ops, s41, I32, pmaxsd, vpmaxsd);
    int3!(ops, s41, I16, pmaxuw, vpmaxuw);
    int3!(ops, s41, I32, pmaxud, vpmaxud);
    int3!(ops, s41, I64, pcmpeqq, vpcmpeqq);
    int3!(ops, s42, I64, pcmpgtq, vpcmpgtq);
    // AVX2 per-element variable shifts (counts drawn from the same lane pool, plus shifts).
    for k in [I32, Kind::Count] {
        ops.push(op(AVX2, k, EXACT, |a| a.vpsllvd(xmm1, xmm3, xmm2)).pairs(false));
        ops.push(op(AVX2, k, EXACT, |a| a.vpsrlvd(ymm1, ymm3, ymm2)).pairs(false));
        ops.push(op(AVX2, k, EXACT, |a| a.vpsravd(ymm1, ymm3, ymmword_ptr(rbx))).pairs(false));
        ops.push(op(AVX2, k, EXACT, |a| a.vpsravd(xmm1, xmm3, xmm2)).pairs(false));
        ops.push(op(AVX2, k, EXACT, |a| a.vpsllvd(ymm1, ymm3, ymm2)).pairs(false));
    }
    for k in [I64, Kind::Count] {
        ops.push(op(AVX2, k, EXACT, |a| a.vpsllvq(ymm1, ymm3, ymm2)).pairs(false));
        ops.push(op(AVX2, k, EXACT, |a| a.vpsrlvq(xmm1, xmm3, xmmword_ptr(rbx))).pairs(false));
        ops.push(op(AVX2, k, EXACT, |a| a.vpsrlvq(ymm1, ymm3, ymm2)).pairs(false));
    }
    check_ops("integer_simd_arithmetic_matches_the_host_cpu", ops);
}

#[test]
fn integer_simd_shifts_match_the_host_cpu() {
    use Kind::{I16, I32, I64};
    let mut ops: Vec<Op> = Vec::new();
    let s2 = Feature::Sse2;
    macro_rules! shift {
        ($k:expr, $leg:ident, $vex:ident) => {{
            for n in [0u32, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 255] {
                ops.push(op(s2, $k, EXACT, |a| a.$leg(xmm1, n)).cases(12));
                ops.push(op(AVX, $k, EXACT, |a| a.$vex(xmm1, xmm3, n)).cases(6));
                ops.push(op(AVX2, $k, EXACT, |a| a.$vex(ymm1, ymm3, n)).cases(6));
            }
            ops.push(
                op(s2, $k, EXACT, |a| a.$leg(xmm1, xmm2))
                    .shift_count()
                    .cases(84),
            );
            ops.push(
                op(s2, $k, EXACT, |a| a.$leg(xmm1, xmmword_ptr(rbx)))
                    .shift_count()
                    .cases(42),
            );
            ops.push(
                op(AVX, $k, EXACT, |a| a.$vex(xmm1, xmm3, xmm2))
                    .shift_count()
                    .cases(42),
            );
            ops.push(
                op(AVX2, $k, EXACT, |a| a.$vex(ymm1, ymm3, xmm2))
                    .shift_count()
                    .cases(84),
            );
            ops.push(
                op(AVX2, $k, EXACT, |a| a.$vex(ymm1, ymm3, xmmword_ptr(rbx)))
                    .shift_count()
                    .cases(42),
            );
        }};
    }
    shift!(I16, psllw, vpsllw);
    shift!(I32, pslld, vpslld);
    shift!(I64, psllq, vpsllq);
    shift!(I16, psrlw, vpsrlw);
    shift!(I32, psrld, vpsrld);
    shift!(I64, psrlq, vpsrlq);
    shift!(I16, psraw, vpsraw);
    shift!(I32, psrad, vpsrad);
    for n in [0u32, 1, 3, 7, 8, 15, 16, 17, 255] {
        ops.push(op(s2, Kind::I8, EXACT, |a| a.pslldq(xmm1, n)).cases(6));
        ops.push(op(s2, Kind::I8, EXACT, |a| a.psrldq(xmm1, n)).cases(6));
        ops.push(op(AVX, Kind::I8, EXACT, |a| a.vpslldq(xmm1, xmm3, n)).cases(6));
        ops.push(op(AVX2, Kind::I8, EXACT, |a| a.vpsrldq(ymm1, ymm3, n)).cases(6));
        ops.push(op(AVX2, Kind::I8, EXACT, |a| a.vpslldq(ymm1, ymm3, n)).cases(6));
    }
    check_ops("integer_simd_shifts_match_the_host_cpu", ops);
}

#[test]
fn integer_simd_shuffles_and_packs_match_the_host_cpu() {
    use Kind::{I8, I16, I32, I64};
    let mut ops: Vec<Op> = Vec::new();
    let (s2, s3, s41) = (Feature::Sse2, Feature::Ssse3, Feature::Sse41);
    int3!(ops, s2, I8, punpcklbw, vpunpcklbw);
    int3!(ops, s2, I8, punpckhbw, vpunpckhbw);
    int3!(ops, s2, I16, punpcklwd, vpunpcklwd);
    int3!(ops, s2, I16, punpckhwd, vpunpckhwd);
    int3!(ops, s2, I32, punpckldq, vpunpckldq);
    int3!(ops, s2, I32, punpckhdq, vpunpckhdq);
    int3!(ops, s2, I64, punpcklqdq, vpunpcklqdq);
    int3!(ops, s2, I64, punpckhqdq, vpunpckhqdq);
    int3!(ops, s2, I16, packsswb, vpacksswb);
    int3!(ops, s2, I32, packssdw, vpackssdw);
    int3!(ops, s2, I16, packuswb, vpackuswb);
    int3!(ops, s41, I32, packusdw, vpackusdw);
    int3!(ops, s3, I8, pshufb, vpshufb);
    for i in 0..256u32 {
        ops.push(op(s2, I32, EXACT, |a| a.pshufd(xmm1, xmm2, i)).cases(3));
        ops.push(op(s2, I16, EXACT, |a| a.pshuflw(xmm1, xmm2, i)).cases(3));
        ops.push(op(s2, I16, EXACT, |a| a.pshufhw(xmm1, xmm2, i)).cases(3));
        ops.push(op(s41, I16, EXACT, |a| a.pblendw(xmm1, xmm2, i)).cases(3));
        ops.push(op(AVX2, I32, EXACT, |a| a.vpblendd(ymm1, ymm3, ymm2, i)).cases(3));
    }
    for i in [0u32, 1, 4, 7, 8, 15, 16, 17, 24, 31, 32, 255] {
        ops.push(op(s3, I8, EXACT, |a| a.palignr(xmm1, xmm2, i)).cases(6));
        ops.push(op(s3, I8, EXACT, |a| a.palignr(xmm1, xmmword_ptr(rbx), i)).cases(4));
        ops.push(op(AVX, I8, EXACT, |a| a.vpalignr(xmm1, xmm3, xmm2, i)).cases(4));
        ops.push(op(AVX2, I8, EXACT, |a| a.vpalignr(ymm1, ymm3, ymm2, i)).cases(6));
    }
    for i in [0x1Bu32, 0xE4, 0x4E, 0xB1, 0x00, 0xFF] {
        ops.push(op(s2, I32, EXACT, |a| a.pshufd(xmm1, xmmword_ptr(rbx), i)).cases(4));
        ops.push(op(AVX, I32, EXACT, |a| a.vpshufd(xmm1, xmm2, i)).cases(4));
        ops.push(op(AVX2, I32, EXACT, |a| a.vpshufd(ymm1, ymmword_ptr(rbx), i)).cases(4));
        ops.push(op(AVX2, I16, EXACT, |a| a.vpshuflw(ymm1, ymm2, i)).cases(4));
        ops.push(op(AVX2, I16, EXACT, |a| a.vpshufhw(ymm1, ymm2, i)).cases(4));
        ops.push(op(AVX, I16, EXACT, |a| a.vpblendw(xmm1, xmm3, xmm2, i)).cases(4));
        ops.push(
            op(AVX2, I16, EXACT, |a| {
                a.vpblendw(ymm1, ymm3, ymmword_ptr(rbx), i)
            })
            .cases(4),
        );
        ops.push(op(AVX2, I32, EXACT, |a| a.vpblendd(xmm1, xmm3, xmm2, i)).cases(4));
    }
    ops.push(op(s41, I8, EXACT, |a| a.pblendvb(xmm1, xmm2)));
    ops.push(op(s41, I8, EXACT, |a| a.pblendvb(xmm1, xmmword_ptr(rbx))));
    ops.push(op(AVX, I8, EXACT, |a| a.vpblendvb(xmm1, xmm3, xmm2, xmm0)));
    ops.push(op(AVX2, I8, EXACT, |a| a.vpblendvb(ymm1, ymm3, ymm2, ymm0)));
    // Element extracts and inserts.
    for i in 0..16u32 {
        ops.push(op(s41, I8, EXACT, |a| a.pextrb(eax, xmm2, i)).cases(4));
        ops.push(op(s41, I8, EXACT, |a| a.pinsrb(xmm1, eax, i)).cases(4));
    }
    for i in 0..8u32 {
        ops.push(op(s2, I16, EXACT, |a| a.pextrw(eax, xmm2, i)).cases(4));
        ops.push(op(s2, I16, EXACT, |a| a.pinsrw(xmm1, eax, i)).cases(4));
        ops.push(op(s41, I16, EXACT, |a| a.pextrw(word_ptr(rbx), xmm2, i)).cases(4));
    }
    for i in 0..4u32 {
        ops.push(op(s41, I32, EXACT, |a| a.pextrd(eax, xmm2, i)).cases(4));
        ops.push(op(s41, I32, EXACT, |a| a.pinsrd(xmm1, eax, i)).cases(4));
        ops.push(op(s41, I32, EXACT, |a| a.pinsrd(xmm1, dword_ptr(rbx), i)).cases(4));
        ops.push(op(AVX, I32, EXACT, |a| a.vpextrd(dword_ptr(rbx), xmm2, i)).cases(4));
        ops.push(op(AVX, I32, EXACT, |a| a.vpinsrd(xmm1, xmm3, eax, i)).cases(4));
    }
    for i in 0..2u32 {
        ops.push(op(s41, I64, EXACT, |a| a.pextrq(rax, xmm2, i)).cases(4));
        ops.push(op(s41, I64, EXACT, |a| a.pinsrq(xmm1, rax, i)).cases(4));
        ops.push(op(AVX, I64, EXACT, |a| a.vpextrq(qword_ptr(rbx), xmm2, i)).cases(4));
        ops.push(
            op(AVX, I64, EXACT, |a| {
                a.vpinsrq(xmm1, xmm3, qword_ptr(rbx), i)
            })
            .cases(4),
        );
    }
    ops.push(op(s41, I8, EXACT, |a| a.pextrb(byte_ptr(rbx), xmm2, 5)));
    ops.push(op(s41, I8, EXACT, |a| a.pinsrb(xmm1, byte_ptr(rbx), 9)));
    ops.push(op(AVX, I8, EXACT, |a| a.vpextrb(eax, xmm2, 3)));
    ops.push(op(AVX, I16, EXACT, |a| a.vpinsrw(xmm1, xmm3, eax, 6)));
    ops.push(op(AVX, I16, EXACT, |a| a.vpextrw(eax, xmm2, 6)));
    check_ops("integer_simd_shuffles_and_packs_match_the_host_cpu", ops);
}

#[test]
fn integer_simd_extends_and_tests_match_the_host_cpu() {
    use Kind::{I8, I16, I32, I64};
    let mut ops: Vec<Op> = Vec::new();
    let s41 = Feature::Sse41;
    macro_rules! extend {
        ($k:expr, $leg:ident, $vex:ident, $mem:ident, $mem2:ident) => {{
            ops.push(op(s41, $k, EXACT, |a| a.$leg(xmm1, xmm2)));
            ops.push(op(s41, $k, EXACT, |a| a.$leg(xmm1, $mem(rbx))));
            ops.push(op(AVX, $k, EXACT, |a| a.$vex(xmm1, xmm2)));
            ops.push(op(AVX2, $k, EXACT, |a| a.$vex(ymm1, xmm2)));
            ops.push(op(AVX2, $k, EXACT, |a| a.$vex(ymm1, $mem2(rbx))));
        }};
    }
    extend!(I8, pmovsxbw, vpmovsxbw, qword_ptr, xmmword_ptr);
    extend!(I8, pmovsxbd, vpmovsxbd, dword_ptr, qword_ptr);
    extend!(I8, pmovsxbq, vpmovsxbq, word_ptr, dword_ptr);
    extend!(I16, pmovsxwd, vpmovsxwd, qword_ptr, xmmword_ptr);
    extend!(I16, pmovsxwq, vpmovsxwq, dword_ptr, qword_ptr);
    extend!(I32, pmovsxdq, vpmovsxdq, qword_ptr, xmmword_ptr);
    extend!(I8, pmovzxbw, vpmovzxbw, qword_ptr, xmmword_ptr);
    extend!(I8, pmovzxbd, vpmovzxbd, dword_ptr, qword_ptr);
    extend!(I8, pmovzxbq, vpmovzxbq, word_ptr, dword_ptr);
    extend!(I16, pmovzxwd, vpmovzxwd, qword_ptr, xmmword_ptr);
    extend!(I16, pmovzxwq, vpmovzxwq, dword_ptr, qword_ptr);
    extend!(I32, pmovzxdq, vpmovzxdq, qword_ptr, xmmword_ptr);
    // ptest/vptest/vtestps/vtestpd: random bits rarely give ZF/CF = 1, so equal and
    // complementary inputs are added as dedicated forms.
    for k in [I8, I64] {
        ops.push(op(s41, k, EXACT, |a| a.ptest(xmm1, xmm2)).pairs(false));
        ops.push(op(s41, k, EXACT, |a| a.ptest(xmm1, xmmword_ptr(rbx))));
        ops.push(op(s41, k, EXACT, |a| a.ptest(xmm1, xmm1)));
        ops.push(op(AVX, k, EXACT, |a| a.vptest(ymm1, ymm2)).pairs(false));
        ops.push(op(AVX, k, EXACT, |a| a.vptest(ymm1, ymmword_ptr(rbx))));
        ops.push(op(AVX, k, EXACT, |a| a.vptest(xmm3, xmm2)));
        ops.push(op(AVX, Kind::F32, EXACT, |a| a.vtestps(xmm1, xmm2)).pairs(false));
        ops.push(op(AVX, Kind::F32, EXACT, |a| a.vtestps(ymm1, ymm2)).pairs(false));
        ops.push(op(AVX, Kind::F64, EXACT, |a| a.vtestpd(ymm1, ymmword_ptr(rbx))).pairs(false));
        ops.push(op(AVX, Kind::F64, EXACT, |a| a.vtestpd(xmm1, xmm2)).pairs(false));
    }
    ops.push(op(s41, I16, EXACT, |a| {
        a.pcmpeqd(xmm2, xmm2)?;
        a.ptest(xmm1, xmm2)
    }));
    ops.push(op(s41, I16, EXACT, |a| {
        a.pxor(xmm2, xmm2)?;
        a.ptest(xmm1, xmm2)
    }));
    // Sign masks of 256-bit registers.
    ops.push(op(AVX, Kind::F32, EXACT, |a| a.vmovmskps(eax, ymm2)));
    ops.push(op(AVX, Kind::F64, EXACT, |a| a.vmovmskpd(eax, ymm2)));
    ops.push(op(AVX2, I8, EXACT, |a| a.vpmovmskb(eax, ymm2)));
    ops.push(op(AVX2, I8, EXACT, |a| a.vpmovmskb(rax, ymm2)));
    check_ops("integer_simd_extends_and_tests_match_the_host_cpu", ops);
}
