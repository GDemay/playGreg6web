//! Host-CPU differential tests for double-precision FMA3, AES-NI/VAES, (V)PCLMULQDQ and
//! `phminposuw` (see [`simd_diff`](super::simd_diff) for the harness).

use iced_x86::code_asm::*;

use super::simd_diff::{Check, Feature, Kind, Op, check_ops, op};

const EXACT: Check = Check::Exact;
const N64: Check = Check::Nan(64);

#[test]
fn fma3_double_precision_matches_the_host_cpu() {
    let fma = Feature::Fma;
    let f64k = Kind::F64;
    let mut ops: Vec<Op> = Vec::new();
    macro_rules! fma {
        ($($name:ident),*) => {$(
            ops.push(op(fma, f64k, N64, |a| a.$name(xmm1, xmm3, xmm2)).pairs(false));
            ops.push(op(fma, f64k, N64, |a| a.$name(xmm1, xmm3, xmmword_ptr(rbx))));
            ops.push(op(fma, f64k, N64, |a| a.$name(ymm1, ymm3, ymm2)).pairs(false));
        )*};
    }
    macro_rules! fma_scalar {
        ($($name:ident),*) => {$(
            ops.push(op(fma, f64k, N64, |a| a.$name(xmm1, xmm3, xmm2)).pairs(true));
            ops.push(op(fma, f64k, N64, |a| a.$name(xmm1, xmm3, qword_ptr(rbx))));
        )*};
    }
    fma!(
        vfmadd132pd,
        vfmadd213pd,
        vfmadd231pd,
        vfmsub132pd,
        vfmsub213pd,
        vfmsub231pd
    );
    fma!(
        vfnmadd132pd,
        vfnmadd213pd,
        vfnmadd231pd,
        vfnmsub132pd,
        vfnmsub213pd,
        vfnmsub231pd
    );
    fma!(vfmaddsub213pd, vfmsubadd231pd);
    fma_scalar!(
        vfmadd132sd,
        vfmadd213sd,
        vfmadd231sd,
        vfmsub231sd,
        vfnmadd213sd,
        vfnmsub231sd
    );
    // Cancellation and ties between operands of similar magnitude, and random operands over the
    // whole exponent range (overflow, underflow, subnormal results, far-apart exponents).
    for k in [Kind::F64Near, Kind::F64Any] {
        ops.push(op(fma, k, N64, |a| a.vfmadd231pd(ymm1, ymm3, ymm2)).cases(6000));
        ops.push(op(fma, k, N64, |a| a.vfmsub213sd(xmm1, xmm3, xmm2)).cases(6000));
        ops.push(op(fma, k, N64, |a| a.vfnmadd132pd(xmm1, xmm3, xmm2)).cases(6000));
    }
    check_ops("fma3_double_precision_matches_the_host_cpu", ops);
}

#[test]
fn aes_clmul_and_phminposuw_match_the_host_cpu() {
    let (aes, clmul) = (Feature::Aes, Feature::Pclmulqdq);
    let (avx, vaes, vclmul) = (Feature::Avx, Feature::Vaes, Feature::Vpclmulqdq);
    let k = Kind::I8;
    let mut ops: Vec<Op> = Vec::new();
    macro_rules! aes {
        ($($leg:ident / $vex:ident),*) => {$(
            ops.push(op(aes, k, EXACT, |a| a.$leg(xmm1, xmm2)).cases(200));
            ops.push(op(aes, k, EXACT, |a| a.$leg(xmm1, xmmword_ptr(rbx))));
            ops.push(op(avx, k, EXACT, |a| a.$vex(xmm1, xmm3, xmm2)));
            ops.push(op(vaes, k, EXACT, |a| a.$vex(ymm1, ymm3, ymm2)).cases(100));
            ops.push(op(vaes, k, EXACT, |a| a.$vex(ymm1, ymm3, ymmword_ptr(rbx))));
        )*};
    }
    aes!(aesenc / vaesenc, aesenclast / vaesenclast, aesdec / vaesdec);
    aes!(aesdeclast / vaesdeclast);
    ops.push(op(aes, k, EXACT, |a| a.aesimc(xmm1, xmm2)).cases(200));
    ops.push(op(aes, k, EXACT, |a| a.aesimc(xmm1, xmmword_ptr(rbx))));
    ops.push(op(avx, k, EXACT, |a| a.vaesimc(xmm1, xmm2)));
    for i in [0u32, 1, 2, 4, 8, 0x10, 0x1B, 0x36, 0x80, 0xFF] {
        ops.push(op(aes, k, EXACT, |a| a.aeskeygenassist(xmm1, xmm2, i)).cases(20));
        ops.push(
            op(avx, k, EXACT, |a| {
                a.vaeskeygenassist(xmm1, xmmword_ptr(rbx), i)
            })
            .cases(8),
        );
    }
    for i in [0x00u32, 0x01, 0x10, 0x11, 0xFE, 0xEF] {
        ops.push(op(clmul, Kind::I64, EXACT, |a| a.pclmulqdq(xmm1, xmm2, i)).cases(100));
        ops.push(op(clmul, Kind::I64, EXACT, |a| {
            a.pclmulqdq(xmm1, xmmword_ptr(rbx), i)
        }));
        ops.push(op(avx, Kind::I64, EXACT, |a| {
            a.vpclmulqdq(xmm1, xmm3, xmm2, i)
        }));
        ops.push(op(vclmul, Kind::I64, EXACT, |a| {
            a.vpclmulqdq(ymm1, ymm3, ymm2, i)
        }));
    }
    let s41 = Feature::Sse41;
    ops.push(op(s41, Kind::I16, EXACT, |a| a.phminposuw(xmm1, xmm2)).pairs(false));
    ops.push(op(s41, Kind::I16, EXACT, |a| {
        a.phminposuw(xmm1, xmmword_ptr(rbx))
    }));
    ops.push(op(avx, Kind::I16, EXACT, |a| a.vphminposuw(xmm1, xmm2)));
    // Many equal minima: ties pick the lowest index.
    ops.push(op(s41, Kind::I8, EXACT, |a| a.phminposuw(xmm1, xmm2)).cases(200));
    check_ops("aes_clmul_and_phminposuw_match_the_host_cpu", ops);
}
