//! Execution tests for lifted SSE code under node (tool-gated like [`exec`](super::exec)): the
//! Win64 floating point calling convention across lifted calls and the import seam, the MXCSR
//! policy and `vzeroall`/`vzeroupper`.

use iced_x86::code_asm::*;

use super::exec::{
    CODE_BASE, Case, R8, RAX, RCX, RDX, RSP, STACK_TOP, STATE_ADDR, build, exec_or_skip, run,
};
use super::ir_simd::random_sse_program;
use super::{Program, asm_function};
use crate::lift::{STATE_XMM_OFFSET, STATE_YMM_HI_OFFSET, lift_module};

const XMM_STATE: u64 = STATE_ADDR + STATE_XMM_OFFSET;
const YMMH_STATE: u64 = STATE_ADDR + STATE_YMM_HI_OFFSET;
/// Host function in `runner.js`: `xmm0 = xmm0 * xmm1 + r8` (doubles), then `ret`.
const HOST_FMA_IMPORT: u64 = 0x7FFE_0010;

fn xmm_bytes(lo: u64, hi: u64) -> Vec<u8> {
    [lo.to_le_bytes(), hi.to_le_bytes()].concat()
}

#[test]
fn win64_float_calls_execute() {
    if !exec_or_skip("win64_float_calls_execute") {
        return;
    }
    let caller = CODE_BASE;
    let callee_d = CODE_BASE + 0x1000;
    let callee_f = CODE_BASE + 0x2000;
    // double callee_d(double a, float b, int c, double d) = 2a + b + c + d.
    // Uses callee-saved xmm6 (saved/restored through the stack) and clobbers volatile xmm2..5.
    let program = Program::new()
        .asm(callee_d, |a| {
            a.sub(rsp, 0x28)?;
            a.movaps(xmmword_ptr(rsp + 0x10), xmm6)?;
            a.cvtss2sd(xmm6, xmm1)?;
            a.addsd(xmm0, xmm0)?;
            a.addsd(xmm0, xmm6)?;
            a.cvtsi2sd(xmm6, r8d)?;
            a.addsd(xmm0, xmm6)?;
            a.addsd(xmm0, xmm3)?;
            a.xorps(xmm2, xmm2)?;
            a.movaps(xmm4, xmm2)?;
            a.cmpeqps(xmm4, xmm2)?;
            a.movaps(xmm5, xmm4)?;
            a.movaps(xmm6, xmmword_ptr(rsp + 0x10))?;
            a.add(rsp, 0x28)?;
            a.ret()
        })
        // float callee_f(float x, int y) = x * y
        .asm(callee_f, |a| {
            a.cvtsi2ss(xmm1, edx)?;
            a.mulss(xmm0, xmm1)?;
            a.ret()
        })
        .asm(caller, |a| {
            a.push(rbx)?;
            a.sub(rsp, 0x40)?;
            a.movaps(xmmword_ptr(rsp + 0x20), xmm6)?;
            a.movaps(xmmword_ptr(rsp + 0x30), xmm7)?;
            a.movq(xmm6, rcx)?;
            a.mov(rax, 2.5f64.to_bits())?;
            a.movq(xmm0, rax)?;
            a.mov(eax, 1.5f32.to_bits())?;
            a.movd(xmm1, eax)?;
            a.mov(r8d, 7)?;
            a.mov(rax, 1.0f64.to_bits())?;
            a.movq(xmm3, rax)?;
            a.call(callee_d)?; // 14.5
            a.movapd(xmm7, xmm0)?;
            a.mov(eax, 3.0f32.to_bits())?;
            a.movd(xmm0, eax)?;
            a.mov(edx, -4)?;
            a.call(callee_f)?; // -12.0f
            a.cvtss2sd(xmm0, xmm0)?;
            a.addsd(xmm7, xmm0)?; // 2.5
            // Import through the runtime dispatcher: 2.5 * 4.0 + 3 = 13.0 in xmm0.
            a.movapd(xmm0, xmm7)?;
            a.mov(rax, 4.0f64.to_bits())?;
            a.movq(xmm1, rax)?;
            a.mov(r8d, 3)?;
            a.mov(rax, HOST_FMA_IMPORT)?;
            a.call(rax)?;
            a.movq(rax, xmm0)?;
            a.movq(rdx, xmm6)?;
            a.movaps(xmm6, xmmword_ptr(rsp + 0x20))?;
            a.movaps(xmm7, xmmword_ptr(rsp + 0x30))?;
            a.add(rsp, 0x40)?;
            a.pop(rbx)?;
            a.ret()
        });
    let functions = program.discover();
    let built = build("win64_float", functions.values());

    let saved6 = xmm_bytes(0x0606_0606_0606_0606, 0x6666_6666_6666_6666);
    let saved7 = xmm_bytes(0x0707_0707_0707_0707, 0x7777_7777_7777_7777);
    let mut case = Case::new(caller).reg(RCX, 0x4142_4344_4546_4748);
    let mut state = vec![0u8; 16 * 16];
    state[16 * 6..16 * 7].copy_from_slice(&saved6);
    state[16 * 7..16 * 8].copy_from_slice(&saved7);
    case.mem.push((XMM_STATE, state));
    case.read.push((XMM_STATE, 16 * 16));
    let o = run(&built, &[case]);
    assert!(o[0].error.is_empty(), "trapped: {}", o[0].error);
    assert_eq!(o[0].regs[RSP], STACK_TOP);
    assert_eq!(f64::from_bits(o[0].regs[RAX]), 13.0);
    assert_eq!(
        o[0].regs[RDX], 0x4142_4344_4546_4748,
        "xmm6 preserved by callees"
    );
    let xmm = &o[0].mem[0];
    assert_eq!(
        xmm[..16],
        xmm_bytes(13.0f64.to_bits(), 0)[..],
        "xmm0 result"
    );
    assert_eq!(xmm[16 * 6..16 * 7], saved6[..], "caller restores its xmm6");
    assert_eq!(xmm[16 * 7..16 * 8], saved7[..], "caller restores its xmm7");
    assert_eq!(
        xmm[16 * 2..16 * 3],
        [0u8; 16][..],
        "volatile xmm2 clobbered by the callee"
    );
    assert_eq!(
        xmm[16 * 5..16 * 6],
        [0xFFu8; 16][..],
        "volatile xmm5 clobbered by the callee"
    );
}

#[test]
fn mxcsr_policy_executes() {
    if !exec_or_skip("mxcsr_policy_executes") {
        return;
    }
    let store = CODE_BASE;
    let load = CODE_BASE + 0x100;
    let outer = CODE_BASE + 0x200;
    let program = Program::new()
        .asm(store, |a| {
            a.stmxcsr(dword_ptr(rcx))?;
            a.vstmxcsr(dword_ptr(rcx + 4))?;
            a.mov(eax, dword_ptr(rcx))?;
            a.mov(edx, dword_ptr(rcx + 4))?;
            a.ret()
        })
        .asm(load, |a| {
            a.ldmxcsr(dword_ptr(rcx))?;
            a.vldmxcsr(dword_ptr(rcx))?;
            a.stmxcsr(dword_ptr(rcx + 4))?;
            a.mov(eax, dword_ptr(rcx + 4))?;
            a.ret()
        })
        // MXCSR lives in the guest state: a value loaded by a callee is seen by its caller.
        .asm(outer, |a| {
            a.sub(rsp, 0x28)?;
            a.call(load)?;
            a.add(rsp, 0x28)?;
            a.vstmxcsr(dword_ptr(rcx + 8))?;
            a.mov(edx, dword_ptr(rcx + 8))?;
            a.ret()
        });
    let functions = program.discover();
    let ldmxcsr_va = functions[&load].blocks[&load].instructions[0].ip();
    let built = build("mxcsr", functions.values());
    let buf = 0x18_0000u64;
    let with = |va: u64, value: u32| {
        let mut c = Case::new(va).reg(RCX, buf);
        c.mem.push((buf, value.to_le_bytes().to_vec()));
        c
    };
    let values = [
        0x1F80u32, 0x9FC0, 0x1F00, 0x0000, 0x3F80, 0x5F80, 0x7F80, 0x1_1F80,
    ];
    let mut cases = vec![with(store, 0)];
    cases.extend(values.iter().map(|&v| with(outer, v)));
    let o = run(&built, &cases);
    assert!(o[0].error.is_empty(), "{}", o[0].error);
    assert_eq!(o[0].regs[RAX], 0x1F80, "stmxcsr of a new thread's MXCSR");
    assert_eq!(o[0].regs[RDX], 0x1F80, "vstmxcsr of a new thread's MXCSR");
    for (v, out) in values.iter().zip(&o[1..]) {
        if v & 0xFFFF_6000 == 0 {
            // FTZ/DAZ and exception masks are stored (and round-trip) but not modelled.
            assert!(out.error.is_empty(), "{v:#x}: {}", out.error);
            assert_eq!(out.regs[RAX], u64::from(*v), "stmxcsr after ldmxcsr");
            assert_eq!(
                out.regs[RDX],
                u64::from(*v),
                "the caller sees the callee's MXCSR"
            );
        } else {
            // Non-default rounding modes and reserved bits fail loudly.
            assert_eq!(
                out.error,
                format!("unimplemented instruction at {ldmxcsr_va:#x}"),
                "{v:#x}"
            );
        }
    }
}

#[test]
fn vzeroall_and_vzeroupper_execute() {
    if !exec_or_skip("vzeroall_and_vzeroupper_execute") {
        return;
    }
    let zero_all = asm_function(CODE_BASE, |a| {
        a.vzeroall()?;
        a.ret()
    });
    let zero_upper = asm_function(CODE_BASE + 0x100, |a| {
        a.vzeroupper()?;
        a.vaddps(xmm1, xmm1, xmm1)?;
        a.ret()
    });
    let built = build("vzero", [&zero_all, &zero_upper]);
    let state: Vec<u8> = (0..256).map(|i| (i * 7 + 1) as u8).collect();
    let mut cases = Vec::new();
    for va in [CODE_BASE, CODE_BASE + 0x100] {
        let mut c = Case::new(va);
        c.mem.push((XMM_STATE, state.clone()));
        c.read.push((XMM_STATE, 256));
        cases.push(c);
    }
    let o = run(&built, &cases);
    assert!(o.iter().all(|o| o.error.is_empty()));
    assert_eq!(o[0].mem[0], vec![0u8; 256], "vzeroall clears xmm0..15");
    let mut expected = state.clone();
    for lane in 0..4 {
        let at = 16 + 4 * lane;
        let v = f32::from_le_bytes(state[at..at + 4].try_into().unwrap());
        expected[at..at + 4].copy_from_slice(&(v + v).to_le_bytes());
    }
    // vaddps xmm1, xmm1, xmm1 (the NaN-free lanes of this pattern) after vzeroupper
    let got = &o[1].mem[0];
    assert_eq!(got[..16], state[..16]);
    assert_eq!(got[32..], state[32..]);
    for lane in 0..4 {
        let at = 16 + 4 * lane;
        let e = f32::from_le_bytes(expected[at..at + 4].try_into().unwrap());
        let g = f32::from_le_bytes(got[at..at + 4].try_into().unwrap());
        assert!(
            e.to_bits() == g.to_bits() || (e.is_nan() && g.is_nan()),
            "lane {lane}"
        );
    }
}

#[test]
fn lifted_random_sse_code_compiles() {
    if !exec_or_skip("lifted_random_sse_code_compiles") {
        return;
    }
    let functions = random_sse_program(200);
    let module = lift_module(functions.values()).expect("lifts");
    let lifted: usize = module
        .functions
        .iter()
        .map(|f| f.ir.matches("load <2 x i64>, ptr %xmm").count())
        .sum();
    assert!(lifted > 100, "random SSE code should lift XMM operations");
    // `build` compiles and links the module; a lifter IR bug fails here.
    let _ = build("random_sse", functions.values());
}

/// Reference model of SSE4a `extrq`/`insertq` (AMD only, so not differentially testable on
/// every host): low-quadword bit fields; the undefined parts as the lifter defines them.
fn sse4a_model(dst: u64, src: u64, len: u64, idx: u64, insert: bool) -> u64 {
    let (len, idx) = (len & 63, idx & 63);
    let mask = if len == 0 { u64::MAX } else { (1 << len) - 1 };
    if insert {
        (dst & !(mask << idx)) | ((src & mask) << idx)
    } else {
        (dst >> idx) & mask
    }
}

#[test]
fn sse4a_bit_fields_execute() {
    if !exec_or_skip("sse4a_bit_fields_execute") {
        return;
    }
    let fields = [
        (0u32, 0u32),
        (1, 0),
        (8, 4),
        (16, 48),
        (32, 31),
        (63, 1),
        (5, 59),
        (0, 7),
    ];
    let mut program = Program::new();
    for (i, &(len, idx)) in fields.iter().enumerate() {
        let va = CODE_BASE + 0x100 * i as u64;
        program = program
            .asm(va, |a| {
                a.extrq_3(xmm1, len, idx)?;
                a.ret()
            })
            .asm(va + 0x40, |a| {
                a.insertq_4(xmm1, xmm2, len, idx)?;
                a.ret()
            });
    }
    let reg_extrq = CODE_BASE + 0x2000;
    let reg_insertq = CODE_BASE + 0x2040;
    let program = program
        .asm(reg_extrq, |a| {
            a.extrq(xmm1, xmm2)?;
            a.ret()
        })
        .asm(reg_insertq, |a| {
            a.insertq(xmm1, xmm2)?;
            a.ret()
        });
    let functions = program.discover();
    let built = build("sse4a", functions.values());
    let (d0, d1) = (0x0123_4567_89AB_CDEFu64, 0xFEDC_BA98_7654_3210u64);
    let s0 = 0xA5A5_5A5A_F00F_0FF0u64;
    let state = |x1: [u64; 2], x2: [u64; 2]| {
        let mut b = vec![0u8; 48];
        b[16..24].copy_from_slice(&x1[0].to_le_bytes());
        b[24..32].copy_from_slice(&x1[1].to_le_bytes());
        b[32..40].copy_from_slice(&x2[0].to_le_bytes());
        b[40..48].copy_from_slice(&x2[1].to_le_bytes());
        b
    };
    let case = |va: u64, x2: [u64; 2]| {
        let mut c = Case::new(va);
        c.mem.push((XMM_STATE, state([d0, d1], x2)));
        c.read.push((XMM_STATE + 16, 16));
        c
    };
    let mut cases = Vec::new();
    let mut expected = Vec::new();
    for (i, &(len, idx)) in fields.iter().enumerate() {
        let va = CODE_BASE + 0x100 * i as u64;
        let (l, x) = (u64::from(len), u64::from(idx));
        cases.push(case(va, [s0, 0x77]));
        expected.push(sse4a_model(d0, s0, l, x, false));
        cases.push(case(va + 0x40, [s0, 0x77]));
        expected.push(sse4a_model(d0, s0, l, x, true));
        // Register forms take the field from xmm2 (bits 5:0 / 13:8, or 69:64 / 77:72).
        cases.push(case(reg_extrq, [l | x << 8 | 0xC0C0, 0]));
        expected.push(sse4a_model(d0, 0, l, x, false));
        cases.push(case(reg_insertq, [s0, l | x << 8 | 0xC0C0]));
        expected.push(sse4a_model(d0, s0, l, x, true));
    }
    let o = run(&built, &cases);
    for (i, (out, want)) in o.iter().zip(&expected).enumerate() {
        assert!(out.error.is_empty(), "case {i}: {}", out.error);
        let lo = u64::from_le_bytes(out.mem[0][..8].try_into().unwrap());
        let hi = u64::from_le_bytes(out.mem[0][8..].try_into().unwrap());
        assert_eq!(lo, *want, "case {i}");
        assert_eq!(hi, d1, "case {i}: upper quadword preserved");
    }
}

#[test]
fn r59_wasm64_simd128_codegen_and_edge_cases() {
    if !exec_or_skip("r59_wasm64_simd128_codegen_and_edge_cases") {
        return;
    }
    use crate::lift::STATE_YMM_HI_OFFSET;
    use crate::lift::compile::{LlvmTool, find_llvm_tool};

    const YMMH_STATE: u64 = STATE_ADDR + STATE_YMM_HI_OFFSET;

    // 1) Verify LLVM emits native Wasm64 SIMD128 instructions (no stack spills or scalar loops).
    if let Some(LlvmTool::Llc { llc, opt }) = find_llvm_tool() {
        let dir = std::env::temp_dir().join(format!(
            "playgta6_r59_asm_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let probes: Vec<(&str, crate::cfg::Function)> = vec![
            (
                "psadbw",
                asm_function(CODE_BASE, |a| {
                    a.psadbw(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "vpermd",
                asm_function(CODE_BASE + 0x100, |a| {
                    a.vpermd(ymm0, ymm1, ymm2)?;
                    a.ret()
                }),
            ),
            (
                "vpermilps",
                asm_function(CODE_BASE + 0x200, |a| {
                    a.vpermilps(xmm0, xmm1, xmm2)?;
                    a.ret()
                }),
            ),
            (
                "vpermilpd",
                asm_function(CODE_BASE + 0x300, |a| {
                    a.vpermilpd(xmm0, xmm1, xmm2)?;
                    a.ret()
                }),
            ),
            (
                "vptest",
                asm_function(CODE_BASE + 0x400, |a| {
                    a.vptest(ymm0, ymm1)?;
                    a.ret()
                }),
            ),
            (
                "pmuludq",
                asm_function(CODE_BASE + 0x500, |a| {
                    a.paddd(xmm0, xmm2)?;
                    a.paddd(xmm1, xmm2)?;
                    a.pmuludq(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "pmuldq",
                asm_function(CODE_BASE + 0x600, |a| {
                    a.paddd(xmm0, xmm2)?;
                    a.paddd(xmm1, xmm2)?;
                    a.pmuldq(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "cvtpd2ps",
                asm_function(CODE_BASE + 0x700, |a| {
                    a.cvtpd2ps(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "cvtpd2dq",
                asm_function(CODE_BASE + 0x800, |a| {
                    a.cvtpd2dq(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "phminposuw",
                asm_function(CODE_BASE + 0x900, |a| {
                    a.phminposuw(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
        ];
        for (name, func) in &probes {
            let m = lift_module([func]).expect("lift");
            let ll = dir.join(format!("{name}.ll"));
            let bc = dir.join(format!("{name}.bc"));
            let asm_path = dir.join(format!("{name}.s"));
            std::fs::write(&ll, &m.ir).expect("write ll");
            let input = if let Some(opt_bin) = &opt {
                let st = std::process::Command::new(opt_bin)
                    .args(["-O2"])
                    .arg(&ll)
                    .arg("-o")
                    .arg(&bc)
                    .status()
                    .expect("run opt");
                assert!(st.success(), "opt failed on {name}");
                &bc
            } else {
                &ll
            };
            let st = std::process::Command::new(&llc)
                .args(["-mtriple=wasm64-unknown-unknown", "-O2", "-filetype=asm"])
                .arg(input)
                .arg("-o")
                .arg(&asm_path)
                .status()
                .expect("run llc");
            assert!(st.success(), "llc failed on {name}");
            let asm = std::fs::read_to_string(&asm_path).expect("read asm");
            match *name {
                "psadbw" => {
                    assert!(asm.contains("i8x16.max_u"), "{asm}");
                    assert!(asm.contains("i8x16.min_u"), "{asm}");
                    assert!(!asm.contains("extract_lane"), "psadbw scalarized:\n{asm}");
                }
                "vpermd" | "vpermilps" => {
                    assert!(asm.contains("i8x16.swizzle"), "{name}:\n{asm}");
                    assert!(!asm.contains("__stack_pointer"), "{name} spilled stack:\n{asm}");
                }
                "vpermilpd" => {
                    assert!(asm.contains("v128.bitselect"), "{asm}");
                    assert!(!asm.contains("__stack_pointer"), "vpermilpd spilled stack:\n{asm}");
                }
                "vptest" => {
                    assert!(asm.contains("v128.any_true"), "{asm}");
                    assert!(!asm.contains("i64x2.extract_lane"), "vptest extracted lanes:\n{asm}");
                }
                "pmuludq" => {
                    // Newer LLVM (23+) `opt -O2` canonicalizes zext(shuffle) into an `and` with
                    // 0xFFFFFFFF that the wasm backend no longer folds back into `extmul`; the
                    // masked `i64x2.mul` is still a pure SIMD128 sequence (no scalarization).
                    assert!(
                        asm.contains("i64x2.extmul_low_i32x4_u")
                            || (asm.contains("v128.and") && asm.contains("4294967295")),
                        "{asm}"
                    );
                    assert!(!asm.contains("extract_lane"), "pmuludq scalarized:
{asm}");
                }
                "pmuldq" => {
                    // See pmuludq: newer LLVM may keep a shl/shr_s sign-extension + `i64x2.mul`.
                    assert!(
                        asm.contains("i64x2.extmul_low_i32x4_s")
                            || (asm.contains("i64x2.shr_s") && asm.contains("i64x2.mul")),
                        "{asm}"
                    );
                    assert!(!asm.contains("extract_lane"), "pmuldq scalarized:
{asm}");
                }
                "cvtpd2ps" => {
                    assert!(asm.contains("f32x4.demote_f64x2_zero"), "{asm}");
                }
                "cvtpd2dq" => {
                    assert!(asm.contains("i32x4.trunc_sat_f64x2_s_zero"), "{asm}");
                }
                "phminposuw" => {
                    assert!(asm.contains("i16x8.min_u"), "{asm}");
                    assert!(asm.contains("i16x8.bitmask"), "{asm}");
                }
                _ => {}
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 2) Execute critical edge cases under Wasm64 (Node):
    //    - pmulhrsw(-32768, -32768) == -32768 (wraps, does NOT saturate to 32767)
    //    - vpmulhrsw (VEX-128) zeroes ymm0[128..255]
    //    - pmaddwd(-32768, -32768) == -2147483648
    //    - psadbw max byte diff (0 vs 255 -> [2040, 2040])
    //    - cvttpd2dq on -2147483648.9 and +inf -> [-2147483648, -2147483648, 0, 0]
    let f_pmulhrsw = CODE_BASE;
    let f_vpmulhrsw = CODE_BASE + 0x100;
    let f_pmaddwd = CODE_BASE + 0x200;
    let f_psadbw = CODE_BASE + 0x300;
    let f_cvttpd2dq = CODE_BASE + 0x400;
    let f_phminposuw = CODE_BASE + 0x500;
    let program = Program::new()
        .asm(f_pmulhrsw, |a| {
            a.pmulhrsw(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_vpmulhrsw, |a| {
            a.vpmulhrsw(xmm0, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_pmaddwd, |a| {
            a.pmaddwd(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_psadbw, |a| {
            a.psadbw(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_cvttpd2dq, |a| {
            a.cvttpd2dq(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_phminposuw, |a| {
            a.phminposuw(xmm0, xmm1)?;
            a.ret()
        });
    let functions = program.discover();
    let built = build("r59_edge", functions.values());

    let words = |ws: [i16; 8]| -> Vec<u8> { ws.iter().flat_map(|w| w.to_le_bytes()).collect() };

    // pmulhrsw edge pairs
    let mut c_hrsw = Case::new(f_pmulhrsw);
    let mut xmms = vec![0u8; 48];
    xmms[0..16].copy_from_slice(&words([-32768, -32768, -1, 32767, 16384, -16384, 0, 1]));
    xmms[16..32].copy_from_slice(&words([-32768, 32767, 1, 32767, 16384, 16384, -32768, 16384]));
    c_hrsw.mem.push((XMM_STATE, xmms.clone()));
    c_hrsw.read.push((XMM_STATE, 16));

    // vpmulhrsw VEX-128 zeroing of upper 128 bits
    let mut c_vhrsw = Case::new(f_vpmulhrsw);
    let mut vxmms = vec![0u8; 48];
    vxmms[16..32].copy_from_slice(&words([-32768; 8]));
    vxmms[32..48].copy_from_slice(&words([-32768; 8]));
    c_vhrsw.mem.push((XMM_STATE, vxmms));
    c_vhrsw.mem.push((YMMH_STATE, vec![0xA5u8; 16]));
    c_vhrsw.read.push((XMM_STATE, 16));
    c_vhrsw.read.push((YMMH_STATE, 16));

    // pmaddwd(-32768, -32768)
    let mut c_madd = Case::new(f_pmaddwd);
    let mut mxmms = vec![0u8; 32];
    mxmms[0..16].copy_from_slice(&words([-32768; 8]));
    mxmms[16..32].copy_from_slice(&words([-32768; 8]));
    c_madd.mem.push((XMM_STATE, mxmms));
    c_madd.read.push((XMM_STATE, 16));

    // psadbw(0x00.., 0xFF..) -> [2040, 2040]
    let mut c_sad = Case::new(f_psadbw);
    let mut sxmms = vec![0u8; 32];
    sxmms[16..32].fill(0xFF);
    c_sad.mem.push((XMM_STATE, sxmms));
    c_sad.read.push((XMM_STATE, 16));

    // cvttpd2dq([-2147483648.9, +inf]) -> [0x80000000, 0x80000000, 0, 0]
    let mut c_cvt = Case::new(f_cvttpd2dq);
    let mut cxmms = vec![0u8; 32];
    cxmms[16..24].copy_from_slice(&(-2147483648.9f64).to_le_bytes());
    cxmms[24..32].copy_from_slice(&f64::INFINITY.to_le_bytes());
    c_cvt.mem.push((XMM_STATE, cxmms));
    c_cvt.read.push((XMM_STATE, 16));

    // phminposuw tie-breaking: lanes 2 and 5 both hold 7 -> min=7, index=2
    let mut c_minpos = Case::new(f_phminposuw);
    let mut pxmms = vec![0u8; 32];
    pxmms[16..32].copy_from_slice(&words([100, 50, 7, 99, 8, 7, 1000, 12]));
    c_minpos.mem.push((XMM_STATE, pxmms));
    c_minpos.read.push((XMM_STATE, 16));

    let out = run(&built, &[c_hrsw, c_vhrsw, c_madd, c_sad, c_cvt, c_minpos]);
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }
    assert_eq!(
        out[0].mem[0],
        words([-32768, -32767, 0, 32766, 8192, -8192, 0, 1]),
        "pmulhrsw edge cases"
    );
    assert_eq!(out[1].mem[0], words([-32768; 8]), "vpmulhrsw lo");
    assert_eq!(out[1].mem[1], vec![0u8; 16], "vpmulhrsw VEX-128 zeroes ymm0_hi");
    assert_eq!(
        out[2].mem[0],
        [-2147483648i32; 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        "pmaddwd -32768*-32768 wrap"
    );
    assert_eq!(out[3].mem[0], xmm_bytes(2040, 2040), "psadbw max diff");
    assert_eq!(
        out[4].mem[0],
        [-2147483648i32, -2147483648, 0, 0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
        "cvttpd2dq edge cases"
    );
    assert_eq!(
        out[5].mem[0],
        words([7, 2, 0, 0, 0, 0, 0, 0]),
        "phminposuw tie-breaking"
    );
}

/// AVX-512VL `vpmin/vpmax{s,u}q` on XMM and YMM registers (no opmask, no broadcast) compute their
/// VEX-style lane-wise result and zero the upper half of an XMM destination.
#[test]
fn evex_quadword_min_max_execute() {
    if !exec_or_skip("evex_quadword_min_max_execute") {
        return;
    }
    const YMMH_STATE: u64 = STATE_ADDR + crate::lift::STATE_YMM_HI_OFFSET;
    let f_minsq = CODE_BASE;
    let f_minuq = CODE_BASE + 0x100;
    let f_maxsq = CODE_BASE + 0x200;
    let f_maxuq = CODE_BASE + 0x300;
    let f_ymm = CODE_BASE + 0x400;
    let program = Program::new()
        .asm(f_minsq, |a| {
            a.vpminsq(xmm1, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_minuq, |a| {
            a.vpminuq(xmm1, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_maxsq, |a| {
            a.vpmaxsq(xmm1, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_maxuq, |a| {
            a.vpmaxuq(xmm1, xmm0, xmm2)?;
            a.ret()
        })
        .asm(f_ymm, |a| {
            a.vpminuq(ymm1, ymm1, ymm2)?;
            a.ret()
        });
    let functions = program.discover();
    let built = build("evex_qword_minmax", functions.values());

    let a = [i64::MIN as u64, 5, u64::MAX, 0x8000_0000_0000_0001];
    let b = [i64::MAX as u64, (-3i64) as u64, 7, 0x8000_0000_0000_0000];
    // Registers xmm0..2 (16 bytes each), then the upper halves ymmh0..2.
    let state = |x0: [u64; 2], x1: [u64; 2], x2: [u64; 2]| {
        [x0, x1, x2]
            .iter()
            .flat_map(|x| xmm_bytes(x[0], x[1]))
            .collect::<Vec<u8>>()
    };
    let case = |func: u64| {
        let mut c = Case::new(func);
        c.mem
            .push((XMM_STATE, state([a[2], b[2]], [a[0], a[1]], [b[0], b[1]])));
        c.mem.push((YMMH_STATE, vec![0xA5u8; 3 * 16]));
        c.read.push((XMM_STATE + 16, 16));
        c.read.push((YMMH_STATE + 16, 16));
        c
    };
    let mut cases = vec![case(f_minsq), case(f_minuq), case(f_maxsq), case(f_maxuq)];
    // 256-bit: both halves are processed.
    let mut wide = Case::new(f_ymm);
    wide.mem
        .push((XMM_STATE, state([0, 0], [a[0], a[1]], [b[0], b[1]])));
    wide.mem.push((
        YMMH_STATE,
        [
            xmm_bytes(0, 0),
            xmm_bytes(a[2], a[3]),
            xmm_bytes(b[2], b[3]),
        ]
        .concat(),
    ));
    wide.read.push((XMM_STATE + 16, 16));
    wide.read.push((YMMH_STATE + 16, 16));
    cases.push(wide);
    let out = run(&built, &cases);
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }
    let lanes =
        |f: fn(u64, u64) -> u64, x: [u64; 2], y: [u64; 2]| xmm_bytes(f(x[0], y[0]), f(x[1], y[1]));
    let (x1, x2) = ([a[0], a[1]], [b[0], b[1]]);
    assert_eq!(
        out[0].mem[0],
        lanes(|p, q| (p as i64).min(q as i64) as u64, x1, x2),
        "vpminsq"
    );
    assert_eq!(out[1].mem[0], lanes(u64::min, x1, x2), "vpminuq");
    assert_eq!(
        out[2].mem[0],
        lanes(|p, q| (p as i64).max(q as i64) as u64, x1, x2),
        "vpmaxsq"
    );
    // vpmaxuq xmm1, xmm0, xmm2: first source is xmm0.
    assert_eq!(
        out[3].mem[0],
        lanes(u64::max, [a[2], b[2]], x2),
        "vpmaxuq (three operand)"
    );
    for o in &out[..4] {
        assert_eq!(
            o.mem[1],
            vec![0u8; 16],
            "VEX/EVEX-128 zeroes the upper half"
        );
    }
    assert_eq!(
        out[4].mem[0],
        lanes(u64::min, x1, x2),
        "vpminuq ymm low half"
    );
    assert_eq!(
        out[4].mem[1],
        lanes(u64::min, [a[2], a[3]], [b[2], b[3]]),
        "vpminuq ymm high half"
    );
}

/// `rdsspq` / `rdsspd` without an enabled shadow stack leave the destination register unchanged.
#[test]
fn rdssp_without_shadow_stack_is_a_nop() {
    if !exec_or_skip("rdssp_without_shadow_stack_is_a_nop") {
        return;
    }
    let f = CODE_BASE;
    let program = Program::new().asm(f, |a| {
        a.rdsspq(r10)?;
        a.rdsspd(r11d)?;
        a.ret()
    });
    let functions = program.discover();
    let built = build("rdssp_nop", functions.values());
    let c = Case::new(f)
        .reg(10, 0x1122_3344_5566_7788)
        .reg(11, 0x99AA_BBCC_DDEE_FF00);
    let out = run(&built, &[c]);
    assert!(out[0].error.is_empty(), "trapped: {}", out[0].error);
    assert_eq!(out[0].regs[10], 0x1122_3344_5566_7788);
    assert_eq!(out[0].regs[11], 0x99AA_BBCC_DDEE_FF00);
}

#[test]
fn r66_wasm64_simd128_codegen_and_edge_cases() {
    if !exec_or_skip("r66_wasm64_simd128_codegen_and_edge_cases") {
        return;
    }
    use crate::lift::compile::{LlvmTool, find_llvm_tool};
    use crate::lift::target::LiftTarget;

    // 1) Verify LLVM emits exact 1-op Wasm64 SIMD128 instructions and compiles cleanly for native
    //    x86_64-pc-windows-msvc (`x86-64-v3`).
    if let Some(LlvmTool::Llc { llc, opt }) = find_llvm_tool() {
        let dir = std::env::temp_dir().join(format!(
            "playgta6_r66_asm_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let probes: Vec<(&str, crate::cfg::Function, &str, Option<&str>)> = vec![
            (
                "packsswb",
                asm_function(CODE_BASE, |a| {
                    a.packsswb(xmm0, xmm1)?;
                    a.ret()
                }),
                "i8x16.narrow_i16x8_s",
                Some("packsswb"),
            ),
            (
                "packuswb",
                asm_function(CODE_BASE + 0x100, |a| {
                    a.packuswb(xmm0, xmm1)?;
                    a.ret()
                }),
                "i8x16.narrow_i16x8_u",
                Some("packuswb"),
            ),
            (
                "packssdw",
                asm_function(CODE_BASE + 0x200, |a| {
                    a.packssdw(xmm0, xmm1)?;
                    a.ret()
                }),
                "i16x8.narrow_i32x4_s",
                Some("packssdw"),
            ),
            (
                "packusdw",
                asm_function(CODE_BASE + 0x300, |a| {
                    a.packusdw(xmm0, xmm1)?;
                    a.ret()
                }),
                "i16x8.narrow_i32x4_u",
                Some("packusdw"),
            ),
            (
                "pmaddwd",
                asm_function(CODE_BASE + 0x400, |a| {
                    a.pmaddwd(xmm0, xmm1)?;
                    a.ret()
                }),
                "i32x4.dot_i16x8_s",
                Some("pmaddwd"),
            ),
            (
                "pavgb",
                asm_function(CODE_BASE + 0x500, |a| {
                    a.pavgb(xmm0, xmm1)?;
                    a.ret()
                }),
                "i8x16.avgr_u",
                Some("pavgb"),
            ),
            (
                "pavgw",
                asm_function(CODE_BASE + 0x600, |a| {
                    a.pavgw(xmm0, xmm1)?;
                    a.ret()
                }),
                "i16x8.avgr_u",
                Some("pavgw"),
            ),
            (
                "psrlw_imm",
                asm_function(CODE_BASE + 0x700, |a| {
                    a.psrlw(xmm0, 3)?;
                    a.ret()
                }),
                "i16x8.shr_u",
                None,
            ),
            (
                "dpps_7f",
                asm_function(CODE_BASE + 0x800, |a| {
                    a.dpps(xmm0, xmm1, 0x7F)?;
                    a.ret()
                }),
                "f32x4.add",
                None,
            ),
        ];
        let native_target = LiftTarget::parse("native:x86_64-pc-windows-msvc", Some("x86-64-v3"))
            .expect("native target");
        for (name, func, want_wasm, want_x86) in &probes {
            let m = lift_module([func]).expect("lift");
            let ll = dir.join(format!("{name}.ll"));
            let bc = dir.join(format!("{name}.bc"));
            let asm_path = dir.join(format!("{name}.s"));
            std::fs::write(&ll, &m.ir).expect("write ll");
            let input = if let Some(opt_bin) = &opt {
                let st = std::process::Command::new(opt_bin)
                    .args(["-O2"])
                    .arg(&ll)
                    .arg("-o")
                    .arg(&bc)
                    .status()
                    .expect("run opt");
                assert!(st.success(), "opt failed on {name}");
                &bc
            } else {
                &ll
            };
            let st = std::process::Command::new(&llc)
                .args(["-mtriple=wasm64-unknown-unknown", "-O2", "-filetype=asm"])
                .arg(input)
                .arg("-o")
                .arg(&asm_path)
                .status()
                .expect("run llc wasm64");
            assert!(st.success(), "llc wasm64 failed on {name}");
            let asm = std::fs::read_to_string(&asm_path).expect("read wasm asm");
            assert!(
                asm.contains(want_wasm),
                "{name} expected `{want_wasm}` in wasm64 asm:\n{asm}"
            );
            if *name == "dpps_7f" {
                assert!(!asm.contains("extract_lane"), "dpps extracted lanes:\n{asm}");
                assert!(!asm.contains("f32.mul"), "dpps scalarized fmul:\n{asm}");
                assert!(!asm.contains("f32.add"), "dpps scalarized fadd:\n{asm}");
            }

            if let Some(x86_mnem) = want_x86 {
                let native_ir = native_target.retarget(m.ir);
                let nll = dir.join(format!("{name}_x86.ll"));
                let nbc = dir.join(format!("{name}_x86.bc"));
                let nasm = dir.join(format!("{name}_x86.s"));
                std::fs::write(&nll, &native_ir).expect("write native ll");
                let ninput = if let Some(opt_bin) = &opt {
                    let st = std::process::Command::new(opt_bin)
                        .args(["-O2"])
                        .arg(&nll)
                        .arg("-o")
                        .arg(&nbc)
                        .status()
                        .expect("run opt native");
                    assert!(st.success(), "opt native failed on {name}");
                    &nbc
                } else {
                    &nll
                };
                let st = std::process::Command::new(&llc)
                    .args([
                        "-mtriple=x86_64-pc-windows-msvc",
                        "-mcpu=x86-64-v3",
                        "-O2",
                        "-filetype=asm",
                    ])
                    .arg(ninput)
                    .arg("-o")
                    .arg(&nasm)
                    .status()
                    .expect("run llc native");
                assert!(st.success(), "llc native failed on {name}");
                let x86_asm = std::fs::read_to_string(&nasm).expect("read x86 asm");
                assert!(
                    x86_asm.contains(x86_mnem),
                    "{name} expected `{x86_mnem}` in x86-64 asm:\n{x86_asm}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 2) Execute all R66 edge cases under Wasm64 (Node):
    let f_packsswb = CODE_BASE;
    let f_packuswb = CODE_BASE + 0x100;
    let f_packssdw = CODE_BASE + 0x200;
    let f_packusdw = CODE_BASE + 0x300;
    let f_pmaddwd = CODE_BASE + 0x400;
    let f_pavgb = CODE_BASE + 0x500;
    let f_pavgw = CODE_BASE + 0x600;
    let f_shifts = CODE_BASE + 0x700;
    let f_palignr = CODE_BASE + 0x800;
    let f_dpps_nan = CODE_BASE + 0x900;
    let f_dpps_neg0 = CODE_BASE + 0xA00;
    let f_dppd_edge = CODE_BASE + 0xB00;
    let f_mov_bcast = CODE_BASE + 0xC00;

    let program = Program::new()
        .asm(f_packsswb, |a| {
            a.packsswb(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_packuswb, |a| {
            a.packuswb(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_packssdw, |a| {
            a.packssdw(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_packusdw, |a| {
            a.packusdw(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_pmaddwd, |a| {
            a.pmaddwd(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_pavgb, |a| {
            a.pavgb(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_pavgw, |a| {
            a.pavgw(xmm0, xmm1)?;
            a.ret()
        })
        .asm(f_shifts, |a| {
            a.psllw(xmm0, 0)?;
            a.psrlw(xmm1, 4)?;
            a.psllw(xmm2, 16)?;
            a.psraw(xmm3, 25)?;
            a.ret()
        })
        .asm(f_palignr, |a| {
            a.palignr(xmm2, xmm1, 0)?; // xmm2 = xmm1
            a.palignr(xmm3, xmm1, 5)?; // (xmm3:xmm1) >> 40
            a.palignr(xmm4, xmm1, 16)?; // xmm4 = xmm4
            a.palignr(xmm5, xmm1, 20)?; // (xmm5:xmm1) >> 160
            a.palignr(xmm6, xmm1, 32)?; // 0
            a.ret()
        })
        .asm(f_dpps_nan, |a| {
            // Mask out lane 3 (which holds NaN/Inf): must NOT poison the dot product.
            a.dpps(xmm0, xmm1, 0x7F)?;
            a.ret()
        })
        .asm(f_dpps_neg0, |a| {
            // Lane 0..3 all -0.0, in_mask=0xF, out_mask=0x5 -> [-0.0, +0.0, -0.0, +0.0]
            a.dpps(xmm0, xmm1, 0xF5)?;
            // Lane 0 is -0.0, lanes 1..3 masked out (become +0.0) -> -0.0 + +0.0 = +0.0!
            a.dpps(xmm2, xmm3, 0x1F)?;
            a.ret()
        })
        .asm(f_dppd_edge, |a| {
            // Mask out lane 1 (NaN): result in lane 0 only, lane 1 = +0.0.
            a.dppd(xmm0, xmm1, 0x11)?;
            a.ret()
        })
        .asm(f_mov_bcast, |a| {
            a.movss(xmm0, xmm1)?;
            a.movsd_2(xmm2, xmm3)?;
            a.movq(xmm4, xmm5)?;
            a.vbroadcastss(xmm6, xmm7)?;
            a.ret()
        });

    let functions = program.discover();
    let built = build("r66_edge", functions.values());

    let i16s = |vals: [i16; 8]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let u16s = |vals: [u16; 8]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let i32s = |vals: [i32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let f32s = |vals: [f32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let f64s = |vals: [f64; 2]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };

    // 1. packsswb & packuswb
    let p_a16 = [i16::MIN, -129, -128, -1, 0, 127, 128, 255];
    let p_b16 = [256, i16::MAX, -1000, 1, -2, 127, 255, 256];
    let mut c_sswb = Case::new(f_packsswb);
    c_sswb.mem.push((XMM_STATE, [i16s(p_a16), i16s(p_b16)].concat()));
    c_sswb.read.push((XMM_STATE, 16));

    let mut c_uswb = Case::new(f_packuswb);
    c_uswb.mem.push((XMM_STATE, [i16s(p_a16), i16s(p_b16)].concat()));
    c_uswb.read.push((XMM_STATE, 16));

    // 2. packssdw & packusdw
    let p_a32 = [i32::MIN, -32769, -32768, -1];
    let p_b32 = [0, 32767, 65535, i32::MAX];
    let mut c_ssdw = Case::new(f_packssdw);
    c_ssdw.mem.push((XMM_STATE, [i32s(p_a32), i32s(p_b32)].concat()));
    c_ssdw.read.push((XMM_STATE, 16));

    let mut c_usdw = Case::new(f_packusdw);
    c_usdw.mem.push((XMM_STATE, [i32s(p_a32), i32s(p_b32)].concat()));
    c_usdw.read.push((XMM_STATE, 16));

    // 3. pmaddwd wrap (-32768*-32768 + -32768*-32768 = 0x80000000) + normal pairs
    let mut c_dot = Case::new(f_pmaddwd);
    c_dot.mem.push((
        XMM_STATE,
        [
            i16s([-32768, -32768, 32767, 32767, -100, 200, 0, -1]),
            i16s([-32768, -32768, 32767, -32768, 3, -4, 999, 1]),
        ]
        .concat(),
    ));
    c_dot.read.push((XMM_STATE, 16));

    // 4. pavgb edge cases
    let ab_a = [0u8, 0, 255, 255, 1, 2, 127, 128, 254, 255, 10, 11, 0, 128, 200, 201];
    let ab_b = [0u8, 1, 255, 0, 1, 3, 128, 127, 255, 254, 20, 20, 254, 128, 200, 202];
    let mut c_avgb = Case::new(f_pavgb);
    c_avgb.mem.push((XMM_STATE, [ab_a.to_vec(), ab_b.to_vec()].concat()));
    c_avgb.read.push((XMM_STATE, 16));

    // 5. pavgw edge cases
    let aw_a = [0u16, 0, 65535, 65535, 32767, 32768, 65534, 1];
    let aw_b = [0u16, 1, 65535, 0, 32768, 32768, 65535, 2];
    let mut c_avgw = Case::new(f_pavgw);
    c_avgw.mem.push((XMM_STATE, [u16s(aw_a), u16s(aw_b)].concat()));
    c_avgw.read.push((XMM_STATE, 16));

    // 6. immediate shifts (0, normal, overshift logical, overshift arith)
    let mut c_sh = Case::new(f_shifts);
    c_sh.mem.push((
        XMM_STATE,
        [
            u16s([0x1234, 0xABCD, 0, 0xFFFF, 1, 2, 3, 4]),
            u16s([0xF000, 0x00F0, 0xFFFF, 0x000F, 0x8000, 0x7FFF, 0x1234, 0]),
            u16s([0xFFFF; 8]),
            i16s([-1, -32768, 0, 1, 32767, -2, 42, -42]),
        ]
        .concat(),
    ));
    c_sh.read.push((XMM_STATE, 64));

    // 7. palignr (0, 5, 16, 20, 32)
    let lo_b: Vec<u8> = (0x10u8..0x20).collect();
    let hi_b: Vec<u8> = (0xA0u8..0xB0).collect();
    let mut c_al = Case::new(f_palignr);
    let mut al_xmms = vec![0u8; 7 * 16];
    al_xmms[16..32].copy_from_slice(&lo_b);
    for reg in 2..7 {
        al_xmms[reg * 16..(reg + 1) * 16].copy_from_slice(&hi_b);
    }
    c_al.mem.push((XMM_STATE, al_xmms));
    c_al.read.push((XMM_STATE + 32, 5 * 16));

    // 8. dpps with NaN/Inf in masked-out lane 3
    let mut c_dp_nan = Case::new(f_dpps_nan);
    c_dp_nan.mem.push((
        XMM_STATE,
        [
            f32s([1.0, 2.0, 3.0, f32::NAN]),
            f32s([4.0, 5.0, 6.0, f32::INFINITY]),
        ]
        .concat(),
    ));
    c_dp_nan.read.push((XMM_STATE, 16));

    // 9. dpps signed-zero edge cases
    let mut c_dp_z = Case::new(f_dpps_neg0);
    c_dp_z.mem.push((
        XMM_STATE,
        [
            f32s([-0.0, -0.0, -0.0, -0.0]),
            f32s([1.0, 1.0, 1.0, 1.0]),
            f32s([-0.0, f32::NAN, f32::INFINITY, -0.0]),
            f32s([1.0, 0.0, 0.0, 1.0]),
        ]
        .concat(),
    ));
    c_dp_z.read.push((XMM_STATE, 16));
    c_dp_z.read.push((XMM_STATE + 32, 16));

    // 10. dppd with NaN in masked-out lane 1
    let mut c_dppd = Case::new(f_dppd_edge);
    c_dppd.mem.push((
        XMM_STATE,
        [f64s([2.5, f64::NAN]), f64s([4.0, f64::INFINITY])].concat(),
    ));
    c_dppd.read.push((XMM_STATE, 16));

    // 11. movss, movsd, movq, vbroadcastss reg-reg
    let mut c_mb = Case::new(f_mov_bcast);
    c_mb.mem.push((
        XMM_STATE,
        [
            i32s([1, 2, 3, 4]),             // xmm0
            i32s([99, 88, 77, 66]),         // xmm1
            xmm_bytes(0x1111, 0x2222),      // xmm2
            xmm_bytes(0xAAAA, 0xBBBB),      // xmm3
            xmm_bytes(0xFFFF, 0xEEEE),      // xmm4
            xmm_bytes(0x1234, 0x5678),      // xmm5
            i32s([0; 4]),                   // xmm6
            i32s([0x5555_6666, 1, 2, 3]),   // xmm7
        ]
        .concat(),
    ));
    c_mb.read.push((XMM_STATE, 8 * 16));

    let out = run(
        &built,
        &[
            c_sswb, c_uswb, c_ssdw, c_usdw, c_dot, c_avgb, c_avgw, c_sh, c_al, c_dp_nan, c_dp_z,
            c_dppd, c_mb,
        ],
    );
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }

    // Check packsswb
    let want_sswb: Vec<u8> = p_a16
        .iter()
        .chain(&p_b16)
        .map(|&v| v.clamp(-128, 127) as i8 as u8)
        .collect();
    assert_eq!(out[0].mem[0], want_sswb, "packsswb");

    // Check packuswb
    let want_uswb: Vec<u8> = p_a16
        .iter()
        .chain(&p_b16)
        .map(|&v| v.clamp(0, 255) as u8)
        .collect();
    assert_eq!(out[1].mem[0], want_uswb, "packuswb");

    // Check packssdw
    let want_ssdw: Vec<u8> = p_a32
        .iter()
        .chain(&p_b32)
        .flat_map(|&v| (v.clamp(i16::MIN as i32, i16::MAX as i32) as i16).to_le_bytes())
        .collect();
    assert_eq!(out[2].mem[0], want_ssdw, "packssdw");

    // Check packusdw
    let want_usdw: Vec<u8> = p_a32
        .iter()
        .chain(&p_b32)
        .flat_map(|&v| (v.clamp(0, u16::MAX as i32) as u16).to_le_bytes())
        .collect();
    assert_eq!(out[3].mem[0], want_usdw, "packusdw");

    // Check pmaddwd
    assert_eq!(
        out[4].mem[0],
        i32s([
            i32::MIN,
            32767 * 32767 + 32767 * -32768,
            -100 * 3 + 200 * -4,
            -1
        ]),
        "pmaddwd"
    );

    // Check pavgb
    let want_avgb: Vec<u8> = ab_a
        .iter()
        .zip(&ab_b)
        .map(|(&a, &b)| ((u16::from(a) + u16::from(b) + 1) >> 1) as u8)
        .collect();
    assert_eq!(out[5].mem[0], want_avgb, "pavgb");

    // Check pavgw
    let want_avgw: Vec<u8> = aw_a
        .iter()
        .zip(&aw_b)
        .flat_map(|(&a, &b)| (((u32::from(a) + u32::from(b) + 1) >> 1) as u16).to_le_bytes())
        .collect();
    assert_eq!(out[6].mem[0], want_avgw, "pavgw");

    // Check immediate shifts
    assert_eq!(
        out[7].mem[0][0..16],
        u16s([0x1234, 0xABCD, 0, 0xFFFF, 1, 2, 3, 4]),
        "psllw 0"
    );
    assert_eq!(
        out[7].mem[0][16..32],
        u16s([0x0F00, 0x000F, 0x0FFF, 0x0000, 0x0800, 0x07FF, 0x0123, 0]),
        "psrlw 4"
    );
    assert_eq!(out[7].mem[0][32..48], vec![0u8; 16], "psllw 16");
    assert_eq!(
        out[7].mem[0][48..64],
        i16s([-1, -1, 0, 0, 0, -1, 0, -1]),
        "psraw 25"
    );

    // Check palignr
    let mut cat = lo_b.clone();
    cat.extend_from_slice(&hi_b);
    assert_eq!(out[8].mem[0][0..16], lo_b, "palignr 0");
    assert_eq!(out[8].mem[0][16..32], cat[5..21], "palignr 5");
    assert_eq!(out[8].mem[0][32..48], hi_b, "palignr 16");
    let mut want_20 = hi_b[4..16].to_vec();
    want_20.extend_from_slice(&[0u8; 4]);
    assert_eq!(out[8].mem[0][48..64], want_20, "palignr 20");
    assert_eq!(out[8].mem[0][64..80], vec![0u8; 16], "palignr 32");

    // Check dpps NaN-masked
    assert_eq!(out[9].mem[0], f32s([32.0; 4]), "dpps 0x7F masks NaN/Inf");

    // Check dpps -0.0 sign preservation
    assert_eq!(
        out[10].mem[0],
        f32s([-0.0, 0.0, -0.0, 0.0]),
        "dpps 0xF5 preserves -0.0 in masked-in lanes and +0.0 in masked-out lanes"
    );
    assert_eq!(
        out[10].mem[1],
        f32s([0.0; 4]),
        "dpps 0x1F sums -0.0 with +0.0 -> +0.0"
    );

    // Check dppd NaN-masked
    assert_eq!(out[11].mem[0], f64s([10.0, 0.0]), "dppd 0x11 masks NaN/Inf");

    // Check movss/movsd/movq/vbroadcastss
    assert_eq!(out[12].mem[0][0..16], i32s([99, 2, 3, 4]), "movss reg-reg");
    assert_eq!(
        out[12].mem[0][32..48],
        xmm_bytes(0xAAAA, 0x2222),
        "movsd reg-reg"
    );
    assert_eq!(
        out[12].mem[0][64..80],
        xmm_bytes(0x1234, 0),
        "movq xmm, xmm"
    );
    assert_eq!(
        out[12].mem[0][96..112],
        i32s([0x5555_6666; 4]),
        "vbroadcastss reg-reg"
    );
}

/// Bit-exact IEEE-754 binary16 -> binary32 reference (with hardware SNaN quieting).
fn ref_f16_to_f32_bits(h: u16) -> u32 {
    let sign = (u32::from(h & 0x8000)) << 16;
    let exp = (h >> 10) & 0x1F;
    let man = u32::from(h & 0x03FF);
    if exp == 0 {
        if man == 0 {
            sign
        } else {
            ((man as f32) * 2f32.powi(-24)).to_bits() | sign
        }
    } else if exp == 31 {
        if man == 0 {
            sign | 0x7F80_0000
        } else {
            sign | 0x7FC0_0000 | (man << 13)
        }
    } else {
        sign | ((u32::from(exp) + 112) << 23) | (man << 13)
    }
}

/// Bit-exact IEEE-754 binary32 -> binary16 reference matching x86 F16C `vcvtps2ph` for all 4
/// rounding modes (0 = nearest-even, 1 = -inf, 2 = +inf, 3 = toward zero).
fn ref_f32_to_f16_bits(bits: u32, mode: u8) -> u16 {
    let s = (bits >> 31) != 0;
    let sign16 = if s { 0x8000u16 } else { 0 };
    let e = (bits >> 23) & 0xFF;
    let m = bits & 0x7F_FFFF;
    if e == 255 {
        return if m == 0 {
            sign16 | 0x7C00
        } else {
            sign16 | 0x7E00 | ((m >> 13) as u16)
        };
    }
    let full = if e == 0 { m } else { m | 0x80_0000 };
    let e_eff = if e == 0 { 1i32 } else { e as i32 };
    let big_e = e_eff - 112;
    if big_e > 30 {
        let ov = match mode {
            0 => 0x7C00,
            1 => if s { 0x7C00 } else { 0x7BFF },
            2 => if s { 0x7BFF } else { 0x7C00 },
            _ => 0x7BFF,
        };
        return sign16 | ov;
    }
    let (shift, base) = if big_e > 0 {
        (13u32, ((big_e - 1) as u32) << 10)
    } else {
        (((14 - big_e) as u32).min(31), 0u32)
    };
    let q = full >> shift;
    let rem = full & ((1u32 << shift) - 1);
    let truncated = base + q;
    let inc = match mode {
        0 => {
            let half = 1u32 << (shift - 1);
            rem > half || (rem == half && (q & 1) != 0)
        }
        1 => rem != 0 && s,
        2 => rem != 0 && !s,
        _ => false,
    };
    sign16 | ((truncated + u32::from(inc)) as u16)
}

#[test]
fn r70_3d_fp_fma_f16c_wasm64_codegen_and_execution() {
    if !exec_or_skip("r70_3d_fp_fma_f16c_wasm64_codegen_and_execution") {
        return;
    }
    use crate::lift::STATE_YMM_HI_OFFSET;
    use crate::lift::compile::{LlvmTool, find_llvm_tool};
    use crate::lift::target::LiftTarget;

    const YMMH_STATE: u64 = STATE_ADDR + STATE_YMM_HI_OFFSET;

    // 1) Verify Wasm64 SIMD128 assembly quality and native x86-64-v3 compilation.
    if let Some(LlvmTool::Llc { llc, opt }) = find_llvm_tool() {
        let dir = std::env::temp_dir().join(format!("playgta6_r70_asm_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let probes: Vec<(&str, crate::cfg::Function)> = vec![
            (
                "vfmadd213ps",
                asm_function(CODE_BASE, |a| {
                    a.vfmadd213ps(xmm0, xmm1, xmm2)?;
                    a.ret()
                }),
            ),
            (
                "vfmaddsub213ps",
                asm_function(CODE_BASE + 0x100, |a| {
                    a.vfmaddsub213ps(xmm0, xmm1, xmm2)?;
                    a.ret()
                }),
            ),
            (
                "addsubps",
                asm_function(CODE_BASE + 0x200, |a| {
                    a.addsubps(xmm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "vcvtpd2dq_ymm",
                asm_function(CODE_BASE + 0x300, |a| {
                    a.vcvtpd2dq(xmm0, ymm1)?;
                    a.ret()
                }),
            ),
            (
                "vcvtpd2ps_ymm",
                asm_function(CODE_BASE + 0x400, |a| {
                    a.vcvtpd2ps(xmm0, ymm1)?;
                    a.ret()
                }),
            ),
            (
                "vcvtps2pd_ymm",
                asm_function(CODE_BASE + 0x500, |a| {
                    a.vcvtps2pd(ymm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "vcvtph2ps_ymm",
                asm_function(CODE_BASE + 0x600, |a| {
                    a.vcvtph2ps(ymm0, xmm1)?;
                    a.ret()
                }),
            ),
            (
                "vcvtps2ph_ymm",
                asm_function(CODE_BASE + 0x700, |a| {
                    a.vcvtps2ph(xmm0, ymm1, 0)?;
                    a.ret()
                }),
            ),
            (
                "movddup_m64",
                asm_function(CODE_BASE + 0x800, |a| {
                    a.movddup(xmm0, qword_ptr(rcx))?;
                    a.ret()
                }),
            ),
            (
                "dpps_71",
                asm_function(CODE_BASE + 0x900, |a| {
                    a.dpps(xmm0, xmm1, 0x71)?;
                    a.ret()
                }),
            ),
        ];
        let native_target = LiftTarget::parse("native:x86_64-pc-windows-msvc", Some("x86-64-v3"))
            .expect("native target");
        for (name, func) in &probes {
            let m = lift_module([func]).expect("lift");
            let ll = dir.join(format!("{name}.ll"));
            let bc = dir.join(format!("{name}.bc"));
            let asm_path = dir.join(format!("{name}.s"));
            std::fs::write(&ll, &m.ir).expect("write ll");
            let input = if let Some(opt_bin) = &opt {
                let st = std::process::Command::new(opt_bin)
                    .args(["-O2"])
                    .arg(&ll)
                    .arg("-o")
                    .arg(&bc)
                    .status()
                    .expect("run opt");
                assert!(st.success(), "opt failed on {name}");
                &bc
            } else {
                &ll
            };
            let st = std::process::Command::new(&llc)
                .args(["-mtriple=wasm64-unknown-unknown", "-O2", "-filetype=asm"])
                .arg(input)
                .arg("-o")
                .arg(&asm_path)
                .status()
                .expect("run llc wasm64");
            assert!(st.success(), "llc wasm64 failed on {name}");
            let asm = std::fs::read_to_string(&asm_path).expect("read wasm asm");
            match *name {
                "vfmadd213ps" => {
                    assert!(asm.contains("f64x2.promote_low_f32x4"), "{asm}");
                    assert_eq!(
                        asm.matches("f32x4.demote_f64x2_zero").count(),
                        2,
                        "fma32 should emit 2 native demote_f64x2_zero:\n{asm}"
                    );
                    assert!(!asm.contains("extract_lane"), "fma32 scalarized:\n{asm}");
                    assert!(!asm.contains("\tcall"), "fma32 must not emit libcalls:\n{asm}");
                }
                "vfmaddsub213ps" => {
                    assert!(asm.contains("v128.xor"), "{asm}");
                    assert!(asm.contains("f64x2.promote_low_f32x4"), "{asm}");
                }
                "addsubps" => {
                    assert!(asm.contains("v128.xor"), "{asm}");
                    assert!(asm.contains("f32x4.add"), "{asm}");
                    assert!(!asm.contains("f32x4.sub"), "addsubps should not emit f32x4.sub:\n{asm}");
                    assert!(!asm.contains("i8x16.shuffle"), "addsubps should not shuffle:\n{asm}");
                }
                "vcvtpd2dq_ymm" => {
                    assert_eq!(
                        asm.matches("i32x4.trunc_sat_f64x2_s_zero").count(),
                        2,
                        "256-bit vcvtpd2dq should emit 2 native trunc_sat_f64x2_s_zero:\n{asm}"
                    );
                }
                "vcvtpd2ps_ymm" => {
                    assert_eq!(
                        asm.matches("f32x4.demote_f64x2_zero").count(),
                        2,
                        "256-bit vcvtpd2ps should emit 2 native demote_f64x2_zero:\n{asm}"
                    );
                }
                "vcvtps2pd_ymm" => {
                    assert_eq!(
                        asm.matches("f64x2.promote_low_f32x4").count(),
                        2,
                        "256-bit vcvtps2pd should emit 2 native promote_low_f32x4:\n{asm}"
                    );
                }
                "vcvtph2ps_ymm" => {
                    assert!(asm.contains("i32x4.extend_low_i16x8_u"), "{asm}");
                    assert!(
                        asm.contains("i32x4.extend_high_i16x8_u")
                            || asm.contains("i32x4.extend_high_i16x8_s"),
                        "{asm}"
                    );
                    assert!(!asm.contains("extract_lane"), "vcvtph2ps scalarized:\n{asm}");
                    assert!(!asm.contains("__stack_pointer"), "vcvtph2ps spilled:\n{asm}");
                }
                "vcvtps2ph_ymm" => {
                    assert!(asm.contains("f32x4.add"), "{asm}");
                    assert!(!asm.contains("__stack_pointer"), "vcvtps2ph spilled:\n{asm}");
                }
                "movddup_m64" => {
                    assert!(asm.contains("v128.load64_splat"), "{asm}");
                }
                "dpps_71" => {
                    assert!(!asm.contains("extract_lane"), "{asm}");
                    assert!(asm.contains("f32x4.mul"), "{asm}");
                    assert!(asm.contains("f32x4.add"), "{asm}");
                }
                _ => {}
            }

            // Ensure LiftTarget::Native compiles cleanly on x86-64-v3.
            let native_ir = native_target.retarget(m.ir);
            let nll = dir.join(format!("{name}_x86.ll"));
            let nasm = dir.join(format!("{name}_x86.s"));
            std::fs::write(&nll, &native_ir).expect("write native ll");
            let st = std::process::Command::new(&llc)
                .args([
                    "-mtriple=x86_64-pc-windows-msvc",
                    "-mcpu=x86-64-v3",
                    "-O2",
                    "-filetype=asm",
                ])
                .arg(&nll)
                .arg("-o")
                .arg(&nasm)
                .status()
                .expect("run llc native");
            assert!(st.success(), "llc native x86-64 failed on {name}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 2) Execute semantic edge cases under Wasm64 (Node):
    let f_fma_ps = CODE_BASE;
    let f_fma_addsub = CODE_BASE + 0x100;
    let f_fma_scalar = CODE_BASE + 0x200;
    let f_addsub = CODE_BASE + 0x300;
    let f_dp_masks = CODE_BASE + 0x400;
    let f_rcp = CODE_BASE + 0x500;
    let f_cvt256 = CODE_BASE + 0x600;
    let f_shuf_blend = CODE_BASE + 0x700;
    let f_vinsert_perm = CODE_BASE + 0x800;
    let f_f16c_ph2ps = CODE_BASE + 0x900;
    let f_f16c_ps2ph = CODE_BASE + 0xA00;
    let f_f16c_modes = CODE_BASE + 0xB00;

    let program = Program::new()
        .asm(f_fma_ps, |a| {
            a.vfmadd213ps(ymm0, ymm1, ymm2)?;
            a.vfnmsub231ps(xmm3, xmm4, xmm5)?;
            a.ret()
        })
        .asm(f_fma_addsub, |a| {
            a.vfmaddsub213ps(xmm0, xmm1, xmm2)?;
            a.vfmsubadd213ps(xmm3, xmm4, xmm5)?;
            a.vfmaddsub213pd(xmm6, xmm7, xmm8)?;
            a.vfmsubadd213pd(xmm9, xmm10, xmm11)?;
            a.ret()
        })
        .asm(f_fma_scalar, |a| {
            a.vfmadd213ss(xmm0, xmm1, xmm2)?;
            a.vfmadd213ss(xmm3, xmm3, xmm4)?; // op0 == op1 reuse
            a.vfmadd213sd(xmm5, xmm6, xmm7)?;
            a.ret()
        })
        .asm(f_addsub, |a| {
            a.addsubps(xmm0, xmm1)?;
            a.addsubpd(xmm2, xmm3)?;
            a.vaddsubps(xmm4, xmm5, xmm6)?;
            a.ret()
        })
        .asm(f_dp_masks, |a| {
            a.dpps(xmm0, xmm1, 0x11)?; // 1-lane, -0.0 -> +0.0
            a.dpps(xmm2, xmm3, 0x4F)?; // 1-lane lane 2 broadcast
            a.dpps(xmm4, xmm5, 0x31)?; // 2-lane, -0.0 + -0.0 -> +0.0
            a.dpps(xmm6, xmm7, 0x5A)?; // 2-lane (0,2) -> out (1,3)
            a.dpps(xmm8, xmm9, 0x71)?; // 3-lane (0,1,2) -> lane 0
            a.dpps(xmm10, xmm11, 0xE2)?; // 3-lane (1,2,3) -> lane 1
            a.dppd(xmm12, xmm13, 0x23)?; // 1-lane dppd broadcast
            a.dpps(xmm14, xmm14, 0x7F)?; // same-reg 3D squared length
            a.ret()
        })
        .asm(f_rcp, |a| {
            a.rcpps(xmm0, xmm1)?;
            a.rsqrtps(xmm2, xmm3)?;
            a.ret()
        })
        .asm(f_cvt256, |a| {
            a.vcvtpd2dq(xmm0, ymm1)?;
            a.vcvttpd2dq(xmm2, ymm1)?;
            a.vcvtpd2ps(xmm3, ymm1)?;
            a.vcvtps2pd(ymm4, xmm5)?;
            a.vcvtdq2pd(ymm6, xmm7)?;
            a.ret()
        })
        .asm(f_shuf_blend, |a| {
            a.pshufd(xmm0, xmm0, 0xE4)?; // identity
            a.pshuflw(xmm1, xmm1, 0xE4)?; // identity
            a.vpermilps(xmm2, xmm2, 0xE4)?; // identity
            a.blendps(xmm3, xmm4, 0x0)?; // only a
            a.blendps(xmm5, xmm4, 0xF)?; // only b
            a.vblendvps(xmm6, xmm7, xmm7, xmm0)?; // a == b
            a.insertps(xmm8, xmm9, 0x62)?; // src=1, dst=2, zmask=0b0010
            a.insertps(xmm10, xmm11, 0x0F)?; // zmask=0xF -> 0
            a.haddps(xmm12, xmm12)?; // same-reg haddps
            a.movddup(xmm13, qword_ptr(rcx))?;
            a.ret()
        })
        .asm(f_vinsert_perm, |a| {
            a.vinsertf128(ymm0, ymm1, xmm2, 0)?;
            a.vinserti128(ymm3, ymm1, xmm2, 1)?;
            a.vpermq(ymm4, ymm1, 0x1B)?; // [3, 2, 1, 0]
            a.vpermq(ymm5, ymm1, 0x00)?; // broadcast lane 0
            a.ret()
        })
        .asm(f_f16c_ph2ps, |a| {
            a.vcvtph2ps(ymm0, xmm1)?;
            a.vcvtph2ps(xmm2, xmm3)?;
            a.ret()
        })
        .asm(f_f16c_ps2ph, |a| {
            a.vcvtps2ph(xmm0, ymm1, 0)?; // Nearest 256-bit
            a.vcvtps2ph(xmm2, xmm3, 0)?; // Nearest 128-bit
            a.ret()
        })
        .asm(f_f16c_modes, |a| {
            a.vcvtps2ph(xmm0, ymm1, 0)?; // Nearest
            a.vcvtps2ph(xmm2, ymm1, 1)?; // Down (-inf)
            a.vcvtps2ph(xmm3, ymm1, 2)?; // Up (+inf)
            a.vcvtps2ph(xmm4, ymm1, 3)?; // Zero (trunc)
            a.ret()
        });

    let functions = program.discover();
    let built = build("r70_edge", functions.values());

    let f32s = |vals: [f32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let u32s = |vals: [u32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let i32s = |vals: [i32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let f64s = |vals: [f64; 2]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let u16s = |vals: [u16; 8]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };

    let read_f32s = |b: &[u8]| -> [u32; 4] {
        std::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
    };
    let read_f64s = |b: &[u8]| -> [u64; 2] {
        std::array::from_fn(|i| u64::from_le_bytes(b[8 * i..8 * i + 8].try_into().unwrap()))
    };
    let read_u16s = |b: &[u8]| -> [u16; 8] {
        std::array::from_fn(|i| u16::from_le_bytes(b[2 * i..2 * i + 2].try_into().unwrap()))
    };

    // Case 0: FMA3 256-bit single-rounding edge cases (catastrophic cancellation & ties).
    // (1 + 2^-23) * (1 - 2^-23) + (-1.0) = -2^-46 (unfused rounds product to 1.0, yielding 0.0!)
    let one_p = f32::from_bits(0x3F80_0001); // 1 + 2^-23
    let one_m = f32::from_bits(0x3F7F_FFFE); // 1 - 2^-23
    // Double-rounding trap: a*b + c where binary64 sum lands on an exact binary32 midpoint with
    // non-zero TwoSum error: a = 1 + 2^-23, b = 1 + 2^-22, c = 2^25 (forces binary64 rounding).
    let trap_a = f32::from_bits(0x3F80_0001);
    let trap_b = f32::from_bits(0x3F80_0002);
    let trap_c = f32::from_bits(0x4C00_0000); // 2^25
    let mut c_fma = Case::new(f_fma_ps);
    let mut xmms = vec![0u8; 6 * 16];
    let mut ymmh = vec![0u8; 6 * 16];
    // ymm0 (op0 = y in 213), ymm1 (op1 = x), ymm2 (op2 = z)
    xmms[0..16].copy_from_slice(&f32s([one_m, trap_b, -0.0, 2.0]));
    xmms[16..32].copy_from_slice(&f32s([one_p, trap_a, 1.0, 3.0]));
    xmms[32..48].copy_from_slice(&f32s([-1.0, trap_c, -0.0, 4.0]));
    ymmh[0..16].copy_from_slice(&f32s([-one_m, 1.5, f32::INFINITY, 1.0]));
    ymmh[16..32].copy_from_slice(&f32s([one_p, 2.5, 2.0, f32::NAN]));
    ymmh[32..48].copy_from_slice(&f32s([1.0, -3.75, -1.0, 5.0]));
    // xmm3..5 for vfnmsub231ps: -(x*y) - z
    xmms[48..64].copy_from_slice(&f32s([1.0, -1.0, 0.5, 10.0]));
    xmms[64..80].copy_from_slice(&f32s([2.0, 3.0, 4.0, -2.0]));
    xmms[80..96].copy_from_slice(&f32s([5.0, 6.0, 7.0, 8.0]));
    c_fma.mem.push((XMM_STATE, xmms));
    c_fma.mem.push((YMMH_STATE, ymmh));
    c_fma.read.push((XMM_STATE, 16));
    c_fma.read.push((YMMH_STATE, 16));
    c_fma.read.push((XMM_STATE + 48, 16));

    // Case 1: vfmaddsub / vfmsubadd (ps & pd)
    let mut c_fmas = Case::new(f_fma_addsub);
    let mut s1 = vec![0u8; 12 * 16];
    s1[0..16].copy_from_slice(&f32s([2.0, 3.0, 4.0, 5.0]));
    s1[16..32].copy_from_slice(&f32s([10.0, 10.0, 10.0, 10.0]));
    s1[32..48].copy_from_slice(&f32s([1.0, 2.0, 3.0, 4.0]));
    s1[48..64].copy_from_slice(&f32s([2.0, 3.0, 4.0, 5.0]));
    s1[64..80].copy_from_slice(&f32s([10.0, 10.0, 10.0, 10.0]));
    s1[80..96].copy_from_slice(&f32s([1.0, 2.0, 3.0, 4.0]));
    s1[96..112].copy_from_slice(&f64s([2.0, 3.0]));
    s1[112..128].copy_from_slice(&f64s([10.0, 10.0]));
    s1[128..144].copy_from_slice(&f64s([1.5, 2.5]));
    s1[144..160].copy_from_slice(&f64s([2.0, 3.0]));
    s1[160..176].copy_from_slice(&f64s([10.0, 10.0]));
    s1[176..192].copy_from_slice(&f64s([1.5, 2.5]));
    c_fmas.mem.push((XMM_STATE, s1));
    c_fmas.read.push((XMM_STATE, 16));
    c_fmas.read.push((XMM_STATE + 48, 16));
    c_fmas.read.push((XMM_STATE + 96, 16));
    c_fmas.read.push((XMM_STATE + 144, 16));

    // Case 2: scalar FMA3 (`ss`/`sd`) preserving upper lanes
    let mut c_fsc = Case::new(f_fma_scalar);
    let mut s2 = vec![0u8; 8 * 16];
    s2[0..16].copy_from_slice(&f32s([one_m, 11.0, 22.0, 33.0]));
    s2[16..32].copy_from_slice(&f32s([one_p, 99.0, 99.0, 99.0]));
    s2[32..48].copy_from_slice(&f32s([-1.0, 88.0, 88.0, 88.0]));
    s2[48..64].copy_from_slice(&f32s([3.0, 44.0, 55.0, 66.0]));
    s2[64..80].copy_from_slice(&f32s([7.0, 0.0, 0.0, 0.0]));
    s2[80..96].copy_from_slice(&f64s([2.5, 777.0]));
    s2[96..112].copy_from_slice(&f64s([4.0, 888.0]));
    s2[112..128].copy_from_slice(&f64s([1.25, 999.0]));
    c_fsc.mem.push((XMM_STATE, s2));
    c_fsc.read.push((XMM_STATE, 16));
    c_fsc.read.push((XMM_STATE + 48, 16));
    c_fsc.read.push((XMM_STATE + 80, 16));

    // Case 3: addsubps & addsubpd signed zero / inf / normal parity
    let mut c_as = Case::new(f_addsub);
    let mut s3 = vec![0u8; 7 * 16];
    s3[0..16].copy_from_slice(&f32s([-0.0, -0.0, 0.0, 10.0]));
    s3[16..32].copy_from_slice(&f32s([0.0, -0.0, 0.0, 3.0]));
    s3[32..48].copy_from_slice(&f64s([-0.0, f64::INFINITY]));
    s3[48..64].copy_from_slice(&f64s([-0.0, 5.0]));
    s3[80..96].copy_from_slice(&f32s([1.0, 2.0, 3.0, 4.0]));
    s3[96..112].copy_from_slice(&f32s([5.0, 6.0, 7.0, 8.0]));
    c_as.mem.push((XMM_STATE, s3));
    c_as.read.push((XMM_STATE, 16));
    c_as.read.push((XMM_STATE + 32, 16));
    c_as.read.push((XMM_STATE + 64, 16));

    // Case 4: dpps/dppd 1/2/3/4-lane fast masks + -0.0 + NaN isolation
    let mut c_dp = Case::new(f_dp_masks);
    let mut s4 = vec![0u8; 15 * 16];
    s4[0..16].copy_from_slice(&f32s([-0.0, f32::NAN, f32::INFINITY, 5.0]));
    s4[16..32].copy_from_slice(&f32s([1.0, 0.0, 0.0, 1.0]));
    s4[32..48].copy_from_slice(&f32s([f32::NAN, f32::INFINITY, 3.5, f32::NAN]));
    s4[48..64].copy_from_slice(&f32s([0.0, 0.0, 2.0, 0.0]));
    s4[64..80].copy_from_slice(&f32s([-0.0, -0.0, f32::NAN, f32::INFINITY]));
    s4[80..96].copy_from_slice(&f32s([1.0, 1.0, 0.0, 0.0]));
    s4[96..112].copy_from_slice(&f32s([2.0, f32::NAN, 5.0, f32::INFINITY]));
    s4[112..128].copy_from_slice(&f32s([3.0, 0.0, 4.0, 0.0]));
    s4[128..144].copy_from_slice(&f32s([1.0, 2.0, 3.0, f32::NAN]));
    s4[144..160].copy_from_slice(&f32s([4.0, 5.0, 6.0, f32::INFINITY]));
    s4[160..176].copy_from_slice(&f32s([f32::NAN, 2.0, 3.0, 4.0]));
    s4[176..192].copy_from_slice(&f32s([f32::INFINITY, 5.0, 6.0, 7.0]));
    s4[192..208].copy_from_slice(&f64s([f64::NAN, -0.0]));
    s4[208..224].copy_from_slice(&f64s([0.0, 1.0]));
    s4[224..240].copy_from_slice(&f32s([3.0, 4.0, 12.0, f32::NAN]));
    c_dp.mem.push((XMM_STATE, s4));
    for reg in [0, 2, 4, 6, 8, 10, 12, 14] {
        c_dp.read.push((XMM_STATE + (reg * 16) as u64, 16));
    }

    // Case 5: rcpps / rsqrtps special cases
    let mut c_rcp = Case::new(f_rcp);
    let mut s5 = vec![0u8; 4 * 16];
    let sub_pos = f32::from_bits(0x0000_0001);
    let sub_neg = f32::from_bits(0x8000_0001);
    let big_pos = f32::from_bits(0x7E80_0000); // 2^126 -> flushes to +0.0
    s5[16..32].copy_from_slice(&f32s([4.0, sub_neg, big_pos, f32::INFINITY]));
    s5[48..64].copy_from_slice(&f32s([4.0, sub_pos, sub_neg, f32::INFINITY]));
    c_rcp.mem.push((XMM_STATE, s5));
    c_rcp.read.push((XMM_STATE, 16));
    c_rcp.read.push((XMM_STATE + 32, 16));

    // Case 6: 256-bit conversions (`vcvtpd2dq`, `vcvttpd2dq`, `vcvtpd2ps`, `vcvtps2pd`, `vcvtdq2pd`)
    let mut c_cvt = Case::new(f_cvt256);
    let mut s6 = vec![0u8; 8 * 16];
    let mut h6 = vec![0u8; 8 * 16];
    s6[16..32].copy_from_slice(&f64s([2.5, -2147483648.9]));
    h6[16..32].copy_from_slice(&f64s([2147483648.0, f64::NAN]));
    s6[80..96].copy_from_slice(&f32s([1.25, -2.5, 3.75, -4.5]));
    s6[112..128].copy_from_slice(&i32s([i32::MIN, -1, 0, i32::MAX]));
    c_cvt.mem.push((XMM_STATE, s6));
    c_cvt.mem.push((YMMH_STATE, h6));
    c_cvt.read.push((XMM_STATE, 16)); // xmm0: vcvtpd2dq
    c_cvt.read.push((YMMH_STATE, 16)); // ymmh0: must be 0
    c_cvt.read.push((XMM_STATE + 32, 16)); // xmm2: vcvttpd2dq
    c_cvt.read.push((XMM_STATE + 48, 16)); // xmm3: vcvtpd2ps
    c_cvt.read.push((XMM_STATE + 64, 16)); // ymm4 lo: vcvtps2pd
    c_cvt.read.push((YMMH_STATE + 64, 16)); // ymm4 hi: vcvtps2pd
    c_cvt.read.push((XMM_STATE + 96, 16)); // ymm6 lo: vcvtdq2pd
    c_cvt.read.push((YMMH_STATE + 96, 16)); // ymm6 hi: vcvtdq2pd

    // Case 7: shuffles, blends, insertps, haddps, movddup m64
    let mem_buf = 0x18_0000u64;
    let mut c_sb = Case::new(f_shuf_blend).reg(RCX, mem_buf);
    c_sb.mem.push((mem_buf, 0xCAFE_BABE_1234_5678u64.to_le_bytes().to_vec()));
    let mut s7 = vec![0u8; 14 * 16];
    s7[0..16].copy_from_slice(&i32s([1, 2, 3, 4]));
    s7[16..32].copy_from_slice(&i32s([5, 6, 7, 8]));
    s7[32..48].copy_from_slice(&i32s([9, 10, 11, 12]));
    s7[48..64].copy_from_slice(&i32s([13, 14, 15, 16]));
    s7[64..80].copy_from_slice(&i32s([17, 18, 19, 20]));
    s7[80..96].copy_from_slice(&i32s([21, 22, 23, 24]));
    s7[112..128].copy_from_slice(&i32s([29, 30, 31, 32]));
    s7[128..144].copy_from_slice(&i32s([100, 200, 300, 400]));
    s7[144..160].copy_from_slice(&i32s([10, 20, 30, 40]));
    s7[160..176].copy_from_slice(&i32s([9, 9, 9, 9]));
    s7[192..208].copy_from_slice(&f32s([1.0, 2.0, 3.0, 4.0]));
    c_sb.mem.push((XMM_STATE, s7));
    for reg in [0, 1, 2, 3, 5, 6, 8, 10, 12, 13] {
        c_sb.read.push((XMM_STATE + (reg * 16) as u64, 16));
    }

    // Case 8: vinsertf128/i128 + vpermq
    let mut c_vp = Case::new(f_vinsert_perm);
    let mut s8 = vec![0u8; 6 * 16];
    let mut h8 = vec![0u8; 6 * 16];
    s8[16..32].copy_from_slice(&xmm_bytes(0x11, 0x22));
    h8[16..32].copy_from_slice(&xmm_bytes(0x33, 0x44));
    s8[32..48].copy_from_slice(&xmm_bytes(0xAA, 0xBB));
    c_vp.mem.push((XMM_STATE, s8));
    c_vp.mem.push((YMMH_STATE, h8));
    for reg in [0, 3, 4, 5] {
        c_vp.read.push((XMM_STATE + (reg * 16) as u64, 16));
        c_vp.read.push((YMMH_STATE + (reg * 16) as u64, 16));
    }

    // Case 9: F16C vcvtph2ps (256-bit & 128-bit) across normal, denormal, +-0, +-inf, SNaN, QNaN
    let h_vals_8: [u16; 8] = [
        0x0000, // +0.0
        0x8000, // -0.0
        0x0001, // smallest +subnormal
        0x83FF, // largest -subnormal
        0x3C00, // +1.0
        0xFC00, // -inf
        0x7C01, // +SNaN (must be quieted to 0x7FC0_2000)
        0xFEAB, // -QNaN with payload
    ];
    let h_vals_4: [u16; 8] = [0x0400, 0x7BFF, 0x7C00, 0xFC01, 0, 0, 0, 0];
    let mut c_ph = Case::new(f_f16c_ph2ps);
    let mut s9 = vec![0u8; 4 * 16];
    s9[16..32].copy_from_slice(&u16s(h_vals_8));
    s9[48..64].copy_from_slice(&u16s(h_vals_4));
    c_ph.mem.push((XMM_STATE, s9));
    c_ph.mem.push((YMMH_STATE, vec![0xCCu8; 4 * 16]));
    c_ph.read.push((XMM_STATE, 16));
    c_ph.read.push((YMMH_STATE, 16));
    c_ph.read.push((XMM_STATE + 32, 16));
    c_ph.read.push((YMMH_STATE + 32, 16));

    // Case 10: F16C vcvtps2ph (256-bit & 128-bit, Round::Nearest) across every boundary
    let f_near_lo: [u32; 4] = [
        0x0000_0000, // +0.0
        0x8000_0000, // -0.0
        0x3300_0000, // 2^-25 (exact tie -> 0)
        0x3300_0001, // just above 2^-25 -> 0x0001
    ];
    let f_near_hi: [u32; 4] = [
        0x387F_F000, // 1023.5 * 2^-24 (tie -> 0x0400 normal!)
        0x477F_EFFF, // just below 65520 -> 0x7BFF
        0x477F_F000, // 65520 (tie -> 0x7C00 inf!)
        0xFF80_2001, // -SNaN -> -QNaN 0xFE01
    ];
    let f_near_128: [u32; 4] = [
        0x33C0_0000, // 1.5 * 2^-24 (tie -> 0x0002 even)
        0x8000_0001, // -2^-149 (f32 subnormal -> -0.0)
        0x7F80_0000, // +inf
        0x7F80_0001, // +SNaN -> 0x7E00
    ];
    let mut c_ps = Case::new(f_f16c_ps2ph);
    let mut s10 = vec![0u8; 4 * 16];
    let mut h10 = vec![0u8; 4 * 16];
    s10[16..32].copy_from_slice(&u32s(f_near_lo));
    h10[16..32].copy_from_slice(&u32s(f_near_hi));
    s10[48..64].copy_from_slice(&u32s(f_near_128));
    c_ps.mem.push((XMM_STATE, s10));
    c_ps.mem.push((YMMH_STATE, h10));
    c_ps.read.push((XMM_STATE, 16));
    c_ps.read.push((XMM_STATE + 32, 16));

    // Case 11: F16C vcvtps2ph all 4 rounding modes (Nearest, Down, Up, Zero)
    let f_mode_lo: [u32; 4] = [
        0x0000_0001, // +2^-149 (f32 subnormal: Up -> 0x0001, others -> 0x0000)
        0x8000_0001, // -2^-149 (f32 subnormal: Down -> 0x8001, others -> 0x8000)
        0x477F_E001, // +65504 + ulp (Up -> +inf, Down/Zero/Nearest -> 0x7BFF)
        0xC77F_E001, // -65504 - ulp (Down -> -inf, Up/Zero/Nearest -> 0xFBFF)
    ];
    let f_mode_hi: [u32; 4] = [
        0x4780_0000, // +65536 (Nearest/Up -> +inf, Down/Zero -> 0x7BFF)
        0x7F80_0000, // +inf (ALL modes -> +inf 0x7C00)
        0xFF80_0000, // -inf (ALL modes -> -inf 0xFC00)
        0x387F_FFFF, // +(2^-14 - ulp) (Nearest/Up -> 0x0400, Down/Zero -> 0x03FF)
    ];
    let mut c_pm = Case::new(f_f16c_modes);
    let mut s11 = vec![0u8; 5 * 16];
    let mut h11 = vec![0u8; 5 * 16];
    s11[16..32].copy_from_slice(&u32s(f_mode_lo));
    h11[16..32].copy_from_slice(&u32s(f_mode_hi));
    c_pm.mem.push((XMM_STATE, s11));
    c_pm.mem.push((YMMH_STATE, h11));
    for reg in [0, 2, 3, 4] {
        c_pm.read.push((XMM_STATE + (reg * 16) as u64, 16));
    }

    let out = run(
        &built,
        &[
            c_fma, c_fmas, c_fsc, c_as, c_dp, c_rcp, c_cvt, c_sb, c_vp, c_ph, c_ps, c_pm,
        ],
    );
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }

    // Verify Case 0: FMA3 256-bit
    let fma_lo = read_f32s(&out[0].mem[0]);
    let fma_hi = read_f32s(&out[0].mem[1]);
    assert_eq!(fma_lo[0], one_p.mul_add(one_m, -1.0).to_bits(), "fma3 cancellation");
    assert_eq!(fma_lo[1], trap_a.mul_add(trap_b, trap_c).to_bits(), "fma3 double-rounding trap");
    assert_eq!(fma_lo[2], (-0.0f32).to_bits(), "fma3 -0.0 + -0.0");
    assert_eq!(fma_lo[3], 10.0f32.to_bits());
    assert_eq!(fma_hi[0], one_p.mul_add(-one_m, 1.0).to_bits(), "fma3 hi cancellation");
    assert_eq!(fma_hi[1], 0.0f32.to_bits());
    assert_eq!(fma_hi[2], f32::INFINITY.to_bits());
    assert!(f32::from_bits(fma_hi[3]).is_nan());
    assert_eq!(
        out[0].mem[2],
        f32s([-11.0, -17.0, -28.5, 6.0]),
        "vfnmsub231ps"
    );

    // Verify Case 1: vfmaddsub / vfmsubadd
    assert_eq!(out[1].mem[0], f32s([19.0, 32.0, 37.0, 54.0]), "vfmaddsub213ps");
    assert_eq!(out[1].mem[1], f32s([21.0, 28.0, 43.0, 46.0]), "vfmsubadd213ps");
    assert_eq!(out[1].mem[2], f64s([18.5, 32.5]), "vfmaddsub213pd");
    assert_eq!(out[1].mem[3], f64s([21.5, 27.5]), "vfmsubadd213pd");

    // Verify Case 2: scalar FMA3
    assert_eq!(
        out[2].mem[0],
        f32s([one_p.mul_add(one_m, -1.0), 11.0, 22.0, 33.0]),
        "vfmadd213ss preserves upper lanes"
    );
    assert_eq!(
        out[2].mem[1],
        f32s([16.0, 44.0, 55.0, 66.0]),
        "vfmadd213ss op0==op1"
    );
    assert_eq!(out[2].mem[2], f64s([11.25, 777.0]), "vfmadd213sd");

    // Verify Case 3: addsubps / addsubpd
    assert_eq!(
        read_f32s(&out[3].mem[0]),
        [
            (-0.0f32).to_bits(),
            (-0.0f32).to_bits(),
            0.0f32.to_bits(),
            13.0f32.to_bits()
        ],
        "addsubps signed zeros"
    );
    assert_eq!(
        read_f64s(&out[3].mem[1]),
        [0.0f64.to_bits(), f64::INFINITY.to_bits()],
        "addsubpd -0.0 - -0.0 = +0.0 and inf + 5 = inf"
    );
    assert_eq!(out[3].mem[2], f32s([-4.0, 8.0, -4.0, 12.0]), "vaddsubps");

    // Verify Case 4: dpps / dppd fast masks
    assert_eq!(read_f32s(&out[4].mem[0]), [0, 0, 0, 0], "dpps 0x11 -0.0 -> +0.0");
    assert_eq!(out[4].mem[1], f32s([7.0; 4]), "dpps 0x4F");
    assert_eq!(read_f32s(&out[4].mem[2]), [0, 0, 0, 0], "dpps 0x31 -0.0+-0.0 -> +0.0");
    assert_eq!(out[4].mem[3], f32s([0.0, 26.0, 0.0, 26.0]), "dpps 0x5A");
    assert_eq!(out[4].mem[4], f32s([32.0, 0.0, 0.0, 0.0]), "dpps 0x71");
    assert_eq!(out[4].mem[5], f32s([0.0, 56.0, 0.0, 0.0]), "dpps 0xE2");
    assert_eq!(read_f64s(&out[4].mem[6]), [0, 0], "dppd 0x23 -0.0 -> +0.0");
    assert_eq!(out[4].mem[7], f32s([169.0; 4]), "dpps same-reg 0x7F");

    // Verify Case 5: rcpps / rsqrtps
    assert_eq!(
        read_f32s(&out[5].mem[0]),
        [
            0.25f32.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            0.0f32.to_bits(),
            0.0f32.to_bits()
        ],
        "rcpps special cases"
    );
    assert_eq!(
        read_f32s(&out[5].mem[1]),
        [
            0.5f32.to_bits(),
            f32::INFINITY.to_bits(),
            f32::NEG_INFINITY.to_bits(),
            0.0f32.to_bits()
        ],
        "rsqrtps special cases"
    );

    // Verify Case 6: 256-bit conversions
    assert_eq!(out[6].mem[0], i32s([2, i32::MIN, i32::MIN, i32::MIN]), "vcvtpd2dq ymm");
    assert_eq!(out[6].mem[1], vec![0u8; 16], "vcvtpd2dq zeroes ymmh0");
    assert_eq!(out[6].mem[2], i32s([2, i32::MIN, i32::MIN, i32::MIN]), "vcvttpd2dq ymm");
    let cvt_ps = read_f32s(&out[6].mem[3]);
    assert_eq!(cvt_ps[0], 2.5f32.to_bits());
    assert_eq!(cvt_ps[1], (-2147483648.9f64 as f32).to_bits());
    assert_eq!(cvt_ps[2], 2147483648.0f32.to_bits());
    assert!(f32::from_bits(cvt_ps[3]).is_nan());
    assert_eq!(out[6].mem[4], f64s([1.25, -2.5]), "vcvtps2pd lo");
    assert_eq!(out[6].mem[5], f64s([3.75, -4.5]), "vcvtps2pd hi");
    assert_eq!(out[6].mem[6], f64s([i32::MIN as f64, -1.0]), "vcvtdq2pd lo");
    assert_eq!(out[6].mem[7], f64s([0.0, i32::MAX as f64]), "vcvtdq2pd hi");

    // Verify Case 7: shuffles/blends/insertps/haddps/movddup
    assert_eq!(out[7].mem[0], i32s([1, 2, 3, 4]), "pshufd 0xE4");
    assert_eq!(out[7].mem[1], i32s([5, 6, 7, 8]), "pshuflw 0xE4");
    assert_eq!(out[7].mem[2], i32s([9, 10, 11, 12]), "vpermilps 0xE4");
    assert_eq!(out[7].mem[3], i32s([13, 14, 15, 16]), "blendps 0x0");
    assert_eq!(out[7].mem[4], i32s([17, 18, 19, 20]), "blendps 0xF");
    assert_eq!(out[7].mem[5], i32s([29, 30, 31, 32]), "vblendvps same-reg");
    assert_eq!(out[7].mem[6], i32s([100, 0, 20, 400]), "insertps 0x62");
    assert_eq!(out[7].mem[7], i32s([0, 0, 0, 0]), "insertps 0x0F");
    assert_eq!(out[7].mem[8], f32s([3.0, 7.0, 3.0, 7.0]), "haddps same-reg");
    assert_eq!(
        out[7].mem[9],
        xmm_bytes(0xCAFE_BABE_1234_5678, 0xCAFE_BABE_1234_5678),
        "movddup m64"
    );

    // Verify Case 8: vinsertf128/i128 + vpermq
    assert_eq!(out[8].mem[0], xmm_bytes(0xAA, 0xBB), "vinsertf128 sel=0 lo");
    assert_eq!(out[8].mem[1], xmm_bytes(0x33, 0x44), "vinsertf128 sel=0 hi");
    assert_eq!(out[8].mem[2], xmm_bytes(0x11, 0x22), "vinserti128 sel=1 lo");
    assert_eq!(out[8].mem[3], xmm_bytes(0xAA, 0xBB), "vinserti128 sel=1 hi");
    assert_eq!(out[8].mem[4], xmm_bytes(0x44, 0x33), "vpermq 0x1B lo");
    assert_eq!(out[8].mem[5], xmm_bytes(0x22, 0x11), "vpermq 0x1B hi");
    assert_eq!(out[8].mem[6], xmm_bytes(0x11, 0x11), "vpermq 0x00 lo");
    assert_eq!(out[8].mem[7], xmm_bytes(0x11, 0x11), "vpermq 0x00 hi");

    // Verify Case 9: F16C vcvtph2ps (bit-exact!)
    let got_ph_lo = read_f32s(&out[9].mem[0]);
    let got_ph_hi = read_f32s(&out[9].mem[1]);
    for i in 0..4 {
        assert_eq!(got_ph_lo[i], ref_f16_to_f32_bits(h_vals_8[i]), "ph2ps 256 lo lane {i}");
        assert_eq!(got_ph_hi[i], ref_f16_to_f32_bits(h_vals_8[4 + i]), "ph2ps 256 hi lane {i}");
    }
    let got_ph_128 = read_f32s(&out[9].mem[2]);
    for i in 0..4 {
        assert_eq!(got_ph_128[i], ref_f16_to_f32_bits(h_vals_4[i]), "ph2ps 128 lane {i}");
    }
    assert_eq!(out[9].mem[3], vec![0u8; 16], "vcvtph2ps 128 zeroes ymmh2");

    // Verify Case 10: F16C vcvtps2ph Nearest (bit-exact!)
    let got_ps_256 = read_u16s(&out[10].mem[0]);
    for i in 0..4 {
        assert_eq!(got_ps_256[i], ref_f32_to_f16_bits(f_near_lo[i], 0), "ps2ph 256 lo {i}");
        assert_eq!(got_ps_256[4 + i], ref_f32_to_f16_bits(f_near_hi[i], 0), "ps2ph 256 hi {i}");
    }
    let got_ps_128 = read_u16s(&out[10].mem[1]);
    for i in 0..4 {
        assert_eq!(got_ps_128[i], ref_f32_to_f16_bits(f_near_128[i], 0), "ps2ph 128 {i}");
        assert_eq!(got_ps_128[4 + i], 0, "ps2ph 128 upper half zeroed {i}");
    }

    // Verify Case 11: F16C vcvtps2ph all 4 rounding modes (bit-exact!)
    for (idx, mode) in [0u8, 1, 2, 3].into_iter().enumerate() {
        let got = read_u16s(&out[11].mem[idx]);
        for i in 0..4 {
            assert_eq!(
                got[i],
                ref_f32_to_f16_bits(f_mode_lo[i], mode),
                "ps2ph mode {mode} lo lane {i}"
            );
            assert_eq!(
                got[4 + i],
                ref_f32_to_f16_bits(f_mode_hi[i], mode),
                "ps2ph mode {mode} hi lane {i}"
            );
        }
    }
}

#[test]
fn wasm64_simd128_bitmask_extadd_fp_fusion_and_idioms_exec() {
    if !exec_or_skip("wasm64_simd128_bitmask_extadd_fp_fusion_and_idioms_exec") {
        return;
    }
    use super::exec::{DATA_ADDR, R8, R9, R10};
    use crate::lift::STATE_YMM_HI_OFFSET;
    use crate::lift::target::LiftTarget;

    const YMMH_STATE: u64 = STATE_ADDR + STATE_YMM_HI_OFFSET;

    let f32s = |vals: [f32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let u32s = |vals: [u32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let i32s = |vals: [i32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let f64s = |vals: [f64; 2]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let u16s = |vals: [u16; 8]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let u64s = |vals: [u64; 2]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let i64s = |vals: [i64; 2]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };

    let f_bitmask = CODE_BASE;
    let f_psadbw = CODE_BASE + 0x100;
    let f_vtest = CODE_BASE + 0x200;
    let f_vpmov = CODE_BASE + 0x300;
    let f_dyn_shift = CODE_BASE + 0x400;
    let f_idioms = CODE_BASE + 0x500;
    let f_vmaskmov = CODE_BASE + 0x600;
    let f_aes = CODE_BASE + 0x700;
    let f_comis = CODE_BASE + 0x800;

    let prog = Program::new()
        .asm(f_bitmask, |a| {
            a.movmskps(eax, xmm0)?;
            a.vmovmskps(ecx, ymm0)?;
            a.movmskpd(edx, xmm1)?;
            a.vmovmskpd(r8, ymm1)?;
            a.pmovmskb(r9d, xmm2)?;
            a.vpmovmskb(r10, ymm2)?;
            a.ret()
        })
        .asm(f_psadbw, |a| {
            a.vpsadbw(ymm0, ymm1, ymm2)?;
            a.psadbw(xmm3, xmm3)?;
            a.ret()
        })
        .asm(f_vtest, |a| {
            // Test ZF-only (`vtestps` + `sete`) and CF-only (`vtestpd` + `setb`) and both.
            a.vtestps(ymm0, ymm1)?;
            a.sete(al)?;
            a.movzx(eax, al)?;
            a.vtestpd(ymm2, ymm3)?;
            a.setb(cl)?;
            a.movzx(ecx, cl)?;
            a.vtestps(xmm4, xmm5)?;
            a.sete(dl)?;
            a.setb(r8b)?;
            a.movzx(edx, dl)?;
            a.movzx(r8d, r8b)?;
            a.ret()
        })
        .asm(f_vpmov, |a| {
            a.vpmovsxbd(ymm0, qword_ptr(rcx))?;
            a.vpmovzxbd(ymm1, xmm2)?;
            a.vpmovsxbq(ymm3, dword_ptr(rdx))?;
            a.vpmovzxbq(ymm4, xmm2)?;
            a.vpmovsxwq(ymm5, qword_ptr(r8))?;
            a.vpmovzxwq(ymm6, xmm7)?;
            a.ret()
        })
        .asm(f_dyn_shift, |a| {
            a.psllw(xmm0, xmm3)?;
            a.psrld(xmm1, xmm4)?;
            a.psrlq(xmm2, xmm5)?;
            a.ret()
        })
        .asm(f_idioms, |a| {
            a.pcmpeqd(xmm0, xmm0)?;
            a.vpcmpeqb(xmm1, xmm1, xmm1)?; // VEX-128 must also zero ymmh1
            a.pcmpgtw(xmm2, xmm2)?;
            a.psubd(xmm3, xmm3)?;
            a.psubsw(xmm4, xmm4)?;
            a.pminsd(xmm5, xmm5)?;
            a.pavgb(xmm6, xmm6)?;
            a.vinsertps(xmm7, xmm8, xmm9, 0xCE)?; // lane 3 of xmm9 -> lane 0, zero lanes 1..3 + ymmh7
            a.vinsertps(xmm10, xmm11, dword_ptr(rcx), 0x1D)?; // [rcx] -> lane 1, zero lanes 0,2,3 + ymmh10
            a.ret()
        })
        .asm(f_vmaskmov, |a| {
            // 1. Store selected lanes to [rdx]
            a.vpmaskmovq(ymmword_ptr(rdx), ymm2, ymm3)?;
            // 2. Load with aliasing dst == mask (ymm1) from [rcx]
            a.vmaskmovps(ymm1, ymm1, ymmword_ptr(rcx))?;
            // 3. All-zero mask at null address (0) must NOT fault!
            a.vxorps(ymm4, ymm4, ymm4)?;
            a.vmaskmovps(ymm5, ymm4, ymmword_ptr(r8))?;
            a.ret()
        })
        .asm(f_aes, |a| {
            a.aesenc(xmm0, xmm1)?;
            a.aesdec(xmm2, xmm3)?;
            a.aeskeygenassist(xmm4, xmm5, 0x01)?;
            a.ret()
        })
        .asm(f_comis, |a| {
            // Test fused `ja` terminator on `ucomisd xmm2, xmm3` -> rdx, and all 16 fused `setcc`
            // predicates on `ucomiss xmm0, xmm1` -> [r9 + 0..16] (memory stores do not alter flags).
            let mut gt_lbl = a.create_label();
            let mut done_lbl = a.create_label();
            a.ucomisd(xmm2, xmm3)?;
            a.ja(gt_lbl)?;
            a.mov(edx, 0)?;
            a.jmp(done_lbl)?;
            a.set_label(&mut gt_lbl)?;
            a.mov(edx, 1)?;
            a.set_label(&mut done_lbl)?;
            a.ucomiss(xmm0, xmm1)?;
            a.seto(byte_ptr(r9))?;
            a.setno(byte_ptr(r9 + 1))?;
            a.setb(byte_ptr(r9 + 2))?;
            a.setae(byte_ptr(r9 + 3))?;
            a.sete(byte_ptr(r9 + 4))?;
            a.setne(byte_ptr(r9 + 5))?;
            a.setbe(byte_ptr(r9 + 6))?;
            a.seta(byte_ptr(r9 + 7))?;
            a.sets(byte_ptr(r9 + 8))?;
            a.setns(byte_ptr(r9 + 9))?;
            a.setp(byte_ptr(r9 + 10))?;
            a.setnp(byte_ptr(r9 + 11))?;
            a.setl(byte_ptr(r9 + 12))?;
            a.setge(byte_ptr(r9 + 13))?;
            a.setle(byte_ptr(r9 + 14))?;
            a.setg(byte_ptr(r9 + 15))?;
            a.ret()
        });

    let functions = prog.discover();
    // Verify BOTH Wasm64 SIMD128 codegen (native bitmask / extadd_pairwise / load8x8) AND
    // native x86-64-v3 retargeting compile cleanly with llc.
    use crate::lift::compile::{LlvmTool, find_llvm_tool};
    if let Some(LlvmTool::Llc { llc, opt }) = find_llvm_tool() {
        let m = lift_module(functions.values()).expect("lift module");
        let dir = std::env::temp_dir().join(format!("playgta6_simd_opt_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let wll = dir.join("simd_opt_wasm.ll");
        let wbc = dir.join("simd_opt_wasm.bc");
        let wasm_s = dir.join("simd_opt_wasm.s");
        std::fs::write(&wll, &m.ir).expect("write wasm ll");
        let w_input = if let Some(opt_bin) = &opt {
            let st = std::process::Command::new(opt_bin)
                .args(["-O2"])
                .arg(&wll)
                .arg("-o")
                .arg(&wbc)
                .status()
                .expect("run opt");
            assert!(st.success(), "opt -O2 failed on wasm IR");
            &wbc
        } else {
            &wll
        };
        let st = std::process::Command::new(&llc)
            .args(["-mtriple=wasm64-unknown-unknown", "-O2", "-filetype=asm"])
            .arg(w_input)
            .arg("-o")
            .arg(&wasm_s)
            .status()
            .expect("run llc wasm64");
        assert!(st.success(), "llc wasm64 failed");
        let wasm_asm = std::fs::read_to_string(&wasm_s).expect("read wasm asm");
        assert!(wasm_asm.contains("i32x4.bitmask"), "{wasm_asm}");
        assert!(wasm_asm.contains("i64x2.bitmask"), "{wasm_asm}");
        assert!(wasm_asm.contains("i8x16.bitmask"), "{wasm_asm}");
        assert!(wasm_asm.contains("i16x8.extadd_pairwise_i8x16_u"), "{wasm_asm}");
        assert!(wasm_asm.contains("i32x4.extadd_pairwise_i16x8_u"), "{wasm_asm}");
        assert!(wasm_asm.contains("i16x8.load8x8_s"), "{wasm_asm}");
        assert!(wasm_asm.contains("i32x4.load16x4_s"), "{wasm_asm}");

        let native_ir = LiftTarget::parse("native:x86_64-pc-windows-msvc", Some("x86-64-v3"))
            .expect("native target")
            .retarget(m.ir);
        let nll = dir.join("simd_opt_x86.ll");
        let nasm = dir.join("simd_opt_x86.s");
        std::fs::write(&nll, &native_ir).expect("write native ll");
        let st = std::process::Command::new(&llc)
            .args([
                "-mtriple=x86_64-pc-windows-msvc",
                "-mcpu=x86-64-v3",
                "-O2",
                "-filetype=asm",
            ])
            .arg(&nll)
            .arg("-o")
            .arg(&nasm)
            .status()
            .expect("run llc native");
        let _ = std::fs::remove_dir_all(&dir);
        assert!(st.success(), "llc native x86-64-v3 failed on retargeted IR");
    }
    let built = build("simd128_opt", functions.values());
    let data = DATA_ADDR;

    // Case 0: bitmask (128 & 256)
    let mut c0 = Case::new(f_bitmask);
    let mut s0 = vec![0u8; 3 * 16];
    let mut h0 = vec![0u8; 3 * 16];
    // ymm0 (f32): lo = [-0.0, +0.0, -NaN, +NaN] => mask 0b0101 = 5; hi = [+1, -1, -2, +3] => 0b0110 = 6 => 256-bit 0x65
    s0[0..16].copy_from_slice(&u32s([0x8000_0000, 0x0000_0000, 0xFFC0_0001, 0x7FC0_0001]));
    h0[0..16].copy_from_slice(&f32s([1.0, -1.0, -2.0, 3.0]));
    // ymm1 (f64): lo = [-0.0, +1.0] => 0b01 = 1; hi = [+0.0, -Inf] => 0b10 = 2 => 256-bit 0b1001 = 9
    s0[16..32].copy_from_slice(&f64s([-0.0, 1.0]));
    h0[16..32].copy_from_slice(&f64s([0.0, f64::NEG_INFINITY]));
    // ymm2 (i8): lo = alternating 0x80/0x7F => 0x5555; hi = all 0xFF except byte 15=0x00 => 0x7FFF => 256-bit 0x7FFF5555
    let lo_b: [u8; 16] = std::array::from_fn(|i| if i % 2 == 0 { 0x80 } else { 0x7F });
    let mut hi_b = [0xFFu8; 16];
    hi_b[15] = 0x00;
    s0[32..48].copy_from_slice(&lo_b);
    h0[32..48].copy_from_slice(&hi_b);
    c0.mem.push((XMM_STATE, s0));
    c0.mem.push((YMMH_STATE, h0));

    // Case 1: psadbw / vpsadbw
    let mut c1 = Case::new(f_psadbw);
    let mut s1 = vec![0u8; 4 * 16];
    let mut h1 = vec![0u8; 4 * 16];
    s1[16..32].copy_from_slice(&[0u8; 16]);
    s1[32..48].copy_from_slice(&[255u8; 16]); // |0-255|*8 = 2040 per qword
    h1[16..32].copy_from_slice(&[10, 20, 30, 40, 50, 60, 70, 80, 1, 2, 3, 4, 5, 6, 7, 8]);
    h1[32..48].copy_from_slice(&[12, 15, 30, 45, 40, 70, 70, 79, 1, 2, 3, 4, 5, 6, 7, 10]);
    // diffs hi lo-qword: 2+5+0+5+10+10+0+1 = 33; hi-qword: 0*7 + 2 = 2
    s1[48..64].copy_from_slice(&[0xABu8; 16]); // xmm3 non-zero before `psadbw xmm3, xmm3`
    c1.mem.push((XMM_STATE, s1));
    c1.mem.push((YMMH_STATE, h1));
    c1.read.push((XMM_STATE, 16));
    c1.read.push((YMMH_STATE, 16));
    c1.read.push((XMM_STATE + 48, 16));

    // Case 2: vtestps / vtestpd
    let mut c2 = Case::new(f_vtest);
    let mut s2 = vec![0u8; 6 * 16];
    let mut h2 = vec![0u8; 6 * 16];
    // ymm0 & ymm1 sign bits disjoint (even though non-sign bits overlap) -> ZF = 1
    s2[0..16].copy_from_slice(&u32s([0x8000_0001, 0x0000_0001, 0x7FFF_FFFF, 0x0000_0000]));
    s2[16..32].copy_from_slice(&u32s([0x7FFF_FFFF, 0x8000_0001, 0x8000_0000, 0x8000_0000]));
    // ymm2 & ymm3: upper half has !a & b sign bit set -> CF = 0
    h2[32..48].copy_from_slice(&u64s([0, 0]));
    h2[48..64].copy_from_slice(&u64s([0, 0x8000_0000_0000_0000]));
    // xmm4 = all sign bits set, xmm5 = one sign bit set -> (a & b) != 0 (ZF=0), (!a & b) == 0 (CF=1)
    s2[64..80].copy_from_slice(&u32s([0x8000_0000; 4]));
    s2[80..96].copy_from_slice(&u32s([0, 0x8000_0000, 0, 0]));
    c2.mem.push((XMM_STATE, s2));
    c2.mem.push((YMMH_STATE, h2));

    // Case 3: 256-bit vpmovsx*/vpmovzx*
    let mut c3 = Case::new(f_vpmov);
    c3 = c3.reg(RCX, data).reg(RDX, data + 16).reg(R8, data + 32);
    c3.mem.push((data, vec![0x80, 0xFF, 0x00, 0x7F, 0x81, 0xFE, 0x01, 0x7E]));
    c3.mem.push((data + 16, vec![0x80, 0xFF, 0x00, 0x7F]));
    let w4: [i16; 4] = [i16::MIN, -1, 0, i16::MAX];
    let mut w4_bytes = Vec::new();
    for x in w4 {
        w4_bytes.extend_from_slice(&x.to_le_bytes());
    }
    c3.mem.push((data + 32, w4_bytes));
    let mut s3 = vec![0u8; 8 * 16];
    s3[32..40].copy_from_slice(&[0x80, 0xFF, 0x00, 0x7F, 0x81, 0xFE, 0x01, 0x7E]); // xmm2
    s3[112..120].copy_from_slice(&[0x00, 0x80, 0xFF, 0xFF, 0x00, 0x00, 0xFF, 0x7F]); // xmm7
    c3.mem.push((XMM_STATE, s3));
    for reg in [0usize, 1, 3, 4, 5, 6] {
        c3.read.push((XMM_STATE + (reg * 16) as u64, 16));
        c3.read.push((YMMH_STATE + (reg * 16) as u64, 16));
    }

    // Case 4 & 5: dynamic psllw/psrld/psrlq (in-range + overshift)
    let make_shift = |cnt_w: u64, cnt_d: u64, cnt_q: u64| {
        let mut c = Case::new(f_dyn_shift);
        let mut s = vec![0u8; 6 * 16];
        s[0..16].copy_from_slice(&u16s([1, 2, 0x8001, 0xFFFF, 0x1234, 0x5678, 0x9ABC, 0xDEF0]));
        s[16..32].copy_from_slice(&u32s([0x8000_0001, 0xFFFF_FFFF, 0x1234_5678, 0]));
        s[32..48].copy_from_slice(&u64s([0x8000_0000_0000_0001, 0xFEDC_BA98_7654_3210]));
        s[48..64].copy_from_slice(&u64s([cnt_w, 0xDEAD_BEEF]));
        s[64..80].copy_from_slice(&u64s([cnt_d, 0xCAFE_BABE]));
        s[80..96].copy_from_slice(&u64s([cnt_q, 0x1234_5678]));
        c.mem.push((XMM_STATE, s));
        c.read.push((XMM_STATE, 48));
        c
    };
    let c4 = make_shift(3, 4, 8);
    let c5 = make_shift(16, 32, 64);

    // Case 6: same-reg idioms + insertps + VEX upper-128-bit zeroing
    let mut c6 = Case::new(f_idioms).reg(RCX, data + 64);
    c6.mem.push((data + 64, 0x1122_3344u32.to_le_bytes().to_vec()));
    let mut s6 = vec![0x55u8; 12 * 16];
    let h6 = vec![0xEEu8; 12 * 16];
    s6[5 * 16..6 * 16].copy_from_slice(&i32s([-10, 0, 42, i32::MAX]));
    s6[6 * 16..7 * 16].copy_from_slice(&[0, 1, 127, 128, 254, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
    s6[9 * 16..10 * 16].copy_from_slice(&u32s([0xAA, 0xBB, 0xCC, 0xDEAD_BEEF]));
    c6.mem.push((XMM_STATE, s6));
    c6.mem.push((YMMH_STATE, h6));
    for reg in [0usize, 1, 2, 3, 4, 5, 6, 7, 10] {
        c6.read.push((XMM_STATE + (reg * 16) as u64, 16));
    }
    c6.read.push((YMMH_STATE + 16, 16)); // ymmh1 zeroed by vpcmpeqb
    c6.read.push((YMMH_STATE + 7 * 16, 16)); // ymmh7 zeroed by vinsertps
    c6.read.push((YMMH_STATE + 10 * 16, 16)); // ymmh10 zeroed by vinsertps

    // Case 7: vmaskmovps (dst==mask + null ptr with zero mask) & vpmaskmovq store
    let mut c7 = Case::new(f_vmaskmov)
        .reg(RCX, data + 0x100)
        .reg(RDX, data + 0x140)
        .reg(R8, 0x7FFF_0000_0000); // unmapped address for all-zero masked load (must not trap!)
    c7.mem.push((data + 0x100, u32s([10, 20, 30, 40]).into_iter().chain(u32s([50, 60, 70, 80])).collect()));
    c7.mem.push((data + 0x140, vec![0x77u8; 32]));
    let mut s7 = vec![0u8; 6 * 16];
    let mut h7 = vec![0u8; 6 * 16];
    // ymm1 mask (and dst!): lo = [neg, pos, neg, pos], hi = [pos, neg, pos, neg]
    s7[16..32].copy_from_slice(&u32s([0x8000_0000, 0x7FFF_FFFF, 0xFFFF_FFFF, 0]));
    h7[16..32].copy_from_slice(&u32s([0, 0x8000_0000, 0x1234_5678, 0x8000_0001]));
    // ymm2 mask for vpmaskmovq store: lo = [neg, pos], hi = [pos, neg]
    s7[32..48].copy_from_slice(&u64s([0x8000_0000_0000_0000, 0]));
    h7[32..48].copy_from_slice(&u64s([0x7FFF_FFFF_FFFF_FFFF, 0xFFFF_FFFF_FFFF_FFFF]));
    // ymm3 store data
    s7[48..64].copy_from_slice(&u64s([0x1111_2222_3333_4444, 0x5555_6666_7777_8888]));
    h7[48..64].copy_from_slice(&u64s([0x9999_AAAA_BBBB_CCCC, 0xDDDD_EEEE_FFFF_0000]));
    c7.mem.push((XMM_STATE, s7));
    c7.mem.push((YMMH_STATE, h7));
    c7.read.push((XMM_STATE + 16, 16));
    c7.read.push((YMMH_STATE + 16, 16));
    c7.read.push((XMM_STATE + 5 * 16, 16));
    c7.read.push((YMMH_STATE + 5 * 16, 16));
    c7.read.push((data + 0x140, 32));

    // Case 8: AES outlined sub_bytes (FIPS-197 zero state/key sanity)
    let mut c8 = Case::new(f_aes);
    let mut s8 = vec![0u8; 6 * 16];
    s8[0..16].copy_from_slice(&[0u8; 16]);
    s8[16..32].copy_from_slice(&[0u8; 16]);
    s8[32..48].copy_from_slice(&[0x63u8; 16]); // InvSubBytes(0x63) = 0x00 -> InvMixColumns(0) = 0
    s8[48..64].copy_from_slice(&[0u8; 16]);
    s8[80..96].copy_from_slice(&[0u8; 16]); // SubBytes(0) = 0x63; rcon = 0x01 on dwords 1,3
    c8.mem.push((XMM_STATE, s8));
    c8.read.push((XMM_STATE, 16));
    c8.read.push((XMM_STATE + 32, 16));
    c8.read.push((XMM_STATE + 64, 16));

    // Cases 9..13: `ucomiss`/`ucomisd` fused across all 16 CCs: `<`, `==`, `-0.0 == +0.0`, `>`, `NaN`
    let make_comis = |a: f32, b: f32, da: f64, db: f64, out_addr: u64| {
        let mut c = Case::new(f_comis).reg(R9, out_addr);
        let mut s = vec![0u8; 4 * 16];
        s[0..16].copy_from_slice(&f32s([a, 0.0, 0.0, 0.0]));
        s[16..32].copy_from_slice(&f32s([b, 0.0, 0.0, 0.0]));
        s[32..48].copy_from_slice(&f64s([da, 0.0]));
        s[48..64].copy_from_slice(&f64s([db, 0.0]));
        c.mem.push((XMM_STATE, s));
        c.read.push((out_addr, 16));
        c
    };
    let c9 = make_comis(1.0, 2.0, 1.0, 2.0, data + 0x200); // less
    let c10 = make_comis(2.0, 2.0, 2.0, 2.0, data + 0x220); // equal
    let c11 = make_comis(-0.0, 0.0, 3.0, 2.0, data + 0x240); // -0.0 == +0.0 (f32), greater (f64)
    let c12 = make_comis(3.0, 2.0, f64::NAN, 1.0, data + 0x260); // greater (f32), NaN (f64)
    let c13 = make_comis(f32::NAN, 1.0, 1.0, f64::NAN, data + 0x280); // NaN (f32 & f64)

    let out = run(
        &built,
        &[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13],
    );
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }

    // Verify Case 0: bitmask
    assert_eq!(out[0].regs[RAX], 0x5, "movmskps 128");
    assert_eq!(out[0].regs[RCX], 0x65, "vmovmskps 256");
    assert_eq!(out[0].regs[RDX], 0x1, "movmskpd 128");
    assert_eq!(out[0].regs[R8], 0x9, "vmovmskpd 256");
    assert_eq!(out[0].regs[R9], 0x5555, "pmovmskb 128");
    assert_eq!(out[0].regs[R10], 0x7FFF_5555, "vpmovmskb 256");

    // Verify Case 1: psadbw
    assert_eq!(out[1].mem[0], u64s([2040, 2040]), "vpsadbw lo");
    assert_eq!(out[1].mem[1], u64s([33, 2]), "vpsadbw hi");
    assert_eq!(out[1].mem[2], vec![0u8; 16], "psadbw same-reg");

    // Verify Case 2: vtestps/vtestpd
    assert_eq!(out[2].regs[RAX], 1, "vtestps disjoint signs -> ZF=1");
    assert_eq!(out[2].regs[RCX], 0, "vtestpd hi !a&b sign bit -> CF=0");
    assert_eq!(out[2].regs[RDX], 0, "vtestps 128 ZF=0");
    assert_eq!(out[2].regs[R8], 1, "vtestps 128 CF=1");

    // Verify Case 3: 256-bit vpmovsx*/vpmovzx*
    assert_eq!(out[3].mem[0], i32s([-128, -1, 0, 127]), "vpmovsxbd lo");
    assert_eq!(out[3].mem[1], i32s([-127, -2, 1, 126]), "vpmovsxbd hi");
    assert_eq!(out[3].mem[2], u32s([0x80, 0xFF, 0x00, 0x7F]), "vpmovzxbd lo");
    assert_eq!(out[3].mem[3], u32s([0x81, 0xFE, 0x01, 0x7E]), "vpmovzxbd hi");
    assert_eq!(out[3].mem[4], i64s([-128, -1]), "vpmovsxbq lo");
    assert_eq!(out[3].mem[5], i64s([0, 127]), "vpmovsxbq hi");
    assert_eq!(out[3].mem[6], u64s([0x80, 0xFF]), "vpmovzxbq lo");
    assert_eq!(out[3].mem[7], u64s([0x00, 0x7F]), "vpmovzxbq hi");
    assert_eq!(out[3].mem[8], i64s([-32768, -1]), "vpmovsxwq lo");
    assert_eq!(out[3].mem[9], i64s([0, 32767]), "vpmovsxwq hi");
    assert_eq!(out[3].mem[10], u64s([0x8000, 0xFFFF]), "vpmovzxwq lo");
    assert_eq!(out[3].mem[11], u64s([0x0000, 0x7FFF]), "vpmovzxwq hi");

    // Verify Cases 4 & 5: dynamic shifts
    assert_eq!(
        &out[4].mem[0][0..16],
        &u16s([8, 16, 8, 0xFFF8, 0x91A0, 0xB3C0, 0xD5E0, 0xF780]),
        "psllw by 3"
    );
    assert_eq!(
        &out[4].mem[0][16..32],
        &u32s([0x0800_0000, 0x0FFF_FFFF, 0x0123_4567, 0]),
        "psrld by 4"
    );
    assert_eq!(
        &out[4].mem[0][32..48],
        &u64s([0x0080_0000_0000_0000, 0x00FE_DCBA_9876_5432]),
        "psrlq by 8"
    );
    assert_eq!(out[5].mem[0], vec![0u8; 48], "overshifted psllw/psrld/psrlq -> 0");

    // Verify Case 6: same-reg idioms + insertps + VEX zeroing
    assert_eq!(out[6].mem[0], vec![0xFFu8; 16], "pcmpeqd same-reg");
    assert_eq!(out[6].mem[1], vec![0xFFu8; 16], "vpcmpeqb same-reg");
    assert_eq!(out[6].mem[2], vec![0u8; 16], "pcmpgtw same-reg");
    assert_eq!(out[6].mem[3], vec![0u8; 16], "psubd same-reg");
    assert_eq!(out[6].mem[4], vec![0u8; 16], "psubsw same-reg");
    assert_eq!(out[6].mem[5], i32s([-10, 0, 42, i32::MAX]), "pminsd same-reg");
    assert_eq!(
        out[6].mem[6],
        vec![0, 1, 127, 128, 254, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "pavgb same-reg"
    );
    assert_eq!(out[6].mem[7], u32s([0xDEAD_BEEF, 0, 0, 0]), "vinsertps 0xCE");
    assert_eq!(out[6].mem[8], u32s([0, 0x1122_3344, 0, 0]), "vinsertps mem 0x1D");
    assert_eq!(out[6].mem[9], vec![0u8; 16], "vpcmpeqb zeroes ymmh1");
    assert_eq!(out[6].mem[10], vec![0u8; 16], "vinsertps zeroes ymmh7");
    assert_eq!(out[6].mem[11], vec![0u8; 16], "vinsertps mem zeroes ymmh10");

    // Verify Case 7: vmaskmovps / vpmaskmovq
    assert_eq!(out[7].mem[0], u32s([10, 0, 30, 0]), "vmaskmovps lo (dst==mask)");
    assert_eq!(out[7].mem[1], u32s([0, 60, 0, 80]), "vmaskmovps hi (dst==mask)");
    assert_eq!(out[7].mem[2], vec![0u8; 16], "vmaskmovps null zero-mask lo");
    assert_eq!(out[7].mem[3], vec![0u8; 16], "vmaskmovps null zero-mask hi");
    let mut expected_store = vec![0x77u8; 32];
    expected_store[0..8].copy_from_slice(&0x1111_2222_3333_4444u64.to_le_bytes());
    expected_store[24..32].copy_from_slice(&0xDDDD_EEEE_FFFF_0000u64.to_le_bytes());
    assert_eq!(out[7].mem[4], expected_store, "vpmaskmovq partial store");

    // Verify Case 8: AES outlined sub_bytes
    assert_eq!(out[8].mem[0], vec![0x63u8; 16], "aesenc(0, 0) = MixColumns(0x63..0x63) = 0x63..0x63");
    assert_eq!(out[8].mem[1], vec![0x00u8; 16], "aesdec(0x63..0x63, 0) = 0");
    assert_eq!(
        out[8].mem[2],
        u32s([0x6363_6363, 0x6363_6362, 0x6363_6363, 0x6363_6362]),
        "aeskeygenassist(0, 1)"
    );

    // Verify Cases 9..13: fused ucomiss/ucomisd across all 16 x86 CCs!
    // Order: [o, no, b, ae, e, ne, be, a, s, ns, p, np, l, ge, le, g]
    let cc_less: [u8; 16] = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1];
    let cc_eq: [u8; 16] = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0];
    let cc_gt: [u8; 16] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
    let cc_nan: [u8; 16] = [0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0];
    assert_eq!(out[9].mem[0], cc_less, "ucomiss <");
    assert_eq!(out[9].regs[RDX], 0, "ucomisd 1.0 > 2.0 -> false");
    assert_eq!(out[10].mem[0], cc_eq, "ucomiss ==");
    assert_eq!(out[10].regs[RDX], 0, "ucomisd 2.0 > 2.0 -> false");
    assert_eq!(out[11].mem[0], cc_eq, "ucomiss -0.0 == +0.0");
    assert_eq!(out[11].regs[RDX], 1, "ucomisd 3.0 > 2.0 -> true");
    assert_eq!(out[12].mem[0], cc_gt, "ucomiss >");
    assert_eq!(out[12].regs[RDX], 0, "ucomisd NaN > 1.0 -> false");
    assert_eq!(out[13].mem[0], cc_nan, "ucomiss NaN");
    assert_eq!(out[13].regs[RDX], 0, "ucomisd 1.0 > NaN -> false");
}

#[test]
fn wasm64_batch2_simd128_codegen_and_exec_bit_exact() {
    use crate::lift::target::LiftTarget;

    if !exec_or_skip("wasm64_batch2_simd128_codegen_and_exec_bit_exact") {
        return;
    }

    // 1) Verify LLVM Wasm64 + Native x86-64-v3 codegen for Batch 2 SIMD128 intrinsics & lowering.
    if let Some(tool) = crate::lift::compile::find_llvm_tool() {
        struct CodegenProbe {
            name: &'static str,
            func: crate::cfg::Function,
            want_wasm: &'static str,
            // (pattern, max tolerated occurrences). LLVM 23 lowers unrelated flag unpacking in the
            // caller via a few `i8x16.replace_lane`; the scalarised mask expansion this guards
            // against emits 16+, so a small budget keeps the probe LLVM-version-agnostic.
            forbid_wasm: Option<(&'static str, usize)>,
            want_x86: Option<&'static str>,
        }
        let probes = [
            CodegenProbe {
                name: "blendvps_bs",
                func: asm_function(0x1000, |a| {
                    a.vblendvps(ymm0, ymm1, ymm2, ymm3)?;
                    a.ret()
                }),
                want_wasm: "v128.bitselect",
                forbid_wasm: Some(("v128.const", 0)),
                want_x86: Some("vblendvps"),
            },
            CodegenProbe {
                name: "pmulhrsw_q15",
                func: asm_function(0x1100, |a| {
                    a.pmulhrsw(xmm0, xmm1)?;
                    a.ret()
                }),
                want_wasm: "i16x8.q15mulr_sat_s",
                forbid_wasm: None,
                want_x86: Some("pmulhrsw"),
            },
            CodegenProbe {
                name: "cvtpd2dq_bs",
                func: asm_function(0x1200, |a| {
                    a.cvttpd2dq(xmm0, xmm1)?;
                    a.ret()
                }),
                want_wasm: "i32x4.trunc_sat_f64x2_s_zero",
                forbid_wasm: Some(("i32x4.shl", 0)),
                want_x86: None,
            },
            CodegenProbe {
                name: "fma32_bs",
                func: asm_function(0x1300, |a| {
                    a.vfmadd213ps(xmm0, xmm1, xmm2)?;
                    a.ret()
                }),
                want_wasm: "v128.bitselect",
                forbid_wasm: None,
                want_x86: None,
            },
            CodegenProbe {
                name: "pcmpistrm_vec",
                func: asm_function(0x1400, |a| {
                    a.pcmpistrm(xmm0, xmm1, 0x40)?;
                    a.ret()
                }),
                want_wasm: "i8x16.ne",
                forbid_wasm: Some(("i8x16.replace_lane", 7)),
                want_x86: None,
            },
            CodegenProbe {
                name: "vpermd_mul",
                func: asm_function(0x1500, |a| {
                    a.vpermd(ymm0, ymm1, ymm2)?;
                    a.ret()
                }),
                want_wasm: "i32x4.mul",
                forbid_wasm: None,
                want_x86: None,
            },
        ];

        let compile_to_asm = |ll: &std::path::Path,
                              bc: &std::path::Path,
                              out_asm: &std::path::Path,
                              triple: &str,
                              cpu: Option<&str>| {
            match &tool {
                crate::lift::compile::LlvmTool::Llc { llc, opt } => {
                    let input = if let Some(opt_bin) = opt {
                        let st = std::process::Command::new(opt_bin)
                            .args(["-O2"])
                            .arg(ll)
                            .arg("-o")
                            .arg(bc)
                            .status()
                            .expect("run opt");
                        assert!(st.success(), "opt failed on {}", ll.display());
                        bc
                    } else {
                        ll
                    };
                    let mut cmd = std::process::Command::new(llc);
                    cmd.arg(format!("-mtriple={triple}"))
                        .args(["-O2", "-filetype=asm"]);
                    if let Some(c) = cpu {
                        cmd.arg(format!("-mcpu={c}"));
                    }
                    let st = cmd
                        .arg(input)
                        .arg("-o")
                        .arg(out_asm)
                        .status()
                        .expect("run llc");
                    assert!(st.success(), "llc failed on {}", ll.display());
                }
                crate::lift::compile::LlvmTool::Clang(clang) => {
                    let mut cmd = std::process::Command::new(clang);
                    cmd.args(["-S", "-O2", "-Wno-override-module"])
                        .arg(format!("--target={triple}"));
                    if let Some(c) = cpu {
                        cmd.arg(format!("-march={c}"));
                    }
                    let st = cmd
                        .arg(ll)
                        .arg("-o")
                        .arg(out_asm)
                        .status()
                        .expect("run clang");
                    assert!(st.success(), "clang failed on {}", ll.display());
                }
            }
            std::fs::read_to_string(out_asm).expect("read asm")
        };

        let dir = std::env::temp_dir().join(format!(
            "playgta6_batch2_simd_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&dir);
        let native_target = LiftTarget::parse("native:x86_64-pc-windows-msvc", Some("x86-64-v3"))
            .expect("native target");
        for CodegenProbe {
            name,
            func,
            want_wasm,
            forbid_wasm,
            want_x86,
        } in &probes
        {
            let m = lift_module([func]).expect("lift");
            let ll = dir.join(format!("{name}.ll"));
            let bc = dir.join(format!("{name}.bc"));
            let asm_path = dir.join(format!("{name}.s"));
            std::fs::write(&ll, &m.ir).expect("write ll");
            let asm = compile_to_asm(&ll, &bc, &asm_path, "wasm64-unknown-unknown", None);
            assert!(
                asm.contains(want_wasm),
                "{name} expected `{want_wasm}` in wasm64 asm:\n{asm}"
            );
            if let Some((forbid, budget)) = forbid_wasm {
                assert!(
                    asm.matches(forbid).count() <= *budget,
                    "{name} must NOT emit more than {budget} `{forbid}` in wasm64 asm:\n{asm}"
                );
            }
            if let Some(x86_mnem) = want_x86 {
                let native_ir = native_target.retarget(m.ir);
                assert!(!native_ir.contains("llvm.wasm."), "leaked wasm intrinsic");
                let nll = dir.join(format!("{name}_x86.ll"));
                let nbc = dir.join(format!("{name}_x86.bc"));
                let nasm = dir.join(format!("{name}_x86.s"));
                std::fs::write(&nll, &native_ir).expect("write native ll");
                let x86_asm = compile_to_asm(
                    &nll,
                    &nbc,
                    &nasm,
                    "x86_64-pc-windows-msvc",
                    Some("x86-64-v3"),
                );
                assert!(
                    x86_asm.contains(x86_mnem),
                    "{name} expected `{x86_mnem}` in x86-64 asm:\n{x86_asm}"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    // 2) Bit-exact Wasm64 Node/V8 execution across all Batch 2 SIMD128 edge cases!
    let f_blendv = CODE_BASE;
    let f_pmulhrsw = CODE_BASE + 0x100;
    let f_cvtpd2dq = CODE_BASE + 0x200;
    let f_fma32 = CODE_BASE + 0x300;
    let f_pcmpstrm = CODE_BASE + 0x400;
    let f_vpermd_ptest = CODE_BASE + 0x500;

    let program = Program::new()
        .asm(f_blendv, |a| {
            // Legacy blendvps (mask=xmm0), blendvpd (mask=xmm0), pblendvb (mask=xmm0), 256-bit vblendvps
            a.blendvps(xmm1, xmm2)?;
            a.blendvpd(xmm3, xmm4)?;
            a.pblendvb(xmm5, xmm6)?;
            a.vblendvps(ymm7, ymm8, ymm9, ymm10)?;
            // VEX-128 vblendvps must zero upper 128 bits of ymm11.
            a.vblendvps(xmm11, xmm8, xmm9, xmm10)?;
            a.ret()
        })
        .asm(f_pmulhrsw, |a| {
            // Distinct-reg (including (-32768, -32768) wrap!), same-reg (-32768, -32768), and 256-bit
            a.pmulhrsw(xmm0, xmm1)?;
            a.pmulhrsw(xmm2, xmm2)?;
            a.vpmulhrsw(ymm3, ymm4, ymm5)?;
            a.ret()
        })
        .asm(f_cvtpd2dq, |a| {
            a.cvtpd2dq(xmm0, xmm1)?;
            a.cvttpd2dq(xmm2, xmm3)?;
            a.vcvtpd2dq(xmm4, ymm5)?;
            a.vcvttpd2dq(xmm6, ymm7)?;
            a.ret()
        })
        .asm(f_fma32, |a| {
            // Single-rounding FMA32 edge cases (exact cancellation & halfway-to-even).
            a.vfmadd213ps(xmm0, xmm1, xmm2)?;
            a.ret()
        })
        .asm(f_pcmpstrm, |a| {
            // Equal-each (imm = 0x48 = 0x40 unit-mask | 0x08 equal-each | 0x00 u8)
            a.pcmpistrm(xmm1, xmm2, 0x48)?;
            a.movdqa(xmm4, xmm0)?;
            // 16-bit equal-each (imm = 0x49 = 0x40 unit-mask | 0x08 equal-each | 0x01 u16) with VEX
            a.vpcmpistrm(xmm3, xmm5, 0x49)?;
            a.ret()
        })
        .asm(f_vpermd_ptest, |a| {
            a.vpermd(ymm0, ymm1, ymm2)?;
            // Same-reg ptest on zero -> ZF=1, CF=1
            a.ptest(xmm3, xmm3)?;
            a.sete(al)?;
            a.setb(cl)?;
            // Same-reg ptest on non-zero -> ZF=0, CF=1
            a.ptest(xmm4, xmm4)?;
            a.sete(dl)?;
            a.setb(r8b)?;
            a.ret()
        });

    let functions = program.discover();
    let built = build("batch2_simd_exec", functions.values());

    let i16s = |vals: [i16; 8]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let i32s = |vals: [i32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let u32s = |vals: [u32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let f32s = |vals: [f32; 4]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };
    let f64s = |vals: [f64; 2]| -> Vec<u8> { vals.iter().flat_map(|v| v.to_le_bytes()).collect() };

    // Case 0: blendv* (including -0.0 sign bit = 1 selecting b, +0.0 sign bit = 0 selecting a, NaN)
    let mut c0 = Case::new(f_blendv);
    let mut xmm0_in = vec![0u8; 12 * 16];
    let mut ymmh0_in = vec![0xAAu8; 12 * 16];
    // xmm0 mask: lane0=-0.0 (bit31=1), lane1=+0.0 (bit31=0), lane2=0x80000000, lane3=0x7FFFFFFF
    xmm0_in[0..16].copy_from_slice(&u32s([0x8000_0000, 0x0000_0000, 0x8000_0001, 0x7FFF_FFFF]));
    xmm0_in[16..32].copy_from_slice(&u32s([10, 20, 30, 40])); // xmm1 (a)
    xmm0_in[32..48].copy_from_slice(&u32s([100, 200, 300, 400])); // xmm2 (b)
    xmm0_in[48..64].copy_from_slice(&xmm_bytes(0x1111, 0x2222)); // xmm3 (a f64)
    xmm0_in[64..80].copy_from_slice(&xmm_bytes(0xAAAA, 0xBBBB)); // xmm4 (b f64)
    xmm0_in[80..96].copy_from_slice(&[1u8; 16]); // xmm5 (a bytes)
    xmm0_in[96..112].copy_from_slice(&[9u8; 16]); // xmm6 (b bytes)
    xmm0_in[128..144].copy_from_slice(&u32s([1, 2, 3, 4])); // xmm8 lo
    ymmh0_in[128..144].copy_from_slice(&u32s([5, 6, 7, 8])); // ymmh8 hi
    xmm0_in[144..160].copy_from_slice(&u32s([11, 22, 33, 44])); // xmm9 lo
    ymmh0_in[144..160].copy_from_slice(&u32s([55, 66, 77, 88])); // ymmh9 hi
    xmm0_in[160..176].copy_from_slice(&u32s([0x8000_0000, 0, 0, 0xFFFF_FFFF])); // xmm10 mask lo
    ymmh0_in[160..176].copy_from_slice(&u32s([0, 0x8000_0000, 0xFFFF_FFFF, 0])); // ymmh10 mask hi
    c0.mem.push((XMM_STATE, xmm0_in));
    c0.mem.push((YMMH_STATE, ymmh0_in));
    c0.read.push((XMM_STATE + 16, 16)); // xmm1
    c0.read.push((XMM_STATE + 48, 16)); // xmm3
    c0.read.push((XMM_STATE + 80, 16)); // xmm5
    c0.read.push((XMM_STATE + 7 * 16, 16)); // xmm7 lo
    c0.read.push((YMMH_STATE + 7 * 16, 16)); // ymmh7 hi
    c0.read.push((XMM_STATE + 11 * 16, 16)); // xmm11 lo
    c0.read.push((YMMH_STATE + 11 * 16, 16)); // ymmh11 hi (must be zeroed!)

    // Case 1: pmulhrsw / vpmulhrsw (including (-32768, -32768) wrap in distinct & same-reg!)
    let mut c1 = Case::new(f_pmulhrsw);
    let mut xmm1_in = vec![0u8; 6 * 16];
    let mut ymmh1_in = vec![0u8; 6 * 16];
    let a16 = [-32768i16, -32768, 32767, 16384, -16384, 1, 0, -1];
    let b16 = [-32768i16, 32767, 32767, 16384, 16384, 16384, -32768, 16384];
    xmm1_in[0..16].copy_from_slice(&i16s(a16));
    xmm1_in[16..32].copy_from_slice(&i16s(b16));
    xmm1_in[32..48].copy_from_slice(&i16s([-32768, 32767, 0, -1, 16384, -16384, 1000, -1000]));
    xmm1_in[64..80].copy_from_slice(&i16s(a16));
    xmm1_in[80..96].copy_from_slice(&i16s(b16));
    ymmh1_in[64..80].copy_from_slice(&i16s([-32768; 8]));
    ymmh1_in[80..96].copy_from_slice(&i16s([-32768; 8]));
    c1.mem.push((XMM_STATE, xmm1_in));
    c1.mem.push((YMMH_STATE, ymmh1_in));
    c1.read.push((XMM_STATE, 16)); // xmm0
    c1.read.push((XMM_STATE + 32, 16)); // xmm2 (same-reg)
    c1.read.push((XMM_STATE + 48, 16)); // ymm3 lo
    c1.read.push((YMMH_STATE + 48, 16)); // ymm3 hi

    // Case 2: cvtpd2dq / cvttpd2dq / 256-bit vcvt(t)pd2dq edge cases (NaN, +-Inf, 2^31, -2^31, ties)
    let mut c2 = Case::new(f_cvtpd2dq);
    let mut xmm2_in = vec![0u8; 8 * 16];
    let mut ymmh2_in = vec![0xFFu8; 8 * 16];
    xmm2_in[16..32].copy_from_slice(&f64s([2.5, f64::NAN])); // xmm1 -> [2, INT_MIN, 0, 0]
    xmm2_in[48..64].copy_from_slice(&f64s([2147483648.0, -2147483648.0])); // xmm3 -> [INT_MIN, INT_MIN, 0, 0]
    xmm2_in[80..96].copy_from_slice(&f64s([3.5, -3.5])); // ymm5 lo -> [4, -4]
    ymmh2_in[80..96].copy_from_slice(&f64s([f64::INFINITY, -f64::INFINITY])); // ymm5 hi -> [INT_MIN, INT_MIN]
    xmm2_in[112..128].copy_from_slice(&f64s([2147483647.9, -2147483648.9])); // ymm7 lo -> [2147483647, INT_MIN]
    ymmh2_in[112..128].copy_from_slice(&f64s([f64::NAN, -0.0])); // ymm7 hi -> [INT_MIN, 0]
    c2.mem.push((XMM_STATE, xmm2_in));
    c2.mem.push((YMMH_STATE, ymmh2_in));
    c2.read.push((XMM_STATE, 16)); // xmm0
    c2.read.push((XMM_STATE + 32, 16)); // xmm2
    c2.read.push((XMM_STATE + 64, 16)); // xmm4
    c2.read.push((XMM_STATE + 96, 16)); // xmm6

    // Case 3: vfmadd213ps single-rounding exactness
    let mut c3 = Case::new(f_fma32);
    let a_f32 = [
        f32::from_bits(0x3F80_0001), // (1 + 2^-23)
        f32::from_bits(0x3F80_0001),
        f32::from_bits(0x4B80_0001), // large exact
        f32::NAN,
    ];
    let b_f32 = [
        f32::from_bits(0x3F80_0001), // (1 + 2^-23)^2 = 1 + 2^-22 + 2^-46
        f32::from_bits(0x3F80_0001),
        2.0f32,
        1.0f32,
    ];
    let c_f32 = [
        -f32::from_bits(0x3F80_0002), // cancels 1 + 2^-22, leaving exact 2^-46!
        f32::from_bits(0x2880_0000),  // 2^-46 -> tests round-to-odd tie-breaking
        -1.0f32,
        2.0f32,
    ];
    c3.mem
        .push((XMM_STATE, [f32s(a_f32), f32s(b_f32), f32s(c_f32)].concat()));
    c3.read.push((XMM_STATE, 16));

    // Case 4: pcmpistrm (n==16) & vpcmpistrm (n==8) unit-mask expansion
    let mut c4 = Case::new(f_pcmpstrm);
    let mut xmm4_in = vec![0u8; 6 * 16];
    let ymmh4_in = vec![0xBBu8; 16];
    // xmm1 vs xmm2 (bytes): match at indices 0, 2, 7, 8, 15
    let s1 = *b"AbCdEfGhIjKlMnOp";
    let s2 = *b"AxCyEzGhIxKxMxXp";
    xmm4_in[16..32].copy_from_slice(&s1);
    xmm4_in[32..48].copy_from_slice(&s2);
    // xmm3 vs xmm5 (u16 words): match at indices 0, 3, 7
    xmm4_in[48..64].copy_from_slice(&i16s([1, 2, 3, 4, 5, 6, 7, 8]));
    xmm4_in[80..96].copy_from_slice(&i16s([1, 9, 9, 4, 9, 9, 9, 8]));
    c4.mem.push((XMM_STATE, xmm4_in));
    c4.mem.push((YMMH_STATE, ymmh4_in));
    c4.read.push((XMM_STATE + 64, 16)); // xmm4 (8-bit unit mask)
    c4.read.push((XMM_STATE, 16)); // xmm0 (16-bit unit mask)
    c4.read.push((YMMH_STATE, 16)); // ymmh0 (zeroed by vpcmpistrm)

    // Case 5: vpermd (dword_swizzle_indices) + same-reg ptest
    let mut c5 = Case::new(f_vpermd_ptest);
    let mut xmm5_in = vec![0u8; 5 * 16];
    let mut ymmh5_in = vec![0u8; 5 * 16];
    xmm5_in[16..32].copy_from_slice(&u32s([7, 0, 5, 2])); // idx lo
    ymmh5_in[16..32].copy_from_slice(&u32s([3, 6, 1, 4])); // idx hi
    xmm5_in[32..48].copy_from_slice(&u32s([10, 20, 30, 40])); // src lo
    ymmh5_in[32..48].copy_from_slice(&u32s([50, 60, 70, 80])); // src hi
    xmm5_in[48..64].copy_from_slice(&u32s([0, 0, 0, 0])); // xmm3 == 0
    xmm5_in[64..80].copy_from_slice(&u32s([0, 1, 0, 0])); // xmm4 != 0
    c5.mem.push((XMM_STATE, xmm5_in));
    c5.mem.push((YMMH_STATE, ymmh5_in));
    c5.read.push((XMM_STATE, 16));
    c5.read.push((YMMH_STATE, 16));

    let out = run(&built, &[c0, c1, c2, c3, c4, c5]);
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }

    // Verify Case 0: blendv*
    assert_eq!(out[0].mem[0], u32s([100, 20, 300, 40]), "blendvps");
    // xmm0 as <2 x u64>: lane0=0x0000_0000_8000_0000 (bit63=0 -> a), lane1=0x7FFF_FFFF_8000_0001 (bit63=0 -> a)
    assert_eq!(out[0].mem[1], xmm_bytes(0x1111, 0x2222), "blendvpd");
    // xmm0 bytes (LE): byte3=0x80, byte11=0x80, bytes 12..=14=0xFF (from 0x7FFF_FFFF) have bit7=1 (sel b);
    // all other bytes (including byte15=0x7F) have bit7=0 (sel a).
    let mut want_pb = [1u8; 16];
    for i in [3, 11, 12, 13, 14] {
        want_pb[i] = 9;
    }
    assert_eq!(out[0].mem[2], want_pb, "pblendvb");
    assert_eq!(out[0].mem[3], u32s([11, 2, 3, 44]), "vblendvps ymm lo");
    assert_eq!(out[0].mem[4], u32s([5, 66, 77, 8]), "vblendvps ymm hi");
    assert_eq!(out[0].mem[5], u32s([11, 2, 3, 44]), "vblendvps xmm lo");
    assert_eq!(out[0].mem[6], vec![0u8; 16], "vblendvps xmm zeroes ymmh11");

    // Verify Case 1: pmulhrsw exact ref
    let ref_hrsw = |x: i16, y: i16| -> i16 {
        ((((i32::from(x) * i32::from(y)) >> 14) + 1) >> 1) as i16
    };
    let want_hrsw: [i16; 8] = std::array::from_fn(|i| ref_hrsw(a16[i], b16[i]));
    assert_eq!(want_hrsw[0], -32768, "sanity: (-32768,-32768) wraps to -32768");
    assert_eq!(out[1].mem[0], i16s(want_hrsw), "pmulhrsw distinct");
    let same_in = [-32768i16, 32767, 0, -1, 16384, -16384, 1000, -1000];
    let want_same: [i16; 8] = std::array::from_fn(|i| ref_hrsw(same_in[i], same_in[i]));
    assert_eq!(out[1].mem[1], i16s(want_same), "pmulhrsw same-reg");
    assert_eq!(out[1].mem[2], i16s(want_hrsw), "vpmulhrsw lo");
    assert_eq!(out[1].mem[3], i16s([-32768; 8]), "vpmulhrsw hi all (-32768,-32768)");

    // Verify Case 2: cvt(t)pd2dq
    assert_eq!(out[2].mem[0], i32s([2, i32::MIN, 0, 0]), "cvtpd2dq");
    assert_eq!(out[2].mem[1], i32s([i32::MIN, i32::MIN, 0, 0]), "cvttpd2dq");
    assert_eq!(out[2].mem[2], i32s([4, -4, i32::MIN, i32::MIN]), "vcvtpd2dq 256");
    assert_eq!(out[2].mem[3], i32s([i32::MAX, i32::MIN, i32::MIN, 0]), "vcvttpd2dq 256");

    // Verify Case 3: vfmadd213ps single-rounding exactness
    let got_fma: Vec<u32> = out[3].mem[0]
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        got_fma[0],
        (2.0f32.powi(-46)).to_bits(),
        "fma32 exact cancellation 2^-46"
    );
    assert_eq!(
        got_fma[1],
        a_f32[1].mul_add(b_f32[1], c_f32[1]).to_bits(),
        "fma32 round-to-odd tie-breaking"
    );
    assert_eq!(
        got_fma[2],
        a_f32[2].mul_add(b_f32[2], c_f32[2]).to_bits(),
        "fma32 large exact"
    );
    assert!(f32::from_bits(got_fma[3]).is_nan(), "fma32 NaN");

    // Verify Case 4: pcmpistrm / vpcmpistrm unit masks
    let want_m8: Vec<u8> = (0..16)
        .map(|i| if s1[i] == s2[i] { 0xFF } else { 0x00 })
        .collect();
    assert_eq!(out[4].mem[0], want_m8, "pcmpistrm 8-bit unit mask");
    assert_eq!(
        out[4].mem[1],
        i16s([-1, 0, 0, -1, 0, 0, 0, -1]),
        "vpcmpistrm 16-bit unit mask"
    );
    assert_eq!(out[4].mem[2], vec![0u8; 16], "vpcmpistrm zeroes ymmh0");

    // Verify Case 5: vpermd + same-reg ptest
    assert_eq!(out[5].mem[0], u32s([80, 10, 60, 30]), "vpermd lo");
    assert_eq!(out[5].mem[1], u32s([40, 70, 20, 50]), "vpermd hi");
    assert_eq!(out[5].regs[RAX] & 0xFF, 1, "ptest(0,0) ZF=1");
    assert_eq!(out[5].regs[RCX] & 0xFF, 1, "ptest(0,0) CF=1");
    assert_eq!(out[5].regs[RDX] & 0xFF, 0, "ptest(nz,nz) ZF=0");
    assert_eq!(out[5].regs[R8] & 0xFF, 1, "ptest(nz,nz) CF=1");
}
