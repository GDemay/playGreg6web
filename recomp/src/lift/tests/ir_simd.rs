//! IR-level tests of the SSE/AVX lifter: XMM state modelling, unsupported forms and fuzzing of
//! the SSE/VEX opcode space.

use std::collections::BTreeMap;

use iced_x86::code_asm::*;

use super::ir::Lcg;
use super::{Program, asm_function};
use crate::cfg;
use crate::lift::{HELPER_UNIMPLEMENTED, LIFTED_FUNCTION_ATTRIBUTES, lift_function, lift_module};

#[test]
fn used_xmm_registers_get_slots_synced_like_gprs() {
    let func = Program::new()
        .asm(0x1000, |a| {
            a.addps(xmm0, xmm5)?;
            a.call(0x2000)?;
            a.movaps(xmm5, xmm0)?;
            a.ret()
        })
        .asm(0x2000, |a| a.ret())
        .function(0x1000);
    let lifted = lift_function(&func).expect("lifts");
    let ir = &lifted.ir;
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    assert!(ir.contains("  %xmm0 = alloca <2 x i64>\n"), "{ir}");
    assert!(ir.contains("  %xmm5 = alloca <2 x i64>\n"), "{ir}");
    assert!(
        !ir.contains("%xmm1 = alloca"),
        "only used registers get a slot"
    );
    let state_slot = |idx: u32| format!("ptr %state, i64 0, i32 5, i64 {idx}\n");
    // entry + after the call + before the call + before the return
    assert_eq!(ir.matches(&state_slot(0)).count(), 4, "{ir}");
    assert_eq!(ir.matches(&state_slot(5)).count(), 4, "{ir}");
    let call = ir.find("call void @sub_2000").expect("call");
    let before = &ir[..call];
    let after = &ir[call..];
    assert!(
        before.rfind("store <2 x i64>").unwrap()
            > before.rfind("load <2 x i64>, ptr %xmm0").unwrap()
    );
    assert!(
        after.contains("load <2 x i64>, ptr %t"),
        "reloaded after the call: {after}"
    );

    let plain = asm_function(0x1000, |a| {
        a.add(rax, rcx)?;
        a.ret()
    });
    let lifted = lift_function(&plain).expect("lifts");
    assert!(!lifted.ir.contains("xmm"), "{}", lifted.ir);
}

#[test]
fn ymm_upper_halves_get_slots_only_where_avx_touches_them() {
    let func = asm_function(0x1000, |a| {
        a.addps(xmm0, xmm1)?; // legacy SSE: upper halves untouched
        a.vaddps(ymm2, ymm3, ymm4)?; // 256-bit: both halves
        a.vmulps(xmm5, xmm5, xmm5)?; // VEX-128: zeroes ymm5's upper half
        a.ret()
    });
    let lifted = lift_function(&func).expect("lifts");
    let ir = &lifted.ir;
    assert!(lifted.unsupported.is_empty(), "{:?}", lifted.unsupported);
    for idx in [2, 3, 4, 5] {
        assert!(
            ir.contains(&format!(
                "  %ymmh{idx} = alloca <2 x i64>
"
            )),
            "{ir}"
        );
        // entry load + exit store of state field 6 (`ymm_hi`)
        assert_eq!(
            ir.matches(&format!(
                "ptr %state, i64 0, i32 6, i64 {idx}
"
            ))
            .count(),
            2,
            "{ir}"
        );
    }
    for idx in [0, 1] {
        assert!(!ir.contains(&format!("%ymmh{idx} =")), "{ir}");
    }
    assert!(
        ir.contains("store <2 x i64> zeroinitializer, ptr %ymmh5"),
        "{ir}"
    );
}

#[test]
fn lifted_functions_require_wasm_simd() {
    let func = asm_function(0x1000, |a| {
        a.sqrtps(xmm0, xmm1)?;
        a.ret()
    });
    let module = lift_module([
        &func,
        &asm_function(0x2000, |a| {
            a.sqrtps(xmm2, xmm3)?;
            a.sqrtss(xmm2, xmm3)?;
            a.ret()
        }),
    ])
    .expect("lifts");
    assert!(
        module
            .ir
            .contains("define void @sub_1000(ptr noalias %state) #0 {")
    );
    assert!(module.ir.contains(&format!(
        "attributes #0 = {{ {LIFTED_FUNCTION_ATTRIBUTES} }}"
    )));
    assert!(LIFTED_FUNCTION_ATTRIBUTES.contains("+simd128"));
    assert_eq!(
        module
            .ir
            .matches("declare <4 x float> @llvm.sqrt.v4f32(<4 x float>)")
            .count(),
        1,
        "intrinsics are declared once"
    );
    assert!(module.ir.contains("declare float @llvm.sqrt.f32(float)"));
    assert!(!module.ir.contains("llvm.x86"), "no x86 intrinsics on wasm");
}

#[test]
fn unsupported_vector_forms_trap_with_precise_reasons() {
    let cases: Vec<(Vec<u8>, &str)> = vec![
        (
            super::assemble(0x1000, |a| a.vmpsadbw(ymm0, ymm1, ymm2, 0)),
            "256-bit form of Vmpsadbw is not supported yet",
        ),
        (
            super::assemble(0x1000, |a| a.sha1rnds4(xmm0, xmm1, 0)),
            "Sha1rnds4 is not supported yet",
        ),
        (
            vec![0x62, 0xF1, 0x74, 0x08, 0x58, 0xC2], // EVEX vaddps xmm0, xmm1, xmm2
            "AVX-512 (EVEX)",
        ),
        (
            super::assemble(0x1000, |a| a.paddb(mm0, mm1)),
            "MMX instructions",
        ),
        (super::assemble(0x1000, |a| a.emms()), "MMX instructions"),
        (
            super::assemble(0x1000, |a| a.pcmpistri(xmm0, xmm1, 0)),
            "Pcmpistri is not supported yet",
        ),
        (
            super::assemble(0x1000, |a| a.maskmovdqu(xmm0, xmm1)),
            "Maskmovdqu is not supported yet",
        ),
        (
            super::assemble(0x1000, |a| {
                a.vpgatherdd_3(xmm0, dword_ptr(rax + xmm1), xmm2)
            }),
            "gather",
        ),
    ];
    for (bytes, reason) in cases {
        let code = [bytes.as_slice(), &[0xC3]].concat();
        let func = Program::new().bytes(0x1000, &code).function(0x1000);
        let lifted = lift_function(&func).expect("lifts");
        assert_eq!(
            lifted.unsupported.len(),
            1,
            "{reason}: {:?}
{}",
            lifted.unsupported,
            lifted.ir
        );
        assert!(
            lifted.unsupported[0].reason.contains(reason),
            "{:?} should mention {reason:?}",
            lifted.unsupported[0]
        );
        assert!(lifted.ir.contains(HELPER_UNIMPLEMENTED));
    }
}

#[test]
fn ldmxcsr_checks_the_rounding_mode_at_run_time() {
    let func = asm_function(0x1000, |a| {
        a.ldmxcsr(dword_ptr(rcx))?;
        a.ret()
    });
    let lifted = lift_function(&func).expect("lifts");
    assert!(lifted.unsupported.is_empty());
    assert!(lifted.ir.contains("and i32 %"), "{}", lifted.ir);
    assert!(
        lifted.ir.contains("-40960"),
        "RC + reserved bit mask: {}",
        lifted.ir
    );
    assert!(
        lifted.ir.contains(&format!(
            "call void @{HELPER_UNIMPLEMENTED}(ptr %state, i64 4096)"
        )),
        "{}",
        lifted.ir
    );
}

/// Random instructions from the SSE/SSE2..4.2 (`[66|F2|F3] [REX] 0F xx`, `0F 38 xx`, `0F 3A xx`)
/// and VEX (`C5`/`C4`) opcode spaces with random ModRM/SIB/displacement/immediate bytes, run
/// through discovery.
pub(super) fn random_sse_program(count: usize) -> BTreeMap<u64, cfg::Function> {
    let mut rng = Lcg(0x55E_F022);
    let mut program = Program::new();
    for i in 0..count {
        let mut bytes = Vec::new();
        for _ in 0..24 {
            let r = rng.next();
            match r % 8 {
                0..=3 => {
                    let prefix = [None, Some(0x66), Some(0xF2), Some(0xF3)][(r >> 8) as usize % 4];
                    bytes.extend(prefix);
                    if r >> 12 & 3 == 0 {
                        bytes.push(0x40 | (r >> 16) as u8 & 0xF);
                    }
                    bytes.push(0x0F);
                    match r >> 20 & 7 {
                        0 => bytes.extend([0x38, (r >> 24) as u8]),
                        1 => bytes.extend([0x3A, (r >> 24) as u8]),
                        _ => bytes.push(0x10 | (r >> 24) as u8),
                    }
                }
                4 | 5 => bytes.extend([0xC5, (r >> 8) as u8 | 0x80, (r >> 16) as u8]),
                _ => bytes.extend([
                    0xC4,
                    (r >> 8) as u8 & 0xE3 | 0xE0,
                    (r >> 16) as u8 & 0x7F,
                    (r >> 24) as u8,
                ]),
            }
            // ModRM (mostly register-register), then SIB/displacement/immediate material.
            let modrm = (r >> 32) as u8;
            bytes.push(if r >> 40 & 3 == 0 {
                modrm
            } else {
                modrm | 0xC0
            });
            let tail = (r >> 44) as usize % 3;
            bytes.extend((0..tail).map(|_| rng.next() as u8));
        }
        bytes.push(0xC3);
        program = program.bytes(0x50_0000 + (i as u64) * 0x400, &bytes);
    }
    program.discover()
}

#[test]
fn random_sse_byte_sequences_never_panic() {
    let functions = random_sse_program(800);
    let module = lift_module(functions.values()).expect("discovered random SSE code lifts");
    assert_eq!(module.functions.len(), functions.len());
    let traps: usize = module.functions.iter().map(|f| f.unsupported.len()).sum();
    let lifted = module.ir.matches("load <2 x i64>, ptr %xmm").count();
    assert!(
        traps > 0 && lifted > 200,
        "traps {traps}, xmm loads {lifted}"
    );
}
