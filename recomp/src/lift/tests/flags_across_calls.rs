//! Flag liveness across calls (`flag_liveness`): which flags a callee may observe on entry, the
//! caller storing only those before a direct call or tail jump, and the results staying exact.

use iced_x86::code_asm::*;

use super::exec::{CODE_BASE, Case, RAX, RCX, RDX, build, exec_or_skip, run};
use super::{Program, asm_function};
use crate::lift::{FlagSummaries, function_symbol, lift_module};

const CF: u8 = 1 << 0;
const PF: u8 = 1 << 1;
const AF: u8 = 1 << 2;
const ZF: u8 = 1 << 3;
const SF: u8 = 1 << 4;
const OF: u8 = 1 << 5;
const ALL: u8 = CF | PF | AF | ZF | SF | OF;

const CALLER: u64 = CODE_BASE;
/// Overwrites every flag but AF before returning (`shr`, `xor`).
const KILL: u64 = CODE_BASE + 0x100;
/// Reads CF (`adc`).
const ADC: u64 = CODE_BASE + 0x200;
/// Touches no flag: all of them reach the caller unchanged.
const PASS: u64 = CODE_BASE + 0x300;
/// Reads ZF (`setz`); also reached by a tail jump.
const SETZ: u64 = CODE_BASE + 0x400;
const ADC_CALLER: u64 = CODE_BASE + 0x500;
const PASS_CALLER: u64 = CODE_BASE + 0x600;
const TAIL_CALLER: u64 = CODE_BASE + 0x700;
const SETZ_CALLER: u64 = CODE_BASE + 0x800;
/// Calls KILL first, then uses CF: needs nothing from its caller but AF (KILL keeps AF).
const NESTED: u64 = CODE_BASE + 0x900;
/// `shl rax, cl`: a zero count keeps the flags, so they are read.
const SHIFT_CL: u64 = CODE_BASE + 0xA00;

fn program() -> Program {
    Program::new()
        // rax = rcx * 3, flags from the compare and imul, then the leaf
        .asm(CALLER, |a| {
            a.cmp(rcx, rdx)?;
            a.imul_3(rax, rcx, 3)?;
            a.call(KILL)?;
            a.ret()
        })
        .asm(KILL, |a| {
            a.push(rbx)?;
            a.mov(rbx, rax)?;
            a.shr(rbx, 3)?;
            a.xor(rax, rbx)?;
            a.pop(rbx)?;
            a.ret()
        })
        .asm(ADC, |a| {
            a.mov(eax, 0)?;
            a.adc(rax, 0)?;
            a.ret()
        })
        .asm(PASS, |a| {
            a.mov(rax, rcx)?;
            a.ret()
        })
        .asm(SETZ, |a| {
            a.mov(eax, 0)?;
            a.setz(al)?;
            a.ret()
        })
        .asm(ADC_CALLER, |a| {
            a.cmp(rcx, rdx)?;
            a.call(ADC)?;
            a.ret()
        })
        .asm(PASS_CALLER, |a| {
            a.cmp(rcx, rdx)?;
            a.call(PASS)?;
            a.ret()
        })
        .asm(TAIL_CALLER, |a| {
            a.cmp(rcx, rdx)?;
            a.jmp(SETZ)
        })
        .asm(SETZ_CALLER, |a| {
            a.call(SETZ)?;
            a.ret()
        })
        .asm(NESTED, |a| {
            a.call(KILL)?;
            a.adc(rax, 0)?;
            a.ret()
        })
        .asm(SHIFT_CL, |a| {
            a.shl(rax, cl)?;
            a.add(rax, 1)?;
            a.ret()
        })
}

#[test]
fn summaries_hold_the_flags_a_callee_may_observe() {
    let functions = program().discover();
    let summaries = FlagSummaries::compute(functions.values());
    let live = |va: u64| summaries.live_in(va).expect("analysed");
    assert_eq!(live(KILL), AF, "shr/xor overwrite all but AF");
    assert_eq!(live(ADC), CF, "adc reads CF and overwrites the rest");
    assert_eq!(live(PASS), ALL);
    assert_eq!(live(SETZ), ALL);
    assert_eq!(live(CALLER), 0, "cmp overwrites every flag first");
    assert_eq!(live(TAIL_CALLER), 0);
    assert_eq!(
        live(SETZ_CALLER),
        ALL,
        "a call passes the callee's live-in on"
    );
    assert_eq!(live(NESTED), AF, "only AF survives the call to KILL");
    assert_eq!(
        live(SHIFT_CL),
        CF | PF | ZF | SF | OF,
        "shl by cl may keep them"
    );
    assert_eq!(summaries.live_in(CODE_BASE + 0xF000), None);
}

#[test]
fn summaries_treat_unknown_code_as_reading_every_flag() {
    // A call to a function outside the analysed set, an indirect call and an instruction the
    // analysis does not model all count as reading every flag.
    let outside = asm_function(0x1000, |a| {
        a.call(0x2000)?;
        a.xor(eax, eax)?;
        a.ret()
    });
    let indirect = asm_function(0x3000, |a| {
        a.call(rax)?;
        a.xor(eax, eax)?;
        a.ret()
    });
    let unmodelled = asm_function(0x4000, |a| {
        a.cpuid()?;
        a.xor(eax, eax)?;
        a.ret()
    });
    let summaries = FlagSummaries::compute([&outside, &indirect, &unmodelled]);
    assert_eq!(summaries.live_in(0x1000), Some(ALL));
    assert_eq!(summaries.live_in(0x3000), Some(ALL));
    assert_eq!(summaries.live_in(0x4000), Some(ALL));
}

/// The caller's LLVM block from its first guest block up to the call or tail call of `callee`.
fn before_transfer(ir: &str, caller: u64, callee: u64) -> String {
    let func = ir
        .split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(caller))))
        .expect("caller defined");
    let call = format!("call void @{}(ptr %state)", function_symbol(callee));
    let end = func.find(&call).expect("calls the callee");
    let start = func.find(&format!("bb_{caller:X}:")).expect("first block");
    func[start..end].to_string()
}

fn stored_flags(text: &str) -> Vec<usize> {
    (0..6)
        .filter(|i| text.contains(&format!("ptr %state.flag{i}\n")))
        .collect()
}

#[test]
fn calls_store_only_the_flags_the_callee_may_observe() {
    let functions = program().discover();
    let ir = lift_module(functions.values()).expect("lifts").ir;
    // KILL observes AF only: CF/OF of the imul (and the rest) are not stored before the call.
    assert_eq!(stored_flags(&before_transfer(&ir, CALLER, KILL)), vec![2]);
    // ADC observes CF, PASS every flag.
    assert_eq!(
        stored_flags(&before_transfer(&ir, ADC_CALLER, ADC)),
        vec![0]
    );
    assert_eq!(
        stored_flags(&before_transfer(&ir, PASS_CALLER, PASS)),
        vec![0, 1, 2, 3, 4, 5]
    );
    // The tail jump into SETZ stores the flags SETZ observes (all of them).
    let tail = ir
        .split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(TAIL_CALLER))))
        .expect("tail caller");
    let exit = tail.find(&format!("ext_{SETZ:X}:")).expect("tail exit");
    assert_eq!(stored_flags(&tail[exit..]), vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn calls_without_summaries_store_every_flag() {
    let functions = program().discover();
    let lifted = crate::lift::lift_function(&functions[&CALLER]).expect("lifts");
    assert_eq!(
        stored_flags(&before_transfer(&lifted.ir, CALLER, KILL)),
        vec![0, 1, 2, 3, 4, 5]
    );
}

#[test]
fn flags_stay_exact_across_calls() {
    if !exec_or_skip("flags_stay_exact_across_calls") {
        return;
    }
    let functions = program().discover();
    let built = build("flags_across_calls", functions.values());
    let pairs = [
        (1u64, 2u64),
        (2, 1),
        (7, 7),
        (0, u64::MAX),
        (0x8000_0000_0000_0000, 1),
    ];
    let mut cases = Vec::new();
    for &(c, d) in &pairs {
        for va in [CALLER, ADC_CALLER, PASS_CALLER, TAIL_CALLER, NESTED] {
            cases.push(Case::new(va).reg(RCX, c).reg(RDX, d).reg(RAX, c ^ 0x5A));
        }
    }
    let outcomes = run(&built, &cases);
    for (i, &(c, d)) in pairs.iter().enumerate() {
        let o = &outcomes[i * 5..i * 5 + 5];
        for (k, out) in o.iter().enumerate() {
            assert!(out.error.is_empty(), "pair {i} case {k}: {}", out.error);
        }
        let diff = c.wrapping_sub(d);
        let cmp_cf = c < d;
        let cmp_zf = c == d;
        let cmp_af = (c ^ d ^ diff) & 0x10 != 0;
        let cmp_sf = diff >> 63 != 0;
        // CALLER: KILL's result and flags, AF from the cmp (imul leaves AF unchanged).
        let x = c.wrapping_mul(3);
        let r = x ^ (x >> 3);
        let parity = (r as u8).count_ones().is_multiple_of(2);
        assert_eq!(o[0].regs[RAX], r, "caller rax ({c:#x}, {d:#x})");
        assert_eq!(
            o[0].flags,
            [false, parity, cmp_af, r == 0, r >> 63 != 0, false],
            "caller flags ({c:#x}, {d:#x})"
        );
        // ADC_CALLER: the callee adds the caller's CF.
        assert_eq!(o[1].regs[RAX], u64::from(cmp_cf), "adc ({c:#x}, {d:#x})");
        // PASS_CALLER: the compare's flags come back unchanged.
        assert_eq!(o[2].regs[RAX], c);
        assert_eq!(
            (o[2].flags[0], o[2].flags[2], o[2].flags[3], o[2].flags[4]),
            (cmp_cf, cmp_af, cmp_zf, cmp_sf),
            "pass ({c:#x}, {d:#x})"
        );
        // TAIL_CALLER: setz after the tail jump sees the compare's ZF.
        assert_eq!(o[3].regs[RAX], u64::from(cmp_zf), "tail ({c:#x}, {d:#x})");
        // NESTED: KILL clears CF, so the adc adds nothing.
        let x = c ^ 0x5A;
        assert_eq!(o[4].regs[RAX], x ^ (x >> 3), "nested ({c:#x}, {d:#x})");
    }
}

#[test]
fn shifts_skip_flags_a_later_instruction_overwrites() {
    // xor rewrites CF, OF, ZF, SF and PF; the shift flags are dead (AF stays unchanged).
    let program = Program::new().asm(CALLER, |a| {
        a.shr(rax, 3)?;
        a.xor(rax, rcx)?;
        a.ret()
    });
    let functions = program.discover();
    let ir = crate::lift::lift_function(&functions[&CALLER])
        .expect("lifts")
        .ir;
    assert_eq!(ir.matches("ctpop").count(), 1, "only the xor PF:\n{ir}");
    assert_eq!(ir.matches("lshr i64").count(), 1, "only the result:\n{ir}");
}

#[test]
fn constant_flags_are_stored_without_a_zext() {
    // sar by one leaves OF = false; the sync-out stores the constant byte directly.
    let program = Program::new().asm(CALLER, |a| {
        a.sar(rax, 1)?;
        a.ret()
    });
    let functions = program.discover();
    let ir = crate::lift::lift_function(&functions[&CALLER])
        .expect("lifts")
        .ir;
    assert!(!ir.contains("zext i1 false"), "{ir}");
    assert!(ir.contains("store i8 0,"), "{ir}");
}

/// Flags reloaded from `%state.flag{i}` in `text` (matching `load i8, ptr %state.flag{i}`).
fn loaded_flags(text: &str) -> Vec<usize> {
    (0..6)
        .filter(|i| text.contains(&format!("load i8, ptr %state.flag{i},")))
        .collect()
}

/// The `after_call` block text following the first call inside `caller`, up to the next block
/// label or return.
fn after_first_call(ir: &str, caller: u64) -> String {
    let func = ir
        .split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(caller))))
        .expect("caller defined");
    let after_lbl = func.find("\nafter_call").expect("after_call label");
    let body = &func[after_lbl + 1..];
    let body = &body[body.find(":\n").expect("label end") + 2..];
    let end = body
        .lines()
        .take_while(|l| l.starts_with("  ") || l.is_empty())
        .map(|l| l.len() + 1)
        .sum::<usize>()
        .min(body.len());
    body[..end].to_string()
}

#[test]
fn r67_dead_flags_after_direct_and_indirect_calls_are_not_reloaded() {
    const DIRECT_DEAD: u64 = CODE_BASE + 0x1000;
    const DIRECT_PARTIAL: u64 = CODE_BASE + 0x1100;
    const DIRECT_LIVE: u64 = CODE_BASE + 0x1200;
    const INDIRECT_DEAD: u64 = CODE_BASE + 0x1300;
    const INDIRECT_LIVE: u64 = CODE_BASE + 0x1400;
    const SET_FLAGS_CALLEE: u64 = CODE_BASE + 0x1500;

    let prog = Program::new()
        // All 6 flags are immediately overwritten by `add rax, rcx` after the direct call.
        .asm(DIRECT_DEAD, |a| {
            a.call(SET_FLAGS_CALLEE)?;
            a.add(rax, rcx)?;
            a.ret()
        })
        // Only CF is read by `adc rax, 0`, which then overwrites all 6 flags.
        .asm(DIRECT_PARTIAL, |a| {
            a.call(SET_FLAGS_CALLEE)?;
            a.adc(rax, 0)?;
            a.ret()
        })
        // Returning immediately after the call leaves all 6 flags live (caller tracks flags via
        // the earlier `cmp`, so all 6 must be reloaded after the call before `ret` stores them).
        .asm(DIRECT_LIVE, |a| {
            a.cmp(rcx, rdx)?;
            a.call(SET_FLAGS_CALLEE)?;
            a.ret()
        })
        // Indirect call whose returned flags are completely overwritten by `sub rax, rcx`.
        .asm(INDIRECT_DEAD, |a| {
            a.call(r11)?;
            a.sub(rax, rcx)?;
            a.ret()
        })
        // Indirect call whose returned ZF is read by `setz`, then all flags are killed by `add`.
        .asm(INDIRECT_LIVE, |a| {
            a.call(r11)?;
            a.setz(dl)?;
            a.add(rax, rdx)?;
            a.ret()
        })
        // Callee sets flags via `cmp rcx, rdx` and returns `rcx` in `rax`.
        .asm(SET_FLAGS_CALLEE, |a| {
            a.mov(rax, rcx)?;
            a.cmp(rcx, rdx)?;
            a.ret()
        });

    let functions = prog.discover();
    let ir = lift_module(functions.values()).expect("lifts").ir;

    assert_eq!(
        loaded_flags(&after_first_call(&ir, DIRECT_DEAD)),
        Vec::<usize>::new(),
        "dead flags after direct call must not be reloaded:\n{}",
        after_first_call(&ir, DIRECT_DEAD)
    );
    assert_eq!(
        loaded_flags(&after_first_call(&ir, DIRECT_PARTIAL)),
        vec![0],
        "only live CF should be reloaded after direct call before adc:\n{}",
        after_first_call(&ir, DIRECT_PARTIAL)
    );
    assert_eq!(
        loaded_flags(&after_first_call(&ir, DIRECT_LIVE)),
        vec![0, 1, 2, 3, 4, 5],
        "all flags live at ret must still be reloaded:\n{}",
        after_first_call(&ir, DIRECT_LIVE)
    );
    assert_eq!(
        loaded_flags(&after_first_call(&ir, INDIRECT_DEAD)),
        Vec::<usize>::new(),
        "dead flags after indirect call must not be reloaded:\n{}",
        after_first_call(&ir, INDIRECT_DEAD)
    );
    assert_eq!(
        loaded_flags(&after_first_call(&ir, INDIRECT_LIVE)),
        vec![3],
        "only live ZF should be reloaded after indirect call before setz+add:\n{}",
        after_first_call(&ir, INDIRECT_LIVE)
    );

    if !exec_or_skip("r67_dead_flags_after_direct_and_indirect_calls_are_not_reloaded") {
        return;
    }
    let built = build("r67_post_call_dead_flags", functions.values());
    let cases = [
        // DIRECT_DEAD: rcx=10, rdx=20 -> callee returns rax=10, caller adds rcx -> 20
        Case::new(DIRECT_DEAD).reg(RCX, 10).reg(RDX, 20),
        // DIRECT_PARTIAL: rcx=10 < rdx=20 -> callee sets CF=1, caller adc rax,0 -> 11
        Case::new(DIRECT_PARTIAL).reg(RCX, 10).reg(RDX, 20),
        // DIRECT_PARTIAL: rcx=20 > rdx=10 -> callee sets CF=0, caller adc rax,0 -> 20
        Case::new(DIRECT_PARTIAL).reg(RCX, 20).reg(RDX, 10),
        // INDIRECT_DEAD: calls SET_FLAGS_CALLEE via r11, then sub rax, rcx -> 0 (ZF=1)
        Case::new(INDIRECT_DEAD)
            .reg(super::exec::R11, SET_FLAGS_CALLEE)
            .reg(RCX, 15)
            .reg(RDX, 25),
        // INDIRECT_LIVE: rcx==rdx=7 -> ZF=1 -> dl=1 -> rax = 7 + (rdx with dl=1)
        Case::new(INDIRECT_LIVE)
            .reg(super::exec::R11, SET_FLAGS_CALLEE)
            .reg(RCX, 7)
            .reg(RDX, 7),
        // INDIRECT_LIVE: rcx=7 != rdx=9 -> ZF=0 -> dl=0 -> rax = 7 + (rdx with dl=0)
        Case::new(INDIRECT_LIVE)
            .reg(super::exec::R11, SET_FLAGS_CALLEE)
            .reg(RCX, 7)
            .reg(RDX, 9),
    ];
    let out = run(&built, &cases);
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }
    assert_eq!(out[0].regs[RAX], 20);
    assert_eq!(out[0].flags[3], false, "DIRECT_DEAD ZF");
    assert_eq!(out[1].regs[RAX], 11, "DIRECT_PARTIAL CF=1");
    assert_eq!(out[2].regs[RAX], 20, "DIRECT_PARTIAL CF=0");
    assert_eq!(out[3].regs[RAX], 0, "INDIRECT_DEAD rax");
    assert_eq!(out[3].flags[3], true, "INDIRECT_DEAD ZF=1");
    assert_eq!(out[4].regs[RAX], 8, "INDIRECT_LIVE ZF=1 -> dl=1, 7+1=8");
    assert_eq!(out[5].regs[RAX], 7, "INDIRECT_LIVE ZF=0 -> dl=0, 7+0=7");
}
