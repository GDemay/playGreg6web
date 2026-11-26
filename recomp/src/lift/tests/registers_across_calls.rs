//! Guest state a callee may touch (`flag_liveness`, "Registers a callee never touches"): the
//! caller keeping the registers a direct callee never reads or writes out of `%state` across the
//! call, and the results staying exact.

use iced_x86::code_asm::*;

use super::exec::{
    CODE_BASE, Case, R8, R9, RAX, RBX, RCX, RDX, RSI, RSP, STATE_ADDR, build, exec_or_skip, run,
};
use super::{Program, asm_function};
use crate::lift::{
    FlagSummaries, STATE_XMM_OFFSET, STATE_YMM_HI_OFFSET, function_symbol, lift_module,
};

const XMM_SHIFT: u32 = 16;
const YMMH_SHIFT: u32 = 32;
const ALL_XMM: u64 = 0xFFFF_u64 << XMM_SHIFT;
const ALL_YMMH: u64 = 0xFFFF_u64 << YMMH_SHIFT;
const OTHER: u64 = 1u64 << 48;
const ALL: u64 = (OTHER << 1) - 1;

const fn bit(reg: usize) -> u64 {
    1u64 << reg
}

const fn xmm_bit(reg: usize) -> u64 {
    1u64 << (XMM_SHIFT + reg as u32)
}

const fn ymmh_bit(reg: usize) -> u64 {
    1u64 << (YMMH_SHIFT + reg as u32)
}

/// Keeps `rcx`, `rdx`, `rsi` and `r9` live across a call to LEAF.
const CALLER: u64 = CODE_BASE;
/// `push rbx; mov rbx, rax; shr rbx, 3; xor rax, rbx; pop rbx; ret`.
const LEAF: u64 = CODE_BASE + 0x100;
/// Adds `r8`, then calls LEAF: touches LEAF's registers and `r8`.
const MID: u64 = CODE_BASE + 0x200;
/// Calls MID with `r8`, `r9` live.
const MID_CALLER: u64 = CODE_BASE + 0x300;
/// Counts `rcx` down recursively.
const REC: u64 = CODE_BASE + 0x400;
/// Tail-jumps to LEAF after touching `rdx`.
const TAIL: u64 = CODE_BASE + 0x500;
/// Uses `xmm0` and `rax`.
const VECTOR: u64 = CODE_BASE + 0x600;
/// Calls VECTOR with `rsi` live: `rsi` stays out of `%state` while `rax` and `rsp` sync.
const VECTOR_CALLER: u64 = CODE_BASE + 0x700;
/// Calls TAIL with `rsi` live.
const TAIL_CALLER: u64 = CODE_BASE + 0x800;
/// Uses an unmodelled/trapping instruction (`ldmxcsr`), so everything must sync.
const UNMODELLED_CALLEE: u64 = CODE_BASE + 0x900;
/// Calls UNMODELLED_CALLEE with `rsi` live: everything is stored and reloaded.
const UNMODELLED_CALLER: u64 = CODE_BASE + 0xA00;

fn program() -> Program {
    Program::new()
        .asm(CALLER, |a| {
            a.mov(r9, rcx)?;
            a.add(rsi, rdx)?;
            a.call(LEAF)?;
            a.add(rax, r9)?;
            a.add(rax, rsi)?;
            a.add(rcx, rdx)?;
            a.ret()
        })
        .asm(LEAF, |a| {
            a.push(rbx)?;
            a.mov(rbx, rax)?;
            a.shr(rbx, 3)?;
            a.xor(rax, rbx)?;
            a.pop(rbx)?;
            a.ret()
        })
        .asm(MID, |a| {
            a.add(rax, r8)?;
            a.call(LEAF)?;
            a.ret()
        })
        .asm(MID_CALLER, |a| {
            a.mov(r9, r8)?;
            a.call(MID)?;
            a.sub(rax, r9)?;
            a.ret()
        })
        .asm(REC, |a| {
            let mut done = a.create_label();
            a.dec(rcx)?;
            a.jz(done)?;
            a.call(REC)?;
            a.set_label(&mut done)?;
            a.ret()
        })
        .asm(TAIL, |a| {
            a.add(rax, rdx)?;
            a.jmp(LEAF)
        })
        .asm(VECTOR, |a| {
            a.movq(xmm0, rax)?;
            a.movq(rax, xmm0)?;
            a.ret()
        })
        .asm(VECTOR_CALLER, |a| {
            a.add(rsi, 1)?;
            a.call(VECTOR)?;
            a.add(rax, rsi)?;
            a.ret()
        })
        .asm(TAIL_CALLER, |a| {
            a.add(rsi, 1)?;
            a.call(TAIL)?;
            a.add(rax, rsi)?;
            a.ret()
        })
        .asm(UNMODELLED_CALLEE, |a| {
            a.cpuid()?;
            a.ret()
        })
        .asm(UNMODELLED_CALLER, |a| {
            a.add(rsi, 1)?;
            a.call(UNMODELLED_CALLEE)?;
            a.add(rax, rsi)?;
            a.ret()
        })
}

#[test]
fn summaries_hold_the_state_a_call_may_touch() {
    let functions = program().discover();
    let summaries = FlagSummaries::compute(functions.values());
    let touched = |va: u64| summaries.touched(va).expect("analysed");
    let leaf = bit(RAX) | bit(RBX) | bit(RSP);
    assert_eq!(touched(LEAF), leaf);
    assert_eq!(
        touched(MID),
        leaf | bit(R8),
        "a call adds the callee's state"
    );
    assert_eq!(
        touched(REC),
        bit(RCX) | bit(RSP),
        "recursion is a fixed point"
    );
    assert_eq!(
        touched(TAIL),
        leaf | bit(RDX),
        "a tail jump adds the target's state"
    );
    assert_eq!(
        touched(VECTOR),
        bit(RAX) | bit(RSP) | xmm_bit(0),
        "legacy SSE tracks only exact xmm0 + GPRs (no ymmh, no OTHER)"
    );
    assert_eq!(
        touched(VECTOR_CALLER),
        bit(RAX) | bit(RSP) | bit(RSI) | xmm_bit(0)
    );
    assert_eq!(touched(UNMODELLED_CALLEE), ALL);
    assert_eq!(touched(UNMODELLED_CALLER), ALL);
    assert_eq!(summaries.touched(CODE_BASE + 0xF000), None);
}

#[test]
fn summaries_treat_unknown_code_as_touching_everything() {
    let outside = asm_function(0x1000, |a| {
        a.call(0x2000)?;
        a.ret()
    });
    let indirect = asm_function(0x3000, |a| {
        a.call(rax)?;
        a.ret()
    });
    let unmodelled = asm_function(0x4000, |a| {
        a.cpuid()?;
        a.ret()
    });
    let trap = asm_function(0x5000, |a| a.int3());
    let dispatch = asm_function(0x6000, |a| a.jmp(rax));
    let looping = asm_function(0x7000, |a| {
        let mut top = a.create_label();
        a.set_label(&mut top)?;
        a.loop_(top)?;
        a.ret()
    });
    let summaries =
        FlagSummaries::compute([&outside, &indirect, &unmodelled, &trap, &dispatch, &looping]);
    for va in [0x1000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000] {
        assert_eq!(summaries.touched(va), Some(ALL), "{va:#x}");
    }
}

/// The caller's function text.
fn function_ir(ir: &str, va: u64) -> &str {
    ir.split("define ")
        .find(|f| f.contains(&format!("@{}(ptr noalias", function_symbol(va))))
        .expect("defined")
}

/// GPR indices whose state slot `text` addresses.
fn gprs(text: &str) -> Vec<usize> {
    (0..16)
        .filter(|i| text.contains(&format!("ptr %state.gpr{i}\n")))
        .collect()
}

/// `(before, after)`: the caller's text from its first block up to the call of `callee`, and
/// from the call to the first instruction after the call's pending test.
fn around_call(ir: &str, caller: u64, callee: u64) -> (String, String) {
    let func = function_ir(ir, caller);
    let call = format!("call void @{}(ptr %state)", function_symbol(callee));
    let at = func.find(&call).expect("calls the callee");
    let start = func.find(&format!("bb_{caller:X}:")).expect("first block");
    let after = &func[at..];
    let resumed = after.find("after_call").expect("pending test");
    let body = &after[resumed..];
    let body = &body[body.find(":\n").expect("label") + 2..];
    // Up to the first lifted instruction after the reloads (each caller continues with add/sub).
    let end = ["= add i64", "= sub i64"]
        .iter()
        .filter_map(|op| body.find(op))
        .min()
        .unwrap_or(body.len());
    (func[start..at].to_string(), body[..end].to_string())
}

#[test]
fn calls_keep_untouched_registers_out_of_the_state() {
    let functions = program().discover();
    let ir = lift_module(functions.values()).expect("lifts").ir;
    // LEAF touches rax, rbx, rsp: rcx, rdx, rsi, r9 stay in the caller.
    let (before, after) = around_call(&ir, CALLER, LEAF);
    assert_eq!(gprs(&before), vec![RAX, RSP], "stores before the call");
    assert_eq!(gprs(&after), vec![RAX, RSP], "reloads after the call");
    // MID also touches r8.
    let (before, after) = around_call(&ir, MID_CALLER, MID);
    assert_eq!(gprs(&before), vec![RAX, RSP, R8]);
    assert_eq!(gprs(&after), vec![RAX, RSP, R8]);
    // TAIL also touches rdx; rsi stays.
    let (before, _) = around_call(&ir, TAIL_CALLER, TAIL);
    assert_eq!(gprs(&before), vec![RAX, RSP]);
    // VECTOR touches rax, rsp, xmm0: rsi stays out of `%state`.
    let (before, after) = around_call(&ir, VECTOR_CALLER, VECTOR);
    assert_eq!(gprs(&before), vec![RAX, RSP]);
    assert_eq!(gprs(&after), vec![RAX, RSP]);
    // UNMODELLED_CALLEE may touch anything: every tracked register goes through the state.
    let (before, after) = around_call(&ir, UNMODELLED_CALLER, UNMODELLED_CALLEE);
    assert_eq!(gprs(&before), vec![RAX, RSP, RSI]);
    assert_eq!(gprs(&after), vec![RAX, RSP, RSI]);
    // The caller's own return still writes every tracked register back.
    let caller = function_ir(&ir, CALLER);
    let ret = caller.rfind("ret void").expect("returns");
    let tail = &caller[..ret];
    let last_call = tail.rfind("after_call").expect("after the call");
    assert_eq!(gprs(&tail[last_call..]), vec![RAX, RCX, RDX, RSP, RSI, R9]);
}

#[test]
fn calls_without_summaries_sync_every_register() {
    let functions = program().discover();
    let lifted = crate::lift::lift_function(&functions[&CALLER]).expect("lifts");
    let (before, after) = around_call(&lifted.ir, CALLER, LEAF);
    assert_eq!(gprs(&before), vec![RAX, RCX, RDX, RSP, RSI, R9]);
    assert_eq!(gprs(&after), vec![RAX, RCX, RDX, RSP, RSI, R9]);
}

#[test]
fn registers_stay_exact_across_calls() {
    if !exec_or_skip("registers_stay_exact_across_calls") {
        return;
    }
    let functions = program().discover();
    let built = build("registers_across_calls", functions.values());
    let inputs = [
        (1u64, 2u64, 3u64),
        (0x1234_5678_9ABC, u64::MAX, 7),
        (5, 0, 1 << 63),
    ];
    let mut cases = Vec::new();
    for &(c, d, s) in &inputs {
        for va in [CALLER, MID_CALLER, TAIL_CALLER, VECTOR_CALLER] {
            cases.push(
                Case::new(va)
                    .reg(RCX, c)
                    .reg(RDX, d)
                    .reg(RSI, s)
                    .reg(R8, d ^ 0x55)
                    .reg(RAX, c ^ 0x5A),
            );
        }
        cases.push(Case::new(REC).reg(RCX, (c & 0xF) + 1));
    }
    let outcomes = run(&built, &cases);
    let leaf = |x: u64| x ^ (x >> 3);
    for (i, &(c, d, s)) in inputs.iter().enumerate() {
        let o = &outcomes[i * 5..i * 5 + 5];
        for (k, out) in o.iter().enumerate() {
            assert!(out.error.is_empty(), "input {i} case {k}: {}", out.error);
        }
        let start = |va: u64| {
            Case::new(va)
                .reg(RCX, c)
                .reg(RDX, d)
                .reg(RSI, s)
                .reg(R8, d ^ 0x55)
                .reg(RAX, c ^ 0x5A)
                .regs
        };
        // CALLER
        let mut want = start(CALLER);
        want[R9] = c;
        want[RSI] = s.wrapping_add(d);
        want[RAX] = leaf(c ^ 0x5A)
            .wrapping_add(c)
            .wrapping_add(s.wrapping_add(d));
        want[RCX] = c.wrapping_add(d);
        assert_eq!(o[0].regs, want, "caller ({c:#x}, {d:#x}, {s:#x})");
        // MID_CALLER
        let mut want = start(MID_CALLER);
        want[R9] = d ^ 0x55;
        want[RAX] = leaf((c ^ 0x5A).wrapping_add(d ^ 0x55)).wrapping_sub(d ^ 0x55);
        assert_eq!(o[1].regs, want, "mid ({c:#x}, {d:#x}, {s:#x})");
        // TAIL_CALLER
        let mut want = start(TAIL_CALLER);
        want[RSI] = s.wrapping_add(1);
        want[RAX] = leaf((c ^ 0x5A).wrapping_add(d)).wrapping_add(s.wrapping_add(1));
        assert_eq!(o[2].regs, want, "tail ({c:#x}, {d:#x}, {s:#x})");
        // VECTOR_CALLER
        let mut want = start(VECTOR_CALLER);
        want[RSI] = s.wrapping_add(1);
        want[RAX] = (c ^ 0x5A).wrapping_add(s.wrapping_add(1));
        assert_eq!(o[3].regs, want, "vector ({c:#x}, {d:#x}, {s:#x})");
        // REC counts rcx down to zero; rsp is back where it started.
        assert_eq!(o[4].regs[RCX], 0);
        assert_eq!(o[4].regs[RSP], Case::new(REC).regs[RSP]);
    }
}

/// XMM indices whose `%state.xmm{i}` slot `text` addresses.
fn state_xmms(text: &str) -> Vec<usize> {
    (0..16)
        .filter(|i| text.contains(&format!("ptr %state.xmm{i},")))
        .collect()
}

/// YMMH indices whose `%state.ymmh{i}` slot `text` addresses.
fn state_ymmhs(text: &str) -> Vec<usize> {
    (0..16)
        .filter(|i| text.contains(&format!("ptr %state.ymmh{i},")))
        .collect()
}

/// `(before, after)`: text around the direct call to `callee` inside `caller` (from the
/// start of `caller`'s first block to the call, and from `after_call` to the next vector op or
/// return).
fn around_vec_call(ir: &str, caller: u64, callee: u64) -> (String, String) {
    let func = function_ir(ir, caller);
    let call = format!("call void @{}(ptr %state)", function_symbol(callee));
    let at = func.find(&call).expect("calls the callee");
    let start = func.find(&format!("bb_{caller:X}:")).expect("first block");
    let after = &func[at..];
    let resumed = after.find("after_call").expect("pending test");
    let body = &after[resumed..];
    let body = &body[body.find(":\n").expect("label") + 2..];
    // Stop at the first non-sync instruction (next call, vector op, or return epilog store).
    let end = body
        .find("call void @")
        .or_else(|| body.find("fadd "))
        .or_else(|| body.find("add <"))
        .or_else(|| body.find("add i64"))
        .unwrap_or(body.len());
    (func[start..at].to_string(), body[..end].to_string())
}

#[test]
fn r61_per_register_xmm_ymmh_summaries_and_spill_elision() {
    const CALLER_DISJOINT: u64 = CODE_BASE + 0x1000;
    const CALLEE_XMM1: u64 = CODE_BASE + 0x1100;
    const CALLEE_PURE_GPR: u64 = CODE_BASE + 0x1200;
    const CALLEE_VXORPS2: u64 = CODE_BASE + 0x1300;
    const CALLER_VXORPS2: u64 = CODE_BASE + 0x1400;
    const CALLEE_VZEROUPPER: u64 = CODE_BASE + 0x1500;
    const CALLER_VZEROUPPER: u64 = CODE_BASE + 0x1600;
    const CALLEE_VZEROALL: u64 = CODE_BASE + 0x1700;
    const CALLEE_BLENDV: u64 = CODE_BASE + 0x1800;
    const CALLEE_LDMXCSR: u64 = CODE_BASE + 0x1900;
    const CALLEE_FXSAVE: u64 = CODE_BASE + 0x1A00;
    const CALLEE_SHA: u64 = CODE_BASE + 0x1B00;
    const CALLEE_VCOMISS: u64 = CODE_BASE + 0x1C00;

    let prog = Program::new()
        // Touches only xmm0 (legacy SSE, no AVX), calls CALLEE_XMM1 and CALLEE_PURE_GPR.
        .asm(CALLER_DISJOINT, |a| {
            a.paddq(xmm0, xmm0)?;
            a.call(CALLEE_XMM1)?;
            a.call(CALLEE_PURE_GPR)?;
            a.paddq(xmm0, xmm0)?;
            a.ret()
        })
        .asm(CALLEE_XMM1, |a| {
            a.paddq(xmm1, xmm1)?;
            a.ret()
        })
        .asm(CALLEE_PURE_GPR, |a| {
            a.add(rax, rcx)?;
            a.ret()
        })
        // VEX-128 `vxorps xmm2, xmm2, xmm2`: writes xmm2 AND zeroes ymmh2.
        .asm(CALLEE_VXORPS2, |a| {
            a.vxorps(xmm2, xmm2, xmm2)?;
            a.ret()
        })
        // Holds ymm0, ymm2, ymm3 across call to CALLEE_VXORPS2.
        .asm(CALLER_VXORPS2, |a| {
            a.vpaddq(ymm0, ymm0, ymm0)?;
            a.vpaddq(ymm2, ymm2, ymm2)?;
            a.vpaddq(ymm3, ymm3, ymm3)?;
            a.call(CALLEE_VXORPS2)?;
            a.vpaddq(ymm0, ymm0, ymm2)?;
            a.vpaddq(ymm0, ymm0, ymm3)?;
            a.ret()
        })
        .asm(CALLEE_VZEROUPPER, |a| {
            a.vzeroupper()?;
            a.ret()
        })
        // Holds ymm0 (xmm0 + ymmh0) and xmm1 across call to CALLEE_VZEROUPPER.
        .asm(CALLER_VZEROUPPER, |a| {
            a.vpaddq(ymm0, ymm0, ymm0)?;
            a.paddq(xmm1, xmm1)?;
            a.call(CALLEE_VZEROUPPER)?;
            a.vpaddq(ymm0, ymm0, ymm0)?;
            a.paddq(xmm1, xmm1)?;
            a.ret()
        })
        .asm(CALLEE_VZEROALL, |a| {
            a.vzeroall()?;
            a.ret()
        })
        .asm(CALLEE_BLENDV, |a| {
            a.blendvps(xmm3, xmm4)?;
            a.ret()
        })
        .asm(CALLEE_LDMXCSR, |a| {
            a.ldmxcsr(dword_ptr(rcx))?;
            a.ret()
        })
        .asm(CALLEE_FXSAVE, |a| {
            a.fxsave(ptr(rcx))?;
            a.ret()
        })
        .asm(CALLEE_SHA, |a| {
            a.sha1rnds4(xmm0, xmm1, 0)?;
            a.ret()
        })
        .asm(CALLEE_VCOMISS, |a| {
            a.vcomiss(xmm0, xmm1)?;
            a.ret()
        });

    let functions = prog.discover();
    let summaries = FlagSummaries::compute(functions.values());
    let touched = |va: u64| summaries.touched(va).expect("analysed");

    // 1. Exact summaries
    assert_eq!(touched(CALLEE_XMM1), xmm_bit(1) | bit(RSP));
    assert_eq!(touched(CALLEE_PURE_GPR), bit(RAX) | bit(RCX) | bit(RSP));
    assert_eq!(
        touched(CALLEE_VXORPS2),
        xmm_bit(2) | ymmh_bit(2) | bit(RSP),
        "VEX-128 vxorps xmm2 must mark BOTH xmm2 and ymmh2"
    );
    assert_eq!(
        touched(CALLEE_VZEROUPPER),
        ALL_YMMH | bit(RSP),
        "vzeroupper touches ymmh0..15 only (no xmm, no OTHER)"
    );
    assert_eq!(
        touched(CALLEE_VZEROALL),
        ALL_XMM | ALL_YMMH | bit(RSP),
        "vzeroall touches xmm0..15 and ymmh0..15"
    );
    assert_eq!(
        touched(CALLEE_BLENDV),
        xmm_bit(0) | xmm_bit(3) | xmm_bit(4) | bit(RSP),
        "legacy blendvps implicitly reads xmm0"
    );
    assert_eq!(
        touched(CALLEE_VCOMISS),
        xmm_bit(0) | xmm_bit(1) | bit(RSP),
        "vcomiss is read-only on op0 so it does not touch ymmh0"
    );
    assert_eq!(
        touched(CALLEE_LDMXCSR),
        ALL,
        "ldmxcsr can trap at runtime and must conservatively mark ALL_STATE"
    );
    assert_eq!(
        touched(CALLEE_FXSAVE),
        ALL,
        "unmodelled fxsave must conservatively mark ALL_STATE"
    );
    assert_eq!(
        touched(CALLEE_SHA),
        ALL,
        "unsupported SIMD sha1rnds4 must conservatively mark ALL_STATE"
    );

    // 2. IR verification
    let ir = lift_module(functions.values()).expect("lifts").ir;

    // Disjoint SSE caller: ZERO %ymmh allocas, ZERO %state.ymmh*, ZERO <2 x i64> spills at call sites!
    let disjoint_ir = function_ir(&ir, CALLER_DISJOINT);
    assert!(
        !disjoint_ir.contains("%ymmh"),
        "pure SSE function must not allocate any %ymmh slot:\n{disjoint_ir}"
    );
    assert!(
        !disjoint_ir.contains("%state.ymmh"),
        "pure SSE function must not touch %state.ymmh:\n{disjoint_ir}"
    );
    let (before_xmm1, after_xmm1) = around_vec_call(&ir, CALLER_DISJOINT, CALLEE_XMM1);
    assert_eq!(
        state_xmms(&before_xmm1),
        Vec::<usize>::new(),
        "disjoint xmm1 callee needs zero xmm spills before call:\n{before_xmm1}"
    );
    assert_eq!(
        state_xmms(&after_xmm1),
        Vec::<usize>::new(),
        "disjoint xmm1 callee needs zero xmm reloads after call:\n{after_xmm1}"
    );
    let (before_gpr, after_gpr) = around_vec_call(&ir, CALLER_DISJOINT, CALLEE_PURE_GPR);
    assert_eq!(
        state_xmms(&before_gpr),
        Vec::<usize>::new(),
        "pure GPR callee needs zero xmm spills before call:\n{before_gpr}"
    );
    assert_eq!(
        state_xmms(&after_gpr),
        Vec::<usize>::new(),
        "pure GPR callee needs zero xmm reloads after call:\n{after_gpr}"
    );

    // VEX-128 caller: spills/reloads ONLY xmm2 + ymmh2 around CALLEE_VXORPS2 (NOT ymm0 or ymm3).
    let (before_vx, after_vx) = around_vec_call(&ir, CALLER_VXORPS2, CALLEE_VXORPS2);
    assert_eq!(state_xmms(&before_vx), vec![2], "before vxorps2:\n{before_vx}");
    assert_eq!(state_ymmhs(&before_vx), vec![2], "before vxorps2:\n{before_vx}");
    assert_eq!(state_xmms(&after_vx), vec![2], "after vxorps2:\n{after_vx}");
    assert_eq!(state_ymmhs(&after_vx), vec![2], "after vxorps2:\n{after_vx}");

    // vzeroupper caller: spills/reloads ONLY ymmh0 (NOT xmm0 or xmm1)!
    let vzu_ir = function_ir(&ir, CALLEE_VZEROUPPER);
    assert!(
        !vzu_ir.contains("%xmm"),
        "vzeroupper callee should not allocate %xmm slots:\n{vzu_ir}"
    );
    let (before_vzu, after_vzu) = around_vec_call(&ir, CALLER_VZEROUPPER, CALLEE_VZEROUPPER);
    assert_eq!(
        state_xmms(&before_vzu),
        Vec::<usize>::new(),
        "vzeroupper must not spill xmm slots:\n{before_vzu}"
    );
    assert_eq!(
        state_ymmhs(&before_vzu),
        vec![0],
        "vzeroupper must spill tracked ymmh0:\n{before_vzu}"
    );
    assert_eq!(
        state_xmms(&after_vzu),
        Vec::<usize>::new(),
        "vzeroupper must not reload xmm slots:\n{after_vzu}"
    );
    assert_eq!(
        state_ymmhs(&after_vzu),
        vec![0],
        "vzeroupper must reload tracked ymmh0:\n{after_vzu}"
    );
}

#[test]
fn r61_xmm_ymmh_vex128_and_vzeroupper_stay_bit_exact_wasm64() {
    if !exec_or_skip("r61_xmm_ymmh_vex128_and_vzeroupper_stay_bit_exact_wasm64") {
        return;
    }
    const XMM_STATE: u64 = STATE_ADDR + STATE_XMM_OFFSET;
    const YMMH_STATE: u64 = STATE_ADDR + STATE_YMM_HI_OFFSET;

    const DISJOINT_CALLER: u64 = CODE_BASE + 0x2000;
    const DISJOINT_CALLEE: u64 = CODE_BASE + 0x2100;
    const VXORPS_CALLER: u64 = CODE_BASE + 0x2200;
    const VXORPS_CALLEE: u64 = CODE_BASE + 0x2300;
    const VZU_CALLER: u64 = CODE_BASE + 0x2400;
    const VZU_CALLEE: u64 = CODE_BASE + 0x2500;

    let prog = Program::new()
        .asm(DISJOINT_CALLER, |a| {
            a.paddq(xmm0, xmm0)?;
            a.call(DISJOINT_CALLEE)?;
            a.paddq(xmm0, xmm0)?; // xmm0 *= 4; xmm1 untouched by caller, *= 2 by callee
            a.ret()
        })
        .asm(DISJOINT_CALLEE, |a| {
            a.paddq(xmm1, xmm1)?;
            a.ret()
        })
        // Caller doubles ymm0, ymm2, ymm3, calls VXORPS_CALLEE (which zeroes xmm2 + ymmh2),
        // then adds ymm2 into ymm3 (should add 0) and leaves ymm0/ymm2/ymm3 visible in state.
        .asm(VXORPS_CALLER, |a| {
            a.vpaddq(ymm0, ymm0, ymm0)?;
            a.vpaddq(ymm2, ymm2, ymm2)?;
            a.vpaddq(ymm3, ymm3, ymm3)?;
            a.call(VXORPS_CALLEE)?;
            a.vpaddq(ymm3, ymm3, ymm2)?;
            a.ret()
        })
        .asm(VXORPS_CALLEE, |a| {
            a.vxorps(xmm2, xmm2, xmm2)?;
            a.ret()
        })
        // Caller doubles ymm0 and xmm1, calls VZU_CALLEE (vzeroupper clears ymmh0..15),
        // then doubles xmm1 again and adds ymm0 to itself (upper half must be 0!).
        .asm(VZU_CALLER, |a| {
            a.vpaddq(ymm0, ymm0, ymm0)?;
            a.paddq(xmm1, xmm1)?;
            a.call(VZU_CALLEE)?;
            a.vpaddq(ymm0, ymm0, ymm0)?;
            a.paddq(xmm1, xmm1)?;
            a.ret()
        })
        .asm(VZU_CALLEE, |a| {
            a.vzeroupper()?;
            a.ret()
        });

    let functions = prog.discover();
    let built = build("r61_vec_calls", functions.values());

    let vec2 = |a: u64, b: u64| -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&a.to_le_bytes());
        out[8..].copy_from_slice(&b.to_le_bytes());
        out
    };

    // Case 0: disjoint xmm0 caller + xmm1 callee
    let mut c0 = Case::new(DISJOINT_CALLER);
    let mut xmm_in0 = vec![0u8; 32];
    xmm_in0[0..16].copy_from_slice(&vec2(10, 20));
    xmm_in0[16..32].copy_from_slice(&vec2(100, 200));
    let ymmh_sentinel = vec![0xA5u8; 32];
    c0.mem.push((XMM_STATE, xmm_in0));
    c0.mem.push((YMMH_STATE, ymmh_sentinel.clone()));
    c0.read.push((XMM_STATE, 32));
    c0.read.push((YMMH_STATE, 32));

    // Case 1: VEX-128 vxorps xmm2, xmm2, xmm2 across call zeroes BOTH xmm2 and ymmh2
    let mut c1 = Case::new(VXORPS_CALLER);
    let mut xmm_in1 = vec![0u8; 64];
    let mut ymmh_in1 = vec![0u8; 64];
    xmm_in1[0..16].copy_from_slice(&vec2(1, 2)); // xmm0
    ymmh_in1[0..16].copy_from_slice(&vec2(3, 4)); // ymmh0
    xmm_in1[32..48].copy_from_slice(&vec2(55, 66)); // xmm2
    ymmh_in1[32..48].copy_from_slice(&vec2(77, 88)); // ymmh2
    xmm_in1[48..64].copy_from_slice(&vec2(5, 6)); // xmm3
    ymmh_in1[48..64].copy_from_slice(&vec2(7, 8)); // ymmh3
    c1.mem.push((XMM_STATE, xmm_in1));
    c1.mem.push((YMMH_STATE, ymmh_in1));
    c1.read.push((XMM_STATE, 64));
    c1.read.push((YMMH_STATE, 64));

    // Case 2: vzeroupper across call clears ymmh0..15 while keeping xmm0 and xmm1 exact
    let mut c2 = Case::new(VZU_CALLER);
    let mut xmm_in2 = vec![0u8; 32];
    xmm_in2[0..16].copy_from_slice(&vec2(11, 22));
    xmm_in2[16..32].copy_from_slice(&vec2(33, 44));
    let ymmh_in2 = vec![0x5Au8; 256];
    c2.mem.push((XMM_STATE, xmm_in2));
    c2.mem.push((YMMH_STATE, ymmh_in2));
    c2.read.push((XMM_STATE, 32));
    c2.read.push((YMMH_STATE, 256));

    let out = run(&built, &[c0, c1, c2]);
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
    }

    // Verify Case 0
    assert_eq!(&out[0].mem[0][0..16], &vec2(40, 80), "xmm0 preserved and *4");
    assert_eq!(&out[0].mem[0][16..32], &vec2(200, 400), "xmm1 *2 by callee");
    assert_eq!(out[0].mem[1], ymmh_sentinel, "SSE-only leaves ymmh untouched");

    // Verify Case 1
    assert_eq!(&out[1].mem[0][0..16], &vec2(2, 4), "xmm0 preserved");
    assert_eq!(&out[1].mem[1][0..16], &vec2(6, 8), "ymmh0 preserved");
    assert_eq!(&out[1].mem[0][32..48], &vec2(0, 0), "vxorps zeroed xmm2");
    assert_eq!(&out[1].mem[1][32..48], &vec2(0, 0), "vxorps zeroed ymmh2");
    assert_eq!(&out[1].mem[0][48..64], &vec2(10, 12), "xmm3 + 0 == 2*xmm3");
    assert_eq!(&out[1].mem[1][48..64], &vec2(14, 16), "ymmh3 + 0 == 2*ymmh3");

    // Verify Case 2
    assert_eq!(&out[2].mem[0][0..16], &vec2(44, 88), "xmm0 *4 across vzeroupper");
    assert_eq!(&out[2].mem[0][16..32], &vec2(132, 176), "xmm1 *4 across vzeroupper");
    assert_eq!(out[2].mem[1], vec![0u8; 256], "vzeroupper cleared ymmh0..15");
}
