//! Tool-gated execution tests for the string, bit, atomic and miscellaneous integer
//! instructions: counter branches, stack frames and RFLAGS, the runtime-provided instructions,
//! large and overlapping `rep movs`/`rep stos`, AMD hints, multiply-high, a multi-threaded test of
//! the atomics on a real shared memory, and compilation of random code from these families.

use std::fmt::Write as _;

use iced_x86::code_asm::*;

use super::exec::{
    CODE_BASE, Case, DATA_ADDR, GUEST_BASE, HarnessMemory, MEMORY_BYTES, Outcome, R8, R9, R10, R11,
    RAX, RBP, RBX, RCX, RDI, RDX, RSI, RSP, STACK_TOP, STATE_ADDR, build, build_with_memory,
    exec_or_skip, run, run_script, shared_memory_link_args, shared_memory_or_skip,
};
use super::ir_integer::random_integer_program;
use super::{Program, asm_function};
use crate::lift::{STATE_FLAGS_OFFSET, STATE_FS_BASE_OFFSET, STATE_GPR_OFFSET};

const RUNNER_THREADS_JS: &str = include_str!("runner_threads.js");
const DF_ADDR: u64 = STATE_ADDR + STATE_FLAGS_OFFSET + 6;

fn run_clean(built: &super::exec::Built, cases: &[Case]) -> Vec<Outcome> {
    let outcomes = run(built, cases);
    for (i, o) in outcomes.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} trapped: {}", o.error);
        assert_eq!(o.regs[RSP], STACK_TOP, "case {i}: rsp not restored by ret");
    }
    outcomes
}

#[test]
fn counter_branches_execute() {
    if !exec_or_skip("counter_branches_execute") {
        return;
    }
    let base = CODE_BASE;
    let functions = Program::new()
        // eax = sum(1..=rcx) with loop, guarded by jrcxz (a 32-bit sum: LLVM rewrites a 64-bit
        // one into a closed form with an i128 multiply, i.e. a compiler-rt `__multi3` call)
        .asm(base, |a| {
            let mut top = a.create_label();
            let mut done = a.create_label();
            a.xor(eax, eax)?;
            a.jrcxz(done)?;
            a.set_label(&mut top)?;
            a.add(eax, ecx)?;
            a.loop_(top)?;
            a.set_label(&mut done)?;
            a.ret()
        })
        // mov eax, 1 ; jecxz +5 (67 E3: the 64-bit-mode encoding) ; mov eax, 2 ; ret
        .bytes(
            base + 0x100,
            &[0xB8, 1, 0, 0, 0, 0x67, 0xE3, 0x05, 0xB8, 2, 0, 0, 0, 0xC3],
        )
        .asm(base + 0x200, |a| {
            let mut zero = a.create_label();
            a.mov(eax, 1)?;
            a.jrcxz(zero)?;
            a.mov(eax, 2)?;
            a.set_label(&mut zero)?;
            a.ret()
        })
        // scan while the byte is zero (loope) / until it is 0x55 (loopne)
        .asm(base + 0x300, |a| {
            let mut top = a.create_label();
            a.mov(rax, -1i64)?;
            a.set_label(&mut top)?;
            a.inc(rax)?;
            a.cmp(byte_ptr(rdx + rax), 0)?;
            a.loope(top)?;
            a.ret()
        })
        .asm(base + 0x400, |a| {
            let mut top = a.create_label();
            a.mov(rax, -1i64)?;
            a.set_label(&mut top)?;
            a.inc(rax)?;
            a.cmp(byte_ptr(rdx + rax), 0x55)?;
            a.loopne(top)?;
            a.ret()
        })
        .discover();
    let built = build("counter_branches", functions.values());

    let sums = [0u64, 1, 5, 1000];
    let tests = [0u64, 1, 0x1_0000_0000, 0xFFFF_FFFF_0000_0000];
    let bytes = [0u8, 0, 0, 7, 0, 0x55, 0, 9, 0x55, 0];
    let counts = [1u64, 2, 3, 4, 5, 6, 7, 10];
    let mut cases = Vec::new();
    cases.extend(sums.iter().map(|&n| Case::new(base).reg(RCX, n)));
    for &v in &tests {
        cases.push(Case::new(base + 0x100).reg(RCX, v));
        cases.push(Case::new(base + 0x200).reg(RCX, v));
    }
    for va in [base + 0x300, base + 0x400] {
        for &n in &counts {
            cases.push(
                Case::new(va)
                    .reg(RCX, n)
                    .reg(RDX, DATA_ADDR)
                    .mem(DATA_ADDR, &bytes),
            );
        }
    }
    let o = run_clean(&built, &cases);
    for (i, &n) in sums.iter().enumerate() {
        assert_eq!(o[i].regs[RAX], n * (n + 1) / 2, "sum({n})");
        assert_eq!(o[i].regs[RCX], 0);
    }
    let o = &o[sums.len()..];
    for (i, &v) in tests.iter().enumerate() {
        let jecxz = o[2 * i].regs[RAX];
        let jrcxz = o[2 * i + 1].regs[RAX];
        assert_eq!(jecxz, if v as u32 == 0 { 1 } else { 2 }, "jecxz {v:#x}");
        assert_eq!(jrcxz, if v == 0 { 1 } else { 2 }, "jrcxz {v:#x}");
    }
    let o = &o[2 * tests.len()..];
    for (k, (target, while_equal)) in [(0u8, true), (0x55u8, false)].into_iter().enumerate() {
        for (i, &n) in counts.iter().enumerate() {
            // Reference model of loope/loopne.
            let (mut idx, mut left) = (u64::MAX, n);
            let zf = loop {
                idx = idx.wrapping_add(1);
                let zf = bytes[idx as usize] == target;
                left -= 1;
                if left == 0 || zf != while_equal {
                    break zf;
                }
            };
            let out = &o[k * counts.len() + i];
            assert_eq!(
                (out.regs[RAX], out.regs[RCX], out.flags[3]),
                (idx, left, zf),
                "loop{} with rcx={n}",
                if while_equal { "e" } else { "ne" }
            );
        }
    }
}

#[test]
fn stack_frames_and_rflags_execute() {
    if !exec_or_skip("stack_frames_and_rflags_execute") {
        return;
    }
    let base = CODE_BASE;
    let functions = Program::new()
        .asm(base, |a| {
            a.enter(0x20u32, 0u32)?;
            a.mov(qword_ptr(rbp - 8), rcx)?;
            a.mov(rax, qword_ptr(rbp - 8))?;
            a.mov(rdx, rsp)?;
            a.mov(r8, rbp)?;
            a.leave()?;
            a.ret()
        })
        .asm(base + 0x100, |a| {
            a.pushfq()?;
            a.pop(rax)?;
            a.ret()
        })
        .asm(base + 0x200, |a| {
            a.push(rcx)?;
            a.popfq()?;
            a.ret()
        })
        .asm(base + 0x300, |a| {
            a.std()?;
            a.pushfq()?;
            a.pop(rax)?;
            a.cld()?;
            a.pushfq()?;
            a.pop(rdx)?;
            a.ret()
        })
        .asm(base + 0x400, |a| {
            a.enter(0x10u32, 1u32)?;
            a.leave()?;
            a.ret()
        })
        .discover();
    let enter_level1 = functions[&(base + 0x400)].entry;
    let built = build("frames", functions.values());
    let rbp_in = 0x1234_5678_9ABC_DEF0u64;
    let flags = [true, false, true, true, false, true]; // CF AF ZF OF set
    let popped = 0x8D5u64 | (1 << 10) | (1 << 9) | (1 << 8); // CF PF AF ZF SF OF, DF, IF, TF
    let cases = [
        Case::new(base)
            .reg(RCX, 0xFEED)
            .reg(RBP, rbp_in)
            .read(STACK_TOP - 16, 8),
        {
            let mut c = Case::new(base + 0x100).mem(DF_ADDR, &[1]);
            c.flags = flags;
            c
        },
        Case::new(base + 0x200)
            .reg(RCX, popped)
            .mem(DF_ADDR, &[0])
            .read(DF_ADDR, 1),
        Case::new(base + 0x300).mem(DF_ADDR, &[0]).read(DF_ADDR, 1),
    ];
    let o = run_clean(&built, &cases);
    assert_eq!(o[0].regs[RAX], 0xFEED);
    assert_eq!(o[0].regs[R8], STACK_TOP - 16, "enter: rbp = frame");
    assert_eq!(
        o[0].regs[RDX],
        STACK_TOP - 16 - 0x20,
        "enter: rsp = frame - size"
    );
    assert_eq!(o[0].regs[RBP], rbp_in, "leave restored rbp");
    assert_eq!(o[0].mem[0], rbp_in.to_le_bytes(), "enter pushed rbp");
    // pushfq: CF(0) PF(2)=0 AF(4) ZF(6) SF(7)=0 OF(11), bit 1, IF, DF
    assert_eq!(
        o[1].regs[RAX],
        1 | 2 | (1 << 4) | (1 << 6) | (1 << 9) | (1 << 10) | (1 << 11)
    );
    // popfq: the six flags and DF come from the popped value; TF/IF are ignored.
    assert_eq!(o[2].flags, [true; 6]);
    assert_eq!(o[2].mem[0], [1], "DF from bit 10");
    assert_eq!(o[3].regs[RAX] >> 10 & 1, 1, "std visible to pushfq");
    assert_eq!(o[3].regs[RDX] >> 10 & 1, 0, "cld visible to pushfq");
    assert_eq!(o[3].mem[0], [0]);

    let trapped = run(&built, &[Case::new(base + 0x400)]);
    assert_eq!(
        trapped[0].error,
        format!("unimplemented instruction at {enter_level1:#x}"),
        "enter with a nesting level traps"
    );
}

#[test]
fn runtime_helper_instructions_execute() {
    if !exec_or_skip("runtime_helper_instructions_execute") {
        return;
    }
    let f = asm_function(CODE_BASE, |a| {
        a.mov(rbx, -1i64)?;
        a.mov(rcx, -1i64)?;
        a.mov(eax, 7)?;
        a.xor(ecx, ecx)?;
        a.cpuid()?;
        a.mov(r8, rax)?;
        a.mov(r9, rbx)?;
        a.mov(r10, rcx)?;
        a.mov(r11, rdx)?;
        a.rdtsc()?;
        a.shl(rdx, 32)?;
        a.or(rax, rdx)?;
        a.mov(rsi, rax)?;
        a.rdtscp()?;
        a.shl(rdx, 32)?;
        a.or(rax, rdx)?;
        a.mov(rdi, rax)?;
        a.mov(rbp, rcx)?;
        a.xor(ecx, ecx)?;
        a.xgetbv()?;
        a.ret()
    });
    let built = build("helpers", &[f]);
    let o = &run_clean(&built, &[Case::new(CODE_BASE)])[0];
    // runner.js stand-ins: cpuid -> leaf ^ 0x11111111, subleaf ^ 0x22222222, 0x33.., 0x44..
    assert_eq!(o.regs[R8], 7 ^ 0x1111_1111);
    assert_eq!(o.regs[R9], 0x2222_2222, "rbx zero-extended");
    assert_eq!(o.regs[R10], 0x3333_3333);
    assert_eq!(o.regs[R11], 0x4444_4444);
    assert!(
        o.regs[RDI] > o.regs[RSI] && o.regs[RSI] > 0,
        "tsc increases"
    );
    assert_eq!(o.regs[RBP], 7, "rdtscp: ecx = TSC_AUX");
    assert_eq!((o.regs[RAX], o.regs[RDX]), (7, 0), "xgetbv(0)");
}

/// Reference model of `rep movs` / `rep stos` on a flat byte array (element by element).
fn model_movs(mem: &mut [u8], base: u64, src: u64, dst: u64, n: u64, size: u64, df: bool) {
    for j in 0..n {
        let (s, d) = if df {
            (src - j * size, dst - j * size)
        } else {
            (src + j * size, dst + j * size)
        };
        let (s, d) = ((s - base) as usize, (d - base) as usize);
        let element: Vec<u8> = mem[s..s + size as usize].to_vec();
        mem[d..d + size as usize].copy_from_slice(&element);
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed) ^ (i >> 8) as u8)
        .collect()
}

#[test]
fn large_and_overlapping_rep_string_operations_execute() {
    if !exec_or_skip("large_and_overlapping_rep_string_operations_execute") {
        return;
    }
    let base = CODE_BASE;
    // fs: rep movsb (segment override on the source)
    let fs_movsb = [0x64, 0xF3, 0xA4, 0xC3];
    let functions = Program::new()
        .asm(base, |a| a.rep().movsb().and_then(|_| a.ret()))
        .asm(base + 0x100, |a| a.rep().movsq().and_then(|_| a.ret()))
        .asm(base + 0x200, |a| a.rep().stosb().and_then(|_| a.ret()))
        .asm(base + 0x300, |a| a.rep().stosq().and_then(|_| a.ret()))
        .asm(base + 0x400, |a| a.rep().stosd().and_then(|_| a.ret()))
        .bytes(base + 0x500, &fs_movsb)
        .discover();
    let built = build("big_strings", functions.values());

    let a_addr = 0x40_0000u64;
    let b_addr = 0x50_0000u64;
    let big = 100_000usize;
    let src = pattern(big, 3);
    let with_df = |c: Case, df: bool| c.mem(DF_ADDR, &[u8::from(df)]);
    // (case, region start, region length, initial bytes, model)
    type Model = Box<dyn Fn(&mut [u8], u64)>;
    let mut cases: Vec<(Case, u64, usize, Model, u64, u64)> = Vec::new();
    // forward, non-overlapping, 100k bytes (bulk path)
    let region = |len: usize| pattern(len, 9);
    cases.push((
        with_df(Case::new(base), false)
            .reg(RSI, a_addr)
            .reg(RDI, b_addr)
            .reg(RCX, big as u64)
            .mem(a_addr, &src),
        b_addr,
        big,
        Box::new(|_, _| {}),
        a_addr + big as u64,
        b_addr + big as u64,
    ));
    // forward overlap by one byte: replicates the first byte (element loop)
    for (gap, n, df, size, va) in [
        (1i64, 1000u64, false, 1u64, base),
        (3, 500, true, 1, base),
        (-5, 700, false, 1, base),
        (4, 300, false, 8, base + 0x100),
        (16, 300, true, 8, base + 0x100),
        (-8, 300, true, 8, base + 0x100),
    ] {
        let src_va = a_addr + 0x8000;
        let dst_va = (src_va as i64 + gap) as u64;
        let model: Model = Box::new(move |mem: &mut [u8], start: u64| {
            model_movs(mem, start, src_va, dst_va, n, size, df);
        });
        let delta = n * size;
        let (rsi_out, rdi_out) = if df {
            (src_va - delta, dst_va - delta)
        } else {
            (src_va + delta, dst_va + delta)
        };
        cases.push((
            with_df(Case::new(va), df)
                .reg(RSI, src_va)
                .reg(RDI, dst_va)
                .reg(RCX, n)
                .mem(a_addr, &region(0x10000)),
            a_addr,
            0x10000,
            model,
            rsi_out,
            rdi_out,
        ));
    }
    let mut outcomes_expected = Vec::new();
    let mut run_cases = Vec::new();
    for (case, start, len, model, rsi_out, rdi_out) in cases {
        let mut initial = vec![0u8; len];
        // The initial bytes of the observed region are whatever the case wrote there.
        for (addr, bytes) in &case.mem {
            if *addr <= start && start < *addr + bytes.len() as u64 {
                let off = (start - addr) as usize;
                let n = len.min(bytes.len() - off);
                initial[..n].copy_from_slice(&bytes[off..off + n]);
            }
        }
        if start == b_addr {
            initial = src.clone();
        } else {
            model(&mut initial, start);
        }
        outcomes_expected.push((initial, rsi_out, rdi_out));
        run_cases.push(case.read(start, len));
    }
    // stores: 100k `stosb`, 10k zero qwords, non-splat dwords downwards
    let c_addr = 0x60_0000u64;
    run_cases.push(
        with_df(Case::new(base + 0x200), false)
            .reg(RDI, c_addr)
            .reg(RCX, big as u64)
            .reg(RAX, 0x5A)
            .read(c_addr, big + 1),
    );
    run_cases.push(
        with_df(Case::new(base + 0x300), false)
            .reg(RDI, c_addr)
            .reg(RCX, 10_000)
            .reg(RAX, 0)
            .mem(c_addr, &vec![0xEE; 80_008])
            .read(c_addr, 80_008),
    );
    run_cases.push(
        with_df(Case::new(base + 0x400), true)
            .reg(RDI, c_addr + 4 * 999)
            .reg(RCX, 1000)
            .reg(RAX, 0x1234_5678)
            .mem(c_addr, &vec![0xEE; 4004])
            .read(c_addr, 4004),
    );
    // fs-relative source
    let fs_base = a_addr;
    run_cases.push(
        with_df(Case::new(base + 0x500), false)
            .reg(RSI, 0x10)
            .reg(RDI, b_addr)
            .reg(RCX, 64)
            .mem(a_addr, &src[..0x100])
            .mem(STATE_ADDR + STATE_FS_BASE_OFFSET, &fs_base.to_le_bytes())
            .read(b_addr, 64),
    );
    let o = run_clean(&built, &run_cases);
    for (i, (expected, rsi_out, rdi_out)) in outcomes_expected.iter().enumerate() {
        assert_eq!(o[i].regs[RCX], 0, "case {i}: rcx");
        assert_eq!(
            (o[i].regs[RSI], o[i].regs[RDI]),
            (*rsi_out, *rdi_out),
            "case {i}"
        );
        assert!(
            o[i].mem[0] == *expected,
            "case {i}: memory differs from the element model"
        );
    }
    let s = &o[outcomes_expected.len()..];
    assert!(s[0].mem[0][..big].iter().all(|&b| b == 0x5A));
    assert_eq!(s[0].mem[0][big], 0, "stosb stops after rcx bytes");
    assert_eq!(s[0].regs[RDI], c_addr + big as u64);
    assert!(s[1].mem[0][..80_000].iter().all(|&b| b == 0));
    assert_eq!(&s[1].mem[0][80_000..], &[0xEE; 8]);
    let dwords: Vec<u32> = s[2].mem[0][..4000]
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert!(dwords.iter().all(|&d| d == 0x1234_5678), "stosd downwards");
    assert_eq!(&s[2].mem[0][4000..], &[0xEE; 4]);
    assert_eq!(
        s[2].regs[RDI],
        c_addr - 4,
        "rdi after 1000 dwords downwards"
    );
    assert_eq!(s[3].mem[0], src[0x10..0x50], "fs: source");
    assert_eq!(s[3].regs[RSI], 0x10 + 64, "rsi excludes the segment base");
}

#[test]
fn amd_hints_clzero_and_multiply_high_execute() {
    if !exec_or_skip("amd_hints_clzero_and_multiply_high_execute") {
        return;
    }
    // clzero; monitorx; mwaitx; prefetchw [rax]; clflush [rax]; ret
    let hints = [
        0x0F, 0x01, 0xFC, 0x0F, 0x01, 0xFA, 0x0F, 0x01, 0xFB, 0x0F, 0x0D, 0x08, 0x0F, 0xAE, 0x38,
        0xC3,
    ];
    let functions = Program::new()
        .bytes(CODE_BASE, &hints)
        // unsigned division by 10 through the high half of a multiply only
        .asm(CODE_BASE + 0x100, |a| {
            a.mov(rax, 0xCCCC_CCCC_CCCC_CCCDu64)?;
            a.mul(rcx)?;
            a.shr(rdx, 3)?;
            a.mov(rax, rdx)?;
            a.ret()
        })
        // mulx with equal destinations keeps only the high half
        .asm(CODE_BASE + 0x200, |a| {
            a.mulx(r8, r8, rcx)?;
            a.ret()
        })
        .discover();
    let built = build("hints", functions.values());
    let line = DATA_ADDR + 0x40;
    let values = [0u64, 9, 10, 99, 12345, u64::MAX, 1 << 63];
    let mut cases = vec![
        Case::new(CODE_BASE)
            .reg(RAX, line + 7)
            .mem(DATA_ADDR, &[0xFF; 0x100])
            .read(DATA_ADDR, 0x100),
    ];
    for &v in &values {
        cases.push(Case::new(CODE_BASE + 0x100).reg(RCX, v));
        cases.push(
            Case::new(CODE_BASE + 0x200)
                .reg(RCX, v)
                .reg(RDX, 0xDEAD_BEEF_1234_5678),
        );
    }
    let o = run_clean(&built, &cases);
    let mut expected = vec![0xFFu8; 0x100];
    expected[0x40..0x80].fill(0);
    assert_eq!(o[0].mem[0], expected, "clzero zeroes the 64-byte line");
    for (i, &v) in values.iter().enumerate() {
        assert_eq!(o[1 + 2 * i].regs[RAX], v / 10, "{v} / 10");
        let hi = ((u128::from(v) * 0xDEAD_BEEF_1234_5678u128) >> 64) as u64;
        assert_eq!(o[2 + 2 * i].regs[R8], hi, "mulx high half of {v:#x}");
    }
}

#[test]
fn misaligned_cmpxchg16b_and_fences_execute() {
    let test = "misaligned_cmpxchg16b_and_fences_execute";
    if !exec_or_skip(test) || !shared_memory_or_skip(test) {
        return;
    }
    let f = asm_function(CODE_BASE, |a| {
        a.mfence()?;
        a.lfence()?;
        a.sfence()?;
        a.lock().cmpxchg16b(xmmword_ptr(rsi))?;
        a.ret()
    });
    let site = f.blocks[&CODE_BASE].instructions[3].ip();
    let built = build("cx16", &[f]);
    let o = run(
        &built,
        &[
            Case::new(CODE_BASE)
                .reg(RSI, DATA_ADDR)
                .reg(RAX, 1)
                .reg(RDX, 2)
                .reg(RBX, 3)
                .reg(RCX, 4)
                .mem(DATA_ADDR, &[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0])
                .read(DATA_ADDR, 16),
            Case::new(CODE_BASE).reg(RSI, DATA_ADDR + 8),
        ],
    );
    assert_eq!(o[0].error, "");
    assert!(o[0].flags[3], "ZF on success");
    assert_eq!(
        o[0].mem[0],
        [3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        o[1].error,
        format!("trap at {site:#x}"),
        "#GP on a misaligned operand"
    );
}

#[test]
fn atomics_have_no_lost_updates_across_threads() {
    atomics_across_threads(
        "atomics_have_no_lost_updates_across_threads",
        &HarnessMemory::from_env(),
    );
}

/// The threads test under a given memory model (R19 runs it with the counters and the guest
/// stacks inside a translated window: `exec_memory.rs`).
pub(super) fn atomics_across_threads(test: &str, memory: &HarnessMemory) {
    if !exec_or_skip(test) || !shared_memory_or_skip(test) {
        return;
    }
    let base = CODE_BASE;
    let functions = Program::new()
        // lock inc qword [rdx], rcx times
        .asm(base, |a| {
            let mut top = a.create_label();
            a.set_label(&mut top)?;
            a.lock().inc(qword_ptr(rdx))?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        // compare-exchange increment loop
        .asm(base + 0x100, |a| {
            let mut top = a.create_label();
            let mut retry = a.create_label();
            a.set_label(&mut top)?;
            a.mov(rax, qword_ptr(rdx))?;
            a.set_label(&mut retry)?;
            a.lea(r8, qword_ptr(rax + 1))?;
            a.lock().cmpxchg(qword_ptr(rdx), r8)?;
            a.jnz(retry)?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        // lock xadd dword [rdx], 1
        .asm(base + 0x200, |a| {
            let mut top = a.create_label();
            a.set_label(&mut top)?;
            a.mov(r8d, 1)?;
            a.lock().xadd(dword_ptr(rdx), r8d)?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        // 128-bit counter with lock cmpxchg16b (carries into the high half)
        .asm(base + 0x300, |a| {
            let mut top = a.create_label();
            let mut retry = a.create_label();
            a.set_label(&mut top)?;
            a.mov(rax, qword_ptr(rsi))?;
            a.mov(rdx, qword_ptr(rsi + 8))?;
            a.set_label(&mut retry)?;
            a.mov(rbx, rax)?;
            a.mov(rcx, rdx)?;
            a.add(rbx, 1)?;
            a.adc(rcx, 0)?;
            a.lock().cmpxchg16b(xmmword_ptr(rsi))?;
            a.jnz(retry)?;
            a.dec(r9)?;
            a.jnz(top)?;
            a.ret()
        })
        // spinlock: acquire with xchg, plain increment inside, release with lock btr
        .asm(base + 0x400, |a| {
            let mut acquire = a.create_label();
            a.set_label(&mut acquire)?;
            a.mov(eax, 1)?;
            a.xchg(dword_ptr(rdx), eax)?;
            a.test(eax, eax)?;
            a.jnz(acquire)?;
            a.inc(qword_ptr(rdx + 8))?;
            a.lock().btr(dword_ptr(rdx), 0)?;
            a.dec(rcx)?;
            a.jnz(acquire)?;
            a.ret()
        })
        // negative control: a plain (non-atomic) increment; the fence keeps LLVM from merging
        // the iterations, so concurrent workers lose updates
        .asm(base + 0x500, |a| {
            let mut top = a.create_label();
            a.set_label(&mut top)?;
            a.inc(qword_ptr(rdx))?;
            a.mfence()?;
            a.dec(rcx)?;
            a.jnz(top)?;
            a.ret()
        })
        .discover();
    let mut extra = vec!["--import-memory".to_string()];
    extra.extend(shared_memory_link_args());
    extra.push("--export-if-defined=__stack_pointer".to_string());
    let built = build_with_memory("threads", functions.values(), &extra, memory, GUEST_BASE);

    const WORKERS: u64 = 4;
    // The lifted loops are short, so on a loaded or small machine one worker can finish before
    // the next is even scheduled. Each attempt is a fresh run with four times the iterations,
    // until the non-atomic control proves that calls really interleaved (it lost updates) or the
    // attempts run out.
    const ATTEMPTS: [u64; 4] = [100_000, 400_000, 1_600_000, 3_200_000];
    let counters = DATA_ADDR;
    let (inc, cas, xadd, wide, spin, plain) = (
        counters,
        counters + 16,
        counters + 32,
        counters + 48,
        counters + 64,
        counters + 80,
    );
    let wide_start: u128 = u128::from(u64::MAX - 1000) | (5u128 << 64);
    let init: String = wide_start
        .to_le_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut last_note = String::new();
    for iterations in ATTEMPTS {
        let mut workers = String::new();
        for w in 0..WORKERS {
            if w > 0 {
                workers.push(',');
            }
            let calls = [
                (base, vec![(RCX, iterations), (RDX, inc)]),
                (base + 0x100, vec![(RCX, iterations), (RDX, cas)]),
                (base + 0x200, vec![(RCX, iterations), (RDX, xadd)]),
                (base + 0x300, vec![(R9, iterations), (RSI, wide)]),
                (base + 0x400, vec![(RCX, iterations), (RDX, spin)]),
                (base + 0x500, vec![(RCX, iterations), (RDX, plain)]),
            ];
            let calls: Vec<String> = calls
                .iter()
                .map(|(va, regs)| {
                    let regs: Vec<String> = regs
                        .iter()
                        .map(|(i, v)| format!("[{i},\"{v:#x}\"]"))
                        .collect();
                    format!("{{\"fn\":\"sub_{va:X}\",\"regs\":[{}]}}", regs.join(","))
                })
                .collect();
            let _ = write!(
                workers,
                "{{\"state\":\"{:#x}\",\"stackTop\":\"{:#x}\",\"shadowStack\":\"{:#x}\",\"calls\":[{}]}}",
                STATE_ADDR + w * 0x200,
                STACK_TOP - w * 0x1_0000,
                0x70_0000 + (w + 1) * 0x1_0000,
                calls.join(",")
            );
        }
        let spec = format!(
            "{{\"pages\":{},\"gprOffset\":{STATE_GPR_OFFSET},\"guestBase\":\"{:#x}\",{}\"init\":[[\"{wide:#x}\",\"{init}\"]],\"read\":[[\"{counters:#x}\",88]],\"workers\":[{workers}]}}",
            MEMORY_BYTES / 65536,
            built.guest_base,
            if built.memory_json.is_empty() {
                String::new()
            } else {
                format!("\"memory\":{},", built.memory_json)
            }
        );
        let out = run_script(&built, "runner_threads.js", RUNNER_THREADS_JS, &[spec]);
        if let Some(reason) = out.trim().strip_prefix("skip: ") {
            println!("SKIP {test}: {reason}");
            return;
        }
        println!("{out}");
        let first = out.lines().next().unwrap_or_default().trim();
        assert!(!first.starts_with("error"), "{first}");
        let bytes: Vec<u8> = (0..first.len() / 2)
            .map(|i| u8::from_str_radix(&first[2 * i..2 * i + 2], 16).expect("hex"))
            .collect();
        let u64_at = |off: usize| u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
        // The no-lost-updates checks are strict on every attempt, whether or not the threads
        // overlapped: a lost update is a bug either way.
        let total = WORKERS * iterations;
        assert_eq!(u64_at(0), total, "lock inc");
        assert_eq!(u64_at(16), total, "lock cmpxchg loop");
        assert_eq!(u64_at(32) & 0xFFFF_FFFF, total, "lock xadd");
        let wide_out = u128::from_le_bytes(bytes[48..64].try_into().unwrap());
        assert_eq!(wide_out, wide_start + u128::from(total), "lock cmpxchg16b");
        assert_eq!(u64_at(64), 0, "spinlock released");
        assert_eq!(
            u64_at(72),
            total,
            "xchg/btr spinlock protects a plain increment"
        );
        // Peak number of calls in flight at once, per call (see runner_threads.js).
        let peaks: Vec<u64> = out
            .lines()
            .filter_map(|l| l.strip_prefix("overlap "))
            .filter_map(|l| l.split(' ').nth(1)?.parse().ok())
            .collect();
        let survived = u64_at(80);
        println!(
            "non-atomic control: {survived} of {total} increments survived; peak concurrent calls {peaks:?}"
        );
        // The mutation check: a plain increment loses updates exactly when calls interleave, so
        // a lost update in the control is proof that the atomic loops above ran under real
        // contention and the harness can see a lost update. Observed overlap alone is not
        // enough (a late worker may just catch the tail of an early one's call).
        if survived < total {
            assert!(
                peaks[5] >= 2,
                "the control lost updates, so calls must have overlapped: {out}"
            );
            println!(
                "OVERLAP OBSERVED {test}: control lost {} updates",
                total - survived
            );
            return;
        }
        last_note = format!("peak concurrent calls {peaks:?}");
    }
    println!(
        "SKIP {test}: the workers never interleaved (the non-atomic control lost no update in \
         any attempt up to {} iterations; {last_note}), so this machine offers no real \
         concurrency; the atomic counters were still exact",
        ATTEMPTS[ATTEMPTS.len() - 1]
    );
}

#[test]
fn lifted_random_integer_code_compiles_and_links() {
    let test = "lifted_random_integer_code_compiles_and_links";
    if !exec_or_skip(test) || !shared_memory_or_skip(test) {
        return;
    }
    for seed in [0x1A7E, 0xBEEF] {
        let functions = random_integer_program(150, seed);
        let _ = build("random_integer", functions.values());
    }
}

/// R60: 100% bit-exact Wasm64 execution tests for all optimized scalar ALU, bit-manipulation,
/// rotate, double-shift, bswap, 32/64-bit division, and fused `CMP`/`TEST`/`SUB`/`DEC`/`INC`/`NEG`
/// -> all 16 `Jcc`/`SETcc`/`CMOVcc` conditions including `INT_MIN`/`INT_MAX` signed overflow
/// boundaries, zero inputs, count-0 flag preservation, and windowed-burst boundaries.
#[test]
fn r60_scalar_bitmanip_div_and_all_16_fused_predicates_are_bit_exact() {
    let test = "r60_scalar_bitmanip_div_and_all_16_fused_predicates_are_bit_exact";
    if !exec_or_skip(test) {
        return;
    }

    const CMP_ALL16: u64 = CODE_BASE;
    const TEST_ALL16: u64 = CODE_BASE + 0x1000;
    const SUB_DEC_INC_NEG: u64 = CODE_BASE + 0x2000;
    const BIT_EDGES: u64 = CODE_BASE + 0x2400;
    const ROL_ROR_EDGES: u64 = CODE_BASE + 0x2800;
    const SHLD_SHRD_EDGES: u64 = CODE_BASE + 0x2C00;
    const DIV32_OK: u64 = CODE_BASE + 0x3000;
    const IDIV32_OK: u64 = CODE_BASE + 0x3100;
    const IDIV64_OK: u64 = CODE_BASE + 0x3200;
    const BURST_CMP_JLE: u64 = CODE_BASE + 0x3300;

    // Emit a function that packs all 16 condition codes (bit 0..15) after `cmp rcx, rdx` via
    // BOTH `setcc` (low 16 bits of rax), `cmovcc` (bits 16..31), and `jcc` (bits 32..47), then
    // overwrites flags with `xor r8d, r8d` so fusion-aware dead-flag elision activates!
    let emit_all16 = |a: &mut CodeAssembler,
                      setup: fn(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>|
     -> Result<(), iced_x86::IcedError> {
        a.xor(eax, eax)?;
        a.mov(r9, 1u64)?;
        // 1) SETcc pack into bits 0..15 (single producer before all 16 setcc!)
        setup(a)?;
        // Note: `setcc` does not modify flags, and `shl`/`or` would, so store each byte to stack
        // (mov is flag-transparent!) and pack afterwards so all 16 setcc fuse with ONE producer!
        a.sub(rsp, 32)?;
        setup(a)?;
        a.seto(byte_ptr(rsp))?;
        a.setno(byte_ptr(rsp + 1))?;
        a.setb(byte_ptr(rsp + 2))?;
        a.setae(byte_ptr(rsp + 3))?;
        a.sete(byte_ptr(rsp + 4))?;
        a.setne(byte_ptr(rsp + 5))?;
        a.setbe(byte_ptr(rsp + 6))?;
        a.seta(byte_ptr(rsp + 7))?;
        a.sets(byte_ptr(rsp + 8))?;
        a.setns(byte_ptr(rsp + 9))?;
        a.setp(byte_ptr(rsp + 10))?;
        a.setnp(byte_ptr(rsp + 11))?;
        a.setl(byte_ptr(rsp + 12))?;
        a.setge(byte_ptr(rsp + 13))?;
        a.setle(byte_ptr(rsp + 14))?;
        a.setg(byte_ptr(rsp + 15))?;
        // 2) CMOVcc pack into bits 16..31 (one producer per cmovcc, flags overwritten right after)
        let r10_acc = |a: &mut CodeAssembler,
                       bit: u32,
                       cmov: fn(&mut CodeAssembler) -> Result<(), iced_x86::IcedError>|
         -> Result<(), iced_x86::IcedError> {
            a.mov(r10, 0u64)?;
            setup(a)?;
            cmov(a)?;
            a.shl(r10, bit)?;
            a.or(rax, r10)?;
            Ok(())
        };
        r10_acc(a, 16, |a| a.cmovo(r10, r9))?;
        r10_acc(a, 17, |a| a.cmovno(r10, r9))?;
        r10_acc(a, 18, |a| a.cmovb(r10, r9))?;
        r10_acc(a, 19, |a| a.cmovae(r10, r9))?;
        r10_acc(a, 20, |a| a.cmove(r10, r9))?;
        r10_acc(a, 21, |a| a.cmovne(r10, r9))?;
        r10_acc(a, 22, |a| a.cmovbe(r10, r9))?;
        r10_acc(a, 23, |a| a.cmova(r10, r9))?;
        r10_acc(a, 24, |a| a.cmovs(r10, r9))?;
        r10_acc(a, 25, |a| a.cmovns(r10, r9))?;
        r10_acc(a, 26, |a| a.cmovp(r10, r9))?;
        r10_acc(a, 27, |a| a.cmovnp(r10, r9))?;
        r10_acc(a, 28, |a| a.cmovl(r10, r9))?;
        r10_acc(a, 29, |a| a.cmovge(r10, r9))?;
        r10_acc(a, 30, |a| a.cmovle(r10, r9))?;
        r10_acc(a, 31, |a| a.cmovg(r10, r9))?;
        // 3) Jcc pack into bits 32..47 (dead flags after each jcc!)
        let jcc_bit = |a: &mut CodeAssembler,
                       bit: u32,
                       emit_j: fn(
            &mut CodeAssembler,
            CodeLabel,
        ) -> Result<(), iced_x86::IcedError>|
         -> Result<(), iced_x86::IcedError> {
            let mut taken = a.create_label();
            let mut next = a.create_label();
            setup(a)?;
            emit_j(a, taken)?;
            a.xor(r8d, r8d)?;
            a.jmp(next)?;
            a.set_label(&mut taken)?;
            a.mov(r10, 1u64 << bit)?;
            a.or(rax, r10)?;
            a.set_label(&mut next)?;
            Ok(())
        };
        jcc_bit(a, 32, |a, l| a.jo(l))?;
        jcc_bit(a, 33, |a, l| a.jno(l))?;
        jcc_bit(a, 34, |a, l| a.jb(l))?;
        jcc_bit(a, 35, |a, l| a.jae(l))?;
        jcc_bit(a, 36, |a, l| a.je(l))?;
        jcc_bit(a, 37, |a, l| a.jne(l))?;
        jcc_bit(a, 38, |a, l| a.jbe(l))?;
        jcc_bit(a, 39, |a, l| a.ja(l))?;
        jcc_bit(a, 40, |a, l| a.js(l))?;
        jcc_bit(a, 41, |a, l| a.jns(l))?;
        jcc_bit(a, 42, |a, l| a.jp(l))?;
        jcc_bit(a, 43, |a, l| a.jnp(l))?;
        jcc_bit(a, 44, |a, l| a.jl(l))?;
        jcc_bit(a, 45, |a, l| a.jge(l))?;
        jcc_bit(a, 46, |a, l| a.jle(l))?;
        jcc_bit(a, 47, |a, l| a.jg(l))?;
        // Pack the 16 `setcc` bytes from [rsp..rsp+16] into bits 0..15 of rax.
        for i in 0..16u32 {
            a.movzx(r10d, byte_ptr(rsp + i as i32))?;
            a.shl(r10, i)?;
            a.or(rax, r10)?;
        }
        a.add(rsp, 32)?;
        a.ret()
    };

    let program = Program::new()
        .asm(CMP_ALL16, |a| emit_all16(a, |a| a.cmp(rcx, rdx)))
        .asm(TEST_ALL16, |a| emit_all16(a, |a| a.test(rcx, rdx)))
        // Verify `dec`, `inc`, `neg` signed predicates at INT_MIN/INT_MAX/0/1/-1.
        .asm(SUB_DEC_INC_NEG, |a| {
            a.xor(eax, eax)?;
            // bit 0..3: dec rcx -> e, l, le, g
            a.mov(r8, rcx)?;
            a.dec(r8)?;
            a.sete(r9b)?;
            a.setl(r10b)?;
            a.setle(r11b)?;
            a.setg(dl)?;
            a.movzx(r9d, r9b)?;
            a.movzx(r10d, r10b)?;
            a.movzx(r11d, r11b)?;
            a.movzx(edx, dl)?;
            a.shl(r10, 1)?;
            a.shl(r11, 2)?;
            a.shl(rdx, 3)?;
            a.or(rax, r9)?;
            a.or(rax, r10)?;
            a.or(rax, r11)?;
            a.or(rax, rdx)?;
            // bit 4..7: inc rcx -> e, l, le, g
            a.mov(r8, rcx)?;
            a.inc(r8)?;
            a.sete(r9b)?;
            a.setl(r10b)?;
            a.setle(r11b)?;
            a.setg(dl)?;
            a.movzx(r9d, r9b)?;
            a.movzx(r10d, r10b)?;
            a.movzx(r11d, r11b)?;
            a.movzx(edx, dl)?;
            a.shl(r9, 4)?;
            a.shl(r10, 5)?;
            a.shl(r11, 6)?;
            a.shl(rdx, 7)?;
            a.or(rax, r9)?;
            a.or(rax, r10)?;
            a.or(rax, r11)?;
            a.or(rax, rdx)?;
            // bit 8..11: neg rcx -> b, l, le, g
            a.mov(r8, rcx)?;
            a.neg(r8)?;
            a.setb(r9b)?;
            a.setl(r10b)?;
            a.setle(r11b)?;
            a.setg(dl)?;
            a.movzx(r9d, r9b)?;
            a.movzx(r10d, r10b)?;
            a.movzx(r11d, r11b)?;
            a.movzx(edx, dl)?;
            a.shl(r9, 8)?;
            a.shl(r10, 9)?;
            a.shl(r11, 10)?;
            a.shl(rdx, 11)?;
            a.or(rax, r9)?;
            a.or(rax, r10)?;
            a.or(rax, r11)?;
            a.or(rax, rdx)?;
            a.ret()
        })
        // BSF(0), BSR(0), TZCNT(0), LZCNT(0), POPCNT, BSWAP 16/32/64.
        // Input: rcx = src; sentinel 0xDEAD_BEEF_1234_5678 in r8/r9 before 32-bit bsf/bsr.
        .asm(BIT_EDGES, |a| {
            a.mov(r8, 0xDEAD_BEEF_1234_5678u64)?;
            a.mov(r9, 0xDEAD_BEEF_1234_5678u64)?;
            a.bsf(r8d, ecx)?; // zero input must preserve full 64-bit r8!
            a.sete(al)?;
            a.bsr(r9d, ecx)?; // zero input must preserve full 64-bit r9!
            a.sete(dl)?;
            a.tzcnt(r10, rcx)?;
            a.setb(r11b)?;
            a.lzcnt(rbx, rcx)?;
            a.popcnt(rsi, rcx)?;
            a.mov(rdi, rcx)?;
            a.bswap(rdi)?;
            a.movzx(eax, al)?;
            a.movzx(edx, dl)?;
            a.movzx(r11d, r11b)?;
            a.shl(rdx, 1)?;
            a.shl(r11, 2)?;
            a.or(rax, rdx)?;
            a.or(rax, r11)?;
            a.ret()
        })
        // ROL/ROR by CL (tests 0, 1, 31, 32, 63, 64) preserving CF/OF on count == 0.
        .asm(ROL_ROR_EDGES, |a| {
            // rdx = val, cl = count; pre-set CF=1 via stc so count==0 keeps CF=1!
            a.stc()?;
            a.mov(eax, edx)?;
            a.rol(eax, cl)?;
            a.setb(r8b)?; // CF after 32-bit rol
            a.stc()?;
            a.mov(r9, rdx)?;
            a.ror(r9, cl)?;
            a.setb(r10b)?; // CF after 64-bit ror
            a.movzx(r8d, r8b)?;
            a.movzx(r10d, r10b)?;
            a.ret()
        })
        // SHLD/SHRD by CL (0, 1, 15, 16, 31, 32, 63, 64) + imm counts.
        .asm(SHLD_SHRD_EDGES, |a| {
            // rdx = dst, r8 = src, cl = count
            a.stc()?;
            a.mov(eax, edx)?;
            a.shld(eax, r8d, cl)?;
            a.setb(r9b)?;
            a.stc()?;
            a.mov(r10, rdx)?;
            a.shrd(r10, r8, cl)?;
            a.setb(r11b)?;
            a.movzx(r9d, r9b)?;
            a.movzx(r11d, r11b)?;
            a.ret()
        })
        .asm(DIV32_OK, |a| {
            a.div(ecx)?;
            a.ret()
        })
        .asm(IDIV32_OK, |a| {
            a.idiv(ecx)?;
            a.ret()
        })
        .asm(IDIV64_OK, |a| {
            a.idiv(rcx)?;
            a.ret()
        })
        // Windowed-burst soundness test: `[rdx]` and `[rdx+8]` form a coalesced burst ending in
        // `cmp`, followed by `jle` AFTER the burst join block!
        .asm(BURST_CMP_JLE, |a| {
            let mut le_lbl = a.create_label();
            a.mov(rax, qword_ptr(rdx))?;
            a.cmp(rax, qword_ptr(rdx + 8))?;
            a.jle(le_lbl)?;
            a.mov(eax, 1)?;
            a.ret()?;
            a.set_label(&mut le_lbl)?;
            a.mov(eax, 2)?;
            a.ret()
        });

    let functions = program.discover();
    let built = build(test, functions.values());

    // 1. All 16 CCs for CMP & TEST across signed/unsigned overflow boundaries.
    let vals: [u64; 9] = [
        0,
        1,
        u64::MAX,
        0x7FFF_FFFF,
        0x8000_0000,
        0xFFFF_FFFF,
        i64::MAX as u64,
        i64::MIN as u64,
        0x1234_5678_9ABC_DEF0,
    ];
    let mut cases = Vec::new();
    let mut cmp_expected = Vec::new();
    let mut test_expected = Vec::new();
    for &a in &vals {
        for &b in &vals {
            let d = a.wrapping_sub(b);
            let cf = a < b;
            let zf = d == 0;
            let sf = (d as i64) < 0;
            let of = (((a ^ b) & (a ^ d)) as i64) < 0;
            let pf = (d as u8).count_ones().is_multiple_of(2);
            let flags = [
                of,
                !of,
                cf,
                !cf,
                zf,
                !zf,
                cf || zf,
                !(cf || zf),
                sf,
                !sf,
                pf,
                !pf,
                sf ^ of,
                !(sf ^ of),
                zf || (sf ^ of),
                !(zf || (sf ^ of)),
            ];
            let mask16: u64 = flags
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &f)| acc | (u64::from(f) << i));
            let pack = mask16 | (mask16 << 16) | (mask16 << 32);
            cases.push(Case::new(CMP_ALL16).reg(RCX, a).reg(RDX, b));
            cmp_expected.push((a, b, pack));

            let t = a & b;
            let tzf = t == 0;
            let tsf = (t as i64) < 0;
            let tpf = (t as u8).count_ones().is_multiple_of(2);
            let tflags = [
                false,
                true,
                false,
                true,
                tzf,
                !tzf,
                tzf,
                !tzf,
                tsf,
                !tsf,
                tpf,
                !tpf,
                tsf,
                !tsf,
                tzf || tsf,
                !(tzf || tsf),
            ];
            let tmask16: u64 = tflags
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &f)| acc | (u64::from(f) << i));
            let tpack = tmask16 | (tmask16 << 16) | (tmask16 << 32);
            cases.push(Case::new(TEST_ALL16).reg(RCX, a).reg(RDX, b));
            test_expected.push((a, b, tpack));
        }
    }
    let out = run(&built, &cases);
    for (idx, ((ca, cb, cwant), (ta, tb, twant))) in
        cmp_expected.iter().zip(&test_expected).enumerate()
    {
        let co = &out[idx * 2];
        assert!(co.error.is_empty(), "cmp({ca:#x},{cb:#x}): {}", co.error);
        assert_eq!(
            co.regs[RAX], *cwant,
            "cmp({ca:#x},{cb:#x}) all-16 pack mismatch: got {:#x}, want {:#x}",
            co.regs[RAX], cwant
        );
        let to = &out[idx * 2 + 1];
        assert!(to.error.is_empty(), "test({ta:#x},{tb:#x}): {}", to.error);
        assert_eq!(
            to.regs[RAX], *twant,
            "test({ta:#x},{tb:#x}) all-16 pack mismatch: got {:#x}, want {:#x}",
            to.regs[RAX], twant
        );
    }

    // 2. DEC / INC / NEG signed predicates including INT_MIN / INT_MAX / 0 / 1 / -1.
    let unary_vals: [i64; 7] = [0, 1, -1, 2, -2, i64::MIN, i64::MAX];
    let unary_cases: Vec<Case> = unary_vals
        .iter()
        .map(|&v| Case::new(SUB_DEC_INC_NEG).reg(RCX, v as u64))
        .collect();
    let unary_out = run(&built, &unary_cases);
    for (&v, o) in unary_vals.iter().zip(&unary_out) {
        assert!(o.error.is_empty(), "unary({v}): {}", o.error);
        let mut want = 0u64;
        // dec: v - 1
        want |= u64::from(v == 1);
        want |= u64::from(v < 1) << 1;
        want |= u64::from(v <= 1) << 2;
        want |= u64::from(v > 1) << 3;
        // inc: v + 1 (same ZF/SF/OF as v - (-1))
        want |= u64::from(v == -1) << 4;
        want |= u64::from(v < -1) << 5;
        want |= u64::from(v <= -1) << 6;
        want |= u64::from(v > -1) << 7;
        // neg: 0 - v
        want |= u64::from(v != 0) << 8;
        want |= u64::from(0 < v) << 9;
        want |= u64::from(0 <= v) << 10;
        want |= u64::from(0 > v) << 11;
        assert_eq!(o.regs[RAX], want, "dec/inc/neg({v:#x})");
    }

    // 3. BSF(0), BSR(0), TZCNT(0), LZCNT(0), POPCNT, BSWAP.
    let bit_inputs = [0u64, 1, 0x8000_0000, 0x8000_0000_0000_0000, 0x0123_4567_89AB_CDEF];
    let bit_cases: Vec<Case> = bit_inputs
        .iter()
        .map(|&v| Case::new(BIT_EDGES).reg(RCX, v))
        .collect();
    let bit_out = run(&built, &bit_cases);
    for (&v, o) in bit_inputs.iter().zip(&bit_out) {
        assert!(o.error.is_empty(), "bit_edges({v:#x}): {}", o.error);
        let v32 = v as u32;
        let want_r8 = if v32 == 0 {
            0xDEAD_BEEF_1234_5678
        } else {
            u64::from(v32.trailing_zeros())
        };
        let want_r9 = if v32 == 0 {
            0xDEAD_BEEF_1234_5678
        } else {
            u64::from(31 - v32.leading_zeros())
        };
        let want_flags =
            u64::from(v32 == 0) | (u64::from(v32 == 0) << 1) | (u64::from(v == 0) << 2);
        assert_eq!(o.regs[R8], want_r8, "bsf32({v:#x})");
        assert_eq!(o.regs[R9], want_r9, "bsr32({v:#x})");
        assert_eq!(o.regs[R10], u64::from(v.trailing_zeros()), "tzcnt64({v:#x})");
        assert_eq!(o.regs[RBX], u64::from(v.leading_zeros()), "lzcnt64({v:#x})");
        assert_eq!(o.regs[RSI], u64::from(v.count_ones()), "popcnt64({v:#x})");
        assert_eq!(o.regs[RDI], v.swap_bytes(), "bswap64({v:#x})");
        assert_eq!(o.regs[RAX], want_flags, "bit_edges flags({v:#x})");
    }

    // 4. ROL/ROR by 0, 1, 31, 32, 63, 64 (verifying count-0 CF preservation!).
    let rot_val = 0x8123_4567_89AB_CDE1u64;
    let rot_counts = [0u64, 1, 31, 32, 63, 64];
    let rot_cases: Vec<Case> = rot_counts
        .iter()
        .map(|&c| Case::new(ROL_ROR_EDGES).reg(RDX, rot_val).reg(RCX, c))
        .collect();
    let rot_out = run(&built, &rot_cases);
    for (&c, o) in rot_counts.iter().zip(&rot_out) {
        assert!(o.error.is_empty(), "rot({c}): {}", o.error);
        let c32 = (c & 31) as u32;
        let r32 = (rot_val as u32).rotate_left(c32);
        let cf32 = if c32 == 0 { 1 } else { u64::from(r32 & 1) };
        let c64 = (c & 63) as u32;
        let r64 = rot_val.rotate_right(c64);
        let cf64 = if c64 == 0 { 1 } else { r64 >> 63 };
        assert_eq!(o.regs[RAX], u64::from(r32), "rol32({c})");
        assert_eq!(o.regs[R8], cf32, "rol32 cf({c})");
        assert_eq!(o.regs[R9], r64, "ror64({c})");
        assert_eq!(o.regs[R10], cf64, "ror64 cf({c})");
    }

    // 5. SHLD/SHRD by 0, 1, 15, 16, 31, 32, 63, 64.
    let sh_dst = 0xF123_4567_89AB_CDE1u64;
    let sh_src = 0x0FED_CBA9_7654_3210u64;
    let sh_counts = [0u64, 1, 15, 16, 31, 32, 63, 64];
    let sh_cases: Vec<Case> = sh_counts
        .iter()
        .map(|&c| {
            Case::new(SHLD_SHRD_EDGES)
                .reg(RDX, sh_dst)
                .reg(R8, sh_src)
                .reg(RCX, c)
        })
        .collect();
    let sh_out = run(&built, &sh_cases);
    for (&c, o) in sh_counts.iter().zip(&sh_out) {
        assert!(o.error.is_empty(), "shld/shrd({c}): {}", o.error);
        let c32 = (c & 31) as u32;
        let d32 = sh_dst as u32;
        let s32 = sh_src as u32;
        let (want_l32, want_cf32) = if c32 == 0 {
            (d32, 1u64)
        } else {
            let r = (d32 << c32) | (s32 >> (32 - c32));
            let cf = u64::from((d32 >> (32 - c32)) & 1);
            (r, cf)
        };
        let c64 = (c & 63) as u32;
        let (want_r64, want_cf64) = if c64 == 0 {
            (sh_dst, 1u64)
        } else {
            let r = (sh_dst >> c64) | (sh_src << (64 - c64));
            let cf = (sh_dst >> (c64 - 1)) & 1;
            (r, cf)
        };
        assert_eq!(o.regs[RAX], u64::from(want_l32), "shld32({c})");
        assert_eq!(o.regs[R9], want_cf32, "shld32 cf({c})");
        assert_eq!(o.regs[R10], want_r64, "shrd64({c})");
        assert_eq!(o.regs[R11], want_cf64, "shrd64 cf({c})");
    }

    // 6. Native 32-bit `div`/`idiv` & fast-path 64-bit `idiv`, including `#DE` traps.
    let div_cases = vec![
        // div32: (0x0000_0001 : 0x0000_0000) / 2 = 0x8000_0000 rem 0
        Case::new(DIV32_OK).reg(RDX, 1).reg(RAX, 0).reg(RCX, 2),
        // div32: #DE overflow (hi >= d)
        Case::new(DIV32_OK).reg(RDX, 2).reg(RAX, 0).reg(RCX, 2),
        // div32: #DE divide by zero
        Case::new(DIV32_OK).reg(RDX, 0).reg(RAX, 5).reg(RCX, 0),
        // idiv32: -100 / 7 = -14 rem -2
        Case::new(IDIV32_OK)
            .reg(RDX, 0xFFFF_FFFF)
            .reg(RAX, (-100i32) as u32 as u64)
            .reg(RCX, 7),
        // idiv32: #DE i32::MIN / -1
        Case::new(IDIV32_OK)
            .reg(RDX, 0xFFFF_FFFF)
            .reg(RAX, 0x8000_0000)
            .reg(RCX, 0xFFFF_FFFF),
        // idiv32: #DE i64::MIN / -1 (EDX = 0x8000_0000, EAX = 0, ECX = -1)
        Case::new(IDIV32_OK)
            .reg(RDX, 0x8000_0000)
            .reg(RAX, 0)
            .reg(RCX, 0xFFFF_FFFF),
        // idiv64 fast path: -100 / 7 = -14 rem -2
        Case::new(IDIV64_OK)
            .reg(RDX, u64::MAX)
            .reg(RAX, (-100i64) as u64)
            .reg(RCX, 7),
        // idiv64 #DE: i64::MIN / -1
        Case::new(IDIV64_OK)
            .reg(RDX, u64::MAX)
            .reg(RAX, i64::MIN as u64)
            .reg(RCX, u64::MAX),
    ];
    let div_out = run(&built, &div_cases);
    assert!(div_out[0].error.is_empty());
    assert_eq!((div_out[0].regs[RAX], div_out[0].regs[RDX]), (0x8000_0000, 0));
    assert!(div_out[1].error.contains("divide error"), "{:?}", div_out[1]);
    assert!(div_out[2].error.contains("divide error"), "{:?}", div_out[2]);
    assert!(div_out[3].error.is_empty());
    assert_eq!(
        (div_out[3].regs[RAX], div_out[3].regs[RDX]),
        ((-14i32) as u32 as u64, (-2i32) as u32 as u64)
    );
    assert!(div_out[4].error.contains("divide error"), "{:?}", div_out[4]);
    assert!(div_out[5].error.contains("divide error"), "{:?}", div_out[5]);
    assert!(div_out[6].error.is_empty());
    assert_eq!(
        (div_out[6].regs[RAX], div_out[6].regs[RDX]),
        ((-14i64) as u64, (-2i64) as u64)
    );
    assert!(div_out[7].error.contains("divide error"), "{:?}", div_out[7]);

    // 7. Windowed-burst soundness: `mov rax, [rdx]; cmp rax, [rdx+8]; jle` under `window` model!
    let win_mem = super::exec::HarnessMemory::window(&super::exec::HarnessMemory::standard_slots());
    let win_built = super::exec::build_with_memory(
        "r60_win_burst_cmp",
        functions.values(),
        &[],
        &win_mem,
        super::exec::GUEST_BASE,
    );
    let mut pair_le = [0u8; 16];
    pair_le[..8].copy_from_slice(&i64::MIN.to_le_bytes());
    pair_le[8..].copy_from_slice(&i64::MAX.to_le_bytes());
    let mut pair_gt = [0u8; 16];
    pair_gt[..8].copy_from_slice(&i64::MAX.to_le_bytes());
    pair_gt[8..].copy_from_slice(&i64::MIN.to_le_bytes());
    let win_cases = vec![
        Case::new(BURST_CMP_JLE).reg(RDX, DATA_ADDR).mem(DATA_ADDR, &pair_le),
        Case::new(BURST_CMP_JLE).reg(RDX, DATA_ADDR).mem(DATA_ADDR, &pair_gt),
    ];
    let win_out = run(&win_built, &win_cases);
    assert!(win_out[0].error.is_empty());
    assert_eq!(win_out[0].regs[RAX], 2, "INT_MIN <= INT_MAX");
    assert!(win_out[1].error.is_empty());
    assert_eq!(win_out[1].regs[RAX], 1, "INT_MAX > INT_MIN");
}

/// R65: End-to-end Wasm64 execution verification of entry-block dead GPR/flag load elision (R65a)
/// and tiny internal leaf `alwaysinline` functions (R65b), including partial-register preservation,
/// incoming-flag preservation (`adc`/`inc`/`setz`), `bsf` zero-source preservation, and calling
/// tiny leaf `cmpxchg`/`xchg`/getter helpers.
#[test]
fn r65_entry_dead_gpr_flag_elision_and_leaf_inlining_are_bit_exact() {
    let test = "r65_entry_dead_gpr_flag_elision_and_leaf_inlining_are_bit_exact";
    if !exec_or_skip(test) {
        return;
    }

    const ELIDE_ENTRY: u64 = CODE_BASE;
    const PARTIAL_AND_BSF: u64 = CODE_BASE + 0x100;
    const PRESERVE_CF_WRAPPER: u64 = CODE_BASE + 0x200;
    const PRESERVE_CF_CALLEE: u64 = CODE_BASE + 0x240;
    const LEAF_GETTER: u64 = CODE_BASE + 0x300;
    const LEAF_CAS: u64 = CODE_BASE + 0x340;
    const LEAF_XCHG: u64 = CODE_BASE + 0x380;
    const CALLER_LEAVES: u64 = CODE_BASE + 0x400;

    let full_functions = Program::new()
        .asm(ELIDE_ENTRY, |a| {
            a.sub(rsp, 0x28)?;
            a.xor(eax, eax)?;
            a.mov(r8d, ecx)?;
            a.lea(r9, ptr(rcx + rdx * 2 + 7))?;
            a.mov(r10, 0x1122_3344_5566_7788u64)?;
            a.add(rax, r8)?;
            a.add(rax, r9)?;
            a.add(rsp, 0x28)?;
            a.ret()
        })
        .asm(PARTIAL_AND_BSF, |a| {
            a.mov(al, 0x5Au32)?;
            a.mov(ch, 0xA5u32)?;
            a.mov(dx, 0x1234u32)?;
            a.xor(bl, bl)?;
            a.bsf(r8d, r9d)?;
            a.ret()
        })
        .asm(PRESERVE_CF_CALLEE, |a| {
            a.inc(rcx)?;
            a.mov(eax, 10)?;
            a.adc(eax, 0)?;
            a.ret()
        })
        .asm(PRESERVE_CF_WRAPPER, |a| {
            let mut set_cf = a.create_label();
            let mut go = a.create_label();
            a.test(rdx, rdx)?;
            a.jnz(set_cf)?;
            a.clc()?;
            a.jmp(go)?;
            a.set_label(&mut set_cf)?;
            a.stc()?;
            a.set_label(&mut go)?;
            a.call(PRESERVE_CF_CALLEE)?;
            a.ret()
        })
        .asm(LEAF_GETTER, |a| {
            a.mov(eax, dword_ptr(rcx))?;
            a.ret()
        })
        .asm(LEAF_CAS, |a| {
            a.mov(eax, edx)?;
            a.cmpxchg(dword_ptr(rcx), r8d)?;
            a.ret()
        })
        .asm(LEAF_XCHG, |a| {
            a.mov(eax, edx)?;
            a.xchg(dword_ptr(rcx), eax)?;
            a.ret()
        })
        .asm(CALLER_LEAVES, |a| {
            a.sub(rsp, 0x28)?;
            a.mov(r10, rcx)?;
            a.call(LEAF_GETTER)?;
            a.mov(r11d, eax)?;
            a.mov(edx, eax)?;
            a.mov(r8d, 0xCAFE_BABEu32)?;
            a.mov(rcx, r10)?;
            a.call(LEAF_CAS)?;
            a.mov(edx, 0x1337_C0DEu32)?;
            a.mov(rcx, r10)?;
            a.call(LEAF_XCHG)?;
            a.mov(rcx, r10)?;
            a.mov(r9d, eax)?;
            a.call(LEAF_GETTER)?;
            a.add(rsp, 0x28)?;
            a.ret()
        })
        .discover();

    let built = build("r65_entry_elision_and_leaves", full_functions.values());
    let init_val = 0xDEAD_BEEFu32.to_le_bytes();
    let cases = vec![
        // 0: ELIDE_ENTRY with dirty incoming rax, r8, r9, r10 (must be cleanly overwritten).
        Case::new(ELIDE_ENTRY)
            .reg(RAX, 0xFFFF_FFFF_DEAD_BEEF)
            .reg(RCX, 0xABCD_0000_0000_0010)
            .reg(RDX, 5)
            .reg(R8, 0xFFFF_FFFF_FFFF_FFFF)
            .reg(R9, 0xAAAA_AAAA_AAAA_AAAA)
            .reg(R10, 0xBBBB_BBBB_BBBB_BBBB),
        // 1: PARTIAL_AND_BSF with r9d = 0 (bsf must preserve full 64-bit r8!).
        Case::new(PARTIAL_AND_BSF)
            .reg(RAX, 0x1122_3344_5566_7788)
            .reg(RCX, 0xAABB_CCDD_EEFF_0011)
            .reg(RDX, 0x5555_6666_7777_8888)
            .reg(RBX, 0xDEAD_BEEF_CAFE_00FF)
            .reg(R8, 0x9988_7766_5544_3322)
            .reg(R9, 0),
        // 2: PARTIAL_AND_BSF with r9d = 0x20 (bsf sets r8 = 5 zero-extended).
        Case::new(PARTIAL_AND_BSF)
            .reg(RAX, 0x1122_3344_5566_7788)
            .reg(RCX, 0xAABB_CCDD_EEFF_0011)
            .reg(RDX, 0x5555_6666_7777_8888)
            .reg(RBX, 0xDEAD_BEEF_CAFE_00FF)
            .reg(R8, 0x9988_7766_5544_3322)
            .reg(R9, 0x20),
        // 3: PRESERVE_CF_WRAPPER with CF = 0 -> eax = 10.
        Case::new(PRESERVE_CF_WRAPPER).reg(RCX, 41).reg(RDX, 0),
        // 4: PRESERVE_CF_WRAPPER with CF = 1 -> eax = 11.
        Case::new(PRESERVE_CF_WRAPPER).reg(RCX, 41).reg(RDX, 1),
        // 5: CALLER_LEAVES calling tiny leaf getter/cmpxchg/xchg.
        Case::new(CALLER_LEAVES)
            .reg(RCX, DATA_ADDR)
            .mem(DATA_ADDR, &init_val),
    ];
    let out = run(&built, &cases);
    for (i, o) in out.iter().enumerate() {
        assert!(o.error.is_empty(), "case {i} failed: {}", o.error);
    }

    // Check case 0: ELIDE_ENTRY
    let rcx0 = 0xABCD_0000_0000_0010u64;
    let r8_want = 0x0000_0010u64;
    let r9_want = rcx0.wrapping_add(5 * 2 + 7);
    assert_eq!(out[0].regs[R8], r8_want);
    assert_eq!(out[0].regs[R9], r9_want);
    assert_eq!(out[0].regs[R10], 0x1122_3344_5566_7788);
    assert_eq!(out[0].regs[RAX], r8_want.wrapping_add(r9_want));

    // Check cases 1 & 2: PARTIAL_AND_BSF
    for idx in [1usize, 2] {
        assert_eq!(out[idx].regs[RAX], 0x1122_3344_5566_775A, "al merge");
        assert_eq!(out[idx].regs[RCX], 0xAABB_CCDD_EEFF_A511, "ch merge");
        assert_eq!(out[idx].regs[RDX], 0x5555_6666_7777_1234, "dx merge");
        assert_eq!(out[idx].regs[RBX], 0xDEAD_BEEF_CAFE_0000, "xor bl,bl merge");
    }
    assert_eq!(
        out[1].regs[R8], 0x9988_7766_5544_3322,
        "bsf with zero src must preserve full 64-bit r8"
    );
    assert_eq!(out[2].regs[R8], 5, "bsf(0x20) == 5 zero-extended");

    // Check cases 3 & 4: PRESERVE_CF_WRAPPER
    assert_eq!(out[3].regs[RCX], 42);
    assert_eq!(out[3].regs[RAX], 10, "inc preserves CF=0");
    assert_eq!(out[4].regs[RCX], 42);
    assert_eq!(out[4].regs[RAX], 11, "inc preserves CF=1");

    // Check case 5: CALLER_LEAVES
    assert_eq!(out[5].regs[R11], 0xDEAD_BEEF);
    assert_eq!(out[5].regs[R9], 0xCAFE_BABE);
    assert_eq!(out[5].regs[RAX], 0x1337_C0DE);
}

fn ref_pext(src: u64, mask: u64, bits: u32) -> u64 {
    let width_mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let (s, mut m) = (src & width_mask, mask & width_mask);
    let (mut r, mut k) = (0u64, 1u64);
    while m != 0 {
        let low = m & m.wrapping_neg();
        if s & low != 0 {
            r |= k;
        }
        m ^= low;
        k <<= 1;
    }
    r
}

fn ref_pdep(src: u64, mask: u64, bits: u32) -> u64 {
    let width_mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let (mut s, mut m) = (src & width_mask, mask & width_mask);
    let mut r = 0u64;
    while m != 0 {
        let low = m & m.wrapping_neg();
        if s & 1 != 0 {
            r |= low;
        }
        m ^= low;
        s >>= 1;
    }
    r
}

#[test]
fn r71_bit_ops_bt_fusion_cmov_and_adcx_adox_execute() {
    if !exec_or_skip("r71_bit_ops_bt_fusion_cmov_and_adcx_adox_execute") {
        return;
    }
    const F_BT: u64 = CODE_BASE;
    const F_CMOV: u64 = CODE_BASE + 0x100;
    const F_BEXTR_BZHI: u64 = CODE_BASE + 0x200;
    const F_PEXT_PDEP: u64 = CODE_BASE + 0x300;
    const F_CNT: u64 = CODE_BASE + 0x400;
    const F_ADCX: u64 = CODE_BASE + 0x500;
    const F_MULX: u64 = CODE_BASE + 0x600;

    let funcs = Program::new()
        // F_BT(rcx=val, rdx=dyn_bit, r8=mem_ptr):
        // Tests bt bit0 + setb, bt bit63 + setae, bts dyn_bit + cmovb, btr [r8], 3 + jc.
        .asm(F_BT, |a| {
            let mut mem_was_set = a.create_label();
            let mut done = a.create_label();
            a.xor(eax, eax)?;
            a.bt(rcx, 0)?;
            a.setb(al)?;
            a.bt(rcx, 63)?;
            a.setae(ah)?;
            a.mov(r9, rcx)?;
            a.mov(r10, 0x1000u64)?;
            a.mov(r11, 0x2000u64)?;
            a.bts(r9, rdx)?;
            a.cmovb(r10, r11)?;
            a.or(rax, r10)?;
            a.btr(qword_ptr(r8), 3)?;
            a.jc(mem_was_set)?;
            a.jmp(done)?;
            a.set_label(&mut mem_was_set)?;
            a.or(rax, 0x10000)?;
            a.set_label(&mut done)?;
            a.xor(edx, edx)?;
            a.ret()
        })
        // F_CMOV(rcx=a, rdx=b, r8=hi_r8, r9=hi_r9):
        // Same-reg 64-bit no-op, 32-bit unconditional zext (even when cond false!), multi-cmov chain.
        .asm(F_CMOV, |a| {
            a.cmp(rcx, rdx)?;
            a.cmove(r8, r8)?; // 64-bit no-op (keeps upper 32 bits)
            a.cmove(r9d, r9d)?; // 32-bit ALWAYS zero-extends even if rcx != rdx
            a.mov(rax, 10u64)?;
            a.mov(r10, 20u64)?;
            a.mov(r11, 30u64)?;
            a.cmp(rcx, rdx)?;
            a.cmovl(rax, r10)?;
            a.cmovg(rax, r11)?;
            a.ret()
        })
        // F_BEXTR_BZHI(rcx=src64):
        // Constant-operand bextr & bzhi covering zero, mid, top, and >=bits cases + flags.
        .asm(F_BEXTR_BZHI, |a| {
            // r8 = bextr64(rcx, start=4, len=12) -> ctrl = 0x0C04
            a.mov(edx, 0x0C04)?;
            a.bextr(r8, rcx, rdx)?;
            // r9d = bextr32(ecx, start=16, len=16) -> top-aligned
            a.mov(edx, 0x1010)?;
            a.bextr(r9d, ecx, edx)?;
            // r10 = bextr64(rcx, start=64, len=8) -> always 0, ZF=1
            a.mov(edx, 0x0840)?;
            a.bextr(r10, rcx, rdx)?;
            a.setz(al)?;
            // r11 = bzhi64(rcx, 12) -> CF=0
            a.mov(edx, 12)?;
            a.bzhi(r11, rcx, rdx)?;
            a.setb(ah)?;
            // rbx = bzhi64(rcx, 64) -> rcx, CF=1
            a.mov(edx, 64)?;
            a.bzhi(rbx, rcx, rdx)?;
            a.setb(dl)?;
            a.shl(rdx, 16)?;
            a.and(rax, 0xFFFF)?;
            a.or(rax, rdx)?;
            // rsi = bzhi64(rcx, 0) -> 0, ZF=1, CF=0
            a.xor(edx, edx)?;
            a.bzhi(rsi, rcx, rdx)?;
            a.setz(dl)?;
            a.shl(rdx, 24)?;
            a.or(rax, rdx)?;
            a.ret()
        })
        // F_PEXT_PDEP(rcx=src64):
        // Constant-mask pext/pdep for 0, all-ones, 1-run, 4-run, and 5-run fallback.
        .asm(F_PEXT_PDEP, |a| {
            a.mov(r8, 0x00FF_00FF_00FF_00FFu64)?; // 4 runs
            a.pext(rax, rcx, r8)?;
            a.pdep(rdx, rcx, r8)?;
            a.mov(r9, 0x0000_FFFF_0000_0000u64)?; // 1 run
            a.pext(r10, rcx, r9)?;
            a.pdep(r11, rcx, r9)?;
            a.mov(ebx, 0xFFFF_0000u32)?; // 32-bit top run
            a.pext(esi, ecx, ebx)?;
            a.pdep(edi, ecx, ebx)?;
            a.mov(rbx, 0x5555_5555_5555_5555u64)?; // >4 runs (helper)
            a.pext(r8, rcx, rbx)?;
            a.pdep(r9, rcx, rbx)?;
            a.ret()
        })
        // F_CNT(rcx=src64): lzcnt & tzcnt result + fused ZF/CF via setz/setb/setae/setnz.
        .asm(F_CNT, |a| {
            a.xor(eax, eax)?;
            a.lzcnt(r8, rcx)?;
            a.setz(al)?; // bit 0: lzcnt ZF (msb == 1)
            a.setb(ah)?; // bit 8: lzcnt CF (src == 0)
            a.tzcnt(r9, rcx)?;
            a.setz(dl)?; // bit 16: tzcnt ZF (lsb == 1)
            a.setnz(dh)?; // bit 24: !tzcnt ZF (lsb == 0)
            a.movzx(rdx, dx)?;
            a.shl(rdx, 16)?;
            a.or(rax, rdx)?;
            a.ret()
        })
        // F_ADCX(rcx=a0, rdx=b0, r8=a1, r9=b1): 128-bit add via adcx (CF=0 start) + adox (OF=0/1).
        .asm(F_ADCX, |a| {
            a.xor(eax, eax)?; // CF=0, OF=0
            a.adcx(rcx, rdx)?;
            a.adcx(r8, r9)?; // trailing CF dead
            a.mov(rax, rcx)?;
            a.mov(rdx, r8)?;
            a.stc()?; // CF=1 constant path
            a.adcx(r10, r11)?; // r10 += r11 + 1, live CF
            a.setb(bl)?;
            a.movzx(rbx, bl)?;
            a.ret()
        })
        // F_MULX(rdx=a, rcx=b): equal-destination 32-bit and 64-bit mulx.
        .asm(F_MULX, |a| {
            a.mulx(r8d, r8d, ecx)?;
            a.mulx(r9, r9, rcx)?;
            a.ret()
        })
        .discover();

    let built = build("r71_bit_ops", funcs.values());
    let mem_addr = DATA_ADDR + 0x100;
    let src64 = 0xDEAD_BEEF_CAFE_BABEu64;
    let cases = [
        // 0: F_BT with bit0=1, bit63=1, bit5=0, mem bit3=1
        Case::new(F_BT)
            .reg(RCX, 0x8000_0000_0000_0001)
            .reg(RDX, 5)
            .reg(R8, mem_addr)
            .mem(mem_addr, &0x08u64.to_le_bytes())
            .read(mem_addr, 8),
        // 1: F_BT with bit0=0, bit63=0, bit5=1, mem bit3=0
        Case::new(F_BT)
            .reg(RCX, 0x20)
            .reg(RDX, 5)
            .reg(R8, mem_addr)
            .mem(mem_addr, &0x00u64.to_le_bytes())
            .read(mem_addr, 8),
        // 2: F_CMOV with rcx < rdx (cmove FALSE: r8 untouched, r9d STILL zero-extended!)
        Case::new(F_CMOV)
            .reg(RCX, 5)
            .reg(RDX, 9)
            .reg(R8, 0x1122_3344_5566_7788)
            .reg(R9, 0xAABB_CCDD_EEFF_0011),
        // 3: F_CMOV with rcx > rdx
        Case::new(F_CMOV)
            .reg(RCX, 9)
            .reg(RDX, 5)
            .reg(R8, 0x1122_3344_5566_7788)
            .reg(R9, 0xAABB_CCDD_EEFF_0011),
        // 4: F_BEXTR_BZHI
        Case::new(F_BEXTR_BZHI).reg(RCX, src64),
        // 5: F_PEXT_PDEP
        Case::new(F_PEXT_PDEP).reg(RCX, src64),
        // 6: F_CNT(0)
        Case::new(F_CNT).reg(RCX, 0),
        // 7: F_CNT(0x8000_0000_0000_0001) (both msb and lsb set!)
        Case::new(F_CNT).reg(RCX, 0x8000_0000_0000_0001),
        // 8: F_CNT(0x0000_0000_0000_0010) (neither msb nor lsb set!)
        Case::new(F_CNT).reg(RCX, 0x10),
        // 9: F_ADCX (with carry from low limb + CF=1 on r10/r11)
        Case::new(F_ADCX)
            .reg(RCX, u64::MAX)
            .reg(RDX, 5)
            .reg(R8, 100)
            .reg(R9, 200)
            .reg(R10, u64::MAX)
            .reg(R11, 0),
        // 10: F_MULX
        Case::new(F_MULX)
            .reg(RDX, 0x1_0000_0000 + 0x8000_0000)
            .reg(RCX, 0x0000_0002_0000_0004),
    ];

    let o = run_clean(&built, &cases);
    // Case 0: bit0=1 (al=1), bit63=1 (ah=0), bit5=0 (r10=0x1000), mem bit3=1 (+0x10000)
    assert_eq!(o[0].regs[RAX], 0x11001);
    assert_eq!(o[0].regs[R9], 0x8000_0000_0000_0021);
    assert_eq!(o[0].mem[0], 0u64.to_le_bytes());
    // Case 1: bit0=0 (al=0), bit63=0 (ah=1 -> 0x100), bit5=1 (r10=0x2000), mem bit3=0
    assert_eq!(o[1].regs[RAX], 0x2100);
    assert_eq!(o[1].regs[R9], 0x20);
    // Cases 2 & 3: F_CMOV
    assert_eq!(o[2].regs[R8], 0x1122_3344_5566_7788, "64-bit same-reg cmov no-op");
    assert_eq!(o[2].regs[R9], 0x0000_0000_EEFF_0011, "32-bit same-reg cmov zext");
    assert_eq!(o[2].regs[RAX], 20, "rcx < rdx -> cmovl");
    assert_eq!(o[3].regs[RAX], 30, "rcx > rdx -> cmovg");
    // Case 4: F_BEXTR_BZHI
    assert_eq!(o[4].regs[R8], (src64 >> 4) & 0xFFF, "bextr64(4, 12)");
    assert_eq!(o[4].regs[R9], (src64 as u32 >> 16) as u64, "bextr32(16, 16)");
    assert_eq!(o[4].regs[R10], 0, "bextr64(64, 8)");
    assert_eq!(o[4].regs[R11], src64 & 0xFFF, "bzhi64(12)");
    assert_eq!(o[4].regs[RBX], src64, "bzhi64(64)");
    assert_eq!(o[4].regs[RSI], 0, "bzhi64(0)");
    // Flags packed in rax: bit0=1 (bextr ZF), bit8=0 (bzhi12 CF), bit16=1 (bzhi64 CF), bit24=1 (bzhi0 ZF)
    assert_eq!(o[4].regs[RAX], 0x0101_0001, "bextr/bzhi flags");
    // Case 5: F_PEXT_PDEP
    assert_eq!(o[5].regs[RAX], ref_pext(src64, 0x00FF_00FF_00FF_00FF, 64));
    assert_eq!(o[5].regs[RDX], ref_pdep(src64, 0x00FF_00FF_00FF_00FF, 64));
    assert_eq!(o[5].regs[R10], ref_pext(src64, 0x0000_FFFF_0000_0000, 64));
    assert_eq!(o[5].regs[R11], ref_pdep(src64, 0x0000_FFFF_0000_0000, 64));
    assert_eq!(o[5].regs[RSI], ref_pext(src64, 0xFFFF_0000, 32));
    assert_eq!(o[5].regs[RDI], ref_pdep(src64, 0xFFFF_0000, 32));
    assert_eq!(o[5].regs[R8], ref_pext(src64, 0x5555_5555_5555_5555, 64));
    assert_eq!(o[5].regs[R9], ref_pdep(src64, 0x5555_5555_5555_5555, 64));
    // Cases 6, 7, 8: F_CNT
    assert_eq!((o[6].regs[R8], o[6].regs[R9], o[6].regs[RAX]), (64, 64, 0x0100_0100));
    assert_eq!((o[7].regs[R8], o[7].regs[R9], o[7].regs[RAX]), (0, 0, 0x0001_0001));
    assert_eq!((o[8].regs[R8], o[8].regs[R9], o[8].regs[RAX]), (59, 4, 0x0100_0000));
    // Case 9: F_ADCX
    assert_eq!(o[9].regs[RAX], 4, "u64::MAX + 5 wraps to 4");
    assert_eq!(o[9].regs[RDX], 301, "100 + 200 + carry(1) = 301");
    assert_eq!(o[9].regs[R10], 0, "u64::MAX + 0 + 1 wraps to 0");
    assert_eq!(o[9].regs[RBX], 1, "carry out of u64::MAX + 0 + 1 is 1");
    // Case 10: F_MULX
    let a = 0x1_0000_0000u64 + 0x8000_0000;
    let b = 0x0000_0002_0000_0004u64;
    let hi32 = ((a as u32 as u64) * (b as u32 as u64)) >> 32;
    let hi64 = ((u128::from(a) * u128::from(b)) >> 64) as u64;
    assert_eq!(o[10].regs[R8], hi32, "mulx32 equal dst high half");
    assert_eq!(o[10].regs[R9], hi64, "mulx64 equal dst high half");
}
